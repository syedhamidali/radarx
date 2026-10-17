#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Reference implementation of R2D2 dealiasing and of the fold margin (NumPy).

This is the specification that the compiled kernel ``_dealias_methods.cpp``
follows step by step, and the fallback when the kernel is not built. The methods
are described in :mod:`radarx.retrieve.dealias_methods`.

A sweep is an ``int32`` array ``(ray, gate)`` of velocities in steps of
``1 / VEL_Q`` m/s with :data:`MISSING` for gates without data. All decisions
are made with integers, so the kernel and this module give identical folds.
Rays are neighbours in index order, and the last ray follows the first one when
``wrap`` is true.
"""

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import csgraph

#: Velocities are kept in steps of 1/256 m/s.
VEL_Q = 256.0
MISSING = np.iinfo(np.int32).min

NEIGHBOURS = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]


def shift(a, di, dj, fill, wrap):
    """``b[i, j] = a[i + di, j + dj]``, ``fill`` outside (rays wrap if ``wrap``)."""
    nray, ngate = a.shape
    if abs(dj) >= ngate or (not wrap and abs(di) >= nray):
        return np.full_like(a, fill)
    if wrap:
        out = np.roll(a, -di, axis=0)
    else:
        out = np.full_like(a, fill)
        if di >= 0:
            out[: nray - di] = a[di:]
        else:
            out[-di:] = a[: nray + di]
    if dj > 0:
        out[:, : ngate - dj] = out[:, dj:].copy()
        out[:, ngate - dj :] = fill
    elif dj < 0:
        out[:, -dj:] = out[:, : ngate + dj].copy()
        out[:, :-dj] = fill
    return out


def round_div(num, den):
    """``round(num / den)`` with halves rounded up (integers, ``den > 0``)."""
    return np.floor_divide(2 * num + den, 2 * den)


def max_difference(cur, wrap):
    """Largest absolute difference of every gate to its valid 8 neighbours (int64)."""
    valid = cur != MISSING
    c = cur.astype(np.int64)
    out = np.zeros(cur.shape, dtype=np.int64)
    for di, dj in NEIGHBOURS:
        nb = shift(cur, di, dj, MISSING, wrap)
        both = valid & (nb != MISSING)
        out = np.where(both, np.maximum(out, np.abs(c - nb)), out)
    return out


def dilate(mask, half, wrap):
    """Gates that have a gate of ``mask`` within ``half`` rays and gates."""
    out = np.zeros(mask.shape, dtype=bool)
    for di in range(-half, half + 1):
        for dj in range(-half, half + 1):
            out |= shift(mask, di, dj, False, wrap)
    return out


def label_regions(mask, wrap):
    """Connected regions (8 neighbours) of ``mask``: ids from 0, -1 outside."""
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return np.full(mask.shape, -1, dtype=np.int64), 0
    if wrap and mask.shape[0] > 1:
        first, last = labels[0], labels[-1]
        edges = []
        for dj in (-1, 0, 1):
            other = shift(last[None, :], 0, dj, 0, False)[0]
            both = (first > 0) & (other > 0)
            edges.append(np.stack([first[both], other[both]]))
        edges = np.concatenate(edges, axis=1)
        graph = sparse.coo_matrix(
            (np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(n + 1, n + 1)
        )
        _, comp = csgraph.connected_components(graph, directed=False)
        labels = np.where(labels > 0, comp[labels], 0)
        _, labels = np.unique(labels, return_inverse=True)
        labels = labels.reshape(mask.shape) - 1  # 0 (outside) becomes -1
        n = int(labels.max()) + 1
        return labels.astype(np.int64), n
    return labels.astype(np.int64) - 1, n


def sum_by(labels, values, n):
    """Exact integer sums of ``values`` (int64) for the labels 0..n-1."""
    out = np.zeros(n, dtype=np.int64)
    np.add.at(out, labels, values)
    return out


# --------------------------------------------------------------------------
# R2D2
# --------------------------------------------------------------------------


def _region_decisions(cur, labels, n, guides, p):
    """Fold step of every region (and whether it is decided) from the guidance."""
    valid = labels >= 0
    decided = np.zeros(n, dtype=bool)
    delta = np.zeros(n, dtype=np.int64)
    if n == 0 or not guides:
        return decided, delta
    nyq2 = p["nyq2"]
    have = np.zeros(n, dtype=np.int64)
    fail = np.zeros(n, dtype=bool)
    first = np.zeros(n, dtype=np.int64)
    for g, fallback in guides:
        ok = valid & (g != MISSING)
        lab = labels[ok]
        d = g[ok].astype(np.int64) - cur[ok]
        cnt = sum_by(lab, np.ones(d.shape, dtype=np.int64), n)
        s1 = sum_by(lab, d, n)
        s2 = sum_by(lab, d * d, n)
        use = cnt >= p["min_overlap"]
        if fallback:
            use &= have == 0
        safe = np.maximum(cnt, 1)
        mean = s1.astype(np.float64) / safe
        var = s2.astype(np.float64) / safe - mean * mean
        spread_fail = use & (var > p["gamma"] * p["gamma"])
        k = round_div(s1, safe * nyq2)
        fail |= spread_fail
        fail |= use & (have > 0) & (k != first)
        first = np.where(use & (have == 0), k, first)
        have += use
    decided = (have > 0) & ~fail
    delta = np.where(decided, first, 0)
    return decided, delta


def _scan(cur, k, res, valid, nyq2, s_q, axis, reverse, wrap):
    """One scanning pass of the spatial dealiasing (in place)."""
    n = cur.shape[axis]
    order = np.arange(n - 1, -1, -1) if reverse else np.arange(n)
    if axis == 0 and wrap:
        order = np.concatenate([order, order])
    step = 1 if reverse else -1  # the predecessor lies at g + step
    for g in order:
        pre = g + step
        if wrap and axis == 0:
            pre %= n
        elif pre < 0 or pre >= n:
            continue
        if axis == 0:
            cg, cp, rg, rp, vg = cur[g], cur[pre], res[g], res[pre], valid[g]
        else:
            cg, cp, rg, rp, vg = (
                cur[:, g],
                cur[:, pre],
                res[:, g],
                res[:, pre],
                valid[:, g],
            )
        todo = vg & ~rg & rp
        if not todo.any():
            continue
        m = round_div(cp - cg, np.int64(nyq2))
        new = cg + m * nyq2
        acc = todo & (np.abs(new - cp) < s_q)
        if axis == 0:
            cur[g] = np.where(acc, new, cg)
            k[g] += np.where(acc, m, 0)
            res[g] = rg | acc
        else:
            cur[:, g] = np.where(acc, new, cg)
            k[:, g] += np.where(acc, m, 0)
            res[:, g] = rg | acc


def _window_pass(cur, k, res, valid, nyq2, half, minimum, wrap):
    """Unresolved gates follow the median of the resolved gates around them."""
    todo = np.argwhere(valid & ~res)
    if todo.size == 0:
        return
    snap_cur = cur.copy()
    snap_res = res.copy()
    nray, ngate = cur.shape
    for i, j in todo:
        rows = np.arange(i - half, i + half + 1)
        if wrap:
            rows = rows % nray
        else:
            rows = rows[(rows >= 0) & (rows < nray)]
        cols = np.arange(max(0, j - half), min(ngate, j + half + 1))
        block = snap_cur[np.ix_(rows, cols)]
        ok = snap_res[np.ix_(rows, cols)]
        values = np.sort(block[ok])
        if values.size < minimum:
            continue
        med = values[(values.size - 1) // 2]
        m = int(round_div(np.int64(med) - cur[i, j], np.int64(nyq2)))
        cur[i, j] += m * nyq2
        k[i, j] += m
        res[i, j] = True


def _guess_pass(cur, k, res, valid, nyq2, ref):
    """Gates that nothing resolved take the fold nearest to the first guess."""
    if ref is None:
        return
    todo = valid & ~res & (ref != MISSING)
    m = round_div(ref.astype(np.int64) - cur, np.int64(nyq2))
    cur += np.where(todo, m * nyq2, 0)
    k += np.where(todo, m, 0)
    res |= todo


def r2d2_sweep(q, nyq2, guidance, wrap, p):
    """
    R2D2 folds of one sweep.

    Parameters
    ----------
    q : numpy.ndarray
        Quantized velocities, ``int32`` ``(ray, gate)``, :data:`MISSING` where
        there are no data.
    nyq2 : int
        Twice the Nyquist velocity in the same units.
    guidance : dict
        ``"ref"`` (first guess), ``"above"`` (sweep above) and ``"prev"``
        (previous volume): dealiased velocities on the grid of the sweep as
        ``int32`` with :data:`MISSING`, or None.
    wrap : bool
        Whether the last ray is next to the first.
    p : dict
        ``beta``, ``gamma``, ``min_overlap``, ``buffer_half``, ``thresholds``,
        ``window_half``, ``window_min``, ``max_iterations``, ``patience`` and
        ``ref_fallback`` (the first guess decides only the regions that the other
        guidance does not cover).

    Returns
    -------
    folds : numpy.ndarray
        ``int32`` folds of the best iteration.
    info : dict
        ``iterations``, ``best`` (iteration of the result), ``unresolved``
        (gates of the result that no step resolved) and ``buffer`` (gates
        in a buffer zone in the last iteration).
    """
    p = dict(p, nyq2=nyq2)
    valid = q != MISSING
    nvalid = int(valid.sum())
    k = np.zeros(q.shape, dtype=np.int64)
    best_k = k.copy()
    cur = q.astype(np.int64)
    cur[~valid] = 0
    shown = np.where(valid, cur, MISSING).astype(np.int32)
    best_sum = int(max_difference(shown, wrap).sum())
    info = {"iterations": 0, "best": 0, "unresolved": nvalid, "buffer": 0}
    if nvalid == 0:
        return best_k.astype(np.int32), info
    coverage = {
        name: (0 if g is None else int(((g != MISSING) & valid).sum()))
        for name, g in guidance.items()
    }
    far = max(coverage["above"], coverage["prev"]) * 10 < nvalid
    res_final = np.zeros(q.shape, dtype=bool)
    for it in range(1, p["max_iterations"] + 1):
        shown = np.where(valid, cur, MISSING).astype(np.int32)
        dmax = max_difference(shown, wrap)
        buffer = dilate(valid & (dmax >= p["beta"]), p["buffer_half"], wrap)
        labels, n = label_regions(valid & ~buffer, wrap)
        guides = [
            (guidance[nm], False)
            for nm in ("above", "prev")
            if guidance[nm] is not None
        ]
        if guidance["ref"] is not None and (p["ref_fallback"] or far or it > 1):
            guides.append((guidance["ref"], p["ref_fallback"]))
        decided, delta = _region_decisions(cur, labels, n, guides, p)
        inreg = labels >= 0
        gate_decided = np.zeros(q.shape, dtype=bool)
        gate_decided[inreg] = decided[labels[inreg]]
        step = np.zeros(q.shape, dtype=np.int64)
        step[inreg] = np.where(decided[labels[inreg]], delta[labels[inreg]], 0)
        fold_before = k.copy()
        k += step
        cur = cur + step * nyq2
        res = gate_decided.copy()
        nbuffer = int((valid & ~res).sum())
        for s_q in p["thresholds"]:
            for axis, reverse in ((1, False), (1, True), (0, False), (0, True)):
                _scan(cur, k, res, valid, nyq2, s_q, axis, reverse, wrap)
        _window_pass(cur, k, res, valid, nyq2, p["window_half"], p["window_min"], wrap)
        _guess_pass(cur, k, res, valid, nyq2, guidance["ref"])
        shown = np.where(valid, cur, MISSING).astype(np.int32)
        total = int(max_difference(shown, wrap).sum())
        info["iterations"] = it
        info["buffer"] = nbuffer
        if total < best_sum:
            best_sum, best_k, info["best"] = total, k.copy(), it
            res_final = res.copy()
        elif it - info["best"] >= p["patience"]:
            break
        if not (k != fold_before).any():
            break
    info["unresolved"] = int((valid & ~res_final).sum()) if info["best"] else nvalid
    return best_k.astype(np.int32), info


# --------------------------------------------------------------------------
# Fold margin
# --------------------------------------------------------------------------


def fold_margin(q, threshold, wrap):
    """
    Regions of a dealiased sweep and the evidence for their fold.

    Parameters
    ----------
    q : numpy.ndarray
        Quantized dealiased velocities (``int32``, :data:`MISSING` for no data).
    threshold : int
        Neighbouring gates whose velocities differ by less than this belong to
        the same region.
    wrap : bool
        Whether the last ray is next to the first.

    Returns
    -------
    region : numpy.ndarray
        Region of every gate (-1 without data).
    size, pairs : numpy.ndarray
        Gates of every region and gate pairs on its boundary.
    total : numpy.ndarray
        Sum over the boundary pairs of the velocity of the neighbour minus
        that of the gate.
    """
    valid = q != MISSING
    c = q.astype(np.int64)
    nray, ngate = q.shape
    # union of neighbouring gates (4 neighbours) that differ by less than the threshold
    index = np.arange(q.size).reshape(q.shape)
    edges = []
    pairs = [(1, 0), (0, 1)]
    for di, dj in pairs:
        nb = shift(q, di, dj, MISSING, wrap)
        nbi = shift(index, di, dj, -1, wrap)
        both = valid & (nb != MISSING)
        near = both & (np.abs(c - nb) < threshold)
        edges.append((index[near], nbi[near]))
    a = np.concatenate([e[0] for e in edges])
    b = np.concatenate([e[1] for e in edges])
    graph = sparse.coo_matrix((np.ones(a.size), (a, b)), shape=(q.size, q.size))
    count, comp = csgraph.connected_components(graph, directed=False)
    comp = comp.reshape(q.shape)
    region = np.full(q.shape, -1, dtype=np.int64)
    ids = np.unique(comp[valid])
    remap = np.full(count, -1, dtype=np.int64)
    remap[ids] = np.arange(ids.size)
    region[valid] = remap[comp[valid]]
    n = ids.size
    size = sum_by(region[valid], np.ones(int(valid.sum()), dtype=np.int64), n)
    npairs = np.zeros(n, dtype=np.int64)
    total = np.zeros(n, dtype=np.int64)
    for di, dj in pairs:
        nb = shift(q, di, dj, MISSING, wrap)
        nbr = shift(region, di, dj, -1, wrap)
        both = valid & (nb != MISSING) & (nbr != region)
        d = nb[both].astype(np.int64) - c[both]
        npairs += sum_by(region[both], np.ones(d.shape, dtype=np.int64), n)
        total += sum_by(region[both], d, n)
        npairs += sum_by(nbr[both], np.ones(d.shape, dtype=np.int64), n)
        total += sum_by(nbr[both], -d, n)
    return region, size, npairs, total
