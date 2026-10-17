#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Object operations on stacks of 2-D frames, with a compiled kernel and a NumPy twin.

Labelling, the nested threshold regions of the feature detection, the watershed,
per-label sums and the overlap counts of two label images. The NumPy functions are
the oracle of the kernel in :file:`_objects.cpp`: both follow the same steps and tie
breaks, so that they give the same labels.
"""

import heapq

import numpy as np

try:
    from . import _objects

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _objects = None
    HAS_COMPILED_KERNEL = False

_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
_BLOCK = np.ones((3, 3), dtype=int)
# neighbour offsets, in the order of the kernel: up, left, right, down, diagonals
_OFFSETS = [(-1, 0), (0, -1), (0, 1), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def check_connectivity(connectivity):
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, not {connectivity!r}")
    return int(connectivity)


def _structure(connectivity):
    return _CROSS if connectivity == 4 else _BLOCK


# --------------------------------------------------------------------------
# smoothing
# --------------------------------------------------------------------------


def smooth(field, sigma, use_compiled, n_threads):
    """Gaussian smoothing (standard deviation ``sigma`` cells) of each frame (nt, ny, nx)."""
    field = np.ascontiguousarray(field, dtype=np.float64)
    if use_compiled:
        return _objects.smooth(field, float(sigma), int(n_threads or 0))
    from scipy import ndimage

    return ndimage.gaussian_filter(
        field, sigma=(0.0, float(sigma), float(sigma)), mode="nearest"
    )


# --------------------------------------------------------------------------
# labelling
# --------------------------------------------------------------------------


def _label_frame_numpy(mask, connectivity):
    from scipy import ndimage

    labels, n = ndimage.label(mask, structure=_structure(connectivity))
    return labels.astype(np.int32), int(n)


def label(mask, connectivity, use_compiled, n_threads):
    """Labels of the true cells of each frame of ``mask`` (nt, ny, nx), and the counts."""
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    if use_compiled:
        return _objects.label(mask, connectivity, int(n_threads or 0))
    labels = np.zeros(mask.shape, dtype=np.int32)
    counts = np.zeros(mask.shape[0], dtype=np.int32)
    for t in range(mask.shape[0]):
        labels[t], counts[t] = _label_frame_numpy(mask[t], connectivity)
    return labels, counts


# --------------------------------------------------------------------------
# nested threshold regions
# --------------------------------------------------------------------------


def _features_frame_numpy(frame, thresholds, min_size, connectivity):
    ny, nx = frame.shape
    nlev = len(thresholds)
    levels, sizes, firsts, kept, has_child = [], [], [], [], []
    with np.errstate(invalid="ignore"):
        finite = np.isfinite(frame)
        for k, thr in enumerate(thresholds):
            lab, n = _label_frame_numpy(finite & (frame >= thr), connectivity)
            size = np.bincount(lab.ravel(), minlength=n + 1)[1:]
            flat = lab.ravel()
            first = np.full(n, -1, dtype=np.int64)
            order = np.flatnonzero(flat)
            if n:
                # first occurrence of each label in raster order
                _, idx = np.unique(flat[order], return_index=True)
                first = order[idx]
            levels.append(lab)
            sizes.append(size)
            firsts.append(first)
            keep = size >= min_size
            kept.append(keep)
            has_child.append(np.zeros(n, dtype=bool))
            if k > 0:
                for c in np.flatnonzero(keep):
                    parent = levels[k - 1].ravel()[first[c]]
                    has_child[k - 1][parent - 1] = True
    leaves = []
    for k in range(nlev):
        for c in np.flatnonzero(kept[k] & ~has_child[k]):
            leaves.append((int(firsts[k][c]), k, int(c)))
    leaves.sort()
    markers = np.zeros((ny, nx), dtype=np.int32)
    level_of = []
    for number, (_, k, c) in enumerate(leaves, start=1):
        markers[levels[k] == c + 1] = number
        level_of.append(k)
    return markers, level_of


def features(field, thresholds, min_size, connectivity, use_compiled, n_threads):
    """
    Feature regions of each frame: nested regions at ascending thresholds, leaves only.

    Returns the markers (nt, ny, nx), the number of features per frame and the
    threshold index of each feature (concatenated over the frames).
    """
    field = np.ascontiguousarray(field, dtype=np.float64)
    thresholds = np.ascontiguousarray(thresholds, dtype=np.float64)
    if use_compiled:
        return _objects.features(
            field, thresholds, int(min_size), connectivity, int(n_threads or 0)
        )
    markers = np.zeros(field.shape, dtype=np.int32)
    counts = np.zeros(field.shape[0], dtype=np.int32)
    levels = []
    for t in range(field.shape[0]):
        markers[t], level_of = _features_frame_numpy(
            field[t], thresholds, min_size, connectivity
        )
        counts[t] = len(level_of)
        levels.extend(level_of)
    return markers, counts, np.asarray(levels, dtype=np.int32)


# --------------------------------------------------------------------------
# watershed
# --------------------------------------------------------------------------


def _watershed_frame_numpy(field, markers, mask, connectivity):
    ny, nx = field.shape
    out = np.zeros((ny, nx), dtype=np.int32)
    heap = []
    age = 0
    ok = mask.astype(bool) & np.isfinite(field)
    for y, x in zip(*np.nonzero((markers > 0) & ok)):
        out[y, x] = markers[y, x]
        heapq.heappush(heap, (-field[y, x], age, y * nx + x))
        age += 1
    offsets = _OFFSETS[:connectivity]
    while heap:
        _, _, p = heapq.heappop(heap)
        y, x = divmod(p, nx)
        for dy, dx in offsets:
            yy, xx = y + dy, x + dx
            if 0 <= yy < ny and 0 <= xx < nx and not out[yy, xx] and ok[yy, xx]:
                out[yy, xx] = out[y, x]
                heapq.heappush(heap, (-field[yy, xx], age, yy * nx + xx))
                age += 1
    return out


def watershed(field, markers, mask, connectivity, use_compiled, n_threads):
    """Marker-controlled watershed of each frame, flooding towards lower values."""
    field = np.ascontiguousarray(field, dtype=np.float64)
    markers = np.ascontiguousarray(markers, dtype=np.int32)
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    if use_compiled:
        return _objects.watershed(
            field, markers, mask, connectivity, int(n_threads or 0)
        )
    out = np.zeros(field.shape, dtype=np.int32)
    for t in range(field.shape[0]):
        out[t] = _watershed_frame_numpy(field[t], markers[t], mask[t], connectivity)
    return out


# --------------------------------------------------------------------------
# region sums
# --------------------------------------------------------------------------


def region_sums(labels, values, nlab, use_compiled, n_threads):
    """
    Sum, count, maximum and minimum of the finite ``values`` (nf, nt, ny, nx) of each label.

    Each result has the shape (nf, nt, nlab); labels run from 1 to ``nlab``.
    """
    labels = np.ascontiguousarray(labels, dtype=np.int32)
    values = np.ascontiguousarray(values, dtype=np.float64)
    nlab = int(nlab)
    if use_compiled:
        return _objects.region_sums(labels, values, nlab, int(n_threads or 0))
    nf, nt = values.shape[:2]
    shape = (nf, nt, nlab)
    total, count = np.zeros(shape), np.zeros(shape)
    high, low = np.full(shape, np.nan), np.full(shape, np.nan)
    for t in range(nt):
        lab = labels[t].ravel()
        inside = (lab > 0) & (lab <= nlab)
        for f in range(nf):
            v = values[f, t].ravel()
            use = inside & np.isfinite(v)
            index = lab[use] - 1
            total[f, t] = np.bincount(index, weights=v[use], minlength=nlab)
            count[f, t] = np.bincount(index, minlength=nlab)
            hi = np.full(nlab, -np.inf)
            lo = np.full(nlab, np.inf)
            np.maximum.at(hi, index, v[use])
            np.minimum.at(lo, index, v[use])
            seen = count[f, t] > 0
            high[f, t, seen] = hi[seen]
            low[f, t, seen] = lo[seen]
    return total, count, high, low


# --------------------------------------------------------------------------
# overlap of two label images
# --------------------------------------------------------------------------


def overlap(l0, l1, n0, n1, dy, dx, use_compiled, n_threads):
    """
    Overlap counts of pairs of label images after a shift of ``l0``.

    ``l0`` and ``l1`` are (np, ny, nx), ``n0`` and ``n1`` the number of labels per
    pair, ``dy`` and ``dx`` integer shifts (np, 1 or ny, 1 or nx) in cells. The
    result holds, for pair p, a block of ``n0[p] * n1[p]`` counts (row: label of
    ``l0``, column: label of ``l1``) after the blocks of the pairs before it.
    """
    l0 = np.ascontiguousarray(l0, dtype=np.int32)
    l1 = np.ascontiguousarray(l1, dtype=np.int32)
    n0 = np.ascontiguousarray(n0, dtype=np.int32)
    n1 = np.ascontiguousarray(n1, dtype=np.int32)
    dy = np.ascontiguousarray(dy, dtype=np.int32)
    dx = np.ascontiguousarray(dx, dtype=np.int32)
    if use_compiled:
        return _objects.overlap(l0, l1, n0, n1, dy, dx, int(n_threads or 0))
    npairs, ny, nx = l0.shape
    blocks = []
    yy, xx = np.mgrid[0:ny, 0:nx]
    for p in range(npairs):
        a = l0[p]
        sy = np.broadcast_to(dy[p], (ny, nx))
        sx = np.broadcast_to(dx[p], (ny, nx))
        ty, tx = yy + sy, xx + sx
        inside = (a > 0) & (a <= n0[p]) & (ty >= 0) & (ty < ny) & (tx >= 0) & (tx < nx)
        b = np.zeros_like(a)
        b[inside] = l1[p][ty[inside], tx[inside]]
        use = inside & (b > 0) & (b <= n1[p])
        index = (a[use].astype(np.int64) - 1) * int(n1[p]) + (b[use] - 1)
        blocks.append(
            np.bincount(index, minlength=int(n0[p]) * int(n1[p])).astype(np.float64)
        )
    return np.concatenate(blocks) if blocks else np.zeros(0)
