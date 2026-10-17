#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
NumPy reference of the volume matching kernel ``_satellite_match.cpp``.

The functions take and return the same arrays as the kernel (see the comments
there) and are the fallback when the compiled module is not available. The
ground radar gates are found with a ``scipy.spatial.cKDTree`` per sweep.
"""

import numpy as np

LN2 = np.log(2.0)


def _reduce(values, modes, weights):
    """Mean, spread and count of ``values`` (nv, m) with the weights (m,)."""
    nv = values.shape[0]
    mean = np.full(nv, np.nan)
    sd = np.full(nv, np.nan)
    count = np.zeros(nv)
    for v in range(nv):
        x = values[v]
        ok = np.isfinite(x)
        count[v] = ok.sum()
        if not ok.any():
            continue
        w = weights[ok]
        x = x[ok]
        sw = w.sum()
        if not sw > 0.0:
            continue
        t = 10.0 ** (0.1 * x) if modes[v] == 1 else x
        mean[v] = (w * t).sum() / sw
        d = x - x[0]  # shifted: no cancellation for a narrow distribution
        m = (w * d).sum() / sw
        sd[v] = np.sqrt(max(0.0, (w * d * d).sum() / sw - m * m))
    return mean, sd, count


def match_gates(
    gx, gy, offsets, values, modes, qx, qy, qs, radius, gaussian=False, n_threads=0
):
    """NumPy version of ``_satellite_match.match_gates``."""
    from scipy.spatial import cKDTree

    gx, gy = np.asarray(gx, float), np.asarray(gy, float)
    values = np.asarray(values, float)
    offsets = np.asarray(offsets, np.int64)
    modes = np.asarray(modes, np.int64)
    qx, qy, qs = np.asarray(qx, float), np.asarray(qy, float), np.asarray(qs)
    nv, nq, ns = values.shape[0], qx.size, offsets.size - 1
    if gx.shape != gy.shape or values.shape[1] != gx.size:
        raise ValueError("gx, gy and values must describe the same gates")
    if offsets[0] != 0 or offsets[-1] != gx.size or np.any(np.diff(offsets) < 0):
        raise ValueError("offsets must run from 0 to the number of gates")
    if not radius > 0.0:
        raise ValueError("radius must be positive")
    if nq and (qs.min() < 0 or qs.max() >= ns):
        raise ValueError("sweep index out of range")
    mean = np.full((nv, nq), np.nan)
    sd = np.full((nv, nq), np.nan)
    count = np.zeros((nv, nq))
    total = np.zeros(nq, np.int64)
    for s in range(ns):
        a, b = offsets[s], offsets[s + 1]
        query = np.flatnonzero((qs == s) & np.isfinite(qx) & np.isfinite(qy))
        finite = np.isfinite(gx[a:b]) & np.isfinite(gy[a:b])
        if query.size == 0 or not finite.any():
            continue
        index = a + np.flatnonzero(finite)
        tree = cKDTree(np.column_stack([gx[index], gy[index]]))
        found = tree.query_ball_point(
            np.column_stack([qx[query], qy[query]]), radius * (1.0 + 1e-12)
        )
        for q, hit in zip(query, found):
            hit = index[np.asarray(hit, dtype=np.int64)]
            # the tree radius is inclusive and slightly larger: apply the kernel's test
            d2 = (gx[hit] - qx[q]) ** 2 + (gy[hit] - qy[q]) ** 2
            keep = d2 <= radius * radius
            hit, d2 = hit[keep], d2[keep]
            total[q] = hit.size
            w = np.exp(-LN2 * d2 / (radius * radius)) if gaussian else np.ones(hit.size)
            mean[:, q], sd[:, q], count[:, q] = _reduce(values[:, hit], modes, w)
    return mean, sd, count, total


def match_bins(z, values, modes, qr, zlo, zhi, zc, gaussian=False, n_threads=0):
    """NumPy version of ``_satellite_match.match_bins``."""
    z = np.asarray(z, float)
    values = np.asarray(values, float)
    modes = np.asarray(modes, np.int64)
    qr = np.asarray(qr, np.int64)
    zlo, zhi, zc = (np.asarray(a, float) for a in (zlo, zhi, zc))
    nv, nq = values.shape[0], qr.size
    if z.ndim != 2 or values.shape[1:] != z.shape:
        raise ValueError("values must have the shape (variables, rays, bins)")
    if nq and (qr.min() < 0 or qr.max() >= z.shape[0]):
        raise ValueError("ray index out of range")
    mean = np.full((nv, nq), np.nan)
    sd = np.full((nv, nq), np.nan)
    count = np.zeros((nv, nq))
    total = np.zeros(nq, np.int64)
    for q in range(nq):
        r = qr[q]
        inside = (z[r] >= zlo[q]) & (z[r] <= zhi[q])
        total[q] = inside.sum()
        h = 0.5 * (zhi[q] - zlo[q])
        if gaussian:
            u = (z[r][inside] - zc[q]) / h if h > 0.0 else np.zeros(total[q])
            w = np.exp(-LN2 * u * u)
        else:
            w = np.ones(total[q])
        mean[:, q], sd[:, q], count[:, q] = _reduce(values[:, r, inside], modes, w)
    return mean, sd, count, total
