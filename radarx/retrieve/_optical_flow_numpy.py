#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
NumPy reference of the optical-flow kernels and dispatch to the compiled ones.

The compiled kernel is ``_optical_flow.cpp``. Both engines have the same
arguments and give the same result. See :mod:`radarx.retrieve.motion`.
"""

from __future__ import annotations

import numpy as np

try:
    from . import _optical_flow

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _optical_flow = None
    HAS_COMPILED_KERNEL = False


def lk_solve_numpy(ix, iy, it, taps):
    """
    Lucas-Kanade increment at every cell, (4, ny, nx): du, dv, lmin, lmax.

    ``ix``, ``iy``, ``it`` are (pair, ny, nx). The products are summed over
    the pairs and smoothed with the separable ``taps`` (edges repeated).
    """
    from scipy.ndimage import correlate1d

    prods = [
        (ix * ix).sum(0),
        (ix * iy).sum(0),
        (iy * iy).sum(0),
        (ix * it).sum(0),
        (iy * it).sum(0),
    ]
    sxx, sxy, syy, sxt, syt = (
        correlate1d(
            correlate1d(p, taps, axis=1, mode="nearest"), taps, axis=0, mode="nearest"
        )
        for p in prods
    )
    tr = 0.5 * (sxx + syy)
    det = sxx * syy - sxy * sxy
    disc = np.sqrt(np.maximum(0.0, tr * tr - det))
    good = (det > 1e-12 * np.maximum(tr * tr, 1e-300)) & (det > 0.0)
    safe = np.where(good, det, 1.0)
    du = np.where(good, -(syy * sxt - sxy * syt) / safe, 0.0)
    dv = np.where(good, -(sxx * syt - sxy * sxt) / safe, 0.0)
    return np.stack([du, dv, tr - disc, tr + disc])


def cosine_basis(order, ny, nx):
    """Cosine basis (M, ny, nx) with ``M = (order + 1)**2``, p-major."""
    p = np.arange(order + 1)
    cy = np.cos(np.pi * p[:, None] * (np.arange(ny) + 0.5)[None, :] / ny)
    cx = np.cos(np.pi * p[:, None] * (np.arange(nx) + 0.5)[None, :] / nx)
    return (cy[:, None, :, None] * cx[None, :, None, :]).reshape(-1, ny, nx)


def darts_normal_numpy(ix, iy, it, order):
    """
    Normal equations of the cosine-basis fit of the flow, (2M, 2M + 1).

    The last column is the right-hand side.
    """
    n, ny, nx = ix.shape
    basis = cosine_basis(order, ny, nx).reshape(-1, ny * nx)  # (M, N)
    m = basis.shape[0]
    d = 2 * m
    out = np.zeros((d, d + 1))
    for k in range(n):
        gx, gy, gt = ix[k].ravel(), iy[k].ravel(), it[k].ravel()
        keep = (gx != 0.0) | (gy != 0.0)
        b = basis[:, keep]
        design = np.concatenate([b * gx[keep], b * gy[keep]], axis=0)  # (2M, N')
        out[:, :d] += design @ design.T
        out[:, d] -= design @ gt[keep]
    return out


def lk_solve(ix, iy, it, taps, use_compiled, n_threads):
    """Run :func:`lk_solve_numpy` or the compiled kernel (same arguments)."""
    if use_compiled:
        return _optical_flow.lk_solve(
            np.ascontiguousarray(ix, dtype=np.float64),
            np.ascontiguousarray(iy, dtype=np.float64),
            np.ascontiguousarray(it, dtype=np.float64),
            np.ascontiguousarray(taps, dtype=np.float64),
            int(n_threads or 0),
        )
    return lk_solve_numpy(ix, iy, it, np.asarray(taps, dtype=np.float64))


def darts_normal(ix, iy, it, order, use_compiled, n_threads):
    """Run :func:`darts_normal_numpy` or the compiled kernel."""
    if use_compiled:
        return _optical_flow.darts_normal(
            np.ascontiguousarray(ix, dtype=np.float64),
            np.ascontiguousarray(iy, dtype=np.float64),
            np.ascontiguousarray(it, dtype=np.float64),
            int(order),
            int(n_threads or 0),
        )
    return darts_normal_numpy(ix, iy, it, int(order))
