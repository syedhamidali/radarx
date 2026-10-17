#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
NumPy reference of the spectral cascade step and dispatch to the compiled one.

The compiled kernel is ``_nowcast.cpp``. Both engines have the same arguments
and give the same result. See :mod:`radarx.retrieve.nowcast`.
"""

from __future__ import annotations

import numpy as np

try:
    from . import _nowcast

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _nowcast = None
    HAS_COMPILED_KERNEL = False


def ar_step_numpy(z1, z2, phi1, phi2, sigma, wh, noise):
    """
    One step of the autoregressive cascade model in the Fourier domain.

    ``z1`` and ``z2`` (member, level, row, column) are the states at the
    latest and the previous time and are updated in place; ``wh`` (level, row,
    column) shapes the innovation, ``noise`` (member, row, column) is the
    transform of the white noise (``None`` for no innovation). Returns the
    spectrum (member, row, column) of ``sum_j sigma_j Z_j``.
    """
    new = phi1[None, :, None, None] * z1 + phi2[None, :, None, None] * z2
    if noise is not None:
        new += wh[None] * noise[:, None]
    z2[...] = z1
    z1[...] = new
    return np.einsum("l,mlyx->myx", sigma, new)


def ar_step(z1, z2, phi1, phi2, sigma, wh, noise, use_compiled, n_threads):
    """Run :func:`ar_step_numpy` or the compiled kernel (same arguments)."""
    if use_compiled:
        empty = np.empty(0, dtype=np.complex128) if noise is None else noise
        return _nowcast.ar_step(
            z1,
            z2,
            np.ascontiguousarray(phi1, dtype=np.float64),
            np.ascontiguousarray(phi2, dtype=np.float64),
            np.ascontiguousarray(sigma, dtype=np.float64),
            np.ascontiguousarray(wh, dtype=np.float64),
            np.ascontiguousarray(empty, dtype=np.complex128),
            int(n_threads or 0),
        )
    return ar_step_numpy(z1, z2, phi1, phi2, sigma, wh, noise)


def rank_match_numpy(fields, cells, ref):
    """
    Probability matching of ``fields`` (field, cell), in place.

    The values at the flat indices ``cells`` of each field are ranked (ties
    keep the order of the cells) and replaced by the sorted reference values
    ``ref`` of the same rank.
    """
    values = fields[:, cells]
    order = np.argsort(values, axis=1, kind="stable")
    matched = np.empty_like(values)
    np.put_along_axis(matched, order, np.broadcast_to(ref, values.shape), axis=1)
    fields[:, cells] = matched


def rank_match(fields, cells, ref, use_compiled, n_threads):
    """Run :func:`rank_match_numpy` or the compiled kernel (same arguments)."""
    if use_compiled:
        _nowcast.rank_match(
            fields,
            np.ascontiguousarray(cells, dtype=np.int64),
            np.ascontiguousarray(ref, dtype=np.float64),
            int(n_threads or 0),
        )
    else:
        rank_match_numpy(fields, np.asarray(cells), np.asarray(ref))
