#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
NumPy reference of the ensemble kernels and the dispatch to the compiled ones.

The counter-based random numbers (Philox-4x64-10, read from
:class:`numpy.random.Philox`), the random errors added to the radar fields,
and the statistics over the members. Every function here has the same
arguments for the NumPy and the compiled engine (``use_compiled``); the
compiled kernel is ``_ensemble.cpp``. See :mod:`radarx.retrieve.ensemble`.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from . import _ensemble

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _ensemble = None
    HAS_COMPILED_KERNEL = False

#: kinds of perturbed field, in the order of the kernel
KINDS = ("dbz", "zdr", "phidp", "rhohv", "vel")
BITS = {"dbz": 1, "zdr": 2, "phidp": 4, "rhohv": 8, "vel": 16}
DB = 10.0 / math.log(10.0)  # dB per unit of fractional power error
_SCALE = 1.0 / 9007199254740992.0  # 2**-53
_TWO_PI = 6.283185307179586
MAX_SEED = 2**64


def make_key(seed, stream):
    """Key (two words) of the generator for a seed and a stream (sweep)."""
    seed = int(seed)
    if not 0 <= seed < MAX_SEED:
        raise ValueError("seed must be an integer in [0, 2**64)")
    return seed, int(stream)


def words(key, member, block, n):
    """Philox-4x64-10 words (n, 4) of the gates 0..n-1 of one member and block."""
    bit_generator = np.random.Philox(
        key=key[0] + (key[1] << 64), counter=[0, int(member), int(block), 0]
    )
    return bit_generator.random_raw(4 * n).reshape(n, 4)


def box_muller(w0, w1):
    """Two Gaussian numbers from two arrays of 64-bit words."""
    u1 = ((w0 >> np.uint64(11)) + np.uint64(1)) * _SCALE
    u2 = (w1 >> np.uint64(11)) * _SCALE
    radius = np.sqrt(-2.0 * np.log(u1))
    angle = _TWO_PI * u2
    return radius * np.cos(angle), radius * np.sin(angle)


def member_normals(key, member, block):
    """Four Gaussian numbers of one member (scalars: calibration and prior)."""
    w = words(key, member, block, 1)[0]
    a, b = box_muller(w[0], w[1])
    c, d = box_muller(w[2], w[3])
    return np.array([a, b, c, d], dtype=np.float64)


def gate_normals(key, member, n, velocity):
    """The five Gaussian numbers of every gate of one member, shape (5, n)."""
    z = np.zeros((5, n))
    w = words(key, member, 0, n)
    z[0], z[1] = box_muller(w[:, 0], w[:, 1])
    z[2], z[3] = box_muller(w[:, 2], w[:, 3])
    if velocity:
        w = words(key, member, 1, n)
        z[4], _ = box_muller(w[:, 0], w[:, 1])
    return z


# --------------------------------------------------------------------------
# random errors of the radar fields
# --------------------------------------------------------------------------


def _apply(kind, v, fac, z, xh, xv, biases):
    """Perturbed values of one field of one member."""
    if kind == "dbz":
        return v + biases[0] + DB * xh
    if kind == "zdr":
        return v + biases[1] + DB * (xh - xv)
    if kind == "phidp":
        return v + fac[3] * z[2]
    if kind == "rhohv":
        return np.clip(v + fac[4] * z[3], 0.0, 1.0)
    return v + fac[5] * z[4]


def _perturb_numpy(values, kinds, fac, key, m0, bias_dbz, bias_zdr):
    n = fac.shape[1]
    out = np.empty((len(bias_dbz), len(kinds), n))
    for j in range(len(bias_dbz)):
        z = gate_normals(key, m0 + j, n, "vel" in kinds)
        xh = fac[0] * z[0]
        xv = fac[1] * z[0] + fac[2] * z[1]
        for s, kind in enumerate(kinds):
            out[j, s] = _apply(
                kind, values[kind], fac, z, xh, xv, (bias_dbz[j], bias_zdr[j])
            )
    return out


def perturb(values, kinds, fac, key, m0, bias_dbz, bias_zdr, use_compiled, n_threads):
    """
    Radar fields with random errors, shape (members, fields, gates).

    ``values`` maps the kinds in ``kinds`` to flat float64 arrays; ``fac`` is
    the (6, gates) array of the error model; ``bias_dbz`` and ``bias_zdr``
    hold one calibration offset per member; the members are numbered from
    ``m0``.
    """
    if use_compiled:
        dummy = np.zeros(1)
        return _ensemble.perturb(
            *[values.get(k, dummy) for k in KINDS],
            sum(BITS[k] for k in kinds),
            fac,
            bias_dbz,
            bias_zdr,
            key[0],
            key[1],
            m0,
            int(n_threads or 0),
        )
    return _perturb_numpy(values, kinds, fac, key, m0, bias_dbz, bias_zdr)


# --------------------------------------------------------------------------
# statistics over the members
# --------------------------------------------------------------------------


def _statistics_numpy(stack, q, min_valid):
    x = stack.astype(np.float64)
    ok = np.isfinite(x)
    count = ok.sum(axis=0)
    total = np.zeros(x.shape[1])
    for m in range(x.shape[0]):
        total += np.where(ok[m], x[m], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
        squares = np.zeros(x.shape[1])
        for m in range(x.shape[0]):
            d = np.where(ok[m], x[m] - mean, 0.0)
            squares += d * d
        std = np.where(count > 1, np.sqrt(squares / (count - 1)), np.nan)
    srt = np.sort(np.where(ok, x, np.inf), axis=0)
    pct = np.empty((len(q), x.shape[1]))
    last = np.maximum(count - 1, 0)
    for k, p in enumerate(q):
        pos = last * (p / 100.0)
        lo = np.floor(pos).astype(np.int64)
        hi = np.minimum(lo + 1, last)
        lower = np.take_along_axis(srt, lo[None], axis=0)[0]
        upper = np.take_along_axis(srt, hi[None], axis=0)[0]
        with np.errstate(invalid="ignore"):
            pct[k] = lower + (upper - lower) * (pos - lo)
    bad = (count < min_valid) | (count == 0)
    mean[bad] = np.nan
    std[bad] = np.nan
    pct[:, bad] = np.nan
    return mean, std, pct


def statistics(stack, q, min_valid, use_compiled, n_threads):
    """
    Mean, standard deviation (``n - 1``) and percentiles over the members.

    ``stack`` is (members, gates); the percentiles ``q`` (0-100) are linear
    interpolations between the order statistics. Gates with fewer than
    ``min_valid`` finite members are NaN.
    """
    stack = np.ascontiguousarray(stack, dtype=np.float32)
    if use_compiled:
        return _ensemble.statistics(
            stack, np.asarray(q, dtype=np.float64), int(min_valid), int(n_threads or 0)
        )
    return _statistics_numpy(stack, q, min_valid)


def _mode_numpy(stack, n_codes, min_valid):
    x = stack.astype(np.float64)
    code = np.where(np.isfinite(x), np.trunc(np.nan_to_num(x)), -1)
    valid = (code >= 0) & (code < n_codes)
    counts = np.stack([((code == c) & valid).sum(axis=0) for c in range(n_codes)])
    best = counts.argmax(axis=0)
    total = valid.sum(axis=0)
    top = np.take_along_axis(counts, best[None], axis=0)[0]
    bad = (total < min_valid) | (total == 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(bad, np.nan, top / total)
    return np.where(bad, np.nan, best.astype(np.float64)), share


def mode(stack, min_valid, use_compiled, n_threads):
    """Most frequent class code (the lowest on ties) and its share of the members."""
    stack = np.ascontiguousarray(stack, dtype=np.float32)
    finite = stack[np.isfinite(stack)]
    n_codes = int(finite.max()) + 1 if finite.size and finite.max() >= 0 else 1
    if use_compiled:
        return _ensemble.mode(stack, n_codes, int(min_valid), int(n_threads or 0))
    return _mode_numpy(stack, n_codes, min_valid)
