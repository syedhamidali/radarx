#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
NumPy reference of the ray-angle check and dispatch to the compiled one.

The compiled kernel is ``_azimuth_qc.cpp``. Both engines have the same
arguments and give the same result. See :mod:`radarx.retrieve.azimuth_qc`.
"""

from __future__ import annotations

import numpy as np

try:
    from . import _azimuth_qc

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _azimuth_qc = None
    HAS_COMPILED_KERNEL = False

OUTLIER = 1
DUPLICATE = 2
GAP = 4
REVERSAL = 8
MISSING = 16
STATIONARY = 0.01  # degrees per ray: a smaller typical step is no scan


def wrap180(x):
    """Angle difference wrapped to [-180, 180) degrees."""
    return x - 360.0 * np.floor((x + 180.0) / 360.0)


def wrap360(x):
    """Angle wrapped to [0, 360) degrees."""
    return x - 360.0 * np.floor(x / 360.0)


def _side_residual(a, p, i, members, delta):
    """Median deviation of ray ``i`` from the positions its neighbours predict."""
    j = np.asarray(members, dtype=np.int64)
    return np.median(wrap180(a[i] - a[j] - (p[i] - p[j]) * delta))


def flag_azimuth_numpy(angle, tolerance, half_window, duplicate_fraction, gap_factor):
    """
    Flags, repaired angles and the signed ray spacing of one sweep.

    ``angle`` holds the angle of every ray in degrees (azimuth of a PPI,
    elevation of an RHI, NaN for a missing value). Returns ``(flags, fixed,
    delta)``: ``flags`` (uint8, bits ``OUTLIER``, ``DUPLICATE``, ``GAP``,
    ``REVERSAL``, ``MISSING``), the angle of every ray with the bad ones
    replaced by an interpolation between the nearest good rays (degrees in
    [0, 360) for the repaired rays, the input for the others) and the median
    angle step per ray, signed (NaN when it cannot be found).

    A ray is an outlier when it deviates by more than ``tolerance`` ray
    spacings from the positions that the rays on its left, and also from the
    positions that the rays on its right, predict (the median of up to
    ``half_window`` rays on each side, extrapolated with the median spacing).
    Needing both sides keeps a persistent jump (a gap) from being reported
    as a run of outliers.
    """
    angle = np.asarray(angle, dtype=np.float64)
    n = angle.size
    flags = np.zeros(n, dtype=np.uint8)
    fixed = angle.copy()
    finite = np.isfinite(angle)
    flags[~finite] |= MISSING
    p = np.flatnonzero(finite)
    m = p.size
    if m < 5:
        return flags, fixed, float("nan")
    a = angle[p]
    step = wrap180(np.diff(a)) / np.diff(p)
    delta = float(np.median(step))
    if abs(delta) < STATIONARY:  # the antenna does not scan: nothing to check
        return flags, fixed, 0.0
    tol = tolerance * abs(delta)

    outlier = np.zeros(m, dtype=bool)
    for i in range(m):
        left = [j for j in range(i - 1, -1, -1) if p[i] - p[j] <= half_window]
        right = [j for j in range(i + 1, m) if p[j] - p[i] <= half_window]
        e_left = _side_residual(a, p, i, left, delta) if len(left) >= 2 else None
        e_right = _side_residual(a, p, i, right, delta) if len(right) >= 2 else None
        if e_left is not None and e_right is not None:
            outlier[i] = min(abs(e_left), abs(e_right)) > tol
        elif e_left is not None:
            outlier[i] = abs(e_left) > tol
        elif e_right is not None:
            outlier[i] = abs(e_right) > tol
    flags[p[outlier]] |= OUTLIER

    # steps between consecutive rays that are not outliers
    duplicate = np.zeros(m, dtype=bool)
    good = np.flatnonzero(~outlier)
    for k in range(1, good.size):
        i0, i1 = good[k - 1], good[k]
        ratio = wrap180(a[i1] - a[i0]) / (delta * (p[i1] - p[i0]))
        if ratio > gap_factor:
            flags[p[i0]] |= GAP
        elif abs(ratio) < duplicate_fraction:
            duplicate[i1] = True
        elif ratio < -duplicate_fraction:
            flags[p[i1]] |= REVERSAL
    flags[p[duplicate]] |= DUPLICATE

    # repair: linear interpolation in the ray number between the good rays
    bad = np.ones(n, dtype=bool)
    bad[p[~outlier & ~duplicate]] = False
    anchors = np.flatnonzero(~bad)
    for i in np.flatnonzero(bad):
        before = anchors[anchors < i]
        after = anchors[anchors > i]
        if before.size and after.size:
            lo, hi = before[-1], after[0]
            value = angle[lo] + wrap180(angle[hi] - angle[lo]) * (i - lo) / (hi - lo)
        elif before.size:
            value = angle[before[-1]] + (i - before[-1]) * delta
        elif after.size:
            value = angle[after[0]] - (after[0] - i) * delta
        else:  # pragma: no cover - needs fewer than one good ray
            continue
        fixed[i] = wrap360(value)
    return flags, fixed, delta


def flag_azimuth(
    angles,
    tolerance,
    half_window,
    duplicate_fraction,
    gap_factor,
    use_compiled,
    n_threads,
):
    """Run :func:`flag_azimuth_numpy` for every sweep or the compiled kernel."""
    if use_compiled:
        return _azimuth_qc.flag_azimuth(
            [np.ascontiguousarray(a, dtype=np.float64) for a in angles],
            float(tolerance),
            int(half_window),
            float(duplicate_fraction),
            float(gap_factor),
            int(n_threads or 0),
        )
    return [
        flag_azimuth_numpy(a, tolerance, half_window, duplicate_fraction, gap_factor)
        for a in angles
    ]
