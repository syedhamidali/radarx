#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Reference implementation of the UNRAVEL dealiasing driver (plain Python).

This is the specification that the compiled kernel ``_unravel.cpp`` follows
step by step, and the fallback when the kernel is not built. It is slow (every
gate is visited in a Python loop) and meant for small sweeps and for the tests
that compare it with the kernel. The method is that of Louf et al. (2020),
described in :mod:`radarx.retrieve.dealias`; the choices that the paper leaves
open are listed there and in :data:`DEFAULTS`.

A sweep is a velocity array ``(ray, gate)`` with NaN for missing gates. The
state of every gate is a flag, as in the paper: -3 missing, 0 not processed,
1 processed and unchanged, 2 processed and unfolded. Rays are neighbours in
index order, and the last ray follows the first one when ``wrap`` is true.
"""

import math

import numpy as np

#: The parameters of the driver. ``alpha`` (0.8), ``alpha_first`` (0.6),
#: ``beta`` (0.6), ``z_cutoff`` (2), ``pre_window`` (10 gates), the range and
#: azimuth windows 6 and the box windows (5, 5), (20, 10), (40, 10), (80, 20)
#: are given in the paper. The others are radarx choices, taken where the paper
#: is silent: the tolerance of the pre-processing, the second window 12, the
#: size of the box around the closest reference, the box of the check and the
#: windows of the 3-D check.
DEFAULTS = {
    "alpha": 0.8,
    "alpha_first": 0.6,
    "alpha_pre": 0.3,
    "z_cutoff": 2.0,
    "pre_window": 10,
    "beta": 0.6,
    "windows": (6, 12),
    "boxes": ((5, 5), (20, 10), (40, 10), (80, 20)),
    "closest_half": 2,
    "check_box": (80, 20),
    "window_3d": (10, 3),
}


def fold_to(vref, v, vn):
    """``v`` moved by a multiple of ``2 vn`` to the value nearest to ``vref``."""
    return v + 2.0 * vn * math.floor((vref - v) / (2.0 * vn) + 0.5)


def accept(vref, v, vn, alpha):
    """Eq. (3) of the paper: the nearest unfolding of ``v`` if within ``alpha vn``."""
    u = fold_to(vref, v, vn)
    return u if abs(u - vref) < alpha * vn else None


def median(values):
    """Median of a list (mean of the two middle values for an even count)."""
    s = sorted(values)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) * 0.5


def mean(values):
    total = 0.0
    for x in values:
        total += x
    return total / len(values)


# --------------------------------------------------------------------------
# initialisation
# --------------------------------------------------------------------------


def preprocess(vel, vn, alpha, z_cutoff, window):
    """
    Modified z-score filter and first unfolding of every ray (paper, 3a 1).

    A window of ``window`` gates slides along the ray from the first gate. For
    its valid gates the median ``m`` and the median absolute deviation are
    computed; a gate with ``|z| > z_cutoff``, ``z = (v - m) / MAD``, is
    unfolded to the mean of the window gates that have the sign of ``m`` if
    that brings it within ``alpha vn``, else to the mean of the others, else it
    is made missing. The modified values stay in the array, so later windows
    see them.
    """
    v = vel.copy()
    nray, ngate = v.shape
    for i in range(nray):
        row = v[i]
        for n in range(ngate):
            idx = [
                k for k in range(n, min(n + window, ngate)) if not math.isnan(row[k])
            ]
            if not idx:
                continue
            vals = [row[k] for k in idx]
            m = median(vals)
            mad = median([abs(x - m) for x in vals])
            pos = [x for x in vals if x > 0]
            neg = [x for x in vals if x < 0]
            mean_pos = mean(pos) if pos else math.nan
            mean_neg = mean(neg) if neg else math.nan
            first, second = (mean_pos, mean_neg) if m >= 0 else (mean_neg, mean_pos)
            for k in idx:
                x = row[k]
                if mad > 0:
                    z = (x - m) / mad
                else:
                    z = 0.0 if x == m else math.inf
                if abs(z) <= z_cutoff:
                    continue
                done = None
                for ref in (first, second):
                    if not math.isnan(ref):
                        done = accept(ref, x, vn, alpha)
                        if done is not None:
                            break
                row[k] = math.nan if done is None else done
    return v


def circ_diff(a, b, wrap):
    d = abs(a - b)
    return min(d, 360.0 - d) if wrap else d


def reference_radials(v, azimuth, vn, beta, wrap):
    """
    The two reference radials (paper, 3a 2). Returns ``(ref1, ref2)``.

    A ray is eligible if it holds at least two thirds of the average number of
    valid gates per ray and at most three gates with ``|v| >= beta vn`` (the
    paper asks for none close to the Nyquist velocity; three tolerate noise).
    If no ray is, the first condition and then the second are dropped. The
    first reference is the eligible ray with the smallest mean ``|v|``. The
    second is the local minimum of that curve, at least 120 degrees from the
    first, that stands deepest below the lower of the two maxima around it (its
    prominence); without one, it is the first reference.
    """
    nray, ngate = v.shape
    valid = ~np.isnan(v)
    count = valid.sum(axis=1)
    if count.sum() == 0:
        return 0, 0
    curve = np.full(nray, math.nan)
    near = np.zeros(nray, dtype=np.int64)
    for i in range(nray):
        if count[i]:
            vals = [abs(x) for x in v[i] if not math.isnan(x)]
            curve[i] = mean(vals)
            near[i] = sum(1 for x in vals if x >= beta * vn)
    average = float(count.sum()) / nray
    quiet = near <= 3
    eligible = (count >= (2.0 / 3.0) * average) & (count > 0) & quiet
    if not eligible.any():
        eligible = (count > 0) & quiet
    if not eligible.any():
        eligible = count > 0
    rays = [i for i in range(nray) if eligible[i]]
    s = [curve[i] for i in rays]
    ref1 = rays[min(range(len(s)), key=lambda k: (s[k], k))]
    e = len(s)
    best, ref2 = -1.0, ref1
    for p in range(e):
        left = s[(p - 1) % e] if wrap or p > 0 else math.inf
        right = s[(p + 1) % e] if wrap or p < e - 1 else math.inf
        if not (s[p] <= left and s[p] <= right):
            continue
        if circ_diff(float(azimuth[rays[p]]), float(azimuth[ref1]), wrap) < 120.0:
            continue
        # prominence: highest value met going each way before a lower one
        highs = []
        for step in (1, -1):
            top, q, steps = s[p], p, 0
            while steps < e - 1:
                q = (q + step) % e if wrap else q + step
                if q < 0 or q >= e:
                    break
                steps += 1
                if s[q] < s[p]:
                    break
                top = max(top, s[q])
            highs.append(top)
        depth = min(highs) - s[p]
        if depth > best:
            best, ref2 = depth, rays[p]
    return ref1, ref2


def init_flags(v, ref1, ref2, vn, beta, alpha, wrap):
    """Flags and values after the reference radials and their neighbours (3a 2)."""
    nray, ngate = v.shape
    out = v.copy()
    flag = np.where(np.isnan(v), -3, 0).astype(np.int8)
    refs = [ref1] if ref1 == ref2 else [ref1, ref2]
    for r in refs:
        for g in range(ngate):
            if flag[r, g] == 0 and abs(v[r, g]) < beta * vn:
                flag[r, g] = 1
    for r in refs:
        for nb in (r - 1, r + 1):
            if wrap:
                nb %= nray
            elif not 0 <= nb < nray:
                continue
            for g in range(ngate):
                if flag[nb, g] == 0 and flag[r, g] > 0:
                    u = accept(out[r, g], v[nb, g], vn, alpha)
                    if u is not None:
                        out[nb, g] = u
                        flag[nb, g] = 1 if u == v[nb, g] else 2
    return out, flag


# --------------------------------------------------------------------------
# dealiasing modules
# --------------------------------------------------------------------------


def _ray(i, nray, wrap):
    if wrap:
        return i % nray
    return i if 0 <= i < nray else -1


def first_pass(v, out, flag, ref1, ref2, vn, alpha, wrap):
    """Azimuthal continuity in four sectors from the reference radials (3a 3)."""
    nray, ngate = v.shape
    if wrap:
        dcw = nray if ref1 == ref2 else (ref2 - ref1) % nray
        dccw = nray - dcw
        counts = {
            1: (dcw // 2, max(0, dccw - 1 - dccw // 2)),
            2: (dccw // 2, max(0, dcw - 1 - dcw // 2)),
        }
    else:
        lo, hi = min(ref1, ref2), max(ref1, ref2)
        d = hi - lo
        if ref1 == ref2:
            counts = {1: (nray - 1 - ref1, ref1), 2: (0, 0)}
        else:
            cnt = {lo: (d // 2, lo), hi: (nray - 1 - hi, max(0, d - 1 - d // 2))}
            counts = {1: cnt[ref1], 2: cnt[ref2]}
    plan = [(ref1, 1, counts[1][0]), (ref1, -1, counts[1][1])]
    if ref2 != ref1 or wrap:
        plan += [(ref2, 1, counts[2][0]), (ref2, -1, counts[2][1])]
    for start, direction, count in plan:
        for k in range(1, count + 1):
            i = _ray(start + direction * k, nray, wrap)
            for g in range(ngate):
                if flag[i, g] != 0:
                    continue
                refs = []
                for j in (1, 2, 3):
                    p = _ray(i - direction * j, nray, wrap)
                    if p >= 0 and flag[p, g] > 0:
                        refs.append(out[p, g])
                if len(refs) < 2:
                    continue
                u = accept(mean(refs), v[i, g], vn, alpha)
                if u is not None:
                    out[i, g] = u
                    flag[i, g] = 1 if u == v[i, g] else 2


def _set(v, out, flag, i, g, u):
    out[i, g] = u
    flag[i, g] = 1 if u == v[i, g] else 2


def range_pass(v, out, flag, vn, alpha, window):
    """Continuity along the ray, onward then backward (3b 1): mean of the window."""
    nray, ngate = v.shape
    for i in range(nray):
        for g in range(ngate):
            if flag[i, g] == 0:
                refs = [
                    out[i, k] for k in range(max(0, g - window), g) if flag[i, k] > 0
                ]
                if refs:
                    u = accept(mean(refs), v[i, g], vn, alpha)
                    if u is not None:
                        _set(v, out, flag, i, g, u)
        for g in range(ngate - 1, -1, -1):
            if flag[i, g] == 0:
                refs = [
                    out[i, k]
                    for k in range(g + 1, min(ngate, g + window + 1))
                    if flag[i, k] > 0
                ]
                if refs:
                    u = accept(mean(refs), v[i, g], vn, alpha)
                    if u is not None:
                        _set(v, out, flag, i, g, u)


def azimuth_pass(v, out, flag, ref1, vn, alpha, window, wrap):
    """Continuity across rays from the first reference, clockwise then counter (3b 2)."""
    nray, ngate = v.shape
    if wrap:
        order = [
            (1, [(ref1 + k) % nray for k in range(1, nray)]),
            (-1, [(ref1 - k) % nray for k in range(1, nray)]),
        ]
    else:
        order = [(1, list(range(ref1 + 1, nray))), (-1, list(range(ref1 - 1, -1, -1)))]
    for direction, rays in order:
        for i in rays:
            for g in range(ngate):
                if flag[i, g] != 0:
                    continue
                refs = []
                for j in range(1, window + 1):
                    p = _ray(i - direction * j, nray, wrap)
                    if p >= 0 and flag[p, g] > 0:
                        refs.append(out[p, g])
                if refs:
                    u = accept(median(refs), v[i, g], vn, alpha)
                    if u is not None:
                        _set(v, out, flag, i, g, u)


def box_pass(v, out, flag, vn, alpha, size, wrap):
    """Median of the processed gates in a box around the gate (3b 3)."""
    nray, ngate = v.shape
    half_g, half_a = size[0] // 2, size[1] // 2
    for i in range(nray):
        for g in range(ngate):
            if flag[i, g] != 0:
                continue
            refs = []
            for a in range(i - half_a, i + half_a + 1):
                p = _ray(a, nray, wrap)
                if p < 0:
                    continue
                for k in range(max(0, g - half_g), min(ngate, g + half_g + 1)):
                    if flag[p, k] > 0:
                        refs.append(out[p, k])
            if refs:
                u = accept(median(refs), v[i, g], vn, alpha)
                if u is not None:
                    _set(v, out, flag, i, g, u)


def closest_pass(v, out, flag, x, y, vn, half, wrap):
    """The remaining gates, with the median around the closest processed gate (3b 4)."""
    nray, ngate = v.shape
    done = flag > 0
    px = x[done]
    py = y[done]
    pi_, pg = np.nonzero(done)
    if px.size == 0:
        return
    snapshot = out.copy()
    todo = np.argwhere(flag == 0)
    for i, g in todo:
        d2 = (px - x[i, g]) * (px - x[i, g]) + (py - y[i, g]) * (py - y[i, g])
        k = int(np.argmin(d2))
        ci, cg = int(pi_[k]), int(pg[k])
        refs = []
        for a in range(ci - half, ci + half + 1):
            p = _ray(a, nray, wrap)
            if p < 0:
                continue
            for q in range(max(0, cg - half), min(ngate, cg + half + 1)):
                if done[p, q]:
                    refs.append(snapshot[p, q])
        _set(v, out, flag, i, g, fold_to(median(refs), v[i, g], vn))


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------


def readjust(vref, x, vn, alpha):
    """The value ``x`` moved to the fold nearest to ``vref`` if it is off by ``alpha vn``."""
    if abs(x - vref) >= alpha * vn:
        u = accept(vref, x, vn, alpha)
        if u is not None:
            return u
    return x


def radial_check(out, flag, rng, vn, alpha, min_gates=10):
    """Least-squares line along each ray; gates are readjusted to it (3c 1)."""
    nray, ngate = out.shape
    new = out.copy()
    for i in range(nray):
        idx = [k for k in range(ngate) if flag[i, k] > 0]
        if len(idx) < min_gates:
            continue
        n = float(len(idx))
        sx = sy = sxx = sxy = 0.0
        for k in idx:
            sx += rng[k]
            sy += out[i, k]
            sxx += rng[k] * rng[k]
            sxy += rng[k] * out[i, k]
        den = n * sxx - sx * sx
        if den == 0.0:
            continue
        a = (n * sxy - sx * sy) / den
        b = (sy - a * sx) / n
        for k in idx:
            new[i, k] = readjust(a * rng[k] + b, out[i, k], vn, alpha)
    out[:] = new


def box_check(out, flag, vn, alpha, size, wrap):
    """Readjust gates to the mean of the processed gates within one sigma around them (3c 2)."""
    nray, ngate = out.shape
    half_g, half_a = size[0] // 2, size[1] // 2
    new = out.copy()
    for i in range(nray):
        for g in range(ngate):
            if flag[i, g] <= 0:
                continue
            vals = []
            for a in range(i - half_a, i + half_a + 1):
                p = _ray(a, nray, wrap)
                if p < 0:
                    continue
                for k in range(max(0, g - half_g), min(ngate, g + half_g + 1)):
                    if flag[p, k] > 0 and not (p == i and k == g):
                        vals.append(out[p, k])
            if not vals:
                continue
            mu = mean(vals)
            var = 0.0
            for x in vals:
                var += (x - mu) * (x - mu)
            sigma = math.sqrt(var / len(vals))
            kept = [x for x in vals if abs(x - mu) <= sigma]
            if kept:
                new[i, g] = readjust(mean(kept), out[i, g], vn, alpha)
    out[:] = new


def unravel_sweep(vel, vn, azimuth, rng, x, y, wrap, params=None):
    """
    Dealias one sweep with the default UNRAVEL driver.

    Returns ``(out, flag)``: the dealiased velocity (NaN where the flag is -3)
    and the flags 1 (unchanged) and 2 (unfolded), -3 for missing gates.
    """
    p = dict(DEFAULTS, **(params or {}))
    v = preprocess(vel, vn, p["alpha_pre"], p["z_cutoff"], p["pre_window"])
    ref1, ref2 = reference_radials(v, azimuth, vn, p["beta"], wrap)
    out, flag = init_flags(v, ref1, ref2, vn, p["beta"], p["alpha_first"], wrap)
    first_pass(v, out, flag, ref1, ref2, vn, p["alpha_first"], wrap)
    for window in p["windows"]:
        if (flag == 0).any():
            range_pass(v, out, flag, vn, p["alpha"], window)
            azimuth_pass(v, out, flag, ref1, vn, p["alpha"], window, wrap)
    for size in p["boxes"]:
        if (flag == 0).any():
            box_pass(v, out, flag, vn, p["alpha"], size, wrap)
    if (flag == 0).any():
        closest_pass(v, out, flag, x, y, vn, p["closest_half"], wrap)
    radial_check(out, flag, rng, vn, p["alpha"])
    box_check(out, flag, vn, p["alpha"], p["check_box"], wrap)
    done = flag > 0
    out = np.where(done, out, np.nan)
    flag = np.where(done, np.where(out == vel, 1, 2), -3).astype(np.int8)
    return out, flag


def check_3d(out, flag, below, below_flag, iray, igate, vn, alpha, window, wrap):
    """
    Continuity with the sweep below (paper, 3c 3).

    ``iray`` and ``igate`` map every gate to the gate of the lower sweep that
    lies under it (negative where there is none). The processed gates of the
    lower sweep within ``window = (gates, rays)`` of that gate that lie within
    one standard deviation of their mean give the reference.
    """
    nray, ngate = out.shape
    nb_ray, nb_gate = below.shape
    half_g, half_a = window
    new = out.copy()
    for i in range(nray):
        for g in range(ngate):
            if flag[i, g] <= 0 or iray[i, g] < 0 or igate[i, g] < 0:
                continue
            vals = []
            for a in range(iray[i, g] - half_a, iray[i, g] + half_a + 1):
                p = _ray(a, nb_ray, wrap)
                if p < 0:
                    continue
                for k in range(
                    max(0, igate[i, g] - half_g), min(nb_gate, igate[i, g] + half_g + 1)
                ):
                    if below_flag[p, k] > 0:
                        vals.append(below[p, k])
            if not vals:
                continue
            mu = mean(vals)
            var = 0.0
            for x in vals:
                var += (x - mu) * (x - mu)
            sigma = math.sqrt(var / len(vals))
            kept = [x for x in vals if abs(x - mu) <= sigma]
            if kept:
                new[i, g] = readjust(mean(kept), out[i, g], vn, alpha)
    out[:] = new
