#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
NumPy reference implementation of the spectral kernels.

Same functions, arguments and results as the compiled
``radarx.level1._moments`` module; used when the extension is not built and
as the test oracle. A spectrum is one row of ``K`` bins, bin ``j`` at the
velocity ``(j - K // 2) dv``. The formulas are in the docstrings of
:mod:`radarx.level1.spectra` and :mod:`radarx.level1.clutter`.
"""

import numpy as np

_CHUNK = 2048  # rows per block, to bound the size of the temporary arrays


def _check(spectra):
    spectra = np.ascontiguousarray(spectra, dtype=np.float64)
    if spectra.ndim != 2:
        raise ValueError("spectra must be 2-D (rows, bins)")
    return spectra


def _check_rows(a, n, name):
    a = np.ascontiguousarray(a, dtype=np.float64)
    if a.ndim != 1 or a.shape[0] != n:
        raise ValueError(f"{name} must have one value per spectrum")
    return a


def noise_rows(spectra, n_avg, min_bins, n_threads=0):
    """Hildebrand and Sekhon (1974) noise level per bin of every row."""
    s = _check(spectra)
    n, k = s.shape
    if min_bins < 1 or min_bins > k:
        raise ValueError("min_bins must be in 1..bins")
    out = np.full(n, np.nan)
    good = np.isfinite(s).all(axis=1)
    for i0 in range(0, n, _CHUNK):
        sl = slice(i0, min(n, i0 + _CHUNK))
        rows = s[sl]
        srt = np.sort(rows, axis=1)
        cs = np.cumsum(srt, axis=1)
        cs2 = np.cumsum(srt * srt, axis=1)
        cnt = np.arange(1, k + 1, dtype=np.float64)
        mean = cs / cnt
        var = cs2 / cnt - mean * mean
        ok = mean * mean >= n_avg * var
        ok[:, : min_bins - 1] = False
        ok[:, min_bins - 1] = True
        idx = k - 1 - np.argmax(ok[:, ::-1], axis=1)
        out[sl] = mean[np.arange(rows.shape[0]), idx]
    out[~good] = np.nan
    return out


def _runs(above):
    """Number of leading True values of every row."""
    return np.cumprod(above, axis=1, dtype=np.int64).sum(axis=1)


def moments(spectra, noise, n_avg, n_sigma, mode, dv, win_var, n_threads=0):
    """Noise, power, mean velocity, width, skewness, kurtosis and bins per row."""
    s = _check(spectra)
    n, k = s.shape
    noise = _check_rows(noise, n, "noise")
    if mode not in (0, 1):
        raise ValueError("mode must be 0 or 1")
    out = np.full((n, 7), np.nan)
    out[:, 0] = noise
    fac = 1.0 + n_sigma / np.sqrt(n_avg)
    valid = np.isfinite(noise) & np.isfinite(s).all(axis=1)
    for i0 in range(0, n, _CHUNK):
        sl = slice(i0, min(n, i0 + _CHUNK))
        rows = np.flatnonzero(valid[sl]) + i0
        if rows.size == 0:
            continue
        sr, nr_ = s[rows], noise[rows]
        thr = nr_ * fac
        r = np.arange(rows.size)
        k0 = np.argmax(sr, axis=1)
        out[rows, 6] = 0.0
        sig = sr[r, k0] > thr
        if not sig.any():
            continue
        rows, sr, nr_, thr, k0 = rows[sig], sr[sig], nr_[sig], thr[sig], k0[sig]
        r = np.arange(rows.size)
        if mode == 0:
            right = (k0[:, None] + np.arange(1, k)[None, :]) % k
            left = (k0[:, None] - np.arange(1, k)[None, :]) % k
            run_r = _runs(sr[r[:, None], right] > thr[:, None])
            run_l = _runs(sr[r[:, None], left] > thr[:, None])
            nl = np.minimum(run_l, k - 1)
            nr = np.minimum(run_r, k - 1 - nl)
            grid = np.arange(-(k - 1), k)
            idx = (k0[:, None] + grid[None, :]) % k
            sel = (grid[None, :] >= -nl[:, None]) & (grid[None, :] <= nr[:, None])
            wgt = (sr[r[:, None], idx] - nr_[:, None]) * sel
            delta = np.broadcast_to(grid[None, :].astype(np.float64), wgt.shape)
            count = sel.sum(axis=1)
        else:
            delta = (
                (np.arange(k)[None, :] - k0[:, None] + k // 2) % k - k // 2
            ).astype(np.float64)
            sel = sr > thr[:, None]
            wgt = (sr - nr_[:, None]) * sel
            count = sel.sum(axis=1)
        power = wgt.sum(axis=1)
        m1 = (wgt * delta).sum(axis=1) / power
        d = delta - m1[:, None]
        m2 = (wgt * d * d).sum(axis=1) / power
        m3 = (wgt * d * d * d).sum(axis=1) / power
        m4 = (wgt * d * d * d * d).sum(axis=1) / power
        out[rows, 1] = power
        out[rows, 2] = (k0 - k // 2 + m1) * dv
        out[rows, 3] = dv * np.sqrt(np.maximum(m2 - win_var, 0.0))
        pos = m2 > 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            out[rows, 4] = np.where(pos, m3 / (m2 * np.sqrt(m2)), np.nan)
            out[rows, 5] = np.where(pos, m4 / (m2 * m2), np.nan)
        out[rows, 6] = count
    return out


def _solve3(a, b):
    """Cramer's rule for stacks of 3 x 3 systems; returns (x, ok)."""

    def det(m):
        return (
            m[:, 0, 0] * (m[:, 1, 1] * m[:, 2, 2] - m[:, 1, 2] * m[:, 2, 1])
            - m[:, 0, 1] * (m[:, 1, 0] * m[:, 2, 2] - m[:, 1, 2] * m[:, 2, 0])
            + m[:, 0, 2] * (m[:, 1, 0] * m[:, 2, 1] - m[:, 1, 1] * m[:, 2, 0])
        )

    d = det(a)
    scale = np.abs(a).max(axis=(1, 2))
    ok = np.abs(d) > 1e-12 * scale**3
    x = np.zeros((a.shape[0], 3))
    safe = np.where(ok, d, 1.0)
    for c in range(3):
        m = a.copy()
        m[:, :, c] = b
        x[:, c] = det(m) / safe
    return x, ok


def _interpolate(row_s, noise, lo, hi, k):
    """Linear fill of bins lo..hi from the bins next to them, per row."""
    a, b = lo - 1, hi + 1
    n = row_s.shape[0]
    fill = np.empty((n, hi - lo + 1))
    for col, kk in enumerate(range(lo, hi + 1)):
        if a < 0 and b >= k:
            fill[:, col] = noise
        elif a < 0:
            fill[:, col] = row_s[:, b]
        elif b >= k:
            fill[:, col] = row_s[:, a]
        else:
            fill[:, col] = row_s[:, a] + (row_s[:, b] - row_s[:, a]) * (kk - a) / (
                b - a
            )
    return fill


def notch(spectra, noise, n_avg, n_sigma, half_width, fill, n_threads=0):
    """Replace the bins around zero velocity; returns spectra and removed power."""
    s = _check(spectra)
    n, k = s.shape
    noise = _check_rows(noise, n, "noise")
    if half_width < 0:
        raise ValueError("half_width must not be negative")
    if fill not in (0, 1, 2):
        raise ValueError("fill must be 0, 1 or 2")
    out = s.copy()
    removed = np.full(n, np.nan)
    c = k // 2
    lo, hi = max(0, c - half_width), min(k - 1, c + half_width)
    fac = 1.0 + n_sigma / np.sqrt(n_avg)
    valid = np.isfinite(noise) & np.isfinite(s).all(axis=1)
    for i0 in range(0, n, _CHUNK):
        sl = slice(i0, min(n, i0 + _CHUNK))
        rows = np.flatnonzero(valid[sl]) + i0
        if rows.size == 0:
            continue
        sr, nr_ = s[rows], noise[rows]
        new = np.empty((rows.size, hi - lo + 1))
        done = np.zeros(rows.size, dtype=bool)
        if fill == 2:
            thr = nr_ * fac
            kk = np.arange(k)
            outside = (kk < lo) | (kk > hi)
            masked = np.where(outside[None, :], sr, -np.inf)
            k0 = np.argmax(masked, axis=1)
            r = np.arange(rows.size)
            has = (masked[r, k0] > thr) & outside.any()
            right = (k0[:, None] + np.arange(1, k)[None, :]) % k
            left = (k0[:, None] - np.arange(1, k)[None, :]) % k
            run_r = _runs((sr[r[:, None], right] > thr[:, None]) & outside[right])
            run_l = _runs((sr[r[:, None], left] > thr[:, None]) & outside[left])
            nl = np.minimum(run_l, k - 1)
            nr = np.minimum(run_r, k - 1 - nl)
            grid = np.arange(-(k - 1), k)
            idx = (k0[:, None] + grid[None, :]) % k
            sel = (grid[None, :] >= -nl[:, None]) & (grid[None, :] <= nr[:, None])
            use = has & (nl + nr + 1 >= 3)
            with np.errstate(invalid="ignore", divide="ignore"):
                y = np.where(sel, sr[r[:, None], idx] - nr_[:, None], 1.0)
                w = np.where(sel, y * y, 0.0)
                ly = np.log(y)
            x = grid[None, :].astype(np.float64)
            powers = [np.ones_like(x), x, x**2, x**3, x**4]
            amat = np.zeros((rows.size, 3, 3))
            bvec = np.zeros((rows.size, 3))
            for p in range(3):
                for q in range(3):
                    amat[:, p, q] = (w * powers[p + q]).sum(axis=1)
                bvec[:, p] = (w * powers[p] * ly).sum(axis=1)
            amat[~use] = np.eye(3)
            coef, ok = _solve3(amat, bvec)
            good = use & ok & (coef[:, 2] < 0.0)
            xn = ((np.arange(lo, hi + 1)[None, :] - k0[:, None] + k // 2) % k) - k // 2
            xn = xn.astype(np.float64)
            with np.errstate(over="ignore", invalid="ignore"):
                model = nr_[:, None] + np.exp(
                    coef[:, 0:1] + coef[:, 1:2] * xn + coef[:, 2:3] * xn * xn
                )
            new[good] = model[good]
            done = good
        rest = ~done
        if rest.any():
            if fill == 0:
                new[rest] = nr_[rest, None]
            else:
                new[rest] = _interpolate(sr[rest], nr_[rest], lo, hi, k)
        res = out[rows]
        res[:, lo : hi + 1] = new
        out[rows] = res
        removed[rows] = (sr[:, lo : hi + 1] - new).sum(axis=1)
    return out, removed


def unfold(vel, ref, va, max_gap, n_threads=0):
    """Unfold the velocities of the rays gate by gate; returns (velocity, folds)."""
    vel = np.ascontiguousarray(vel, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if vel.ndim != 2:
        raise ValueError("vel must be 2-D (rays, gates)")
    has_ref = ref.size > 0
    if has_ref and ref.shape != vel.shape:
        raise ValueError("ref must have the shape of vel")
    if not va > 0.0:
        raise ValueError("the Nyquist velocity must be positive")
    n, g = vel.shape
    out = vel.copy()
    fold = np.zeros((n, g), dtype=np.int32)
    last = np.full(n, np.nan)
    missing = np.zeros(n, dtype=np.int64)
    for j in range(g):
        v = vel[:, j]
        fin = np.isfinite(v)
        use_ref = fin & np.isfinite(ref[:, j]) if has_ref else np.zeros(n, bool)
        use_last = fin & ~use_ref & np.isfinite(last) & (missing <= max_gap)
        target = np.where(
            use_ref, ref[:, j] if has_ref else 0.0, np.where(use_last, last, np.nan)
        )
        has = np.isfinite(target)
        with np.errstate(invalid="ignore"):
            nf = np.floor((target - v) / (2.0 * va) + 0.5)
        nf = np.where(has, nf, 0.0)
        fold[:, j] = nf.astype(np.int32)
        out[:, j] = np.where(fin, v + 2.0 * va * nf, v)
        last = np.where(fin, out[:, j], last)
        missing = np.where(fin, 0, missing + 1)
    return out, fold
