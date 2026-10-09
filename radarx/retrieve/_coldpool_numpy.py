#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
NumPy reference implementation of the cold-pool kernel.

Same functions, arguments and results as the compiled
``radarx.retrieve._coldpool`` module; used when the extension is not built and
as the test oracle.
"""

import numpy as np

RD = 287.04749  # gas constant of dry air [J kg-1 K-1]
RV = 461.52311  # gas constant of water vapour [J kg-1 K-1]
EPS = RD / RV
CPD = 1005.7  # specific heat of dry air at constant pressure [J kg-1 K-1]
KAPPA = RD / CPD
T0 = 273.15
P0 = 100000.0


def _esat(t):
    tc = t - T0
    return 611.2 * np.exp(17.67 * tc / (tc + 243.5))


def thermo(t, p, td, n_threads=0):
    """Mixing ratio, theta, theta_v and theta_e (rows) of 1-D inputs."""
    t, p, td = (np.asarray(a, dtype=np.float64) for a in (t, p, td))
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        d = np.where(td < t, td, t)
        d = np.where(np.isnan(td), np.nan, d)
        th = t * (P0 / p) ** KAPPA
        e = _esat(d)
        r = EPS * e / (p - e)
        thv = th * (1.0 + r / EPS) / (1.0 + r)
        tl = 1.0 / (1.0 / (d - 56.0) + np.log(t / d) / 800.0) + 56.0
        rg = 1000.0 * r
        the = (
            t
            * (P0 / p) ** (0.2854 * (1.0 - 0.28e-3 * rg))
            * np.exp((3.376 / tl - 0.00254) * rg * (1.0 + 0.81e-3 * rg))
        )
    return np.stack([r, th, thv, the])


def _cold_pool_column(z, b, bottom, threshold, top):
    ok = np.isfinite(z) & np.isfinite(b)
    if np.isfinite(bottom):
        first = np.flatnonzero(ok & (z >= bottom))
    else:
        first = np.flatnonzero(ok)
    if first.size == 0:
        return np.nan, np.nan, np.nan
    k0 = first[0]
    zz = z[k0:][ok[k0:]]
    bb = b[k0:][ok[k0:]]
    z0 = zz[0]
    if np.isfinite(top):
        zt = z0 + top
        above = np.flatnonzero(zz[1:] >= zt)
        if above.size == 0:
            return np.nan, np.nan, np.nan
        k = above[0] + 1
        bt = bb[k - 1] + (bb[k] - bb[k - 1]) * (zt - zz[k - 1]) / (zz[k] - zz[k - 1])
        seg = 0.5 * (-bb[:k][:-1] - bb[1:k]) * np.diff(zz[:k])
        integral = seg.sum() + 0.5 * (-bb[k - 1] - bt) * (zt - zz[k - 1])
        return np.sqrt(max(2.0 * integral, 0.0)), top, 0.0
    if bb[0] >= threshold:
        return 0.0, 0.0, 0.0
    warm = np.flatnonzero(bb[1:] >= threshold)
    if warm.size == 0:
        integral = (0.5 * (-bb[:-1] - bb[1:]) * np.diff(zz)).sum()
        return np.sqrt(max(2.0 * integral, 0.0)), zz[-1] - z0, 1.0
    k = warm[0] + 1
    zc = zz[k - 1] + (threshold - bb[k - 1]) * (zz[k] - zz[k - 1]) / (bb[k] - bb[k - 1])
    seg = 0.5 * (-bb[:k][:-1] - bb[1:k]) * np.diff(zz[:k])
    integral = seg.sum() + 0.5 * (-bb[k - 1] - threshold) * (zc - zz[k - 1])
    return np.sqrt(max(2.0 * integral, 0.0)), zc - z0, 0.0


def cold_pool(z, b, bottom, threshold, top, n_threads=0):
    """Cold-pool intensity, depth and open-top flag (rows) of every column."""
    z, b = np.asarray(z, np.float64), np.asarray(b, np.float64)
    bottom, top = np.asarray(bottom, np.float64), np.asarray(top, np.float64)
    out = np.full((3, b.shape[0]), np.nan)
    for c in range(b.shape[0]):
        out[:, c] = _cold_pool_column(z[c], b[c], bottom[c], threshold, top[c])
    return out


def _at(z, u, v, h):
    """Winds at height ``h`` by linear interpolation of valid levels."""
    if h < z[0] or h > z[-1]:
        return None
    return np.interp(h, z, u), np.interp(h, z, v)


def _profile_column(z, u, v, ground, bottom, top, cu, cv):
    out = np.full(7, np.nan)
    ok = np.isfinite(z) & np.isfinite(u) & np.isfinite(v)
    if not ok.any():
        return out
    z, u, v = z[ok], u[ok], v[ok]
    g = ground if np.isfinite(ground) else z[0]
    zb, zt = g + bottom, g + top
    lo, hi = _at(z, u, v, zb), _at(z, u, v, zt)
    if lo is None or hi is None:
        return out
    out[:4] = lo[0], lo[1], hi[0], hi[1]
    inner = (z > zb) & (z < zt)
    zs = np.concatenate([[zb], z[inner], [zt]])
    us = np.concatenate([[lo[0]], u[inner], [hi[0]]])
    vs = np.concatenate([[lo[1]], v[inner], [hi[1]]])
    if np.isfinite(cu) and np.isfinite(cv):
        ur, vr = us - cu, vs - cv
        out[4] = np.sum(ur[1:] * vr[:-1] - ur[:-1] * vr[1:])
    if zt > zb:
        dz = np.diff(zs)
        out[5] = np.sum(0.5 * (us[1:] + us[:-1]) * dz) / (zt - zb)
        out[6] = np.sum(0.5 * (vs[1:] + vs[:-1]) * dz) / (zt - zb)
    else:
        out[5], out[6] = lo
    return out


def profile(z, u, v, ground, bottom, top, cu, cv, n_threads=0):
    """Winds at ``bottom`` and ``top``, helicity and layer-mean wind (rows)."""
    z, u, v = (np.asarray(a, np.float64) for a in (z, u, v))
    ground, cu, cv = (np.asarray(a, np.float64) for a in (ground, cu, cv))
    out = np.full((7, u.shape[0]), np.nan)
    for c in range(u.shape[0]):
        out[:, c] = _profile_column(
            z[c], u[c], v[c], ground[c], bottom, top, cu[c], cv[c]
        )
    return out


def _deriv(f, x, axis):
    """Centred, one-sided or NaN derivative of ``f`` along ``axis``."""
    f = np.moveaxis(f, axis, -1)
    n = f.shape[-1]
    out = np.full(f.shape, np.nan)
    fin = np.isfinite(f)
    lo = np.zeros(f.shape, bool)
    hi = np.zeros(f.shape, bool)
    lo[..., 1:] = fin[..., :-1]
    hi[..., :-1] = fin[..., 1:]
    if n > 1:
        fm = np.empty(f.shape)
        fp = np.empty(f.shape)
        fm[..., 1:] = f[..., :-1]
        fm[..., 0] = np.nan
        fp[..., :-1] = f[..., 1:]
        fp[..., -1] = np.nan
        xm = np.concatenate([[np.nan], x[:-1]])
        xp = np.concatenate([x[1:], [np.nan]])
        with np.errstate(invalid="ignore"):
            centred = (fp - fm) / (xp - xm)
            forward = (fp - f) / (xp - x)
            backward = (f - fm) / (x - xm)
        both = fin & lo & hi
        out[both] = centred[both]
        only_hi = fin & hi & ~lo
        out[only_hi] = forward[only_hi]
        only_lo = fin & lo & ~hi
        out[only_lo] = backward[only_lo]
    return np.moveaxis(out, -1, axis)


def gradient(f, x, y, n_threads=0):
    """d/dx and d/dy (first axis) of ``f`` on (batch, y, x)."""
    f = np.asarray(f, np.float64)
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    return np.stack([_deriv(f, x, 2), _deriv(f, y, 1)])


def vad(vr, az, cos_el, min_gates, min_spread, n_threads=0):
    """VAD fit u, v, offset, rms residual and gate count (rows) per ring."""
    vr, az = np.asarray(vr, np.float64), np.asarray(az, np.float64)
    cos_el = np.asarray(cos_el, np.float64)
    ok = np.isfinite(vr) & np.isfinite(az)
    n = ok.sum(axis=1).astype(float)
    s = np.where(ok, np.sin(az), 0.0)
    c = np.where(ok, np.cos(az), 0.0)
    v = np.where(ok, vr, 0.0)
    out = np.full((5, vr.shape[0]), np.nan)
    out[4] = n
    with np.errstate(invalid="ignore", divide="ignore"):
        ms, mc, mv = s.sum(1) / n, c.sum(1) / n, v.sum(1) / n
        ds = np.where(ok, s - ms[:, None], 0.0)
        dc = np.where(ok, c - mc[:, None], 0.0)
        dv = np.where(ok, v - mv[:, None], 0.0)
        css, ccc, csc = (ds * ds).sum(1), (dc * dc).sum(1), (ds * dc).sum(1)
        csv, ccv = (ds * dv).sum(1), (dc * dv).sum(1)
        det = css * ccc - csc * csc
        good = (n >= max(min_gates, 3)) & (det / (n * n) >= min_spread)
        b1 = (ccc * csv - csc * ccv) / det
        b2 = (css * ccv - csc * csv) / det
        a0 = mv - b1 * ms - b2 * mc
        res = np.where(ok, vr - a0[:, None] - b1[:, None] * s - b2[:, None] * c, 0.0)
        rms = np.sqrt((res * res).sum(1) / n)
    out[0] = np.where(good, b1 / cos_el, np.nan)
    out[1] = np.where(good, b2 / cos_el, np.nan)
    out[2] = np.where(good, a0, np.nan)
    out[3] = np.where(good, rms, np.nan)
    return out
