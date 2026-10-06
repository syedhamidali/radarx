#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
NumPy reference implementation of the profile kernels.

Same functions, arguments and results as the compiled ``radarx.io._sounding``
module; used when the extension is not built and as the test oracle.
"""

import numpy as np

RD = 287.04749  # gas constant of dry air [J kg-1 K-1]
RV = 461.52311  # gas constant of water vapour [J kg-1 K-1]
EPS = RD / RV
CPD = 1005.7  # specific heat of dry air [J kg-1 K-1]
CPV = 1875.0  # specific heat of water vapour [J kg-1 K-1]
T0 = 273.15
G0 = 9.80665  # standard gravity [m s-2]
RE = 6371008.8  # mean Earth radius [m]


def _esat(t):
    tc = t - T0
    return 611.2 * np.exp(17.67 * tc / (tc + 243.5))


def _dewpoint(e):
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(e > 0, e, np.nan)
        lg = np.log(e / 611.2)
        return T0 + 243.5 * lg / (17.67 - lg)


def _latent_heat(t):
    return (2.501 - 0.00237 * (t - T0)) * 1e6


def _mixing_ratio(e, p):
    return EPS * e / (p - e)


def _wet_bulb(p, t, td):
    td = np.minimum(td, t)
    r = _mixing_ratio(_esat(td), p)
    cp = CPD + r * CPV

    def f(tw):
        return cp * (t - tw) - _latent_heat(tw) * (_mixing_ratio(_esat(tw), p) - r)

    lo, hi = td.copy(), t.copy()
    tw = 0.5 * (lo + hi)
    h = 1e-4
    for _ in range(60):
        fv = f(tw)
        pos = fv > 0
        lo = np.where(pos, tw, lo)
        hi = np.where(pos, hi, tw)
        df = (f(tw + h) - f(tw - h)) / (2 * h)
        with np.errstate(divide="ignore", invalid="ignore"):
            nxt = tw - fv / df
        bad = ~((nxt > lo) & (nxt < hi))
        nxt = np.where(bad, 0.5 * (lo + hi), nxt)
        done = (hi - lo < 1e-7) | (np.abs(nxt - tw) < 1e-7)
        tw = np.where(hi - lo < 1e-7, tw, nxt)
        if np.all(done | np.isnan(tw)):
            break
    return np.where(np.isnan(p) | np.isnan(t) | np.isnan(td), np.nan, tw)


def thermo(op, inputs, n_threads=0):
    """Elementwise thermodynamics (see the compiled kernel for ``op``)."""
    a = [np.asarray(x, dtype=np.float64).ravel() for x in inputs]
    if op == "esat":
        return _esat(a[0])
    if op == "dewpoint":
        return _dewpoint(a[0])
    if op == "vapor_pressure":
        q, p = a
        return q * p / (EPS + (1.0 - EPS) * q)
    if op == "specific_humidity":
        e, p = a
        return EPS * e / (p - (1.0 - EPS) * e)
    if op == "wet_bulb":
        return _wet_bulb(*a)
    if op == "height":
        h = a[0] / G0
        return RE * h / (RE - h)
    if op == "density":
        p, t, q = a
        return p / (RD * t * (1.0 + (1.0 / EPS - 1.0) * q))
    raise ValueError(f"unknown op {op}")


def interp_vertical(z, values, target, log_var, extrapolate=False, n_threads=0):
    """Interpolate (nvar, nprof, nlev) profiles to (nrow, nt) target heights."""
    z = np.asarray(z, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    nvar, nprof, nlev = values.shape
    nrow, nt = target.shape
    out = np.full((nvar, nrow, nt), np.nan)
    for row in range(nrow):
        prof = 0 if nprof == 1 else row
        zc = z[prof]
        zt = target[row]
        for k in range(nvar):
            vc = values[k, prof]
            ok = np.isfinite(vc) & np.isfinite(zc)
            if not ok.any():
                continue
            zv, vv = zc[ok], vc[ok]
            if log_var[k]:
                with np.errstate(divide="ignore", invalid="ignore"):
                    res = np.exp(np.interp(zt, zv, np.log(vv)))
            else:
                res = np.interp(zt, zv, vv)
            if not extrapolate:
                res = np.where((zt < zv[0]) | (zt > zv[-1]), np.nan, res)
            res = np.where(np.isnan(zt), np.nan, res)
            out[k, row] = res
    return out


def level_crossing(z, values, level, highest=True, n_threads=0):
    """Height where values fall through ``level`` going up, per column."""
    z = np.asarray(z, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    out = np.full(z.shape[0], np.nan)
    for c in range(z.shape[0]):
        ok = np.isfinite(values[c]) & np.isfinite(z[c])
        zc, vc = z[c, ok], values[c, ok]
        hits = np.nonzero((vc[:-1] >= level) & (vc[1:] < level))[0]
        if hits.size:
            i = hits[-1] if highest else hits[0]
            w = (vc[i] - level) / (vc[i] - vc[i + 1])
            out[c] = zc[i] + w * (zc[i + 1] - zc[i])
    return out


def layer_mean(z, values, bottom, top, n_threads=0):
    """Height-weighted mean over [bottom, top] of each profile."""
    if not top > bottom:
        raise ValueError("top must be above bottom")
    z = np.asarray(z, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    nvar, ncol, _ = values.shape
    out = np.full((nvar, ncol), np.nan)
    for k in range(nvar):
        for c in range(ncol):
            ok = np.isfinite(values[k, c]) & np.isfinite(z[c])
            zc, vc = z[c, ok], values[k, c, ok]
            if zc.size < 2 or zc[0] > bottom or zc[-1] < top:
                continue
            inner = zc[(zc > bottom) & (zc < top)]
            zz = np.concatenate([[bottom], inner, [top]])
            vv = np.interp(zz, zc, vc)
            out[k, c] = np.sum(0.5 * (vv[1:] + vv[:-1]) * np.diff(zz)) / (top - bottom)
    return out


def bilinear_columns(fields, time_weights, lat, lon, qlat, qlon, n_threads=0):
    """Columns at (qlat, qlon) from (ntime, nvar, nlev, nlat, nlon) fields."""
    fields = np.asarray(fields, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    qlat = np.asarray(qlat, dtype=np.float64).ravel()
    qlon = np.asarray(qlon, dtype=np.float64).ravel()
    inside = (qlat >= lat[0]) & (qlat <= lat[-1]) & (qlon >= lon[0]) & (qlon <= lon[-1])
    j = np.clip(np.searchsorted(lat, qlat, side="right") - 1, 0, lat.size - 2)
    i = np.clip(np.searchsorted(lon, qlon, side="right") - 1, 0, lon.size - 2)
    fy = (qlat - lat[j]) / (lat[j + 1] - lat[j])
    fx = (qlon - lon[i]) / (lon[i + 1] - lon[i])
    out = 0.0
    for t, wt in enumerate(np.asarray(time_weights, dtype=np.float64)):
        if wt == 0.0:
            continue
        f = fields[t]  # (nvar, nlev, nlat, nlon)
        out = out + wt * (
            ((1 - fy) * (1 - fx)) * f[..., j, i]
            + ((1 - fy) * fx) * f[..., j, i + 1]
            + (fy * (1 - fx)) * f[..., j + 1, i]
            + (fy * fx) * f[..., j + 1, i + 1]
        )
    out = np.where(inside, out, np.nan)  # (nvar, nlev, npts)
    return np.ascontiguousarray(np.moveaxis(out, -1, 1))


def rotate_wind(u, v, angle, n_threads=0):
    """Rotate earth-relative winds to grid axes (see the compiled kernel)."""
    u, v, angle = (np.asarray(a, dtype=np.float64).ravel() for a in (u, v, angle))
    s, c = np.sin(np.radians(angle)), np.cos(np.radians(angle))
    return np.stack([u * c + v * s, -u * s + v * c])
