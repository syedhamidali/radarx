#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
NumPy reference of the raindrop trajectory kernel.

Every operation follows ``radarx/retrieve/_rain_trajectories.cpp`` in the same
order, vectorised over the drops. It is the fallback without the compiled
kernel and the oracle of the tests.
"""

from __future__ import annotations

import math

import numpy as np

RD = 287.04
RV = 461.5
EPS = RD / RV
RHO_W = 1000.0
T0 = 273.15
D_FLOOR = 1.0e-3
BISECT = 60
NEWTON = 3
NOUT = 9
ALOFT, LANDED, EVAPORATED, INVALID = 0, 1, 2, 3

_G = 0x9E3779B97F4A7C15
_C = 0xD1B54A32D192ED03
_MASK = (1 << 64) - 1


def _u64(x):
    return np.uint64(x & _MASK)


def _mix64(z):
    with np.errstate(over="ignore"):
        z = z + _u64(_G)
        z = (z ^ (z >> np.uint64(30))) * _u64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * _u64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def _normal(key, ctr):
    with np.errstate(over="ignore"):
        h1 = _mix64(key + _u64((2 * ctr) * _C))
        h2 = _mix64(key + _u64((2 * ctr + 1) * _C))
    scale = 1.0 / 9007199254740992.0
    u1 = ((h1 >> np.uint64(11)).astype(np.float64) + 0.5) * scale
    u2 = ((h2 >> np.uint64(11)).astype(np.float64) + 0.5) * scale
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * math.pi * u2)


def _locate(ax, q):
    """Index, weight and whether ``q`` is within the axis."""
    n = ax.size
    if n <= 1:
        zero = np.zeros(np.shape(q))
        return np.zeros(np.shape(q), dtype=np.int64), zero, np.ones(np.shape(q), bool)
    i = np.clip(np.searchsorted(ax, q, side="right") - 1, 0, n - 2)
    w = np.clip((q - ax[i]) / (ax[i + 1] - ax[i]), 0.0, 1.0)
    return i, w, (q >= ax[0]) & (q <= ax[-1])


def _background(m, z):
    ax = m["bz"]
    i, w, inside = _locate(ax, z)
    j = np.minimum(i + 1, ax.size - 1)
    out = [a[i] + w * (a[j] - a[i]) for a in (m["bu"], m["bv"], m["bw"])]
    dwdz = np.zeros(np.shape(z))
    if m["wdiv"] and ax.size > 1:
        with np.errstate(all="ignore"):
            slope = (m["bw"][j] - m["bw"][i]) / (ax[j] - ax[i])
        dwdz = np.where(inside, slope, 0.0)
    return out[0], out[1], out[2], np.zeros(np.shape(z)), np.zeros(np.shape(z)), dwdz


def _wind(m, t, x, y, z):
    """(u, v, w, du/dx, dv/dy, dw/dz) at points (see the kernel)."""
    bg = _background(m, z)
    if m["nt"] == 0:
        return bg
    n = np.shape(t)
    wt, wz, wy, wx = m["wt"], m["wz"], m["wy"], m["wx"]
    nz, ny, nx = wz.size, wy.size, wx.size
    it, wtt, _ = _locate(wt, t)
    iz, wzz, in_z = _locate(wz, z)
    ok = np.ones(n, dtype=bool)
    if nz > 1:
        ok &= z <= wz[-1]
    nslice_two = wtt != 0.0
    flat_uvw = m["uvw"].reshape(-1, 3)
    acc = [np.zeros(n) for _ in range(3)]
    der = [np.zeros(n) for _ in range(3)]  # du/dx, dv/dy, dw/dz
    invz = np.zeros(n)
    if nz > 1:
        with np.errstate(all="ignore"):
            invz = np.where(in_z, 1.0 / (wz[np.minimum(iz + 1, nz - 1)] - wz[iz]), 0.0)
    for s in (0, 1):
        used = np.ones(n, dtype=bool) if s == 0 else nslice_two
        sl = np.minimum(it + s, wt.size - 1)
        sw = (1.0 - wtt) if s == 0 else wtt
        dt = t - wt[sl]
        iy, wyy, in_y = _locate(wy, y - m["cy"] * dt)
        ix, wxx, in_x = _locate(wx, x - m["cx"] * dt)
        if nx > 1:
            ok &= in_x | ~used
        if ny > 1:
            ok &= in_y | ~used
        invy = np.zeros(n)
        invx = np.zeros(n)
        with np.errstate(all="ignore"):
            if ny > 1:
                invy = np.where(
                    in_y, 1.0 / (wy[np.minimum(iy + 1, ny - 1)] - wy[iy]), 0.0
                )
            if nx > 1:
                invx = np.where(
                    in_x, 1.0 / (wx[np.minimum(ix + 1, nx - 1)] - wx[ix]), 0.0
                )
        for corner in range(8):
            hz, hy, hx = (corner >> 2) & 1, (corner >> 1) & 1, corner & 1
            wzc = wzz if hz else 1.0 - wzz
            wyc = wyy if hy else 1.0 - wyy
            wxc = wxx if hx else 1.0 - wxx
            weight = (wzc * wyc * wxc) * sw
            flat = (
                (sl * nz + np.minimum(iz + hz, nz - 1)) * (ny * nx)
                + np.minimum(iy + hy, ny - 1) * nx
                + np.minimum(ix + hx, nx - 1)
            )
            q = flat_uvw[flat]
            needed = used & ((weight != 0.0) | bool(m["wdiv"]))
            ok &= np.isfinite(q).all(axis=-1) | ~needed
            qs = np.where(np.isfinite(q) & needed[:, None], q, 0.0)
            for k in range(3):
                acc[k] = acc[k] + weight * qs[:, k]
            if m["wdiv"]:
                dz = wyc * wxc * ((1.0 if hz else -1.0) * invz)
                dy = wzc * wxc * ((1.0 if hy else -1.0) * invy)
                dx = wzc * wyc * ((1.0 if hx else -1.0) * invx)
                der[2] = der[2] + (dz * sw) * qs[:, 2]
                der[1] = der[1] + (dy * sw) * qs[:, 1]
                der[0] = der[0] + (dx * sw) * qs[:, 0]
    grid = (acc[0], acc[1], acc[2], der[0], der[1], der[2])
    return tuple(np.where(ok, g, b) for g, b in zip(grid, bg))


def _esat(t):
    """Buck (1981), Eq. 8, saturation vapour pressure over water [Pa], ``t`` in K

    The enhancement factor of moist air is neglected (radarx choice).
    """
    tc = t - T0
    return 611.21 * np.exp(17.502 * tc / (240.97 + tc))


def thermo_nodes(temp, p, rh):
    """Air density and evaporation properties at the levels of a profile."""
    es = _esat(temp)
    e = rh * es
    qv = EPS * e / (p - (1.0 - EPS) * e)
    rho = p / (RD * temp * (1.0 + (1.0 / EPS - 1.0) * qv))
    lv = 2.499e6 * np.power(T0 / temp, 0.167 + 3.67e-4 * temp)
    k = (0.441635 + 0.0071 * temp) * 1.0e-2
    dv = 2.11e-5 * np.power(temp / T0, 1.94) * (1.0e5 / p)
    nu = (0.379565 + 0.0049 * temp) * 1.0e-5 / rho
    fkd = (lv / (RV * temp) - 1.0) * lv / (k * temp) + RV * temp / (dv * es)
    return rho, nu, fkd, rh - 1.0, np.cbrt(nu / dv)


def _environment(m, z):
    """Density, its slope and the evaporation properties at heights ``z``."""
    ax = m["ez"]
    i, w, inside = _locate(ax, z)
    j = np.minimum(i + 1, ax.size - 1)

    def lin(a):
        return a[i] + w * (a[j] - a[i])

    rho = lin(m["erho"])
    drho = np.zeros(np.shape(z))
    if ax.size > 1:
        with np.errstate(all="ignore"):
            slope = (m["erho"][j] - m["erho"][i]) / (ax[j] - ax[i])
        drho = np.where(inside, slope, 0.0)
    if m["evap"]:
        return rho, drho, lin(m["enu"]), lin(m["efkd"]), lin(m["essat"]), lin(m["ecs"])
    zero = np.zeros(np.shape(z))
    return rho, drho, zero, zero, zero, zero


def _v0(m, d):
    """Sea-level fall speed (at least zero) and its derivative with D."""
    kind = m["fall_kind"]
    fc = m["fc"]
    with np.errstate(all="ignore"):
        if kind == 0:
            e = np.exp(-0.6 * d)
            v = 9.65 - 10.3 * e
            dv = 6.18 * e
        elif kind == 1:
            v = fc[0] * np.power(d, fc[1]) * np.exp(-fc[2] * d)
            dv = v * (fc[1] / d - fc[2])
        else:
            v = np.zeros(np.shape(d))
            dv = np.zeros(np.shape(d))
            for c in fc[::-1]:
                dv = dv * d + v
                v = v * d + c
    pos = v > 0.0
    return np.where(pos, v, 0.0), np.where(pos, dv, 0.0)


def _rhs(m, t, y, turb):
    """``dir * dy/dt`` for the states ``y`` (5, n)."""
    x, yy, z, s, _ = y
    u, v, w, dudx, dvdy, dwdz = _wind(m, t, x, yy, z)
    rho, drho, nu, fkd, ssat, cs = _environment(m, z)
    d = np.sqrt(np.maximum(s, D_FLOOR * D_FLOOR))
    v0, dv0 = _v0(m, d)
    corr = np.power(m["rho0"] / rho, 0.4) if m["dens"] else 1.0
    vt = v0 * corr
    f = np.zeros_like(y)
    f[0] = u + turb[0]
    f[1] = v + turb[1]
    f[2] = w + turb[2] - vt
    dm = d * 1.0e-3
    if m["evap"]:
        arg = vt * dm / nu
        fv = m["vav"] + m["vbv"] * cs * np.sqrt(arg)
        kc = 8.0e6 * ssat / (fkd * RHO_W)
        sdot = kc * fv
        f[3] = sdot
    if m["ratio"]:
        div = np.zeros_like(z)
        if m["wdiv"]:
            div = dudx + dvdy + dwdz
        if m["dens"]:
            div = div + 0.4 * vt * drho / rho
        if m["evap"]:
            with np.errstate(all="ignore"):
                dfv = np.where(
                    arg > 0.0,
                    m["vbv"]
                    * cs
                    * (0.5 / np.sqrt(arg))
                    * (dv0 * corr * dm + vt * 1.0e-3)
                    / nu,
                    0.0,
                )
            div = div + kc * dfv / (2.0 * d) - sdot / (2.0 * d * d)
        f[4] = -div
    return m["dir"] * f


def _step(m, t, y, k1, turb, h):
    dr = m["dir"]
    if m["scheme"] == 2:
        k2 = _rhs(m, t + dr * h, y + h * k1, turb)
        return y + 0.5 * h * (k1 + k2)
    k2 = _rhs(m, t + dr * 0.5 * h, y + 0.5 * h * k1, turb)
    k3 = _rhs(m, t + dr * 0.5 * h, y + 0.5 * h * k2, turb)
    k4 = _rhs(m, t + dr * h, y + h * k3, turb)
    return y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _hermite(th, a0, fa0, a1, fa1):
    t2 = th * th
    t3 = t2 * th
    return (
        (2.0 * t3 - 3.0 * t2 + 1.0) * a0
        + (t3 - 2.0 * t2 + th) * fa0
        + (-2.0 * t3 + 3.0 * t2) * a1
        + (t3 - t2) * fa1
    )


def _stop(m, x, y, zconst):
    """Stop height and horizontal gradient at points (see the kernel)."""
    sy, sx, sz = m["sy"], m["sx"], m["sz"]
    zero = np.zeros(np.shape(x))
    if sy.size == 0:
        return zconst, zero, zero
    iy, wy, in_y = _locate(sy, y)
    ix, wx, in_x = _locate(sx, x)
    jy = np.minimum(iy + 1, sy.size - 1)
    jx = np.minimum(ix + 1, sx.size - 1)
    z00, z01 = sz[iy, ix], sz[iy, jx]
    z10, z11 = sz[jy, ix], sz[jy, jx]
    z = (1.0 - wy) * ((1.0 - wx) * z00 + wx * z01) + wy * ((1.0 - wx) * z10 + wx * z11)
    gx, gy = zero, zero
    with np.errstate(all="ignore"):
        if sx.size > 1:
            gx = np.where(
                in_x,
                ((1.0 - wy) * (z01 - z00) + wy * (z11 - z10)) / (sx[ix + 1] - sx[ix]),
                0.0,
            )
        if sy.size > 1:
            gy = np.where(
                in_y,
                ((1.0 - wx) * (z10 - z00) + wx * (z11 - z01)) / (sy[iy + 1] - sy[iy]),
                0.0,
            )
    return z, gx, gy


def _hermite_root(scale, lev0, lev1, a0, fa0, a1, fa1):
    lo = np.zeros_like(a0)
    hi = np.ones_like(a0)
    for _ in range(BISECT):
        mid = 0.5 * (lo + hi)
        g = scale * (
            _hermite(mid, a0, fa0, a1, fa1) - ((1.0 - mid) * lev0 + mid * lev1)
        )
        up = g > 0.0
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    return 0.5 * (lo + hi)


def integrate(m, x0, y0, z0, t0, d0, zstop, seed, nrec):
    """Integrate all drops; returns ``out (n, 9)`` and ``path (n, nrec, 5)``."""
    n = x0.size
    out = np.full((n, NOUT), np.nan)
    out[:, 6] = INVALID
    path = np.full((n, nrec, 5), np.nan)
    valid = (
        np.isfinite(x0)
        & np.isfinite(y0)
        & np.isfinite(z0)
        & np.isfinite(t0)
        & np.isfinite(d0)
        & np.isfinite(zstop)
        & (d0 > 0.0)
    )
    with np.errstate(invalid="ignore"):
        valid &= np.isfinite(_stop(m, x0, y0, zstop)[0])
    idx = np.nonzero(valid)[0]
    if idx.size == 0:
        return out, path
    dr = m["dir"]
    h = m["dt"]
    nsteps = int(math.ceil(m["max_time"] / h))
    y = np.stack([x0[idx], y0[idx], z0[idx], d0[idx] ** 2, np.zeros(idx.size)])
    tz = t0[idx]
    zs = zstop[idx]
    spz = _stop(m, y[0], y[1], zs)[0]
    turb = np.zeros((3, idx.size))
    key = None
    if m["turb"]:
        with np.errstate(over="ignore"):
            key = _mix64(
                _u64(seed) + (idx.astype(np.uint64) + np.uint64(1)) * _u64(_G),
            )
        ar = math.exp(-h / m["timescale"])
        ar_s = math.sqrt(1.0 - ar * ar)
        sig = (m["sigma_h"], m["sigma_h"], m["sigma_w"])
        for k in range(3):
            turb[k] = sig[k] * _normal(key, k)
    f0 = _rhs(m, tz, y, turb)
    vz0 = -dr * f0[2]
    rec = np.zeros(idx.size, dtype=np.int64)

    def record(sel, elapsed, ys):
        if nrec == 0:
            return
        sel = sel[rec[sel] < nrec]
        r = rec[sel]
        path[idx[sel], r, 0] = elapsed[sel] if np.ndim(elapsed) else elapsed
        path[idx[sel], r, 1] = ys[0][sel]
        path[idx[sel], r, 2] = ys[1][sel]
        path[idx[sel], r, 3] = ys[2][sel]
        path[idx[sel], r, 4] = np.sqrt(np.maximum(ys[3][sel], 0.0))
        rec[sel] += 1

    def finish(sel, elapsed, ys, status, fend):
        gi = idx[sel]
        out[gi, 0] = ys[0]
        out[gi, 1] = ys[1]
        out[gi, 2] = ys[2]
        out[gi, 3] = elapsed
        out[gi, 4] = np.sqrt(np.maximum(ys[3], 0.0))
        out[gi, 5] = ys[4]
        out[gi, 6] = status
        out[gi, 7] = vz0[sel]
        out[gi, 8] = -dr * fend[2]

    everyone = np.arange(idx.size)
    zero = np.zeros(idx.size)
    record(everyone, zero, y)
    active = np.ones(idx.size, dtype=bool)
    # already at the stop level or already evaporated
    at_stop = dr * (y[2] - spz) <= 0.0
    sel = np.nonzero(at_stop)[0]
    if sel.size:
        finish(sel, 0.0, y[:, sel], LANDED, f0[:, sel])
        record(sel, 0.0, y)
        active[sel] = False
    gone = active & (y[3] <= m["smin"])
    sel = np.nonzero(gone)[0]
    if sel.size:
        finish(sel, 0.0, y[:, sel], EVAPORATED, f0[:, sel])
        record(sel, 0.0, y)
        active[sel] = False
    fk = f0.copy()
    stride = int(m["stride"])
    for step in range(nsteps):
        act = np.nonzero(active)[0]
        if act.size == 0:
            break
        t = tz[act] + dr * (step * h)
        ya = y[:, act]
        ta = turb[:, act]
        fa = fk[:, act]
        y1 = _step(m, t, ya, fa, ta, h)
        sp1 = _stop(m, y1[0], y1[1], zs[act])[0]
        hit_z = dr * (y1[2] - sp1) <= 0.0
        hit_s = y1[3] <= m["smin"]
        ev = hit_z | hit_s
        if ev.any():
            e = np.nonzero(ev)[0]
            ye, y1e, fe0 = ya[:, e], y1[:, e], fa[:, e]
            f1 = _rhs(m, t[e] + dr * h, y1e, ta[:, e])
            th = np.full(e.size, 2.0)
            status = np.full(e.size, LANDED)
            hz = hit_z[e]
            if hz.any():
                th[hz] = _hermite_root(
                    dr,
                    spz[act][e][hz],
                    sp1[e][hz],
                    ye[2][hz],
                    h * fe0[2][hz],
                    y1e[2][hz],
                    h * f1[2][hz],
                )
            hs = hit_s[e]
            if hs.any():
                ts = _hermite_root(
                    1.0,
                    m["smin"],
                    m["smin"],
                    ye[3][hs],
                    h * fe0[3][hs],
                    y1e[3][hs],
                    h * f1[3][hs],
                )
                better = ts < th[hs]
                sub = np.nonzero(hs)[0]
                th[sub[better]] = ts[better]
                status[sub[better]] = EVAPORATED
            end = np.empty_like(ye)
            for k in range(5):
                end[k] = _hermite(th, ye[k], h * fe0[k], y1e[k], h * f1[k])
            end[3] = np.where(status == LANDED, end[3], m["smin"])
            land = np.nonzero(status == LANDED)[0]
            if land.size:
                # Newton iterations with real, shortened steps (see the kernel)
                tl, yl, fl = t[e][land], ye[:, land], fe0[:, land]
                ul, zl = ta[:, e][:, land], zs[act][e][land]
                thl = th[land]
                live = np.ones(land.size, dtype=bool)
                for _ in range(NEWTON):
                    el = _step(m, tl, yl, fl, ul, thl * h)
                    fel = _rhs(m, tl + dr * (thl * h), el, ul)
                    se, sgx, sgy = _stop(m, el[0], el[1], zl)
                    g = dr * (el[2] - se)
                    gp = dr * h * (fel[2] - sgx * fel[0] - sgy * fel[1])
                    live &= (gp != 0.0) & np.isfinite(gp)
                    with np.errstate(all="ignore"):
                        nxt = np.clip(thl - g / gp, 0.0, 1.0)
                    thl = np.where(live, nxt, thl)
                el = _step(m, tl, yl, fl, ul, thl * h)
                el[2] = _stop(m, el[0], el[1], zl)[0]
                end[:, land] = el
                th[land] = thl
            fend = _rhs(m, t[e] + dr * (th * h), end, ta[:, e])
            sel = act[e]
            elapsed = (step + th) * h
            gi = idx[sel]
            out[gi, 0] = end[0]
            out[gi, 1] = end[1]
            out[gi, 2] = end[2]
            out[gi, 3] = elapsed
            out[gi, 4] = np.sqrt(np.maximum(end[3], 0.0))
            out[gi, 5] = end[4]
            out[gi, 6] = status
            out[gi, 7] = vz0[sel]
            out[gi, 8] = -dr * fend[2]
            if nrec:
                r = rec[sel]
                keep = r < nrec
                path[gi[keep], r[keep], 0] = elapsed[keep]
                for k in range(3):
                    path[gi[keep], r[keep], 1 + k] = end[k][keep]
                path[gi[keep], r[keep], 4] = np.sqrt(np.maximum(end[3][keep], 0.0))
                rec[sel[keep]] += 1
            active[sel] = False
        cont = np.nonzero(~ev)[0]
        if cont.size == 0:
            continue
        sel = act[cont]
        y[:, sel] = y1[:, cont]
        spz[sel] = sp1[cont]
        if m["turb"]:
            mstep = step + 1
            for k in range(3):
                turb[k, sel] = ar * turb[k, sel] + ar_s * sig[k] * _normal(
                    key[sel], 3 * mstep + k
                )
        fk[:, sel] = _rhs(m, tz[sel] + dr * ((step + 1) * h), y[:, sel], turb[:, sel])
        if stride > 0 and ((step + 1) % stride) == 0:
            record(sel, (step + 1) * h, y)
    sel = np.nonzero(active)[0]
    if sel.size:
        finish(sel, nsteps * h, y[:, sel], ALOFT, fk[:, sel])
        record(sel, nsteps * h, y)
    return out, path
