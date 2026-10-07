#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
NumPy reference of the compiled trajectory and DLA kernel.

Same steps, in the same order, as ``radarx/retrieve/_lagrangian.cpp``,
vectorised over trajectories. Used when the kernel is not built and as the
test oracle of the kernel.
"""

import math

import numpy as np

# Trajectory flags
ENV_DBZ = 1
ENV_W = 2
BOUNDARY = 4
END_OF_DATA = 8
MAX_STEPS = 16
MISSING = 32
NW = 6  # wind pack: u, v, w, Z_H, valid, environment mask
ENVIRONMENT = ENV_DBZ | ENV_W | BOUNDARY

# Process switches
COND = 1
REVP = 2
RACW = 4
GACW = 8
GMLT = 16
GSUB = 32
GFR = 64
DAMP = 128
FLUX = 256
MICRO = REVP | RACW | GACW | GMLT | GSUB | GFR

T0 = 273.15
P0 = 1.0e5
RD = 287.04
KAPPA = 0.2854
CP = RD / KAPPA
RV = 461.5
EPS = RD / RV
ES0 = 611.2

LV_L = 2.5e6
LF_L = 3.336e5
LS_L = 2.8336e6
CW = 4.187e3
A_R = 841.99
B_R = 0.8
C_D = 0.6
G = 9.805
A_PR = 0.66
B_PR = 100.0
RHO_W = 1000.0

# thermodynamic parameter indices (same order as the kernel)
THERMO_KEYS = (
    "dt_small",
    "cd",
    "b",
    "w0",
    "ld1",
    "ld2",
    "lw1",
    "lw2",
    "cmin",
    "cmax",
    "qp0",
    "q0",
    "q1",
    "z_bl",
    "b_f",
    "z_sfc",
    "rho_g_sfc",
    "rho_g_5km",
    "rho0",
    "flux_theta",
    "flux_qv",
    "switches",
)
Q = {k: i for i, k in enumerate(THERMO_KEYS)}
N_BUDGET = 7


class Grid:
    """Gridded fields (nt, nz, ny, nx, nvar) on a grid moving with (cx, cy)."""

    def __init__(self, x, y, z, t, data, cx=0.0, cy=0.0, ext_before=0.0, ext_after=0.0):
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        self.z = np.asarray(z, float)
        self.t = np.asarray(t, float)
        self.data = np.asarray(data, np.float32)
        self.nvar = self.data.shape[-1]
        self.flat = self.data.reshape(-1, self.nvar)
        self.cx, self.cy = float(cx), float(cy)
        self.eb, self.ea = float(ext_before), float(ext_after)


def _bracket(c, v):
    i = np.searchsorted(c, v, side="right") - 1
    return np.clip(i, 0, c.size - 2)


def sample(g, xp, yp, zp, tp, check_nan=True):
    """Values (n, nvar) and status (n,) at the points, as the kernel's sample."""
    xp, yp, zp, tp = (np.asarray(a, float) for a in (xp, yp, zp, tp))
    n = xp.size
    status = np.zeros(n, np.int32)
    tol = 1.0e-6
    lev0 = np.zeros(n, np.int64)
    lev1 = np.zeros(n, np.int64)
    wt0 = np.ones(n)
    wt1 = np.zeros(n)
    before = tp < g.t[0]
    after = tp > g.t[-1]
    status[before & (tp < g.t[0] - g.eb - tol)] = END_OF_DATA
    status[after & (tp > g.t[-1] + g.ea + tol)] = END_OF_DATA
    lev0[after] = g.t.size - 1
    inside = ~before & ~after
    if g.t.size > 1:
        i = _bracket(g.t, tp)
        a = (tp - g.t[i]) / (g.t[i + 1] - g.t[i])
        lev0 = np.where(inside, i, lev0)
        lev1 = np.where(inside, i + 1, lev1)
        wt0 = np.where(inside, 1.0 - a, wt0)
        wt1 = np.where(inside, a, wt1)
    out = np.zeros((n, g.nvar))
    k = _bracket(g.z, zp)
    fz = (zp - g.z[k]) / (g.z[k + 1] - g.z[k])
    nx, ny, nz = g.x.size, g.y.size, g.z.size
    for lev, wt in ((lev0, wt0), (lev1, wt1)):
        use = (wt > 0.0) & (status == 0)
        dtl = tp - g.t[lev]
        xs = xp - g.cx * dtl
        ys = yp - g.cy * dtl
        bad = use & ((xs < g.x[0]) | (xs > g.x[-1]) | (ys < g.y[0]) | (ys > g.y[-1]))
        status[bad] = BOUNDARY
        use &= ~bad
        i = _bracket(g.x, xs)
        j = _bracket(g.y, ys)
        fx = (xs - g.x[i]) / (g.x[i + 1] - g.x[i])
        fy = (ys - g.y[j]) / (g.y[j + 1] - g.y[j])
        base = ((lev * nz + k) * ny + j) * nx + i
        sx, sy, sz = 1, nx, ny * nx

        def node(off):
            return g.flat[np.where(use, base + off, 0)].astype(float)

        w000 = (1 - fz) * (1 - fy) * (1 - fx)
        w001 = (1 - fz) * (1 - fy) * fx
        w010 = (1 - fz) * fy * (1 - fx)
        w011 = (1 - fz) * fy * fx
        w100 = fz * (1 - fy) * (1 - fx)
        w101 = fz * (1 - fy) * fx
        w110 = fz * fy * (1 - fx)
        w111 = fz * fy * fx
        c = (
            w000[:, None] * node(0)
            + w001[:, None] * node(sx)
            + w010[:, None] * node(sy)
            + w011[:, None] * node(sy + sx)
            + w100[:, None] * node(sz)
            + w101[:, None] * node(sz + sx)
            + w110[:, None] * node(sz + sy)
            + w111[:, None] * node(sz + sy + sx)
        )
        out = np.where(use[:, None], out + wt[:, None] * c, out)
    if check_nan:
        miss = (status == 0) & np.isnan(out[:, :3]).any(axis=1)
        status[miss] = MISSING
    return out, status


def build_paths(g, par, starts):
    """Paths of all start points: pos (n, m, 4), val (n, m, 6), npts, flags."""
    dt, n_iter, max_steps = par[0], int(par[1]), int(par[2])
    h = (-1.0 if par[3] < 0 else 1.0) * dt
    mode, min_steps = int(par[4]), int(par[5])
    env_dbz, env_w, env_w_steps = par[6], par[7], int(par[8])
    env_dbz_steps, cold_pool_depth, z_sfc = int(par[9]), par[10], par[11]
    starts = np.asarray(starts, float).reshape(-1, 3)
    n, m = starts.shape[0], max_steps + 1
    pos = np.full((n, m, 4), np.nan)
    val = np.full((n, m, NW), np.nan)
    npts = np.zeros(n, np.int64)
    flags = np.zeros(n, np.int32)
    zlo, zhi = g.z[0], g.z[-1]

    def outside(xa, ya):
        return (xa < g.x[0]) | (xa > g.x[-1]) | (ya < g.y[0]) | (ya > g.y[-1])

    xn, yn = starts[:, 0].copy(), starts[:, 1].copy()
    zn = np.clip(starts[:, 2], zlo, zhi)
    tn = np.zeros(n)
    out0 = outside(xn, yn)
    flags[out0] = BOUNDARY
    vn = np.zeros((n, NW))
    ok = ~out0
    v, st = sample(g, xn[ok], yn[ok], zn[ok], tn[ok])
    idx = np.flatnonzero(ok)
    flags[idx] = st
    good = idx[st == 0]
    vn[idx] = v
    pos[good, 0] = np.column_stack([xn[good], yn[good], zn[good], tn[good]])
    val[good, 0] = vn[good]
    npts[good] = 1
    active = np.zeros(n, bool)
    active[good] = True
    wcount = np.zeros(n, np.int64)
    dcount = np.zeros(n, np.int64)
    for step in range(1, max_steps + 1):
        a = np.flatnonzero(active)
        if a.size == 0:
            break
        t1 = tn[a] + h
        xs = xn[a] + h * vn[a, 0]
        ys = yn[a] + h * vn[a, 1]
        zs = np.clip(zn[a] + h * vn[a, 2], zlo, zhi)
        ok = np.ones(a.size, bool)
        bad = outside(xs, ys)
        flags[a[bad]] |= BOUNDARY
        ok &= ~bad
        v1 = np.zeros((a.size, NW))
        for _ in range(n_iter):
            r = np.flatnonzero(ok)
            vv, st = sample(g, xs[r], ys[r], zs[r], t1[r])
            flags[a[r[st != 0]]] |= st[st != 0]
            ok[r[st != 0]] = False
            r = r[st == 0]
            v1[r] = vv[st == 0]
            xs[r] = xn[a[r]] + 0.5 * h * (vn[a[r], 0] + v1[r, 0])
            ys[r] = yn[a[r]] + 0.5 * h * (vn[a[r], 1] + v1[r, 1])
            zs[r] = np.clip(zn[a[r]] + 0.5 * h * (vn[a[r], 2] + v1[r, 2]), zlo, zhi)
            bad = outside(xs[r], ys[r])
            flags[a[r[bad]]] |= BOUNDARY
            ok[r[bad]] = False
        r = np.flatnonzero(ok)
        vv, st = sample(g, xs[r], ys[r], zs[r], t1[r])
        flags[a[r[st != 0]]] |= st[st != 0]
        ok[r[st != 0]] = False
        r = r[st == 0]
        v1[r] = vv[st == 0]
        active[a[~ok]] = False
        b = a[r]
        pos[b, step] = np.column_stack([xs[r], ys[r], zs[r], t1[r]])
        val[b, step] = v1[r]
        npts[b] = step + 1
        xn[b], yn[b], zn[b], tn[b] = xs[r], ys[r], zs[r], t1[r]
        vn[b] = v1[r]
        if mode == 1:
            wcount[b] = np.where(v1[r, 2] < env_w, wcount[b] + 1, 0)
            if step > min_steps:
                flags[b[v1[r, 3] < env_dbz]] |= ENV_DBZ
                flags[b[wcount[b] >= env_w_steps]] |= ENV_W
                active[b[flags[b] != 0]] = False
        elif mode == 2:
            dcount[b] = np.where(v1[r, 3] < env_dbz, dcount[b] + 1, 0)
            if step > min_steps:
                env = (dcount[b] >= env_dbz_steps) & (
                    (zs[r] - z_sfc >= cold_pool_depth) | (v1[r, 5] >= 0.5)
                )
                flags[b[env]] |= ENV_DBZ
                active[b[env]] = False
        if step == max_steps:
            flags[np.flatnonzero(active)] |= MAX_STEPS
    return pos, val, npts, flags


# ---------------------------------------------------------------------------
# Thermodynamics
# ---------------------------------------------------------------------------


def es_water(t):
    tc = t - T0
    return ES0 * np.exp(17.67 * tc / (tc + 243.5))


def qvs_water(t, p):
    e = es_water(t)
    return EPS * e / np.maximum(p - e, 1.0)


def es_ice(t):
    return ES0 * np.exp(LS_L / RV * (1.0 / T0 - 1.0 / t))


def lv_bolton(t):
    return 2.501e6 - 2370.0 * (t - T0)


def exner(p):
    return np.power(p / P0, KAPPA)


def air_density(theta, p):
    return P0 * np.power(p / P0, 1.0 - KAPPA) / (RD * theta)


def adjust(th, qv, qc, p):
    """Saturation adjustment; returns new (theta, qv, qc) and the heating."""
    pi = exner(p)
    t = th * pi
    need = (qv > qvs_water(t, p)) | (qc > 0.0)
    lv = lv_bolton(t)
    dq = np.zeros_like(t)
    for _ in range(6):
        tn = t + lv * dq / CP
        e = es_water(tn)
        qs = EPS * e / np.maximum(p - e, 1.0)
        tc = tn - T0
        dlnes = 17.67 * 243.5 / ((tc + 243.5) * (tc + 243.5))
        dqs = qs * p / np.maximum(p - e, 1.0) * dlnes
        f = qv - dq - qs
        fp = -1.0 - lv / CP * dqs
        dq = dq - f / fp
    dq = np.where(dq < -qc, -qc, dq)
    dq = np.where(need, dq, 0.0)
    dth = lv * dq / (CP * pi)
    return th + dth, qv - dq, qc + dq, dth


def air_props(t, p, rho):
    ka = (0.441635 + 0.0071 * t) * 1.0e-2
    psi = 2.11e-5 * np.power(t / T0, 1.94) * (1.0e5 / p)
    nu = (0.379565 + 0.0049 * t) * 1.0e-5 / rho
    return ka, psi, nu, nu / psi


def lfo_rates(t, p, rho, qv, qc, qr, nr, qg, ng, rhog, rho0, sw):
    """LFO83 rates (n, 7): revp, racw, gacw, gacr, gmlt, gsub, gfr."""
    arrs = np.broadcast_arrays(
        *(np.asarray(a, float) for a in (t, p, rho, qv, qc, qr, nr, qg, ng, rhog))
    )
    t, p, rho, qv, qc, qr, nr, qg, ng, rhog = arrs
    sw = int(sw)
    zero = np.zeros_like(t)
    rain = (qr > 1.0e-12) & (nr > 0.0)
    graupel = (qg > 1.0e-12) & (ng > 0.0)
    ka, psi, nu, sc = air_props(t, p, rho)
    tc = t - T0
    with np.errstate(all="ignore"):
        lr = np.where(
            rain, np.cbrt(np.pi * RHO_W * nr / (rho * np.where(rain, qr, 1.0))), 1.0
        )
        n0r = nr * lr
        lg = np.where(
            graupel,
            np.cbrt(np.pi * rhog * ng / (rho * np.where(graupel, qg, 1.0))),
            1.0,
        )
        n0g = ng * lg
        gfall = np.sqrt(4.0 * G * rhog / (3.0 * C_D * rho))
        racw = zero
        if sw & RACW:
            racw = np.where(
                rain & (qc > 0.0),
                np.pi
                * n0r
                * A_R
                * qc
                * math.gamma(3.0 + B_R)
                / (4.0 * np.power(lr, 3.0 + B_R))
                * np.sqrt(rho0 / rho),
                0.0,
            )
        qs = qvs_water(t, p)
        s = qv / qs
        revp = zero
        if sw & REVP:
            vent = 0.78 / (lr * lr) + 0.31 * np.cbrt(sc) * math.gamma(
                0.5 * (B_R + 5.0)
            ) * np.sqrt(A_R) / np.sqrt(nu) * np.power(rho0 / rho, 0.25) * np.power(
                lr, -0.5 * (B_R + 5.0)
            )
            den = LV_L * LV_L / (ka * RV * t * t) + 1.0 / (rho * qs * psi)
            revp = np.where(
                rain & (s < 1.0), 2.0 * np.pi * (s - 1.0) * n0r * vent / rho / den, 0.0
            )
        gfr = zero
        if sw & GFR:
            gfr = np.where(
                rain & (t < T0),
                20.0
                * np.pi
                * np.pi
                * B_PR
                * n0r
                * (RHO_W / rho)
                * (np.exp(A_PR * (T0 - t)) - 1.0)
                * np.power(lr, -7.0),
                0.0,
            )
        gacw = zero
        if sw & (GACW | GMLT):
            gacw = np.where(
                graupel & (qc > 0.0),
                np.pi * n0g * qc * math.gamma(3.5) / (4.0 * np.power(lg, 3.5)) * gfall,
                0.0,
            )
        vent_g = 0.78 / (lg * lg) + 0.31 * np.cbrt(sc) * math.gamma(2.75) * np.sqrt(
            gfall
        ) / np.sqrt(nu) * np.power(lg, -2.75)
        gacr = zero
        gmlt = zero
        if sw & GMLT:
            melt = graupel & (t >= T0)
            ur = (
                A_R
                * math.gamma(4.0 + B_R)
                / (6.0 * np.power(lr, B_R))
                * np.sqrt(rho0 / rho)
            )
            ug = math.gamma(4.5) / (6.0 * np.sqrt(lg)) * gfall
            gacr = np.where(
                melt & rain,
                np.pi
                * np.pi
                * n0g
                * n0r
                * np.abs(ug - ur)
                * (RHO_W / rho)
                * (
                    5.0 / (np.power(lr, 6.0) * lg)
                    + 2.0 / (np.power(lr, 5.0) * lg * lg)
                    + 0.5 / (np.power(lr, 4.0) * lg * lg * lg)
                ),
                0.0,
            )
            drs = EPS * ES0 / (p - ES0) - qv
            gm = -2.0 * np.pi / (rho * LF_L) * (
                ka * tc - LV_L * psi * rho * drs
            ) * n0g * vent_g - CW * tc / LF_L * (gacw + gacr)
            gmlt = np.where(melt, np.minimum(gm, 0.0), 0.0)
        gsub = zero
        if sw & GSUB:
            ei = es_ice(t)
            qsi = EPS * ei / np.maximum(p - ei, 1.0)
            si = qv / qsi
            a2 = LS_L * LS_L / (ka * RV * t * t)
            b2 = 1.0 / (rho * qsi * psi)
            gsub = np.where(
                graupel & (t < T0) & (qc <= 0.0) & (si < 1.0),
                2.0 * np.pi * (si - 1.0) / (rho * (a2 + b2)) * n0g * vent_g,
                0.0,
            )
        if not sw & GACW:
            gacw = zero
    out = np.stack(
        np.broadcast_arrays(revp, racw, gacw, gacr, gmlt, gsub, gfr), axis=-1
    )
    return np.where(np.isfinite(out), out, 0.0) * 1.0


def tendencies(r, th, p, qv, qc, qr, qg, dt):
    """Limited tendencies (theta, qv, qc) and the theta parts (revp, gmlt, gsub, frz)."""
    revp, racw, gacw, gacr, gmlt, gsub, gfr = (r[..., k].copy() for k in range(7))
    pi = exner(p)
    t = th * pi
    with np.errstate(all="ignore"):
        qs = qvs_water(t, p)
        gap = np.maximum(qs - qv, 0.0) / (1.0 + LV_L * LV_L * qs / (CP * RV * t * t))
        lim = np.minimum(gap, qr) / dt
        revp = np.where((revp < 0.0) & (-revp > lim), -lim, revp)
        ei = es_ice(t)
        qsi = EPS * ei / np.maximum(p - ei, 1.0)
        gap = np.maximum(qsi - qv, 0.0) / (1.0 + LS_L * LS_L * qsi / (CP * RV * t * t))
        lim = np.minimum(gap, qg) / dt
        gsub = np.where((gsub < 0.0) & (-gsub > lim), -lim, gsub)
        col = racw + gacw
        scale = (col * dt > qc) & (col > 0.0)
        f = np.where(scale, qc / np.where(scale, col * dt, 1.0), 1.0)
        racw = np.where(scale, racw * f, racw)
        gacw = np.where(scale, gacw * f, gacw)
        gmlt = np.where(-gmlt * dt > qg, -qg / dt, gmlt)
        gfr = np.where(gfr * dt > qr, qr / dt, gfr)
    c = 1.0 / (CP * pi)
    d_revp = c * LV_L * revp
    d_gsub = c * LS_L * gsub
    d_gmlt = c * LF_L * gmlt
    d_frz = c * LF_L * (gfr + np.where(t < T0, gacw, 0.0))
    dth = d_revp + d_gsub + d_gmlt + d_frz
    return dth, -revp - gsub, -(racw + gacw), d_revp, d_gmlt, d_gsub, d_frz


def profile(table, z0, dz, zp):
    n = table.shape[0]
    f = np.clip((zp - z0) / dz, 0.0, float(n - 1))
    i = np.minimum(f.astype(np.int64), n - 2)
    a = f - i
    return (1.0 - a)[:, None] * table[i] + a[:, None] * table[i + 1]


def damping_rate(q, w, u, v, ub, vb, qp, zagl):
    aw = np.abs(w)
    f = np.clip(qp / q[Q["qp0"]], 0.0, 1.0)
    c0 = (1.0 - f) * q[Q["cmin"]] + f * q[Q["cmax"]]
    vel = np.where(
        (w > q[Q["w0"]]) | (w < -q[Q["w0"]]),
        aw,
        np.sqrt((u - ub) * (u - ub) + (v - vb) * (v - vb)),
    )
    ld = np.where(
        w > q[Q["w0"]],
        q[Q["ld1"]] + (aw - q[Q["w0"]]) * q[Q["lw1"]],
        np.where(
            w < -q[Q["w0"]],
            q[Q["ld2"]] + (aw - q[Q["w0"]]) * q[Q["lw2"]],
            q[Q["cd"]] / c0,
        ),
    )
    return q[Q["cd"]] * vel / (ld * np.exp(q[Q["b"]] * zagl / 1000.0))


def dla(g, par, starts, surface, base, base_z0, base_dz, precip, meso, grad, q):
    """Diabatic Lagrangian analysis (NumPy reference of the kernel's dla)."""
    pos, val, npts, flags = build_paths(g, par, starts)
    n = npts.size
    dt = par[0]
    sw = int(q[Q["switches"]])
    zsfc = q[Q["z_sfc"]]
    out = np.full((n, 3), np.nan)
    bud = np.full((n, N_BUDGET), np.nan)
    org = np.full((n, 5), np.nan)
    has = npts > 0
    org[has, :4] = pos[np.flatnonzero(has), npts[has] - 1]
    steps = np.arange(val.shape[1])[None, :] < npts[:, None]
    nvalid = ((val[:, :, 4] >= 0.5) & steps).sum(axis=1)
    org[has, 4] = nvalid[has] / npts[has]
    env = has & ((flags & ENVIRONMENT) != 0)
    idx = np.flatnonzero(env)
    if idx.size == 0:
        return out, bud, org, npts, flags
    surf = np.asarray(surface)[idx] != 0
    k = npts[idx]

    def base_at(xp, yp, zp, tp):
        if meso is not None:
            m, _ = sample(meso, xp, yp, zp, tp, False)
            return m[:, 0], m[:, 1]
        b = profile(base, base_z0, base_dz, zp)
        return b[:, 1], b[:, 2]

    p0 = pos[idx, k - 1]
    theta, qv = base_at(p0[:, 0], p0[:, 1], p0[:, 2], p0[:, 3])
    qc = np.zeros(idx.size)
    b = np.zeros((idx.size, N_BUDGET))
    pa = profile(base, base_z0, base_dz, p0[:, 2])[:, 0]
    if sw & COND:
        theta, qv, qc, dth = adjust(theta, qv, qc, pa)
        b[:, 0] += dth
    ns = max(1, int(math.ceil(dt / q[Q["dt_small"]] - 1e-9)))
    h = dt / ns
    qthr = np.where(surf, q[Q["q0"]], q[Q["q1"]])
    for j in range(int(k.max()) - 1):
        m = k - 1 - j
        r = np.flatnonzero(m >= 1)
        rows = idx[r]
        A = pos[rows, m[r]]
        B = pos[rows, m[r] - 1]
        VA = val[rows, m[r]]
        zagl = A[:, 2] - zsfc
        pb = profile(base, base_z0, base_dz, B[:, 2])[:, 0]
        if precip is not None:
            pr, st = sample(precip, A[:, 0], A[:, 1], A[:, 2], A[:, 3], False)
            pr[st != 0] = 0.0
            pr = np.where(pr > 0.0, pr, 0.0)
        else:
            pr = np.zeros((r.size, 4))
        qr, nr, qg, ng = pr.T
        th, qvr, qcr, par_ = theta[r], qv[r], qc[r], pa[r]
        if sw & MICRO:
            t = th * exner(par_)
            rho = air_density(th, par_)
            rhog = q[Q["rho_g_sfc"]] + (
                q[Q["rho_g_5km"]] - q[Q["rho_g_sfc"]]
            ) * np.clip(zagl / 5000.0, 0.0, 1.0)
            rates = lfo_rates(
                t, par_, rho, qvr, qcr, qr, nr, qg, ng, rhog, q[Q["rho0"]], sw
            )
            d = tendencies(rates, th, par_, qvr, qcr, qr, qg, dt)
        else:
            zz = np.zeros(r.size)
            d = (zz,) * 7
        fth = np.zeros(r.size)
        fqv = np.zeros(r.size)
        if sw & FLUX:
            use = (zagl <= q[Q["z_bl"]]) & (qcr + qr + qg <= q[Q["q1"]])
            gth = np.zeros(r.size)
            gqv = np.zeros(r.size)
            if grad is not None:
                gr, st = sample(
                    grad,
                    A[:, 0],
                    A[:, 1],
                    np.full(r.size, grad.z[0]),
                    A[:, 3],
                    False,
                )
                ok = st == 0
                gth = np.where(ok, VA[:, 0] * gr[:, 0] + VA[:, 1] * gr[:, 1], 0.0)
                gqv = np.where(ok, VA[:, 0] * gr[:, 2] + VA[:, 1] * gr[:, 3], 0.0)
            e = np.exp(-q[Q["b_f"]] * zagl / 1000.0)
            fth = np.where(use, e * (gth + q[Q["flux_theta"]]), 0.0)
            fqv = np.where(use, e * (gqv + q[Q["flux_qv"]]), 0.0)
        bb = b[r]
        for s in range(1, ns + 1):
            ps = par_ + (pb - par_) * float(s) / float(ns)
            th = th + h * (d[0] + fth)
            qvr = qvr + h * (d[1] + fqv)
            qcr = qcr + h * d[2]
            qvr = np.where(qvr < 0.0, 0.0, qvr)
            qcr = np.where(qcr < 0.0, 0.0, qcr)
            if sw & COND:
                th, qvr, qcr, dth = adjust(th, qvr, qcr, ps)
                bb[:, 0] += dth
        bb[:, 1] += dt * d[3]
        bb[:, 2] += dt * d[4]
        bb[:, 3] += dt * d[5]
        bb[:, 4] += dt * d[6]
        bb[:, 6] += dt * fth
        if sw & DAMP:
            use = qr + qg >= qthr[r]
            pab = profile(base, base_z0, base_dz, A[:, 2])
            kd = damping_rate(
                q, VA[:, 2], VA[:, 0], VA[:, 1], pab[:, 3], pab[:, 4], qr + qg, zagl
            )
            f = np.exp(-kd * dt)
            thb, qvb = base_at(B[:, 0], B[:, 1], B[:, 2], B[:, 3])
            th_new = thb + (th - thb) * f
            bb[:, 5] += np.where(use, th_new - th, 0.0)
            th = np.where(use, th_new, th)
            qvr = np.where(use, qvb + (qvr - qvb) * f, qvr)
            qcr = np.where(use, qcr * f, qcr)
        theta[r], qv[r], qc[r] = th, qvr, qcr
        b[r] = bb
        pa[r] = pb
    out[idx] = np.column_stack([theta, qv, qc])
    bud[idx] = b
    return out, bud, org, npts, flags


def rates(theta, p, qv, qc, qr, nr, qg, ng, rhog, rho0, switches, dt):
    """LFO83 rates (n, 7) and limited tendencies (n, 3) for 1-D states."""
    theta, p = np.asarray(theta, float), np.asarray(p, float)
    t = theta * exner(p)
    rho = air_density(theta, p)
    r = lfo_rates(t, p, rho, qv, qc, qr, nr, qg, ng, rhog, rho0, switches)
    d = tendencies(
        r,
        theta,
        p,
        np.asarray(qv, float),
        np.asarray(qc, float),
        np.asarray(qr, float),
        np.asarray(qg, float),
        dt,
    )
    return r, np.column_stack(d[:3])
