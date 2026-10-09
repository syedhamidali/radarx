#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
NumPy reference of the compiled trajectory and DLA kernel.

Same steps, in the same order, as ``radarx/retrieve/_lagrangian.cpp``,
vectorised over trajectories. Used when the kernel is not built and as the
test oracle of the kernel.

Sources of the numbers (details and the full reference list are in the module
documentation of :mod:`radarx.retrieve.lagrangian` and
:mod:`radarx.retrieve.diabatic_lagrangian`; the pointers were checked against
the papers): Z13a = Ziegler (2013a, J. Atmos. Oceanic Technol. 30, 2248-2265,
https://doi.org/10.1175/JTECH-D-12-00194.1); Z07 = Ziegler et al. (2007, Mon.
Wea. Rev. 135, 2417-2442, https://doi.org/10.1175/MWR3396.1); LFO83 = Lin et al.
(1983, J. Climate Appl. Meteor. 22, 1065-1092,
https://doi.org/10.1175/1520-0450(1983)022<1065:BPOTSF>2.0.CO;2); Tao et al.
(1989, Mon. Wea. Rev. 117, 231-235,
https://doi.org/10.1175/1520-0493(1989)117<0231:AIWSA>2.0.CO;2); Hsie et al.
(1980, J. Appl. Meteor. 19, 950-977,
https://doi.org/10.1175/1520-0450(1980)019<0950:NSOIPC>2.0.CO;2). Numbers
without a pointer are radarx choices. The LFO83 rates are implemented from
LFO83, not from the "modified LFO" supplement of Gilmore et al. (2004a), which
was not consulted.
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
VALID_FRACTION = 64  # (DLA, set in Python) too little time in valid winds
BOUNDARY_STORM = 128  # left through a lateral boundary that is not environment
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
ICE = 512  # mixed-phase saturation adjustment (Tao et al. 1989) with cloud ice
MICRO = REVP | RACW | GACW | GMLT | GSUB | GFR
INSITU = 256  # DLA flag: initial state from in situ observations (Ziegler et al. 2007)

T0 = 273.15  # melting temperature, LFO83 appendix, p. 1091
P0 = 1.0e5  # Pa
RD = 287.04  # J kg-1 K-1, Z13a sect. 2a (p. 2250)
KAPPA = 0.2854  # Z13a sect. 2a (p. 2250)
CP = RD / KAPPA  # radarx choice (1005.8; LFO83 list 1005, appendix p. 1089)
RV = 461.5  # R_w, LFO83 appendix, p. 1091
EPS = RD / RV  # radarx
ES0 = (
    611.2  # e_s(0 degC) of the Bolton (1980) fit, Pa; not re-checked against the paper
)

# LFO83 appendix (pp. 1089-1092), SI units
LV_L = 2.5e6  # L_v, J kg-1
LF_L = 3.336e5  # L_f, J kg-1
LS_L = 2.8336e6  # L_s, J kg-1
CW = 4.187e3  # C_w, J kg-1 K-1
A_R = 841.99  # a = 2115 cm^0.2 s-1 in m^0.2 s-1, eq. (7), p. 1069
B_R = 0.8  # b, eq. (7)
C_D = 0.6  # hail drag coefficient, eq. (9), p. 1069
G = 9.805  # g = 980.5 cm s-2
A_PR = 0.66  # A', K-1, Bigg freezing, eq. (45), p. 1075
B_PR = 100.0  # B', m-3 s-1, eq. (45)
RHO_W = 1000.0  # density of water, LFO83 appendix (1 g cm-3)

# Tao, Simpson and McCumber (1989), eqs. (3a), (3b), p. 232: a = 17.2693882
# and 21.8745584, b = 3.8 / P with P in mb; eqs. (6c), (6d) use 237.3 and 265.5
# (= 273.16 - 35.86 and 273.16 - 7.66)
TAO_A1 = 17.2693882
TAO_A2 = 21.8745584
TAO_B = 3.8
# homogeneous freezing of cloud water at T <= -40 degC (LFO83 sect. 3f, p. 1077;
# Hsie et al. 1980, sect. 3b5, p. 956)
T_HOM = 233.15

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
    "t00",
)
Q = {k: i for i, k in enumerate(THERMO_KEYS)}
N_BUDGET = 7
N_OBS_PAR = 6  # window, radius, z_tolerance, kappa_s, tau_i, tau_L


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


WEST, EAST, SOUTH, NORTH = 1, 2, 4, 8


def sides_of(g, xp, yp):
    """Sides (bit mask) of the domain beyond which the points lie."""
    return (
        np.where(xp < g.x[0], WEST, 0)
        | np.where(xp > g.x[-1], EAST, 0)
        | np.where(yp < g.y[0], SOUTH, 0)
        | np.where(yp > g.y[-1], NORTH, 0)
    ).astype(np.int32)


def exit_sides(g, xp, yp, tp):
    """Sides of the fixed grid, else of the analysis moving with the storm,
    beyond which the points lie (as the kernel's exit_sides)."""
    xp, yp, tp = (np.asarray(a, float) for a in (xp, yp, tp))
    s = sides_of(g, xp, yp)
    n = xp.size
    lev0 = np.zeros(n, np.int64)
    lev1 = np.zeros(n, np.int64)
    wt1 = np.zeros(n)
    after = tp > g.t[-1]
    lev0[after] = g.t.size - 1
    inside = (tp >= g.t[0]) & ~after
    if g.t.size > 1:
        i = _bracket(g.t, tp)
        a = (tp - g.t[i]) / (g.t[i + 1] - g.t[i])
        lev0 = np.where(inside, i, lev0)
        lev1 = np.where(inside, i + 1, lev1)
        wt1 = np.where(inside, a, wt1)
    wt0 = np.where(inside, 1.0 - wt1, 1.0)
    for lev, wt in ((lev0, wt0), (lev1, wt1)):
        dtl = tp - g.t[lev]
        sl = sides_of(g, xp - g.cx * dtl, yp - g.cy * dtl)
        s = np.where((s == 0) & (wt > 0.0), sl, s)
    return s


def boundary_flag(par, sides, vlast):
    """BOUNDARY or BOUNDARY_STORM for exits through ``sides`` (as the kernel)."""
    mode, rule, allowed = int(par[4]), int(par[12]), int(par[13])
    sides = np.asarray(sides, np.int32)
    env = np.ones(sides.shape, bool)
    if mode != 0:
        if rule == 1:
            env = vlast[:, 5] >= 0.5
        elif rule == 2:
            env = (vlast[:, 3] < par[6]) | (vlast[:, 5] >= 0.5)
        elif rule == 3:
            env = (sides & allowed) != 0
    return np.where(env, BOUNDARY, BOUNDARY_STORM).astype(np.int32)


def _classify(g, par, st, xp, yp, tp, vlast):
    """Sample status with BOUNDARY (left the analysis moving with the storm)
    classified by :func:`boundary_flag`."""
    f = np.array(st, np.int32, copy=True)
    bnd = f == BOUNDARY
    if bnd.any():
        f[bnd] = boundary_flag(
            par, exit_sides(g, xp[bnd], yp[bnd], tp[bnd]), vlast[bnd]
        )
    return f


def build_paths(g, par, starts):
    """Paths of all start points: pos (n, m, 4), val (n, m, 6), npts, flags."""
    dt, n_iter, max_steps = par[0], int(par[1]), int(par[2])
    # Euler predictor and n_iter trapezoidal corrector iterations: radarx's reading
    # of the "first-order predictor corrector scheme as in Z07" with three
    # iterations (Z13a sect. 2b, p. 2250; Z07 p. 2422)
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
        flags[a[bad]] |= boundary_flag(par, sides_of(g, xs[bad], ys[bad]), vn[a[bad]])
        ok &= ~bad
        v1 = np.zeros((a.size, NW))

        for _ in range(n_iter):
            r = np.flatnonzero(ok)
            vv, st = sample(g, xs[r], ys[r], zs[r], t1[r])
            flags[a[r[st != 0]]] |= _classify(
                g, par, st, xs[r], ys[r], t1[r], vn[a[r]]
            )[st != 0]
            ok[r[st != 0]] = False
            r = r[st == 0]
            v1[r] = vv[st == 0]
            xs[r] = xn[a[r]] + 0.5 * h * (vn[a[r], 0] + v1[r, 0])
            ys[r] = yn[a[r]] + 0.5 * h * (vn[a[r], 1] + v1[r, 1])
            zs[r] = np.clip(zn[a[r]] + 0.5 * h * (vn[a[r], 2] + v1[r, 2]), zlo, zhi)
            bad = outside(xs[r], ys[r])
            rb = r[bad]
            flags[a[rb]] |= boundary_flag(par, sides_of(g, xs[rb], ys[rb]), vn[a[rb]])
            ok[rb] = False
        r = np.flatnonzero(ok)
        vv, st = sample(g, xs[r], ys[r], zs[r], t1[r])
        flags[a[r[st != 0]]] |= _classify(g, par, st, xs[r], ys[r], t1[r], vn[a[r]])[
            st != 0
        ]
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
    # Bolton (1980) fit: 611.2 Pa, 17.67, 243.5 K (radarx choice, not checked
    # against the paper)
    tc = t - T0
    return ES0 * np.exp(17.67 * tc / (tc + 243.5))


def qvs_water(t, p):
    e = es_water(t)
    return EPS * e / np.maximum(p - e, 1.0)


def es_ice(t):
    return ES0 * np.exp(LS_L / RV * (1.0 / T0 - 1.0 / t))


def lv_bolton(t):
    # radarx choice (2.501e6 - 2370 (T - T0) J kg-1); not from Z13a
    return 2.501e6 - 2370.0 * (t - T0)


def exner(p):
    return np.power(p / P0, KAPPA)


def air_density(theta, p):
    # Z13a (sect. 2a, p. 2250) prints rho = 1e5 [(p/1000 mb)^0.2854]^2.509 /
    # (287.04 theta), exponent 0.7161; radarx uses 1 - kappa = 0.7146 (< 0.1 %
    # difference in density above 500 hPa)
    return P0 * np.power(p / P0, 1.0 - KAPPA) / (RD * theta)


def adjust(th, qv, qc, p):
    """Saturation adjustment; returns new (theta, qv, qc) and the heating.

    Z13a sect. 2g (p. 2257) applies "ideas from the Eulerian frame modeling
    approach of Soong and Ogura (1973)"; the six Newton iterations on
    q_v - dq = q_vs(T + L dq / c_p) are radarx's implementation.
    """
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


def tao_saturation(t, p):
    """Saturation mixing ratios over water and ice, Tao et al. (1989) (3a), (3b)."""
    b = TAO_B / (p / 100.0)
    qws = b * np.exp(TAO_A1 * (t - 273.16) / (t - 35.86))
    qis = b * np.exp(TAO_A2 * (t - 273.16) / (t - 7.66))
    return qws, qis


def adjust_ice(th, qv, qc, qi, p, t00):
    """Ice-water saturation adjustment of Tao et al. (1989), after melting of
    cloud ice above 0 degC and homogeneous freezing of cloud water at or below
    -40 degC (LFO83, Hsie et al. 1980); returns new (theta, q_v, q_c, q_i) and
    the heating of the adjustment and of the phase changes."""
    pi = exner(p)
    t = th * pi
    # P_IMLT and P_IHOM, heating from Tao et al. (4a) with dq_c = -dq_i
    melt = np.where(t > T0, qi, 0.0)
    frz = np.where(t <= T_HOM, qc, 0.0)
    dth_f = (LS_L - LV_L) * (frz - melt) / (CP * pi)
    qc = qc + melt - frz
    qi = qi - melt + frz
    th = th + dth_f
    t = th * pi
    cnd = np.clip((t - t00) / (T0 - t00), 0.0, 1.0)
    dep = 1.0 - cnd
    qws, qis = tao_saturation(t, p)
    cloud = qc + qi
    has = cloud > 0.0
    den = np.where(has, cloud, 1.0)
    wc = np.where(has, qc / den, cnd)
    wi = np.where(has, qi / den, dep)
    qvs = wc * qws + wi * qis
    a1 = 237.3 * TAO_A1 * pi / ((t - 35.86) * (t - 35.86))
    a2 = 265.5 * TAO_A2 * pi / ((t - 7.66) * (t - 7.66))
    r1 = qv - qvs  # (6a)
    r2 = a1 * wc * qws + a2 * wi * qis  # (6b)
    a3 = (LV_L * cnd + LS_L * dep) / (CP * pi)  # (6e)
    dq = r1 / (1.0 + r2 * a3)  # (7b)
    dqc = np.maximum(dq * cnd, -qc)  # (2b), limited by q_c
    dqi = np.maximum(dq * dep, -qi)  # (2c), limited by q_i
    need = (r1 > 0.0) | has
    dqc = np.where(need, dqc, 0.0)
    dqi = np.where(need, dqi, 0.0)
    dth = (LV_L * dqc + LS_L * dqi) / (CP * pi)  # (4a)
    return th + dth, qv - dqc - dqi, qc + dqc, qi + dqi, dth, dth_f


def air_props(t, p, rho):
    # Thermal conductivity, vapour diffusivity, kinematic viscosity of air as
    # in Kumjian and Ryzhkov (2010, appendix); not checked against that
    # paper, same expressions as radarx.retrieve.evaporation.
    ka = (0.441635 + 0.0071 * t) * 1.0e-2
    psi = 2.11e-5 * np.power(t / T0, 1.94) * (1.0e5 / p)
    nu = (0.379565 + 0.0049 * t) * 1.0e-5 / rho
    return ka, psi, nu, nu / psi


def lfo_rates(t, p, rho, qv, qc, qr, nr, qg, ng, rhog, rho0, sw, qi=0.0):
    """LFO83 rates (n, 7): revp, racw, gacw, gacr, gmlt, gsub, gfr; q_i only
    enters the in-cloud test of the graupel sublimation (delta_1, eq. 20, p. 1070).

    Equations of Lin et al. (1983): lambda_R, lambda_G (4), (6), p. 1068; fall
    speeds (7), (9) and mass-weighted velocities (11), (13), pp. 1068-1069;
    P_GACW (40) and P_GACR (42), p. 1075; P_GFR (45), p. 1075; P_GSUB (46) with
    A'', B'' of (31), pp. 1072 and 1076; P_GMLT (47), p. 1076; P_RACW (51),
    p. 1076; P_REVP (52), p. 1077. Eq. (46) is printed with
    (4 g rho_G / 3 C_D rho)^(1/4) but eq. (47) without rho; the form with rho,
    which the fall speed (9) requires dimensionally, is used for both.
    """
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
                graupel & (t < T0) & (qc + qi <= 0.0) & (si < 1.0),
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
    """Limited tendencies (theta, qv, qc) and the theta parts (revp, gmlt, gsub, frz).

    The limits (evaporation to saturation and to the available rain or
    graupel, collection to the available cloud water, melting and freezing to
    the available graupel and rain) are numerical safeguards of radarx, not
    from LFO83 or Z13a. The heating is L / (c_p Pi) times the rate (radarx
    thermodynamics, cf. LFO83 eq. 53); freezing of cloud collected by graupel
    below 0 degC is heated with L_f.
    """
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
    # Ziegler (2013a) eqs. (22)-(26), pp. 2257-2258: K = c_d V / (L_d exp(b z)),
    # with z in km (radarx's reading; the unit is not stated in the paper)
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


def insitu_match(pos, npts, dt, obs, op):
    """Initial states from in situ observations (Ziegler et al. 2007, eq. 1,
    p. 2422: first-pass Barnes weight exp(-r^2/kappa_s - t_i^2/tau_i - t_L^2/tau_L)).

    Returns the weight sum (0 without a candidate), the weighted theta and
    q_v and the index of the stored path point nearest the time of the
    candidate with the largest weight (-1 without a candidate)."""
    n, m = npts.size, pos.shape[1]
    window, radius, ztol, kap, tau_i, tau_l = (float(v) for v in op)
    h = -float(dt)
    wsum = np.zeros(n)
    sth = np.zeros(n)
    sqv = np.zeros(n)
    wbest = np.zeros(n)
    kbest = np.full(n, -1, np.int64)
    rows = np.arange(n)
    for xo, yo, zo, to, tho, qvo in np.asarray(obs, float).reshape(-1, 6):
        if not (to <= 0.0 and -to <= window):
            continue
        f = to / h
        k0 = int(math.floor(f))
        a = f - k0
        need = k0 + (1 if a > 0.0 else 0)
        if need > m - 1:
            continue
        ok = npts - 1 >= need
        p0 = pos[rows, k0]
        if a > 0.0:
            p1 = pos[rows, k0 + 1]
            pt = (1.0 - a) * p0 + a * p1
        else:
            pt = p0
        with np.errstate(invalid="ignore"):
            dx = pt[:, 0] - xo
            dy = pt[:, 1] - yo
            r2 = dx * dx + dy * dy
            ok &= (r2 <= radius * radius) & (np.abs(pt[:, 2] - zo) <= ztol)
        w = np.exp(-r2 / kap - to * to / tau_i - to * to / tau_l)
        w = np.where(ok, w, 0.0)
        wsum += w
        sth += w * tho
        sqv += w * qvo
        better = ok & (w > wbest)
        wbest = np.where(better, w, wbest)
        kbest = np.where(better, int(math.floor(f + 0.5)), kbest)
    with np.errstate(invalid="ignore", divide="ignore"):
        return wsum, sth / wsum, sqv / wsum, kbest


class _Forward:
    """Forward integration of the DLA along stored paths (as the kernel's
    integrate), vectorised over trajectories."""

    def __init__(self, base, base_z0, base_dz, precip, meso, grad, q, dt):
        self.base, self.z0, self.dz = base, base_z0, base_dz
        self.precip, self.meso, self.grad, self.q, self.dt = precip, meso, grad, q, dt
        self.sw = int(q[Q["switches"]])
        self.ice = bool(self.sw & ICE)

    def pressure(self, zp):
        return profile(self.base, self.z0, self.dz, zp)[:, 0]

    def base_at(self, xp, yp, zp, tp):
        if self.meso is not None:
            m, _ = sample(self.meso, xp, yp, zp, tp, False)
            return m[:, 0], m[:, 1]
        b = profile(self.base, self.z0, self.dz, zp)
        return b[:, 1], b[:, 2]

    def cond(self, s, p, b):
        """Saturation adjustment of the state s = [theta, q_v, q_c, q_i]."""
        if not self.sw & COND:
            return s
        if self.ice:
            th, v, c, i, d1, d2 = adjust_ice(*s, p, self.q[Q["t00"]])
            b[:, 4] += d2
        else:
            th, v, c, d1 = adjust(s[0], s[1], s[2], p)
            i = s[3]
        b[:, 0] += d1
        return [th, v, c, i]

    def precipitation(self, A):
        if self.precip is None:
            return np.zeros((A.shape[0], 4))
        pr, st = sample(self.precip, A[:, 0], A[:, 1], A[:, 2], A[:, 3], False)
        pr[st != 0] = 0.0
        return np.where(pr > 0.0, pr, 0.0)

    def micro(self, s, p, pr, zagl):
        q = self.q
        if not self.sw & MICRO:
            return (np.zeros(p.size),) * 7
        th, qvr, qcr, qir = s
        qr, nr, qg, ng = pr.T
        t = th * exner(p)
        rho = air_density(th, p)
        rhog = q[Q["rho_g_sfc"]] + (q[Q["rho_g_5km"]] - q[Q["rho_g_sfc"]]) * np.clip(
            zagl / 5000.0, 0.0, 1.0
        )
        rates = lfo_rates(
            t, p, rho, qvr, qcr, qr, nr, qg, ng, rhog, q[Q["rho0"]], self.sw, qir
        )
        return tendencies(rates, th, p, qvr, qcr, qr, qg, self.dt)

    def flux(self, s, pr, A, VA, zagl):
        q, n = self.q, A.shape[0]
        if not self.sw & FLUX:
            return np.zeros(n), np.zeros(n)
        use = (zagl <= q[Q["z_bl"]]) & (s[2] + s[3] + pr[:, 0] + pr[:, 2] <= q[Q["q1"]])
        gth = np.zeros(n)
        gqv = np.zeros(n)
        if self.grad is not None:
            g = self.grad
            gr, st = sample(g, A[:, 0], A[:, 1], np.full(n, g.z[0]), A[:, 3], False)
            ok = st == 0
            gth = np.where(ok, VA[:, 0] * gr[:, 0] + VA[:, 1] * gr[:, 1], 0.0)
            gqv = np.where(ok, VA[:, 0] * gr[:, 2] + VA[:, 1] * gr[:, 3], 0.0)
        e = np.exp(-q[Q["b_f"]] * zagl / 1000.0)
        return (
            np.where(use, e * (gth + q[Q["flux_theta"]]), 0.0),
            np.where(use, e * (gqv + q[Q["flux_qv"]]), 0.0),
        )

    def damp(self, s, pr, A, B, VA, zagl, qthr, b):
        if not self.sw & DAMP:
            return s
        qp = pr[:, 0] + pr[:, 2]
        use = qp >= qthr
        pab = profile(self.base, self.z0, self.dz, A[:, 2])
        kd = damping_rate(
            self.q, VA[:, 2], VA[:, 0], VA[:, 1], pab[:, 3], pab[:, 4], qp, zagl
        )
        f = np.exp(-kd * self.dt)
        thb, qvb = self.base_at(B[:, 0], B[:, 1], B[:, 2], B[:, 3])
        th_new = thb + (s[0] - thb) * f
        b[:, 5] += np.where(use, th_new - s[0], 0.0)
        return [
            np.where(use, th_new, s[0]),
            np.where(use, qvb + (s[1] - qvb) * f, s[1]),
            np.where(use, s[2] * f, s[2]),
            np.where(use, s[3] * f, s[3]),
        ]

    def step(self, s, A, B, VA, pa, qthr, b):
        """One step from point A to point B with sub-steps dt_small."""
        dt, q = self.dt, self.q
        zagl = A[:, 2] - q[Q["z_sfc"]]
        pb = self.pressure(B[:, 2])
        pr = self.precipitation(A)
        d = self.micro(s, pa, pr, zagl)
        fth, fqv = self.flux(s, pr, A, VA, zagl)
        ns = max(1, int(math.ceil(dt / q[Q["dt_small"]] - 1e-9)))
        h = dt / ns
        for k in range(1, ns + 1):
            ps = pa + (pb - pa) * float(k) / float(ns)
            s = [
                s[0] + h * (d[0] + fth),
                np.maximum(s[1] + h * (d[1] + fqv), 0.0),
                np.maximum(s[2] + h * d[2], 0.0),
                s[3],
            ]
            s = self.cond(s, ps, b)
        b[:, 1] += dt * d[3]
        b[:, 2] += dt * d[4]
        b[:, 3] += dt * d[5]
        b[:, 4] += dt * d[6]
        b[:, 6] += dt * fth
        return self.damp(s, pr, A, B, VA, zagl, qthr, b), pb


def dla(
    g,
    par,
    starts,
    surface,
    base,
    base_z0,
    base_dz,
    precip,
    meso,
    grad,
    q,
    obs=None,
    obs_par=None,
):
    """Diabatic Lagrangian analysis (NumPy reference of the kernel's dla).

    Returns (n, 4) theta, q_v, q_c, q_i, the (n, 7) theta budget, (n, 5)
    origin, the number of points, the flags and (n, 2) in situ weight and
    start time."""
    pos, val, npts, flags = build_paths(g, par, starts)
    n = npts.size
    dt = par[0]
    out = np.full((n, 4), np.nan)
    bud = np.full((n, N_BUDGET), np.nan)
    org = np.full((n, 5), np.nan)
    init = np.full((n, 2), np.nan)
    has = npts > 0
    org[has, :4] = pos[np.flatnonzero(has), npts[has] - 1]
    steps = np.arange(val.shape[1])[None, :] < npts[:, None]
    nvalid = ((val[:, :, 4] >= 0.5) & steps).sum(axis=1)
    org[has, 4] = nvalid[has] / npts[has]
    env = has & ((flags & ENVIRONMENT) != 0)
    kstart = npts - 1
    matched = np.zeros(n, bool)
    if obs is not None and len(obs):
        wsum, th_obs, qv_obs, kbest = insitu_match(pos, npts, dt, obs, obs_par)
        matched = has & (kbest >= 0) & (wsum > 0.0)
        kstart = np.where(matched, kbest, kstart)
        flags = np.where(matched, flags | INSITU, flags).astype(np.int32)
        init[matched, 0] = wsum[matched]
        init[matched, 1] = pos[np.flatnonzero(matched), kbest[matched], 3]
    idx = np.flatnonzero(env | matched)
    if idx.size == 0:
        return out, bud, org, npts, flags, init
    fw = _Forward(base, base_z0, base_dz, precip, meso, grad, q, dt)
    k = kstart[idx] + 1
    p0 = pos[idx, k - 1]
    theta, qv = fw.base_at(p0[:, 0], p0[:, 1], p0[:, 2], p0[:, 3])
    mi = matched[idx]
    if mi.any():
        theta = np.where(mi, th_obs[idx], theta)
        qv = np.where(mi, qv_obs[idx], qv)
    b = np.zeros((idx.size, N_BUDGET))
    pa = fw.pressure(p0[:, 2])
    state = np.stack(
        fw.cond([theta, qv, np.zeros(idx.size), np.zeros(idx.size)], pa, b)
    )
    qthr = np.where(np.asarray(surface)[idx] != 0, q[Q["q0"]], q[Q["q1"]])
    for j in range(int(k.max()) - 1):
        m = k - 1 - j
        r = np.flatnonzero(m >= 1)
        rows = idx[r]
        bb = b[r]
        s, pb = fw.step(
            list(state[:, r]),
            pos[rows, m[r]],
            pos[rows, m[r] - 1],
            val[rows, m[r]],
            pa[r],
            qthr[r],
            bb,
        )
        state[:, r] = np.stack(s)
        b[r] = bb
        pa[r] = pb
    out[idx] = state.T
    bud[idx] = b
    return out, bud, org, npts, flags, init


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


def adjust_states(theta, p, qv, qc, qi, t00, ice):
    """Saturation adjustment of 1-D states as the kernel's adjust: (n, 4)
    theta, q_v, q_c, q_i and (n, 2) heating of the adjustment and of the
    freezing or melting of cloud condensate."""
    theta, p, qv, qc, qi = (np.asarray(a, float) for a in (theta, p, qv, qc, qi))
    if ice:
        th, v, c, x, d1, d2 = adjust_ice(theta, qv, qc, qi, p, t00)
    else:
        th, v, c, d1 = adjust(theta, qv, qc, p)
        x, d2 = qi, np.zeros_like(theta)
    return np.column_stack([th, v, c, x]), np.column_stack([d1, d2])
