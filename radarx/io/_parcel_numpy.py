# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
NumPy reference of the parcel kernel (``radarx/io/_parcel.cpp``).

Scalar code per column, the same arithmetic as the kernel (the same order of
operations, bisections of the same number of steps), so that the two agree to
rounding. It is the fallback when the kernel is not built and the oracle of
the tests. See :mod:`radarx.io.parcel` for the definitions.
"""

from __future__ import annotations

import math

import numpy as np

NAN = float("nan")

# constants, in sync with radarx/io/_sounding.cpp and the kernel
RD = 287.04749
RV = 461.52311
EPS = RD / RV
G0 = 9.80665
T0 = 273.15
P_REF = 1.0e5  # Pa, the 1000 hPa of Bolton's potential temperatures
LI_PRESSURE = 50000.0  # Pa
ROWS = (
    "cape",
    "cin",
    "lifted_index",
    "lcl_pressure",
    "lcl_height",
    "lfc_pressure",
    "lfc_height",
    "el_pressure",
    "el_height",
    "parcel_pressure",
    "parcel_temperature",
    "parcel_dewpoint",
)
_BISECT = 80


def esat(t):
    """Bolton (1980) eq. (10), Pa."""
    tc = t - T0
    return 611.2 * math.exp(17.67 * tc / (tc + 243.5))


def dewpoint(e):
    """Eq. (10) inverted, K."""
    ratio = math.log(e / 611.2)
    return T0 + 243.5 * ratio / (17.67 - ratio)


def mixing_ratio(e, p):
    return EPS * e / (p - e)


def kappa(r):
    """Exponent of the potential temperature, Bolton (1980) eq. (6)."""
    return 0.2854 * (1.0 - 0.28 * r)


def lcl_temperature(t, td):
    """Bolton (1980) eq. (15), K."""
    if td >= t:
        return t
    return 1.0 / (1.0 / (td - 56.0) + math.log(t / td) / 800.0) + 56.0


def theta_e_exponent(r, tl):
    """exp of the latent term of Bolton (1980) eq. (43)."""
    return math.exp(r * (1.0 + 0.81 * r) * (3376.0 / tl - 2.54))


def theta_e(t, td, p):
    """Equivalent potential temperature, eq. (43), K."""
    e = esat(td)
    r = mixing_ratio(e, p)
    tl = lcl_temperature(t, td)
    return t * (P_REF / p) ** kappa(r) * theta_e_exponent(r, tl)


def virtual(t, r, use):
    return t * (1.0 + r / EPS) / (1.0 + r) if use else t


def _interp_lnp(p, f, pt):
    """Linear in ln p between the two levels that bracket ``pt``."""
    n = len(p)
    for k in range(n - 1):
        if p[k] >= pt >= p[k + 1]:
            w = math.log(pt / p[k]) / math.log(p[k + 1] / p[k])
            return f[k] + w * (f[k + 1] - f[k])
    return NAN


def _parcel_start(kind, depth, z, p, t, td):
    """Pressure, temperature, dew point and index of the starting parcel."""
    if kind == 0:
        return p[0], t[0], td[0], 0
    top = p[0] - depth
    if kind == 2:
        best, arg = -1.0, 0
        for k in range(len(p)):
            if p[k] < top:
                break
            th = theta_e(t[k], td[k], p[k])
            if th > best:
                best, arg = th, k
        return p[arg], t[arg], td[arg], arg
    # mixed layer: pressure-weighted mean of the potential temperature and the
    # mixing ratio over the lowest ``depth`` Pa (trapezoid in p)
    pp, th, rr = [], [], []
    for k in range(len(p)):
        if p[k] < top:
            break
        r = mixing_ratio(esat(td[k]), p[k])
        pp.append(p[k])
        rr.append(r)
        th.append(t[k] * (P_REF / p[k]) ** kappa(r))
    if pp[-1] > top and len(pp) < len(p):  # the layer top lies between levels
        k = len(pp)
        w = math.log(top / p[k - 1]) / math.log(p[k] / p[k - 1])
        r = mixing_ratio(esat(td[k - 1]), p[k - 1])
        r2 = mixing_ratio(esat(td[k]), p[k])
        t_top = t[k - 1] + w * (t[k] - t[k - 1])
        r_top = r + w * (r2 - r)
        pp.append(top)
        rr.append(r_top)
        th.append(t_top * (P_REF / top) ** kappa(r_top))
    if len(pp) < 2:
        return p[0], t[0], td[0], 0
    num_t = num_r = 0.0
    for k in range(len(pp) - 1):
        dp = pp[k] - pp[k + 1]
        num_t += 0.5 * (th[k] + th[k + 1]) * dp
        num_r += 0.5 * (rr[k] + rr[k + 1]) * dp
    span = pp[0] - pp[-1]
    th_m, r_m = num_t / span, num_r / span
    p0 = p[0]
    t0 = th_m * (p0 / P_REF) ** kappa(r_m)
    e0 = r_m * p0 / (EPS + r_m)
    return p0, t0, min(dewpoint(e0), t0), 0


class _Parcel:
    """A parcel lifted from (p0, t0, td0): dry adiabat, then the Bolton pseudo-adiabat."""

    def __init__(self, p0, t0, td0):
        self.p0, self.t0 = p0, t0
        self.r0 = mixing_ratio(esat(td0), p0)
        self.kappa0 = kappa(self.r0)
        self.theta0 = t0 * (P_REF / p0) ** self.kappa0
        self.tl = lcl_temperature(t0, td0)
        self.pl = p0 * (self.tl / t0) ** (1.0 / self.kappa0)
        self.theta_e0 = self.theta0 * theta_e_exponent(self.r0, self.tl)

    def temperature(self, p):
        """Parcel temperature and mixing ratio at pressure ``p``."""
        if p >= self.pl:
            return self.theta0 * (p / P_REF) ** self.kappa0, self.r0

        def g(t):
            es = esat(t)
            r = mixing_ratio(es, p)
            return t * (P_REF / p) ** kappa(r) * theta_e_exponent(r, t) - self.theta_e0

        lo, hi = 100.0, self.tl + 1.0
        for _ in range(_BISECT):
            mid = 0.5 * (lo + hi)
            if g(mid) > 0.0:
                hi = mid
            else:
                lo = mid
        t = 0.5 * (lo + hi)
        return t, mixing_ratio(esat(t), p)


def _positive_area(z, b):
    """Integral of max(b, 0) dz for b linear between the nodes."""
    total = 0.0
    for k in range(len(z) - 1):
        a, c, dz = b[k], b[k + 1], z[k + 1] - z[k]
        if a >= 0.0 and c >= 0.0:
            total += 0.5 * (a + c) * dz
        elif a > 0.0 > c:
            total += 0.5 * a * dz * a / (a - c)
        elif c > 0.0 > a:
            total += 0.5 * c * dz * c / (c - a)
    return total


def _negative_area(z, b):
    return -_positive_area(z, [-v for v in b])


def parcel_column(z, p, t, td, kind, depth, use_virtual):
    out = [NAN] * 12
    keep_z, keep_p, keep_t, keep_td = [], [], [], []
    for zk, pk, tk, dk in zip(z, p, t, td):
        if not (math.isfinite(zk) and math.isfinite(pk) and math.isfinite(tk)):
            continue
        if not math.isfinite(dk) or pk <= 0.0 or tk <= 0.0 or dk <= 0.0:
            continue
        if keep_p and not (pk < keep_p[-1] and zk > keep_z[-1]):
            continue
        keep_z.append(zk)
        keep_p.append(pk)
        keep_t.append(tk)
        keep_td.append(min(dk, tk))
    n = len(keep_p)
    if n < 3:
        return out
    z, p, t, td = keep_z, keep_p, keep_t, keep_td

    p0, t0, td0, i0 = _parcel_start(kind, depth, z, p, t, td)
    parcel = _Parcel(p0, t0, td0)
    out[9], out[10], out[11] = p0, t0, td0
    r_env = [mixing_ratio(esat(d), pk) for d, pk in zip(td, p)]

    def buoyancy(tp, rp, te, re):
        tvp, tve = virtual(tp, rp, use_virtual), virtual(te, re, use_virtual)
        return G0 * (tvp - tve) / tve

    # lifted index: environment minus parcel at 500 hPa
    if p[-1] <= LI_PRESSURE <= p0:
        te = _interp_lnp(p, t, LI_PRESSURE)
        out[2] = te - parcel.temperature(LI_PRESSURE)[0]

    # nodes: the start, the LCL and the environment levels above the start
    nodes = []  # (z, p, B)
    start_re = r_env[i0]
    nodes.append((z[i0], p0, buoyancy(t0, parcel.r0, t[i0], start_re)))
    lcl_node = -1
    have_lcl = parcel.pl < p0 and parcel.pl >= p[-1]
    if parcel.pl >= p0:  # saturated at the start
        have_lcl = True
        lcl_node = 0
    lcl_z = lcl_p = NAN
    if have_lcl and lcl_node == 0:
        lcl_z, lcl_p = z[i0], p0
    for k in range(i0 + 1, n):
        if have_lcl and lcl_node < 0 and p[k] <= parcel.pl:
            pl = parcel.pl
            lcl_p = pl
            lcl_z = _interp_lnp(p, z, pl)
            te = _interp_lnp(p, t, pl)
            re = _interp_lnp(p, r_env, pl)
            if p[k] < pl:  # a level exactly at the LCL is the node itself
                nodes.append((lcl_z, pl, buoyancy(parcel.tl, parcel.r0, te, re)))
                lcl_node = len(nodes) - 1
        tp, rp = parcel.temperature(p[k])
        nodes.append((z[k], p[k], buoyancy(tp, rp, t[k], r_env[k])))
        if lcl_node < 0 and have_lcl and p[k] == parcel.pl:
            lcl_node = len(nodes) - 1
    out[3], out[4] = lcl_p, lcl_z
    if lcl_node < 0:
        out[0] = 0.0
        return out

    nz = [v[0] for v in nodes]
    npr = [v[1] for v in nodes]
    nb = [v[2] for v in nodes]

    def p_at(zc, j):
        """Pressure at height zc in the segment j, j+1 (linear in z of ln p)."""
        w = (zc - nz[j]) / (nz[j + 1] - nz[j])
        return npr[j] * (npr[j + 1] / npr[j]) ** w

    # level of free convection: the first rise of the buoyancy above zero at or
    # above the LCL
    lfc_z = lfc_p = NAN
    lfc_b = 0.0
    lfc_seg = -1
    if nb[lcl_node] > 0.0:
        lfc_z, lfc_p, lfc_b, lfc_seg = (
            nz[lcl_node],
            npr[lcl_node],
            nb[lcl_node],
            lcl_node,
        )
    else:
        for j in range(lcl_node, len(nodes) - 1):
            if nb[j] <= 0.0 < nb[j + 1]:
                zc = nz[j] + (nz[j + 1] - nz[j]) * (0.0 - nb[j]) / (nb[j + 1] - nb[j])
                lfc_z, lfc_p, lfc_seg = zc, p_at(zc, j), j
                break
    if lfc_seg < 0:
        out[0] = 0.0
        return out
    out[5], out[6] = lfc_p, lfc_z

    # equilibrium level: the last fall of the buoyancy to zero above the LFC, or
    # the top of the profile if the parcel is still buoyant there
    el_z, el_p, el_b = nz[-1], npr[-1], nb[-1]
    for j in range(len(nodes) - 2, lfc_seg - 1, -1):
        if nb[j] > 0.0 >= nb[j + 1]:
            zc = nz[j] + (nz[j + 1] - nz[j]) * (0.0 - nb[j]) / (nb[j + 1] - nb[j])
            el_z, el_p, el_b = zc, p_at(zc, j), 0.0
            break
    if nb[-1] > 0.0:
        el_z, el_p, el_b = nz[-1], npr[-1], nb[-1]
    out[7], out[8] = el_p, el_z

    # CAPE: the positive buoyancy between the LFC and the EL
    cz = [lfc_z] + [v for v in nz if lfc_z < v < el_z] + [el_z]
    cb = [lfc_b]
    for v, bv in zip(nz, nb):
        if lfc_z < v < el_z:
            cb.append(bv)
    cb.append(el_b)
    out[0] = _positive_area(cz, cb)
    # CIN: the negative buoyancy from the start to the LFC
    iz = [v for v in nz if v < lfc_z] + [lfc_z]
    ib = [bv for v, bv in zip(nz, nb) if v < lfc_z] + [lfc_b]
    out[1] = _negative_area(iz, ib)
    return out


def parcel(z, p, t, td, kind, depth, use_virtual, n_threads=0):
    """Diagnostics (12 rows, see ``ROWS``) of every column of (ncol, nlev) arrays."""
    z, p, t, td = (np.asarray(a, dtype=np.float64) for a in (z, p, t, td))
    out = np.full((12, z.shape[0]), np.nan)
    for c in range(z.shape[0]):
        out[:, c] = parcel_column(
            z[c].tolist(),
            p[c].tolist(),
            t[c].tolist(),
            td[c].tolist(),
            int(kind),
            float(depth),
            bool(use_virtual),
        )
    return out
