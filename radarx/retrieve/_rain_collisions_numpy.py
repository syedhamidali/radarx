# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
NumPy reference of the raindrop collision kernel ``_rain_collisions.cpp``.

Every operation follows the compiled kernel in the same order. It is the
fallback without the compiled kernel and the oracle of the tests. The model
(Low and List 1982 coalescence efficiency and fragment distributions, a
mass-conserving closure, fixed pivots) is described in
:mod:`radarx.retrieve.rain_collisions`.

Lengths of the parameterisation are in cm and energies in J, as in the paper;
the diameters of the interface are in mm.
"""

from __future__ import annotations

import math

import numpy as np

SIGMA = 7.28e-2  # N m-1, surface tension of Low and List (1982)
RHO_W = 1000.0  # kg m-3
D0 = 0.01  # cm, smallest fragment of the log-normal distributions (LL82, Part II, p. 1611)
CKE0 = 0.893e-6  # J, filament breakup only below (LL82 Part II Fig. 1)
W0 = 0.86  # sheet breakup only above this Weber number (LL82 Part II Eq. 4.44)
E_T_MAX = 5.0e-6  # J, no coalescence above this total energy (LL82 Part I Eq. 4.6)
X_MIN = 0.1  # smallest share of the mass that the large remnant must carry
SIGMA_MAX = 8.0  # largest log-normal spread searched
NONE, COALESCENCE, LOW_LIST = 0, 1, 2


# --------------------------------------------------------------------------
# Low and List (1982)
# --------------------------------------------------------------------------


def energies(dl, ds, vl, vs):
    """
    Collision kinetic energy, surface energies and total energy (J).

    ``dl`` and ``ds`` are the diameters of the large and small drop in cm,
    ``vl`` and ``vs`` their fall speeds (m s-1). CKE is Eq. (3.1) of Part I,
    ``S_T`` Eq. (4.2), ``S_C`` Eq. (4.3), ``dS = S_T - S_C`` Eq. (4.1).
    """
    dl_m, ds_m = dl * 1e-2, ds * 1e-2
    cke = (
        (RHO_W * math.pi / 12.0)
        * dl_m**3
        * ds_m**3
        / (dl_m**3 + ds_m**3)
        * (vl - vs) ** 2
    )
    st = math.pi * SIGMA * (dl_m**2 + ds_m**2)
    sc = math.pi * SIGMA * (dl_m**3 + ds_m**3) ** (2.0 / 3.0)
    return cke, st, sc, st - sc


def coalescence_efficiency(dl, ds, cke, sc, dsig):
    """Eqs. (4.5) and (4.6) of Part I."""
    et = cke + dsig
    if et >= E_T_MAX:
        return 0.0
    return 0.778 * (1.0 + ds / dl) ** -2 * math.exp(-2.61e6 * SIGMA * et * et / sc)


def breakup_fractions_raw(cke, st):
    """Fractions of filament, sheet and disk breakups as printed (Part II, Eqs. 4.40-4.46)."""
    rf = 1.0 if cke < CKE0 else 1.11e-4 * cke**-0.654
    w2 = cke / st
    rs = 0.685 * (1.0 - math.exp(-1.63 * (w2 - W0))) if w2 >= W0 else 0.0
    rd = 1.0 - (rf + rs) if rf + rs <= 1.0 else 0.0
    return rf, rs, rd


def breakup_fractions(cke, st):
    """
    The fractions of the three breakup types, rescaled to a sum of one.

    The printed fractions add up to more than one where :math:`R_f + R_s > 1`
    (1.18 for the drop pair of 1.8 and 0.715 mm in Table 2 of Part II); every
    event has to produce the mass of the two drops once.
    """
    rf, rs, rd = breakup_fractions_raw(cke, st)
    total = rf + rs + rd
    return rf / total, rs / total, rd / total


def _filament_numbers(dl, ds):
    f1 = (
        (-2.25e4 * (dl - 0.403) ** 2 - 37.9) * ds**2.5 + 9.67 * (dl + 0.170) ** 2 + 4.95
    )
    f2 = 1.02e4 * ds**2.83 + 2.0
    return f1, f2


def mean_numbers(dl, ds, cke, st):
    """
    Mean number of fragments of filament, sheet and disk breakups.

    Eqs. (3.3), (3.4), (3.7) and (3.9) of Part II: the filament number is
    the smaller of the two branches ``F_f1`` and ``F_f2`` (they cross at
    ``D_S0``), never below 2; the sheet number is never below 2 and the disk
    number never below 2.
    """
    f1, f2 = _filament_numbers(dl, ds)
    ff = max(2.0, min(f1, f2))
    fs = max(2.0, 5.0 * math.erf((st - 2.53e-6) / 1.85e-6) + 6.0)
    fd = max(2.0, 297.5 + 23.7 * math.log(cke)) if cke > 0.0 else 2.0
    return ff, fs, fd


def _crossing(dl):
    """Small-drop size ``D_S0`` (cm) where the two filament branches cross."""
    lo, hi = 1e-5, 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f1, f2 = _filament_numbers(dl, mid)
        if f1 > f2:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _lognormal_sigma(mode, peak, number):
    """
    Spread of the log-normal distribution of Eq. (4.15) with a given mode and
    peak value whose area above ``D0`` is ``number`` (Eqs. 4.22-4.23, solved
    by bisection). NaN if there is none.
    """
    if not (number > 0.0 and peak > 0.0 and mode > 0.0):
        return math.nan
    lmode, ld0 = math.log(mode), math.log(D0)

    def area(sig):
        h = peak * mode * math.exp(0.5 * sig * sig)
        mu = lmode + sig * sig
        return (
            h
            * sig
            * math.sqrt(math.pi / 2.0)
            * (1.0 - math.erf((ld0 - mu) / (math.sqrt(2.0) * sig)))
        )

    lo, hi = 1e-3, SIGMA_MAX
    if not (area(lo) < number < area(hi)):
        return math.nan
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if area(mid) < number:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _gauss_moments(mu, sig, a, b, h):
    """Number and ``integral D^3 P`` of ``h exp(-0.5 ((D - mu)/sig)^2)`` on ``[a, b]``."""
    if not b > a:
        return 0.0, 0.0
    s2 = math.sqrt(2.0) * sig
    za, zb = (a - mu) / s2, (b - mu) / s2
    ea, eb = math.exp(-za * za), math.exp(-zb * zb)
    i0 = math.sqrt(math.pi / 2.0) * sig * (math.erf(zb) - math.erf(za))
    i1 = mu * i0 - sig * sig * (eb - ea)
    i2 = mu * i1 + sig * sig * (i0 - (b * eb - a * ea))
    i3 = mu * i2 + sig * sig * (2.0 * i1 - (b * b * eb - a * a * ea))
    return h * i0, h * i3


def _lognormal_moments(mu, sig, a, b, h):
    """Number and ``integral D^3 P`` of ``(h/D) exp(-0.5 ((ln D - mu)/sig)^2)`` on ``[a, b]``."""
    if not b > a:
        return 0.0, 0.0
    s2 = math.sqrt(2.0) * sig
    la, lb = math.log(a), math.log(b)
    n0 = (
        h
        * sig
        * math.sqrt(math.pi / 2.0)
        * (math.erf((lb - mu) / s2) - math.erf((la - mu) / s2))
    )
    shift = mu + 3.0 * sig * sig
    n3 = (
        h
        * math.exp(3.0 * mu + 4.5 * sig * sig)
        * sig
        * math.sqrt(math.pi / 2.0)
        * (math.erf((lb - shift) / s2) - math.erf((la - shift) / s2))
    )
    return n0, n3


def _components(dl, ds, cke):
    """
    Small-fragment components of the three breakup types of a collision.

    Returns, for ``"f"``, ``"s"`` and ``"d"``, a list of components
    ``(kind, mu, sigma, height, lower, upper)`` (``"g"`` Gaussian, ``"l"``
    log-normal; cm) and a flag that says whether the log-normal part could be
    constructed. The large-drop remnant (Gaussians ``P_f1``, ``P_s1``,
    ``P_d1`` of the paper) is not a component: its size follows from the
    conservation of mass.
    """
    d_coal = (dl**3 + ds**3) ** (1.0 / 3.0)
    st = math.pi * SIGMA * ((dl * 1e-2) ** 2 + (ds * 1e-2) ** 2)
    ff, fs, fd = mean_numbers(dl, ds, cke, st)
    out = {}
    # filament: the Gaussian small-drop remnant (Eqs. 4.12, 4.13) and the log-normal bridge
    hf2 = 4.18 * ds**-1.17
    parts = [("g", ds, 1.0 / (math.sqrt(2.0 * math.pi) * hf2), hf2, 0.0, d_coal)]
    ds0 = _crossing(dl)
    mode = 0.241 * ds + 0.0129  # Eq. (4.14)
    p1 = 1.68e5 * ds**2.33  # Eq. (4.18)
    p2 = (
        (43.4 * (dl + 1.81) ** 2 - 159.0) / ds - 3870.0 * (dl - 0.285) ** 2 - 58.1
    )  # Eq. (4.19)
    if ds <= ds0:
        peak = p1
    elif ds >= 1.2 * ds0:
        peak = p2
    else:
        alpha = (ds - ds0) / (0.2 * ds0)  # Eq. (4.21), see the module documentation
        peak = (1.0 - alpha) * p1 + alpha * p2
    if not peak > 0.0:
        peak = p1
    ok = True
    if ff - 2.0 > 1e-9:
        sig = _lognormal_sigma(mode, peak, ff - 2.0)
        if sig == sig:
            parts.append(
                (
                    "l",
                    math.log(mode) + sig * sig,
                    sig,
                    peak * mode * math.exp(0.5 * sig * sig),
                    D0,
                    d_coal,
                )
            )
    out["f"] = (parts, ok)
    # sheet: cloud-droplet part, Eqs. (4.27)-(4.31)
    mode_s = 0.254 * ds**0.413 * math.exp(3.53 * ds**2.51 * (dl - ds))
    peak_s = 0.23 * ds**-3.93 * dl ** (14.2 * math.exp(-17.2 * ds))
    sig = _lognormal_sigma(mode_s, peak_s, fs - 1.0)
    parts, ok = [], sig == sig
    if ok:
        parts.append(
            (
                "l",
                math.log(mode_s) + sig * sig,
                sig,
                peak_s * mode_s * math.exp(0.5 * sig * sig),
                D0,
                d_coal,
            )
        )
    out["s"] = (parts, ok)
    # disk: cloud-droplet part, Eqs. (4.37)-(4.39)
    mode_d = math.exp(-17.4 * ds - 0.671 * (dl - ds)) * ds
    peak_d = 0.0884 * ds**-2.52 * (dl - ds) ** (0.007 * ds**-2.54)
    sig = _lognormal_sigma(mode_d, peak_d, fd - 1.0)
    parts, ok = [], sig == sig
    if ok:
        parts.append(
            (
                "l",
                math.log(mode_d) + sig * sig,
                sig,
                peak_d * mode_d * math.exp(0.5 * sig * sig),
                D0,
                d_coal,
            )
        )
    out["d"] = (parts, ok)
    return out


# --------------------------------------------------------------------------
# fixed pivots
# --------------------------------------------------------------------------


def edges_cm(diam_mm):
    """Edges (cm) of the size bins of the pivots: geometric means, extrapolated at the ends."""
    d = np.asarray(diam_mm, dtype=np.float64) * 0.1
    inner = np.sqrt(d[:-1] * d[1:])
    first = d[0] * d[0] / inner[0]
    last = d[-1] * d[-1] / inner[-1]
    return np.concatenate([[first], inner, [last]])


def assign(x, number, cube, out):
    """
    Add ``number`` particles with ``D^3 = x`` to the pivots (``cube`` = ``D^3``).

    Between two pivots the number and ``D^3`` are conserved; beyond the ends
    the ``D^3`` is conserved and the number follows.
    """
    if x <= cube[0]:
        out[0] += number * x / cube[0]
    elif x >= cube[-1]:
        out[-1] += number * x / cube[-1]
    else:
        k = int(np.searchsorted(cube, x, side="right")) - 1
        t = (x - cube[k]) / (cube[k + 1] - cube[k])
        out[k] += number * (1.0 - t)
        out[k + 1] += number * t
    return out


def _bin_component(comp, edges, cube, out):
    """Distribute a component over the pivots; returns the ``D^3`` it carries."""
    kind, mu, sig, h, lo, hi = comp
    carried = 0.0
    n = cube.size
    for k in range(n):
        a = lo if k == 0 else max(lo, edges[k])
        b = hi if k == n - 1 else min(hi, edges[k + 1])
        a = max(a, lo)
        b = min(b, hi)
        if kind == "g":
            num, x3 = _gauss_moments(mu, sig, a, b, h)
        else:
            num, x3 = _lognormal_moments(mu, sig, a, b, h)
        if num > 0.0 and x3 > 0.0:
            assign(x3 / num, num, cube, out)
            carried += x3
    return carried


def remnants(dl, ds, cke):
    """
    Large remnant of each breakup type of a collision (diagnostic).

    Returns ``{"f": (D, ok), "s": (D, ok), "d": (D, ok)}``: the diameter (cm)
    of the drop that carries the mass the small fragments do not (with the
    components integrated exactly, not on the pivots) and whether the
    distribution of the type could be built and the remnant carries at least
    ``X_MIN`` of the mass.
    """
    x0 = dl**3 + ds**3
    out = {}
    for t, (parts, ok) in _components(dl, ds, cke).items():
        carried = 0.0
        for kind, mu, sig, h, lo, hi in parts:
            moments = _gauss_moments if kind == "g" else _lognormal_moments
            carried += moments(mu, sig, lo, hi, h)[1]
        remnant = x0 - carried
        out[t] = (max(remnant, 0.0) ** (1.0 / 3.0), bool(ok and remnant >= X_MIN * x0))
    return out


def pair_outcome(dl, ds, vl, vs, edges, cube):
    """
    Average outcome of the collision of a large and a small drop (cm, m s-1).

    Returns ``(efficiency, breakup, coalesced)``: the coalescence efficiency,
    the mean number of drops per breakup event at the pivots (the mass of
    ``dl**3 + ds**3`` is conserved) and the coalesced drop at the pivots.
    """
    cke, st, sc, dsig = energies(dl, ds, vl, vs)
    eff = coalescence_efficiency(dl, ds, cke, sc, dsig)
    x0 = dl**3 + ds**3
    rf, rs, rd = breakup_fractions(cke, st)
    comps = _components(dl, ds, cke)
    n = cube.size
    brk = np.zeros(n)
    weight = {"f": rf, "s": rs, "d": rd}
    types = {}
    for t in ("f", "s", "d"):
        parts, ok = comps[t]
        out = np.zeros(n)
        carried = 0.0
        for comp in parts:
            carried += _bin_component(comp, edges, cube, out)
        remnant = x0 - carried
        if t != "f" and (not ok or remnant < X_MIN * x0):
            weight["f"] += weight[t]
            weight[t] = 0.0
            continue
        assign(remnant, 1.0, cube, out)
        types[t] = out
    for t, out in types.items():
        brk += weight[t] * out
    coal = np.zeros(n)
    assign(x0, 1.0, cube, coal)
    return eff, brk, coal


# --------------------------------------------------------------------------
# kernel tensors and the evolution of a shaft
# --------------------------------------------------------------------------


def build(diam_mm, v0, mode, n_threads=0):
    """
    Collision kernel and event tensor of a set of pivot diameters.

    Parameters
    ----------
    diam_mm, v0 : array of float
        Pivot diameters (mm, ascending) and sea-level fall speeds (m s-1).
    mode : int
        ``0`` no interactions, ``1`` every collision coalesces, ``2`` the
        coalescence efficiency and breakup of Low and List (1982).
    n_threads : int
        Ignored (the signature of the compiled kernel).

    Returns
    -------
    kernel : ndarray (n, n)
        ``pi/4 (D_i + D_j)^2 |V_i - V_j|`` in m3 s-1.
    gain : ndarray (n, n, n)
        ``gain[i, j, k]`` for ``i > j``: change of the number of drops at
        pivot ``k`` per collision of a drop of pivot ``i`` with one of pivot
        ``j``, losses included. Zero where ``i <= j``.
    efficiency : ndarray (n, n)
        Coalescence efficiency of the pairs.
    """
    d = np.asarray(diam_mm, dtype=np.float64)
    v = np.asarray(v0, dtype=np.float64)
    n = d.size
    dm = d * 1e-3
    kernel = (
        0.25
        * math.pi
        * (dm[:, None] + dm[None, :]) ** 2
        * np.abs(v[:, None] - v[None, :])
    )
    gain = np.zeros((n, n, n))
    eff = np.zeros((n, n))
    if mode == NONE:
        return kernel, gain, eff
    dcm = d * 0.1
    cube = dcm**3
    edges = edges_cm(d)
    for i in range(1, n):
        for j in range(i):
            if not kernel[i, j] > 0.0:
                continue
            if mode == COALESCENCE:
                e = 1.0
                coal = np.zeros(n)
                assign(cube[i] + cube[j], 1.0, cube, coal)
                g = coal
            else:
                e, brk, coal = pair_outcome(dcm[i], dcm[j], v[i], v[j], edges, cube)
                g = (1.0 - e) * brk + e * coal
            g = g.copy()
            g[i] -= 1.0
            g[j] -= 1.0
            gain[i, j] = g
            eff[i, j] = e
    return kernel, gain, eff


def _rates(kernel, gain, conc, scale):
    """Net rate of change (m-3 s-1) of the number concentrations."""
    rate = scale * np.tril(kernel, -1) * conc[:, None] * conc[None, :]
    return np.einsum("ij,ijk->k", rate, gain)


def evolve(kernel, gain, v0, n0, z_out, dz_max, cfl, zs, ss, n_threads=0):
    """
    Evolve the number concentrations of columns of rain falling in still air.

    Steady state of ``d(V N_k)/dz = dN_k/dt|coll``, with ``V = V0 s(z)``, where
    ``s`` is the air density correction at the distance ``z`` below the top of
    the shaft, integrated with the classical Runge-Kutta method in ``z`` with
    steps of at most ``dz_max`` and at most ``cfl`` of the distance in which a
    drop collides with another.

    Parameters
    ----------
    kernel, gain : ndarray
        From :func:`build`.
    v0 : ndarray (n,)
        Sea-level fall speeds (m s-1), all positive.
    n0 : ndarray (ncol, n)
        Number concentrations (m-3) at the top.
    z_out : ndarray (nout,)
        Ascending distances below the top (m) where the state is wanted.
    dz_max, cfl : float
        Longest step (m) and the share of the collision distance.
    zs, ss : ndarray
        Distances below the top (m, ascending) and the factor ``s`` there;
        linear in between, constant outside.
    n_threads : int
        Ignored.

    Returns
    -------
    ndarray (nout, ncol, n)
        Number concentrations at the distances ``z_out``.
    """
    n0 = np.atleast_2d(np.asarray(n0, dtype=np.float64))
    z_out = np.asarray(z_out, dtype=np.float64)
    out = np.empty((z_out.size, n0.shape[0], n0.shape[1]))
    for c in range(n0.shape[0]):
        z = 0.0
        flux = n0[c] * v0 * np.interp(0.0, zs, ss)
        for q, target in enumerate(z_out):
            while z < target:
                s = float(np.interp(z, zs, ss))
                conc = flux / (v0 * s)
                # collisions per unit fall distance of the busiest size (independent of s)
                lam = float(np.max((kernel @ conc) / v0))
                dz = min(dz_max, target - z)
                if lam > 0.0:
                    dz = min(dz, cfl / lam)
                flux = _rk4(kernel, gain, v0, flux, z, dz, zs, ss)
                z = z + dz if z + dz < target else target
            out[q, c] = flux / (v0 * float(np.interp(z, zs, ss)))
    return out


def _deriv(kernel, gain, v0, flux, z, zs, ss):
    s = float(np.interp(z, zs, ss))
    conc = flux / (v0 * s)
    return _rates(kernel, gain, conc, s)


def _rk4(kernel, gain, v0, flux, z, dz, zs, ss):
    k1 = _deriv(kernel, gain, v0, flux, z, zs, ss)
    k2 = _deriv(kernel, gain, v0, flux + 0.5 * dz * k1, z + 0.5 * dz, zs, ss)
    k3 = _deriv(kernel, gain, v0, flux + 0.5 * dz * k2, z + 0.5 * dz, zs, ss)
    k4 = _deriv(kernel, gain, v0, flux + dz * k3, z + dz, zs, ss)
    return flux + dz / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
