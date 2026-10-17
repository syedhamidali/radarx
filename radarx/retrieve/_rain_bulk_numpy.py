# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
NumPy reference of the two-moment rain shaft of :mod:`radarx.retrieve.rain_bulk`.

The compiled kernel ``_rain_bulk`` does the same steps in the same order; this
module is the fallback and the test oracle. Everything is in SI units (m,
kg, s) except where a name says otherwise.
"""

from __future__ import annotations

import math

import numpy as np

A_V = 9.65  # m s-1, Eq. (A4) of Seifert (2008)
B_V = 9.8  # m s-1
C_V = 600.0  # m-1
K_RR = 4.33  # m3 kg-1 s-1, Eq. (A12)
K_BR = 1.0e3  # m-1 (1 mm-1), Eq. (A13)
D_EQ = 1.1e-3  # m, equilibrium mean volume diameter
D_BR_MIN = 0.3e-3  # m, no breakup of smaller drops
C_1 = 4.0e3  # m-1, Eq. (20)
C_2 = 1.0e3  # m-1, Eq. (20)
RHO_W = 1000.0
DM_LOW = 5.0e-5  # m, bracket of the mean volume diameter
DM_HIGH = 2.5e-2  # m
BISECTIONS = 60


def _shape_scalar(dm):
    t = dm - D_EQ
    if t <= 0.0:
        return 6.0 * math.tanh(C_1 * t) ** 2 + 1.0
    return 30.0 * math.tanh(C_2 * t) ** 2 + 1.0


def _sedimentation_scalar(dm):
    """:func:`sedimentation` for one diameter, without the array overhead."""
    mu = _shape_scalar(dm)
    lam = ((mu + 3.0) * (mu + 2.0) * (mu + 1.0)) ** (1.0 / 3.0) / dm
    q = 1.0 + C_V / lam
    return A_V - B_V * q ** (-(mu + 1.0)), A_V - B_V * q ** (-(mu + 4.0))


def shape_parameter(dm):
    """Gamma shape parameter from the mean volume diameter (m), Eq. (20)."""
    t = np.asarray(dm, dtype=np.float64) - D_EQ
    return np.where(
        t <= 0.0, 6.0 * np.tanh(C_1 * t) ** 2 + 1.0, 30.0 * np.tanh(C_2 * t) ** 2 + 1.0
    )


def slope(dm, mu):
    """Slope (m-1) of the gamma distribution, Eq. (19)."""
    return np.cbrt((mu + 3.0) * (mu + 2.0) * (mu + 1.0)) / dm


def sedimentation(dm):
    """Number- and mass-weighted sea-level fall speeds (m s-1), Eq. (A10)."""
    mu = shape_parameter(dm)
    lam = slope(dm, mu)
    q = 1.0 + C_V / lam
    return A_V - B_V * q ** (-(mu + 1.0)), A_V - B_V * q ** (-(mu + 4.0))


def mass_of_mean_drop(dm):
    """Mass (kg) of the drop of the mean volume diameter ``dm`` (m)."""
    return np.pi / 6.0 * RHO_W * dm**3


def flux_ratio(dm):
    """``F_N / F_L`` (kg-1) of the shaft with the mean volume diameter ``dm``."""
    vn, vl = sedimentation(dm)
    return vn / vl / mass_of_mean_drop(dm)


def mean_diameter_of_flux_ratio(ratio):
    """
    Mean volume diameter (m) whose flux ratio is ``ratio``; bisection in the
    logarithm of the diameter, clamped to the bracket. The ratio decreases
    with the diameter over the bracket.
    """
    lo, hi = math.log(DM_LOW), math.log(DM_HIGH)
    for _ in range(BISECTIONS):
        mid = 0.5 * (lo + hi)
        dm = math.exp(mid)
        vn, vl = _sedimentation_scalar(dm)
        if vn / vl / (math.pi / 6.0 * RHO_W * dm**3) > ratio:
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))


def concentrations(flux_n, flux_l, s):
    """``(N, L, Dm)`` from the two fluxes (m-2 s-1, kg m-2 s-1) at the factor ``s``."""
    dm = mean_diameter_of_flux_ratio(flux_n / flux_l)
    vl = _sedimentation_scalar(dm)[1]
    water = flux_l / (s * vl)
    return water / (math.pi / 6.0 * RHO_W * dm**3), water, dm


def source(n, water, dm):
    """Net change of the number concentration (m-3 s-1) by self-collection and breakup."""
    sc = -K_RR * n * water
    if dm > D_BR_MIN:
        return sc - (K_BR * (dm - D_EQ) + 1.0) * sc
    return sc


def factor_at(zs, ss, z):
    return float(np.interp(z, zs, ss))


def evolve(n0, water0, z_out, dz_max, cfl, zs, ss, n_threads=0):
    """
    Evolve columns of rain falling in still air with the two-moment scheme.

    Steady state of ``d(s V_N N)/dz = s dN/dt|collisions`` with the water
    flux ``s V_L L`` constant, in the distance ``z`` below the top of the
    shaft, integrated with the classical Runge-Kutta method in the flux of
    the number of drops with steps of at most ``dz_max`` and at most ``cfl``
    times the distance over which the collisions change the number.

    Parameters
    ----------
    n0, water0 : ndarray (ncol,)
        Number concentration (m-3) and rainwater content (kg m-3) at the top.
    z_out : ndarray (nout,)
        Ascending distances below the top (m).
    dz_max, cfl : float
        Longest step (m) and its share of the collision distance.
    zs, ss : ndarray
        Distances (m, ascending) and the factor ``s`` there; linear between,
        constant outside.
    n_threads : int
        Ignored.

    Returns
    -------
    ndarray (nout, ncol, 3)
        Number concentration (m-3), rainwater content (kg m-3) and mean volume
        diameter (m) at the distances; NaN for a column whose input is not a
        positive finite pair or whose mean volume diameter at the top is
        outside the bracket.
    """
    n0 = np.atleast_1d(np.asarray(n0, dtype=np.float64))
    water0 = np.atleast_1d(np.asarray(water0, dtype=np.float64))
    z_out = np.asarray(z_out, dtype=np.float64)
    out = np.full((z_out.size, n0.size, 3), np.nan)
    s0 = factor_at(zs, ss, 0.0)
    for c in range(n0.size):
        n, w = n0[c], water0[c]
        if not (np.isfinite(n) and np.isfinite(w) and n > 0.0 and w > 0.0):
            continue
        dm0 = (w / n / (math.pi / 6.0 * RHO_W)) ** (1.0 / 3.0)
        if not (DM_LOW <= dm0 <= DM_HIGH):
            continue
        vn, vl = _sedimentation_scalar(dm0)
        flux_n = s0 * vn * n
        flux_l = s0 * vl * w
        z = 0.0

        def rhs(f, zz):
            s = factor_at(zs, ss, zz)
            nn, ww, dm = concentrations(f, flux_l, s)
            return s * source(nn, ww, dm)

        for q, target in enumerate(z_out):
            while z < target:
                s = factor_at(zs, ss, z)
                nn, ww, dm = concentrations(flux_n, flux_l, s)
                vn_now = _sedimentation_scalar(dm)[0]
                dz = min(dz_max, target - z)
                rate = K_RR * ww
                if rate > 0.0:
                    dz = min(dz, cfl * s * vn_now / rate)
                k1 = rhs(flux_n, z)
                k2 = rhs(flux_n + 0.5 * dz * k1, z + 0.5 * dz)
                k3 = rhs(flux_n + 0.5 * dz * k2, z + 0.5 * dz)
                k4 = rhs(flux_n + dz * k3, z + dz)
                flux_n = flux_n + dz / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                z = z + dz if z + dz < target else target
            sz = factor_at(zs, ss, z)
            out[q, c] = concentrations(flux_n, flux_l, sz)
    return out
