#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Rain rate and radar variables of simulated drop size distributions.

Every sample is a normalized-gamma drop size distribution (Testud et al.
2001) with the mass-weighted mean diameter ``Dm``, the normalized intercept
``Nw`` and the shape ``mu``. The radar variables (Z_H, Z_DR, K_DP, rho_hv)
are integrals over the distribution of the single-drop T-matrix tables of
radarx (``radarx.retrieve.dsd``; drop shape of Brandes et al. 2002, 7 degrees
of canting), and the rain rate is the third-moment integral with the
fall speed of Atlas et al. (1973), as in ``radarx.retrieve.radar_from_dsd``.
Z_H, K_DP and the rain rate are proportional to ``Nw``; Z_DR and rho_hv do not
depend on it, so they are tabulated once on a (Dm, mu) grid per band and
water temperature and interpolated.

The distributions are drawn from a mixture of regimes (stratiform, maritime
and continental convection, heavy tropical rain, and a broad component) whose
widths are radarx choices that cover the ranges of Dm and log10(Nw) reported
for different climatic regimes by Bringi et al. (2003); they are not fitted
to a data set. On top of the scatter of the distributions the measurements
get random errors (``add_noise``): a calibration bias and noise of Z_H and
Z_DR, noise on K_DP and rho_hv.
"""

from __future__ import annotations

import functools

import numpy as np
from scipy.special import gamma as gamma_fn

from radarx.retrieve.dsd import _single_drop, _trapezoid_weights

BANDS = ("S", "C", "X")
#: Radar wavelengths (cm) of the scattering tables (the feature ``wavelength``).
WAVELENGTH = {"S": 10.70687, "C": 5.35344, "X": 3.18928}
TEMPS = (5.0, 15.0, 25.0)
DM = np.linspace(0.3, 4.5, 211)
MU = np.linspace(-2.0, 12.0, 57)
FEATURES = ("dbz", "zdr", "kdp", "rhohv", "wavelength")

# Regimes: weight, Dm mean and sd (mm), log10(Nw) at the mean Dm, its slope
# against Dm, its sd. A radar-gauge-free choice, see the module docstring.
REGIMES = {
    "stratiform": (0.25, 1.4, 0.30, 3.9, -0.8, 0.35),
    "maritime": (0.20, 1.6, 0.30, 3.9, -0.8, 0.30),
    "continental": (0.20, 2.0, 0.35, 3.4, -1.0, 0.35),
    "tropical": (0.15, 1.3, 0.30, 4.4, -0.8, 0.30),
    "broad": (0.20, 1.8, 0.70, 3.8, -0.5, 0.60),
}
RAIN_RANGE = (0.1, 400.0)  # mm h-1 kept in the training set


@functools.lru_cache(maxsize=16)
def table(band, temperature):
    """Per unit Nw on the (Dm, mu) grid: Z_h, Z_v (mm6 m-3), copolar term, K_DP, rain rate."""
    wl, d, data = _single_drop(band, temperature)
    w = _trapezoid_weights(d)
    dm = DM[:, None, None]
    mu = MU[None, :, None]
    f = 6.0 / 4.0**4 * (4.0 + mu) ** (mu + 4.0) / gamma_fn(mu + 4.0)
    with np.errstate(over="ignore", under="ignore"):
        nd = f * (d / dm) ** mu * np.exp(-(4.0 + mu) * d / dm) * w
    k = wl**4 / (np.pi**5 * 0.93)
    zh = k * nd @ data[:, 0]
    zv = k * nd @ data[:, 1]
    copol = -k * (nd @ data[:, 2] + 1j * nd @ data[:, 3])
    kdp = nd @ data[:, 4]
    speed = 9.65 - 10.3 * np.exp(-0.6 * d)
    rr = 6.0e-4 * np.pi * nd @ (speed * d**3)
    return zh, zv, copol, kdp, rr


def _interp(grid, dm, mu):
    """Bilinear interpolation on the (DM, MU) grid."""
    x = np.clip((dm - DM[0]) / (DM[1] - DM[0]), 0, DM.size - 1.000001)
    y = np.clip((mu - MU[0]) / (MU[1] - MU[0]), 0, MU.size - 1.000001)
    i, j = x.astype(int), y.astype(int)
    fx, fy = x - i, y - j
    return (
        grid[i, j] * (1 - fx) * (1 - fy)
        + grid[i + 1, j] * fx * (1 - fy)
        + grid[i, j + 1] * (1 - fx) * fy
        + grid[i + 1, j + 1] * fx * fy
    )


def radar_variables(band, temperature, log_nw, dm, mu):
    """Z_H (dBZ), Z_DR (dB), K_DP (deg/km), rho_hv and rain rate (mm/h) of the distributions."""
    zh, zv, copol, kdp, rr = table(band, temperature)
    nw = 10.0**log_nw
    zh_i, zv_i = _interp(zh, dm, mu), _interp(zv, dm, mu)
    c = _interp(copol.real, dm, mu) + 1j * _interp(copol.imag, dm, mu)
    return (
        10.0 * np.log10(nw * zh_i),
        10.0 * np.log10(zh_i / zv_i),
        nw * _interp(kdp, dm, mu),
        np.minimum(np.abs(c) / np.sqrt(zh_i * zv_i), 1.0),
        nw * _interp(rr, dm, mu),
    )


def draw_dsd(rng, n, regimes=None):
    """``n`` draws of (log10 Nw, Dm, mu) from the mixture of regimes."""
    names = list(REGIMES if regimes is None else regimes)
    weights = np.array([REGIMES[k][0] for k in names])
    pick = rng.choice(len(names), size=n, p=weights / weights.sum())
    par = np.array([REGIMES[k][1:] for k in names])[pick]
    dm = np.clip(rng.normal(par[:, 0], par[:, 1]), 0.5, 4.0)
    log_nw = par[:, 2] + par[:, 3] * (dm - par[:, 0]) + rng.normal(0, par[:, 4])
    log_nw = np.clip(log_nw, 2.0, 5.5)
    mu = np.clip(rng.gamma(2.5, 1.4, n) - 1.0, -1.0, 10.0)
    return log_nw, dm, mu


def add_noise(rng, band_idx, dbz, zdr, kdp, rhohv, level=1.0):
    """
    Measurement errors: Z_H bias N(0, 0.5 dB) and noise U(0.3, 1.5) dB, Z_DR
    bias N(0, 0.1 dB) and noise U(0.05, 0.3) dB, K_DP noise U(0.03, 0.3)
    degree/km plus 10 % of its value, rho_hv noise U(0.002, 0.01). ``level``
    scales all of them (0: no noise).
    """
    n = dbz.size
    dbz = dbz + level * (
        rng.normal(0, 0.5, n) + rng.uniform(0.3, 1.5, n) * rng.normal(size=n)
    )
    zdr = zdr + level * (
        rng.normal(0, 0.1, n) + rng.uniform(0.05, 0.3, n) * rng.normal(size=n)
    )
    kdp = kdp + level * (
        rng.uniform(0.03, 0.3, n) * rng.normal(size=n) + 0.1 * kdp * rng.normal(size=n)
    )
    rhohv = np.minimum(
        rhohv + level * rng.uniform(0.002, 0.01, n) * rng.normal(size=n), 1.0
    )
    return dbz, zdr, kdp, np.clip(rhohv, 0.5, 1.0)


def simulate(n, seed, bands=BANDS, regimes=None, noise=1.0, rain_range=RAIN_RANGE):
    """
    ``n`` samples of the features and the true rain rate.

    Returns a dict of float32 arrays: ``features`` (n, 5) in the order of
    ``FEATURES``, ``rain_rate`` (mm/h), and ``band`` (0, 1, 2 for S, C, X),
    ``dm``, ``log_nw`` and the noise-free ``dbz`` for diagnostics.
    """
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("features", "rain_rate", "band", "dm", "log_nw", "dbz_true")}
    got = 0
    while got < n:
        m = int(1.3 * (n - got)) + 1000
        log_nw, dm, mu = draw_dsd(rng, m, regimes)
        band = rng.integers(len(bands), size=m)
        temp = rng.choice(TEMPS, size=m)
        dbz = np.empty(m)
        zdr = np.empty(m)
        kdp = np.empty(m)
        rho = np.empty(m)
        rr = np.empty(m)
        for ib, b in enumerate(bands):
            for t in TEMPS:
                sel = (band == ib) & (temp == t)
                if sel.any():
                    dbz[sel], zdr[sel], kdp[sel], rho[sel], rr[sel] = radar_variables(
                        b, t, log_nw[sel], dm[sel], mu[sel]
                    )
        keep = (rr >= rain_range[0]) & (rr <= rain_range[1]) & np.isfinite(dbz)
        idx = np.flatnonzero(keep)[: n - got]
        z0 = dbz[idx]
        nz = add_noise(rng, band[idx], z0, zdr[idx], kdp[idx], rho[idx], noise)
        wl = np.array([WAVELENGTH[bands[i]] for i in range(len(bands))])[band[idx]]
        out["features"].append(np.stack([*nz, wl], axis=1))
        out["rain_rate"].append(rr[idx])
        out["band"].append(np.array([BANDS.index(bands[i]) for i in band[idx]]))
        out["dm"].append(dm[idx])
        out["log_nw"].append(log_nw[idx])
        out["dbz_true"].append(z0)
        got += idx.size
    return {
        k: np.concatenate(v).astype(np.float32 if k != "band" else np.int8)
        for k, v in out.items()
    }
