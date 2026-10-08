#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Simulated differential phase rays with known KDP and backscatter phase.

Every ray is built from a range profile of normalized-gamma raindrop size
distributions (DSDs). The radar variables of each gate (Z_H, Z_DR, K_DP,
the backscatter differential phase delta and rho_hv) are integrals of the
single-drop T-matrix scattering tables shipped with radarx
(``radarx.retrieve.dsd.scattering_table``) over the DSD, so K_DP and delta
are physically consistent with each other and with Z_H. On top of the rain
profile come the things a KDP estimator has to cope with:

- convective cells and stratiform rain, size sorting (large drops, low
  concentration) at cell edges;
- a melting layer with a bright band, a rho_hv dip and a backscatter phase
  bump, above which K_DP is small (ice);
- hail cores with an extra backscatter phase and lower rho_hv;
- occasional slightly negative K_DP (vertically aligned ice);
- attenuation of Z_H (two-way, proportional to Phi_DP);
- phase noise that grows at low signal-to-noise ratio and low rho_hv;
- non-meteorological gates (clear air with random phase, clutter spikes),
  data gaps (no-data gates);
- a random system offset, folding into [-180, 180) or [0, 360), and the
  reversed sign convention of some systems.

The measured phase is

    Psi_DP = sign * (offset + 2 * integral(K_DP) dr + delta + noise)  (folded)

and the truth (K_DP, delta) is returned with the network inputs computed by
``radarx.retrieve.kdp._ml_features`` (the same function used at inference).
"""

from __future__ import annotations

import functools

import numpy as np
from scipy.special import gamma as gamma_fn

from radarx.retrieve.dsd import _single_drop, _trapezoid_weights
from radarx.retrieve.kdp import _decide_sign, _mask_numpy, _ml_features

BANDS = ("S", "C", "X")
# two-way attenuation per degree of Phi_DP (dB/deg), typical rain values
ALPHA = {"S": 0.016, "C": 0.08, "X": 0.28}
# maximum melting-layer / hail backscatter phase (degrees)
DELTA_ML = {"S": 6.0, "C": 9.0, "X": 12.0}
DELTA_HAIL = {"S": 4.0, "C": 15.0, "X": 15.0}
GATE_SPACINGS = (0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.5)  # km
DM = np.linspace(0.5, 4.0, 141)
MU = np.linspace(-1.0, 8.0, 19)
TEMPS = (5.0, 15.0, 25.0)

# default estimate_kdp masking parameters (in gates, filled per dr)
MASK_PARAMS = {
    "rhohv_min": 0.85,
    "texture_max": 20.0,
    "n_offset": 10,
    "offset_mode": "sweep",
    "offset": 0.0,
}


@functools.lru_cache(maxsize=16)
def lookup(band, temperature):
    """
    Radar variables of normalized-gamma DSDs with N_w = 1 m-3 mm-1 on the
    (Dm, mu) grid: Z_h, Z_v (mm6 m-3), copolar term (complex), K_DP.
    Z and K_DP scale with N_w; Z_DR, delta and rho_hv do not.
    """
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
    # the tables store S_hh S_vv* in a convention with a 180 degree shift
    copol = -k * (nd @ data[:, 2] + 1j * nd @ data[:, 3])
    kdp = nd @ data[:, 4]
    return zh, zv, copol, kdp


def _interp2(table, dm, mu):
    """Bilinear interpolation on the (DM, MU) grid."""
    x = np.clip((dm - DM[0]) / (DM[1] - DM[0]), 0, DM.size - 1.000001)
    y = np.clip((mu - MU[0]) / (MU[1] - MU[0]), 0, MU.size - 1.000001)
    i = x.astype(int)
    j = y.astype(int)
    fx = x - i
    fy = y - j
    return (
        table[i, j] * (1 - fx) * (1 - fy)
        + table[i + 1, j] * fx * (1 - fy)
        + table[i, j + 1] * (1 - fx) * fy
        + table[i + 1, j + 1] * fx * fy
    )


def rain_variables(band, temperature, log_nw, dm, mu):
    """Z_H (dBZ), Z_DR (dB), K_DP (deg/km), delta (deg), rho_hv per gate."""
    zh, zv, copol, kdp = lookup(band, temperature)
    nw = 10.0**log_nw
    zh = nw * _interp2(zh, dm, mu)
    zv = nw * _interp2(zv, dm, mu)
    c = nw * (_interp2(copol.real, dm, mu) + 1j * _interp2(copol.imag, dm, mu))
    kdp = nw * _interp2(kdp, dm, mu)
    return (
        10 * np.log10(zh),
        10 * np.log10(zh / zv),
        kdp,
        np.degrees(np.angle(c)),
        np.abs(c) / np.sqrt(zh * zv),
    )


def _smooth_noise(rng, n, scale_gates, size=1.0):
    """Gaussian random field along range with a given correlation length."""
    white = rng.normal(0.0, 1.0, n + 6 * scale_gates)
    x = np.arange(-3 * scale_gates, 3 * scale_gates + 1)
    kern = np.exp(-0.5 * (x / max(scale_gates, 1)) ** 2)
    kern /= np.sqrt((kern**2).sum())
    out = np.convolve(white, kern, mode="same")[3 * scale_gates : 3 * scale_gates + n]
    return size * out


def _bump(r, center, width, power=2.0):
    return np.exp(-0.5 * np.abs((r - center) / width) ** power)


def simulate_ray(rng, n_gates, dr, band=None):
    """
    One simulated ray.

    Returns a dict with the measured ``phi`` (deg), ``rho``, ``z`` (dBZ, or
    None), and the truth ``kdp`` (deg/km), ``delta`` (deg), ``echo`` (gates
    inside precipitation), ``phi_true`` (2 * integral of K_DP, deg) and
    the metadata ``band``, ``dr``, ``offset``, ``sign``, ``sigma``.
    """
    band = band or BANDS[rng.integers(3)]
    temp = TEMPS[rng.integers(len(TEMPS))]
    r0 = rng.uniform(1.0, 60.0)
    r = r0 + dr * np.arange(n_gates)
    length = r[-1] - r[0]
    corr = max(int(round(2.0 / dr)), 1)  # ~2 km correlation of DSD noise

    # --- precipitation intensity: stratiform background and cells
    intensity = np.full(n_gates, rng.uniform(0.0, 0.8) * (rng.random() < 0.8))
    n_cells = rng.integers(0, 6)
    cells = []
    for _ in range(n_cells):
        c = rng.uniform(r[0] - 5, r[-1] + 5)
        w = rng.uniform(0.7, 12.0)
        a = rng.uniform(0.3, 1.6)
        intensity += a * _bump(r, c, w, rng.choice([2.0, 2.0, 4.0]))
        cells.append((c, w, a))
    intensity += _smooth_noise(rng, n_gates, corr, 0.08)
    intensity = np.clip(intensity, 0.0, 2.4)

    dm = 0.9 + 0.65 * intensity + _smooth_noise(rng, n_gates, corr, 0.12)
    log_nw = 3.0 + 0.65 * intensity + _smooth_noise(rng, n_gates, corr, 0.2)
    if cells and rng.random() < 0.4:  # size sorting at the leading edge
        c, w, a = cells[rng.integers(len(cells))]
        edge = _bump(r, c + rng.choice([-1, 1]) * 1.5 * w, 0.6 * w)
        dm += rng.uniform(0.5, 1.5) * edge
        log_nw -= rng.uniform(0.5, 1.5) * edge
    dm = np.clip(dm, 0.6, 3.8)
    log_nw = np.clip(log_nw, 1.5, 5.3)
    mu = np.clip(rng.uniform(0, 8) + _smooth_noise(rng, n_gates, corr, 1.0), -1, 8)

    z, zdr, kdp, delta, rho = rain_variables(band, temp, log_nw, dm, mu)
    rho = np.minimum(rho, 0.999)

    # --- echo extent: the ray leaves precipitation where Z drops low and in
    # clear-air segments
    echo = z > rng.uniform(-5.0, 10.0)
    for _ in range(rng.integers(0, 3)):
        a = rng.uniform(r[0], r[-1])
        b = a + rng.uniform(2.0, 0.4 * length + 2.0)
        echo &= ~((r >= a) & (r <= b))

    # --- melting layer: ice beyond it (ray rising through the 0 C level)
    if rng.random() < 0.35:
        rm = rng.uniform(r[0] + 0.1 * length, r[-1] + 20.0)
        wm = rng.uniform(0.3, 2.5)
        above = r > rm + wm
        ml = _bump(r, rm, wm)
        kdp = np.where(above, rng.uniform(0.0, 0.25), kdp)
        z = np.where(above, z - rng.uniform(5.0, 12.0) * (1 - np.exp(-(r - rm) / 5)), z)
        z = z + rng.uniform(4.0, 10.0) * ml
        delta = np.where(above, 0.0, delta) + rng.uniform(0.0, DELTA_ML[band]) * ml
        rho = rho - rng.uniform(0.02, 0.1) * ml
        rho = np.where(above, rng.uniform(0.97, 0.995), rho)

    # --- hail in the strongest core
    if cells and rng.random() < 0.2:
        c, w, a = max(cells, key=lambda t: t[2])
        hail = _bump(r, c, rng.uniform(0.3, 1.0) * min(w, 4.0))
        delta = delta + rng.uniform(1.0, DELTA_HAIL[band]) * hail
        rho = rho - rng.uniform(0.02, 0.08) * hail
        kdp = kdp * (1 - rng.uniform(0.0, 0.7) * hail)
        z = z + rng.uniform(2.0, 8.0) * hail

    # --- vertically aligned ice: slightly negative K_DP
    if rng.random() < 0.05:
        c = rng.uniform(r[0], r[-1])
        seg = _bump(r, c, rng.uniform(1.0, 5.0), 4.0)
        kdp = kdp * (1 - seg) - rng.uniform(0.1, 0.5) * seg
        z = np.minimum(z, 30.0) * seg + z * (1 - seg)

    kdp = np.where(echo, kdp, 0.0)
    delta = np.where(echo, delta, 0.0)
    phi_true = np.concatenate([[0.0], np.cumsum(dr * (kdp[:-1] + kdp[1:]))])
    z = z - ALPHA[band] * phi_true

    # --- measurement noise
    snr = z - 20 * np.log10(r / 50.0) + rng.uniform(5.0, 20.0)
    sigma0 = rng.uniform(1.5, 5.0)
    sigma = sigma0 * (1 + np.clip(0.99 - rho, 0, None) / 0.03)
    sigma = np.clip(sigma * (1 + 3 * np.exp(-np.clip(snr, -20, None) / 5)), 0, 60)
    echo &= snr > -3.0
    offset = rng.uniform(-180.0, 180.0)
    phi = offset + phi_true + delta + rng.normal(0.0, 1.0, n_gates) * sigma
    rho_m = np.clip(rho + rng.normal(0.0, 0.005, n_gates), 0.0, 1.0)
    z_m = z + rng.normal(0.0, 1.0, n_gates)

    noise = ~echo
    # clutter spikes inside echo
    for _ in range(rng.poisson(0.6)):
        g = rng.integers(n_gates)
        k = rng.integers(1, 6)
        noise[g : g + k] = True
    # near-radar clutter
    if rng.random() < 0.3 and r[0] < 20:
        noise[: rng.integers(1, max(2, int(3 / dr)))] = True
    nn = noise.sum()
    phi[noise] = rng.uniform(-180, 180, nn)
    rho_m[noise] = rng.uniform(0.2, 0.85, nn)
    z_m[noise] = rng.uniform(-20, 20, nn)
    if rng.random() < 0.5:  # many systems store no data outside echo
        phi[noise & ~echo] = np.nan
        rho_m[noise & ~echo] = np.nan
        z_m[noise & ~echo] = np.nan
    for _ in range(rng.poisson(0.3)):  # data gaps
        g = rng.integers(n_gates)
        sl = slice(g, g + rng.integers(1, 20))
        phi[sl] = rho_m[sl] = z_m[sl] = np.nan

    sign = -1 if rng.random() < 0.2 else 1
    phi = sign * phi
    if rng.random() < 0.5:
        phi = (phi + 180.0) % 360.0 - 180.0
    else:
        phi = phi % 360.0
    return {
        "phi": phi,
        "rho": rho_m,
        "z": z_m,
        "kdp": kdp,
        "delta": delta,
        "echo": echo & ~noise,
        "phi_true": phi_true,
        "zdr": zdr,
        "band": band,
        "dr": dr,
        "offset": offset,
        "sign": sign,
        "sigma": sigma,
    }


def mask_params(dr, texture_window=2.0):
    """estimate_kdp masking parameters for gate spacing ``dr`` (km)."""
    return {**MASK_PARAMS, "htex": max(1, round(0.5 * texture_window / dr))}


def simulate_batch(rng, n_rays, n_gates, dr=None, band=None):
    """
    A batch of rays with one gate spacing, featurised like at inference.

    The rays of a batch are treated as one sweep for the sign test, as
    ``estimate_kdp`` does; all rays share one sign convention and offset
    mode (per ray), so the batch looks like a sweep of different storms.
    """
    dr = dr or GATE_SPACINGS[rng.integers(len(GATE_SPACINGS))]
    rays = [simulate_ray(rng, n_gates, dr, band) for _ in range(n_rays)]
    sign = rays[0]["sign"]
    phi = np.stack([sign * ray["sign"] * ray["phi"] for ray in rays])
    rho = None if rng.random() < 0.05 else np.stack([ray["rho"] for ray in rays])
    z = None if rng.random() < 0.1 else np.stack([ray["z"] for ray in rays])
    params = {**mask_params(dr), "offset_mode": "ray"}
    mask = _mask_numpy(phi, rho, z, params)
    s = _decide_sign([mask[3]], 0)  # truth below is in the detected convention
    feats, psi, valid, good, _ = _ml_features(phi, rho, z, dr, params, s, mask)
    return {
        "features": feats,
        "psi": np.nan_to_num(psi).astype(np.float32),
        "valid": valid,
        "kdp": np.stack([s * sign * ray["kdp"] for ray in rays]).astype(np.float32),
        "delta": np.stack([s * sign * ray["delta"] for ray in rays]).astype(np.float32),
        "phi": phi,
        "rho": rho,
        "z": z,
        "sign_ok": s == sign,
        "dr": dr,
        "bands": [ray["band"] for ray in rays],
        "echo": np.stack([ray["echo"] for ray in rays]),
    }
