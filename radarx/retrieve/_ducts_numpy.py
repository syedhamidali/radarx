# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
NumPy reference of the trapping-layer search of :mod:`radarx.retrieve.ducts`.

The compiled kernel ``_ducts`` does the same steps in the same order; this
module is the fallback and the test oracle.
"""

from __future__ import annotations

import math

import numpy as np

NFIELDS = 7  # base, top, thickness, deficit, angle, radar angle, surface-based


def _interp(h, n, z):
    """Refractivity at ``z`` by linear interpolation of the valid levels."""
    return float(np.interp(z, h, n))


def trapping_layers(
    height,
    refractivity,
    radar_height,
    earth_radius,
    max_layers,
    min_deficit,
    n_threads=0,
):
    """
    Trapping layers of profiles of refractivity.

    Parameters
    ----------
    height : ndarray (nlev,)
        Heights (m), ascending.
    refractivity : ndarray (nprof, nlev)
        Refractivity in N-units; NaN levels are left out of their profile.
    radar_height : float
        Height of the radar (m) or NaN.
    earth_radius : float
        Radius of the Earth (m) of the modified refractivity.
    max_layers : int
        Number of layers stored per profile.
    min_deficit : float
        Layers whose M-deficit is not above this are not counted.
    n_threads : int
        Ignored.

    Returns
    -------
    fields : ndarray (nprof, max_layers, 7)
        Base height (m), top height (m), thickness (m), M-deficit (M-units),
        trapping angle at the base (rad), trapping angle at the radar (rad) and
        1 for a layer that begins at the lowest level, NaN for unused slots.
    count : ndarray (nprof,) of int
        Number of layers found (may exceed ``max_layers``).
    """
    height = np.asarray(height, dtype=np.float64)
    big_n = np.atleast_2d(np.asarray(refractivity, dtype=np.float64))
    out = np.full((big_n.shape[0], max_layers, NFIELDS), np.nan)
    count = np.zeros(big_n.shape[0], dtype=np.int64)
    scale = 1.0e6 / earth_radius
    for p in range(big_n.shape[0]):
        ok = np.isfinite(big_n[p])
        h, n = height[ok], big_n[p][ok]
        if h.size < 2:
            continue
        m = n + h * scale
        i = 0
        while i < h.size - 1:
            if not (m[i + 1] < m[i]):
                i += 1
                continue
            b = i
            while i < h.size - 1 and m[i + 1] < m[i]:
                i += 1
            t = i
            deficit = m[b] - m[t]
            if not (deficit > min_deficit):
                continue
            if count[p] < max_layers:
                nb = 1.0 + 1.0e-6 * n[b]
                nt = 1.0 + 1.0e-6 * n[t]
                rb, rt = earth_radius + h[b], earth_radius + h[t]
                angle = math.acos(min(nt * rt / (nb * rb), 1.0))
                radar = math.nan
                if np.isfinite(radar_height) and h[b] <= radar_height <= h[t]:
                    nr = 1.0 + 1.0e-6 * _interp(h, n, radar_height)
                    rr = earth_radius + radar_height
                    radar = math.acos(min(nt * rt / (nr * rr), 1.0))
                out[p, count[p]] = (
                    h[b],
                    h[t],
                    h[t] - h[b],
                    deficit,
                    angle,
                    radar,
                    1.0 if b == 0 else 0.0,
                )
            count[p] += 1
    return out, count
