# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
NumPy reference of the ray tracing kernel ``_refraction.cpp``.

The equations, the integration (classical Runge-Kutta from gate to gate, equal
steps of at most ``step`` between two gates, shortened to end at the next
level of the profile, not below 1 mm) and the profile convention
(piecewise linear refractive index, constant below the first and from the last
level on) are those of the kernel and are described in
:mod:`radarx.fundamentals.refraction`. All elevations are advanced together.
"""

import numpy as np

DEG = np.pi / 180.0


def _layer(h, n, hh, sin_th):
    """
    Layer of every ray: ``n(h) = na + dn (h - ha)``.

    Piecewise linear between the levels, constant below the first and from the
    last level on; a ray exactly on a level that travels downward is in the
    layer below it.
    """
    last = h.size - 1
    idx = np.searchsorted(h, hh, side="right") - 1  # last level at or below hh
    on_level = (sin_th < 0.0) & (idx >= 0) & (hh == h[np.clip(idx, 0, last)])
    idx = np.where(on_level, idx - 1, idx)
    flat = (idx < 0) | (idx >= last)
    i = np.clip(idx, 0, max(last - 1, 0))
    j = np.minimum(i + 1, last)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (n[j] - n[i]) / (h[j] - h[i])
    ha = np.where(idx < 0, h[0], np.where(idx >= last, h[-1], h[i]))
    na = np.where(idx < 0, n[0], np.where(idx >= last, n[-1], n[i]))
    return ha, na, np.where(flat, 0.0, slope)


def _deriv(layer, a, r, th):
    ha, na, dn = layer
    nn = na + dn * (r - a - ha)
    c = np.cos(th)
    return np.sin(th), c / r, c * (1.0 / r + dn / nn)


def _step(h, n, a, r, phi, th, ds):
    layer = _layer(h, n, r - a, np.sin(th))
    k1 = _deriv(layer, a, r, th)
    k2 = _deriv(layer, a, r + 0.5 * ds * k1[0], th + 0.5 * ds * k1[2])
    k3 = _deriv(layer, a, r + 0.5 * ds * k2[0], th + 0.5 * ds * k2[2])
    k4 = _deriv(layer, a, r + ds * k3[0], th + ds * k3[2])
    return (
        r + ds / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
        phi + ds / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
        th + ds / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]),
    )


def _cut_length(h, hh, sin_th):
    """Path length to the next level of the profile in the direction of travel."""
    idx = np.searchsorted(h, hh, side="right") - 1  # last level at or below hh
    last = h.size - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        up_ok = idx + 1 <= last
        up = np.where(up_ok, h[np.clip(idx + 1, 0, last)], np.nan)
        on_level = (idx >= 0) & (hh > h[np.clip(idx, 0, last)])
        dn_a = h[np.clip(idx, 0, last)]
        dn_b_ok = idx - 1 >= 0
        dn_b = h[np.clip(idx - 1, 0, last)]
        down = np.where(on_level, dn_a, np.where(dn_b_ok, dn_b, np.nan))
        target = np.where(sin_th > 0.0, up, down)
        cut = (target - hh) / sin_th
    return np.where((np.abs(sin_th) < 1e-12) | np.isnan(target), np.inf, cut)


def trace(rng, elev, h, n, radar_height, earth_radius, step=100.0, n_threads=0):
    """
    Height, ground distance and elevation of the ray at every gate.

    Returns an array ``(len(elev), len(rng), 3)``: the height above sea level
    (m), the distance on the ground at sea level (m) and the elevation of the
    ray at the gate (degrees). ``n_threads`` is accepted for the signature of
    the compiled kernel.
    """
    rng = np.asarray(rng, dtype=np.float64)
    elev = np.asarray(elev, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    if rng.ndim != 1 or elev.ndim != 1 or h.ndim != 1 or n.ndim != 1:
        raise ValueError("rng, elev, h and n must be one-dimensional")
    if h.size != n.size or h.size < 1:
        raise ValueError("the profile needs matching heights and indices")
    if not step > 0.0:
        raise ValueError("step must be positive")
    if rng.size > 1 and not np.all(rng[1:] >= rng[:-1]):
        raise ValueError("rng must not decrease")
    out = np.full((elev.size, rng.size, 3), np.nan)
    ok = np.isfinite(elev)
    r = np.full(ok.sum(), earth_radius + radar_height)
    phi = np.zeros(r.shape)
    th = elev[ok] * DEG
    s = 0.0
    for k, target in enumerate(rng):
        length = target - s
        if length > 0.0:
            m = int(np.ceil(length / step))
            nominal = length / m
            rem = np.full(r.shape, length)
            active = np.ones(r.shape, dtype=bool)
            while active.any():
                sin_th = np.sin(th)
                cut = _cut_length(h, r - earth_radius, sin_th)
                # not shorter than 1 mm, nor than a path that changes r by 1e-8 m
                # (r is known to 1e-9 m), so that a step always crosses the level
                least = np.maximum(1e-3, 1e-8 / np.maximum(np.abs(sin_th), 1e-300))
                ds = np.minimum(nominal, np.maximum(cut, least))
                last = ds >= rem
                ds = np.where(last, rem, ds)
                r2, phi2, th2 = _step(h, n, earth_radius, r, phi, th, ds)
                r = np.where(active, r2, r)
                phi = np.where(active, phi2, phi)
                th = np.where(active, th2, th)
                rem = np.where(last, 0.0, rem - ds)
                active = active & ~last
            s = target
        out[ok, k, 0] = r - earth_radius
        out[ok, k, 1] = earth_radius * phi
        out[ok, k, 2] = th / DEG
    return out
