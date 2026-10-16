#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Mixed-Order Cone Gridding
=========================

Cone gridding of a radar volume with the mixed-order ordered linear
interpolation of Collis et al. (2010), for radial velocities that go into a
variational wind retrieval.

:func:`radarx.grid.grid_cones` interpolates inside every sweep first and
between the sweeps in height last ("vertical last"). Where the radial
velocity has a strong vertical gradient and the beams are far apart, that
order leaves periodic undulations at the horizontal spacing of the beams, and
the horizontal divergence of such a perturbation is turned into a spurious
vertical velocity by the mass continuity equation of the retrieval (Collis et
al. 2010, 1.8 m s-1 in their synthetic case, 5 m s-1 in their TWP-ICE case).
The remedy of the paper is to grid the data twice, once in each order of
interpolation, and to return a weighted sum, with the weight of the
horizontal-last result growing with the elevation of the beam: horizontal
last (no undulations, smoother horizontally) at high elevations, where the
beams are far apart, and vertical last (the sharper result) at low elevations.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = ["grid_cones_mixed"]

__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np

from .._provenance import provenance
from .cone import (
    EARTH_RADIUS,
    K_E,
    _check_k_e,
    _load_volume,
    _select_sweeps,
    _sweep_arrays,
    grid_cones,
)

try:
    from . import _mixed

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _mixed = None
    HAS_COMPILED_KERNEL = False


def _use_compiled(engine):
    """Whether to run the compiled kernel for the requested ``engine``."""
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled mixed-order kernel is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


def _beam_height(r, sin_el, big_r, sr):
    return np.sqrt(2.0 * sin_el * r * sr + r * r + sr * sr) - big_r


def _sweep_geometry(arrays, big_r, sr):
    """Gate ground distances, ray order and spacing of a sweep (as the kernel)."""
    data, azimuth, elevation, rng = arrays
    el_med = np.radians(np.median(elevation))
    sin_m, cos_m = np.sin(el_med), np.cos(el_med)
    gate_ground = big_r * np.arcsin(
        rng * cos_m / (big_r + _beam_height(rng, sin_m, big_r, sr))
    )
    ray_az = np.mod(azimuth, 360.0)
    order = np.argsort(ray_az, kind="stable")
    a = ray_az[order]
    keep = np.r_[True, np.diff(a) > 1e-6]
    order, a = order[keep], a[keep]
    if a.size < 3:
        raise ValueError("sweep needs at least 3 distinct azimuths")
    return dict(
        data=data,
        gate_ground=gate_ground,
        a_ext=np.r_[a[-1] - 360.0, a, a[0] + 360.0],
        ray_ext=np.r_[order[-1], order, order[0]],
        spacing=np.median(np.diff(a)),
        sin_m=sin_m,
        cos_m=cos_m,
    )


def _crossing(g, z, big_r, sr):
    """Ground distance (m) at which the beam of a sweep is at height ``z``."""
    rad = sr * sr * (g["sin_m"] ** 2 - 1.0) + (big_r + z) ** 2
    if not rad >= 0.0:
        return np.nan
    r = -sr * g["sin_m"] + np.sqrt(rad)
    if not r > 0.0:
        return np.nan
    return big_r * np.arcsin(min(1.0, r * g["cos_m"] / (big_r + z)))


def _value_at(g, ground, az, max_gap, min_weight):
    """Bilinear value of a sweep at ground distance ``ground`` and azimuth ``az``."""
    gate_ground, a_ext, ray_ext = g["gate_ground"], g["a_ext"], g["ray_ext"]
    ng = gate_ground.size
    inside = (ground >= gate_ground[0]) & (ground <= gate_ground[-1])
    i = np.clip(np.searchsorted(gate_ground, ground, side="right") - 1, 0, ng - 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        fr = (ground - gate_ground[i]) / (gate_ground[i + 1] - gate_ground[i])
    j = np.clip(np.searchsorted(a_ext, az, side="right") - 1, 0, a_ext.size - 2)
    da = a_ext[j + 1] - a_ext[j]
    fa = (az - a_ext[j]) / da
    ok = inside & (da <= max_gap * g["spacing"])
    r0, r1 = ray_ext[j], ray_ext[j + 1]
    num = np.zeros(az.shape)
    den = np.zeros(az.shape)
    for ray, wa in ((r0, 1 - fa), (r1, fa)):
        for gate, wr in ((i, 1 - fr), (i + 1, fr)):
            value = g["data"][ray, gate]
            weight = wa * wr
            valid = np.isfinite(value)
            num += np.where(valid, weight * value, 0.0)
            den += np.where(valid, weight, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok & (den >= min_weight), num / den, np.nan)


def _mixed_numpy(ground, az, z, sweeps, site_altitude, max_gap, min_weight, k_e):
    """NumPy reference of ``_mixed.mixed_columns`` (same results)."""
    big_r = EARTH_RADIUS * k_e
    sr = big_r + site_altitude
    shape = ground.shape
    s, a = ground.ravel(), np.mod(az.ravel(), 360.0)
    geo = [_sweep_geometry(arrays, big_r, sr) for arrays in sweeps]
    out = np.full((z.size,) + s.shape, np.nan, dtype=np.float32)
    for iz, level in enumerate(z):
        cross = [_crossing(g, level, big_r, sr) for g in geo]
        done = np.zeros(s.shape, dtype=bool)
        for k in range(len(geo) - 1):
            s0, s1 = cross[k], cross[k + 1]
            if not abs(s1 - s0) > 1e-9:
                continue
            enclosed = ~done & ((s - s0) * (s - s1) <= 0.0)
            if not enclosed.any():
                continue
            v0 = _value_at(geo[k], np.full(s.shape, s0), a, max_gap, min_weight)
            v1 = _value_at(geo[k + 1], np.full(s.shape, s1), a, max_gap, min_weight)
            value = v0 + (s - s0) / (s1 - s0) * (v1 - v0)
            out[iz] = np.where(enclosed, value, out[iz])
            done |= enclosed
    return out.reshape((z.size,) + shape)


def horizontal_last(
    ground,
    azimuth,
    z,
    sweeps,
    site_altitude,
    *,
    k_e=K_E,
    max_gap=2.0,
    min_weight=0.5,
    n_threads=None,
    engine="auto",
):
    """
    Horizontal-last interpolation of one field of one radar onto columns.

    ``ground`` and ``azimuth`` (degrees) give the ground distance (m) and the
    azimuth of every column as seen from the radar, on ``(y, x)``; ``sweeps``
    is a list of ``(data, azimuth, elevation, range)`` arrays sorted by
    elevation (the output of the sweep selection of :func:`grid_cones`).
    Returns float32 on ``(z, y, x)``.
    """
    ground = np.ascontiguousarray(ground, dtype=np.float64)
    azimuth = np.ascontiguousarray(azimuth, dtype=np.float64)
    z = np.ascontiguousarray(z, dtype=np.float64)
    if _use_compiled(engine):
        data, az, el, rng = (list(a) for a in zip(*sweeps))
        return np.asarray(
            _mixed.mixed_columns(
                ground,
                azimuth,
                z,
                data,
                az,
                el,
                rng,
                float(site_altitude),
                earth_radius=EARTH_RADIUS,
                max_gap=max_gap,
                min_weight=min_weight,
                n_threads=int(n_threads or 0),
                k_e=k_e,
            )
        )
    return _mixed_numpy(
        ground, azimuth, z, sweeps, float(site_altitude), max_gap, min_weight, k_e
    )


def antenna_elevation(ground, z, site_altitude, k_e=K_E):
    """
    Antenna elevation (degrees) of the beam to ground distance ``ground``, height ``z``.

    From the triangle of the Earth centre, the radar and the point on the
    effective Earth radius ``k_e a`` (Doviak and Zrnic 1993, Eqs. 2.28b-d);
    the same relation that :func:`radarx.retrieve.radar_geometry` uses before it
    adds the central angle.
    """
    big_r = EARTH_RADIUS * k_e
    a = big_r + site_altitude
    b = big_r + np.asarray(z, dtype=np.float64)
    gamma = np.asarray(ground, dtype=np.float64) / big_r
    r = np.sqrt(np.maximum(a * a + b * b - 2.0 * a * b * np.cos(gamma), 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        sin_el = (b * b - r * r - a * a) / (2.0 * r * a)
    return np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))


def horizontal_weight(theta, theta_max, order):
    """
    Weight of the horizontal-last result, Eq. 6 of Collis et al. (2010).

    :math:`w_h = [(\\sin(\\pi \\theta/\\theta_{max} - \\pi/2) + 1) / 2]^n` for
    the antenna elevation :math:`\\theta` of the point, between 0 and
    :math:`\\theta_{max}` (values outside are clipped).
    """
    if theta_max > 0:
        t = np.clip(np.asarray(theta, dtype=np.float64) / theta_max, 0.0, 1.0)
    else:
        t = np.zeros(np.shape(theta))
    return ((np.sin(np.pi * t - np.pi / 2.0) + 1.0) / 2.0) ** order


@provenance(
    "Mixed-order ordered linear interpolation of radar volumes (Collis et al. 2010)",
    extra_refs=("doviak-zrnic-1993",),
)
def grid_cones_mixed(
    obj,
    data_vars=None,
    x=None,
    y=None,
    z=None,
    *,
    order=0.5,
    k_e=K_E,
    max_gap=2.0,
    min_weight=0.5,
    n_threads=None,
    engine="auto",
):
    """
    Grid a radar volume with mixed-order interpolation (Collis et al. 2010).

    Every field is gridded twice, by :func:`radarx.grid.grid_cones` (inside each
    sweep first, then in height: "vertical last", :math:`G_v`) and with the
    reverse order of interpolation ("horizontal last", :math:`G_h`: in height
    first, then in azimuth and range), and the two are blended

    .. math::

       G = w_h\\, G_h + (1 - w_h)\\, G_v , \\qquad
       w_h = \\left[\\frac{\\sin(\\pi \\theta/\\theta_{max} - \\pi/2) + 1}{2}\\right]^n ,

    where :math:`\\theta` is the antenna elevation of the beam to the point
    and :math:`\\theta_{max}` the highest sweep elevation of the field (Eq. 6 of
    Collis et al. 2010 [1]_). The weight grows from 0 at the lowest elevation
    to 1 at the highest; it is smaller for larger ``order`` :math:`n`
    (:math:`n = 1/2` is :math:`w_h = \\sin(\\pi\\theta / 2\\theta_{max})`).

    The horizontal-last grid takes, for an output point at height :math:`z`,
    the value of every sweep at the point of its beam that is at height
    :math:`z` (at the ground distance :math:`s_k(z)`, with the azimuth
    interpolated between the rays), and interpolates linearly in ground
    distance between the two sweeps whose crossing points enclose the point.
    Between sparse beams a field that varies strongly in the vertical but
    little in the horizontal, such as a jet, is therefore reproduced, where
    the vertical-last order leaves undulations with the horizontal spacing of
    the beams.

    Parameters
    ----------
    obj : xarray.DataTree
        Radar volume, as for :func:`radarx.grid.grid_cones`.
    data_vars : str or list of str, optional
        Fields to grid; by default every field in the lowest sweep.
    x, y, z : array-like
        Grid coordinates (m) east and north of the radar and above sea level.
    order : float, optional
        The exponent :math:`n`, at least 0 (0 gives horizontal last
        everywhere, large values vertical last except at the top). Default
        0.5, the value Collis et al. chose for the 0.5 to 42 degree scan
        strategy of C-POL; it will differ for other scan strategies.
    k_e, max_gap, min_weight, n_threads, engine :
        As for :func:`radarx.grid.grid_cones`.

    Returns
    -------
    xarray.Dataset
        As for :func:`radarx.grid.grid_cones`, with the attribute
        ``gridding_method = "mixed-order cone"`` and ``mixed_order``. Cells
        where only one of the two orders gives a value take that value.
        Levels below the lowest sweep stay empty (no ``fill_below``).

    Notes
    -----
    Source of the method. The two orders of ordered linear interpolation, the
    weighted sum and the weight :math:`w_h` of Eq. 6 are those of Collis et
    al. (2010) [1]_. The paper's text gives the weight of the vertical-last
    result as :math:`w_v = w_h - 1`, which cannot be a weight of a weighted
    sum; ``1 - w_h`` is used. The paper describes the horizontal-last order
    in one sentence ("interpolate in height, azimuth, and then radially", with
    "very poor horizontal resolution" as the price) and gives no equations
    for it; the construction above, the beam crossing points at the output
    height, is radarx's reading of that sentence, chosen because it has the two
    properties the paper reports (no vertical-gradient undulations, poor
    horizontal resolution). It is not their code. The paper's data had the
    rays and gates of one radar; the sweeps here need not share them. The beam
    height is that of the effective Earth radius model (Doviak and Zrnić 1993
    [2]_, Eq. 2.28b), as in :func:`radarx.grid.grid_cones`. The use of the
    median elevation of a sweep for the crossing points, ``max_gap`` and
    ``min_weight`` are radarx choices.

    What it does and does not do. Horizontal last removes the undulations at the
    beam spacing and smooths the field horizontally at high elevations (the
    paper: "at the cost of reduced horizontal resolution at higher
    elevations"). Whether it also removes real small-scale updrafts has not
    been studied (the paper, section 6, leaves it for later). The paper tested
    the scheme on synthetic flow and on one TWP-ICE case with two C-band radars;
    radarx checks it on synthetic flow (see the tests) and does not claim more.

    References
    ----------
    .. [1] Collis, S., A. Protat, and K.-S. Chung, 2010: The effect of radial
       velocity gridding artifacts on variationally retrieved vertical
       velocities. *J. Atmos. Oceanic Technol.*, **27**, 1239-1246,
       https://doi.org/10.1175/2010JTECHA1402.1
    .. [2] Doviak, R. J., and D. S. Zrnić, 1993: *Doppler Radar and Weather
       Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
       DOI).

    Examples
    --------
    >>> grid = radarx.grid.grid_cones_mixed(dtree, "VRADH", x, y, z)  # doctest: +SKIP
    """
    use_compiled = _use_compiled(engine)
    if x is None or y is None or z is None:
        raise ValueError("x, y and z target coordinates are required.")
    if not np.isfinite(order) or order < 0:
        raise ValueError(f"order must be a number of at least 0, not {order!r}")
    k_e = _check_k_e(k_e)
    x, y, z = (np.asarray(a, dtype=np.float64) for a in (x, y, z))
    sweeps, site, _ = _load_volume(obj)
    if data_vars is None:
        data_vars = [
            name for name, da in sweeps[0].data_vars.items() if {"range"} < set(da.dims)
        ]
    elif isinstance(data_vars, str):
        data_vars = [data_vars]
    out = grid_cones(
        obj,
        data_vars,
        x,
        y,
        z,
        k_e=k_e,
        max_gap=max_gap,
        min_weight=min_weight,
        n_threads=n_threads,
        engine="compiled" if use_compiled else "numpy",
    )
    gx, gy = np.meshgrid(x, y)
    ground = np.hypot(gx, gy)
    azimuth = np.mod(np.degrees(np.arctan2(gx, gy)), 360.0)
    for name in data_vars:
        out[name] = out[name].copy(
            data=mix_orders(
                out[name].values,
                sweeps,
                site.get("altitude", 0.0),
                name,
                ground,
                azimuth,
                z,
                order=order,
                k_e=k_e,
                max_gap=max_gap,
                min_weight=min_weight,
                n_threads=n_threads,
                engine="compiled" if use_compiled else "numpy",
            )
        )
    out.attrs["gridding_method"] = "mixed-order cone"
    out.attrs["mixed_order"] = float(order)
    return out


def mix_orders(
    vertical_last,
    sweeps,
    site_altitude,
    name,
    ground,
    azimuth,
    z,
    *,
    order=0.5,
    k_e=K_E,
    max_gap=2.0,
    min_weight=0.5,
    n_threads=None,
    engine="auto",
):
    """
    Mixed-order result of one field from its vertical-last grid.

    ``vertical_last`` is the cone-gridded field on ``(z, y, x)``; ``sweeps`` the
    sweep datasets of the volume (``_load_volume``), ``ground`` and ``azimuth``
    the ground distance (m) and the azimuth (degrees) of the columns seen from
    the radar on ``(y, x)``. Returns the weighted sum of the two orders (float32).
    """
    selected = _select_sweeps(sweeps, name)
    arrays = [_sweep_arrays(ds, name) for ds in selected]
    h = horizontal_last(
        ground,
        azimuth,
        z,
        arrays,
        site_altitude,
        k_e=k_e,
        max_gap=max_gap,
        min_weight=min_weight,
        n_threads=n_threads,
        engine=engine,
    )
    theta_max = max(float(np.nanmedian(ds["elevation"].values)) for ds in selected)
    theta = antenna_elevation(
        np.asarray(ground)[None], np.asarray(z)[:, None, None], site_altitude, k_e
    )
    w_h = horizontal_weight(theta, theta_max, order)
    v = np.asarray(vertical_last)
    both = np.isfinite(v) & np.isfinite(h)
    blended = np.where(both, w_h * h + (1.0 - w_h) * v, np.where(np.isfinite(v), v, h))
    return blended.astype(np.float32)
