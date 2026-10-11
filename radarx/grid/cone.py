#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx Cone Gridding
====================

Grid a radar volume onto a Cartesian 3D grid by interpolating within each
sweep and then between sweeps.

A radar volume is not a cloud of scattered points: each sweep is a cone of
constant elevation sampled on a regular polar grid. Cone gridding uses that
structure directly:

1. Every output column ``(x, y)`` is mapped to ground distance and azimuth.
   On each sweep, the value and beam height at that column are interpolated
   bilinearly from the four surrounding gates, using the measured ray
   azimuths and elevations.
2. Every output level is interpolated linearly in height between the two
   sweeps that bracket it in that column.

There is no radius of influence or smoothing length to tune. Measured values
are reproduced at the gates, results never overshoot the data, and cells
that are not bracketed by two sweeps (below the lowest, above the highest,
or in a gap) stay empty instead of being filled with data from another
altitude. Optionally, the lowest sweep fills the levels below it
(``fill_below=True``, a pseudo-CAPPI).

The work is done by a compiled C++ kernel; if it is not available, an
equivalent NumPy implementation is used.

Provenance. The method is radarx's own combination of two standard steps and
is not a reproduction of one publication. Its closest published relative is
the vertical interpolation (VI) scheme of Zhang et al. (2005): linear
interpolation between the two tilts around the grid
cell (their Eqs. 5-7, p. 36), nearest neighbour in azimuth and range. Cone
gridding instead interpolates bilinearly in azimuth and range within each
sweep and then linearly in beam height (not in elevation angle) between
sweeps. The beam height uses the 4/3 effective Earth radius model (Doviak and
Zrnić 1993, Eqs. 2.28b-d, pp. 21-22; Zhang et al. 2005, Eqs. 3-4, p. 32). Full
references are listed in :func:`grid_cones`.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = ["grid_cones"]

__doc__ = __doc__.format("\n   ".join(__all__))

import warnings

import numpy as np
import xarray as xr

from .._compat import renamed_arguments
from .._fields import node_dataset
from .._gridcf import aeqd_crs, crs_variable, georeference
from .._provenance import provenance

try:
    from . import _cone

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _cone = None
    HAS_COMPILED_KERNEL = False

# Earth radius [m] of the beam geometry (multiplied by k_e in the kernels):
# the commonly used mean radius, a radarx choice that matches xradar.georeference.
# Doviak and Zrnic (1993), Eqs. 2.28, only require the radius "a" of the Earth
# and the factor k_e (4/3 in a standard atmosphere, Eq. 2.28d).
EARTH_RADIUS = 6371000.0

# Default effective Earth radius factor of the standard atmosphere.
K_E = 4.0 / 3.0


def _antenna_kwargs(k_e):
    """
    Keyword of ``xradar.georeference.antenna_to_cartesian`` for ``k_e``.

    Nothing for the standard 4/3, which every xradar version uses by default.
    """
    if k_e == K_E:
        return {}
    import inspect

    from xradar.georeference import antenna_to_cartesian

    params = inspect.signature(antenna_to_cartesian).parameters
    for name in ("effective_radius_fraction", "ke"):
        if name in params:
            return {name: k_e}
    raise ImportError(  # pragma: no cover
        "this xradar version cannot change k_e; use engine='compiled' or upgrade"
    )


_VELOCITY_NAMES = ("VRAD", "VEL", "VR", "velocity", "mean_doppler_velocity")


def _is_velocity(name, attrs=None):
    """Whether a field is a radial velocity (by name or CF standard name)."""
    attrs = attrs or {}
    standard = str(attrs.get("standard_name", "")).lower()
    return (
        "doppler_velocity" in standard
        or "radial_velocity" in standard
        or str(name).startswith(_VELOCITY_NAMES)
    )


def _sweep_names(dtree):
    return sorted(
        (name for name in dtree.children if name.startswith("sweep")),
        key=lambda name: int(name.rsplit("_", 1)[-1]) if name[-1].isdigit() else name,
    )


def _sweep_dataset(dtree, name):
    """Sweep node as a Dataset including the radar site coordinates."""
    return node_dataset(dtree[name])


def _select_sweeps(sweeps, variable, tolerance=0.1):
    """
    Pick one sweep per elevation that contains ``variable``.

    Volumes with split cuts (e.g. NEXRAD) scan some elevations twice; the cut
    that reaches farthest is used. This rule and the default tolerance are
    radarx choices, not taken from a publication.

    Parameters
    ----------
    sweeps : list of xarray.Dataset
        Sweep datasets.
    variable : str
        Field to grid.
    tolerance : float, optional
        Elevations closer than this (degrees) count as the same sweep.

    Returns
    -------
    list of xarray.Dataset
        Sweeps sorted by elevation, lowest first.
    """
    best = {}
    for ds in sweeps:
        if variable not in ds:
            continue
        elevation = float(np.nanmedian(ds["elevation"].values))
        key = round(elevation / tolerance)
        reach = float(ds["range"].values[-1])
        if key not in best or reach > best[key][1]:
            best[key] = (ds, reach, elevation)
    return [ds for ds, _, _ in sorted(best.values(), key=lambda item: item[2])]


REPEATS = ("farthest", "mean", "latest", "nearest")


def _sweep_groups(sweeps, variable, repeats, analysis_time, tolerance=0.1):
    """
    Groups of sweeps of one elevation that contain ``variable``, lowest first.

    ``repeats="farthest"`` keeps one sweep per elevation (:func:`_select_sweeps`).
    The other choices keep all of them, ordered by priority within the group:
    ``"mean"`` by time, ``"latest"`` the latest first and ``"nearest"`` the one
    closest to ``analysis_time`` first; sweeps that tie come with the farthest
    reach first. Returns ``(groups, flat list of sweeps)``.
    """
    if repeats not in REPEATS:
        raise ValueError(f"repeats must be one of {REPEATS}, not {repeats!r}")
    if repeats == "farthest":
        flat = _select_sweeps(sweeps, variable, tolerance)
        return [1] * len(flat), flat
    t0 = np.datetime64(analysis_time, "ns")
    found = {}
    for ds in sweeps:
        if variable not in ds:
            continue
        elevation = float(np.nanmedian(ds["elevation"].values))
        time = _mean_time(ds)
        seconds = 0.0 if np.isnat(time) else float((time - t0) / np.timedelta64(1, "s"))
        reach = float(ds["range"].values[-1])
        priority = {"mean": seconds, "latest": -seconds, "nearest": abs(seconds)}
        found.setdefault(round(elevation / tolerance), []).append(
            (priority[repeats], -reach, elevation, ds)
        )
    ordered = sorted(
        (sorted(group, key=lambda m: m[:2]) for group in found.values()),
        key=lambda group: float(np.median([m[2] for m in group])),
    )
    return [len(g) for g in ordered], [m[3] for g in ordered for m in g]


def _reduce_groups(values, heights, groups, mean):
    """
    One cone per group of repeated sweeps from stacks of values and heights.

    The mean of the members that have a value (and of their heights), or the
    first member that has a value, as in the compiled kernel.
    """
    out_v, out_h = [], []
    start = 0
    for size in groups:
        v, h = values[start : start + size], heights[start : start + size]
        start += size
        if mean:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out_v.append(np.nanmean(v, axis=0))
                out_h.append(np.nanmean(h, axis=0))
            continue
        first_v, first_h = v[0].copy(), h[0].copy()
        for member_v, member_h in zip(v[1:], h[1:]):
            take = np.isnan(first_v) & (np.isfinite(member_v) | np.isnan(first_h))
            first_v = np.where(take, member_v, first_v)
            first_h = np.where(take, member_h, first_h)
        out_v.append(first_v)
        out_h.append(first_h)
    return np.stack(out_v), np.stack(out_h)


def _mean_time(ds):
    """Mean time of a sweep as datetime64[ns], or ``NaT``."""
    if "time" not in ds:
        return np.datetime64("NaT", "ns")
    t = np.ravel(ds["time"].values).astype("datetime64[ns]")
    t = t[~np.isnat(t)]
    if t.size == 0:
        return np.datetime64("NaT", "ns")
    t0 = t.min()
    return t0 + (t - t0).astype(np.int64).mean().astype("timedelta64[ns]")


def _motion_pair(motion):
    """``(u, v)`` in m/s from the ``motion`` argument."""
    try:
        u, v = (float(m) for m in motion)
    except (TypeError, ValueError) as err:
        raise ValueError(
            "motion must be a pair (u, v) of eastward and northward speed in m/s"
        ) from err
    return u, v


def _sweep_shifts(selected, motion, time, default_time):
    """
    Displacement (dx, dy) in metres of every sweep to the analysis time.

    A cell at ``x`` at the analysis time ``time`` holds what the sweep saw
    at ``x - (u, v) (time - t_sweep)``, so the cone is sampled there: the
    storm is assumed to move steadily with ``(u, v)``.
    """
    u, v = _motion_pair(motion)
    t0 = np.datetime64(default_time if time is None else time, "ns")
    shifts = []
    for ds in selected:
        dt = (t0 - _mean_time(ds)) / np.timedelta64(1, "s")
        dt = 0.0 if np.isnan(dt) else float(dt)
        shifts.append((u * dt, v * dt))
    return shifts


def _volume_time(sweeps):
    """Mean time of all rays of the volume as datetime64[ns], or ``NaT``."""
    times = [np.ravel(ds["time"].values) for ds in sweeps if "time" in ds]
    if not times:
        return np.datetime64("NaT", "ns")
    t = np.concatenate(times).astype("datetime64[ns]")
    t = t[~np.isnat(t)]
    if t.size == 0:
        return np.datetime64("NaT", "ns")
    t0 = t.min()
    return t0 + (t - t0).astype(np.int64).mean().astype("timedelta64[ns]")


def _sweep_arrays(ds, variable):
    """Contiguous float64 arrays (data, azimuth, elevation, range) of a sweep."""
    ray_dim = (
        ds[variable].dims[0]
        if ds[variable].dims[0] != "range"
        else ds[variable].dims[1]
    )
    data = ds[variable].transpose(ray_dim, "range").values
    return (
        np.ascontiguousarray(data, dtype=np.float64),
        np.ascontiguousarray(ds["azimuth"].values, dtype=np.float64),
        np.ascontiguousarray(
            np.broadcast_to(ds["elevation"].values, ds["azimuth"].shape),
            dtype=np.float64,
        ),
        np.ascontiguousarray(ds["range"].values, dtype=np.float64),
    )


def _cone_numpy(
    data, azimuth, elevation, rng, s, az, site_altitude, max_gap, min_weight, k_e=K_E
):
    """
    Value and beam height of one sweep at ground distance ``s``, azimuth ``az``.

    NumPy reference of the compiled kernel: bilinear interpolation of the four
    surrounding gates (weights from azimuth and ground distance), the beam
    height from the effective Earth radius model of xradar (Doviak and Zrnic
    1993, Eq. 2.28b, ``k_e`` = 4/3 unless given); ``max_gap`` and
    ``min_weight`` are radarx choices.
    """
    from xradar.georeference import antenna_to_cartesian

    el_med = float(np.median(elevation))
    kw = _antenna_kwargs(k_e)
    gx, gy, _ = antenna_to_cartesian(
        rng, 0.0, el_med, site_altitude=site_altitude, **kw
    )
    ground = np.hypot(gx, gy)
    ir = np.interp(s, ground, np.arange(rng.size), left=np.nan, right=np.nan)

    ray_az = np.mod(azimuth, 360.0)
    order = np.argsort(ray_az, kind="stable")
    a = ray_az[order]
    keep = np.r_[True, np.diff(a) > 1e-6]
    order, a = order[keep], a[keep]
    a_ext = np.r_[a[-1] - 360.0, a, a[0] + 360.0]
    ray_ext = np.r_[order[-1], order, order[0]]
    spacing = np.median(np.diff(a))

    j = np.clip(np.searchsorted(a_ext, az, side="right") - 1, 0, a_ext.size - 2)
    da = a_ext[j + 1] - a_ext[j]
    fa = (az - a_ext[j]) / da
    too_wide = da > max_gap * spacing

    inside = np.isfinite(ir)
    i0 = np.clip(np.floor(np.where(inside, ir, 0)).astype(int), 0, rng.size - 2)
    fr = np.where(inside, ir - i0, 0.0)
    r0, r1 = ray_ext[j], ray_ext[j + 1]
    _, _, gate_z = antenna_to_cartesian(
        rng[None, :], 0.0, elevation[:, None], site_altitude=site_altitude, **kw
    )

    num = np.zeros(s.shape)
    den = np.zeros(s.shape)
    height = np.zeros(s.shape)
    for ray, wa in ((r0, 1 - fa), (r1, fa)):
        for gate, wr in ((i0, 1 - fr), (i0 + 1, fr)):
            weight = wa * wr
            value = data[ray, gate]
            valid = np.isfinite(value)
            num += np.where(valid, weight * value, 0.0)
            den += np.where(valid, weight, 0.0)
            height += weight * gate_z[ray, gate]
    value = np.where(den >= min_weight, num / np.where(den > 0, den, 1.0), np.nan)
    unusable = ~inside | too_wide
    value[unusable] = np.nan
    height[unusable] = np.nan
    return value, height


def _grid_numpy(
    x,
    y,
    z,
    arrays,
    site_altitude,
    max_gap,
    min_weight,
    fill_below,
    k_e=K_E,
    fill_depth=np.inf,
    shifts=None,
    groups=None,
    group_mean=False,
):
    """NumPy implementation of the compiled kernel (same results)."""
    X, Y = np.meshgrid(x, y)
    s = np.hypot(X, Y)
    az = np.mod(np.degrees(np.arctan2(X, Y)), 360.0)
    return _columns_numpy(
        s,
        az,
        z,
        arrays,
        site_altitude,
        max_gap,
        min_weight,
        fill_below,
        k_e,
        fill_depth,
        shifts,
        groups,
        group_mean,
    )


def _displaced(s, az, dx, dy):
    """Ground distance and azimuth (deg) of the columns shifted by (-dx, -dy)."""
    if dx == 0.0 and dy == 0.0:
        return s, az
    a = np.radians(az)
    px = s * np.sin(a) - dx
    py = s * np.cos(a) - dy
    return np.hypot(px, py), np.mod(np.degrees(np.arctan2(px, py)), 360.0)


def _columns_numpy(
    s,
    az,
    z,
    arrays,
    site_altitude,
    max_gap,
    min_weight,
    fill_below,
    k_e=K_E,
    fill_depth=np.inf,
    shifts=None,
    groups=None,
    group_mean=False,
):
    """
    Cone gridding of columns at ground distance ``s`` and azimuth ``az``.

    ``s`` and ``az`` (degrees) have the shape of the output columns; the
    result is float32 on ``(z,) + s.shape``. Between the two cones that
    bracket a level the value is interpolated linearly in beam height (compare
    the vertical interpolation of Zhang et al. 2005, Eqs. 5-7, which
    interpolates in elevation angle). With a finite ``fill_depth`` the lowest
    cone fills at most that many metres below itself and not below the radar
    altitude. ``shifts`` gives a displacement ``(dx, dy)`` in metres per
    sweep: the cone is sampled at the column position minus the shift. With
    ``groups`` (sizes of runs of consecutive sweeps) every run is merged into one
    cone first, by its mean or by its first member that has a value.
    """
    shifts = shifts if shifts is not None else [(0.0, 0.0)] * len(arrays)
    values, heights = zip(
        *(
            _cone_numpy(
                *sweep,
                *_displaced(s, az, dx, dy),
                site_altitude,
                max_gap,
                min_weight,
                k_e,
            )
            for sweep, (dx, dy) in zip(arrays, shifts)
        )
    )
    V = np.stack(values)
    H = np.stack(heights)
    if groups is not None:
        V, H = _reduce_groups(V, H, groups, group_mean)
    nk = V.shape[0]
    # Only cones that reach a column take part (low tilts can start farther
    # out than high ones): move missing cones to the end of each column.
    H_search = np.where(np.isfinite(H), H, np.inf)
    order = np.argsort(H_search, axis=0, kind="stable")
    H_search = np.take_along_axis(H_search, order, 0)
    H = np.where(np.isfinite(H_search), H_search, np.nan)
    V = np.take_along_axis(V, order, 0)
    out = np.full((z.size,) + s.shape, np.nan, dtype=np.float32)
    for iz, level in enumerate(z):
        k = (H_search <= level).sum(axis=0) - 1
        lo = np.clip(k, 0, max(nk - 2, 0))
        hi = np.minimum(lo + 1, nk - 1)
        h0 = np.take_along_axis(H, lo[None], 0)[0]
        h1 = np.take_along_axis(H, hi[None], 0)[0]
        v0 = np.take_along_axis(V, lo[None], 0)[0]
        v1 = np.take_along_axis(V, hi[None], 0)[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            t = (level - h0) / (h1 - h0)
        result = np.where(
            (k >= 0) & (k <= nk - 2) & np.isfinite(h1), v0 + t * (v1 - v0), np.nan
        )
        if fill_below:
            fill = (k < 0) & np.isfinite(H[0])
            if np.isfinite(fill_depth):
                fill &= (level >= H[0] - fill_depth) & (level >= site_altitude)
            result = np.where(fill, V[0], result)  # lowest present cone
        out[iz] = result
    return out


def _lonlat_axes(x, y, latitude, longitude):
    """Longitude along x (at y = 0) and latitude along y (at x = 0)."""
    import pyproj

    aeqd = pyproj.Proj(proj="aeqd", lat_0=latitude, lon_0=longitude, datum="WGS84")
    lon, _ = aeqd(x, np.zeros_like(x), inverse=True)
    _, lat = aeqd(np.zeros_like(y), y, inverse=True)
    return np.asarray(lon), np.asarray(lat)


@renamed_arguments(dtree="obj")
@provenance(
    "Cone gridding of radar volumes, radarx's own method",
    extra_refs=("doviak-zrnic-1993",),
)
def grid_cones(
    obj,
    data_vars=None,
    x=None,
    y=None,
    z=None,
    *,
    fill_below=False,
    fill_depth=None,
    fill_velocity=False,
    k_e=K_E,
    motion=None,
    time=None,
    repeats="farthest",
    max_gap=2.0,
    min_weight=0.5,
    n_threads=None,
    engine="auto",
):
    """
    Grid a radar volume onto a Cartesian grid with cone interpolation.

    Parameters
    ----------
    obj : xarray.DataTree
        Radar volume with ``sweep_*`` groups and the radar site
        ``latitude``, ``longitude`` and ``altitude``, e.g. from xradar.
    data_vars : str or list of str, optional
        Fields to grid. By default every field present in the lowest sweep
        on ``(azimuth, range)``.
    x, y : array-like
        Target coordinates east and north of the radar, in metres.
    z : array-like
        Target heights above sea level, in metres.
    fill_below : bool, optional
        Fill levels below the lowest sweep with the lowest sweep's value
        (pseudo-CAPPI). Default ``False`` leaves them empty.
    fill_depth : float, optional
        Fill at most this many metres below the lowest sweep, and never below
        the radar altitude. Default ``None`` (or ``numpy.inf``) fills every
        level below the lowest sweep, as before; at long range that spreads the
        lowest beam (2.5 km up at 150 km) down to the ground. Only used with
        ``fill_below=True``.
    fill_velocity : bool, optional
        Also fill radial velocity fields. Default ``False``: a velocity
        is a projection on the beam direction, which differs from level to
        level, so it is never copied downward. ``True`` restores the old
        behaviour.
    k_e : float, optional
        Effective Earth radius factor of the beam height, ``a_e = k_e a``
        (Doviak and Zrnić 1993, Eq. 2.28). Default 4/3, the standard
        atmosphere. :func:`radarx.fundamentals.geometry.k_e_from_refractivity`
        derives it from a sounding or reanalysis profile.
    motion : tuple of float, optional
        Storm motion ``(u, v)`` in m/s. When given, every sweep is moved from
        its own mean time to the analysis time ``time`` before the vertical
        interpolation (Gal-Chen 1982 [4]_, equations not checked against the
        paper): low and high tilts of a volume are minutes apart, and a storm
        moving at 15 m/s travels 4.5 km in five minutes, so an echo seen by
        the lowest and the highest tilt would otherwise appear at positions
        that differ by that distance. The motion is steady and the same
        everywhere; the displacement is applied per sweep, not per ray.
        Default ``None``: no advection, as before.
    time : datetime-like, optional
        Analysis time for ``motion``. Default: the mean time of the volume.
        Also the ``time`` of the result when given.
    repeats : {"farthest", "mean", "latest", "nearest"}, optional
        What to do with several sweeps of the same elevation (within 0.1
        degrees). ``"farthest"`` (default, as before) takes the one that reaches
        farthest, which sorts out the split cuts of NEXRAD. The other choices use
        all of them, for volumes that repeat the lowest tilt (SAILS and
        MESO-SAILS of NEXRAD, or any other repeated sweep): ``"mean"`` averages
        the repeats that have a value at the column, ``"latest"`` takes the
        latest repeat that has a value and ``"nearest"`` the one closest in time
        to ``time`` (default: the mean time of the volume). With ``motion`` every
        repeat is moved to the analysis time before they are combined, so
        ``"mean"`` averages the echo at one position instead of smearing it.
        Merging is by elevation, not by time: repeated sweeps of the
        highest tilts are merged as well.
    max_gap : float, optional
        Do not interpolate between rays more than this many median ray
        spacings apart (sector edges, missing rays). Default 2.
    min_weight : float, optional
        Minimum share of the bilinear weight that must come from valid gates
        for a sweep value to be defined. Default 0.5, so values do not spread
        more than half a gate into missing data.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled
        kernel and falls back to NumPy.

    Returns
    -------
    xarray.Dataset
        Gridded fields on ``(z, y, x)`` as float32, georeferenced as in CF:
        the azimuthal equidistant CRS centred on the radar as the scalar
        ``crs_wkt`` (``grid_mapping`` of every field), 2-D ``lat(y, x)`` and
        ``lon(y, x)`` computed from it, ``standard_name``, ``units``, ``axis``
        and ``positive`` on ``x``, ``y``, ``z``, the radar site and the mean
        time of the volume as a scalar ``time`` coordinate, so grids of several
        volumes stack along time with ``xr.concat(grids, "time")``. Before
        radarx 0.5 ``lat`` was 1-D along ``y`` and ``lon`` along ``x``, which is
        wrong away from the radar (633 m in the corner of a 120 km wide grid).

    Raises
    ------
    ValueError
        If no sweep contains a requested variable, or the grid is missing.
    ImportError
        If ``engine="compiled"`` and the compiled kernel is not available.

    Notes
    -----
    Beam height. The height of a gate above sea level is computed with the
    effective Earth radius model, h = [r^2 + (k_e a)^2 + 2 r k_e a
    sin(theta_e)]^(1/2) - k_e a with k_e = 4/3 by default (Doviak and Zrnić
    1993 [2]_, Eq. 2.28b, p. 21; the same model is Eqs. 3-4, p. 32, of Zhang
    et al. 2005 [3]_). Other values of ``k_e`` describe sub- or super-refractive
    conditions.
    The Earth radius ``a`` (6371 km) is a radarx choice.

    What is taken from the literature, and what is not. Bilinear
    interpolation within a sweep and linear interpolation in height between
    the two sweeps around a cell are standard operations, here combined as
    described in the module docstring; the vertical step is analogous to the
    VI scheme of Zhang et al. (2005) [3]_ (Eqs. 5-7) but interpolates in beam
    height instead of elevation angle. ``max_gap`` (2 median ray spacings),
    ``min_weight`` (0.5), the 0.1 degree tolerance with which sweeps of equal
    elevation are identified, and the rule that cells outside the bracketing
    cones stay empty are radarx choices, not values from a paper.

    Repeated sweeps. ``repeats="mean"``, ``"latest"`` and ``"nearest"`` are
    radarx's own constructions; no publication was used for them. The repeats of
    one elevation (within 0.1 degrees) become one cone before the vertical
    interpolation, so a repeat adds no extra layer and no zero-thickness
    interval. Merging by the mean after moving each repeat to the analysis time
    assumes the steady motion of ``motion``.

    Velocity gridding. Interpolating radial velocity from the polar grid
    (here and in the multi-Doppler input) can leave periodic gridding
    artifacts that map into spurious vertical velocities in variational
    retrievals (Collis et al. 2010 [1]_); their mixed-order linear
    interpolation is not implemented.

    References
    ----------
    .. [1] Collis, S., A. Protat, and K.-S. Chung, 2010: The effect of radial
       velocity gridding artifacts on variationally retrieved vertical
       velocities. *J. Atmos. Oceanic Technol.*, **27**, 1239-1246,
       https://doi.org/10.1175/2010JTECHA1402.1
    .. [2] Doviak, R. J., and D. S. Zrnić, 1993: *Doppler Radar and Weather
       Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
       DOI).
    .. [3] Zhang, J., K. Howard, and J. J. Gourley, 2005: Constructing
       three-dimensional multiple-radar reflectivity mosaics: Examples of
       convective storms and stratiform rain echoes. *J. Atmos. Oceanic
       Technol.*, **22**, 30-42, https://doi.org/10.1175/JTECH-1689.1
    .. [4] Gal-Chen, T., 1982: Errors in fixed and moving frame of references:
       Applications for conventional and Doppler radar analysis. *J. Atmos.
       Sci.*, **39**, 2279-2300,
       https://doi.org/10.1175/1520-0469(1982)039<2279:EIFAMF>2.0.CO;2

    Examples
    --------
    >>> import numpy as np  # doctest: +SKIP
    >>> x = y = np.arange(-150e3, 150e3 + 1, 500.0)  # doctest: +SKIP
    >>> z = np.arange(500.0, 15e3 + 1, 500.0)  # doctest: +SKIP
    >>> grid = radarx.grid.grid_cones(dtree, "DBZH", x, y, z)  # doctest: +SKIP
    """
    use_compiled = _use_compiled(engine)
    if x is None or y is None or z is None:
        raise ValueError("x, y and z target coordinates are required.")
    x, y, z = (np.asarray(a, dtype=np.float64) for a in (x, y, z))
    sweeps, site, site_coords = _load_volume(obj)
    if data_vars is None:
        data_vars = [
            name for name, da in sweeps[0].data_vars.items() if {"range"} < set(da.dims)
        ]
    elif isinstance(data_vars, str):
        data_vars = [data_vars]

    k_e = _check_k_e(k_e)
    fill_depth = _fill_depth(fill_depth)
    volume_time = _volume_time(sweeps)
    gridded = {}
    for variable in data_vars:
        sizes, selected = _sweep_groups(
            sweeps,
            variable,
            repeats,
            volume_time if time is None else time,
        )
        if not selected:
            raise ValueError(f"No sweep contains {variable!r}.")
        arrays = [_sweep_arrays(ds, variable) for ds in selected]
        attrs = dict(selected[0][variable].attrs)
        shifts = (
            _sweep_shifts(selected, motion, time, volume_time)
            if motion is not None
            else None
        )
        options = dict(
            max_gap=max_gap,
            min_weight=min_weight,
            fill_below=bool(
                fill_below and (fill_velocity or not _is_velocity(variable, attrs))
            ),
            k_e=k_e,
            fill_depth=fill_depth,
            shifts=shifts,
        )
        if repeats != "farthest":
            options["groups"] = sizes
            options["group_mean"] = repeats == "mean"
        site_altitude = site.get("altitude", 0.0)
        if use_compiled:
            field = _grid_compiled(x, y, z, arrays, site_altitude, n_threads, **options)
        else:
            field = _grid_numpy(x, y, z, arrays, site_altitude, **options)
        gridded[variable] = (("z", "y", "x"), field, attrs)
    out = _to_dataset(gridded, x, y, z, sweeps, site, obj.attrs)
    out.attrs["k_e"] = k_e
    if motion is not None:
        u, v = _motion_pair(motion)
        out = out.assign_coords(
            time=volume_time if time is None else np.datetime64(time, "ns")
        )
        out.attrs["advection"] = (
            f"each sweep moved from its own time to the analysis time with "
            f"u = {u} m/s, v = {v} m/s"
        )
    return _georeferenced(out, site_coords, site)


def _check_k_e(k_e):
    """``k_e`` as a positive float."""
    k_e = float(k_e)
    if not np.isfinite(k_e) or k_e <= 0:
        raise ValueError(f"k_e must be a positive number, not {k_e!r}")
    return k_e


def _fill_depth(depth):
    """``fill_depth`` as a float, ``numpy.inf`` for no limit."""
    if depth is None:
        return np.inf
    depth = float(depth)
    if depth < 0:
        raise ValueError(f"fill_depth must not be negative, not {depth!r}")
    return depth


def _use_compiled(engine):
    """Whether to run the compiled kernel for the requested ``engine``."""
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled cone-gridding kernel is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


def _load_volume(dtree):
    """
    Sweep datasets of ``dtree`` and the radar site location.

    Returns
    -------
    sweeps : list of xarray.Dataset
    site : dict
        ``latitude``, ``longitude`` (and ``altitude``) as floats.
    site_coords : dict
        The same as scalar DataArrays with their attributes.
    """
    names = _sweep_names(dtree)
    if not names:
        raise ValueError("No sweep groups found in DataTree.")
    sweeps = [_sweep_dataset(dtree, name) for name in names]
    first = sweeps[0]
    root = dtree.root.to_dataset()
    site_coords = {}
    for key in ("latitude", "longitude", "altitude"):
        # sweeps usually inherit the site; older xarray only exposes it on the root
        source = first if key in first else root if key in root else None
        if source is not None:
            site_coords[key] = source[key].reset_coords(drop=True)
    if "latitude" not in site_coords or "longitude" not in site_coords:
        raise ValueError("The volume needs the radar site 'latitude' and 'longitude'.")
    site = {key: float(value) for key, value in site_coords.items()}
    return sweeps, site, site_coords


def _grid_compiled(x, y, z, arrays, site_altitude, n_threads, **options):
    """Run the C++ kernel on per-sweep (data, azimuth, elevation, range) arrays."""
    data, azimuth, elevation, rng = (list(a) for a in zip(*arrays))
    shifts = options.pop("shifts", None)
    if shifts is not None:
        options["shift"] = [float(c) for pair in shifts for c in pair]
    return _cone.grid_cones(
        x,
        y,
        z,
        data,
        azimuth,
        elevation,
        rng,
        site_altitude,
        earth_radius=EARTH_RADIUS,
        n_threads=int(n_threads or 0),
        **options,
    )


def _crs_wkt(latitude, longitude):
    """The grid's azimuthal equidistant CRS as a CF ``crs_wkt`` coordinate."""
    return crs_variable(latitude, longitude)


def _to_dataset(gridded, x, y, z, sweeps, site, attrs):
    """Wrap gridded fields with coordinates, site, CRS, time and attributes."""
    out = xr.Dataset(
        gridded,
        coords={
            "z": ("z", z, {"units": "m", "long_name": "height above sea level"}),
            "y": ("y", y, {"units": "m", "long_name": "distance north of the radar"}),
            "x": ("x", x, {"units": "m", "long_name": "distance east of the radar"}),
        },
    )
    times = [np.ravel(ds["time"].values) for ds in sweeps if "time" in ds]
    if times:
        out = out.assign_coords(
            time=xr.DataArray(np.concatenate(times).astype("datetime64[ns]")).mean()
        )
    out.attrs = dict(attrs)
    out.attrs["radar_name"] = out.attrs.get("instrument_name", "")
    out.attrs["gridding_method"] = "cone"
    return out


def _georeferenced(out, site_coords, site):
    """The grid with the radar site, the CRS and the CF georeferencing."""
    out = out.assign_coords(site_coords)
    mapping = aeqd_crs(site["latitude"], site["longitude"]).to_cf()
    return georeference(out, mapping)
