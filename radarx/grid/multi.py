#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx Multi-Radar Gridding
===========================

Grid several radars, fixed or mobile, onto one Cartesian grid, merge them and
reconcile their calibrations.

:func:`grid_radars` cone-grids every radar (see :mod:`radarx.grid.cone`) onto
a shared grid: one origin, one azimuthal equidistant projection, heights above
sea level. Each radar keeps its own fields along a ``radar`` dimension,
together with the beam geometry from that radar to every cell (azimuth,
elevation and slant range), which is what a multi-Doppler wind retrieval
needs. Optionally the radars are moved to a common analysis time along the
storm motion (:func:`radarx.retrieve.advect`), calibrated against a reference
radar (:func:`network_bias`) and merged into one field
(:func:`merge_radars`).

All heavy loops (cone gridding of every radar and field in one call, the
per-cell beam geometry, the weighted merge and the pairwise comparison
histograms) run in compiled, multithreaded C++ kernels; if they are not
available an equivalent NumPy implementation is used.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = ["grid_radars", "merge_radars", "network_bias"]

__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np
import xarray as xr

from .._provenance import provenance
from .._registry import accessor_method
from . import cone

try:
    from . import _multi

    HAS_COMPILED_KERNEL = cone.HAS_COMPILED_KERNEL
except ImportError:  # pragma: no cover - depends on the build
    _multi = None
    HAS_COMPILED_KERNEL = False

EARTH_RADIUS = cone.EARTH_RADIUS
_LN_005 = np.log(0.005)
_GEOMETRY = ("azimuth", "elevation", "range")


def _use_compiled(engine):
    """Whether to run the compiled kernels for the requested ``engine``."""
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled multi-radar kernels are not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def _aeqd(latitude, longitude):
    import pyproj

    return pyproj.Proj(proj="aeqd", lat_0=latitude, lon_0=longitude, datum="WGS84")


def _column_geometry(x, y, origin, sites, use_compiled=False, n_threads=None):
    """
    Column geometry of the shared grid as seen from every radar.

    The compiled kernel solves the geodesics on the WGS84 ellipsoid with
    Vincenty's (1975) direct and inverse formulae (https://doi.org/10.1179/
    sre.1975.23.176.88; equation numbers not checked against the paper). The
    NumPy fallback uses PROJ's azimuthal equidistant projections
    (:func:`_column_geometry_numpy`). The WGS84 ellipsoid constants
    (a = 6378137 m, 1/f = 298.257223563) are its defining parameters (NIMA
    TR8350.2).

    Returns
    -------
    dict
        ``lon``, ``lat`` (ny, nx); per radar lists ``ground`` (ground
        distance, m), ``antenna_azimuth`` (azimuth of the column seen from the
        radar, degrees) and ``azimuth`` (direction of the beam at the column,
        degrees clockwise from the grid's +y axis); ``radar_x``, ``radar_y``.
    """
    if not use_compiled:
        return _column_geometry_numpy(x, y, origin, sites)
    lat, lon, ground, antenna, heading = _multi.column_geometry(
        x,
        y,
        origin[0],
        origin[1],
        [s["latitude"] for s in sites],
        [s["longitude"] for s in sites],
        n_threads=int(n_threads or 0),
    )
    out = dict(lon=lon, lat=lat, ground=list(ground), antenna_azimuth=list(antenna))
    out["azimuth"] = list(heading)
    out["radar_x"], out["radar_y"] = _radar_positions(origin, sites)
    return out


def _radar_positions(origin, sites):
    """Radar positions (m) in the grid's azimuthal equidistant projection."""
    grid = _aeqd(*origin)
    xy = [grid(site["longitude"], site["latitude"]) for site in sites]
    return [float(a) for a, _ in xy], [float(b) for _, b in xy]


def _column_geometry_numpy(x, y, origin, sites, step=100.0):
    """
    Column geometry from PROJ azimuthal equidistant projections.

    Parameters
    ----------
    x, y : numpy.ndarray
        Grid axes (m) in the azimuthal equidistant projection at ``origin``.
    origin : tuple of float
        ``(latitude, longitude)`` of the grid origin.
    sites : list of dict
        Radar sites with ``latitude`` and ``longitude``.
    step : float, optional
        Distance (m) along the beam used for the beam direction at the cell.

    Returns
    -------
    dict
        ``lon``, ``lat`` (ny, nx); per radar lists ``ground`` (ground
        distance, m), ``antenna_azimuth`` (azimuth of the column seen from the
        radar, degrees) and ``azimuth`` (direction of the beam at the column,
        degrees clockwise from the grid's +y axis); ``radar_x``, ``radar_y``.
    """
    grid = _aeqd(*origin)
    X, Y = np.meshgrid(x, y)
    lon, lat = (np.asarray(a) for a in grid(X, Y, inverse=True))
    out = dict(lon=lon, lat=lat, ground=[], antenna_azimuth=[], azimuth=[])
    for site in sites:
        proj = _aeqd(site["latitude"], site["longitude"])
        xr_, yr_ = (np.asarray(a) for a in proj(lon, lat))
        s = np.hypot(xr_, yr_)
        antenna = np.mod(np.degrees(np.arctan2(xr_, yr_)), 360.0)
        # a point a little farther along the same beam, back in the grid frame
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.where(s > 0, (s + step) / s, 1.0)
        far_x = np.where(s > 0, xr_ * scale, 0.0)
        far_y = np.where(s > 0, yr_ * scale, step)
        far_lon, far_lat = proj(far_x, far_y, inverse=True)
        gx, gy = (np.asarray(a) for a in grid(far_lon, far_lat))
        heading = np.mod(np.degrees(np.arctan2(gx - X, gy - Y)), 360.0)
        out["ground"].append(np.ascontiguousarray(s))
        out["antenna_azimuth"].append(np.ascontiguousarray(antenna))
        out["azimuth"].append(heading)
    out["radar_x"], out["radar_y"] = _radar_positions(origin, sites)
    return out


def _cell_angles(ground, z, site_altitude, earth_radius=EARTH_RADIUS):
    """
    Slant range, local and antenna elevation (deg) of cells (4/3 Earth).

    The radar is at distance ``a = R + site_altitude`` and the cell at
    ``b = R + z`` from the centre of an Earth of effective radius
    ``R = 4/3 * earth_radius``, separated by the central angle ``ground / R``.
    The slant range is the law of cosines in that triangle; the elevation
    angles follow from the same triangle. This is the geometry behind the
    effective Earth radius model, Doviak and Zrnic (1993), Eqs. 2.28b-d,
    pp. 21-22 (k_e = 4/3, Eq. 2.28d), solved for range and elevation instead
    of height; the algebra is radarx's own. The Earth radius of 6371 km is a
    radarx choice.
    """
    R = earth_radius * 4.0 / 3.0
    a = R + site_altitude
    b = R + np.asarray(z, dtype=np.float64).reshape((-1,) + (1,) * np.ndim(ground))
    th = np.asarray(ground) / R
    c, s = np.cos(th), np.sin(th)
    rng = np.sqrt(np.maximum(0.0, a * a + b * b - 2.0 * a * b * c))
    local = np.degrees(np.arctan2(b - a * c, a * s))
    antenna = np.degrees(np.arctan2(b * c - a, b * s))
    return rng, local, antenna


def _geometry_numpy(ground, z, site_altitude, earth_radius=EARTH_RADIUS):
    """NumPy implementation of ``_multi.beam_geometry``."""
    rng, el = [], []
    for g, alt in zip(ground, site_altitude):
        r, local, _ = _cell_angles(g, z, alt, earth_radius)
        rng.append(r.astype(np.float32))
        el.append(local.astype(np.float32))
    return np.stack(rng), np.stack(el)


def _beam_weight_numpy(e, elevations, beamwidth):
    """
    Elevation weight of Lakshmanan et al. (2006): ``exp(|a|^3 ln 0.005)``.

    ``a`` is the angular distance of the cell from the nearest beam axis as a
    fraction of the larger of the beamwidth and the spacing to the next beam
    on that side. This is Eq. (6) of Lakshmanan et al. (2006), p. 808:
    ``delta_e = exp[alpha^3 ln(0.005)]`` with ``alpha = (e - theta_i) /
    (|theta_(i+-1) - theta_i| V b_i)``, where ``V`` is the maximum operator,
    ``b_i`` the beamwidth and ``theta_i`` the elevation of the beam centre.
    The paper states the weight is 1 at the beam centre, 0.5 at half a
    beamwidth and below 0.01 at a beamwidth (the exact values of the formula
    are 0.516 at ``alpha = 0.5`` and 0.005 at ``alpha = 1``). Which neighbour
    beam is used (the one on the side of the cell) follows the paper's
    ``theta_(i+-1)``; the choice of the nearest axis is radarx's.
    """
    el = np.sort(np.asarray(elevations, dtype=np.float64))
    n = el.size
    if n == 0:
        return np.ones_like(e)
    k = np.searchsorted(el, e, side="left")
    lower = el[np.clip(k - 1, 0, n - 1)]
    upper = el[np.clip(k, 0, n - 1)]
    i = np.where(
        k == 0, 0, np.where(k == n, n - 1, np.where(e - lower <= upper - e, k - 1, k))
    )
    axis = el[i]
    nb = np.where(e >= axis, i + 1, i - 1)
    valid_nb = (nb >= 0) & (nb < n)
    gap = np.abs(el[np.clip(nb, 0, n - 1)] - axis)
    spacing = np.where(valid_nb, np.maximum(beamwidth, gap), beamwidth)
    with np.errstate(invalid="ignore", divide="ignore"):
        alpha = np.abs(e - axis) / spacing
    return np.where(spacing > 0, np.exp(alpha**3 * _LN_005), 1.0)


def _merge_numpy(
    values,
    ground,
    z,
    site_altitude,
    elevations,
    beamwidth,
    time_offset,
    range_scale,
    time_scale,
    earth_radius=EARTH_RADIUS,
):
    """
    NumPy implementation of ``_multi.merge``.

    Weight of radar ``r``: ``exp(-(range/range_scale)^2)`` (the Gaussian
    distance weight of Zhang et al. 2005, Fig. 15, p. 41, with R = 50 km) times
    the elevation weight of Lakshmanan et al. (2006), Eq. (6), times
    ``exp(-(dt/time_scale)^2)`` (radarx's own time weight, not Eq. 7 of
    Lakshmanan et al. 2006; see :func:`merge_radars`).
    """
    num = np.zeros(values.shape[1:])
    den = np.zeros(values.shape[1:])
    for r in range(values.shape[0]):
        rng, _, antenna = _cell_angles(ground[r], z, site_altitude[r], earth_radius)
        w = _beam_weight_numpy(antenna, elevations[r], beamwidth[r])
        if range_scale > 0:
            w = w * np.exp(-((rng / range_scale) ** 2))
        if time_scale > 0:
            w = w * np.exp(-((time_offset[r] / time_scale) ** 2))
        v = values[r].astype(np.float64)
        ok = np.isfinite(v)
        num += np.where(ok, w * np.where(ok, v, 0.0), 0.0)
        den += np.where(ok, w, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        merged = np.where(den > 0, num / den, np.nan)
    return merged.astype(np.float32), den.astype(np.float32)


def _pair_histograms_numpy(values, rng, max_range_ratio, bin_width, max_difference):
    """NumPy implementation of ``_multi.pair_histograms``."""
    nr = values.shape[0]
    nbin = int(np.ceil(2.0 * max_difference / bin_width))
    pairs = [(i, j) for i in range(nr) for j in range(i + 1, nr)]
    counts = np.zeros((len(pairs), nbin), dtype=np.int64)
    sums = np.zeros(len(pairs))
    sumsq = np.zeros(len(pairs))
    for p, (i, j) in enumerate(pairs):
        ok = np.isfinite(values[i]) & np.isfinite(values[j])
        if max_range_ratio > 0:
            lo = np.minimum(rng[i], rng[j]).astype(np.float64)
            hi = np.maximum(rng[i], rng[j]).astype(np.float64)
            ok &= hi <= max_range_ratio * lo
        d = values[i][ok].astype(np.float64) - values[j][ok].astype(np.float64)
        b = np.floor((d + max_difference) / bin_width).astype(np.int64)
        keep = (b >= 0) & (b < nbin)
        counts[p] = np.bincount(b[keep], minlength=nbin)[:nbin]
        sums[p] = d[keep].sum()
        sumsq[p] = (d[keep] ** 2).sum()
    return counts, sums, sumsq


# ---------------------------------------------------------------------------
# reading the radars
# ---------------------------------------------------------------------------


def _radar_name(dtree, index):
    name = dtree.attrs.get("instrument_name") or dtree.root.attrs.get("instrument_name")
    if not name or str(name) == "None":
        return f"radar_{index}"
    return str(name)


def _unique(names):
    seen = {}
    out = []
    for name in names:
        if name in seen:
            seen[name] += 1
            out.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            out.append(name)
    return out


def _volume_time(sweeps):
    times = [np.ravel(ds["time"].values) for ds in sweeps if "time" in ds]
    if not times:
        return np.datetime64("NaT", "ns")
    t = np.concatenate(times).astype("datetime64[ns]")
    t = t[~np.isnat(t)]
    if t.size == 0:
        return np.datetime64("NaT", "ns")
    t0 = t.min()
    return t0 + (t - t0).astype(np.int64).mean().astype("timedelta64[ns]")


def _task_arrays(ds, name, rhohv, rhohv_min):
    """
    Data (float32 or float64 as stored, masked where the co-polar correlation
    is low) and float64 azimuth, elevation and range of one sweep.
    """
    da = ds[name]
    ray_dim = da.dims[0] if da.dims[0] != "range" else da.dims[1]
    data = da.transpose(ray_dim, "range").values
    if data.dtype not in (np.float32, np.float64):
        data = data.astype(np.float64)
    if rhohv_min is not None and name != rhohv and rhohv in ds:
        rho = ds[rhohv].transpose(ray_dim, "range").values
        data = np.where(rho < rhohv_min, data.dtype.type(np.nan), data)
    return (
        np.ascontiguousarray(data),
        np.ascontiguousarray(ds["azimuth"].values, dtype=np.float64),
        np.ascontiguousarray(
            np.broadcast_to(ds["elevation"].values, ds["azimuth"].shape),
            dtype=np.float64,
        ),
        np.ascontiguousarray(ds["range"].values, dtype=np.float64),
    )


def _sweep_elevations(selected):
    return [float(np.nanmedian(ds["elevation"].values)) for ds in selected]


# ---------------------------------------------------------------------------
# gridding
# ---------------------------------------------------------------------------


@provenance("Multi-radar gridding on a shared grid", extra_refs=("doviak-zrnic-1993",))
def grid_radars(
    radars,
    x=None,
    y=None,
    z=None,
    origin=None,
    data_vars=None,
    *,
    names=None,
    time=None,
    motion=None,
    merge=None,
    calibrate=None,
    reference=0,
    rhohv="RHOHV",
    rhohv_min=None,
    max_range=None,
    beamwidth=1.0,
    range_scale=50e3,
    time_scale=300.0,
    fill_below=False,
    max_gap=2.0,
    min_weight=0.5,
    n_threads=None,
    engine="auto",
):
    """
    Grid several radars onto one shared Cartesian grid.

    Every radar volume is cone-gridded (:func:`radarx.grid.grid_cones`) onto
    the same grid: an azimuthal equidistant projection centred on ``origin``
    with heights above sea level. For each radar the columns of the shared
    grid are located by their geodesic ground distance and azimuth from that
    radar (WGS84), so radars far from the origin are placed as accurately as
    the radar at it. All fields of all radars are gridded in one call of the
    compiled kernel.

    The result keeps every radar separately along a ``radar`` dimension,
    together with the beam geometry from each radar to each cell (4/3 Earth
    radius model, as in :mod:`xradar.georeference`; Doviak and Zrnić 1993 [4]_,
    Eqs. 2.28b-d, pp. 21-22):

    * ``azimuth``: direction of the beam at the cell, in degrees clockwise
      from the grid's ``+y`` axis (grid north). This is the direction of the
      great circle from the radar where it crosses the cell, so it already
      contains the convergence of meridians and of the projection.
    * ``elevation``: angle of the beam above the local horizontal at the cell,
      in degrees. It exceeds the antenna elevation by the central angle
      between radar and cell (earth curvature).
    * ``range``: slant range from the radar to the cell, in metres.

    so that a radial velocity is ``u sin(az) cos(el) + v cos(az) cos(el) +
    w sin(el)`` with ``(u, v, w)`` in the grid's ``x``, ``y``, ``z`` axes.

    Optionally

    * the fields of each radar are advected to a common analysis ``time``
      along ``motion`` (Gal-Chen 1982 [1]_; :func:`radarx.retrieve.advect`),
    * the fields in ``calibrate`` are corrected for their relative bias
      against the ``reference`` radar (:func:`network_bias`),
    * the fields in ``merge`` are merged into one weighted field
      (:func:`merge_radars`), stored as ``<field>_merged``.

    Parameters
    ----------
    radars : list of xarray.DataTree
        Radar volumes with ``sweep_*`` groups and the radar site
        ``latitude``, ``longitude`` and ``altitude``, e.g. from xradar. A
        mobile radar's site is its position during the volume.
    x, y : array-like
        Grid coordinates (m) east and north of ``origin`` in its azimuthal
        equidistant projection.
    z : array-like
        Grid heights above sea level, in metres.
    origin : tuple of float, optional
        ``(latitude, longitude)`` of the grid origin. Default: the site of the
        first radar.
    data_vars : str or list of str, optional
        Fields to grid. Default: every field on ``(azimuth, range)`` found in
        the lowest sweep of any radar. A radar without a field (or without
        any sweep containing it) gets NaN for it.
    names : list of str, optional
        Radar names for the ``radar`` coordinate. Default: each volume's
        ``instrument_name``.
    time : datetime-like, optional
        Analysis time. Default: the volume time of the first radar.
    motion : xarray.Dataset or tuple of float, optional
        Storm motion, the output of :func:`radarx.retrieve.estimate_motion` or
        ``(u, v)`` in m/s. When given, every radar's fields are advected from
        its volume time to ``time``; the geometry is not moved.
    merge : str or list of str, optional
        Fields to merge over all radars into ``<field>_merged``.
    calibrate : str or list of str, optional
        Fields (e.g. ``["DBZH", "ZDR"]``) whose relative biases are estimated
        with :func:`network_bias` and subtracted from every radar; the biases
        are stored as ``<field>_bias`` on the ``radar`` dimension.
    reference : int or str, optional
        Reference radar (index or name) of the calibration. Default: the
        first radar.
    rhohv : str, optional
        Co-polar correlation field used for ``rhohv_min``. Default
        ``"RHOHV"``.
    rhohv_min : float, optional
        Mask gates with a co-polar correlation below this (non-meteorological
        echoes) in every field before gridding, in sweeps that have
        ``rhohv``. Default: no masking.
    max_range : float, optional
        Discard cells farther than this slant range (m) from a radar.
    beamwidth : float or list of float, optional
        Half-power beamwidth (degrees) of each radar, for the merge weights.
        Default 1.
    range_scale, time_scale : float, optional
        Merge weight scales, see :func:`merge_radars`. Defaults 50 km (the
        value of Fig. 15 of Zhang et al. 2005 [3]_) and 300 s (a radarx
        choice).
    fill_below, max_gap, min_weight : optional
        Cone gridding options, see :func:`radarx.grid.grid_cones`.
    n_threads : int, optional
        Threads for the compiled kernels. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled
        kernels and falls back to NumPy.

    Returns
    -------
    xarray.Dataset
        On dimensions ``radar``, ``z``, ``y``, ``x``:

        * every gridded field on ``(radar, z, y, x)``, float32;
        * ``azimuth`` on ``(radar, y, x)``, ``elevation`` and ``range`` on
          ``(radar, z, y, x)``, float32;
        * coordinates ``radar`` (names), ``radar_x``, ``radar_y``,
          ``radar_z`` (radar position in the grid frame, m; ``radar_z`` above
          sea level), ``radar_latitude``, ``radar_longitude``,
          ``radar_altitude``, ``time`` (volume time of each radar),
          ``sweep_elevation`` on ``(radar, sweep)``;
        * ``x``, ``y``, ``z``, 2-D ``lat`` and ``lon`` on ``(y, x)`` and the
          grid's ``crs_wkt``;
        * optionally ``<field>_merged`` on ``(z, y, x)`` and ``<field>_bias``
          on ``radar``.

    Raises
    ------
    ValueError
        If no radar or no grid is given, or a volume has no sweeps or site.
    ImportError
        If ``engine="compiled"`` and the compiled kernels are not available.

    See Also
    --------
    merge_radars, network_bias, radarx.grid.grid_cones, radarx.retrieve.advect

    Notes
    -----
    Source of each ingredient. The geodesic geometry of the columns is
    Vincenty (1975) [2]_ (equation numbers not checked against the paper).
    Moving the fields to a common time along the storm motion is the
    frame-of-reference correction of Gal-Chen (1982) [1]_ (equations not
    checked against the paper). It assumes steady translation. The beam
    geometry is the 4/3 Earth model [4]_. Gridding, the merge weights and the
    calibration are described in :func:`grid_cones`, :func:`merge_radars` and
    :func:`network_bias`. The Earth radius (6371 km), ``beamwidth`` (1 degree)
    and ``time_scale`` defaults are radarx choices, not values from the
    references.

    References
    ----------
    .. [1] Gal-Chen, T., 1982: Errors in fixed and moving frame of references:
       Applications for conventional and Doppler radar analysis. *J. Atmos.
       Sci.*, **39**, 2279-2300,
       https://doi.org/10.1175/1520-0469(1982)039<2279:EIFAMF>2.0.CO;2
    .. [2] Vincenty, T., 1975: Direct and inverse solutions of geodesics on
       the ellipsoid with application of nested equations. *Survey Review*,
       **23**, 88-93, https://doi.org/10.1179/sre.1975.23.176.88
    .. [3] Zhang, J., K. Howard, and J. J. Gourley, 2005: Constructing
       three-dimensional multiple-radar reflectivity mosaics: Examples of
       convective storms and stratiform rain echoes. *J. Atmos. Oceanic
       Technol.*, **22**, 30-42, https://doi.org/10.1175/JTECH-1689.1
    .. [4] Doviak, R. J., and D. S. Zrnić, 1993: *Doppler Radar and Weather
       Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
       DOI).

    Examples
    --------
    >>> x = y = np.arange(-150e3, 150e3 + 1, 1000.0)  # doctest: +SKIP
    >>> z = np.arange(500.0, 12e3 + 1, 500.0)  # doctest: +SKIP
    >>> grid = radarx.grid.grid_radars(
    ...     [kgwx, knqa], x, y, z, data_vars=["DBZH", "ZDR", "VRADH"],
    ...     merge="DBZH", calibrate=["DBZH", "ZDR"],
    ... )  # doctest: +SKIP
    """
    use_compiled = _use_compiled(engine)
    if isinstance(radars, xr.DataTree):
        radars = [radars]
    radars = list(radars)
    if not radars:
        raise ValueError("give at least one radar volume")
    if x is None or y is None or z is None:
        raise ValueError("x, y and z target coordinates are required.")
    x, y, z = (np.asarray(a, dtype=np.float64) for a in (x, y, z))
    if isinstance(data_vars, str):
        data_vars = [data_vars]

    volumes = [cone._load_volume(dtree) for dtree in radars]
    sites = [site for _, site, _ in volumes]
    if origin is None:
        origin = (sites[0]["latitude"], sites[0]["longitude"])
    origin = (float(origin[0]), float(origin[1]))
    if names is None:
        names = [_radar_name(dtree, i) for i, dtree in enumerate(radars)]
    names = _unique([str(n) for n in names])
    if len(names) != len(radars):
        raise ValueError("give one name per radar")
    if data_vars is None:
        data_vars = []
        for sweeps, _, _ in volumes:
            for name, da in sweeps[0].data_vars.items():
                if {"range"} < set(da.dims) and name not in data_vars:
                    data_vars.append(name)
    altitudes = [float(site.get("altitude", 0.0)) for site in sites]
    beamwidths = np.broadcast_to(np.asarray(beamwidth, dtype=float), (len(radars),))

    columns = _column_geometry(x, y, origin, sites, use_compiled, n_threads)
    tasks, task_keys = [], []
    sweep_el = []
    for r, (sweeps, _, _) in enumerate(volumes):
        elevations = []
        for name in data_vars:
            selected = cone._select_sweeps(sweeps, name)
            if not selected:
                continue
            arrays = [_task_arrays(ds, name, rhohv, rhohv_min) for ds in selected]
            tasks.append((r, arrays))
            task_keys.append((r, name, dict(selected[0][name].attrs)))
            if len(selected) > len(elevations):
                elevations = _sweep_elevations(selected)
        sweep_el.append(elevations)
    if not tasks:
        raise ValueError(f"No sweep of any radar contains {data_vars!r}.")

    options = dict(max_gap=max_gap, min_weight=min_weight, fill_below=fill_below)
    gridded = _grid_tasks(
        columns, z, altitudes, tasks, use_compiled, n_threads, **options
    )
    if use_compiled:
        rng, elevation = _multi.beam_geometry(
            columns["ground"],
            z,
            altitudes,
            earth_radius=EARTH_RADIUS,
            n_threads=int(n_threads or 0),
        )
    else:
        rng, elevation = _geometry_numpy(columns["ground"], z, altitudes)

    out = _assemble(
        gridded,
        task_keys,
        data_vars,
        names,
        columns,
        rng,
        elevation,
        x,
        y,
        z,
        origin,
        sites,
        altitudes,
        [_volume_time(sweeps) for sweeps, _, _ in volumes],
        sweep_el,
        beamwidths,
        max_range,
    )
    if motion is not None:
        out = _advect_radars(out, data_vars, motion, time, engine, n_threads)
    elif time is not None:
        out.attrs["analysis_time"] = str(np.datetime64(time, "ns"))

    for name in [calibrate] if isinstance(calibrate, str) else calibrate or []:
        bias = network_bias(
            out, name, reference=reference, engine=engine, n_threads=n_threads
        )
        out[name] = out[name] - bias["bias"].fillna(0.0).astype(np.float32)
        out[name].attrs = dict(out[name].attrs, comment="calibrated, see bias")
        out[f"{name}_bias"] = bias["bias"]
    for name in [merge] if isinstance(merge, str) else merge or []:
        merged = merge_radars(
            out,
            name,
            time=time,
            range_scale=range_scale,
            time_scale=time_scale,
            engine=engine,
            n_threads=n_threads,
        )
        out[f"{name}_merged"] = merged[name]
    return out


def _grid_tasks(columns, z, altitudes, tasks, use_compiled, n_threads, **options):
    """Cone-grid every (radar, field) task onto the shared columns."""
    if use_compiled:
        radar = [r for r, _ in tasks]
        parts = [list(zip(*arrays)) for _, arrays in tasks]
        return cone._cone.grid_columns(
            columns["ground"],
            columns["antenna_azimuth"],
            z,
            radar,
            [list(p[0]) for p in parts],
            [list(p[1]) for p in parts],
            [list(p[2]) for p in parts],
            [list(p[3]) for p in parts],
            altitudes,
            earth_radius=EARTH_RADIUS,
            n_threads=int(n_threads or 0),
            **options,
        )
    return [
        cone._columns_numpy(
            columns["ground"][r],
            columns["antenna_azimuth"][r],
            z,
            arrays,
            altitudes[r],
            options["max_gap"],
            options["min_weight"],
            options["fill_below"],
        )
        for r, arrays in tasks
    ]


def _assemble(
    gridded,
    task_keys,
    data_vars,
    names,
    columns,
    rng,
    elevation,
    x,
    y,
    z,
    origin,
    sites,
    altitudes,
    times,
    sweep_el,
    beamwidths,
    max_range,
):
    """Stack per-radar results into the output Dataset."""
    nr = len(names)
    shape = (nr, z.size, y.size, x.size)
    dims = ("radar", "z", "y", "x")
    too_far = rng > max_range if max_range is not None else None
    fields = {}
    for name in data_vars:
        attrs = next((a for _, n, a in task_keys if n == name), None)
        if attrs is None:
            continue
        stack = np.full(shape, np.nan, dtype=np.float32)
        for (r, n, _), arr in zip(task_keys, gridded):
            if n == name:
                stack[r] = arr
        if too_far is not None:
            stack[too_far] = np.nan
        fields[name] = (dims, stack, attrs)
    fields["azimuth"] = (
        ("radar", "y", "x"),
        np.stack(columns["azimuth"]).astype(np.float32),
        {
            "standard_name": "sensor_to_target_azimuth_angle",
            "long_name": "direction of the beam at the cell, clockwise from grid north",
            "units": "degrees",
        },
    )
    fields["elevation"] = (
        dims,
        elevation,
        {
            "long_name": "elevation of the beam above the local horizontal at the cell",
            "units": "degrees",
        },
    )
    fields["range"] = (
        dims,
        rng,
        {"long_name": "slant range from the radar to the cell", "units": "m"},
    )
    nk = max([len(e) for e in sweep_el] + [1])
    sweep_table = np.full((nr, nk), np.nan)
    for r, e in enumerate(sweep_el):
        sweep_table[r, : len(e)] = e
    coords = {
        "radar": ("radar", np.asarray(names, dtype=object), {"long_name": "radar"}),
        "z": ("z", z, {"units": "m", "long_name": "height above sea level"}),
        "y": ("y", y, {"units": "m", "long_name": "distance north of the origin"}),
        "x": ("x", x, {"units": "m", "long_name": "distance east of the origin"}),
        "lat": (("y", "x"), columns["lat"], {"units": "degrees_north"}),
        "lon": (("y", "x"), columns["lon"], {"units": "degrees_east"}),
        "crs_wkt": cone._crs_wkt(*origin),
        "radar_x": (
            "radar",
            np.asarray(columns["radar_x"]),
            {"units": "m", "long_name": "radar position east of the origin"},
        ),
        "radar_y": (
            "radar",
            np.asarray(columns["radar_y"]),
            {"units": "m", "long_name": "radar position north of the origin"},
        ),
        "radar_z": (
            "radar",
            np.asarray(altitudes),
            {"units": "m", "long_name": "radar altitude above sea level"},
        ),
        "radar_latitude": (
            "radar",
            np.array([s["latitude"] for s in sites]),
            {"units": "degrees_north", "standard_name": "latitude"},
        ),
        "radar_longitude": (
            "radar",
            np.array([s["longitude"] for s in sites]),
            {"units": "degrees_east", "standard_name": "longitude"},
        ),
        "radar_altitude": (
            "radar",
            np.asarray(altitudes),
            {"units": "m", "standard_name": "altitude"},
        ),
        "time": (
            "radar",
            np.asarray(times, dtype="datetime64[ns]"),
            {"long_name": "volume time of each radar"},
        ),
        "sweep_elevation": (
            ("radar", "sweep"),
            sweep_table,
            {"long_name": "sweep elevation angles", "units": "degrees"},
        ),
        "beamwidth": (
            "radar",
            np.asarray(beamwidths, dtype=float),
            {"long_name": "half-power beamwidth", "units": "degrees"},
        ),
    }
    out = xr.Dataset(fields, coords=coords)
    out.attrs = {
        "gridding_method": "cone",
        "origin_latitude": origin[0],
        "origin_longitude": origin[1],
        "comment": "multi-radar grid; heights above sea level",
    }
    return out


def _advect_radars(out, data_vars, motion, time, engine, n_threads):
    """Advect every radar's fields from its volume time to the analysis time."""
    from ..retrieve import advect

    if time is None:
        time = out["time"].values[0]
    time = np.datetime64(time, "ns")
    if isinstance(motion, xr.Dataset):
        u, v = motion, None
    else:
        u, v = motion
    fields = [name for name in data_vars if name in out]
    moved = []
    for r in range(out.sizes["radar"]):
        part = out[fields].isel(radar=r).drop_vars("time")
        dt = float((time - out["time"].values[r]) / np.timedelta64(1, "s"))
        moved.append(advect(part, u, v, dt=dt, engine=engine, n_threads=n_threads))
    for name in fields:
        dims = out[name].dims
        stack = np.stack([m[name].transpose(*dims[1:]).values for m in moved])
        out[name] = out[name].copy(data=stack)
    out.attrs["analysis_time"] = str(time)
    out.attrs["advection"] = "fields advected from each volume time to analysis_time"
    return out


# ---------------------------------------------------------------------------
# merging
# ---------------------------------------------------------------------------


def _radar_columns(grid, use_compiled=False, n_threads=None):
    """Ground distance of every column from every radar of a multi-radar grid."""
    origin = (grid.attrs["origin_latitude"], grid.attrs["origin_longitude"])
    sites = [
        {"latitude": float(la), "longitude": float(lo)}
        for la, lo in zip(grid["radar_latitude"].values, grid["radar_longitude"].values)
    ]
    x, y = grid["x"].values.astype(np.float64), grid["y"].values.astype(np.float64)
    return _column_geometry(x, y, origin, sites, use_compiled, n_threads)


@provenance("Weighted merge of gridded radars, after Zhang et al. (2005)")
def merge_radars(
    grid,
    data_vars="DBZH",
    *,
    bias=None,
    time=None,
    range_scale=50e3,
    time_scale=300.0,
    beamwidth=None,
    n_threads=None,
    engine="auto",
):
    """
    Weighted merge of the radars of a multi-radar grid.

    At every cell the defined values of all radars are averaged with weights

    .. math::

        w = \\exp(-r^2 / L^2) \\; \\delta_e \\; \\exp(-\\Delta t^2 / \\tau^2)

    * ``r`` is the slant range from the radar and ``L = range_scale``: the
      nearer radar, with the smaller, lower beam, dominates. This is the
      Gaussian distance weight ``w = exp(-d^2 / R^2)`` with ``R = 50 km`` of
      Zhang et al. (2005) [1]_, shown in their Fig. 15 (p. 41), where ``d`` is
      the distance between the grid cell and the radar. The paper gives the
      function only as the legend of that figure; its text (p. 39-40) says
      the weight decreases monotonically with range and that the steep
      function (this one) is preferred over a flat alternative. radarx
      uses the slant range for ``d`` and keeps 50 km as the default;
    * ``delta_e`` is the elevation weight of Lakshmanan et al. (2006) [2]_,
      Eq. (6), p. 808, ``exp(|a|^3 ln 0.005)``, where ``a`` is the angular
      distance of the cell from the nearest beam axis as a fraction of the
      larger of the beamwidth and the spacing to the next sweep: 1 on a beam
      axis, 0.52 half way between sweeps and 0.005 one spacing away (the paper
      quotes 0.5 and below 0.01). Cone gridding fills the gaps between
      sweeps; this weight lets a radar whose beam passes through the cell
      override one that interpolates across a gap. This factor reproduces the
      paper's formula;
    * ``dt`` is the time between the radar's volume and the analysis time and
      ``tau = time_scale``, so the most recent radar dominates. This time
      factor is a radarx choice (a Gaussian with ``tau`` = 300 s) and is not
      taken from either paper.

    Difference from Lakshmanan et al. (2006). That paper weights an
    observation (an "agent", including repeated scans of the same radar) by
    ``delta = delta_e exp[-(t^2 r^2 / beta)]`` (their Eq. 7, p. 809), with
    ``t`` the time between the observation and the grid time in seconds,
    ``r`` the range of the gate in km and ``beta = 17.36 s^2 km^2``, "chosen
    through experimentation". There the range matters only through the
    product ``t r``: at ``t = 0`` every range has weight 1 and the time weight
    is tighter at long range. radarx instead multiplies a range Gaussian and
    a time Gaussian that act independently, so the range weight also
    separates two radars observed at the same time. radarx therefore does
    not reproduce the merger of Lakshmanan et al. (2006); only the elevation
    weight is theirs and only the range weight is Zhang et al.'s. (With the
    paper's units a 60 s old scan at 50 km would have weight
    ``exp(-5.2e5)``, so ``beta`` is meaningful only for time offsets of a few
    seconds; the paper does not resolve this.) The paper's text before Eq. 7
    also describes the agent weight simply as an exponentially declining
    function of the distance from the radar.

    A scale of ``None`` or 0 switches its factor off. Values are averaged in
    the units of the field (dBZ for reflectivity).

    Parameters
    ----------
    grid : xarray.Dataset
        Output of :func:`grid_radars`.
    data_vars : str or list of str, optional
        Fields to merge. Default ``"DBZH"``.
    bias : xarray.Dataset or dict, optional
        Biases to subtract first: the output of :func:`network_bias` (for a
        single field) or ``{field: DataArray on radar}``.
    time : datetime-like, optional
        Analysis time for the time weights. Default: the grid's
        ``analysis_time`` attribute or the first radar's time.
    range_scale : float, optional
        ``L`` in metres. Default 50 km.
    time_scale : float, optional
        ``tau`` in seconds. Default 300 s.
    beamwidth : float or list of float, optional
        Beamwidth (degrees) per radar. Default: the grid's ``beamwidth``.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use.

    Returns
    -------
    xarray.Dataset
        Merged fields on ``(z, y, x)`` with ``<field>_weight``, the sum of the
        weights, and the grid's coordinates without the ``radar`` dimension.

    Raises
    ------
    ValueError
        If a field is not in ``grid``.

    References
    ----------
    .. [1] Zhang, J., K. Howard, and J. J. Gourley, 2005: Constructing
       three-dimensional multiple-radar reflectivity mosaics: Examples of
       convective storms and stratiform rain echoes. *J. Atmos. Oceanic
       Technol.*, **22**, 30-42, https://doi.org/10.1175/JTECH-1689.1
    .. [2] Lakshmanan, V., T. Smith, K. Hondl, G. J. Stumpf, and A. Witt,
       2006: A real-time, three-dimensional, rapidly updating, heterogeneous
       radar merger technique for reflectivity, velocity, and derived
       products. *Wea. Forecasting*, **21**, 802-823,
       https://doi.org/10.1175/WAF942.1

    Examples
    --------
    >>> merged = radarx.grid.merge_radars(grid, "DBZH")  # doctest: +SKIP
    """
    use_compiled = _use_compiled(engine)
    if isinstance(data_vars, str):
        data_vars = [data_vars]
    missing = [name for name in data_vars if name not in grid]
    if missing:
        raise ValueError(f"grid has no {missing}")
    if time is None:
        time = grid.attrs.get("analysis_time", grid["time"].values[0])
    time = np.datetime64(time, "ns")
    offsets = list(
        ((grid["time"].values - time) / np.timedelta64(1, "s")).astype(float)
    )
    offsets = [0.0 if np.isnan(o) else o for o in offsets]
    if beamwidth is None:
        beamwidth = grid["beamwidth"].values if "beamwidth" in grid else 1.0
    nr = grid.sizes["radar"]
    beamwidth = [float(b) for b in np.broadcast_to(beamwidth, (nr,))]
    elevations = [
        [float(e) for e in row if np.isfinite(e)]
        for row in grid["sweep_elevation"].values
    ]
    altitudes = [float(a) for a in grid["radar_altitude"].values]
    ground = _radar_columns(grid, use_compiled, n_threads)["ground"]
    z = grid["z"].values.astype(np.float64)
    L = float(range_scale or 0.0)
    tau = float(time_scale or 0.0)

    out = {}
    for name in data_vars:
        da = grid[name].transpose("radar", "z", "y", "x")
        if bias is not None:
            b = bias[name] if not isinstance(bias, xr.Dataset) else bias["bias"]
            da = da - b.fillna(0.0).astype(np.float32)
        values = np.ascontiguousarray(da.values, dtype=np.float32)
        args = (values, ground, z, altitudes, elevations, beamwidth, offsets, L, tau)
        if use_compiled:
            merged, weight = _multi.merge(
                *args, earth_radius=EARTH_RADIUS, n_threads=int(n_threads or 0)
            )
        else:
            merged, weight = _merge_numpy(*args)
        attrs = dict(grid[name].attrs)
        attrs["comment"] = "weighted merge of all radars"
        out[name] = (("z", "y", "x"), merged, attrs)
        out[f"{name}_weight"] = (
            ("z", "y", "x"),
            weight,
            {"long_name": f"sum of merge weights of {name}", "units": "1"},
        )
    coords = {
        k: c for k, c in grid.coords.items() if "radar" not in c.dims and k != "time"
    }
    ds = xr.Dataset(out, coords=coords)
    ds["time"] = xr.DataArray(time, attrs={"long_name": "analysis time"})
    ds.attrs = dict(grid.attrs)
    return ds


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def _histogram_quantiles(counts, edges, q):
    """Quantiles ``q`` of each row of a histogram (linear within bins)."""
    out = np.full((counts.shape[0], len(q)), np.nan)
    for p, row in enumerate(counts):
        total = row.sum()
        if total == 0:
            continue
        cdf = np.concatenate([[0.0], np.cumsum(row) / total])
        for k, qq in enumerate(q):
            out[p, k] = np.interp(qq, cdf, edges)
    return out


def _is_reflectivity(da):
    return str(da.attrs.get("units", "")).lower() in ("dbz", "dbz ") or str(
        da.name
    ).upper().startswith(("DBZ", "REFL"))


def _network_solution(pairs, d, w, nr, ref):
    """Weighted least squares ``b_i - b_j = d_ij`` with ``b_ref = 0``."""
    bias = np.full(nr, np.nan)
    error = np.full(nr, np.nan)
    # radars connected to the reference
    linked = {ref}
    changed = True
    while changed:
        changed = False
        for i, j in pairs:
            if (i in linked) != (j in linked):
                linked |= {i, j}
                changed = True
    unknowns = sorted(linked - {ref})
    bias[ref] = 0.0
    error[ref] = 0.0
    rows = [(k, i, j) for k, (i, j) in enumerate(pairs) if i in linked and j in linked]
    if not unknowns or not rows:
        return bias, error, np.zeros(len(pairs))
    col = {r: c for c, r in enumerate(unknowns)}
    A = np.zeros((len(rows), len(unknowns)))
    rhs = np.zeros(len(rows))
    sw = np.zeros(len(rows))
    for n, (k, i, j) in enumerate(rows):
        if i in col:
            A[n, col[i]] = 1.0
        if j in col:
            A[n, col[j]] = -1.0
        rhs[n] = d[k]
        sw[n] = np.sqrt(w[k])
    sol, *_ = np.linalg.lstsq(A * sw[:, None], rhs * sw, rcond=None)
    for r, c in col.items():
        bias[r] = sol[c]
    cov = np.linalg.pinv((A * sw[:, None]).T @ (A * sw[:, None]))
    for r, c in col.items():
        error[r] = np.sqrt(cov[c, c])
    residual = np.zeros(len(pairs))
    for n, (k, i, j) in enumerate(rows):
        residual[k] = d[k] - (bias[i] - bias[j])
    return bias, error, residual


@provenance("Relative radar biases from overlaps, radarx's own after Seo et al. (2014)")
def network_bias(
    grid,
    field="DBZH",
    reference=0,
    *,
    valid_range=None,
    reflectivity="DBZH",
    reflectivity_range=None,
    z_range=None,
    max_range=None,
    max_range_ratio=None,
    min_count=200,
    bin_width=0.01,
    max_difference=20.0,
    n_threads=None,
    engine="auto",
):
    """
    Relative calibration biases of a radar network from its overlap regions.

    For every pair of radars ``(i, j)`` the differences ``X_i - X_j`` of a
    field are collected over all cells of the multi-radar grid observed by
    both radars at the same height and (after advection) the same time. Seo
    et al. (2014) [1]_ compare two ground-based radars by matching their
    observations in space and time; radarx takes only that idea. Their
    matching and statistics are not reproduced. The pair bias ``d_ij`` is the
    median difference, resistant to residual clutter, partial beam blockage
    and attenuation; its spread ``s_ij`` is
    the interquartile range divided by 1.349. The pair biases are then
    reconciled across the network by weighted least squares,

    .. math::

        \\min_b \\sum_{(i,j)} \\frac{n_{ij}}{s_{ij}^2}
        \\left(b_i - b_j - d_{ij}\\right)^2, \\qquad b_{ref} = 0,

    so that every radar connected to the reference through a chain of
    overlaps gets a bias, closed loops of overlaps are made consistent and
    well-sampled pairs count most. The corrected field is ``X - bias``.

    The median pair bias, the interquartile-range spread (1.349 is the ratio
    of the interquartile range of a normal distribution to its standard
    deviation, so IQR / 1.349 estimates sigma), the weights ``n / s^2`` and
    the least-squares network solution are radarx's construction, not taken
    from Seo et al. (2014). The defaults (valid reflectivity 15 to 50 dBZ,
    companion reflectivity 20 to 40 dBZ, ``min_count`` 200, ``bin_width``
    0.01, ``max_difference`` 20) are radarx choices without a published
    source.

    Parameters
    ----------
    grid : xarray.Dataset
        Output of :func:`grid_radars`.
    field : str, optional
        Field to compare, e.g. ``"DBZH"`` or ``"ZDR"``. Default ``"DBZH"``.
    reference : int or str, optional
        Reference radar (index or name), whose bias is 0. Default: the first.
    valid_range : tuple of float, optional
        Use only values inside this range on both radars. Default (15, 50)
        dBZ for reflectivity (above the noise, below hail), no limit for
        other fields.
    reflectivity : str, optional
        Reflectivity field used by ``reflectivity_range``. Default ``"DBZH"``.
    reflectivity_range : tuple of float, optional
        For fields other than reflectivity: use only cells where both radars
        measure a reflectivity in this range. Default (20, 40) dBZ, rain
        without hail, if ``reflectivity`` is in ``grid``.
    z_range : tuple of float, optional
        Height range (m above sea level) to use, e.g. below the melting
        layer. Default: all levels.
    max_range : float, optional
        Use only cells within this slant range (m) of both radars.
    max_range_ratio : float, optional
        Use only cells whose larger range is at most this times the smaller
        one (similar sample volumes near the line midway between radars).
    min_count : int, optional
        Pairs with fewer matched cells are not used. Default 200.
    bin_width, max_difference : float, optional
        Resolution and span of the difference histograms. Differences beyond
        ``max_difference`` (default 20) are discarded as outliers.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use.

    Returns
    -------
    xarray.Dataset
        ``bias`` and ``bias_error`` on ``radar`` (the formal standard error,
        a lower bound because neighbouring cells are not independent), and
        ``pair_bias`` (median of row minus column radar), ``pair_spread``,
        ``pair_count`` and ``pair_residual`` on ``(radar, other)``.

    Raises
    ------
    ValueError
        If ``field`` is not in ``grid`` or ``reference`` is unknown.

    References
    ----------
    .. [1] Seo, B.-C., W. F. Krajewski, and J. A. Smith, 2014:
       Four-dimensional reflectivity data comparison between two
       ground-based radars: methodology and statistical analysis.
       *Hydrol. Sci. J.*, **59**, 1320-1334,
       https://doi.org/10.1080/02626667.2013.839872

    Examples
    --------
    >>> bias = radarx.grid.network_bias(grid, "DBZH", reference="KGWX")  # doctest: +SKIP
    >>> corrected = grid.DBZH - bias.bias  # doctest: +SKIP
    """
    use_compiled = _use_compiled(engine)
    if field not in grid:
        raise ValueError(f"grid has no {field!r}")
    names = [str(n) for n in grid["radar"].values]
    nr = len(names)
    if isinstance(reference, str):
        if reference not in names:
            raise ValueError(f"unknown reference radar {reference!r}")
        ref = names.index(reference)
    else:
        ref = int(reference)
        if not 0 <= ref < nr:
            raise ValueError(f"reference index {reference} out of range")

    da = grid[field].transpose("radar", "z", "y", "x")
    keep = xr.ones_like(da, dtype=bool)
    is_refl = _is_reflectivity(grid[field])
    if valid_range is None and is_refl:
        valid_range = (15.0, 50.0)
    if valid_range is not None:
        keep &= (da >= valid_range[0]) & (da <= valid_range[1])
    if not is_refl and reflectivity in grid:
        if reflectivity_range is None:
            reflectivity_range = (20.0, 40.0)
        zh = grid[reflectivity].transpose(*da.dims)
        keep &= (zh >= reflectivity_range[0]) & (zh <= reflectivity_range[1])
    if z_range is not None:
        keep &= (grid["z"] >= z_range[0]) & (grid["z"] <= z_range[1])
    rng = grid["range"].transpose(*da.dims)
    if max_range is not None:
        keep &= rng <= max_range
    values = np.ascontiguousarray(
        da.where(keep).values.reshape(nr, -1), dtype=np.float32
    )
    ratio = float(max_range_ratio or 0.0)
    rng_arr = (
        np.ascontiguousarray(rng.values.reshape(nr, -1), dtype=np.float32)
        if ratio > 0
        else np.zeros((0, 0), dtype=np.float32)
    )
    if use_compiled:
        counts, sums, sumsq = _multi.pair_histograms(
            values,
            rng_arr,
            ratio,
            float(bin_width),
            float(max_difference),
            n_threads=int(n_threads or 0),
        )
    else:
        counts, sums, sumsq = _pair_histograms_numpy(
            values, rng_arr, ratio, float(bin_width), float(max_difference)
        )
    edges = -max_difference + bin_width * np.arange(counts.shape[1] + 1)
    quant = _histogram_quantiles(counts, edges, [0.25, 0.5, 0.75])
    n = counts.sum(axis=1)
    pairs = [(i, j) for i in range(nr) for j in range(i + 1, nr)]
    median = quant[:, 1]
    spread = (quant[:, 2] - quant[:, 0]) / 1.349
    spread = np.where(spread > 0, spread, bin_width)
    used = [k for k in range(len(pairs)) if n[k] >= min_count]
    w = np.where(n >= min_count, n / spread**2, 0.0)
    bias, error, residual = _network_solution(
        [pairs[k] for k in used], median[used], w[used], nr, ref
    )
    full_residual = np.full(len(pairs), np.nan)
    full_residual[used] = residual

    def square(vals, antisymmetric):
        m = np.full((nr, nr), np.nan)
        for k, (i, j) in enumerate(pairs):
            m[i, j] = vals[k]
            m[j, i] = -vals[k] if antisymmetric else vals[k]
        return m

    valid = n >= min_count
    units = grid[field].attrs.get("units", "")
    other = ("other", names)
    out = xr.Dataset(
        {
            "bias": (
                "radar",
                bias,
                {"long_name": f"{field} bias relative to {names[ref]}", "units": units},
            ),
            "bias_error": (
                "radar",
                error,
                {"long_name": "formal standard error of the bias", "units": units},
            ),
            "pair_bias": (
                ("radar", "other"),
                square(np.where(valid, median, np.nan), True),
                {
                    "long_name": f"median {field} difference radar - other",
                    "units": units,
                },
            ),
            "pair_spread": (
                ("radar", "other"),
                square(np.where(n > 0, spread, np.nan), False),
                {
                    "long_name": "robust spread (IQR / 1.349) of the differences",
                    "units": units,
                },
            ),
            "pair_count": (
                ("radar", "other"),
                np.nan_to_num(square(n.astype(float), False)).astype(np.int64),
                {"long_name": "number of matched cells"},
            ),
            "pair_residual": (
                ("radar", "other"),
                square(full_residual, True),
                {"long_name": "pair bias minus network solution", "units": units},
            ),
        },
        coords={"radar": grid["radar"].values, "other": other},
    )
    out.attrs = {
        "field": field,
        "reference": names[ref],
        "method": "median pair differences, weighted least-squares network solution",
    }
    with np.errstate(invalid="ignore"):
        out["pair_mean"] = (
            ("radar", "other"),
            square(np.where(n > 0, sums / np.maximum(n, 1), np.nan), True),
            {"long_name": f"mean {field} difference radar - other", "units": units},
        )
    return out


@accessor_method("dataset", name="network_bias")
def _network_bias_dataset_accessor(self, field="DBZH", reference=0, **kwargs):
    """
    Relative calibration biases of the radars of a multi-radar grid.

    Parameters
    ----------
    field : str, optional
        Field to compare. Default ``"DBZH"``.
    reference : int or str, optional
        Reference radar. Default: the first.
    **kwargs
        See :func:`radarx.grid.network_bias`.

    Returns
    -------
    xarray.Dataset
        ``bias`` on ``radar`` and the pair statistics.

    See Also
    --------
    radarx.grid.network_bias
    """
    return network_bias(self.xarray_obj, field, reference, **kwargs)


@accessor_method("dataset", name="merge_radars")
def _merge_radars_dataset_accessor(self, data_vars="DBZH", **kwargs):
    """
    Weighted merge of the radars of a multi-radar grid.

    Parameters
    ----------
    data_vars : str or list of str, optional
        Fields to merge. Default ``"DBZH"``.
    **kwargs
        See :func:`radarx.grid.merge_radars`.

    Returns
    -------
    xarray.Dataset
        Merged fields on ``(z, y, x)``.

    See Also
    --------
    radarx.grid.merge_radars
    """
    return merge_radars(self.xarray_obj, data_vars, **kwargs)


@accessor_method("datatree", name="grid_radars")
def _grid_radars_datatree_accessor(self, others=(), x=None, y=None, z=None, **kwargs):
    """
    Grid this radar together with ``others`` onto one shared grid.

    Parameters
    ----------
    others : xarray.DataTree or list of xarray.DataTree, optional
        The other radar volumes.
    x, y, z : array-like
        Grid coordinates (m); ``z`` above sea level.
    **kwargs
        See :func:`radarx.grid.grid_radars`. The default origin is this
        radar's site.

    Returns
    -------
    xarray.Dataset
        Multi-radar grid with a ``radar`` dimension, this radar first.

    See Also
    --------
    radarx.grid.grid_radars
    """
    if isinstance(others, xr.DataTree):
        others = [others]
    return grid_radars([self.xarray_obj, *others], x, y, z, **kwargs)
