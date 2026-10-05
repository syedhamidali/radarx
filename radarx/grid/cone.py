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

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = ["grid_cones"]

__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np
import xarray as xr

try:
    from . import _cone

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _cone = None
    HAS_COMPILED_KERNEL = False

EARTH_RADIUS = 6371000.0


def _sweep_names(dtree):
    return sorted(
        (name for name in dtree.children if name.startswith("sweep")),
        key=lambda name: int(name.rsplit("_", 1)[-1]) if name[-1].isdigit() else name,
    )


def _sweep_dataset(dtree, name):
    """Sweep node as a Dataset including the radar site coordinates."""
    try:
        return dtree[name].to_dataset(inherit="all_coords")
    except (TypeError, ValueError):  # xarray without "all_coords"
        return dtree[name].to_dataset()


def _select_sweeps(sweeps, variable, tolerance=0.1):
    """
    Pick one sweep per elevation that contains ``variable``.

    Volumes with split cuts (e.g. NEXRAD) scan some elevations twice; the cut
    that reaches farthest is used.

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
    data, azimuth, elevation, rng, s, az, site_altitude, max_gap, min_weight
):
    """Value and beam height of one sweep at ground distance ``s``, azimuth ``az``."""
    from xradar.georeference import antenna_to_cartesian

    el_med = float(np.median(elevation))
    gx, gy, _ = antenna_to_cartesian(rng, 0.0, el_med, site_altitude=site_altitude)
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
        rng[None, :], 0.0, elevation[:, None], site_altitude=site_altitude
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


def _grid_numpy(x, y, z, arrays, site_altitude, max_gap, min_weight, fill_below):
    """NumPy implementation of the compiled kernel (same results)."""
    X, Y = np.meshgrid(x, y)
    s = np.hypot(X, Y)
    az = np.mod(np.degrees(np.arctan2(X, Y)), 360.0)
    values, heights = zip(
        *(
            _cone_numpy(*sweep, s, az, site_altitude, max_gap, min_weight)
            for sweep in arrays
        )
    )
    V = np.stack(values)
    H = np.stack(heights)
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
            result = np.where(
                (k < 0) & np.isfinite(H[0]), V[0], result
            )  # lowest present cone
        out[iz] = result
    return out


def _lonlat_axes(x, y, latitude, longitude):
    """Longitude along x (at y = 0) and latitude along y (at x = 0)."""
    import pyproj

    aeqd = pyproj.Proj(proj="aeqd", lat_0=latitude, lon_0=longitude, datum="WGS84")
    lon, _ = aeqd(x, np.zeros_like(x), inverse=True)
    _, lat = aeqd(np.zeros_like(y), y, inverse=True)
    return np.asarray(lon), np.asarray(lat)


def grid_cones(
    dtree,
    data_vars=None,
    x=None,
    y=None,
    z=None,
    *,
    fill_below=False,
    max_gap=2.0,
    min_weight=0.5,
    n_threads=None,
    engine="auto",
):
    """
    Grid a radar volume onto a Cartesian grid with cone interpolation.

    Parameters
    ----------
    dtree : xarray.DataTree
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
        Gridded fields on ``(z, y, x)`` as float32, with ``lat``/``lon``
        axis coordinates, the radar site, ``crs_wkt`` and the volume time.

    Raises
    ------
    ValueError
        If no sweep contains a requested variable, or the grid is missing.
    ImportError
        If ``engine="compiled"`` and the compiled kernel is not available.

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
    sweeps, site, site_coords = _load_volume(dtree)
    if data_vars is None:
        data_vars = [
            name for name, da in sweeps[0].data_vars.items() if {"range"} < set(da.dims)
        ]
    elif isinstance(data_vars, str):
        data_vars = [data_vars]

    options = dict(max_gap=max_gap, min_weight=min_weight, fill_below=fill_below)
    gridded = {}
    for variable in data_vars:
        selected = _select_sweeps(sweeps, variable)
        if not selected:
            raise ValueError(f"No sweep contains {variable!r}.")
        arrays = [_sweep_arrays(ds, variable) for ds in selected]
        site_altitude = site.get("altitude", 0.0)
        if use_compiled:
            field = _grid_compiled(x, y, z, arrays, site_altitude, n_threads, **options)
        else:
            field = _grid_numpy(x, y, z, arrays, site_altitude, **options)
        gridded[variable] = (("z", "y", "x"), field, dict(selected[0][variable].attrs))
    out = _to_dataset(gridded, x, y, z, sweeps, site, dtree.attrs)
    return out.assign_coords(site_coords)


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
    import pyproj

    crs = pyproj.CRS.from_dict(
        {"proj": "aeqd", "lat_0": latitude, "lon_0": longitude, "datum": "WGS84"}
    )
    return xr.DataArray(0, attrs=crs.to_cf())


def _to_dataset(gridded, x, y, z, sweeps, site, attrs):
    """Wrap gridded fields with coordinates, site, CRS, time and attributes."""
    lon, lat = _lonlat_axes(x, y, site["latitude"], site["longitude"])
    out = xr.Dataset(
        gridded,
        coords={
            "z": ("z", z, {"units": "m", "long_name": "height above sea level"}),
            "y": ("y", y, {"units": "m", "long_name": "distance north of the radar"}),
            "x": ("x", x, {"units": "m", "long_name": "distance east of the radar"}),
            "lat": ("y", lat, {"units": "degrees_north"}),
            "lon": ("x", lon, {"units": "degrees_east"}),
            "crs_wkt": _crs_wkt(site["latitude"], site["longitude"]),
        },
    )
    times = [np.ravel(ds["time"].values) for ds in sweeps if "time" in ds]
    if times:
        out["time"] = xr.DataArray(
            np.concatenate(times).astype("datetime64[ns]")
        ).mean()
    out.attrs = dict(attrs)
    out.attrs["radar_name"] = out.attrs.get("instrument_name", "")
    out.attrs["gridding_method"] = "cone"
    return out
