#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Quasi-Vertical Profiles
=======================

Quasi-vertical profiles (QVPs, Ryzhkov et al. 2016) are azimuthal averages of
the polarimetric variables measured on a high-elevation PPI. Each range gate
of the sweep is reduced over all rays to a single value and assigned to the
beam height of that gate, so one sweep gives one vertical profile and a
sequence of volumes gives a time-height display.

Following Ryzhkov et al. (2016), only gates with ``rhohv > 0.6`` and
``Z > -10 dBZ`` are used, and a value is defined only where at least 30 gates
on the circle are valid. Quantities in decibels (reflectivity, differential
reflectivity) are averaged in linear units and converted back; the other
variables are averaged as they are. A median is available as a robust
alternative. The melting layer can be detected in the profiles from the
co-located rhohv minimum and ZDR / Z maxima (after Giangrande et al. 2008),
its top and bottom placed where rhohv returns to its background (Griffin et
al. 2020), compared with the 0 °C and wet-bulb 0 °C heights of a sounding or
ERA5 profile, and checked for consistency along the time series.

The reductions over (time, azimuth, range) are done by a compiled C++ kernel
in a single multithreaded pass; if it is not available, an equivalent NumPy
implementation is used.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}

References
----------
Ryzhkov, A., P. Zhang, H. Reeves, M. Kumjian, T. Tschallener, S. Trömel, and
C. Simmer, 2016: Quasi-vertical profiles—A new way to look at polarimetric
radar data. *J. Atmos. Oceanic Technol.*, **33**, 551–562,
https://doi.org/10.1175/JTECH-D-15-0020.1

Trömel, S., M. R. Kumjian, A. V. Ryzhkov, C. Simmer, and M. Diederich, 2013:
Backscatter differential phase—Estimation and variability. *J. Appl. Meteor.
Climatol.*, **52**, 2529–2548, https://doi.org/10.1175/JAMC-D-13-0124.1

Giangrande, S. E., J. M. Krause, and A. V. Ryzhkov, 2008: Automatic
designation of the melting layer with a polarimetric prototype of the WSR-88D
radar. *J. Appl. Meteor. Climatol.*, **47**, 1354–1364,
https://doi.org/10.1175/2007JAMC1634.1

Griffin, E. M., T. J. Schuur, and A. V. Ryzhkov, 2020: A polarimetric radar
analysis of ice microphysical processes in melting layers of winter storms
using S-band quasi-vertical profiles. *J. Appl. Meteor. Climatol.*, **59**,
751–767, https://doi.org/10.1175/JAMC-D-19-0128.1
"""

from __future__ import annotations

__all__ = ["melting_layer", "qvp", "qvp_timeseries"]

__doc__ = __doc__.format("\n   ".join(__all__))

import math
import warnings

import numpy as np
import xarray as xr

try:
    from . import _qvp

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _qvp = None
    HAS_COMPILED_KERNEL = False

EARTH_RADIUS = 6371000.0
_LN10_OVER_10 = math.log(10.0) / 10.0
_MODES = {"mean": 0, "mean_db": 1, "median": 2}
_DB_UNITS = {"db", "dbz", "dbm"}

_DBZ_NAMES = ("DBZH", "DBZ", "reflectivity", "corrected_reflectivity", "DBTH")
_DBZ_STANDARD = "radar_equivalent_reflectivity_factor_h"
_RHOHV_NAMES = ("RHOHV", "rhohv", "cross_correlation_ratio", "RHO", "copol_coeff")
_RHOHV_STANDARD = "radar_correlation_coefficient_hv"
_ZDR_NAMES = ("ZDR", "differential_reflectivity", "corrected_differential_reflectivity")
_ZDR_STANDARD = "radar_differential_reflectivity_hv"


def _use_compiled(engine):
    """Whether to run the compiled kernel for the requested ``engine``."""
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled QVP kernel is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


def _find_field(ds, names, standard_name):
    """First variable of ``ds`` named in ``names``, else by ``standard_name``."""
    for name in names:
        if name in ds.data_vars:
            return name
    for name, da in ds.data_vars.items():
        if da.attrs.get("standard_name") == standard_name:
            return name
    return None


def _resolve_field(ds, value, names, standard_name, what):
    """Field name for an ``"auto"`` / explicit / ``None`` argument."""
    if value is None:
        return None
    if value == "auto":
        return _find_field(ds, names, standard_name)
    if value not in ds.data_vars:
        raise ValueError(f"{what} field {value!r} not found in the sweep.")
    return value


# ---------------------------------------------------------------------------
# sweeps and geometry
# ---------------------------------------------------------------------------


def _sweep_dataset(dtree, name):
    """Sweep node as a Dataset including the radar site coordinates."""
    try:
        return dtree[name].to_dataset(inherit="all_coords")
    except (TypeError, ValueError):  # xarray without "all_coords"
        return dtree[name].to_dataset()


def _sweep_elevations(dtree, names):
    """Elevation of each sweep: the root's ``sweep_fixed_angle`` if it lists
    every ``sweep_<i>`` (xradar layout, cheap), else the median ray elevation."""
    root = dtree.ds
    fixed = root["sweep_fixed_angle"].values if "sweep_fixed_angle" in root else []
    try:
        index = {name: int(name.rsplit("_", 1)[1]) for name in names}
    except (IndexError, ValueError):
        index = None
    if (
        index is not None
        and len(fixed) == len(names)
        and set(index.values()) == set(range(len(names)))
    ):
        return {name: float(fixed[i]) for name, i in index.items()}
    return {
        name: float(np.nanmedian(dtree[name]["elevation"].values)) for name in names
    }


def _select_sweep(obj, sweep=None, elevation=None):
    """The sweep Dataset to use, with the radar site as coordinates."""
    if isinstance(obj, xr.Dataset):
        return obj
    if not isinstance(obj, xr.DataTree):
        raise TypeError(
            f"expected an xarray.DataTree volume or Dataset sweep, not {type(obj)}"
        )
    names = [name for name in obj.children if name.startswith("sweep")]
    if not names:
        raise ValueError("No sweep groups found in DataTree.")
    if sweep is not None:
        name = f"sweep_{sweep}" if isinstance(sweep, (int, np.integer)) else sweep
        if name not in obj.children:
            raise ValueError(f"Sweep {name!r} not found in DataTree.")
        ds = _sweep_dataset(obj, name)
    else:
        elevations = _sweep_elevations(obj, names)
        if elevation is None:
            name = max(names, key=elevations.get)
        else:
            name = min(names, key=lambda n: abs(elevations[n] - elevation))
        ds = _sweep_dataset(obj, name)
    root = obj.root.to_dataset()
    for key in ("latitude", "longitude", "altitude"):
        if key not in ds.coords and key in root:
            ds = ds.assign_coords({key: root[key].reset_coords(drop=True)})
    return ds


def _ray_dim(da):
    dims = [d for d in da.dims if d != "range"]
    if len(dims) != 1 or "range" not in da.dims:
        raise ValueError(
            f"{da.name!r} must be 2-D on (azimuth, range), got dims {da.dims}"
        )
    return dims[0]


def _default_vars(ds):
    return [
        name
        for name, da in ds.data_vars.items()
        if da.ndim == 2 and "range" in da.dims and np.issubdtype(da.dtype, np.number)
    ]


def _site_altitude(ds):
    return float(ds["altitude"].values) if "altitude" in ds else 0.0


def _beam_height(rng, elevation, site_altitude):
    """Gate height above sea level and ground distance (4/3 Earth, as xradar)."""
    from xradar.georeference import antenna_to_cartesian

    x, y, z = antenna_to_cartesian(
        np.asarray(rng, dtype=np.float64),
        0.0,
        elevation,
        earth_radius=EARTH_RADIUS,
        site_altitude=site_altitude,
    )
    return np.asarray(z), np.hypot(x, y)


def _sweep_time(ds):
    if "time" not in ds.coords and "time" not in ds:
        return np.datetime64("NaT", "ns")
    t = np.ravel(ds["time"].values).astype("datetime64[ns]")
    t = t[~np.isnat(t)]
    if t.size == 0:
        return np.datetime64("NaT", "ns")
    return t.min() + (t - t.min()).mean()


# ---------------------------------------------------------------------------
# reductions
# ---------------------------------------------------------------------------


def _reduce_numpy(data, quality, thresholds, modes, min_valid):
    """NumPy implementation of the compiled ``azimuthal_reduce`` (same results)."""
    values, counts = [], []
    for fields, qual, need in zip(data, quality, min_valid):
        nray, ngate = fields[0].shape
        ok = np.ones((nray, ngate), dtype=bool)
        for q, t in zip(qual, thresholds):
            ok &= q.astype(np.float64) > t
        val = np.full((len(modes), ngate), np.nan)
        cnt = np.zeros((len(modes), ngate), dtype=np.int32)
        for v, (x, mode) in enumerate(zip(fields, modes)):
            x = x.astype(np.float64)
            valid = ok & ~np.isnan(x)
            n = valid.sum(axis=0)
            cnt[v] = n
            if mode == _MODES["median"]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # empty columns
                    res = np.nanmedian(np.where(valid, x, np.nan), axis=0)
            else:
                if mode == _MODES["mean_db"]:
                    x = np.exp(x * _LN10_OVER_10)
                with np.errstate(all="ignore"):
                    res = np.where(valid, x, 0.0).sum(axis=0) / n
                    if mode == _MODES["mean_db"]:
                        res = 10.0 * np.log10(res)
            val[v] = np.where((n >= 1) & (n >= need), res, np.nan)
        values.append(val)
        counts.append(cnt)
    return values, counts


def _sweep_arrays(ds, data_vars, quality_vars):
    """Contiguous float32 (ray, range) arrays of a sweep."""
    ray = _ray_dim(ds[data_vars[0]])

    def arr(name):
        da = ds[name]
        if _ray_dim(da) != ray:
            raise ValueError(f"{name!r} is not on the same rays as {data_vars[0]!r}")
        if da.dims != (ray, "range"):
            da = da.transpose(ray, "range")
        return np.ascontiguousarray(da.values, np.float32)

    return [arr(v) for v in data_vars], [arr(q) for q in quality_vars]


def _modes(ds, data_vars, reduction, linear):
    if reduction not in ("mean", "median"):
        raise ValueError(f"reduction must be 'mean' or 'median', not {reduction!r}")
    if reduction == "median":
        return [_MODES["median"]] * len(data_vars)
    if linear is None:
        linear = [
            v
            for v in data_vars
            if str(ds[v].attrs.get("units", "")).strip().lower() in _DB_UNITS
        ]
    elif isinstance(linear, str):
        linear = [linear]
    return [_MODES["mean_db" if v in linear else "mean"] for v in data_vars]


def _profiles(
    sweeps,
    data_vars,
    *,
    min_rhohv,
    min_dbz,
    rhohv,
    dbz,
    min_count,
    min_fraction,
    reduction,
    linear,
    n_threads,
    engine,
):
    """Reduce every sweep; returns the per-sweep profiles and metadata."""
    use_compiled = _use_compiled(engine)
    first = sweeps[0]
    if data_vars is None:
        data_vars = _default_vars(first)
    elif isinstance(data_vars, str):
        data_vars = [data_vars]
    data_vars = list(data_vars)
    if not data_vars:
        raise ValueError("No (azimuth, range) data variables to profile.")
    for v in data_vars:
        if v not in first.data_vars:
            raise ValueError(f"Variable {v!r} not found in the sweep.")
    quality, thresholds = [], []
    for value, limit, names, std, what in (
        (rhohv, min_rhohv, _RHOHV_NAMES, _RHOHV_STANDARD, "rhohv"),
        (dbz, min_dbz, _DBZ_NAMES, _DBZ_STANDARD, "reflectivity"),
    ):
        if limit is None:
            continue
        name = _resolve_field(first, value, names, std, what)
        if name is not None:
            quality.append(name)
            thresholds.append(float(limit))
    modes = _modes(first, data_vars, reduction, linear)

    data, qual, need = [], [], []
    for ds in sweeps:
        d, q = _sweep_arrays(ds, data_vars, quality)
        data.append(d)
        qual.append(q)
        need.append(max(int(min_count), math.ceil(min_fraction * d[0].shape[0]), 1))
    if use_compiled:
        values, counts = _qvp.azimuthal_reduce(
            data, qual, thresholds, modes, need, n_threads=int(n_threads or 0)
        )
    else:
        values, counts = _reduce_numpy(data, qual, thresholds, modes, need)
    filters = {q: t for q, t in zip(quality, thresholds)}
    return data_vars, modes, values, counts, filters


def _var_attrs(ds, name, mode, filters, min_count):
    attrs = dict(ds[name].attrs)
    method = "median" if mode == _MODES["median"] else "mean"
    attrs["cell_methods"] = f"azimuth: {method}"
    if mode == _MODES["mean_db"]:
        attrs["comment"] = "averaged in linear units"
    attrs["qvp_filter"] = (
        " and ".join(f"{q} > {t:g}" for q, t in filters.items()) or "none"
    )
    attrs["qvp_min_count"] = int(min_count)
    return attrs


def _height_coords(height, rng, ground):
    return {
        "height": (
            "height",
            height,
            {
                "standard_name": "altitude",
                "long_name": "beam height above sea level",
                "units": "m",
            },
        ),
        "range": (
            "height",
            rng,
            {"long_name": "slant range of the gate", "units": "m"},
        ),
        "ground_range": (
            "height",
            ground,
            {
                "long_name": "radius of the averaging circle (ground distance)",
                "units": "m",
            },
        ),
    }


def _site_coords(ds):
    return {
        key: ds[key].reset_coords(drop=True)
        for key in ("latitude", "longitude", "altitude")
        if key in ds
    }


def _geometry(ds):
    elevation = float(np.nanmedian(ds["elevation"].values))
    rng = np.asarray(ds["range"].values, dtype=np.float64)
    height, ground = _beam_height(rng, elevation, _site_altitude(ds))
    return elevation, rng, height, ground


_ELEVATION_ATTRS = {
    "standard_name": "sensor_to_target_elevation_angle",
    "long_name": "median elevation angle of the sweep",
    "units": "degrees",
}

_QVP_PARAMS = """
    data_vars : str or list of str, optional
        Variables to profile. By default every numeric ``(azimuth, range)``
        variable of the sweep.
    min_rhohv : float or None, optional
        Only gates with ``rhohv > min_rhohv`` are used. Default 0.6
        (Ryzhkov et al. 2016). ``None`` disables the test.
    min_dbz : float or None, optional
        Only gates with reflectivity ``> min_dbz`` are used. Default -10 dBZ
        (Ryzhkov et al. 2016). ``None`` disables the test.
    rhohv, dbz : str or None, optional
        Names of the fields used for the two tests. ``"auto"`` (default) looks
        for common names (``RHOHV``, ``DBZH``, ...) and the CF standard name;
        a test is skipped if no field is found or the name is ``None``.
    min_count : int, optional
        Minimum number of valid gates on the circle for a defined value.
        Default 30, the number Ryzhkov et al. (2016) require to be exceeded.
    min_fraction : float, optional
        Minimum fraction of the rays that must be valid. Default 0. The
        stricter of ``min_count`` and ``min_fraction`` applies.
    reduction : {"mean", "median"}, optional
        Azimuthal mean (default) or median.
    linear : str or list of str, optional
        Variables averaged in linear units (``10**(x/10)``) and converted back
        to dB. By default the variables with units ``dB``, ``dBZ`` or ``dBm``
        (reflectivity, differential reflectivity). Ignored for the median.
    counts : bool, optional
        Also return ``<name>_count``, the number of valid gates per height.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled
        kernel and falls back to NumPy."""


def qvp(
    obj,
    data_vars=None,
    *,
    sweep=None,
    elevation=None,
    min_rhohv=0.6,
    min_dbz=-10.0,
    rhohv="auto",
    dbz="auto",
    min_count=30,
    min_fraction=0.0,
    reduction="mean",
    linear=None,
    counts=False,
    n_threads=None,
    engine="auto",
):
    """
    Quasi-vertical profile of one sweep.

    Every range gate of a high-elevation PPI is reduced over all rays and
    assigned to its beam height (4/3 Earth model, as in xradar and
    :func:`radarx.grid.grid_cones`), using the median elevation of the sweep.

    Parameters
    ----------
    obj : xarray.DataTree or xarray.Dataset
        A radar volume with ``sweep_*`` groups (xradar layout) or a single
        sweep on ``(azimuth, range)``.
    sweep : int or str, optional
        Sweep of a volume to use, by number or group name. By default the
        sweep with the highest elevation (Ryzhkov et al. 2016 recommend
        elevations of 10°–20° or more).
    elevation : float, optional
        Use the sweep whose elevation is closest to this angle (degrees)
        instead.
    {params}

    Returns
    -------
    xarray.Dataset
        Profiles on the ``height`` dimension (metres above sea level), with
        ``range`` and ``ground_range`` (radius of the averaging circle) as
        coordinates, the sweep ``time`` and ``elevation`` and the radar site.
        Variables keep their attributes; ``cell_methods`` records the
        reduction.

    See Also
    --------
    qvp_timeseries, melting_layer

    References
    ----------
    .. [1] Ryzhkov, A., P. Zhang, H. Reeves, M. Kumjian, T. Tschallener,
       S. Trömel, and C. Simmer, 2016: Quasi-vertical profiles—A new way to
       look at polarimetric radar data. *J. Atmos. Oceanic Technol.*, **33**,
       551–562, https://doi.org/10.1175/JTECH-D-15-0020.1
    .. [2] Trömel, S., M. R. Kumjian, A. V. Ryzhkov, C. Simmer, and
       M. Diederich, 2013: Backscatter differential phase—Estimation and
       variability. *J. Appl. Meteor. Climatol.*, **52**, 2529–2548,
       https://doi.org/10.1175/JAMC-D-13-0124.1

    Examples
    --------
    >>> profile = radarx.retrieve.qvp(dtree, ["DBZH", "ZDR", "RHOHV"])  # doctest: +SKIP
    >>> profile = dtree.radarx.qvp(sweep=9)  # doctest: +SKIP
    """
    ds = _select_sweep(obj, sweep=sweep, elevation=elevation)
    data_vars, modes, values, cnts, filters = _profiles(
        [ds],
        data_vars,
        min_rhohv=min_rhohv,
        min_dbz=min_dbz,
        rhohv=rhohv,
        dbz=dbz,
        min_count=min_count,
        min_fraction=min_fraction,
        reduction=reduction,
        linear=linear,
        n_threads=n_threads,
        engine=engine,
    )
    el, rng, height, ground = _geometry(ds)
    out = {}
    for v, (name, mode) in enumerate(zip(data_vars, modes)):
        out[name] = (
            "height",
            values[0][v].astype(np.float32),
            _var_attrs(ds, name, mode, filters, min_count),
        )
        if counts:
            out[f"{name}_count"] = (
                "height",
                cnts[0][v],
                {"long_name": f"number of valid gates of {name}", "units": "1"},
            )
    coords = _height_coords(height, rng, ground)
    coords["time"] = _sweep_time(ds)
    coords["elevation"] = ((), el, _ELEVATION_ATTRS)
    result = xr.Dataset(out, coords=coords).assign_coords(_site_coords(ds))
    result.attrs = {
        "title": "quasi-vertical profile",
        "references": "Ryzhkov et al. (2016), doi:10.1175/JTECH-D-15-0020.1",
    }
    return result


def _interp_nan(x_new, x, y):
    """Linear interpolation that is NaN wherever a neighbouring sample is NaN."""
    nan = np.isnan(y)
    v = np.interp(x_new, x, np.where(nan, 0.0, y), left=np.nan, right=np.nan)
    bad = np.interp(x_new, x, nan.astype(np.float64), left=1.0, right=1.0)
    return np.where(bad > 0, np.nan, v)


def qvp_timeseries(
    volumes,
    data_vars=None,
    *,
    heights=None,
    sweep=None,
    elevation=None,
    min_rhohv=0.6,
    min_dbz=-10.0,
    rhohv="auto",
    dbz="auto",
    min_count=30,
    min_fraction=0.0,
    reduction="mean",
    linear=None,
    counts=False,
    n_threads=None,
    engine="auto",
):
    """
    Time-height quasi-vertical profiles from a sequence of volumes.

    All sweeps are reduced in one call of the compiled kernel, then put on a
    common height axis.

    Parameters
    ----------
    volumes : sequence of xarray.DataTree or xarray.Dataset
        Radar volumes (or sweeps) in time order.
    heights : array-like, optional
        Common heights above sea level (m). By default the gate heights of the
        first sweep. Profiles whose gate heights differ (other elevation or
        gate spacing) are interpolated linearly in height; values next to a
        missing gate stay missing.
    sweep, elevation : optional
        Sweep selection in each volume, as in :func:`qvp`.
    {params}

    Returns
    -------
    xarray.Dataset
        Profiles on ``(time, height)``, with the ``elevation`` of each sweep
        on ``time``. ``range`` and ``ground_range`` are those of the first
        sweep when ``heights`` is not given.

    See Also
    --------
    qvp, melting_layer

    References
    ----------
    .. [1] Ryzhkov, A., P. Zhang, H. Reeves, M. Kumjian, T. Tschallener,
       S. Trömel, and C. Simmer, 2016: Quasi-vertical profiles—A new way to
       look at polarimetric radar data. *J. Atmos. Oceanic Technol.*, **33**,
       551–562, https://doi.org/10.1175/JTECH-D-15-0020.1

    Examples
    --------
    >>> import xradar as xd  # doctest: +SKIP
    >>> volumes = [xd.io.open_iris_datatree(f) for f in files]  # doctest: +SKIP
    >>> tqvp = radarx.retrieve.qvp_timeseries(volumes, ["DBZH", "ZDR", "RHOHV"])  # doctest: +SKIP
    """
    if isinstance(volumes, (xr.Dataset, xr.DataTree)):
        volumes = [volumes]
    sweeps = [_select_sweep(v, sweep=sweep, elevation=elevation) for v in volumes]
    if not sweeps:
        raise ValueError("No volumes given.")
    data_vars, modes, values, cnts, filters = _profiles(
        sweeps,
        data_vars,
        min_rhohv=min_rhohv,
        min_dbz=min_dbz,
        rhohv=rhohv,
        dbz=dbz,
        min_count=min_count,
        min_fraction=min_fraction,
        reduction=reduction,
        linear=linear,
        n_threads=n_threads,
        engine=engine,
    )
    geometry = [_geometry(ds) for ds in sweeps]
    if heights is None:
        _, rng0, target, ground0 = geometry[0]
        coords = _height_coords(target, rng0, ground0)
    else:
        target = np.asarray(heights, dtype=np.float64)
        coords = {"height": _height_coords(target, target, target)["height"]}
    nt, nh = len(sweeps), target.size
    stacked = np.full((len(data_vars), nt, nh), np.nan, dtype=np.float32)
    stacked_n = np.zeros((len(data_vars), nt, nh), dtype=np.int32)
    for t, (val, cnt, (_, _, h, _)) in enumerate(zip(values, cnts, geometry)):
        same = h.size == nh and np.array_equal(h, target)
        for v in range(len(data_vars)):
            if same:
                stacked[v, t] = val[v]
                stacked_n[v, t] = cnt[v]
            else:
                stacked[v, t] = _interp_nan(target, h, val[v])
                stacked_n[v, t] = np.rint(
                    np.interp(target, h, cnt[v], left=0, right=0)
                ).astype(np.int32)
    out = {}
    for v, (name, mode) in enumerate(zip(data_vars, modes)):
        out[name] = (
            ("time", "height"),
            stacked[v],
            _var_attrs(sweeps[0], name, mode, filters, min_count),
        )
        if counts:
            out[f"{name}_count"] = (
                ("time", "height"),
                stacked_n[v],
                {"long_name": f"number of valid gates of {name}", "units": "1"},
            )
    coords["time"] = (
        "time",
        np.array([_sweep_time(ds) for ds in sweeps], dtype="datetime64[ns]"),
        {"standard_name": "time"},
    )
    coords["elevation"] = ("time", [g[0] for g in geometry], _ELEVATION_ATTRS)
    result = xr.Dataset(out, coords=coords).assign_coords(_site_coords(sweeps[0]))
    result.attrs = {
        "title": "quasi-vertical profiles",
        "references": "Ryzhkov et al. (2016), doi:10.1175/JTECH-D-15-0020.1",
    }
    return result


qvp.__doc__ = qvp.__doc__.replace("{params}", _QVP_PARAMS.strip("\n").lstrip())
qvp_timeseries.__doc__ = qvp_timeseries.__doc__.replace(
    "{params}", _QVP_PARAMS.strip("\n").lstrip()
)


# ---------------------------------------------------------------------------
# melting layer
# ---------------------------------------------------------------------------


def _layer_edge(x, h, i0, step, depth, sign, fraction):
    """NumPy twin of the kernel's ``layer_edge``: last gate before the anomaly
    ``sign * x`` has fallen by ``fraction`` of its prominence, or -1."""
    n = h.size
    peak = sign * x[i0]
    gates = []
    j = i0 + step
    while 0 <= j < n and abs(h[j] - h[i0]) <= depth:
        if not np.isnan(x[j]):
            gates.append(j)
        j += step
    base = min([peak] + [sign * x[j] for j in gates])
    if not base < peak:
        return -1
    threshold = peak - fraction * (peak - base)
    last = i0
    for j in gates:
        if sign * x[j] <= threshold:
            return last
        last = j
    return -1


def _ml_anchor(z, r, d, h, lo, hi, hmin, hmax, rho_lo, rho_hi, zdr_min, dbz_min):
    """Largest ZDR peak with a rhohv minimum and a Z maximum nearby."""
    best = best_rho = -1
    for j in range(h.size):
        if not (hmin <= h[j] <= hmax) or np.isnan(d[j]) or d[j] < zdr_min:
            continue
        if best >= 0 and d[j] <= d[best]:
            continue
        rw, zw = r[lo[j] : hi[j]], z[lo[j] : hi[j]]
        if np.isnan(rw).all():
            continue
        jr = lo[j] + int(np.nanargmin(rw))
        zmax = np.nanmax(zw) if not np.isnan(zw).all() else -np.inf
        if not (rho_lo <= r[jr] <= rho_hi) or zmax < dbz_min:
            continue
        best, best_rho = j, jr
    return best, best_rho


def _onset_edge(x, h, i0, step, depth, rho_onset, onset_fraction):
    """NumPy twin of the kernel's ``onset_edge``: height where rhohv, walking
    away from its minimum at ``i0``, first reaches the background threshold
    ``min(rho_onset, bg - onset_fraction * (bg - x[i0]))``, or NaN."""
    n = h.size
    gates = []
    j = i0 + step
    while 0 <= j < n and abs(h[j] - h[i0]) <= depth:
        if not np.isnan(x[j]):
            gates.append(j)
        j += step
    bg = max([x[i0]] + [x[j] for j in gates])
    t = min(rho_onset, bg - onset_fraction * (bg - x[i0]))
    if not t > x[i0]:
        return np.nan
    prev = i0
    for j in gates:
        if x[j] >= t:
            w = (t - x[prev]) / (x[j] - x[prev])
            return h[prev] + w * (h[j] - h[prev])
        prev = j
    return np.nan  # pragma: no cover - bg >= t lies inside the window


def _melting_layer_numpy(zh, rho, zdr, h, hmin, hmax, *params):
    """NumPy implementation of the compiled ``melting_layer`` (same results)."""
    (
        rho_lo,
        rho_hi,
        zdr_min,
        dbz_min,
        window,
        depth,
        fraction,
        boundaries,
        rho_onset,
        onset_fraction,
    ) = params
    out = np.full((3, zh.shape[0]), np.nan)
    # co-location windows [lo, hi) around every gate
    lo = np.searchsorted(h, h - window, side="left")
    hi = np.searchsorted(h, h + window, side="right")
    for i, (z, r, d) in enumerate(zip(zh, rho, zdr)):
        top_h = hmax[i]
        finite_z = np.flatnonzero(~np.isnan(z))
        if finite_z.size:  # the layer must lie below the echo top
            top_h = min(top_h, h[finite_z[-1]])
        best, best_rho = _ml_anchor(
            z, r, d, h, lo, hi, hmin[i], top_h, rho_lo, rho_hi, zdr_min, dbz_min
        )
        if best < 0:
            continue
        up = _layer_edge(d, h, best, 1, depth, 1.0, fraction)
        dn = _layer_edge(d, h, best, -1, depth, 1.0, fraction)
        if up < 0 or dn < 0:
            continue
        if boundaries == _BOUNDARIES["onset"]:
            top = _onset_edge(r, h, best_rho, 1, depth, rho_onset, onset_fraction)
            bottom = _onset_edge(r, h, best_rho, -1, depth, rho_onset, onset_fraction)
            top = h[up] if np.isnan(top) else top
            bottom = h[dn] if np.isnan(bottom) else bottom
            out[:, i] = top, bottom, h[best]
            continue
        top, bottom = h[up], h[dn]
        rup = _layer_edge(r, h, best_rho, 1, depth, -1.0, fraction)
        rdn = _layer_edge(r, h, best_rho, -1, depth, -1.0, fraction)
        if rup >= 0:
            top = max(top, h[rup])
        if rdn >= 0:
            bottom = min(bottom, h[rdn])
        out[:, i] = top, bottom, h[best]
    return out


_ML_FLAGS = {"none": 0, "detected": 1, "rejected": 2, "filled": 3}
_BOUNDARIES = {"half_prominence": 0, "onset": 1}


def _running_median(x, size):
    """Centred running median ignoring NaN (shorter windows at the ends)."""
    half = size // 2
    out = np.full(x.size, np.nan)
    for i in range(x.size):
        w = x[max(0, i - half) : i + half + 1]
        if np.isfinite(w).any():
            out[i] = np.nanmedian(w)
    return out


def _time_consistency(top, bottom, peak, time, median_window, max_jump, max_fill):
    """Reject detections far from the running median and fill short gaps."""
    flag = np.where(np.isfinite(peak), _ML_FLAGS["detected"], _ML_FLAGS["none"])
    if median_window > 1 and top.size > 2:
        mid = 0.5 * (top + bottom)
        with np.errstate(invalid="ignore"):
            bad = np.abs(mid - _running_median(mid, median_window)) > max_jump
        flag[bad] = _ML_FLAGS["rejected"]
    good = flag == _ML_FLAGS["detected"]
    top, bottom, peak = (np.where(good, v, np.nan) for v in (top, bottom, peak))
    if max_fill > 0 and good.sum() >= 2:
        idx = np.flatnonzero(good)
        t = (time - time[0]).astype("timedelta64[ns]").astype(np.float64)
        pos = np.arange(top.size)
        before = np.searchsorted(idx, pos, side="right") - 1
        after = np.searchsorted(idx, pos, side="left")
        inner = (before >= 0) & (after < idx.size)
        gap = np.full(top.size, top.size)
        gap[inner] = idx[after[inner]] - idx[before[inner]] - 1
        fill = ~good & inner & (gap <= max_fill)
        for v in (top, bottom, peak):
            v[fill] = np.interp(t[fill], t[good], v[good])
        flag[fill] = _ML_FLAGS["filled"]
    return top, bottom, peak, flag


def _environment_levels(environment, engine, n_threads):
    """0 °C and wet-bulb 0 °C heights from a sounding or ERA5 profile, or from
    a mapping with ``freezing_level`` and ``wet_bulb_zero_height``."""
    if environment is None:
        return None, None
    if isinstance(environment, xr.Dataset) and "height" in environment.dims:
        from ..io.sounding import isotherm_height, wet_bulb_zero_height

        if "temperature" not in environment:
            raise ValueError("the environment profile needs a 'temperature'.")
        fl = isotherm_height(environment, engine=engine, n_threads=n_threads)
        wbz = None
        if {"pressure", "dewpoint"} <= set(environment.data_vars):
            wbz = wet_bulb_zero_height(environment, engine=engine, n_threads=n_threads)
        return fl, wbz
    if isinstance(environment, (xr.Dataset, dict)):
        fl = environment.get("freezing_level")
        wbz = environment.get("wet_bulb_zero_height")
        if fl is None and wbz is None:
            raise ValueError(
                "environment needs 'freezing_level' or 'wet_bulb_zero_height'."
            )
        return fl, wbz
    raise TypeError(
        "environment must be a profile Dataset on 'height' (radarx.io.sounding) "
        f"or a mapping of reference heights, not {type(environment)}"
    )


def _reference_height(value, template, other):
    """A scalar or DataArray height broadcast to the profiles' non-height
    dimensions; a ``time`` dimension is interpolated linearly to the profile
    times (nearest value outside its range)."""
    da = value if isinstance(value, xr.DataArray) else xr.DataArray(value)
    da = da.astype(np.float64).reset_coords(drop=True)
    if "time" in da.dims and "time" not in da.coords:
        da = da.isel(time=0, drop=True) if da.sizes["time"] == 1 else da
    elif "time" in da.dims:
        if "time" in template.coords and da.sizes["time"] > 1:
            t = template["time"].values
            near = da.sel(time=t, method="nearest").drop_vars("time")
            da = da.interp(time=t).drop_vars("time").fillna(near)
            if np.ndim(t):
                da = da.assign_coords(time=t)
        elif da.sizes["time"] == 1:
            da = da.isel(time=0, drop=True)
        else:
            raise ValueError(
                "a reference height on several times needs profiles with a time."
            )
    ref = xr.broadcast(da, template.isel(height=0, drop=True))[0]
    return ref.transpose(*other).reset_coords(drop=True)


_REFERENCE_LABELS = {
    "freezing_level": ("0 degC", "height of the 0 degC level above sea level"),
    "wet_bulb_zero_height": (
        "wet-bulb 0 degC",
        "height of the 0 degC wet-bulb temperature above sea level",
    ),
    "reference": ("reference", "reference height above sea level"),
}


def _reference_heights(environment, freezing_level, template, other, engine, n_threads):
    """Reference heights to report, the height the search is centred on, and
    whether the reported ``freezing_level`` is the user's own."""
    env_fl, env_wbz = _environment_levels(environment, engine, n_threads)
    given = environment is None and freezing_level is not None
    refs = {
        key: _reference_height(value, template, other)
        for key, value in (
            ("freezing_level", freezing_level if given else env_fl),
            ("wet_bulb_zero_height", env_wbz),
        )
        if value is not None
    }
    if freezing_level is not None:
        return refs, _reference_height(freezing_level, template, other), given
    search = refs.get("wet_bulb_zero_height", refs.get("freezing_level"))
    if search is not None and "freezing_level" in refs:
        # the 0 °C height where the wet-bulb 0 °C height is missing
        search = search.fillna(refs["freezing_level"])
    return refs, search, given


def _ml_method(boundaries, onset_fraction, rhohv_onset):
    """``method`` attribute of the melting-layer heights."""
    method = "co-located rhohv minimum and ZDR/Z maximum (after Giangrande et al. 2008)"
    if boundaries == "half_prominence":
        return (
            method + "; top and bottom at half the prominence of the ZDR/rhohv anomaly"
        )
    method += (
        "; top and bottom where rhohv departs from its background by "
        f"{float(onset_fraction):g} of the dip"
    )
    if rhohv_onset is not None:
        method += f" or reaches {float(rhohv_onset):g}"
    return method + " (after Griffin et al. 2020)"


def _search_range(reference, height_range, window, n_prof):
    """Per-profile (hmin, hmax) for the ZDR peak: ``window`` around the
    reference height, else (and where it is missing) ``height_range``."""
    hmin = np.full(n_prof, float(height_range[0]))
    hmax = np.full(n_prof, float(height_range[1]))
    if reference is None:
        return hmin, hmax
    ref = np.asarray(reference.values, np.float64).reshape(-1)
    ok = np.isfinite(ref)
    hmin[ok] = ref[ok] + float(window[0])
    hmax[ok] = ref[ok] + float(window[1])
    return hmin, hmax


def _apply_time_consistency(top, bottom, peak, flag, template, other, **kwargs):
    """Run :func:`_time_consistency` along ``time`` for every other index."""
    axis = other.index("time")
    arrays = [np.moveaxis(v, axis, -1).copy() for v in (top, bottom, peak, flag)]
    time = np.asarray(template["time"].values)
    if time.dtype.kind != "M":
        time = time.astype(np.float64).astype("datetime64[ns]")
    for idx in np.ndindex(arrays[0].shape[:-1]):
        res = _time_consistency(*(a[idx] for a in arrays[:3]), time, **kwargs)
        for a, v in zip(arrays, res):
            a[idx] = v
    return [np.moveaxis(a, -1, axis) for a in arrays]


def melting_layer(
    profiles,
    *,
    dbz="auto",
    rhohv="auto",
    zdr="auto",
    boundaries="onset",
    environment=None,
    height_range=(1000.0, 6000.0),
    freezing_level=None,
    freezing_level_window=(-1000.0, 500.0),
    rhohv_range=(0.80, 0.97),
    zdr_min=0.5,
    dbz_min=20.0,
    window=500.0,
    depth=1000.0,
    rhohv_onset=None,
    onset_fraction=0.1,
    edge_fraction=0.5,
    median_window=5,
    max_jump=500.0,
    max_fill=3,
    n_threads=None,
    engine="auto",
):
    """
    Melting-layer top and bottom from quasi-vertical profiles.

    The melting layer shows up in polarimetric data as a ρhv minimum together
    with maxima of ZDR and Z, and the ρhv signature discriminates best
    (Giangrande et al. 2008). This function looks for that co-located
    signature in each profile, measures its depth, and checks the result for
    consistency along a time series:

    1. **Signature.** Within the search range (``height_range``, or
       ``freezing_level_window`` around the reference height, and always
       below the echo top), the anchor is the largest ZDR value
       ``>= zdr_min`` that has, within ``window`` metres, a ρhv minimum inside
       ``rhohv_range`` and a Z maximum ``>= dbz_min``. ρhv dips without a ZDR
       and Z enhancement (e.g. near the echo top or in noisy data aloft) are
       not taken, nor is a ZDR peak that does not fall to half its prominence
       within ``depth`` on both sides.
    2. **Top and bottom.** ``boundaries`` selects the definition:

       - ``"onset"`` (default): where the signature begins and ends. The ρhv
         dip is the most objective marker of melting: its top is at the
         0 °C wet-bulb level, where melting starts, and its bottom where the
         snow has melted (Ryzhkov and Krause 2022). Going up and down from
         the ρhv minimum, as Griffin et al. (2020) do, the edge is the first
         height where ρhv is back at its background: within
         ``onset_fraction`` (10 %) of the dip depth from the background,
         the largest ρhv within ``depth`` on that side. Griffin et al. (2020)
         use a fixed background, ρhv ``>= 0.97`` in S-band QVPs; pass
         ``rhohv_onset=0.97`` for their definition (the edge is then
         wherever either test is met first). Heights are interpolated
         linearly between gates. If ρhv does not rise on one side, the ZDR
         half-prominence edge is used.
       - ``"half_prominence"``: going up and down from the ZDR peak and from
         the ρhv minimum, each edge is the last gate before the anomaly has
         fallen by ``edge_fraction`` of its prominence over the background
         within ``depth`` metres; the layer spans the union of the ZDR and
         ρhv anomalies. This is the width at half height of the anomaly, so
         the layer is shallower than with ``"onset"`` and its top lies below
         the level where melting starts.

    3. **Time consistency** (profiles with a ``time`` dimension). Detections
       whose mid-height differs by more than ``max_jump`` from the running
       median over ``median_window`` profiles are rejected, and gaps of up to
       ``max_fill`` profiles between accepted detections are filled by linear
       interpolation in time. ``melting_layer_flag`` records what happened.

    Steps 1 and 2 run in the compiled kernel, one profile per work item.

    Parameters
    ----------
    profiles : xarray.Dataset
        QVPs on an ascending ``height`` dimension (m above sea level), e.g.
        from :func:`qvp` or :func:`qvp_timeseries`, with Z, ZDR and ρhv.
    dbz, rhohv, zdr : str, optional
        Variable names; ``"auto"`` (default) looks for common names and CF
        standard names.
    boundaries : {"onset", "half_prominence"}, optional
        Definition of the top and bottom (step 2). Default ``"onset"``.
    environment : xarray.Dataset or dict, optional
        The thermodynamic environment: a sounding or ERA5 profile from
        :mod:`radarx.io.sounding` (:func:`~radarx.io.sounding.read_sounding`,
        :func:`~radarx.io.sounding.era5_profile`, ``dtree.radarx.sounding()``;
        several profiles concatenated on ``time`` are interpolated to the
        profile times), or a mapping or Dataset with ``freezing_level`` and/or
        ``wet_bulb_zero_height`` (m above sea level, scalars or DataArrays).
        Its 0 °C height (:func:`~radarx.io.sounding.isotherm_height`) and
        wet-bulb 0 °C height (:func:`~radarx.io.sounding.wet_bulb_zero_height`,
        if the profile has ``pressure`` and ``dewpoint``) are returned with
        the offsets of the layer top from them, and, unless
        ``freezing_level`` is given, the search is constrained to
        ``freezing_level_window`` around the wet-bulb 0 °C height (the
        0 °C height if there is none).
    height_range : tuple of float, optional
        Heights (m above sea level) searched for the ZDR peak without a
        reference height (and where the reference height is missing).
        Default 1–6 km (Giangrande et al. 2008 only flag gates below 6 km).
    freezing_level : float or xarray.DataArray, optional
        Reference height for the search (m above sea level), e.g. the 0 °C
        level, a scalar or one value per profile (e.g. on ``time``). If
        given, the ZDR peak is searched within ``freezing_level_window`` of
        it; it overrides the reference height from ``environment``.
    freezing_level_window : tuple of float, optional
        Search range for the ZDR peak relative to the reference height,
        default −1000 to +500 m (melting happens below the 0 °C level).
    rhohv_range : tuple of float, optional
        Range for the ρhv minimum near the ZDR peak, default (0.80, 0.97).
    zdr_min : float, optional
        Minimum ZDR peak, default 0.5 dB.
    dbz_min : float, optional
        Minimum Z maximum near the ZDR peak, default 20 dBZ.
    window : float, optional
        Co-location distance of the ρhv minimum and Z maximum from the ZDR
        peak, default 500 m.
    depth : float, optional
        How far above and below the peak the background is sought, default
        1000 m.
    rhohv_onset : float, optional
        A fixed background ρhv for ``boundaries="onset"``, e.g. 0.97 as in
        Griffin et al. (2020). Default None: only ``onset_fraction``.
    onset_fraction : float, optional
        With ``boundaries="onset"``, the edge is where ρhv is within this
        fraction of the dip depth from the background, default 0.1.
    edge_fraction : float, optional
        Fraction of the prominence the anomaly must fall by at the edges with
        ``boundaries="half_prominence"``, default 0.5.
    median_window : int, optional
        Profiles in the running median, default 5. 1 disables the check.
    max_jump : float, optional
        Largest accepted distance from the running median, default 500 m.
    max_fill : int, optional
        Longest gap (in profiles) filled by interpolation, default 3; 0
        disables filling.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation of steps 1 and 2.

    Returns
    -------
    xarray.Dataset
        ``melting_layer_top``, ``melting_layer_bottom`` and
        ``melting_layer_peak`` (height of the ZDR peak), in m above sea level,
        and ``melting_layer_flag`` (0 none, 1 detected, 2 rejected as
        inconsistent in time, 3 filled in time), on the non-height dimensions
        of ``profiles``. With a reference height (``environment`` or
        ``freezing_level``) also ``freezing_level`` (0 °C height, or the
        given ``freezing_level``), ``wet_bulb_zero_height`` (if the
        environment has it) and ``melting_layer_top_offset_freezing_level`` /
        ``melting_layer_top_offset_wet_bulb_zero``, the height of the layer
        top above those levels.

    Notes
    -----
    Giangrande et al. (2008) flag individual radar gates with
    0.90 < ρhv < 0.97 and nearby Z of 30–47 dBZ and ZDR > 0.8 dB. Azimuthal
    averaging in a QVP smooths and weakens these extremes (Z is averaged over
    the whole circle, including weaker echo), so the defaults here are lower.
    Tune them for other radars and elevations.

    Griffin et al. (2020) define the top and bottom of the melting layer in
    S-band QVPs by searching upward and downward from the ρhv minimum for the
    first ρhv ``>= 0.97``; the reflectivity-curvature method of Fabry and
    Zawadzki (1995) gave tops about 200 m higher and bottoms within about
    50 m. Where the background is close to 1, as at S band, a fixed 0.97 is
    reached well inside the dip; the default relative test finds where ρhv
    first departs from its background, wherever that background lies (it is
    lower at C and X band and in noisy data). On the KGWX QVPs of 30–31
    March 2022 the default top is about 190 m above the 0.97 top, the
    difference Griffin et al. (2020) report against the curvature method.
    Ryzhkov and Krause (2022) place the top of the ρhv dip at the 0 °C
    wet-bulb level and its bottom near +3 °C and estimate both from QVPs to
    about 0.1 km, so the ``"onset"`` top is expected near the wet-bulb 0 °C
    height of the environment; the returned offsets quantify the difference.

    References
    ----------
    .. [1] Giangrande, S. E., J. M. Krause, and A. V. Ryzhkov, 2008: Automatic
       designation of the melting layer with a polarimetric prototype of the
       WSR-88D radar. *J. Appl. Meteor. Climatol.*, **47**, 1354–1364,
       https://doi.org/10.1175/2007JAMC1634.1
    .. [2] Ryzhkov, A., P. Zhang, H. Reeves, M. Kumjian, T. Tschallener,
       S. Trömel, and C. Simmer, 2016: Quasi-vertical profiles—A new way to
       look at polarimetric radar data. *J. Atmos. Oceanic Technol.*, **33**,
       551–562, https://doi.org/10.1175/JTECH-D-15-0020.1
    .. [3] Griffin, E. M., T. J. Schuur, and A. V. Ryzhkov, 2020: A
       polarimetric radar analysis of ice microphysical processes in melting
       layers of winter storms using S-band quasi-vertical profiles. *J.
       Appl. Meteor. Climatol.*, **59**, 751–767,
       https://doi.org/10.1175/JAMC-D-19-0128.1
    .. [4] Ryzhkov, A., and J. Krause, 2022: New polarimetric radar algorithm
       for melting-layer detection and determination of its height. *J.
       Atmos. Oceanic Technol.*, **39**, 529–543,
       https://doi.org/10.1175/JTECH-D-21-0130.1
    .. [5] Fabry, F., and I. Zawadzki, 1995: Long-term radar observations of
       the melting layer of precipitation and their interpretation. *J.
       Atmos. Sci.*, **52**, 838–851,
       https://doi.org/10.1175/1520-0469(1995)052<0838:LTROOT>2.0.CO;2

    Examples
    --------
    >>> tqvp = radarx.retrieve.qvp_timeseries(volumes)  # doctest: +SKIP
    >>> ml = radarx.retrieve.melting_layer(tqvp)  # doctest: +SKIP
    """
    use_compiled = _use_compiled(engine)
    if not isinstance(profiles, xr.Dataset) or "height" not in profiles.dims:
        raise ValueError("profiles must be a Dataset with a 'height' dimension.")
    names = {}
    for key, value, cands, std in (
        ("dbz", dbz, _DBZ_NAMES, _DBZ_STANDARD),
        ("rhohv", rhohv, _RHOHV_NAMES, _RHOHV_STANDARD),
        ("zdr", zdr, _ZDR_NAMES, _ZDR_STANDARD),
    ):
        names[key] = _resolve_field(profiles, value, cands, std, key)
        if names[key] is None:
            raise ValueError(f"melting_layer needs a {key} profile.")
    h = np.asarray(profiles["height"].values, dtype=np.float64)
    if h.size > 1 and np.any(np.diff(h) <= 0):
        raise ValueError("height must be strictly ascending.")
    template = profiles[names["dbz"]]
    other = [d for d in template.dims if d != "height"]
    shape = tuple(template.sizes[d] for d in other)

    def as2d(name):
        da = profiles[name].transpose(*other, "height")
        return np.ascontiguousarray(da.values.reshape(-1, h.size), np.float64)

    if boundaries not in _BOUNDARIES:
        raise ValueError(
            f"boundaries must be 'onset' or 'half_prominence', not {boundaries!r}"
        )
    zh, rho, zd = (as2d(names[k]) for k in ("dbz", "rhohv", "zdr"))
    refs, search, given = _reference_heights(
        environment, freezing_level, template, other, engine, n_threads
    )
    hmin, hmax = _search_range(search, height_range, freezing_level_window, len(zh))
    params = (
        *(float(v) for v in (*rhohv_range, zdr_min, dbz_min, window, depth)),
        float(edge_fraction),
        _BOUNDARIES[boundaries],
        np.inf if rhohv_onset is None else float(rhohv_onset),
        float(onset_fraction),
    )
    if use_compiled:
        top, bottom, peak = _qvp.melting_layer(
            zh, rho, zd, h, hmin, hmax, *params, n_threads=int(n_threads or 0)
        )
    else:
        top, bottom, peak = _melting_layer_numpy(zh, rho, zd, h, hmin, hmax, *params)
    top, bottom, peak = (np.asarray(v).reshape(shape) for v in (top, bottom, peak))
    flag = np.where(np.isfinite(peak), _ML_FLAGS["detected"], _ML_FLAGS["none"])
    if "time" in other:
        top, bottom, peak, flag = _apply_time_consistency(
            top,
            bottom,
            peak,
            flag,
            template,
            other,
            median_window=int(median_window),
            max_jump=float(max_jump),
            max_fill=int(max_fill),
        )
    coords = {name: c for name, c in template.coords.items() if "height" not in c.dims}
    method = _ml_method(boundaries, onset_fraction, rhohv_onset)

    def wrap(values, long_name):
        return xr.DataArray(
            values,
            dims=other,
            coords=coords,
            attrs={
                "standard_name": "altitude",
                "long_name": long_name,
                "units": "m",
                "method": method,
            },
        )

    out = xr.Dataset(
        {
            "melting_layer_top": wrap(top, "melting layer top above sea level"),
            "melting_layer_bottom": wrap(
                bottom, "melting layer bottom above sea level"
            ),
            "melting_layer_peak": wrap(
                peak, "height of the ZDR peak in the melting layer above sea level"
            ),
        }
    )
    out["melting_layer_flag"] = xr.DataArray(
        np.asarray(flag, dtype=np.int8),
        dims=other,
        coords=coords,
        attrs={
            "long_name": "melting layer detection quality flag",
            "flag_values": np.array(list(_ML_FLAGS.values()), dtype=np.int8),
            "flag_meanings": " ".join(_ML_FLAGS),
        },
    )
    out.attrs["melting_layer_boundaries"] = boundaries
    for key, ref in refs.items():
        what, long_name = _REFERENCE_LABELS["reference" if given else key]
        values = np.asarray(ref.values, np.float64).reshape(shape)
        out[key] = xr.DataArray(
            values,
            dims=other,
            coords=coords,
            attrs={"standard_name": "altitude", "long_name": long_name, "units": "m"},
        )
        suffix = "wet_bulb_zero" if key == "wet_bulb_zero_height" else key
        out[f"melting_layer_top_offset_{suffix}"] = xr.DataArray(
            np.asarray(top, np.float64) - values,
            dims=other,
            coords=coords,
            attrs={
                "long_name": f"melting layer top minus the {what} height",
                "units": "m",
            },
        )
    return out
