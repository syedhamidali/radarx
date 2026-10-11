#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Quasi-Vertical Profiles
=======================

Quasi-vertical profiles (QVPs, Ryzhkov et al. 2016 [1]) are azimuthal
averages of the polarimetric variables measured on a high-elevation PPI.
Each range gate of the sweep is reduced over all rays to a single value and
assigned to the beam height of that gate, so one sweep gives one vertical
profile and a sequence of volumes gives a time-height display. Ryzhkov et al.
(2016) average at elevations exceeding 10 to 20 degrees (p. 552); the idea
of azimuthal averaging at high elevation to reduce the noise of the melting
layer signatures goes back to Trömel et al. (2013) [2] (as summarised by
Ryzhkov et al. 2016, p. 552; not checked against the 2013 paper).

Following Ryzhkov et al. (2016) [1], p. 553, only gates with ``rhohv > 0.6``
and ``Z > -10 dBZ`` (both strict inequalities, as in the paper) are used. The
paper requires that the number of valid gates along the circle "should
exceed 30" for an estimate; radarx defines a value where the number of valid
gates is at least ``min_count`` (default 30), so a circle with exactly 30
valid gates gives a value here although it does not in the paper (``min_count=31``
reproduces the paper). Quantities in decibels (reflectivity, differential
reflectivity) are averaged in linear units and converted back; the other
variables are averaged as they are. A median is available as a robust
alternative. Linear averaging of decibel quantities and the median are
radarx's own choices, not statements of the paper. The melting layer can be
detected in the profiles from the co-located rhohv minimum and ZDR / Z
maxima (after Giangrande et al. 2008 [3]; see :func:`melting_layer` for what
differs), its top and bottom placed where rhohv returns to its background
(Griffin et al. 2020 [4]), compared with the 0 °C and wet-bulb 0 °C heights
of a sounding or ERA5 profile, and checked for consistency along the time
series (radarx).

Profiles that do not need a high sweep. :func:`rd_qvp` combines the
azimuthal means of all sweeps within a range of the radar (the range-defined
QVP of Tobin and Kumjian 2017 [5], as described by Murphy et al. 2020 [6]; the
weights are radarx's own), :func:`cvp` is the columnar vertical profile of a
sector around a point (Murphy et al. 2020 [6]) and :func:`rhi_qvp` the profile
of an RHI scan (Allabakash et al. 2019 [7]). :func:`melting_layer_ppi` finds
the melting layer on PPIs between 4 and 10 degrees with the radial-profile rules
of Giangrande et al. (2008) [3], and :func:`mask_above_melting_layer` removes
the products of rain-only retrievals above its bottom.

The reductions over (time, azimuth, range) and the averaging in height are done
by a compiled C++ kernel in a single multithreaded pass; if it is not available,
an equivalent NumPy implementation is used.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}

References
----------
.. [1] Ryzhkov, A., P. Zhang, H. Reeves, M. Kumjian, T. Tschallener, S.
   Trömel, and C. Simmer, 2016: Quasi-vertical profiles—A new way to look at
   polarimetric radar data. *J. Atmos. Oceanic Technol.*, **33** (3),
   551–562, https://doi.org/10.1175/JTECH-D-15-0020.1
.. [2] Trömel, S., M. R. Kumjian, A. V. Ryzhkov, C. Simmer, and M.
   Diederich, 2013: Backscatter differential phase—Estimation and
   variability. *J. Appl. Meteor. Climatol.*, **52** (11), 2529–2548,
   https://doi.org/10.1175/JAMC-D-13-0124.1
.. [3] Giangrande, S. E., J. M. Krause, and A. V. Ryzhkov, 2008: Automatic
   designation of the melting layer with a polarimetric prototype of the
   WSR-88D radar. *J. Appl. Meteor. Climatol.*, **47** (5), 1354–1364,
   https://doi.org/10.1175/2007JAMC1634.1
.. [4] Griffin, E. M., T. J. Schuur, and A. V. Ryzhkov, 2020: A polarimetric
   radar analysis of ice microphysical processes in melting layers of winter
   storms using S-band quasi-vertical profiles. *J. Appl. Meteor. Climatol.*,
   **59** (4), 751–767, https://doi.org/10.1175/JAMC-D-19-0128.1
.. [5] Tobin, D. M., and M. R. Kumjian, 2017: Polarimetric radar and
   surface-based precipitation-type observations of ice pellet to freezing rain
   transitions. *Wea. Forecasting*, **32** (6), 2065–2082,
   https://doi.org/10.1175/WAF-D-17-0054.1
.. [6] Murphy, A. M., A. Ryzhkov, and P. Zhang, 2020: Columnar vertical profile
   (CVP) methodology for validating polarimetric radar retrievals in ice using
   in situ aircraft measurements. *J. Atmos. Oceanic Technol.*, **37** (9),
   1623–1642, https://doi.org/10.1175/JTECH-D-20-0011.1
.. [7] Allabakash, S., S. Lim, and B.-J. Jang, 2019: Melting layer detection and
   characterization based on range height indicator–quasi vertical profiles.
   *Remote Sens.*, **11** (23), 2848, https://doi.org/10.3390/rs11232848
"""

from __future__ import annotations

__all__ = [
    "cvp",
    "mask_above_melting_layer",
    "melting_layer",
    "melting_layer_ppi",
    "qvp",
    "qvp_timeseries",
    "rd_qvp",
    "rhi_qvp",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import math
import warnings

import numpy as np
import xarray as xr

from .._provenance import provenance
from .._registry import accessor_method

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
    return _with_site(ds, obj)


def _with_site(ds, dtree):
    """``ds`` with the radar site of the root of ``dtree`` as coordinates."""
    root = dtree.root.to_dataset()
    for key in ("latitude", "longitude", "altitude"):
        if key not in ds.coords and key in root:
            ds = ds.assign_coords({key: root[key].reset_coords(drop=True)})
    return ds


def _volume_sweeps(obj):
    """Every sweep of a volume (xradar layout), or the sweep itself, as Datasets."""
    if isinstance(obj, xr.Dataset):
        return [obj]
    if not isinstance(obj, xr.DataTree):
        raise TypeError(
            f"expected an xarray.DataTree volume or Dataset sweep, not {type(obj)}"
        )
    names = [name for name in obj.children if name.startswith("sweep")]
    if not names:
        raise ValueError("No sweep groups found in DataTree.")
    return [_with_site(_sweep_dataset(obj, name), obj) for name in names]


def _ray_dim(da):
    dims = [d for d in da.dims if d != "range"]
    if len(dims) != 1 or "range" not in da.dims:
        raise ValueError(
            f"{da.name!r} must be 2-D on (azimuth, range), got dims {da.dims}"
        )
    return dims[0]


_ZH_ROLES = (
    _DBZ_NAMES,
    _ZDR_NAMES,
    _RHOHV_NAMES,
    ("KDP", "specific_differential_phase", "corrected_specific_differential_phase"),
)
_PHASE_NAMES = {
    "phidp",
    "uphidp",
    "phidp_processed",
    "phi",
    "differential_phase",
    "uncorrected_differential_phase",
    "corrected_differential_phase",
}


def _is_2d_numeric(da):
    return da.ndim == 2 and "range" in da.dims and np.issubdtype(da.dtype, np.number)


def _default_vars(ds):
    """
    Reflectivity, differential reflectivity, rhohv and KDP, as far as the
    sweep has them. Radial velocity and the differential phase do not have a
    useful azimuthal mean as a default. If the sweep has none of the four,
    every numeric (azimuth, range) variable.
    """
    names = []
    for candidates in _ZH_ROLES:
        found = next((n for n in candidates if n in ds.data_vars), None)
        if found is not None and _is_2d_numeric(ds[found]):
            names.append(found)
    if names:
        return names
    return [name for name, da in ds.data_vars.items() if _is_2d_numeric(da)]


def _is_phase(ds, name):
    return name.lower() in _PHASE_NAMES and str(
        ds[name].attrs.get("units", "degrees")
    ).strip().lower() in ("degrees", "degree", "deg", "degrees_north", "")


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


def _fields_of(first, data_vars, rhohv, dbz, min_rhohv, min_dbz):
    """Names of the data variables and the quality fields (with thresholds)."""
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
    return data_vars, quality, thresholds


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
    elevation_correction=False,
    warn_low=True,
):
    """Reduce every sweep; returns the per-sweep profiles and metadata."""
    use_compiled = _use_compiled(engine)
    first = sweeps[0]
    data_vars, quality, thresholds = _fields_of(
        first, data_vars, rhohv, dbz, min_rhohv, min_dbz
    )
    modes = _modes(first, data_vars, reduction, linear)
    # the mean of a phase is the circular mean: reduce cos and sin instead
    circular = [
        v
        for v, name in enumerate(data_vars)
        if reduction == "mean" and _is_phase(first, name)
    ]
    kernel_modes = list(modes) + [_MODES["mean"]] * (2 * len(circular))

    data, qual, need = [], [], []
    for ds in sweeps:
        d, q = _sweep_arrays(ds, data_vars, quality)
        for v in circular:
            rad = np.deg2rad(d[v].astype(np.float64))
            d.extend([np.cos(rad).astype(np.float32), np.sin(rad).astype(np.float32)])
        data.append(d)
        qual.append(q)
        need.append(max(int(min_count), math.ceil(min_fraction * d[0].shape[0]), 1))
    if use_compiled:
        values, counts = _qvp.azimuthal_reduce(
            data, qual, thresholds, kernel_modes, need, n_threads=int(n_threads or 0)
        )
    else:
        values, counts = _reduce_numpy(data, qual, thresholds, kernel_modes, need)
    if circular:
        nv = len(data_vars)
        values, counts = list(values), list(counts)
        for i, (val, cnt) in enumerate(zip(values, counts)):
            val = np.array(val, dtype=np.float64)
            for j, v in enumerate(circular):
                c, sn = val[nv + 2 * j], val[nv + 2 * j + 1]
                with np.errstate(invalid="ignore"):
                    angle = np.rad2deg(np.arctan2(sn, c))
                val[v] = np.where(np.isnan(c) | np.isnan(sn), np.nan, angle)
            values[i], counts[i] = val[:nv], cnt[:nv]
    filters = {q: t for q, t in zip(quality, thresholds)}
    _check_elevation(sweeps, data_vars, values, elevation_correction, warn_low)
    return data_vars, modes, values, counts, filters


_KDP_NAMES = {n.lower() for n in _ZH_ROLES[3]}
_ZDR_LOWER = {n.lower() for n in _ZDR_NAMES}
_LOW_ELEVATION = 10.0


def _zdr_to_horizontal(zdr, elevation):
    """
    ZDR (dB) that the scatterers would show at 0 degrees elevation, from the
    ZDR (dB) measured at ``elevation`` (degrees).

    The inverse of the relation of :func:`radarx.retrieve.zdr_offset`
    (``ZDR(t) = ZDR0 - 20 log10(10**(ZDR0/20) sin(t)**2 + cos(t)**2)``,
    Rayleigh scattering by spheroids with a vertical axis and no canting,
    derived in radarx and not checked against a paper):
    ``ZDR0 = ZDR(t) + 20 log10(cos(t)**2) - 20 log10(1 - 10**(ZDR(t)/20) sin(t)**2)``.
    NaN where the measured ZDR is outside what the relation can produce.
    """
    t = np.deg2rad(elevation)
    r = np.power(10.0, 0.05 * np.asarray(zdr, dtype=np.float64))
    s2, c2 = np.sin(t) ** 2, np.cos(t) ** 2
    denom = 1.0 - r * s2
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (
            zdr
            + 20.0 * np.log10(c2)
            - 20.0 * np.log10(np.where(denom > 0, denom, np.nan))
        )
    return out


def _check_elevation(sweeps, data_vars, values, correct, warn=True):
    """
    Warn for low sweeps; if asked, scale KDP by 1 / cos(elevation)**2 and
    refer ZDR to 0 degrees elevation.
    """
    low = []
    for i, ds in enumerate(sweeps):
        el = float(np.nanmedian(ds["elevation"].values))
        if el < _LOW_ELEVATION:
            low.append(el)
        if correct:
            for v, name in enumerate(data_vars):
                if name.lower() in _KDP_NAMES:
                    values[i][v] = values[i][v] / np.cos(np.deg2rad(el)) ** 2
                elif name.lower() in _ZDR_LOWER:
                    values[i][v] = _zdr_to_horizontal(values[i][v], el)
    if low and warn:
        warnings.warn(
            f"QVP of a sweep at {min(low):.1f} degrees elevation: the method "
            "assumes high elevations (more than 10 to 20 degrees, Ryzhkov et al. "
            "2016), and the azimuthal means of ZDR and KDP are lower than "
            "their values at the horizontal (KDP by about cos(elevation)**2; "
            "elevation_correction=True corrects KDP and ZDR)",
            UserWarning,
            stacklevel=4,
        )


def _var_attrs(ds, name, mode, filters, min_count, correct=False):
    attrs = dict(ds[name].attrs)
    if correct and name.lower() in _KDP_NAMES:
        attrs["elevation_correction"] = "divided by cos(elevation)**2"
    elif correct and name.lower() in _ZDR_LOWER:
        attrs["elevation_correction"] = "referred to 0 degrees elevation"
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
        Variables to profile. By default the reflectivity, differential
        reflectivity, correlation coefficient and KDP that the sweep has (if
        it has none of them, every numeric ``(azimuth, range)`` variable).
        The mean of the radial velocity is not a wind and is left out; name
        a variable to profile it. The mean of a differential phase (named
        ``PHIDP``, ``UPHIDP``, ... in degrees) is the circular mean.
    min_rhohv : float or None, optional
        Only gates with ``rhohv > min_rhohv`` are used. Default 0.6
        (Ryzhkov et al. 2016, p. 553). ``None`` disables the test.
    min_dbz : float or None, optional
        Only gates with reflectivity ``> min_dbz`` are used. Default -10 dBZ
        (Ryzhkov et al. 2016, p. 553). ``None`` disables the test.
    rhohv, dbz : str or None, optional
        Names of the fields used for the two tests. ``"auto"`` (default) looks
        for common names (``RHOHV``, ``DBZH``, ...) and the CF standard name;
        a test is skipped if no field is found or the name is ``None``.
    min_count : int, optional
        Minimum number of valid gates on the circle for a defined value: the
        value is defined where the number of valid gates is at least
        ``min_count``. Default 30. Ryzhkov et al. (2016, p. 553) require the
        number of valid gates to *exceed* 30, i.e. at least 31, so the default
        accepts one gate fewer than the paper; use ``min_count=31`` for the
        paper's rule.
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
    elevation_correction : bool, optional
        Refer KDP and ZDR to the horizontal. KDP is divided by
        ``cos(elevation)**2``, the factor by which its azimuthal mean at the
        elevation of the sweep is lower than at the horizontal (approximate
        scaling for small oblate scatterers; not checked against a paper).
        ZDR (dB) is corrected with the inverse of the relation used by
        :func:`radarx.retrieve.zdr_offset`,
        ``ZDR0 = ZDR + 20 log10(cos(e)**2) - 20 log10(1 - 10**(ZDR/20)
        sin(e)**2)`` for the elevation ``e``: Rayleigh scattering by spheroids
        with a vertical axis and no canting. That formula is derived in
        radarx, not checked against a paper, and applied to the azimuthal
        mean of ZDR. Canted or tumbling particles (and ice with a
        horizontal-axis preference) do not follow it. Default False. Sweeps
        below 10 degrees raise a warning either way.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled
        kernel and falls back to NumPy."""


@provenance("Quasi-vertical profile of a sweep, after Ryzhkov et al. (2016)")
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
    elevation_correction=False,
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
        sweep with the highest elevation (Ryzhkov et al. 2016 [1] average at
        elevations exceeding 10°–20°, p. 552).
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

    Notes
    -----
    The filters (``rhohv > 0.6``, ``Z > -10`` dBZ) are those of the initial
    QVP methodology of Ryzhkov et al. (2016) [1] (p. 553). The count rule
    differs by one gate (see ``min_count``), and the heights of the gates are
    those of the 4/3 Earth model, not a statement of the paper. The diameter
    of the averaging circle grows with height (Ryzhkov et al. 2016, p. 553),
    whose radius is returned as ``ground_range``. Azimuthal averaging at high
    elevation was suggested for the backscatter differential phase by Trömel
    et al. (2013) [2].

    References
    ----------
    .. [1] Ryzhkov, A., P. Zhang, H. Reeves, M. Kumjian, T. Tschallener,
       S. Trömel, and C. Simmer, 2016: Quasi-vertical profiles—A new way to
       look at polarimetric radar data. *J. Atmos. Oceanic Technol.*, **33**
       (3), 551–562, https://doi.org/10.1175/JTECH-D-15-0020.1
    .. [2] Trömel, S., M. R. Kumjian, A. V. Ryzhkov, C. Simmer, and
       M. Diederich, 2013: Backscatter differential phase—Estimation and
       variability. *J. Appl. Meteor. Climatol.*, **52** (11), 2529–2548,
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
        elevation_correction=elevation_correction,
    )
    el, rng, height, ground = _geometry(ds)
    out = {}
    for v, (name, mode) in enumerate(zip(data_vars, modes)):
        out[name] = (
            "height",
            values[0][v].astype(np.float32),
            _var_attrs(ds, name, mode, filters, min_count, elevation_correction),
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


@provenance("Time-height quasi-vertical profiles, after Ryzhkov et al. (2016)")
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
    elevation_correction=False,
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
       look at polarimetric radar data. *J. Atmos. Oceanic Technol.*, **33**
       (3), 551–562, https://doi.org/10.1175/JTECH-D-15-0020.1

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
        elevation_correction=elevation_correction,
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
            _var_attrs(sweeps[0], name, mode, filters, min_count, elevation_correction),
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
            da = da.interp(time=np.asarray(t)).drop_vars("time").fillna(near)
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


@provenance(
    "Melting layer from QVPs, radarx's own, loosely after Giangrande et al. (2008)"
)
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
    (Giangrande et al. 2008 [1]). This function looks for that co-located
    signature in each profile, measures its depth, and checks the result for
    consistency along a time series. It is not an implementation of the
    algorithm of Giangrande et al. (2008) or of Ryzhkov and Krause (2022)
    [4]: the signature test is loosely modelled on the former, with
    different thresholds and window (see Notes), and every default below
    that is not attributed to a paper is radarx's own choice:

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
         dip is the most objective marker of melting; Ryzhkov and Krause
         (2022) [4] (p. 533) state that its top corresponds to the freezing
         level or the height of the wet-bulb 0 °C level and its bottom, on
         average, to a temperature of about +3 °C, citing an earlier study.
         Going up and down from
         the ρhv minimum, as Griffin et al. (2020) [3] do, the edge is the first
         height where ρhv is back at its background: within
         ``onset_fraction`` (10 %) of the dip depth from the background,
         the largest ρhv within ``depth`` on that side. Griffin et al. (2020)
         use a fixed background, ρhv ``>= 0.97`` in S-band QVPs (not checked against their
         paper); pass
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
        Default 1–6 km: the upper limit is the 6 km below which Giangrande
        et al. (2008) [1] (p. 1356, step 1) identify melting snow; the 1 km
        lower limit is radarx's choice.
    freezing_level : float or xarray.DataArray, optional
        Reference height for the search (m above sea level), e.g. the 0 °C
        level, a scalar or one value per profile (e.g. on ``time``). If
        given, the ZDR peak is searched within ``freezing_level_window`` of
        it; it overrides the reference height from ``environment``.
    freezing_level_window : tuple of float, optional
        Search range for the ZDR peak relative to the reference height,
        default −1000 to +500 m (radarx choice: melting happens below the
        0 °C level).
    rhohv_range : tuple of float, optional
        Range for the ρhv minimum near the ZDR peak, default (0.80, 0.97).
        The upper bound is the 0.97 of Giangrande et al. (2008) [1] (p. 1356),
        but the lower bound deviates from their 0.90, which they chose to
        mitigate contamination by non-meteorological scatterers (Sect. 2d,
        p. 1357). radarx accepts lower values because the azimuthal mean of
        ρhv in a QVP (over gates with ρhv > 0.6) and gates that mix
        melting and non-melting echo can fall below 0.90; this is a
        radarx choice, not tested against the paper. Pass
        ``(0.90, 0.97)`` for the paper's range.
    zdr_min : float, optional
        Minimum ZDR peak, default 0.5 dB (radarx choice; Giangrande et al.
        2008 [1] require a ZDR maximum between 0.8 and 2.5 dB, and there is no
        upper limit here).
    dbz_min : float, optional
        Minimum Z maximum near the ZDR peak, default 20 dBZ (radarx choice;
        Giangrande et al. 2008 [1] require a Z maximum between 30 and 47 dBZ,
        and there is no upper limit here).
    window : float, optional
        Co-location distance of the ρhv minimum and Z maximum from the ZDR
        peak, default 500 m, on both sides of the peak (the 500 m of
        Giangrande et al. 2008 [1] is a window above the gate of the ρhv
        signature).
    depth : float, optional
        How far above and below the peak the background is sought, default
        1000 m (radarx choice).
    rhohv_onset : float, optional
        A fixed background ρhv for ``boundaries="onset"``, e.g. 0.97 as in
        Griffin et al. (2020). Default None: only ``onset_fraction``.
    onset_fraction : float, optional
        With ``boundaries="onset"``, the edge is where ρhv is within this
        fraction of the dip depth from the background, default 0.1 (radarx
        choice).
    edge_fraction : float, optional
        Fraction of the prominence the anomaly must fall by at the edges with
        ``boundaries="half_prominence"``, default 0.5 (radarx choice).
    median_window : int, optional
        Profiles in the running median, default 5 (radarx choice). 1 disables
        the check.
    max_jump : float, optional
        Largest accepted distance from the running median, default 500 m
        (radarx choice).
    max_fill : int, optional
        Longest gap (in profiles) filled by interpolation, default 3 (radarx
        choice); 0 disables filling.
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
    *Giangrande et al. (2008) [1]* (Sect. 2b, p. 1356, steps 1 to 3 of their
    algorithm) flag individual radar gates with 0.90 < ρhv < 0.97 (below
    6 km), search for the maxima of Z and ZDR in a 500 m window above each
    such gate and call the gate a melting-layer point if the maximum of Z lies
    between 30 and 47 dBZ and the maximum of ZDR between 0.8 and 2.5 dB. They
    then take the heights below which 80 % (top) and 20 % (bottom) of the
    points of an azimuth sector lie (p. 1357); this is not done here. The
    differences of this function are:

    - the search is anchored on the largest ZDR value (not on the ρhv gates)
      and looks for the ρhv minimum and the Z maximum within ``window`` metres
      on both sides of it;
    - ``rhohv_range`` is (0.80, 0.97), not (0.90, 0.97) (see that parameter);
    - ``zdr_min=0.5`` dB and ``dbz_min=20`` dBZ replace 0.8 dB and 30 dBZ,
      and there is no upper limit of 2.5 dB and 47 dBZ. The lower defaults
      are radarx's choice, made because azimuthal averaging in a QVP smooths
      and weakens the extremes (Z is averaged over the whole circle,
      including weaker echo); they were not derived from the paper. Tune
      them for other radars and elevations.

    *Griffin et al. (2020) [3]* define the top and bottom of the melting
    layer in S-band QVPs by searching upward and downward from the ρhv
    minimum for the first ρhv ``>= 0.97``; they report that the
    reflectivity-curvature method of Fabry and Zawadzki (1995) [5] gave tops
    about 200 m higher and bottoms within about 50 m (not checked against the
    paper). Where the
    background is close to 1, as at S band, a fixed 0.97 is reached well
    inside the dip; the default relative test finds where ρhv first departs
    from its background, wherever that background lies (it is lower at C and
    X band and in noisy data). On the KGWX QVPs of 30–31 March 2022 the
    default top is about 190 m above the 0.97 top; this is a radarx result and
    is not the same comparison as that of Griffin et al.

    *Ryzhkov and Krause (2022) [4]* state (p. 533, citing an earlier study for
    the second part) that the top of the ρhv (CC) dip of the melting layer
    corresponds to the freezing level or the height of the wet-bulb 0 °C
    level, and that its bottom, the end of snow melting, occurs on average at
    about +3 °C; they add that the dip can be detected at temperatures as
    high as 7 °C and that other authors report different bottoms. The
    ``"onset"`` top is therefore expected near the freezing level or the
    wet-bulb 0 °C height of the environment; the returned offsets quantify
    the difference. Their statement that the top and bottom of the dip can
    be estimated from QVPs with an accuracy of about 0.1 km (p. 540) is
    attributed by them to earlier QVP studies (among them Griffin et al.
    2020 [3]); it is not a result of their paper, and it is not a validation
    of this function. Their own algorithm (radial profiles of ρhv at
    elevations below about 5–6 degrees matched to model profiles, Sect. 3;
    average bias against their RD-QVP products of −128 m for the top and
    −107 m for the bottom in four events, p. 541) is not QVP-based and is not implemented here.

    References
    ----------
    .. [1] Giangrande, S. E., J. M. Krause, and A. V. Ryzhkov, 2008:
       Automatic designation of the melting layer with a polarimetric
       prototype of the WSR-88D radar. *J. Appl. Meteor. Climatol.*, **47**
       (5), 1354–1364, https://doi.org/10.1175/2007JAMC1634.1
    .. [2] Ryzhkov, A., P. Zhang, H. Reeves, M. Kumjian, T. Tschallener,
       S. Trömel, and C. Simmer, 2016: Quasi-vertical profiles—A new way to
       look at polarimetric radar data. *J. Atmos. Oceanic Technol.*, **33**
       (3), 551–562, https://doi.org/10.1175/JTECH-D-15-0020.1
    .. [3] Griffin, E. M., T. J. Schuur, and A. V. Ryzhkov, 2020: A
       polarimetric radar analysis of ice microphysical processes in melting
       layers of winter storms using S-band quasi-vertical profiles. *J.
       Appl. Meteor. Climatol.*, **59** (4), 751–767,
       https://doi.org/10.1175/JAMC-D-19-0128.1
    .. [4] Ryzhkov, A., and J. Krause, 2022: New polarimetric radar algorithm
       for melting-layer detection and determination of its height. *J.
       Atmos. Oceanic Technol.*, **39** (5), 529–543,
       https://doi.org/10.1175/JTECH-D-21-0130.1
    .. [5] Fabry, F., and I. Zawadzki, 1995: Long-term radar observations of
       the melting layer of precipitation and their interpretation. *J.
       Atmos. Sci.*, **52** (7), 838–851,
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


# ---------------------------------------------------------------------------
# profiles from several sweeps: range-defined, columnar and RHI profiles
# ---------------------------------------------------------------------------

_KERNELS = {"top_hat": 0, "cressman": 1}
_VERTICAL_MEAN = ("linear", "db")


def _blend_numpy(height, weight, values, grid, radius, kernel, modes, min_samples):
    """NumPy implementation of the compiled ``height_blend`` (same results)."""
    nv, m = values.shape[0], grid.size
    out = np.full((nv, m), np.nan)
    cnt = np.zeros((nv, m), dtype=np.int32)
    r2 = radius * radius
    lo = np.searchsorted(height, grid - radius, side="left")
    hi = np.searchsorted(height, grid + radius, side="left")
    for k in range(m):
        w = weight[lo[k] : hi[k]]
        if kernel == _KERNELS["cressman"]:
            d2 = (height[lo[k] : hi[k]] - grid[k]) ** 2
            w = np.where(d2 < r2, w * (r2 - d2) / (r2 + d2), 0.0)
        for v in range(nv):
            x = values[v, lo[k] : hi[k]]
            ok = ~np.isnan(x) & (w > 0)
            n = int(ok.sum())
            cnt[v, k] = n
            if n < 1 or n < min_samples:
                continue
            xx = x[ok]
            if modes[v] == _MODES["mean_db"]:
                xx = np.exp(xx * _LN10_OVER_10)
            mean = np.sum(w[ok] * xx) / np.sum(w[ok])
            out[v, k] = 10.0 * np.log10(mean) if modes[v] == _MODES["mean_db"] else mean
    return out, cnt


def _blend(
    height, weight, values, grid, radius, kernel, modes, min_samples, n_threads, engine
):
    """
    Weighted means of samples on a height grid.

    A sample within ``radius`` of a grid height contributes with its
    ``weight`` times the kernel (``"top_hat"``: 1 on ``[-radius, radius)``;
    ``"cressman"``: ``(R**2 - d**2) / (R**2 + d**2)``, Cressman 1959).
    """
    use_compiled = _use_compiled(engine)
    if not radius > 0:
        raise ValueError("the vertical radius must be positive")
    order = np.argsort(height, kind="stable")
    height = np.ascontiguousarray(height[order], dtype=np.float64)
    weight = np.ascontiguousarray(weight[order], dtype=np.float64)
    values = np.ascontiguousarray(values[:, order], dtype=np.float64)
    grid = np.ascontiguousarray(grid, dtype=np.float64)
    modes = [int(v) for v in modes]
    args = (height, weight, values, grid, float(radius), _KERNELS[kernel])
    if use_compiled:
        return _qvp.height_blend(
            *args, modes, int(min_samples), n_threads=int(n_threads or 0)
        )
    return _blend_numpy(*args, modes, int(min_samples))


def _blend_modes(ds, data_vars, linear, vertical_mean):
    """Reduction mode of the averaging in height for every variable."""
    if vertical_mean not in _VERTICAL_MEAN:
        raise ValueError(
            f"vertical_mean must be 'linear' or 'db', not {vertical_mean!r}"
        )
    if vertical_mean == "db":
        return [_MODES["mean"]] * len(data_vars)
    return _modes(ds, data_vars, "mean", linear)


def _usable_sweeps(sweeps, data_vars, rhohv, dbz, min_rhohv, min_dbz):
    """The sweeps that have the fields of the first sweep that has any."""
    ref = None
    for ds in sweeps:
        try:
            ref = _fields_of(ds, data_vars, rhohv, dbz, min_rhohv, min_dbz)
        except ValueError:
            continue
        break
    if ref is None:
        raise ValueError("No sweep has the variables to profile.")
    names = ref[0] + ref[1]
    return [ds for ds in sweeps if all(n in ds.data_vars for n in names)], ref[0]


def _as_volumes(volumes):
    """``(list of volumes, whether a sequence was given)``."""
    if isinstance(volumes, (xr.Dataset, xr.DataTree)):
        return [volumes], False
    volumes = list(volumes)
    if not volumes:
        raise ValueError("No volumes given.")
    return volumes, True


def _volume_time(sweeps):
    """Mean time of the rays of the sweeps."""
    t = np.array([_sweep_time(ds) for ds in sweeps], dtype="datetime64[ns]")
    t = t[~np.isnat(t)]
    if t.size == 0:
        return np.datetime64("NaT", "ns")
    return t.min() + (t - t.min()).mean()


def _height_grid(heights, lowest, highest, dz):
    """The common heights: given, or a regular grid over the samples."""
    if heights is not None:
        grid = np.asarray(heights, dtype=np.float64)
        if grid.ndim != 1 or (grid.size > 1 and np.any(np.diff(grid) <= 0)):
            raise ValueError("heights must be 1-D and strictly ascending.")
        return grid
    if not dz > 0:
        raise ValueError("dz must be positive.")
    if not (np.isfinite(lowest) and np.isfinite(highest)):
        return np.zeros(0)
    return np.arange(math.ceil(lowest / dz) * dz, highest + 0.5 * dz, dz)


def _blend_attrs(name, ds, mode, filters, min_count, text, correct=False):
    attrs = _var_attrs(ds, name, mode, filters, min_count, correct)
    attrs["cell_methods"] = text
    return attrs


def _blended_dataset(
    names,
    results,
    grid,
    times,
    series,
    ds,
    modes,
    filters,
    min_count,
    text,
    counts,
    extra,
):
    """Dataset of ``(time,) height`` profiles from the blended volumes."""
    dims = ("time", "height") if series else ("height",)
    out = {}
    for v, (name, mode) in enumerate(zip(names, modes)):
        value = np.stack([r[0][v] for r in results]).astype(np.float32)
        out[name] = (
            dims,
            value if series else value[0],
            _blend_attrs(name, ds, mode, filters, min_count, text),
        )
        if counts:
            n = np.stack([r[1][v] for r in results])
            out[f"{name}_count"] = (
                dims,
                n if series else n[0],
                {"long_name": f"number of samples of {name}", "units": "1"},
            )
    coords = {"height": _height_coords(grid, grid, grid)["height"]}
    if series:
        coords["time"] = (
            "time",
            np.array(times, dtype="datetime64[ns]"),
            {"standard_name": "time"},
        )
    else:
        coords["time"] = times[0]
    coords.update(extra)
    return xr.Dataset(out, coords=coords).assign_coords(_site_coords(ds))


def _db_text(vertical_mean):
    return "linear units for dB quantities" if vertical_mean == "linear" else "dB"


def _profile_kwargs(
    min_rhohv, min_dbz, rhohv, dbz, min_count, min_fraction, reduction, linear
):
    return dict(
        min_rhohv=min_rhohv,
        min_dbz=min_dbz,
        rhohv=rhohv,
        dbz=dbz,
        min_count=min_count,
        min_fraction=min_fraction,
        reduction=reduction,
        linear=linear,
    )


def _elevation_limits(sweeps, minimum, maximum):
    """The sweeps whose elevation is within the limits."""
    kept = []
    for ds in sweeps:
        el = float(np.nanmedian(ds["elevation"].values))
        if (minimum is None or el >= minimum) and (maximum is None or el <= maximum):
            kept.append(ds)
    return kept


@provenance(
    "Range-defined quasi-vertical profile, after Tobin and Kumjian (2017) "
    "as described by Murphy et al. (2020); weights are radarx's own"
)
def rd_qvp(
    volumes,
    data_vars=None,
    *,
    rmax=50000.0,
    power=2.0,
    heights=None,
    dz=100.0,
    radius=None,
    min_elevation=None,
    max_elevation=None,
    min_rhohv=0.6,
    min_dbz=-10.0,
    rhohv="auto",
    dbz="auto",
    min_count=30,
    min_fraction=0.0,
    reduction="mean",
    linear=None,
    vertical_mean="linear",
    min_samples=1,
    counts=False,
    elevation_correction=False,
    n_threads=None,
    engine="auto",
):
    """
    Range-defined quasi-vertical profile from all sweeps of a volume.

    A quasi-vertical profile (:func:`qvp`) needs a high elevation. The
    range-defined QVP (RD-QVP) of Tobin and Kumjian (2017) [1]_ builds a
    profile that reaches the ground from the low and intermediate sweeps as
    well: the azimuthal means of every sweep, taken over the gates within
    ``rmax`` of the radar, are combined with weights that decrease with the
    distance from the radar. It is used for the structure near the surface
    and close to the radar, and as every QVP it assumes that the
    precipitation is the same all around the radar (Murphy et al. 2020 [2]_,
    who describe it).

    The paper of Tobin and Kumjian (2017) could not be read (it is not open
    access); this function follows the description of Murphy et al. (2020):
    QVPs of several elevations, each from data within 50 km of the radar,
    combined by inverse distance weighting. What is radarx's own, and **not
    checked against the paper**: the form of the weight, ``ground_range **
    -power`` with ``power = 2``; the vertical radius over which the samples of
    the sweeps are combined, with the weights of Cressman (1959) [3]_; the
    height grid; and the use of the azimuthal reduction of :func:`qvp`
    (thresholds, minimum count, averaging of dB quantities in linear units).

    Parameters
    ----------
    volumes : xarray.DataTree, xarray.Dataset or sequence of them
        A radar volume with ``sweep_*`` groups (xradar layout) or a sweep, or
        a sequence of volumes in time order. Sweeps that lack the variables
        (for example Doppler-only cuts) are left out.
    data_vars : str or list of str, optional
        Variables to profile, as in :func:`qvp`.
    rmax : float, optional
        Only gates within this ground distance (m) of the radar are used.
        Default 50000 (Murphy et al. 2020 [2]_).
    power : float, optional
        Exponent of the weight ``ground_range ** -power`` of a sample (the
        ground range is at least 1 m). 0 gives equal weights. Default 2
        (radarx choice).
    heights : array-like, optional
        Heights (m above sea level) of the profile. By default a regular grid
        of ``dz`` over the heights of the gates.
    dz : float, optional
        Spacing (m) of the default height grid. Default 100.
    radius : float, optional
        Samples within this vertical distance (m) of a grid height are
        combined, with the weights of Cressman (1959). Default ``dz``.
    min_elevation, max_elevation : float, optional
        Use only the sweeps within these elevations (degrees).
    min_rhohv, min_dbz, rhohv, dbz, min_count, min_fraction, reduction, linear
        As in :func:`qvp`, for the azimuthal reduction of every sweep.
    vertical_mean : {"linear", "db"}, optional
        Whether dB quantities are averaged in height in linear units
        (default) or in dB.
    min_samples : int, optional
        Minimum number of samples (gates of the sweeps) for a defined value.
        Default 1.
    counts : bool, optional
        Also return ``<name>_count``, the number of samples used.
    elevation_correction : bool, optional
        Refer KDP and ZDR of every sweep to the horizontal as in :func:`qvp`.
        There is no warning for low sweeps here. Default False.
    n_threads : int, optional
        Threads for the compiled kernels. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use.

    Returns
    -------
    xarray.Dataset
        Profiles on ``height`` (m above sea level) for a single volume, on
        ``(time, height)`` for a sequence, with the radar site.

    See Also
    --------
    qvp, cvp, melting_layer

    Notes
    -----
    Not implemented: any weighting of the gates other than by the ground
    range of their sweep. The hard cut at ``rmax`` is the one described by
    Murphy et al. (2020); weights beyond ``rmax`` (as in the Py-ART
    implementation) are not offered.

    References
    ----------
    .. [1] Tobin, D. M., and M. R. Kumjian, 2017: Polarimetric radar and
       surface-based precipitation-type observations of ice pellet to
       freezing rain transitions. *Wea. Forecasting*, **32** (6), 2065-2082,
       https://doi.org/10.1175/WAF-D-17-0054.1
    .. [2] Murphy, A. M., A. Ryzhkov, and P. Zhang, 2020: Columnar vertical
       profile (CVP) methodology for validating polarimetric radar retrievals
       in ice using in situ aircraft measurements. *J. Atmos. Oceanic
       Technol.*, **37** (9), 1623-1642,
       https://doi.org/10.1175/JTECH-D-20-0011.1
    .. [3] Cressman, G. P., 1959: An operational objective analysis system.
       *Mon. Wea. Rev.*, **87** (10), 367-374,
       https://doi.org/10.1175/1520-0493(1959)087<0367:AOOAS>2.0.CO;2

    Examples
    --------
    >>> profile = radarx.retrieve.rd_qvp(dtree, ["DBZH", "ZDR", "RHOHV"])  # doctest: +SKIP
    >>> profile = dtree.radarx.rd_qvp(rmax=30e3)  # doctest: +SKIP
    """
    volumes, series = _as_volumes(volumes)
    if not rmax > 0:
        raise ValueError("rmax must be positive.")
    kw = _profile_kwargs(
        min_rhohv, min_dbz, rhohv, dbz, min_count, min_fraction, reduction, linear
    )
    samples = []
    for volume in volumes:
        sweeps, names = _usable_sweeps(
            _elevation_limits(_volume_sweeps(volume), min_elevation, max_elevation),
            data_vars,
            rhohv,
            dbz,
            min_rhohv,
            min_dbz,
        )
        cut, height, weight = [], [], []
        for ds in sweeps:
            _, _, h, ground = _geometry(ds)
            n = int(np.searchsorted(ground, rmax, side="right"))
            if n:
                cut.append(ds.isel(range=slice(0, n)))
                height.append(h[:n])
                weight.append(np.maximum(ground[:n], 1.0) ** -float(power))
        if not cut:
            raise ValueError(f"No sweep has gates within rmax={rmax:g} m.")
        names, modes, values, cnts, filters = _profiles(
            cut,
            names,
            n_threads=n_threads,
            engine=engine,
            elevation_correction=elevation_correction,
            warn_low=False,
            **kw,
        )
        samples.append(
            (
                np.concatenate(height),
                np.concatenate(weight),
                np.concatenate([np.asarray(v, np.float64) for v in values], axis=1),
                _volume_time(cut),
                cut[0],
            )
        )
    lowest = min(s[0].min() for s in samples)
    highest = max(s[0].max() for s in samples)
    grid = _height_grid(heights, lowest, highest, dz)
    blend_modes = _blend_modes(samples[0][4], names, linear, vertical_mean)
    vertical_radius = float(dz if radius is None else radius)
    results = [
        _blend(
            h,
            w,
            v,
            grid,
            vertical_radius,
            "cressman",
            blend_modes,
            min_samples,
            n_threads,
            engine,
        )
        for h, w, v, _, _ in samples
    ]
    text = (
        f"ground range <= {rmax:g} m: azimuth: mean per sweep; height: weighted "
        f"mean (weight ground_range**-{power:g}, Cressman radius "
        f"{vertical_radius:g} m) in {_db_text(vertical_mean)}"
    )
    out = _blended_dataset(
        names,
        results,
        grid,
        [s[3] for s in samples],
        series,
        samples[0][4],
        blend_modes,
        filters,
        min_count,
        text,
        counts,
        {},
    )
    out.attrs = {
        "title": "range-defined quasi-vertical profile",
        "references": "Tobin and Kumjian (2017), doi:10.1175/WAF-D-17-0054.1; "
        "Murphy et al. (2020), doi:10.1175/JTECH-D-20-0011.1",
    }
    return out


def _height_at_ground(elevation, ground, site_altitude):
    """Beam height (m above sea level) at a ground distance (4/3 Earth model)."""
    cos = max(math.cos(math.radians(elevation)), 0.05)
    rng = np.linspace(0.0, 1.5 * ground / cos + 1000.0, 2001)
    height, dist = _beam_height(rng, elevation, site_altitude)
    return float(np.interp(ground, dist, height))


@provenance("Columnar vertical profile, after Murphy et al. (2020)")
def cvp(
    volumes,
    data_vars=None,
    *,
    azimuth,
    ground_range,
    half_range=10000.0,
    half_azimuth=10.0,
    heights=None,
    dz=50.0,
    radius=100.0,
    min_rhohv=0.6,
    min_dbz=-10.0,
    rhohv="auto",
    dbz="auto",
    min_count=1,
    min_fraction=0.0,
    reduction="mean",
    linear=None,
    vertical_mean="db",
    min_samples=1,
    counts=False,
    elevation_correction=False,
    n_threads=None,
    engine="auto",
):
    """
    Columnar vertical profile above a point of a volume.

    The columnar vertical profile (CVP) of Murphy et al. (2020) [1]_ is a
    vertical profile of a sector of the radar volume. A centre is chosen in
    azimuth and ground range, and a sector around it
    (``half_range`` up- and down-radial, ``half_azimuth`` clockwise and
    counter-clockwise). For every sweep, the data of the sector are averaged
    over the azimuths for each range gate, and the mean of each gate is
    projected horizontally to the centre, which gives samples irregularly
    spaced in height. A sample of a sweep is only used between the heights, at
    the centre, of the two "intermediate elevation angles" half-way to the
    sweeps above and below. The samples of all sweeps are then averaged on a
    regular height grid with the weights of Cressman (1959) [2]_.

    The sector size (20 km in range and 20 degrees in azimuth), the Cressman
    radius of 100 m and the 50 m spacing are those of Murphy et al. (2020),
    who use them for stratiform precipitation. Defaults here that are not
    stated in the paper are radarx's: the thresholds of :func:`qvp`, the
    minimum of 1 gate per azimuthal mean (``min_count``; the 30 gates of
    :func:`qvp` cannot be reached in a sector), and the averaging of dB
    quantities in the azimuthal mean in linear units. The paper states that
    the Cressman averaging is done on Z and ZDR in logarithmic units
    (``vertical_mean="db"``, the default here); how the azimuthal averaging
    treats them is not stated.

    Parameters
    ----------
    volumes : xarray.DataTree, xarray.Dataset or sequence of them
        A radar volume with ``sweep_*`` groups (xradar layout), or a sequence
        of volumes in time order. Sweeps that lack the variables are left
        out. Use a different centre for each volume with
        :func:`radarx.retrieve.cvp` called per volume.
    data_vars : str or list of str, optional
        Variables to profile, as in :func:`qvp`.
    azimuth : float
        Azimuth (degrees clockwise from north) of the centre of the sector.
    ground_range : float
        Ground distance (m) of the centre from the radar. Murphy et al.
        (2020) use centres within 100 km of the radar: beyond that, the
        beams are wider than the spacing of the elevations and the profile
        has gaps.
    half_range : float, optional
        Half the radial extent (m) of the sector. Default 10000.
    half_azimuth : float, optional
        Half the azimuthal extent (degrees) of the sector. Default 10.
    heights : array-like, optional
        Heights (m above sea level) of the profile. By default a regular grid
        of ``dz`` over the heights of the samples.
    dz : float, optional
        Spacing (m) of the default height grid. Default 50.
    radius : float, optional
        Cressman radius (m): samples within this vertical distance of a
        grid height are averaged. Default 100.
    min_rhohv, min_dbz, rhohv, dbz, min_fraction, reduction, linear
        As in :func:`qvp`, for the azimuthal reduction.
    min_count : int, optional
        Minimum number of gates of the sector in the azimuthal mean of a
        range gate. Default 1.
    vertical_mean : {"db", "linear"}, optional
        Whether dB quantities are averaged in height in dB (default, as
        Murphy et al. 2020) or in linear units.
    min_samples : int, optional
        Minimum number of samples for a defined value. Default 1.
    counts : bool, optional
        Also return ``<name>_count``, the number of samples used.
    elevation_correction : bool, optional
        Refer KDP and ZDR of every sweep to the horizontal as in :func:`qvp`.
        Default False.
    n_threads : int, optional
        Threads for the compiled kernels. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use.

    Returns
    -------
    xarray.Dataset
        Profiles on ``height`` (m above sea level) for a single volume, on
        ``(time, height)`` for a sequence, with the centre as the scalar
        coordinates ``azimuth`` and ``ground_range`` and the radar site.

    See Also
    --------
    qvp, rd_qvp, melting_layer

    Notes
    -----
    Not implemented: the moving CVP that follows an aircraft or a storm (give
    a different centre per volume by calling the function for each volume and
    concatenating the results), and the KDP processing of Griffin et al.
    (2018) that the paper applies (use :func:`radarx.retrieve.estimate_kdp`).
    The 1.3 % displacement of the mean location of the projected points
    reported in the paper was not checked.

    References
    ----------
    .. [1] Murphy, A. M., A. Ryzhkov, and P. Zhang, 2020: Columnar vertical
       profile (CVP) methodology for validating polarimetric radar retrievals
       in ice using in situ aircraft measurements. *J. Atmos. Oceanic
       Technol.*, **37** (9), 1623-1642,
       https://doi.org/10.1175/JTECH-D-20-0011.1
    .. [2] Cressman, G. P., 1959: An operational objective analysis system.
       *Mon. Wea. Rev.*, **87** (10), 367-374,
       https://doi.org/10.1175/1520-0493(1959)087<0367:AOOAS>2.0.CO;2

    Examples
    --------
    >>> profile = radarx.retrieve.cvp(dtree, azimuth=225.0, ground_range=30e3)  # doctest: +SKIP
    >>> profile = dtree.radarx.cvp(azimuth=225.0, ground_range=30e3)  # doctest: +SKIP
    """
    volumes, series = _as_volumes(volumes)
    if not (half_range > 0 and half_azimuth > 0 and ground_range > 0):
        raise ValueError("half_range, half_azimuth and ground_range must be positive.")
    kw = _profile_kwargs(
        min_rhohv, min_dbz, rhohv, dbz, min_count, min_fraction, reduction, linear
    )
    samples = []
    for volume in volumes:
        sweeps, names = _usable_sweeps(
            _volume_sweeps(volume), data_vars, rhohv, dbz, min_rhohv, min_dbz
        )
        elevation = [_geometry(ds)[0] for ds in sweeps]
        levels = np.unique(np.round(elevation, 2))
        site = _site_altitude(sweeps[0])
        cut, height = [], []
        for ds, el in zip(sweeps, elevation):
            if "azimuth" not in ds:
                raise ValueError("every sweep needs an 'azimuth' coordinate.")
            _, _, h, ground = _geometry(ds)
            raydim = ds["azimuth"].dims[0]
            dist = np.abs((ds["azimuth"].values - azimuth + 180.0) % 360.0 - 180.0)
            rays = np.flatnonzero(dist <= half_azimuth)
            g0 = int(np.searchsorted(ground, ground_range - half_range, side="left"))
            g1 = int(np.searchsorted(ground, ground_range + half_range, side="right"))
            if rays.size == 0 or g1 <= g0:
                continue
            cut.append(ds.isel({raydim: rays, "range": slice(g0, g1)}))
            height.append(
                (
                    h[g0:g1],
                    *_intermediate_band(levels, el, ground_range, site),
                )
            )
        if not cut:
            raise ValueError("No sweep has gates in the sector.")
        names, modes, values, cnts, filters = _profiles(
            cut,
            names,
            n_threads=n_threads,
            engine=engine,
            elevation_correction=elevation_correction,
            warn_low=False,
            **kw,
        )
        h_all, v_all = [], []
        for (h, low, high), val in zip(height, values):
            val = np.array(val, dtype=np.float64)
            val[:, (h < low) | (h > high)] = np.nan
            h_all.append(h)
            v_all.append(val)
        h_all = np.concatenate(h_all)
        samples.append(
            (
                h_all,
                np.ones(h_all.size),
                np.concatenate(v_all, axis=1),
                _volume_time(cut),
                cut[0],
            )
        )
    lowest = min(s[0].min() for s in samples)
    highest = max(s[0].max() for s in samples)
    grid = _height_grid(heights, lowest, highest, dz)
    blend_modes = _blend_modes(samples[0][4], names, linear, vertical_mean)
    results = [
        _blend(
            h,
            w,
            v,
            grid,
            radius,
            "cressman",
            blend_modes,
            min_samples,
            n_threads,
            engine,
        )
        for h, w, v, _, _ in samples
    ]
    text = (
        f"sector {ground_range:g} +- {half_range:g} m, azimuth {azimuth:g} +- "
        f"{half_azimuth:g} degrees: azimuth: mean per sweep; height: Cressman "
        f"mean (radius {radius:g} m) in {_db_text(vertical_mean)}"
    )
    extra = {
        "azimuth": (
            (),
            float(azimuth),
            {"long_name": "azimuth of the CVP centre", "units": "degrees"},
        ),
        "ground_range": (
            (),
            float(ground_range),
            {"long_name": "ground range of the CVP centre", "units": "m"},
        ),
    }
    out = _blended_dataset(
        names,
        results,
        grid,
        [s[3] for s in samples],
        series,
        samples[0][4],
        blend_modes,
        filters,
        min_count,
        text,
        counts,
        extra,
    )
    out.attrs = {
        "title": "columnar vertical profile",
        "references": "Murphy et al. (2020), doi:10.1175/JTECH-D-20-0011.1",
    }
    return out


def _intermediate_band(levels, elevation, ground, site_altitude):
    """
    Lowest and highest height at which a sweep contributes to a CVP.

    Heights, at ground distance ``ground``, of the beams at the "intermediate
    elevation angles" half-way to the next lower and higher elevation of the
    volume (Murphy et al. 2020, Sect. 3a); unbounded below the lowest sweep
    and above the highest.
    """
    k = int(np.argmin(np.abs(levels - elevation)))
    low, high = -np.inf, np.inf
    if k > 0:
        low = _height_at_ground(
            0.5 * (levels[k - 1] + levels[k]), ground, site_altitude
        )
    if k < levels.size - 1:
        high = _height_at_ground(
            0.5 * (levels[k] + levels[k + 1]), ground, site_altitude
        )
    return low, high


@provenance(
    "Quasi-vertical profile of an RHI (RHI-QVP), after Allabakash et al. (2019)"
)
def rhi_qvp(
    rhi,
    data_vars=None,
    *,
    sweep=0,
    rmin=6000.0,
    rmax=20000.0,
    max_elevation=45.0,
    min_rhohv=0.6,
    min_dbz=0.0,
    rhohv="auto",
    dbz="auto",
    dz=100.0,
    heights=None,
    linear=None,
    min_samples=1,
    counts=False,
    n_threads=None,
    engine="auto",
):
    """
    Vertical profile of a range-height indicator (RHI) scan.

    The range of every gate of an RHI is converted to height, and the gates
    of all elevations within a height bin are averaged, which gives a
    vertical profile at the azimuth of the RHI (the range-height-indicator
    quasi-vertical profile, R-QVP, of Allabakash et al. 2019 [1]_, [2]_). A
    sequence of RHIs gives a time-height profile. The profile has the
    variables and the ``height`` dimension of :func:`qvp` and can be given to
    :func:`melting_layer`.

    The settings are those of Allabakash et al. (2019) [1]_, who use X-band
    RHIs: gates with ``rhohv > 0.6`` and ``Z >= 0 dBZ``, between 6 and 20 km
    range and below 45 degrees elevation (to reduce ground clutter and beam
    broadening), averaged in 100 m bins. They do not say in which scale dB
    quantities are averaged: radarx averages them in linear units, as
    :func:`qvp` does. The gate heights use the 4/3 Earth model of
    :func:`qvp`.

    Parameters
    ----------
    rhi : xarray.Dataset, xarray.DataTree or sequence of them
        An RHI sweep on ``(ray, range)`` with an ``elevation`` for every ray
        (xradar layout), or a volume of RHI sweeps (then ``sweep`` is used),
        or a sequence in time order.
    data_vars : str or list of str, optional
        Variables to profile, as in :func:`qvp`.
    sweep : int or str, optional
        The sweep of a volume to use. Default 0.
    rmin, rmax : float, optional
        Range limits (m) of the gates used. Default 6000 and 20000.
    max_elevation : float, optional
        Only rays below this elevation (degrees) are used. Default 45.
    min_rhohv, min_dbz : float or None, optional
        Gates with ``rhohv > min_rhohv`` and ``Z >= min_dbz`` are used.
        Default 0.6 and 0 dBZ. ``None`` disables a test.
    rhohv, dbz : str or None, optional
        Names of the fields of the two tests, as in :func:`qvp`.
    dz : float, optional
        Height of the bins (m). Default 100.
    heights : array-like, optional
        Centres of the bins (m above sea level). By default a regular grid of
        ``dz`` over the heights of the gates.
    linear : str or list of str, optional
        Variables averaged in linear units, as in :func:`qvp`.
    min_samples : int, optional
        Minimum number of gates in a bin. Default 1.
    counts : bool, optional
        Also return ``<name>_count``, the number of gates in each bin.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use.

    Returns
    -------
    xarray.Dataset
        Profiles on ``height`` (m above sea level), or ``(time, height)`` for
        a sequence, with the ``azimuth`` of the RHI and the radar site.

    See Also
    --------
    qvp, melting_layer

    Notes
    -----
    The melting layer of Allabakash et al. (2019) is found from the
    gradients of Z, ZDR and rhohv and of their products with 1 - rhohv in the
    R-QVP; that method is not implemented. :func:`melting_layer` applied to
    the R-QVP is a different method (radarx's, see there).

    References
    ----------
    .. [1] Allabakash, S., S. Lim, and B.-J. Jang, 2019: Melting layer
       detection and characterization based on range height
       indicator-quasi vertical profiles. *Remote Sens.*, **11** (23), 2848,
       https://doi.org/10.3390/rs11232848
    .. [2] Allabakash, S., S. Lim, V. Chandrasekar, K.-H. Min, J. Choi, and
       B.-J. Jang, 2019: X-band dual-polarization radar observations of snow
       growth processes of a severe winter storm: Case of 12 December 2013 in
       South Korea. *J. Atmos. Oceanic Technol.*, **36** (7), 1217-1235,
       https://doi.org/10.1175/JTECH-D-18-0076.1

    Examples
    --------
    >>> profile = radarx.retrieve.rhi_qvp(rhi, ["DBZH", "ZDR", "RHOHV"])  # doctest: +SKIP
    >>> ml = radarx.retrieve.melting_layer(profile)  # doctest: +SKIP
    """
    if isinstance(rhi, (xr.Dataset, xr.DataTree)):
        scans, series = [rhi], False
    else:
        scans, series = list(rhi), True
        if not scans:
            raise ValueError("No scans given.")
    sweeps = [
        _select_sweep(s, sweep=sweep) if isinstance(s, xr.DataTree) else s
        for s in scans
    ]
    samples = [
        _rhi_samples(
            ds, data_vars, rmin, rmax, max_elevation, min_rhohv, min_dbz, rhohv, dbz
        )
        for ds in sweeps
    ]
    names = samples[0][0]
    heights_in = [s[1] for s in samples if s[1].size]
    lowest = min((h.min() for h in heights_in), default=np.inf)
    highest = max((h.max() for h in heights_in), default=-np.inf)
    grid = _height_grid(heights, lowest, highest, dz)
    modes = _modes(sweeps[0], names, "mean", linear)
    results = [
        _blend(
            h,
            np.ones(h.size),
            values,
            grid,
            0.5 * dz,
            "top_hat",
            modes,
            min_samples,
            n_threads,
            engine,
        )
        for _, h, values in samples
    ]
    first = sweeps[0]
    extra = {}
    if "azimuth" in first:
        extra["azimuth"] = (
            (),
            float(np.nanmedian(first["azimuth"].values)),
            {"long_name": "median azimuth of the RHI", "units": "degrees"},
        )
    out = _blended_dataset(
        names,
        results,
        grid,
        [_sweep_time(ds) for ds in sweeps],
        series,
        first,
        modes,
        {},
        1,
        f"height bins of {dz:g} m: mean of the gates of all elevations",
        counts,
        extra,
    )
    for name in names:
        out[name].attrs["qvp_filter"] = (
            f"rhohv > {min_rhohv} and Z >= {min_dbz} dBZ, range {rmin:g} to "
            f"{rmax:g} m, elevation < {max_elevation:g} degrees"
        )
        out[name].attrs.pop("qvp_min_count", None)
    out.attrs = {
        "title": "RHI quasi-vertical profile",
        "references": "Allabakash et al. (2019), doi:10.3390/rs11232848",
    }
    return out


def _rhi_samples(
    ds, data_vars, rmin, rmax, max_elevation, min_rhohv, min_dbz, rhohv, dbz
):
    """Names, heights and values of the gates of an RHI that pass the tests."""
    names = _fields_of(ds, data_vars, rhohv, dbz, min_rhohv, min_dbz)[0]
    ray = _ray_dim(ds[names[0]])
    arrays, _ = _sweep_arrays(ds, names, [])
    el = np.asarray(ds["elevation"].values, dtype=np.float64)
    if el.shape != (ds.sizes[ray],):
        raise ValueError("the elevation must have one value for every ray.")
    rng = np.asarray(ds["range"].values, dtype=np.float64)
    ok = (
        ((rng >= rmin) & (rng <= rmax))[None, :]
        & (el < max_elevation)[:, None]
        & np.ones((el.size, rng.size), dtype=bool)
    )
    for value, limit, cands, std, what, strict in (
        (rhohv, min_rhohv, _RHOHV_NAMES, _RHOHV_STANDARD, "rhohv", True),
        (dbz, min_dbz, _DBZ_NAMES, _DBZ_STANDARD, "reflectivity", False),
    ):
        name = None if limit is None else _resolve_field(ds, value, cands, std, what)
        if name is not None:
            q = _sweep_arrays(ds, [name], [])[0][0].astype(np.float64)
            ok &= (q > limit) if strict else (q >= limit)
    _, _, z = _antenna_to_cartesian(rng[None, :], el[:, None], _site_altitude(ds))
    height = np.broadcast_to(z, ok.shape)[ok]
    values = np.stack([a[ok].astype(np.float64) for a in arrays])
    return names, height, values


def _antenna_to_cartesian(rng, elevation, site_altitude):
    from xradar.georeference import antenna_to_cartesian

    return antenna_to_cartesian(
        rng, 0.0, elevation, earth_radius=EARTH_RADIUS, site_altitude=site_altitude
    )


# ---------------------------------------------------------------------------
# melting layer on PPIs
# ---------------------------------------------------------------------------


def _ppi_flags_numpy(zh, rho, zdr, hgate, hi, params):
    """Melting-layer points of one sweep: a boolean (ray, gate) array."""
    rho_lo, rho_hi, zdr_lo, zdr_hi, dbz_lo, dbz_hi, hmax = params
    zh, rho, zdr = (np.asarray(a, dtype=np.float64) for a in (zh, rho, zdr))
    flag = np.zeros(zh.shape, dtype=bool)
    candidate = (rho > rho_lo) & (rho < rho_hi) & (hgate[None, :] < hmax)
    zf = np.where(np.isnan(zh), -np.inf, zh)
    df = np.where(np.isnan(zdr), -np.inf, zdr)
    for j in np.flatnonzero(candidate.any(axis=0)):
        zmax = zf[:, j : hi[j]].max(axis=1)
        dmax = df[:, j : hi[j]].max(axis=1)
        flag[:, j] = (
            candidate[:, j]
            & (zmax >= dbz_lo)
            & (zmax <= dbz_hi)
            & (dmax >= zdr_lo)
            & (dmax <= zdr_hi)
        )
    return flag


def _ppi_layer_numpy(sweeps, target, sector, q_top, q_bottom, min_points, params):
    """NumPy implementation of the compiled ``ppi_melting_layer``."""
    pooled = []  # (azimuth of the ray, heights of its points)
    for zh, rho, zdr, az, hgate, hi in sweeps:
        flag = _ppi_flags_numpy(zh, rho, zdr, hgate, hi, params)
        pooled += [(a, hgate[f]) for a, f in zip(az, flag)]
    azimuth = np.array([p[0] for p in pooled])

    def percentiles(points):
        n = points.size
        if n == 0 or n < min_points:
            return np.nan, np.nan, n
        return (
            float(np.percentile(points, q_top)),
            float(np.percentile(points, q_bottom)),
            n,
        )

    top = np.full(target.size, np.nan)
    bottom = np.full(target.size, np.nan)
    npts = np.zeros(target.size, dtype=np.int64)
    for k, t in enumerate(target):
        d = np.abs(np.fmod(azimuth - t, 360.0))
        d = np.where(d > 180.0, 360.0 - d, d)
        take = [p[1] for p, near in zip(pooled, d <= 0.5 * sector) if near]
        points = np.concatenate(take) if take else np.zeros(0)
        top[k], bottom[k], npts[k] = percentiles(points)
    everything = np.concatenate([p[1] for p in pooled]) if pooled else np.zeros(0)
    return top, bottom, npts, *percentiles(everything)


@provenance("Melting layer from the PPIs of a volume, after Giangrande et al. (2008)")
def melting_layer_ppi(
    volumes,
    *,
    dbz="auto",
    rhohv="auto",
    zdr="auto",
    elevation_range=(4.0, 10.0),
    sector=20.0,
    percentiles=(80.0, 20.0),
    rhohv_range=(0.90, 0.97),
    zdr_range=(0.8, 2.5),
    dbz_range=(30.0, 47.0),
    window=500.0,
    height_max=6000.0,
    min_points=20,
    azimuth_step=1.0,
    n_threads=None,
    engine="auto",
):
    """
    Melting-layer top and bottom from the PPIs of a volume.

    This is the radial-profile method of Giangrande et al. (2008) [1]_ for
    PPIs at moderate elevation, where the melting layer shows as a ring of
    low rhohv with a Z and ZDR enhancement. On every ray the gates with
    ``0.90 < rhohv < 0.97`` below 6 km are flagged as melting-layer points if
    the maxima of Z and ZDR in a window of 500 m above them are between 30 and
    47 dBZ and between 0.8 and 2.5 dB (the thresholds and the window of the
    journal paper as given in :func:`melting_layer`, the sector width of the
    conference version of the paper [2]_). The height of a gate is its beam
    height (4/3 Earth model, as in :func:`qvp`). The bottom and top of the
    layer are the 20th and 80th percentile of the heights of the points of
    all rays within a running azimuthal window (Ryzhkov and Krause 2022 [3]_,
    who describe this designation, p. 530), here 20 degrees wide [2]_. The
    sweeps with elevations between 4 and 10 degrees are used together
    [2]_, where the signature is pronounced and the beam does not blur the
    layer.

    Unlike the azimuthal mean of a QVP, the result keeps the dependence on
    the azimuth, and the sweeps are not averaged before the detection. It is
    meant for the layer at heights and ranges where the beam at 4 to 10
    degrees crosses it, that is for a layer between about 1 and 6 km. The
    effect of the beam width on the layer (which Ryzhkov and Krause 2022
    correct with simulated profiles, a method that is not implemented here)
    is not corrected.

    Parameters
    ----------
    volumes : xarray.DataTree, xarray.Dataset or sequence of them
        A radar volume with ``sweep_*`` groups (xradar layout), a sweep, or a
        sequence of volumes in time order. The sweeps with an elevation in
        ``elevation_range`` that have Z, ZDR and rhohv are used. A sweep
        given directly is used whatever its elevation (with a warning if it
        is outside the range).
    dbz, rhohv, zdr : str, optional
        Variable names; ``"auto"`` (default) looks for common names and the
        CF standard names.
    elevation_range : tuple of float, optional
        Elevations (degrees) of the sweeps used. Default ``(4, 10)``.
    sector : float, optional
        Width (degrees) of the azimuthal window. Default 20. The window of a
        target azimuth contains the points of the rays within half of it on
        both sides.
    percentiles : tuple of float, optional
        Percentiles of the heights of the points that give the top and the
        bottom. Default ``(80, 20)``.
    rhohv_range, zdr_range, dbz_range : tuple of float, optional
        Ranges of rhohv at the gate (strictly between), of the maximum of ZDR
        (dB) and of the maximum of Z (dBZ) above it. Defaults
        ``(0.90, 0.97)``, ``(0.8, 2.5)`` and ``(30, 47)``.
    window : float, optional
        Height (m) of the window above the gate in which the maxima are
        searched. Default 500.
    height_max : float, optional
        Gates must be below this height (m above sea level). Default 6000.
    min_points : int, optional
        Minimum number of points in a window for a top and a bottom. Default
        20 (the paper leaves this a parameter; the value is radarx's choice).
    azimuth_step : float, optional
        Spacing (degrees) of the azimuths of the result. Default 1.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use.

    Returns
    -------
    xarray.Dataset
        ``melting_layer_top``, ``melting_layer_bottom`` (m above sea level)
        and ``melting_layer_points`` (number of points) on ``azimuth``; the
        same for all azimuths together as ``melting_layer_top_all``,
        ``melting_layer_bottom_all`` and ``melting_layer_points_all``. A
        sequence of volumes adds the ``time`` dimension. NaN where there are
        fewer than ``min_points`` points.

    See Also
    --------
    melting_layer, qvp, rhi_qvp

    Notes
    -----
    Left out: the weighting of the elevations (the conference paper says the
    higher elevations are weighted), the fuzzy-logic filter of clutter and
    hail, the use of earlier volumes and of model values where nothing is
    found, and the gaps filled in time, all of which belong to the
    operational algorithm.

    References
    ----------
    .. [1] Giangrande, S. E., J. M. Krause, and A. V. Ryzhkov, 2008:
       Automatic designation of the melting layer with a polarimetric
       prototype of the WSR-88D radar. *J. Appl. Meteor. Climatol.*, **47**
       (5), 1354-1364, https://doi.org/10.1175/2007JAMC1634.1
    .. [2] Giangrande, S. E., A. V. Ryzhkov, and J. Krause: Automatic
       detection of the melting layer with a polarimetric prototype of the
       WSR-88D radar. Preprint 11R.2, Conf. on Radar Meteorology, Amer. Meteor.
       Soc. (conference version of the journal paper; no DOI; the edition and
       year of the conference are not given in the preprint),
       https://ams.confex.com/ams/pdfpapers/95894.pdf
    .. [3] Ryzhkov, A., and J. Krause, 2022: New polarimetric radar algorithm
       for melting-layer detection and determination of its height. *J.
       Atmos. Oceanic Technol.*, **39** (5), 529-543,
       https://doi.org/10.1175/JTECH-D-21-0130.1

    Examples
    --------
    >>> ml = radarx.retrieve.melting_layer_ppi(dtree)  # doctest: +SKIP
    >>> ml = dtree.radarx.melting_layer_ppi(sector=30.0)  # doctest: +SKIP
    """
    use_compiled = _use_compiled(engine)
    volumes, series = _as_volumes(volumes)
    if not azimuth_step > 0:
        raise ValueError("azimuth_step must be positive.")
    if not sector > 0:
        raise ValueError("sector must be positive.")
    target = np.arange(0.0, 360.0, float(azimuth_step))
    params = (*map(float, (*rhohv_range, *zdr_range, *dbz_range)), float(height_max))
    top, bottom, npts, tops, bottoms, nall, times = [], [], [], [], [], [], []
    for volume in volumes:
        used = _ppi_sweeps(volume, dbz, rhohv, zdr, elevation_range)
        sweeps = []
        for ds, names in used:
            _, _, h, _ = _geometry(ds)
            if np.any(np.diff(h) <= 0):
                raise ValueError("the beam height must increase with range.")
            arrays, _ = _sweep_arrays(ds, names, [])
            az = np.asarray(ds["azimuth"].values, dtype=np.float64)
            hi = np.searchsorted(h, h + float(window), side="right").astype(np.int64)
            sweeps.append((*arrays, az, h, hi))
        args = (
            target,
            float(sector),
            float(percentiles[0]),
            float(percentiles[1]),
            int(min_points),
        )
        if use_compiled:
            res = _qvp.ppi_melting_layer(
                [s[0] for s in sweeps],
                [s[1] for s in sweeps],
                [s[2] for s in sweeps],
                [s[3] for s in sweeps],
                [s[4] for s in sweeps],
                [s[5] for s in sweeps],
                target,
                *params,
                float(sector),
                float(percentiles[0]),
                float(percentiles[1]),
                int(min_points),
                n_threads=int(n_threads or 0),
            )
        else:
            res = _ppi_layer_numpy(
                [(s[0], s[1], s[2], s[3], s[4], s[5]) for s in sweeps], *args, params
            )
        for store, value in zip((top, bottom, npts, tops, bottoms, nall), res):
            store.append(value)
        times.append(_volume_time([u[0] for u in used]))
    coords = {
        "azimuth": (
            "azimuth",
            target,
            {"long_name": "azimuth of the window centre", "units": "degrees"},
        )
    }
    method = (
        "gates with 0.90 < rhohv < 0.97 and the Z and ZDR maxima above them in range "
        "(after Giangrande et al. 2008); "
        f"{percentiles[0]:g}th and {percentiles[1]:g}th percentile of their heights "
        f"within {sector:g} degrees of azimuth"
    )
    dims = ("time", "azimuth") if series else ("azimuth",)
    scalar = ("time",) if series else ()

    def stack(values, dtype):
        a = np.array(values, dtype=dtype)
        return a if series else a[0]

    height_attrs = {
        "standard_name": "altitude",
        "units": "m",
        "method": method,
    }
    out = xr.Dataset(
        {
            "melting_layer_top": (
                dims,
                stack(top, np.float64),
                {**height_attrs, "long_name": "melting layer top above sea level"},
            ),
            "melting_layer_bottom": (
                dims,
                stack(bottom, np.float64),
                {**height_attrs, "long_name": "melting layer bottom above sea level"},
            ),
            "melting_layer_points": (
                dims,
                stack(npts, np.int64),
                {"long_name": "melting layer points in the window", "units": "1"},
            ),
            "melting_layer_top_all": (
                scalar,
                stack(tops, np.float64),
                {
                    **height_attrs,
                    "long_name": "melting layer top above sea level, all azimuths",
                },
            ),
            "melting_layer_bottom_all": (
                scalar,
                stack(bottoms, np.float64),
                {
                    **height_attrs,
                    "long_name": "melting layer bottom above sea level, all azimuths",
                },
            ),
            "melting_layer_points_all": (
                scalar,
                stack(nall, np.int64),
                {"long_name": "melting layer points, all azimuths", "units": "1"},
            ),
        },
        coords=coords,
    )
    if series:
        out = out.assign_coords(time=("time", np.array(times, dtype="datetime64[ns]")))
    else:
        out = out.assign_coords(time=times[0])
    return out


def _ppi_sweeps(volume, dbz, rhohv, zdr, elevation_range):
    """``[(sweep, (dbz, rhohv, zdr names))]`` of the sweeps used for the PPI layer."""
    direct = isinstance(volume, xr.Dataset)
    chosen = []
    for ds in _volume_sweeps(volume):
        el = float(np.nanmedian(ds["elevation"].values))
        inside = elevation_range[0] <= el <= elevation_range[1]
        if direct and not inside:
            warnings.warn(
                f"the sweep is at {el:.1f} degrees, outside the {elevation_range[0]:g} "
                f"to {elevation_range[1]:g} degrees of the method",
                UserWarning,
                stacklevel=3,
            )
        if not (inside or direct):
            continue
        names = []
        for value, cands, std, what in (
            (dbz, _DBZ_NAMES, _DBZ_STANDARD, "dbz"),
            (rhohv, _RHOHV_NAMES, _RHOHV_STANDARD, "rhohv"),
            (zdr, _ZDR_NAMES, _ZDR_STANDARD, "zdr"),
        ):
            names.append(_resolve_field(ds, value, cands, std, what))
        if None in names:
            if direct:
                raise ValueError("the sweep needs Z, ZDR and rhohv fields.")
            continue
        chosen.append((ds, names))
    if not chosen:
        raise ValueError(
            f"No sweep between {elevation_range[0]:g} and {elevation_range[1]:g} "
            "degrees has reflectivity, ZDR and rhohv."
        )
    return chosen


# ---------------------------------------------------------------------------
# rain-only products below the melting layer
# ---------------------------------------------------------------------------

#: Products of the retrievals that are valid in rain only, set to NaN above
#: the melting layer by default: the rain rates of :mod:`radarx.retrieve.qpe`
#: and the drop size distribution parameters of :func:`radarx.retrieve.dsd`.
RAIN_ONLY_PRODUCTS = (
    "RAIN_RATE",
    "RAIN_RATE_Z",
    "RAIN_RATE_KDP",
    "RAIN_RATE_Z_ZDR",
    "RAIN_RATE_A",
    "N0",
    "NW",
    "D0",
    "DM",
    "MU",
    "LAMBDA",
    "LWC",
)

#: Value of ``RAIN_RATE_FLAG`` where the rain rate was removed above the layer.
ABOVE_MELTING_LAYER_FLAG = 4


def _gate_height(obj, height, beam_width):
    """
    Height (m above sea level) of the gates of a sweep, or the height of the
    lower edge of the beam if ``beam_width`` (degrees) is given.
    """
    if height is None:
        if "elevation" not in obj.coords or "range" not in obj.coords:
            raise ValueError(
                "the gate heights need the elevation and range coordinates of the "
                "sweep, or give height="
            )
        el = np.asarray(obj["elevation"].values, dtype=np.float64)
        rng = np.asarray(obj["range"].values, dtype=np.float64)
        alt = float(obj["altitude"].values) if "altitude" in obj.coords else 0.0
        if el.ndim == 0:
            z = _antenna_to_cartesian(rng, float(el), alt)[2]
            dims = ("range",)
        elif el.ndim == 1:
            z = _antenna_to_cartesian(rng[None, :], el[:, None], alt)[2]
            dims = (obj["elevation"].dims[0], "range")
        else:
            raise ValueError("the elevation must be a scalar or one value per ray.")
        height = xr.DataArray(np.asarray(z), dims=dims)
    if beam_width:
        rng = obj["range"].reset_coords(drop=True)
        el = np.deg2rad(obj["elevation"].reset_coords(drop=True))
        half = 0.5 * np.deg2rad(float(beam_width)) * rng * np.cos(el)
        height = height - half
    return height


def _cap_height(melting_layer, bottom, margin):
    """Height (m) above which rain-only products are removed."""
    if isinstance(melting_layer, xr.Dataset):
        if bottom not in melting_layer:
            raise ValueError(f"the melting layer has no {bottom!r}.")
        cap = melting_layer[bottom]
    elif isinstance(melting_layer, xr.DataArray):
        cap = melting_layer
    else:
        cap = xr.DataArray(np.asarray(melting_layer, dtype=np.float64))
    return cap.astype(np.float64).reset_coords(drop=True) + float(margin)


def _align_azimuth(cap, obj):
    """``cap`` on the azimuth of the rays of ``obj`` (nearest, circular)."""
    if "azimuth" not in cap.dims or "azimuth" not in cap.coords:
        return cap
    if "azimuth" not in obj.coords or obj["azimuth"].ndim != 1:
        raise ValueError(
            "a melting layer that depends on the azimuth needs products with an "
            "azimuth coordinate."
        )
    ray = obj["azimuth"].dims[0]
    az = np.asarray(obj["azimuth"].values, dtype=np.float64)
    ref = np.asarray(cap["azimuth"].values, dtype=np.float64)
    d = np.abs((az[:, None] - ref[None, :] + 180.0) % 360.0 - 180.0)
    picked = cap.isel(azimuth=xr.DataArray(np.argmin(d, axis=1), dims=ray))
    return picked.drop_vars("azimuth", errors="ignore")


def _capped_flag_attrs():
    from .qpe import FLAG_MEANINGS

    return {
        "long_name": "why the rain rate is missing",
        "flag_values": np.arange(len(FLAG_MEANINGS) + 1, dtype=np.int8),
        "flag_meanings": " ".join((*FLAG_MEANINGS, "above_melting_layer")),
        "comment": (
            "0: estimated; 1: no valid relation for the hydrometeor class; "
            "2: an input of the relation is missing; 3: the gate is not "
            "classified; 4: removed above the melting layer"
        ),
    }


def _cap_sweep(obj, melting_layer, bottom, height, margin, beam_width, variables):
    cap = _align_azimuth(_cap_height(melting_layer, bottom, margin), obj)
    edge = _gate_height(obj, height, beam_width)
    above = edge > cap  # NaN where the layer or the height is unknown: kept
    text = (
        f"set to NaN where the beam is above the melting layer bottom + {margin:g} m"
        + (f" (lower edge of a {beam_width:g} degree beam)" if beam_width else "")
    )
    if isinstance(obj, xr.DataArray):
        out = obj.where(~above)
        out.attrs["comment"] = "; ".join(filter(None, [obj.attrs.get("comment"), text]))
        return out
    if variables is None:
        names = [v for v in RAIN_ONLY_PRODUCTS if v in obj.data_vars]
    else:
        names = [variables] if isinstance(variables, str) else list(variables)
        missing = [v for v in names if v not in obj.data_vars]
        if missing:
            raise ValueError(f"no {missing} in the dataset.")
    out = obj.copy()
    removed = None
    if "RAIN_RATE_FLAG" in obj and "RAIN_RATE" in obj:
        removed = above & np.isfinite(obj["RAIN_RATE"])
    for name in names:
        da = obj[name]
        out[name] = da.where(~above)
        out[name].attrs = dict(da.attrs)
        out[name].attrs["comment"] = "; ".join(
            filter(None, [da.attrs.get("comment"), text])
        )
    if removed is not None and "RAIN_RATE" in names:
        flag = xr.where(removed, ABOVE_MELTING_LAYER_FLAG, obj["RAIN_RATE_FLAG"])
        out["RAIN_RATE_FLAG"] = flag.astype(obj["RAIN_RATE_FLAG"].dtype)
        out["RAIN_RATE_FLAG"].attrs = _capped_flag_attrs()
        if "RAIN_RATE_RELATION" in obj:
            rel = xr.where(removed, 0, obj["RAIN_RATE_RELATION"])
            out["RAIN_RATE_RELATION"] = rel.astype(obj["RAIN_RATE_RELATION"].dtype)
            out["RAIN_RATE_RELATION"].attrs = dict(obj["RAIN_RATE_RELATION"].attrs)
    return out


@provenance(
    "Rain-only products set to NaN above the melting layer bottom, radarx's own"
)
def mask_above_melting_layer(
    products,
    melting_layer,
    *,
    bottom="melting_layer_bottom",
    height=None,
    margin=0.0,
    beam_width=None,
    variables=None,
):
    """
    Remove the products of rain-only retrievals above the melting layer.

    The rain rates of :func:`radarx.retrieve.rain_rate_z`, ``rain_rate_kdp``,
    ``rain_rate_z_zdr``, ``rain_rate_a`` and the drop size distribution
    parameters of :func:`radarx.retrieve.dsd` hold for rain. Above the
    melting layer the radar sees snow and ice, and in it wet snow, for which
    the relations do not apply. :func:`radarx.retrieve.rain_rate` already
    leaves the gates of the hydrometeor classes without a relation (including
    the melting layer, "wet snow") empty, but depends on the classification;
    this function makes a cap from a melting-layer height instead, for example
    from :func:`melting_layer` (profiles of a QVP, with the time of the
    volume) or :func:`melting_layer_ppi` (with the azimuth).

    A gate is removed where its height is above the bottom of the melting
    layer plus ``margin``. The bottom is the end of snow melting (Ryzhkov
    and Krause 2022 [1]_, p. 533), so below it there are no wet snow and no
    ice: the rain-only relations apply from there downward. The ``margin``
    (default 0 m) moves the cap, and ``beam_width`` makes it use the lower
    edge of the beam instead of its centre, because a beam whose lower half
    is in rain and upper half in the layer is contaminated. Both are radarx's
    definition and are not taken from a paper. Gates are kept where the
    melting layer is not known (NaN): the function does not assume that there
    is no melting layer.

    Parameters
    ----------
    products : xarray.DataArray, xarray.Dataset or xarray.DataTree
        A rain-only product (DataArray), a sweep with such products
        (Dataset), or a volume. Datasets need the ``elevation`` and ``range``
        coordinates (and ``altitude``) for the gate heights.
    melting_layer : xarray.Dataset, xarray.DataArray or float
        The bottom of the melting layer, in m above sea level: the output of
        :func:`melting_layer` or :func:`melting_layer_ppi` (variable
        ``bottom``), a DataArray, or a number. A DataArray or variable on
        ``azimuth`` is matched to the azimuths of the rays (nearest, circular);
        one without dimensions applies to all gates. Select one time of a
        time series first (``isel(time=...)``).
    bottom : str, optional
        Variable of a Dataset that holds the bottom. Default
        ``"melting_layer_bottom"``.
    height : xarray.DataArray, optional
        Height of the gates (m above sea level). By default the beam height
        of :func:`qvp` (4/3 Earth model) from the ``elevation``, ``range`` and
        ``altitude`` coordinates.
    margin : float, optional
        Added to the bottom (m); gates above the sum are removed. Default 0.
    beam_width : float, optional
        Two-way half-power beam width (degrees). With it the height compared
        with the cap is the lower edge of the beam, ``height - 0.5 * width *
        range * cos(elevation)``. Default None: the centre of the beam.
    variables : str or list of str, optional
        The variables of a Dataset to remove. Default: those of
        :data:`RAIN_ONLY_PRODUCTS` the Dataset has.

    Returns
    -------
    xarray.DataArray, xarray.Dataset or xarray.DataTree
        The products with NaN above the layer; other variables are
        unchanged. In a Dataset with ``RAIN_RATE`` and ``RAIN_RATE_FLAG`` the
        flag is 4 (``above_melting_layer``) and ``RAIN_RATE_RELATION`` 0
        where an estimate was removed.

    See Also
    --------
    melting_layer, melting_layer_ppi, radarx.retrieve.rain_rate

    Notes
    -----
    Nothing here estimates the rain rate in the melting layer or above it:
    the relations scaled for wet snow and dry snow of Giangrande and Ryzhkov
    (2007) are listed in :mod:`radarx.retrieve.qpe` and are chosen there with
    ``class_relations``.

    References
    ----------
    .. [1] Ryzhkov, A., and J. Krause, 2022: New polarimetric radar algorithm
       for melting-layer detection and determination of its height. *J.
       Atmos. Oceanic Technol.*, **39** (5), 529-543,
       https://doi.org/10.1175/JTECH-D-21-0130.1

    Examples
    --------
    >>> ml = radarx.retrieve.melting_layer_ppi(dtree)  # doctest: +SKIP
    >>> rr = radarx.retrieve.rain_rate_z(sweep)  # doctest: +SKIP
    >>> rr = radarx.retrieve.mask_above_melting_layer(rr, ml, margin=-200.0)  # doctest: +SKIP
    """
    if isinstance(products, xr.DataTree):
        from ._fieldtools import map_sweeps

        return map_sweeps(
            products,
            lambda ds, name: _cap_sweep(
                ds, melting_layer, bottom, height, margin, beam_width, variables
            ),
            "mask_above_melting_layer",
        )
    if not isinstance(products, (xr.DataArray, xr.Dataset)):
        raise TypeError("products must be an xarray DataArray, Dataset or DataTree.")
    return _cap_sweep(
        products, melting_layer, bottom, height, margin, beam_width, variables
    )


# ---------------------------------------------------------------------------
# accessors
# ---------------------------------------------------------------------------


@accessor_method("dataset", "datatree", name="rd_qvp")
def _rd_qvp_accessor(self, data_vars=None, **kwargs):
    """
    Range-defined quasi-vertical profile of this volume.

    Parameters
    ----------
    data_vars : str or list of str, optional
        Variables to profile.
    **kwargs
        Options of :func:`radarx.retrieve.rd_qvp` (``rmax``, ``power``,
        ``dz``, ...).

    Returns
    -------
    xarray.Dataset
        Profiles on the ``height`` dimension.

    See Also
    --------
    radarx.retrieve.rd_qvp
    """
    return rd_qvp(self.xarray_obj, data_vars, **kwargs)


@accessor_method("dataset", "datatree", name="cvp")
def _cvp_accessor(self, data_vars=None, **kwargs):
    """
    Columnar vertical profile of this volume.

    Parameters
    ----------
    data_vars : str or list of str, optional
        Variables to profile.
    **kwargs
        Options of :func:`radarx.retrieve.cvp`; ``azimuth`` and
        ``ground_range`` of the centre are required.

    Returns
    -------
    xarray.Dataset
        Profiles on the ``height`` dimension.

    See Also
    --------
    radarx.retrieve.cvp
    """
    return cvp(self.xarray_obj, data_vars, **kwargs)


@accessor_method("dataset", "datatree", name="rhi_qvp")
def _rhi_qvp_accessor(self, data_vars=None, **kwargs):
    """
    Vertical profile of this RHI scan.

    Parameters
    ----------
    data_vars : str or list of str, optional
        Variables to profile.
    **kwargs
        Options of :func:`radarx.retrieve.rhi_qvp` (``rmin``, ``rmax``,
        ``dz``, ...).

    Returns
    -------
    xarray.Dataset
        Profiles on the ``height`` dimension.

    See Also
    --------
    radarx.retrieve.rhi_qvp
    """
    return rhi_qvp(self.xarray_obj, data_vars, **kwargs)


@accessor_method("dataset", "datatree", name="melting_layer_ppi")
def _melting_layer_ppi_accessor(self, **kwargs):
    """
    Melting-layer top and bottom from the PPIs of this volume.

    Parameters
    ----------
    **kwargs
        Options of :func:`radarx.retrieve.melting_layer_ppi`
        (``elevation_range``, ``sector``, ``percentiles``, ...).

    Returns
    -------
    xarray.Dataset
        ``melting_layer_top`` and ``melting_layer_bottom`` (m above sea
        level) on ``azimuth``, and the same for all azimuths.

    See Also
    --------
    radarx.retrieve.melting_layer_ppi
    """
    return melting_layer_ppi(self.xarray_obj, **kwargs)


@accessor_method("dataarray", "dataset", "datatree", name="mask_above_melting_layer")
def _mask_above_melting_layer_accessor(self, melting_layer, **kwargs):
    """
    Remove rain-only products above the melting layer.

    Parameters
    ----------
    melting_layer : xarray.Dataset, xarray.DataArray or float
        The bottom of the melting layer (m above sea level), for example the
        output of ``melting_layer_ppi``.
    **kwargs
        Options of :func:`radarx.retrieve.mask_above_melting_layer`
        (``margin``, ``beam_width``, ...).

    Returns
    -------
    xarray.DataArray, xarray.Dataset or xarray.DataTree
        The products with NaN above the layer.

    See Also
    --------
    radarx.retrieve.mask_above_melting_layer
    """
    return mask_above_melting_layer(self.xarray_obj, melting_layer, **kwargs)
