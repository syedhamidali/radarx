#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx Hydrometeor Size Sorting
===============================

The separation of the enhanced KDP and ZDR regions at low levels of a storm, from a
polarimetric radar grid, after Loeffler and Kumjian (2018).

The storm-relative wind sorts the falling raindrops by size: the large drops fall
out first and give high differential reflectivity on the inflow side of the storm,
the small drops are carried farther and give the high specific differential phase
that comes from their number. The vector from the region of high KDP to the region of
high ZDR, the separation vector, is thought to be antiparallel to the mean
storm-relative wind of the sorting layer. Its length follows the strength of that wind
and its direction relative to the storm motion its turning with height (Loeffler and
Kumjian 2018).

Algorithm
---------
:func:`storm_size_sorting` follows the method of Loeffler and Kumjian (2018, section 2) for
the lowest level of a grid, in every volume:

1. A grid point of an enhancement region lies in the analysis region, has the copolar
   correlation coefficient at least 0.93 and a KDP (ZDR) at or above the threshold of
   the region.
2. The threshold is adaptive: it is the largest multiple of 0.5 degrees km-1 (dB), from
   zero up, for which at least ``min_cells`` grid points (10 in the paper) belong to
   the region. So only the number of points is set, and the thresholds follow the
   strength of the signature (and are insensitive to a ZDR bias of 1 to 2 dB).
3. The centroid of a region is the median of the x and of the y coordinates of its
   points.
4. The separation vector points from the KDP centroid to the ZDR centroid. Its
   orientation is the angle, clockwise, from the storm motion.
5. The storm motion is, if it is not given, the mean of the velocities of straight
   lines that are fitted by least squares to the KDP and to the ZDR centroids over
   the volumes.

The size sorting parameter of the paper, :math:`\\mathrm{SSP} = D\\,|\\sin A|` with the
separation distance :math:`D` and the orientation :math:`A`, is returned for every volume;
the paper uses the medians of both within 10 min of a tornado report.

What the paper leaves open, and what radarx does:

- The paper works on the lowest elevation angle (0.5 degrees) of the WSR-88D in a box of
  20 km by 20 km around a rotation. Here the box (``box``) or any mask (``region``,
  for example one cell of :func:`radarx.retrieve.track_cells`) is the analysis region and the
  default is the whole grid, which can hold several storms.
- Gates are replaced by grid points: choose ``min_cells`` for the grid spacing (10
  gates of the WSR-88D cover roughly 2 km2).
- The velocity of a centroid series is the least squares slope of its x and y against
  time. The paper fits a line to the centroids and takes the speed from the first and
  the last point.
- The orientation is wrapped to the range -180 to 180 degrees, positive to the right of
  the storm motion. The thresholds are not allowed to fall below zero: a volume whose
  ``min_cells``-th largest value is negative has no region.
- The angle to the storm-relative wind is added (radarx): the angle between the
  separation vector and the direction opposite to the storm-relative wind
  ``wind`` minus the storm motion, 0 for a perfect alignment.

Fields to prepare first
-----------------------
The paper takes the operational KDP and ZDR of the WSR-88D (Level III) in storms within
60 km of the radar, and the grid should resolve the regions: use the lowest tilt
(:func:`radarx.grid.grid_cones` at a fine spacing), a calibrated ZDR and the KDP of
:func:`radarx.retrieve.estimate_kdp`. :func:`radarx.retrieve.zdr_arc` measures the same
signature with the object-based algorithm of Wilson and Van Den Broeke (2021) for one
supercell.

Implementation
--------------
A compiled, multithreaded C++ kernel finds the thresholds and the centroids of all
volumes, with an equivalent NumPy implementation as fallback and test oracle.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = ["storm_size_sorting"]

__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

import numpy as np
import xarray as xr

from .._provenance import provenance
from .._registry import accessor_method
from .._sweepwise import dask_support
from ._fieldtools import KDP_NAMES, RHOHV_NAMES, ZDR_NAMES, choose_engine, find_field

try:
    from . import _sorting

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _sorting = None
    HAS_COMPILED_KERNEL = False

_VERTICAL_NAMES = ("z", "altitude", "height", "level")


# --------------------------------------------------------------------------
# the adaptive threshold: NumPy reference of the kernel
# --------------------------------------------------------------------------


def _adaptive_numpy(values, x, y, min_cells, step, floor_value):
    """Threshold, median x, median y and number of cells per frame; the kernel's twin."""
    nt = values.shape[0]
    out = np.full((4, nt), np.nan)
    out[3] = 0.0
    for t in range(nt):
        v = values[t]
        finite = np.isfinite(v)
        if finite.sum() < min_cells:
            continue
        kth = np.sort(v[finite])[::-1][min_cells - 1]
        level = np.floor(kth / step) * step
        if level < floor_value:
            continue
        chosen = finite & (v >= level)
        out[:, t] = (
            level,
            np.median(x[chosen]),
            np.median(y[chosen]),
            float(chosen.sum()),
        )
    return out


def _adaptive(values, x, y, min_cells, step, floor_value, use_compiled, n_threads):
    values = np.ascontiguousarray(values, dtype=np.float64)
    if use_compiled:
        res = _sorting.adaptive(
            values,
            x,
            y,
            int(min_cells),
            float(step),
            float(floor_value),
            int(n_threads or 0),
        )
        return np.stack(res)
    return _adaptive_numpy(
        values, x, y, int(min_cells), float(step), float(floor_value)
    )


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def _lowest(ds, level):
    """The fields at the lowest vertical level (or the one nearest to ``level``)."""
    for name in _VERTICAL_NAMES:
        if name in ds.dims:
            values = ds[name].values
            k = int(
                np.argmin(values) if level is None else np.argmin(abs(values - level))
            )
            return ds.isel({name: k})
    return ds


def _frames(da, name):
    """The field as (time, y, x) values; a grid without time gets one frame."""
    extra = [d for d in da.dims if d not in ("time", "y", "x")]
    if extra or not {"y", "x"} <= set(da.dims):
        raise ValueError(
            f"{name} needs the dimensions (time,) y and x only after the vertical "
            f"level is chosen, not {da.dims}"
        )
    if "time" not in da.dims:
        da = da.expand_dims("time")
    return da.transpose("time", "y", "x")


def _grid_spacing(ds):
    out = []
    for d in ("x", "y"):
        if d not in ds.coords or ds[d].size < 2:
            raise ValueError(f"the grid needs a coordinate {d!r} (m)")
        out.append(float(np.median(np.abs(np.diff(ds[d].values)))))
    return out


def _seconds(time):
    """Time in seconds from the first value; None for a missing or single time."""
    if time is None or time.size < 2:
        return None
    values = time.values
    if np.issubdtype(values.dtype, np.datetime64):
        return (values - values[0]) / np.timedelta64(1, "s")
    return np.asarray(values, dtype=float) - float(values[0])


def _velocity(seconds, x, y):
    """Least squares slope of a centroid series against time, NaN without two points."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2 or np.ptp(seconds[ok]) == 0.0:
        return np.nan, np.nan
    return (
        float(np.polyfit(seconds[ok], x[ok], 1)[0]),
        float(np.polyfit(seconds[ok], y[ok], 1)[0]),
    )


def _given_motion(motion, nt):
    """Storm motion (u, v) per volume from a tuple, DataArrays or a dataset with ``u`` and ``v``."""
    if isinstance(motion, xr.Dataset):
        u, v = motion["u"], motion["v"]
    else:
        try:
            u, v = motion
        except (TypeError, ValueError):
            raise TypeError(
                "motion must be (u, v) in m/s or a dataset with u and v"
            ) from None
    out = []
    for comp in (u, v):
        if isinstance(comp, xr.DataArray) and comp.ndim:
            if "time" not in comp.dims or comp.sizes["time"] != nt:
                raise ValueError("a motion with a dimension needs one value per time")
            out.append(np.asarray(comp.values, dtype=float))
        else:
            out.append(np.full(nt, float(comp)))
    return out[0], out[1]


def _bearing(dx, dy):
    """Bearing in degrees clockwise from north of the vector (dx east, dy north)."""
    return np.mod(np.degrees(np.arctan2(dx, dy)), 360.0)


def _wrap(angle):
    """Angle in degrees wrapped to (-180, 180]."""
    return 180.0 - np.mod(180.0 - np.asarray(angle, dtype=float), 360.0)


# --------------------------------------------------------------------------
# the algorithm
# --------------------------------------------------------------------------


@dask_support(
    "computes",
    "the storm motion is fitted to the centroids of all volumes.",
)
@provenance(
    "KDP-ZDR separation vector and size sorting parameter of Loeffler and Kumjian (2018)"
)
def storm_size_sorting(
    obj,
    kdp=None,
    zdr=None,
    rhohv=None,
    *,
    level=None,
    box=None,
    region=None,
    motion=None,
    wind=None,
    min_rhohv=0.93,
    min_cells=10,
    step=0.5,
    n_threads=None,
    engine="auto",
):
    """
    KDP-ZDR separation vector and size sorting parameter of the volumes of a grid.

    See :mod:`radarx.retrieve.sorting` for the method of Loeffler and Kumjian (2018)
    and for what the paper leaves open.

    Parameters
    ----------
    obj : xarray.Dataset
        Fields on a grid with the coordinates ``x`` (east) and ``y`` (north) in m and, if
        there are several volumes, ``time``. A vertical dimension (``z``, ``altitude``,
        ``height`` or ``level``) is reduced to its lowest level, or the one nearest
        to ``level``.
    kdp, zdr, rhohv : str, optional
        Names of the specific differential phase (degrees km-1), the differential
        reflectivity (dB) and the copolar correlation coefficient. Default: the first
        usual name of KDP and ZDR; ``rhohv`` has no default and without it the
        correlation coefficient is not used.
    level : float, optional
        Height (m) of the grid level to use. Default: the lowest.
    box : tuple of float, optional
        Analysis region ``(xmin, xmax, ymin, ymax)`` in m. Default: the whole grid.
    region : xarray.DataArray of bool, optional
        Analysis region as a mask on the grid (and time), for example
        ``cells == 5``. Combined with ``box``.
    motion : tuple, xarray.Dataset or None, optional
        Storm motion ``(u, v)`` in m/s, or a dataset with ``u`` and ``v`` (scalars or
        one value per time, e.g. from :func:`radarx.retrieve.estimate_motion`). Default:
        fitted to the centroids over the volumes (needs at least two volumes with
        centroids).
    wind : tuple of float, optional
        Mean environmental wind ``(u, v)`` in m/s of the sorting layer. If given, the
        angle to the storm-relative wind is returned.
    min_rhohv : float, optional
        Smallest correlation coefficient of a grid point of the regions. Default 0.93.
        Used only with ``rhohv``.
    min_cells : int, optional
        Smallest number of grid points of a region (``N_G`` of the paper). Default 10.
    step : float, optional
        Step of the adaptive thresholds, in degrees km-1 and dB. Default 0.5.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled kernel and
        falls back to NumPy.

    Returns
    -------
    xarray.Dataset
        On ``time`` (NaN where a region does not exist): ``KDP_THRESHOLD`` and
        ``ZDR_THRESHOLD``, ``KDP_CENTROID_X``, ``KDP_CENTROID_Y``, ``ZDR_CENTROID_X`` and
        ``ZDR_CENTROID_Y`` (m), ``KDP_REGION_AREA`` and ``ZDR_REGION_AREA`` (km2),
        ``STORM_MOTION_U`` and ``STORM_MOTION_V`` (m/s), ``SEPARATION_DISTANCE`` (km),
        ``SEPARATION_BEARING`` (degrees clockwise from north, direction of the vector),
        ``SEPARATION_ORIENTATION`` (degrees, from -180 to 180, positive clockwise from the
        storm motion, i.e. to its right), ``SIZE_SORTING_PARAMETER`` (km, the
        distance times the absolute sine of the orientation) and, with ``wind``,
        ``SORTING_WIND_ANGLE`` (degrees, 0 to 180).

    Raises
    ------
    ValueError
        For an unknown ``engine``, a field with other dimensions than ``y``, ``x`` and
        ``time`` (after the vertical level is chosen), a missing grid coordinate, a
        ``min_cells`` below 1, a ``step`` that is not positive, or a time dimension
        without a time coordinate when the motion is fitted.
    KeyError
        If a field is not found.
    TypeError
        If ``obj`` is not a dataset or ``motion`` is not a pair.

    See Also
    --------
    zdr_arc, estimate_motion

    References
    ----------
    Loeffler, S. D., and M. R. Kumjian, 2018: Quantifying the separation of enhanced ZDR
    and KDP regions in nonsupercell tornadic storms. *Wea. Forecasting*, **33**,
    1143-1157, https://doi.org/10.1175/WAF-D-18-0011.1

    Examples
    --------
    >>> res = radarx.retrieve.storm_size_sorting(grid, rhohv="RHOHV", box=(0, 20e3, 0, 20e3))  # doctest: +SKIP
    >>> res.SEPARATION_ORIENTATION  # doctest: +SKIP
    """
    if not isinstance(obj, xr.Dataset):
        raise TypeError("obj must be an xarray.Dataset holding the fields")
    use_compiled = choose_engine(engine, HAS_COMPILED_KERNEL, "size sorting")
    if int(min_cells) < 1:
        raise ValueError("min_cells must be at least 1")
    if not step > 0:
        raise ValueError("step must be positive")
    ds = _lowest(obj, level)
    kdp_name = find_field(ds, kdp, KDP_NAMES, "specific differential phase")
    zdr_name = find_field(ds, zdr, ZDR_NAMES, "differential reflectivity")
    rho_name = (
        None
        if rhohv is None
        else find_field(ds, rhohv, RHOHV_NAMES, "correlation coefficient")
    )
    dx, dy = _grid_spacing(ds)
    kdp_f = _frames(ds[kdp_name], kdp_name)
    zdr_f = _frames(ds[zdr_name], zdr_name).broadcast_like(kdp_f).transpose(*kdp_f.dims)
    rho = None if rho_name is None else _frames(ds[rho_name], rho_name)
    eligible = _eligible(kdp_f, rho, min_rhohv, box, region)
    xx, yy = np.meshgrid(
        kdp_f["x"].values.astype(float), kdp_f["y"].values.astype(float)
    )
    xs, ys = np.ascontiguousarray(xx.ravel()), np.ascontiguousarray(yy.ravel())
    nt = kdp_f.sizes["time"]
    results = {}
    for key, field in (("KDP", kdp_f), ("ZDR", zdr_f)):
        values = np.where(eligible, field.transpose(*kdp_f.dims).values, np.nan)
        results[key] = _adaptive(
            values.reshape(nt, -1),
            xs,
            ys,
            min_cells,
            step,
            0.0,
            use_compiled,
            n_threads,
        )
    time = kdp_f["time"] if "time" in kdp_f.coords else None
    u, v = _storm_motion(motion, results, nt, _seconds(time))
    out = _separation_dataset(results, u, v, dx * dy / 1e6, time)
    if wind is not None:
        _add_wind_angle(out, wind, u, v)
    out.attrs["comment"] = (
        f"adaptive thresholds with at least {int(min_cells)} grid points in steps of {step:g}"
        + (
            f", correlation coefficient >= {min_rhohv:g}"
            if rho_name is not None
            else ""
        )
    )
    return out


def _eligible(kdp_f, rho, min_rhohv, box, region):
    """Grid points that may belong to an enhancement region, as (time, y, x) values."""
    eligible = xr.ones_like(kdp_f, dtype=bool)
    if rho is not None:
        eligible = eligible & (rho >= float(min_rhohv))
    if box is not None:
        xmin, xmax, ymin, ymax = (float(b) for b in box)
        eligible = (
            eligible
            & (kdp_f["x"] >= xmin)
            & (kdp_f["x"] <= xmax)
            & (kdp_f["y"] >= ymin)
            & (kdp_f["y"] <= ymax)
        )
    if region is not None:
        mask = region.astype(bool)
        if "time" not in mask.dims and "time" in kdp_f.dims:
            mask = mask.expand_dims(time=kdp_f["time"])
        eligible = eligible & mask.broadcast_like(kdp_f)
    return eligible.transpose(*kdp_f.dims).values


def _storm_motion(motion, results, nt, seconds):
    """The storm motion (u, v) per volume: given, or fitted to the centroids."""
    if motion is not None:
        return _given_motion(motion, nt)
    if seconds is None:
        return np.full(nt, np.nan), np.full(nt, np.nan)
    velocities = [
        _velocity(seconds, results[k][1], results[k][2]) for k in ("KDP", "ZDR")
    ]
    good = [vel for vel in velocities if np.isfinite(vel[0])]
    mean = np.mean(good, axis=0) if good else (np.nan, np.nan)
    return np.full(nt, mean[0]), np.full(nt, mean[1])


_SORTING_ATTRS = {
    "KDP_THRESHOLD": (
        "Adaptive KDP threshold of the enhancement region",
        "degree km-1",
    ),
    "ZDR_THRESHOLD": ("Adaptive ZDR threshold of the enhancement region", "dB"),
    "KDP_CENTROID_X": ("x of the centroid of the KDP enhancement region", "m"),
    "KDP_CENTROID_Y": ("y of the centroid of the KDP enhancement region", "m"),
    "ZDR_CENTROID_X": ("x of the centroid of the ZDR enhancement region", "m"),
    "ZDR_CENTROID_Y": ("y of the centroid of the ZDR enhancement region", "m"),
    "KDP_REGION_AREA": ("Area of the KDP enhancement region", "km2"),
    "ZDR_REGION_AREA": ("Area of the ZDR enhancement region", "km2"),
    "STORM_MOTION_U": ("Eastward storm motion", "m s-1"),
    "STORM_MOTION_V": ("Northward storm motion", "m s-1"),
    "SEPARATION_DISTANCE": ("Distance from the KDP to the ZDR centroid", "km"),
    "SEPARATION_BEARING": (
        "Direction of the separation vector, clockwise from north",
        "degree",
    ),
    "SEPARATION_ORIENTATION": (
        "Direction of the separation vector, clockwise from the storm motion",
        "degree",
    ),
    "SIZE_SORTING_PARAMETER": (
        "Separation distance times the absolute sine of the orientation",
        "km",
    ),
}


def _separation_dataset(results, u, v, cell_km2, time):
    """The thresholds, centroids and separation vector of every volume as a dataset."""
    ddx = results["ZDR"][1] - results["KDP"][1]
    ddy = results["ZDR"][2] - results["KDP"][2]
    distance = np.hypot(ddx, ddy) / 1e3
    bearing = _bearing(ddx, ddy)
    with np.errstate(invalid="ignore"):
        orientation = _wrap(bearing - _bearing(u, v))
        parameter = distance * np.abs(np.sin(np.radians(orientation)))
    dims = ("time",)

    def area(key):
        return np.where(results[key][3] > 0, results[key][3] * cell_km2, np.nan)

    out = xr.Dataset(
        {
            "KDP_THRESHOLD": (dims, results["KDP"][0]),
            "ZDR_THRESHOLD": (dims, results["ZDR"][0]),
            "KDP_CENTROID_X": (dims, results["KDP"][1]),
            "KDP_CENTROID_Y": (dims, results["KDP"][2]),
            "ZDR_CENTROID_X": (dims, results["ZDR"][1]),
            "ZDR_CENTROID_Y": (dims, results["ZDR"][2]),
            "KDP_REGION_AREA": (dims, area("KDP")),
            "ZDR_REGION_AREA": (dims, area("ZDR")),
            "STORM_MOTION_U": (dims, u),
            "STORM_MOTION_V": (dims, v),
            "SEPARATION_DISTANCE": (dims, distance),
            "SEPARATION_BEARING": (
                dims,
                np.where(np.isfinite(distance), bearing, np.nan),
            ),
            "SEPARATION_ORIENTATION": (dims, orientation),
            "SIZE_SORTING_PARAMETER": (dims, parameter),
        },
        coords={"time": time.values} if time is not None else {},
    )
    for name, (long_name, unit) in _SORTING_ATTRS.items():
        out[name].attrs = {"long_name": long_name, "units": unit}
    return out


def _add_wind_angle(out, wind, u, v):
    """Add the angle between the separation vector and the direction opposite to the SR wind."""
    try:
        wu, wv = (float(w) for w in wind)
    except (TypeError, ValueError):
        raise TypeError("wind must be (u, v) in m/s") from None
    with np.errstate(invalid="ignore"):
        toward = _bearing(-(wu - u), -(wv - v))  # opposite to the storm-relative wind
        angle = np.abs(_wrap(out["SEPARATION_BEARING"].values - toward))
    out["SORTING_WIND_ANGLE"] = (("time",), angle)
    out["SORTING_WIND_ANGLE"].attrs = {
        "long_name": "Angle between the separation vector and the direction opposite to the storm-relative wind",
        "units": "degree",
    }


@accessor_method("dataset", name="storm_size_sorting", wraps=storm_size_sorting)
def _size_sorting_accessor(self):
    """
    KDP-ZDR separation vector and size sorting parameter of the volumes of a grid.

    See :func:`radarx.retrieve.storm_size_sorting` for the parameters.

    Returns
    -------
    xarray.Dataset
        The thresholds, centroids, separation vector and size sorting parameter per
        time.
    """
