#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx ZDR Columns
==================

The depth of the ZDR column of a convective storm and its size, from a
polarimetric radar volume on a grid.

Drops that are lofted by an updraft above the environmental freezing level
freeze slowly, so that a column of enhanced differential reflectivity,
:math:`Z_{DR} \\ge 1` dB, stands above the 0 degree Celsius level
(Kumjian et al. 2014; Snyder et al. 2015). Its depth follows the strength of
the updraft, and a growing column precedes the intensification of the storm.

Algorithm
---------
:func:`zdr_column` follows the algorithm of Snyder et al. (2015, section 3). At
every horizontal grid point it counts the vertically consecutive grid points
with :math:`Z_{DR} \\ge 1` dB above the freezing level, so that the result is the
vertically continuous depth of the column. Grid points with a copolar
correlation coefficient below about 0.80 are removed, which keeps the three-body
scatter signature (hail spikes) and the depolarization streaks out of the
product.

What the paper leaves open, and what radarx does:

- The count starts at the first grid level at or above the freezing level, and the
  depth is the sum of the layer thicknesses of the levels counted (the number of
  levels times 250 m for the grid of the paper). A column that does not reach the freezing
  level, or whose first level is missing or below 1 dB, has a depth of 0, so
  a ZDR column above a layer of lower ZDR is not a column of this algorithm
  (the vertical continuity requirement of the paper).
- The depth is NaN where no level at or above the freezing level has a ZDR
  value, and 0 where there is a value but no column.
- The paper has no condition on the reflectivity, and the columns need one on real
  data (radarx addition, off by default: ``min_dbz``). ZDR is noisy where the echo is
  weak, and a grid point of weak echo with :math:`Z_{DR} \\ge 1` dB is not an updraft.
  On the squall line of 30 March 2022 (PERiLS IOP2, KGWX, 250 m grid, freezing
  level 3.7 km from the 2211 UTC sounding) the volume of 22:11 UTC has columns of at least 1 km
  over 1383 km2 with a depth of up to 6.5 km without the condition, and over 141 km2 with a
  depth of up to 2.25 km if the grid points below 20 dBZ are left out, the
  size of the columns of the paper; 30 and 40 dBZ give 115 and 80 km2.
- The paper smooths the output with a Gaussian filter and does not give its
  width, so no smoothing is applied; filter the depth with
  :func:`scipy.ndimage.gaussian_filter` if you want it.
- The freezing level comes from the user, as it does in the paper from an
  analysis of a model (the hourly 13 km Rapid Refresh): a height in the vertical
  reference of the grid, from a sounding or a model. The columns are sensitive to
  it and to the calibration of :math:`Z_{DR}`: a negative bias shortens
  them and a positive one lengthens them (Snyder et al. 2015).

Fields to prepare first
-----------------------
Snyder et al. (2015) average ZDR over five range gates before gridding (three
for the reflectivity) and grid the WSR-88D volumes with 0.0025 degrees in latitude and
longitude and 250 m in height. Use a grid of a similar size
(:func:`radarx.grid.grid_cones`) and a calibrated :math:`Z_{DR}`; at C and X
band correct the differential attenuation first.
A volume gives a column only where its sweeps cover the levels above the
freezing level: the scanning strategy decides how deep a column can be seen
(Snyder et al. 2015, Figs. 8 and 9), and a column near the radar goes through the
top of the highest sweep.

KDP columns, a second signature of the updraft, are not included: the paper
shows them but gives no algorithm for them.

Implementation
--------------
A compiled, multithreaded C++ kernel scans the columns, with an equivalent NumPy
implementation as fallback and test oracle. The functions work on
:class:`xarray.DataArray` and :class:`xarray.Dataset` grids with extra leading
dimensions (time), keep dask arrays lazy and never change their input.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = ["zdr_column", "zdr_column_metrics"]

__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

import numpy as np
import xarray as xr

from .._provenance import provenance
from .._registry import accessor_method
from .._sweepwise import dask_support
from ._fieldtools import (
    DBZ_NAMES,
    RHOHV_NAMES,
    ZDR_NAMES,
    choose_engine,
    detach,
    find_field,
)

try:
    from . import _zdr_column

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _zdr_column = None
    HAS_COMPILED_KERNEL = False

_VERTICAL_NAMES = ("z", "altitude", "height", "level")
_HORIZONTAL_NAMES = ("y", "x")


# --------------------------------------------------------------------------
# the column scan: NumPy reference of the kernel
# --------------------------------------------------------------------------


def _depth_numpy(zdr, keep, z, thickness, level, threshold):
    """Depth of the ZDR column of the rows of ``zdr`` (n, nz); the kernel's twin."""
    with np.errstate(invalid="ignore"):
        below = z[None, :] < level[:, None]
        finite = np.isfinite(zdr)
        ok = finite & (zdr >= threshold) & (keep != 0.0)
        run = np.cumprod(ok | below, axis=1).astype(bool)
        depth = (thickness[None, :] * (run & ~below)).sum(axis=1)
        valid = (finite & ~below).any(axis=1) & ~np.isnan(level)
    return np.where(valid, depth, np.nan)


def _depth(zdr, keep, z, thickness, level, threshold, use_compiled, n_threads):
    """Depth of the columns of the (..., nz) blocks ``zdr`` and ``keep``."""
    nz = z.size
    lead = zdr.shape[:-1]
    flat = np.ascontiguousarray(zdr, dtype=np.float64).reshape(-1, nz)
    flat_keep = np.ascontiguousarray(
        np.broadcast_to(keep, zdr.shape), dtype=np.float64
    ).reshape(-1, nz)
    lv = np.ascontiguousarray(
        np.broadcast_to(np.asarray(level, dtype=np.float64), lead).reshape(-1)
    )
    if use_compiled:
        out = _zdr_column.depth(
            flat, flat_keep, z, thickness, lv, float(threshold), int(n_threads or 0)
        )
    else:
        out = _depth_numpy(flat, flat_keep, z, thickness, lv, float(threshold))
    return out.reshape(lead)


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def _vertical_name(da):
    for name in _VERTICAL_NAMES:
        if name in da.dims:
            return name
    raise ValueError(
        f"no vertical dimension ({'/'.join(_VERTICAL_NAMES)}) in {da.dims}: "
        "the ZDR column needs a grid"
    )


def _layer_thickness(z):
    """Thickness of each level: from halfway to the level below to halfway to the one above."""
    if z.size == 1:
        raise ValueError("the grid needs at least two levels")
    edges = np.r_[
        z[0] - 0.5 * (z[1] - z[0]),
        0.5 * (z[1:] + z[:-1]),
        z[-1] + 0.5 * (z[-1] - z[-2]),
    ]
    return np.diff(edges)


def _field(obj, spec, candidates, what, required=True):
    if isinstance(spec, xr.DataArray):
        return spec
    if isinstance(obj, xr.DataArray):
        if spec is not None:
            raise ValueError(f"pass the {what} as a DataArray when obj is a DataArray")
        return None if not required else obj
    name = find_field(obj, spec, candidates, what, required=required)
    return None if name is None else obj[name]


def _freezing_level(obj, level):
    if isinstance(level, str):
        if not isinstance(obj, xr.Dataset) or level not in obj:
            raise KeyError(f"{level!r} is not in the dataset")
        level = obj[level]
    if isinstance(level, xr.DataArray):
        if str(level.attrs.get("units", "m")).lower() in (
            "km",
            "kilometer",
            "kilometre",
        ):
            level = level * 1000.0
        return level
    try:
        return xr.DataArray(float(level))
    except (TypeError, ValueError):
        raise TypeError(
            "freezing_level must be a height in m, a DataArray or the name of a field"
        ) from None


# --------------------------------------------------------------------------
# the ZDR column
# --------------------------------------------------------------------------


@dask_support("lazy")
@provenance("ZDR column depth algorithm of Snyder et al. (2015)")
def zdr_column(
    obj,
    zdr=None,
    rhohv=None,
    dbz=None,
    *,
    freezing_level,
    threshold=1.0,
    min_rhohv=0.8,
    min_dbz=None,
    n_threads=None,
    engine="auto",
):
    """
    Depth of the ZDR column above the freezing level on a grid.

    At every horizontal grid point, the vertically continuous depth, from the
    freezing level up, in which the differential reflectivity is at least
    ``threshold`` (Snyder et al. 2015); see :mod:`radarx.retrieve.columns` for
    the algorithm and for what the paper leaves open.

    Parameters
    ----------
    obj : xarray.DataArray or xarray.Dataset
        The differential reflectivity (DataArray) or a dataset with it, on a grid
        with a vertical dimension (``z``, ``altitude``, ``height`` or ``level``,
        in m) and any horizontal and leading (time) dimensions.
    zdr : str or xarray.DataArray, optional
        Name of the differential reflectivity (dB) in ``obj`` or the field. Default:
        the first of ``ZDR``, ``differential_reflectivity`` and
        ``corrected_differential_reflectivity``.
    rhohv : str or xarray.DataArray, optional
        The copolar correlation coefficient, to remove the grid points below
        ``min_rhohv``. Default: no removal.
    dbz : str or xarray.DataArray, optional
        The reflectivity (dBZ), used with ``min_dbz``. Default: the first of
        ``DBZH``, ``DBZ``, ``reflectivity`` and ``corrected_reflectivity``.
    freezing_level : float, str or xarray.DataArray
        Height of the 0 degree Celsius level in m, in the vertical reference of the
        grid: a number, a field of ``obj`` (name) or a DataArray that
        broadcasts to the horizontal dimensions (a field in km is converted).
    threshold : float, optional
        Differential reflectivity (dB) that a grid point needs to belong to the
        column. Default 1, as Snyder et al. (2015).
    min_rhohv : float, optional
        Smallest correlation coefficient of a grid point that counts. Default 0.8
        ("approximately 0.80" in the paper); used only with ``rhohv``.
    min_dbz : float, optional
        Smallest reflectivity (dBZ) of a grid point that counts. Default ``None``:
        no condition, as in the paper; 20 dBZ keeps the noise of weak echo out of the
        columns (see :mod:`radarx.retrieve.columns`). A radarx addition.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled
        kernel and falls back to NumPy.

    Returns
    -------
    xarray.DataArray
        ``ZDR_COLUMN_DEPTH`` (m) on the horizontal dimensions of the grid: the
        depth of the column, 0 where there is none and NaN where no level at or
        above the freezing level has a ZDR value.

    Raises
    ------
    ValueError
        For an unknown ``engine``, a grid without a vertical dimension or with a
        single level.
    KeyError
        If a field is not found.

    References
    ----------
    Snyder, J. C., A. V. Ryzhkov, M. R. Kumjian, A. P. Khain, and J. Picca,
    2015: A ZDR column detection algorithm to examine convective storm
    updrafts. *Wea. Forecasting*, **30**, 1819-1844,
    https://doi.org/10.1175/WAF-D-15-0068.1

    Examples
    --------
    >>> depth = radarx.retrieve.zdr_column(grid, freezing_level=4200.0)  # doctest: +SKIP
    >>> depth = grid.radarx.zdr_column(rhohv="RHOHV", freezing_level="T0_HEIGHT")  # doctest: +SKIP
    """
    use_compiled = choose_engine(engine, HAS_COMPILED_KERNEL, "ZDR column")
    zda = _field(obj, zdr, ZDR_NAMES, "differential reflectivity")
    rho = (
        None
        if rhohv is None
        else _field(obj, rhohv, RHOHV_NAMES, "correlation coefficient")
    )
    level = _freezing_level(obj, freezing_level)
    vname = _vertical_name(zda)
    if not np.all(np.diff(zda[vname].values) > 0):
        zda = zda.sortby(vname)
        rho = None if rho is None else rho.sortby(vname)
    z = np.asarray(zda[vname].values, dtype=np.float64)
    thickness = _layer_thickness(z)
    keep = xr.ones_like(zda, dtype=np.float64)
    if rho is not None:
        keep = keep * xr.where(rho >= float(min_rhohv), 1.0, 0.0)
    if min_dbz is not None:
        reflectivity = _field(obj, dbz, DBZ_NAMES, "reflectivity")
        keep = keep * xr.where(reflectivity >= float(min_dbz), 1.0, 0.0)
        if vname not in reflectivity.dims:
            raise ValueError("the reflectivity needs the vertical dimension of the ZDR")

    def block(zdr_block, keep_block, level_block):
        return _depth(
            zdr_block,
            keep_block,
            z,
            thickness,
            level_block,
            threshold,
            use_compiled,
            n_threads,
        )

    arrays = detach(zda, keep, level)
    out = xr.apply_ufunc(
        block,
        *arrays,
        input_core_dims=[[vname], [vname], []],
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[np.float32],
        keep_attrs=False,
    )
    out = out.astype(np.float32)
    out.name = "ZDR_COLUMN_DEPTH"
    out.attrs = {
        "long_name": "Depth of the ZDR column above the freezing level",
        "units": "m",
        "comment": (
            f"vertically continuous depth with ZDR >= {threshold:g} dB from the first "
            "level at or above the freezing level"
            + (f", correlation coefficient >= {min_rhohv:g}" if rho is not None else "")
            + (f", reflectivity >= {min_dbz:g} dBZ" if min_dbz is not None else "")
            + "; 0 where there is no column, NaN where no level above the freezing "
            "level has a ZDR value"
        ),
    }
    return out


# --------------------------------------------------------------------------
# size of the columns
# --------------------------------------------------------------------------


def _horizontal_dims(depth):
    named = [d for d in _HORIZONTAL_NAMES if d in depth.dims]
    if len(named) == 2:
        return named
    if depth.ndim < 2:
        raise ValueError("the depth needs two horizontal dimensions")
    return list(depth.dims[-2:])


def _cell_area(depth, dims, cell_area):
    if cell_area is not None:
        return cell_area
    sizes = []
    for d in dims:
        if d not in depth.coords or depth[d].size < 2:
            raise ValueError(
                f"no coordinate {d!r} to take the grid spacing from: pass cell_area (m2)"
            )
        sizes.append(float(np.median(np.abs(np.diff(depth[d].values)))))
    return sizes[0] * sizes[1]


@dask_support("lazy")
@provenance("ZDR column size metrics: maximum depth, area and volume")
def zdr_column_metrics(obj, *, min_depth=1000.0, cell_area=None):
    """
    Maximum depth, area and volume of the ZDR columns of a map of depths.

    Parameters
    ----------
    obj : xarray.DataArray or xarray.Dataset
        The depth of the ZDR column (m), ``ZDR_COLUMN_DEPTH`` of
        :func:`zdr_column`, on the horizontal dimensions ``y`` and ``x``
        (or the last two) and any leading (time) dimensions.
    min_depth : float, optional
        Depth (m) from which a grid column counts as a ZDR column for the area and the
        volume. Default 1000, a radarx choice: Snyder et al. (2015, Fig. 17)
        contour the depth at 1, 2 and 3 km, and a single 250 m level is as
        likely to be noise as a column.
    cell_area : float or xarray.DataArray, optional
        Area of a grid cell in m2. Default: the product of the median spacings of
        the ``y`` and ``x`` coordinates.

    Returns
    -------
    xarray.Dataset
        ``ZDR_COLUMN_MAX_DEPTH`` (m, of the whole map), ``ZDR_COLUMN_AREA`` (km2) of
        the grid columns with at least ``min_depth`` and ``ZDR_COLUMN_VOLUME`` (km3),
        the sum of their depths times the cell area, per leading index.

    Raises
    ------
    ValueError
        If the grid spacing cannot be found and ``cell_area`` is not given.

    References
    ----------
    Snyder, J. C., A. V. Ryzhkov, M. R. Kumjian, A. P. Khain, and J. Picca,
    2015: A ZDR column detection algorithm to examine convective storm
    updrafts. *Wea. Forecasting*, **30**, 1819-1844,
    https://doi.org/10.1175/WAF-D-15-0068.1

    Examples
    --------
    >>> metrics = radarx.retrieve.zdr_column_metrics(depth, min_depth=2000.0)  # doctest: +SKIP
    """
    if isinstance(obj, xr.Dataset):
        obj = obj["ZDR_COLUMN_DEPTH"]
    dims = _horizontal_dims(obj)
    area = _cell_area(obj, dims, cell_area)
    column = obj >= float(min_depth)
    out = xr.Dataset(
        {
            "ZDR_COLUMN_MAX_DEPTH": obj.max(dims, skipna=True),
            "ZDR_COLUMN_AREA": column.sum(dims) * area / 1e6,
            "ZDR_COLUMN_VOLUME": obj.where(column).sum(dims) * area / 1e9,
        }
    )
    out["ZDR_COLUMN_MAX_DEPTH"].attrs = {
        "long_name": "Maximum depth of the ZDR column",
        "units": "m",
    }
    out["ZDR_COLUMN_AREA"].attrs = {
        "long_name": f"Area of the ZDR columns deeper than {min_depth:g} m",
        "units": "km2",
    }
    out["ZDR_COLUMN_VOLUME"].attrs = {
        "long_name": f"Volume of the ZDR columns deeper than {min_depth:g} m",
        "units": "km3",
    }
    return out


# --------------------------------------------------------------------------
# accessors
# --------------------------------------------------------------------------


@accessor_method("dataarray", "dataset", name="zdr_column", wraps=zdr_column)
def _zdr_column_accessor(self):
    """
    Depth of the ZDR column above the freezing level on a grid.

    See :func:`radarx.retrieve.zdr_column` for the parameters.

    Returns
    -------
    xarray.DataArray
        ``ZDR_COLUMN_DEPTH`` (m).
    """


@accessor_method(
    "dataarray", "dataset", name="zdr_column_metrics", wraps=zdr_column_metrics
)
def _zdr_column_metrics_accessor(self):
    """
    Maximum depth, area and volume of the ZDR columns of a map of depths.

    See :func:`radarx.retrieve.zdr_column_metrics` for the parameters.

    Returns
    -------
    xarray.Dataset
        ``ZDR_COLUMN_MAX_DEPTH``, ``ZDR_COLUMN_AREA`` and ``ZDR_COLUMN_VOLUME``.
    """
