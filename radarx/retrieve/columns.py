#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx ZDR and KDP Columns
==========================

The depth of the ZDR column of a convective storm and its size, and the KDP
columns above the freezing level, from a polarimetric radar volume on a grid.

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

KDP columns
-----------
Rain and liquid-coated hydrometeors that an updraft carries above the freezing level
give positive specific differential phase there, so that a KDP column is a second
signature of the updraft (van Lier-Walqui et al. 2016). Snyder et al. (2015) show it
but give no algorithm; :func:`kdp_column` follows the one of van Lier-Walqui et al.
(2016, section 2e): KDP is integrated in a slab above the freezing level, the integral
is smoothed with a Gaussian filter, the regional maxima of the smoothed integral are
the cores of the columns, and a watershed segmentation gives each column its extent.
Within a column the highest level with
:math:`K_{DP} > 0.75` degrees km-1 (S band; 1.5 at C band, in inverse proportion to
the wavelength) is the top of the column, and the KDP volume is the volume with KDP
above the threshold in the slab.

What the paper leaves open, and what radarx does (:func:`kdp_column`):

- The slab is 2 km deep (the paper has 2 km in one place and 3 km in another for the
  volume); ``slab_depth`` changes it. It starts at the first level at or above the
  freezing level.
- The integral adds the positive KDP values only, in degrees: the algorithm of the
  paper assumes no negative KDP in the rain above the freezing level, and
  negative values are noise.
- The Gaussian width of the smoothing is not given (it is 1 grid cell here) and
  neither is the extent of a column: a grid column belongs to a KDP column where the
  smoothed integral reaches that of a layer of ``min_depth`` (1000 m) at the
  threshold. A layer of 500 m, one level of a 500 m grid, lets single noisy levels
  make columns: on the squall line of 30 March 2022 (KGWX, 1 km grid) the volume of
  23:19 UTC has 154 columns with 500 m and 27 with 1000 m, over 4700 and 1350 km2.
- The top is the height of the highest level at or above the freezing level with KDP
  at or above the threshold anywhere in the column of the grid, not only in the slab,
  which is the maximum height of the 0.75 degrees km-1 contour of the paper.
- ZDR columns in the paper are measured with a different condition (a negative vertical
  gradient of ZDR, below the homogeneous freezing level); the continuous depth of
  :func:`zdr_column` is used for them here.

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

__all__ = ["kdp_column", "kdp_column_metrics", "zdr_column", "zdr_column_metrics"]

__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

import numpy as np
import xarray as xr

from .._provenance import provenance
from .._registry import accessor_method
from .._sweepwise import dask_support
from . import _cells
from ._fieldtools import (
    DBZ_NAMES,
    KDP_NAMES,
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
# KDP columns
# --------------------------------------------------------------------------


def _slab_numpy(kdp, keep, z, thickness, level, threshold, slab_depth):
    """Integral, thickness and top of the KDP slab of the rows (n, nz); the kernel's twin."""
    with np.errstate(invalid="ignore"):
        above = z[None, :] >= level[:, None]
        counts = np.isfinite(kdp) & (keep != 0.0) & above
        in_slab = counts & (z[None, :] < level[:, None] + slab_depth)
        high = counts & (kdp >= threshold)
        integral = (np.where(in_slab & (kdp > 0.0), kdp, 0.0) * thickness).sum(
            axis=1
        ) / 1000.0
        depth = (thickness[None, :] * (in_slab & high)).sum(axis=1)
        valid = in_slab.any(axis=1) & ~np.isnan(level)
        top = np.where(high, z[None, :], -np.inf).max(axis=1)
    top = np.where(np.isfinite(top) & ~np.isnan(level), top, np.nan)
    return (
        np.where(valid, integral, np.nan),
        np.where(valid, depth, np.nan),
        top,
    )


def _slab(
    kdp, keep, z, thickness, level, threshold, slab_depth, use_compiled, n_threads
):
    """Integral, thickness and top for the (..., nz) blocks ``kdp`` and ``keep``."""
    nz = z.size
    lead = kdp.shape[:-1]
    flat = np.ascontiguousarray(kdp, dtype=np.float64).reshape(-1, nz)
    flat_keep = np.ascontiguousarray(
        np.broadcast_to(keep, kdp.shape), dtype=np.float64
    ).reshape(-1, nz)
    lv = np.ascontiguousarray(
        np.broadcast_to(np.asarray(level, dtype=np.float64), lead).reshape(-1)
    )
    args = (flat, flat_keep, z, thickness, lv, float(threshold), float(slab_depth))
    if use_compiled:
        out = _zdr_column.kdp_slab(*args, int(n_threads or 0))
    else:
        out = _slab_numpy(*args)
    return tuple(o.reshape(lead) for o in out)


def _column_objects(integral, floor, sigma, use_compiled, n_threads):
    """Labels of the KDP columns in the (..., ny, nx) blocks of the slab integral."""
    from scipy import ndimage

    ny, nx = integral.shape[-2:]
    lead = integral.shape[:-2]
    smooth = np.nan_to_num(np.asarray(integral, dtype=np.float64), nan=0.0).reshape(
        -1, ny, nx
    )
    if sigma:
        smooth = _cells.smooth(smooth, sigma, use_compiled, n_threads)
    local_max = ndimage.maximum_filter(
        smooth, size=(1, 3, 3), mode="constant", cval=-np.inf
    )
    mask = smooth >= floor
    peaks = mask & (smooth >= local_max) & (smooth > 0.0)
    markers, _ = _cells.label(peaks, 8, use_compiled, n_threads)
    labels = _cells.watershed(smooth, markers, mask, 8, use_compiled, n_threads)
    return labels.reshape(lead + (ny, nx))


@dask_support("lazy")
@provenance("KDP columns above the freezing level after van Lier-Walqui et al. (2016)")
def kdp_column(
    obj,
    kdp=None,
    rhohv=None,
    dbz=None,
    *,
    freezing_level,
    threshold=0.75,
    slab_depth=2000.0,
    min_rhohv=0.8,
    min_dbz=None,
    smooth=1.0,
    min_depth=1000.0,
    n_threads=None,
    engine="auto",
):
    """
    KDP columns above the freezing level on a grid.

    KDP is integrated in a slab above the freezing level and smoothed, the maxima of
    the integral are the cores of the columns and a watershed segmentation gives their
    extent (van Lier-Walqui et al. 2016); see :mod:`radarx.retrieve.columns` for the
    algorithm and for what the paper leaves open.

    Parameters
    ----------
    obj : xarray.DataArray or xarray.Dataset
        The specific differential phase (DataArray) or a dataset with it, on a grid
        with a vertical dimension (``z``, ``altitude``, ``height`` or ``level``, in m),
        the horizontal dimensions ``y`` and ``x`` and any leading (time) dimensions.
    kdp : str or xarray.DataArray, optional
        Name of the specific differential phase (degrees km-1) in ``obj`` or the
        field. Default: the first of ``KDP`` and ``specific_differential_phase``.
    rhohv : str or xarray.DataArray, optional
        The copolar correlation coefficient, to remove the grid points below
        ``min_rhohv``. Default: no removal.
    dbz : str or xarray.DataArray, optional
        The reflectivity (dBZ), used with ``min_dbz``. Default: the first of ``DBZH``,
        ``DBZ``, ``reflectivity`` and ``corrected_reflectivity``.
    freezing_level : float, str or xarray.DataArray
        Height of the 0 degree Celsius level in m, in the vertical reference of the
        grid: a number, a field of ``obj`` (name) or a DataArray that broadcasts to the
        horizontal dimensions (a field in km is converted).
    threshold : float, optional
        Specific differential phase (degrees km-1) of a level that counts for the top
        and the thickness of a column. Default 0.75, the S-band value of the paper; use
        1.5 at C band.
    slab_depth : float, optional
        Depth (m) of the slab above the freezing level. Default 2000.
    min_rhohv : float, optional
        Smallest correlation coefficient of a grid point that counts. Default 0.8; used
        only with ``rhohv``.
    min_dbz : float, optional
        Smallest reflectivity (dBZ) of a grid point that counts. Default ``None``.
    smooth : float, optional
        Width (standard deviation, grid cells) of the Gaussian filter of the slab
        integral. Default 1; 0 or ``None`` for none.
    min_depth : float, optional
        A grid column belongs to a KDP column where the smoothed integral is at least
        that of a layer of this depth (m) with KDP equal to ``threshold``. Default 1000.
    n_threads : int, optional
        Threads for the compiled kernels. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled kernels and
        falls back to NumPy.

    Returns
    -------
    xarray.Dataset
        On the horizontal grid: ``KDP_SLAB_INTEGRAL`` (degree, the integral of the
        positive KDP in the slab), ``KDP_SLAB_DEPTH`` (m, the thickness of the levels
        of the slab with KDP at or above ``threshold``), ``KDP_COLUMN_TOP`` (m, the
        height of the highest such level above the freezing level, NaN where there is
        none) and ``KDP_COLUMN_OBJECT`` (the number of the KDP column of the grid
        point, 0 for none; the numbers run from 1 in every volume).

    Raises
    ------
    ValueError
        For an unknown ``engine``, a grid without a vertical dimension or with a single
        level, or a field without the horizontal dimensions ``y`` and ``x``.
    KeyError
        If a field is not found.

    See Also
    --------
    kdp_column_metrics, zdr_column

    References
    ----------
    van Lier-Walqui, M., A. M. Fridlind, A. S. Ackerman, S. Collis, J. Helmus,
    D. R. MacGorman, K. North, P. Kollias, and D. J. Posselt, 2016: On polarimetric radar
    signatures of deep convection for model evaluation: Columns of specific differential
    phase observed during MC3E. *Mon. Wea. Rev.*, **144**, 737-758,
    https://doi.org/10.1175/MWR-D-15-0100.1

    Examples
    --------
    >>> cols = radarx.retrieve.kdp_column(grid, freezing_level=4200.0)  # doctest: +SKIP
    >>> cols.KDP_COLUMN_TOP  # doctest: +SKIP
    """
    use_compiled = choose_engine(engine, HAS_COMPILED_KERNEL, "KDP column")
    use_objects = choose_engine(engine, _cells.HAS_COMPILED_KERNEL, "object")
    kda = _field(obj, kdp, KDP_NAMES, "specific differential phase")
    rho = (
        None
        if rhohv is None
        else _field(obj, rhohv, RHOHV_NAMES, "correlation coefficient")
    )
    level = _freezing_level(obj, freezing_level)
    vname = _vertical_name(kda)
    for d in ("y", "x"):
        if d not in kda.dims:
            raise ValueError(
                f"the grid needs the horizontal dimensions y and x, not {kda.dims}"
            )
    if not np.all(np.diff(kda[vname].values) > 0):
        kda = kda.sortby(vname)
        rho = None if rho is None else rho.sortby(vname)
    z = np.asarray(kda[vname].values, dtype=np.float64)
    thickness = _layer_thickness(z)
    keep = xr.ones_like(kda, dtype=np.float64)
    if rho is not None:
        keep = keep * xr.where(rho >= float(min_rhohv), 1.0, 0.0)
    if min_dbz is not None:
        reflectivity = _field(obj, dbz, DBZ_NAMES, "reflectivity")
        if vname not in reflectivity.dims:
            raise ValueError("the reflectivity needs the vertical dimension of the KDP")
        keep = keep * xr.where(reflectivity >= float(min_dbz), 1.0, 0.0)

    def block(kdp_block, keep_block, level_block):
        return _slab(
            kdp_block,
            keep_block,
            z,
            thickness,
            level_block,
            threshold,
            slab_depth,
            use_compiled,
            n_threads,
        )

    arrays = detach(kda, keep, level)
    integral, depth, top = xr.apply_ufunc(
        block,
        *arrays,
        input_core_dims=[[vname], [vname], []],
        output_core_dims=[[], [], []],
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[np.float32] * 3,
        keep_attrs=False,
    )
    floor = float(threshold) * float(min_depth) / 1000.0

    def objects(plane):
        return _column_objects(plane, floor, smooth, use_objects, n_threads)

    labels = xr.apply_ufunc(
        objects,
        detach(integral)[0],
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[np.int32],
        keep_attrs=False,
    )
    out = xr.Dataset(
        {
            "KDP_SLAB_INTEGRAL": integral.astype(np.float32),
            "KDP_SLAB_DEPTH": depth.astype(np.float32),
            "KDP_COLUMN_TOP": top.astype(np.float32),
            "KDP_COLUMN_OBJECT": labels.astype(np.int32),
        }
    )
    out["KDP_SLAB_INTEGRAL"].attrs = {
        "long_name": f"Integral of the positive KDP in the {slab_depth:g} m above the freezing level",
        "units": "degree",
    }
    out["KDP_SLAB_DEPTH"].attrs = {
        "long_name": f"Thickness of the levels with KDP >= {threshold:g} degrees km-1 in the slab above the freezing level",
        "units": "m",
    }
    out["KDP_COLUMN_TOP"].attrs = {
        "long_name": f"Height of the highest level with KDP >= {threshold:g} degrees km-1 above the freezing level",
        "units": "m",
    }
    out["KDP_COLUMN_OBJECT"].attrs = {
        "long_name": "Number of the KDP column (watershed of the smoothed slab integral)",
        "comment": (
            f"integral smoothed with a Gaussian of {smooth or 0:g} cells; columns where it "
            f"is at least {floor:g} degree"
        ),
    }
    return out


@dask_support("lazy")
@provenance(
    "KDP column count, area, volume and top after van Lier-Walqui et al. (2016)"
)
def kdp_column_metrics(obj, *, cell_area=None):
    """
    Number, area, volume and greatest top of the KDP columns of a map.

    Parameters
    ----------
    obj : xarray.Dataset
        The result of :func:`kdp_column` (``KDP_COLUMN_OBJECT``, ``KDP_COLUMN_TOP`` and
        ``KDP_SLAB_DEPTH``), on the horizontal dimensions ``y`` and ``x`` (or the last
        two) and any leading (time) dimensions.
    cell_area : float or xarray.DataArray, optional
        Area of a grid cell in m2. Default: the product of the median spacings of the
        ``y`` and ``x`` coordinates.

    Returns
    -------
    xarray.Dataset
        Per leading index: ``KDP_COLUMN_COUNT`` (number of KDP columns),
        ``KDP_COLUMN_AREA`` (km2, of the grid points in a column), ``KDP_COLUMN_MAX_TOP``
        (m, the highest top of the columns) and ``KDP_VOLUME`` (km3, the volume with KDP at
        or above the threshold in the slab, over the whole map).

    Raises
    ------
    ValueError
        If the grid spacing cannot be found and ``cell_area`` is not given.

    See Also
    --------
    kdp_column

    References
    ----------
    van Lier-Walqui, M., A. M. Fridlind, A. S. Ackerman, S. Collis, J. Helmus,
    D. R. MacGorman, K. North, P. Kollias, and D. J. Posselt, 2016: On polarimetric radar
    signatures of deep convection for model evaluation: Columns of specific differential
    phase observed during MC3E. *Mon. Wea. Rev.*, **144**, 737-758,
    https://doi.org/10.1175/MWR-D-15-0100.1

    Examples
    --------
    >>> metrics = radarx.retrieve.kdp_column_metrics(cols)  # doctest: +SKIP
    """
    labels = obj["KDP_COLUMN_OBJECT"]
    dims = _horizontal_dims(labels)
    area = _cell_area(labels, dims, cell_area)
    inside = labels > 0
    out = xr.Dataset(
        {
            "KDP_COLUMN_COUNT": labels.max(dims).astype(np.int32),
            "KDP_COLUMN_AREA": inside.sum(dims) * area / 1e6,
            "KDP_COLUMN_MAX_TOP": obj["KDP_COLUMN_TOP"]
            .where(inside)
            .max(dims, skipna=True),
            "KDP_VOLUME": obj["KDP_SLAB_DEPTH"].sum(dims, skipna=True) * area / 1e9,
        }
    )
    out["KDP_COLUMN_COUNT"].attrs = {"long_name": "Number of KDP columns"}
    out["KDP_COLUMN_AREA"].attrs = {
        "long_name": "Area of the KDP columns",
        "units": "km2",
    }
    out["KDP_COLUMN_MAX_TOP"].attrs = {
        "long_name": "Greatest top of the KDP columns",
        "units": "m",
    }
    out["KDP_VOLUME"].attrs = {
        "long_name": "Volume with KDP at or above the threshold in the slab above the freezing level",
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


@accessor_method("dataarray", "dataset", name="kdp_column", wraps=kdp_column)
def _kdp_column_accessor(self):
    """
    KDP columns above the freezing level on a grid.

    See :func:`radarx.retrieve.kdp_column` for the parameters.

    Returns
    -------
    xarray.Dataset
        ``KDP_SLAB_INTEGRAL``, ``KDP_SLAB_DEPTH``, ``KDP_COLUMN_TOP`` and
        ``KDP_COLUMN_OBJECT``.
    """


@accessor_method("dataset", name="kdp_column_metrics", wraps=kdp_column_metrics)
def _kdp_column_metrics_accessor(self):
    """
    Number, area, volume and greatest top of the KDP columns of a map.

    See :func:`radarx.retrieve.kdp_column_metrics` for the parameters.

    Returns
    -------
    xarray.Dataset
        ``KDP_COLUMN_COUNT``, ``KDP_COLUMN_AREA``, ``KDP_COLUMN_MAX_TOP`` and
        ``KDP_VOLUME``.
    """
