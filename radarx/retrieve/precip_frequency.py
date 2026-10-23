# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Rainfall against precipitation frequency and climatology
========================================================

Tools to put a rainfall depth in context: how rare is it, and how unusual is
it for the place?

* :func:`return_period`: the return period of a depth from precipitation
  frequency estimates, the depth for a few return periods (for example the
  2, 10 and 100 year depths of NOAA Atlas 14), interpolated in the logarithm
  of both. A depth outside the table is flagged, not extrapolated.
* :func:`areal_reduction_factor`: the factor that turns a point depth into
  the mean depth over an area, interpolated in the logarithm of the area from a
  table that the user supplies. radarx ships no factors: published curves
  depend on the region, the duration and the return period.
* :func:`climatological_reference`: the quantile of a record of rainfall
  depths at each place, for example the 99.5th percentile of the one-hour
  totals, to compare a depth with ``depth / reference``.

None of these is a flood forecast. A return period of a point depth says how
often that depth is reached at a point; it knows nothing of soil, terrain or
drainage.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = ["return_period", "areal_reduction_factor", "climatological_reference"]

__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

import numpy as np
import xarray as xr

from .._dtypes import FIELD_DTYPE
from .._provenance import provenance
from .._sweepwise import dask_support


@dask_support("computes", "it looks the depth up in the table of every cell together.")
@provenance("Return period from precipitation frequency estimates")
def return_period(depth, periods, depths, *, period_dim="period"):
    """
    Return period of a rainfall depth, from depth-for-return-period estimates.

    For each cell the depths of the table at the return periods
    ``periods`` give a curve; the return period of ``depth`` is read from it by
    linear interpolation of ``ln(period)`` against ``ln(depth)``. A depth below
    the depth of the shortest period gets that period and the flag ``-1``, a
    depth above the depth of the longest period gets that period and the flag
    ``+1``; nothing is extrapolated. A cell is NaN where the depth or any of its
    table depths is missing or not positive, or where the table depths fall
    with the return period.

    Parameters
    ----------
    depth : xarray.DataArray
        Rainfall depth, in the units of ``depths`` (for example mm).
    periods : array_like
        Return periods in years, increasing, at least two.
    depths : xarray.DataArray or array_like
        Depth for each return period: a DataArray with the dimension
        ``period_dim`` (length ``len(periods)``) and the dimensions of
        ``depth``, or a 1-D array-like for one table that is used everywhere.
    period_dim : str, optional
        Name of the return-period dimension of ``depths``. Default
        ``"period"``.

    Returns
    -------
    xarray.Dataset
        ``return_period`` (years, ``float32``) and ``flag`` (``int8``: -1
        below the table, 0 inside, +1 above) on the dimensions of ``depth``.

    Raises
    ------
    ValueError
        If ``periods`` are not increasing, positive and at least two, or
        ``depths`` does not match them.

    See Also
    --------
    areal_reduction_factor, climatological_reference

    References
    ----------
    .. [1] Chow, V. T., D. R. Maidment, and L. W. Mays, 1988: *Applied
       Hydrology*. McGraw-Hill, chapter 12 (frequency analysis).

    Examples
    --------
    >>> rp = radarx.retrieve.return_period(
    ...     rain, [2, 10, 100], [40.0, 80.0, 160.0]
    ... )  # doctest: +SKIP
    """
    per = np.asarray(periods, dtype=np.float64)
    if per.ndim != 1 or per.size < 2 or np.any(per <= 0) or np.any(np.diff(per) <= 0):
        raise ValueError(
            "periods must be at least two increasing positive numbers (years)"
        )
    if isinstance(depths, xr.DataArray) and period_dim in depths.dims:
        table = depths.transpose(period_dim, ...)
        if table.sizes[period_dim] != per.size:
            raise ValueError("depths needs one value per return period")
        stack = table.broadcast_like(depth).transpose(period_dim, *depth.dims).values
    else:
        flat = np.asarray(depths, dtype=np.float64)
        if flat.shape != per.shape:
            raise ValueError("a 1-D table needs one depth per return period")
        stack = np.broadcast_to(
            flat.reshape((-1,) + (1,) * depth.ndim), (per.size,) + depth.shape
        )
    d = np.asarray(depth.values, dtype=np.float64)
    stack = np.asarray(stack, dtype=np.float64)
    ok = np.isfinite(d) & (d > 0) & np.all(np.isfinite(stack) & (stack > 0), axis=0)
    ok &= np.all(np.diff(stack, axis=0) >= 0, axis=0)
    ln_d = np.log(np.where(ok, d, 1.0))
    ln_s = np.log(np.where(ok[None], stack, 1.0))
    lnp = np.log(per)
    below = ok & (ln_d < ln_s[0])
    above = ok & (ln_d > ln_s[-1])
    inside = ok & ~below & ~above
    k = np.zeros(d.shape, dtype=int)
    for j in range(1, per.size - 1):
        k = np.where(ln_d >= ln_s[j], j, k)
    lo = np.take_along_axis(ln_s, k[None], 0)[0]
    hi = np.take_along_axis(ln_s, k[None] + 1, 0)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(hi > lo, (ln_d - lo) / (hi - lo), 0.0)
    years = np.full(d.shape, np.nan)
    years[below], years[above] = per[0], per[-1]
    years[inside] = np.exp(lnp[k] + t * (lnp[k + 1] - lnp[k]))[inside]
    flag = np.where(below, -1, np.where(above, 1, 0)).astype(np.int8)
    coords = depth.coords
    return xr.Dataset(
        {
            "return_period": (
                depth.dims,
                years.astype(FIELD_DTYPE),
                {"units": "year", "long_name": "return period of the depth"},
            ),
            "flag": (
                depth.dims,
                flag,
                {
                    "long_name": "-1 below the table, 0 inside, +1 above (not extrapolated)",
                    "units": "1",
                },
            ),
        },
        coords=coords,
    )


@provenance("Areal reduction factor from a table of the user")
def areal_reduction_factor(area, areas, factors):
    """
    Areal reduction factor of an area, from a table supplied by the user.

    The mean depth over an area is smaller than the point depth of the same
    return period; the factor (between 0 and 1) is the ratio. It is
    interpolated linearly in ``ln(area)`` between the table values and held at
    the first and last value outside the table (not extrapolated). A single
    value is used for every area. radarx contains no table: published curves
    depend on the region, the duration and the return period, so the user
    supplies, and cites, the one that fits.

    To find the return period of an *area mean* depth, divide it by the factor
    first, which gives the point depth that has the same areal mean, and look
    that up with :func:`return_period`. Without a factor the return period of
    an area mean found directly from a point table is too short (a lower bound).

    Parameters
    ----------
    area : float or array_like
        Area (any unit, the one of ``areas``).
    areas : float or array_like
        Areas of the table, positive, increasing.
    factors : float or array_like
        Factors of the table, each in (0, 1], same length as ``areas``.

    Returns
    -------
    numpy.ndarray or float
        The factor at ``area``.

    Raises
    ------
    ValueError
        If the table is not positive and increasing, or a factor is outside
        (0, 1].

    See Also
    --------
    return_period

    Examples
    --------
    >>> radarx.retrieve.areal_reduction_factor(
    ...     100.0, [10, 1000], [1.0, 0.8]
    ... )  # doctest: +SKIP
    """
    a = np.atleast_1d(np.asarray(areas, dtype=np.float64))
    f = np.atleast_1d(np.asarray(factors, dtype=np.float64))
    if a.shape != f.shape or a.ndim != 1:
        raise ValueError("areas and factors must be 1-D with the same length")
    if np.any(a <= 0) or np.any(np.diff(a) <= 0):
        raise ValueError("areas must be positive and increasing")
    if np.any(~np.isfinite(f)) or np.any(f <= 0) or np.any(f > 1):
        raise ValueError("areal reduction factors must be in (0, 1]")
    x = np.asarray(area, dtype=np.float64)
    if a.size == 1:
        out = np.full(x.shape, f[0])
    else:
        out = np.interp(np.log(np.maximum(x, 1e-300)), np.log(a), f)
    return float(out) if np.ndim(out) == 0 else out


@dask_support("computes", "it takes a quantile over the whole record.")
@provenance("Climatological reference depth from a record")
def climatological_reference(record, *, quantile=0.995, dim="time", min_count=30):
    """
    Quantile of a record of rainfall depths at each place.

    The reference for ``depth / reference``: with the 99.5th percentile of the
    one-hour totals of a record, a ratio of 1 means the hour was as wet as
    the wettest half percent of the hours at that place. Places with fewer than
    ``min_count`` valid values get NaN: the quantile of a handful of hours says
    nothing about a rare quantile.

    Parameters
    ----------
    record : xarray.DataArray
        Depths (mm) with the dimension ``dim``.
    quantile : float, optional
        Quantile, 0.5 to below 1. Default 0.995.
    dim : str, optional
        Dimension of the record. Default ``"time"``.
    min_count : int, optional
        Fewest valid values for a result. Default 30.

    Returns
    -------
    xarray.DataArray
        The quantile on the other dimensions, ``float32``, in the units of
        ``record``.

    Raises
    ------
    ValueError
        If ``quantile`` is outside 0.5 to 1 or ``dim`` is not a dimension.

    See Also
    --------
    return_period

    Examples
    --------
    >>> ref = radarx.retrieve.climatological_reference(hourly, dim="time")  # doctest: +SKIP
    >>> relative = rain / ref  # doctest: +SKIP
    """
    if not 0.5 <= quantile < 1.0:
        raise ValueError("quantile must be from 0.5 up to (not including) 1")
    if dim not in record.dims:
        raise ValueError(f"the record has no dimension {dim!r}")
    q = record.quantile(quantile, dim=dim, skipna=True).drop_vars("quantile")
    n = record.notnull().sum(dim)
    out = q.where(n >= min_count).astype(FIELD_DTYPE)
    out.name = "climatological_reference"
    out.attrs = {
        "units": record.attrs.get("units", ""),
        "long_name": f"{quantile:g} quantile of the record",
        "quantile": float(quantile),
    }
    return out
