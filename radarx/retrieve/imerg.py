#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx Comparison with IMERG
============================

Puts the rain rate of a ground radar and the Integrated Multi-satellitE Retrievals for
GPM (IMERG; Huffman et al. 2020) on the same grid and time steps, for the comparison of
the two, with :func:`radarx.retrieve.match_statistics`.

IMERG is a gridded product of 0.1 degrees every 30 minutes that merges the passive
microwave estimates of the GPM constellation, the infrared estimates of geostationary
satellites and gauge analyses. A radar scan is a snapshot every few minutes on a grid of a
few hundred metres, so the radar rate is averaged

1. in time over the IMERG step, weighting every scan by the time that it represents
   (the interval between the midpoints to its neighbours, cut at the step), and
2. in space over the cells of the radar grid whose centres lie inside an IMERG cell.

Choices of radarx
-----------------
- ``time_label`` says what the time of an IMERG value is: the start of the half hour
  (``"start"``, as in the IMERG V07 files), the end or the centre.
- A radar pixel needs scans for at least ``min_time_coverage`` (default 0.8) of the
  step; a cell needs radar pixels for at least ``min_coverage`` (default 0.5) of the
  pixels that fit into it. Missing pixels (NaN) are outside the coverage of the radar:
  fill the rain-free pixels inside the coverage with 0 before the call.
- The radar pixels are binned to the cell that holds their centre; the grid of IMERG
  is regular in longitude and latitude.
- Values below zero of IMERG (the fill value of the files is -9999.9) are missing.
  The calibrated precipitation (``precipitation``, mm/h) is the usual field.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = ["match_imerg"]

__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

import re

import numpy as np
import xarray as xr

from .._provenance import provenance
from .._registry import accessor_method
from .._sweepwise import dask_support

EARTH_RADIUS = 6371000.0
_LON = ("lon", "longitude")
_LAT = ("lat", "latitude")
_OUTPUTS = ("rate", "accumulation")
_LABELS = ("start", "end", "center")


_UNITS = {"s": 1.0, "sec": 1.0, "min": 60.0, "m": 60.0, "h": 3600.0, "hr": 3600.0}


def _seconds(interval):
    """Length of the step in seconds from a number, a timedelta or a string like "30min"."""
    if isinstance(interval, np.timedelta64):
        return float(interval / np.timedelta64(1, "s"))
    if isinstance(interval, (int, float)):
        return float(interval)
    match = re.fullmatch(r"\s*([0-9.]+)\s*([a-z]+)\s*", str(interval))
    if match is None or match.group(2) not in _UNITS:
        raise ValueError(
            f"interval must be seconds, a timedelta or a string like '30min', not {interval!r}"
        )
    return float(match.group(1)) * _UNITS[match.group(2)]


def _find_dim(da, names, what):
    for name in names:
        if name in da.dims:
            return name
    raise ValueError(f"cannot find the {what} dimension of {da.name!r} among {da.dims}")


def _radar_lonlat(radar):
    """2-D longitude and latitude of the pixels, their dimensions and area (m2)."""
    from .._gridcf import lonlat

    for lon_name, lat_name in zip(_LON, _LAT):
        if lon_name in radar.coords and lat_name in radar.coords:
            lon, lat = radar[lon_name], radar[lat_name]
            if lon.ndim == 2 and lat.ndim == 2:
                dims = lon.dims
                break
            if lon.ndim == 1 and lat.ndim == 1 and lon.dims != lat.dims:
                lat2, lon2 = xr.broadcast(lat, lon)
                lon, lat, dims = lon2, lat2, lat2.dims
                break
    else:
        if not {"x", "y"} <= set(radar.dims):
            raise ValueError(
                "radar needs 2-D 'lon' and 'lat' coordinates, or x and y with a "
                "'crs_wkt' grid mapping"
            )
        crs = radar.coords.get("crs_wkt")
        if crs is None or "crs_wkt" not in crs.attrs:
            raise ValueError("radar has x and y but no 'crs_wkt' grid mapping")
        lon, lat = lonlat(crs.attrs, radar["x"].values, radar["y"].values)
        lon = xr.DataArray(lon, dims=("y", "x"))
        lat = xr.DataArray(lat, dims=("y", "x"))
        dims = ("y", "x")
    lon_v, lat_v = np.asarray(lon.transpose(*dims).values), np.asarray(
        lat.transpose(*dims).values
    )
    if {"x", "y"} <= set(radar.dims) and "x" in radar.coords and "y" in radar.coords:
        dx = float(np.median(np.abs(np.diff(radar["x"].values))))
        dy = float(np.median(np.abs(np.diff(radar["y"].values))))
        area = np.full(lon_v.shape, dx * dy)
    else:
        dlat = np.radians(np.abs(np.median(np.diff(lat_v, axis=0))))
        dlon = np.radians(np.abs(np.median(np.diff(lon_v, axis=-1))))
        area = EARTH_RADIUS**2 * np.cos(np.radians(lat_v)) * dlat * dlon
    return lon_v, lat_v, dims, area


def _edges(centres):
    """Edges of a regular axis (ascending) from its centres."""
    step = float(np.median(np.diff(centres)))
    return np.concatenate([centres - step / 2.0, [centres[-1] + step / 2.0]]), step


def _time_weights(times, windows, default):
    """Overlap (s) of the interval of every scan with every window: ``(nt, nw)``."""
    t = times.astype("datetime64[ns]").astype(np.int64) / 1e9
    if t.size == 1:
        half = np.array([default / 2.0])
        low, high = t - half, t + half
    else:
        mid = 0.5 * (t[1:] + t[:-1])
        low = np.concatenate([[t[0] - (mid[0] - t[0])], mid])
        high = np.concatenate([mid, [t[-1] + (t[-1] - mid[-1])]])
    start = windows[0].astype("datetime64[ns]").astype(np.int64) / 1e9
    end = windows[1].astype("datetime64[ns]").astype(np.int64) / 1e9
    return np.clip(
        np.minimum(high[:, None], end[None, :])
        - np.maximum(low[:, None], start[None, :]),
        0.0,
        None,
    )


@dask_support(
    "computes",
    "the scans of every step and the cells of the coarse grid are averaged in memory.",
)
@provenance("Averaging of a radar rain rate onto the grid and steps of IMERG")
def match_imerg(
    obj,
    imerg,
    *,
    interval="30min",
    time_label="start",
    min_coverage=0.5,
    min_time_coverage=0.8,
    output="rate",
):
    """
    Average a radar rain rate onto the grid and time steps of IMERG.

    Parameters
    ----------
    obj : xarray.DataArray
        Radar rain rate (mm/h) on a radar grid, ``(time, y, x)`` or ``(y, x)``, with the
        2-D coordinates ``lon`` and ``lat`` (or ``longitude`` and ``latitude``), or
        ``x`` and ``y`` with a ``crs_wkt`` grid mapping as the grids of radarx have.
        NaN is outside the coverage of the radar.
    imerg : xarray.DataArray
        IMERG rain rate (mm/h), ``(time, lat, lon)`` in any order, or without time.
    interval : str, float or numpy.timedelta64, optional
        Length of an IMERG step: a string like ``"30min"`` or ``"1h"``, seconds or a
        timedelta. Default ``"30min"``.
    time_label : {"start", "end", "center"}, optional
        The time of an IMERG value is the start, the end or the centre of its step.
        Default ``"start"``.
    min_coverage : float, optional
        Smallest fraction of the radar pixels that fit into a cell that must have a
        value. Default 0.5.
    min_time_coverage : float, optional
        Smallest fraction of a step that the scans of a pixel must cover. Default 0.8.
    output : {"rate", "accumulation"}, optional
        ``"rate"`` (default): mean rate in mm/h. ``"accumulation"``: mm in the
        step.

    Returns
    -------
    xarray.Dataset
        On the cells of IMERG that hold radar pixels (cropped to their bounding box,
        ``lat`` and ``lon`` ascending) and the steps of IMERG: ``radar``, ``imerg``,
        ``radar_coverage`` (fraction of the pixels that fit into the cell that have a
        value) and ``radar_time_coverage`` (mean fraction of the step covered).

    Raises
    ------
    TypeError
        If an input is not a DataArray.
    ValueError
        For unknown options, a radar without time for an IMERG with several steps, a
        grid that is not regular, or radar pixels outside the IMERG grid.

    References
    ----------
    Huffman, G. J., D. T. Bolvin, D. Braithwaite, K.-L. Hsu, R. J. Joyce, C. Kidd, E.
    J. Nelkin, S. Sorooshian, E. F. Stocker, J. Tan, D. B. Wolff, and P. Xie, 2020:
    Integrated Multi-satellitE Retrievals for the Global Precipitation Measurement
    (GPM) mission (IMERG). *Satellite Precipitation Measurement*, Springer, 343-353,
    https://doi.org/10.1007/978-3-030-24568-9_19

    Examples
    --------
    >>> both = radarx.retrieve.match_imerg(rain_rate, imerg["precipitation"])  # doctest: +SKIP
    >>> radarx.retrieve.match_statistics(both.imerg, both.radar, threshold=0.5)  # doctest: +SKIP
    """
    radar = obj
    for da in (radar, imerg):
        if not isinstance(da, xr.DataArray):
            raise TypeError("radar and imerg must be xarray.DataArray")
    if output not in _OUTPUTS:
        raise ValueError(f"output must be one of {_OUTPUTS}, not {output!r}")
    if time_label not in _LABELS:
        raise ValueError(f"time_label must be one of {_LABELS}, not {time_label!r}")
    for name, value in (
        ("min_coverage", min_coverage),
        ("min_time_coverage", min_time_coverage),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    seconds = _seconds(interval)
    if not seconds > 0:
        raise ValueError("interval must be positive")

    lon_dim = _find_dim(imerg, _LON, "longitude")
    lat_dim = _find_dim(imerg, _LAT, "latitude")
    imerg = imerg.sortby([lat_dim, lon_dim]).where(lambda a: a >= 0)
    lat_c, lon_c = imerg[lat_dim].values, imerg[lon_dim].values
    if lat_c.size < 2 or lon_c.size < 2:
        raise ValueError("the IMERG grid needs at least two cells in each direction")
    lat_edges, dlat = _edges(lat_c)
    lon_edges, dlon = _edges(lon_c)
    for c, d in ((lat_c, dlat), (lon_c, dlon)):
        if not np.allclose(np.diff(c), d, rtol=1e-3, atol=1e-6):
            raise ValueError("the IMERG grid must be regular")
    has_time = "time" in imerg.dims
    imerg_t = imerg.transpose(*(["time"] if has_time else []), lat_dim, lon_dim)
    times = imerg_t["time"].values if has_time else None

    # radar pixels -> cells
    lon, lat, rdims, area = _radar_lonlat(radar)
    ilat = np.searchsorted(lat_edges, lat, side="right") - 1
    ilon = np.searchsorted(lon_edges, lon, side="right") - 1
    inside = (ilat >= 0) & (ilat < lat_c.size) & (ilon >= 0) & (ilon < lon_c.size)
    if not inside.any():
        raise ValueError("no radar pixel lies inside the IMERG grid")
    rows, cols = np.unique(ilat[inside]), np.unique(ilon[inside])
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    nr, nc = r1 - r0, c1 - c0
    cell = np.where(inside, (ilat - r0) * nc + (ilon - c0), -1)
    cell_area = (
        EARTH_RADIUS**2
        * np.cos(np.radians(lat_c[r0:r1]))[:, None]
        * np.radians(dlat)
        * np.radians(dlon)
        * np.ones((1, nc))
    )
    expected = np.maximum(cell_area.ravel() / np.median(area[inside]), 1.0)

    # time windows
    if "time" in radar.dims:
        r_times = radar["time"].values
        order = np.argsort(r_times)
        r_data = radar.transpose("time", *rdims).values[order]
        r_times = r_times[order]
    else:
        r_times, r_data = None, radar.transpose(*rdims).values[None]
    if has_time:
        shift = {"start": (0.0, 1.0), "end": (-1.0, 0.0), "center": (-0.5, 0.5)}[
            time_label
        ]
        t_ns = times.astype("datetime64[ns]")
        start = t_ns + np.timedelta64(int(round(shift[0] * seconds * 1e9)), "ns")
        stop = t_ns + np.timedelta64(int(round(shift[1] * seconds * 1e9)), "ns")
        if r_times is None:
            if times.size != 1:
                raise ValueError(
                    "radar has no time dimension but IMERG has several steps"
                )
            weights = np.ones((1, 1)) * seconds
        else:
            weights = _time_weights(r_times, (start, stop), seconds)
    else:
        weights = np.ones((r_data.shape[0], 1)) * seconds
    n_steps = weights.shape[1]

    flat = cell.ravel()
    use = flat >= 0
    radar_out = np.full((n_steps, nr * nc), np.nan)
    cover_out = np.zeros((n_steps, nr * nc))
    time_cover = np.zeros(n_steps)
    data = r_data.reshape(r_data.shape[0], -1)[:, use]
    cells = flat[use]
    for j in range(n_steps):
        w = weights[:, j]
        finite = np.isfinite(data)
        wt = np.where(finite, w[:, None], 0.0)  # (nt, npixel)
        total = wt.sum(axis=0)
        covered = total >= float(min_time_coverage) * seconds
        rate = np.where(
            covered,
            (np.where(finite, data, 0.0) * wt).sum(axis=0)
            / np.where(total > 0, total, 1.0),
            np.nan,
        )
        ok = np.isfinite(rate)
        time_cover[j] = (
            float(np.mean(np.minimum(total, seconds) / seconds)) if total.size else 0.0
        )
        count = np.bincount(cells[ok], minlength=nr * nc)
        total_rate = np.bincount(cells[ok], weights=rate[ok], minlength=nr * nc)
        good = count >= float(min_coverage) * expected
        with np.errstate(invalid="ignore", divide="ignore"):
            radar_out[j] = np.where(
                good & (count > 0), total_rate / np.where(count > 0, count, 1), np.nan
            )
        cover_out[j] = np.minimum(count / expected, 1.0)
    scale = seconds / 3600.0 if output == "accumulation" else 1.0
    units = "mm" if output == "accumulation" else "mm h-1"
    dims = ("time", lat_dim, lon_dim) if has_time else (lat_dim, lon_dim)
    shape = (n_steps, nr, nc)
    sub = imerg_t.isel({lat_dim: slice(r0, r1), lon_dim: slice(c0, c1)})
    squeeze = (lambda a: a) if has_time else (lambda a: a[0])
    out = xr.Dataset(
        {
            "radar": (dims, squeeze(radar_out.reshape(shape) * scale)),
            "imerg": (dims, np.asarray(sub.values) * scale),
            "radar_coverage": (dims, squeeze(cover_out.reshape(shape))),
        },
        coords={k: v for k, v in sub.coords.items()},
    )
    out["radar_time_coverage"] = (
        ("time",) if has_time else (),
        time_cover if has_time else time_cover[0],
    )
    out["radar"].attrs = {
        "long_name": "radar rain " + ("accumulation" if scale != 1 else "rate"),
        "units": units,
    }
    out["imerg"].attrs = {
        "long_name": "IMERG rain " + ("accumulation" if scale != 1 else "rate"),
        "units": units,
    }
    out["radar_coverage"].attrs = {
        "long_name": "fraction of the radar pixels of the cell that have a value",
        "units": "1",
    }
    out["radar_time_coverage"].attrs = {
        "long_name": "mean fraction of the step that the radar scans cover",
        "units": "1",
    }
    out.attrs.update(
        {
            "interval_seconds": seconds,
            "time_label": time_label,
            "min_coverage": float(min_coverage),
            "min_time_coverage": float(min_time_coverage),
        }
    )
    return out


@accessor_method("dataarray", name="match_imerg", wraps=match_imerg)
def _match_imerg_accessor(self):
    """
    Average this radar rain rate onto the grid and time steps of IMERG.

    See :func:`radarx.retrieve.match_imerg` for the parameters.

    Returns
    -------
    xarray.Dataset
        ``radar``, ``imerg``, ``radar_coverage`` and ``radar_time_coverage``.
    """
