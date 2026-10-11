#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
CF georeferencing of grids
==========================

One builder for the coordinates of a gridded result, used by
:func:`radarx.grid.grid_cones`, :func:`radarx.grid.grid_radars`,
:func:`radarx.retrieve.create_cappi` and the CF writers of
:mod:`radarx.io.export`.

A grid on ``x`` and ``y`` in metres is georeferenced by its CRS. radarx grids
are azimuthal equidistant (AEQD) around a point, the radar or the origin of a
network. :func:`georeference` gives such a Dataset

* a grid mapping variable with the CF attributes of the CRS, and
  ``grid_mapping`` on every field that has ``x`` and ``y``;
* ``lat(y, x)`` and ``lon(y, x)`` computed from the CRS (on an AEQD grid the
  latitude depends on ``x`` as well as ``y``, so 1-D axes are 190 m off at the
  corner of a 60 km grid and about 4 km at 230 km range);
* ``standard_name``, ``units``, ``axis`` and ``positive`` on ``x``, ``y``,
  ``z``, and ``standard_name`` and ``axis`` on ``time``.

Attributes the object already has are kept.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = [
    "AXES",
    "CRS_NAMES",
    "aeqd_crs",
    "crs_variable",
    "georeference",
    "lonlat",
    "user_crs",
]

CRS_NAMES = ("crs_wkt", "spatial_ref", "crs")

AXES = {
    "x": {
        "standard_name": "projection_x_coordinate",
        "long_name": "x coordinate of the projection",
        "axis": "X",
        "units": "m",
    },
    "y": {
        "standard_name": "projection_y_coordinate",
        "long_name": "y coordinate of the projection",
        "axis": "Y",
        "units": "m",
    },
    "z": {"long_name": "height", "axis": "Z", "positive": "up", "units": "m"},
}

_SITE_UNITS = (
    ("latitude", "degrees_north"),
    ("longitude", "degrees_east"),
    ("altitude", "m"),
)


def aeqd_crs(latitude, longitude):
    """The azimuthal equidistant CRS (WGS84) centred on a point, as pyproj.CRS."""
    import pyproj

    return pyproj.CRS.from_dict(
        {
            "proj": "aeqd",
            "lat_0": float(latitude),
            "lon_0": float(longitude),
            "datum": "WGS84",
        }
    )


def crs_variable(latitude, longitude):
    """The AEQD CRS as a scalar variable with the CF grid mapping attributes."""
    return xr.DataArray(0, attrs=aeqd_crs(latitude, longitude).to_cf())


def user_crs(crs):
    """
    A pyproj.CRS from anything the CF writers accept.

    A ``(latitude, longitude)`` pair is the AEQD CRS centred there; anything
    else is passed to ``pyproj.CRS.from_user_input``.
    """
    import pyproj

    if isinstance(crs, (tuple, list)):
        return aeqd_crs(*(float(v) for v in crs))
    return pyproj.CRS.from_user_input(crs)


def lonlat(crs_attrs, x, y):
    """2-D longitude and latitude (degrees) of the cells of a projected grid."""
    import pyproj

    crs = pyproj.CRS.from_wkt(crs_attrs["crs_wkt"])
    transformer = pyproj.Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)
    gx, gy = np.meshgrid(np.asarray(x, "f8"), np.asarray(y, "f8"))
    lon, lat = transformer.transform(gx, gy)
    return np.asarray(lon), np.asarray(lat)


def _set_attrs(var, defaults):
    for key, value in defaults.items():
        var.attrs.setdefault(key, value)


def georeference(ds, mapping=None, *, name="crs_wkt", recompute=False, lonlat_=True):
    """
    Add the CF georeferencing of a grid to a Dataset (in place) and return it.

    Parameters
    ----------
    ds : xarray.Dataset
        Grid with ``x`` and ``y`` dimensions. It is changed; pass a copy.
    mapping : dict or None, optional
        CF grid mapping attributes of the CRS (``pyproj.CRS.to_cf()``). Without
        them no mapping, ``grid_mapping`` or ``lat`` and ``lon`` is added, and
        a ``grid_mapping`` the fields carry is removed.
    name : str, optional
        Name of the grid mapping variable. Default ``"crs_wkt"``, which the
        sounding, lightning and beam blockage functions look for.
    recompute : bool, optional
        Compute ``lat`` and ``lon`` from the CRS even when the Dataset has
        2-D ones (the CRS was given, not found). Default False: 2-D ones are
        kept.
    lonlat_ : bool, optional
        Write ``lat`` and ``lon``. Default True.

    Returns
    -------
    xarray.Dataset
        ``ds``, with the mapping variable as a coordinate.
    """
    for axis, attrs in AXES.items():
        if axis in ds.coords:
            _set_attrs(ds[axis], attrs)
            if (
                axis == "z"
                and ds[axis].ndim == 1
                and "sea level" in str(ds[axis].attrs.get("long_name", ""))
            ):
                _set_attrs(ds[axis], {"standard_name": "altitude"})
    if "time" in ds.coords:
        _set_attrs(ds["time"], {"standard_name": "time", "axis": "T"})
    for site, units in _SITE_UNITS:
        if site in ds.coords:
            _set_attrs(ds[site], {"units": units, "long_name": f"{site} of the radar"})

    if mapping is None:
        for var in ds.data_vars.values():
            var.attrs.pop("grid_mapping", None)
    else:
        ds = ds.assign_coords({name: xr.DataArray(0, attrs=dict(mapping))})
        have = ds.coords.get("lat"), ds.coords.get("lon")
        if lonlat_ and (
            recompute or any(c is None or c.ndim == 1 for c in have)  # 1-D axes
        ):
            lon, lat = lonlat(mapping, ds["x"].values, ds["y"].values)
            ds = ds.drop_vars(["lat", "lon"], errors="ignore")
            ds = ds.assign_coords(lat=(("y", "x"), lat), lon=(("y", "x"), lon))
        for var in ds.data_vars.values():
            if "x" in var.dims and "y" in var.dims:
                var.attrs["grid_mapping"] = name
    for coord, units, standard in (
        ("lat", "degrees_north", "latitude"),
        ("lon", "degrees_east", "longitude"),
    ):
        if coord in ds.coords:
            _set_attrs(ds[coord], {"units": units, "standard_name": standard})
    return ds
