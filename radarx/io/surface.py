#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Surface Stations
================

Readers for deployable surface weather stations into one xarray layout: an
:py:class:`xarray.Dataset` on ``(station, time)`` with the station
``latitude``, ``longitude`` and ``altitude`` as coordinates on ``station``
and the variables

======================  ===============================  ========
variable                CF standard name                 units
======================  ===============================  ========
``temperature``         ``air_temperature``              K
``relative_humidity``   ``relative_humidity``            1
``dewpoint``            ``dew_point_temperature``        K
``pressure``            ``surface_air_pressure``         Pa
``wind_speed``          ``wind_speed``                   m s-1
``wind_direction``      ``wind_from_direction``          degree
``u``, ``v``            ``eastward_wind``, ``northward_wind``  m s-1
======================  ===============================  ========

Stations with different sampling times share the union of their times
(missing samples are NaN). Every reader adds the string coordinates
``platform``, ``deployment`` and ``array_type`` on ``station``, so networks
from different readers combine with
``xarray.concat([a, b], dim="station", join="outer")``. The dew point is
derived from temperature and relative humidity (Bolton 1980, as in
:mod:`radarx.io.sounding`). These datasets feed
:func:`radarx.retrieve.potential_temperatures` and
:func:`radarx.retrieve.cold_pool_perturbation`.

Formats
-------
- :func:`read_sticknet`: Texas Tech University StickNet text files, e.g. the PERiLS 2022 archive (Weiss and McDonald 2022):
  ``<ID>_IOP<n>_level<k>.txt`` with the columns ``Time, T, RH, P, WS, WD``
  (degC, %, hPa, m s-1, degree; level 1 at 10 Hz with the QC flags ``TFLAG``
  and ``WFLAG``, levels 2 and 3 at 1 Hz) and the station table
  ``IOP<n>_StickNet_Locations.csv`` (``ID, Latitude, Longitude, Elevation,
  Array_Type``).
- :func:`read_pips`: the conventional (meteorological) netCDF files of the
  Portable In situ Precipitation Stations (PIPS), ``conventional_raw_*.nc``,
  with ``fasttemp``/``slowtemp`` (degC), ``RH`` (%), ``pressure`` (hPa),
  ``windspd`` and ``winddirabs`` and the station location in the ``location``
  attribute.

References
----------
Bolton, D., 1980: The computation of equivalent potential temperature. *Mon.
Wea. Rev.*, **108** (7), 1046-1053,
https://doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2

Weiss, C., and J. McDonald, 2022: PERiLS_2022: TTU StickNet Data. Version 1.0.
UCAR/NCAR Earth Observing Laboratory, https://doi.org/10.26023/93M9-AE8F-SX07

.. autosummary::
   :nosignatures:
   :toctree: generated/

   read_sticknet
   read_sticknet_locations
   read_pips
"""

from __future__ import annotations

__all__ = ["read_sticknet", "read_sticknet_locations", "read_pips"]

import ast
import io
import re
from pathlib import Path

import numpy as np
import xarray as xr

T0 = 273.15

_ATTRS = {
    "temperature": {
        "standard_name": "air_temperature",
        "long_name": "Air temperature",
        "units": "K",
    },
    "relative_humidity": {
        "standard_name": "relative_humidity",
        "long_name": "Relative humidity",
        "units": "1",
    },
    "dewpoint": {
        "standard_name": "dew_point_temperature",
        "long_name": "Dew point temperature (from temperature and RH)",
        "units": "K",
    },
    "pressure": {
        "standard_name": "surface_air_pressure",
        "long_name": "Station pressure",
        "units": "Pa",
    },
    "wind_speed": {
        "standard_name": "wind_speed",
        "long_name": "Wind speed",
        "units": "m s-1",
    },
    "wind_direction": {
        "standard_name": "wind_from_direction",
        "long_name": "Wind direction (from, clockwise from north)",
        "units": "degree",
    },
    "u": {
        "standard_name": "eastward_wind",
        "long_name": "Eastward wind",
        "units": "m s-1",
    },
    "v": {
        "standard_name": "northward_wind",
        "long_name": "Northward wind",
        "units": "m s-1",
    },
}
_COORD_ATTRS = {
    "latitude": {"standard_name": "latitude", "units": "degrees_north"},
    "longitude": {"standard_name": "longitude", "units": "degrees_east"},
    "altitude": {
        "standard_name": "altitude",
        "long_name": "Station elevation above sea level",
        "units": "m",
    },
}


def _dewpoint(t, rh):
    """Dew point (K) from temperature (K) and relative humidity (fraction)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        tc = t - T0
        e = rh * 611.2 * np.exp(17.67 * tc / (tc + 243.5))
        lg = np.log(np.where(e > 0, e, np.nan) / 611.2)
        return T0 + 243.5 * lg / (17.67 - lg)


def _station_dataset(records, *, source, attrs=None):
    """Combine per-station (times, {var: values}, meta) on the union of times."""
    names = [r[0] for r in records]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate station names: {names}")
    times = np.unique(np.concatenate([r[1] for r in records]))
    data = {k: np.full((len(records), times.size), np.nan) for k in _ATTRS}
    for i, (_, t, values, _) in enumerate(records):
        idx = np.searchsorted(times, t)
        for k, val in values.items():
            data[k][i, idx] = val
    t, rh = data["temperature"], data["relative_humidity"]
    data["dewpoint"] = _dewpoint(t, rh)
    ws, wd = data["wind_speed"], data["wind_direction"]
    rad = np.radians(wd)
    data["u"] = -ws * np.sin(rad)
    data["v"] = -ws * np.cos(rad)
    coords = {"station": ("station", names), "time": ("time", times)}
    for key in ("latitude", "longitude", "altitude"):
        coords[key] = (
            "station",
            np.array([r[3].get(key, np.nan) for r in records], dtype=float),
            _COORD_ATTRS[key],
        )
    # the same string coordinates for every reader, so networks concatenate
    coords["platform"] = ("station", np.array([source] * len(records)))
    for key in ("deployment", "array_type"):
        coords[key] = ("station", np.array([str(r[3].get(key, "")) for r in records]))
    ds = xr.Dataset(
        {k: (("station", "time"), data[k], dict(_ATTRS[k])) for k in _ATTRS},
        coords=coords,
    )
    ds.attrs.update(source=source, **(attrs or {}))
    return ds


def _select(times, values, time):
    if time is None:
        return times, values
    start = np.datetime64(time.start) if time.start is not None else times.min()
    stop = np.datetime64(time.stop) if time.stop is not None else times.max()
    keep = (times >= start) & (times <= stop)
    return times[keep], {k: v[keep] for k, v in values.items()}


# --------------------------------------------------------------------------
# StickNet
# --------------------------------------------------------------------------


def read_sticknet_locations(path):
    """
    Read a StickNet location table.

    Parameters
    ----------
    path : str or os.PathLike
        ``IOP<n>_StickNet_Locations.csv`` with the columns ``ID, Latitude,
        Longitude, Elevation, Array_Type``.

    Returns
    -------
    xarray.Dataset
        ``latitude``, ``longitude``, ``altitude`` and ``array_type`` on
        ``station`` (the ID as in the file, e.g. ``"101A"``).
    """
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    header = [h.strip().lower() for h in lines[0].split(",")]
    rows = [[c.strip() for c in ln.split(",")] for ln in lines[1:]]
    col = {name: i for i, name in enumerate(header)}
    for need in ("id", "latitude", "longitude", "elevation"):
        if need not in col:
            raise ValueError(f"{path}: column {need!r} not found in {header}")
    ids = [r[col["id"]] for r in rows]

    def num(name):
        return np.array([float(r[col[name]]) for r in rows])

    ds = xr.Dataset(
        coords={
            "station": ("station", ids),
            "latitude": ("station", num("latitude"), _COORD_ATTRS["latitude"]),
            "longitude": ("station", num("longitude"), _COORD_ATTRS["longitude"]),
            "altitude": ("station", num("elevation"), _COORD_ATTRS["altitude"]),
        }
    )
    if "array_type" in col:
        ds = ds.assign_coords(
            array_type=("station", np.array([r[col["array_type"]] for r in rows]))
        )
    return ds


def _sticknet_id(name):
    """Station ID of a StickNet file name, e.g. ``0101A_IOP2_level3`` -> 101A."""
    m = re.match(r"0*(\d+[A-Za-z]?)_IOP(\d+)", name)
    if m is None:
        raise ValueError(f"not a StickNet file name: {name}")
    return m.group(1), m.group(2)


def _read_sticknet_file(path, apply_flags):
    """Times and SI values of one StickNet file."""
    text = Path(path).read_text()
    first, _, body = text.partition("\n")
    header = [h.strip().upper() for h in first.split(",")]
    col = {name: i for i, name in enumerate(header)}
    for need in ("T", "RH", "P", "WS", "WD"):
        if need not in col:
            raise ValueError(f"{path}: column {need!r} not found in {header}")
    # empty fields are missing values
    body = body.replace("\r", "").replace(",,", ",nan,").replace(",,", ",nan,")
    body = body.replace(",\n", ",nan\n")
    if body.endswith(","):
        body += "nan"
    times = np.loadtxt(
        io.StringIO(body), delimiter=",", usecols=0, dtype="datetime64[ms]", ndmin=1
    ).astype("datetime64[ns]")
    numeric = list(range(1, len(header)))
    vals = np.loadtxt(io.StringIO(body), delimiter=",", usecols=numeric, ndmin=2)

    def get(name):
        return vals[:, numeric.index(col[name])].astype(np.float64)

    t, rh, p, ws, wd = (get(n) for n in ("T", "RH", "P", "WS", "WD"))
    if apply_flags and "TFLAG" in col:
        bad = get("TFLAG") != 0
        t[bad] = rh[bad] = p[bad] = np.nan
    if apply_flags and "WFLAG" in col:
        bad = get("WFLAG") != 0
        ws[bad] = wd[bad] = np.nan
    values = {
        "temperature": t + T0,
        "relative_humidity": rh / 100.0,
        "pressure": p * 100.0,
        "wind_speed": ws,
        "wind_direction": wd,
    }
    return times, values


def read_sticknet(
    files, locations=None, *, iop=None, level=3, time=None, apply_flags=True
):
    """
    Read TTU StickNet files into a station network.

    Parameters
    ----------
    files : str, os.PathLike or list of them
        StickNet text files (``<ID>_IOP<n>_level<k>.txt``), one per station,
        or a directory: all files of deployment ``iop`` and level ``level``
        in it.
    locations : str, os.PathLike or xarray.Dataset, optional
        Station table (:func:`read_sticknet_locations`). Default:
        ``IOP<n>_StickNet_Locations.csv`` next to the data files, if present.
    iop : int, optional
        Deployment (IOP) to read from a directory; required when the
        directory holds several.
    level : int, optional
        Data level to read from a directory: 1 (10 Hz, QC flags), 2 (1 Hz,
        filtered) or 3 (level 2 bias-corrected, recommended). Default 3.
    time : slice, optional
        Keep only this time window (e.g.
        ``slice("2022-03-30T18:00", "2022-03-31T03:00")``); saves memory for
        the 10-Hz level-1 files.
    apply_flags : bool, optional
        For level-1 files, set values that failed the QC (``TFLAG`` for
        temperature, humidity and pressure, ``WFLAG`` for wind) to NaN.
        Default True.

    Returns
    -------
    xarray.Dataset
        Station network on ``(station, time)`` (see :mod:`radarx.io.surface`)
        with the string coordinates ``platform``, ``deployment`` (``IOP<n>``)
        and ``array_type`` (``Coarse`` or ``Fine``, from the table) on
        ``station``.

    Notes
    -----
    The StickNet documentation lists, per deployment, stations whose
    temperature or wind direction could not be calibrated after the
    deployment, and two stations (104A and 109A) with a slower temperature
    and humidity sensor (time constant 42 s instead of 10 s); check it before
    comparing stations across sharp gradients.

    References
    ----------
    Weiss, C., and J. McDonald, 2022: PERiLS_2022: TTU StickNet Data. Version
    1.0. UCAR/NCAR Earth Observing Laboratory,
    https://doi.org/10.26023/93M9-AE8F-SX07
    """
    if isinstance(files, (str, Path)):
        path = Path(files)
        if path.is_dir():
            pattern = f"*_IOP{iop}_level{level}.txt" if iop else f"*_level{level}.txt"
            files = sorted(path.glob(pattern))
            iops = {_sticknet_id(f.name)[1] for f in files}
            if len(iops) > 1:
                raise ValueError(
                    f"{path} holds several deployments {sorted(iops)}; pass iop="
                )
        else:
            files = [path]
    files = [Path(f) for f in files]
    if not files:
        raise FileNotFoundError("no StickNet files given")
    if locations is None:
        _, iop = _sticknet_id(files[0].name)
        table = files[0].parent / f"IOP{iop}_StickNet_Locations.csv"
        locations = read_sticknet_locations(table) if table.exists() else None
    elif not isinstance(locations, xr.Dataset):
        locations = read_sticknet_locations(locations)
    records = []
    for f in files:
        sid, iop = _sticknet_id(f.name)
        times, values = _read_sticknet_file(f, apply_flags)
        times, values = _select(times, values, time)
        meta = {"deployment": f"IOP{iop}"}
        if locations is not None and sid in locations["station"].values:
            loc = locations.sel(station=sid)
            meta.update(
                latitude=float(loc["latitude"]),
                longitude=float(loc["longitude"]),
                altitude=float(loc["altitude"]),
            )
            if "array_type" in loc.coords:
                meta["array_type"] = str(loc["array_type"].values)
        records.append((sid, times, values, meta))
    found = re.search(r"level(\d)", files[0].name)
    return _station_dataset(
        records,
        source="TTU StickNet",
        attrs={
            "level": int(found.group(1)) if found else -1,
            "references": "https://doi.org/10.26023/93M9-AE8F-SX07",
        },
    )


# --------------------------------------------------------------------------
# PIPS
# --------------------------------------------------------------------------


def read_pips(files, *, temperature="fasttemp", time=None):
    """
    Read PIPS conventional (meteorological) netCDF files into a station
    network.

    Parameters
    ----------
    files : str, os.PathLike or list of them
        ``conventional_raw_*.nc`` files, one per probe; a directory reads all
        of them in it.
    temperature : {"fasttemp", "slowtemp"}, optional
        Thermistor used for ``temperature``. Default ``"fasttemp"``.
    time : slice, optional
        Keep only this time window.

    Returns
    -------
    xarray.Dataset
        Station network on ``(station, time)`` (see :mod:`radarx.io.surface`),
        stations named by the ``probe_name`` attribute.
    """
    if isinstance(files, (str, Path)):
        path = Path(files)
        files = sorted(path.glob("conventional_raw_*.nc")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError("no PIPS files given")
    records = []
    for f in files:
        with xr.open_dataset(f) as ds:
            if temperature not in ds:
                raise KeyError(f"{f}: no variable {temperature!r}")
            times = ds["time"].values.astype("datetime64[ns]")
            values = {
                "temperature": ds[temperature].values.astype(float) + T0,
                "relative_humidity": ds["RH"].values.astype(float) / 100.0,
                "pressure": ds["pressure"].values.astype(float) * 100.0,
                "wind_speed": ds["windspd"].values.astype(float),
                "wind_direction": ds["winddirabs"].values.astype(float),
            }
            name = str(ds.attrs.get("probe_name", Path(f).stem))
            meta = {}
            loc = ds.attrs.get("location")
            if isinstance(loc, str):
                loc = ast.literal_eval(loc)
            if loc is not None and len(loc) == 3:
                meta = dict(
                    latitude=float(loc[0]),
                    longitude=float(loc[1]),
                    altitude=float(loc[2]),
                )
            if "deployment_name" in ds.attrs:
                meta["deployment"] = str(ds.attrs["deployment_name"])
        order = np.argsort(times, kind="stable")
        times = times[order]
        values = {k: v[order] for k, v in values.items()}
        times, values = _select(times, values, time)
        records.append((name, times, values, meta))
    return _station_dataset(
        records, source="PIPS", attrs={"temperature_sensor": temperature}
    )
