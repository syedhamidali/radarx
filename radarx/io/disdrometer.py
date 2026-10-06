#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Disdrometer Readers
===================

Read OTT Parsivel / Parsivel2 laser disdrometer records, as logged by the
Portable In situ Precipitation Stations (PIPS) or by any data logger that
stores the Parsivel telegram, into :py:class:`xarray.Dataset` objects.

Dataset format
--------------

Every reader returns a dataset on the dimensions ``time``, ``velocity`` and
``diameter`` (the 32 fall-speed and 32 size classes of the Parsivel) with

=========================  ===========================================  =========
variable                   meaning                                      units
=========================  ===========================================  =========
``counts``                 particles per (velocity, diameter) class     1
``sample_interval``        integration time of each record              s
``rain_rate_instrument``   rain intensity computed by the Parsivel      mm h-1
``reflectivity_instrument``  reflectivity computed by the Parsivel      dBZ
``particle_count``         particles detected by the Parsivel           1
=========================  ===========================================  =========

and, when the logger records them, ``rain_accumulation``,
``signal_amplitude``, ``sensor_temperature``, ``supply_voltage`` and the
station's meteorological measurements averaged over each record
(``wind_speed``, ``wind_speed_max``, ``wind_direction``,
``air_temperature``, ``relative_humidity``, ``air_pressure``).

The coordinates are ``diameter`` (class centres, mm) with ``bin_width``,
``diameter_lower`` and ``diameter_upper``; ``velocity`` (class centres,
m s-1) with ``velocity_width``, ``velocity_lower`` and ``velocity_upper``;
and the scalar ``station``, ``latitude``, ``longitude`` and ``altitude`` (m
above sea level). Several stations combine with
``xr.concat(datasets, dim="station")``.

The analysis of these datasets (quality control, N(D), gamma fits, radar
variables and radar matching) is in :mod:`radarx.retrieve.disdrometer`.

Formats
-------
:func:`read_parsivel`
    Text files holding the Parsivel telegram (the ``;``-separated fields
    serial number, rain intensity, rain amount, reflectivity, sample
    interval, signal amplitude, particle count, sensor temperature, supply
    voltage, time, date and the 1024 counts of the raw spectrum, ordered
    by velocity class, then diameter class): PIPS merged CSV files (1 Hz
    meteorological records with a ``ParsivelStr`` column), Campbell
    Scientific TOA5 files with a ``ParsivelStr`` column, or files with one
    telegram per line.
:func:`read_pips_netcdf`
    The netCDF files of PIPS deployments (``parsivel_combined_*.nc``), e.g.
    from PERiLS, with a ``VD_matrix`` of counts.

References
----------
Löffler-Mang, M., and J. Joss, 2000: An optical disdrometer for measuring
size and velocity of hydrometeors. *J. Atmos. Oceanic Technol.*, **17** (2),
130-139, https://doi.org/10.1175/1520-0426(2000)017<0130:AODFMS>2.0.CO;2

Tokay, A., D. B. Wolff, and W. A. Petersen, 2014: Evaluation of the new
version of the laser-optical disdrometer, OTT Parsivel2. *J. Atmos. Oceanic
Technol.*, **31** (6), 1276-1288, https://doi.org/10.1175/JTECH-D-13-00174.1

.. autosummary::
   :nosignatures:
   :toctree: generated/

   read_parsivel
   read_pips_netcdf
   parsivel_classes
"""

from __future__ import annotations

__all__ = ["read_parsivel", "read_pips_netcdf", "parsivel_classes"]

import os
import re

import numpy as np
import xarray as xr

N_CLASSES = 32

# OTT Parsivel size classes [mm]: centres and widths (classes 1-2 are not
# measured by the instrument)
DIAMETER_WIDTHS = np.array(
    [0.125] * 10 + [0.25] * 5 + [0.5] * 5 + [1.0] * 5 + [2.0] * 5 + [3.0] * 2
)
DIAMETER_CENTERS = np.cumsum(DIAMETER_WIDTHS) - DIAMETER_WIDTHS / 2
# OTT Parsivel fall-speed classes [m s-1]: centres and widths
VELOCITY_WIDTHS = np.array(
    [0.1] * 10 + [0.2] * 5 + [0.4] * 5 + [0.8] * 5 + [1.6] * 5 + [3.2] * 2
)
VELOCITY_CENTERS = np.cumsum(VELOCITY_WIDTHS) - VELOCITY_WIDTHS / 2

_TELEGRAM_FIELDS = (
    ("serial", None),
    ("rain_rate_instrument", float),
    ("rain_accumulation", float),
    ("reflectivity_instrument", float),
    ("sample_interval", float),
    ("signal_amplitude", float),
    ("particle_count", float),
    ("sensor_temperature", float),
    ("supply_voltage", float),
)
_N_HEADER = 11  # telegram fields before the spectrum (incl. time and date)

_ATTRS = {
    "counts": {
        "long_name": "Number of particles per fall speed and diameter class",
        "units": "1",
    },
    "sample_interval": {"long_name": "Sample interval", "units": "s"},
    "rain_rate_instrument": {
        "standard_name": "rainfall_rate",
        "long_name": "Rain intensity computed by the instrument",
        "units": "mm h-1",
    },
    "rain_accumulation": {
        "long_name": "Rain amount accumulated by the instrument",
        "units": "mm",
    },
    "reflectivity_instrument": {
        "standard_name": "equivalent_reflectivity_factor",
        "long_name": "Reflectivity computed by the instrument",
        "units": "dBZ",
    },
    "signal_amplitude": {"long_name": "Laser signal amplitude", "units": "1"},
    "particle_count": {"long_name": "Number of detected particles", "units": "1"},
    "sensor_temperature": {
        "long_name": "Temperature of the sensor head",
        "units": "degC",
    },
    "supply_voltage": {"long_name": "Supply voltage", "units": "V"},
    "wind_speed": {
        "standard_name": "wind_speed",
        "long_name": "Wind speed (mean over the sample interval)",
        "units": "m s-1",
    },
    "wind_speed_max": {
        "standard_name": "wind_speed_of_gust",
        "long_name": "Maximum wind speed over the sample interval",
        "units": "m s-1",
    },
    "wind_direction": {
        "standard_name": "wind_from_direction",
        "long_name": "Wind direction (mean over the sample interval)",
        "units": "degree",
    },
    "air_temperature": {
        "standard_name": "air_temperature",
        "long_name": "Air temperature",
        "units": "degC",
    },
    "relative_humidity": {
        "standard_name": "relative_humidity",
        "long_name": "Relative humidity",
        "units": "percent",
    },
    "air_pressure": {
        "standard_name": "air_pressure",
        "long_name": "Air pressure",
        "units": "hPa",
    },
}

# PIPS logger column -> (variable, reduction over the sample interval)
_PIPS_MET = {
    "WS_ms": ("wind_speed", "mean"),
    "WindDirAbs": ("wind_direction", "circmean"),
    "SlowTemp": ("air_temperature", "mean"),
    "RH": ("relative_humidity", "mean"),
    "Pressure": ("air_pressure", "mean"),
}


def parsivel_classes():
    """
    Size and fall-speed classes of the OTT Parsivel disdrometer.

    Returns
    -------
    xarray.Dataset
        Empty dataset with the coordinates ``diameter`` (32 class centres,
        mm) with ``bin_width``, ``diameter_lower`` and ``diameter_upper``,
        and ``velocity`` (32 class centres, m s-1) with ``velocity_width``,
        ``velocity_lower`` and ``velocity_upper``. The two smallest size
        classes are not measured by the instrument.

    References
    ----------
    Tokay, A., D. B. Wolff, and W. A. Petersen, 2014: Evaluation of the
    new version of the laser-optical disdrometer, OTT Parsivel2. *J. Atmos.
    Oceanic Technol.*, **31** (6), 1276-1288,
    https://doi.org/10.1175/JTECH-D-13-00174.1
    """
    d, dw = DIAMETER_CENTERS, DIAMETER_WIDTHS
    v, vw = VELOCITY_CENTERS, VELOCITY_WIDTHS
    mm = {"units": "mm"}
    ms = {"units": "m s-1"}
    return xr.Dataset(
        coords={
            "diameter": (
                "diameter",
                d,
                {"long_name": "Drop diameter (class centre)", **mm},
            ),
            "bin_width": ("diameter", dw, {"long_name": "Size class width", **mm}),
            "diameter_lower": (
                "diameter",
                d - dw / 2,
                {"long_name": "Lower size class edge", **mm},
            ),
            "diameter_upper": (
                "diameter",
                d + dw / 2,
                {"long_name": "Upper size class edge", **mm},
            ),
            "velocity": (
                "velocity",
                v,
                {"long_name": "Fall speed (class centre)", **ms},
            ),
            "velocity_width": (
                "velocity",
                vw,
                {"long_name": "Fall speed class width", **ms},
            ),
            "velocity_lower": (
                "velocity",
                v - vw / 2,
                {"long_name": "Lower fall speed class edge", **ms},
            ),
            "velocity_upper": (
                "velocity",
                v + vw / 2,
                {"long_name": "Upper fall speed class edge", **ms},
            ),
        },
        attrs={"instrument": "OTT Parsivel"},
    )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _build(time, counts, scalars, station, latitude, longitude, altitude, attrs):
    """Assemble the reader output."""
    order = np.argsort(time, kind="stable")
    time = time[order]
    keep = np.ones(time.size, bool)
    keep[1:] = time[1:] != time[:-1]  # drop duplicated records
    sel = order[keep]
    ds = parsivel_classes()
    ds = ds.assign_coords(time=("time", time[keep], {"long_name": "Time"}))
    ds["counts"] = (
        ("time", "velocity", "diameter"),
        counts[sel],
        dict(_ATTRS["counts"]),
    )
    for name, values in scalars.items():
        ds[name] = ("time", np.asarray(values, float)[sel], dict(_ATTRS[name]))
    ds = ds.assign_coords(
        station=((), str(station), {"long_name": "Station name"}),
        latitude=(
            (),
            float(latitude),
            {"standard_name": "latitude", "units": "degrees_north"},
        ),
        longitude=(
            (),
            float(longitude),
            {"standard_name": "longitude", "units": "degrees_east"},
        ),
        altitude=(
            (),
            float(altitude),
            {
                "standard_name": "altitude",
                "long_name": "Altitude of the instrument",
                "units": "m",
            },
        ),
    )
    ds.attrs.update(attrs)
    return ds


def _parse_telegrams(telegrams):
    """
    Split telegram strings into the header fields, the instrument clock and
    the (n, 32, 32) count spectra. Telegrams without a full spectrum are
    flagged invalid.
    """
    n = len(telegrams)
    nspec = N_CLASSES * N_CLASSES
    counts = np.zeros((n, nspec), np.int32)
    fields = np.full((n, len(_TELEGRAM_FIELDS)), np.nan)
    clock = np.full(n, np.datetime64("NaT", "s"))
    serial = ""
    valid = np.zeros(n, bool)
    for i, tel in enumerate(telegrams):
        parts = tel.strip().strip('"').split(";")
        if len(parts) < _N_HEADER + nspec:
            continue
        try:
            spec = np.array(parts[_N_HEADER : _N_HEADER + nspec], dtype=np.int32)
            head = [
                float(p) if conv else np.nan
                for p, (_, conv) in zip(
                    parts[: len(_TELEGRAM_FIELDS)], _TELEGRAM_FIELDS
                )
            ]
        except ValueError:
            continue
        counts[i] = spec
        fields[i] = head
        serial = serial or parts[0].strip()
        clock[i] = _parsivel_clock(parts[9], parts[10])
        valid[i] = True
    return valid, fields, clock, counts.reshape((n, N_CLASSES, N_CLASSES)), serial


def _parsivel_clock(hms, dmy):
    """Instrument time from the ``HH:MM:SS`` and ``DD.MM.YYYY`` fields."""
    try:
        d, m, y = dmy.strip().split(".")
        return np.datetime64(f"{y}-{m}-{d}T{hms.strip()}", "s")
    except ValueError:
        return np.datetime64("NaT", "s")


def _gps_degrees(value, hemisphere):
    """
    Degrees from a PIPS GPS value ``dd.mmmm`` (degrees, then minutes times
    100 after the point) and the hemisphere letter.
    """
    with np.errstate(invalid="ignore"):
        deg = np.trunc(value)
        out = deg + (value - deg) * 100.0 / 60.0
    sign = np.where(np.isin(hemisphere, ["S", "W"]), -1.0, 1.0)
    return out * sign


def _float_column(column):
    out = np.full(len(column), np.nan)
    for i, v in enumerate(column):
        try:
            out[i] = float(v)
        except ValueError:
            pass
    return out


def _reduce(values, ends, how):
    """Reduce 1 Hz values over the segments (ends[i-1], ends[i]]."""
    starts = np.concatenate([[0], ends[:-1] + 1])
    out = np.full(ends.size, np.nan)
    for i, (s, e) in enumerate(zip(starts, ends + 1)):
        seg = values[s:e]
        seg = seg[np.isfinite(seg)]
        if seg.size == 0:
            continue
        if how == "mean":
            out[i] = seg.mean()
        elif how == "max":
            out[i] = seg.max()
        else:  # circular mean of directions in degrees
            r = np.deg2rad(seg)
            out[i] = np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360
    return out


def _station_name(path, serial):
    name = os.path.basename(str(path))
    m = re.search(r"PIPS\d[A-Z]", name)
    if m:
        return m.group(0)
    return f"Parsivel {serial}" if serial else name


# --------------------------------------------------------------------------
# readers
# --------------------------------------------------------------------------


def _read_table(path):
    """Header names and rows of a CSV/TOA5 file (quotes removed)."""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    if not lines:
        raise ValueError(f"{path} is empty")
    first = lines[0].strip()
    if first.startswith('"TOA5"') or first.startswith("TOA5"):
        header = [h.strip('"') for h in lines[1].split(",")]
        body = lines[4:]
    elif first.replace('"', "").startswith("TIMESTAMP"):
        header = [h.strip('"') for h in first.split(",")]
        body = lines[1:]
    else:
        return None, lines
    rows = [r.split(",") for r in body if r.strip()]
    return header, rows


def read_parsivel(
    path,
    *,
    time="auto",
    station=None,
    latitude=None,
    longitude=None,
    altitude=None,
):
    """
    Read Parsivel telegrams from a PIPS, TOA5 or plain text file.

    Parameters
    ----------
    path : str or path-like
        A PIPS merged CSV file (``TIMESTAMP, ..., ParsivelStr``), a
        Campbell Scientific TOA5 file with a ``ParsivelStr`` column, or a
        text file with one Parsivel telegram per line (see
        :mod:`radarx.io.disdrometer` for the telegram fields).
    time : {"auto", "gps", "logger", "instrument"}, optional
        Time stamp of each record: the GPS time logged with the telegram
        (PIPS), the logger time stamp, or the Parsivel clock. ``"auto"``
        (default) takes the GPS time where it is available (records
        without a GPS fix are shifted by the median GPS-logger offset),
        else the logger time, else the Parsivel clock.
    station : str, optional
        Station name. Default: ``PIPS..`` from the file name, else the
        Parsivel serial number.
    latitude, longitude, altitude : float, optional
        Station position (degrees north and east, m above sea level).
        Default: the median of the logged GPS fixes (PIPS), else NaN.

    Returns
    -------
    xarray.Dataset
        ``counts`` on ``(time, velocity, diameter)`` with the instrument
        variables and, for PIPS files, the meteorological measurements
        averaged over each sample interval; see
        :mod:`radarx.io.disdrometer`. Records without a complete spectrum
        (e.g. ``NAN`` telegrams) are skipped.

    Examples
    --------
    >>> ds = read_parsivel("PIPS1A_IOP2_033022_merged.txt")  # doctest: +SKIP
    >>> ds.counts.sum(("velocity", "diameter")).plot()  # doctest: +SKIP
    """
    if time not in ("auto", "gps", "logger", "instrument"):
        raise ValueError(
            "time must be 'auto', 'gps', 'logger' or 'instrument', " f"not {time!r}"
        )
    header, rows = _read_table(path)
    if header is None:  # plain telegrams
        if time in ("gps", "logger"):
            raise ValueError(f"{path} has no {time} time stamps")
        valid, fields, clock, counts, serial = _parse_telegrams(rows)
        logger = None
        gps_time = None
        met = {}
        gps_pos = None
        tel_rows = np.flatnonzero(valid)
    else:
        if "ParsivelStr" not in header:
            raise ValueError(f"{path} has no ParsivelStr column")
        ntel = len(header) - 1
        # the telegram is the last column and itself holds no commas
        tels = [",".join(r[ntel:]) if len(r) > ntel else "" for r in rows]
        valid, fields, clock, counts, serial = _parse_telegrams(tels)
        tel_rows = np.flatnonzero(valid)
        col = {h: i for i, h in enumerate(header)}

        def column(name):
            i = col[name]
            return [r[i].strip('"') if len(r) > i else "" for r in rows]

        stamps = np.array(
            [s.replace(" ", "T") for s in column("TIMESTAMP")], "datetime64[s]"
        )
        logger = stamps[tel_rows]
        met = {}
        for name, (var, how) in _PIPS_MET.items():
            if name in col:
                vals = _float_column(column(name))
                met[var] = _reduce(vals, tel_rows, how)
                if var == "wind_speed":
                    met["wind_speed_max"] = _reduce(vals, tel_rows, "max")
        gps_pos = None
        gps_time = None
        if {"GPSTime", "GPSDate"} <= col.keys():
            gt, gd = column("GPSTime"), column("GPSDate")
            gps_time = np.full(len(rows), np.datetime64("NaT", "s"))
            for i, (t_, d_) in enumerate(zip(gt, gd)):
                if len(t_) == 6 and len(d_) == 6 and t_.isdigit() and d_.isdigit():
                    gps_time[i] = np.datetime64(
                        f"20{d_[4:]}-{d_[2:4]}-{d_[:2]}T{t_[:2]}:{t_[2:4]}:{t_[4:]}",
                        "s",
                    )
            gps_time = gps_time[tel_rows]
        if {"GPSLat", "GPSLatHem", "GPSLon", "GPSLonHem"} <= col.keys():
            lat = _gps_degrees(
                _float_column(column("GPSLat")), np.array(column("GPSLatHem"))
            )
            lon = _gps_degrees(
                _float_column(column("GPSLon")), np.array(column("GPSLonHem"))
            )
            alt = (
                _float_column(column("GPSAlt"))
                if "GPSAlt" in col
                else np.full(lat.size, np.nan)
            )
            good = np.isfinite(lat) & np.isfinite(lon)
            if good.any():
                gps_pos = (
                    float(np.median(lat[good])),
                    float(np.median(lon[good])),
                    (
                        float(np.nanmedian(alt[good]))
                        if np.isfinite(alt[good]).any()
                        else np.nan
                    ),
                )

    if tel_rows.size == 0:
        raise ValueError(f"no complete Parsivel telegram in {path}")
    fields, clock, counts = fields[tel_rows], clock[tel_rows], counts[tel_rows]

    if time == "instrument" or (time == "auto" and logger is None):
        stamp = clock
        source = "instrument clock"
    elif time == "logger":
        stamp = logger
        source = "logger"
    else:
        have = gps_time is not None and (~np.isnat(gps_time)).any()
        if not have:
            if time == "gps":
                raise ValueError(f"{path} has no GPS time")
            stamp, source = logger, "logger"
        else:
            ok = ~np.isnat(gps_time)
            offset = np.median((gps_time[ok] - logger[ok]).astype(np.int64))
            stamp = np.where(ok, gps_time, logger + np.timedelta64(int(offset), "s"))
            source = "GPS"
    good = ~np.isnat(stamp)
    scalars = {
        name: fields[good, j]
        for j, (name, conv) in enumerate(_TELEGRAM_FIELDS)
        if conv is not None
    }
    scalars.update({k: v[good] for k, v in met.items()})
    pos = gps_pos or (np.nan, np.nan, np.nan)
    return _build(
        stamp[good].astype("datetime64[ns]"),
        counts[good],
        scalars,
        station or _station_name(path, serial),
        pos[0] if latitude is None else latitude,
        pos[1] if longitude is None else longitude,
        pos[2] if altitude is None else altitude,
        {
            "instrument": "OTT Parsivel",
            "serial_number": serial,
            "time_source": source,
            "source_file": os.path.basename(str(path)),
        },
    )


def read_pips_netcdf(path, *, station=None):
    """
    Read the raw spectra of a PIPS netCDF file.

    Parameters
    ----------
    path : str or path-like
        A PIPS ``parsivel_combined_*.nc`` file with ``VD_matrix`` (counts
        on ``time``, ``fallspeed_bin``, ``diameter_bin``).
    station : str, optional
        Station name. Default: the ``probe_name`` attribute.

    Returns
    -------
    xarray.Dataset
        ``counts`` on ``(time, velocity, diameter)`` and the instrument and
        meteorological variables in the format of :func:`read_parsivel`;
        the station position comes from the ``location`` attribute
        (latitude, longitude, altitude). Products stored in the file (e.g.
        quality-controlled spectra or fits) are not read.

    Examples
    --------
    >>> ds = read_pips_netcdf("parsivel_combined_IOP2_033022_PIPS1A_60s.nc")
    ... # doctest: +SKIP
    """
    names = {
        "precipintensity": "rain_rate_instrument",
        "precipaccum": "rain_accumulation",
        "parsivel_dBZ": "reflectivity_instrument",
        "sample_interval": "sample_interval",
        "signal_amplitude": "signal_amplitude",
        "pcount": "particle_count",
        "sensor_temp": "sensor_temperature",
        "pvoltage": "supply_voltage",
        "windspd": "wind_speed",
        "windgust": "wind_speed_max",
        "winddirabs": "wind_direction",
        "slowtemp": "air_temperature",
        "RH": "relative_humidity",
        "pressure": "air_pressure",
    }
    with xr.open_dataset(path) as src:
        if "VD_matrix" not in src:
            raise ValueError(f"{path} has no VD_matrix")
        vd = src["VD_matrix"].transpose("time", "fallspeed_bin", "diameter_bin")
        counts = np.nan_to_num(vd.values).astype(np.int32)
        scalars = {new: src[old].values for old, new in names.items() if old in src}
        loc = [
            float(x)
            for x in re.findall(
                r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", str(src.attrs.get("location", ""))
            )
        ]
        loc = (loc + [np.nan] * 3)[:3]
        attrs = {
            "instrument": "OTT Parsivel",
            "time_source": "PIPS netCDF",
            "source_file": os.path.basename(str(path)),
        }
        for key in ("deployment_name", "parsivel_angle"):
            if key in src.attrs:
                attrs[key] = src.attrs[key]
        return _build(
            src["time"].values.astype("datetime64[ns]"),
            counts,
            scalars,
            station or src.attrs.get("probe_name", _station_name(path, "")),
            *loc,
            attrs,
        )
