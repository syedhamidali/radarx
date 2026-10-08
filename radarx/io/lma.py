#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Lightning Mapping Array Data
============================

Reader of the VHF source ("event") files of a Lightning Mapping Array (LMA;
Rison et al. 1999; Thomas et al. 2004): the ASCII ``.dat`` files written by
the New Mexico Tech ``lma_analysis`` program (``LYLOUT_YYMMDD_HHMMSS_SSSS.dat``,
optionally gzip- or bzip2-compressed).

Every file has a header (analysis program, start time, network centre,
station table, station statistics, and a ``Data:`` line naming the columns),
then a line ``*** data ***`` and one source per line: time in UT seconds of
the day, latitude, longitude, altitude (m above mean sea level), reduced
chi-square of the time-of-arrival solution, power (dBW) and a hexadecimal
mask of the contributing stations. Columns are matched by name, so files
with other column sets (e.g. without power) are read as well.

Dataset layout
--------------

:func:`read_lma` returns an :py:class:`xarray.Dataset` with the source
variables on a ``number_of_events`` dimension and the station table on
``number_of_stations``, named as in the CF layout of the xlma-python package
(so its NetCDF files open with :func:`xarray.open_dataset` and work with the
functions of :mod:`radarx.retrieve.lightning` too):

=============================  =========================================
variable                       meaning
=============================  =========================================
``event_time``                 source time (datetime64, coordinate)
``event_latitude``             degrees north (coordinate)
``event_longitude``            degrees east (coordinate)
``event_altitude``             m above mean sea level
``event_chi2``                 reduced chi-square of the solution
``event_power``                received power, dBW
``event_mask``                 bit mask of contributing stations
``event_stations``             number of contributing stations
``station_code`` ...           station letter, name, location
``network_center_latitude``    network centre (scalar)
=============================  =========================================

Parsing the data section runs in a compiled kernel (``radarx.io._lma``,
multithreaded over chunks of the text) with a NumPy fallback.

References
----------
Rison, W., R. J. Thomas, P. R. Krehbiel, T. Hamlin, and J. Harlin, 1999: A
GPS-based three-dimensional lightning mapping system: Initial observations
in central New Mexico. *Geophys. Res. Lett.*, **26** (23), 3573-3576,
https://doi.org/10.1029/1999GL010856

Thomas, R. J., P. R. Krehbiel, W. Rison, S. J. Hunyady, W. P. Winn,
T. Hamlin, and J. Harlin, 2004: Accuracy of the Lightning Mapping Array.
*J. Geophys. Res.*, **109**, D14207, https://doi.org/10.1029/2004JD004549

.. autosummary::
   :nosignatures:
   :toctree: generated/

   read_lma
"""

from __future__ import annotations

__all__ = ["read_lma"]

import bz2
import datetime as dt
import gzip
import os
import re

import numpy as np
import xarray as xr

try:
    from . import _lma

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _lma = None
    HAS_COMPILED_KERNEL = False

_DATA_MARK = b"*** data ***"

# column name in the "Data:" header line -> (variable, matcher)
_COLUMNS = (
    ("event_time", re.compile(r"^time")),
    ("event_latitude", re.compile(r"^lat")),
    ("event_longitude", re.compile(r"^lon")),
    ("event_altitude", re.compile(r"^alt")),
    ("event_chi2", re.compile(r"chi")),
    ("event_power", re.compile(r"^p\b|^p\(|power")),
    ("event_mask", re.compile(r"mask")),
    ("event_stations", re.compile(r"stations|^#|^nsta")),
)

_ATTRS = {
    "event_time": {"long_name": "Time of the VHF source"},
    "event_latitude": {
        "standard_name": "latitude",
        "long_name": "Latitude of the VHF source",
        "units": "degrees_north",
    },
    "event_longitude": {
        "standard_name": "longitude",
        "long_name": "Longitude of the VHF source",
        "units": "degrees_east",
    },
    "event_altitude": {
        "standard_name": "altitude",
        "long_name": "Altitude of the VHF source above mean sea level",
        "units": "m",
        "positive": "up",
    },
    "event_chi2": {
        "long_name": "Reduced chi-square of the time-of-arrival solution",
        "units": "1",
    },
    "event_power": {"long_name": "Received power of the VHF source", "units": "dBW"},
    "event_mask": {"long_name": "Bit mask of the contributing stations"},
    "event_stations": {"long_name": "Number of contributing stations", "units": "1"},
    "station_latitude": {"standard_name": "latitude", "units": "degrees_north"},
    "station_longitude": {"standard_name": "longitude", "units": "degrees_east"},
    "station_altitude": {"standard_name": "altitude", "units": "m"},
    "network_center_latitude": {
        "standard_name": "latitude",
        "long_name": "Latitude of the network centre",
        "units": "degrees_north",
    },
    "network_center_longitude": {
        "standard_name": "longitude",
        "long_name": "Longitude of the network centre",
        "units": "degrees_east",
    },
    "network_center_altitude": {
        "standard_name": "altitude",
        "long_name": "Altitude of the network centre",
        "units": "m",
    },
}


def _use_compiled(engine):
    """Whether to run the compiled parser for the requested ``engine``."""
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled LMA parser is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


def _open_bytes(path):
    path = os.fspath(path)
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return f.read()
    if path.endswith(".bz2"):
        with bz2.open(path, "rb") as f:
            return f.read()
    with open(path, "rb") as f:
        return f.read()


def _header_value(header, key):
    m = re.search(rf"^{re.escape(key)}:\s*(.*)$", header, re.MULTILINE)
    return m.group(1).strip() if m else None


def _parse_header(header, path):
    """Start time, column names, network centre and stations of one file."""
    start = _header_value(header, "Data start time")
    if start is None:
        raise ValueError(f"{path}: no 'Data start time' in the LMA header")
    start = dt.datetime.strptime(start, "%m/%d/%y %H:%M:%S")
    names = _header_value(header, "Data")
    if names is None:
        raise ValueError(f"{path}: no 'Data:' column line in the LMA header")
    columns = [c.strip().lower() for c in names.split(",")]
    center = _header_value(header, "Coordinate center (lat,lon,alt)")
    center = [float(v) for v in center.split()] if center else [np.nan] * 3
    stations = []
    for line in re.findall(r"^Sta_info:\s*(.*)$", header, re.MULTILINE):
        # id, name (may contain blanks), lat, lon, alt, delay, board, channel
        parts = line.split()
        nums = parts[-6:]
        stations.append(
            (parts[0], " ".join(parts[1:-6]), *(float(v) for v in nums[:3]))
        )
    station_data = {}
    for line in re.findall(r"^Sta_data:\s*(.*)$", header, re.MULTILINE):
        parts = line.split()
        # id, name, win, dec_win, data_ver, rms_error, sources, %, <P/P_m>, active
        try:
            station_data[parts[0]] = (float(parts[-3]), parts[-1])
        except (ValueError, IndexError):  # pragma: no cover - unusual header
            pass
    return start, columns, center, stations, station_data


def _variables(columns):
    """Variable name for every data column (None for unknown columns)."""
    out = []
    for col in columns:
        name = None
        for var, pattern in _COLUMNS:
            if pattern.search(col) and var not in out:
                name = var
                break
        out.append(name)
    if "event_time" not in out or "event_latitude" not in out:
        raise ValueError(f"unrecognised LMA data columns: {columns}")
    return out


def _parse_numpy(body, ncol, hex_column):
    tokens = body.split()
    if len(tokens) % ncol:
        raise ValueError("a data line has fewer columns than expected")
    tokens = np.array(tokens, dtype=object).reshape(-1, ncol)
    keep = [j for j in range(ncol) if j != hex_column]
    values = tokens[:, keep].astype(np.float64)
    if hex_column >= 0:
        hexes = np.array([int(t, 16) for t in tokens[:, hex_column]], dtype=np.int64)
    else:
        hexes = np.full(len(tokens), -1, dtype=np.int64)
    return values, hexes


def _read_one(path, use_compiled, n_threads):
    raw = _open_bytes(path)
    pos = raw.find(_DATA_MARK)
    if pos < 0:
        raise ValueError(f"{path}: no '*** data ***' line; not an LMA ASCII file")
    header = raw[:pos].decode("latin-1")
    body = raw[raw.index(b"\n", pos) + 1 if b"\n" in raw[pos:] else len(raw) :]
    start, columns, center, stations, station_data = _parse_header(header, path)
    names = _variables(columns)
    hex_column = names.index("event_mask") if "event_mask" in names else -1
    if use_compiled:
        values, hexes = _lma.parse(body, len(names), hex_column, int(n_threads or 0))
    else:
        values, hexes = _parse_numpy(body, len(names), hex_column)
    float_names = [n for j, n in enumerate(names) if j != hex_column]
    data = {n: values[:, j] for j, n in enumerate(float_names) if n is not None}
    if hex_column >= 0:
        data["event_mask"] = hexes
    day = np.datetime64(start.date(), "ns")
    data["event_time"] = day + np.round(data["event_time"] * 1e9).astype(
        "timedelta64[ns]"
    )
    return data, header, center, stations, station_data


def _popcount(mask):
    """Number of set bits of every non-negative int64 mask."""
    m = np.where(mask < 0, 0, mask).astype(np.uint64)
    count = np.zeros(m.shape, dtype=np.int16)
    while np.any(m):
        count += (m & np.uint64(1)).astype(np.int16)
        m >>= np.uint64(1)
    return count


def read_lma(
    paths,
    *,
    max_chi2=None,
    min_stations=None,
    altitude=None,
    engine="auto",
    n_threads=None,
):
    """
    Read Lightning Mapping Array VHF source files into an xarray Dataset.

    Parameters
    ----------
    paths : str, path-like or list of them
        LMA ASCII ``.dat`` files (``.gz`` and ``.bz2`` are decompressed).
        Several files are concatenated in time order.
    max_chi2 : float, optional
        Keep only sources with a reduced chi-square at most this value
        (commonly 1 to 5). Default: keep all.
    min_stations : int, optional
        Keep only sources located by at least this many stations (commonly
        6, or 5 for small networks). Needs the station mask or a station
        count column. Default: keep all.
    altitude : (float, float), optional
        Keep only sources with ``altitude[0] <= alt <= altitude[1]`` (m above
        mean sea level), e.g. ``(0, 20e3)``. Default: keep all.
    engine : {"auto", "compiled", "numpy"}, optional
        Parser to use. ``"auto"`` (default) prefers the compiled kernel.
    n_threads : int, optional
        Threads of the compiled parser. Default: all cores.

    Returns
    -------
    xarray.Dataset
        Sources on ``number_of_events`` (sorted by time; ``event_time``,
        ``event_latitude`` and ``event_longitude`` are coordinates), the
        station table of the first file on ``number_of_stations`` and the
        network centre. ``attrs`` record the analysis program, the location
        name and the files read.

    Raises
    ------
    ValueError
        If a file is not an LMA ASCII file or its columns are not recognised.

    References
    ----------
    Thomas, R. J., P. R. Krehbiel, W. Rison, S. J. Hunyady, W. P. Winn,
    T. Hamlin, and J. Harlin, 2004: Accuracy of the Lightning Mapping Array.
    *J. Geophys. Res.*, **109**, D14207, https://doi.org/10.1029/2004JD004549

    Examples
    --------
    >>> import glob  # doctest: +SKIP
    >>> files = sorted(glob.glob("LYLOUT_220330_23*.dat.gz"))  # doctest: +SKIP
    >>> lma = radarx.io.read_lma(files, max_chi2=1.0, min_stations=6)  # doctest: +SKIP
    """
    use_compiled = _use_compiled(engine)
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    paths = list(paths)
    if not paths:
        raise ValueError("no LMA files given")
    parts = []
    first = None
    for path in paths:
        data, header, center, stations, station_data = _read_one(
            path, use_compiled, n_threads
        )
        if first is None:
            first = (header, center, stations, station_data)
        parts.append(data)
    keys = [k for k in parts[0] if all(k in p for p in parts)]
    data = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    if "event_mask" in data and "event_stations" not in data:
        data["event_stations"] = _popcount(data["event_mask"])

    keep = np.ones(data["event_time"].shape, dtype=bool)
    if max_chi2 is not None:
        if "event_chi2" not in data:
            raise ValueError("max_chi2 needs a chi-square column")
        keep &= data["event_chi2"] <= max_chi2
    if min_stations is not None:
        if "event_stations" not in data:
            raise ValueError("min_stations needs a station mask or count column")
        keep &= data["event_stations"] >= min_stations
    if altitude is not None:
        keep &= (data["event_altitude"] >= altitude[0]) & (
            data["event_altitude"] <= altitude[1]
        )
    order = np.argsort(data["event_time"][keep], kind="stable")
    data = {k: v[keep][order] for k, v in data.items()}

    header, center, stations, station_data = first
    dim = "number_of_events"
    coords = {
        k: (dim, data.pop(k), dict(_ATTRS[k]))
        for k in ("event_time", "event_latitude", "event_longitude")
    }
    variables = {k: (dim, v, dict(_ATTRS[k])) for k, v in data.items()}
    if "event_stations" in variables:
        variables["event_stations"] = (
            dim,
            np.asarray(data["event_stations"]).astype(np.int16),
            dict(_ATTRS["event_stations"]),
        )
    sdim = "number_of_stations"
    codes = [s[0] for s in stations]
    variables.update(
        {
            "station_code": (sdim, np.array(codes, dtype=str)),
            "station_name": (sdim, np.array([s[1] for s in stations], dtype=str)),
            "station_latitude": (
                sdim,
                np.array([s[2] for s in stations], dtype=float),
                dict(_ATTRS["station_latitude"]),
            ),
            "station_longitude": (
                sdim,
                np.array([s[3] for s in stations], dtype=float),
                dict(_ATTRS["station_longitude"]),
            ),
            "station_altitude": (
                sdim,
                np.array([s[4] for s in stations], dtype=float),
                dict(_ATTRS["station_altitude"]),
            ),
            "station_event_fraction": (
                sdim,
                np.array([station_data.get(c, (np.nan, ""))[0] / 100.0 for c in codes]),
                {"long_name": "Fraction of sources the station contributed to"},
            ),
            "station_active": (
                sdim,
                np.array([station_data.get(c, (0, ""))[1] == "A" for c in codes]),
                {"long_name": "Station active during the first file"},
            ),
        }
    )
    for key, value in zip(("latitude", "longitude", "altitude"), center):
        name = f"network_center_{key}"
        variables[name] = ((), float(value), dict(_ATTRS[name]))
    attrs = {
        "title": "Lightning Mapping Array VHF sources",
        "source": "VHF Lightning Mapping Array",
        "location": _header_value(header, "Location") or "",
        "event_algorithm_name": _header_value(header, "Analysis program") or "",
        "event_algorithm_version": _header_value(header, "Analysis program version")
        or "",
        "files": ", ".join(os.path.basename(os.fspath(p)) for p in paths),
    }
    return xr.Dataset(variables, coords=coords, attrs=attrs)
