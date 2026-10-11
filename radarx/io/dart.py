#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx DART Radar Observation Writer
====================================

Writes the super-observations of :func:`radarx.retrieve.superobs` as an ASCII
``obs_seq`` file of the Data Assimilation Research Testbed (DART).

The layout was written from the DART source and documentation (the files of
the ``NCAR/DART`` repository named below), not from memory. **The file was not
tested with DART itself**, which is not installed where radarx is developed:
the tests read the file back with a reader that uses only the layout below.

Layout (``obs_sequence_mod.f90``, ``write_obs_seq`` and ``write_obs``)::

    obs_sequence
    obs_type_definitions
              3
              1 DOPPLER_RADIAL_VELOCITY
              2 RADAR_REFLECTIVITY
              3 RADAR_CLEARAIR_REFLECTIVITY
      num_copies:            1  num_qc:            1
      num_obs:            4  max_num_obs:            4
    observation
    Data QC
      first:            1  last:            4
     OBS            1
       value
       qc
                -1           2          -1        previous, next, covariance group
    obdef
    loc3d
       lon  lat  height  3                          radians, radians, m, VERTISHEIGHT
    kind
               1
    platform                                        radial velocity only
    loc3d
       radar lon  radar lat  radar height  3
    dir3d
       beam_direction(1) (2) (3)
       nyquist_velocity
       velocity key
        secs        days
       error variance

The numbers are written in free format, which DART reads in free format, so
only the order of the items matters.

Fields
------

* **Types** (``obs_def_radar_mod.f90``): ``DOPPLER_RADIAL_VELOCITY``,
  ``RADAR_REFLECTIVITY`` and ``RADAR_CLEARAIR_REFLECTIVITY``. The names in the
  table of the file are mapped by DART to the types of the program that reads
  the file, so the program has to be built with ``obs_def_radar_mod``. The
  README of the DART radar converter describes the clear-air type as
  "clear-air reflectivity observations where no radar echoes are observed"; it
  is used here for the boxes flagged ``nonprecip``.
* **Location**: longitude from 0 to 2 pi and latitude, both in radians, and the
  height above sea level in m with the vertical coordinate code 3
  (``VERTISHEIGHT``), at the mean position of the gates of the box.
* **Radial velocity** (m s-1, positive away from the radar) carries the
  position of the radar (height in m, code 3, as required by DART), the
  ``beam_direction`` and the Nyquist velocity. DART defines the beam direction
  as ``(sin az cos el, cos az cos el, sin el)`` and projects the model wind as
  ``b1 u + b2 v + b3 (w - fall speed)``: this is the mean radial unit vector
  (``radial_east``, ``radial_north``, ``radial_up``) of the super-observation,
  which is not normalised after the averaging. The key at the end of the block
  is written as the number of the velocity record; DART discards it on reading.
* **Reflectivity** is in dBZ. The documentation does not give the value of a
  clear-air observation; it is the argument ``clearair_dbz`` (default -10, the
  default of ``lowest_reflectivity_obs`` of DART).
* **Error**: DART takes the error *variance* (``(m/s)^2`` and ``dBZ^2``), so the
  error estimates of ``superobs`` are squared.
* **Time**: ``secs days`` (``(i6,1x,i10)``), the days counted from 1 January
  1601 of the Gregorian calendar (``time_manager_mod.f90``), as DART needs
  with ``calendar = 'Gregorian'`` in ``input.nml``. Records are written in time
  order and the previous and next keys link them in that order; the first
  record has previous key -1 and the last next key -1.
* **Copies**: one copy named ``observation`` (DART looks for that word) and one
  quality-control field named ``Data QC`` (a name with ``QC``), always 0.

Not in the Dataset and therefore arguments: the Nyquist velocity
(``nyquist_velocity``; ``None`` writes DART's missing value -888888, with
which DART does not unfold the velocity, as its own converter does for
synthetic observations) and the value of clear-air reflectivity.

Speed
-----

The records are formatted in a compiled, multithreaded formatter (blocks of
records are formatted into their own buffers and joined in order, so the file is
the same for any number of threads); a Python formatter writes the same bytes
and is used if the extension is not built. Time to write the records of 100 000
super-observation boxes (172 903 records, 59 MB) and of 1 000 000 boxes
(1 739 911 records, 599 MB), tiled from KGWX super-observations, on a 10-core
laptop that was busy with other jobs (best of three, wall time):

======================  ===============  ===============
formatter               100 000 boxes    1 000 000 boxes
======================  ===============  ===============
Python                  1.16 s           12.2 s
compiled, 1 thread      0.16 s           1.9 s
compiled, 10 threads    0.08 s           1.9 s
======================  ===============  ===============

The time of the large file is mostly the NumPy preparation of the records
(0.4 s) and writing 600 MB. :func:`radarx.io.write_wrfda_radar` takes 0.44 s,
0.08 s and 0.06 s for 100 000 observations (9 850 columns) with the same three
formatters.

References
----------
El Gharamti, M., H. Kershaw, K. Raeder, B. Raczka, B. Johnson, M. Smith, J.
Anderson, D. Amrhein, N. Collins, T. Hoar, B. Gaubert, I. Grooms, and L.
Kugler, 2025: The Data Assimilation Research Testbed: A robust, scalable
software facility with groundbreaking capabilities for model-data integration.
*Bull. Amer. Meteor. Soc.*, **106** (11), E2328-E2345,
https://doi.org/10.1175/BAMS-D-24-0214.1

Anderson, J., T. Hoar, K. Raeder, H. Liu, N. Collins, R. Torn, and A.
Arellano, 2009: The Data Assimilation Research Testbed: A community facility.
*Bull. Amer. Meteor. Soc.*, **90** (9), 1283-1296,
https://doi.org/10.1175/2009BAMS2618.1

Data Assimilation Research Testbed (software). NSF NCAR Computational and
Information Systems Laboratory, https://doi.org/10.5065/D6WQ0202

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = ["write_dart_obs_seq"]

__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

import numpy as np
import pandas as pd
import xarray as xr

try:
    from . import _obswriter

    HAS_COMPILED_FORMATTER = True
except ImportError:  # pragma: no cover - depends on the build
    _obswriter = None
    HAS_COMPILED_FORMATTER = False

MISSING_R8 = -888888.0
VERTISHEIGHT = 3
DOPPLER_RADIAL_VELOCITY = "DOPPLER_RADIAL_VELOCITY"
RADAR_REFLECTIVITY = "RADAR_REFLECTIVITY"
RADAR_CLEARAIR_REFLECTIVITY = "RADAR_CLEARAIR_REFLECTIVITY"
_EPOCH = np.datetime64("1601-01-01T00:00:00", "s")
_REQUIRED = (
    "reflectivity",
    "velocity",
    "reflectivity_error",
    "velocity_error",
    "nonprecip",
    "latitude",
    "longitude",
    "height",
    "radial_east",
    "radial_north",
    "radial_up",
    "radar_index",
    "radar_latitude",
    "radar_longitude",
    "radar_altitude",
    "radar_time",
)

# kinds of a record: 0 velocity, 1 reflectivity, 2 clear-air reflectivity
_NAMES = (DOPPLER_RADIAL_VELOCITY, RADAR_REFLECTIVITY, RADAR_CLEARAIR_REFLECTIVITY)

_E = "%25.16E"
_LOC = f" {_E} {_E} {_E} %2d\n"
_HEAD = " OBS %12d\n" + _E + "\n%12d\n%12d%12d%12d\nobdef\nloc3d\n" + _LOC
_VEL = (
    _HEAD
    + "kind\n%12d\nplatform\nloc3d\n"
    + _LOC
    + f"dir3d\n {_E} {_E} {_E}\n {_E}\n%12d\n%6d %10d\n"
    + _E
    + "\n"
)
_REF = _HEAD + "kind\n%12d\n%6d %10d\n" + _E + "\n"


def _use_compiled(engine):
    if engine not in ("auto", "compiled", "python"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'python', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_FORMATTER:
        raise ImportError("the compiled formatter is not available")
    return HAS_COMPILED_FORMATTER and engine != "python"


def _format_compiled(rec, n_threads):
    """Bytes of the observations, formatted by the compiled formatter."""
    return _obswriter.format_dart(
        rec["kind"],
        rec["value"],
        rec["variance"],
        rec["lon"],
        rec["lat"],
        rec["height"],
        rec["days"],
        rec["secs"],
        rec["radar_lon"],
        rec["radar_lat"],
        rec["radar_height"],
        np.ascontiguousarray(rec["beam"]),
        rec["nyquist"],
        rec["velkey"],
        VERTISHEIGHT,
        n_threads=int(n_threads or 0),
    )


def _time_seconds(superobs, time):
    """Seconds since 1601-01-01 of every observation."""
    if time is None:
        stamps = np.asarray(superobs["radar_time"].values, dtype="datetime64[ns]")[
            np.asarray(superobs["radar_index"].values)
        ]
    else:
        stamp = pd.Timestamp(time.replace("_", " ") if isinstance(time, str) else time)
        stamps = np.full(superobs.sizes["obs"], stamp.to_datetime64(), "datetime64[ns]")
    if np.isnat(stamps).any():
        raise ValueError("no time: pass time= or use a volume with ray times")
    # whole seconds: 1601 is more than 292 years before the data, the limit of nanoseconds
    return (stamps.astype("datetime64[s]") - _EPOCH).astype(np.int64)


def _records(superobs, time, clear_air, clearair_dbz, nyquist_velocity):
    """Arrays of the records, in file order, for the formatters."""
    ds = superobs
    values = {n: np.asarray(ds[n].values, dtype=np.float64) for n in _REQUIRED[:4]}
    nonprecip = np.asarray(ds["nonprecip"].values).astype(bool)
    lat = np.asarray(ds["latitude"].values, dtype=np.float64)
    lon = np.asarray(ds["longitude"].values, dtype=np.float64)
    height = np.asarray(ds["height"].values, dtype=np.float64)
    radar = np.asarray(ds["radar_index"].values).astype(np.int64)
    beam = np.stack(
        [np.asarray(ds[n].values, dtype=np.float64) for n in _REQUIRED[8:11]], axis=1
    )
    seconds = _time_seconds(ds, time)

    has_v = np.isfinite(values["velocity"])
    has_z = np.isfinite(values["reflectivity"]) & ~nonprecip
    has_c = nonprecip & bool(clear_air)
    if np.any(has_v & ~np.isfinite(values["velocity_error"])):
        raise ValueError("a velocity has no error estimate")
    if np.any((has_z | has_c) & ~np.isfinite(values["reflectivity_error"])):
        raise ValueError("a reflectivity has no error estimate")

    for name, a in (("latitude", lat), ("longitude", lon), ("height", height)):
        if not np.all(np.isfinite(a)):
            raise ValueError(f"every box needs a finite {name}")
    if np.any(~np.isfinite(beam[has_v])):
        raise ValueError("a velocity has no radial unit vector")

    rank = np.empty(seconds.size, np.int64)
    rank[np.argsort(seconds, kind="stable")] = np.arange(seconds.size)
    first = np.nonzero(has_v)[0]
    second = np.nonzero(has_z | has_c)[0]
    obs = np.concatenate([first, second])
    slot = np.concatenate(
        [np.zeros(first.size, np.int64), np.ones(second.size, np.int64)]
    )
    order = np.lexsort((slot, rank[obs]))
    obs, slot = obs[order], slot[order]
    kind = np.where(slot == 0, 0, np.where(has_c[obs], 2, 1)).astype(np.int64)

    value = np.where(
        kind == 0,
        values["velocity"][obs],
        np.where(kind == 2, float(clearair_dbz), values["reflectivity"][obs]),
    )
    error = np.where(
        kind == 0, values["velocity_error"][obs], values["reflectivity_error"][obs]
    )
    deg = np.pi / 180.0
    r = radar[obs]
    ny = np.broadcast_to(
        np.asarray(
            MISSING_R8 if nyquist_velocity is None else nyquist_velocity, np.float64
        ),
        (np.asarray(ds["radar_latitude"].values).size,),
    )
    if np.any((ny < 0) & (ny != MISSING_R8)):
        raise ValueError("the Nyquist velocity must not be negative")
    rec = {
        "kind": kind,
        "value": value,
        "variance": error * error,
        "lon": np.mod(lon[obs], 360.0) * deg,
        "lat": lat[obs] * deg,
        "height": height[obs],
        "days": seconds[obs] // 86400,
        "secs": seconds[obs] % 86400,
        "radar_lon": np.mod(
            np.asarray(ds["radar_longitude"].values, np.float64)[r], 360.0
        )
        * deg,
        "radar_lat": np.asarray(ds["radar_latitude"].values, np.float64)[r] * deg,
        "radar_height": np.asarray(ds["radar_altitude"].values, np.float64)[r],
        "beam": np.clip(beam[obs], -1.0, 1.0),
        "nyquist": ny[r],
    }
    rec["velkey"] = np.cumsum(kind == 0)
    return rec


def _header(rec, copy_name, qc_name):
    used = [k for k in range(3) if np.any(rec["kind"] == k)]
    lines = ["obs_sequence", "obs_type_definitions", f"{len(used):12d}"]
    lines += [f"{k + 1:12d} {_NAMES[k]}" for k in used]
    n = rec["kind"].size
    lines += [
        f"  num_copies: {1:12d}  num_qc: {1:12d}",
        f"  num_obs: {n:12d}  max_num_obs: {n:12d}",
        copy_name,
        qc_name,
        f"  first: {1 if n else -1:12d}  last: {n if n else -1:12d}",
    ]
    return "\n".join(lines) + "\n"


def _format_python(rec):
    """Text of the observations, formatted in Python."""
    n = rec["kind"].size
    key = np.arange(1, n + 1)
    prev = key - 1
    prev[:1] = -1
    nxt = key + 1
    nxt[-1:] = -1
    common = [
        key,
        rec["value"],
        np.zeros(n, np.int64),
        prev,
        nxt,
        np.full(n, -1),
        rec["lon"],
        rec["lat"],
        rec["height"],
        np.full(n, VERTISHEIGHT),
        rec["kind"] + 1,
    ]
    out = []
    for i in range(n):
        kind = int(rec["kind"][i])
        row = [c[i] for c in common]
        if kind == 0:
            row += [
                rec["radar_lon"][i],
                rec["radar_lat"][i],
                rec["radar_height"][i],
                VERTISHEIGHT,
                rec["beam"][i, 0],
                rec["beam"][i, 1],
                rec["beam"][i, 2],
                rec["nyquist"][i],
                rec["velkey"][i],
            ]
            template = _VEL
        else:
            template = _REF
        row += [rec["secs"][i], rec["days"][i], rec["variance"][i]]
        out.append(template % tuple(row))
    return "".join(out)


def write_dart_obs_seq(
    superobs,
    path,
    *,
    time=None,
    clear_air=True,
    clearair_dbz=-10.0,
    nyquist_velocity=None,
    copy_name="observation",
    qc_name="Data QC",
    n_threads=None,
    engine="auto",
):
    """
    Write super-observations as a DART ``obs_seq`` file.

    See :mod:`radarx.io.dart` for the layout of the file and the source of
    every field. The file was implemented from the DART source and
    documentation and **was not tested with DART itself**.

    Parameters
    ----------
    superobs : xarray.Dataset
        Result of :func:`radarx.retrieve.superobs`.
    path : str or os.PathLike
        File to write.
    time : str or datetime-like, optional
        Time of all observations, e.g. the analysis time. By default the
        ``radar_time`` of each radar.
    clear_air : bool, optional
        Write the boxes flagged ``nonprecip`` as ``RADAR_CLEARAIR_REFLECTIVITY``
        observations. Default True.
    clearair_dbz : float, optional
        Value (dBZ) of these observations. The DART documentation does not
        define it. Default -10, the default ``lowest_reflectivity_obs`` of DART.
    nyquist_velocity : float or array-like, optional
        Nyquist velocity (m s-1) of the velocities, one value or one per radar.
        The Dataset does not hold it. Default None: DART's missing value
        (-888888), with which DART does not unfold the velocities.
    copy_name, qc_name : str, optional
        Metadata strings of the observation copy and of the quality-control
        field (at most 64 characters). DART looks for ``observation`` in the
        first and ``QC`` in the second.
    n_threads : int, optional
        Threads of the compiled formatter. Default: all cores.
    engine : {"auto", "compiled", "python"}, optional
        Formatter to use. ``"auto"`` (default) prefers the compiled one and
        falls back to Python; both write the same bytes.

    Returns
    -------
    dict
        ``observations``, ``velocities``, ``reflectivities`` and
        ``clearair`` (the numbers of records written).

    Raises
    ------
    ValueError
        If ``superobs`` is not a result of :func:`radarx.retrieve.superobs`, an
        error estimate or the time is missing, or a name is too long.

    Notes
    -----
    Timings of the formatting, for records tiled from the KGWX
    super-observations, are in the notebook of the super-observations.

    References
    ----------
    El Gharamti, M., H. Kershaw, K. Raeder, B. Raczka, B. Johnson, M. Smith, J.
    Anderson, D. Amrhein, N. Collins, T. Hoar, B. Gaubert, I. Grooms, and L.
    Kugler, 2025: The Data Assimilation Research Testbed: A robust, scalable
    software facility with groundbreaking capabilities for model-data
    integration. *Bull. Amer. Meteor. Soc.*, **106** (11), E2328-E2345,
    https://doi.org/10.1175/BAMS-D-24-0214.1

    Anderson, J., T. Hoar, K. Raeder, H. Liu, N. Collins, R. Torn, and A.
    Arellano, 2009: The Data Assimilation Research Testbed: A community
    facility. *Bull. Amer. Meteor. Soc.*, **90** (9), 1283-1296,
    https://doi.org/10.1175/2009BAMS2618.1

    Data Assimilation Research Testbed (software). NSF NCAR Computational and
    Information Systems Laboratory, https://doi.org/10.5065/D6WQ0202
    """
    if not isinstance(superobs, xr.Dataset) or any(
        v not in superobs.variables for v in _REQUIRED
    ):
        raise ValueError(
            "superobs must be the result of radarx.retrieve.superobs "
            f"(it needs {', '.join(_REQUIRED)})"
        )
    for name in (copy_name, qc_name):
        if not name.strip() or len(name.strip()) > 64 or "\n" in name:
            raise ValueError("metadata names must have 1 to 64 characters on one line")
    use = _use_compiled(engine)
    rec = _records(superobs, time, clear_air, clearair_dbz, nyquist_velocity)
    body = (
        _format_compiled(rec, n_threads) if use else _format_python(rec).encode("ascii")
    )
    with open(path, "wb") as handle:
        handle.write(_header(rec, copy_name.strip(), qc_name.strip()).encode("ascii"))
        handle.write(body)
    kind = rec["kind"]
    return {
        "observations": int(kind.size),
        "velocities": int((kind == 0).sum()),
        "reflectivities": int((kind == 1).sum()),
        "clearair": int((kind == 2).sum()),
    }
