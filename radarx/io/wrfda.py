#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx WRFDA Radar Observation Writer
=====================================

Writes the super-observations of :func:`radarx.retrieve.superobs` as the
``ob.radar`` text file that WRFDA reads for the assimilation of radar radial
velocity and reflectivity.

The file has fixed-width fields. The layout below is the one of the writer
this module comes from, which WRFDA 4.7 read without errors; it was not
checked here against the WRFDA reader (``da_read_obs_radar.inc``).

.. code-block:: text

    TOTAL NUMBER =  1                      count of radars in columns 15 to 17
    #-----------------#

    RADAR  KGWX         -88.329    33.897     179.0  2022-03-31_00:00:00   n_col n_lev
    #---...---#
    #  MM/DD/YYYY  lat  lon  elv  levels  height  rv  qc  err  rf  qc  err
    FM-128 RADAR   2022-03-31_00:00:00      lat  lon  elv  n_lev
       height  rv  qc  rv_error    rf  qc  rf_error        (one line per level)

``n_col`` and ``n_lev`` are the numbers of columns and levels (``I6`` each).
Each column gives the latitude and longitude of its super-observations, the
height of the terrain, and one line per layer with the layer-centre height
(m above sea level), the radial velocity (m s-1) and the reflectivity (dBZ),
each with a quality flag (0: used, -88: missing) and an error. Missing values
are -888888.0. A box without echo is written as the *non-precipitating* value
-999.99 with flag 0, so that WRFDA can remove model rain where the radar sees
none; ``nonprecip=None`` writes it as missing instead.

The lines are formatted in a compiled, multithreaded formatter (the same bytes
for any number of threads) or, if the extension is not built, in Python. For
100 000 observations in 9 850 columns it takes 0.44 s in Python, 0.08 s with the
compiled formatter on one thread and 0.06 s on ten; for 1 000 000 observations
(95 944 columns) about 5.5 s of CPU time in Python and 0.9 s compiled (machine
busy with other jobs). :func:`radarx.io.write_dart_obs_seq` writes the DART
format.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = ["write_wrfda_radar"]

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

MISSING = -888888.0
MISSING_QC = -88
NONPRECIP = -999.99
_RULE = "#" + "-" * 79 + "#"
_REQUIRED = (
    "reflectivity",
    "velocity",
    "reflectivity_error",
    "velocity_error",
    "nonprecip",
    "latitude",
    "longitude",
    "level",
    "radar_index",
    "column",
    "radar_name",
    "radar_latitude",
    "radar_longitude",
    "radar_altitude",
    "level_edges",
)


def _use_compiled(engine):
    if engine not in ("auto", "compiled", "python"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'python', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_FORMATTER:
        raise ImportError("the compiled formatter is not available")
    return HAS_COMPILED_FORMATTER and engine != "python"


_COLUMN = "FM-128 RADAR   %-19s  %12.3f  %12.3f  %8.1f  %6d\n"
_LEVEL = "   %12.1f%12.3f%4d%12.3f  %12.3f%4d%12.3f  \n"


def _format_columns_python(
    centres, col_lat, col_lon, col_elv, rv, rv_qc, rv_err, rf, rf_qc, rf_err, stamp
):
    """Column lines and level lines of one radar, formatted in Python."""
    nlev = len(centres)
    centres = centres.tolist()
    rows = [a.tolist() for a in (rv, rv_qc, rv_err, rf, rf_qc, rf_err)]
    out = []
    for i, head in enumerate(zip(col_lat.tolist(), col_lon.tolist(), col_elv.tolist())):
        out.append(_COLUMN % (stamp, *head, nlev))
        out.extend(_LEVEL % (centres[k], *(r[i][k] for r in rows)) for k in range(nlev))
    return "".join(out)


def _time_string(value):
    stamp = pd.Timestamp(value.replace("_", " ") if isinstance(value, str) else value)
    if pd.isna(stamp):
        raise ValueError("no time: pass time= or use a volume with ray times")
    return stamp.strftime("%Y-%m-%d_%H:%M:%S")


def write_wrfda_radar(
    superobs,
    path,
    *,
    time=None,
    nonprecip=NONPRECIP,
    missing_error=(2.0, 5.0),
    n_threads=None,
    engine="auto",
):
    """
    Write super-observations as a WRFDA ``ob.radar`` file.

    See :mod:`radarx.io.wrfda` for the layout of the file.

    Parameters
    ----------
    superobs : xarray.Dataset
        Result of :func:`radarx.retrieve.superobs`. Radars without
        observations are not written.
    path : str or os.PathLike
        File to write.
    time : str or datetime-like, optional
        Time of the observations, e.g. the analysis time. By default the
        ``radar_time`` of each radar (the middle of the volume). A string must
        be ``YYYY-MM-DD_HH:MM:SS`` or anything that pandas reads.
    nonprecip : float or None, optional
        Reflectivity written for boxes flagged ``nonprecip``. Default -999.99,
        the non-precipitating value; None writes them as missing.
    missing_error : tuple of float, optional
        Errors (velocity m s-1, reflectivity dBZ) written on lines whose value
        is missing, which WRFDA ignores. Default (2, 5).
    n_threads : int, optional
        Threads of the compiled formatter. Default: all cores.
    engine : {"auto", "compiled", "python"}, optional
        Formatter to use. ``"auto"`` (default) prefers the compiled one and
        falls back to Python; both write the same bytes.

    Returns
    -------
    dict
        ``radars``, ``columns``, ``velocities`` and ``reflectivities``: the
        numbers written (the last two count the values that are not missing,
        non-precipitating ones included in the reflectivities).

    Raises
    ------
    ValueError
        If ``superobs`` is not a result of :func:`radarx.retrieve.superobs`,
        has no observations, or has more than 999 radars.
    """
    if not isinstance(superobs, xr.Dataset) or any(
        v not in superobs.variables for v in _REQUIRED
    ):
        raise ValueError(
            "superobs must be the result of radarx.retrieve.superobs "
            f"(it needs {', '.join(_REQUIRED)})"
        )
    ds = superobs
    edges = np.asarray(ds["level_edges"].values, dtype=np.float64)
    nlev = edges.size - 1
    centres = 0.5 * (edges[:-1] + edges[1:])
    radar = np.asarray(ds["radar_index"].values)
    column = np.asarray(ds["column"].values)
    level = np.asarray(ds["level"].values)
    lat = np.asarray(ds["latitude"].values, dtype=np.float64)
    lon = np.asarray(ds["longitude"].values, dtype=np.float64)
    terrain = (
        np.asarray(ds["terrain_height"].values, dtype=np.float64)
        if "terrain_height" in ds
        else np.zeros(lat.shape)
    )
    rv, rf = (
        np.asarray(ds[n].values, dtype=np.float64) for n in ("velocity", "reflectivity")
    )
    rv_err, rf_err = (
        np.asarray(ds[n].values, dtype=np.float64)
        for n in ("velocity_error", "reflectivity_error")
    )
    flat = np.asarray(ds["nonprecip"].values).astype(bool)
    err_v, err_z = (float(e) for e in missing_error)

    used = [i for i in range(ds.sizes["radar"]) if np.any(radar == i)]
    if not used:
        raise ValueError("there are no observations to write")
    if len(used) > 999:
        raise ValueError("a WRFDA file holds at most 999 radars")
    use = _use_compiled(engine)

    lines = [f"TOTAL NUMBER ={len(used):3d}", "#-----------------#"]
    stats = {"radars": len(used), "columns": 0, "velocities": 0, "reflectivities": 0}
    chunks = []
    for i in used:
        in_radar = np.nonzero(radar == i)[0]
        name = str(ds["radar_name"].values[i])[:12]
        stamp = _time_string(ds["radar_time"].values[i] if time is None else time)

        # columns of the radar, from sorted ids, and their level tables
        ids, inv = np.unique(column[in_radar], return_inverse=True)
        inv = inv.ravel()
        ncol = ids.size
        count = np.bincount(inv, minlength=ncol)
        col_lat = np.bincount(inv, weights=lat[in_radar], minlength=ncol) / count
        col_lon = np.bincount(inv, weights=lon[in_radar], minlength=ncol) / count
        terr = terrain[in_radar]
        known = np.isfinite(terr)
        n_terr = np.bincount(inv, weights=known, minlength=ncol)
        col_elv = np.bincount(inv, weights=np.where(known, terr, 0.0), minlength=ncol)
        with np.errstate(invalid="ignore", divide="ignore"):
            col_elv = col_elv / n_terr
        lv = level[in_radar]

        def table(values, fill):
            out = np.full((ncol, nlev), fill)
            out[inv, lv] = values[in_radar]
            return out

        present = np.zeros((ncol, nlev), bool)
        present[inv, lv] = True
        rv_t, rf_t = table(rv, np.nan), table(rf, np.nan)
        rve_t, rfe_t = table(rv_err, np.nan), table(rf_err, np.nan)
        dry = present & table(flat, False).astype(bool) & (nonprecip is not None)
        v_ok = present & np.isfinite(rv_t)
        z_ok = dry | (present & np.isfinite(rf_t))
        z_val = np.where(dry, float(nonprecip) if nonprecip is not None else 0.0, rf_t)
        z_err = np.where(z_ok & np.isfinite(rfe_t), rfe_t, err_z)
        v_err = np.where(v_ok & np.isfinite(rve_t), rve_t, err_v)
        v_val = np.where(v_ok, rv_t, MISSING)
        z_val = np.where(z_ok, z_val, MISSING)
        v_qc = np.where(v_ok, 0, MISSING_QC)
        z_qc = np.where(z_ok, 0, MISSING_QC)
        stats["columns"] += ncol
        stats["velocities"] += int(v_ok.sum())
        stats["reflectivities"] += int(z_ok.sum())

        lines += [
            "",
            f"{'RADAR':5s}  {name:12s}{float(ds['radar_longitude'].values[i]):8.3f}  "
            f"{float(ds['radar_latitude'].values[i]):8.3f}  "
            f"{float(ds['radar_altitude'].values[i]):8.1f}  {stamp:19s}"
            f"{ncol:6d}{nlev:6d}",
            _RULE,
            "#  MM/DD/YYYY  lat  lon  elv  levels  height  rv  qc  err  rf  qc  err",
        ]
        args = (
            centres,
            col_lat,
            col_lon,
            np.nan_to_num(col_elv, nan=0.0),
            v_val,
            v_qc.astype(np.int64),
            v_err,
            z_val,
            z_qc.astype(np.int64),
            z_err,
            stamp,
        )
        chunks.append(("\n".join(lines) + "\n").encode("ascii"))
        lines = []
        if use:
            chunks.append(_obswriter.format_wrfda(*args, n_threads=int(n_threads or 0)))
        else:
            chunks.append(_format_columns_python(*args).encode("ascii"))
    with open(path, "wb") as handle:
        handle.write(b"".join(chunks))
    return stats
