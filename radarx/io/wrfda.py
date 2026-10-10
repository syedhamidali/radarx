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


def _time_string(value):
    stamp = pd.Timestamp(value.replace("_", " ") if isinstance(value, str) else value)
    if pd.isna(stamp):
        raise ValueError("no time: pass time= or use a volume with ray times")
    return stamp.strftime("%Y-%m-%d_%H:%M:%S")


def write_wrfda_radar(
    superobs, path, *, time=None, nonprecip=NONPRECIP, missing_error=(2.0, 5.0)
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

    lines = [f"TOTAL NUMBER ={len(used):3d}", "#-----------------#"]
    stats = {"radars": len(used), "columns": 0, "velocities": 0, "reflectivities": 0}
    for i in used:
        in_radar = np.nonzero(radar == i)[0]
        name = str(ds["radar_name"].values[i])[:12]
        stamp = _time_string(ds["radar_time"].values[i] if time is None else time)
        columns = np.unique(column[in_radar])
        lines += [
            "",
            f"{'RADAR':5s}  {name:12s}{float(ds['radar_longitude'].values[i]):8.3f}  "
            f"{float(ds['radar_latitude'].values[i]):8.3f}  "
            f"{float(ds['radar_altitude'].values[i]):8.1f}  {stamp:19s}"
            f"{columns.size:6d}{nlev:6d}",
            _RULE,
            "#  MM/DD/YYYY  lat  lon  elv  levels  height  rv  qc  err  rf  qc  err",
        ]
        for col in columns:
            rows = in_radar[column[in_radar] == col]
            lines.append(
                f"{'FM-128 RADAR':12s}   {stamp:19s}  {lat[rows].mean():12.3f}  "
                f"{lon[rows].mean():12.3f}  {np.nanmean(terrain[rows]):8.1f}  {nlev:6d}"
            )
            by_level = {int(level[r]): r for r in rows}
            for k in range(nlev):
                r = by_level.get(k)
                v_ok = r is not None and np.isfinite(rv[r])
                if r is not None and flat[r] and nonprecip is not None:
                    z_val, z_qc, z_err = float(nonprecip), 0, rf_err[r]
                elif r is not None and np.isfinite(rf[r]):
                    z_val, z_qc, z_err = rf[r], 0, rf_err[r]
                else:
                    z_val, z_qc, z_err = MISSING, MISSING_QC, err_z
                if not np.isfinite(z_err):
                    z_err = err_z
                v_val, v_qc, v_e = (
                    (rv[r], 0, rv_err[r]) if v_ok else (MISSING, MISSING_QC, err_v)
                )
                if not np.isfinite(v_e):
                    v_e = err_v
                stats["velocities"] += int(v_ok)
                stats["reflectivities"] += int(z_qc == 0)
                lines.append(
                    f"   {centres[k]:12.1f}{v_val:12.3f}{v_qc:4d}{v_e:12.3f}  "
                    f"{z_val:12.3f}{z_qc:4d}{z_err:12.3f}  "
                )
            stats["columns"] += 1
    with open(path, "w", encoding="ascii") as handle:
        handle.write("\n".join(lines) + "\n")
    return stats
