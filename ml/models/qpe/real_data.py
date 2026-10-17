#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Compare the rain rate network with the physical estimators and with gauges on
real NEXRAD volumes.

    python real_data.py radarx-qpe-v1.onnx gauges.csv out_dir KGWX*_V06

The volumes (public ``unidata-nexrad-level2`` AWS bucket) are read with xradar;
the no-data codes are masked (DBZH <= -32, ZDR <= -12.9, RHOHV <= 0.21). For
the lowest sweep that has a differential phase, K_DP comes from
``estimate_kdp``, and the rain rate from R(Z), from a hybrid (R(K_DP) where
K_DP is at least 0.3 degree/km, R(Z, ZDR) elsewhere) and from the network,
on the gates that are rain by ``rhohv >= 0.95`` and ``dbz >= 10``. The rates are
averaged on a 1 km grid, integrated linearly in time over the hours of the gauge
file and read at the gauges. ``gauges.csv`` is a request of the Iowa Environmental
Mesonet ASOS service (columns ``station, valid, lon, lat, p01i`` in inches and ``metar``, reports
near the hour). Prints the bias of the total, the normalized mean absolute error
and the correlation for every estimator, and writes ``real_data.json``.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyproj
import xarray as xr
import xradar as xd

from radarx import ml
from radarx.retrieve import (
    estimate_kdp,
    rain_rate_kdp,
    rain_rate_ml,
    rain_rate_z,
    rain_rate_z_zdr,
)

EDGES = np.arange(-150e3, 150.1e3, 1000.0)


def rate_fields(path, model):
    """Gridded rain rates (mm/h) of one volume by estimator, and the volume time."""
    stamp = re.search(r"(\d{8})_(\d{6})", path).groups()
    time = np.datetime64(
        f"{stamp[0][:4]}-{stamp[0][4:6]}-{stamp[0][6:]}T"
        f"{stamp[1][:2]}:{stamp[1][2:4]}:{stamp[1][4:]}"
    )
    dtree = xd.io.open_nexradlevel2_datatree(
        path, sweep=[0, 1, 2, 3]
    ).xradar.georeference()
    for name in sorted(n for n in dtree.children if n.startswith("sweep")):
        ds = dtree[name].to_dataset()
        if "PHIDP" in ds and "ZDR" in ds:
            break
    else:
        return time, None
    ds["DBZH"] = ds.DBZH.where(ds.DBZH > -32)
    ds["ZDR"] = ds.ZDR.where(ds.ZDR > -12.9)
    ds["RHOHV"] = ds.RHOHV.where(ds.RHOHV > 0.21)
    ds["KDP"] = estimate_kdp(ds)["KDP"]
    rain = (ds.RHOHV >= 0.95) & (ds.DBZH >= 10)
    r_z = rain_rate_z(ds).where(rain)
    r_zdr = rain_rate_z_zdr(ds).where(rain)
    r_kdp = rain_rate_kdp(ds, negative="zero").where(rain)
    hybrid = xr.where(ds.KDP >= 0.3, r_kdp, r_zdr)
    r_ml = rain_rate_ml(ds, model=model, mask=rain).RAIN_RATE_ML
    out = {"R(Z)": r_z, "hybrid": hybrid, "network": r_ml}
    gx, gy = ds.x.values.ravel(), ds.y.values.ravel()
    grids = {}
    for key, field in out.items():
        v = field.fillna(0.0).where(rain, 0.0).values.ravel()
        ok = np.isfinite(v) & (np.abs(gx) < 150e3) & (np.abs(gy) < 150e3)
        s, _, _ = np.histogram2d(gy[ok], gx[ok], bins=[EDGES, EDGES], weights=v[ok])
        n, _, _ = np.histogram2d(gy[ok], gx[ok], bins=[EDGES, EDGES])
        grids[key] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return time, grids


def main(model_file, gauge_file, out_dir, *files):
    ml.register_model(
        "qpe-real-data",
        model_file,
        None,
        "MIT",
        "radarx",
        features=["dbz", "zdr", "kdp", "rhohv", "wavelength"],
        overwrite=True,
    )
    times, series = [], []
    for f in sorted(files):
        t, g = rate_fields(f, "qpe-real-data")
        if g is not None:
            times.append(t)
            series.append(g)
    times = np.array(times)
    rows = list(csv.DictReader(open(gauge_file)))
    # keep the stations whose precipitation sensor reports (a P group in two or
    # more reports); AWOS stations without one give 0.00 in rain
    groups = {}
    for r in rows:
        if re.search(r"\sP\d{4}(\s|$)", r.get("metar", "")):
            groups[r["station"]] = groups.get(r["station"], 0) + 1
    rows = [r for r in rows if groups.get(r["station"], 0) >= 2]
    lat0 = xd.io.open_nexradlevel2_datatree(files[0], sweep=[0]).latitude.item()
    lon0 = xd.io.open_nexradlevel2_datatree(files[0], sweep=[0]).longitude.item()
    tr = pyproj.Transformer.from_crs(
        "EPSG:4326",
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84",
        always_xy=True,
    )
    hours = sorted(
        {
            np.datetime64(r["valid"].replace(" ", "T")).astype("datetime64[h]")
            + np.timedelta64(55, "m")
            for r in rows
        }
    )
    hours = [
        h
        for h in hours
        if times.min() <= h - np.timedelta64(60, "m") and h <= times.max()
    ]
    result = {k: {"est": [], "obs": []} for k in series[0]}
    for h in hours:
        t0 = h - np.timedelta64(60, "m")
        grid_t = np.linspace(0, 1, 121)
        tt = (times - t0) / np.timedelta64(1, "s") / 3600.0
        idx = np.clip(np.searchsorted(tt, grid_t), 1, len(tt) - 1)
        w = np.clip((grid_t - tt[idx - 1]) / (tt[idx] - tt[idx - 1]), 0, 1)
        for r in rows:
            t = np.datetime64(r["valid"].replace(" ", "T"))
            if abs((t - h) / np.timedelta64(1, "m")) > 5 or r["p01i"] in ("M", ""):
                continue
            obs = 0.0 if r["p01i"] == "T" else float(r["p01i"]) * 25.4
            x, y = tr.transform(float(r["lon"]), float(r["lat"]))
            if np.hypot(x, y) > 140e3:
                continue
            i, j = int((y + 150e3) // 1000), int((x + 150e3) // 1000)
            for key in result:
                vals = np.array(
                    [
                        [
                            s[key][a, b]
                            for a in range(i - 1, i + 2)
                            for b in range(j - 1, j + 2)
                        ]
                        for s in series
                    ]
                )
                v = np.nanmean(vals, axis=1)
                est = float(np.nanmean(v[idx - 1] * (1 - w) + v[idx] * w))
                result[key]["est"].append(est)
                result[key]["obs"].append(obs)
    summary = {}
    for key, d in result.items():
        e, o = np.array(d["est"]), np.array(d["obs"])
        ok = np.isfinite(e)
        summary[key] = {
            "n": int(ok.sum()),
            "bias_percent": float(100 * (e[ok].sum() / o[ok].sum() - 1)),
            "nmae": float(np.abs(e[ok] - o[ok]).sum() / o[ok].sum()),
            "corr": float(np.corrcoef(e[ok], o[ok])[0, 1]),
        }
        print(key, summary[key])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "real_data.json").write_text(
        json.dumps({"summary": summary, "pairs": result}, indent=1)
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
