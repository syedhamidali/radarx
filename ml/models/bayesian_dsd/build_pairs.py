"""
Build naive and trajectory-matched radar-disdrometer pairs for one IOP.

    python build_pairs.py IOP2 --radar-dir DIR --era5 FILE --out DIR

Radar volumes: NEXRAD level-2 or CfRadial files of the lowest sweep. Storm
motion: median of radarx.retrieve.estimate_motion between consecutive
volumes. Wind: layer mean of an ERA5 profile from radarx.io.sounding (or the
disdrometer's surface wind).
"""

from __future__ import annotations

import argparse
import glob
import os

import match
import numpy as np
import pips
import radar
import xarray as xr

from radarx.retrieve import estimate_motion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("iop")
    ap.add_argument("--radar-dir", required=True)
    ap.add_argument("--era5", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra-sweeps", type=int, nargs="*", default=[1, 2])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    probes = [pips.load(f) for f in pips.files(a.iop)]
    sites = {
        raw.attrs["probe"]: (raw.attrs["latitude"], raw.attrs["longitude"])
        for raw, _ in probes
    }
    t0 = min(raw.time.values[0] for raw, _ in probes) - np.timedelta64(20, "m")
    t1 = max(raw.time.values[-1] for raw, _ in probes) + np.timedelta64(5, "m")
    files = sorted(glob.glob(os.path.join(a.radar_dir, "K???2022*_V06*")))
    files = [
        f for f in files if not f.endswith(".part") and t0 <= radar.volume_time(f) <= t1
    ]
    print(len(files), "volumes", flush=True)
    cache = os.path.join(a.out, "cache")
    vols = [radar.process(f, sites, cache) for f in files]
    pts_list = [v[0] for v in vols]
    for sweep in a.extra_sweeps:
        pts_list += [radar.process(f, sites, cache, sweep)[0] for f in files]
    grids = [v[1] for v in vols]

    # storm motion from consecutive volumes
    motions = []
    for g0, g1 in zip(grids[:-1], grids[1:]):
        dt = (
            np.datetime64(g1.attrs["volume_time"])
            - np.datetime64(g0.attrs["volume_time"])
        ) / np.timedelta64(1, "s")
        try:
            m = estimate_motion(
                g0.DBZH.fillna(0.0).to_dataset(name="DBZH"),
                g1.DBZH.fillna(0.0).to_dataset(name="DBZH"),
                "DBZH",
                dt=dt,
                observed=g0.observed & g1.observed,
            )
        except Exception as err:  # pragma: no cover - diagnostics only
            print("motion failed", err)
            continue
        u, v = float(m["u"]), float(m["v"])
        motions.append((u, v, float(m["quality"])))
    motions = np.array(motions)
    good = motions[:, 2] >= 0.3
    motion = np.nanmedian(motions[good, :2], axis=0)
    print("storm motion", motion, "from", len(motions), flush=True)

    profile = xr.open_dataset(a.era5) if a.era5 else None
    lat0 = pts_list[0].attrs["radar_latitude"]
    lon0 = pts_list[0].attrs["radar_longitude"]
    out_n, out_t = [], []
    for raw, _ in probes:
        xy = radar.project(lat0, lon0, raw.attrs["latitude"], raw.attrs["longitude"])
        n, t = match.pairs(raw, pts_list, (float(xy[0]), float(xy[1])), profile, motion)
        for lst, ds in ((out_n, n), (out_t, t)):
            if ds is not None:
                ds = match.with_parameters(ds)
                ds["probe"] = ("pair", [raw.attrs["probe"]] * ds.sizes["pair"])
                ds.attrs = {}
                lst.append(ds)
        print(
            raw.attrs["probe"],
            0 if n is None else n.sizes["pair"],
            0 if t is None else t.sizes["pair"],
            flush=True,
        )
    attrs = {"iop": a.iop, "storm_motion_u": motion[0], "storm_motion_v": motion[1]}
    for name, lst in (("naive", out_n), ("traj", out_t)):
        ds = xr.concat(lst, "pair")
        ds.attrs = attrs
        ds.to_netcdf(os.path.join(a.out, f"pairs_{a.iop}_{name}.nc"))


if __name__ == "__main__":
    main()
