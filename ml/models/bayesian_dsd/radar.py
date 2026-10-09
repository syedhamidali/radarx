"""
NEXRAD lowest-sweep processing for radar-disdrometer pairs.

Each volume is reduced to the gates of the lowest sweep near the
disdrometers (with KDP from :func:`radarx.retrieve.estimate_kdp`) and a
1-km Cartesian reflectivity grid around them for storm-motion estimation
with :func:`radarx.retrieve.estimate_motion`. The results are cached as
NetCDF next to the volumes.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pyproj
import xarray as xr
import xradar as xd
from scipy.spatial import cKDTree

from radarx.retrieve import estimate_kdp

NODATA = {"DBZH": -32.0, "ZDR": -12.9, "RHOHV": 0.21}


def volume_time(path):
    m = re.search(r"(\d{8})_(\d{6})", os.path.basename(path))
    s = m.group(1) + "T" + m.group(2)
    return np.datetime64(f"{s[:4]}-{s[4:6]}-{s[6:8]}T{s[9:11]}:{s[11:13]}:{s[13:15]}")


CFRAD_NAMES = {"REF": "DBZH", "ZDR": "ZDR", "RHO": "RHOHV", "PHI": "PHIDP"}


def read_sweep(path, sweep=0):
    """Lowest sweep (NEXRAD level 2 or CfRadial), georeferenced, masked, with KDP."""
    if path.endswith(".nc"):
        dt = xd.io.open_cfradial1_datatree(path, sweep=[sweep])
    else:
        dt = xd.io.open_nexradlevel2_datatree(path, sweep=[sweep])
    dt = dt.xradar.georeference()
    ds = dt[[c for c in dt.children if c.startswith("sweep")][0]].to_dataset()
    if "REF" in ds:
        ds = ds.drop_vars([v for v in ("KDP",) if v in ds]).rename(
            {k: v for k, v in CFRAD_NAMES.items() if k in ds}
        )
    for name, lim in NODATA.items():
        ds[name] = ds[name].where(ds[name] > lim)
    kdp = estimate_kdp(ds)
    ds["KDP"] = kdp["KDP"]
    ds.attrs["latitude"] = float(ds.latitude)
    ds.attrs["longitude"] = float(ds.longitude)
    ds.attrs["altitude"] = float(ds.altitude)
    return ds


def project(lat0, lon0, lat, lon):
    """Azimuthal equidistant x, y (m) around (lat0, lon0)."""
    p = pyproj.Proj(proj="aeqd", lat_0=lat0, lon_0=lon0, datum="WGS84")
    return p(np.asarray(lon), np.asarray(lat))


def reduce(ds, center_xy, box=40e3, grid_half=80e3, grid_step=1e3):
    """
    Gates within ``box`` of ``center_xy`` as a point cloud, and a gridded
    reflectivity (nearest gate within 1.5 km) for motion estimation.
    """
    cx, cy = center_xy
    x = ds.x.values
    y = ds.y.values
    z = ds.z.values
    t = np.broadcast_to(ds.time.values[:, None], x.shape)
    near = (np.abs(x - cx) <= box) & (np.abs(y - cy) <= box)
    pts = xr.Dataset(
        {
            name: ("gate", ds[name].values[near].astype(np.float32))
            for name in ("DBZH", "ZDR", "KDP", "RHOHV")
        }
        | {
            "x": ("gate", x[near]),
            "y": ("gate", y[near]),
            "z": ("gate", z[near]),
            "gate_time": ("gate", t[near]),
        }
    )
    gx = cx + np.arange(-grid_half, grid_half + 1, grid_step)
    gy = cy + np.arange(-grid_half, grid_half + 1, grid_step)
    sel = (np.abs(x - cx) <= grid_half + 2e3) & (np.abs(y - cy) <= grid_half + 2e3)
    tree = cKDTree(np.column_stack([x[sel], y[sel]]))
    xx, yy = np.meshgrid(gx, gy)
    dist, idx = tree.query(np.column_stack([xx.ravel(), yy.ravel()]))
    dbz = ds.DBZH.values[sel][idx]
    dbz = np.where(dist <= 1.5e3, dbz, np.nan).reshape(xx.shape)
    observed = (dist <= 1.5e3).reshape(xx.shape)
    grid = xr.Dataset(
        {
            "DBZH": (("y", "x"), dbz.astype(np.float32)),
            "observed": (("y", "x"), observed),
        },
        coords={"x": gx, "y": gy},
    )
    return pts, grid


def process(path, sites, cache_dir, sweep=0):
    """
    Reduce one volume for the given sites (dict name -> (lat, lon)); cached.
    Returns (points, grid, attrs).
    """
    name = os.path.basename(path)
    cache = os.path.join(cache_dir, name + (f".s{sweep}" if sweep else "") + ".nc")
    if os.path.exists(cache):
        pts = xr.open_dataset(cache, group="points").load()
        grid = xr.open_dataset(cache, group="grid").load()
        return pts, grid
    ds = read_sweep(path, sweep)
    lat0, lon0 = ds.attrs["latitude"], ds.attrs["longitude"]
    lats = np.array([s[0] for s in sites.values()])
    lons = np.array([s[1] for s in sites.values()])
    sx, sy = project(lat0, lon0, lats, lons)
    pts, grid = reduce(ds, (float(np.mean(sx)), float(np.mean(sy))))
    attrs = {
        "radar_latitude": lat0,
        "radar_longitude": lon0,
        "radar_altitude": ds.attrs["altitude"],
        "volume_time": str(volume_time(path)),
        "elevation": float(ds.sweep_fixed_angle),
    }
    pts.attrs = attrs
    grid.attrs = attrs
    os.makedirs(cache_dir, exist_ok=True)
    pts.to_netcdf(cache, group="points", mode="w")
    grid.to_netcdf(cache, group="grid", mode="a")
    return pts, grid
