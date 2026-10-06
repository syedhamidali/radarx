"""
Build the dual-Doppler training, validation and test cases from NEXRAD.

For every analysis time of a case, the volume of each radar nearest in time
is downloaded from the AWS NEXRAD Level II archive, its no-data codes masked,
its radial velocity dealiased (``radarx.retrieve.dealias_velocity``) and both
radars are cone-gridded onto one 1 km x 1 km x 500 m grid centred between
them and moved to the analysis time (the start of the first radar's volume) with
the ERA5 0-6 km mean wind as the storm motion (``multi_doppler_input``). The ERA5 background
(``era5_column``) and the multi-Doppler retrieval (``multi_doppler``) are
added. One netCDF file per analysis time is written; it holds both radars,
so every file gives two single-Doppler samples (one per radar) with the same
multi-Doppler target.

Runs in the radarx environment (no PyTorch needed)::

    python make_dataset.py OUTDIR [CASE ...]
"""

import gzip
import os
import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import xarray as xr
import xradar as xd
from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

import radarx  # noqa: F401  (registers the accessors)
from radarx.io import sounding
from radarx.io.aws_data import list_available_files
from radarx.retrieve import multidoppler as md

BUCKET = "unidata-nexrad-level2"

# name: (split, (radar A, radar B), first time, last time, step in minutes)
CASES = {
    # squall line over Mississippi and Alabama (the radarx example case)
    "20220330_KGWX_KBMX": (
        "test",
        ("KGWX", "KBMX"),
        "2022-03-30T23:00",
        "2022-03-31T02:00",
        30,
    ),
    # Mayfield, Kentucky, tornadic supercell and QLCS
    "20211211_KPAH_KHPX": (
        "val",
        ("KPAH", "KHPX"),
        "2021-12-11T01:30",
        "2021-12-11T04:00",
        30,
    ),
    # 27 April 2011 outbreak: Tuscaloosa supercell
    "20110427_KBMX_KGWX": (
        "train",
        ("KBMX", "KGWX"),
        "2011-04-27T20:00",
        "2011-04-27T23:00",
        30,
    ),
    # Moore, Oklahoma, supercell and later storms
    "20130520_KTLX_KINX": (
        "train",
        ("KTLX", "KINX"),
        "2013-05-20T19:30",
        "2013-05-20T22:30",
        30,
    ),
    # Barnsdall, Oklahoma, supercells
    "20240507_KTLX_KINX": (
        "train",
        ("KTLX", "KINX"),
        "2024-05-07T01:00",
        "2024-05-07T04:00",
        30,
    ),
    # Iowa and Illinois outbreak
    "20230331_KDVN_KILX": (
        "train",
        ("KDVN", "KILX"),
        "2023-03-31T20:00",
        "2023-03-31T23:00",
        30,
    ),
}

X = np.arange(-100e3, 100e3 + 1, 1000.0)
Y = np.arange(-100e3, 100e3 + 1, 1000.0)
Z = np.arange(500.0, 12e3 + 1, 500.0)

_S3 = None


def _s3():
    global _S3
    if _S3 is None:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config

        _S3 = boto3.client(
            "s3", config=Config(signature_version=UNSIGNED, max_pool_connections=64)
        )
    return _S3


def _scan_time(name):
    stamp = name.split("_")[0][4:] + name.split("_")[1]
    return np.datetime64(
        f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[8:10]}:{stamp[10:12]}:{stamp[12:14]}"
    )


def list_volumes(site, t0, t1):
    """Keys and start times of the volumes of ``site`` between t0 and t1."""
    days = np.arange(
        np.datetime64(t0, "D") - 1, np.datetime64(t1, "D") + 1, np.timedelta64(1, "D")
    )
    keys = []
    for day in days:
        d = str(day).replace("-", "/")
        keys += list_available_files(BUCKET, f"{d}/{site}/")
    keys = sorted(k for k in keys if not k.endswith("MDM") and "_V0" in k)
    times = np.array([_scan_time(os.path.basename(k)) for k in keys])
    return keys, times


# directories searched for already downloaded volumes (NEXRAD_DIRS=dir1:dir2)
LOCAL_DIRS = [d for d in os.environ.get("NEXRAD_DIRS", "").split(":") if d]


def fetch(key, workdir):
    """Local copy of an archive volume (ranged parallel download)."""
    from boto3.s3.transfer import TransferConfig

    name = os.path.basename(key)
    for d in LOCAL_DIRS:
        for cand in (name, name[:-3] if name.endswith(".gz") else name):
            if os.path.exists(os.path.join(d, cand)):
                return os.path.join(d, cand)
    local = os.path.join(workdir, name)
    if not os.path.exists(local):
        config = TransferConfig(
            multipart_threshold=256 * 1024,
            multipart_chunksize=256 * 1024,
            max_concurrency=48,
        )
        _s3().download_file(BUCKET, key, local + ".part", Config=config)
        os.replace(local + ".part", local)
    if local.endswith(".gz"):
        plain = local[:-3]
        with gzip.open(local) as src, open(plain, "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.remove(local)
        local = plain
    return local


def nexrad_volume(path):
    """Volume with no-data codes masked and the Nyquist velocity of each sweep."""
    with NEXRADLevel2File(path) as nf:
        nyquist = [
            h["msg_31_data_header"]["RAD"]["nyquist_vel"] / 100.0
            for h in nf.msg_31_data_header
        ]
    dtree = xd.io.open_nexradlevel2_datatree(path)
    keep = {}
    for i, name in enumerate(n for n in dtree.children if n.startswith("sweep")):
        ds = dtree[name].to_dataset()
        if "DBZH" in ds:
            ds["DBZH"] = ds.DBZH.where(ds.DBZH > -32)
        if "VRADH" in ds:
            ds["VRADH"] = ds.VRADH.where(ds.VRADH > -63.9)
        keep[name] = ds.assign_coords(nyquist_velocity=nyquist[i])
    for name, ds in keep.items():
        dtree[name] = ds
    return dtree


def dealiased(path):
    vol = nexrad_volume(path)
    return vol.radarx.assign(vol.radarx.dealias("VRADH", name="VRADH"))


def _log(msg):
    if os.environ.get("VERBOSE"):
        print(msg, flush=True)


def build_time(case, t, workdir):
    split, radars, *_ = CASES[case]
    vols = {}
    for site in radars:
        keys, times = list_volumes(
            site, t - np.timedelta64(30, "m"), t + np.timedelta64(30, "m")
        )
        dt = np.abs((times - t).astype("timedelta64[s]").astype(float))
        i = int(np.argmin(dt))
        if dt[i] > 360:
            raise RuntimeError(f"{site}: no volume within 6 min of {t}")
        if site == radars[0]:
            t = times[i]  # analysis time: the start of the first radar's volume
        t_ = time.perf_counter()
        path = fetch(keys[i], workdir)
        _log(f"{site} download {time.perf_counter() - t_:.0f} s")
        vols[site] = dealiased(path)
        _log(f"{site} read + dealias {time.perf_counter() - t_:.0f} s")
    a, b = vols[radars[0]], vols[radars[1]]
    lat = 0.5 * (float(a["latitude"].values) + float(b["latitude"].values))
    lon = 0.5 * (float(a["longitude"].values) + float(b["longitude"].values))
    # ERA5 background on the grid (nearest hour, one request per hour), and
    # the storm motion as its 0-6 km mean wind (it only moves the second
    # radar by its few minutes of time offset)
    empty = xr.Dataset(coords={"z": Z, "y": Y, "x": X}).assign_coords(
        crs_wkt=md._origin_crs(lat, lon)
    )
    bg = sounding.era5_column(empty, str(t), time_interpolation="nearest")
    low = bg[["u", "v"]].where(bg.z <= 6000.0)
    motion = (float(low.u.mean()), float(low.v.mean()))
    _log("background")
    grids = md.multi_doppler_input(
        [a, b], X, Y, Z, origin=(lat, lon), time=t, motion=motion
    )
    _log("gridded")
    bg = bg.drop_vars([c for c in bg.coords if c not in ("z", "y", "x")])
    wind = md.multi_doppler(grids, bg)
    _log(f"multi-Doppler {wind.attrs['run_time_s']:.0f} s")
    out = xr.Dataset(
        {
            "VRADH": grids["VRADH"].astype(np.float32),
            "DBZH": grids["DBZH"].astype(np.float32),
            "radar_x": grids["radar_x"],
            "radar_y": grids["radar_y"],
            "radar_altitude": grids["radar_altitude"],
            "u_bg": bg["u"].astype(np.float32),
            "v_bg": bg["v"].astype(np.float32),
            "air_density": bg["air_density"].astype(np.float32),
            "freezing_level": bg["freezing_level"].astype(np.float32),
            "u": wind["u"],
            "v": wind["v"],
            "w": wind["w"],
            "fall_speed": wind["fall_speed"],
            "n_radars": wind["n_radars"],
            "beam_crossing_angle": wind["beam_crossing_angle"],
        }
    )
    out = out.drop_vars(
        [c for c in ("lat", "lon", "crs_wkt", "time") if c in out.coords]
    )
    out = out.assign_coords(radar_name=("radar", list(radars)))
    out.attrs = {
        "case": case,
        "split": split,
        "analysis_time": str(t),
        "motion_u": motion[0],
        "motion_v": motion[1],
        "origin_latitude": lat,
        "origin_longitude": lon,
        "multi_doppler_iterations": wind.attrs["iterations"],
    }
    return out


def build_case(case, outdir):
    split, radars, t0, t1, step = CASES[case]
    times = np.arange(
        np.datetime64(t0), np.datetime64(t1) + 1, np.timedelta64(step, "m")
    )
    os.makedirs(os.path.join(outdir, split), exist_ok=True)
    workdir = tempfile.mkdtemp(prefix=case, dir=outdir)
    done = []
    for t in times:
        fname = os.path.join(outdir, split, f"{case}_{str(t)[:16].replace(':', '')}.nc")
        if os.path.exists(fname):
            done.append(fname)
            continue
        t_start = time.perf_counter()
        try:
            ds = build_time(case, t, workdir)
        except Exception:  # noqa: BLE001 - keep going with the other times
            print(case, t, "FAILED\n", traceback.format_exc(), flush=True)
            continue
        enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
        ds.to_netcdf(fname, encoding=enc)
        good = int(((ds.n_radars >= 2) & (ds.beam_crossing_angle > 30)).sum())
        print(
            f"{case} {t}: {good} dual-Doppler cells, {time.perf_counter() - t_start:.0f} s",
            flush=True,
        )
        done.append(fname)
        for f in os.listdir(workdir):  # keep the disk use small
            os.remove(os.path.join(workdir, f))
    shutil.rmtree(workdir, ignore_errors=True)
    return done


if __name__ == "__main__":
    outdir = sys.argv[1]
    cases = sys.argv[2:] or list(CASES)
    workers = int(os.environ.get("WORKERS", "3"))
    with ProcessPoolExecutor(workers) as ex:
        for files in ex.map(build_case, cases, [outdir] * len(cases)):
            print(len(files), "files", flush=True)
