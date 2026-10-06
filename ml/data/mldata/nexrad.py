"""
NEXRAD Level II access: list and fetch volumes from the NOAA open-data
archive on AWS and read them as xradar DataTrees ready for radarx.

The archive is the public ``unidata-nexrad-level2`` bucket (NOAA Open Data
Dissemination program), read anonymously with radarx's S3 helpers.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

BUCKET = "unidata-nexrad-level2"

#: No-data floors of the NEXRAD moments as decoded by xradar: values at or
#: below these are flags (below threshold, range folded), not data.
NODATA = {"DBZH": -32.0, "VRADH": -63.9, "ZDR": -12.9, "RHOHV": 0.21, "PHIDP": -0.1}

_KEY = re.compile(
    r"^(?P<radar>[A-Z]{4})(?P<date>\d{8})_(?P<time>\d{6})(_V\d\d)?(\.gz)?$"
)


def parse_key(key):
    """
    Radar and start time of an archive key, or None for other files.

    ``2022/03/30/KGWX/KGWX20220330_234639_V06`` gives
    ``("KGWX", datetime(2022, 3, 30, 23, 46, 39))``; ``*_MDM`` (metadata)
    files and unknown names give None.
    """
    name = key.rsplit("/", 1)[-1]
    m = _KEY.match(name)
    if m is None:
        return None
    when = datetime.strptime(m["date"] + m["time"], "%Y%m%d%H%M%S")
    return m["radar"], when


def volume_id(radar, when):
    """Identifier ``RADAR_YYYYmmddTHHMMSS`` of a volume."""
    return f"{radar}_{when:%Y%m%dT%H%M%S}"


def select_keys(keys, start, end):
    """
    Archive keys of volumes starting in ``[start, end]``, in time order.

    Compressed duplicates (``.gz`` next to the same volume uncompressed) and
    non-volume files are dropped.
    """
    chosen = {}
    for key in keys:
        parsed = parse_key(key)
        if parsed is None:
            continue
        radar, when = parsed
        if not start <= when <= end:
            continue
        vid = volume_id(radar, when)
        if vid not in chosen or chosen[vid].endswith(".gz"):
            chosen[vid] = key
    return [chosen[k] for k in sorted(chosen)]


def list_volumes(radar, start, end, lister=None):
    """
    Archive keys of ``radar`` volumes starting between ``start`` and ``end``.

    Parameters
    ----------
    radar : str
        ICAO identifier, e.g. ``"KGWX"``.
    start, end : datetime
        Time window (UTC, inclusive).
    lister : callable, optional
        ``lister(bucket, prefix) -> list of keys``; default
        :func:`radarx.io.aws_data.list_available_files` (anonymous).
    """
    if lister is None:
        from radarx.io.aws_data import list_available_files as lister
    keys = []
    day = datetime(start.year, start.month, start.day)
    while day <= end:
        keys += lister(BUCKET, f"{day:%Y/%m/%d}/{radar}/")
        day += timedelta(days=1)
    return select_keys(keys, start, end)


def fetch(key, cache_dir):
    """Local path of archive ``key``, downloaded once into ``cache_dir``."""
    cache = Path(cache_dir)
    path = cache / key
    if path.exists() and path.stat().st_size > 0:
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from radarx.io.aws_data import get_s3_client

    tmp = path.with_suffix(path.suffix + ".part")
    get_s3_client(anonymous=True).download_file(BUCKET, key, str(tmp))
    tmp.replace(path)
    return str(path)


def sweep_headers(path):
    """Per-sweep ``(elevation, nyquist_velocity)`` from the Message 31 headers."""
    from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

    with NEXRADLevel2File(path) as nf:
        return [
            (
                float(h["msg_31_header"]["elevation_angle"]),
                h["msg_31_data_header"]["RAD"]["nyquist_vel"] / 100.0,
            )
            for h in nf.msg_31_data_header
        ]


def mask_nodata(ds):
    """Set the NEXRAD no-data codes of the moments in a sweep to NaN."""
    for name, floor in NODATA.items():
        if name in ds:
            ds[name] = ds[name].where(ds[name] > floor)
    return ds


def read_volume(path, max_elevation=None):
    """
    Read a NEXRAD Level II volume for radarx.

    Only sweeps up to ``max_elevation`` (degrees) are read. Every sweep gets
    its Nyquist velocity (from the Message 31 radial headers) as the
    ``nyquist_velocity`` coordinate, and the no-data codes of ``DBZH``,
    ``VRADH``, ``ZDR``, ``RHOHV`` and ``PHIDP`` (:data:`NODATA`) are set to
    NaN.

    Returns
    -------
    xarray.DataTree
        Loaded into memory; sweeps keep their index in the volume
        (``sweep_<i>``).
    """
    import xarray as xr
    import xradar as xd

    headers = sweep_headers(path)
    index = [
        i
        for i, (elev, _) in enumerate(headers)
        if max_elevation is None or elev <= max_elevation + 0.25
    ]
    dtree = xd.io.open_nexradlevel2_datatree(path, sweep=index)
    nodes = {"/": dtree.root.to_dataset(inherit=False)}
    for name in dtree.children:
        if not name.startswith("sweep_"):
            nodes[name] = dtree[name].to_dataset(inherit=False)
            continue
        i = int(name.split("_")[1])
        ds = dtree[name].to_dataset(inherit=False).load()
        ds = mask_nodata(ds).assign_coords(nyquist_velocity=np.float64(headers[i][1]))
        nodes[name] = ds
    return xr.DataTree.from_dict(nodes)


def sweep_kind(ds):
    """``"polarimetric"`` (has ZDR), ``"doppler"`` (VRADH only) or None."""
    if "DBZH" not in ds:
        return None
    if "ZDR" in ds:
        return "polarimetric"
    if "VRADH" in ds:
        return "doppler"
    return None


def unique_sweeps(dtree, kind):
    """
    Names of the sweeps of ``kind``, skipping repeated elevations.

    NEXRAD SAILS/MESO-SAILS scans repeat the lowest cut several times in a
    volume; the repeats are nearly identical samples, so only the first cut
    of each elevation is kept.
    """
    seen, names = [], []
    for name in sorted(
        (n for n in dtree.children if n.startswith("sweep_")),
        key=lambda n: int(n.split("_")[1]),
    ):
        ds = dtree[name].to_dataset()
        if kind == "velocity":
            ok = "VRADH" in ds
        else:
            ok = sweep_kind(ds) == kind
        if not ok:
            continue
        elev = float(ds["sweep_fixed_angle"])
        if any(abs(elev - e) < 0.2 for e in seen):
            continue
        seen.append(elev)
        names.append(name)
    return names
