"""
Dataset build: volumes in parallel, then sequences, pairs and Zarr stores.

Stages
------
1. **Volumes** (one process per volume, :func:`process_volume`): fetch, read,
   run radarx's ``echo_mask``, ``estimate_kdp``, ``hid`` and
   ``dealias_velocity`` on the whole volume (each one compiled, multithreaded
   kernel call), build the polar samples of every task and the composite
   frame for nowcasting, and write them as per-volume parts (temporary pickles).
2. **Nowcasting** (:func:`nowcast_parts`): consecutive frames of each case
   become sequences; motion and extrapolation come from radarx.
3. **Multi-Doppler** (:func:`process_pair`, one process per radar pair).
4. **Consolidation** (:func:`consolidate`): the parts are appended into one
   chunked Zarr store per task and split, ``<out>/<task>/<split>.zarr``,
   with provenance attributes; ``<out>/manifest.json`` lists the cases,
   volumes, sample counts and timings.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xarray as xr

from . import labels
from .cases import SPLITS, assign_splits, check_no_leakage, expand_cases, sample_rng
from .nexrad import fetch, read_volume, unique_sweeps
from .polar import PolarGrid, resample_sweep
from .schema import SCHEMA_VERSION, SCHEMAS, attrs_for

LICENCE = (
    "NEXRAD Level II data: NOAA open data, no restrictions on use; distributed "
    "through the NOAA Open Data Dissemination program on AWS "
    "(s3://unidata-nexrad-level2). Cite: NOAA National Weather Service Radar "
    "Operations Center (1991): NOAA Next Generation Radar (NEXRAD) Level II "
    "Base Data. NOAA National Centers for Environmental Information, "
    "https://doi.org/10.7289/V5W9574V. Labels: radarx (MIT licence)."
)

_DATETIME_ENCODING = {"units": "milliseconds since 1970-01-01", "dtype": "int64"}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def git_sha(path=None):
    """Commit of the radarx checkout the builder runs from (or "unknown")."""
    here = Path(path or __file__).resolve().parent
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=here,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _sweep_time(ds):
    t = np.asarray(ds["time"].values).astype("datetime64[ns]")
    return t.min() if t.size else np.datetime64("NaT", "ns")


def _sample_coords(task, ds, name, record, site):
    """Per-sample coordinates of a polar sample, typed as in the schema."""
    spec = SCHEMAS[task]["coords"]
    values = {
        "radar": record["radar"],
        "volume_id": record["volume_id"],
        "sweep": int(name.split("_")[1]),
        "elevation": float(ds["sweep_fixed_angle"]),
        "time": _sweep_time(ds),
        "latitude": site["latitude"],
        "longitude": site["longitude"],
        "altitude": site["altitude"],
        "case": record["case"],
    }
    return {
        k: ("sample", np.asarray([values[k]]).astype(spec[k]["dtype"])) for k in spec
    }


def _finish(task, ds, coords):
    """Add the sample dimension, schema attributes and coordinates."""
    ds = ds.expand_dims("sample")
    for name in ds.data_vars:
        if name in SCHEMAS[task]["variables"]:
            keep = {k: v for k, v in ds[name].attrs.items() if k.startswith("flag")}
            ds[name].attrs = {**attrs_for(task, name), **keep}
    return ds.assign_coords(coords)


def _write_part(ds, path):
    """Write a temporary part (a pickled in-memory Dataset; fast, uncompressed)."""
    path = Path(path).with_suffix(".pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(ds.load(), f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def _read_part(path):
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301 - parts are written by this build


def _site(dtree):
    root = dtree.root.to_dataset()
    return {k: float(root[k]) for k in ("latitude", "longitude", "altitude")}


# --------------------------------------------------------------------------
# stage 1: one volume
# --------------------------------------------------------------------------


def process_volume(job):
    """
    Build all polar samples and the nowcasting frame of one volume.

    Parameters
    ----------
    job : dict
        ``record`` (from :func:`mldata.cases.expand_cases`), ``config``
        (normalised), ``cache_dir``, ``parts_dir``, ``n_threads`` and
        ``temperature`` (profile Dataset or None).

    Returns
    -------
    dict
        ``volume_id``, ``split``, ``samples`` per task, ``seconds`` per stage,
        ``frame`` (path of the composite or None) and ``error`` (None).
    """
    from radarx.retrieve import (
        apply_mask,
        dealias_velocity,
        echo_mask,
        estimate_kdp,
        hid,
    )

    rec, cfg = job["record"], job["config"]
    tasks, nt = cfg["tasks"], job.get("n_threads")
    parts = Path(job["parts_dir"])
    grid = PolarGrid.from_config(cfg["polar"])
    seconds, counts = {}, {}
    summary = {"volume_id": rec["volume_id"], "split": rec["split"], "error": None}

    def tick(stage, t0):
        seconds[stage] = round(time.perf_counter() - t0, 3)
        return time.perf_counter()

    t = time.perf_counter()
    path = fetch(rec["key"], job["cache_dir"])
    t = tick("fetch", t)
    dtree = read_volume(path, cfg["max_elevation"])
    site = _site(dtree)
    t = tick("read", t)
    pol = unique_sweeps(dtree, "polarimetric")
    vel = unique_sweeps(dtree, "velocity")

    qc = echo_mask(dtree, n_threads=nt)
    clean = apply_mask(dtree, qc)
    t = tick("echo_mask", t)
    kdp = None
    if {"kdp", "hid"} & set(tasks) and pol:
        kdp = estimate_kdp(dtree, n_threads=nt)
        t = tick("estimate_kdp", t)
    hid_tree = None
    if "hid" in tasks and kdp is not None:
        merged = dtree.radarx.assign(kdp).radarx.assign(qc)
        hid_tree = hid(
            merged,
            job.get("temperature"),
            mask="METEO_MASK",
            scores=False,
            n_threads=nt,
        )
        t = tick("hid", t)
    dealiased = None
    if "dealias" in tasks and vel:
        dealiased = dealias_velocity(dtree, n_threads=nt)
        t = tick("dealias_velocity", t)

    out = {task: [] for task in tasks}
    for name in pol:
        sweep = dtree[name].to_dataset()
        if "qc" in tasks and name in qc.children:
            s = labels.qc_sample(sweep, qc[name].to_dataset())
            out["qc"].append(("qc", s, name, sweep))
        if "kdp" in tasks and kdp is not None and name in kdp.children:
            s = labels.kdp_sample(sweep, kdp[name].to_dataset())
            out["kdp"].append(("kdp", s, name, sweep))
        if "hid" in tasks and hid_tree is not None and name in hid_tree.children:
            merged_sweep = sweep.assign(
                KDP=kdp[name]["KDP"], METEO_MASK=qc[name]["METEO_MASK"]
            )
            temp = None
            if job.get("temperature") is not None:
                temp = labels.gate_temperature(
                    sweep, job["temperature"], site["altitude"]
                )
            s = labels.hid_sample(merged_sweep, hid_tree[name].to_dataset(), temp)
            for key in ("flag_values", "flag_meanings"):
                if key in hid_tree[name]["HID"].attrs:
                    s["HID"].attrs[key] = hid_tree[name]["HID"].attrs[key]
            out["hid"].append(("hid", s, name, sweep))
        if "inpaint" in tasks:
            dbz = resample_sweep(clean[name].to_dataset(), ["DBZH"], grid)["DBZH"]
            rng = sample_rng(cfg["seed"], rec["volume_id"], name, "inpaint")
            opts = {k: v for k, v in tasks["inpaint"].items()}
            s = labels.inpaint_sample(dbz, rng, **opts)
            out["inpaint"].append(("inpaint", s, name, sweep))

    if dealiased is not None:
        opts = tasks["dealias"]
        for name in vel:
            if name not in dealiased.children:
                continue
            sweep = dtree[name].to_dataset()
            nyq = float(sweep["nyquist_velocity"])
            found = labels.velocity_truth(
                sweep["VRADH"],
                dealiased[name]["VRADH_dealiased"],
                nyq,
                sources=opts["sources"],
                max_jump_fraction=opts["max_jump_fraction"],
                min_gates=opts["min_gates"],
            )
            if found is None:
                continue
            truth, source = found
            rng = sample_rng(cfg["seed"], rec["volume_id"], name, "dealias")
            vn = float(rng.uniform(*opts["nyquist"]))
            s = labels.dealias_sample(sweep, truth, vn)
            s = s.assign(
                nyquist_velocity=np.float32(vn),
                nyquist_velocity_original=np.float32(nyq),
                truth_source=np.int8(source),
                folded_fraction=np.float32(s.attrs.pop("folded_fraction")),
            )
            out["dealias"].append(("dealias", s, name, sweep))
    t = tick("samples", t)

    for task, items in out.items():
        if task in ("nowcast", "multidoppler"):
            continue
        done = []
        for _, s, name, sweep in items:
            polar = [v for v in s.data_vars if s[v].ndim == 2]
            scalar = s[[v for v in s.data_vars if s[v].ndim == 0]]
            if (
                "azimuth" in s.dims
                and s.sizes["azimuth"] == grid.n_azimuth
                and (np.allclose(s["azimuth"].values, grid.azimuth))
            ):
                fixed = s[polar]
            else:
                fixed = resample_sweep(s, polar, grid)
            fixed = fixed.assign({k: v for k, v in scalar.data_vars.items()}).drop_vars(
                [c for c in fixed.coords if c not in ("azimuth", "range")],
            )
            coords = _sample_coords(task, sweep, name, rec, site)
            done.append(_finish(task, fixed, coords))
        counts[task] = len(done)
        if done:
            ds = xr.concat(done, dim="sample")
            _write_part(ds, parts / task / rec["split"] / f"{rec['volume_id']}.zarr")
    t = tick("write", t)

    summary["frame"] = None
    if "nowcast" in tasks:
        frame = labels.composite_frame(clean, tasks["nowcast"]["grid"], n_threads=nt)
        frame = frame.to_dataset(name="DBZH")
        frame.attrs.update(
            {k: rec[k] for k in ("volume_id", "radar", "case", "split", "event")}
        )
        fpath = _write_part(frame, parts / "_frames" / f"{rec['volume_id']}.pkl")
        summary["frame"] = str(fpath)
        t = tick("composite", t)
    summary["samples"] = counts
    summary["seconds"] = seconds
    return summary


def _safe(func, job):
    """Run ``func(job)``; an exception becomes the ``error`` of the summary."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return func(job)
    except Exception as err:  # noqa: BLE001 - one bad volume must not stop a build
        rec = job.get("record", {})
        return {
            "volume_id": rec.get("volume_id", job.get("pair_id")),
            "split": rec.get("split", job.get("split")),
            "error": f"{type(err).__name__}: {err}",
            "traceback": traceback.format_exc(),
            "samples": {},
            "seconds": {},
            "frame": None,
        }


# --------------------------------------------------------------------------
# stage 2: nowcasting sequences
# --------------------------------------------------------------------------


def _frames(frame_paths):
    frames = [_read_part(p) for p in frame_paths]
    return sorted(frames, key=lambda f: f["time"].values)


def sequences(times, length, max_gap_s):
    """Start indices of runs of ``length`` frames with gaps <= ``max_gap_s``."""
    t = np.asarray(times).astype("datetime64[ns]").astype(np.int64) / 1e9
    starts = []
    for i in range(len(t) - length + 1):
        if np.all(np.diff(t[i : i + length]) <= max_gap_s):
            starts.append(i)
    return starts


def nowcast_parts(frame_paths, cfg, parts_dir, n_threads=None):
    """Write the nowcasting samples of all cases; returns counts per split."""
    opts = cfg["tasks"]["nowcast"]
    n_in, n_out = int(opts["n_input"]), int(opts["n_target"])
    frames = _frames(frame_paths)
    by_case = {}
    for f in frames:
        by_case.setdefault(f.attrs["case"], []).append(f)
    counts = {}
    for case, seq in by_case.items():
        times = [f["time"].values for f in seq]
        starts = sequences(times, n_in + n_out, 60.0 * opts["max_gap_minutes"])
        samples = []
        for i in starts:
            window = [
                f["DBZH"].assign_coords(time=f["time"])
                for f in seq[i : i + n_in + n_out]
            ]
            s = labels.nowcast_sample(window, n_in, opts["motion"], n_threads=n_threads)
            if s is None:
                continue
            a = seq[i].attrs
            spec = SCHEMAS["nowcast"]["coords"]
            coords = {
                "radar": (
                    "sample",
                    np.asarray([a["radar"]]).astype(spec["radar"]["dtype"]),
                ),
                "case": (
                    "sample",
                    np.asarray([a["case"]]).astype(spec["case"]["dtype"]),
                ),
                "time": (
                    "sample",
                    np.asarray([s["time"].values]).astype("datetime64[ns]"),
                ),
            }
            s = s.drop_vars("time")
            s.attrs = {}
            samples.append(_finish("nowcast", s, coords))
        if samples:
            split = seq[0].attrs["split"]
            ds = xr.concat(samples, dim="sample")
            _write_part(ds, Path(parts_dir) / "nowcast" / split / f"{case}.zarr")
            counts[split] = counts.get(split, 0) + len(samples)
    return counts


# --------------------------------------------------------------------------
# stage 3: multi-Doppler pairs
# --------------------------------------------------------------------------


def _pair_motion(frame_paths, radar, when, max_gap_s):
    """Echo motion of ``radar`` from its two frames nearest before ``when``."""
    from radarx.retrieve import estimate_motion

    frames = [f for f in _frames(frame_paths) if f.attrs["radar"] == radar]
    t = np.datetime64(when, "ns")
    before = [f for f in frames if f["time"].values <= t + np.timedelta64(10, "m")]
    if len(before) < 2:
        return None
    a, b = before[-2], before[-1]
    dt = (b["time"].values - a["time"].values) / np.timedelta64(1, "s")
    if dt > max_gap_s:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = estimate_motion(a["DBZH"], b["DBZH"], floor=10.0)
    if not np.isfinite(float(m["u"])):
        return None
    return float(m["u"]), float(m["v"])


def process_pair(job):
    """
    Multi-Doppler sample of one radar pair (run in a worker process).

    The volumes are read again from the cache, quality-controlled
    (``echo_mask``) and dealiased (``dealias_velocity``) before
    :func:`mldata.labels.multidoppler_sample`.
    """
    from radarx.retrieve import apply_mask, dealias_velocity, echo_mask

    cfg, nt = job["config"], job.get("n_threads")
    t0 = time.perf_counter()
    vols = []
    for rec in job["records"]:
        dtree = read_volume(
            fetch(rec["key"], job["cache_dir"]), job.get("max_elevation")
        )
        clean = apply_mask(dtree, echo_mask(dtree, n_threads=nt))
        clean = clean.radarx.assign(dealias_velocity(clean, n_threads=nt, name="VRADH"))
        vols.append(clean)
    md = cfg["tasks"]["multidoppler"]
    when = np.datetime64(job["time"], "ns")
    motion = job.get("motion")
    s = labels.multidoppler_sample(
        vols,
        md["grid"],
        time=when if motion is not None else None,
        motion=motion,
        n_threads=nt,
    )
    spec = SCHEMAS["multidoppler"]["coords"]
    coords = {
        "radar": (
            ("sample", "radar_index"),
            s["radar"].values[None, :].astype(spec["radar"]["dtype"]),
        ),
        "case": ("sample", np.asarray([job["case"]]).astype(spec["case"]["dtype"])),
        "time": ("sample", np.asarray([when]).astype("datetime64[ns]")),
    }
    s = _finish("multidoppler", s.drop_vars("radar"), coords)
    s.attrs = {
        "motion": "none" if motion is None else f"{motion[0]:.2f} {motion[1]:.2f}"
    }
    _write_part(
        s,
        Path(job["parts_dir"])
        / "multidoppler"
        / job["split"]
        / f"{job['pair_id']}.zarr",
    )
    return {
        "volume_id": job["pair_id"],
        "split": job["split"],
        "error": None,
        "samples": {"multidoppler": 1},
        "seconds": {"pair": round(time.perf_counter() - t0, 3)},
        "frame": None,
    }


def pair_jobs(cfg, records, ok_ids):
    """Jobs for the configured radar pairs; volumes nearest the analysis time."""
    jobs, skipped = [], []
    for i, pair in enumerate(cfg["tasks"].get("multidoppler", {}).get("pairs", [])):
        when = np.datetime64(str(pair["time"]).replace("Z", ""), "ns")
        tol = np.timedelta64(int(60 * float(pair.get("tolerance_minutes", 6))), "s")
        chosen = []
        for radar in pair["radars"]:
            cands = [
                r
                for r in records
                if r["radar"] == radar.upper() and r["volume_id"] in ok_ids
            ]
            cands.sort(key=lambda r: abs(np.datetime64(r["time"], "ns") - when))
            if cands and abs(np.datetime64(cands[0]["time"], "ns") - when) <= tol:
                chosen.append(cands[0])
        splits = {r["split"] for r in chosen}
        if len(chosen) != len(pair["radars"]) or len(splits) != 1:
            skipped.append(
                {"pair": pair, "reason": "volumes missing or in several splits"}
            )
            continue
        jobs.append(
            {
                "pair_id": f"pair{i}_" + "_".join(r["radar"] for r in chosen),
                "records": chosen,
                "time": str(when),
                "split": splits.pop(),
                "case": chosen[0]["case"],
            }
        )
    return jobs, skipped


# --------------------------------------------------------------------------
# stage 4: consolidation
# --------------------------------------------------------------------------


def _encoding(ds):
    enc = {}
    for name, var in ds.variables.items():
        if "sample" not in var.dims:
            continue
        e = {}
        if var.dtype.kind == "M":
            e.update(_DATETIME_ENCODING)
        if name in ds.data_vars:
            e["chunks"] = tuple(1 if d == "sample" else ds.sizes[d] for d in var.dims)
        if e:
            enc[name] = e
    return enc


def consolidate(parts_dir, out_dir, provenance, batch_bytes=2**30):
    """
    Append the parts of every task and split into ``<task>/<split>.zarr``.

    Parts are written in batches of about ``batch_bytes`` (in memory).

    Returns
    -------
    dict
        ``{task: {split: n_samples}}``.
    """
    import zarr

    parts_dir, out_dir = Path(parts_dir), Path(out_dir)
    counts = {}
    for task in SCHEMAS:
        for split in SPLITS:
            files = sorted((parts_dir / task / split).glob("*.pkl"))
            if not files:
                continue
            target = out_dir / task / f"{split}.zarr"
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            n = 0
            cases = set()
            batch, size = [], 0

            def flush(batch, first):
                ds = xr.concat(batch, dim="sample") if len(batch) > 1 else batch[0]
                opts = {"zarr_format": 2, "consolidated": False}
                if first:
                    ds.attrs = {}
                    ds.to_zarr(target, mode="w", encoding=_encoding(ds), **opts)
                else:
                    fixed = [v for v in ds.variables if "sample" not in ds[v].dims]
                    ds.drop_vars(fixed).to_zarr(target, append_dim="sample", **opts)

            # parts are gathered in memory up to ``batch_bytes`` and written
            # in one call: each Zarr append has a fixed cost
            for f in files:
                ds = _read_part(f)
                cases.update(str(c) for c in np.atleast_1d(ds["case"].values))
                n += ds.sizes["sample"]
                batch.append(ds)
                size += ds.nbytes
                if size >= batch_bytes:
                    flush(batch, first=n == sum(b.sizes["sample"] for b in batch))
                    batch, size = [], 0
            if batch:
                flush(batch, first=n == sum(b.sizes["sample"] for b in batch))
            attrs = dict(provenance)
            attrs.update(
                {
                    "task": task,
                    "split": split,
                    "description": SCHEMAS[task]["description"],
                    "n_samples": n,
                    "cases": sorted(cases),
                }
            )
            group = zarr.open_group(str(target), mode="r+", zarr_format=2)
            group.attrs.update(attrs)
            zarr.consolidate_metadata(str(target), zarr_format=2)
            counts.setdefault(task, {})[split] = n
    return counts


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def provenance(cfg):
    """Attributes stored with every output store."""
    import radarx

    return {
        "title": f"{cfg['name']}: radar ML training data with radarx labels",
        "schema_version": SCHEMA_VERSION,
        "radarx_version": radarx.__version__,
        "git_sha": git_sha(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": "NEXRAD Level II, s3://unidata-nexrad-level2 (anonymous)",
        "licence": LICENCE,
        "builder": "ml/data/build_dataset.py",
        "config_name": cfg["name"],
        "seed": int(cfg["seed"]),
    }


def _jsonable(cfg):
    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, list | tuple):
            return [conv(v) for v in o]
        if isinstance(o, datetime):
            return o.isoformat()
        return o

    return conv(cfg)


def build(
    cfg,
    out_dir,
    *,
    cache_dir,
    workers=None,
    keep_parts=False,
    temperature=None,
    log=print,
    lister=None,
):
    """
    Build the dataset of a normalised config.

    Parameters
    ----------
    cfg : dict
        From :func:`mldata.cases.load_config`.
    out_dir : path-like
        Output folder (created; existing stores are replaced).
    cache_dir : path-like
        Download cache of the NEXRAD files.
    workers : int, optional
        Worker processes (default: all cores up to the number of volumes).
        Each worker gives the radarx kernels ``cores // workers`` threads.
    keep_parts : bool
        Keep the per-volume parts after consolidation.
    temperature : dict, optional
        Temperature profile per case name (for ``hid``); default from the
        case's ``freezing_level`` (:func:`mldata.labels.standard_profile`).

    Returns
    -------
    dict
        The manifest (also written to ``<out_dir>/manifest.json``).
    """
    t_start = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = out_dir / "_parts"
    if parts.exists():
        shutil.rmtree(parts)
    assign_splits(cfg)
    check_no_leakage(
        cfg["cases"], cfg["splits"]["min_gap_hours"], cfg["splits"]["holdout_radars"]
    )
    records = expand_cases(cfg, lister=lister)
    log(f"{len(records)} volumes in {len(cfg['cases'])} cases")
    cores = os.cpu_count() or 1
    workers = max(1, min(workers or cores, len(records) or 1))
    n_threads = max(1, cores // workers)
    profiles = dict(temperature or {})
    for c in cfg["cases"]:
        if c["name"] not in profiles and c.get("freezing_level") is not None:
            profiles[c["name"]] = labels.standard_profile(float(c["freezing_level"]))
    jobs = [
        {
            "record": r,
            "config": cfg,
            "cache_dir": str(cache_dir),
            "parts_dir": str(parts),
            "n_threads": n_threads,
            "temperature": profiles.get(r["case"]),
        }
        for r in records
    ]
    summaries = []
    t_vol = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_safe, process_volume, j) for j in jobs]
        for fut in as_completed(futures):
            s = fut.result()
            summaries.append(s)
            state = s["error"] or " ".join(f"{k}={v}" for k, v in s["samples"].items())
            log(
                f"[{len(summaries)}/{len(jobs)}] {s['volume_id']} ({s['split']}): {state}"
            )
    t_vol = time.perf_counter() - t_vol
    ok = {s["volume_id"] for s in summaries if not s["error"]}
    frames = [s["frame"] for s in summaries if s.get("frame")]

    t_now = time.perf_counter()
    nowcast = {}
    if "nowcast" in cfg["tasks"] and frames:
        nowcast = nowcast_parts(frames, cfg, parts, n_threads=cores)
        log(f"nowcast samples: {nowcast}")
    t_now = time.perf_counter() - t_now

    t_pair = time.perf_counter()
    pair_summaries, skipped = [], []
    md = cfg["tasks"].get("multidoppler")
    if md and md.get("pairs") and md.get("grid"):
        pjobs, skipped = pair_jobs(cfg, records, ok)
        for j in pjobs:
            j.update(
                config=cfg,
                cache_dir=str(cache_dir),
                parts_dir=str(parts),
                n_threads=max(1, cores // max(1, min(workers, len(pjobs)))),
                max_elevation=md.get("max_elevation"),
            )
            gap = 60.0 * cfg["tasks"].get("nowcast", {}).get("max_gap_minutes", 12.0)
            j["motion"] = _pair_motion(frames, j["records"][0]["radar"], j["time"], gap)
        with ProcessPoolExecutor(
            max_workers=max(1, min(workers, len(pjobs) or 1))
        ) as pool:
            for s in pool.map(_safe, [process_pair] * len(pjobs), pjobs):
                pair_summaries.append(s)
                log(f"pair {s['volume_id']}: {s['error'] or 'ok'}")
    t_pair = time.perf_counter() - t_pair

    t_cons = time.perf_counter()
    prov = provenance(cfg)
    counts = consolidate(parts, out_dir, prov)
    t_cons = time.perf_counter() - t_cons
    if not keep_parts:
        shutil.rmtree(parts, ignore_errors=True)

    by_id = {s["volume_id"]: s for s in summaries}
    volumes = []
    for r in records:
        s = by_id.get(r["volume_id"], {})
        volumes.append(
            {
                **r,
                "samples": s.get("samples", {}),
                "seconds": s.get("seconds", {}),
                "error": s.get("error"),
            }
        )
    manifest = {
        **prov,
        "config": _jsonable(cfg),
        "cases": [
            {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in c.items()}
            for c in cfg["cases"]
        ],
        "volumes": volumes,
        "pairs": pair_summaries,
        "skipped_pairs": _jsonable(skipped),
        "samples": counts,
        "timing_seconds": {
            "volumes": round(t_vol, 2),
            "nowcast": round(t_now, 2),
            "multidoppler": round(t_pair, 2),
            "consolidate": round(t_cons, 2),
            "total": round(time.perf_counter() - t_start, 2),
        },
        "workers": workers,
        "threads_per_worker": n_threads,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1, default=str)
    failed = [s for s in summaries + pair_summaries if s["error"]]
    for s in failed:
        log(f"FAILED {s['volume_id']}: {s['error']}\n{s.get('traceback', '')}")
    return manifest
