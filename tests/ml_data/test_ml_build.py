"""The build pipeline end to end on synthetic volumes (no network)."""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import xarray as xr
from mldata import build as bd
from mldata import cases as cs
from mldata import nexrad, open_dataset, validate

from .conftest import synthetic_sweep, synthetic_volume

pytest.importorskip("zarr")

T0 = np.datetime64("2022-03-30T23:00:00")


def _key(radar, i):
    t = (T0 + np.timedelta64(300 * i, "s")).astype("datetime64[s]").item()
    return f"{t:%Y/%m/%d}/{radar}/{radar}{t:%Y%m%d_%H%M%S}_V06"


def _volume_for(key, max_elevation=None):
    radar, when = nexrad.parse_key(key.rsplit("/", 1)[-1])
    i = int((np.datetime64(when) - T0) / np.timedelta64(300, "s"))
    vol = synthetic_volume(t0=str(np.datetime64(when)), seed=i, radar=radar)
    nodes = {k: v.to_dataset(inherit=False) for k, v in vol.subtree_with_keys}
    # the echo turns 2 degrees per volume, so it moves between frames
    for name in [n for n in nodes if n.startswith("sweep")]:
        nodes[name] = nodes[name].roll(azimuth=2 * i, roll_coords=False)
    if radar == "KTWO":  # a second radar 60 km to the east
        nodes["."] = nodes["."].assign_coords(longitude=-87.65)
    return xr.DataTree.from_dict(nodes)


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(bd, "fetch", lambda key, cache: key)
    monkeypatch.setattr(bd, "read_volume", _volume_for)
    monkeypatch.setattr(bd, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(bd, "git_sha", lambda path=None: "abc123")


def _lister(bucket, prefix):
    radar = prefix.rstrip("/").rsplit("/", 1)[-1]
    n = 4 if radar == "KTST" else 1
    return [_key(radar, i) for i in range(n) if _key(radar, i).startswith(prefix)]


def _config():
    return cs.normalise(
        {
            "name": "synthetic",
            "seed": 5,
            "max_elevation": 2.0,
            "polar": {"n_azimuth": 90, "gate_spacing": 1000.0, "n_gates": 100},
            "tasks": {
                "qc": {},
                "kdp": {},
                "hid": {},
                "dealias": {"nyquist": [8.0, 12.0], "min_gates": 100},
                "inpaint": {},
                "nowcast": {
                    "grid": {"size": 64, "spacing": 2000.0, "z": [1000, 2000]},
                    "motion": {"tile": None, "floor": 10.0, "min_quality": 0.0},
                    "max_gap_minutes": 10,
                },
                "multidoppler": {
                    "grid": {
                        "x": [-20000, 60000, 4000],
                        "y": [-120000, -60000, 4000],
                        "z": [500, 2500, 1000],
                    },
                    "pairs": [
                        {"radars": ["KTST", "KTWO"], "time": "2022-03-30T23:00:00"},
                        {"radars": ["KTST", "KNONE"], "time": "2022-03-30T23:00:00"},
                    ],
                },
            },
            "cases": [
                {
                    "name": "synthetic-case",
                    "radar": "KTST",
                    "start": "2022-03-30T22:59:00",
                    "end": "2022-03-30T23:20:00",
                    "freezing_level": 3500,
                    "event": "e1",
                    "split": "train",
                },
                {
                    "name": "second-radar",
                    "radar": "KTWO",
                    "start": "2022-03-30T22:59:00",
                    "end": "2022-03-30T23:01:00",
                    "event": "e1",
                },
            ],
        }
    )


def test_build_end_to_end(tmp_path, offline):
    cfg = _config()
    lines = []
    manifest = bd.build(
        cfg,
        tmp_path / "out",
        cache_dir=tmp_path / "cache",
        workers=2,
        lister=_lister,
        log=lines.append,
    )
    assert not [v for v in manifest["volumes"] if v["error"]], lines
    assert manifest["samples"]["qc"] == {"train": 10}  # 5 volumes x 2 pol sweeps
    for task in ("qc", "kdp", "hid", "inpaint", "dealias", "nowcast", "multidoppler"):
        assert task in manifest["samples"], (task, lines)
        ds = open_dataset(tmp_path / "out", task, "train")
        validate(ds, task)
        assert ds.attrs["git_sha"] == "abc123"
        assert ds.attrs["task"] == task and ds.attrs["split"] == "train"
        assert "NOAA" in ds.attrs["licence"]
    qc = open_dataset(tmp_path / "out", "qc", "train")
    assert qc["DBZH"].encoding["chunks"] == (1, 90, 100)
    assert sorted(set(qc["radar"].values)) == ["KTST", "KTWO"]
    # time order of the samples and readable times
    times = qc["time"].values[qc["radar"].values == "KTST"]
    assert np.all(np.diff(times) >= np.timedelta64(0, "s"))
    dl = open_dataset(tmp_path / "out", "dealias", "train").load()
    assert np.all((dl.nyquist_velocity >= 8) & (dl.nyquist_velocity <= 12))
    recon = dl.VRADH_folded + 2 * dl.nyquist_velocity * dl.FOLD
    ok = np.isfinite(dl.VRADH)
    np.testing.assert_allclose(recon.values[ok], dl.VRADH.values[ok], atol=1e-3)
    now = open_dataset(tmp_path / "out", "nowcast", "train")
    assert now.sizes["sample"] == 2  # 4 frames, windows of 3
    md = open_dataset(tmp_path / "out", "multidoppler", "train")
    assert md["radar"].values.tolist() == [["KTST", "KTWO"]]
    with open(tmp_path / "out" / "manifest.json") as f:
        saved = json.load(f)
    assert saved["skipped_pairs"][0]["pair"]["radars"] == ["KTST", "KNONE"]
    assert saved["pairs"][0]["error"] is None
    assert not (tmp_path / "out" / "_parts").exists()
    # the same seed gives the same samples; small batches give the same store
    cfg["tasks"] = {"inpaint": cfg["tasks"]["inpaint"]}
    again = bd.build(
        cfg,
        tmp_path / "out2",
        cache_dir=tmp_path / "cache",
        workers=1,
        lister=_lister,
        log=lambda m: None,
        keep_parts=True,
    )
    assert again["samples"]["inpaint"] == manifest["samples"]["inpaint"]
    counts = bd.consolidate(tmp_path / "out2" / "_parts", tmp_path / "out3", {}, 1)
    assert counts == again["samples"]
    c = open_dataset(tmp_path / "out3", "inpaint", "train")
    np.testing.assert_array_equal(
        c.BLOCKAGE.values,
        open_dataset(tmp_path / "out2", "inpaint", "train").BLOCKAGE.values,
    )
    a = open_dataset(tmp_path / "out", "inpaint", "train").sortby("time")
    b = open_dataset(tmp_path / "out2", "inpaint", "train").sortby("time")
    np.testing.assert_array_equal(a.BLOCKAGE.values, b.BLOCKAGE.values)


def test_failed_volume_is_reported(tmp_path, offline, monkeypatch):
    def broken(key, max_elevation=None):
        raise OSError("corrupt file")

    monkeypatch.setattr(bd, "read_volume", broken)
    cfg = _config()
    cfg["tasks"].pop("multidoppler")
    lines = []
    manifest = bd.build(
        cfg,
        tmp_path / "out",
        cache_dir=tmp_path,
        workers=1,
        lister=_lister,
        log=lines.append,
    )
    assert all("corrupt file" in v["error"] for v in manifest["volumes"])
    assert manifest["samples"] == {}
    assert any(line.startswith("FAILED") for line in lines)


def test_process_volume_skips_sweeps_without_fields(tmp_path, monkeypatch):
    vol = synthetic_volume()
    nodes = {k: v.to_dataset(inherit=False) for k, v in vol.subtree_with_keys}
    nodes["sweep_1"] = nodes["sweep_1"].drop_vars("VRADH")
    monkeypatch.setattr(bd, "fetch", lambda key, cache: key)
    monkeypatch.setattr(
        bd, "read_volume", lambda p, m=None: xr.DataTree.from_dict(nodes)
    )
    cfg = _config()
    cfg["tasks"] = {"qc": {}, "dealias": cfg["tasks"]["dealias"]}
    rec = {
        "volume_id": "KTST_x",
        "key": "k",
        "radar": "KTST",
        "case": "c",
        "event": "e",
        "split": "val",
        "time": "t",
    }
    s = bd.process_volume(
        {
            "record": rec,
            "config": cfg,
            "cache_dir": "",
            "parts_dir": str(tmp_path),
            "n_threads": 1,
        }
    )
    assert s["samples"]["qc"] == 2 and s["frame"] is None
    assert s["samples"]["dealias"] == 1  # only the 1.5 degree sweep has VRADH


def test_sequences_and_git_sha():
    t = np.array(
        [
            "2022-01-01T00:00",
            "2022-01-01T00:05",
            "2022-01-01T00:10",
            "2022-01-01T00:40",
            "2022-01-01T00:45",
        ],
        "datetime64[ns]",
    )
    assert bd.sequences(t, 3, 600.0) == [0]
    assert bd.sequences(t, 2, 600.0) == [0, 1, 3]
    assert isinstance(bd.git_sha(), str)
    assert bd.git_sha("/") in ("unknown",) or len(bd.git_sha("/")) == 40


def test_unique_sweeps_and_kind():
    vol = synthetic_volume()
    assert nexrad.unique_sweeps(vol, "polarimetric") == ["sweep_0", "sweep_2"]
    assert nexrad.unique_sweeps(vol, "velocity") == ["sweep_1", "sweep_2"]
    assert nexrad.sweep_kind(synthetic_sweep(polarimetric=False)) == "doppler"
    assert (
        nexrad.sweep_kind(synthetic_sweep(polarimetric=False, velocity=False)) is None
    )
    assert nexrad.sweep_kind(xr.Dataset()) is None


def test_mask_nodata():
    ds = xr.Dataset(
        {
            "DBZH": ("x", np.array([-32.0, -31.5])),
            "VRADH": ("x", np.array([-64.0, 3.0])),
            "PHIDP": ("x", np.array([-0.5, 10.0])),
        }
    )
    out = nexrad.mask_nodata(ds)
    assert np.isnan(out.DBZH[0]) and float(out.DBZH[1]) == -31.5
    assert np.isnan(out.VRADH[0]) and np.isnan(out.PHIDP[0])
