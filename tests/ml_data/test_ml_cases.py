"""Case configs, splits, leakage checks and the dataset schemas."""

from datetime import datetime

import numpy as np
import pytest
import xarray as xr
from mldata import SCHEMAS, TASKS, nexrad, open_dataset, validate
from mldata import cases as cs
from mldata.schema import attrs_for, variables


def _cfg(**over):
    cfg = {
        "seed": 1,
        "cases": [
            {
                "name": "a",
                "radar": "kgwx",
                "start": "2022-03-30T23:00Z",
                "end": "2022-03-31T00:00",
            },
            {
                "name": "b",
                "radar": "KBMX",
                "start": "2022-03-30T23:30",
                "end": "2022-03-31T00:00",
                "event": "2022-03-30",
            },
            {
                "name": "c",
                "radar": "KHGX",
                "start": datetime(2017, 8, 27, 12),
                "end": "2017-08-27T13:00",
            },
            {
                "name": "d",
                "radar": "KOKX",
                "start": "2022-01-29T12:00",
                "end": "2022-01-29T13:00",
            },
        ],
    }
    cfg.update(over)
    return cs.normalise(cfg)


def test_normalise_defaults_and_cases():
    cfg = _cfg(tasks={"qc": {}, "dealias": {"nyquist": [5, 9]}, "hid": False})
    assert set(cfg["tasks"]) == {"qc", "dealias"}
    assert cfg["tasks"]["dealias"]["nyquist"] == [5, 9]
    assert cfg["tasks"]["dealias"]["max_jump_fraction"] == 5e-4  # default kept
    a = cfg["cases"][0]
    assert a["radar"] == "KGWX" and a["event"] == "2022-03-30"
    assert a["start"] == datetime(2022, 3, 30, 23)
    assert cfg["polar"]["n_azimuth"] == 360
    assert set(_cfg()["tasks"]) == set(cs.DEFAULTS["tasks"])


@pytest.mark.parametrize(
    "case, match",
    [
        ({"name": "x", "radar": "K", "start": "2020-01-01"}, "lacks"),
        (
            {"name": "x", "radar": "K", "start": "2020-01-02", "end": "2020-01-01"},
            "ends before",
        ),
        (
            {
                "name": "x",
                "radar": "K",
                "start": "2020-01-01",
                "end": "2020-01-01",
                "split": "dev",
            },
            "unknown split",
        ),
    ],
)
def test_normalise_errors(case, match):
    with pytest.raises(ValueError, match=match):
        cs.normalise({"cases": [case]})


def test_normalise_duplicate_and_fractions():
    c = {"name": "x", "radar": "K", "start": "2020-01-01", "end": "2020-01-01"}
    with pytest.raises(ValueError, match="duplicate"):
        cs.normalise({"cases": [c, c]})
    with pytest.raises(ValueError, match="fractions"):
        cs.normalise({"splits": {"fractions": {"train": 0.5, "val": 0.1}}})


def test_assign_splits_by_event_and_holdout():
    cfg = _cfg(splits={"holdout_radars": ["KOKX"]})
    cs.assign_splits(cfg)
    split = {c["name"]: c["split"] for c in cfg["cases"]}
    assert split["a"] == split["b"]  # same event
    assert split["d"] == "test"  # held-out radar
    assert split["c"] == cs.hash_split("2017-08-27", 1, cfg["splits"]["fractions"])
    cs.check_no_leakage(cfg["cases"], 24, ["KOKX"])
    # explicit split propagates to the whole event
    cfg = _cfg()
    cfg["cases"][0]["split"] = "val"
    cs.assign_splits(cfg)
    assert cfg["cases"][1]["split"] == "val"


def test_assign_splits_conflicts():
    cfg = _cfg()
    cfg["cases"][0]["split"], cfg["cases"][1]["split"] = "train", "test"
    with pytest.raises(ValueError, match="several|splits"):
        cs.assign_splits(cfg)
    cfg = _cfg(splits={"holdout_radars": ["KBMX"]})
    cfg["cases"][0]["split"] = "train"
    with pytest.raises(ValueError, match="held-out"):
        cs.assign_splits(cfg)


def test_hash_split_is_deterministic_and_balanced():
    fr = {"train": 0.7, "val": 0.15, "test": 0.15}
    days = [f"2020-{m:02d}-{d:02d}" for m in range(1, 13) for d in range(1, 29)]
    got = [cs.hash_split(d, 0, fr) for d in days]
    assert got == [cs.hash_split(d, 0, fr) for d in days]
    share = got.count("train") / len(got)
    assert 0.6 < share < 0.8
    assert cs.hash_split("x", 0, {"train": 0.0, "val": 0.0, "test": 0.0}) == "test"


def test_check_no_leakage():
    cfg = _cfg()
    for c, s in zip(cfg["cases"], ["train", "val", "test", "test"]):
        c["split"] = s
    with pytest.raises(ValueError, match="less than"):
        cs.check_no_leakage(cfg["cases"], 24)
    cfg["cases"][1]["split"] = "train"
    cs.check_no_leakage(cfg["cases"], 24)
    with pytest.raises(ValueError, match="held-out"):
        cs.check_no_leakage(cfg["cases"], 24, ["KGWX"])


def test_select_and_expand_with_fake_archive():
    keys = [
        "2022/03/30/KGWX/KGWX20220330_225959_V06",
        "2022/03/30/KGWX/KGWX20220330_230555_V06",
        "2022/03/30/KGWX/KGWX20220330_230555_V06.gz",
        "2022/03/30/KGWX/KGWX20220330_231240_V06.gz",
        "2022/03/30/KGWX/KGWX20220330_235959_V06_MDM",
        "2022/03/30/KGWX/readme.txt",
    ]
    picked = nexrad.select_keys(keys, datetime(2022, 3, 30, 23), datetime(2022, 3, 31))
    assert picked == [keys[1], keys[3]]
    assert nexrad.parse_key(keys[-2]) is None
    assert nexrad.parse_key(keys[0]) == ("KGWX", datetime(2022, 3, 30, 22, 59, 59))

    prefixes = []

    def lister(bucket, prefix):
        prefixes.append(prefix)
        return [k for k in keys if k.startswith(prefix)]

    cfg = cs.normalise(
        {
            "max_volumes_per_case": 1,
            "cases": [
                {
                    "name": "a",
                    "radar": "KGWX",
                    "start": "2022-03-30T23:00",
                    "end": "2022-03-31T00:30",
                    "split": "train",
                }
            ],
        }
    )
    recs = cs.expand_cases(cfg, lister=lister)
    assert prefixes == ["2022/03/30/KGWX/", "2022/03/31/KGWX/"]
    assert len(recs) == 1
    r = recs[0]
    assert r["volume_id"] == "KGWX_20220330T230555" and r["split"] == "train"


def test_sample_rng_reproducible():
    a = cs.sample_rng(3, "KGWX_x", "sweep_0", "inpaint").uniform(size=3)
    b = cs.sample_rng(3, "KGWX_x", "sweep_0", "inpaint").uniform(size=3)
    c = cs.sample_rng(3, "KGWX_x", "sweep_1", "inpaint").uniform(size=3)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_load_config_yaml(tmp_path):
    pytest.importorskip("yaml")
    from pathlib import Path

    demo = Path(cs.__file__).resolve().parents[1] / "configs" / "demo.yaml"
    cfg = cs.load_config(demo)
    cs.assign_splits(cfg)
    cs.check_no_leakage(
        cfg["cases"], cfg["splits"]["min_gap_hours"], cfg["splits"]["holdout_radars"]
    )
    assert {c["split"] for c in cfg["cases"]} == {"train", "val", "test"}
    other = Path(cs.__file__).resolve().parents[1] / "configs" / "conus_cases.yaml"
    cfg = cs.load_config(other)
    cs.assign_splits(cfg)
    cs.check_no_leakage(
        cfg["cases"], cfg["splits"]["min_gap_hours"], cfg["splits"]["holdout_radars"]
    )


def test_schema_tables():
    assert TASKS == (
        "qc",
        "kdp",
        "hid",
        "dealias",
        "inpaint",
        "nowcast",
        "multidoppler",
    )
    for task in TASKS:
        spec = SCHEMAS[task]
        assert spec["description"]
        assert variables(task, "label"), task
        assert variables(task, "input"), task
        for name in spec["variables"]:
            a = attrs_for(task, name)
            assert {"long_name", "role"} <= set(a)
            assert ("units" in a) == (
                np.dtype(spec["variables"][name]["dtype"]).kind != "M"
            )
    assert attrs_for("qc", "ECHO_CLASS")["flag_values"] == [0, 1, 2, 3]


def test_validate_reports_problems():
    ds = xr.Dataset(
        {"DBZH": (("sample", "azimuth", "range"), np.zeros((1, 2, 3), "float64"))}
    )
    with pytest.raises(ValueError) as err:
        validate(ds, "qc")
    msg = str(err.value)
    assert "missing ECHO_CLASS" in msg and "DBZH: dtype" in msg
    ds = xr.Dataset({"radar": (("sample",), np.zeros(1))})
    with pytest.raises(ValueError, match="not a string"):
        validate(ds, "nowcast")
    ds = xr.Dataset({"DBZH": (("sample", "y", "x"), np.zeros((1, 2, 2), "f4"))})
    with pytest.raises(ValueError, match="dims"):
        validate(ds, "nowcast")
    with pytest.raises(ValueError, match="unknown task"):
        validate(ds, "nope")
    with pytest.raises(ValueError, match="task must be"):
        open_dataset(".", "nope", "train")
