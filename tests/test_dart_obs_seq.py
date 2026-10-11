#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the DART obs_seq writer
=================================

The reader below follows only the layout that the DART source defines
(``write_obs_seq``, ``write_obs``, ``write_obs_def``, ``write_location``,
``write_time``, ``write_radial_vel``; quoted in the module docstring of
``radarx.io.dart``). Every field of a small example is checked by hand. DART
itself is not installed, so these tests do not show that DART accepts the
file.
"""

import datetime as dt
import math

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.io import dart, write_dart_obs_seq, write_wrfda_radar
from radarx.retrieve import superobs

ENGINES = ["python"] + (["compiled"] if dart.HAS_COMPILED_FORMATTER else [])
compiled_only = pytest.mark.skipif(
    not dart.HAS_COMPILED_FORMATTER, reason="compiled formatter not built"
)
MISSING_R8 = -888888.0


def gregorian_days(year, month, day):
    """Days since 1601-01-01 by the algorithm of set_date_gregorian (time_manager_mod)."""
    leap = lambda y: y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)  # noqa: E731
    base = 1601
    n_leap = (year - base) // 4 - (year - base) // 100 + (year - base) // 400
    per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    ndays = sum(per_month[: month - 1]) + (1 if leap(year) and month > 2 else 0)
    return day - 1 + ndays + 365 * (year - base - n_leap) + 366 * n_leap


def example():
    """Two radars, four boxes, written down by hand."""
    obs = {
        # radar, lat, lon, height, V, V error, Z, Z error, nonprecip, e, n, u
        "radar_index": [0, 0, 1],
        "latitude": [35.5, 35.75, 36.0],
        "longitude": [-97.0, -96.5, 10.25],
        "height": [1250.0, 2250.0, 750.0],
        "velocity": [10.0, np.nan, -5.5],
        "velocity_error": [2.0, np.nan, 3.0],
        "reflectivity": [30.0, np.nan, np.nan],
        "reflectivity_error": [5.0, 5.0, np.nan],
        "nonprecip": np.array([0, 1, 0], np.int8),
        "radial_east": [0.6, np.nan, -0.28],
        "radial_north": [0.8, np.nan, 0.96],
        "radial_up": [0.0, np.nan, 0.0],
    }
    data = {k: ("obs", np.asarray(v)) for k, v in obs.items()}
    return xr.Dataset(
        data,
        coords={
            "radar_name": ("radar", ["AAAA", "BBBB"]),
            "radar_latitude": ("radar", [35.25, 50.0]),
            "radar_longitude": ("radar", [-97.5, -10.5]),
            "radar_altitude": ("radar", [300.0, 40.0]),
            "radar_time": (
                "radar",
                np.array(
                    ["2022-03-31T00:05:00", "2022-03-31T00:00:00"], "datetime64[ns]"
                ),
            ),
        },
    )


def read_obs_seq(path):
    """Reader written from the layout of the DART write routines only."""
    lines = open(path, encoding="ascii").read().split("\n")
    assert lines.pop() == ""
    at = 0

    def take():
        nonlocal at
        at += 1
        return lines[at - 1]

    assert take().strip() == "obs_sequence"
    assert take() == "obs_type_definitions"  # starts in column 1
    ntypes = int(take())
    types = {}
    for _ in range(ntypes):
        index, name = take().split()
        types[int(index)] = name
    t = take().split()
    assert t[0] == "num_copies:" and t[2] == "num_qc:"
    num_copies, num_qc = int(t[1]), int(t[3])
    t = take().split()
    assert t[0] == "num_obs:" and t[2] == "max_num_obs:"
    num_obs, max_num_obs = int(t[1]), int(t[3])
    copies = [take() for _ in range(num_copies)]
    qcs = [take() for _ in range(num_qc)]
    t = take().split()
    assert t[0] == "first:" and t[2] == "last:"
    first, last = int(t[1]), int(t[3])
    records = []
    for _ in range(num_obs):
        t = take().split()
        assert t[0] == "OBS"
        rec = {"key": int(t[1])}
        rec["values"] = [float(take()) for _ in range(num_copies)]
        rec["qc"] = [int(float(take())) for _ in range(num_qc)]
        rec["prev"], rec["next"], rec["cov"] = (int(x) for x in take().split())
        assert take() == "obdef"
        assert take() == "loc3d"
        lon, lat, vloc, which = take().split()
        rec["location"] = (float(lon), float(lat), float(vloc), int(which))
        assert take() == "kind"
        rec["type"] = types[int(take())]
        if rec["type"] == "DOPPLER_RADIAL_VELOCITY":
            assert take() == "platform"
            assert take() == "loc3d"
            lon, lat, vloc, which = take().split()
            rec["radar"] = (float(lon), float(lat), float(vloc), int(which))
            assert take() == "dir3d"
            rec["beam"] = tuple(float(x) for x in take().split())
            assert len(rec["beam"]) == 3
            rec["nyquist"] = float(take())
            rec["velkey"] = int(take())
        secs, days = take().split()
        rec["time"] = (int(secs), int(days))
        rec["variance"] = float(take())
        records.append(rec)
    assert at == len(lines)
    return {
        "types": types,
        "num_copies": num_copies,
        "num_qc": num_qc,
        "num_obs": num_obs,
        "max_num_obs": max_num_obs,
        "copies": copies,
        "qcs": qcs,
        "first": first,
        "last": last,
        "records": records,
    }


@pytest.mark.parametrize("engine", ENGINES)
def test_every_field_of_a_small_example(tmp_path, engine):
    path = tmp_path / "obs_seq.out"
    stats = write_dart_obs_seq(example(), path, engine=engine)
    assert stats == {
        "observations": 4,
        "velocities": 2,
        "reflectivities": 1,
        "clearair": 1,
    }
    got = read_obs_seq(path)
    # header
    assert got["types"] == {
        1: "DOPPLER_RADIAL_VELOCITY",
        2: "RADAR_REFLECTIVITY",
        3: "RADAR_CLEARAIR_REFLECTIVITY",
    }
    assert (got["num_copies"], got["num_qc"]) == (1, 1)
    assert got["num_obs"] == got["max_num_obs"] == 4
    assert got["copies"] == ["observation"] and got["qcs"] == ["Data QC"]
    assert (got["first"], got["last"]) == (1, 4)
    r = got["records"]
    # time order: radar BBBB (00:00:00) before AAAA (00:05:00); the velocity of a box
    # before its reflectivity
    assert [x["type"] for x in r] == [
        "DOPPLER_RADIAL_VELOCITY",
        "DOPPLER_RADIAL_VELOCITY",
        "RADAR_REFLECTIVITY",
        "RADAR_CLEARAIR_REFLECTIVITY",
    ]
    assert [x["key"] for x in r] == [1, 2, 3, 4]
    assert [x["prev"] for x in r] == [-1, 1, 2, 3]
    assert [x["next"] for x in r] == [2, 3, 4, -1]
    assert all(x["cov"] == -1 and x["qc"] == [0] for x in r)
    deg = math.pi / 180.0
    days = gregorian_days(2022, 3, 31)
    # key 1: velocity of the box of BBBB
    assert r[0]["values"] == [-5.5]
    assert r[0]["location"] == pytest.approx(
        (10.25 * deg, 36.0 * deg, 750.0, 3), rel=1e-15
    )
    assert r[0]["radar"] == pytest.approx((349.5 * deg, 50.0 * deg, 40.0, 3), rel=1e-15)
    assert r[0]["beam"] == pytest.approx((-0.28, 0.96, 0.0), rel=1e-15, abs=1e-15)
    assert r[0]["nyquist"] == MISSING_R8
    assert r[0]["velkey"] == 1
    assert r[0]["time"] == (0, days)
    assert r[0]["variance"] == pytest.approx(9.0, rel=1e-15)
    # key 2: velocity of the first box of AAAA, longitude -97 as 263 degrees
    assert r[1]["values"] == [10.0]
    assert r[1]["location"] == pytest.approx(
        (263.0 * deg, 35.5 * deg, 1250.0, 3), rel=1e-15
    )
    assert r[1]["radar"] == pytest.approx(
        (262.5 * deg, 35.25 * deg, 300.0, 3), rel=1e-15
    )
    assert r[1]["beam"] == pytest.approx((0.6, 0.8, 0.0), rel=1e-15, abs=1e-15)
    assert r[1]["velkey"] == 2
    assert r[1]["time"] == (300, days)
    assert r[1]["variance"] == pytest.approx(4.0, rel=1e-15)
    # key 3: its reflectivity, which has no radar block
    assert r[2]["values"] == [30.0] and "radar" not in r[2]
    assert r[2]["location"] == r[1]["location"]
    assert r[2]["time"] == (300, days)
    assert r[2]["variance"] == pytest.approx(25.0, rel=1e-15)
    # key 4: the box without echo as clear-air reflectivity at -10 dBZ
    assert r[3]["values"] == [-10.0]
    assert r[3]["location"] == pytest.approx(
        (263.5 * deg, 35.75 * deg, 2250.0, 3), rel=1e-15
    )
    assert r[3]["variance"] == pytest.approx(25.0, rel=1e-15)


def test_days_since_1601():
    # the algorithm of the time manager and the calendar agree
    for date in [(1601, 1, 1), (1900, 3, 1), (2000, 2, 29), (2022, 3, 31)]:
        want = (dt.date(*date) - dt.date(1601, 1, 1)).days
        assert gregorian_days(*date) == want
    assert gregorian_days(1601, 1, 1) == 0


@pytest.mark.parametrize("engine", ENGINES)
def test_options(tmp_path, engine):
    ds = example()
    path = tmp_path / "a"
    stats = write_dart_obs_seq(
        ds,
        path,
        engine=engine,
        clear_air=False,
        nyquist_velocity=[27.5, 13.0],
        time="2020-02-29_12:30:15",
        copy_name="observation NEXRAD",
        qc_name="radarx QC",
    )
    assert stats["clearair"] == 0 and stats["observations"] == 3
    got = read_obs_seq(path)
    assert "RADAR_CLEARAIR_REFLECTIVITY" not in got["types"].values()
    assert got["copies"] == ["observation NEXRAD"] and got["qcs"] == ["radarx QC"]
    days = gregorian_days(2020, 2, 29)
    for rec in got["records"]:
        assert rec["time"] == (12 * 3600 + 30 * 60 + 15, days)
    # equal times keep the order of the Dataset, with the velocity before the reflectivity
    velocities = [x for x in got["records"] if x["type"] == "DOPPLER_RADIAL_VELOCITY"]
    assert [x["values"][0] for x in velocities] == [10.0, -5.5]
    assert [x["nyquist"] for x in velocities] == [27.5, 13.0]
    assert [x["type"] for x in got["records"]] == [
        "DOPPLER_RADIAL_VELOCITY",
        "RADAR_REFLECTIVITY",
        "DOPPLER_RADIAL_VELOCITY",
    ]
    b = tmp_path / "b"
    write_dart_obs_seq(ds, b, engine=engine, clearair_dbz=-3.5, nyquist_velocity=20.0)
    clear = [
        x
        for x in read_obs_seq(b)["records"]
        if x["type"].endswith("CLEARAIR_REFLECTIVITY")
    ]
    assert [x["values"] for x in clear] == [[-3.5]]
    assert all(
        x["nyquist"] == 20.0
        for x in read_obs_seq(b)["records"]
        if x["type"] == "DOPPLER_RADIAL_VELOCITY"
    )


def test_nothing_to_write_gives_a_valid_file(tmp_path):
    ds = example().isel(obs=slice(0, 0))
    path = tmp_path / "empty"
    stats = write_dart_obs_seq(ds, path, engine="python")
    assert stats["observations"] == 0
    text = open(path).read().split("\n")
    assert text[:3] == ["obs_sequence", "obs_type_definitions", "           0"]
    assert text[-2].split() == ["first:", "-1", "last:", "-1"]


@compiled_only
def test_compiled_formatter_gives_identical_bytes(tmp_path):
    ds = example()
    files = []
    for name, kwargs in (
        ("p", dict(engine="python")),
        ("c1", dict(engine="compiled", n_threads=1)),
        ("c5", dict(engine="compiled", n_threads=5)),
    ):
        path = tmp_path / name
        write_dart_obs_seq(ds, path, nyquist_velocity=26.25, **kwargs)
        files.append(path.read_bytes())
    assert files[0] == files[1] == files[2]


@compiled_only
def test_formatters_agree_on_many_records(tmp_path):
    """More records than one block of the formatter (2048), tiled from the example."""
    base = example()
    idx = np.arange(5000) % 3
    big = base.isel(obs=idx)
    times = {}
    for name, kwargs in (
        ("p", dict(engine="python")),
        ("c1", dict(engine="compiled", n_threads=1)),
        ("c7", dict(engine="compiled", n_threads=7)),
    ):
        path = tmp_path / name
        write_dart_obs_seq(big, path, **kwargs)
        times[name] = path.read_bytes()
    assert times["p"] == times["c1"] == times["c7"]
    got = read_obs_seq(tmp_path / "p")
    assert got["num_obs"] == len(got["records"]) == 6667  # 2 + 1 + 1 per three boxes
    keys = [x["key"] for x in got["records"]]
    assert keys == list(range(1, got["num_obs"] + 1))
    assert got["records"][0]["prev"] == -1 and got["records"][-1]["next"] == -1
    assert [x["next"] for x in got["records"][:-1]] == keys[1:]


def test_errors(tmp_path):
    ds = example()
    with pytest.raises(ValueError, match="superobs"):
        write_dart_obs_seq(xr.Dataset(), tmp_path / "x")
    with pytest.raises(ValueError, match="engine"):
        write_dart_obs_seq(ds, tmp_path / "x", engine="gpu")
    with pytest.raises(ValueError, match="metadata"):
        write_dart_obs_seq(ds, tmp_path / "x", copy_name="x" * 65)
    with pytest.raises(ValueError, match="negative"):
        write_dart_obs_seq(ds, tmp_path / "x", nyquist_velocity=-1.0)
    bad = ds.copy(deep=True)
    bad["velocity_error"].values[0] = np.nan
    with pytest.raises(ValueError, match="velocity"):
        write_dart_obs_seq(bad, tmp_path / "x")
    stamp = ds.assign_coords(
        radar_time=("radar", np.array(["NaT", "NaT"], "datetime64[ns]"))
    )
    with pytest.raises(ValueError, match="time"):
        write_dart_obs_seq(stamp, tmp_path / "x")
    write_dart_obs_seq(
        stamp, tmp_path / "ok", time=np.datetime64("2001-09-09T01:46:40")
    )


# --------------------------------------------------------------------------
# super-observations from sweeps, and the WRFDA formatter
# --------------------------------------------------------------------------


def test_superobs_end_to_end(tmp_path):
    from tests.test_superobs import volume

    so = superobs(volume(), 8.0)
    path = tmp_path / "obs_seq"
    stats = write_dart_obs_seq(so, path)
    got = read_obs_seq(path)
    n_v = int(so.velocity.notnull().sum())
    n_z = int((so.reflectivity.notnull()).sum())
    n_c = int(so.nonprecip.sum())
    assert stats == {
        "observations": n_v + n_z + n_c,
        "velocities": n_v,
        "reflectivities": n_z,
        "clearair": n_c,
    }
    assert got["num_obs"] == n_v + n_z + n_c
    vel = [x for x in got["records"] if x["type"] == "DOPPLER_RADIAL_VELOCITY"]
    np.testing.assert_allclose(
        sorted(x["values"][0] for x in vel), np.sort(so.velocity.dropna("obs").values)
    )
    np.testing.assert_allclose(
        sorted(x["variance"] for x in vel),
        np.sort(
            so.velocity_error.where(so.velocity.notnull()).dropna("obs").values ** 2
        ),
    )
    for x in got["records"]:
        lon, lat, height, which = x["location"]
        assert 0.0 <= lon < 2 * math.pi and -math.pi / 2 <= lat <= math.pi / 2
        assert which == 3 and 0 <= height < 12000
    assert all(max(abs(c) for c in x["beam"]) <= 1.0 for x in vel)


@compiled_only
def test_wrfda_formatters_give_identical_bytes(tmp_path):
    from tests.test_superobs import model_grid, volume

    so = superobs(
        volume(elevations=(0.5, 1.5), ngate=140),
        model_grid(n=60, dx=3000.0),
        layers=500.0,
        top=6000.0,
        terrain=np.full((60, 60), 123.0),
    )
    # tile the boxes into more than one block of the formatter (2048 columns)
    tiles = 60
    big = so.isel(obs=np.arange(so.sizes["obs"] * tiles) % so.sizes["obs"])
    shift = np.arange(big.sizes["obs"]) // so.sizes["obs"] * 10**6
    big = big.assign_coords(column=("obs", big.column.values + shift))
    files = []
    for name, kwargs in (
        ("p", dict(engine="python")),
        ("c1", dict(engine="compiled", n_threads=1)),
        ("c6", dict(engine="compiled", n_threads=6)),
    ):
        path = tmp_path / name
        stats = write_wrfda_radar(big, path, time="2022-03-31_00:00:00", **kwargs)
        files.append(path.read_bytes())
    assert files[0] == files[1] == files[2]
    assert stats["columns"] > 2048
    with pytest.raises(ValueError, match="engine"):
        write_wrfda_radar(so, tmp_path / "x", engine="gpu")


def _wrfda_example(n_radars):
    """One velocity box for each of ``n_radars`` radars, in the layout of superobs."""
    k = np.arange(n_radars)
    return xr.Dataset(
        {
            "reflectivity": ("obs", np.full(n_radars, 25.0)),
            "velocity": ("obs", np.full(n_radars, 5.0)),
            "reflectivity_error": ("obs", np.full(n_radars, 5.0)),
            "velocity_error": ("obs", np.full(n_radars, 2.0)),
            "nonprecip": ("obs", np.zeros(n_radars, bool)),
            "latitude": ("obs", np.full(n_radars, 35.0)),
            "longitude": ("obs", np.full(n_radars, -97.0)),
            "level": ("obs", np.zeros(n_radars, np.int64)),
            "radar_index": ("obs", k),
            "column": ("obs", np.zeros(n_radars, np.int64)),
            "radar_name": ("radar", np.array([f"R{i:03d}" for i in k])),
            "radar_latitude": ("radar", np.full(n_radars, 35.5)),
            "radar_longitude": ("radar", np.full(n_radars, -97.5)),
            "radar_altitude": ("radar", np.full(n_radars, 100.0)),
            "level_edges": ("edge", np.array([0.0, 500.0, 1000.0])),
        }
    )


def test_compiled_formatter_is_required_when_asked(tmp_path, monkeypatch):
    from radarx.io import wrfda

    so = _wrfda_example(1)
    stamp = "2022-03-31_00:00:00"
    monkeypatch.setattr(dart, "HAS_COMPILED_FORMATTER", False)
    monkeypatch.setattr(wrfda, "HAS_COMPILED_FORMATTER", False)
    with pytest.raises(ImportError, match="compiled formatter is not available"):
        write_dart_obs_seq(example(), tmp_path / "a", engine="compiled")
    with pytest.raises(ImportError, match="compiled formatter is not available"):
        write_wrfda_radar(so, tmp_path / "b", time=stamp, engine="compiled")
    # 'auto' falls back to the Python formatter
    write_dart_obs_seq(example(), tmp_path / "c")
    write_wrfda_radar(so, tmp_path / "d", time=stamp)
    assert (tmp_path / "d").read_text().startswith("TOTAL NUMBER =  1")


def test_wrfda_holds_at_most_999_radars(tmp_path):
    stamp = "2022-03-31_00:00:00"
    write_wrfda_radar(_wrfda_example(999), tmp_path / "ok", time=stamp, engine="python")
    with pytest.raises(ValueError, match="at most 999 radars"):
        write_wrfda_radar(
            _wrfda_example(1000), tmp_path / "x", time=stamp, engine="python"
        )


def test_dart_checks_every_value_it_needs(tmp_path):
    ds = example()
    # a reflectivity without an error estimate (box 0 has a reflectivity)
    no_err = ds.copy(deep=True)
    no_err["reflectivity_error"].values[0] = np.nan
    with pytest.raises(ValueError, match="a reflectivity has no error estimate"):
        write_dart_obs_seq(no_err, tmp_path / "x")
    # the clear-air box (1) needs its error only when clear air is written
    clear = ds.copy(deep=True)
    clear["reflectivity_error"].values[1] = np.nan
    with pytest.raises(ValueError, match="a reflectivity has no error estimate"):
        write_dart_obs_seq(clear, tmp_path / "x")
    write_dart_obs_seq(clear, tmp_path / "ok", clear_air=False)
    for name in ("latitude", "longitude", "height"):
        bad = ds.copy(deep=True)
        bad[name].values[2] = np.nan
        with pytest.raises(ValueError, match=f"every box needs a finite {name}"):
            write_dart_obs_seq(bad, tmp_path / "x")
    no_beam = ds.copy(deep=True)
    no_beam["radial_east"].values[0] = np.nan
    with pytest.raises(ValueError, match="velocity has no radial unit vector"):
        write_dart_obs_seq(no_beam, tmp_path / "x")
