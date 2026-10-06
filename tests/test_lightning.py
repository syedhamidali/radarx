#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for radarx.io.lma and radarx.retrieve.lightning."""

import gzip
import importlib

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.io import read_lma
from radarx.retrieve import (
    cell_flash_rate,
    cluster_flashes,
    grid_lightning,
    lightning_jump,
    vertical_source_distribution,
)

L = importlib.import_module("radarx.retrieve.lightning")
lma_io = importlib.import_module("radarx.io.lma")

ENGINES = ["numpy"] + (["compiled"] if L.HAS_COMPILED_KERNEL else [])
IO_ENGINES = ["numpy"] + (["compiled"] if lma_io.HAS_COMPILED_KERNEL else [])
needs_kernel = pytest.mark.skipif(
    not L.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)

LAT0, LON0 = 33.5, -88.5
DAY = np.datetime64("2022-03-30T00:00:00", "ns")

HEADER = """Lightning Mapping Array analyzed data
Analysis program: lma_analysis -d 20220330 -t 230000 -s 600
Analysis program version: 10.14.9R
File created: Thu Mar 31 04:46:26 2022
Data start time: 03/30/22 23:00:00
Number of seconds analyzed: 600
Location: TEST
Coordinate center (lat,lon,alt): 33.5000000 -88.5000000 0.00
Coordinate frame: cartesian
Number of stations: 3
Station information: id, name, lat(d), lon(d), alt(m), delay(ns), board_rev, rec_ch
Sta_info: A  Site 1             33.8896328  -89.0188344    83.52  100 52  3
Sta_info: B  Site 2             33.4405831  -88.8309586    74.73  100 52  3
Sta_info: C  long name here     33.7498089  -88.6912139    56.38  100 52  3
Station data: id, name, win(us), dec_win(us), data_ver, rms_error(ns), sources, %, <P/P_m>, active
Sta_data: A  Site 1              80    12   70   914793  82.6  1.82   A
Sta_data: B  Site 2              80    12   70   218586  19.7  2.15   A
Sta_data: C  long name here      80    12   70   946493  85.5  1.30   NA
Metric file version: 4
Station mask order: CBA
Data: time (UT sec of day), lat, lon, alt(m), reduced chi^2, P(dBW), mask
Data format: 15.9f 12.8f 13.8f 9.2f 6.2f 5.1f 6x
Number of events: 4
*** data ***
"""
ROWS = """82800.002232494  33.58691706  -88.89849663  13434.35   3.90   8.4 0xf9
82800.000124000  33.50000000  -88.50000000   9000.00   0.10  11.0 0x7
82801.500000000  33.51000000  -88.51000000   8000.50   0.50  -2.5 0x3

82802.250000000  33.52000000  -88.52000000   7000.25   1.50  20.2 0x1ff
"""


def _write(tmp_path, name="LYLOUT_220330_230000_0600.dat", rows=ROWS, compress=None):
    text = HEADER + rows
    path = tmp_path / (name + (".gz" if compress == "gz" else ""))
    if compress == "gz":
        with gzip.open(path, "wt") as f:
            f.write(text)
    else:
        path.write_text(text)
    return path


def _sources(flashes, seed=0, spacing_s=1.0):
    """Synthetic flashes (y, x, n): compact source clouds well apart in time."""
    rng = np.random.default_rng(seed)
    t, lat, lon, alt = [], [], [], []
    truth = []
    for k, (y, x, n) in enumerate(flashes):
        t.append(k * spacing_s + np.sort(rng.uniform(0, 0.2, n)))
        dlat = (y + rng.normal(0, 300.0, n)) / 111.2e3
        dlon = (x + rng.normal(0, 300.0, n)) / (111.2e3 * np.cos(np.radians(LAT0)))
        lat.append(LAT0 + dlat)
        lon.append(LON0 + dlon)
        alt.append(rng.uniform(6e3, 10e3, n))
        truth.append(np.full(n, k))
    t = np.concatenate(t)
    order = np.argsort(t, kind="stable")
    ds = xr.Dataset(
        {
            "event_altitude": ("number_of_events", np.concatenate(alt)[order]),
            "event_chi2": ("number_of_events", np.full(t.size, 0.5)),
        },
        coords={
            "event_time": (
                "number_of_events",
                DAY + (t[order] * 1e9).astype("timedelta64[ns]"),
            ),
            "event_latitude": ("number_of_events", np.concatenate(lat)[order]),
            "event_longitude": ("number_of_events", np.concatenate(lon)[order]),
        },
    )
    return ds, np.concatenate(truth)[order]


# --------------------------------------------------------------------------
# reader
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", IO_ENGINES)
@pytest.mark.parametrize("compress", [None, "gz"])
def test_read_lma(tmp_path, engine, compress):
    ds = read_lma(_write(tmp_path, compress=compress), engine=engine)
    assert ds.sizes["number_of_events"] == 4
    # sorted by time
    assert np.all(np.diff(ds.event_time.values) >= np.timedelta64(0, "ns"))
    t0 = np.datetime64("2022-03-30T23:00:00.000124", "ns")
    assert ds.event_time.values[0] == t0
    np.testing.assert_allclose(
        ds.event_altitude.values, [9000.0, 13434.35, 8000.5, 7000.25]
    )
    np.testing.assert_array_equal(ds.event_mask.values, [7, 0xF9, 3, 0x1FF])
    np.testing.assert_array_equal(ds.event_stations.values, [3, 6, 2, 9])
    np.testing.assert_allclose(ds.event_power.values, [11.0, 8.4, -2.5, 20.2])
    assert list(ds.station_code.values) == ["A", "B", "C"]
    assert ds.station_name.values[2] == "long name here"
    np.testing.assert_allclose(ds.station_event_fraction.values, [0.826, 0.197, 0.855])
    np.testing.assert_array_equal(ds.station_active.values, [True, True, False])
    assert float(ds.network_center_latitude) == 33.5
    assert ds.attrs["location"] == "TEST"
    assert ds.event_latitude.attrs["units"] == "degrees_north"


def test_read_lma_filters_and_concat(tmp_path):
    a = _write(tmp_path)
    b = _write(tmp_path, name="LYLOUT_220330_231000_0600.dat")
    ds = read_lma([b, a], max_chi2=1.0, min_stations=3, altitude=(0, 12e3))
    # two files, kept rows: chi2 0.1 (3 stations); others fail a filter
    assert ds.sizes["number_of_events"] == 2
    assert np.all(ds.event_chi2.values <= 1.0)


@pytest.mark.parametrize("engine", IO_ENGINES)
def test_read_lma_errors(tmp_path, engine):
    bad = tmp_path / "bad.dat"
    bad.write_text("no header\n")
    with pytest.raises(ValueError, match="data"):
        read_lma(bad, engine=engine)
    with pytest.raises(ValueError, match="no LMA files"):
        read_lma([])
    short = _write(tmp_path, name="short.dat", rows="82800.1 33.5 -88.5 9000.0 0.1\n")
    with pytest.raises(ValueError, match="fewer columns"):
        read_lma(short, engine=engine)
    with pytest.raises(ValueError, match="engine"):
        read_lma(short, engine="bogus")


@pytest.mark.parametrize("engine", IO_ENGINES)
def test_read_lma_other_columns(tmp_path, engine):
    header = HEADER.replace(
        "Data: time (UT sec of day), lat, lon, alt(m), reduced chi^2, P(dBW), mask",
        "Data: time (UT sec of day), lat, lon, alt(m), reduced chi^2, # of stations",
    )
    path = tmp_path / "x.dat"
    path.write_text(header + "82800.5 33.5 -88.5 9000.0 0.10 7\n")
    ds = read_lma(path, min_stations=6, engine=engine)
    assert int(ds.event_stations[0]) == 7
    assert "event_mask" not in ds
    with pytest.raises(ValueError, match="chi-square"):
        read_lma(
            _write(tmp_path, rows="")
            .with_name("y.dat")
            .write_text(
                HEADER.replace(", reduced chi^2", "")
                + "82800.5 33.5 -88.5 9000.0 1.0 0x3\n"
            )
            and tmp_path / "y.dat",
            max_chi2=1,
        )


def test_read_lma_bz2_and_bad_headers(tmp_path):
    import bz2

    path = tmp_path / "a.dat.bz2"
    path.write_bytes(bz2.compress((HEADER + ROWS).encode()))
    assert read_lma(path).sizes["number_of_events"] == 4
    nostart = tmp_path / "b.dat"
    nostart.write_text(HEADER.replace("Data start time", "Start") + ROWS)
    with pytest.raises(ValueError, match="start time"):
        read_lma(nostart)
    nodata = tmp_path / "c.dat"
    nodata.write_text(HEADER.replace("Data: time", "Columns: time") + ROWS)
    with pytest.raises(ValueError, match="column line"):
        read_lma(nodata)
    nomask = tmp_path / "d.dat"
    nomask.write_text(
        HEADER.replace(", P(dBW), mask", ", P(dBW)")
        + "82800.5 33.5 -88.5 9000.0 0.10 3.0\n"
    )
    with pytest.raises(ValueError, match="station mask"):
        read_lma(nomask, min_stations=6)


def test_read_lma_unknown_columns(tmp_path):
    path = tmp_path / "z.dat"
    path.write_text(
        HEADER.replace("time (UT sec of day), lat", "foo, bar") + "1 2 3 4 5 0x1\n"
    )
    with pytest.raises(ValueError, match="unrecognised"):
        read_lma(path)


@pytest.mark.skipif(not lma_io.HAS_COMPILED_KERNEL, reason="compiled kernel not built")
def test_read_lma_engines_agree_large(tmp_path):
    rng = np.random.default_rng(3)
    n = 50_000
    t = np.sort(rng.uniform(82800, 83400, n))
    rows = "".join(
        f"{a:15.9f} {b:12.8f} {c:13.8f} {d:9.2f} {e:6.2f} {f:5.1f} 0x{g:x}\n"
        for a, b, c, d, e, f, g in zip(
            t,
            rng.uniform(33, 34, n),
            rng.uniform(-89, -88, n),
            rng.uniform(0, 2e4, n),
            rng.uniform(0, 5, n),
            rng.uniform(-10, 40, n),
            rng.integers(1, 2**20, n),
        )
    )
    path = _write(tmp_path, rows=rows)
    a = read_lma(path, engine="numpy")
    b = read_lma(path, engine="compiled", n_threads=4)
    xr.testing.assert_identical(a, b)


# --------------------------------------------------------------------------
# clustering
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_cluster_recovers_separated_flashes(engine):
    flashes = [(0, 0, 40), (30e3, 0, 25), (0, 30e3, 5), (-20e3, -20e3, 60)]
    ds, truth = _sources(flashes)
    out = cluster_flashes(ds, engine=engine)
    assert out.sizes["number_of_flashes"] == 4
    np.testing.assert_array_equal(out.event_parent_flash_id.values, truth)
    np.testing.assert_array_equal(out.flash_event_count.values, [40, 25, 5, 60])
    assert np.all(out.flash_time_start <= out.flash_time_end)
    assert np.all(out.flash_duration >= 0) and np.all(out.flash_duration < 0.2)
    # initiation = first source, centre = mean
    first = (
        out.event_time.values
        == out.flash_time_start.values[out.event_parent_flash_id.values]
    )
    assert first.sum() >= 4
    assert np.all(
        (out.flash_center_altitude > 6e3) & (out.flash_center_altitude < 10e3)
    )
    # plan area of a 300 m (1 sigma) cloud: a few km2
    assert np.all(out.flash_area.values[[0, 1, 3]] > 0.3)
    assert np.all(out.flash_area.values < 10)
    assert out.flash_area.attrs["units"] == "km2"


@pytest.mark.parametrize("engine", ENGINES)
def test_cluster_link_criterion(engine):
    # two sources 2.9 km and 0.0 s apart join, 3.1 km do not; in time 0.14 s / 0.16 s
    lat = np.array(
        [
            LAT0,
            LAT0 + 2900 / 111.2e3,
            LAT0 + 100.0,
            LAT0 + 100.0 + 3100 / 111.2e3,
            LAT0 + 50,
            LAT0 + 50,
            LAT0 + 60,
            LAT0 + 60,
        ]
    )
    t = np.array([0.0, 0.0, 10.0, 10.0, 20.0, 20.14, 30.0, 30.16])
    ds = xr.Dataset(
        {"event_altitude": ("number_of_events", np.full(8, 8e3))},
        coords={
            "event_time": (
                "number_of_events",
                DAY + (t * 1e9).astype("timedelta64[ns]"),
            ),
            "event_latitude": ("number_of_events", lat),
            "event_longitude": ("number_of_events", np.full(8, LON0)),
        },
    )
    out = cluster_flashes(ds, engine=engine)
    np.testing.assert_array_equal(
        out.event_parent_flash_id.values, [0, 0, 1, 2, 3, 3, 4, 5]
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_cluster_unsorted_and_chain(engine):
    # a chain of sources 2 km apart forms one flash (single linkage)
    n = 20
    t = np.linspace(0, 1.9, n)
    lat = LAT0 + np.arange(n) * 2000 / 111.2e3
    perm = np.random.default_rng(1).permutation(n)
    ds = xr.Dataset(
        {"event_altitude": ("number_of_events", np.full(n, 8e3)[perm])},
        coords={
            "event_time": (
                "number_of_events",
                (DAY + (t * 1e9).astype("timedelta64[ns]"))[perm],
            ),
            "event_latitude": ("number_of_events", lat[perm]),
            "event_longitude": ("number_of_events", np.full(n, LON0)),
        },
    )
    out = cluster_flashes(ds, engine=engine)
    assert out.sizes["number_of_flashes"] == 1
    assert np.all(np.diff(out.event_time.values) >= np.timedelta64(0, "ns"))
    np.testing.assert_allclose(out.flash_area.values, [0.0], atol=1e-9)  # collinear


@needs_kernel
def test_cluster_engines_agree_random():
    rng = np.random.default_rng(7)
    n = 20_000
    t = np.sort(rng.uniform(0, 20, n))
    ds = xr.Dataset(
        {"event_altitude": ("number_of_events", rng.uniform(2e3, 14e3, n))},
        coords={
            "event_time": (
                "number_of_events",
                DAY + (t * 1e9).astype("timedelta64[ns]"),
            ),
            "event_latitude": ("number_of_events", LAT0 + rng.uniform(-0.3, 0.3, n)),
            "event_longitude": ("number_of_events", LON0 + rng.uniform(-0.3, 0.3, n)),
        },
    )
    a = cluster_flashes(ds, engine="numpy")
    b = cluster_flashes(ds, engine="compiled", n_threads=4)
    c = cluster_flashes(ds, engine="compiled", n_threads=1)
    np.testing.assert_array_equal(a.event_parent_flash_id, b.event_parent_flash_id)
    np.testing.assert_array_equal(b.event_parent_flash_id, c.event_parent_flash_id)
    np.testing.assert_allclose(a.flash_area, b.flash_area, rtol=1e-9, atol=1e-12)
    xr.testing.assert_allclose(a, b)


def test_cluster_errors():
    ds, _ = _sources([(0, 0, 5)])
    with pytest.raises(ValueError, match="positive"):
        cluster_flashes(ds, distance=0)
    with pytest.raises(ValueError, match="event_altitude"):
        cluster_flashes(ds.drop_vars("event_altitude"))
    with pytest.raises(ValueError, match="engine"):
        cluster_flashes(ds, engine="fast")
    empty = ds.isel(number_of_events=slice(0, 0))
    assert cluster_flashes(empty, engine="numpy").sizes["number_of_flashes"] == 0


def test_engine_compiled_unavailable(monkeypatch):
    monkeypatch.setattr(L, "HAS_COMPILED_KERNEL", False)
    ds, _ = _sources([(0, 0, 5)])
    with pytest.raises(ImportError):
        cluster_flashes(ds, engine="compiled")
    assert cluster_flashes(ds).sizes["number_of_flashes"] == 1
    monkeypatch.setattr(lma_io, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError):
        read_lma("x.dat", engine="compiled")


# --------------------------------------------------------------------------
# gridding
# --------------------------------------------------------------------------


def _grid(n=41, dx=1000.0):
    x = (np.arange(n) - n // 2) * dx
    return xr.Dataset(
        coords={
            "x": x,
            "y": x,
            "z": np.arange(1000.0, 15001.0, 1000.0),
            "latitude": LAT0,
            "longitude": LON0,
        }
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_grid_lightning_counts(engine):
    flashes = [(0, 0, 40), (10e3, 0, 25), (0, 10e3, 5), (-10e3, -10e3, 60)]
    ds, _ = _sources(flashes)
    out = grid_lightning(ds, _grid(), interval="1min", engine=engine)
    assert out.flash_extent_density.dims == ("time", "y", "x")
    assert out.sizes["time"] == 1
    # all sources inside the grid
    assert int(out.source_density.sum()) == 130
    # 3 flashes with >= 10 sources, each initiated once
    assert int(out.flash_initiation_density.sum()) == 3
    fed = out.flash_extent_density.isel(time=0)
    # every flash covers at least one box and never counts twice in one box
    assert int(fed.max()) == 1
    assert int(fed.sel(x=0, y=0, method="nearest")) == 1
    assert int(fed.sel(x=10e3, y=0, method="nearest")) == 0  # 5-source flash dropped
    # min_sources=1 keeps it
    out1 = grid_lightning(ds, _grid(), interval="1min", min_sources=1, engine=engine)
    assert int(out1.flash_initiation_density.sum()) == 4
    assert out.lat.size == 41 and float(out.latitude) == LAT0
    assert out.time_bounds.shape == (1, 2)


@pytest.mark.parametrize("engine", ENGINES)
def test_grid_lightning_3d_and_time(engine):
    flashes = [(0, 0, 40), (10e3, 0, 25)] * 3
    ds, _ = _sources(flashes, spacing_s=30.0)
    out = grid_lightning(ds, _grid(), z=True, interval="30s", engine=engine)
    assert out.flash_extent_density.dims == ("time", "z", "y", "x")
    assert out.sizes["time"] == 6
    np.testing.assert_array_equal(
        out.flash_initiation_density.sum(("z", "y", "x")), [1] * 6
    )
    # sources between 6 and 10 km only
    by_z = out.source_density.sum(("time", "y", "x"))
    assert int(by_z.sel(z=slice(None, 5000)).sum()) == 0
    assert int(by_z.sum()) == 195
    # column FED <= sum of 3-D FED
    col = grid_lightning(ds, _grid(), interval="30s", engine=engine)
    assert np.all(col.flash_extent_density <= out.flash_extent_density.sum("z"))


@needs_kernel
def test_grid_engines_agree():
    rng = np.random.default_rng(5)
    flashes = [
        (rng.uniform(-30e3, 30e3), rng.uniform(-30e3, 30e3), int(rng.integers(1, 80)))
        for _ in range(300)
    ]
    ds, _ = _sources(flashes, spacing_s=0.4)
    kw = dict(z=True, interval="20s", min_sources=3)
    a = grid_lightning(ds, _grid(81, 800.0), engine="numpy", **kw)
    b = grid_lightning(ds, _grid(81, 800.0), engine="compiled", n_threads=3, **kw)
    xr.testing.assert_identical(a, b)
    va = vertical_source_distribution(
        ds, np.arange(500.0, 15e3, 1000.0), engine="numpy"
    )
    vb = vertical_source_distribution(
        ds, np.arange(500.0, 15e3, 1000.0), engine="compiled"
    )
    xr.testing.assert_identical(va, vb)


def test_grid_lightning_with_xlma_flash_ids():
    ds, truth = _sources([(0, 0, 20), (5e3, 0, 30)])
    ids = np.array([131072, 7])[truth]
    xlma = ds.assign(event_parent_flash_id=("number_of_events", ids)).assign_coords(
        flash_id=("number_of_flashes", np.array([7, 131072], dtype=np.uint64))
    )
    out = grid_lightning(xlma, _grid(), interval="1min")
    assert int(out.flash_initiation_density.sum()) == 2
    no_ids = ds.assign(event_parent_flash_id=("number_of_events", ids))
    assert (
        int(
            grid_lightning(
                no_ids, _grid(), interval="1min"
            ).flash_initiation_density.sum()
        )
        == 2
    )


def test_grid_lightning_origin_and_errors():
    ds, _ = _sources([(0, 0, 20)])
    g = _grid().drop_vars(["latitude", "longitude"])
    with pytest.raises(ValueError, match="origin"):
        grid_lightning(ds, g)
    out = grid_lightning(ds, g, latitude=LAT0, longitude=LON0)
    assert int(out.flash_extent_density.sum()) >= 1
    pyart = g.assign(
        origin_latitude=("time", [LAT0]), origin_longitude=("time", [LON0])
    )
    assert int(grid_lightning(ds, pyart).source_density.sum()) == 20
    import pyproj

    crs = pyproj.CRS.from_dict(
        {"proj": "aeqd", "lat_0": LAT0, "lon_0": LON0, "datum": "WGS84"}
    )
    cf = g.assign_coords(crs_wkt=xr.DataArray(0, attrs=crs.to_cf()))
    out = grid_lightning(ds, cf)
    assert "crs_wkt" in out.coords and int(out.source_density.sum()) == 20
    attrs = g.assign_attrs(latitude=LAT0, longitude=LON0)
    assert int(grid_lightning(ds, attrs).source_density.sum()) == 20
    with pytest.raises(ValueError, match="x and y"):
        grid_lightning(ds, latitude=LAT0, longitude=LON0)
    with pytest.raises(ValueError, match="z=True"):
        grid_lightning(
            ds, x=[0.0, 1.0], y=[0.0, 1.0], z=True, latitude=LAT0, longitude=LON0
        )
    with pytest.raises(ValueError, match="increase"):
        grid_lightning(ds, x=[1.0, 0.0], y=[0.0, 1.0], latitude=LAT0, longitude=LON0)
    with pytest.raises(ValueError, match="interval"):
        grid_lightning(ds, _grid(), interval="soon")
    with pytest.raises(ValueError, match="positive"):
        grid_lightning(ds, _grid(), interval="-1min")
    with pytest.raises(ValueError, match="time_edges"):
        grid_lightning(ds, _grid(), time_edges=[DAY])
    one = grid_lightning(
        ds, x=[0.0], y=[0.0], z=[8000.0], latitude=LAT0, longitude=LON0
    )
    assert one.sizes["x"] == 1
    import datetime

    td = grid_lightning(ds, _grid(), interval=datetime.timedelta(minutes=2))
    assert td.sizes["time"] == 1
    with pytest.raises(ValueError, match="1-D"):
        grid_lightning(ds, x=[[0.0]], y=[0.0], latitude=LAT0, longitude=LON0)
    with pytest.raises(ValueError, match="no sources"):
        grid_lightning(ds.isel(number_of_events=slice(0, 0)), _grid())


def test_vertical_source_distribution():
    ds, _ = _sources([(0, 0, 40), (10e3, 0, 25)])
    z = np.arange(500.0, 15e3, 1000.0)
    out = vertical_source_distribution(ds, z)
    assert out.source_count.dims == ("time", "z") and out.sizes["time"] == 1
    assert int(out.source_count.sum()) == 65
    assert int(out.flash_initiation_count.sum()) == 2
    assert int(out.source_count.sel(z=slice(None, 5000)).sum()) == 0
    timed = vertical_source_distribution(ds, z, interval="1s")
    assert int(timed.source_count.sum()) == 65
    with pytest.raises(ValueError, match="no sources"):
        vertical_source_distribution(ds.isel(number_of_events=slice(0, 0)), z)


# --------------------------------------------------------------------------
# cells
# --------------------------------------------------------------------------


def _mask(times):
    x = np.arange(-20e3, 20.01e3, 1000.0)
    m = np.zeros((len(times), x.size, x.size), dtype=np.int32)
    m[:, 15:26, 15:26] = 7  # cell 7 around the origin
    m[:, 15:26, 26:36] = 3  # cell 3 east of it (x 6..15 km)
    return xr.DataArray(
        m,
        dims=("time", "y", "x"),
        coords={"time": times, "y": x, "x": x, "latitude": LAT0, "longitude": LON0},
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_cell_flash_rate(engine):
    flashes = [(0, 0, 40), (0, 10e3, 25), (0, 0, 30), (19e3, 19e3, 30)]
    ds, _ = _sources(flashes, spacing_s=20.0)
    times = DAY + np.array([0, 60], dtype="timedelta64[s]")
    out = cell_flash_rate(
        ds, _mask(times), z=np.arange(500.0, 15e3, 1000.0), engine=engine
    )
    assert list(out.cell.values) == [3, 7]
    assert out.flash_count.dims == ("cell", "time")
    total = out.flash_count.sum("time")
    np.testing.assert_array_equal(total, [1, 2])
    assert out.flash_rate.attrs["units"] == "min-1"
    np.testing.assert_allclose(out.flash_rate.sum("time"), total)  # 1-min intervals
    assert int(out.source_count.sel(cell=7).sum()) == 70
    ext = cell_flash_rate(ds, _mask(times), count="extent", engine=engine)
    assert np.all(ext.flash_count.sum("time") >= total)
    assert "source_count" not in ext


@needs_kernel
def test_cell_engines_agree():
    rng = np.random.default_rng(11)
    flashes = [
        (rng.uniform(-15e3, 15e3), rng.uniform(-15e3, 15e3), int(rng.integers(5, 50)))
        for _ in range(200)
    ]
    ds, _ = _sources(flashes, spacing_s=0.9)
    times = DAY + np.arange(0, 200, 30).astype("timedelta64[s]")
    mask = _mask(times).astype(float)
    mask[2] = np.nan
    for count in ("initiation", "extent"):
        kw = dict(z=np.arange(500.0, 15e3, 1000.0), count=count, interval="30s")
        a = cell_flash_rate(ds, mask, engine="numpy", **kw)
        b = cell_flash_rate(ds, mask, engine="compiled", **kw)
        xr.testing.assert_identical(a, b)


def test_cell_flash_rate_errors_and_offsets():
    ds, _ = _sources([(0, 0, 40)], spacing_s=20.0)
    times = DAY + np.array([0, 60], dtype="timedelta64[s]")
    m = _mask(times)
    with pytest.raises(ValueError, match="count"):
        cell_flash_rate(ds, m, count="all")
    with pytest.raises(ValueError, match="DataArray"):
        cell_flash_rate(ds, m.isel(time=0))
    with pytest.raises(ValueError, match="increase"):
        cell_flash_rate(ds, m.isel(time=[1, 0]))
    # sources far from any mask time are not attributed
    far = cell_flash_rate(ds, m.assign_coords(time=times + np.timedelta64(1, "h")))
    assert int(far.flash_count.sum()) == 0
    near = cell_flash_rate(ds, m, max_offset="10s", time_edges=times)
    assert int(near.flash_count.sum()) == 1
    single = cell_flash_rate(ds, m.isel(time=[0]))
    assert int(single.flash_count.sum()) == 1
    for count in ("initiation", "extent"):
        empty = cell_flash_rate(
            ds.isel(number_of_events=slice(0, 0)), m, count=count, engine="numpy"
        )
        assert int(empty.flash_count.sum()) == 0


# --------------------------------------------------------------------------
# lightning jump
# --------------------------------------------------------------------------


def _rate(values, step="1min"):
    t = DAY + np.arange(len(values)) * np.timedelta64(pd_minutes(step), "m")
    return xr.DataArray(np.asarray(values, float), dims="time", coords={"time": t})


def pd_minutes(step):
    return int(step.rstrip("min"))


def test_lightning_jump_detects_and_ends():
    # 2-min means: flat 12 with small noise, then jump, then fall
    base = [12, 12, 13, 12, 13, 12, 12, 13, 12, 12, 13, 12, 12, 12]
    rise = [20, 20, 30, 30, 34, 34, 20, 20, 12, 12]
    out = lightning_jump(_rate(base + rise))
    assert out.sizes["time"] == 12
    np.testing.assert_allclose(out.flash_rate.values[:3], [12, 12.5, 12.5])
    np.testing.assert_allclose(out.dfrdt.values[1], 0.25)
    assert np.isnan(out.sigma_level.values[:6]).all()
    starts = np.nonzero(out.jump_start.values)[0]
    assert list(starts) == [7]  # first period at 20 flashes/min
    # the jump continues while the sigma level stays >= 0, ends when it drops
    assert out.jump.values[7] and out.jump.values[8] and out.jump.values[9]
    assert not out.jump.values[10]
    assert out.dfrdt.attrs["units"] == "min-2"


def test_lightning_jump_rate_threshold_grouping_and_cells():
    low = [2, 2, 3, 2, 3, 2, 2, 3, 2, 2, 3, 2, 2, 2, 6, 6, 9, 9]
    out = lightning_jump(_rate(low))
    assert not out.jump.any()  # below 10 flashes/min
    out = lightning_jump(_rate(low), min_rate=5.0)
    assert out.jump_start.sum() == 1
    # two cells on a 2-D array
    hi = [12, 12, 13, 12, 13, 12, 12, 13, 12, 12, 13, 12, 12, 12, 30, 30, 31, 31]
    da = xr.concat([_rate(low), _rate(hi)], dim=xr.DataArray([1, 2], dims="cell"))
    out = da.radarx.lightning_jump()
    assert out.jump_start.dims == ("cell", "time")
    np.testing.assert_array_equal(out.jump_start.sum("time"), [0, 1])
    # a renewed jump inside the 6-min grouping window is not a new start
    seq = [12] * 14 + [20, 20, 12, 12, 40, 40, 40, 40]
    out = lightning_jump(_rate(seq), sigma=1.0, group="10min")
    assert out.jump_start.sum() == 1
    out = lightning_jump(_rate(seq), sigma=1.0, group="2min")
    assert out.jump_start.sum() == 2


def test_lightning_jump_flat_history_and_errors():
    flat = [12] * 14 + [20, 20, 20, 20]
    out = lightning_jump(_rate(flat))
    assert np.isinf(out.sigma_level.values[7])  # zero sigma, positive DFRDT
    assert out.jump_start.values[7]
    with pytest.raises(ValueError, match="time"):
        lightning_jump(xr.DataArray([1.0, 2.0], dims="x"))
    with pytest.raises(ValueError, match="history"):
        lightning_jump(_rate(flat), history=1)
    with pytest.raises(ValueError, match="two times"):
        lightning_jump(_rate([1.0]))


def test_lightning_jump_from_cell_rates():
    ds, _ = _sources([(0, 0, 40)] * 3, spacing_s=20.0)
    times = DAY + np.array([0, 60], dtype="timedelta64[s]")
    rates = cell_flash_rate(ds, _mask(times))
    out = lightning_jump(rates.flash_rate)
    assert out.flash_rate.dims == ("cell", "time")


# --------------------------------------------------------------------------
# accessors and real data
# --------------------------------------------------------------------------


def test_accessors():
    ds, _ = _sources([(0, 0, 40), (10e3, 0, 25)])
    fl = ds.radarx.cluster_flashes()
    assert fl.sizes["number_of_flashes"] == 2
    out = fl.radarx.grid_lightning(_grid(), interval="1min")
    assert int(out.flash_initiation_density.sum()) == 2


def test_real_xlma_sample():
    """The 21 084 sources of the xlma-python West Texas example files."""
    pooch = pytest.importorskip("pooch")
    base = (
        "https://raw.githubusercontent.com/deeplycloudy/xlma-python/"
        "97f8aaa88d8730dad62686d076e8bd74c4007be6/examples/data/"
    )
    try:
        path = pooch.retrieve(base + "WTLMA_231224_005701_0001.dat.gz", known_hash=None)
    except Exception as err:  # pragma: no cover - network
        pytest.skip(f"sample file not available: {err}")
    ds = read_lma(path, max_chi2=5.0)
    assert ds.sizes["number_of_events"] > 100
    assert ds.attrs["location"] == "WestTexas"
    fl = cluster_flashes(ds)
    assert 1 <= fl.sizes["number_of_flashes"] < ds.sizes["number_of_events"]
    out = grid_lightning(
        fl,
        x=np.arange(-200e3, 200e3, 2e3),
        y=np.arange(-200e3, 200e3, 2e3),
        latitude=float(ds.network_center_latitude),
        longitude=float(ds.network_center_longitude),
        min_sources=1,
    )
    assert int(out.flash_initiation_density.sum()) <= fl.sizes["number_of_flashes"]


def test_projection_threads_match():
    rng = np.random.default_rng(2)
    lon = LON0 + rng.uniform(-3, 3, 250_000)
    lat = LAT0 + rng.uniform(-3, 3, 250_000)
    x1, y1 = L._project(lon, lat, LAT0, LON0, n_threads=1)
    x4, y4 = L._project(lon, lat, LAT0, LON0, n_threads=4)
    np.testing.assert_array_equal(x1, x4)
    np.testing.assert_array_equal(y1, y4)
    # due east and north of the origin along the axes
    x, y = L._project([LON0, LON0 + 1.0], [LAT0 + 1.0, LAT0], LAT0, LON0)
    assert abs(x[0]) < 1e-6 and y[0] > 110e3 and abs(y[1]) < 2e3 and x[1] > 90e3
