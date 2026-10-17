#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the beam weighting of gate gridding and the blockage weight of the merge
===================================================================================
"""
import numpy as np
import pyproj
import pytest
import xarray as xr
from xradar.georeference import antenna_to_cartesian

import radarx  # noqa: F401
from radarx.grid import gates, multi

ENGINES = ["numpy"] + (["compiled"] if gates.HAS_COMPILED_KERNEL else [])
needs_kernel = pytest.mark.skipif(
    not gates.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)
T0 = np.datetime64("2026-01-01T00:00", "ns")
LN2 = np.log(2.0)


def _tree(data, azimuth, ranges, elevation=0.0, altitude=0.0, times=None):
    """One sweep with ``data`` on (azimuth, range) at a fixed elevation."""
    azimuth = np.asarray(azimuth, dtype=float)
    ds = xr.Dataset(
        {"DBZH": (("azimuth", "range"), np.asarray(data, dtype=float))},
        coords={
            "azimuth": azimuth,
            "range": np.asarray(ranges, dtype=float),
            "elevation": ("azimuth", np.full(azimuth.size, float(elevation))),
            "time": ("azimuth", np.full(azimuth.size, T0)),
        },
    )
    root = xr.Dataset(
        coords={"latitude": 45.0, "longitude": 10.0, "altitude": float(altitude)}
    )
    return xr.DataTree.from_dict({"/": root, "sweep_0": ds})


def _point(az, rng, el=0.0, altitude=0.0):
    x, y, z = antenna_to_cartesian(
        np.array([rng]), np.array([az]), np.array([el]), site_altitude=altitude
    )
    return [float(x[0])], [float(y[0])], [float(z[0])]


def _beam(tree, point, engine, **kw):
    x, y, z = point
    options = dict(weighting="beam", beamwidth=1.0, roi=2000.0, engine=engine)
    options.update(kw)
    return gates.grid_gates(tree, "DBZH", x, y, z, **options)["DBZH"].values.item()


# ---------------------------------------------------------------------------
# the weight against the formula of the book
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("engine", ENGINES)
def test_weight_across_the_beam_is_the_two_way_gaussian(engine):
    """Two beams, 0.5 degrees apart: the point on the first one gives 0.25 to the other.

    Doviak and Zrnic (1993), Eq. 5.75: the two-way power pattern of a Gaussian
    beam of half-power width theta_1 has the variance theta_1^2 / (16 ln 2), so
    its value is 0.5^2 at half the beamwidth off the axis.
    """
    tree = _tree([[10.0, np.nan], [50.0, np.nan]], [0.0, 0.5], [30e3, 30.25e3])
    value = _beam(tree, _point(0.0, 30e3), engine)
    assert value == pytest.approx((10.0 + 0.25 * 50.0) / 1.25, rel=2e-3)
    # the same point from the other beam: the weights swap
    value = _beam(tree, _point(0.5, 30e3), engine)
    assert value == pytest.approx((50.0 + 0.25 * 10.0) / 1.25, rel=2e-3)


@pytest.mark.parametrize("engine", ENGINES)
def test_weight_between_beams_is_a_logistic_of_the_offsets(engine):
    tree = _tree([[0.0, np.nan], [10.0, np.nan]], [0.0, 1.0], [30e3, 30.25e3])
    sigma = np.radians(1.0) / (4.0 * np.sqrt(LN2)) * 30e3
    s = 30e3  # ground distance of the gates, to 0.1 percent
    for az in (0.1, 0.25, 0.5, 0.8):
        d_a, d_b = s * np.radians(az), s * np.radians(1.0 - az)
        w_b = np.exp(-0.5 * (d_b**2 - d_a**2) / sigma**2)
        expected = 10.0 * w_b / (1.0 + w_b)
        assert _beam(tree, _point(az, 30e3), engine) == pytest.approx(
            expected, abs=2e-3 * 10.0
        )


@pytest.mark.parametrize("engine", ENGINES)
def test_weight_along_the_beam_follows_the_range_resolution(engine):
    spacing = 250.0
    tree = _tree([[0.0, 10.0]], [0.0], [30e3, 30e3 + spacing])
    for resolution, kw in ((None, {}), (500.0, {"range_resolution": 500.0})):
        sigma = (spacing if resolution is None else resolution) / (
            2.0 * np.sqrt(2.0 * LN2)
        )
        for offset in (40.0, 125.0, 200.0):
            w_b = np.exp(-0.5 * ((offset - spacing) ** 2 - offset**2) / sigma**2)
            expected = 10.0 * w_b / (1.0 + w_b)
            point = _point(0.0, 30e3 + offset)
            assert _beam(tree, point, engine, **kw) == pytest.approx(expected, rel=2e-3)


@pytest.mark.parametrize("engine", ENGINES)
def test_weights_are_relative_to_the_best_gate(engine):
    """Between two beams 90 degrees apart no weight is above 1e-300, still a mean."""
    tree = _tree([[10.0, np.nan], [50.0, np.nan]], [0.0, 90.0], [30e3, 30.25e3])
    x, y, z = _point(45.0, 30e3)
    value = _beam(tree, (x, y, z), engine, roi=60e3)
    assert value == pytest.approx(30.0, rel=1e-6)


@pytest.mark.parametrize("engine", ENGINES)
def test_radius_still_limits_the_gates(engine):
    tree = _tree([[10.0, np.nan], [50.0, np.nan]], [0.0, 90.0], [30e3, 30.25e3])
    far = _beam(tree, _point(45.0, 30e3), engine, roi=5e3)
    assert np.isnan(far)


# ---------------------------------------------------------------------------
# a velocity field
# ---------------------------------------------------------------------------
def _edge_field(gx, gy, gz):
    """+20 east of the north-south line through the radar, -20 west of it."""
    return np.where(gx > 0, 20.0, -20.0)


def _ppi_volume(f, elevations=(0.5, 1.5)):
    rng_m = np.arange(125.0, 60e3, 250.0)
    sweeps = {}
    for k, el in enumerate(elevations):
        az = np.roll(np.arange(0.5, 360.0, 1.0), 37 * (k + 1))
        gx, gy, gz = antenna_to_cartesian(
            rng_m[None, :], az[:, None], np.full((az.size, 1), el), site_altitude=0.0
        )
        sweeps[f"sweep_{k}"] = xr.Dataset(
            {"DBZH": (("azimuth", "range"), f(gx, gy, gz))},
            coords={
                "azimuth": az,
                "range": rng_m,
                "elevation": ("azimuth", np.full(az.size, el)),
                "time": ("azimuth", np.full(az.size, T0)),
            },
        )
    root = xr.Dataset(coords={"latitude": 45.0, "longitude": 10.0, "altitude": 0.0})
    return xr.DataTree.from_dict({"/": root, **sweeps})


@pytest.mark.parametrize("engine", ENGINES)
def test_beam_weighting_keeps_a_sharp_azimuthal_edge(engine):
    """Across an edge of the field the beams that see each side keep their value.

    The radius of 2 km at 30 to 40 km range reaches several beams to either
    side, so Cressman weights blur the edge; the beam weights follow the beams
    that pass the point. On a smooth field the beam weights are the noisier ones,
    as they follow single beams (1 degree apart, about as wide as the beam).
    """
    tree = _ppi_volume(_edge_field)
    grid = dict(
        x=np.array([-6e3, -4e3, -3e3, -2e3, -1e3, 1e3, 2e3, 3e3, 4e3, 6e3]),
        y=np.array([30e3, 40e3]),
        z=np.array([700.0, 1100.0]),
    )
    X, Y = np.meshgrid(grid["x"], grid["y"])
    truth = _edge_field(X, Y, 0.0)
    error = {}
    for weighting in ("cressman", "beam"):
        out = gates.grid_gates(
            tree,
            "DBZH",
            **grid,
            weighting=weighting,
            roi=2000.0,
            beamwidth=1.0,
            engine=engine,
        )["DBZH"]
        mean = out.mean("z").values
        assert np.isfinite(mean).all()
        error[weighting] = np.sqrt(np.mean((mean - truth) ** 2))
    assert error["beam"] < 0.1 * error["cressman"]
    assert error["beam"] < 1.0


# ---------------------------------------------------------------------------
# engines and options
# ---------------------------------------------------------------------------
def _random_volume(seed=0, nan_fraction=0.1):
    rnd = np.random.default_rng(seed)
    tree = _ppi_volume(lambda gx, gy, gz: rnd.uniform(-10, 60, gx.shape))
    for name in ("sweep_0", "sweep_1"):
        ds = tree[name].to_dataset()
        data = ds["DBZH"].values
        data[rnd.random(data.shape) < nan_fraction] = np.nan
        ds["DBZH"] = (ds["DBZH"].dims, data)
        tree[name] = xr.DataTree(ds)
    return tree


@needs_kernel
@pytest.mark.parametrize(
    "options",
    [
        dict(roi=1500.0, beamwidth=1.0),
        dict(roi=800.0, beamwidth=0.5, range_resolution=400.0),
        dict(roi=1000.0, beamwidth=1.5, roi_range_factor=0.01, min_gates=3),
        dict(roi=1500.0, beamwidth=1.0, motion=(10.0, -5.0), time="2026-01-01T00:03"),
    ],
)
def test_engines_agree(options):
    tree = _random_volume()
    grid = dict(
        x=np.arange(-30e3, 30e3 + 1, 2e3) + 17.3,
        y=np.arange(-30e3, 30e3 + 1, 2e3) + 34.6,
        z=np.arange(500.0, 3000.0 + 1, 500.0) + 51.9,
    )
    a = gates.grid_gates(
        tree, "DBZH", **grid, weighting="beam", engine="compiled", **options
    )
    b = gates.grid_gates(
        tree, "DBZH", **grid, weighting="beam", engine="numpy", **options
    )
    A, B = a.DBZH.values, b.DBZH.values
    np.testing.assert_array_equal(np.isnan(A), np.isnan(B))
    np.testing.assert_allclose(A, B, atol=1e-3, equal_nan=True)
    assert np.isfinite(A).mean() > 0.3


@needs_kernel
def test_threads_do_not_change_the_result():
    tree = _random_volume(1)
    grid = dict(
        x=np.arange(-20e3, 20e3 + 1, 2e3),
        y=np.arange(-20e3, 20e3 + 1, 2e3),
        z=np.arange(500.0, 3000.0 + 1, 500.0),
    )
    one = gates.grid_gates(
        tree, "DBZH", **grid, weighting="beam", beamwidth=1.0, n_threads=1
    )
    many = gates.grid_gates(
        tree, "DBZH", **grid, weighting="beam", beamwidth=1.0, n_threads=4
    )
    np.testing.assert_array_equal(one.DBZH.values, many.DBZH.values)


@pytest.mark.parametrize("engine", ENGINES)
def test_attributes_and_the_other_weightings_do_not_change(engine):
    tree = _random_volume(2)
    grid = dict(x=np.arange(-10e3, 10e3 + 1, 2e3), y=[0.0, 5e3], z=[500.0, 1500.0])
    out = gates.grid_gates(
        tree,
        "DBZH",
        **grid,
        weighting="beam",
        beamwidth=1.0,
        range_resolution=300.0,
        engine=engine,
    )
    assert out.attrs["weighting"] == "beam"
    assert out.attrs["beamwidth"] == 1.0
    assert out.attrs["range_resolution"] == 300.0
    default = gates.grid_gates(tree, "DBZH", **grid, engine=engine)
    assert default.attrs["weighting"] == "cressman"
    assert "beamwidth" not in default.attrs


@pytest.mark.parametrize("engine", ENGINES)
def test_empty_volume_and_missing_gates(engine):
    tree = _tree([[np.nan, np.nan]], [0.0], [1e3, 1.25e3])
    out = gates.grid_gates(
        tree,
        "DBZH",
        [1e3],
        [1e3],
        [10.0],
        weighting="beam",
        beamwidth=1.0,
        engine=engine,
    )
    assert np.isnan(out.DBZH.values).all()


@pytest.mark.parametrize("engine", ENGINES)
def test_rhi_section_with_beam_weights(engine):
    elevation = np.arange(0.25, 30.0, 0.5)
    rng_m = np.arange(125.0, 40e3, 250.0)
    gx, gy, gz = antenna_to_cartesian(
        rng_m[None, :],
        np.full((elevation.size, 1), 90.0),
        elevation[:, None],
        site_altitude=0.0,
    )
    ds = xr.Dataset(
        {"DBZH": (("time", "range"), 20.0 + 1e-3 * gz)},
        coords={
            "azimuth": ("time", np.full(elevation.size, 90.0)),
            "elevation": ("time", elevation),
            "range": rng_m,
            "time": T0 + np.arange(elevation.size) * np.timedelta64(1, "s"),
        },
    )
    root = xr.Dataset(coords={"latitude": 45.0, "longitude": 10.0, "altitude": 0.0})
    tree = xr.DataTree.from_dict({"/": root, "sweep_0": ds})
    distance = np.arange(5e3, 30e3, 2e3)
    heights = np.arange(500.0, 6000.0, 500.0)
    out = gates.grid_rhi(
        tree,
        "DBZH",
        distance=distance,
        z=heights,
        weighting="beam",
        beamwidth=1.0,
        roi=1000.0,
        engine=engine,
    )
    assert out.DBZH.dims == ("z", "distance")
    ok = np.isfinite(out.DBZH.values)
    assert ok.mean() > 0.5
    truth = 20.0 + 1e-3 * heights[:, None] * np.ones_like(distance)[None, :]
    assert (
        np.abs(out.DBZH.values[ok] - truth[ok]).max() < 1.0
    )  # gradient 1e-3 per m, radius 1 km


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
def test_errors():
    tree = _tree([[1.0, 2.0]], [0.0], [1e3, 1.25e3])
    point = ([1e3], [1e3], [10.0])
    with pytest.raises(ValueError, match="needs beamwidth"):
        gates.grid_gates(tree, "DBZH", *point, weighting="beam")
    with pytest.raises(ValueError, match="applies to weighting='beam'"):
        gates.grid_gates(tree, "DBZH", *point, range_resolution=250.0)
    for bad in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="range_resolution must be a positive"):
            gates.grid_gates(
                tree,
                "DBZH",
                *point,
                weighting="beam",
                beamwidth=1.0,
                range_resolution=bad,
            )
    one_gate = _tree([[1.0]], [0.0], [1e3])
    with pytest.raises(ValueError, match="needs range_resolution"):
        gates.grid_gates(one_gate, "DBZH", *point, weighting="beam", beamwidth=1.0)
    out = gates.grid_gates(
        one_gate,
        "DBZH",
        *point,
        weighting="beam",
        beamwidth=1.0,
        range_resolution=100.0,
    )
    assert out.DBZH.shape == (1, 1, 1)
    with pytest.raises(ValueError, match="weighting must be one of"):
        gates.grid_gates(tree, "DBZH", *point, weighting="triangle")


@needs_kernel
def test_kernel_argument_checks():
    from radarx.grid import _cone

    data = [np.ones((2, 2))]
    args = (
        np.array([1e3]),
        np.array([1e3]),
        np.array([10.0]),
        data,
        [np.array([0.0, 1.0])],
        [np.array([0.0, 0.0])],
        [np.array([1e3, 1.25e3])],
        0.0,
    )
    kw = dict(weighting=3, roi=1000.0)
    with pytest.raises(ValueError, match="positive beam_sigma"):
        _cone.grid_gates(*args, **kw, beam_sigma=0.0, along_sigma=[100.0])
    with pytest.raises(ValueError, match="along_sigma per sweep"):
        _cone.grid_gates(*args, **kw, beam_sigma=0.005, along_sigma=[])
    with pytest.raises(ValueError, match="along_sigma must be positive"):
        _cone.grid_gates(*args, **kw, beam_sigma=0.005, along_sigma=[0.0])
    with pytest.raises(ValueError, match="unknown weighting"):
        _cone.grid_gates(*args, weighting=4, roi=1000.0)
    ok = _cone.grid_gates(*args, **kw, beam_sigma=0.005, along_sigma=[100.0])
    assert ok.shape == (1, 1)


# ---------------------------------------------------------------------------
# the real volume
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def nexrad():
    from open_radar_data import DATASETS
    from xradar.io import open_nexradlevel2_datatree

    dtree = open_nexradlevel2_datatree(DATASETS.fetch("KLBB20160601_150025_V06"))
    for name in [n for n in dtree.children if n.startswith("sweep")]:
        ds = dtree[name].to_dataset()
        if "VRADH" in ds:
            ds["VRADH"] = ds["VRADH"].where(ds["VRADH"] > -63.9)
        dtree[name] = ds
    return dtree


def test_real_velocity_is_closer_to_the_cones_with_beam_weights(nexrad):
    """KLBB: the beam weights follow the gates of the beams, like the cone grid."""
    from radarx.grid import grid_cones

    x = y = np.arange(-100e3, 100e3 + 1, 1e3)
    z = np.arange(1000.0, 5000.0 + 1, 1000.0)
    ref = grid_cones(nexrad, "VRADH", x, y, z).VRADH.values
    rms = {}
    for weighting in ("cressman", "beam"):
        out = gates.grid_gates(
            nexrad, "VRADH", x, y, z, weighting=weighting, roi=1000.0, beamwidth=1.0
        ).VRADH.values
        both = np.isfinite(out) & np.isfinite(ref)
        assert both.sum() > 10_000
        rms[weighting] = np.sqrt(np.mean((out[both] - ref[both]) ** 2))
    assert rms["beam"] < rms["cressman"]


# ---------------------------------------------------------------------------
# blockage weight of the merge
# ---------------------------------------------------------------------------
ORIGIN = (35.0, -90.0)
MERGE_GRID = dict(
    x=np.arange(-40e3, 60e3 + 1, 4e3),
    y=np.arange(-30e3, 30e3 + 1, 4e3),
    z=np.arange(1000.0, 4000.0 + 1, 1000.0),
)


def _aeqd(lat, lon):
    return pyproj.Proj(proj="aeqd", lat_0=lat, lon_0=lon, datum="WGS84")


def _site(x, y):
    lon, lat = _aeqd(*ORIGIN)(x, y, inverse=True)
    return float(lat), float(lon)


def _radar(lat, lon, bias):
    """A volume that measures 30 dBZ plus ``bias`` everywhere."""
    rng_m = np.arange(250.0, 90e3, 500.0)
    az = np.arange(0.5, 360.0, 1.0)
    sweeps = {}
    for k, el in enumerate((0.5, 1.5, 3.0, 5.0, 8.0)):
        sweeps[f"sweep_{k}"] = xr.Dataset(
            {
                "DBZH": (
                    ("azimuth", "range"),
                    np.full((az.size, rng_m.size), 30.0 + bias),
                )
            },
            coords={
                "azimuth": az,
                "range": rng_m,
                "elevation": ("azimuth", np.full(az.size, el)),
                "time": ("azimuth", np.full(az.size, T0)),
            },
        )
    root = xr.Dataset(coords={"latitude": lat, "longitude": lon, "altitude": 100.0})
    return xr.DataTree.from_dict({"/": root, **sweeps})


@pytest.fixture(scope="module")
def pair():
    a = _radar(*ORIGIN, bias=0.0)
    b = _radar(*_site(20e3, 5e3), bias=10.0)
    return multi.grid_radars([a, b], **MERGE_GRID, origin=ORIGIN, data_vars="DBZH")


def _with_blockage(grid, first, second):
    b = np.zeros(grid["DBZH"].shape, dtype=np.float32)
    b[0], b[1] = first, second
    return grid.assign(CBB=(grid["DBZH"].dims, b))


@pytest.mark.parametrize("engine", ENGINES)
def test_blockage_weight_is_one_minus_the_blockage(pair, engine):
    plain = multi.merge_radars(pair, "DBZH", no_echo=None, engine=engine)
    both = np.isfinite(pair.DBZH.isel(radar=0)) & np.isfinite(pair.DBZH.isel(radar=1))
    assert both.values.sum() > 100
    # radar B (30 dBZ + 10) pulls the merge up by 10 w_b / (w_a + w_b)
    share = ((plain.DBZH - 30.0) / 10.0).values[both.values]
    odds = share / (1.0 - share)
    for blocked in (0.2, 0.4):
        grid = _with_blockage(pair, 0.0, blocked)
        out = multi.merge_radars(
            grid, "DBZH", no_echo=None, blockage="CBB", engine=engine
        )
        got = ((out.DBZH - 30.0) / 10.0).values[both.values]
        odds_b = odds * (1.0 - blocked)
        np.testing.assert_allclose(got, odds_b / (1.0 + odds_b), atol=2e-5)
        assert out.DBZH.attrs["merge_blockage"].startswith("weight (1 - b)")
        # the sum of the weights is the sum of the weighted weights
        assert (
            out.DBZH_weight.values[both.values] < plain.DBZH_weight.values[both.values]
        ).all()


@pytest.mark.parametrize("engine", ENGINES)
def test_a_blocked_radar_is_dropped(pair, engine):
    grid = _with_blockage(pair, 0.0, 0.6)
    out = multi.merge_radars(grid, "DBZH", blockage="CBB", engine=engine)
    only_a = multi.merge_radars(
        pair.isel(radar=[0]), "DBZH", no_echo=None, engine=engine
    )
    finite = np.isfinite(only_a.DBZH.values)
    np.testing.assert_allclose(out.DBZH.values[finite], 30.0, atol=1e-4)
    # at 0.6 the limit of 0.5 drops it; a limit of 0.7 keeps it with weight 0.4
    kept = multi.merge_radars(
        grid, "DBZH", blockage="CBB", max_blockage=0.7, engine=engine
    )
    assert np.nanmax(kept.DBZH.values) > 30.5
    # blockage of the first radar instead: the second one decides
    swapped = multi.merge_radars(
        _with_blockage(pair, 1.0, 0.0), "DBZH", blockage="CBB", engine=engine
    )
    only_b = multi.merge_radars(
        pair.isel(radar=[1]), "DBZH", no_echo=None, engine=engine
    )
    finite = np.isfinite(only_b.DBZH.values)
    np.testing.assert_allclose(swapped.DBZH.values[finite], 40.0, atol=1e-4)


@pytest.mark.parametrize("engine", ENGINES)
def test_no_blockage_nan_blockage_and_dataarray(pair, engine):
    plain = multi.merge_radars(pair, "DBZH", engine=engine)
    clear = multi.merge_radars(
        _with_blockage(pair, 0.0, 0.0), "DBZH", blockage="CBB", engine=engine
    )
    np.testing.assert_allclose(
        clear.DBZH.values, plain.DBZH.values, equal_nan=True, atol=1e-5
    )
    nan = multi.merge_radars(
        _with_blockage(pair, np.nan, np.nan), "DBZH", blockage="CBB", engine=engine
    )
    np.testing.assert_allclose(
        nan.DBZH.values, plain.DBZH.values, equal_nan=True, atol=1e-5
    )
    grid = _with_blockage(pair, 0.0, 0.3)
    named = multi.merge_radars(grid, "DBZH", blockage="CBB", engine=engine)
    given = multi.merge_radars(
        pair,
        "DBZH",
        blockage=grid["CBB"].transpose("radar", "x", "y", "z"),
        engine=engine,
    )
    np.testing.assert_array_equal(named.DBZH.values, given.DBZH.values)
    # a blockage that does not depend on the height is broadcast
    flat = grid["CBB"].isel(z=0, drop=True)
    again = multi.merge_radars(pair, "DBZH", blockage=flat, engine=engine)
    np.testing.assert_array_equal(named.DBZH.values, again.DBZH.values)


@needs_kernel
def test_merge_engines_agree_with_blockage(pair):
    rnd = np.random.default_rng(3)
    b = rnd.uniform(0.0, 0.9, pair["DBZH"].shape).astype(np.float32)
    b[rnd.random(b.shape) < 0.1] = np.nan
    grid = pair.assign(CBB=(pair["DBZH"].dims, b))
    a = multi.merge_radars(grid, "DBZH", blockage="CBB", engine="compiled")
    n = multi.merge_radars(grid, "DBZH", blockage="CBB", engine="numpy")
    np.testing.assert_array_equal(np.isnan(a.DBZH.values), np.isnan(n.DBZH.values))
    np.testing.assert_allclose(a.DBZH.values, n.DBZH.values, atol=1e-4, equal_nan=True)
    np.testing.assert_allclose(a.DBZH_weight.values, n.DBZH_weight.values, rtol=1e-4)


def test_blockage_errors(pair):
    grid = _with_blockage(pair, 0.0, 0.0)
    with pytest.raises(ValueError, match="no blockage field"):
        multi.merge_radars(grid, "DBZH", blockage="PBB")
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="max_blockage must be in"):
            multi.merge_radars(grid, "DBZH", blockage="CBB", max_blockage=bad)


@needs_kernel
def test_merge_kernel_checks_the_quality_shape(pair):
    from radarx.grid import _multi

    z = pair["z"].values
    ground = [np.ones(10)]
    values = np.ones((1, z.size, 10), dtype=np.float32)
    kw = dict(
        z=z,
        site_altitude=[0.0],
        elevations=[[0.5]],
        beamwidth=[1.0],
        time_offset=[0.0],
        range_scale=0.0,
        time_scale=0.0,
    )
    with pytest.raises(ValueError, match="quality must have the shape"):
        _multi.merge(values, ground, quality=np.ones((1, 2), dtype=np.float32), **kw)
    merged, weight = _multi.merge(values, ground, quality=np.ones_like(values), **kw)
    assert merged.shape == (z.size, 10)
