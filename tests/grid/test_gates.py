#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for radius of influence gridding of gates
===============================================
"""
import numpy as np
import pytest
import xarray as xr
from xradar.georeference import antenna_to_cartesian

import radarx  # noqa: F401
from radarx.grid import gates

ALT = 300.0
ELEVATIONS = (0.5, 1.5, 3.0, 6.0, 10.0)
ENGINES = ["numpy"] + (["compiled"] if gates.HAS_COMPILED_KERNEL else [])
T0 = np.datetime64("2026-01-01T00:00", "ns")

FIELDS = {
    "linear": lambda x, y, z: 30.0 + 1e-4 * x - 5e-5 * y - 2e-3 * (z - 2000.0),
    "plane_wave": lambda x, y, z: 20.0
    + 10.0 * np.sin(2 * np.pi * (0.6 * x + 0.8 * y) / 20e3),
    "gaussian_blob": lambda x, y, z: 10.0
    + 40.0
    * np.exp(-((x - 15e3) ** 2 + (y + 10e3) ** 2 + (z - 3e3) ** 2) / (2 * 6e3**2)),
}

GRID = dict(
    x=np.arange(-30e3, 30e3 + 1, 2e3),
    y=np.arange(-30e3, 30e3 + 1, 2e3),
    z=np.arange(1000.0, 5000.0 + 1, 500.0),
)


def _gate_xyz(az, el, rng_m):
    return antenna_to_cartesian(
        rng_m[None, :], az[:, None], el[:, None], site_altitude=ALT
    )


def _volume(
    f=None,
    elevations=ELEVATIONS,
    azimuth=None,
    nan_fraction=0.0,
    times=None,
    max_range=60e3,
    seed=0,
    name="DBZH",
):
    """
    PPI volume that measures ``f(x, y, z)`` at the gates (random if ``f`` is None).

    ``times`` gives the start time (seconds after T0) of each sweep; the field
    ``f`` may use a fourth argument ``t`` in seconds for a moving pattern.
    """
    rng_m = np.arange(125.0, max_range, 250.0)
    azimuth = np.arange(0.5, 360.0, 1.0) if azimuth is None else azimuth
    rnd = np.random.default_rng(seed)
    sweeps = {}
    for k, el in enumerate(elevations):
        az = np.roll(azimuth, 37 * (k + 1))
        gx, gy, gz = _gate_xyz(az, np.full(az.size, el), rng_m)
        t = 0.0 if times is None else times[k]
        if f is None:
            data = rnd.uniform(-10, 60, gz.shape)
        else:
            data = f(gx, gy, gz, t) if f.__code__.co_argcount == 4 else f(gx, gy, gz)
        if nan_fraction:
            data[rnd.random(data.shape) < nan_fraction] = np.nan
        stamp = T0 + np.timedelta64(int(t), "s")
        sweeps[f"sweep_{k}"] = xr.Dataset(
            {name: (("azimuth", "range"), data, {"units": "dBZ"})},
            coords={
                "azimuth": az,
                "range": rng_m,
                "elevation": ("azimuth", np.full(az.size, el)),
                "time": ("azimuth", np.full(az.size, stamp)),
            },
        )
    root = xr.Dataset(coords={"latitude": 45.0, "longitude": 10.0, "altitude": ALT})
    return xr.DataTree.from_dict({"/": root, **sweeps})


def _rhi_volume(f, azimuths=(90.0,), max_range=60e3, name="DBZH"):
    """RHI sweeps: a fixed azimuth, elevation 0 to 30 degrees along the rays."""
    rng_m = np.arange(125.0, max_range, 250.0)
    elevation = np.arange(0.25, 30.0, 0.5)
    sweeps = {}
    for k, azimuth in enumerate(azimuths):
        az = np.full(elevation.size, azimuth)
        gx, gy, gz = _gate_xyz(az, elevation, rng_m)
        sweeps[f"sweep_{k}"] = xr.Dataset(
            {name: (("time", "range"), f(gx, gy, gz), {"units": "dBZ"})},
            coords={
                "azimuth": ("time", az),
                "elevation": ("time", elevation),
                "range": rng_m,
                "time": T0 + np.arange(elevation.size) * np.timedelta64(1, "s"),
            },
        )
    root = xr.Dataset(coords={"latitude": 45.0, "longitude": 10.0, "altitude": ALT})
    return xr.DataTree.from_dict({"/": root, **sweeps})


def _brute(volume, points, weighting, roi, range_factor=0.0, height_factor=0.0):
    """Reference: every gate against every point, in plain NumPy."""
    xs, ys, zs, vs = [], [], [], []
    for name in ("sweep_0", "sweep_1", "sweep_2", "sweep_3", "sweep_4"):
        ds = volume[name].to_dataset()
        gx, gy, gz = _gate_xyz(
            ds["azimuth"].values, ds["elevation"].values, ds["range"].values
        )
        v = ds["DBZH"].values
        ok = np.isfinite(v)
        xs.append(gx[ok]), ys.append(gy[ok]), zs.append(gz[ok]), vs.append(v[ok])
    gx, gy, gz, gv = (np.concatenate(a) for a in (xs, ys, zs, vs))
    out = []
    for px, py, pz in points:
        r = max(roi, range_factor * np.hypot(px, py) + height_factor * (pz - ALT))
        d2 = (gx - px) ** 2 + (gy - py) ** 2 + (gz - pz) ** 2
        near = d2 < r * r
        if not near.any():
            out.append(np.nan)
        elif weighting == "nearest":
            out.append(gv[near][np.argmin(d2[near])])
        else:
            w = (
                (r * r - d2[near]) / (r * r + d2[near])
                if weighting == "cressman"
                else np.exp(-4 * d2[near] / r**2)
            )
            out.append(np.sum(w * gv[near]) / np.sum(w))
    return np.array(out)


@pytest.fixture(scope="module")
def random_volume():
    return _volume(nan_fraction=0.1)


# ---------------------------------------------------------------------------
# weights against a brute-force reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("weighting", ["cressman", "barnes", "nearest"])
def test_weights_match_brute_force(random_volume, engine, weighting):
    rnd = np.random.default_rng(4)
    points = np.column_stack(
        [
            rnd.uniform(-25e3, 25e3, 40),
            rnd.uniform(-25e3, 25e3, 40),
            rnd.uniform(900, 4000, 40),
        ]
    )
    expected = _brute(random_volume, points, weighting, 2000.0, 0.02, 0.1)
    for p, e in zip(points, expected):
        out = gates.grid_gates(
            random_volume,
            "DBZH",
            [p[0]],
            [p[1]],
            [p[2]],
            weighting=weighting,
            roi=2000.0,
            roi_range_factor=0.02,
            roi_height_factor=0.1,
            engine=engine,
        )["DBZH"]
        np.testing.assert_allclose(out.values.item(), e, rtol=2e-6, equal_nan=True)


@pytest.mark.parametrize("weighting", ["cressman", "barnes"])
@pytest.mark.parametrize("engine", ENGINES)
def test_a_gate_at_the_point_and_a_gate_at_the_radius(weighting, engine):
    """Weight 1 at distance 0 and Cressman 0 at the radius (excluded)."""
    ds = xr.Dataset(
        {"DBZH": (("azimuth", "range"), np.array([[10.0, 50.0]] * 4))},
        coords={
            "azimuth": [0.0, 90.0, 180.0, 270.0],
            "range": [1000.0, 3000.0],
            "elevation": ("azimuth", np.zeros(4)),
            "time": ("azimuth", np.full(4, T0)),
        },
    )
    root = xr.Dataset(coords={"latitude": 45.0, "longitude": 10.0, "altitude": 0.0})
    tree = xr.DataTree.from_dict({"/": root, "sweep_0": ds})
    gx, gy, gz = antenna_to_cartesian(
        np.array([[1000.0, 3000.0]]),
        np.array([[0.0]]),
        np.array([[0.0]]),
        site_altitude=0.0,
    )
    out = gates.grid_gates(
        tree,
        "DBZH",
        [float(gx[0, 0])],
        [float(gy[0, 0])],
        [float(gz[0, 0])],
        weighting=weighting,
        roi=2000.0,
        engine=engine,
    )
    # the other gates of range 1000 m are 1414 m away (sqrt(2) r) or 2000 m
    value = out["DBZH"].values.item()
    assert 10.0 <= value < 20.0
    none = gates.grid_gates(
        tree,
        "DBZH",
        [float(gx[0, 0])],
        [float(gy[0, 0])],
        [float(gz[0, 0])],
        weighting=weighting,
        roi=1e-3,
        min_gates=1,
        engine=engine,
    )
    assert none["DBZH"].values.item() == pytest.approx(10.0, abs=1e-5)


# ---------------------------------------------------------------------------
# engines, threads, options
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not gates.HAS_COMPILED_KERNEL, reason="compiled kernel not built")
@pytest.mark.parametrize("weighting", ["cressman", "barnes", "nearest"])
@pytest.mark.parametrize(
    "options",
    [
        dict(roi=1500.0),
        dict(roi=800.0, beamwidth=1.0),
        dict(roi=1000.0, roi_range_factor=0.01, roi_height_factor=0.2, min_gates=3),
        dict(roi=1500.0, motion=(10.0, -5.0), time="2026-01-01T00:03"),
    ],
)
def test_engines_agree(weighting, options):
    dtree = _volume(nan_fraction=0.1, times=[0, 60, 120, 180, 240])
    # off-grid points: on the symmetric grid two gates can be equally near
    grid = {k: v + 17.3 * (i + 1) for i, (k, v) in enumerate(GRID.items())}
    kw = dict(weighting=weighting, **options)
    a = gates.grid_gates(dtree, "DBZH", **grid, engine="compiled", **kw)
    b = gates.grid_gates(dtree, "DBZH", **grid, engine="numpy", **kw)
    A, B = a.DBZH.values, b.DBZH.values
    np.testing.assert_array_equal(np.isnan(A), np.isnan(B))
    np.testing.assert_allclose(A, B, atol=1e-4, equal_nan=True)


@pytest.mark.skipif(not gates.HAS_COMPILED_KERNEL, reason="compiled kernel not built")
def test_threads_do_not_change_the_result(random_volume):
    one = gates.grid_gates(random_volume, "DBZH", **GRID, n_threads=1)
    many = gates.grid_gates(random_volume, "DBZH", **GRID, n_threads=4)
    np.testing.assert_array_equal(one.DBZH.values, many.DBZH.values)


@pytest.mark.parametrize("engine", ENGINES)
def test_bigger_radius_fills_more_and_beam_width_grows_with_range(
    random_volume, engine
):
    small = gates.grid_gates(random_volume, "DBZH", **GRID, roi=700.0, engine=engine)
    large = gates.grid_gates(random_volume, "DBZH", **GRID, roi=2500.0, engine=engine)
    assert np.isfinite(large.DBZH.values).mean() > np.isfinite(small.DBZH.values).mean()
    # the radius is s tan(1 degree) = 0.017 s: 0.9 km at 50 km, below roi near the radar
    fixed = gates.grid_gates(random_volume, "DBZH", **GRID, roi=200.0, engine=engine)
    beam = gates.grid_gates(
        random_volume, "DBZH", **GRID, roi=200.0, beamwidth=1.0, engine=engine
    )
    far = dict(x=slice(20e3, 30e3), y=slice(20e3, 30e3))
    assert (
        np.isfinite(beam.DBZH.sel(**far).values).mean()
        > np.isfinite(fixed.DBZH.sel(**far).values).mean()
    )
    near = dict(x=slice(-5e3, 5e3), y=slice(-5e3, 5e3))
    np.testing.assert_array_equal(
        np.isfinite(beam.DBZH.sel(**near).values),
        np.isfinite(fixed.DBZH.sel(**near).values),
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_min_gates_and_empty_volume(random_volume, engine):
    one = gates.grid_gates(random_volume, "DBZH", **GRID, roi=1200.0, engine=engine)
    three = gates.grid_gates(
        random_volume, "DBZH", **GRID, roi=1200.0, min_gates=3, engine=engine
    )
    assert np.isfinite(three.DBZH.values).sum() < np.isfinite(one.DBZH.values).sum()
    empty = _volume(nan_fraction=1.0)
    out = gates.grid_gates(empty, "DBZH", **GRID, engine=engine)
    assert np.isnan(out.DBZH.values).all()


# ---------------------------------------------------------------------------
# truth
# ---------------------------------------------------------------------------
def _rms_against_truth(out, f):
    z, y, x = xr.broadcast(out["z"], out["y"], out["x"])
    truth = f(
        x.values.transpose(1, 0, 2),
        y.values.transpose(1, 0, 2),
        z.values.transpose(1, 0, 2),
    )
    return truth


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "weighting,tolerance", [("cressman", 1.0), ("barnes", 1.0), ("nearest", 1.5)]
)
@pytest.mark.parametrize("name", list(FIELDS))
def test_known_fields(name, weighting, tolerance, engine):
    f = FIELDS[name]
    dtree = _volume(f)
    out = gates.grid_gates(
        dtree,
        "DBZH",
        **GRID,
        weighting=weighting,
        roi=1500.0,
        beamwidth=1.0,
        engine=engine,
    )["DBZH"]
    X, Y, Z = np.meshgrid(GRID["x"], GRID["y"], GRID["z"], indexing="ij")
    truth = f(X, Y, Z).transpose(2, 1, 0)
    ok = np.isfinite(out.values)
    assert ok.mean() > 0.7
    rms = np.sqrt(np.mean((out.values[ok] - truth[ok]) ** 2))
    assert rms < tolerance, (name, weighting, rms)


def test_smooth_weights_beat_the_nearest_gate_on_a_wave():
    f = FIELDS["plane_wave"]
    dtree = _volume(f)
    X, Y, Z = np.meshgrid(GRID["x"], GRID["y"], GRID["z"], indexing="ij")
    truth = f(X, Y, Z).transpose(2, 1, 0)
    rms = {}
    for weighting in ("cressman", "barnes", "nearest"):
        out = gates.grid_gates(
            dtree, "DBZH", **GRID, weighting=weighting, roi=1500.0, engine="numpy"
        )["DBZH"].values
        ok = np.isfinite(out)
        rms[weighting] = np.sqrt(np.mean((out[ok] - truth[ok]) ** 2))
    assert rms["cressman"] < rms["nearest"] and rms["barnes"] < rms["nearest"]


# ---------------------------------------------------------------------------
# scan geometries
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("engine", ENGINES)
def test_sector_scan_leaves_the_rest_empty(engine):
    f = FIELDS["linear"]
    dtree = _volume(f, azimuth=np.arange(0.5, 90.0, 1.0))
    out = gates.grid_gates(dtree, "DBZH", **GRID, roi=1500.0, engine=engine)["DBZH"]
    assert np.isfinite(out.sel(x=20e3, y=20e3, z=2000.0))  # inside the sector
    assert np.isnan(out.sel(x=-20e3, y=-20e3).values).all()
    inside = out.sel(x=slice(5e3, 25e3), y=slice(5e3, 25e3)).values
    assert np.isfinite(inside).any()


@pytest.mark.parametrize("engine", ENGINES)
def test_rhi_section(engine):
    f = FIELDS["gaussian_blob"]
    dtree = _rhi_volume(f, azimuths=(90.0,))
    distance = np.arange(2e3, 50e3, 1e3)
    z = np.arange(500.0, 8000.0, 500.0)
    out = gates.grid_rhi(
        dtree, "DBZH", distance=distance, z=z, roi=800.0, beamwidth=0.5, engine=engine
    )
    assert out["DBZH"].dims == ("z", "distance")
    assert float(out.azimuth) == pytest.approx(90.0)
    ok = np.isfinite(out["DBZH"].values)
    assert ok.mean() > 0.5
    truth = f(distance[None, :], np.zeros((1, 1)), z[:, None])
    rms = np.sqrt(
        np.mean((out["DBZH"].values[ok] - np.broadcast_to(truth, ok.shape)[ok]) ** 2)
    )
    assert rms < 1.0
    # a section needs the gates near its plane: another azimuth finds none
    far = gates.grid_rhi(
        dtree, "DBZH", azimuth=270.0, distance=distance, z=z, roi=800.0, engine=engine
    )
    assert np.isnan(far["DBZH"].values).all()
    # the same section through the grid function
    grid = gates.grid_gates(
        dtree,
        "DBZH",
        x=distance,
        y=[0.0],
        z=z,
        roi=800.0,
        beamwidth=0.5,
        engine=engine,
    )["DBZH"].isel(y=0)
    np.testing.assert_allclose(
        grid.values, out["DBZH"].values, equal_nan=True, atol=1e-4
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_rhi_plane_azimuth_defaults_to_the_volume_azimuth(engine):
    dtree = _rhi_volume(FIELDS["linear"], azimuths=(45.0,))
    out = gates.grid_rhi(
        dtree,
        "DBZH",
        distance=np.arange(2e3, 20e3, 2e3),
        z=[1000.0, 2000.0],
        roi=1000.0,
        engine=engine,
    )
    assert float(out.azimuth) == pytest.approx(45.0)
    assert np.isfinite(out["DBZH"].values).any()
    assert out["DBZH"].attrs["units"] == "dBZ"
    assert out.attrs["gridding_method"] == "gates"


@pytest.mark.parametrize("engine", ENGINES)
def test_rhis_in_nearby_azimuths_fill_the_section_between_them(engine):
    f = FIELDS["linear"]
    dtree = _rhi_volume(f, azimuths=(89.0, 91.0))
    out = gates.grid_rhi(
        dtree,
        "DBZH",
        azimuth=90.0,
        distance=np.arange(5e3, 30e3, 5e3),
        z=[2000.0, 4000.0],
        roi=2500.0,
        engine=engine,
    )
    X = out["distance"].values[None, :]
    truth = f(X, 0.0 * X, out["z"].values[:, None])
    ok = np.isfinite(out["DBZH"].values)
    assert ok.any()
    assert (
        np.abs(out["DBZH"].values[ok] - np.broadcast_to(truth, ok.shape)[ok]).max()
        < 3.0
    )


# ---------------------------------------------------------------------------
# repeated low tilts
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("engine", ENGINES)
def test_repeated_low_tilts_are_moved_to_the_analysis_time(engine):
    """A pattern that moves 10 m/s east, seen by the 0.5 degree tilt at 0, 60 and 120 s."""
    u = 10.0

    def moving(x, y, z, t):
        return 30.0 + 20.0 * np.exp(-(((x - 10e3 - u * t) / 4e3) ** 2 + (y / 6e3) ** 2))

    dtree = _volume(
        moving, elevations=(0.5, 0.5, 0.5), times=[0, 600, 1200], max_range=40e3
    )
    kw = dict(
        x=np.arange(0, 30e3, 1e3), y=np.arange(-10e3, 10e3, 1e3), z=[400.0, 600.0]
    )
    t_end = "2026-01-01T00:20"
    plain = gates.grid_gates(dtree, "DBZH", **kw, roi=1500.0, engine=engine)
    moved = gates.grid_gates(
        dtree, "DBZH", **kw, roi=1500.0, motion=(u, 0.0), time=t_end, engine=engine
    )
    truth = moving(kw["x"][None, :], kw["y"][:, None], 0.0, 1200.0)

    def error(grid):
        v = grid["DBZH"].sel(z=400.0).values
        ok = np.isfinite(v)
        return np.sqrt(np.mean((v[ok] - truth[ok]) ** 2))

    assert error(moved) < 0.5 * error(plain)
    assert str(moved.time.values)[:16] == "2026-01-01T00:20"
    assert "u = 10.0" in moved.attrs["advection"]


# ---------------------------------------------------------------------------
# metadata, input and errors
# ---------------------------------------------------------------------------
def test_output_layout_and_georeferencing(random_volume):
    out = gates.grid_gates(random_volume, None, **GRID)
    assert out["DBZH"].dims == ("z", "y", "x") and out["DBZH"].dtype == np.float32
    assert out["DBZH"].attrs["units"] == "dBZ"
    assert out["DBZH"].attrs["grid_mapping"] == "crs_wkt"
    assert out["lat"].dims == ("y", "x") and "time" in out.coords
    assert out.attrs["gridding_method"] == "gates"
    assert out.attrs["weighting"] == "cressman" and out.attrs["roi"] == 1000.0
    assert out.attrs["radarx_method"].startswith("Radius-of-influence")
    assert (
        "10.1175/1520-0493(1959)087<0367:AOOAS>2.0.CO;2"
        in out.attrs["radarx_references"]
    )
    assert (
        "10.1175/1520-0450(1964)003<0396:ATFMDI>2.0.CO;2"
        in out.attrs["radarx_references"]
    )


def test_input_is_not_changed(random_volume):
    before = random_volume["sweep_0"].to_dataset().copy(deep=True)
    for engine in ENGINES:
        gates.grid_gates(random_volume, "DBZH", **GRID, engine=engine)
    xr.testing.assert_identical(random_volume["sweep_0"].to_dataset(), before)


def test_accessors(random_volume):
    out = random_volume.radarx.grid_gates("DBZH", **GRID, weighting="barnes")
    assert out.attrs["weighting"] == "barnes"
    rhi = _rhi_volume(FIELDS["linear"])
    section = rhi.radarx.grid_rhi(
        "DBZH", distance=np.arange(5e3, 20e3, 5e3), z=[1000.0, 2000.0], roi=1000.0
    )
    assert section["DBZH"].dims == ("z", "distance")


def test_errors(random_volume, monkeypatch):
    with pytest.raises(ValueError, match="required"):
        gates.grid_gates(random_volume, "DBZH", x=GRID["x"])
    with pytest.raises(ValueError, match="required"):
        gates.grid_rhi(random_volume, "DBZH", z=GRID["z"])
    with pytest.raises(ValueError, match="finite"):
        gates.grid_gates(random_volume, "DBZH", x=[np.nan], y=[0.0], z=[1000.0])
    with pytest.raises(ValueError, match="finite"):
        gates.grid_rhi(random_volume, "DBZH", distance=[1e3], z=[np.inf])
    with pytest.raises(ValueError, match="weighting must be"):
        gates.grid_gates(random_volume, "DBZH", **GRID, weighting="kriging")
    with pytest.raises(ValueError, match="weighting must be"):
        gates.grid_gates(random_volume, "DBZH", **GRID, weighting=None)
    with pytest.raises(ValueError, match="engine must be"):
        gates.grid_gates(random_volume, "DBZH", **GRID, engine="gpu")
    with pytest.raises(ValueError, match="No sweep contains"):
        gates.grid_gates(random_volume, "VRADH", **GRID)
    for bad in (
        dict(roi=0.0),
        dict(roi=np.nan),
        dict(roi_range_factor=-1.0),
        dict(roi_height_factor=-0.1),
        dict(beamwidth=0.0),
        dict(beamwidth=95.0),
        dict(min_gates=0),
    ):
        with pytest.raises(ValueError):
            gates.grid_gates(random_volume, "DBZH", **GRID, **bad)
    with pytest.raises(ValueError, match="positive"):
        gates.grid_gates(random_volume, "DBZH", **GRID, k_e=-1.0)
    monkeypatch.setattr(gates, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError, match="compiled"):
        gates.grid_gates(random_volume, "DBZH", **GRID, engine="compiled")
    assert not gates._use_compiled("auto")


@pytest.mark.skipif(not gates.HAS_COMPILED_KERNEL, reason="compiled kernel not built")
def test_kernel_argument_checks():
    from radarx.grid import _cone

    one = np.zeros((4, 3))
    args = dict(
        cx=np.zeros(1),
        cy=np.zeros(1),
        z=np.zeros(1),
        data=[one],
        azimuth=[np.zeros(4)],
        elevation=[np.zeros(4)],
        range=[np.ones(3)],
        site_altitude=0.0,
    )
    _cone.grid_gates(**args)
    for change in (
        dict(azimuth=[np.zeros(3)]),
        dict(data=[np.zeros(4)]),
        dict(elevation=[]),
        dict(shift=[0.0]),
        dict(weighting=7),
        dict(roi=-1.0),
        dict(min_gates=0),
        dict(cy=np.zeros(2)),
    ):
        with pytest.raises(ValueError):
            _cone.grid_gates(**{**args, **change})
    assert _cone.grid_gates(**{**args, "z": np.zeros(0)}).shape == (0, 1)


def test_numpy_without_any_gate_in_reach():
    out = gates._map_numpy(
        np.array([0.0]),
        np.array([0.0]),
        np.array([100.0]),
        [
            (
                np.full((4, 3), np.nan),
                np.arange(4) * 90.0,
                np.zeros(4),
                np.array([1e3, 2e3, 3e3]),
            )
        ],
        [(0.0, 0.0)],
        0.0,
        4 / 3,
        "cressman",
        500.0,
        0.0,
        0.0,
        1,
    )
    assert out.shape == (1, 1) and np.isnan(out).all()
