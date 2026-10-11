#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for radarx cone gridding
==============================
"""
import numpy as np
import pytest
import xarray as xr
from xradar.georeference import antenna_to_cartesian

import radarx  # noqa: F401
from radarx.grid import cone

ALT = 300.0
ELEVATIONS = (0.5, 1.5, 3.0, 6.0, 10.0)
ENGINES = ["numpy"] + (["compiled"] if cone.HAS_COMPILED_KERNEL else [])


def _volume(
    field="height",
    elevations=ELEVATIONS,
    azimuth=None,
    nan_fraction=0.0,
    seed=0,
    name="DBZH",
    k_e=cone.K_E,
):
    """
    Synthetic PPI volume; ``field="height"`` stores each gate's beam height
    (for the effective Earth radius factor ``k_e``) in the variable ``name``.
    """
    rng_m = np.arange(125.0, 60_000.0, 250.0)
    azimuth = np.arange(0.5, 360.0, 1.0) if azimuth is None else azimuth
    rnd = np.random.default_rng(seed)
    sweeps = {}
    for k, el in enumerate(elevations):
        # rays start mid-rotation and wobble a little in elevation, like real scans
        az = np.roll(azimuth, 37 * (k + 1))
        ray_el = el + 0.05 * np.sin(np.deg2rad(az))
        _, _, gz = antenna_to_cartesian(
            rng_m[None, :],
            az[:, None],
            ray_el[:, None],
            site_altitude=ALT,
            **cone._antenna_kwargs(k_e),
        )
        if field == "height":
            data = gz.copy()
        else:
            data = rnd.uniform(-10, 60, gz.shape)
        if nan_fraction:
            data[rnd.random(data.shape) < nan_fraction] = np.nan
        sweeps[f"sweep_{k}"] = xr.Dataset(
            {name: (("azimuth", "range"), data, {"units": "dBZ"})},
            coords={
                "azimuth": az,
                "range": rng_m,
                "elevation": ("azimuth", ray_el),
                "time": (
                    "azimuth",
                    np.full(az.size, np.datetime64("2026-01-01T00:00")),
                ),
            },
        )
    root = xr.Dataset(coords={"latitude": 45.0, "longitude": 10.0, "altitude": ALT})
    return xr.DataTree.from_dict({"/": root, **sweeps})


GRID = dict(
    x=np.arange(-40e3, 40e3 + 1, 2e3),
    y=np.arange(-40e3, 40e3 + 1, 2e3),
    z=np.arange(500.0, 6000.0 + 1, 250.0),
)


@pytest.mark.parametrize("engine", ENGINES)
def test_height_field_is_reproduced(engine):
    """Gridding each gate's own height must return the level height."""
    out = cone.grid_cones(_volume(), "DBZH", **GRID, engine=engine)["DBZH"]
    level = xr.broadcast(out["z"], out)[0]
    filled = np.isfinite(out.values)
    assert filled.mean() > 0.3
    np.testing.assert_allclose(out.values[filled], level.values[filled], atol=2.0)


@pytest.mark.parametrize("engine", ENGINES)
def test_no_overshoot_and_gaps_respected(engine):
    dtree = _volume(field="random", nan_fraction=0.2)
    out = cone.grid_cones(dtree, "DBZH", **GRID, engine=engine)["DBZH"].values
    assert np.nanmin(out) >= -10 - 1e-4
    assert np.nanmax(out) <= 60 + 1e-4


@pytest.mark.skipif(not cone.HAS_COMPILED_KERNEL, reason="compiled kernel not built")
@pytest.mark.parametrize("fill_below", [False, True])
def test_engines_agree(fill_below):
    dtree = _volume(field="random", nan_fraction=0.1)
    a = cone.grid_cones(dtree, "DBZH", **GRID, engine="compiled", fill_below=fill_below)
    b = cone.grid_cones(dtree, "DBZH", **GRID, engine="numpy", fill_below=fill_below)
    np.testing.assert_array_equal(np.isnan(a.DBZH.values), np.isnan(b.DBZH.values))
    np.testing.assert_allclose(a.DBZH.values, b.DBZH.values, atol=1e-4, equal_nan=True)


@pytest.mark.parametrize("engine", ENGINES)
def test_fill_below_and_cone_of_silence(engine):
    dtree = _volume()
    strict = cone.grid_cones(dtree, "DBZH", **GRID, engine=engine)["DBZH"]
    filled = cone.grid_cones(dtree, "DBZH", **GRID, engine=engine, fill_below=True)[
        "DBZH"
    ]
    # far from the radar the lowest level lies below the 0.5 deg beam
    far = dict(x=40e3, y=0.0, z=500.0)
    assert np.isnan(float(strict.sel(**far)))
    assert np.isfinite(float(filled.sel(**far)))
    # straight above the radar, above the 10 deg cone: nothing to interpolate
    assert np.isnan(float(filled.sel(x=2e3, y=0.0, z=6000.0)))


@pytest.mark.parametrize("engine", ENGINES)
def test_sector_scan_does_not_bridge_gap(engine):
    dtree = _volume(azimuth=np.arange(0.5, 90.0, 1.0))
    out = cone.grid_cones(dtree, "DBZH", **GRID, engine=engine)["DBZH"]
    assert np.isfinite(out.sel(x=20e3, y=20e3, z=1000.0))  # inside the sector
    assert np.isnan(out.sel(x=-20e3, y=-20e3).values).all()  # opposite side


def test_output_layout_and_metadata():
    out = cone.grid_cones(_volume(), None, **GRID)
    assert out["DBZH"].dims == ("z", "y", "x")
    assert out["DBZH"].dtype == np.float32
    assert out["DBZH"].attrs["units"] == "dBZ"
    for name in ("lat", "lon", "latitude", "longitude", "altitude", "time"):
        assert name in out.variables
    assert "time" in out.coords and out["time"].ndim == 0
    assert out.lat.dims == ("y", "x") and out.lon.dims == ("y", "x")
    np.testing.assert_allclose(float(out.lat.sel(x=0.0, y=0.0)), 45.0)
    assert out.attrs["gridding_method"] == "cone"


def test_split_cuts_use_the_longest_range():
    near = _volume(elevations=(0.5,))["sweep_0"].to_dataset().isel(range=slice(0, 50))
    far = _volume(elevations=(0.5,))["sweep_0"].to_dataset()
    selected = cone._select_sweeps([near, far], "DBZH")
    assert len(selected) == 1 and selected[0].sizes["range"] == far.sizes["range"]
    assert cone._select_sweeps([near], "VRADH") == []


def test_grid_radar_methods():
    dtree = _volume()
    kw = dict(
        x_lim=(-20e3, 20e3),
        y_lim=(-20e3, 20e3),
        z_lim=(500, 3000),
        x_step=2e3,
        y_step=2e3,
        z_step=500,
    )
    out = dtree.radarx.to_grid(data_vars=["DBZH"], **kw)
    assert out.attrs["gridding_method"] == "cone"
    assert out["DBZH"].shape == (6, 21, 21)
    with pytest.raises(ValueError, match="method must be"):
        dtree.radarx.to_grid(data_vars=["DBZH"], method="kriging", **kw)


def test_errors(monkeypatch):
    dtree = _volume()
    with pytest.raises(ValueError, match="required"):
        cone.grid_cones(dtree, "DBZH", x=GRID["x"])
    with pytest.raises(ValueError, match="engine must be"):
        cone.grid_cones(dtree, "DBZH", **GRID, engine="gpu")
    with pytest.raises(ValueError, match="No sweep contains"):
        cone.grid_cones(dtree, "VRADH", **GRID)
    with pytest.raises(ValueError, match="No sweep groups"):
        cone.grid_cones(xr.DataTree(xr.Dataset()), "DBZH", **GRID)
    no_site = xr.DataTree.from_dict(
        {"/": xr.Dataset(), "sweep_0": dtree["sweep_0"].to_dataset(inherit=False)}
    )
    with pytest.raises(ValueError, match="radar site"):
        cone.grid_cones(no_site, "DBZH", **GRID)
    monkeypatch.setattr(cone, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError, match="compiled"):
        cone.grid_cones(dtree, "DBZH", **GRID, engine="compiled")


def test_sweep_dataset_falls_back_for_older_xarray(monkeypatch):
    dtree = _volume()
    node = dtree["sweep_0"]
    original = type(node).to_dataset

    def old_to_dataset(self, inherit=True):
        if inherit == "all_coords":
            raise TypeError("unsupported")
        return original(self, inherit=inherit)

    monkeypatch.setattr(type(node), "to_dataset", old_to_dataset)
    assert "DBZH" in cone._sweep_dataset(dtree, "sweep_0")


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("k_e", [1.0, 1.2])
def test_k_e_sets_the_beam_height(engine, k_e):
    """Gridding gate heights computed with k_e returns the level height."""
    dtree = _volume(k_e=k_e)
    out = cone.grid_cones(dtree, "DBZH", **GRID, k_e=k_e, engine=engine)["DBZH"]
    level = xr.broadcast(out["z"], out)[0]
    filled = np.isfinite(out.values)
    assert filled.mean() > 0.3
    np.testing.assert_allclose(out.values[filled], level.values[filled], atol=2.0)
    assert cone.grid_cones(dtree, "DBZH", **GRID, k_e=k_e, engine=engine).attrs[
        "k_e"
    ] == pytest.approx(k_e)
    # the default 4/3 puts the same gates at other heights
    default = cone.grid_cones(dtree, "DBZH", **GRID, engine=engine)["DBZH"]
    both = np.isfinite(default.values) & filled
    assert np.abs(default.values[both] - level.values[both]).max() > 20.0


def test_k_e_default_is_four_thirds():
    dtree = _volume()
    for engine in ENGINES:
        a = cone.grid_cones(dtree, "DBZH", **GRID, engine=engine)
        b = cone.grid_cones(dtree, "DBZH", **GRID, k_e=4.0 / 3.0, engine=engine)
        np.testing.assert_array_equal(a.DBZH.values, b.DBZH.values)


@pytest.mark.skipif(not cone.HAS_COMPILED_KERNEL, reason="compiled kernel not built")
@pytest.mark.parametrize("fill_depth", [None, 100.0, 1000.0])
def test_engines_agree_with_k_e_and_fill_depth(fill_depth):
    dtree = _volume(field="random", nan_fraction=0.1)
    kw = dict(k_e=1.1, fill_below=True, fill_depth=fill_depth)
    z = np.arange(100.0, 3000.0, 150.0)
    grid = dict(x=GRID["x"], y=GRID["y"], z=z)
    a = cone.grid_cones(dtree, "DBZH", **grid, engine="compiled", **kw)
    b = cone.grid_cones(dtree, "DBZH", **grid, engine="numpy", **kw)
    np.testing.assert_array_equal(np.isnan(a.DBZH.values), np.isnan(b.DBZH.values))
    np.testing.assert_allclose(a.DBZH.values, b.DBZH.values, atol=1e-4, equal_nan=True)


@pytest.mark.parametrize("engine", ENGINES)
def test_k_e_must_be_positive(engine):
    with pytest.raises(ValueError, match="k_e"):
        cone.grid_cones(_volume(), "DBZH", **GRID, k_e=0.0, engine=engine)


@pytest.mark.parametrize("engine", ENGINES)
def test_fill_depth_limits_the_pseudo_cappi(engine):
    """The lowest beam fills only a limited depth, never below the radar."""
    dtree = _volume()
    z = np.array([100.0, 250.0, 400.0, 500.0, 600.0, 800.0])
    grid = dict(x=GRID["x"], y=GRID["y"], z=z)
    far = dict(x=40e3, y=0.0)
    kw = dict(fill_below=True, engine=engine)
    unlimited = cone.grid_cones(dtree, "DBZH", **grid, **kw)["DBZH"].sel(**far)
    # as before: every level below the lowest beam gets its value, even the
    # levels below the radar altitude (300 m)
    assert np.isfinite(unlimited.values[:4]).all()
    limited = cone.grid_cones(dtree, "DBZH", **grid, fill_depth=200.0, **kw)["DBZH"]
    limited = limited.sel(**far)
    # the 0.5 degree beam at 40 km is near 740 m: 600 m is within 200 m of it,
    # 400 m is not, and 250 m lies below the radar
    assert np.isnan(limited.sel(z=250.0)) and np.isnan(limited.sel(z=400.0))
    assert np.isfinite(limited.sel(z=600.0))
    deep = cone.grid_cones(dtree, "DBZH", **grid, fill_depth=1000.0, **kw)["DBZH"]
    deep = deep.sel(**far)
    assert np.isnan(deep.sel(z=100.0)) and np.isnan(deep.sel(z=250.0))
    assert np.isfinite(deep.sel(z=400.0))
    inf = cone.grid_cones(dtree, "DBZH", **grid, fill_depth=np.inf, **kw)["DBZH"]
    np.testing.assert_array_equal(inf.sel(**far).values, unlimited.values)


@pytest.mark.parametrize("engine", ENGINES)
def test_velocity_is_not_filled_below_the_lowest_beam(engine):
    dtree = _volume(name="VRADH")
    far = dict(x=40e3, y=0.0, z=500.0)
    kw = dict(fill_below=True, engine=engine)
    default = cone.grid_cones(dtree, "VRADH", **GRID, **kw)["VRADH"]
    assert np.isnan(float(default.sel(**far)))
    old = cone.grid_cones(dtree, "VRADH", **GRID, fill_velocity=True, **kw)["VRADH"]
    assert np.isfinite(float(old.sel(**far)))
    # a reflectivity field is still filled
    refl = cone.grid_cones(_volume(), "DBZH", **GRID, **kw)["DBZH"]
    assert np.isfinite(float(refl.sel(**far)))


def _moving_blob_volume(u, v, step=60.0, x0=15e3, y0=5e3, width=10e3):
    """
    Volume of a storm drifting with (u, v) m/s, seen by one tilt every ``step`` s.

    The field is independent of height: tilt k shows the blob where it is at
    ``t0 + k step``.
    """
    rng_m = np.arange(125.0, 60_000.0, 250.0)
    azimuth = np.arange(0.5, 360.0, 1.0)
    t0 = np.datetime64("2026-01-01T00:00:00", "ns")
    sweeps = {}
    for k, el in enumerate(ELEVATIONS):
        gx, gy, _ = antenna_to_cartesian(
            rng_m[None, :], azimuth[:, None], np.full((azimuth.size, 1), el)
        )
        t = k * step
        data = 40.0 * np.exp(
            -((gx - x0 - u * t) ** 2 + (gy - y0 - v * t) ** 2) / width**2
        )
        when = t0 + np.timedelta64(int(t * 1e9), "ns")
        sweeps[f"sweep_{k}"] = xr.Dataset(
            {"DBZH": (("azimuth", "range"), data, {"units": "dBZ"})},
            coords={
                "azimuth": azimuth,
                "range": rng_m,
                "elevation": ("azimuth", np.full(azimuth.size, el)),
                "time": ("azimuth", np.full(azimuth.size, when)),
            },
        )
    root = xr.Dataset(coords={"latitude": 45.0, "longitude": 10.0, "altitude": 0.0})
    tree = xr.DataTree.from_dict({"/": root, **sweeps})
    return tree, t0, (x0, y0, width)


@pytest.mark.parametrize("engine", ENGINES)
def test_each_sweep_is_advected_from_its_own_time(engine):
    """Tilts a minute apart show a storm at 20 m/s 1.2 km further each (#223)."""
    u, v = 20.0, -5.0
    dtree, t0, (x0, y0, width) = _moving_blob_volume(u, v)
    grid = dict(
        x=np.arange(-40e3, 60e3 + 1, 2e3),
        y=np.arange(-40e3, 40e3 + 1, 2e3),
        z=np.arange(500.0, 5000.0 + 1, 250.0),
    )
    X, Y = np.meshgrid(grid["x"], grid["y"])
    # the blob at the analysis time t0, the time of the lowest tilt
    truth = 40.0 * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / width**2)
    still = cone.grid_cones(dtree, "DBZH", **grid, engine=engine)["DBZH"]
    moved = cone.grid_cones(
        dtree, "DBZH", **grid, engine=engine, motion=(u, v), time=t0
    )
    near = (np.hypot(X, Y) > 8e3) & (np.hypot(X, Y) < 35e3)
    both = np.isfinite(moved["DBZH"].values) & np.isfinite(still.values) & near
    assert both.sum() > 1000
    # all levels hold the blob where it is at t0, up to the 2 km cells
    err = np.abs(moved["DBZH"].values - truth)[both]
    assert err.max() < 0.6
    # without advection the tilts disagree by up to 5 km and the layers blur
    assert np.abs(still.values - truth)[both].max() > 5.0
    assert moved["time"].values == t0
    assert "advection" in moved.attrs
    assert "advection" not in still.attrs


@pytest.mark.skipif(not cone.HAS_COMPILED_KERNEL, reason="compiled kernel not built")
def test_advected_engines_agree():
    dtree, t0, _ = _moving_blob_volume(20.0, -5.0)
    kw = dict(motion=(20.0, -5.0), time=t0, fill_below=True)
    a = cone.grid_cones(dtree, "DBZH", **GRID, engine="compiled", **kw)
    b = cone.grid_cones(dtree, "DBZH", **GRID, engine="numpy", **kw)
    np.testing.assert_allclose(a.DBZH.values, b.DBZH.values, atol=1e-4, equal_nan=True)


def test_motion_must_be_a_pair():
    dtree, _, _ = _moving_blob_volume(0.0, 0.0)
    with pytest.raises(ValueError, match="motion"):
        cone.grid_cones(dtree, "DBZH", **GRID, motion=5.0)


def test_negative_fill_depth_is_rejected():
    with pytest.raises(ValueError, match="fill_depth must not be negative"):
        cone.grid_cones(_volume(), "DBZH", **GRID, fill_depth=-1.0)
    assert cone._fill_depth(None) == np.inf
    assert cone._fill_depth(0) == 0.0


def _timed_sweep(offsets_s, t0="2026-01-01T00:00:00"):
    """A sweep dataset whose rays are ``offsets_s`` seconds after ``t0``."""
    t = np.datetime64(t0, "ns") + np.array(offsets_s) * np.timedelta64(1, "s")
    return xr.Dataset({"time": ("azimuth", t)}, coords={"azimuth": np.arange(t.size)})


def test_mean_time_of_a_sweep():
    t0 = np.datetime64("2026-01-01T00:00:00", "ns")
    sec = np.timedelta64(1, "s")
    assert cone._mean_time(_timed_sweep([0, 30, 60, 90])) == t0 + 45 * sec
    nat = np.datetime64("NaT", "ns")
    # NaT rays are left out
    holey = _timed_sweep([0, 60, 0])
    holey["time"][2] = nat
    assert cone._mean_time(holey) == t0 + 30 * sec
    # no time, or only NaT: the mean is NaT
    assert np.isnat(cone._mean_time(xr.Dataset({"x": ("azimuth", [1.0, 2.0])})))
    only_nat = _timed_sweep([0, 0])
    only_nat["time"][:] = nat
    assert np.isnat(cone._mean_time(only_nat))


def test_mean_time_of_a_volume():
    t0 = np.datetime64("2026-01-01T00:00:00", "ns")
    sec = np.timedelta64(1, "s")
    a, b = _timed_sweep([0, 20]), _timed_sweep([40, 100])
    assert cone._volume_time([a, b]) == t0 + 40 * sec
    # sweeps without a time are ignored; none at all, or only NaT, gives NaT
    bare = xr.Dataset({"x": ("azimuth", [1.0])})
    assert cone._volume_time([bare, a]) == t0 + 10 * sec
    assert np.isnat(cone._volume_time([bare]))
    only_nat = _timed_sweep([0, 0])
    only_nat["time"][:] = np.datetime64("NaT", "ns")
    assert np.isnat(cone._volume_time([only_nat, bare]))
