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
    field="height", elevations=ELEVATIONS, azimuth=None, nan_fraction=0.0, seed=0
):
    """Synthetic PPI volume; ``field="height"`` stores each gate's beam height."""
    rng_m = np.arange(125.0, 60_000.0, 250.0)
    azimuth = np.arange(0.5, 360.0, 1.0) if azimuth is None else azimuth
    rnd = np.random.default_rng(seed)
    sweeps = {}
    for k, el in enumerate(elevations):
        # rays start mid-rotation and wobble a little in elevation, like real scans
        az = np.roll(azimuth, 37 * (k + 1))
        ray_el = el + 0.05 * np.sin(np.deg2rad(az))
        _, _, gz = antenna_to_cartesian(
            rng_m[None, :], az[:, None], ray_el[:, None], site_altitude=ALT
        )
        if field == "height":
            data = gz.copy()
        else:
            data = rnd.uniform(-10, 60, gz.shape)
        if nan_fraction:
            data[rnd.random(data.shape) < nan_fraction] = np.nan
        sweeps[f"sweep_{k}"] = xr.Dataset(
            {"DBZH": (("azimuth", "range"), data, {"units": "dBZ"})},
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
    assert out.lat.dims == ("y",) and out.lon.dims == ("x",)
    np.testing.assert_allclose(float(out.lat.sel(y=0.0)), 45.0)
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
