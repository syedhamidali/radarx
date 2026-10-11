#!/usr/bin/env python
# Copyright (c) 2024-2025, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

import numpy as np
import pytest
import xarray as xr
import xradar as xd
from open_radar_data import DATASETS

from radarx.grid import grid_radar, make_3d_grid, stack_data
from radarx.grid.grid import _sweep_dataset


@pytest.fixture
def mock_dtree():
    """Fixture to create a mock radar DataTree."""
    file = DATASETS.fetch("KLBB20160601_150025_V06")
    dtree = xd.io.open_nexradlevel2_datatree(file, sweep=[0, 1])
    return dtree


def test_stack_data(mock_dtree):
    """Test `stack_data` function."""
    stacked_ds = stack_data(mock_dtree, data_vars=["DBZH"], geo=False)
    # Assertions
    assert isinstance(
        stacked_ds, xr.Dataset
    ), "Returned object is not an xarray Dataset"
    assert (
        "xyz" in stacked_ds.coords
    ), "'xyz' coordinate is missing in the stacked dataset"
    assert (
        "DBZH" in stacked_ds.data_vars
    ), "'DBZH' variable is missing in the stacked dataset"
    assert (
        stacked_ds["xyz"].shape[1] == 3
    ), "'xyz' coordinate does not have three dimensions"


def test_make_3d_grid(mock_dtree):
    """Test `make_3d_grid` function."""
    ds = _sweep_dataset(mock_dtree, "sweep_0")
    lat, lon, x, y, z, trg_crs = make_3d_grid(ds)

    # Assertions
    assert isinstance(lat, np.ndarray), "Latitude is not a numpy array"
    assert isinstance(lon, np.ndarray), "Longitude is not a numpy array"
    assert isinstance(x, np.ndarray), "x-coordinates are not a numpy array"
    assert isinstance(y, np.ndarray), "y-coordinates are not a numpy array"
    assert isinstance(z, np.ndarray), "z-coordinates are not a numpy array"
    assert trg_crs is not None, "Target CRS is missing"
    assert lat.shape == (401,), "Latitude grid shape mismatch"
    assert lon.shape == (401,), "Longitude grid shape mismatch"
    assert lon.shape == x.shape, "Shapes should be equal"


def test_make_3d_grid_axes_are_independent(mock_dtree):
    """lon runs along y = 0 and lat along x = 0, so unequal axes work (#169)."""
    import pyproj

    ds = _sweep_dataset(mock_dtree, "sweep_0")
    lat, lon, x, y, z, _ = make_3d_grid(ds, x_lim=(-100e3, 100e3), y_lim=(-50e3, 50e3))
    assert lon.shape == x.shape and lat.shape == y.shape
    assert lon.shape != lat.shape
    site = (float(ds["latitude"]), float(ds["longitude"]))
    aeqd = pyproj.Proj(proj="aeqd", lat_0=site[0], lon_0=site[1], datum="WGS84")
    lon0, _ = aeqd(x, np.zeros_like(x), inverse=True)
    _, lat0 = aeqd(np.zeros_like(y), y, inverse=True)
    assert np.abs(lon - lon0).max() * 111.3e3 < 5.0
    assert np.abs(lat - lat0).max() * 111.3e3 < 5.0


def test_grid_radar_barnes_cells_are_where_they_are_labelled(mock_dtree):
    """Barnes runs on the metric grid; lat/lon are those of every cell (#225)."""
    import pyproj

    pytest.importorskip("fastbarnes")
    gdt = mock_dtree.xradar.georeference()
    for name in ("sweep_0", "sweep_1"):
        node = gdt[name]
        node["DBZH"] = node["x"].astype("float64").copy()  # the field is x [m]
        node["DBZH"].attrs = {"units": "dBZ"}
    out = grid_radar(
        gdt,
        data_vars=["DBZH"],
        pseudo_cappi=False,
        x_lim=(-60e3, 60e3),
        y_lim=(-60e3, 60e3),
        z_lim=(1250, 3250),
        x_step=2000,
        y_step=2000,
        z_step=500,
        method="barnes",
    )
    x = out["x"].values[None, None, :] * np.ones(out["DBZH"].shape)
    err = np.abs(out["DBZH"].values - x)
    # with the regular lon/lat lattice of #225 the median error was 380 m
    assert np.nanmedian(err) < 50.0
    assert out["lat"].dims == out["lon"].dims == ("y", "x")
    site = _sweep_dataset(gdt, "sweep_0")
    aeqd = pyproj.Proj(
        proj="aeqd",
        lat_0=float(site["latitude"]),
        lon_0=float(site["longitude"]),
        datum="WGS84",
    )
    X, Y = np.meshgrid(out["x"].values, out["y"].values)
    lon, lat = aeqd(X, Y, inverse=True)
    np.testing.assert_allclose(out["lon"].values, lon, atol=1e-6)
    np.testing.assert_allclose(out["lat"].values, lat, atol=1e-6)


@pytest.mark.parametrize("pseudo_cappi", [True, False])
def test_grid_radar_barnes(mock_dtree, pseudo_cappi):
    """The Barnes method stays available next to the default cone method."""
    pytest.importorskip("fastbarnes")
    gridded_ds = grid_radar(
        mock_dtree,
        data_vars=["DBZH"],
        pseudo_cappi=pseudo_cappi,
        x_lim=(-50e3, 50e3),
        y_lim=(-50e3, 50e3),
        z_lim=(0, 5e3),
        x_step=2000,
        y_step=2000,
        z_step=1000,
        method="barnes",
    )
    assert gridded_ds["DBZH"].shape == (6, 51, 51)
    assert np.isfinite(gridded_ds["DBZH"].values).any()
    assert "gridding_method" not in gridded_ds.attrs


def test_grid_radar_barnes_missing(mock_dtree, monkeypatch):
    """method="barnes" raises a clear ImportError without fast-barnes-py."""
    import sys

    monkeypatch.setitem(sys.modules, "fastbarnes", None)
    monkeypatch.setitem(sys.modules, "fastbarnes.interpolation", None)
    with pytest.raises(ImportError, match="fast-barnes-py"):
        grid_radar(mock_dtree, data_vars=["DBZH"], method="barnes")


def test_grid_radar(mock_dtree):
    """Test `grid_radar` function."""
    gridded_ds = grid_radar(
        mock_dtree,
        data_vars=["DBZH"],
        pseudo_cappi=True,
        x_lim=(-50e3, 50e3),
        y_lim=(-50e3, 50e3),
        z_lim=(1250, 6250),  # KLBB is at 1029 m: levels below it are not filled
        x_step=2000,
        y_step=2000,
        z_step=1000,
    )
    # Assertions
    assert isinstance(
        gridded_ds, xr.Dataset
    ), "Returned object is not an xarray Dataset"
    assert (
        "DBZH" in gridded_ds.data_vars
    ), "'DBZH' variable is missing in the gridded dataset"
    assert "lon" in gridded_ds.coords, "'lon' coordinate is missing"
    assert "lat" in gridded_ds.coords, "'lat' coordinate is missing"
    assert "z" in gridded_ds.coords, "'z' coordinate is missing"
    assert gridded_ds["DBZH"].shape == (6, 51, 51), "Gridded dataset shape mismatch"
    assert np.isfinite(
        gridded_ds["DBZH"].values
    ).any(), "No finite values in gridded data"


if __name__ == "__main__":
    pytest.main()


def test_sweep_dataset_falls_back_for_older_xarray(mock_dtree, monkeypatch):
    """Without inherit="all_coords" support, plain to_dataset() is used."""
    node = mock_dtree["sweep_0"]
    original = type(node).to_dataset

    def old_to_dataset(self, inherit=True):
        if inherit == "all_coords":
            raise TypeError("unsupported")
        return original(self, inherit=inherit)

    monkeypatch.setattr(type(node), "to_dataset", old_to_dataset)
    ds = _sweep_dataset(mock_dtree, "sweep_0")
    assert isinstance(ds, xr.Dataset)
    assert "DBZH" in ds


def test_grid_radar_passes_fill_depth_and_k_e(mock_dtree):
    kw = dict(
        data_vars=["DBZH"],
        x_lim=(-50e3, 50e3),
        y_lim=(-50e3, 50e3),
        z_lim=(0, 5e3),
        x_step=2000,
        y_step=2000,
        z_step=1000,
    )
    # the lowest beam is above 1029 m; levels at the radar altitude and below
    # are only filled with fill_depth=inf (the earlier behaviour)
    limited = grid_radar(mock_dtree, **kw)
    old = grid_radar(mock_dtree, fill_depth=np.inf, **kw)
    assert np.isfinite(limited["DBZH"].values[:2]).sum() == 0
    assert np.isfinite(old["DBZH"].values[:2]).sum() > 0
    assert limited.attrs["k_e"] == pytest.approx(4 / 3)
    assert grid_radar(mock_dtree, k_e=1.2, **kw).attrs["k_e"] == pytest.approx(1.2)


def test_grid_radar_barnes_arguments_and_layout(mock_dtree, monkeypatch):
    """What the Barnes branch hands to fast-barnes-py, without the package.

    A stand-in for ``fastbarnes`` records the call and returns the mean of the
    gate values on every cell, so the lattice (origin, steps, size), the
    smoothing lengths, the gates passed and the layout of the output are
    checked independently of the interpolation itself.
    """
    import sys
    import types

    calls = []

    def barnes(xyz, values, sigma, x0, steps, size, max_dist):
        calls.append((xyz, values, sigma, x0, steps, size, max_dist))
        return np.full(tuple(size)[::-1], values.mean())

    interp = types.ModuleType("fastbarnes.interpolation")
    interp.barnes = barnes
    interp.get_half_kernel_size = lambda sigma, steps, num_iter: 1
    package = types.ModuleType("fastbarnes")
    package.interpolation = interp
    monkeypatch.setitem(sys.modules, "fastbarnes", package)
    monkeypatch.setitem(sys.modules, "fastbarnes.interpolation", interp)

    out = grid_radar(
        mock_dtree,
        data_vars=["DBZH"],
        pseudo_cappi=False,
        x_lim=(-20e3, 20e3),
        y_lim=(-10e3, 10e3),
        z_lim=(1000, 3000),
        x_step=2000,
        y_step=2000,
        z_step=500,
        x_smth=0.5,
        y_smth=0.25,
        z_smth=2,
        method="barnes",
    )
    ((xyz, values, sigma, x0, steps, size, max_dist),) = calls
    assert size == (21, 11, 5) and max_dist == 4
    np.testing.assert_allclose(steps, [2000.0, 2000.0, 500.0])
    np.testing.assert_allclose(x0, [-20e3, -10e3, 1000.0])
    np.testing.assert_allclose(sigma, [0.5 * 2000.0, 0.25 * 2000.0, 2 * 500.0])
    # the gates of the two sweeps with a value, in metres from the radar
    stacked = stack_data(mock_dtree, data_vars=["DBZH"], geo=False)
    n_valid = int(np.isfinite(stacked["DBZH"].values).sum())
    assert values.size == n_valid and xyz.shape == (n_valid, 3)
    assert np.isfinite(values).all()
    np.testing.assert_allclose(out["x"], np.arange(-20e3, 20e3 + 1, 2000.0))
    np.testing.assert_allclose(out["y"], np.arange(-10e3, 10e3 + 1, 2000.0))
    np.testing.assert_allclose(out["z"], np.arange(1000.0, 3000.1, 500.0))
    assert out["DBZH"].dims == ("z", "y", "x") and out["DBZH"].shape == (5, 11, 21)
    np.testing.assert_allclose(out["DBZH"], values.mean())
    assert out["lat"].dims == out["lon"].dims == ("y", "x")
    assert out["lat"].attrs["units"] == "degrees_north"
    assert out["lon"].attrs["units"] == "degrees_east"
    assert out.attrs["radar_name"] == out.attrs.get("instrument_name", "")
    assert out["time"].ndim == 0
