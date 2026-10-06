#!/usr/bin/env python
# Copyright (c) 2024-2025, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for Radarx Accessors
===========================
"""

import numpy as np
import pytest
import xarray as xr
import xradar as xd
from open_radar_data import DATASETS

import radarx  # noqa: F401


@pytest.fixture
def mock_dtree():
    """Fixture to create a mock radar DataTree."""
    file = DATASETS.fetch("swx_20120520_0641.nc")
    dtree = xd.io.open_cfradial1_datatree(file, sweep=[0, 1, 2, 3, 5, 7])
    return dtree


def test_to_grid_accessor(mock_dtree):
    """Test the `to_grid` accessor from `RadarxDataTreeAccessor`."""
    gridded_ds = mock_dtree.radarx.to_grid(
        data_vars=["corrected_reflectivity_horizontal"],
        pseudo_cappi=True,
        x_lim=(-50e3, 50e3),
        y_lim=(-50e3, 50e3),
        z_lim=(0, 5e3),
        x_step=2000,
        y_step=2000,
        z_step=1000,
    )

    # Assertions
    assert isinstance(
        gridded_ds, xr.Dataset
    ), "Returned object is not an xarray Dataset"
    assert (
        "corrected_reflectivity_horizontal" in gridded_ds.data_vars
    ), "'corrected_reflectivity_horizontal' variable is missing in the gridded dataset"
    assert "lon" in gridded_ds.coords, "'lon' coordinate is missing"
    assert "lat" in gridded_ds.coords, "'lat' coordinate is missing"
    assert "z" in gridded_ds.coords, "'z' coordinate is missing"
    assert gridded_ds["corrected_reflectivity_horizontal"].shape == (
        6,
        51,
        51,
    ), "Gridded dataset shape mismatch"


def test_plot_maxcappi_accessor(mock_dtree, tmp_path):
    """Test the `plot_max_cappi` method from `RadarxDataSetAccessor`."""
    # Grid the radar data
    gridded_ds = mock_dtree.radarx.to_grid(
        data_vars=["corrected_reflectivity_horizontal"],
        pseudo_cappi=True,
        x_lim=(-50e3, 50e3),
        y_lim=(-50e3, 50e3),
        z_lim=(0, 5e3),
        x_step=2000,
        y_step=2000,
        z_step=1000,
    )

    # Create a valid directory for saving the plot
    save_dir = tmp_path / "plots"
    save_dir.mkdir()

    # Plot Max-CAPPI using the accessor
    ax = gridded_ds.radarx.plot_max_cappi(
        data_var="corrected_reflectivity_horizontal",
        cmap="viridis",
        vmin=0,
        vmax=60,
        title="Test Max-CAPPI",
        add_map=False,
        colorbar=True,
        show_figure=False,
        savedir=str(save_dir),
    )

    assert hasattr(ax, "figure")

    # Dynamically construct the expected file name
    radar_name = gridded_ds.attrs.get("instrument_name", "Radar")
    time_str = gridded_ds["time"].dt.strftime("%Y%m%d%H%M%S").values.item()
    expected_file = save_dir / f"Test Max-CAPPI_{radar_name}_{time_str}.png"

    # Assert the plot was saved
    assert expected_file.exists(), f"Expected file {expected_file} was not created."


def test_to_grid_keeps_time_as_coordinate(mock_dtree):
    """Grids carry ``time`` as a scalar coordinate and stack along it."""
    kw = dict(
        data_vars=["corrected_reflectivity_horizontal"],
        x_lim=(-20e3, 20e3),
        y_lim=(-20e3, 20e3),
        z_lim=(0, 2e3),
        x_step=4000,
        y_step=4000,
        z_step=1000,
    )
    grid = mock_dtree.radarx.to_grid(**kw)
    assert "time" in grid.coords and "time" not in grid.data_vars
    assert grid["time"].ndim == 0
    later = grid.assign_coords(time=grid.time + np.timedelta64(5, "m"))
    stacked = xr.concat([grid, later], "time")
    assert stacked["corrected_reflectivity_horizontal"].dims[0] == "time"
    assert stacked.sizes["time"] == 2
    assert "time" not in stacked["lat"].dims  # grid coordinates are not stacked


def test_assign_products_real_volume(mock_dtree):
    """Products of a real volume merge into the matching sweeps."""
    dtree = mock_dtree
    products = dtree.radarx.llsd("mean_doppler_velocity")
    assert set(products.children) == set(dtree.children)  # root kept
    xr.testing.assert_identical(products.to_dataset(), dtree.to_dataset())
    merged = dtree.radarx.assign(products)
    for name in products.children:
        own = merged[name].to_dataset(inherit=False)
        before = dtree[name].to_dataset(inherit=False)
        assert set(own.data_vars) == set(before.data_vars) | {
            "azimuthal_shear",
            "radial_divergence",
        }
        assert set(own.coords) == set(before.coords)  # no duplicated coords
        xr.testing.assert_identical(
            own["azimuthal_shear"], products[name]["azimuthal_shear"]
        )
    kdp = dtree.radarx.kdp(phidp="diff_phase", rhohv="copol_coeff")
    merged = merged.radarx.assign(kdp)
    assert {"KDP", "PHIDP_processed", "azimuthal_shear"} <= set(
        merged["sweep_0"].data_vars
    )
    # a sweep
    ds = dtree["sweep_0"].to_dataset()
    out = ds.radarx.assign(ds.radarx.azimuthal_shear("mean_doppler_velocity"))
    assert "azimuthal_shear" in out.data_vars
    assert "azimuthal_shear" not in ds.data_vars


def test_assign_aligns_and_checks():
    """Partial products are aligned (NaN elsewhere); bad input raises."""
    az = np.arange(0.5, 360.0, 1.0)
    rng = np.arange(10) * 100.0
    ds = xr.Dataset(
        {"DBZH": (("azimuth", "range"), np.ones((az.size, rng.size)))},
        coords={"azimuth": az, "range": rng, "elevation": ("azimuth", az * 0 + 1)},
    )
    part = (2 * ds["DBZH"].isel(range=slice(0, 5))).rename("DBZH_x2")
    out = ds.radarx.assign(part)
    assert out["DBZH_x2"].shape == ds["DBZH"].shape
    assert np.isnan(out["DBZH_x2"].isel(range=slice(5, None))).all()
    assert (out["DBZH_x2"].isel(range=slice(0, 5)) == 2).all()
    # a product of the same name replaces the variable
    replaced = ds.radarx.assign((ds["DBZH"] + 1).to_dataset())
    assert (replaced["DBZH"] == 2).all()
    with pytest.raises(ValueError, match="name"):
        ds.radarx.assign(xr.DataArray(np.zeros(az.size), dims="azimuth"))
    with pytest.raises(TypeError):
        ds.radarx.assign({"DBZH_x2": part})
    tree = xr.DataTree.from_dict({"/": xr.Dataset(), "sweep_0": ds})
    with pytest.raises(TypeError):
        ds.radarx.assign(tree)
    with pytest.raises(TypeError):
        tree.radarx.assign(part)
    other = xr.DataTree.from_dict({"/": xr.Dataset(), "sweep_1": part.to_dataset()})
    with pytest.raises(KeyError, match="sweep_1"):
        tree.radarx.assign(other)
    # nodes without products (and the root) are skipped
    empty = xr.DataTree.from_dict({"/": xr.Dataset({"a": 1}), "sweep_0": None})
    xr.testing.assert_identical(tree.radarx.assign(empty), tree)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
