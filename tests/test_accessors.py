#!/usr/bin/env python
# Copyright (c) 2024-2025, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for Radarx Accessors
===========================
"""

import pytest
import xarray as xr
import xradar as xd
from open_radar_data import DATASETS


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
    gridded_ds.radarx.plot_max_cappi(
        data_var="corrected_reflectivity_horizontal",
        cmap="viridis",
        vmin=0,
        vmax=60,
        title="Test Max-CAPPI",
        add_map=True,
        colorbar=True,
        show_figure=False,
        savedir=str(save_dir),
    )

    # Dynamically construct the expected file name
    radar_name = gridded_ds.attrs.get("instrument_name", "Radar")
    time_str = gridded_ds["time"].dt.strftime("%Y%m%d%H%M%S").values.item()
    expected_file = save_dir / f"Test Max-CAPPI_{radar_name}_{time_str}.png"

    # Assert the plot was saved
    assert expected_file.exists(), f"Expected file {expected_file} was not created."


if __name__ == "__main__":
    pytest.main(["-s", __file__])
