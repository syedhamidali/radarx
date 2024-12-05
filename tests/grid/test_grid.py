#!/usr/bin/env python
# Copyright (c) 2024-2025, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

import pytest
import xarray as xr
import numpy as np
import xradar as xd
from radarx.grid import stack_data, make_3d_grid, grid_radar
from open_radar_data import DATASETS


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
    ds = mock_dtree["sweep_0"].to_dataset()
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


def test_grid_radar(mock_dtree):
    """Test `grid_radar` function."""
    gridded_ds = grid_radar(
        mock_dtree,
        data_vars=["DBZH"],
        pseudo_cappi=True,
        x_lim=(-50e3, 50e3),
        y_lim=(-50e3, 50e3),
        z_lim=(0, 5e3),
        x_step=2000,
        y_step=2000,
        z_step=1000,
    )
    print(gridded_ds["DBZH"].shape)
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
