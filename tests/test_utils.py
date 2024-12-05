#!/usr/bin/env python
# Copyright (c) 2024-2025, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for Radarx Utils
======================
"""
import pytest
import xarray as xr
import numpy as np
import xradar as xd
from xarray import DataTree
from radarx.utils import (
    get_geocoords,
    cartesian_to_geographic_aeqd,
    find_multidim_vars,
    combine_nexrad_sweeps,
)


def test_get_geocoords():
    """
    Test `get_geocoords` function with CRS-based mock radar data.
    """
    radar = xd.model.create_sweep_dataset()
    tree = xr.DataTree.from_dict({"sweep_0": radar})
    tree = tree.xradar.georeference()
    ds = tree["sweep_0"].to_dataset()
    ds = ds.pipe(get_geocoords)

    # Assertions
    assert "lon" in ds.coords, "Longitude coordinate is missing"
    assert "lat" in ds.coords, "Latitude coordinate is missing"
    assert "alt" in ds.coords, "Altitude coordinate is missing"
    assert ds.lon.shape == ds.x.shape, "Longitude shape mismatch"
    assert ds.lat.shape == ds.y.shape, "Latitude shape mismatch"


def test_cartesian_to_geographic_aeqd():
    """Test `cartesian_to_geographic_aeqd` function."""
    x = np.array([0, 1000, -1000])
    y = np.array([0, 1000, -1000])
    lon_0, lat_0 = -90.0, 30.0
    earth_radius = 6371000

    lon, lat = cartesian_to_geographic_aeqd(x, y, lon_0, lat_0, earth_radius)

    assert len(lon) == len(x)
    assert len(lat) == len(y)
    assert np.isclose(lon[0], lon_0)
    assert np.isclose(lat[0], lat_0)


def test_find_multidim_vars():
    """Test `find_multidim_vars` function."""
    ds = xr.Dataset(
        {
            "var1": (("x", "y"), np.random.rand(10, 10)),
            "var2": (("x",), np.random.rand(10)),
            "var3": (("y",), np.random.rand(10)),
        }
    )
    multidim_vars = find_multidim_vars(ds)

    assert "var1" in multidim_vars
    assert "var2" not in multidim_vars
    assert "var3" not in multidim_vars


def test_combine_nexrad_sweeps():
    """Test `combine_nexrad_sweeps` function."""
    # Create a mock DataTree with two sweeps
    ds1 = xr.Dataset(
        {"DBZH": (("range"), np.random.rand(100))},
        coords={
            "sweep_fixed_angle": 0.5,
            "range": ("range", np.arange(100)),  # Add range coordinate
        },
    )
    ds2 = xr.Dataset(
        {"DBZH": (("range"), np.random.rand(50))},
        coords={
            "sweep_fixed_angle": 0.5,
            "range": ("range", np.arange(50)),  # Add range coordinate
        },
    )
    dtree = DataTree.from_dict(
        {
            "/sweep_0": DataTree(ds1),
            "/sweep_1": DataTree(ds2),
        }
    )
    combined = combine_nexrad_sweeps(dtree)
    assert "/sweep_0" in combined.groups
    assert "/sweep_1" not in combined.groups
    assert "DBZH" in combined["/sweep_0"].to_dataset()
    assert "DBZH_SHORT" in combined["/sweep_0"].to_dataset()
    assert combined["/sweep_0"].to_dataset()["range"].size == 100


if __name__ == "__main__":
    pytest.main()
