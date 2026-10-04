#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the radarx UGRID / uxarray conversion
===============================================
"""
import numpy as np
import pytest
import xarray as xr

try:
    import uxarray as ux  # noqa: F401
except Exception as err:  # some uxarray/numba combinations fail at import
    pytest.skip(f"uxarray not usable: {err}", allow_module_level=True)

import radarx  # noqa: E402,F401
from radarx.grid import ugrid  # noqa: E402

EARTH_RADIUS = 6371008.8


def _sweep(azimuth, n_range=20, gate=250.0, first=125.0):
    rng = first + gate * np.arange(n_range)
    data = np.arange(azimuth.size * n_range, dtype=float).reshape(azimuth.size, -1)
    return xr.Dataset(
        {
            "DBZH": (("azimuth", "range"), data, {"units": "dBZ"}),
            "VRADH": (("azimuth", "range"), -data),
        },
        coords={
            "azimuth": azimuth,
            "range": rng,
            "elevation": ("azimuth", np.full(azimuth.size, 0.5)),
            "latitude": 45.0,
            "longitude": 10.0,
            "altitude": 100.0,
        },
    )


@pytest.fixture
def full_sweep():
    # unsorted rays, as delivered by a radar starting mid-rotation
    return _sweep(np.roll(np.arange(0.5, 360, 1.0), 90))


@pytest.fixture
def sector_sweep():
    return _sweep(np.arange(10.5, 40, 1.0))


def test_edges():
    np.testing.assert_allclose(ugrid._edges([1.0, 2.0, 4.0]), [0.5, 1.5, 3.0, 5.0])
    np.testing.assert_allclose(
        ugrid._edges([0.5, 120.5, 240.5], full_circle=True), [300.5, 60.5, 180.5]
    )


def test_is_full_circle(full_sweep, sector_sweep):
    assert ugrid._is_full_circle(full_sweep.azimuth)
    assert not ugrid._is_full_circle(sector_sweep.azimuth)
    assert not ugrid._is_full_circle([0.0, 1.0])


def test_full_sweep_topology_and_data(full_sweep):
    uxds = full_sweep.radarx.to_uxarray()
    grid = uxds.uxgrid
    n_az, n_r = full_sweep.sizes["azimuth"], full_sweep.sizes["range"]
    assert grid.n_face == n_az * n_r
    assert grid.n_node == n_az * (n_r + 1)  # closed circle, no duplicate edge
    assert set(uxds.data_vars) == {"DBZH", "VRADH"}
    assert uxds["DBZH"].attrs["units"] == "dBZ"
    # faces are ordered by azimuth, then range
    order = np.argsort(full_sweep.azimuth.values)
    np.testing.assert_array_equal(
        uxds["DBZH"].values, full_sweep.DBZH.values[order].ravel()
    )
    np.testing.assert_array_equal(
        uxds["azimuth"].values[::n_r], full_sweep.azimuth.values[order]
    )


def test_face_areas_match_gate_geometry(full_sweep):
    grid = full_sweep.radarx.to_uxarray("DBZH").uxgrid
    area = grid.face_areas.values.reshape(360, -1) * EARTH_RADIUS**2
    expected = 250.0 * full_sweep.range.values * np.deg2rad(1.0)
    np.testing.assert_allclose(area.mean(axis=0), expected, rtol=5e-3)
    assert (area > 0).all()  # counter-clockwise faces have positive area


def test_sector_sweep_and_dataarray(sector_sweep):
    uxds = sector_sweep["DBZH"].radarx.to_uxarray()
    grid = uxds.uxgrid
    n_az, n_r = sector_sweep.sizes["azimuth"], sector_sweep.sizes["range"]
    assert grid.n_face == n_az * n_r
    assert grid.n_node == (n_az + 1) * (n_r + 1)  # open sector keeps both ends
    assert list(uxds.data_vars) == ["DBZH"]


def test_errors(full_sweep, monkeypatch):
    import sys

    with pytest.raises(ValueError, match="latitude"):
        full_sweep.drop_vars(["latitude", "longitude"]).radarx.to_uxarray()
    with pytest.raises(ValueError, match="PPI sweep"):
        full_sweep.isel(range=0).radarx.to_uxarray()
    with pytest.raises(ValueError, match="No \\(azimuth, range\\)"):
        full_sweep.drop_vars(["DBZH", "VRADH"]).radarx.to_uxarray()
    monkeypatch.setitem(sys.modules, "uxarray", None)
    with pytest.raises(ImportError, match="pip install uxarray"):
        full_sweep.radarx.to_uxarray()
