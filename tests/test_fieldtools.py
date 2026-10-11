#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the helpers shared by the noise and calibration modules
==================================================================

Field lookup, the choice of the engine, the per-sweep arguments of a volume and
the geometry checks of a field.
"""

import numpy as np
import pytest
import xarray as xr

from radarx.retrieve import _fieldtools as ft


def sweep(dims=("azimuth", "range"), shape=(4, 6), **coords):
    return xr.DataArray(
        np.zeros(shape),
        dims=dims,
        coords={"range": 500.0 + 250.0 * np.arange(shape[-1]), **coords},
    )


def test_find_field_by_name_candidate_and_missing():
    ds = xr.Dataset({"DBZ": ("x", [1.0]), "OTHER": ("x", [2.0])})
    assert ft.find_field(ds, "OTHER", ft.DBZ_NAMES, "reflectivity") == "OTHER"
    assert ft.find_field(ds, None, ft.DBZ_NAMES, "reflectivity") == "DBZ"
    with pytest.raises(ft.MissingField, match="'NOPE' is not in the dataset"):
        ft.find_field(ds, "NOPE", ft.DBZ_NAMES, "reflectivity")
    assert ft.find_field(ds, None, ft.ZDR_NAMES, "ZDR", required=False) is None
    with pytest.raises(ft.MissingField):
        ft.find_field(ds, None, ft.ZDR_NAMES, "ZDR")


def test_primary_field_prefers_a_given_dataarray():
    ds = xr.Dataset({"DBZH": ("x", [1.0, 2.0])})
    given = xr.DataArray([5.0], dims="y")
    assert ft.primary_field(ds, given, ft.DBZ_NAMES, "reflectivity") is given
    assert ft.primary_field(ds, "DBZH", ft.DBZ_NAMES, "reflectivity").dims == ("x",)
    da = ds.DBZH
    assert ft.primary_field(da, None, ft.DBZ_NAMES, "reflectivity") is da


def test_choose_engine():
    assert ft.choose_engine("auto", True, "x") is True
    assert ft.choose_engine("auto", False, "x") is False
    assert ft.choose_engine("numpy", True, "x") is False
    assert ft.choose_engine("compiled", True, "x") is True
    with pytest.raises(ImportError, match="the compiled foo kernel is not available"):
        ft.choose_engine("compiled", False, "foo")
    with pytest.raises(ValueError, match="engine must be"):
        ft.choose_engine("gpu", True, "x")


def test_node_arg_takes_the_entry_of_a_sweep():
    tree = xr.DataTree.from_dict(
        {"/": xr.Dataset(), "sweep_0": xr.Dataset({"a": ("x", [1.0])})}
    )
    assert ft.node_arg(3.0, "sweep_0") == 3.0
    assert "a" in ft.node_arg(tree, "sweep_0")
    plain = xr.Dataset({"a": ("x", [2.0])})
    assert ft.node_arg({"sweep_0": plain}, "sweep_0") is plain
    with pytest.raises(ft.MissingField, match="no entry for 'sweep_1'"):
        ft.node_arg(tree, "sweep_1")
    with pytest.raises(ft.MissingField, match="no entry for 'sweep_1'"):
        ft.node_arg({"sweep_0": plain}, "sweep_1")


def test_ray_dimension_names_and_errors():
    assert ft.ray_dimension(sweep()) == "azimuth"
    assert ft.ray_dimension(sweep(dims=("time", "range"))) == "time"
    assert ft.ray_dimension(sweep(dims=("ray", "range"))) == "ray"
    both = sweep(dims=("sweep", "time", "range"), shape=(2, 3, 5))
    assert ft.ray_dimension(both) == "time"
    with pytest.raises(ValueError, match="needs a 'range' dimension"):
        ft.ray_dimension(xr.DataArray(np.zeros((2, 3)), dims=("azimuth", "gate")))
    with pytest.raises(ValueError, match="needs a ray dimension besides 'range'"):
        ft.ray_dimension(xr.DataArray(np.zeros(4), dims=("range",)))


def test_range_meters_needs_the_coordinate():
    r = ft.range_meters(sweep())
    assert r.dtype == np.float64
    np.testing.assert_array_equal(r.values, 500.0 + 250.0 * np.arange(6))
    bare = xr.DataArray(np.zeros((2, 3)), dims=("azimuth", "range"))
    with pytest.raises(ValueError, match="has no 'range' coordinate"):
        ft.range_meters(bare)
