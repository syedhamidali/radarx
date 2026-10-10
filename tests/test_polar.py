#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for the helpers shared by the polar-volume modules."""
import numpy as np
import xarray as xr

from radarx import _polar


def test_nearest_ray_wraps_and_is_unsorted():
    az = np.array([90.0, 359.0, 1.0, 180.0])
    idx, dist = _polar.nearest_ray(az, [0.2, 359.6, 91.0, 135.0, 365.0], distance=True)
    assert list(idx[:3]) == [2, 1, 0]
    np.testing.assert_allclose(dist[:3], [0.8, 0.6, 1.0])
    assert idx[3] == 3 and np.isclose(dist[3], 45.0)  # tie: the larger azimuth
    assert idx[4] == 2  # 365 deg is 5 deg
    assert _polar.nearest_ray(az, [0.2]).shape == (1,)


def test_nearest_ray_of_regular_scan():
    az = (np.arange(360) + 0.5) * 1.0
    target = np.array([0.0, 10.4, 10.6, 359.9])
    idx = _polar.nearest_ray(az, target)
    assert list(idx) == [0, 10, 10, 359]


def test_beam_width_lookup():
    root = xr.Dataset(coords={"latitude": 1.0})
    tree = xr.DataTree.from_dict({"/": root})
    assert _polar.beam_width(tree) == 1.0
    assert _polar.beam_width(tree, default=0.95) == 0.95
    params = xr.Dataset({"radar_beam_width_h": 0.9})
    tree = xr.DataTree.from_dict({"/": root, "radar_parameters": params})
    assert _polar.beam_width(tree) == 0.9
    params = xr.Dataset({"radar_beam_width_v": 1.1, "radar_beam_width_h": 0.9})
    tree = xr.DataTree.from_dict({"/": root, "radar_parameters": params})
    assert _polar.beam_width(tree) == 1.1
    tree = xr.DataTree.from_dict({"/": xr.Dataset({"radar_beam_width_v": 0.8})})
    assert _polar.beam_width(tree) == 0.8
