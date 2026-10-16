#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the ZDR arc and KDP-ZDR separation algorithm
======================================================

A synthetic supercell on a 500 m grid: a Gaussian reflectivity core at (30, 32) km, an arc
of high ZDR on its south (inflow) side and a patch of high ZDR on its north side, and a
KDP foot inside the core. The forward-flank downdraft vector points north (0 degrees).
The areas, centroids, separation vector and angle follow from the construction.
"""

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.retrieve import zdr_arc

arc_mod = __import__("radarx.retrieve.zdr_arc", fromlist=["x"])

X = np.arange(0.0, 60001.0, 500.0)
Y = np.arange(0.0, 60001.0, 500.0)
XX, YY = np.meshgrid(X, Y)
CELL = 0.25  # km2


def supercell(north_patch=True, low_rho=False):
    r2 = (XX - 30e3) ** 2 + (YY - 32e3) ** 2
    dbz = 60.0 * np.exp(-r2 / (2 * 9e3**2))
    zdr = np.full(XX.shape, 0.5)
    arc = (XX >= 26e3) & (XX <= 34e3) & (YY >= 20e3) & (YY <= 23e3)
    zdr[arc] = 4.0
    if north_patch:
        zdr[(XX >= 26e3) & (XX <= 34e3) & (YY >= 41e3) & (YY <= 43e3)] = 4.0
    rho = np.full(XX.shape, 0.98)
    if low_rho:
        rho[(XX <= 27e3) & arc] = 0.4
    kdp = np.where(np.hypot(XX - 30e3, YY - 29e3) <= 3e3, 2.5, 0.1)
    return xr.Dataset(
        {
            "DBZH": (("y", "x"), dbz),
            "ZDR": (("y", "x"), zdr),
            "RHOHV": (("y", "x"), rho),
            "KDP": (("y", "x"), kdp),
        },
        coords={"x": X, "y": Y},
    )


KW = dict(rhohv="RHOHV", kdp="KDP", ffd_direction=0.0, storm_motion=90.0)


def test_the_arc_and_the_separation_follow_from_the_construction():
    res = zdr_arc(supercell(), **KW)
    assert res.sizes["storm"] == 1 and res.sizes["candidate"] == 1
    s = res.isel(storm=0)
    assert float(s.storm_x) == pytest.approx(30e3, abs=1.0)
    assert float(s.storm_y) == pytest.approx(32e3, abs=1.0)
    n_arc = 17 * 7
    assert float(s.arc_area) == pytest.approx(n_arc * CELL)
    assert float(s.arc_mean_zdr) == pytest.approx(4.0)
    assert float(s.arc_top10_zdr) == pytest.approx(4.0)
    assert float(s.arc_x) == pytest.approx(30e3) and float(s.arc_y) == pytest.approx(
        21.5e3
    )
    foot_cells = int((np.hypot(XX - 30e3, YY - 29e3) <= 3e3).sum())
    assert float(s.foot_area) == pytest.approx(foot_cells * CELL)
    assert float(s.foot_x) == pytest.approx(30e3) and float(s.foot_y) == pytest.approx(
        29e3
    )
    assert float(s.separation_length) == pytest.approx(
        7.5
    )  # km, from the foot to the arc
    # the vector points south (180 degrees) and the storm moves east: 90 degrees
    assert float(s.separation_angle) == pytest.approx(90.0)
    # the maps hold the objects
    assert int((res.ZDR_ARC_STORM == 0).sum()) == n_arc
    assert int((res.KDP_FOOT_STORM == 0).sum()) == foot_cells
    assert int((res.ZDR_ARC_CANDIDATE > 0).sum()) == n_arc
    assert res.separation_angle.attrs["units"] == "degree"


@pytest.mark.parametrize(
    ("motion", "angle"),
    [(90.0, 90.0), (270.0, 90.0), (0.0, 180.0), (180.0, 0.0), (-180.0, 0.0)],
)
def test_separation_angle_is_the_angle_to_the_storm_motion(motion, angle):
    res = zdr_arc(supercell(), **{**KW, "storm_motion": motion})
    assert float(res.separation_angle[0]) == pytest.approx(angle)


def test_the_inflow_mask_removes_the_north_side():
    res = zdr_arc(supercell(north_patch=True), **KW)
    assert res.sizes["candidate"] == 1  # the north patch has a gradient pointing south
    # a looser limit lets it through, and with no classifier it joins the arc
    res = zdr_arc(supercell(north_patch=True), **{**KW, "max_gradient_angle": 180.0})
    assert res.sizes["candidate"] == 2 and bool(res.candidate_arc.all())
    assert float(res.arc_area[0]) == pytest.approx((17 * 7 + 17 * 5) * CELL)
    # the features have the order of Table 1 and a classifier can use them
    assert (
        arc_mod.FEATURES[9] == "angle_to_ffd"
        and arc_mod.FEATURES[1] == "storm_distance"
    )
    seen = {}

    def classifier(features):
        seen["shape"] = features.shape
        return features[:, 9] > 90.0  # the arc lies opposite to the FFD vector

    res = zdr_arc(
        supercell(), **{**KW, "max_gradient_angle": 180.0, "classifier": classifier}
    )
    assert seen["shape"] == (2, len(arc_mod.FEATURES))
    assert list(res.candidate_arc.values) == [True, False] or list(
        res.candidate_arc.values
    ) == [False, True]
    assert float(res.arc_area[0]) == pytest.approx(17 * 7 * CELL)

    # a model with predict is accepted
    class Model:
        def predict(self, features):
            return features[:, 9] > 90.0

    res = zdr_arc(
        supercell(), **{**KW, "max_gradient_angle": 180.0, "classifier": Model()}
    )
    assert float(res.arc_area[0]) == pytest.approx(17 * 7 * CELL)
    with pytest.raises(ValueError, match="one value per candidate"):
        zdr_arc(supercell(), **{**KW, "classifier": lambda f: np.ones(5)})


def test_the_candidate_features():
    res = zdr_arc(supercell(), **KW)
    c = res.isel(candidate=0)
    assert float(c.candidate_area) == pytest.approx(17 * 7 * CELL)
    assert (
        float(c.candidate_mean_zdr) == pytest.approx(4.0)
        and float(c.candidate_max_zdr) == 4.0
    )
    assert float(c.candidate_mean_rhohv) == pytest.approx(0.98)
    assert float(c.candidate_storm_distance) == pytest.approx(10.5)
    assert float(c.candidate_angle_to_ffd) == pytest.approx(180.0)
    assert float(c.candidate_storm_vector_y) == pytest.approx(-10.5)
    assert abs(float(c.candidate_mean_gradient_direction)) < 25.0  # toward the core
    assert (
        float(c.candidate_mean_gradient) > 0
        and 20.0 < float(c.candidate_mean_dbz) < 45.0
    )


def test_correlation_coefficient_threshold_and_zdr_threshold():
    res = zdr_arc(supercell(low_rho=True), **KW)
    assert float(res.arc_area[0]) == pytest.approx(
        (17 - 3) * 7 * CELL
    )  # x = 26, 26.5, 27 removed
    res = zdr_arc(supercell(low_rho=True), **{**KW, "min_rhohv": 0.3})
    assert float(res.arc_area[0]) == pytest.approx(17 * 7 * CELL)
    res = zdr_arc(supercell(), **{**KW, "zdr_threshold": 4.5})
    assert res.sizes["candidate"] == 0 and np.isnan(float(res.arc_area[0]))
    # without the correlation coefficient the field is not used
    ds = supercell(low_rho=True)
    res = zdr_arc(ds, kdp="KDP", ffd_direction=0.0, storm_motion=90.0)
    assert float(res.arc_area[0]) == pytest.approx(17 * 7 * CELL)


def test_without_kdp_or_motion():
    res = zdr_arc(supercell(), ffd_direction=0.0, storm_motion=90.0)
    assert np.isnan(float(res.foot_area[0])) and np.isnan(
        float(res.separation_length[0])
    )
    assert float(res.arc_area[0]) > 0
    res = zdr_arc(supercell(), rhohv="RHOHV", kdp="KDP", ffd_direction=0.0)
    assert float(res.separation_length[0]) == pytest.approx(7.5) and np.isnan(
        float(res.separation_angle[0])
    )


def test_directions_are_bearings():
    # the same storm seen with the FFD vector 360 degrees round, or pointing the other way
    a = zdr_arc(supercell(), **KW)
    b = zdr_arc(supercell(), **{**KW, "ffd_direction": 360.0})
    assert float(a.arc_area[0]) == float(b.arc_area[0])
    c = zdr_arc(supercell(), **{**KW, "ffd_direction": 180.0})
    assert c.sizes["candidate"] == 1  # now the north patch is on the inflow side
    assert float(c.candidate_y[0]) > 40e3


def test_storm_objects_and_the_distance_to_them():
    ds = supercell()
    res = zdr_arc(
        ds, **{**KW, "max_distance": 5.0}
    )  # the arc is 10.5 km from the storm
    assert res.sizes["candidate"] == 1 and int(res.candidate_storm[0]) == -1
    assert not bool(res.candidate_arc[0]) and np.isnan(float(res.arc_area[0]))
    weak = ds.assign(DBZH=ds.DBZH.clip(max=40.0))  # no 45 dBZ region: no storm
    res = zdr_arc(weak, **KW)
    assert res.sizes["storm"] == 0 and int(res.candidate_storm[0]) == -1
    # a region above 300 km2 is split into its 50 dBZ cores
    r2a = (XX - 22e3) ** 2 + (YY - 30e3) ** 2
    r2b = (XX - 36e3) ** 2 + (YY - 30e3) ** 2
    two = ds.assign(
        DBZH=(
            ("y", "x"),
            np.maximum(65 * np.exp(-r2a / 162e6), 65 * np.exp(-r2b / 162e6)),
        )
    )
    res = zdr_arc(two, **KW)
    assert res.sizes["storm"] == 2
    assert sorted(np.round(res.storm_x.values / 1e3)) == [22.0, 36.0]
    # a region of 13 km2 is not a storm object (the limit is 20 km2)
    small = ds.assign(
        DBZH=ds.DBZH.where(np.hypot(XX - 30e3, YY - 32e3) < 2e3, 30.0).clip(min=0)
    )
    assert zdr_arc(small, **KW).sizes["storm"] == 0
    # smoothing is accepted
    assert zdr_arc(ds, **{**KW, "smooth": 1.0}).sizes["storm"] == 1


def test_vertical_dimension_and_errors():
    ds = supercell()
    cube = xr.concat([ds, ds * 0.0 + 1.0], dim="z").assign_coords(z=[500.0, 1500.0])
    assert zdr_arc(cube, **KW).sizes["storm"] == 1  # the lowest level
    low = zdr_arc(cube, level=1400.0, **KW)  # the level near 1500 m has no storm
    assert low.sizes["storm"] == 0
    with pytest.raises(ValueError, match="dimensions y and x"):
        zdr_arc(ds.expand_dims(time=2), **KW)
    with pytest.raises(KeyError):
        zdr_arc(ds.drop_vars("ZDR"), **KW)
    with pytest.raises(KeyError):
        zdr_arc(ds, zdr="NOPE", **KW)
    with pytest.raises(TypeError):
        zdr_arc(ds.ZDR, ffd_direction=0.0)
    with pytest.raises(ValueError, match="coordinate"):
        zdr_arc(ds.drop_vars("x"), **KW)


def test_input_is_not_changed_provenance_and_accessor():
    ds = supercell()
    before = ds.copy(deep=True)
    res = ds.radarx.zdr_arc(**KW)
    xr.testing.assert_identical(ds, before)
    assert "10.1175/JTECH-D-20-0056.1" in res.attrs["radarx_references"]
    assert res.attrs["ffd_direction"] == 0.0


def test_a_grid_without_echo_gives_empty_results():
    ds = supercell()
    empty = ds.assign(DBZH=ds.DBZH * 0.0 - 10.0)
    res = zdr_arc(empty, **KW)
    assert res.sizes["candidate"] == 0 and res.sizes["storm"] == 0
    assert int((res.ZDR_ARC_STORM >= 0).sum()) == 0
