#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the mixed-order cone gridding
=======================================

A jet in the vertical profile of the wind, seen by a volume of sweeps with
a few degrees between them, shows the gridding artifacts of Collis et al.
(2010): the order of interpolation that ends in height leaves undulations
with the horizontal spacing of the beams.
"""

import numpy as np
import pytest
import xarray as xr
from xradar.georeference import antenna_to_cartesian

import radarx  # noqa: F401
from radarx.grid import cone, grid_cones, grid_cones_mixed, mixed

ENGINES = ["numpy"] + (["compiled"] if mixed.HAS_COMPILED_KERNEL else [])
ELEVATIONS = (0.5, 1.5, 3.0, 5.0, 7.5, 10.0, 13.0, 17.0, 22.0, 28.0, 35.0, 42.0)
ALT = 100.0


def jet(z):
    """A wind jet of 20 m/s at 7 km, 3 km deep."""
    return 20.0 * np.exp(-(((z - 7000.0) / 1500.0) ** 2))


def volume(
    profile,
    elevations=ELEVATIONS,
    azimuth=None,
    nan_fraction=0.0,
    name="VRADH",
    projected=True,
):
    """Radial velocity of the horizontal wind ``profile(z)`` blowing towards +x."""
    rng_m = np.arange(250.0, 80_000.0, 250.0)
    azimuth = np.arange(0.5, 360.0, 1.0) if azimuth is None else azimuth
    rnd = np.random.default_rng(0)
    sweeps = {}
    for k, el in enumerate(elevations):
        az = np.roll(azimuth, 11 * k)
        _, _, gz = antenna_to_cartesian(
            rng_m[None, :], az[:, None], el, site_altitude=ALT
        )
        data = np.broadcast_to(profile(gz), (az.size, rng_m.size)).copy()
        if projected:
            data = data * np.cos(np.radians(el)) * np.sin(np.radians(az))[:, None]
        if nan_fraction:
            data[rnd.random(data.shape) < nan_fraction] = np.nan
        sweeps[f"sweep_{k}"] = xr.Dataset(
            {name: (("azimuth", "range"), data, {"units": "m/s"})},
            coords={
                "azimuth": az,
                "range": rng_m,
                "elevation": ("azimuth", np.full(az.size, el)),
                "time": (
                    "azimuth",
                    np.full(az.size, np.datetime64("2026-01-01T00:00")),
                ),
            },
        )
    root = xr.Dataset(coords={"latitude": 45.0, "longitude": 10.0, "altitude": ALT})
    return xr.DataTree.from_dict({"/": root, **sweeps})


X = np.arange(-60e3, 60e3 + 1, 2e3)
Z = np.arange(1000.0, 13_000.0, 500.0)


def truth(x, y, z):
    """Radial velocity of the jet at grid points, with the local beam elevation."""
    s = np.hypot(x, y)
    el = np.radians(mixed.antenna_elevation(s, z, ALT))
    return jet(z) * np.cos(el) * x / np.where(s > 0, s, 1.0)


def errors(field):
    s = np.hypot(*np.meshgrid(X, X))
    t = truth(X[None, None, :], X[None, :, None], Z[:, None, None])
    d = field - t
    ok = np.isfinite(d) & (s[None] > 10e3)
    return d[ok]


@pytest.fixture(scope="module")
def jet_volume():
    return volume(jet)


@pytest.mark.parametrize("engine", ENGINES)
def test_horizontal_last_removes_the_vertical_undulations(jet_volume, engine):
    """Order 0 is horizontal last everywhere; it halves the error of the jet."""
    vertical = grid_cones(jet_volume, "VRADH", X, X, Z, engine=engine)["VRADH"]
    last = grid_cones_mixed(jet_volume, "VRADH", X, X, Z, order=0.0, engine=engine)[
        "VRADH"
    ]
    blend = grid_cones_mixed(jet_volume, "VRADH", X, X, Z, engine=engine)["VRADH"]
    rms = [np.sqrt(np.mean(errors(f.values) ** 2)) for f in (vertical, blend, last)]
    # 1.61, 1.22 and 0.71 m/s
    assert rms[0] > rms[1] > rms[2]
    assert rms[2] < 0.6 * rms[0]
    assert np.isfinite(last.values).sum() > 0.5 * np.isfinite(vertical.values).sum()


@pytest.mark.skipif(not mixed.HAS_COMPILED_KERNEL, reason="compiled kernel not built")
def test_engines_agree(jet_volume):
    a = grid_cones_mixed(jet_volume, "VRADH", X, X, Z, engine="numpy")["VRADH"]
    b = grid_cones_mixed(jet_volume, "VRADH", X, X, Z, engine="compiled")["VRADH"]
    np.testing.assert_array_equal(np.isnan(a.values), np.isnan(b.values))
    np.testing.assert_allclose(a.values, b.values, atol=1e-4, equal_nan=True)
    assert a.dtype == np.float32


@pytest.mark.skipif(not mixed.HAS_COMPILED_KERNEL, reason="compiled kernel not built")
def test_engines_agree_with_missing_gates_and_rays():
    azimuth = np.r_[np.arange(0.5, 100.0, 1.0), np.arange(130.5, 360.0, 1.0)]
    v = volume(jet, azimuth=azimuth, nan_fraction=0.05)
    a = mixed.horizontal_last(
        *columns(),
        sweeps_of(v),
        ALT,
        engine="numpy",
    )
    b = mixed.horizontal_last(*columns(), sweeps_of(v), ALT, engine="compiled")
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_allclose(a, b, atol=1e-4, equal_nan=True)
    # the gap of 30 degrees without rays stays empty
    in_gap = (columns()[1] > 102.0) & (columns()[1] < 128.0)
    assert np.isnan(b[:, in_gap]).all()
    assert np.isfinite(b[:, ~in_gap]).any()


def columns():
    gx, gy = np.meshgrid(X, X)
    return (
        np.hypot(gx, gy),
        np.mod(np.degrees(np.arctan2(gx, gy)), 360.0),
        Z,
    )


def sweeps_of(dtree, name="VRADH"):
    sweeps, _, _ = cone._load_volume(dtree)
    return [cone._sweep_arrays(ds, name) for ds in cone._select_sweeps(sweeps, name)]


def test_field_that_does_not_change_horizontally_is_reproduced(jet_volume):
    """The wind of the jet blowing along the beams is exact for horizontal last."""
    dtree = volume(jet, projected=False)
    last = mixed.horizontal_last(*columns(), sweeps_of(dtree), ALT)
    ground = np.hypot(*np.meshgrid(X, X))
    ok = np.isfinite(last) & (ground[None] > 10e3)
    assert ok.sum() > 10_000
    exact = np.broadcast_to(jet(Z)[:, None, None], last.shape)
    np.testing.assert_allclose(last[ok], exact[ok], atol=0.05)
    vertical = grid_cones(dtree, "VRADH", X, X, Z)["VRADH"].values
    assert np.nanmax(np.abs(vertical - exact)[ok]) > 1.0  # vertical last is not


def test_weight_of_the_horizontal_last_order():
    theta = np.array([0.0, 10.0, 21.0, 42.0, 50.0, -3.0])
    w = mixed.horizontal_weight(theta, 42.0, 1.0)
    # Eq. 6: [(sin(pi t - pi/2) + 1) / 2]^n, 0 at the lowest, 1 at the highest
    np.testing.assert_allclose(
        w[[0, 2, 3, 4, 5]], [0.0, 0.5, 1.0, 1.0, 0.0], atol=1e-12
    )
    half = mixed.horizontal_weight(theta, 42.0, 0.5)
    np.testing.assert_allclose(
        half, np.sin(np.pi * np.clip(theta, 0, 42) / 84), atol=1e-12
    )
    assert (mixed.horizontal_weight(theta, 42.0, 2.0) <= half + 1e-12).all()
    np.testing.assert_allclose(mixed.horizontal_weight(theta, 42.0, 0.0), 1.0)
    np.testing.assert_allclose(mixed.horizontal_weight(theta, 0.0, 0.5), 0.0)


def test_antenna_elevation_matches_the_beam_geometry():
    el = 5.0
    gx, gy, gz = antenna_to_cartesian(np.array([30e3]), 0.0, el, site_altitude=ALT)
    ground = np.hypot(gx, gy)
    # xradar's ground distance is along the surface of the effective Earth
    np.testing.assert_allclose(mixed.antenna_elevation(ground, gz, ALT), el, atol=0.01)


def test_blend_uses_the_available_order(jet_volume):
    """A cell with a value in only one order takes that value."""
    vertical = grid_cones(jet_volume, "VRADH", X, X, Z)["VRADH"]
    blend = grid_cones_mixed(jet_volume, "VRADH", X, X, Z)["VRADH"]
    # everything the vertical-last grid has, the blend has
    assert (np.isfinite(vertical.values) <= np.isfinite(blend.values)).all()
    assert blend.attrs["units"] == "m/s"
    ds = grid_cones_mixed(jet_volume, "VRADH", X, X, Z, order=0.25)
    assert ds.attrs["gridding_method"] == "mixed-order cone"
    assert ds.attrs["mixed_order"] == 0.25
    # all fields of the lowest sweep when none are named
    assert "VRADH" in grid_cones_mixed(jet_volume, None, X, X, Z)


def test_errors(jet_volume, monkeypatch):
    with pytest.raises(ValueError, match="x, y and z"):
        grid_cones_mixed(jet_volume, "VRADH")
    with pytest.raises(ValueError, match="order"):
        grid_cones_mixed(jet_volume, "VRADH", X, X, Z, order=-0.1)
    with pytest.raises(ValueError, match="order"):
        grid_cones_mixed(jet_volume, "VRADH", X, X, Z, order=float("nan"))
    with pytest.raises(ValueError, match="engine"):
        grid_cones_mixed(jet_volume, "VRADH", X, X, Z, engine="gpu")
    with pytest.raises(ValueError, match="k_e"):
        grid_cones_mixed(jet_volume, "VRADH", X, X, Z, k_e=-1.0)
    with pytest.raises(ValueError, match="No sweep contains"):
        grid_cones_mixed(jet_volume, "DBZH", X, X, Z)
    monkeypatch.setattr(mixed, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError):
        grid_cones_mixed(jet_volume, "VRADH", X, X, Z, engine="compiled")
    assert not mixed._use_compiled("auto")


@pytest.mark.parametrize("engine", ENGINES)
def test_sweeps_with_too_few_rays_are_rejected(engine):
    gates = np.arange(250.0, 5000.0, 250.0)
    data = np.ones((3, gates.size))
    sweep = (data, np.array([10.0, 10.0, 10.0]), np.full(3, 1.0), gates)
    with pytest.raises(ValueError, match="3 distinct azimuths"):
        mixed.horizontal_last(
            np.array([[1000.0]]),
            np.array([[20.0]]),
            Z,
            [sweep, sweep],
            ALT,
            engine=engine,
        )
    small = (np.ones((3, 1)), np.array([10.0, 20.0, 30.0]), np.full(3, 1.0), gates[:1])
    if engine == "compiled":
        with pytest.raises(ValueError, match="too small"):
            mixed.horizontal_last(
                np.array([[1000.0]]), np.array([[20.0]]), Z, [small], ALT, engine=engine
            )


@pytest.mark.parametrize("engine", ENGINES)
def test_levels_without_a_crossing_stay_empty(engine, jet_volume):
    """Levels below the radar, absurdly low levels, columns beyond the last gate."""
    levels = np.array([-6e6, ALT - 50.0, 3000.0, 9000.0])
    out = mixed.horizontal_last(
        np.array([[50e3, 100e3]]),
        np.array([[45.0, 45.0]]),
        levels,
        sweeps_of(jet_volume),
        ALT,
        engine=engine,
    )
    assert out.shape == (4, 1, 2)
    assert np.isnan(out[:2]).all()
    assert np.isfinite(out[2:, 0, 0]).all()
    assert np.isnan(out[:, 0, 1]).all()
