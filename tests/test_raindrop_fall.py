# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the raindrop fall speed shared by the disdrometer, trajectory and
evaporation modules (issue 231)
============================================================================

The drag-balance speed of small drops is checked against a root finder that
shares no code with the Newton iterations of the library, and the three
modules are checked against each other.
"""

import importlib

import numpy as np
import pytest
import xarray as xr
from scipy.optimize import brentq, least_squares

import radarx  # noqa: F401
from radarx.retrieve import _rain_trajectories_numpy as ref
from radarx.retrieve import (
    _raindrop_fall,
    drop_evaporation_rate,
    rain_trajectories,
    terminal_fall_speed,
)

evaporation = importlib.import_module("radarx.retrieve.evaporation")
rt = importlib.import_module("radarx.retrieve.rain_trajectories")
ENGINES = ["numpy"] + (["compiled"] if rt.HAS_COMPILED_KERNEL else [])
RHO0 = 1.204


def sphere(d_mm):
    """Drag balance of a sphere solved by Brent's method."""
    dm = d_mm * 1e-3
    best = 4.0 / 3.0 * 9.81 * RHO0 * (1000.0 - RHO0) * dm**3 / 1.81e-5**2
    re = brentq(lambda r: 24.0 * r + 3.6 * r**1.687 - best, 1e-12, 1e7, xtol=1e-15)
    return re * 1.81e-5 / (RHO0 * dm)


def atlas(d):
    return 9.65 - 10.3 * np.exp(-0.6 * d)


def test_small_drops_follow_the_drag_balance():
    for d in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
        v, _ = _raindrop_fall.sea_level_speed(d)
        assert v == pytest.approx(sphere(d), rel=1e-10)
    # the numbers of the validation of issue 231
    assert _raindrop_fall.sea_level_speed(0.2)[0] == pytest.approx(0.71, abs=0.005)
    assert _raindrop_fall.sea_level_speed(0.1)[0] == pytest.approx(0.25, abs=0.005)
    # the Atlas law gives 27 % and 100 % less there
    old = _raindrop_fall.sea_level_speed(np.array([0.2, 0.1]), "atlas1973")[0]
    np.testing.assert_allclose(old, [0.515, 0.0], atol=1e-3)


def test_law_is_continuous_and_matches_atlas_above_the_join():
    dj = _raindrop_fall.D_JOIN
    assert 0.51 < dj < 0.52
    lo, hi = (_raindrop_fall.sea_level_speed(dj + e)[0] for e in (-1e-9, 1e-9))
    assert lo == pytest.approx(hi, rel=1e-8)
    d = np.array([0.6, 1.0, 2.0, 5.0, 8.0])
    np.testing.assert_allclose(_raindrop_fall.sea_level_speed(d)[0], atlas(d))
    with pytest.raises(ValueError, match="law"):
        _raindrop_fall.sea_level_speed(1.0, "beard")


def test_slope_is_the_derivative():
    d = np.array([0.05, 0.2, 0.4, 0.8, 3.0])
    for law in ("drag_balance", "atlas1973"):
        _, dv = _raindrop_fall.sea_level_speed(d, law)
        h = 1e-6
        num = (
            _raindrop_fall.sea_level_speed(d + h, law)[0]
            - _raindrop_fall.sea_level_speed(d - h, law)[0]
        ) / (2 * h)
        np.testing.assert_allclose(dv[d > 0.12], num[d > 0.12], rtol=1e-6)


def test_exponential_fit_for_the_evaporation_integrals():
    d = np.geomspace(0.15, 5.0, 300)
    v = _raindrop_fall.sea_level_speed(d)[0]
    a, b, f = _raindrop_fall.ABF
    err = a * d**b * np.exp(-f * d) / v - 1.0
    assert np.abs(err).max() < 0.055
    # the coefficients are the minimax fit described in the module
    res = least_squares(
        lambda p: p[0] * d ** p[1] * np.exp(-p[2] * d) / v - 1.0, x0=_raindrop_fall.ABF
    )
    np.testing.assert_allclose(res.x, _raindrop_fall.ABF, rtol=0.05)
    wide = np.geomspace(0.1, 8.0, 100)
    vw = _raindrop_fall.sea_level_speed(wide)[0]
    ew = a * wide**b * np.exp(-f * wide) / vw - 1.0
    assert np.abs(ew[wide < 0.15]).max() < 0.25 and np.abs(ew).max() < 0.35
    # the earlier fit to the Atlas law was 38 % high at 0.2 mm
    old = evaporation.FALL_SPEED["atlas1973"]
    at = old[0] * 0.2 ** old[1] * np.exp(-old[2] * 0.2)
    assert at / _raindrop_fall.sea_level_speed(0.2)[0] > 1.3
    assert evaporation.FALL_SPEED["drag_balance"] == _raindrop_fall.ABF


def test_disdrometer_trajectories_and_evaporation_agree():
    d = np.array([0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 3.0])
    disdrometer = terminal_fall_speed(d).values
    np.testing.assert_allclose(disdrometer, _raindrop_fall.sea_level_speed(d)[0])
    z = np.arange(0.0, 3001.0, 500.0)
    prof = xr.Dataset(
        {
            "temperature": ("height", np.full(z.size, 293.15)),
            "pressure": ("height", np.full(z.size, 101325.0)),
            "u": ("height", np.zeros(z.size)),
            "v": ("height", np.zeros(z.size)),
        },
        coords={"height": z},
    )
    for engine in ENGINES:
        out = rain_trajectories(
            {"x": 0.0, "y": 0.0, "z": 1000.0},
            d,
            profile=prof,
            evaporation=False,
            density_correction=False,
            max_time=0.0,
            engine=engine,
        )
        np.testing.assert_allclose(
            out.fall_speed_start.values.ravel(), disdrometer, rtol=1e-12
        )
    # the evaporation rate of a single drop uses the (a, b, f) fit of the law:
    # the fit to the Atlas law was 38 % high at 0.2 mm, so the rate of the
    # small drops falls (by 8 % at 0.2 mm in this air)
    new = drop_evaporation_rate(d, 293.15, 101325.0, 0.006)
    old = drop_evaporation_rate(d, 293.15, 101325.0, 0.006, fall_speed="atlas1973")
    ratio = (new.diameter_rate / old.diameter_rate).values
    assert (new.diameter_rate < 0).all()
    assert 0.85 < ratio[1] < 0.97  # 0.2 mm
    assert ratio[-1] == pytest.approx(1.0, abs=0.2)  # 3 mm: the fits are close


def test_fall_distance_of_small_drops():
    """A 0.3 mm drop falls 12 % farther in a minute than with the Atlas law."""
    z = np.arange(0.0, 3001.0, 500.0)
    prof = xr.Dataset(
        {
            "temperature": ("height", np.full(z.size, 293.15)),
            "pressure": ("height", np.full(z.size, 101325.0)),
            "u": ("height", np.zeros(z.size)),
            "v": ("height", np.zeros(z.size)),
        },
        coords={"height": z},
    )
    kw = dict(profile=prof, evaporation=False, density_correction=False)
    for engine in ENGINES:
        new = rain_trajectories(
            {"x": 0.0, "y": 0.0, "z": 3000.0}, [0.3], max_time=60.0, engine=engine, **kw
        )
        old = rain_trajectories(
            {"x": 0.0, "y": 0.0, "z": 3000.0},
            [0.3],
            max_time=60.0,
            fall_speed="atlas1973",
            engine=engine,
            **kw,
        )
        fell_new = 3000.0 - float(new.landing_z.values.ravel()[0])
        fell_old = 3000.0 - float(old.landing_z.values.ravel()[0])
        assert fell_new == pytest.approx(60.0 * sphere(0.3), rel=1e-6)
        assert fell_old == pytest.approx(60.0 * atlas(0.3), rel=1e-6)
        assert fell_new / fell_old == pytest.approx(1.12, abs=0.01)


@pytest.mark.skipif(not rt.HAS_COMPILED_KERNEL, reason="compiled kernel not built")
def test_trajectory_engines_agree_for_small_drops():
    z = np.arange(0.0, 4001.0, 500.0)
    prof = xr.Dataset(
        {
            "temperature": ("height", 293.15 - 6.5e-3 * z),
            "pressure": ("height", 101325.0 * (1 - 6.5e-3 * z / 293.15) ** 5.2559),
            "relative_humidity": ("height", np.full(z.size, 0.7)),
            "u": ("height", 0.004 * z),
            "v": ("height", np.zeros(z.size)),
        },
        coords={"height": z},
    )
    d = [0.12, 0.2, 0.3, 0.5, 0.52, 1.0]
    a = rain_trajectories(
        {"x": 0.0, "y": 0.0, "z": 3000.0}, d, profile=prof, engine="numpy"
    )
    b = rain_trajectories(
        {"x": 0.0, "y": 0.0, "z": 3000.0}, d, profile=prof, engine="compiled"
    )
    for name in (
        "landing_x",
        "landing_time",
        "landing_diameter",
        "concentration_ratio",
    ):
        np.testing.assert_allclose(a[name], b[name], rtol=1e-9, atol=1e-9)


def test_numpy_reference_has_the_drag_balance_kind():
    d = np.array([0.2, 0.6])
    m = {"fall_kind": 3, "fc": np.array([_raindrop_fall.D_JOIN])}
    v, dv = ref._v0(m, d)
    np.testing.assert_allclose(v, [sphere(0.2), atlas(0.6)], rtol=1e-10)
    assert dv[0] > 0 and dv[1] > 0


def test_rain_trajectories_reject_unknown_law():
    with pytest.raises(ValueError, match="drag_balance"):
        rt._fall("beard1976")
