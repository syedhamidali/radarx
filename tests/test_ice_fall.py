#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests of the hail and melting-snow fall speeds
==============================================

The hail laws are compared with the numbers printed in the corrigendum of
Heymsfield et al. (2020), the melting snow law with the weight of Eq. (15) of
Frick et al. (2013), and the trajectory model with the public functions and
with exact fall times.
"""

import importlib
import warnings

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.retrieve import (
    _ice_fall,
    _raindrop_fall,
    hail_fall_speed,
    melting_snow_fall_speed,
    rain_source_points,
    rain_trajectories,
    terminal_fall_speed,
)

rt = importlib.import_module("radarx.retrieve.rain_trajectories")
ENGINES = ["numpy"] + (["compiled"] if rt.HAS_COMPILED_KERNEL else [])
D_X = (8.4 / 7.6) ** (1.0 / 0.22)  # cm


def standard_profile(top=8000.0):
    z = np.arange(0.0, top + 1.0, 100.0)
    return xr.Dataset(
        {
            "temperature": ("height", 288.15 - 0.0065 * z),
            "pressure": ("height", 101325.0 * (1.0 - 2.25577e-5 * z) ** 5.25588),
            "relative_humidity": ("height", np.full(z.size, 0.5)),
            "u": ("height", np.zeros(z.size)),
            "v": ("height", np.zeros(z.size)),
        },
        coords={"height": z},
    )


# --------------------------------------------------------------------------
# hail
# --------------------------------------------------------------------------


def test_hail_laws_are_the_printed_ones():
    # Eqs. (2a) and (2b) with D in cm
    for d_cm in (0.6, 1.0, 1.5):
        assert float(hail_fall_speed(10.0 * d_cm)) == pytest.approx(
            7.6 * d_cm**0.89, rel=1e-12
        )
    for d_cm in (1.6, 2.0, 5.0, 10.0):
        assert float(hail_fall_speed(10.0 * d_cm)) == pytest.approx(
            8.4 * d_cm**0.67, rel=1e-12
        )
    # Table 2, row 3
    assert float(hail_fall_speed(30.0, "graupel_hail_obs")) == pytest.approx(
        7.2 * 3.0**0.84
    )
    # Table 3
    table = {
        "p10": (6.73, 0.68),
        "p25": (8.33, 0.62),
        "p50": (9.45, 0.62),
        "p75": (10.54, 0.63),
        "p90": (11.58, 0.58),
    }
    for name, (a, b) in table.items():
        got = float(hail_fall_speed(25.0, f"heymsfield2020_{name}"))
        assert got == pytest.approx(a * 2.5**b, rel=1e-12)
    assert set(table) == set(_ice_fall.PERCENTILES)
    assert (
        len(_ice_fall.HAIL_LAWS) == 8
    )  # the 7 of Heymsfield et al. and Milbrandt and Yau


def test_hail_speed_is_continuous_and_increases_with_size():
    # the two branches of the default law cross at 1.576 cm
    left = float(hail_fall_speed(10.0 * D_X * (1 - 1e-12)))
    right = float(hail_fall_speed(10.0 * D_X * (1 + 1e-12)))
    assert left == pytest.approx(right, rel=1e-9)
    assert 7.6 * D_X**0.89 == pytest.approx(8.4 * D_X**0.67, rel=1e-12)
    d = np.geomspace(5.0, 120.0, 400)
    for law in _ice_fall.HAIL_LAWS:
        v = hail_fall_speed(d, law).values
        assert np.all(np.diff(v) > 0.0), law
        assert v[0] > 4.0 and v[-1] < 60.0
    # the derivative of the kernel helper, by differences
    v, dv = _ice_fall.hail_speed(np.array([8.0, 20.0, 40.0]))
    eps = 1e-6
    num = (_ice_fall.hail_speed(np.array([8.0, 20.0, 40.0]) + eps)[0] - v) / eps
    np.testing.assert_allclose(dv, num, rtol=1e-4)
    # the percentiles are ordered at the sizes of the data
    for d_mm in (10.0, 25.0, 50.0):
        speeds = [
            float(hail_fall_speed(d_mm, f"heymsfield2020_p{p}"))
            for p in (10, 25, 50, 75, 90)
        ]
        assert speeds == sorted(speeds)


def test_milbrandt_yau_hail_law_and_its_density_factor():
    # V = 206.89 D^0.6384 with D in m (Table 2 of Milbrandt and Yau 2005, Part II)
    for d_mm in (8.0, 20.0, 50.0):
        want = 206.89 * (d_mm * 1e-3) ** 0.6384
        assert float(hail_fall_speed(d_mm, "milbrandt_yau2005")) == pytest.approx(want)
    # the density factor (rho0/rho)^(1/2) of Part I, Eq. A1, rho0 = 1.204 kg m-3
    rho = xr.DataArray([1.204, 0.9, 0.6], dims="level")
    v = hail_fall_speed(20.0, "milbrandt_yau2005", air_density=rho)
    sea = float(hail_fall_speed(20.0, "milbrandt_yau2005"))
    np.testing.assert_allclose(v, sea * np.sqrt(1.204 / rho.values), rtol=1e-13)
    assert v.dims == ("level",)
    assert float(hail_fall_speed(20.0, "milbrandt_yau2005", air_density=1.204)) == sea
    assert _ice_fall.density_exponent("milbrandt_yau2005") == 0.5
    assert _ice_fall.density_exponent("heymsfield2020") == 0.0
    # a law without a density dependence refuses an air density
    with pytest.raises(ValueError, match="no air density dependence"):
        hail_fall_speed(20.0, air_density=1.0)
    with pytest.raises(ValueError, match="positive"):
        hail_fall_speed(20.0, "milbrandt_yau2005", air_density=[1.0, -1.0])
    # the derivative of the law used by the trajectory model
    v, dv = _ice_fall.hail_speed(np.array([8.0, 20.0, 40.0]), "milbrandt_yau2005")
    eps = 1e-6
    num = (
        _ice_fall.hail_speed(np.array([8.0, 20.0, 40.0]) + eps, "milbrandt_yau2005")[0]
        - v
    ) / eps
    np.testing.assert_allclose(dv, num, rtol=1e-5)


def test_hail_xarray_warning_and_errors():
    d = xr.DataArray(
        [6.0, 12.0, 24.0],
        dims="size",
        coords={"size": ("size", [1, 2, 3], {"units": "index"})},
        attrs={"units": "mm"},
        name="D",
    )
    before = d.copy(deep=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = hail_fall_speed(d)
    xr.testing.assert_identical(d, before)
    assert out.dims == ("size",) and out.name == "terminal_fall_speed"
    assert out.attrs["units"] == "m s-1" and out.attrs["law"] == "heymsfield2020"
    assert out["size"].attrs == {"units": "index"}
    with pytest.warns(UserWarning, match="below 5 mm"):
        small = hail_fall_speed([3.0, 10.0])
    assert np.isfinite(small).all()
    with pytest.raises(ValueError, match="positive"):
        hail_fall_speed([0.0, 10.0])
    with pytest.raises(ValueError, match="hail law must be"):
        hail_fall_speed(10.0, "knight1983")
    with pytest.raises(ValueError, match="hail law must be"):
        hail_fall_speed(10.0, "heymsfield2020_p13")
    assert np.isnan(float(hail_fall_speed(np.nan)))
    # provenance
    assert "10.1175/JAS-D-19-0185.1" in out.attrs["radarx_references"]
    # accessor
    via = d.radarx.hail_fall_speed(law="graupel_hail_obs")
    np.testing.assert_allclose(via, hail_fall_speed(d, "graupel_hail_obs"))


# --------------------------------------------------------------------------
# melting snow
# --------------------------------------------------------------------------


def test_melting_snow_weight_and_limits():
    ell = np.array([0.0, 0.1, 0.5, 0.7, 0.9, 1.0])
    expected = 0.246 * ell + 0.754 * ell**7
    np.testing.assert_allclose(_ice_fall.psi(ell), expected, rtol=1e-14)
    assert _ice_fall.psi(0.0) == 0.0 and _ice_fall.psi(1.0) == pytest.approx(
        1.0, abs=1e-15
    )
    # slow and nearly linear below 0.5, fast above 0.7 (Fig. 2 of Mitra et al.)
    assert _ice_fall.psi(0.5) < 0.13 and _ice_fall.psi(0.9) > 0.55
    d = np.array([0.5, 1.0, 2.0, 4.0])
    snow = (1.2, 0.35)
    dry = melting_snow_fall_speed(d, 0.0, snow=snow)
    np.testing.assert_allclose(dry, 1.2 * d**0.35, rtol=1e-14)
    rain = melting_snow_fall_speed(d, 1.0, snow=snow)
    np.testing.assert_allclose(rain, terminal_fall_speed(d), rtol=1e-12)
    mid = melting_snow_fall_speed(d, 0.6, snow=snow).values
    w = 0.246 * 0.6 + 0.754 * 0.6**7
    np.testing.assert_allclose(mid, (1 - w) * dry.values + w * rain.values, rtol=1e-12)
    assert np.all((mid > dry.values) & (mid < rain.values))
    # the Atlas law as the rain law
    atlas = melting_snow_fall_speed(d, 1.0, snow=snow, rain="atlas1973")
    np.testing.assert_allclose(
        atlas, terminal_fall_speed(d, law="atlas1973"), rtol=1e-12
    )
    # increasing with size at every liquid fraction (a snow law that increases)
    dd = np.geomspace(0.3, 8.0, 200)
    for fraction in (0.0, 0.3, 0.9, 1.0):
        assert np.all(
            np.diff(melting_snow_fall_speed(dd, fraction, snow=snow).values) > 0.0
        )
    # monotone in the liquid fraction when rain falls faster than snow
    fraction = np.linspace(0.0, 1.0, 51)
    v = melting_snow_fall_speed(2.0, xr.DataArray(fraction, dims="l"), snow=snow).values
    assert np.all(np.diff(v) > 0.0)


def test_melting_snow_xarray_and_errors():
    d = xr.DataArray([1.0, 2.0, 3.0], dims="diameter", attrs={"units": "mm"})
    ell = xr.DataArray([0.0, 0.5, 1.0, np.nan], dims="fraction")
    out = melting_snow_fall_speed(d, ell, snow=(1.0, 0.3))
    assert out.dims == ("diameter", "fraction")
    assert np.isnan(out.isel(fraction=3)).all()
    assert out.attrs["units"] == "m s-1" and out.name == "terminal_fall_speed"
    assert "10.5194/gmd-6-1925-2013" in out.attrs["radarx_references"]
    assert d.attrs == {"units": "mm"}
    with pytest.raises(ValueError, match="between 0 and 1"):
        melting_snow_fall_speed(d, 1.2, snow=(1.0, 0.3))
    with pytest.raises(ValueError, match="between 0 and 1"):
        melting_snow_fall_speed(d, -0.1, snow=(1.0, 0.3))
    with pytest.raises(ValueError, match="snow needs"):
        melting_snow_fall_speed(d, 0.5, snow=(0.0, 0.3))
    with pytest.raises(ValueError, match="snow needs"):
        melting_snow_fall_speed(d, 0.5, snow=(1.0, -1.0))
    with pytest.raises(ValueError, match="rain must be"):
        melting_snow_fall_speed(d, 0.5, snow=(1.0, 0.3), rain="beard1976")
    with pytest.raises(ValueError, match="positive"):
        melting_snow_fall_speed([-1.0], 0.5, snow=(1.0, 0.3))
    with pytest.raises(TypeError):
        melting_snow_fall_speed(d, 0.5)
    via = d.radarx.melting_snow_fall_speed(0.5, snow=(1.0, 0.3))
    np.testing.assert_allclose(via, melting_snow_fall_speed(d, 0.5, snow=(1.0, 0.3)))


# --------------------------------------------------------------------------
# the trajectory model
# --------------------------------------------------------------------------

DIAMS = [8.0, 12.0, 15.0, 16.0, 30.0, 60.0]


@pytest.mark.parametrize("engine", ENGINES)
def test_hail_trajectories_fall_at_the_hail_speed(engine):
    prof = standard_profile()
    point = {"x": 0.0, "y": 0.0, "z": 6000.0}
    for law in ("heymsfield2020", "graupel_hail_obs", "heymsfield2020_p50"):
        res = rain_trajectories(
            point,
            DIAMS,
            profile=prof,
            fall_speed=("hail", law),
            time_step=10.0,
            engine=engine,
        )
        v = hail_fall_speed(DIAMS, law).values
        # no density correction and no evaporation: constant speed, exact fall time
        np.testing.assert_allclose(res.fall_speed_start.values.ravel(), v, rtol=1e-12)
        np.testing.assert_allclose(res.fall_speed_end.values.ravel(), v, rtol=1e-12)
        np.testing.assert_allclose(res.fall_time.values.ravel(), 6000.0 / v, rtol=1e-9)
        np.testing.assert_allclose(
            res.landing_diameter.values.ravel(), DIAMS, rtol=1e-12
        )
        assert (res.status == 1).all()
        # the number flux is conserved without a density correction
        np.testing.assert_allclose(res.concentration_ratio, 1.0, atol=1e-12)
        np.testing.assert_allclose(res.evaporated_mass_fraction, 0.0, atol=1e-15)
    # the rain law with the density correction is not the same
    rain = rain_trajectories(point, [3.0], profile=prof, time_step=10.0, engine=engine)
    assert abs(float(rain.concentration_ratio.values.ravel()[0]) - 1.0) > 1e-3


@pytest.mark.parametrize("engine", ENGINES)
def test_milbrandt_yau_hail_trajectories_have_the_density_factor(engine):
    prof = standard_profile()
    point = {"x": 0.0, "y": 0.0, "z": 6000.0}
    law = ("hail", "milbrandt_yau2005")
    off = rain_trajectories(
        point,
        DIAMS,
        profile=prof,
        fall_speed=law,
        density_correction=False,
        time_step=10.0,
        engine=engine,
    )
    v = hail_fall_speed(DIAMS, "milbrandt_yau2005").values
    np.testing.assert_allclose(off.fall_speed_start.values.ravel(), v, rtol=1e-12)
    np.testing.assert_allclose(off.fall_time.values.ravel(), 6000.0 / v, rtol=1e-9)
    on = rain_trajectories(
        point, DIAMS, profile=prof, fall_speed=law, time_step=10.0, engine=engine
    )
    # faster aloft by (rho0/rho)^0.5 with the density of the profile at the start,
    # and the same factor at the ground within the profile's density there
    ratio = on.fall_speed_start.values.ravel() / v
    assert np.all(ratio > 1.3) and np.all(
        ratio < 1.6
    )  # (1.204/0.66)^0.5 = 1.35 at 6 km
    assert (on.fall_time.values < off.fall_time.values).all()
    # the law of the other family keeps its sea-level speed (kind 4, no factor)
    hm = rain_trajectories(
        point, DIAMS, profile=prof, fall_speed=("hail", "heymsfield2020"), engine=engine
    )
    np.testing.assert_allclose(
        hm.fall_speed_start.values.ravel(), hail_fall_speed(DIAMS).values, rtol=1e-12
    )
    # trajectory speed at the start equals the public function with the profile density
    rho = float(
        np.interp(6000.0, prof.height, prof.pressure / (287.04 * prof.temperature))
    )
    np.testing.assert_allclose(
        on.fall_speed_start.values.ravel(),
        hail_fall_speed(DIAMS, "milbrandt_yau2005", air_density=rho).values,
        rtol=0.02,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_melting_snow_trajectories_follow_the_mixed_speed(engine):
    prof = standard_profile()
    point = {"x": 0.0, "y": 0.0, "z": 3000.0}
    d = np.array([0.8, 1.5, 3.0])
    for ell in (0.0, 0.4, 0.8, 1.0):
        res = rain_trajectories(
            point,
            d,
            profile=prof,
            fall_speed=("melting_snow", ell, 1.1, 0.3),
            time_step=10.0,
            engine=engine,
        )
        v = melting_snow_fall_speed(d, ell, snow=(1.1, 0.3)).values
        np.testing.assert_allclose(res.fall_speed_start.values.ravel(), v, rtol=1e-12)
        np.testing.assert_allclose(res.fall_time.values.ravel(), 3000.0 / v, rtol=1e-9)
        np.testing.assert_allclose(res.concentration_ratio, 1.0, atol=1e-12)
    # more liquid water: faster fall, earlier landing
    slow = rain_trajectories(
        point, d, profile=prof, fall_speed=("melting_snow", 0.1, 1.1, 0.3)
    )
    fast = rain_trajectories(
        point, d, profile=prof, fall_speed=("melting_snow", 0.9, 1.1, 0.3)
    )
    assert (fast.fall_time.values < slow.fall_time.values).all()


@pytest.mark.parametrize("engine", ENGINES)
def test_ice_laws_drift_with_the_wind_and_run_backward(engine):
    prof = standard_profile()
    prof["u"] = ("height", np.full(prof.height.size, 12.0))
    fall = ("hail", "heymsfield2020")
    fwd = rain_trajectories(
        {"x": 0.0, "y": 0.0, "z": 5000.0},
        [20.0],
        profile=prof,
        fall_speed=fall,
        engine=engine,
    )
    v = float(hail_fall_speed(20.0))
    assert fwd.landing_x.values.ravel()[0] == pytest.approx(12.0 * 5000.0 / v, rel=1e-8)
    back = rain_source_points(
        {"x": fwd.landing_x.values.ravel()[0], "y": 0.0, "time": 0.0},
        [20.0],
        source_height=5000.0,
        profile=prof,
        fall_speed=fall,
        engine=engine,
    )
    assert back.source_x.values.ravel()[0] == pytest.approx(0.0, abs=1e-3)


def test_ice_laws_and_evaporation():
    prof = standard_profile()
    point = {"x": 0.0, "y": 0.0, "z": 3000.0}
    with pytest.raises(ValueError, match="raindrops only"):
        rain_trajectories(
            point,
            [10.0],
            profile=prof,
            fall_speed=("hail", "heymsfield2020"),
            evaporation=True,
        )
    # the default switches evaporation off for ice although the profile is humid
    res = rain_trajectories(
        point, [10.0], profile=prof, fall_speed=("hail", "heymsfield2020")
    )
    assert int(res.attrs["evaporation"]) == 0
    res = rain_trajectories(point, [1.0], profile=prof)
    assert int(res.attrs["evaporation"]) == 1


def test_fall_speed_specifications():
    kind, fc = rt._fall(("hail", "heymsfield2020"))
    assert kind == 4
    np.testing.assert_allclose(fc, [2.0, 10.0 * D_X, 7.6, 0.89, 8.4, 0.67])
    kind, fc = rt._fall(("hail", "graupel_hail_obs"))
    np.testing.assert_allclose(fc, [1.0, 7.2, 0.84])
    kind, fc = rt._fall(("hail", "milbrandt_yau2005"))
    assert kind == 6
    np.testing.assert_allclose(fc, [1.0, 206.89 * 0.01**0.6384, 0.6384])
    kind, fc = rt._fall(("melting_snow", 0.5, 1.1, 0.3))
    assert kind == 5
    np.testing.assert_allclose(
        fc, [0.246 * 0.5 + 0.754 * 0.5**7, 1.1, 0.3, _raindrop_fall.D_JOIN]
    )
    for bad in (
        ("hail",),
        ("hail", "a", "b"),
        ("hail", "nope"),
        ("melting_snow", 0.5, 1.0),
        ("melting_snow", 1.5, 1.0, 0.3),
        ("melting_snow", 0.5, 0.0, 0.3),
        ("melting_snow", 0.5, 1.0, -1.0),
        "hail",
        ("polynomial",),
        (1.0, 2.0),
    ):
        with pytest.raises(ValueError):
            rt._fall(bad)


@pytest.mark.skipif(not rt.HAS_COMPILED_KERNEL, reason="compiled kernel not built")
def test_kernel_rejects_malformed_ice_coefficients():
    prof = standard_profile()
    su = rt._setup(
        wind=None,
        profile=prof,
        storm_motion=None,
        pattern_time=None,
        evaporation=False,
        fall_speed=("hail", "heymsfield2020"),
        density_correction=True,
        wind_divergence=False,
        concentration=True,
        dispersion=None,
        time_step=5.0,
        max_time=600.0,
        scheme="rk4",
        evaporated_diameter=0.12,
        direction="forward",
        store_path=0,
        epoch_hint=np.datetime64("1970-01-01"),
    )
    args = ([0.0], [0.0], [1000.0], [0.0], [20.0], [0.0], 0, True, 0)
    out, _ = rt._integrate(su, *args)
    assert out.shape[0] == 1
    for fc in (
        np.array([2.0, 10.0, 7.6, 0.89]),  # too short
        np.array(
            [3.0, 20.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        ),  # breaks do not increase
        np.array([0.0]),  # no segment
    ):
        su.fc = fc
        with pytest.raises(ValueError, match="hail"):
            rt._integrate(su, *args)
    # a melting snow law of the wrong length
    su.ipar[1] = 5
    su.fc = np.array([0.5, 1.0, 0.3])
    with pytest.raises(ValueError, match="melting snow"):
        rt._integrate(su, *args)
    su.ipar[1] = 9
    with pytest.raises(ValueError, match="fall_kind"):
        rt._integrate(su, *args)


def test_numpy_and_compiled_agree_for_ice_laws():
    if not rt.HAS_COMPILED_KERNEL:
        pytest.skip("compiled kernel not built")
    prof = standard_profile()
    prof["u"] = ("height", 3.0 + 0.003 * prof.height.values)
    point = {"x": [0.0, 500.0], "y": [0.0, 0.0], "z": [4000.0, 7000.0]}
    for fall in (("hail", "heymsfield2020"), ("melting_snow", 0.6, 1.1, 0.3)):
        a = rain_trajectories(
            point,
            [9.0, 18.0, 40.0] if fall[0] == "hail" else [1.0, 2.0, 4.0],
            profile=prof,
            fall_speed=fall,
            engine="numpy",
        )
        b = rain_trajectories(
            point,
            [9.0, 18.0, 40.0] if fall[0] == "hail" else [1.0, 2.0, 4.0],
            profile=prof,
            fall_speed=fall,
            engine="compiled",
        )
        for name in ("landing_x", "landing_time", "fall_time", "concentration_ratio"):
            np.testing.assert_allclose(a[name], b[name], rtol=1e-9, atol=1e-9)
