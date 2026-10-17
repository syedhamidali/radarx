#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the attenuation of Ka and W band cloud radar signals
===============================================================

The line-by-line gas attenuation is checked against values of an independent
implementation of Rec. ITU-R P.676 (the ``itur`` package, same line tables) and
against the physics (a uniform atmosphere has a path attenuation that is its
specific attenuation times the length), the cloud liquid attenuation against
the Rayleigh absorption of the permittivity model of Ray (1972), the rain relation
against the specific attenuation of Rec. ITU-R P.838, the integral against closed
forms and the Hitschfeld and Bordan solution against a forward model. Real data: the
Cloudnet categorization file of Lindenberg, 1 May 2023 (35.5 GHz), whose gas
attenuation is computed with another absorption model.
"""

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401  registers the accessors
from radarx.fundamentals.geometry import beam_center_height
from radarx.retrieve import cloud_attenuation as ca
from radarx.retrieve import (
    cloud_liquid_attenuation,
    gas_path_attenuation,
    gas_specific_attenuation,
    rain_attenuation_relation,
    rain_path_attenuation,
)
from radarx.retrieve.cloud_attenuation import (
    _integrate_numpy,
    _liquid_coefficient_numpy,
)

compiled_only = pytest.mark.skipif(
    not ca.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)
ENGINES = ["numpy", pytest.param("compiled", marks=compiled_only)]

# gamma_o and gamma_w of the itur package (Rec. ITU-R P.676-12 tables, which are those
# of P.676-13): frequency (GHz), dry-air pressure (hPa), temperature (K), water vapour
# density (g m-3), oxygen and water vapour attenuation (dB km-1)
ITUR = [
    (35.0, 1013.25, 288.15, 7.5, 0.03184303386435486, 0.06961429562456616),
    (94.0, 1013.25, 288.15, 7.5, 0.03448072305000696, 0.3736481599889677),
    (22.235, 1013.25, 288.15, 7.5, 0.013292678183376018, 0.17897799237293666),
    (118.75, 1013.25, 288.15, 7.5, 1.3339530073242092, 0.6149752830074323),
    (183.31, 1013.25, 288.15, 7.5, 0.012746473180202167, 28.00772010224626),
    (35.0, 700.0, 270.0, 2.0, 0.018246037444022475, 0.014079627894009052),
    (94.0, 500.0, 250.0, 0.5, 0.013793879768280632, 0.016660823852194313),
    (94.0, 900.0, 300.0, 20.0, 0.02399582668203379, 1.0206924341663255),
    (52.123, 311.276, 249.873, 4.3575, 0.09169544819054302, 0.04243173284624131),
    (178.106, 960.068, 234.725, 11.3364, 0.025287160866170937, 20.281899626213598),
]


def total_pressure(p_dry, temperature, density):
    return p_dry + density * temperature / 216.7


# ---- specific attenuation of the gases ------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("row", ITUR)
def test_gas_matches_the_itur_package(row, engine):
    f, p, t, rho, oxygen, vapor = row
    out = gas_specific_attenuation(
        f, total_pressure(p, t, rho), t, vapor_density=rho, engine=engine
    )
    assert float(out.A_OXYGEN) == pytest.approx(oxygen, rel=2e-6)
    assert float(out.A_WATER_VAPOR) == pytest.approx(vapor, rel=2e-6)
    assert float(out.A_GAS) == pytest.approx(oxygen + vapor, rel=2e-6)
    assert out.A_GAS.attrs["units"] == "dB km-1"
    assert out.A_GAS.dtype == np.float32


def test_standard_atmosphere_values_of_the_module_docstring():
    p = total_pressure(1003.0, 288.15, 7.5)
    ka = gas_specific_attenuation(35.0, p, 288.15, vapor_density=7.5)
    w = gas_specific_attenuation(94.0, p, 288.15, vapor_density=7.5)
    assert float(ka.A_GAS) == pytest.approx(0.10, abs=0.005)
    assert float(w.A_GAS) == pytest.approx(0.40, abs=0.01)
    assert float(ka.A_OXYGEN) == pytest.approx(0.03, abs=0.005)
    assert float(w.A_OXYGEN) == pytest.approx(0.03, abs=0.005)


@compiled_only
def test_engines_agree_and_threads_do_not_matter():
    rnd = np.random.default_rng(1)
    n = 400
    f = xr.DataArray(rnd.uniform(5, 300, n), dims="x")
    p = xr.DataArray(rnd.uniform(100, 1013, n), dims="x")
    t = xr.DataArray(rnd.uniform(220, 310, n), dims="x")
    e = xr.DataArray(rnd.uniform(0.0, 25, n), dims="x")
    ref = gas_specific_attenuation(f, p, t, e, engine="numpy")
    for threads in (1, 4):
        got = gas_specific_attenuation(f, p, t, e, n_threads=threads, engine="compiled")
        for name in ref:
            np.testing.assert_allclose(got[name], ref[name], rtol=2e-6, atol=1e-9)


def test_inputs_are_broadcast_and_keep_their_coordinates():
    f = xr.DataArray([35.0, 94.0], dims="frequency", coords={"frequency": [35.0, 94.0]})
    h = xr.DataArray(
        [0.0, 2000.0, 4000.0], dims="height", coords={"height": [0.0, 2000.0, 4000.0]}
    )
    p = 1013.25 * np.exp(-h / 8000.0)
    t = 288.15 - 6.5e-3 * h
    out = gas_specific_attenuation(f, p, t, vapor_density=7.5 * np.exp(-h / 2000.0))
    assert out.A_GAS.dims == ("frequency", "height")
    assert out.A_GAS.shape == (2, 3)
    assert (out.A_GAS.sel(frequency=94.0) > out.A_GAS.sel(frequency=35.0)).all()
    assert (out.A_GAS.diff("height") < 0).all()  # thinner and drier with height


def test_vapor_pressure_and_density_are_the_same_humidity():
    rho, t = 9.0, 285.0
    e = rho * t / 216.7
    a = gas_specific_attenuation(94.0, 1000.0, t, vapor_density=rho)
    b = gas_specific_attenuation(94.0, 1000.0, t, e)
    np.testing.assert_allclose(a.A_GAS, b.A_GAS, rtol=1e-6)


def test_dry_air_has_no_water_vapour_attenuation():
    out = gas_specific_attenuation(94.0, 1000.0, 280.0, 0.0)
    assert float(out.A_WATER_VAPOR) == 0.0
    assert float(out.A_OXYGEN) > 0


def test_missing_and_invalid_points_are_nan():
    out = gas_specific_attenuation(
        xr.DataArray([94.0, np.nan, 94.0, 0.0]),
        1000.0,
        xr.DataArray([280.0, 280.0, -1.0, 280.0]),
        5.0,
    )
    assert np.isfinite(out.A_GAS.values[0])
    assert np.isnan(out.A_GAS.values[1:]).all()
    numpy = gas_specific_attenuation(
        xr.DataArray([94.0, np.nan, 94.0, 0.0]),
        1000.0,
        xr.DataArray([280.0, 280.0, -1.0, 280.0]),
        5.0,
        engine="numpy",
    )
    np.testing.assert_array_equal(np.isnan(numpy.A_GAS), np.isnan(out.A_GAS))


def test_specific_attenuation_errors():
    with pytest.raises(ValueError, match="exactly one"):
        gas_specific_attenuation(94.0, 1000.0, 280.0)
    with pytest.raises(ValueError, match="exactly one"):
        gas_specific_attenuation(94.0, 1000.0, 280.0, 5.0, vapor_density=5.0)
    with pytest.raises(ValueError, match="engine"):
        gas_specific_attenuation(94.0, 1000.0, 280.0, 5.0, engine="gpu")
    with pytest.raises(ValueError, match="engine"):
        cloud_liquid_attenuation(94.0, 280.0, engine="gpu")


def test_the_kernel_is_not_required_for_the_numpy_engine(monkeypatch):
    monkeypatch.setattr(ca, "HAS_COMPILED_KERNEL", False)
    out = gas_specific_attenuation(94.0, 1000.0, 280.0, 5.0)
    assert np.isfinite(out.A_GAS)
    with pytest.raises(ImportError):
        gas_specific_attenuation(94.0, 1000.0, 280.0, 5.0, engine="compiled")


# ---- cloud liquid water ----------------------------------------------------------


def rayleigh_coefficient(frequency, celsius):
    """Absorption of liquid water per g m-3 in the Rayleigh limit, dB km-1, from Ray (1972)."""
    from radarx.fundamentals.tmatrix import water_refractive_index

    wavelength = 299.792458 / frequency  # mm
    m = water_refractive_index(wavelength, celsius)
    m = complex(m.real, -m.imag)  # the n - jk convention of the dielectric factor
    k = (m**2 - 1) / (m**2 + 2)
    per_m = 6.0 * np.pi / (wavelength * 1e-3) * (-k.imag) * 1e-6
    return 4.343 * per_m * 1e3


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("frequency", [35.0, 94.0])
@pytest.mark.parametrize("celsius", [0.0, 10.0, 20.0])
def test_cloud_liquid_is_the_rayleigh_absorption_of_water(frequency, celsius, engine):
    k = cloud_liquid_attenuation(frequency, celsius + 273.15, engine=engine)
    # two permittivity models (double Debye of Liebe et al. 1991 and Ray 1972)
    assert float(k) == pytest.approx(rayleigh_coefficient(frequency, celsius), rel=0.08)
    assert k.attrs["units"] == "dB km-1"


def test_cloud_liquid_values_and_scaling():
    k94 = cloud_liquid_attenuation(94.0, 273.15)
    k35 = cloud_liquid_attenuation(35.0, 273.15)
    assert float(k94) == pytest.approx(4.55, abs=0.05)
    assert float(k35) == pytest.approx(1.02, abs=0.02)
    lwc = xr.DataArray([0.0, 0.1, 0.5], dims="gate")
    scaled = cloud_liquid_attenuation(94.0, 273.15, lwc)
    np.testing.assert_allclose(scaled, float(k94) * lwc.values, rtol=1e-6)
    # warmer water absorbs less at W band, more at Ka band is not true: both fall
    assert float(cloud_liquid_attenuation(94.0, 293.15)) < float(k94)


def test_cloud_liquid_engines_and_missing_values():
    f = xr.DataArray([35.0, 94.0, np.nan, 94.0])
    t = xr.DataArray([273.0, 260.0, 273.0, np.nan])
    a = cloud_liquid_attenuation(f, t, 0.3, engine="numpy")
    assert np.isnan(a.values[2:]).all() and np.isfinite(a.values[:2]).all()
    if ca.HAS_COMPILED_KERNEL:
        b = cloud_liquid_attenuation(f, t, 0.3, engine="compiled", n_threads=2)
        np.testing.assert_allclose(b, a, rtol=1e-6)
    np.testing.assert_allclose(
        _liquid_coefficient_numpy(np.array([94.0]), np.array([273.15])),
        [4.5465],
        atol=1e-3,
    )


# ---- the integral ----------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_integral_of_a_constant_is_twice_the_length(engine):
    rng = np.arange(100.0, 3100.0, 30.0)
    gamma = np.full((3, rng.size), 0.2)
    out = ca._integrate(rng, gamma, gamma[:, 0], engine == "compiled", 1)
    np.testing.assert_allclose(
        out, 2 * 0.2 * rng[None, :] * 1e-3 * np.ones((3, 1)), rtol=1e-12
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_integral_of_a_ramp_is_exact_for_the_trapezoid(engine):
    rng = np.arange(50.0, 2000.0, 50.0)
    gamma = (0.001 * rng)[None, :]  # dB/km, linear in range
    out = ca._integrate(rng, gamma, gamma[:, 0], engine == "compiled", 1)
    # the first interval 0..50 m has the value of the first gate, then the trapezoid
    # of the ramp is exact: one way 0.001 * (50 * 50e-3 + (r**2 - 50**2) / 2 * 1e-3) dB
    one_way = 0.001 * (50.0 * 50e-3 + (rng**2 - 50.0**2) / 2.0 * 1e-3)
    np.testing.assert_allclose(out[0], 2.0 * one_way, rtol=1e-10)


@pytest.mark.parametrize("engine", ENGINES)
def test_integral_carries_over_missing_gates(engine):
    rng = np.arange(100.0, 600.0, 100.0)
    gamma = np.array([[np.nan, 1.0, np.nan, np.nan, 3.0]])
    out = ca._integrate(rng, gamma, np.array([np.nan]), engine == "compiled", 2)
    # gate 0 has no value and neither has the radar: 0 up to the first value
    carry = [0.0, 1.0, 1.0, 1.0, 3.0]
    start, expected = 0.0, []
    total = 0.0
    prev_r = 0.0
    for r, g in zip(rng, carry):
        total += 0.5 * (start + g) * (r - prev_r) * 1e-3
        start, prev_r = g, r
        expected.append(2 * total)
    np.testing.assert_allclose(out[0], expected, rtol=1e-12)


def test_integral_kernel_and_reference_agree_on_random_fields():
    rnd = np.random.default_rng(2)
    rng = np.cumsum(rnd.uniform(10, 40, 90))
    gamma = rnd.uniform(0, 2, (37, 90))
    gamma[rnd.random(gamma.shape) < 0.1] = np.nan
    g0 = rnd.uniform(0, 1, 37)
    ref = _integrate_numpy(rng, gamma, g0)
    if ca.HAS_COMPILED_KERNEL:
        np.testing.assert_allclose(
            ca._integrate(rng, gamma, g0, True, 3), ref, rtol=1e-12
        )


@compiled_only
def test_kernel_input_checks():
    from radarx.retrieve import _gas

    with pytest.raises(ValueError, match="same length"):
        _gas.specific(
            np.ones(3), np.ones(2), np.ones(3), np.ones(3), ca._OXYGEN, ca._WATER_VAPOR
        )
    with pytest.raises(ValueError, match="1-D"):
        _gas.specific(
            np.ones((2, 2)),
            np.ones(4),
            np.ones(4),
            np.ones(4),
            ca._OXYGEN,
            ca._WATER_VAPOR,
        )
    with pytest.raises(ValueError, match="7 columns"):
        _gas.specific(
            np.ones(3),
            np.ones(3),
            np.ones(3),
            np.ones(3),
            np.ones((2, 5)),
            ca._WATER_VAPOR,
        )
    with pytest.raises(ValueError, match="same length"):
        _gas.cloud_liquid(np.ones(3), np.ones(2), np.ones(3))
    with pytest.raises(ValueError, match="1-D"):
        _gas.cloud_liquid(np.ones((2, 2)), np.ones(4), np.ones(4))
    with pytest.raises(ValueError, match="2-D"):
        _gas.integrate(np.ones(3), np.ones(3), np.ones(1))
    with pytest.raises(ValueError, match="one value per gate"):
        _gas.integrate(np.ones(4), np.ones((1, 3)), np.ones(1))
    with pytest.raises(ValueError, match="one value per ray"):
        _gas.integrate(np.ones(3), np.ones((2, 3)), np.ones(1))


# ---- the path attenuation of a sweep or a section -------------------------------------


def profile(constant=False):
    height = np.arange(0.0, 12001.0, 250.0)
    if constant:
        p = np.full(height.size, 90000.0)
        t = np.full(height.size, 280.0)
        q = np.full(height.size, 0.006)
    else:
        p = 101325.0 * np.exp(-height / 8000.0)
        t = 288.15 - 6.5e-3 * np.minimum(height, 11000.0)
        q = 0.008 * np.exp(-height / 2500.0)
    return xr.Dataset(
        {
            "pressure": ("height", p),
            "temperature": ("height", t),
            "specific_humidity": ("height", q),
        },
        coords={"height": height},
    )


def sweep(elevation=30.0, n_az=6, n_gates=200, altitude=100.0, frequency=94.0):
    rng = 100.0 + 30.0 * np.arange(n_gates)
    az = np.linspace(0.0, 350.0, n_az)
    return xr.Dataset(
        {
            "DBZ": (("azimuth", "range"), np.full((n_az, n_gates), 10.0)),
            "radar_frequency": ((), frequency),
        },
        coords={
            "azimuth": az,
            "range": rng,
            "elevation": ("azimuth", np.full(n_az, elevation)),
            "altitude": altitude,
        },
    )


def vapor_pressure_of(prof):
    from radarx.io.sounding import specific_humidity_from_dewpoint  # noqa: F401
    from radarx.retrieve.beam_path import _vapor_pressure

    return _vapor_pressure(prof, prof.pressure, prof.temperature, "auto")


@pytest.mark.parametrize("engine", ENGINES)
def test_a_uniform_atmosphere_attenuates_by_its_specific_attenuation(engine):
    prof = profile(constant=True)
    ds = sweep(elevation=20.0)
    out = gas_path_attenuation(ds, prof, engine=engine)
    e = vapor_pressure_of(prof).values[0] / 100.0
    gamma = gas_specific_attenuation(94.0, 900.0, 280.0, e)
    expected = 2.0 * float(gamma.A_GAS) * ds.range.values * 1e-3
    np.testing.assert_allclose(out.PIA_GAS.isel(azimuth=0), expected, rtol=2e-6)
    np.testing.assert_allclose(
        out.PIA_OXYGEN + out.PIA_WATER_VAPOR, out.PIA_GAS, rtol=1e-6, atol=1e-7
    )
    assert "PIA_CLOUD_LIQUID" not in out
    assert out.PIA_GAS.dims == ("azimuth", "range")
    assert out.PIA_GAS.attrs["units"] == "dB"
    assert out.attrs["frequency_GHz"] == 94.0


def test_a_vertical_section_matches_a_fine_integral_of_the_profile():
    prof = profile()
    height = np.arange(150.0, 8000.0, 30.0)
    section = xr.Dataset(
        {"Z": (("time", "range"), np.zeros((3, height.size)))},
        coords={
            "time": np.arange(3),
            "range": height - 100.0,
            "height": ("range", height),
            "radar_frequency": 35.5,
        },
    )
    out = gas_path_attenuation(section, prof)
    # fine reference with the specific attenuation of the profile at 1 m steps
    z = np.arange(100.0, 8000.0, 1.0)
    p = np.interp(z, prof.height, prof.pressure) / 100.0
    t = np.interp(z, prof.height, prof.temperature)
    e = np.interp(z, prof.height, vapor_pressure_of(prof)) / 100.0
    gamma = gas_specific_attenuation(35.5, p, t, e).A_GAS.values.astype(float)
    cumulative = 2e-3 * np.concatenate([[0], np.cumsum(0.5 * (gamma[1:] + gamma[:-1]))])
    index = np.searchsorted(z, height)
    # the integral from the radar at 100 m (range 0) to the height of the gate
    reference = cumulative[index]
    assert out.PIA_GAS.dims == ("time", "range")
    np.testing.assert_allclose(
        out.PIA_GAS.isel(time=1), reference, rtol=5e-3, atol=2e-4
    )
    # monotonic, positive
    assert (out.PIA_GAS.diff("range") > 0).all()


def test_cloud_liquid_is_integrated_with_the_temperature_of_the_profile():
    prof = profile(constant=True)
    ds = sweep(elevation=90.0, n_gates=120, altitude=0.0, frequency=35.0)
    lwc = xr.zeros_like(ds.DBZ)
    lwc[:, 40:60] = 0.4
    lwc[:, 90] = np.nan  # a missing value counts as no water
    ds["LWC"] = lwc
    out = gas_path_attenuation(ds, prof, lwc="LWC")
    k = float(cloud_liquid_attenuation(35.0, 280.0))
    thickness = 20 * 30.0 * 1e-3  # km
    last = out.PIA_CLOUD_LIQUID.isel(azimuth=0, range=-1)
    # the trapezoid puts half a gate of the layer at each edge: 20 gates of 0.4 g m-3
    assert float(last) == pytest.approx(2 * k * 0.4 * thickness, rel=0.06)
    assert float(out.PIA_CLOUD_LIQUID.isel(azimuth=0, range=20)) == pytest.approx(
        0.0, abs=1e-9
    )
    assert (out.PIA_CLOUD_LIQUID.diff("range") >= -1e-6).all()
    assert out.PIA_GAS.dims == out.PIA_CLOUD_LIQUID.dims


def test_lwc_can_be_a_dataarray_and_the_height_can_be_given():
    prof = profile()
    ds = sweep(elevation=45.0, n_gates=80)
    lwc = xr.full_like(ds.DBZ, 0.1)
    height = beam_center_height(ds.range, 45.0, radar_height=100.0) + 5.0
    a = gas_path_attenuation(ds, prof, lwc=lwc, height=height)
    b = gas_path_attenuation(ds, prof, lwc=lwc, radar_height=105.0)
    # a height of +5 m is a radar 5 m higher, up to the curvature of the Earth
    np.testing.assert_allclose(a.PIA_GAS, b.PIA_GAS, rtol=2e-3)
    ds["h"] = height
    c = gas_path_attenuation(ds, prof, lwc=lwc, height="h")
    np.testing.assert_allclose(c.PIA_GAS, a.PIA_GAS, rtol=1e-6)


def test_frequency_comes_from_the_data_in_ghz_or_hz():
    prof = profile()
    ghz = gas_path_attenuation(sweep(frequency=94.0), prof)
    hz = gas_path_attenuation(sweep(frequency=94.0e9), prof)
    given = gas_path_attenuation(sweep(frequency=35.0), prof, frequency=94.0)
    np.testing.assert_allclose(ghz.PIA_GAS, hz.PIA_GAS)
    np.testing.assert_allclose(ghz.PIA_GAS, given.PIA_GAS)
    assert (ghz.PIA_GAS.isel(azimuth=0, range=-1) > 0).all()
    ka = gas_path_attenuation(sweep(frequency=35.0), prof)
    assert (ka.PIA_GAS < ghz.PIA_GAS).all()


@pytest.mark.parametrize("engine", ENGINES)
def test_a_sweep_gives_the_same_with_dask(engine):
    pytest.importorskip("dask")
    prof = profile()
    ds = sweep(n_az=12)
    ds["LWC"] = xr.full_like(ds.DBZ, 0.05)
    expected = gas_path_attenuation(ds, prof, lwc="LWC", engine=engine)
    lazy = gas_path_attenuation(
        ds.chunk({"azimuth": 4}), prof, lwc="LWC", engine=engine
    )
    assert lazy.PIA_GAS.chunks is not None
    xr.testing.assert_allclose(lazy.compute(), expected)


def test_a_volume_skips_the_sweeps_without_a_field_with_a_range():
    prof = profile()
    good = sweep(elevation=2.0)
    other = sweep(elevation=5.0)
    empty = xr.Dataset({"x": ("t", np.arange(3.0))})
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "/sweep_0": good,
            "/sweep_1": other,
            "/sweep_2": empty,
        }
    )
    out = gas_path_attenuation(tree, prof)
    assert set(out.children) == {"sweep_0", "sweep_1"}
    expected = gas_path_attenuation(good, prof)
    xr.testing.assert_allclose(out["sweep_0"].to_dataset().PIA_GAS, expected.PIA_GAS)
    # a lower elevation stays longer in the humid air
    assert float(out["sweep_0"].PIA_GAS.isel(azimuth=0, range=-1)) > float(
        out["sweep_1"].PIA_GAS.isel(azimuth=0, range=-1)
    )
    nothing = xr.DataTree.from_dict({"/": xr.Dataset(), "/sweep_0": empty})
    with pytest.raises(KeyError, match="no sweep"):
        gas_path_attenuation(nothing, prof)


def test_the_accessor_calls_the_function():
    prof = profile()
    ds = sweep()
    direct = gas_path_attenuation(ds, prof)
    via = ds.radarx.gas_path_attenuation(prof)
    xr.testing.assert_allclose(via, direct)
    da = ds.DBZ.assign_coords(radar_frequency=94.0)
    out = da.radarx.gas_path_attenuation(prof)
    np.testing.assert_allclose(out.PIA_GAS, direct.PIA_GAS)


def test_the_input_is_not_changed_and_provenance_is_written():
    prof = profile()
    ds = sweep()
    before = ds.copy(deep=True)
    out = gas_path_attenuation(ds, prof)
    xr.testing.assert_identical(ds, before)
    assert "P.676" in out.attrs["radarx_method"]
    assert out.attrs["radarx_references"] == "itu-r-p676-13 itu-r-p840-9"


def test_path_attenuation_errors():
    prof = profile()
    ds = sweep()
    with pytest.raises(TypeError, match="obj"):
        gas_path_attenuation(np.zeros(3), prof)
    with pytest.raises(TypeError, match="profile"):
        gas_path_attenuation(ds, {"height": 1})
    with pytest.raises(ValueError, match="height"):
        gas_path_attenuation(
            ds,
            prof.drop_vars("height").assign(pressure=("level", prof.pressure.values)),
        )
    with pytest.raises(ValueError, match="pressure"):
        gas_path_attenuation(ds, prof.drop_vars("pressure"))
    with pytest.raises(ValueError, match="humidity|vapour"):
        gas_path_attenuation(ds, prof.drop_vars("specific_humidity"))
    with pytest.raises(ValueError, match="increasing"):
        gas_path_attenuation(ds, prof.isel(height=slice(None, None, -1)))
    no_frequency = ds.drop_vars("radar_frequency")
    with pytest.raises(ValueError, match="frequency"):
        gas_path_attenuation(no_frequency, prof)
    with pytest.raises(ValueError, match="1 to 1000"):
        gas_path_attenuation(ds, prof, frequency=2000.0)
    with pytest.raises(ValueError, match="range"):
        gas_path_attenuation(
            xr.Dataset({"DBZ": ("azimuth", np.zeros(3))}), prof, dbz="DBZ"
        )
    no_altitude = ds.drop_vars("altitude")
    with pytest.raises(ValueError, match="altitude"):
        gas_path_attenuation(no_altitude, prof)
    no_elevation = ds.drop_vars("elevation")
    with pytest.raises(ValueError, match="elevation"):
        gas_path_attenuation(no_elevation, prof)
    with pytest.raises(ValueError, match="height must"):
        gas_path_attenuation(ds, prof, height=xr.DataArray([1.0, 2.0], dims="x"))
    with pytest.raises(TypeError, match="lwc"):
        gas_path_attenuation(ds, prof, lwc=0.1)
    with pytest.raises(KeyError):
        gas_path_attenuation(ds, prof, lwc="nope")
    backwards = ds.assign_coords(range=ds.range.values[::-1])
    with pytest.raises(ValueError, match="increase"):
        gas_path_attenuation(backwards, prof)
    with pytest.raises(KeyError):
        gas_path_attenuation(ds, prof, dbz="nope")


def test_a_data_array_without_a_name_and_the_zenith_angle():
    prof = profile()
    ds = sweep(elevation=90.0, altitude=100.0)
    da = ds.DBZ.assign_coords(radar_frequency=94.0, zenith_angle=0.0, altitude=100.0)
    out = gas_path_attenuation(da.drop_vars("elevation"), prof)
    assert out.PIA_GAS.dims == da.dims
    plain = ds.DBZ.drop_vars("elevation").assign_coords(
        zenith_angle=0.0, altitude=100.0, radar_frequency=94.0
    )
    out2 = gas_path_attenuation(plain.rename(None), prof)
    np.testing.assert_allclose(out2.PIA_GAS, out.PIA_GAS)


# ---- rain ---------------------------------------------------------------------------

# Rec. ITU-R P.838 specific attenuation at 10 mm/h (the itur package, horizontal
# polarization, 0 degrees elevation): dB km-1
ITU838 = {35.0: 2.709, 94.0: 6.436}


@compiled_only
@pytest.mark.parametrize("frequency, tolerance", [(35.0, 0.12), (94.0, 0.12)])
def test_rain_attenuation_of_a_marshall_palmer_rain_agrees_with_itu_r_p838(
    frequency, tolerance
):
    from radarx.fundamentals.tmatrix import raindrop_scattering

    d = np.arange(0.05, 8.001, 0.05)
    drops = raindrop_scattering(d, 299.792458 / frequency, 0.0, ddelt=1e-5)
    speed = 9.65 - 10.3 * np.exp(-0.6 * d)
    n0 = 8000.0
    rate_per_n0 = (
        6e-4 * np.pi * (np.exp(-4.1 * 10.0**-0.21 * d) * speed * d**3).sum() * 0.05
    )
    scale = 10.0 / (
        n0 * rate_per_n0
    )  # N0 that gives 10 mm/h for the Marshall-Palmer slope
    nd = n0 * scale * np.exp(-4.1 * 10.0**-0.21 * d)
    k = (nd * drops.ah.values).sum() * 0.05
    assert k == pytest.approx(ITU838[frequency], rel=tolerance)


@compiled_only
def test_rain_relation_is_a_power_law_of_the_right_size():
    ka = rain_attenuation_relation(35.0)
    w = rain_attenuation_relation(94.0)
    assert ka.attrs["frequency_GHz"] == 35.0 and ka.attrs["distributions"] > 1000
    # rain of 20 dBZ at Ka band attenuates by about 0.09 dB/km, of 40 dBZ by about 5 dB/km
    assert float(ka.a) * 100.0 ** float(ka.b) == pytest.approx(0.09, rel=0.5)
    assert float(ka.a) * 10000.0 ** float(ka.b) == pytest.approx(4.8, rel=0.35)
    # W band attenuates more at the same Z in light rain
    assert float(w.a) * 100.0 ** float(w.b) > float(ka.a) * 100.0 ** float(ka.b)
    assert 0.5 < float(ka.b) < 1.3 and 0.5 < float(w.b) < 1.3
    assert 0 < float(ka.scatter) < 0.3
    assert ka.a.attrs["units"] == "1"
    again = rain_attenuation_relation(35.0)
    assert float(again.a) == float(ka.a)  # cached
    warm = rain_attenuation_relation(35.0, temperature=20.0)
    assert float(warm.a) != float(ka.a)


def test_rain_relation_errors():
    with pytest.raises(ValueError, match="1 to 1000"):
        rain_attenuation_relation(0.5)
    with pytest.raises(ValueError, match="1 to 1000"):
        rain_attenuation_relation(1500.0)


def rain_column(z_dbz=30.0, a=0.0015, b=0.88, gate=30.0, n=100, start=100.0):
    rng = start + gate * np.arange(n)
    z = xr.DataArray(np.full(n, z_dbz), dims="range", coords={"range": rng})
    k = a * (10.0 ** (z / 10.0)) ** b
    pia = (
        2.0 * (k * gate * 1e-3).cumsum("range") - k * gate * 1e-3
    )  # at the gate centre
    return z, pia, rng


def test_the_hitschfeld_bordan_path_recovers_a_forward_model():
    z, pia, _ = rain_column()
    measured = z - pia
    out = rain_path_attenuation(measured, 35.0, a=0.0015, b=0.88, max_pia=40.0)
    np.testing.assert_allclose(out, pia, atol=1e-3)
    assert out.name == "PIA_RAIN" and out.attrs["units"] == "dB"
    assert out.dtype == np.float32
    assert "k = 0.0015 Z**0.88" == out.attrs["k_Z_relation"]


def test_gates_behind_the_rain_carry_the_whole_path_and_masks_are_respected():
    z, pia, rng = rain_column(n=100)
    measured = (z - pia).copy()
    rain = xr.DataArray(rng < 2000.0, dims="range", coords={"range": rng})
    ice = measured.where(rain, -5.0)  # echo above the rain that must not attenuate
    out = rain_path_attenuation(ice, a=0.0015, b=0.88, mask=rain, max_pia=40.0)
    last_rain = int(rain.values.sum()) - 1
    np.testing.assert_allclose(out.values[last_rain:], out.values[last_rain], atol=1e-6)
    assert float(out.isel(range=0)) < 0.1
    assert (out.diff("range") >= 0).all()
    # the same with a melting level instead of a mask
    height = xr.DataArray(rng, dims="range", coords={"range": rng})
    out2 = rain_path_attenuation(
        ice, a=0.0015, b=0.88, melting=2000.0, height=height, max_pia=40.0
    )
    np.testing.assert_allclose(out2, out, atol=1e-6)
    # a mask given by name in a dataset
    ds = xr.Dataset({"DBZ": ice, "wet": rain})
    named = rain_path_attenuation(ds, a=0.0015, b=0.88, mask="wet", max_pia=40.0)
    np.testing.assert_allclose(named, out, atol=1e-6)


@compiled_only
def test_rain_path_uses_the_fitted_relation_by_default():
    z, pia, _ = rain_column(z_dbz=25.0, n=40)
    ka = rain_attenuation_relation(35.0)
    explicit = rain_path_attenuation(z, 35.0, a=float(ka.a), b=float(ka.b))
    fitted = rain_path_attenuation(z, 35.0)
    np.testing.assert_allclose(fitted, explicit, rtol=1e-5)
    only_a = rain_path_attenuation(z, 35.0, a=2.0 * float(ka.a))
    assert float(only_a.max()) > float(fitted.max())
    # the frequency comes from a dataset
    ds = xr.Dataset({"Z": z, "radar_frequency": 35.0})
    from_data = rain_path_attenuation(ds)
    np.testing.assert_allclose(from_data, fitted, rtol=1e-5)


def test_rain_path_on_a_volume_and_with_dask():
    z, pia, rng = rain_column(n=60)
    sweep0 = xr.Dataset(
        {"DBZH": (("azimuth", "range"), np.tile((z - pia).values, (4, 1)))},
        coords={"azimuth": np.arange(4.0), "range": rng},
    )
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "/sweep_0": sweep0,
            "/sweep_1": xr.Dataset({"x": ("t", np.arange(2.0))}),
        }
    )
    out = rain_path_attenuation(tree, a=0.0015, b=0.88, max_pia=40.0)
    assert list(out.children) == ["sweep_0"]
    expected = rain_path_attenuation(sweep0, a=0.0015, b=0.88, max_pia=40.0)
    xr.testing.assert_allclose(out["sweep_0"].to_dataset().PIA_RAIN, expected)
    pytest.importorskip("dask")
    lazy = rain_path_attenuation(
        sweep0.chunk({"azimuth": 2}), a=0.0015, b=0.88, max_pia=40.0
    )
    assert lazy.chunks is not None
    xr.testing.assert_allclose(lazy.compute(), expected)
    with pytest.raises(KeyError, match="no sweep"):
        rain_path_attenuation(
            xr.DataTree.from_dict(
                {
                    "/": xr.Dataset(),
                    "/sweep_0": xr.Dataset({"x": ("t", np.arange(2.0))}),
                }
            ),
            a=0.0015,
            b=0.88,
        )


def test_rain_path_errors():
    z, _, rng = rain_column(n=20)
    with pytest.raises(TypeError, match="obj"):
        rain_path_attenuation(np.zeros(3), a=1.0, b=1.0)
    with pytest.raises(KeyError):
        rain_path_attenuation(xr.Dataset({"x": z}), a=1.0, b=1.0)
    with pytest.raises(KeyError):
        rain_path_attenuation(xr.Dataset({"DBZ": z}), a=1.0, b=1.0, dbz="nope")
    with pytest.raises(ValueError, match="range"):
        rain_path_attenuation(xr.DataArray(np.zeros(3), dims="x"), a=1.0, b=1.0)
    with pytest.raises(ValueError, match="engine"):
        rain_path_attenuation(z, a=1.0, b=1.0, engine="gpu")
    with pytest.raises(ValueError, match="b must"):
        rain_path_attenuation(z, a=1.0, b=-1.0)
    with pytest.raises(ValueError, match="height"):
        rain_path_attenuation(
            z.drop_vars("range").assign_coords(range=rng), a=1.0, b=1.0, melting=1000.0
        )


def test_the_rain_accessor_calls_the_function():
    z, pia, _ = rain_column(n=30)
    via = (z - pia).radarx.rain_path_attenuation(a=0.0015, b=0.88)
    direct = rain_path_attenuation(z - pia, a=0.0015, b=0.88)
    xr.testing.assert_allclose(via, direct)


# ---- real data: the Cloudnet categorization file of Lindenberg, 1 May 2023 -----------------

CLOUDNET = (
    "https://cloudnet.fmi.fi/api/download/product/231c0919-f399-4fea-aa04-3435754f96b9/"
    "20230501_lindenberg_categorize.nc"
)


@pytest.fixture(scope="module")
def categorize(tmp_path_factory):
    """Cloudnet categorization file (MIRA-35, 35.5 GHz) of Lindenberg, 2023-05-01."""
    import urllib.request

    path = tmp_path_factory.mktemp("cloudnet") / "20230501_lindenberg_categorize.nc"
    error = None
    for _ in range(3):  # the server sometimes ends a transfer early
        try:
            urllib.request.urlretrieve(CLOUDNET, path)  # noqa: S310 - fixed https URL
            return xr.open_dataset(path).load()
        except Exception as err:  # offline, or the file is not available
            error = err
    pytest.skip(f"Cloudnet data unavailable: {error}")


def test_gas_path_attenuation_agrees_with_the_cloudnet_categorization(categorize):
    ds = categorize
    altitude = float(ds.altitude.mean())
    sec = xr.Dataset(
        {"Z": ds.Z.rename(height="range")},
        coords={
            "range": ("range", ds.height.values.astype(float) - altitude),
            "height": ("range", ds.height.values.astype(float)),
            "radar_frequency": float(ds.radar_frequency),
        },
    )
    for index in (200, 1200, 2000):
        model_time = ds.model_time.sel(
            model_time=ds.time.values[index], method="nearest"
        )
        prof = (
            ds[["temperature", "pressure", "q"]]
            .sel(model_time=model_time)
            .rename(model_height="height", q="specific_humidity")
            .drop_vars("model_time")
        )
        out = gas_path_attenuation(sec.isel(time=[index]), prof).PIA_GAS.isel(time=0)
        reference = ds.radar_gas_atten.isel(time=index)
        ok = np.isfinite(reference.values) & (ds.height.values > 400.0)
        assert ok.sum() > 200
        # two absorption models (Rec. ITU-R P.676 and the Liebe model of Cloudnet)
        np.testing.assert_allclose(out.values[ok], reference.values[ok], rtol=0.06)
