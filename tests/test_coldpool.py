#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for radarx.retrieve.coldpool (cold pools, baroclinic generation)."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401  (registers the accessors)
from radarx.io import sounding
from radarx.retrieve import _coldpool_numpy as ref
from radarx.retrieve import coldpool as cp

ENGINES = ["numpy"] + (["compiled"] if cp.HAS_COMPILED_KERNEL else [])
compiled = pytest.mark.skipif(
    not cp.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)
G = 9.80665
DATA = Path(__file__).parent / "io" / "data"


def column(z, b):
    return xr.DataArray(b, dims="height", coords={"height": z})


# ---------------------------------------------------------------------------
# thermodynamics


@pytest.mark.parametrize("engine", ENGINES)
def test_potential_temperatures_known_values(engine):
    ds = xr.Dataset(
        {
            "temperature": ("x", [300.0, 290.0, 273.15]),
            "pressure": ("x", [100000.0, 85000.0, 70000.0]),
            "dewpoint": ("x", [290.0, 280.0, np.nan]),
        },
        coords={"x": [0, 1, 2]},
    )
    out = cp.potential_temperatures(ds, engine=engine)
    kappa = ref.RD / ref.CPD
    th = ds.temperature * (1e5 / ds.pressure) ** kappa
    np.testing.assert_allclose(out.potential_temperature, th, rtol=1e-12)
    # Bolton (1980) eq. 10 for e, mixing ratio, theta_v
    e = 611.2 * np.exp(17.67 * (290.0 - 273.15) / (290.0 - 273.15 + 243.5))
    r = ref.EPS * e / (1e5 - e)
    assert out.mixing_ratio[0] == pytest.approx(r, rel=1e-12)
    assert out.virtual_potential_temperature[0] == pytest.approx(
        300.0 * (1 + r / ref.EPS) / (1 + r), rel=1e-12
    )
    # theta_e of a moist surface parcel: about 340 K (Bolton 1980 eq. 43)
    assert 335 < float(out.equivalent_potential_temperature[0]) < 345
    assert np.isnan(out.virtual_potential_temperature[2])
    assert np.isfinite(out.potential_temperature[2])
    assert out.potential_temperature.attrs["units"] == "K"
    assert out.potential_temperature.attrs["standard_name"] == (
        "air_potential_temperature"
    )
    assert "x" in out.coords


def test_potential_temperatures_from_relative_humidity():
    t = np.array([300.0, 285.0])
    td = np.array([293.0, 280.0])
    rh = sounding.relative_humidity_from_dewpoint(
        xr.DataArray(t, dims="i"), xr.DataArray(td, dims="i")
    )
    a = cp.potential_temperatures(
        xr.Dataset(
            {
                "temperature": ("i", t),
                "pressure": ("i", [1e5, 9e4]),
                "dewpoint": ("i", td),
            }
        )
    )
    b = cp.potential_temperatures(
        xr.Dataset(
            {
                "temperature": ("i", t),
                "pressure": ("i", [1e5, 9e4]),
                "relative_humidity": ("i", rh.values),
            }
        )
    )
    np.testing.assert_allclose(
        a.virtual_potential_temperature, b.virtual_potential_temperature, rtol=1e-6
    )
    # no humidity at all: only theta is finite
    c = cp.potential_temperatures(
        xr.Dataset({"temperature": ("i", t), "pressure": ("i", [1e5, 9e4])})
    )
    assert np.isnan(c.mixing_ratio).all() and np.isfinite(c.potential_temperature).all()


def test_potential_temperatures_unit_checks():
    ds = xr.Dataset({"temperature": ("i", [25.0]), "pressure": ("i", [1e5])})
    with pytest.raises(ValueError, match="K"):
        cp.potential_temperatures(ds)
    ds = xr.Dataset({"temperature": ("i", [298.0]), "pressure": ("i", [1000.0])})
    with pytest.raises(ValueError, match="Pa"):
        cp.potential_temperatures(ds)
    ds = xr.Dataset(
        {
            "temperature": ("i", [298.0]),
            "pressure": ("i", [1e5]),
            "relative_humidity": ("i", [80.0]),
        }
    )
    with pytest.raises(ValueError, match="fraction"):
        cp.potential_temperatures(ds)


def test_potential_temperatures_vs_metpy():
    mpcalc = pytest.importorskip("metpy.calc")
    from metpy.units import units

    t = np.array([300.0, 290.0, 280.0])
    p = np.array([100000.0, 90000.0, 80000.0])
    td = np.array([295.0, 285.0, 270.0])
    ds = xr.Dataset(
        {"temperature": ("i", t), "pressure": ("i", p), "dewpoint": ("i", td)}
    )
    out = cp.potential_temperatures(ds)
    the = mpcalc.equivalent_potential_temperature(
        p * units.Pa, t * units.K, td * units.K
    ).m
    np.testing.assert_allclose(out.equivalent_potential_temperature, the, atol=0.05)
    th = mpcalc.potential_temperature(p * units.Pa, t * units.K).m
    np.testing.assert_allclose(out.potential_temperature, th, rtol=1e-4)


# ---------------------------------------------------------------------------
# buoyancy and perturbations


def test_buoyancy_with_condensate():
    b = cp.buoyancy(xr.DataArray([297.0, 300.0]), 300.0, condensate=0.001)
    np.testing.assert_allclose(b, [G * -3 / 300 - G * 1e-3, -G * 1e-3])
    assert b.attrs["units"] == "m s-2"


def station_network():
    """Two stations, one cooling by 5 K after 01:00, one unchanged."""
    time = np.arange("2022-03-31T00:00", "2022-03-31T02:00", 60, dtype="datetime64[s]")
    n = time.size
    t = np.full((2, n), 295.0)
    t[0, time >= np.datetime64("2022-03-31T01:00")] = 290.0
    return xr.Dataset(
        {
            "temperature": (("station", "time"), t),
            "pressure": (("station", "time"), np.full((2, n), 99000.0)),
            "dewpoint": (("station", "time"), np.full((2, n), 288.0)),
        },
        coords={
            "station": ["A", "B"],
            "time": time.astype("datetime64[ns]"),
            "latitude": ("station", [33.7, 33.8]),
        },
    )


def test_cold_pool_perturbation_time_window():
    ds = station_network()
    out = cp.cold_pool_perturbation(ds, slice("2022-03-31T00:00", "2022-03-31T00:30"))
    dth = out["potential_temperature_perturbation"]
    assert dth.dims == ("station", "time")
    assert "latitude" in out.coords
    late = dth.sel(time=slice("2022-03-31T01:00", None))
    np.testing.assert_allclose(late.sel(station="B"), 0.0, atol=1e-12)
    expected = -5.0 * (1e5 / 99000.0) ** (ref.RD / ref.CPD)
    np.testing.assert_allclose(late.sel(station="A"), expected, rtol=1e-12)
    assert float(out.buoyancy.sel(station="A").min()) < -0.15
    assert out.buoyancy.attrs["units"] == "m s-2"
    assert "temperature_perturbation" in out


def test_cold_pool_perturbation_reference_sounding_interpolated():
    z = np.arange(0.0, 3001.0, 100.0)
    env = xr.Dataset(
        {
            "temperature": ("height", 300.0 - 0.0065 * z),
            "pressure": ("height", 1e5 * np.exp(-z / 8000.0)),
            "dewpoint": ("height", 290.0 - 0.002 * z),
        },
        coords={"height": z},
    )
    cold = env.copy()
    cold["temperature"] = env.temperature - 4.0 * np.clip(1 - z / 1000.0, 0, None)
    coarse = env.isel(height=slice(None, None, 2))
    out = cp.cold_pool_perturbation(cold, coarse)
    np.testing.assert_allclose(
        out.temperature_perturbation, cold.temperature - env.temperature, atol=1e-9
    )
    with pytest.raises(TypeError):
        cp.cold_pool_perturbation(cold, 3.0)


# ---------------------------------------------------------------------------
# cold-pool intensity


@pytest.mark.parametrize("engine", ENGINES)
def test_cold_pool_intensity_linear_profile(engine):
    # B = -b0 (1 - z/H): C^2 = b0 H exactly with the trapezoidal rule
    b0, h = 0.1, 2000.0
    z = np.linspace(0, 5000, 51)
    b = -b0 * np.clip(1 - z / h, 0, None) + 0.01 * (z > h)
    out = cp.cold_pool_intensity(column(z, b), engine=engine)
    assert float(out.cold_pool_intensity) == pytest.approx(np.sqrt(b0 * h), rel=1e-12)
    assert float(out.cold_pool_depth) == pytest.approx(h, rel=1e-12)
    assert bool(out.cold_pool_top_found)
    assert out.cold_pool_intensity.attrs["units"] == "m s-1"


@pytest.mark.parametrize("engine", ENGINES)
def test_cold_pool_intensity_uniform_and_threshold(engine):
    z = np.arange(0.0, 3001.0, 10.0)
    b = np.where(z < 1000, -0.05, -0.001)
    out = cp.cold_pool_intensity(column(z, b), threshold=-0.01, engine=engine)
    # integral of 0.05 over 990 m + half the crossing layer: crossing at the
    # threshold inside the 990-1000 m layer
    zc = 990 + (-0.01 + 0.05) / (-0.001 + 0.05) * 10
    integral = 0.05 * 990 + 0.5 * (0.05 + 0.01) * (zc - 990)
    assert float(out.cold_pool_intensity) == pytest.approx(np.sqrt(2 * integral))
    assert float(out.cold_pool_depth) == pytest.approx(zc)
    # without threshold the deficit reaches the top: open top, lower bound
    out = cp.cold_pool_intensity(column(z, b), engine=engine)
    assert not bool(out.cold_pool_top_found)
    assert float(out.cold_pool_depth) == pytest.approx(3000.0)


@pytest.mark.parametrize("engine", ENGINES)
def test_cold_pool_intensity_fixed_depth_bottom_and_edge_cases(engine):
    z = np.arange(0.0, 2001.0, 100.0)
    b = np.full(z.size, -0.02)
    b[:3] = np.nan  # gaps at the bottom
    out = cp.cold_pool_intensity(column(z, b), depth=1000.0, engine=engine)
    assert float(out.cold_pool_intensity) == pytest.approx(np.sqrt(2 * 0.02 * 1000))
    out = cp.cold_pool_intensity(column(z, b), depth=5000.0, engine=engine)
    assert np.isnan(out.cold_pool_intensity)  # profile too short
    out = cp.cold_pool_intensity(column(z, b), bottom=1000.0, engine=engine)
    assert float(out.cold_pool_depth) == pytest.approx(1000.0)
    warm = cp.cold_pool_intensity(column(z, -b), engine=engine)
    assert float(warm.cold_pool_intensity) == 0 and float(warm.cold_pool_depth) == 0
    empty = cp.cold_pool_intensity(column(z, b * np.nan), engine=engine)
    assert np.isnan(empty.cold_pool_intensity) and not bool(empty.cold_pool_top_found)


@pytest.mark.parametrize("engine", ENGINES)
def test_cold_pool_intensity_grid_and_descending(engine):
    zc = np.arange(0.0, 4001.0, 250.0)
    depth = xr.DataArray([[500.0, 1000.0], [2000.0, 3000.0]], dims=("y", "x"))
    b = -0.03 * (1 - xr.DataArray(zc, dims="z", coords={"z": zc}) / depth).clip(min=0)
    b = b.transpose("z", "y", "x").assign_coords(y=[0.0, 1.0], x=[5.0, 6.0])
    out = cp.cold_pool_intensity(b, dim="z", engine=engine)
    assert out.cold_pool_intensity.dims == ("y", "x")
    np.testing.assert_allclose(out.cold_pool_depth, depth)
    np.testing.assert_allclose(out.cold_pool_intensity, np.sqrt(0.03 * depth))
    flipped = cp.cold_pool_intensity(b.isel(z=slice(None, None, -1)), dim="z")
    np.testing.assert_allclose(flipped.cold_pool_depth, depth)
    # per-column bottom and height given as a variable
    out2 = cp.cold_pool_intensity(
        b.rename(z="level").drop_vars("level"),
        dim="level",
        height=xr.DataArray(zc, dims="level"),
        bottom=xr.DataArray([0.0, 0.0], dims="x"),
        engine=engine,
    )
    np.testing.assert_allclose(out2.cold_pool_depth, depth)
    with pytest.raises(ValueError, match="height"):
        cp.cold_pool_intensity(b.rename(z="level").drop_vars("level"), dim="level")
    with pytest.raises(ValueError, match="dimension"):
        cp.cold_pool_intensity(b, dim="height", height=b["z"])


def test_surface_and_pressure_estimates_consistent():
    # linear cold pool: C^2 = b0 H = 2 dp / rho for a hydrostatic excess
    b0, h, rho = 0.08, 1500.0, 1.15
    c_int = np.sqrt(b0 * h)
    dp = xr.DataArray([rho * 0.5 * b0 * h, -50.0])
    np.testing.assert_allclose(
        cp.cold_pool_intensity_from_pressure(dp, density=rho), [c_int, 0.0]
    )
    bs = xr.DataArray([-b0, 0.02])
    np.testing.assert_allclose(cp.cold_pool_intensity_from_surface(bs, h), [c_int, 0.0])
    np.testing.assert_allclose(
        cp.cold_pool_intensity_from_surface(bs, h, shape="uniform")[0],
        np.sqrt(2) * c_int,
    )
    with pytest.raises(ValueError):
        cp.cold_pool_intensity_from_surface(bs, h, shape="cubic")
    r = cp.rkw_ratio(xr.DataArray([20.0]), 10.0)
    assert float(r[0]) == 2.0 and r.name == "rkw_ratio"


# ---------------------------------------------------------------------------
# baroclinic generation


@pytest.mark.parametrize("engine", ENGINES)
def test_baroclinic_generation_linear_field(engine):
    x = np.arange(0.0, 10001.0, 500.0)
    y = np.arange(0.0, 6001.0, 1000.0)
    z = np.array([250.0, 500.0])
    ax, ay = -2e-5, 1e-5  # dB/dx, dB/dy (s-2)
    b = (
        ax * xr.DataArray(x, dims="x", coords={"x": x})
        + ay * xr.DataArray(y, dims="y", coords={"y": y})
        + 0 * xr.DataArray(z, dims="z", coords={"z": z})
    ).transpose("z", "y", "x")
    b[0, 2, 3] = np.nan  # a gap: neighbours use one-sided differences
    out = cp.baroclinic_generation(b, engine=engine)
    ok = np.isfinite(b.values)
    np.testing.assert_allclose(out.dB_dx.values[ok], ax, rtol=1e-9)
    np.testing.assert_allclose(out.vorticity_y_generation.values[ok], -ax, rtol=1e-9)
    np.testing.assert_allclose(out.vorticity_x_generation.values[ok], ay, rtol=1e-9)
    assert np.isnan(out.dB_dx[0, 2, 3])
    assert out.dB_dx.dims == b.dims
    assert out.horizontal_vorticity_generation.attrs["units"] == "s-2"
    # storm-relative wind along +x: streamwise = x component
    u = xr.full_like(b, 10.0)
    v = xr.zeros_like(b)
    sw = cp.baroclinic_generation(b, u=u, v=v, storm_motion=(5.0, 0.0), engine=engine)
    np.testing.assert_allclose(sw.streamwise_vorticity_generation.values[ok], ay)
    np.testing.assert_allclose(sw.crosswise_vorticity_generation.values[ok], -ax)
    motion = xr.Dataset({"u": 20.0, "v": 0.0})
    sw2 = cp.baroclinic_generation(b, u=u, v=v, storm_motion=motion, engine=engine)
    np.testing.assert_allclose(sw2.streamwise_vorticity_generation.values[ok], -ay)


def test_baroclinic_generation_cross_section_and_errors():
    x = np.arange(0.0, 5001.0, 250.0)
    b = xr.DataArray(
        np.tile(-1e-5 * x, (3, 1)), dims=("z", "x"), coords={"x": x, "z": [0, 1, 2]}
    )
    out = cp.baroclinic_generation(b)
    np.testing.assert_allclose(out.vorticity_y_generation, 1e-5)
    assert "dB_dy" not in out and out.dB_dx.dims == ("z", "x")
    with pytest.raises(ValueError, match="dimension"):
        cp.baroclinic_generation(b, x="lon")


def test_baroclinic_generation_single_point_rows():
    b = xr.DataArray([[1.0]], dims=("y", "x"))
    assert np.isnan(cp.baroclinic_generation(b, engine="numpy").dB_dx).all()


# ---------------------------------------------------------------------------
# engines, accessors, real sounding


@compiled
def test_engines_agree():
    rng = np.random.default_rng(1)
    t = 270 + 30 * rng.random(5000)
    p = 5e4 + 5e4 * rng.random(5000)
    td = t - 20 * rng.random(5000)
    td[::7] = np.nan
    np.testing.assert_allclose(
        cp._coldpool.thermo(t, p, td), ref.thermo(t, p, td), rtol=1e-12, equal_nan=True
    )
    z = np.tile(np.sort(rng.random(60)) * 4000, (300, 1))
    b = -0.05 + 0.08 * np.sort(rng.random((300, 60)), axis=1)
    b[rng.random(b.shape) < 0.05] = np.nan
    bot = np.where(rng.random(300) < 0.3, 200.0, np.nan)
    for top in (np.full(300, np.nan), np.full(300, 1500.0)):
        np.testing.assert_allclose(
            cp._coldpool.cold_pool(z, b, bot, -0.01, top),
            ref.cold_pool(z, b, bot, -0.01, top),
            rtol=1e-10,
            equal_nan=True,
        )
    u, v = 10 * rng.standard_normal((2, 300, 60))
    u[rng.random(u.shape) < 0.05] = np.nan
    g = np.where(rng.random(300) < 0.5, np.nan, 100.0)
    cu, cv = rng.standard_normal((2, 300))
    np.testing.assert_allclose(
        cp._coldpool.profile(z, u, v, g, 0.0, 1000.0, cu, cv),
        ref.profile(z, u, v, g, 0.0, 1000.0, cu, cv),
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
    )
    f = rng.standard_normal((3, 20, 30))
    f[rng.random(f.shape) < 0.1] = np.nan
    x, y = np.cumsum(rng.random(30)), np.cumsum(rng.random(20))
    np.testing.assert_allclose(
        cp._coldpool.gradient(f, x, y),
        ref.gradient(f, x, y),
        rtol=1e-12,
        equal_nan=True,
    )
    vr = rng.standard_normal((50, 360))
    az = np.tile(np.linspace(0, 2 * np.pi, 360, endpoint=False), (50, 1))
    vr[:10, 100:] = np.nan
    np.testing.assert_allclose(
        cp._coldpool.vad(vr, az, np.full(50, 0.99), 50, 0.1),
        ref.vad(vr, az, np.full(50, 0.99), 50, 0.1),
        rtol=1e-9,
        atol=1e-12,
        equal_nan=True,
    )
    # threads change nothing
    np.testing.assert_array_equal(
        cp._coldpool.gradient(f, x, y, n_threads=1),
        cp._coldpool.gradient(f, x, y, n_threads=4),
    )


def test_engine_argument():
    with pytest.raises(ValueError, match="engine"):
        cp._use_compiled("fortran")
    if not cp.HAS_COMPILED_KERNEL:  # pragma: no cover
        with pytest.raises(ImportError):
            cp._use_compiled("compiled")


def test_accessors():
    ds = station_network()
    th = ds.radarx.potential_temperatures()
    assert "virtual_potential_temperature" in th
    z = np.linspace(0, 3000, 31)
    b = column(z, -0.02 * np.clip(1 - z / 1000, 0, None))
    assert float(b.radarx.cold_pool_intensity().cold_pool_depth) == pytest.approx(1000)
    grid = xr.DataArray(
        np.zeros((3, 4)), dims=("y", "x"), coords={"y": [0, 1, 2], "x": [0, 1, 2, 3]}
    )
    assert "vorticity_x_generation" in grid.radarx.baroclinic_generation()


def test_real_sounding_cold_pool_jan():
    """A cooled copy of the JAN 00 UTC 31 March 2022 sounding."""
    env = sounding.open_sounding_file(DATA / "uwyo_72235_2022033100.csv")
    env = env.where(env.height < env.height.min() + 6000, drop=True)
    agl = env.height - env.height.min()
    cold = env.copy()
    cold["temperature"] = env.temperature - 6.0 * (1 - agl / 1500.0).clip(min=0)
    pert = cp.cold_pool_perturbation(cold, env)
    out = cp.cold_pool_intensity(pert.buoyancy)
    # B ~ -g 6 K / 300 K at the ground, decreasing linearly over 1.5 km:
    # C ~ sqrt(0.196 * 1500) ~ 17 m/s
    assert 14 < float(out.cold_pool_intensity) < 19
    assert float(out.cold_pool_depth) == pytest.approx(1500, abs=150)
