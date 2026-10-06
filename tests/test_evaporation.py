#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for radarx.retrieve.evaporation."""

import importlib
import math

import numpy as np
import pytest
import xarray as xr
from scipy.integrate import trapezoid

import radarx  # noqa: F401
from radarx.retrieve import (
    drop_evaporation_rate,
    evaporation,
    integrate_evaporation,
)

evap = importlib.import_module("radarx.retrieve.evaporation")

ENGINES = ["numpy"] + (["compiled"] if evap.HAS_COMPILED_KERNEL else [])
needs_kernel = pytest.mark.skipif(
    not evap.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)


def _mp(rain_rate):
    """Marshall-Palmer DSD (N0 = 8000 m-3 mm-1)."""
    return xr.Dataset({"N0": 8000.0, "MU": 0.0, "LAMBDA": 4.1 * rain_rate**-0.21})


def _random(n, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(-0.9, 10.0, n)
    lam = rng.uniform(0.5, 15.0, n) + 0.3 * np.maximum(mu, 0)
    dm = (mu + 4) / lam
    nw = 10 ** rng.uniform(2, 5, n)
    # N0 from Nw and Dm (Testud et al. 2001 normalization)
    f = 6 / 4**4 * (4 + mu) ** (mu + 4) / np.exp(evap.gammaln(mu + 4))
    n0 = nw * f * dm ** (-mu)
    t = rng.uniform(265.0, 310.0, n)
    p = rng.uniform(5.0e4, 1.03e5, n)
    rh = rng.uniform(0.05, 1.1, n)
    q = evap._from_rh(t, p, rh)
    return n0, mu, lam, t, p, q


# --------------------------------------------------------------------------
# physics
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_saturated_air_does_not_evaporate(engine):
    out = evaporation(
        _mp(5.0),
        temperature=290.0,
        pressure=9.0e4,
        relative_humidity=xr.DataArray([0.6, 1.0, 1.05], dims="rh"),
        engine=engine,
    )
    assert out.EVAPORATION_RATE[0] > 0 and out.COOLING_RATE[0] > 0
    assert out.DBZ_TENDENCY[0] < 0
    np.testing.assert_allclose(out.EVAPORATION_RATE[1], 0.0, atol=1e-15)
    np.testing.assert_allclose(out.COOLING_RATE_HOURLY[1], 0.0, atol=1e-9)
    # supersaturated: drops grow, the air warms
    assert out.EVAPORATION_RATE[2] < 0 and out.COOLING_RATE[2] < 0
    np.testing.assert_allclose(out.SATURATION_DEFICIT, [-0.4, 0.0, 0.05], atol=1e-12)
    np.testing.assert_allclose(out.COOLING_RATE_HOURLY, out.COOLING_RATE * 3600)


def test_bulk_equals_integral_of_single_drops():
    """The closed form equals the numerical integral of dm/dt over N(D)."""
    t, p = 288.0, 8.5e4
    q = float(evap._from_rh(t, p, 0.65))
    d = np.linspace(1e-4, 40.0, 400001)
    for n0, mu, lam in ((8000.0, 0.0, 2.5), (3.0e4, 2.5, 4.0), (500.0, -0.5, 1.5)):
        nd = n0 * d**mu * np.exp(-lam * d)
        single = drop_evaporation_rate(d, t, p, q).mass_rate.values
        rho = evap._air(t, p, q)["rho"]
        expected = -trapezoid(single * nd, d) / rho
        ds = xr.Dataset({"N0": n0, "MU": mu, "LAMBDA": lam})
        out = evaporation(ds, temperature=t, pressure=p, specific_humidity=q)
        np.testing.assert_allclose(out.EVAPORATION_RATE, expected, rtol=2e-3)
        # reflectivity tendency: 6 int D^5 dD/dt N dD over Z
        dd = drop_evaporation_rate(d, t, p, q).diameter_rate.values
        zrate = trapezoid(6 * d**5 * dd * nd, d)
        z = trapezoid(d**6 * nd, d)
        np.testing.assert_allclose(
            out.DBZ_TENDENCY, 3600 * 10 / math.log(10) * zrate / z, rtol=2e-3
        )


@pytest.mark.parametrize("engine", ENGINES)
def test_monodisperse_limit(engine):
    """A very narrow gamma DSD evaporates like its drops (mean diameter)."""
    t, p = 293.15, 9.0e4
    q = float(evap._from_rh(t, p, 0.7))
    nt = 1000.0  # drops per m3
    for d1 in (0.5, 1.0, 2.0, 3.0):
        mu = 150.0
        lam = (mu + 1.0) / d1
        n0 = nt * math.exp((mu + 1) * math.log(lam) - evap.gammaln(mu + 1))
        ds = xr.Dataset({"N0": n0, "MU": mu, "LAMBDA": lam})
        out = evaporation(
            ds, temperature=t, pressure=p, specific_humidity=q, engine=engine
        )
        single = drop_evaporation_rate(d1, t, p, q).mass_rate
        rho = evap._air(t, p, q)["rho"]
        np.testing.assert_allclose(out.EVAPORATION_RATE, -nt * single / rho, rtol=5e-3)


def test_scaling_with_intercept():
    """Rates are linear in N0; the dBZ tendency does not depend on N0."""
    ds = xr.Dataset(
        {
            "N0": ("n", [1.0e3, 1.0e4]),
            "MU": ("n", [2.0, 2.0]),
            "LAMBDA": ("n", [3.0, 3.0]),
        }
    )
    out = evaporation(ds, temperature=295.0, pressure=9.5e4, relative_humidity=0.5)
    np.testing.assert_allclose(out.EVAPORATION_RATE[1] / out.EVAPORATION_RATE[0], 10.0)
    np.testing.assert_allclose(out.DBZ_TENDENCY[1], out.DBZ_TENDENCY[0])
    # drier and warmer air evaporates more
    dry = evaporation(ds, temperature=295.0, pressure=9.5e4, relative_humidity=0.3)
    warm = evaporation(ds, temperature=300.0, pressure=9.5e4, relative_humidity=0.5)
    assert (dry.COOLING_RATE > out.COOLING_RATE).all()
    assert (warm.COOLING_RATE > out.COOLING_RATE).all()


def test_fall_speed_fit_follows_atlas():
    """The closed-form fall speed is within 10 % of Atlas et al. (1973)."""
    a, b, f = evap.FALL_SPEED["atlas1973"]
    d = np.linspace(0.5, 7.0, 500)
    atlas = 9.65 - 10.3 * np.exp(-0.6 * d)
    fit = a * d**b * np.exp(-f * d)
    assert np.abs(fit / atlas - 1).max() < 0.1
    assert np.sqrt(np.mean((fit / atlas - 1) ** 2)) < 0.025


def _dstar(depth, rh, dz=10.0):
    """Diameter (mm) that just evaporates falling ``depth`` (Li and Srivastava 2001).

    Layer top at 600 hPa and 0 degC, lapse rate 9 K/km, constant RH (their
    Fig. 2); drops of many initial sizes are followed down at once.
    """
    a, b, f = evap.FALL_SPEED["atlas1973"]
    d = np.linspace(0.05, 1.5, 581)
    t, p = 273.15, 6.0e4
    for _ in range(int(round(depth / dz))):
        q = evap._from_rh(t, p, rh)
        rate = drop_evaporation_rate(np.maximum(d, 1e-6), t, p, q).diameter_rate.values
        v = (
            (evap.RHO0 / evap._air(t, p, q)["rho"]) ** 0.4
            * a
            * np.maximum(d, 0) ** b
            * np.exp(-f * d)
        )
        d = np.where(d > 1e-3, d + rate / np.maximum(v, 1e-3) * dz, 0.0)
        t_new = t + 9.0e-3 * dz
        p = p * math.exp(9.80665 * dz / (evap.RD * 0.5 * (t + t_new)))
        t = t_new
    init = np.linspace(0.05, 1.5, 581)
    return init[np.argmax(d > 1e-3)]


def test_li_srivastava_dstar():
    """D* near 0.04 cm for 1 km and 0.06 cm for 1.6 km at 70 % RH (LS01, Fig. 2).

    radarx gives about 0.05 and 0.066 cm: the Atlas et al. (1973) fall speeds
    are too slow for drops below 0.5 mm, which then evaporate a little faster
    on their way down than with the Gunn and Kinzer speeds of LS01.
    """
    assert 0.35 < _dstar(1000.0, 0.7) < 0.6
    assert 0.5 < _dstar(1600.0, 0.7) < 0.8
    assert _dstar(1000.0, 0.5) > _dstar(1000.0, 0.7)


@pytest.mark.parametrize("engine", ENGINES)
def test_missing_and_empty(engine):
    ds = xr.Dataset(
        {
            "N0": ("g", [0.0, np.nan, 8000.0, 8000.0]),
            "MU": ("g", [0.0, 0.0, 0.0, 0.0]),
            "LAMBDA": ("g", [2.0, 2.0, 2.0, 2.0]),
        }
    )
    t = xr.DataArray([290.0, 290.0, 290.0, np.nan], dims="g")
    out = evaporation(
        ds, temperature=t, pressure=9e4, relative_humidity=0.5, engine=engine
    )
    assert float(out.EVAPORATION_RATE[0]) == 0.0
    assert np.isnan(out.EVAPORATION_RATE[1])
    assert out.EVAPORATION_RATE[2] > 0
    assert np.isnan(out.EVAPORATION_RATE[3])


@needs_kernel
def test_compiled_matches_numpy():
    n0, mu, lam, t, p, q = _random(20000)
    n0[::97] = np.nan
    n0[::101] = 0.0
    ds = xr.Dataset({"N0": ("g", n0), "MU": ("g", mu), "LAMBDA": ("g", lam)})
    kw = dict(temperature=xr.DataArray(t, dims="g"), pressure=xr.DataArray(p, dims="g"))
    kw["specific_humidity"] = xr.DataArray(q, dims="g")
    a = evaporation(ds, engine="compiled", n_threads=3, **kw)
    b = evaporation(ds, engine="numpy", **kw)
    for name in a.data_vars:
        np.testing.assert_allclose(a[name], b[name], rtol=1e-10, atol=0, equal_nan=True)
    # the Lanczos log-gamma of the kernel
    assert abs(
        float(a.EVAPORATION_RATE[5]) - float(b.EVAPORATION_RATE[5])
    ) <= 1e-10 * abs(float(b.EVAPORATION_RATE[5]))


@needs_kernel
def test_kernel_argument_checks():
    k = evap._evaporation
    one = np.ones(3)
    with pytest.raises(ValueError):
        k.rates(np.ones((2, 2)), one, one, one, one, one, 1, 1, 0, 0.78, 0.3, 1.2)
    with pytest.raises(ValueError):
        k.rates(one, np.ones(2), one, one, one, one, 1, 1, 0, 0.78, 0.3, 1.2)
    two = np.ones((2, 3))
    with pytest.raises(ValueError):
        k.integrate(
            one, one, one, one, one, one, np.ones(1), 60, 1, 1, 0, 0.78, 0.3, 1.2
        )
    with pytest.raises(ValueError):
        k.integrate(
            two,
            np.ones((2, 2)),
            two,
            one,
            one,
            one,
            np.ones(1),
            60,
            1,
            1,
            0,
            0.78,
            0.3,
            1.2,
        )
    with pytest.raises(ValueError):
        k.integrate(
            two, two, two, one, one, one, np.ones(2), 60, 1, 1, 0, 0.78, 0.3, 1.2
        )
    with pytest.raises(ValueError):
        k.integrate(
            two, two, two, one, one, one, np.ones(1), 0, 1, 1, 0, 0.78, 0.3, 1.2
        )


# --------------------------------------------------------------------------
# time integration
# --------------------------------------------------------------------------


def _series(nt=5, nh=4, dt_min=5.0):
    time = np.datetime64("2022-03-30T23:00") + np.arange(nt) * np.timedelta64(
        int(dt_min * 60), "s"
    )
    height = np.linspace(200.0, 2000.0, nh)
    shape = (nt, nh)
    lam = np.full(shape, 4.1 * 5.0**-0.21)
    n0 = np.full(shape, 8000.0)
    n0[2, 0] = np.nan  # a missing DSD: no rain
    return xr.Dataset(
        {
            "N0": (("time", "height"), n0),
            "MU": (("time", "height"), np.zeros(shape)),
            "LAMBDA": (("time", "height"), lam),
        },
        coords={"time": time, "height": height},
    )


def _profile(rh=0.6):
    h = np.linspace(0.0, 4000.0, 41)
    t = 295.0 - 6.5e-3 * h
    p = 1.0e5 * np.exp(-h / 8000.0)
    return xr.Dataset(
        {
            "temperature": ("height", t),
            "pressure": ("height", p),
            "specific_humidity": ("height", evap._from_rh(t, p, rh)),
        },
        coords={"height": h},
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_integration_cools_and_moistens(engine):
    series = _series()
    out = integrate_evaporation(series, _profile(), engine=engine)
    assert out.temperature.dims == ("time", "height")
    dtemp = out.TEMPERATURE_CHANGE.isel(time=-1)
    assert (dtemp < 0).all()
    assert (out.relative_humidity.diff("time") >= -1e-12).all()
    assert (out.relative_humidity <= 1.0 + 1e-9).all()
    # moist enthalpy is (nearly) conserved: cp dT + Lv dq = 0
    pres = _profile().interp(height=series.height).pressure.values
    air = evap._air(out.temperature.values[0], pres, out.specific_humidity.values[0])
    dq = out.specific_humidity.isel(time=-1) - out.specific_humidity.isel(time=0)
    balance = air["cp"] * dtemp + air["lv"] * dq
    np.testing.assert_allclose(
        balance, 0.0, atol=2e-3 * 1005.0 * abs(float(dtemp.min()))
    )
    # first 5 min: a little less than the initial rate times 300 s, as the
    # air cools and moistens (the rate itself drops accordingly)
    first = out.COOLING_RATE.isel(time=0) * 300.0
    change = -out.TEMPERATURE_CHANGE.isel(time=1)
    assert ((change < first) & (change > 0.9 * first)).all()
    assert (out.COOLING_RATE.isel(time=1) < out.COOLING_RATE.isel(time=0)).all()
    one_step = integrate_evaporation(
        series.isel(time=[0, 1]), _profile(), max_step=1.0, engine=engine
    )
    np.testing.assert_allclose(one_step.temperature[1], out.temperature[1], rtol=1e-6)
    # the gate without rain in the third interval does not change then
    np.testing.assert_allclose(out.temperature[3, 0], out.temperature[2, 0])


@pytest.mark.parametrize("engine", ENGINES)
def test_integration_stops_at_saturation(engine):
    series = _series(nt=3, nh=2, dt_min=600.0)  # 10 h of heavy evaporation
    series["N0"] = series.N0.fillna(8000.0) * 20
    out = integrate_evaporation(series, _profile(0.5), max_step=30.0, engine=engine)
    rh = out.relative_humidity.isel(time=-1)
    np.testing.assert_allclose(rh, 1.0, atol=2e-3)
    assert (
        out.EVAPORATION_RATE.isel(time=-1) < 1e-3 * out.EVAPORATION_RATE.isel(time=0)
    ).all()


@needs_kernel
def test_integration_compiled_matches_numpy():
    n = 3000
    n0, mu, lam, t, p, q = _random(n * 4, seed=3)
    q = np.minimum(q, evap._from_rh(t, p, 0.95))
    shape = (4, n)
    ds = xr.Dataset(
        {
            "N0": (("time", "g"), n0.reshape(shape)),
            "MU": (("time", "g"), mu.reshape(shape)),
            "LAMBDA": (("time", "g"), lam.reshape(shape)),
        },
        coords={"time": [0.0, 300.0, 400.0, 1000.0]},
    )
    kw = dict(
        temperature=xr.DataArray(t[:n], dims="g"),
        pressure=xr.DataArray(p[:n], dims="g"),
        specific_humidity=xr.DataArray(q[:n], dims="g"),
        max_step=45.0,
    )
    a = integrate_evaporation(ds, engine="compiled", **kw)
    b = integrate_evaporation(ds, engine="numpy", **kw)
    # identical to rounding; where the air reaches saturation the rates fall
    # to ~0 and only agree to a tiny fraction of their typical size
    for name in a.data_vars:
        scale = float(np.nanmax(np.abs(b[name])))
        np.testing.assert_allclose(
            a[name], b[name], rtol=1e-9, atol=1e-6 * scale, equal_nan=True
        )


def test_integration_errors():
    series = _series()
    with pytest.raises(ValueError, match="dimension"):
        integrate_evaporation(series, _profile(), time_dim="t")
    with pytest.raises(ValueError, match="max_step"):
        integrate_evaporation(series, _profile(), max_step=0)
    with pytest.raises(ValueError, match="increase"):
        integrate_evaporation(series.isel(time=[1, 0, 2]), _profile())
    with pytest.raises(TypeError):
        integrate_evaporation(series.N0, _profile())
    # an environment with a time dimension starts from its first time
    env = xr.concat([_profile(), _profile(0.9)], dim="time")
    out = integrate_evaporation(
        series, env.drop_vars("height").assign_coords(height=env.height)
    )
    ref = integrate_evaporation(series, _profile())
    np.testing.assert_allclose(out.temperature, ref.temperature)


# --------------------------------------------------------------------------
# environments, volumes, accessors, errors
# --------------------------------------------------------------------------


def test_profile_interpolated_to_qvp_heights():
    series = _series()
    out = evaporation(series, _profile())
    env = _profile().interp(height=series.height)
    ref = evaporation(
        series,
        temperature=env.temperature,
        pressure=env.pressure,
        specific_humidity=env.specific_humidity,
    )
    np.testing.assert_allclose(out.EVAPORATION_RATE, ref.EVAPORATION_RATE, rtol=1e-6)
    assert out.EVAPORATION_RATE.dims == ("time", "height")
    assert out.EVAPORATION_RATE.attrs["units"] == "kg kg-1 s-1"
    # environment already on the QVP heights: used as is
    same = evaporation(series, env)
    np.testing.assert_allclose(same.EVAPORATION_RATE, ref.EVAPORATION_RATE)
    # relative humidity in percent
    rh = xr.Dataset(
        {
            "temperature": env.temperature,
            "pressure": env.pressure,
            "relative_humidity": (env.temperature * 0 + 60.0).assign_attrs(units="%"),
        }
    )
    pct = evaporation(series, rh)
    frac = evaporation(series, env[["temperature", "pressure"]], relative_humidity=0.6)
    np.testing.assert_allclose(pct.EVAPORATION_RATE, frac.EVAPORATION_RATE)


def _sweep(seed=0, with_dsd=True):
    rng = np.random.default_rng(seed)
    az, rg = np.arange(0, 360, 10.0), np.arange(1000.0, 30000.0, 1000.0)
    shape = (az.size, rg.size)
    z = xr.DataArray(np.broadcast_to(rg * 0.05 + 100, shape), dims=("azimuth", "range"))
    data = {}
    if with_dsd:
        data = {
            "N0": (("azimuth", "range"), 10 ** rng.uniform(3, 4, shape)),
            "MU": (("azimuth", "range"), rng.uniform(0, 4, shape)),
            "LAMBDA": (("azimuth", "range"), rng.uniform(2, 6, shape)),
        }
    else:
        data = {"DBZH": (("azimuth", "range"), np.zeros(shape))}
    return xr.Dataset(data, coords={"azimuth": az, "range": rg, "z": z})


def test_datatree_volume_and_accessor():
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs={"site": "x"}),
            "sweep_0": _sweep(0),
            "sweep_1": _sweep(1),
            "sweep_2": _sweep(2, with_dsd=False),
        }
    )
    out = tree.radarx.evaporation(_profile())
    assert set(out.children) == {"sweep_0", "sweep_1"}
    one = _sweep(1).radarx.evaporation(_profile())
    xr.testing.assert_allclose(out["sweep_1"].to_dataset(), one)
    assert one.EVAPORATION_RATE.dims == ("azimuth", "range")
    assert "z" in one.coords
    # an environment per sweep: sweeps without one are skipped
    env = xr.DataTree.from_dict({"sweep_1": _profile()})
    part = evaporation(tree, env)
    assert set(part.children) == {"sweep_1"}
    with pytest.raises(KeyError, match="no node"):
        evaporation(tree, xr.DataTree.from_dict({"sweep_9": _profile()}))
    with pytest.raises(KeyError, match="N0, MU and LAMBDA"):
        evaporation(
            xr.DataTree.from_dict({"sweep_0": _sweep(0, with_dsd=False)}), _profile()
        )
    # integration accessor
    series = _series()
    a = series.radarx.integrate_evaporation(_profile())
    b = integrate_evaporation(series, _profile())
    xr.testing.assert_allclose(a, b)


def test_errors():
    ds = _mp(5.0)
    with pytest.raises(ValueError, match="engine"):
        evaporation(
            ds, temperature=290, pressure=9e4, relative_humidity=0.5, engine="x"
        )
    with pytest.raises(ValueError, match="fall_speed"):
        evaporation(
            ds, temperature=290, pressure=9e4, relative_humidity=0.5, fall_speed="x"
        )
    with pytest.raises(ValueError, match="fall_speed"):
        evaporation(
            ds,
            temperature=290,
            pressure=9e4,
            relative_humidity=0.5,
            fall_speed=(-1, 1, 0),
        )
    with pytest.raises(KeyError, match="pressure"):
        evaporation(ds, temperature=290, relative_humidity=0.5)
    with pytest.raises(KeyError, match="humidity"):
        evaporation(ds, temperature=290, pressure=9e4)
    with pytest.raises(KeyError, match="LAMBDA"):
        evaporation(
            ds.drop_vars("LAMBDA"), temperature=290, pressure=9e4, relative_humidity=0.5
        )
    with pytest.raises(TypeError):
        evaporation(ds.N0, temperature=290, pressure=9e4, relative_humidity=0.5)
    with pytest.raises(TypeError, match="environment"):
        evaporation(ds, {"temperature": 290})
    with pytest.raises(ValueError, match="coordinate"):
        evaporation(ds, _profile())
    if not evap.HAS_COMPILED_KERNEL:  # pragma: no cover - depends on the build
        with pytest.raises(ImportError):
            evaporation(
                ds,
                temperature=290,
                pressure=9e4,
                relative_humidity=0.5,
                engine="compiled",
            )
    # custom fall speed is used
    slow = evaporation(
        ds,
        temperature=290,
        pressure=9e4,
        relative_humidity=0.5,
        fall_speed=(1.0, 1.0, 0.2),
    )
    fast = evaporation(ds, temperature=290, pressure=9e4, relative_humidity=0.5)
    assert slow.EVAPORATION_RATE < fast.EVAPORATION_RATE
    assert "1 D^1" in slow.attrs["fall_speed"]


def test_drop_rates():
    q = float(evap._from_rh(293.15, 9.0e4, 0.7))
    out = drop_evaporation_rate([0.5, 1.0, 2.0, 4.0], 293.15, 9.0e4, q)
    assert (out.mass_rate < 0).all()
    # small drops shrink faster in diameter, large drops lose more mass
    assert (np.diff(out.diameter_rate) > 0).all()
    assert (np.diff(out.mass_rate) < 0).all()


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------


def test_real_nexrad_volume():
    xd = pytest.importorskip("xradar")
    from open_radar_data import DATASETS

    file = DATASETS.fetch("KLBB20160601_150025_V06")
    dtree = xd.io.open_nexradlevel2_datatree(file, sweep=[0, 1])
    dtree = dtree.xradar.georeference()
    for name in ("sweep_0", "sweep_1"):
        ds = dtree[name].to_dataset(inherit=False)
        for var, lim in (("DBZH", -32.0), ("ZDR", -12.9), ("RHOHV", 0.21)):
            if var in ds:
                ds[var] = ds[var].where(ds[var] > lim)
        dtree[name] = ds
    params = dtree.radarx.dsd()
    # a warm, dry boundary layer (like a West Texas summer afternoon)
    out = params.radarx.evaporation(_profile(0.4))
    res = out["sweep_0"].to_dataset()
    ds = dtree["sweep_0"].to_dataset()
    rain = ((ds.RHOHV > 0.97) & (ds.DBZH > 30) & (ds.DBZH < 50) & (ds.z < 3000)).values
    assert rain.sum() > 200
    cool = res.COOLING_RATE_HOURLY.values[rain]
    assert np.isfinite(cool).mean() > 0.9
    # a few K/h at 30-50 dBZ in air of 40 % RH
    assert 0.5 < np.nanmedian(cool) < 50.0
    assert (res.DBZ_TENDENCY.values[rain][np.isfinite(cool)] < 0).all()
    if evap.HAS_COMPILED_KERNEL:
        ref = evaporation(params, _profile(0.4), engine="numpy")
        np.testing.assert_allclose(
            res.EVAPORATION_RATE, ref["sweep_0"].EVAPORATION_RATE, rtol=1e-10
        )
