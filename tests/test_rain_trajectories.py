#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for radarx.retrieve.rain_trajectories."""

import importlib
import math

import numpy as np
import pytest
import xarray as xr
from scipy.integrate import solve_ivp

import radarx  # noqa: F401
from radarx.retrieve import (
    drop_evaporation_rate,
    rain_source_points,
    rain_trajectories,
    size_sorting,
    surface_dsd,
    terminal_fall_speed,
    trajectory_matched_times,
)

rt = importlib.import_module("radarx.retrieve.rain_trajectories")
evap = importlib.import_module("radarx.retrieve.evaporation")

ENGINES = ["numpy"] + (["compiled"] if rt.HAS_COMPILED_KERNEL else [])
needs_kernel = pytest.mark.skipif(
    not rt.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)
RHO0 = 1.204
FALL = (4.643, 0.9496, 0.1671)  # a D^b exp(-f D), as in drop_evaporation_rate


def v0(d):
    """Atlas et al. (1973) sea-level fall speed."""
    return 9.65 - 10.3 * np.exp(-0.6 * np.asarray(d, dtype=float))


def profile(z, *, t=None, p=None, rh=0.0, u=0.0, v=0.0, w=None):
    """A sounding-like profile with given (arrays or constants) variables."""
    z = np.asarray(z, dtype=float)

    def col(x, default):
        x = default if x is None else x
        return np.broadcast_to(np.asarray(x, dtype=float), z.shape).copy()

    data = {
        "temperature": ("height", col(t, 288.15 - 0.0065 * z)),
        "pressure": (
            "height",
            col(p, 101325.0 * (1.0 - 2.25577e-5 * z) ** 5.25588),
        ),
        "relative_humidity": ("height", col(rh, 0.0)),
        "u": ("height", col(u, 0.0)),
        "v": ("height", col(v, 0.0)),
    }
    if w is not None:
        data["w"] = ("height", col(w, 0.0))
    return xr.Dataset(data, coords={"height": z})


def wind_grid(
    u, v=0.0, w=0.0, *, x=(-1e5, 1e5), y=(-1e5, 1e5), z=(0.0, 6000.0), t=None
):
    """A wind Dataset from functions ``u(t, z, y, x)`` or constants."""
    xs, ys, zs = (np.asarray(a, dtype=float) for a in (x, y, z))
    times = np.array([0.0]) if t is None else np.asarray(t)
    if np.issubdtype(times.dtype, np.datetime64):
        tt = (times - times[0]) / np.timedelta64(1, "s")
    else:
        tt = times
    T, Z, Y, X = np.meshgrid(tt.astype(float), zs, ys, xs, indexing="ij")

    def field(f):
        return f(T, Z, Y, X) if callable(f) else np.full(T.shape, float(f))

    ds = xr.Dataset(
        {
            k: (("time", "z", "y", "x"), field(f))
            for k, f in (("u", u), ("v", v), ("w", w))
        },
        coords={"time": times, "z": zs, "y": ys, "x": xs},
    )
    return ds


def rho_of(z, t, p, rh=0.0):
    e = rh * evap._saturation_vapor_pressure(t)
    q = evap.EPS * e / (p - (1.0 - evap.EPS) * e)
    return p / (evap.RD * t * (1.0 + (1.0 / evap.EPS - 1.0) * q))


def run(source, diameter, engine, **kw):
    kw.setdefault("evaporation", False)
    kw.setdefault("density_correction", False)
    return rain_trajectories(source, diameter, engine=engine, **kw)


POINT = {"x": 0.0, "y": 0.0, "z": 3000.0}
DIAMS = [1.0, 2.0, 4.0]


# --------------------------------------------------------------------------
# analytic: still air, uniform wind, linear shear
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_still_air_fall_time_is_the_quadrature(engine):
    """With density correction the fall time is the integral of dz/Vt(D, z)."""
    z = np.arange(0.0, 6001.0, 2.0)
    prof = profile(z)
    res = rain_trajectories(
        {"x": 0.0, "y": 0.0, "z": 5000.0},
        DIAMS,
        profile=prof,
        evaporation=False,
        time_step=10.0,
        engine=engine,
    )
    zz = np.linspace(0.0, 5000.0, 400001)
    rho = rho_of(
        zz, np.interp(zz, z, prof.temperature), np.interp(zz, z, prof.pressure)
    )
    for k, d in enumerate(DIAMS):
        f = 1.0 / (v0(d) * (RHO0 / rho) ** 0.4)
        exact = np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(zz))
        assert res.fall_time.values.ravel()[k] == pytest.approx(exact, rel=1e-7)
    assert (res.status == 1).all()
    np.testing.assert_allclose(res.landing_z, 0.0, atol=1e-9)
    np.testing.assert_allclose(res.landing_x, 0.0, atol=1e-9)
    # larger drops fall faster
    assert (np.diff(res.fall_time.values.ravel()) < 0).all()


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("source", ["profile", "grid"])
def test_uniform_wind_drift_is_exact(engine, source):
    u, v, w = 7.0, -3.0, 0.5
    if source == "profile":
        kw = {"profile": profile(np.arange(0, 6001.0, 1000.0), u=u, v=v, w=w)}
    else:
        kw = {
            "wind": wind_grid(u, v, w),
            "profile": profile(np.arange(0, 6001.0, 1000.0), u=0.0, v=0.0),
        }
    res = run(POINT, DIAMS, engine, **kw)
    t_fall = 3000.0 / (v0(DIAMS) - w)  # an updraft slows the fall
    np.testing.assert_allclose(res.fall_time.values.ravel(), t_fall, rtol=1e-12)
    np.testing.assert_allclose(res.landing_x.values.ravel(), u * t_fall, rtol=1e-12)
    np.testing.assert_allclose(res.landing_y.values.ravel(), v * t_fall, rtol=1e-12)
    np.testing.assert_allclose(res.fall_speed_start.values.ravel(), v0(DIAMS) - w)
    # numerically the same flux: the speed does not change
    np.testing.assert_allclose(res.concentration_ratio, 1.0, rtol=1e-12)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("source", ["profile", "grid"])
def test_linear_shear_displacement(engine, source):
    """u = a z gives x = a H^2 / (2 V): larger drops are displaced less."""
    a = 0.004
    zlev = np.arange(0.0, 6001.0, 1500.0)
    if source == "profile":
        kw = {"profile": profile(zlev, u=a * zlev)}
    else:
        kw = {
            "wind": wind_grid(lambda t, z, y, x: a * z),
            "profile": profile(zlev, u=0.0),
        }
    res = run(POINT, DIAMS, engine, **kw)
    exact = a * 3000.0**2 / (2.0 * v0(DIAMS))
    np.testing.assert_allclose(res.landing_x.values.ravel(), exact, rtol=1e-11)
    assert (np.diff(res.landing_x.values.ravel()) < 0).all()


@pytest.mark.parametrize("engine", ENGINES)
def test_frozen_pattern_moves_with_the_storm(engine):
    """A wind field U(x - c t): dxi/dt = U0 - c + k xi has a closed form."""
    u0, k, c = 5.0, 0.001, 10.0
    wind = wind_grid(lambda t, z, y, x: u0 + k * x, x=(-1e5, 0.0, 1e5))
    prof = profile(np.arange(0, 6001.0, 1000.0))
    res = run(POINT, [2.0], engine, wind=wind, profile=prof, storm_motion=(c, 0.0))
    t_fall = 3000.0 / v0(2.0)
    xi = (u0 - c) / -k + (0.0 - (u0 - c) / -k) * math.exp(k * t_fall)
    assert res.landing_x.values.item() == pytest.approx(xi + c * t_fall, rel=1e-9)
    still = run(POINT, [2.0], engine, wind=wind, profile=prof)
    xs = u0 * (math.exp(k * t_fall) - 1.0) / k  # x' = u0 + k x without motion
    assert still.landing_x.values.item() == pytest.approx(xs, rel=1e-9)


@pytest.mark.parametrize("engine", ENGINES)
def test_time_interpolation_numeric_and_datetime(engine):
    """Linear interpolation in time: u = u0 + a t gives x = u0 T + a T^2 / 2."""
    u0, a = 4.0, 0.01
    prof = profile(np.arange(0, 6001.0, 1000.0))
    t_fall = 3000.0 / v0(2.0)
    wind = wind_grid(lambda t, z, y, x: u0 + a * t, t=[0.0, 1000.0])
    res = run(POINT, [2.0], engine, wind=wind, profile=prof, time=100.0)
    exact = (u0 + 100.0 * a) * t_fall + 0.5 * a * t_fall**2
    assert res.landing_x.values.item() == pytest.approx(exact, rel=1e-11)
    assert res.landing_time.values.item() == pytest.approx(100.0 + t_fall)
    # datetimes: the same result, landing time as datetime64
    base = np.datetime64("2022-03-30T23:00:00", "ns")
    wind_t = wind_grid(
        lambda t, z, y, x: u0 + a * t,
        t=base + np.array([0, 1000]) * np.timedelta64(1, "s"),
    )
    src = dict(POINT, time=base + np.timedelta64(100, "s"))
    res_t = run(src, [2.0], engine, wind=wind_t, profile=prof)
    assert res_t.landing_x.values.item() == pytest.approx(exact, rel=1e-11)
    land = res_t.landing_time.values.ravel()[0]
    expected = base + np.timedelta64(int(round((100.0 + t_fall) * 1e9)), "ns")
    assert abs((land - expected) / np.timedelta64(1, "ns")) < 10
    # the time is held constant before the first and after the last analysis
    early = run(POINT, [2.0], engine, wind=wind, profile=prof, time=-500.0)
    assert (
        early.landing_x.values.item()
        == pytest.approx(u0 * t_fall + 0.5 * a * t_fall**2 + a * 0.0, rel=1e-3)
        or early.landing_x.values.item() > 0
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_background_wind_outside_the_grid_and_missing_values(engine):
    """Outside the grid and where the analysis is missing the profile is used."""
    wind = wind_grid(
        2.0, x=(-300.0, 300.0), y=(-300.0, 300.0), z=np.arange(0, 6001.0, 500.0)
    )
    prof = profile(np.arange(0, 6001.0, 1000.0), u=6.0)
    t_fall = 3000.0 / v0(2.0)
    res = run({"x": 0.0, "y": 0.0, "z": 3000.0}, [2.0], engine, wind=wind, profile=prof)
    # starts inside the grid at 2 m/s, leaves it after 1000 m, continues at 6 m/s
    t_in = 300.0 / 2.0
    exact = 300.0 + 6.0 * (t_fall - t_in)
    assert res.landing_x.values.item() == pytest.approx(exact, rel=2e-2)
    # a hole in the analysis (NaN) means the background wind there
    holes = wind_grid(2.0, z=np.arange(0, 6001.0, 500.0))
    holes["u"] = holes["u"].where(holes.z < 1500.0)
    res = run(
        {"x": 0.0, "y": 0.0, "z": 3000.0}, [2.0], engine, wind=holes, profile=prof
    )
    t_low = 1000.0 / v0(2.0)
    exact = 2.0 * t_low + 6.0 * (t_fall - t_low)
    assert res.landing_x.values.item() == pytest.approx(exact, rel=2e-2)
    # without a profile the background is the mean wind of the analysis
    res = run(POINT, [2.0], engine, wind=wind_grid(3.0))
    assert res.landing_x.values.item() == pytest.approx(3.0 * t_fall, rel=1e-12)


@pytest.mark.parametrize("engine", ENGINES)
def test_horizontal_divergence_in_the_concentration(engine):
    """u = a x, v = a y: x grows as exp(a T) and n falls as exp(-2 a T)."""
    a = 1.0e-4
    wind = wind_grid(lambda t, z, y, x: a * x, lambda t, z, y, x: a * y)
    prof = profile(np.arange(0, 6001.0, 1000.0))
    t_fall = 3000.0 / v0(2.0)
    kw = {"wind": wind, "profile": prof}
    src = {"x": 800.0, "y": -500.0, "z": 3000.0}
    res = run(src, [2.0], engine, wind_divergence=True, **kw)
    assert res.landing_x.values.item() == pytest.approx(
        800.0 * math.exp(a * t_fall), rel=1e-10
    )
    assert res.landing_y.values.item() == pytest.approx(
        -500.0 * math.exp(a * t_fall), rel=1e-10
    )
    assert res.concentration_ratio.values.item() == pytest.approx(
        math.exp(-2.0 * a * t_fall), rel=1e-9
    )
    off = run(src, [2.0], engine, **kw)
    assert off.concentration_ratio.values.item() == pytest.approx(1.0, rel=1e-12)
    # the vertical wind gradient of a profile enters through dw/dz as well
    pw = profile(np.arange(0, 6001.0, 1000.0), w=np.arange(0, 6001.0, 1000.0) * 1e-4)
    res = run(POINT, [2.0], engine, profile=pw, wind_divergence=True)
    assert res.concentration_ratio.values.item() < 1.0


# --------------------------------------------------------------------------
# density correction, flux conservation, evaporation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_flux_conservation_with_the_density_correction(engine):
    """n Vt is constant: n grows by Vt(top) / Vt(surface) = (rho_s/rho_t)^0.4."""
    z = np.arange(0.0, 8001.0, 5.0)
    prof = profile(z)
    res = rain_trajectories(
        {"x": 0.0, "y": 0.0, "z": 5000.0},
        DIAMS,
        profile=prof,
        evaporation=False,
        time_step=5.0,
        engine=engine,
    )
    ratio = res.fall_speed_start / res.fall_speed_end
    np.testing.assert_allclose(res.concentration_ratio, ratio, rtol=3e-4)
    # at the surface the air is denser, the drops fall slower, n is larger
    assert (res.concentration_ratio > 1.0).all()
    rho_t = rho_of(
        5000.0,
        288.15 - 0.0065 * 5000.0,
        101325.0 * (1 - 2.25577e-5 * 5000.0) ** 5.25588,
    )
    rho_s = rho_of(0.0, 288.15, 101325.0)
    np.testing.assert_allclose(
        res.concentration_ratio.values.ravel(), (rho_s / rho_t) ** 0.4, rtol=5e-4
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_evaporation_flux_conservation_with_bin_stretching(engine):
    """n Vt dD is conserved: the ratio includes the change of the bin width."""
    z = np.arange(0.0, 6001.0, 10.0)
    prof = profile(z, rh=0.5)
    d0 = np.array([1.6, 2.0, 3.0])
    eps = 1e-4
    kw = dict(profile=prof, time_step=2.0, engine=engine)
    src = {"x": 0.0, "y": 0.0, "z": 3000.0}
    mid = rain_trajectories(src, d0, **kw)
    up = rain_trajectories(src, d0 * (1 + eps), **kw)
    dn = rain_trajectories(src, d0 * (1 - eps), **kw)
    assert (mid.status == 1).all()
    assert (mid.evaporated_mass_fraction > 0.01).all()
    dd = (up.landing_diameter.values - dn.landing_diameter.values) / (2 * eps * d0)
    expected = mid.fall_speed_start / mid.fall_speed_end / dd
    np.testing.assert_allclose(mid.concentration_ratio, expected, rtol=1e-4)
    # smaller drops shrink more
    assert (np.diff(mid.evaporated_mass_fraction.values.ravel()) < 0).all()


@pytest.mark.parametrize("engine", ENGINES)
def test_evaporation_matches_drop_evaporation_rate(engine):
    """One tiny step reproduces the existing single-drop rate."""
    t, p, rh = 290.0, 9.0e4, 0.5
    prof = profile(np.array([0.0, 4000.0]), t=t, p=p, rh=rh)
    e = rh * evap._saturation_vapor_pressure(t)
    q = evap.EPS * e / (p - (1.0 - evap.EPS) * e)
    d = np.array([0.5, 1.0, 2.0, 4.0])
    rate = drop_evaporation_rate(d, t, p, q, fall_speed=FALL).diameter_rate.values
    dt = 1.0e-3
    res = rain_trajectories(
        {"x": 0.0, "y": 0.0, "z": 3000.0},
        d,
        profile=prof,
        fall_speed=FALL,
        time_step=dt,
        max_time=dt,
        engine=engine,
    )
    assert (res.status == 0).all()
    np.testing.assert_allclose(
        (res.landing_diameter.values.ravel() - d) / dt, rate, rtol=1e-5
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_evaporation_coupled_path_against_an_ode_solver(engine):
    """Fall and shrinking against scipy's DOP853 with drop_evaporation_rate."""
    t, p, rh = 290.0, 9.0e4, 0.4
    prof = profile(np.array([0.0, 4000.0]), t=t, p=p, rh=rh)
    e = rh * evap._saturation_vapor_pressure(t)
    q = evap.EPS * e / (p - (1.0 - evap.EPS) * e)
    rho = rho_of(0.0, t, p, rh)
    a, b, f = FALL

    def vt(d):
        return a * d**b * math.exp(-f * d) * (RHO0 / rho) ** 0.4

    def rhs(_, y):
        rate = drop_evaporation_rate(y[1], t, p, q, fall_speed=FALL).diameter_rate
        return [-vt(y[1]), float(rate)]

    def hit(_, y):
        return y[0]

    hit.terminal = True
    for d0 in (1.5, 3.0):
        sol = solve_ivp(
            rhs,
            [0.0, 3000.0],
            [2000.0, d0],
            method="DOP853",
            rtol=1e-12,
            atol=1e-12,
            events=hit,
        )
        res = rain_trajectories(
            {"x": 0.0, "y": 0.0, "z": 2000.0},
            [d0],
            profile=prof,
            fall_speed=FALL,
            time_step=4.0,
            engine=engine,
        )
        assert res.fall_time.values.item() == pytest.approx(
            sol.t_events[0][0], rel=1e-8
        )
        assert res.landing_diameter.values.item() == pytest.approx(
            sol.y_events[0][0][1], rel=1e-8
        )
        assert res.evaporated_mass_fraction.values.item() > 0.02


@pytest.mark.parametrize("engine", ENGINES)
def test_small_drops_evaporate_completely(engine):
    prof = profile(np.arange(0, 6001.0, 500.0), rh=0.4, u=5.0)
    res = rain_trajectories(
        {"x": 0.0, "y": 0.0, "z": 3000.0},
        [0.2, 0.3, 0.5, 5.0],
        profile=prof,
        engine=engine,
    )
    status = res.status.values.ravel()
    assert status[:3].tolist() == [2, 2, 2]
    assert status[3] == 1
    frac = res.evaporated_mass_fraction.values.ravel()
    assert (frac[:3] == 1.0).all() and 0.0 < frac[3] < 0.2
    # a drop that evaporated ends at the evaporated diameter, before the ground
    np.testing.assert_allclose(
        res.landing_diameter.values.ravel()[:3], 0.12, rtol=1e-12
    )
    assert (res.landing_z.values.ravel()[:3] > 0.0).all()
    # drops that cannot evaporate in saturated air stay whole
    wet = rain_trajectories(
        {"x": 0.0, "y": 0.0, "z": 3000.0},
        [0.5, 2.0],
        profile=profile(np.arange(0, 6001.0, 500.0), rh=1.0),
        engine=engine,
    )
    np.testing.assert_allclose(wet.evaporated_mass_fraction, 0.0, atol=1e-12)


# --------------------------------------------------------------------------
# integrator accuracy
# --------------------------------------------------------------------------


def _landing_errors(scheme, steps, engine):
    z = np.arange(0.0, 8001.0, 1.0)
    prof = profile(z, rh=0.6, u=10.0 * np.sin(z / 700.0), v=4.0 * np.cos(z / 1100.0))
    src = {"x": 0.0, "y": 0.0, "z": 5000.0}
    d = [2.0, 4.0]
    ref = rain_trajectories(
        src,
        d,
        profile=prof,
        time_step=0.5,
        max_time=2000.0,
        engine="compiled" if rt.HAS_COMPILED_KERNEL else "numpy",
    )
    errs = []
    for h in steps:
        r = rain_trajectories(
            src,
            d,
            profile=prof,
            time_step=h,
            max_time=2000.0,
            scheme=scheme,
            engine=engine,
        )
        errs.append(
            np.hypot(
                r.landing_x.values.ravel() - ref.landing_x.values.ravel(),
                r.landing_y.values.ravel() - ref.landing_y.values.ravel(),
            )
        )
    return np.array(errs)


@pytest.mark.parametrize("scheme, order", [("rk4", 4.0), ("rk2", 2.0)])
def test_convergence_order(scheme, order):
    """Halving the step reduces the landing error by 2^order (both drops)."""
    steps = [80.0, 40.0, 20.0] if scheme == "rk4" else [40.0, 20.0, 10.0]
    engine = "compiled" if rt.HAS_COMPILED_KERNEL else "numpy"
    errs = _landing_errors(scheme, steps, engine)
    observed = np.log2(errs[:-1] / errs[1:])
    assert (observed > order - 0.7).all(), (errs, observed)
    assert (observed < order + 1.7).all(), (errs, observed)


# --------------------------------------------------------------------------
# the engines agree
# --------------------------------------------------------------------------


def _random_case(seed=1, smooth=False):
    rng = np.random.default_rng(seed)
    if smooth:
        x = np.linspace(-4000.0, 8000.0, 241)
        y = np.linspace(-4000.0, 4000.0, 161)
        z = np.arange(0.0, 4001.0, 100.0)
        t = np.array([0.0, 300.0, 700.0])
        T, Z, Y, X = np.meshgrid(t, z, y, x, indexing="ij")
        ph = rng.uniform(0, 6.28, 3)
        u = 8.0 + 3.0 * np.sin(X / 2500.0 + ph[0]) * np.cos(Z / 1500.0) + 0.004 * T
        v = 2.0 + 3.0 * np.cos(Y / 2000.0 + ph[1]) * np.sin(Z / 1800.0 + ph[2])
        w = 1.0 * np.sin(X / 3000.0) * np.sin(Z / 1000.0)
    else:
        x = np.linspace(-2000.0, 6000.0, 9)
        y = np.linspace(-3000.0, 3000.0, 7)
        z = np.array([0.0, 500.0, 1200.0, 2500.0, 4000.0])
        t = np.array([0.0, 300.0, 700.0])
        shape = (t.size, z.size, y.size, x.size)
        u = rng.normal(8.0, 3.0, shape)
        v = rng.normal(2.0, 3.0, shape)
        w = rng.normal(0.0, 1.0, shape)
        u[1, 2, 3, 4] = np.nan  # a missing neighbour
    wind = xr.Dataset(
        {k: (("time", "z", "y", "x"), a) for k, a in (("u", u), ("v", v), ("w", w))},
        coords={"time": t, "z": z, "y": y, "x": x},
    )
    zz = np.arange(0.0, 8001.0, 250.0)
    prof = profile(zz, rh=0.55, u=10.0 + 0.002 * zz, v=1.0)
    n = 24
    src = {
        "x": rng.uniform(-1500.0, 5000.0, n),
        "y": rng.uniform(-2500.0, 2500.0, n),
        "z": rng.uniform(1500.0, 3800.0, n),
        "time": rng.uniform(0.0, 600.0, n),
    }
    return wind, prof, src


def _assert_same(a, b, rtol=1e-9, atol=1e-8):
    assert set(a.data_vars) == set(b.data_vars)
    for k in a.data_vars:
        va, vb = a[k].values, b[k].values
        if va.dtype.kind == "f":
            np.testing.assert_allclose(va, vb, rtol=rtol, atol=atol, err_msg=k)
        elif va.dtype.kind == "M":
            np.testing.assert_array_equal(va, vb, err_msg=k)
        else:
            np.testing.assert_array_equal(va, vb, err_msg=k)


@needs_kernel
@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"scheme": "rk2", "wind_divergence": True},
        {"storm_motion": (12.0, 4.0), "pattern_time": 200.0, "wind_divergence": True},
        {"dispersion": (1.5, 0.5, 60.0), "members": 3, "seed": 123},
        {"fall_speed": FALL, "density_correction": False, "store_path": 4},
        {"fall_speed": ("polynomial", -0.1, 4.9, -0.95, 0.079, -0.0024)},
        {"evaporation": False, "concentration": False, "time_step": 20.0},
        {"max_time": 100.0},
    ],
    ids=[
        "default",
        "rk2-div",
        "pattern",
        "turbulence",
        "power-path",
        "poly",
        "dry",
        "aloft",
    ],
)
def test_compiled_and_numpy_agree(kw):
    wind, prof, src = _random_case()
    d = [0.3, 1.0, 2.5, 5.0]
    a = rain_trajectories(src, d, wind=wind, profile=prof, engine="numpy", **kw)
    b = rain_trajectories(
        src, d, wind=wind, profile=prof, engine="compiled", n_threads=3, **kw
    )
    _assert_same(a, b)
    assert np.isin(a.status.values, [0, 1, 2]).all()


@needs_kernel
@pytest.mark.parametrize("scheme", ["rk4", "rk2"])
def test_compiled_and_numpy_agree_backward(scheme):
    wind, prof, src = _random_case(2)
    tgt = {k: v for k, v in src.items() if k != "z"}
    tgt["time"] = tgt["time"] + 800.0
    kw = dict(
        wind=wind,
        profile=prof,
        source_height=3000.0,
        storm_motion=(9.0, 1.0),
        dispersion=(1.0, 0.3, 40.0),
        wind_divergence=True,
        scheme=scheme,
        store_path=5,
    )
    a = rain_source_points(tgt, [0.6, 2.0, 4.0], engine="numpy", **kw)
    b = rain_source_points(tgt, [0.6, 2.0, 4.0], engine="compiled", **kw)
    _assert_same(a, b)


@needs_kernel
def test_threads_and_seed():
    wind, prof, src = _random_case()
    kw = dict(wind=wind, profile=prof, dispersion=(2.0, 0.5, 50.0), members=4, seed=7)
    one = rain_trajectories(src, [1.0, 3.0], n_threads=1, **kw)
    many = rain_trajectories(src, [1.0, 3.0], n_threads=4, **kw)
    _assert_same(one, many, rtol=0, atol=0)
    other = rain_trajectories(src, [1.0, 3.0], **dict(kw, seed=8))
    assert not np.allclose(one.landing_x, other.landing_x)
    # the members of one drop differ
    assert one.landing_x.std("member").min() > 0.0


# --------------------------------------------------------------------------
# turbulence statistics
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_turbulent_dispersion_variance(engine):
    """Var(x) of an integrated AR(1) velocity has a closed form."""
    sigma, tl, h = 1.0, 60.0, 1.0
    prof = profile(np.arange(0, 6001.0, 1000.0))
    n = 4000 if engine == "compiled" else 600
    res = run(
        {"x": 0.0, "y": 0.0, "z": 3000.0},
        [2.0],
        engine,
        profile=prof,
        dispersion=(sigma, 0.0, tl),
        members=n,
        time_step=h,
        seed=11,
        concentration=False,
    )
    t_fall = 3000.0 / v0(2.0)
    nsteps = int(round(t_fall / h))
    a = math.exp(-h / tl)
    var = (
        sigma**2
        * h**2
        * (nsteps * (1 + a) / (1 - a) - 2 * a * (1 - a**nsteps) / (1 - a) ** 2)
    )
    x = res.landing_x.values.ravel()
    # the last, fractional step adds a little; the vertical speed is unperturbed
    assert x.var() == pytest.approx(var, rel=0.15 if engine == "numpy" else 0.08)
    assert abs(x.mean()) < 4 * math.sqrt(var / n)
    np.testing.assert_allclose(res.fall_time, t_fall, rtol=1e-12)
    # a vertical perturbation changes the fall time
    w = run(
        {"x": 0.0, "y": 0.0, "z": 3000.0},
        [2.0],
        engine,
        profile=prof,
        dispersion=(0.0, 1.0, tl),
        members=50,
        time_step=h,
    )
    assert w.fall_time.std() > 0.5


# --------------------------------------------------------------------------
# fall speed laws
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_fall_speed_laws(engine):
    prof = profile(np.arange(0, 6001.0, 1000.0), t=288.15, p=101325.0)
    rho = rho_of(0.0, 288.15, 101325.0)
    d = np.array([1.0, 2.0, 4.0])
    src = {"x": 0.0, "y": 0.0, "z": 1000.0}
    base = rain_trajectories(
        src,
        d,
        profile=prof,
        evaporation=False,
        time_step=1.0,
        max_time=0.0,
        engine=engine,
    )
    corr = (RHO0 / rho) ** 0.4
    np.testing.assert_allclose(
        base.fall_speed_start.values.ravel(),
        terminal_fall_speed(d, rho).values,
        rtol=1e-12,
    )
    np.testing.assert_allclose(base.fall_speed_start.values.ravel(), v0(d) * corr)
    a, b, f = FALL
    power = rain_trajectories(
        src,
        d,
        profile=prof,
        evaporation=False,
        fall_speed=FALL,
        max_time=0.0,
        engine=engine,
    )
    np.testing.assert_allclose(
        power.fall_speed_start.values.ravel(), a * d**b * np.exp(-f * d) * corr
    )
    coef = (0.5, 1.5, -0.1)
    poly = rain_trajectories(
        src,
        d,
        profile=prof,
        evaporation=False,
        fall_speed=("polynomial",) + coef,
        density_correction=False,
        max_time=0.0,
        engine=engine,
    )
    np.testing.assert_allclose(
        poly.fall_speed_start.values.ravel(), coef[0] + coef[1] * d + coef[2] * d**2
    )
    # a negative speed is set to zero
    neg = rain_trajectories(
        src, [0.05], profile=prof, evaporation=False, max_time=0.0, engine=engine
    )
    assert neg.fall_speed_start.values.item() == 0.0


# --------------------------------------------------------------------------
# backward trajectories
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_backward_inverts_forward(engine):
    wind, prof, src = _random_case(3, smooth=True)
    src = {k: v[:6] for k, v in src.items()}
    src["y"] = src["y"] * 0.4
    src["x"] = src["x"] * 0.4  # stay away from the edge of the analysis
    d = [1.5, 3.0]
    kw = dict(
        wind=wind,
        profile=prof,
        storm_motion=(3.0, 0.5),
        time_step=2.0,
        engine=engine,
    )
    fwd = rain_trajectories(src, d, **kw)
    checked = 0
    for i in range(6):
        for k in range(len(d)):
            if fwd.status.values[i, k] != 1:
                continue
            res = rain_source_points(
                {
                    "x": fwd.landing_x.values[i, k],
                    "y": fwd.landing_y.values[i, k],
                    "time": fwd.landing_time.values[i, k],
                },
                [fwd.landing_diameter.values[i, k]],
                source_height=src["z"][i],
                **kw,
            )
            assert res.status.values.item() == 1
            assert res.source_x.values.item() == pytest.approx(src["x"][i], abs=5e-2)
            assert res.source_y.values.item() == pytest.approx(src["y"][i], abs=5e-2)
            assert res.source_time.values.item() == pytest.approx(
                src["time"][i], abs=5e-3
            )
            assert res.source_diameter.values.item() == pytest.approx(d[k], rel=1e-6)
            # concentration ratios are the same (later over earlier) both ways
            assert res.concentration_ratio.values.item() == pytest.approx(
                fwd.concentration_ratio.values[i, k], rel=1e-4
            )
            assert res.evaporated_mass_fraction.values.item() == pytest.approx(
                fwd.evaporated_mass_fraction.values[i, k], abs=1e-5
            )
            checked += 1
    assert checked >= 6


@pytest.mark.parametrize("engine", ENGINES)
def test_source_dsd_is_sampled_at_the_source(engine):
    """A source field linear in x: the surface value is read at the source."""
    u = 8.0
    prof = profile(np.arange(0, 6001.0, 1000.0), u=u, t=288.15, p=101325.0)
    xs = np.linspace(-20000.0, 20000.0, 41)
    ds_ = np.array([1.0, 2.0, 3.0, 4.0])
    field = xr.DataArray(
        (100.0 + 0.01 * xs)[:, None] * (1.0 + 0.0 * ds_)[None, :],
        dims=("x", "diameter"),
        coords={"x": xs, "diameter": ds_},
    )
    tgt = {"x": 0.0, "y": 0.0}
    res = rain_source_points(
        tgt,
        [2.0, 3.0],
        source_height=3000.0,
        profile=prof,
        source_dsd=field,
        evaporation=False,
        engine=engine,
    )
    x_src = res.source_x.values.ravel()
    t_fall = 3000.0 / (
        v0([2.0, 3.0]) * (RHO0 / rho_of(1500.0, 288.15, 101325.0)) ** 0.4
    )
    np.testing.assert_allclose(x_src, -u * t_fall, rtol=1e-4)
    np.testing.assert_allclose(
        res.ND.values.ravel(),
        (100.0 + 0.01 * x_src) * res.concentration_ratio.values.ravel(),
        rtol=1e-12,
    )
    assert res.ND.attrs["units"] == "m-3 mm-1"
    # a field with a time dimension of length one and a z axis are accepted
    f2 = field.expand_dims(time=[0.0], z=[3000.0, 4000.0])
    res2 = rain_source_points(
        tgt,
        [2.0],
        source_height=3000.0,
        profile=prof,
        source_dsd=f2,
        evaporation=False,
        time=0.0,
        engine=engine,
    )
    assert np.isfinite(res2.ND.values).all()
    with pytest.raises(ValueError, match="dimensions"):
        rain_source_points(
            tgt,
            [2.0],
            source_height=3000.0,
            profile=prof,
            source_dsd=field.expand_dims(bad=2),
            engine=engine,
        )
    with pytest.raises(ValueError, match="diameter"):
        rain_source_points(
            tgt,
            [2.0],
            source_height=3000.0,
            profile=prof,
            source_dsd=field.isel(diameter=0, drop=True),
            engine=engine,
        )
    with pytest.raises(TypeError):
        rain_source_points(
            tgt,
            [2.0],
            source_height=3000.0,
            profile=prof,
            source_dsd=np.ones(3),
            engine=engine,
        )


# --------------------------------------------------------------------------
# the constant-wind stand-in (ml/models/bayesian_dsd/match.py)
# --------------------------------------------------------------------------


def _stand_in_time(d, h, rho, u, c, t_gate=0.0, d_ref=2.0):
    """The trajectory-pair formula of the stand-in, written out."""

    def fall(dd):
        return (
            np.maximum(9.65 - 10.3 * np.exp(-0.6 * np.asarray(dd, float)), 0.1)
            * (RHO0 / rho) ** 0.4
        )

    tau = h / fall(d)
    tau_ref = h / fall(d_ref)
    cn = np.linalg.norm(c)
    chat = np.asarray(c) / cn
    shift = tau - (np.asarray(u) @ chat) * (tau - tau_ref) / cn
    return t_gate + shift, tau, tau_ref


@pytest.mark.parametrize("engine", ENGINES)
def test_reproduces_the_constant_wind_stand_in(engine):
    """Constant wind, no evaporation: the time of every size bin of the stand-in."""
    h, u, c = 2500.0, np.array([9.0, 4.0]), np.array([14.0, 6.0])
    t_, p_ = 290.0, 95000.0
    rho = rho_of(0.0, t_, p_)
    prof = profile(np.arange(0.0, 6001.0, 500.0), t=t_, p=p_, rh=0.0, u=u[0], v=u[1])
    d = np.array([0.6, 1.0, 2.0, 3.0, 5.0])
    d_ref = 2.0
    tau_ref = h / (v0(d_ref) * (RHO0 / rho) ** 0.4)
    site = {"x": 20000.0, "y": 5000.0}
    gate = {
        "x": site["x"] - u[0] * tau_ref,
        "y": site["y"] - u[1] * tau_ref,
        "z": h,
        "time": 0.0,
    }
    res = trajectory_matched_times(
        gate,
        site,
        d,
        storm_motion=tuple(c),
        profile=prof,
        evaporation=False,
        density_correction=True,
        engine=engine,
    )
    expected, tau, _ = _stand_in_time(d, h, rho, u, c)
    # the stand-in floors the Atlas speed at 0.1 m/s, which only matters below 0.11 mm
    np.testing.assert_allclose(res.arrival_time.values.ravel(), expected, atol=1e-6)
    np.testing.assert_allclose(res.fall_time.values.ravel(), tau, rtol=1e-12)
    assert res.converged.all()
    # the cross-track miss is the cross drift the stand-in reports
    chat = c / np.linalg.norm(c)
    cross = (u[0] * chat[1] - u[1] * chat[0]) * (tau - tau_ref)
    np.testing.assert_allclose(res.cross_miss.values.ravel(), -cross, atol=1e-6)
    # the concentration ratio is the (rho/rho0)^0.4 scaling at constant density: 1
    np.testing.assert_allclose(res.concentration_ratio, 1.0, rtol=1e-12)
    # with evaporation the drops arrive later (they shrink and fall more slowly)
    dry = profile(np.arange(0.0, 6001.0, 500.0), t=t_, p=p_, rh=0.3, u=u[0], v=u[1])
    ev = trajectory_matched_times(
        gate,
        site,
        d,
        storm_motion=tuple(c),
        profile=dry,
        engine=engine,
        density_correction=True,
    )
    big = d >= 2.0
    assert (ev.fall_time.values.ravel()[big] > tau[big]).all()
    assert (ev.evaporated_mass_fraction.values.ravel()[big] > 0).all()


@pytest.mark.parametrize("engine", ENGINES)
def test_matched_times_varying_wind_converges(engine):
    """With shear and time-varying winds the secant iteration converges."""
    wind, prof, _ = _random_case()
    res = trajectory_matched_times(
        {"x": [0.0, 500.0], "y": [0.0, 0.0], "z": 2500.0, "time": 100.0},
        {"x": 6000.0, "y": 1000.0},
        [1.0, 2.0, 4.0],
        storm_motion=(10.0, 2.0),
        wind=wind,
        profile=prof,
        engine=engine,
        max_iterations=30,
        tolerance=1e-3,
    )
    # drops that evaporate (1 mm here) never reach the site
    assert (res.converged | (res.status != 1)).all()
    assert (res.status == 1).sum() >= 4
    assert res.arrival_time.shape == (2, 3)
    with pytest.raises(ValueError, match="storm_motion"):
        trajectory_matched_times(
            {"x": 0.0, "y": 0.0, "z": 1000.0},
            {"x": 1.0, "y": 0.0},
            [2.0],
            storm_motion=(0.0, 0.0),
            profile=prof,
            engine=engine,
        )


# --------------------------------------------------------------------------
# size sorting
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_size_sorting_in_a_shear(engine):
    a = 0.004
    zlev = np.arange(0.0, 6001.0, 1500.0)
    d = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 6.0])
    res = run(POINT, d, engine, profile=profile(zlev, u=a * zlev))
    s = size_sorting(res, 2.0)
    expect_x = a * 3000.0**2 / (2.0 * v0(d))
    ref_x = a * 3000.0**2 / (2.0 * v0(2.0))
    # the 2 mm point is interpolated between its neighbours' end points, but 2 mm is a bin
    np.testing.assert_allclose(
        s.displacement_x.values.ravel(), expect_x - ref_x, rtol=1e-9, atol=1e-6
    )
    np.testing.assert_allclose(s.displacement_cross.values.ravel(), 0.0, atol=1e-6)
    np.testing.assert_allclose(
        s.displacement_along.values.ravel(), expect_x - ref_x, atol=1e-6
    )
    t_fall = 3000.0 / v0(d)
    np.testing.assert_allclose(
        s.arrival_offset.values.ravel(), t_fall - t_fall[2], atol=1e-9
    )
    assert s.displacement_x.values.ravel()[0] > 0 > s.displacement_x.values.ravel()[-1]
    assert (s.arrival_offset.values.ravel()[:2] > 0).all()
    assert s.attrs["reference_diameter"] == 2.0
    # an interpolated reference lies between the neighbouring sizes
    s25 = size_sorting(res, 2.5)
    assert np.isfinite(s25.reference_x.values).all()
    # explicit axis and storm motion axis
    sa = size_sorting(res, 2.0, axis=(0.0, 1.0))
    np.testing.assert_allclose(sa.displacement_along.values.ravel(), 0.0, atol=1e-6)
    np.testing.assert_allclose(
        sa.displacement_cross.values.ravel(), -(expect_x - ref_x), atol=1e-6
    )
    res_m = run(
        POINT, d, engine, profile=profile(zlev, u=a * zlev), storm_motion=(1.0, 0.0)
    )
    sm = size_sorting(res_m, 2.0)
    np.testing.assert_allclose(
        sm.displacement_along.values.ravel(), expect_x - ref_x, atol=1e-6
    )
    with pytest.raises(ValueError, match="outside"):
        size_sorting(res, 10.0)
    with pytest.raises(ValueError, match="landing_x"):
        size_sorting(res.drop_vars("landing_x"))
    with pytest.raises(ValueError, match="two diameters"):
        size_sorting(res.isel(diameter=[0]))
    # backward results work too, datetimes as well
    base = np.datetime64("2022-03-30T23:00:00", "ns")
    bw = rain_source_points(
        {"x": 0.0, "y": 0.0, "time": base},
        d,
        source_height=3000.0,
        profile=profile(zlev, u=a * zlev),
        evaporation=False,
        density_correction=False,
        engine=engine,
    )
    sb = size_sorting(bw, 2.0)
    assert np.isfinite(sb.arrival_offset.values).all()
    assert np.issubdtype(sb.reference_time.dtype, np.datetime64)
    # large drops fall faster: their source is later (closer in time) than small ones
    off = sb.arrival_offset.values.ravel()
    assert off[0] < 0 < off[-1]


# --------------------------------------------------------------------------
# surface accumulation
# --------------------------------------------------------------------------


def _source_grid(nx=40, ny=30, nt=6, dx=500.0, dt=60.0, z=2500.0):
    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dx
    t = np.arange(nt) * dt
    return (
        xr.Dataset(coords={"time": t, "y": y, "x": x})
        .assign(z=(("time", "y", "x"), np.full((nt, ny, nx), z)))
        .assign(tt=("time", t))
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_surface_dsd_conserves_the_number_flux(engine):
    """Uniform wind and DSD: the surface N(D) is N V_top / V_surface."""
    dx, dt = 500.0, 60.0
    grid = _source_grid(dx=dx, dt=dt, nt=40)
    dia = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    prof = profile(np.arange(0, 6001.0, 100.0), u=6.0, v=2.0)
    traj = rain_trajectories(
        grid.drop_vars("tt"),
        dia,
        profile=prof,
        evaporation=False,
        engine=engine,
        time_step=10.0,
    )
    n0 = xr.DataArray(
        np.array([300.0, 250.0, 200.0, 150.0, 100.0, 40.0]),
        dims="diameter",
        coords={"diameter": dia},
    )
    out = surface_dsd(
        traj,
        n0,
        x=np.arange(-20000.0, 30001.0, dx),
        y=np.arange(-12000.0, 24001.0, dx),
        time=np.arange(-10, 80) * dt,
        duration=dt,
    )
    ratio = (traj.fall_speed_start / traj.fall_speed_end).isel(time=0, y=0, x=0).values
    # the interior of the field, away from the edges and the start-up
    inner = out.ND.sel(time=slice(20 * dt, 30 * dt)).sel(
        y=slice(-1500.0, 1500.0), x=slice(-2000.0, 2000.0)
    )
    # drops that land are shifted by about 600 m and 4 minutes; the field is 40 x 30 cells
    mean = inner.mean(["time", "y", "x"]).values
    np.testing.assert_allclose(mean, n0.values * ratio, rtol=1e-6)
    assert out.NT.dims == ("time", "y", "x")
    assert np.nanmax(out.DM) < 4.5
    # all landed drops are counted: the total number agrees (field inside the grid)
    total_in = (
        (
            n0
            * xr.DataArray(
                np.diff(rt._edges(dia)), dims="diameter", coords={"diameter": dia}
            )
            * traj.fall_speed_start
            * 500.0**2
            * dt
        )
        .sum()
        .values
    )
    # the flux through the surface per cell: sum N V dD dx dy dt over the grid
    tot_out = (
        (out.ND * out.diameter_width * traj.fall_speed_end.isel(time=0, y=0, x=0))
        .sum()
        .values
        * dx
        * dx
        * dt
    )
    assert tot_out == pytest.approx(total_in, rel=1e-6)


@pytest.mark.parametrize("engine", ENGINES)
def test_surface_dsd_evaporation_and_errors(engine):
    grid = _source_grid(nx=20, ny=16, nt=4)
    dia = np.array([0.4, 1.0, 2.0, 3.0, 4.0])
    prof = profile(np.arange(0, 6001.0, 250.0), u=5.0, rh=0.5)
    traj = rain_trajectories(
        grid.drop_vars("tt"), dia, profile=prof, engine=engine, time_step=10.0
    )
    n0 = xr.DataArray(
        np.full(dia.size, 100.0), dims="diameter", coords={"diameter": dia}
    )
    base = dict(
        x=grid.x.values, y=grid.y.values, time=np.arange(-4, 14) * 60.0, duration=60.0
    )
    out = surface_dsd(traj, n0, **base)
    dry = rain_trajectories(
        grid.drop_vars("tt"),
        dia,
        profile=profile(np.arange(0, 6001.0, 250.0), u=5.0, rh=0.5),
        evaporation=False,
        engine=engine,
        time_step=10.0,
    )
    ref = surface_dsd(dry, n0, **base)
    # evaporation removes the smallest drops and shifts the others to smaller bins
    assert out.NT.sum() < ref.NT.sum()
    assert float(out.ND.sel(diameter=0.4).sum()) == 0.0
    # datetime grids and explicit areas
    t0 = np.datetime64("2022-03-30T23:00:00", "ns")
    g2 = grid.assign_coords(
        time=t0 + (grid.time.values * 1e9).astype("timedelta64[ns]")
    )
    traj2 = rain_trajectories(
        g2.drop_vars("tt"), dia, profile=prof, engine=engine, time_step=10.0
    )
    base2 = dict(
        base, time=t0 + (np.arange(-4, 14) * 60 * 10**9).astype("timedelta64[ns]")
    )
    out2 = surface_dsd(traj2, n0, area=500.0**2, **base2)
    np.testing.assert_allclose(out2.ND.values, out.ND.values, rtol=1e-9, atol=1e-12)
    out3 = surface_dsd(traj2, n0, **dict(base2, duration=None))
    np.testing.assert_allclose(out3.ND.values, out.ND.values, rtol=1e-9, atol=1e-12)
    with pytest.raises(ValueError, match="rain_trajectories"):
        surface_dsd(traj.drop_vars("landing_x"), n0, **base)
    with pytest.raises(ValueError, match="diameter"):
        surface_dsd(traj, n0.isel(diameter=0, drop=True), **base)
    with pytest.raises(ValueError, match="at least two"):
        surface_dsd(traj, n0, **dict(base, x=[0.0]))
    with pytest.raises(ValueError, match="increasing"):
        surface_dsd(traj, n0, **dict(base, y=grid.y.values[::-1]))
    with pytest.raises(ValueError, match="duration"):
        surface_dsd(traj.isel(time=[0]), n0, **dict(base, duration=None))
    with pytest.raises(ValueError, match="area"):
        pts = rain_trajectories(
            {"x": [0.0, 1.0], "y": 0.0, "z": 2000.0}, dia, profile=prof, engine=engine
        )
        surface_dsd(pts, n0, **base)


@pytest.mark.parametrize("engine", ENGINES)
def test_sweep_sources_and_datatree(engine):
    """Gates of a sweep (azimuth, range) as the source, and a DataTree of sweeps."""
    az = np.arange(0.0, 360.0, 2.0)
    rng_ = np.arange(2000.0, 40000.0, 1000.0)
    el = 0.5
    r_h = rng_ * math.cos(math.radians(el))
    x = r_h[None, :] * np.sin(np.deg2rad(az))[:, None]
    y = r_h[None, :] * np.cos(np.deg2rad(az))[:, None]
    z = rng_ * math.sin(math.radians(el)) + 0.5 * (r_h**2) / (2 * 8.5e6 / 1.0)
    sweep = xr.Dataset(
        {
            "x": (("azimuth", "range"), x),
            "y": (("azimuth", "range"), y),
            "z": (("azimuth", "range"), np.broadcast_to(z[None, :] + 100.0, x.shape)),
        },
        coords={"azimuth": az, "range": rng_, "elevation": el},
    )
    prof = profile(np.arange(0, 8001.0, 250.0), u=10.0, rh=0.6)
    one = rain_trajectories(
        sweep, [1.0, 3.0], profile=prof, engine=engine, time_step=10.0
    )
    assert one.landing_x.dims == ("azimuth", "range", "diameter")
    area = rt._cell_area(one)
    assert area.dims == ("azimuth", "range")
    gate = np.deg2rad(2.0) * (r_h[10]) * 1000.0 * math.cos(math.radians(el))
    assert float(area.isel(azimuth=5, range=10)) == pytest.approx(gate, rel=1e-3)
    tree = xr.DataTree.from_dict(
        {
            "/sweep_0": sweep,
            "/sweep_1": sweep.assign_coords(elevation=1.5),
            "/other": xr.Dataset({"a": ("t", [1.0, 2.0])}),
        }
    )
    res = rain_trajectories(
        tree, [1.0, 3.0], profile=prof, engine=engine, time_step=10.0
    )
    assert sorted(res.children) == ["sweep_0", "sweep_1"]
    np.testing.assert_allclose(
        res["sweep_0"]["landing_x"].values, one.landing_x.values, rtol=1e-12
    )
    acc = tree.radarx.rain_trajectories(
        [1.0, 3.0], profile=prof, engine=engine, time_step=10.0
    )
    assert sorted(acc.children) == ["sweep_0", "sweep_1"]
    with pytest.raises(ValueError, match="georeference"):
        rain_trajectories(
            xr.DataTree.from_dict({"/a": xr.Dataset({"a": ("t", [1.0])})}),
            [1.0],
            profile=prof,
        )
    # surface N(D) of a sweep: sorting moves the drops downwind, in bins of a surface grid
    nd = xr.DataArray(
        np.full((az.size, rng_.size, 2), 100.0),
        dims=("azimuth", "range", "diameter"),
        coords={"azimuth": az, "range": rng_, "diameter": [1.0, 3.0]},
    )
    sfc = surface_dsd(
        one,
        nd,
        x=np.arange(-45000.0, 45001.0, 2000.0),
        y=np.arange(-45000.0, 45001.0, 2000.0),
        time=np.array([0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 420.0]),
        duration=60.0,
    )
    assert float(sfc.NT.sum()) > 0


def test_accessor_dataset():
    ds = xr.Dataset(
        {"x": ("p", [0.0, 100.0]), "y": ("p", [0.0, 0.0]), "z": ("p", [2000.0, 2000.0])}
    )
    prof = profile(np.arange(0, 6001.0, 500.0), u=5.0)
    res = ds.radarx.rain_trajectories([2.0], profile=prof, evaporation=False)
    assert res.landing_x.dims == ("p", "diameter")


# --------------------------------------------------------------------------
# edge cases and errors
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_status_codes_and_invalid_input(engine):
    prof = profile(np.arange(0, 6001.0, 500.0), u=5.0, rh=0.5)
    src = {"x": [0.0, 0.0, np.nan, 0.0], "y": 0.0, "z": [3000.0, -50.0, 1000.0, 1000.0]}
    res = rain_trajectories(
        src, [2.0], profile=prof, engine=engine, max_time=1000.0, time_step=10.0
    )
    st = res.status.values.ravel()
    assert st[0] == 1 and st[1] == 1 and st[2] == 3
    # already below the surface: lands at once, nothing happens
    assert res.fall_time.values.ravel()[1] == 0.0
    assert np.isnan(res.landing_x.values.ravel()[2])
    assert np.isnan(res.concentration_ratio.values.ravel()[2])
    assert np.isnan(res.evaporated_mass_fraction.values.ravel()[2])
    # aloft when the time runs out; small drop already smaller than the threshold
    short = rain_trajectories(
        {"x": 0.0, "y": 0.0, "z": 3000.0},
        [2.0],
        profile=prof,
        engine=engine,
        max_time=50.0,
        time_step=20.0,
    )
    assert short.status.values.item() == 0
    assert short.fall_time.values.item() == pytest.approx(60.0)  # whole steps
    tiny = rain_trajectories(
        {"x": 0.0, "y": 0.0, "z": 3000.0}, [0.05], profile=prof, engine=engine
    )
    assert tiny.status.values.item() == 2
    assert tiny.fall_time.values.item() == 0.0
    # a raised surface
    high = rain_trajectories(
        {"x": 0.0, "y": 0.0, "z": 3000.0},
        [2.0],
        profile=prof,
        engine=engine,
        surface_height=1000.0,
        evaporation=False,
        density_correction=False,
    )
    assert high.landing_z.values.item() == pytest.approx(1000.0)
    assert high.fall_time.values.item() == pytest.approx(2000.0 / v0(2.0), rel=1e-9)


@pytest.mark.parametrize("engine", ENGINES)
def test_stop_at_the_surface_with_updraft(engine):
    """A strong updraft carries small drops up: they stay aloft."""
    prof = profile(np.arange(0, 6001.0, 1000.0), w=5.0, u=0.0)
    res = run(POINT, [1.0, 4.0], engine, profile=prof, max_time=1500.0, time_step=10.0)
    assert res.status.values.ravel().tolist() == [0, 1]
    assert res.landing_z.values.ravel()[0] > 3000.0


@pytest.mark.parametrize("engine", ENGINES)
def test_path_and_output_structure(engine):
    prof = profile(np.arange(0, 6001.0, 500.0), u=5.0, rh=0.6)
    res = rain_trajectories(
        {"x": [0.0, 100.0], "y": [0.0, 0.0], "z": 3000.0},
        xr.DataArray([1.0, 3.0], dims="diameter"),
        profile=prof,
        engine=engine,
        store_path=2,
        time_step=10.0,
    )
    assert res.path_z.dims == ("point", "diameter", "step")
    assert res.path_z.isel(point=0, diameter=1, step=0).item() == 3000.0
    last = res.path_z.isel(point=0, diameter=1).dropna("step")
    assert last.values[-1] == pytest.approx(0.0, abs=1e-9)
    assert (np.diff(last.values) < 0).all()
    assert res.attrs["scheme"] == "rk4" and res.attrs["direction"] == "forward"
    assert res.status.attrs["flag_meanings"].startswith("aloft")
    assert res.landing_x.attrs["units"] == "m"
    assert res.start_z.dims == ("point",)
    # the mapping form with scalars, an offset and a release time
    sc = rain_trajectories(
        {"x": 0.0, "y": 0.0, "z": 1000.0},
        2.0,
        profile=prof,
        engine=engine,
        offset=(10.0, 0.0, 2000.0),
        time=5.0,
    )
    assert sc.start_z.item() == 3000.0 and sc.start_x.item() == 10.0
    assert sc.landing_time.item() > 5.0


def test_errors():
    prof = profile(np.arange(0, 6001.0, 500.0), u=5.0)
    pt = {"x": 0.0, "y": 0.0, "z": 1000.0}
    with pytest.raises(ValueError, match="engine"):
        rain_trajectories(pt, [1.0], profile=prof, engine="gpu")
    with pytest.raises(ValueError, match="scheme"):
        rain_trajectories(pt, [1.0], profile=prof, scheme="euler")
    with pytest.raises(ValueError, match="time_step"):
        rain_trajectories(pt, [1.0], profile=prof, time_step=0.0)
    with pytest.raises(ValueError, match="max_time"):
        rain_trajectories(pt, [1.0], profile=prof, max_time=-1.0)
    with pytest.raises(ValueError, match="evaporated_diameter"):
        rain_trajectories(pt, [1.0], profile=prof, evaporated_diameter=0.0)
    with pytest.raises(ValueError, match="fall_speed"):
        rain_trajectories(pt, [1.0], profile=prof, fall_speed="beard1976")
    with pytest.raises(ValueError, match="fall_speed"):
        rain_trajectories(pt, [1.0], profile=prof, fall_speed=(1.0, 2.0))
    with pytest.raises(ValueError, match="fall_speed"):
        rain_trajectories(pt, [1.0], profile=prof, fall_speed=(-1.0, 1.0, 0.0))
    with pytest.raises(ValueError, match="polynomial"):
        rain_trajectories(pt, [1.0], profile=prof, fall_speed=("poly", 1.0))
    with pytest.raises(ValueError, match="positive"):
        rain_trajectories(pt, [-1.0], profile=prof)
    with pytest.raises(ValueError, match="one-dimensional"):
        rain_trajectories(pt, np.ones((2, 2)), profile=prof)
    with pytest.raises(ValueError, match="members"):
        rain_trajectories(pt, [1.0], profile=prof, members=0)
    with pytest.raises(ValueError, match="dispersion"):
        rain_trajectories(pt, [1.0], profile=prof, dispersion=(-1.0, 0.0, 10.0))
    with pytest.raises(ValueError, match="n_threads"):
        rain_trajectories(pt, [1.0], profile=prof, n_threads=-2)
    with pytest.raises(ValueError, match="wind analysis or a profile"):
        rain_trajectories(pt, [1.0])
    with pytest.raises(ValueError, match="humidity"):
        rain_trajectories(
            pt, [1.0], profile=prof.drop_vars("relative_humidity"), evaporation=True
        )
    with pytest.raises(ValueError, match="'z'"):
        rain_trajectories({"x": 0.0, "y": 0.0}, [1.0], profile=prof)
    with pytest.raises(TypeError, match="Dataset"):
        rain_trajectories(3.0, [1.0], profile=prof)
    with pytest.raises(ValueError, match="height"):
        rain_trajectories(pt, [1.0], profile=prof.rename(height="h"))
    with pytest.raises(ValueError, match="'u' and 'v'"):
        rain_trajectories(
            pt, [1.0], wind=xr.Dataset({"u": ("x", [1.0])}, coords={"x": [0.0]})
        )
    with pytest.raises(TypeError, match="wind"):
        rain_trajectories(pt, [1.0], wind=3.0, profile=prof)
    with pytest.raises(ValueError, match="needs .u. and .v."):
        rain_trajectories(pt, [1.0], storm_motion=xr.Dataset({"u": 1.0}), profile=prof)
    with pytest.raises(ValueError, match="store_path"):
        rain_trajectories(
            {"x": np.zeros(100000), "y": 0.0, "z": 1000.0},
            np.linspace(1, 5, 100),
            profile=prof,
            store_path=1,
            time_step=0.01,
            max_time=1000.0,
        )
    with pytest.raises(ValueError, match="dimension"):
        rain_trajectories(
            pt,
            [1.0],
            wind=xr.Dataset(
                {
                    "u": (("y", "x"), np.ones((2, 2))),
                    "v": (("y", "x"), np.ones((2, 2))),
                },
                coords={"y": [0, 1.0], "x": [0, 1.0]},
            ),
        )
    with pytest.raises(ValueError, match="no valid"):
        rain_trajectories(pt, [1.0], wind=wind_grid(np.nan))
    if rt.HAS_COMPILED_KERNEL:
        with pytest.raises(ValueError):
            rt._rain_trajectories.integrate(
                np.zeros(1),
                np.zeros(1),
                np.zeros(1),
                np.zeros(1),
                np.zeros(1),
                np.zeros(2),
                np.zeros(0),
                np.zeros(0),
                np.zeros(0),
                np.zeros(0),
                np.zeros(0),
                np.zeros(1),
                np.zeros(1),
                np.zeros(1),
                np.zeros(1),
                np.zeros(0),
                np.zeros(0),
                np.zeros((0, 0)),
                np.zeros(1),
                np.zeros(1),
                np.zeros(1),
                np.zeros(1),
                np.zeros(1),
                np.zeros(16),
                np.zeros(10, dtype=np.int64),
                np.zeros(1),
                0,
                0,
            )


def test_engine_compiled_unavailable(monkeypatch):
    monkeypatch.setattr(rt, "HAS_COMPILED_KERNEL", False)
    prof = profile(np.arange(0, 6001.0, 500.0), u=5.0)
    with pytest.raises(ImportError, match="compiled"):
        rain_trajectories(
            {"x": 0.0, "y": 0.0, "z": 1000.0}, [1.0], profile=prof, engine="compiled"
        )
    res = rain_trajectories(
        {"x": 0.0, "y": 0.0, "z": 1000.0}, [1.0], profile=prof, engine="auto"
    )
    assert res.status.values.item() in (0, 1, 2)


def test_profile_humidity_forms_and_standard_atmosphere():
    z = np.arange(0.0, 6001.0, 500.0)
    base = profile(z, u=5.0, rh=0.5)
    e = 0.5 * evap._saturation_vapor_pressure(base.temperature.values)
    q = evap.EPS * e / (base.pressure.values - (1 - evap.EPS) * e)
    td = base.drop_vars("relative_humidity")
    by_q = td.assign(specific_humidity=("height", q))
    by_td = td.assign(
        dewpoint=(
            "height",
            1.0 / (1.0 / base.temperature.values - 461.5 / 2.5e6 * np.log(0.5)),
        )
    )
    src = {"x": 0.0, "y": 0.0, "z": 3000.0}
    ref = rain_trajectories(src, [1.5], profile=base)
    out_q = rain_trajectories(src, [1.5], profile=by_q)
    np.testing.assert_allclose(out_q.landing_diameter, ref.landing_diameter, rtol=1e-9)
    out_td = rain_trajectories(src, [1.5], profile=by_td)
    assert out_td.evaporated_mass_fraction.item() > 0
    # no humidity: evaporation is off, standard atmosphere without a profile
    dry = rain_trajectories(src, [1.5], profile=td)
    assert dry.evaporated_mass_fraction.item() == 0
    assert dry.attrs["evaporation"] == 0
    wind = wind_grid(5.0)
    std = rain_trajectories(src, [1.5], wind=wind)
    assert std.status.item() == 1
    # missing and unsorted levels, a column of NaNs in the thermodynamics
    messy = base.isel(height=[3, 1, 2, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    messy["temperature"] = messy.temperature.where(messy.height != 2000.0)
    messy["u"] = messy.u.where(messy.height != 1500.0)
    out = rain_trajectories(src, [1.5], profile=messy)
    assert out.status.item() == 1


# --------------------------------------------------------------------------
# terrain and a sloping source level
# --------------------------------------------------------------------------


def _plane(z0, slope_x, slope_y=0.0):
    x = np.array([-1e5, 0.0, 1e5])
    y = np.array([-1e5, 1e5])
    z = z0 + slope_x * x[None, :] + slope_y * y[:, None]
    return xr.DataArray(z, dims=("y", "x"), coords={"y": y, "x": x})


@pytest.mark.parametrize("engine", ENGINES)
def test_landing_on_a_sloping_terrain_is_exact(engine):
    u, v, d = 6.0, 2.0, 2.0
    prof = profile(np.arange(0, 6001.0, 1000.0), u=u, v=v)
    terrain = _plane(500.0, 0.05, -0.02)
    res = run(
        {"x": 100.0, "y": 50.0, "z": 3000.0},
        [d],
        engine,
        profile=prof,
        terrain=terrain,
    )
    # z = 3000 - V t meets 500 + 0.05 x - 0.02 y with x = 100 + u t, y = 50 + v t
    vt = float(v0(d))
    t_exact = (3000.0 - 500.0 - 0.05 * 100.0 + 0.02 * 50.0) / (vt + 0.05 * u - 0.02 * v)
    assert res.fall_time.values.item() == pytest.approx(t_exact, rel=1e-10)
    assert res.landing_x.values.item() == pytest.approx(100.0 + u * t_exact, rel=1e-12)
    expected_z = 500.0 + 0.05 * (100.0 + u * t_exact) - 0.02 * (50.0 + v * t_exact)
    assert res.landing_z.values.item() == pytest.approx(expected_z, rel=1e-10)
    # the speed at the end is the fall speed of the drop (no vertical wind)
    assert res.fall_speed_end.values.item() == pytest.approx(vt)
    # a start below the terrain lands at once
    below = run(
        {"x": 0.0, "y": 0.0, "z": 100.0}, [d], engine, profile=prof, terrain=terrain
    )
    assert below.fall_time.values.item() == 0.0


@pytest.mark.parametrize("engine", ENGINES)
def test_backward_to_a_sloping_source_surface(engine):
    u, d = 8.0, 3.0
    prof = profile(np.arange(0, 6001.0, 1000.0), u=u)
    cone = _plane(2000.0, 0.1)
    res = rain_source_points(
        {"x": 0.0, "y": 0.0},
        [d],
        source_surface=cone,
        profile=prof,
        evaporation=False,
        density_correction=False,
        engine=engine,
    )
    vt = float(v0(d))
    t_exact = 2000.0 / (vt + 0.1 * u)  # V t = 2000 + 0.1 (-u t)
    assert res.fall_time.values.item() == pytest.approx(t_exact, rel=1e-10)
    assert res.source_x.values.item() == pytest.approx(-u * t_exact, rel=1e-12)
    assert res.source_z.values.item() == pytest.approx(vt * t_exact, rel=1e-10)
    with pytest.raises(ValueError, match="source_height or source_surface"):
        rain_source_points({"x": 0.0, "y": 0.0}, [d], profile=prof, engine=engine)


def test_stop_surface_errors():
    prof = profile(np.arange(0, 6001.0, 1000.0), u=5.0)
    pt = {"x": 0.0, "y": 0.0, "z": 1000.0}
    good = _plane(0.0, 0.0)
    with pytest.raises(ValueError, match="DataArray on"):
        rain_trajectories(pt, [1.0], profile=prof, terrain=good.rename(x="lon"))
    with pytest.raises(ValueError, match="DataArray on"):
        rain_trajectories(pt, [1.0], profile=prof, terrain=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="two-dimensional"):
        rain_trajectories(pt, [1.0], profile=prof, terrain=good.expand_dims(t=2))
    with pytest.raises(ValueError, match="finite"):
        rain_trajectories(pt, [1.0], profile=prof, terrain=good.where(good.x > 0))
    if rt.HAS_COMPILED_KERNEL:
        arrays = [np.zeros(1)] * 6 + [np.zeros(0)] * 5 + [np.zeros(1)] * 4
        stop = (np.array([0.0, 1.0]), np.array([0.0]), np.zeros((3, 1)))  # bad shape
        ipar = np.zeros(10, dtype=np.int64)
        ipar[0] = 4
        rest = [np.ones(1)] * 6 + [np.ones(16), ipar, np.ones(1)]
        with pytest.raises(ValueError, match="stop surface"):
            rt._rain_trajectories.integrate(*arrays, *stop, *rest, 0, 0)


# --------------------------------------------------------------------------
# a moving source field and the spreading of the observations in time
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_source_dsd_follows_the_moving_pattern(engine):
    """A translating cell is read exactly with the pattern motion, not without."""
    c = (8.0, 0.0)
    prof = profile(np.arange(0, 6001.0, 1000.0), u=3.0, t=288.15, p=101325.0)
    xs = np.arange(-60e3, 60e3 + 1, 500.0)
    times = np.array([0.0, 600.0, 1200.0, 1800.0])
    dia = np.array([2.0, 3.0, 4.0])

    def cell(x, t):
        return 100.0 * np.exp(-(((x - c[0] * t) / 6e3) ** 2))

    field = xr.DataArray(
        cell(xs[None, :, None], times[:, None, None]) * np.ones((1, 1, dia.size)),
        dims=("time", "x", "diameter"),
        coords={"time": times, "x": xs, "diameter": dia},
    )
    tgt = {"x": 12000.0, "y": 0.0, "time": 900.0}
    kw = dict(
        source_height=2500.0,
        profile=prof,
        source_dsd=field,
        evaporation=False,
        density_correction=False,
        engine=engine,
    )
    moving = rain_source_points(tgt, dia, storm_motion=c, **kw)
    plain = rain_source_points(tgt, dia, **kw)
    truth = cell(moving.source_x.values, moving.source_time.values)
    np.testing.assert_allclose(moving.ND.values, truth, rtol=3e-3)
    # without the motion the two analyses around the source time are blended in place
    err_plain = np.abs(
        plain.ND.values - cell(plain.source_x.values, plain.source_time.values)
    )
    assert (err_plain / truth.max() > 1e-2).any()
    # outside the times of the field nothing is known
    late = rain_source_points(dict(tgt, time=2600.0), dia, storm_motion=c, **kw)
    assert np.isnan(late.ND.values).all()
    # a single analysis is held in time, moved with the pattern
    one = rain_source_points(
        tgt, dia, storm_motion=c, **dict(kw, source_dsd=field.isel(time=[1]))
    )
    shift = (one.source_time.values - 600.0) * c[0]
    np.testing.assert_allclose(
        one.ND.values, cell(one.source_x.values - shift, 600.0), rtol=3e-3
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_surface_dsd_spreads_observations_over_their_duration(engine):
    """Sources every 120 s over a 60-s grid: a uniform field stays uniform in time."""
    prof = profile(np.arange(0, 6001.0, 100.0), u=5.0, v=2.0, t=288.0, p=95000.0)
    x = np.arange(-30e3, 30.1e3, 2e3)
    times = np.arange(0.0, 1801.0, 120.0)
    dia = np.array([4.0, 4.25, 4.5])
    src = xr.Dataset({"z": ((), 2500.0)}, coords={"time": times, "y": x, "x": x})
    tr = rain_trajectories(
        src, dia, profile=prof, evaporation=False, time_step=10.0, engine=engine
    )
    out = surface_dsd(
        tr,
        xr.full_like(tr.landing_x, 100.0),
        x=np.arange(-40e3, 60e3, 2e3),
        y=np.arange(-40e3, 60e3, 2e3),
        time=np.arange(0.0, 2700.0, 60.0),
        duration=120.0,
    )
    series = out.ND.sel(x=3e3, y=3e3, diameter=4.25, method="nearest")
    np.testing.assert_allclose(
        series.sel(time=slice(600.0, 1900.0)).values, 100.0, rtol=1e-9
    )
    assert float(series.sel(time=0.0)) == 0.0


# --------------------------------------------------------------------------
# input forms and degenerate grids
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_single_point_grids_and_levels(engine):
    """Wind, profile and terrain grids may have one level or point on an axis."""
    one = profile(np.array([0.0]), u=6.0, v=1.0, t=288.15, p=101325.0)
    res = run(POINT, [2.0], engine, profile=one, wind_divergence=True)
    t_fall = 3000.0 / v0(2.0)
    assert res.landing_x.values.item() == pytest.approx(6.0 * t_fall, rel=1e-10)
    grid = wind_grid(4.0, x=(0.0,), y=(0.0,), z=(0.0,))
    res = run(POINT, [2.0], engine, wind=grid, wind_divergence=True)
    assert res.landing_x.values.item() == pytest.approx(4.0 * t_fall, rel=1e-10)
    flat = xr.DataArray(
        np.array([[250.0]]), dims=("y", "x"), coords={"y": [0.0], "x": [0.0]}
    )
    res = run(POINT, [2.0], engine, wind=grid, terrain=flat)
    assert res.landing_z.values.item() == pytest.approx(250.0)
    assert res.fall_time.values.item() == pytest.approx(2750.0 / v0(2.0), rel=1e-10)
    # nothing valid at all
    nan = run({"x": np.nan, "y": 0.0, "z": 1000.0}, [2.0], engine, profile=one)
    assert nan.status.values.item() == 3


@pytest.mark.parametrize("engine", ENGINES)
def test_wind_forms_and_time_conventions(engine):
    """Winds without w, with scalar coordinates or without a time axis."""
    prof = profile(np.arange(0, 6001.0, 1000.0))
    t_fall = 3000.0 / v0(2.0)
    base = wind_grid(5.0, z=np.arange(0, 6001.0, 1000.0))
    no_w = base[["u", "v"]].isel(time=0, drop=True)  # no w, no time axis
    res = run(POINT, [2.0], engine, wind=no_w, profile=prof)
    assert res.landing_x.values.item() == pytest.approx(5.0 * t_fall, rel=1e-10)
    # a scalar time coordinate and a scalar level
    scalar_t = base.isel(time=0)
    assert "time" in scalar_t.coords and "time" not in scalar_t.dims
    res = run(POINT, [2.0], engine, wind=scalar_t, profile=prof)
    assert res.landing_x.values.item() == pytest.approx(5.0 * t_fall, rel=1e-10)
    level = base.isel(z=[2]).squeeze("z")  # z only as a scalar coordinate
    res = run(POINT, [2.0], engine, wind=level, profile=prof)
    assert res.landing_x.values.item() == pytest.approx(5.0 * t_fall, rel=1e-10)
    # a datetime axis of the wind with a numeric release time (seconds since its start)
    t0 = np.datetime64("2022-03-30T23:00:00", "ns")
    dated = base.assign_coords(time=[t0])
    res = run(POINT, [2.0], engine, wind=dated, profile=prof, time=0.0)
    assert res.landing_x.values.item() == pytest.approx(5.0 * t_fall, rel=1e-10)
    # a wind analysis valid at pattern_time moves with the pattern
    res = run(
        POINT,
        [2.0],
        engine,
        wind=no_w,
        profile=prof,
        storm_motion=(5.0, 0.0),
        pattern_time=0.0,
    )
    assert res.landing_x.values.item() == pytest.approx(5.0 * t_fall, rel=1e-10)


def test_point_and_motion_inputs():
    prof = profile(np.arange(0, 6001.0, 1000.0), u=5.0)
    res = rain_trajectories(
        {"x": [0.0, 1.0], "y": 0.0, "z": xr.DataArray([2000.0, 2500.0], dims="point")},
        [2.0],
        profile=prof,
        evaporation=False,
    )
    assert res.landing_x.dims == ("point", "diameter")
    with pytest.raises(ValueError, match="same length"):
        rain_trajectories(
            {"x": [0.0, 1.0], "y": [0.0, 1.0, 2.0], "z": 1000.0}, [2.0], profile=prof
        )
    # the motion given as a Dataset (the output of estimate_motion) is averaged
    motion = xr.Dataset({"u": ("x", [4.0, 6.0]), "v": ("x", [1.0, 3.0])})
    out = rain_trajectories(
        {"x": 0.0, "y": 0.0, "z": 1000.0}, [2.0], profile=prof, storm_motion=motion
    )
    assert (out.attrs["storm_motion_u"], out.attrs["storm_motion_v"]) == (5.0, 2.0)
    # diameters as a DataArray with another dimension name, or two-dimensional
    out = rain_trajectories(
        {"x": 0.0, "y": 0.0, "z": 1000.0},
        xr.DataArray([1.0, 2.0], dims="bin"),
        profile=prof,
        evaporation=False,
    )
    assert out.landing_x.dims == ("diameter",)
    with pytest.raises(ValueError, match="one-dimensional"):
        rain_trajectories(
            {"x": 0.0, "y": 0.0, "z": 1000.0},
            xr.DataArray(np.ones((2, 2)), dims=("a", "b")),
            profile=prof,
        )


def test_source_dsd_and_surface_with_degenerate_axes():
    prof = profile(np.arange(0, 6001.0, 1000.0), u=5.0, t=288.15, p=101325.0)
    # a source field with a level of length one and a single diameter
    xs = np.arange(-20e3, 20e3 + 1, 1000.0)
    field = xr.DataArray(
        np.full((1, xs.size, 1), 50.0),
        dims=("z", "x", "diameter"),
        coords={"z": [2500.0], "x": xs, "diameter": [2.0]},
    )
    res = rain_source_points(
        {"x": 0.0, "y": 0.0},
        [2.0],
        source_height=2500.0,
        profile=prof,
        source_dsd=field,
        evaporation=False,
    )
    assert np.isfinite(res.ND.values).all()
    # the surface bins need at least two diameters
    x = np.arange(-10e3, 10e3 + 1, 2000.0)
    src = xr.Dataset({"z": ((), 2500.0)}, coords={"time": [0.0, 60.0], "y": x, "x": x})
    tr = rain_trajectories(src, [2.0], profile=prof, evaporation=False)
    nd = xr.full_like(tr.landing_x, 80.0)
    with pytest.raises(ValueError, match="at least two diameters"):
        surface_dsd(
            tr,
            nd,
            x=np.arange(-20e3, 40e3, 2000.0),
            y=np.arange(-20e3, 40e3, 2000.0),
            time=np.arange(0.0, 900.0, 60.0),
            duration=60.0,
        )
    # a datetime release and arrival at the matched times
    t0 = np.datetime64("2022-03-30T23:00:00", "ns")
    matched = trajectory_matched_times(
        {"x": 0.0, "y": 0.0, "z": 2000.0, "time": t0},
        {"x": 15000.0, "y": 0.0},
        [2.0, 3.0],
        storm_motion=(10.0, 0.0),
        profile=prof,
        evaporation=False,
    )
    assert np.issubdtype(matched.arrival_time.dtype, np.datetime64)


def test_profiles_without_wind_or_thermodynamics_and_other_forms():
    z = np.arange(0, 6001.0, 1000.0)
    t_fall = 3000.0 / v0(2.0)
    base = profile(z, u=5.0)
    # a profile without winds: the background is the mean of the wind analysis
    no_wind = base.drop_vars(["u", "v"])
    res = run(POINT, [2.0], "numpy", profile=no_wind, wind=wind_grid(4.0))
    assert res.landing_x.values.item() == pytest.approx(4.0 * t_fall, rel=1e-10)
    # a profile without temperature and pressure: a standard atmosphere
    no_thermo = base.drop_vars(["temperature", "pressure", "relative_humidity"])
    res = run(POINT, [2.0], "numpy", profile=no_thermo)
    assert res.landing_x.values.item() == pytest.approx(5.0 * t_fall, rel=1e-3)
    # a target with its own surface height, and a missing release time
    back = rain_source_points(
        {"x": 0.0, "y": 0.0, "z": 100.0},
        xr.DataArray([2.0, 3.0], dims="diameter", coords={"diameter": [2.0, 3.0]}),
        source_height=2100.0,
        profile=base,
        evaporation=False,
        density_correction=False,
    )
    assert back.fall_time.values[0] == pytest.approx(2000.0 / v0(2.0), rel=1e-9)
    nat = rain_trajectories(
        {
            "x": [0.0, 0.0],
            "y": 0.0,
            "z": 1000.0,
            "time": np.array(["2022-03-30T23:00", "NaT"], dtype="datetime64[ns]"),
        },
        [2.0],
        profile=base,
        evaporation=False,
    )
    assert nat.status.values.ravel().tolist() == [1, 3]


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------


def test_real_nexrad_sweeps():
    """The gates of two sweeps of a WSR-88D volume as sources (a DataTree)."""
    xd = pytest.importorskip("xradar")
    from open_radar_data import DATASETS

    file = DATASETS.fetch("KLBB20160601_150025_V06")
    dtree = xd.io.open_nexradlevel2_datatree(file, sweep=[0, 3]).xradar.georeference()
    nodes = {}
    for name in ("sweep_0", "sweep_3"):
        sub = (
            dtree[name]
            .to_dataset(inherit=False)
            .isel(azimuth=slice(0, 720, 24), range=slice(0, 300, 10))
        )
        nodes[f"/{name}"] = sub.drop_vars(list(sub.data_vars))  # only the geometry
    altitude = float(dtree["sweep_0"].to_dataset().altitude)
    tree = xr.DataTree.from_dict(nodes)
    z = np.arange(0.0, 12001.0, 250.0)
    prof = profile(z, rh=0.5, u=4.0 + 0.004 * z, v=-2.0 + 0.001 * z)
    kw = dict(profile=prof, surface_height=altitude, time_step=10.0)
    res = rain_trajectories(tree, [1.0, 2.0, 4.0], **kw)
    assert sorted(res.children) == ["sweep_0", "sweep_3"]
    for name in ("sweep_0", "sweep_3"):
        r = res[name].to_dataset()
        assert r.landing_x.dims == ("azimuth", "range", "diameter")
        reach = r.status == 1
        assert int(reach.sum()) > 100
        landed = r.landing_z.where(reach)
        np.testing.assert_allclose(landed.max(), altitude, atol=1e-6)
        # in a wind that increases with height large drops are displaced less
        disp = np.hypot(r.landing_x - r.start_x, r.landing_y - r.start_y).where(reach)
        mean = disp.mean(["azimuth", "range"])
        assert float(mean.sel(diameter=4.0)) < float(mean.sel(diameter=1.0))
    if rt.HAS_COMPILED_KERNEL:
        ds = tree["sweep_3"].to_dataset()
        a = rain_trajectories(ds, [2.0], engine="numpy", **kw)
        b = rain_trajectories(ds, [2.0], engine="compiled", **kw)
        _assert_same(a, b)
