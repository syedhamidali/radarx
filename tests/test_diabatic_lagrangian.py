#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests of radarx.retrieve.diabatic_lagrangian (Ziegler 2013 DLA)."""

import importlib
import math

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401  registers the accessors
from radarx.retrieve import (
    diabatic_lagrangian,
    microphysical_rates,
    polarimetric_precipitation,
    ziegler2013_precipitation,
)
from radarx.retrieve import lagrangian as lg

dl = importlib.import_module("radarx.retrieve.diabatic_lagrangian")

needs_kernel = pytest.mark.skipif(
    not lg.HAS_COMPILED_KERNEL, reason="compiled trajectory kernel not built"
)

DIMS = ("time", "z", "y", "x")
T0 = np.datetime64("2022-03-30T23:00:00", "ns")
KAPPA = 0.2854
EPS = 287.04 / 461.5


def _winds(u=0.0, v=0.0, w=0.0, dbz=45.0, nt=3, step=1800.0, extra=None):
    c = np.arange(0.0, 8001.0, 1000.0)
    z = np.arange(0.0, 6001.0, 500.0)
    t = np.arange(nt) * step
    tt, zz, yy, xx = np.meshgrid(t, z, c, c, indexing="ij")
    data = {}
    fields = {"u": u, "v": v, "w": w, "DBZ": dbz, **(extra or {})}
    for name, f in fields.items():
        val = f(tt, xx, yy, zz) if callable(f) else np.full(tt.shape, float(f))
        data[name] = (DIMS, val)
    times = T0 + (t * 1e9).astype("timedelta64[ns]")
    return xr.Dataset(data, coords={"time": times, "z": z, "y": c, "x": c})


def _sounding(theta_lapse=None, q_low=0.014, q_high=0.004, ground_theta=300.0):
    """Hydrostatic sounding; theta linear in height if theta_lapse is given."""
    h = np.arange(0.0, 12001.0, 50.0)
    if theta_lapse is None:
        t = 300.0 - 0.0065 * h
        p = 1e5 * (t / 300.0) ** (9.80665 / (287.04 * 0.0065))
    else:
        th = ground_theta + theta_lapse * h
        # hydrostatic Exner function: dPi/dz = -g / (cp theta)
        cp = 287.04 / KAPPA
        pi = 1.0 - np.concatenate(
            ([0.0], np.cumsum(9.80665 / (cp * 0.5 * (th[1:] + th[:-1])) * np.diff(h)))
        )
        p = 1e5 * pi ** (1.0 / KAPPA)
        t = th * pi
    q = np.where(h < 1500, q_low, q_high)
    return xr.Dataset(
        {
            "pressure": ("height", p),
            "temperature": ("height", t),
            "specific_humidity": ("height", q),
            "u": ("height", 0.0 * h),
            "v": ("height", 0.0 * h),
        },
        coords={"height": h},
    )


def _precip(ds, qr=0.0, nr=0.0, qg=0.0, ng=0.0):
    shape = tuple(ds.sizes[d] for d in DIMS)
    return xr.Dataset(
        {
            k: (DIMS, np.full(shape, val))
            for k, val in (("qr", qr), ("nr", nr), ("qg", qg), ("ng", ng))
        },
        coords={d: ds[d] for d in DIMS},
    )


def _es(t):
    tc = t - 273.15
    return 611.2 * np.exp(17.67 * tc / (tc + 243.5))


def _theta_e(t, p, r):
    """Bolton (1980) eq. (43) equivalent potential temperature."""
    e = p * r / (EPS + r) / 100.0
    tl = 2840.0 / (3.5 * np.log(t) - np.log(e) - 4.805) + 55.0
    return (
        t
        * (1e5 / p) ** (0.2854 * (1 - 0.28 * r))
        * np.exp((3.376 / tl - 0.00254) * 1e3 * r * (1 + 0.81 * r))
    )


NONE = {k: False for k in dl.PROCESSES}


# --------------------------------------------------------------------------
# thermodynamics
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ["auto", "numpy"])
def test_dry_adiabatic_descent_conserves_theta(engine):
    ds = _winds(w=-2.0, dbz=40.0)
    snd = _sounding(theta_lapse=0.004, q_low=0.002, q_high=0.002)
    out = diabatic_lagrangian(
        ds,
        snd,
        precipitation="none",
        processes={"surface_downdraft": False},
        engine=engine,
    )
    # theta of every parcel is the base-state theta of its origin
    zo = out.origin_z.values
    np.testing.assert_allclose(out.theta.values, 300.0 + 0.004 * zo, atol=1e-3)
    assert (out.origin_z.isel(z=slice(1, 9)) > out.z.isel(z=slice(1, 9))).all()
    assert float(abs(out.qc).max()) == 0.0
    assert out.environment.all()
    assert float(abs(out.dtheta_condensation).max()) == 0.0
    # warm, dry descent: positive delta theta_v
    assert (out.delta_theta_v.isel(z=slice(1, 8)) > 0).all()


@pytest.mark.parametrize("engine", ["auto", "numpy"])
def test_saturated_ascent_conserves_theta_e(engine):
    ds = _winds(w=lambda t, x, y, z: np.where(z > 0, 3.0, 0.0), dbz=10.0)
    snd = _sounding()
    out = diabatic_lagrangian(
        ds, snd, precipitation="none", engine=engine, filter_passes=0, hole_fill=False
    )
    col = out.isel(x=4, y=4)
    p = col.pressure_base.values
    t = col.temperature.values
    qv, qc = col.qv.values, col.qc.values
    r0 = 0.014 / (1 - 0.014)
    cloud = qc > 1e-4
    assert cloud.sum() >= 8 and not cloud[0]
    # saturated, total water conserved: q_c = q_t - q_vs(T) on the moist adiabat
    qvs = EPS * _es(t) / (p - _es(t))
    np.testing.assert_allclose(qv[cloud], qvs[cloud], rtol=1e-6)
    np.testing.assert_allclose((qv + qc)[cloud], r0, rtol=1e-9)
    # theta_e of the parcel (Bolton 1980) conserved within the accuracy of the
    # reversible adjustment (condensate kept) and Bolton's fit
    zo = col.origin_z.values
    p0 = np.interp(zo, snd.height.values, snd.pressure.values)
    t0 = np.interp(zo, snd.height.values, snd.temperature.values)
    the0 = _theta_e(t0, p0, r0)
    the = _theta_e(t[cloud], p[cloud], qv[cloud])
    assert np.abs(the - the0[cloud]).max() < 1.5
    # the latent heating is the condensation term of the budget
    th0 = np.interp(
        zo, snd.height.values, (snd.temperature * (1e5 / snd.pressure) ** KAPPA).values
    )
    np.testing.assert_allclose(
        col.dtheta_condensation.values, col.theta.values - th0, atol=1e-3
    )


def test_damping_rate_and_relaxation():
    # dry descent at w = -1 through a stable layer, damping only, b = 0:
    # d theta/dt = -K (theta - theta_B(z(t))), K = c_d |w| / (L_d0 + (|w| - W0) L_W)
    lapse = 0.005
    ds = _winds(w=-1.0, dbz=40.0)
    snd = _sounding(theta_lapse=lapse, q_low=0.001, q_high=0.001)
    pr = _precip(ds, qr=3e-3, nr=1e3)
    procs = {**NONE, "damping": True}
    out = diabatic_lagrangian(
        ds,
        snd,
        precipitation=pr,
        processes=procs,
        parameters={"b": 0.0},
        filter_passes=0,
    )
    k = 0.2 * 1.0 / (300.0 + (1.0 - 0.1) * 100.0)
    f = math.exp(-k * 20.0)
    col = out.isel(x=4, y=4)
    for lev in (2, 4, 6):
        n = int(col.n_steps[lev])
        assert n == 77 and float(col.origin_z[lev]) == float(col.z[lev]) + 20.0 * n
        # e = theta - theta_B(z): the base state warms by lapse * |w| * dt per
        # step and the departure decays by exp(-K dt) (exact over a step)
        e = 0.0
        for _ in range(n):
            e = (e + lapse * 20.0) * f
        th_b = 300.0 + lapse * float(col.z[lev])
        np.testing.assert_allclose(float(col.theta[lev]), th_b + e, atol=2e-4)
        # close to the continuous solution e = lapse/K (1 - exp(-K t))
        cont = lapse / k * (1.0 - math.exp(-k * 20.0 * n))
        assert abs(e - cont) < 0.02 * cont
        assert float(col.dtheta_damping[lev]) < -0.1
    # below the precipitation threshold there is no damping
    out = diabatic_lagrangian(
        ds,
        snd,
        precipitation=_precip(ds, qr=1e-5, nr=1e3),
        processes=procs,
        filter_passes=0,
    )
    assert float(abs(out.dtheta_damping).max()) == 0.0


def test_damping_coefficient_regimes():
    from radarx.retrieve import _lagrangian_numpy as nk

    q = np.array(
        [
            float(
                {
                    **dl.DLA_DEFAULTS,
                    "z_sfc": 0,
                    "rho0": 1.2,
                    "flux_theta": 0,
                    "flux_qv": 0,
                    "switches": 0,
                }[k]
            )
            for k in nk.THERMO_KEYS
        ]
    )
    w = np.array([10.0, -5.0, 0.0, 0.0])
    qp = np.array([0.0, 0.0, 0.0, 5e-4])
    kd = nk.damping_rate(
        q, w, np.array([0, 0, 10.0, 10.0]), 0.0, 0.0, 0.0, qp, np.full(4, 3000.0)
    )
    e = math.exp(1.0)
    expect = [
        0.2 * 10 / ((5000 + 9.9 * 2000) * e),
        0.2 * 5 / ((300 + 4.9 * 100) * e),
        7e-5 * 10 / e,
        (0.5 * 7e-5 + 0.5 * 2e-4) * 10 / e,
    ]
    np.testing.assert_allclose(kd, expect, rtol=1e-12)


def test_surface_flux():
    # surface parcels in horizontal flow gain F exp(-b_F z) per second
    ds = _winds(u=10.0, dbz=-10.0)
    snd = _sounding()
    procs = {**NONE, "surface_flux": True}
    out = diabatic_lagrangian(
        ds,
        snd,
        precipitation="none",
        processes=procs,
        surface_flux=(1e-3, 0.0),
        filter_passes=0,
    )
    sfc = out.isel(z=0)
    steps = sfc.n_steps.values
    expect = 1e-3 * math.exp(-3.0 * 0.01) * 20.0 * steps
    np.testing.assert_allclose(sfc.dtheta_surface_flux, expect, rtol=1e-9)
    th10 = float(dl._base_state(snd, np.array([0.0, 10.0]), 0.0)[3].theta[1])
    np.testing.assert_allclose(sfc.theta - th10, expect, atol=1e-6)
    # above z_BL there is no flux
    assert float(abs(out.dtheta_surface_flux.sel(z=1500.0)).max()) == 0.0


@needs_kernel
def test_mesoscale_flux_and_engines_agree():
    def u(t, x, y, z):
        return 6.0 + 1e-4 * y

    def w(t, x, y, z):
        return np.where(z > 0, -1.5 * np.cos(4e-4 * x), 0.0)

    def dbz(t, x, y, z):
        return 55.0 - 4e-3 * np.hypot(x - 4e3, y - 4e3) - 2e-3 * z

    extra = {"ZDR": lambda t, x, y, z: 0.5 + (dbz(t, x, y, z) - 20) / 20.0}
    ds = _winds(u=u, v=1.0, w=w, dbz=dbz, extra=extra)
    snd = _sounding()
    zz, yy, xx = np.meshgrid(ds.z, ds.y, ds.x, indexing="ij")
    base = dl._base_state(snd, ds.z.values, 0.0)[3]
    meso = xr.Dataset(
        {
            "theta": (("z", "y", "x"), base.theta.values[:, None, None] + 1e-4 * xx),
            "qv": (("z", "y", "x"), base.qv.values[:, None, None] * (1 + 1e-5 * yy)),
        },
        coords={"z": ds.z, "y": ds.y, "x": ds.x},
    )
    kw = dict(
        mesoscale=meso, surface_flux=(1e-4, 1e-8), storm_motion=(2.0, 1.0), extend=600.0
    )
    a = diabatic_lagrangian(ds, snd, engine="compiled", n_threads=2, **kw)
    b = diabatic_lagrangian(ds, snd, engine="numpy", **kw)
    for k in ("theta", "qv", "qc", "qr", "qg", "flags", "origin_z"):
        np.testing.assert_allclose(a[k], b[k], rtol=1e-9, atol=1e-12, equal_nan=True)
    for k in a:
        if k.startswith("dtheta"):
            np.testing.assert_allclose(
                a[k], b[k], rtol=1e-8, atol=1e-10, equal_nan=True
            )
    assert float(abs(a.dtheta_surface_flux).max()) > 0
    assert float(abs(a.dtheta_rain_evaporation).max()) > 0.1
    assert float(a.qr.max()) > 1e-4 and float(a.qg.max()) > 1e-5
    # the mesoscale analysis sets the origin values
    c = diabatic_lagrangian(
        ds, snd, processes=NONE, precipitation="none", mesoscale=meso, filter_passes=0
    )
    ok = c.environment.values
    org = meso.theta.interp(
        x=xr.DataArray(c.origin_x.values[ok]),
        y=xr.DataArray(c.origin_y.values[ok]),
        z=xr.DataArray(c.origin_z.values[ok]),
    )
    np.testing.assert_allclose(
        c.theta.values[ok], org.values, atol=1e-4
    )  # float32 storage


@pytest.mark.parametrize(
    "test, budget",
    [
        ("RVAP", "dtheta_rain_evaporation"),
        ("GMLT", "dtheta_graupel_melting"),
        ("NOLD", "dtheta_damping"),
    ],
)
def test_sensitivity_switches(test, budget):
    ds = _winds(w=lambda t, x, y, z: np.where(z > 0, -2.0, 0.0), dbz=50.0)
    snd = _sounding()
    pr = _precip(ds, qr=2e-3, nr=500.0, qg=1e-3, ng=50.0)
    full = diabatic_lagrangian(ds, snd, precipitation=pr, filter_passes=0)
    red = diabatic_lagrangian(
        ds, snd, precipitation=pr, processes=test, filter_passes=0
    )
    assert float(abs(full[budget]).max()) > 0.05
    assert float(abs(red[budget]).max()) == 0.0
    assert float(abs(full.theta - red.theta).max()) > 0.05
    assert (
        test in red.attrs["processes"]
        or budget.split("_", 1)[1] not in red.attrs["processes"]
    )


def test_collection_and_surface_downdraft_switches():
    ds = _winds(w=lambda t, x, y, z: np.where(z > 0, 3.0, 0.0), dbz=50.0)
    snd = _sounding()
    pr = _precip(ds, qr=2e-3, nr=500.0, qg=1e-3, ng=50.0)
    full = diabatic_lagrangian(ds, snd, precipitation=pr, filter_passes=0)
    nocol = diabatic_lagrangian(
        ds, snd, precipitation=pr, processes="NOCOL", filter_passes=0
    )
    top = dict(z=6000.0)
    assert float(nocol.qc.sel(**top).mean()) > float(full.qc.sel(**top).mean()) + 1e-4
    ds = _winds(w=lambda t, x, y, z: np.where(z > 0, -3.0, 0.0), dbz=55.0)
    a = diabatic_lagrangian(
        ds,
        snd,
        precipitation="none",
        processes={**NONE, "surface_downdraft": True},
        filter_passes=0,
    )
    b = diabatic_lagrangian(
        ds, snd, precipitation="none", processes="WSFC", filter_passes=0
    )
    # surface parcels come from aloft sooner with the surface downdraft
    assert float(a.origin_time.isel(z=0).max()) == float(b.origin_time.isel(z=0).max())
    assert (
        float(a.origin_z.isel(z=0).mean()) > float(b.origin_z.isel(z=0).mean()) + 500.0
    )


# --------------------------------------------------------------------------
# microphysical rates (LFO83), checked against the printed equations
# --------------------------------------------------------------------------


def _props(t, p, rho):
    ka = (0.441635 + 0.0071 * t) * 1e-2
    psi = 2.11e-5 * (t / 273.15) ** 1.94 * (1e5 / p)
    nu = (0.379565 + 0.0049 * t) * 1e-5 / rho
    return ka, psi, nu


def _state(theta, p):
    t = theta * (p / 1e5) ** KAPPA
    rho = 1e5 * (p / 1e5) ** (1 - KAPPA) / (287.04 * theta)
    return t, rho


def _n0_lambda(q, n, rho, rhox):
    lam = (math.pi * rhox * n / (rho * q)) ** (1.0 / 3.0)
    return n * lam, lam


def test_rates_rain_by_hand():
    theta, p, qv, qc, qr, nr = 302.0, 90000.0, 0.008, 5e-4, 1.5e-3, 800.0
    t, rho = _state(theta, p)
    rho0 = 1.15
    n0, lam = _n0_lambda(qr, nr, rho, 1000.0)
    a, b = 841.99, 0.8
    # LFO83 eq. (51)
    racw = (
        math.pi
        * n0
        * a
        * qc
        * math.gamma(3 + b)
        / (4 * lam ** (3 + b))
        * (rho0 / rho) ** 0.5
    )
    r = microphysical_rates(theta, p, qv, qc, qr, nr, 0.0, 0.0, rho0=rho0)
    np.testing.assert_allclose(float(r.P_RACW), racw, rtol=1e-10)
    # eq. (52) in subsaturated air (no cloud)
    ka, psi, nu = _props(t, p, rho)
    rs = EPS * _es(t) / (p - _es(t))
    s = qv / rs
    vent = 0.78 / lam**2 + 0.31 * (nu / psi) ** (1 / 3) * math.gamma(
        (b + 5) / 2
    ) * a**0.5 * nu**-0.5 * (rho0 / rho) ** 0.25 * lam ** (-(b + 5) / 2)
    revp = (
        2
        * math.pi
        * (s - 1)
        * n0
        * vent
        / rho
        / (2.5e6**2 / (ka * 461.5 * t**2) + 1 / (rho * rs * psi))
    )
    r = microphysical_rates(theta, p, qv, 0.0, qr, nr, 0.0, 0.0, rho0=rho0)
    np.testing.assert_allclose(float(r.P_REVP), revp, rtol=1e-10)
    assert revp < 0 and float(r.P_RACW) == 0.0
    assert float(r.dtheta_dt) < 0 and float(r.dqv_dt) > 0
    # eq. (45) Bigg freezing below 0 degC
    theta2, p2 = 290.0, 60000.0
    t2, rho2 = _state(theta2, p2)
    n02, lam2 = _n0_lambda(qr, nr, rho2, 1000.0)
    gfr = (
        20
        * math.pi**2
        * 100.0
        * n02
        * (1000.0 / rho2)
        * (math.exp(0.66 * (273.15 - t2)) - 1)
        * lam2**-7
    )
    r = microphysical_rates(theta2, p2, 0.002, 0.0, qr, nr, 0.0, 0.0, rho0=rho0)
    assert t2 < 273.15
    np.testing.assert_allclose(float(r.P_GFR), gfr, rtol=1e-10)
    assert float(r.dtheta_dt) > 0


def test_rates_graupel_by_hand():
    rho0, rhog = 1.15, 690.0
    qg, ng, qr, nr = 2e-3, 100.0, 1e-3, 800.0
    # melting above 0 degC with cloud and rain accreted, eqs. (40), (42), (47)
    theta, p, qv, qc = 300.0, 85000.0, 0.009, 3e-4
    t, rho = _state(theta, p)
    n0g, lg_ = _n0_lambda(qg, ng, rho, rhog)
    n0r, lr = _n0_lambda(qr, nr, rho, 1000.0)
    ka, psi, nu = _props(t, p, rho)
    gf = (4 * 9.805 * rhog / (3 * 0.6 * rho)) ** 0.5
    gacw = math.pi * n0g * qc * math.gamma(3.5) / (4 * lg_**3.5) * gf
    ur = 841.99 * math.gamma(4.8) / (6 * lr**0.8) * (rho0 / rho) ** 0.5
    ug = math.gamma(4.5) / (6 * lg_**0.5) * gf
    gacr = (
        math.pi**2
        * n0g
        * n0r
        * abs(ug - ur)
        * (1000.0 / rho)
        * (5 / (lr**6 * lg_) + 2 / (lr**5 * lg_**2) + 0.5 / (lr**4 * lg_**3))
    )
    vent = (
        0.78 / lg_**2
        + 0.31
        * (nu / psi) ** (1 / 3)
        * math.gamma(2.75)
        * gf**0.5
        * nu**-0.5
        * lg_**-2.75
    )
    drs = EPS * 611.2 / (p - 611.2) - qv
    tc = t - 273.15
    gmlt = -2 * math.pi / (rho * 3.336e5) * (
        ka * tc - 2.5e6 * psi * rho * drs
    ) * n0g * vent - 4187.0 * tc / 3.336e5 * (gacw + gacr)
    r = microphysical_rates(
        theta, p, qv, qc, qr, nr, qg, ng, graupel_density=rhog, rho0=rho0
    )
    np.testing.assert_allclose(float(r.P_GACW), gacw, rtol=1e-10)
    np.testing.assert_allclose(float(r.P_GACR), gacr, rtol=1e-10)
    np.testing.assert_allclose(float(r.P_GMLT), gmlt, rtol=1e-10)
    assert gmlt < 0 and float(r.P_GSUB) == 0.0
    # sublimation below 0 degC outside cloud, eqs. (46), (31)
    theta, p, qv = 295.0, 55000.0, 0.0003
    t, rho = _state(theta, p)
    n0g, lg_ = _n0_lambda(qg, ng, rho, rhog)
    ka, psi, nu = _props(t, p, rho)
    gf = (4 * 9.805 * rhog / (3 * 0.6 * rho)) ** 0.5
    vent = (
        0.78 / lg_**2
        + 0.31
        * (nu / psi) ** (1 / 3)
        * math.gamma(2.75)
        * gf**0.5
        * nu**-0.5
        * lg_**-2.75
    )
    ei = 611.2 * math.exp(2.8336e6 / 461.5 * (1 / 273.15 - 1 / t))
    rsi = EPS * ei / (p - ei)
    a2 = 2.8336e6**2 / (ka * 461.5 * t**2)
    b2 = 1 / (rho * rsi * psi)
    gsub = 2 * math.pi * (qv / rsi - 1) / (rho * (a2 + b2)) * n0g * vent
    r = microphysical_rates(
        theta, p, qv, 0.0, 0.0, 0.0, qg, ng, graupel_density=rhog, rho0=rho0
    )
    assert t < 273.15
    np.testing.assert_allclose(float(r.P_GSUB), gsub, rtol=1e-10)
    assert gsub < 0 and float(r.P_GMLT) == 0.0


def test_rate_limits_and_switches():
    # evaporation can not overshoot saturation, collection not exceed the cloud
    r = microphysical_rates(300.0, 90000.0, 0.001, 1e-6, 5e-3, 1e4, 0.0, 0.0, dt=600.0)
    t, _ = _state(300.0, 90000.0)
    rs = EPS * _es(t) / (90000.0 - _es(t))
    assert float(r.dqv_dt) * 600.0 <= rs - 0.001
    assert float(r.dqc_dt) * 600.0 >= -1e-6 - 1e-18
    off = microphysical_rates(
        300.0, 90000.0, 0.001, 1e-4, 5e-3, 1e4, 0.0, 0.0, processes=NONE
    )
    assert all(float(abs(off[k])) == 0.0 for k in off)
    da = xr.DataArray(np.array([290.0, 300.0]), dims="z")
    out = microphysical_rates(
        da, 90000.0, 0.005, 0.0, 1e-3, 1e3, 0.0, 0.0, engine="numpy"
    )
    assert out.P_REVP.dims == ("z",)
    if lg.HAS_COMPILED_KERNEL:
        ref = microphysical_rates(
            da, 90000.0, 0.005, 0.0, 1e-3, 1e3, 0.0, 0.0, engine="compiled"
        )
        xr.testing.assert_allclose(out, ref, rtol=1e-10)
    with pytest.raises(ValueError, match="unknown processes"):
        microphysical_rates(300.0, 9e4, 0, 0, 0, 0, 0, 0, processes={"melting": False})
    with pytest.raises(ValueError, match="sensitivity test"):
        microphysical_rates(300.0, 9e4, 0, 0, 0, 0, 0, 0, processes="XYZ")


# --------------------------------------------------------------------------
# precipitation closures
# --------------------------------------------------------------------------


def _base(ds):
    return dl._base_state(_sounding(), ds.z.values, 0.0)[3]


def test_polarimetric_closure():
    ds = _winds(dbz=lambda t, x, y, z: 30.0 + 2e-3 * x, extra={"ZDR": 1.2})
    base = _base(ds)
    pr = polarimetric_precipitation(ds, base, melting_depth=0.0)
    hm = base.attrs["melting_level"]
    assert 4000.0 < hm < 4300.0
    below = pr.sel(z=slice(0, 3500))
    above = pr.sel(z=slice(4500, None))
    assert (below.qr > 0).all() and (below.qg == 0).all()
    assert (above.qr == 0).all() and (above.qg > 0).all()
    # graupel reflectivity of eq. (8) gives back Z_H
    g = above.isel(time=0, z=0)
    rho = float(base.rho.sel(z=4500.0))
    rhog = 690.0 + (630.0 - 690.0) * 0.9
    n0, lam = np.vectorize(_n0_lambda)(g.qg.values, g.ng.values, rho, rhog)
    zg = 0.224 * 7.295e19 * (np.pi * rhog / 1000.0) ** 2 * n0 / lam**7
    np.testing.assert_allclose(
        10 * np.log10(zg), ds.DBZ.isel(time=0).sel(z=4500.0), atol=1e-6
    )
    # rain from the radarx DSD retrieval
    from radarx.retrieve import dsd

    d = dsd(ds.isel(time=0).sel(z=1000.0), band="S")
    np.testing.assert_allclose(
        below.qr.isel(time=0).sel(z=1000.0),
        d.LWC * 1e-3 / float(base.rho.sel(z=1000.0)),
        rtol=1e-10,
    )
    # an HID graupel class below the melting level adds graupel; scaling N_g
    hid = xr.full_like(ds.DBZ, 4, dtype=np.int8)
    hid.attrs = {
        "flag_values": np.arange(1, 4 + 1),
        "flag_meanings": "dry_snow wet_snow ice_crystals graupel",
    }
    ds2 = ds.assign(HID=hid, DBZ=ds.DBZ + 15.0)
    pr2 = polarimetric_precipitation(ds2, base, graupel_scale=2.0)
    assert float(pr2.qg.sel(z=1000.0).max()) > 0
    pr3 = polarimetric_precipitation(ds2, base)
    np.testing.assert_allclose(
        pr2.ng.sel(z=5000.0), 2 * pr3.ng.sel(z=5000.0), rtol=1e-10
    )
    # without ZDR the rain comes from Z with a fixed intercept
    with pytest.warns(UserWarning, match="no ZDR"):
        pr4 = polarimetric_precipitation(ds.drop_vars("ZDR"), base)
    q = pr4.isel(time=0).sel(z=1000.0, x=0.0, y=0.0)
    zlin = 10**3.0
    lam = (1e18 * 720 * 8e5 / zlin) ** (1 / 7)
    np.testing.assert_allclose(
        float(q.qr),
        np.pi * 1000 * 8e5 / (float(base.rho.sel(z=1000.0)) * lam**4),
        rtol=1e-10,
    )


def test_polarimetric_closure_is_continuous_across_the_melting_level():
    c = np.arange(0.0, 2001.0, 1000.0)
    z = np.arange(0.0, 7001.0, 100.0)
    ds = xr.Dataset(
        {
            "DBZ": (("z", "y", "x"), np.full((z.size, c.size, c.size), 42.0)),
            "ZDR": (("z", "y", "x"), np.full((z.size, c.size, c.size), 1.3)),
        },
        coords={"z": z, "y": c, "x": c},
    )
    base = dl._base_state(_sounding(), z, 0.0)[3]
    hm = base.attrs["melting_level"]
    for depth, smooth in ((1000.0, True), (0.0, False)):
        pr = polarimetric_precipitation(ds, base, melting_depth=depth).isel(y=0, x=0)
        for k in ("qr", "qg"):
            col = pr[k].values
            jump = np.abs(np.diff(col)).max() / col.max()
            # 100-m levels through a 1-km layer (q_g grows as f^(4/7) at its base)
            assert (jump < 0.25) == smooth, (k, depth, jump)
        # rain below, graupel above the layer, both inside it
        assert float(pr.qg.sel(z=hm - 600.0, method="nearest")) == 0.0 or not smooth
        assert float(pr.qr.sel(z=hm + 600.0, method="nearest")) == 0.0
    mid = (
        polarimetric_precipitation(ds, base)
        .isel(y=0, x=0)
        .sel(z=round(hm, -2), method="nearest")
    )
    assert float(mid.qr) > 0 and float(mid.qg) > 0
    lay = polarimetric_precipitation(ds, base, melting_layer=(2000.0, 3000.0)).isel(
        y=0, x=0
    )
    assert float(lay.qr.sel(z=3000.0)) == 0.0 and float(lay.qg.sel(z=2000.0)) == 0.0


def _profiles():
    zs = np.array([0.0, 2500.0, 5000.0])
    return xr.Dataset(
        {
            "Z0r": ("z_star", [45.0, 47.0, 50.0]),
            "Z0g": ("z_star", [55.0, 50.0, 44.19]),
            "S0_qr": ("z_star", [8.0, 8.0, 8.0]),
            "S0_qg": ("z_star", [7.0, 7.0, 7.0]),
            "S0_n0g": ("z_star", [3e3, 3e3, 3e3]),
        },
        coords={"z_star": zs},
    )


def test_ziegler_closure():
    ds = _winds(w=lambda t, x, y, z: -2.0 + 3e-3 * x, dbz=50.0)
    base = _base(ds)
    with pytest.raises(ValueError, match="does not\\s+tabulate"):
        ziegler2013_precipitation(ds, base)
    with pytest.raises(ValueError, match="unknown constants"):
        ziegler2013_precipitation(ds, base, profiles=_profiles(), constants={"x": 1})
    pr = ziegler2013_precipitation(ds, base, profiles=_profiles())
    hm = base.attrs["melting_level"]
    zst = 1000.0 + 3900.0 - hm
    z0r = np.interp(zst, [0, 2500, 5000], [45, 47, 50])
    # eq. (9) where w < W_min
    cell = pr.isel(time=0).sel(z=1000.0, y=0.0, x=0.0)
    np.testing.assert_allclose(
        float(cell.qr), 0.5e-3 * math.exp((50.0 - z0r) / 8.0), rtol=1e-10
    )
    # eq. (13): no graupel below the melting level in strong updrafts (w > 20)
    strong = pr.isel(time=0).sel(z=1000.0, x=8000.0)
    assert float(strong.qg.max()) == 0.0
    # rain partitions the reflectivity left by graupel in updrafts (eq. 7)
    up = pr.isel(time=0).sel(z=1000.0, y=0.0, x=3000.0)
    rho = float(base.rho.sel(z=1000.0))
    n0g, lgm = _n0_lambda(float(up.qg), float(up.ng), rho, 690.0 + (-60) * 0.2)
    zeg = 0.224 * 7.295e19 * (np.pi * (690.0 - 12.0) / 1000) ** 2 * n0g / lgm**7
    lr = (1e18 * 720 * 8e5 / (1e5 - zeg)) ** (1 / 7)
    np.testing.assert_allclose(
        float(up.qr), np.pi * 1000 * 8e5 / (rho * lr**4), rtol=1e-8
    )
    out = diabatic_lagrangian(
        ds,
        _sounding(),
        precipitation="ziegler2013",
        precipitation_kwargs={"profiles": _profiles()},
    )
    assert float(out.qr.max()) > 0


# --------------------------------------------------------------------------
# gridding, output, errors
# --------------------------------------------------------------------------


def test_hole_fill_and_filter():
    a = np.ones((2, 5, 6))
    a[0, 2, 3] = np.nan
    a[0, 0, :2] = np.nan
    a[1] = np.nan
    f = dl._hole_fill(a)
    assert np.isfinite(f[0]).all() and np.isnan(f[1]).all()
    np.testing.assert_allclose(f[0], 1.0)
    chk = np.indices((1, 8, 8)).sum(axis=0) % 2 * 2.0 - 1.0
    sm = dl._nine_point(chk, 1)
    assert np.abs(sm[0, 1:-1, 1:-1]).max() < 1e-12
    np.testing.assert_allclose(dl._nine_point(np.full((1, 4, 4), 3.0), 2), 3.0)


def test_output_and_accessor():
    ds = _winds(w=-1.0, dbz=lambda t, x, y, z: 30.0 + 2e-3 * x, extra={"ZDR": 1.0})
    snd = _sounding()
    out = ds.radarx.diabatic_lagrangian(snd, levels=[0, 2])
    assert out.theta.dims == ("z", "y", "x") and out.sizes["z"] == 2
    for k in (
        "theta_v",
        "delta_theta_v",
        "qv",
        "qc",
        "qr",
        "nr",
        "qg",
        "ng",
        "rain_rate",
        "flags",
        "origin_time",
        "theta_v_base",
    ):
        assert k in out
    assert out.time.values == ds.time.values[-1]
    assert out.rain_rate.attrs["units"] == "mm h-1"
    assert (out.rain_rate.isel(z=0) > 0).all()
    assert 0 <= out.attrs["environment_fraction"] <= 1
    # hole filling fills grid points without an environmental trajectory
    wnd = ds.copy()
    wnd["u"] = wnd.u.where(~((wnd.x == 4000.0) & (wnd.y == 4000.0)))
    a = diabatic_lagrangian(wnd, snd, levels=[2], precipitation="none")
    assert np.isfinite(a.theta).all()
    assert not bool(a.environment.sel(x=4000.0, y=4000.0).item())
    b = diabatic_lagrangian(wnd, snd, levels=[2], precipitation="none", hole_fill=False)
    assert np.isnan(b.theta.sel(x=4000.0, y=4000.0)).all()
    # sounding with dewpoint instead of specific humidity
    s2 = snd.drop_vars("specific_humidity").assign(dewpoint=snd.temperature - 10.0)
    c = diabatic_lagrangian(ds, s2, levels=[1], precipitation="none")
    assert np.isfinite(c.qv).all()
    # mesoscale analysis from temperature and specific humidity
    base = _base(ds)
    meso = xr.Dataset(
        {
            "temperature": ("z", base.temperature.values),
            "specific_humidity": ("z", base.qv.values / (1 + base.qv.values)),
        },
        coords={"z": ds.z.values},
    ).expand_dims(y=ds.y.values, x=ds.x.values)
    d = diabatic_lagrangian(
        ds, snd, levels=[1], precipitation="none", mesoscale=meso, processes=NONE
    )
    e = diabatic_lagrangian(ds, snd, levels=[1], precipitation="none", processes=NONE)
    np.testing.assert_allclose(d.theta, e.theta, atol=5e-3)


def test_custom_closure_callable():
    ds = _winds(w=-1.0)
    calls = []

    def closure(radar, base):
        calls.append(base.attrs["melting_level"])
        return _precip(radar, qr=1e-3, nr=1e3).isel(time=0, drop=True)

    out = diabatic_lagrangian(ds, _sounding(), precipitation=closure, levels=[1])
    assert calls and np.allclose(out.qr, 1e-3)


def test_errors():
    ds = _winds()
    snd = _sounding()
    with pytest.raises(ValueError, match="height"):
        diabatic_lagrangian(ds, snd.rename(height="z"))
    with pytest.raises(ValueError, match="temperature"):
        diabatic_lagrangian(ds, snd.drop_vars("temperature"))
    with pytest.raises(ValueError, match="dewpoint"):
        diabatic_lagrangian(ds, snd.drop_vars("specific_humidity"))
    with pytest.raises(ValueError, match="two valid"):
        diabatic_lagrangian(ds, snd.isel(height=[0]))
    with pytest.raises(ValueError, match="precipitation must be"):
        diabatic_lagrangian(ds, snd, precipitation="radar")
    with pytest.raises(ValueError, match="must return"):
        diabatic_lagrangian(ds, snd, precipitation=_precip(ds).drop_vars("ng"))
    with pytest.raises(ValueError, match="mesoscale analysis needs 'theta'"):
        diabatic_lagrangian(
            ds,
            snd,
            precipitation="none",
            mesoscale=xr.Dataset(coords={"x": ds.x, "y": ds.y, "z": ds.z}),
        )
    with pytest.raises(ValueError, match="'qv'"):
        diabatic_lagrangian(
            ds,
            snd,
            precipitation="none",
            mesoscale=xr.Dataset({"theta": ds.u.isel(time=0)}),
        )
    with pytest.raises(ValueError, match="coordinate"):
        diabatic_lagrangian(
            ds, snd, precipitation="none", mesoscale=xr.Dataset({"theta": ("q", [1.0])})
        )
    with pytest.raises(ValueError, match="dbzh"):
        polarimetric_precipitation(ds.drop_vars("DBZ"), _base(ds))
    with pytest.raises(ValueError, match="not found"):
        polarimetric_precipitation(ds, _base(ds), dbzh="ZH")
    cold = snd.assign(temperature=snd.temperature - 60.0)
    with pytest.raises(ValueError, match="melting level"):
        ziegler2013_precipitation(
            ds, dl._base_state(cold, ds.z.values, 0.0)[3], profiles=_profiles()
        )


def test_numpy_branches_and_inputs():
    ds = _winds(w=-1.0, dbz=50.0, extra={"ZDR": 1.5})
    ds = ds.assign_coords(lat=(("y", "x"), np.zeros((ds.sizes["y"], ds.sizes["x"]))))
    snd = _sounding().drop_vars("u")
    snd["v"] = snd.v * np.nan
    pr = _precip(ds, qr=2e-3, nr=500.0, qg=1e-3, ng=50.0)
    kw = dict(precipitation=pr, levels=[2], filter_passes=0)
    for procs in ("NOCOL", {**NONE, "damping": True}):
        a = diabatic_lagrangian(ds, snd, engine="numpy", processes=procs, **kw)
        if lg.HAS_COMPILED_KERNEL:
            b = diabatic_lagrangian(ds, snd, engine="compiled", processes=procs, **kw)
            np.testing.assert_allclose(a.theta, b.theta, rtol=1e-10)
    assert "lat" in a.coords
    # no trajectory reaches the environment: all missing
    c = diabatic_lagrangian(ds, snd, engine="numpy", max_steps=5, hole_fill=False, **kw)
    assert np.isnan(c.theta).all() and not c.environment.any()
    # mesoscale analysis on another grid, with a time axis, mixing ratio and gaps
    base = dl._base_state(snd, ds.z.values, 0.0)[3]
    xx = np.arange(-1000.0, 9001.0, 2000.0)
    th = np.broadcast_to(
        base.theta.values[:, None, None], (ds.sizes["z"], xx.size, xx.size)
    ).copy()
    th[3, 0, 0] = np.nan
    meso = xr.Dataset(
        {
            "theta": (("z", "y", "x"), th),
            "mixing_ratio": (
                ("z", "y", "x"),
                np.broadcast_to(base.qv.values[:, None, None], th.shape),
            ),
        },
        coords={"z": ds.z.values, "y": xx, "x": xx},
    ).expand_dims(time=[ds.time.values[-1]])
    d = diabatic_lagrangian(
        ds, snd, mesoscale=meso, processes=NONE, precipitation="none", levels=[2]
    )
    e = diabatic_lagrangian(ds, snd, processes=NONE, precipitation="none", levels=[2])
    np.testing.assert_allclose(d.theta, e.theta, atol=5e-3)
    # explicit closure names and constants
    q = polarimetric_precipitation(ds, base, dbzh="DBZ", zdr="ZDR")
    assert float(q.qr.max()) > 0
    z = ziegler2013_precipitation(
        ds, base, profiles=_profiles(), constants={"n0r": 4e5}
    )
    assert float(z.qr.max()) > 0


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------


def test_real_nexrad_volume():
    xd = pytest.importorskip("xradar")
    from open_radar_data import DATASETS

    from radarx.grid import grid_radar

    try:
        file = DATASETS.fetch("KLBB20160601_150025_V06")
    except Exception as err:  # noqa: BLE001  # pragma: no cover - network
        pytest.skip(f"sample data not available: {err}")
    dtree = xd.io.open_nexradlevel2_datatree(file, sweep=[0, 1, 2, 3, 4, 5])
    dtree = dtree.xradar.georeference()
    for name in [n for n in dtree.children if n.startswith("sweep")]:
        ds = dtree[name].to_dataset(inherit=False)
        for var, lim in (("DBZH", -32.0), ("ZDR", -12.9)):
            if var in ds:
                ds[var] = ds[var].where(ds[var] > lim)
        dtree[name] = ds
    g = grid_radar(
        dtree,
        data_vars=["DBZH", "ZDR"],
        x_lim=(-60e3, 60e3),
        y_lim=(-60e3, 60e3),
        z_lim=(0, 6e3),
        x_step=2000,
        y_step=2000,
        z_step=500,
        pseudo_cappi=False,
    )
    g = g.drop_vars("time").astype("float64")
    rain = (g.DBZH > 35).fillna(False)
    # a steady storm moving with the mean wind, sinking 1 m/s in echo
    winds = g.assign(
        u=xr.full_like(g.DBZH, 5.0),
        v=xr.full_like(g.DBZH, 0.0),
        w=xr.where(rain & (g.z > 0), -1.0, 0.0),
    ).assign_coords(time=T0)
    snd = _sounding(q_low=0.006, q_high=0.003)
    k = int(np.argmax(rain.sum(("y", "x")).values))
    out = diabatic_lagrangian(
        winds, snd, storm_motion=(5.0, 0.0), extend=(2400.0, 0.0), levels=[k]
    )
    lev = out.isel(z=0)
    core = rain.isel(z=k).values
    assert core.sum() > 20
    assert (lev.qr.values[core] > 1e-4).mean() > 0.9
    rr = np.nanmedian(lev.rain_rate.values[core])
    assert 2.0 < rr < 200.0
    # evaporatively cooled air under the rain of a dry boundary layer
    assert float(lev.dtheta_rain_evaporation.where(rain.isel(z=k)).mean()) < -0.3
    assert out.attrs["environment_fraction"] > 0.9


def test_time_dependent_mesoscale_valid_fraction_and_termination():
    def dbz(t, x, y, z):
        return np.where(x > 5000.0, 45.0, -10.0)

    ds = _winds(u=4.0, w=0.0, dbz=dbz)
    ds["dd_valid"] = (ds.x < 3000.0).broadcast_like(ds.u)
    snd = _sounding()
    base = dl._base_state(snd, ds.z.values, 0.0)[3]
    shape = (2, ds.sizes["z"], ds.sizes["y"], ds.sizes["x"])
    # an environment that cools by 2 K between the first and the last wind time
    th = (
        base.theta.values[None, :, None, None]
        + np.array([2.0, 0.0])[:, None, None, None]
    )
    qv = np.broadcast_to(base.qv.values[None, :, None, None], shape)
    dims = ("time", "z", "y", "x")
    meso = xr.Dataset(
        {"theta": (dims, np.broadcast_to(th, shape)), "qv": (dims, qv)},
        coords={"time": ds.time.values[[0, -1]], "z": ds.z, "y": ds.y, "x": ds.x},
    )
    kw = dict(
        precipitation="none",
        processes=NONE,
        filter_passes=0,
        hole_fill=False,
        levels=[2],
    )
    out = {}
    engines = ("numpy", "compiled") if lg.HAS_COMPILED_KERNEL else ("numpy",)
    for eng in engines:
        out[eng] = diabatic_lagrangian(
            ds,
            snd,
            mesoscale=meso,
            valid="dd_valid",
            min_valid_fraction=0.5,
            engine=eng,
            **kw,
        )
    a = out["numpy"]
    if "compiled" in out:
        for k in ("theta", "qv", "valid_fraction", "flags"):
            np.testing.assert_allclose(
                a[k], out["compiled"][k], rtol=1e-9, equal_nan=True
            )
    # parcels start from the environment at their origin time (linear in time)
    ok = a.environment.values
    span = float(ds.time[-1] - ds.time[0]) / 1e9
    th0 = base.theta.sel(z=1000.0).item()
    expect = th0 + 2.0 * (-a.origin_time.values[ok] / span)
    np.testing.assert_allclose(a.theta.values[ok], expect, atol=1e-3)
    # endpoints whose trajectories ran mostly outside valid winds are masked
    low = a.valid_fraction.values < 0.5
    assert low.any() and (~low).any()
    assert np.isnan(a.theta.values[low]).all() and (a.flags.values[low] & 64).all()
    assert not a.environment.values[low].any()
    # the alternative termination keeps parcels in rain longer
    b = diabatic_lagrangian(
        ds,
        snd,
        termination="precipitation",
        parameters={"cold_pool_depth": 500.0, "min_steps": 0},
        **kw,
    )
    assert float(b.n_steps.mean()) != float(a.n_steps.mean())
    assert b.attrs["termination"] == "precipitation"
    with pytest.raises(ValueError, match="termination"):
        diabatic_lagrangian(ds, snd, termination=False, **kw)
    with pytest.raises(ValueError, match="distinct"):
        diabatic_lagrangian(ds, snd, mesoscale=xr.concat([meso, meso], "time"), **kw)
