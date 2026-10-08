#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests of the ice-water saturation adjustment (Tao et al. 1989) and the in
situ initialization (Ziegler et al. 2007) of the diabatic Lagrangian
analysis."""

import importlib
import math

import numpy as np
import pytest
import xarray as xr

from radarx.retrieve import diabatic_lagrangian
from radarx.retrieve import lagrangian as lg

dl = importlib.import_module("radarx.retrieve.diabatic_lagrangian")
nk = importlib.import_module("radarx.retrieve._lagrangian_numpy")

needs_kernel = pytest.mark.skipif(
    not lg.HAS_COMPILED_KERNEL, reason="compiled trajectory kernel not built"
)

DIMS = ("time", "z", "y", "x")
T0 = np.datetime64("2022-03-30T23:00:00", "ns")
KAPPA = 0.2854
CP = 287.04 / KAPPA
LV, LS = 2.5e6, 2.8336e6
G = 9.80665
ENGINES = ["numpy"] + (["compiled"] if lg.HAS_COMPILED_KERNEL else [])


def _winds(u=0.0, w=0.0, dbz=10.0, ztop=6000.0, nt=3, step=1800.0, extra=None):
    c = np.arange(0.0, 8001.0, 1000.0)
    z = np.arange(0.0, ztop + 1.0, 500.0)
    t = np.arange(nt) * step
    tt, zz, yy, xx = np.meshgrid(t, z, c, c, indexing="ij")
    fields = {"u": u, "v": 0.0, "w": w, "DBZ": dbz, **(extra or {})}
    data = {}
    for name, f in fields.items():
        val = f(tt, xx, yy, zz) if callable(f) else np.full(tt.shape, float(f))
        data[name] = (DIMS, val)
    times = T0 + (t * 1e9).astype("timedelta64[ns]")
    return xr.Dataset(data, coords={"time": times, "z": z, "y": c, "x": c})


def _sounding(top=16000.0, q_low=0.014, dry=False):
    h = np.arange(0.0, top + 1.0, 50.0)
    t = np.maximum(300.0 - 0.0065 * h, 210.0)
    p = 1e5 * (t / 300.0) ** (G / (287.04 * 0.0065))
    q = np.where(h < 1500, q_low, 0.004 * np.exp(-h / 3000.0))
    if dry:
        q = np.full(h.shape, 1e-4)
    return xr.Dataset(
        {
            "pressure": ("height", p),
            "temperature": ("height", t),
            "specific_humidity": ("height", q),
        },
        coords={"height": h},
    )


def _tao_step(theta, p, qv, qc, qi, t00=233.15):
    """One step of Tao et al. (1989) written out from the printed equations."""
    pi = (p / 1e5) ** KAPPA
    t = pi * theta
    t0 = 273.15
    cnd = min(max((t - t00) / (t0 - t00), 0.0), 1.0)  # (2b)
    dep = 1.0 - cnd  # (2c)
    b = 3.8 / (p / 100.0)
    a1, a2 = 17.2693882, 21.8745584
    qws = b * math.exp(a1 * (pi * theta - 273.16) / (pi * theta - 35.86))  # (3a)
    qis = b * math.exp(a2 * (pi * theta - 273.16) / (pi * theta - 7.66))  # (3b)
    wc, wi = (qc, qi) if qc + qi > 0 else (cnd, dep)
    r1 = qv - (wc * qws + wi * qis) / (wc + wi)  # (6a)
    A1 = 237.3 * a1 * pi / (pi * theta - 35.86) ** 2  # (6c)
    A2 = 265.5 * a2 * pi / (pi * theta - 7.66) ** 2  # (6d)
    r2 = (A1 * wc * qws + A2 * wi * qis) / (wc + wi)  # (6b)
    A3 = (LV * cnd + LS * dep) / (CP * pi)  # (6e)
    theta1 = theta + r1 * A3 / (1.0 + r2 * A3)  # (7a)
    qv1 = qv - r1 / (1.0 + r2 * A3)  # (7b)
    dq = qv - qv1
    return theta1, qv1, qc + dq * cnd, qi + dq * dep, qws, qis


def _adjust(engine, *args, ice=True, t00=233.15):
    a = [np.atleast_1d(np.asarray(x, float)) for x in args]
    if engine == "compiled":
        st, ht = lg._lagrangian.adjust(*a, t00, ice)
        return np.asarray(st), np.asarray(ht)
    return nk.adjust_states(*a, t00, ice)


# --------------------------------------------------------------------------
# ice-water saturation adjustment
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "state",
    [
        # theta, p, qv, qc, qi: supersaturated, mixed phase, both condensates
        (310.0, 5.0e4, 1.8e-3, 1.0e-3, 0.5e-3),
        # first condensation in the mixed-phase range (weights CND, DEP)
        (305.0, 5.5e4, 2.1e-3, 0.0, 0.0),
        # warm: liquid only
        (300.0, 9.0e4, 0.015, 0.0, 0.0),
        # cold, cloud ice only
        (330.0, 3.0e4, 0.32e-3, 0.0, 0.2e-3),
    ],
)
def test_tao_adjustment_by_hand(engine, state):
    th, p, qv, qc, qi = state
    st, ht = _adjust(engine, th, p, qv, qc, qi)
    th1, qv1, qc1, qi1, qws, qis = _tao_step(*state)
    np.testing.assert_allclose(st[0], [th1, qv1, qc1, qi1], rtol=1e-12, atol=1e-15)
    # moist-adiabatic constraint (4a) and total water
    pi = (p / 1e5) ** KAPPA
    dqc, dqi = st[0, 2] - qc, st[0, 3] - qi
    np.testing.assert_allclose(
        st[0, 0] - th, (LV * dqc + LS * dqi) / (CP * pi), rtol=1e-12
    )
    np.testing.assert_allclose(st[0, 1:].sum(), qv + qc + qi, rtol=1e-13)
    assert ht[0, 0] == pytest.approx(st[0, 0] - th, rel=1e-12)
    assert ht[0, 1] == 0.0
    # after the step q_v is the weighted saturation (4b) at the new
    # temperature to first order in d theta (5)
    t1 = st[0, 0] * pi
    b = 3.8 / (p / 100.0)
    qws1 = b * math.exp(17.2693882 * (t1 - 273.16) / (t1 - 35.86))
    qis1 = b * math.exp(21.8745584 * (t1 - 273.16) / (t1 - 7.66))
    c, i = (qc, qi) if qc + qi > 0 else (st[0, 2], st[0, 3])
    qvs1 = (c * qws1 + i * qis1) / (c + i)
    assert abs(st[0, 1] - qvs1) < 0.01 * qvs1


@pytest.mark.parametrize("engine", ENGINES)
def test_tao_adjustment_partition_and_limits(engine):
    p = 6.0e4
    pi = (p / 1e5) ** KAPPA
    t00 = 233.15
    # first condensation at -20 degC: new condensate split CND : DEP
    th = 253.15 / pi
    st, _ = _adjust(engine, th, p, 3e-3, 0.0, 0.0)
    t = 253.15
    cnd = (t - t00) / (273.15 - t00)
    dqc, dqi = st[0, 2], st[0, 3]
    assert dqc > 0 and dqi > 0
    np.testing.assert_allclose(dqc / (dqc + dqi), cnd, rtol=1e-12)
    # subsaturated, no condensate: nothing happens
    st, ht = _adjust(engine, th, p, 1e-4, 0.0, 0.0)
    np.testing.assert_array_equal(st[0], [th, 1e-4, 0.0, 0.0])
    assert ht[0, 0] == 0.0
    # subsaturated with ice only at -10 degC: sublimation limited to the ice
    th = 263.15 / pi
    st, _ = _adjust(engine, th, p, 1e-4, 0.0, 1e-5)
    assert st[0, 2] == 0.0 and st[0, 3] == pytest.approx(0.0, abs=1e-18)
    np.testing.assert_allclose(st[0, 1], 1e-4 + 1e-5, rtol=1e-12)
    # subsaturated with much ice: q_i loses the DEP share of the deficit only
    st, _ = _adjust(engine, th, p, 1e-4, 0.0, 1e-3)
    th1, qv1, _, _, _, _ = _tao_step(th, p, 1e-4, 0.0, 1e-3)
    dep = (273.15 - 263.15) / (273.15 - t00)
    np.testing.assert_allclose(st[0, 3] - 1e-3, (qv1 - 1e-4) * -dep, rtol=1e-10)
    # without ice: water saturation (Bolton), cloud ice untouched
    st, ht = _adjust(engine, th, p, 3e-3, 0.0, 1e-4, ice=False)
    assert st[0, 3] == 1e-4 and ht[0, 1] == 0.0 and st[0, 2] > 0
    t1 = st[0, 0] * pi
    es = 611.2 * math.exp(17.67 * (t1 - 273.15) / (t1 - 273.15 + 243.5))
    np.testing.assert_allclose(st[0, 1], 287.04 / 461.5 * es / (p - es), rtol=1e-6)
    # T00 = -25 degC: ice only below
    th = 243.15 / pi
    st, _ = _adjust(engine, th, p, 2e-3, 0.0, 0.0, t00=248.15)
    assert st[0, 2] == 0.0 and st[0, 3] > 0


@pytest.mark.parametrize("engine", ENGINES)
def test_homogeneous_freezing_and_melting_by_hand(engine):
    p = 2.5e4
    pi = (p / 1e5) ** KAPPA
    # cloud water at -45 degC freezes (P_IHOM), heating L_f q_c / (c_p Pi)
    th = 228.15 / pi
    qc = 1.5e-3
    st, ht = _adjust(engine, th, p, 0.0, qc, 0.0)
    dth_f = (LS - LV) * qc / (CP * pi)
    np.testing.assert_allclose(ht[0, 1], dth_f, rtol=1e-12)
    assert st[0, 2] == 0.0
    # the vapour then deposits or the ice sublimates to ice saturation (DEP = 1)
    th1, qv1, qc1, qi1, _, _ = _tao_step(th + dth_f, p, 0.0, 0.0, qc)
    np.testing.assert_allclose(st[0], [th1, qv1, qc1, qi1], rtol=1e-11, atol=1e-16)
    # cloud ice above 0 degC melts (P_IMLT)
    p = 9.0e4
    pi = (p / 1e5) ** KAPPA
    th = 275.15 / pi
    st, ht = _adjust(engine, th, p, 6e-3, 0.0, 1e-3)
    np.testing.assert_allclose(ht[0, 1], -(LS - LV) * 1e-3 / (CP * pi), rtol=1e-12)
    assert st[0, 3] == 0.0 and st[0, 2] > 0.0


@needs_kernel
def test_adjustment_engines_agree():
    rng = np.random.default_rng(7)
    n = 5000
    th = rng.uniform(280, 345, n)
    p = rng.uniform(2e4, 1e5, n)
    qv = rng.uniform(0, 0.02, n)
    qc = rng.uniform(0, 3e-3, n) * (rng.random(n) < 0.6)
    qi = rng.uniform(0, 3e-3, n) * (rng.random(n) < 0.6)
    for ice in (False, True):
        a, b = _adjust("compiled", th, p, qv, qc, qi, ice=ice)
        c, d = _adjust("numpy", th, p, qv, qc, qi, ice=ice)
        np.testing.assert_allclose(a, c, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(b, d, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("engine", ["auto", "numpy"])
def test_saturated_ascent_through_the_freezing_level(engine):
    """Updraft from the moist boundary layer to -50 degC with ice=True."""
    ds = _winds(w=lambda t, x, y, z: np.where(z > 0, 3.0, 0.0), ztop=12000.0, nt=4)
    snd = _sounding()
    kw = dict(precipitation="none", filter_passes=0, hole_fill=False, max_steps=400)
    out = diabatic_lagrangian(ds, snd, ice=True, engine=engine, **kw)
    water = diabatic_lagrangian(ds, snd, engine=engine, **kw)
    assert "qi" in out and "qi" not in water
    assert out.attrs["ice"] == 1 and "Tao" in out.attrs["ice_adjustment"]
    col, wcol = out.isel(x=4, y=4), water.isel(x=4, y=4)
    t, qv, qc, qi = (col[k].values for k in ("temperature", "qv", "qc", "qi"))
    r0 = 0.014 / (1 - 0.014)
    cloud = (qc + qi) > 1e-4
    assert cloud.sum() >= 20 and not cloud[0]
    assert bool(out.environment.all()) and bool(water.environment.all())
    # total water conserved
    np.testing.assert_allclose((qv + qc + qi)[cloud], r0, rtol=1e-9)
    # liquid only above 0 degC, ice only below -40 degC, both in between
    assert np.all(qi[t > 273.15] == 0.0)
    assert np.all(qc[t < 233.15] == 0.0) and np.all(qi[t < 233.15] > 1e-3)
    mixed = cloud & (t < 268.15) & (t > 238.15)
    assert np.all(qc[mixed] > 0) and np.all(qi[mixed] > 0)
    # the theta change is the latent heating of eq. (4a) along the path
    zo = col.origin_z.values
    th0 = np.interp(
        zo, snd.height.values, (snd.temperature * (1e5 / snd.pressure) ** KAPPA).values
    )
    np.testing.assert_allclose(
        col.dtheta_condensation.values + col.dtheta_freezing.values,
        col.theta.values - th0,
        atol=1e-4,
    )
    # conserved: ice-liquid water potential temperature in the form of eq.
    # (4a) along the path, theta_il = theta - sum (L_v dq_c + L_s dq_i) /
    # (c_p Pi), minus the theta of the parcel's origin. In this steady,
    # uniform updraft the grid levels hold parcels of the same total water at
    # successive stages, so the sum is taken between levels with the
    # trapezoidal rule in 1 / Pi (Pi of the base state); it holds across the
    # freezing level and the homogeneous freezing at -40 degC
    pi = (col.pressure_base.values / 1e5) ** KAPPA
    dh = LV * np.diff(qc) + LS * np.diff(qi)
    inv = 0.5 * (1.0 / pi[1:] + 1.0 / pi[:-1])
    th_il = col.theta.values - np.concatenate(([0.0], np.cumsum(dh * inv / CP))) - th0
    assert np.abs(th_il).max() < 0.06
    # the ice run is warmer above the freezing level (fusion, deposition)
    wt = wcol.temperature.values
    assert np.all(t[mixed] >= wt[mixed] - 0.05)
    assert np.all(t[t < 233.15] > wt[t < 233.15] + 3.0)


def test_ice_in_the_dla_engines_agree_with_precipitation():
    if not lg.HAS_COMPILED_KERNEL:
        pytest.skip("compiled trajectory kernel not built")
    ds = _winds(
        u=lambda t, x, y, z: 3.0 + 0.0 * x,
        w=lambda t, x, y, z: np.where(z > 0, 2.0 * np.sin(np.pi * x / 8000.0), 0.0),
        dbz=lambda t, x, y, z: 30.0 + 10.0 * np.cos(np.pi * x / 8000.0),
        ztop=10000.0,
    )
    shape = tuple(ds.sizes[d] for d in DIMS)
    zz = np.broadcast_to(ds.z.values[None, :, None, None], shape)
    precip = xr.Dataset(
        {
            "qr": (DIMS, np.where(zz < 4000, 1e-3, 0.0)),
            "nr": (DIMS, np.where(zz < 4000, 2e3, 0.0)),
            "qg": (DIMS, np.where(zz > 2000, 1e-3, 0.0)),
            "ng": (DIMS, np.where(zz > 2000, 1e2, 0.0)),
        },
        coords={d: ds[d] for d in DIMS},
    )
    snd = _sounding()
    kw = dict(precipitation=precip, ice=True, filter_passes=0, hole_fill=False)
    a = diabatic_lagrangian(ds, snd, engine="compiled", **kw)
    b = diabatic_lagrangian(ds, snd, engine="numpy", **kw)
    for k in ("theta", "qv", "qc", "qi", "dtheta_freezing", "dtheta_condensation"):
        np.testing.assert_allclose(a[k].values, b[k].values, rtol=1e-9, atol=1e-12)
    assert float(a.qi.max()) > 0


def test_ice_with_mesoscale_surface_flux_by_hand():
    """NumPy path of the surface flux from the mesoscale gradient with ice:
    dtheta/dt = exp(-b_F z) u dtheta/dx along a dry surface parcel."""
    ds = _winds(u=6.0, dbz=45.0, nt=2, step=1800.0)
    snd = _sounding(dry=True)
    base = dl._base_state(snd, ds.z.values, 0.0)[3]
    zz, yy, xx = np.meshgrid(ds.z, ds.y, ds.x, indexing="ij")
    meso = xr.Dataset(
        {
            "theta": (("z", "y", "x"), base.theta.values[:, None, None] + 1e-4 * xx),
            "qv": (("z", "y", "x"), base.qv.values[:, None, None] + 0.0 * xx),
        },
        coords={"z": ds.z, "y": ds.y, "x": ds.x},
    )
    out = diabatic_lagrangian(
        ds,
        snd,
        precipitation="none",
        processes={**DRY, "surface_flux": True},
        mesoscale=meso,
        ice=True,
        engine="numpy",
        filter_passes=0,
        hole_fill=False,
        levels=[0],
    ).isel(z=0)
    ok = out.environment.values & (out.n_steps.values > 0)
    expect = out.n_steps.values * 20.0 * math.exp(-3.0 * 10.0 / 1000.0) * 6.0 * 1e-4
    np.testing.assert_allclose(
        out.dtheta_surface_flux.values[ok], expect[ok], rtol=1e-6
    )
    assert float(abs(out.qi).max()) == 0.0


def test_ice_errors():
    ds = _winds()
    with pytest.raises(ValueError, match="t00"):
        diabatic_lagrangian(
            ds, _sounding(), precipitation="none", ice=True, parameters={"t00": 280.0}
        )


# --------------------------------------------------------------------------
# in situ initialization
# --------------------------------------------------------------------------


def _stations(x, y, t, theta, qv, z=0.0):
    """Station observations on (station, time); t in s from the analysis."""
    x, y = np.atleast_1d(x).astype(float), np.atleast_1d(y).astype(float)
    t = np.atleast_1d(t).astype(float)
    th = np.broadcast_to(np.asarray(theta, float), (x.size, t.size))
    q = np.broadcast_to(np.asarray(qv, float), (x.size, t.size))
    times = T0 + np.timedelta64(3600, "s") + (t * 1e9).astype("timedelta64[ns]")
    return xr.Dataset(
        {
            "x": ("station", x),
            "y": ("station", y),
            "z": ("station", np.full(x.size, float(z))),
            "theta": (("station", "time"), th),
            "qv": (("station", "time"), q),
        },
        coords={"time": times, "station": np.arange(x.size)},
    )


DRY = {k: False for k in dl.PROCESSES}


@pytest.mark.parametrize("engine", ["auto", "numpy"])
def test_insitu_uniform_flow(engine):
    """Uniform u = 10 m/s: the surface parcel at the analysis time at (5, 4) km
    was at (2, 4) km 300 s earlier, where the station observed it."""
    ds = _winds(u=10.0, dbz=45.0, nt=2, step=3600.0)
    snd = _sounding(dry=True)
    obs = _stations(2000.0, 4000.0, [-300.0], 290.0, 2e-3)
    out = diabatic_lagrangian(
        ds,
        snd,
        precipitation="none",
        processes=DRY,
        observations=obs,
        observation_options={"radius": 400.0},
        engine=engine,
        filter_passes=0,
        hole_fill=False,
    )
    hit = (out.flags.values & 256) != 0
    assert hit.sum() == 1 and hit[0, 4, 5]
    s = out.isel(z=0, y=4, x=5)
    assert float(s.theta) == pytest.approx(290.0, abs=1e-9)
    assert float(s.qv) == pytest.approx(2e-3, abs=1e-12)
    assert float(s.insitu_time) == pytest.approx(-300.0)
    tau_i, tau_l = dl.INSITU_DEFAULTS["tau_i"], dl.INSITU_DEFAULTS["tau_l"]
    assert float(s.insitu_weight) == pytest.approx(
        math.exp(-(300.0**2) / tau_i - 300.0**2 / tau_l), rel=1e-6
    )
    # every other point keeps its environmental initial state and NaN weight
    other = out.isel(z=0).where(~xr.DataArray(hit[0], dims=("y", "x")))
    assert np.isnan(other.insitu_weight).all()
    np.testing.assert_allclose(
        other.theta.values[np.isfinite(other.theta.values)],
        float(out.theta_base.isel(z=0)),
        atol=0.1,
    )
    assert out.attrs["insitu_observations"] == 1
    assert out.attrs["insitu_fraction"] == pytest.approx(1 / hit.size)
    flags = out.flags
    assert 256 in flags.attrs["flag_masks"]
    assert flags.attrs["flag_meanings"].endswith("initialized_from_in_situ_observation")


@pytest.mark.parametrize("engine", ["auto", "numpy"])
def test_insitu_weighting_window_and_tolerances(engine):
    ds = _winds(u=10.0, dbz=45.0, nt=2, step=3600.0)
    snd = _sounding(dry=True)
    # station A at (2, 4) km at -300 s on the path of the surface parcel
    # ending at (5, 4) km; station B at (2.3, 4.25) km at -270 s, 250 m from
    # that parcel (at 2.3 km at -270 s); station C
    # outside the window, station D after the analysis time, station E at
    # 500 m height (matches level 1 only)
    obs = xr.Dataset(
        {
            "x": ("obs", [2000.0, 2300.0, 2000.0, 5000.0, 2000.0]),
            "y": ("obs", [4000.0, 4250.0, 4000.0, 4000.0, 4000.0]),
            "z": ("obs", [0.0, 0.0, 0.0, 0.0, 500.0]),
            "time": ("obs", [-300.0, -270.0, -900.0, 60.0, -300.0]),
            "theta": ("obs", [290.0, 292.0, 280.0, 250.0, 305.0]),
            "qv": ("obs", [2e-3, 3e-3, 1e-3, 1e-3, 1e-3]),
        }
    )
    opts = {"radius": 400.0, "window": 600.0}
    kw = dict(precipitation="none", processes=DRY, filter_passes=0, hole_fill=False)
    out = diabatic_lagrangian(
        ds, snd, observations=obs, observation_options=opts, engine=engine, **kw
    )
    hit = (out.flags.values & 256) != 0
    assert hit.sum() == 2 and hit[0, 4, 5] and hit[1, 4, 5]
    s = out.isel(z=0, y=4, x=5)
    o = dl.INSITU_DEFAULTS
    w = [
        math.exp(-(r**2) / o["kappa_s"] - t**2 / o["tau_i"] - t**2 / o["tau_l"])
        for r, t in ((0.0, -300.0), (250.0, -270.0))
    ]
    np.testing.assert_allclose(float(s.insitu_weight), sum(w), rtol=1e-9)
    np.testing.assert_allclose(
        float(s.theta), (w[0] * 290 + w[1] * 292) / sum(w), rtol=1e-12
    )
    # the start is at the observation with the largest weight
    assert float(s.insitu_time) == pytest.approx(-300.0 if w[0] > w[1] else -280.0)
    assert float(out.theta.isel(z=1, y=4, x=5)) == pytest.approx(305.0)
    # a tighter z tolerance leaves the surface parcel (started at the offset
    # height, 10 m) unmatched; a smaller radius drops station B
    out = diabatic_lagrangian(
        ds,
        snd,
        observations=obs,
        observation_options={**opts, "z_tolerance": 5.0},
        engine=engine,
        **kw,
    )
    assert not (out.flags.values[0] & 256).any()
    out = diabatic_lagrangian(
        ds,
        snd,
        observations=obs,
        observation_options={**opts, "radius": 200.0},
        engine=engine,
        **kw,
    )
    s = out.isel(z=0, y=4, x=5)
    assert float(s.theta) == pytest.approx(290.0)
    assert float(s.insitu_time) == pytest.approx(-300.0)


@pytest.mark.parametrize("engine", ["auto", "numpy"])
def test_insitu_initialises_trajectories_inside_the_storm(engine):
    """Trajectories that stop inside the storm (max_steps) are analysed only
    where an observation initialises them."""
    ds = _winds(u=10.0, dbz=45.0, nt=2, step=3600.0)
    snd = _sounding(dry=True)
    obs = _stations([1000.0, 3000.0], [2000.0, 6000.0], [-100.0], 295.0, 4e-3)
    kw = dict(precipitation="none", processes=DRY, filter_passes=0, hole_fill=False)
    out = diabatic_lagrangian(
        ds,
        snd,
        observations=obs,
        observation_options={"radius": 300.0},
        max_steps=10,
        boundary=["east"],
        engine=engine,
        **kw,
    )
    sfc = out.isel(z=0)
    assert not bool(sfc.environment.any())
    ok = np.isfinite(sfc.theta.values)
    assert ok.sum() == 2 and ok[2, 2] and ok[6, 4]
    np.testing.assert_allclose(sfc.theta.values[ok], 295.0)
    assert ((sfc.flags.values[ok] & 16) != 0).all()
    # an observation older than the (10-step) trajectories is not used
    old = _stations([1000.0], [2000.0], [-1000.0], 295.0, 4e-3)
    out = diabatic_lagrangian(
        ds,
        snd,
        observations=old,
        observation_options={"radius": 300.0, "window": 1500.0},
        max_steps=10,
        boundary=["east"],
        engine=engine,
        **kw,
    )
    assert not ((out.flags.values & 256) != 0).any()


@needs_kernel
def test_insitu_engines_agree_with_processes():
    ds = _winds(
        u=lambda t, x, y, z: 8.0 + 0.0 * x,
        w=lambda t, x, y, z: np.where(z > 0, -1.0 * np.cos(np.pi * x / 8000.0), 0.0),
        dbz=lambda t, x, y, z: 35.0 + 5.0 * np.sin(np.pi * y / 8000.0),
    )
    shape = tuple(ds.sizes[d] for d in DIMS)
    precip = xr.Dataset(
        {
            "qr": (DIMS, np.full(shape, 1e-3)),
            "nr": (DIMS, np.full(shape, 2e3)),
            "qg": (DIMS, np.full(shape, 2e-4)),
            "ng": (DIMS, np.full(shape, 50.0)),
        },
        coords={d: ds[d] for d in DIMS},
    )
    rng = np.random.default_rng(3)
    obs = _stations(
        rng.uniform(0, 8000, 40),
        rng.uniform(0, 8000, 40),
        np.arange(-600.0, 1.0, 60.0),
        rng.uniform(292, 298, (40, 11)),
        rng.uniform(8e-3, 12e-3, (40, 11)),
    )
    kw = dict(
        precipitation=precip,
        observations=obs,
        observation_options={"radius": 600.0, "z_tolerance": 600.0},
        ice=True,
        filter_passes=0,
        hole_fill=False,
    )
    a = diabatic_lagrangian(ds, _sounding(), engine="compiled", **kw)
    b = diabatic_lagrangian(ds, _sounding(), engine="numpy", **kw)
    np.testing.assert_array_equal(a.flags.values, b.flags.values)
    assert ((a.flags.values & 256) != 0).sum() > 20
    for k in ("theta", "qv", "qc", "qi", "insitu_weight", "insitu_time"):
        np.testing.assert_allclose(
            a[k].values, b[k].values, rtol=1e-9, atol=1e-12, equal_nan=True
        )


def test_insitu_input_forms_and_errors():
    pyproj = pytest.importorskip("pyproj")
    ds = _winds(u=10.0, dbz=45.0, nt=2, step=3600.0)
    ds.attrs.update(origin_latitude=33.9, origin_longitude=-88.3)
    snd = _sounding(dry=True)
    kw = dict(
        precipitation="none",
        processes=DRY,
        filter_passes=0,
        hole_fill=False,
        engine="numpy",
        observation_options={"radius": 400.0},
    )
    ref = diabatic_lagrangian(
        ds, snd, observations=_stations(2000.0, 4000.0, [-300.0], 290.0, 2e-3), **kw
    )
    # latitude/longitude, temperature and pressure, dewpoint, datetime times
    proj = pyproj.Proj(proj="aeqd", lat_0=33.9, lon_0=-88.3, datum="WGS84")
    lon, lat = proj(2000.0, 4000.0, inverse=True)
    p = float(ref.pressure_base.isel(z=0))
    t = 290.0 * (p / 1e5) ** KAPPA
    e = 2e-3 * p / (287.04 / 461.5 + 2e-3)
    tc = np.log(e / 611.2)
    td = 273.15 + 243.5 * tc / (17.67 - tc)
    obs = xr.Dataset(
        {
            "latitude": ("station", [lat]),
            "longitude": ("station", [lon]),
            "temperature": ("station", [t]),
            "dewpoint": ("station", [td]),
            "pressure": ("station", [p]),
        },
        coords={"time": ("station", [T0 + np.timedelta64(3300, "s")])},
    )
    out = diabatic_lagrangian(ds, snd, observations=obs, **kw)
    np.testing.assert_allclose(out.theta.values, ref.theta.values, equal_nan=True)
    np.testing.assert_allclose(out.qv.values, ref.qv.values, rtol=1e-9, equal_nan=True)
    for name, alt in (("qv", "mixing_ratio"), ("qv", "specific_humidity")):
        o2 = _stations(2000.0, 4000.0, [-300.0], 290.0, 2e-3).rename({name: alt})
        if alt == "specific_humidity":
            o2[alt] = o2[alt] / (1 + o2[alt])
        o2 = o2.drop_vars("z")
        out = diabatic_lagrangian(ds, snd, observations=o2, **kw)
        np.testing.assert_allclose(out.qv.values, ref.qv.values, rtol=1e-9)
    # no observation matched: identical to the run without observations
    far = _stations(2000.0, 4000.0, [-3000.0], 290.0, 2e-3)
    a = diabatic_lagrangian(ds, snd, observations=far, **kw)
    b = diabatic_lagrangian(ds, snd, **kw)
    np.testing.assert_array_equal(a.theta.values, b.theta.values)
    assert np.isnan(a.insitu_weight).all() and "insitu_weight" not in b
    good = _stations(2000.0, 4000.0, [-300.0], 290.0, 2e-3)
    with pytest.raises(TypeError, match="Dataset"):
        diabatic_lagrangian(ds, snd, observations=good.theta, **kw)
    with pytest.raises(ValueError, match="unknown observation options"):
        diabatic_lagrangian(
            ds, snd, observations=good, **{**kw, "observation_options": {"r": 1}}
        )
    with pytest.raises(ValueError, match="positive"):
        diabatic_lagrangian(
            ds, snd, observations=good, **{**kw, "observation_options": {"window": 0}}
        )
    with pytest.raises(ValueError, match="'time'"):
        diabatic_lagrangian(ds, snd, observations=good.isel(time=0, drop=True), **kw)
    with pytest.raises(ValueError, match="'x' and 'y'"):
        diabatic_lagrangian(ds, snd, observations=good.drop_vars("x"), **kw)
    with pytest.raises(ValueError, match="origin_latitude"):
        bare = ds.copy()
        bare.attrs = {}
        diabatic_lagrangian(bare, snd, observations=obs, **kw)
    with pytest.raises(ValueError, match="'theta' or 'temperature'"):
        diabatic_lagrangian(ds, snd, observations=good.drop_vars("theta"), **kw)
    with pytest.raises(ValueError, match="'qv'"):
        diabatic_lagrangian(ds, snd, observations=good.drop_vars("qv"), **kw)


def test_real_nexrad_volume_with_ice_and_stations():
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
        z_lim=(0, 8e3),
        x_step=2000,
        y_step=2000,
        z_step=500,
        pseudo_cappi=False,
    )
    g = g.drop_vars("time").astype("float64")
    echo = (g.DBZH > 30).fillna(False)
    # a steady storm moving with the mean wind, rising 3 m/s in echo aloft
    winds = g.assign(
        u=xr.full_like(g.DBZH, 5.0),
        v=xr.full_like(g.DBZH, 0.0),
        w=xr.where(echo & (g.z > 0), 3.0, 0.0),
    ).assign_coords(time=T0)
    snd = _sounding()
    # stations at the ground under the echo, 3 K colder than the sounding
    sfc = echo.isel(z=0).values
    jj, ii = np.nonzero(sfc)
    pick = slice(0, None, max(1, jj.size // 15))
    th0 = float(snd.temperature[0]) * (1e5 / float(snd.pressure[0])) ** KAPPA
    st = xr.Dataset(
        {
            "x": ("station", g.x.values[ii[pick]]),
            "y": ("station", g.y.values[jj[pick]]),
            "theta": ("station", np.full(ii[pick].size, th0 - 3.0)),
            "qv": ("station", np.full(ii[pick].size, 0.012)),
        },
        coords={"time": ("station", np.full(ii[pick].size, T0))},
    )
    out = diabatic_lagrangian(
        winds,
        snd,
        storm_motion=(5.0, 0.0),
        extend=(3600.0, 0.0),
        ice=True,
        observations=st,
        observation_options={"radius": 1000.0, "kappa_s": 1e6},
        filter_passes=0,
        hole_fill=False,
    )
    hit = (out.flags.isel(z=0).values & 256) != 0
    assert hit.sum() == st.sizes["station"]
    np.testing.assert_allclose(out.theta.isel(z=0).values[hit], th0 - 3.0, atol=0.5)
    up = echo.values & (out.temperature.values < 263.15)
    assert up.sum() > 10
    assert float(out.qi.where(up).max()) > 0
    assert np.isfinite(out.theta.values[out.environment.values]).all()
