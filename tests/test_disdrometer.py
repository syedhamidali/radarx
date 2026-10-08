#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the disdrometer analysis
==================================
"""

import importlib

import numpy as np
import pytest
import xarray as xr
from scipy.special import gamma as gamma_fn

import radarx  # noqa: F401
from radarx.io import parsivel_classes
from radarx.retrieve import (
    disdrometer_qc,
    dsd_moments,
    fit_gamma,
    fit_gamma_moments,
    match_radar,
    number_concentration,
    process_disdrometer,
    radar_at_location,
    radar_from_dsd,
    raupach_berne_correction,
    terminal_fall_speed,
)

dmod = importlib.import_module("radarx.retrieve.disdrometer")

ENGINES = ["numpy"] + (["compiled"] if dmod.HAS_COMPILED_KERNEL else [])
compiled_only = pytest.mark.skipif(
    not dmod.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)


def _gamma(d, n0, mu, lam):
    return n0 * d**mu * np.exp(-lam * d)


def _station(n0=8000.0, mu=2.0, lam=3.0, ntime=12, dt=10.0, scale=None):
    """Parsivel counts of a gamma DSD, every drop at its velocity class."""
    ds = parsivel_classes()
    d = ds.diameter.values
    v = ds.velocity.values
    vt = np.maximum(9.65 - 10.3 * np.exp(-0.6 * d), 0.0)
    nd = _gamma(d, n0, mu, lam)
    nd[d < 0.25] = 0.0
    nd[d > 8.0] = 0.0
    area = 180e-6 * (30.0 - d / 2.0)
    counts = np.zeros((32, 32))
    for i in range(32):
        k = np.searchsorted(ds.velocity_upper.values, vt[i], side="right")
        k = min(k, 31)
        counts[k, i] = nd[i] * area[i] * ds.bin_width.values[i] * dt * v[k]
    scale = np.ones(ntime) if scale is None else np.asarray(scale, float)
    allc = scale[:, None, None] * counts[None]
    time = np.datetime64("2022-03-31T00:00:00", "ns") + np.arange(
        ntime
    ) * np.timedelta64(int(dt), "s")
    ds = ds.assign_coords(time=time)
    ds["counts"] = (("time", "velocity", "diameter"), allc)
    ds["sample_interval"] = ("time", np.full(ntime, dt))
    ds["rain_rate_instrument"] = ("time", np.full(ntime, 1.5))
    ds = ds.assign_coords(station="S1", latitude=33.75, longitude=-88.45, altitude=70.0)
    return ds, nd


def _fine(n0, mu, lam, dmax=8.0, step=0.01, extra=None):
    d = np.arange(step / 2, dmax, step)
    nd = _gamma(d, n0, mu, lam)
    coords = {
        "diameter": d,
        "bin_width": ("diameter", np.full(d.size, step)),
    }
    da = xr.DataArray(nd, dims="diameter", coords=coords)
    if extra:
        da = da.expand_dims(extra)
    return da


# --------------------------------------------------------------------------
# fall speed and quality control
# --------------------------------------------------------------------------


def test_terminal_fall_speed():
    v = terminal_fall_speed(np.array([0.05, 1.0, 5.0]))
    np.testing.assert_allclose(
        v, [0.0, 9.65 - 10.3 * np.exp(-0.6), 9.65 - 10.3 * np.exp(-3.0)]
    )
    aloft = terminal_fall_speed(xr.DataArray([2.0], dims="d"), air_density=0.9)
    np.testing.assert_allclose(aloft / terminal_fall_speed(2.0), (1.204 / 0.9) ** 0.4)
    assert aloft.attrs["units"] == "m s-1"


def test_qc_relative_band():
    ds, _ = _station(ntime=2)
    spikes = ds.counts.copy()
    spikes[:, 30, 3] = 50.0  # small and fast: splashing
    spikes[:, 2, 22] = 50.0  # large and slow: strong-wind artifact
    spikes[:, 25, 24] = 50.0  # 9.5 mm drop near its fall speed
    spikes[:, 4, 1] = 50.0  # unmeasured class (0.19 mm at 0.45 m/s)
    ds["counts"] = spikes
    out = disdrometer_qc(ds)
    assert out.qc_mask.dims == ("velocity", "diameter")  # no air data
    kept = out.counts_qc.isel(time=0)
    assert kept[30, 3] == 0 and kept[2, 22] == 0 and kept[25, 24] == 0
    assert kept[4, 1] == 0
    on_curve = _station(ntime=2)[0].counts.isel(time=0)
    on_curve[:, :2] = 0.0
    on_curve[:, 24:] = 0.0
    np.testing.assert_allclose(kept, on_curve)
    assert out.valid.all() and "relative" in out.attrs["qc"]
    wide = disdrometer_qc(ds, max_diameter=None, drop_unmeasured=False)
    assert wide.counts_qc[0, 4, 1] == 50.0
    rb = disdrometer_qc(ds, method="raupach2015")
    assert rb.counts_qc[0, 30, 3] == 0 and rb.counts_qc[0, 2, 22] == 0
    with pytest.raises(ValueError, match="method"):
        disdrometer_qc(ds, method="x")
    with pytest.raises(ValueError, match="tolerance"):
        disdrometer_qc(ds, tolerance=0)


def test_qc_wind_and_density():
    ds, _ = _station(ntime=3)
    ds["wind_speed"] = ("time", [3.0, 25.0, np.nan])
    out = disdrometer_qc(ds, max_wind=20.0)
    np.testing.assert_array_equal(out.valid, [True, False, True])
    assert np.isnan(out.counts_qc[1]).all()
    nd = number_concentration(out)
    assert np.isnan(nd[1]).all() and np.isfinite(nd[0]).all()
    with pytest.raises(ValueError, match="max_wind needs"):
        disdrometer_qc(ds.drop_vars("wind_speed"), max_wind=20.0)
    ds["air_pressure"] = ("time", [700.0, 1000.0, np.nan])
    ds["air_temperature"] = ("time", [0.0, 20.0, 20.0])
    ds["relative_humidity"] = ("time", [90.0, 50.0, 50.0])
    out = disdrometer_qc(ds)
    assert out.qc_mask.dims == ("time", "velocity", "diameter")
    vt = out.terminal_fall_speed
    assert vt[0, 15] > vt[1, 15]  # faster aloft (low density)
    np.testing.assert_allclose(vt[2], terminal_fall_speed(ds.diameter))  # gap
    flat = disdrometer_qc(ds, density_correction=False)
    assert flat.qc_mask.dims == ("velocity", "diameter")


def test_input_checks():
    ds, _ = _station(ntime=1)
    with pytest.raises(TypeError):
        disdrometer_qc(ds.counts)
    with pytest.raises(ValueError, match="no 'counts'"):
        disdrometer_qc(ds.drop_vars("counts"))
    with pytest.raises(ValueError, match="must be on"):
        disdrometer_qc(ds.assign(counts=ds.counts.isel(velocity=0)))
    with pytest.raises(ValueError, match="engine"):
        number_concentration(ds, engine="fast")


# --------------------------------------------------------------------------
# Raupach and Berne (2015)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_velocity_shift(engine):
    ds = parsivel_classes()
    c = np.zeros((2, 32, 32))
    c[0, 20, 10] = 30.0  # 1.38 mm drops at 4.4 m/s, below 5.1 m/s
    c[0, 21, 10] = 10.0
    c[1, 0, 2] = 8.0  # shifted below zero: lost
    vt = np.tile(np.maximum(9.65 - 10.3 * np.exp(-0.6 * ds.diameter.values), 0), (2, 1))
    vt[1, 2] = -1.0
    lo, up = ds.velocity_lower.values, ds.velocity_upper.values
    if engine == "compiled":
        out = dmod._disdrometer.velocity_shift(c, lo, up, vt, 0.1, 2)
    else:
        out = dmod._velocity_shift_numpy(c, lo, up, vt, 0.1)
    assert out[0, :, 10].sum() == pytest.approx(40.0)
    mean = (out[0, :, 10] * ds.velocity.values).sum() / 40.0
    assert abs(mean - vt[0, 10]) < 0.45  # within the class widths
    assert out[1].sum() < 8.0
    np.testing.assert_array_equal(out[0, :, 0], 0.0)


@compiled_only
def test_velocity_shift_engines_agree():
    rnd = np.random.default_rng(3)
    ds = parsivel_classes()
    c = rnd.poisson(2.0, (40, 32, 32)).astype(float)
    vt = np.tile(
        np.maximum(9.65 - 10.3 * np.exp(-0.6 * ds.diameter.values), 0), (40, 1)
    )
    vt[3, 5] = np.nan
    lo, up = ds.velocity_lower.values, ds.velocity_upper.values
    a = dmod._disdrometer.velocity_shift(c, lo, up, vt, 0.1, 0)
    b = dmod._velocity_shift_numpy(c, lo, up, vt, 0.1)
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(a[3, :, 5], c[3, :, 5])
    with pytest.raises(ValueError):
        dmod._disdrometer.velocity_shift(c, lo, up, vt[:, :5], 0.1, 0)
    with pytest.raises(ValueError):
        dmod._disdrometer.velocity_shift(c, lo, up, vt, 0.0, 0)
    with pytest.raises(ValueError):
        dmod._disdrometer.velocity_shift(c, lo[::-1].copy(), up, vt, 0.1, 0)
    with pytest.raises(ValueError):
        dmod._disdrometer.velocity_shift(c[0], lo, up, vt, 0.1, 0)


@pytest.mark.parametrize("engine", ENGINES)
def test_raupach_berne_correction(engine):
    ds, _ = _station(ntime=4)
    ds["rain_rate_instrument"] = ("time", [0.05, 1.5, 50.0, np.nan])
    out = raupach_berne_correction(ds, engine=engine)
    cf = out.concentration_factor
    # Parsivel2 (Table 10): class 3 at [0, 0.1) and [1, 2) mm/h, class 22 at
    # [2, 200), class 14 at [0, 0.1) without factor, classes 23+ uncorrected
    assert cf[0, 2] == 0.02 and cf[1, 2] == 0.06
    assert cf[2, 21] == 0.32 and cf[0, 13] == 1.0 and cf[2, 22] == 1.0
    assert np.isnan(cf[3]).all()
    nd = number_concentration(out, engine=engine)
    plain = number_concentration(disdrometer_qc(ds, method="raupach2015"))
    assert nd[1, 2] < plain[1, 2]
    p1 = raupach_berne_correction(ds, instrument="parsivel")
    assert p1.concentration_factor[1, 2] == 0.09  # Table 3, [1, 2)
    with pytest.raises(ValueError, match="instrument"):
        raupach_berne_correction(ds, instrument="x")
    with pytest.raises(ValueError, match="rain_rate_instrument"):
        raupach_berne_correction(ds.drop_vars("rain_rate_instrument"))
    with pytest.raises(ValueError, match="32 x 32"):
        raupach_berne_correction(ds.isel(diameter=slice(0, 30)))


# --------------------------------------------------------------------------
# N(D) and moments
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_number_concentration_recovers_gamma(engine):
    ds, truth = _station(scale=np.arange(1, 13))
    nd = number_concentration(ds, counts="counts", engine=engine)
    assert nd.dims == ("time", "diameter") and "bin_width" in nd.coords
    assert nd.station.item() == "S1"
    np.testing.assert_allclose(nd[0], truth, rtol=1e-12)
    np.testing.assert_allclose(nd[5], 6 * truth, rtol=1e-12)
    term = number_concentration(ds, counts="counts", velocity="terminal", engine=engine)
    good = truth > 0
    ratio = (term[0] / truth).values[good]
    assert np.all((ratio > 0.8) & (ratio < 1.25))  # class centre vs v_t
    half = number_concentration(ds, counts="counts", sample_interval=20.0)
    np.testing.assert_allclose(half[0], truth / 2, rtol=1e-12)
    no_dt = number_concentration(ds.drop_vars("sample_interval"), counts="counts")
    np.testing.assert_allclose(no_dt[0], truth, rtol=1e-12)
    with pytest.raises(ValueError, match="velocity"):
        number_concentration(ds, velocity="x")


@compiled_only
def test_number_concentration_engines_agree():
    rnd = np.random.default_rng(1)
    ds, _ = _station(ntime=30)
    ds["counts"] = ds.counts.copy(data=rnd.poisson(3.0, ds.counts.shape))
    ds["air_pressure"] = ("time", rnd.uniform(800, 1000, 30))
    ds["air_temperature"] = ("time", rnd.uniform(0, 30, 30))
    q = disdrometer_qc(ds)
    for velocity in ("measured", "terminal"):
        a = number_concentration(q, velocity=velocity, engine="compiled", n_threads=3)
        b = number_concentration(q, velocity=velocity, engine="numpy")
        np.testing.assert_allclose(a, b, rtol=1e-12)
    with pytest.raises(ValueError):
        dmod._disdrometer.number_concentration(
            q.counts.values.astype(float), np.ones((2, 32, 32)), np.ones((30, 32))
        )
    with pytest.raises(ValueError):
        dmod._disdrometer.number_concentration(
            q.counts.values[0].astype(float), np.ones((1, 32, 32)), np.ones((30, 32))
        )


def test_dsd_moments_closed_form():
    n0, mu, lam = 5000.0, 3.0, 4.0
    nd = _fine(n0, mu, lam, extra={"time": 2})
    mom = dsd_moments(nd)
    m3 = n0 * gamma_fn(mu + 4) / lam ** (mu + 4)
    m6 = n0 * gamma_fn(mu + 7) / lam ** (mu + 7)
    np.testing.assert_allclose(mom.DM, (mu + 4) / lam, rtol=1e-4)
    np.testing.assert_allclose(mom.D0, (mu + 3.67) / lam, rtol=5e-3)
    np.testing.assert_allclose(mom.LWC, np.pi / 6e3 * m3, rtol=1e-4)
    np.testing.assert_allclose(mom.DBZ_RAYLEIGH, 10 * np.log10(m6), atol=1e-3)
    np.testing.assert_allclose(
        mom.NT, n0 * gamma_fn(mu + 1) / lam ** (mu + 1), rtol=1e-4
    )
    sig = np.sqrt(mu + 4) / lam  # mass spectrum is gamma(mu + 3)
    np.testing.assert_allclose(mom.SIGMA_M, sig, rtol=1e-3)
    fit = fit_gamma(nd)
    np.testing.assert_allclose(mom.RAIN_RATE, fit.RAIN_RATE, rtol=1e-3)
    zero = dsd_moments(nd * 0.0)
    assert np.isnan(zero.DM).all() and np.isnan(zero.D0).all()
    assert (zero.NT == 0).all()
    with pytest.raises(ValueError, match="no 'D'"):
        dsd_moments(nd, dim="D")


# --------------------------------------------------------------------------
# gamma fits
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("moments", [(2, 4, 6), (3, 4, 6), (2, 3, 4)])
def test_fit_recovers_gamma(engine, moments):
    truth = [(8000.0, 0.0, 2.5), (2000.0, 3.0, 5.0), (300.0, 7.0, 9.0)]
    nd = xr.concat([_fine(*t) for t in truth], "time")
    fit = fit_gamma(nd, moments=moments, engine=engine)
    np.testing.assert_allclose(fit.MU, [t[1] for t in truth], atol=2e-3)
    np.testing.assert_allclose(fit.LAMBDA, [t[2] for t in truth], rtol=1e-3)
    np.testing.assert_allclose(fit.N0, [t[0] for t in truth], rtol=1e-2)
    assert (fit.FIT_ITERATIONS == 0).all()
    assert fit.attrs["fit"] == "MM" + "".join(map(str, moments))
    # truncation of a nearly untruncated DSD changes nothing
    tr = fit_gamma(nd, moments=moments, truncated=True, engine=engine)
    np.testing.assert_allclose(tr.MU, [t[1] for t in truth], atol=2e-3)


@pytest.mark.parametrize("engine", ENGINES)
def test_truncated_fit_recovers_truncated_gamma(engine):
    truth = [(4000.0, 2.0, 2.0, 2.5), (4000.0, -0.5, 1.5, 3.0), (900.0, 5.0, 4.0, 1.5)]
    nds = [_fine(n0, mu, lam, dmax=dmax).rename("ND") for n0, mu, lam, dmax in truth]
    nd = xr.concat(nds, "time", join="outer", fill_value=0.0)
    nd = nd.assign_coords(bin_width=("diameter", np.full(nd.diameter.size, 0.01)))
    fit = fit_gamma(nd, truncated=True, engine=engine)
    np.testing.assert_allclose(fit.MU, [t[1] for t in truth], atol=0.02)
    np.testing.assert_allclose(fit.LAMBDA, [t[2] for t in truth], rtol=0.01)
    np.testing.assert_allclose(fit.N0, [t[0] for t in truth], rtol=0.05)
    assert (fit.FIT_ITERATIONS > 0).all()
    plain = fit_gamma(nd, engine=engine)
    assert plain.MU[0] > 3.0  # the untruncated fit is biased
    low = fit_gamma(nd, truncated=True, lower_truncation=True, engine=engine)
    np.testing.assert_allclose(low.MU, fit.MU, atol=0.02)


@compiled_only
@pytest.mark.parametrize("truncated", [False, True])
@pytest.mark.parametrize("moments", [(2, 4, 6), (3, 4, 6), (2, 3, 4)])
def test_fit_engines_agree(truncated, moments):
    rnd = np.random.default_rng(7)
    ds, _ = _station(ntime=200)
    lam = rnd.uniform(1.0, 8.0, 200)
    mu = rnd.uniform(-0.5, 6.0, 200)
    d = ds.diameter.values
    counts = rnd.poisson(
        (2e4 * d[None, :] ** mu[:, None] * np.exp(-lam[:, None] * d[None, :])).clip(
            0, 5e3
        )[:, None, :]
        * (ds.counts.values[0] > 0)
        / 50.0
    ).astype(float)
    ds["counts"] = ds.counts.copy(data=counts)
    nd = number_concentration(disdrometer_qc(ds))
    nd[:3] = 0.0  # empty spectra
    nd[3, 5] = np.nan
    kw = dict(moments=moments, truncated=truncated, lower_truncation=truncated)
    a = fit_gamma(nd, engine="compiled", n_threads=4, **kw)
    b = fit_gamma(nd, engine="numpy", **kw)
    np.testing.assert_array_equal(np.isnan(a.MU), np.isnan(b.MU))
    assert np.isfinite(a.MU).sum() > 120
    np.testing.assert_allclose(a.MU, b.MU, atol=1e-5)
    np.testing.assert_allclose(a.LAMBDA, b.LAMBDA, rtol=1e-6)
    np.testing.assert_allclose(a.N0, b.N0, rtol=1e-5)
    assert np.isnan(a.MU[:4]).all()


def test_fit_matches_existing_m246_and_errors():
    nd = xr.concat([_fine(5000.0, 2.0, 3.0), _fine(800.0, 4.0, 6.0)], "t")
    a = fit_gamma(nd)
    b = fit_gamma_moments(nd)
    np.testing.assert_allclose(a.MU, b.MU, rtol=1e-8)
    np.testing.assert_allclose(a.NW, b.NW, rtol=1e-8)
    with pytest.raises(ValueError, match="three orders"):
        fit_gamma(nd, moments=(4, 2, 6))
    with pytest.raises(ValueError, match="mu_range"):
        fit_gamma(nd, mu_range=(-4.0, 10.0))
    narrow = fit_gamma(nd, mu_range=(2.5, 10.0))
    assert np.isnan(narrow.MU[0]) and np.isfinite(narrow.MU[1])


@compiled_only
def test_fit_kernel_checks():
    x = np.ones((2, 5))
    d = np.arange(1.0, 6.0)
    args = (x, d, np.ones(5), d - 0.5, d + 0.5)
    with pytest.raises(ValueError):
        dmod._disdrometer.fit_gamma(
            *args, 4, 2, 6, False, False, -1.0, 50.0, 10, 1e-8, 0
        )
    with pytest.raises(ValueError):
        dmod._disdrometer.fit_gamma(
            *args, 2, 4, 6, False, False, -4.0, 50.0, 10, 1e-8, 0
        )
    with pytest.raises(ValueError):
        dmod._disdrometer.fit_gamma(
            x[0], *args[1:], 2, 4, 6, False, False, -1.0, 50.0, 10, 1e-8, 0
        )
    with pytest.raises(ValueError):
        dmod._disdrometer.fit_gamma(
            x, d[:3], *args[2:], 2, 4, 6, False, False, -1.0, 50.0, 10, 1e-8, 0
        )


# --------------------------------------------------------------------------
# all products
# --------------------------------------------------------------------------


def test_process_disdrometer_and_accessor():
    ds, truth = _station(ntime=6)
    out = ds.radarx.disdrometer(fits=("MM246", "TMM346"), band="C")
    assert out.ND.dims == ("time", "diameter")
    for name in ("DBZH", "ZDR", "KDP", "RAIN_RATE", "DM", "D0", "NW", "NT"):
        assert name in out and out[name].dims == ("time",)
    assert "MU_MM246" in out and "LAMBDA_TMM346" in out
    assert out.attrs["band"] == "C" and out.station.item() == "S1"
    ref = radar_from_dsd(out.ND, band="C")
    np.testing.assert_allclose(out.DBZH, ref.DBZH)
    np.testing.assert_allclose(out.MU_MM246, 2.0, atol=0.5)
    rb = process_disdrometer(ds, qc="raupach_berne")
    assert np.isfinite(rb.DBZH).all()
    raw = process_disdrometer(ds, qc=None)
    assert raw.attrs["qc"] == "none"
    empty = ds.assign(counts=ds.counts * 0.0)
    nil = process_disdrometer(empty)
    assert np.isnan(nil.DBZH).all()
    with pytest.raises(ValueError, match="unknown fit"):
        process_disdrometer(ds, fits=("XX",))
    with pytest.raises(ValueError, match="qc must be"):
        process_disdrometer(ds, qc="x")


# --------------------------------------------------------------------------
# radar matching
# --------------------------------------------------------------------------

R_EARTH = 6371000.0


def _destination(lat0, lon0, az, s):
    p0, l0, a, d = np.deg2rad(lat0), np.deg2rad(lon0), np.deg2rad(az), s / R_EARTH
    p1 = np.arcsin(np.sin(p0) * np.cos(d) + np.cos(p0) * np.sin(d) * np.cos(a))
    l1 = l0 + np.arctan2(
        np.sin(a) * np.sin(d) * np.cos(p0), np.cos(d) - np.sin(p0) * np.sin(p1)
    )
    return np.rad2deg(p1), np.rad2deg(l1)


def _sweep(t0, angle=0.5, offset=0.0):
    az = np.arange(0.5, 360.0, 1.0)
    rng = np.arange(125.0, 60000.0, 250.0)
    time = np.datetime64(t0, "ns") + (az * 1e8).astype("timedelta64[ns]")
    dbz = 20.0 + offset + 0.0 * az[:, None] + rng[None, :] / 2000.0
    return xr.Dataset(
        {
            "DBZH": (("azimuth", "range"), dbz, {"units": "dBZ"}),
            "VEL": (
                ("azimuth", "range"),
                np.tile(az[:, None], (1, rng.size)),
                {"units": "m/s"},
            ),
        },
        coords={
            "azimuth": az,
            "range": rng,
            "elevation": ("azimuth", np.full(az.size, angle)),
            "time": ("azimuth", time),
            "sweep_fixed_angle": angle,
            "latitude": 33.9,
            "longitude": -88.33,
            "altitude": 150.0,
        },
    )


def test_radar_at_location_geometry():
    lat, lon = _destination(33.9, -88.33, 45.2, 20000.0)
    sweep = _sweep("2022-03-31T00:30:00")
    out = radar_at_location(sweep, lat, lon, 70.0)
    assert out.sizes["time"] == 1
    assert out.azimuth.item() == 45.5
    assert abs(out.range.item() - 20000.0) <= 125.0
    assert out.gate_distance.item() < 200.0
    re = 4.0 / 3.0 * R_EARTH
    r = out.range.item()
    h = np.sqrt(r**2 + re**2 + 2 * r * re * np.sin(np.deg2rad(0.5))) - re + 150.0
    assert out.beam_height.item() == pytest.approx(h)
    assert out.height_above_ground.item() == pytest.approx(h - 70.0)
    assert out.DBZH.item() == pytest.approx(20.0 + r / 2000.0)
    assert str(out.time.values[0])[:21] == "2022-03-31T00:30:04.5"
    assert out.DBZH.attrs["units"] == "dBZ"
    avg = radar_at_location(sweep, lat, lon, radius=600.0, fields=["DBZH", "VEL", "XX"])
    assert avg.DBZH.item() == pytest.approx(out.DBZH.item(), abs=0.2)
    assert 44.0 < avg.VEL.item() < 47.0 and np.isnan(avg.XX.item())
    with pytest.raises(ValueError, match="no radar"):
        radar_at_location(sweep.drop_vars("latitude"), lat, lon)
    with pytest.raises(ValueError, match="no radar sweeps"):
        radar_at_location([], lat, lon)


def _volume(t0, offset=0.0):
    root = xr.Dataset(coords={"latitude": 33.9, "longitude": -88.33, "altitude": 150.0})
    s0 = _sweep(t0, 0.5, offset).drop_vars(["latitude", "longitude", "altitude"])
    s1 = _sweep(t0, 1.5, offset + 5).drop_vars(["latitude", "longitude", "altitude"])
    return xr.DataTree.from_dict({"/": root, "sweep_0": s0, "sweep_1": s1})


def test_radar_at_location_datatree():
    lat, lon = _destination(33.9, -88.33, 100.0, 10000.0)
    vols = [_volume(f"2022-03-31T00:{m:02d}:00", offset=m) for m in (35, 30)]
    out = radar_at_location(vols, lat, lon)
    assert out.sizes["time"] == 2 and out.time[0] < out.time[1]  # sorted
    by_angle = radar_at_location(vols, lat, lon, sweep=1.4)
    assert (by_angle.elevation == 1.5).all()
    assert (by_angle.DBZH - out.DBZH == 5.0).all()
    with pytest.raises(ValueError, match="no sweep_7"):
        radar_at_location(vols[0], lat, lon, sweep=7)
    with pytest.raises(ValueError, match="no sweep_"):
        radar_at_location(xr.DataTree(), lat, lon)


def test_match_radar():
    ds, truth = _station(ntime=60, dt=10.0)  # 10 minutes from 00:00
    lat, lon = _destination(33.9, -88.33, 200.0, 15000.0)
    ds = ds.assign_coords(latitude=lat, longitude=lon)
    vols = [_volume(f"2022-03-31T00:0{m}:00") for m in (2, 5)]
    pairs = match_radar(ds, vols, window="60s")
    assert pairs.sizes["time"] == 2
    assert (pairs.n_records == 6).all()
    ref = radar_from_dsd(number_concentration(disdrometer_qc(ds)).isel(time=0))
    np.testing.assert_allclose(pairs.DBZH_disdrometer, ref.DBZH, rtol=1e-10)
    assert "DBZH" in pairs and "RAIN_RATE_disdrometer" in pairs
    assert pairs.ND.dims == ("time", "diameter")
    assert (pairs.delay == 0).all() and pairs.attrs["station"] == "S1"
    fall = match_radar(ds, vols, delay="fall")
    h = fall.height_above_ground.values
    assert np.all(fall.delay > h / 10.0) and np.all(fall.delay < h / 3.0)
    fixed = match_radar(ds, vols, delay="2min", window="30s")
    assert (fixed.delay == 120.0).all() and (fixed.n_records == 3).all()
    late = match_radar(ds, vols, delay="20min")
    assert np.isnan(late.DBZH_disdrometer).all() and (late.n_records == 0).all()
    prod = process_disdrometer(ds)
    again = match_radar(prod, vols)
    np.testing.assert_allclose(again.DBZH_disdrometer, pairs.DBZH_disdrometer)
    rbm = match_radar(ds, vols, qc="raupach_berne")
    assert np.isfinite(rbm.DBZH_disdrometer).all()
    nan_alt = match_radar(ds.assign_coords(altitude=np.nan), vols)
    assert np.isfinite(nan_alt.height_above_ground).all()
    with pytest.raises(ValueError, match="unknown qc"):
        match_radar(ds, vols, qc="x")
    with pytest.raises(ValueError, match="no 'latitude'"):
        match_radar(ds.drop_vars("latitude"), vols)
    two = xr.concat([ds, ds], "station")
    with pytest.raises(ValueError, match="select one station"):
        match_radar(two, vols)


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def klbb():
    xd = pytest.importorskip("xradar")
    from open_radar_data import DATASETS

    file = DATASETS.fetch("KLBB20160601_150025_V06")
    dtree = xd.io.open_nexradlevel2_datatree(file, sweep=[0])
    ds = dtree["sweep_0"].to_dataset(inherit=False)
    ds["DBZH"] = ds["DBZH"].where(ds["DBZH"] > -32.0)
    dtree["sweep_0"] = ds
    return dtree


def test_real_nexrad_gate_above_instrument(klbb):
    """A disdrometer placed under a rainy KLBB gate gets that gate."""
    sweep = klbb["sweep_0"].to_dataset()
    dbz = sweep.DBZH.transpose("azimuth", "range")
    ia, ir = np.unravel_index(
        int(np.nanargmax(dbz.where(dbz.range < 60e3).values)), dbz.shape
    )
    az = float(sweep.azimuth[ia])
    r = float(sweep.range[ir])
    lat0, lon0 = float(klbb["latitude"]), float(klbb["longitude"])
    re = 4.0 / 3.0 * R_EARTH
    el = float(sweep.elevation[ia])
    h = np.sqrt(r**2 + re**2 + 2 * r * re * np.sin(np.deg2rad(el))) - re
    s = re * np.arcsin(r * np.cos(np.deg2rad(el)) / (re + h))
    lat, lon = _destination(lat0, lon0, az, s)
    out = radar_at_location(klbb, lat, lon, fields=["DBZH", "ZDR"])
    assert out.range.item() == r and out.azimuth.item() == pytest.approx(az)
    assert out.DBZH.item() == pytest.approx(float(dbz[ia, ir]))
    assert out.gate_distance.item() < 50.0
    assert out.time.values[0] == sweep.time.values[ia]
    # a station of gamma rain under that gate gives a comparable reflectivity
    ds, _ = _station(n0=8000.0, mu=2.0, lam=2.0, ntime=30)
    ds = ds.assign_coords(
        latitude=lat,
        longitude=lon,
        time=sweep.time.values[ia]
        - np.timedelta64(150, "s")
        + np.arange(30) * np.timedelta64(10, "s"),
    )
    pairs = match_radar(ds, klbb, fields=["DBZH"])
    assert np.isfinite(pairs.DBZH_disdrometer).all()
    assert 30.0 < pairs.DBZH_disdrometer.item() < 60.0
