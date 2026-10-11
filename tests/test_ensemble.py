#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the ensembles and the uncertainty of the retrievals
=============================================================

The error model is checked against closed-form values written from the
equations and against Monte Carlo simulations of the estimators built from
complex Gaussian signals; the random numbers against a Philox-4x64-10 written
with Python integers; the statistics against NumPy. The compiled kernel and
the NumPy implementation draw the same 64-bit words and agree to rounding
(rtol 1e-12) elsewhere.
"""
import math

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.retrieve import _ensemble_numpy as kernels
from radarx.retrieve import (
    correct_attenuation,
    dsd,
    dsd_bayesian,
    ensemble,
    ensemble_retrieval,
    ensemble_statistics,
    estimate_kdp,
    hid,
    measurement_error,
    perturb_radar,
    rain_rate_a,
    rain_rate_kdp,
    rain_rate_z,
    rain_rate_z_zdr,
    relation_scatter,
)

from .provenance_helpers import without_provenance
from .test_dsd import _sweep as dsd_sweep

ENGINES = ["numpy"] + (["compiled"] if ensemble.HAS_COMPILED_KERNEL else [])
compiled_only = pytest.mark.skipif(
    not ensemble.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)

M_MULT = (0xD2E7470EE14C6C93, 0xCA5A826395121157)
W_ADD = (0x9E3779B97F4A7C15, 0xBB67AE8584CAA73B)
MASK = (1 << 64) - 1
DB = 10.0 / math.log(10.0)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _sweep(nray=8, ng=160, seed=0):
    """Rain cell with ZH, ZDR, a noisy phase, SNR, a velocity and a temperature."""
    ds, dm, nw, kdp = dsd_sweep(nray=nray, ng=ng, seed=seed)
    shape = (nray, ng)
    r = ds["range"].values / 1000.0
    ds = ds.assign(
        SNR=(("azimuth", "range"), np.broadcast_to(35.0 - 0.2 * r, shape).copy()),
        VRADH=(("azimuth", "range"), np.broadcast_to(5.0 + 0.0 * r, shape).copy()),
        TEMP=(("azimuth", "range"), np.full(shape, 15.0)),
        nyquist_velocity=("azimuth", np.full(nray, 20.0)),
    )
    ds["DBZH"].attrs["units"] = "dBZ"
    return ds


def _constant(snr=None, rho=0.97, zdr=0.5, **extra):
    """One gate with the given values."""
    data = {
        "DBZH": 30.0,
        "ZDR": zdr,
        "PHIDP": 20.0,
        "RHOHV": rho,
        "VRADH": 5.0,
        **extra,
    }
    if snr is not None:
        data["SNR"] = snr
    return xr.Dataset(
        {k: (("azimuth", "range"), np.full((1, 1), v)) for k, v in data.items()}
    )


def _volume():
    first = _sweep(nray=6, ng=120, seed=1)
    second = _sweep(nray=5, ng=100, seed=2)
    empty = xr.Dataset({"OTHER": (("azimuth", "range"), np.zeros((3, 4)))})
    root = xr.Dataset(coords={"latitude": 10.0, "longitude": 20.0, "altitude": 5.0})
    return xr.DataTree.from_dict(
        {"/": root, "sweep_0": first, "sweep_1": second, "sweep_2": empty}
    )


def _rain_ds():
    """A sweep with all fields and a uniform temperature, for every retrieval."""
    return _sweep(nray=6, ng=260, seed=3)


# --------------------------------------------------------------------------
# random numbers
# --------------------------------------------------------------------------


def philox_reference(counter, key, rounds=10):
    """Philox-4x64-10 with Python integers (Salmon et al. 2011)."""
    c = list(counter)
    k = list(key)
    for _ in range(rounds):
        p0 = M_MULT[0] * c[0]
        p1 = M_MULT[1] * c[2]
        c = [(p1 >> 64) ^ c[1] ^ k[0], p1 & MASK, (p0 >> 64) ^ c[3] ^ k[1], p0 & MASK]
        k = [(k[0] + W_ADD[0]) & MASK, (k[1] + W_ADD[1]) & MASK]
    return c


def test_words_are_philox_4x64_10():
    key = (12345, 678)
    for member, block in [(0, 0), (3, 1), (9, 2)]:
        w = kernels.words(key, member, block, 6)
        for i in range(6):
            expect = philox_reference((i + 1, member, block, 0), key)
            assert [int(x) for x in w[i]] == expect


@compiled_only
def test_compiled_words_equal_numpy_words():
    key = (2**63 + 5, 17)
    for member, block in [(0, 0), (11, 1), (4, 2)]:
        a = ensemble._kernels._ensemble.words(key[0], key[1], member, block, 50)
        assert np.array_equal(a, kernels.words(key, member, block, 50))


def test_gaussian_numbers():
    key = kernels.make_key(7, 2)
    z = kernels.gate_normals(key, 3, 200000, True)
    assert z.shape == (5, 200000)
    assert np.all(np.abs(z.mean(axis=1)) < 0.012)
    assert np.all(np.abs(z.std(axis=1) - 1.0) < 0.012)
    corr = np.corrcoef(z)
    assert np.max(np.abs(corr - np.eye(5))) < 0.012
    assert not np.any(kernels.gate_normals(key, 3, 10, False)[4])
    # members and sweeps have their own numbers
    other = kernels.gate_normals(key, 4, 200000, False)
    assert abs(np.corrcoef(z[0], other[0])[0, 1]) < 0.012
    other = kernels.gate_normals(kernels.make_key(7, 3), 3, 200000, False)
    assert abs(np.corrcoef(z[0], other[0])[0, 1]) < 0.012
    four = kernels.member_normals(key, 5, 2)
    assert four.shape == (4,) and np.all(np.isfinite(four))


def test_seed_is_checked():
    for bad in (-1, 2**64):
        with pytest.raises(ValueError, match="seed"):
            kernels.make_key(bad, 0)
    assert kernels.make_key(2**64 - 1, 3) == (2**64 - 1, 3)


# --------------------------------------------------------------------------
# error model
# --------------------------------------------------------------------------


def independent_samples(m, sigma_vn):
    """M_I (Eq. A7) and M_I1 (Eq. C6) by explicit sums."""
    s1 = sum(
        (1 - k / m) * math.exp(-((math.pi * sigma_vn * k) ** 2)) for k in range(1, m)
    )
    s2 = sum(
        (1 - k / m) * math.exp(-((math.pi * sigma_vn * k) ** 2))
        for k in range(1, m - 1)
    )
    return m / (1 + 2 * s1), (m - 1) / (1 + 2 * s2)


def reference_sd(m, rho, snr_db, zdr_db, width, va):
    """The standard deviations from the equations, with linear SNRs."""
    sigma_vn = width / va
    mi, mi1 = independent_samples(m, sigma_vn)
    sh = 10 ** (snr_db / 10)
    sv = sh / 10 ** (zdr_db / 10)
    vh = (2 * sh + 1) / (m * sh**2) + 1 / mi
    vv = (2 * sv + 1) / (m * sv**2) + 1 / mi
    zdr = DB * math.sqrt(vh + vv - 2 * rho**2 / mi)
    phi = math.degrees(
        math.sqrt((sh + sv + 1) / (m * sh * sv) + (1 - rho**2) / mi)
        / (math.sqrt(2) * rho)
    )
    rho_sd = math.sqrt(
        (1 - 2 * sh) * rho**2 / (4 * m * sh**2)
        + (1 - 2 * sv) * rho**2 / (4 * m * sv**2)
        + (sh + sv + 1) / (2 * m * sh * sv)
        + (1 - rho**2) ** 2 / (2 * mi)
    )
    r1 = math.exp(-((math.pi * sigma_vn) ** 2))
    vel = (
        va
        / (math.sqrt(2) * math.pi * math.sqrt(r1))
        * math.sqrt((2 * sh * (1 - r1) + 1) / ((m - 1) * sh**2) + (1 - r1) / mi1)
    )
    return {
        "dbz": DB * math.sqrt(vh),
        "zdr": zdr,
        "phidp": phi,
        "rhohv": rho_sd,
        "vel": vel,
    }


@pytest.mark.parametrize(
    "m,rho,snr,zdr,width",
    [
        (64, 0.99, 20.0, 0.0, 3.0),
        (32, 0.95, 8.0, 2.5, 4.0),
        (128, 0.9, 12.0, -0.5, 1.5),
    ],
)
def test_measurement_error_equals_the_equations(m, rho, snr, zdr, width):
    ds = _constant(snr=snr, rho=rho, zdr=zdr)
    err = measurement_error(ds, n_samples=m, spectrum_width=width, nyquist=25.0)
    ref = reference_sd(m, rho, snr, zdr, width, 25.0)
    for kind, name in (
        ("dbz", "DBZH"),
        ("zdr", "ZDR"),
        ("phidp", "PHIDP"),
        ("rhohv", "RHOHV"),
        ("vel", "VRADH"),
    ):
        got = float(err[f"{name}_error"].values.ravel()[0])
        assert got == pytest.approx(ref[kind], rel=1e-5)


def test_high_snr_limits():
    """Without receiver noise the equations reduce to the signal terms."""
    ds = _constant(rho=0.99)
    err = measurement_error(ds, n_samples=64, spectrum_width=3.0, nyquist=25.0)
    mi, _ = independent_samples(64, 3.0 / 25.0)
    assert float(err.DBZH_error.squeeze()) == pytest.approx(
        DB / math.sqrt(mi), rel=1e-5
    )
    assert float(err.ZDR_error.squeeze()) == pytest.approx(
        DB * math.sqrt(2 * (1 - 0.99**2) / mi), rel=1e-5
    )
    assert float(err.RHOHV_error.squeeze()) == pytest.approx(
        (1 - 0.99**2) / math.sqrt(2 * mi), rel=1e-5
    )
    # Eq. A9, the approximation for 0.04 <= sigma_vn <= 0.6
    assert mi == pytest.approx(64 * math.sqrt(math.pi) * 0.12, rel=0.1)
    units = {v: err[v].attrs["units"] for v in err.data_vars}
    assert units == {
        "DBZH_error": "dB",
        "ZDR_error": "dB",
        "PHIDP_error": "degrees",
        "RHOHV_error": "1",
        "VRADH_error": "m s-1",
    }


def test_snr_forms_agree():
    ds = _constant(snr=14.0)
    base = measurement_error(ds, n_samples=40)
    assert float(
        measurement_error(ds, n_samples=40, snr=14.0).PHIDP_error.squeeze()
    ) == (pytest.approx(float(base.PHIDP_error.squeeze())))
    by_name = measurement_error(ds.rename(SNR="S"), snr="S", n_samples=40)
    array = measurement_error(ds, snr=ds["SNR"], n_samples=40)
    for other in (by_name, array):
        xr.testing.assert_allclose(other.PHIDP_error, base.PHIDP_error)
    # a higher SNR makes the errors smaller; a lower one larger
    low = measurement_error(ds, n_samples=40, snr=3.0)
    high = measurement_error(ds, n_samples=40, snr=30.0)
    for name in ("ZDR_error", "PHIDP_error", "RHOHV_error", "DBZH_error"):
        assert float(low[name].squeeze()) > float(high[name].squeeze())
    # below -10 dB the inverse SNR is limited, the errors stay finite
    assert np.isfinite(measurement_error(ds, snr=-40.0).ZDR_error.values).all()
    # NaN SNR gives NaN
    assert np.isnan(measurement_error(ds, snr=np.nan).ZDR_error.values).all()


def test_spectrum_width_as_field_and_nyquist_from_data():
    ds = _sweep(nray=3, ng=40).assign(WIDTH=lambda d: d.DBZH * 0 + 3.0)
    from_field = measurement_error(ds, spectrum_width="WIDTH")
    scalar = measurement_error(ds, spectrum_width=3.0)
    xr.testing.assert_allclose(from_field.DBZH_error, scalar.DBZH_error, rtol=1e-3)
    # the Nyquist velocity of the data (20 m/s) is used; 40 m/s gives narrower spectra
    wide = measurement_error(ds, spectrum_width=3.0, nyquist=40.0)
    assert float(wide.DBZH_error.mean()) > float(scalar.DBZH_error.mean())
    with pytest.raises(KeyError):
        measurement_error(ds, spectrum_width="nope")
    # the default without the variable
    plain = measurement_error(ds.drop_vars("nyquist_velocity"), spectrum_width=3.0)
    explicit = measurement_error(ds, spectrum_width=3.0, nyquist=25.0)
    xr.testing.assert_allclose(plain.DBZH_error, explicit.DBZH_error)
    assert float(plain.DBZH_error.mean()) != float(scalar.DBZH_error.mean())
    with pytest.raises(ValueError, match="nyquist"):
        measurement_error(ds, nyquist=-1.0)
    # NaN Nyquist velocities are ignored
    nan = ds.assign(nyquist_velocity=("azimuth", np.full(3, np.nan)))
    assert np.isfinite(measurement_error(nan).DBZH_error.values).all()


def test_estimate_snr_option():
    ds = _sweep(nray=4, ng=60)
    estimated = measurement_error(ds.drop_vars("SNR"), snr="estimate")
    from radarx.retrieve import snr as snr_of

    expect = measurement_error(ds.drop_vars("SNR"), snr=snr_of(ds, dbz="DBZH"))
    xr.testing.assert_allclose(estimated.ZDR_error, expect.ZDR_error)
    with pytest.raises(Exception, match="reflectivity"):
        measurement_error(ds[["ZDR", "RHOHV"]], snr="estimate")


def simulate(m, rho, snr_db, nrep, seed):
    """SD of ZDR (dB), the phase (degrees) and the correlation of the SHV estimators."""
    rng = np.random.default_rng(seed)
    s_h = 10 ** (snr_db / 10)

    def gauss():
        return (
            rng.standard_normal((nrep, m)) + 1j * rng.standard_normal((nrep, m))
        ) / np.sqrt(2)

    common, other = gauss(), gauss()
    h = np.sqrt(s_h) * common + gauss()
    v = np.sqrt(s_h) * (rho * common + np.sqrt(1 - rho**2) * other) + gauss()
    sh = (abs(h) ** 2).mean(axis=1) - 1.0
    sv = (abs(v) ** 2).mean(axis=1) - 1.0
    cross = (h * v.conj()).mean(axis=1)
    ok = (sh > 0) & (sv > 0)
    zdr = 10 * np.log10(sh[ok] / sv[ok])
    rho_hat = abs(cross[ok]) / np.sqrt(sh[ok] * sv[ok])
    return zdr.std(), np.degrees(np.angle(cross)).std(), rho_hat.std()


@pytest.mark.parametrize("rho,snr", [(0.99, 20.0), (0.95, 10.0), (0.9, 12.0)])
def test_measurement_error_against_simulated_estimators(rho, snr):
    """For uncorrelated pulses (sigma_vn = 1 gives M_I = M) the formulas are Monte Carlo results."""
    m = 32
    ds = _constant(snr=snr, rho=rho, zdr=0.0)
    err = measurement_error(ds, n_samples=m, spectrum_width=1000.0, nyquist=1000.0)
    sd_zdr, sd_phi, sd_rho = simulate(m, rho, snr, 40000, seed=11)
    assert float(err.ZDR_error.squeeze()) == pytest.approx(sd_zdr, rel=0.04)
    assert float(err.PHIDP_error.squeeze()) == pytest.approx(sd_phi, rel=0.04)
    assert float(err.RHOHV_error.squeeze()) == pytest.approx(sd_rho, rel=0.05)


def test_velocity_and_power_errors_against_correlated_time_series():
    """A Gaussian Doppler spectrum: power estimate (A6) and pulse-pair velocity (C10)."""
    m, sigma_vn, snr = 32, 0.15, 15.0
    rng = np.random.default_rng(3)
    lag = np.abs(np.arange(m)[:, None] - np.arange(m)[None, :])
    chol = np.linalg.cholesky(
        np.exp(-((np.pi * sigma_vn * lag) ** 2) / 2) + 1e-12 * np.eye(m)
    )
    shift = np.exp(1j * np.pi * 0.3 * np.arange(m))
    s = 10 ** (snr / 10)
    nrep = 40000

    def gauss():
        return (
            rng.standard_normal((nrep, m)) + 1j * rng.standard_normal((nrep, m))
        ) / np.sqrt(2)

    h = np.sqrt(s) * (gauss() @ chol.T) * shift + gauss()
    power = (abs(h) ** 2).mean(axis=1) - 1.0
    lag1 = (h[:, :-1].conj() * h[:, 1:]).mean(axis=1)
    sim_dbz = (10 * np.log10(power / s)).std()
    sim_vel = np.angle(lag1).std() / np.pi * 25.0
    ds = _constant(snr=snr, rho=0.99)
    err = measurement_error(
        ds, n_samples=m, spectrum_width=sigma_vn * 25.0, nyquist=25.0
    )
    assert float(err.DBZH_error.squeeze()) == pytest.approx(sim_dbz, rel=0.03)
    assert float(err.VRADH_error.squeeze()) == pytest.approx(sim_vel, rel=0.03)


def test_invalid_noise_options():
    ds = _constant()
    with pytest.raises(ValueError, match="n_samples"):
        measurement_error(ds, n_samples=3)
    with pytest.raises(KeyError):
        measurement_error(xr.Dataset({"OTHER": (("a",), np.zeros(3))}))
    with pytest.raises(ValueError, match="name"):
        measurement_error(xr.DataArray(np.zeros((2, 2)), dims=("azimuth", "range")))
    with pytest.raises(TypeError, match="xarray"):
        measurement_error(np.zeros(3))
    with pytest.raises(ValueError, match="unknown noise"):
        ensemble_retrieval(_constant(), "rain_rate_z", noise={"bad": 1})


# --------------------------------------------------------------------------
# perturbation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_errors_have_the_modelled_moments(engine):
    """Over many gates of a uniform field the errors have the SDs and the correlation of the model."""
    shape = (400, 500)
    base = {
        "DBZH": 30.0,
        "ZDR": 1.0,
        "PHIDP": 20.0,
        "RHOHV": 0.96,
        "VRADH": 5.0,
        "SNR": 14.0,
    }
    ds = xr.Dataset(
        {k: (("azimuth", "range"), np.full(shape, v)) for k, v in base.items()}
    )
    out = perturb_radar(
        ds, members=2, seed=5, calibration=None, snr="SNR", engine=engine
    )
    ref = measurement_error(ds.isel(azimuth=[0], range=[0]), snr="SNR")
    diff = {k: (out[k] - ds[k]).values for k in ("DBZH", "ZDR", "PHIDP", "VRADH")}
    for k in diff:
        sd = float(ref[f"{k}_error"].squeeze())
        assert np.abs(diff[k].mean()) < 4 * sd / np.sqrt(diff[k].size)
        assert diff[k].std() == pytest.approx(sd, rel=0.01)
    rho = out["RHOHV"].values - 0.96
    assert rho.std() == pytest.approx(float(ref.RHOHV_error.squeeze()), rel=0.03)
    # Z and ZDR errors are correlated like the powers: cov = rho^2 / M_I
    mi, _ = independent_samples(64, 3.0 / 25.0)
    var_h = (2 * 10 ** (-1.4) + 10 ** (-2.8)) / 64 + 1 / mi
    expected = DB**2 * (var_h - 0.96**2 / mi)
    cov = np.mean(diff["DBZH"] * diff["ZDR"])
    assert cov == pytest.approx(expected, rel=0.05)
    # the phase error is independent of the powers
    assert abs(np.corrcoef(diff["PHIDP"].ravel(), diff["DBZH"].ravel())[0, 1]) < 0.01


@pytest.mark.parametrize("engine", ENGINES)
def test_calibration_offsets(engine):
    ds = _constant()
    out = perturb_radar(
        ds, members=4000, seed=1, noise=False, calibration=(1.0, 0.2), engine=engine
    )
    z = (out.DBZH - 30.0).values.ravel()
    zdr = (out.ZDR - 0.5).values.ravel()
    assert z.std() == pytest.approx(1.0, rel=0.04)
    assert zdr.std() == pytest.approx(0.2, rel=0.04)
    assert abs(z.mean()) < 1e-9 and abs(zdr.mean()) < 1e-9  # antithetic pairs
    assert np.allclose(z[0::2], -z[1::2]) and np.allclose(zdr[0::2], -zdr[1::2])
    assert abs(np.corrcoef(z[::2], zdr[::2])[0, 1]) < 0.05
    odd = perturb_radar(
        ds, members=5, seed=1, noise=False, calibration=(1.0, 0.2), engine=engine
    )
    assert np.allclose(odd.DBZH.values[:4], out.DBZH.values[:4])
    # the offset is the same at all gates of a sweep
    wide = perturb_radar(_sweep(nray=3, ng=20), members=3, noise=False, engine=engine)
    off = (wide.DBZH - _sweep(nray=3, ng=20).DBZH).values.reshape(3, -1)
    assert np.allclose(off, off[:, :1])
    # no offsets and no noise return the input
    same = perturb_radar(ds, members=3, noise=False, calibration=None, engine=engine)
    for name in same.data_vars:
        assert np.array_equal(same[name].values[0], ds[name].values)


@compiled_only
def test_engines_agree():
    ds = _sweep()
    kw = dict(members=5, seed=9, snr="SNR")
    a = perturb_radar(ds, engine="numpy", **kw)
    b = perturb_radar(ds, engine="compiled", **kw)
    for name in a.data_vars:
        np.testing.assert_allclose(
            a[name].values, b[name].values, rtol=1e-12, atol=1e-12
        )
    # the number of threads does not change the numbers
    one = perturb_radar(ds, engine="compiled", n_threads=1, **kw)
    many = perturb_radar(ds, engine="compiled", n_threads=4, **kw)
    for name in a.data_vars:
        assert np.array_equal(one[name].values, many[name].values)


def test_members_do_not_depend_on_the_ensemble_size():
    ds = _sweep()
    small = perturb_radar(ds, members=3, seed=4)
    large = perturb_radar(ds, members=6, seed=4)
    for name in small.data_vars:
        assert np.array_equal(small[name].values, large[name].values[:3])
    other = perturb_radar(ds, members=3, seed=5)
    assert not np.array_equal(small.DBZH.values, other.DBZH.values)
    assert not np.array_equal(small.DBZH.values[0], small.DBZH.values[1])


def test_perturb_keeps_structure_and_the_input():
    ds = _sweep()
    before = ds.copy(deep=True)
    out = perturb_radar(ds, members=4, seed=2)
    xr.testing.assert_identical(ds, before)
    assert set(out.data_vars) == {"DBZH", "ZDR", "PHIDP", "RHOHV", "VRADH"}
    assert out.DBZH.dims == ("member", "azimuth", "range")
    assert out.sizes["member"] == 4
    assert out.DBZH.attrs["units"] == "dBZ"
    xr.testing.assert_equal(out["range"], ds["range"])
    assert out.RHOHV.min() >= 0 and out.RHOHV.max() <= 1
    assert out.DBZH.dtype == ds.DBZH.dtype
    # float32 fields stay float32, integer fields become float32
    f32 = ds.assign(DBZH=ds.DBZH.astype(np.float32))
    assert perturb_radar(f32, members=2).DBZH.dtype == np.float32
    const = _constant()
    ints = const.assign(DBZH=const.DBZH.astype(np.int32))
    assert perturb_radar(ints, members=2).DBZH.dtype == np.float32


def test_missing_values_stay_missing():
    ds = _sweep(nray=4, ng=30)
    ds["DBZH"] = ds["DBZH"].where(ds["range"] > 2000.0)
    out = perturb_radar(ds, members=3, seed=1)
    assert np.isnan(out.DBZH.values[:, :, :8]).all()
    assert np.isfinite(out.DBZH.values[:, :, 12:]).all()


def test_field_names_and_dataarray():
    ds = _sweep(nray=3, ng=20)
    renamed = ds.rename(DBZH="refl", ZDR="zdr_x")
    out = perturb_radar(renamed, dbz="refl", zdr="zdr_x", members=2)
    assert {"refl", "zdr_x"} <= set(out.data_vars)
    da = perturb_radar(ds["DBZH"], members=3, seed=1)
    assert isinstance(da, xr.DataArray) and da.dims[0] == "member" and da.name == "DBZH"
    single = perturb_radar(ds[["DBZH"]], members=3, seed=1)
    assert np.array_equal(single.DBZH.values, da.values)
    with pytest.raises(KeyError):
        perturb_radar(ds[["TEMP"]], members=2)
    with pytest.raises(KeyError):
        perturb_radar(ds, dbz="missing", members=2)


def test_perturb_volume():
    dtree = _volume()
    out = perturb_radar(dtree, members=3)
    assert list(out.children) == ["sweep_0", "sweep_1"]
    assert out["sweep_0"]["DBZH"].dims[0] == "member"
    # every sweep has its own numbers
    a = out["sweep_0"]["DBZH"].values - dtree["sweep_0"]["DBZH"].values
    b = perturb_radar(dtree["sweep_0"].to_dataset(), members=3)["DBZH"].values
    assert np.array_equal(out["sweep_0"]["DBZH"].values, b)
    c = out["sweep_1"]["DBZH"].values - dtree["sweep_1"]["DBZH"].values
    assert a.shape != c.shape
    with pytest.raises(KeyError):
        perturb_radar(
            xr.DataTree.from_dict({"sweep_0": xr.Dataset({"X": (("a",), [1.0])})})
        )


def test_perturb_argument_errors():
    ds = _constant()
    with pytest.raises(ValueError, match="members"):
        perturb_radar(ds, members=1)
    with pytest.raises(ValueError, match="calibration"):
        perturb_radar(ds, calibration=(1.0,))
    with pytest.raises(ValueError, match="calibration"):
        perturb_radar(ds, calibration=(-1.0, 0.1))
    with pytest.raises(ValueError, match="seed"):
        perturb_radar(ds, seed=-3)
    with pytest.raises(ValueError, match="engine"):
        perturb_radar(ds, engine="nope")


@pytest.mark.skipif(ensemble.HAS_COMPILED_KERNEL, reason="the kernel is built")
def test_compiled_engine_without_kernel():
    with pytest.raises(ImportError, match="compiled"):
        perturb_radar(_constant(), engine="compiled")


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def _members(n=40, shape=(30,), seed=0, nan=0.1):
    rng = np.random.default_rng(seed)
    x = rng.normal(5.0, 2.0, (n,) + shape) * rng.uniform(0.5, 2.0, shape)
    x[rng.random(x.shape) < nan] = np.nan
    x[:, 0] = np.nan  # no member at one gate
    x[:30, 1] = np.nan  # a few members at another
    return x


@pytest.mark.parametrize("engine", ENGINES)
def test_statistics_equal_numpy(engine):
    x = _members()
    da = xr.DataArray(x.astype(np.float32), dims=("member", "x"), name="V")
    out = ensemble_statistics(da, percentiles=(5, 25, 95), engine=engine)
    xf = x.astype(np.float32).astype(np.float64)
    ok = np.isfinite(xf).sum(axis=0)
    keep = ok >= 20
    with np.errstate(all="ignore"):
        assert np.allclose(
            out.V_mean.values[keep], np.nanmean(xf, axis=0)[keep], rtol=2e-6
        )
        assert np.allclose(
            out.V_std.values[keep], np.nanstd(xf, axis=0, ddof=1)[keep], rtol=2e-6
        )
        for name, q in (("median", 50), ("p05", 5), ("p25", 25), ("p95", 95)):
            expect = np.nanpercentile(xf, q, axis=0)
            assert np.allclose(out[f"V_{name}"].values[keep], expect[keep], rtol=2e-6)
    # fewer than half of the members: NaN
    assert np.isnan(out.V_mean.values[~keep]).all()
    assert np.isnan(out.V_p95.values[~keep]).all()
    assert out.V_mean.dtype == np.float32
    assert out.V_mean.attrs["ensemble_members"] == 40
    assert "member" not in out.dims


@compiled_only
def test_statistics_engines_agree():
    x = _members(n=64, shape=(500,), seed=2)
    a = kernels.statistics(x, [50.0, 5.0, 95.0], 5, False, None)
    b = kernels.statistics(x, [50.0, 5.0, 95.0], 5, True, 3)
    for u, v in zip(a, b):
        np.testing.assert_allclose(u, v, rtol=1e-12, atol=0, equal_nan=True)


def test_statistics_special_cases():
    x = np.array(
        [[1.0, 2.0, np.nan], [3.0, np.inf, np.nan], [5.0, 4.0, 7.0]], dtype=np.float32
    )
    mean, std, pct = kernels.statistics(x, [50.0, 100.0 - 1e-9], 1, False, None)
    assert mean[0] == pytest.approx(3.0) and std[0] == pytest.approx(2.0)
    assert mean[1] == pytest.approx(3.0)  # the infinite member is left out
    assert mean[2] == 7.0 and np.isnan(std[2])  # one member: no spread
    assert pct[0, 0] == 3.0
    # no member at all
    mean, std, pct = kernels.statistics(np.full((3, 2), np.nan), [50.0], 1, False, None)
    assert np.isnan(mean).all() and np.isnan(pct).all()
    with pytest.raises(ValueError, match="percentiles"):
        ensemble_statistics(
            xr.DataArray(x, dims=("member", "x"), name="V"), percentiles=(0, 50)
        )
    with pytest.raises(ValueError, match="min_fraction"):
        ensemble_statistics(
            xr.DataArray(x, dims=("member", "x"), name="V"), min_fraction=1.5
        )


@pytest.mark.parametrize("engine", ENGINES)
def test_mode_and_agreement(engine):
    codes = np.array(
        [
            [1, 2, 3, np.nan, 0, 4],
            [1, 2, 4, np.nan, 0, 4],
            [1, 3, 4, np.nan, 5, 4],
            [2, 3, 3, np.nan, 5, np.nan],
            [1, 3, 3, np.nan, 5, 4],
        ]
    )
    da = xr.DataArray(
        codes, dims=("member", "x"), name="HID", attrs={"flag_values": [0, 1]}
    )
    out = ensemble_statistics(da, categorical=["HID"], engine=engine, min_fraction=0.0)
    assert set(out.data_vars) == {"HID_mode", "HID_agreement"}
    mode = out.HID_mode.values
    share = out.HID_agreement.values
    np.testing.assert_array_equal(mode[[0, 1, 2, 4, 5]], [1, 3, 3, 5, 4])
    assert np.isnan(mode[3])
    np.testing.assert_allclose(share[[0, 1, 2, 4, 5]], [0.8, 0.6, 0.6, 0.6, 1.0])
    # ties go to the lowest code
    tie = xr.DataArray(np.array([[2.0], [3.0]]), dims=("member", "x"), name="C")
    assert (
        ensemble_statistics(tie, categorical=["C"], engine=engine).C_mode.values[0] == 2
    )
    assert out.HID_mode.attrs["flag_values"] == [0, 1]
    # not enough members
    strict = ensemble_statistics(
        da, categorical=["HID"], engine=engine, min_fraction=0.9
    )
    assert np.isnan(strict.HID_mode.values[5]) and not np.isnan(
        strict.HID_mode.values[0]
    )
    # negative and huge codes are not classes
    odd = xr.DataArray(
        np.array([[-3.0, 1e6], [-1.0, 1e6]]), dims=("member", "x"), name="C"
    )
    res = ensemble_statistics(odd, categorical=["C"], engine=engine, min_fraction=0.0)
    assert np.isnan(res.C_mode.values).any() or res.C_mode.size == 2


@compiled_only
def test_mode_engines_agree():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 12, (33, 400)).astype(np.float32)
    x[rng.random(x.shape) < 0.2] = np.nan
    a = kernels.mode(x, 3, False, None)
    b = kernels.mode(x, 3, True, 3)
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u, v)


def test_statistics_containers():
    ds = _sweep(nray=3, ng=10)
    ens = perturb_radar(ds, members=6)
    stats = ensemble_statistics(ens)
    assert {"DBZH_mean", "DBZH_std", "ZDR_median", "VRADH_p95"} <= set(stats.data_vars)
    assert stats.DBZH_std.dims == ("azimuth", "range")
    assert (stats.DBZH_std >= 0).all()
    # variables without the dimension are skipped, and a Dataset without any is an error
    mixed = ens.assign(extra=ds["TEMP"])
    assert "extra_mean" not in ensemble_statistics(mixed)
    with pytest.raises(KeyError):
        ensemble_statistics(ds)
    with pytest.raises(KeyError):
        ensemble_statistics(ens, dim="other")
    # volumes
    tree = perturb_radar(_volume(), members=4)
    out = ensemble_statistics(tree)
    assert list(out.children) == ["sweep_0", "sweep_1"]
    assert "DBZH_std" in out["sweep_0"].data_vars
    # a named DataArray and the dimension name
    da = ens["DBZH"].rename(member="k")
    assert "DBZH_mean" in ensemble_statistics(da, dim="k")
    # the percentile names
    names = ensemble_statistics(ens["DBZH"], percentiles=(2.5, 90)).data_vars
    assert {"DBZH_p2.5", "DBZH_p90"} <= set(names)
    # accessors
    xr.testing.assert_identical(
        without_provenance(ens.radarx.ensemble_statistics()), without_provenance(stats)
    )
    assert "DBZH_mean" in ens["DBZH"].radarx.ensemble_statistics()


# --------------------------------------------------------------------------
# relation_scatter
# --------------------------------------------------------------------------


def test_relation_scatter():
    sc = relation_scatter("S")
    assert list(sc.relation.values) == ["z", "kdp", "z_zdr", "a"]
    ln_sd = sc.ln_sd.to_series()
    # R(Z, ZDR) and R(A) are less sensitive to the DSD than R(Z) and R(KDP)
    assert ln_sd["z_zdr"] < ln_sd["z"] and ln_sd["a"] < ln_sd["z"]
    assert ln_sd["kdp"] < ln_sd["z"]
    assert 0.3 < ln_sd["z"] < 0.7 and 0.05 < ln_sd["z_zdr"] < 0.2
    assert sc.exponent.sel(relation="z") == pytest.approx(0.855, abs=0.03)
    assert np.isnan(sc.zdr_exponent.sel(relation="kdp"))
    assert sc.zdr_exponent.sel(relation="z_zdr") < -3
    assert 0.01 < float(sc.alpha) < 0.02 and 0.1 < float(sc.alpha_ln_sd) < 0.4
    # a prior given as a Dataset, the same numbers
    from radarx.retrieve import dsd_prior

    again = relation_scatter("S", prior=dsd_prior("generic"))
    xr.testing.assert_allclose(again.ln_sd, sc.ln_sd)
    other = relation_scatter("C", 10.0, prior="perils2022")
    assert other.attrs["band"] == "C" and not np.allclose(other.ln_sd, sc.ln_sd)
    with pytest.raises(ValueError, match="band"):
        relation_scatter("L")


def test_relation_scatter_equals_a_direct_fit():
    """The fit by weighted least squares on a hand-made sample."""
    rng = np.random.default_rng(0)
    x = np.exp(rng.normal(0, 1, 5000))
    y = 3.0 * x**0.8 * np.exp(rng.normal(0, 0.25, x.size))
    sd, beta = ensemble._weighted_fit(y, [x], np.ones_like(x))
    assert sd == pytest.approx(0.25, rel=0.05)
    assert beta[1] == pytest.approx(0.8, abs=0.02)
    assert np.exp(beta[0]) == pytest.approx(3.0, rel=0.05)


# --------------------------------------------------------------------------
# retrievals
# --------------------------------------------------------------------------

ZERO = dict(calibration=None, noise=False, prior=False)


def _direct(name, ds, **kw):
    if name == "kdp":
        return estimate_kdp(ds)["KDP"]
    if name == "attenuation":
        return correct_attenuation(ds, band="S", **kw)
    if name == "rain_rate_z":
        return rain_rate_z(ds)
    if name == "rain_rate_kdp":
        return rain_rate_kdp(ds.assign(KDP=estimate_kdp(ds)["KDP"]))
    if name == "rain_rate_z_zdr":
        return rain_rate_z_zdr(ds)
    if name == "rain_rate_a":
        return rain_rate_a(ds.assign(AH=correct_attenuation(ds, band="S")["AH"]))
    if name == "dsd":
        return dsd(ds)
    return dsd_bayesian(ds)


@pytest.mark.parametrize(
    "name,variable",
    [
        ("kdp", "KDP"),
        ("rain_rate_z", "RAIN_RATE_Z"),
        ("rain_rate_kdp", "RAIN_RATE_KDP"),
        ("rain_rate_z_zdr", "RAIN_RATE_Z_ZDR"),
        ("rain_rate_a", "RAIN_RATE_A"),
        ("dsd", "DM"),
        ("dsd_bayesian", "DM"),
    ],
)
def test_without_perturbation_the_ensemble_is_the_retrieval(name, variable):
    ds = _rain_ds()
    out = ensemble_retrieval(ds, name, members=3, band="S", **ZERO)
    direct = _direct(name, ds)
    direct = direct[variable] if isinstance(direct, xr.Dataset) else direct
    np.testing.assert_allclose(
        out[f"{variable}_mean"].values, direct.values, rtol=1e-6, equal_nan=True
    )
    std = out[f"{variable}_std"].values
    assert np.nanmax(std) == 0.0
    assert out[f"{variable}_mean"].dims == direct.dims
    assert out.attrs["ensemble_members"] == 3


def test_attenuation_chain_and_its_prior():
    ds = _rain_ds()
    out = ensemble_retrieval(ds, "attenuation", members=3, band="C", **ZERO)
    direct = correct_attenuation(ds, band="C")
    for name in ("DBZH_corrected", "ZDR_corrected", "AH"):
        np.testing.assert_allclose(
            out[f"{name}_mean"].values, direct[name].values, rtol=1e-6, equal_nan=True
        )
    assert float(out.AH_std.max()) == 0.0
    # the coefficients vary with the prior: the specific attenuation scales with alpha
    varied = ensemble_retrieval(
        ds,
        "attenuation",
        members=60,
        band="C",
        calibration=None,
        noise=False,
        keep_members=True,
        attenuation={"method": "linear"},
    )
    ratio = (varied.AH_members / direct.AH).where(direct.AH > 0.05)
    log_sd = float(np.log(ratio).std("member").mean())
    expected = float(relation_scatter("C").alpha_ln_sd)
    assert log_sd == pytest.approx(expected, rel=0.35)
    given = ensemble_retrieval(
        ds,
        "attenuation",
        members=4,
        band="C",
        calibration=None,
        noise=False,
        prior={"alpha": 0.3},
        attenuation={"alpha": 0.1, "beta": 0.02, "b": 0.8, "method": "zphi"},
    )
    assert float(given.AH_std.max()) > 0
    # a ZDR-free sweep gives no corrected ZDR
    no_zdr = ensemble_retrieval(
        ds.drop_vars("ZDR"), "attenuation", members=3, band="S", **ZERO
    )
    assert "ZDR_corrected_mean" not in no_zdr


def test_rain_rate_with_attenuation_and_prior():
    ds = _rain_ds()
    plain = ensemble_retrieval(ds, "rain_rate_z", members=3, band="X", **ZERO)
    att = ensemble_retrieval(
        ds, "rain_rate_z", members=3, band="X", attenuation="zphi", **ZERO
    )
    corrected = correct_attenuation(ds, band="X")
    direct = rain_rate_z(ds.assign(DBZH=corrected["DBZH_corrected"]), band="X")
    np.testing.assert_allclose(
        att.RAIN_RATE_Z_mean.values, direct.values, rtol=1e-6, equal_nan=True
    )
    assert float(att.RAIN_RATE_Z_mean.mean()) > float(plain.RAIN_RATE_Z_mean.mean())
    # the attenuation correction also feeds the other chains
    for method in ("linear", True, {"method": "zphi", "min_dphi": 1.0}):
        out = ensemble_retrieval(
            ds, "dsd", members=2, band="S", attenuation=method, **ZERO
        )
        assert "DM_mean" in out
    with pytest.raises(TypeError, match="attenuation"):
        ensemble_retrieval(ds, "rain_rate_z", attenuation=3)
    # the rain rate prefactor: ln R varies with the sd of relation_scatter
    out = ensemble_retrieval(
        ds,
        "rain_rate_z",
        members=1000,
        seed=2,
        calibration=None,
        noise=False,
        keep_members=True,
    )
    ratio = np.log(
        out.RAIN_RATE_Z_members / out.RAIN_RATE_Z_mean.where(out.RAIN_RATE_Z_mean > 0)
    )
    expected = float(relation_scatter("S").ln_sd.sel(relation="z"))
    assert float(ratio.std("member").median()) == pytest.approx(expected, rel=0.08)
    # the common change is the same at all gates of a member
    assert float((ratio.std("range") / ratio.std("member").mean()).max()) < 1e-3
    custom = ensemble_retrieval(
        ds,
        "rain_rate_z",
        members=1000,
        seed=2,
        calibration=None,
        noise=False,
        prior={"relation": 0.2},
        keep_members=True,
    )
    ratio = np.log(
        custom.RAIN_RATE_Z_members
        / custom.RAIN_RATE_Z_mean.where(custom.RAIN_RATE_Z_mean > 0)
    )
    assert float(ratio.std("member").median()) == pytest.approx(0.2, rel=0.1)


def test_calibration_only_ensemble_follows_the_power_law():
    ds = _rain_ds()
    out = ensemble_retrieval(
        ds,
        "rain_rate_z",
        members=600,
        seed=3,
        calibration=(1.0, 0.1),
        noise=False,
        prior=False,
        keep_members=True,
    )
    base = rain_rate_z(ds)
    ratio = np.log(out.RAIN_RATE_Z_members / base).where(base > 1.0)
    # b ln(10)/10 per dB with the exponent 0.714 of the WSR-88D relation
    expected = 0.714 * math.log(10) / 10 * 1.0
    assert float(ratio.std("member").median()) == pytest.approx(expected, rel=0.08)
    # R(Z, ZDR) feels both offsets
    both = ensemble_retrieval(
        ds,
        "rain_rate_z_zdr",
        members=300,
        seed=3,
        calibration=(1.0, 0.1),
        noise=False,
        prior=False,
        keep_members=True,
    )
    z_only = ensemble_retrieval(
        ds,
        "rain_rate_z_zdr",
        members=300,
        seed=3,
        calibration=(1.0, 0.0),
        noise=False,
        prior=False,
        keep_members=True,
    )
    assert float(both.RAIN_RATE_Z_ZDR_std.mean()) > float(
        z_only.RAIN_RATE_Z_ZDR_std.mean()
    )
    # R(A) does not depend on the reflectivity calibration but its coefficients do
    a_only = ensemble_retrieval(
        ds,
        "rain_rate_a",
        members=20,
        seed=3,
        calibration=(5.0, 0.0),
        noise=False,
        prior=False,
    )
    assert float(a_only.RAIN_RATE_A_std.max()) < 0.05 * float(
        a_only.RAIN_RATE_A_mean.max()
    )


def test_matches_the_public_pieces():
    """The ensemble is perturb_radar, the retrieval on each member, and ensemble_statistics."""
    ds = _rain_ds()
    kw = dict(
        members=6, seed=8, noise={"snr": "SNR"}, calibration=(1.0, 0.1), prior=False
    )
    out = ensemble_retrieval(ds, "rain_rate_z", **kw)
    members = perturb_radar(ds, members=6, seed=8, snr="SNR")
    rates = xr.concat(
        [rain_rate_z(members.isel(member=m, drop=True)) for m in range(6)], dim="member"
    )
    manual = ensemble_statistics(rates.rename("RAIN_RATE_Z"))
    for name in ("mean", "std", "median", "p05", "p95"):
        np.testing.assert_allclose(
            out[f"RAIN_RATE_Z_{name}"].values,
            manual[f"RAIN_RATE_Z_{name}"].values,
            rtol=1e-5,
            equal_nan=True,
        )


@pytest.mark.parametrize("engine", ENGINES)
def test_hid_ensemble(engine):
    ds = _rain_ds()
    out = ensemble_retrieval(
        ds,
        "hid",
        members=7,
        seed=1,
        engine=engine,
        noise={"snr": "SNR"},
        options={"temperature": "TEMP", "band": "S"},
    )
    assert set(out.data_vars) == {"HID_mode", "HID_agreement"}
    assert float(out.HID_agreement.min()) >= 1 / 7 - 1e-6
    assert float(out.HID_agreement.max()) <= 1.0
    direct = hid(ds, "TEMP", band="S")
    zero = ensemble_retrieval(
        ds, "hid", members=3, options={"temperature": "TEMP", "band": "S"}, **ZERO
    )
    np.testing.assert_array_equal(zero.HID_mode.values, direct.HID.values)
    assert float(zero.HID_agreement.min()) == 1.0
    assert "flag_meanings" in out.HID_mode.attrs


def test_dsd_and_bayesian_priors():
    ds = _rain_ds()
    base = ensemble_retrieval(ds, "dsd", members=2, band="S", **ZERO)
    shifted = ensemble_retrieval(
        ds,
        "dsd",
        members=40,
        band="S",
        calibration=None,
        noise=False,
        prior={"mu": 1.0},
        keep_members=True,
    )
    wide = ensemble_retrieval(
        ds,
        "dsd",
        members=40,
        band="S",
        calibration=None,
        noise=False,
        prior={"mu": 2.0},
        keep_members=True,
    )
    assert 0.5 < float(shifted.MU_members.std("member").median()) < 3.0
    assert float(wide.MU_members.std("member").median()) > float(
        shifted.MU_members.std("member").median()
    )
    assert float(shifted.MU_mean.mean()) == pytest.approx(
        float(base.MU_mean.mean()), abs=1.0
    )
    # the default width is that of mu under the generic prior
    default = ensemble_retrieval(
        ds,
        "dsd",
        members=40,
        band="S",
        calibration=None,
        noise=False,
        keep_members=True,
    )
    assert 1.5 < float(default.MU_members.std("member").median()) < 4.5
    # the normalized method shifts mu itself
    norm = ensemble_retrieval(
        ds,
        "dsd",
        members=30,
        band="S",
        calibration=None,
        noise=False,
        prior={"mu": 0.8},
        options={"method": "normalized", "mu": 2.0},
        keep_members=True,
    )
    assert float(norm.MU_members.mean("member").mean()) == pytest.approx(2.0, abs=0.5)
    assert float(norm.MU_members.min()) >= -0.9
    custom = ensemble_retrieval(
        ds,
        "dsd",
        members=4,
        band="S",
        calibration=None,
        noise=False,
        options={"mu_lambda": (-0.0201, 0.902, -1.718)},
    )
    assert float(custom.MU_std.mean()) > 0
    # the Bayesian prior: tilt and shift change the posterior
    bayes = ensemble_retrieval(
        ds,
        "dsd_bayesian",
        members=6,
        band="S",
        calibration=None,
        noise=False,
        prior={"tilt": 1.0, "log10_nw": 0.5},
    )
    assert float(bayes.DM_std.mean()) > 0
    corrected = ensemble_retrieval(
        ds, "dsd_bayesian", members=3, band="S", attenuation="zphi", **ZERO
    )
    assert np.isfinite(corrected.DM_mean).any()
    quiet = ensemble_retrieval(
        ds,
        "dsd_bayesian",
        members=4,
        band="S",
        calibration=None,
        noise=False,
        prior=False,
    )
    assert float(quiet.DM_std.max()) == 0.0
    from radarx.retrieve import dsd_prior

    named = ensemble_retrieval(
        ds,
        "dsd_bayesian",
        members=3,
        band="S",
        calibration=None,
        noise=False,
        options={"prior": dsd_prior("perils2022")},
        prior={"tilt": 0.5, "log10_nw": 0.0},
    )
    assert float(named.DM_std.mean()) > 0


def test_bayesian_total_spread_combines_posterior_and_ensemble():
    ds = _rain_ds()
    kw = dict(members=8, seed=3, band="S", noise={"snr": "SNR"})
    default = ensemble_retrieval(ds, "dsd_bayesian", **kw)
    assert "DM_SD_mean" not in default and "DM_total_std" in default
    raw = ensemble_retrieval(
        ds, "dsd_bayesian", variables=["DM", "DM_SD"], keep_members=True, **kw
    )
    assert "DM_total_std" not in raw and "DM_SD_mean" in raw
    expected = np.sqrt(
        (raw.DM_SD_members**2).mean("member")
        + raw.DM_members.std("member", ddof=1) ** 2
    )
    np.testing.assert_allclose(
        default.DM_total_std.values, expected.values, rtol=1e-4, equal_nan=True
    )
    assert (default.DM_total_std >= default.DM_std - 1e-6).all()


def test_tilted_prior_moves_the_mean():
    from radarx.retrieve import dsd_prior

    prior = dsd_prior("generic").transpose("dm", "mu")
    draws = {"tilt_dm": 1.0, "tilt_mu": 0.0, "nw": 0.2}
    tilted = ensemble._tilted_prior(prior, draws)
    assert float(tilted.prior_mass.sum()) == pytest.approx(1.0)
    marginal = prior.prior_mass.sum("mu")
    mean = float((marginal * prior.dm).sum())
    sd = float(np.sqrt((marginal * (prior.dm - mean) ** 2).sum()))
    new = tilted.prior_mass.sum("mu")
    # an exponential tilt of one standard deviation moves the mean by a fraction of it
    assert float((new * prior.dm).sum()) > mean + 0.2 * sd
    np.testing.assert_allclose(tilted.log10_nw_mean, prior.log10_nw_mean + 0.2)
    xr.testing.assert_identical(prior, dsd_prior("generic").transpose("dm", "mu"))


def test_options_variables_and_callables():
    ds = _rain_ds()
    out = ensemble_retrieval(
        ds, "kdp", members=3, options={"method": "vulpiani", "kdp_options": {}}, **ZERO
    )
    expected = estimate_kdp(ds, method="vulpiani")["KDP"]
    np.testing.assert_allclose(
        out.KDP_mean.values, expected.values, rtol=1e-6, equal_nan=True
    )
    # the keyword arguments of the phase processing reach the KDP step
    a = ensemble_retrieval(
        ds,
        "rain_rate_kdp",
        members=2,
        options={"kdp_options": {"method": "monotone"}},
        **ZERO,
    )
    b = ensemble_retrieval(ds, "rain_rate_kdp", members=2, **ZERO)
    assert not np.allclose(a.RAIN_RATE_KDP_mean, b.RAIN_RATE_KDP_mean, equal_nan=True)
    # without a phase the KDP of the input is used as it is
    given = ds.assign(KDP=estimate_kdp(ds)["KDP"]).drop_vars("PHIDP")
    fixed = ensemble_retrieval(given, "rain_rate_kdp", members=3, **ZERO)
    np.testing.assert_allclose(
        fixed.RAIN_RATE_KDP_mean.values,
        rain_rate_kdp(given).values,
        rtol=1e-6,
        equal_nan=True,
    )
    # variables
    sel = ensemble_retrieval(
        ds, "dsd", members=3, variables=["NW", "D0"], band="S", **ZERO
    )
    assert {"NW_mean", "D0_mean"} == {v for v in sel.data_vars if v.endswith("_mean")}
    with pytest.raises(KeyError):
        ensemble_retrieval(ds, "dsd", members=3, variables=["nope"], band="S", **ZERO)
    # a function that gets the sweep with the standard names
    seen = {}

    def fraction(sweep, scale=1.0):
        seen["names"] = set(sweep.data_vars)
        return (sweep["DBZH"] * scale).rename("scaled")

    got = ensemble_retrieval(ds, fraction, members=4, options={"scale": 2.0}, **ZERO)
    assert {"DBZH", "ZDR", "PHIDP", "RHOHV", "VRADH", "TEMP", "SNR"} <= seen["names"]
    np.testing.assert_allclose(got.scaled_mean.values, 2.0 * ds.DBZH.values)
    # a function returning a Dataset, and the members
    both = ensemble_retrieval(
        ds, lambda s: s[["DBZH", "ZDR"]], members=3, keep_members=True, band="S", **ZERO
    )
    assert both.DBZH_members.dims == ("member", "azimuth", "range")
    assert both.ZDR_members.shape[0] == 3
    # the field names of the input are not left in the member
    renamed = ds.rename(DBZH="my_dbz", ZDR="my_zdr")
    again = ensemble_retrieval(
        renamed, "rain_rate_z_zdr", dbz="my_dbz", zdr="my_zdr", members=2, **ZERO
    )
    np.testing.assert_allclose(
        again.RAIN_RATE_Z_ZDR_mean.values,
        rain_rate_z_zdr(ds).values,
        rtol=1e-6,
        equal_nan=True,
    )


def test_min_fraction_and_attributes():
    ds = _rain_ds()
    out = ensemble_retrieval(ds, "dsd", members=8, seed=4, band="S", min_fraction=1.0)
    assert out.attrs["ensemble_seed"] == 4
    assert (
        "calibration offsets SD 1 dB (Z), 0.1 dB (ZDR)" in out.attrs["ensemble_method"]
    )
    assert "measurement noise of 64 samples" in out.attrs["ensemble_method"]
    assert "prior SD mu" in out.attrs["ensemble_method"]
    quiet = ensemble_retrieval(ds, "dsd", members=3, band="S", **ZERO)
    assert quiet.attrs["ensemble_method"] == "3 members, seed 0"
    assert out.DM_mean.attrs["units"] == "mm"
    assert "over the ensemble" in out.DM_mean.attrs["long_name"]
    loose = ensemble_retrieval(ds, "dsd", members=8, seed=4, band="S", min_fraction=0.0)
    assert int(np.isfinite(loose.DM_mean).sum()) >= int(np.isfinite(out.DM_mean).sum())
    for field in ("DM_median", "DM_p05", "DM_p95"):
        assert field in loose


def test_ensemble_volume_dataarray_and_accessors():
    dtree = _volume()
    out = ensemble_retrieval(dtree, "rain_rate_z", members=4, seed=2)
    assert list(out.children) == ["sweep_0", "sweep_1"]
    single = ensemble_retrieval(
        dtree["sweep_1"].to_dataset(), "rain_rate_z", members=4, seed=2
    )
    # the second sweep has its own random numbers, so it differs from a single sweep
    assert not np.allclose(
        out["sweep_1"]["RAIN_RATE_Z_mean"].values,
        single["RAIN_RATE_Z_mean"].values,
        equal_nan=True,
    )
    first = ensemble_retrieval(
        dtree["sweep_0"].to_dataset(), "rain_rate_z", members=4, seed=2
    )
    np.testing.assert_allclose(
        out["sweep_0"]["RAIN_RATE_Z_mean"].values,
        first["RAIN_RATE_Z_mean"].values,
        equal_nan=True,
    )
    # a named DataArray
    da = ensemble_retrieval(
        dtree["sweep_0"].to_dataset()["DBZH"], "rain_rate_z", members=4, seed=2
    )
    only = ensemble_retrieval(
        dtree["sweep_0"].to_dataset()[["DBZH"]], "rain_rate_z", members=4, seed=2
    )
    xr.testing.assert_allclose(da.RAIN_RATE_Z_mean, only.RAIN_RATE_Z_mean)
    # accessors
    ds = _rain_ds()
    via = ds.radarx.ensemble_retrieval("rain_rate_z", members=3, seed=1)
    xr.testing.assert_identical(
        without_provenance(via),
        without_provenance(ensemble_retrieval(ds, "rain_rate_z", members=3, seed=1)),
    )
    err = ds.radarx.measurement_error()
    xr.testing.assert_identical(
        without_provenance(err), without_provenance(measurement_error(ds))
    )
    ens = ds.radarx.perturb_radar(members=2)
    assert ens.sizes["member"] == 2
    assert ds["DBZH"].radarx.perturb_radar(members=2).dims[0] == "member"
    assert "DBZH_error" in ds["DBZH"].radarx.measurement_error()
    assert (
        "RAIN_RATE_Z_std"
        in dtree.radarx.ensemble_retrieval("rain_rate_z", members=2)["sweep_0"]
    )


def test_ensemble_argument_errors():
    ds = _rain_ds()
    with pytest.raises(ValueError, match="retrieval"):
        ensemble_retrieval(ds, "nope")
    with pytest.raises(ValueError, match="members"):
        ensemble_retrieval(ds, "rain_rate_z", members=1)
    with pytest.raises(TypeError, match="prior"):
        ensemble_retrieval(ds, "rain_rate_z", prior="yes")
    with pytest.raises(ValueError, match="unknown prior"):
        ensemble_retrieval(ds, "rain_rate_z", prior={"nope": 1.0})
    with pytest.raises(ValueError, match="negative"):
        ensemble_retrieval(ds, "rain_rate_z", prior={"relation": -1.0})
    with pytest.raises(ValueError, match="percentiles"):
        ensemble_retrieval(ds, "rain_rate_z", percentiles=(100,))
    with pytest.raises(ValueError, match="min_fraction"):
        ensemble_retrieval(ds, "rain_rate_z", min_fraction=-0.1)
    with pytest.raises(KeyError):
        ensemble_retrieval(ds[["TEMP"]], "rain_rate_z")
    with pytest.raises(KeyError):
        ensemble_retrieval(
            xr.DataTree.from_dict({"sweep_0": xr.Dataset({"X": (("a",), [1.0])})}),
            "rain_rate_z",
        )


def test_band_resolution():
    ds = _rain_ds()
    run = ensemble._Run(band=None, name="rain_rate_z", att=None)
    assert run.band_for(ds) == "S"
    run = ensemble._Run(band="X", name="dsd", att=None)
    assert run.band_for(ds) == "X"
    run = ensemble._Run(band=None, name="dsd", att=None)
    with pytest.warns(UserWarning, match="band"):
        assert run.band_for(ds) == "S"


# --------------------------------------------------------------------------
# spread against the error on a case with known truth
# --------------------------------------------------------------------------


def _truth_and_observations(n_real):
    ds = _sweep(nray=4, ng=240, seed=5)
    truth_kdp = dsd_sweep(nray=4, ng=240, seed=5)[3]
    dr = 0.25
    phi = 2.0 * np.cumsum(truth_kdp) * dr + 40.0
    truth = ds.assign(
        PHIDP=(("azimuth", "range"), np.broadcast_to(phi, ds.PHIDP.shape).copy()),
        SNR=ds["SNR"] * 0.0 + 22.0,
    )
    obs = perturb_radar(truth, members=n_real, seed=1234, snr="SNR")
    return truth, [
        obs.isel(member=k, drop=True).assign(SNR=truth["SNR"]) for k in range(n_real)
    ]


def _reliability(truth, observed, retrieval, variable):
    """
    Scale of the normalized error of the ensemble median against the retrieval
    on the true fields (median of the absolute value over 0.6745; 1 for a
    Gaussian error with the stated spread).
    """
    ref = ensemble_retrieval(truth, retrieval, members=2, band="S", **ZERO)[
        f"{variable}_mean"
    ]
    z = []
    for k, ds in enumerate(observed):
        ens = ensemble_retrieval(
            ds,
            retrieval,
            members=24,
            seed=100 + k,
            band="S",
            noise={"snr": "SNR"},
            prior=False,
        )
        spread = ens[f"{variable}_std"].values
        good = (ref > 1e-3 if retrieval != "kdp" else ref > 0.05).values & (spread > 0)
        err = (ens[f"{variable}_median"].values - ref.values)[good]
        z.append(err / ens[f"{variable}_std"].values[good])
    return np.median(np.abs(np.concatenate(z))) / 0.6745


def test_spread_matches_the_actual_error():
    """The members scatter like the observations do around the truth (RMS of the normalized error near 1)."""
    truth, observed = _truth_and_observations(32)
    # the error of one realization against the spread of its ensemble
    assert 0.7 < _reliability(truth, observed, "rain_rate_z", "RAIN_RATE_Z") < 1.4
    assert (
        0.5 < _reliability(truth, observed, "rain_rate_z_zdr", "RAIN_RATE_Z_ZDR") < 1.4
    )
    assert 0.4 < _reliability(truth, observed, "kdp", "KDP") < 1.4


def test_spread_with_the_prior_covers_the_true_rain_rate():
    """With the DSD prior the 5-95 percent range holds the true rain rate in most gates."""
    ds, dm, nw, _ = dsd_sweep(nray=4, ng=240, seed=5)
    from radarx.retrieve import forward_grid

    fwd = forward_grid("S", 20.0)
    mu = 3.0
    rate = np.empty_like(dm)
    for i, (d, n) in enumerate(zip(dm, nw)):
        node = fwd.sel(dm=round(d / 0.05) * 0.05, mu=mu, method="nearest")
        rate[i] = float(node.RAIN_RATE) * n
    truth_rate = np.broadcast_to(rate, ds.DBZH.shape)
    out = ensemble_retrieval(
        ds, "rain_rate_z_zdr", members=48, seed=6, band="S", noise={"snr": 25.0}
    )
    inside = (truth_rate >= out.RAIN_RATE_Z_ZDR_p05.values) & (
        truth_rate <= out.RAIN_RATE_Z_ZDR_p95.values
    )
    good = truth_rate > 1.0
    assert inside[good].mean() > 0.6
