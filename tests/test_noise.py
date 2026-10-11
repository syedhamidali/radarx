#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the noise level, the SNR and the noise correction of RHOHV and ZDR
============================================================================

The truth is known: a sweep whose reflectivity is the sum of a signal and a
noise with a known level, and complex Gaussian voltages of two channels with a
known correlation, power ratio and noise, from which the measured moments are
formed the way a radar does. The compiled kernel and the NumPy implementation
must agree to rounding (rtol 1e-12).
"""
import importlib

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx._provenance import load_registry, read_provenance
from radarx.retrieve import (
    correct_rhohv_noise,
    correct_zdr_noise,
    noise_level,
    snr,
)

noisemod = importlib.import_module("radarx.retrieve.noise")

ENGINES = ["numpy"] + (["compiled"] if noisemod.HAS_COMPILED_KERNEL else [])
compiled_only = pytest.mark.skipif(
    not noisemod.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)
ZN = -40.0  # noise level, dBZ at 1 km


def synthetic_sweep(nray=90, ngate=700, seed=1, samples=64, noise=ZN, threshold=None):
    """A sweep with a rain patch and noise of a known level; the truth is ``noise``."""
    rng = np.random.default_rng(seed)
    rng_m = 2000.0 + 250.0 * np.arange(ngate)
    r_km = rng_m / 1000.0
    signal = np.full((nray, ngate), -np.inf)
    signal[20:60, 100:400] = 25.0 + 8.0 * rng.standard_normal((40, 300))
    p_sig = 10 ** (0.1 * (signal - 20 * np.log10(r_km)[None]))
    p_noise = 10 ** (0.1 * noise)
    power = p_sig * rng.gamma(samples, 1 / samples, (nray, ngate))
    power += p_noise * rng.gamma(samples, 1 / samples, (nray, ngate))
    dbz = 10 * np.log10(power) + 20 * np.log10(r_km)[None]
    if threshold is not None:
        snr_db = dbz - noise - 20 * np.log10(r_km)[None]
        dbz = np.where(snr_db >= threshold, dbz, np.nan)
    return xr.Dataset(
        {
            "DBZH": (
                ("azimuth", "range"),
                dbz.astype(np.float32),
                {
                    "units": "dBZ",
                    "standard_name": "radar_equivalent_reflectivity_factor_h",
                },
            )
        },
        coords={"azimuth": np.arange(nray) * 4.0, "range": rng_m},
    )


def weak_echo_sweep(nray=90, ngate=700, seed=4, noise=ZN, threshold=3.0):
    """Echo with an SNR spread from -5 to 25 dB everywhere, thresholded on SNR."""
    rng = np.random.default_rng(seed)
    rng_m = 2000.0 + 250.0 * np.arange(ngate)
    r_db = 20 * np.log10(rng_m / 1000.0)[None]
    snr_true = rng.uniform(-5.0, 25.0, (nray, ngate))
    p = 10 ** (0.1 * (noise + snr_true)) * rng.gamma(64, 1 / 64, (nray, ngate))
    p += 10 ** (0.1 * noise) * rng.gamma(64, 1 / 64, (nray, ngate))
    dbz = 10 * np.log10(p) + r_db
    snr_db = dbz - noise - r_db
    dbz = np.where(snr_db >= threshold, dbz, np.nan)
    return xr.Dataset(
        {"DBZH": (("azimuth", "range"), dbz.astype(np.float32))},
        coords={"azimuth": np.arange(nray) * 4.0, "range": rng_m},
    )


def volume(**kwargs):
    ds0 = synthetic_sweep(seed=1, **kwargs)
    ds1 = synthetic_sweep(seed=2, **kwargs)
    return xr.DataTree.from_dict({"/": xr.Dataset(), "sweep_0": ds0, "sweep_1": ds1})


# --------------------------------------------------------------------------
# SNR
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_snr_closed_form(engine):
    rng_m = np.array([1000.0, 2000.0, 10000.0, 100000.0])
    ds = xr.Dataset(
        {"DBZH": (("azimuth", "range"), np.array([[0.0, 10.0, 20.0, 40.0]] * 2))},
        coords={"azimuth": [0.0, 1.0], "range": rng_m},
    )
    out = snr(ds, noise=-30.0, engine=engine)
    expected = np.array([0.0, 10.0, 20.0, 40.0]) + 30.0 - 20 * np.log10(rng_m / 1000.0)
    np.testing.assert_allclose(out.values, np.tile(expected, (2, 1)), rtol=1e-6)
    assert out.attrs["units"] == "dB"
    assert out.attrs["standard_name"] == "signal_to_noise_ratio"
    assert out.dtype == np.float32
    assert out.dims == ("azimuth", "range")
    assert out.attrs["noise_level"] == -30.0


@pytest.mark.parametrize("engine", ENGINES)
def test_snr_with_noise_in_the_signal(engine):
    """A reflectivity that holds the noise power loses it first."""
    ds = xr.Dataset(
        {"DBZH": (("range",), np.array([-30.0, -20.0, 0.0]))},
        coords={"range": [1000.0, 1000.0 + 1e-3, 1000.0 + 2e-3]},
    )
    out = snr(ds, noise=-30.0, signal_includes_noise=True, engine=engine)
    lin = 10 ** (0.1 * np.array([0.0, 10.0, 30.0])) - 1.0
    assert np.isnan(out.values[0])  # no power above the noise
    np.testing.assert_allclose(out.values[1:], 10 * np.log10(lin[1:]), atol=1e-4)


def test_snr_per_ray_noise_and_estimated_noise():
    ds = synthetic_sweep()
    per_ray = xr.DataArray(
        np.where(np.arange(ds.sizes["azimuth"]) < 45, -40.0, -38.0),
        dims="azimuth",
    )
    out = snr(ds, noise=per_ray)
    base = snr(ds, noise=-40.0)
    np.testing.assert_allclose(
        (base - out).isel(azimuth=slice(45, None)).values, 2.0, atol=1e-4
    )
    est = snr(ds)  # noise estimated with Hildebrand and Sekhon
    assert abs(est.attrs["noise_level"] - ZN) < 0.5


# --------------------------------------------------------------------------
# noise level
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_hildebrand_sekhon_recovers_the_noise_level(engine):
    ds = synthetic_sweep()
    out = noise_level(ds, engine=engine)
    assert abs(float(out.NOISE_LEVEL) - ZN) < 0.3
    assert out.NOISE_LEVEL.attrs["units"] == "dBZ"
    # most of the gates are noise (the signal patch is 15 % of the sweep)
    assert int(out.NOISE_SAMPLES) > 0.7 * ds.DBZH.size
    assert out.attrs["noise_method"] == "hildebrand"


@pytest.mark.parametrize("level", [-55.0, -40.0, -25.0])
def test_noise_level_over_a_range_of_levels(level):
    out = noise_level(synthetic_sweep(noise=level))
    assert abs(float(out.NOISE_LEVEL) - level) < 0.3


def test_n_samples_matters():
    """Too few samples accept signal as noise, too many reject noise."""
    ds = synthetic_sweep()
    low = float(noise_level(ds, n_samples=1.0).NOISE_LEVEL)
    right = float(noise_level(ds, n_samples=64.0).NOISE_LEVEL)
    high = float(noise_level(ds, n_samples=256.0).NOISE_LEVEL)
    assert low > right > high
    # no set of gates is as steady as that: no estimate
    assert np.isnan(float(noise_level(ds, n_samples=1.0e5).NOISE_LEVEL))
    assert abs(right - ZN) < 0.3


@pytest.mark.parametrize("engine", ENGINES)
def test_floor_method_on_thresholded_data(engine):
    """With gates below an SNR of 3 dB removed, the floor is the threshold."""
    ds = weak_echo_sweep(threshold=3.0)
    out = noise_level(
        ds, method="floor", snr_threshold=3.0, percentile=0.5, engine=engine
    )
    assert abs(float(out.NOISE_LEVEL) - ZN) < 0.6
    plain = noise_level(ds, method="floor", percentile=0.5, engine=engine)
    assert float(plain.NOISE_LEVEL) - float(out.NOISE_LEVEL) == pytest.approx(
        3.0, abs=1e-4
    )


@pytest.mark.parametrize("method", ["hildebrand", "floor"])
def test_by_ray_and_range_limits(method):
    ds = synthetic_sweep()
    out = noise_level(ds, method=method, by="ray", min_count=20)
    assert out.NOISE_LEVEL.dims == ("azimuth",)
    assert out.NOISE_LEVEL.shape == (ds.sizes["azimuth"],)
    far = noise_level(ds, range_limits=(120.0, 175.0))
    assert abs(float(far.NOISE_LEVEL) - ZN) < 0.5
    assert int(far.NOISE_SAMPLES) < int(noise_level(ds).NOISE_SAMPLES)


def test_too_few_gates_give_nan():
    ds = xr.Dataset(
        {"DBZH": (("azimuth", "range"), np.full((3, 10), -30.0))},
        coords={"azimuth": [0.0, 1.0, 2.0], "range": 1000.0 + 250.0 * np.arange(10)},
    )
    assert np.isnan(float(noise_level(ds, min_count=100).NOISE_LEVEL))


@compiled_only
@pytest.mark.parametrize("method", ["hildebrand", "floor"])
@pytest.mark.parametrize("by", ["sweep", "ray"])
def test_engines_agree_noise_level(method, by):
    ds = weak_echo_sweep(threshold=1.0)
    ds["DBZH"][3:6, 40:80] = np.nan
    a = noise_level(ds, method=method, by=by, engine="numpy", min_count=20)
    b = noise_level(ds, method=method, by=by, engine="compiled", min_count=20)
    np.testing.assert_allclose(a.NOISE_LEVEL, b.NOISE_LEVEL, rtol=1e-6, equal_nan=True)
    np.testing.assert_array_equal(a.NOISE_SAMPLES, b.NOISE_SAMPLES)


@compiled_only
def test_engines_agree_elementwise_kernels():
    rng = np.random.default_rng(5)
    n = 100_000
    rho = rng.uniform(0.2, 1.0, n)
    snr_db = rng.uniform(-5.0, 40.0, n)
    zdr = rng.uniform(-1.0, 5.0, n)
    for a in (rho, snr_db, zdr):
        a[rng.integers(0, n, 500)] = np.nan
    k = noisemod._noise
    for zdr_arg, has_noise in ((None, False), (zdr, False), (zdr, True)):
        args = (rho, snr_db, zdr_arg, 0.7, has_noise, 0.0, True)
        ref = noisemod._correct_rhohv_numpy(*args)
        got = k.correct_rhohv(
            rho,
            snr_db,
            np.empty(0) if zdr_arg is None else zdr_arg,
            0.7,
            has_noise,
            0.0,
            True,
        )
        np.testing.assert_allclose(got, ref, rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(
        k.correct_zdr(zdr, snr_db, 0.7, 0.0),
        noisemod._correct_zdr_numpy(zdr, snr_db, 0.7, 0.0),
        rtol=1e-9,
        equal_nan=True,
    )
    rng_m = rng.uniform(500.0, 400_000.0, n)
    dbz = rng.uniform(-30.0, 60.0, n)
    for inc in (False, True):
        np.testing.assert_allclose(
            k.snr_db(dbz, np.full(n, -41.0), rng_m, inc),
            noisemod._snr_numpy(dbz, np.full(n, -41.0), rng_m, inc),
            rtol=1e-9,
            equal_nan=True,
        )


# --------------------------------------------------------------------------
# correction of RHOHV and ZDR against simulated voltages
# --------------------------------------------------------------------------


def simulate_channels(rho, zdr_db, snr_h_db, alpha_db, n=400_000, seed=0):
    """
    Measured RHOHV and ZDR from complex Gaussian voltages with noise.

    The signal power of H is ``snr_h`` times its noise power, V follows from
    ``zdr_db`` and the correlation between the signals is ``rho``; the noise
    powers differ by ``alpha_db`` (H minus V). Returns the measured moments.
    """
    rng = np.random.default_rng(seed)

    def cn(size):
        return (rng.standard_normal(size) + 1j * rng.standard_normal(size)) / np.sqrt(2)

    n_h = 1.0
    n_v = 10 ** (-0.1 * alpha_db)
    p_h = n_h * 10 ** (0.1 * snr_h_db)
    p_v = p_h / 10 ** (0.1 * zdr_db)
    a, b = cn(n), cn(n)
    s_h = np.sqrt(p_h) * a
    s_v = np.sqrt(p_v) * (rho * a + np.sqrt(1 - rho**2) * b)
    m_h = s_h + np.sqrt(n_h) * cn(n)
    m_v = s_v + np.sqrt(n_v) * cn(n)
    ph, pv = np.mean(abs(m_h) ** 2), np.mean(abs(m_v) ** 2)
    rho_m = abs(np.mean(np.conj(m_h) * m_v)) / np.sqrt(ph * pv)
    return rho_m, 10 * np.log10(ph / pv)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("alpha_db", [0.0, 1.5])
@pytest.mark.parametrize("snr_db", [0.0, 3.0, 10.0, 20.0])
def test_corrections_against_simulated_voltages(engine, snr_db, alpha_db):
    rho, zdr = 0.99, 1.2
    rho_m, zdr_m = simulate_channels(rho, zdr, snr_db, alpha_db)
    ds = xr.Dataset(
        {
            "RHOHV": ("g", [rho_m]),
            "ZDR": ("g", [zdr_m]),
            "SNR": ("g", [snr_db]),
        }
    )
    rho_c = correct_rhohv_noise(
        ds,
        zdr=True,
        noise_ratio=alpha_db,
        zdr_includes_noise=True,
        min_snr=-10.0,
        engine=engine,
    )
    zdr_c = correct_zdr_noise(ds, noise_ratio=alpha_db, min_snr=-10.0, engine=engine)
    rho_c, zdr_c = float(rho_c[0]), float(zdr_c[0])
    assert abs(rho_c - rho) < 0.006
    assert abs(zdr_c - zdr) < 0.06
    if snr_db <= 3.0:  # the correction is large and must have helped
        assert abs(rho_c - rho) < 0.1 * abs(rho_m - rho)
        assert abs(zdr_c - zdr) < 0.1 * abs(zdr_m - zdr)


@pytest.mark.parametrize("engine", ENGINES)
def test_rhohv_correction_without_zdr_is_equal_snr_form(engine):
    ds = xr.Dataset({"RHOHV": ("g", [0.9, 0.5]), "SNR": ("g", [10.0, 3.0])})
    out = correct_rhohv_noise(ds, engine=engine, clip=False, min_snr=0.0)
    s = 10 ** (0.1 * np.array([10.0, 3.0]))
    np.testing.assert_allclose(
        out.values, np.array([0.9, 0.5]) * (1 + 1 / s), rtol=1e-6
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_correction_limits(engine):
    ds = xr.Dataset(
        {
            "RHOHV": ("g", [0.995, 0.9, np.nan, 0.9]),
            "ZDR": ("g", [4.0, 0.5, 0.5, 0.5]),
            "SNR": ("g", [1.0, -2.0, 10.0, 10.0]),
        }
    )
    clipped = correct_rhohv_noise(ds, engine=engine, min_snr=0.0)
    assert clipped.values[0] == pytest.approx(1.0)  # limited to 1
    assert np.isnan(clipped.values[1])  # below min_snr
    assert np.isnan(clipped.values[2])
    free = correct_rhohv_noise(ds, engine=engine, clip=False, min_snr=0.0)
    assert free.values[0] > 1.0
    # a measured ZDR no signal can have at this SNR: s + 1 <= zdr_linear
    zdr = correct_zdr_noise(ds, engine=engine, min_snr=-5.0)
    assert np.isnan(zdr.values[0])
    assert np.isfinite(zdr.values[1])
    assert np.isnan(zdr.values[2]) is np.False_ or np.isfinite(zdr.values[2])
    assert zdr.values[3] == pytest.approx(0.5, abs=0.06)


def test_high_snr_changes_nothing():
    ds = xr.Dataset({"RHOHV": ("g", [0.97]), "ZDR": ("g", [0.8]), "SNR": ("g", [45.0])})
    assert float(correct_rhohv_noise(ds)[0]) == pytest.approx(0.97, abs=1e-4)
    assert float(correct_zdr_noise(ds)[0]) == pytest.approx(0.8, abs=1e-4)


# --------------------------------------------------------------------------
# xarray behaviour
# --------------------------------------------------------------------------


def corrected_inputs():
    ds = synthetic_sweep()
    s = snr(ds, noise=ZN)
    rng = np.random.default_rng(2)
    ds["RHOHV"] = (
        ("azimuth", "range"),
        np.clip(rng.normal(0.95, 0.02, ds.DBZH.shape), 0, 1).astype("f4"),
    )
    ds["ZDR"] = (("azimuth", "range"), rng.normal(0.6, 0.3, ds.DBZH.shape).astype("f4"))
    ds["SNR"] = s
    return ds


def test_inputs_are_not_changed_and_coordinates_kept():
    ds = corrected_inputs()
    before = ds.copy(deep=True)
    for out in (
        correct_rhohv_noise(ds),
        correct_zdr_noise(ds),
        snr(ds, noise=ZN),
        noise_level(ds),
    ):
        xr.testing.assert_identical(ds, before)
    out = correct_rhohv_noise(ds)
    assert out.dims == ds.RHOHV.dims
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out["range"], ds["range"])
    np.testing.assert_array_equal(out["azimuth"], ds["azimuth"])
    assert out.attrs["units"] == "1"
    assert out.name == "RHOHV"
    z = correct_zdr_noise(ds)
    assert z.attrs["units"] == "dB" and z.name == "ZDR"


def test_dataarray_input_and_accessors():
    ds = corrected_inputs()
    a = correct_rhohv_noise(ds.RHOHV, snr=ds.SNR)
    xr.testing.assert_allclose(a, correct_rhohv_noise(ds))
    xr.testing.assert_allclose(ds.RHOHV.radarx.correct_rhohv_noise(snr=ds.SNR), a)
    xr.testing.assert_allclose(ds.radarx.correct_zdr_noise(), correct_zdr_noise(ds))
    xr.testing.assert_allclose(ds.DBZH.radarx.snr(noise=ZN), snr(ds, noise=ZN))
    xr.testing.assert_allclose(
        ds.radarx.noise_level().NOISE_LEVEL, noise_level(ds).NOISE_LEVEL
    )
    with pytest.raises(ValueError, match="DataArray"):
        correct_rhohv_noise(ds.RHOHV)


def test_snr_from_reflectivity_when_no_snr_field():
    ds = corrected_inputs().drop_vars("SNR")
    a = correct_rhohv_noise(ds, noise=ZN)
    b = correct_rhohv_noise(ds.assign(SNR=snr(ds, noise=ZN)))
    xr.testing.assert_allclose(a, b)
    # estimated noise gives nearly the same
    c = correct_rhohv_noise(ds)
    assert float(abs(a - c).max()) < 0.05


def test_dask_stays_lazy():
    dask = pytest.importorskip("dask")  # noqa: F841
    ds = corrected_inputs()
    lazy = ds.chunk({"azimuth": 30})
    for out in (
        correct_rhohv_noise(lazy),
        correct_zdr_noise(lazy),
        snr(lazy, noise=ZN),
    ):
        assert out.chunks is not None
    xr.testing.assert_allclose(
        correct_rhohv_noise(lazy).compute(), correct_rhohv_noise(ds)
    )
    xr.testing.assert_allclose(snr(lazy, noise=ZN).compute(), snr(ds, noise=ZN))
    # a noise level estimated lazily is a dask scalar; its value is recorded
    est = snr(lazy)
    assert est.attrs["noise_level"] == pytest.approx(float(noise_level(ds).NOISE_LEVEL))
    nl = noise_level(lazy)
    assert nl.NOISE_LEVEL.chunks is not None
    assert float(nl.NOISE_LEVEL.compute()) == pytest.approx(
        float(noise_level(ds).NOISE_LEVEL)
    )


def test_datatree_sweeps():
    tree = volume()
    nl = noise_level(tree)
    assert set(nl.children) == {"sweep_0", "sweep_1"}
    s = snr(tree, noise=nl)
    assert s["sweep_0"]["SNR"].dims == ("azimuth", "range")
    tree["sweep_0"]["RHOHV"] = xr.DataArray(
        np.full(tree["sweep_0"]["DBZH"].shape, 0.9, "f4"), dims=("azimuth", "range")
    )
    tree["sweep_1"]["RHOHV"] = tree["sweep_0"]["RHOHV"]
    out = tree.radarx.correct_rhohv_noise(noise=nl)
    assert set(out.children) == {"sweep_0", "sweep_1"}
    assert (
        (out["sweep_0"]["RHOHV"] >= 0.9 - 1e-6)
        .where(out["sweep_0"]["RHOHV"].notnull())
        .all()
    )
    # sweeps without the field are skipped; none at all is an error
    with pytest.raises(KeyError):
        correct_zdr_noise(tree)


def test_errors():
    ds = synthetic_sweep()
    with pytest.raises(ValueError, match="method"):
        noise_level(ds, method="nope")
    with pytest.raises(ValueError, match="by"):
        noise_level(ds, by="column")
    with pytest.raises(ValueError, match="engine"):
        noise_level(ds, engine="gpu")
    with pytest.raises(KeyError):
        noise_level(ds, dbz="MISSING")
    with pytest.raises(TypeError):
        snr(np.zeros(3))


def test_noise_level_argument_ranges():
    ds = synthetic_sweep()
    with pytest.raises(ValueError, match="n_samples must be positive"):
        noise_level(ds, n_samples=0.0)
    with pytest.raises(ValueError, match="percentile must be between 0 and 100"):
        noise_level(ds, method="floor", percentile=101.0)
    with pytest.raises(ValueError, match="percentile must be between 0 and 100"):
        noise_level(ds, method="floor", percentile=-1.0)
    # n_samples is not used by the floor method
    out = noise_level(ds, method="floor", n_samples=0.0, min_count=20)
    assert np.isfinite(float(out.NOISE_LEVEL))


@pytest.mark.parametrize("method", ["hildebrand", "floor"])
def test_too_few_gates_give_nan_with_both_engines(method):
    ds = xr.Dataset(
        {"DBZH": (("azimuth", "range"), np.full((3, 10), -30.0))},
        coords={"azimuth": [0.0, 1.0, 2.0], "range": 1000.0 + 250.0 * np.arange(10)},
    )
    out = noise_level(ds, method=method, min_count=100, engine="numpy")
    assert np.isnan(float(out.NOISE_LEVEL))
    assert int(out.NOISE_SAMPLES) == 0
    if method == "floor":  # all 30 gates are used once the minimum allows it
        ok = noise_level(ds, method=method, min_count=30, engine="numpy")
        assert int(ok.NOISE_SAMPLES) == 30
        corrected = -30.0 - 20 * np.log10(ds.range.values / 1000.0)
        assert float(ok.NOISE_LEVEL) == pytest.approx(
            np.percentile(np.tile(corrected, 3), 1.0), abs=1e-5
        )


def test_compiled_engine_needs_the_kernel(monkeypatch):
    ds = synthetic_sweep()
    monkeypatch.setattr(noisemod, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError, match="compiled noise kernel"):
        noise_level(ds, engine="compiled")
    assert np.isfinite(float(noise_level(ds, engine="auto").NOISE_LEVEL))


def test_snr_and_correction_inputs_by_dataset_array_or_name():
    ds = corrected_inputs()
    ref = correct_rhohv_noise(ds)
    # SNR from a Dataset that holds it
    from_ds = correct_rhohv_noise(ds.drop_vars("SNR"), snr=xr.Dataset({"SNR": ds.SNR}))
    xr.testing.assert_allclose(from_ds, ref)
    # a name that is not in the sweep
    with pytest.raises(KeyError, match="'SNR_USER' is not in the dataset"):
        correct_rhohv_noise(ds, snr="SNR_USER")
    named = correct_rhohv_noise(ds.rename(SNR="SNR_USER"), snr="SNR_USER")
    xr.testing.assert_allclose(named, ref)
    # a DataArray of ZDR with the reflectivity and the noise
    z = correct_zdr_noise(ds.ZDR, dbz=ds.DBZH, noise=ZN)
    xr.testing.assert_allclose(z, correct_zdr_noise(ds, noise=ZN))
    # a named SNR together with a noise level is not accepted
    with pytest.raises(ValueError, match="exclusive"):
        correct_rhohv_noise(ds, snr="SNR", noise=ZN)
    with pytest.raises(ValueError, match="exclusive"):
        correct_zdr_noise(ds, snr="SNR", noise=ZN)
    # a name that is not there is still reported as missing
    with pytest.raises(KeyError, match="'SNR_USER' is not in the dataset"):
        correct_rhohv_noise(ds, snr="SNR_USER", noise=ZN)
    # a DataArray and no way to find the SNR
    with pytest.raises(ValueError, match="pass snr as a DataArray"):
        correct_zdr_noise(ds.ZDR)


def test_rhohv_zdr_correction_fields():
    ds = corrected_inputs()
    base = correct_rhohv_noise(ds, zdr=True)
    by_name = correct_rhohv_noise(ds, zdr="ZDR")
    by_array = correct_rhohv_noise(ds, zdr=ds.ZDR)
    xr.testing.assert_allclose(by_name, base)
    xr.testing.assert_allclose(by_array, base)
    # the ZDR changes the result compared with none (ratio 1)
    assert float(abs(base - correct_rhohv_noise(ds)).max()) > 1e-4
    with pytest.raises(ValueError, match="pass ZDR as a DataArray"):
        correct_rhohv_noise(ds.RHOHV, snr=ds.SNR, zdr=True)
    with pytest.raises(KeyError):
        correct_rhohv_noise(ds, zdr="ZDR_NONE")


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def test_provenance_and_references():
    ds = corrected_inputs()
    registry = load_registry()
    for out in (
        noise_level(ds),
        snr(ds, noise=ZN),
        correct_rhohv_noise(ds),
        correct_zdr_noise(ds),
    ):
        (info,) = read_provenance(out)
        assert info["method"] and info["references"]
        assert all(r in registry for r in info["references"])
    (info,) = read_provenance(correct_rhohv_noise(ds))
    assert "10.1175/JTECH1912.1" in info["references"]
    (info,) = read_provenance(noise_level(ds))
    assert "10.1175/1520-0450(1974)013<0808:ODOTNL>2.0.CO;2" in info["references"]


# --------------------------------------------------------------------------
# real NEXRAD volume
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def klbb():
    xd = pytest.importorskip("xradar")
    from open_radar_data import DATASETS

    file = DATASETS.fetch("KLBB20160601_150025_V06")
    dtree = xd.io.open_nexradlevel2_datatree(file, sweep=[0, 2])
    for name in ("sweep_0", "sweep_2"):
        ds = dtree[name].to_dataset(inherit=False)
        for var, lim in (("DBZH", -32.0), ("ZDR", -12.9), ("RHOHV", 0.21)):
            ds[var] = ds[var].where(ds[var] > lim)
        dtree[name] = ds
    return dtree


def test_real_nexrad_noise_floor_and_snr(klbb):
    nl = noise_level(klbb, method="floor")
    for name in ("sweep_0", "sweep_2"):
        level = float(nl[name].NOISE_LEVEL)
        assert (
            -45.0 < level < -38.0
        )  # the WSR-88D reports down to about -41 dBZ at 1 km
    # on thresholded data the objective method also lands on the floor
    hs = float(noise_level(klbb["sweep_0"].to_dataset()).NOISE_LEVEL)
    assert abs(hs - float(nl["sweep_0"].NOISE_LEVEL)) < 2.0
    s = snr(klbb, noise=nl)
    snr0 = s["sweep_0"]["SNR"]
    assert float(snr0.min()) > -1.0  # nothing below the lowest reportable value
    assert float(snr0.max()) > 60.0


def test_real_nexrad_noise_correction_leaves_strong_signal_alone(klbb):
    ds = klbb["sweep_0"].to_dataset()
    s = snr(ds, noise=-41.0)
    zdr = correct_zdr_noise(ds, snr=s)
    rho = correct_rhohv_noise(ds, snr=s)
    strong = (s > 25.0) & ds.ZDR.notnull() & ds.RHOHV.notnull()
    assert int(strong.sum()) > 1000
    assert float(abs(zdr - ds.ZDR).where(strong).max()) < 0.1
    assert float((rho - ds.RHOHV).where(strong).max()) < 0.01
    weak = (s < 6.0) & ds.RHOHV.notnull() & rho.notnull() & (rho < 1.0)
    assert float((rho - ds.RHOHV).where(weak).min()) >= 0.0
    assert float(rho.max()) <= 1.0
