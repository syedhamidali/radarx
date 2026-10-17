#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the ground clutter filters
====================================

Weather with a known velocity is mixed with ground clutter, a narrow echo at
zero velocity that is 30 to 50 dB stronger. The filters must bring the
velocity back to the truth, and must not touch the gates they are not asked
to filter.
"""
import numpy as np
import pytest
import xarray as xr
from level1_data import PRT, WAVELENGTH, gates, sweep, volume

from radarx import level1
from radarx._provenance import load_registry, read_provenance
from radarx.level1 import clutter as cl
from radarx.level1 import spectra as sp

ENGINES = ["numpy"] + (["compiled"] if sp.HAS_COMPILED_KERNEL else [])
compiled_only = pytest.mark.skipif(
    not sp.HAS_COMPILED_KERNEL, reason="compiled kernels not built"
)


def dirty(clutter=1e4, n=600, seed=2, n_pulses=64, window_rich=False):
    """Weather of 100 (20 dB over the noise) plus clutter 20 dB stronger."""
    return gates(
        n=n,
        vmax=12.0,
        width=1.5,
        snr_db=20.0,
        n_pulses=n_pulses,
        clutter=clutter,
        clutter_width=0.2,
        seed=seed,
    )


def far_from_zero(truth, limit=5.0):
    return np.abs(truth) > limit


@pytest.mark.parametrize("engine", ENGINES)
def test_the_regression_filter_restores_the_velocity(engine):
    iq, truth = dirty(clutter=1e5)
    far = far_from_zero(truth)
    raw = level1.pulse_pair_moments(iq, noise=1.0)
    assert np.mean(np.abs(raw.VRADH.values[far] - truth[far])) > 3.0  # clutter wins
    errors = []
    for order in (1, 2, 3):
        filtered = level1.regression_filter(iq, order=order, engine=engine)
        pp = level1.pulse_pair_moments(filtered["IQ"], noise=1.0)
        errors.append(np.mean(np.abs(pp.VRADH.values[far] - truth[far])))
    assert errors[-1] < 0.6
    assert errors[-1] < errors[0]  # a higher order removes more of the clutter
    removed = 10 * np.log10(np.mean(10 ** (0.1 * filtered.CLUTTER_POWER.values)))
    assert removed == pytest.approx(50.0, abs=0.5)  # 1e5 of clutter power, 50 dB


def test_a_tone_at_zero_velocity_is_removed_by_the_mean_filter():
    dc = np.full((3, 32), 2.0 + 1.0j)
    iq = level1.as_iq(dc, dims=("range", "pulse"), wavelength=WAVELENGTH, prt=PRT)
    for engine in ENGINES:
        out = level1.regression_filter(iq, order=0, engine=engine)
        assert float(np.abs(out["IQ"]).max()) < 1e-6
        assert float(out.CLUTTER_POWER.mean()) == pytest.approx(
            10 * np.log10(5.0), abs=1e-4
        )
    # nothing to remove: the power removed is NaN
    zero = iq * 0
    out = level1.regression_filter(zero, order=1)
    assert np.all(np.isnan(out.CLUTTER_POWER.values))


def test_the_regression_filter_leaves_a_fast_weather_signal_nearly_unchanged():
    iq, truth = gates(n=60, vmax=22.0, width=1.0, snr_db=30.0, seed=4)
    fast = np.abs(truth) > 15.0
    out = level1.regression_filter(iq.isel(range=fast), order=2)
    kept = float(
        (np.abs(out["IQ"]) ** 2).mean() / (np.abs(iq.isel(range=fast)) ** 2).mean()
    )
    assert kept > 0.97
    assert out["IQ"].dims == iq.dims and out["IQ"].dtype == iq.dtype
    assert out["IQ"].attrs["wavelength"] == WAVELENGTH


@pytest.mark.parametrize("engine", ENGINES)
def test_where_restricts_the_regression_filter_to_flagged_gates(engine):
    iq, truth = dirty(n=40)
    flag = xr.DataArray(np.arange(40) % 2 == 0, dims="range")
    out = level1.regression_filter(iq, order=2, where=flag, engine=engine)
    full = level1.regression_filter(iq, order=2, engine=engine)
    np.testing.assert_array_equal(out["IQ"].values[:, 1::2], iq.values[:, 1::2])
    np.testing.assert_allclose(out["IQ"].values[:, ::2], full["IQ"].values[:, ::2])
    assert np.all(out.CLUTTER_POWER.values[1::2] == -np.inf) or np.all(
        np.isnan(out.CLUTTER_POWER.values[1::2])
    )
    scalar = level1.regression_filter(iq, order=1, where=True)
    np.testing.assert_allclose(
        scalar["IQ"], level1.regression_filter(iq, order=1)["IQ"]
    )


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("fill", ["noise", "interpolate", "gaussian"])
def test_the_notch_filter_restores_the_velocity(engine, fill):
    iq, truth = dirty(clutter=1e4, n_pulses=128, seed=3)
    far = far_from_zero(truth, 6.0)
    spectra = level1.doppler_spectra(iq, window="blackman")
    raw = level1.spectral_moments(spectra, engine=engine)
    assert np.mean(np.abs(raw.VRADH.values[far] - truth[far])) > 2.0
    out = level1.notch_filter(spectra, half_width=3, fill=fill, engine=engine)
    m = level1.spectral_moments(out["SPECTRUM"], engine=engine)
    err = np.abs(m.VRADH.values[far] - truth[far])
    assert np.mean(err) < 1.0 if fill != "noise" else np.mean(err) < 1.5
    assert float(out.CLUTTER_POWER.mean()) > 15.0  # clutter power removed, dB
    assert out["SPECTRUM"].attrs["n_average"] == 1.0
    assert out["SPECTRUM"].dims == spectra.SPECTRUM.dims
    assert out.attrs["wavelength"] == WAVELENGTH


def test_interpolation_fills_the_notch_with_a_line_and_noise_fill_with_the_noise_level():
    iq, _ = dirty(n=5)
    spectra = level1.doppler_spectra(iq)
    s = spectra.SPECTRUM.values.astype(np.float64)
    c = 32
    lin = level1.notch_filter(spectra, half_width=2, fill="interpolate").SPECTRUM.values
    for g in range(5):
        a, b = s[g, c - 3], s[g, c + 3]
        np.testing.assert_allclose(
            lin[g, c - 2 : c + 3], a + (b - a) * np.arange(1, 6) / 6, rtol=1e-5
        )
        np.testing.assert_allclose(lin[g, : c - 2], s[g, : c - 2], rtol=1e-6)
    noise = level1.spectral_noise(spectra)
    fill = level1.notch_filter(
        spectra, half_width=1, fill="noise", noise=noise
    ).SPECTRUM.values
    per_bin = 10 ** (0.1 * noise.NOISE_POWER.values) / 64
    np.testing.assert_allclose(fill[:, c], per_bin, rtol=1e-4)


@pytest.mark.parametrize("engine", ENGINES)
def test_where_restricts_the_notch_filter(engine):
    iq, _ = dirty(n=12)
    spectra = level1.doppler_spectra(iq, window="blackman")
    flag = xr.DataArray(np.arange(12) < 6, dims="range")
    out = level1.notch_filter(spectra, half_width=2, where=flag, engine=engine)
    np.testing.assert_array_equal(out.SPECTRUM.values[6:], spectra.SPECTRUM.values[6:])
    assert np.all(np.isnan(out.CLUTTER_POWER.values[6:]))
    full = level1.notch_filter(spectra, half_width=2, engine=engine)
    np.testing.assert_allclose(out.SPECTRUM.values[:6], full.SPECTRUM.values[:6])
    given = level1.notch_filter(
        spectra, noise=1.0, half_width=1, where=flag, engine=engine
    )
    assert given.SPECTRUM.shape == spectra.SPECTRUM.shape


@compiled_only
def test_engines_agree_on_the_filters():
    iq, _ = dirty(n=40, n_pulses=48)
    for order in (0, 2):
        a = level1.regression_filter(iq, order=order, engine="numpy")
        b = level1.regression_filter(iq, order=order, engine="compiled")
        np.testing.assert_allclose(a["IQ"], b["IQ"], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(
            a.CLUTTER_POWER, b.CLUTTER_POWER, rtol=1e-5, equal_nan=True
        )
    spectra = level1.doppler_spectra(iq, segments=2)
    for fill in ("noise", "interpolate", "gaussian"):
        a = level1.notch_filter(spectra, half_width=1, fill=fill, engine="numpy")
        b = level1.notch_filter(spectra, half_width=1, fill=fill, engine="compiled")
        np.testing.assert_allclose(a.SPECTRUM, b.SPECTRUM, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(
            a.CLUTTER_POWER, b.CLUTTER_POWER, rtol=1e-4, atol=1e-4, equal_nan=True
        )


def test_volumes_and_datasets():
    iq, _ = sweep(n_az=2, n_rng=6, n_pulses=32)
    tree = volume(iq)
    reg = level1.regression_filter(tree, order=1)
    assert set(reg.children) == {"sweep_0", "sweep_1"}
    assert "IQ" in reg["sweep_0"].dataset
    spectra = level1.doppler_spectra(tree)
    notch = level1.notch_filter(spectra)
    assert set(notch.children) == {"sweep_0", "sweep_1"}
    ds = xr.Dataset({"IQ": iq})
    assert "CLUTTER_POWER" in level1.regression_filter(ds)
    per_sweep = level1.regression_filter(
        tree, order=1, where={"sweep_0": True, "sweep_1": False}
    )
    assert set(per_sweep.children) == {"sweep_0", "sweep_1"}


def test_dask_stays_lazy():
    pytest.importorskip("dask")
    iq, _ = sweep(n_az=4, n_rng=6, n_pulses=32)
    chunked = iq.chunk({"azimuth": 2})
    out = level1.regression_filter(chunked, order=1)
    assert out["IQ"].chunks is not None and out.CLUTTER_POWER.chunks is not None
    ref = level1.regression_filter(iq, order=1)
    np.testing.assert_allclose(out["IQ"].compute(), ref["IQ"], rtol=1e-4, atol=1e-5)
    spectra = level1.doppler_spectra(chunked)
    notch = level1.notch_filter(spectra)
    assert notch.SPECTRUM.chunks is not None
    np.testing.assert_allclose(
        notch.SPECTRUM.compute(),
        level1.notch_filter(level1.doppler_spectra(iq)).SPECTRUM,
        rtol=1e-5,
    )


def test_inputs_unchanged_and_provenance():
    iq, _ = dirty(n=10)
    before = iq.copy(deep=True)
    out = level1.regression_filter(iq, order=1)
    xr.testing.assert_identical(iq, before)
    spectra = level1.doppler_spectra(iq)
    snapshot = spectra.SPECTRUM.copy(deep=True)
    notch = level1.notch_filter(spectra)
    xr.testing.assert_identical(spectra.SPECTRUM, snapshot)
    registry = load_registry()
    for result, doi in (
        (out, "10.1175/1520-0426(1999)016<1364:GCCWAR>2.0.CO;2"),
        (notch, "10.1109/MSP.2011.941846"),
    ):
        prov = read_provenance(result)
        assert doi in prov[0]["references"]
        assert all(r in registry for r in prov[0]["references"])
    assert out["IQ"].dtype == iq.dtype
    assert out.CLUTTER_POWER.dtype == np.float32
    assert notch.CLUTTER_POWER.attrs["units"] == "dB"


def test_polynomial_basis_is_orthonormal_and_spans_polynomials():
    for m, order in ((16, 0), (64, 3), (9, 7)):
        q = cl.polynomial_basis(order, m)
        assert q.shape == (m, order + 1)
        np.testing.assert_allclose(q.T @ q, np.eye(order + 1), atol=1e-12)
    t = np.linspace(-1, 1, 30)
    q = cl.polynomial_basis(3, 30)
    for k in range(4):
        np.testing.assert_allclose(q @ (q.T @ t**k), t**k, atol=1e-10)


def test_errors():
    iq, _ = dirty(n=6, n_pulses=16)
    spectra = level1.doppler_spectra(iq)
    with pytest.raises(ValueError, match="order"):
        level1.regression_filter(iq, order=-1)
    with pytest.raises(ValueError, match="degrees of freedom"):
        level1.regression_filter(iq, order=15)
    with pytest.raises(ValueError, match="complex"):
        level1.regression_filter(iq.real)
    with pytest.raises(ValueError, match="no 'slow'"):
        level1.regression_filter(iq, dim="slow")
    with pytest.raises(ValueError, match="engine"):
        level1.regression_filter(iq, engine="gpu")
    with pytest.raises(KeyError):
        level1.regression_filter(xr.Dataset({"x": (("a",), [1.0])}))
    with pytest.raises(TypeError):
        level1.regression_filter([1, 2, 3])
    with pytest.raises(ValueError, match="fill"):
        level1.notch_filter(spectra, fill="gauss")
    with pytest.raises(ValueError, match="half_width"):
        level1.notch_filter(spectra, half_width=-1)
    with pytest.raises(ValueError, match="n_sigma"):
        level1.notch_filter(spectra, n_sigma=-1.0)
    with pytest.raises(ValueError, match="min_bins"):
        level1.notch_filter(spectra, min_bins=0)
    with pytest.raises(KeyError):
        level1.notch_filter(xr.Dataset({"x": (("a",), [1.0])}))


def test_missing_kernels(monkeypatch):
    iq, _ = dirty(n=4, n_pulses=16)
    monkeypatch.setattr(sp, "HAS_COMPILED_SPECTRA", False)
    monkeypatch.setattr(sp, "HAS_COMPILED_MOMENTS", False)
    monkeypatch.setattr(sp, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError):
        level1.regression_filter(iq, engine="compiled")
    out = level1.regression_filter(iq)
    assert out["IQ"].shape == iq.shape
    assert level1.notch_filter(level1.doppler_spectra(iq)).SPECTRUM.shape[0] == 4
