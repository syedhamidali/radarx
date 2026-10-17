#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Level I processing on the velocity field of a real storm
========================================================

Open radar data has no IQ time series, so the IQ of a real scene is simulated
from the Level II fields of a NEXRAD Doppler cut (KLBB, 1 June 2016): the mean
velocity, the spectrum width and a signal-to-noise ratio from the reflectivity
and the range. The whole Level I chain then has to bring the velocity back to
what the radar measured, with and without aliasing by a short Nyquist
interval. The file comes from open-radar-data and the test is skipped when it
cannot be fetched.
"""
import numpy as np
import pytest
import xarray as xr

from radarx import level1

WAVELENGTH = 0.1071  # m, NEXRAD (2.8 GHz)
NOISE_DBZ_1KM = -42.0  # equivalent reflectivity of the noise at 1 km
NODATA = -63.9


@pytest.fixture(scope="module")
def scene():
    xd = pytest.importorskip("xradar")
    open_radar_data = pytest.importorskip("open_radar_data")
    try:
        file = open_radar_data.DATASETS.fetch("KLBB20160601_150025_V06")
        dtree = xd.io.open_nexradlevel2_datatree(file, sweep=[1])
    except Exception as err:  # pragma: no cover - offline
        pytest.skip(f"open radar data not available: {err}")
    ds = dtree["sweep_1"].to_dataset()
    ds = ds.isel(azimuth=slice(100, 164), range=slice(0, 400))
    vel, width, dbz = ds["VRADH"], ds["WRADH"], ds["DBZH"]
    valid = (vel > NODATA) & (width >= 0) & (dbz > -20) & np.isfinite(vel)
    r_km = (ds["range"] / 1000.0).astype(float)
    snr = (dbz - NOISE_DBZ_1KM - 20 * np.log10(r_km)).where(valid)
    return {
        "velocity": vel.where(valid, 0.0),
        "width": width.where(valid, 1.0).clip(min=0.3),
        "snr": snr,
        "valid": valid & (snr > 8.0),
        "power": xr.where(valid, 10 ** (0.1 * snr.fillna(0.0)), 0.0),
    }


def make_iq(scene, nyquist, n_pulses=64, seed=1):
    prt = WAVELENGTH / (4 * nyquist)
    return level1.simulate_iq(
        scene["velocity"].values,
        width=scene["width"].values,
        power=scene["power"].values,
        noise=1.0,
        n_pulses=n_pulses,
        wavelength=WAVELENGTH,
        prt=prt,
        dims=("azimuth", "range"),
        seed=seed,
    )


def mean_db(values):
    """Mean power of values in dB (the average of dB values is biased low)."""
    values = values[np.isfinite(values)]
    return 10 * np.log10(np.mean(10 ** (0.1 * values)))


def wrap_error(a, b, nyquist):
    return (a - b + nyquist) % (2 * nyquist) - nyquist


def test_the_chain_recovers_the_radar_velocity(scene):
    nyq = 30.0  # above every velocity of the cut: no aliasing
    iq = make_iq(scene, nyq)
    ok = scene["valid"].values
    truth = scene["velocity"].values
    pp = level1.pulse_pair_moments(iq, noise=1.0)
    err = (pp.VRADH.values - truth)[ok]
    assert np.sqrt(np.mean(err**2)) < 1.0
    assert abs(np.mean(err)) < 0.1
    assert np.nanmean(pp.WRADH.values[ok]) == pytest.approx(
        np.mean(scene["width"].values[ok]), rel=0.1
    )
    assert mean_db(pp.SNR.values[ok]) == pytest.approx(
        mean_db(scene["snr"].values[ok]), abs=0.5
    )
    spectra = level1.doppler_spectra(iq)
    sm = level1.spectral_moments(spectra, min_snr=3.0)
    err = (sm.VRADH.values - truth)[ok]
    assert np.sqrt(np.nanmean(err**2)) < 2.0
    assert abs(np.nanmean(err)) < 0.15
    assert mean_db(sm.SNR.values[ok]) == pytest.approx(
        mean_db(scene["snr"].values[ok]), abs=1.0
    )


def test_engines_agree_on_the_scene(scene):
    iq = make_iq(scene, 30.0, n_pulses=32)
    a = level1.spectral_moments(
        level1.doppler_spectra(iq, engine="numpy"), engine="numpy"
    )
    b = level1.spectral_moments(
        level1.doppler_spectra(iq, engine="auto"), engine="auto"
    )
    for name in a.data_vars:
        np.testing.assert_allclose(
            a[name], b[name], rtol=1e-4, atol=1e-4, equal_nan=True
        )


def test_dealiasing_with_the_spectra_in_a_short_nyquist_interval(scene):
    nyq = 6.0  # most of the cut is near -7 m/s: it folds
    iq = make_iq(scene, nyq)
    truth = scene["velocity"].values
    # a spectrum as wide as a third of the Nyquist interval fills it, and the
    # noise level cannot be told from the signal: keep the narrower echoes
    ok = scene["valid"].values & (scene["width"].values < 2.0)
    assert np.mean(np.abs(truth[ok]) > nyq) > 0.5  # there is aliasing to undo
    spectra = level1.doppler_spectra(iq)
    moments = level1.spectral_moments(spectra, min_snr=3.0)
    folded = moments.VRADH.values
    assert np.nanmax(np.abs(folded)) <= nyq
    # the measured velocity is the true one modulo the Nyquist interval
    err = wrap_error(folded, truth, nyq)[ok]
    assert np.sqrt(np.nanmean(err**2)) < 2.5
    # a coarse reference (a model wind with 2.5 m/s error) restores the field
    rng = np.random.default_rng(3)
    reference = xr.DataArray(
        truth + rng.normal(0, 2.5, truth.shape), dims=("azimuth", "range")
    )
    out = level1.dealias_spectra(spectra, reference=reference)
    unfolded = out.VRADH.values
    found = np.isfinite(unfolded)[ok]
    assert np.mean(found) > 0.9  # the gates the SNR limit keeps
    good = np.abs(unfolded - truth)[ok][found] < 4.0
    assert np.mean(good) > 0.95
    assert int(out.FOLD.min()) <= -1
