#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the Doppler spectra and the spectral and pulse pair moments
=====================================================================

The truth is known: complex Gaussian time series with a Gaussian Doppler
spectrum of given mean velocity, width and power (Zrnic 1975), plus white
noise. Averaged over many gates the estimators must return the truth within
the sampling error of a 64 pulse dwell, with both engines and from any
dimension order.
"""
import numpy as np
import pytest
import xarray as xr
from level1_data import NYQUIST, PRT, WAVELENGTH, gates, sweep, volume

from radarx import level1
from radarx._provenance import load_registry, read_provenance
from radarx.level1 import spectra as sp

ENGINES = ["numpy"] + (["compiled"] if sp.HAS_COMPILED_KERNEL else [])
compiled_only = pytest.mark.skipif(
    not sp.HAS_COMPILED_KERNEL, reason="compiled kernels not built"
)


def wrap(v):
    return (v + NYQUIST) % (2 * NYQUIST) - NYQUIST


def circular_error(a, b):
    return (a - b + NYQUIST) % (2 * NYQUIST) - NYQUIST


def tone(v, n=64, power=4.0):
    """A pure tone of radial velocity v (away from the radar)."""
    t = np.arange(n) * PRT
    return power**0.5 * np.exp(-4j * np.pi * v * t / WAVELENGTH)


# ---- spectra ---------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("n", [64, 50])
def test_a_tone_peaks_at_its_velocity_and_the_power_adds_up(engine, n):
    v = 7.0
    iq = level1.as_iq(
        tone(v, n)[None, :], dims=("range", "pulse"), wavelength=WAVELENGTH, prt=PRT
    )
    iq = iq.transpose("range", "pulse")
    spectra = level1.doppler_spectra(iq, window="rect", engine=engine)
    s = spectra.SPECTRUM
    assert s.dims == ("range", "velocity")
    peak = float(s.velocity[s.argmax("velocity")].item())
    dv = WAVELENGTH / (2 * n * PRT)
    assert abs(peak - v) <= dv / 2 + 1e-9
    # the bins add up to the mean power of the series
    assert float(s.sum("velocity")[0]) == pytest.approx(4.0, rel=1e-5)
    assert spectra.attrs["nyquist_velocity"] == pytest.approx(NYQUIST)
    assert float(s.velocity[n // 2]) == 0.0


def test_the_velocity_axis_and_attributes():
    iq, _ = gates(n=5, n_pulses=32)
    s = level1.doppler_spectra(iq, window="blackman", segments=2)
    assert s.sizes["velocity"] == 16
    dv = WAVELENGTH / (2 * 16 * PRT)
    np.testing.assert_allclose(np.diff(s.velocity), dv)
    assert s.velocity.dtype == np.float64
    assert s.SPECTRUM.dtype == np.float32
    assert s.SPECTRUM.attrs["n_average"] == 2.0
    assert s.attrs["window"] == "blackman"
    assert s.attrs["window_variance_bins2"] > 0
    assert s.velocity.attrs["units"] == "m s-1"


def test_window_variance_values():
    assert sp.make_window("rect", 32)[1] == 0.0
    w, var = sp.make_window("hann", 64)
    assert var == pytest.approx(1 / 3, rel=0.01)  # 1/3 bin^2 for the Hann window
    np.testing.assert_allclose(w[0], 0.0)
    assert sp.make_window("hamming", 64)[1] > 0
    assert sp.make_window("blackman", 64)[1] > 0
    with pytest.raises(ValueError, match="window"):
        sp.make_window("kaiser", 8)


@pytest.mark.parametrize("window", ["rect", "hann", "hamming", "blackman"])
def test_noise_floor_of_white_noise(window):
    iq = level1.simulate_iq(
        np.zeros(500),
        width=1.0,
        power=0.0,
        noise=2.0,
        n_pulses=64,
        wavelength=WAVELENGTH,
        prt=PRT,
        dims=("range",),
        seed=4,
    )
    s = level1.doppler_spectra(iq, window=window).SPECTRUM
    # white noise: the bins hold noise / bins on average, and add up to the power
    assert float(s.mean()) == pytest.approx(2.0 / 64, rel=0.02)
    assert float(s.sum("velocity").mean()) == pytest.approx(2.0, rel=0.02)


@pytest.mark.parametrize("engine", ENGINES)
def test_spectral_noise_finds_the_noise_level(engine):
    iq, _ = gates(n=800, snr_db=15.0, n_pulses=128, seed=3)
    s = level1.doppler_spectra(iq, segments=2)
    noise = level1.spectral_noise(s, engine=engine).NOISE_POWER
    assert noise.attrs["units"] == "dB"
    # true total noise power is 1 (0 dB); the objective method is biased slightly low
    assert -1.0 < float(noise.mean()) < 0.2


# ---- moments against the truth ---------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("width", [1.0, 3.0])
def test_spectral_moments_recover_the_truth(engine, width):
    iq, truth = gates(n=1500, width=width, snr_db=30.0, n_pulses=64)
    m = level1.spectral_moments(level1.doppler_spectra(iq), engine=engine)
    err = m.VRADH.values - truth
    assert abs(err.mean()) < 0.05
    assert err.std() < 0.5 + 0.2 * width
    assert np.nanmean(m.WRADH.values) == pytest.approx(width, rel=0.05)
    assert np.nanmean(m.SNR.values) == pytest.approx(30.0, abs=1.0)
    assert np.nanmean(m.NOISE_POWER.values) == pytest.approx(0.0, abs=0.8)
    assert m.VRADH.attrs["units"] == "m s-1"
    assert m.VRADH.dtype == np.float32


def test_skewness_and_kurtosis_of_a_gaussian_spectrum():
    # averaged periodograms: skewness 0 and kurtosis 3 of a Gaussian
    iq, _ = gates(n=800, width=3.0, snr_db=30.0, n_pulses=256, vmax=10.0)
    m = level1.spectral_moments(level1.doppler_spectra(iq, segments=4))
    assert abs(np.nanmean(m.SKEW.values)) < 0.1
    assert np.nanmean(m.KURT.values) == pytest.approx(3.0, abs=0.5)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("method, tolerance", [("r0r1", 0.06), ("r1r2", 0.04)])
def test_pulse_pair_recovers_the_truth(engine, method, tolerance):
    iq, truth = gates(n=1500, width=2.0, snr_db=30.0, n_pulses=64)
    m = level1.pulse_pair_moments(iq, noise=1.0, width_method=method, engine=engine)
    assert abs(np.mean(m.VRADH.values - truth)) < 0.03
    assert np.mean(m.WRADH.values) == pytest.approx(2.0, rel=tolerance)
    assert float(m.SNR.mean()) == pytest.approx(30.0, abs=0.3)
    assert float(m.SIGNAL_POWER.mean()) == pytest.approx(30.0, abs=0.3)
    assert list(m.data_vars) == ["SIGNAL_POWER", "VRADH", "WRADH", "NOISE_POWER", "SNR"]


def test_pulse_pair_without_noise_returns_the_total_power():
    iq, truth = gates(n=300, snr_db=10.0)
    m = level1.pulse_pair_moments(iq)
    assert list(m.data_vars) == ["SIGNAL_POWER", "VRADH", "WRADH"]
    # signal 10 dB above a noise of 1: total power 1 + 10 = 11
    assert float(m.SIGNAL_POWER.mean()) == pytest.approx(10 * np.log10(11), abs=0.3)
    assert "total power" in m.SIGNAL_POWER.attrs["long_name"]


def test_pulse_pair_with_the_spectral_noise_estimate():
    iq, truth = gates(n=1500, width=2.0, snr_db=15.0, n_pulses=128)
    m = level1.pulse_pair_moments(iq, noise="spectral")
    assert float(m.NOISE_POWER.mean()) == pytest.approx(0.0, abs=1.0)
    assert float(m.SNR.mean()) == pytest.approx(15.0, abs=1.5)
    assert abs(np.mean(m.VRADH.values - truth)) < 0.05


def test_noise_as_a_dataset_a_db_array_and_a_number_agree():
    iq, _ = gates(n=200, snr_db=20.0, n_pulses=128)
    spectra = level1.doppler_spectra(iq)
    noise = level1.spectral_noise(spectra)
    a = level1.spectral_moments(spectra, noise=noise)
    b = level1.spectral_moments(spectra, noise=noise.NOISE_POWER)
    linear = 10 ** (0.1 * noise.NOISE_POWER)
    linear.attrs = {}
    c = level1.spectral_moments(spectra, noise=linear)
    for other in (b, c):
        for name in a.data_vars:
            np.testing.assert_allclose(
                a[name], other[name], rtol=1e-5, atol=1e-5, equal_nan=True
            )
    d = level1.spectral_moments(spectra, noise=1.0)
    assert float(d.NOISE_POWER.mean()) == pytest.approx(0.0, abs=1e-4)
    p = level1.pulse_pair_moments(iq, noise=noise)
    q = level1.pulse_pair_moments(iq, noise=1.0)
    assert float(abs(p.NOISE_POWER - q.NOISE_POWER).mean()) < 1.0


def test_aliased_velocities_wrap_to_the_nyquist_interval():
    truth = np.array([-60.0, -30.0, 28.0, 40.0, 60.0])
    iq = level1.simulate_iq(
        truth,
        width=1.0,
        power=500.0,
        noise=1.0,
        n_pulses=64,
        wavelength=WAVELENGTH,
        prt=PRT,
        dims=("range",),
        seed=2,
    )
    expected = wrap(truth)
    for m in (
        level1.pulse_pair_moments(iq, noise=1.0),
        level1.spectral_moments(level1.doppler_spectra(iq)),
    ):
        assert np.all(np.abs(circular_error(m.VRADH.values, expected)) < 1.0)
        assert float(m.VRADH.min()) >= -NYQUIST and float(m.VRADH.max()) < NYQUIST


@pytest.mark.parametrize("engine", ENGINES)
def test_a_spectrum_straddling_the_nyquist_velocity_is_not_split(engine):
    # mean velocity at the Nyquist velocity, width 3 m/s: the spectrum is cut in two
    truth = np.full(400, NYQUIST - 0.5)
    iq = level1.simulate_iq(
        truth,
        width=3.0,
        power=1000.0,
        noise=1.0,
        n_pulses=64,
        wavelength=WAVELENGTH,
        prt=PRT,
        dims=("range",),
        seed=7,
    )
    spectra = level1.doppler_spectra(iq)
    m = level1.spectral_moments(spectra, engine=engine)
    err = circular_error(m.VRADH.values, truth)
    assert abs(np.mean(err)) < 0.15
    assert np.mean(m.WRADH.values) == pytest.approx(3.0, rel=0.06)
    # the first moment over the principal interval would be near zero instead
    s = spectra.SPECTRUM.values.astype(float)
    naive = (s * spectra.velocity.values).sum(1) / s.sum(1)
    assert np.mean(np.abs(naive)) < 10.0 < NYQUIST


def test_region_all_collects_a_second_peak_that_the_peak_region_ignores():
    k = 64
    v = (np.arange(k) - k // 2) * (WAVELENGTH / (2 * k * PRT))
    j = np.arange(k)
    s = (
        1.0
        + 200.0 * np.exp(-0.5 * ((j - 40) / 2.0) ** 2)
        + 100.0 * np.exp(-0.5 * ((j - 10) / 1.5) ** 2)
    )
    spectra = xr.DataArray(
        s[None, :].astype(np.float32),
        dims=("range", "velocity"),
        coords={"velocity": v},
        attrs={"n_average": 1.0},
    )
    peak = level1.spectral_moments(
        spectra, region="peak", noise=float(k) * 1.0, n_sigma=3.0
    )
    allb = level1.spectral_moments(
        spectra, region="all", noise=float(k) * 1.0, n_sigma=3.0
    )
    peak, allb = (r.isel(range=0) for r in (peak, allb))
    assert float(peak.VRADH) == pytest.approx(v[40], abs=0.3)
    assert float(peak.WRADH) == pytest.approx(2.0 * v[1] - 2.0 * v[0], rel=0.1)
    assert float(allb.WRADH) > 3 * float(peak.WRADH)
    assert float(allb.SIGNAL_POWER) > float(peak.SIGNAL_POWER)


def test_no_signal_gives_nan_moments_and_a_nan_gate_is_masked():
    iq, _ = gates(n=6, snr_db=20.0, n_pulses=32)
    noise_only = level1.simulate_iq(
        np.zeros(3),
        width=1.0,
        power=0.0,
        noise=1.0,
        n_pulses=32,
        wavelength=WAVELENGTH,
        prt=PRT,
        dims=("range",),
        seed=3,
    )
    s = level1.doppler_spectra(iq)
    s["SPECTRUM"].values[1, 3] = np.nan
    m = level1.spectral_moments(s)
    assert np.isnan(m.VRADH.values[1])
    assert np.isfinite(m.VRADH.values[0])
    # a flat noise spectrum: nothing exceeds the threshold at a high n_sigma
    flat = level1.doppler_spectra(noise_only)
    m = level1.spectral_moments(flat, n_sigma=8.0)
    assert np.all(np.isnan(m.VRADH.values))
    assert np.all(np.isnan(m.SNR.values))


def test_a_negative_width_estimate_is_zero():
    # a pure tone: |R1| = S, ln(S/|R1|) is zero or slightly negative from rounding
    iq = level1.as_iq(
        tone(5.0)[None, :], dims=("range", "pulse"), wavelength=WAVELENGTH, prt=PRT
    )
    iq = iq.transpose("range", "pulse")
    m = level1.pulse_pair_moments(iq, noise=0.0).isel(range=0)
    assert float(m.WRADH) == pytest.approx(0.0, abs=0.05)
    assert float(m.VRADH) == pytest.approx(5.0, abs=1e-4)
    # a power below the noise has no signal
    m = level1.pulse_pair_moments(iq, noise=100.0).isel(range=0)
    assert np.isnan(float(m.WRADH)) and np.isnan(float(m.SIGNAL_POWER))


# ---- engines, dimension order, containers ----------------------------------


@compiled_only
@pytest.mark.parametrize("window", ["hann", "rect"])
def test_engines_agree_on_every_function(window):
    iq, _ = sweep(n_az=3, n_rng=20, n_pulses=48)
    spectra = {}
    for engine in ("numpy", "compiled"):
        spectra[engine] = level1.doppler_spectra(
            iq, window=window, segments=2, engine=engine
        )
    np.testing.assert_allclose(
        spectra["numpy"].SPECTRUM, spectra["compiled"].SPECTRUM, rtol=1e-5
    )
    s = spectra["numpy"]
    noise_n = level1.spectral_noise(s, engine="numpy")
    noise_c = level1.spectral_noise(s, engine="compiled")
    np.testing.assert_allclose(noise_n.NOISE_POWER, noise_c.NOISE_POWER, rtol=1e-5)
    for region in ("peak", "all"):
        a = level1.spectral_moments(s, region=region, engine="numpy")
        b = level1.spectral_moments(s, region=region, engine="compiled")
        for name in a.data_vars:
            np.testing.assert_allclose(
                a[name], b[name], rtol=1e-5, atol=1e-5, equal_nan=True
            )
    for method in ("r0r1", "r1r2"):
        for noise in (None, 1.0, "spectral"):
            a = level1.pulse_pair_moments(
                iq, noise=noise, width_method=method, engine="numpy"
            )
            b = level1.pulse_pair_moments(
                iq, noise=noise, width_method=method, engine="compiled"
            )
            for name in a.data_vars:
                np.testing.assert_allclose(
                    a[name], b[name], rtol=1e-5, atol=1e-5, equal_nan=True
                )


def test_the_pulse_dimension_may_be_anywhere_and_threads_do_not_matter():
    iq, _ = sweep(n_az=3, n_rng=12, n_pulses=32)
    ref = level1.pulse_pair_moments(iq, noise=1.0)
    moved = iq.transpose("pulse", "range", "azimuth")
    out = level1.pulse_pair_moments(moved, noise=1.0, n_threads=1)
    np.testing.assert_allclose(out.transpose("azimuth", "range").VRADH, ref.VRADH)
    spec = level1.doppler_spectra(moved, n_threads=1)
    assert spec.SPECTRUM.dims[-1] == "velocity"
    np.testing.assert_allclose(
        spec.SPECTRUM.transpose("azimuth", "range", "velocity"),
        level1.doppler_spectra(iq).SPECTRUM,
        rtol=1e-6,
    )
    renamed = iq.rename(pulse="slow_time")
    out = level1.doppler_spectra(renamed, dim="slow_time")
    assert "velocity" in out.dims


def test_a_dataset_supplies_the_radar_constants_and_a_volume_is_mapped():
    iq, _ = sweep(n_az=2, n_rng=8, n_pulses=32)
    bare = iq.copy()
    bare.attrs = {}
    with pytest.raises(ValueError, match="wavelength"):
        level1.doppler_spectra(bare)
    with pytest.raises(ValueError, match="pulse repetition"):
        level1.doppler_spectra(bare, wavelength=WAVELENGTH)
    explicit = level1.doppler_spectra(bare, wavelength=WAVELENGTH, prt=PRT)
    np.testing.assert_allclose(explicit.SPECTRUM, level1.doppler_spectra(iq).SPECTRUM)
    # from variables of the sweep: radar_frequency and prt
    ds = xr.Dataset(
        {"IQ": bare, "radar_frequency": 299792458.0 / WAVELENGTH, "prt": PRT}
    )
    from_ds = level1.doppler_spectra(ds)
    np.testing.assert_allclose(from_ds.SPECTRUM, explicit.SPECTRUM)
    ds2 = xr.Dataset({"IQ": bare, "wavelength": WAVELENGTH}, attrs={"prt": PRT})
    np.testing.assert_allclose(level1.doppler_spectra(ds2).SPECTRUM, explicit.SPECTRUM)
    # a constant that varies over the data is refused
    ds3 = xr.Dataset(
        {"IQ": bare, "wavelength": WAVELENGTH, "prt": ("azimuth", [PRT, 2 * PRT])}
    )
    with pytest.raises(ValueError, match="varies"):
        level1.doppler_spectra(ds3)
    tree = volume(iq)
    out = level1.doppler_spectra(tree)
    assert set(out.children) == {"sweep_0", "sweep_1"}  # sweep_9 has no IQ
    mom = level1.spectral_moments(out)
    assert set(mom.children) == {"sweep_0", "sweep_1"}
    pp = level1.pulse_pair_moments(tree, noise=1.0)
    assert set(pp.children) == {"sweep_0", "sweep_1"}
    assert "VRADH" in pp["sweep_0"].dataset
    noise = level1.spectral_noise(out)
    assert set(noise.children) == {"sweep_0", "sweep_1"}
    # a volume where no sweep has the field
    empty = xr.DataTree.from_dict(
        {"/": xr.Dataset(), "sweep_0": xr.Dataset({"DBZH": (("a",), [1.0])})}
    )
    with pytest.raises(KeyError):
        level1.doppler_spectra(empty)


def test_spectral_functions_accept_a_dataarray_and_a_per_sweep_noise_tree():
    iq, _ = sweep(n_az=2, n_rng=6, n_pulses=32)
    spec = level1.doppler_spectra(iq)
    from_da = level1.spectral_moments(spec.SPECTRUM)
    ref = level1.spectral_moments(spec)
    np.testing.assert_allclose(from_da.VRADH, ref.VRADH)
    tree = level1.doppler_spectra(volume(iq))
    out = level1.spectral_moments(tree, noise=1.0)
    assert float(out["sweep_1"].dataset.NOISE_POWER.mean()) == pytest.approx(
        0.0, abs=1e-4
    )


def test_dask_arrays_stay_lazy_and_agree():
    pytest.importorskip("dask")
    iq, _ = sweep(n_az=4, n_rng=10, n_pulses=32)
    chunked = iq.chunk({"azimuth": 2, "range": 5, "pulse": 16})
    spectra = level1.doppler_spectra(chunked)
    assert spectra.SPECTRUM.chunks is not None
    ref = level1.doppler_spectra(iq)
    np.testing.assert_allclose(spectra.SPECTRUM.compute(), ref.SPECTRUM, rtol=1e-6)
    m = level1.spectral_moments(spectra)
    assert m.VRADH.chunks is not None
    np.testing.assert_allclose(
        m.VRADH.compute(), level1.spectral_moments(ref).VRADH, rtol=1e-5, equal_nan=True
    )
    n = level1.spectral_noise(spectra)
    assert n.NOISE_POWER.chunks is not None
    pp = level1.pulse_pair_moments(chunked, noise=chunked.isel(pulse=0).real * 0 + 1.0)
    assert pp.VRADH.chunks is not None
    np.testing.assert_allclose(
        pp.VRADH.compute(),
        level1.pulse_pair_moments(iq, noise=1.0).VRADH,
        rtol=1e-5,
        equal_nan=True,
    )


def test_inputs_are_not_changed_and_provenance_is_written():
    iq, _ = gates(n=20)
    before = iq.copy(deep=True)
    spectra = level1.doppler_spectra(iq)
    xr.testing.assert_identical(iq, before)
    prov = read_provenance(spectra)
    assert prov and "Windowed periodogram" in prov[0]["method"][0]
    registry = load_registry()
    assert all(ref in registry for ref in prov[0]["references"])
    assert "10.1109/PROC.1978.10837" in prov[0]["references"]
    prov = read_provenance(level1.pulse_pair_moments(iq))
    assert "doviak-zrnic-1993" in prov[0]["references"]
    assert "10.1109/TAES.1977.308467" in prov[0]["references"]
    prov = read_provenance(level1.spectral_moments(spectra))
    assert "10.1175/1520-0450(1974)013<0808:ODOTNL>2.0.CO;2" in prov[0]["references"]


# ---- dealiasing ------------------------------------------------------------


def ramp_ray(n_gates=120, span=2.4, width=2.0, seed=5, snr_db=25.0):
    truth = np.linspace(-span * NYQUIST, span * NYQUIST, n_gates)
    iq = level1.simulate_iq(
        truth[None, :],
        width=width,
        power=10 ** (0.1 * snr_db),
        noise=1.0,
        n_pulses=64,
        wavelength=WAVELENGTH,
        prt=PRT,
        seed=seed,
        dims=("azimuth", "range"),
    )
    return iq, truth


@pytest.mark.parametrize("engine", ENGINES)
def test_dealias_spectra_by_continuity_along_the_ray(engine):
    iq, truth = ramp_ray()
    # start inside the Nyquist interval: the ramp is -60..60 so shift to begin near 0
    truth = truth - truth[0]
    truth = truth * 0.8  # 0..96, steps well below the Nyquist velocity
    iq = level1.simulate_iq(
        truth[None, :],
        width=2.0,
        power=300.0,
        noise=1.0,
        n_pulses=64,
        wavelength=WAVELENGTH,
        prt=PRT,
        seed=5,
        dims=("azimuth", "range"),
    )
    out = level1.dealias_spectra(level1.doppler_spectra(iq), engine=engine)
    assert out.FOLD.dtype == np.int8
    err = out.VRADH.values[0] - truth
    assert np.mean(np.abs(err) < 4.0) > 0.97
    assert int(out.FOLD.max()) >= 1
    assert out.FOLD.attrs["comment"].startswith("each interval")
    assert out.VRADH.attrs["units"] == "m s-1"


@pytest.mark.parametrize("engine", ENGINES)
def test_dealias_spectra_with_a_reference_wind(engine):
    iq, truth = ramp_ray(span=2.4)
    spectra = level1.doppler_spectra(iq)
    reference = xr.DataArray(truth + 8.0 * np.sin(np.arange(truth.size)), dims="range")
    out = level1.dealias_spectra(spectra, reference=reference, engine=engine)
    err = out.VRADH.values[0] - truth
    assert np.mean(np.abs(err) < 4.0) > 0.97
    # a number as the reference unfolds every gate towards it
    flat = level1.dealias_spectra(spectra, reference=0.0, engine=engine)
    assert float(np.abs(flat.VRADH).max()) <= NYQUIST + 1e-3
    with pytest.raises(TypeError):
        level1.dealias_spectra(spectra, reference=xr.Dataset({"a": reference}))


def test_dealias_spectra_errors_and_options():
    iq, _ = gates(n=10)
    spectra = level1.doppler_spectra(iq)
    no_range = spectra.SPECTRUM.isel(range=0)
    with pytest.raises(ValueError, match="range"):
        level1.dealias_spectra(no_range)
    with pytest.raises(ValueError, match="max_gap"):
        level1.dealias_spectra(spectra, max_gap=-1)
    out = level1.dealias_spectra(
        spectra, region="all", n_sigma=3.0, channel="V", max_gap=0
    )
    assert "VRADV" in out
    tree = level1.dealias_spectra(
        level1.doppler_spectra(volume(sweep(n_az=2, n_rng=8)[0]))
    )
    assert set(tree.children) == {"sweep_0", "sweep_1"}
    per_sweep = level1.dealias_spectra(
        level1.doppler_spectra(volume(sweep(n_az=2, n_rng=8)[0])),
        reference={"sweep_0": 0.0, "sweep_1": 5.0},
    )
    assert set(per_sweep.children) == {"sweep_0", "sweep_1"}


# ---- errors ----------------------------------------------------------------


def test_argument_errors():
    iq, _ = gates(n=8, n_pulses=32)
    spectra = level1.doppler_spectra(iq)
    with pytest.raises(TypeError):
        level1.doppler_spectra(np.zeros(4))
    with pytest.raises(ValueError, match="window"):
        level1.doppler_spectra(iq, window="kaiser")
    with pytest.raises(ValueError, match="segments"):
        level1.doppler_spectra(iq, segments=0)
    with pytest.raises(ValueError, match="fewer than 3"):
        level1.doppler_spectra(iq, segments=16)
    with pytest.raises(ValueError, match="complex"):
        level1.doppler_spectra(iq.real)
    with pytest.raises(ValueError, match="no 'slow'"):
        level1.doppler_spectra(iq, dim="slow")
    with pytest.raises(ValueError, match="at least 3 pulses"):
        level1.doppler_spectra(iq.isel(pulse=slice(0, 2)))
    with pytest.raises(ValueError, match="engine"):
        level1.doppler_spectra(iq, engine="gpu")
    with pytest.raises(ValueError, match="positive"):
        level1.doppler_spectra(iq, wavelength=-1.0)
    with pytest.raises(ValueError, match="positive"):
        level1.doppler_spectra(iq, prt=0.0)
    with pytest.raises(ValueError, match="region"):
        level1.spectral_moments(spectra, region="some")
    with pytest.raises(ValueError, match="n_sigma"):
        level1.spectral_moments(spectra, n_sigma=-1.0)
    with pytest.raises(ValueError, match="channel"):
        level1.spectral_moments(spectra, channel="X")
    with pytest.raises(ValueError, match="noise"):
        level1.spectral_moments(spectra, noise=-1.0)
    with pytest.raises(ValueError, match="NOISE_POWER"):
        level1.spectral_moments(spectra, noise=xr.Dataset({"a": 1.0}))
    with pytest.raises(ValueError, match="n_average"):
        level1.spectral_moments(spectra, n_average=0.5)
    with pytest.raises(ValueError, match="min_bins"):
        level1.spectral_moments(spectra, min_bins=0)
    with pytest.raises(ValueError, match="min_bins"):
        level1.spectral_noise(spectra, min_bins=1000)
    with pytest.raises(ValueError, match="width_method"):
        level1.pulse_pair_moments(iq, width_method="r2r3")
    with pytest.raises(ValueError, match="spectral"):
        level1.pulse_pair_moments(iq, noise="other")
    with pytest.raises(ValueError, match="channel"):
        level1.pulse_pair_moments(iq, channel="X")
    with pytest.raises(KeyError):
        level1.spectral_moments(xr.Dataset({"x": (("a",), [1.0])}))
    bad = spectra.SPECTRUM.copy()
    with pytest.raises(ValueError, match="velocity"):
        level1.spectral_moments(bad, dim="nope")
    bad = bad.assign_coords(
        velocity=bad.velocity.values**3 * np.sign(bad.velocity.values)
    )
    with pytest.raises(ValueError, match="equal steps"):
        level1.spectral_moments(bad)
    shifted = spectra.SPECTRUM.assign_coords(velocity=spectra.velocity + 0.37)
    with pytest.raises(ValueError, match="zero velocity"):
        level1.spectral_moments(shifted)
    no_coord = spectra.SPECTRUM.drop_vars("velocity")
    with pytest.raises(ValueError, match="coordinate"):
        level1.spectral_moments(no_coord)
    with pytest.raises(ValueError, match="at least 3 bins"):
        level1.spectral_moments(spectra.SPECTRUM.isel(velocity=slice(0, 2)))


def test_the_compiled_engine_is_refused_when_the_kernels_are_missing(monkeypatch):
    iq, _ = gates(n=4, n_pulses=16)
    monkeypatch.setattr(sp, "HAS_COMPILED_SPECTRA", False)
    monkeypatch.setattr(sp, "HAS_COMPILED_MOMENTS", False)
    monkeypatch.setattr(sp, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError):
        level1.doppler_spectra(iq, engine="compiled")
    out = level1.doppler_spectra(iq)  # auto falls back to NumPy
    assert out.SPECTRUM.sizes["velocity"] == 16
    assert level1.spectral_moments(out).VRADH.shape == (4,)


def test_units_and_dtypes_of_the_products():
    iq, _ = sweep(n_az=2, n_rng=6, n_pulses=32)
    spectra = level1.doppler_spectra(iq)
    for ds in (
        spectra,
        level1.spectral_moments(spectra),
        level1.spectral_noise(spectra),
        level1.pulse_pair_moments(iq, noise=1.0),
        level1.dealias_spectra(spectra),
    ):
        for name, var in ds.data_vars.items():
            if var.dtype.kind == "f":
                assert var.dtype == np.float32, name
            assert "units" in var.attrs, name
            assert var.attrs["units"] in ("dB", "m s-1", "1"), name


def test_min_snr_masks_weak_gates_in_the_moments_and_the_unfolding():
    strong, _ = gates(n=20, snr_db=25.0, n_pulses=64, seed=1)
    weak = level1.simulate_iq(
        np.zeros(20),
        width=1.0,
        power=0.0,
        noise=1.0,
        n_pulses=64,
        wavelength=WAVELENGTH,
        prt=PRT,
        dims=("range",),
        seed=2,
    )
    mixed = xr.concat([strong, weak], dim="range")
    spectra = level1.doppler_spectra(mixed)
    plain = level1.spectral_moments(spectra)
    assert np.isfinite(plain.VRADH.values).sum() > 30  # noise spikes give velocities
    masked = level1.spectral_moments(spectra, min_snr=5.0)
    assert np.all(np.isfinite(masked.VRADH.values[:20]))
    assert np.all(np.isnan(masked.VRADH.values[20:]))
    assert np.all(np.isnan(masked.WRADH.values[20:])) and np.all(
        np.isnan(masked.KURT.values[20:])
    )
    np.testing.assert_allclose(masked.SNR, plain.SNR)  # the powers are kept
    # unfolding: the weak gates do not take part and come back as NaN, no fold
    out = level1.dealias_spectra(spectra, min_snr=5.0)
    assert np.all(np.isnan(out.VRADH.values[20:])) and np.all(out.FOLD.values[20:] == 0)
    kept = level1.dealias_spectra(spectra, min_snr=None)
    assert np.isfinite(kept.VRADH.values).sum() > 30


# ---- textbook: the variance of the estimators ------------------------------
#
# Doviak and Zrnic (1993) give the variance of the mean velocity from pulse
# pairs (Eq. 6.22a, with the high SNR limit 6.23), of the spectral mean velocity
# (Eq. 6.26) and of the pulse pair width (Eq. 6.30a), for a Gaussian spectrum of
# width sigma_vn normalised to the Nyquist interval, and a signal-to-noise
# ratio S/N. The simulated gates of Zrnic's method must reproduce them.


def normalised_width(width):
    return width / (2 * NYQUIST)


def var_velocity_pulse_pair(width, m, snr_db):
    """Eq. 6.22a for contiguous pairs (m pulses, T = Ts)."""
    sn = normalised_width(width)
    ns = 10 ** (-snr_db / 10)
    rho = np.exp(-2 * np.pi**2 * sn**2)
    rho2 = rho**4  # rho(2 Ts)
    return (
        WAVELENGTH**2
        / (32 * np.pi**2 * m * rho**2 * PRT**2)
        * ((1 - rho**2) / (2 * sn * np.sqrt(np.pi)) + ns**2 + 2 * ns * (1 - rho2))
    )


def var_velocity_spectral(width, m, snr_db):
    """Eq. 6.26: the spectral mean over the unambiguous interval."""
    sn = normalised_width(width)
    ns = 10 ** (-snr_db / 10)
    return (
        WAVELENGTH**2
        / (4 * m * PRT**2)
        * (sn / (4 * np.sqrt(np.pi)) + 2 * sn**2 * ns + ns**2 / 12)
    )


def var_width_pulse_pair(width, m, snr_db):
    """Eq. 6.30a for contiguous pairs, with the sum over the lags."""
    sn = normalised_width(width)
    ns = 10 ** (-snr_db / 10)
    r1 = np.exp(-2 * np.pi**2 * sn**2)

    def rho(k):
        return r1 ** (k**2.0)

    lag = np.arange(-(m - 1), m)
    braces = (
        2 * rho(lag) ** 2
        + rho(lag) ** 2 / r1**2
        + rho(lag) ** 2
        - 4 * rho(lag + 1) * rho(lag) / r1
    )
    total = np.sum(braces * (1 - np.abs(lag) / m))
    lead = WAVELENGTH**2 / (128 * m * np.pi**4 * sn**2 * r1**2 * PRT**2)
    first = 2 * (1 - 2 * r1**2 + r1**4) * ns
    second = (1 + 2 * r1**2) * ns**2
    return lead * (first + second + r1**2 * total)


@pytest.fixture(scope="module")
def textbook_gates():
    """6000 gates of 128 pulses, width 2 m/s, SNR 20 dB, random velocities."""
    truth = np.random.default_rng(5).uniform(-8, 8, 6000)
    iq = level1.simulate_iq(
        truth,
        width=2.0,
        power=100.0,
        noise=1.0,
        n_pulses=128,
        wavelength=WAVELENGTH,
        prt=PRT,
        dims=("range",),
        seed=3,
    )
    return iq, truth


@pytest.mark.parametrize("engine", ENGINES)
def test_the_pulse_pair_velocity_error_follows_Doviak_Zrnic_6_22a(
    textbook_gates, engine
):
    iq, truth = textbook_gates
    m = level1.pulse_pair_moments(iq, noise=1.0, engine=engine)
    sd = np.std(m.VRADH.values - truth)
    assert sd == pytest.approx(
        np.sqrt(var_velocity_pulse_pair(2.0, 128, 20.0)), rel=0.05
    )


def test_the_high_snr_limit_6_23():
    truth = np.random.default_rng(6).uniform(-8, 8, 6000)
    iq = level1.simulate_iq(
        truth,
        width=1.0,
        power=1e4,
        noise=1.0,
        n_pulses=128,
        wavelength=WAVELENGTH,
        prt=PRT,
        dims=("range",),
        seed=4,
    )
    sd = np.std(level1.pulse_pair_moments(iq, noise=1.0).VRADH.values - truth)
    # sigma_v lambda / (8 M Ts sqrt(pi))
    expected = 1.0 * WAVELENGTH / (8 * 128 * PRT * np.sqrt(np.pi))
    assert sd == pytest.approx(np.sqrt(expected), rel=0.05)


@pytest.mark.parametrize("engine", ENGINES)
def test_the_spectral_velocity_error_follows_Doviak_Zrnic_6_26(textbook_gates, engine):
    iq, truth = textbook_gates
    # the unwindowed periodogram of the book, every bin above the noise
    spectra = level1.doppler_spectra(iq, window="rect")
    m = level1.spectral_moments(spectra, region="all", n_sigma=3.0, engine=engine)
    sd = np.nanstd(m.VRADH.values - truth)
    assert sd == pytest.approx(np.sqrt(var_velocity_spectral(2.0, 128, 20.0)), rel=0.05)


def test_the_pulse_pair_width_error_follows_Doviak_Zrnic_6_30a(textbook_gates):
    iq, _ = textbook_gates
    m = level1.pulse_pair_moments(iq, noise=1.0)
    width = m.WRADH.values
    assert np.mean(width) == pytest.approx(2.0, rel=0.01)
    # a perturbation result: it holds to about 15 percent at 128 pulses
    assert np.std(width) == pytest.approx(
        np.sqrt(var_width_pulse_pair(2.0, 128, 20.0)), rel=0.2
    )
    assert np.std(width) > np.sqrt(
        var_width_pulse_pair(2.0, 128, 20.0)
    )  # 6.30a is a bit low
