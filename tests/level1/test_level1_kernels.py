#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the Level I kernels
=============================

The compiled kernels (``radarx.level1._spectra`` and ``_moments``) and their
NumPy twins must agree to rounding on random data and on the edge cases: a
length that is not a power of two (Bluestein), several segments, rows without
signal, rows with a NaN, spectra that wrap around the Nyquist velocity, and a
notch that covers the whole spectrum.
"""
import numpy as np
import pytest

from radarx.level1 import _moments_numpy as mnp
from radarx.level1 import _spectra_numpy as snp
from radarx.level1 import spectra as sp

try:
    from radarx.level1 import _moments, _spectra

    COMPILED = True
except ImportError:  # pragma: no cover
    COMPILED = False

compiled_only = pytest.mark.skipif(not COMPILED, reason="compiled kernels not built")
RTOL = 1e-9


def rows(n, k, seed=0, signal=True):
    """Exponentially distributed noise spectra, most with a Gaussian peak."""
    rng = np.random.default_rng(seed)
    s = rng.exponential(1.0, (n, k))
    if signal:
        centre = rng.uniform(0, k, n)
        width = rng.uniform(0.5, 4.0, n)
        j = np.arange(k)[None, :]
        d = (j - centre[:, None] + k / 2) % k - k / 2
        s += 60.0 * np.exp(-0.5 * (d / width[:, None]) ** 2)
    return s


def same(a, b, rtol=RTOL):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=1e-12, equal_nan=True)


# ---- spectra ---------------------------------------------------------------


@compiled_only
@pytest.mark.parametrize(
    "m, nseg", [(64, 1), (100, 1), (37, 1), (128, 4), (90, 3), (7, 1), (3, 1)]
)
def test_power_spectra_agree(m, nseg):
    rng = np.random.default_rng(m)
    x = rng.normal(size=(9, m)) + 1j * rng.normal(size=(9, m))
    length = m // nseg
    window, _ = sp.make_window("hann", length)
    for nt in (1, 3):
        same(
            _spectra.power_spectra(x, window, nseg, n_threads=nt),
            snp.power_spectra(x, window, nseg),
            1e-11,
        )


def test_power_spectra_match_the_fft_and_add_up_to_the_power():
    rng = np.random.default_rng(1)
    m = 50
    x = rng.normal(size=(4, m)) + 1j * rng.normal(size=(4, m))
    window = np.ones(m)
    ps = snp.power_spectra(x, window, 1)
    # Parseval with the rectangular window: the bins add up to the mean power
    same(ps.sum(axis=1), np.mean(np.abs(x) ** 2, axis=1), 1e-12)
    # bin j is the FFT bin (m/2 - j) mod m
    ref = np.abs(np.fft.fft(x, axis=1)) ** 2 / (m * m)
    same(ps, ref[:, (m // 2 - np.arange(m)) % m], 1e-12)


@compiled_only
def test_power_spectra_errors():
    x = np.ones((2, 8), dtype=complex)
    w = np.ones(8)
    for module in (_spectra, snp):
        with pytest.raises(ValueError):
            module.power_spectra(np.ones(8, dtype=complex), w, 1)
        with pytest.raises(ValueError):
            module.power_spectra(x, w, 0)
        with pytest.raises(ValueError):
            module.power_spectra(x, w, 8)  # one pulse per segment
        with pytest.raises(ValueError):
            module.power_spectra(x, np.ones(5), 1)
        with pytest.raises(ValueError):
            module.power_spectra(x, np.zeros(8), 1)


@compiled_only
def test_lag_sums_agree_and_errors():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(11, 40)) + 1j * rng.normal(size=(11, 40))
    same(_spectra.lag_sums(x), snp.lag_sums(x), 1e-12)
    for module in (_spectra, snp):
        with pytest.raises(ValueError):
            module.lag_sums(x[:, :2])
        with pytest.raises(ValueError):
            module.lag_sums(x[0])


def test_lag_sums_of_a_tone():
    n = np.arange(32)
    x = 2.0 * np.exp(1j * 0.7 * n)[None, :]
    p0, r1r, r1i, r2r, r2i = snp.lag_sums(x)[0]
    assert p0 == pytest.approx(4.0)
    assert np.angle(r1r + 1j * r1i) == pytest.approx(0.7)
    assert np.angle(r2r + 1j * r2i) == pytest.approx(1.4)
    assert abs(r1r + 1j * r1i) == pytest.approx(4.0)


@compiled_only
@pytest.mark.parametrize("order", [0, 1, 3])
def test_regression_filter_agrees_and_removes_polynomials(order):
    from radarx.level1.clutter import polynomial_basis

    rng = np.random.default_rng(3)
    m = 48
    basis = polynomial_basis(order, m)
    np.testing.assert_allclose(basis.T @ basis, np.eye(order + 1), atol=1e-12)
    x = rng.normal(size=(7, m)) + 1j * rng.normal(size=(7, m))
    y_c, rem_c = _spectra.regression_filter(x, basis, n_threads=2)
    y_n, rem_n = snp.regression_filter(x, basis)
    same(y_c, y_n, 1e-12)
    same(rem_c, rem_n, 1e-12)
    # a polynomial of the same degree is removed entirely
    t = np.linspace(-1, 1, m)
    poly = sum((k + 1) * (1 + 1j) * t**k for k in range(order + 1))[None, :]
    y, removed = snp.regression_filter(poly, basis)
    assert np.abs(y).max() < 1e-10
    assert removed[0] == pytest.approx(np.mean(np.abs(poly) ** 2))


@compiled_only
def test_regression_filter_errors():
    x = np.ones((2, 10), dtype=complex)
    for module in (_spectra, snp):
        with pytest.raises(ValueError):
            module.regression_filter(x, np.ones((9, 2)))
        with pytest.raises(ValueError):
            module.regression_filter(x[0], np.ones((10, 2)))


# ---- noise and moments -----------------------------------------------------


@compiled_only
@pytest.mark.parametrize("k", [16, 33, 64])
@pytest.mark.parametrize("n_avg", [1.0, 4.0])
def test_noise_rows_agree(k, n_avg):
    s = rows(200, k, seed=k)
    s[3, 2] = np.nan
    s[4] = 2.5  # flat spectrum: zero variance
    min_bins = max(3, k // 8)
    a = _moments.noise_rows(s, n_avg, min_bins, n_threads=2)
    b = mnp.noise_rows(s, n_avg, min_bins)
    same(a, b)
    assert np.isnan(a[3])
    assert a[4] == pytest.approx(2.5)


def test_noise_of_noise_only_spectra_is_near_its_mean():
    rng = np.random.default_rng(5)
    s = rng.exponential(1.0, (3000, 64))
    noise = mnp.noise_rows(s, 1.0, 8)
    # the objective method trims the strongest bins, so it is biased low
    assert 0.75 < np.mean(noise) < 1.0
    averaged = rng.gamma(8.0, 1.0 / 8.0, (3000, 64))
    noise = mnp.noise_rows(averaged, 8.0, 8)
    assert 0.9 < np.mean(noise) < 1.02


@compiled_only
@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("k", [16, 33, 64])
def test_moments_agree(mode, k):
    s = rows(300, k, seed=10 + k)
    s[0, :] = 0.5  # no signal
    s[1, 4] = np.nan
    s[2, :] = 50.0  # every bin above the threshold: wraps all the way round
    s[5, :] = 1.0
    s[5, k // 2] = 100.0  # a single bin
    noise = mnp.noise_rows(np.where(np.isfinite(s), s, 1.0), 1.0, 3)
    noise[6] = np.nan
    noise[2] = 1.0
    args = (noise, 1.0, 1.0, mode, 0.37, 0.33)
    a = _moments.moments(s, *args, n_threads=3)
    b = mnp.moments(s, *args)
    same(a, b, 1e-8)
    assert np.isnan(a[1, 2])  # NaN bin
    assert a[0, 6] == 0 and np.isnan(a[0, 2])  # no signal
    assert a[5, 6] == 1 and a[5, 3] == 0.0 and np.isnan(a[5, 4])  # one bin
    assert np.isnan(a[6, 1])  # NaN noise


@compiled_only
def test_moments_of_a_wrapped_spectrum():
    # a Gaussian centred on the edge of the axis: the peak mode follows it round
    k = 64
    j = np.arange(k)
    d = (j - 0.0 + k / 2) % k - k / 2  # centred on bin 0 (velocity -va)
    s = 1.0 + 200.0 * np.exp(-0.5 * (d / 2.0) ** 2)
    out = mnp.moments(s[None, :], np.array([1.0]), 1.0, 1.0, 0, 1.0, 0.0)[0]
    # the mean position is bin 0 - k/2 = -32 in velocity units of 1
    assert out[2] == pytest.approx(-32.0, abs=1e-6)
    assert out[3] == pytest.approx(2.0, rel=0.02)
    same(_moments.moments(s[None, :], np.array([1.0]), 1.0, 1.0, 0, 1.0, 0.0)[0], out)


def test_moments_errors():
    s = rows(4, 16)
    noise = np.ones(4)
    mods = [mnp] + ([_moments] if COMPILED else [])
    for module in mods:
        with pytest.raises(ValueError):
            module.moments(s[0], noise, 1.0, 1.0, 0, 1.0, 0.0)
        with pytest.raises(ValueError):
            module.moments(s, noise[:2], 1.0, 1.0, 0, 1.0, 0.0)
        with pytest.raises(ValueError):
            module.moments(s, noise, 1.0, 1.0, 2, 1.0, 0.0)
        with pytest.raises(ValueError):
            module.noise_rows(s, 1.0, 0)
        with pytest.raises(ValueError):
            module.noise_rows(s, 1.0, 17)
        with pytest.raises(ValueError):
            module.noise_rows(s[0], 1.0, 3)


# ---- notch -----------------------------------------------------------------


@compiled_only
@pytest.mark.parametrize("fill", [0, 1, 2])
@pytest.mark.parametrize("half_width", [0, 1, 3])
@pytest.mark.parametrize("k", [16, 33, 64])
def test_notch_agrees(fill, half_width, k):
    s = rows(250, k, seed=20 + k)
    # clutter spike at zero velocity and a weather peak that touches the notch
    s[:, k // 2] += 400.0
    s[:60, k // 2 - 1 : k // 2 + 2] += 80.0
    s[7, 3] = np.nan
    noise = mnp.noise_rows(np.where(np.isfinite(s), s, 1.0), 1.0, 3)
    noise[9] = np.nan
    args = (noise, 1.0, 1.0, half_width, fill)
    a, ra = _moments.notch(s, *args, n_threads=2)
    b, rb = mnp.notch(s, *args)
    same(a, b, 1e-7)
    same(ra, rb, 1e-7)
    assert np.isnan(ra[7]) and np.isnan(ra[9])
    np.testing.assert_array_equal(a[7], s[7])  # a row with a NaN is not touched


@compiled_only
@pytest.mark.parametrize("fill", [0, 1, 2])
def test_notch_covering_the_whole_spectrum(fill):
    s = rows(10, 8, seed=1)
    noise = np.ones(10)
    args = (noise, 1.0, 1.0, 8, fill)
    a, _ = _moments.notch(s, *args)
    b, _ = mnp.notch(s, *args)
    same(a, b)
    np.testing.assert_allclose(a, 1.0)  # everything becomes the noise level


def test_notch_at_the_edge_uses_the_available_side():
    # k = 4: bin 2 is the zero velocity bin
    s = np.array([[5.0, 6.0, 70.0, 8.0]])
    one = np.array([1.0])
    for module in [mnp] + ([_moments] if COMPILED else []):
        # half_width 0: the bin between its neighbours
        out, removed = module.notch(s, one, 1.0, 1.0, 0, 1)
        np.testing.assert_allclose(out[0], [5.0, 6.0, 7.0, 8.0])
        assert removed[0] == pytest.approx(63.0)
        # half_width 1 covers bins 1..3: only the left neighbour is left
        out, _ = module.notch(s, one, 1.0, 1.0, 1, 1)
        np.testing.assert_allclose(out[0], [5.0, 5.0, 5.0, 5.0])
        # a notch that reaches the left edge uses the right neighbour
        s2 = np.array([[5.0, 6.0, 7.0, 8.0, 9.0, 4.0]])
        out, _ = module.notch(s2, one, 1.0, 1.0, 3, 1)  # covers bins 0..5
        np.testing.assert_allclose(out, 1.0)
        out, _ = module.notch(s2, one, 1.0, 1.0, 2, 1)  # bins 1..5, left is 5
        np.testing.assert_allclose(out[0], [5.0] * 6)
        s3 = np.array([[5.0, 6.0, 7.0, 8.0, 9.0, 4.0, 3.0, 2.0]])
        out, _ = module.notch(s3, one, 1.0, 1.0, 4, 1)  # bins 0..7, no side
        np.testing.assert_allclose(out, 1.0)
        s4 = np.array([[9.0, 6.0, 7.0, 8.0, 9.0, 4.0, 3.0, 2.0, 1.5, 2.5]])
        out, _ = module.notch(s4, one, 1.0, 1.0, 5, 1)  # bins 0..9
        np.testing.assert_allclose(out, 1.0)
        # a notch that reaches the right edge uses the left neighbour
        s5 = np.array([[5.0, 6.0, 7.0, 8.0, 9.0, 4.0]])
        out, _ = module.notch(
            s5, one, 1.0, 1.0, 2, 1
        )  # k = 6, zero at bin 3: bins 1..5
        assert out[0, 5] == pytest.approx(5.0)


def test_gaussian_fill_recovers_a_gaussian_under_the_notch():
    k = 64
    j = np.arange(k)
    truth = 1.0 + 300.0 * np.exp(-0.5 * ((j - (k // 2 + 2)) / 3.0) ** 2)
    corrupted = truth.copy()
    corrupted[k // 2 - 1 : k // 2 + 2] += 5000.0  # clutter in the notch
    for module in [mnp] + ([_moments] if COMPILED else []):
        out, removed = module.notch(corrupted[None, :], np.array([1.0]), 1.0, 1.0, 1, 2)
        np.testing.assert_allclose(
            out[0, k // 2 - 1 : k // 2 + 2], truth[k // 2 - 1 : k // 2 + 2], rtol=1e-3
        )
        assert removed[0] == pytest.approx(3 * 5000.0, rel=1e-2)


def test_notch_errors():
    s = rows(4, 16)
    noise = np.ones(4)
    mods = [mnp] + ([_moments] if COMPILED else [])
    for module in mods:
        with pytest.raises(ValueError):
            module.notch(s[0], noise, 1.0, 1.0, 1, 1)
        with pytest.raises(ValueError):
            module.notch(s, noise[:3], 1.0, 1.0, 1, 1)
        with pytest.raises(ValueError):
            module.notch(s, noise, 1.0, 1.0, -1, 1)
        with pytest.raises(ValueError):
            module.notch(s, noise, 1.0, 1.0, 1, 3)


# ---- unfolding -------------------------------------------------------------


def ray_velocities(seed=0, n=40, g=60, va=10.0):
    rng = np.random.default_rng(seed)
    true = np.cumsum(rng.uniform(-3.0, 3.0, (n, g)), axis=1) * 0.5
    true += rng.uniform(-30, 30, (n, 1))
    wrapped = (true + va) % (2 * va) - va
    return true, wrapped


@compiled_only
@pytest.mark.parametrize("with_ref", [False, True])
def test_unfold_agrees_and_restores_the_continuous_field(with_ref):
    va = 10.0
    true, wrapped = ray_velocities(va=va)
    wrapped[3, 10:14] = np.nan
    wrapped[5, 20] = np.nan
    ref = np.empty((0, 0))
    if with_ref:
        ref = true + np.random.default_rng(1).uniform(-4, 4, true.shape)
        ref[:, ::7] = np.nan  # where there is no reference, continuity is used
    a, fa = _moments.unfold(wrapped, ref, va, 3, n_threads=2)
    b, fb = mnp.unfold(wrapped, ref, va, 3)
    same(a, b)
    np.testing.assert_array_equal(fa, fb)
    assert fa.dtype == np.int32
    ok = np.isfinite(wrapped)
    if with_ref:
        # a good reference restores the truth at the gates that have one
        has = ok & np.isfinite(ref)
        np.testing.assert_allclose(a[has], true[has], atol=1e-9)
    else:
        # continuity restores the field up to a whole number of intervals per ray
        # where the first gate was already folded; gaps of up to 3 gates link
        err = (a - true)[ok]
        assert np.all(np.abs(err / (2 * va) - np.round(err / (2 * va))) < 1e-9)
    assert np.array_equal(np.isfinite(a), ok)


def test_unfold_gap_restarts_in_the_principal_interval():
    va = 10.0
    v = np.array([[9.0, -9.0, np.nan, np.nan, -9.0, 9.0]])
    for module in [mnp] + ([_moments] if COMPILED else []):
        long_gap, _ = module.unfold(v, np.empty((0, 0)), va, 3)
        np.testing.assert_allclose(long_gap[0], [9.0, 11.0, np.nan, np.nan, 11.0, 9.0])
        short_gap, folds = module.unfold(v, np.empty((0, 0)), va, 1)
        np.testing.assert_allclose(
            short_gap[0], [9.0, 11.0, np.nan, np.nan, -9.0, -11.0]
        )
        assert folds[0].tolist() == [0, 1, 0, 0, 0, -1]


def test_unfold_errors():
    v = np.zeros((2, 5))
    for module in [mnp] + ([_moments] if COMPILED else []):
        with pytest.raises(ValueError):
            module.unfold(v[0], np.empty((0, 0)), 10.0, 3)
        with pytest.raises(ValueError):
            module.unfold(v, np.zeros((2, 4)), 10.0, 3)
        with pytest.raises(ValueError):
            module.unfold(v, np.empty((0, 0)), 0.0, 3)
