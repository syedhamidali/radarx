import numpy as np
import pytest

from radarx.fundamentals import doppler
from radarx.fundamentals.constants import C


def test_nyquist_velocity():
    prf = 1000  # Hz
    wavelength = 0.03  # 3 cm
    expected = prf * wavelength / 4
    assert np.isclose(doppler.nyquist_velocity(prf, wavelength), expected)


def test_max_unambiguous_range():
    prf = 1200  # Hz
    expected = C / (2 * prf)
    assert np.isclose(doppler.unambiguous_range(prf), expected, rtol=1e-6)


def test_doppler_dilemma():
    vmax = 15.0  # m/s
    wavelength = 0.032  # 3.2 cm
    rmax = doppler.doppler_dilemma(vmax, wavelength)
    assert rmax > 0


def test_dual_prf_velocity():
    prf1 = 1000  # Hz
    prf2 = 800  # Hz
    wavelength = 0.05  # 5 cm
    v_dual = doppler.dual_prf_velocity(wavelength, prf1, prf2)
    assert np.isclose(v_dual, 50.0, rtol=1e-6)


def test_dual_prf_velocity_is_positive_for_either_order():
    # lambda / (4 (T2 - T1)) with T = 1/PRF: 0.1 / (4 (1/750 - 1/1000)) = 75
    assert np.isclose(doppler.dual_prf_velocity(0.1, 1000, 750), 75.0)
    assert np.isclose(doppler.dual_prf_velocity(0.1, 750, 1000), 75.0)
    v = doppler.dual_prf_velocity(0.1, np.array([1000.0, 500.0]), 750.0)
    np.testing.assert_allclose(v, [75.0, 37.5])


def test_max_frequency():
    prf = 2000  # Hz
    expected = prf / 2
    assert np.isclose(doppler.max_frequency(prf), expected)


def test_doppler_frequency_shift_basic():
    frequency = 3e9  # Hz
    vr = 15.0  # m/s
    expected = 2 * frequency * vr / C
    assert np.isclose(doppler.doppler_frequency_shift(frequency, vr), expected)


def test_doppler_frequency_shift_exact():
    frequency = 3e9  # Hz
    vr = 15.0  # m/s
    expected = 2 * frequency * vr / (C - vr)
    result = doppler.doppler_frequency_shift(frequency, vr, exact=True)
    assert np.isclose(result, expected)


def test_doppler_frequency_shift_relativistic():
    # two-way: the reflected frequency is f (1 + b) / (1 - b)
    frequency = 3e9  # Hz
    vr = 15.0  # m/s
    beta = vr / C
    expected = frequency * ((1 + beta) / (1 - beta) - 1)
    result = doppler.doppler_frequency_shift(frequency, vr, relativistic=True)
    assert np.isclose(result, expected)


def test_doppler_frequency_shift_relativistic_is_the_two_way_shift():
    # 5.6 GHz, 10 m/s: 373.592 Hz, not the one-way 186.796 Hz
    f, v = 5.6e9, 10.0
    relativistic = doppler.doppler_frequency_shift(f, v, relativistic=True)
    assert relativistic == pytest.approx(373.592, abs=1e-3)
    assert relativistic == pytest.approx(
        doppler.doppler_frequency_shift(f, v, exact=True), rel=1e-9
    )
    assert relativistic == pytest.approx(
        doppler.doppler_frequency_shift(f, v), rel=1e-6
    )
    # receding target: negative shift of about the same size
    receding = doppler.doppler_frequency_shift(f, -v, relativistic=True)
    assert receding == pytest.approx(-373.592, abs=1e-3)


def test_doppler_frequency_shift_array_input():
    frequency = np.array([3e9, 3e9])
    vr = np.array([15.0, -15.0])
    expected = 2 * frequency * vr / C
    result = doppler.doppler_frequency_shift(frequency, vr)
    assert np.allclose(result, expected)


def test_doppler_shift_relativistic_overrides_exact():
    frequency = 3e9
    vr = 15.0
    result = doppler.doppler_frequency_shift(
        frequency, vr, exact=True, relativistic=True
    )
    beta = vr / C
    expected = frequency * ((1 + beta) / (1 - beta) - 1)
    assert np.isclose(result, expected)


def test__doppler_shift_basic():
    frequency = 3e9
    vr = 10
    expected = 2 * frequency * vr / C
    from radarx.fundamentals.doppler import _doppler_shift_basic

    assert np.isclose(_doppler_shift_basic(frequency, vr), expected)


def test__doppler_shift_exact():
    frequency = 3e9
    vr = 10
    expected = 2 * frequency * vr / (C - vr)
    from radarx.fundamentals.doppler import _doppler_shift_exact

    assert np.isclose(_doppler_shift_exact(frequency, vr), expected)


def test__doppler_shift_relativistic():
    frequency = 3e9
    vr = 10
    expected = frequency * ((1 + vr / C) / (1 - vr / C) - 1)
    from radarx.fundamentals.doppler import _doppler_shift_relativistic

    assert np.isclose(_doppler_shift_relativistic(frequency, vr), expected)
