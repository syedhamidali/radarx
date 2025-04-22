import numpy as np
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
    assert np.isclose(np.abs(v_dual), 50.0, rtol=1e-6)


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
    frequency = 3e9  # Hz
    vr = 15.0  # m/s
    expected = frequency * (np.sqrt((1 + vr / C) / (1 - vr / C)) - 1)
    result = doppler.doppler_frequency_shift(frequency, vr, relativistic=True)
    assert np.isclose(result, expected)


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
    expected = frequency * (np.sqrt((1 + vr / C) / (1 - vr / C)) - 1)
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
    expected = frequency * (np.sqrt((1 + vr / C) / (1 - vr / C)) - 1)
    from radarx.fundamentals.doppler import _doppler_shift_relativistic

    assert np.isclose(_doppler_shift_relativistic(frequency, vr), expected)
