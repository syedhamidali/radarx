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
