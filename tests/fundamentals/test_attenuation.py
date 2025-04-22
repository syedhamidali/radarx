import numpy as np
from radarx.fundamentals import attenuation


def test_k_complex_water():
    m = 7.14 - 2.89j
    k = attenuation.k_complex(m)
    assert isinstance(k, complex)
    assert k.real > 0


def test_absorption_coefficient():
    diam = 1e-3  # 1 mm
    wavelength = 0.03  # 3 cm
    m = 7.14 - 2.89j
    qa = attenuation.absorption_coefficient(diam, wavelength, m)
    assert np.isfinite(qa)
    assert qa > 0


def test_scattering_coefficient():
    diam = 1e-3  # 1 mm
    wavelength = 0.03  # 3 cm
    m = 7.14 - 2.89j
    qs = attenuation.scattering_coefficient(diam, wavelength, m)
    assert np.isfinite(qs)
    assert qs > 0


def test_extinction_coefficient():
    diam = np.array([1e-3, 2e-3])
    wavelength = 0.03
    m = 7.14 - 2.89j
    qe = attenuation.extinction_coefficient(diam, wavelength, m)
    assert np.all(np.isfinite(qe))
    assert np.all(qe > 0)
    assert np.allclose(
        qe,
        attenuation.absorption_coefficient(diam, wavelength, m)
        + attenuation.scattering_coefficient(diam, wavelength, m),
    )
