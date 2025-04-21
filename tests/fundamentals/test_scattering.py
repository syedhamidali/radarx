import numpy as np
from radarx.fundamentals import scattering


def test_backscatter_cross_section():
    diameter = 0.001  # 1 mm
    wavelength = 0.03  # 3 cm
    sigma = scattering.backscatter_cross_section(diameter, wavelength)
    assert np.isfinite(sigma)
    assert sigma > 0


def test_normalized_backscatter_cross_section():
    diameter = 0.001
    wavelength = 0.03
    normalized = scattering.normalized_backscatter_cross_section(diameter, wavelength)
    assert np.isfinite(normalized)
    assert normalized > 0


def test_size_parameter():
    diameter = np.array([0.001, 0.002])
    wavelength = 0.05
    size_param = scattering.size_parameter(diameter, wavelength)
    assert size_param.shape == diameter.shape
    assert np.all(size_param > 0)
