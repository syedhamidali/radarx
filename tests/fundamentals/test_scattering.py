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


def test_absorption_coefficient():
    radius = 0.0005  # 0.5 mm
    wavelength = 0.03  # 3 cm
    refractive_index = 8.5 + 1.0j
    coeff = scattering.absorption_coefficient(radius, wavelength, refractive_index)
    assert np.isfinite(coeff)
    assert coeff >= 0


def test_scattering_coefficient():
    radius = 0.0005
    wavelength = 0.03
    refractive_index = 8.5 + 1.0j
    coeff = scattering.scattering_coefficient(radius, wavelength, refractive_index)
    assert np.isfinite(coeff)
    assert coeff >= 0


def test_scattering_coefficient_zero_radius():
    radius = 0.0
    wavelength = 0.03
    refractive_index = 8.5 + 1.0j
    coeff = scattering.scattering_coefficient(radius, wavelength, refractive_index)
    assert coeff == 0.0


def test_scattering_coefficient_large_radius():
    radius = 0.01  # 10 mm
    wavelength = 0.03
    refractive_index = 8.5 + 1.0j
    coeff = scattering.scattering_coefficient(radius, wavelength, refractive_index)
    assert np.isfinite(coeff)
    assert coeff > 0


def test_scattering_coefficient_real_index_only():
    radius = 0.0005
    wavelength = 0.03
    refractive_index = 1.33 + 0.0j  # No imaginary part
    coeff = scattering.scattering_coefficient(radius, wavelength, refractive_index)
    assert np.isfinite(coeff)
    assert coeff >= 0


def test_extinction_coefficient():
    radius = 0.0005
    wavelength = 0.03
    refractive_index = 8.5 + 1.0j
    coeff = scattering.extinction_coefficient(radius, wavelength, refractive_index)
    assert np.isfinite(coeff)
    assert coeff >= 0
