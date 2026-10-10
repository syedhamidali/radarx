import numpy as np

from radarx.fundamentals import scattering


def test_backscatter_cross_section():
    diameter = 0.001  # 1 mm
    wavelength = 0.03  # 3 cm
    sigma = scattering.backscatter_cross_section(diameter, wavelength)
    assert np.isfinite(sigma)
    assert sigma > 0


def test_backscatter_cross_section_is_the_rayleigh_value():
    # D = 1 mm, 10 cm: pi^5 |K|^2 D^6 / lambda^4, |K|^2 = 0.927 -> 2.837e-12 m2
    sigma = scattering.backscatter_cross_section(1e-3, 0.1, dielectric=0.927)
    assert np.isclose(sigma, np.pi**5 * 0.927 * 1e-18 / 1e-4)
    assert np.isclose(sigma, 2.837e-12, rtol=1e-3)
    # the default |K|^2 = 0.93 is used as it is, not squared
    default = scattering.backscatter_cross_section(1e-3, 0.1)
    assert np.isclose(default, np.pi**5 * 0.93 * 1e-14)


def test_backscatter_cross_section_matches_k_complex():
    from radarx.fundamentals import attenuation

    m = np.sqrt(80.0 - 17.0j)  # water at about 3 GHz, n - jk convention
    k2 = np.abs(attenuation.k_complex(m)) ** 2
    sigma = scattering.backscatter_cross_section(1e-3, 0.1, dielectric=k2)
    assert np.isclose(sigma, np.pi**5 * k2 * 1e-14)


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


def test_absorption_coefficient_either_sign_convention():
    radius, wavelength = 0.0005, 0.03
    plus = scattering.absorption_coefficient(radius, wavelength, 8.5 + 1.0j)
    minus = scattering.absorption_coefficient(radius, wavelength, 8.5 - 1.0j)
    assert plus > 0
    assert np.isclose(plus, minus)
    assert np.isclose(
        scattering.extinction_coefficient(radius, wavelength, 8.5 - 1.0j),
        scattering.extinction_coefficient(radius, wavelength, 8.5 + 1.0j),
    )


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
