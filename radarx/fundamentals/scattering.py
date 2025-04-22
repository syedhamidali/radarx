"""
Rayleigh Scattering Approximations
==================================

Functions for computing backscatter cross-sections, size parameters,
and absorption, scattering, and extinction coefficients under Rayleigh
scattering assumptions.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   backscatter_cross_section
   normalized_backscatter_cross_section
   size_parameter
   absorption_coefficient
   scattering_coefficient
   extinction_coefficient

References
----------
- Rinehart (1997). Radar for Meteorologists.
- Battan (1973). Radar Observations of the Atmosphere.
"""

import numpy as np


def backscatter_cross_section(diameter, wavelength, dielectric=0.93):
    """
    Rayleigh backscatter cross-section for a water sphere.

    Parameters
    ----------
    diameter : float or array-like
        Drop diameter [m]
    wavelength : float
        Radar wavelength [m]
    dielectric : float
        Dielectric factor (default: 0.93 for water)

    Returns
    -------
    float or array-like
        Backscatter cross-section [m^2]
    """
    d = np.asarray(diameter)
    return (np.pi**5 * dielectric**2 * d**6) / wavelength**4


def normalized_backscatter_cross_section(diameter, wavelength, dielectric=0.93):
    """
    Normalize Rayleigh backscatter cross-section by projected area.

    Parameters
    ----------
    diameter : float or array-like
        Drop diameter [m]
    wavelength : float
        Radar wavelength [m]
    dielectric : float, optional
        Dielectric factor (default: 0.93)

    Returns
    -------
    float or array-like
        Normalized cross-section [unitless]
    """
    sigma = backscatter_cross_section(diameter, wavelength, dielectric)
    area = np.pi * (np.asarray(diameter) / 2.0) ** 2
    return sigma / area


def size_parameter(diameter, wavelength):
    """
    Calculate size parameter alpha = π * D / λ.

    Parameters
    ----------
    diameter : float or array-like
        Drop diameter [m]
    wavelength : float
        Radar wavelength [m]

    Returns
    -------
    float or array-like
        Size parameter (unitless)
    """
    return np.pi * np.asarray(diameter) / wavelength


def _size_parameter(radius, wavelength):
    """
    Calculate size parameter x = 2π * radius / wavelength.

    Parameters
    ----------
    radius : float or array-like
        Particle radius [m]
    wavelength : float
        Radar wavelength [m]

    Returns
    -------
    float or array-like
        Size parameter (unitless)
    """
    return 2 * np.pi * np.asarray(radius) / wavelength


def _complex_ratio(refractive_index):
    """
    Calculate the complex ratio (m^2 - 1) / (m^2 + 2).

    Parameters
    ----------
    refractive_index : complex
        Complex refractive index of the particle

    Returns
    -------
    complex or array-like of complex
        Complex ratio used in scattering calculations
    """
    m = refractive_index
    m2 = m * m
    return (m2 - 1) / (m2 + 2)


def absorption_coefficient(radius, wavelength, refractive_index):
    """
    Compute Rayleigh absorption coefficient.

    Parameters
    ----------
    radius : float or array-like
        Particle radius [m]
    wavelength : float
        Radar wavelength [m]
    refractive_index : complex
        Complex refractive index of the particle

    Returns
    -------
    float or array-like
        Absorption efficiency (Qa)
    """
    x = 2 * np.pi * radius / wavelength
    m = refractive_index
    m2 = m * m
    return np.maximum(4 * x * np.imag((m2 - 1) / (m2 + 2)), 0.0)


def scattering_coefficient(radius, wavelength, refractive_index):
    """
    Compute Rayleigh scattering coefficient.

    Parameters
    ----------
    radius : float or array-like
        Particle radius [m]
    wavelength : float
        Radar wavelength [m]
    refractive_index : complex
        Complex refractive index of the particle

    Returns
    -------
    float or array-like
        Scattering efficiency (Qs)
    """
    x = _size_parameter(radius, wavelength)
    ratio = _complex_ratio(refractive_index)
    return (8 / 3) * x**4 * np.abs(ratio) ** 2


def extinction_coefficient(radius, wavelength, refractive_index):
    """
    Compute Rayleigh extinction coefficient (Qa + Qs).

    Parameters
    ----------
    radius : float or array-like
        Particle radius [m]
    wavelength : float
        Radar wavelength [m]
    refractive_index : complex
        Complex refractive index of the particle

    Returns
    -------
    float or array-like
        Extinction efficiency (Qe)
    """
    return np.maximum(
        absorption_coefficient(radius, wavelength, refractive_index)
        + scattering_coefficient(radius, wavelength, refractive_index),
        0.0,
    )


__all__ = [
    "backscatter_cross_section",
    "normalized_backscatter_cross_section",
    "size_parameter",
    "absorption_coefficient",
    "scattering_coefficient",
    "extinction_coefficient",
]
