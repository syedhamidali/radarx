"""
Rayleigh Scattering Approximations
==================================

Functions for computing backscatter cross-sections and size parameters
under Rayleigh scattering assumptions.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   backscatter_cross_section
   normalized_backscatter_cross_section
   size_parameter

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


__all__ = [
    "backscatter_cross_section",
    "normalized_backscatter_cross_section",
    "size_parameter",
]
