"""
Attenuation and Scattering Coefficients
=======================================

Functions related to absorption, scattering, and extinction for spherical particles.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   absorption_coefficient
   extinction_coefficient
   k_complex
   scattering_coefficient

References
----------
- Doviak and Zrnic (1993). Doppler Radar and Weather Observations.
- Battan (1973). Radar Observations of the Atmosphere.
"""

import numpy as np


def k_complex(m):
    """
    Complex dielectric factor used in scattering calculations.

    Parameters
    ----------
    m : complex
        Complex refractive index of the medium.

    Returns
    -------
    complex
        Dielectric factor.
    """
    return (m**2 - 1) / (m**2 + 2)


def absorption_coefficient(diameter, wavelength, m):
    """
    Absorption coefficient Qa for a spherical particle.

    Parameters
    ----------
    diameter : float or array-like
        Particle diameter [m]
    wavelength : float
        Radar wavelength [m]
    m : complex
        Complex refractive index

    Returns
    -------
    float or array-like
        Absorption coefficient (unitless)
    """
    Km_im = np.imag(-1 * k_complex(m))
    return (np.pi**2 * np.asarray(diameter) ** 3 / wavelength) * Km_im


def scattering_coefficient(diameter, wavelength, m):
    """
    Scattering coefficient Qs for a spherical particle.

    Parameters
    ----------
    diameter : float or array-like
        Particle diameter [m]
    wavelength : float
        Radar wavelength [m]
    m : complex
        Complex refractive index

    Returns
    -------
    float or array-like
        Scattering coefficient (unitless)
    """
    Km_abs = np.abs(k_complex(m))
    return (2 * np.pi**5 * np.asarray(diameter) ** 6 / (3 * wavelength**4)) * (
        Km_abs**2
    )


def extinction_coefficient(diameter, wavelength, m):
    """
    Extinction coefficient Qe (absorption + scattering).

    Parameters
    ----------
    diameter : float or array-like
        Particle diameter [m]
    wavelength : float
        Radar wavelength [m]
    m : complex
        Complex refractive index

    Returns
    -------
    float or array-like
        Extinction coefficient (unitless)
    """
    Qa = absorption_coefficient(diameter, wavelength, m)
    Qs = scattering_coefficient(diameter, wavelength, m)
    return Qa + Qs


__all__ = [
    "absorption_coefficient",
    "extinction_coefficient",
    "k_complex",
    "scattering_coefficient",
]
