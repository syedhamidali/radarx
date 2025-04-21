"""
Radar-Derived Variables
=======================

Functions related to reflectivity, differential reflectivity, and radial velocity.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   reflectivity_factor
   differential_reflectivity
   linear_depolarization_ratio
   circular_depolarization_ratio
   radial_velocity

References
----------
- Rinehart, R. E. (1997). Radar for Meteorologists.
- Doviak and Zrnic (1993). Doppler Radar and Weather Observations.
"""

import numpy as np


def reflectivity_factor(p_return, radar_const, dielectric=0.93, range_m=1000.0):
    """
    Compute reflectivity factor Z [mm^6/m^3].

    Parameters
    ----------
    p_return : float or array-like
        Received power from target [W]
    radar_const : float
        Radar constant (unitless)
    dielectric : float
        Dielectric factor (default is 0.93 for water)
    range_m : float or array-like
        Range to target [m]

    Returns
    -------
    float or array-like
        Reflectivity factor Z [mm^6/m^3]
    """
    return (
        np.asarray(p_return) * np.asarray(range_m) ** 2 / (radar_const * dielectric**2)
    )


def differential_reflectivity(z_h, z_v):
    """
    Compute differential reflectivity ZDR [dB].

    Parameters
    ----------
    z_h : float or array-like
        Horizontal reflectivity [mm^6/m^3]
    z_v : float or array-like
        Vertical reflectivity [mm^6/m^3]

    Returns
    -------
    float or array-like
        ZDR in dB
    """
    return 10.0 * np.log10(np.asarray(z_h) / np.asarray(z_v))


def linear_depolarization_ratio(z_h, z_v):
    """
    Compute linear depolarization ratio LDR [dB].

    Parameters
    ----------
    z_h : float or array-like
        Horizontal reflectivity [mm^6/m^3]
    z_v : float or array-like
        Vertical reflectivity [mm^6/m^3]

    Returns
    -------
    float or array-like
        LDR in dB
    """
    return 10.0 * np.log10(np.asarray(z_v) / np.asarray(z_h))


def circular_depolarization_ratio(z_parallel, z_orthogonal):
    """
    Compute circular depolarization ratio CDR [dB].

    Parameters
    ----------
    z_parallel : float or array-like
        Power or reflectivity in the parallel channel [mm^6/m^3]
    z_orthogonal : float or array-like
        Power or reflectivity in the orthogonal channel [mm^6/m^3]

    Returns
    -------
    float or array-like
        CDR in dB
    """
    return 10.0 * np.log10(np.asarray(z_parallel) / np.asarray(z_orthogonal))


def radial_velocity(f_shift, wavelength):
    """
    Compute radial velocity from Doppler frequency shift.

    Parameters
    ----------
    f_shift : float or array-like
        Doppler frequency shift [Hz]
    wavelength : float
        Radar wavelength [m]

    Returns
    -------
    float or array-like
        Radial velocity [m/s]
    """
    return np.asarray(f_shift) * np.asarray(wavelength) / 2.0


__all__ = [
    "reflectivity_factor",
    "differential_reflectivity",
    "linear_depolarization_ratio",
    "circular_depolarization_ratio",
    "radial_velocity",
]
