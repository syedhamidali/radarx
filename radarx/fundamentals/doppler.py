"""
Doppler Radar Calculations
===========================

Functions related to Doppler radar performance and velocity limits.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   max_frequency
   nyquist_velocity
   unambiguous_range
   doppler_dilemma
   dual_prf_velocity

References
----------
- Rinehart, R. E. (1997). Radar for Meteorologists.
- Doviak, R. J., & Zrnić, D. S. (1993). Doppler Radar and Weather Observations.
"""

import numpy as np
from .constants import C


def max_frequency(prf):
    """
    Compute the maximum observable Doppler frequency.

    Parameters
    ----------
    prf : float or array-like
        Pulse repetition frequency [Hz]

    Returns
    -------
    float or array-like
        Maximum Doppler frequency [Hz]
    """
    return np.asarray(prf) / 2.0


def nyquist_velocity(prf, wavelength):
    """
    Compute Nyquist velocity (maximum unambiguous Doppler velocity).

    Parameters
    ----------
    prf : float or array-like
        Pulse repetition frequency [Hz]
    wavelength : float or array-like
        Radar wavelength [m]

    Returns
    -------
    float or array-like
        Nyquist velocity [m/s]
    """
    return np.asarray(prf) * np.asarray(wavelength) / 4.0


def unambiguous_range(prf):
    """
    Compute maximum unambiguous range for a given PRF.

    Parameters
    ----------
    prf : float or array-like
        Pulse repetition frequency [Hz]

    Returns
    -------
    float or array-like
        Maximum unambiguous range [m]
    """
    return C / (2.0 * np.asarray(prf))


def doppler_dilemma(value, wavelength):
    """
    Solve the Doppler dilemma equation: trade-off between unambiguous range and velocity.

    Parameters
    ----------
    value : float or array-like
        Either Nyquist velocity [m/s] or unambiguous range [m]
    wavelength : float
        Radar wavelength [m]

    Returns
    -------
    float or array-like
        Corresponding range or velocity [m or m/s]
    """
    return (C * wavelength / 8.0) / np.asarray(value)


def dual_prf_velocity(wavelength, prf1, prf2):
    """
    Compute Nyquist velocity using dual-PRF scheme.

    Parameters
    ----------
    wavelength : float
        Radar wavelength [m]
    prf1 : float
        First PRF [Hz]
    prf2 : float
        Second PRF [Hz]

    Returns
    -------
    float
        Maximum unambiguous velocity from dual-PRF scheme [m/s]

    Notes
    -----
    The sign of the result depends on the order of PRF values. Ensure prf1 > prf2 for positive velocity.
    """
    return wavelength / (4.0 * (1.0 / prf1 - 1.0 / prf2))


__all__ = [
    "doppler_dilemma",
    "dual_prf_velocity",
    "max_frequency",
    "nyquist_velocity",
    "unambiguous_range",
]
