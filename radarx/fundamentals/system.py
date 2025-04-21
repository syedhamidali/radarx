"""
Radar System Characteristics
============================

Core functions related to radar system components like antenna gain, pulse parameters,
wavelength/frequency conversion, and radar constants.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   antenna_gain
   ant_eff_area
   frequency
   frequency_from_wavelength
   power_return_target
   pulse_duration
   pulse_duration_from_length
   pulse_length
   pulse_length_from_duration
   size_param
   wavelength
   wavelength_from_frequency
"""

import numpy as np
from .constants import C


def antenna_gain(p_beam, p_iso):
    """
    Compute antenna gain in dB from power ratio.

    Parameters
    ----------
    p_beam : float or array-like
        Power on the beam axis [W]
    p_iso : float or array-like
        Power from an isotropic antenna [W]

    Returns
    -------
    float or array-like
        Antenna gain in dB.
    """
    return 10.0 * np.log10(np.asarray(p_beam) / np.asarray(p_iso))


def frequency_from_wavelength(wavelength):
    """
    Compute frequency from radar wavelength.

    Parameters
    ----------
    wavelength : float or array-like
        Radar wavelength [m]

    Returns
    -------
    float or array-like
        Frequency [Hz]
    """
    return C / np.asarray(wavelength)


def wavelength_from_frequency(freq):
    """
    Compute wavelength from radar frequency.

    Parameters
    ----------
    freq : float or array-like
        Frequency [Hz]

    Returns
    -------
    float or array-like
        Wavelength [m]
    """
    return C / np.asarray(freq)


def pulse_length_from_duration(pulse_duration):
    """
    Compute physical pulse length from pulse duration.

    Parameters
    ----------
    pulse_duration : float or array-like
        Pulse duration [s]

    Returns
    -------
    float or array-like
        Pulse length [m]
    """
    return C * np.asarray(pulse_duration) / 2.0


def pulse_duration_from_length(pulse_length):
    """
    Compute pulse duration from physical pulse length.

    Parameters
    ----------
    pulse_length : float or array-like
        Pulse length [m]

    Returns
    -------
    float or array-like
        Pulse duration [s]
    """
    return 2.0 * np.asarray(pulse_length) / C


def ant_eff_area(gain_dbi, wavelength):
    """
    Compute effective antenna area from gain and wavelength.

    Parameters
    ----------
    gain_dbi : float
        Antenna gain [dBi]
    wavelength : float
        Wavelength [m]

    Returns
    -------
    float
        Effective antenna area [m^2]
    """
    gain_linear = 10 ** (gain_dbi / 10)
    return (gain_linear * wavelength**2) / (4 * np.pi)


def size_param(diameter, wavelength):
    """
    Compute size parameter alpha.

    Parameters
    ----------
    diameter : float
        Diameter of the particle [m]
    wavelength : float
        Radar wavelength [m]

    Returns
    -------
    float
        Size parameter alpha
    """
    return np.pi * diameter / wavelength


def power_return_target(power_tx, gain_dbi, wavelength, sigma, range_m):
    """
    Compute received power from a radar target using the radar equation.

    Parameters
    ----------
    power_tx : float
        Transmitted power [W]
    gain_dbi : float
        Antenna gain [dBi]
    wavelength : float
        Radar wavelength [m]
    sigma : float
        Radar cross-section [m^2]
    range_m : float
        Range to target [m]

    Returns
    -------
    float
        Received power [W]
    """
    gain_linear = 10 ** (gain_dbi / 10)
    return (power_tx * gain_linear**2 * wavelength**2 * sigma) / (
        (4 * np.pi) ** 3 * range_m**4
    )


def frequency(wavelength):
    """
    Alias for frequency_from_wavelength.
    """
    return frequency_from_wavelength(wavelength)


def wavelength(freq):
    """
    Alias for wavelength_from_frequency.
    """
    return wavelength_from_frequency(freq)


def pulse_length(pulse_duration):
    """
    Alias for pulse_length_from_duration.
    """
    return pulse_length_from_duration(pulse_duration)


def pulse_duration(pulse_length_val):
    """
    Alias for pulse_duration_from_length.
    """
    return pulse_duration_from_length(pulse_length_val)


__all__ = [
    "ant_eff_area",
    "antenna_gain",
    "frequency",
    "frequency_from_wavelength",
    "power_return_target",
    "pulse_duration",
    "pulse_duration_from_length",
    "pulse_length",
    "pulse_length_from_duration",
    "size_param",
    "wavelength",
    "wavelength_from_frequency",
]
