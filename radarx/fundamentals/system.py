"""
Radar System Characteristics
============================

Core functions related to radar system components like antenna gain, pulse parameters,
wavelength/frequency conversion, and radar constants.

.. autosummary::
    :nosignatures:
    :toctree: generated/

    ant_eff_area
    antenna_gain
    frequency
    frequency_from_wavelength
    power_return_target
    pulse_duration
    pulse_duration_from_length
    pulse_length
    pulse_length_from_duration
    radar_const
    radar_equation
    size_param
    solve_peak_power
    wavelength
    wavelength_from_frequency
"""

import numpy as np
from .constants import C


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


def frequency(wavelength):
    """
    Alias for frequency_from_wavelength.
    """
    return frequency_from_wavelength(wavelength)


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


def pulse_duration(pulse_length_val):
    """
    Alias for pulse_duration_from_length.
    """
    return pulse_duration_from_length(pulse_length_val)


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


def pulse_length(pulse_duration):
    """
    Alias for pulse_length_from_duration.
    """
    return pulse_length_from_duration(pulse_duration)


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


def radar_const(power_t, gain, tau, wavelength, bw_h, bw_v, aloss, rloss):
    """
    Compute the radar constant (unitless).

    Adapted from CSU Radar Meteorology tools (AT741 notes).

    Parameters
    ----------
    power_t : float
         Transmitted power [W]
    gain : float
         Antenna Gain [dB]
    tau : float
         Pulse Width [s]
    wavelength : float
         Radar wavelength [m]
    bw_h : float
         Horizontal antenna beamwidth [degrees]
    bw_v : float
         Vertical antenna beamwidth [degrees]
    aloss : float
         Antenna/waveguide/coupler loss [dB]
    rloss : float
         Receiver loss [dB]

    Returns
    -------
    float
         Radar constant (unitless)
    """
    alosslin = 10 ** (aloss / 10.0)
    rlosslin = 10 ** (rloss / 10.0)
    gainlin = 10 ** (gain / 10.0)

    bw_hr = np.deg2rad(bw_h)
    bw_vr = np.deg2rad(bw_v)

    numer = (
        np.pi**3 * C * power_t * gainlin**2 * tau * bw_hr * bw_vr * alosslin * rlosslin
    )
    denom = 1024.0 * np.log(2) * wavelength**2
    return numer / denom


def radar_equation(
    pt: float,
    g_tx: float,
    g_rx: float,
    wavelength: float,
    sigma: float,
    r: float,
    loss: float = 1.0,
) -> float:
    """
    Compute the received power using the radar range equation.

    Parameters
    ----------
    pt : float
         Transmitter peak power [W].
    g_tx : float
         Transmit antenna gain (linear).
    g_rx : float
         Receive antenna gain (linear).
    wavelength : float
         Radar wavelength [m].
    sigma : float
         Radar cross-section [m^2].
    r : float
         Range to target [m].
    loss : float, optional
         System loss factor (default is 1.0).

    Returns
    -------
    float
         Received power [W].
    """
    numerator = pt * g_tx * g_rx * wavelength**2 * sigma
    denominator = ((4 * np.pi) ** 3) * r**4 * loss
    return numerator / denominator


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


def solve_peak_power(
    pr: float,
    g_tx: float,
    g_rx: float,
    wavelength: float,
    sigma: float,
    r: float,
    loss: float = 1.0,
) -> float:
    """
    Solve for transmitter peak power using the radar equation.

    Parameters
    ----------
    pr : float
         Received power [W].
    g_tx : float
         Transmit antenna gain (linear).
    g_rx : float
         Receive antenna gain (linear).
    wavelength : float
         Radar wavelength [m].
    sigma : float
         Radar cross-section [m^2].
    r : float
         Range to target [m].
    loss : float, optional
         System loss factor (default is 1.0).

    Returns
    -------
    float
         Required transmitter peak power [W].
    """
    numerator = pr * ((4 * np.pi) ** 3) * r**4 * loss
    denominator = g_tx * g_rx * wavelength**2 * sigma
    return numerator / denominator


def wavelength(freq):
    """
    Alias for wavelength_from_frequency.
    """
    return wavelength_from_frequency(freq)


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
    "radar_const",
    "radar_equation",
    "size_param",
    "solve_peak_power",
    "wavelength",
    "wavelength_from_frequency",
]
