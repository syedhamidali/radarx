"""
Radar Principles
================

Core physical and theoretical principles in radar meteorology.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   radar_range
   compute_nyquist_velocity
   compute_range_resolution
   compute_doppler_shift
   compute_snr

References
----------
- Rinehart, R. E. (2004). Radar for Meteorologists. 4th ed. Rinehart Publications.
- Doviak, R. J., & Zrnić, D. S. (1993). Doppler Radar and Weather Observations. Academic Press.
"""

__all__ = [
    "compute_doppler_shift",
    "compute_nyquist_velocity",
    "compute_range_resolution",
    "compute_snr",
    "doppler_frequency_shift",
    "radar_range",
    "range_resolution",
    "round_trip_time",
    "snr",
]

import numpy as np
from .constants import C, K_BOLTZMANN


def _compute_numerator(transmit_power, gain, wavelength, rcs):
    return transmit_power * gain**2 * wavelength**2 * rcs


def _compute_denominator(system_loss, min_detectable_power):
    return (4 * np.pi) ** 3 * system_loss * min_detectable_power


def radar_range(
    transmit_power, gain, wavelength, rcs, system_loss, min_detectable_power
):
    """
    Compute maximum radar detection range using the radar range equation.

    Parameters
    ----------
    transmit_power : float
        Transmitted power [W]
    gain : float
        Antenna gain (linear, not dB)
    wavelength : float
        Radar wavelength [m]
    rcs : float
        Radar cross-section [m^2]
    system_loss : float
        System loss factor (linear, not dB)
    min_detectable_power : float
        Minimum detectable signal power [W]

    Returns
    -------
    float
        Maximum radar range [m]

    References
    ----------
    - Rinehart (2004), Eq. 2.1
    - Doviak and Zrnić (1993), Section 3.2
    """
    numerator = _compute_numerator(transmit_power, gain, wavelength, rcs)
    denominator = _compute_denominator(system_loss, min_detectable_power)
    return (numerator / denominator) ** 0.25


def compute_nyquist_velocity(prf, wavelength):
    """
    Compute the Nyquist velocity for Doppler radar.

    Parameters
    ----------
    prf : float
        Pulse repetition frequency [Hz]
    wavelength : float
        Radar wavelength [m]

    Returns
    -------
    float
        Nyquist velocity [m/s]

    References
    ----------
    - Doviak and Zrnić (1993), Section 6.3.1
    """
    return wavelength * prf / 4


def compute_range_resolution(pulse_width):
    """
    Compute the radar range resolution.

    Parameters
    ----------
    pulse_width : float
        Pulse width [s]

    Returns
    -------
    float
        Range resolution [m]

    References
    ----------
    - Rinehart (2004), Eq. 2.6
    """
    return C * pulse_width / 2


def compute_doppler_shift(velocity, wavelength):
    """
    Compute Doppler frequency shift.

    Parameters
    ----------
    velocity : float
        Target radial velocity [m/s]
    wavelength : float
        Radar wavelength [m]

    Returns
    -------
    float
        Doppler shift [Hz]

    References
    ----------
    - Doviak and Zrnić (1993), Eq. 6.2.1
    """
    return 2 * velocity / wavelength


def compute_snr(power_received, noise_bandwidth, system_temp):
    """
    Compute signal-to-noise ratio (SNR).

    Parameters
    ----------
    power_received : float
        Received signal power [W]
    noise_bandwidth : float
        Noise bandwidth [Hz]
    system_temp : float
        System temperature [K]

    Returns
    -------
    float
        Signal-to-noise ratio (linear)

    References
    ----------
    - Doviak and Zrnić (1993), Eq. 3.1.12
    """
    noise_power = K_BOLTZMANN * system_temp * noise_bandwidth
    return power_received / noise_power


def range_resolution(pulse_width):
    """
    Alias for compute_range_resolution for compatibility.

    Parameters
    ----------
    pulse_width : float
        Pulse width [s]

    Returns
    -------
    float
        Range resolution [m]
    """
    return compute_range_resolution(pulse_width)


def snr(signal, noise):
    """
    Compute signal-to-noise ratio in linear scale.

    Parameters
    ----------
    signal : float
        Signal power [W]
    noise : float
        Noise power [W]

    Returns
    -------
    float
        SNR (linear)
    """
    return signal / noise


def doppler_frequency_shift(v_radial, wavelength):
    """
    Compute Doppler frequency shift given radial velocity and wavelength.

    Parameters
    ----------
    v_radial : float
        Radial velocity [m/s]
    wavelength : float
        Radar wavelength [m]

    Returns
    -------
    float
        Doppler frequency shift [Hz]
    """
    return 2 * v_radial / wavelength


def round_trip_time(distance):
    """
    Compute round trip time for a given distance.

    Parameters
    ----------
    distance : float
        Distance to the target [m]

    Returns
    -------
    float
        Time for signal to travel to the target and back [s]
    """
    return 2 * distance / C
