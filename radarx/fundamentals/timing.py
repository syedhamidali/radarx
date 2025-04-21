"""
Timing Calculations
===================

Functions for radar timing-related calculations: PRF, duty cycle, blind range, etc.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   compute_blind_range
   compute_duty_cycle
   compute_max_unambiguous_range
   compute_max_unambiguous_velocity
   compute_prf
   compute_pulse_repetition_interval

References
----------
- Doviak, R. J., & Zrnić, D. S. (1993). Doppler Radar and Weather Observations. Academic Press.
"""

__all__ = [
    "compute_blind_range",
    "compute_duty_cycle",
    "compute_max_unambiguous_range",
    "compute_max_unambiguous_velocity",
    "compute_prf",
    "compute_pulse_repetition_interval",
]

from .constants import C


def compute_prf(pulse_width, duty_cycle):
    """
    Compute Pulse Repetition Frequency (PRF).

    Parameters
    ----------
    pulse_width : float
        Pulse width in seconds.
    duty_cycle : float
        Duty cycle (0 < duty_cycle < 1).

    Returns
    -------
    float
        PRF in Hz.

    References
    ----------
    - Doviak and Zrnić (1993), Eq. 3.3.2
    """
    return duty_cycle / pulse_width


def compute_duty_cycle(prf, pulse_width):
    """
    Compute the duty cycle of the radar.

    Parameters
    ----------
    prf : float
        Pulse repetition frequency [Hz].
    pulse_width : float
        Pulse width [s].

    Returns
    -------
    float
        Duty cycle (0 < value < 1).

    References
    ----------
    - Doviak and Zrnić (1993), Eq. 3.3.3
    """
    return prf * pulse_width


def compute_blind_range(pulse_width):
    """
    Compute blind range (minimum detectable range) caused by transmission pulse.

    Parameters
    ----------
    pulse_width : float
        Pulse width [s].

    Returns
    -------
    float
        Blind range [m].

    Notes
    -----
    Targets within this range cannot be detected because the receiver is turned off.

    References
    ----------
    - Doviak and Zrnić (1993), Section 3.3
    """
    return C * pulse_width / 2


def compute_max_unambiguous_range(prf):
    """
    Compute the maximum unambiguous range.

    Parameters
    ----------
    prf : float
        Pulse repetition frequency [Hz].

    Returns
    -------
    float
        Maximum unambiguous range [m].

    References
    ----------
    - Doviak and Zrnić (1993), Eq. 3.3.4
    """
    return C / (2 * prf)


def compute_max_unambiguous_velocity(prf, wavelength):
    """
    Compute the maximum unambiguous velocity.

    Parameters
    ----------
    prf : float
        Pulse repetition frequency [Hz].
    wavelength : float
        Radar wavelength [m].

    Returns
    -------
    float
        Maximum unambiguous velocity [m/s].

    References
    ----------
    - Doviak and Zrnić (1993), Eq. 6.3.5
    """
    return prf * wavelength / 4


def compute_pulse_repetition_interval(prf):
    """
    Compute Pulse Repetition Interval (PRI) from PRF.

    Parameters
    ----------
    prf : float
        Pulse repetition frequency [Hz].

    Returns
    -------
    float
        Pulse repetition interval [s].

    References
    ----------
    - Doviak and Zrnić (1993), Eq. 3.3.1
    """
    return 1.0 / prf
