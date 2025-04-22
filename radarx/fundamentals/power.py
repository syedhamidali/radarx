"""
Radar Power Calculations
=========================

Functions for computing peak power, average power, and minimum detectable signal.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   compute_peak_power
   compute_average_power
   compute_min_detectable_signal

References
----------
- Rinehart, R. E. (2004). Radar for Meteorologists. 4th ed.
- Doviak, R. J., & Zrnić, D. S. (1993). Doppler Radar and Weather Observations.
"""

__all__ = [
    "compute_average_power",
    "compute_min_detectable_signal",
    "compute_peak_power",
]

from .constants import K_BOLTZMANN


def compute_peak_power(voltage, impedance):
    """
    Compute peak power from voltage and impedance.

    Parameters
    ----------
    voltage : float
        Peak voltage [V].
    impedance : float
        System impedance [Ohms].

    Returns
    -------
    float
        Peak power [W].

    References
    ----------
    - Rinehart (2004), Ch. 2
    """
    return voltage**2 / impedance


def compute_average_power(peak_power, duty_cycle):
    """
    Compute average power from peak power and duty cycle.

    Parameters
    ----------
    peak_power : float
        Peak transmitted power [W].
    duty_cycle : float
        Duty cycle (0 < value < 1).

    Returns
    -------
    float
        Average power [W].

    References
    ----------
    - Rinehart (2004), Ch. 2
    """
    return peak_power * duty_cycle


def compute_min_detectable_signal(bandwidth, system_temp, snr_threshold=1):
    """
    Compute minimum detectable signal power.

    Parameters
    ----------
    bandwidth : float
        Receiver bandwidth [Hz].
    system_temp : float
        System temperature [K].
    snr_threshold : float, optional
        Minimum required SNR to detect signal (default is 1).

    Returns
    -------
    float
        Minimum detectable signal power [W].

    References
    ----------
    - Doviak and Zrnić (1993), Eq. 3.1.12
    """
    return snr_threshold * K_BOLTZMANN * system_temp * bandwidth
