"""
Radar Power Calculations
=========================

Functions for computing peak power, average power, and minimum detectable signal.

Sources: average power and duty cycle follow Rinehart [1]_; the thermal noise
power ``k T B`` follows Doviak and Zrnic [2]_. The earlier citations of "Rinehart
(2004), Ch. 2" and "Doviak and Zrnic (1993), Eq. 3.1.12" could not be matched
to the books on disk (2nd edition 1991 and 2nd edition 1993) and were removed.

References
----------
.. [1] Rinehart, R. E., 1991: *Radar for Meteorologists*, 2nd ed. (3rd
       printing 1994). Rinehart Publications, Grand Forks, ND, ISBN
       0-9608700-7-5 (book, no DOI). Chapter 13, p. 193.
.. [2] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
       Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
       DOI). Eqs. (3.31)-(3.33), pp. 54-56.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = [
    "compute_average_power",
    "compute_min_detectable_signal",
    "compute_peak_power",
]

__doc__ = __doc__.format("\n   ".join(__all__))

from .constants import K_BOLTZMANN


def compute_peak_power(voltage, impedance):
    """
    Compute peak power from voltage and impedance.

    ``P = V**2 / Z`` (Ohm's law for a constant, or rms, voltage). If
    ``voltage`` is the peak amplitude of a sinusoid the mean power over a
    cycle is ``V**2 / (2 Z)``; the function does not include that factor of
    2. Elementary circuit relation, no radar source (the earlier "Rinehart
    (2004), Ch. 2" pointer could not be verified).

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
    """
    return voltage**2 / impedance


def compute_average_power(peak_power, duty_cycle):
    """
    Compute average power from peak power and duty cycle.

    ``P_ave = P_t * f`` with duty cycle ``f = tau * PRF``, Rinehart [1]_
    (Chapter 13, p. 193, "Quantification").

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
    .. [1] Rinehart, R. E., 1991: *Radar for Meteorologists*, 2nd ed. (3rd
           printing 1994). Rinehart Publications, Grand Forks, ND, ISBN
           0-9608700-7-5 (book, no DOI).
    """
    return peak_power * duty_cycle


def compute_min_detectable_signal(bandwidth, system_temp, snr_threshold=1):
    """
    Compute minimum detectable signal power.

    ``S_min = SNR_min * k * T_sy * B_n``: the thermal noise power
    ``N = k T_sy B_n`` of Eq. (3.32) of [1]_ (p. 55; ``k`` = Boltzmann
    constant, ``T_sy`` = system noise temperature referenced to the
    receiver input including the receiver noise temperature, Eq. 3.33,
    ``B_n`` = noise bandwidth) times the required linear SNR. The caller must
    supply ``system_temp`` as the full system noise temperature; no separate
    noise figure is applied. ``snr_threshold`` is linear (not dB).

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
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    """
    return snr_threshold * K_BOLTZMANN * system_temp * bandwidth
