"""
Timing Calculations
===================

Functions for radar timing-related calculations: PRF, duty cycle, blind range, etc.

Sources: duty cycle ``f = tau * PRF`` from Rinehart [2]_ (Chapter 13,
p. 193); unambiguous range and velocity from Doviak and Zrnic [1]_,
Eqs. (3.40a) and (3.40b), pp. 60-61.

References
----------
.. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
       Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
       DOI).
.. [2] Rinehart, R. E., 1991: *Radar for Meteorologists*, 2nd ed. (3rd
       printing 1994). Rinehart Publications, Grand Forks, ND, ISBN
       0-9608700-7-5 (book, no DOI).

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = [
    "compute_blind_range",
    "compute_duty_cycle",
    "compute_max_unambiguous_range",
    "compute_max_unambiguous_velocity",
    "compute_prf",
    "compute_pulse_repetition_interval",
]

__doc__ = __doc__.format("\n   ".join(__all__))

from .constants import C


def compute_prf(pulse_width, duty_cycle):
    """
    Compute Pulse Repetition Frequency (PRF).

    ``PRF = f / tau``: the duty cycle is ``f = tau * PRF`` (Rinehart [1]_,
    Chapter 13, p. 193), solved for the PRF.

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
    .. [1] Rinehart, R. E., 1991: *Radar for Meteorologists*, 2nd ed. (3rd
           printing 1994). Rinehart Publications, Grand Forks, ND, ISBN
           0-9608700-7-5 (book, no DOI).
    """
    return duty_cycle / pulse_width


def compute_duty_cycle(prf, pulse_width):
    """
    Compute the duty cycle of the radar.

    ``f = tau * PRF``, the fraction of time the transmitter is on, Rinehart
    [1]_ (Chapter 13, p. 193).

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
    .. [1] Rinehart, R. E., 1991: *Radar for Meteorologists*, 2nd ed. (3rd
           printing 1994). Rinehart Publications, Grand Forks, ND, ISBN
           0-9608700-7-5 (book, no DOI).
    """
    return prf * pulse_width


def compute_blind_range(pulse_width):
    """
    Compute blind range (minimum detectable range) caused by transmission pulse.

    ``c * tau / 2``: the range an echo travels while the pulse of width
    ``tau`` is being transmitted. Elementary geometry (round-trip delay
    ``2 r / c``); no textbook equation is claimed. Receiver recovery time is
    not included, so the real blind range is larger.

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
    """
    return C * pulse_width / 2


def compute_max_unambiguous_range(prf):
    """
    Compute the maximum unambiguous range.

    ``r_a = c T_s / 2 = c / (2 PRF)``, Eq. (3.40a) of [1]_ (p. 60).

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
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    """
    return C / (2 * prf)


def compute_max_unambiguous_velocity(prf, wavelength):
    """
    Compute the maximum unambiguous velocity.

    ``v_a = lambda * PRF / 4``, Eq. (3.40b) of [1]_ (p. 61); duplicate of
    :func:`radarx.fundamentals.doppler.nyquist_velocity`.

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
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    """
    return prf * wavelength / 4


def compute_pulse_repetition_interval(prf):
    """
    Compute Pulse Repetition Interval (PRI) from PRF.

    ``T_s = 1 / PRF`` (definition of the pulse repetition time ``T_s`` used
    in Eqs. 3.40a,b of [1]_).

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
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    """
    return 1.0 / prf
