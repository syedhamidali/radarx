"""
Radar Principles
================

Core physical and theoretical principles in radar meteorology.

Sources are [1]_ (radar equation, Doppler shift, noise) and [2]_ (Nyquist
velocity).

Name clashes: :func:`doppler_frequency_shift` here has the signature
``(v_radial, wavelength)`` whereas
:func:`radarx.fundamentals.doppler.doppler_frequency_shift` takes
``(frequency, vr)``. Because ``radarx.fundamentals`` star-imports this module
after ``doppler``, ``radarx.fundamentals.doppler_frequency_shift`` is the
function of this module. Use the explicit module path to be sure.

References
----------
.. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
       Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
       DOI). Eqs. (3.24) p. 46, (3.30), (3.32) p. 55, (3.40a,b) pp. 60-61.
.. [2] Rinehart, R. E., 1991: *Radar for Meteorologists*, 2nd ed. (3rd
       printing 1994). Rinehart Publications, Grand Forks, ND, ISBN
       0-9608700-7-5 (book, no DOI). Chapter 6.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
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

__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np

from .constants import K_BOLTZMANN, C


def _compute_numerator(transmit_power, gain, wavelength, rcs):
    """
    Numerator ``Pt G**2 lambda**2 sigma`` of the point-target radar equation,
    Eq. (3.24) of Doviak and Zrnic (1993), 2nd ed., p. 46 (same antenna for
    transmit and receive, ``g_t = g_r = g``). See :func:`radar_range`.
    """
    return transmit_power * gain**2 * wavelength**2 * rcs


def _compute_denominator(system_loss, min_detectable_power):
    """
    Denominator ``(4 pi)**3 L S_min`` of the radar equation solved for
    ``r**4``: Eq. (3.24) of Doviak and Zrnic (1993), 2nd ed., p. 46, with
    ``P_r = S_min`` and an extra loss factor ``L >= 1`` (radarx addition; in
    Doviak and Zrnic Eq. 3.24 the losses are contained in ``g``).
    """
    return (4 * np.pi) ** 3 * system_loss * min_detectable_power


def radar_range(
    transmit_power, gain, wavelength, rcs, system_loss, min_detectable_power
):
    """
    Compute maximum radar detection range using the radar range equation.

    Solves the point-target radar equation
    ``P_r = P_t G**2 lambda**2 sigma / ((4 pi)**3 r**4)``, Eq. (3.24) of [1]_
    (p. 46), for ``r`` with ``P_r = min_detectable_power`` and divides by
    ``system_loss`` (``L >= 1``, linear): ``r_max = [P_t G**2 lambda**2 sigma
    / ((4 pi)**3 L S_min)]**(1/4)``. The loss factor is a radarx addition
    (in [1]_ losses are inside ``g``); note it divides, whereas the
    ``aloss``/``rloss`` of :func:`radarx.fundamentals.system.radar_const`
    multiply.

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
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    """
    numerator = _compute_numerator(transmit_power, gain, wavelength, rcs)
    denominator = _compute_denominator(system_loss, min_detectable_power)
    return (numerator / denominator) ** 0.25


def compute_nyquist_velocity(prf, wavelength):
    """
    Compute the Nyquist velocity for Doppler radar.

    ``v_a = lambda * PRF / 4``, Eq. (3.40b) of [1]_ (p. 61); Chapter 6 of
    [2]_.

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
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    .. [2] Rinehart, R. E., 1991: *Radar for Meteorologists*, 2nd ed. (3rd
           printing 1994). Rinehart Publications, Grand Forks, ND, ISBN
           0-9608700-7-5 (book, no DOI).
    """
    return wavelength * prf / 4


def compute_range_resolution(pulse_width):
    """
    Compute the radar range resolution.

    ``c * tau / 2``: for a rectangular pulse of width ``tau`` and a receiver
    bandwidth much larger than ``1 / tau`` only scatterers within a range
    interval ``c tau / 2`` contribute to a sample, Section 4.4.1 of [1]_
    (p. 75). The effective resolution depends on the receiver filter (Section
    4.4.2); this is the idealized infinite-bandwidth value. ``c`` is
    :data:`radarx.fundamentals.constants.C`.

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
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    """
    return C * pulse_width / 2


def compute_doppler_shift(velocity, wavelength):
    """
    Compute Doppler frequency shift.

    ``f_d = 2 v / lambda``: the Doppler shift in radians per second is
    ``omega_d = 4 pi v / lambda``, Eq. (3.30) of [1]_, divided by ``2 pi``.
    Positive ``velocity`` (as given) gives a positive shift here; in [1]_
    the sign of the phase rate follows the range-rate convention of that
    equation.

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
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    """
    return 2 * velocity / wavelength


def compute_snr(power_received, noise_bandwidth, system_temp):
    """
    Compute signal-to-noise ratio (SNR).

    ``SNR = P_r / (k T B)``, with the noise power ``N = k T_sy B_n`` of Eq.
    (3.32) of [1]_ (p. 55). ``system_temp`` must be the full system noise
    temperature (including the receiver noise temperature, Eq. 3.33); no
    noise figure is applied. ``k`` is
    :data:`radarx.fundamentals.constants.K_BOLTZMANN`.

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
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
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

    Plain ratio ``signal / noise`` (definition; no source needed).

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

    ``f_d = 2 v_r / lambda``, Eq. (3.30) of [1]_ divided by ``2 pi``. This
    function shadows :func:`radarx.fundamentals.doppler.doppler_frequency_shift`
    in the ``radarx.fundamentals`` namespace although the signature differs
    (see the module docstring).

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

    References
    ----------
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    """
    return 2 * v_radial / wavelength


def round_trip_time(distance):
    """
    Compute round trip time for a given distance.

    ``t = 2 d / c``: the echo from range ``d`` is delayed by ``2 r / c``,
    see the factor ``t - 2 r / c`` in Eq. (3.25) of [1]_.

    Parameters
    ----------
    distance : float
        Distance to the target [m]

    Returns
    -------
    float
        Time for signal to travel to the target and back [s]

    References
    ----------
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    """
    return 2 * distance / C
