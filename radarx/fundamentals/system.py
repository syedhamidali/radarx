"""
Radar System Characteristics
============================

Core functions related to radar system components like antenna gain, pulse parameters,
wavelength/frequency conversion, and radar constants.

Sources: the radar equation, antenna gain and effective area are from Doviak
and Zrnic [1]_ (Eqs. 3.3, 3.21, 3.24, pp. 34, 45, 46); the weather radar
constant from their Eqs. (4.14), (4.16), (4.31), pp. 74-75, 82 (see
:func:`radar_const` for the derivation and its unit and sign conventions);
the size parameter from Bringi and Chandrasekar [2]_. The ``c`` used is
:data:`radarx.fundamentals.constants.C`.

References
----------
.. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
       Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
       DOI).
.. [2] Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler
       Weather Radar: Principles and Applications*. Cambridge University
       Press, https://doi.org/10.1017/CBO9780511541094 (book; ISBN
       0-521-62384-7). Section 2.5.

.. autosummary::
    :nosignatures:
    :toctree: generated/

    {}
"""

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

__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np

from .constants import C


def ant_eff_area(gain_dbi, wavelength):
    """
    Compute effective antenna area from gain and wavelength.

    ``A_e = g lambda**2 / (4 pi)`` on the beam axis, Eq. (3.21) of [1]_
    (p. 45), with ``g = 10**(gain_dbi / 10)``.

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

    References
    ----------
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    """
    gain_linear = 10 ** (gain_dbi / 10)
    return (gain_linear * wavelength**2) / (4 * np.pi)


def antenna_gain(p_beam, p_iso):
    """
    Compute antenna gain in dB from power ratio.

    ``10 log10(p_beam / p_iso)``: the gain is the ratio of the peak power
    density to that of an isotropic radiator, Eq. (3.3) of [1]_ (p. 34),
    expressed in decibels.

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

    References
    ----------
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
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

    ``f = c / lambda`` (definition of wavelength; ``c`` in vacuum, not
    corrected for the refractive index of air).

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

    ``P_r = P_t g**2 lambda**2 sigma / ((4 pi)**3 r**4)``, Eq. (3.24) of [1]_
    (p. 46), with ``g = 10**(gain_dbi / 10)`` used for transmit and receive
    and no extra loss factor.

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

    References
    ----------
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
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

    ``tau = 2 L / c``. Note ``pulse_length`` is interpreted as the
    range depth ``c tau / 2`` (the round-trip convention of
    :func:`pulse_length_from_duration`), not the pulse length in space
    ``c tau`` (the thickness of the transmitted shell, Section 3.1 of [1]_,
    p. 34). Definition.

    Parameters
    ----------
    pulse_length : float or array-like
         Pulse length [m]

    Returns
    -------
    float or array-like
         Pulse duration [s]

    References
    ----------
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
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

    Returns ``c tau / 2``, the range depth of a pulse (the range resolution
    of a rectangular pulse in the infinite-bandwidth limit, Section 4.4.1 of
    [1]_, p. 75), not the pulse length in space ``c tau``. Feeding this into
    :func:`radarx.fundamentals.geometry.sample_volume_gaussian` therefore
    yields half of the effective volume.

    Parameters
    ----------
    pulse_duration : float or array-like
         Pulse duration [s]

    Returns
    -------
    float or array-like
         Pulse length [m]

    References
    ----------
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    """
    return C * np.asarray(pulse_duration) / 2.0


def radar_const(power_t, gain, tau, wavelength, bw_h, bw_v, aloss, rloss):
    """
    Compute the radar constant (unitless).

    Implements ``C = pi**3 c P_t G**2 tau theta phi l_a l_r / (1024 ln 2
    lambda**2)`` with ``G = 10**(gain/10)``, beamwidths in radians, and
    ``l_a = 10**(aloss/10)``, ``l_r = 10**(rloss/10)``. Without ``l_a`` and
    ``l_r`` this is the constant that multiplies ``|K|**2 Z / r**2`` in the
    weather radar equation: combining the point-target equation (3.24), the
    effective volume of Eq. (4.14) (``pi theta**2 c tau / (16 ln 2)`` for a
    Gaussian circular beam, generalized here to ``theta phi``) and
    ``eta = pi**5 |K|**2 Z / lambda**4``, Eq. (4.31), of [1]_ gives
    ``P_r = [pi**3 c tau P_t G**2 theta phi / (1024 ln 2 lambda**2)]
    |K|**2 Z / r**2`` (``1024 = 64 * 16``). This is derived here from Eqs.
    (3.24), (4.14) and (4.31); the equation numbers of the final form in
    [1]_ (Eqs. 4.34, 4.35) are not cited. The Gaussian-beam ``ln 2`` form is
    that of [2]_ (content not checked against the original paper).

    Units: with ``c`` in m/s, ``tau`` in s and ``P_t`` in W the result has the
    unit W/m (not unitless) and gives ``P_r`` in W for ``Z`` in m^3
    (``m^6 m^-3``, SI) and ``r`` in m. Using it with ``Z`` in mm^6 m^-3 needs
    a factor 1e-18 (see :func:`radarx.fundamentals.variables.reflectivity_factor`).
    Finite receiver bandwidth loss ``l_r`` of Eq. (4.15)-(4.16) of [1]_ and
    path attenuation are not included unless entered through ``aloss`` and
    ``rloss``.

    Sign convention of the losses: ``aloss`` and ``rloss`` multiply the
    constant by ``10**(loss/10)``. A loss that reduces the received power
    must therefore be entered as negative dB (a positive value raises the
    constant). In [1]_ all loss factors (``l``, ``l_r``) are >= 1 and sit in
    the denominator of the radar equation.

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
         Antenna/waveguide/coupler loss [dB]; negative for a loss (see above)
    rloss : float
         Receiver loss [dB]; negative for a loss (see above)

    Returns
    -------
    float
         Radar constant (units W/m, see above)

    References
    ----------
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    .. [2] Probert-Jones, J. R., 1962: The radar equation in meteorology.
           *Q. J. R. Meteorol. Soc.*, **88** (378), 485-495,
           https://doi.org/10.1002/qj.49708837810
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

    ``P_r = P_t g_tx g_rx lambda**2 sigma / ((4 pi)**3 r**4 L)``: Eq. (3.24)
    of [1]_ (p. 46, there with ``g_t = g_r = g``) with separate transmit and
    receive gains and a loss factor ``L >= 1`` that divides (radarx addition;
    in [1]_ losses are inside ``g``).

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

    References
    ----------
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    """
    numerator = pt * g_tx * g_rx * wavelength**2 * sigma
    denominator = ((4 * np.pi) ** 3) * r**4 * loss
    return numerator / denominator


def size_param(diameter, wavelength):
    """
    Compute size parameter alpha.

    ``alpha = pi D / lambda = k0 a``, the size parameter of Section 2.5 of
    [1]_ (duplicate of :func:`radarx.fundamentals.scattering.size_parameter`).

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

    References
    ----------
    .. [1] Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler
           Weather Radar: Principles and Applications*. Cambridge University
           Press, https://doi.org/10.1017/CBO9780511541094 (book; ISBN
           0-521-62384-7).
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

    Algebraic inverse of :func:`radar_equation` (Eq. 3.24 of [1]_, p. 46):
    ``P_t = P_r (4 pi)**3 r**4 L / (g_tx g_rx lambda**2 sigma)``.

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

    References
    ----------
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
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

    ``lambda = c / f`` (vacuum speed of light, see
    :func:`frequency_from_wavelength`).

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
