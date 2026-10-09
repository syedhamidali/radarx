"""
Doppler Radar Calculations
===========================

Functions related to Doppler radar performance and velocity limits.

References
----------
.. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
       Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
       DOI). Eqs. (3.30), (3.40a,b), (7.1), (7.6b); Sections 3.6, 7.2,
       7.4.3.
.. [2] Rinehart, R. E., 1991: *Radar for Meteorologists*, 2nd ed. (3rd
       printing 1994). Rinehart Publications, Grand Forks, ND, ISBN
       0-9608700-7-5 (book, no DOI). Chapter 6 (Section 6.2, "Doppler
       dilemma").
.. [3] Einstein, A., 1905: Zur Elektrodynamik bewegter Koerper. *Ann.
       Phys.*, **322** (10), 891-921,
       https://doi.org/10.1002/andp.19053221004
.. [4] Zrnic, D. S., and P. Mahapatra, 1985: Two methods of ambiguity
       resolution in pulse Doppler weather radars. *IEEE Trans. Aerosp.
       Electron. Syst.*, **AES-21** (4), 470-483,
       https://doi.org/10.1109/TAES.1985.310635

The relativistic option of :func:`doppler_frequency_shift` returns the
one-way shift and :func:`dual_prf_velocity` is negative for ``prf1 > prf2``.
The two-way classical results [1]_ (see also [2]_) are the default and
``exact`` options; the relativistic form is that of [3]_ and the dual-PRF
method that of [4]_.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = [
    "doppler_dilemma",
    "doppler_frequency_shift",
    "dual_prf_velocity",
    "max_frequency",
    "nyquist_velocity",
    "unambiguous_range",
    "_doppler_shift_basic",
    "_doppler_shift_exact",
    "_doppler_shift_relativistic",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np

from .constants import C


def max_frequency(prf):
    """
    Compute the maximum observable Doppler frequency.

    This is the Nyquist (folding) frequency ``f_N = 1 / (2 T_s) = PRF / 2``
    for pulse spacing ``T_s``; Doppler shifts above it alias. See the text
    below Fig. 3.14 of [1]_ (p. 61) and Chapter 6 of [2]_.

    Parameters
    ----------
    prf : float or array-like
        Pulse repetition frequency [Hz]

    Returns
    -------
    float or array-like
        Maximum Doppler frequency [Hz]

    References
    ----------
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    .. [2] Rinehart, R. E., 1991: *Radar for Meteorologists*, 2nd ed. (3rd
           printing 1994). Rinehart Publications, Grand Forks, ND, ISBN
           0-9608700-7-5 (book, no DOI).
    """
    return np.asarray(prf) / 2.0


def nyquist_velocity(prf, wavelength):
    """
    Compute Nyquist velocity (maximum unambiguous Doppler velocity).

    ``v_a = lambda * PRF / 4 = lambda / (4 T_s)``, Eq. (3.40b) of [1]_
    (p. 61), derived from the Nyquist frequency ``PRF / 2`` and the
    two-way Doppler shift ``f_d = 2 v / lambda`` (Eq. 3.30). Same
    expression in Chapter 6 of [2]_.

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

    References
    ----------
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    .. [2] Rinehart, R. E., 1991: *Radar for Meteorologists*, 2nd ed. (3rd
           printing 1994). Rinehart Publications, Grand Forks, ND, ISBN
           0-9608700-7-5 (book, no DOI).
    """
    return np.asarray(prf) * np.asarray(wavelength) / 4.0


def unambiguous_range(prf):
    """
    Compute maximum unambiguous range for a given PRF.

    ``r_a = c T_s / 2 = c / (2 PRF)``, Eq. (3.40a) of [1]_ (p. 60), with
    ``c`` the speed of light in vacuum
    (:data:`radarx.fundamentals.constants.C`).

    Parameters
    ----------
    prf : float or array-like
        Pulse repetition frequency [Hz]

    Returns
    -------
    float or array-like
        Maximum unambiguous range [m]

    References
    ----------
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    """
    return C / (2.0 * np.asarray(prf))


def _doppler_shift_relativistic(frequency, vr):
    """
    One-way relativistic Doppler shift ``f (sqrt((1+b)/(1-b)) - 1)``,
    ``b = vr / c`` (Einstein 1905 [1]_, one-way source/observer shift).

    A radar measures the two-way (reflected) shift, which is about twice
    this, ``f ((1+b)/(1-b) - 1) = 2 f vr / (c - vr)`` (the ``exact`` result);
    this helper therefore returns about half the radar value.

    References
    ----------
    .. [1] Einstein, A., 1905: Zur Elektrodynamik bewegter Koerper. *Ann.
           Phys.*, **322** (10), 891-921,
           https://doi.org/10.1002/andp.19053221004 (section not cited)
    """
    return frequency * (np.sqrt((1 + vr / C) / (1 - vr / C)) - 1)


def _doppler_shift_exact(frequency, vr):
    """
    Two-way Doppler shift ``2 f vr / (c - vr)``: reflection from a target
    approaching at ``vr`` shifts the frequency to ``f (c + vr) / (c - vr)``
    (classical moving-reflector result; equals the two-way relativistic
    ``f (1 + b) / (1 - b)`` shift). Derived here from the moving-reflector
    argument; ``c`` is :data:`radarx.fundamentals.constants.C`.
    """
    return 2.0 * frequency * vr / (C - vr)


def _doppler_shift_basic(frequency, vr):
    """
    First-order two-way Doppler shift ``2 f vr / c = 2 vr / lambda``, the
    frequency form of Eq. (3.30) of [1]_ (``omega_d = 4 pi vr / lambda``,
    Section 3.4.3).

    References
    ----------
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book,
           no DOI).
    """
    return 2.0 * frequency * vr / C


def doppler_frequency_shift(frequency, vr, exact=False, relativistic=False):
    """
    Compute Doppler frequency shift.

    The default is the first-order two-way shift ``2 f vr / c`` (Eq. 3.30 of
    [1]_ divided by 2 pi). ``exact=True`` gives ``2 f vr / (c - vr)``, the
    exact reflected shift of a moving target. ``relativistic=True`` returns
    ``f (sqrt((1+b)/(1-b)) - 1)`` with ``b = vr/c`` [2]_, which is the one-way
    relativistic shift, about half the radar (two-way) value (e.g. 186.8 Hz
    instead of 373.6 Hz for 5.6 GHz and 10 m/s). Positive ``vr`` (approaching)
    gives a positive shift.

    Parameters
    ----------
    frequency : float or array-like
        Transmitted radar frequency [Hz]
    vr : float or array-like
        Radial velocity of the target [m/s], positive if approaching
    exact : bool, optional
        If True, uses the classical but accurate Doppler formula: (2 * f * v) / (c - v).
    relativistic : bool, optional
        If True, uses the relativistic Doppler formula. Overrides `exact` if both are True.

    Returns
    -------
    float or array-like
        Doppler frequency shift [Hz]

    References
    ----------
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    .. [2] Einstein, A., 1905: Zur Elektrodynamik bewegter Koerper. *Ann.
           Phys.*, **322** (10), 891-921,
           https://doi.org/10.1002/andp.19053221004
    """
    frequency = np.asarray(frequency)
    vr = np.asarray(vr)

    if relativistic:
        return _doppler_shift_relativistic(frequency, vr)
    if exact:
        return _doppler_shift_exact(frequency, vr)
    return _doppler_shift_basic(frequency, vr)


def doppler_dilemma(value, wavelength):
    """
    Solve the Doppler dilemma equation: trade-off between unambiguous range and velocity.

    ``r_a * v_a = c * lambda / 8``: eliminating PRF between Eqs. (3.40a) and
    (3.40b) of [1]_; the range-velocity product is Eq. (7.1) of [1]_
    (Section 7.2). See Section 6.2 of [2]_. The function returns ``c lambda / 8 / value``, i.e. the range for a
    velocity or the velocity for a range.

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

    References
    ----------
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    .. [2] Rinehart, R. E., 1991: *Radar for Meteorologists*, 2nd ed. (3rd
           printing 1994). Rinehart Publications, Grand Forks, ND, ISBN
           0-9608700-7-5 (book, no DOI).
    """
    return (C * wavelength / 8.0) / np.asarray(value)


def dual_prf_velocity(wavelength, prf1, prf2):
    """
    Compute Nyquist velocity using dual-PRF scheme.

    Implements ``lambda / (4 (1/prf1 - 1/prf2))``, which is the extended
    unambiguous velocity ``v_m = lambda / (4 (T_s2 - T_s1))`` of the
    staggered / dual-PRT scheme, Eq. (7.6b) of [1]_ (Section 7.4.3; method
    of [2]_), where ``T_s = 1/PRF`` and ``T_s2 > T_s1``. With
    ``T_s1 = 1/prf1`` and ``T_s2 = 1/prf2`` the code computes
    ``lambda / (4 (T_s1 - T_s2))``, the opposite order. It is therefore
    positive when ``prf1 < prf2`` and negative when ``prf1 > prf2``; take
    ``abs()`` of the result for the magnitude.

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
    The sign of the result depends on the order of PRF values. The result is
    positive for ``prf1 < prf2`` and negative for ``prf1 > prf2``, e.g.
    ``dual_prf_velocity(0.1, 1000, 750) == -75`` and
    ``dual_prf_velocity(0.1, 750, 1000) == 75``.

    References
    ----------
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    .. [2] Zrnic, D. S., and P. Mahapatra, 1985: Two methods of ambiguity
           resolution in pulse Doppler weather radars. *IEEE Trans. Aerosp.
           Electron. Syst.*, **AES-21** (4), 470-483,
           https://doi.org/10.1109/TAES.1985.310635
    """
    return wavelength / (4.0 * (1.0 / prf1 - 1.0 / prf2))
