"""
Radar-Derived Variables
=======================

Functions related to reflectivity, differential reflectivity, and radial velocity.

Sources: reflectivity factor, Doppler relations and the 10 log10 definitions
follow Doviak and Zrnic [1]_ and Bringi and Chandrasekar [2]_ as detailed in
each docstring. The depolarization ratios here are plain channel ratios; see
:func:`linear_depolarization_ratio` and :func:`circular_depolarization_ratio`
for how they relate to the definitions of [2]_.

References
----------
.. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
       Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
       DOI). Eqs. (3.30), (4.31), (4.32).
.. [2] Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler
       Weather Radar: Principles and Applications*. Cambridge University
       Press, https://doi.org/10.1017/CBO9780511541094 (book; ISBN
       0-521-62384-7). Eqs. (2.54), (2.55), (3.91a), (3.167a), (3.168),
       (3.169).

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = [
    "reflectivity_factor",
    "differential_reflectivity",
    "linear_depolarization_ratio",
    "circular_depolarization_ratio",
    "radial_velocity",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np


def reflectivity_factor(p_return, radar_const, dielectric=0.93, range_m=1000.0):
    """
    Compute reflectivity factor Z [mm^6/m^3].

    Implements ``Z = P_r r**2 / (radar_const * dielectric**2)``, the
    inversion of the weather radar equation ``P_r = C |K|**2 Z / r**2``
    (Eqs. 4.14, 4.16, 4.31 of [1]_, with ``Z = lambda**4 eta / (pi**5 |K|**2)``,
    Eq. 3.167a of [2]_). Two caveats, documented without changing the code.
    (1) ``dielectric`` is SQUARED although the default 0.93 is the
    literature value of ``|K|**2`` for water (Section 3.2 of [1]_, p. 36),
    so with the default the result is about 7.5 percent (0.3 dB) too large
    relative to the textbook inversion. (2) The units follow
    :func:`radarx.fundamentals.system.radar_const`: with its SI constant
    (W/m), ``p_return`` in W and ``range_m`` in m the result is in m^3
    (``m^6 m^-3``), i.e. 1e-18 times the value in mm^6 m^-3 stated below.
    The default ``range_m = 1000`` m is an arbitrary radarx choice.

    Parameters
    ----------
    p_return : float or array-like
        Received power from target [W]
    radar_const : float
        Radar constant (unitless)
    dielectric : float
        Dielectric factor (default is 0.93 for water)
    range_m : float or array-like
        Range to target [m]

    Returns
    -------
    float or array-like
        Reflectivity factor Z (mm^6/m^3 only if ``radar_const`` is scaled
        accordingly, see above)

    References
    ----------
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    .. [2] Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler
           Weather Radar: Principles and Applications*. Cambridge University
           Press, https://doi.org/10.1017/CBO9780511541094 (book; ISBN
           0-521-62384-7).
    """
    return (
        np.asarray(p_return) * np.asarray(range_m) ** 2 / (radar_const * dielectric**2)
    )


def differential_reflectivity(z_h, z_v):
    """
    Compute differential reflectivity ZDR [dB].

    ``ZDR = 10 log10(Z_h / Z_v)``, Eq. (3.168) of [1]_ (the ratio of the
    horizontally and vertically polarized reflectivities, from Eq. 2.54 for
    a single particle).

    Parameters
    ----------
    z_h : float or array-like
        Horizontal reflectivity [mm^6/m^3]
    z_v : float or array-like
        Vertical reflectivity [mm^6/m^3]

    Returns
    -------
    float or array-like
        ZDR in dB

    References
    ----------
    .. [1] Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler
           Weather Radar: Principles and Applications*. Cambridge University
           Press, https://doi.org/10.1017/CBO9780511541094 (book; ISBN
           0-521-62384-7).
    """
    return 10.0 * np.log10(np.asarray(z_h) / np.asarray(z_v))


def linear_depolarization_ratio(z_h, z_v):
    """
    Compute linear depolarization ratio LDR [dB].

    Returns ``10 log10(z_v / z_h)``. This is ``-ZDR``, not the LDR of [1]_:
    Eq. (3.169) defines ``LDR_vh = 10 log10(eta_vh / eta_hh)`` as the ratio
    of the CROSS-polar return (transmit H, receive V) to the co-polar
    return (Eq. 2.55 for a single particle). The result equals ``LDR_vh``
    only if ``z_v`` is the cross-polar reflectivity ``eta_vh`` and ``z_h``
    the co-polar ``eta_hh``; with horizontal and vertical co-polar
    reflectivities as named, it is simply ``-ZDR``. Behaviour unchanged.

    Parameters
    ----------
    z_h : float or array-like
        Horizontal reflectivity [mm^6/m^3]
    z_v : float or array-like
        Vertical reflectivity [mm^6/m^3]

    Returns
    -------
    float or array-like
        LDR in dB

    References
    ----------
    .. [1] Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler
           Weather Radar: Principles and Applications*. Cambridge University
           Press, https://doi.org/10.1017/CBO9780511541094 (book; ISBN
           0-521-62384-7). Eqs. (2.55), (3.169).
    """
    return 10.0 * np.log10(np.asarray(z_v) / np.asarray(z_h))


def circular_depolarization_ratio(z_parallel, z_orthogonal):
    """
    Compute circular depolarization ratio CDR [dB].

    Returns ``10 log10(z_parallel / z_orthogonal)``. In [1]_, Eq. (3.91a),
    ``CDR = 10 log10(|S_RR|**2 / |S_LR|**2)``, and by Eq. (3.91b)
    ``CDR = 10 log10(|alpha - alpha_zb|**2 / |alpha + alpha_zb|**2)``
    vanishes (-infinity dB) for a sphere, so the numerator of [1]_ is the
    depolarized (weak) channel. To obtain CDR in the convention of [1]_ pass
    that channel as ``z_parallel``; with the names as written (co-polar
    over cross-polar) the result is the negative of it.

    Parameters
    ----------
    z_parallel : float or array-like
        Power or reflectivity in the parallel channel [mm^6/m^3]
    z_orthogonal : float or array-like
        Power or reflectivity in the orthogonal channel [mm^6/m^3]

    Returns
    -------
    float or array-like
        CDR in dB

    References
    ----------
    .. [1] Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler
           Weather Radar: Principles and Applications*. Cambridge University
           Press, https://doi.org/10.1017/CBO9780511541094 (book; ISBN
           0-521-62384-7).
    """
    return 10.0 * np.log10(np.asarray(z_parallel) / np.asarray(z_orthogonal))


def radial_velocity(f_shift, wavelength):
    """
    Compute radial velocity from Doppler frequency shift.

    ``v = f_d lambda / 2``, the inverse of the two-way relation
    ``omega_d = 4 pi v / lambda`` (Eq. 3.30 of [1]_) with
    ``f_d = omega_d / (2 pi)``.

    Parameters
    ----------
    f_shift : float or array-like
        Doppler frequency shift [Hz]
    wavelength : float
        Radar wavelength [m]

    Returns
    -------
    float or array-like
        Radial velocity [m/s]

    References
    ----------
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    """
    return np.asarray(f_shift) * np.asarray(wavelength) / 2.0
