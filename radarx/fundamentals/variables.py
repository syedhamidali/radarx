"""
Radar-Derived Variables
=======================

Functions related to reflectivity, differential reflectivity, and radial velocity.

Sources: reflectivity factor, Doppler relations and the 10 log10 definitions
follow Doviak and Zrnic [1]_ and Bringi and Chandrasekar [2]_ as detailed in
each docstring. The depolarization ratios are cross-polar over co-polar channel ratios; see
:func:`linear_depolarization_ratio` and :func:`circular_depolarization_ratio`
for which input is which channel.

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
    Compute reflectivity factor Z (m^3 with the SI radar constant, see below).

    Implements ``Z = P_r r**2 / (radar_const * dielectric)``, the
    inversion of the weather radar equation ``P_r = C |K|**2 Z / r**2``
    (Eqs. 4.14, 4.16, 4.31 of [1]_, with ``Z = lambda**4 eta / (pi**5 |K|**2)``,
    Eq. 3.167a of [2]_). ``dielectric`` is ``|K|**2`` itself (the literature
    value 0.93 for water, Section 3.2 of [1]_, p. 36), not ``|K|``. Before
    this release it was squared, which made the result about 7.5 percent
    (0.3 dB) too large with the default.

    Units follow :func:`radarx.fundamentals.system.radar_const`: with its SI
    constant (W/m), ``p_return`` in W and ``range_m`` in m the result is in
    m^3 (``m^6 m^-3``); multiply by 1e18 for mm^6 m^-3. The default
    ``range_m = 1000`` m is an arbitrary radarx choice.

    Parameters
    ----------
    p_return : float or array-like
        Received power from target [W]
    radar_const : float
        Radar constant (W/m for :func:`radarx.fundamentals.system.radar_const`)
    dielectric : float
        Dielectric factor ``|K|**2`` (default is 0.93 for water)
    range_m : float or array-like
        Range to target [m]

    Returns
    -------
    float or array-like
        Reflectivity factor Z (m^3 for the SI ``radar_const``, mm^6/m^3 only
        if ``radar_const`` is scaled accordingly, see above)

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
    return np.asarray(p_return) * np.asarray(range_m) ** 2 / (radar_const * dielectric)


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


def linear_depolarization_ratio(z_co, z_cross):
    """
    Compute linear depolarization ratio LDR [dB].

    ``LDR = 10 log10(z_cross / z_co)``: the ratio of the cross-polar return to
    the co-polar return, Eq. (3.169) of [1]_ (``LDR_vh = 10 log10(eta_vh /
    eta_hh)`` for horizontal transmission, Eq. 2.55 for a single particle).
    For ``LDR_vh`` pass the horizontal co-polar reflectivity ``Z_hh`` as
    ``z_co`` and the cross-polar reflectivity ``Z_vh`` (transmit H, receive V)
    as ``z_cross``. The ratio is negative in dB for ordinary hydrometeors.

    The arguments were called ``z_h`` and ``z_v`` before this release, which
    suggested the horizontal and vertical co-polar reflectivities. Fed with
    those, the result is ``-ZDR`` and not an LDR. The numbers for positional
    calls are unchanged.

    Parameters
    ----------
    z_co : float or array-like
        Co-polar reflectivity or power (``Z_hh``) [mm^6/m^3]
    z_cross : float or array-like
        Cross-polar reflectivity or power (``Z_vh``) [mm^6/m^3]

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
    return 10.0 * np.log10(np.asarray(z_cross) / np.asarray(z_co))


def circular_depolarization_ratio(z_parallel, z_orthogonal):
    """
    Compute circular depolarization ratio CDR [dB].

    ``CDR = 10 log10(z_parallel / z_orthogonal)``. In [1]_, Eq. (3.91a),
    ``CDR = 10 log10(|S_RR|**2 / |S_LR|**2)``, and by Eq. (3.91b) it vanishes
    (-infinity dB) for a sphere. The channels are named by the sense of the
    receive circular polarization relative to the transmit sense:

    - ``z_parallel`` is the channel received in the same circular sense as
      transmitted (``S_RR``). A sphere returns the opposite sense, so this is
      the weak, depolarized channel; it is the "cross-polar" channel of the
      AMS Glossary definition (cross-polar over co-polar).
    - ``z_orthogonal`` is the channel received in the opposite sense
      (``S_LR``), the strong one for spheres.

    With these inputs the result is the CDR of [1]_ and negative in dB. Do
    not pass the strong channel as ``z_parallel``: the result is then the
    negative of the CDR.

    Parameters
    ----------
    z_parallel : float or array-like
        Power or reflectivity of the same-sense channel, ``S_RR`` [mm^6/m^3]
    z_orthogonal : float or array-like
        Power or reflectivity of the opposite-sense channel, ``S_LR``
        [mm^6/m^3]

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
