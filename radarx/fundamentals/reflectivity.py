"""
Reflectivity-Based Retrievals
=============================

Functions for reflectivity-based rainfall estimation, attenuation correction,
and reflectivity conversions. These functions are based on established
meteorological principles and are commonly used in radar meteorology.

The Marshall-Palmer Z-R pair (Z = 200 R^1.6) is attributed to Marshall and
Palmer [1]_ by Rinehart [2]_ and Bringi and Chandrasekar [3]_; the paper [1]_
itself is not on disk and the pair was not checked against it.
:func:`dbz_attenuation_correction` is NOT a published method (see its
docstring).

References
----------
.. [1] Marshall, J. S., and W. McK. Palmer, 1948: The distribution of
       raindrops with size. *J. Meteor.*, **5** (4), 165-166,
       https://doi.org/10.1175/1520-0469(1948)005<0165:TDORWS>2.0.CO;2
.. [2] Rinehart, R. E., 1991: *Radar for Meteorologists*, 2nd ed. (3rd
       printing 1994). Rinehart Publications, Grand Forks, ND, ISBN
       0-9608700-7-5 (book, no DOI). Chapter 9, Section 9.2.
.. [3] Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler
       Weather Radar: Principles and Applications*. Cambridge University
       Press, https://doi.org/10.1017/CBO9780511541094 (book; ISBN
       0-521-62384-7). Eq. (8.8b).

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = [
    "dbz_attenuation_correction",
    "z_to_r_custom",
    "z_to_r_marshall_palmer",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np


def z_to_r_marshall_palmer(dbz):
    """
    Estimate rain rate [mm/hr] from reflectivity [dBZ] using Marshall-Palmer Z-R relation.

    Inverts ``Z = 200 R**1.6`` (``Z`` in mm^6 m^-3, ``R`` in mm/h):
    ``R = (Z / 200)**(1 / 1.6)``. The relation is attributed to Marshall and
    Palmer [1]_ and described as "the most commonly used Z-R relationship" in
    Chapter 9 of [2]_; [3]_ Eq. (8.8b) gives the equivalent
    ``R = 0.0365 Z**0.625`` (200**-0.625 = 0.0365). The constants were
    checked against [2]_ and [3]_ but not against the original paper [1]_.
    The relation applies to rain only; no hail cap is applied.

    Parameters
    ----------
    dbz : float or array-like
        Reflectivity in dBZ.

    Returns
    -------
    float or array-like
        Rain rate [mm/hr]

    References
    ----------
    .. [1] Marshall, J. S., and W. McK. Palmer, 1948: The distribution of
           raindrops with size. *J. Meteor.*, **5** (4), 165-166,
           https://doi.org/10.1175/1520-0469(1948)005<0165:TDORWS>2.0.CO;2
    .. [2] Rinehart, R. E., 1991: *Radar for Meteorologists*, 2nd ed. (3rd
           printing 1994). Rinehart Publications, Grand Forks, ND, ISBN
           0-9608700-7-5 (book, no DOI). Chapter 9, Section 9.2.
    .. [3] Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler
           Weather Radar: Principles and Applications*. Cambridge University
           Press, https://doi.org/10.1017/CBO9780511541094 (book; ISBN
           0-521-62384-7). Eq. (8.8b).
    """
    z = 10.0 ** (0.1 * np.asarray(dbz))
    return (z / 200.0) ** (1.0 / 1.6)


def z_to_r_custom(dbz, a=200.0, b=1.6):
    """
    Custom Z-R relation to estimate rain rate [mm/hr].

    Inverts the power law ``Z = a R**b``: ``R = (Z / a)**(1 / b)``, with
    ``Z`` in mm^6 m^-3. The power-law Z-R form is standard [1]_ (Eq. 8.8a
    writes it as ``R = c Z**a``, i.e. ``c = a**(-1/b)`` here); ``a`` and ``b``
    depend on the drop size distribution and must be chosen for the case
    [1]_. The defaults 200 and 1.6 are the Marshall-Palmer pair (see
    :func:`z_to_r_marshall_palmer`), not a recommendation for any storm type.

    Parameters
    ----------
    dbz : float or array-like
        Reflectivity in dBZ.
    a : float
        Z-R coefficient (default: 200)
    b : float
        Z-R exponent (default: 1.6)

    Returns
    -------
    float or array-like
        Rain rate [mm/hr]

    References
    ----------
    .. [1] Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler
           Weather Radar: Principles and Applications*. Cambridge University
           Press, https://doi.org/10.1017/CBO9780511541094 (book; ISBN
           0-521-62384-7). Section 8.1, Eq. (8.8a-c).
    """
    z = 10.0 ** (0.1 * np.asarray(dbz))
    return (z / a) ** (1.0 / b)


def dbz_attenuation_correction(dbz, alpha=0.01, beta=0.85):
    """
    Simple attenuation correction using a power-law fit.

    Adds ``alpha * max(0, dBZ)**beta`` dB to the observed reflectivity. This
    is an ad hoc radarx formula: the defaults ``alpha = 0.01`` and
    ``beta = 0.85`` are radarx choices, not taken from any publication, and
    the correction depends only on the local value, not on the path-integrated
    attenuation along the ray, so it is not a physically based correction.
    For a range-recursive method see Hitschfeld and Bordan [1]_ (not
    implemented here; cited only as the classic reference, not checked).

    Parameters
    ----------
    dbz : float or array-like
        Observed reflectivity [dBZ]
    alpha : float
        Attenuation coefficient scale factor (unitless)
    beta : float
        Attenuation exponent (unitless)

    Returns
    -------
    float or array-like
        Attenuation-corrected reflectivity [dBZ]

    References
    ----------
    .. [1] Hitschfeld, W., and J. Bordan, 1954: Errors inherent in the radar
           measurement of rainfall at attenuating wavelengths. *J. Meteor.*,
           **11** (1), 58-67,
           https://doi.org/10.1175/1520-0469(1954)011<0058:EIITRM>2.0.CO;2
    """
    att = alpha * (np.maximum(0, dbz) ** beta)
    return dbz + att
