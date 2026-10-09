"""
Common Utilities
================

Shared helper functions for internal use across radarx.fundamentals modules.

Unit conventions: reflectivity is converted with ``10 log10 Z`` for ``Z`` in
mm^6 m^-3 [1]_ [2]_. Length conversions use the exact factor 1000. The
knot conversion uses 1 kn = 1852 m/h exactly (international nautical mile;
BIPM SI brochure [3]_), i.e. 0.514444... m/s, and this module
uses the rounded factor 0.514444 whereas :mod:`radarx.core.conversion` uses
the rounded 1/1.943844 (the two agree to about 1e-6 relative).

References
----------
.. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
       Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
       DOI). p. 82, text after Eq. (4.33).
.. [2] Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler
       Weather Radar: Principles and Applications*. Cambridge University
       Press, https://doi.org/10.1017/CBO9780511541094 (book; ISBN
       0-521-62384-7). Chapter 3, text before Eq. (3.167a).
.. [3] BIPM, 2019: *The International System of Units (SI)*, 9th ed.
       Bureau International des Poids et Mesures, Sevres,
       https://www.bipm.org/en/publications/si-brochure (document, no DOI;
       document, no DOI).

.. module:: radarx.fundamentals.common
   :synopsis: Shared helper functions for unit conversion and internal use across radarx.fundamentals modules.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   dbz_from_z
   dbz_to_z
   ensure_positive
   kts_to_mps
   kts_to_si
   km_to_m
   km_to_si
   kilometers_to_meters
   kilometers_to_si
   linearize_dbz
   m_to_km
   meters_to_kilometers
   mps_to_kts
   mps_to_knots
   si_to_kilometers
   si_to_km
   si_to_kts
   z_to_dbz
"""

__all__ = [
    "dbz_from_z",
    "dbz_to_z",
    "ensure_positive",
    "kts_to_mps",
    "kts_to_si",
    "km_to_m",
    "km_to_si",
    "kilometers_to_meters",
    "kilometers_to_si",
    "linearize_dbz",
    "m_to_km",
    "meters_to_kilometers",
    "mps_to_kts",
    "mps_to_knots",
    "si_to_kilometers",
    "si_to_km",
    "si_to_kts",
    "z_to_dbz",
]

import numpy as np


def ensure_positive(value, name="value"):
    """
    Raise a ValueError if the input value is not positive.

    Parameters
    ----------
    value : float
        Numeric value to validate.
    name : str
        Optional name of the variable for error messages.

    Returns
    -------
    float
        The validated positive value.

    Raises
    ------
    ValueError
        If the value is not positive.
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def z_to_dbz(z):
    """
    Convert linear reflectivity Z [mm^6/m^3] to logarithmic dBZ.

    ``dBZ = 10 log10(Z)`` with ``Z`` in mm^6 m^-3: D&Z [1]_ (p. 82, after
    Eq. 4.33) and [2]_ (text before Eq. 3.167a, "decibels of Z relative to
    1 mm^6 m^-3"). ``Z = 0`` gives ``-inf``.

    Parameters
    ----------
    z : float or array-like
        Linear reflectivity [mm^6/m^3].

    Returns
    -------
    float or array-like
        Reflectivity in dBZ.

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
    z = np.asarray(z)
    with np.errstate(divide="ignore"):
        return 10 * np.log10(z)


def dbz_to_z(dbz):
    """
    Convert logarithmic reflectivity dBZ to linear Z [mm^6/m^3].

    ``Z = 10**(dBZ / 10)``, inverse of :func:`z_to_dbz` (same sources [1]_ [2]_).

    Parameters
    ----------
    dbz : float or array-like
        Reflectivity in dBZ.

    Returns
    -------
    float or array-like
        Linear reflectivity [mm^6/m^3].

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
    return 10 ** (np.asarray(dbz) / 10)


def meters_to_kilometers(meters):
    """
    Convert meters to kilometers.

    Parameters
    ----------
    meters : float or array-like
        Distance in meters.

    Returns
    -------
    float or array-like
        Distance in kilometers.
    """
    return np.asarray(meters) / 1000.0


def kilometers_to_meters(kilometers):
    """
    Convert kilometers to meters.

    Parameters
    ----------
    kilometers : float or array-like
        Distance in kilometers.

    Returns
    -------
    float or array-like
        Distance in meters.
    """
    return np.asarray(kilometers) * 1000.0


def knots_to_mps(knots):
    """
    Convert knots to meters per second.

    Multiplies by 0.514444, the six-digit rounding of 1852/3600 =
    0.5144444... (1 kn = 1 nautical mile per hour, 1 nautical mile = 1852 m
    exactly [1]_). The relative rounding error is about 1e-6.

    Parameters
    ----------
    knots : float or array-like
        Speed in knots.

    Returns
    -------
    float or array-like
        Speed in meters per second.

    References
    ----------
    .. [1] BIPM, 2019: *The International System of Units (SI)*, 9th ed.
           Bureau International des Poids et Mesures, Sevres,
           https://www.bipm.org/en/publications/si-brochure (document, no
           DOI).
    """
    return np.asarray(knots) * 0.514444


def mps_to_knots(mps):
    """
    Convert meters per second to knots.

    Divides by 0.514444, the rounded 1852/3600 (see :func:`knots_to_mps` and [1]_).

    Parameters
    ----------
    mps : float or array-like
        Speed in meters per second.

    Returns
    -------
    float or array-like
        Speed in knots.

    References
    ----------
    .. [1] BIPM, 2019: *The International System of Units (SI)*, 9th ed.
           Bureau International des Poids et Mesures, Sevres,
           https://www.bipm.org/en/publications/si-brochure (document, no
           DOI).
    """
    return np.asarray(mps) / 0.514444


def si_to_kilometers(value, unit="m"):
    """
    Convert SI distance to kilometers.

    Parameters
    ----------
    value : float or array-like
        Distance in SI units (meters by default).
    unit : str
        Unit of input value ("m" or "km").

    Returns
    -------
    float or array-like
        Distance in kilometers.
    """
    if unit == "m":
        return meters_to_kilometers(value)
    elif unit == "km":
        return value
    else:
        raise ValueError(f"Unsupported unit '{unit}' for distance.")


def kilometers_to_si(value):
    """
    Convert kilometers to meters (SI).

    Parameters
    ----------
    value : float or array-like
        Distance in kilometers.

    Returns
    -------
    float or array-like
        Distance in meters (SI).
    """
    return kilometers_to_meters(value)


def linearize_dbz(dbz):
    """
    Alias for dbz_to_z for compatibility with legacy terminology.

    Parameters
    ----------
    dbz : float or array-like
        Reflectivity in dBZ.

    Returns
    -------
    float or array-like
        Linear reflectivity Z [mm^6/m^3].
    """
    return dbz_to_z(dbz)


def dbz_from_z(z):
    """
    Alias for z_to_dbz for compatibility with legacy terminology.

    Parameters
    ----------
    z : float or array-like
        Linear reflectivity [mm^6/m^3].

    Returns
    -------
    float or array-like
        Reflectivity in dBZ.
    """
    return z_to_dbz(z)


def si_to_kts(value):
    """
    Convert meters per second to knots.

    Parameters
    ----------
    value : float or array-like
        Speed in meters per second.

    Returns
    -------
    float or array-like
        Speed in knots.
    """
    return mps_to_knots(value)


def kts_to_si(value):
    """
    Convert knots to meters per second.

    Parameters
    ----------
    value : float or array-like
        Speed in knots.

    Returns
    -------
    float or array-like
        Speed in meters per second.
    """
    return knots_to_mps(value)


def m_to_km(meters):
    """
    Convert meters to kilometers.

    Parameters
    ----------
    meters : float or array-like
        Distance in meters.

    Returns
    -------
    float or array-like
        Distance in kilometers.
    """
    return meters_to_kilometers(meters)


def km_to_m(kilometers):
    """
    Convert kilometers to meters.

    Parameters
    ----------
    kilometers : float or array-like
        Distance in kilometers.

    Returns
    -------
    float or array-like
        Distance in meters.
    """
    return kilometers_to_meters(kilometers)


def mps_to_kts(mps):
    """
    Convert meters per second to knots.

    Parameters
    ----------
    mps : float or array-like
        Speed in meters per second.

    Returns
    -------
    float or array-like
        Speed in knots.
    """
    return mps_to_knots(mps)


def kts_to_mps(knots):
    """
    Convert knots to meters per second.

    Parameters
    ----------
    knots : float or array-like
        Speed in knots.

    Returns
    -------
    float or array-like
        Speed in meters per second.
    """
    return knots_to_mps(knots)


def si_to_km(value):
    """
    Convert SI distance (meters) to kilometers.

    Parameters
    ----------
    value : float or array-like
        Distance in meters.

    Returns
    -------
    float or array-like
        Distance in kilometers.
    """
    return meters_to_kilometers(value)


def km_to_si(kilometers):
    """
    Convert kilometers to meters (SI).

    Parameters
    ----------
    kilometers : float or array-like
        Distance in kilometers.

    Returns
    -------
    float or array-like
        Distance in meters (SI).
    """
    return kilometers_to_meters(kilometers)
