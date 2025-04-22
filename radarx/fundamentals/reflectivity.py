"""
Reflectivity-Based Retrievals
=============================

Functions for reflectivity-based rainfall estimation, attenuation correction,
and reflectivity conversions. These functions are based on established
meteorological principles and are commonly used in radar meteorology.

References:
- Marshall, J. S., & Palmer, W. M. (1948). The distribution of raindrops with size. Journal of Meteorology, 5(4), 165-166.
- Bringi, V. N., & Chandrasekar, V. (2001). Polarimetric Doppler Weather Radar: Principles and Applications. Cambridge University Press.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   z_to_r_marshall_palmer
   z_to_r_custom
   dbz_attenuation_correction
"""

import numpy as np


def z_to_r_marshall_palmer(dbz):
    """
    Estimate rain rate [mm/hr] from reflectivity [dBZ] using Marshall-Palmer Z-R relation.

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
    Marshall, J. S., & Palmer, W. M. (1948). The distribution of raindrops with size. Journal of Meteorology, 5(4), 165-166.
    """
    z = 10.0 ** (0.1 * np.asarray(dbz))
    return (z / 200.0) ** (1.0 / 1.6)


def z_to_r_custom(dbz, a=200.0, b=1.6):
    """
    Custom Z-R relation to estimate rain rate [mm/hr].

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
    """
    z = 10.0 ** (0.1 * np.asarray(dbz))
    return (z / a) ** (1.0 / b)


def dbz_attenuation_correction(dbz, alpha=0.01, beta=0.85):
    """
    Simple attenuation correction using a power-law fit.

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
    """
    att = alpha * (np.maximum(0, dbz) ** beta)
    return dbz + att


__all__ = [
    "dbz_attenuation_correction",
    "z_to_r_custom",
    "z_to_r_marshall_palmer",
]
