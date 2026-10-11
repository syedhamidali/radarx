"""
Common Utilities
================

Shared helper functions for internal use across radarx.fundamentals modules.

Array types: the converters accept scalars, lists and arrays, and return the
type they are given for :class:`xarray.DataArray` and dask input. A
DataArray keeps its dims, coords, name and attrs (``units`` and ``long_name``
are set for the converted quantity, and the attrs that describe the old
quantity, such as ``standard_name`` and ``valid_range``, are dropped when the
quantity changes), and a dask-backed DataArray or dask array stays lazy.
Scalars and lists give NumPy results as before.

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
import xarray as xr

# attributes that describe the quantity itself; they are wrong after a
# conversion that changes the quantity (dBZ to Z, m/s to knots keeps the
# quantity, see ``_describe``)
_QUANTITY_ATTRS = (
    "units",
    "long_name",
    "standard_name",
    "valid_min",
    "valid_max",
    "valid_range",
    "actual_range",
    "_FillValue",
    "missing_value",
    "flag_values",
    "flag_meanings",
)


def _prepare(x):
    """
    Return ``x`` unchanged if it supports NumPy ufuncs (ndarray, DataArray,
    dask array), otherwise ``np.asarray(x)`` (scalars, lists, tuples).
    """
    return x if hasattr(x, "__array_ufunc__") else np.asarray(x)


def _describe(source, result, units, long_name=None, same_quantity=False):
    """
    Give a converted DataArray the metadata of its source.

    Dims, coords and name are those of ``result`` (they come from ``source``
    through the arithmetic). The attrs of ``source`` are kept, except that the
    ones describing the quantity (``units``, ``long_name``, ``standard_name``,
    ``valid_range`` and so on) are dropped unless ``same_quantity`` is True
    (length and speed conversions, where only ``units`` changes). ``units``
    is set and ``long_name`` is set if given. Anything that is not a
    DataArray is returned as it is.
    """
    if not isinstance(source, xr.DataArray) or not isinstance(result, xr.DataArray):
        return result
    attrs = dict(source.attrs)
    if not same_quantity:
        for key in _QUANTITY_ATTRS:
            attrs.pop(key, None)
    attrs["units"] = units
    if long_name is not None:
        attrs["long_name"] = long_name
    elif same_quantity and "long_name" in source.attrs:
        attrs["long_name"] = source.attrs["long_name"]
    result.attrs = attrs
    return result


def _safe_log10(a):
    with np.errstate(divide="ignore"):
        return np.log10(a)


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
    z = _prepare(z)
    if isinstance(z, xr.DataArray):
        log_z = xr.apply_ufunc(
            _safe_log10, z, dask="parallelized", output_dtypes=[np.float64]
        )
        return _describe(z, 10 * log_z, "dBZ", "Reflectivity factor")
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
    dbz = _prepare(dbz)
    return _describe(dbz, 10 ** (dbz / 10), "mm6 m-3", "Reflectivity factor")


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
    meters = _prepare(meters)
    return _describe(meters, meters / 1000.0, "km", same_quantity=True)


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
    kilometers = _prepare(kilometers)
    return _describe(kilometers, kilometers * 1000.0, "m", same_quantity=True)


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
    knots = _prepare(knots)
    return _describe(knots, knots * 0.514444, "m s-1", same_quantity=True)


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
    mps = _prepare(mps)
    return _describe(mps, mps / 0.514444, "knot", same_quantity=True)


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
