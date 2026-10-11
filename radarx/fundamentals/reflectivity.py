"""
Reflectivity-Based Retrievals
=============================

Functions for reflectivity-based rainfall estimation, attenuation correction,
and reflectivity conversions. These functions are based on established
meteorological principles and are commonly used in radar meteorology.

The Marshall-Palmer Z-R pair (Z = 200 R^1.6) is attributed to Marshall and
Palmer [1]_ by Rinehart [2]_ and Bringi and Chandrasekar [3]_; the paper [1]_
itself was not consulted for the pair.
:func:`hitschfeld_bordan_correction` corrects the attenuation along the ray
with the solution of Hitschfeld and Bordan [4]_.
:func:`dbz_attenuation_correction` is not a published method, it is
deprecated (see its docstring).

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
.. [4] Hitschfeld, W., and J. Bordan, 1954: Errors inherent in the radar
       measurement of rainfall at attenuating wavelengths. *J. Meteor.*,
       **11** (1), 58-67,
       https://doi.org/10.1175/1520-0469(1954)011<0058:EIITRM>2.0.CO;2

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = [
    "dbz_attenuation_correction",
    "hitschfeld_bordan_correction",
    "z_to_r_custom",
    "z_to_r_marshall_palmer",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import warnings

import numpy as np
import xarray as xr

from .common import _describe, _prepare

try:
    from . import _hitschfeld_bordan

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _hitschfeld_bordan = None
    HAS_COMPILED_KERNEL = False


def z_to_r_marshall_palmer(dbz):
    """
    Estimate rain rate [mm/hr] from reflectivity [dBZ] using Marshall-Palmer Z-R relation.

    Inverts ``Z = 200 R**1.6`` (``Z`` in mm^6 m^-3, ``R`` in mm/h):
    ``R = (Z / 200)**(1 / 1.6)``. The relation is attributed to Marshall and
    Palmer [1]_ and described as "the most commonly used Z-R relationship" in
    Chapter 9 of [2]_; [3]_ Eq. (8.8b) gives the equivalent
    ``R = 0.0365 Z**0.625`` (200**-0.625 = 0.0365). The constants
    follow [2]_ and [3]_; not checked against the original paper [1]_.
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
    dbz = _prepare(dbz)
    z = 10.0 ** (0.1 * dbz)
    return _describe(dbz, (z / 200.0) ** (1.0 / 1.6), "mm h-1", "Rain rate")


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
    dbz = _prepare(dbz)
    z = 10.0 ** (0.1 * dbz)
    return _describe(dbz, (z / a) ** (1.0 / b), "mm h-1", "Rain rate")


def _hitschfeld_bordan_numpy(dbz, dr_km, a, b, max_pia):
    """
    NumPy implementation of the kernel ``_hitschfeld_bordan.pia`` (same results).

    ``dbz`` has the shape (rays, gates), ``dr_km`` is (gates,). Returns the
    two-way path-integrated attenuation [dB] at the centre of every gate, NaN
    where ``dbz`` is NaN.
    """
    c = 0.2 * b * np.log(10.0)
    floor_value = 10.0 ** (-b * max_pia / 10.0)
    missing = np.isnan(dbz)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        term = a * (10.0 ** (dbz / 10.0)) ** b * dr_km
    term = np.where(missing, 0.0, term)
    before = np.cumsum(term, axis=-1) - term
    centre = before + 0.5 * term
    with np.errstate(invalid="ignore"):
        rest = np.maximum(1.0 - c * centre, floor_value)
    pia = -(10.0 / b) * np.log10(rest)
    return np.where(missing, np.nan, pia)


def _use_compiled(engine):
    """Whether to run the compiled kernel for the requested ``engine``."""
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled Hitschfeld-Bordan kernel is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


def _pia_last_axis(dbz, dr_km, a, b, max_pia, use_compiled, n_threads):
    """The PIA of an array whose last axis is the range."""
    dbz = np.asarray(dbz, dtype=np.float64)
    flat = np.ascontiguousarray(dbz.reshape(-1, dbz.shape[-1]))
    dr_km = np.ascontiguousarray(dr_km, dtype=np.float64)
    if use_compiled and HAS_COMPILED_KERNEL:
        pia = _hitschfeld_bordan.pia(flat, dr_km, a, b, max_pia, int(n_threads or 0))
    else:
        pia = _hitschfeld_bordan_numpy(flat, dr_km, a, b, max_pia)
    return pia.reshape(dbz.shape)


def hitschfeld_bordan_correction(
    dbz,
    range_m=None,
    *,
    a,
    b,
    max_correction_db=10.0,
    range_dim="range",
    axis=-1,
    return_pia=False,
    n_threads=None,
    engine="auto",
    use_compiled=None,
):
    """
    Correct reflectivity for attenuation along the ray (Hitschfeld-Bordan).

    The specific attenuation is the power law ``k = a Z**b`` [dB km-1] of the
    unattenuated reflectivity ``Z`` [mm^6 m^-3]. The measured reflectivity
    is ``Zm = Z 10**(-0.2 A)`` with ``A`` the one-way integral of ``k`` along
    the ray (a two-way attenuation of ``2 A`` dB). Hitschfeld and Bordan [1]_
    solved this for ``Z`` from ``Zm`` alone,

    ``Z = Zm (1 - c S)**(-1/b)``, ``c = 0.2 b ln(10)``,
    ``S(r) = integral from 0 to r of a Zm**b ds`` (``s`` in km),

    so that the correction in dB is the path-integrated attenuation
    ``PIA = -(10 / b) log10(1 - c S)``, added to the measured dBZ. The closed
    form above is derived here from the power law and the definition of
    ``Zm``; the text of [1]_ could not be read for this, and the form agrees
    with a forward model of the attenuation in the tests.

    ``S`` is evaluated at the centre of each gate (the gates before it in
    full, half of the gate itself) with the gate length from the spacing of
    ``range_m``. Attenuation before the first gate is not included. A gate
    without a value (NaN) adds nothing to ``S`` and stays NaN.

    The solution is unstable: ``1 - c S`` goes to zero for strong
    attenuation and any error in ``a`` or ``b`` grows with range [1]_. The
    correction is therefore capped at ``max_correction_db`` (default 10 dB,
    a radarx choice) where ``1 - c S`` would fall below
    ``10**(-b max_correction_db / 10)``. The caller must treat capped gates
    as unreliable. The method needs ``a`` and ``b`` for the radar band, the
    temperature and the drop size distribution; radarx does not provide
    them. It replaces :func:`dbz_attenuation_correction`, which used only the
    local value of the reflectivity.

    The calculation runs in a compiled multithreaded kernel with a NumPy
    fallback of the same results (``engine="numpy"`` selects it).

    Parameters
    ----------
    dbz : xarray.DataArray or array-like
        Measured (attenuated) reflectivity [dBZ], with the range along
        ``range_dim`` (DataArray) or ``axis`` (array). A dask-backed DataArray
        stays lazy; the range dimension is then made a single chunk.
    range_m : array-like, optional
        Range of the gates [m], increasing. Taken from the coordinate
        ``range_dim`` of a DataArray if not given; required for arrays.
    a : float
        Coefficient of ``k = a Z**b`` [dB km^-1 (mm^6 m^-3)^-b].
    b : float
        Exponent of ``k = a Z**b``, positive.
    max_correction_db : float, optional
        Upper limit of the correction [dB], positive and finite.
    range_dim : str, optional
        Name of the range dimension of a DataArray.
    axis : int, optional
        Range axis of an array.
    return_pia : bool, optional
        Also return the correction ``PIA`` [dB] (two-way).
    n_threads : int, optional
        Threads of the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Kernel implementation. ``"auto"`` uses the compiled kernel if it was
        built.
    use_compiled : bool, optional
        Deprecated. ``False`` is ``engine="numpy"``.

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        Corrected reflectivity [dBZ] of the shape of ``dbz`` (a DataArray
        keeps its coords and attrs); with ``return_pia`` a tuple of the
        corrected reflectivity and the correction [dB].

    References
    ----------
    .. [1] Hitschfeld, W., and J. Bordan, 1954: Errors inherent in the radar
           measurement of rainfall at attenuating wavelengths. *J. Meteor.*,
           **11** (1), 58-67,
           https://doi.org/10.1175/1520-0469(1954)011<0058:EIITRM>2.0.CO;2
    """
    if use_compiled is not None:
        warnings.warn(
            "hitschfeld_bordan_correction(): the argument 'use_compiled' is "
            "deprecated and will be removed in a future release; use "
            "engine='numpy' for False",
            DeprecationWarning,
            stacklevel=2,
        )
        engine = "auto" if use_compiled else "numpy"
    compiled = _use_compiled(engine)
    a, b, cap = float(a), float(b), float(max_correction_db)
    if not (np.isfinite(a) and a >= 0.0):
        raise ValueError(f"a must be finite and not negative, got {a}")
    if not (np.isfinite(b) and b > 0.0):
        raise ValueError(f"b must be finite and positive, got {b}")
    if not (np.isfinite(cap) and cap > 0.0):
        raise ValueError(f"max_correction_db must be finite and positive, got {cap}")

    is_xarray = isinstance(dbz, xr.DataArray)
    if is_xarray:
        if range_dim not in dbz.dims:
            raise ValueError(f"{range_dim!r} is not a dimension of dbz {dbz.dims}")
        n_gates = dbz.sizes[range_dim]
        rng = dbz[range_dim].values if range_m is None else range_m
    else:
        if range_m is None:
            raise ValueError("range_m is required for input that is not a DataArray")
        dbz = np.asarray(dbz, dtype=np.float64)
        n_gates = dbz.shape[axis]
        rng = range_m
    rng = np.asarray(rng, dtype=np.float64)
    if rng.shape != (n_gates,):
        raise ValueError(f"range_m must have {n_gates} values, got shape {rng.shape}")
    if n_gates < 2 or not np.all(np.diff(rng) > 0):
        raise ValueError("range_m must have at least two gates and increase")
    options = dict(
        dr_km=np.gradient(rng) / 1000.0,
        a=a,
        b=b,
        max_pia=cap,
        use_compiled=compiled,
        n_threads=n_threads,
    )

    if is_xarray:
        pia = xr.apply_ufunc(
            _pia_last_axis,
            dbz,
            input_core_dims=[[range_dim]],
            output_core_dims=[[range_dim]],
            kwargs=options,
            dask="parallelized",
            output_dtypes=[np.float64],
            dask_gufunc_kwargs={"allow_rechunk": True},
        ).transpose(*dbz.dims)
        pia.attrs = {
            "units": "dB",
            "long_name": "Two-way path-integrated attenuation (Hitschfeld-Bordan)",
        }
        corrected = dbz + pia
        corrected.attrs = dict(dbz.attrs)
        corrected.attrs.setdefault("units", "dBZ")
    else:
        pia = np.moveaxis(
            _pia_last_axis(np.moveaxis(dbz, axis, -1), **options), -1, axis
        )
        corrected = dbz + pia
    return (corrected, pia) if return_pia else corrected


def dbz_attenuation_correction(dbz, alpha=0.01, beta=0.85):
    """
    Simple attenuation correction using a power-law fit (deprecated).

    Deprecated: use :func:`hitschfeld_bordan_correction`, which integrates the
    attenuation along the ray. This function is kept for one release.

    Adds ``alpha * max(0, dBZ)**beta`` dB to the observed reflectivity. This
    is an ad hoc radarx formula: the defaults ``alpha = 0.01`` and
    ``beta = 0.85`` are radarx choices, not taken from any publication, and
    the correction depends only on the local value, not on the path-integrated
    attenuation along the ray, so it is not a physically based correction.
    For a range-recursive method see Hitschfeld and Bordan [1]_
    (:func:`hitschfeld_bordan_correction`).

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
    warnings.warn(
        "dbz_attenuation_correction adds a correction from the local value only "
        "and is deprecated; use hitschfeld_bordan_correction, which integrates "
        "the attenuation along the ray.",
        DeprecationWarning,
        stacklevel=2,
    )
    att = alpha * (np.maximum(0, dbz) ** beta)
    return dbz + att
