#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx Quantitative Precipitation Estimation
============================================

Gate-wise rain rate from the reflectivity, the differential reflectivity, the
specific differential phase and the specific attenuation, and a blend that
chooses the relation from the hydrometeor class.

All relations are power laws, fitted to drop size distributions, with the rain
rate :math:`R` in mm h-1:

.. math::

    R(Z) = a\\,Z^{b}, \\qquad
    R(K_{DP}) = a\\,|K_{DP}|^{b}\\,\\mathrm{sign}(K_{DP}), \\qquad
    R(Z, Z_{DR}) = a\\,Z^{b}\\,z_{dr}^{c}, \\qquad
    R(A) = a\\,A^{b}

with :math:`Z` in mm6 m-3 (from the reflectivity in dBZ), :math:`z_{dr} =
10^{Z_{DR}/10}` the differential reflectivity in linear units,
:math:`K_{DP}` in degrees km-1 and the specific attenuation :math:`A` in
dB km-1. The coefficients depend on the radar band, and on the drop size
distribution (regime) and the temperature they were fitted for, so they are
explicit parameters (``coefficients``) with the defaults below.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}

Fields to prepare first
-----------------------
The functions take the fields as they are. A rain rate is only as good as its
inputs, so correct the fields before the estimate:

* the reflectivity and the differential reflectivity for attenuation and
  differential attenuation (both are biased low behind heavy rain, at C and X
  band strongly), and for the calibration of the radar (a 1 dB bias of
  :math:`Z_H` changes :math:`R(Z)` by about 18 %, and a 0.2 dB bias of
  :math:`Z_{DR}` changes :math:`R(Z, Z_{DR})` by about 7 %); the fit of ``R(Z, ZDR)`` is for the horizontal
  reflectivity and a differential reflectivity at the scan elevation of
  about 0.5 degrees;
* :math:`K_{DP}` from a processed (unfolded, offset-free, filtered) differential
  phase, see :func:`radarx.retrieve.estimate_kdp`; it is noisy below about
  0.3 degrees km-1, where :math:`R(K_{DP})` is not useful;
* for :func:`rain_rate_a`, the specific attenuation :math:`A` of a method that
  uses the differential phase, for example :math:`A = \\alpha K_{DP}` or the
  ZPHI method (Ryzhkov et al. 2014), because its advantage is that it does not
  depend on the reflectivity calibration or on partial beam blockage;
* for :func:`rain_rate`, the gates of non-meteorological echo masked
  (``mask`` of :func:`radarx.retrieve.hid`, which leaves them unclassified).

Defaults per band
-----------------
Every default is a value printed in a paper; nothing is fitted here. Where no
paper that could be read gives a relation for a band, there is no default and
the function raises an error that says so: pass ``coefficients``.

* **S band**: ``z`` 0.017, 0.714 (the WSR-88D relation), ``kdp`` 44.0, 0.822
  and ``z_zdr`` 1.42e-2, 0.770, -1.67 from Giangrande and Ryzhkov (2007), Eqs.
  1-3 (Ryzhkov et al. 2014, Eqs. 18-19 give the same ``z`` and ``kdp``);
  ``a`` from Table 1 of Ryzhkov et al. (2014).
* **C band**: ``a`` from Table 1 of Ryzhkov et al. (2014) only.
* **X band**: ``z`` 0.029, 0.67 and ``kdp`` 16.9, 0.801 from Ryzhkov et al.
  (2014), Eqs. 21-22 (the X-band radar BoXPol, Bonn, Germany); ``a`` from
  their Table 1.

* The S-band ``z_zdr`` and ``kdp`` relations were selected by the authors for
  rain of central Oklahoma, from 2002-2003 disdrometer and gauge data of the
  JPOLE experiment. The WSR-88D ``z`` relation is the operational one of the
  United States (:math:`Z = 300 R^{1.4}`). All three depend on the regime:
  they overestimate rain with large drops and underestimate rain with many
  small drops, which is why a tropical or maritime regime needs other
  coefficients.
* The X-band ``z`` relation is the Marshall-Palmer type relation used for the
  Bonn radar in Germany; the X-band ``kdp`` relation is the one used by
  Ryzhkov et al. (2014) for that radar. No X-band ``z_zdr``, and no C-band
  ``z``, ``kdp`` or ``z_zdr``, could be verified, so these have no default.
* ``a`` is the optimal :math:`R(A)` relation of Table 1 of Ryzhkov et al.
  (2014) for the polarization of the specific attenuation (``polarization``,
  horizontal by default), at 0, 10, 20 and 30 degrees Celsius for S, C and X
  band, simulated from 47 144 drop size distributions measured in Oklahoma
  (S band: :math:`R = 4.12\\times10^{3} A_h^{1.03}` at 20 degrees Celsius). The
  prefactor and the exponent are interpolated linearly in the temperature of
  the gate (``temperature``, default 20 degrees Celsius), as the authors
  suggest, and kept constant outside 0-30 degrees Celsius.

Hydrometeor-class blend
-----------------------
:func:`rain_rate` follows Giangrande and Ryzhkov (2008): the relation depends on
the class that :func:`radarx.retrieve.hid` assigns to the gate, and rain is
only estimated where a relation is valid. The classes are read from the
``flag_values`` and ``flag_meanings`` of the ``HID`` variable, so it works
with every method and band of :func:`radarx.retrieve.hid`. The defaults are:

==============================================================  ===========
Class (any method)                                              Relation
==============================================================  ===========
``drizzle``, ``rain``, ``light_and_moderate_rain``              ``z_zdr``
``heavy_rain``, ``big_drops``                                   ``z_zdr``
``rain_hail_mixture``                                           ``kdp``
all others (hail, graupel, snow, ice, wet snow, ...)            none, NaN
==============================================================  ===========

The first three rows and the ``kdp`` row follow the algorithm of Giangrande and
Ryzhkov (2007). Where no relation exists the rain rate is NaN and
``RAIN_RATE_FLAG`` says why. Hail, graupel, ice crystals, snow and the
melting layer (wet snow) are left without an estimate by default because no
single relation is valid there. That algorithm does estimate them, with the
relation of rain scaled by a factor fitted to gauges in Oklahoma
(wet snow 0.6 :math:`R(Z)`, graupel and rain-hail mixture above the freezing
level 0.8 :math:`R(Z)`, dry snow :math:`R(Z)`, crystals 2.8 :math:`R(Z)`),
which is not transferable to another radar or climate. Use ``class_relations``
to switch such a class on, for example ``{"wet_snow": ("z", 0.6)}``.
The ``z_zdr`` of those authors for heavy rain and big drops is the S-band
relation of this module; the "synthetic" algorithm of Ryzhkov et al. (2005),
which chooses among relations by the rain rate :math:`R(Z)` and uses another
:math:`R(Z, Z_{DR})`, is not implemented because its final coefficients could
not be verified.

Implementation
--------------
The relations and the blend run in a compiled, multithreaded C++ kernel that
takes blocks of gates from an atomic counter. An equivalent NumPy
implementation is the fallback and the test oracle. The Python functions work
on :class:`xarray.DataArray`, :class:`xarray.Dataset` and
:class:`xarray.DataTree` sweeps, keep dask arrays lazy, keep the coordinates
and never change their input.
"""

from __future__ import annotations

__all__ = [
    "rain_rate_z",
    "rain_rate_kdp",
    "rain_rate_z_zdr",
    "rain_rate_a",
    "rain_rate",
]

__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

import numpy as np
import xarray as xr

from .._compat import renamed_arguments
from .._provenance import provenance
from .._registry import accessor_method
from ._fieldtools import detach

try:
    from . import _qpe

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _qpe = None
    HAS_COMPILED_KERNEL = False

BANDS = ("S", "C", "X")
RELATIONS = ("z", "kdp", "z_zdr", "a")
_CODE = {name: i for i, name in enumerate(RELATIONS)}

#: Default ``(a, b)`` of ``z`` and ``kdp`` and ``(a, b, c)`` of ``z_zdr``, by band.
COEFFICIENTS = {
    "S": {
        "z": (1.70e-2, 0.714),
        "kdp": (44.0, 0.822),
        "z_zdr": (1.42e-2, 0.770, -1.67),
    },
    "C": {},
    "X": {
        "z": (0.029, 0.67),
        "kdp": (16.9, 0.801),
    },
}

#: Temperatures (degrees Celsius) of the R(A) relations of Ryzhkov et al. (2014), Table 1.
A_TEMPERATURES = (0.0, 10.0, 20.0, 30.0)

#: ``(a, b)`` of R(A) at ``A_TEMPERATURES``, by band and polarization (Table 1).
A_COEFFICIENTS = {
    "S": {
        "h": ((2.23e3, 1.03), (3.10e3, 1.03), (4.12e3, 1.03), (5.33e3, 1.03)),
        "v": ((3.02e3, 1.06), (4.12e3, 1.06), (5.51e3, 1.06), (7.19e3, 1.06)),
    },
    "C": {
        "h": ((221.0, 0.92), (250.0, 0.91), (294.0, 0.89), (352.0, 0.89)),
        "v": ((281.0, 0.95), (326.0, 0.94), (393.0, 0.93), (483.0, 0.93)),
    },
    "X": {
        "h": ((49.1, 0.87), (45.5, 0.83), (43.5, 0.79), (43.0, 0.76)),
        "v": ((57.8, 0.89), (53.3, 0.85), (51.1, 0.81), (51.0, 0.78)),
    },
}

#: Reflectivity (dBZ) above which Z is capped, 53 dBZ (Giangrande and Ryzhkov 2007).
Z_CAP = 53.0

#: Default relation of each hydrometeor class (see the module documentation).
DEFAULT_CLASS_RELATIONS = {
    "drizzle": "z_zdr",
    "rain": "z_zdr",
    "light_and_moderate_rain": "z_zdr",
    "heavy_rain": "z_zdr",
    "big_drops": "z_zdr",
    "rain_hail_mixture": "kdp",
}

FLAG_MEANINGS = (
    "estimated",
    "no_relation_for_class",
    "missing_input",
    "not_classified",
)
RELATION_MEANINGS = ("none",) + RELATIONS

_NAMES = {
    "dbz": ("DBZH", "DBZ", "reflectivity", "corrected_reflectivity"),
    "zdr": (
        "ZDR",
        "differential_reflectivity",
        "corrected_differential_reflectivity",
    ),
    "kdp": (
        "KDP",
        "specific_differential_phase",
        "corrected_specific_differential_phase",
    ),
    "a": ("AH", "specific_attenuation", "A"),
}
_WHAT = {
    "dbz": "reflectivity",
    "zdr": "differential reflectivity",
    "kdp": "specific differential phase",
    "a": "specific attenuation",
}
_NEEDS = {
    "z": ("dbz",),
    "kdp": ("kdp",),
    "z_zdr": ("dbz", "zdr"),
    "a": ("a",),
}
_KELVIN = ("K", "kelvin", "Kelvin", "degK")


# --------------------------------------------------------------------------
# coefficients and options
# --------------------------------------------------------------------------


class _Setup:
    """Coefficients and options of one call, in the form the kernel takes."""

    def __init__(self, coef, a_nodes, z_cap, min_zdr, negative):
        self.coef = [float(c) for c in coef]
        self.a_t, self.a_p0, self.a_p1 = (
            np.asarray(v, dtype=np.float64) for v in a_nodes
        )
        self.z_cap = float("nan") if z_cap is None else float(z_cap)
        self.zdr_min = float("nan") if min_zdr is None else float(min_zdr)
        self.negative = 1 if negative == "zero" else 0
        self.fixed_a = self.a_t.size == 1


def _check_band(band):
    if band not in BANDS:
        raise ValueError(f"band must be one of {BANDS}, not {band!r}")


def _check_options(negative, polarization):
    if negative not in ("keep", "zero"):
        raise ValueError(f"negative must be 'keep' or 'zero', not {negative!r}")
    if polarization not in ("h", "v"):
        raise ValueError(f"polarization must be 'h' or 'v', not {polarization!r}")


def _power_law(value, n, what):
    try:
        out = tuple(float(v) for v in value)
    except TypeError:
        raise ValueError(f"{what} must be a sequence of {n} numbers") from None
    if len(out) != n:
        raise ValueError(f"{what} must have {n} numbers, not {len(out)}")
    return out


def _coefficients(relations, band, coefficients, polarization):
    """
    The coefficients of the requested ``relations``.

    ``coefficients`` is a dict by relation name; a relation that is missing is
    taken from the defaults of the band. Returns the 7 power-law numbers
    (unused ones are 1) and the R(A) nodes ``(temperature, a, b)``.
    """
    _check_band(band)
    given = dict(coefficients or {})
    unknown = sorted(set(given) - set(RELATIONS))
    if unknown:
        raise ValueError(
            f"unknown relations {unknown} in coefficients, use {RELATIONS}"
        )
    sizes = {"z": 2, "kdp": 2, "z_zdr": 3, "a": 2}
    out = {}
    for rel in relations:
        if rel == "a":
            continue
        if rel in given:
            out[rel] = _power_law(given[rel], sizes[rel], f"coefficients[{rel!r}]")
        elif rel in COEFFICIENTS[band]:
            out[rel] = COEFFICIENTS[band][rel]
        else:
            n = sizes[rel]
            raise ValueError(
                f"there is no verified default for the {rel!r} relation at "
                f"{band} band; pass coefficients={{{rel!r}: {tuple('abc'[:n])}}} "
                "from a fit for your radar and rain regime"
            )
    z = out.get("z", (1.0, 1.0))
    k = out.get("kdp", (1.0, 1.0))
    zz = out.get("z_zdr", (1.0, 1.0, 1.0))
    if "a" in relations and "a" in given:
        pa = _power_law(given["a"], 2, "coefficients['a']")
        nodes = ([0.0], [pa[0]], [pa[1]])
    elif "a" in relations:
        table = A_COEFFICIENTS[band][polarization]
        nodes = (
            list(A_TEMPERATURES),
            [row[0] for row in table],
            [row[1] for row in table],
        )
    else:
        nodes = ([0.0], [1.0], [1.0])
    return [*z, *k, *zz], nodes


# --------------------------------------------------------------------------
# NumPy implementation of the kernel (same results)
# --------------------------------------------------------------------------


def _signed_power(c, x, e, negative):
    with np.errstate(invalid="ignore"):
        mag = c * np.power(np.abs(x), e)
        if negative == 1:
            out = np.where(x > 0, mag, 0.0)
        else:
            out = np.where(x > 0, mag, np.where(x == 0, 0.0, -mag))
    return np.where(np.isnan(x), np.nan, out)


def _rate_numpy(rel, dbz, zdr, kdp, a, t, s):
    """Rain rate of relation ``rel`` (code) on arrays that broadcast together."""
    cap = np.inf if np.isnan(s.z_cap) else s.z_cap
    zmin = -np.inf if np.isnan(s.zdr_min) else s.zdr_min
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        if rel == 0:
            z = np.power(10.0, np.minimum(dbz, cap) / 10.0)
            return s.coef[0] * np.power(z, s.coef[1])
        if rel == 1:
            return _signed_power(s.coef[2], kdp, s.coef[3], s.negative)
        if rel == 2:
            z = np.power(10.0, np.minimum(dbz, cap) / 10.0)
            zd = np.power(10.0, np.maximum(zdr, zmin) / 10.0)
            return s.coef[4] * np.power(z, s.coef[5]) * np.power(zd, s.coef[6])
        if s.a_t.size == 1:
            return _signed_power(s.a_p0[0], a, s.a_p1[0], s.negative)
        p0 = np.interp(t, s.a_t, s.a_p0)
        p1 = np.interp(t, s.a_t, s.a_p1)
        return np.where(np.isnan(t), np.nan, _signed_power(p0, a, p1, s.negative))


def _blend_numpy(cls, dbz, zdr, kdp, a, t, lut_rel, lut_fac, s):
    """NumPy implementation of the kernel ``_qpe.blend`` (same results)."""
    cls = np.asarray(cls)
    shape = cls.shape
    nl = lut_rel.size
    inside = (cls > 0) & (cls < nl)
    safe = np.where(inside, cls, 0)
    rel = np.where(inside, lut_rel[safe], -1)
    fac = lut_fac[safe]
    rate = np.full(shape, np.nan)
    for code in range(4):
        use = rel == code
        if use.any():
            r = _rate_numpy(code, dbz, zdr, kdp, a, t, s)
            rate = np.where(use, r, rate)
    flag = np.full(shape, 3, dtype=np.int8)
    flag = np.where(cls > 0, np.int8(1), flag)
    has = rel >= 0
    flag = np.where(has & np.isnan(rate), np.int8(2), flag)
    ok = has & ~np.isnan(rate)
    flag = np.where(ok, np.int8(0), flag)
    out = np.where(ok, rate * fac, np.nan)
    used = np.where(ok, rel + 1, 0).astype(np.int8)
    return out, flag.astype(np.int8), used


# --------------------------------------------------------------------------
# engine selection and the array level
# --------------------------------------------------------------------------


def _use_compiled(engine):
    """Whether to run the compiled kernel for the requested ``engine``."""
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled QPE kernel is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


def _flat(values, shape):
    """Contiguous float64 array of ``values`` broadcast to ``shape`` (size 1: one value)."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 1 and values.ndim == 0:
        return np.ascontiguousarray(values.reshape(1))
    return np.ascontiguousarray(np.broadcast_to(values, shape).reshape(-1))


def _kernel_inputs(shape, dbz, zdr, kdp, a, t):
    nan = np.full(1, np.nan)
    return [nan if v is None else _flat(v, shape) for v in (dbz, zdr, kdp, a, t)]


def _rate_array(rel, dbz, zdr, kdp, a, t, setup, use_compiled, n_threads):
    """Rain rate (float32) of one relation on NumPy arrays or scalars."""
    arrays = [v for v in (dbz, zdr, kdp, a, t) if v is not None]
    shape = np.broadcast_shapes(*[np.shape(v) for v in arrays])
    n = int(np.prod(shape))
    if use_compiled:
        flat = _kernel_inputs(shape, dbz, zdr, kdp, a, t)
        out = _qpe.relation(
            rel,
            *flat,
            n,
            setup.coef,
            setup.a_t,
            setup.a_p0,
            setup.a_p1,
            setup.z_cap,
            setup.zdr_min,
            setup.negative,
            int(n_threads or 0),
        )
        out = out.reshape(shape)
    else:
        nan = np.nan
        vals = [
            nan if v is None else np.asarray(v, dtype=np.float64)
            for v in (dbz, zdr, kdp, a, t)
        ]
        out = np.broadcast_to(_rate_numpy(rel, *vals, setup), shape)
    return out.astype(np.float32)


def _blend_array(
    cls, dbz, zdr, kdp, a, t, lut_rel, lut_fac, setup, use_compiled, n_threads
):
    """Blended rain rate (float32), flag and relation (int8) on NumPy arrays."""
    cls = np.asarray(cls)
    if cls.dtype.kind == "f":
        cls = np.nan_to_num(cls, nan=0.0)
    cls = cls.astype(np.int32)
    arrays = [cls] + [v for v in (dbz, zdr, kdp, a, t) if v is not None]
    shape = np.broadcast_shapes(*[np.shape(v) for v in arrays])
    cls = np.ascontiguousarray(np.broadcast_to(cls, shape))
    if use_compiled:
        flat = _kernel_inputs(shape, dbz, zdr, kdp, a, t)
        out, flag, used = _qpe.blend(
            cls.reshape(-1),
            *flat,
            lut_rel,
            lut_fac,
            setup.coef,
            setup.a_t,
            setup.a_p0,
            setup.a_p1,
            setup.z_cap,
            setup.zdr_min,
            setup.negative,
            int(n_threads or 0),
        )
        return (
            out.reshape(shape).astype(np.float32),
            flag.reshape(shape),
            used.reshape(shape),
        )
    vals = [
        np.nan if v is None else np.asarray(v, dtype=np.float64)
        for v in (dbz, zdr, kdp, a, t)
    ]
    out, flag, used = _blend_numpy(cls, *vals, lut_rel, lut_fac, setup)
    return out.astype(np.float32), flag, used


# --------------------------------------------------------------------------
# fields
# --------------------------------------------------------------------------


def _find(ds, spec, key):
    """A field of the dataset ``ds`` as a DataArray: by name, by default name or as given."""
    if isinstance(spec, xr.DataArray):
        return spec
    if ds is None:
        raise ValueError(
            f"the {_WHAT[key]} must be passed as a DataArray when obj is a DataArray"
        )
    if spec is not None:
        if spec not in ds:
            raise KeyError(f"{spec!r} is not in the dataset")
        return ds[spec]
    for cand in _NAMES[key]:
        if cand in ds:
            return ds[cand]
    raise KeyError(f"no {_WHAT[key]} found among {_NAMES[key]}; pass the field name")


def _temperature(ds, spec):
    """Temperature (degrees Celsius) as a float, a DataArray or None for the default."""
    if spec is None:
        return 20.0
    if isinstance(spec, (int, float, np.integer, np.floating)):
        return float(spec)
    if isinstance(spec, str) and ds is None:
        raise ValueError("pass the temperature as a DataArray or a number")
    da = ds[spec] if isinstance(spec, str) else spec
    if not isinstance(da, xr.DataArray):
        raise TypeError("temperature must be a number, a field name or a DataArray")
    units = da.attrs.get("units")
    values = da.values if not hasattr(da.data, "dask") else None
    kelvin = units in _KELVIN or (
        units is None
        and values is not None
        and np.isfinite(values).any()
        and np.nanmedian(values) > 100.0
    )
    return da - 273.15 if kelvin else da


def _sweep_names(dtree):
    return [n for n in dtree.children if n.startswith("sweep")]


def _is_tree(obj):
    return isinstance(obj, xr.DataTree)


def _describe(rel, band, setup, polarization, temperature):
    """Text of the relation and the coefficients that were used."""
    c = setup.coef
    cap = "" if np.isnan(setup.z_cap) else f", Z capped at {setup.z_cap:g} dBZ"
    if rel == "z":
        return f"R = {c[0]:g} Z^{c[1]:g}{cap}"
    if rel == "kdp":
        return f"R = {c[2]:g} |KDP|^{c[3]:g} sign(KDP)"
    if rel == "z_zdr":
        return f"R = {c[4]:g} Z^{c[5]:g} zdr^{c[6]:g}{cap}"
    if setup.fixed_a:
        return f"R = {setup.a_p0[0]:g} A^{setup.a_p1[0]:g}"
    return (
        f"R = a(T) A^b(T), Table 1 of Ryzhkov et al. (2014), band {band}, "
        f"polarization {polarization}, linearly interpolated in temperature"
    )


# --------------------------------------------------------------------------
# single relations
# --------------------------------------------------------------------------

_LONG_NAME = {
    "z": "rain rate from reflectivity, R(Z)",
    "kdp": "rain rate from specific differential phase, R(KDP)",
    "z_zdr": "rain rate from reflectivity and differential reflectivity, R(Z,ZDR)",
    "a": "rain rate from specific attenuation, R(A)",
}
_OUT_NAME = {
    "z": "RAIN_RATE_Z",
    "kdp": "RAIN_RATE_KDP",
    "z_zdr": "RAIN_RATE_Z_ZDR",
    "a": "RAIN_RATE_A",
}


def _rate_attrs(rel, text):
    return {
        "standard_name": "rainfall_rate",
        "long_name": _LONG_NAME[rel],
        "units": "mm h-1",
        "comment": text,
    }


def _rate_dataarray(rel, fields, temperature, setup, text, engine, n_threads):
    """Apply the relation to DataArrays (lazy for dask)."""
    use_compiled = _use_compiled(engine)
    keys = [k for k in ("dbz", "zdr", "kdp", "a") if k in fields]
    arrays = [fields[k] for k in keys]
    scalar_t = None
    if rel == _CODE["a"]:
        if setup.fixed_a:
            scalar_t = 0.0
        elif isinstance(temperature, xr.DataArray):
            keys.append("t")
            arrays.append(temperature)
        else:
            scalar_t = temperature

    def func(*vals):
        got = dict(zip(keys, vals))
        return _rate_array(
            rel,
            got.get("dbz"),
            got.get("zdr"),
            got.get("kdp"),
            got.get("a"),
            got.get("t", scalar_t),
            setup,
            use_compiled,
            n_threads,
        )

    out = xr.apply_ufunc(
        func,
        *detach(*arrays),
        dask="parallelized",
        output_dtypes=[np.float32],
        keep_attrs=False,
    )
    out.name = _OUT_NAME[RELATIONS[rel]]
    out.attrs = _rate_attrs(RELATIONS[rel], text)
    return out


def _single(
    obj,
    rel,
    specs,
    band,
    coefficients,
    polarization,
    z_cap,
    min_zdr,
    negative,
    temperature,
    engine,
    n_threads,
):
    """Driver of the four single relations for a DataArray, Dataset or DataTree."""
    _check_options(negative, polarization)
    _use_compiled(engine)
    name = RELATIONS[rel]
    coef, nodes = _coefficients([name], band, coefficients, polarization)
    setup = _Setup(coef, nodes, z_cap, min_zdr, negative)
    text = _describe(name, band, setup, polarization, temperature)
    needs = _NEEDS[name]

    def one(ds, primary=None):
        fields = {}
        for key in needs:
            spec = specs.get(key)
            if primary is not None and key == needs[0]:
                fields[key] = primary
            else:
                fields[key] = _find(ds, spec, key)
        t = _temperature(ds, temperature) if name == "a" else None
        return _rate_dataarray(rel, fields, t, setup, text, engine, n_threads)

    if isinstance(obj, xr.DataArray):
        return one(None, primary=obj)
    if isinstance(obj, xr.Dataset):
        return one(obj)
    if not _is_tree(obj):
        raise TypeError("obj must be an xarray.DataArray, Dataset or DataTree")
    nodes_out = {"/": obj.root.to_dataset(inherit=False)}
    for sweep in _sweep_names(obj):
        ds = obj[sweep].to_dataset()
        try:
            for key in needs:
                _find(ds, specs.get(key), key)
        except KeyError:
            continue
        nodes_out[sweep] = one(ds).to_dataset()
    if len(nodes_out) == 1:
        raise KeyError(f"no sweep contains the fields {[_WHAT[k] for k in needs]}")
    return xr.DataTree.from_dict(nodes_out)


_Z_CAP_DOC = """    z_cap : float or None, optional
        Reflectivity (dBZ) above which Z is set to this value, to limit the
        contribution of hail. Default 53 dBZ (Giangrande and Ryzhkov 2007);
        ``None`` for no cap.
"""

_COMMON_DOC = """    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled
        kernel and falls back to NumPy."""


def _fill(func, coefficients_doc, z_cap=True):
    """Put the shared parameter text into the docstring of ``func``."""
    text = (
        '    band : {"S", "C", "X"}, optional\n'
        '        Radar band, which selects the default coefficients. Default ``"S"``.\n'
        "    coefficients : sequence of float, optional\n"
        f"        {coefficients_doc}, required where there is no default for the\n"
        "        band.\n" + (_Z_CAP_DOC if z_cap else "") + _COMMON_DOC
    )
    func.__doc__ = func.__doc__.replace("    {common}", text)


@provenance("Rain rate from reflectivity, R(Z) power law")
def rain_rate_z(
    obj,
    dbz=None,
    *,
    band="S",
    coefficients=None,
    z_cap=Z_CAP,
    n_threads=None,
    engine="auto",
):
    """
    Rain rate from the reflectivity, ``R = a Z**b``.

    Z is the reflectivity in mm6 m-3 from the horizontal reflectivity in dBZ.
    The default for S band is the operational WSR-88D relation
    ``R = 1.70e-2 Z**0.714`` (:math:`Z = 300 R^{1.4}`), the one for X band the
    Marshall-Palmer type relation ``R = 0.029 Z**0.67`` used for the Bonn
    radar; see :mod:`radarx.retrieve.qpe` for the sources and for what to
    correct before using it. There is no default for C band.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        The reflectivity (dBZ) as a DataArray, a sweep or grid that holds it,
        or a volume with ``sweep_*`` groups (sweeps without the field are
        skipped).
    dbz : str, optional
        Name of the reflectivity. By default the first found of ``DBZH``,
        ``DBZ``, ``reflectivity`` and ``corrected_reflectivity``. Correct the
        reflectivity for attenuation and calibration first and pass its name.
    {common}

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        ``RAIN_RATE_Z`` (mm h-1) on the coordinates of the reflectivity (float32,
        NaN where the reflectivity is missing); for a DataTree one node per
        sweep with that variable. The input is not changed.

    Raises
    ------
    KeyError
        If the reflectivity is not found.
    ValueError
        For an unknown ``band`` or ``engine``, a ``coefficients`` of the wrong
        length, or a band without a default and without ``coefficients``.

    References
    ----------
    Ryzhkov, A., M. Diederich, P. Zhang, and C. Simmer, 2014: Potential
    utilization of specific attenuation for rainfall estimation, mitigation of
    partial beam blockage, and radar networking. *J. Atmos. Oceanic
    Technol.*, **31** (3), 599-619,
    https://doi.org/10.1175/JTECH-D-13-00038.1 (Eq. 18, S band; Eq. 21,
    X band).

    Giangrande, S. E., and A. V. Ryzhkov, 2008: Estimation of rainfall based
    on the results of polarimetric echo classification. *J. Appl. Meteor.
    Climatol.*, **47** (9), 2445-2462,
    https://doi.org/10.1175/2008JAMC1753.1 (the relations and the 53 dBZ
    cap are taken from the extended abstract of the same title, 33rd Conf. on
    Radar Meteorology, 2007, paper P6B.7, Eq. 1; the journal article was not
    available).

    Examples
    --------
    >>> rr = radarx.retrieve.rain_rate_z(sweep)  # doctest: +SKIP
    >>> rr = dtree.radarx.rain_rate_z(dbz="DBZH_corrected")  # doctest: +SKIP
    """
    return _single(
        obj,
        0,
        {"dbz": dbz},
        band,
        {"z": coefficients} if coefficients is not None else None,
        "h",
        z_cap,
        None,
        "keep",
        None,
        engine,
        n_threads,
    )


@provenance("Rain rate from specific differential phase, R(KDP) power law")
def rain_rate_kdp(
    obj,
    kdp=None,
    *,
    band="S",
    coefficients=None,
    negative="keep",
    n_threads=None,
    engine="auto",
):
    """
    Rain rate from the specific differential phase, ``R = a |KDP|**b sign(KDP)``.

    :math:`K_{DP}` is in degrees km-1. The default for S band is
    ``R = 44.0 |KDP|**0.822`` of Giangrande and Ryzhkov (2007); for X band
    ``R = 16.9 |KDP|**0.801`` as used by Ryzhkov et al. (2014) for the Bonn
    radar (see :mod:`radarx.retrieve.qpe`). There is no default for C band.
    The relation is independent of the reflectivity calibration and of
    attenuation, but KDP is noisy below about 0.3 degrees km-1 and
    the relation is not meant for light rain.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        KDP as a DataArray, a sweep or grid that holds it, or a volume (sweeps
        without the field are skipped).
    kdp : str, optional
        Name of the specific differential phase. By default the first found of
        ``KDP``, ``specific_differential_phase`` and
        ``corrected_specific_differential_phase``; see
        :func:`radarx.retrieve.estimate_kdp`.
    negative : {"keep", "zero"}, optional
        What to do with negative KDP (noise). ``"keep"`` (default) gives a
        negative rain rate with the sign of KDP, as the relation is
        written in the paper, so that averages are not biased; ``"zero"`` gives
        zero (a radarx option).
    {common}

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        ``RAIN_RATE_KDP`` (mm h-1) on the coordinates of KDP (float32, NaN where KDP
        is missing); for a DataTree one node per sweep. The input is not
        changed.

    Raises
    ------
    KeyError
        If KDP is not found.
    ValueError
        For an unknown ``band``, ``engine`` or ``negative``, a ``coefficients``
        of the wrong length, or a band without a default and without
        ``coefficients``.

    References
    ----------
    Ryzhkov, A., M. Diederich, P. Zhang, and C. Simmer, 2014: Potential
    utilization of specific attenuation for rainfall estimation, mitigation of
    partial beam blockage, and radar networking. *J. Atmos. Oceanic
    Technol.*, **31** (3), 599-619,
    https://doi.org/10.1175/JTECH-D-13-00038.1 (Eq. 19, S band; Eq. 22,
    X band).

    Giangrande, S. E., and A. V. Ryzhkov, 2008: Estimation of rainfall based
    on the results of polarimetric echo classification. *J. Appl. Meteor.
    Climatol.*, **47** (9), 2445-2462,
    https://doi.org/10.1175/2008JAMC1753.1 (the S-band relation is taken from
    the extended abstract of the same title, 33rd Conf. on Radar Meteorology,
    2007, paper P6B.7, Eq. 2; the journal article was not available).

    Examples
    --------
    >>> rr = radarx.retrieve.rain_rate_kdp(sweep)  # doctest: +SKIP
    >>> rr = dtree.radarx.rain_rate_kdp(negative="zero")  # doctest: +SKIP
    """
    return _single(
        obj,
        1,
        {"kdp": kdp},
        band,
        {"kdp": coefficients} if coefficients is not None else None,
        "h",
        None,
        None,
        negative,
        None,
        engine,
        n_threads,
    )


_fill(
    rain_rate_kdp,
    "``(a, b)`` of the relation instead of the default of the band",
    z_cap=False,
)


@renamed_arguments(zdr_min="min_zdr")
@provenance("Rain rate from reflectivity and differential reflectivity, R(Z, ZDR)")
def rain_rate_z_zdr(
    obj,
    dbz=None,
    zdr=None,
    *,
    band="S",
    coefficients=None,
    z_cap=Z_CAP,
    min_zdr=None,
    n_threads=None,
    engine="auto",
):
    """
    Rain rate from the reflectivity and the differential reflectivity,
    ``R = a Z**b zdr**c``.

    ``zdr`` is the differential reflectivity in linear units,
    ``10**(ZDR/10)``. The differential reflectivity corrects the dependence
    of ``R(Z)`` on the drop size: large drops (high ZDR) give a lower rain rate
    for the same reflectivity. The default for S band is the relation
    ``R = 1.42e-2 Z**0.770 zdr**-1.67`` that Giangrande and Ryzhkov (2007)
    selected for central Oklahoma; there is no default for C and X band. The
    differential reflectivity must be calibrated (a bias of 0.2 dB changes the
    rain rate by about 7 %), and corrected for differential attenuation; see
    :mod:`radarx.retrieve.qpe`.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        The reflectivity (dBZ) as a DataArray (then pass ``zdr`` as a
        DataArray on the same coordinates), a sweep or grid that holds both
        fields, or a volume (sweeps without both fields are skipped).
    dbz : str, optional
        Name of the reflectivity. By default the first found of ``DBZH``,
        ``DBZ``, ``reflectivity`` and ``corrected_reflectivity``.
    zdr : str or xarray.DataArray, optional
        Name of the differential reflectivity (dB), by default the first found
        of ``ZDR``, ``differential_reflectivity`` and
        ``corrected_differential_reflectivity``, or the field itself.
    min_zdr : float or None, optional
        Differential reflectivity (dB) below which the value is replaced by
        this one, because the relation grows quickly for negative values (a
        radarx option). Default None: no limit.
    {common}

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        ``RAIN_RATE_Z_ZDR`` (mm h-1) on the coordinates of the reflectivity (float32,
        NaN where a field is missing); for a DataTree one node per sweep. The
        input is not changed.

    Raises
    ------
    KeyError
        If a field is not found.
    ValueError
        For an unknown ``band`` or ``engine``, a ``coefficients`` of the wrong
        length, or a band without a default and without ``coefficients``.

    References
    ----------
    Giangrande, S. E., and A. V. Ryzhkov, 2008: Estimation of rainfall based
    on the results of polarimetric echo classification. *J. Appl. Meteor.
    Climatol.*, **47** (9), 2445-2462,
    https://doi.org/10.1175/2008JAMC1753.1 (the relation is taken from the
    extended abstract of the same title, 33rd Conf. on Radar Meteorology,
    2007, paper P6B.7, Eq. 3; the journal article was not available).

    Ryzhkov, A. V., S. E. Giangrande, and T. J. Schuur, 2005: Rainfall
    estimation with a polarimetric prototype of WSR-88D. *J. Appl. Meteor.*,
    **44** (4), 502-515, https://doi.org/10.1175/JAM2213.1 (the form of the
    relation; the same coefficients are in a conference preprint of that
    paper, the journal article was not available).

    Examples
    --------
    >>> rr = radarx.retrieve.rain_rate_z_zdr(sweep)  # doctest: +SKIP
    >>> rr = dtree.radarx.rain_rate_z_zdr(min_zdr=0.0)  # doctest: +SKIP
    """
    return _single(
        obj,
        2,
        {"dbz": dbz, "zdr": zdr},
        band,
        {"z_zdr": coefficients} if coefficients is not None else None,
        "h",
        z_cap,
        min_zdr,
        "keep",
        None,
        engine,
        n_threads,
    )


@provenance("Rain rate from specific attenuation, R(A) power law")
def rain_rate_a(
    obj,
    a=None,
    *,
    band="S",
    coefficients=None,
    temperature=None,
    polarization="h",
    negative="keep",
    n_threads=None,
    engine="auto",
):
    """
    Rain rate from the specific attenuation, ``R = a(T) A**b(T)``.

    ``A`` is the specific attenuation in dB km-1, estimated from the
    differential phase (for example :math:`A = \\alpha K_{DP}` or the ZPHI
    method, see :mod:`radarx.retrieve.qpe`). The rain rate then does not depend
    on the reflectivity calibration, on attenuation or on partial beam
    blockage, and the relation is little affected by the drop size
    distribution (Ryzhkov et al. 2014). The default is the optimal relation of
    their Table 1 for the band, polarization and temperature, for example
    ``R = 4.12e3 A**1.03`` at S band and 20 degrees Celsius (their Eq. 24).

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        The specific attenuation as a DataArray, a sweep or grid that holds it,
        or a volume (sweeps without the field are skipped).
    a : str, optional
        Name of the specific attenuation (dB km-1). By default the first found
        of ``AH``, ``specific_attenuation`` and ``A``.
    temperature : float, str or xarray.DataArray, optional
        Temperature (degrees Celsius, or K if the ``units`` say so) of the
        rain at each gate, a number, the name of a field or a field. The
        coefficients are interpolated linearly between 0, 10, 20 and 30
        degrees and constant outside that range. Default 20 degrees Celsius.
        Ignored when ``coefficients`` is given.
    polarization : {"h", "v"}, optional
        Polarization of the specific attenuation. Default ``"h"``.
    negative : {"keep", "zero"}, optional
        What to do with negative A (noise): keep the sign (default) or give
        zero (a radarx option).
    {common}

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        ``RAIN_RATE_A`` (mm h-1) on the coordinates of A (float32, NaN where A or
        the temperature is missing); for a DataTree one node per sweep. The
        input is not changed.

    Raises
    ------
    KeyError
        If the specific attenuation is not found.
    ValueError
        For an unknown ``band``, ``polarization``, ``negative`` or ``engine``,
        or a ``coefficients`` of the wrong length.

    Notes
    -----
    The table was fitted to drop size distributions measured in Oklahoma at
    S, C and X band. R(A) relies on the specific attenuation being right: with
    attenuation estimated from a path-integrated differential phase it fails in
    areas of hail (Ryzhkov et al. 2014, section 6).

    References
    ----------
    Ryzhkov, A., M. Diederich, P. Zhang, and C. Simmer, 2014: Potential
    utilization of specific attenuation for rainfall estimation, mitigation of
    partial beam blockage, and radar networking. *J. Atmos. Oceanic
    Technol.*, **31** (3), 599-619,
    https://doi.org/10.1175/JTECH-D-13-00038.1 (Table 1; Eq. 24)

    Examples
    --------
    >>> rr = radarx.retrieve.rain_rate_a(sweep, a="AH")  # doctest: +SKIP
    >>> rr = sweep.radarx.rain_rate_a(temperature=15.0, band="C")  # doctest: +SKIP
    """
    return _single(
        obj,
        3,
        {"a": a},
        band,
        {"a": coefficients} if coefficients is not None else None,
        polarization,
        None,
        None,
        negative,
        temperature,
        engine,
        n_threads,
    )


_fill(rain_rate_z, "``(a, b)`` of the relation instead of the default of the band")
_fill(
    rain_rate_z_zdr, "``(a, b, c)`` of the relation instead of the default of the band"
)
_fill(
    rain_rate_a,
    "``(a, b)`` of the relation for all temperatures instead of Table 1",
    z_cap=False,
)


# --------------------------------------------------------------------------
# class blend
# --------------------------------------------------------------------------


def _spec_relation(spec):
    """``(relation, factor)`` of a class relation given as a name, a pair or None."""
    if spec is None:
        return None, 1.0
    if isinstance(spec, str):
        name, factor = spec, 1.0
    else:
        try:
            name, factor = spec
        except (TypeError, ValueError):
            raise ValueError(
                "a class relation is a relation name, a (name, factor) pair or None"
            ) from None
    if name not in RELATIONS:
        raise ValueError(f"unknown relation {name!r}, use one of {RELATIONS}")
    factor = float(factor)
    if not factor >= 0.0:
        raise ValueError("the factor of a class relation must not be negative")
    return name, factor


def _class_tables(hid_da, class_relations):
    """
    Look-up tables of the blend from the classes of a ``HID`` array.

    Returns ``(lut_rel, lut_fac, relations)`` with one entry per class code
    (``-1`` for no relation) and the set of relations that are used.
    """
    meanings = hid_da.attrs.get("flag_meanings")
    values = hid_da.attrs.get("flag_values")
    if meanings is None or values is None:
        raise ValueError(
            "the class field needs flag_values and flag_meanings attributes, "
            "as the output of radarx.retrieve.hid has"
        )
    names = str(meanings).split()
    values = [int(v) for v in np.asarray(values).reshape(-1)]
    if len(names) != len(values):
        raise ValueError("flag_values and flag_meanings differ in length")
    mapping = dict(DEFAULT_CLASS_RELATIONS)
    for key in class_relations or {}:
        if key not in names:
            raise ValueError(
                f"class {key!r} is not one of the classes {names} of the "
                "hydrometeor classification"
            )
    mapping.update(class_relations or {})
    size = max(values + [0]) + 1
    lut_rel = np.full(size, -1, dtype=np.int32)
    lut_fac = np.ones(size, dtype=np.float64)
    used = set()
    for name, code in zip(names, values):
        if name not in mapping or code < 1:
            continue
        rel, factor = _spec_relation(mapping[name])
        if rel is None:
            continue
        lut_rel[code] = _CODE[rel]
        lut_fac[code] = factor
        used.add(rel)
    return lut_rel, lut_fac, used


def _hid_for(ds, spec, name):
    """The class array of the sweep ``name`` from ``spec`` (None: the ``HID`` of ``ds``)."""
    if spec is None:
        return ds["HID"] if "HID" in ds else None
    if isinstance(spec, xr.DataArray):
        return spec
    if isinstance(spec, str):
        if spec not in ds:
            raise KeyError(f"{spec!r} is not in the dataset")
        return ds[spec]
    if isinstance(spec, xr.DataTree):
        if name is None or name not in spec.children:
            return None
        spec = spec[name].to_dataset()
    if isinstance(spec, xr.Dataset):
        if "HID" not in spec:
            raise KeyError("the hydrometeor classification has no 'HID' variable")
        return spec["HID"]
    raise TypeError("hid must be a field name, a DataArray, a Dataset or a DataTree")


def _blend_flags():
    return {
        "RAIN_RATE_FLAG": {
            "long_name": "why the rain rate is missing",
            "flag_values": np.array([0, 1, 2, 3], dtype=np.int8),
            "flag_meanings": " ".join(FLAG_MEANINGS),
            "comment": (
                "0: estimated; 1: no valid relation for the hydrometeor class; "
                "2: an input of the relation is missing; 3: the gate is not "
                "classified"
            ),
        },
        "RAIN_RATE_RELATION": {
            "long_name": "relation that gave the rain rate",
            "flag_values": np.array([0, 1, 2, 3, 4], dtype=np.int8),
            "flag_meanings": " ".join(RELATION_MEANINGS),
            "comment": "0: none; 1: R(Z); 2: R(KDP); 3: R(Z,ZDR); 4: R(A)",
        },
    }


def _blend_dataset(
    ds,
    hid_da,
    relations,
    lut_rel,
    lut_fac,
    specs,
    band,
    setup,
    polarization,
    temperature,
    text,
    engine,
    n_threads,
):
    use_compiled = _use_compiled(engine)
    fields = {}
    for rel in relations:
        for key in _NEEDS[rel]:
            if key not in fields:
                fields[key] = _find(ds, specs.get(key), key)
    t = _temperature(ds, temperature) if "a" in relations else None
    keys = ["hid"] + list(fields)
    arrays = [hid_da] + [fields[k] for k in fields]
    scalar_t = None
    if isinstance(t, xr.DataArray) and not setup.fixed_a:
        keys.append("t")
        arrays.append(t)
    elif "a" in relations:
        scalar_t = (
            0.0 if setup.fixed_a else (t if not isinstance(t, xr.DataArray) else None)
        )

    def func(*vals):
        got = dict(zip(keys, vals))
        return _blend_array(
            got["hid"],
            got.get("dbz"),
            got.get("zdr"),
            got.get("kdp"),
            got.get("a"),
            got.get("t", scalar_t),
            lut_rel,
            lut_fac,
            setup,
            use_compiled,
            n_threads,
        )

    rate, flag, used = xr.apply_ufunc(
        func,
        *detach(*arrays),
        dask="parallelized",
        output_core_dims=[[], [], []],
        output_dtypes=[np.float32, np.int8, np.int8],
        keep_attrs=False,
    )
    rate.attrs = {
        "standard_name": "rainfall_rate",
        "long_name": "rain rate",
        "units": "mm h-1",
        "comment": text,
    }
    flags = _blend_flags()
    flag.attrs = flags["RAIN_RATE_FLAG"]
    used.attrs = flags["RAIN_RATE_RELATION"]
    return xr.Dataset(
        {"RAIN_RATE": rate, "RAIN_RATE_FLAG": flag, "RAIN_RATE_RELATION": used}
    )


def _blend_text(
    relations, lut_rel, lut_fac, hid_da, band, setup, polarization, temperature
):
    parts = [
        f"{rel}: {_describe(rel, band, setup, polarization, temperature)}"
        for rel in RELATIONS
        if rel in relations
    ]
    names = str(hid_da.attrs["flag_meanings"]).split()
    values = [int(v) for v in np.asarray(hid_da.attrs["flag_values"]).reshape(-1)]
    chosen = []
    for name, code in zip(names, values):
        if 0 < code < lut_rel.size and lut_rel[code] >= 0:
            fac = "" if lut_fac[code] == 1.0 else f" x {lut_fac[code]:g}"
            chosen.append(f"{name}: {RELATIONS[lut_rel[code]]}{fac}")
    return (
        "relation chosen from the hydrometeor class, after Giangrande and "
        f"Ryzhkov (2008); {'; '.join(chosen)}; other classes have no estimate. "
        + "; ".join(parts)
    )


@renamed_arguments(zdr_min="min_zdr")
@provenance("Rain rate with the relation chosen from the hydrometeor class")
def rain_rate(
    obj,
    hid=None,
    *,
    band="S",
    class_relations=None,
    coefficients=None,
    dbz=None,
    zdr=None,
    kdp=None,
    a=None,
    temperature=None,
    polarization="h",
    z_cap=Z_CAP,
    min_zdr=None,
    negative="keep",
    hid_kwargs=None,
    n_threads=None,
    engine="auto",
):
    """
    Rain rate with the relation chosen from the hydrometeor class of the gate.

    Each gate gets the rain rate of the relation that
    :mod:`radarx.retrieve.qpe` assigns to its class (``R(Z, ZDR)`` in rain and
    ``R(KDP)`` in a rain-hail mixture by default) and NaN where no relation
    is valid: hail, graupel, snow, ice crystals, the melting layer (wet snow)
    and gates without a class. ``RAIN_RATE_FLAG`` tells which. The classes are
    read from the attributes of the ``HID`` variable, so every method and band
    of :func:`radarx.retrieve.hid` works.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataTree
        A sweep or grid with the fields of the relations that are used and,
        unless ``hid`` is given, the classes in ``HID``; or a volume with
        ``sweep_*`` groups (sweeps without the fields are skipped). With the
        default relations the fields are the reflectivity, ``ZDR`` and ``KDP``
        (a field is only needed if a class that occurs uses its relation).
    hid : str, xarray.DataArray, xarray.Dataset or xarray.DataTree, optional
        The classification: the name of the field, the array, or the result of
        :func:`radarx.retrieve.hid` (for a volume, with the same sweep names).
        Default: ``HID`` of ``obj``; if there is none, the classification is
        computed with :func:`radarx.retrieve.hid` (``band`` and
        ``hid_kwargs``).
    band : {"S", "C", "X"}, optional
        Radar band, which selects the default coefficients (and the band of a
        classification that is computed here). Default ``"S"``.
    class_relations : dict, optional
        Changes to the default of each class: a mapping from the name of a
        class (as in ``flag_meanings``) to ``"z"``, ``"kdp"``, ``"z_zdr"``,
        ``"a"``, a pair ``(relation, factor)`` that scales the rain rate, or
        ``None`` for no estimate. For example ``{"wet_snow": ("z", 0.6)}`` adds
        the wet-snow estimate of Giangrande and Ryzhkov (2007) and
        ``{"heavy_rain": "kdp"}`` takes ``R(KDP)`` in heavy rain. A name that
        is not a class of the classification raises an error.
    coefficients : dict, optional
        Coefficients by relation, ``{"z": (a, b), "kdp": (a, b), "z_zdr": (a,
        b, c), "a": (a, b)}``, instead of the defaults of the band; required
        for a relation that has none at that band.
    dbz, zdr, kdp, a : str or xarray.DataArray, optional
        Names of the fields, or the fields. By default the names of
        :func:`rain_rate_z`, :func:`rain_rate_z_zdr`, :func:`rain_rate_kdp` and
        :func:`rain_rate_a`.
    temperature, polarization : optional
        For ``R(A)`` only, see :func:`rain_rate_a`.
    z_cap, min_zdr, negative : optional
        See :func:`rain_rate_z`, :func:`rain_rate_z_zdr` and
        :func:`rain_rate_kdp`.
    hid_kwargs : dict, optional
        Options of :func:`radarx.retrieve.hid` for a classification computed
        here, for example ``{"temperature": profile}``.
    n_threads, engine : optional
        See :func:`rain_rate_z`.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``RAIN_RATE`` (mm h-1, float32), ``RAIN_RATE_FLAG`` (0 estimated, 1 no relation for
        the class, 2 an input is missing, 3 not classified) and
        ``RAIN_RATE_RELATION`` (0 none, 1 R(Z), 2 R(KDP), 3 R(Z,ZDR), 4 R(A)),
        on the coordinates of the fields; for a DataTree one node per sweep.
        The input is not changed.

    Raises
    ------
    KeyError
        If a field that a used relation needs, or a sweep that has all of them,
        is not found.
    ValueError
        For an unknown option, a class name that does not exist, or a
        relation without a default at ``band`` and without ``coefficients``.

    Notes
    -----
    The default relations are those of Giangrande and Ryzhkov (2007) for S
    band; at C and X band pass ``coefficients`` for the relations of your
    radar, or choose with ``class_relations`` relations that have a default
    (``"kdp"`` and ``"z"`` at X band, ``"a"`` at C and X band). The factors
    fitted to gauges for wet snow, graupel, dry snow and crystals in that
    paper are for Oklahoma and are not applied unless you ask for them.

    References
    ----------
    Giangrande, S. E., and A. V. Ryzhkov, 2008: Estimation of rainfall based
    on the results of polarimetric echo classification. *J. Appl. Meteor.
    Climatol.*, **47** (9), 2445-2462,
    https://doi.org/10.1175/2008JAMC1753.1 (the class table is taken from the
    extended abstract of the same title, 33rd Conf. on Radar Meteorology,
    2007, paper P6B.7, Eq. 4; the journal article was not available).

    Ryzhkov, A. V., S. E. Giangrande, and T. J. Schuur, 2005: Rainfall
    estimation with a polarimetric prototype of WSR-88D. *J. Appl. Meteor.*,
    **44** (4), 502-515, https://doi.org/10.1175/JAM2213.1

    Ryzhkov, A., M. Diederich, P. Zhang, and C. Simmer, 2014: Potential
    utilization of specific attenuation for rainfall estimation, mitigation of
    partial beam blockage, and radar networking. *J. Atmos. Oceanic
    Technol.*, **31** (3), 599-619,
    https://doi.org/10.1175/JTECH-D-13-00038.1 (R(A) relations)

    Examples
    --------
    >>> volume = volume.radarx.assign(volume.radarx.hid(profile))  # doctest: +SKIP
    >>> rr = radarx.retrieve.rain_rate(volume)  # doctest: +SKIP
    >>> rr = sweep.radarx.rain_rate(class_relations={"wet_snow": ("z", 0.6)})  # doctest: +SKIP
    """
    _check_options(negative, polarization)
    _check_band(band)
    _use_compiled(engine)
    if isinstance(obj, xr.DataArray) or not isinstance(obj, (xr.Dataset, xr.DataTree)):
        raise TypeError("obj must be an xarray.Dataset or xarray.DataTree")
    specs = {"dbz": dbz, "zdr": zdr, "kdp": kdp, "a": a}
    hid_kwargs = dict(hid_kwargs or {})

    def build(hid_da):
        lut_rel, lut_fac, relations = _class_tables(hid_da, class_relations)
        coef, nodes = _coefficients(sorted(relations), band, coefficients, polarization)
        setup = _Setup(coef, nodes, z_cap, min_zdr, negative)
        text = _blend_text(
            relations, lut_rel, lut_fac, hid_da, band, setup, polarization, temperature
        )
        return lut_rel, lut_fac, relations, setup, text

    if isinstance(obj, xr.Dataset):
        hid_da = _hid_for(obj, hid, None)
        if hid_da is None:
            from .hid import hid as _hid

            hid_da = _hid(obj, band=band, **hid_kwargs)["HID"]
        lut_rel, lut_fac, relations, setup, text = build(hid_da)
        return _blend_dataset(
            obj,
            hid_da,
            relations,
            lut_rel,
            lut_fac,
            specs,
            band,
            setup,
            polarization,
            temperature,
            text,
            engine,
            n_threads,
        )

    computed = None
    # the classification is only computed if no sweep and no argument has one
    given = hid is not None or any(
        "HID" in obj[name].to_dataset() for name in _sweep_names(obj)
    )
    nodes_out = {"/": obj.root.to_dataset(inherit=False)}
    for sweep in _sweep_names(obj):
        ds = obj[sweep].to_dataset()
        hid_da = _hid_for(ds, hid, sweep)
        if hid_da is None and given:
            continue  # a classification exists, and it has no such sweep
        if hid_da is None:
            if computed is None:
                from .hid import hid as _hid

                computed = _hid(obj, band=band, **hid_kwargs)
            if sweep not in computed.children:
                continue
            hid_da = computed[sweep].to_dataset()["HID"]
        lut_rel, lut_fac, relations, setup, text = build(hid_da)
        try:
            for rel in relations:
                for key in _NEEDS[rel]:
                    _find(ds, specs.get(key), key)
        except KeyError:
            continue
        nodes_out[sweep] = _blend_dataset(
            ds,
            hid_da,
            relations,
            lut_rel,
            lut_fac,
            specs,
            band,
            setup,
            polarization,
            temperature,
            text,
            engine,
            n_threads,
        )
    if len(nodes_out) == 1:
        raise KeyError(
            "no sweep has the classification and the fields of the relations"
        )
    return xr.DataTree.from_dict(nodes_out)


# --------------------------------------------------------------------------
# accessors
# --------------------------------------------------------------------------


@accessor_method(
    "dataarray", "dataset", "datatree", name="rain_rate_z", wraps=rain_rate_z
)
def _rain_rate_z_accessor(self):
    """
    Rain rate from the reflectivity, R(Z).

    See :func:`radarx.retrieve.rain_rate_z` for the parameters.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        ``RAIN_RATE_Z`` (mm h-1).
    """


@accessor_method(
    "dataarray", "dataset", "datatree", name="rain_rate_kdp", wraps=rain_rate_kdp
)
def _rain_rate_kdp_accessor(self):
    """
    Rain rate from the specific differential phase, R(KDP).

    See :func:`radarx.retrieve.rain_rate_kdp` for the parameters.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        ``RAIN_RATE_KDP`` (mm h-1).
    """


@accessor_method("dataset", "datatree", name="rain_rate_z_zdr", wraps=rain_rate_z_zdr)
def _rain_rate_z_zdr_accessor(self):
    """
    Rain rate from the reflectivity and the differential reflectivity, R(Z, ZDR).

    See :func:`radarx.retrieve.rain_rate_z_zdr` for the parameters.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        ``RAIN_RATE_Z_ZDR`` (mm h-1).
    """


@accessor_method(
    "dataarray", "dataset", "datatree", name="rain_rate_a", wraps=rain_rate_a
)
def _rain_rate_a_accessor(self):
    """
    Rain rate from the specific attenuation, R(A).

    See :func:`radarx.retrieve.rain_rate_a` for the parameters.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        ``RAIN_RATE_A`` (mm h-1).
    """


@accessor_method("dataset", "datatree", name="rain_rate", wraps=rain_rate)
def _rain_rate_accessor(self):
    """
    Rain rate with the relation chosen from the hydrometeor class.

    See :func:`radarx.retrieve.rain_rate` for the parameters.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``RAIN_RATE`` (mm h-1), ``RAIN_RATE_FLAG`` and ``RAIN_RATE_RELATION``.
    """
