#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Attenuation of Ka and W Band Cloud Radar Signals
================================================

A cloud radar at 35 GHz (Ka band) or 94 GHz (W band) loses signal to the gases
of the air, to cloud liquid water and to rain, and the loss differs between the
two bands. The dual-frequency ratio of :mod:`radarx.retrieve.dual_frequency` and
the reflectivity itself are only meaningful once the attenuation is removed. This
module gives the specific attenuation of each constituent and the two-way
path-integrated attenuation along the beam, which is added to the measured
reflectivity in dB.

Gases
-----
:func:`gas_specific_attenuation` evaluates the line-by-line model of Rec. ITU-R
P.676-13, Annex 1 [1]_: the sum of the 44 oxygen lines and 35 water vapour lines
(the last is the pseudo-line of the wet continuum) of its Tables 1 and 2, with
the line strengths, widths and interference terms of its Eqs. (1) to (9), plus
the dry continuum (pressure-induced nitrogen absorption and the non-resonant
Debye spectrum of oxygen). The result is the one-way attenuation in dB km-1 of
oxygen and of water vapour for a pressure, temperature and vapour pressure. At
sea level in a standard atmosphere (1013.25 hPa, 15 degrees C, 7.5 g m-3 of
vapour) it is 0.10 dB km-1 at 35 GHz and 0.41 dB km-1 at 94 GHz, of which the
oxygen is 0.03 dB km-1 in both and the rest is water vapour (the wing of the
line at 183 GHz and the continuum). The model is valid to 1000 GHz.
:func:`gas_path_attenuation` interpolates a sounding to the gates, integrates
along the beam and gives the two-way attenuation, ``PIA_GAS`` in dB.

Cloud liquid water
------------------
:func:`cloud_liquid_attenuation` is the Rayleigh specific attenuation of
Rec. ITU-R P.840-9, Annex 1, Eqs. (1) to (10) [2]_: with the double-Debye
permittivity of liquid water [3]_ the one-way attenuation is
:math:`K_l \\rho_l`, with :math:`K_l = 0.819 f / [\\epsilon'' (1 + \\eta^2)]`
(dB km-1 per g m-3) and :math:`\\eta = (2 + \\epsilon') / \\epsilon''`. It is
valid for droplets of less than about 0.1 mm up to 200 GHz and applies to
supercooled water. :math:`K_l` is 1.0 dB km-1 per g m-3 at 35 GHz and 4.5 at
94 GHz for water at 0 degrees C, so a liquid water path of 0.3 kg m-2
(a thin stratocumulus) costs a vertically pointing W band radar
:math:`2 \\times 4.5 \\times 0.3 \\approx 2.7` dB two-way. Pass the liquid
water content of the cloud (from a radiometer, an adiabatic profile or a model) to
:func:`gas_path_attenuation` to integrate it.

Rain
----
Rain at these frequencies attenuates by several dB km-1 and the Rayleigh
approximation fails: the measured reflectivity of rain of a few mm drop size
saturates at W band. :func:`rain_attenuation_relation` fits the power law
:math:`k = a Z^b` (dB km-1, Z in mm6 m-3) to the T-matrix scattering of the
raindrops of gamma size distributions at the given frequency, with Z the
reflectivity the radar measures at that frequency, so that the relation is
the one the Hitschfeld and Bordan [4]_ solution needs. :func:`rain_path_attenuation`
applies that solution along the beam. Both are an estimate: the relation depends
on the drop size distribution and the temperature, and the correction of rain
at W band is unstable beyond a few dB of path attenuation (the cap
``max_pia`` limits it).

Conventions
-----------
The attenuation functions return the **two-way** attenuation in dB (``PIA_*``),
which is added to the reflectivity in dB to correct it, as the other attenuation
functions of radarx do, or the **one-way** specific attenuation in dB km-1
(``A_*``). The reflectivity factor of a cloud radar is the equivalent
reflectivity factor with :math:`|K_w|^2 = 0.93`, the value of the T-matrix tables of
:mod:`radarx.retrieve.dsd` at every frequency.

Implementation
--------------
A compiled, multithreaded C++ kernel sums the lines and integrates along the
beam (``radarx.retrieve._gas``), with equivalent NumPy implementations as
fallback and test oracle. The functions work on :class:`xarray.DataArray`,
:class:`xarray.Dataset` and :class:`xarray.DataTree` objects and on time-height
sections of a vertically pointing radar.

References
----------
.. [1] Rec. ITU-R P.676-13, 2022: Attenuation by atmospheric gases and related
       effects. International Telecommunication Union, Geneva,
       https://www.itu.int/rec/R-REC-P.676 (recommendation, no DOI).
.. [2] Rec. ITU-R P.840-9, 2023: Attenuation due to clouds and fog.
       International Telecommunication Union, Geneva,
       https://www.itu.int/rec/R-REC-P.840 (recommendation, no DOI).
.. [3] Liebe, H. J., G. A. Hufford, and T. Manabe, 1991: A model for the complex
       permittivity of water at frequencies below 1 THz. *Int. J. Infrared
       Millimeter Waves*, **12**, 659-675, https://doi.org/10.1007/BF01008897
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
    "gas_specific_attenuation",
    "cloud_liquid_attenuation",
    "gas_path_attenuation",
    "rain_attenuation_relation",
    "rain_path_attenuation",
]

__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

import functools

import numpy as np
import xarray as xr

from .._dtypes import as_field
from .._provenance import provenance
from .._registry import accessor_method
from ..fundamentals.geometry import beam_center_height
from ..fundamentals.reflectivity import hitschfeld_bordan_correction
from ._fieldtools import DBZ_NAMES, MissingField, choose_engine, detach, map_sweeps
from ._gas_lines import OXYGEN, WATER_VAPOR

try:
    from . import _gas

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _gas = None
    HAS_COMPILED_KERNEL = False

_RANGE = "range"
_OXYGEN = np.asarray(OXYGEN, dtype=np.float64)
_WATER_VAPOR = np.asarray(WATER_VAPOR, dtype=np.float64)
_HUMIDITY = ("vapor_pressure", "specific_humidity", "dewpoint", "relative_humidity")
_HEIGHT_NAMES = ("beam_height", "height", "z")
_ELEVATION_NAMES = ("elevation", "sweep_fixed_angle")
_CLOUD_DBZ_NAMES = ("Z", "Zh", "Ze")  # Cloudnet files
_CHUNK = 65536  # points per block of the NumPy line sum


# --------------------------------------------------------------------------
# NumPy references of the kernel
# --------------------------------------------------------------------------


def _line_shape(f, fi, df, delta):
    """Line-shape factor of Rec. ITU-R P.676-13, Eq. (5)."""
    a, b = fi - f, fi + f
    return (f / fi) * (
        (df - delta * a) / (a * a + df * df) + (df - delta * b) / (b * b + df * df)
    )


def _specific_numpy(f, p, t, e):
    """Oxygen and water vapour attenuation [dB/km] of flat arrays; the kernel's twin."""
    f, p, t, e = (np.asarray(x, dtype=np.float64) for x in (f, p, t, e))
    out_o = np.full(f.shape, np.nan)
    out_w = np.full(f.shape, np.nan)
    ok = (f > 0) & (t > 0) & (p >= 0) & (e >= 0) & np.isfinite(f + t + p + e)
    index = np.flatnonzero(ok)
    for start in range(0, index.size, _CHUNK):
        sel = index[start : start + _CHUNK]
        ff, pp, tt, ee = f[sel, None], p[sel, None], t[sel, None], e[sel, None]
        th = 300.0 / tt
        fi, a1, a2, a3, a4, a5, a6 = (_OXYGEN[None, :, k] for k in range(7))
        s = a1 * 1e-7 * pp * th**3 * np.exp(a2 * (1.0 - th))
        df = a3 * 1e-4 * (pp * th ** (0.8 - a4) + 1.1 * ee * th)
        df = np.sqrt(df * df + 2.25e-6)
        delta = (a5 + a6 * th) * 1e-4 * (pp + ee) * th**0.8
        n_ox = (s * _line_shape(ff, fi, df, delta)).sum(axis=1)
        th1, ff1, pp1, ee1 = th[:, 0], ff[:, 0], pp[:, 0], ee[:, 0]
        d = 5.6e-4 * (pp1 + ee1) * th1**0.8
        n_ox = n_ox + ff1 * pp1 * th1**2 * (
            6.14e-5 / (d * (1.0 + (ff1 / d) ** 2))
            + 1.4e-12 * pp1 * th1**1.5 / (1.0 + 1.9e-5 * ff1**1.5)
        )
        fi, b1, b2, b3, b4, b5, b6 = (_WATER_VAPOR[None, :, k] for k in range(7))
        s = b1 * 1e-1 * ee * th**3.5 * np.exp(b2 * (1.0 - th))
        df = b3 * 1e-4 * (pp * th**b4 + b5 * ee * th**b6)
        df = 0.535 * df + np.sqrt(0.217 * df * df + 2.1316e-12 * fi * fi / th)
        n_wv = (s * _line_shape(ff, fi, df, 0.0)).sum(axis=1)
        out_o[sel] = 0.1820 * ff1 * n_ox
        out_w[sel] = 0.1820 * ff1 * n_wv
    return out_o, out_w


def _liquid_coefficient_numpy(f, t):
    """Rec. ITU-R P.840-9 coefficient K_l [dB/km per g/m3] of flat arrays."""
    f = np.asarray(f, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        th = 300.0 / t
        e0 = 77.66 + 103.3 * (th - 1.0)
        e1 = 0.0671 * e0
        e2 = 3.52
        fp = 20.20 - 146.0 * (th - 1.0) + 316.0 * (th - 1.0) ** 2
        fs = 39.8 * fp
        epp = f * (e0 - e1) / (fp * (1.0 + (f / fp) ** 2)) + f * (e1 - e2) / (
            fs * (1.0 + (f / fs) ** 2)
        )
        ep = (e0 - e1) / (1.0 + (f / fp) ** 2) + (e1 - e2) / (1.0 + (f / fs) ** 2) + e2
        eta = (2.0 + ep) / epp
        k = 0.819 * f / (epp * (1.0 + eta**2))
    return np.where((f > 0) & (t > 0) & np.isfinite(f + t), k, np.nan)


def _integrate_numpy(range_m, gamma, gamma0):
    """Two-way attenuation [dB] along the last axis; the kernel's twin."""
    gamma = np.asarray(gamma, dtype=np.float64)
    start = np.where(np.isfinite(gamma0), gamma0, 0.0)[:, None]
    stack = np.concatenate([start, gamma], axis=1)
    finite = np.isfinite(stack)
    finite[:, 0] = True
    last = np.maximum.accumulate(np.where(finite, np.arange(stack.shape[1]), 0), axis=1)
    filled = np.take_along_axis(stack, last, axis=1)
    dr = np.diff(np.concatenate([[0.0], np.asarray(range_m, dtype=np.float64)])) * 1e-3
    one_way = np.cumsum(0.5 * (filled[:, :-1] + filled[:, 1:]) * dr, axis=1)
    return 2.0 * one_way


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------


def _specific(f, p, t, e, use_compiled, n_threads):
    f, p, t, e = (np.ascontiguousarray(x, dtype=np.float64) for x in (f, p, t, e))
    if use_compiled:
        return _gas.specific(f, p, t, e, _OXYGEN, _WATER_VAPOR, int(n_threads or 0))
    return _specific_numpy(f, p, t, e)


def _cloud_liquid(f, t, lwc, use_compiled, n_threads):
    f, t, lwc = (np.ascontiguousarray(x, dtype=np.float64) for x in (f, t, lwc))
    if use_compiled:
        return _gas.cloud_liquid(f, t, lwc, int(n_threads or 0))
    return _liquid_coefficient_numpy(f, t) * lwc


def _integrate(range_m, gamma, gamma0, use_compiled, n_threads):
    range_m = np.ascontiguousarray(range_m, dtype=np.float64)
    gamma = np.ascontiguousarray(gamma, dtype=np.float64)
    gamma0 = np.ascontiguousarray(gamma0, dtype=np.float64)
    if use_compiled:
        return _gas.integrate(range_m, gamma, gamma0, int(n_threads or 0))
    return _integrate_numpy(range_m, gamma, gamma0)


# --------------------------------------------------------------------------
# specific attenuation on any broadcast of the inputs
# --------------------------------------------------------------------------


def _as_data_array(value, name):
    if isinstance(value, xr.DataArray):
        return value
    return xr.DataArray(np.asarray(value, dtype=np.float64), name=name)


def _broadcast(**inputs):
    """The inputs as DataArrays broadcast against each other, with flat float64 values."""
    arrays = xr.broadcast(*[_as_data_array(v, k) for k, v in inputs.items()])
    template = arrays[0]
    flat = {
        k: np.ascontiguousarray(
            a.transpose(*template.dims).values, dtype=np.float64
        ).reshape(-1)
        for k, a in zip(inputs, arrays)
    }
    return template, flat


def _rewrap(template, values, name, attrs):
    out = xr.DataArray(
        as_field(np.asarray(values).reshape(template.shape)),
        dims=template.dims,
        coords=dict(template.coords),
        name=name,
        attrs=attrs,
    )
    return out


@provenance(
    "Specific attenuation of oxygen and water vapour by the line-by-line model of "
    "Rec. ITU-R P.676-13",
    extra_refs=("itu-r-p676-13",),
)
def gas_specific_attenuation(
    frequency,
    pressure,
    temperature,
    vapor_pressure=None,
    *,
    vapor_density=None,
    n_threads=None,
    engine="auto",
):
    """
    Specific attenuation of oxygen and water vapour at a radar frequency.

    The line-by-line model of Rec. ITU-R P.676-13, Annex 1 (see
    :mod:`radarx.retrieve.cloud_attenuation`): the one-way attenuation of the dry
    air (oxygen, pressure-induced nitrogen and the non-resonant Debye spectrum)
    and of water vapour at the frequency, for the air pressure, temperature
    and water vapour of every point. The inputs are broadcast against each other,
    so a frequency axis gives a spectrum and a sounding gives a profile.

    Parameters
    ----------
    frequency : float or xarray.DataArray
        Frequency in GHz (1 to 1000), for example 35.5 or 94.
    pressure : float or xarray.DataArray
        Total air pressure in hPa (dry air plus water vapour).
    temperature : float or xarray.DataArray
        Air temperature in K.
    vapor_pressure : float or xarray.DataArray, optional
        Partial pressure of water vapour in hPa.
    vapor_density : float or xarray.DataArray, optional
        Water vapour density in g m-3, instead of ``vapor_pressure``; they are
        related by ``e = rho T / 216.7`` (Rec. ITU-R P.676-13, Eq. 4). Give one
        of the two.
    n_threads : int, optional
        Threads of the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation. ``"auto"`` (default) prefers the compiled kernel.

    Returns
    -------
    xarray.Dataset
        ``A_OXYGEN`` (dry air), ``A_WATER_VAPOR`` and ``A_GAS`` (their sum), the
        one-way specific attenuation in dB km-1, on the broadcast dimensions of
        the inputs. Gates with a non-positive or missing input are NaN.

    Raises
    ------
    ValueError
        If neither or both of ``vapor_pressure`` and ``vapor_density`` are given,
        or for an unknown ``engine``.
    ImportError
        If ``engine="compiled"`` and the kernel is not available.

    References
    ----------
    Rec. ITU-R P.676-13, 2022: Attenuation by atmospheric gases and related
    effects. International Telecommunication Union, Geneva,
    https://www.itu.int/rec/R-REC-P.676 (recommendation, no DOI).

    Examples
    --------
    >>> gas = gas_specific_attenuation(94.0, 1013.25, 288.15, vapor_density=7.5)  # doctest: +SKIP
    >>> round(float(gas.A_GAS), 2)  # doctest: +SKIP
    0.41
    """
    use_compiled = choose_engine(engine, HAS_COMPILED_KERNEL, "gas attenuation")
    if (vapor_pressure is None) == (vapor_density is None):
        raise ValueError("give exactly one of vapor_pressure and vapor_density")
    if vapor_pressure is None:
        vapor_pressure = _as_data_array(vapor_density, "vapor_density") * (
            _as_data_array(temperature, "temperature") / 216.7
        )
    template, flat = _broadcast(
        frequency=frequency,
        pressure=pressure,
        temperature=temperature,
        vapor_pressure=vapor_pressure,
    )
    e = flat["vapor_pressure"]
    dry = flat["pressure"] - e  # P.676 uses the dry-air pressure p = p_tot - e
    gamma_o, gamma_w = _specific(
        flat["frequency"], dry, flat["temperature"], e, use_compiled, n_threads
    )
    attrs = {
        "A_OXYGEN": "Specific attenuation of dry air (oxygen and nitrogen)",
        "A_WATER_VAPOR": "Specific attenuation of water vapour",
        "A_GAS": "Specific attenuation of oxygen and water vapour",
    }
    values = {"A_OXYGEN": gamma_o, "A_WATER_VAPOR": gamma_w, "A_GAS": gamma_o + gamma_w}
    return xr.Dataset(
        {
            k: _rewrap(
                template,
                values[k],
                k,
                {"long_name": text + " (one way)", "units": "dB km-1"},
            )
            for k, text in attrs.items()
        }
    )


@provenance(
    "Specific attenuation of cloud liquid water by the Rayleigh double-Debye model of "
    "Rec. ITU-R P.840-9",
    extra_refs=("itu-r-p840-9",),
)
def cloud_liquid_attenuation(
    frequency, temperature, lwc=1.0, *, n_threads=None, engine="auto"
):
    """
    Specific attenuation of cloud liquid water at a radar frequency.

    ``K_l * lwc`` with the coefficient ``K_l`` of Rec. ITU-R P.840-9, Annex 1,
    Eqs. (2) to (10): the Rayleigh absorption of liquid water with the
    double-Debye permittivity of Liebe et al. (1991), one way, for droplets of
    less than about 0.1 mm at up to 200 GHz. With the default ``lwc`` of
    1 g m-3 the result is the coefficient ``K_l`` in dB km-1 per g m-3.

    Parameters
    ----------
    frequency : float or xarray.DataArray
        Frequency in GHz.
    temperature : float or xarray.DataArray
        Temperature of the liquid water in K (supercooled water is valid).
    lwc : float or xarray.DataArray, optional
        Liquid water content in g m-3. Default 1, which gives ``K_l``.
    n_threads : int, optional
        Threads of the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation. ``"auto"`` (default) prefers the compiled kernel.

    Returns
    -------
    xarray.DataArray
        ``A_CLOUD_LIQUID`` (dB km-1, one way) on the broadcast dimensions of the
        inputs; NaN where an input is missing.

    Raises
    ------
    ValueError
        For an unknown ``engine``.
    ImportError
        If ``engine="compiled"`` and the kernel is not available.

    References
    ----------
    Rec. ITU-R P.840-9, 2023: Attenuation due to clouds and fog. International
    Telecommunication Union, Geneva, https://www.itu.int/rec/R-REC-P.840
    (recommendation, no DOI).

    Liebe, H. J., G. A. Hufford, and T. Manabe, 1991: A model for the complex
    permittivity of water at frequencies below 1 THz. *Int. J. Infrared
    Millimeter Waves*, **12**, 659-675, https://doi.org/10.1007/BF01008897

    Examples
    --------
    >>> k94 = cloud_liquid_attenuation(94.0, 273.15)  # doctest: +SKIP
    """
    use_compiled = choose_engine(engine, HAS_COMPILED_KERNEL, "gas attenuation")
    template, flat = _broadcast(frequency=frequency, temperature=temperature, lwc=lwc)
    values = _cloud_liquid(
        flat["frequency"], flat["temperature"], flat["lwc"], use_compiled, n_threads
    )
    return _rewrap(
        template,
        values,
        "A_CLOUD_LIQUID",
        {
            "long_name": "Specific attenuation of cloud liquid water (one way)",
            "units": "dB km-1",
        },
    )


# --------------------------------------------------------------------------
# geometry and profile of a sweep or a time-height section
# --------------------------------------------------------------------------


def _frequency_ghz(obj, frequency):
    """Radar frequency in GHz: the argument or the ``radar_frequency`` of the data."""
    if frequency is not None:
        value = float(frequency)
    else:
        sources = [obj]
        if isinstance(obj, xr.DataTree):
            sources = [n.to_dataset(inherit=False) for n in obj.subtree]
        value = None
        for src in sources:
            value = _scalar(src, ("radar_frequency", "frequency"))
            if value is not None:
                break
        if value is None:
            raise ValueError(
                "the frequency is not in the data (radar_frequency): pass frequency in GHz"
            )
    if value > 1e6:  # xradar writes the frequency in Hz
        value = value / 1e9
    if not 1.0 <= value <= 1000.0:
        raise ValueError(f"the frequency must be 1 to 1000 GHz, not {value:g}")
    return value


def _vapor_pressure_pa(profile, engine):
    from .beam_path import _vapor_pressure

    pressure, temperature = profile["pressure"], profile["temperature"]
    return _vapor_pressure(profile, pressure, temperature, engine)


def _profile_gamma(profile, frequency, use_compiled, n_threads, engine):
    """Specific attenuation [dB/km] and temperature [K] of a profile at its levels."""
    if not isinstance(profile, xr.Dataset):
        raise TypeError("profile must be an xarray.Dataset on a 'height' coordinate")
    if "height" not in profile.coords:
        raise ValueError("the profile needs a 'height' coordinate (m)")
    for name in ("pressure", "temperature"):
        if name not in profile:
            raise ValueError(
                f"the profile needs {name!r} (Pa, K) and a humidity variable "
                f"{_HUMIDITY}"
            )
    height = np.asarray(profile["height"].values, dtype=np.float64)
    if height.ndim != 1 or height.size < 2 or not np.all(np.diff(height) > 0):
        raise ValueError("the profile height must be one-dimensional and increasing")
    p_total = np.asarray(profile["pressure"].values, dtype=np.float64) / 100.0
    t = np.asarray(profile["temperature"].values, dtype=np.float64)
    e = np.asarray(_vapor_pressure_pa(profile, engine).values, dtype=np.float64) / 100.0
    gamma_o, gamma_w = _specific(
        np.full(height.shape, frequency), p_total - e, t, e, use_compiled, n_threads
    )
    return height, gamma_o, gamma_w, t


def _reference_field(obj, dbz):
    """The field that gives the gates: ``dbz`` or the first field with a range."""
    if isinstance(obj, xr.DataArray):
        return obj
    if isinstance(dbz, xr.DataArray):
        return dbz
    if dbz is not None:
        if dbz not in obj:
            raise MissingField(f"{dbz!r} is not in the dataset")
        return obj[dbz]
    for name in DBZ_NAMES + _CLOUD_DBZ_NAMES:
        if name in obj and _RANGE in obj[name].dims:
            return obj[name]
    for name in obj.data_vars:
        if _RANGE in obj[name].dims:
            return obj[name]
    raise MissingField("no field with a 'range' dimension in the sweep")


def _reflectivity(obj, dbz):
    """The reflectivity of a sweep: ``obj``, ``dbz`` or the first usual name."""
    if isinstance(dbz, xr.DataArray):
        return dbz
    if isinstance(obj, xr.DataArray):
        return obj
    if dbz is not None:
        if dbz not in obj:
            raise MissingField(f"{dbz!r} is not in the dataset")
        return obj[dbz]
    for name in DBZ_NAMES + _CLOUD_DBZ_NAMES:
        if name in obj:
            return obj[name]
    raise MissingField("no reflectivity found: pass its name as dbz")


def _get(source, name):
    """A coordinate or variable of a DataArray or Dataset, or None."""
    if source is None:
        return None
    if isinstance(source, xr.DataArray):
        return source.coords[name] if name in source.coords else None
    return source[name] if name in source.variables else None


def _scalar(source, names):
    for name in names:
        found = _get(source, name)
        if found is not None:
            arr = np.asarray(found.values, dtype=np.float64).ravel()
            arr = arr[np.isfinite(arr)]
            if arr.size:
                return float(arr[0])
    return None


def _gate_height(ref, obj, height, radar_height):
    """Height of the gates (m above sea level) as a DataArray with a range dimension."""
    ds = obj if isinstance(obj, xr.Dataset) else None
    if height is not None:
        out = ds[height] if isinstance(height, str) and ds is not None else height
        if not isinstance(out, xr.DataArray) or _RANGE not in out.dims:
            raise ValueError("height must be a DataArray with a 'range' dimension")
        return out
    for src in (ref, ds):
        for name in _HEIGHT_NAMES:
            found = _get(src, name)
            if found is not None and _RANGE in found.dims:
                return found
    if _RANGE not in ref.coords:
        raise ValueError("the sweep needs a 'range' coordinate (m)")
    elevation = None
    for src in (ref, ds):
        if elevation is not None:
            break
        for name in _ELEVATION_NAMES:
            elevation = _get(src, name)
            if elevation is not None:
                break
        if elevation is None and _get(src, "zenith_angle") is not None:
            elevation = 90.0 - _get(src, "zenith_angle")
    if elevation is None:
        raise ValueError(
            "no gate height: give height, or a sweep with an 'elevation' "
            "(or 'zenith_angle') and the radar altitude"
        )
    altitude = radar_height
    if altitude is None:
        for src in (ds, ref):
            altitude = _scalar(src, ("altitude",))
            if altitude is not None:
                break
    if altitude is None:
        raise ValueError(
            "the radar altitude is not in the data; pass radar_height (m above sea level)"
        )
    return beam_center_height(
        ref[_RANGE].astype(np.float64),
        elevation.astype(np.float64).reset_coords(drop=True),
        radar_height=float(altitude),
    )


def _lwc_field(obj, lwc):
    if lwc is None:
        return None
    if isinstance(lwc, xr.DataArray):
        return lwc
    if isinstance(obj, xr.Dataset) and isinstance(lwc, str):
        if lwc not in obj:
            raise MissingField(f"{lwc!r} is not in the dataset")
        return obj[lwc]
    raise TypeError("lwc must be a DataArray or the name of a field of the sweep")


def _path_block(
    height,
    liquid,
    *,
    range_m,
    profile_height,
    gamma_o,
    gamma_w,
    profile_t,
    frequency,
    use_compiled,
    n_threads,
    with_liquid,
):
    """PIA of the gates of a block with the range on the last axis."""
    shape = height.shape
    h2 = np.ascontiguousarray(height, dtype=np.float64).reshape(-1, shape[-1])
    results = []
    for gamma in (gamma_o, gamma_w):
        g = np.interp(h2, profile_height, gamma)
        results.append(
            _integrate(range_m, g, g[:, 0], use_compiled, n_threads).reshape(shape)
        )
    if with_liquid:
        t = np.interp(h2, profile_height, profile_t)
        lwc = np.where(np.isfinite(liquid), liquid, 0.0).reshape(-1)
        g = _cloud_liquid(
            np.full(t.size, frequency), t.reshape(-1), lwc, use_compiled, n_threads
        ).reshape(h2.shape)
        results.append(
            _integrate(range_m, g, g[:, 0], use_compiled, n_threads).reshape(shape)
        )
    return tuple(results)


def _path_sweep(
    obj,
    profile,
    frequency,
    lwc,
    dbz,
    height,
    radar_height,
    n_threads,
    engine,
    use_compiled,
):
    ref = _reference_field(obj, dbz)
    if _RANGE not in ref.dims:
        raise ValueError(f"the field needs a 'range' dimension, not {ref.dims}")
    freq = _frequency_ghz(obj, frequency)
    prof_h, gamma_o, gamma_w, prof_t = _profile_gamma(
        profile, freq, use_compiled, n_threads, engine
    )
    h = _gate_height(ref, obj, height, radar_height)
    h = h.broadcast_like(ref).transpose(*ref.dims)
    liquid = _lwc_field(obj, lwc)
    with_liquid = liquid is not None
    if with_liquid:
        liquid = liquid.broadcast_like(ref).transpose(*ref.dims)
    else:
        liquid = xr.zeros_like(h)
    range_m = np.asarray(ref[_RANGE].values, dtype=np.float64)
    if range_m.size < 2 or not np.all(np.diff(range_m) > 0):
        raise ValueError("range must have at least two gates and increase")
    n_out = 3 if with_liquid else 2
    outputs = xr.apply_ufunc(
        _path_block,
        *detach(h.reset_coords(drop=True), liquid.reset_coords(drop=True)),
        input_core_dims=[[_RANGE], [_RANGE]],
        output_core_dims=[[_RANGE]] * n_out,
        kwargs=dict(
            range_m=range_m,
            profile_height=prof_h,
            gamma_o=gamma_o,
            gamma_w=gamma_w,
            profile_t=prof_t,
            frequency=freq,
            use_compiled=use_compiled,
            n_threads=n_threads,
            with_liquid=with_liquid,
        ),
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[np.float64] * n_out,
    )
    names = ["PIA_OXYGEN", "PIA_WATER_VAPOR", "PIA_CLOUD_LIQUID"][:n_out]
    texts = [
        "Two-way path-integrated attenuation by oxygen",
        "Two-way path-integrated attenuation by water vapour",
        "Two-way path-integrated attenuation by cloud liquid water",
    ]
    data = {}
    for name, text, arr in zip(names, texts, outputs):
        arr = as_field(arr.transpose(*ref.dims))
        arr = arr.assign_coords({k: v for k, v in ref.coords.items() if k in arr.dims})
        arr.attrs = {"long_name": text, "units": "dB"}
        data[name] = arr
    gas = data["PIA_OXYGEN"] + data["PIA_WATER_VAPOR"]
    gas.attrs = {
        "long_name": "Two-way path-integrated attenuation by oxygen and water vapour",
        "units": "dB",
    }
    ordered = {"PIA_GAS": gas, **data}
    if "PIA_CLOUD_LIQUID" in ordered:
        ordered["PIA_CLOUD_LIQUID"] = ordered.pop("PIA_CLOUD_LIQUID")
    out = xr.Dataset(ordered)
    out.attrs = {"frequency_GHz": freq}
    return out


@provenance(
    "Two-way gas and cloud liquid attenuation along the beam: Rec. ITU-R P.676-13 "
    "line-by-line gases and the Rayleigh liquid water of Rec. ITU-R P.840-9",
    extra_refs=("itu-r-p676-13", "itu-r-p840-9"),
)
def gas_path_attenuation(
    obj,
    profile,
    frequency=None,
    *,
    lwc=None,
    dbz=None,
    height=None,
    radar_height=None,
    n_threads=None,
    engine="auto",
):
    """
    Two-way gas (and cloud liquid) attenuation along the beam of a cloud radar.

    The specific attenuation of oxygen and water vapour is computed with
    :func:`gas_specific_attenuation` at every level of the profile, interpolated
    linearly in height to the gates and integrated along the range from the radar
    with the trapezoid rule; the result is the two-way attenuation. The
    attenuation between the radar and the first gate uses the value of the first
    gate. With ``lwc`` the cloud liquid water of every gate is integrated in the
    same way with :func:`cloud_liquid_attenuation`, at the temperature of the
    profile.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        A sweep or a time-height section with a ``range`` dimension (m), or a
        volume with ``sweep_*`` groups (sweeps without a field with a range are
        skipped). The gates are placed at the height given by ``height``, else
        by a ``beam_height``, ``height`` or ``z`` variable with a ``range``
        dimension (the height above sea level of a vertically pointing radar), else by
        the 4/3 Earth model from the ``elevation`` (or ``zenith_angle``) and the
        radar altitude.
    profile : xarray.Dataset
        Sounding or model profile on a ``height`` coordinate (m above sea
        level, increasing) with ``pressure`` (Pa), ``temperature`` (K) and one
        of ``vapor_pressure`` (Pa), ``specific_humidity``, ``dewpoint`` or
        ``relative_humidity``, as the readers of :mod:`radarx.io.sounding`
        return. Gates above the profile get its top value.
    frequency : float, optional
        Frequency in GHz. Default: the ``radar_frequency`` of the data (GHz, or
        Hz as xradar writes it).
    lwc : str or xarray.DataArray, optional
        Liquid water content of the gates in g m-3 (a field of ``obj`` or a
        DataArray); NaN counts as 0. Adds ``PIA_CLOUD_LIQUID``.
    dbz : str or xarray.DataArray, optional
        The field whose gates define the output grid (for a ``Dataset``). Default:
        the first of ``DBZH``, ``DBZ``, ``reflectivity``, ``Z``, ``Zh``, else the
        first field with a range.
    height : str or xarray.DataArray, optional
        Height of the gates in m above sea level (a DataArray with a ``range``
        dimension, for example the ``beam_height`` of
        :func:`radarx.retrieve.beam_path`).
    radar_height : float, optional
        Radar altitude in m above sea level for the 4/3 Earth model. Default: the
        ``altitude`` of the data.
    n_threads : int, optional
        Threads of the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation. ``"auto"`` (default) prefers the compiled kernel.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        On the gates of the field: ``PIA_OXYGEN``, ``PIA_WATER_VAPOR``,
        ``PIA_GAS`` (their sum) and, with ``lwc``, ``PIA_CLOUD_LIQUID``, the
        two-way attenuation in dB at the gate (add it to the reflectivity in dBZ).
        A volume gives a DataTree with the root of the input and one node per
        sweep. The input is not changed.

    Raises
    ------
    ValueError
        For a profile without pressure, temperature or humidity, gates without
        a height, a frequency outside 1 to 1000 GHz, an unknown ``engine`` or a
        range that does not increase.
    TypeError
        If ``profile`` is not a Dataset or ``lwc`` of the wrong type.
    KeyError
        If no sweep has a field with a range.
    ImportError
        If ``engine="compiled"`` and the kernel is not available.

    Notes
    -----
    The gas attenuation of a vertically pointing radar below 3 km is about
    0.3 dB at 35 GHz and 1.2 dB at 94 GHz, two-way; for a scanning radar at low
    elevation it grows with the path through the humid boundary layer. The
    sounding is the dominant uncertainty: take it near in time and place, or
    from a model.

    References
    ----------
    Rec. ITU-R P.676-13, 2022: Attenuation by atmospheric gases and related
    effects. International Telecommunication Union, Geneva,
    https://www.itu.int/rec/R-REC-P.676 (recommendation, no DOI).

    Rec. ITU-R P.840-9, 2023: Attenuation due to clouds and fog. International
    Telecommunication Union, Geneva, https://www.itu.int/rec/R-REC-P.840
    (recommendation, no DOI).

    Examples
    --------
    >>> pia = gas_path_attenuation(sweep, sounding, frequency=94.0)  # doctest: +SKIP
    >>> dbz_corrected = sweep.DBZ + pia.PIA_GAS  # doctest: +SKIP
    >>> pia = sweep.radarx.gas_path_attenuation(sounding, lwc="LWC")  # doctest: +SKIP
    """
    use_compiled = choose_engine(engine, HAS_COMPILED_KERNEL, "gas attenuation")

    def one(sweep, name=None):
        return _path_sweep(
            sweep,
            profile,
            frequency,
            lwc,
            dbz,
            height,
            radar_height,
            n_threads,
            engine,
            use_compiled,
        )

    if isinstance(obj, xr.DataTree):
        return map_sweeps(obj, one, "a gas path attenuation")
    if not isinstance(obj, (xr.DataArray, xr.Dataset)):
        raise TypeError("obj must be an xarray DataArray, Dataset or DataTree")
    if isinstance(obj, xr.DataArray) and obj.name is None:
        obj = obj.rename("field")
    return one(obj)


# --------------------------------------------------------------------------
# rain
# --------------------------------------------------------------------------

_KW2 = 0.93  # |K_w|^2 of the equivalent reflectivity factor (radarx.retrieve.dsd)


def _gamma_dsd_ensemble(diameter, n_members=1500, seed=1):
    """
    Normalized-gamma drop size distributions N(D), shape (members, diameters).

    Median volume diameter D_m of 0.5-3.5 mm, log10 N_w of 2.5-4.5 (N_w in
    m-3 mm-1) and shape parameter mu of 0-6, uniformly random (radarx choices that
    cover the observed range of rain; see :func:`rain_attenuation_relation`).
    """
    rnd = np.random.default_rng(seed)
    dm = rnd.uniform(0.5, 3.5, n_members)
    nw = 10.0 ** rnd.uniform(2.5, 4.5, n_members)
    mu = rnd.uniform(0.0, 6.0, n_members)
    from scipy.special import gammaln

    d = diameter[None, :]
    dmc, nwc, muc = dm[:, None], nw[:, None], mu[:, None]
    f_mu = 6.0 / 4.0**4 * (muc + 4.0) ** (muc + 4.0) / np.exp(gammaln(muc + 4.0))
    x = d / dmc
    return nwc * f_mu * x**muc * np.exp(-(4.0 + muc) * x)


@functools.lru_cache(maxsize=16)
def _zk_fit(frequency, temperature):
    from ..fundamentals.tmatrix import raindrop_scattering

    diameter = np.arange(0.05, 8.001, 0.05)
    wavelength = 299.792458 / frequency
    drops = raindrop_scattering(diameter, wavelength, temperature, ddelt=1e-4)
    nd = _gamma_dsd_ensemble(diameter)
    dd = 0.05
    z = wavelength**4 / (np.pi**5 * _KW2) * (nd * drops.sigma_h.values).sum(1) * dd
    k = (nd * drops.ah.values).sum(1) * dd
    ok = (z > 10 ** (-0.5)) & (k > 0)  # Z above -5 dBZ
    slope, intercept = np.polyfit(np.log10(z[ok]), np.log10(k[ok]), 1)
    resid = np.log10(k[ok]) - (intercept + slope * np.log10(z[ok]))
    return float(10.0**intercept), float(slope), float(resid.std()), int(ok.sum())


@provenance(
    "Power-law relation between the specific attenuation and the reflectivity of rain "
    "at a radar frequency, fitted to T-matrix scattering by gamma drop size distributions",
)
def rain_attenuation_relation(frequency, temperature=10.0):
    """
    Coefficients ``a`` and ``b`` of ``k = a Z**b`` for rain at a radar frequency.

    The specific attenuation :math:`k` (dB km-1, one way) of rain follows a power
    law of the reflectivity :math:`Z` (mm6 m-3) that the radar measures at the
    same frequency (``|K_w|**2 = 0.93``). The coefficients are the least-squares
    fit of :math:`\\log k` on :math:`\\log Z` over 1500 gamma drop size
    distributions (normalized gamma, :math:`D_m` 0.5-3.5 mm, :math:`\\log_{10} N_w`
    2.5-4.5, :math:`\\mu` 0-6, a radarx choice) for :math:`Z` above -5 dBZ, with
    :func:`radarx.fundamentals.tmatrix.raindrop_scattering` (oblate drops
    with the axis ratio of Brandes et al. 2002, 7 degrees canting) at horizontal
    incidence. It is a statement about rain, not about drizzle or cloud droplets,
    and at W band the relation is poor where the reflectivity saturates (Mie
    resonances of drops of 1.5 mm and larger): see ``scatter``.

    Parameters
    ----------
    frequency : float
        Radar frequency in GHz, 1 to 1000 (35 for Ka band, 94 for W band).
    temperature : float, optional
        Temperature of the water in degrees Celsius. Default 10.

    Returns
    -------
    xarray.Dataset
        Scalar variables ``a`` (k in dB km-1 and Z in mm6 m-3) and ``b``, and ``scatter``,
        the standard deviation of the fit in decades of ``k``; the attributes
        hold the frequency, the temperature and the number of distributions.

    Raises
    ------
    ValueError
        For a frequency outside 1 to 1000 GHz.
    ImportError
        If the compiled T-matrix kernel is not available.

    References
    ----------
    Mishchenko, M. I., and L. D. Travis, 1994: T-matrix computations of light
    scattering by large spheroidal particles. *Opt. Commun.*, **109**, 16-21,
    https://doi.org/10.1016/0030-4018(94)90731-5

    Brandes, E. A., G. Zhang, and J. Vivekanandan, 2002: Experiments in rainfall
    estimation with a polarimetric radar in a subtropical environment. *J.
    Appl. Meteor.*, **41**, 674-685,
    https://doi.org/10.1175/1520-0450(2002)041<0674:EIREWA>2.0.CO;2

    Examples
    --------
    >>> rain_attenuation_relation(35.0)  # doctest: +SKIP
    """
    frequency = float(frequency)
    if not 1.0 <= frequency <= 1000.0:
        raise ValueError(f"the frequency must be 1 to 1000 GHz, not {frequency:g}")
    a, b, scatter, n = _zk_fit(frequency, float(temperature))
    out = xr.Dataset(
        {
            "a": (
                (),
                a,
                {
                    "long_name": "Coefficient of k = a Z**b (k in dB km-1, Z in mm6 m-3)",
                    "units": "1",
                },
            ),
            "b": ((), b, {"long_name": "Exponent of k = a Z**b", "units": "1"}),
            "scatter": (
                (),
                scatter,
                {
                    "long_name": "Standard deviation of log10 k about the fit",
                    "units": "1",
                },
            ),
        }
    )
    out.attrs = {
        "frequency_GHz": frequency,
        "temperature_celsius": float(temperature),
        "distributions": n,
        "method": "k = a Z**b, T-matrix, gamma drop size distributions",
    }
    return out


@provenance(
    "Hitschfeld and Bordan (1954) correction of the rain attenuation of a cloud radar "
    "with a fitted power-law relation between specific attenuation and reflectivity",
)
def rain_path_attenuation(
    obj,
    frequency=None,
    *,
    dbz=None,
    temperature=10.0,
    a=None,
    b=None,
    mask=None,
    melting=None,
    height=None,
    radar_height=None,
    max_pia=10.0,
    n_threads=None,
    engine="auto",
):
    """
    Two-way rain attenuation along the beam by the Hitschfeld and Bordan solution.

    With the relation ``k = a Z**b`` of :func:`rain_attenuation_relation` the
    solution of Hitschfeld and Bordan (1954),
    :func:`radarx.fundamentals.reflectivity.hitschfeld_bordan_correction`, gives
    the path-integrated attenuation at every gate from the measured reflectivity
    alone. Only gates of rain count: gates that are not in ``mask`` or are at or
    above ``melting`` add nothing, and gates behind the rain carry the
    attenuation of the whole rain path (the attenuation does not decrease).

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        The measured (attenuated) reflectivity in dBZ on a ``range``
        dimension, a dataset with it, or a volume. Correct the gas attenuation
        first if it matters.
    frequency : float, optional
        Frequency in GHz. Default: the ``radar_frequency`` of the data.
    dbz : str or xarray.DataArray, optional
        The reflectivity. Default: ``obj`` itself for a DataArray, else the first of
        ``DBZH``, ``DBZ``, ``reflectivity``, ``Z``, ``Zh``.
    temperature : float, optional
        Temperature of the rain in degrees Celsius for the relation. Default 10.
    a, b : float, optional
        Coefficients of ``k = a Z**b`` to use instead of the fit.
    mask : str or xarray.DataArray, optional
        True for the gates that can hold rain.
    melting : float, optional
        Height in m above sea level of the melting level; gates at or above it do
        not count. Needs the gate heights (see ``height``).
    height : str or xarray.DataArray, optional
        Height of the gates in m above sea level, as for
        :func:`gas_path_attenuation`.
    radar_height : float, optional
        Radar altitude in m above sea level for the 4/3 Earth model.
    max_pia : float, optional
        Largest two-way attenuation in dB that the solution may reach (it is
        unstable beyond). Default 10.
    n_threads : int, optional
        Threads of the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation. ``"auto"`` (default) prefers the compiled kernel.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        ``PIA_RAIN`` (dB, two-way) on the gates of the reflectivity; 0 before
        the first rain gate. A volume gives a DataTree with the root of the
        input and one node per sweep.

    Raises
    ------
    ValueError
        For a frequency outside 1 to 1000 GHz, an unknown ``engine`` or ``melting``
        without gate heights.
    KeyError
        If no reflectivity is found.

    References
    ----------
    Hitschfeld, W., and J. Bordan, 1954: Errors inherent in the radar
    measurement of rainfall at attenuating wavelengths. *J. Meteor.*, **11**
    (1), 58-67,
    https://doi.org/10.1175/1520-0469(1954)011<0058:EIITRM>2.0.CO;2

    Examples
    --------
    >>> pia = rain_path_attenuation(sweep, 35.0, melting=3000.0)  # doctest: +SKIP
    >>> dbz_corrected = sweep.DBZ + pia  # doctest: +SKIP
    """
    choose_engine(engine, True, "Hitschfeld-Bordan")

    def one(sweep, name=None):
        ref = _reflectivity(sweep, dbz)
        if _RANGE not in ref.dims:
            raise ValueError(
                f"the reflectivity needs a 'range' dimension, not {ref.dims}"
            )
        if a is None or b is None:
            relation = rain_attenuation_relation(
                _frequency_ghz(sweep, frequency), temperature
            )
            coef_a = float(relation.a) if a is None else float(a)
            coef_b = float(relation.b) if b is None else float(b)
        else:
            coef_a, coef_b = float(a), float(b)
        valid = np.isfinite(ref)
        if mask is not None:
            m = sweep[mask] if isinstance(mask, str) else mask
            valid = valid & m.astype(bool)
        if melting is not None:
            h = _gate_height(ref, sweep, height, radar_height)
            valid = valid & (h < float(melting))
        rain = ref.where(valid)
        _, pia = hitschfeld_bordan_correction(
            rain,
            a=coef_a,
            b=coef_b,
            max_correction_db=max_pia,
            return_pia=True,
            n_threads=n_threads,
            engine=engine,
        )
        # the attenuation of the rain path carries on behind the rain gates
        pia = xr.apply_ufunc(
            lambda x: np.maximum.accumulate(np.nan_to_num(x, nan=0.0), axis=-1),
            pia,
            input_core_dims=[[_RANGE]],
            output_core_dims=[[_RANGE]],
            dask="parallelized",
            dask_gufunc_kwargs={"allow_rechunk": True},
            output_dtypes=[np.float64],
        ).transpose(*ref.dims)
        out = as_field(pia.rename("PIA_RAIN"))
        out.attrs = {
            "long_name": "Two-way path-integrated attenuation by rain (Hitschfeld-Bordan)",
            "units": "dB",
            "k_Z_relation": f"k = {coef_a:.4g} Z**{coef_b:.3g}",
        }
        return out

    if isinstance(obj, xr.DataTree):
        return map_sweeps(obj, one, "a reflectivity")
    if not isinstance(obj, (xr.DataArray, xr.Dataset)):
        raise TypeError("obj must be an xarray DataArray, Dataset or DataTree")
    return one(obj)


# --------------------------------------------------------------------------
# accessors
# --------------------------------------------------------------------------


@accessor_method(
    "dataarray",
    "dataset",
    "datatree",
    name="gas_path_attenuation",
    wraps=gas_path_attenuation,
)
def _gas_path_attenuation_accessor(self):
    """
    Two-way gas (and cloud liquid) attenuation along the beam of a cloud radar.

    See :func:`radarx.retrieve.gas_path_attenuation` for the parameters.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``PIA_OXYGEN``, ``PIA_WATER_VAPOR``, ``PIA_GAS`` and, with ``lwc``,
        ``PIA_CLOUD_LIQUID`` (dB, two-way).
    """


@accessor_method(
    "dataarray",
    "dataset",
    "datatree",
    name="rain_path_attenuation",
    wraps=rain_path_attenuation,
)
def _rain_path_attenuation_accessor(self):
    """
    Two-way rain attenuation along the beam by the Hitschfeld and Bordan solution.

    See :func:`radarx.retrieve.rain_path_attenuation` for the parameters.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        ``PIA_RAIN`` (dB, two-way).
    """
