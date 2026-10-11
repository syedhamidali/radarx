#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx Hail Products
====================

Hail diagnostics of the reflectivity columns of a radar volume: the severe
hail index (SHI), the maximum expected size of hail (MESH), the probability of
severe hail (POSH) and the hail kinetic energy flux. They are the products of
the hail detection algorithm of the WSR-88D (Witt et al. 1998), computed on
the same columns as :func:`radarx.retrieve.vil`.

Columns
-------

A column is a list of reflectivity samples at known heights. A polar volume
(``xarray.DataTree`` with ``sweep_*`` groups) gives one column per (azimuth,
ground range) position of its lowest sweep, filled from the nearest ray and
gate of every other sweep at its beam height; a grid (``xarray.Dataset`` with
``z``, ``y``, ``x``) gives one column per (y, x); a quasi-vertical profile
(``height``, usually ``time``) gives one column per profile. See
:mod:`radarx.retrieve.vil` for the details of the three kinds.

Equations
---------

All equations are those of Witt et al. (1998), read in a copy of the paper
(Eqs. 1 to 6 and the worked example of their Fig. 4). With ``Z`` in dBZ and
heights ``H`` above the radar,

.. math::

    \\dot E = 5\\times10^{-6}\\,10^{0.084 Z}\\,W(Z)
    \\quad [\\mathrm{J\\,m^{-2}\\,s^{-1}}]

.. math::

    W(Z) = \\begin{cases} 0 & Z \\le Z_L \\\\
    (Z - Z_L)/(Z_U - Z_L) & Z_L < Z < Z_U \\\\ 1 & Z \\ge Z_U \\end{cases}
    \\qquad
    W_T(H) = \\begin{cases} 0 & H \\le H_0 \\\\
    (H - H_0)/(H_{m20} - H_0) & H_0 < H < H_{m20} \\\\ 1 & H \\ge H_{m20}
    \\end{cases}

.. math::

    \\mathrm{SHI} = 0.1 \\int_{H_0}^{H_T} W_T(H)\\,\\dot E\\,\\mathrm{d}H
    \\quad [\\mathrm{J\\,m^{-1}\\,s^{-1}}]

with ``Z_L = 40`` dBZ and ``Z_U = 50`` dBZ (the paper's defaults, "adaptable"),
``H_0`` and ``H_m20`` the heights of the 0 and -20 degC levels, ``H_T`` the top
of the storm and ``H`` in m. The discrete sum of the paper is used: the
reflectivity of each sample is applied over a layer of depth ``dH_i`` which is
half the distance between its two neighbours for an interior sample, the
distance to the next lower sample for the top sample, and, for the lowest
sample when it is above ``H_0``, the layer from ``H_0`` to the middle between
it and the next sample. A column needs two samples. Then

.. math::

    \\mathrm{MESH} = 2.54\\,\\mathrm{SHI}^{0.5}\\ [\\mathrm{mm}],\\qquad
    \\mathrm{WT} = \\max(57.5\\,H_0 - 121,\\ 20),\\qquad
    \\mathrm{POSH} = 29 \\ln(\\mathrm{SHI}/\\mathrm{WT}) + 50

with ``H_0`` in km above radar level, POSH limited to 0 to 100 %. MESH was fitted
so that about 75 % of 147 reports were below it (Witt et al. 1998, Eq. 6 and
Table 6). WT is the warning threshold of the paper (the SHI at which POSH is
50 %), an empirical fit of the best threshold of 8 storm days to the melting
level. The operational algorithm rounds POSH to 10 % and MESH to 6.35 mm; the
values here are not rounded.

Hail kinetic energy. ``hail_kinetic_energy`` is the flux ``E`` of Eq. 1 at the
lowest sample of each column, the closest to the ground. The relation between
reflectivity and the kinetic energy flux of hail is from the hailstone spectra
of Waldvogel et al. (1978a, b), as given in Eq. 1 of Witt et al. (1998) and in
Eq. 2.4 of Holleman (2001), who writes it as ``E = 5.0e-6 Z^0.84`` with ``Z``
in mm6 m-3, the same law. Waldvogel et al. (1978) were not read, so the fit,
its scatter and its range of validity are not checked against them.

Heights and levels
------------------

The heights of the samples are above sea level, as in :func:`radarx.retrieve.vil`.
The 0 and -20 degC levels (``freezing_level``, ``minus20_level``) are in the
same frame, as the profile functions of :mod:`radarx.io.sounding` give them. They
can be numbers, arrays on the columns (e.g. the freezing level of an ERA5
column per grid cell) or taken from a sounding or ERA5 profile
(``profile=``, through :func:`radarx.io.sounding.isotherm_height`). ``W_T``
only needs their difference; the warning threshold needs the height of the 0 degC
level above the radar, which is the level minus ``radar_height`` (the site
altitude by default).

Not implemented
---------------

* The probability of hail of any size (POH) of Witt et al. (1998) is read off a
  figure (their Fig. 2) that is derived from Waldvogel et al. (1979), which was
  not read; the paper gives no equation. It is left out.
* Alternative fits of MESH to hail reports (e.g. 75th and 95th percentiles of
  Murillo and Homeyer 2019) were not read and are not included.
* Witt et al. (1998) apply the equations to the storm cells of a cell tracking
  algorithm (the maximum reflectivity of each 2D component). Here the same sum
  is applied to the reflectivity of every column, which is what the MRMS
  version does on its 3D grid according to the NOAA Warning Decision Training
  Division product guide. The guide gives the same Eqs. 1 to 5 (as images) and
  the equation image on its MESH page reads ``MESH(inch) = sqrt(SHI)``, which
  differs by a factor 10 from Eq. 6 of the paper (``2.54 SHI^0.5`` mm, or
  ``0.1 SHI^0.5`` inch); that was not resolved and the paper is followed.
* A comparison with the MRMS MESH product was not possible (no MESH data).

The column sums run in a compiled, multithreaded C++ kernel; if it is not
available an equivalent NumPy implementation is used.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = ["hail_products", "shi", "mesh", "posh", "hail_kinetic_energy"]

__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

import numpy as np
import xarray as xr

from .._provenance import provenance
from .._registry import accessor_method
from ._fieldtools import choose_engine
from .vil import (
    _find_dbz,
    _kind_of,
    _polar_columns,
    _vertical_dim,
    _volume_field,
    _wrap,
)

try:
    from . import _hail

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _hail = None
    HAS_COMPILED_KERNEL = False

# Witt et al. (1998)
ENERGY_COEFFICIENT = 5.0e-6  # J m-2 s-1, Eq. 1
ENERGY_EXPONENT = 0.084  # per dBZ, Eq. 1
SHI_FACTOR = 0.1  # Eq. 3
WARNING_SLOPE = 57.5  # J m-1 s-1 per km, Eq. 4
WARNING_OFFSET = 121.0  # J m-1 s-1, Eq. 4
WARNING_MINIMUM = 20.0  # J m-1 s-1, below Eq. 4
POSH_COEFFICIENT = 29.0  # Eq. 5
POSH_OFFSET = 50.0  # %, Eq. 5
MESH_COEFFICIENT = 2.54  # mm, Eq. 6
MESH_EXPONENT = 0.5  # Eq. 6
ZERO_C = 273.15  # K
MINUS_20_C = 253.15  # K

_CHUNK = 2_000_000  # samples per kernel call of a polar volume, as in vil
_PROFILE_HEIGHTS = ("height", "z")


# --------------------------------------------------------------------------
# column products: NumPy reference of the kernel
# --------------------------------------------------------------------------


def _energy_flux(dbz, zl, zu):
    """E (J m-2 s-1) of Eq. 1 for reflectivity ``dbz`` (dBZ); 0 where W(Z) = 0."""
    with np.errstate(invalid="ignore", over="ignore"):
        w = np.where(dbz >= zu, 1.0, (dbz - zl) / (zu - zl))
        e = ENERGY_COEFFICIENT * np.power(10.0, ENERGY_EXPONENT * dbz) * w
    return np.where(dbz > zl, e, 0.0)


def _columns_numpy(
    h, v, h0, hm20, shared, radar_height, zl, zu, warning_offset, posh_offset
):
    """
    NumPy implementation of the kernel ``_hail.columns`` (same results).

    ``v`` has the shape (levels, columns); ``h`` is (levels,) when ``shared``
    and (levels, columns) otherwise; ``h0`` and ``hm20`` are (columns,).
    Returns the array (SHI, MESH, POSH, WT, hail kinetic energy flux of the
    lowest sample, reflectivity of the highest sample) of shape (6, columns).
    """
    nk, nc = v.shape
    h = np.broadcast_to(h[:, None], v.shape) if shared else h
    valid = np.isfinite(h) & ~np.isnan(v)
    order = np.argsort(np.where(valid, h, np.inf), axis=0, kind="stable")
    hs = np.take_along_axis(h, order, 0)
    vs = np.take_along_axis(v, order, 0)
    n = valid.sum(axis=0)
    idx = np.arange(nk)[:, None]
    live = idx < n[None, :]

    levels = np.isfinite(h0) & np.isfinite(hm20) & (hm20 > h0)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        e = np.where(live, _energy_flux(vs, zl, zu), 0.0)
        wt = np.clip((hs - h0[None]) / (hm20 - h0)[None], 0.0, 1.0)
        below = np.vstack([hs[:1], hs[:-1]])
        above = np.vstack([hs[1:], hs[-1:]])
        depth = np.where(
            idx == 0,
            0.5 * (hs + above) - h0[None],
            np.where(idx == n[None, :] - 1, hs - below, 0.5 * (above - below)),
        )
        use = live & (e > 0.0) & (wt > 0.0)
        total = np.where(use, wt * e * depth, 0.0).sum(axis=0)
        shi = SHI_FACTOR * total
        mesh = MESH_COEFFICIENT * np.power(shi, MESH_EXPONENT)
        warn = np.maximum(
            WARNING_SLOPE * (h0 - radar_height) / 1000.0 - warning_offset,
            WARNING_MINIMUM,
        )
        posh = np.where(
            shi > 0.0,
            np.clip(POSH_COEFFICIENT * np.log(shi / warn) + posh_offset, 0.0, 100.0),
            0.0,
        )
    ok = (n > 1) & levels
    nan = np.nan
    first = np.where(n > 0, _energy_flux(vs[0], zl, zu), nan)
    last = np.where(
        n > 0, np.take_along_axis(vs, np.maximum(n - 1, 0)[None], 0)[0], nan
    )
    return np.stack(
        [
            np.where(ok, shi, nan),
            np.where(ok, mesh, nan),
            np.where(ok, posh, nan),
            np.where(ok, warn, nan),
            first,
            last,
        ]
    )


def _columns(h, v, h0, hm20, shared, options, radar_height, use_compiled, n_threads):
    """Run the kernel, or its NumPy reference, on the columns ``v``."""
    args = (
        np.ascontiguousarray(h, dtype=np.float64),
        np.ascontiguousarray(v, dtype=np.float64),
        np.ascontiguousarray(h0, dtype=np.float64),
        np.ascontiguousarray(hm20, dtype=np.float64),
        bool(shared),
        float(radar_height),
        float(options["zl"]),
        float(options["zu"]),
        float(options["warning_offset"]),
        float(options["posh_offset"]),
    )
    if use_compiled:
        return _hail.columns(*args, n_threads=int(n_threads or 0))
    return _columns_numpy(*args)


# --------------------------------------------------------------------------
# the 0 and -20 degC levels
# --------------------------------------------------------------------------


def _profile_levels(profile):
    """Heights (m) of the 0 and -20 degC levels of a profile, from the temperature."""
    from ..io.sounding import isotherm_height

    if not isinstance(profile, xr.Dataset):
        raise TypeError("profile must be an xarray.Dataset with 'temperature'")
    if "temperature" not in profile:
        raise KeyError("profile has no 'temperature'")
    if "height" not in profile.dims:
        for name in _PROFILE_HEIGHTS:
            if name in profile.dims:
                profile = profile.rename({name: "height"})
                break
    if "height" not in profile.dims:
        raise ValueError("profile needs a 'height' (or 'z') dimension")
    return (
        isotherm_height(profile, ZERO_C),
        isotherm_height(profile, MINUS_20_C),
    )


def _level_field(value, template, name):
    """Height (m) of a level on the columns of ``template``, as a flat array."""
    if isinstance(value, xr.DataArray):
        try:
            out = value.broadcast_like(template).transpose(*template.dims)
        except ValueError as err:
            raise ValueError(
                f"{name} {value.dims} does not fit the columns {template.dims}"
            ) from err
        if out.shape != template.shape:
            raise ValueError(
                f"{name} {dict(value.sizes)} does not match the columns "
                f"{dict(template.sizes)}"
            )
        return np.asarray(out.values, dtype=np.float64).reshape(-1)
    try:
        number = float(value)
    except (TypeError, ValueError) as err:
        raise TypeError(
            f"{name} must be a number, an xarray.DataArray or None, not "
            f"{type(value).__name__}"
        ) from err
    return np.full(template.size, number)


def _levels(freezing_level, minus20_level, profile, template, need_levels=True):
    """Flat 0 and -20 degC heights on the columns, from numbers, arrays or a profile."""
    if not need_levels:
        nan = np.full(template.size, np.nan)
        return nan, nan
    if profile is not None:
        h0_p, hm_p = _profile_levels(profile)
        freezing_level = h0_p if freezing_level is None else freezing_level
        minus20_level = hm_p if minus20_level is None else minus20_level
    if freezing_level is None or minus20_level is None:
        raise ValueError(
            "the heights of the 0 and -20 degC levels are needed: give "
            "freezing_level and minus20_level, or profile= (a sounding or "
            "ERA5 profile with 'temperature')"
        )
    h0 = _level_field(freezing_level, template, "freezing_level")
    hm20 = _level_field(minus20_level, template, "minus20_level")
    scalar = not isinstance(freezing_level, xr.DataArray) and not isinstance(
        minus20_level, xr.DataArray
    )
    if scalar and np.isfinite(h0[0]) and np.isfinite(hm20[0]) and not hm20[0] > h0[0]:
        raise ValueError(
            "minus20_level must be above freezing_level "
            f"({float(minus20_level):g} m is not above {float(freezing_level):g} m)"
        )
    return h0, hm20


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def _options(zl, zu, warning_offset, posh_offset):
    if not zu > zl:
        raise ValueError(f"zu ({zu}) must be above zl ({zl})")
    return {
        "zl": float(zl),
        "zu": float(zu),
        "warning_offset": float(warning_offset),
        "posh_offset": float(posh_offset),
    }


def _hail_columns(
    obj,
    dbz,
    kind,
    options,
    freezing_level,
    minus20_level,
    profile,
    missing,
    radar_height,
    tolerance,
    n_threads,
    engine,
    need_levels=True,
):
    """
    Run the column sums.

    Returns the 6 products as arrays on ``template`` and ``template`` (a
    DataArray with the output dims and coordinates).
    """
    use_compiled = choose_engine(engine, HAS_COMPILED_KERNEL, "hail")
    kind = _kind_of(obj, kind)
    if missing not in (None, "clear", "gap"):
        raise ValueError(f"missing must be 'clear' or 'gap', not {missing!r}")
    clear = (missing == "clear") if missing is not None else kind == "volume"

    if kind == "volume":
        name = dbz or _volume_field(obj)
        template, ground, altitude, blocks = _polar_columns(
            obj, name, tolerance, None, clear
        )
        radar = altitude if radar_height is None else float(radar_height)
        h0, hm20 = _levels(
            freezing_level, minus20_level, profile, template, need_levels
        )
        out = np.full((6, template.size), np.nan)
        ns = ground.size
        for rows, h, _, v in blocks():
            cols = (rows[:, None] * ns + np.arange(ns)[None, :]).reshape(-1)
            out[:, cols] = _columns(
                h,
                v,
                h0[cols],
                hm20[cols],
                False,
                options,
                radar,
                use_compiled,
                n_threads,
            )
        template = template.assign_coords(
            ground_range=(
                "range",
                ground,
                {"long_name": "ground range of the gate", "units": "m"},
            )
        )
    else:
        name = _find_dbz(obj, dbz)
        da = obj[name]
        zdim = _vertical_dim(da, kind)
        da = da.transpose(zdim, ...)
        template = da.isel({zdim: 0}, drop=True)
        z = np.asarray(da[zdim].values, dtype=np.float64)
        site = obj["altitude"] if "altitude" in obj else None
        if radar_height is not None:
            radar = float(radar_height)
        elif site is not None and site.size == 1:
            radar = float(site)
        else:
            radar = 0.0
        v = da.values.reshape(z.size, -1)
        if clear:
            seen = np.maximum.accumulate(~np.isnan(v), axis=0)
            v = np.where(seen & np.isnan(v), -np.inf, v)
        h0, hm20 = _levels(
            freezing_level, minus20_level, profile, template, need_levels
        )
        out = _columns(z, v, h0, hm20, True, options, radar, use_compiled, n_threads)
    products = [out[i].reshape(template.shape) for i in range(6)]
    return products, template


# --------------------------------------------------------------------------
# result variables
# --------------------------------------------------------------------------

_SOURCE = "Witt et al. (1998)"


def _comment(options):
    return (
        f"{_SOURCE}, ZL = {options['zl']:g} dBZ, ZU = {options['zu']:g} dBZ; "
        "heights of the samples and levels above sea level"
    )


def _shi_array(template, shi_, options):
    return _wrap(
        template,
        shi_,
        "SHI",
        {
            "long_name": "severe hail index",
            "units": "J m-1 s-1",
            "comment": _comment(options)
            + "; SHI = 0.1 sum of W_T(H) E dH over the samples of the column",
        },
    )


def _mesh_array(template, mesh_, options):
    return _wrap(
        template,
        mesh_,
        "MESH",
        {
            "long_name": "maximum expected size of hail",
            "units": "mm",
            "comment": _comment(options)
            + "; MESH = 2.54 SHI^0.5, not rounded to 6.35 mm",
        },
    )


def _posh_array(template, posh_, options):
    return _wrap(
        template,
        posh_,
        "POSH",
        {
            "long_name": "probability of severe hail",
            "units": "%",
            "comment": _comment(options)
            + f"; POSH = 29 ln(SHI / WT) + {options['posh_offset']:g}, limited to "
            "0-100, not rounded to 10 %",
        },
    )


def _warning_array(template, warn, options):
    return _wrap(
        template,
        warn,
        "SHI_WARNING_THRESHOLD",
        {
            "long_name": "warning threshold of the probability of severe hail",
            "units": "J m-1 s-1",
            "comment": _comment(options)
            + f"; WT = max(57.5 H0 - {options['warning_offset']:g}, 20) with H0 in "
            "km above the radar",
        },
    )


def _hke_array(template, hke, options):
    return _wrap(
        template,
        hke,
        "HKE",
        {
            "long_name": "hail kinetic energy flux",
            "units": "J m-2 s-1",
            "comment": (
                f"{_SOURCE} Eq. 1, E = 5e-6 10^(0.084 Z) W(Z), ZL = "
                f"{options['zl']:g} dBZ, ZU = {options['zu']:g} dBZ, of the "
                "lowest sample of the column; no temperature weighting"
            ),
        },
    )


def _top_flag(template, shi_, last, zl):
    """1 where the highest sample of the column still has hail-sized reflectivity."""
    with np.errstate(invalid="ignore"):
        cut = np.isfinite(shi_) & np.isfinite(last) & (last > zl)
    out = template.copy(data=cut.astype(np.int8))
    out.attrs = {
        "long_name": "hail echo continues above the highest sample",
        "comment": (
            "1: the reflectivity of the highest valid sample is above ZL, so the "
            "part of the storm above it (e.g. above the highest beam near the "
            "radar) is not included in the SHI"
        ),
        "flag_values": np.array([0, 1], dtype=np.int8),
        "flag_meanings": "complete top_truncated",
    }
    out.name = "SHI_TOP_TRUNCATED"
    return out


# --------------------------------------------------------------------------
# documentation shared by the public functions
# --------------------------------------------------------------------------

_P_DBZ = """
    dbz : str, optional
        Name of the reflectivity (dBZ). By default the first found of
        ``DBZH``, ``DBZ``, ``reflectivity`` and ``corrected_reflectivity``
        (for a volume, in the first sweep that has one)."""

_P_LEVELS = """
    freezing_level, minus20_level : float or xarray.DataArray, optional
        Height (m above sea level) of the 0 and the -20 degC level: a number,
        or an array that broadcasts against the columns (a 2-D field on the
        grid or the polar columns, or one value per time of a profile). NaN
        where unknown, which gives NaN in the products. Not needed where
        ``profile`` gives them.
    profile : xarray.Dataset, optional
        A sounding or ERA5 profile with ``temperature`` (K) on a ``height``
        (or ``z``) dimension, e.g. from :func:`radarx.io.sounding.read_sounding`,
        :func:`radarx.io.sounding.era5_profile` or
        :func:`radarx.io.sounding.era5_column`. The heights of the 0 and -20
        degC isotherms (the highest crossing, see
        :func:`radarx.io.sounding.isotherm_height`) are used for the levels that
        are not given."""

_P_WEIGHT = """
    zl, zu : float, optional
        Reflectivities (dBZ) between which the weight W(Z) rises from 0 to 1.
        Defaults 40 and 50, the values of Witt et al. (1998), who note that
        they are adaptable."""

_P_WARNING = """
    warning_offset, posh_offset : float, optional
        The constants 121 (J m-1 s-1) of the warning threshold
        ``WT = 57.5 H0 - 121`` and 50 (%) of ``POSH = 29 ln(SHI/WT) + 50`` of
        Witt et al. (1998). The Western Region technical attachment 98-27
        (Graham and Nelson 1998) reports that the threshold model over-warned
        for radars high above sea level and quotes ``WT = 57.5 H0 - 5.8`` and an
        offset of 30 % for summer storms as the later NSSL adjustment, which
        these two options can reproduce. That document was read as a scan; the
        operational values were not checked in another source."""

_P_COLUMNS = """
    kind : {"volume", "grid", "qvp"}, optional
        Kind of input. By default a DataTree is a polar volume, a Dataset with
        ``x`` and ``y`` is a grid and one with ``height`` a quasi-vertical
        profile.
    missing : {"clear", "gap"}, optional
        What a missing (NaN) value means. ``"clear"``: no echo, which is how
        radar volumes mark it (for a grid or a profile, only the values above
        the lowest valid level); ``"gap"``: no data, left out. Default
        ``"clear"`` for a volume and ``"gap"`` for a grid or a profile."""

_P_RADAR = """
    radar_height : float, optional
        Height of the radar (m above sea level) that the warning threshold
        takes the 0 degC level above. Default: the ``altitude`` of the radar
        site, else 0."""

_P_RUN = """
    tolerance : float, optional
        Volume only: sweeps closer than this in elevation (degrees) count as
        the same cut, and the one that reaches farthest is used. Default 0.1.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled
        kernel and falls back to NumPy."""

_RAISES = """
    Raises
    ------
    KeyError
        If the reflectivity or no sweep is found.
    ValueError
        If the levels are missing or do not fit the columns, or ``zu`` is not
        above ``zl``.
    ImportError
        If ``engine="compiled"`` and the compiled kernel is not available."""

_REFERENCES = """
    References
    ----------
    Witt, A., M. D. Eilts, G. J. Stumpf, J. T. Johnson, E. D. Mitchell, and
    K. W. Thomas, 1998: An enhanced hail detection algorithm for the WSR-88D.
    *Wea. Forecasting*, **13** (2), 286-303,
    https://doi.org/10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2
    (Eqs. 1-6 and Fig. 4, read in a copy of the paper).

    Waldvogel, A., W. Schmid, and B. Federer, 1978: The kinetic energy of
    hailfalls. Part I: Hailstone spectra. *J. Appl. Meteor.*, **17** (4),
    515-520, https://doi.org/10.1175/1520-0450(1978)017<0515:TKEOHP>2.0.CO;2
    (not read; the reflectivity to energy flux relation is taken from Witt et
    al. 1998 and Holleman 2001).

    Waldvogel, A., B. Federer, W. Schmid, and J. F. Mezeix, 1978: The kinetic
    energy of hailfalls. Part II: Radar and hailpads. *J. Appl. Meteor.*,
    **17** (11), 1680-1693,
    https://doi.org/10.1175/1520-0450(1978)017<1680:TKEOHP>2.0.CO;2
    (not read; the relation is taken from Witt et al. 1998 and Holleman 2001).

    Holleman, I., 2001: Hail detection using single-polarization radar.
    KNMI Scientific Report WR-2001-01 (Eq. 2.4 for ``E = 5.0e-6 Z^0.84`` and
    Eqs. 2.5 to 2.7 for the SHI).

    Graham, R. A., and J. A. Nelson, 1998: An examination into the utility of
    the Build 9 hail detection algorithm (HDA) across northern Utah. NWS
    Western Region Technical Attachment No. 98-27 (warning threshold and POSH
    offsets of the later adjustment)."""


def _document(func, warning=True, levels=True):
    """Fill the shared parameter blocks into the docstring of ``func``."""
    parts = [_P_DBZ]
    if levels:
        parts.append(_P_LEVELS)
    parts.append(_P_WEIGHT)
    if warning:
        parts.append(_P_WARNING)
    parts.append(_P_COLUMNS)
    if warning:
        parts.append(_P_RADAR)
    parts.append(_P_RUN)
    func.__doc__ = (
        func.__doc__.replace(
            "{params}", "\n".join(p.strip("\n") for p in parts).strip()
        )
        .replace("{raises}", _RAISES.strip())
        .replace("{references}", _REFERENCES.strip())
    )
    return func


def _compute(
    obj,
    dbz,
    kind,
    levels,
    zl,
    zu,
    warning_offset,
    posh_offset,
    missing,
    radar_height,
    tolerance,
    n_threads,
    engine,
):
    """Products on the columns of ``obj``; ``levels`` is None or a 3-tuple."""
    options = _options(zl, zu, warning_offset, posh_offset)
    freezing_level, minus20_level, profile = levels or (None, None, None)
    products, template = _hail_columns(
        obj,
        dbz,
        kind,
        options,
        freezing_level,
        minus20_level,
        profile,
        missing,
        radar_height,
        tolerance,
        n_threads,
        engine,
        need_levels=levels is not None,
    )
    return products, template, options


# --------------------------------------------------------------------------
# public functions
# --------------------------------------------------------------------------


@provenance(
    "Hail detection algorithm of Witt et al. (1998): SHI, MESH, POSH, HKE",
    extra_refs=("holleman-2001", "graham-nelson-1998"),
)
def hail_products(
    obj,
    dbz=None,
    *,
    freezing_level=None,
    minus20_level=None,
    profile=None,
    zl=40.0,
    zu=50.0,
    warning_offset=WARNING_OFFSET,
    posh_offset=POSH_OFFSET,
    kind=None,
    missing=None,
    radar_height=None,
    tolerance=0.1,
    n_threads=None,
    engine="auto",
):
    """
    Severe hail index, maximum expected hail size, probability of severe hail
    and hail kinetic energy flux of the reflectivity columns.

    Implements the hail detection algorithm of Witt et al. (1998), with the
    equations of the paper (see :mod:`radarx.retrieve.hail`): the kinetic
    energy flux ``E = 5e-6 10^(0.084 Z) W(Z)``, the severe hail index
    ``SHI = 0.1 sum W_T(H) E dH`` weighted by the heights of the 0 and -20
    degC levels, ``MESH = 2.54 SHI^0.5`` and ``POSH = 29 ln(SHI/WT) + 50``.
    Where the column is below the 0 degC level, or has no reflectivity above
    ``zl``, SHI, MESH and POSH are 0.

    Parameters
    ----------
    obj : xarray.DataTree or xarray.Dataset
        A polar volume with ``sweep_*`` groups (sweeps without the field are
        skipped), a grid, or a quasi-vertical profile (time series).
    {params}

    Returns
    -------
    xarray.Dataset
        On the horizontal coordinates of the input (the polar grid of the
        lowest sweep for a volume, with ``ground_range``, the times for a QVP):

        ``SHI`` (J m-1 s-1)
            The severe hail index. NaN where the column has fewer than two
            valid samples or a level is not known.
        ``MESH`` (mm)
            ``2.54 SHI^0.5``.
        ``POSH`` (%)
            Probability of severe hail, 0 to 100.
        ``SHI_WARNING_THRESHOLD`` (J m-1 s-1)
            The warning threshold WT of POSH.
        ``HKE`` (J m-2 s-1)
            Hail kinetic energy flux of the lowest sample, see
            :func:`hail_kinetic_energy`.
        ``SHI_TOP_TRUNCATED``
            1 where the highest valid sample has reflectivity above ``zl``, so
            hail echo may continue above the highest beam (near the radar).

    {raises}

    Notes
    -----
    SHI integrates only above the 0 degC level, and the beams of a volume are
    far apart near the radar and at long range, so the columns there have few
    samples above that level; ``SHI_TOP_TRUNCATED`` marks the top of the cone
    of silence. The products are diagnostics of the reflectivity profile,
    calibrated by Witt et al. (1998) on 147 reports of nine storm days: they
    are not a measurement of the hail at the ground.

    {references}

    Examples
    --------
    >>> out = radarx.retrieve.hail_products(
    ...     dtree, freezing_level=4200.0, minus20_level=7500.0
    ... )  # doctest: +SKIP
    >>> out = dtree.radarx.hail_products(profile=sounding)  # doctest: +SKIP
    """
    products, template, options = _compute(
        obj,
        dbz,
        kind,
        (freezing_level, minus20_level, profile),
        zl,
        zu,
        warning_offset,
        posh_offset,
        missing,
        radar_height,
        tolerance,
        n_threads,
        engine,
    )
    shi_, mesh_, posh_, warn, hke, last = products
    return xr.Dataset(
        {
            "SHI": _shi_array(template, shi_, options),
            "MESH": _mesh_array(template, mesh_, options),
            "POSH": _posh_array(template, posh_, options),
            "SHI_WARNING_THRESHOLD": _warning_array(template, warn, options),
            "HKE": _hke_array(template, hke, options),
            "SHI_TOP_TRUNCATED": _top_flag(template, shi_, last, options["zl"]),
        }
    )


_document(hail_products)


@provenance(
    "Severe hail index of Witt et al. (1998)",
    extra_refs=("holleman-2001", "graham-nelson-1998"),
)
def shi(
    obj,
    dbz=None,
    *,
    freezing_level=None,
    minus20_level=None,
    profile=None,
    zl=40.0,
    zu=50.0,
    kind=None,
    missing=None,
    tolerance=0.1,
    n_threads=None,
    engine="auto",
):
    """
    Severe hail index (SHI) of the reflectivity columns.

    ``SHI = 0.1 sum W_T(H) E dH`` (J m-1 s-1) of Witt et al. (1998) Eq. 3, with
    the kinetic energy flux ``E = 5e-6 10^(0.084 Z) W(Z)`` of Eq. 1, the
    reflectivity weight ``W(Z)`` rising from 0 at ``zl`` to 1 at ``zu`` and the
    temperature weight ``W_T(H)`` rising from 0 at the 0 degC level to 1 at the
    -20 degC level. See :mod:`radarx.retrieve.hail` for the discrete sum.

    Parameters
    ----------
    obj : xarray.DataTree or xarray.Dataset
        A polar volume, a grid or a quasi-vertical profile.
    {params}

    Returns
    -------
    xarray.DataArray
        ``SHI`` (J m-1 s-1); 0 below the 0 degC level or where no sample is
        above ``zl``, NaN where the column has fewer than two valid samples or a
        level is not known.

    {raises}

    {references}

    Examples
    --------
    >>> index = radarx.retrieve.shi(
    ...     dtree, freezing_level=4200.0, minus20_level=7500.0
    ... )  # doctest: +SKIP
    """
    products, template, options = _compute(
        obj,
        dbz,
        kind,
        (freezing_level, minus20_level, profile),
        zl,
        zu,
        WARNING_OFFSET,
        POSH_OFFSET,
        missing,
        None,
        tolerance,
        n_threads,
        engine,
    )
    return _shi_array(template, products[0], options)


_document(shi, warning=False)


@provenance(
    "Maximum expected size of hail of Witt et al. (1998)",
    extra_refs=("holleman-2001", "graham-nelson-1998"),
)
def mesh(
    obj,
    dbz=None,
    *,
    freezing_level=None,
    minus20_level=None,
    profile=None,
    zl=40.0,
    zu=50.0,
    kind=None,
    missing=None,
    tolerance=0.1,
    n_threads=None,
    engine="auto",
):
    """
    Maximum expected size of hail (MESH) of the reflectivity columns.

    ``MESH = 2.54 SHI^0.5`` in mm (Witt et al. 1998, Eq. 6), fitted so that
    about 75 % of 147 hail reports were smaller (77, 70 and 73 % for the
    19-33, 33-60 and above 60 mm reports of their Table 6). It is an upper
    estimate of the largest stone, not the size at a point. The operational
    value is rounded to 6.35 mm; this one is not.

    Parameters
    ----------
    obj : xarray.DataTree or xarray.Dataset
        A polar volume, a grid or a quasi-vertical profile.
    {params}

    Returns
    -------
    xarray.DataArray
        ``MESH`` (mm); 0 where SHI is 0.

    {raises}

    {references}

    Examples
    --------
    >>> size = radarx.retrieve.mesh(
    ...     dtree, freezing_level=4200.0, minus20_level=7500.0
    ... )  # doctest: +SKIP
    """
    products, template, options = _compute(
        obj,
        dbz,
        kind,
        (freezing_level, minus20_level, profile),
        zl,
        zu,
        WARNING_OFFSET,
        POSH_OFFSET,
        missing,
        None,
        tolerance,
        n_threads,
        engine,
    )
    return _mesh_array(template, products[1], options)


_document(mesh, warning=False)


@provenance(
    "Probability of severe hail of Witt et al. (1998)",
    extra_refs=("holleman-2001", "graham-nelson-1998"),
)
def posh(
    obj,
    dbz=None,
    *,
    freezing_level=None,
    minus20_level=None,
    profile=None,
    zl=40.0,
    zu=50.0,
    warning_offset=WARNING_OFFSET,
    posh_offset=POSH_OFFSET,
    kind=None,
    missing=None,
    radar_height=None,
    tolerance=0.1,
    n_threads=None,
    engine="auto",
):
    """
    Probability of severe hail (POSH) of the reflectivity columns.

    ``POSH = 29 ln(SHI / WT) + 50`` in % (Witt et al. 1998, Eq. 5), limited to
    0 to 100, with the warning threshold ``WT = max(57.5 H0 - 121, 20)``
    (J m-1 s-1, ``H0`` in km above the radar) of Eq. 4. POSH is 50 % when SHI
    equals WT. The curve was calibrated with reliability diagrams of ten storm
    days, mostly in Oklahoma and Florida, so it depends on the region and
    the radar height (see ``warning_offset`` and ``posh_offset``). The
    operational value is rounded to 10 %; this one is not.

    Parameters
    ----------
    obj : xarray.DataTree or xarray.Dataset
        A polar volume, a grid or a quasi-vertical profile.
    {params}

    Returns
    -------
    xarray.DataArray
        ``POSH`` (%); 0 where SHI is 0.

    {raises}

    {references}

    Examples
    --------
    >>> prob = radarx.retrieve.posh(
    ...     dtree, freezing_level=4200.0, minus20_level=7500.0
    ... )  # doctest: +SKIP
    """
    products, template, options = _compute(
        obj,
        dbz,
        kind,
        (freezing_level, minus20_level, profile),
        zl,
        zu,
        warning_offset,
        posh_offset,
        missing,
        radar_height,
        tolerance,
        n_threads,
        engine,
    )
    return _posh_array(template, products[2], options)


_document(posh)


@provenance(
    "Hail kinetic energy flux of Waldvogel et al. (1978) as used by Witt et al. (1998)",
    extra_refs=("holleman-2001", "graham-nelson-1998"),
)
def hail_kinetic_energy(
    obj,
    dbz=None,
    *,
    zl=40.0,
    zu=50.0,
    kind=None,
    missing=None,
    tolerance=0.1,
    n_threads=None,
    engine="auto",
):
    """
    Hail kinetic energy flux of the lowest sample of each column.

    ``E = 5e-6 10^(0.084 Z) W(Z)`` in J m-2 s-1 (Witt et al. 1998, Eq. 1), the
    relation between reflectivity and the kinetic energy flux of hail that
    Waldvogel et al. (1978) fitted to hailstone spectra, with the weight
    ``W(Z)`` that rises from 0 at ``zl`` to 1 at ``zu`` to leave out the
    reflectivities of rain. It is the flux at the lowest beam, the closest to
    the ground, with no temperature weighting; the vertical sum of the flux
    weighted with the temperature is the severe hail index (:func:`shi`).
    It needs no freezing level. The flux is 0 for a lowest sample without
    echo or at or below ``zl``.

    The time integral of the flux over the duration of the hailfall (the
    kinetic energy per area of hailpads, J m-2) needs a series of volumes
    and is not computed here.

    Parameters
    ----------
    obj : xarray.DataTree or xarray.Dataset
        A polar volume, a grid or a quasi-vertical profile.
    {params}

    Returns
    -------
    xarray.DataArray
        ``HKE`` (J m-2 s-1); NaN where the column has no valid sample.

    {raises}

    {references}

    Examples
    --------
    >>> flux = radarx.retrieve.hail_kinetic_energy(dtree)  # doctest: +SKIP
    """
    products, template, options = _compute(
        obj,
        dbz,
        kind,
        None,
        zl,
        zu,
        WARNING_OFFSET,
        POSH_OFFSET,
        missing,
        None,
        tolerance,
        n_threads,
        engine,
    )
    return _hke_array(template, products[4], options)


_document(hail_kinetic_energy, warning=False, levels=False)


# --------------------------------------------------------------------------
# accessors
# --------------------------------------------------------------------------


@accessor_method("dataset", "datatree", name="hail_products")
def _hail_products_accessor(self, **kwargs):
    """
    SHI, MESH, POSH and hail kinetic energy flux of a volume, grid or profile.

    See :func:`radarx.retrieve.hail_products` for the parameters.

    Returns
    -------
    xarray.Dataset
        ``SHI``, ``MESH``, ``POSH``, ``SHI_WARNING_THRESHOLD``, ``HKE`` and
        ``SHI_TOP_TRUNCATED``.
    """
    return hail_products(self.xarray_obj, **kwargs)


@accessor_method("dataset", "datatree", name="shi")
def _shi_accessor(self, **kwargs):
    """
    Severe hail index of a volume, grid or profile.

    See :func:`radarx.retrieve.shi` for the parameters.

    Returns
    -------
    xarray.DataArray
        ``SHI`` (J m-1 s-1).
    """
    return shi(self.xarray_obj, **kwargs)


@accessor_method("dataset", "datatree", name="mesh")
def _mesh_accessor(self, **kwargs):
    """
    Maximum expected size of hail of a volume, grid or profile.

    See :func:`radarx.retrieve.mesh` for the parameters.

    Returns
    -------
    xarray.DataArray
        ``MESH`` (mm).
    """
    return mesh(self.xarray_obj, **kwargs)


@accessor_method("dataset", "datatree", name="posh")
def _posh_accessor(self, **kwargs):
    """
    Probability of severe hail of a volume, grid or profile.

    See :func:`radarx.retrieve.posh` for the parameters.

    Returns
    -------
    xarray.DataArray
        ``POSH`` (%).
    """
    return posh(self.xarray_obj, **kwargs)


@accessor_method("dataset", "datatree", name="hail_kinetic_energy")
def _hail_kinetic_energy_accessor(self, **kwargs):
    """
    Hail kinetic energy flux of a volume, grid or profile.

    See :func:`radarx.retrieve.hail_kinetic_energy` for the parameters.

    Returns
    -------
    xarray.DataArray
        ``HKE`` (J m-2 s-1).
    """
    return hail_kinetic_energy(self.xarray_obj, **kwargs)
