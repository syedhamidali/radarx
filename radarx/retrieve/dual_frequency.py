#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Dual-Frequency Ratio Retrievals for Rain and Snow
=================================================

Two radars at different frequencies, usually Ka band (35 GHz) and W band
(94 GHz), see the same particles with different reflectivity once the size of the
particles is no longer small against the shorter wavelength. The
dual-frequency ratio (DFR, called the dual-wavelength ratio, DWR, for ice)

.. math::

    \\mathrm{DFR} = 10 \\log_{10} (Z_{Ka} / Z_{W}) = Z_{Ka} - Z_{W}\\ \\text{(dB)}

is zero for particles in the Rayleigh regime at both frequencies and grows with the
size of the particles, so it carries the size of the particles without the
particle concentration and without a calibration of the absolute reflectivity
(only the relative calibration of the two radars matters). It also changes with the
attenuation of the two signals, which differs strongly (see
:mod:`radarx.retrieve.cloud_attenuation`), so the reflectivities must be corrected
for the attenuation of gases, cloud liquid water and rain before they are
subtracted.

What the functions do
---------------------
:func:`dual_frequency_ratio`
    Matches the two reflectivities in time and range (exactly, by the nearest
    gate or by interpolation), adds the attenuation corrections and gives the
    ratio.
:func:`dfr_rain`
    Mass-weighted mean drop diameter :math:`D_m`, normalized intercept parameter
    :math:`N_w` and liquid water content of rain from the ratio and the
    reflectivity of the short wavelength (Ka band). Gamma size distributions
    of fixed shape :math:`\\mu` (normalized, :math:`N(D) = N_w f(\\mu)
    (D/D_m)^\\mu \\exp[-(4 + \\mu) D/D_m]`), oblate raindrops of the axis ratio of
    Brandes et al. (2002), T-matrix scattering
    (:func:`radarx.fundamentals.tmatrix.raindrop_scattering`) at horizontal
    incidence. The ratio does not depend on :math:`N_w`, so a monotonic table
    of the ratio against :math:`D_m` gives the size, and the reflectivity gives
    :math:`N_w`. Tridon and Battaglia (2015) derive the drop size distribution
    from the Doppler spectra at the two frequencies; this function uses the
    reflectivities only and needs the shape :math:`\\mu` as an assumption.
:func:`dfr_snow`
    Mass-weighted mean maximum dimension :math:`D_m` and ice water content of snow
    and ice clouds, in the way of Hogan et al. (2000) and Matrosov (1998) who relate
    the ratio of 35 and 94 GHz to the size of the crystals. The particles are
    spheres of the maximum dimension :math:`D` with the mass of Brown and Francis
    (1995), :math:`m = 0.0185 D^{1.9}` (SI units), limited to solid ice, mixed with air by
    the Maxwell Garnett rule (Maxwell Garnett 1904) to an effective refractive
    index, ice of Warren and Brandt (2008), Mie scattering through the T-matrix
    code for spheres, and an exponential size distribution of maximum dimension
    (a gamma distribution with ``mu``). Soft spheres exaggerate the resonance of
    the aggregates at W band; see the Notes of the function.

Choices of radarx
-----------------
* The retrievals use only the reflectivities and assume the shape of the
  distribution; the full-spectrum methods are not implemented.
* Both retrievals return NaN where the ratio is outside the table: below the
  smallest tabulated ratio by more than ``tolerance`` and above the largest, where
  the ratio no longer grows with the size (saturation, a ratio that two sizes can
  have). A ratio that is below the smallest tabulated ratio by at most
  ``tolerance`` (noise around the Rayleigh limit) gets the smallest size of the table
  and no :math:`N_w`, liquid or ice water content: the particles are small and
  the reflectivity alone does not tell their number.
* The reflectivity of both frequencies is the equivalent reflectivity factor with
  :math:`|K_w|^2 = 0.93`, also for ice.
  Data calibrated in the way of Cloudnet (cloud droplets give the same dBZ at both
  frequencies, :math:`|K_w(f)|^2` of each frequency) are converted by
  ``calibration="liquid"`` of :func:`dual_frequency_ratio`, which adds 1.1 dB at 35 and
  94 GHz and 0 degrees Celsius.
* Hail is not retrieved: the ratio of Ka and W band of hail is dominated by Mie
  oscillations of the stones and is not monotonic in the size, so no table of
  the ratio against the size exists.

Implementation
--------------
A compiled, multithreaded C++ kernel interpolates the tables
(``radarx.retrieve._dfr``), with the equivalent NumPy implementation as
fallback and test oracle. The tables are built once per set of parameters from the
T-matrix scattering and cached. The functions work on :class:`xarray.DataArray`
fields of any dimensions and keep dask arrays lazy.

References
----------
.. [1] Hogan, R. J., A. J. Illingworth, and H. Sauvageot, 2000: Measuring crystal
       size in cirrus using 35- and 94-GHz radars. *J. Atmos. Oceanic Technol.*,
       **17**, 27-37,
       https://doi.org/10.1175/1520-0426(2000)017<0027:MCSICU>2.0.CO;2
.. [2] Matrosov, S. Y., 1998: A dual-wavelength radar method to measure snowfall
       rate. *J. Appl. Meteor.*, **37**, 1510-1521,
       https://doi.org/10.1175/1520-0450(1998)037<1510:ADWRMT>2.0.CO;2
.. [3] Tridon, F., and A. Battaglia, 2015: Dual-frequency radar Doppler spectral
       retrieval of rain drop size distributions and entangled dynamics
       variables. *J. Geophys. Res. Atmos.*, **120**, 5585-5601,
       https://doi.org/10.1002/2014JD023023
.. [4] Brown, P. R. A., and P. N. Francis, 1995: Improved measurements of the ice
       water content in cirrus using a total-water probe. *J. Atmos. Oceanic
       Technol.*, **12**, 410-414,
       https://doi.org/10.1175/1520-0426(1995)012<0410:IMOTIW>2.0.CO;2

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = ["dual_frequency_ratio", "dfr_rain", "dfr_snow"]

__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

import functools

import numpy as np
import xarray as xr
from scipy.special import gammaln

from .._dtypes import as_field
from .._provenance import provenance
from .._registry import accessor_method
from ._fieldtools import choose_engine, detach

try:
    from . import _dfr

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _dfr = None
    HAS_COMPILED_KERNEL = False

_KW2 = 0.93  # |K_w|^2 of the equivalent reflectivity factor
_MATCH = ("exact", "nearest", "interp")
_ICE_DENSITY = 917.0  # kg m-3
# Brown and Francis (1995): m = A D**B with m in kg and D in m
_BROWN_FRANCIS = (0.0185, 1.9)

# Refractive index of ice, Warren and Brandt (2008) compilation, microwave part
# (Maetzler 2006 at 266 K): wavelength [um], real part, imaginary part.
_ICE_INDEX = np.array(
    [
        [1.3e3, 1.7868, 5.173e-3],
        [5.0e3, 1.7861, 1.337e-3],
        [1.9e4, 1.7861, 3.574e-4],
        [3.9e4, 1.7861, 1.839e-4],
        [6.1e4, 1.7861, 1.294e-4],
    ]
)


# --------------------------------------------------------------------------
# scattering tables
# --------------------------------------------------------------------------


def _check_frequencies(frequencies, low=1.0, high=1000.0):
    try:
        f1, f2 = (float(f) for f in frequencies)
    except (TypeError, ValueError):
        raise ValueError("frequencies must be a pair (lower, higher) in GHz") from None
    if not (low <= f1 < f2 <= high):
        raise ValueError(
            f"frequencies must be two increasing values in {low:g}-{high:g} GHz, "
            f"not {frequencies!r}"
        )
    return f1, f2


def _gamma_shape(d, dm, mu):
    """Normalized gamma size distribution for N_w = 1; ``d`` (n), ``dm`` (m) -> (m, n)."""
    f_mu = 6.0 / 4.0**4 * (mu + 4.0) ** (mu + 4.0) / np.exp(gammaln(mu + 4.0))
    x = d[None, :] / dm[:, None]
    return f_mu * x**mu * np.exp(-(4.0 + mu) * x)


def _monotonic_segment(ratio):
    """
    Slice of the table over which the ratio increases with the size.

    From the smallest ratio before the largest one (the ratio of rain dips a
    little below its Rayleigh value before it rises, since the W band drops reach
    their first resonance) to the largest ratio, where the ratio stops growing.
    """
    top = int(np.argmax(ratio))
    bottom = int(np.argmin(ratio[: top + 1]))
    if top - bottom < 1:  # pragma: no cover - a table that never rises
        raise ValueError("the ratio does not increase with the size")
    return slice(bottom, top + 1)


@functools.lru_cache(maxsize=8)
def _rain_table(f1, f2, temperature, mu):
    """Table of a rain DFR: ratio, D_m, log10 of Z/N_w at f1 and of LWC/N_w."""
    from ..fundamentals.tmatrix import raindrop_scattering

    d = np.arange(0.02, 8.0001, 0.02)
    dd = 0.02
    sig = []
    for f in (f1, f2):
        drops = raindrop_scattering(d, 299.792458 / f, temperature, ddelt=1e-5)
        sig.append((299.792458 / f) ** 4 / (np.pi**5 * _KW2) * drops.sigma_h.values)
    dm = np.arange(0.2, 4.0001, 0.01)
    nd = _gamma_shape(d, dm, mu)
    z1 = (nd * sig[0]).sum(axis=1) * dd
    z2 = (nd * sig[1]).sum(axis=1) * dd
    lwc = np.pi / 6.0 * 1e-3 * (nd * d**3).sum(axis=1) * dd  # g m-3 for N_w = 1
    ratio = 10.0 * np.log10(z1 / z2)
    keep = _monotonic_segment(ratio)
    return np.vstack([ratio, dm, np.log10(z1), np.log10(lwc)])[:, keep]


def _ice_index(frequency):
    """Refractive index of ice (n + ik) at a frequency in GHz, Warren and Brandt (2008)."""
    wavelength = 299792458.0 / (frequency * 1e9) * 1e6  # um
    if not _ICE_INDEX[0, 0] <= wavelength <= _ICE_INDEX[-1, 0]:
        raise ValueError(
            "the refractive index of ice is tabulated for 4.9 to 230 GHz, "
            f"not {frequency:g} GHz"
        )
    lw = np.log(_ICE_INDEX[:, 0])
    n = np.interp(np.log(wavelength), lw, _ICE_INDEX[:, 1])
    k = np.exp(np.interp(np.log(wavelength), lw, np.log(_ICE_INDEX[:, 2])))
    return complex(n, k)


def _maxwell_garnett(eps_ice, fraction):
    """Effective permittivity of ice inclusions in air (Maxwell Garnett)."""
    return 1.0 + 3.0 * fraction * (eps_ice - 1.0) / (
        eps_ice + 2.0 - fraction * (eps_ice - 1.0)
    )


def _ice_sphere_mass(d, coefficients):
    """Mass of the ice particle of maximum dimension ``d`` (mm) in kg."""
    a, b = coefficients
    sphere = _ICE_DENSITY * np.pi / 6.0 * (d * 1e-3) ** 3
    return np.minimum(a * (d * 1e-3) ** b, sphere)


def _snow_backscatter(d, frequency, coefficients):
    """Backscatter cross sections (mm2) of soft ice spheres of diameter ``d`` (mm)."""
    from ..fundamentals.tmatrix import BACKSCATTER, spheroid_scattering

    eps_ice = _ice_index(frequency) ** 2
    fraction = _ice_sphere_mass(d, coefficients) / (
        _ICE_DENSITY * np.pi / 6.0 * (d * 1e-3) ** 3
    )
    eps = _maxwell_garnett(eps_ice, fraction)
    index = np.sqrt(eps)
    index = np.where(index.imag < 0, index.conjugate(), index)
    wavelength = 299.792458 / frequency
    out = np.empty(d.size)
    orientation = np.array([[0.0, 0.0, 1.0]])
    for i, (di, mi) in enumerate(zip(d, index)):
        _, z, _ = spheroid_scattering(
            [di / 2.0],
            1.0,
            wavelength,
            complex(mi),
            np.array([BACKSCATTER]),
            orientation,
            ddelt=1e-4,
        )
        zb = z[0, 0]
        out[i] = 2.0 * np.pi * (zb[0, 0] - zb[0, 1] - zb[1, 0] + zb[1, 1])
    return out


@functools.lru_cache(maxsize=8)
def _snow_table(f1, f2, mu, coefficients):
    """Table of a snow DFR: ratio, D_m, log10 of Z/IWC at f1."""
    d = np.geomspace(0.05, 40.0, 220)
    sig = []
    for f in (f1, f2):
        sig.append(
            (299.792458 / f) ** 4
            / (np.pi**5 * _KW2)
            * _snow_backscatter(d, f, coefficients)
        )
    weight = np.gradient(d)
    mass = _ice_sphere_mass(d, coefficients)  # kg
    slope = np.geomspace(0.04, 20.0, 400)  # 1/mm
    nd = (d[None, :] ** mu) * np.exp(-slope[:, None] * d[None, :])
    z1 = (nd * sig[0] * weight).sum(axis=1)
    z2 = (nd * sig[1] * weight).sum(axis=1)
    iwc = (nd * mass * 1e3 * weight).sum(
        axis=1
    )  # g m-3 for N_0 = 1 (m-3 mm-1-mu), D in mm
    dm = (nd * d * mass * weight).sum(axis=1) / (nd * mass * weight).sum(axis=1)
    ratio = 10.0 * np.log10(z1 / z2)
    order = np.argsort(dm)
    ratio, dm, z1, iwc = ratio[order], dm[order], z1[order], iwc[order]
    keep = _monotonic_segment(ratio)
    return np.vstack([ratio, dm, np.log10(z1 / iwc)])[:, keep]


# --------------------------------------------------------------------------
# look-up
# --------------------------------------------------------------------------


def _lookup_numpy(x, table, tolerance):
    """NumPy twin of the kernel ``_dfr.lookup``: ``table`` has the ratio as first row."""
    x = np.asarray(x, dtype=np.float64)
    lut_x, lut_y = table[0], table[1:]
    ok = np.isfinite(x) & (x <= lut_x[-1]) & (x >= lut_x[0] - tolerance)
    out = np.full((lut_y.shape[0], x.size), np.nan)
    xs = x[ok]
    for row in range(lut_y.shape[0]):
        out[row, ok] = np.interp(xs, lut_x, lut_y[row])
    return out


def _lookup(x, table, tolerance, use_compiled, n_threads):
    flat = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
    table = np.ascontiguousarray(table, dtype=np.float64)
    if use_compiled:
        out = _dfr.lookup(
            flat, table[0], table[1:], float(tolerance), int(n_threads or 0)
        )
    else:
        out = _lookup_numpy(flat, table, tolerance)
    return out.reshape((table.shape[0] - 1,) + np.shape(x))


# --------------------------------------------------------------------------
# the ratio
# --------------------------------------------------------------------------


def _shared_dims(a, b):
    return [d for d in a.dims if d in b.dims]


def _match(a, b, match, tolerance):
    """The second field on the coordinates of the first, in the way of ``match``."""
    if match == "exact":
        return xr.align(a, b, join="inner")
    dims = [d for d in _shared_dims(a, b) if d in a.coords and d in b.coords]
    if match == "interp":
        return a, b.interp({d: a[d] for d in dims})
    tolerance = tolerance or {}
    for d in dims:
        b = b.reindex({d: a[d]}, method="nearest", tolerance=tolerance.get(d))
    return a, b


def _liquid_k2(frequency, celsius):
    """Dielectric factor |K_w|^2 of liquid water at a frequency (GHz), Ray (1972)."""
    from ..fundamentals.tmatrix import water_refractive_index

    m = water_refractive_index(299.792458 / frequency, celsius)
    k = (m * m - 1.0) / (m * m + 2.0)
    return float(abs(k) ** 2)


@provenance(
    "Dual-frequency ratio of the reflectivities of two radars, corrected for path attenuation"
)
def dual_frequency_ratio(
    dbz_ka,
    dbz_w,
    *,
    pia_ka=None,
    pia_w=None,
    match="exact",
    tolerance=None,
    calibration="kw2",
    frequencies=None,
    temperature=0.0,
):
    """
    Dual-frequency ratio in dB of the reflectivities of two radars.

    ``DFR = (dbz_ka + pia_ka) - (dbz_w + pia_w)``: the reflectivity at the lower
    frequency (Ka band) minus the reflectivity at the higher one (W band), each
    corrected for its two-way path attenuation
    (:mod:`radarx.retrieve.cloud_attenuation`), in dB.

    Parameters
    ----------
    dbz_ka, dbz_w : xarray.DataArray
        The reflectivity (dBZ) at the lower and the higher frequency, with
        coordinates to match them (``time``, ``range`` or ``height``).
    pia_ka, pia_w : float or xarray.DataArray, optional
        Two-way path-integrated attenuation (dB, positive) at the two
        frequencies, for example ``PIA_GAS + PIA_CLOUD_LIQUID``. Default: none.
    match : {"exact", "nearest", "interp"}, optional
        How the fields are put on the same coordinates. ``"exact"`` (default)
        keeps the coordinate values the two fields share; ``"nearest"`` takes
        the nearest value of ``dbz_w`` for every coordinate value of
        ``dbz_ka`` (within ``tolerance``); ``"interp"`` interpolates ``dbz_w``
        linearly to the coordinates of ``dbz_ka``.
    tolerance : dict, optional
        For ``"nearest"``: the largest distance per dimension (for time
        a ``numpy.timedelta64``), as a dict ``{dim: value}``. Default: none.
    calibration : {"kw2", "liquid"}, optional
        The convention of the reflectivities. ``"kw2"`` (default): the equivalent
        reflectivity factor with the dielectric factor of water
        :math:`|K_w|^2 = 0.93` at both frequencies, as the retrievals of this
        module assume. ``"liquid"``: calibrated so that cloud droplets of the same
        size distribution give the same dBZ at both frequencies, with
        :math:`|K_w(f)|^2` of liquid water at each frequency, the convention of
        Cloudnet (and CloudnetPy). The ratio is then converted to ``"kw2"``,
        which adds :math:`10 \\log_{10} (|K_w(f_{Ka})|^2 / |K_w(f_W)|^2)` (1.1 dB
        for 35 and 94 GHz at 0 degrees Celsius), so that ice particles in the Rayleigh regime
        have a ratio of zero. Needs ``frequencies``.
    frequencies : tuple of float, optional
        The lower and the higher frequency in GHz, for ``calibration="liquid"``.
    temperature : float, optional
        Temperature of the water in degrees Celsius for its dielectric factor,
        for ``calibration="liquid"``. Default 0.

    Returns
    -------
    xarray.DataArray
        ``DFR`` (dB) on the matched coordinates; NaN where either field is missing.

    Raises
    ------
    ValueError
        For an unknown ``match`` or ``calibration``, or ``calibration="liquid"``
        without two increasing ``frequencies`` (GHz).
    TypeError
        If a reflectivity is not a DataArray.

    Notes
    -----
    A ratio below zero is possible in noise and for calibration offsets of a
    few tenths of a dB; the retrievals of :mod:`radarx.retrieve.dual_frequency`
    accept values down to ``tolerance`` below the smallest tabulated ratio.

    References
    ----------
    Hogan, R. J., A. J. Illingworth, and H. Sauvageot, 2000: Measuring crystal
    size in cirrus using 35- and 94-GHz radars. *J. Atmos. Oceanic Technol.*,
    **17**, 27-37,
    https://doi.org/10.1175/1520-0426(2000)017<0027:MCSICU>2.0.CO;2

    Tukiainen, S., E. O'Connor, and A. Korpinen, 2020: CloudnetPy: A Python
    package for processing cloud remote sensing data. *J. Open Source Softw.*,
    **5**, 2123, https://doi.org/10.21105/joss.02123

    Examples
    --------
    >>> dfr = dual_frequency_ratio(ka.DBZ, w.DBZ, pia_ka=pia_ka.PIA_GAS, pia_w=pia_w.PIA_GAS,
    ...                            match="nearest")  # doctest: +SKIP
    """
    if match not in _MATCH:
        raise ValueError(f"match must be one of {_MATCH}, not {match!r}")
    if calibration not in ("kw2", "liquid"):
        raise ValueError(f"calibration must be 'kw2' or 'liquid', not {calibration!r}")
    offset = 0.0
    if calibration == "liquid":
        if frequencies is None:
            raise ValueError("calibration='liquid' needs the frequencies")
        f1, f2 = _check_frequencies(frequencies)
        offset = 10.0 * np.log10(
            _liquid_k2(f1, float(temperature)) / _liquid_k2(f2, float(temperature))
        )
    for name, field in (("dbz_ka", dbz_ka), ("dbz_w", dbz_w)):
        if not isinstance(field, xr.DataArray):
            raise TypeError(f"{name} must be an xarray.DataArray")
    a, b = _match(dbz_ka, dbz_w, match, tolerance)
    if pia_ka is not None:
        a = a + pia_ka
    if pia_w is not None:
        b = b + pia_w
    out = (a - b + offset).rename("DFR")
    out = as_field(out)
    out.attrs = {
        "long_name": "Dual-frequency ratio of the reflectivities (lower minus higher frequency)",
        "units": "dB",
    }
    return out


# --------------------------------------------------------------------------
# the retrievals
# --------------------------------------------------------------------------


def _retrieve(dfr, dbz, table, tolerance, names, finish, use_compiled, n_threads):
    """Apply the table to the ratio and the reflectivity; ``finish`` makes the products.

    ``finish(rows, dbz, clipped)`` gets the interpolated rows of the table, the
    reflectivity and where the ratio is below the table (and was set to its first size).
    """
    if not isinstance(dfr, xr.DataArray):
        raise TypeError("dfr must be an xarray.DataArray")
    if not isinstance(dbz, xr.DataArray):
        raise TypeError("dbz must be an xarray.DataArray")
    if tolerance < 0:
        raise ValueError(f"tolerance must not be negative, not {tolerance!r}")

    def block(ratio, reflectivity):
        ratio, reflectivity = np.broadcast_arrays(ratio, reflectivity)
        rows = _lookup(ratio, table, tolerance, use_compiled, n_threads)
        return tuple(
            np.asarray(v, dtype=np.float64)
            for v in finish(rows, reflectivity.astype(np.float64), ratio < table[0, 0])
        )

    outputs = xr.apply_ufunc(
        block,
        *detach(dfr, dbz),
        output_core_dims=[[]] * len(names),
        dask="parallelized",
        output_dtypes=[np.float64] * len(names),
    )
    return outputs


@provenance(
    "Rain drop size from the Ka-W dual-frequency ratio with T-matrix scattering of "
    "gamma drop size distributions, in the way of Tridon and Battaglia (2015)"
)
def dfr_rain(
    dfr,
    dbz,
    *,
    frequencies=(35.0, 94.0),
    temperature=10.0,
    mu=3.0,
    tolerance=1.0,
    n_threads=None,
    engine="auto",
):
    """
    Drop size and water content of rain from the dual-frequency ratio.

    Looks up the mass-weighted mean diameter :math:`D_m` in the table of the
    ratio against :math:`D_m` of normalized gamma distributions of shape ``mu``
    (see :mod:`radarx.retrieve.dual_frequency`), and gets :math:`N_w` and the
    liquid water content from the reflectivity of the lower frequency.

    Parameters
    ----------
    dfr : xarray.DataArray
        Dual-frequency ratio in dB of the attenuation-corrected reflectivities
        (:func:`dual_frequency_ratio`).
    dbz : xarray.DataArray
        Attenuation-corrected reflectivity (dBZ) at the lower frequency, on the
        same gates.
    frequencies : tuple of float, optional
        The lower and the higher frequency in GHz of the ratio. Default
        ``(35.0, 94.0)``.
    temperature : float, optional
        Temperature of the water in degrees Celsius, for its refractive index. Default 10.
    mu : float, optional
        Shape parameter of the gamma distribution. Default 3, a radarx choice for
        the middle of the range of observed rain (0-6). The ratio of
        a given :math:`D_m` changes by about 0.5-2 dB over this range at
        :math:`D_m` above 1 mm.
    tolerance : float, optional
        How far (dB) below the smallest tabulated ratio a ratio is still
        accepted and set to the smallest size (without ``NW`` and ``LWC``). Default 1.
    n_threads : int, optional
        Threads of the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation. ``"auto"`` (default) prefers the compiled kernel.

    Returns
    -------
    xarray.Dataset
        ``DM`` (mm), ``NW`` (m-3 mm-1) and ``LWC`` (g m-3) on the gates of the
        input, NaN where the ratio is outside the table (see
        :mod:`radarx.retrieve.dual_frequency`).

    Raises
    ------
    ValueError
        For frequencies that are not two increasing values, a negative
        ``tolerance`` or an unknown ``engine``.
    TypeError
        If an input is not a DataArray.
    ImportError
        If ``engine="compiled"`` and the kernel is not available, or the T-matrix
        kernel is not.

    Notes
    -----
    Rain at W band attenuates by several dB per kilometre and the reflectivity of
    drops larger than 1.5 mm saturates, so the ratio needs an attenuation correction
    that is good to a fraction of a dB over the path. Without it the size is
    overestimated.

    References
    ----------
    Tridon, F., and A. Battaglia, 2015: Dual-frequency radar Doppler spectral
    retrieval of rain drop size distributions and entangled dynamics variables.
    *J. Geophys. Res. Atmos.*, **120**, 5585-5601,
    https://doi.org/10.1002/2014JD023023

    Brandes, E. A., G. Zhang, and J. Vivekanandan, 2002: Experiments in rainfall
    estimation with a polarimetric radar in a subtropical environment. *J. Appl.
    Meteor.*, **41**, 674-685,
    https://doi.org/10.1175/1520-0450(2002)041<0674:EIREWA>2.0.CO;2

    Examples
    --------
    >>> rain = dfr_rain(dfr, ka_corrected, frequencies=(35.5, 94.0))  # doctest: +SKIP
    >>> rain.DM  # doctest: +SKIP
    """
    use_compiled = choose_engine(engine, HAS_COMPILED_KERNEL, "dual-frequency ratio")
    f1, f2 = _check_frequencies(frequencies)
    table = _rain_table(f1, f2, float(temperature), float(mu))

    def finish(rows, z, clipped):
        dm, log_g1, log_lwc = rows
        with np.errstate(invalid="ignore", over="ignore"):
            nw = np.where(clipped, np.nan, 10.0 ** (z / 10.0 - log_g1))
            lwc = nw * 10.0**log_lwc
        return dm, nw, lwc

    dm, nw, lwc = _retrieve(
        dfr, dbz, table, tolerance, ("DM", "NW", "LWC"), finish, use_compiled, n_threads
    )
    out = xr.Dataset({"DM": dm, "NW": nw, "LWC": lwc})
    meta = {
        "DM": ("Mass-weighted mean drop diameter", "mm"),
        "NW": (
            "Normalized intercept parameter of the drop size distribution",
            "m-3 mm-1",
        ),
        "LWC": ("Liquid water content", "g m-3"),
    }
    for name, (text, unit) in meta.items():
        out[name] = as_field(out[name])
        out[name].attrs = {"long_name": text, "units": unit}
    out.attrs = {
        "frequencies_GHz": (f1, f2),
        "mu": float(mu),
        "temperature_celsius": float(temperature),
    }
    return out


@provenance(
    "Snow and ice particle size from the Ka-W dual-frequency ratio with Mie scattering by "
    "soft ice spheres, in the way of Hogan et al. (2000) and Matrosov (1998)"
)
def dfr_snow(
    dfr,
    dbz,
    *,
    frequencies=(35.0, 94.0),
    mu=0.0,
    mass_size=_BROWN_FRANCIS,
    tolerance=1.0,
    n_threads=None,
    engine="auto",
):
    """
    Particle size and ice water content of snow and ice cloud from the dual-frequency ratio.

    Looks up the mass-weighted mean maximum dimension :math:`D_m` in the table of
    the ratio against :math:`D_m` of gamma size distributions (``mu``) of soft
    ice spheres, and gets the ice water content from the reflectivity of the
    lower frequency (see :mod:`radarx.retrieve.dual_frequency`).

    Parameters
    ----------
    dfr : xarray.DataArray
        Dual-frequency ratio in dB of the attenuation-corrected reflectivities
        (:func:`dual_frequency_ratio`).
    dbz : xarray.DataArray
        Attenuation-corrected reflectivity (dBZ) at the lower frequency, on the
        same gates.
    frequencies : tuple of float, optional
        The lower and the higher frequency in GHz (4.9-230). Default ``(35.0, 94.0)``.
    mu : float, optional
        Shape parameter of the gamma distribution in the maximum dimension;
        0 (default) is the exponential distribution.
    mass_size : tuple of float, optional
        Coefficients ``(a, b)`` of the mass-size relation ``m = a D**b`` (kg, D in m).
        Default the relation of Brown and Francis (1995), ``(0.0185, 1.9)``.
    tolerance : float, optional
        How far (dB) below the smallest tabulated ratio a ratio is still
        accepted and set to the smallest size (without ``IWC``). Default 1.
    n_threads : int, optional
        Threads of the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation. ``"auto"`` (default) prefers the compiled kernel.

    Returns
    -------
    xarray.Dataset
        ``DM`` (mm) and ``IWC`` (g m-3) on the gates of the input, NaN where the ratio
        is outside the table.

    Raises
    ------
    ValueError
        For frequencies that are not two increasing values (4.9-230 GHz), a
        negative ``tolerance`` or an unknown ``engine``.
    TypeError
        If an input is not a DataArray.
    ImportError
        If ``engine="compiled"`` and the kernel is not available, or the T-matrix
        kernel is not.

    Notes
    -----
    Soft spheres scatter more strongly at the resonances than the aggregates
    they stand for, and the ratio of a sphere grows faster with the size than that
    of a snowflake of the same mass and maximum dimension (Kneifel et al. 2015
    discuss the dependence of the dual-wavelength ratio on the particle shape and
    structure). The size is therefore a lower bound of what a nonspherical
    model gives for the same ratio. The table saturates at a ratio of about 15 dB.

    References
    ----------
    Hogan, R. J., A. J. Illingworth, and H. Sauvageot, 2000: Measuring crystal
    size in cirrus using 35- and 94-GHz radars. *J. Atmos. Oceanic Technol.*,
    **17**, 27-37,
    https://doi.org/10.1175/1520-0426(2000)017<0027:MCSICU>2.0.CO;2

    Matrosov, S. Y., 1998: A dual-wavelength radar method to measure snowfall
    rate. *J. Appl. Meteor.*, **37**, 1510-1521,
    https://doi.org/10.1175/1520-0450(1998)037<1510:ADWRMT>2.0.CO;2

    Brown, P. R. A., and P. N. Francis, 1995: Improved measurements of the ice
    water content in cirrus using a total-water probe. *J. Atmos. Oceanic
    Technol.*, **12**, 410-414,
    https://doi.org/10.1175/1520-0426(1995)012<0410:IMOTIW>2.0.CO;2

    Warren, S. G., and R. E. Brandt, 2008: Optical constants of ice from the
    ultraviolet to the microwave: A revised compilation. *J. Geophys. Res.*,
    **113**, D14220, https://doi.org/10.1029/2007JD009744

    Kneifel, S., A. von Lerber, J. Tiira, D. Moisseev, P. Kollias, and J.
    Leinonen, 2015: Observed relations between snowfall microphysics and
    triple-frequency radar measurements. *J. Geophys. Res. Atmos.*, **120**,
    6034-6055, https://doi.org/10.1002/2015JD023156

    Examples
    --------
    >>> snow = dfr_snow(dfr, ka_corrected)  # doctest: +SKIP
    >>> snow.IWC  # doctest: +SKIP
    """
    use_compiled = choose_engine(engine, HAS_COMPILED_KERNEL, "dual-frequency ratio")
    f1, f2 = _check_frequencies(frequencies, low=4.9, high=230.0)
    try:
        coefficients = (float(mass_size[0]), float(mass_size[1]))
    except (TypeError, ValueError, IndexError):
        raise ValueError("mass_size must be a pair (a, b)") from None
    if not (coefficients[0] > 0 and 1.0 < coefficients[1] < 3.0):
        raise ValueError("mass_size needs a > 0 and 1 < b < 3")
    table = _snow_table(f1, f2, float(mu), coefficients)

    def finish(rows, z, clipped):
        dm, log_ratio = rows
        with np.errstate(invalid="ignore", over="ignore"):
            iwc = np.where(clipped, np.nan, 10.0 ** (z / 10.0 - log_ratio))
        return dm, iwc

    dm, iwc = _retrieve(
        dfr, dbz, table, tolerance, ("DM", "IWC"), finish, use_compiled, n_threads
    )
    out = xr.Dataset({"DM": as_field(dm), "IWC": as_field(iwc)})
    out["DM"].attrs = {
        "long_name": "Mass-weighted mean maximum dimension of the ice particles",
        "units": "mm",
    }
    out["IWC"].attrs = {
        "standard_name": "mass_concentration_of_cloud_ice_in_air",
        "long_name": "Ice water content",
        "units": "g m-3",
    }
    out.attrs = {
        "frequencies_GHz": (f1, f2),
        "mu": float(mu),
        "mass_size": coefficients,
    }
    return out


# --------------------------------------------------------------------------
# accessors
# --------------------------------------------------------------------------


@accessor_method("dataarray", name="dfr_rain", wraps=dfr_rain)
def _dfr_rain_accessor(self):
    """
    Drop size and water content of rain from the dual-frequency ratio.

    See :func:`radarx.retrieve.dfr_rain` for the parameters.

    Returns
    -------
    xarray.Dataset
        ``DM``, ``NW`` and ``LWC``.
    """


@accessor_method("dataarray", name="dfr_snow", wraps=dfr_snow)
def _dfr_snow_accessor(self):
    """
    Particle size and ice water content of snow from the dual-frequency ratio.

    See :func:`radarx.retrieve.dfr_snow` for the parameters.

    Returns
    -------
    xarray.Dataset
        ``DM`` and ``IWC``.
    """
