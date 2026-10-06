#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Drop Size Distribution Retrieval
================================

Retrieve the parameters of a gamma raindrop size distribution (DSD)

.. math::

    N(D) = N_0 D^{\\mu} \\exp(-\\Lambda D)

(Ulbrich 1983; :math:`N` in m\\ :sup:`-3` mm\\ :sup:`-1`, :math:`D` in mm)
from polarimetric radar variables, and derived quantities, for sweeps,
volumes, grids or QVPs.

Methods
-------
``"constrained"`` (default)
    Constrained-gamma DSD of Zhang et al. (2001): the shape :math:`\\mu` is
    tied to the slope :math:`\\Lambda` by an empirical relation
    :math:`\\mu = c_2 \\Lambda^2 + c_1 \\Lambda + c_0`, either that of Cao et
    al. (2008, Oklahoma 2D video disdrometer data, the default) or that of
    Zhang et al. (2001, Florida), or one given by the user. Because
    :math:`Z_{DR}` does not depend on :math:`N_0`, it fixes :math:`\\Lambda`
    (and :math:`\\mu`); :math:`N_0` then follows from :math:`Z_H`
    (Zhang et al. 2001; Vivekanandan et al. 2004).
``"normalized"``
    Normalized gamma DSD (Testud et al. 2001; Bringi et al. 2002)

    .. math::

        N(D) = N_w f(\\mu) \\left(\\frac{D}{D_m}\\right)^{\\mu}
        \\exp\\left(-(4 + \\mu) \\frac{D}{D_m}\\right),\\quad
        f(\\mu) = \\frac{6}{4^4} \\frac{(4 + \\mu)^{\\mu + 4}}{\\Gamma(\\mu + 4)},

    where :math:`D_m` is the mass-weighted mean diameter and :math:`N_w` the
    intercept of the exponential DSD with the same water content and
    :math:`D_m`. For a given shape :math:`\\mu` (``mu``, default 3),
    :math:`Z_{DR}` depends on :math:`D_m` only, which it fixes, and
    :math:`N_w` follows from :math:`Z_H`.

    The shape cannot be retrieved from :math:`K_{DP}` in addition: at a fixed
    :math:`Z_{DR}`, :math:`K_{DP}/Z_H` (which does not depend on the
    intercept either) changes by less than 0.03 in :math:`\\log_{10}` (7 %)
    for :math:`\\mu` from -0.5 to 10 at S band, and by less than 0.15 at C
    and X band (for :math:`Z_{DR}` of 0.5-3 dB), well within the noise of
    :math:`K_{DP}`.

With :math:`K_{DP}` (``kdp``), both methods take the intercept from
:math:`K_{DP}` instead of :math:`Z_H` where :math:`K_{DP}` is at least
``kdp_min``: for the DSD shape fixed by :math:`Z_{DR}`, :math:`K_{DP}` is
proportional to the intercept, so the intercept (and the rain rate and water
content) is then immune to the calibration of :math:`Z_H`, attenuation and
partial beam blockage. Elsewhere it comes from :math:`Z_H`.

Both methods use lookup tables of :math:`Z_H`, :math:`Z_{DR}` and
:math:`K_{DP}` for the DSD family, computed (and cached) from the tabulated
single-drop scattering properties when first needed, so no gate is fitted
iteratively. Values of :math:`Z_{DR}` outside the range of the table are
clipped to it. Per gate, a compiled C++ kernel (multithreaded over all gates
of all sweeps at once, with an identical NumPy fallback) looks up the DSD
parameters and evaluates the closed-form moments
:math:`M_n = N_0 \\Gamma(\\mu + n + 1) / \\Lambda^{\\mu + n + 1}` of the
untruncated gamma DSD:

- :math:`D_m = M_4 / M_3 = (4 + \\mu)/\\Lambda`;
- the median volume diameter :math:`D_0 \\approx (3.67 + \\mu)/\\Lambda`
  (Ulbrich 1983);
- the liquid water content :math:`W = \\frac{\\pi}{6} \\rho_w M_3`;
- :math:`N_w = \\frac{4^4}{\\pi \\rho_w} \\frac{W}{D_m^4}` (Testud et al. 2001);
- the rain rate :math:`R = 6\\pi \\times 10^{-4} \\int v(D) D^3 N(D)\\,dD`
  (mm h\\ :sup:`-1`) with the fall speed
  :math:`v(D) = 9.65 - 10.3 \\exp(-0.6 D)` m s\\ :sup:`-1` of Atlas et al.
  (1973) at sea level, which also integrates in closed form.

Scattering tables
-----------------
``radarx/retrieve/data/dsd_scattering.csv`` holds single-drop backscatter
cross sections, copolar correlation terms, :math:`K_{DP}` and specific
attenuation at S (2.8 GHz), C (5.6 GHz) and X band (9.4 GHz), for liquid
water at 0, 10, 20 and 30 °C (linearly interpolated in between), for
equal-volume diameters of 0.05-8 mm (drops larger than 8 mm are ignored).
They were computed with the T-matrix method (Mishchenko and Travis 1998,
through the ``pytmatrix`` interface of Leinonen 2014) by
``ci/build_dsd_tables.py``, assuming

- oblate spheroids with the axis ratio of Brandes et al. (2002),
  :math:`b/a = 0.9951 + 0.0251 D - 0.03644 D^2 + 0.005303 D^3 - 0.0002492 D^4`;
- Gaussian canting angles with zero mean and 7° standard deviation (Huang
  et al. 2008);
- horizontal incidence (low elevation angles);
- the refractive index of water of Ray (1972), and
  :math:`|K_w|^2 = 0.93`.

Rain only
---------
The retrieval assumes every gate is rain. Pass ``mask`` (``True`` for rain
gates, e.g. from a hydrometeor classification or a quality-control step) to
leave other gates empty; snow, hail, the melting layer and non-meteorological
echo otherwise give meaningless DSDs. As a safeguard, gates whose :math:`N_w`
falls outside ``nw_range`` (by default :math:`10`-:math:`10^6`
m\\ :sup:`-3` mm\\ :sup:`-1`, about one decade beyond the
:math:`\\log_{10} N_w` of 2-5.5 found in rain of different climates by
Bringi et al. 2003) are left empty too: their :math:`Z_H` and :math:`Z_{DR}`
are not consistent with rain.

Disdrometers
------------
:func:`dsd_spectrum` rebuilds :math:`N(D)` from the retrieved parameters on
disdrometer size bins (the 32 classes of the OTT Parsivel by default, see
:func:`parsivel_bins`), :func:`fit_gamma_moments` fits a gamma DSD to measured
spectra by the method of moments (second, fourth and sixth moments, as
evaluated by Cao and Zhang 2009) and :func:`radar_from_dsd` computes radar
variables from measured or rebuilt spectra, for radar-disdrometer comparisons.

References
----------
Ulbrich, C. W., 1983: Natural variations in the analytical form of the
raindrop size distribution. *J. Climate Appl. Meteor.*, **22** (10),
1764-1775, https://doi.org/10.1175/1520-0450(1983)022<1764:NVITAF>2.0.CO;2

Zhang, G., J. Vivekanandan, and E. Brandes, 2001: A method for estimating
rain rate and drop size distribution from polarimetric radar measurements.
*IEEE Trans. Geosci. Remote Sens.*, **39** (4), 830-841,
https://doi.org/10.1109/36.917906

Testud, J., S. Oury, R. A. Black, P. Amayenc, and X. Dou, 2001: The concept
of "normalized" distribution to describe raindrop spectra: A tool for cloud
physics and cloud remote sensing. *J. Appl. Meteor.*, **40** (6), 1118-1140,
https://doi.org/10.1175/1520-0450(2001)040<1118:TCONDT>2.0.CO;2

Brandes, E. A., G. Zhang, and J. Vivekanandan, 2002: Experiments in rainfall
estimation with a polarimetric radar in a subtropical environment. *J. Appl.
Meteor.*, **41** (6), 674-685,
https://doi.org/10.1175/1520-0450(2002)041<0674:EIREWA>2.0.CO;2

Bringi, V. N., G.-J. Huang, V. Chandrasekar, and E. Gorgucci, 2002: A
methodology for estimating the parameters of a gamma raindrop size
distribution model from polarimetric radar data: Application to a
squall-line event from the TRMM/Brazil campaign. *J. Atmos. Oceanic
Technol.*, **19** (5), 633-645,
https://doi.org/10.1175/1520-0426(2002)019<0633:AMFETP>2.0.CO;2

Bringi, V. N., V. Chandrasekar, J. Hubbert, E. Gorgucci, W. L. Randeu, and
M. Schoenhuber, 2003: Raindrop size distribution in different climatic
regimes from disdrometer and dual-polarized radar analysis. *J. Atmos.
Sci.*, **60** (2), 354-365,
https://doi.org/10.1175/1520-0469(2003)060<0354:RSDIDC>2.0.CO;2

Vivekanandan, J., G. Zhang, and E. Brandes, 2004: Polarimetric radar
estimators based on a constrained gamma drop size distribution model.
*J. Appl. Meteor.*, **43** (2), 217-230,
https://doi.org/10.1175/1520-0450(2004)043<0217:PREBOA>2.0.CO;2

Cao, Q., G. Zhang, E. Brandes, T. Schuur, A. Ryzhkov, and K. Ikeda, 2008:
Analysis of video disdrometer and polarimetric radar data to characterize
rain microphysics in Oklahoma. *J. Appl. Meteor. Climatol.*, **47** (8),
2238-2255, https://doi.org/10.1175/2008JAMC1732.1

Bringi, V. N., C. R. Williams, M. Thurai, and P. T. May, 2009: Using
dual-polarized radar and dual-frequency profiler for DSD characterization: A
case study from Darwin, Australia. *J. Atmos. Oceanic Technol.*, **26** (10),
2107-2122, https://doi.org/10.1175/2009JTECHA1258.1

Cao, Q., and G. Zhang, 2009: Errors in estimating raindrop size distribution
parameters employing disdrometer and simulated raindrop spectra. *J. Appl.
Meteor. Climatol.*, **48** (2), 406-425,
https://doi.org/10.1175/2008JAMC2026.1

Atlas, D., R. C. Srivastava, and R. S. Sekhon, 1973: Doppler radar
characteristics of precipitation at vertical incidence. *Rev. Geophys.*,
**11** (1), 1-35, https://doi.org/10.1029/RG011i001p00001

Mishchenko, M. I., and L. D. Travis, 1998: Capabilities and limitations of a
current FORTRAN implementation of the T-matrix method for randomly oriented,
rotationally symmetric scatterers. *J. Quant. Spectrosc. Radiat. Transfer*,
**60** (3), 309-324, https://doi.org/10.1016/S0022-4073(98)00008-9

Leinonen, J., 2014: High-level interface to T-matrix scattering
calculations: architecture, capabilities and limitations. *Opt. Express*,
**22** (2), 1655-1660, https://doi.org/10.1364/OE.22.001655

Huang, G.-J., V. N. Bringi, and M. Thurai, 2008: Orientation angle
distributions of drops after an 80-m fall using a 2D video disdrometer.
*J. Atmos. Oceanic Technol.*, **25** (9), 1717-1723,
https://doi.org/10.1175/2008JTECHA1075.1

Ray, P. S., 1972: Broadband complex refractive indices of ice and water.
*Appl. Opt.*, **11** (8), 1836-1844, https://doi.org/10.1364/AO.11.001836

Tokay, A., D. B. Wolff, and W. A. Petersen, 2014: Evaluation of the new
version of the laser-optical disdrometer, OTT Parsivel2. *J. Atmos. Oceanic
Technol.*, **31** (6), 1276-1288, https://doi.org/10.1175/JTECH-D-13-00174.1

.. autosummary::
   :nosignatures:
   :toctree: generated/

   dsd
   dsd_spectrum
   fit_gamma_moments
   radar_from_dsd
   scattering_table
   parsivel_bins
"""

from __future__ import annotations

__all__ = [
    "dsd",
    "dsd_spectrum",
    "fit_gamma_moments",
    "radar_from_dsd",
    "scattering_table",
    "parsivel_bins",
]

import functools
import warnings
from importlib import resources

import numpy as np
import xarray as xr
from scipy.special import gamma as _gamma_fn

from .._registry import accessor_method

try:
    from . import _dsd

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _dsd = None
    HAS_COMPILED_KERNEL = False

KW2 = 0.93  # dielectric factor |K_w|^2 of the reflectivity
BANDS = ("S", "C", "X")
MU_LAMBDA = {
    # mu = c2 Lambda^2 + c1 Lambda + c0
    "cao2008": (-0.0201, 0.902, -1.718),
    "zhang2001": (-0.016, 1.213, -1.957),
}
_LAMBDA_MAX = 20.0  # mm-1, upper end of the constrained-gamma table
_N_LAMBDA = 2000
_DM_RANGE = (0.3, 4.0)  # mm, normalized-gamma table
_N_DM = 741
_OUT_NAMES = ("N0", "NW", "D0", "DM", "MU", "LAMBDA", "RAIN_RATE", "LWC")
_OUT_ATTRS = {
    "N0": {
        "long_name": "Intercept parameter of the gamma drop size distribution",
        "units": "m-3 mm-(1+mu)",
    },
    "NW": {
        "long_name": "Normalized intercept parameter of the drop size distribution",
        "units": "m-3 mm-1",
    },
    "D0": {"long_name": "Median volume diameter", "units": "mm"},
    "DM": {"long_name": "Mass-weighted mean diameter", "units": "mm"},
    "MU": {
        "long_name": "Shape parameter of the gamma drop size distribution",
        "units": "1",
    },
    "LAMBDA": {
        "long_name": "Slope parameter of the gamma drop size distribution",
        "units": "mm-1",
    },
    "RAIN_RATE": {
        "standard_name": "rainfall_rate",
        "long_name": "Rain rate",
        "units": "mm h-1",
    },
    "LWC": {
        "standard_name": "mass_concentration_of_liquid_water_in_air",
        "long_name": "Liquid water content",
        "units": "g m-3",
    },
}

# OTT Parsivel size classes [mm]: centres and widths
_PARSIVEL_CENTERS = np.array(
    [0.062, 0.187, 0.312, 0.437, 0.562, 0.687, 0.812, 0.937, 1.062, 1.187]
    + [1.375, 1.625, 1.875, 2.125, 2.375, 2.75, 3.25, 3.75, 4.25, 4.75]
    + [5.5, 6.5, 7.5, 8.5, 9.5, 11.0, 13.0, 15.0, 17.0, 19.0, 21.5, 24.5]
)
_PARSIVEL_WIDTHS = np.array(
    [0.125] * 10 + [0.25] * 5 + [0.5] * 5 + [1.0] * 5 + [2.0] * 5 + [3.0] * 2
)


# --------------------------------------------------------------------------
# scattering tables
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _raw_table():
    """The packaged single-drop scattering table, by (band, temperature)."""
    path = resources.files("radarx.retrieve") / "data" / "dsd_scattering.csv"
    with resources.as_file(path) as p:
        band = np.loadtxt(
            p, delimiter=",", comments="#", skiprows=5, usecols=0, dtype=str
        )
        vals = np.loadtxt(
            p, delimiter=",", comments="#", skiprows=5, usecols=range(1, 11)
        )
    out = {}
    for b in BANDS:
        rows = vals[band == b]
        temps = np.unique(rows[:, 1])
        per_t = [rows[rows[:, 1] == t] for t in temps]
        out[b] = {
            "wavelength": float(rows[0, 0]),
            "temperature": temps,
            "diameter": per_t[0][:, 2],
            "data": np.stack([r[:, 3:] for r in per_t]),  # (temp, diameter, 7)
        }
    return out


_COLUMNS = ("sigma_h", "sigma_v", "copol_re", "copol_im", "kdp", "ah", "av")


def _check_band(band):
    b = str(band).upper()
    if b not in BANDS:
        raise ValueError(f"band must be one of {BANDS}, not {band!r}")
    return b


@functools.lru_cache(maxsize=64)
def _single_drop(band, temperature):
    """Per-diameter scattering arrays at one temperature (linear in T)."""
    tab = _raw_table()[band]
    temps = tab["temperature"]
    t = float(temperature)
    if not temps[0] <= t <= temps[-1]:
        raise ValueError(
            f"temperature must be within {temps[0]:g}-{temps[-1]:g} degrees "
            f"Celsius, not {temperature!r}"
        )
    j = min(int(np.searchsorted(temps, t, side="right")) - 1, temps.size - 2)
    w = (t - temps[j]) / (temps[j + 1] - temps[j])
    data = (1.0 - w) * tab["data"][j] + w * tab["data"][j + 1]
    return tab["wavelength"], tab["diameter"], data


def scattering_table(band="S", temperature=20.0):
    """
    Single-drop scattering properties of raindrops.

    Parameters
    ----------
    band : {"S", "C", "X"}, optional
        Radar band: 2.8, 5.6 or 9.4 GHz. Default ``"S"``.
    temperature : float, optional
        Water temperature in degrees Celsius, 0-30 (linear interpolation
        between the tables at 0, 10, 20 and 30 °C). Default 20.

    Returns
    -------
    xarray.Dataset
        On the equal-volume ``diameter`` (mm): the backscatter cross sections
        ``sigma_h`` and ``sigma_v`` (mm2), the copolar correlation terms
        ``copol_re`` and ``copol_im`` (mm2), and the per-drop contributions
        (for one drop per m3) to ``kdp`` (degrees/km) and to the specific
        attenuation ``ah`` and ``av`` (dB/km), with the drop ``axis_ratio``.
        The assumptions are listed in :mod:`radarx.retrieve.dsd`.

    Examples
    --------
    >>> tab = scattering_table("C", temperature=10.0)  # doctest: +SKIP
    """
    band = _check_band(band)
    wl, diam, data = _single_drop(band, float(temperature))
    units = ("mm2", "mm2", "mm2", "mm2", "degrees/km", "dB/km", "dB/km")
    names = (
        "Horizontal backscatter cross section",
        "Vertical backscatter cross section",
        "Real part of the copolar correlation term",
        "Imaginary part of the copolar correlation term",
        "Specific differential phase per drop per m3",
        "Horizontal specific attenuation per drop per m3",
        "Vertical specific attenuation per drop per m3",
    )
    data_vars = {
        c: ("diameter", data[:, i], {"long_name": n, "units": u})
        for i, (c, n, u) in enumerate(zip(_COLUMNS, names, units))
    }
    ratio = (
        0.9951
        + 0.0251 * diam
        - 0.03644 * diam**2
        + 0.005303 * diam**3
        - 0.0002492 * diam**4
    )
    data_vars["axis_ratio"] = (
        "diameter",
        np.minimum(ratio, 1.0),
        {"long_name": "Drop axis ratio (Brandes et al. 2002)", "units": "1"},
    )
    return xr.Dataset(
        data_vars,
        coords={
            "diameter": (
                "diameter",
                diam,
                {"long_name": "Equal-volume drop diameter", "units": "mm"},
            )
        },
        attrs={
            "band": band,
            "wavelength": wl,
            "wavelength_units": "mm",
            "temperature": float(temperature),
            "temperature_units": "degrees_Celsius",
            "canting": "Gaussian, 0 mean, 7 degrees standard deviation",
            "method": "T-matrix (pytmatrix), built by ci/build_dsd_tables.py",
        },
    )


def _trapezoid_weights(d):
    """Trapezoid-rule weights on the diameter grid."""
    w = np.zeros_like(d)
    step = np.diff(d)
    w[:-1] += 0.5 * step
    w[1:] += 0.5 * step
    return w


def _gamma_integrals(band, temperature, n0, mu, lam):
    """
    Z_H, Z_V [mm6 m-3] and K_DP of gamma DSDs (broadcast arrays) on the table
    grid. Used to build the lookup tables.
    """
    wl, d, data = _single_drop(band, temperature)
    w = _trapezoid_weights(d)
    n0, mu, lam = np.broadcast_arrays(*(np.asarray(a, float) for a in (n0, mu, lam)))
    with np.errstate(over="ignore", under="ignore"):
        nd = n0[..., None] * d ** mu[..., None] * np.exp(-lam[..., None] * d)
    nd = nd * w
    k = wl**4 / (np.pi**5 * KW2)
    zh = k * nd @ data[:, 0]
    zv = k * nd @ data[:, 1]
    kdp = nd @ data[:, 4]
    return zh, zv, kdp


# --------------------------------------------------------------------------
# lookup tables
# --------------------------------------------------------------------------


def _increasing_prefix(x):
    """Length of the strictly increasing leading part of x."""
    bad = np.flatnonzero(~(np.diff(x) > 0))
    return int(bad[0] + 1) if bad.size else x.size


def _f_mu(mu):
    return 6.0 / 4.0**4 * (4.0 + mu) ** (mu + 4.0) / _gamma_fn(mu + 4.0)


_TABLE_KEYS = ("zdr", "logz", "logk", "logn0", "mu", "lam")


@functools.lru_cache(maxsize=32)
def _lookup_table(band, temperature, method, relation, mu):
    """
    Lookup table against strictly increasing ZDR: a dict of 1-D arrays
    (zdr, logz, logk, logn0, mu, lam) for a unit intercept.
    """
    if method == "constrained":
        c2, c1, c0 = relation
        # decreasing Lambda: increasing ZDR
        lam = np.geomspace(0.3, _LAMBDA_MAX, _N_LAMBDA)[::-1]
        m = c2 * lam**2 + c1 * lam + c0
        keep = m > -0.99
        lam, m = lam[keep], m[keep]
        logn0 = np.zeros_like(lam)
    else:
        dm = np.linspace(*_DM_RANGE, _N_DM)
        m = np.full_like(dm, mu)
        lam = (4.0 + mu) / dm
        logn0 = np.log10(_f_mu(mu)) - mu * np.log10(dm)
    zh, zv, kdp = _gamma_integrals(band, temperature, 10.0**logn0, m, lam)
    zdr = 10.0 * np.log10(zh / zv)
    n = _increasing_prefix(zdr)
    if n < 2:
        raise ValueError("the DSD family gives no usable ZDR lookup table")
    cols = {
        "zdr": zdr,
        "logz": np.log10(zh),
        "logk": np.log10(kdp),
        "logn0": logn0,
        "mu": m,
        "lam": lam,
    }
    out = {}
    for k in _TABLE_KEYS:
        out[k] = np.ascontiguousarray(cols[k][:n], dtype=np.float64)
        out[k].flags.writeable = False
    return out


# --------------------------------------------------------------------------
# NumPy reference implementation (same steps and order as the C++ kernel)
# --------------------------------------------------------------------------


def _lookup_numpy(tab, x):
    """Clipped position of x in the table and the interpolated values."""
    z = tab["zdr"]
    n = z.size
    xc = np.minimum(np.maximum(x, z[0]), z[n - 1])
    i = np.searchsorted(z, xc, side="right") - 1
    i = np.minimum(np.maximum(i, 0), n - 2)
    w = (xc - z[i]) / (z[i + 1] - z[i])
    return {
        k: tab[k][i] + w * (tab[k][i + 1] - tab[k][i])
        for k in ("logz", "logk", "logn0", "mu", "lam")
    }


def _moments_numpy(logn0, mu, lam):
    """Closed-form N0, Nw, D0, Dm, mu, Lambda, R, LWC of gamma DSDs."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        n0 = 10.0**logn0
        m3 = n0 * _gamma_fn(mu + 4.0) * lam ** (-(mu + 4.0))
        dm = (mu + 4.0) / lam
        nw = 256.0 / 6.0 * m3 / (dm * dm * dm * dm)
        d0 = (mu + 3.67) / lam
        rate = 6.0e-4 * np.pi * m3 * (9.65 - 10.3 * (lam / (lam + 0.6)) ** (mu + 4.0))
        lwc = np.pi / 6.0 * 1.0e-3 * m3
    return np.stack([n0, nw, d0, dm, mu, lam, rate, lwc])


def _retrieve_numpy(z, zdr, kdp, mask, tab, kdp_min, nw_range=(0.0, np.inf)):
    """NumPy implementation of the compiled kernel (same results)."""
    out = np.full((len(_OUT_NAMES), z.size), np.nan)
    good = np.isfinite(z) & np.isfinite(zdr)
    if mask is not None:
        good &= mask.astype(bool)
    if not good.any():
        return out
    e = _lookup_numpy(tab, zdr[good])
    logn0 = 0.1 * z[good] - e["logz"] + e["logn0"]
    if kdp is not None:
        k = kdp[good]
        with np.errstate(invalid="ignore", divide="ignore"):
            use = np.isfinite(k) & (k >= kdp_min) & (k > 0.0)
            from_kdp = np.log10(np.where(use, k, 1.0)) - e["logk"] + e["logn0"]
        logn0 = np.where(use, from_kdp, logn0)
    values = _moments_numpy(logn0, e["mu"], e["lam"])
    nw = values[1]
    plausible = (nw >= nw_range[0]) & (nw <= nw_range[1])
    out[:, good] = np.where(plausible, values, np.nan)
    return out


def _retrieve_compiled(arrays, tab, kdp_min, nw_range, n_threads):
    return _dsd.retrieve(
        [a[0] for a in arrays],
        [a[1] for a in arrays],
        [a[2] for a in arrays],
        [a[3] for a in arrays],
        *(tab[k] for k in _TABLE_KEYS),
        float(kdp_min),
        float(nw_range[0]),
        float(nw_range[1]),
        int(n_threads or 0),
    )


# --------------------------------------------------------------------------
# xarray layer
# --------------------------------------------------------------------------


def _use_compiled(engine):
    """Whether to run the compiled kernel for the requested ``engine``."""
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled DSD kernel is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


_DBZH_NAMES = ("DBZH", "DBZ", "reflectivity", "corrected_reflectivity")
_ZDR_NAMES = (
    "ZDR",
    "differential_reflectivity",
    "corrected_differential_reflectivity",
)
_KDP_NAMES = ("KDP", "specific_differential_phase")


def _find(ds, name, candidates, required):
    if name is not None:
        if name in ds:
            return name
        if required:
            raise KeyError(f"{name!r} is not in the dataset")
        return None
    for cand in candidates:
        if cand in ds:
            return cand
    if required:
        raise KeyError(f"none of {candidates} found; pass the field name")
    return None


def _band_from_frequency(freq):
    ghz = float(freq) / 1e9
    for band, lo, hi in (("S", 2.0, 4.0), ("C", 4.0, 8.0), ("X", 8.0, 12.5)):
        if lo <= ghz < hi:
            return band
    raise ValueError(
        f"radar frequency {ghz:.2f} GHz is not S, C or X band; pass "
        "band='S', 'C' or 'X' to use one of the tables"
    )


def _infer_band(objs):
    """Band from frequency metadata; S band for WSR-88D volumes."""
    for o in objs:
        for key in ("frequency", "radar_frequency"):
            if key in getattr(o, "variables", {}):
                values = np.atleast_1d(np.asarray(o[key].values, dtype=float))
                values = values[np.isfinite(values)]
                if values.size:
                    return _band_from_frequency(values[0])
            if key in o.attrs:
                return _band_from_frequency(o.attrs[key])
    for o in objs:
        if str(o.attrs.get("scan_name", "")).startswith("VCP"):
            return "S"  # NEXRAD WSR-88D volume coverage pattern
    warnings.warn(
        "the radar band could not be determined from the data; assuming S "
        "band (pass band='S', 'C' or 'X')",
        UserWarning,
        stacklevel=3,
    )
    return "S"


def _relation(mu_lambda):
    if isinstance(mu_lambda, str):
        if mu_lambda not in MU_LAMBDA:
            raise ValueError(
                f"mu_lambda must be one of {sorted(MU_LAMBDA)} or three "
                f"coefficients (c2, c1, c0), not {mu_lambda!r}"
            )
        return mu_lambda, MU_LAMBDA[mu_lambda]
    coeffs = tuple(float(c) for c in mu_lambda)
    if len(coeffs) != 3:
        raise ValueError("mu_lambda needs three coefficients (c2, c1, c0)")
    return "custom", coeffs


def _as_mask(mask, ds, ref):
    """Boolean mask array broadcast to the reflectivity field, or None."""
    if mask is None:
        return None
    if isinstance(mask, str):
        if mask not in ds:
            raise KeyError(f"mask {mask!r} is not in the dataset")
        mask = ds[mask]
    if not isinstance(mask, xr.DataArray):
        raise TypeError("mask must be a variable name or a boolean DataArray")
    mask = mask.fillna(0).astype(bool)
    _, mask = xr.broadcast(ref, mask)
    return np.ascontiguousarray(mask.transpose(*ref.dims).values.ravel())


def _prepare(ds, fields, mask, kdp_da):
    """Contiguous 1-D arrays of one dataset and the reference field."""
    dbzh, zdr = fields
    zname = _find(ds, dbzh, _DBZH_NAMES, True)
    dname = _find(ds, zdr, _ZDR_NAMES, True)
    ref = ds[zname]

    def flat(da):
        _, da = xr.broadcast(ref, da)
        values = da.transpose(*ref.dims).values.ravel()
        return np.ascontiguousarray(values, dtype=np.float64)

    arrays = (
        flat(ref),
        flat(ds[dname]),
        None if kdp_da is None else flat(kdp_da),
        _as_mask(mask, ds, ref),
    )
    kname = None if kdp_da is None else (kdp_da.name or "KDP")
    return {"ref": ref, "fields": (zname, dname, kname), "arrays": arrays}


def _wrap(item, values, attrs):
    """Output Dataset on the coordinates of the reflectivity field."""
    ref = item["ref"]
    data_vars = {}
    for i, name in enumerate(_OUT_NAMES):
        data_vars[name] = (
            ref.dims,
            values[i].reshape(ref.shape),
            dict(_OUT_ATTRS[name]),
        )
    src = ", ".join(f for f in item["fields"] if f)
    out = xr.Dataset(data_vars, coords=dict(ref.coords), attrs=dict(attrs))
    out.attrs["source_fields"] = src
    return out


def _run(items, tab, kdp_min, nw_range, attrs, n_threads, use_compiled):
    if use_compiled:
        arrays = [it["arrays"] for it in items]
        results = _retrieve_compiled(arrays, tab, kdp_min, nw_range, n_threads)
    else:
        results = [
            _retrieve_numpy(*it["arrays"], tab, kdp_min, nw_range) for it in items
        ]
    return [_wrap(it, res, attrs) for it, res in zip(items, results)]


def _kdp_for(ds, kdp, estimated):
    """KDP DataArray for one dataset, or None."""
    if kdp is None:
        return None
    if isinstance(kdp, xr.DataArray):
        return kdp
    if isinstance(kdp, str) and kdp == "estimate":
        return None if estimated is None else estimated
    name = _find(ds, kdp, _KDP_NAMES, False)
    return None if name is None else ds[name]


def dsd(
    obj,
    method="constrained",
    *,
    dbzh=None,
    zdr=None,
    kdp=None,
    mask=None,
    band=None,
    temperature=20.0,
    mu_lambda="cao2008",
    mu=3.0,
    kdp_min=1.0,
    nw_range=(1.0e1, 1.0e6),
    n_threads=None,
    engine="auto",
):
    """
    Retrieve gamma raindrop size distribution parameters from radar data.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataTree
        A sweep, a grid, a QVP or any dataset holding reflectivity and
        differential reflectivity on the same coordinates, or a volume with
        ``sweep_*`` groups (sweeps without both fields are skipped).
    method : {"constrained", "normalized"}, optional
        ``"constrained"`` (default): constrained-gamma DSD from :math:`Z_H`
        and :math:`Z_{DR}` (Zhang et al. 2001; Cao et al. 2008).
        ``"normalized"``: normalized gamma DSD (Testud et al. 2001; Bringi
        et al. 2002) with the shape ``mu``. See :mod:`radarx.retrieve.dsd`.
    dbzh, zdr : str, optional
        Names of the reflectivity (dBZ) and differential reflectivity (dB).
        By default the first found of ``DBZH``, ``DBZ``, ``reflectivity``,
        ``corrected_reflectivity`` and of ``ZDR``,
        ``differential_reflectivity``, ``corrected_differential_reflectivity``.
        Both should be calibrated and corrected for attenuation.
    kdp : str, xarray.DataArray, xarray.DataTree or "estimate", optional
        Specific differential phase (degrees/km): the name of a field
        (``KDP`` or ``specific_differential_phase``), a DataArray (a DataTree
        with a ``KDP`` per sweep for a volume), or ``"estimate"`` to compute
        it with :func:`radarx.retrieve.estimate_kdp`. Where it is at least
        ``kdp_min``, the intercept is taken from :math:`K_{DP}` instead of
        :math:`Z_H`. Default: not used.
    mask : str, xarray.DataArray or xarray.DataTree, optional
        Rain gates (``True``), e.g. from a hydrometeor classification; the
        other gates are left empty (NaN). A variable name, a DataArray that
        broadcasts against the reflectivity, or, for a volume, a DataTree
        whose sweep nodes hold one boolean variable each. Default: all gates.
    band : {"S", "C", "X"}, optional
        Radar band of the scattering tables. By default it is found from the
        ``frequency`` metadata; WSR-88D volumes are S band; otherwise S band
        is assumed with a warning.
    temperature : float, optional
        Rain temperature in degrees Celsius (0-30). Default 20.
    mu_lambda : {"cao2008", "zhang2001"} or (float, float, float), optional
        ``"constrained"``: the relation :math:`\\mu = c_2 \\Lambda^2 + c_1
        \\Lambda + c_0`; ``"cao2008"`` (default) is
        :math:`-0.0201 \\Lambda^2 + 0.902 \\Lambda - 1.718` (Cao et al.
        2008), ``"zhang2001"`` is :math:`-0.016 \\Lambda^2 + 1.213 \\Lambda -
        1.957` (Zhang et al. 2001), or give ``(c2, c1, c0)``. The table
        covers :math:`\\Lambda` up to 20 mm-1 where :math:`\\mu > -0.99`.
    mu : float, optional
        ``"normalized"``: shape parameter (> -1). Default 3.
    kdp_min : float, optional
        Minimum :math:`K_{DP}` (degrees/km) for taking the intercept from
        :math:`K_{DP}`; below it :math:`K_{DP}` is too noisy and
        :math:`Z_H` is used. Default 1.
    nw_range : (float, float) or None, optional
        Plausible range of :math:`N_w` (m-3 mm-1) in rain. Gates outside it
        are left empty (NaN): their :math:`Z_H` and :math:`Z_{DR}` are not
        consistent with rain (e.g. hail, with high :math:`Z_H` and
        :math:`Z_{DR}` near 0 dB, reads as an enormous number of tiny drops).
        Default ``(1e1, 1e6)``, about one decade beyond the
        :math:`\\log_{10} N_w` of 2-5.5 found in rain by Bringi et al.
        (2003); ``None`` keeps all gates.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled
        kernel and falls back to NumPy.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        On the coordinates of the reflectivity: ``N0`` (m-3 mm-(1+mu)),
        ``NW`` (m-3 mm-1), ``D0`` and ``DM`` (mm), ``MU``, ``LAMBDA``
        (mm-1), ``RAIN_RATE`` (mm/h) and ``LWC`` (g/m3); NaN where an input is
        missing or the mask is ``False``. For a volume, a DataTree with one
        such node per sweep and the root of the input. The options used are
        stored in the attributes.

    Raises
    ------
    KeyError
        If a requested field is missing, or no sweep has the fields.
    ValueError
        For unknown options.
    ImportError
        If ``engine="compiled"`` and the compiled kernel is not available.

    References
    ----------
    Zhang, G., J. Vivekanandan, and E. Brandes, 2001: A method for
    estimating rain rate and drop size distribution from polarimetric radar
    measurements. *IEEE Trans. Geosci. Remote Sens.*, **39** (4), 830-841,
    https://doi.org/10.1109/36.917906

    Testud, J., S. Oury, R. A. Black, P. Amayenc, and X. Dou, 2001: The
    concept of "normalized" distribution to describe raindrop spectra: A tool
    for cloud physics and cloud remote sensing. *J. Appl. Meteor.*, **40**
    (6), 1118-1140,
    https://doi.org/10.1175/1520-0450(2001)040<1118:TCONDT>2.0.CO;2

    Bringi, V. N., G.-J. Huang, V. Chandrasekar, and E. Gorgucci, 2002: A
    methodology for estimating the parameters of a gamma raindrop size
    distribution model from polarimetric radar data: Application to a
    squall-line event from the TRMM/Brazil campaign. *J. Atmos. Oceanic
    Technol.*, **19** (5), 633-645,
    https://doi.org/10.1175/1520-0426(2002)019<0633:AMFETP>2.0.CO;2

    Cao, Q., G. Zhang, E. Brandes, T. Schuur, A. Ryzhkov, and K. Ikeda,
    2008: Analysis of video disdrometer and polarimetric radar data to
    characterize rain microphysics in Oklahoma. *J. Appl. Meteor.
    Climatol.*, **47** (8), 2238-2255, https://doi.org/10.1175/2008JAMC1732.1

    Examples
    --------
    >>> out = radarx.retrieve.dsd(dtree["sweep_0"].ds)  # doctest: +SKIP
    >>> out = dtree.radarx.dsd(method="normalized", kdp="estimate")  # doctest: +SKIP
    """
    if method not in ("constrained", "normalized"):
        raise ValueError(
            f"method must be 'constrained' or 'normalized', not {method!r}"
        )
    if method == "normalized" and not float(mu) > -1.0:
        raise ValueError(f"mu must be larger than -1, not {mu!r}")
    use_compiled = _use_compiled(engine)
    is_tree = isinstance(obj, xr.DataTree)
    if band is None:
        objs = [obj] if not is_tree else [obj.ds]
        if is_tree:
            objs += [obj[c].ds for c in ("radar_parameters",) if c in obj.children] + [
                obj.root.ds
            ]
        band = _infer_band(objs)
    band = _check_band(band)
    temperature = float(temperature)
    relation_name, relation = _relation(mu_lambda)
    tab = _lookup_table(
        band,
        temperature,
        method,
        relation if method == "constrained" else None,
        float(mu) if method == "normalized" else None,
    )
    kdp_min = float(kdp_min)
    if nw_range is None:
        nw_range = (0.0, np.inf)
    nw_range = (float(nw_range[0]), float(nw_range[1]))
    attrs = {
        "method": method,
        "band": band,
        "temperature": temperature,
        "comment": (
            "gamma drop size distribution retrieved by radarx.retrieve.dsd; "
            "ZDR outside the lookup table is clipped to it"
        ),
    }
    if method == "constrained":
        attrs["mu_lambda"] = (
            f"{relation_name}: {relation[0]:g} L^2 + {relation[1]:g} L + {relation[2]:g}"
        )
    else:
        attrs["mu"] = float(mu)
    attrs["nw_range"] = list(nw_range) if np.isfinite(nw_range[1]) else "none"
    if kdp is not None:
        attrs["kdp_min"] = kdp_min
        attrs["comment"] += (
            f"; intercept from KDP where KDP >= {kdp_min:g} degrees/km, "
            "otherwise from DBZH"
        )
    fields = (dbzh, zdr)

    if not is_tree:
        est = None
        if isinstance(kdp, str) and kdp == "estimate":
            from .kdp import estimate_kdp

            est = estimate_kdp(obj)["KDP"]
        kdp_da = _kdp_for(obj, kdp, est)
        if isinstance(kdp, str) and kdp != "estimate" and kdp_da is None:
            raise KeyError(f"{kdp!r} is not in the dataset")
        item = _prepare(obj, fields, mask, kdp_da)
        return _run([item], tab, kdp_min, nw_range, attrs, n_threads, use_compiled)[0]

    names = [
        name
        for name in obj.children
        if name.startswith("sweep")
        and _find(obj[name].ds, dbzh, _DBZH_NAMES, False) is not None
        and _find(obj[name].ds, zdr, _ZDR_NAMES, False) is not None
    ]
    if not names:
        raise KeyError("no sweep contains both reflectivity and ZDR")
    est_tree = None
    if isinstance(kdp, str) and kdp == "estimate":
        from .kdp import estimate_kdp

        est_tree = estimate_kdp(obj)
    items = []
    for name in names:
        ds = obj[name].to_dataset(inherit=False)
        m = mask
        if isinstance(mask, (xr.DataTree, dict)):
            node = mask[name] if name in mask else None
            if node is None:
                m = None
            elif isinstance(node, xr.DataArray):
                m = node
            else:
                node = node.to_dataset() if isinstance(node, xr.DataTree) else node
                if len(node.data_vars) != 1:
                    raise ValueError(
                        "each mask node must hold exactly one boolean variable"
                    )
                m = node[next(iter(node.data_vars))]
        est = None
        if est_tree is not None and name in est_tree.children:
            est = est_tree[name].ds["KDP"]
        k = kdp
        if isinstance(kdp, xr.DataTree):
            k = kdp[name].ds["KDP"] if name in kdp.children else None
        items.append(_prepare(ds, fields, m, _kdp_for(ds, k, est)))
    results = _run(items, tab, kdp_min, nw_range, attrs, n_threads, use_compiled)
    nodes = {"/": obj.root.to_dataset(inherit=False)}
    nodes.update(zip(names, results))
    return xr.DataTree.from_dict(nodes)


# --------------------------------------------------------------------------
# disdrometer helpers
# --------------------------------------------------------------------------


def parsivel_bins():
    """
    Size classes of the OTT Parsivel disdrometer.

    Returns
    -------
    xarray.Dataset
        The 32 class centres ``diameter`` (mm) with ``bin_width``,
        ``diameter_lower`` and ``diameter_upper`` (mm). The two smallest
        classes are not measured by the instrument.

    References
    ----------
    Tokay, A., D. B. Wolff, and W. A. Petersen, 2014: Evaluation of the new
    version of the laser-optical disdrometer, OTT Parsivel2. *J. Atmos.
    Oceanic Technol.*, **31** (6), 1276-1288,
    https://doi.org/10.1175/JTECH-D-13-00174.1
    """
    c, w = _PARSIVEL_CENTERS, _PARSIVEL_WIDTHS
    mm = {"units": "mm"}
    return xr.Dataset(
        {
            "bin_width": ("diameter", w, {"long_name": "Size class width", **mm}),
            "diameter_lower": (
                "diameter",
                c - w / 2,
                {"long_name": "Lower class edge", **mm},
            ),
            "diameter_upper": (
                "diameter",
                c + w / 2,
                {"long_name": "Upper class edge", **mm},
            ),
        },
        coords={"diameter": ("diameter", c, {"long_name": "Drop diameter", **mm})},
        attrs={"instrument": "OTT Parsivel"},
    )


def _diameters(diameter):
    """Diameter coordinate (mm) with a bin_width coordinate."""
    if diameter is None:
        diameter = parsivel_bins()
    if isinstance(diameter, xr.Dataset):
        d = diameter["diameter"]
        width = diameter["bin_width"] if "bin_width" in diameter else None
    elif isinstance(diameter, xr.DataArray):
        d = diameter
        width = diameter.coords.get("bin_width")
    else:
        d = xr.DataArray(np.asarray(diameter, float), dims="diameter")
        width = None
    d = xr.DataArray(d.values, dims="diameter", attrs={"units": "mm"})
    if width is None:
        width = np.gradient(d.values) if d.size > 1 else np.ones(1)
    width = np.asarray(getattr(width, "values", width), float)
    return d.assign_coords(
        diameter=d.values, bin_width=("diameter", width, {"units": "mm"})
    )


def dsd_spectrum(params, diameter=None):
    """
    Rebuild the drop size distribution N(D) from gamma DSD parameters.

    Parameters
    ----------
    params : xarray.Dataset
        Gamma DSD parameters ``N0``, ``MU`` and ``LAMBDA``, e.g. the output
        of :func:`dsd` or :func:`fit_gamma_moments` (select the gates of
        interest first: the result has one more dimension).
    diameter : array-like, xarray.DataArray or xarray.Dataset, optional
        Diameters (mm), or size bins with a ``bin_width`` such as
        :func:`parsivel_bins` (the default).

    Returns
    -------
    xarray.DataArray
        ``ND`` (m-3 mm-1) on the dimensions of the parameters and
        ``diameter``, with the ``bin_width`` coordinate.

    Examples
    --------
    >>> nd = dsd_spectrum(out.isel(azimuth=10, range=100))  # doctest: +SKIP
    """
    d = _diameters(diameter)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        nd = params["N0"] * d ** params["MU"] * np.exp(-params["LAMBDA"] * d)
    nd = nd.transpose(*params["N0"].dims, "diameter")
    nd.name = "ND"
    nd.attrs = {
        "long_name": "Number concentration of drops per unit diameter",
        "units": "m-3 mm-1",
    }
    return nd


def _bin_widths(nd, dim):
    if "bin_width" in nd.coords:
        return nd["bin_width"]
    d = nd[dim].values
    width = np.gradient(d) if d.size > 1 else np.ones(1)
    return xr.DataArray(width, dims=dim, coords={dim: nd[dim]})


def fit_gamma_moments(nd, dim="diameter"):
    """
    Gamma DSD parameters of measured spectra by the method of moments.

    The shape follows from :math:`\\eta = M_4^2 / (M_2 M_6)`, which for a
    gamma DSD equals :math:`(\\mu + 3)(\\mu + 4) / ((\\mu + 5)(\\mu + 6))`,
    then :math:`\\Lambda = \\sqrt{(\\mu + 3)(\\mu + 4) M_2 / M_4}` and
    :math:`N_0 = M_2 \\Lambda^{\\mu + 3} / \\Gamma(\\mu + 3)` (the
    2-4-6 moment estimator evaluated by Cao and Zhang 2009).

    Parameters
    ----------
    nd : xarray.DataArray
        N(D) (m-3 mm-1) with a diameter dimension (mm) and, for bins of
        varying width, a ``bin_width`` coordinate (as from
        :func:`dsd_spectrum` with :func:`parsivel_bins`); without it the
        spacing of the diameters is used.
    dim : str, optional
        The diameter dimension. Default ``"diameter"``.

    Returns
    -------
    xarray.Dataset
        ``N0``, ``NW``, ``D0``, ``DM``, ``MU``, ``LAMBDA``, ``RAIN_RATE`` and
        ``LWC`` of the fitted gamma DSD, as returned by :func:`dsd`; NaN
        where the moments do not define a gamma DSD.

    References
    ----------
    Cao, Q., and G. Zhang, 2009: Errors in estimating raindrop size
    distribution parameters employing disdrometer and simulated raindrop
    spectra. *J. Appl. Meteor. Climatol.*, **48** (2), 406-425,
    https://doi.org/10.1175/2008JAMC2026.1
    """
    width = _bin_widths(nd, dim)
    d = nd[dim]
    m2, m4, m6 = ((nd * d**n * width).sum(dim) for n in (2, 4, 6))
    eta = (m4 * m4 / (m2 * m6)).values
    with np.errstate(invalid="ignore", divide="ignore"):
        a = eta - 1.0
        b = 11.0 * eta - 7.0
        c = 30.0 * eta - 12.0
        mu = (-b - np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
        mu = np.where(mu > -1.0, mu, np.nan)
        lam = np.sqrt((mu + 3.0) * (mu + 4.0) * m2.values / m4.values)
        logn0 = (
            np.log10(m2.values)
            + (mu + 3.0) * np.log10(lam)
            - np.log10(_gamma_fn(mu + 3.0))
        )
    values = _moments_numpy(logn0, mu, lam)
    data_vars = {
        name: (m2.dims, values[i], dict(_OUT_ATTRS[name]))
        for i, name in enumerate(_OUT_NAMES)
    }
    return xr.Dataset(
        data_vars,
        coords=dict(m2.coords),
        attrs={"method": "method of moments (M2, M4, M6)"},
    )


def radar_from_dsd(nd, band="S", temperature=20.0, dim="diameter"):
    """
    Polarimetric radar variables of drop size distributions.

    Parameters
    ----------
    nd : xarray.DataArray
        N(D) (m-3 mm-1) on a diameter dimension (mm), e.g. disdrometer
        spectra or :func:`dsd_spectrum`. Bin widths are taken from a
        ``bin_width`` coordinate if present, otherwise from the diameter
        spacing. Drops larger than 8 mm are ignored.
    band : {"S", "C", "X"}, optional
        Radar band. Default ``"S"``.
    temperature : float, optional
        Water temperature in degrees Celsius (0-30). Default 20.
    dim : str, optional
        The diameter dimension. Default ``"diameter"``.

    Returns
    -------
    xarray.Dataset
        ``DBZH`` (dBZ), ``ZDR`` (dB), ``KDP`` (degrees/km), ``RHOHV``,
        ``AH`` and ``ADP`` (dB/km), and the moment quantities ``DM`` (mm),
        ``NW`` (m-3 mm-1), ``LWC`` (g/m3) and ``RAIN_RATE`` (mm/h, fall speed
        of Atlas et al. 1973) on the remaining dimensions, from the
        scattering tables of :func:`scattering_table`.
    """
    tab = scattering_table(band, temperature)
    width = _bin_widths(nd, dim)
    d = nd[dim].values
    inside = d <= tab.diameter.values[-1] + 1e-9
    weights = (nd * width).where(
        xr.DataArray(inside, dims=dim, coords={dim: nd[dim]}), 0.0
    )

    def per_drop(name):
        v = np.interp(d, tab.diameter.values, tab[name].values, left=0.0, right=0.0)
        if name in ("sigma_h", "sigma_v", "copol_re", "copol_im"):
            # Rayleigh scaling (D^6) below the first tabulated diameter
            d0 = tab.diameter.values[0]
            small = d < d0
            v[small] = tab[name].values[0] * (d[small] / d0) ** 6
        return xr.DataArray(v, dims=dim, coords={dim: nd[dim]})

    k = tab.attrs["wavelength"] ** 4 / (np.pi**5 * KW2)
    zh = k * (weights * per_drop("sigma_h")).sum(dim)
    zv = k * (weights * per_drop("sigma_v")).sum(dim)
    cre = k * (weights * per_drop("copol_re")).sum(dim)
    cim = k * (weights * per_drop("copol_im")).sum(dim)
    kdp = (weights * per_drop("kdp")).sum(dim)
    ah = (weights * per_drop("ah")).sum(dim)
    av = (weights * per_drop("av")).sum(dim)
    dd = nd[dim]
    full = nd * width
    m3 = (full * dd**3).sum(dim)
    m4 = (full * dd**4).sum(dim)
    speed = 9.65 - 10.3 * np.exp(-0.6 * dd)
    with np.errstate(divide="ignore", invalid="ignore"):
        dm = m4 / m3
        out = xr.Dataset(
            {
                "DBZH": 10.0 * np.log10(zh),
                "ZDR": 10.0 * np.log10(zh / zv),
                "KDP": kdp,
                "RHOHV": np.sqrt(cre**2 + cim**2) / np.sqrt(zh * zv),
                "AH": ah,
                "ADP": ah - av,
                "DM": dm,
                "NW": 256.0 / 6.0 * m3 / dm**4,
                "LWC": np.pi / 6.0 * 1.0e-3 * m3,
                "RAIN_RATE": 6.0e-4 * np.pi * (full * speed * dd**3).sum(dim),
            }
        )
    attrs = {
        "DBZH": {
            "standard_name": "equivalent_reflectivity_factor",
            "long_name": "Equivalent reflectivity factor H",
            "units": "dBZ",
        },
        "ZDR": {
            "standard_name": "radar_differential_reflectivity_hv",
            "long_name": "Log differential reflectivity H/V",
            "units": "dB",
        },
        "KDP": {
            "standard_name": "radar_specific_differential_phase_hv",
            "long_name": "Specific differential phase HV",
            "units": "degrees/km",
        },
        "RHOHV": {
            "standard_name": "radar_correlation_coefficient_hv",
            "long_name": "Correlation coefficient HV",
            "units": "unitless",
        },
        "AH": {"long_name": "Specific attenuation H", "units": "dB/km"},
        "ADP": {"long_name": "Specific differential attenuation", "units": "dB/km"},
        "DM": dict(_OUT_ATTRS["DM"]),
        "NW": dict(_OUT_ATTRS["NW"]),
        "LWC": dict(_OUT_ATTRS["LWC"]),
        "RAIN_RATE": dict(_OUT_ATTRS["RAIN_RATE"]),
    }
    for name, a in attrs.items():
        out[name].attrs = a
    out.attrs = {"band": tab.attrs["band"], "temperature": float(temperature)}
    return out


@accessor_method("dataset", name="dsd")
def _dsd_dataset_accessor(self, method="constrained", **kwargs):
    """
    Retrieve gamma raindrop size distribution parameters.

    Parameters
    ----------
    method : {"constrained", "normalized"}, optional
        Retrieval method. Default ``"constrained"``.
    **kwargs
        Options of :func:`radarx.retrieve.dsd`, e.g. ``mask``, ``kdp``,
        ``band``.

    Returns
    -------
    xarray.Dataset
        ``N0``, ``NW``, ``D0``, ``DM``, ``MU``, ``LAMBDA``, ``RAIN_RATE``
        and ``LWC``.

    See Also
    --------
    radarx.retrieve.dsd
    """
    return dsd(self.xarray_obj, method, **kwargs)


@accessor_method("datatree", name="dsd")
def _dsd_datatree_accessor(self, method="constrained", **kwargs):
    """
    Retrieve gamma raindrop size distribution parameters for every sweep.

    All gates of all sweeps are processed in one call of the compiled
    kernel.

    Parameters
    ----------
    method : {"constrained", "normalized"}, optional
        Retrieval method. Default ``"constrained"``.
    **kwargs
        Options of :func:`radarx.retrieve.dsd`, e.g. ``mask``, ``kdp``,
        ``band``.

    Returns
    -------
    xarray.DataTree
        The root of the volume and one node per sweep with ``N0``,
        ``NW``, ``D0``, ``DM``, ``MU``, ``LAMBDA``, ``RAIN_RATE`` and
        ``LWC``.

    See Also
    --------
    radarx.retrieve.dsd
    """
    return dsd(self.xarray_obj, method, **kwargs)
