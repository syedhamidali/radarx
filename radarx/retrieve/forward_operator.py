#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Polarimetric Radar Forward Operator for WRF and MPAS Output
===========================================================

A forward operator turns the microphysics state of a numerical model into the
fields a radar would measure, so that a model run can be compared with radar
data in radar space. :func:`forward_operator` computes the horizontal
reflectivity :math:`Z_H`, the differential reflectivity :math:`Z_{DR}`, the
specific differential phase :math:`K_{DP}`, the correlation coefficient
:math:`\\rho_{hv}` and the reflectivity-weighted fall speed on the grid of the
model, from the mixing ratios of rain, snow, hail and graupel (and the rain
number concentration if the scheme has one). :func:`to_radar_grid` places the
gates of a radar sweep in the model (4/3 effective Earth radius, the site
altitude in the Earth radius, as in :mod:`radarx.fundamentals.geometry`) and
interpolates the fields to them, with the radial velocity from the model wind
and the fall speed. :func:`wrf_state` and :func:`mpas_state` turn a WRF
``wrfout`` Dataset or an MPAS history or diagnostics Dataset (on its
unstructured mesh) into the standard state Dataset that both functions take.

The operator follows the structure of the operators of Jung et al. (2008) and
of Ryzhkov et al. (2011, as reproduced in the review of Ryzhkov et al. 2020):
every hydrometeor class has a size distribution that follows from its mixing
ratio, a shape, a bulk density, an orientation (a Gaussian canting angle with
zero mean) and a scattering model, and the radar variables are the sums over
the classes of the moments of the scattering amplitudes. The model does not
predict the shape, the orientation or the density of most classes, so the
operator assumes them; every such assumption is an argument or a documented
default.

Equations
---------
Variables are those of the review of Ryzhkov et al. (2020, Appendix A; the
horizontal incidence case, the antenna at 0 degrees elevation, which is the
only incidence implemented). With :math:`s_a` and :math:`s_b` the backscatter
amplitudes (mm) along the symmetry axis and along the transverse axis of a
spheroid, :math:`\\lambda` the wavelength (mm), :math:`\\langle\\cdot\\rangle`
the integral over the size distribution :math:`N(D)` (m\\ :sup:`-3`
mm\\ :sup:`-1`, :math:`D` the equal-volume diameter in mm) and
:math:`|K_w|^2 = 0.93`,

.. math::

    Z_h = \\frac{4\\lambda^4}{\\pi^4 |K_w|^2}\\left\\{\\langle |s_b|^2\\rangle
    - 2\\,\\mathrm{Re}\\langle s_b^*(s_b - s_a)\\rangle A_2
    + \\langle |s_b - s_a|^2\\rangle A_4\\right\\},

and the same with :math:`A_1` and :math:`A_3` for :math:`Z_v`,
:math:`Z_{DR} = Z_h / Z_v`,
:math:`K_{DP} = (0.18\\lambda/\\pi)\\langle \\mathrm{Re}(s_b - s_a)\\rangle
(A_1 - A_2)` (degrees per km, forward amplitudes) and

.. math::

    \\rho_{hv} = \\frac{4\\lambda^4}{\\pi^4 |K_w|^2}
    \\frac{\\left|\\langle|s_b|^2\\rangle + \\langle|s_b - s_a|^2\\rangle A_5
    - \\langle s_b^*(s_b - s_a)\\rangle A_1
    - \\langle s_b (s_b - s_a)^*\\rangle A_2\\right|}{\\sqrt{Z_h Z_v}}.

The prefactor printed in the review for :math:`Z_h` and :math:`Z_v` (its
Eqs. A1 and A2) has :math:`\\pi^5`, while its Eq. A4 and Jung et al. (2008,
Eq. 5) have :math:`\\pi^4`; with :math:`\\sigma_b = 4\\pi |s|^2` and the
usual :math:`Z = \\lambda^4 \\sigma_b / (\\pi^5 |K_w|^2)` the power is 4, which
radarx uses. For horizontal incidence the angular moments of a Gaussian
canting angle with zero mean and standard deviation :math:`\\sigma` are
:math:`A_1 = (1 + e^{-2\\sigma^2})/2`, :math:`A_2 = (1 - e^{-2\\sigma^2})/2`,
:math:`A_3 = (3 + 4e^{-2\\sigma^2} + e^{-8\\sigma^2})/8`,
:math:`A_4 = (3 - 4e^{-2\\sigma^2} + e^{-8\\sigma^2})/8` and
:math:`A_5 = (1 - e^{-8\\sigma^2})/8` (the :math:`\\langle\\cos^4\\rangle`,
:math:`\\langle\\sin^4\\rangle` and :math:`\\langle\\sin^2\\cos^2\\rangle` of
Jung et al. 2008, Eq. 5; the definition of the moments is Eq. A10 of the
review at :math:`\\psi = 90^\\circ`). They are written out here because the
review refers to Ryzhkov (2001) for them, which was not read; they follow
from the Gaussian average of the powers of :math:`\\cos\\alpha` and
:math:`\\sin\\alpha`.

Rain
----
Rain uses the T-matrix scattering table of :func:`radarx.retrieve.scattering_table`
(Brandes et al. 2002 axis ratio, 7 degree canting, Ray 1972 refractive index at
0, 10, 20 and 30 degrees Celsius) at S, C or X band, with the table
temperature interpolated linearly per cell from the model temperature (clipped
to 0-30 degrees Celsius; ``rain_temperature`` where the model has none). The
sums over the table diameters (0.05-8 mm, trapezoid weights) are those of
:func:`radarx.retrieve.radar_from_dsd` and agree with it to rounding for the
same size distribution. The size distribution is the gamma distribution

.. math::

    N(D) = N_0 D^\\mu e^{-\\Lambda D}.

The mixing ratio :math:`q_r` and the air density :math:`\\rho_a` fix the third
moment through :math:`\\rho_a q_r = (\\pi/6)\\rho_w \\int D^3 N\\,dD` (a
definition, no source needed). Then

- with a fixed :math:`N_0` (``n0_rain``, default :math:`8\\times10^6`
  m\\ :sup:`-4`, the default intercept of the single-moment scheme of Lin et
  al. as quoted by Jung et al. 2008) and shape ``mu_rain`` (default 0, the
  exponential distribution), :math:`\\Lambda = (N_0\\Gamma(\\mu+4) /
  M_3)^{1/(\\mu+4)}`;
- with the rain number concentration :math:`N_t` (``nrain``, e.g. WRF
  ``QNRAIN``, "# kg-1" in the WRF Registry) and ``mu_rain``,
  :math:`\\Lambda = (N_t\\Gamma(\\mu+4)/(\\Gamma(\\mu+1) M_3))^{1/3}` and
  :math:`N_0 = N_t\\Lambda^{\\mu+1}/\\Gamma(\\mu+1)`.

The rain size distribution of a double-moment scheme (the shape parameter and
the diameter limits of Thompson, Morrison, NSSL or P3) is scheme-dependent and
is **not** encoded: the scheme-specific relations could not be read, so
``n0_rain`` and ``mu_rain`` are the assumptions, and the user names them. The
fall speed is that of Atlas et al. (1973) (:math:`9.65 - 10.3 e^{-0.6D}`,
at least zero, sea-level, no density correction), weighted by the
backscattering cross section.

Snow, hail and graupel
----------------------
Each ice class is an exponential size distribution
:math:`N(D) = N_0 e^{-\\Lambda D}` with a fixed intercept :math:`N_0`, the
slope :math:`\\Lambda = (\\pi\\rho_i N_0 / (\\rho_a q))^{1/4}` of the single-moment
scheme of Lin et al. (Jung et al. 2008; it is the mass relation above for
particles of bulk density :math:`\\rho_i`), an oblate spheroidal shape with the
axis ratio ``axis_ratio`` (symmetry axis over transverse axis) and a Gaussian
canting angle of standard deviation ``sigma`` degrees. Scattering is
Rayleigh scattering by uniformly filled spheroids (Ryzhkov et al. 2020,
Eqs. 3-5):

.. math::

    s_{a,b} = \\frac{\\pi^2 D^3}{6\\lambda^2}\\,
    \\frac{1}{L_{a,b} + 1/(\\varepsilon - 1)},

with the shape factors :math:`L_a` and :math:`L_b = (1 - L_a)/2` of the review
(Eq. 4 for oblate, Eq. 5 for prolate spheroids). Eq. 5 prints
:math:`\\ln\\frac{1-e}{1-e}`, a misprint; radarx uses
:math:`\\ln\\frac{1+e}{1-e}`, which gives :math:`L_a = 1/3` for a sphere.
The dielectric constant :math:`\\varepsilon` of the ice-air mixture comes from
the Maxwell Garnett mixing formula for ice inclusions in air
(:math:`(\\varepsilon-1)/(\\varepsilon+2) = \\delta K_i`, with the ice volume
fraction :math:`\\delta = \\rho_i/913` for the ice density of 913 kg m\\ :sup:`-3`
of Jung et al. 2008 and :math:`K_i = (\\varepsilon_i - 1)/(\\varepsilon_i + 2)`)
and a dielectric constant of ice :math:`\\varepsilon_i = 3.17`, the value that
reproduces the dry-hail amplitudes of Jung et al. (2008) (see below) and the
:math:`|K_i|^2 = 0.176` of :mod:`radarx.fundamentals.constants`; its dependence
on temperature and wavelength is neglected. Moments over the exponential distribution are closed forms:
:math:`\\langle D^6\\rangle = 720 N_0\\Lambda^{-7}` and
:math:`\\langle D^3\\rangle = 6 N_0\\Lambda^{-4}`.

The defaults for ``snow`` and ``hail`` are those of Jung et al. (2008) for dry
particles: snow with :math:`N_0 = 3\\times10^6` m\\ :sup:`-4`, density 100 kg
m\\ :sup:`-3`, axis ratio 0.75 and canting 20 degrees; hail with
:math:`N_0 = 4\\times10^4` m\\ :sup:`-4`, density 913 kg m\\ :sup:`-3`, axis ratio
0.75 and canting 60 degrees (the paper's :math:`60^\\circ(1 - c f_w)` at no
water). Both are the **dry** forms; the melting classes of the paper (a
rain-snow and a rain-hail mixture whose water fraction follows from the ratio
of the rain and snow mixing ratios) are **not** implemented: the paper gives
the fitted amplitudes (its Eq. 9) but not the intercept or the slope of the
mixture's size distribution, so a melting-layer bright band is missing and a
snow and a rain contribution just add where they coexist. Graupel has no
preset in any source read: ``graupel_params`` must give ``n0``, ``rho``,
``axis_ratio`` and ``sigma``. Cloud water and cloud ice (``QCLOUD``,
``QICE``) are not scatterers in the operator.

Jung et al. (2008, Eq. 9) give the amplitudes of dry snow and dry hail as
numbers fitted to Rayleigh scattering at S band; the formulas above reproduce
them to 2.6 % (snow) and 0.4 % (hail) at :math:`\\lambda` = 107 mm (see the
notebook), so there is one code path and the amplitudes scale with the band
as :math:`\\lambda^{-2}`, as Rayleigh scattering does. The Rayleigh
approximation holds while the resonance parameter :math:`D\\sqrt{|\\varepsilon|}
/\\lambda` stays below 0.3-0.4 (Ryzhkov et al. 2020); large hail at C and X band
violates it, and the operator does not check it. The fall speed of ice classes
is a constant (``fall_speed`` of the parameters, default 0 m s\\ :sup:`-1`,
i.e. the ice moves with the air): the fall speed relations of the schemes
were not read.

Radial velocity and sampling
----------------------------
At a gate with azimuth :math:`\\phi` (clockwise from north) and elevation
:math:`\\theta` the radial velocity is
:math:`u\\sin\\phi\\cos\\theta + v\\cos\\phi\\cos\\theta + (w - v_t)\\sin\\theta`
(positive away from the radar), with the model wind and the reflectivity-weighted
fall speed :math:`v_t` of the gate. The gate at slant range :math:`r` is at the
height and ground distance of Doviak and Zrnic (1993, Eqs. 2.28) with the
effective Earth radius :math:`k_e a` (``k_e``, default 4/3) and the site
altitude in the radius, as in :func:`radarx.fundamentals.geometry.beam_center_height`
and :func:`radarx.retrieve.beam_blockage`. The gate is placed on the model
columns by great-circle distance (the horizontal neighbours with inverse
distance weights, a KD-tree on the sphere: it works for the curvilinear WRF
grid and for the unstructured MPAS mesh alike), in each column the linear
fields are interpolated linearly in height, and the columns are averaged. The
radar variables that are not linear (dBZ, :math:`Z_{DR}`, :math:`\\rho_{hv}`)
are formed after the sampling from the sampled :math:`Z_h`, :math:`Z_v` and
covariance, so a gate averages in linear units.

What is left out: the beam pattern (the gate is sampled at the beam centre,
without the weighting over the beam width, the pulse volume or the beam
broadening), attenuation and differential attenuation, non-zero elevation
in the orientation moments (the scattering is for horizontal incidence, as
the rain tables; see :mod:`radarx.retrieve.dsd`), the melting classes,
resonance effects for ice, the beam blockage and the radar noise.

Reading model output
--------------------
:func:`wrf_state` takes the variables of a ``wrfout`` file: ``T`` (the
perturbation of the potential temperature over 300 K; the moist potential
temperature :math:`\\theta(1 + 1.61 q_v)` if the file has ``USE_THETA_M`` equal
to 1, as the WRF Registry states), ``P`` and ``PB``, ``PH`` and ``PHB``
(geopotential on the staggered levels, divided by 9.81 m s\\ :sup:`-2`),
``QVAPOR``, ``QRAIN``, ``QSNOW``, ``QGRAUP``, ``QHAIL``, ``QNRAIN``, the
staggered ``U``, ``V`` and ``W`` (averaged to the mass points) and ``XLAT`` and
``XLONG``. The temperature is :math:`\\theta (p/10^5\\,\\mathrm{Pa})^{R_d/c_p}`
and the dry-air density (the air the mixing ratios refer to)
:math:`p / (R_d T (1 + 1.61 q_v))`, with :math:`R_d` = 287 J kg\\ :sup:`-1`
K\\ :sup:`-1` and :math:`c_p = 7 R_d / 2` (the values of
``module_model_constants.F`` of WRF). :func:`mpas_state` takes ``qr``, ``qs``,
``qg``, ``nr`` ("nb kg-1" in the MPAS-Atmosphere Registry), ``rho`` (dry air
density), ``pressure`` (or ``pressure_p`` and ``pressure_base``), ``theta``,
``uReconstructZonal``, ``uReconstructMeridional``, ``w`` and, from the grid
file if the output has none, ``lonCell``, ``latCell`` (radians) and ``zgrid``
(heights of the layer interfaces, averaged to the layer centres). Read
the files with xarray (``xr.open_dataset``); radarx does not download
model data. No WRF output exists on the machine this was developed on, so the
operator was tested on synthetic WRF-like and MPAS-like Datasets built from
the variable definitions of the two Registries.

Sources
-------
Read in full for this module: Jung et al. (2008), the open copy of the
Center for Analysis and Prediction of Storms (Eqs. 1-20: the shape, the
orientation, the intercepts and densities of the Lin scheme, the amplitudes,
the closed forms for exponential distributions); Ryzhkov et al. (2020), the
open copy in the NOAA repository (Eqs. 3-5, A1-A13 and the descriptions of the
operators). Not read: Ryzhkov et al. (2011), which the review cites for its
equations and mixing rules (no open copy was found), so the ice
dielectric-constant mixing of that paper is not used; Jung et al. (2010, the
two-moment operator); Pfeifer et al. (2008), Augros et al. (2018) and
Wolfensberger and Berne (2018). The WRF and MPAS variable names, units and
constants are from the Registry files and ``module_model_constants.F`` of the
two models.

References
----------
Jung, Y., G. Zhang, and M. Xue, 2008: Assimilation of simulated polarimetric
radar data for a convective storm using the ensemble Kalman filter. Part I:
Observation operators for reflectivity and polarimetric variables. *Mon.
Wea. Rev.*, **136** (6), 2228-2245, https://doi.org/10.1175/2007MWR2083.1

Ryzhkov, A. V., J. Snyder, J. T. Carlin, A. Khain, and M. Pinsky, 2020: What
polarimetric weather radars offer to cloud modelers: Forward radar operators
and microphysical/thermodynamic retrievals. *Atmosphere*, **11** (4), 362,
https://doi.org/10.3390/atmos11040362

Ryzhkov, A., M. Pinsky, A. Pokrovsky, and A. Khain, 2011: Polarimetric radar
observation operator for a cloud model with spectral microphysics. *J. Appl.
Meteor. Climatol.*, **50** (4), 873-894,
https://doi.org/10.1175/2010JAMC2363.1

Maxwell Garnett, J. C., 1904: Colours in metal glasses and in metallic films.
*Philos. Trans. R. Soc. Lond. A*, **203**, 385-420,
https://doi.org/10.1098/rsta.1904.0024

.. autosummary::
   :nosignatures:
   :toctree: generated/

   forward_operator
   to_radar_grid
   wrf_state
   mpas_state
"""

from __future__ import annotations

__all__ = ["forward_operator", "to_radar_grid", "wrf_state", "mpas_state"]

import functools
import importlib

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
from scipy.special import gamma as _gamma

from .._compat import renamed_arguments
from .._provenance import provenance
from .._registry import accessor_method
from ..fundamentals.constants import EARTH_RADIUS
from ..fundamentals.geometry import beam_center_height, ground_range
from ._fieldtools import broadcast_block, choose_engine
from ._raindrop_fall import sea_level_speed
from .beam_blockage import _locate

try:
    from . import _forward_operator

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _forward_operator = None
    HAS_COMPILED_KERNEL = False

_dsd = importlib.import_module("radarx.retrieve.dsd")

_DEG = np.pi / 180.0
_BLOCK = 2048  # cells per block of the NumPy implementation
_GATE_BLOCK = 20000  # gates per block of the NumPy sampling
# pi / 6 * 1000 kg m-3 * 1e-9 m3 mm-3: kg m-3 of a third moment of 1 mm3 m-3
_MASS = np.pi / 6.0 * 1.0e-6
_ICE_DENSITY = 913.0  # kg m-3, the density of ice in Jung et al. (2008)
_EPS_ICE = 3.17

# WRF: module_model_constants.F (r_d, cp = 7 r_d / 2, g, p1000mb, t0) and the
# "theta_m = theta (1 + 1.61 Qv)" of the Registry (use_theta_m)
_R_D = 287.0
_CP = 7.0 * _R_D / 2.0
_G = 9.81
_P0 = 1.0e5
_T0 = 300.0
_RV_RD = 1.61

#: Dry snow and dry hail of Jung et al. (2008): intercept (m-4), bulk density
#: (kg m-3), axis ratio, canting angle standard deviation (degrees) and the
#: constant fall speed (m s-1).
SNOW_PARAMS = {
    "n0": 3.0e6,
    "rho": 100.0,
    "axis_ratio": 0.75,
    "sigma": 20.0,
    "fall_speed": 0.0,
}
HAIL_PARAMS = {
    "n0": 4.0e4,
    "rho": 913.0,
    "axis_ratio": 0.75,
    "sigma": 60.0,
    "fall_speed": 0.0,
}
_ICE_KEYS = ("n0", "rho", "axis_ratio", "sigma", "fall_speed")
_STATE_NAMES = {
    "rain": "QRAIN",
    "snow": "QSNOW",
    "hail": "QHAIL",
    "graupel": "QGRAUP",
    "nrain": "QNRAIN",
    "density": "AIR_DENSITY",
    "temperature": "TEMPERATURE",
}
_MOMENTS_ORDER = ("Z_H", "Z_V", "Z_HV_REAL", "Z_HV_IMAG", "KDP", "ZH_VT")
_CARRIED = ("HEIGHT", "U", "V", "W")


# --------------------------------------------------------------------------
# scattering by spheroids and the ice classes
# --------------------------------------------------------------------------


def _shape_factors(axis_ratio):
    """L_a, L_b of a spheroid with the given axis ratio (symmetry axis over
    transverse axis), Ryzhkov et al. (2020), Eqs. 4-5."""
    r = float(axis_ratio)
    if abs(r - 1.0) < 1.0e-9:
        return 1.0 / 3.0, 1.0 / 3.0
    if r < 1.0:  # oblate
        f = np.sqrt(1.0 / r**2 - 1.0)
        la = (1.0 + f**2) / f**2 * (1.0 - np.arctan(f) / f)
    else:  # prolate
        e = np.sqrt(1.0 - 1.0 / r**2)
        la = (1.0 - e**2) / e**2 * (np.log((1.0 + e) / (1.0 - e)) / (2.0 * e) - 1.0)
    return la, 0.5 * (1.0 - la)


def _dielectric_constant(rho, eps_ice=_EPS_ICE):
    """Maxwell Garnett dielectric constant of ice inclusions in air."""
    k_ice = (eps_ice - 1.0) / (eps_ice + 2.0)
    mixed = (rho / _ICE_DENSITY) * k_ice
    return (1.0 + 2.0 * mixed) / (1.0 - mixed)


def rayleigh_amplitudes(rho, axis_ratio, wavelength, eps_ice=_EPS_ICE):
    """
    Rayleigh backscatter amplitudes per unit D**3 (mm-2) of dry ice spheroids.

    Returns ``(alpha_a, alpha_b)``: along the symmetry axis and the transverse
    axis, so that ``s = alpha D**3`` in mm for D in mm and ``wavelength`` in mm.
    """
    la, lb = _shape_factors(axis_ratio)
    inv = 1.0 / (_dielectric_constant(rho, eps_ice) - 1.0)
    c = np.pi**2 / (6.0 * wavelength**2)
    return c / (la + inv), c / (lb + inv)


def _orientation_moments(sigma_deg):
    """A1 to A5 of a Gaussian canting angle (zero mean) at horizontal incidence."""
    s = np.deg2rad(float(sigma_deg))
    e1, e2 = np.exp(-2.0 * s * s), np.exp(-8.0 * s * s)
    return (
        0.5 * (1.0 + e1),
        0.5 * (1.0 - e1),
        (3.0 + 4.0 * e1 + e2) / 8.0,
        (3.0 - 4.0 * e1 + e2) / 8.0,
        (1.0 - e2) / 8.0,
    )


def _ice_constants(params, wavelength):
    """Constants of one ice class: Z_H, Z_V, covariance and K_DP per unit moment."""
    a_a, a_b = rayleigh_amplitudes(params["rho"], params["axis_ratio"], wavelength)
    a1, a2, a3, a4, a5 = _orientation_moments(params["sigma"])
    d = a_b - a_a
    pref = 4.0 * wavelength**4 / (np.pi**4 * _dsd.KW2)
    return (
        pref * (a_b**2 - 2.0 * a_b * d * a2 + d**2 * a4),
        pref * (a_b**2 - 2.0 * a_b * d * a1 + d**2 * a3),
        pref * (a_b**2 + d**2 * a5 - a_b * d * (a1 + a2)),
        0.18 * wavelength / np.pi * d * (a1 - a2),
    )


def _check_ice(name, params, defaults):
    """Parameters of an ice class: the defaults updated by the user's."""
    if defaults is None and params is None:
        raise ValueError(
            f"{name}_params must give n0, rho, axis_ratio and sigma: there is no "
            f"default for {name}"
        )
    merged = {"fall_speed": 0.0}
    merged.update(defaults or {})
    merged.update(params or {})
    unknown = set(merged) - set(_ICE_KEYS)
    missing = [k for k in _ICE_KEYS if k not in merged]
    if unknown or missing:
        raise ValueError(
            f"{name}_params: unknown keys {sorted(unknown)}, missing {missing}; "
            f"the keys are {_ICE_KEYS}"
        )
    merged = {k: float(merged[k]) for k in _ICE_KEYS}
    if not (merged["n0"] > 0 and 0 < merged["rho"] <= _ICE_DENSITY):
        raise ValueError(
            f"{name}_params: n0 must be positive and rho between 0 and "
            f"{_ICE_DENSITY:g} kg m-3"
        )
    if not (merged["axis_ratio"] > 0 and merged["sigma"] >= 0):
        raise ValueError(
            f"{name}_params: axis_ratio must be positive and sigma not negative"
        )
    if not merged["fall_speed"] >= 0:
        raise ValueError(f"{name}_params: fall_speed must not be negative")
    return merged


# --------------------------------------------------------------------------
# NumPy reference of the kernels
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=4)
def _rain_table(band):
    """The rain scattering table of one band: wavelength and kernel arrays."""
    raw = _dsd._raw_table()[band]
    d = raw["diameter"]
    return {
        "wavelength": raw["wavelength"],
        "temps": np.ascontiguousarray(raw["temperature"], dtype=np.float64),
        "diam": np.ascontiguousarray(d, dtype=np.float64),
        "weights": _dsd._trapezoid_weights(d),
        "vfall": np.asarray(sea_level_speed(d, "atlas1973")[0], dtype=np.float64),
        "table": np.ascontiguousarray(raw["data"][:, :, :5], dtype=np.float64),
    }


def _cells_numpy(
    q_rain,
    n_rain,
    rho,
    temp_c,
    tab,
    kfac,
    n0_rain,
    mu_rain,
    q_min,
    temp_default,
    nr_per_kg,
    ice_q,
    ice_par,
):
    """NumPy implementation of ``_forward_operator.cells`` (same results)."""
    n = rho.size
    out = np.zeros((6, n))
    air = np.isfinite(rho) & (rho > 0.0)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore", under="ignore"):
        if q_rain is not None:
            _rain_numpy(
                out,
                np.flatnonzero(air & (q_rain > q_min)),
                q_rain,
                n_rain,
                rho,
                temp_c,
                tab,
                kfac,
                n0_rain,
                mu_rain,
                temp_default,
                nr_per_kg,
            )
        for q, par in zip(ice_q, ice_par):
            n0_p, rho_i, zh_c, zv_c, cov_c, kdp_c, vt = par
            i = np.flatnonzero(air & (q > q_min))
            lam = (np.pi * rho_i * n0_p / (rho[i] * q[i])) ** 0.25 * 1.0e-3
            n0 = n0_p * 1.0e-3
            z7 = 720.0 * n0 * lam**-7.0
            zhc = zh_c * z7
            out[0, i] += zhc
            out[1, i] += zv_c * z7
            out[2, i] += cov_c * z7
            out[4, i] += kdp_c * 6.0 * n0 * lam**-4.0
            out[5, i] += vt * zhc
    return out.astype(np.float32)


def _rain_numpy(
    out,
    sel,
    q_rain,
    n_rain,
    rho,
    temp_c,
    tab,
    kfac,
    n0_rain,
    mu_rain,
    temp_default,
    nr_per_kg,
):
    mu = mu_rain
    g_mu1, g_mu4 = _gamma(mu + 1.0), _gamma(mu + 4.0)
    n0m = n0_rain * 1.0e-3
    temps, d, w, vfall, table = (
        tab[k] for k in ("temps", "diam", "weights", "vfall", "table")
    )
    logd = np.log(d)
    nt = temps.size
    for s in range(0, sel.size, _BLOCK):
        i = sel[s : s + _BLOCK]
        m3 = q_rain[i] * rho[i] / _MASS
        if n_rain is not None:
            nt_cell = n_rain[i] * rho[i] if nr_per_kg else n_rain[i]
            use = np.isfinite(nt_cell) & (nt_cell > 0.0)
        else:
            nt_cell = np.full(i.size, -1.0)
            use = np.zeros(i.size, dtype=bool)
        lam = np.where(
            use,
            (nt_cell * g_mu4 / (g_mu1 * m3)) ** (1.0 / 3.0),
            (n0m * g_mu4 / m3) ** (1.0 / (mu + 4.0)),
        )
        n0 = np.where(use, nt_cell * lam ** (mu + 1.0) / g_mu1, n0m)
        temp = np.full(i.size, float(temp_default))
        if temp_c is not None:
            tc = temp_c[i]
            temp = np.where(np.isfinite(tc), tc, temp)
        temp = np.clip(temp, temps[0], temps[-1])
        k = np.clip(np.searchsorted(temps, temp, side="right") - 1, 0, nt - 2)
        wt = (temp - temps[k]) / (temps[k + 1] - temps[k])
        d0 = table[k]
        d1 = table[np.minimum(k + 1, nt - 1)]
        data = d0 + wt[:, None, None] * (d1 - d0)
        nd = n0[:, None] * np.exp(mu * logd[None, :] - lam[:, None] * d[None, :])
        nd = nd * w[None, :]
        a = nd * data[:, :, 0]
        out[0, i] += kfac * a.sum(1)
        out[1, i] += kfac * (nd * data[:, :, 1]).sum(1)
        # the table holds the covariance with the opposite sign (see the kernel)
        out[2, i] -= kfac * (nd * data[:, :, 2]).sum(1)
        out[3, i] -= kfac * (nd * data[:, :, 3]).sum(1)
        out[4, i] += (nd * data[:, :, 4]).sum(1)
        out[5, i] += kfac * (a * vfall[None, :]).sum(1)


def _sample_numpy(fields, height, idx, weight, gate_h):
    """NumPy implementation of ``_forward_operator.sample`` (same results)."""
    nz = height.shape[1]
    ng, k = idx.shape
    out = np.full((len(fields), ng), np.nan, dtype=np.float32)
    heights = np.asarray(height, dtype=np.float32)
    fields = [np.asarray(f, dtype=np.float32) for f in fields]
    for g0 in range(0, ng, _GATE_BLOCK):
        sl = slice(g0, min(ng, g0 + _GATE_BLOCK))
        h = gate_h[sl]
        total = np.zeros((len(fields), h.size))
        wsum = np.zeros((len(fields), h.size))
        for j in range(k):
            col = idx[sl, j]
            w = weight[sl, j]
            ok = (col >= 0) & (w > 0.0) & np.isfinite(h)
            c = np.where(ok, col, 0)
            hh = heights[c].astype(np.float64)
            ok &= (h >= hh[:, 0]) & (h <= hh[:, -1])
            lo = np.clip((hh <= h[:, None]).sum(1) - 1, 0, max(nz - 2, 0))
            up = np.minimum(lo + 1, nz - 1)
            rows = np.arange(h.size)
            hl, hu = hh[rows, lo], hh[rows, up]
            with np.errstate(invalid="ignore", divide="ignore"):
                t = np.where(hu > hl, (h - hl) / (hu - hl), 0.0)
            for f, field in enumerate(fields):
                a = field[c, lo].astype(np.float64)
                b = field[c, up].astype(np.float64)
                v = a + t * (b - a)
                good = ok & np.isfinite(v)
                total[f] += np.where(good, w * v, 0.0)
                wsum[f] += np.where(good, w, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, sl] = np.where(wsum > 0.0, total / wsum, np.nan)
    return out


# --------------------------------------------------------------------------
# model states
# --------------------------------------------------------------------------


def _destagger(da, dim, new):
    """Average a staggered dimension to the mass points."""
    if dim not in da.dims:
        raise KeyError(f"{da.name!r} has no dimension {dim!r}")
    lo = da.isel({dim: slice(None, -1)}).variable
    hi = da.isel({dim: slice(1, None)}).variable
    out = xr.DataArray(0.5 * (lo + hi), name=da.name, attrs=da.attrs)
    return out.rename({dim: new})


_STATE_ATTRS = {
    "TEMPERATURE": ("Air temperature", "K"),
    "PRESSURE": ("Air pressure", "Pa"),
    "AIR_DENSITY": ("Dry air density", "kg m-3"),
    "HEIGHT": ("Height of the model levels above sea level", "m"),
    "U": ("Eastward wind at the mass points", "m s-1"),
    "V": ("Northward wind at the mass points", "m s-1"),
    "W": ("Vertical wind at the mass points", "m s-1"),
    "QVAPOR": ("Water vapor mixing ratio", "kg kg-1"),
    "QRAIN": ("Rain mixing ratio", "kg kg-1"),
    "QSNOW": ("Snow mixing ratio", "kg kg-1"),
    "QGRAUP": ("Graupel mixing ratio", "kg kg-1"),
    "QHAIL": ("Hail mixing ratio", "kg kg-1"),
    "QNRAIN": ("Rain number concentration", "kg-1"),
}


def _finish_state(variables, latitude, longitude):
    ds = xr.Dataset(variables)
    for name, (long_name, units) in _STATE_ATTRS.items():
        if name in ds:
            ds[name].attrs = {"long_name": long_name, "units": units}
    ds = ds.assign_coords(
        latitude=latitude.assign_attrs(
            standard_name="latitude", long_name="Latitude", units="degrees_north"
        ),
        longitude=longitude.assign_attrs(
            standard_name="longitude", long_name="Longitude", units="degrees_east"
        ),
    )
    return ds


def _need(ds, names, what):
    missing = [n for n in names if n not in ds]
    if missing:
        raise KeyError(f"{what} needs the variables {missing}")


@renamed_arguments(ds="obj")
@provenance("WRF output as the state of the polarimetric forward operator")
def wrf_state(obj, *, use_theta_m=None):
    """
    The state of a WRF run for the forward operator.

    Parameters
    ----------
    obj : xarray.Dataset
        The variables of a ``wrfout`` file (read with ``xr.open_dataset``):
        ``T``, ``P``, ``PB``, ``PH``, ``PHB``, ``U``, ``V``, ``W``, ``XLAT``,
        ``XLONG`` and the hydrometeors ``QRAIN``, ``QSNOW``, ``QGRAUP``,
        ``QHAIL``, ``QNRAIN`` that the run has (``QVAPOR`` for the moisture
        correction of the temperature and the density).
    use_theta_m : bool, optional
        Whether ``T`` is the moist potential temperature
        :math:`\\theta(1 + 1.61 q_v) - 300` K. Default: the ``USE_THETA_M``
        attribute of the file, else False.

    Returns
    -------
    xarray.Dataset
        On the mass grid of the model (``Time``, ``bottom_top``,
        ``south_north``, ``west_east``): ``TEMPERATURE`` (K), ``PRESSURE``
        (Pa), ``AIR_DENSITY`` (dry air, kg m-3), ``HEIGHT`` (m above sea
        level at the mass levels), ``U``, ``V``, ``W`` (m s-1 at the mass
        points) and the hydrometeor fields that ``obj`` has, with the
        coordinates ``latitude`` and ``longitude`` (degrees). The input is
        not changed.

    Raises
    ------
    KeyError
        If a variable that the state needs is missing.

    Notes
    -----
    Constants and the relations are listed in
    :mod:`radarx.retrieve.forward_operator`: the base temperature 300 K, the
    gas constant 287 J kg-1 K-1, :math:`c_p = 7 R_d/2`, g = 9.81 m s-2 and
    :math:`p_0` = 1e5 Pa of WRF. Staggered fields are averaged to the mass
    points. The terrain-following levels are kept, so ``HEIGHT`` varies in
    the horizontal.

    References
    ----------
    Jung, Y., G. Zhang, and M. Xue, 2008: Assimilation of simulated
    polarimetric radar data for a convective storm using the ensemble Kalman
    filter. Part I: Observation operators for reflectivity and polarimetric
    variables. *Mon. Wea. Rev.*, **136** (6), 2228-2245,
    https://doi.org/10.1175/2007MWR2083.1
    """
    if not isinstance(obj, xr.Dataset):
        raise TypeError("ds must be an xarray Dataset")
    _need(obj, ("T", "P", "PB", "PH", "PHB", "XLAT", "XLONG"), "wrf_state")
    if use_theta_m is None:
        use_theta_m = bool(int(obj.attrs.get("USE_THETA_M", 0)))
    qv = obj["QVAPOR"] if "QVAPOR" in obj else xr.zeros_like(obj["T"])
    theta = obj["T"] + _T0
    if use_theta_m:
        theta = theta / (1.0 + _RV_RD * qv)
    pressure = obj["P"] + obj["PB"]
    temperature = theta * (pressure / _P0) ** (_R_D / _CP)
    density = pressure / (_R_D * temperature * (1.0 + _RV_RD * qv))
    geopotential = _destagger(obj["PH"] + obj["PHB"], "bottom_top_stag", "bottom_top")
    variables = {
        "TEMPERATURE": temperature,
        "PRESSURE": pressure,
        "AIR_DENSITY": density,
        "HEIGHT": geopotential / _G,
    }
    for name, dim, new in (
        ("U", "west_east_stag", "west_east"),
        ("V", "south_north_stag", "south_north"),
        ("W", "bottom_top_stag", "bottom_top"),
    ):
        if name in obj:
            variables[name] = _destagger(obj[name], dim, new)
    if "QVAPOR" in obj:
        variables["QVAPOR"] = obj["QVAPOR"]
    for name in ("QRAIN", "QSNOW", "QGRAUP", "QHAIL", "QNRAIN"):
        if name in obj:
            variables[name] = obj[name]

    def horizontal(da):
        da = da.isel(Time=0, drop=True) if "Time" in da.dims else da
        return da.reset_coords(drop=True)

    out = _finish_state(
        {k: v.reset_coords(drop=True) for k, v in variables.items()},
        horizontal(obj["XLAT"]),
        horizontal(obj["XLONG"]),
    )
    out.attrs = {"source": "WRF", "use_theta_m": int(use_theta_m)}
    return out


@renamed_arguments(ds="obj")
@provenance("MPAS output as the state of the polarimetric forward operator")
def mpas_state(obj, grid=None):
    """
    The state of an MPAS-Atmosphere run for the forward operator.

    Parameters
    ----------
    obj : xarray.Dataset
        An MPAS history or diagnostics Dataset on the unstructured mesh, with
        ``qr``, ``qs``, ``qg``, ``nr``, ``rho``, ``pressure`` (or
        ``pressure_p`` and ``pressure_base``), ``theta``,
        ``uReconstructZonal``, ``uReconstructMeridional``, ``w`` (``exner``
        gives the temperature if present) and the hydrometeors that the run
        has. The variables ``lonCell``, ``latCell`` and ``zgrid`` are taken
        from ``obj`` if it has them.
    grid : xarray.Dataset, optional
        The static grid file, for ``lonCell``, ``latCell`` (radians) and
        ``zgrid`` (m, ``nCells`` by ``nVertLevelsP1``) when ``obj`` has none.

    Returns
    -------
    xarray.Dataset
        On ``nCells`` and ``nVertLevels`` (and ``Time``): ``TEMPERATURE``,
        ``PRESSURE``, ``AIR_DENSITY`` (dry air), ``HEIGHT`` (layer centres,
        m above sea level), ``U``, ``V``, ``W`` and the hydrometeors
        ``QRAIN``, ``QSNOW``, ``QGRAUP``, ``QNRAIN`` the run has, with the
        coordinates ``latitude`` and ``longitude`` in degrees on ``nCells``.
        The input is not changed.

    Raises
    ------
    KeyError
        If a variable that the state needs is missing.

    Notes
    -----
    The temperature is ``theta * exner`` or :math:`\\theta (p/10^5)^{R_d/c_p}`
    with the constants of :mod:`radarx.retrieve.forward_operator`. Layer
    interfaces (``zgrid``) and the vertical velocity are averaged to the layer
    centres. The variable names and units are those of the MPAS-Atmosphere
    Registry.

    References
    ----------
    Jung, Y., G. Zhang, and M. Xue, 2008: Assimilation of simulated
    polarimetric radar data for a convective storm using the ensemble Kalman
    filter. Part I: Observation operators for reflectivity and polarimetric
    variables. *Mon. Wea. Rev.*, **136** (6), 2228-2245,
    https://doi.org/10.1175/2007MWR2083.1
    """
    if not isinstance(obj, xr.Dataset):
        raise TypeError("ds must be an xarray Dataset")
    if grid is not None and not isinstance(grid, xr.Dataset):
        raise TypeError("grid must be an xarray Dataset")

    def pick(name):
        for holder in (obj, grid):
            if holder is not None and name in holder.variables:
                return holder[name]
        raise KeyError(
            f"mpas_state needs {name!r} in ds or in grid (the static grid file)"
        )

    for name in ("lonCell", "latCell", "zgrid"):
        pick(name)
    _need(obj, ("rho", "theta"), "mpas_state")
    if "pressure" in obj:
        pressure = obj["pressure"]
    else:
        _need(obj, ("pressure_p", "pressure_base"), "mpas_state")
        pressure = obj["pressure_p"] + obj["pressure_base"]
    if "exner" in obj:
        temperature = obj["theta"] * obj["exner"]
    else:
        temperature = obj["theta"] * (pressure / _P0) ** (_R_D / _CP)
    zgrid = pick("zgrid").reset_coords(drop=True)
    zgrid = zgrid.transpose(..., "nVertLevelsP1")
    variables = {
        "TEMPERATURE": temperature,
        "PRESSURE": pressure,
        "AIR_DENSITY": obj["rho"],
        "HEIGHT": _destagger(zgrid, "nVertLevelsP1", "nVertLevels"),
    }
    for name, new in (("uReconstructZonal", "U"), ("uReconstructMeridional", "V")):
        if name in obj:
            variables[new] = obj[name]
    if "w" in obj:
        variables["W"] = _destagger(obj["w"], "nVertLevelsP1", "nVertLevels")
    for name, new in (
        ("qv", "QVAPOR"),
        ("qr", "QRAIN"),
        ("qs", "QSNOW"),
        ("qg", "QGRAUP"),
        ("nr", "QNRAIN"),
    ):
        if name in obj:
            variables[new] = obj[name]

    def degrees(name):
        da = pick(name).reset_coords(drop=True)
        return da.copy(data=np.rad2deg(da.data))

    out = _finish_state(
        {k: v.reset_coords(drop=True) for k, v in variables.items()},
        degrees("latCell"),
        degrees("lonCell"),
    )
    out.attrs = {"source": "MPAS-Atmosphere"}
    return out


# --------------------------------------------------------------------------
# forward operator on the model grid
# --------------------------------------------------------------------------

_ATTRS = {
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
        "units": "degree km-1",
    },
    "RHOHV": {
        "standard_name": "radar_correlation_coefficient_hv",
        "long_name": "Correlation coefficient HV",
        "units": "1",
    },
    "FALL_SPEED": {
        "long_name": "Reflectivity-weighted terminal fall speed",
        "units": "m s-1",
        "comment": "positive downward",
    },
    "Z_H": {"long_name": "Reflectivity factor H, linear", "units": "mm6 m-3"},
    "Z_V": {"long_name": "Reflectivity factor V, linear", "units": "mm6 m-3"},
    "Z_HV_REAL": {
        "long_name": "Real part of the H-V covariance, in units of Z",
        "units": "mm6 m-3",
    },
    "Z_HV_IMAG": {
        "long_name": "Imaginary part of the H-V covariance, in units of Z",
        "units": "mm6 m-3",
    },
    "ZH_VT": {
        "long_name": "Linear reflectivity H times the fall speed",
        "units": "mm6 m-3 m s-1",
    },
    "VRADH": {
        "standard_name": "radial_velocity_of_scatterers_away_from_instrument",
        "long_name": "Radial velocity of scatterers away from instrument",
        "units": "m s-1",
    },
}


def _choose(state, value, key, what, optional=True):
    """The DataArray named by an argument: ``"auto"`` is the standard name."""
    if value is None:
        return None
    if isinstance(value, str) and value == "auto":
        name = _STATE_NAMES[key]
        return state[name] if name in state else None
    if not isinstance(value, str) or value not in state:
        raise KeyError(f"{what}={value!r} is not a variable of the state")
    return state[value]


def _air_density(state, density):
    rho = _choose(state, density, "density", "density")
    if rho is not None:
        return rho
    if "PRESSURE" in state and "TEMPERATURE" in state:
        qv = state["QVAPOR"] if "QVAPOR" in state else 0.0
        return state["PRESSURE"] / (_R_D * state["TEMPERATURE"] * (1.0 + _RV_RD * qv))
    raise ValueError(
        "the state needs the air density (AIR_DENSITY, or PRESSURE and TEMPERATURE); "
        "wrf_state and mpas_state build it"
    )


def _kernel_call(compiled, n_threads, spec):
    """The block function of ``apply_ufunc``: kernel or NumPy on flat blocks."""

    def block(*blocks):
        shape, flat = broadcast_block(*blocks)
        it = iter(flat)
        rho = next(it)
        q_rain = next(it) if spec["rain"] else None
        n_rain = next(it) if spec["nrain"] else None
        temp_c = next(it) if spec["temperature"] else None
        ice_q = [next(it) for _ in range(spec["n_ice"])]
        tab = spec["table"]
        args = (
            q_rain,
            n_rain,
            rho,
            temp_c,
        )
        scalars = (
            spec["kfac"],
            spec["n0_rain"],
            spec["mu_rain"],
            spec["q_min"],
            spec["temp_default"],
            spec["nr_per_kg"],
        )
        if compiled:
            out = _forward_operator.cells(
                *args,
                tab["temps"],
                tab["diam"],
                tab["weights"],
                tab["vfall"],
                tab["table"],
                *scalars,
                ice_q,
                spec["ice_par"],
                int(n_threads or 0),
            )
        else:
            out = _cells_numpy(*args, tab, *scalars, ice_q, spec["ice_par"])
        return tuple(out[j].reshape(shape) for j in range(6))

    return block


@provenance(
    "Polarimetric radar forward operator: T-matrix rain and Rayleigh dry snow, "
    "hail and graupel from the mixing ratios of a numerical model, after Jung "
    "et al. (2008) and Ryzhkov et al. (2020)"
)
def forward_operator(
    state,
    *,
    band="S",
    rain="auto",
    snow="auto",
    hail=None,
    graupel=None,
    nrain=None,
    n0_rain=8.0e6,
    mu_rain=0.0,
    snow_params=None,
    hail_params=None,
    graupel_params=None,
    density="auto",
    temperature="auto",
    rain_temperature=10.0,
    nrain_per_kg=True,
    min_mixing_ratio=1.0e-9,
    min_dbz=-30.0,
    n_threads=None,
    engine="auto",
):
    """
    Polarimetric radar variables of a model state on the model grid.

    Parameters
    ----------
    state : xarray.Dataset
        The microphysics state: the hydrometeor mixing ratios (kg kg-1), the
        dry air density (kg m-3, ``AIR_DENSITY``; or ``PRESSURE`` and
        ``TEMPERATURE``), the temperature (K) and, for sampling, ``HEIGHT``,
        ``U``, ``V``, ``W`` and the coordinates ``latitude`` and
        ``longitude``. :func:`wrf_state` and :func:`mpas_state` build it. Any
        dimensions work (the operator is cell by cell); dask arrays give a
        lazy result.
    band : {"S", "C", "X"}, optional
        Radar band (the rain table: 2.8, 5.6, 9.4 GHz). Default ``"S"``.
    rain, snow, hail, graupel : str or None, optional
        Names of the mixing ratios in ``state``. ``"auto"`` (the default of
        ``rain`` and ``snow``) takes ``QRAIN`` and ``QSNOW`` if the state has
        them; ``None`` leaves the class out. ``hail`` and ``graupel`` are
        left out by default: name ``QHAIL`` or ``QGRAUP`` to include them.
    nrain : str or None, optional
        Name of the rain number concentration (``QNRAIN``; default None, the
        intercept ``n0_rain`` is fixed). A cell with no positive value falls
        back to ``n0_rain``.
    n0_rain : float, optional
        Rain intercept (m-4) where the number concentration is not used.
        Default 8e6, the default intercept of the single-moment scheme of Lin
        et al. in Jung et al. (2008). The intercept or the shape of the
        rain size distribution of a double-moment scheme is not taken from
        the scheme: set the assumption you want to test.
    mu_rain : float, optional
        Shape of the rain gamma distribution, greater than -1. Default 0
        (exponential).
    snow_params, hail_params, graupel_params : dict, optional
        ``n0`` (m-4), ``rho`` (bulk density, kg m-3), ``axis_ratio``,
        ``sigma`` (canting angle standard deviation, degrees) and
        ``fall_speed`` (m s-1, default 0) of the class. Snow and hail default
        to the dry values of Jung et al. (2008) (see
        :mod:`radarx.retrieve.forward_operator`) and keys given here replace
        them; graupel has no default, all keys but ``fall_speed`` are
        required.
    density, temperature : str or None, optional
        Names of the dry air density and the temperature (K) in ``state``;
        ``"auto"`` takes ``AIR_DENSITY`` and ``TEMPERATURE``. Without a
        temperature the rain table uses ``rain_temperature``.
    rain_temperature : float, optional
        Water temperature (degrees Celsius, 0-30) of the rain table where
        the model temperature is not used. Default 10.
    nrain_per_kg : bool, optional
        Whether ``nrain`` is per kg of air (WRF, MPAS; default) or per m3.
    min_mixing_ratio : float, optional
        Classes with a mixing ratio at or below this (kg kg-1) do not
        contribute. Default 1e-9.
    min_dbz : float, optional
        Cells with a reflectivity below this (dBZ) have no echo: their
        ``DBZH``, ``ZDR``, ``RHOHV`` and ``FALL_SPEED`` are NaN. Default -30
        (a radarx choice).
    n_threads : int, optional
        Threads of the compiled kernel (per dask chunk). Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation; ``"auto"`` prefers the compiled kernel.

    Returns
    -------
    xarray.Dataset
        On the grid of the mixing ratios: ``DBZH`` (dBZ), ``ZDR`` (dB),
        ``KDP`` (degree km-1), ``RHOHV`` and ``FALL_SPEED`` (m s-1, positive
        downward), the linear moments ``Z_H``, ``Z_V``, ``Z_HV_REAL``,
        ``Z_HV_IMAG`` (mm6 m-3) and ``ZH_VT`` that :func:`to_radar_grid`
        interpolates, and the ``HEIGHT``, ``U``, ``V``, ``W`` of the state if
        it has them. The input is not changed.

    Raises
    ------
    KeyError
        If a named variable is not in the state.
    ValueError
        For no hydrometeor class, an unknown option, a missing air density or
        incomplete ``graupel_params``.
    TypeError
        If ``state`` is not a Dataset.
    ImportError
        If ``engine="compiled"`` and the kernel is not available.

    Notes
    -----
    The equations, the choices of radarx and what is left out (melting
    classes, attenuation, non-zero elevation, resonance) are in
    :mod:`radarx.retrieve.forward_operator`. No WRF output could be read on
    the development machine; the operator is checked against closed forms,
    :func:`radarx.retrieve.radar_from_dsd` and the published amplitudes.

    References
    ----------
    Jung, Y., G. Zhang, and M. Xue, 2008: Assimilation of simulated
    polarimetric radar data for a convective storm using the ensemble Kalman
    filter. Part I: Observation operators for reflectivity and polarimetric
    variables. *Mon. Wea. Rev.*, **136** (6), 2228-2245,
    https://doi.org/10.1175/2007MWR2083.1

    Ryzhkov, A. V., J. Snyder, J. T. Carlin, A. Khain, and M. Pinsky, 2020:
    What polarimetric weather radars offer to cloud modelers: Forward radar
    operators and microphysical/thermodynamic retrievals. *Atmosphere*,
    **11** (4), 362, https://doi.org/10.3390/atmos11040362

    Maxwell Garnett, J. C., 1904: Colours in metal glasses and in metallic
    films. *Philos. Trans. R. Soc. Lond. A*, **203**, 385-420,
    https://doi.org/10.1098/rsta.1904.0024

    Brandes, E. A., G. Zhang, and J. Vivekanandan, 2002: Experiments in
    rainfall estimation with a polarimetric radar in a subtropical
    environment. *J. Appl. Meteor.*, **41** (6), 674-685,
    https://doi.org/10.1175/1520-0450(2002)041<0674:EIREWA>2.0.CO;2

    Mishchenko, M. I., and L. D. Travis, 1998: Capabilities and limitations of
    a current FORTRAN implementation of the T-matrix method for randomly
    oriented, rotationally symmetric scatterers. *J. Quant. Spectrosc.
    Radiat. Transfer*, **60** (3), 309-324,
    https://doi.org/10.1016/S0022-4073(98)00008-9

    Atlas, D., R. C. Srivastava, and R. S. Sekhon, 1973: Doppler radar
    characteristics of precipitation at vertical incidence. *Rev. Geophys.*,
    **11** (1), 1-35, https://doi.org/10.1029/RG011i001p00001

    Examples
    --------
    >>> state = wrf_state(xr.open_dataset("wrfout_d03.nc"))  # doctest: +SKIP
    >>> fields = forward_operator(state, nrain="QNRAIN", mu_rain=2.0)  # doctest: +SKIP
    >>> fields = forward_operator(state, graupel="QGRAUP", graupel_params={
    ...     "n0": 4e6, "rho": 400.0, "axis_ratio": 0.8, "sigma": 40.0})  # doctest: +SKIP
    """
    if not isinstance(state, xr.Dataset):
        raise TypeError("state must be an xarray Dataset")
    band = _dsd._check_band(band)
    compiled = choose_engine(engine, HAS_COMPILED_KERNEL, "forward operator")
    if not mu_rain > -1.0:
        raise ValueError(f"mu_rain must be greater than -1, not {mu_rain!r}")
    if not n0_rain > 0.0:
        raise ValueError(f"n0_rain must be positive, not {n0_rain!r}")
    if not min_mixing_ratio >= 0.0:
        raise ValueError("min_mixing_ratio must not be negative")
    if not 0.0 <= rain_temperature <= 30.0:
        raise ValueError("rain_temperature must be within 0-30 degrees Celsius")

    q_rain = _choose(state, rain, "rain", "rain")
    n_rain = _choose(state, nrain, "nrain", "nrain")
    ice = []
    for label, value, params, defaults in (
        ("snow", snow, snow_params, SNOW_PARAMS),
        ("hail", hail, hail_params, HAIL_PARAMS),
        ("graupel", graupel, graupel_params, None),
    ):
        da = _choose(state, value, label, label)
        if da is not None:
            ice.append((label, da, _check_ice(label, params, defaults)))
    if q_rain is None and not ice:
        raise ValueError("no hydrometeor class: the state has none of the named fields")
    rho = _air_density(state, density)
    temp = _choose(state, temperature, "temperature", "temperature")

    tab = _rain_table(band)
    wl = tab["wavelength"]
    ice_par = np.zeros((len(ice), 7))
    for c, (_, _, p) in enumerate(ice):
        ice_par[c] = (p["n0"], p["rho"], *_ice_constants(p, wl), p["fall_speed"])
    spec = {
        "rain": q_rain is not None,
        "nrain": n_rain is not None and q_rain is not None,
        "temperature": temp is not None and q_rain is not None,
        "n_ice": len(ice),
        "table": tab,
        "kfac": wl**4 / (np.pi**5 * _dsd.KW2),
        "n0_rain": float(n0_rain),
        "mu_rain": float(mu_rain),
        "q_min": float(min_mixing_ratio),
        "temp_default": float(rain_temperature),
        "nr_per_kg": bool(nrain_per_kg),
        "ice_par": np.ascontiguousarray(ice_par),
    }
    arrays = [rho]
    if q_rain is not None:
        arrays.append(q_rain)
        if spec["nrain"]:
            arrays.append(n_rain)
        if spec["temperature"]:
            arrays.append(temp - 273.15)
    arrays.extend(da for _, da, _ in ice)

    reference = q_rain if q_rain is not None else ice[0][1]
    moments = xr.apply_ufunc(
        _kernel_call(compiled, n_threads, spec),
        *[a.copy(deep=False) for a in arrays],
        dask="parallelized",
        output_core_dims=[[]] * 6,
        output_dtypes=[np.float32] * 6,
        keep_attrs=False,
    )
    out = xr.Dataset(
        {
            name: da.transpose(*reference.dims, ...)
            for name, da in zip(_MOMENTS_ORDER, moments)
        }
    )
    zh, zv = out["Z_H"], out["Z_V"]
    echo = zh > 10.0 ** (0.1 * min_dbz)
    zh_e, zv_e = zh.where(echo), zv.where(echo)  # NaN, not 0, where there is no echo
    products = {
        "DBZH": 10.0 * np.log10(zh_e),
        "ZDR": 10.0 * np.log10(zh_e / zv_e),
        "RHOHV": np.hypot(out["Z_HV_REAL"], out["Z_HV_IMAG"]).where(echo)
        / np.sqrt(zh_e * zv_e),
        "FALL_SPEED": out["ZH_VT"] / zh_e,
    }
    result = xr.Dataset(
        {
            "DBZH": products["DBZH"],
            "ZDR": products["ZDR"],
            "KDP": out["KDP"],
            "RHOHV": products["RHOHV"].clip(max=1.0),
            "FALL_SPEED": products["FALL_SPEED"],
            "Z_H": zh,
            "Z_V": zv,
            "Z_HV_REAL": out["Z_HV_REAL"],
            "Z_HV_IMAG": out["Z_HV_IMAG"],
            "ZH_VT": out["ZH_VT"],
        }
    ).astype(np.float32)
    for name, attrs in _ATTRS.items():
        if name in result:
            result[name].attrs = dict(attrs)
    for name in _CARRIED:
        if name in state:
            result[name] = state[name]
    result.attrs = {
        "forward_operator_band": band,
        "forward_operator_classes": ", ".join(
            (["rain"] if q_rain is not None else []) + [c[0] for c in ice]
        ),
    }
    return result


# --------------------------------------------------------------------------
# sampling at the gates of a radar
# --------------------------------------------------------------------------

_LAT_NAMES = ("latitude", "lat", "XLAT")
_LON_NAMES = ("longitude", "lon", "XLONG")
_VERTICAL_NAMES = ("bottom_top", "nVertLevels", "lev", "level", "z")
_SAMPLED = ("Z_H", "Z_V", "Z_HV_REAL", "Z_HV_IMAG", "KDP", "ZH_VT")
_EDGE_FACTOR = 1.5  # a column further than this many spacings is not a neighbour
_AZIMUTH = "azimuth"
_RANGE = "range"


def _first(ds, names, what):
    for name in names:
        if name in ds.variables:
            return ds[name]
    raise KeyError(f"no {what} found in the fields (tried {names})")


def _unit_vectors(lat, lon):
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    return np.stack(
        [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1
    )


def _arc(chord):
    return EARTH_RADIUS * 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))


def _vertical_dim(height, horizontal, vertical_dim):
    if vertical_dim is not None:
        if vertical_dim not in height.dims:
            raise ValueError(f"HEIGHT has no dimension {vertical_dim!r}")
        return vertical_dim
    left = [d for d in height.dims if d not in horizontal]
    for name in _VERTICAL_NAMES:
        if name in left:
            return name
    if len(left) == 1:
        return left[0]
    raise ValueError(
        f"cannot tell the vertical dimension of HEIGHT {height.dims}; pass vertical_dim"
    )


def _columns(lat, lon):
    """KD-tree of the model columns and the spacing of each column (m)."""
    ll = _unit_vectors(
        np.asarray(lat.values, dtype=np.float64).ravel(),
        np.asarray(lon.values, dtype=np.float64).ravel(),
    )
    tree = cKDTree(ll)
    spacing = _arc(tree.query(ll, k=2)[0][:, 1]) if len(ll) > 1 else np.full(1, np.inf)
    return tree, spacing


def _neighbours(tree, spacing, gate_lat, gate_lon, neighbors, max_distance):
    """Column indices and inverse-distance weights of the gates (ng, neighbors)."""
    k = min(int(neighbors), tree.n)
    chord, idx = tree.query(_unit_vectors(gate_lat.ravel(), gate_lon.ravel()), k=k)
    chord, idx = chord.reshape(-1, k), idx.reshape(-1, k)
    dist = _arc(chord)
    limit = _EDGE_FACTOR * spacing[idx] if max_distance is None else max_distance
    weight = np.where(dist <= limit, 1.0 / np.maximum(dist, 1.0), 0.0)
    return np.ascontiguousarray(idx, dtype=np.int64), np.ascontiguousarray(weight)


def _sample_block(sampler, nh, naz, nrng):
    """Block function of ``apply_ufunc``: sample every field at the gates."""

    def run(*arrs):
        nf = len(arrs) - 1
        bshape = tuple(np.broadcast_shapes(*[a.shape[: a.ndim - nh - 1] for a in arrs]))
        out = np.full((nf,) + bshape + (naz * nrng,), np.nan, dtype=np.float32)

        def take(a, b):
            # a field without some of the leading dimensions broadcasts
            nb = a.ndim - nh - 1
            sel = tuple(
                0 if s == 1 else i for s, i in zip(a.shape[:nb], b[len(b) - nb :])
            )
            return a[sel].reshape(-1, a.shape[-1])

        for b in np.ndindex(*bshape):
            res = sampler([take(a, b) for a in arrs[:-1]], take(arrs[-1], b))
            out[(slice(None),) + b] = res
        out = out.reshape((nf,) + bshape + (naz, nrng))
        return tuple(out[j] for j in range(nf))

    return run


def _sweep_fields(fields, args, vertical, horizontal, wanted):
    """The moments of one sweep, sampled at its gates, by name."""
    idx, weight, gate_h, naz, nrng, compiled, n_threads = args

    def sampler(arrays, height):
        if compiled:
            return _forward_operator.sample(
                arrays, height, idx, weight, gate_h, int(n_threads or 0)
            )
        return _sample_numpy(arrays, height, idx, weight, gate_h)

    core = list(horizontal) + [vertical]
    arrays = [fields[n].copy(deep=False) for n in list(wanted) + ["HEIGHT"]]
    arrays = [
        a.chunk({d: -1 for d in core if d in a.dims}) if a.chunks is not None else a
        for a in arrays
    ]
    sampled = xr.apply_ufunc(
        _sample_block(sampler, len(horizontal), naz, nrng),
        *arrays,
        input_core_dims=[core] * len(arrays),
        output_core_dims=[[_AZIMUTH, _RANGE]] * len(wanted),
        dask="parallelized",
        output_dtypes=[np.float32] * len(wanted),
        dask_gufunc_kwargs={
            "output_sizes": {_AZIMUTH: naz, _RANGE: nrng},
            "allow_rechunk": True,
        },
        keep_attrs=False,
    )
    return dict(zip(wanted, sampled))


def _sweep_products(sampled, azimuth_da, range_da, elevation, site, thr, velocity):
    """DBZH, ZDR, KDP, RHOHV and VRADH of one sweep from the sampled moments."""
    echo = sampled["Z_H"] > thr
    zh, zv = sampled["Z_H"].where(echo), sampled["Z_V"].where(echo)
    el = np.deg2rad(elevation)
    products = {
        "DBZH": 10.0 * np.log10(zh),
        "ZDR": 10.0 * np.log10(zh / zv),
        "KDP": sampled["KDP"],
        "RHOHV": (
            np.hypot(sampled["Z_HV_REAL"], sampled["Z_HV_IMAG"]) / np.sqrt(zh * zv)
        ).clip(max=1.0),
    }
    if velocity:
        phi = np.deg2rad(azimuth_da)
        fall = sampled["ZH_VT"] / zh
        w = sampled["W"] if "W" in sampled else 0.0
        products["VRADH"] = (
            sampled["U"] * np.sin(phi) + sampled["V"] * np.cos(phi)
        ) * np.cos(el) + (w - fall) * np.sin(el)
    ds = xr.Dataset({k: v.where(echo).astype(np.float32) for k, v in products.items()})
    for name in ds.data_vars:
        ds[name].attrs = dict(_ATTRS[name])
    ds = ds.assign_coords(
        {
            _AZIMUTH: azimuth_da,
            _RANGE: range_da,
            "elevation": (_AZIMUTH, np.full(azimuth_da.size, float(elevation))),
            "latitude": site[0],
            "longitude": site[1],
            "altitude": site[2],
        }
    )
    ds["elevation"].attrs = {"units": "degree"}
    ds["latitude"].attrs = {"units": "degrees_north"}
    ds["longitude"].attrs = {"units": "degrees_east"}
    ds["altitude"].attrs = {"units": "m"}
    return ds.transpose(..., _AZIMUTH, _RANGE)


@renamed_arguments(fields="obj")
@provenance(
    "Sampling of the model radar variables along the beam of a radar (4/3 "
    "effective Earth radius, inverse-distance neighbour columns, linear in height)",
    extra_refs=("doviak-zrnic-1993",),
)
def to_radar_grid(
    obj,
    site,
    elevation,
    *,
    azimuth=None,
    gate_range=None,
    k_e=4.0 / 3.0,
    neighbors=4,
    max_distance=None,
    vertical_dim=None,
    min_dbz=-30.0,
    n_threads=None,
    engine="auto",
):
    """
    Sample the model radar variables on the polar grid of a radar.

    Parameters
    ----------
    obj : xarray.Dataset
        The result of :func:`forward_operator` of a state with ``HEIGHT`` and
        the coordinates ``latitude`` and ``longitude`` (also ``XLAT`` and
        ``XLONG``), and ``U``, ``V`` (and ``W``, taken as 0 if absent) for the
        radial velocity.
    site : tuple of float
        ``(latitude, longitude, altitude)`` of the radar: degrees, degrees,
        m above sea level.
    elevation : float or sequence of float
        Elevation angle(s) in degrees. A sequence gives a volume.
    azimuth : array-like, optional
        Azimuths (degrees clockwise from north). Default 0.5, 1.5, ..., 359.5.
    gate_range : array-like, optional
        Slant ranges of the gates (m). Default 250 m to 150 km every 250 m.
    k_e : float, optional
        Effective Earth radius factor. Default 4/3.
    neighbors : int, optional
        Model columns around a gate that are averaged with inverse-distance
        weights. Default 4.
    max_distance : float, optional
        A column further than this (m) from a gate is not used. Default: 1.5
        times the distance of the column to its nearest column (so gates
        outside the model domain are NaN).
    vertical_dim : str, optional
        The vertical dimension of ``HEIGHT``; found among ``bottom_top``,
        ``nVertLevels``, ``lev``, ``level`` and ``z`` by default.
    min_dbz : float, optional
        Gates with a reflectivity below this (dBZ) have no echo and are NaN
        in every variable. Default -30.
    n_threads : int, optional
        Threads of the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation; ``"auto"`` prefers the compiled kernel.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        For one elevation a sweep with ``DBZH``, ``ZDR``, ``KDP``,
        ``RHOHV`` and ``VRADH`` (m s-1, positive away from the radar) on
        (``azimuth``, ``range``) and the dimensions of ``obj`` that are
        not spatial (``Time``), the coordinates ``elevation`` (per ray),
        ``latitude``, ``longitude`` and ``altitude`` of the site. For several
        elevations a DataTree with the site in the root and one such
        ``sweep_<n>`` node each. NaN where the gate is outside the model
        (horizontally or vertically, below the lowest model level) or has no
        echo. Dask arrays give a lazy result. The input is not changed.

    Raises
    ------
    KeyError
        If the fields lack a moment of :func:`forward_operator`, ``HEIGHT``
        or the coordinates.
    ValueError
        For an invalid site, elevation, ranges or options.
    TypeError
        If ``obj`` is not a Dataset.
    ImportError
        If ``engine="compiled"`` and the kernel is not available.

    Notes
    -----
    The gate at the slant range ``r`` and elevation ``e`` is at the height
    ``sqrt(r^2 + (k_e a + h0)^2 + 2 r (k_e a + h0) sin e) - k_e a`` above sea
    level and at the ground distance ``k_e a arcsin(r cos e / (k_e a + h))``
    along the great circle of the azimuth (Doviak and Zrnic 1993, Eqs. 2.28),
    ``h0`` being the site altitude and ``a`` 6371 km. The beam centre is
    sampled; the beam pattern is not integrated. The linear moments are
    interpolated and the logarithmic variables are formed afterwards. See
    :mod:`radarx.retrieve.forward_operator`.

    References
    ----------
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
       Observations*, 2nd ed. Academic Press (book, no DOI). Eqs. (2.28).
    .. [2] Jung, Y., G. Zhang, and M. Xue, 2008: Assimilation of simulated
       polarimetric radar data for a convective storm using the ensemble
       Kalman filter. Part I: Observation operators for reflectivity and
       polarimetric variables. *Mon. Wea. Rev.*, **136** (6), 2228-2245,
       https://doi.org/10.1175/2007MWR2083.1

    Examples
    --------
    >>> fields = forward_operator(wrf_state(ds))  # doctest: +SKIP
    >>> sweep = to_radar_grid(fields, (35.33, -97.28, 390.0), 0.5)  # doctest: +SKIP
    >>> volume = fields.radarx.to_radar_grid((35.33, -97.28, 390.0), [0.5, 1.5])  # doctest: +SKIP
    """
    if not isinstance(obj, xr.Dataset):
        raise TypeError("fields must be an xarray Dataset")
    compiled = choose_engine(engine, HAS_COMPILED_KERNEL, "forward operator")
    site = tuple(float(v) for v in site)
    if len(site) != 3 or not all(np.isfinite(site)) or not -90.0 <= site[0] <= 90.0:
        raise ValueError(
            "site must be (latitude, longitude, altitude) of finite numbers"
        )
    scalar = np.ndim(elevation) == 0
    elevations = np.atleast_1d(np.asarray(elevation, dtype=np.float64))
    if elevations.size == 0 or not np.all(np.abs(elevations) < 90.0):
        raise ValueError("elevation must be one or more angles within +-90 degrees")
    az = np.asarray(
        np.arange(0.5, 360.0, 1.0) if azimuth is None else azimuth, dtype=np.float64
    )
    rng = np.asarray(
        np.arange(250.0, 150001.0, 250.0) if gate_range is None else gate_range,
        dtype=np.float64,
    )
    if az.ndim != 1 or az.size == 0 or not np.all(np.isfinite(az)):
        raise ValueError("azimuth must be a non-empty 1-D array of finite values")
    if (
        rng.ndim != 1
        or rng.size == 0
        or not (np.all(np.isfinite(rng)) and rng.min() > 0)
    ):
        raise ValueError("gate_range must be a non-empty 1-D array of positive values")
    if not (np.isfinite(k_e) and k_e > 0):
        raise ValueError(f"k_e must be positive, not {k_e!r}")
    if int(neighbors) != neighbors or neighbors < 1:
        raise ValueError(f"neighbors must be a positive integer, not {neighbors!r}")
    if max_distance is not None and not max_distance > 0:
        raise ValueError("max_distance must be positive")
    missing = [n for n in _SAMPLED + ("HEIGHT",) if n not in obj]
    if missing:
        raise KeyError(
            f"the fields lack {missing}: pass the result of forward_operator of a "
            "state with HEIGHT"
        )
    lat = _first(obj, _LAT_NAMES, "latitude")
    lon = _first(obj, _LON_NAMES, "longitude")
    horizontal = tuple(lat.dims)
    if lon.dims != lat.dims:
        raise ValueError("latitude and longitude must have the same dimensions")
    vertical = _vertical_dim(obj["HEIGHT"], horizontal, vertical_dim)
    velocity = "U" in obj and "V" in obj
    wanted = list(_SAMPLED) + (["U", "V"] if velocity else [])
    if velocity and "W" in obj:
        wanted.append("W")
    tree, spacing = _columns(lat, lon)
    thr = 10.0 ** (0.1 * min_dbz)
    reff = k_e * EARTH_RADIUS
    geo = {
        "geographic": True,
        "site_a": site[0],
        "site_b": site[1],
        "earth_radius": EARTH_RADIUS,
        "lon_center": site[1],
    }
    azimuth_da = xr.DataArray(
        az, dims=_AZIMUTH, coords={_AZIMUTH: az}, attrs={"units": "degree"}
    )
    range_da = xr.DataArray(
        rng, dims=_RANGE, coords={_RANGE: rng}, attrs={"units": "m"}
    )
    sweeps = []
    for el in elevations:
        gate_h = beam_center_height(rng, el, radar_height=site[2], reff=reff)
        s = ground_range(rng, el, reff=reff, radar_height=site[2])
        gate_lat, gate_lon = _locate(s[None, :], az[:, None], geo)
        idx, weight = _neighbours(
            tree, spacing, gate_lat, gate_lon, neighbors, max_distance
        )
        height = np.ascontiguousarray(
            np.broadcast_to(gate_h[None, :], (az.size, rng.size)).ravel()
        )
        args = (idx, weight, height, az.size, rng.size, compiled, n_threads)
        sampled = _sweep_fields(obj, args, vertical, horizontal, wanted)
        sweeps.append(
            _sweep_products(sampled, azimuth_da, range_da, el, site, thr, velocity)
        )
    if scalar:
        return sweeps[0]
    nodes = {
        "/": xr.Dataset(
            coords={"latitude": site[0], "longitude": site[1], "altitude": site[2]}
        )
    }
    for number, ds in enumerate(sweeps):
        nodes[f"sweep_{number}"] = ds.drop_vars(["latitude", "longitude", "altitude"])
    return xr.DataTree.from_dict(nodes)


# --------------------------------------------------------------------------
# accessors
# --------------------------------------------------------------------------


@accessor_method("dataset", name="forward_operator", wraps=forward_operator)
def _forward_operator_accessor(self):
    """
    Polarimetric radar variables of a model state on the model grid.

    See :func:`radarx.retrieve.forward_operator` for the parameters.

    Returns
    -------
    xarray.Dataset
        ``DBZH``, ``ZDR``, ``KDP``, ``RHOHV``, ``FALL_SPEED`` and the linear
        moments.
    """


@accessor_method("dataset", name="to_radar_grid", wraps=to_radar_grid)
def _to_radar_grid_accessor(self):
    """
    Sample the model radar variables on the polar grid of a radar.

    See :func:`radarx.retrieve.to_radar_grid` for the parameters.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``DBZH``, ``ZDR``, ``KDP``, ``RHOHV`` and ``VRADH`` on the gates of
        the sweep (a DataTree of sweeps for several elevations).
    """
