#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Disdrometer Analysis
====================

Quality control, drop size distributions, gamma fits, radar variables and
radar matching for laser disdrometer (OTT Parsivel / Parsivel2) data in the
format of :mod:`radarx.io.disdrometer`: ``counts`` of particles on
``(time, velocity, diameter)`` classes.

Workflow
--------
1. :func:`disdrometer_qc` removes particles that are not plausible
   raindrops, or :func:`raupach_berne_correction` applies the correction of
   Raupach and Berne (2015);
2. :func:`number_concentration` converts counts to N(D);
3. :func:`dsd_moments` gives the integral quantities of the measured
   spectra, :func:`fit_gamma` fits gamma DSDs by moments or truncated
   moments, and :func:`radarx.retrieve.radar_from_dsd` gives radar
   variables from the T-matrix tables of :mod:`radarx.retrieve.dsd`;
4. :func:`match_radar` pairs the disdrometer with the radar gate above it.

:func:`process_disdrometer` (also ``ds.radarx.disdrometer()``) runs steps 1-3.

Fall speed
----------
The terminal fall speed of raindrops is that of Atlas et al. (1973),
:math:`v_t(D) = 9.65 - 10.3 e^{-0.6 D}` m s\\ :sup:`-1` (:math:`D` in mm,
zero below 0.08 mm) at the sea-level air density
:math:`\\rho_0 = 1.204` kg m\\ :sup:`-3`, scaled by
:math:`(\\rho_0/\\rho)^{0.4}` (Foote and du Toit 1969) where the station
records pressure and temperature (:func:`terminal_fall_speed`).

- The Atlas et al. (1973) law is quoted as Eq. 7.65b of Bringi and
  Chandrasekar (2001), where it is the sea-level fit to the Gunn and Kinzer
  (1949) measurements. It is negative below 0.109 mm (clipped to zero
  here). Its range of validity in the original paper is not checked; radarx
  uses it for all classes up to the maximum diameter of the quality control.
- The factor :math:`(\\rho_0/\\rho)^{0.4}` is the correction attributed to
  Foote and du Toit (1969), as quoted by Li and Srivastava (2001, after
  their Eq. 4) and Kumjian and Ryzhkov (2010, Eq. 3). The equation number and
  the range of validity in the original paper are not checked. :math:`\\rho_0` = 1.204 kg m\\ :sup:`-3` is the density of dry air at
  1013.25 hPa and 20 °C, taken as the density of the sea-level law (radarx
  choice).
- Raupach and Berne (2015, Sect. 5.1) used the terminal velocities of Beard
  (1976) as the reference of their velocity correction and of their filter
  (Eqs. 9-11), and the factors of their Tables 3 and 10 were trained with
  that reference. radarx substitutes the Atlas et al. law (a deviation from
  the paper), which differs from Beard (1976) by a few per cent for 1-5 mm
  (not quantified here).

Quality control
---------------
:func:`disdrometer_qc` keeps the (velocity, diameter) classes whose centre
velocity :math:`V` is within a relative ``tolerance`` of the fall speed,
:math:`|V - v_t(D)| \\le` ``tolerance`` :math:`v_t(D)` (default 60 %).
Particles far from the fall speed-diameter relation are mostly not single
raindrops falling through the beam: margin fallers, splashing drops, and
the strong-wind artifact of Friedrich et al. (2013), large (> 5 mm) and
slow (< 1 m s\\ :sup:`-1`) particles caused by particles crossing the beam
at an angle. Friedrich et al. (2013) also removed drops larger than 8 mm
(``max_diameter``). With ``method="raupach2015"`` the absolute filter of
Raupach and Berne (2015, their Eqs. 9-11) is used instead: particles are
removed if :math:`D > 7.5` mm, :math:`V > v_t(D) + 4` or
:math:`V < v_t(D) - 3` m s\\ :sup:`-1` (their Eqs. 9-11, as printed;
:math:`v_t` is the Atlas et al. law here, see above). The default relative
tolerance of 60 % of the ``"relative"`` filter is a radarx choice, not taken
from a paper. The two smallest size classes, which the Parsivel does not
measure, are removed (Raupach and Berne 2015 also ignore them: their tables
start at class 3), and records with winds above ``max_wind`` can be
discarded.

Raupach and Berne (2015) correction
-----------------------------------
:func:`raupach_berne_correction` shifts the velocities of each size class
so that their mean matches the terminal fall speed (the counts are split
into 0.1 m s\\ :sup:`-1` sub-classes, shifted and regrouped), applies the
filter above and multiplies N(D) by the per-class correction factors that
Raupach and Berne (2015) calibrated against a 2D video disdrometer for
classes of the Parsivel rain intensity (their Table 3 for the first
generation Parsivel, from the SOP2013 campaign, and Table 10 for Parsivel2,
from HyMeX 2013; the numbers in this module agree with both tables). The velocity shift is their Sect. 5.1 (classes subsampled to 0.1 m
s\\ :sup:`-1`, shifted so that the mean velocity equals the terminal velocity,
regrouped) and the concentration factors :math:`P(i)` their Sect. 5.2.
Classes without a factor are not corrected. The factors were trained in the
Cévennes (France); their transferability to other climates is limited
(Raupach and Berne 2015, Sect. 8).

Drop size distribution
----------------------
With :math:`C_{v,i}` particles in velocity class :math:`v` and size class
:math:`i` during :math:`\\Delta t` (Raupach and Berne 2015, Eqs. 5-6),

.. math::

    N(D_i) = \\frac{1}{S_i \\Delta D_i \\Delta t} \\sum_v \\frac{C_{v,i}}{V_v},
    \\qquad S_i = 10^{-6} L (B - D_i / 2),

with the beam length :math:`L = 180` mm and width :math:`B = 30` mm (the
values of Raupach and Berne 2015, Sect. 4, who give :math:`S_i` as their
Eq. 5 and :math:`N` as their Eq. 6 and attribute the sampling area to
Löffler-Mang and Joss 2000 and Battaglia et al. 2010): drops only partly
inside the beam are discarded by the instrument, which shrinks the sampling
area of large drops. ``velocity="terminal"`` uses :math:`v_t(D_i)` instead
of the measured :math:`V_v` (a radarx option, not in the paper).

Gamma fits
----------
:func:`fit_gamma` fits :math:`N(D) = N_0 D^{\\mu} e^{-\\Lambda D}` from three
moments :math:`M_n = \\sum_i N(D_i) D_i^n \\Delta D_i` of orders
:math:`i < j < k` (e.g. 2-4-6 or 3-4-6). For the untruncated gamma DSD,
:math:`M_n = N_0 \\Gamma(\\mu + n + 1) / \\Lambda^{\\mu + n + 1}`, the ratio
:math:`M_j^{k-i} / (M_i^{k-j} M_k^{j-i})` depends on :math:`\\mu` only and
is solved for it, then :math:`\\Lambda` and :math:`N_0` follow (method of
moments; Ulbrich and Atlas 1998, Cao and Zhang 2009; the closed-form ratio is
derived from the moments of the gamma DSD, not copied from a paper, and
the bounds and tolerances below are radarx numerical choices). With
``truncated=True`` the moments are those of the gamma DSD truncated at the
largest observed diameter :math:`D_{max}` (upper edge of the largest class
with drops), following the idea of the truncated moments of Ulbrich and
Atlas (1998) and the truncated moment fit used by Cao et al. (2008) and
Vivekanandan et al. (2004) (the way of solving the two log moment ratios
for :math:`\\mu` and :math:`\\ln\\Lambda` by a damped Newton iteration is
radarx's own; the truncation and solution in those papers are not compared),
and
with ``lower_truncation=True`` also at the smallest, :math:`D_{min}` (lower
edge of the smallest class with drops; else :math:`D_{min} = 0`):

.. math::

    M_n = N_0 \\frac{\\Gamma(\\mu + n + 1)}{\\Lambda^{\\mu + n + 1}}
    \\left[P(\\mu + n + 1, \\Lambda D_{max}) - P(\\mu + n + 1, \\Lambda D_{min})\\right],

with :math:`P` the regularized incomplete gamma function. The two log
moment ratios :math:`\\ln(M_j/M_i)` and :math:`\\ln(M_k/M_j)` are solved for
:math:`\\mu` and :math:`\\ln\\Lambda` by a damped Newton iteration from the
untruncated fit, to a residual of ``tol``. Truncated fits can have
:math:`\\mu < -1` (their moments stay finite); spectra without a solution
with :math:`10^{-3} < \\Lambda < 10^3` mm\\ :sup:`-1` (e.g. a few
neighbouring small classes) are NaN.

The compiled C++ kernel processes all spectra at once on all cores; an
identical NumPy implementation is the fallback and the test reference.

References
----------
Atlas, D., R. C. Srivastava, and R. S. Sekhon, 1973: Doppler radar
characteristics of precipitation at vertical incidence. *Rev. Geophys.*,
**11** (1), 1-35, https://doi.org/10.1029/RG011i001p00001

Foote, G. B., and P. S. du Toit, 1969: Terminal velocity of raindrops aloft.
*J. Appl. Meteor.*, **8** (2), 249-253,
https://doi.org/10.1175/1520-0450(1969)008<0249:TVORA>2.0.CO;2

Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler Weather
Radar: Principles and Applications*. Cambridge University Press, 636 pp.,
https://doi.org/10.1017/CBO9780511541094

Gunn, R., and G. D. Kinzer, 1949: The terminal velocity of fall for water
droplets in stagnant air. *J. Meteor.*, **6** (4), 243-248,
https://doi.org/10.1175/1520-0469(1949)006<0243:TTVOFF>2.0.CO;2

Li, X., and R. C. Srivastava, 2001: An analytical solution for raindrop
evaporation and its application to radar rainfall measurements. *J. Appl.
Meteor.*, **40** (9), 1607-1616,
https://doi.org/10.1175/1520-0450(2001)040<1607:AASFRE>2.0.CO;2

Kumjian, M. R., and A. V. Ryzhkov, 2010: The impact of evaporation on
polarimetric characteristics of rain: Theoretical model and practical
implications. *J. Appl. Meteor. Climatol.*, **49** (6), 1247-1267,
https://doi.org/10.1175/2010JAMC2243.1

Vivekanandan, J., G. Zhang, and E. Brandes, 2004: Polarimetric radar
estimators based on a constrained gamma drop size distribution model. *J.
Appl. Meteor.*, **43** (2), 217-230,
https://doi.org/10.1175/1520-0450(2004)043<0217:PREBOA>2.0.CO;2

Cao, Q., G. Zhang, E. Brandes, T. Schuur, A. Ryzhkov, and K. Ikeda, 2008:
Analysis of video disdrometer and polarimetric radar data to characterize
rain microphysics in Oklahoma. *J. Appl. Meteor. Climatol.*, **47** (8),
2238-2255, https://doi.org/10.1175/2008JAMC1732.1

Battaglia, A., E. Rustemeier, A. Tokay, U. Blahak, and C. Simmer, 2010:
PARSIVEL snow observations: A critical assessment. *J. Atmos. Oceanic
Technol.*, **27** (2), 333-344, https://doi.org/10.1175/2009JTECHA1332.1

Beard, K. V., 1976: Terminal velocity and shape of cloud and precipitation
drops aloft. *J. Atmos. Sci.*, **33** (5), 851-864,
https://doi.org/10.1175/1520-0469(1976)033<0851:TVASOC>2.0.CO;2

Ulbrich, C. W., and D. Atlas, 1998: Rainfall microphysics and radar
properties: Analysis methods for drop size spectra. *J. Appl. Meteor.*,
**37** (9), 912-923,
https://doi.org/10.1175/1520-0450(1998)037<0912:RMARPA>2.0.CO;2

Löffler-Mang, M., and J. Joss, 2000: An optical disdrometer for measuring
size and velocity of hydrometeors. *J. Atmos. Oceanic Technol.*, **17** (2),
130-139, https://doi.org/10.1175/1520-0426(2000)017<0130:AODFMS>2.0.CO;2


Cao, Q., and G. Zhang, 2009: Errors in estimating raindrop size
distribution parameters employing disdrometer and simulated raindrop
spectra. *J. Appl. Meteor. Climatol.*, **48** (2), 406-425,
https://doi.org/10.1175/2008JAMC2026.1

Friedrich, K., S. Higgins, F. J. Masters, and C. R. Lopez, 2013:
Articulating and stationary PARSIVEL disdrometer measurements in conditions
with strong winds and heavy rainfall. *J. Atmos. Oceanic Technol.*, **30**
(9), 2063-2080, https://doi.org/10.1175/JTECH-D-12-00254.1

Raupach, T. H., and A. Berne, 2015: Correction of raindrop size
distributions measured by Parsivel disdrometers, using a two-dimensional
video disdrometer as a reference. *Atmos. Meas. Tech.*, **8** (1), 343-365,
https://doi.org/10.5194/amt-8-343-2015

.. autosummary::
   :nosignatures:
   :toctree: generated/

   terminal_fall_speed
   disdrometer_qc
   raupach_berne_correction
   number_concentration
   dsd_moments
   fit_gamma
   process_disdrometer
   radar_at_location
   match_radar
"""

from __future__ import annotations

__all__ = [
    "terminal_fall_speed",
    "disdrometer_qc",
    "raupach_berne_correction",
    "number_concentration",
    "dsd_moments",
    "fit_gamma",
    "process_disdrometer",
    "radar_at_location",
    "match_radar",
]

import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import gammainc, gammaincc, gammaincinv, gammaln

from .._registry import accessor_method
from ..fundamentals.geometry import beam_center_height
from . import dsd as _dsdmod

try:
    from . import _disdrometer

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _disdrometer = None
    HAS_COMPILED_KERNEL = False

# kg m-3, air density of the sea-level fall speeds: dry air at 1013.25 hPa and
# 20 degC (radarx choice, as in evaporation.py)
RHO0 = 1.204
_ACCEPT = 1e-7  # residual accepted when the truncated Newton fit stalls
_BISECT = 64  # bisection steps of the untruncated fit
_LOG_LAM_MAX = np.log(1e3)  # truncated fits: Lambda within 1e-3 .. 1e3 mm-1
# Parsivel laser beam, L = 180 mm and B = 30 mm: Raupach and Berne (2015), Sect. 4
BEAM_LENGTH = 180.0  # mm
BEAM_WIDTH = 30.0  # mm
EARTH_RADIUS = 6371000.0  # m

# Raupach and Berne (2015) concentration correction factors P(i) by class of
# Parsivel rain intensity (mm h-1), for the size classes i = 3, 4, ... (1-based);
# None: no correction (blank entries of the tables). Transcribed from their
# Table 3 (first-generation Parsivel, SOP2013, classes 3-21) and Table 10
# (Parsivel2, HyMeX 2013, classes 3-22) and checked against both tables.
_RB_INTENSITY = {
    "parsivel": [0.0, 0.5, 1.0, 2.0, 200.0],  # Table 3
    "parsivel2": [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 200.0],  # Table 10
}
_N = None
_RB_FACTORS = {
    "parsivel": [  # classes 3-21
        [0.05, 0.06, 0.09, 0.12],
        [0.12, 0.15, 0.24, 0.28],
        [0.38, 0.44, 0.63, 0.66],
        [0.48, 0.54, 0.71, 0.85],
        [0.70, 0.77, 0.95, 1.13],
        [0.73, 0.74, 0.97, 1.09],
        [0.84, 0.84, 1.03, 1.26],
        [0.90, 0.84, 1.04, 1.27],
        [0.84, 0.81, 1.00, 1.21],
        [0.75, 0.71, 0.88, 1.03],
        [0.74, 0.57, 0.77, 0.96],
        [0.66, 0.54, 0.71, 0.88],
        [0.51, 0.56, 0.63, 0.83],
        [0.47, 0.45, 0.47, 0.77],
        [0.42, 0.46, 0.39, 0.71],
        [0.47, _N, 0.46, 0.53],
        [_N, _N, _N, 0.43],
        [_N, _N, _N, 0.20],
        [_N, _N, _N, 0.42],
    ],
    "parsivel2": [  # classes 3-22
        [0.02, 0.04, 0.04, 0.05, 0.06, 0.07],
        [0.03, 0.05, 0.05, 0.07, 0.11, 0.16],
        [0.11, 0.16, 0.19, 0.22, 0.30, 0.36],
        [0.20, 0.26, 0.29, 0.36, 0.45, 0.54],
        [0.36, 0.47, 0.52, 0.53, 0.71, 0.78],
        [0.55, 0.55, 0.67, 0.67, 0.80, 0.86],
        [0.86, 0.85, 0.94, 0.89, 1.01, 1.03],
        [0.74, 0.84, 1.08, 0.90, 1.17, 1.03],
        [1.04, 1.13, 1.22, 1.12, 1.36, 1.12],
        [1.10, 1.20, 1.35, 1.19, 1.37, 1.10],
        [1.14, 0.97, 1.34, 1.17, 1.41, 1.04],
        [_N, _N, 1.25, 1.17, 1.22, 0.97],
        [_N, _N, 1.29, 1.17, 1.43, 1.06],
        [_N, _N, 1.43, _N, 1.37, 1.07],
        [_N, _N, 0.51, _N, 1.31, 1.02],
        [_N, _N, _N, _N, _N, 0.97],
        [_N, _N, _N, _N, _N, 0.73],
        [_N, _N, _N, _N, _N, 0.58],
        [_N, _N, _N, _N, _N, 0.45],
        [_N, _N, _N, _N, _N, 0.32],
    ],
}

_FITS = {
    "MM246": ((2, 4, 6), False),
    "MM346": ((3, 4, 6), False),
    "MM234": ((2, 3, 4), False),
    "TMM246": ((2, 4, 6), True),
    "TMM346": ((3, 4, 6), True),
    "TMM234": ((2, 3, 4), True),
}

_ND_ATTRS = {
    "long_name": "Number concentration of drops per unit diameter",
    "units": "m-3 mm-1",
}


def _use_compiled(engine):
    """Whether to run the compiled kernel for the requested ``engine``."""
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled disdrometer kernel is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


def _threads(n_threads):
    return int(n_threads or 0)


def _check_counts(ds, name="counts"):
    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"expected an xarray.Dataset, not {type(ds).__name__}")
    if name not in ds:
        raise ValueError(f"the dataset has no {name!r} variable")
    da = ds[name]
    if set(da.dims) != {"time", "velocity", "diameter"}:
        raise ValueError(
            f"{name!r} must be on ('time', 'velocity', 'diameter'), not {da.dims}"
        )
    return da.transpose("time", "velocity", "diameter")


def _edges(ds, dim):
    """Class centres, widths, lower and upper edges of a dimension."""
    c = np.asarray(ds[dim].values, float)
    width = "bin_width" if dim == "diameter" else f"{dim}_width"
    w = np.asarray(ds[width].values, float) if width in ds.coords else None
    if w is None:
        w = np.gradient(c) if c.size > 1 else np.ones(1)
    lo = ds[f"{dim}_lower"].values if f"{dim}_lower" in ds.coords else c - w / 2
    up = ds[f"{dim}_upper"].values if f"{dim}_upper" in ds.coords else c + w / 2
    return c, w, np.asarray(lo, float), np.asarray(up, float)


# --------------------------------------------------------------------------
# fall speed
# --------------------------------------------------------------------------


def terminal_fall_speed(diameter, air_density=None):
    """
    Terminal fall speed of raindrops.

    Parameters
    ----------
    diameter : xarray.DataArray or array-like
        Equal-volume drop diameter in mm.
    air_density : xarray.DataArray or float, optional
        Air density in kg m-3; the speed is scaled by
        :math:`(\\rho_0/\\rho)^{0.4}` with :math:`\\rho_0 = 1.204`
        kg m\\ :sup:`-3`. Default: :math:`\\rho_0` (sea level).

    Returns
    -------
    xarray.DataArray
        Fall speed :math:`9.65 - 10.3 e^{-0.6 D}` m s-1 (Atlas et al.
        1973; Bringi and Chandrasekar 2001, Eq. 7.65b), at least zero,
        broadcast over ``diameter`` and ``air_density``.

    Notes
    -----
    The law is negative below 0.109 mm (clipped to zero here, a radarx
    choice). Its range of validity in the original paper is not checked. The exponent 0.4 of the density correction is the
    one attributed to Foote and du Toit (1969) in Li and Srivastava (2001,
    text after their Eq. 4) and Kumjian and Ryzhkov (2010, Eq. 3); the
    equation and range of validity of the original paper are not checked. Beard (1976), used as the reference by Raupach
    and Berne (2015), is not implemented.

    References
    ----------
    Atlas, D., R. C. Srivastava, and R. S. Sekhon, 1973: Doppler radar
    characteristics of precipitation at vertical incidence. *Rev.
    Geophys.*, **11** (1), 1-35, https://doi.org/10.1029/RG011i001p00001

    Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler Weather
    Radar: Principles and Applications*. Cambridge University Press, 636 pp.,
    https://doi.org/10.1017/CBO9780511541094

    Li, X., and R. C. Srivastava, 2001: An analytical solution for raindrop
    evaporation and its application to radar rainfall measurements. *J.
    Appl. Meteor.*, **40** (9), 1607-1616,
    https://doi.org/10.1175/1520-0450(2001)040<1607:AASFRE>2.0.CO;2

    Kumjian, M. R., and A. V. Ryzhkov, 2010: The impact of evaporation on
    polarimetric characteristics of rain: Theoretical model and practical
    implications. *J. Appl. Meteor. Climatol.*, **49** (6), 1247-1267,
    https://doi.org/10.1175/2010JAMC2243.1

    Foote, G. B., and P. S. du Toit, 1969: Terminal velocity of raindrops
    aloft. *J. Appl. Meteor.*, **8** (2), 249-253,
    https://doi.org/10.1175/1520-0450(1969)008<0249:TVORA>2.0.CO;2

    Beard, K. V., 1976: Terminal velocity and shape of cloud and precipitation
    drops aloft. *J. Atmos. Sci.*, **33** (5), 851-864,
    https://doi.org/10.1175/1520-0469(1976)033<0851:TVASOC>2.0.CO;2

    Raupach, T. H., and A. Berne, 2015: Correction of raindrop size
    distributions measured by Parsivel disdrometers, using a two-dimensional
    video disdrometer as a reference. *Atmos. Meas. Tech.*, **8** (1),
    343-365, https://doi.org/10.5194/amt-8-343-2015
    """
    d = diameter if isinstance(diameter, xr.DataArray) else xr.DataArray(diameter)
    v = np.maximum(9.65 - 10.3 * np.exp(-0.6 * d), 0.0)
    if air_density is not None:
        v = v * (RHO0 / air_density) ** 0.4
    v = v.rename("terminal_fall_speed")
    v.attrs = {
        "long_name": "Terminal fall speed of raindrops",
        "units": "m s-1",
    }
    return v


def _air_density(ds):
    """Air density (time) from the station measurements, or None."""
    if not {"air_pressure", "air_temperature"} <= set(ds.data_vars):
        return None
    from ..io.sounding import air_density, saturation_vapor_pressure

    p = ds["air_pressure"] * 100.0
    t = ds["air_temperature"] + 273.15
    q = 0.0
    if "relative_humidity" in ds:
        e = ds["relative_humidity"].fillna(0.0) / 100.0 * saturation_vapor_pressure(t)
        q = 0.622 * e / (p - 0.378 * e)
    rho = air_density(p, t, q)
    return rho.where(np.isfinite(rho), RHO0).rename("air_density")


def _fall_speed(ds, density_correction):
    """
    Terminal fall speed on (time, diameter) with air density, else on
    (diameter).
    """
    rho = _air_density(ds) if density_correction else None
    vt = terminal_fall_speed(ds["diameter"], rho)
    return vt.transpose(*[d for d in ("time", "diameter") if d in vt.dims])


def _fall_speed_values(vt, n):
    """Fall speed as a (time, diameter) array."""
    v = np.asarray(vt.values, float)
    return np.ascontiguousarray(np.broadcast_to(v, (n, v.shape[-1])))


# --------------------------------------------------------------------------
# quality control
# --------------------------------------------------------------------------


def _wind_valid(ds, max_wind):
    valid = xr.ones_like(ds["time"], dtype=bool)
    if max_wind is None:
        return valid
    for name in ("wind_speed_max", "wind_speed"):
        if name in ds:
            return ~(ds[name] > float(max_wind))
    raise ValueError("max_wind needs 'wind_speed' or 'wind_speed_max'")


def _finish_qc(ds, keep, valid, vt, counts, attrs, extra=None):
    out = ds.copy()
    keep = keep.transpose(
        *[d for d in ("time", "velocity", "diameter") if d in keep.dims]
    )
    qc = (counts * keep).astype(float)
    out["counts_qc"] = qc if bool(valid.all()) else qc.where(valid)
    out["counts_qc"].attrs = {
        "long_name": "Number of particles per class after quality control",
        "units": "1",
    }
    out["qc_mask"] = keep
    out["qc_mask"].attrs = {"long_name": "Classes kept by the quality control"}
    out["valid"] = valid
    out["valid"].attrs = {"long_name": "Records kept by the quality control"}
    out["terminal_fall_speed"] = vt
    for k, v in (extra or {}).items():
        out[k] = v
    out.attrs["qc"] = attrs
    return out


def disdrometer_qc(
    ds,
    *,
    method="relative",
    tolerance=0.6,
    max_diameter=8.0,
    drop_unmeasured=True,
    max_wind=None,
    density_correction=True,
):
    """
    Remove particles that are not plausible raindrops.

    Parameters
    ----------
    ds : xarray.Dataset
        Disdrometer data with ``counts`` on ``(time, velocity, diameter)``,
        e.g. from :func:`radarx.io.read_parsivel`.
    method : {"relative", "raupach2015"}, optional
        ``"relative"`` (default) keeps classes whose velocity is within
        ``tolerance`` times the terminal fall speed; ``"raupach2015"``
        uses the filter of Raupach and Berne (2015, Eqs. 9-11). See
        :mod:`radarx.retrieve.disdrometer`.
    tolerance : float, optional
        Relative tolerance of the ``"relative"`` filter. Default 0.6.
    max_diameter : float or None, optional
        Remove classes larger than this (mm; class centre). Default 8
        (Friedrich et al. 2013); ``"raupach2015"`` always removes
        :math:`D > 7.5` mm.
    drop_unmeasured : bool, optional
        Remove the two smallest size classes (below 0.25 mm), which the
        Parsivel does not measure. Default True.
    max_wind : float, optional
        Discard records whose wind speed (``wind_speed_max``, else
        ``wind_speed``, m s-1) exceeds this. Default None (keep all).
        Friedrich et al. (2013) found the strong-wind artifact mostly above
        20 m s-1, sometimes from 10 m s-1.
    density_correction : bool, optional
        Scale fall speeds to the air density from ``air_pressure``,
        ``air_temperature`` and ``relative_humidity`` when present. Default
        True.

    Returns
    -------
    xarray.Dataset
        ``ds`` with ``counts_qc`` (counts of the kept classes, NaN in
        discarded records), ``qc_mask`` (kept classes), ``valid`` (kept
        records) and ``terminal_fall_speed`` (m s-1, on ``diameter``, and
        on ``time`` with the air density).

    References
    ----------
    Friedrich, K., S. Higgins, F. J. Masters, and C. R. Lopez, 2013:
    Articulating and stationary PARSIVEL disdrometer measurements in
    conditions with strong winds and heavy rainfall. *J. Atmos. Oceanic
    Technol.*, **30** (9), 2063-2080,
    https://doi.org/10.1175/JTECH-D-12-00254.1

    Raupach, T. H., and A. Berne, 2015: Correction of raindrop size
    distributions measured by Parsivel disdrometers, using a
    two-dimensional video disdrometer as a reference. *Atmos. Meas.
    Tech.*, **8** (1), 343-365, https://doi.org/10.5194/amt-8-343-2015

    The default ``tolerance`` of 0.6 and the ``drop_unmeasured`` threshold of
    0.25 mm are radarx choices, not from these papers; the fall speed is that
    of :func:`terminal_fall_speed`.
    """
    counts = _check_counts(ds)
    if method not in ("relative", "raupach2015"):
        raise ValueError(f"method must be 'relative' or 'raupach2015', not {method!r}")
    if not tolerance > 0:
        raise ValueError("tolerance must be positive")
    vt = _fall_speed(ds, density_correction)
    v = ds["velocity"]
    d = ds["diameter"]
    if method == "relative":
        keep = abs(v - vt) <= float(tolerance) * vt
    else:
        keep = (v <= vt + 4.0) & (v >= vt - 3.0) & (d <= 7.5)
    if max_diameter is not None:
        keep = keep & (d <= float(max_diameter))
    if drop_unmeasured:
        keep = keep & (d > 0.25)
    valid = _wind_valid(ds, max_wind)
    attrs = (
        f"method={method}, tolerance={tolerance}, max_diameter={max_diameter}, "
        f"drop_unmeasured={drop_unmeasured}, max_wind={max_wind}"
    )
    return _finish_qc(ds, keep, valid, vt, counts, attrs)


def _velocity_shift_numpy(counts, lo, up, vt, step):
    n, nv, nd = counts.shape
    nsub = np.maximum(1, np.rint((up - lo) / step).astype(int))
    centre = 0.5 * (lo + up)
    total = counts.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (counts * centre[None, :, None]).sum(1) / total
    shift = np.round((vt - mean) / step) * step  # (n, nd)
    ok = (total > 0) & np.isfinite(vt)
    out = np.where(ok[:, None, :], 0.0, counts)
    for v in range(nv):
        part = counts[:, v, :] / nsub[v]
        for s in range(nsub[v]):
            x = lo[v] + (s + 0.5) * step + shift
            inside = ok & (x >= lo[0]) & (x < up[-1]) & (counts[:, v, :] != 0)
            w = np.clip(np.searchsorted(lo, x, side="right") - 1, 0, nv - 1)
            inside &= x < up[w]
            t_idx, d_idx = np.nonzero(inside)
            np.add.at(out, (t_idx, w[t_idx, d_idx], d_idx), part[t_idx, d_idx])
    return out


def raupach_berne_correction(
    ds,
    *,
    instrument="parsivel2",
    counts="counts",
    max_wind=None,
    density_correction=True,
    engine="auto",
    n_threads=None,
):
    """
    Correct Parsivel spectra with the method of Raupach and Berne (2015).

    The velocities of each size class are shifted so that their mean
    matches the terminal fall speed, implausible particles are removed
    (their Eqs. 9-11), and per-class concentration correction factors are
    attached for :func:`number_concentration` (see
    :mod:`radarx.retrieve.disdrometer`).

    Parameters
    ----------
    ds : xarray.Dataset
        Disdrometer data with ``counts`` on ``(time, velocity, diameter)``
        and the instrument rain intensity ``rain_rate_instrument``
        (mm h-1), which selects the correction factors.
    instrument : {"parsivel2", "parsivel"}, optional
        Factors for the Parsivel2 (Table 10 of Raupach and Berne 2015,
        default) or the first-generation Parsivel (Table 3).
    counts : str, optional
        Counts to correct. Default ``"counts"`` (raw, as in Raupach and
        Berne 2015); ``"counts_qc"`` corrects the output of
        :func:`disdrometer_qc`, so that particles far from the fall speed
        (e.g. splashing drops) do not shift the mean velocity of their size
        class.
    max_wind, density_correction : optional
        See :func:`disdrometer_qc`.
    engine : {"auto", "compiled", "numpy"}, optional
        Compiled kernel or NumPy implementation of the velocity shift.
    n_threads : int, optional
        Threads of the compiled kernel; default all cores.

    Returns
    -------
    xarray.Dataset
        ``ds`` with ``counts_qc`` (shifted and filtered counts), ``qc_mask``,
        ``valid``, ``terminal_fall_speed`` and ``concentration_factor`` (on
        ``time`` and ``diameter``; NaN where the intensity is missing).

    References
    ----------
    Raupach, T. H., and A. Berne, 2015: Correction of raindrop size
    distributions measured by Parsivel disdrometers, using a
    two-dimensional video disdrometer as a reference. *Atmos. Meas.
    Tech.*, **8** (1), 343-365, https://doi.org/10.5194/amt-8-343-2015
    """
    name = counts
    counts = _check_counts(ds, name).fillna(0.0)
    if instrument not in _RB_FACTORS:
        raise ValueError(
            f"instrument must be one of {tuple(_RB_FACTORS)}, not {instrument!r}"
        )
    if "rain_rate_instrument" not in ds:
        raise ValueError("the correction needs 'rain_rate_instrument'")
    if ds.sizes["diameter"] != 32 or ds.sizes["velocity"] != 32:
        raise ValueError("the correction needs the 32 x 32 Parsivel classes")
    vt = _fall_speed(ds, density_correction)
    _, _, vlo, vup = _edges(ds, "velocity")
    c = np.asarray(counts.values, float)
    vtv = _fall_speed_values(vt, counts.shape[0])
    if _use_compiled(engine):
        shifted = _disdrometer.velocity_shift(
            c, vlo, vup, vtv, 0.1, _threads(n_threads)
        )
    else:
        shifted = _velocity_shift_numpy(c, vlo, vup, vtv, 0.1)
    shifted = counts.copy(data=shifted)
    v = ds["velocity"]
    d = ds["diameter"]
    keep = (v <= vt + 4.0) & (v >= vt - 3.0) & (d <= 7.5)
    valid = _wind_valid(ds, max_wind)
    # concentration factors by intensity class
    table = np.array(
        [[np.nan if x is None else x for x in row] for row in _RB_FACTORS[instrument]]
    )
    edges = np.array(_RB_INTENSITY[instrument])
    rate = ds["rain_rate_instrument"].values
    cls = np.searchsorted(edges, rate, side="right") - 1
    ok = np.isfinite(rate) & (cls >= 0) & (cls < len(edges) - 1)
    factor = np.ones((rate.size, 32))
    rows = table[:, np.clip(cls, 0, len(edges) - 2)].T  # (time, classes 3..)
    factor[:, 2 : 2 + table.shape[0]] = np.where(np.isfinite(rows), rows, 1.0)
    factor[~ok] = np.nan
    cf = xr.DataArray(
        factor,
        dims=("time", "diameter"),
        coords={"time": ds["time"], "diameter": ds["diameter"]},
        attrs={
            "long_name": "Drop concentration correction factor (Raupach and Berne 2015)",
            "units": "1",
        },
    )
    attrs = (
        f"method=raupach_berne_2015, instrument={instrument}, counts={name}, "
        f"max_wind={max_wind}"
    )
    if name == "counts_qc" and "qc" in ds.attrs:
        attrs = f"{ds.attrs['qc']}; {attrs}"
        valid = valid & ds["valid"] if "valid" in ds else valid
    return _finish_qc(
        ds, keep, valid, vt, shifted, attrs, extra={"concentration_factor": cf}
    )


# --------------------------------------------------------------------------
# N(D)
# --------------------------------------------------------------------------


def number_concentration(
    ds,
    *,
    counts=None,
    velocity="measured",
    sample_interval=None,
    engine="auto",
    n_threads=None,
):
    """
    Drop number concentration N(D) from particle counts.

    Parameters
    ----------
    ds : xarray.Dataset
        Disdrometer data with counts on ``(time, velocity, diameter)``.
    counts : str, optional
        Counts variable. Default ``"counts_qc"`` if present (output of
        :func:`disdrometer_qc` or :func:`raupach_berne_correction`), else
        ``"counts"``.
    velocity : {"measured", "terminal"}, optional
        Fall speed of each particle: the centre of its velocity class
        (default) or the terminal fall speed of its size class
        (``terminal_fall_speed`` of the dataset, or sea-level speeds).
    sample_interval : float, optional
        Integration time in s. Default: the ``sample_interval`` variable,
        else the spacing of ``time``.
    engine : {"auto", "compiled", "numpy"}, optional
        Compiled kernel or NumPy implementation.
    n_threads : int, optional
        Threads of the compiled kernel; default all cores.

    Returns
    -------
    xarray.DataArray
        ``ND`` (m-3 mm-1) on ``(time, diameter)`` with the ``bin_width``
        coordinate, multiplied by ``concentration_factor`` when present.
        NaN for records discarded by the quality control.

    Notes
    -----
    :math:`N(D_i)` is Eq. 6 of Raupach and Berne (2015) with the sampling
    area :math:`S_i = 10^{-6} L (B - D_i/2)` of their Eq. 5 (:math:`L` =
    180 mm, :math:`B` = 30 mm, Sect. 4; attributed there to Löffler-Mang and
    Joss 2000 and Battaglia et al. 2010). Raw counts are multiplied by
    ``concentration_factor`` only when it is present (their Sect. 5.2).

    References
    ----------
    Raupach, T. H., and A. Berne, 2015: Correction of raindrop size
    distributions measured by Parsivel disdrometers, using a
    two-dimensional video disdrometer as a reference. *Atmos. Meas.
    Tech.*, **8** (1), 343-365, https://doi.org/10.5194/amt-8-343-2015

    Löffler-Mang, M., and J. Joss, 2000: An optical disdrometer for
    measuring size and velocity of hydrometeors. *J. Atmos. Oceanic
    Technol.*, **17** (2), 130-139,
    https://doi.org/10.1175/1520-0426(2000)017<0130:AODFMS>2.0.CO;2

    Battaglia, A., E. Rustemeier, A. Tokay, U. Blahak, and C. Simmer, 2010:
    PARSIVEL snow observations: A critical assessment. *J. Atmos. Oceanic
    Technol.*, **27** (2), 333-344, https://doi.org/10.1175/2009JTECHA1332.1
    """
    if counts is None:
        counts = "counts_qc" if "counts_qc" in ds else "counts"
    c = _check_counts(ds, counts)
    if velocity not in ("measured", "terminal"):
        raise ValueError(f"velocity must be 'measured' or 'terminal', not {velocity!r}")
    d, dd, _, _ = _edges(ds, "diameter")
    n, nv, nd = c.shape
    if sample_interval is not None:
        dt = np.full(n, float(sample_interval))
    elif "sample_interval" in ds:
        dt = ds["sample_interval"].values.astype(float)
    else:
        t = ds["time"].values.astype("datetime64[ns]").astype(np.int64) / 1e9
        dt = np.full(n, float(np.median(np.diff(t))) if n > 1 else np.nan)
    area = 1e-6 * BEAM_LENGTH * (BEAM_WIDTH - d / 2.0)  # m2
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = 1.0 / (area[None, :] * dd[None, :] * dt[:, None])
        if "concentration_factor" in ds:
            scale = (
                scale * ds["concentration_factor"].transpose("time", "diameter").values
            )
        if velocity == "measured":
            v = np.asarray(ds["velocity"].values, float)
            weight = np.broadcast_to((1.0 / v)[None, :, None], (1, nv, nd))
        else:
            vt = _fall_speed_values(
                (
                    ds["terminal_fall_speed"]
                    if "terminal_fall_speed" in ds
                    else _fall_speed(ds, False)
                ),
                n,
            )
            weight = np.where(vt > 0, 1.0 / vt, 0.0)[:, None, :]
            weight = np.broadcast_to(weight, (n, nv, nd))
    weight = np.ascontiguousarray(weight, float)
    scale = np.ascontiguousarray(scale, float)
    values = np.asarray(c.values, float)
    if _use_compiled(engine):
        out = _disdrometer.number_concentration(
            values, weight, scale, _threads(n_threads)
        )
    else:
        out = (
            np.einsum("tvd,wvd->td", values, weight)
            if weight.shape[0] == 1
            else (np.einsum("tvd,tvd->td", values, weight))
        )
        out = out * scale
    nd_da = xr.DataArray(
        out,
        dims=("time", "diameter"),
        coords={
            "time": ds["time"],
            "diameter": ds["diameter"],
            "bin_width": ("diameter", dd, {"units": "mm"}),
        },
        name="ND",
        attrs=dict(_ND_ATTRS),
    )
    for k, val in ds.coords.items():
        if val.ndim == 0:
            nd_da = nd_da.assign_coords({k: val})
    return nd_da


# --------------------------------------------------------------------------
# moments
# --------------------------------------------------------------------------


def _diameter_dim(nd, dim):
    if dim not in nd.dims:
        raise ValueError(f"nd has no {dim!r} dimension")
    d = np.asarray(nd[dim].values, float)
    w = _dsdmod._bin_widths(nd, dim).values
    return d, w


def dsd_moments(nd, dim="diameter"):
    """
    Integral quantities of measured drop size distributions.

    Parameters
    ----------
    nd : xarray.DataArray
        N(D) (m-3 mm-1) on a diameter dimension (mm) with a ``bin_width``
        coordinate (as from :func:`number_concentration`).
    dim : str, optional
        The diameter dimension. Default ``"diameter"``.

    Returns
    -------
    xarray.Dataset
        On the other dimensions: total concentration ``NT`` (m-3), liquid
        water content ``LWC`` (g m-3), rain rate ``RAIN_RATE`` (mm h-1, with
        the sea-level fall speed of :func:`terminal_fall_speed`), Rayleigh
        reflectivity ``DBZ_RAYLEIGH`` (dBZ, :math:`10 \\log_{10} M_6`),
        mass-weighted mean diameter ``DM`` (:math:`M_4/M_3`, mm), its
        standard deviation ``SIGMA_M`` (mm), median volume diameter ``D0``
        (mm, drops spread uniformly within each class) and normalized
        intercept ``NW`` (:math:`4^4 M_3 / (6 D_m^4)`, m-3 mm-1, Testud et
        al. 2001; see :mod:`radarx.retrieve.dsd`). NaN where N(D) is NaN.

    Notes
    -----
    The moments are :math:`M_n = \\sum N(D_i) D_i^n \\Delta D_i`. The rain rate
    is :math:`0.6\\pi\\times10^{-3} \\sum v D^3 N \\Delta D` (Bringi and
    Chandrasekar 2001, Eq. 7.66a) with the Atlas et al. (1973) fall speed,
    :math:`D_m = M_4/M_3` (their Eq. 7.13), :math:`N_w` the :math:`D_m` form of
    Testud et al. (2001). The standard deviation of the mass distribution and
    the within-class interpolation of :math:`D_0` are the usual definitions,
    implemented here without a specific paper's equation. The moment
    definitions follow Ulbrich and Atlas (1998).

    References
    ----------
    Ulbrich, C. W., and D. Atlas, 1998: Rainfall microphysics and radar
    properties: Analysis methods for drop size spectra. *J. Appl. Meteor.*,
    **37** (9), 912-923,
    https://doi.org/10.1175/1520-0450(1998)037<0912:RMARPA>2.0.CO;2

    Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler Weather
    Radar: Principles and Applications*. Cambridge University Press, 636 pp.,
    https://doi.org/10.1017/CBO9780511541094

    Testud, J., S. Oury, R. A. Black, P. Amayenc, and X. Dou, 2001: The
    concept of "normalized" distribution to describe raindrop spectra: A tool
    for cloud physics and cloud remote sensing. *J. Appl. Meteor.*, **40**
    (6), 1118-1140,
    https://doi.org/10.1175/1520-0450(2001)040<1118:TCONDT>2.0.CO;2

    Atlas, D., R. C. Srivastava, and R. S. Sekhon, 1973: Doppler radar
    characteristics of precipitation at vertical incidence. *Rev. Geophys.*,
    **11** (1), 1-35, https://doi.org/10.1029/RG011i001p00001
    """
    d, w = _diameter_dim(nd, dim)
    nd = nd.transpose(..., dim)
    x = np.asarray(nd.values, float)
    nw_ = x * w

    def mom(k):
        return (nw_ * d**k).sum(-1)

    m0, m3, m4, m6 = mom(0), mom(3), mom(4), mom(6)
    vt = np.maximum(9.65 - 10.3 * np.exp(-0.6 * d), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dm = m4 / m3
        sig = np.sqrt((nw_ * d**3 * (d - dm[..., None]) ** 2).sum(-1) / m3)
        # median volume diameter: cumulative M3 at the class edges
        lo = d - w / 2
        mass = nw_ * d**3
        cum = np.cumsum(mass, -1)
        half = 0.5 * m3
        k = np.argmax(cum >= half[..., None], -1)
        before = (
            np.take_along_axis(cum, k[..., None], -1)[..., 0]
            - np.take_along_axis(mass, k[..., None], -1)[..., 0]
        )
        frac = (half - before) / np.take_along_axis(mass, k[..., None], -1)[..., 0]
        d0 = lo[k] + frac * w[k]
        d0 = np.where(m3 > 0, d0, np.nan)
        values = {
            "NT": m0,
            "LWC": np.pi / 6.0 * 1.0e-3 * m3,
            "RAIN_RATE": 6.0e-4 * np.pi * (nw_ * vt * d**3).sum(-1),
            "DBZ_RAYLEIGH": 10.0 * np.log10(m6),
            "DM": np.where(m3 > 0, dm, np.nan),
            "SIGMA_M": np.where(m3 > 0, sig, np.nan),
            "D0": d0,
            "NW": np.where(m3 > 0, 256.0 / 6.0 * m3 / dm**4, np.nan),
        }
    attrs = {
        "NT": {"long_name": "Total drop number concentration", "units": "m-3"},
        "LWC": dict(_dsdmod._OUT_ATTRS["LWC"]),
        "RAIN_RATE": dict(_dsdmod._OUT_ATTRS["RAIN_RATE"]),
        "DBZ_RAYLEIGH": {
            "long_name": "Rayleigh reflectivity factor of the drop size distribution",
            "units": "dBZ",
        },
        "DM": dict(_dsdmod._OUT_ATTRS["DM"]),
        "SIGMA_M": {
            "long_name": "Standard deviation of the mass spectrum about Dm",
            "units": "mm",
        },
        "D0": dict(_dsdmod._OUT_ATTRS["D0"]),
        "NW": dict(_dsdmod._OUT_ATTRS["NW"]),
    }
    dims = nd.dims[:-1]
    coords = {k: v for k, v in nd.coords.items() if dim not in v.dims}
    return xr.Dataset(
        {k: (dims, v, attrs[k]) for k, v in values.items()}, coords=coords
    )


# --------------------------------------------------------------------------
# gamma fits
# --------------------------------------------------------------------------


def _ratio_numpy(ijk, mu):
    i, j, k = ijk
    return (
        (k - i) * gammaln(mu + j + 1)
        - (k - j) * gammaln(mu + i + 1)
        - (j - i) * gammaln(mu + k + 1)
    )


def _solve_numpy(ijk, mi, mj, mk, mu_lo, mu_hi):
    i, j, k = ijk
    with np.errstate(divide="ignore", invalid="ignore"):
        target = (k - i) * np.log(mj) - (k - j) * np.log(mi) - (j - i) * np.log(mk)
        ok = (mi > 0) & (mj > 0) & (mk > 0)
        ok &= (target > _ratio_numpy(ijk, mu_lo)) & (target < _ratio_numpy(ijk, mu_hi))
        lo = np.full(target.shape, float(mu_lo))
        hi = np.full(target.shape, float(mu_hi))
        for _ in range(_BISECT):
            mid = 0.5 * (lo + hi)
            below = _ratio_numpy(ijk, mid) < target
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        mu = 0.5 * (lo + hi)
        ll = (np.log(mi) + gammaln(mu + j + 1) - np.log(mj) - gammaln(mu + i + 1)) / (
            j - i
        )
        ln0 = np.log(mi) + (mu + i + 1) * ll - gammaln(mu + i + 1)
        lam = np.exp(ll)
    nan = np.nan
    return (
        np.where(ok, mu, nan),
        np.where(ok, lam, nan),
        np.where(ok, ln0, nan),
        ok,
    )


def _log_window(a, x0, x1):
    """log of P(a, x1) - P(a, x0) (regularized incomplete gamma)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        upper = x0 >= a + 1.0
        diff = np.where(
            upper,
            gammaincc(a, x0) - gammaincc(a, x1),
            gammainc(a, x1) - gammainc(a, x0),
        )
        return np.where(x0 > 0, np.log(diff), np.log(gammainc(a, x1)))


def _log_moment(n, mu, ll, d0, d1):
    a = mu + n + 1.0
    lam = np.exp(ll)
    return gammaln(a) - a * ll + _log_window(a, lam * d0, lam * d1)


def _residual(ijk, mu, ll, d0, d1, r1, r2):
    i, j, k = ijk
    li, lj, lk = (_log_moment(n, mu, ll, d0, d1) for n in (i, j, k))
    return lj - li - r1, lk - lj - r2


def _truncated_numpy(ijk, mi, mj, mk, d0, d1, mu, lam, mu_lo, mu_hi, max_iter, tol):
    """Damped Newton iteration of the truncated-moment fit (all spectra)."""
    n = mu.size
    with np.errstate(divide="ignore", invalid="ignore"):
        r1, r2 = np.log(mj / mi), np.log(mk / mj)
        m, ll = mu.copy(), np.log(lam)
        f0, f1 = _residual(ijk, m, ll, d0, d1, r1, r2)
    iters = np.full(n, -1.0)
    active = np.isfinite(m) & np.isfinite(ll)
    for it in range(max_iter + 1):
        fin = np.isfinite(f0) & np.isfinite(f1)
        active &= fin
        conv = active & (np.maximum(np.abs(f0), np.abs(f1)) < tol)
        iters[conv] = it
        active &= ~conv
        if it == max_iter or not active.any():
            break
        a_ = np.flatnonzero(active)
        h = 1e-6
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            g0, g1 = _residual(ijk, m[a_] + h, ll[a_], d0[a_], d1[a_], r1[a_], r2[a_])
            h0, h1 = _residual(ijk, m[a_], ll[a_] + h, d0[a_], d1[a_], r1[a_], r2[a_])
            ja, jb = (g0 - f0[a_]) / h, (h0 - f0[a_]) / h
            jc, jd = (g1 - f1[a_]) / h, (h1 - f1[a_]) / h
            det = ja * jd - jb * jc
            dm = -(jd * f0[a_] - jb * f1[a_]) / det
            dl = -(-jc * f0[a_] + ja * f1[a_]) / det
        okdet = (np.abs(det) > 0) & np.isfinite(det)
        norm = np.maximum(np.abs(f0[a_]), np.abs(f1[a_]))
        moved = np.zeros(a_.size, bool)
        new_m, new_l = m[a_].copy(), ll[a_].copy()
        new_f0, new_f1 = f0[a_].copy(), f1[a_].copy()
        t = 1.0
        for _ in range(30):
            todo = okdet & ~moved
            if not todo.any():
                break
            cm = m[a_] + t * dm
            cl = ll[a_] + t * dl
            inside = todo & (cm > mu_lo) & (cm < mu_hi) & (np.abs(cl) < _LOG_LAM_MAX)
            q = np.flatnonzero(inside)
            if q.size:
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    c0, c1 = _residual(
                        ijk, cm[q], cl[q], d0[a_][q], d1[a_][q], r1[a_][q], r2[a_][q]
                    )
                better = (
                    np.isfinite(c0)
                    & np.isfinite(c1)
                    & (np.maximum(np.abs(c0), np.abs(c1)) < norm[q])
                )
                b = q[better]
                new_m[b], new_l[b] = cm[b], cl[b]
                new_f0[b], new_f1[b] = c0[better], c1[better]
                moved[b] = True
            t *= 0.5
        stuck = ~moved
        # the line search stalls at the precision of the moment ratios:
        # accept a residual below _ACCEPT
        iters[a_[stuck & (norm < _ACCEPT)]] = it + 1
        m[a_], ll[a_], f0[a_], f1[a_] = new_m, new_l, new_f0, new_f1
        active[a_[stuck]] = False
    ok = iters >= 0
    i = ijk[0]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ln0 = np.log(mi) - _log_moment(i, m, ll, d0, d1)
        lam_out = np.exp(ll)
    nan = np.nan
    return (
        np.where(ok, m, nan),
        np.where(ok, lam_out, nan),
        np.where(ok, ln0, nan),
        np.where(ok, iters, nan),
    )


def _fit_numpy(
    x, d, dd, lo, up, ijk, truncated, lower_cut, mu_lo, mu_hi, max_iter, tol
):
    i, j, k = ijk
    w = x * dd
    mi, mj, mk = ((w * d**p).sum(-1) for p in (i, j, k))
    mu, lam, ln0, ok = _solve_numpy(ijk, mi, mj, mk, mu_lo, mu_hi)
    iters = np.where(ok, 0.0, np.nan)
    if truncated:
        pos = x > 0
        first = np.argmax(pos, -1)
        last = x.shape[-1] - 1 - np.argmax(pos[:, ::-1], -1)
        d0 = lo[first] if lower_cut else np.zeros(x.shape[0])
        d1 = up[last]
        with np.errstate(divide="ignore", invalid="ignore"):
            lam0 = np.exp(
                (gammaln(j + 1.0) - gammaln(i + 1.0) - np.log(mj / mi)) / (j - i)
            )
        start_mu = np.where(ok, mu, 0.0)
        start_lam = np.where(ok, lam, lam0)
        has = (mi > 0) & (mj > 0) & (mk > 0)
        start_mu = np.where(has, start_mu, np.nan)
        mu, lam, ln0, iters = _truncated_numpy(
            ijk, mi, mj, mk, d0, d1, start_mu, start_lam, mu_lo, mu_hi, max_iter, tol
        )
    return np.stack([ln0 / np.log(10.0), mu, lam, iters])


def _truncated_quantities(logn0, mu, lam, dmin, dmax):
    """
    N0, NW, D0, DM, MU, LAMBDA, RAIN_RATE and LWC of gamma DSDs truncated to
    [dmin, dmax] (closed form with incomplete gamma functions).
    """
    with np.errstate(all="ignore"):
        n0 = 10.0**logn0

        def window(a, scale):
            return gammainc(a, scale * dmax) - gammainc(a, scale * dmin)

        def moment(n):
            a = mu + n + 1.0
            return n0 * np.exp(gammaln(a) - a * np.log(lam)) * window(a, lam)

        m3, m4 = moment(3), moment(4)
        dm = m4 / m3
        a4 = mu + 4.0
        # fall speed 9.65 - 10.3 exp(-0.6 D) (Atlas et al. 1973; Bringi and
        # Chandrasekar 2001, Eq. 7.65b), integrated over the truncation window
        m3v = n0 * np.exp(gammaln(a4) - a4 * np.log(lam + 0.6)) * window(a4, lam + 0.6)
        rain = 6.0e-4 * np.pi * (9.65 * m3 - 10.3 * m3v)
        lo = gammainc(a4, lam * dmin)
        d0 = gammaincinv(a4, lo + 0.5 * (gammainc(a4, lam * dmax) - lo)) / lam
        nw = 256.0 / 6.0 * m3 / dm**4
    return np.stack([n0, nw, d0, dm, mu, lam, rain, np.pi / 6.0 * 1.0e-3 * m3])


def fit_gamma(
    nd,
    *,
    moments=(2, 4, 6),
    truncated=False,
    lower_truncation=False,
    dim="diameter",
    mu_range=None,
    max_iter=50,
    tol=1e-10,
    engine="auto",
    n_threads=None,
):
    """
    Fit gamma drop size distributions to measured spectra.

    Parameters
    ----------
    nd : xarray.DataArray
        N(D) (m-3 mm-1) on a diameter dimension (mm) with a ``bin_width``
        coordinate (as from :func:`number_concentration`); class edges come
        from ``diameter_lower``/``diameter_upper`` coordinates if present,
        else centre -/+ half width.
    moments : tuple of int, optional
        The three moment orders :math:`i < j < k`. Default ``(2, 4, 6)``;
        ``(3, 4, 6)`` and ``(2, 3, 4)`` are common alternatives.
    truncated : bool, optional
        Fit the gamma DSD truncated at the largest observed diameter
        (truncated moments, Ulbrich and Atlas 1998) instead of the
        untruncated one (method of moments). Default False.
    lower_truncation : bool, optional
        Also truncate at the smallest observed diameter. Default False.
    dim : str, optional
        The diameter dimension. Default ``"diameter"``.
    mu_range : (float, float), optional
        Admissible shape parameters; spectra whose fit falls outside are
        NaN. Default -1 to 50, and :math:`-(i + 1) + 0.05` to 50 for
        truncated fits (whose moments stay finite for :math:`\\mu < -1`).
    max_iter, tol : optional
        Newton iterations of the truncated fit (default 50) and tolerance
        on the log moment ratios (default 1e-10).
    engine : {"auto", "compiled", "numpy"}, optional
        Compiled kernel or NumPy implementation.
    n_threads : int, optional
        Threads of the compiled kernel; default all cores.

    Returns
    -------
    xarray.Dataset
        ``N0`` (m-3 mm-(1+mu)), ``MU``, ``LAMBDA`` (mm-1) and the
        quantities ``NW``, ``D0``, ``DM``, ``RAIN_RATE`` and ``LWC`` (as
        from :func:`radarx.retrieve.dsd`) of the fitted gamma DSD,
        truncated like the fit, with ``FIT_ITERATIONS`` (0 for the
        untruncated fit); NaN where the spectrum has no fit.

    Notes
    -----
    The method of moments (untruncated) follows Ulbrich and Atlas (1998) and
    the 2-4-6 estimator evaluated by Cao and Zhang (2009); the truncated fit
    follows the idea of the truncated moments of Ulbrich and Atlas (1998) and
    the truncated moment fit of Cao et al. (2008) and Vivekanandan et al.
    (2004); the equations are not compared with those papers. The closed forms are those of the gamma moments and incomplete gamma
    functions, solved by radarx's own damped Newton iteration. The ``mu_range``,
    ``max_iter`` and ``tol`` defaults are radarx choices. The rain rate of the
    fitted DSD uses the Atlas et al. (1973) fall speed (Bringi and
    Chandrasekar 2001, Eq. 7.65b).

    References
    ----------
    Ulbrich, C. W., and D. Atlas, 1998: Rainfall microphysics and radar
    properties: Analysis methods for drop size spectra. *J. Appl.
    Meteor.*, **37** (9), 912-923,
    https://doi.org/10.1175/1520-0450(1998)037<0912:RMARPA>2.0.CO;2

    Cao, Q., and G. Zhang, 2009: Errors in estimating raindrop size
    distribution parameters employing disdrometer and simulated raindrop
    spectra. *J. Appl. Meteor. Climatol.*, **48** (2), 406-425,
    https://doi.org/10.1175/2008JAMC2026.1

    Cao, Q., G. Zhang, E. Brandes, T. Schuur, A. Ryzhkov, and K. Ikeda,
    2008: Analysis of video disdrometer and polarimetric radar data to
    characterize rain microphysics in Oklahoma. *J. Appl. Meteor.
    Climatol.*, **47** (8), 2238-2255,
    https://doi.org/10.1175/2008JAMC1732.1

    Vivekanandan, J., G. Zhang, and E. Brandes, 2004: Polarimetric radar
    estimators based on a constrained gamma drop size distribution model.
    *J. Appl. Meteor.*, **43** (2), 217-230,
    https://doi.org/10.1175/1520-0450(2004)043<0217:PREBOA>2.0.CO;2

    Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler Weather
    Radar: Principles and Applications*. Cambridge University Press, 636 pp.,
    https://doi.org/10.1017/CBO9780511541094

    Atlas, D., R. C. Srivastava, and R. S. Sekhon, 1973: Doppler radar
    characteristics of precipitation at vertical incidence. *Rev. Geophys.*,
    **11** (1), 1-35, https://doi.org/10.1029/RG011i001p00001

    Examples
    --------
    >>> fit = fit_gamma(nd, moments=(2, 4, 6), truncated=True)  # doctest: +SKIP
    """
    ijk = tuple(int(m) for m in moments)
    if len(ijk) != 3 or not (0 <= ijk[0] < ijk[1] < ijk[2]):
        raise ValueError(
            f"moments must be three orders 0 <= i < j < k, not {moments!r}"
        )
    if mu_range is None:
        mu_range = (-(ijk[0] + 1) + 0.05 if truncated else -1.0, 50.0)
    mu_lo, mu_hi = (float(m) for m in mu_range)
    if not (mu_lo > -(ijk[0] + 1) and mu_hi > mu_lo):
        raise ValueError(f"mu_range must lie above {-(ijk[0] + 1)}, not {mu_range!r}")
    d, dd = _diameter_dim(nd, dim)
    lo = nd[f"{dim}_lower"].values if f"{dim}_lower" in nd.coords else d - dd / 2
    up = nd[f"{dim}_upper"].values if f"{dim}_upper" in nd.coords else d + dd / 2
    nd = nd.transpose(..., dim)
    other = nd.dims[:-1]
    x = np.ascontiguousarray(np.asarray(nd.values, float).reshape(-1, d.size))
    if _use_compiled(engine):
        res = _disdrometer.fit_gamma(
            x,
            d,
            dd,
            np.asarray(lo, float),
            np.asarray(up, float),
            *ijk,
            bool(truncated),
            bool(lower_truncation),
            mu_lo,
            mu_hi,
            int(max_iter),
            float(tol),
            _threads(n_threads),
        )
    else:
        res = _fit_numpy(
            x,
            d,
            dd,
            np.asarray(lo, float),
            np.asarray(up, float),
            ijk,
            bool(truncated),
            bool(lower_truncation),
            mu_lo,
            mu_hi,
            int(max_iter),
            float(tol),
        )
    shape = nd.shape[:-1]
    logn0, mu, lam, iters = (r.reshape(shape) for r in res)
    if truncated:
        pos = x > 0
        first = np.argmax(pos, -1)
        last = x.shape[-1] - 1 - np.argmax(pos[:, ::-1], -1)
        dmin = (
            np.asarray(lo, float)[first] if lower_truncation else np.zeros(x.shape[0])
        )
        dmax = np.asarray(up, float)[last]
        values = _truncated_quantities(
            logn0.ravel(), mu.ravel(), lam.ravel(), dmin, dmax
        )
    else:
        values = _dsdmod._moments_numpy(logn0.ravel(), mu.ravel(), lam.ravel())
    data = {
        name: (other, values[q].reshape(shape), dict(_dsdmod._OUT_ATTRS[name]))
        for q, name in enumerate(_dsdmod._OUT_NAMES)
    }
    data["FIT_ITERATIONS"] = (
        other,
        iters,
        {"long_name": "Newton iterations of the truncated moment fit", "units": "1"},
    )
    name = ("TMM" if truncated else "MM") + "".join(str(m) for m in ijk)
    coords = {k: v for k, v in nd.coords.items() if dim not in v.dims}
    return xr.Dataset(
        data,
        coords=coords,
        attrs={
            "method": ("truncated moment fit" if truncated else "method of moments")
            + f" (M{ijk[0]}, M{ijk[1]}, M{ijk[2]})",
            "fit": name,
        },
    )


# --------------------------------------------------------------------------
# all products
# --------------------------------------------------------------------------

_RADAR_VARS = ("DBZH", "ZDR", "KDP", "RHOHV", "AH", "ADP")


def process_disdrometer(
    ds,
    *,
    qc="relative",
    fits=("MM246", "TMM246"),
    band="S",
    temperature=20.0,
    velocity="measured",
    engine="auto",
    n_threads=None,
    **qc_kwargs,
):
    """
    Quality control, N(D), moments, gamma fits and radar variables.

    Parameters
    ----------
    ds : xarray.Dataset
        Disdrometer data with ``counts`` (e.g. from
        :func:`radarx.io.read_parsivel`), or the output of
        :func:`disdrometer_qc` / :func:`raupach_berne_correction` (then
        ``qc=None``).
    qc : {"relative", "raupach2015", "raupach_berne", None}, optional
        :func:`disdrometer_qc` with that method (default ``"relative"``),
        :func:`raupach_berne_correction` (``"raupach_berne"``), or none.
    fits : sequence of str, optional
        Gamma fits among ``MM246``, ``MM346``, ``MM234`` (method of
        moments) and ``TMM246``, ``TMM346``, ``TMM234`` (truncated moments).
        Default ``("MM246", "TMM246")``.
    band : {"S", "C", "X"}, optional
        Radar band of the radar variables. Default ``"S"``.
    temperature : float, optional
        Water temperature (degrees Celsius) of the scattering tables.
        Default 20.
    velocity : {"measured", "terminal"}, optional
        See :func:`number_concentration`.
    engine, n_threads : optional
        Compiled kernel or NumPy implementation and threads.
    **qc_kwargs
        Options of the quality control (e.g. ``tolerance``, ``max_wind``).

    Returns
    -------
    xarray.Dataset
        On ``time``: ``ND`` (on ``time`` and ``diameter``), the quantities
        of :func:`dsd_moments`, the radar variables ``DBZH``, ``ZDR``,
        ``KDP``, ``RHOHV``, ``AH`` and ``ADP`` of
        :func:`radarx.retrieve.radar_from_dsd`, and per fit ``N0_<fit>``,
        ``MU_<fit>``, ``LAMBDA_<fit>``, ``DM_<fit>``, ``D0_<fit>`` and
        ``NW_<fit>``; with the station coordinates of ``ds``.

    Notes
    -----
    Chains :func:`disdrometer_qc` or :func:`raupach_berne_correction`
    (Raupach and Berne 2015), :func:`number_concentration`,
    :func:`dsd_moments` and :func:`fit_gamma` (Ulbrich and Atlas 1998; Cao and
    Zhang 2009) and :func:`radarx.retrieve.radar_from_dsd`; the equations,
    tables and radarx choices are described in the documentation of each.

    References
    ----------
    Raupach, T. H., and A. Berne, 2015: Correction of raindrop size
    distributions measured by Parsivel disdrometers, using a two-dimensional
    video disdrometer as a reference. *Atmos. Meas. Tech.*, **8** (1),
    343-365, https://doi.org/10.5194/amt-8-343-2015

    Ulbrich, C. W., and D. Atlas, 1998: Rainfall microphysics and radar
    properties: Analysis methods for drop size spectra. *J. Appl. Meteor.*,
    **37** (9), 912-923,
    https://doi.org/10.1175/1520-0450(1998)037<0912:RMARPA>2.0.CO;2

    Cao, Q., and G. Zhang, 2009: Errors in estimating raindrop size
    distribution parameters employing disdrometer and simulated raindrop
    spectra. *J. Appl. Meteor. Climatol.*, **48** (2), 406-425,
    https://doi.org/10.1175/2008JAMC2026.1

    Examples
    --------
    >>> from radarx.io import read_parsivel
    >>> out = read_parsivel("PIPS1A_merged.txt").radarx.disdrometer()
    ... # doctest: +SKIP
    """
    for f in fits:
        if f not in _FITS:
            raise ValueError(f"unknown fit {f!r}; choose from {tuple(_FITS)}")
    if qc in ("relative", "raupach2015"):
        ds = disdrometer_qc(ds, method=qc, **qc_kwargs)
    elif qc == "raupach_berne":
        ds = raupach_berne_correction(
            ds, engine=engine, n_threads=n_threads, **qc_kwargs
        )
    elif qc is not None:
        raise ValueError(
            "qc must be 'relative', 'raupach2015', 'raupach_berne' or None, "
            f"not {qc!r}"
        )
    nd = number_concentration(ds, velocity=velocity, engine=engine, n_threads=n_threads)
    out = dsd_moments(nd)
    valid = np.isfinite(out["NT"])
    radar = _dsdmod.radar_from_dsd(nd.fillna(0.0), band=band, temperature=temperature)
    for name in _RADAR_VARS:
        out[name] = radar[name].where(valid & (out["NT"] > 0))
        out[name].attrs = radar[name].attrs
    for f in fits:
        ijk, trunc = _FITS[f]
        fit = fit_gamma(
            nd, moments=ijk, truncated=trunc, engine=engine, n_threads=n_threads
        )
        for name in ("N0", "MU", "LAMBDA", "DM", "D0", "NW"):
            out[f"{name}_{f}"] = fit[name]
            out[f"{name}_{f}"].attrs = dict(fit[name].attrs, method=fit.attrs["method"])
    out["ND"] = nd
    out.attrs = {
        "band": radar.attrs["band"],
        "temperature": float(temperature),
        "qc": ds.attrs.get("qc", "none"),
        "velocity": velocity,
    }
    return out


@accessor_method("dataset", name="disdrometer")
def _disdrometer_accessor(self, **kwargs):
    """
    Disdrometer products: quality control, N(D), moments, fits, radar variables.

    Parameters
    ----------
    **kwargs
        Options of :func:`radarx.retrieve.process_disdrometer`, e.g. ``qc``,
        ``fits``, ``band``.

    Returns
    -------
    xarray.Dataset
        See :func:`radarx.retrieve.process_disdrometer`.
    """
    return process_disdrometer(self.xarray_obj, **kwargs)


# --------------------------------------------------------------------------
# radar matching
# --------------------------------------------------------------------------


def _ground_position(lat0, lon0, lat, lon):
    """Great-circle distance (m) and bearing (degrees) from (lat0, lon0)."""
    p0, p1 = np.deg2rad(lat0), np.deg2rad(lat)
    dl = np.deg2rad(lon - lon0)
    a = np.sin((p1 - p0) / 2) ** 2 + np.cos(p0) * np.cos(p1) * np.sin(dl / 2) ** 2
    s = 2 * EARTH_RADIUS * np.arcsin(np.sqrt(a))
    az = np.rad2deg(
        np.arctan2(
            np.sin(dl) * np.cos(p1),
            np.cos(p0) * np.sin(p1) - np.sin(p0) * np.cos(p1) * np.cos(dl),
        )
    )
    return s, az % 360.0


def _sweeps(radar, sweep):
    """Sweep datasets of a radar volume, list of volumes, or sweeps."""
    if isinstance(radar, xr.Dataset):
        return [radar]
    if isinstance(radar, xr.DataTree):
        return [_pick_sweep(radar, sweep)]
    out = []
    for r in radar:
        out.extend(_sweeps(r, sweep))
    return out


def _pick_sweep(tree, sweep):
    names = sorted(
        (n for n in tree.children if n.startswith("sweep_")),
        key=lambda n: int(n.split("_")[1]),
    )
    if not names:
        raise ValueError("the radar DataTree has no sweep_* nodes")
    if isinstance(sweep, (int, np.integer)):
        name = f"sweep_{int(sweep)}"
        if name not in tree.children:
            raise ValueError(f"the volume has no {name}")
    else:
        angles = [float(tree[n].to_dataset()["sweep_fixed_angle"]) for n in names]
        name = names[int(np.argmin(np.abs(np.array(angles) - float(sweep))))]
    ds = tree[name].to_dataset()
    for c in ("latitude", "longitude", "altitude"):
        if c not in ds.coords and c in tree.coords:
            ds = ds.assign_coords({c: tree[c]})
        elif c not in ds.coords and c in tree.to_dataset():
            ds = ds.assign_coords({c: tree.to_dataset()[c]})
    return ds


def radar_at_location(
    radar,
    latitude,
    longitude,
    altitude=None,
    *,
    sweep=0,
    fields=None,
    radius=None,
):
    """
    Radar variables in the gate above a ground location.

    Parameters
    ----------
    radar : xarray.DataTree, xarray.Dataset or sequence of them
        Radar volumes (xradar DataTrees) or sweeps, e.g. one volume per
        time; each gives one sample.
    latitude, longitude : float
        Location (degrees north and east), e.g. a disdrometer.
    altitude : float, optional
        Altitude of the location (m above sea level). Default: the radar
        altitude.
    sweep : int or float, optional
        Sweep index (``sweep_<n>``, default 0) or the fixed angle (degrees)
        of the sweep to take from DataTrees.
    fields : sequence of str, optional
        Variables to sample. Default: all variables on ``(azimuth, range)``.
    radius : float, optional
        Average the gates whose centres are within this horizontal distance
        (m) of the location (variables in dB averaged in linear units)
        instead of taking the nearest gate.

    Returns
    -------
    xarray.Dataset
        On ``time`` (time of the sampled ray): the fields, and
        ``beam_height`` (m above sea level), ``height_above_ground``
        (beam height above the location), ``gate_distance`` (horizontal
        distance from the gate to the location, m), ``range``, ``azimuth``
        and ``elevation`` of the gate. The beam is a straight line on an
        Earth of 4/3 its radius.

    Notes
    -----
    No published method is implemented here: the gate sampling is radarx's
    own, and the beam height is that of
    :func:`radarx.fundamentals.geometry.beam_center_height` (4/3 effective
    Earth radius; see that function).
    """
    rows = []
    for ds in _sweeps(radar, sweep):
        for c in ("latitude", "longitude", "altitude"):
            if c not in ds.coords and c not in ds:
                raise ValueError(f"the sweep has no radar {c!r}")
        lat0, lon0 = float(ds["latitude"]), float(ds["longitude"])
        alt0 = float(ds["altitude"])
        s_loc, az_loc = _ground_position(lat0, lon0, float(latitude), float(longitude))
        xl, yl = s_loc * np.sin(np.deg2rad(az_loc)), s_loc * np.cos(np.deg2rad(az_loc))
        rng = np.asarray(ds["range"].values, float)
        az = np.asarray(ds["azimuth"].values, float)
        el = (
            np.asarray(ds["elevation"].values, float)
            if "elevation" in ds
            else np.full(az.size, float(ds["sweep_fixed_angle"]))
        )
        re = 4.0 / 3.0 * EARTH_RADIUS
        h = beam_center_height(rng[None, :], el[:, None], alt0, re)
        s = re * np.arcsin(
            rng[None, :] * np.cos(np.deg2rad(el[:, None])) / (re + h - alt0)
        )
        x = s * np.sin(np.deg2rad(az[:, None]))
        y = s * np.cos(np.deg2rad(az[:, None]))
        dist = np.hypot(x - xl, y - yl)
        ia, ir = np.unravel_index(np.nanargmin(dist), dist.shape)
        names = fields or [
            v for v in ds.data_vars if set(ds[v].dims) == {"azimuth", "range"}
        ]
        row = {}
        for name in names:
            if name not in ds:
                row[name] = np.nan
                continue
            vals = ds[name].transpose("azimuth", "range").values.astype(float)
            if radius is None:
                row[name] = vals[ia, ir]
            else:
                sel = vals[dist <= float(radius)]
                units = str(ds[name].attrs.get("units", "")).lower()
                if units.startswith("db"):
                    with np.errstate(divide="ignore"):
                        row[name] = 10 * np.log10(np.nanmean(10 ** (sel / 10)))
                else:
                    row[name] = np.nanmean(sel) if np.isfinite(sel).any() else np.nan
        ray_time = ds["time"].values
        t = ray_time[ia] if np.ndim(ray_time) else ray_time
        alt = alt0 if altitude is None else float(altitude)
        row.update(
            time=np.datetime64(t, "ns"),
            beam_height=h[ia, ir],
            height_above_ground=h[ia, ir] - alt,
            gate_distance=dist[ia, ir],
            range=rng[ir],
            azimuth=az[ia],
            elevation=el[ia],
        )
        rows.append((row, {n: ds[n].attrs for n in names if n in ds}))
    if not rows:
        raise ValueError("no radar sweeps given")
    names = [k for k in rows[0][0] if k != "time"]
    time = np.array([r["time"] for r, _ in rows], "datetime64[ns]")
    out = xr.Dataset(
        {
            k: ("time", np.array([r.get(k, np.nan) for r, _ in rows], float))
            for k in names
        },
        coords={"time": time},
    )
    for n, a in rows[0][1].items():
        out[n].attrs = dict(a)
    geo = {
        "beam_height": ("Height of the beam centre above sea level", "m"),
        "height_above_ground": ("Height of the beam centre above the location", "m"),
        "gate_distance": ("Horizontal distance from the gate to the location", "m"),
        "range": ("Range of the gate", "m"),
        "azimuth": ("Azimuth of the gate", "degrees"),
        "elevation": ("Elevation of the gate", "degrees"),
    }
    for k, (ln, u) in geo.items():
        out[k].attrs = {"long_name": ln, "units": u}
    out = out.sortby("time").assign_coords(
        latitude=float(latitude), longitude=float(longitude)
    )
    return out


def _window_mean(t_src, values, weights, t_lo, t_hi):
    """Weighted mean of values (n, ...) with t_lo <= t_src < t_hi."""
    i0 = np.searchsorted(t_src, t_lo, side="left")
    i1 = np.searchsorted(t_src, t_hi, side="left")
    ok = np.isfinite(values).all(axis=tuple(range(1, values.ndim)))
    w = np.where(ok, weights, 0.0)
    v = np.where(ok.reshape((-1,) + (1,) * (values.ndim - 1)), values, 0.0)
    cw = np.concatenate([[0.0], np.cumsum(w)])
    cv = np.concatenate(
        [
            np.zeros((1,) + values.shape[1:]),
            np.cumsum(v * w.reshape((-1,) + (1,) * (values.ndim - 1)), axis=0),
        ]
    )
    sw = cw[i1] - cw[i0]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (cv[i1] - cv[i0]) / sw.reshape((-1,) + (1,) * (values.ndim - 1))
    return out, sw


def match_radar(
    ds,
    radar,
    *,
    sweep=0,
    fields=None,
    window="60s",
    delay=None,
    radius=None,
    band="S",
    temperature=20.0,
    qc="relative",
    engine="auto",
    n_threads=None,
):
    """
    Pair disdrometer spectra with the radar gate above the instrument.

    Parameters
    ----------
    ds : xarray.Dataset
        Disdrometer data of one station (``counts`` and the scalar
        coordinates ``latitude``, ``longitude``, ``altitude``), e.g. from
        :func:`radarx.io.read_parsivel`; or the output of
        :func:`process_disdrometer`/:func:`number_concentration` with an
        ``ND`` variable and the station coordinates.
    radar : xarray.DataTree, xarray.Dataset or sequence of them
        Radar volumes or sweeps, see :func:`radar_at_location`.
    sweep, fields, radius : optional
        See :func:`radar_at_location`.
    window : str or numpy.timedelta64, optional
        Length of the disdrometer averaging window centred on each radar
        time (plus ``delay``). Default 60 s.
    delay : None, "fall", str or numpy.timedelta64, optional
        Time drops take from the radar gate to the instrument: none
        (default), a fixed lag, or ``"fall"`` for the height of the gate
        above the instrument divided by the terminal fall speed at the
        mass-weighted mean diameter of the (undelayed) window (the Atlas et
        al. 1973 law, at least 0.1 m s-1, without density correction: a
        radarx choice). Advection by the wind is not modelled.
    band, temperature, qc, engine, n_threads : optional
        Options of :func:`process_disdrometer` for the radar variables of
        the averaged spectra.

    Returns
    -------
    xarray.Dataset
        On the radar ``time``: the radar fields and gate geometry of
        :func:`radar_at_location`, the window-averaged ``ND`` and, with the
        suffix ``_disdrometer``, the radar variables (``DBZH``, ``ZDR``,
        ``KDP``, ...) and moment quantities (``RAIN_RATE``, ``DM``, ``D0``,
        ``NW``, ``LWC``, ``NT``) of the averaged spectra, with
        ``n_records`` and ``delay`` (s).

    Notes
    -----
    The windows, delay and averaging are radarx's own construction (no
    published method); ``delay="fall"`` uses the fall speed of Atlas et al.
    (1973) at the mass-weighted mean diameter.

    References
    ----------
    Atlas, D., R. C. Srivastava, and R. S. Sekhon, 1973: Doppler radar
    characteristics of precipitation at vertical incidence. *Rev. Geophys.*,
    **11** (1), 1-35, https://doi.org/10.1029/RG011i001p00001

    Examples
    --------
    >>> pairs = match_radar(pips, [vol1, vol2, vol3], window="60s")
    ... # doctest: +SKIP
    >>> (pairs.DBZH - pairs.DBZH_disdrometer).mean()  # doctest: +SKIP
    """
    for c in ("latitude", "longitude"):
        if c not in ds.coords:
            raise ValueError(f"the disdrometer dataset has no {c!r} coordinate")
    if "station" in ds.dims:
        raise ValueError("select one station (ds.isel(station=i)) or loop over them")
    lat, lon = float(ds["latitude"]), float(ds["longitude"])
    alt = float(ds["altitude"]) if "altitude" in ds.coords else None
    if alt is not None and not np.isfinite(alt):
        alt = None
    pts = radar_at_location(
        radar, lat, lon, alt, sweep=sweep, fields=fields, radius=radius
    )
    if "ND" in ds:
        nd = ds["ND"].transpose("time", "diameter")
    else:
        if qc in ("relative", "raupach2015"):
            ds = disdrometer_qc(ds, method=qc)
        elif qc == "raupach_berne":
            ds = raupach_berne_correction(ds, engine=engine, n_threads=n_threads)
        elif qc is not None:
            raise ValueError(f"unknown qc {qc!r}")
        nd = number_concentration(ds, engine=engine, n_threads=n_threads)
    win = pd.to_timedelta(window).to_timedelta64().astype("timedelta64[ns]")
    t_src = nd["time"].values.astype("datetime64[ns]")
    weights = (
        ds["sample_interval"].values.astype(float)
        if "sample_interval" in ds
        else np.ones(t_src.size)
    )
    t_rad = pts["time"].values
    x = nd.values
    if delay is None:
        lag = np.zeros(t_rad.size, "timedelta64[ns]")
    elif isinstance(delay, str) and delay == "fall":
        avg, _ = _window_mean(t_src, x, weights, t_rad - win // 2, t_rad + win // 2)
        d = nd["diameter"].values
        w = _dsdmod._bin_widths(nd, "diameter").values
        with np.errstate(invalid="ignore", divide="ignore"):
            dm = (avg * w * d**4).sum(-1) / (avg * w * d**3).sum(-1)
        # fall speed of the mass-weighted mean drop, Atlas et al. (1973), at
        # least 0.1 m s-1, without density correction (radarx choice)
        vt = np.maximum(9.65 - 10.3 * np.exp(-0.6 * dm), 0.1)
        sec = pts["height_above_ground"].values / vt
        lag = np.where(np.isfinite(sec), sec * 1e9, 0).astype("timedelta64[ns]")
    else:
        lag = np.full(
            t_rad.size,
            pd.to_timedelta(delay).to_timedelta64().astype("timedelta64[ns]"),
        )
    centre = t_rad + lag
    avg, sw = _window_mean(t_src, x, weights, centre - win // 2, centre + win // 2)
    nd_avg = xr.DataArray(
        avg,
        dims=("time", "diameter"),
        coords={
            "time": pts["time"],
            "diameter": nd["diameter"],
            "bin_width": nd["bin_width"],
        },
        name="ND",
        attrs=dict(_ND_ATTRS),
    )
    mom = dsd_moments(nd_avg)
    out = pts.copy()
    out["ND"] = nd_avg
    good = np.isfinite(mom["NT"]) & (mom["NT"] > 0)
    rad = _dsdmod.radar_from_dsd(nd_avg.fillna(0.0), band=band, temperature=temperature)
    for name in _RADAR_VARS:
        out[f"{name}_disdrometer"] = rad[name].where(good)
        out[f"{name}_disdrometer"].attrs = dict(rad[name].attrs)
    for name in ("RAIN_RATE", "DM", "D0", "NW", "LWC", "NT", "DBZ_RAYLEIGH"):
        out[f"{name}_disdrometer"] = mom[name]
    out["n_records"] = (
        "time",
        np.searchsorted(t_src, centre + win // 2)
        - np.searchsorted(t_src, centre - win // 2),
    )
    out["n_records"].attrs = {
        "long_name": "Disdrometer records in the window",
        "units": "1",
    }
    out["delay"] = (
        "time",
        lag.astype(np.int64) / 1e9,
        {"long_name": "Fall delay", "units": "s"},
    )
    out.attrs = {
        "window": str(window),
        "band": rad.attrs["band"],
        "station": str(ds["station"].values) if "station" in ds.coords else "",
    }
    return out
