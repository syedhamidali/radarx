#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Rain Evaporation and Evaporative Cooling
========================================

Bulk evaporation rate of rain and the resulting cooling of the air, from the
parameters of a gamma raindrop size distribution (DSD, e.g. from
:func:`radarx.retrieve.dsd`) and the temperature, pressure and humidity of the
air (a sounding or ERA5 profile from :mod:`radarx.io.sounding`, or fields on
the same coordinates).

Single drop
-----------
A drop of diameter :math:`D` falling through air with the saturation ratio
:math:`S = e/e_s` changes its mass by diffusion of water vapour (Rogers and
Yau 1989; Pruppacher and Klett 1997). The form used here is that of
Kumjian and Ryzhkov (2010, their Eq. 2 and Appendix Eqs. A1-A2, written
there for the radius, :math:`r\\,dr/dt = (S - 1)/(F_K + F_D)`), rewritten
for the mass of the drop

.. math::

    \\frac{dm}{dt} = \\frac{2 \\pi D f_v(D) (S - 1)}{F_K + F_D},\\qquad
    F_K = \\left(\\frac{L_v}{R_v T} - 1\\right) \\frac{L_v}{K T},\\qquad
    F_D = \\frac{R_v T}{D_v e_s(T)},

with the ventilation coefficient of Pruppacher and Klett (1997), as given
by Li and Srivastava (2001, their Eq. 2), Kumjian and Ryzhkov (2010,
Appendix Eq. A3) and Seifert (2008, his Eq. 8, who uses the same 0.78 and
0.308)

.. math::

    f_v = 0.78 + 0.308\\, N_{Sc}^{1/3} N_{Re}^{1/2},\\qquad
    N_{Sc} = \\nu / D_v,\\quad N_{Re} = V(D) D / \\nu .

The drop is assumed at its equilibrium (wet-bulb) temperature, which is what
the single-drop law of Rogers and Yau (1989) is derived for, and the
ventilation coefficients for vapour and heat are taken equal (so that
:math:`f_v` multiplies both :math:`F_K` and :math:`F_D`). The numbers 0.78 and
0.308 are those printed in the three papers above (they are Pruppacher and
Klett's; the equation is not checked in the book).

Differences from Kumjian and Ryzhkov (2010)
-------------------------------------------
radarx takes the single-drop law and the thermodynamic fits of Kumjian and
Ryzhkov (2010) but integrates it over a gamma DSD in closed form instead of
following size bins down a rain shaft with feedback on the sounding. It
deviates from the paper in these points (the first four change the numbers):

- the ventilation coefficient for heat is set equal to that for vapour
  (Kumjian and Ryzhkov, Eq. A3, use the Prandtl number for heat);
- the saturation vapour pressure is Buck (1981) instead of their
  :math:`e_s = A \\exp(-5420/T)`, :math:`A = 2.53\\times10^9` hPa (Eq. A8);
- the fall speed is the exponential fit to Atlas et al. (1973), below,
  instead of the power law :math:`3.78 D^{0.67} (\\rho_0/\\rho)^{0.4}` m
  s\\ :sup:`-1` of Atlas and Ulbrich (1977) in their Eq. 3;
- the air temperature and humidity are those of the profile at the cell
  (or stepped by :func:`integrate_evaporation`), not the layer-by-layer
  feedback of their model;
- collision, coalescence and breakup are absent in both.

Thermodynamic properties
------------------------
- saturation vapour pressure over water, Buck (1981, his Eq. 8 for liquid
  water; equation number and constants not checked against the paper):
  :math:`e_s = 611.21 \\exp[17.502\\, t / (240.97 + t)]` Pa, :math:`t` in °C.
  Buck's enhancement factor of moist air, :math:`1.0007 + 3.46\\times10^{-8}
  p` with :math:`p` in Pa (about 1.004 at 1000 hPa), is neglected (radarx
  choice);
- latent heat of vaporization, thermal conductivity of air :math:`K`,
  diffusivity of water vapour :math:`D_v` and dynamic viscosity of air
  :math:`\\eta` as functions of temperature (:math:`T` in K) and pressure,
  copied from the Appendix of Kumjian and Ryzhkov (2010), who state that
  the dependence on :math:`T` follows Rasmussen and Heymsfield (1987,
  Part I) and give the equations in SI units:
  :math:`L_v = 2.499 \\times 10^6 (273.15/T)^{0.167 + 3.67\\times10^{-4} T}`
  J kg\\ :sup:`-1` (Eqs. A4-A5),
  :math:`K = (0.441635 + 0.0071 T) \\times 10^{-2}`
  W m\\ :sup:`-1` K\\ :sup:`-1` (Eq. A6),
  :math:`D_v = 2.11 \\times 10^{-5} (T / 273.15)^{1.94} (p_0/p)`
  m\\ :sup:`2` s\\ :sup:`-1` (Eq. A7),
  :math:`\\eta = (0.379565 + 0.0049 T) \\times 10^{-5}` kg m\\ :sup:`-1`
  s\\ :sup:`-1` (Eq. A10, stated for :math:`T > 273` K; radarx applies it
  at all temperatures);
- air density of moist air from the ideal gas law (as in Kumjian and Ryzhkov
  2010, after Eq. A10), specific heat
  :math:`c_p = 1005.7 (1 - q_v) + 1870 q_v` J kg\\ :sup:`-1` K\\ :sup:`-1`
  and the gas constants 287.04 and 461.5 J kg\\ :sup:`-1` K\\ :sup:`-1`
  are common textbook values chosen by radarx, not taken from the cited
  papers.

Reference pressure of :math:`D_v`
---------------------------------
Eq. A7 of Kumjian and Ryzhkov (2010) states that :math:`p_0` "is the
reference level pressure, taken as 1000 hPa in this study", and radarx uses
that value (``1.0e5 / p`` in ``_air``). The fit of Pruppacher and Klett
(1997), from which this form originates, is defined for
:math:`p_0 = 1013.25` hPa (not checked in the book). With 1013.25 hPa :math:`D_v` would be 1.3 % larger,
:math:`F_D` 1.3 % smaller and the evaporation rates larger by 0.3-0.8 %
(computed for 0-30 °C at 800-1000 hPa and :math:`q_v` = 5 g kg\\ :sup:`-1`;
the larger the colder the air, as :math:`F_D` is a larger part of
:math:`F_K + F_D` there). radarx rates are therefore, if that reading of the
fit is right, up to 0.8 % too small. This is smaller than the uncertainty of the
ventilation coefficient and of the DSD, and the value is kept for
consistency with Kumjian and Ryzhkov (2010).

Fall speed
----------
The terminal fall speed of Atlas et al. (1973),
:math:`V = 9.65 - 10.3 e^{-0.6 D}` m s\\ :sup:`-1` (:math:`D` in mm; the law
is quoted at sea level as Eq. 7.65b of Bringi and Chandrasekar 2001, where
it is described as a fit to the Gunn and Kinzer 1949 measurements), does
not integrate in closed form under the square root of the ventilation term.
The Atlas et al. law is negative below :math:`D` = 0.109 mm and its range of
validity in the original paper is not checked. radarx uses the exponential
fit below, which is meant for 0.5-7 mm; its extrapolation to small drops is a
radarx choice.
It is therefore represented by :math:`V = a D^b e^{-f D}` with
:math:`a = 4.643` m s\\ :sup:`-1` mm\\ :sup:`-b`, :math:`b = 0.9496`,
:math:`f = 0.1671` mm\\ :sup:`-1`, a least-squares fit (relative error) to
the Atlas et al. (1973) law for 0.5-7 mm (radarx's own fit, not from the
cited papers): 2 % r.m.s. and at most 9.5 % (at 0.5 mm; -3.7 % at 7 mm), i.e. at most 5 %
on :math:`V^{1/2}`. Other coefficients can be given
(``fall_speed=(a, b, f)``). Fall speeds aloft are increased by
:math:`(\\rho_0 / \\rho)^{0.4}`, the correction attributed to Foote and
du Toit (1969) in Li and Srivastava (2001, text after their Eq. 4:
:math:`V = V_m (\\rho_m/\\rho)^{0.4}`) and Kumjian and Ryzhkov (2010, Eq.
3); the equation and range of densities of the original paper are not
checked. :math:`\\rho_0` = 1.204 kg
m\\ :sup:`-3` is the density of dry air at 1013.25 hPa and 20 °C, the
density at which the sea-level fall speeds are taken to hold (a radarx
choice; the cited papers only call it the surface reference density).

Bulk rates
----------
With :math:`N(D) = N_0 D^\\mu e^{-\\Lambda D}` (untruncated) all integrals
are closed-form gamma-function moments. Writing

.. math::

    I_k = \\int_0^\\infty D^k f_v(D) N(D)\\, dD
        = N_0 \\left[0.78 \\frac{\\Gamma(\\mu + k + 1)}{\\Lambda^{\\mu + k + 1}}
        + B \\frac{\\Gamma(\\mu + k + 1 + \\frac{b + 1}{2})}
        {(\\Lambda + f/2)^{\\mu + k + 1 + \\frac{b + 1}{2}}}\\right],\\quad
    B = 0.308\\, N_{Sc}^{1/3} \\sqrt{a (\\rho_0/\\rho)^{0.4} / \\nu},

(in consistent units) the rate of evaporation per unit volume is
:math:`E_v = -2\\pi (S - 1) I_1 / (F_K + F_D)`, the evaporation rate
:math:`E = E_v / \\rho` (kg kg\\ :sup:`-1` s\\ :sup:`-1`, positive for
evaporation), the cooling rate :math:`L_v E / c_p` (K s\\ :sup:`-1`), and the
tendency of the (Rayleigh) reflectivity factor
:math:`dZ/dt = 6 \\int D^5 (dD/dt) N\\, dD \\propto (S - 1) I_4`, which does
not depend on :math:`N_0` (the last two expressions are derived here from the
single-drop law, not taken from a paper). The closed-form integration of the
ventilated single-drop law over a gamma DSD is the one of bulk microphysics
schemes: the ventilation factor integrated over a gamma DSD is Eq. 8 (with
the thermodynamic function of Eq. 9) of Milbrandt and Yau (2005, Part II),
and the same approach is used by Ferrier (1994; equation number not checked).
radarx's moment :math:`I_1` is the same integral with the fall-speed law of
this module. Supersaturated air gives negative rates (growth by
condensation, a radarx extension); saturated air none.

Time integration
----------------
:func:`integrate_evaporation` follows the temperature and humidity of the air
at each cell (QVP height, gate, grid point) through a sequence of DSDs, e.g.
radar volumes: between two volumes the DSD of the earlier one is held, and
:math:`T` and :math:`q_v` are stepped (sub-steps of at most ``max_step``)
with :math:`q_v \\mathrel{+}= E \\Delta t` and
:math:`T \\mathrel{-}= L_v E \\Delta t / c_p`, recomputing the thermodynamic
properties every sub-step; each sub-step is a Heun (trapezoidal
predictor-corrector) step, second-order accurate in time. A sub-step never
evaporates more than brings the air to saturation (linearized saturation
adjustment; a radarx choice, not from the cited papers). Pressure is constant,
and there is no advection, mixing or vertical motion: the result is the
cooling a column would feel if it stayed under the observed rain.

Both computations run in a compiled kernel (``radarx.retrieve._evaporation``,
multithreaded over all cells of all inputs) with an identical NumPy reference
implementation as fallback.

References
----------
Atlas, D., R. C. Srivastava, and R. S. Sekhon, 1973: Doppler radar
characteristics of precipitation at vertical incidence. *Rev. Geophys.*,
**11** (1), 1-35, https://doi.org/10.1029/RG011i001p00001

Atlas, D., and C. W. Ulbrich, 1977: Path- and area-integrated rainfall
measurement by microwave attenuation in the 1-3 cm band. *J. Appl. Meteor.*,
**16** (12), 1322-1331,
https://doi.org/10.1175/1520-0450(1977)016<1322:PAAIRM>2.0.CO;2

Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler Weather
Radar: Principles and Applications*. Cambridge University Press, 636 pp.,
https://doi.org/10.1017/CBO9780511541094

Buck, A. L., 1981: New equations for computing vapor pressure and
enhancement factor. *J. Appl. Meteor.*, **20** (12), 1527-1532,
https://doi.org/10.1175/1520-0450(1981)020<1527:NEFCVP>2.0.CO;2

Ferrier, B. S., 1994: A double-moment multiple-phase four-class bulk ice
scheme. Part I: Description. *J. Atmos. Sci.*, **51** (2), 249-280,
https://doi.org/10.1175/1520-0469(1994)051<0249:ADMMPF>2.0.CO;2

Foote, G. B., and P. S. du Toit, 1969: Terminal velocity of raindrops aloft.
*J. Appl. Meteor.*, **8** (2), 249-253,
https://doi.org/10.1175/1520-0450(1969)008<0249:TVORA>2.0.CO;2

Gunn, R., and G. D. Kinzer, 1949: The terminal velocity of fall for water
droplets in stagnant air. *J. Meteor.*, **6** (4), 243-248,
https://doi.org/10.1175/1520-0469(1949)006<0243:TTVOFF>2.0.CO;2

Kumjian, M. R., and A. V. Ryzhkov, 2010: The impact of evaporation on
polarimetric characteristics of rain: Theoretical model and practical
implications. *J. Appl. Meteor. Climatol.*, **49** (6), 1247-1267,
https://doi.org/10.1175/2010JAMC2243.1

Li, X., and R. C. Srivastava, 2001: An analytical solution for raindrop
evaporation and its application to radar rainfall measurements. *J. Appl.
Meteor.*, **40** (9), 1607-1616,
https://doi.org/10.1175/1520-0450(2001)040<1607:AASFRE>2.0.CO;2

Milbrandt, J. A., and M. K. Yau, 2005: A multimoment bulk microphysics
parameterization. Part II: A proposed three-moment closure and scheme
description. *J. Atmos. Sci.*, **62** (9), 3065-3081,
https://doi.org/10.1175/JAS3535.1

Pruppacher, H. R., and J. D. Klett, 1997: *Microphysics of Clouds and
Precipitation*. 2nd rev. and enl. ed., Kluwer Academic Publishers (reprinted by
Springer, 2010), https://doi.org/10.1007/978-0-306-48100-0

Rasmussen, R. M., and A. J. Heymsfield, 1987: Melting and shedding of
graupel and hail. Part I: Model physics. *J. Atmos. Sci.*, **44** (19),
2754-2763,
https://doi.org/10.1175/1520-0469(1987)044<2754:MASOGA>2.0.CO;2

Rogers, R. R., and M. K. Yau, 1989: *A Short Course in Cloud Physics*. 3rd
ed., Elsevier (Butterworth-Heinemann), 290 pp., ISBN 978-0-7506-3215-7 (no
DOI).

Seifert, A., 2008: On the parameterization of evaporation of raindrops as
simulated by a one-dimensional rainshaft model. *J. Atmos. Sci.*, **65** (11),
3608-3619, https://doi.org/10.1175/2008JAS2586.1

.. autosummary::
   :nosignatures:
   :toctree: generated/

   evaporation
   integrate_evaporation
   drop_evaporation_rate
"""

from __future__ import annotations

__all__ = ["evaporation", "integrate_evaporation", "drop_evaporation_rate"]

import math

import numpy as np
import xarray as xr
from scipy.special import gammaln

from .._registry import accessor_method

try:
    from . import _evaporation

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _evaporation = None
    HAS_COMPILED_KERNEL = False

RD = 287.04  # J kg-1 K-1, dry air
RV = 461.5  # J kg-1 K-1, water vapour
EPS = RD / RV
CPD = 1005.7  # J kg-1 K-1
CPV = 1870.0  # J kg-1 K-1
RHO_W = 1000.0  # kg m-3
T0 = 273.15
# ventilation coefficient f_v = 0.78 + 0.308 N_Sc^(1/3) N_Re^(1/2): Pruppacher
# and Klett (1997) as printed in Li and Srivastava (2001), Eq. 2, Kumjian and
# Ryzhkov (2010), Eq. A3 and Seifert (2008), Eq. 8
VENTILATION = (0.78, 0.308)
# kg m-3, dry air at 1013.25 hPa and 20 degC: the density at which the sea-level
# fall speeds are taken to hold (radarx choice; Foote and du Toit 1969 call it
# the surface reference density)
RHO0 = 1.204
# V = a D^b exp(-f D) (D in mm, V in m s-1): radarx least-squares fit (relative
# error) to Atlas et al. (1973), V = 9.65 - 10.3 exp(-0.6 D), over 0.5-7 mm
FALL_SPEED = {"atlas1973": (4.643, 0.9496, 0.1671)}

_OUT_NAMES = (
    "EVAPORATION_RATE",
    "COOLING_RATE",
    "COOLING_RATE_HOURLY",
    "DBZ_TENDENCY",
    "SATURATION_DEFICIT",
)
_OUT_ATTRS = {
    "EVAPORATION_RATE": {
        "long_name": "Mass of rain evaporated per unit mass of air per unit time",
        "units": "kg kg-1 s-1",
        "comment": "positive for evaporation, negative for condensational growth",
    },
    "COOLING_RATE": {
        "long_name": "Cooling rate of the air by evaporation of rain",
        "units": "K s-1",
        "comment": "positive for cooling; the temperature tendency is its negative",
    },
    "COOLING_RATE_HOURLY": {
        "long_name": "Cooling rate of the air by evaporation of rain",
        "units": "K h-1",
        "comment": "positive for cooling; the temperature tendency is its negative",
    },
    "DBZ_TENDENCY": {
        "long_name": "Tendency of the Rayleigh reflectivity factor by evaporation",
        "units": "dB h-1",
        "comment": "10 log10 rate of change of the sixth moment of the DSD",
    },
    "SATURATION_DEFICIT": {
        "long_name": "Saturation ratio minus one, e / e_s(T) - 1",
        "units": "1",
    },
}
_STATE_ATTRS = {
    "temperature": {
        "standard_name": "air_temperature",
        "long_name": "Air temperature",
        "units": "K",
    },
    "specific_humidity": {
        "standard_name": "specific_humidity",
        "long_name": "Specific humidity",
        "units": "kg kg-1",
    },
    "relative_humidity": {
        "standard_name": "relative_humidity",
        "long_name": "Relative humidity over water",
        "units": "1",
    },
    "TEMPERATURE_CHANGE": {
        "long_name": "Accumulated change of air temperature by evaporation of rain",
        "units": "K",
    },
}
_DSD_NAMES = ("N0", "MU", "LAMBDA")


# --------------------------------------------------------------------------
# NumPy reference (the compiled kernel follows it in the same order)
# --------------------------------------------------------------------------


def _saturation_vapor_pressure(t):
    """Buck (1981), Eq. 8, saturation vapour pressure over water [Pa], ``t`` in K.

    The enhancement factor of moist air is neglected (radarx choice).
    """
    tc = t - T0
    return 611.21 * np.exp(17.502 * tc / (240.97 + tc))


def _air(t, p, qv):
    """Thermodynamic properties of moist air (all SI)."""
    es = _saturation_vapor_pressure(t)
    e = qv * p / (EPS + (1.0 - EPS) * qv)
    rho = p / (RD * t * (1.0 + (1.0 / EPS - 1.0) * qv))
    # Kumjian and Ryzhkov (2010), Eqs. A4-A5 (L_v), A6 (K), A10 (eta, below)
    lv = 2.499e6 * (T0 / t) ** (0.167 + 3.67e-4 * t)
    k = (0.441635 + 0.0071 * t) * 1.0e-2
    # Kumjian and Ryzhkov (2010), Eq. A7, with their p0 = 1000 hPa (see the
    # module docstring: the fit of Pruppacher and Klett uses 1013.25 hPa)
    dv = 2.11e-5 * (t / T0) ** 1.94 * (1.0e5 / p)
    nu = (0.379565 + 0.0049 * t) * 1.0e-5 / rho
    fkd = (lv / (RV * t) - 1.0) * lv / (k * t) + RV * t / (dv * es)
    cp = CPD * (1.0 - qv) + CPV * qv
    qs = EPS * es / (p - (1.0 - EPS) * es)
    return {
        "es": es,
        "ssat": e / es - 1.0,
        "rho": rho,
        "lv": lv,
        "dv": dv,
        "nu": nu,
        "fkd": fkd,
        "cp": cp,
        "qs": qs,
    }


def _moment(logn0, mu, lam, k, bterm, fall, vent):
    """``I_k`` of the ventilated gamma DSD (D in mm, N0 in m-3 mm-(1+mu))."""
    _, b, f = fall
    x1 = mu + k + 1.0
    x2 = x1 + 0.5 * (b + 1.0)
    return vent[0] * np.exp(logn0 + gammaln(x1) - x1 * np.log(lam)) + bterm * np.exp(
        logn0 + gammaln(x2) - x2 * np.log(lam + 0.5 * f)
    )


def _rates_numpy(n0, mu, lam, t, p, qv, fall, vent, rho0):
    """Evaporation rate, cooling rate, dBZ tendency and S - 1 per cell."""
    a, b, f = fall
    with np.errstate(all="ignore"):
        air = _air(t, p, qv)
        corr = (rho0 / air["rho"]) ** 0.4
        bterm = (
            vent[1]
            * (air["nu"] / air["dv"]) ** (1.0 / 3.0)
            * np.sqrt(1.0e-3 * corr * a / air["nu"])
        )
        ok = np.isfinite(n0) & np.isfinite(mu) & np.isfinite(lam) & (n0 > 0) & (lam > 0)
        logn0 = np.log(np.where(ok, n0, 1.0))
        mu_ = np.where(ok, mu, 0.0)
        lam_ = np.where(ok, lam, 1.0)
        i1 = _moment(logn0, mu_, lam_, 1.0, bterm, fall, vent)
        i4 = _moment(logn0, mu_, lam_, 4.0, bterm, fall, vent)
        z = np.exp(logn0 + gammaln(mu_ + 7.0) - (mu_ + 7.0) * np.log(lam_))
        evol = -2.0e-3 * math.pi * air["ssat"] / air["fkd"] * i1
        evap = evol / air["rho"]
        cool = air["lv"] * evap / air["cp"]
        zrate = 2.4e7 * air["ssat"] / (RHO_W * air["fkd"]) * i4
        dbz = 3600.0 * 10.0 / math.log(10.0) * zrate / z
        # no rain (N0 = 0) evaporates nothing; missing DSD or air: NaN
        zero = np.isfinite(n0) & (n0 == 0) & np.isfinite(air["ssat"])
        out = []
        for x in (evap, cool, dbz):
            x = np.where(ok, x, np.nan)
            out.append(np.where(zero, 0.0, x))
        out.append(air["ssat"])
    return out


def _limit(dq, gap):
    """Limit a vapour increment to the saturation adjustment ``gap``."""
    dq = np.where(dq > 0.0, np.minimum(dq, np.maximum(gap, 0.0)), dq)
    return np.where(dq < 0.0, np.maximum(dq, np.minimum(gap, 0.0)), dq)


def _integrate_numpy(n0, mu, lam, t0, qv0, p, dt, max_step, fall, vent, rho0):
    """March T and qv through the DSDs (``(nt, n)``); see the module notes."""
    nt = n0.shape[0]
    t = np.array(t0, dtype=np.float64, copy=True)
    q = np.array(qv0, dtype=np.float64, copy=True)
    shape = (nt,) + t.shape
    t_out = np.empty(shape)
    q_out = np.empty(shape)
    e_out = np.empty(shape)
    c_out = np.empty(shape)
    for i in range(nt):
        t_out[i] = t
        q_out[i] = q
        e, c, _, _ = _rates_numpy(n0[i], mu[i], lam[i], t, p, q, fall, vent, rho0)
        e_out[i] = e
        c_out[i] = c
        if i == nt - 1:
            break
        nsub = max(1, int(math.ceil(dt[i] / max_step)))
        h = dt[i] / nsub
        for _ in range(nsub):
            # Heun (trapezoidal predictor-corrector) step, limited at saturation
            with np.errstate(all="ignore"):
                air = _air(t, p, q)
                gap = (air["qs"] - q) / (
                    1.0 + air["lv"] ** 2 * air["qs"] / (air["cp"] * RV * t * t)
                )
                e1 = _rates_numpy(n0[i], mu[i], lam[i], t, p, q, fall, vent, rho0)[0]
                e1 = np.where(np.isfinite(e1), e1, 0.0)
                dq = _limit(e1 * h, gap)
                t1 = t - air["lv"] * dq / air["cp"]
                e2 = _rates_numpy(
                    n0[i], mu[i], lam[i], t1, p, q + dq, fall, vent, rho0
                )[0]
                e2 = np.where(np.isfinite(e2), e2, 0.0)
                dq = _limit(0.5 * (e1 + e2) * h, gap)
                q = q + dq
                t = t - air["lv"] * dq / air["cp"]
    return t_out, q_out, e_out, c_out


# --------------------------------------------------------------------------
# engine selection
# --------------------------------------------------------------------------


def _use_compiled(engine):
    """Whether to run the compiled kernel for the requested ``engine``."""
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled evaporation kernel is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


def _fall(fall_speed):
    if isinstance(fall_speed, str):
        if fall_speed not in FALL_SPEED:
            raise ValueError(
                f"fall_speed must be one of {tuple(FALL_SPEED)} or (a, b, f), "
                f"not {fall_speed!r}"
            )
        return FALL_SPEED[fall_speed]
    a, b, f = (float(v) for v in fall_speed)
    if not (a > 0 and b > -1 and f >= 0):
        raise ValueError("fall_speed (a, b, f) needs a > 0, b > -1 and f >= 0")
    return a, b, f


def _f64(x):
    return np.ascontiguousarray(x, dtype=np.float64)


def _rates(arrays, fall, vent, rho0, n_threads, use_compiled):
    """Rates for a list of (n0, mu, lam, t, p, qv) 1-D arrays, in one call."""
    sizes = [a[0].size for a in arrays]
    cat = [_f64(np.concatenate([a[j] for a in arrays])) for j in range(6)]
    if use_compiled:
        res = np.asarray(
            _evaporation.rates(
                *cat,
                fall[0],
                fall[1],
                fall[2],
                vent[0],
                vent[1],
                rho0,
                int(n_threads or 0),
            )
        )
    else:
        res = np.stack(_rates_numpy(*cat, fall, vent, rho0))
    return np.split(res, np.cumsum(sizes)[:-1], axis=1)


# --------------------------------------------------------------------------
# xarray layer
# --------------------------------------------------------------------------


def drop_evaporation_rate(
    diameter, temperature, pressure, specific_humidity, *, fall_speed="atlas1973"
):
    """
    Evaporation rate of single raindrops.

    Parameters
    ----------
    diameter : float, array-like or xarray.DataArray
        Equal-volume drop diameter (mm).
    temperature, pressure, specific_humidity : float, array-like or xarray.DataArray
        Air temperature (K), pressure (Pa) and specific humidity (kg kg-1).
    fall_speed : "atlas1973" or (float, float, float), optional
        Fall speed :math:`V = a D^b e^{-f D}` (m s-1, :math:`D` in mm) at
        :math:`\\rho_0` = 1.204 kg m-3; see :mod:`radarx.retrieve.evaporation`.

    Returns
    -------
    xarray.Dataset
        ``mass_rate`` :math:`dm/dt` (kg s-1, negative for evaporation) and
        ``diameter_rate`` :math:`dD/dt` (mm s-1), broadcast over the inputs.

    Notes
    -----
    The single-drop law (Rogers and Yau 1989, in the form of Kumjian and
    Ryzhkov 2010), the ventilation coefficient (Pruppacher and Klett 1997),
    the thermodynamic fits (Rasmussen and Heymsfield 1987, as given by Kumjian
    and Ryzhkov), the fall-speed fit to Atlas et al. (1973) and the
    differences from the cited papers are described in
    :mod:`radarx.retrieve.evaporation` (the :math:`D_v` reference pressure is
    1000 hPa as in Kumjian and Ryzhkov 2010, Eq. A7).

    References
    ----------
    Kumjian, M. R., and A. V. Ryzhkov, 2010: The impact of evaporation on
    polarimetric characteristics of rain: Theoretical model and practical
    implications. *J. Appl. Meteor. Climatol.*, **49** (6), 1247-1267,
    https://doi.org/10.1175/2010JAMC2243.1

    Rogers, R. R., and M. K. Yau, 1989: *A Short Course in Cloud Physics*. 3rd
    ed., Elsevier (Butterworth-Heinemann), 290 pp., ISBN 978-0-7506-3215-7.

    Pruppacher, H. R., and J. D. Klett, 1997: *Microphysics of Clouds and
    Precipitation*. 2nd rev. and enl. ed., Kluwer Academic Publishers
    (reprinted by Springer, 2010),
    https://doi.org/10.1007/978-0-306-48100-0

    Rasmussen, R. M., and A. J. Heymsfield, 1987: Melting and shedding of
    graupel and hail. Part I: Model physics. *J. Atmos. Sci.*, **44** (19),
    2754-2763,
    https://doi.org/10.1175/1520-0469(1987)044<2754:MASOGA>2.0.CO;2

    Atlas, D., R. C. Srivastava, and R. S. Sekhon, 1973: Doppler radar
    characteristics of precipitation at vertical incidence. *Rev.
    Geophys.*, **11** (1), 1-35, https://doi.org/10.1029/RG011i001p00001

    Examples
    --------
    >>> drop_evaporation_rate(1.0, 293.15, 9.0e4, 0.008)  # doctest: +SKIP
    """
    a, b, f = _fall(fall_speed)
    d, t, p, q = (
        x if isinstance(x, xr.DataArray) else xr.DataArray(np.asarray(x, dtype=float))
        for x in (diameter, temperature, pressure, specific_humidity)
    )
    air = _air(t, p, q)
    v = (RHO0 / air["rho"]) ** 0.4 * a * d**b * np.exp(-f * d)
    dm = d * 1.0e-3
    re = v * dm / air["nu"]
    vent = VENTILATION[0] + VENTILATION[1] * (air["nu"] / air["dv"]) ** (
        1.0 / 3.0
    ) * np.sqrt(re)
    mass = 2.0 * math.pi * dm * vent * air["ssat"] / air["fkd"]
    ddt = 1.0e3 * 2.0 * mass / (math.pi * RHO_W * dm**2)
    return xr.Dataset(
        {
            "mass_rate": mass.assign_attrs(
                long_name="Rate of change of drop mass", units="kg s-1"
            ),
            "diameter_rate": ddt.assign_attrs(
                long_name="Rate of change of drop diameter", units="mm s-1"
            ),
        }
    )


def _height_of(ds):
    """Heights above sea level (m) of the cells of a DSD dataset."""
    for name in ("z", "height", "altitude"):
        if name in ds.coords or name in ds.dims:
            return ds[name]
    raise ValueError(
        "the DSD needs a 'z' or 'height' coordinate (m above sea level) to "
        "interpolate a profile to it"
    )


def _from_rh(t, p, rh):
    e = rh * _saturation_vapor_pressure(t)
    return EPS * e / (p - (1.0 - EPS) * e)


def _environment(ds, environment, temperature, pressure, qv, rh, engine, n_threads):
    """Temperature (K), pressure (Pa) and specific humidity on the DSD cells."""
    env = {}
    if environment is not None:
        if not isinstance(environment, xr.Dataset):
            raise TypeError("environment must be an xarray.Dataset")
        want = [
            v
            for v in (
                "temperature",
                "pressure",
                "specific_humidity",
                "relative_humidity",
            )
            if v in environment.data_vars
        ]
        if "height" in environment.dims and not (
            "height" in ds.dims
            and "height" in environment.coords
            and environment.sizes["height"] == ds.sizes["height"]
            and np.array_equal(environment["height"].values, ds["height"].values)
        ):
            from ..io.sounding import interpolate_profile

            environment = interpolate_profile(
                environment, _height_of(ds), want, engine=engine, n_threads=n_threads
            )
        env = {v: environment[v] for v in want}
    for name, value in (
        ("temperature", temperature),
        ("pressure", pressure),
        ("specific_humidity", qv),
        ("relative_humidity", rh),
    ):
        if value is not None:
            env[name] = (
                value if isinstance(value, xr.DataArray) else xr.DataArray(value)
            )
    for name in ("temperature", "pressure"):
        if name not in env:
            raise KeyError(f"no {name} given (environment or {name}=)")
    t, p = env["temperature"], env["pressure"]
    if "specific_humidity" in env and (qv is not None or rh is None):
        q = env["specific_humidity"]
    elif "relative_humidity" in env:
        rhv = env["relative_humidity"]
        if rhv.attrs.get("units") == "%":
            rhv = rhv / 100.0
        q = _from_rh(t, p, rhv)
    else:
        raise KeyError("no humidity given (specific_humidity or relative_humidity)")
    return t, p, q


def _dsd_vars(ds):
    missing = [v for v in _DSD_NAMES if v not in ds.data_vars]
    if missing:
        raise KeyError(
            f"the DSD dataset lacks {missing}; use the output of radarx.retrieve.dsd"
        )
    return [ds[v] for v in _DSD_NAMES]


def _broadcast(ds, t, p, q):
    """NumPy arrays of N0, mu, Lambda, T, p, qv on the common shape."""
    n0, mu, lam = _dsd_vars(ds)
    arrs = xr.broadcast(n0, mu, lam, t, p, q)
    ref = arrs[0]
    dims = ref.dims
    flat = [_f64(a.transpose(*dims).values).ravel() for a in arrs]
    coords = {k: v for k, v in ref.coords.items() if set(v.dims) <= set(dims)}
    return flat, dims, ref.shape, coords


def _wrap(res, dims, shape, coords, attrs):
    data_vars = {}
    evap, cool, dbz, ssat = res
    values = {
        "EVAPORATION_RATE": evap,
        "COOLING_RATE": cool,
        "COOLING_RATE_HOURLY": cool * 3600.0,
        "DBZ_TENDENCY": dbz,
        "SATURATION_DEFICIT": ssat,
    }
    for name in _OUT_NAMES:
        data_vars[name] = (dims, values[name].reshape(shape), dict(_OUT_ATTRS[name]))
    return xr.Dataset(data_vars, coords=coords, attrs=dict(attrs))


def _attrs(fall, rho0):
    return {
        "fall_speed": f"V = {fall[0]:g} D^{fall[1]:g} exp(-{fall[2]:g} D) m s-1 "
        f"(D in mm) at {rho0:g} kg m-3, times (rho0/rho)^0.4",
        "comment": (
            "bulk evaporation of rain from a gamma DSD by radarx.retrieve.evaporation: "
            "ventilated diffusional evaporation integrated in closed form"
        ),
        "references": (
            "Kumjian and Ryzhkov (2010), doi:10.1175/2010JAMC2243.1; "
            "Buck (1981), doi:10.1175/1520-0450(1981)020<1527:NEFCVP>2.0.CO;2; "
            "Atlas et al. (1973), doi:10.1029/RG011i001p00001; "
            "Rasmussen and Heymsfield (1987), "
            "doi:10.1175/1520-0469(1987)044<2754:MASOGA>2.0.CO;2"
        ),
    }


def _env_for_node(environment, name):
    if isinstance(environment, xr.DataTree):
        if name not in environment.children:
            return None
        return environment[name].to_dataset()
    return environment


def evaporation(
    dsd,
    environment=None,
    *,
    temperature=None,
    pressure=None,
    specific_humidity=None,
    relative_humidity=None,
    fall_speed="atlas1973",
    rho0=RHO0,
    n_threads=None,
    engine="auto",
):
    """
    Evaporation and evaporative cooling rates of rain from a gamma DSD.

    Parameters
    ----------
    dsd : xarray.Dataset or xarray.DataTree
        Gamma DSD parameters ``N0`` (m-3 mm-(1+mu)), ``MU`` and ``LAMBDA``
        (mm-1), as returned by :func:`radarx.retrieve.dsd` for a sweep, a
        grid, a QVP or QVP time series; or a volume with such ``sweep_*``
        nodes (nodes without them are skipped). Only rain should be passed
        (mask the DSD retrieval); levels above the melting layer give
        meaningless rates.
    environment : xarray.Dataset or xarray.DataTree, optional
        ``temperature`` (K), ``pressure`` (Pa) and ``specific_humidity``
        (kg kg-1) or ``relative_humidity`` (fraction, or ``units="%"``),
        either as a vertical profile on ``height`` (e.g. from
        :func:`radarx.io.sounding.era5_profile` or
        :func:`radarx.io.sounding.read_sounding`), which is interpolated
        with :func:`radarx.io.sounding.interpolate_profile` to the ``z``
        (or ``height``) coordinate of the DSD, or as fields that broadcast
        against the DSD (e.g. :func:`radarx.io.sounding.profile_to_grid`
        output). For a volume, a profile or a DataTree with one node per
        sweep.
    temperature, pressure, specific_humidity, relative_humidity : xarray.DataArray or float, optional
        Fields that override (or replace) those of ``environment``, in the
        same units. ``relative_humidity`` is used only if no specific
        humidity is given.
    fall_speed : "atlas1973" or (float, float, float), optional
        Raindrop fall speed :math:`V = a D^b e^{-f D}` (m s-1, :math:`D` in
        mm) at the density ``rho0``. ``"atlas1973"`` (default) is the fit to
        Atlas et al. (1973) given in :mod:`radarx.retrieve.evaporation`.
    rho0 : float, optional
        Air density (kg m-3) at which ``fall_speed`` holds. Default 1.204.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled
        kernel and falls back to NumPy.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        On the coordinates of the DSD: ``EVAPORATION_RATE`` (kg kg-1 s-1,
        positive for evaporation), ``COOLING_RATE`` (K s-1) and
        ``COOLING_RATE_HOURLY`` (K h-1), positive for cooling,
        ``DBZ_TENDENCY`` (dB h-1, the tendency of the Rayleigh reflectivity
        factor by evaporation) and ``SATURATION_DEFICIT`` (:math:`S - 1`).
        Rates are 0 where ``N0`` is 0 and NaN where the DSD or the air is
        missing. A volume gives a DataTree with one such node per sweep.

    Raises
    ------
    KeyError
        If the DSD or the environment lacks a field, or no sweep of a volume
        has a DSD.
    ValueError
        For unknown options.

    Notes
    -----
    The rates integrate the single-drop law of Rogers and Yau (1989), as
    written by Kumjian and Ryzhkov (2010) with the ventilation coefficient of
    Pruppacher and Klett (1997) and the thermodynamic fits of Rasmussen and
    Heymsfield (1987), over a gamma DSD in closed form, with the saturation
    vapour pressure of Buck (1981) and the fall speed of Atlas et al. (1973)
    (fit and density correction as described in
    :mod:`radarx.retrieve.evaporation`). The :math:`D_v` reference pressure is
    1000 hPa, as in Kumjian and Ryzhkov (2010, Eq. A7); with 1013.25 hPa the
    rates would be 0.3-0.8 % larger.

    References
    ----------
    Kumjian, M. R., and A. V. Ryzhkov, 2010: The impact of evaporation on
    polarimetric characteristics of rain: Theoretical model and practical
    implications. *J. Appl. Meteor. Climatol.*, **49** (6), 1247-1267,
    https://doi.org/10.1175/2010JAMC2243.1

    Buck, A. L., 1981: New equations for computing vapor pressure and
    enhancement factor. *J. Appl. Meteor.*, **20** (12), 1527-1532,
    https://doi.org/10.1175/1520-0450(1981)020<1527:NEFCVP>2.0.CO;2

    Atlas, D., R. C. Srivastava, and R. S. Sekhon, 1973: Doppler radar
    characteristics of precipitation at vertical incidence. *Rev.
    Geophys.*, **11** (1), 1-35, https://doi.org/10.1029/RG011i001p00001

    Rogers, R. R., and M. K. Yau, 1989: *A Short Course in Cloud Physics*. 3rd
    ed., Elsevier (Butterworth-Heinemann), 290 pp., ISBN 978-0-7506-3215-7.

    Pruppacher, H. R., and J. D. Klett, 1997: *Microphysics of Clouds and
    Precipitation*. 2nd rev. and enl. ed., Kluwer Academic Publishers
    (reprinted by Springer, 2010),
    https://doi.org/10.1007/978-0-306-48100-0

    Rasmussen, R. M., and A. J. Heymsfield, 1987: Melting and shedding of
    graupel and hail. Part I: Model physics. *J. Atmos. Sci.*, **44** (19),
    2754-2763,
    https://doi.org/10.1175/1520-0469(1987)044<2754:MASOGA>2.0.CO;2

    Examples
    --------
    >>> params = radarx.retrieve.dsd(qvps)  # doctest: +SKIP
    >>> profile = radarx.io.sounding.era5_profile(33.9, -88.3, "2022-03-30T23:46")  # doctest: +SKIP
    >>> rates = radarx.retrieve.evaporation(params, profile)  # doctest: +SKIP
    """
    use_compiled = _use_compiled(engine)
    fall = _fall(fall_speed)
    rho0 = float(rho0)
    attrs = _attrs(fall, rho0)
    kw = dict(
        temperature=temperature,
        pressure=pressure,
        qv=specific_humidity,
        rh=relative_humidity,
        engine=engine,
        n_threads=n_threads,
    )
    if isinstance(dsd, xr.DataTree):
        names = [
            n for n in dsd.children if all(v in dsd[n].ds.data_vars for v in _DSD_NAMES)
        ]
        if not names:
            raise KeyError("no node of the DataTree holds N0, MU and LAMBDA")
        items, done = [], []
        for name in names:
            ds = dsd[name].to_dataset(inherit=False)
            env = _env_for_node(environment, name)
            if env is None and temperature is None:
                continue
            t, p, q = _environment(ds, env, **kw)
            items.append(_broadcast(ds, t, p, q))
            done.append(name)
        if not items:
            raise KeyError("the environment has no node for any DSD sweep")
        res = _rates(
            [it[0] for it in items], fall, VENTILATION, rho0, n_threads, use_compiled
        )
        nodes = {"/": dsd.root.to_dataset(inherit=False)}
        for name, it, r in zip(done, items, res):
            nodes[name] = _wrap(r, it[1], it[2], it[3], attrs)
        return xr.DataTree.from_dict(nodes)
    if not isinstance(dsd, xr.Dataset):
        raise TypeError("dsd must be an xarray.Dataset or xarray.DataTree")
    t, p, q = _environment(dsd, environment, **kw)
    flat, dims, shape, coords = _broadcast(dsd, t, p, q)
    res = _rates([flat], fall, VENTILATION, rho0, n_threads, use_compiled)[0]
    return _wrap(res, dims, shape, coords, attrs)


def integrate_evaporation(
    dsd,
    environment=None,
    *,
    temperature=None,
    pressure=None,
    specific_humidity=None,
    relative_humidity=None,
    time_dim="time",
    max_step=60.0,
    fall_speed="atlas1973",
    rho0=RHO0,
    n_threads=None,
    engine="auto",
):
    """
    Temperature and humidity of the air under a sequence of DSDs.

    Starting from the environment at the first time, every cell is cooled
    and moistened by the evaporating rain of each DSD until the next one
    (see :mod:`radarx.retrieve.evaporation`), so that the rates respond to
    the cooling and moistening already done.

    Parameters
    ----------
    dsd : xarray.Dataset
        Gamma DSD parameters ``N0``, ``MU`` and ``LAMBDA`` with a ``time_dim``
        dimension, e.g. :func:`radarx.retrieve.dsd` of a
        :func:`radarx.retrieve.qvp_timeseries`. Missing DSDs (NaN) are
        treated as no rain.
    environment : xarray.Dataset, optional
        Initial state, as in :func:`evaporation` (a profile on ``height`` or
        fields without ``time_dim``; with ``time_dim``, its first time is
        used).
    temperature, pressure, specific_humidity, relative_humidity : optional
        Initial fields overriding ``environment``, as in :func:`evaporation`.
        The pressure stays constant.
    time_dim : str, optional
        Time dimension of ``dsd``. Default ``"time"``.
    max_step : float, optional
        Longest integration step (s). Default 60.
    fall_speed, rho0, n_threads, engine : optional
        As in :func:`evaporation`.

    Returns
    -------
    xarray.Dataset
        On the coordinates of the DSD: ``temperature`` (K),
        ``specific_humidity`` (kg kg-1) and ``relative_humidity`` (fraction)
        at each time (before the evaporation of that time's DSD),
        ``TEMPERATURE_CHANGE`` (K, since the first time) and the
        ``EVAPORATION_RATE``, ``COOLING_RATE`` and ``COOLING_RATE_HOURLY``
        in that state.

    Raises
    ------
    KeyError
        If a field is missing.
    ValueError
        If ``dsd`` has no ``time_dim`` or times do not increase.

    Notes
    -----
    The single-drop law (Rogers and Yau 1989, in the form of Kumjian and
    Ryzhkov 2010), the thermodynamic fits and the departures from the cited
    papers are listed in :mod:`radarx.retrieve.evaporation`. The
    sub-stepping (Heun step, saturation limit) is a radarx numerical choice,
    not taken from a paper.

    References
    ----------
    Kumjian, M. R., and A. V. Ryzhkov, 2010: The impact of evaporation on
    polarimetric characteristics of rain: Theoretical model and practical
    implications. *J. Appl. Meteor. Climatol.*, **49** (6), 1247-1267,
    https://doi.org/10.1175/2010JAMC2243.1

    Rogers, R. R., and M. K. Yau, 1989: *A Short Course in Cloud Physics*. 3rd
    ed., Elsevier (Butterworth-Heinemann), 290 pp., ISBN 978-0-7506-3215-7.

    Examples
    --------
    >>> state = radarx.retrieve.integrate_evaporation(params, profile)  # doctest: +SKIP
    """
    use_compiled = _use_compiled(engine)
    fall = _fall(fall_speed)
    rho0 = float(rho0)
    max_step = float(max_step)
    if not max_step > 0:
        raise ValueError("max_step must be positive")
    if not isinstance(dsd, xr.Dataset):
        raise TypeError("dsd must be an xarray.Dataset")
    if time_dim not in dsd.dims:
        raise ValueError(f"the DSD has no {time_dim!r} dimension")
    if environment is not None and time_dim in environment.dims:
        environment = environment.isel({time_dim: 0}, drop=True)
    first = dsd.isel({time_dim: 0})
    t, p, q = _environment(
        first,
        environment,
        temperature=temperature,
        pressure=pressure,
        qv=specific_humidity,
        rh=relative_humidity,
        engine=engine,
        n_threads=n_threads,
    )
    n0, mu, lam = _dsd_vars(dsd)
    n0, mu, lam = xr.broadcast(n0, mu, lam)
    cell_dims = [d for d in n0.dims if d != time_dim]
    t, p, q = (
        x.drop_vars(time_dim, errors="ignore")
        for x in xr.broadcast(t, p, q, first["N0"])[:3]
    )
    t, p, q = (x.transpose(*cell_dims) for x in (t, p, q))
    shape = tuple(dsd.sizes[d] for d in cell_dims)
    dims = (time_dim, *cell_dims)
    nt = dsd.sizes[time_dim]
    stack = [_f64(x.transpose(*dims).values).reshape(nt, -1) for x in (n0, mu, lam)]
    t0, p0, q0 = (_f64(x.values).reshape(-1) for x in (t, p, q))
    tc = dsd[time_dim].values
    if np.issubdtype(np.asarray(tc).dtype, np.datetime64):
        sec = (tc - tc[0]) / np.timedelta64(1, "s")
    else:
        sec = np.asarray(tc, dtype=np.float64)
    dt = _f64(np.diff(sec))
    if np.any(~(dt > 0)):
        raise ValueError(f"{time_dim} must increase")
    if use_compiled:
        res = _evaporation.integrate(
            *stack,
            t0,
            q0,
            p0,
            dt,
            max_step,
            fall[0],
            fall[1],
            fall[2],
            VENTILATION[0],
            VENTILATION[1],
            rho0,
            int(n_threads or 0),
        )
        tt, qq, ee, cc = (np.asarray(r) for r in res)
    else:
        tt, qq, ee, cc = _integrate_numpy(
            *stack, t0, q0, p0, dt, max_step, fall, VENTILATION, rho0
        )
    full = (nt, *shape)
    tt, qq, ee, cc = (x.reshape(full) for x in (tt, qq, ee, cc))
    with np.errstate(all="ignore"):
        pp = np.broadcast_to(p0.reshape(shape), full)
        e = qq * pp / (EPS + (1.0 - EPS) * qq)
        rh = e / _saturation_vapor_pressure(tt)
    coords = {k: v for k, v in n0.coords.items() if set(v.dims) <= set(dims)}
    data_vars = {
        "temperature": (dims, tt, dict(_STATE_ATTRS["temperature"])),
        "specific_humidity": (dims, qq, dict(_STATE_ATTRS["specific_humidity"])),
        "relative_humidity": (dims, rh, dict(_STATE_ATTRS["relative_humidity"])),
        "TEMPERATURE_CHANGE": (
            dims,
            tt - tt[0:1],
            dict(_STATE_ATTRS["TEMPERATURE_CHANGE"]),
        ),
        "EVAPORATION_RATE": (dims, ee, dict(_OUT_ATTRS["EVAPORATION_RATE"])),
        "COOLING_RATE": (dims, cc, dict(_OUT_ATTRS["COOLING_RATE"])),
        "COOLING_RATE_HOURLY": (
            dims,
            cc * 3600.0,
            dict(_OUT_ATTRS["COOLING_RATE_HOURLY"]),
        ),
    }
    attrs = _attrs(fall, rho0)
    attrs["max_step"] = max_step
    attrs["comment"] += (
        "; temperature and humidity integrated in time under each DSD until the "
        "next, at constant pressure, without advection or mixing"
    )
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


# --------------------------------------------------------------------------
# accessors
# --------------------------------------------------------------------------


@accessor_method("dataset", "datatree", name="evaporation")
def _evaporation_accessor(self, environment=None, **kwargs):
    """
    Evaporation and evaporative cooling rates of rain from a gamma DSD.

    The object holds the DSD parameters (``N0``, ``MU``, ``LAMBDA``, e.g. the
    output of ``.radarx.dsd()``). See :func:`radarx.retrieve.evaporation`
    for the parameters.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        Evaporation rate, cooling rates and reflectivity tendency.
    """
    return evaporation(self.xarray_obj, environment, **kwargs)


@accessor_method("dataset", name="integrate_evaporation")
def _integrate_evaporation_accessor(self, environment=None, **kwargs):
    """
    Temperature and humidity of the air under a time series of DSDs.

    See :func:`radarx.retrieve.integrate_evaporation` for the parameters.

    Returns
    -------
    xarray.Dataset
        Temperature, humidity, accumulated cooling and rates per time.
    """
    return integrate_evaporation(self.xarray_obj, environment, **kwargs)
