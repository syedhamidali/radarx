#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Diabatic Lagrangian Analysis
============================

Potential temperature :math:`\\theta`, water vapour and cloud water mixing
ratios :math:`q_v`, :math:`q_c` and virtual buoyancy retrieved from a time
series of 3-D multi-Doppler winds and radar data by the diabatic Lagrangian
analysis (DLA) of Ziegler (2013a, b) [1, 2].

Sources and notation
--------------------
In this module Ziegler (2013a) [1] is Z13a, Ziegler (2013b) [2] Z13b, Ziegler
et al. (2007) [3] Z07 and Lin et al. (1983) [12] LFO83; equation, table, section
and page numbers (pages of the journal articles) are taken from the papers for
Tao et al. (1989) [15], Hsie et al. (1980) [9], Lin et al. (1983) [12],
Gilmore et al. (2004a, b) [7, 8] and Z13a/b. Bolton (1980) [4],
Soong and Ogura (1973) [14], Ferrier (1994) [6], Koenig (1971) [10], Kumjian and
Ryzhkov (2010) [11] and Shapiro (1970) [13] are cited as the papers themselves
are cited by those sources, or for a statement that is not checked
against the paper, and each such place says so. Every number is either
given with its pointer or labelled "radarx choice" or "not checked".

Algorithm
---------
1. A backward trajectory is computed from every grid point at the analysis
   time through the time-dependent winds (:func:`radarx.retrieve.trajectories`:
   predictor-corrector with three iterations, :math:`\\Delta t` = 20 s,
   trilinear/linear interpolation, optional storm-motion advected grid,
   surface parcels from the offset height :math:`H_0` with the parameterised
   surface downdraft, eqs. 2-3) until it reaches the storm environment
   (Z13a sect. 2a-b, pp. 2250-2251).
2. Along each trajectory that reached the environment the ordinary
   differential equations (Z13a eq. 1, p. 2250)

   .. math::

       \\frac{d\\phi}{dt} = M_\\phi + D_\\phi + F_\\psi,\\qquad
       \\phi = (\\theta, q_v, q_c),\\ \\psi = (\\theta, q_v),

   are integrated forward in time, from :math:`\\theta` and :math:`q_v` of the
   environment at the origin of the trajectory (an environmental sounding,
   or a 3-D mesoscale analysis, Z13b sect. 3b) and :math:`q_c = 0`,
   back to the grid point.
3. The end values form the 3-D fields at the analysis time; grid points
   whose trajectory did not reach the environment are hole-filled from their
   neighbours and the fields are smoothed with a horizontal nine-point
   low-pass filter (Z13a sect. 2a, p. 2250: "hole filled from surrounding
   nonmissing grid points" and a "horizontal nine-point elliptic low-pass
   filter"). The weights of the filter and the way holes are filled are not
   given in Z13a; radarx uses the 1-2-1 by 1-2-1 weights (radarx choice, the
   second-order Shapiro 1970 [13] filter in each direction, not checked
   against that paper) and the mean of the valid horizontal neighbours.

The integration runs along the stored backward path (reversed), so every
forward integration ends exactly at its grid point.

Thermodynamics
--------------
Pressure is the base-state pressure :math:`p_B(z)` of the sounding (the
perturbation pressure is neglected, Z13a sect. 2a, p. 2250), and the air
density is :math:`\\rho_a = 10^5 (p_B/10^5)^{1-\\kappa} / (R_d \\theta)` with
:math:`\\kappa = 0.2854` and :math:`R_d = 287.04` J kg\\ :sup:`-1`
K\\ :sup:`-1`. Z13a (p. 2250) prints
:math:`\\rho_a = 10^5 [(p_B/1000)^{0.2854}]^{2.509} / (287.04\\,\\theta)` with
:math:`p_B` in mb, an exponent :math:`0.2854 \\times 2.509 = 0.7161` of
:math:`p_B/p_0` against :math:`1-\\kappa = 0.7146` here (radarx choice; the
densities differ by less than 0.1 % for :math:`p_B \\ge 500` hPa). The
specific heat :math:`c_p = R_d/\\kappa` (1005.8 J kg\\ :sup:`-1` K\\ :sup:`-1`)
follows from these two numbers (radarx choice; LFO83 list 1.005 x 10\\ :sup:`3`,
appendix p. 1089). The saturation vapour pressure over
water is the fit attributed to Bolton (1980) [4] (611.2 Pa, 17.67, 243.5) and
the latent heat of vaporization :math:`L_v = 2.501 \\times 10^6 - 2370\\,(T -
273.15)` J kg\\ :sup:`-1` is also radarx's choice (Z13a only cites Bolton 1980
for the equivalent potential temperature, p. 2257; the coefficients are not
checked against that paper). After each
displacement :math:`\\theta` and :math:`q_v` are conserved while the air is
subsaturated; supersaturation is condensed to cloud water and cloud water in
subsaturated air is evaporated until saturation or :math:`q_c = 0` by an
isobaric saturation adjustment applying the ideas of Soong and Ogura (1973)
[14] (Z13a sect. 2g, p. 2257), in sub-steps of 4 s (Z13a sect. 2g,
:math:`\\Delta t_{small}` = 4 s). The adjustment is a Newton iteration of the
saturation condition (radarx's implementation; Soong and Ogura's
non-iterative form is not checked) and conserves the equivalent potential
temperature of saturated parcels to the accuracy of the sub-stepping.

Ice processes (optional, ``ice=True``)
--------------------------------------
Z13a keeps cloud updrafts saturated with respect to water at all
temperatures (sect. 2g, p. 2257). With ``ice=True`` cloud ice :math:`q_i` is
carried as a fourth Lagrangian variable and the adjustment is the ice-water
saturation adjustment of Tao et al. (1989) [15] (Tao et al. propose it for a
cloud model with the LFO83 microphysics, p. 231; Gilmore et al. 2004a [7],
p. 1901, say their LFO83-like scheme uses a saturation adjustment similar to
it instead of the adjustment of LFO83). It is not part of Z13a.
Equation numbers are those of Tao et al. (1989), pp. 231-233:

- the saturation mixing ratio is the mass-weighted mix (eq. 1, p. 231)
  :math:`q_{vs} = (q_c q_{ws} + q_i q_{is}) / (q_c + q_i)` of Teten's values
  over water and ice (eqs. 3a, 3b, p. 232, with :math:`a` = 17.2693882 and
  21.8745584, offsets 273.16, 35.86 and 7.66 K, :math:`b = 3.8/P` with
  :math:`P` in mb);
- the excess vapour (or deficit) :math:`\\delta q = r_1 / (1 + r_2 A_3)`
  (eqs. 5-7, pp. 232-233, with :math:`A_1`, :math:`A_2`, :math:`A_3` of eqs.
  6c-6e) is split into cloud water and cloud ice in proportions
  :math:`CND = (T - T_{00}) / (T_0 - T_{00})` and :math:`DEP = 1 - CND`
  (eqs. 2b, 2c, p. 232; only liquid above :math:`T_0`, only ice below
  :math:`T_{00}`, ``parameters["t00"]``). The default :math:`T_{00}` = -40
  degC is that of the first run of Tao et al. (Table 1, p. 233); the paper
  calls the choice "simply academic" and says it typically ranges from -30 to
  -40 degC (p. 231). :math:`T_0` is 273.15 K here, as in LFO83. The
  heating is :math:`d\\theta = (L_v\\, dq_c + L_s\\, dq_i)/(c_p \\Pi)` (eq. 4a,
  p. 232), one non-iterative step per sub-step;
- evaporation and sublimation are limited by the available :math:`q_c` and
  :math:`q_i` (p. 232, sect. 2);
- cloud ice melts instantaneously above 0 degC (LFO83 [12], sect. 3f,
  p. 1077) and cloud water freezes instantaneously at or below -40 degC
  (:math:`P_{IMLT}`, :math:`P_{IHOM}`; "colder than -40 degC, homogeneous
  nucleation will occur naturally", LFO83 sect. 3f, p. 1077, and Hsie et al.
  1980 [9], sect. 3b5, p. 956, who also use an isobaric freezing equation and
  melt cloud ice above 0 degC), with the heating of
  eq. (4a) for :math:`dq_i = -dq_c` (:math:`\\pm L_f q / c_p \\Pi`, budget
  term ``dtheta_freezing``), the vapour then adjusting to ice saturation. The
  instantaneous full freezing of all cloud water at -40 degC is how radarx
  reads "occurs naturally" (neither paper gives an equation for it), and the
  heating uses the full latent heat difference rather than an isobaric
  freezing equation (radarx choice).

Not printed in Tao et al. (1989) and chosen here: the weights of eq. (1)
before any condensate exists are :math:`CND` and :math:`DEP` (those of the
condensate the step produces); :math:`L_v`, :math:`L_s` are the constants of
LFO83 (2.5 x 10\\ :sup:`6` and 2.8336 x 10\\ :sup:`6` J kg\\ :sup:`-1`, appendix
pp. 1090-1091). The adjustment conserves total water and, for each isobaric step,
:math:`\\theta - (L_v q_c + L_s q_i)/(c_p \\Pi)`; along an ascent it conserves
the ice-liquid water potential temperature in the form of eq. (4a).
The depositional (Bergeron) growth of cloud ice at the expense of cloud
water, :math:`P_{IDW}` (LFO83, sect. 3f, p. 1077) or :math:`P_{CNWD}` (Hsie et
al. 1980, sect. 3b6, p. 957), is not included: its rate
:math:`(N_n/1000\\rho)\\, a_1 m_n^{a_2}` (Hsie et al.) needs the
temperature-dependent coefficients :math:`a_1(T)`, :math:`a_2(T)` of Koenig
(1971) [10], which neither paper tabulates. Cloud ice enters the in-cloud
test of the graupel sublimation (:math:`\\delta_1` of LFO83 eq. 20), the
surface-flux threshold and the damping like cloud water; collection of cloud
ice by rain or graupel is not represented. ``ice=False`` (default) leaves the
analysis of Z13a unchanged.

In situ initialization (``observations=``)
------------------------------------------
Z13a (sects. 1 and 4, pp. 2249 and 2263) lets a parcel's saturation point be
initialised from a closely neighbouring in situ observation following Z07 [3].
radarx checks each backward trajectory against every observation
(surface station, mobile mesonet, sounding or aircraft datum) taken within the
data window before the analysis time (Z07 p. 2423: observations within
+-12 min of the nominal analysis time in the 22 May case, 15 min on 24 May):
the trajectory position at
the observation time (linear between stored points) is a candidate if it is
within ``radius`` horizontally and ``z_tolerance`` vertically, and its weight
is the first-pass Barnes weight of Z07 (eq. 1, p. 2422)

.. math::

    w = \\exp\\left(-\\frac{r^2}{\\kappa_s} - \\frac{t_i^2}{\\tau_i}
        - \\frac{t_L^2}{\\tau_L}\\right),

with :math:`r` the distance, :math:`t_i = t_o` the time of the observation
relative to the analysis time and :math:`t_L = |t_o|` the integration time
along the trajectory from the observation to the grid point. The initial
:math:`\\theta`, :math:`q_v` are the weighted means of the candidates (each
assumed conserved following the parcel, which is the local conservation of
:math:`\\theta` and :math:`q_v` of Z07, Z13a p. 2249), :math:`q_c = q_i = 0`
followed by the saturation adjustment, at the stored trajectory point nearest
the time of the candidate with the largest weight; the ODEs are integrated
forward from there. Trajectories initialised this way get flag 256 and are
analysed even if they did not reach the environment; ``insitu_weight`` and
``insitu_time`` report the weight sum and the start time. Defaults
(:data:`INSITU_DEFAULTS`) are the Z07 22 May parameters (Table 1, p. 2421:
:math:`\\kappa_s` = 0.076, :math:`\\tau_i` = 364363, :math:`\\tau_L` = 640856; the
table prints no units, km\\ :sup:`2` and s\\ :sup:`2` follow from eq. 1 and
the quoted response wavelengths and periods), second-pass values tuned to
mobile mesonet legs; scale them to the station
spacing of other networks. The cut-off radius (default
:math:`2\\sqrt{\\kappa_s}`) and the vertical tolerance (100 m) are not given
in Z07. Only the first pass of Z07 applies (a single trajectory has no
first-guess field for the correcting pass), and observations after the
analysis time are not on the backward trajectories and are not used.

Microphysics :math:`M_\\phi`
-----------------------------
Z13a (sect. 2g, p. 2257) says that rain collection and graupel accretion of
cloud, rain evaporation and freezing, graupel sublimation and graupel melting
"follow the modified LFO formulation used by G04", i.e. Gilmore et al. (2004a)
[7], and refers to the supplemental material of that paper for the equations;
that supplement (doi:10.1175/MWR2760s1; cited by Gilmore et al. 2004a, pp.
1897, 1899, 1900 and 1901, and 2004b, p. 2611) was not consulted.
radarx implements the rates from the original equations of LFO83
[12] (which the supplement is stated to be "based upon", Gilmore et al.
2004a, p. 1899), with the rain and graupel size distributions of the DLA.
Nothing numerical is taken from Gilmore et al. (2004b) [8]; it is cited for the
discussion of fixed intercepts and densities (below). The equation numbers
below are those of LFO83 (pages 1068-1077):

=====================================  ===================  ==============
process                                LFO83 equation       switch
=====================================  ===================  ==============
rain evaporation :math:`P_{REVP}`      (52), p. 1077        ``rain_evaporation``
collection of cloud by rain            (51), p. 1076        ``cloud_collection``
collection of cloud by graupel         (40), p. 1075        ``cloud_collection``
graupel melting :math:`P_{GMLT}`       (47), p. 1076, (42)  ``graupel_melting``
graupel sublimation :math:`P_{GSUB}`   (46), p. 1076, (31)  ``graupel_sublimation``
rain freezing (Bigg) :math:`P_{GFR}`   (45), p. 1075        ``rain_freezing``
=====================================  ===================  ==============

with the constants of the LFO83 appendix (pp. 1089-1092): the fall-speed
coefficients :math:`a` = 2115 cm\\ :sup:`0.2` s\\ :sup:`-1` (in SI 841.99
m\\ :sup:`0.2` s\\ :sup:`-1`) and :math:`b` = 0.8 of the rain fall speed (eq. 7,
p. 1069), :math:`C_D` = 0.6 (hail drag coefficient, eq. 9, p. 1069),
:math:`g` = 980.5 cm s\\ :sup:`-2`, the Bigg-freezing constants
:math:`A'` = 0.66 K\\ :sup:`-1` and :math:`B'` = 100 m\\ :sup:`-3`
s\\ :sup:`-1` (eq. 45), :math:`L_v` = 2.5 x 10\\ :sup:`6`,
:math:`L_f` = 3.336 x 10\\ :sup:`5` and :math:`L_s` = 2.8336 x 10\\ :sup:`6`
J kg\\ :sup:`-1`, :math:`C_w` = 4.187 x 10\\ :sup:`3` J kg\\ :sup:`-1`
K\\ :sup:`-1`, :math:`R_w` = 461.5 J kg\\ :sup:`-1` K\\ :sup:`-1`, and the
collection efficiencies :math:`E_{RW} = E_{GW} = E_{GR} = 1` (text of eqs. 40-42
and 51, pp. 1075-1076); the mixing-ratio-to-slope relations are eqs. (4)-(6),
p. 1068. The :math:`L_v` of the rain evaporation and graupel melting is the LFO83
constant, whereas the water saturation adjustment uses the temperature-dependent
:math:`L_v` of the thermodynamics section (radarx choice). The thermal
conductivity, vapour diffusivity and kinematic viscosity of air are those of
Kumjian and Ryzhkov (2010, appendix) [11] (the same expressions as
:mod:`radarx.retrieve.evaporation`; not checked against that paper;
LFO83 only lists them as functions of temperature in its
appendix); the saturation vapour pressure over ice is the
Clausius-Clapeyron equation with the constant :math:`L_s` of LFO83 (radarx
choice). The heating is :math:`L_v` (evaporation), :math:`L_s` (sublimation) and
:math:`L_f` (melting, Bigg freezing, freezing of cloud collected by
graupel below 0 degC) times the rate over :math:`c_p \\Pi`
(radarx's thermodynamics; LFO83 eq. 53). Rates are held constant over a time
step (Z13a sect. 2g, p. 2257: "applied as substantial derivatives held
constant during a Lagrangian time step"). Limiting them so that a
step does not evaporate beyond saturation or remove more cloud, rain or
graupel than present is a numerical safeguard of radarx, not part of LFO83 or
Z13a. LFO83 processes that are not implemented are the autoconversion
(eq. 50), all snow and cloud-ice terms, graupel wet growth (eq. 43) and the
collection of rain by graupel below 0 degC (eq. 42 is used only in the
sensible-heat term of eq. 47).

Differences from Gilmore et al. (2004a, b) and Z13a, as far as the
published papers state them: (i) the "Li" scheme of Gilmore et al. 2004a [7]
(p. 1900) does not assume, contrary to LFO83, that the diameter and terminal
fall speed of cloud ice and cloud water are zero in the accretion equations,
and its modified forms were in the supplement, which was not consulted; radarx
uses the LFO83 forms (40) and (51); (ii) Gilmore et al. (2004a, p. 1901, eq. 1-2
and the text below them) used constant intercepts and densities
(:math:`n_{0r} = 8 \\times 10^6`, :math:`n_{0h} = 4 \\times 10^4` m\\ :sup:`-4`,
:math:`\\rho_h` = 900 kg m\\ :sup:`-3`; the same values as LFO83, p. 1068, where
they are 8 x 10\\ :sup:`-2` and 4 x 10\\ :sup:`-4` cm\\ :sup:`-4`); the DLA
takes :math:`n_0` and :math:`\\lambda` of rain and graupel from the precipitation
closure and the graupel density of Z13a (Table 2, p. 2252); the sensitivity of
precipitation to such fixed intercepts and densities is the subject of
Gilmore et al. (2004b) [8], which was used for this discussion only (hail
intercepts observed between 10\\ :sup:`2` and more than 10\\ :sup:`8`
m\\ :sup:`-4`, p. 2612); (iii) eq. (46) is printed in LFO83 (p. 1076) with
:math:`(4 g \\rho_G / 3 C_D \\rho)^{1/4}` but eq. (47) with
:math:`(4 g \\rho_G / 3 C_D)^{1/4}`; dimensional consistency with the fall
speed (9) requires the density in the denominator also in eq. (47), and the
form with :math:`\\rho` is used for both.

Damping and surface flux
------------------------
Mixing is represented by the Lagrangian damping of Z13a
(eqs. 22-26, pp. 2257-2258) toward the base state :math:`\\phi_B` at the parcel,

.. math::

    D_\\phi = -\\frac{c_d V}{L_d e^{\\beta z}} (\\phi - \\phi_B),

with :math:`V = |w|`, :math:`L_d = L_{d0}^{\\pm} + (|w| - W_0)
L_W^{\\pm}` in updrafts (+, :math:`w > W_0`) and downdrafts (-,
:math:`w < -W_0`), and :math:`V = |\\mathbf{V}_h - \\mathbf{V}_B|`,
:math:`L_d = c_d / C_{d0}`, :math:`C_{d0} = (1 - q_p/q_{p0}) C_{min0} +
(q_p/q_{p0}) C_{max0}` in quasi-horizontal flow, applied only where the
precipitation mixing ratio :math:`q_p = q_r + q_g` reaches :math:`q_0`
(surface grid points) or :math:`q_1` (elevated grid points); :math:`z` is
the height above the ground in km (Z13a does not state the unit of :math:`z`
in eqs. 22 and 27 or of :math:`\\beta` and :math:`\\beta_F` in Table 1; km is
radarx's reading, not checked, it is consistent with
:math:`z_{BL}` being given in km). It is integrated exactly over a step,
:math:`\\phi \\leftarrow \\phi_B + (\\phi - \\phi_B) e^{-K \\Delta t}`.
The surface flux is (Z13a eq. 27, p. 2258) :math:`F_\\psi = e^{-\\beta_F z}
(\\mathbf{V}_h \\cdot \\nabla \\psi|_{sfc})` below :math:`z_{BL}` where
:math:`q_c + q_r + q_g \\le q_1`, with the horizontal gradient of the lowest
level of the mesoscale analysis (zero without one) plus optional constant
rates (``surface_flux``, radarx addition). Defaults are those of Z13a (Table 1,
p. 2251), :data:`DLA_DEFAULTS`; Z13a gives :math:`q_{p0}`, :math:`q_0` and :math:`q_1` in g kg\\ :sup:`-1`
(converted to kg kg\\ :sup:`-1` in ``DLA_DEFAULTS``). Z13a states that damping is zero if
:math:`q_p < q_0` (surface grid points) or :math:`q_p < q_1` (elevated grid
points), p. 2258, and sets :math:`F_\\psi = 0` outside
:math:`z \\le z_{BL}` and :math:`q_{hydro} = q_c + q_r + q_g \\le q_1`.

Precipitation closures
----------------------
The rain and graupel mixing ratios and number concentrations (inverse
exponential size distributions, Z13a eqs. 4-6, p. 2252) come from a
pluggable closure, evaluated on the analysis grid at every wind time and
interpolated to the parcels like the winds:

- ``"polarimetric"`` (default, :func:`polarimetric_precipitation`): rain
  below the melting level from the gamma DSD of :func:`radarx.retrieve.dsd`
  (:math:`Z_H`, :math:`Z_{DR}`, optionally :math:`K_{DP}`), graupel where the
  hydrometeor classification has graupel or hail, above the melting level, or
  where the DSD retrieval fails in strong echo (rain-hail mixtures), from the
  reflectivity left after rain with the graupel reflectivity of Ferrier
  (1994) [6] in the form of Z13a (eq. 8, p. 2252: "following Ferrier (1994)
  and substituting the definitions of the distribution moments"; the
  coefficients :math:`a_r` = 0.224 and :math:`C_g` = 7.295 x 10\\ :sup:`19`
  are those of Table 2, p. 2252) and a fixed intercept. This closure is
  radarx's own and not part of Z13a; its thresholds (``graupel_min_dbz`` =
  40 dBZ, ``min_dbz`` = 0 dBZ, the 1 km melting layer, the fixed graupel
  intercept :math:`(n_{0g})_0` of Table 2) are radarx choices;
- ``"ziegler2013"`` (:func:`ziegler2013_precipitation`): the reflectivity-only
  closure of Z13a (eqs. 9-21, pp. 2252-2257, constants of Table 2,
  p. 2252). Its regression profiles
  :math:`Z_{0r}(z^*)`, :math:`Z_{0g}(z^*)`, :math:`S_0(z^*)` were derived by
  Z13a from a simulated supercell (Binger storm, 10-ICE scheme) and are
  output to look-up tables at 100 m that are not tabulated in the paper
  (only :math:`Z_{0g}` = 44.19 dBZ at :math:`z^*` = 5 km is quoted, Fig. 3
  caption, p. 2255). radarx ships
  the same regressions fitted to a simulated squall line
  (:func:`ziegler2013_profiles`), used by default with a warning; pass your
  own with ``profiles=``;
- any callable ``f(winds, base) -> Dataset`` returning ``qr``, ``nr``,
  ``qg``, ``ng`` on the winds grid, or such a Dataset itself.

Rain rate (Z13b [2], eqs. 7-9, p. 2275, coefficients of Table 1, p. 2270):
:math:`R = 3.6 \\times 10^6 \\rho_{sfc} q_r \\bar V_r / \\rho_w`
mm h\\ :sup:`-1` (eq. 9) with
:math:`\\bar V_r = (1.225/\\rho_{sfc})^{1/2} a_r [1 - (1 + f_r D_r)^{-4}]`
(eq. 8, the integral over the rain size distribution of the drop fall speed
:math:`a_r [1 - \\exp(-f_r D_r)]` of eq. 7), :math:`a_r` = 10 m s\\ :sup:`-1`,
:math:`f_r` = 516.575 m\\ :sup:`-1`, :math:`D_r = 1/\\lambda_r`. radarx deviates
from Z13b in using the local air density :math:`\\rho` instead of the surface
density :math:`\\rho_{sfc}` in eqs. (8) and (9) (the output is a rain rate at
every level).

Sensitivity tests
-----------------
``processes`` switches every term (and the surface downdraft) off; the
acronyms of Z13a (Table 3, p. 2259) are accepted: ``"CNTL"``, ``"GMLT"``
(no graupel melting), ``"NOCOL"`` (no cloud collection), ``"NOLD"`` (no
damping), ``"RVAP"`` (no rain evaporation), ``"WSFC"`` (no surface
downdraft); ``"NGSC"`` is ``graupel_scale`` of the closures (:math:`a_N` = 2 in Table 3).

Applying DLA to observed QLCS cases
-----------------------------------
Z13a, b analysed supercells with winds blended to a sounding
outside the radar coverage (Z13b sect. 2b). This section and the evidence
below describe radarx's own experience and tests; none of it is from the cited
papers. Squall lines with a long-lived trailing cold pool
and multi-Doppler winds of limited coverage need some care:

- *Winds outside the coverage and untrusted winds.* Fill them (e.g. with the
  background wind of :func:`radarx.retrieve.multi_doppler` and :math:`w = 0`)
  so the parcels can continue, and pass the coverage as ``valid`` (e.g.
  ``dd_valid``). ``valid_fraction`` tells how much of each trajectory used
  analysed winds, and ``min_valid_fraction`` masks (and hole-fills) the
  others. Columns in which the vertical integration of mass continuity
  diverges (|w| of tens of m s\\ :sup:`-1` aloft above unobserved levels) also
  carry spurious broad descent of several m s\\ :sup:`-1` at low levels, which
  warms parcels adiabatically by several K: treat them as not valid.
- *Unobserved lowest level.* If the ground level is below the lowest beams,
  extrapolate the winds and radar fields to it from the level above (and their
  validity), so that surface parcels see the precipitation that cools them.
- *Environment.* A single sounding cannot represent a heterogeneous, evolving
  inflow (e.g. evening cooling). Pass a time-dependent ``mesoscale`` analysis
  (as in Z13b, sect. 3b) built from pre-storm soundings and surface
  stations, and keep the stations used for it apart from those used for
  validation.
- *Termination.* Ziegler's test (ii), :math:`w < 0.5` m s\\ :sup:`-1` for five
  steps after 76 steps, ends surface trajectories in a stratiform cold pool
  after about 26 min, still inside the outflow. ``termination="precipitation"``
  follows them until they are outside echo and either ahead of the gust front
  (``environment_mask``) or above the cold-pool depth (``cold_pool_depth``);
  set ``min_steps`` to 0 with it, and a maximum duration with ``max_steps``.
  Longer trajectories are more sensitive to errors of :math:`w`.
- *Time morphing.* The air of a mature squall-line cold pool is often older
  than the wind series. Without more winds the "precipitation" termination
  then leaves most surface trajectories inside the cold pool when the data run
  out (flag 8 or 16, not environmental). ``extend_before`` reuses the first
  analysis, moved with ``storm_motion``, before the series (Ziegler 2013b,
  sect. 2c, p. 2268, and Fig. 1; a storm steady in its own frame).
- *Lateral boundaries.* By default every exit through a lateral boundary
  counts as environment (Ziegler 2013a, sect. 2a). Where an edge of the
  analysis cuts through the storm (e.g. the rear of a trailing cold pool),
  restrict it with ``boundary``: a list of the sides that lie in the
  environment (e.g. ``["east", "north", "south"]`` for a line moving east),
  ``"environment_mask"`` or ``"no_echo"``. The other exits get flag 128, are
  not environmental and are hole-filled.
- *Melting layer.* The polarimetric closure blends rain and graupel through a
  melting layer of finite depth (default 1 km about the 0 degC level, or the
  wet-snow band of the HID), so latent cooling is continuous in height.
- *Storm motion.* Use the motion with which the analyses were advected to
  their times (``storm_motion``). ``storm_motion="estimate"`` takes the median
  of :func:`radarx.retrieve.estimate_motion` between consecutive analyses;
  it tracks echoes, and the cells of a squall line can move along and across
  the line differently from the line itself, so check it against the motion
  of the gust front. The motion must be in the frame of the analysis
  coordinates: for a model domain that translates with the storm it is the
  domain motion, not the echo motion within the domain.
- *Ice processes.* By default the DLA follows Ziegler (2013a) and has no
  deposition or freezing of cloud water in updrafts: cloudy updraft air above
  the melting level stays saturated with respect to water and misses the
  latent heat of freezing and of deposition onto ice. ``ice=True`` adds the
  ice-water saturation adjustment of Tao et al. (1989) [15] (see "Ice processes"
  above); the Bergeron conversion of supercooled cloud water is still missing.

Recommended squall-line configuration (Ziegler's defaults are unchanged)::

    diabatic_lagrangian(
        winds, sounding,
        termination="precipitation", parameters={"min_steps": 0},
        environment_mask="ahead",          # True ahead of the gust front
        storm_motion=(cx, cy), extend_before=7200.0,
        boundary=["east", "north", "south"],  # edges in the environment
    )

*Evidence from an observing-system simulation experiment.* The DLA was given
the perfect winds and reflectivity (every 2 min for 100 min, 1-km grid) of a
CM1 (release 21.1; Bryan and Fritsch 2002 [5]) squall line with GSR-LFO
three-ice microphysics, the inflow sounding and the Ziegler closure with
profiles fitted to the same simulation (:func:`ziegler2013_profiles`), and
was scored against the model at the analysis time (surface cold pool: 0-60 km
behind the gust front where the truth :math:`\\Delta\\theta_v < -1` K, mean
over the points whose trajectory reached the environment):

==============================================  =====  =========  =======
configuration                                   valid  truth (K)  DLA (K)
==============================================  =====  =========  =======
``termination="ziegler2013"``                   100 %  -5.38      -1.17
``"precipitation"``                             1 %    -5.30      -0.00
``"precipitation"``, ``extend_before=7200``     61 %   -5.47      -4.47
==============================================  =====  =========  =======

Ziegler's test ends all surface cold-pool trajectories after 77 steps
(25.7 min), about 50 m above the ground inside the cold pool, so they start
with inflow values; the "precipitation" test alone leaves 98 % of them in the
cold pool when the data run out; with two hours of time morphing they reach
the inflow and the DLA recovers about 80 % of the cold pool (in the whole
hole-filled surface cold pool -4.28 K against -5.38 K; storm-volume
:math:`\\theta_v` RMSE 2.27 K, bias -0.80 K). With radar-like winds (3-min
cadence, 1-km smoothing, 1.5 m s\\ :sup:`-1` noise, no data below 250 m,
:math:`w` from mass continuity) the same configuration gives -4.53 K against
-5.55 K. In that domain 8 % of the trajectories leave through its rear (west)
edge, inside the cold pool; ``boundary=["east", "north", "south"]`` flags them
(128) and the environmental points are then exactly those of a manual
exclusion of rear exits, with the same scores (-4.47 K, RMSE 2.27 K); the
hole-filled fields change by less than 0.05 K. ``"no_echo"`` flags only the
2.8 % of exits in echo. Above the line there is a spurious cold anomaly:
line-averaged :math:`\\Delta\\theta_v` at 5-7 km, 0-30 km behind the gust
front, is -2.2 K in the DLA (minimum -4.7 K) against -0.1 K (minimum
-0.7 K) in the truth, independent of the termination, boundary and
time-morphing options. ``ice=True`` reduces it to -1.9 K (minimum -4.2 K;
deposition and fusion add 0.36 K of latent heating there) and the
storm-volume :math:`\\theta_v` RMSE from 2.27 to 2.16 K (bias -0.79 to
-0.70 K), leaving the surface unchanged (within 0.01 K); the rest of the
anomaly, which starts at the melting level, does not come from cloud ice.

With in situ observations: 200 "stations" at random surface grid points of
the cold pool report the truth :math:`\\theta`, :math:`q_v` every 2 min for
the 12 min before the analysis time (``observation_options={"radius": 1500,
"kappa_s": 5e5}``), and 100 other cold-pool points at least 3 km from every
station are held out for scoring. 1.7 % of all trajectories start from a
station, and the analysed (environmental or station-initialised) part of the
surface cold pool grows from 61 % to 87 %; at the hold-out
points the surface :math:`\\Delta\\theta_v` RMSE drops from 1.89 to 1.36 K
and the bias from +0.98 to +0.19 K (truth mean -5.18 K), and the hole-filled
surface cold pool mean is -5.18 K against -5.38 K in the truth (-4.28 K
without stations). With 400 stations the hold-out RMSE is 1.17 K (bias
+0.10 K); with the Z07 mobile-mesonet defaults (:math:`\\kappa_s` = 0.076
km\\ :sup:`2`, radius 0.55 km) 200 stations give 1.52 K (bias +0.56 K).

Computation
-----------
Trajectories and the forward integration of all grid points run in one call
of the compiled kernel (``radarx.retrieve._lagrangian``, C++,
multithreaded over grid points, GIL released, no storage of the
trajectories); an identical NumPy implementation is used when the kernel is
not built (``engine="numpy"``) and serves as its test oracle.

References
----------
[1] Ziegler, C. L., 2013a: A diabatic Lagrangian technique for the analysis of
convective storms. Part I: Description and validation via an observing system
simulation experiment. *J. Atmos. Oceanic Technol.*, **30** (10), 2248-2265,
https://doi.org/10.1175/JTECH-D-12-00194.1

[2] Ziegler, C. L., 2013b: A diabatic Lagrangian technique for the analysis of
convective storms. Part II: Application to a radar-observed storm. *J. Atmos.
Oceanic Technol.*, **30** (10), 2266-2280,
https://doi.org/10.1175/JTECH-D-13-00036.1

[3] Ziegler, C. L., M. S. Buban, and E. N. Rasmussen, 2007: A Lagrangian objective
analysis technique for assimilating in situ observations with
multiple-radar-derived airflow. *Mon. Wea. Rev.*, **135** (7), 2417-2442,
https://doi.org/10.1175/MWR3396.1

[4] Bolton, D., 1980: The computation of equivalent potential temperature.
*Mon. Wea. Rev.*, **108** (7), 1046-1053,
https://doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2

[5] Bryan, G. H., and J. M. Fritsch, 2002: A benchmark simulation for moist
nonhydrostatic numerical models. *Mon. Wea. Rev.*, **130** (12), 2917-2928,
https://doi.org/10.1175/1520-0493(2002)130<2917:ABSFMN>2.0.CO;2

[6] Ferrier, B. S., 1994: A double-moment multiple-phase four-class bulk ice
scheme. Part I: Description. *J. Atmos. Sci.*, **51** (2), 249-280,
https://doi.org/10.1175/1520-0469(1994)051<0249:ADMMPF>2.0.CO;2

[7] Gilmore, M. S., J. M. Straka, and E. N. Rasmussen, 2004a: Precipitation and
evolution sensitivity in simulated deep convective storms: Comparisons
between liquid-only and simple ice and liquid phase microphysics. *Mon. Wea.
Rev.*, **132** (8), 1897-1916,
https://doi.org/10.1175/1520-0493(2004)132<1897:PAESIS>2.0.CO;2

[8] Gilmore, M. S., J. M. Straka, and E. N. Rasmussen, 2004b: Precipitation
uncertainty due to variations in precipitation particle parameters within a
simple microphysics scheme. *Mon. Wea. Rev.*, **132** (11), 2610-2627,
https://doi.org/10.1175/MWR2810.1

[9] Hsie, E.-Y., R. D. Farley, and H. D. Orville, 1980: Numerical simulation of
ice-phase convective cloud seeding. *J. Appl. Meteor.*, **19** (8), 950-977,
https://doi.org/10.1175/1520-0450(1980)019<0950:NSOIPC>2.0.CO;2

[10] Koenig, L. R., 1971: Numerical modeling of ice deposition. *J. Atmos.
Sci.*, **28** (2), 226-237,
https://doi.org/10.1175/1520-0469(1971)028<0226:NMOID>2.0.CO;2

[11] Kumjian, M. R., and A. V. Ryzhkov, 2010: The impact of evaporation on
polarimetric characteristics of rain: Theoretical model and practical
implications. *J. Appl. Meteor. Climatol.*, **49** (6), 1247-1267,
https://doi.org/10.1175/2010JAMC2243.1

[12] Lin, Y.-L., R. D. Farley, and H. D. Orville, 1983: Bulk parameterization of
the snow field in a cloud model. *J. Climate Appl. Meteor.*, **22** (6),
1065-1092, https://doi.org/10.1175/1520-0450(1983)022<1065:BPOTSF>2.0.CO;2

[13] Shapiro, R., 1970: Smoothing, filtering, and boundary effects. *Rev.
Geophys.*, **8** (2), 359-387, https://doi.org/10.1029/RG008i002p00359

[14] Soong, S.-T., and Y. Ogura, 1973: A comparison between axisymmetric and
slab-symmetric cumulus cloud models. *J. Atmos. Sci.*, **30** (5), 879-893,
https://doi.org/10.1175/1520-0469(1973)030<0879:ACBAAS>2.0.CO;2

[15] Tao, W.-K., J. Simpson, and M. McCumber, 1989: An ice-water saturation
adjustment. *Mon. Wea. Rev.*, **117** (1), 231-235,
https://doi.org/10.1175/1520-0493(1989)117<0231:AIWSA>2.0.CO;2

.. autosummary::
   :nosignatures:
   :toctree: generated/

   diabatic_lagrangian
   polarimetric_precipitation
   ziegler2013_precipitation
   ziegler2013_profiles
   microphysical_rates
"""

__all__ = [
    "diabatic_lagrangian",
    "polarimetric_precipitation",
    "ziegler2013_precipitation",
    "ziegler2013_profiles",
    "microphysical_rates",
]

import math
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from .._provenance import provenance
from . import _lagrangian_numpy as _nk
from . import lagrangian as _traj

#: Damping, surface-flux and graupel-density parameters (Ziegler 2013a, Table 1,
#: p. 2251, and Table 2, p. 2252; the table values in g kg-1 or km are converted
#: to SI here), the trajectory parameters of
#: :data:`radarx.retrieve.lagrangian.TRAJECTORY_DEFAULTS` and the condensation
#: sub-step (Ziegler 2013a, sect. 2g, p. 2257: 4 s). ``t00`` is Tao et al.
#: (1989), Table 1, p. 233 (run 1: -40 degC), used only with ``ice=True``.
DLA_DEFAULTS = {
    **_traj.TRAJECTORY_DEFAULTS,
    "dt_small": 4.0,  # s, Delta t_small of sect. 2g, p. 2257
    "cd": 0.2,  # damping coefficient, eq. (22), Table 1
    "b": 1.0
    / 3.0,  # height-scale coefficient, eq. (22), Table 1 (unit of z not given; km assumed)
    "w0": 0.1,  # W0, m s-1, Table 1
    "ld1": 5000.0,  # L_d0 (updraft), m, Table 1, eq. (23)
    "ld2": 300.0,  # L_d0 (downdraft), m, Table 1, eq. (24)
    "lw1": 2000.0,  # dL/dw (updraft), s, Table 1
    "lw2": 100.0,  # dL/dw (downdraft), s, Table 1
    "cmin": 7.0e-5,  # C_min0, m-1, Table 1, eq. (26)
    "cmax": 2.0e-4,  # C_max0, m-1, Table 1, eq. (26)
    "qp0": 1.0e-3,  # q_p0, kg kg-1 (1 g kg-1), Table 1, eq. (26)
    "q0": 2.0e-3,  # surface threshold hydrometeor q, kg kg-1 (2 g kg-1), Table 1
    "q1": 1.0e-4,  # threshold hydrometeor q, kg kg-1 (0.1 g kg-1), Table 1
    "z_bl": 1000.0,  # boundary-layer height, m (1.0 km), Table 1, eq. (27)
    "b_f": 3.0,  # height-scale coefficient, eq. (27), Table 1 (unit of z not given; km assumed)
    "rho_g_sfc": 690.0,  # graupel density at the surface, kg m-3 (Table 2, p. 2252)
    "rho_g_5km": 630.0,  # graupel density at 5 km AGL, kg m-3 (Table 2, p. 2252)
    "t00": 233.15,  # T00 of the ice-water adjustment (ice=True), K: -40 degC, Tao et al. (1989) Table 1, run 1
}

#: Options of the in situ initialization (``observations=``): data window
#: (s, observations at most this long before the analysis time; Ziegler et al.
#: 2007, p. 2423: observations within +-12 min of the analysis time in the
#: 22 May case), cut-off radius (m; None: 2 sqrt(kappa_s)), vertical
#: tolerance (m) and the Barnes parameters of Ziegler et al. (2007, eq. 1,
#: p. 2422, and Table 1, p. 2421, 22 May case, second-pass values): kappa_s
#: (0.076, 7.6e4 m2 if the table's unit is km2), tau_i, tau_L (s2; the table
#: prints no units). The radius and the tolerance are not given by Ziegler et
#: al. (2007) and are radarx choices.
INSITU_DEFAULTS = {
    "window": 720.0,
    "radius": None,
    "z_tolerance": 100.0,
    "kappa_s": 7.6e4,
    "tau_i": 364363.0,
    "tau_l": 640856.0,
}

#: Physical processes of the DLA (all on by default).
PROCESSES = (
    "condensation",
    "rain_evaporation",
    "cloud_collection",
    "graupel_melting",
    "graupel_sublimation",
    "rain_freezing",
    "damping",
    "surface_flux",
    "surface_downdraft",
)

#: Sensitivity tests of Ziegler (2013a, Table 3) as process switches.
SENSITIVITY_TESTS = {
    "CNTL": {},
    "GMLT": {"graupel_melting": False},
    "NOCOL": {"cloud_collection": False},
    "NOLD": {"damping": False},
    "RVAP": {"rain_evaporation": False},
    "WSFC": {"surface_downdraft": False},
}

_BITS = {
    "condensation": _nk.COND,
    "rain_evaporation": _nk.REVP,
    "cloud_collection": _nk.RACW | _nk.GACW,
    "graupel_melting": _nk.GMLT,
    "graupel_sublimation": _nk.GSUB,
    "rain_freezing": _nk.GFR,
    "damping": _nk.DAMP,
    "surface_flux": _nk.FLUX,
}

# Ziegler (2013a) Table 2 (p. 2252) and (2013b) Table 1 (p. 2270) constants of
# the closures (the two tables give the same values except H_melt, H_frz and
# the Z13b rain-rate coefficients).
C_R = 1.0e18  # mm6 m-3 per m6 m-3, eq. (7)
A_R_DIEL = 0.224  # a_r of eq. (8) (dielectric factor ratio; Table 2 gives no unit)
C_G = 7.295e19  # eq. (8), Table 2
RHO_W = 1000.0  # density of water, kg m-3, Table 2
ZIEGLER_CONSTANTS = {
    "eps_r": 0.5,  # g kg-1, eq. (9), Table 2 (unit g kg-1 inferred from q* in g kg-1)
    "eps_g": 1.0,  # g kg-1, eq. (11), Table 2
    "n0g0": 3.355e5,  # (n0g)0, m-4, eq. (10), Table 2
    "n0r": 8.0e5,  # m-4, Table 2
    "h_melt_ref": 3900.0,  # m, H_melt = 3.9 km AGL of the regression storm, Table 2
    "a_frz": 0.1,  # a_frz of C_frz (text below eq. 16), Table 2
    "w_min": 5.0,  # m s-1, Table 2, eqs. (12)-(15)
    "w_max": 20.0,  # m s-1, Table 2, eqs. (13)-(16)
}

_BUDGET = (
    ("condensation", "condensation, deposition, evaporation and sublimation of cloud"),
    ("rain_evaporation", "rain evaporation"),
    ("graupel_melting", "graupel melting"),
    ("graupel_sublimation", "graupel sublimation"),
    (
        "freezing",
        "rain freezing, riming of cloud by graupel and freezing or melting of cloud ice",
    ),
    ("damping", "Lagrangian damping"),
    ("surface_flux", "surface flux"),
)


# ---------------------------------------------------------------------------
# Base state
# ---------------------------------------------------------------------------


def _es_bolton(t):
    # Bolton (1980) fit (611.2 Pa, 17.67, 243.5 K); radarx choice, the
    # coefficients are not checked against the paper.
    tc = t - 273.15
    return 611.2 * np.exp(17.67 * tc / (tc + 243.5))


def _crossing(z, t, level):
    """Highest height where t falls through level (NaN if none)."""
    d = t - level
    idx = np.flatnonzero((d[:-1] >= 0) & (d[1:] < 0))
    if idx.size == 0:
        return np.nan
    i = idx[-1]
    return float(z[i] + (z[i + 1] - z[i]) * d[i] / (d[i] - d[i + 1]))


def _base_state(sounding, z, ground, dz=10.0):
    """Base-state table (uniform in height) and profiles on the grid levels."""
    if not isinstance(sounding, xr.Dataset) or "height" not in sounding.dims:
        raise ValueError(
            "sounding must be an xarray.Dataset on 'height' (radarx.io.sounding format)"
        )
    for name in ("pressure", "temperature"):
        if name not in sounding:
            raise ValueError(f"the sounding needs {name!r}")
    s = sounding.sortby("height")
    h = np.asarray(s["height"].values, float)
    p = np.asarray(s["pressure"].values, float)
    t = np.asarray(s["temperature"].values, float)
    if "specific_humidity" in s:
        q = np.asarray(s["specific_humidity"].values, float)
        r = q / (1.0 - q)
    elif "dewpoint" in s:
        e = _es_bolton(np.asarray(s["dewpoint"].values, float))
        r = _nk.EPS * e / (p - e)
    else:
        raise ValueError("the sounding needs 'specific_humidity' or 'dewpoint'")
    u = np.asarray(s["u"].values, float) if "u" in s else np.zeros_like(h)
    v = np.asarray(s["v"].values, float) if "v" in s else np.zeros_like(h)

    zb = np.arange(min(ground, z[0]), z[-1] + dz, dz)

    def interp(a, log=False):
        ok = np.isfinite(a) & np.isfinite(h)
        if ok.sum() < 2:
            if a is u or a is v:
                return np.zeros_like(zb)
            raise ValueError("the sounding has fewer than two valid levels")
        vals = np.log(a[ok]) if log else a[ok]
        out = np.interp(zb, h[ok], vals)
        return np.exp(out) if log else out

    pb = interp(p, log=True)
    tb = interp(t)
    rb = np.maximum(interp(r), 0.0)
    ub, vb = interp(u), interp(v)
    thb = tb * (_nk.P0 / pb) ** _nk.KAPPA
    table = np.column_stack([pb, thb, rb, ub, vb])
    zagl = zb - ground
    h_melt = _crossing(zagl, tb, 273.15)
    h_15 = _crossing(zagl, tb, 258.15)

    def at(a):
        return np.interp(z, zb, a)

    thv = thb * (1.0 + rb / _nk.EPS) / (1.0 + rb)
    rho = _nk.air_density(thb, pb)
    base = xr.Dataset(
        {
            "pressure": ("z", at(pb), {"standard_name": "air_pressure", "units": "Pa"}),
            "temperature": (
                "z",
                at(tb),
                {"standard_name": "air_temperature", "units": "K"},
            ),
            "theta": (
                "z",
                at(thb),
                {"standard_name": "air_potential_temperature", "units": "K"},
            ),
            "qv": (
                "z",
                at(rb),
                {"standard_name": "humidity_mixing_ratio", "units": "kg kg-1"},
            ),
            "theta_v": (
                "z",
                at(thv),
                {"long_name": "virtual potential temperature", "units": "K"},
            ),
            "rho": ("z", at(rho), {"standard_name": "air_density", "units": "kg m-3"}),
            "u": ("z", at(ub), {"units": "m s-1"}),
            "v": ("z", at(vb), {"units": "m s-1"}),
        },
        coords={"z": z},
        attrs={
            "ground_height": float(ground),
            "melting_level": h_melt,  # m above the ground
            "minus15_level": h_15,
            "rho0": float(
                _nk.air_density(
                    thb[np.argmin(abs(zb - ground))], pb[np.argmin(abs(zb - ground))]
                )
            ),
        },
    )
    return table, float(zb[0]), float(dz), base


# ---------------------------------------------------------------------------
# Precipitation closures
# ---------------------------------------------------------------------------


def _graupel_density(zagl, rho_sfc, rho_5km):
    # linear between the surface and 5 km AGL (Ziegler 2013a, p. 2254; values
    # of Table 2); constant above 5 km (radarx choice)
    return rho_sfc + (rho_5km - rho_sfc) * np.clip(zagl / 5000.0, 0.0, 1.0)


def _scale_graupel(qg, ng, rho, rhog, scale):
    """Ziegler (2013a, eqs. 18-21, pp. 2256-2257): N_g scaled by a_N at constant
    graupel reflectivity."""
    if np.all(np.asarray(scale) == 1.0):
        return qg, ng
    with np.errstate(all="ignore"):
        lam = np.cbrt(np.pi * rhog * ng / (rho * qg))
        zg = A_R_DIEL * C_G * np.pi * rho * rhog * qg / (RHO_W**2 * lam**3)  # (19)
        ng2 = scale * ng  # (18)
        qg2 = np.sqrt(zg * ng2 / (A_R_DIEL * C_G * (rho / RHO_W) ** 2))  # (20)
    ok = (qg > 0) & (ng > 0)
    return np.where(ok, qg2, qg), np.where(ok, ng2, ng)


def _dbz_name(ds, dbzh):
    if dbzh not in (None, "auto"):
        if dbzh not in ds:
            raise ValueError(f"reflectivity {dbzh!r} not found")
        return dbzh
    for name in _traj._REFLECTIVITY:
        if name in ds:
            return name
    raise ValueError("no reflectivity variable found; pass dbzh=")


def _like(ds, name):
    return ds[name].transpose(
        *[d for d in ("time", "z", "y", "x") if d in ds[name].dims]
    )


def _closure_output(ref, qr, nr, qg, ng, attrs):
    dims = ref.dims
    meta = {
        "qr": (
            {
                "standard_name": "mass_fraction_of_rain_in_air",
                "long_name": "rain mixing ratio",
                "units": "kg kg-1",
            }
        ),
        "nr": ({"long_name": "rain number concentration", "units": "m-3"}),
        "qg": ({"long_name": "graupel mixing ratio", "units": "kg kg-1"}),
        "ng": ({"long_name": "graupel number concentration", "units": "m-3"}),
    }
    data = {"qr": qr, "nr": nr, "qg": qg, "ng": ng}
    return xr.Dataset(
        {
            k: (dims, np.where(np.isfinite(a) & (a > 0), a, 0.0), meta[k])
            for k, a in data.items()
        },
        coords={
            c: ref.coords[c] for c in ref.coords if set(ref.coords[c].dims) <= set(dims)
        },
        attrs=attrs,
    )


@provenance(
    "Rain and graupel from polarimetric data, radarx's own closure after Ziegler (2013a)"
)
def polarimetric_precipitation(
    radar,
    base,
    *,
    dbzh="auto",
    zdr=None,
    kdp=None,
    hid="auto",
    band="S",
    mu_lambda="cao2008",
    graupel_intercept=3.355e5,
    rain_intercept=8.0e5,
    graupel_min_dbz=40.0,
    min_dbz=0.0,
    graupel_scale=1.0,
    graupel_density=(690.0, 630.0),
    melting_layer=None,
    melting_depth=1000.0,
    engine="auto",
    n_threads=None,
):
    """
    Rain and graupel from polarimetric radar data (default DLA closure).

    This closure is radarx's own; it is not part of Ziegler (2013a) [1], which
    uses reflectivity only (:func:`ziegler2013_precipitation`). It uses the
    inverse exponential size distributions of Z13a (eqs. 4-8, p. 2252) with
    the coefficients of Z13a Table 2, and the graupel reflectivity of Ferrier
    (1994) [2] as given in Z13a eq. (8).

    Below the melting level rain comes from the constrained-gamma DSD of
    :func:`radarx.retrieve.dsd` (:math:`Z_H`, :math:`Z_{DR}`, optional
    :math:`K_{DP}`): :math:`q_r` from the liquid water content and
    :math:`N_r = N_0 \\Gamma(\\mu+1)/\\Lambda^{\\mu+1}`. Graupel is diagnosed
    where the hydrometeor classification (``hid``) has a graupel or hail
    class, everywhere above the melting level, and below it where the DSD
    retrieval fails with :math:`Z_H \\ge` ``graupel_min_dbz`` (rain-hail
    mixtures). The graupel reflectivity is what is left after rain,
    :math:`Z_g = Z_H - Z_r(\\mathrm{DSD})`; with a fixed intercept
    :math:`n_{0g}` it gives :math:`\\lambda_g` from Z13a [1] eq. (8) (the
    graupel reflectivity of Ferrier 1994 [2]), then :math:`q_g` and
    :math:`N_g` from eqs. (6) and (5) (p. 2252). Rain cells below the melting
    level without a DSD (e.g. no :math:`Z_{DR}`) use eq. (7) (p. 2252) with
    ``rain_intercept``. The gamma-DSD rain uses :math:`Z_r = N_0
    \\Gamma(\\mu+7)/\\Lambda^{\\mu+7}` and :math:`N_r = N_0
    \\Gamma(\\mu+1)/\\Lambda^{\\mu+1}` (moments of the gamma distribution;
    not from Z13a, which assumes an exponential rain distribution).

    Rain and graupel are blended through a melting layer of finite depth
    (by default 1 km centred on the environmental 0 degC level): with the ice
    fraction :math:`f` rising linearly from 0 at its bottom to 1 at its top,
    the rain contents are scaled by :math:`1 - f` and the graupel
    reflectivity is :math:`f Z_H + (1 - f) Z_g^{below}` (the linear ramp and the 1 km depth are
    radarx choices; Z13a has no melting-layer blending), so the diagnosed
    contents, and the melting and evaporation they drive, are continuous in
    height.

    Parameters
    ----------
    radar : xarray.Dataset
        Reflectivity (dBZ), differential reflectivity (dB) and optionally
        :math:`K_{DP}` and an HID on the analysis grid ``(time, z, y, x)``.
    base : xarray.Dataset
        Base state on ``z`` with ``rho`` and the attributes
        ``ground_height`` and ``melting_level`` (m above the ground), as
        passed by :func:`diabatic_lagrangian`.
    dbzh, zdr : str, optional
        Variable names (default: found automatically).
    kdp : str, optional
        :math:`K_{DP}` variable (degrees/km). Default: not used.
    hid : str or None, optional
        Hydrometeor classification (``flag_meanings`` attribute as written by
        :func:`radarx.retrieve.hid`); classes whose names contain "graupel" or
        "hail" are graupel. ``"auto"``: ``HID`` if present.
    band : {"S", "C", "X"}, optional
        Radar band of the DSD scattering tables. Default ``"S"``.
    mu_lambda : optional
        :math:`\\mu`-:math:`\\Lambda` relation of :func:`radarx.retrieve.dsd`.
    graupel_intercept : float, optional
        :math:`n_{0g}` in m-4. Default :math:`3.355 \\times 10^5`
        (:math:`(n_{0g})_0` of Z13a Table 2, p. 2252, the intercept of the
        regression of eq. 10 at :math:`Z_H` = 0; using it as a constant
        intercept is a radarx choice).
    rain_intercept : float, optional
        :math:`n_{0r}` (m-4) of the fallback rain. Default :math:`8 \\times
        10^5` (Ziegler 2013a, Table 2).
    graupel_min_dbz : float, optional
        Minimum :math:`Z_H` for graupel where the DSD fails. Default 40
        (radarx choice, not from the cited papers).
    min_dbz : float, optional
        No precipitation below this reflectivity. Default 0 (radarx choice).
    graupel_scale : float, optional
        Graupel concentration scale :math:`a_N` (eqs. 18-21). Default 1.
    graupel_density : (float, float), optional
        Graupel density at the ground and at 5 km above it (kg m-3), linear in
        between. Default (690, 630) (Z13a Table 2, p. 2252; linear variation,
        p. 2254).
    melting_layer : (float, float), optional
        Bottom and top of the melting layer in m above the ground (e.g. from
        the wet-snow band of the HID or :func:`radarx.retrieve.melting_layer`).
        Default: ``melting_depth`` centred on the melting level of ``base``.
    melting_depth : float, optional
        Depth (m) of the default melting layer. Default 1000 (radarx choice); 0
        switches abruptly at the melting level.
    engine, n_threads : optional
        Passed to :func:`radarx.retrieve.dsd`.

    Returns
    -------
    xarray.Dataset
        ``qr``, ``qg`` (kg kg-1), ``nr``, ``ng`` (m-3) on the radar grid.

    References
    ----------
    [1] Ziegler, C. L., 2013a: A diabatic Lagrangian technique for the analysis
        of convective storms. Part I: Description and validation via an
        observing system simulation experiment. *J. Atmos. Oceanic Technol.*,
        **30** (10), 2248-2265, https://doi.org/10.1175/JTECH-D-12-00194.1

    [2] Ferrier, B. S., 1994: A double-moment multiple-phase four-class bulk ice
        scheme. Part I: Description. *J. Atmos. Sci.*, **51** (2), 249-280,
        https://doi.org/10.1175/1520-0469(1994)051<0249:ADMMPF>2.0.CO;2
    """
    from .dsd import dsd

    zname = _dbz_name(radar, dbzh)
    ref = _like(radar, zname)
    zh = np.asarray(ref.values, float)
    zz = np.asarray(radar["z"].values, float)
    zax = ref.dims.index("z")
    shape = [1] * ref.ndim
    shape[zax] = zz.size
    zagl = (zz - base.attrs["ground_height"]).reshape(shape)
    rho = np.asarray(base["rho"].interp(z=zz).values, float).reshape(shape)
    hmelt = base.attrs["melting_level"]
    if melting_layer is not None:
        ml_bot, ml_top = (float(v) for v in melting_layer)
    elif np.isfinite(hmelt):
        ml_bot, ml_top = hmelt - 0.5 * melting_depth, hmelt + 0.5 * melting_depth
    else:
        ml_bot = ml_top = np.inf
    if ml_top > ml_bot:
        f_ice = np.clip((zagl - ml_bot) / (ml_top - ml_bot), 0.0, 1.0)
    else:
        f_ice = (zagl >= ml_top).astype(float)
    below = f_ice < 1.0  # liquid present
    echo = np.isfinite(zh) & (zh >= min_dbz)
    zlin = np.where(echo, 10.0 ** (np.nan_to_num(zh) / 10.0), 0.0)
    rhog = _graupel_density(zagl, *graupel_density)
    dname = zdr
    if dname is None:
        dname = next(
            (
                n
                for n in (
                    "ZDR",
                    "differential_reflectivity",
                    "corrected_differential_reflectivity",
                )
                if n in radar
            ),
            None,
        )
    qr = np.zeros(zh.shape)
    nr = np.zeros(zh.shape)
    zr = np.zeros(zh.shape)
    dsd_ok = np.zeros(zh.shape, bool)
    if dname is not None:
        sub = radar[[zname, dname] + ([kdp] if kdp else [])]
        d = dsd(
            sub,
            dbzh=zname,
            zdr=dname,
            kdp=kdp,
            band=band,
            mu_lambda=mu_lambda,
            engine=engine,
            n_threads=n_threads,
        )
        n0 = np.asarray(d["N0"].transpose(*ref.dims).values, float)
        mu = np.asarray(d["MU"].transpose(*ref.dims).values, float)
        lam = np.asarray(d["LAMBDA"].transpose(*ref.dims).values, float)
        lwc = np.asarray(d["LWC"].transpose(*ref.dims).values, float)
        from scipy.special import gammaln

        with np.errstate(all="ignore"):
            nt = n0 * np.exp(gammaln(mu + 1.0)) / lam ** (mu + 1.0)
            zd = n0 * np.exp(gammaln(mu + 7.0)) / lam ** (mu + 7.0)
        dsd_ok = (
            below
            & echo
            & np.isfinite(lwc)
            & (lwc > 0)
            & np.isfinite(nt)
            & np.isfinite(zd)
        )
        qr = np.where(dsd_ok, lwc * 1.0e-3 / rho, 0.0)
        nr = np.where(dsd_ok, nt, 0.0)
        zr = np.where(dsd_ok, zd, 0.0)
    else:
        warnings.warn(
            "no ZDR: rain from reflectivity with a fixed intercept", stacklevel=3
        )
    hname = ("HID" if "HID" in radar else None) if hid == "auto" else hid
    hid_g = np.zeros(zh.shape, bool)
    if hname is not None:
        h = _like(radar, hname)
        names = str(h.attrs.get("flag_meanings", "")).split()
        codes = np.asarray(h.attrs.get("flag_values", np.arange(1, len(names) + 1)))
        gcodes = [int(c) for c, n in zip(codes, names) if "graupel" in n or "hail" in n]
        hid_g = np.isin(np.asarray(h.values), gcodes)
    hail = below & echo & ~dsd_ok & (zh >= graupel_min_dbz)
    # fallback rain from Z with a fixed intercept (Ziegler 2013a, eqs. 5-7)
    fallback = below & echo & ~dsd_ok & ~hail & ~hid_g
    with np.errstate(all="ignore"):
        lam_r = (C_R * math.gamma(7.0) * rain_intercept / zlin) ** (1.0 / 7.0)
        qr = np.where(fallback, np.pi * RHO_W * rain_intercept / (rho * lam_r**4), qr)
        nr = np.where(fallback, rain_intercept / lam_r, nr)
        zr = np.where(fallback, zlin, zr)
        # graupel reflectivity: below the melting layer what rain leaves in HID
        # graupel/hail or rain-hail cells; blended to all of Z_H above it
        resid = np.where(hid_g | hail, np.maximum(zlin - zr, 0.0), 0.0)
        zg = np.where(echo, f_ice * zlin + (1.0 - f_ice) * resid, 0.0)
        wr = 1.0 - f_ice
        qr, nr = qr * wr, nr * wr
        lam_g = (
            A_R_DIEL * C_G * (np.pi * rhog / RHO_W) ** 2 * graupel_intercept / zg
        ) ** (1.0 / 7.0)
        qg = np.where(zg > 0, np.pi * rhog * graupel_intercept / (rho * lam_g**4), 0.0)
        ng = np.where(zg > 0, graupel_intercept / lam_g, 0.0)
    qg, ng = _scale_graupel(qg, ng, rho, rhog, graupel_scale)
    return _closure_output(
        ref,
        qr,
        nr,
        qg,
        ng,
        {"precipitation_closure": "polarimetric (radarx DSD + HID)"},
    )


_PROFILE_FILES = {
    "cm1_squall_line": "ziegler2013_profiles_cm1_squall_line.csv",
}
_PROFILE_META = {
    "Z0r": ("Z0r(z*) of Ziegler (2013a) eq. (9)", "dBZ"),
    "S0_qr": ("[S0(z*)]_qr of Ziegler (2013a) eq. (9)", "dB"),
    "Z0g": ("Z0g(z*) of Ziegler (2013a) eq. (11)", "dBZ"),
    "S0_qg": ("[S0(z*)]_qg of Ziegler (2013a) eq. (11)", "dB"),
    "S0_n0g": ("[S0(z*)]_n0g of Ziegler (2013a) eq. (10)", "m-4 dBZ-1"),
}


@provenance("Regression profiles of the Ziegler (2013a) closure")
def ziegler2013_profiles(name="cm1_squall_line"):
    """
    Regression profiles of the Ziegler (2013a) closure shipped with radarx.

    Ziegler (2013a) [1] (sect. 2d, pp. 2252-2253) fitted the profiles
    :math:`Z_{0r}(z^*)`, :math:`S_{0,qr}(z^*)`, :math:`Z_{0g}(z^*)`,
    :math:`S_{0,qg}(z^*)` and :math:`S_{0,n0g}(z^*)` of eqs. (9)-(11) to a
    simulated supercell by Levenberg-Marquardt non-linear least squares at
    500 m levels below 5 km, spline-fitted them and wrote look-up tables at
    100 m, but did not publish them (only :math:`Z_{0g}` = 44.19 dBZ at 5 km,
    Fig. 3 caption, p. 2255). ``"cm1_squall_line"`` are the same kind of
    regressions fitted by radarx to a CM1 (release 21.1; Bryan and Fritsch
    2002 [2]) squall-line simulation with
    the GSR-LFO single-moment three-ice microphysics (rain intercept
    :math:`8 \\times 10^6` m\\ :sup:`-4`, graupel intercept
    :math:`3.355 \\times 10^5` m\\ :sup:`-4`, graupel density 660 kg
    m\\ :sup:`-3`): Levenberg-Marquardt fits per level at points with
    :math:`w < 5` m s\\ :sup:`-1`, smoothed with a cubic smoothing spline and
    tabulated every 100 m of :math:`z^* = z + (3.9\\ \\mathrm{km} -
    H_{melt})`, with :math:`H_{melt}` = 3.48 km (the 3.9 km reference is Table 2 of Z13a;
    the fit settings and the smoothing spline are radarx's). The provenance is in the
    header of the data file (``radarx/retrieve/data``) and in the attributes.
    They describe a squall line, not Ziegler's supercell; the closure is most
    consistent with them with ``constants={"n0r": 8e6}`` and
    ``graupel_density=(660, 660)``.

    Parameters
    ----------
    name : str, optional
        Table name. Default (and only): ``"cm1_squall_line"``.

    Returns
    -------
    xarray.Dataset
        ``Z0r``, ``S0_qr``, ``Z0g``, ``S0_qg``, ``S0_n0g`` on ``z_star`` (m).

    References
    ----------
    [1] Ziegler, C. L., 2013a: A diabatic Lagrangian technique for the analysis
        of convective storms. Part I: Description and validation via an
        observing system simulation experiment. *J. Atmos. Oceanic Technol.*,
        **30** (10), 2248-2265, https://doi.org/10.1175/JTECH-D-12-00194.1

    [2] Bryan, G. H., and J. M. Fritsch, 2002: A benchmark simulation for moist
        nonhydrostatic numerical models. *Mon. Wea. Rev.*, **130** (12),
        2917-2928,
        https://doi.org/10.1175/1520-0493(2002)130<2917:ABSFMN>2.0.CO;2
    """
    if name not in _PROFILE_FILES:
        raise ValueError(f"unknown profiles {name!r}; one of {sorted(_PROFILE_FILES)}")
    path = Path(__file__).parent / "data" / _PROFILE_FILES[name]
    lines = path.read_text().splitlines()
    notes = [ln[1:].strip() for ln in lines if ln.startswith("#")]
    rows = [ln for ln in lines if ln and not ln.startswith("#")]
    cols = [c.strip() for c in rows[0].split(",")]
    data = np.array([[float(v) for v in r.split(",")] for r in rows[1:]])
    zstar = {
        "long_name": "height above the ground with the melting level at 3.9 km",
        "units": "m",
    }
    return xr.Dataset(
        {
            k: ("z_star", data[:, cols.index(k)], {"long_name": ln, "units": u})
            for k, (ln, u) in _PROFILE_META.items()
        },
        coords={"z_star": ("z_star", data[:, cols.index("z_star")], zstar)},
        attrs={
            "title": notes[0],
            "source": "CM1 r21.1 squall-line simulation (GSR-LFO microphysics)",
            "history": " ".join(notes[1:]),
            "h_melt_m": 3476.6,
            "n0r_m-4": 8.0e6,
            "graupel_density_kg_m-3": 660.0,
        },
    )


@provenance("Rain and graupel from reflectivity, closure of Ziegler (2013a)")
def ziegler2013_precipitation(
    radar,
    base,
    *,
    profiles=None,
    w="w",
    dbzh="auto",
    constants=None,
    freezing_height=None,
    graupel_scale=1.0,
    graupel_density=(690.0, 630.0),
):
    """
    Rain and graupel from reflectivity with the closure of Ziegler (2013a) [1].

    Provisional contents and the graupel intercept (Z13a eqs. 9-11, p. 2253)

    .. math::

        q_r^* = \\epsilon_r e^{[Z_H - Z_{0r}(z^*)]/S_{0,qr}(z^*)},\\quad
        n_{0g} = (n_{0g})_0 - S_{0,n0g}(z^*) Z_H,\\quad
        q_g^* = \\epsilon_g e^{[Z_H - Z_{0g}(z^*)]/S_{0,qg}(z^*)},

    with :math:`z^* = z + (3.9\\ \\mathrm{km} - H_{melt})`, then graupel from
    the updraft and height rules (eqs. 12-16), :math:`\\lambda_g` from
    eq. (17) or (6), rain partitioned from the reflectivity left after
    graupel (eq. 7) where :math:`w \\ge W_{min}`, and the optional graupel
    concentration scaling (eqs. 18-21). The height rules are evaluated in
    the :math:`z^*` frame, in which the melting level is at 3.9 km. The
    conditions are those of eqs. (12)-(16), p. 2254, with
    :math:`w^* = (w - W_{min})/(W_{max} - W_{min})` and
    :math:`C_{frz} = \\exp[-a_{frz}(z^* - H_{melt})/(H_{frz} - H_{melt})]`
    (p. 2256); :math:`\\lambda_g` is from eq. (17) when :math:`z^* \\ge
    H_{melt}` and :math:`q_g = q_g^*`, else eq. (6); eqs. (18)-(21),
    pp. 2256-2257, are used for ``graupel_scale``. All constants are those of
    Z13a Table 2 (p. 2252): :math:`\\epsilon_r` = 0.5, :math:`\\epsilon_g` = 1.0
    (g kg\\ :sup:`-1`), :math:`(n_{0g})_0` = 3.355 x 10\\ :sup:`5`, :math:`n_{0r}`
    = 8 x 10\\ :sup:`5` m\\ :sup:`-4`, :math:`W_{min}` = 5, :math:`W_{max}` = 20
    m s\\ :sup:`-1`, :math:`a_{frz}` = 0.1, :math:`H_{melt}` = 3.9 km. Z13b
    [2] (Table 1, p. 2270) uses :math:`H_{melt}` = 3.7 km and :math:`H_{frz}` =
    7.1 km for the Greensburg storm instead of 3.9 and 7.0 km (Z13a Table 2);
    here :math:`H_{melt}` and :math:`H_{frz}` default to the 0 and -15 degC
    levels of the sounding (radarx choice, as Z13a/b take them from the
    inflow sounding or an updraft-core estimate).

    Parameters
    ----------
    radar : xarray.Dataset
        Reflectivity (dBZ) and vertical velocity on ``(time, z, y, x)``.
    base : xarray.Dataset
        Base state as passed by :func:`diabatic_lagrangian`.
    profiles : xarray.Dataset or str, optional
        Regression profiles on a ``z_star`` coordinate (m above the ground in
        the scaled frame): ``Z0r``, ``Z0g`` (dBZ), ``S0_qr``, ``S0_qg``
        (dB) and ``S0_n0g`` (m-4 dBZ-1), or the name of a table of
        :func:`ziegler2013_profiles`. Ziegler (2013a) fitted them to a
        simulated supercell and does not tabulate them (only
        :math:`Z_{0g}` = 44.19 dBZ at 5 km is quoted). Default: the
        ``"cm1_squall_line"`` tables, fitted to a simulated squall line, with
        a warning. Values beyond the profile take the nearest end.
    w : str, optional
        Vertical velocity variable. Default ``"w"``.
    dbzh : str, optional
        Reflectivity variable. Default: found automatically.
    constants : dict, optional
        Overrides of :data:`ZIEGLER_CONSTANTS` (Z13a Table 2: ``eps_r``,
        ``eps_g`` in g kg-1, ``n0g0``, ``n0r`` in m-4, ``h_melt_ref`` in m,
        ``a_frz``, ``w_min``, ``w_max`` in m s-1).
    freezing_height : float, optional
        :math:`H_{frz}`, height of -15 degC in the updraft core (m above the
        ground; Z13a uses 7.0 km in Table 2, Z13b 7.1 km). Default: the
        -15 degC level of the base state (radarx choice).
    graupel_scale : float, optional
        :math:`a_N` (eq. 18). Default 1.
    graupel_density : (float, float), optional
        Graupel density at the ground and at 5 km (kg m-3). Default
        (690, 630) (Z13a Table 2).

    Returns
    -------
    xarray.Dataset
        ``qr``, ``qg`` (kg kg-1), ``nr``, ``ng`` (m-3).

    References
    ----------
    [1] Ziegler, C. L., 2013a: A diabatic Lagrangian technique for the analysis
        of convective storms. Part I: Description and validation via an
        observing system simulation experiment. *J. Atmos. Oceanic Technol.*,
        **30** (10), 2248-2265, https://doi.org/10.1175/JTECH-D-12-00194.1

    [2] Ziegler, C. L., 2013b: A diabatic Lagrangian technique for the analysis
        of convective storms. Part II: Application to a radar-observed storm.
        *J. Atmos. Oceanic Technol.*, **30** (10), 2266-2280,
        https://doi.org/10.1175/JTECH-D-13-00036.1
    """
    need = ("Z0r", "Z0g", "S0_qr", "S0_qg", "S0_n0g")
    if profiles is None or isinstance(profiles, str):
        name = "cm1_squall_line" if profiles is None else profiles
        profiles = ziegler2013_profiles(name)
        warnings.warn(
            "Ziegler (2013a) did not publish the regression profiles of eqs. 9-11; "
            f"using the radarx {name!r} tables, fitted to a CM1 squall-line "
            "simulation (not Ziegler's supercell). Pass profiles= to use your own.",
            stacklevel=2,
        )
    if not all(k in profiles for k in need) or "z_star" not in profiles.coords:
        raise ValueError(
            "the Ziegler (2013a) closure needs the regression profiles Z0r, Z0g, S0_qr, "
            "S0_qg and S0_n0g on a 'z_star' coordinate (profiles=)"
        )
    c = dict(ZIEGLER_CONSTANTS)
    if constants:
        unknown = set(constants) - set(c)
        if unknown:
            raise ValueError(f"unknown constants: {sorted(unknown)}")
        c.update(constants)
    zname = _dbz_name(radar, dbzh)
    ref = _like(radar, zname)
    zh = np.asarray(ref.values, float)
    ww = np.asarray(_like(radar, w).values, float)
    zz = np.asarray(radar["z"].values, float)
    zax = ref.dims.index("z")
    shape = [1] * ref.ndim
    shape[zax] = zz.size
    zagl = zz - base.attrs["ground_height"]
    hmelt = base.attrs["melting_level"]
    if not np.isfinite(hmelt):
        raise ValueError("the base state has no melting level")
    hfrz = (
        base.attrs["minus15_level"]
        if freezing_height is None
        else float(freezing_height)
    )
    shift = c["h_melt_ref"] - hmelt
    zstar = zagl + shift
    hm_s = c["h_melt_ref"]
    hf_s = hfrz + shift
    zp = np.asarray(profiles["z_star"].values, float)
    o = np.argsort(zp)

    def prof(name):
        return np.interp(
            zstar, zp[o], np.asarray(profiles[name].values, float)[o]
        ).reshape(shape)

    z0r, z0g, sqr, sqg, sn0g = (prof(k) for k in need)
    zs = zstar.reshape(shape)
    rho = np.asarray(base["rho"].interp(z=zz).values, float).reshape(shape)
    rhog = _graupel_density(zagl.reshape(shape), *graupel_density)
    valid = np.isfinite(zh)
    zh0 = np.nan_to_num(zh, nan=-100.0)
    zlin = np.where(valid, 10.0 ** (zh0 / 10.0), 0.0)
    with np.errstate(all="ignore"):
        qr_s = 1e-3 * c["eps_r"] * np.exp((zh0 - z0r) / sqr)  # (9)
        n0g = c["n0g0"] - sn0g * zh0  # (10)
        qg_s = 1e-3 * c["eps_g"] * np.exp((zh0 - z0g) / sqg)  # (11)
        wstar = (ww - c["w_min"]) / (c["w_max"] - c["w_min"])
        cfrz = np.exp(-c["a_frz"] * (zs - hm_s) / (hf_s - hm_s))
        mid = (ww >= c["w_min"]) & (ww <= c["w_max"])
        rule12 = (ww < c["w_min"]) | (zs > hf_s)
        qg = np.select(
            [
                rule12,
                (ww > c["w_max"]) & (zs < hm_s),  # (13)
                mid & (zs < hm_s),  # (14)
                mid & (zs >= hm_s),  # (15)
                (ww > c["w_max"]) & (zs >= hm_s),  # (16)
            ],
            [
                qg_s,
                0.0,
                (1.0 - wstar) * qg_s,
                cfrz * (1.0 - wstar) * qg_s,
                (1.0 - cfrz) * qg_s,
            ],
            default=qg_s,
        )
        qg = np.where(valid & (n0g > 0) & np.isfinite(ww), qg, 0.0)
        lam17 = np.cbrt(
            A_R_DIEL * C_G * np.pi * rho * rhog * qg / (RHO_W**2 * zlin)
        )  # (17)
        lam6 = (np.pi * rhog * n0g / (rho * qg)) ** 0.25  # (6)
        lam_g = np.where((zs >= hm_s) & rule12, lam17, lam6)
        ng = np.where(qg > 0, rho * qg * lam_g**3 / (np.pi * rhog), 0.0)  # (5), (6)
        n0g_eff = ng * lam_g
        zeg = np.where(
            qg > 0,
            A_R_DIEL * C_G * (np.pi * rhog / RHO_W) ** 2 * n0g_eff / lam_g**7,
            0.0,
        )
        zer = zlin - zeg
        lam_r7 = (C_R * math.gamma(7.0) * c["n0r"] / zer) ** (1.0 / 7.0)  # (7)
        lam_r6 = (np.pi * RHO_W * c["n0r"] / (rho * qr_s)) ** 0.25  # (6)
        updraft = ww >= c["w_min"]
        lam_r = np.where(updraft, lam_r7, lam_r6)
        qr = np.where(
            updraft,
            np.where(zer > 0, np.pi * RHO_W * c["n0r"] / (rho * lam_r**4), 0.0),
            qr_s,
        )
        qr = np.where(valid & np.isfinite(ww), qr, 0.0)
        nr = np.where(qr > 0, c["n0r"] / lam_r, 0.0)
    qg, ng = _scale_graupel(qg, ng, rho, rhog, graupel_scale)
    return _closure_output(
        ref, qr, nr, qg, ng, {"precipitation_closure": "Ziegler (2013a) eqs. 9-21"}
    )


# ---------------------------------------------------------------------------
# Microphysical rates
# ---------------------------------------------------------------------------


def _switches(processes):
    if processes is None:
        processes = {}
    if isinstance(processes, str):
        key = processes.upper()
        if key not in SENSITIVITY_TESTS:
            raise ValueError(
                f"unknown sensitivity test {processes!r}; one of {sorted(SENSITIVITY_TESTS)}"
            )
        processes = SENSITIVITY_TESTS[key]
    unknown = set(processes) - set(PROCESSES)
    if unknown:
        raise ValueError(f"unknown processes: {sorted(unknown)}; known: {PROCESSES}")
    on = {k: bool(processes.get(k, True)) for k in PROCESSES}
    bits = 0
    for k, b in _BITS.items():
        if on[k]:
            bits |= b
    return on, bits


@provenance("Microphysical rates of Lin et al. (1983) and the resulting DLA tendencies")
def microphysical_rates(
    theta,
    pressure,
    qv,
    qc,
    qr,
    nr,
    qg,
    ng,
    *,
    graupel_density=690.0,
    rho0=1.2,
    dt=20.0,
    processes=None,
    engine="auto",
    n_threads=None,
):
    """
    LFO83 microphysical rates and the resulting DLA tendencies.

    Rain evaporation (eq. 52), collection of cloud water by rain (eq. 51) and
    graupel (eq. 40), graupel melting (eq. 47 with eq. 42), graupel
    sublimation (eq. 46) and Bigg freezing of rain (eq. 45) of Lin et al.
    (1983) [1], with the constants listed in the module documentation (the
    equation, appendix and page pointers are given there). Z13a [2] says the
    DLA uses the modified LFO formulation of Gilmore et al. (2004a), whose
    supplement was not consulted; see the module documentation for what
    differs. The tendencies are limited as a numerical safeguard of radarx.

    Parameters
    ----------
    theta : array-like or xarray.DataArray
        Potential temperature (K).
    pressure : array-like or xarray.DataArray
        Pressure (Pa).
    qv, qc, qr, qg : array-like or xarray.DataArray
        Mixing ratios of vapour, cloud, rain and graupel (kg kg-1).
    nr, ng : array-like or xarray.DataArray
        Number concentrations of rain and graupel (m-3).
    graupel_density : float or array-like, optional
        Graupel density (kg m-3). Default 690 (surface value of Z13a [2],
        Table 2).
    rho0 : float, optional
        Air density at the ground (kg m-3) of the fall speeds (:math:`\\rho_0`
        of eqs. 7 and 51). Default 1.2 (radarx choice; LFO83 give no value).
    dt : float, optional
        Time step (s) of the limits on the tendencies. Default 20 (Z13a sect.
        2b).
    processes : dict or str, optional
        Process switches (see :func:`diabatic_lagrangian`).
    engine, n_threads : optional
        Implementation and threads.

    Returns
    -------
    xarray.Dataset
        The rates ``P_REVP``, ``P_RACW``, ``P_GACW``, ``P_GACR``, ``P_GMLT``,
        ``P_GSUB``, ``P_GFR`` (kg kg-1 s-1, signed source terms of rain or
        graupel as in LFO83) and the limited tendencies ``dtheta_dt`` (K s-1),
        ``dqv_dt``, ``dqc_dt`` (kg kg-1 s-1).

    References
    ----------
    [1] Lin, Y.-L., R. D. Farley, and H. D. Orville, 1983: Bulk parameterization
        of the snow field in a cloud model. *J. Climate Appl. Meteor.*, **22**
        (6), 1065-1092,
        https://doi.org/10.1175/1520-0450(1983)022<1065:BPOTSF>2.0.CO;2

    [2] Ziegler, C. L., 2013a: A diabatic Lagrangian technique for the analysis
        of convective storms. Part I: Description and validation via an
        observing system simulation experiment. *J. Atmos. Oceanic Technol.*,
        **30** (10), 2248-2265, https://doi.org/10.1175/JTECH-D-12-00194.1
    """
    _, bits = _switches(processes)
    arrs = [theta, pressure, qv, qc, qr, nr, qg, ng, graupel_density]
    das = [a for a in arrs if isinstance(a, xr.DataArray)]
    if das:
        b = xr.broadcast(
            *[a if isinstance(a, xr.DataArray) else xr.DataArray(a) for a in arrs]
        )
        dims, coords = b[0].dims, b[0].coords
        vals = [np.asarray(x.values, float) for x in b]
    else:
        vals = [
            np.asarray(a, float)
            for a in np.broadcast_arrays(*[np.asarray(a, float) for a in arrs])
        ]
        dims = tuple(f"dim_{i}" for i in range(vals[0].ndim))
        coords = None
    shp = vals[0].shape
    flat = [np.ascontiguousarray(v.ravel()) for v in vals]
    if _traj._use_compiled(engine):
        r, d = _traj._lagrangian.rates(
            *flat, float(rho0), int(bits), float(dt), n_threads=int(n_threads or 0)
        )
        r, d = np.asarray(r), np.asarray(d)
    else:
        r, d = _nk.rates(*flat, float(rho0), int(bits), float(dt))
    names = ("P_REVP", "P_RACW", "P_GACW", "P_GACR", "P_GMLT", "P_GSUB", "P_GFR")
    data = {
        n: (dims, r[:, i].reshape(shp), {"units": "kg kg-1 s-1"})
        for i, n in enumerate(names)
    }
    data["dtheta_dt"] = (dims, d[:, 0].reshape(shp), {"units": "K s-1"})
    data["dqv_dt"] = (dims, d[:, 1].reshape(shp), {"units": "kg kg-1 s-1"})
    data["dqc_dt"] = (dims, d[:, 2].reshape(shp), {"units": "kg kg-1 s-1"})
    return xr.Dataset(data, coords=coords, attrs={"source": "Lin et al. (1983) rates"})


# ---------------------------------------------------------------------------
# Gridding: hole filling and the nine-point filter
# ---------------------------------------------------------------------------


def _hole_fill(a):
    """Fill NaN on each level with the mean of valid horizontal neighbours.

    Ziegler (2013a, sect. 2a, p. 2250) hole-fills "from surrounding nonmissing
    grid points" without giving the rule; the repeated 8-neighbour mean is a
    radarx choice.
    """
    out = np.array(a, dtype=float, copy=True)
    for k in range(out.shape[0]):
        f = out[k]
        miss = np.isnan(f)
        if not miss.any() or miss.all():
            continue
        for _ in range(f.shape[0] + f.shape[1]):
            if not miss.any():
                break
            p = np.pad(f, 1, constant_values=np.nan)
            s = np.zeros_like(f)
            n = np.zeros_like(f)
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    if dj == 0 and di == 0:
                        continue
                    nb = p[1 + dj : 1 + dj + f.shape[0], 1 + di : 1 + di + f.shape[1]]
                    ok = np.isfinite(nb)
                    s += np.where(ok, nb, 0.0)
                    n += ok
            fill = miss & (n > 0)
            f[fill] = s[fill] / n[fill]
            miss = np.isnan(f)
        out[k] = f
    return out


def _nine_point(a, passes=1):
    """Horizontal nine-point (1-2-1 x 1-2-1) low-pass filter, edges repeated.

    Ziegler (2013a, sect. 2a, p. 2250) applies "a horizontal nine-point
    elliptic low-pass filter" without giving the weights; the 1-2-1 by 1-2-1
    weights (the second-order Shapiro 1970 filter in each direction, not
    checked against that paper) are a radarx choice, as is the repetition
    of edge values and the leaving of missing values unchanged.
    """
    out = np.array(a, dtype=float, copy=True)
    for _ in range(int(passes)):
        p = np.pad(out, ((0, 0), (1, 1), (1, 1)), mode="edge")
        c = p[:, 1:-1, 1:-1]
        sides = p[:, :-2, 1:-1] + p[:, 2:, 1:-1] + p[:, 1:-1, :-2] + p[:, 1:-1, 2:]
        corners = p[:, :-2, :-2] + p[:, :-2, 2:] + p[:, 2:, :-2] + p[:, 2:, 2:]
        new = 0.25 * c + 0.125 * sides + 0.0625 * corners
        out = np.where(np.isnan(new), out, new)
    return out


# ---------------------------------------------------------------------------
# DLA
# ---------------------------------------------------------------------------


def _mesoscale(meso, prep, base):
    """Mesoscale theta and q_v packed (nt, nz, ny, nx, 2), surface gradients
    (nt, 2, ny, nx, 4) and the times (s relative to the analysis time)."""
    m = meso
    for c in ("x", "y", "z"):
        if c not in m.coords:
            raise ValueError(f"the mesoscale analysis needs a {c!r} coordinate")
    if not (
        np.array_equal(m["x"].values, prep["x"])
        and np.array_equal(m["y"].values, prep["y"])
        and np.array_equal(m["z"].values, prep["z"])
    ):
        m = m.interp(x=prep["x"], y=prep["y"], z=prep["z"])
    if "time" in m.dims:
        m = m.sortby("time")
        mt = (
            m["time"].values.astype("datetime64[ns]") - prep["time"]
        ) / np.timedelta64(1, "s")
        mt = np.asarray(mt, dtype=np.float64)
        if np.any(np.diff(mt) <= 0):
            raise ValueError("the mesoscale times must be distinct")
    else:
        m = m.expand_dims(time=[prep["time"]])
        mt = np.zeros(1)
    order = ("time", "z", "y", "x")
    if "theta" in m:
        th = m["theta"]
    elif "temperature" in m:
        p = m["pressure"] if "pressure" in m else base["pressure"]
        th = m["temperature"] * (_nk.P0 / p) ** _nk.KAPPA
    else:
        raise ValueError("the mesoscale analysis needs 'theta' or 'temperature'")
    if "qv" in m:
        qv = m["qv"]
    elif "mixing_ratio" in m:
        qv = m["mixing_ratio"]
    elif "specific_humidity" in m:
        qv = m["specific_humidity"] / (1.0 - m["specific_humidity"])
    else:
        raise ValueError(
            "the mesoscale analysis needs 'qv', 'mixing_ratio' or 'specific_humidity'"
        )
    th = np.asarray(th.broadcast_like(m).transpose(*order).values, float)
    qv = np.asarray(qv.broadcast_like(m).transpose(*order).values, float)
    if np.isnan(th).any() or np.isnan(qv).any():
        th = np.where(np.isnan(th), base["theta"].values[None, :, None, None], th)
        qv = np.where(np.isnan(qv), base["qv"].values[None, :, None, None], qv)
    packed = np.stack([th, qv], axis=-1).astype(np.float32)
    grads = []
    for k in range(th.shape[0]):
        gy_t, gx_t = np.gradient(th[k, 0], prep["y"], prep["x"])
        gy_q, gx_q = np.gradient(qv[k, 0], prep["y"], prep["x"])
        g = np.stack([gx_t, gy_t, gx_q, gy_q], axis=-1)
        grads.append(np.stack([g, g]))
    grad = np.stack(grads).astype(np.float32)
    return packed, grad, mt


_OBS_KEYS = ("window", "radius", "z_tolerance", "kappa_s", "tau_i", "tau_l")


def _observation_options(options):
    """Options of the in situ initialization with defaults and checks."""
    opts = dict(INSITU_DEFAULTS)
    unknown = set(options or {}) - set(opts)
    if unknown:
        raise ValueError(
            f"unknown observation options: {sorted(unknown)}; known: {sorted(opts)}"
        )
    opts.update(options or {})
    if opts["radius"] is None:
        opts["radius"] = 2.0 * math.sqrt(opts["kappa_s"])
    for k in _OBS_KEYS:
        if not float(opts[k]) > 0:
            raise ValueError(f"observation option {k!r} must be positive")
    return opts


def _observation_xy(flat, ds0):
    """Grid coordinates of the observations (x/y or latitude/longitude)."""
    if "x" in flat and "y" in flat:
        return flat["x"].astype(float), flat["y"].astype(float)
    lat = flat.get("lat", flat.get("latitude"))
    lon = flat.get("lon", flat.get("longitude"))
    if lat is None or lon is None:
        raise ValueError("the observations need 'x' and 'y' or latitude/longitude")
    if "origin_latitude" not in ds0.attrs or "origin_longitude" not in ds0.attrs:
        raise ValueError(
            "observations given by latitude/longitude need the grid's "
            "'origin_latitude' and 'origin_longitude' attributes"
        )
    from ..grid.multi import _aeqd

    proj = _aeqd(ds0.attrs["origin_latitude"], ds0.attrs["origin_longitude"])
    x, y = proj(lon.astype(float), lat.astype(float))
    return np.asarray(x, float), np.asarray(y, float)


def _observation_theta(flat, p):
    if "theta" in flat:
        return flat["theta"].astype(float)
    if "temperature" in flat:
        return flat["temperature"].astype(float) * (_nk.P0 / p) ** _nk.KAPPA
    raise ValueError("the observations need 'theta' or 'temperature'")


def _observation_qv(flat, p):
    if "qv" in flat:
        return flat["qv"].astype(float)
    if "mixing_ratio" in flat:
        return flat["mixing_ratio"].astype(float)
    if "specific_humidity" in flat:
        q = flat["specific_humidity"].astype(float)
        return q / (1.0 - q)
    if "dewpoint" in flat:
        e = _es_bolton(flat["dewpoint"].astype(float))
        return _nk.EPS * e / (p - e)
    raise ValueError(
        "the observations need 'qv', 'mixing_ratio', 'specific_humidity' or 'dewpoint'"
    )


def _observations(obs, prep, base, options, ds0):
    """In situ observations as (n, 6) x, y, z, t (s from the analysis time),
    theta, q_v and the matching options (window, radius, z tolerance,
    kappa_s, tau_i, tau_L)."""
    if not isinstance(obs, xr.Dataset):
        raise TypeError("observations must be an xarray.Dataset")
    opts = _observation_options(options)
    o = obs.reset_coords()
    if "time" not in o:
        raise ValueError("the observations need 'time'")
    names = [n for n in (*o.data_vars, *o.coords) if o[n].dtype.kind in "fiumM"]
    b = xr.broadcast(*[o[n].reset_coords(drop=True) for n in names])
    flat = {n: np.asarray(a.values).ravel() for n, a in zip(names, b)}
    tt = flat["time"]
    if np.issubdtype(tt.dtype, np.datetime64):
        tt = (tt.astype("datetime64[ns]") - prep["time"]) / np.timedelta64(1, "s")
    t = np.asarray(tt, float)
    x, y = _observation_xy(flat, ds0)
    zname = next((n for n in ("z", "height", "altitude") if n in flat), None)
    if zname is None:
        z = np.full(t.shape, float(prep["z"][0]))
    else:
        z = flat[zname].astype(float)
    if "pressure" in flat:
        p = flat["pressure"].astype(float)
    else:
        p = np.interp(z, base["z"].values, base["pressure"].values)
    arr = np.column_stack(
        [x, y, z, t, _observation_theta(flat, p), _observation_qv(flat, p)]
    )
    arr = np.ascontiguousarray(arr[np.isfinite(arr).all(axis=1)])
    par = np.array([float(opts[k]) for k in _OBS_KEYS])
    return arr, par, opts


def _precipitation(precipitation, prep, base, kwargs):
    ds = prep["ds"]
    if precipitation is None or (
        isinstance(precipitation, str) and precipitation == "none"
    ):
        return None
    if isinstance(precipitation, xr.Dataset):
        pr = precipitation
    else:
        if precipitation == "polarimetric":
            func = polarimetric_precipitation
        elif precipitation == "ziegler2013":
            func = ziegler2013_precipitation
        elif callable(precipitation):
            func = precipitation
        else:
            raise ValueError(
                "precipitation must be 'polarimetric', 'ziegler2013', 'none', a callable or a Dataset"
            )
        pr = func(ds, base, **(kwargs or {}))
    for k in ("qr", "nr", "qg", "ng"):
        if k not in pr:
            raise ValueError(f"the precipitation closure must return {k!r}")
    if "time" not in pr.dims:
        pr = pr.expand_dims(time=ds["time"].values)
    pr = pr.reindex(time=ds["time"].values, method="nearest")
    order = ("time", "z", "y", "x")
    packed = np.empty(tuple(ds.sizes[d] for d in order) + (4,), dtype=np.float32)
    for i, k in enumerate(("qr", "nr", "qg", "ng")):
        packed[..., i] = pr[k].transpose(*order).values
    np.nan_to_num(packed, copy=False, nan=0.0)
    return pr, packed


def _ice_insitu_output(data, attrs, dims, qi, init, flags, ice, obs, obs_opts, params):
    """Output variables and attributes of the ice and in situ options."""
    if ice:
        data["qi"] = (
            dims,
            qi,
            {"long_name": "cloud ice mixing ratio", "units": "kg kg-1"},
        )
    if obs is not None:
        data["insitu_weight"] = (
            dims,
            init[..., 0],
            {
                "long_name": "sum of the Barnes weights of the in situ observations "
                "that initialised the trajectory (Ziegler et al. 2007, eq. 1)",
                "units": "1",
            },
        )
        data["insitu_time"] = (
            dims,
            init[..., 1],
            {
                "long_name": "start time of the trajectory at the in situ observation "
                "relative to the analysis time",
                "units": "s",
            },
        )
    if ice:
        attrs["ice_adjustment"] = (
            f"Tao et al. (1989), T00 = {params['t00']} K; homogeneous freezing at "
            "233.15 K and melting of cloud ice above 273.15 K"
        )
    if obs is not None:
        attrs["insitu_observations"] = int(obs.shape[0])
        attrs["insitu_options"] = ", ".join(f"{k}={obs_opts[k]}" for k in obs_opts)
        attrs["insitu_fraction"] = float(np.mean((flags & _nk.INSITU) != 0))


def _run_kernel(
    use_compiled,
    prep,
    par,
    starts,
    surface,
    table,
    precip,
    meso,
    meso_t,
    grad,
    thermo,
    obs,
    obs_par,
    n_threads,
):
    """The DLA of all start points with the compiled kernel or the NumPy
    reference."""
    table, bz0, bdz = table
    xyz = (prep["x"], prep["y"], prep["z"])
    moving = (prep["cx"], prep["cy"], prep["eb"], prep["ea"])
    if use_compiled:  # pragma: no cover - depends on the build
        dummy4 = np.zeros((1, 2, 2, 2, 4), np.float32)
        dummy2 = np.zeros((1, 2, 2, 2, 2), np.float32)
        return _traj._lagrangian.dla(
            prep["packed"],
            *xyz,
            prep["t"],
            starts,
            surface,
            par,
            *moving,
            table,
            bz0,
            bdz,
            dummy4 if precip is None else precip,
            precip is not None,
            dummy2 if meso is None else meso,
            meso_t,
            meso is not None,
            dummy4 if grad is None else grad,
            grad is not None,
            thermo,
            np.zeros((0, 6)) if obs is None else obs,
            np.zeros(6) if obs_par is None else obs_par,
            obs is not None,
            n_threads=int(n_threads or 0),
        )
    g = _nk.Grid(*xyz, prep["t"], prep["packed"], *moving)
    gp = None if precip is None else _nk.Grid(*xyz, prep["t"], precip, *moving)
    gm = None if meso is None else _nk.Grid(*xyz, meso_t, meso, 0, 0, 1e30, 1e30)
    gg = (
        None
        if grad is None
        else _nk.Grid(*xyz[:2], [0.0, 1.0], meso_t, grad, 0, 0, 1e30, 1e30)
    )
    return _nk.dla(
        g, par, starts, surface, table, bz0, bdz, gp, gm, gg, thermo, obs, obs_par
    )


@provenance(
    "Diabatic Lagrangian analysis of Ziegler (2013a, b), radarx choices where silent"
)
def diabatic_lagrangian(
    winds,
    sounding,
    time=None,
    *,
    mesoscale=None,
    precipitation="polarimetric",
    precipitation_kwargs=None,
    processes=None,
    parameters=None,
    ice=False,
    observations=None,
    observation_options=None,
    surface_flux=(0.0, 0.0),
    storm_motion=None,
    extend=0.0,
    extend_before=None,
    extend_after=None,
    dt=20.0,
    iterations=3,
    max_steps=None,
    levels=None,
    hole_fill=True,
    filter_passes=1,
    u="u",
    v="v",
    w="w",
    reflectivity="auto",
    termination=True,
    boundary="environment",
    valid=None,
    environment_mask=None,
    min_valid_fraction=None,
    engine="auto",
    n_threads=None,
):
    """
    Diabatic Lagrangian analysis (DLA) of Ziegler (2013a, b) [1, 2].

    The method, equations and defaults (Z13a Tables 1-3) are documented, with
    page-level pointers and the differences from the papers, in the module
    documentation.

    Parameters
    ----------
    winds : xarray.Dataset
        Time series of 3-D winds and radar data on ``(time, z, y, x)``
        (``x``, ``y``, ``z`` in m, increasing; the lowest level is the
        ground), with ``u``, ``v``, ``w`` (m s-1), reflectivity (dBZ) and the
        variables the precipitation closure needs (for the default closure
        ``ZDR`` and optionally ``KDP`` and ``HID``).
    sounding : xarray.Dataset
        Environmental sounding on ``height`` (same datum as ``z``), e.g. from
        :func:`radarx.io.sounding.read_sounding` or
        :func:`radarx.io.sounding.era5_profile`: ``pressure`` (Pa),
        ``temperature`` (K), ``specific_humidity`` (or ``dewpoint``) and
        ``u``, ``v``. It gives the base-state pressure, the base state of the
        damping and of :math:`\\Delta\\theta_v`, and (without ``mesoscale``)
        the initial :math:`\\theta`, :math:`q_v` of the trajectories.
    time : datetime-like, optional
        Analysis time. Default: the last wind time.
    mesoscale : xarray.Dataset, optional
        Heterogeneous environment on the grid (``z``, ``y``, ``x``, optionally
        ``time``): ``theta`` or ``temperature`` (with ``pressure``, else the
        base-state pressure), and ``qv``, ``mixing_ratio`` or
        ``specific_humidity``, e.g. :func:`radarx.io.sounding.era5_column`
        output (Z13b [2], sect. 3b). Used for the initial values at the
        origin and time of each trajectory, the damping base state and the
        surface-flux gradient; with a ``time`` axis it is interpolated
        linearly in time (and held constant before its first and after its
        last time).
    precipitation : str, callable, xarray.Dataset or None, optional
        Precipitation closure: ``"polarimetric"`` (default,
        :func:`polarimetric_precipitation`), ``"ziegler2013"``
        (:func:`ziegler2013_precipitation`, needs ``profiles`` in
        ``precipitation_kwargs``), a callable ``f(winds, base) -> Dataset``,
        a Dataset with ``qr``, ``nr``, ``qg``, ``ng`` on the winds grid, or
        ``"none"`` (no precipitation).
    precipitation_kwargs : dict, optional
        Keyword arguments of the closure.
    processes : dict or str, optional
        Switches of :data:`PROCESSES` (all True by default) or a sensitivity
        test of :data:`SENSITIVITY_TESTS` (``"NOLD"``, ...).
    parameters : dict, optional
        Overrides of :data:`DLA_DEFAULTS` (the source of every default is given
        there).
    ice : bool, optional
        Carry cloud ice :math:`q_i` and use the ice-water saturation
        adjustment of Tao et al. (1989) [4] with melting of cloud ice above
        0 degC and homogeneous freezing of cloud water at or below -40 degC
        (see "Ice processes" in the module documentation). Default False
        (Z13a [1]: water saturation only). :math:`T_{00}` is
        ``parameters["t00"]`` (default 233.15 K).
    observations : xarray.Dataset, optional
        In situ observations (e.g. surface stations or mobile mesonet) that
        initialise the trajectories passing near them (Z07 [3];
        Z13a, sect. 1, p. 2249), on any dimensions (e.g. ``station``,
        ``time``): ``time`` (datetime64, or seconds relative to the analysis
        time), ``x``, ``y`` (m, grid coordinates) or ``lat``/``latitude`` and
        ``lon``/``longitude`` (with the grid attributes ``origin_latitude``,
        ``origin_longitude``), ``z`` (or ``height``, ``altitude``; default:
        the ground), ``theta`` or ``temperature``, ``qv``,
        ``mixing_ratio``, ``specific_humidity`` or ``dewpoint``, and
        ``pressure`` (Pa; default: base-state pressure). Missing values are
        dropped. See "In situ initialization" in the module documentation.
    observation_options : dict, optional
        Overrides of :data:`INSITU_DEFAULTS`: ``window`` (s), ``radius`` (m),
        ``z_tolerance`` (m), ``kappa_s`` (m2), ``tau_i``, ``tau_l`` (s2).
    surface_flux : (float, float), optional
        Constant surface fluxes of :math:`\\theta` (K s-1) and :math:`q_v`
        (kg kg-1 s-1) added to the mesoscale advective flux of eq. (27).
        Default (0, 0).
    storm_motion, extend, extend_before, extend_after : optional
        Storm motion (a pair, an :func:`radarx.retrieve.estimate_motion`
        result or ``"estimate"``) and time morphing (s) before the first and
        after the last analysis (Z13b, sect. 2c, p. 2268), see
        :func:`radarx.retrieve.trajectories` and "Applying DLA to observed
        QLCS cases" in the module documentation.
    dt, iterations, max_steps : optional
        Trajectory options, see :func:`radarx.retrieve.trajectories` (``dt`` 20 s
        and ``iterations`` 3: Z13a sect. 2b).
    levels : sequence of int, optional
        Grid levels to analyse. Default: all.
    hole_fill : bool, optional
        Fill grid points without a valid trajectory from their neighbours
        (Z13a sect. 2a, p. 2250). Default True.
    filter_passes : int, optional
        Passes of the nine-point low-pass filter. Default 1; 0 for none (Z13a
        sect. 2a applies one light smoothing; the number of passes and the
        weights are radarx choices).
    u, v, w, reflectivity : str, optional
        Variable names, see :func:`radarx.retrieve.trajectories`.
    termination : {True, "ziegler2013", "precipitation"}, optional
        Environment test of the trajectories, see
        :func:`radarx.retrieve.trajectories`. Default: Z13a sect. 2a.
    boundary : str or sequence of str, optional
        Which lateral-boundary exits count as environmental: ``"environment"``
        (default, every exit, Z13a sect. 2a), ``"environment_mask"``,
        ``"no_echo"`` or a list of sides; other exits get flag 128 and are
        hole-filled. See :func:`radarx.retrieve.trajectories`.
    valid, environment_mask : str or xarray.DataArray, optional
        Validity of the winds (e.g. ``"dd_valid"``) and the mask of
        environmental air for ``termination="precipitation"``, see
        :func:`radarx.retrieve.trajectories`.
    min_valid_fraction : float, optional
        Grid points whose trajectory spent a smaller fraction of its points in
        valid winds are set missing (flag 64) before hole filling. Default:
        not applied (``valid_fraction`` is still returned).
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation. ``"auto"`` (default) prefers the compiled kernel.
    n_threads : int, optional
        Threads of the compiled kernel. Default: all cores.

    Returns
    -------
    xarray.Dataset
        On ``(z, y, x)`` at the analysis time: ``theta``, ``temperature``,
        ``theta_v``, ``delta_theta_v`` (relative to the base state),
        ``qv``, ``qc`` (and ``qi`` with ``ice=True``), the diagnosed ``qr``, ``nr``, ``qg``, ``ng`` and
        ``rain_rate``; per grid point ``flags``, ``n_steps``, ``environment``,
        the trajectory origin ``origin_x``, ``origin_y``, ``origin_z``,
        ``origin_time``, ``valid_fraction`` (with ``observations`` also
        ``insitu_weight`` and ``insitu_time``, NaN where no observation
        matched) and the accumulated :math:`\\theta` change of each
        process ``dtheta_<process>`` (unfiltered, NaN without a valid
        trajectory); the base-state profiles ``theta_base``,
        ``theta_v_base``, ``pressure_base`` on ``z``.

    References
    ----------
    [1] Ziegler, C. L., 2013a: A diabatic Lagrangian technique for the analysis
        of convective storms. Part I: Description and validation via an
        observing system simulation experiment. *J. Atmos. Oceanic Technol.*,
        **30** (10), 2248-2265, https://doi.org/10.1175/JTECH-D-12-00194.1

    [2] Ziegler, C. L., 2013b: A diabatic Lagrangian technique for the analysis
        of convective storms. Part II: Application to a radar-observed storm.
        *J. Atmos. Oceanic Technol.*, **30** (10), 2266-2280,
        https://doi.org/10.1175/JTECH-D-13-00036.1

    [3] Ziegler, C. L., M. S. Buban, and E. N. Rasmussen, 2007: A Lagrangian
        objective analysis technique for assimilating in situ observations with
        multiple-radar-derived airflow. *Mon. Wea. Rev.*, **135** (7),
        2417-2442, https://doi.org/10.1175/MWR3396.1

    [4] Tao, W.-K., J. Simpson, and M. McCumber, 1989: An ice-water saturation
        adjustment. *Mon. Wea. Rev.*, **117** (1), 231-235,
        https://doi.org/10.1175/1520-0493(1989)117<0231:AIWSA>2.0.CO;2
    """
    params = _traj._parameters(parameters, DLA_DEFAULTS)
    on, bits = _switches(processes)
    if ice:
        bits |= _nk.ICE
        if not 0.0 < params["t00"] < _nk.T0:
            raise ValueError("parameters['t00'] must be between 0 K and 273.15 K")
    use_compiled = _traj._use_compiled(engine)
    prep = _traj._prepare(
        winds,
        time,
        "backward",
        u=u,
        v=v,
        w=w,
        reflectivity=reflectivity,
        storm_motion=storm_motion,
        extend=extend,
        surface_downdraft_on=on["surface_downdraft"],
        params=params,
        valid=valid,
        environment_mask=environment_mask,
        extend_before=extend_before,
        extend_after=extend_after,
    )
    if termination in (False, None):
        raise ValueError("the DLA needs a termination test of the trajectories")
    par = _traj._path_params(
        prep, "backward", dt, iterations, max_steps, termination, params, boundary
    )
    ground = float(prep["z"][0])
    table, bz0, bdz, base = _base_state(sounding, prep["z"], ground)
    pr = _precipitation(precipitation, prep, base, precipitation_kwargs)
    precip_ds, precip = (None, None) if pr is None else pr
    meso = grad = None
    meso_t = np.zeros(1)
    if mesoscale is not None:
        meso, grad, meso_t = _mesoscale(mesoscale, prep, base)
    q = dict(params)
    q.update(
        z_sfc=ground,
        rho0=base.attrs["rho0"],
        flux_theta=float(surface_flux[0]),
        flux_qv=float(surface_flux[1]),
        switches=float(bits),
    )
    thermo = np.array([float(q[k]) for k in _nk.THERMO_KEYS])
    obs = obs_par = obs_opts = None
    if observations is not None:
        obs, obs_par, obs_opts = _observations(
            observations, prep, base, observation_options, prep["ds"]
        )
    starts, index, ks = _traj._grid_starts(prep, levels, params["offset_height"])
    surface = (index[0] == 0).astype(np.int8)
    out, bud, org, npts, flags, init = _run_kernel(
        use_compiled,
        prep,
        par,
        starts,
        surface,
        (table, bz0, bdz),
        precip,
        meso,
        meso_t,
        grad,
        thermo,
        obs,
        obs_par,
        n_threads,
    )
    shape = (ks.size, prep["y"].size, prep["x"].size)
    out = np.asarray(out).reshape(shape + (4,))
    init = np.asarray(init).reshape(shape + (2,))
    bud = np.asarray(bud).reshape(shape + (_nk.N_BUDGET,))
    org = np.asarray(org).reshape(shape + (5,))
    npts = np.asarray(npts).reshape(shape)
    flags = np.asarray(flags).reshape(shape).astype(np.int32)
    if min_valid_fraction is not None:
        low = ~(org[..., 4] >= float(min_valid_fraction))
        flags = np.where(low, flags | 64, flags)
        out = np.where(low[..., None], np.nan, out)
        bud = np.where(low[..., None], np.nan, bud)
    theta, qv, qc, qi = (out[..., i] for i in range(4))
    if hole_fill:
        theta, qv, qc, qi = (_hole_fill(a) for a in (theta, qv, qc, qi))
    if filter_passes:
        theta, qv, qc, qi = (_nine_point(a, filter_passes) for a in (theta, qv, qc, qi))
    zsel = prep["z"][ks]
    bsel = base.isel(z=ks)
    pz = bsel["pressure"].values[:, None, None]
    temp = theta * (pz / _nk.P0) ** _nk.KAPPA
    thv = theta * (1.0 + qv / _nk.EPS) / (1.0 + qv)
    dims = ("z", "y", "x")
    ds0 = prep["ds"]
    coords = {
        "z": ("z", zsel, dict(ds0["z"].attrs)),
        "y": ds0["y"],
        "x": ds0["x"],
        "time": prep["time"],
    }
    for name in ds0.coords:
        if name not in coords and set(ds0[name].dims) <= {"y", "x"}:
            coords[name] = ds0[name]
    data = {
        "theta": (
            dims,
            theta,
            {"standard_name": "air_potential_temperature", "units": "K"},
        ),
        "temperature": (dims, temp, {"standard_name": "air_temperature", "units": "K"}),
        "theta_v": (
            dims,
            thv,
            {"long_name": "virtual potential temperature", "units": "K"},
        ),
        "delta_theta_v": (
            dims,
            thv - bsel["theta_v"].values[:, None, None],
            {
                "long_name": "virtual potential temperature minus base state",
                "units": "K",
            },
        ),
        "qv": (
            dims,
            qv,
            {"standard_name": "humidity_mixing_ratio", "units": "kg kg-1"},
        ),
        "qc": (dims, qc, {"long_name": "cloud water mixing ratio", "units": "kg kg-1"}),
        "flags": (
            dims,
            flags.astype(np.int32),
            {
                "long_name": "trajectory termination flags",
                "flag_masks": np.append(_traj.FLAG_MASKS, np.int32(_nk.INSITU)),
                "flag_meanings": _traj.FLAG_MEANINGS
                + " initialized_from_in_situ_observation",
            },
        ),
        "environment": (
            dims,
            ((flags & _nk.ENVIRONMENT) != 0) & ((flags & 64) == 0),
            {"long_name": "trajectory reached the storm environment"},
        ),
        "n_steps": (
            dims,
            np.maximum(npts - 1, 0),
            {"long_name": "number of trajectory steps"},
        ),
        "origin_x": (
            dims,
            org[..., 0],
            {"long_name": "x of the trajectory origin", "units": "m"},
        ),
        "origin_y": (
            dims,
            org[..., 1],
            {"long_name": "y of the trajectory origin", "units": "m"},
        ),
        "origin_z": (
            dims,
            org[..., 2],
            {"long_name": "height of the trajectory origin", "units": "m"},
        ),
        "origin_time": (
            dims,
            org[..., 3],
            {
                "long_name": "time of the trajectory origin relative to the analysis time",
                "units": "s",
            },
        ),
        "valid_fraction": (
            dims,
            org[..., 4],
            {
                "long_name": "fraction of the trajectory points with valid winds",
                "units": "1",
            },
        ),
        "theta_base": (
            "z",
            bsel["theta"].values,
            {"long_name": "base-state potential temperature", "units": "K"},
        ),
        "theta_v_base": (
            "z",
            bsel["theta_v"].values,
            {"long_name": "base-state virtual potential temperature", "units": "K"},
        ),
        "pressure_base": (
            "z",
            bsel["pressure"].values,
            {"long_name": "base-state pressure", "units": "Pa"},
        ),
    }
    for i, (key, text) in enumerate(_BUDGET):
        data[f"dtheta_{key}"] = (
            dims,
            bud[..., i],
            {
                "long_name": f"potential temperature change by {text} along the trajectory",
                "units": "K",
            },
        )
    if precip_ds is not None:
        pa = precip_ds.isel(z=ks)
        pt = pa.interp(time=prep["time"]) if pa.sizes["time"] > 1 else pa.isel(time=0)
        for k in ("qr", "nr", "qg", "ng"):
            data[k] = (
                dims,
                np.asarray(pt[k].transpose(*dims).values, float),
                dict(precip_ds[k].attrs),
            )
        rho = bsel["rho"].values[:, None, None]
        qr_, nr_ = data["qr"][1], data["nr"][1]
        with np.errstate(all="ignore"):
            lam = np.cbrt(np.pi * RHO_W * nr_ / (rho * qr_))
            vbar = np.sqrt(1.225 / rho) * 10.0 * (1.0 - (1.0 + 516.575 / lam) ** -4.0)
            rr = np.where(qr_ > 0, 3.6e6 * rho * qr_ * vbar / RHO_W, 0.0)
        data["rain_rate"] = (
            dims,
            rr,
            {
                "standard_name": "rainfall_rate",
                "long_name": "rain rate (Ziegler 2013b eqs. 7-9)",
                "units": "mm h-1",
            },
        )
    attrs = {
        "method": "diabatic Lagrangian analysis (Ziegler 2013a, b)",
        "microphysics": "Lin et al. (1983) rates; see radarx.retrieve.diabatic_lagrangian",
        "processes": ", ".join(k for k in PROCESSES if on[k]),
        "storm_motion": (prep["cx"], prep["cy"]),
        "extend": (prep["eb"], prep["ea"]),
        "boundary": str(boundary),
        "dt": float(dt),
        "melting_level": base.attrs["melting_level"],
        "ground_height": ground,
        "environment_fraction": float(np.mean((flags & _nk.ENVIRONMENT) != 0)),
        "termination": str(termination),
        "ice": int(bool(ice)),
    }
    _ice_insitu_output(data, attrs, dims, qi, init, flags, ice, obs, obs_opts, params)
    return xr.Dataset(data, coords=coords, attrs=attrs)


from .._registry import accessor_method  # noqa: E402


@accessor_method("dataset", name="diabatic_lagrangian")
def _diabatic_lagrangian_dataset_accessor(self, sounding, time=None, **kwargs):
    """
    Diabatic Lagrangian analysis of the winds and radar data in this dataset.

    See :func:`radarx.retrieve.diabatic_lagrangian` for the parameters.

    Returns
    -------
    xarray.Dataset
        Potential temperature, humidity, cloud water and virtual buoyancy at
        the analysis time.
    """
    return diabatic_lagrangian(self.xarray_obj, sounding, time, **kwargs)
