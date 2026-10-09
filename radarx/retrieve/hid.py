#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Hydrometeor Classification
==========================

Fuzzy-logic hydrometeor classification (HID) of every gate of a sweep, a
volume, a grid or a QVP from the reflectivity :math:`Z_H`, the differential
reflectivity :math:`Z_{DR}`, the specific differential phase :math:`K_{DP}`,
the copolar correlation coefficient :math:`\\rho_{hv}` and, optionally, the
temperature at each gate.

For each class :math:`i` and variable :math:`j` a membership function
:math:`P^{(i)}(V_j)` describes how typical the measured value :math:`V_j` is
of the class. The memberships of each class are aggregated into a score and
the class with the highest score is assigned to the gate. The score of the
assigned class is returned as ``HID_confidence`` and the scores of all
classes as ``HID_scores``.

Methods
-------
``"park"`` (S band, default for ``band="S"``)
    The WSR-88D algorithm of Park et al. (2009) [1]: trapezoidal membership
    functions (their Table 1, p. 733, with the :math:`Z_{DR}` and LKdp
    :math:`= 10 \\log_{10} K_{DP}` corners of the rain classes depending on
    :math:`Z_H` through their Eqs. 4 and 5, p. 733; LKdp as in their Eq. 1,
    p. 732), the class-dependent weights of their Table 2 (p. 734, the
    columns of :math:`Z_H`, :math:`Z_{DR}`, :math:`\\rho_{hv}` and LKdp) and
    additive aggregation :math:`A_i = \\sum_j W_{ij} Q_j P^{(i)}(V_j) /
    \\sum_j W_{ij} Q_j` (their Eq. 3, p. 732). The confidence vector
    :math:`Q` (their Eqs. 14-17, 23 and 25, pp. 734-737) accounts for
    attenuation (:math:`\\Phi_{DP}`, threshold 250 degrees), low
    :math:`\\rho_{hv}` (:math:`\\Delta\\rho_{hv}^{(1)} = 0.2`, switched off
    below 0.8) and partial beam blockage (:math:`a/50`, their Eq. 13). Classes
    that fail the hard thresholds of their Table 3 (p. 737) are skipped in
    favour of the next highest score. When the melting layer is known
    (``melting_layer=`` or from ``temperature``), the classes allowed in five
    intervals of beam height relative to the melting layer bottom and top,
    accounting for the beam width (their Fig. 2 and Eq. 24, p. 736), are
    enforced. Classes: dry snow, wet snow, ice crystals, graupel, big drops,
    light and moderate rain, heavy rain and rain-hail mixture. The numbers of
    Tables 1 to 3 and Eqs. 4, 5, 24 were checked against the paper; what
    radarx leaves out of the paper's algorithm is listed in the Notes.
``"dolan"`` (default for ``band="C"`` and ``"X"``)
    Theory-based beta membership functions
    :math:`\\beta = 1 / (1 + [((x - m)/a)^2]^b)` (Dolan and Rutledge 2009
    [2], their Eq. 14, p. 2080) and the hybrid aggregation of Dolan et al.
    (2013) [3] (their Eq. 8, p. 2167): :math:`\\mu_i = \\beta_{T,i}\\,
    \\beta_{Z,i}\\, (0.8\\,\\beta_{Z_{DR},i} + 1.0\\,\\beta_{K_{DP},i} +
    0.1\\,\\beta_{\\rho_{hv},i}) / 1.9`, with the weights 0.8, 1.0 and 0.1,
    which those authors determined subjectively. At C band the ten classes
    and membership functions of Dolan et al. (2013, Table A2, p. 2183):
    drizzle, rain, ice crystals, aggregates, wet snow, vertically aligned
    ice, low- and high-density graupel, hail and big drops. At X and S band
    the seven classes of Dolan and Rutledge (2009) with the variable ranges
    of their X-band (XMBF) and S-band (SMBF) membership functions (Tables
    3-9, pp. 2078-2079; :math:`m` and :math:`a` are the middle and half the
    width of the printed minimum and maximum). **This is not the algorithm of
    the 2009 paper.** There the beta score :math:`\\beta` of every variable
    "is calculated ... and then multiplied by a weight, and the result for
    each variable is then added together to define a score" (Sect. 3a,
    p. 2080), i.e. a purely additive aggregation; for the X-band case study
    with the CASA IP1 radars the weights were reflectivity 1.5, :math:`K_{DP}`
    1.0, temperature 0.5, :math:`Z_{DR}` 0.4 and :math:`\\rho_{hv}` 0.2, the
    last two low because of the data quality of that campaign (Sect. 3b,
    p. 2081). radarx instead applies the hybrid rule of the 2013 C-band paper
    (:math:`T` and :math:`Z_H` multiply the polarimetric score) with the
    weights 0.8, 1.0 and 0.1 to **all** bands, so at X and S band the scores
    and class boundaries differ from those of the 2009 algorithm although the
    :math:`m` and :math:`a` values match its Tables 3-9. The 2009 paper gives
    the ranges but not the slopes :math:`b` or temperature membership
    functions; these are taken from the same classes of Dolan et al. (2013),
    Table A2. The ``references`` attribute of the output names both papers
    for ``band="X"`` and ``"S"``.
``"thompson"`` (winter precipitation)
    The winter classification of Thompson et al. (2014) [4], their Table 5
    (p. 1470), with their band-dependent :math:`K_{DP}` functions and
    class-dependent weights, in additive aggregation. A melting-layer
    detection step separates wet snow from other echo; the median height of
    the wet snow gates between 5 and 35 km range defines the melting layer
    (p. 1466: top, median and base are the heights below which 80, 50 and 20
    % of the wet snow gates lie, after Giangrande et al. 2008, in each
    10 degree azimuth sector, using gates with SNR above 10 dB; radarx uses
    only the median, for the whole volume and without the SNR condition). If
    at least ``ml_gates[1]`` gates are wet snow (complete melting), rain and
    freezing/frozen rain are classified below it and plates, dendrites, ice
    crystals and aggregates above it; with at least ``ml_gates[0]`` (partial
    melting) the above-melting-layer classes are used everywhere and the wet
    snow is kept; otherwise the above-melting-layer classes are used
    everywhere. The defaults 100 and 10 000 gates are the values of Thompson
    et al. (p. 1466), which they tested for the very high spatial resolution
    of the OU-PRIME and CSU-CHILL RHIs and describe as dependent on the data
    quality and resolution of the radar.

Temperature
-----------
``temperature`` may be a sounding or ERA5 profile from
:mod:`radarx.io.sounding` (interpolated to the gate heights ``z`` with
:func:`radarx.io.sounding.interpolate_profile`), a field per gate or
``None``. The Dolan and Thompson methods use it as a membership variable
(without temperature it is left out of the aggregation). The Park method uses
it only to place the melting layer: its top is the wet-bulb 0 °C height of
the profile (the 0 °C height without humidity; Park et al. 2009 [1], p. 736,
call the top of the melting layer typically coincident with the 0 °C wet-bulb
height) and its bottom ``ml_thickness`` lower (500 m, radarx's own choice,
not a value of the paper); ``melting_layer=`` overrides this, e.g. with the
output of :func:`radarx.retrieve.melting_layer`.

Non-meteorological echo is not removed here: pass ``mask=`` (True where
gates hold meteorological echo, e.g. from a clutter or echo classification)
to leave the other gates unclassified (class 0).

Implementation
--------------
A compiled C++ kernel classifies all gates of all sweeps of a volume in one
multithreaded call: for every gate a single pass evaluates all memberships,
aggregates them, applies the restrictions and takes the argmax, without
intermediate arrays. An equivalent NumPy implementation is the fallback and
the test oracle.

Notes
-----
- Table A2 of Dolan et al. (2013) [3] gives a half-width of 21 °C for the big
  drops temperature function (centre 48 °C), which would rule out big drops
  below 27 °C and contradicts the range :math:`T > -3` °C of their Table A1
  (p. 2182). The half-width of 51 °C used for rain, which matches Table A1,
  is used instead (a deliberate deviation from the printed Table A2).
- Table 5 of Thompson et al. (2014) [4] repeats the reflectivity parameters
  in the wet snow :math:`Z_{DR}` row; the wet snow :math:`Z_{DR}` function
  (:math:`3 \\pm 5` dB, :math:`b = 10`) is read from their Fig. 6. The
  half-widths of the rain and freezing rain reflectivity functions are
  those of the ranges given in the table. Melting-layer heights are
  estimated for the whole volume rather than per 10° azimuth sector.
- The texture fields (SD(Z), SD(:math:`\\Phi_{DP}`)) and the
  non-meteorological classes (ground clutter, biological scatterers) of
  Park et al. (2009) [1] are not used: non-meteorological echo should be
  removed with ``mask=``. Their confidence vector is computed without the
  non-uniform beam filling and signal-to-noise terms (their Eqs. 9, 12, 14-17
  contain them) and for the four variables :math:`Z_H`, :math:`Z_{DR}`,
  :math:`\\rho_{hv}` and :math:`K_{DP}` only, their convective/stratiform
  separation is not applied, and the smoothing of :math:`Z`, :math:`Z_{DR}`
  and :math:`\\rho_{hv}` along the radial and the attenuation correction
  of their Sect. 2a are left to the caller. In Table 3 the two rules for
  ground clutter and biological scatterers are not used.
- Values that are radarx's own choices and not from the cited papers:
  ``ml_thickness=500`` m, ``beamwidth=1`` degree (the paper draws the
  :math:`\\pm 0.5` degree beam extent of the 3 dB beamwidth in its Fig. 2),
  and the 5 m histogram bins used for the median melting-layer height.

References
----------
.. [1] Park, H. S., A. V. Ryzhkov, D. S. Zrnić, and K.-E. Kim, 2009: The
   hydrometeor classification algorithm for the polarimetric WSR-88D:
   Description and application to an MCS. *Wea. Forecasting*, **24** (3),
   730-748, https://doi.org/10.1175/2008WAF2222205.1
.. [2] Dolan, B., and S. A. Rutledge, 2009: A theory-based hydrometeor
   identification algorithm for X-band polarimetric radars. *J. Atmos.
   Oceanic Technol.*, **26** (10), 2071-2088,
   https://doi.org/10.1175/2009JTECHA1208.1
.. [3] Dolan, B., S. A. Rutledge, S. Lim, V. Chandrasekar, and M. Thurai,
   2013: A robust C-band hydrometeor identification algorithm and
   application to a long-term polarimetric radar dataset. *J. Appl. Meteor.
   Climatol.*, **52** (9), 2162-2186,
   https://doi.org/10.1175/JAMC-D-12-0275.1
.. [4] Thompson, E. J., S. A. Rutledge, B. Dolan, V. Chandrasekar, and B. L.
   Cheong, 2014: A dual-polarization radar hydrometeor classification
   algorithm for winter precipitation. *J. Atmos. Oceanic Technol.*,
   **31** (7), 1457-1481, https://doi.org/10.1175/JTECH-D-13-00119.1

.. autosummary::
   :nosignatures:
   :toctree: generated/

   hid
   hid_classes
"""

from __future__ import annotations

__all__ = ["hid", "hid_classes"]

import numpy as np
import xarray as xr

from .._registry import accessor_method

try:
    from . import _hid

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _hid = None
    HAS_COMPILED_KERNEL = False

# variables of the membership table
_VARS = ("Z", "ZDR", "KDP", "RHOHV", "T")
_VIDX = {v: i for i, v in enumerate(_VARS)}
_NONE, _BETA, _TRAP = 0, 1, 2
_ADDITIVE, _HYBRID, _WINTER = 0, 1, 2
# reflectivity-dependent trapezoid corners, Park et al. (2009) Eqs. (4)-(5)
_FSEL = {None: 0, "f1": 1, "f2": 2, "f3": 3, "g1": 4, "g2": 5}

REFERENCES = {
    "park": "Park et al. (2009), https://doi.org/10.1175/2008WAF2222205.1",
    "dolan2009": "Dolan and Rutledge (2009), https://doi.org/10.1175/2009JTECHA1208.1",
    "dolan2013": "Dolan et al. (2013), https://doi.org/10.1175/JAMC-D-12-0275.1",
    "thompson": "Thompson et al. (2014), https://doi.org/10.1175/JTECH-D-13-00119.1",
}


def _zfunc(sel, z):
    """Park et al. (2009) Eqs. (4) and (5) (selector 0: zero)."""
    if sel == 1:
        return -0.50 + 2.50e-3 * z + 7.50e-4 * z * z
    if sel == 2:
        return 0.68 - 4.81e-2 * z + 2.92e-3 * z * z
    if sel == 3:
        return 1.42 + 6.67e-2 * z + 4.85e-4 * z * z
    if sel == 4:
        return -44.0 + 0.8 * z
    if sel == 5:
        return -22.0 + 0.5 * z
    return np.zeros_like(z)


# --------------------------------------------------------------------------
# Published membership functions
# --------------------------------------------------------------------------

# Park et al. (2009), Table 1 (p. 733; trapezoid corners x1-x4, a corner is a
# number or (offset, "f1".."g2")) and Table 2 (p. 734; weights of Z, ZDR, rhohv,
# LKdp). Only the rows of the eight meteorological classes and the four
# variables are used; the GC/AP and BS classes and the SD(Z), SD(PhiDP)
# columns are not. All numbers checked against the paper.
_P = "f1", "f2", "f3", "g1", "g2"
_PARK = [
    # abbr, name, Z, ZDR, RHOHV, LKdp, weights (Z, ZDR, RHOHV, LKdp)
    ("DS", "dry_snow", (5, 10, 35, 40), (-0.3, 0.0, 0.3, 0.6),
     (0.95, 0.98, 1.00, 1.01), (-30, -25, 10, 20), (1.0, 0.8, 0.6, 0.0)),
    ("WS", "wet_snow", (25, 30, 40, 50), (0.5, 1.0, 2.0, 3.0),
     (0.88, 0.92, 0.95, 0.985), (-30, -25, 10, 20), (0.6, 0.8, 1.0, 0.0)),
    ("CR", "ice_crystals", (0, 5, 20, 25), (0.1, 0.4, 3.0, 3.3),
     (0.95, 0.98, 1.00, 1.01), (-5, 0, 10, 15), (1.0, 0.6, 0.4, 0.5)),
    ("GR", "graupel", (25, 35, 50, 55), (-0.3, 0.0, (0, "f1"), (0.3, "f1")),
     (0.90, 0.97, 1.00, 1.01), (-30, -25, 10, 20), (0.8, 1.0, 0.4, 0.0)),
    ("BD", "big_drops", (20, 25, 45, 50),
     ((-0.3, "f2"), (0, "f2"), (0, "f3"), (1.0, "f3")),
     (0.92, 0.95, 1.00, 1.01), ((-1, "g1"), (0, "g1"), (0, "g2"), (1, "g2")),
     (0.8, 1.0, 0.6, 0.0)),
    ("RA", "light_and_moderate_rain", (5, 10, 45, 50),
     ((-0.3, "f1"), (0, "f1"), (0, "f2"), (0.5, "f2")),
     (0.95, 0.97, 1.00, 1.01), ((-1, "g1"), (0, "g1"), (0, "g2"), (1, "g2")),
     (1.0, 0.8, 0.6, 0.0)),
    ("HR", "heavy_rain", (40, 45, 55, 60),
     ((-0.3, "f1"), (0, "f1"), (0, "f2"), (0.5, "f2")),
     (0.92, 0.95, 1.00, 1.01), ((-1, "g1"), (0, "g1"), (0, "g2"), (1, "g2")),
     (1.0, 0.8, 0.6, 1.0)),
    ("RH", "rain_hail_mixture", (45, 50, 75, 80),
     (-0.3, 0.0, (0, "f1"), (0.5, "f1")),
     (0.85, 0.90, 1.00, 1.01), (-10, -4, (0, "g1"), (1, "g1")),
     (1.0, 0.8, 0.6, 1.0)),
]  # fmt: skip
# Park et al. (2009), Table 3 (p. 737): hard thresholds (class, variable, ">"
# or "<", threshold). The GC/AP (V > 1 m/s) and BS (rhohv > 0.97) rules are not
# used; the paper prints the GR rule as "<10 Z or >60 dBZ" and it is read as
# 10 dBZ.
_PARK_RULES = [
    ("DS", "ZDR", ">", 2.0),
    ("WS", "Z", "<", 20.0),
    ("WS", "ZDR", "<", 0.0),
    ("CR", "Z", ">", 40.0),
    ("GR", "Z", "<", 10.0),
    ("GR", "Z", ">", 60.0),
    ("BD", "ZDR", "<", (-0.3, "f2")),
    ("RA", "Z", ">", 50.0),
    ("HR", "Z", "<", 30.0),
    ("RH", "Z", "<", 40.0),
]
# Park et al. (2009), Eq. (24), p. 736 (GC/AP and BS left out): classes
# allowed by position of the beam relative to the
# melting layer (beam below / centre below bottom / centre in the layer /
# centre above top, lower edge below / beam above).
_PARK_ZONES = [
    ("BD", "RA", "HR", "RH"),
    ("WS", "GR", "BD", "RA", "HR", "RH"),
    ("DS", "WS", "GR", "BD", "RH"),
    ("DS", "WS", "CR", "GR", "BD", "RH"),
    ("DS", "CR", "GR", "RH"),
]

# Dolan et al. (2013), Table A2 (p. 2183): (m, a, b) of Z, ZDR, KDP, rhohv,
# T [degC] (every entry checked against the rendered table). Big drops T: half-width 51
# instead of the 21 printed (deviation, see module notes).
_DOLAN_C = [
    ("DZ", "drizzle", (1.75, 29, 10.0), (0.46, 0.46, 5.0), (0.03, 0.03, 2.0),
     (1.0, 0.018, 3.0), (40.0, 41.0, 50.0)),
    ("RN", "rain", (39, 19, 10.0), (2.3, 2.2, 9.0), (5.5, 5.5, 10.0),
     (1.0, 0.025, 3.0), (48.0, 51.0, 30.0)),
    ("CR", "ice_crystals", (-2.8, 22.1, 20.0), (2.9, 2.7, 10.0),
     (0.08, 0.08, 6.0), (0.98, 0.025, 3.0), (-50.0, 50.0, 25.0)),
    ("AG", "aggregates", (17.0, 18.1, 10.0), (1.0, 1.1, 7.0),
     (-0.008, 0.3, 1.0), (0.93, 0.07, 3.0), (-25.0, 26.0, 15.0)),
    ("WS", "wet_snow", (24.0, 21.3, 10.0), (1.3, 0.9, 10.0), (0.25, 0.43, 6.0),
     (0.74, 0.25, 10.0), (1.0, 3.5, 5.0)),
    ("HDG", "high_density_graupel", (44.3, 10.2, 6.0), (1.6, 1.2, 3.0),
     (1.9, 1.9, 3.0), (1.0, 0.04, 2.0), (-2.5, 20.0, 2.0)),
    ("LDG", "low_density_graupel", (37.0, 9.2, 0.8), (0.9, 0.9, 6.0),
     (0.1, 0.08, 3.0), (1.0, 0.025, 1.0), (-50.0, 50.0, 25.0)),
    ("HA", "hail", (62.3, 14.3, 10.0), (0.14, 0.56, 8.0), (0.6, 3.5, 6.0),
     (0.97, 0.1, 3.0), (0.0, 100.0, 5.0)),
    ("BD", "big_drops", (57.8, 8.5, 10.0), (4.4, 1.9, 8.0), (3.4, 3.3, 6.0),
     (0.99, 0.03, 3.0), (48.0, 51.0, 30.0)),
    ("VI", "vertically_aligned_ice", (-1.0, 25.0, 20.0), (-0.90, 0.9, 10.0),
     (-0.75, 0.75, 30.0), (0.975, 0.022, 3.0), (-50.0, 50.0, 25.0)),
]  # fmt: skip
# Dolan et al. (2013), Eq. (8), p. 2167: weights of ZDR, KDP and rhohv
# ("subjectively determined"). Used for ALL bands, including X and S band where
# Dolan and Rutledge (2009) used an additive sum with other weights (Zh 1.5,
# Kdp 1.0, T 0.5, Zdr 0.4, rhohv 0.2, Sect. 3b, p. 2081).
_DOLAN_WEIGHTS = {"ZDR": 0.8, "KDP": 1.0, "RHOHV": 0.1}

# Dolan and Rutledge (2009), Tables 3-9 (pp. 2078-2079): (min, max) of the
# X-band (XMBF) and S-band (SMBF) membership functions of Z, ZDR, KDP and
# rhohv (the "XMBF" and "SMBF" rows).
_DOLAN_RANGES = {
    "X": {
        "DZ": ((-27, 31), (0.0, 0.9), (0.0, 0.06), (0.985, 1.0)),
        "RN": ((25, 59), (0.1, 5.6), (0.0, 25.5), (0.98, 1.0)),
        "AG": ((-1.0, 33), (0.0, 1.4), (0.0, 0.4), (0.978, 1.0)),
        "CR": ((-25, 19), (0.6, 5.8), (0.0, 0.3), (0.97, 1.0)),
        "LDG": ((24, 44), (-0.7, 1.3), (-1.4, 2.8), (0.985, 1.0)),
        "HDG": ((32, 54), (-1.3, 3.7), (-2.5, 7.6), (0.965, 1.0)),
        "VI": ((-25, 32), (-2.1, 0.5), (-0.15, 0.0), (0.93, 1.0)),
    },
    "S": {
        "DZ": ((-27, 21), (0.0, 0.7), (0.0, 0.02), (0.99, 1.0)),
        "RN": ((26, 57), (0.1, 5.1), (0.0, 7.4), (0.98, 1.0)),
        "AG": ((0, 34), (0.0, 1.2), (0.0, 0.08), (0.978, 1.0)),
        "CR": ((-25, 19), (0.0, 5.8), (0.0, 0.09), (0.98, 1.0)),
        "LDG": ((25, 45), (-0.5, 1.1), (-0.4, 0.8), (0.99, 1.0)),
        "HDG": ((32, 58), (-0.9, 2.9), (-0.6, 1.7), (0.975, 1.0)),
        "VI": ((-26, 32), (-2.1, 0.5), (-0.04, 0.0), (0.93, 1.0)),
    },
}
_DOLAN_2009_CLASSES = ("DZ", "RN", "AG", "CR", "LDG", "HDG", "VI")

# Thompson et al. (2014), Table 5 (p. 1470; every entry checked against the
# rendered table, except the printed wet snow ZDR row, see the module notes):
# per class the weight, b, m, a of each
# variable (KDP m, a per band); group 0: melting-layer detection, 1: below,
# 2: above the melting layer.
_THOMPSON = [
    # abbr, name, group, {var: (w, b, m, a) or (w, b, {band: (m, a)})}
    ("PL", "plates", 2, {
        "Z": (0.20, 10, 12, 13), "ZDR": (0.36, 10, 5.5, 3.7),
        "KDP": (0.44, 5, {"X": (0.46, 0.45), "C": (0.27, 0.26),
                          "S": (0.13, 0.13)})}),
    ("DN", "dendrites", 2, {
        "Z": (0.20, 5, 17, 14), "ZDR": (0.36, 10, 2.6, 1.3),
        "KDP": (0.44, 5, {"X": (1.32, 1.28), "C": (1.0, 0.7),
                          "S": (0.31, 0.3)})}),
    ("IC", "ice_crystals", 2, {
        "Z": (0.48, 5, 6, 11), "ZDR": (0.24, 20, 0.0, 1.0),
        "KDP": (0.28, 5, {"X": (0.0, 0.55), "C": (0.0, 0.325),
                          "S": (0.0, 0.2)})}),
    ("AG", "aggregates", 2, {
        "Z": (0.24, 10, 28, 12), "ZDR": (0.36, 20, 0.0, 1.0),
        "KDP": (0.40, 5, {"X": (0.0, 0.55), "C": (0.0, 0.325),
                          "S": (0.0, 0.2)})}),
    ("WS", "wet_snow", 0, {
        "Z": (0.16, 10, 25, 20), "ZDR": (0.28, 10, 3.0, 5.0),
        "RHOHV": (0.56, 30, 0.75, 0.2)}),
    ("FZ", "freezing_or_frozen_rain", 1, {
        "Z": (0.33, 15, 11, 28), "T": (0.66, 20, -4, 3)}),
    ("RN", "rain", 1, {"Z": (0.33, 15, 19, 30), "T": (0.66, 40, 25, 25)}),
    ("OT", "other", 0, {
        "Z": (0.16, 5, 16, 17), "ZDR": (0.28, 15, 0.5, 1.5),
        "RHOHV": (0.56, 10, 0.96, 0.06)}),
]  # fmt: skip
# wet snow gates for the melting-layer statistics: 5-35 km, Thompson et al.
# (2014), p. 1466
_THOMPSON_STATS_RANGE = (5000.0, 35000.0)


def _empty_table(nc):
    return {
        "kind": np.zeros((nc, 5), dtype=np.int64),
        "par": np.zeros((nc, 5, 4)),
        "fsel": np.zeros((nc, 5, 4), dtype=np.int64),
        "weight": np.zeros((nc, 5)),
        "group": np.zeros(nc, dtype=np.int64),
    }


def _corner(c):
    if isinstance(c, tuple):
        return float(c[0]), _FSEL[c[1]]
    return float(c), 0


def _park_table():
    t = _empty_table(len(_PARK))
    for i, (_, _, *mfs, w) in enumerate(_PARK):
        for name, mf, wt in zip(
            ("Z", "ZDR", "RHOHV", "KDP"), mfs, (w[0], w[1], w[2], w[3])
        ):
            j = _VIDX[name]
            t["kind"][i, j] = _TRAP
            for k, c in enumerate(mf):
                t["par"][i, j, k], t["fsel"][i, j, k] = _corner(c)
            t["weight"][i, j] = wt
    abbr = [c[0] for c in _PARK]
    rules = [
        (abbr.index(c), _VIDX[v], 0 if op == ">" else 1, *_corner(thr))
        for c, v, op, thr in _PARK_RULES
    ]
    t["rules"] = rules
    t["zones"] = np.array(
        [sum(1 << abbr.index(c) for c in z) for z in _PARK_ZONES], dtype=np.int64
    )
    t.update(mode=_ADDITIVE, kdp_log=True, quality=True)
    return [(c[0], c[1]) for c in _PARK], t


def _beta_table(rows):
    """rows: (abbr, name, {var: (m, a, b)})"""
    t = _empty_table(len(rows))
    for i, (_, _, mfs) in enumerate(rows):
        for name, (m, a, b) in mfs.items():
            j = _VIDX[name]
            t["kind"][i, j] = _BETA
            t["par"][i, j, :3] = (m, a, b)
            t["weight"][i, j] = _DOLAN_WEIGHTS.get(name, 1.0)
    t.update(rules=[], zones=None, mode=_HYBRID, kdp_log=False, quality=False)
    return [(r[0], r[1]) for r in rows], t


def _dolan_table(band):
    if band == "C":
        rows = [
            (a, n, dict(zip(_VARS, (z, zdr, kdp, rho, temp))))
            for a, n, z, zdr, kdp, rho, temp in _DOLAN_C
        ]
        return _beta_table(rows)
    c13 = {r[0]: r for r in _DOLAN_C}
    rows = []
    for abbr in _DOLAN_2009_CLASSES:
        ref = c13[abbr]
        mfs = {}
        for name, (lo, hi), slope in zip(
            ("Z", "ZDR", "KDP", "RHOHV"),
            _DOLAN_RANGES[band][abbr],
            (ref[2][2], ref[3][2], ref[4][2], ref[5][2]),
        ):
            mfs[name] = (0.5 * (lo + hi), 0.5 * (hi - lo), slope)
        mfs["T"] = ref[6]
        rows.append((abbr, ref[1], mfs))
    return _beta_table(rows)


def _thompson_table(band):
    t = _empty_table(len(_THOMPSON))
    for i, (_, _, group, mfs) in enumerate(_THOMPSON):
        t["group"][i] = group
        for name, spec in mfs.items():
            j = _VIDX[name]
            w, b = spec[0], spec[1]
            m, a = spec[2][band] if isinstance(spec[2], dict) else spec[2:4]
            t["kind"][i, j] = _BETA
            t["par"][i, j, :3] = (m, a, b)
            t["weight"][i, j] = w
    abbr = [c[0] for c in _THOMPSON]
    t.update(
        rules=[],
        zones=None,
        mode=_WINTER,
        kdp_log=False,
        quality=False,
        ws=abbr.index("WS"),
        ot=abbr.index("OT"),
    )
    return [(c[0], c[1]) for c in _THOMPSON], t


_METHODS = ("park", "dolan", "thompson")
_BANDS = ("S", "C", "X")


def _scheme(method, band):
    band = str(band).upper()
    if band not in _BANDS:
        raise ValueError(f"band must be one of {_BANDS}, not {band!r}")
    if method == "auto":
        method = "park" if band == "S" else "dolan"
    if method not in _METHODS:
        raise ValueError(f"method must be 'auto' or one of {_METHODS}, not {method!r}")
    if method == "park":
        if band != "S":
            raise ValueError("the Park et al. (2009) method is for S band only")
        classes, table = _park_table()
        ref = REFERENCES["park"]
    elif method == "dolan":
        classes, table = _dolan_table(band)
        ref = REFERENCES["dolan2013"]
        if band != "C":
            ref = REFERENCES["dolan2009"] + "; " + ref
    else:
        classes, table = _thompson_table(band)
        ref = REFERENCES["thompson"]
    table["method"], table["band"], table["references"] = method, band, ref
    return classes, table


def hid_classes(method="auto", band="S"):
    """
    Classes of a hydrometeor classification method.

    Parameters
    ----------
    method : {"auto", "park", "dolan", "thompson"}, optional
        Classification method, see :mod:`radarx.retrieve.hid`.
    band : {"S", "C", "X"}, optional
        Radar band.

    Returns
    -------
    list of (int, str, str)
        Class code (as in the ``HID`` output; 0 is unclassified),
        abbreviation and name.

    References
    ----------
    The classes are those of Park et al. (2009) [1] for ``"park"`` (Table 1,
    p. 733), of Dolan et al. (2013) [3] for ``"dolan"`` at C band (Table A2,
    p. 2183) and of Dolan and Rutledge (2009) [2] for ``"dolan"`` at X and S
    band (Tables 3-9, pp. 2078-2079), and of Thompson et al. (2014) [4] for
    ``"thompson"`` (Table 5, p. 1470).

    .. [1] Park, H. S., A. V. Ryzhkov, D. S. Zrnić, and K.-E. Kim, 2009: The
       hydrometeor classification algorithm for the polarimetric WSR-88D:
       Description and application to an MCS. *Wea. Forecasting*, **24** (3),
       730-748, https://doi.org/10.1175/2008WAF2222205.1
    .. [2] Dolan, B., and S. A. Rutledge, 2009: A theory-based hydrometeor
       identification algorithm for X-band polarimetric radars. *J. Atmos.
       Oceanic Technol.*, **26** (10), 2071-2088,
       https://doi.org/10.1175/2009JTECHA1208.1
    .. [3] Dolan, B., S. A. Rutledge, S. Lim, V. Chandrasekar, and M. Thurai,
       2013: A robust C-band hydrometeor identification algorithm and
       application to a long-term polarimetric radar dataset. *J. Appl.
       Meteor. Climatol.*, **52** (9), 2162-2186,
       https://doi.org/10.1175/JAMC-D-12-0275.1
    .. [4] Thompson, E. J., S. A. Rutledge, B. Dolan, V. Chandrasekar, and B.
       L. Cheong, 2014: A dual-polarization radar hydrometeor classification
       algorithm for winter precipitation. *J. Atmos. Oceanic Technol.*,
       **31** (7), 1457-1481, https://doi.org/10.1175/JTECH-D-13-00119.1

    Examples
    --------
    >>> from radarx.retrieve import hid_classes
    >>> hid_classes("dolan", "C")[1]
    (2, 'RN', 'rain')
    """
    classes, _ = _scheme(method, band)
    return [(i + 1, a, n) for i, (a, n) in enumerate(classes)]


# --------------------------------------------------------------------------
# NumPy reference implementation (same steps and order as the C++ kernel)
# --------------------------------------------------------------------------


def _membership(t, c, v, x, z):
    p = t["par"][c, v]
    if t["kind"][c, v] == _BETA:
        u = (x - p[0]) / p[1]
        return 1.0 / (1.0 + np.power(u * u, p[2]))
    s = t["fsel"][c, v]
    x1, x2, x3, x4 = (p[k] + _zfunc(s[k], z) for k in range(4))
    with np.errstate(invalid="ignore", divide="ignore"):
        rise = (x - x1) / (x2 - x1)
        fall = (x4 - x) / (x4 - x3)
    out = np.where(x < x2, rise, np.where(x <= x3, 1.0, fall))
    return np.where((x < x1) | (x > x4), 0.0, out)


def _score_numpy(t, c, x, q):
    z = x[0]
    if t["mode"] == _HYBRID:
        pz = np.ones_like(z)
        pt = np.ones_like(z)
        num = np.zeros_like(z)
        den = np.zeros_like(z)
        for v in np.flatnonzero(t["kind"][c] != _NONE):
            ok = ~np.isnan(x[v])
            pv = np.where(ok, _membership(t, c, v, x[v], z), np.nan)
            if v == 0:
                pz = np.where(ok, pv, 1.0)
            elif v == 4:
                pt = np.where(ok, pv, 1.0)
            else:
                w = t["weight"][c, v]
                num = num + np.where(ok, w * pv, 0.0)
                den = den + np.where(ok, w, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            pol = np.where(den > 0, num / den, 1.0)
        return pt * pz * pol
    num = np.zeros_like(z)
    den = np.zeros_like(z)
    for v in range(5):
        w = t["weight"][c, v]
        if t["kind"][c, v] == _NONE or w <= 0:
            continue
        ok = ~np.isnan(x[v])
        wq = w * q[v]
        pv = _membership(t, c, v, x[v], z)
        num = num + np.where(ok, wq * pv, 0.0)
        den = den + np.where(ok, wq, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, 0.0)


def _confidence_numpy(phidp, rho, block):
    """Park et al. (2009) confidence vector Q for (Z, ZDR, rhohv, KDP, T).

    Q_Z = exp(-0.69 [(PhiDP / 250)^2 + (a / 50)^2]) (Eq. 14), Q_ZDR adds
    ((1 - rhohv) / 0.2)^2 (Eq. 15), Q_rhohv and Q_KDP keep that term
    (Eqs. 16 and 17), with the thresholds of Eq. 25 and the rhohv term
    switched off below 0.8 (Eq. 23). The non-uniform beam filling and SNR
    terms of the paper are not included; Q of the temperature is 1.
    """
    fphi = np.where(np.isnan(phidp), 0.0, (phidp / 250.0) ** 2)
    fblk = np.where(np.isnan(block), 0.0, (block / 50.0) ** 2)
    with np.errstate(invalid="ignore"):
        chi = np.where(~np.isnan(rho) & (rho >= 0.8), ((1.0 - rho) / 0.2) ** 2, 0.0)
    return [
        np.exp(-0.69 * (fphi + fblk)),
        np.exp(-0.69 * (fphi + chi + fblk)),
        np.exp(-0.69 * chi),
        np.exp(-0.69 * chi),
        np.ones_like(fphi),
    ]


def _load_numpy(t, b):
    """Gate variables (flattened), quality vector and the classified gates."""
    n = b["zh"].size
    nan = np.full(n, np.nan)

    def flat(key):
        return nan if b.get(key) is None else np.asarray(b[key], float).ravel()

    x = [flat("zh"), flat("zdr"), flat("kdp"), flat("rhohv"), flat("temperature")]
    ok = ~np.isnan(x[0])
    if b.get("valid") is not None:
        ok &= np.asarray(b["valid"]).ravel().astype(bool)
    if t["kdp_log"]:
        k = x[2]
        with np.errstate(invalid="ignore", divide="ignore"):
            x[2] = np.where(
                np.isnan(k), np.nan, np.where(k > 1e-3, 10.0 * np.log10(k), -30.0)
            )
    if t["quality"]:
        q = _confidence_numpy(flat("phidp"), x[3], flat("blockage"))
    else:
        q = [np.ones(n)] * 5
    return x, q, ok


def _ml_zone_numpy(h, half, bottom, top):
    zone = np.full(h.shape, 4)
    zone = np.where(h - half < top, 3, zone)
    zone = np.where(h < top, 2, zone)
    zone = np.where(h < bottom, 1, zone)
    return np.where(h + half < bottom, 0, zone)


def _detect_melting_numpy(blocks, prepared, t, opts):
    """Winter pass 1: wet snow gates and their median height (as the kernel)."""
    lo, width, nbin = opts["hist_lo"], opts["hist_bin"], opts["hist_n"]
    counts = np.zeros(nbin, dtype=np.int64)
    for b, (_, _, ok, s) in zip(blocks, prepared):
        ws = ok & (s[t["ws"]] > s[t["ot"]])
        h = np.asarray(b["height"], float).ravel()
        use = ws & ~np.isnan(h)
        if b.get("range") is not None:
            r = np.broadcast_to(b["range"], b["zh"].shape).ravel()
            use &= (r >= opts["stats_rmin"]) & (r <= opts["stats_rmax"])
        k = np.floor((h[use] - lo) / width).astype(np.int64)
        counts += np.bincount(np.clip(k, 0, nbin - 1), minlength=nbin)
    n_ws = int(counts.sum())
    state = 0
    if n_ws >= opts["ml_gates_complete"]:
        state = 2
    elif n_ws >= opts["ml_gates_partial"]:
        state = 1
    ml_height = np.nan
    if n_ws:
        k = int(np.searchsorted(np.cumsum(counts), (n_ws - 1) // 2 + 1))
        ml_height = lo + (k + 0.5) * width
    return {"n_wet_snow": n_ws, "melting_layer_height": ml_height, "melting": state}


def _allowed_winter(b, t, ok, s, info):
    nc = s.shape[0]
    ws = ok & (s[t["ws"]] > s[t["ot"]])
    h = np.asarray(b["height"], float).ravel()
    below = np.zeros(h.size, dtype=bool)
    if info["melting"] == 2:
        with np.errstate(invalid="ignore"):
            below = h < info["melting_layer_height"]
    allowed = t["group"][:, None] == np.where(below, 1, 2)[None, :]
    if info["melting"] > 0:
        allowed = np.where(ws[None, :], np.arange(nc)[:, None] == t["ws"], allowed)
    return allowed


def _allowed_zones(b, t, nc, opts):
    """Park et al. (2009) classes by beam position, or None (no restriction)."""
    if t.get("zones") is None or any(
        b.get(k) is None for k in ("ml_bottom", "ml_top", "height")
    ):
        return None
    nrow, ncol = b["zh"].shape
    bottom = np.repeat(np.asarray(b["ml_bottom"], float), ncol)
    top = np.repeat(np.asarray(b["ml_top"], float), ncol)
    h = np.asarray(b["height"], float).ravel()
    half = np.zeros(h.size)
    if b.get("range") is not None:
        half = np.tile(np.asarray(b["range"], float) * opts["sin_half_beam"], nrow)
    use = ~np.isnan(bottom) & ~np.isnan(top) & ~np.isnan(h)
    bits = t["zones"][np.where(use, _ml_zone_numpy(h, half, bottom, top), 0)]
    zallowed = ((bits[None, :] >> np.arange(nc)[:, None]) & 1).astype(bool)
    return np.where(use[None, :], zallowed, True)


def _apply_rules(allowed, t, x):
    """Park et al. (2009) Table 3: drop classes failing a hard threshold."""
    for c, v, op, thr, fsel in t["rules"]:
        val = x[v]
        limit = thr + _zfunc(fsel, x[0])
        with np.errstate(invalid="ignore"):
            bad = (val > limit) if op == 0 else (val < limit)
        allowed[c] &= ~(bad & ~np.isnan(val))
    return allowed


def _classify_numpy(blocks, t, opts):
    """NumPy implementation of the compiled kernel (same results)."""
    nc = t["kind"].shape[0]
    prepared = []
    for b in blocks:
        x, q, ok = _load_numpy(t, b)
        s = np.stack([_score_numpy(t, c, x, q) for c in range(nc)])
        prepared.append((x, q, ok, s))
    info = {"n_wet_snow": 0, "melting_layer_height": np.nan, "melting": 0}
    if t["mode"] == _WINTER:
        info = _detect_melting_numpy(blocks, prepared, t, opts)

    out = []
    for b, (x, _, ok, s) in zip(blocks, prepared):
        if t["mode"] == _WINTER:
            allowed = _allowed_winter(b, t, ok, s, info)
        else:
            allowed = _allowed_zones(b, t, nc, opts)
            if allowed is None:
                allowed = np.ones(s.shape, dtype=bool)
        allowed = _apply_rules(allowed, t, x)
        masked = np.where(allowed, s, -1.0)
        best = np.argmax(masked, axis=0)
        best_s = np.take_along_axis(masked, best[None], 0)[0]
        found = ok & (best_s > -1.0)
        shape = b["zh"].shape
        cls = np.where(found, best + 1, 0).astype(np.int8).reshape(shape)
        conf = np.where(found, best_s, np.nan).astype(np.float32).reshape(shape)
        scores = None
        if opts["want_scores"]:
            scores = np.where(ok[None], s, np.nan).astype(np.float32)
            scores = scores.reshape((nc,) + shape)
        out.append((cls, conf, scores))
    return out, info


def _classify_compiled(blocks, t, opts, n_threads):
    rules = t["rules"]
    arr = np.asarray

    def col(key):
        return [b.get(key) for b in blocks]

    cls, conf, scores, info = _hid.classify(
        [np.ascontiguousarray(b["zh"], dtype=np.float64) for b in blocks],
        col("zdr"),
        col("kdp"),
        col("rhohv"),
        col("temperature"),
        col("phidp"),
        col("blockage"),
        col("valid"),
        col("height"),
        col("range"),
        col("ml_bottom"),
        col("ml_top"),
        t["kind"],
        t["par"],
        t["fsel"],
        t["weight"],
        t["group"],
        arr([r[0] for r in rules], dtype=np.int64),
        arr([r[1] for r in rules], dtype=np.int64),
        arr([r[2] for r in rules], dtype=np.int64),
        arr([r[4] for r in rules], dtype=np.int64),
        arr([r[3] for r in rules], dtype=np.float64),
        t.get("zones"),
        t["mode"],
        t["kdp_log"],
        t["quality"],
        opts["sin_half_beam"],
        t.get("ws", -1),
        t.get("ot", -1),
        opts["ml_gates_partial"],
        opts["ml_gates_complete"],
        opts["stats_rmin"],
        opts["stats_rmax"],
        opts["hist_lo"],
        opts["hist_bin"],
        opts["hist_n"],
        opts["want_scores"],
        int(n_threads or 0),
    )
    return list(zip(cls, conf, scores)), dict(info)


# --------------------------------------------------------------------------
# xarray layer
# --------------------------------------------------------------------------

_NAMES = {
    "dbzh": ("DBZH", "DBZ", "reflectivity", "corrected_reflectivity"),
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
    "rhohv": ("RHOHV", "cross_correlation_ratio", "copol_correlation_coeff"),
    "phidp": ("PHIDP_processed", "corrected_differential_phase"),
}
_KELVIN = ("K", "kelvin", "Kelvin", "degK")


def _use_compiled(engine):
    """Whether to run the compiled kernel for the requested ``engine``."""
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled HID kernel is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


def _find(ds, name, key, required=False):
    if name is not None:
        if name not in ds:
            raise KeyError(f"{name!r} is not in the dataset")
        return name
    for cand in _NAMES[key]:
        if cand in ds:
            return cand
    if required:
        raise KeyError(f"none of {_NAMES[key]} found; pass the {key} field name")
    return None


def _is_profile(obj):
    return isinstance(obj, (xr.Dataset, xr.DataArray)) and "height" in obj.dims


def _to_celsius(values, units):
    values = np.asarray(values, dtype=np.float64)
    if units in _KELVIN:
        return values - 273.15
    if units is None and np.isfinite(values).any() and np.nanmedian(values) > 100.0:
        return values - 273.15
    return values


def _profile_dataset(profile):
    if isinstance(profile, xr.DataArray):
        return profile.to_dataset(name="temperature")
    if "temperature" not in profile:
        raise KeyError("the temperature profile needs a 'temperature' variable")
    return profile


def _on(arr, da):
    """``arr`` on the dimensions (and order) of ``da``."""
    if set(arr.dims) == set(da.dims) and arr.shape and arr.sizes == da.sizes:
        return arr.transpose(*da.dims)
    return arr.broadcast_like(da).transpose(*da.dims)


def _heights(ds, da):
    """Gate heights above sea level (m) on the dims of ``da``, or None."""
    if "z" in ds.variables:
        return _on(ds["z"], da)
    if "height" in ds.coords:
        return _on(ds["height"], da)
    if "range" in da.dims and "elevation" in ds.variables:
        from xradar.georeference import antenna_to_cartesian

        elev = ds["elevation"].broadcast_like(da)
        rng = ds["range"].broadcast_like(da)
        alt = float(ds["altitude"].values) if "altitude" in ds.variables else 0.0
        _, _, z = antenna_to_cartesian(
            rng.values.astype(np.float64), 0.0, elev.values, site_altitude=alt
        )
        return xr.DataArray(np.asarray(z), dims=elev.dims).transpose(*da.dims)
    return None


def _field_values(ds, spec, da, what):
    """A gate field from a name or a DataArray, on the dims of ``da``."""
    if spec is None:
        return None
    if isinstance(spec, str):
        if spec not in ds:
            raise KeyError(f"{what} field {spec!r} is not in the dataset")
        spec = ds[spec]
    if not isinstance(spec, xr.DataArray):
        raise TypeError(f"{what} must be a field name or a DataArray")
    return _on(spec, da)


def _flat2d(values):
    """(rows, columns) view with the last dimension as columns."""
    values = np.asarray(values)
    if values.ndim == 1:
        return values.reshape(1, -1)
    return values.reshape(-1, values.shape[-1])


def _c2d(values, dtype=np.float64):
    """Contiguous (rows, columns) array."""
    return np.ascontiguousarray(_flat2d(values), dtype=dtype)


def _temperature_values(ds, da, temperature, height, opts):
    """Temperature (degC) on the gates of ``da``."""
    if _is_profile(temperature):
        from ..io.sounding import interpolate_profile

        prof = _profile_dataset(temperature)
        env = interpolate_profile(
            prof, height, ["temperature"], n_threads=opts["n_threads"]
        )["temperature"]
        return _to_celsius(env.values, prof["temperature"].attrs.get("units"))
    tda = _field_values(ds, temperature, da, "temperature")
    return _to_celsius(tda.values, tda.attrs.get("units"))


def _prepare(ds, fields, opts):
    """Arrays of one sweep or grid for the kernel, and wrapping info."""
    dbzh, zdr, kdp, rhohv, phidp = fields
    zname = _find(ds, dbzh, "dbzh", True)
    da = ds[zname]
    if "range" in da.dims:
        da = da.transpose(..., "range")
    names = {
        "zdr": _find(ds, zdr, "zdr"),
        "kdp": _find(ds, kdp, "kdp"),
        "rhohv": _find(ds, rhohv, "rhohv"),
        "phidp": _find(ds, phidp, "phidp"),
    }
    block = {"zh": _c2d(da.values)}
    for key, name in names.items():
        block[key] = None if name is None else _c2d(_on(ds[name], da).values)

    temperature = opts["temperature"]
    height = None
    if opts["need_height"] or _is_profile(temperature):
        height = _heights(ds, da)
        if height is None:
            raise ValueError(
                "gate heights are needed: the dataset has no 'z', 'height' or "
                "'elevation' and 'range'"
            )
        block["height"] = _c2d(height.values)
    if temperature is not None:
        block["temperature"] = _c2d(
            _temperature_values(ds, da, temperature, height, opts)
        )
    mask = _field_values(ds, opts["mask"], da, "mask")
    if mask is not None:
        block["valid"] = _c2d(np.asarray(mask.values).astype(bool), np.uint8)
    blk = _field_values(ds, opts["blockage"], da, "blockage")
    if blk is not None:
        block["blockage"] = _c2d(blk.values)
    if "range" in da.dims:
        block["range"] = np.ascontiguousarray(ds["range"].values, dtype=np.float64)
    if opts["ml"] is not None:
        nrow = block["zh"].shape[0]
        block["ml_bottom"] = np.full(nrow, opts["ml"][0])
        block["ml_top"] = np.full(nrow, opts["ml"][1])
    used = [zname] + [n for n in names.values() if n]
    return block, {"da": da, "fields": used}


def _zero_height_profile(temperature, opts):
    """Wet-bulb 0 degC height of a profile (0 degC without humidity), or NaN."""
    from ..io.sounding import isotherm_height, wet_bulb_zero_height

    prof = _profile_dataset(temperature)
    units = prof["temperature"].attrs.get("units")
    vals = np.asarray(prof["temperature"].values, dtype=np.float64)
    kelvin = units in _KELVIN or (units is None and np.nanmedian(vals) > 100)
    top = np.array([np.nan])
    if kelvin and "dewpoint" in prof and "pressure" in prof:
        top = wet_bulb_zero_height(prof, n_threads=opts["n_threads"]).values
    if not np.isfinite(top).any():
        top = isotherm_height(
            prof, 273.15 if kelvin else 0.0, n_threads=opts["n_threads"]
        ).values
    return float(np.nanmedian(top)) if np.isfinite(top).any() else np.nan


def _zero_height_gates(temperature, datasets, opts):
    """Median height of the gates within 0.5 K of 0 degC, or NaN."""
    hs = []
    for ds in datasets:
        da = ds[_find(ds, opts["dbzh"], "dbzh", True)]
        h = _heights(ds, da)
        if h is None:
            continue
        t = _field_values(ds, temperature, da, "temperature")
        tc = _to_celsius(t.values, t.attrs.get("units"))
        hs.append(np.asarray(h.values)[np.abs(tc) <= 0.5])
    hs = np.concatenate(hs) if hs else np.array([])
    return float(np.median(hs)) if hs.size else np.nan


def _melting_layer_heights(melting_layer, temperature, ml_thickness, datasets, opts):
    """(bottom, top) of the melting layer in m above sea level, or None."""
    if melting_layer is not None:
        if isinstance(melting_layer, xr.Dataset):
            bottom = float(np.nanmedian(melting_layer["melting_layer_bottom"].values))
            top = float(np.nanmedian(melting_layer["melting_layer_top"].values))
        else:
            bottom, top = (float(v) for v in melting_layer)
        if not bottom <= top:
            raise ValueError("the melting layer bottom must not be above its top")
        return bottom, top
    if temperature is None:
        return None
    if _is_profile(temperature):
        top = _zero_height_profile(temperature, opts)
    else:
        top = _zero_height_gates(temperature, datasets, opts)
    if not np.isfinite(top):
        return None
    return top - float(ml_thickness), top


def _wrap(prep, cls, conf, scores, classes, table, info):
    da = prep["da"]
    shape = da.shape
    coords = {k: v for k, v in da.coords.items()}
    abbr = [c[0] for c in classes]
    meanings = " ".join(c[1] for c in classes)
    method = table["method"]
    comment = (
        f"fuzzy-logic hydrometeor classification, method={method!r}, "
        f"band={table['band']!r}, from {', '.join(prep['fields'])}"
    )
    if table["mode"] == _WINTER:
        state = ("no melting", "partial melting", "complete melting")[info["melting"]]
        comment += (
            f"; {state} detected ({info['n_wet_snow']} wet snow gates, "
            f"median height {info['melting_layer_height']:.0f} m)"
        )
    hid_attrs = {
        "long_name": "Hydrometeor classification",
        "flag_values": np.arange(1, len(classes) + 1, dtype=np.int8),
        "flag_meanings": meanings,
        "classes": " ".join(abbr),
        "method": method,
        "band": table["band"],
        "comment": comment + "; 0: not classified",
        "references": table["references"],
        "_FillValue": np.int8(0),
    }
    out = xr.Dataset(
        {
            "HID": (da.dims, cls.reshape(shape), hid_attrs),
            "HID_confidence": (
                da.dims,
                conf.reshape(shape),
                {
                    "long_name": "Score of the assigned hydrometeor class",
                    "units": "1",
                    "comment": "aggregated membership of the assigned class (0-1)",
                },
            ),
        },
        coords=coords,
    )
    if scores is not None:
        out["HID_scores"] = xr.DataArray(
            scores.reshape((len(classes),) + shape),
            dims=("hid_class",) + da.dims,
            attrs={
                "long_name": "Aggregated membership score of each hydrometeor class",
                "units": "1",
            },
        )
        out = out.assign_coords(
            hid_class=("hid_class", abbr, {"long_name": "hydrometeor class"})
        )
    if table["mode"] == _WINTER:
        out["melting_layer_height"] = xr.DataArray(
            info["melting_layer_height"],
            attrs={
                "long_name": "median height of wet snow gates (melting layer)",
                "units": "m",
            },
        )
    return out


def _run(datasets, fields, classes, table, opts, n_threads, use_compiled):
    if table["mode"] == _HYBRID or table["mode"] == _ADDITIVE:
        opts["need_height"] = opts["ml"] is not None
    preps = [_prepare(ds, fields, opts) for ds in datasets]
    blocks = [p[0] for p in preps]
    if use_compiled:
        results, info = _classify_compiled(blocks, table, opts, n_threads)
    else:
        results, info = _classify_numpy(blocks, table, opts)
    return [
        _wrap(p[1], c, f, s, classes, table, info)
        for p, (c, f, s) in zip(preps, results)
    ]


def _check_inputs(obj, temperature, mask, blockage, ml_gates):
    """Validate the input combinations; returns the ``ml_gates`` thresholds."""
    lo, hi = (int(v) for v in ml_gates)
    if lo > hi or lo < 0:
        raise ValueError(
            "ml_gates must be (partial, complete) with partial <= complete"
        )
    if isinstance(temperature, xr.DataTree):
        raise TypeError("temperature must be a profile, a DataArray or a field name")
    if not isinstance(obj, xr.DataTree):
        return lo, hi
    if isinstance(temperature, xr.DataArray) and not _is_profile(temperature):
        raise TypeError(
            "for a volume, temperature must be a profile on 'height' or a field name"
        )
    for name, spec in (("mask", mask), ("blockage", blockage)):
        if spec is not None and not isinstance(spec, str):
            raise TypeError(f"for a volume, {name} must be a field name")
    return lo, hi


def hid(
    obj,
    temperature=None,
    *,
    band="S",
    method="auto",
    dbzh=None,
    zdr=None,
    kdp=None,
    rhohv=None,
    phidp=None,
    mask=None,
    blockage=None,
    melting_layer=None,
    ml_thickness=500.0,
    beamwidth=1.0,
    ml_gates=(100, 10000),
    quality=True,
    scores=True,
    n_threads=None,
    engine="auto",
):
    """
    Classify hydrometeors with fuzzy logic for a sweep, a volume or a grid.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataTree
        A sweep (e.g. from xradar, on ``(azimuth, range)``), a grid or a
        QVP, or a volume with ``sweep_*`` groups. Volumes are classified in
        one kernel call; sweeps without reflectivity or differential
        reflectivity (e.g. NEXRAD Doppler cuts) are skipped.
    temperature : xarray.Dataset, xarray.DataArray or str, optional
        Temperature (K or °C, from the ``units`` attribute; values above 100
        without units are taken as K): a profile on ``height`` (m above sea
        level), e.g. from :func:`radarx.io.sounding.era5_profile`,
        :func:`radarx.io.sounding.read_sounding` or ``dtree.radarx.sounding()``,
        interpolated to the gate heights; a DataArray per gate (sweep or
        grid input); or the name of a temperature field in each sweep.
        Default None: no temperature.
    band : {"S", "C", "X"}, optional
        Radar band. Default ``"S"``.
    method : {"auto", "park", "dolan", "thompson"}, optional
        ``"auto"`` (default) uses ``"park"`` (Park et al. 2009 [1]) at S band
        and ``"dolan"`` at C and X band: the membership functions of Dolan et
        al. (2013) [3] at C band, and the variable ranges of Dolan and
        Rutledge (2009) [2] at X and S band, **both with the aggregation of
        Dolan et al. (2013)** and not the additive weighted sum of the 2009
        paper (see :mod:`radarx.retrieve.hid`). ``"thompson"`` is the winter
        classification of Thompson et al. (2014) [4].
    dbzh, zdr, kdp, rhohv, phidp : str, optional
        Field names of the reflectivity (dBZ), differential reflectivity
        (dB), specific differential phase (°/km), copolar correlation
        coefficient and the processed (unfolded, offset-free) differential
        phase (°, only for the attenuation term of the Park et al. 2009
        confidence vector). By default the first name found is used:
        ``DBZH``, ``DBZ``, ``reflectivity``, ``corrected_reflectivity``;
        ``ZDR``, ``differential_reflectivity``,
        ``corrected_differential_reflectivity``; ``KDP``,
        ``specific_differential_phase``,
        ``corrected_specific_differential_phase``; ``RHOHV``,
        ``cross_correlation_ratio``, ``copol_correlation_coeff``;
        ``PHIDP_processed``, ``corrected_differential_phase``. If there is no
        KDP field but a differential phase, KDP is estimated with
        :func:`radarx.retrieve.estimate_kdp` (default options). Missing
        variables are left out of the aggregation.
    mask : xarray.DataArray or str, optional
        True at gates to classify (meteorological echo); other gates get
        class 0. A DataArray (sweep or grid input) or the name of a boolean
        field in each sweep.
    blockage : xarray.DataArray or str, optional
        Partial beam blockage (percent) per gate, for the Park et al. (2009)
        confidence vector.
    melting_layer : (float, float) or xarray.Dataset, optional
        Bottom and top of the melting layer (m above sea level), or the
        output of :func:`radarx.retrieve.melting_layer` (median over time),
        used by the Park method. Default: from ``temperature``.
    ml_thickness : float, optional
        Park method: depth (m) of the melting layer below its top when it is
        derived from ``temperature``. Default 500 (radarx choice, not a value
        of Park et al. 2009 [1]).
    beamwidth : float, optional
        Half-power beam width (degrees) for the beam extent relative to the
        melting layer (Park method, Fig. 2 and Eq. 24 of [1]). Default 1
        (radarx choice, about the WSR-88D beam width).
    ml_gates : (int, int), optional
        Thompson method: number of wet snow gates for partial and complete
        melting. Default ``(100, 10000)``, the values of Thompson et al.
        (2014) [4] (p. 1466); they depend on the radar resolution.
    quality : bool, optional
        Park method: apply the confidence vector (Eqs. 14-17 of [1], without
        the beam filling and SNR terms). Default True.
    scores : bool, optional
        Also return the scores of all classes (``HID_scores``; one float32
        value per class and gate). Default True.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled
        kernel and falls back to NumPy.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        For a sweep or grid, a Dataset on the input coordinates with ``HID``
        (int8 class code; ``flag_values`` and ``flag_meanings`` attributes;
        0 where not classified), ``HID_confidence`` (score of the assigned
        class, 0-1) and, with ``scores=True``, ``HID_scores`` on
        ``(hid_class, ...)``. The Thompson method adds the scalar
        ``melting_layer_height``. For a volume, a DataTree with one such node
        per classified sweep and the root of the input.

    Raises
    ------
    KeyError
        If a requested field is missing.
    ValueError
        For unknown options, or if gate heights are needed but cannot be
        found.
    ImportError
        If ``engine="compiled"`` and the compiled kernel is not available.

    Notes
    -----
    The ``"dolan"`` method at X and S band uses the variable ranges of Dolan
    and Rutledge (2009) [2] with the hybrid aggregation of Dolan et al. (2013)
    [3], not the 2009 additive aggregation; see :mod:`radarx.retrieve.hid`
    for the differences from every paper, the checked table and equation
    numbers, and the values that are radarx choices.

    References
    ----------
    .. [1] Park, H. S., A. V. Ryzhkov, D. S. Zrnić, and K.-E. Kim, 2009: The
       hydrometeor classification algorithm for the polarimetric WSR-88D:
       Description and application to an MCS. *Wea. Forecasting*, **24** (3),
       730-748, https://doi.org/10.1175/2008WAF2222205.1
    .. [2] Dolan, B., and S. A. Rutledge, 2009: A theory-based hydrometeor
       identification algorithm for X-band polarimetric radars. *J. Atmos.
       Oceanic Technol.*, **26** (10), 2071-2088,
       https://doi.org/10.1175/2009JTECHA1208.1
    .. [3] Dolan, B., S. A. Rutledge, S. Lim, V. Chandrasekar, and M. Thurai,
       2013: A robust C-band hydrometeor identification algorithm and
       application to a long-term polarimetric radar dataset. *J. Appl.
       Meteor. Climatol.*, **52** (9), 2162-2186,
       https://doi.org/10.1175/JAMC-D-12-0275.1
    .. [4] Thompson, E. J., S. A. Rutledge, B. Dolan, V. Chandrasekar, and B.
       L. Cheong, 2014: A dual-polarization radar hydrometeor classification
       algorithm for winter precipitation. *J. Atmos. Oceanic Technol.*,
       **31** (7), 1457-1481, https://doi.org/10.1175/JTECH-D-13-00119.1

    Examples
    --------
    >>> profile = dtree.radarx.sounding()  # doctest: +SKIP
    >>> out = radarx.retrieve.hid(dtree, profile, band="S")  # doctest: +SKIP
    >>> out = sweep.radarx.hid(profile, band="C")  # doctest: +SKIP
    """
    classes, table = _scheme(method, band)
    use_compiled = _use_compiled(engine)
    lo, hi = _check_inputs(obj, temperature, mask, blockage, ml_gates)
    opts = {
        "temperature": temperature,
        "mask": mask,
        "blockage": blockage,
        "dbzh": dbzh,
        "n_threads": n_threads,
        "need_height": table["mode"] == _WINTER,
        "sin_half_beam": float(np.sin(np.deg2rad(0.5 * float(beamwidth)))),
        "ml_gates_partial": lo,
        "ml_gates_complete": hi,
        "stats_rmin": _THOMPSON_STATS_RANGE[0],
        "stats_rmax": _THOMPSON_STATS_RANGE[1],
        "hist_lo": -1000.0,
        "hist_bin": 5.0,
        "hist_n": 6000,
        "want_scores": bool(scores),
    }
    table = dict(table, quality=table["quality"] and bool(quality))
    fields = (dbzh, zdr, kdp, rhohv, phidp)

    if isinstance(obj, xr.Dataset):
        datasets = [_with_kdp(obj, fields, use_compiled, n_threads)]
        names = None
    else:
        names = [
            name
            for name in obj.children
            if name.startswith("sweep")
            and _find(obj[name].to_dataset(), dbzh, "dbzh") is not None
            and _find(obj[name].to_dataset(), zdr, "zdr") is not None
        ]
        if not names:
            raise KeyError(
                "no sweep contains reflectivity and differential reflectivity"
            )
        datasets = _with_kdp_tree(obj, names, fields, use_compiled, n_threads)
    opts["ml"] = None
    if table["zones"] is not None:
        opts["ml"] = _melting_layer_heights(
            melting_layer, temperature, ml_thickness, datasets, opts
        )
    results = _run(datasets, fields, classes, table, opts, n_threads, use_compiled)
    if names is None:
        return results[0]
    nodes = {"/": obj.root.to_dataset(inherit=False)}
    nodes.update(zip(names, results))
    return xr.DataTree.from_dict(nodes)


def _has_phase(ds):
    from .kdp import _PHIDP_NAMES

    return any(n in ds for n in _PHIDP_NAMES)


def _kdp_options(fields):
    dbzh, _, _, rhohv, _ = fields
    return {"dbzh": dbzh, "rhohv": rhohv}


def _with_kdp(ds, fields, use_compiled, n_threads):
    """The sweep, with KDP (and processed PHIDP) estimated if missing."""
    if (
        _find(ds, fields[2], "kdp") is not None
        or not _has_phase(ds)
        or "range" not in ds.dims
    ):
        return ds
    from .kdp import estimate_kdp

    out = estimate_kdp(
        ds,
        **_kdp_options(fields),
        n_threads=n_threads,
        engine="auto" if use_compiled else "numpy",
    )
    return ds.assign(KDP=out["KDP"], PHIDP_processed=out["PHIDP_processed"])


def _with_kdp_tree(dtree, names, fields, use_compiled, n_threads):
    datasets = [dtree[n].to_dataset() for n in names]
    todo = [
        i
        for i, ds in enumerate(datasets)
        if _find(ds, fields[2], "kdp") is None and _has_phase(ds) and "range" in ds.dims
    ]
    if not todo:
        return datasets
    from .kdp import estimate_kdp

    sub = xr.DataTree.from_dict(
        {
            "/": dtree.root.to_dataset(inherit=False),
            **{names[i]: dtree[names[i]].to_dataset(inherit=False) for i in todo},
        }
    )
    out = estimate_kdp(
        sub,
        **_kdp_options(fields),
        n_threads=n_threads,
        engine="auto" if use_compiled else "numpy",
    )
    for i in todo:
        if names[i] in out.children:
            res = out[names[i]].to_dataset()
            datasets[i] = datasets[i].assign(
                KDP=res["KDP"], PHIDP_processed=res["PHIDP_processed"]
            )
    return datasets


@accessor_method("dataset", name="hid")
def _hid_dataset_accessor(self, temperature=None, *, band="S", **kwargs):
    """
    Classify hydrometeors with fuzzy logic for this sweep or grid.

    Parameters
    ----------
    temperature : xarray.Dataset, xarray.DataArray or str, optional
        Temperature profile on ``height`` (e.g. ``dtree.radarx.sounding()``),
        a temperature field or its name. Default: none.
    band : {"S", "C", "X"}, optional
        Radar band. Default ``"S"``.
    **kwargs
        Options of :func:`radarx.retrieve.hid`, e.g. ``method``, ``mask``
        or the field names.

    Returns
    -------
    xarray.Dataset
        ``HID`` (class code), ``HID_confidence`` and ``HID_scores``.

    See Also
    --------
    radarx.retrieve.hid
    """
    return hid(self.xarray_obj, temperature, band=band, **kwargs)


@accessor_method("datatree", name="hid")
def _hid_datatree_accessor(self, temperature=None, *, band="S", **kwargs):
    """
    Classify hydrometeors with fuzzy logic for every sweep.

    All gates of all sweeps are classified in one call of the compiled
    kernel; sweeps without reflectivity or differential reflectivity are
    skipped.

    Parameters
    ----------
    temperature : xarray.Dataset, xarray.DataArray or str, optional
        Temperature profile on ``height`` (e.g. ``dtree.radarx.sounding()``)
        or the name of a temperature field in each sweep. Default: none.
    band : {"S", "C", "X"}, optional
        Radar band. Default ``"S"``.
    **kwargs
        Options of :func:`radarx.retrieve.hid`, e.g. ``method``, ``mask``
        or the field names.

    Returns
    -------
    xarray.DataTree
        The root of the volume and one node per classified sweep with
        ``HID``, ``HID_confidence`` and ``HID_scores``.

    See Also
    --------
    radarx.retrieve.hid
    """
    return hid(self.xarray_obj, temperature, band=band, **kwargs)
