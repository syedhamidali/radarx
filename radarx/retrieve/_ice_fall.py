# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Terminal fall speeds of hail and melting snow, shared by
:mod:`radarx.retrieve.ice_fall` and the trajectory model.

Hail
    Power laws of the maximum dimension, from the corrigendum of Heymsfield et
    al. (2020, J. Atmos. Sci., 77, 405-412) to Heymsfield et al. (2018): the
    terminal velocity of hail and graupel in m s-1 against the maximum
    dimension :math:`D` in cm,

    ============================  =====================================  ====================
    law                           :math:`V_t`                            source (corrigendum)
    ============================  =====================================  ====================
    ``"heymsfield2020"``          :math:`7.6 D^{0.89}` for :math:`D<D_x`  Eqs. (2a), (2b),
                                  :math:`8.4 D^{0.67}` for :math:`D>D_x`  Table 2 row 4
    ``"graupel_hail_obs"``        :math:`7.2 D^{0.84}`                    Table 2 row 3
    ``"heymsfield2020_p10"`` ...  :math:`a D^b`, the 10th, 25th, 50th,    Table 3
    ``"..._p90"``                 75th and 90th percentile
    ============================  =====================================  ====================

    The two branches of the first law cross at :math:`D_x = (8.4/7.6)^{1/0.22}
    = 1.576` cm, where the law is switched so that it is continuous (the text
    prints the break at 1.5 cm and Table 2 at 1.59 cm). The first law is a fit
    to hail observed in the field and to wind-tunnel measurements of printed
    hail replicas for diameters above 0.5 cm; the validity ranges of the other
    laws are not given in the corrigendum, and 0.5 cm is used for all.
    No dependence on air density is given there, and none is applied.

    ``"milbrandt_yau2005"`` is the hail law of the multimoment bulk scheme of
    Milbrandt and Yau (2005; Table 2 of Part II, from Ferrier 1994),
    :math:`V_t = \\gamma\\,206.89\\,D^{0.6384}` with :math:`D` in m, :math:`V_t`
    in m s-1 and the density factor :math:`\\gamma = (\\rho_0/\\rho)^{1/2}`
    (Part I, Eq. A1). It is the one hail law here with an air density
    dependence; no size range is stated.

Melting snow
    :math:`V_m = V_s + (V_r - V_s)\\,\\Psi(\\ell)` with the terminal velocities
    :math:`V_s` and :math:`V_r` of the dry snowflake and of the raindrop of the
    same mass and :math:`\\Psi(\\ell) = \\alpha \\ell + (1 - \\alpha) \\ell^7`,
    :math:`\\alpha = 0.246`, for the liquid water fraction :math:`\\ell`: the
    form of Eqs. (14) and (15) of Frick et al. (2013), who fitted it to the
    wind-tunnel data of Mitra et al. (1990, their Fig. 2). It rises slowly and
    linearly at small :math:`\\ell` and fast above 0.7.
"""

from __future__ import annotations

import numpy as np

HAIL_MIN_DIAMETER = 5.0  # mm; the hail fit starts at 0.5 cm
PSI_ALPHA = 0.246  # Frick et al. (2013), Eq. (15)
# (a, b) of V = a D^b, D in cm, V in m s-1: Table 3 of the corrigendum
PERCENTILES = {
    "p10": (6.73, 0.68),
    "p25": (8.33, 0.62),
    "p50": (9.45, 0.62),
    "p75": (10.54, 0.63),
    "p90": (11.58, 0.58),
}
# Milbrandt and Yau (2005b, Table 2, after Ferrier 1994): V = gamma a D^b, D in m, a = 206.89
# m^(1-b) s-1, b = 0.6384, gamma = (rho0 / rho)^(1/2) (Milbrandt and Yau 2005a, Eq. A1)
MY2005_A = 206.89
MY2005_B = 0.6384
MY2005_DENSITY_EXPONENT = 0.5
HAIL_LAWS = (
    "heymsfield2020",
    "graupel_hail_obs",
    *(f"heymsfield2020_{p}" for p in PERCENTILES),
    "milbrandt_yau2005",
)
_BRANCH_CROSSING_CM = (8.4 / 7.6) ** (1.0 / 0.22)


def hail_segments(law):
    """
    Segments of a hail law as ``(breaks_mm, a, b)`` of ``V = a (D/10)^b``.

    ``breaks_mm`` are the diameters (mm) where the next segment starts; ``a``
    (m s-1) and ``b`` have one more entry.
    """
    if law == "heymsfield2020":
        return (
            np.array([10.0 * _BRANCH_CROSSING_CM]),
            np.array([7.6, 8.4]),
            np.array([0.89, 0.67]),
        )
    if law == "graupel_hail_obs":
        return np.zeros(0), np.array([7.2]), np.array([0.84])
    if (
        isinstance(law, str)
        and law.startswith("heymsfield2020_")
        and law[15:] in PERCENTILES
    ):
        a, b = PERCENTILES[law[15:]]
        return np.zeros(0), np.array([a]), np.array([b])
    if law == "milbrandt_yau2005":
        # V = a D_m^b with D_m in m is a (0.01 D_cm)^b with D_cm in cm
        return np.zeros(0), np.array([MY2005_A * 0.01**MY2005_B]), np.array([MY2005_B])
    raise ValueError(f"hail law must be one of {HAIL_LAWS}, not {law!r}")


def density_exponent(law):
    """
    Exponent ``e`` of the air density factor ``(rho0 / rho)^e`` of a hail law.

    Only ``"milbrandt_yau2005"`` has one (0.5); the laws of Heymsfield et al.
    (2020) are given without a density dependence (0).
    """
    return MY2005_DENSITY_EXPONENT if law == "milbrandt_yau2005" else 0.0


def hail_speed(d, law="heymsfield2020"):
    """
    Terminal speed (m s-1) and its derivative with D (m s-1 mm-1) of hail of
    maximum dimension ``d`` (mm).
    """
    d = np.asarray(d, dtype=np.float64)
    breaks, a, b = hail_segments(law)
    k = np.searchsorted(breaks, d, side="right")
    with np.errstate(all="ignore"):
        v = a[k] * np.power(d / 10.0, b[k])
        dv = b[k] * v / d
    return v, dv


def psi(liquid_fraction):
    """The weight of the raindrop speed in melting snow, Eq. (15) of Frick et al. (2013)."""
    ell = np.asarray(liquid_fraction, dtype=np.float64)
    return PSI_ALPHA * ell + (1.0 - PSI_ALPHA) * ell**7
