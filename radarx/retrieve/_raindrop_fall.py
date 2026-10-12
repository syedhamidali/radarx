# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Sea-level terminal fall speed of raindrops, shared by the disdrometer,
raindrop trajectory and evaporation modules.

Two laws are available:

``"atlas1973"``
    :math:`V = 9.65 - 10.3 e^{-0.6 D}` (m s-1, D in mm) of Atlas et al.
    (1973), as quoted by Bringi and Chandrasekar (2001, Eq. 7.65b), at least
    zero. It is negative below 0.109 mm, and for drops under 0.5 mm it is
    smaller than the measured speeds.

``"drag_balance"`` (default)
    For drops smaller than ``D_JOIN`` (0.516 mm) the speed at which the weight
    of a sphere, less its buoyancy, balances the drag, with the drag
    coefficient of a sphere :math:`C_D = 24 / Re\\,(1 + 0.15 Re^{0.687})`
    (Schiller and Naumann 1935; the textbook form of the formula, not
    checked against the paper). Dividing the balance by
    the Reynolds number squared gives the Best number
    :math:`C_D Re^2 = (4/3)\\, g\\, \\rho_a (\\rho_w - \\rho_a) D^3 / \\mu^2 =
    24 Re + 3.6 Re^{1.687}`, solved for :math:`Re = \\rho_a V D / \\mu` by
    Newton iterations. For larger drops the Atlas et al. law, which drops
    flatten and the sphere law then overestimates. The two curves cross at
    ``D_JOIN``, so the speed is continuous there. The air constants are
    :math:`\\rho_a` = 1.204 kg m-3 (the density of the sea-level law,
    ``RHO0``), :math:`\\mu` = 1.81e-5 Pa s, :math:`\\rho_w` = 1000 kg m-3 and
    :math:`g` = 9.81 m s-2, radarx choices.

The speed of a sphere under 0.5 mm is the standard small-drop physics but
has not been compared with the Gunn and Kinzer (1949) table or Beard (1976).
The 0.71 m s-1 at 0.2 mm and 0.25 m s-1 at
0.1 mm it gives are the values of an independent drag-balance calculation of
the validation of issue 231; Gunn and Kinzer give about 0.7 m s-1
at 0.2 mm (not checked). Between 0.6 and 2 mm the Atlas law is within 0.05
m s-1 of the Gunn-Kinzer fit of Bringi and Chandrasekar (as quoted in that
issue; not checked).

The module also holds ``ABF``, the coefficients :math:`(a, b, f)` of
:math:`V = a D^b e^{-f D}` that approximate the ``"drag_balance"`` law, for
the closed-form drop-size-distribution integrals of
:mod:`radarx.retrieve.evaporation`.
"""

from __future__ import annotations

import numpy as np

LAWS = ("drag_balance", "atlas1973")
RHO0 = 1.204  # kg m-3, density of the sea-level fall speeds
RHO_W = 1000.0  # kg m-3
G = 9.81  # m s-2
MU = 1.81e-5  # Pa s, dynamic viscosity of air at 20 degC (radarx choice)
_NEWTON = 12


def atlas(d):
    """Atlas et al. (1973) fall speed (m s-1) and its derivative, D in mm."""
    d = np.asarray(d, dtype=np.float64)
    e = np.exp(-0.6 * d)
    return 9.65 - 10.3 * e, 6.18 * e


def sphere(d):
    """
    Drag-balance fall speed of a sphere (m s-1) and its derivative, D in mm.

    Newton iterations on :math:`h(Re) = 24 Re + 3.6 Re^{1.687} - X`, convex
    and increasing, from an upper bound of the root.
    """
    d = np.asarray(d, dtype=np.float64)
    dm = np.maximum(d, 1e-9) * 1e-3
    x = 4.0 / 3.0 * G * RHO0 * (RHO_W - RHO0) * dm**3 / MU**2
    re = np.minimum(x / 24.0, (x / 3.6) ** (1.0 / 1.687))
    for _ in range(_NEWTON):
        h = 24.0 * re + 3.6 * re**1.687 - x
        re = re - h / (24.0 + 6.0732 * re**0.687)
    slope = 24.0 + 6.0732 * re**0.687
    dre = 3.0 * x / dm / slope
    v = re * MU / (RHO0 * dm)
    dv = 1e-3 * MU / RHO0 * (dre / dm - re / dm**2)
    return v, dv


def _join():
    lo, hi = 0.3, 0.8  # the curves cross once in this range
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if sphere(mid)[0] > atlas(mid)[0]:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


D_JOIN = _join()

# (a, b, f) of V = a D^b exp(-f D) (D in mm), fitted to the "drag_balance" law
# by minimising the largest relative error (plus 0.001 times the rms error) at
# 300 diameters spaced logarithmically from 0.15 to 5 mm (Nelder-Mead). The
# error is below 5.4 % from 0.15 to 5 mm, 24 % at 0.1 mm and 34 % at 8 mm.
# tests/test_raindrop_fall.py checks these bounds and the coefficients.
ABF = (5.28695537, 1.22273346, 0.29531149)


def sea_level_speed(d, law="drag_balance"):
    """
    Sea-level terminal fall speed (m s-1, at least zero) and its derivative
    with D (m s-1 mm-1) of drops of diameter ``d`` (mm).
    """
    d = np.asarray(d, dtype=np.float64)
    with np.errstate(all="ignore"):
        if law == "atlas1973":
            v, dv = atlas(d)
        elif law == "drag_balance":
            vs, dvs = sphere(d)
            va, dva = atlas(d)
            small = d < D_JOIN
            v, dv = np.where(small, vs, va), np.where(small, dvs, dva)
        else:
            raise ValueError(f"law must be one of {LAWS}, not {law!r}")
    pos = v > 0.0
    return np.where(pos, v, 0.0), np.where(pos, dv, 0.0)
