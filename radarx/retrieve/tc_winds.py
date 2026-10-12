#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

r"""
Tropical-Cyclone Winds from One Doppler Radar: Centre, GBVTD and Decomposition
==============================================================================

A single radar measures the wind component along its beams. When a vortex is
seen from one side, the Doppler velocity on a ring around the vortex centre
still holds the axisymmetric tangential wind, the axisymmetric radial wind and
the low azimuthal wavenumbers of the tangential wind, because the beam crosses
the ring at an angle that changes along the ring. The *ground-based velocity
track display* (GBVTD, Lee et al. 1999) fits Fourier series to the Doppler
velocity on such rings. This module has three steps:

- :func:`tc_center` finds the circulation centre as the position that maximises
  the axisymmetric tangential wind retrieved by GBVTD (the GBVTD-simplex idea
  of Lee and Marks 2000), from many starting points, and chooses the radius of
  maximum wind and the centre from the consensus of the searches (Bell and Lee
  2012).
- :func:`gbvtd` retrieves, on rings around a centre, the axisymmetric
  tangential and radial wind, the wavenumber-1 and wavenumber-2 tangential
  wind and the along-beam mean wind, and the radius of maximum wind.
- :func:`tc_decompose` rebuilds the winds on a polar grid and splits them into
  the symmetric part and the asymmetric parts.

:func:`synthetic_vortex` forward-models a Doppler sweep of an analytic vortex
(the modified Rankine-combined vortex of Lee et al. 1999 as used by Chen et
al. 2013) with a mean wind, asymmetries, noise, beam height and data gaps, to
test the retrieval against a known truth.

Sources, and what was and was not read
--------------------------------------
Not read: the text of Lee et al. (1999) and of Lee and Marks (2000); the
publisher's server refused anonymous downloads. The GBVTD equations below are
taken from the conference paper of Liou et al. (2005), which reproduces the
equations of Lee et al. (1999) (its Eqs. 1 to 11), and from Chen et al. (2013,
Eq. 1, with the sign of the :math:`B_3` term that Chen et al. corrected), and
the Doppler-velocity projection (Eq. 1 of Liou et al.) was rebuilt from the
geometry. The derivation reproduces Eqs. 5 to 11 of Liou et al. and Eq. 1 of
Chen et al. from it. Everything that depends on the paper itself (the exact
treatment of the angle :math:`\psi` and of the data gaps, the ring spacing) is
*not checked against Lee et al. (1999)*. The centre finding follows the
description of Bell and Lee (2012, section 2 and 3a, including the weights; the
NORM score gives the values of their Table 1 for the best peak and a ranking
that agrees with it, but values 0.03 to 0.06 lower for the second peak); the simplex cost
function is the one Bell and Lee describe in words (the axisymmetric tangential
wind of GBVTD) and the details of Lee and Marks (2000), which were not read,
are not reproduced. The temporal steps of Bell and Lee (polynomial fits over
time and the simplex centre likelihood, their sections 3b and 3c) need a series
of volumes and are not implemented. Not implemented either: the modified GBVTD
that retrieves the cross-beam mean wind (Chen et al. 2013, MGBVTD), because its
HVVP step was only partly readable, the extended GBVTD with two radars (Liou et
al. 2005), and diagnostics of the eyewall or the rainbands.

Geometry and the equations
--------------------------
The radar is at the origin, the vortex centre at the horizontal distance
:math:`R_T` in the direction :math:`\theta_T`, and a ring of radius :math:`R`
around the centre has :math:`\sin\alpha_{max} = R / R_T` (the radar must be
outside the ring). At a gate on the ring the angle :math:`\psi` is measured from
the beam to the outward radial direction of the ring (so that :math:`\psi = 0`
at the far side of the ring and :math:`\pi` at the near side), and the
horizontal projection of the Doppler velocity, :math:`\hat V_d = V_d /
\cos\phi` for the beam elevation :math:`\phi`, is, for a uniform mean wind
:math:`V_M` blowing towards :math:`\theta_M`, a tangential wind :math:`V_T`
(positive anticlockwise, cyclonic in the northern hemisphere) and a radial wind
:math:`V_R` (positive outward),

.. math::

    \hat V_d = V_M \Big\{ \cos(\theta_T - \theta_M)
        \Big[\tfrac{1 - \cos\alpha_{max}}{2}\cos 2\psi
        + \tfrac{1 + \cos\alpha_{max}}{2}\Big]
        - \sin(\theta_T - \theta_M)\sin\alpha_{max}\sin\psi \Big\}
        - V_T \sin\psi + V_R \cos\psi .

(The mean-wind term is the exact projection with
:math:`\cos\beta = \sqrt{1 - \sin^2\alpha_{max}\sin^2\psi}` replaced by its
value at :math:`\psi = 0, \pi/2, \pi`, :math:`1 - (1 - \cos\alpha_{max})\sin^2\psi`.)
Expanding :math:`V_T` and :math:`V_R` in Fourier series of :math:`\psi` and
fitting

.. math::

    \hat V_d(\psi) = \sum_{n=0}^{3} \big(A_n \cos n\psi + B_n \sin n\psi\big)

by weighted least squares gives (Liou et al. 2005, Eqs. 5 to 11)

.. math::

    V_M \cos(\theta_T - \theta_M) &= A_0 + A_2 - V_{RC1}, \\
    V_{TC0} &= -B_1 - B_3 - V_M \sin(\theta_T - \theta_M)\sin\alpha_{max}
        + V_{RS2}, \\
    V_{RC0} &= A_1 + A_3 - V_{RC2}, \\
    V_{TS1} &= A_2 - A_0 + (A_0 + A_2 - V_{RC1})\cos\alpha_{max}, \\
    V_{TC1} &= -2 B_2 + V_{RS1}, \\
    V_{TS2} &= 2 A_3 - V_{RC2}, \qquad V_{TC2} = -2 B_3 + V_{RS2},

where :math:`V_{TCn}` and :math:`V_{TSn}` are the cosine and sine parts of the
wavenumber :math:`n` of :math:`V_T` (and likewise for :math:`V_R`). There are
more unknowns than equations. Lee et al. (1999) close the system by neglecting
the asymmetric radial wind (:math:`V_{RCn} = V_{RSn} = 0` for :math:`n \ge 1`),
which is what :func:`gbvtd` does. The along-beam mean wind
:math:`V_M\cos(\theta_T - \theta_M)` is then :math:`A_0 + A_2`. The cross-beam
mean wind :math:`V_M\sin(\theta_T - \theta_M)` cannot be found and goes into
:math:`V_{TC0}`; if a mean wind is given (the storm motion is used as a proxy,
Chen et al. 2013, Harasti et al. 2004 as cited there) the term is removed.
The largest Fourier order that the data allow depends on the largest angular
gap on the ring; gaps up to 30, 60 and 120 degrees allow orders 3, 2 and 1
(Chen et al. 2013), and a ring with a wider gap is not retrieved.

The angle :math:`\psi` is not the azimuth around the centre :math:`\gamma`
(measured clockwise from north): :math:`\gamma = \theta_T + \psi +
\arcsin(\sin\alpha_{max}\sin\psi)` in mathematical angles. The Fourier
coefficients belong to :math:`\psi`; the series in :math:`\psi` therefore
distorts the shape of a vortex asymmetry seen at the azimuth (the geometric
distortion of GBVTD, Lee et al. 1999, Lee et al. 2006). :func:`gbvtd` reports the
coefficients and the azimuth of the maximum of the wavenumber-1 wind mapped
with this relation, which is plain geometry; whether Lee et al. (1999) map the
coefficients in this way was not checked.

Ring fit
--------
Every gate with :math:`|d - R| < h` (:math:`d` the distance from the centre,
:math:`h` the half width of the ring) takes part with the weight
:math:`w\,(1 - |d - R|/h)`, where :math:`w` is the ground range of the gate
(the area of a polar gate is proportional to its range, so the fit sees the
ring like a uniformly sampled annulus), so the fit is a continuous function of
the centre, which the simplex search needs. Lee et al. (1999) and Bell and Lee
(2012) analyse constant-altitude maps on a Cartesian grid; radarx fits the gates
of a sweep directly and does not interpolate. A plan position indicator is not
at constant height: the winds of rings at different ranges are sampled at
different heights (the beam height is returned by :func:`synthetic_vortex` for
tests). The elevation of the beam at a gate is that of the effective Earth
radius model, :math:`\phi_g = \phi_0 + s / (k_e a)` for the ground distance
:math:`s`; :math:`\hat V_d = V_d / \cos\phi_g`. The vertical wind and the fall
speed are neglected.

Centre finding
--------------
A first guess of the centre and of the radius of maximum wind comes from the
user or, when none is given, from the midpoint and half the separation of the
strongest inbound and outbound velocities (a velocity dipole, the idea behind
Wood and Brown 1992 as described by Chen et al. 2013; the details here are
radarx's own: the mean position of the 0.5 percent most inbound and 0.5 percent
most outbound gates). Following Bell and Lee (2012) a grid of starting centres
around the first guess (4 by 4) is each refined for every ring radius by a
Nelder and Mead (1965) simplex search that maximises the axisymmetric
tangential wind of GBVTD with the Fourier orders 0, 1 and 2 (the choice of
Murillo et al. 2011 as reported by Bell and Lee: only the wavenumbers 0 and 1).
The mean and the spread of the centres that converge are formed, the centres
farther than one standard deviation from the mean are dropped and the mean and
spread recomputed (their steps 4 to 6). Radii within 1 km of a local maximum of
the tangential wind profile qualify as radius of maximum wind and the
normalised NORM score ranks them,

.. math::

    \mathrm{NORM} = \frac{n_{V} + 3 n_{\sigma} + n_{N}}{5},\quad
    n_V = e^{(V_T - V_{max})/(1\ \mathrm{m\,s^{-1}})},\quad
    n_\sigma = \sigma_{min}/\sigma,\quad
    n_N = 1 + \ln(N_{conv}/N_{start}),

which gives the scores of Table 1 of Bell and Lee (2012) for the radii of the best
peak (15 to 17 km) and the same ranking of the radii; for the second peak (22 to 24
km) it gives 0.03 to 0.06 less than the table, whose printed parts do not add up to
the printed scores with these weights. The centre of the
best radius is the result and its spread an estimate of its uncertainty. Not
from the paper: the use of the first guess for the range of radii (a fixed grid
is used), the floor of 10 m for :math:`\sigma`, the convergence criterion of the
simplex (0.05 km), the limit on the distance between the centre and the first
guess, and the mean tangential wind of a radius being the mean of the winds of
the converging centres (their Table 1 does not say how it is formed).

Sensitivity to the centre
-------------------------
The retrieval depends strongly on the centre. Lee and Marks (2000) found, as
reported by Bell and Lee (2012), errors of more than 20 percent in the
wavenumber-1 asymmetry for a displacement of 1 km in a storm with a radius of
maximum wind of 20 km. The tests and the notebook of radarx repeat this with
displacements of 1, 2 and 5 km.

Compiled kernel
---------------
The ring fits and the simplex searches run in a multithreaded C++ kernel
(``_tc_winds.cpp``: a uniform grid of the gates, work blocks handed out to the
threads through an atomic counter). If it is not available a NumPy
implementation with the same steps gives the same results, much more slowly.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = [
    "gbvtd",
    "synthetic_vortex",
    "tc_center",
    "tc_decompose",
    "vortex_profile",
]

__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

import numpy as np
import xarray as xr

from .._compat import renamed_arguments
from .._fields import VELOCITY, as_sweep, find_field, node_dataset
from .._provenance import provenance
from .._registry import accessor_method
from .._sweepwise import dask_support
from ..fundamentals.constants import EARTH_RADIUS
from ..fundamentals.geometry import beam_center_height, ground_range
from ..grid.cone import _sweep_names
from ._fieldtools import check_object, choose_engine, detach, node_arg, ray_dimension
from ._products import product_tree

try:
    from . import _tc_winds

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _tc_winds = None
    HAS_COMPILED_KERNEL = False

_DEG = np.pi / 180.0
_TWO_PI = 2.0 * np.pi
_BAD = 1.0e30
_RANGE = "range"
_NCOEF = 7  # A0, A1, B1, A2, B2, A3, B3

# order of the rows returned by the blocks
_CENTER_ROWS = (
    "VT0",
    "RING_CENTER_X",
    "RING_CENTER_Y",
    "CENTER_SPREAD",
    "N_CONVERGING",
    "NORM_SCORE",
    "SELECTED",
    "GUESS_X",
    "GUESS_Y",
    "GUESS_RMW",
)
_GBVTD_ROWS = (
    "VT0",
    "VR0",
    "VT1_COS",
    "VT1_SIN",
    "VT2_COS",
    "VT2_SIN",
    "VM_ALONG",
    "N_POINTS",
    "DATA_GAP",
    "FIT_ORDER",
    "FIT_RMS",
)

_NORM_WEIGHTS = (1.0, 3.0, 1.0)  # tangential wind, centre spread, converging centres
_NORM_EFOLD = 1.0  # m s-1
_SIGMA_FLOOR = 10.0  # m
_DIPOLE_FRACTION = 0.005
_SIMPLEX_STEP = 2000.0  # m, size of the first simplex

_UNITS = {
    "VT0": ("Axisymmetric tangential wind", "m s-1"),
    "VR0": ("Axisymmetric radial wind", "m s-1"),
    "VT1_COS": ("Cosine part of the wavenumber-1 tangential wind in psi", "m s-1"),
    "VT1_SIN": ("Sine part of the wavenumber-1 tangential wind in psi", "m s-1"),
    "VT2_COS": ("Cosine part of the wavenumber-2 tangential wind in psi", "m s-1"),
    "VT2_SIN": ("Sine part of the wavenumber-2 tangential wind in psi", "m s-1"),
    "VM_ALONG": ("Along-beam mean wind of the fit, A0 + A2", "m s-1"),
    "N_POINTS": ("Number of gates in the ring", "1"),
    "DATA_GAP": ("Largest angular gap of the data on the ring", "degree"),
    "FIT_ORDER": ("Highest Fourier order of the fit", "1"),
    "FIT_RMS": ("Weighted rms of the residual of the ring fit", "m s-1"),
}


# --------------------------------------------------------------------------
# analytic vortex and its Doppler velocity
# --------------------------------------------------------------------------


def vortex_profile(radius, vmax, rmw, xt=1.0, radial=None):
    """
    Axisymmetric winds of the modified Rankine-combined vortex.

    Eqs. 15 and 16 of Chen et al. (2013): the tangential wind rises linearly
    to ``vmax`` at ``rmw`` and decays as ``(rmw / radius) ** xt`` outside. The
    optional radial wind is outflow ``c1 sqrt((rmw - R) R)`` inside the radius
    of maximum wind and inflow ``-c2 sqrt(R - rmw) rmw / R`` outside, with the
    radii in km and the winds in m s-1 (the units of ``c1`` and ``c2`` in the
    paper are not consistent, so these are the units that give winds of a few
    m s-1).

    Parameters
    ----------
    radius : array-like
        Distance from the centre (m).
    vmax : float
        Maximum tangential wind (m s-1).
    rmw : float
        Radius of maximum wind (m).
    xt : float, optional
        Exponent of the decay outside the radius of maximum wind. Default 1
        (a Rankine vortex).
    radial : tuple of float, optional
        ``(c1, c2)`` of the radial wind. Default None: no radial wind.

    Returns
    -------
    tuple of numpy.ndarray
        Tangential and radial wind (m s-1) of every radius.
    """
    r = np.asarray(radius, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        inside = r <= rmw
        vt = np.where(inside, vmax * r / rmw, vmax * (rmw / r) ** xt)
        vr = np.zeros_like(r)
        if radial is not None:
            c1, c2 = radial
            rk, mk = r / 1000.0, rmw / 1000.0
            vr = np.where(
                inside,
                c1 * np.sqrt(np.maximum((mk - rk) * rk, 0.0)),
                -c2 * np.sqrt(np.maximum(rk - mk, 0.0)) * mk / rk,
            )
    return np.nan_to_num(vt, nan=0.0), np.nan_to_num(vr, nan=0.0)


def _asymmetry(gamma_deg, spec):
    """``sum_n a_n cos(n (gamma - delta_n))`` for ``spec = {n: (a_n, delta_n)}``."""
    total = 0.0
    for n, (amp, phase) in (spec or {}).items():
        total = total + amp * np.cos(n * (gamma_deg - phase) * _DEG)
    return total


@provenance(
    "Analytic modified Rankine-combined vortex with a mean wind, forward-modelled "
    "to Doppler velocity"
)
def synthetic_vortex(
    center=(60000.0, 0.0),
    vmax=50.0,
    rmw=20000.0,
    *,
    xt=1.0,
    radial=None,
    asymmetry=None,
    radial_asymmetry=None,
    mean_wind=(0.0, 0.0),
    height_scale=None,
    elevation=0.5,
    noise=0.0,
    seed=0,
    azimuth_step=1.0,
    range_step=250.0,
    max_range=150000.0,
    blank_sectors=(),
    site=(20.0, 80.0, 50.0),
    k_e=4.0 / 3.0,
):
    r"""
    Doppler velocity of an analytic tropical cyclone seen by one radar.

    The wind at a gate is the sum of a uniform mean wind, the tangential wind
    :math:`V_T = V_{T0}(R)\,[1 + \sum_n A_n \cos n(\gamma - \delta_n)]` and the
    radial wind :math:`V_R = V_{R0}(R) + \sum_n a_n \cos n(\gamma - \delta_n)`,
    with :math:`V_{T0}, V_{R0}` from :func:`vortex_profile`, the distance
    :math:`R` from the centre and the azimuth :math:`\gamma` of the gate seen
    from the centre in degrees clockwise from north. The Doppler velocity is the
    projection of the horizontal wind on the beam with the exact geometry (no
    approximation of the mean-wind term), :math:`V_d = (u \sin\alpha + v
    \cos\alpha)\cos\phi_g`, with the azimuth :math:`\alpha` and the local beam
    elevation :math:`\phi_g = \phi_0 + s/(k_e a)` of the effective Earth radius
    model. The vertical wind and fall speed are zero. This is the construction
    of the test vortex of Lee et al. (1999) as described by Chen et al. (2013)
    (a radar at the origin, the vortex 60 to 80 km away, a modified Rankine
    vortex); the asymmetric radial wind is zero by default, as there.

    Parameters
    ----------
    center : tuple of float, optional
        ``(x, y)`` of the vortex centre in m east and north of the radar.
        Default ``(60000, 0)``.
    vmax, rmw : float, optional
        Maximum tangential wind (m s-1) and radius of maximum wind (m).
    xt : float, optional
        Exponent of the decay outside the radius of maximum wind.
    radial : tuple of float, optional
        ``(c1, c2)`` of the axisymmetric radial wind, see :func:`vortex_profile`.
    asymmetry : dict, optional
        ``{n: (A_n, delta_n)}``: the relative amplitude of the wavenumber-``n``
        tangential asymmetry and the azimuth (degrees clockwise from north) of
        its maximum.
    radial_asymmetry : dict, optional
        ``{n: (a_n, delta_n)}``: the amplitude (m s-1, the same at all radii) of
        an asymmetric radial wind. GBVTD assumes that there is none.
    mean_wind : tuple of float, optional
        ``(u, v)`` of the uniform mean wind (m s-1).
    height_scale : float, optional
        If given (m), the vortex wind (not the mean wind) is multiplied by
        ``exp(-z / height_scale)`` with the beam height ``z`` above the radar,
        so that rings at different ranges of a plan position indicator see
        different winds.
    elevation : float, optional
        Beam elevation (degrees). Default 0.5.
    noise : float, optional
        Standard deviation (m s-1) of Gaussian noise added to the velocity.
    seed : int, optional
        Seed of the noise.
    azimuth_step, range_step, max_range : float, optional
        Ray spacing (degrees), gate spacing and the last gate (m).
    blank_sectors : sequence of tuple, optional
        ``(start, end)`` azimuths (degrees clockwise from north, seen from the
        vortex centre) in which there are no data, to test data gaps on the
        rings.
    site : tuple of float, optional
        ``(latitude, longitude, altitude)`` of the radar (degrees, degrees, m).
    k_e : float, optional
        Effective Earth radius factor of the beam geometry. Default 4/3.

    Returns
    -------
    xarray.Dataset
        A sweep with the dimensions ``azimuth`` and ``range`` (m), ``elevation``
        along the rays, the radar site, the Doppler velocity ``VRADH`` (m s-1),
        the beam height ``BEAM_HEIGHT`` (m above the radar) and the true
        tangential and radial winds ``VT_TRUE`` and ``VR_TRUE`` (m s-1, relative
        to the mean wind) of every gate. The parameters of the vortex are in the
        attributes.

    References
    ----------
    .. [1] Chen, X., K. Zhao, W.-C. Lee, B. J.-D. Jou, M. Xue, and P. R.
       Harasti, 2013: The improvement to the environmental wind and tropical
       cyclone circulation retrievals with the modified GBVTD (MGBVTD)
       technique. *J. Appl. Meteor. Climatol.*, **52** (11), 2493-2508,
       https://doi.org/10.1175/JAMC-D-13-031.1
    """
    if not (rmw > 0 and vmax >= 0):
        raise ValueError("rmw must be positive and vmax not negative")
    if not (range_step > 0 and azimuth_step > 0 and max_range > range_step):
        raise ValueError("azimuth_step, range_step and max_range must be positive")
    cx, cy = (float(v) for v in center)
    um, vm = (float(v) for v in mean_wind)
    az = (np.arange(0.0, 360.0, azimuth_step) + 0.5 * azimuth_step) % 360.0
    rng = np.arange(range_step, max_range + 0.5 * range_step, range_step)
    reff = k_e * EARTH_RADIUS
    lat0, lon0, alt0 = (float(v) for v in site)
    s = ground_range(rng[None, :], elevation, reff=reff, radar_height=alt0)
    s = np.broadcast_to(s, (az.size, rng.size))
    sin_a, cos_a = np.sin(az * _DEG)[:, None], np.cos(az * _DEG)[:, None]
    x, y = s * sin_a, s * cos_a
    phi = elevation * _DEG + s / reff
    dx, dy = x - cx, y - cy
    dist = np.hypot(dx, dy)
    vt0, vr0 = vortex_profile(dist, vmax, rmw, xt, radial)
    gamma = np.arctan2(dx, dy) / _DEG
    vt = vt0 * (1.0 + _asymmetry(gamma, asymmetry))
    vr = vr0 + _asymmetry(gamma, radial_asymmetry)
    z = beam_center_height(rng[None, :], elevation, radar_height=alt0, reff=reff) - alt0
    z = np.broadcast_to(z, s.shape)
    if height_scale is not None:
        factor = np.exp(-z / float(height_scale))
        vt, vr = vt * factor, vr * factor
    with np.errstate(divide="ignore", invalid="ignore"):
        rx = np.where(dist > 0, dx / dist, 0.0)
        ry = np.where(dist > 0, dy / dist, 0.0)
    u = um + vr * rx - vt * ry
    v = vm + vr * ry + vt * rx
    vd = (u * sin_a + v * cos_a) * np.cos(phi)
    if noise:
        vd = vd + np.random.default_rng(seed).normal(0.0, noise, vd.shape)
    vd = np.where(s <= 0, np.nan, vd)
    for start, end in blank_sectors:
        lo, hi = start % 360.0, end % 360.0
        g = gamma % 360.0
        inside = (g >= lo) & (g <= hi) if lo <= hi else (g >= lo) | (g <= hi)
        vd = np.where(inside, np.nan, vd)
    ds = xr.Dataset(
        {
            "VRADH": (
                ("azimuth", _RANGE),
                vd,
                {"units": "m s-1", "long_name": "Doppler velocity"},
            ),
            "BEAM_HEIGHT": (
                ("azimuth", _RANGE),
                z,
                {"units": "m", "long_name": "Beam height above the radar"},
            ),
            "VT_TRUE": (("azimuth", _RANGE), vt, {"units": "m s-1"}),
            "VR_TRUE": (("azimuth", _RANGE), vr, {"units": "m s-1"}),
        },
        coords={
            "azimuth": ("azimuth", az, {"units": "degree"}),
            _RANGE: (_RANGE, rng, {"units": "m"}),
            "elevation": ("azimuth", np.full(az.size, float(elevation))),
            "latitude": lat0,
            "longitude": lon0,
            "altitude": alt0,
        },
    )
    ds.attrs.update(
        {
            "vortex_center_x": cx,
            "vortex_center_y": cy,
            "vortex_vmax": float(vmax),
            "vortex_rmw": float(rmw),
            "vortex_xt": float(xt),
            "mean_wind_u": um,
            "mean_wind_v": vm,
        }
    )
    return ds


# --------------------------------------------------------------------------
# geometry of a sweep and the NumPy reference of the kernel
# --------------------------------------------------------------------------


def _scalar_of(ds, name, default=None):
    if name in ds.variables:
        value = np.asarray(ds[name].values, dtype=np.float64).ravel()
        value = value[np.isfinite(value)]
        if value.size:
            return float(value[0])
    return default


def _elevation_of(ds, ray_dim, n):
    for name in ("elevation", "sweep_fixed_angle"):
        if name in ds.variables:
            el = ds[name]
            if el.ndim == 0:
                return np.full(n, float(el.values))
            if el.dims == (ray_dim,):
                return np.asarray(el.values, dtype=np.float64)
            raise ValueError(f"{name!r} must be a scalar or have one value per ray")
    raise ValueError(
        "the sweep needs an 'elevation' (one value per ray) or a "
        "'sweep_fixed_angle' variable"
    )


class _Sweep:
    """Gates of one sweep: horizontal positions, projection factors and weights."""

    def __init__(self, obj, field, k_e):
        if isinstance(obj, xr.DataArray):
            name = obj.name or "field"
            ds = obj.to_dataset(name=name)
            vel = ds[name]
        else:
            ds = obj
            vel = ds[find_field(ds, VELOCITY, field)]
        self.ray_dim = ray_dimension(vel)
        if vel.ndim != 2:
            raise ValueError(
                f"the velocity needs the two dimensions (ray, range), not {vel.dims}"
            )
        vel = vel.transpose(self.ray_dim, _RANGE)
        if "azimuth" not in ds.variables:
            raise ValueError("the sweep needs an 'azimuth' coordinate")
        az = ds["azimuth"]
        if az.dims != (self.ray_dim,):
            raise ValueError("azimuth must have one value per ray")
        az = np.asarray(az.values, dtype=np.float64)
        rng = np.asarray(vel[_RANGE].values, dtype=np.float64)
        el = _elevation_of(ds, self.ray_dim, az.size)
        self.altitude = _scalar_of(ds, "altitude", 0.0)
        self.lat = _scalar_of(ds, "latitude")
        self.lon = _scalar_of(ds, "longitude")
        reff = k_e * EARTH_RADIUS
        s = ground_range(
            rng[None, :], el[:, None], reff=reff, radar_height=self.altitude
        )
        sin_a, cos_a = np.sin(az * _DEG)[:, None], np.cos(az * _DEG)[:, None]
        dims = (self.ray_dim, _RANGE)
        self.x = xr.DataArray(s * sin_a, dims=dims)
        self.y = xr.DataArray(s * cos_a, dims=dims)
        self.w = xr.DataArray(s, dims=dims)
        cos_phi = np.cos(el[:, None] * _DEG + s / reff)
        self.velocity = vel
        self.vh = vel / xr.DataArray(cos_phi, dims=dims)


def _gate_arrays(vh, x, y, w):
    """Flat float64 arrays of the valid gates of one block."""
    vh, x, y, w = (np.asarray(a, dtype=np.float64).ravel() for a in (vh, x, y, w))
    ok = np.isfinite(vh) & np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    return x[ok], y[ok], vh[ok], w[ok]


def _crop(gates, anchor, radius):
    """The gates within ``radius`` of ``anchor`` (changes no ring that is inside)."""
    x, y, v, w = gates
    keep = np.hypot(x - anchor[0], y - anchor[1]) <= radius
    return x[keep], y[keep], v[keep], w[keep]


class _NumpyGates:
    """NumPy implementation of the ring fit of the kernel (same steps)."""

    def __init__(self, gates):
        self.x, self.y, self.v, self.w = gates
        self.azim = np.arctan2(self.y, self.x)

    def fit(self, cx, cy, radius, half_width, max_order, min_points, max_gap):
        """``(coefficients, diagnostics)`` of one ring, None if it cannot be fitted."""
        t = 1.0 - np.abs(np.hypot(self.x - cx, self.y - cy) - radius) / half_width
        sel = (t > 0.0) & np.isfinite(self.v)
        m = int(sel.sum())
        if m < min_points or m < 2:
            return None
        psi = np.arctan2(self.y[sel] - cy, self.x[sel] - cx) - self.azim[sel]
        psi = psi - _TWO_PI * np.floor(psi / _TWO_PI)
        ordered = np.sort(psi)
        gap = max(
            ordered[0] + _TWO_PI - ordered[-1],
            float(np.max(np.diff(ordered))) if m > 1 else 0.0,
        )
        order = 0
        for limit, value in ((30.0, 3), (60.0, 2), (120.0, 1)):
            if gap <= limit * _DEG:
                order = value
                break
        order = min(order, max_order)
        if gap > max_gap * _DEG or order < 1:
            return None
        ncol = 2 * order + 1  # fewer gates than columns would leave a gap > 120 degrees
        wt = self.w[sel] * t[sel]
        val = self.v[sel]
        basis = np.empty((m, ncol))
        basis[:, 0] = 1.0
        for n in range(1, order + 1):
            basis[:, 2 * n - 1] = np.cos(n * psi)
            basis[:, 2 * n] = np.sin(n * psi)
        ata = (basis * wt[:, None]).T @ basis
        atb = (basis * wt[:, None]).T @ val
        sol = np.linalg.solve(ata, atb)
        resid = val - basis @ sol
        coef = np.zeros(_NCOEF)
        coef[:ncol] = sol
        rms = np.sqrt(np.sum(wt * resid * resid) / np.sum(wt))
        return coef, np.array([m, gap / _DEG, order, rms], dtype=np.float64)


def _rings_numpy(gates, cx, cy, radii, cfg):
    ng = _NumpyGates(gates)
    coef = np.full((len(cx), len(radii), _NCOEF), np.nan)
    diag = np.full((len(cx), len(radii), 4), np.nan)
    for i, (xc, yc) in enumerate(zip(cx, cy)):
        for k, radius in enumerate(radii):
            res = ng.fit(
                xc,
                yc,
                radius,
                cfg["half_width"],
                cfg["max_order"],
                cfg["min_points"],
                cfg["max_gap"],
            )
            if res is not None:
                coef[i, k], diag[i, k] = res
    return coef, diag


def _vt0_of(coef, cx, cy, radius, um, vm):
    """Axisymmetric tangential wind of a ring (``-B1 - B3 - VM sin(dT) sin(a)``)."""
    rt = np.hypot(cx, cy)
    return -coef[..., 2] - coef[..., 6] - (cy * um - cx * vm) / rt * (radius / rt)


def _search_numpy(gates, starts, radii, cfg, search):
    """Nelder-Mead search of every start and radius, as in the kernel."""
    ng = _NumpyGates(gates)
    um, vm = cfg["mean_wind"]
    ax, ay, limit = search["anchor"][0], search["anchor"][1], search["limit"]
    step, tol, max_iter = search["step"], search["tolerance"], search["max_iter"]

    def make_cost(radius, counter):
        def cost(px, py):
            counter[0] += 1
            if np.hypot(px - ax, py - ay) > limit:
                return _BAD
            rt = np.hypot(px, py)
            if not rt > radius * 1.0001:
                return _BAD
            res = ng.fit(
                px,
                py,
                radius,
                cfg["half_width"],
                cfg["max_order"],
                cfg["min_points"],
                cfg["max_gap"],
            )
            if res is None:
                return _BAD
            return -float(_vt0_of(res[0], px, py, radius, um, vm))

        return cost

    out = np.full((len(starts), len(radii), 5), np.nan)
    for ig, (sx, sy) in enumerate(starts):
        for ir, radius in enumerate(radii):
            counter = [0]
            cost = make_cost(radius, counter)
            vx = np.array([sx, sx + step, sx])
            vy = np.array([sy, sy, sy + step])
            fv = np.array([cost(vx[k], vy[k]) for k in range(3)])
            converged = False
            for _ in range(max_iter):
                o = np.argsort(fv, kind="stable")
                ib, im, iw = int(o[0]), int(o[1]), int(o[2])
                size = max(
                    np.hypot(vx[im] - vx[ib], vy[im] - vy[ib]),
                    np.hypot(vx[iw] - vx[ib], vy[iw] - vy[ib]),
                )
                if fv[ib] >= _BAD:
                    break
                if size < tol:
                    converged = True
                    break
                mx, my = 0.5 * (vx[ib] + vx[im]), 0.5 * (vy[ib] + vy[im])
                rx, ry = mx + (mx - vx[iw]), my + (my - vy[iw])
                fr = cost(rx, ry)
                shrink = False
                if fv[ib] <= fr < fv[im]:
                    vx[iw], vy[iw], fv[iw] = rx, ry, fr
                elif fr < fv[ib]:
                    ex, ey = mx + 2.0 * (mx - vx[iw]), my + 2.0 * (my - vy[iw])
                    fe = cost(ex, ey)
                    if fe < fr:
                        vx[iw], vy[iw], fv[iw] = ex, ey, fe
                    else:
                        vx[iw], vy[iw], fv[iw] = rx, ry, fr
                elif fr < fv[iw]:
                    cxo, cyo = mx + 0.5 * (rx - mx), my + 0.5 * (ry - my)
                    fc = cost(cxo, cyo)
                    if fc <= fr:
                        vx[iw], vy[iw], fv[iw] = cxo, cyo, fc
                    else:
                        shrink = True
                else:
                    cxi, cyi = mx + 0.5 * (vx[iw] - mx), my + 0.5 * (vy[iw] - my)
                    fc = cost(cxi, cyi)
                    if fc < fv[iw]:
                        vx[iw], vy[iw], fv[iw] = cxi, cyi, fc
                    else:
                        shrink = True
                if shrink:
                    for k in (im, iw):
                        vx[k] = vx[ib] + 0.5 * (vx[k] - vx[ib])
                        vy[k] = vy[ib] + 0.5 * (vy[k] - vy[ib])
                        fv[k] = cost(vx[k], vy[k])
            best = int(np.argmin(fv))
            ok = fv[best] < _BAD
            out[ig, ir] = (
                vx[best] if ok else np.nan,
                vy[best] if ok else np.nan,
                -fv[best] if ok else np.nan,
                counter[0],
                1.0 if (ok and converged) else 0.0,
            )
    return out


def _rings(gates, cx, cy, radii, cfg, compiled, n_threads):
    if compiled:
        return _tc_winds.gbvtd_rings(
            *gates,
            np.asarray(cx, dtype=np.float64),
            np.asarray(cy, dtype=np.float64),
            np.asarray(radii, dtype=np.float64),
            cfg["half_width"],
            cfg["max_order"],
            cfg["min_points"],
            cfg["max_gap"],
            int(n_threads or 0),
        )
    return _rings_numpy(gates, cx, cy, radii, cfg)


def _search(gates, starts, radii, cfg, search, compiled, n_threads):
    if compiled:
        um, vm = cfg["mean_wind"]
        return _tc_winds.center_search(
            *gates,
            np.asarray(starts, dtype=np.float64),
            np.asarray(radii, dtype=np.float64),
            cfg["half_width"],
            cfg["max_order"],
            cfg["min_points"],
            cfg["max_gap"],
            um,
            vm,
            search["step"],
            search["tolerance"],
            search["max_iter"],
            search["limit"],
            search["anchor"][0],
            search["anchor"][1],
            int(n_threads or 0),
        )
    return _search_numpy(gates, starts, radii, cfg, search)


# --------------------------------------------------------------------------
# centre finding
# --------------------------------------------------------------------------


def _dipole(gates):
    """First guess ``(x, y, rmw)``: the strongest inbound and outbound velocities."""
    x, y, v, _ = gates
    if v.size < 20:
        return None
    n = max(1, int(round(_DIPOLE_FRACTION * v.size)))
    order = np.argsort(v, kind="stable")
    inbound, outbound = order[:n], order[-n:]
    xi, yi = x[inbound].mean(), y[inbound].mean()
    xo, yo = x[outbound].mean(), y[outbound].mean()
    return 0.5 * (xi + xo), 0.5 * (yi + yo), 0.5 * float(np.hypot(xo - xi, yo - yi))


def _consensus(res, radii, n_start):
    """
    Mean centre, spread and wind of the converging simplex centres of every radius.

    Steps 4 to 6 of Bell and Lee (2012, section 2): the centres farther than
    one standard deviation from the mean are dropped, then the mean and the
    standard deviation are formed again. The standard deviation of a set of
    centres is the root mean square of their distances from the mean.
    """
    nr = len(radii)
    out = np.full((6, nr), np.nan)  # vt, x, y, spread, n_conv, (unused)
    out[4] = 0.0
    for k in range(nr):
        ok = res[:, k, 4] == 1.0
        if not ok.any():
            continue
        px, py, pv = res[ok, k, 0], res[ok, k, 1], res[ok, k, 2]
        d = np.hypot(px - px.mean(), py - py.mean())
        sigma = np.sqrt(np.mean(d * d))
        keep = d <= sigma if sigma > 0 else np.ones(d.shape, dtype=bool)
        px, py, pv = px[keep], py[keep], pv[keep]
        mx, my = px.mean(), py.mean()
        out[0, k] = pv.mean()
        out[1, k], out[2, k] = mx, my
        out[3, k] = np.sqrt(np.mean((px - mx) ** 2 + (py - my) ** 2))
        out[4, k] = keep.sum()
    return out


def _qualified(vt, radii, window):
    """Radii within ``window`` of a local maximum of the wind profile."""
    finite = np.isfinite(vt)
    peaks = [
        k
        for k in range(1, len(vt) - 1)
        if finite[k - 1 : k + 2].all() and vt[k] > vt[k - 1] and vt[k] >= vt[k + 1]
    ]
    if not peaks:
        return finite
    mask = np.zeros(len(vt), dtype=bool)
    for k in peaks:
        mask |= np.abs(radii - radii[k]) <= window + 1e-9
    return mask & finite


def _norm_scores(vt, spread, n_conv, n_start, qualified):
    """The NORM score of Bell and Lee (2012, section 3a); NaN where not qualified."""
    score = np.full(len(vt), np.nan)
    use = qualified & (n_conv > 0)
    if not use.any():
        return score
    n_vt = np.exp((vt[use] - vt[use].max()) / _NORM_EFOLD)
    sigma = np.maximum(spread[use], _SIGMA_FLOOR)
    n_sigma = sigma.min() / sigma
    n_n = 1.0 + np.log(n_conv[use] / n_start)
    w = _NORM_WEIGHTS
    score[use] = (w[0] * n_vt + w[1] * n_sigma + w[2] * n_n) / sum(w)
    return score


def _center_block(vh, x, y, w, *, radii, cfg, opts, compiled, n_threads):
    """Rows of ``_CENTER_ROWS`` for one sweep (all NaN if there is no first guess)."""
    nr = len(radii)
    out = np.full((len(_CENTER_ROWS), nr), np.nan)
    gates = _gate_arrays(vh, x, y, w)
    guess = opts["first_guess"]
    rough = _dipole(gates)
    if guess is None:
        if rough is None:
            return out
        guess = rough[:2]
    rough_rmw = rough[2] if rough is not None else float(np.median(radii))
    spacing = opts["guess_spacing"]
    if spacing is None:
        spacing = float(np.clip(0.25 * rough_rmw, 1000.0, 10000.0))
    limit = opts["search_radius"]
    if limit is None:
        limit = max(2.0 * rough_rmw, 10000.0)
    out[7], out[8], out[9] = guess[0], guess[1], rough_rmw
    n = opts["guess_grid"]
    offsets = (np.arange(n) - 0.5 * (n - 1)) * spacing
    starts = np.array([(guess[0] + a, guess[1] + b) for b in offsets for a in offsets])
    crop = limit + float(radii.max()) + cfg["half_width"] + 1.0
    gates = _crop(gates, guess, crop)
    search = {
        "step": opts["step"],
        "tolerance": opts["tolerance"],
        "max_iter": opts["max_iter"],
        "limit": limit,
        "anchor": guess,
    }
    res = _search(gates, starts, radii, cfg, search, compiled, n_threads)
    cons = _consensus(res, radii, len(starts))
    qualified = _qualified(cons[0], radii, opts["inflection_window"])
    score = _norm_scores(cons[0], cons[3], cons[4], len(starts), qualified)
    out[0:5] = cons[0:5]
    out[5] = score
    out[6] = 0.0
    if np.isfinite(score).any():
        out[6, int(np.nanargmax(score))] = 1.0
    return out


def _radii_of(radii, min_radius, max_radius, radius_step):
    if radii is None:
        if not (0 < min_radius <= max_radius and radius_step > 0):
            raise ValueError("need 0 < min_radius <= max_radius and radius_step > 0")
        radii = np.arange(min_radius, max_radius + 0.5 * radius_step, radius_step)
    radii = np.atleast_1d(np.asarray(radii, dtype=np.float64))
    if radii.ndim != 1 or radii.size == 0 or not np.all(radii > 0):
        raise ValueError(
            "radii must be a non-empty 1-D array of positive distances (m)"
        )
    return radii


def _offset_to_latlon(x, y, lat0, lon0):
    """Latitude and longitude of the point ``(x, y)`` m east and north of the radar."""
    delta = np.hypot(x, y) / EARTH_RADIUS
    bearing = np.arctan2(x, y)
    phi0 = lat0 * _DEG
    sin_lat = np.sin(phi0) * np.cos(delta) + np.cos(phi0) * np.sin(delta) * np.cos(
        bearing
    )
    lat = np.arcsin(sin_lat)
    lon = lon0 * _DEG + np.arctan2(
        np.sin(bearing) * np.sin(delta) * np.cos(phi0),
        np.cos(delta) - np.sin(phi0) * sin_lat,
    )
    return lat / _DEG, lon / _DEG


def _latlon_to_offset(lat, lon, lat0, lon0):
    """Position ``(x, y)`` in m east and north of the radar of a latitude and longitude."""
    p0, p1 = lat0 * _DEG, lat * _DEG
    dlon = (lon - lon0) * _DEG
    bearing = np.arctan2(
        np.sin(dlon) * np.cos(p1),
        np.cos(p0) * np.sin(p1) - np.sin(p0) * np.cos(p1) * np.cos(dlon),
    )
    a = np.sin((p1 - p0) / 2) ** 2 + np.cos(p0) * np.cos(p1) * np.sin(dlon / 2) ** 2
    dist = 2.0 * EARTH_RADIUS * np.arcsin(np.sqrt(a))
    return float(dist * np.sin(bearing)), float(dist * np.cos(bearing))


def _mean_wind(mean_wind):
    if mean_wind is None:
        return (0.0, 0.0)
    try:
        u, v = (float(m) for m in mean_wind)
    except (TypeError, ValueError) as err:
        raise ValueError("mean_wind must be a pair (u, v) in m s-1") from err
    if not (np.isfinite(u) and np.isfinite(v)):
        raise ValueError("mean_wind must be finite")
    return (u, v)


def _fit_config(ring_width, max_wavenumber, min_points, max_gap, mean_wind):
    if not ring_width > 0:
        raise ValueError(f"ring_width must be positive, not {ring_width!r}")
    if int(max_wavenumber) != max_wavenumber or not 1 <= max_wavenumber <= 3:
        raise ValueError(
            f"max_wavenumber (the Fourier order of the fit) must be 1, 2 or 3, "
            f"not {max_wavenumber!r}"
        )
    if int(min_points) != min_points or min_points < 2:
        raise ValueError(
            f"min_points must be an integer of at least 2, not {min_points!r}"
        )
    if not 0 < max_gap <= 360:
        raise ValueError(f"max_gap must be between 0 and 360 degrees, not {max_gap!r}")
    return {
        "half_width": 0.5 * float(ring_width),
        "max_order": int(max_wavenumber),
        "min_points": int(min_points),
        "max_gap": float(max_gap),
        "mean_wind": _mean_wind(mean_wind),
    }


def _stack(sweep, block, rows, n_out):
    """Apply a block function over the (ray, range) gates, lazily for dask."""
    ray = sweep.ray_dim
    core = [[ray, _RANGE]] * 4
    out = xr.apply_ufunc(
        block,
        *detach(sweep.vh, sweep.x, sweep.y, sweep.w),
        input_core_dims=core,
        output_core_dims=[["quantity", "radius"]],
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={
            "output_sizes": {"quantity": len(rows), "radius": n_out},
            "allow_rechunk": True,
        },
        keep_attrs=False,
    )
    return out


def _variables(out, rows, radii):
    """The rows of a stacked block result as variables on the ``radius`` axis."""
    out = out.reset_coords(drop=True)
    data = {name: out.isel(quantity=i, drop=True) for i, name in enumerate(rows)}
    ds = xr.Dataset(data).assign_coords(
        radius=(
            "radius",
            radii,
            {"units": "m", "long_name": "Distance from the centre"},
        )
    )
    return ds


def _scalar_at(ds, name, selected):
    return ds[name].where(selected).sum("radius", min_count=1)


def _center_dataset(sweep, out, radii):
    ds = _variables(out, _CENTER_ROWS, radii)
    selected = ds["SELECTED"] == 1
    center_x = _scalar_at(ds, "RING_CENTER_X", selected)
    center_y = _scalar_at(ds, "RING_CENTER_Y", selected)
    scalars = {
        "CENTER_X": center_x,
        "CENTER_Y": center_y,
        "RMW": ds["radius"].where(selected).sum("radius", min_count=1),
        "VT_MAX": _scalar_at(ds, "VT0", selected),
        "CENTER_UNCERTAINTY": _scalar_at(ds, "CENTER_SPREAD", selected),
        "FIRST_GUESS_X": ds["GUESS_X"].isel(radius=0, drop=True),
        "FIRST_GUESS_Y": ds["GUESS_Y"].isel(radius=0, drop=True),
        "FIRST_GUESS_RMW": ds["GUESS_RMW"].isel(radius=0, drop=True),
    }
    ds = ds.drop_vars(["GUESS_X", "GUESS_Y", "GUESS_RMW"])
    for name, value in scalars.items():
        ds[name] = value
    if sweep.lat is not None and sweep.lon is not None:
        lat, lon = _offset_to_latlon(center_x, center_y, sweep.lat, sweep.lon)
        ds["CENTER_LATITUDE"] = lat
        ds["CENTER_LONGITUDE"] = lon
        ds["CENTER_LATITUDE"].attrs = {"units": "degrees_north"}
        ds["CENTER_LONGITUDE"].attrs = {"units": "degrees_east"}
    attrs = {
        "VT0": ("Mean axisymmetric tangential wind of the converging centres", "m s-1"),
        "RING_CENTER_X": ("Consensus centre of the ring, east of the radar", "m"),
        "RING_CENTER_Y": ("Consensus centre of the ring, north of the radar", "m"),
        "CENTER_SPREAD": ("Root mean square distance of the converging centres", "m"),
        "N_CONVERGING": ("Number of converging simplex centres", "1"),
        "NORM_SCORE": ("Normalised score of the ring as radius of maximum wind", "1"),
        "SELECTED": ("1 for the ring that gives the centre", "1"),
        "CENTER_X": ("Circulation centre, metres east of the radar", "m"),
        "CENTER_Y": ("Circulation centre, metres north of the radar", "m"),
        "RMW": ("Radius of maximum wind of the best ring", "m"),
        "VT_MAX": (
            "Axisymmetric tangential wind at the radius of maximum wind",
            "m s-1",
        ),
        "CENTER_UNCERTAINTY": (
            "Spread of the converging centres of the best ring",
            "m",
        ),
        "FIRST_GUESS_X": ("First guess of the centre, east of the radar", "m"),
        "FIRST_GUESS_Y": ("First guess of the centre, north of the radar", "m"),
        "FIRST_GUESS_RMW": ("First guess of the radius of maximum wind", "m"),
    }
    for name, (long_name, units) in attrs.items():
        ds[name].attrs = {"long_name": long_name, "units": units}
    return ds


def _guess_of(first_guess, first_guess_latlon, sweep):
    if first_guess is not None and first_guess_latlon is not None:
        raise ValueError("give first_guess or first_guess_latlon, not both")
    if first_guess_latlon is not None:
        if sweep.lat is None or sweep.lon is None:
            raise ValueError("the sweep has no latitude and longitude of the radar")
        lat, lon = (float(v) for v in first_guess_latlon)
        return _latlon_to_offset(lat, lon, sweep.lat, sweep.lon)
    if first_guess is None:
        return None
    x, y = (float(v) for v in first_guess)
    return (x, y)


def _per_sweep(obj, func):
    """Apply ``func(sweep_dataset, name)`` to a sweep or to every sweep of a volume."""
    obj = as_sweep(obj)
    if isinstance(obj, xr.DataTree):
        nodes = {
            name: func(node_dataset(obj[name]), name) for name in _sweep_names(obj)
        }
        return product_tree(obj, nodes)
    return func(obj, None)


@dask_support("lazy_only")
@renamed_arguments(velocity="field")
@provenance(
    "Tropical-cyclone centre and radius of maximum wind from the GBVTD-simplex "
    "consensus of Bell and Lee (2012), single volume",
    extra_refs=("liou-2005",),
)
def tc_center(
    obj,
    *,
    field=None,
    first_guess=None,
    first_guess_latlon=None,
    radii=None,
    min_radius=5000.0,
    max_radius=60000.0,
    radius_step=1000.0,
    ring_width=2000.0,
    max_wavenumber=2,
    mean_wind=None,
    guess_grid=4,
    guess_spacing=None,
    search_radius=None,
    inflection_window=1000.0,
    min_points=30,
    max_gap=120.0,
    k_e=4.0 / 3.0,
    n_threads=None,
    engine="auto",
):
    """
    Circulation centre and radius of maximum wind of a tropical cyclone.

    Searches, from a grid of starting points around a first guess and for every
    ring radius, the centre that maximises the axisymmetric tangential wind
    retrieved by GBVTD, takes the consensus of the converging searches and
    selects the radius with the best NORM score (Bell and Lee 2012, one volume).
    The method, the sources and what is not taken from them are in
    :mod:`radarx.retrieve.tc_winds`.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        A plan position indicator of the Doppler velocity with ``azimuth`` (one
        value per ray), ``range`` (m) and ``elevation`` (per ray, else
        ``sweep_fixed_angle``) and the radar ``altitude`` (``latitude`` and
        ``longitude`` for geographic output); a DataArray is the velocity
        itself. For a volume every sweep is processed. Dask-backed input gives
        a lazy result. The velocity must be dealiased
        (:func:`radarx.retrieve.dealias_velocity`).
    field : str, optional
        Name of the velocity field. Default: the first of ``VRADH``, ``VRAD``
        and the like.
    first_guess : tuple of float, optional
        ``(x, y)`` of the first guess of the centre in m east and north of the
        radar (a best-track or eye-fix position). Default: the midpoint of the
        strongest inbound and outbound velocities.
    first_guess_latlon : tuple of float, optional
        ``(latitude, longitude)`` of the first guess instead of ``first_guess``.
    radii : array-like, optional
        Ring radii (m). Default ``min_radius`` to ``max_radius`` in steps of
        ``radius_step``.
    min_radius, max_radius, radius_step : float, optional
        Default radii (m): 5 to 60 km in 1 km steps.
    ring_width : float, optional
        Full width of the triangular weight of a ring (m). Default 2000.
    max_wavenumber : int, optional
        Highest Fourier order of the ring fit (1 to 3). Default 2, the
        wavenumbers 0 and 1 of the vortex.
    mean_wind : tuple of float, optional
        ``(u, v)`` mean wind or storm motion (m s-1) that removes the cross-beam
        mean wind from the tangential wind. Default None: not removed.
    guess_grid : int, optional
        The starting points form a ``guess_grid`` by ``guess_grid`` grid.
        Default 4 (16 searches per ring, as in Bell and Lee).
    guess_spacing : float, optional
        Spacing of the starting points (m). Default: a quarter of the radius
        of maximum wind of the dipole estimate, between 1 and 10 km.
    search_radius : float, optional
        A centre farther than this (m) from the first guess is rejected.
        Default: twice the dipole estimate of the radius of maximum wind, at
        least 10 km.
    inflection_window : float, optional
        Radii within this distance (m) of a local maximum of the tangential
        wind profile qualify as radius of maximum wind. Default 1000.
    min_points : int, optional
        Fewest gates in a ring. Default 30.
    max_gap : float, optional
        Largest angular gap (degrees) of the data of a ring. Default 120.
    k_e : float, optional
        Effective Earth radius factor of the beam geometry. Default 4/3.
    n_threads : int, optional
        Threads of the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation. ``"auto"`` (default) prefers the compiled kernel.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        On the dimension ``radius`` (m): ``VT0`` (the mean axisymmetric
        tangential wind of the converging centres), ``RING_CENTER_X`` and
        ``RING_CENTER_Y``, ``CENTER_SPREAD``, ``N_CONVERGING``, ``NORM_SCORE``
        (NaN for radii that do not qualify) and ``SELECTED``. Scalars:
        ``CENTER_X`` and ``CENTER_Y`` (m east and north of the radar),
        ``CENTER_LATITUDE`` and ``CENTER_LONGITUDE``, ``RMW`` (m), ``VT_MAX``
        (m s-1), ``CENTER_UNCERTAINTY`` (m) and the first guess. All NaN if no
        ring could be fitted. For a volume a DataTree with one node per sweep.
        The input is not changed.

    Raises
    ------
    ValueError
        For invalid options or a sweep without the geometry.
    TypeError
        If ``obj`` is not an xarray object.
    ImportError
        If ``engine="compiled"`` and the kernel is not available.

    Notes
    -----
    Not checked against Lee and Marks (2000) (not read); see the module
    documentation for the choices that are radarx's own.

    References
    ----------
    .. [1] Lee, W.-C., and F. D. Marks, 2000: Tropical cyclone kinematic
       structure retrieved from single-Doppler radar observations. Part II: The
       GBVTD-simplex center finding algorithm. *Mon. Wea. Rev.*, **128** (6),
       1925-1936, https://doi.org/10.1175/1520-0493(2000)128<1925:TCKSRF>2.0.CO;2
    .. [2] Bell, M. M., and W.-C. Lee, 2012: Objective tropical cyclone center
       tracking using single-Doppler radar. *J. Appl. Meteor. Climatol.*,
       **51** (5), 878-896, https://doi.org/10.1175/JAMC-D-11-0167.1
    .. [3] Lee, W.-C., B. J.-D. Jou, P.-L. Chang, and S.-M. Deng, 1999: Tropical
       cyclone kinematic structure retrieved from single-Doppler radar
       observations. Part I: Interpretation of Doppler velocity patterns and
       the GBVTD technique. *Mon. Wea. Rev.*, **127** (10), 2419-2439,
       https://doi.org/10.1175/1520-0493(1999)127<2419:TCKSRF>2.0.CO;2
    .. [4] Nelder, J. A., and R. Mead, 1965: A simplex method for function
       minimization. *Comput. J.*, **7** (4), 308-313,
       https://doi.org/10.1093/comjnl/7.4.308
    .. [5] Liou, Y.-C., T.-C. C. Wang, W.-C. Lee, and Y.-J. Chang, 2005: The
       extended GBVTD and the recovery of the tropical cyclones asymmetric
       structures. *Preprints, 32nd Conf. on Radar Meteorology*, Albuquerque,
       NM, Amer. Meteor. Soc., JP2J.8 (conference paper, no DOI).
    """
    check_object(obj)
    compiled = choose_engine(engine, HAS_COMPILED_KERNEL, "tc_winds")
    radii = _radii_of(radii, min_radius, max_radius, radius_step)
    cfg = _fit_config(ring_width, max_wavenumber, min_points, max_gap, mean_wind)
    if int(guess_grid) != guess_grid or guess_grid < 1:
        raise ValueError(f"guess_grid must be a positive integer, not {guess_grid!r}")
    for name, value in (
        ("guess_spacing", guess_spacing),
        ("search_radius", search_radius),
    ):
        if value is not None and not value > 0:
            raise ValueError(f"{name} must be positive, not {value!r}")
    if not inflection_window >= 0:
        raise ValueError("inflection_window must not be negative")
    if not (np.isfinite(k_e) and k_e > 0):
        raise ValueError(f"k_e must be positive, not {k_e!r}")

    def one(ds, _name):
        sweep = _Sweep(ds, field, k_e)
        opts = {
            "first_guess": _guess_of(first_guess, first_guess_latlon, sweep),
            "guess_grid": int(guess_grid),
            "guess_spacing": guess_spacing,
            "search_radius": search_radius,
            "inflection_window": float(inflection_window),
            "step": _SIMPLEX_STEP,
            "tolerance": 50.0,
            "max_iter": 200,
        }

        def block(vh, x, y, w):
            return _center_block(
                vh,
                x,
                y,
                w,
                radii=radii,
                cfg=cfg,
                opts=opts,
                compiled=compiled,
                n_threads=n_threads,
            )

        out = _stack(sweep, block, _CENTER_ROWS, radii.size)
        ds_out = _center_dataset(sweep, out, radii)
        ds_out.attrs.update(
            {
                "tc_center_ring_width": float(ring_width),
                "tc_center_max_wavenumber": int(max_wavenumber),
                "tc_center_mean_wind": cfg["mean_wind"],
            }
        )
        return ds_out

    return _per_sweep(obj, one)


# --------------------------------------------------------------------------
# GBVTD
# --------------------------------------------------------------------------


def _derived(coef, diag, cx, cy, radii, mean_wind):
    """Wind quantities of the rows of ``_GBVTD_ROWS`` from the ring coefficients."""
    a0, a1, b1, a2, b2, a3, b3 = (coef[..., i] for i in range(_NCOEF))
    rt = np.hypot(cx, cy)
    um, vm = mean_wind
    with np.errstate(invalid="ignore", divide="ignore"):
        sin_a = radii / rt
        cos_a = np.sqrt(1.0 - sin_a * sin_a)
        valid = sin_a < 1.0
        cross = (cy * um - cx * vm) / rt  # V_M sin(theta_T - theta_M)
        vt0 = -b1 - b3 - cross * sin_a
        vr0 = a1 + a3
        vt1c = -2.0 * b2
        vt1s = a2 - a0 + (a0 + a2) * cos_a
        vt2c = -2.0 * b3
        vt2s = 2.0 * a3
    rows = np.stack(
        [vt0, vr0, vt1c, vt1s, vt2c, vt2s, a0 + a2]
        + [diag[..., 0], diag[..., 1], diag[..., 2], diag[..., 3]]
    )
    order = diag[..., 2]
    rows[6:7, order < 2] = np.nan  # the along-beam wind needs A2
    rows[2:4, order < 2] = np.nan  # the wavenumber-1 wind needs B2 and A2
    rows[4:6, order < 3] = np.nan  # the wavenumber-2 wind needs A3 and B3
    rows[:, ~valid] = np.nan
    return rows


def _gbvtd_block(vh, x, y, w, *, center, radii, cfg, compiled, n_threads):
    gates = _gate_arrays(vh, x, y, w)
    cx, cy = center
    nr = len(radii)
    if gates[0].size == 0:
        return np.full((len(_GBVTD_ROWS), nr), np.nan)
    gates = _crop(gates, center, float(radii.max()) + cfg["half_width"] + 1.0)
    rt = np.hypot(cx, cy)
    usable = radii < rt * 0.9999
    coef = np.full((1, nr, _NCOEF), np.nan)
    diag = np.full((1, nr, 4), np.nan)
    if usable.any():
        c, d = _rings(gates, [cx], [cy], radii[usable], cfg, compiled, n_threads)
        coef[:, usable], diag[:, usable] = c, d
    return _derived(coef[0], diag[0], cx, cy, radii, cfg["mean_wind"])


def _peak(radii, values):
    """Radius and value of the maximum of a profile, refined by a parabola."""
    out = np.array([np.nan, np.nan])
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return out
    k = int(np.nanargmax(np.where(finite, values, -np.inf)))
    if k == 0 or k == len(values) - 1 or not (finite[k - 1] and finite[k + 1]):
        return out
    y0, y1, y2 = values[k - 1], values[k], values[k + 1]
    # y1 is the first maximum, so y0 < y1 >= y2 and the denominator is negative
    shift = 0.5 * (y0 - y2) / (y0 - 2.0 * y1 + y2)
    step = radii[k + 1] - radii[k]
    return np.array([radii[k] + shift * step, y1 - 0.25 * (y0 - y2) * shift])


def _center_of(center, center_latlon, sweep, name):
    """``(x, y)`` m of a centre given as a pair, a ``tc_center`` result or lat/lon."""
    if center is not None and center_latlon is not None:
        raise ValueError("give center or center_latlon, not both")
    if center_latlon is not None:
        if sweep.lat is None or sweep.lon is None:
            raise ValueError("the sweep has no latitude and longitude of the radar")
        lat, lon = (float(v) for v in center_latlon)
        return _latlon_to_offset(lat, lon, sweep.lat, sweep.lon)
    if center is None:
        raise ValueError("give the centre: center=(x, y) or center_latlon=(lat, lon)")
    center = node_arg(center, name) if name is not None else center
    if isinstance(center, xr.DataTree):
        center = center.to_dataset()
    if isinstance(center, xr.Dataset):
        if "CENTER_X" not in center or "CENTER_Y" not in center:
            raise ValueError(
                "the dataset has no CENTER_X and CENTER_Y (a tc_center result)"
            )
        x, y = float(center["CENTER_X"]), float(center["CENTER_Y"])
    else:
        try:
            x, y = (float(v) for v in center)
        except (TypeError, ValueError) as err:
            raise ValueError(
                "center must be (x, y) in m, or the result of tc_center"
            ) from err
    if not (np.isfinite(x) and np.isfinite(y)):
        raise ValueError("the centre is not finite (the centre finding failed?)")
    if x == 0.0 and y == 0.0:
        raise ValueError("the centre is at the radar")
    return x, y


@dask_support("lazy_only")
@renamed_arguments(velocity="field")
@provenance(
    "Ground-based velocity track display (GBVTD) of a tropical cyclone, with the "
    "closure of Lee et al. (1999)",
    extra_refs=("liou-2005",),
)
def gbvtd(
    obj,
    center=None,
    *,
    center_latlon=None,
    field=None,
    radii=None,
    min_radius=1000.0,
    max_radius=80000.0,
    radius_step=1000.0,
    ring_width=2000.0,
    max_wavenumber=3,
    mean_wind=None,
    min_points=30,
    max_gap=120.0,
    k_e=4.0 / 3.0,
    n_threads=None,
    engine="auto",
):
    """
    Axisymmetric and wavenumber-1 and 2 winds of a vortex from one radar.

    Fits Fourier series to the Doppler velocity on rings around ``center``
    (the ground-based velocity track display of Lee et al. 1999, equations in
    :mod:`radarx.retrieve.tc_winds`) with the closure that the asymmetric
    radial wind is zero.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        A dealiased plan position indicator of the Doppler velocity, see
        :func:`tc_center`. For a volume every sweep is processed.
    center : tuple of float or xarray.Dataset or xarray.DataTree, optional
        ``(x, y)`` of the circulation centre in m east and north of the radar,
        or the result of :func:`tc_center` (for a volume, its tree). A lazy
        ``tc_center`` result is computed.
    center_latlon : tuple of float, optional
        ``(latitude, longitude)`` of the centre instead of ``center``.
    field : str, optional
        Name of the velocity field. Default: ``VRADH``, ``VRAD`` and the like.
    radii : array-like, optional
        Ring radii (m). Default ``min_radius`` to ``max_radius`` in steps of
        ``radius_step``; a ring that does not fit (the radar inside it, too few
        gates or a wide gap) is NaN.
    min_radius, max_radius, radius_step : float, optional
        Default radii (m): 1 to 80 km in 1 km steps.
    ring_width : float, optional
        Full width of the triangular weight of a ring (m). Default 2000.
    max_wavenumber : int, optional
        Highest Fourier order of the fit (1 to 3), reduced where the data have
        wide gaps. Default 3, which gives the wavenumbers 0 to 2 of the vortex.
    mean_wind : tuple of float, optional
        ``(u, v)`` of the mean wind or storm motion (m s-1). Its cross-beam
        part is removed from ``VT0`` (Eq. 6 of Liou et al. 2005); without it
        the cross-beam mean wind is aliased into ``VT0``, with an error that
        grows with the radius as ``sin(alpha_max)``. The winds are relative to
        the mean wind. Default None: not removed.
    min_points : int, optional
        Fewest gates in a ring. Default 30.
    max_gap : float, optional
        Largest angular gap (degrees) of the data of a ring. Default 120.
    k_e : float, optional
        Effective Earth radius factor of the beam geometry. Default 4/3.
    n_threads : int, optional
        Threads of the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation. ``"auto"`` (default) prefers the compiled kernel.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        On the dimension ``radius`` (m): ``VT0`` and ``VR0`` (axisymmetric
        tangential and radial wind, m s-1), ``VT1_COS`` and ``VT1_SIN`` (the
        wavenumber-1 tangential wind in the angle ``psi`` of the fit),
        ``VT1_AMPLITUDE`` (m s-1) and ``VT1_AZIMUTH`` (degrees clockwise from
        north, seen from the centre, where the wavenumber-1 tangential wind is
        largest), ``VT2_COS``, ``VT2_SIN`` and ``VT2_AMPLITUDE``, ``VM_ALONG``
        (the along-beam mean wind of the fit), and ``N_POINTS``, ``DATA_GAP``,
        ``FIT_ORDER`` and ``FIT_RMS`` of the fits. Scalars: ``CENTER_X``,
        ``CENTER_Y`` (m), ``CENTER_LATITUDE``, ``CENTER_LONGITUDE``, ``RMW``
        (the radius of the maximum of ``VT0`` refined with a parabola, NaN if
        the maximum is at the edge of the radii) and ``VT_MAX``. For a volume
        a DataTree with one node per sweep. The input is not changed.

    Raises
    ------
    ValueError
        For invalid options or a sweep without the geometry, or a centre that
        is not finite or at the radar.
    TypeError
        If ``obj`` is not an xarray object.
    ImportError
        If ``engine="compiled"`` and the kernel is not available.

    Notes
    -----
    The equations are from the conference paper of Liou et al. (2005) and
    Chen et al. (2013); the paper of Lee et al. (1999) was not read and the
    code is not checked against it. The retrieval is sensitive to the centre
    (see :mod:`radarx.retrieve.tc_winds`). The wavenumber-1 radial wind is
    assumed zero; a real one leaks into the wavenumber-1 tangential wind.

    References
    ----------
    .. [1] Lee, W.-C., B. J.-D. Jou, P.-L. Chang, and S.-M. Deng, 1999: Tropical
       cyclone kinematic structure retrieved from single-Doppler radar
       observations. Part I: Interpretation of Doppler velocity patterns and
       the GBVTD technique. *Mon. Wea. Rev.*, **127** (10), 2419-2439,
       https://doi.org/10.1175/1520-0493(1999)127<2419:TCKSRF>2.0.CO;2
    .. [2] Chen, X., K. Zhao, W.-C. Lee, B. J.-D. Jou, M. Xue, and P. R.
       Harasti, 2013: The improvement to the environmental wind and tropical
       cyclone circulation retrievals with the modified GBVTD (MGBVTD)
       technique. *J. Appl. Meteor. Climatol.*, **52** (11), 2493-2508,
       https://doi.org/10.1175/JAMC-D-13-031.1
    .. [3] Liou, Y.-C., T.-C. C. Wang, W.-C. Lee, and Y.-J. Chang, 2005: The
       extended GBVTD and the recovery of the tropical cyclones asymmetric
       structures. *Preprints, 32nd Conf. on Radar Meteorology*, Albuquerque,
       NM, Amer. Meteor. Soc., JP2J.8 (conference paper, no DOI).
    .. [4] Lee, W.-C., P. R. Harasti, M. Bell, B. J.-D. Jou, and M.-H. Chang,
       2006: Doppler velocity signatures of idealized elliptical vortices.
       *Terr. Atmos. Ocean. Sci.*, **17** (2), 429-446,
       https://doi.org/10.3319/TAO.2006.17.2.429(A)
    """
    check_object(obj)
    compiled = choose_engine(engine, HAS_COMPILED_KERNEL, "tc_winds")
    radii = _radii_of(radii, min_radius, max_radius, radius_step)
    cfg = _fit_config(ring_width, max_wavenumber, min_points, max_gap, mean_wind)
    if not (np.isfinite(k_e) and k_e > 0):
        raise ValueError(f"k_e must be positive, not {k_e!r}")

    def one(ds, name):
        sweep = _Sweep(ds, field, k_e)
        cx, cy = _center_of(center, center_latlon, sweep, name)

        def block(vh, x, y, w):
            return _gbvtd_block(
                vh,
                x,
                y,
                w,
                center=(cx, cy),
                radii=radii,
                cfg=cfg,
                compiled=compiled,
                n_threads=n_threads,
            )

        out = _stack(sweep, block, _GBVTD_ROWS, radii.size)
        return _gbvtd_dataset(sweep, out, radii, (cx, cy), cfg)

    return _per_sweep(obj, one)


def _gbvtd_dataset(sweep, out, radii, center, cfg):
    ds = _variables(out, _GBVTD_ROWS, radii)
    cx, cy = center
    sin_a = ds["radius"] / np.hypot(cx, cy)
    psi = np.arctan2(ds["VT1_SIN"], ds["VT1_COS"])
    ds["VT1_AMPLITUDE"] = np.hypot(ds["VT1_COS"], ds["VT1_SIN"])
    # azimuth of the maximum (mathematical angle of the point seen from the centre)
    phi = np.arctan2(cy, cx) + psi + np.arcsin(sin_a * np.sin(psi))
    ds["VT1_AZIMUTH"] = (90.0 - phi / _DEG) % 360.0
    ds["VT2_AMPLITUDE"] = np.hypot(ds["VT2_COS"], ds["VT2_SIN"])
    peak = xr.apply_ufunc(
        lambda r, v: _peak(r, v),
        ds["radius"],
        ds["VT0"],
        input_core_dims=[["radius"], ["radius"]],
        output_core_dims=[["peak"]],
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {"peak": 2}, "allow_rechunk": True},
    )
    ds["RMW"] = peak.isel(peak=0, drop=True)
    ds["VT_MAX"] = peak.isel(peak=1, drop=True)
    ds["CENTER_X"] = xr.DataArray(cx)
    ds["CENTER_Y"] = xr.DataArray(cy)
    if sweep.lat is not None and sweep.lon is not None:
        lat, lon = _offset_to_latlon(cx, cy, sweep.lat, sweep.lon)
        ds["CENTER_LATITUDE"] = xr.DataArray(
            float(lat), attrs={"units": "degrees_north"}
        )
        ds["CENTER_LONGITUDE"] = xr.DataArray(
            float(lon), attrs={"units": "degrees_east"}
        )
    for name, (long_name, units) in _UNITS.items():
        ds[name].attrs = {"long_name": long_name, "units": units}
    ds["VT1_AMPLITUDE"].attrs = {
        "long_name": "Amplitude of the wavenumber-1 tangential wind",
        "units": "m s-1",
    }
    ds["VT2_AMPLITUDE"].attrs = {
        "long_name": "Amplitude of the wavenumber-2 tangential wind",
        "units": "m s-1",
    }
    ds["VT1_AZIMUTH"].attrs = {
        "long_name": "Azimuth of the maximum wavenumber-1 tangential wind",
        "units": "degree",
        "comment": "clockwise from north, seen from the centre",
    }
    ds["RMW"].attrs = {"long_name": "Radius of maximum wind", "units": "m"}
    ds["VT_MAX"].attrs = {
        "long_name": "Maximum of the axisymmetric tangential wind",
        "units": "m s-1",
    }
    ds["CENTER_X"].attrs = {
        "long_name": "Centre, metres east of the radar",
        "units": "m",
    }
    ds["CENTER_Y"].attrs = {
        "long_name": "Centre, metres north of the radar",
        "units": "m",
    }
    ds.attrs.update(
        {
            "gbvtd_ring_width": 2.0 * cfg["half_width"],
            "gbvtd_max_wavenumber": cfg["max_order"],
            "gbvtd_mean_wind": cfg["mean_wind"],
            "gbvtd_closure": "asymmetric radial wind neglected",
        }
    )
    return ds


# --------------------------------------------------------------------------
# symmetric and asymmetric decomposition
# --------------------------------------------------------------------------


def _psi_of_azimuth(phi, theta_t, sin_a):
    """Angle ``psi`` of the points at the mathematical azimuth ``phi`` (radians)."""
    delta = np.mod(phi - theta_t + np.pi, _TWO_PI) - np.pi  # in [-pi, pi)
    delta = np.broadcast_to(delta, np.broadcast(delta, sin_a).shape)
    psi = delta.copy()
    for _ in range(30):
        g = psi + np.arcsin(sin_a * np.sin(psi)) - delta
        dg = 1.0 + sin_a * np.cos(psi) / np.sqrt(1.0 - (sin_a * np.sin(psi)) ** 2)
        psi = psi - g / dg
    return psi


def _decompose(ds, azimuth):
    cx, cy = float(ds["CENTER_X"]), float(ds["CENTER_Y"])
    radius = np.asarray(ds["radius"].values, dtype=np.float64)
    rt = np.hypot(cx, cy)
    sin_a = (radius / rt)[:, None]
    az = np.asarray(azimuth, dtype=np.float64)
    phi = (90.0 - az[None, :]) * _DEG  # mathematical angle seen from the centre
    with np.errstate(invalid="ignore"):
        psi = _psi_of_azimuth(phi, np.arctan2(cy, cx), sin_a)
    coef = {
        k: np.asarray(ds[k].values, dtype=np.float64)[:, None] for k in _GBVTD_ROWS[:6]
    }
    vt0 = np.broadcast_to(coef["VT0"], psi.shape)
    wn1 = coef["VT1_COS"] * np.cos(psi) + coef["VT1_SIN"] * np.sin(psi)
    wn2 = coef["VT2_COS"] * np.cos(2 * psi) + coef["VT2_SIN"] * np.sin(2 * psi)
    vr = np.broadcast_to(coef["VR0"], psi.shape)
    vt = vt0 + wn1 + wn2
    return phi, vt0, wn1, wn2, vt, vr


@dask_support("lazy_only")
@provenance(
    "Symmetric and wavenumber-1 and 2 asymmetric parts of the GBVTD winds on a "
    "polar grid, with the psi to azimuth mapping of the GBVTD geometry",
    extra_refs=("liou-2005",),
)
def tc_decompose(result, *, azimuth=None):
    """
    Winds of a GBVTD retrieval on a polar grid, split by wavenumber.

    The coefficients of :func:`gbvtd` belong to the angle ``psi`` of the ring
    fit; they are evaluated at the physical azimuths with the exact relation
    between ``psi`` and the azimuth seen from the centre (geometry of
    :mod:`radarx.retrieve.tc_winds`, not checked against Lee et al. 1999).

    Parameters
    ----------
    result : xarray.Dataset or xarray.DataTree
        The result of :func:`gbvtd` (for a volume, its tree). Dask-backed
        results are computed.
    azimuth : array-like, optional
        Azimuths (degrees clockwise from north, seen from the centre). Default
        every 5 degrees.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        On the dimensions ``radius`` and ``azimuth``: ``VT_SYMMETRIC`` (the
        axisymmetric tangential wind ``VT0``), ``VT_WAVENUMBER1`` and
        ``VT_WAVENUMBER2`` (the asymmetric parts), ``VT`` (their sum), ``VR``
        (the axisymmetric radial wind: the asymmetric radial wind is not
        retrieved) and the Cartesian winds ``U`` and ``V`` relative to the mean
        wind and the positions ``X`` and ``Y`` (m east and north of the radar).
        On ``radius``: ``ASYMMETRY`` (the root mean square of the asymmetric
        tangential wind over the azimuths divided by ``VT0``). On ``azimuth``:
        ``RMW_AZIMUTH``, the radius of the maximum of ``VT`` at the azimuth
        (the radius of maximum wind of ``VT0`` per azimuth, NaN at the edge of
        the radii). Scalars ``CENTER_X`` and ``CENTER_Y`` and the radius of
        maximum wind of ``VT0``, ``RMW``.

    Raises
    ------
    ValueError
        If ``result`` is not a result of :func:`gbvtd`.
    TypeError
        If ``result`` is not a Dataset or a DataTree.

    References
    ----------
    .. [1] Lee, W.-C., B. J.-D. Jou, P.-L. Chang, and S.-M. Deng, 1999: Tropical
       cyclone kinematic structure retrieved from single-Doppler radar
       observations. Part I: Interpretation of Doppler velocity patterns and
       the GBVTD technique. *Mon. Wea. Rev.*, **127** (10), 2419-2439,
       https://doi.org/10.1175/1520-0493(1999)127<2419:TCKSRF>2.0.CO;2
    .. [2] Liou, Y.-C., T.-C. C. Wang, W.-C. Lee, and Y.-J. Chang, 2005: The
       extended GBVTD and the recovery of the tropical cyclones asymmetric
       structures. *Preprints, 32nd Conf. on Radar Meteorology*, Albuquerque,
       NM, Amer. Meteor. Soc., JP2J.8 (conference paper, no DOI).
    """
    if isinstance(result, xr.DataTree):
        nodes = {
            name: _decompose_one(node_dataset(result[name]), azimuth)
            for name in _sweep_names(result)
        }
        if not nodes:
            raise ValueError("the volume has no sweeps")
        return product_tree(result, nodes)
    if not isinstance(result, xr.Dataset):
        raise TypeError("result must be an xarray Dataset or DataTree from gbvtd")
    return _decompose_one(result, azimuth)


def _decompose_one(ds, azimuth):
    needed = (
        "VT0",
        "VR0",
        "VT1_COS",
        "VT1_SIN",
        "VT2_COS",
        "VT2_SIN",
        "CENTER_X",
        "CENTER_Y",
    )
    missing = [n for n in needed if n not in ds]
    if missing:
        raise ValueError(f"not a gbvtd result, missing {missing}")
    az = (
        np.arange(0.0, 360.0, 5.0)
        if azimuth is None
        else np.asarray(azimuth, dtype=np.float64)
    )
    if az.ndim != 1 or az.size == 0:
        raise ValueError("azimuth must be a non-empty 1-D array (degrees)")
    ds = ds.compute()
    phi, vt0, wn1, wn2, vt, vr = _decompose(ds, az)
    radius = np.asarray(ds["radius"].values)
    cx, cy = float(ds["CENTER_X"]), float(ds["CENTER_Y"])
    xx = cx + radius[:, None] * np.cos(phi)
    yy = cy + radius[:, None] * np.sin(phi)
    u = vr * np.cos(phi) - vt * np.sin(phi)
    v = vr * np.sin(phi) + vt * np.cos(phi)
    asym = np.sqrt(np.mean((wn1 + wn2) ** 2, axis=1)) / np.abs(vt0[:, 0])
    rmw_az = np.array([_peak(radius, vt[:, j])[0] for j in range(az.size)])
    dims = ("radius", "azimuth")
    out = xr.Dataset(
        {
            "VT_SYMMETRIC": (dims, vt0),
            "VT_WAVENUMBER1": (dims, wn1),
            "VT_WAVENUMBER2": (dims, wn2),
            "VT": (dims, vt),
            "VR": (dims, vr),
            "U": (dims, u),
            "V": (dims, v),
            "X": (dims, xx),
            "Y": (dims, yy),
            "ASYMMETRY": (("radius",), asym),
            "RMW_AZIMUTH": (("azimuth",), rmw_az),
            "CENTER_X": ((), cx),
            "CENTER_Y": ((), cy),
            "RMW": ((), float(ds["RMW"])) if "RMW" in ds else ((), np.nan),
        },
        coords={
            "radius": ("radius", radius, {"units": "m"}),
            "azimuth": ("azimuth", az, {"units": "degree"}),
        },
    )
    for name in (
        "VT_SYMMETRIC",
        "VT_WAVENUMBER1",
        "VT_WAVENUMBER2",
        "VT",
        "VR",
        "U",
        "V",
    ):
        out[name].attrs = {"units": "m s-1"}
    for name in ("X", "Y", "RMW_AZIMUTH", "CENTER_X", "CENTER_Y", "RMW"):
        out[name].attrs = {"units": "m"}
    out["ASYMMETRY"].attrs = {
        "long_name": "Asymmetric tangential wind relative to the axisymmetric wind",
        "units": "1",
    }
    out["VT_WAVENUMBER1"].attrs["long_name"] = "Wavenumber-1 tangential wind"
    out["VT_WAVENUMBER2"].attrs["long_name"] = "Wavenumber-2 tangential wind"
    out["VT_SYMMETRIC"].attrs["long_name"] = "Axisymmetric tangential wind"
    return out


# --------------------------------------------------------------------------
# accessors
# --------------------------------------------------------------------------


@accessor_method("dataarray", "dataset", "datatree", name="tc_center", wraps=tc_center)
def _tc_center_accessor(self):
    """
    Circulation centre and radius of maximum wind of a tropical cyclone.

    See :func:`radarx.retrieve.tc_center` for the parameters.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        The centre, the radius of maximum wind and the profiles of the search.
    """


@accessor_method("dataarray", "dataset", "datatree", name="gbvtd", wraps=gbvtd)
def _gbvtd_accessor(self):
    """
    Axisymmetric and wavenumber-1 and 2 winds of a vortex from one radar (GBVTD).

    See :func:`radarx.retrieve.gbvtd` for the parameters.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        The ring winds, the radius of maximum wind and the fit diagnostics.
    """
