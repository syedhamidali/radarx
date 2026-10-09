#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Raindrop Trajectories and Size Sorting
======================================

Lagrangian model of raindrops falling from a radar gate, or from any point of
a three-dimensional grid, to the ground. Drops of different size fall at
different speeds through a sheared, time-varying wind and so reach the ground
at different places and times (size sorting; Kumjian and Ryzhkov 2012, Dawson
et al. 2015). The model integrates the path of one drop per source point, size
bin and (optionally) turbulent ensemble member, together with the shrinking of
the drop by evaporation, and returns where and when it lands and how much of
it is left. The same equations run backward in time from a surface instrument
(for example a disdrometer) to the source point and time aloft of each size
bin.

Equations
---------
A drop of equal-volume diameter :math:`D` follows the air horizontally and
falls through it at its terminal speed (inertia is neglected, as to my
knowledge in Dawson et al. 2015, whose paper was not available to check; the statement that drops reach their terminal speed within
about a second is a rule of thumb that was not checked against a paper
here),

.. math::

    \\frac{dx}{dt} = u + u',\\quad \\frac{dy}{dt} = v + v',\\quad
    \\frac{dz}{dt} = w + w' - V_t(D, \\rho),\\quad
    \\frac{d(D^2)}{dt} = 2 D \\dot D_{\\mathrm{evap}}(D, V_t, T, p, q_v),

where :math:`(u, v, w)` is the wind at the position of the drop and
:math:`(u', v', w')` an optional turbulent perturbation. The squared diameter
is integrated instead of :math:`D` (a radarx numerical choice) because :math:`\\dot D \\propto 1/D` is
singular for vanishing drops while :math:`d(D^2)/dt` stays smooth, so the
integration is stable up to complete evaporation.

Fall speed
----------
By default :math:`V_t = V_0(D) (\\rho_0/\\rho)^{0.4}` with the sea-level speed
of Atlas et al. (1973), :math:`V_0 = 9.65 - 10.3 e^{-0.6 D}` m s\\ :sup:`-1`
(:math:`D` in mm, at least zero, negative below 0.109 mm; the same as
:func:`radarx.retrieve.terminal_fall_speed`, where the source of the formula
and the unverified validity range are discussed), and the air-density
correction of Foote and du Toit (1969) as quoted by Li and Srivastava (2001)
and Kumjian and Ryzhkov (2010, Eq. 3), where :math:`\\rho` is the density of
moist air at the height of the drop and :math:`\\rho_0` = 1.204 kg
m\\ :sup:`-3` (dry air at 1013.25 hPa and 20 °C, a radarx choice).
``fall_speed=(a, b, f)`` gives :math:`V_0 = a D^b e^{-f D}` (the form of
:func:`radarx.retrieve.drop_evaporation_rate`) and
``fall_speed=("polynomial", c0, c1, ...)`` the polynomial
:math:`\\sum_k c_k D^k`, so that other published relations can be used by
giving their coefficients. The relations of Beard (1976) and Gunn and Kinzer
(1949) are not included.

Wind
----
The wind comes from a three-dimensional analysis (``wind``, e.g. the output
of :func:`radarx.retrieve.multi_doppler`, on ``(z, y, x)`` or
``(time, z, y, x)``) interpolated trilinearly in space and linearly in time
between the analyses (held constant before the first and after the last
time). With ``storm_motion`` :math:`\\mathbf c` every analysis is a frozen
pattern that moves with the storm from its own valid time :math:`t_a`: the
wind at :math:`(\\mathbf x, t)` is the blend, linear in time, of the two
analyses around :math:`t` read at :math:`\\mathbf x - \\mathbf c\\,(t - t_a)`
(the moving frame of reference of Gal-Chen 1982, as in
:func:`radarx.retrieve.interpolate_time`). A wind analysis without a time
coordinate is valid at ``pattern_time``. Wherever a drop leaves the
horizontal extent of the grid, rises above its top or a neighbouring value is
missing, the wind of a background profile (``profile``, a sounding or ERA5
profile from :mod:`radarx.io.sounding`) is used, and below the lowest grid
level the wind of that level. Without a wind analysis the profile alone is
used, a layer wind that varies with height only. Without a profile the
background is the horizontal mean wind of the analysis.

Evaporation
-----------
With humidity in the profile, every drop evaporates by ventilated diffusion
of water vapour exactly as in :func:`radarx.retrieve.drop_evaporation_rate`
(Kumjian and Ryzhkov 2010; Appendix Eqs. A1-A10, with the ventilation
coefficient of Pruppacher and Klett 1997 as in
:mod:`radarx.retrieve.evaporation`, where the departures from the paper
and the 1000 hPa reference pressure of :math:`D_v` are described), with the
fall speed of the drop at its current size and height in the ventilation
coefficient; the saturation vapour pressure is that of Buck (1981, Eq. 8,
without the enhancement factor). The temperature and humidity of the profile
are not changed by the evaporation (no cooling or moistening feedback; see
:func:`radarx.retrieve.integrate_evaporation` for that). A drop smaller than
``evaporated_diameter`` has evaporated.

Time stepping
-------------
The equations are integrated with fixed steps of the classical fourth-order
Runge-Kutta method (``scheme="rk4"``) or Heun's method (``"rk2"``). The step
in which the drop would pass the surface (or the sloping stop surface) is
repeated with a shortened step: the cubic Hermite interpolant of the step
gives the first estimate of its length, and three Newton iterations with real
Runge-Kutta steps then end the path exactly on the surface, so the landing
point and time are as accurate as the scheme even though the environment is
not defined below the surface. A drop that shrinks below the minimum size is
placed with the Hermite interpolant. The tests show fourth order convergence
of the landing point (second order for Heun's method).

Concentration
-------------
Along the path the concentration of the drop population per unit diameter,
:math:`n(D)`, follows from the continuity equation
:math:`\\partial_t n + \\nabla\\cdot(n \\mathbf v) + \\partial_D (n \\dot D) = 0`,
i.e. :math:`d \\ln n / dt = -\\nabla\\cdot\\mathbf v - \\partial_D \\dot D`
along the drop, with :math:`\\mathbf v = (u, v, w - V_t)`. Without wind
divergence (the default; ``wind_divergence=True`` adds
:math:`\\partial_x u + \\partial_y v + \\partial_z w` of the wind analysis,
which is noisy in real analyses) this conserves the number flux
:math:`n V_t` (so that :math:`n` grows as :math:`(\\rho/\\rho_0)^{0.4}` toward
the ground, the correction used with the constant-wind model) and accounts
for the change of the width of a size bin by evaporation. The result is the
ratio ``concentration_ratio`` of the concentration at the later point of the
path to that at the earlier one.

Turbulent dispersion
--------------------
With ``dispersion=(sigma_h, sigma_w, timescale)`` every drop is released
``members`` times with velocity perturbations that follow a first-order
autoregressive (Langevin) process for homogeneous, stationary Gaussian
turbulence (Thomson 1987; Wilson and Sawford 1996: the discretization is the standard
one of these reviews, their equations were not checked here, and the default
parameters are the user's choice, there are none from a paper),
:math:`u'_{n+1} = a u'_n + \\sqrt{1 - a^2}\\,\\sigma \\xi_n` with
:math:`a = e^{-\\Delta t / T_L}`, standard deviations ``sigma_h`` (horizontal)
and ``sigma_w`` (vertical) and Lagrangian time scale ``timescale``. The random
numbers are a counter-based hash of ``seed`` and the drop, so results do not
depend on the number of threads or on the engine.

Outputs and applications
------------------------
:func:`rain_trajectories` returns, for every source point and size bin, the
landing position and time, the diameter at landing, the evaporated mass
fraction and the concentration ratio. :func:`size_sorting` expresses the
landing of each size relative to a reference size (displacement and arrival
time offset). :func:`surface_dsd` accumulates the landed drops of a gridded or
radial (sweep) source onto a surface grid in space and time, conserving the
number flux. :func:`rain_source_points` integrates backward from surface
points (disdrometers) to the source level, optionally sampling a source DSD
there, and :func:`trajectory_matched_times` finds, for every size bin, the
time at which drops from a point of the moving echo pattern reach a surface
site: the trajectory-pair construction of
``ml/models/bayesian_dsd/match.py`` for a uniform wind without evaporation,
generalised to variable winds and evaporation.

Not included
------------
Collisional growth and breakup are not modelled: they change the number of
drops in each size bin through the interaction of all sizes in the same
volume, which independent per-size trajectories do not carry (it takes a
bin-microphysics column or grid model). Melting is not modelled either, so
sources must be in rain below the melting layer. The horizontal convergence
of the drop flow is accounted for only through ``wind_divergence`` and
:func:`surface_dsd`. The surface is a horizontal plane (``surface_height``) or a height field on
a regular ``(y, x)`` grid (``terrain``); it has no other effect on the wind.

The trajectories of all drops (source points x sizes x members) are
integrated in one call of a compiled kernel
(``radarx.retrieve._rain_trajectories``, multithreaded with dynamic
scheduling and the GIL released) with an identical NumPy reference as
fallback and test oracle.

References
----------
Dawson, D. T., E. R. Mansell, and M. R. Kumjian, 2015: Does wind shear cause
hydrometeor size sorting? *J. Atmos. Sci.*, **72** (1), 340-348,
https://doi.org/10.1175/JAS-D-14-0084.1

Kumjian, M. R., and A. V. Ryzhkov, 2012: The impact of size sorting on the
polarimetric radar variables. *J. Atmos. Sci.*, **69** (6), 2042-2060,
https://doi.org/10.1175/JAS-D-11-0125.1

Atlas, D., R. C. Srivastava, and R. S. Sekhon, 1973: Doppler radar
characteristics of precipitation at vertical incidence. *Rev. Geophys.*,
**11** (1), 1-35, https://doi.org/10.1029/RG011i001p00001

Foote, G. B., and P. S. du Toit, 1969: Terminal velocity of raindrops aloft.
*J. Appl. Meteor.*, **8** (2), 249-253,
https://doi.org/10.1175/1520-0450(1969)008<0249:TVORA>2.0.CO;2

Kumjian, M. R., and A. V. Ryzhkov, 2010: The impact of evaporation on
polarimetric characteristics of rain: Theoretical model and practical
implications. *J. Appl. Meteor. Climatol.*, **49** (6), 1247-1267,
https://doi.org/10.1175/2010JAMC2243.1

Buck, A. L., 1981: New equations for computing vapor pressure and enhancement
factor. *J. Appl. Meteor.*, **20** (12), 1527-1532,
https://doi.org/10.1175/1520-0450(1981)020<1527:NEFCVP>2.0.CO;2

Thomson, D. J., 1987: Criteria for the selection of stochastic models of
particle trajectories in turbulent flows. *J. Fluid Mech.*, **180**, 529-556,
https://doi.org/10.1017/S0022112087001940

Wilson, J. D., and B. L. Sawford, 1996: Review of Lagrangian stochastic
models for trajectories in the turbulent atmosphere. *Bound.-Layer Meteor.*,
**78** (1-2), 191-210, https://doi.org/10.1007/BF00122492

Beard, K. V., 1976: Terminal velocity and shape of cloud and precipitation
drops aloft. *J. Atmos. Sci.*, **33** (5), 851-864,
https://doi.org/10.1175/1520-0469(1976)033<0851:TVASOC>2.0.CO;2

Gunn, R., and G. D. Kinzer, 1949: The terminal velocity of fall for water
droplets in stagnant air. *J. Meteor.*, **6** (4), 243-248,
https://doi.org/10.1175/1520-0469(1949)006<0243:TTVOFF>2.0.CO;2

Li, X., and R. C. Srivastava, 2001: An analytical solution for raindrop
evaporation and its application to radar rainfall measurements. *J. Appl.
Meteor.*, **40** (9), 1607-1616,
https://doi.org/10.1175/1520-0450(2001)040<1607:AASFRE>2.0.CO;2

Pruppacher, H. R., and J. D. Klett, 1997: *Microphysics of Clouds and
Precipitation*. 2nd rev. and enl. ed., Kluwer Academic Publishers (reprinted
by Springer, 2010), https://doi.org/10.1007/978-0-306-48100-0

Gal-Chen, T., 1982: Errors in fixed and moving frame of references:
Applications for conventional and Doppler radar analysis. *J. Atmos. Sci.*,
**39** (10), 2279-2300,
https://doi.org/10.1175/1520-0469(1982)039<2279:EIFAMF>2.0.CO;2

.. autosummary::
   :nosignatures:
   :toctree: generated/

   rain_trajectories
   rain_source_points
   size_sorting
   surface_dsd
   trajectory_matched_times
"""

from __future__ import annotations

__all__ = [
    "rain_trajectories",
    "rain_source_points",
    "size_sorting",
    "surface_dsd",
    "trajectory_matched_times",
]

import math
import warnings
from collections.abc import Mapping

import numpy as np
import xarray as xr

from .._registry import accessor_method
from . import _rain_trajectories_numpy as _ref

try:
    from . import _rain_trajectories

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _rain_trajectories = None
    HAS_COMPILED_KERNEL = False

# kg m-3, air density of the sea-level fall speeds: dry air at 1013.25 hPa and
# 20 degC (radarx choice, as in evaporation.py)
RHO0 = 1.204
# ventilation coefficient 0.78 + 0.308 N_Sc^(1/3) N_Re^(1/2): Pruppacher and
# Klett (1997) as printed in Kumjian and Ryzhkov (2010), Eq. A3, as in
# evaporation.py
VENTILATION = (0.78, 0.308)
SCHEMES = {"rk2": 2, "rk4": 4}
STATUS = {"aloft": 0, "reached": 1, "evaporated": 2, "invalid": 3}
_MAX_PATH_SAMPLES = 50_000_000

_STATUS_ATTRS = {
    "long_name": "Outcome of the trajectory",
    "flag_values": np.array([0, 1, 2, 3], dtype=np.int8),
    "flag_meanings": "aloft_at_max_time reached_stop_height evaporated invalid_input",
}


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def _use_compiled(engine):
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled rain trajectory kernel is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


def _f64(x):
    return np.ascontiguousarray(x, dtype=np.float64)


def _fall(fall_speed):
    """(kind, coefficients) of a fall-speed specification."""
    if isinstance(fall_speed, str):
        if fall_speed != "atlas1973":
            raise ValueError(
                f"fall_speed must be 'atlas1973', (a, b, f) or ('polynomial', "
                f"c0, c1, ...), not {fall_speed!r}"
            )
        return 0, np.zeros(1)
    seq = list(fall_speed)
    if seq and isinstance(seq[0], str):
        if seq[0] != "polynomial" or len(seq) < 2:
            raise ValueError("fall_speed polynomial needs ('polynomial', c0, ...)")
        return 2, _f64([float(c) for c in seq[1:]])
    if len(seq) != 3:
        raise ValueError("fall_speed must be 'atlas1973', (a, b, f) or a polynomial")
    a, b, f = (float(v) for v in seq)
    if not (a > 0 and b > -1 and f >= 0):
        raise ValueError("fall_speed (a, b, f) needs a > 0, b > -1 and f >= 0")
    return 1, _f64([a, b, f])


def _is_time(arr):
    return np.issubdtype(np.asarray(arr).dtype, np.datetime64)


def _seconds(t, epoch):
    """Seconds since ``epoch`` (datetime64) or the numbers themselves."""
    t = np.asarray(t)
    if np.issubdtype(t.dtype, np.datetime64):
        return (t - epoch) / np.timedelta64(1, "s")
    return t.astype(np.float64)


def _from_seconds(s, epoch):
    return epoch + (np.asarray(s, dtype=np.float64) * 1e9).astype("timedelta64[ns]")


def _as_dataset(obj, names):
    """A Dataset with the given names from a Dataset or a mapping of arrays."""
    if isinstance(obj, xr.Dataset):
        return obj
    if isinstance(obj, Mapping):
        arrays = {
            k: v if isinstance(v, xr.DataArray) else np.asarray(v)
            for k, v in obj.items()
        }
        sizes = {np.size(v) for v in arrays.values() if not isinstance(v, xr.DataArray)}
        sizes.discard(1)
        if len(sizes) > 1:
            raise ValueError("the arrays of the points must have the same length")
        n = sizes.pop() if sizes else 1
        data = {}
        for k, v in arrays.items():
            if isinstance(v, xr.DataArray):
                data[k] = v
            elif v.ndim == 0 and n == 1:
                data[k] = xr.DataArray(v)
            else:
                data[k] = ("point", np.broadcast_to(np.atleast_1d(v), (n,)).copy())
        return xr.Dataset(data)
    raise TypeError(f"expected an xarray.Dataset or a mapping with {names}")


def _component(ds, name):
    return ds[name] if name in ds else None


def _points(source, offset=(0.0, 0.0, 0.0), time=None):
    """Broadcast coordinates (m) and time of the source points."""
    ds = _as_dataset(source, ("x", "y", "z"))
    comps = []
    for name, off in zip(("x", "y", "z"), offset):
        c = _component(ds, name)
        if c is None:
            raise ValueError(f"the points need a {name!r} variable or coordinate")
        c = c.reset_coords(drop=True)
        comps.append(c + off if off else c)
    tcomp = _component(ds, "time")
    if time is not None:
        t = xr.DataArray(time)
    elif tcomp is not None:
        t = tcomp.reset_coords(drop=True)
    else:
        t = xr.DataArray(0.0)
    allc = xr.broadcast(*comps, t)
    return allc[:3], allc[3]


def _motion(storm_motion):
    """(cx, cy) of a storm motion given as a pair or a Dataset."""
    if storm_motion is None:
        return 0.0, 0.0
    if isinstance(storm_motion, xr.Dataset):
        if "u" not in storm_motion or "v" not in storm_motion:
            raise ValueError("storm_motion Dataset needs 'u' and 'v'")
        return float(storm_motion["u"].mean()), float(storm_motion["v"].mean())
    u, v = (float(c) for c in storm_motion)
    return u, v


def _profile_tables(profile):
    """Wind and thermodynamic tables from a profile Dataset, or Nones."""
    if profile is None:
        return None
    if not isinstance(profile, xr.Dataset) or "height" not in profile.coords:
        raise ValueError("profile must be a Dataset on a 'height' coordinate (m)")
    z = np.asarray(profile["height"].values, dtype=np.float64)

    def col(name):
        if name in profile:
            return np.asarray(profile[name].values, dtype=np.float64)
        return np.full(z.shape, np.nan)

    tab = {}
    # wind
    u, v, w = col("u"), col("v"), col("w")
    okw = np.isfinite(z) & np.isfinite(u) & np.isfinite(v)
    if okw.any():
        order = np.argsort(z[okw])
        zz = z[okw][order]
        keep = np.concatenate([[True], np.diff(zz) > 0])
        tab["wind"] = (
            zz[keep],
            u[okw][order][keep],
            v[okw][order][keep],
            np.nan_to_num(w[okw][order][keep]),
        )
    # thermodynamics
    temp, p = col("temperature"), col("pressure")
    okt = np.isfinite(z) & np.isfinite(temp) & np.isfinite(p) & (p > 0) & (temp > 0)
    if okt.any():
        order = np.argsort(z[okt])
        zz = z[okt][order]
        keep = np.concatenate([[True], np.diff(zz) > 0])
        zz = zz[keep]

        def pick(a):
            return a[okt][order][keep]

        tt, pp = pick(temp), pick(p)
        rh = pick(col("relative_humidity"))
        if not np.isfinite(rh).any():
            q = pick(col("specific_humidity"))
            td = pick(col("dewpoint"))
            if np.isfinite(q).any():
                e = q * pp / (_ref.EPS + (1.0 - _ref.EPS) * q)
                rh = e / _ref._esat(tt)
            elif np.isfinite(td).any():
                rh = _ref._esat(td) / _ref._esat(tt)
        humid = bool(np.isfinite(rh).any())
        if humid:
            good = np.isfinite(rh)
            rh = np.interp(zz, zz[good], rh[good])
        else:
            rh = np.zeros(tt.size)
        tab["thermo"] = (zz, tt, np.log(pp), rh, humid)
    return tab


def _stop_surface(surface):
    """Axes and values of a stop surface DataArray on (y, x), or empties."""
    if surface is None:
        return np.zeros(0), np.zeros(0), np.zeros((0, 0))
    if not isinstance(surface, xr.DataArray) or not {"y", "x"} <= set(surface.dims):
        raise ValueError("a terrain or source surface must be a DataArray on (y, x)")
    if surface.ndim != 2:
        raise ValueError("a terrain or source surface must be two-dimensional (y, x)")
    surface = surface.sortby(["y", "x"]).transpose("y", "x")
    vals = np.asarray(surface.values, dtype=np.float64)
    if not np.isfinite(vals).all():
        raise ValueError("a terrain or source surface must be finite everywhere")
    return (
        _f64(surface["y"].values),
        _f64(surface["x"].values),
        np.ascontiguousarray(vals),
    )


def _standard_atmosphere():
    z = np.arange(-500.0, 20001.0, 500.0)
    t = np.where(z < 11000.0, 288.15 - 0.0065 * z, 216.65)
    p = np.where(
        z < 11000.0,
        101325.0 * (1.0 - 2.25577e-5 * z) ** 5.25588,
        22632.1 * np.exp(-(z - 11000.0) / 6341.62),
    )
    return z, t, np.log(p), np.zeros(z.size), False


def _prepare_wind(wind, valid_time):
    """Axes and ``(nt, nz, ny, nx, 3)`` winds of a wind Dataset."""
    if not isinstance(wind, xr.Dataset):
        raise TypeError("wind must be an xarray.Dataset with u, v (and w)")
    for name in ("u", "v"):
        if name not in wind:
            raise ValueError("wind needs the variables 'u' and 'v'")
    ds = wind
    for d in ("z", "y", "x"):
        if d not in ds.dims:
            if d not in ds.coords:
                raise ValueError(f"wind needs the dimension or coordinate {d!r}")
            ds = ds.expand_dims(d)
    if "time" not in ds.dims:
        if "time" in ds.coords:
            ds = ds.expand_dims("time")
        else:
            ds = ds.expand_dims(time=[0.0 if valid_time is None else valid_time])
    ds = ds.sortby(["time", "z", "y", "x"])
    dims = ("time", "z", "y", "x")
    u = ds["u"].transpose(*dims).values
    v = ds["v"].transpose(*dims).values
    if "w" in ds:
        w = ds["w"].transpose(*dims).values
    else:
        w = np.zeros_like(u)
    uvw = np.ascontiguousarray(np.stack([u, v, w], axis=-1), dtype=np.float64)
    axes = {d: np.asarray(ds[d].values) for d in dims}
    return uvw, axes


# --------------------------------------------------------------------------
# core
# --------------------------------------------------------------------------


class _Setup:
    """Everything the kernels need besides the drops."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _setup(
    *,
    wind,
    profile,
    storm_motion,
    pattern_time,
    evaporation,
    fall_speed,
    density_correction,
    wind_divergence,
    concentration,
    dispersion,
    time_step,
    max_time,
    scheme,
    evaporated_diameter,
    direction,
    store_path,
    epoch_hint,
    terrain=None,
):
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {tuple(SCHEMES)}, not {scheme!r}")
    if not time_step > 0:
        raise ValueError("time_step must be positive")
    if not max_time >= 0:
        raise ValueError("max_time must be non-negative")
    if not evaporated_diameter > 0:
        raise ValueError("evaporated_diameter must be positive")
    kind, fc = _fall(fall_speed)
    tabs = _profile_tables(profile) or {}
    if wind is None and "wind" not in tabs:
        raise ValueError("give a wind analysis or a profile with u and v")
    epoch = epoch_hint
    # wind grid
    if wind is not None:
        uvw, axes = _prepare_wind(wind, pattern_time)
        wt = axes["time"]
        if np.issubdtype(wt.dtype, np.datetime64):
            wt = _seconds(wt, epoch)
        wt = _f64(wt)
        grid = (wt, _f64(axes["z"]), _f64(axes["y"]), _f64(axes["x"]), uvw)
    else:
        empty = np.zeros(0)
        grid = (empty, empty, empty, empty, empty)
    # background wind
    if "wind" in tabs:
        bz, bu, bv, bw = tabs["wind"]
    else:
        zc = grid[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # levels without data
            bu = np.nanmean(uvw[..., 0], axis=(0, 2, 3))
            bv = np.nanmean(uvw[..., 1], axis=(0, 2, 3))
        bw = np.zeros_like(bu)
        good = np.isfinite(bu) & np.isfinite(bv)
        if not good.any():
            raise ValueError("the wind analysis has no valid values")
        bz, bu, bv, bw = zc[good], bu[good], bv[good], bw[good]
    # thermodynamics
    humid = False
    if "thermo" in tabs:
        ez, et, elnp, erh, humid = tabs["thermo"]
    else:
        ez, et, elnp, erh, humid = _standard_atmosphere()
    if evaporation is None:
        evaporation = humid
    if evaporation and not humid:
        raise ValueError(
            "evaporation needs humidity (relative_humidity, specific_humidity or "
            "dewpoint) in the profile"
        )
    cx, cy = _motion(storm_motion)
    sh = sw = 0.0
    tl = 1.0
    turb = 0
    if dispersion is not None:
        sh, sw, tl = (float(v) for v in dispersion)
        if sh < 0 or sw < 0 or not tl > 0:
            raise ValueError(
                "dispersion needs sigma_h >= 0, sigma_w >= 0, timescale > 0"
            )
        turb = int(sh > 0 or sw > 0)
    par = np.zeros(16)
    par[:13] = [
        1.0 if direction == "forward" else -1.0,
        float(time_step),
        float(max_time),
        float(evaporated_diameter) ** 2,
        RHO0,
        VENTILATION[0],
        VENTILATION[1],
        cx,
        cy,
        0.0,
        sh,
        sw,
        tl,
    ]
    nsteps = int(math.ceil(max_time / time_step))
    ipar = np.zeros(10, dtype=np.int64)
    ipar[:7] = [
        SCHEMES[scheme],
        kind,
        int(bool(density_correction)),
        int(bool(evaporation)),
        int(bool(concentration)),
        int(bool(wind_divergence)),
        turb,
    ]
    stride = int(store_path) if store_path else 0
    ipar[7] = stride
    ipar[8] = nsteps // stride + 2 if stride else 0
    return _Setup(
        grid=grid,
        bg=tuple(_f64(a) for a in (bz, bu, bv, bw)),
        stop=_stop_surface(terrain),
        env=tuple(_f64(a) for a in (ez,) + _ref.thermo_nodes(et, np.exp(elnp), erh)),
        par=par,
        ipar=ipar,
        fc=fc,
        epoch=epoch,
        evaporation=bool(evaporation),
        motion=(cx, cy),
    )


def _model_dict(su):
    """The NumPy reference's view of a setup."""
    wt, wz, wy, wx, uvw = su.grid
    p, q = su.par, su.ipar
    m = {
        "nt": wt.size,
        "wt": wt,
        "wz": wz,
        "wy": wy,
        "wx": wx,
        "uvw": uvw,
        "bz": su.bg[0],
        "bu": su.bg[1],
        "bv": su.bg[2],
        "bw": su.bg[3],
        "sy": su.stop[0],
        "sx": su.stop[1],
        "sz": su.stop[2],
        "ez": su.env[0],
        "erho": su.env[1],
        "enu": su.env[2],
        "efkd": su.env[3],
        "essat": su.env[4],
        "ecs": su.env[5],
        "dir": p[0],
        "dt": p[1],
        "max_time": p[2],
        "smin": p[3],
        "rho0": p[4],
        "vav": p[5],
        "vbv": p[6],
        "cx": p[7],
        "cy": p[8],
        "sigma_h": p[10],
        "sigma_w": p[11],
        "timescale": p[12],
        "scheme": int(q[0]),
        "fall_kind": int(q[1]),
        "dens": int(q[2]),
        "evap": int(q[3]),
        "ratio": int(q[4]),
        "wdiv": int(q[5]),
        "turb": int(q[6]),
        "stride": int(q[7]),
        "fc": su.fc,
    }
    return m


def _integrate(su, x, y, z, t, d, zstop, seed, use_compiled, n_threads):
    """Run the kernel on 1-D arrays of drops; returns ``(out, path)``."""
    arrays = [_f64(a) for a in (x, y, z, t, d, zstop)]
    nrec = int(su.ipar[8]) if su.ipar[7] else 0
    if nrec and arrays[0].size * nrec > _MAX_PATH_SAMPLES:
        raise ValueError(
            "store_path would keep too many samples; use a larger stride or fewer drops"
        )
    seed = int(seed) & ((1 << 64) - 1)
    if use_compiled:
        out, path = _rain_trajectories.integrate(
            *arrays,
            *su.grid,
            *su.bg,
            *su.stop,
            *su.env,
            su.par,
            su.ipar,
            su.fc,
            seed,
            int(n_threads or 0),
        )
        return np.asarray(out), np.asarray(path)
    return _ref.integrate(_model_dict(su), *arrays, seed, nrec)


def _validate_diameter(diameter):
    if isinstance(diameter, xr.DataArray):
        d = diameter
    else:
        arr = np.atleast_1d(np.asarray(diameter, dtype=np.float64))
        if arr.ndim != 1:
            raise ValueError("diameter must be one-dimensional")
        d = xr.DataArray(arr, dims="diameter")
    if d.ndim != 1:
        raise ValueError("diameter must be one-dimensional")
    if d.dims[0] != "diameter":
        d = d.rename({d.dims[0]: "diameter"})
    vals = np.asarray(d.values, dtype=np.float64)
    if vals.size == 0 or not np.all(np.isfinite(vals)) or np.any(vals <= 0):
        raise ValueError("diameter must be positive and finite (mm)")
    if "diameter" not in d.coords:
        d = d.assign_coords(diameter=vals)
    d.attrs.setdefault("units", "mm")
    return d


def _epoch_of(time_values, wind):
    """Reference epoch of the second-based times: the earliest datetime64."""
    if _is_time(time_values):
        ok = ~np.isnat(time_values)
        if ok.any():
            return time_values[ok].min()
    if (
        isinstance(wind, xr.Dataset)
        and "time" in wind.coords
        and _is_time(wind["time"].values)
        and wind["time"].size
    ):
        return wind["time"].values.min()
    return None


def _run(
    start,
    stop_height,
    time,
    diameter,
    members,
    seed,
    direction,
    engine,
    n_threads,
    offset,
    **kw,
):
    """Shared driver: broadcast the points, run the kernel, build the output."""
    use_compiled = _use_compiled(engine)
    diameter = _validate_diameter(diameter)
    if members < 1 or int(members) != members:
        raise ValueError("members must be a positive integer")
    members = int(members)
    (x, y, z), tcomp = _points(start, offset=offset, time=time)
    zs = (
        stop_height
        if isinstance(stop_height, xr.DataArray)
        else xr.DataArray(float(stop_height))
    )
    x, y, z, tcomp, zs = xr.broadcast(x, y, z, tcomp, zs)
    epoch = _epoch_of(tcomp.values, kw.get("wind"))
    su = _setup(direction=direction, epoch_hint=epoch, **kw)
    tsec = _seconds(tcomp.values, epoch)
    dims, shape = x.dims, x.shape
    nd = diameter.size
    npts = int(np.prod(shape, dtype=np.int64))
    n = npts * nd * members

    def per_drop(a):
        a = np.asarray(a, dtype=np.float64).reshape(npts, 1, 1)
        return np.broadcast_to(a, (npts, nd, members)).reshape(n)

    dvals = np.broadcast_to(diameter.values.reshape(1, nd, 1), (npts, nd, members))
    out, path = _integrate(
        su,
        per_drop(x.values),
        per_drop(y.values),
        per_drop(z.values),
        per_drop(tsec),
        dvals.reshape(n),
        per_drop(zs.values),
        seed,
        use_compiled,
        n_threads,
    )
    layout = (dims, shape, nd, members)
    return su, epoch, layout, diameter, (x, y, z, tcomp, tsec), out, path


def _wrap(su, epoch, layout, diameter, pts, out, path, direction, store_path):
    dims, shape, nd, nm = layout
    x, y, z, tcomp, tsec = pts
    ens = nm > 1
    odims = tuple(dims) + ("diameter",) + (("member",) if ens else ())
    oshape = tuple(shape) + (nd,) + ((nm,) if ens else ())
    o = out.reshape(oshape + (out.shape[-1],))
    fwd = direction == "forward"
    coords = dict(x.coords)
    coords["diameter"] = diameter
    if ens:
        coords["member"] = np.arange(nm)
    status = o[..., 6]
    invalid = status == STATUS["invalid"]
    ones = (1,) * len(dims)
    d0 = np.broadcast_to(
        diameter.values.reshape(ones + (nd,) + ((1,) if ens else ())), oshape
    )
    d_end = o[..., 4]
    with np.errstate(invalid="ignore", divide="ignore"):
        lost = 1.0 - (d_end / d0) ** 3 if fwd else 1.0 - (d0 / d_end) ** 3
    if fwd:
        lost = np.where(status == STATUS["evaporated"], 1.0, lost)
    lost = np.where(invalid, np.nan, lost)
    tstart = np.broadcast_to(
        tsec.reshape(tuple(shape) + (1,) + ((1,) if ens else ())), oshape
    )
    t_end = tstart + (1.0 if fwd else -1.0) * o[..., 3]
    if _is_time(tcomp.values):
        t_end = np.where(
            np.isfinite(t_end),
            _from_seconds(np.nan_to_num(t_end), epoch),
            np.datetime64("NaT", "ns"),
        )
    ratio = np.exp(o[..., 5]) if fwd else np.exp(-o[..., 5])
    ratio = np.where(invalid, np.nan, ratio)
    pre = "landing" if fwd else "source"
    data = {
        f"{pre}_x": (
            odims,
            o[..., 0],
            {"units": "m", "long_name": f"x of the {pre} point"},
        ),
        f"{pre}_y": (
            odims,
            o[..., 1],
            {"units": "m", "long_name": f"y of the {pre} point"},
        ),
        f"{pre}_z": (
            odims,
            o[..., 2],
            {"units": "m", "long_name": f"Height above sea level of the {pre} point"},
        ),
        f"{pre}_time": (
            odims,
            t_end,
            {"standard_name": "time", "long_name": f"Time at the {pre} point"},
        ),
        "fall_time": (
            odims,
            o[..., 3],
            {"units": "s", "long_name": "Time along the path from its start"},
        ),
        f"{pre}_diameter": (
            odims,
            d_end,
            {"units": "mm", "long_name": f"Drop diameter at the {pre} point"},
        ),
        "evaporated_mass_fraction": (
            odims,
            lost,
            {
                "units": "1",
                "long_name": "Fraction of the mass of the drop lost by evaporation",
            },
        ),
        "concentration_ratio": (
            odims,
            ratio,
            {
                "units": "1",
                "long_name": "Concentration per unit diameter at the later end of "
                "the path over that at the earlier end",
            },
        ),
        "fall_speed_start": (
            odims,
            o[..., 7],
            {"units": "m s-1", "long_name": "Downward speed at the start of the path"},
        ),
        "fall_speed_end": (
            odims,
            o[..., 8],
            {"units": "m s-1", "long_name": "Downward speed at the end of the path"},
        ),
        "status": (odims, status.astype(np.int8), dict(_STATUS_ATTRS)),
    }
    sdims = tuple(dims)
    start = {
        "start_x": (sdims, x.values, {"units": "m"}),
        "start_y": (sdims, y.values, {"units": "m"}),
        "start_z": (sdims, z.values, {"units": "m"}),
        "start_time": (sdims, tcomp.values, {"standard_name": "time"}),
    }
    ds = xr.Dataset({**start, **data}, coords=coords)
    if store_path:
        p = path.reshape(oshape + path.shape[1:])
        ds["path_time"] = (
            odims + ("step",),
            p[..., 0],
            {"units": "s", "long_name": "Time since the start of the path"},
        )
        for k, name in enumerate(("path_x", "path_y", "path_z", "path_diameter"), 1):
            ds[name] = (
                odims + ("step",),
                p[..., k],
                {"units": "mm" if k == 4 else "m"},
            )
    cx, cy = su.motion
    par = su.par
    ds.attrs = {
        "title": "Raindrop trajectories" if fwd else "Backward raindrop trajectories",
        "direction": direction,
        "scheme": "rk4" if su.ipar[0] == 4 else "rk2",
        "time_step": float(par[1]),
        "max_time": float(par[2]),
        "evaporated_diameter": float(math.sqrt(par[3])),
        "evaporation": int(su.evaporation),
        "density_correction": int(su.ipar[2]),
        "wind_divergence": int(su.ipar[5]),
        "storm_motion_u": cx,
        "storm_motion_v": cy,
        "reference": "Dawson et al. 2015, J. Atmos. Sci., 72, 340-348",
    }
    return ds


def _check_threads(n_threads):
    if n_threads is not None and int(n_threads) < 0:
        raise ValueError("n_threads must be non-negative")


def rain_trajectories(
    source,
    diameter,
    *,
    wind=None,
    profile=None,
    storm_motion=None,
    pattern_time=None,
    time=None,
    surface_height=0.0,
    terrain=None,
    offset=(0.0, 0.0, 0.0),
    evaporation=None,
    fall_speed="atlas1973",
    density_correction=True,
    wind_divergence=False,
    concentration=True,
    dispersion=None,
    members=1,
    seed=0,
    time_step=5.0,
    max_time=3600.0,
    scheme="rk4",
    evaporated_diameter=0.12,
    store_path=0,
    engine="auto",
    n_threads=None,
):
    """
    Trajectories of raindrops from source points to the ground.

    One drop of every size in ``diameter`` (and every turbulent member) is
    released at each source point and followed until it reaches the surface,
    evaporates, or ``max_time`` is exceeded; see the module description for
    the equations.

    Parameters
    ----------
    source : xarray.Dataset, xarray.DataTree or mapping
        Source points with ``x``, ``y`` (m, in the frame of the wind
        analysis), ``z`` (m above sea level) and, optionally, ``time``
        (datetime64 or seconds), as variables or coordinates of any shape:
        a grid (``z, y, x``), the gates of a sweep (``azimuth, range`` with
        georeferenced ``x, y, z``) or a list of points. Arrays or scalars in
        a mapping are one-dimensional points. A DataTree is processed sweep
        by sweep (nodes without ``x``, ``y``, ``z`` are skipped).
    diameter : array-like or xarray.DataArray
        Drop diameters (mm), a one-dimensional ``diameter`` axis.
    wind : xarray.Dataset, optional
        Three-dimensional wind ``u``, ``v`` and (optionally) ``w`` (m s-1) on
        ``(z, y, x)`` or ``(time, z, y, x)`` with ``x``, ``y`` (m) and ``z``
        (m above sea level), e.g. from :func:`radarx.retrieve.multi_doppler`.
    profile : xarray.Dataset, optional
        Sounding or ERA5 profile on ``height`` (m above sea level) with
        ``temperature`` (K), ``pressure`` (Pa), humidity (``relative_humidity``,
        ``specific_humidity`` or ``dewpoint``) and ``u``, ``v`` (and ``w``),
        see :mod:`radarx.io.sounding`. It gives the air density, the humidity
        for evaporation and the background wind. Default: a standard
        atmosphere without evaporation.
    storm_motion : tuple of float or xarray.Dataset, optional
        Motion ``(u, v)`` (m s-1) of the wind pattern (the output of
        :func:`radarx.retrieve.estimate_motion` is averaged). Default: the
        pattern is fixed.
    pattern_time : datetime-like or float, optional
        Valid time of a wind analysis that has no ``time`` coordinate (for the
        moving pattern). Default 0.
    time : datetime-like or float, optional
        Release time, if the source has none.
    surface_height : float or xarray.DataArray, optional
        Height of the surface (m above sea level), a plane, broadcast against
        the source points. Default 0.
    terrain : xarray.DataArray, optional
        Height of the surface (m above sea level) on a regular ``(y, x)`` grid,
        interpolated bilinearly at the position of the drop; replaces
        ``surface_height``.
    offset : tuple of float, optional
        ``(x, y, z)`` added to the source coordinates, e.g. the position of
        the radar in the frame of the wind grid.
    evaporation : bool, optional
        Evaporate the drops. Default: if the profile has humidity.
    fall_speed : "atlas1973", (a, b, f) or ("polynomial", c0, c1, ...)
        Sea-level terminal speed law (see the module description).
    density_correction : bool, optional
        Multiply by :math:`(\\rho_0/\\rho)^{0.4}`. Default True.
    wind_divergence : bool, optional
        Include the divergence of the wind analysis in the concentration
        ratio. Default False.
    concentration : bool, optional
        Compute ``concentration_ratio`` (costs about a third more). Default
        True.
    dispersion : tuple of float, optional
        ``(sigma_h, sigma_w, timescale)``: standard deviations (m s-1) of the
        horizontal and vertical turbulent velocity and the Lagrangian time
        scale (s).
    members : int, optional
        Ensemble members per drop (``member`` dimension if above 1); only
        useful with ``dispersion``.
    seed : int, optional
        Seed of the counter-based random numbers. Default 0.
    time_step : float, optional
        Integration step (s). Default 5.
    max_time : float, optional
        Longest time followed (s), rounded up to whole steps. Default 3600.
    scheme : {"rk4", "rk2"}, optional
        Fourth-order Runge-Kutta or Heun. Default ``"rk4"``.
    evaporated_diameter : float, optional
        Drops smaller than this (mm) count as evaporated. Default 0.12.
    store_path : int, optional
        Keep the path every ``store_path`` steps as ``path_x``, ``path_y``,
        ``path_z``, ``path_diameter`` and ``path_time`` on a ``step`` axis.
        Default 0 (no path).
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation. The NumPy reference gives the same results.
    n_threads : int, optional
        Threads of the compiled kernel. Default: all cores.

    Returns
    -------
    xarray.Dataset
        On the dimensions of the source points plus ``diameter`` (and
        ``member``): ``landing_x``, ``landing_y``, ``landing_z``,
        ``landing_time``, ``fall_time`` (s), ``landing_diameter`` (mm),
        ``evaporated_mass_fraction``, ``concentration_ratio``,
        ``fall_speed_start``, ``fall_speed_end`` (downward speeds, m s-1),
        ``status`` (0 aloft at ``max_time``, 1 reached the surface, 2
        evaporated, 3 invalid input) and the start of every path as
        ``start_x``, ``start_y``, ``start_z`` and ``start_time``.

    Notes
    -----
    Sources (Dawson et al. 2015 for the trajectory model; Kumjian and Ryzhkov
    2010 for the evaporation; Atlas et al. 1973 and Foote and du Toit 1969 for
    the fall speed; Buck 1981 for the saturation vapour pressure; Thomson
    1987 and Wilson and Sawford 1996 for the turbulent dispersion; Gal-Chen
    1982 for the moving frame). Defaults that are radarx choices, not from a
    paper: ``time_step`` 5 s,
    ``max_time`` 3600 s, ``evaporated_diameter`` 0.12 mm, the RK4 scheme and
    the Hermite landing interpolation. The fall speed, density correction,
    evaporation and turbulence formulations are described, with their
    sources and the points that could not be checked, in the module
    docstring.

    References
    ----------
    Dawson, D. T., E. R. Mansell, and M. R. Kumjian, 2015: Does wind shear
    cause hydrometeor size sorting? *J. Atmos. Sci.*, **72** (1), 340-348,
    https://doi.org/10.1175/JAS-D-14-0084.1

    Kumjian, M. R., and A. V. Ryzhkov, 2010: The impact of evaporation on
    polarimetric characteristics of rain: Theoretical model and practical
    implications. *J. Appl. Meteor. Climatol.*, **49** (6), 1247-1267,
    https://doi.org/10.1175/2010JAMC2243.1

    Atlas, D., R. C. Srivastava, and R. S. Sekhon, 1973: Doppler radar
    characteristics of precipitation at vertical incidence. *Rev. Geophys.*,
    **11** (1), 1-35, https://doi.org/10.1029/RG011i001p00001

    Foote, G. B., and P. S. du Toit, 1969: Terminal velocity of raindrops
    aloft. *J. Appl. Meteor.*, **8** (2), 249-253,
    https://doi.org/10.1175/1520-0450(1969)008<0249:TVORA>2.0.CO;2

    Buck, A. L., 1981: New equations for computing vapor pressure and
    enhancement factor. *J. Appl. Meteor.*, **20** (12), 1527-1532,
    https://doi.org/10.1175/1520-0450(1981)020<1527:NEFCVP>2.0.CO;2

    Thomson, D. J., 1987: Criteria for the selection of stochastic models of
    particle trajectories in turbulent flows. *J. Fluid Mech.*, **180**,
    529-556, https://doi.org/10.1017/S0022112087001940

    Wilson, J. D., and B. L. Sawford, 1996: Review of Lagrangian stochastic
    models for trajectories in the turbulent atmosphere. *Bound.-Layer
    Meteor.*, **78** (1-2), 191-210, https://doi.org/10.1007/BF00122492

    Gal-Chen, T., 1982: Errors in fixed and moving frame of references:
    Applications for conventional and Doppler radar analysis. *J. Atmos.
    Sci.*, **39** (10), 2279-2300,
    https://doi.org/10.1175/1520-0469(1982)039<2279:EIFAMF>2.0.CO;2

    Examples
    --------
    >>> pts = {"x": 0.0, "y": 0.0, "z": 2000.0}  # doctest: +SKIP
    >>> out = rain_trajectories(pts, [1.0, 2.0, 4.0], profile=snd)  # doctest: +SKIP
    >>> out.landing_x  # doctest: +SKIP
    """
    _check_threads(n_threads)
    if isinstance(source, xr.DataTree):
        return _tree(
            source,
            lambda ds: rain_trajectories(
                ds,
                diameter,
                wind=wind,
                profile=profile,
                storm_motion=storm_motion,
                pattern_time=pattern_time,
                time=time,
                surface_height=surface_height,
                terrain=terrain,
                offset=offset,
                evaporation=evaporation,
                fall_speed=fall_speed,
                density_correction=density_correction,
                wind_divergence=wind_divergence,
                concentration=concentration,
                dispersion=dispersion,
                members=members,
                seed=seed,
                time_step=time_step,
                max_time=max_time,
                scheme=scheme,
                evaporated_diameter=evaporated_diameter,
                store_path=store_path,
                engine=engine,
                n_threads=n_threads,
            ),
        )
    su, epoch, layout, dia, pts, out, path = _run(
        source,
        surface_height,
        time,
        diameter,
        members,
        seed,
        "forward",
        engine,
        n_threads,
        offset,
        wind=wind,
        profile=profile,
        storm_motion=storm_motion,
        pattern_time=pattern_time,
        evaporation=evaporation,
        fall_speed=fall_speed,
        density_correction=density_correction,
        wind_divergence=wind_divergence,
        concentration=concentration,
        dispersion=dispersion,
        time_step=time_step,
        max_time=max_time,
        scheme=scheme,
        evaporated_diameter=evaporated_diameter,
        store_path=store_path,
        terrain=terrain,
    )
    return _wrap(su, epoch, layout, dia, pts, out, path, "forward", store_path)


def _tree(tree, func):
    """Apply ``func`` to the sweeps of a DataTree that hold x, y and z."""
    nodes = {}
    for name, node in tree.children.items():
        ds = node.to_dataset()
        if all(_component(ds, c) is not None for c in ("x", "y", "z")):
            nodes[name] = func(ds)
    if not nodes:
        raise ValueError("no sweep of the DataTree has x, y and z (georeference it)")
    return xr.DataTree.from_dict({f"/{k}": v for k, v in nodes.items()})


def _surface_points(target, surface_height):
    """Target points with a ``z`` (the surface) if they have none."""
    ds = _as_dataset(target, ("x", "y"))
    if _component(ds, "z") is None:
        z = (
            surface_height
            if isinstance(surface_height, xr.DataArray)
            else xr.DataArray(float(surface_height))
        )
        ds = ds.assign(z=z)
    return ds


def _sample_source_dsd(field, ds, motion=(0.0, 0.0)):
    """
    Linear interpolation of a source DSD at the source points of ``ds``.

    With several times the field is interpolated linearly in time between the
    two analyses around each source time; the echo pattern is frozen and moves
    with ``motion``, so each analysis is read at the source point moved back by
    ``motion`` times the time since that analysis. Outside the first and last
    times the result is missing.
    """
    if not isinstance(field, xr.DataArray):
        raise TypeError("source_dsd must be an xarray.DataArray")
    known = {
        "x": "source_x",
        "y": "source_y",
        "z": "source_z",
        "time": "source_time",
        "diameter": "source_diameter",
    }
    extra = [d for d in field.dims if d not in known]
    if extra:
        raise ValueError(
            f"source_dsd may only have the dimensions {tuple(known)}, not {extra}"
        )
    if "diameter" not in field.dims:
        raise ValueError("source_dsd needs a 'diameter' dimension")
    shape = ds["source_x"].shape
    dims = ds["source_x"].dims
    pos = {k: ds[v].values.ravel() for k, v in known.items() if k != "time"}
    n = pos["x"].size
    cx, cy = motion
    if "time" in field.dims:
        tvals = field["time"].values
        t_is_dt = _is_time(tvals)
        t0 = tvals[0]
        tf = _seconds(tvals, t0) if t_is_dt else tvals.astype(np.float64)
        ts = ds["source_time"].values.ravel()
        ts = _seconds(ts, t0) if t_is_dt else ts.astype(np.float64)
        nt = tf.size
        if nt > 1:
            ia = np.clip(np.searchsorted(tf, ts, side="right") - 1, 0, nt - 2)
            wb = (ts - tf[ia]) / (tf[ia + 1] - tf[ia])
            inside = (ts >= tf[0]) & (ts <= tf[-1])
        else:
            ia = np.zeros(n, dtype=np.int64)
            wb = np.zeros(n)
            inside = np.ones(n, dtype=bool)
    else:
        tf = ts = None
        ia = np.zeros(n, dtype=np.int64)
        wb = np.zeros(n)
        inside = np.ones(n, dtype=bool)

    def sample(index):
        """The field of time ``index`` at the sources moved back by the motion."""
        out = np.full(n, np.nan)
        for j in np.unique(index[inside]):
            m = inside & (index == j)
            f2 = field.isel(time=int(j), drop=True) if tf is not None else field
            shift = ts[m] - tf[j] if tf is not None else 0.0
            idx = {}
            for dim, vals in pos.items():
                v = vals[m]
                if dim == "x":
                    v = v - cx * shift
                elif dim == "y":
                    v = v - cy * shift
                if dim in f2.dims and f2.sizes[dim] > 1:
                    idx[dim] = xr.DataArray(v, dims="_p")
                elif dim in f2.dims:
                    f2 = f2.isel({dim: 0}, drop=True)
            got = f2.interp(**idx) if idx else f2
            vals_ = (
                np.asarray(got.transpose("_p", ...).values)
                if "_p" in got.dims
                else np.asarray(got.values)
            )
            out[m] = vals_ if vals_.ndim else float(vals_)
        return out

    if tf is not None and tf.size > 1:
        first = sample(ia)
        vals = np.where(wb == 0.0, first, (1.0 - wb) * first + wb * sample(ia + 1))
    else:
        vals = sample(ia)
    return xr.DataArray(vals.reshape(shape), dims=dims, coords=ds["source_x"].coords)


def rain_source_points(
    target,
    diameter,
    *,
    source_height=None,
    source_surface=None,
    wind=None,
    profile=None,
    storm_motion=None,
    pattern_time=None,
    time=None,
    surface_height=0.0,
    offset=(0.0, 0.0, 0.0),
    source_dsd=None,
    evaporation=None,
    fall_speed="atlas1973",
    density_correction=True,
    wind_divergence=False,
    dispersion=None,
    members=1,
    seed=0,
    time_step=5.0,
    max_time=3600.0,
    scheme="rk4",
    evaporated_diameter=0.12,
    store_path=0,
    engine="auto",
    n_threads=None,
):
    """
    Source points and times aloft of drops arriving at the surface.

    The equations of :func:`rain_trajectories` are integrated backward in
    time from drops of diameter ``diameter`` arriving at the target points
    (e.g. disdrometers) at the given times, up to the height
    ``source_height``. Evaporation is integrated backward too, so the
    diameter at the source is larger than that at the surface; a drop that
    would have to be smaller than ``evaporated_diameter`` at some point of
    its past has ``status`` 2.

    Parameters
    ----------
    target : xarray.Dataset or mapping
        Target points with ``x``, ``y`` (m), optionally ``z`` (m above sea
        level; default ``surface_height``) and the arrival ``time``
        (datetime64 or seconds).
    diameter : array-like or xarray.DataArray
        Diameters (mm) of the drops **at the target**.
    source_height : float or xarray.DataArray, optional
        Height (m above sea level) to trace the drops back to, e.g. the beam
        height over the site.
    source_surface : xarray.DataArray, optional
        The source level as a surface on a regular ``(y, x)`` grid (m above
        sea level), e.g. the height of the gates of a sweep, interpolated
        bilinearly at the position of the drop; replaces ``source_height``.
    source_dsd : xarray.DataArray, optional
        Concentration (m-3 mm-1) of the drops aloft, e.g. a retrieved DSD on
        ``(time, z, y, x, diameter)`` (any subset of these dimensions;
        dimensions of length one other than ``time`` are used as constants).
        It is interpolated linearly to the source point and diameter. With
        several times it is blended linearly between the two times around the
        source time; with ``storm_motion`` the echo pattern is frozen and
        moves, so each time is read at the source point moved back by the
        motion times the time since it (a time before the first or after the
        last gives NaN); a single time is held. The result, multiplied by
        ``concentration_ratio``, is returned as ``ND``: the concentration at
        the target.
    **kwargs
        As in :func:`rain_trajectories`.

    Returns
    -------
    xarray.Dataset
        On the dimensions of the target points plus ``diameter``:
        ``source_x``, ``source_y``, ``source_z``, ``source_time``,
        ``source_diameter``, ``fall_time`` (s), ``evaporated_mass_fraction``
        (of the source drop), ``concentration_ratio`` (target over source),
        ``fall_speed_start`` (at the target), ``fall_speed_end`` (at the
        source), ``status``, and ``ND`` with ``source_dsd``.

    Notes
    -----
    The backward integration of the trajectory equations (Dawson et al. 2015;
    Kumjian and Ryzhkov 2010 for the evaporation) and the sampling of
    ``source_dsd`` are radarx constructions: no paper gives this backward
    scheme, and its defaults are those of :func:`rain_trajectories`.

    References
    ----------
    Dawson, D. T., E. R. Mansell, and M. R. Kumjian, 2015: Does wind shear
    cause hydrometeor size sorting? *J. Atmos. Sci.*, **72** (1), 340-348,
    https://doi.org/10.1175/JAS-D-14-0084.1

    Kumjian, M. R., and A. V. Ryzhkov, 2010: The impact of evaporation on
    polarimetric characteristics of rain: Theoretical model and practical
    implications. *J. Appl. Meteor. Climatol.*, **49** (6), 1247-1267,
    https://doi.org/10.1175/2010JAMC2243.1
    """
    _check_threads(n_threads)
    if source_height is None and source_surface is None:
        raise ValueError("give source_height or source_surface")
    pts = _surface_points(target, surface_height)
    su, epoch, layout, dia, p, out, path = _run(
        pts,
        0.0 if source_height is None else source_height,
        time,
        diameter,
        members,
        seed,
        "backward",
        engine,
        n_threads,
        offset,
        wind=wind,
        profile=profile,
        storm_motion=storm_motion,
        pattern_time=pattern_time,
        evaporation=evaporation,
        fall_speed=fall_speed,
        density_correction=density_correction,
        wind_divergence=wind_divergence,
        concentration=True,
        dispersion=dispersion,
        time_step=time_step,
        max_time=max_time,
        scheme=scheme,
        evaporated_diameter=evaporated_diameter,
        store_path=store_path,
        terrain=source_surface,
    )
    ds = _wrap(su, epoch, layout, dia, p, out, path, "backward", store_path)
    if source_dsd is not None:
        n_src = _sample_source_dsd(source_dsd, ds, su.motion)
        nd = n_src * ds["concentration_ratio"]
        nd.attrs = {
            "units": "m-3 mm-1",
            "long_name": "Concentration of drops at the target per unit diameter",
            "comment": "source concentration times concentration_ratio",
        }
        ds["ND"] = nd
    return ds


def size_sorting(result, reference_diameter=2.0, *, axis=None):
    """
    Landing (or source) of each size relative to that of a reference size.

    Drops of the reference diameter, found by linear interpolation along
    ``diameter`` between the sizes of the result, are the baseline: for
    every size the displacement of the end point and the offset of its time
    relative to those of the reference drop from the same start. Large drops
    fall faster, so they arrive earlier and are displaced less by the wind
    than the reference when the wind shears; the sign of the displacement
    shows the direction of the sorting.

    Parameters
    ----------
    result : xarray.Dataset
        The output of :func:`rain_trajectories` (``landing_*``) or
        :func:`rain_source_points` (``source_*``), including the reference
        diameter within its range of diameters.
    reference_diameter : float, optional
        Reference diameter (mm). Default 2.
    axis : tuple of float, optional
        Direction ``(ex, ey)`` of the along-axis displacement. Default: the
        storm motion of the result if it is not zero, else the direction of
        the displacement of the reference drop from its start.

    Returns
    -------
    xarray.Dataset
        ``displacement_x``, ``displacement_y`` and ``displacement`` (m),
        ``displacement_along`` and ``displacement_cross`` (m, along and
        to the left of ``axis``), ``arrival_offset`` (s, negative for drops
        that arrive earlier than the reference), and the reference end point
        ``reference_x``, ``reference_y``, ``reference_time`` (dimension
        ``diameter`` removed).

    Notes
    -----
    Size sorting follows the concept of Kumjian and Ryzhkov (2012); the
    displacement and arrival offset relative to a reference size are radarx
    definitions.

    References
    ----------
    Kumjian, M. R., and A. V. Ryzhkov, 2012: The impact of size sorting on
    the polarimetric radar variables. *J. Atmos. Sci.*, **69** (6),
    2042-2060, https://doi.org/10.1175/JAS-D-11-0125.1
    """
    if "landing_x" in result:
        pre = "landing"
    elif "source_x" in result:
        pre = "source"
    else:
        raise ValueError("result needs landing_x or source_x (see rain_trajectories)")
    if "diameter" not in result.dims or result.sizes["diameter"] < 2:
        raise ValueError("result needs at least two diameters")
    d = result["diameter"].values
    if not (d.min() <= reference_diameter <= d.max()):
        raise ValueError(
            f"reference_diameter {reference_diameter} mm is outside the diameters "
            f"{d.min():g}-{d.max():g} mm of the result"
        )
    names = [f"{pre}_x", f"{pre}_y", f"{pre}_time"]
    t_var = result[names[2]]
    time_is_dt = _is_time(t_var.values)
    sub = result[names[:2]]
    t_sec = (t_var - t_var.min()) / np.timedelta64(1, "s") if time_is_dt else t_var
    t_zero = t_var.min() if time_is_dt else 0.0
    sub = sub.assign(_t=t_sec)
    ref = sub.interp(diameter=reference_diameter)
    dx = sub[names[0]] - ref[names[0]]
    dy = sub[names[1]] - ref[names[1]]
    dt = sub["_t"] - ref["_t"]
    if axis is None:
        cx = result.attrs.get("storm_motion_u", 0.0)
        cy = result.attrs.get("storm_motion_v", 0.0)
        if cx == 0.0 and cy == 0.0:
            ex = ref[names[0]] - result["start_x"]
            ey = ref[names[1]] - result["start_y"]
        else:
            ex, ey = xr.DataArray(cx), xr.DataArray(cy)
    else:
        ex, ey = xr.DataArray(float(axis[0])), xr.DataArray(float(axis[1]))
    norm = np.hypot(ex, ey)
    with np.errstate(invalid="ignore", divide="ignore"):
        ex, ey = ex / norm, ey / norm
    along = dx * ex + dy * ey
    cross = -dx * ey + dy * ex
    ref_t = ref["_t"]
    if time_is_dt:
        ref_t = t_zero + (ref["_t"] * 1e9).astype("timedelta64[ns]")
    out = xr.Dataset(
        {
            "displacement_x": dx.assign_attrs(units="m"),
            "displacement_y": dy.assign_attrs(units="m"),
            "displacement": np.hypot(dx, dy).assign_attrs(units="m"),
            "displacement_along": along.assign_attrs(units="m"),
            "displacement_cross": cross.assign_attrs(units="m"),
            "arrival_offset": dt.assign_attrs(
                units="s",
                long_name="Time of the end point relative to the reference drop",
            ),
            "reference_x": ref[names[0]].assign_attrs(units="m"),
            "reference_y": ref[names[1]].assign_attrs(units="m"),
            "reference_time": ref_t,
        }
    )
    out.attrs = {
        "reference_diameter": float(reference_diameter),
        "reference": "Kumjian and Ryzhkov 2012, J. Atmos. Sci., 69, 2042-2060",
    }
    return out


def _cell_area(traj):
    """Horizontal area (m2) of the source cells of a grid or a sweep."""
    dims = traj["start_x"].dims
    if "x" in dims and "y" in dims and "x" in traj.coords and "y" in traj.coords:
        dx = float(np.abs(np.diff(traj["x"].values)).mean())
        dy = float(np.abs(np.diff(traj["y"].values)).mean())
        return xr.DataArray(dx * dy)
    if "range" in dims and "azimuth" in dims and "range" in traj.coords:
        r = traj["range"].values
        az = np.deg2rad(traj["azimuth"].values)
        dr = np.gradient(r)
        daz = np.abs(np.gradient(np.unwrap(az)))
        el = np.deg2rad(traj["elevation"].values) if "elevation" in traj.coords else 0.0
        ce = np.cos(el)
        area = xr.DataArray(
            (r * ce * dr * ce)[None, :] * daz[:, None],
            dims=("azimuth", "range"),
            coords={"azimuth": traj["azimuth"], "range": traj["range"]},
        )
        return area
    raise ValueError(
        "give the area (m2) of the source cells: it cannot be derived from the "
        "dimensions of the trajectories"
    )


def _isnull(a):
    return np.isnat(a) if _is_time(a) else np.isnan(a)


def _edges(centers):
    c = np.asarray(centers, dtype=np.float64)
    mid = 0.5 * (c[1:] + c[:-1])
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


def surface_dsd(
    trajectories,
    nd,
    *,
    x,
    y,
    time,
    area=None,
    duration=None,
):
    """
    Drop size distribution at the surface from landed drops.

    Every source point releases, during the ``duration`` its observation
    represents, the drops that cross its level: in every size bin
    :math:`N(D) \\Delta D\\, V_s\\, A\\, T` drops, where :math:`V_s` is the
    downward speed of the drop at the source (``fall_speed_start``) and
    :math:`A` the horizontal area of the source cell. They are carried to
    their landing points by :func:`rain_trajectories` and accumulated onto
    the surface grid (shared out to the neighbouring grid points in ``x``,
    ``y`` and ``time`` with linear weights) and between the two size bins
    nearest their diameter at landing. Counting at the surface gives
    :math:`N_s(D) = \\sum n / (V_e\\, \\Delta D\\, \\Delta x\\, \\Delta y\\, \\Delta t)`
    with :math:`V_e` the downward speed at landing (``fall_speed_end``), the
    way a disdrometer counts them, so the number flux is conserved exactly:
    including the correction of the fall speed with air density, the shrinking
    of drops by evaporation (drops that evaporated are lost) and the sorting
    in space and time.

    Parameters
    ----------
    trajectories : xarray.Dataset
        The output of :func:`rain_trajectories` (not for a DataTree; apply it
        to a sweep). Its ``diameter`` bins are those of ``nd``.
    nd : xarray.DataArray
        Concentration (m-3 mm-1) of the source points on their dimensions and
        ``diameter``; it is broadcast against the trajectories.
    x, y, time : array-like
        Regular, increasing surface grid (at least two points each): the
        coordinates (m) and the times (datetime64 or seconds).
    area : float or xarray.DataArray, optional
        Horizontal area of the source cells (m2). Default: ``dx dy`` of a
        regular ``(y, x)`` grid or the gate area of an ``(azimuth, range)``
        sweep.
    duration : float or xarray.DataArray, optional
        Time (s) represented by each source observation, centred on its time:
        its drops are spread evenly over it. Default: the median time step of
        the ``time`` dimension of the source.

    Returns
    -------
    xarray.Dataset
        ``ND`` (m-3 mm-1) on ``(time, y, x, diameter)``, the total
        concentration ``NT`` (m-3) and the mass-weighted mean diameter
        ``DM`` (mm). Drops landing outside the grid are not counted.

    Notes
    -----
    The number-flux bookkeeping above is a radarx construction (the
    conservation of the number flux, with the bin widths of the source
    diameters); it is motivated by the size sorting of Kumjian and Ryzhkov
    (2012) and Dawson et al. (2015), who do not give this accumulation.

    References
    ----------
    Kumjian, M. R., and A. V. Ryzhkov, 2012: The impact of size sorting on
    the polarimetric radar variables. *J. Atmos. Sci.*, **69** (6),
    2042-2060, https://doi.org/10.1175/JAS-D-11-0125.1

    Dawson, D. T., E. R. Mansell, and M. R. Kumjian, 2015: Does wind shear
    cause hydrometeor size sorting? *J. Atmos. Sci.*, **72** (1), 340-348,
    https://doi.org/10.1175/JAS-D-14-0084.1
    """
    for name in ("landing_x", "landing_y", "landing_time", "landing_diameter"):
        if name not in trajectories:
            raise ValueError("trajectories must be the output of rain_trajectories")
    if "diameter" not in nd.dims:
        raise ValueError("nd needs a 'diameter' dimension")
    xg, yg = (np.asarray(a, dtype=np.float64) for a in (x, y))
    tg = np.asarray(time)
    if min(xg.size, yg.size, tg.size) < 2:
        raise ValueError("x, y and time need at least two points each")
    dx, dy = float(xg[1] - xg[0]), float(yg[1] - yg[0])
    t_is_dt = _is_time(tg)
    t0 = tg[0]
    tsec = _seconds(tg, t0) if t_is_dt else tg.astype(np.float64)
    dt = float(tsec[1] - tsec[0])
    if not (dx > 0 and dy > 0 and dt > 0):
        raise ValueError("x, y and time must be increasing")
    land = trajectories["landing_x"]
    dims = land.dims
    nd = nd.broadcast_like(land).transpose(*dims)
    dia = trajectories["diameter"].values
    if dia.size < 2:
        raise ValueError("surface_dsd needs at least two diameters to define the bins")
    edges = _edges(dia)
    width = np.diff(edges)
    wsrc = xr.DataArray(width, dims="diameter", coords={"diameter": dia})
    if area is None:
        area = _cell_area(trajectories)
    if duration is None:
        times = trajectories["start_time"]
        if "time" in times.dims and times.sizes["time"] > 1:
            tt = np.unique(times.values[~_isnull(times.values)])
            step = np.diff(tt)
            duration = float(
                np.median(step / np.timedelta64(1, "s")) if t_is_dt else np.median(step)
            )
        else:
            raise ValueError("give the duration (s) represented by each source")
    n_drops = (
        nd
        * wsrc
        * trajectories["fall_speed_start"]
        * xr.DataArray(area)
        * xr.DataArray(duration)
    ).transpose(*dims)
    status = trajectories["status"].transpose(*dims).values
    vz_end = trajectories["fall_speed_end"].transpose(*dims).values
    d_end = trajectories["landing_diameter"].transpose(*dims).values
    lx = land.values
    ly = trajectories["landing_y"].transpose(*dims).values
    lt = trajectories["landing_time"].transpose(*dims).values
    lts = _seconds(lt, t0) if t_is_dt else lt.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        weight = n_drops.values / vz_end
    ok = (
        (status == STATUS["reached"])
        & np.isfinite(weight)
        & (vz_end > 0)
        & np.isfinite(lx)
        & np.isfinite(ly)
        & np.isfinite(lts)
        & (weight > 0)
    )
    ib = np.searchsorted(edges, d_end, side="right") - 1
    ok &= (ib >= 0) & (ib < dia.size)
    ok &= np.isfinite(d_end)
    nx, ny, nt, nb = xg.size, yg.size, tg.size, dia.size
    counts = np.zeros(nt * ny * nx * nb)
    sel = np.nonzero(ok.ravel())[0]
    wv = weight.ravel()[sel]
    fx = (lx.ravel()[sel] - xg[0]) / dx
    fy = (ly.ravel()[sel] - yg[0]) / dy
    # the drops are shared linearly between the two size bins whose centres
    # enclose the diameter at landing (the number is conserved; assigning all of
    # them to one bin would make the bin concentrations alias)
    qd = np.interp(d_end.ravel()[sel], dia, np.arange(nb, dtype=np.float64))
    b0 = np.minimum(np.floor(qd).astype(np.int64), nb - 2)
    fb = np.clip(qd - b0, 0.0, 1.0)
    # an observation represents an interval of its ``duration`` centred on its
    # time: the drops are spread evenly over it in sub-packets
    dur = (xr.zeros_like(land) + xr.DataArray(duration)).transpose(*dims)
    dur = np.asarray(dur.values, dtype=np.float64).ravel()[sel]
    nsub = max(1, int(round(float(dur.max()) / dt))) if dur.size else 1
    i0 = np.floor(fx).astype(np.int64)
    j0 = np.floor(fy).astype(np.int64)
    ax, ay = fx - i0, fy - j0
    for sub in range(nsub):
        offset = ((sub + 0.5) / nsub - 0.5) * dur
        ft = (lts.ravel()[sel] + offset - tsec[0]) / dt
        k0 = np.floor(ft).astype(np.int64)
        at = ft - k0
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    for db in (0, 1):
                        w = (
                            (wv / nsub)
                            * (ax if di else 1 - ax)
                            * (ay if dj else 1 - ay)
                            * (at if dk else 1 - at)
                            * (fb if db else 1 - fb)
                        )
                        ii, jj, kk = i0 + di, j0 + dj, k0 + dk
                        inside = (
                            (ii >= 0)
                            & (ii < nx)
                            & (jj >= 0)
                            & (jj < ny)
                            & (kk >= 0)
                            & (kk < nt)
                        )
                        flat = ((kk * ny + jj) * nx + ii) * nb + b0 + db
                        counts += np.bincount(
                            flat[inside], weights=w[inside], minlength=counts.size
                        )
    counts = counts.reshape(nt, ny, nx, nb)
    nd_s = counts / (dx * dy * dt * width)
    dcoord = dia
    dens = xr.DataArray(
        nd_s,
        dims=("time", "y", "x", "diameter"),
        coords={"time": tg, "y": yg, "x": xg, "diameter": dcoord},
        name="ND",
        attrs={
            "units": "m-3 mm-1",
            "long_name": "Drop concentration per unit diameter at the surface",
        },
    )
    dens = dens.assign_coords(diameter_width=("diameter", width))
    m3 = (dens * dens["diameter"] ** 3 * wsrc).sum("diameter")
    m4 = (dens * dens["diameter"] ** 4 * wsrc).sum("diameter")
    nt_ = (dens * wsrc).sum("diameter")
    with np.errstate(invalid="ignore", divide="ignore"):
        dm = (m4 / m3).where(m3 > 0)
    return xr.Dataset(
        {
            "ND": dens,
            "NT": nt_.assign_attrs(units="m-3", long_name="Total drop concentration"),
            "DM": dm.assign_attrs(units="mm", long_name="Mass-weighted mean diameter"),
        },
        attrs={
            "reference": "Kumjian and Ryzhkov 2012; Dawson et al. 2015",
            "comment": "number flux conserving accumulation of landed drops",
        },
    )


def trajectory_matched_times(
    source,
    target,
    diameter,
    *,
    storm_motion,
    wind=None,
    profile=None,
    pattern_time=None,
    time=None,
    surface_height=0.0,
    offset=(0.0, 0.0, 0.0),
    evaporation=None,
    fall_speed="atlas1973",
    density_correction=True,
    wind_divergence=False,
    time_step=5.0,
    max_time=3600.0,
    scheme="rk4",
    evaporated_diameter=0.12,
    tolerance=1e-3,
    max_iterations=12,
    engine="auto",
    n_threads=None,
):
    """
    Times at which drops from a moving echo pattern reach a surface site.

    The echo pattern is frozen and moves with ``storm_motion`` :math:`c`. A
    point of it, observed at time :math:`t_g` at :math:`X_g` (a radar gate),
    is at :math:`X_g + c\\,\\tau` at time :math:`t_g + \\tau`. For every
    diameter this finds the release offset :math:`\\tau` for which drops
    released from the pattern point land at the target site along the storm
    motion, by a secant iteration on the along-motion miss distance, and so
    the time :math:`t_g + \\tau + T_f` at which the drops of that size
    observed aloft reach the site (:math:`T_f` is their fall time). The
    distance by which they miss the site across the motion cannot be removed
    with one site and is returned (``cross_miss``).

    For a uniform wind :math:`u` without evaporation this is the
    trajectory-pair construction of the Bayesian DSD retrieval
    (``ml/models/bayesian_dsd/match.py``),

    .. math::

        t(D) = t_g + \\tau_D - \\mathbf u\\cdot\\hat{\\mathbf c}\\,
               (\\tau_D - \\tau_\\mathrm{ref}) / |\\mathbf c| ,

    with the fall times :math:`\\tau_D` of the drops, and the tests check that
    it is reproduced; here variable winds, the air density along the path and
    evaporation are included.

    Parameters
    ----------
    source : xarray.Dataset or mapping
        Pattern points with ``x``, ``y``, ``z`` (m) and ``time``.
    target : xarray.Dataset or mapping
        Surface sites with ``x``, ``y`` and optionally ``z``; broadcast
        against ``source``.
    diameter : array-like or xarray.DataArray
        Drop diameters (mm) **at the source**.
    storm_motion : tuple of float
        Motion ``(u, v)`` (m s-1), not zero. It also moves the wind pattern of
        ``wind``.
    tolerance : float, optional
        Largest along-motion miss (m) accepted. Default 1e-3.
    max_iterations : int, optional
        Most secant iterations. Default 12.
    **kwargs
        As in :func:`rain_trajectories`.

    Returns
    -------
    xarray.Dataset
        On the broadcast dimensions plus ``diameter``: ``arrival_time``,
        ``release_offset`` (s), ``cross_miss`` (m, to the left of the motion),
        ``fall_time`` (s), ``landing_diameter``, ``evaporated_mass_fraction``,
        ``concentration_ratio``, ``status`` and ``converged``.

    Notes
    -----
    The secant search for the release offset (``tolerance`` 1e-3 m and
    ``max_iterations`` 12 are radarx choices) is a radarx construction on top
    of :func:`rain_trajectories` (Dawson et al. 2015; Kumjian and Ryzhkov
    2012 for size sorting).

    References
    ----------
    Dawson, D. T., E. R. Mansell, and M. R. Kumjian, 2015: Does wind shear
    cause hydrometeor size sorting? *J. Atmos. Sci.*, **72** (1), 340-348,
    https://doi.org/10.1175/JAS-D-14-0084.1

    Kumjian, M. R., and A. V. Ryzhkov, 2012: The impact of size sorting on
    the polarimetric radar variables. *J. Atmos. Sci.*, **69** (6),
    2042-2060, https://doi.org/10.1175/JAS-D-11-0125.1
    """
    _check_threads(n_threads)
    cx, cy = _motion(storm_motion)
    speed = math.hypot(cx, cy)
    if not speed > 0:
        raise ValueError("storm_motion must not be zero")
    ex, ey = cx / speed, cy / speed
    use_compiled = _use_compiled(engine)
    diameter = _validate_diameter(diameter)
    (xs, ys, zs), ts = _points(source, offset=offset, time=time)
    tgt = _surface_points(target, surface_height)
    txc = _component(tgt, "x").reset_coords(drop=True)
    tyc = _component(tgt, "y").reset_coords(drop=True)
    tzc = _component(tgt, "z").reset_coords(drop=True)
    xs, ys, zs, ts, txc, tyc, tzc = xr.broadcast(xs, ys, zs, ts, txc, tyc, tzc)
    epoch = _epoch_of(ts.values, wind)
    su = _setup(
        wind=wind,
        profile=profile,
        storm_motion=storm_motion,
        pattern_time=pattern_time,
        evaporation=evaporation,
        fall_speed=fall_speed,
        density_correction=density_correction,
        wind_divergence=wind_divergence,
        concentration=True,
        dispersion=None,
        time_step=time_step,
        max_time=max_time,
        scheme=scheme,
        evaporated_diameter=evaporated_diameter,
        direction="forward",
        store_path=0,
        epoch_hint=epoch,
    )
    tsec = _seconds(ts.values, epoch)
    dims, shape = xs.dims, xs.shape
    npts = int(np.prod(shape, dtype=np.int64))
    nd = diameter.size
    n = npts * nd

    def rep(a):
        a = np.asarray(a, dtype=np.float64).reshape(npts, 1)
        return np.broadcast_to(a, (npts, nd)).reshape(n)

    x0, y0, z0, t0 = (rep(a) for a in (xs.values, ys.values, zs.values, tsec))
    xt, yt, zt = (rep(a.values) for a in (txc, tyc, tzc))
    dval = np.broadcast_to(diameter.values.reshape(1, nd), (npts, nd)).reshape(n)
    tau = np.zeros(n)
    prev = None
    for it in range(int(max_iterations)):
        out, _p = _integrate(
            su,
            x0 + cx * tau,
            y0 + cy * tau,
            z0,
            t0 + tau,
            dval,
            zt,
            0,
            use_compiled,
            n_threads,
        )
        miss_x, miss_y = out[:, 0] - xt, out[:, 1] - yt
        g = miss_x * ex + miss_y * ey
        cross = -miss_x * ey + miss_y * ex
        converged = np.isfinite(g) & (np.abs(g) <= tolerance)
        tau_eval = tau
        if converged.all() or not np.isfinite(g).any() or it == max_iterations - 1:
            break
        slope = np.full(n, speed)
        if prev is not None:
            dtau = tau - prev[0]
            with np.errstate(invalid="ignore", divide="ignore"):
                sec = (g - prev[1]) / dtau
            usable = np.isfinite(sec) & (np.abs(dtau) > 0) & (sec > 0.1 * speed)
            slope = np.where(usable, sec, slope)
        prev = (tau, g)
        tau = np.where(np.isfinite(g) & ~converged, tau - g / slope, tau)
    layout = (dims, shape, nd, 1)
    pts = (xs, ys, zs, ts, tsec)
    ds = _wrap(su, epoch, layout, diameter, pts, out, np.zeros((n, 0, 5)), "forward", 0)
    ds["release_offset"] = (
        tuple(dims) + ("diameter",),
        tau_eval.reshape(shape + (nd,)),
        {"units": "s", "long_name": "Release time after the observation"},
    )
    t_arr = (
        tsec.reshape(shape + (1,))
        + ds["release_offset"].values
        + ds["fall_time"].values
    )
    if _is_time(ts.values):
        t_arr = _from_seconds(t_arr, epoch)
    ds["arrival_time"] = (
        tuple(dims) + ("diameter",),
        t_arr,
        {"standard_name": "time", "long_name": "Arrival time of the drops at the site"},
    )
    ds["cross_miss"] = (
        tuple(dims) + ("diameter",),
        cross.reshape(shape + (nd,)),
        {"units": "m", "long_name": "Miss distance across the storm motion"},
    )
    ds["converged"] = (
        tuple(dims) + ("diameter",),
        converged.reshape(shape + (nd,)),
        {"long_name": "Along-motion miss within tolerance"},
    )
    return ds.drop_vars(["landing_time"])


@accessor_method("dataset", "datatree", name="rain_trajectories")
def _rain_trajectories_accessor(self, diameter, **kwargs):
    """
    Trajectories of raindrops from the points of this object to the ground.

    The object holds the source points (``x``, ``y``, ``z`` and optionally
    ``time``), e.g. a gridded or georeferenced radar volume. See
    :func:`radarx.retrieve.rain_trajectories` for the parameters.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        Landing positions and times, diameters, evaporated mass fractions and
        concentration ratios per source point and size.
    """
    return rain_trajectories(self.xarray_obj, diameter, **kwargs)
