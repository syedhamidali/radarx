#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Air Trajectories Through Multi-Doppler Winds
============================================

Backward and forward air trajectories from the points of a 3-D grid through a
time series of gridded winds (for example the analyses of
:func:`radarx.retrieve.multi_doppler`), as used by the diabatic Lagrangian
analysis of Ziegler (2013a, b; :mod:`radarx.retrieve.diabatic_lagrangian`).

Integration
-----------
Each time step :math:`\\Delta t` (20 s by default, Ziegler 2013a) is an Euler
predictor followed by three iterations of the trapezoidal corrector
(Ziegler et al. 2007; Ziegler 2013a, sect. 2b)

.. math::

    \\mathbf{x}^{(0)} = \\mathbf{x}_n \\pm \\Delta t\\,
    \\mathbf{v}(\\mathbf{x}_n, t_n),\\qquad
    \\mathbf{x}^{(i+1)} = \\mathbf{x}_n \\pm \\frac{\\Delta t}{2}
    \\left[\\mathbf{v}(\\mathbf{x}_n, t_n) +
    \\mathbf{v}(\\mathbf{x}^{(i)}, t_n \\pm \\Delta t)\\right],

with the sign negative for backward trajectories. The winds :math:`u, v, w`
and the reflectivity :math:`Z_H` are interpolated trilinearly in space from
the eight nodes of the grid cell holding the parcel and linearly in time
between the two analyses that bracket it. The parcel height is kept between
the lowest and the highest grid level.

Storm-motion advected grid
--------------------------
With a constant storm motion :math:`(c_x, c_y)` each analysis is moved with
the storm between its time :math:`t_i` and the parcel time :math:`t`: the
value of analysis :math:`i` at :math:`(x, y)` is read at
:math:`(x - c_x (t - t_i), y - c_y (t - t_i))` before the linear time
interpolation (Ziegler 2013a, sect. 2b). This is the advection of the grid
coordinates in a time-to-space sense described by Ziegler, evaluated directly
at the parcel (one interpolation instead of a bilinear re-gridding followed by
the trilinear interpolation). With ``extend_before`` / ``extend_after`` (or
``extend``) the first and last analyses are also moved with the storm,
unchanged, before the first and after the last analysis time (the time
morphing of Ziegler 2013b, sect. 2c and Fig. 1, which assumes a storm steady
in its own frame). ``storm_motion="estimate"`` takes the motion from the
reflectivity of the first and last analyses with
:func:`radarx.retrieve.estimate_motion`.

Surface parcels
---------------
Trajectories from the lowest grid level (the ground) start at the offset
height :math:`H_0` above it (10 m, Ziegler 2013a, Table 1). In precipitation
downdrafts the vertical velocity at the ground is replaced by the
parameterised surface downdraft of Ziegler (2013a, eqs. 2-3)

.. math::

    w_{sfc} = \\max\\left(Z^* w_{mix0}\\, w_{k=2},\\ w_{mix1}\\right)
    \\ \\text{if } w_{k=2} < 0,\\ \\text{else } 0,\\qquad
    Z^* = \\min\\left\\{\\max\\left[\\frac{Z_{H,k=1} - Z_0}
    {Z_{DDC} - Z_0}, 0\\right], 1\\right\\},

where :math:`w_{k=2}` is the vertical velocity at the first level above the
ground and :math:`Z_{H,k=1}` the reflectivity at the ground, with
:math:`w_{mix0} = 0.5`, :math:`w_{mix1} = -0.75` m s\\ :sup:`-1`,
:math:`Z_0 = 40` and :math:`Z_{DDC} = 50` dBZ (Table 1). The lowest grid
level must be the ground.

Termination
-----------
By default (``termination="ziegler2013"``) a backward trajectory has reached
the storm environment (Ziegler 2013a, sect. 2a) when, after more than
:math:`N = 76` steps, (i) :math:`Z_H < 0` dBZ or (ii) :math:`w < 0.5`
m s\\ :sup:`-1` for at least five consecutive steps, or (iii) when it leaves
the domain through a lateral boundary. Test (ii) is met by any parcel that is
not in an updraft, so in a long-lived cold pool (e.g. under the stratiform
rain of a squall line) surface trajectories end after about 26 min while still
inside the outflow.

``termination="precipitation"`` (an option of radarx, not part of Ziegler
2013a) instead requires the parcel to be outside precipitation, :math:`Z_H`
below ``env_dbz`` for ``env_dbz_steps`` consecutive steps (after
``min_steps``), *and* either at least ``cold_pool_depth`` above the ground or
where ``environment_mask`` is true (for example the air ahead of the gust
front). Trajectories that never meet the test stop at ``max_steps`` (the
maximum backward duration) or at the start of the data and are not
environmental. Air in a long-lived squall-line cold pool is often older than
the wind series; time morphing before the first analysis
(``extend_before``) lets it be traced back to the inflow.

Lateral boundaries
------------------
By default a parcel that leaves the analysed domain through a lateral
boundary has reached the environment (test iii). Where the domain edge lies
inside the storm (e.g. the rear edge of an analysis that cuts through a
trailing cold pool) this gives such parcels environmental values.
``boundary`` restricts the rule: to exits where ``environment_mask`` is true
(``"environment_mask"``), where the boundary point is outside precipitation
or in the mask (``"no_echo"``), or to a list of sides (e.g. ``["east",
"north", "south"]``). Other exits end the trajectory with flag 128; they are
not environmental (the DLA hole-fills them).

The tests are applied to backward trajectories only. A trajectory also stops
at the end of the wind time series, after ``max_steps`` steps, and at a
missing (NaN) wind. The reason is returned as a bit mask (``flags``):

=====  ====================================================================
bit    meaning
=====  ====================================================================
1      environment: outside precipitation (:math:`Z_H` test)
2      environment: :math:`w` below ``env_w`` for ``env_w_steps`` steps
4      left the analysed domain through a lateral boundary
8      reached the end of the wind time series
16     reached ``max_steps``
32     missing (NaN) wind at the parcel
64     (diabatic Lagrangian analysis) too little time in valid winds
128    left through a lateral boundary that does not count as environment
       (``boundary``)
=====  ====================================================================

Missing reflectivity is treated as no echo (``dbz_floor``). Missing winds stop
a trajectory: fill them first (e.g. with the background wind outside the
multi-Doppler coverage) where the parcels should continue, and pass the
coverage as ``valid`` to obtain ``valid_fraction``, the fraction of the
trajectory points in analysed winds.

Computation
-----------
All trajectories run in one call of a compiled kernel
(``radarx.retrieve._lagrangian``, C++, multithreaded over trajectories with
the GIL released); an identical NumPy implementation is used when the kernel
is not built (``engine="numpy"``) and serves as its test oracle.

References
----------
Ziegler, C. L., 2013a: A diabatic Lagrangian technique for the analysis of
convective storms. Part I: Description and validation via an observing system
simulation experiment. *J. Atmos. Oceanic Technol.*, **30** (10), 2248-2265,
https://doi.org/10.1175/JTECH-D-12-00194.1

Ziegler, C. L., 2013b: A diabatic Lagrangian technique for the analysis of
convective storms. Part II: Application to a radar-observed storm. *J. Atmos.
Oceanic Technol.*, **30** (10), 2266-2280,
https://doi.org/10.1175/JTECH-D-13-00036.1

Ziegler, C. L., M. S. Buban, and E. N. Rasmussen, 2007: A Lagrangian
objective analysis technique for assimilating in situ observations with
multiple-radar-derived airflow. *Mon. Wea. Rev.*, **135** (7), 2417-2442,
https://doi.org/10.1175/MWR3396.1

.. autosummary::
   :nosignatures:
   :toctree: generated/

   trajectories
"""

__all__ = ["trajectories"]

import math
import warnings

import numpy as np
import xarray as xr

from . import _lagrangian_numpy as _np_kernel

try:
    from . import _lagrangian

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _lagrangian = None
    HAS_COMPILED_KERNEL = False

#: Parameters of the trajectories and the surface downdraft (Ziegler 2013a,
#: Table 1 and sect. 2a).
TRAJECTORY_DEFAULTS = {
    "offset_height": 10.0,  # H0, m
    "wmix0": 0.5,
    "wmix1": -0.75,  # m s-1
    "z0_dbz": 40.0,  # Z0, dBZ
    "zddc_dbz": 50.0,  # Z_DDC, dBZ
    "min_steps": 76,
    "env_dbz": 0.0,
    "env_w": 0.5,
    "env_w_steps": 5,
    "env_dbz_steps": 5,  # termination="precipitation"
    "cold_pool_depth": 2000.0,  # m above the ground, termination="precipitation"
    "dbz_floor": -30.0,
}

#: Termination modes of backward trajectories.
TERMINATION = {"ziegler2013": 1, "precipitation": 2}

_REFLECTIVITY = ("DBZ", "DBZH", "reflectivity", "corrected_reflectivity", "dbz")

FLAG_MEANINGS = (
    "environment_reflectivity environment_weak_vertical_velocity lateral_boundary "
    "end_of_data max_steps missing_wind outside_valid_winds "
    "lateral_boundary_not_environment"
)
FLAG_MASKS = np.array([1, 2, 4, 8, 16, 32, 64, 128], np.int32)

#: Rules for lateral-boundary exits of backward trajectories (``boundary=``).
BOUNDARY_RULES = {"environment": 0, "environment_mask": 1, "no_echo": 2}
_SIDES = {"west": 1, "east": 2, "south": 4, "north": 8}


def _use_compiled(engine):
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:  # pragma: no cover
        raise ImportError("the compiled trajectory kernel is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


def _parameters(parameters, defaults):
    out = dict(defaults)
    if parameters:
        unknown = set(parameters) - set(defaults)
        if unknown:
            raise ValueError(f"unknown parameters: {sorted(unknown)}")
        out.update(parameters)
    return out


def _axis(ds, name):
    if name not in ds.coords:
        raise ValueError(f"the winds need a {name!r} coordinate")
    c = np.asarray(ds[name].values, dtype=np.float64)
    if c.ndim != 1 or c.size < 2 or not np.all(np.diff(c) > 0):
        raise ValueError(f"{name!r} must be 1-D, increasing, with at least 2 points")
    return c


def _reflectivity_name(ds, reflectivity):
    if reflectivity is None:
        return None
    if reflectivity != "auto":
        if reflectivity not in ds:
            raise ValueError(f"reflectivity {reflectivity!r} not in the winds")
        return reflectivity
    for name in _REFLECTIVITY:
        if name in ds:
            return name
    return None


def surface_downdraft(w, dbz, wmix0=0.5, wmix1=-0.75, z0_dbz=40.0, zddc_dbz=50.0):
    """Parameterised surface downdraft of Ziegler (2013a, eqs. 2-3).

    ``w`` and ``dbz`` are NumPy arrays with the height on axis -3; returns the
    vertical velocity at the lowest level.
    """
    w2 = w[..., 1, :, :]
    zs = np.clip(
        (np.nan_to_num(dbz[..., 0, :, :], nan=-np.inf) - z0_dbz) / (zddc_dbz - z0_dbz),
        0.0,
        1.0,
    )
    return np.where(w2 < 0.0, np.maximum(zs * wmix0 * w2, wmix1), 0.0)


def _analysis_time(times, time, direction):
    if time is None:
        return times[-1] if direction == "backward" else times[0]
    if isinstance(time, xr.DataArray):
        time = time.values
    return np.datetime64(np.asarray(time).astype("datetime64[ns]").item(), "ns")


def _prepare(
    winds,
    time,
    direction,
    *,
    u,
    v,
    w,
    reflectivity,
    storm_motion,
    extend,
    surface_downdraft_on,
    params,
    valid=None,
    environment_mask=None,
    extend_before=None,
    extend_after=None,
):
    """Winds as packed float32 (nt, nz, ny, nx, 6) plus coordinates."""
    if not isinstance(winds, xr.Dataset):
        raise TypeError("winds must be an xarray.Dataset")
    for name in (u, v, w):
        if name not in winds:
            raise ValueError(f"the winds need the variable {name!r}")
    ds = winds
    if "time" not in ds.dims:
        if "time" not in ds.coords:
            ds = ds.assign_coords(time=np.datetime64("1970-01-01T00:00:00", "ns"))
        ds = ds.expand_dims("time")
    ds = ds.sortby("time")
    x, y, z = (_axis(ds, c) for c in ("x", "y", "z"))
    times = ds["time"].values.astype("datetime64[ns]")
    ta = np.datetime64(_analysis_time(times, time, direction), "ns")
    t = (times - ta) / np.timedelta64(1, "s")
    t = np.asarray(t, dtype=np.float64)
    if np.any(np.diff(t) <= 0):
        raise ValueError("the wind times must be distinct")
    order = ("time", "z", "y", "x")

    shape = tuple(ds.sizes[d] for d in order)
    packed = np.empty(shape + (6,), dtype=np.float32)
    for i, name in enumerate((u, v, w)):
        packed[..., i] = ds[name].transpose(*order).values
    for i, (item, default) in ((4, (valid, 1.0)), (5, (environment_mask, 0.0))):
        if item is None:
            packed[..., i] = default
            continue
        da = ds[item] if isinstance(item, str) else item
        if not isinstance(da, xr.DataArray):
            raise TypeError(
                "valid and environment_mask must be variable names or DataArrays"
            )
        if "time" not in da.dims and "time" in da.coords:
            da = da.drop_vars("time")
        da = da.broadcast_like(ds[u]).transpose(*order)
        packed[..., i] = (
            np.nan_to_num(np.asarray(da.values, dtype=np.float64)) > 0
        ).astype(np.float32)
    zname = _reflectivity_name(ds, reflectivity)
    if zname is None:
        if reflectivity is not None and params["min_steps"] >= 0:
            warnings.warn(
                "no reflectivity in the winds: the reflectivity test of the "
                "environment is not applied",
                stacklevel=3,
            )
        packed[..., 3] = 1.0e3
    else:
        packed[..., 3] = ds[zname].transpose(*order).values
        dbz = packed[..., 3]
        dbz[np.isnan(dbz)] = params["dbz_floor"]
    if surface_downdraft_on:
        packed[:, 0, :, :, 2] = surface_downdraft(
            packed[:, :2, :, :, 2].astype(np.float64),
            packed[:, :1, :, :, 3].astype(np.float64),
            params["wmix0"],
            params["wmix1"],
            params["z0_dbz"],
            params["zddc_dbz"],
        )
    cx, cy = _storm_motion(storm_motion, ds, zname)
    if np.ndim(extend) == 0:
        eb = ea = float(extend)
    else:
        eb, ea = (float(e) for e in extend)
    if extend_before is not None:
        eb = float(extend_before)
    if extend_after is not None:
        ea = float(extend_after)
    if eb < 0 or ea < 0:
        raise ValueError("extend, extend_before and extend_after must be >= 0")
    if (eb > 0 or ea > 0) and storm_motion is None and t.size > 1:
        warnings.warn(
            "time morphing (extend) without storm_motion holds the first and last "
            "analyses fixed; pass the storm motion (or 'estimate')",
            stacklevel=3,
        )
    if t.size == 1 and eb == 0 and ea == 0 and storm_motion is None:
        raise ValueError(
            "a single wind analysis needs storm_motion and extend (time morphing)"
        )
    return {
        "ds": ds,
        "packed": packed,
        "x": x,
        "y": y,
        "z": z,
        "t": t,
        "time": ta,
        "cx": cx,
        "cy": cy,
        "eb": eb,
        "ea": ea,
        "reflectivity": zname,
    }


def _storm_motion(storm_motion, ds, zname):
    """(c_x, c_y) from a pair, an estimate_motion Dataset or "estimate"."""
    if storm_motion is None:
        return 0.0, 0.0
    if isinstance(storm_motion, str):
        if storm_motion != "estimate":
            raise ValueError(
                f"storm_motion must be (c_x, c_y), a Dataset or 'estimate', not {storm_motion!r}"
            )
        if zname is None or ds.sizes["time"] < 2:
            raise ValueError(
                "storm_motion='estimate' needs reflectivity at two or more wind times"
            )
        from .advection import estimate_motion

        # median of the motions between consecutive analyses (robust to echoes
        # entering or leaving the domain and to storm evolution)
        refl = ds[[zname]]
        for c in ("x", "y"):
            # coordinates stored in float32 are evenly spaced only to rounding
            a = np.asarray(refl[c].values, np.float64)
            even = np.linspace(a[0], a[-1], a.size)
            if np.allclose(a, even, rtol=0.0, atol=1e-4 * abs(a[1] - a[0])):
                refl = refl.assign_coords({c: even})
        uv = []
        for i in range(ds.sizes["time"] - 1):
            m = estimate_motion(refl.isel(time=i), refl.isel(time=i + 1), field=zname)
            uv.append((float(m["u"]), float(m["v"])))
        uv = np.array(uv)
        uv = uv[np.isfinite(uv).all(axis=1)]
        storm_motion = (
            (np.nan, np.nan) if uv.size == 0 else tuple(np.median(uv, axis=0))
        )
    if isinstance(storm_motion, xr.Dataset):
        if "u" not in storm_motion or "v" not in storm_motion:
            raise ValueError("a storm_motion Dataset needs 'u' and 'v'")
        if storm_motion["u"].ndim or storm_motion["v"].ndim:
            raise ValueError("storm_motion must be a single (domain-wide) motion")
        storm_motion = (float(storm_motion["u"]), float(storm_motion["v"]))
    cx, cy = (float(storm_motion[0]), float(storm_motion[1]))
    if not (np.isfinite(cx) and np.isfinite(cy)):
        raise ValueError(
            "the storm motion is not finite (e.g. the estimate found no correlation)"
        )
    return cx, cy


def _boundary_rule(boundary):
    """(rule, sides) of the lateral-boundary option."""
    if isinstance(boundary, str):
        if boundary in BOUNDARY_RULES:
            return BOUNDARY_RULES[boundary], 0
        boundary = [boundary]
    try:
        sides = [str(b).lower() for b in boundary]
    except TypeError:
        sides = [None]
    if not sides or any(b not in _SIDES for b in sides):
        raise ValueError(
            f"boundary must be one of {sorted(BOUNDARY_RULES)} or a sequence of the "
            f"sides {sorted(_SIDES)}, not {boundary!r}"
        )
    bits = 0
    for b in sides:
        bits |= _SIDES[b]
    return 3, bits


def _path_params(
    prep,
    direction,
    dt,
    iterations,
    max_steps,
    termination,
    params,
    boundary="environment",
):
    if direction not in ("backward", "forward"):
        raise ValueError("direction must be 'backward' or 'forward'")
    if not dt > 0:
        raise ValueError("dt must be positive")
    if iterations < 0:
        raise ValueError("iterations must be >= 0")
    t = prep["t"]
    if max_steps is None:
        span = (-t[0] + prep["eb"]) if direction == "backward" else (t[-1] + prep["ea"])
        max_steps = max(int(math.ceil(span / dt - 1e-9)), 0)
    sign = -1.0 if direction == "backward" else 1.0
    if termination is True:
        termination = "ziegler2013"
    if termination in (False, None):
        mode = 0
    elif termination in TERMINATION:
        mode = TERMINATION[termination]
    else:
        raise ValueError(
            f"termination must be True, False, 'ziegler2013' or 'precipitation', not {termination!r}"
        )
    env = float(mode) if direction == "backward" else 0.0
    rule, sides = _boundary_rule(boundary)
    return np.array(
        [
            float(dt),
            float(iterations),
            float(max_steps),
            sign,
            env,
            float(params["min_steps"]),
            float(params["env_dbz"]),
            float(params["env_w"]),
            float(params["env_w_steps"]),
            float(params["env_dbz_steps"]),
            float(params["cold_pool_depth"]),
            float(prep["z"][0]),
            float(rule),
            float(sides),
        ]
    )


def _grid_starts(prep, levels, offset_height):
    """All grid points (or those of ``levels``) as (n, 3) starts and indices."""
    z, y, x = prep["z"], prep["y"], prep["x"]
    ks = np.arange(z.size) if levels is None else np.asarray(levels, dtype=np.int64)
    kk, jj, ii = np.meshgrid(ks, np.arange(y.size), np.arange(x.size), indexing="ij")
    zs = z[kk] + np.where(kk == 0, offset_height, 0.0)
    starts = np.column_stack([x[ii].ravel(), y[jj].ravel(), zs.ravel()])
    return starts, (kk.ravel(), jj.ravel(), ii.ravel()), ks


def _run_paths(prep, par, starts, use_compiled, n_threads):
    if use_compiled:
        return _lagrangian.trajectories(
            prep["packed"],
            prep["x"],
            prep["y"],
            prep["z"],
            prep["t"],
            np.ascontiguousarray(starts, dtype=np.float64),
            par,
            prep["cx"],
            prep["cy"],
            prep["eb"],
            prep["ea"],
            n_threads=int(n_threads or 0),
        )
    g = _np_kernel.Grid(
        prep["x"],
        prep["y"],
        prep["z"],
        prep["t"],
        prep["packed"],
        prep["cx"],
        prep["cy"],
        prep["eb"],
        prep["ea"],
    )
    return _np_kernel.build_paths(g, par, starts)


def _valid_fraction(val, npts):
    steps = np.arange(val.shape[1])[None, :] < npts[:, None]
    nvalid = ((np.nan_to_num(val[..., 4]) >= 0.5) & steps).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(npts > 0, nvalid / np.maximum(npts, 1), np.nan)


def trajectories(
    winds,
    time=None,
    *,
    direction="backward",
    start=None,
    levels=None,
    dt=20.0,
    iterations=3,
    max_steps=None,
    storm_motion=None,
    extend=0.0,
    extend_before=None,
    extend_after=None,
    surface_downdraft=True,
    termination=True,
    boundary="environment",
    parameters=None,
    u="u",
    v="v",
    w="w",
    reflectivity="auto",
    valid=None,
    environment_mask=None,
    engine="auto",
    n_threads=None,
):
    """
    Backward or forward air trajectories through a time series of 3-D winds.

    Parameters
    ----------
    winds : xarray.Dataset
        Winds on ``(time, z, y, x)`` (any order; ``time`` may be absent for a
        single analysis used with ``storm_motion`` and ``extend``), with
        coordinates ``x``, ``y`` (m, increasing) and ``z`` (m, increasing;
        the lowest level is the ground), e.g. a time series of
        :func:`radarx.retrieve.multi_doppler` analyses joined with
        ``xr.concat(..., "time")``. Ground-relative winds in m s-1.
    time : datetime-like, optional
        Analysis time at which the trajectories start. Default: the last wind
        time for backward and the first for forward trajectories. It need not
        be one of the wind times.
    direction : {"backward", "forward"}, optional
        Integration direction. Default ``"backward"``.
    start : xarray.Dataset or dict, optional
        Start points as ``x``, ``y``, ``z`` arrays (m) of the same shape.
        Default: every grid point (those of ``levels``), with points of the
        lowest level started at the offset height above it.
    levels : sequence of int, optional
        Grid level indices to start from when ``start`` is not given. Default:
        all levels.
    dt : float, optional
        Time step in s. Default 20 (Ziegler 2013a).
    iterations : int, optional
        Corrector iterations per step. Default 3.
    max_steps : int, optional
        Maximum number of steps. Default: enough to reach the end of the wind
        time series (plus ``extend``).
    storm_motion : (float, float), xarray.Dataset or "estimate", optional
        Constant storm motion :math:`(c_x, c_y)` in m s-1 with which every
        analysis is moved between its time and the parcel time: a pair, the
        domain-wide output of :func:`radarx.retrieve.estimate_motion` (``u``,
        ``v``), or ``"estimate"`` to estimate it with
        :func:`radarx.retrieve.estimate_motion` from the reflectivity of the
        first and last wind times. Default: no motion (fixed analyses).
    extend : float or (float, float), optional
        Seconds by which the series is extended before the first and after the
        last analysis by moving those analyses with ``storm_motion`` (time
        morphing, Ziegler 2013b, sect. 2c). Default 0.
    extend_before, extend_after : float, optional
        Seconds of time morphing before the first and after the last analysis;
        override the corresponding part of ``extend``. For backward
        trajectories ``extend_before`` lets parcels of an air mass older than
        the wind series (e.g. a squall-line cold pool) reach the environment.
    surface_downdraft : bool, optional
        Replace the vertical velocity at the ground with the parameterised
        surface downdraft (Ziegler 2013a, eqs. 2-3). Default True.
    termination : {True, "ziegler2013", "precipitation", False}, optional
        Environment test of backward trajectories. ``True`` or
        ``"ziegler2013"`` (default): the tests of Ziegler (2013a, sect. 2a).
        ``"precipitation"``: the parcel is outside precipitation
        (:math:`Z_H` below ``env_dbz`` for ``env_dbz_steps`` steps) and
        either above ``cold_pool_depth`` or where ``environment_mask`` is
        true (e.g. ahead of the gust front); see the module documentation.
        ``False``: no test (trajectories run to ``max_steps`` or the data).
    boundary : str or sequence of str, optional
        Which exits through a lateral boundary count as reaching the
        environment (backward trajectories with a termination test).
        ``"environment"`` (default, Ziegler 2013a, sect. 2a, test iii): every
        exit. ``"environment_mask"``: only where ``environment_mask`` is true
        at the last point inside the domain. ``"no_echo"``: only where the
        reflectivity there is below ``env_dbz`` or the mask is true. A
        sequence of sides (``"west"``, ``"east"``, ``"south"``, ``"north"``;
        the domain edges at the smallest and largest ``x`` and ``y``): only
        exits through those sides. Other exits stop the trajectory with flag
        128 and are not environmental.
    parameters : dict, optional
        Overrides of :data:`TRAJECTORY_DEFAULTS`: ``offset_height`` (m),
        ``wmix0``, ``wmix1`` (m s-1), ``z0_dbz``, ``zddc_dbz`` (dBZ),
        ``min_steps``, ``env_dbz`` (dBZ), ``env_w`` (m s-1), ``env_w_steps``,
        ``env_dbz_steps``, ``cold_pool_depth`` (m above the ground) and
        ``dbz_floor`` (dBZ given to missing reflectivity).
    u, v, w : str, optional
        Names of the wind components. Default ``"u"``, ``"v"``, ``"w"``.
    reflectivity : str or None, optional
        Reflectivity (dBZ) variable, ``"auto"`` (default: the first of
        ``DBZ``, ``DBZH``, ``reflectivity``, ``corrected_reflectivity``,
        ``dbz``) or None (not used; the reflectivity test of the environment
        is then never met).
    valid : str or xarray.DataArray, optional
        True (non-zero) where the winds are analysed, e.g. ``"dd_valid"`` of
        :func:`radarx.retrieve.multi_doppler` output, and false where they are
        a background or filled. Interpolated to the parcels; the fraction of
        trajectory points with a value of at least 0.5 is returned as
        ``valid_fraction``. Default: all valid.
    environment_mask : str or xarray.DataArray, optional
        True where a parcel may count as environmental for
        ``termination="precipitation"`` below ``cold_pool_depth`` (e.g. the
        air ahead of the gust front). Default: nowhere.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation. ``"auto"`` (default) prefers the compiled kernel.
    n_threads : int, optional
        Threads of the compiled kernel. Default: all cores.

    Returns
    -------
    xarray.Dataset
        On ``(trajectory, step)``: ``x``, ``y``, ``z`` (m), ``time``
        (datetime64), ``u``, ``v``, ``w`` (m s-1, ``w`` with the surface
        downdraft), ``reflectivity`` (dBZ) and ``valid`` (0-1) at every
        point (NaN after the end); on ``trajectory``: ``n_points``,
        ``valid_fraction``, ``flags`` (bit mask, see the
        module documentation), ``environment`` (True where the trajectory
        reached the storm environment or a lateral boundary) and the start
        point ``start_x``, ``start_y``, ``start_z`` (plus the grid indices
        ``k``, ``j``, ``i`` for gridpoint trajectories).

    References
    ----------
    Ziegler, C. L., 2013a, *J. Atmos. Oceanic Technol.*, **30**, 2248-2265,
    https://doi.org/10.1175/JTECH-D-12-00194.1

    Examples
    --------
    >>> import numpy as np, xarray as xr
    >>> from radarx.retrieve import trajectories
    >>> c = np.arange(0.0, 10001.0, 1000.0)
    >>> t = np.array(["2022-03-30T23:00", "2022-03-30T23:10"], "datetime64[ns]")
    >>> ds = xr.Dataset(
    ...     {k: (("time", "z", "y", "x"), np.full((2, 11, 11, 11), val))
    ...      for k, val in (("u", 5.0), ("v", 0.0), ("w", 0.0))},
    ...     coords={"time": t, "z": c, "y": c, "x": c})
    >>> tr = trajectories(ds, start={"x": [8000.0], "y": [5000.0], "z": [3000.0]},
    ...                   reflectivity=None)
    >>> float(tr.x.isel(step=int(tr.n_points[0]) - 1)[0])
    5000.0
    """
    params = _parameters(parameters, TRAJECTORY_DEFAULTS)
    use_compiled = _use_compiled(engine)
    prep = _prepare(
        winds,
        time,
        direction,
        u=u,
        v=v,
        w=w,
        reflectivity=reflectivity,
        storm_motion=storm_motion,
        extend=extend,
        surface_downdraft_on=surface_downdraft,
        params=params,
        valid=valid,
        environment_mask=environment_mask,
        extend_before=extend_before,
        extend_after=extend_after,
    )
    par = _path_params(
        prep, direction, dt, iterations, max_steps, termination, params, boundary
    )
    index = None
    if start is None:
        starts, index, _ = _grid_starts(prep, levels, params["offset_height"])
    else:
        sx, sy, sz = (
            np.asarray(start[k], dtype=np.float64).ravel() for k in ("x", "y", "z")
        )
        if not (sx.size == sy.size == sz.size):
            raise ValueError("start x, y and z must have the same size")
        starts = np.column_stack([sx, sy, sz])
    m = int(par[2]) + 1
    nbytes = starts.shape[0] * m * 8 * 8
    if nbytes > 8e9:
        raise MemoryError(
            f"{starts.shape[0]} trajectories of {m} points need {nbytes / 1e9:.0f} GB; "
            "pass fewer start points (start=, levels=) or a smaller max_steps"
        )
    pos, val, npts, flags = _run_paths(prep, par, starts, use_compiled, n_threads)
    pos = np.asarray(pos)
    val = np.asarray(val)
    npts = np.asarray(npts, dtype=np.int64)
    flags = np.asarray(flags, dtype=np.int32)
    tsec = pos[..., 3]
    tt = np.where(
        np.isnan(tsec),
        np.datetime64("NaT", "ns"),
        prep["time"] + np.round(np.nan_to_num(tsec) * 1e9).astype("timedelta64[ns]"),
    )
    dims = ("trajectory", "step")
    out = xr.Dataset(
        {
            "x": (dims, pos[..., 0], {"long_name": "parcel x", "units": "m"}),
            "y": (dims, pos[..., 1], {"long_name": "parcel y", "units": "m"}),
            "z": (dims, pos[..., 2], {"long_name": "parcel height", "units": "m"}),
            "time": (dims, tt, {"long_name": "parcel time"}),
            "u": (
                dims,
                val[..., 0],
                {"standard_name": "eastward_wind", "units": "m s-1"},
            ),
            "v": (
                dims,
                val[..., 1],
                {"standard_name": "northward_wind", "units": "m s-1"},
            ),
            "w": (
                dims,
                val[..., 2],
                {"standard_name": "upward_air_velocity", "units": "m s-1"},
            ),
            "reflectivity": (
                dims,
                val[..., 3],
                {"long_name": "radar reflectivity at the parcel", "units": "dBZ"},
            ),
            "valid": (
                dims,
                val[..., 4],
                {
                    "long_name": "interpolated validity of the winds at the parcel",
                    "units": "1",
                },
            ),
            "valid_fraction": (
                "trajectory",
                _valid_fraction(val, npts),
                {
                    "long_name": "fraction of trajectory points with valid winds",
                    "units": "1",
                },
            ),
            "n_points": (
                "trajectory",
                npts,
                {"long_name": "number of trajectory points"},
            ),
            "flags": (
                "trajectory",
                flags,
                {
                    "long_name": "trajectory termination flags",
                    "flag_masks": FLAG_MASKS,
                    "flag_meanings": FLAG_MEANINGS,
                },
            ),
            "environment": (
                "trajectory",
                (flags & _np_kernel.ENVIRONMENT) != 0,
                {"long_name": "trajectory reached the storm environment"},
            ),
        },
        coords={
            "step": ("step", np.arange(m)),
            "start_x": ("trajectory", starts[:, 0], {"units": "m"}),
            "start_y": ("trajectory", starts[:, 1], {"units": "m"}),
            "start_z": ("trajectory", starts[:, 2], {"units": "m"}),
            "analysis_time": prep["time"],
        },
        attrs={
            "direction": direction,
            "dt": float(dt),
            "iterations": int(iterations),
            "storm_motion": (prep["cx"], prep["cy"]),
            "extend": (prep["eb"], prep["ea"]),
            "surface_downdraft": int(bool(surface_downdraft)),
            "termination": str(termination),
            "boundary": str(boundary),
            "method": "Ziegler (2013a) gridpoint trajectories",
        },
    )
    if index is not None:
        out = out.assign_coords(
            k=("trajectory", index[0]),
            j=("trajectory", index[1]),
            i=("trajectory", index[2]),
        )
    return out


from .._registry import accessor_method  # noqa: E402


@accessor_method("dataset", name="trajectories")
def _trajectories_dataset_accessor(self, time=None, **kwargs):
    """
    Air trajectories through the time series of 3-D winds in this dataset.

    See :func:`radarx.retrieve.trajectories` for the parameters.

    Returns
    -------
    xarray.Dataset
        Trajectory positions, times, winds and termination flags.
    """
    return trajectories(self.xarray_obj, time, **kwargs)
