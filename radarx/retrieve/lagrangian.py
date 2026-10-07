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
the trilinear interpolation). With ``extend`` the first and last analyses are
also moved with the storm, unchanged, before the first and after the last
analysis time (the time morphing of Ziegler 2013b, sect. 2c, which assumes a
storm steady in its own frame).

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
A backward trajectory has reached the storm environment (Ziegler 2013a,
sect. 2a) when, after more than :math:`N = 76` steps, (i) :math:`Z_H < 0`
dBZ or (ii) :math:`w < 0.5` m s\\ :sup:`-1` for at least five consecutive
steps, or (iii) when it leaves the domain through a lateral boundary. These
tests are applied to backward trajectories only. A trajectory also stops at
the end of the wind time series, after ``max_steps`` steps, and at a missing
(NaN) wind. The reason is returned as a bit mask (``flags``):

=====  ====================================================================
bit    meaning
=====  ====================================================================
1      environment: :math:`Z_H` below ``env_dbz`` after ``min_steps`` steps
2      environment: :math:`w` below ``env_w`` for ``env_w_steps`` steps
4      left the analysed domain through a lateral boundary
8      reached the end of the wind time series
16     reached ``max_steps``
32     missing (NaN) wind at the parcel
=====  ====================================================================

Missing reflectivity is treated as no echo (``dbz_floor``). Missing winds stop
a trajectory: fill them first (e.g. with the background wind outside the
multi-Doppler coverage) where the parcels should continue.

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
    "dbz_floor": -30.0,
}

_REFLECTIVITY = ("DBZ", "DBZH", "reflectivity", "corrected_reflectivity", "dbz")

FLAG_MEANINGS = (
    "environment_reflectivity environment_weak_vertical_velocity lateral_boundary "
    "end_of_data max_steps missing_wind"
)


def _use_compiled(engine):
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
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
):
    """Winds as packed float32 (nt, nz, ny, nx, 4) plus coordinates."""
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

    def arr(name):
        return np.asarray(ds[name].transpose(*order).values, dtype=np.float64)

    uu, vv, ww = arr(u), arr(v), arr(w)
    zname = _reflectivity_name(ds, reflectivity)
    if zname is None:
        if reflectivity is not None and params["min_steps"] >= 0:
            warnings.warn(
                "no reflectivity in the winds: the reflectivity test of the "
                "environment is not applied",
                stacklevel=3,
            )
        dbz = np.full(uu.shape, 1.0e3)
    else:
        dbz = arr(zname)
        dbz = np.where(np.isnan(dbz), params["dbz_floor"], dbz)
    if surface_downdraft_on:
        ww = ww.copy()
        ww[:, 0] = surface_downdraft(
            ww,
            dbz,
            params["wmix0"],
            params["wmix1"],
            params["z0_dbz"],
            params["zddc_dbz"],
        )
    packed = np.stack([uu, vv, ww, dbz], axis=-1).astype(np.float32)
    cx, cy = (
        (0.0, 0.0)
        if storm_motion is None
        else (float(storm_motion[0]), float(storm_motion[1]))
    )
    if np.ndim(extend) == 0:
        eb = ea = float(extend)
    else:
        eb, ea = (float(e) for e in extend)
    if eb < 0 or ea < 0:
        raise ValueError("extend must be >= 0")
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


def _path_params(prep, direction, dt, iterations, max_steps, termination, params):
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
    env = 1.0 if (termination and direction == "backward") else 0.0
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
    surface_downdraft=True,
    termination=True,
    parameters=None,
    u="u",
    v="v",
    w="w",
    reflectivity="auto",
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
    storm_motion : (float, float), optional
        Constant storm motion :math:`(c_x, c_y)` in m s-1 with which every
        analysis is moved between its time and the parcel time. Default: no
        motion (fixed analyses).
    extend : float or (float, float), optional
        Seconds by which the series is extended before the first and after the
        last analysis by moving those analyses with ``storm_motion`` (time
        morphing). Default 0.
    surface_downdraft : bool, optional
        Replace the vertical velocity at the ground with the parameterised
        surface downdraft (Ziegler 2013a, eqs. 2-3). Default True.
    termination : bool, optional
        Stop backward trajectories that reach the storm environment
        (Ziegler 2013a, sect. 2a). Default True.
    parameters : dict, optional
        Overrides of :data:`TRAJECTORY_DEFAULTS`: ``offset_height`` (m),
        ``wmix0``, ``wmix1`` (m s-1), ``z0_dbz``, ``zddc_dbz`` (dBZ),
        ``min_steps``, ``env_dbz`` (dBZ), ``env_w`` (m s-1), ``env_w_steps``
        and ``dbz_floor`` (dBZ given to missing reflectivity).
    u, v, w : str, optional
        Names of the wind components. Default ``"u"``, ``"v"``, ``"w"``.
    reflectivity : str or None, optional
        Reflectivity (dBZ) variable, ``"auto"`` (default: the first of
        ``DBZ``, ``DBZH``, ``reflectivity``, ``corrected_reflectivity``,
        ``dbz``) or None (not used; the reflectivity test of the environment
        is then never met).
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation. ``"auto"`` (default) prefers the compiled kernel.
    n_threads : int, optional
        Threads of the compiled kernel. Default: all cores.

    Returns
    -------
    xarray.Dataset
        On ``(trajectory, step)``: ``x``, ``y``, ``z`` (m), ``time``
        (datetime64), ``u``, ``v``, ``w`` (m s-1, ``w`` with the surface
        downdraft) and ``reflectivity`` (dBZ) at every point (NaN after the
        end); on ``trajectory``: ``n_points``, ``flags`` (bit mask, see the
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
    )
    par = _path_params(prep, direction, dt, iterations, max_steps, termination, params)
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
                    "flag_masks": np.array([1, 2, 4, 8, 16, 32], np.int32),
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
            "surface_downdraft": int(bool(surface_downdraft)),
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
