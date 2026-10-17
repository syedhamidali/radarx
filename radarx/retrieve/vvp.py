#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

r"""
Volume Velocity Processing: Linear Wind Fields from One Doppler Radar
=====================================================================

A single Doppler radar measures the radial velocity :math:`v_r` only. If the
wind is a linear function of position inside a region (an *analysis volume*),
the radial velocities of all gates of the region over-determine the wind at its
centre and the gradients of the wind. This is volume velocity processing (VVP,
Waldteufel and Corbin [1]_), the generalisation of the velocity-azimuth display
(VAD, :func:`radarx.retrieve.vad_profile`) from a circle around the radar to a
volume anywhere in the scan. radarx follows the formulation of Doviak and
Zrnic [2]_ (section 9.3).

Model
-----
About the centre :math:`(x_0, y_0, z_0)` of an analysis volume the wind
:math:`(u, v, w)` is a first-order Taylor series (Eq. 9.7 of [2]_),

.. math::

    \mathbf{V}(x, y, z) = \mathbf{V}_0 + \frac{\partial\mathbf{V}}{\partial x}
    (x - x_0) + \frac{\partial\mathbf{V}}{\partial y}(y - y_0)
    + \frac{\partial\mathbf{V}}{\partial z}(z - z_0).

A gate at the azimuth :math:`\alpha` (clockwise from north) and the local
elevation angle :math:`\theta` of the beam has the radial velocity
:math:`v_r = \mathbf{V}\cdot(\cos\theta\sin\alpha, \cos\theta\cos\alpha,
\sin\theta)`. With the horizontal gate position :math:`x = s\sin\alpha`,
:math:`y = s\cos\alpha` this is linear in eleven unknowns (Eqs. 9.11 and 9.13
of [2]_),

.. math::

    v_r &= u_0'\cos\theta\sin\alpha + u_x\cos\theta\sin\alpha\,(x - x_0)
    + u_z\cos\theta\sin\alpha\,(z - z_0) \\
    &+ v_0'\cos\theta\cos\alpha + v_y\cos\theta\cos\alpha\,(y - y_0)
    + v_z\cos\theta\cos\alpha\,(z - z_0) \\
    &+ (u_y + v_x)\,\tfrac12\cos\theta\,[(x - x_0)\cos\alpha
    + (y - y_0)\sin\alpha] \\
    &+ w_0\sin\theta + w_x\sin\theta\,(x - x_0) + w_y\sin\theta\,(y - y_0)
    + w_z\sin\theta\,(z - z_0),

with the *modified* wind components :math:`u_0' = u_0 + \tfrac12 y_0
(v_x - u_y)` and :math:`v_0' = v_0 - \tfrac12 x_0 (v_x - u_y)`. The vorticity
:math:`v_x - u_y` has the same dependence on the gate position as :math:`u_0`
and :math:`v_0` and cannot be separated from them (Doviak and Zrnic [2]_);
:math:`u_0'` and :math:`v_0'` are the wind at the centre only if the vorticity
term :math:`\tfrac12 y_0\zeta` is small, which it is not for a volume 50 km from
the radar in a cyclone (a vorticity of :math:`10^{-3}` s-1 gives 25 m s-1).
The option ``vorticity`` removes the term for a vorticity that is known.
The unknowns are found by least squares from the gates of the volume; the
covariance of the estimates is :math:`s^2 (P^T P)^{-1}` with the residual
variance :math:`s^2` (Eq. 9.18 of [2]_), which gives the standard errors and
the error of the divergence :math:`u_x + v_y`.

Models
------
The number of unknowns is chosen with ``model`` (the model must be adequate,
that is, include the terms that matter, or the estimates are biased: Eq. 9.17
of [2]_).

``"uniform"``
    :math:`u_0', v_0'` (2 unknowns): a wind that is uniform in the volume, the
    VAD wind of an arc.
``"linear"``
    :math:`u_0', v_0', u_x, v_y, u_y + v_x, u_z, v_z` (7): the horizontal wind
    with its horizontal gradients and vertical shear, the vertical motion
    neglected; the model of the low-level divergence example of [2]_
    (section 9.3.4). This is the default.
``"linear_w"``
    ``"linear"`` and :math:`w_0, w_z` (9): :math:`w_x = w_y = 0` above flat
    ground (Doviak and Zrnic, problem 9.9).
``"full"``
    all eleven unknowns.

Velocity :math:`w` and the fall speed :math:`v_t` enter the radial velocity
only as :math:`(w - v_t)\sin\theta`; with ``fall_speed`` the fall speed is
removed from :math:`v_r` before the fit (:math:`v_r + v_t\sin\theta`),
otherwise :math:`w_0` is the vertical motion of the hydrometeors.

Choices that are radarx's own
-----------------------------
The analysis volume is a cylinder of radius ``radius`` and depth ``depth`` in
the Cartesian frame of the radar (Doviak and Zrnic use sectors in range,
azimuth and elevation, Fig. 9.7); the centres are the nodes of a grid
(``x``, ``y``, ``z``); the minimum number of gates and the default sizes are
not values of the papers. The beam geometry is the effective Earth radius model
(:func:`radarx.fundamentals.geometry.beam_center_height`); the elevation angle of
the beam at a gate is the elevation of the sweep plus the angle that the ground
distance subtends at the effective Earth radius (Eq. 9.9 of [2]_ with the
effective radius). The wind of the volume is the wind of the gates in the
cylinder: gates are not weighted.

Checks
------
Eq. 9.18 of [2]_ with the analysis volume of its section 9.3.4 (a sector 30 degrees
wide and 30 km deep with 0.4 and 0.8 degree sweeps, 1 degree and 450 m spacing,
1 m s-1 error of the radial velocity) gives a standard error of the divergence
of :math:`1.6\times10^{-5}` s-1; the notebook repeats this with the
predictors of this module. A wind that is exactly linear is returned exactly
(tests), also with noise to the standard errors predicted by Eq. 9.18.

Compiled kernel
---------------
The fits run in a multithreaded C++ kernel (``_vvp.cpp``: the gates are sorted
into a grid once and the analysis volumes are handed out to the threads in
blocks through an atomic counter). If it is not available a NumPy
implementation with the same steps gives the same results, more slowly.

References
----------
.. [1] Waldteufel, P., and H. Corbin, 1979: On the analysis of single-Doppler
   radar data. *J. Appl. Meteor.*, **18** (4), 532-542,
   https://doi.org/10.1175/1520-0450(1979)018<0532:OTAOSD>2.0.CO;2

.. [2] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
   Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no DOI).

.. autosummary::
   :nosignatures:
   :toctree: generated/

   vvp_winds
"""

from __future__ import annotations

__all__ = ["vvp_winds"]

import numpy as np
import xarray as xr

from .._compat import renamed_arguments
from .._dtypes import fields_as_float32
from .._fields import VELOCITY, as_sweep, find_field, node_dataset
from .._provenance import provenance
from .._registry import accessor_method
from .._sweepwise import dask_support
from ..fundamentals.constants import EARTH_RADIUS
from ..fundamentals.geometry import beam_center_height, ground_range
from ..grid.cone import _sweep_names
from ._fieldtools import check_object, choose_engine, ray_dimension

try:
    from . import _vvp

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _vvp = None
    HAS_COMPILED_KERNEL = False

_DEG = np.pi / 180.0
_RANGE = "range"
_PIVOT_TOLERANCE = 1.0e-10

# the unknowns in the order of the kernel (Eq. 9.12 of Doviak and Zrnic)
_UNKNOWNS = (
    "u",
    "du_dx",
    "du_dz",
    "v",
    "dv_dy",
    "dv_dz",
    "shear",
    "w",
    "dw_dx",
    "dw_dy",
    "dw_dz",
)
_MODELS = {
    "uniform": (0, 3),
    "linear": (0, 1, 2, 3, 4, 5, 6),
    "linear_w": (0, 1, 2, 3, 4, 5, 6, 7, 10),
    "full": tuple(range(11)),
}
_LATERAL = (1, 4, 6, 8, 9)
_VERTICAL = (2, 5, 10)

_ATTRS = {
    "u": ("eastward_wind", "Eastward wind (VVP, modified component u0')", "m s-1"),
    "v": ("northward_wind", "Northward wind (VVP, modified component v0')", "m s-1"),
    "w": (
        "upward_air_velocity",
        "Vertical velocity of the hydrometeors (VVP)",
        "m s-1",
    ),
    "du_dx": (None, "Eastward gradient of the eastward wind du/dx (VVP)", "s-1"),
    "dv_dy": (None, "Northward gradient of the northward wind dv/dy (VVP)", "s-1"),
    "shear": (None, "Shearing deformation du/dy + dv/dx (VVP)", "s-1"),
    "du_dz": (None, "Vertical shear of the eastward wind du/dz (VVP)", "s-1"),
    "dv_dz": (None, "Vertical shear of the northward wind dv/dz (VVP)", "s-1"),
    "dw_dx": (None, "Eastward gradient of the vertical velocity dw/dx (VVP)", "s-1"),
    "dw_dy": (None, "Northward gradient of the vertical velocity dw/dy (VVP)", "s-1"),
    "dw_dz": (None, "Vertical gradient of the vertical velocity dw/dz (VVP)", "s-1"),
}


# --------------------------------------------------------------------------
# gates of a volume
# --------------------------------------------------------------------------


def _scalar_of(ds, name):
    if name in ds.variables:
        value = np.asarray(ds[name].values, dtype=np.float64).ravel()
        value = value[np.isfinite(value)]
        if value.size:
            return float(value[0])
    return None


def _elevations(ds, ray_dim, n):
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


def _sweep_gates(ds, field, fall_speed, k_e, max_range, elevation_limits):
    """Gate arrays of one sweep (valid gates only) or None."""
    try:
        name = find_field(ds, VELOCITY, field)
    except KeyError:
        return None
    if name not in ds:
        return None
    vel = ds[name]
    ray_dim = ray_dimension(vel)
    if vel.ndim != 2:
        raise ValueError(
            f"the velocity needs the two dimensions (ray, range), not {vel.dims}"
        )
    vel = vel.transpose(ray_dim, _RANGE)
    if "azimuth" not in ds.variables or ds["azimuth"].dims != (ray_dim,):
        raise ValueError("the sweep needs an 'azimuth' with one value per ray")
    az = np.asarray(ds["azimuth"].values, dtype=np.float64)
    rng = np.asarray(vel[_RANGE].values, dtype=np.float64)
    el = _elevations(ds, ray_dim, az.size)
    lo, hi = elevation_limits
    if not (lo <= float(np.nanmedian(el)) <= hi):
        return None
    altitude = _scalar_of(ds, "altitude") or 0.0
    reff = k_e * EARTH_RADIUS
    s = ground_range(rng[None, :], el[:, None], reff=reff, radar_height=altitude)
    z = beam_center_height(rng[None, :], el[:, None], radar_height=altitude, reff=reff)
    theta = el[:, None] * _DEG + s / reff
    sa, ca = np.sin(az * _DEG)[:, None], np.cos(az * _DEG)[:, None]
    shape = vel.shape
    vr = np.asarray(vel.values, dtype=np.float64)
    if isinstance(fall_speed, str):
        if fall_speed not in ds:
            raise KeyError(f"the fall speed field {fall_speed!r} is not in the sweep")
        vt = np.asarray(ds[fall_speed].transpose(ray_dim, _RANGE).values, np.float64)
    elif fall_speed is None:
        vt = 0.0
    else:
        vt = float(fall_speed)
    vr = vr + vt * np.sin(theta)
    ok = np.isfinite(vr) & np.isfinite(s) & np.isfinite(z)
    if max_range is not None:
        ok &= np.broadcast_to(rng[None, :] <= max_range, shape)
    gate = {
        "x": np.broadcast_to(s * sa, shape)[ok],
        "y": np.broadcast_to(s * ca, shape)[ok],
        "z": np.broadcast_to(z, shape)[ok],
        "vr": vr[ok],
        "c": np.broadcast_to(np.cos(theta), shape)[ok],
        "sn": np.broadcast_to(np.sin(theta), shape)[ok],
        "sa": np.broadcast_to(sa, shape)[ok],
        "ca": np.broadcast_to(ca, shape)[ok],
    }
    site = {k: _scalar_of(ds, k) for k in ("latitude", "longitude", "altitude")}
    return gate, site


def _volume_gates(obj, field, fall_speed, k_e, max_range, elevation_limits):
    obj = as_sweep(obj)
    if isinstance(obj, xr.DataTree):
        sweeps = [node_dataset(obj[n]) for n in _sweep_names(obj)]
    else:
        sweeps = [obj]
    found = [
        g
        for g in (
            _sweep_gates(ds, field, fall_speed, k_e, max_range, elevation_limits)
            for ds in sweeps
        )
        if g is not None
    ]
    if not found:
        raise KeyError(
            "no sweep has a radial velocity field in the elevation range"
            if field is None
            else f"no sweep has the field {field!r} in the elevation range"
        )
    gates = {k: np.concatenate([g[0][k] for g in found]) for k in found[0][0]}
    if gates["vr"].size == 0:
        raise ValueError("the volume has no valid radial velocity")
    return gates, found[0][1]


# --------------------------------------------------------------------------
# NumPy reference of the kernel
# --------------------------------------------------------------------------


def _predictors(g, x0, y0, z0, radius, half_depth):
    """The eleven predictors of Eq. 9.13 with the offsets scaled by L and D."""
    ex = (g["x"] - x0) / radius
    ey = (g["y"] - y0) / radius
    ez = (g["z"] - z0) / half_depth
    c, sn, sa, ca = g["c"], g["sn"], g["sa"], g["ca"]
    return np.stack(
        [
            c * sa,
            c * sa * ex,
            c * sa * ez,
            c * ca,
            c * ca * ey,
            c * ca * ez,
            0.5 * c * (ca * ex + sa * ey),
            sn,
            sn * ex,
            sn * ey,
            sn * ez,
        ],
        axis=1,
    )


def _fit_numpy(gates, x0, y0, z0, radius, half_depth, mask, min_gates):
    nv = x0.size
    coef = np.full((nv, 11), np.nan)
    err = np.full((nv, 11), np.nan)
    extra = np.full((nv, 3), np.nan)
    idx = [k for k in range(11) if mask & (1 << k)]
    m = len(idx)
    order = np.argsort(gates["x"], kind="stable")
    g = {k: v[order] for k, v in gates.items()}
    gx = g["x"]
    scale = np.ones(11)
    scale[list(_LATERAL)] = 1.0 / radius
    scale[list(_VERTICAL)] = 1.0 / half_depth
    for t in range(nv):
        lo = np.searchsorted(gx, x0[t] - radius, side="left")
        hi = np.searchsorted(gx, x0[t] + radius, side="right")
        sel = {k: v[lo:hi] for k, v in g.items()}
        dx, dy, dz = sel["x"] - x0[t], sel["y"] - y0[t], sel["z"] - z0[t]
        inside = (dx * dx + dy * dy <= radius * radius) & (np.abs(dz) <= half_depth)
        n = int(inside.sum())
        if n < min_gates or n <= m:
            continue
        sel = {k: v[inside] for k, v in sel.items()}
        p = _predictors(sel, x0[t], y0[t], z0[t], radius, half_depth)[:, idx]
        v = sel["vr"]
        normal, rhs = p.T @ p, p.T @ v
        try:
            chol = np.linalg.cholesky(normal)
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.diag(chol) ** 2 > _PIVOT_TOLERANCE * np.diag(normal)):
            continue
        k_hat = np.linalg.solve(normal, rhs)
        inv = np.linalg.inv(normal)
        rss = max(float(v @ v - k_hat @ rhs), 0.0)
        s2 = rss / (n - m)
        coef[t, idx] = k_hat * scale[idx]
        err[t, idx] = np.sqrt(s2 * np.diag(inv)) * scale[idx]
        if mask & 2 and mask & 16:
            a, b = idx.index(1), idx.index(4)
            var = inv[a, a] + inv[b, b] + 2.0 * inv[a, b]
            extra[t, 0] = np.sqrt(s2 * max(var, 0.0)) / radius
        extra[t, 1] = np.sqrt(rss / n)
        extra[t, 2] = n
    return coef, err, extra


def _fit(gates, x0, y0, z0, radius, half_depth, mask, min_gates, compiled, n_threads):
    if compiled:
        arrays = [
            np.ascontiguousarray(gates[k], dtype=np.float64)
            for k in ("x", "y", "z", "vr", "c", "sn", "sa", "ca")
        ]
        return _vvp.vvp_fit(
            *arrays,
            x0,
            y0,
            z0,
            float(radius),
            float(half_depth),
            int(mask),
            int(min_gates),
            int(n_threads or 0),
        )
    return _fit_numpy(gates, x0, y0, z0, radius, half_depth, mask, min_gates)


# --------------------------------------------------------------------------
# the product
# --------------------------------------------------------------------------


def _axis(values, name, lo, hi, step):
    if values is not None:
        a = np.atleast_1d(np.asarray(values, dtype=np.float64))
        if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)):
            raise ValueError(f"{name} must be a non-empty 1-D array of finite values")
        return a
    first = np.floor(lo / step) * step
    return np.arange(first, hi + step, step)


def _levels(values, lo, hi, depth):
    if values is not None:
        return _axis(values, "z", lo, hi, depth)
    first = np.floor(lo / depth) * depth
    edges = np.arange(first, hi + depth, depth)
    return edges[:-1] + 0.5 * depth


def _attrs(name):
    standard, long_name, units = _ATTRS[name]
    out = {"long_name": long_name, "units": units}
    if standard:
        out["standard_name"] = standard
    return out


@dask_support("computes", "it fits the wind to all gates of the volume.")
@renamed_arguments(velocity="field")
@provenance(
    "Volume velocity processing (VVP) of Waldteufel and Corbin (1979) in the form of "
    "Doviak and Zrnic (1993), cylindrical analysis volumes",
    extra_refs=("doviak-zrnic-1993",),
)
def vvp_winds(
    obj,
    field=None,
    *,
    model="linear",
    radius=10000.0,
    depth=1000.0,
    x=None,
    y=None,
    z=None,
    min_gates=100,
    fall_speed=None,
    vorticity=None,
    min_elevation=0.0,
    max_elevation=90.0,
    max_range=None,
    k_e=4.0 / 3.0,
    n_threads=None,
    engine="auto",
):
    """
    Linear wind in analysis volumes from the radial velocities of one radar.

    Fits the linear wind model of volume velocity processing (equations in
    :mod:`radarx.retrieve.vvp`) to the gates inside a cylinder around every
    node of a grid, and returns the wind, its gradients, the divergence and the
    deformation with their standard errors.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataTree
        A sweep or a volume with ``sweep_*`` groups (xradar layout) with the
        radial velocity, ``azimuth``, ``range``, ``elevation`` (or
        ``sweep_fixed_angle``) and the radar ``altitude``. The velocities must
        be dealiased (e.g. :func:`radarx.retrieve.dealias_velocity`) and
        no-data codes masked. Sweeps without the velocity field are skipped.
    field : str, optional
        Velocity variable. Default: the first of ``VRADH``, ``VRADV``,
        ``VELOCITY``, ``velocity``, ``VEL``.
    model : {"uniform", "linear", "linear_w", "full"}, optional
        Unknowns of the fit, see :mod:`radarx.retrieve.vvp`. Default
        ``"linear"``: the horizontal wind, its horizontal gradients and its
        vertical shear. Terms that the model omits are not in the result.
    radius : float, optional
        Radius of the cylinder of an analysis volume (m). Default 10000.
    depth : float, optional
        Depth of the cylinder (m); the volume reaches ``depth / 2`` above and
        below the centre. Default 1000.
    x, y : array-like, optional
        Centres of the volumes east and north of the radar (m). Default: a
        grid with the spacing ``radius`` that covers the gates.
    z : array-like, optional
        Heights of the centres above sea level (m). Default: the middles of
        layers of the thickness ``depth`` that cover the gates.
    min_gates : int, optional
        Fewest gates in a volume. Default 100. A volume with fewer gates, or
        whose gates do not determine the unknowns (for instance the vertical
        terms of a single sweep), is NaN.
    fall_speed : float or str, optional
        Terminal fall speed of the hydrometeors (m s-1, positive downward) or
        the name of a field of it, removed from the radial velocity as
        :math:`v_r + v_t \\sin\\theta`. Default None: not removed, ``w`` then
        includes the fall speed.
    vorticity : float, optional
        Vertical vorticity (s-1) of the flow, used to turn the modified wind
        components into the wind at the centre of a volume,
        :math:`u = u_0' - y_0\\zeta/2`, :math:`v = v_0' + x_0\\zeta/2`.
        Default None: ``u`` and ``v`` are the modified components.
    min_elevation, max_elevation : float, optional
        Elevation angles (degrees) of the sweeps used. Default 0 to 90.
    max_range : float, optional
        Use only gates up to this slant range (m).
    k_e : float, optional
        Effective Earth radius factor of the beam geometry. Default 4/3.
    n_threads : int, optional
        Threads of the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation. ``"auto"`` (default) prefers the compiled kernel.

    Returns
    -------
    xarray.Dataset
        On the grid ``(z, y, x)`` (coordinates in m; ``z`` above sea level):
        ``u`` and ``v`` (m s-1), and for the models that have them ``w``,
        ``du_dx``, ``dv_dy``, ``shear`` (:math:`u_y + v_x`), ``du_dz``,
        ``dv_dz``, ``dw_dx``, ``dw_dy`` and ``dw_dz`` (s-1), each with its
        standard error as ``<name>_error``; ``divergence`` (:math:`u_x + v_y`),
        ``divergence_error``, ``stretching_deformation`` (:math:`u_x - v_y`),
        ``shearing_deformation`` and ``deformation`` (their root sum of
        squares, s-1) when the model has the horizontal gradients;
        ``vvp_rms`` (the r.m.s. residual, m s-1) and ``vvp_gates`` (gates in
        the volume). The vorticity is not retrievable.

    Raises
    ------
    ValueError
        For invalid options, a sweep without the geometry, or a volume without
        a valid velocity.
    KeyError
        If no sweep has the velocity field.
    TypeError
        If ``obj`` is not an xarray Dataset or DataTree.
    ImportError
        If ``engine="compiled"`` and the kernel is not available.

    Notes
    -----
    The retrieval is biased where the wind is not linear in the volume or the
    model leaves out a term that matters; the standard errors describe the
    random error of the radial velocity only. Use a larger volume to reduce the
    random error and a smaller one to reduce the bias.

    References
    ----------
    .. [1] Waldteufel, P., and H. Corbin, 1979: On the analysis of
           single-Doppler radar data. *J. Appl. Meteor.*, **18** (4), 532-542,
           https://doi.org/10.1175/1520-0450(1979)018<0532:OTAOSD>2.0.CO;2
    .. [2] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    """
    check_object(obj)
    if isinstance(obj, xr.DataArray):
        raise TypeError("vvp_winds needs a Dataset sweep or a DataTree volume")
    compiled = choose_engine(engine, HAS_COMPILED_KERNEL, "vvp")
    if model not in _MODELS:
        raise ValueError(f"model must be one of {sorted(_MODELS)}, not {model!r}")
    for label, value in (("radius", radius), ("depth", depth), ("k_e", k_e)):
        if not (np.isfinite(value) and value > 0):
            raise ValueError(f"{label} must be positive, not {value!r}")
    if int(min_gates) < 1:
        raise ValueError("min_gates must be at least 1")
    if not min_elevation <= max_elevation:
        raise ValueError("min_elevation must not exceed max_elevation")
    gates, site = _volume_gates(
        obj, field, fall_speed, k_e, max_range, (min_elevation, max_elevation)
    )
    xs = _axis(
        x,
        "x",
        float(gates["x"].min()),
        float(gates["x"].max()),
        radius,
    )
    ys = _axis(
        y,
        "y",
        float(gates["y"].min()),
        float(gates["y"].max()),
        radius,
    )
    zs = _levels(z, float(gates["z"].min()), float(gates["z"].max()), depth)
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    unknowns = _MODELS[model]
    mask = sum(1 << k for k in unknowns)
    coef, err, extra = _fit(
        gates,
        np.ascontiguousarray(xx.ravel()),
        np.ascontiguousarray(yy.ravel()),
        np.ascontiguousarray(zz.ravel()),
        radius,
        0.5 * depth,
        mask,
        min_gates,
        compiled,
        n_threads,
    )
    shape = (zs.size, ys.size, xs.size)
    coords = {
        "x": ("x", xs, {"long_name": "East of the radar", "units": "m"}),
        "y": ("y", ys, {"long_name": "North of the radar", "units": "m"}),
        "z": (
            "z",
            zs,
            {
                "standard_name": "altitude",
                "long_name": "Height above sea level",
                "units": "m",
            },
        ),
    }
    for key in ("latitude", "longitude", "altitude"):
        if site.get(key) is not None:
            coords[key] = ((), site[key])
    dims = ("z", "y", "x")
    out = xr.Dataset(coords=coords)

    def put(name, values, attrs):
        out[name] = xr.DataArray(values.reshape(shape), dims=dims).assign_attrs(attrs)

    u, v = coef[:, 0].copy(), coef[:, 3].copy()
    if vorticity is not None:
        zeta = float(vorticity)
        u = u - 0.5 * yy.ravel() * zeta
        v = v + 0.5 * xx.ravel() * zeta
    put("u", u, _attrs("u"))
    put("v", v, _attrs("v"))
    out["u_error"] = (
        out["u"]
        .copy(data=err[:, 0].reshape(shape))
        .assign_attrs(long_name="Standard error of u", units="m s-1")
    )
    out["v_error"] = (
        out["v"]
        .copy(data=err[:, 3].reshape(shape))
        .assign_attrs(long_name="Standard error of v", units="m s-1")
    )
    for k in unknowns:
        name = _UNKNOWNS[k]
        if name in ("u", "v"):
            continue
        put(name, coef[:, k], _attrs(name))
        put(
            f"{name}_error",
            err[:, k],
            {"long_name": f"Standard error of {name}", "units": _ATTRS[name][2]},
        )
    if 1 in unknowns and 4 in unknowns:
        ux, vy, sh = coef[:, 1], coef[:, 4], coef[:, 6]
        stretch = ux - vy
        put(
            "divergence",
            ux + vy,
            {
                "standard_name": "divergence_of_wind",
                "long_name": "Horizontal divergence du/dx + dv/dy (VVP)",
                "units": "s-1",
            },
        )
        put(
            "divergence_error",
            extra[:, 0],
            {"long_name": "Standard error of the divergence", "units": "s-1"},
        )
        put(
            "stretching_deformation",
            stretch,
            {"long_name": "Stretching deformation du/dx - dv/dy (VVP)", "units": "s-1"},
        )
        put(
            "shearing_deformation",
            sh,
            {"long_name": "Shearing deformation du/dy + dv/dx (VVP)", "units": "s-1"},
        )
        put(
            "deformation",
            np.hypot(stretch, sh),
            {"long_name": "Total deformation (VVP)", "units": "s-1"},
        )
    put(
        "vvp_rms",
        extra[:, 1],
        {"long_name": "R.m.s. residual of the VVP fit", "units": "m s-1"},
    )
    out["vvp_gates"] = xr.DataArray(
        np.nan_to_num(extra[:, 2]).reshape(shape).astype(np.int32), dims=dims
    ).assign_attrs(long_name="Number of gates in the analysis volume", units="1")
    out.attrs.update(
        source="VVP (Waldteufel and Corbin 1979; Doviak and Zrnic 1993)",
        vvp_model=model,
        vvp_radius=float(radius),
        vvp_depth=float(depth),
    )
    return fields_as_float32(out)


# --------------------------------------------------------------------------
# accessors
# --------------------------------------------------------------------------


@accessor_method("dataset", "datatree", name="vvp_winds", wraps=vvp_winds)
def _vvp_winds_accessor(self):
    """
    Linear wind in analysis volumes of the sweep or volume (VVP).

    See :func:`radarx.retrieve.vvp_winds` for the parameters.

    Returns
    -------
    xarray.Dataset
        Winds, gradients, divergence and deformation on ``(z, y, x)``.
    """
