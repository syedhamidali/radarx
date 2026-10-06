# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Multi-Doppler Wind Retrieval
============================

Three-dimensional winds ``(u, v, w)`` from the radial velocities of two or
more Doppler radars, by a variational method (Gao et al. 1999). The wind on
a regular Cartesian grid minimises

.. math::

   J = J_o + J_m + J_s + J_b + J_v

- :math:`J_o`, the misfit to the radial velocities of every radar,
  corrected for the fall speed of the precipitation;
- :math:`J_m`, the anelastic mass continuity equation as a weak constraint;
- :math:`J_s`, a smoothness penalty on the second derivatives;
- :math:`J_b`, the distance to a background wind from a sounding or ERA5
  (:mod:`radarx.io.sounding`);
- :math:`J_v`, optionally, the steady vertical vorticity equation in a frame
  moving with the storm (Shapiro et al. 2009; Potvin et al. 2012).

The cost and its exact gradient are computed in one fused, multithreaded
pass of a compiled C++ kernel (with an identical NumPy implementation as a
fallback and test oracle). Without the vorticity term the cost is quadratic
and is minimised by conjugate gradients preconditioned with the inverse of
the smoothness operator in a cosine basis; with it, by L-BFGS-B. Both run
first on coarser grids. The vertical velocity at the lower boundary (and optionally at the top
of the grid or above the echo top) is held at zero by leaving it out of the
control vector.

The input is a gridded :class:`xarray.Dataset` with a ``radar`` dimension:
per-radar radial velocity (``VRADH``) and reflectivity on ``(radar, z, y,
x)`` and the radar positions ``radar_x``, ``radar_y``, ``radar_altitude`` on
``radar``; :func:`multi_doppler_input` builds it from radar volumes with cone
gridding, and :func:`radar_geometry` adds the beam azimuth and elevation of
every radar at every grid cell.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = ["fall_speed", "multi_doppler", "multi_doppler_input", "radar_geometry"]

__doc__ = __doc__.format("\n   ".join(__all__))

import time as _time
import warnings

import numpy as np
import xarray as xr

try:
    from . import _multidoppler

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _multidoppler = None
    HAS_COMPILED_KERNEL = False

EARTH_RADIUS = 6371000.0
OMEGA = 7.292115e-5  # Earth's angular velocity [s-1]
TERMS = ("observation", "mass_continuity", "smoothness", "background", "vorticity")
_VREF = 10.0  # velocity scale of the vorticity residual [m s-1]
_PRECONDITIONER_LAMBDA = 0.05  # observation share of the preconditioner


def _use_compiled(engine):
    """Whether to run the compiled kernel for the requested ``engine``."""
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled multi-Doppler kernel is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


# ---------------------------------------------------------------------------
# discrete operators (NumPy reference) and the cost function
# ---------------------------------------------------------------------------


def _d1(f, axis, d):
    """First derivative: centred inside, one-sided at the edges."""
    return np.gradient(f, d, axis=axis, edge_order=1)


def _d1T(g, axis, d):
    """Adjoint (transpose) of :func:`_d1`."""
    g = np.moveaxis(g, axis, 0)
    out = np.zeros_like(g)
    out[2:] += 0.5 * g[1:-1] / d
    out[:-2] -= 0.5 * g[1:-1] / d
    out[1] += g[0] / d
    out[0] -= g[0] / d
    out[-1] += g[-1] / d
    out[-2] -= g[-1] / d
    return np.moveaxis(out, 0, axis)


def _d2(f, axis):
    """Second difference in grid units, zero at the edges."""
    f = np.moveaxis(f, axis, 0)
    out = np.zeros_like(f)
    out[1:-1] = f[:-2] - 2.0 * f[1:-1] + f[2:]
    return np.moveaxis(out, 0, axis)


def _d2T(g, axis):
    """Adjoint (transpose) of :func:`_d2`."""
    g = np.moveaxis(g, axis, 0)
    out = np.zeros_like(g)
    gi = g[1:-1]
    out[:-2] += gi
    out[1:-1] -= 2.0 * gi
    out[2:] += gi
    return np.moveaxis(out, 0, axis)


def _cost_gradient_numpy(
    state,
    coef,
    target,
    weight,
    rho,
    bg,
    bg_weight,
    vort_weight,
    dx,
    dy,
    dz,
    cm,
    csx,
    csy,
    csz,
    cv,
    ut,
    vt,
    coriolis,
    h,
    vort_scale,
    n_threads=0,
):
    """NumPy reference of ``_multidoppler.cost_gradient`` (same arguments)."""
    u, v, w = state
    nr = target.shape[0]
    coef = coef.reshape((nr, 3) + u.shape)
    terms = np.zeros(5)
    grad = np.zeros_like(state)

    r = coef[:, 0] * u + coef[:, 1] * v + coef[:, 2] * w - target
    terms[0] = np.sum(weight * r * r)
    t = 2.0 * weight * r
    for q in range(3):
        grad[q] += np.sum(t * coef[:, q], axis=0)

    rb = np.where(bg_weight != 0, state - bg, 0.0)
    terms[3] = np.sum(bg_weight * rb * rb)
    grad += 2.0 * bg_weight * rb

    for q in range(3):
        for axis, c in ((2, csx), (1, csy), (0, csz)):
            s = _d2(state[q], axis)
            terms[2] += c * np.sum(s * s)
            grad[q] += 2.0 * c * _d2T(s, axis)

    m = h / rho * (_d1(rho * u, 2, dx) + _d1(rho * v, 1, dy) + _d1(rho * w, 0, dz))
    terms[1] = cm * np.sum(m * m)
    a = 2.0 * cm * h * m / rho
    grad[0] += rho * _d1T(a, 2, dx)
    grad[1] += rho * _d1T(a, 1, dy)
    grad[2] += rho * _d1T(a, 0, dz)

    if cv != 0.0:
        zeta = _d1(v, 2, dx) - _d1(u, 1, dy)
        zx, zy, zz = _d1(zeta, 2, dx), _d1(zeta, 1, dy), _d1(zeta, 0, dz)
        div = _d1(u, 2, dx) + _d1(v, 1, dy)
        uz, vz = _d1(u, 0, dz), _d1(v, 0, dz)
        wx, wy = _d1(w, 2, dx), _d1(w, 1, dy)
        res = (
            (u - ut) * zx
            + (v - vt) * zy
            + w * zz
            + (zeta + coriolis) * div
            + wx * vz
            - wy * uz
        )
        wv = cv * vort_weight
        terms[4] = np.sum(wv * vort_scale**2 * res * res)
        g = 2.0 * wv * vort_scale**2 * res
        gz = (
            _d1T(g * (u - ut), 2, dx)
            + _d1T(g * (v - vt), 1, dy)
            + _d1T(g * w, 0, dz)
            + g * div
        )
        bz = g * (zeta + coriolis)
        grad[0] += g * zx - _d1T(gz, 1, dy) + _d1T(bz, 2, dx) - _d1T(g * wy, 0, dz)
        grad[1] += g * zy + _d1T(gz, 2, dx) + _d1T(bz, 1, dy) + _d1T(g * wx, 0, dz)
        grad[2] += g * zz + _d1T(g * vz, 2, dx) - _d1T(g * uz, 1, dy)
    return terms, grad


def _cost_gradient(problem, state, use_compiled, n_threads):
    """Cost terms and gradient of a prepared problem at ``state``."""
    p = problem
    args = (
        state,
        p["coef"],
        p["target"],
        p["weight"],
        p["rho"],
        p["bg"],
        p["bg_weight"],
        p["vort_weight"],
        p["dx"],
        p["dy"],
        p["dz"],
        p["cm"],
        p["csx"],
        p["csy"],
        p["csz"],
        p["cv"],
        p["ut"],
        p["vt"],
        p["coriolis"],
        p["h"],
        p["vort_scale"],
        int(n_threads or 0),
    )
    if use_compiled:
        terms, grad = _multidoppler.cost_gradient(*args)
        return np.asarray(terms), np.asarray(grad)
    return _cost_gradient_numpy(*args)


# ---------------------------------------------------------------------------
# fall speed and beam geometry
# ---------------------------------------------------------------------------


def fall_speed(
    reflectivity,
    air_density=None,
    freezing_level=None,
    *,
    rain=(2.65, 0.114),
    ice=(0.817, 0.063),
    reference_density=1.2,
):
    """
    Terminal fall speed of precipitation from reflectivity.

    :math:`V_t = a Z^b (\\rho_0 / \\rho)^{0.4}` with :math:`Z` in
    mm\\ :sup:`6` m\\ :sup:`-3`. Below the freezing level the rain relation
    :math:`V_t = 2.65 Z^{0.114}` of Atlas et al. (1973) [1]_ is used, above it
    the snow relation :math:`V_t = 0.817 Z^{0.063}` (also Atlas et al.
    1973). The factor :math:`(\\rho_0 / \\rho)^{0.4}` corrects for the lower
    air density aloft (Foote and du Toit 1969 [2]_).

    Parameters
    ----------
    reflectivity : xarray.DataArray
        Reflectivity factor in dBZ on ``(..., z, y, x)``.
    air_density : xarray.DataArray, optional
        Air density in kg m-3, broadcastable to ``reflectivity``. Without it
        no density correction is made.
    freezing_level : float or xarray.DataArray, optional
        Height of the 0 degC level (m above sea level), e.g. ``(y, x)``.
        Without it the rain relation is used everywhere.
    rain, ice : tuple of float, optional
        Coefficients ``(a, b)`` of the rain and ice relations.
    reference_density : float, optional
        Surface reference density :math:`\\rho_0` in kg m-3. Default 1.2.

    Returns
    -------
    xarray.DataArray
        Fall speed in m s-1, positive downward; NaN where the reflectivity is
        missing.

    References
    ----------
    .. [1] Atlas, D., R. C. Srivastava, and R. S. Sekhon, 1973: Doppler radar
       characteristics of precipitation at vertical incidence. *Rev.
       Geophys.*, **11**, 1-35, https://doi.org/10.1029/RG011i001p00001
    .. [2] Foote, G. B., and P. S. du Toit, 1969: Terminal velocity of
       raindrops aloft. *J. Appl. Meteor.*, **8**, 249-253,
       https://doi.org/10.1175/1520-0450(1969)008<0249:TVORA>2.0.CO;2
    """
    dbz = reflectivity
    z_lin = 10.0 ** (dbz.clip(max=70.0) / 10.0)
    vt = rain[0] * z_lin ** rain[1]
    if freezing_level is not None:
        vt_ice = ice[0] * z_lin ** ice[1]
        vt = xr.where(dbz["z"] > freezing_level, vt_ice, vt)
    if air_density is not None:
        vt = vt * (reference_density / air_density) ** 0.4
    vt = vt.where(np.isfinite(dbz))
    vt.attrs = {
        "long_name": "terminal fall speed of precipitation (positive downward)",
        "units": "m s-1",
    }
    return vt.rename("fall_speed")


def _beam_angles(dx, dy, z, radar_altitude, earth_radius=EARTH_RADIUS):
    """Azimuth and local beam elevation (deg) seen from a radar (4/3 Earth)."""
    R = earth_radius * 4.0 / 3.0
    s = np.hypot(dx, dy)
    azimuth = np.degrees(np.arctan2(dx, dy)) % 360.0
    a = R + radar_altitude
    b = R + z
    gamma = s / R
    r = np.sqrt(np.maximum(a * a + b * b - 2.0 * a * b * np.cos(gamma), 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        sin_el = (b * b - r * r - a * a) / (2.0 * r * a)
    antenna = np.arcsin(np.clip(sin_el, -1.0, 1.0))
    elevation = np.degrees(antenna + gamma)  # beam angle above the local horizon
    elevation = np.where(r > 0, elevation, np.nan)
    return azimuth, elevation


def radar_geometry(grid, *, earth_radius=EARTH_RADIUS):
    """
    Beam azimuth and elevation of every radar at every grid cell.

    The beam is a straight line on an Earth of 4/3 its radius. The
    elevation is the local angle of the beam above the horizon at the grid
    cell (the antenna elevation plus the angle the beam has travelled around
    the Earth), so that the radial velocity is :math:`u \\cos\\phi \\sin\\alpha
    + v \\cos\\phi \\cos\\alpha + w \\sin\\phi` with ``u``, ``v`` along the grid
    axes.

    Parameters
    ----------
    grid : xarray.Dataset
        Grid with ``z`` (m above sea level), ``y``, ``x`` (m) and the radar
        positions ``radar_x``, ``radar_y`` (m, in the grid's coordinates) and
        ``radar_altitude`` (m above sea level) on the ``radar`` dimension.
    earth_radius : float, optional
        Earth radius in m.

    Returns
    -------
    xarray.Dataset
        ``grid`` with ``azimuth`` and ``elevation`` on ``(radar, z, y, x)``
        (degrees; azimuth clockwise from the grid ``y`` axis).
    """
    for name in ("radar_x", "radar_y", "radar_altitude"):
        if name not in grid:
            raise ValueError(f"grid needs {name!r} on the 'radar' dimension")
    x = grid["x"].values[None, None, None, :]
    y = grid["y"].values[None, None, :, None]
    z = grid["z"].values[None, :, None, None]
    rx_ = grid["radar_x"].values[:, None, None, None]
    ry_ = grid["radar_y"].values[:, None, None, None]
    rz_ = grid["radar_altitude"].values[:, None, None, None]
    shape = (grid.sizes["radar"], grid.sizes["z"], grid.sizes["y"], grid.sizes["x"])
    az, el = _beam_angles(
        np.broadcast_to(x - rx_, shape),
        np.broadcast_to(y - ry_, shape),
        np.broadcast_to(z, shape),
        np.broadcast_to(rz_, shape),
        earth_radius,
    )
    dims = ("radar", "z", "y", "x")
    return grid.assign(
        azimuth=(
            dims,
            az.astype(np.float32),
            {
                "long_name": "beam azimuth, clockwise from the grid y axis",
                "units": "degree",
            },
        ),
        elevation=(
            dims,
            el.astype(np.float32),
            {"long_name": "beam elevation above the local horizon", "units": "degree"},
        ),
    )


# ---------------------------------------------------------------------------
# input assembly
# ---------------------------------------------------------------------------


def _origin_crs(latitude, longitude):
    from ..grid.cone import _crs_wkt

    return _crs_wkt(latitude, longitude)


def _site(dtree):
    from ..grid.cone import _load_volume

    _, site, _ = _load_volume(dtree)
    return site


def multi_doppler_input(
    volumes,
    x,
    y,
    z,
    *,
    origin=None,
    velocity="VRADH",
    reflectivity="DBZH",
    time=None,
    motion=None,
    fill_below=False,
    n_threads=None,
    engine="auto",
):
    """
    Grid several radar volumes onto one grid for :func:`multi_doppler`.

    Every volume is cone-gridded (:func:`radarx.grid.grid_cones`) onto the
    same grid, centred on ``origin`` (an azimuthal equidistant projection);
    the grids are stacked on a ``radar`` dimension and the beam geometry is
    added with :func:`radar_geometry`. With ``time`` and ``motion`` every
    radar's fields are first moved to the common analysis time
    (:func:`radarx.retrieve.advect`).

    Radial velocities must be dealiased first
    (:func:`radarx.retrieve.dealias_velocity`).

    Parameters
    ----------
    volumes : sequence of xarray.DataTree
        Radar volumes with ``sweep_*`` groups and the site ``latitude``,
        ``longitude`` and ``altitude``.
    x, y : array-like
        Grid coordinates (m) east and north of ``origin``.
    z : array-like
        Grid heights above sea level (m).
    origin : tuple of float, optional
        ``(latitude, longitude)`` of the grid centre. Default: the first
        radar.
    velocity, reflectivity : str, optional
        Field names. Default ``"VRADH"`` and ``"DBZH"``. A missing
        reflectivity field is skipped.
    time : datetime-like, optional
        Analysis time to advect every radar to (requires ``motion``).
    motion : xarray.Dataset or tuple, optional
        Storm motion: the result of :func:`radarx.retrieve.estimate_motion`
        or ``(u, v)`` in m s-1.
    fill_below : bool, optional
        Fill levels below the lowest sweep. Default False.
    n_threads : int, optional
        Threads for the compiled kernels. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Gridding implementation.

    Returns
    -------
    xarray.Dataset
        ``velocity`` and ``reflectivity`` on ``(radar, z, y, x)``, ``azimuth``
        and ``elevation`` (see :func:`radar_geometry`), and on ``radar``:
        ``radar_x``, ``radar_y``, ``radar_altitude``, ``radar_latitude``,
        ``radar_longitude``, ``radar_name`` and ``time`` (the time of each
        radar's data); plus ``lat``, ``lon`` and ``crs_wkt`` of the grid.
    """
    import pyproj

    from ..grid.cone import _lonlat_axes, grid_cones
    from .advection import advect

    if (time is None) != (motion is None):
        raise ValueError("give both time and motion, or neither")
    x, y, z = (np.asarray(a, dtype=np.float64) for a in (x, y, z))
    sites = [_site(v) for v in volumes]
    if origin is None:
        origin = (sites[0]["latitude"], sites[0]["longitude"])
    lat0, lon0 = map(float, origin)
    aeqd = pyproj.Proj(proj="aeqd", lat_0=lat0, lon_0=lon0, datum="WGS84")
    grids, rx_, ry_ = [], [], []
    for dtree, site in zip(volumes, sites):
        xs, ys = aeqd(site["longitude"], site["latitude"])
        fields = [velocity]
        sweeps = [n for n in dtree.children if n.startswith("sweep")]
        if any(reflectivity in dtree[n].data_vars for n in sweeps):
            fields.append(reflectivity)
        g = grid_cones(
            dtree,
            fields,
            x - xs,
            y - ys,
            z,
            fill_below=fill_below,
            n_threads=n_threads,
            engine=engine,
        )
        g = g.drop_vars(["lat", "lon", "crs_wkt"], errors="ignore")
        g = g.assign_coords(x=x, y=y)
        if time is not None:
            m = motion if isinstance(motion, xr.Dataset) else None
            if m is None:
                g = advect(g, float(motion[0]), float(motion[1]), time=time)
            else:
                g = advect(g, m, time=time)
        grids.append(g)
        rx_.append(xs)
        ry_.append(ys)
    names = [
        str(v.attrs.get("instrument_name", f"radar_{i}")) for i, v in enumerate(volumes)
    ]
    times = [
        g["time"].values if "time" in g else np.datetime64("NaT", "ns") for g in grids
    ]
    stacked = xr.concat(
        [
            g.drop_vars(["time", "latitude", "longitude", "altitude"], errors="ignore")
            for g in grids
        ],
        dim="radar",
        join="outer",
        combine_attrs="drop",
    )
    lon, lat = _lonlat_axes(x, y, lat0, lon0)
    stacked = stacked.assign_coords(
        radar=np.arange(len(volumes)),
        radar_name=("radar", names),
        time=("radar", np.asarray(times, dtype="datetime64[ns]")),
        lat=("y", lat, {"units": "degrees_north"}),
        lon=("x", lon, {"units": "degrees_east"}),
        crs_wkt=_origin_crs(lat0, lon0),
    )
    stacked["x"].attrs = {"units": "m", "long_name": "distance east of the grid origin"}
    stacked["y"].attrs = {
        "units": "m",
        "long_name": "distance north of the grid origin",
    }
    stacked["z"].attrs = {"units": "m", "long_name": "height above sea level"}
    stacked = stacked.assign(
        radar_x=("radar", np.asarray(rx_), {"units": "m", "long_name": "radar x"}),
        radar_y=("radar", np.asarray(ry_), {"units": "m", "long_name": "radar y"}),
        radar_altitude=(
            "radar",
            np.asarray([s.get("altitude", 0.0) for s in sites]),
            {"units": "m", "long_name": "radar altitude above sea level"},
        ),
        radar_latitude=(
            "radar",
            np.asarray([s["latitude"] for s in sites]),
            {"units": "degrees_north"},
        ),
        radar_longitude=(
            "radar",
            np.asarray([s["longitude"] for s in sites]),
            {"units": "degrees_east"},
        ),
    )
    stacked.attrs["origin_latitude"] = lat0
    stacked.attrs["origin_longitude"] = lon0
    return radar_geometry(stacked)


def _as_multi_radar(grids, velocity):
    """Dataset with a ``radar`` dimension (and beam geometry) from the input."""
    if isinstance(grids, xr.Dataset):
        ds = grids
    else:
        ds = xr.concat(list(grids), dim="radar", combine_attrs="drop_conflicts")
    if "radar" not in ds.dims:
        raise ValueError("the grids need a 'radar' dimension (one entry per radar)")
    if velocity not in ds:
        raise ValueError(f"no {velocity!r} in the grids")
    if "azimuth" not in ds or "elevation" not in ds:
        ds = radar_geometry(ds)
    for name in ("x", "y", "z"):
        if ds.sizes.get(name, 0) < 3:
            raise ValueError("the grid needs at least 3 points along x, y and z")
    return ds


# ---------------------------------------------------------------------------
# problem setup
# ---------------------------------------------------------------------------


def _spacing(coord):
    v = np.asarray(coord.values, dtype=np.float64)
    d = np.diff(v)
    if not np.allclose(d, d[0], rtol=1e-6):
        raise ValueError(f"{coord.name} must be evenly spaced")
    return float(d[0])


def _fill_column(a):
    """Fill NaN levels by the nearest valid level, then any rest by the mean."""
    a = np.array(a, dtype=np.float64)
    nz = a.shape[0]
    idx = np.where(np.isfinite(a), np.arange(nz)[:, None, None], -1)
    idx = np.maximum.accumulate(idx, axis=0)
    filled = np.take_along_axis(a, np.maximum(idx, 0), axis=0)
    filled[idx < 0] = np.nan
    rev = np.where(np.isfinite(filled[::-1]), np.arange(nz)[:, None, None], -1)
    rev = np.maximum.accumulate(rev, axis=0)
    back = np.take_along_axis(filled[::-1], np.maximum(rev, 0), axis=0)[::-1]
    filled = np.where(np.isfinite(filled), filled, back)
    if not np.isfinite(filled).all():
        mean = np.nanmean(filled) if np.isfinite(filled).any() else 0.0
        filled = np.where(np.isfinite(filled), filled, mean)
    return filled


def _standard_density(z):
    """Density of a standard atmosphere approximation, 1.2 exp(-z / 10 km)."""
    return 1.2 * np.exp(-np.asarray(z, dtype=np.float64) / 10000.0)


def _w_free(ds, w_boundary, valid_any):
    """Mask of grid cells where w is a control variable."""
    nz, ny, nx = ds.sizes["z"], ds.sizes["y"], ds.sizes["x"]
    free = np.ones((nz, ny, nx), dtype=bool)
    if isinstance(w_boundary, str):
        w_boundary = (w_boundary,)
    w_boundary = tuple(w_boundary or ())
    for item in w_boundary:
        if item not in ("bottom", "top", "echo_top"):
            raise ValueError(
                f"w_boundary items must be bottom, top or echo_top, not {item!r}"
            )
    if "bottom" in w_boundary:
        free[0] = False
    if "top" in w_boundary:
        free[-1] = False
    if "echo_top" in w_boundary:
        lev = np.arange(nz)[:, None, None]
        top = np.where(valid_any, lev, -1).max(axis=0)  # highest echo level per column
        free &= lev <= top[None] + 1  # w = 0 from one level above the echo top
    return free


def _prepare(
    ds,
    background,
    *,
    velocity,
    reflectivity,
    use_fall_speed,
    weights,
    storm_motion,
    w_boundary,
):
    """NumPy arrays of the variational problem on the grid of ``ds``."""
    nr = ds.sizes["radar"]
    dims = ("radar", "z", "y", "x")
    vr = ds[velocity].transpose(*dims).values.astype(np.float64)
    el = np.radians(ds["elevation"].transpose(*dims).values.astype(np.float64))
    az = np.radians(ds["azimuth"].transpose(*dims).values.astype(np.float64))
    coef = np.stack(
        [np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)], axis=1
    )  # (nr, 3, nz, ny, nx)
    zc = ds["z"]

    # density, background and freezing level
    shape3 = (ds.sizes["z"], ds.sizes["y"], ds.sizes["x"])
    rho = None
    bg = np.zeros((3,) + shape3)
    bg_ok = np.zeros((3,) + shape3, dtype=bool)
    freezing = None
    if background is not None:
        background = background.transpose(..., "z", "y", "x")
        if "air_density" in background:
            rho = _fill_column(
                np.broadcast_to(background["air_density"].values, shape3)
            )
        for q, name in enumerate(("u", "v", "w")):
            if name in background:
                b = np.broadcast_to(background[name].values, shape3).astype(np.float64)
                bg_ok[q] = np.isfinite(b)
                bg[q] = _fill_column(b)
        if "freezing_level" in background:
            freezing = background["freezing_level"]
    if rho is None:
        rho = np.broadcast_to(
            _standard_density(zc.values)[:, None, None], shape3
        ).copy()

    vt = np.zeros((nr,) + shape3)
    if use_fall_speed:
        if reflectivity in ds:
            dbz_mean = ds[reflectivity]
            if "radar" in dbz_mean.dims:
                # reflectivity per cell: mean (in dBZ) over the radars that see it
                dbz_mean = dbz_mean.mean("radar", skipna=True)
            dbz_mean = dbz_mean.transpose("z", "y", "x")
            vt_da = fall_speed(
                dbz_mean,
                xr.DataArray(rho, dims=("z", "y", "x"), coords={"z": zc}),
                freezing,
            )
            vt = np.broadcast_to(np.nan_to_num(vt_da.values), (nr,) + shape3)
        else:
            warnings.warn(
                f"no {reflectivity!r} in the grids: no fall speed correction",
                stacklevel=3,
            )
    target = vr + coef[:, 2] * vt  # a u + b v + c w = vr + c Vt
    valid = np.isfinite(vr) & np.isfinite(coef).all(axis=1)
    weight = np.where(valid, weights["observation"], 0.0)
    target = np.where(valid, target, 0.0)
    coef = np.where(valid[:, None], coef, 0.0)

    bw = np.zeros((3,) + shape3)
    bw[0] = np.where(bg_ok[0], weights["background"], 0.0)
    bw[1] = np.where(bg_ok[1], weights["background"], 0.0)
    bw[2] = np.where(bg_ok[2], weights.get("background_w", 0.0), 0.0)
    first_guess = bg.copy()
    first_guess[2] = 0.0
    bg = np.where(bw != 0, bg, 0.0)

    n_obs = valid.sum(axis=0)
    valid_any = n_obs > 0
    if reflectivity in ds:
        echo = np.isfinite(ds[reflectivity].transpose(..., "z", "y", "x").values)
        valid_any = valid_any | (echo.any(0) if echo.ndim == 4 else echo)
    free = _w_free(ds, w_boundary, valid_any)

    dx, dy, dz = _spacing(ds["x"]), _spacing(ds["y"]), _spacing(ds["z"])
    h = float(np.sqrt(abs(dx * dy)))
    lat = ds.attrs.get("origin_latitude")
    if lat is None and "lat" in ds.coords:
        lat = float(np.mean(ds["lat"].values))
    coriolis = 2.0 * OMEGA * np.sin(np.radians(lat)) if lat is not None else 0.0
    ut, vt_storm = storm_motion
    problem = dict(
        coef=np.ascontiguousarray(coef.reshape((nr * 3,) + shape3)),
        target=np.ascontiguousarray(target),
        weight=np.ascontiguousarray(weight),
        rho=np.ascontiguousarray(rho),
        bg=np.ascontiguousarray(bg),
        bg_weight=np.ascontiguousarray(bw),
        vort_weight=np.ascontiguousarray((n_obs >= 2).astype(np.float64)),
        dx=dx,
        dy=dy,
        dz=dz,
        co=float(weights["observation"]),
        cb=float(weights["background"]),
        cm=float(weights["mass_continuity"]),
        csx=float(weights["smoothness"]),
        csy=float(weights["smoothness"]),
        csz=float(weights.get("smoothness_vertical", weights["smoothness"])),
        cv=float(weights["vorticity"]),
        ut=float(ut),
        vt=float(vt_storm),
        coriolis=float(coriolis),
        h=h,
        vort_scale=h * h / _VREF,
        free=free,
        first_guess=first_guess,
        vr=vr,
        vfall=vt,
        n_obs=n_obs,
        shape=shape3,
    )
    return problem


def _coarsen_problem(p, factor=2):
    """The same problem on a grid coarsened by ``factor`` in x and y."""
    nz, ny, nx = p["shape"]
    cy, cx = ny // factor, nx // factor
    sy, sx = cy * factor, cx * factor

    def block_mean(a, weight=None):
        lead = a.shape[:-2]
        a = a[..., :sy, :sx].reshape(lead + (cy, factor, cx, factor))
        if weight is None:
            return a.mean(axis=(-3, -1))
        w = np.broadcast_to(weight[..., :sy, :sx], a.shape[:-4] + (sy, sx))
        w = w.reshape(lead + (cy, factor, cx, factor))
        ws = w.sum(axis=(-3, -1))
        with np.errstate(invalid="ignore", divide="ignore"):
            out = (a * w).sum(axis=(-3, -1)) / ws
        return np.where(ws > 0, out, 0.0)

    nr = p["target"].shape[0]
    w = p["weight"]
    q = dict(p)
    q["coef"] = block_mean(p["coef"].reshape((nr, 3) + p["shape"]), w[:, None]).reshape(
        (nr * 3, nz, cy, cx)
    )
    q["target"] = block_mean(p["target"], w)
    # weight of a coarse cell: Co times the number of fine observations in it
    q["weight"] = block_mean(w) * factor * factor
    q["rho"] = block_mean(p["rho"])
    q["bg"] = block_mean(p["bg"], p["bg_weight"])
    q["bg_weight"] = block_mean(p["bg_weight"]) * factor * factor
    q["vort_weight"] = (block_mean(p["vort_weight"]) > 0.5).astype(np.float64)
    q["free"] = block_mean(p["free"].astype(np.float64)) > 0.5
    q["first_guess"] = block_mean(p["first_guess"])
    q["dx"], q["dy"] = p["dx"] * factor, p["dy"] * factor
    q["h"] = p["h"] * factor
    q["vort_scale"] = q["h"] ** 2 / _VREF
    # Keep every term per unit area for a smooth field: a coarse cell stands
    # for factor**2 fine cells; horizontal second differences in grid units
    # grow by factor**2 (their squares by factor**4), the scaled mass residual
    # h D by factor and the scaled vorticity residual h**2 R / U by factor**2.
    a = factor * factor
    q["csx"], q["csy"] = p["csx"] / a, p["csy"] / a
    q["csz"] = p["csz"] * a
    q["cm"] = p["cm"]
    q["co"], q["cb"] = p["co"] * a, p["cb"] * a
    q["cv"] = p["cv"] / a
    q["shape"] = (nz, cy, cx)
    for c in ("coef", "target", "weight", "rho", "bg", "bg_weight", "first_guess"):
        q[c] = np.ascontiguousarray(q[c])
    return q


def _refine(state, coarse_shape, fine_shape, factor=2):
    """Interpolate a coarse state to the fine grid (linear, edges extended)."""
    from scipy.interpolate import RegularGridInterpolator

    nz, cy, cx = coarse_shape
    _, ny, nx = fine_shape
    yc = (np.arange(cy) + 0.5) * factor - 0.5
    xc = (np.arange(cx) + 0.5) * factor - 0.5
    yf, xf = np.arange(ny, dtype=float), np.arange(nx, dtype=float)
    Y, X = np.meshgrid(
        np.clip(yf, yc[0], yc[-1]), np.clip(xf, xc[0], xc[-1]), indexing="ij"
    )
    out = np.empty((3, nz, ny, nx))
    for q in range(3):
        for k in range(nz):
            f = RegularGridInterpolator((yc, xc), state[q, k])
            out[q, k] = f((Y, X))
    return out


def _solve(
    problem, state0, use_compiled, n_threads, max_iterations, tolerance, history
):
    """L-BFGS-B on the control vector (u, v and the free w)."""
    from scipy.optimize import minimize

    free = problem["free"]
    n = free.size
    last = {}

    def to_state(xv):
        s = np.zeros((3,) + free.shape)
        s[0] = xv[:n].reshape(free.shape)
        s[1] = xv[n : 2 * n].reshape(free.shape)
        s[2][free] = xv[2 * n :]
        return s

    def fun(xv):
        terms, grad = _cost_gradient(problem, to_state(xv), use_compiled, n_threads)
        last["terms"] = terms
        g = np.concatenate([grad[0].ravel(), grad[1].ravel(), grad[2][free]])
        return float(terms.sum()), g

    def callback(*_):
        history.append(np.array(last["terms"]))

    x0 = np.concatenate([state0[0].ravel(), state0[1].ravel(), state0[2][free]])
    fun(x0)
    history.append(np.array(last["terms"]))
    res = minimize(
        fun,
        x0,
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        options={
            "maxiter": int(max_iterations),
            "ftol": tolerance,
            "gtol": 1e-10,
            "maxcor": 10,
        },
    )
    return to_state(res.x), res


def _spectral_preconditioner(problem):
    """
    Inverse of ``lambda + Hessian of the smoothness term`` in a cosine basis.

    The second differences of the smoothness term are diagonal in the basis
    of the discrete cosine transform (DCT-II), with eigenvalues
    :math:`-4 \\sin^2(\\pi k / 2n)`. Dividing by
    :math:`\\lambda + 2 \\sum_d C_{sd}\\, 16 \\sin^4(\\pi k_d / 2 n_d)` removes
    the stiffness of the smoothness penalty, which dominates the condition
    number; :math:`\\lambda` stands for the observation and background terms.
    """
    from scipy.fft import dctn, idctn

    nz, ny, nx = problem["shape"]

    def eig(n):
        return 16.0 * np.sin(np.pi * np.arange(n) / (2.0 * n)) ** 4

    lam = _PRECONDITIONER_LAMBDA * problem["co"] + 2.0 * problem["cb"]
    h = lam + 2.0 * (
        problem["csx"] * eig(nx)[None, None, :]
        + problem["csy"] * eig(ny)[None, :, None]
        + problem["csz"] * eig(nz)[:, None, None]
    )
    inv = 1.0 / h

    def apply(r):
        out = np.empty_like(r)
        for q in range(3):
            c = dctn(r[q], type=2, norm="ortho", workers=-1)
            c *= inv
            out[q] = idctn(c, type=2, norm="ortho", workers=-1)
        return out

    return apply


def _solve_cg(
    problem, state0, use_compiled, n_threads, max_iterations, tolerance, history
):
    """
    Preconditioned conjugate gradients for the quadratic cost.

    Without the vorticity term the cost is quadratic, :math:`J(x) = x^T A x / 2
    - b^T x + c`, with gradient :math:`A x - b`. The product :math:`A p` is
    the gradient of the same problem with zero observations and background,
    evaluated at ``p``. Every iteration takes the exact minimising step along
    the conjugate direction (no line search). The preconditioner is
    :func:`_spectral_preconditioner`. ``w`` at the fixed boundary cells stays
    zero: the iteration runs on the free cells only.
    """
    from scipy.optimize import OptimizeResult

    free = problem["free"]
    mask = np.ones((3,) + free.shape)
    mask[2][~free] = 0.0
    homogeneous = dict(
        problem,
        target=np.zeros_like(problem["target"]),
        bg=np.zeros_like(problem["bg"]),
    )
    precondition = _spectral_preconditioner(problem)
    x = np.array(state0, dtype=np.float64)
    x[2][~free] = 0.0
    terms, grad = _cost_gradient(problem, x, use_compiled, n_threads)
    history.append(np.array(terms))
    r = -grad * mask
    z = precondition(r) * mask
    rz = float(np.vdot(r, z))
    # stop relative to the right-hand side b = -grad J(0), so that the
    # criterion does not depend on the first guess
    _, g0 = _cost_gradient(problem, np.zeros_like(x), use_compiled, n_threads)
    g0 *= mask
    rz0 = float(np.vdot(g0, precondition(g0) * mask))
    p = z.copy()
    nit, message, success = 0, "maximum number of iterations reached", False
    for it in range(1, int(max_iterations) + 1):
        if rz <= tolerance**2 * rz0:
            message, success = "preconditioned gradient norm below tolerance", True
            break
        _, ap = _cost_gradient(homogeneous, p, use_compiled, n_threads)
        ap *= mask
        pap = float(np.vdot(p, ap))
        if not pap > 0.0:  # pragma: no cover - A is positive semi-definite
            message = "direction of non-positive curvature"
            break
        x += (rz / pap) * p
        terms, grad = _cost_gradient(problem, x, use_compiled, n_threads)
        history.append(np.array(terms))
        nit = it
        r = -grad * mask
        z = precondition(r) * mask
        rz_new = float(np.vdot(r, z))
        p *= rz_new / rz
        p += z
        rz = rz_new
    else:
        if rz <= tolerance**2 * rz0:
            message, success = "preconditioned gradient norm below tolerance", True
    res = OptimizeResult(
        x=x, nit=nit, nfev=2 * nit + 2, success=success, message=message
    )
    return x, res


DEFAULT_WEIGHTS = {
    "observation": 1.0,
    "mass_continuity": 10.0,
    "smoothness": 0.5,
    "smoothness_vertical": None,
    "background": 0.001,
    "background_w": 0.0,
    "vorticity": 0.0,
}


def multi_doppler(
    grids,
    background=None,
    *,
    velocity="VRADH",
    reflectivity="DBZH",
    weights=None,
    fall_speed_correction=True,
    w_boundary="bottom",
    storm_motion=(0.0, 0.0),
    first_guess=None,
    levels=3,
    max_iterations=500,
    tolerance=None,
    solver="auto",
    engine="auto",
    n_threads=None,
):
    """
    Retrieve the three-dimensional wind from two or more Doppler radars.

    The wind ``(u, v, w)`` on the grid minimises the cost function
    :math:`J = J_o + J_m + J_s + J_b + J_v` (Gao et al. 1999 [1]_):

    .. math::

       J_o &= C_o \\sum_k \\sum_i \\left(u \\cos\\phi_k \\sin\\alpha_k
              + v \\cos\\phi_k \\cos\\alpha_k + (w - V_t) \\sin\\phi_k
              - v_{r,k}\\right)^2 \\\\
       J_m &= C_m \\sum_i \\left(\\frac{h}{\\rho}\\left[
              \\frac{\\partial \\rho u}{\\partial x}
              + \\frac{\\partial \\rho v}{\\partial y}
              + \\frac{\\partial \\rho w}{\\partial z}\\right]\\right)^2 \\\\
       J_s &= C_s \\sum_i \\sum_{f=u,v,w} (\\delta_{xx} f)^2 + (\\delta_{yy} f)^2
              + \\frac{C_{sz}}{C_s} (\\delta_{zz} f)^2 \\\\
       J_b &= C_b \\sum_i (u - u_b)^2 + (v - v_b)^2 \\\\
       J_v &= C_v \\sum_i \\left(\\frac{h^2}{U} R_\\zeta\\right)^2

    over the grid cells :math:`i` and radars :math:`k`, where
    :math:`\\alpha_k`, :math:`\\phi_k` are the beam azimuth and local
    elevation, :math:`V_t` the fall speed (:func:`fall_speed`), :math:`\\rho`
    the air density, :math:`h = \\sqrt{\\Delta x \\Delta y}`, :math:`\\delta`
    second differences in grid units and :math:`U = 10` m s-1. The scaling
    makes every term a squared velocity, so the weights are dimensionless
    and per grid cell. :math:`R_\\zeta` is the residual of the vertical
    vorticity equation (Shapiro et al. 2009 [2]_; Potvin et al. 2012 [3]_),
    taken as steady in a frame moving with ``storm_motion``,

    .. math::

       R_\\zeta = (u - U_s) \\zeta_x + (v - V_s) \\zeta_y + w \\zeta_z
                  + (\\zeta + f)(u_x + v_y) + w_x v_z - w_y u_z ,

    applied where at least two radars observe. Derivatives are centred
    differences (one-sided at the edges); the smoothness penalty is applied
    to the second derivatives in each direction (Potvin et al. 2012).

    The cost and its exact gradient are computed in one fused, multithreaded
    pass of a compiled kernel. Without the vorticity term the cost is
    quadratic and is minimised by conjugate gradients (exact steps, no line
    search), preconditioned with the inverse of the smoothness operator
    (plus a constant) in the basis of the discrete cosine transform; with
    the vorticity term SciPy's L-BFGS-B is used. The minimisation runs
    first on grids coarsened by two horizontally (``levels``), each solution
    starting the next finer one. The first guess is the background wind with
    ``w = 0``. ``w`` is held at zero at the boundaries named in
    ``w_boundary`` by removing those cells from the control vector.

    Parameters
    ----------
    grids : xarray.Dataset or sequence of xarray.Dataset
        Gridded, dealiased and advection-corrected radial velocities on a
        common grid with a ``radar`` dimension (see
        :func:`multi_doppler_input`): ``velocity`` on ``(radar, z, y, x)``
        and either ``azimuth`` and ``elevation`` on ``(radar, z, y, x)`` or
        ``radar_x``, ``radar_y`` and ``radar_altitude`` on ``radar``. A
        sequence of per-radar Datasets is concatenated along ``radar``.
    background : xarray.Dataset, optional
        Background on the grid, from
        :func:`radarx.io.sounding.era5_column` or
        :func:`radarx.io.sounding.profile_to_grid`: ``u``, ``v`` (along the
        grid axes), optionally ``w``, ``air_density`` and
        ``freezing_level``. Without it there is no background term and a
        standard density profile is used.
    velocity, reflectivity : str, optional
        Field names. Default ``"VRADH"`` and ``"DBZH"``.
    weights : dict, optional
        Overrides of the dimensionless weights ``observation`` (:math:`C_o`,
        default 1), ``mass_continuity`` (:math:`C_m`, 1), ``smoothness``
        (:math:`C_s`, 0.5), ``smoothness_vertical`` (:math:`C_{sz}`, default
        the same as ``smoothness``), ``background`` (:math:`C_b`, 0.01),
        ``background_w`` (weight of ``w - w_b``, 0) and ``vorticity``
        (:math:`C_v`, 0, i.e. off).
    fall_speed_correction : bool, optional
        Correct for the precipitation fall speed. Default True.
    w_boundary : str or sequence of str, optional
        Where ``w = 0``: any of ``"bottom"`` (lowest level, default),
        ``"top"`` (highest level) and ``"echo_top"`` (from one level above
        the highest echo in each column). ``None`` for no boundary condition.
    storm_motion : tuple of float, optional
        ``(u, v)`` of the frame in which the vorticity is steady, m s-1.
    first_guess : xarray.Dataset, optional
        Start from these ``u``, ``v``, ``w`` instead of the background (e.g.
        the previous analysis of a time series); then ``levels`` is 1.
    levels : int, optional
        Number of grid levels of the coarse-to-fine minimisation; each
        coarser level halves the horizontal resolution (only while the grid
        keeps at least 12 points). Default 3.
    max_iterations : int, optional
        Maximum iterations per level. Default 500.
    tolerance : float, optional
        Stopping tolerance. For ``"cg"``: the preconditioned gradient norm
        relative to that of the cost at zero wind (default 3e-4); for
        ``"lbfgsb"``: the relative reduction of the cost (``ftol``, default
        1e-7).
    solver : {"auto", "cg", "lbfgsb"}, optional
        Minimiser. ``"auto"`` (default) uses conjugate gradients when the
        vorticity term is off and L-BFGS-B otherwise.
    engine : {"auto", "compiled", "numpy"}, optional
        Cost function implementation. ``"auto"`` prefers the compiled kernel.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.

    Returns
    -------
    xarray.Dataset
        On ``(z, y, x)``: ``u``, ``v`` (along the grid axes), ``w``,
        ``fall_speed``, ``mass_residual`` (anelastic mass divergence, s-1),
        ``n_radars`` (radars observing each cell) and, with two or more
        radars, ``beam_crossing_angle``; ``vr_residual`` (model minus
        observed radial velocity) on ``(radar, z, y, x)``; ``cost`` on
        ``term`` (final value of every term) and ``cost_history`` on
        ``(iteration, term)``. ``attrs`` record the weights, iterations,
        convergence message and run time.

    References
    ----------
    .. [1] Gao, J., M. Xue, A. Shapiro, and K. K. Droegemeier, 1999: A
       variational method for the analysis of three-dimensional wind fields
       from two Doppler radars. *Mon. Wea. Rev.*, **127**, 2128-2142,
       https://doi.org/10.1175/1520-0493(1999)127<2128:AVMFTA>2.0.CO;2
    .. [2] Shapiro, A., C. K. Potvin, and J. Gao, 2009: Use of a vertical
       vorticity equation in variational dual-Doppler wind analysis. *J.
       Atmos. Oceanic Technol.*, **26**, 2089-2106,
       https://doi.org/10.1175/2009JTECHA1256.1
    .. [3] Potvin, C. K., A. Shapiro, and M. Xue, 2012: Impact of a vertical
       vorticity constraint in variational dual-Doppler wind analysis: Tests
       with real and simulated supercell data. *J. Atmos. Oceanic Technol.*,
       **29**, 32-49, https://doi.org/10.1175/JTECH-D-11-00019.1
    .. [4] Collis, S., A. Protat, and K.-S. Chung, 2010: The effect of radial
       velocity gridding artifacts on variationally retrieved vertical
       velocities. *J. Atmos. Oceanic Technol.*, **27**, 1239-1246,
       https://doi.org/10.1175/2010JTECHA1402.1

    Examples
    --------
    >>> grids = radarx.retrieve.multi_doppler_input([kgwx, kcbm], x, y, z)  # doctest: +SKIP
    >>> bg = radarx.io.sounding.era5_column(grids)  # doctest: +SKIP
    >>> wind = radarx.retrieve.multi_doppler(grids, bg)  # doctest: +SKIP
    """
    use_compiled = _use_compiled(engine)
    if solver not in ("auto", "cg", "lbfgsb"):
        raise ValueError(f"solver must be 'auto', 'cg' or 'lbfgsb', not {solver!r}")
    w_ = dict(DEFAULT_WEIGHTS)
    if weights:
        unknown = set(weights) - set(w_)
        if unknown:
            raise ValueError(f"unknown weights: {sorted(unknown)}")
        w_.update(weights)
    if w_["smoothness_vertical"] is None:
        w_["smoothness_vertical"] = w_["smoothness"]
    if solver == "auto":
        solver = "lbfgsb" if w_["vorticity"] != 0.0 else "cg"
    if solver == "cg" and w_["vorticity"] != 0.0:
        raise ValueError("the vorticity term is not quadratic: use solver='lbfgsb'")
    solve = _solve_cg if solver == "cg" else _solve
    if tolerance is None:
        tolerance = 3e-4 if solver == "cg" else 1e-7
    if int(levels) < 1:
        raise ValueError("levels must be at least 1")
    if background is None and w_["background"] != 0.0:
        w_["background"] = 0.0
    ds = _as_multi_radar(grids, velocity)
    if background is not None:
        for name in ("z", "y", "x"):
            if name in background.dims and background.sizes[name] != ds.sizes[name]:
                raise ValueError("the background must be on the grid of the radar data")
    t_start = _time.perf_counter()
    problem = _prepare(
        ds,
        background,
        velocity=velocity,
        reflectivity=reflectivity,
        use_fall_speed=fall_speed_correction,
        weights=w_,
        storm_motion=storm_motion,
        w_boundary=w_boundary,
    )
    state0 = problem["first_guess"]
    if first_guess is not None:
        state0 = np.stack(
            [
                np.nan_to_num(
                    first_guess[n].transpose("z", "y", "x").values.astype(float)
                )
                for n in ("u", "v", "w")
            ]
        )
        levels = 1
    history, results, level_of = [], [], []
    hierarchy = [problem]
    while len(hierarchy) < levels and min(hierarchy[-1]["shape"][1:]) >= 12:
        hierarchy.append(_coarsen_problem(hierarchy[-1]))
    guesses = [state0]
    for _ in hierarchy[1:]:
        guesses.append(_coarsen_first_guess(guesses[-1]))
    state = guesses[-1]
    for level in range(len(hierarchy) - 1, -1, -1):
        p = hierarchy[level]
        if level < len(hierarchy) - 1:
            state = _refine(state, hierarchy[level + 1]["shape"], p["shape"])
            state[2][~p["free"]] = 0.0
        n0 = len(history)
        state, res = solve(
            p, state, use_compiled, n_threads, max_iterations, tolerance, history
        )
        level_of += [level] * (len(history) - n0)
        results.append(res)
    elapsed = _time.perf_counter() - t_start
    final, _ = _cost_gradient(problem, state, use_compiled, n_threads)
    return _output(ds, problem, state, final, history, level_of, results, w_, elapsed)


def _coarsen_first_guess(state, factor=2):
    nz, ny, nx = state.shape[1:]
    cy, cx = ny // factor, nx // factor
    a = state[:, :, : cy * factor, : cx * factor].reshape(3, nz, cy, factor, cx, factor)
    return a.mean(axis=(3, 5))


def _output(ds, p, state, final, history, level_of, results, weights, elapsed):
    """Wrap the solution and diagnostics as an xarray Dataset."""
    dims = ("z", "y", "x")
    u, v, w = state
    nr = p["target"].shape[0]
    coef = p["coef"].reshape((nr, 3) + p["shape"])
    valid = p["weight"] > 0
    model = coef[:, 0] * u + coef[:, 1] * v + coef[:, 2] * (w - p["vfall"])
    resid = np.where(valid, model - p["vr"], np.nan)
    rho = p["rho"]
    mass = (
        _d1(rho * u, 2, p["dx"]) + _d1(rho * v, 1, p["dy"]) + _d1(rho * w, 0, p["dz"])
    ) / rho
    vt = p["vfall"][0]
    coords = {c: ds.coords[c] for c in ds.coords if set(ds.coords[c].dims) <= set(dims)}
    out = xr.Dataset(
        {
            "u": (
                dims,
                u.astype(np.float32),
                {
                    "standard_name": "x_wind",
                    "long_name": "retrieved wind component along the grid x axis",
                    "units": "m s-1",
                },
            ),
            "v": (
                dims,
                v.astype(np.float32),
                {
                    "standard_name": "y_wind",
                    "long_name": "retrieved wind component along the grid y axis",
                    "units": "m s-1",
                },
            ),
            "w": (
                dims,
                w.astype(np.float32),
                {
                    "standard_name": "upward_air_velocity",
                    "long_name": "retrieved vertical air velocity",
                    "units": "m s-1",
                },
            ),
            "fall_speed": (
                dims,
                np.where(valid.any(0), vt, np.nan).astype(np.float32),
                {
                    "long_name": "terminal fall speed of precipitation (positive downward)",
                    "units": "m s-1",
                },
            ),
            "mass_residual": (
                dims,
                mass.astype(np.float32),
                {
                    "long_name": "anelastic mass divergence of the retrieved wind",
                    "units": "s-1",
                },
            ),
            "n_radars": (
                dims,
                p["n_obs"].astype(np.int8),
                {
                    "long_name": "number of radars observing the grid cell",
                    "units": "1",
                },
            ),
            "vr_residual": (
                ("radar",) + dims,
                resid.astype(np.float32),
                {
                    "long_name": "retrieved minus observed radial velocity",
                    "units": "m s-1",
                },
            ),
            "cost": (
                ("term",),
                np.asarray(final, dtype=np.float64),
                {
                    "long_name": "final value of each cost function term",
                    "units": "m2 s-2",
                },
            ),
            "cost_history": (
                ("iteration", "term"),
                np.asarray(history),
                {
                    "long_name": "cost function terms at each L-BFGS-B iteration",
                    "units": "m2 s-2",
                },
            ),
        },
        coords=coords,
    )
    out = out.assign_coords(
        term=("term", list(TERMS)),
        iteration=("iteration", np.arange(len(history))),
        grid_level=(
            "iteration",
            np.asarray(level_of, dtype=np.int8),
            {
                "long_name": "grid level of the iteration (0: full resolution, "
                "n: coarsened by 2**n)",
            },
        ),
    )
    if "radar" in ds.coords:
        out = out.assign_coords(radar=ds["radar"])
    for c in ("radar_name", "time"):
        if c in ds.coords and ds.coords[c].dims == ("radar",):
            out = out.assign_coords({c: ds.coords[c]})
    if nr >= 2:
        out["beam_crossing_angle"] = (
            dims,
            _crossing_angle(ds, valid).astype(np.float32),
            {
                "long_name": "largest horizontal angle between two observing beams",
                "units": "degree",
            },
        )
    out.attrs = {
        "method": "variational multi-Doppler analysis (Gao et al. 1999)",
        "weights": str({k: float(v) for k, v in weights.items()}),
        "w_boundary_cells": int((~p["free"]).sum()),
        "iterations": str([int(r.nit) for r in results]),
        "function_evaluations": str([int(r.nfev) for r in results]),
        "converged": int(all(r.success for r in results)),
        "message": str([str(r.message) for r in results]),
        "run_time_s": float(elapsed),
    }
    return out


def _crossing_angle(ds, valid):
    """Largest horizontal angle (0-90 deg) between beams of two radars."""
    az = np.radians(
        ds["azimuth"].transpose("radar", "z", "y", "x").values.astype(float)
    )
    best = np.zeros(az.shape[1:])
    nr = az.shape[0]
    for i in range(nr):
        for j in range(i + 1, nr):
            d = np.abs(np.degrees(np.arcsin(np.abs(np.sin(az[i] - az[j])))))
            d = np.where(valid[i] & valid[j], d, 0.0)
            best = np.maximum(best, d)
    return np.where(valid.sum(0) >= 2, best, np.nan)
