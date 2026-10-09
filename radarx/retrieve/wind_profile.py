#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Wind Profiles: Shear, Storm Motion and Helicity
===============================================

Kinematic parameters of a vertical wind profile: any :py:class:`xarray.Dataset`
with ``u`` and ``v`` (m s-1) on a vertical dimension, such as a sounding or an
ERA5 profile or column from :mod:`radarx.io.sounding`, or a wind profiler
(:func:`radarx.io.read_wind_profiler`, on ``(time, height)``). All columns of
the input (times, grid points) are computed in one kernel call.

Layers
------
Layer limits are heights (m) above ``ground``: by default the lowest level
with a valid wind of every column (the surface for a sounding, the first gate
of a profiler); pass ``ground=0`` for heights that are already above ground
level, or the station altitude for heights above sea level. Winds at the
limits are interpolated linearly between valid levels; a layer that the valid
winds do not span gives NaN.

Definitions
-----------
- bulk shear: the vector wind difference :math:`\\mathbf{V}(z_t) -
  \\mathbf{V}(z_b)`; its component along a direction, e.g. normal to a squall
  line, is the :math:`\\Delta u` of RKW theory (Rotunno et al. [3]_; Weisman and
  Rotunno [4]_). The default layer of 0-6 km in :func:`bulk_shear` is the
  conventional deep-layer shear; it is a radarx default, not a layer taken
  from the cited papers (RKW theory uses the lowest few kilometres; pass
  ``bottom`` and ``top`` for it).
- storm-relative helicity, defined from the streamwise-vorticity framework of
  Davies-Jones [2]_ (equation numbers not checked against the paper; the discrete
  sum is derived here),

  .. math::

      H = -\\int_{z_b}^{z_t} \\mathbf{k} \\cdot (\\mathbf{V} - \\mathbf{c})
      \\times \\frac{\\partial \\mathbf{V}}{\\partial z}\\, dz
      = \\sum_k (u_{k+1} - c_x)(v_k - c_y) - (u_k - c_x)(v_{k+1} - c_y),

  exact for winds varying linearly between levels.
- velocity-azimuth display (VAD, Browning and Wexler [1]_): on every range
  ring of a radar sweep the radial velocity is fitted by least squares with
  :math:`v_r = a_0 + u \\cos\\phi \\sin\\alpha + v \\cos\\phi \\cos\\alpha`
  (azimuth :math:`\\alpha`, elevation :math:`\\phi`), assuming a horizontally
  uniform wind across the ring; :func:`vad_profile` averages the ring winds
  of all sweeps in height bins into a profile usable by the functions here.
  The defaults of :func:`vad_profile` (at least 50 gates per ring, azimuthal
  spread 0.1, 100-m bins up to 12 km, 1-45 degree elevations) are radarx
  choices; none comes from Browning and Wexler [1]_.
- Bunkers et al. [5]_ "internal dynamics" storm motion as implemented: the
  0-6 km (non-pressure-weighted, height-weighted here) mean wind plus a
  deviation of 7.5 m s-1 at right angles to the shear vector between the
  0-0.5 km and the 5.5-6 km mean winds; to the right of the shear for right
  movers, to the left for left movers. These layers and the 7.5 m s-1 are the
  values the implementation uses for that method; they and the choice of a
  height-weighted mean are not checked against the text or tables of the
  paper.

The layer integrals run in the compiled kernel ``radarx.retrieve._coldpool``
(multithreaded over columns) with an identical NumPy implementation as
fallback.

References
----------
.. [1] Browning, K. A., and R. Wexler, 1968: The determination of kinematic
   properties of a wind field using Doppler radar. *J. Appl. Meteor.*, **7**
   (1), 105-113,
   https://doi.org/10.1175/1520-0450(1968)007<0105:TDOKPO>2.0.CO;2

.. [2] Davies-Jones, R., 1984: Streamwise vorticity: The origin of updraft
   rotation in supercell storms. *J. Atmos. Sci.*, **41** (20), 2991-3006,
   https://doi.org/10.1175/1520-0469(1984)041<2991:SVTOOU>2.0.CO;2

.. [3] Rotunno, R., J. B. Klemp, and M. L. Weisman, 1988: A theory for strong,
   long-lived squall lines. *J. Atmos. Sci.*, **45** (3), 463-485,
   https://doi.org/10.1175/1520-0469(1988)045<0463:ATFSLL>2.0.CO;2

.. [4] Weisman, M. L., and R. Rotunno, 2004: "A theory for strong long-lived
   squall lines" revisited. *J. Atmos. Sci.*, **61** (4), 361-382,
   https://doi.org/10.1175/1520-0469(2004)061<0361:ATFSLS>2.0.CO;2

.. [5] Bunkers, M. J., B. A. Klimowski, J. W. Zeitler, R. L. Thompson, and
   M. L. Weisman, 2000: Predicting supercell motion using a new hodograph
   technique. *Wea. Forecasting*, **15** (1), 61-79,
   https://doi.org/10.1175/1520-0434(2000)015<0061:PSMUAN>2.0.CO;2

.. autosummary::
   :nosignatures:
   :toctree: generated/

   bulk_shear
   layer_mean_wind
   bunkers_storm_motion
   storm_relative_wind
   storm_relative_helicity
   vad_profile
"""

from __future__ import annotations

__all__ = [
    "bulk_shear",
    "layer_mean_wind",
    "bunkers_storm_motion",
    "storm_relative_wind",
    "storm_relative_helicity",
    "vad_profile",
]

import numpy as np
import xarray as xr

from .._registry import accessor_method
from .coldpool import _columns, _f64, _height_of, _kernel, _per_column, _threads

# Deviation of the Bunkers et al. (2000) method [m s-1]; value as used by the
# method, not checked against the paper.
BUNKERS_DEVIATION = 7.5  # m s-1


def _motion(storm_motion):
    if storm_motion is None:
        return None, None
    if isinstance(storm_motion, xr.Dataset):
        return storm_motion["u"], storm_motion["v"]
    cx, cy = storm_motion
    return cx, cy


def _layer(
    profile, bottom, top, *, dim, height, ground, storm_motion, engine, n_threads
):
    """Kernel results (7 rows) for every column, with the output layout."""
    if "u" not in profile or "v" not in profile:
        raise ValueError("the profile needs wind components 'u' and 'v'")
    if not top >= bottom:
        raise ValueError("top must not be below bottom")
    z = _height_of(profile["u"], dim, height)
    (u, v), zc, other, coords, shape = _columns([profile["u"], profile["v"]], dim, z)
    g = _per_column(ground, other, coords, shape)
    cx, cy = _motion(storm_motion)
    cu = _per_column(cx, other, coords, shape)
    cv = _per_column(cy, other, coords, shape)
    res = np.asarray(
        _kernel(engine).profile(
            zc,
            u,
            v,
            g,
            float(bottom),
            float(top),
            cu,
            cv,
            n_threads=_threads(n_threads),
        )
    )
    return res, other, coords, shape


def _da(values, other, coords, shape, name, attrs=None):
    return xr.DataArray(
        values.reshape(shape), dims=other, coords=coords, name=name, attrs=attrs or {}
    )


def _wind_vars(u, v, prefix, what):
    """u, v, speed and direction (from) of a wind vector."""
    speed = np.hypot(u, v)
    direction = (np.degrees(np.arctan2(-u, -v)) % 360.0).where(speed > 0)
    return {
        f"{prefix}u": u.rename(f"{prefix}u").assign_attrs(
            long_name=f"{what}, eastward (x) component", units="m s-1"
        ),
        f"{prefix}v": v.rename(f"{prefix}v").assign_attrs(
            long_name=f"{what}, northward (y) component", units="m s-1"
        ),
        f"{prefix}speed": speed.rename(f"{prefix}speed").assign_attrs(
            long_name=f"{what}, speed", units="m s-1"
        ),
        f"{prefix}direction": direction.rename(f"{prefix}direction").assign_attrs(
            long_name=f"{what}, direction it comes from (clockwise from north)",
            units="degree",
        ),
    }


def _along(u, v, azimuth):
    a = np.radians(azimuth)
    return u * np.sin(a) + v * np.cos(a)


def bulk_shear(
    profile,
    bottom=0.0,
    top=6000.0,
    *,
    normal=None,
    dim="height",
    height=None,
    ground=None,
    engine="auto",
    n_threads=None,
):
    """
    Bulk wind difference (bulk shear) over a layer.

    The vector difference of the (linearly interpolated) winds at ``top`` and
    ``bottom``; its component along ``normal`` is the :math:`\\Delta u` of
    RKW theory (Weisman and Rotunno [1]_; Rotunno et al. [2]_). The 0-6 km
    default layer is a radarx choice, not a layer prescribed by those papers.

    Parameters
    ----------
    profile : xarray.Dataset
        ``u`` and ``v`` (m s-1) on the vertical dimension ``dim``.
    bottom, top : float, optional
        Layer limits (m above ``ground``). Default 0-6 km.
    normal : float or xarray.DataArray, optional
        Azimuth (degrees clockwise from north) of a direction, e.g. the
        normal to a squall line pointing toward its inflow (its direction of
        motion). The shear component along it is returned as
        ``shear_normal``, the :math:`\\Delta u` of RKW theory.
    dim : str, optional
        Vertical dimension. Default ``"height"``.
    height : str or xarray.DataArray, optional
        Heights (m) if not the ``dim`` coordinate.
    ground : float or xarray.DataArray, optional
        Height (m) of the ground in the units of the heights; default the
        lowest level with a valid wind.
    engine : {"auto", "compiled", "numpy"}, optional
        Kernel implementation.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.

    Returns
    -------
    xarray.Dataset
        ``shear_u``, ``shear_v``, ``shear_speed``, ``shear_direction`` (the
        direction the shear vector points from, as for winds) and, with
        ``normal``, ``shear_normal``.

    References
    ----------
    .. [1] Weisman, M. L., and R. Rotunno, 2004: "A theory for strong
           long-lived squall lines" revisited. *J. Atmos. Sci.*, **61** (4),
           361-382, https://doi.org/10.1175/1520-0469(2004)061<0361:ATFSLS>2.0.CO;2

    .. [2] Rotunno, R., J. B. Klemp, and M. L. Weisman, 1988: A theory for
           strong, long-lived squall lines. *J. Atmos. Sci.*, **45** (3),
           463-485,
           https://doi.org/10.1175/1520-0469(1988)045<0463:ATFSLL>2.0.CO;2
    """
    res, other, coords, shape = _layer(
        profile,
        bottom,
        top,
        dim=dim,
        height=height,
        ground=ground,
        storm_motion=None,
        engine=engine,
        n_threads=n_threads,
    )
    du = _da(res[2] - res[0], other, coords, shape, "shear_u")
    dv = _da(res[3] - res[1], other, coords, shape, "shear_v")
    what = f"Bulk wind difference {bottom:g}-{top:g} m"
    out = xr.Dataset(_wind_vars(du, dv, "shear_", what))
    if normal is not None:
        out["shear_normal"] = _along(du, dv, normal).assign_attrs(
            long_name=f"{what}, component along the given normal direction",
            units="m s-1",
        )
    out.attrs.update(layer_bottom=float(bottom), layer_top=float(top))
    return out


def layer_mean_wind(
    profile,
    bottom=0.0,
    top=6000.0,
    *,
    dim="height",
    height=None,
    ground=None,
    engine="auto",
    n_threads=None,
):
    """
    Height-weighted (non-pressure-weighted) mean wind of a layer.

    The trapezoidal integral of the winds over height divided by the layer
    depth. The 0-6 km default is a radarx choice (it is the layer of the mean
    wind in the Bunkers et al. method, :func:`bunkers_storm_motion`).

    Parameters
    ----------
    profile : xarray.Dataset
        ``u`` and ``v`` (m s-1) on the vertical dimension ``dim``.
    bottom, top : float, optional
        Layer limits (m above ``ground``). Default 0-6 km.
    dim, height, ground, engine, n_threads : optional
        As in :func:`bulk_shear`.

    Returns
    -------
    xarray.Dataset
        ``u``, ``v``, ``speed`` and ``direction`` of the mean wind.
    """
    res, other, coords, shape = _layer(
        profile,
        bottom,
        top,
        dim=dim,
        height=height,
        ground=ground,
        storm_motion=None,
        engine=engine,
        n_threads=n_threads,
    )
    u = _da(res[5], other, coords, shape, "u")
    v = _da(res[6], other, coords, shape, "v")
    out = xr.Dataset(_wind_vars(u, v, "", f"Mean wind {bottom:g}-{top:g} m"))
    out.attrs.update(layer_bottom=float(bottom), layer_top=float(top))
    return out


def bunkers_storm_motion(
    profile,
    *,
    mover="right",
    deviation=BUNKERS_DEVIATION,
    dim="height",
    height=None,
    ground=None,
    engine="auto",
    n_threads=None,
):
    """
    Supercell motion of Bunkers et al. [1]_ ("internal dynamics" method).

    Mean wind of 0-6 km plus (right mover) or minus (left mover) a deviation
    of 7.5 m s-1 perpendicular to the shear vector from the 0-0.5 km mean wind
    to the 5.5-6 km mean wind. The layers, the 7.5 m s-1 and the use of
    height-weighted rather than pressure-weighted means are as implemented
    for the method; they are not checked against the text of the paper (the pressure-weighted variant is not offered).

    Parameters
    ----------
    profile : xarray.Dataset
        ``u`` and ``v`` (m s-1) on the vertical dimension ``dim``, reaching
        6 km above ``ground``.
    mover : {"right", "left", "mean"}, optional
        Right mover (default), left mover or the 0-6 km mean wind.
    deviation : float, optional
        Deviation from the mean wind (m s-1). Default 7.5 (the value of the
        Bunkers et al. method as implemented; not checked, see above).
    dim, height, ground, engine, n_threads : optional
        As in :func:`bulk_shear`.

    Returns
    -------
    xarray.Dataset
        ``u``, ``v``, ``speed`` and ``direction`` (from) of the storm motion,
        usable as ``storm_motion`` of :func:`storm_relative_helicity`.

    References
    ----------
    .. [1] Bunkers, M. J., B. A. Klimowski, J. W. Zeitler, R. L. Thompson, and
           M. L. Weisman, 2000: Predicting supercell motion using a new
           hodograph technique. *Wea. Forecasting*, **15** (1), 61-79,
           https://doi.org/10.1175/1520-0434(2000)015<0061:PSMUAN>2.0.CO;2
    """
    sign = {"right": 1.0, "left": -1.0, "mean": 0.0}
    if mover not in sign:
        raise ValueError(f"mover must be 'right', 'left' or 'mean', not {mover!r}")
    kw = dict(dim=dim, height=height, ground=ground, engine=engine, n_threads=n_threads)
    mean = layer_mean_wind(profile, 0.0, 6000.0, **kw)
    low = layer_mean_wind(profile, 0.0, 500.0, **kw)
    high = layer_mean_wind(profile, 5500.0, 6000.0, **kw)
    su, sv = high["u"] - low["u"], high["v"] - low["v"]
    mag = np.hypot(su, sv)
    # (shear x k) / |shear| points 90 degrees to the right of the shear
    u = mean["u"] + sign[mover] * deviation * sv / mag
    v = mean["v"] - sign[mover] * deviation * su / mag
    out = xr.Dataset(_wind_vars(u, v, "", f"Bunkers et al. (2000) {mover} motion"))
    out.attrs.update(method="Bunkers et al. (2000) internal dynamics", mover=mover)
    return out


def storm_relative_wind(profile, storm_motion, *, normal=None):
    """
    Storm-relative wind profile.

    Parameters
    ----------
    profile : xarray.Dataset
        ``u`` and ``v`` (m s-1).
    storm_motion : tuple of float or xarray.Dataset
        Storm motion ``(cx, cy)`` (m s-1) or a Dataset with ``u`` and ``v``
        (e.g. :func:`bunkers_storm_motion`, or a squall-line motion).
    normal : float or xarray.DataArray, optional
        Azimuth (degrees clockwise from north) of a direction, e.g. the
        normal to a squall line toward its inflow; the storm-relative wind
        along it is returned as ``storm_relative_normal`` (negative: flowing
        into the line from the inflow side).

    Returns
    -------
    xarray.Dataset
        ``storm_relative_u``, ``storm_relative_v``, ``storm_relative_speed``,
        ``storm_relative_direction`` and, with ``normal``,
        ``storm_relative_normal``, on the coordinates of ``profile``.
    """
    cx, cy = _motion(storm_motion)
    u = profile["u"] - cx
    v = profile["v"] - cy
    out = xr.Dataset(_wind_vars(u, v, "storm_relative_", "Storm-relative wind"))
    if normal is not None:
        out["storm_relative_normal"] = _along(u, v, normal).assign_attrs(
            long_name="Storm-relative wind along the given normal direction",
            units="m s-1",
        )
    return out


def storm_relative_helicity(
    profile,
    storm_motion="right",
    bottom=0.0,
    top=3000.0,
    *,
    dim="height",
    height=None,
    ground=None,
    engine="auto",
    n_threads=None,
):
    """
    Storm-relative helicity of a layer.

    :math:`H = \\sum_k (u_{k+1} - c_x)(v_k - c_y) - (u_k - c_x)(v_{k+1} -
    c_y)` over the layer (see the module docstring), the discrete form of the
    storm-relative helicity built on the streamwise vorticity of
    Davies-Jones [1]_ (the sum is exact for winds linear between levels).
    The 0-3 km default layer is the conventional low-level layer, a radarx
    default not taken from that paper; the ``"right"``/``"left"`` motions are
    those of Bunkers et al. [2]_ (:func:`bunkers_storm_motion`). The paper
    equations are not checked.

    Parameters
    ----------
    profile : xarray.Dataset
        ``u`` and ``v`` (m s-1) on the vertical dimension ``dim``.
    storm_motion : {"right", "left"}, tuple of float or xarray.Dataset, optional
        Storm motion: the Bunkers et al. (2000) right (default) or left mover
        of the same profile, ``(cx, cy)`` in m s-1, or a Dataset with ``u``
        and ``v`` (one motion per column, e.g. per time).
    bottom, top : float, optional
        Layer limits (m above ``ground``). Default 0-3 km.
    dim, height, ground, engine, n_threads : optional
        As in :func:`bulk_shear`.

    Returns
    -------
    xarray.DataArray
        ``storm_relative_helicity`` (m2 s-2), positive for streamwise
        vorticity (a hodograph turning clockwise relative to the storm in the
        northern hemisphere).

    References
    ----------
    .. [1] Davies-Jones, R., 1984: Streamwise vorticity: The origin of updraft
           rotation in supercell storms. *J. Atmos. Sci.*, **41** (20),
           2991-3006,
           https://doi.org/10.1175/1520-0469(1984)041<2991:SVTOOU>2.0.CO;2

    .. [2] Bunkers, M. J., B. A. Klimowski, J. W. Zeitler, R. L. Thompson, and
           M. L. Weisman, 2000: Predicting supercell motion using a new
           hodograph technique. *Wea. Forecasting*, **15** (1), 61-79,
           https://doi.org/10.1175/1520-0434(2000)015<0061:PSMUAN>2.0.CO;2
    """
    if isinstance(storm_motion, str):
        storm_motion = bunkers_storm_motion(
            profile,
            mover=storm_motion,
            dim=dim,
            height=height,
            ground=ground,
            engine=engine,
            n_threads=n_threads,
        )
    res, other, coords, shape = _layer(
        profile,
        bottom,
        top,
        dim=dim,
        height=height,
        ground=ground,
        storm_motion=storm_motion,
        engine=engine,
        n_threads=n_threads,
    )
    return _da(
        res[4],
        other,
        coords,
        shape,
        "storm_relative_helicity",
        {
            "long_name": f"Storm-relative helicity {bottom:g}-{top:g} m",
            "units": "m2 s-2",
        },
    )


# --------------------------------------------------------------------------
# VAD
# --------------------------------------------------------------------------

_VELOCITY_NAMES = ("VRADH", "VRADV", "VELOCITY", "velocity", "VEL")


def _sweep_rings(ds, velocity, min_elevation, max_elevation, max_range):
    """Range rings (vr, azimuth, cos elevation, height) of one sweep or None."""
    name = velocity
    if name is None:
        name = next((n for n in _VELOCITY_NAMES if n in ds), None)
    if name is None or name not in ds:
        return None
    vr = ds[name]
    if "azimuth" not in vr.dims or "range" not in vr.dims:
        return None
    vr = vr.transpose("range", "azimuth")
    if max_range is not None:
        vr = vr.where(vr["range"] <= max_range, drop=True)
        if vr.sizes["range"] == 0:
            return None
    elevation = ds["elevation"]
    el = float(elevation.median()) if elevation.size > 1 else float(elevation)
    if not (min_elevation <= el <= max_elevation):
        return None
    from .vertical_profiles import _beam_height, _site_altitude

    rng = vr["range"].values
    z, _ = _beam_height(rng, el, _site_altitude(ds))
    az = np.broadcast_to(np.radians(vr["azimuth"].values), vr.shape)
    return (
        _f64(vr.values),
        _f64(az),
        np.full(rng.size, np.cos(np.radians(el))),
        np.asarray(z, dtype=np.float64),
    )


def vad_profile(
    obj,
    velocity=None,
    *,
    height_bins=None,
    min_gates=50,
    min_spread=0.1,
    max_rms=None,
    min_elevation=1.0,
    max_elevation=45.0,
    max_range=None,
    engine="auto",
    n_threads=None,
):
    """
    Wind profile from radar radial velocities by the velocity-azimuth display.

    The ring fit is the linear VAD of Browning and Wexler [1]_ (uniform wind
    across the ring, first harmonic only; the paper's equation numbers are
    not checked). The sweep selection and rejection thresholds
    (``min_gates``, ``min_spread``, ``max_rms``, elevation limits) and the
    height binning are radarx choices, not values from that paper.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataTree
        A sweep (``azimuth``, ``range``, ``elevation``, ``altitude``) or a
        volume with ``sweep_*`` groups (xradar layout). The velocities must be
        dealiased (e.g. :func:`radarx.retrieve.dealias_velocity`) and
        no-data codes masked; sweeps without the velocity field are skipped.
    velocity : str, optional
        Velocity variable; default the first of ``VRADH``, ``VRADV``,
        ``VELOCITY``, ``velocity``, ``VEL``.
    height_bins : array-like, optional
        Edges of the height bins (m above sea level) in which ring winds are
        averaged (weighted by their gate counts). Default 100-m bins from the
        radar altitude to 12 km above it.
    min_gates : int, optional
        Fewest valid gates on a ring. Default 50.
    min_spread : float, optional
        Smallest azimuthal coverage of the valid gates, the determinant of the
        covariance of :math:`(\\sin\\alpha, \\cos\\alpha)` (0.25 for a full
        circle, 0 for one direction). Default 0.1.
    max_rms : float, optional
        Reject rings whose r.m.s. fit residual exceeds this (m s-1).
    min_elevation, max_elevation : float, optional
        Elevation angles (degrees) of the sweeps used. Default 1-45.
    max_range : float, optional
        Use only range rings up to this range (m). Far rings of low sweeps
        average the wind over a circle hundreds of kilometres wide; limiting
        the range keeps the profile local to the radar.
    engine : {"auto", "compiled", "numpy"}, optional
        Kernel implementation.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.

    Returns
    -------
    xarray.Dataset
        ``u``, ``v``, ``wind_speed``, ``wind_direction``, ``vad_rms`` (mean
        fit residual) and ``vad_rings`` (rings per bin) on ``height`` (bin
        centres, m above sea level), with the radar ``latitude``,
        ``longitude`` and ``altitude``.

    References
    ----------
    .. [1] Browning, K. A., and R. Wexler, 1968: The determination of
           kinematic properties of a wind field using Doppler radar.
           *J. Appl. Meteor.*, **7** (1), 105-113,
           https://doi.org/10.1175/1520-0450(1968)007<0105:TDOKPO>2.0.CO;2
    """
    if isinstance(obj, xr.Dataset):
        datasets = [obj]
        site = obj
    elif not isinstance(obj, xr.DataTree):
        raise TypeError("vad_profile needs an xarray.Dataset sweep or a DataTree")
    else:
        datasets = [
            obj[name].to_dataset(inherit="all_coords")
            for name in obj.children
            if name.startswith("sweep")
        ]
        site = datasets[0] if datasets else obj.to_dataset()
    rings = [
        r
        for r in (
            _sweep_rings(ds, velocity, min_elevation, max_elevation, max_range)
            for ds in datasets
        )
        if r is not None
    ]
    if not rings:
        raise KeyError("no sweep has a radial velocity field in the elevation range")
    naz = max(r[0].shape[1] for r in rings)

    def pad(a):
        return np.pad(a, ((0, 0), (0, naz - a.shape[1])), constant_values=np.nan)

    vr = _f64(np.concatenate([pad(r[0]) for r in rings]))
    az = _f64(np.concatenate([pad(r[1]) for r in rings]))
    cos_el = _f64(np.concatenate([r[2] for r in rings]))
    z = np.concatenate([r[3] for r in rings])
    res = np.asarray(
        _kernel(engine).vad(
            vr, az, cos_el, int(min_gates), float(min_spread), _threads(n_threads)
        )
    )
    u, v, _, rms, n = res
    good = np.isfinite(u)
    if max_rms is not None:
        good &= rms <= max_rms
    alt = float(site["altitude"].values) if "altitude" in site else 0.0
    if height_bins is None:
        height_bins = np.arange(alt, alt + 12000.0 + 1.0, 100.0)
    edges = np.asarray(height_bins, dtype=np.float64)
    k = np.digitize(z, edges) - 1
    good &= (k >= 0) & (k < edges.size - 1)
    nb = edges.size - 1
    w = np.where(good, n, 0.0)
    sw = np.bincount(k[good], w[good], nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        um = np.bincount(k[good], (w * u)[good], nb) / sw
        vm = np.bincount(k[good], (w * v)[good], nb) / sw
        rm = np.bincount(k[good], (w * rms)[good], nb) / sw
    centres = 0.5 * (edges[1:] + edges[:-1])
    coords = {
        "height": (
            "height",
            centres,
            {
                "standard_name": "altitude",
                "long_name": "Height above sea level",
                "units": "m",
            },
        )
    }
    for key in ("latitude", "longitude", "altitude"):
        if key in site:
            coords[key] = site[key].reset_coords(drop=True)
    uda = xr.DataArray(um, dims="height", coords=coords)
    vda = xr.DataArray(vm, dims="height", coords=coords)
    out = xr.Dataset(coords=coords)
    out["u"] = uda.assign_attrs(
        standard_name="eastward_wind", long_name="Eastward wind (VAD)", units="m s-1"
    )
    out["v"] = vda.assign_attrs(
        standard_name="northward_wind", long_name="Northward wind (VAD)", units="m s-1"
    )
    out["wind_speed"] = np.hypot(uda, vda).assign_attrs(
        standard_name="wind_speed", units="m s-1"
    )
    out["wind_direction"] = (np.degrees(np.arctan2(-uda, -vda)) % 360.0).assign_attrs(
        standard_name="wind_from_direction", units="degree"
    )
    out["vad_rms"] = xr.DataArray(rm, dims="height", coords=coords).assign_attrs(
        long_name="Mean r.m.s. residual of the VAD fits", units="m s-1"
    )
    out["vad_rings"] = xr.DataArray(
        np.bincount(k[good], minlength=nb)[:nb], dims="height", coords=coords
    ).assign_attrs(long_name="Number of range rings averaged", units="1")
    out.attrs.update(source="VAD (Browning and Wexler 1968)")
    if "time" in site.coords or "time" in site:
        t = np.ravel(site["time"].values)
        if t.size and np.issubdtype(t.dtype, np.datetime64):
            out = out.assign_coords(time=t.min())
    return out


# --------------------------------------------------------------------------
# accessors
# --------------------------------------------------------------------------


@accessor_method("dataset", name="bulk_shear")
def _bulk_shear_accessor(self, bottom=0.0, top=6000.0, **kwargs):
    """
    Bulk wind difference of the ``u``, ``v`` profile over a layer.

    See :func:`radarx.retrieve.bulk_shear` for the parameters.

    Returns
    -------
    xarray.Dataset
        Shear components, speed, direction (and line-normal component).
    """
    return bulk_shear(self.xarray_obj, bottom, top, **kwargs)


@accessor_method("dataset", name="bunkers_storm_motion")
def _bunkers_accessor(self, **kwargs):
    """
    Bunkers et al. (2000) storm motion of the ``u``, ``v`` profile.

    See :func:`radarx.retrieve.bunkers_storm_motion` for the parameters.

    Returns
    -------
    xarray.Dataset
        ``u``, ``v``, ``speed`` and ``direction`` of the storm motion.
    """
    return bunkers_storm_motion(self.xarray_obj, **kwargs)


@accessor_method("dataset", name="storm_relative_helicity")
def _srh_accessor(self, storm_motion="right", bottom=0.0, top=3000.0, **kwargs):
    """
    Storm-relative helicity of the ``u``, ``v`` profile.

    See :func:`radarx.retrieve.storm_relative_helicity` for the parameters.

    Returns
    -------
    xarray.DataArray
        Storm-relative helicity (m2 s-2).
    """
    return storm_relative_helicity(self.xarray_obj, storm_motion, bottom, top, **kwargs)


@accessor_method("dataset", "datatree", name="vad_profile")
def _vad_accessor(self, velocity=None, **kwargs):
    """
    Wind profile by the velocity-azimuth display of the sweep or volume.

    See :func:`radarx.retrieve.vad_profile` for the parameters.

    Returns
    -------
    xarray.Dataset
        ``u``, ``v``, speed and direction on ``height``.
    """
    return vad_profile(self.xarray_obj, velocity, **kwargs)
