#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx Gate Gridding
====================

Grid radar gates onto a Cartesian grid with a radius of influence, the
objective analysis way: every output point takes the weighted mean of the gates
within a radius :math:`R` of it, with the weight falling from the point to the
radius.

:func:`grid_cones` interpolates in the polar structure of a sweep and has no
radius to tune. This is the other family, which does not care how the gates were
scanned. A gate is a point in space (its position follows from azimuth,
elevation and range with the effective Earth radius model, as in
:func:`grid_cones`), so the same code grids full volumes, sector scans, RHI
sections and volumes with repeated low tilts (SAILS, MESO-SAILS: every repeat
is a set of gates, and with a storm motion they are moved to the analysis time
before they are averaged).

The weights, with :math:`d` the distance between the gate and the output point
and :math:`d < R`, are

``"cressman"``
    :math:`w = (R^2 - d^2) / (R^2 + d^2)`, the weight of Cressman (1959),
``"barnes"``
    :math:`w = \\exp(-4 d^2 / R^2)`, the Gaussian weight of Barnes (1964),
    :math:`\\exp(-d^2/\\kappa)` with :math:`\\kappa = R^2/4`, cut at the radius,
``"nearest"``
    the value of the closest gate.

The radius is :math:`R = \\max(R_0,\\; a\\,s + b\\,(z - z_0))` with the distance
:math:`s` of the output point from the radar and its height above the radar
:math:`z - z_0`. With ``roi_range_factor = tan(beamwidth)`` the radius follows
the width of the beam. The three parameters are those of the ``dist`` function
of Py-ART (``min_radius``, ``xy_factor``, ``z_factor``) and ``beamwidth`` is its
``dist_beam`` term for the beam, so that the settings carry over from there.

The work is done by a compiled C++ kernel (multithreaded, gates sorted into
buckets); if it is not available an equivalent NumPy and SciPy implementation
is used.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = ["grid_gates", "grid_rhi"]

__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np
import xarray as xr

from .._compat import renamed_arguments
from .._provenance import provenance
from .._registry import accessor_method
from . import cone
from .cone import K_E

try:
    from . import _cone

    HAS_COMPILED_KERNEL = cone.HAS_COMPILED_KERNEL
except ImportError:  # pragma: no cover - depends on the build
    _cone = None
    HAS_COMPILED_KERNEL = False

WEIGHTINGS = {"cressman": 0, "barnes": 1, "nearest": 2}


def _weighting(weighting):
    try:
        return WEIGHTINGS[weighting]
    except (KeyError, TypeError):
        raise ValueError(
            f"weighting must be one of {sorted(WEIGHTINGS)}, not {weighting!r}"
        ) from None


def _radius_parameters(roi, roi_range_factor, roi_height_factor, beamwidth, min_gates):
    """Validated ``(roi, range factor, height factor, min_gates)``."""
    roi = float(roi)
    if not np.isfinite(roi) or roi <= 0:
        raise ValueError(f"roi must be a positive number of metres, not {roi!r}")
    range_factor = float(roi_range_factor)
    height_factor = float(roi_height_factor)
    if range_factor < 0 or height_factor < 0:
        raise ValueError("roi_range_factor and roi_height_factor must not be negative")
    if beamwidth is not None:
        beamwidth = float(beamwidth)
        if not 0.0 < beamwidth < 90.0:
            raise ValueError("beamwidth must be between 0 and 90 degrees")
        range_factor += float(np.tan(np.radians(beamwidth)))
    min_gates = int(min_gates)
    if min_gates < 1:
        raise ValueError("min_gates must be at least 1")
    return roi, range_factor, height_factor, min_gates


def _gate_positions(arrays, shifts, site_altitude, k_e):
    """Positions ``(n, 3)`` and values of the valid gates of all sweeps."""
    from xradar.georeference import antenna_to_cartesian

    kwargs = cone._antenna_kwargs(k_e)
    points, values = [], []
    for (data, azimuth, elevation, rng), (dx, dy) in zip(arrays, shifts):
        gx, gy, gz = antenna_to_cartesian(
            rng[None, :],
            azimuth[:, None],
            elevation[:, None],
            site_altitude=site_altitude,
            **kwargs,
        )
        ok = np.isfinite(data)
        points.append(np.column_stack([gx[ok] + dx, gy[ok] + dy, gz[ok]]))
        values.append(data[ok])
    return np.concatenate(points), np.concatenate(values)


def _map_numpy(
    cx,
    cy,
    z,
    arrays,
    shifts,
    site_altitude,
    k_e,
    weighting,
    roi,
    range_factor,
    height_factor,
    min_gates,
):
    """NumPy and SciPy implementation of the compiled kernel (same results)."""
    from scipy.spatial import cKDTree

    nz, ncol = z.size, cx.size
    out = np.full((nz, ncol), np.nan, dtype=np.float32)
    points, values = _gate_positions(arrays, shifts, site_altitude, k_e)
    if values.size == 0:
        return out
    tree = cKDTree(points)
    s = np.hypot(cx, cy)
    radius = np.maximum(
        roi, range_factor * s[:, None] + height_factor * (z[None, :] - site_altitude)
    )  # (ncol, nz)
    step = max(1, 200_000 // nz)
    for c0 in range(0, ncol, step):
        c1 = min(ncol, c0 + step)
        rad = radius[c0:c1].ravel()
        query = np.column_stack(
            [
                np.repeat(cx[c0:c1], nz),
                np.repeat(cy[c0:c1], nz),
                np.tile(z, c1 - c0),
            ]
        )
        pairs = cKDTree(query).sparse_distance_matrix(
            tree, float(rad.max()), output_type="ndarray"
        )
        i, j, d = pairs["i"], pairs["j"], pairs["v"]
        inside = d**2 < rad[i] ** 2
        i, j, d = i[inside], j[inside], d[inside]
        n = rad.size
        count = np.bincount(i, minlength=n)
        if weighting == "nearest":
            order = np.lexsort((d, i))
            first = np.unique(i[order], return_index=True)
            result = np.full(n, np.nan)
            result[first[0]] = values[j[order][first[1]]]
        else:
            r2, d2 = rad[i] ** 2, d**2
            w = (
                (r2 - d2) / (r2 + d2)
                if weighting == "cressman"
                else np.exp(-4 * d2 / r2)
            )
            den = np.bincount(i, w, minlength=n)
            num = np.bincount(i, w * values[j], minlength=n)
            result = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
        result[count < min_gates] = np.nan
        out[:, c0:c1] = result.reshape(c1 - c0, nz).T.astype(np.float32)
    return out


def _map_compiled(
    cx,
    cy,
    z,
    arrays,
    shifts,
    site_altitude,
    k_e,
    weighting,
    roi,
    range_factor,
    height_factor,
    min_gates,
    n_threads,
):
    data, azimuth, elevation, rng = (list(a) for a in zip(*arrays))
    return _cone.grid_gates(
        cx,
        cy,
        z,
        data,
        azimuth,
        elevation,
        rng,
        site_altitude,
        earth_radius=cone.EARTH_RADIUS,
        k_e=k_e,
        weighting=_weighting(weighting),
        roi=roi,
        roi_range_factor=range_factor,
        roi_height_factor=height_factor,
        min_gates=min_gates,
        n_threads=int(n_threads or 0),
        shift=[float(c) for pair in shifts for c in pair],
    )


def _use_compiled(engine):
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled gate-gridding kernel is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


def _finite(*axes):
    for axis in axes:
        if not np.isfinite(axis).all():
            raise ValueError("the target coordinates must be finite numbers")


def _variables(sweeps, data_vars):
    if data_vars is None:
        return [
            name for name, da in sweeps[0].data_vars.items() if {"range"} < set(da.dims)
        ]
    return [data_vars] if isinstance(data_vars, str) else list(data_vars)


def _map_fields(
    dtree,
    data_vars,
    cx,
    cy,
    z,
    *,
    weighting,
    roi,
    roi_range_factor,
    roi_height_factor,
    beamwidth,
    min_gates,
    k_e,
    motion,
    time,
    n_threads,
    engine,
):
    """Gate-map every field onto the points; returns fields, volume pieces."""
    _weighting(weighting)
    use_compiled = _use_compiled(engine)
    roi, range_factor, height_factor, min_gates = _radius_parameters(
        roi, roi_range_factor, roi_height_factor, beamwidth, min_gates
    )
    k_e = cone._check_k_e(k_e)
    sweeps, site, site_coords = cone._load_volume(dtree)
    names = _variables(sweeps, data_vars)
    volume_time = cone._volume_time(sweeps)
    site_altitude = site.get("altitude", 0.0)
    fields = {}
    for variable in names:
        selected = [
            ds for ds in sweeps if variable in ds and "range" in ds[variable].dims
        ]
        if not selected:
            raise ValueError(f"No sweep contains {variable!r}.")
        arrays = [cone._sweep_arrays(ds, variable) for ds in selected]
        shifts = (
            cone._sweep_shifts(selected, motion, time, volume_time)
            if motion is not None
            else [(0.0, 0.0)] * len(selected)
        )
        options = (
            shifts,
            site_altitude,
            k_e,
            weighting,
            roi,
            range_factor,
            height_factor,
            min_gates,
        )
        if use_compiled:
            field = _map_compiled(cx, cy, z, arrays, *options, n_threads)
        else:
            field = _map_numpy(cx, cy, z, arrays, *options)
        fields[variable] = (field, dict(selected[0][variable].attrs))
    info = dict(
        sweeps=sweeps,
        site=site,
        site_coords=site_coords,
        volume_time=volume_time,
        k_e=k_e,
        roi=roi,
        range_factor=range_factor,
        height_factor=height_factor,
        min_gates=min_gates,
    )
    return fields, info


def _describe(out, info, weighting, motion, time):
    """Attributes common to the results of the gate gridding functions."""
    out.attrs["gridding_method"] = "gates"
    out.attrs["weighting"] = weighting
    out.attrs["roi"] = info["roi"]
    out.attrs["roi_range_factor"] = info["range_factor"]
    out.attrs["roi_height_factor"] = info["height_factor"]
    out.attrs["min_gates"] = info["min_gates"]
    out.attrs["k_e"] = info["k_e"]
    if motion is not None:
        u, v = cone._motion_pair(motion)
        out = out.assign_coords(
            time=info["volume_time"] if time is None else np.datetime64(time, "ns")
        )
        out.attrs["advection"] = (
            f"each sweep moved from its own time to the analysis time with "
            f"u = {u} m/s, v = {v} m/s"
        )
    return out


@renamed_arguments(dtree="obj", beam_width="beamwidth")
@provenance(
    "Radius-of-influence gridding of radar gates with Cressman and Barnes weights, "
    "radarx implementation; radius functions as in Py-ART",
    extra_refs=("doviak-zrnic-1993",),
)
def grid_gates(
    obj,
    data_vars=None,
    x=None,
    y=None,
    z=None,
    *,
    weighting="cressman",
    roi=1000.0,
    roi_range_factor=0.0,
    roi_height_factor=0.0,
    beamwidth=None,
    min_gates=1,
    k_e=K_E,
    motion=None,
    time=None,
    n_threads=None,
    engine="auto",
):
    """
    Grid radar gates onto a Cartesian grid with a radius of influence.

    Every output point is the weighted mean of the gates within the radius of
    influence :math:`R` of it (Cressman 1959 [1]_ or Barnes 1964 [2]_ weights,
    or the nearest gate). All sweeps of the volume take part, whatever the scan
    (PPI, sector, RHI, repeated tilts); see :mod:`radarx.grid.gates`.

    Parameters
    ----------
    obj : xarray.DataTree
        Radar volume with ``sweep_*`` groups and the radar site ``latitude``,
        ``longitude`` and ``altitude``, e.g. from xradar.
    data_vars : str or list of str, optional
        Fields to grid. By default every field present in the lowest sweep on
        ``(azimuth, range)``.
    x, y : array-like
        Target coordinates east and north of the radar, in metres.
    z : array-like
        Target heights above sea level, in metres.
    weighting : {"cressman", "barnes", "nearest"}, optional
        Weight of a gate at the distance :math:`d < R`: Cressman,
        :math:`(R^2-d^2)/(R^2+d^2)` (default); Barnes, :math:`\\exp(-4d^2/R^2)`;
        or the value of the nearest gate.
    roi : float, optional
        Radius of influence in metres, and the smallest one when it grows
        with the distance from the radar. Default 1000.
    roi_range_factor, roi_height_factor : float, optional
        Increase of the radius in metres per metre of horizontal distance of the
        output point from the radar, and per metre of height above the radar
        (Py-ART's ``xy_factor`` and ``z_factor``). Default 0: a constant
        radius.
    beamwidth : float, optional
        Half-power beamwidth in degrees. When given, ``tan(beamwidth)`` is
        added to ``roi_range_factor``: the radius is the width of the beam at
        the distance of the output point (the ``dist_beam`` term of Py-ART), so
        that the weights reach as far as the gates of the beam do, at any range.
        Default ``None``.
    min_gates : int, optional
        Fewest gates within the radius for a value. Default 1.
    k_e : float, optional
        Effective Earth radius factor of the beam, see :func:`grid_cones`.
    motion : tuple of float, optional
        Storm motion ``(u, v)`` in m/s. Every sweep is moved from its mean time
        to the analysis time before the gates are averaged, which matters for
        repeated low tilts minutes apart. See :func:`grid_cones`. Default
        ``None``.
    time : datetime-like, optional
        Analysis time for ``motion``. Default: the mean time of the volume.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled
        kernel and falls back to NumPy and SciPy.

    Returns
    -------
    xarray.Dataset
        Gridded fields on ``(z, y, x)`` as float32, georeferenced like the
        result of :func:`grid_cones`. Points without a gate within the radius
        are NaN.

    Raises
    ------
    ValueError
        If no sweep contains a requested variable, the grid is missing or a
        parameter is out of range.
    ImportError
        If ``engine="compiled"`` and the compiled kernel is not available.

    See Also
    --------
    grid_cones, grid_rhi

    Notes
    -----
    Cressman's (1959) weight and the Gaussian weight of Barnes (1964) are the
    ones of the objective analysis schemes of these papers, with the analysis
    radius :math:`R` as the cut-off. The Gaussian is
    :math:`\\exp(-d^2/\\kappa)` with :math:`\\kappa = R^2/4`, so the weight is
    0.018 at the radius. A radius that grows with the range and height is the
    ``dist`` function of Py-ART [3]_, and ``beamwidth`` its ``dist_beam`` term;
    the formulas were read from the Py-ART source and are implemented here, not
    copied. The distance is Euclidean between the gate and the output point in
    metres, in the frame of the radar (beam height with the effective Earth
    radius model, Doviak and Zrnić 1993, Eqs. 2.28), and the radius is
    evaluated at the output point. The default radius, ``min_gates`` and the
    rule that a point without a gate is empty are radarx choices.

    The mean of the gates in the radius is not the cone interpolation of
    :func:`grid_cones`: it smooths over :math:`R`, does not reproduce the gate
    values, and a velocity gridded this way mixes gates of different beams (see
    the note on velocity in :func:`grid_cones`). A radius that follows the beam
    width, or at most a gate spacing, limits that.

    References
    ----------
    .. [1] Cressman, G. P., 1959: An operational objective analysis system.
       *Mon. Wea. Rev.*, **87**, 367-374,
       https://doi.org/10.1175/1520-0493(1959)087<0367:AOOAS>2.0.CO;2
    .. [2] Barnes, S. L., 1964: A technique for maximizing details in numerical
       weather map analysis. *J. Appl. Meteor.*, **3**, 396-409,
       https://doi.org/10.1175/1520-0450(1964)003<0396:ATFMDI>2.0.CO;2
    .. [3] Helmus, J. J., and S. M. Collis, 2016: The Python ARM Radar Toolkit
       (Py-ART), a library for working with weather radar data in the Python
       programming language. *J. Open Res. Softw.*, **4**, e25,
       https://doi.org/10.5334/jors.119
    .. [4] Doviak, R. J., and D. S. Zrnić, 1993: *Doppler Radar and Weather
       Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
       DOI).

    Examples
    --------
    >>> x = y = np.arange(-100e3, 100e3 + 1, 1000.0)  # doctest: +SKIP
    >>> z = np.arange(500.0, 10e3 + 1, 500.0)  # doctest: +SKIP
    >>> grid = radarx.grid.grid_gates(  # doctest: +SKIP
    ...     dtree, "DBZH", x, y, z, weighting="barnes", roi=1500.0, beamwidth=1.0
    ... )
    """
    if x is None or y is None or z is None:
        raise ValueError("x, y and z target coordinates are required.")
    x, y, z = (np.asarray(a, dtype=np.float64) for a in (x, y, z))
    _finite(x, y, z)
    cx, cy = (np.ascontiguousarray(a.ravel()) for a in np.meshgrid(x, y))
    fields, info = _map_fields(
        obj,
        data_vars,
        cx,
        cy,
        z,
        weighting=weighting,
        roi=roi,
        roi_range_factor=roi_range_factor,
        roi_height_factor=roi_height_factor,
        beamwidth=beamwidth,
        min_gates=min_gates,
        k_e=k_e,
        motion=motion,
        time=time,
        n_threads=n_threads,
        engine=engine,
    )
    gridded = {
        name: (("z", "y", "x"), field.reshape(z.size, y.size, x.size), attrs)
        for name, (field, attrs) in fields.items()
    }
    out = cone._to_dataset(gridded, x, y, z, info["sweeps"], info["site"], obj.attrs)
    out = _describe(out, info, weighting, motion, time)
    return cone._georeferenced(out, info["site_coords"], info["site"])


def _plane_azimuth(dtree, azimuth):
    """The azimuth (degrees) of the plane: given, or the median of the volume."""
    if azimuth is not None:
        return float(azimuth) % 360.0
    sweeps, _, _ = cone._load_volume(dtree)
    angles = np.radians(
        np.concatenate([np.ravel(ds["azimuth"].values) for ds in sweeps])
    )
    mean = np.degrees(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()))
    return float(mean % 360.0)


@renamed_arguments(dtree="obj", beam_width="beamwidth")
@provenance(
    "Radius-of-influence gridding of radar gates with Cressman and Barnes weights, "
    "radarx implementation; radius functions as in Py-ART",
    extra_refs=("doviak-zrnic-1993",),
)
def grid_rhi(
    obj,
    data_vars=None,
    azimuth=None,
    distance=None,
    z=None,
    *,
    weighting="cressman",
    roi=500.0,
    roi_range_factor=0.0,
    roi_height_factor=0.0,
    beamwidth=None,
    min_gates=1,
    k_e=K_E,
    motion=None,
    time=None,
    n_threads=None,
    engine="auto",
):
    """
    Grid radar gates onto a vertical section through the radar.

    The section is the plane of the azimuth ``azimuth`` through the radar; the
    points are ``distance`` metres from the radar along that azimuth at the
    heights ``z``. The gates of an RHI sweep (a fixed azimuth and the elevation
    varying along the ray) fall in the plane, those of other azimuths and of PPI
    sweeps are weighted by their distance from it, so a few RHIs in nearby
    azimuths, or a PPI volume, also give a section. The weights and the radius
    are those of :func:`grid_gates`, to which the parameters not listed here
    refer.

    Parameters
    ----------
    obj : xarray.DataTree
        Radar volume with ``sweep_*`` groups and the radar site.
    data_vars : str or list of str, optional
        Fields to grid. By default every field present in the lowest sweep.
    azimuth : float, optional
        Azimuth of the plane in degrees clockwise from north. Default: the mean
        azimuth of the gates of the volume, which is the azimuth of an RHI.
    distance : array-like
        Horizontal distances from the radar along the plane, in metres.
    z : array-like
        Target heights above sea level, in metres.
    roi : float, optional
        Radius of influence in metres. Default 500, smaller than for
        :func:`grid_gates`, as an RHI is dense in the plane.

    Returns
    -------
    xarray.Dataset
        Gridded fields on ``(z, distance)`` as float32, with the scalar
        coordinates ``azimuth``, the radar site and ``time``.

    See Also
    --------
    grid_gates

    Notes
    -----
    Weights and radius as in :func:`grid_gates` (Cressman 1959 [1]_, Barnes
    1964 [2]_, the radius functions of Py-ART [3]_, beam geometry of Doviak and
    Zrnić 1993 [4]_). Choices of radarx: the plane is vertical and straight (the section follows a
    great circle only to the extent that the Cartesian frame of the radar does)
    and the default azimuth and radius.

    References
    ----------
    .. [1] Cressman, G. P., 1959: An operational objective analysis system.
       *Mon. Wea. Rev.*, **87**, 367-374,
       https://doi.org/10.1175/1520-0493(1959)087<0367:AOOAS>2.0.CO;2
    .. [2] Barnes, S. L., 1964: A technique for maximizing details in numerical
       weather map analysis. *J. Appl. Meteor.*, **3**, 396-409,
       https://doi.org/10.1175/1520-0450(1964)003<0396:ATFMDI>2.0.CO;2
    .. [3] Helmus, J. J., and S. M. Collis, 2016: The Python ARM Radar Toolkit
       (Py-ART), a library for working with weather radar data in the Python
       programming language. *J. Open Res. Softw.*, **4**, e25,
       https://doi.org/10.5334/jors.119
    .. [4] Doviak, R. J., and D. S. Zrnić, 1993: *Doppler Radar and Weather
       Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
       DOI).

    Examples
    --------
    >>> d = np.arange(1000.0, 60e3, 500.0)  # doctest: +SKIP
    >>> z = np.arange(0.0, 15e3, 250.0)  # doctest: +SKIP
    >>> section = radarx.grid.grid_rhi(  # doctest: +SKIP
    ...     rhi, "DBZH", azimuth=270.0, distance=d, z=z
    ... )
    """
    if distance is None or z is None:
        raise ValueError("distance and z target coordinates are required.")
    distance, z = (np.asarray(a, dtype=np.float64) for a in (distance, z))
    _finite(distance, z)
    phi = _plane_azimuth(obj, azimuth)
    cx = np.ascontiguousarray(distance * np.sin(np.radians(phi)))
    cy = np.ascontiguousarray(distance * np.cos(np.radians(phi)))
    # the radius grows with the distance of the point from the radar, the plane
    # distance in this frame
    fields, info = _map_fields(
        obj,
        data_vars,
        cx,
        cy,
        z,
        weighting=weighting,
        roi=roi,
        roi_range_factor=roi_range_factor,
        roi_height_factor=roi_height_factor,
        beamwidth=beamwidth,
        min_gates=min_gates,
        k_e=k_e,
        motion=motion,
        time=time,
        n_threads=n_threads,
        engine=engine,
    )
    out = xr.Dataset(
        {
            name: (("z", "distance"), field, attrs)
            for name, (field, attrs) in fields.items()
        },
        coords={
            "z": ("z", z, {"units": "m", "long_name": "height above sea level"}),
            "distance": (
                "distance",
                distance,
                {"units": "m", "long_name": "horizontal distance from the radar"},
            ),
            "azimuth": (
                (),
                phi,
                {"units": "degree", "long_name": "azimuth of the section"},
            ),
        },
    )
    times = [np.ravel(ds["time"].values) for ds in info["sweeps"] if "time" in ds]
    if times:
        out = out.assign_coords(
            time=xr.DataArray(np.concatenate(times).astype("datetime64[ns]")).mean()
        )
    out.attrs = dict(obj.attrs)
    out.attrs["radar_name"] = out.attrs.get("instrument_name", "")
    out = _describe(out, info, weighting, motion, time)
    return out.assign_coords(info["site_coords"])


@accessor_method("datatree", name="grid_gates", wraps=grid_gates)
def _grid_gates_accessor(self):
    """
    Grid the radar gates with a radius of influence.

    Parameters
    ----------
    data_vars, x, y, z, **kwargs
        See :func:`radarx.grid.grid_gates`.

    Returns
    -------
    xarray.Dataset

    See Also
    --------
    radarx.grid.grid_gates
    """


@accessor_method("datatree", name="grid_rhi", wraps=grid_rhi)
def _grid_rhi_accessor(self):
    """
    Grid the radar gates onto a vertical section through the radar.

    Parameters
    ----------
    data_vars, azimuth, distance, z, **kwargs
        See :func:`radarx.grid.grid_rhi`.

    Returns
    -------
    xarray.Dataset

    See Also
    --------
    radarx.grid.grid_rhi
    """
