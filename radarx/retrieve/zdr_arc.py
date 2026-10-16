#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx ZDR Arc and KDP-ZDR Separation
=====================================

The ZDR arc and the KDP-ZDR separation of a supercell, from the lowest level of a
polarimetric radar volume on a grid, after Wilson and Van Den Broeke (2021).

Precipitation that falls through the forward flank of a supercell is sorted by the
storm-relative wind: the big drops fall out near the inflow edge, where they form an
arc of high differential reflectivity (the ZDR arc), and the small drops are carried
farther into the storm, below the updraft, where the specific differential phase is
enhanced (the KDP foot). The distance between the two and its orientation to the
storm motion indicate the low-level shear and helicity (Loeffler and Kumjian 2018;
Loeffler et al. 2020; Wilson and Van Den Broeke 2021).

Algorithm
---------
:func:`zdr_arc` follows the object-based algorithm of the paper, for one grid:

1. The reflectivity gradient is computed at the grid points above 20 dBZ. Its
   direction, which points to higher reflectivity, is compared with the direction of
   the forward-flank downdraft (FFD) vector that the user gives, a vector
   perpendicular to the forward flank that points into the storm. ZDR is masked where
   the two differ by more than 120 degrees (the grid point is not on the inflow
   side) and where the correlation coefficient is below 0.6.
2. The masked ZDR is contoured at 3.25 dB (the arc is defined as ZDR above 3.5 dB; the
   threshold is lowered for the smoothing of the gridding), and every closed region
   is a candidate arc object.
3. Storm objects are the centroids of the regions of at least 45 dBZ that cover more than
   20 km2; inside a region of more than 300 km2 the 50 dBZ regions are used. A candidate
   belongs to the closest storm object within 30 km.
4. The candidates that a classifier accepts are combined into the arc of their storm.
5. KDP is masked below 35 dBZ and contoured at 1.5 degrees km-1; the regions within 15 km
   of a storm form its KDP foot. The separation vector points from the centroid of the
   foot to that of the arc, and the separation angle is its angle to the storm motion.

The result per storm is the arc area, the mean ZDR of the arc, the mean of its 10
highest ZDR values, the length of the separation vector and the separation angle
(Table 2 of the paper).

What is not in this function:

- The paper removes the candidates that are not part of the arc (the rear flank,
  the north side, small storms) with a random forest of 100 trees that was trained
  on 4724 candidates labelled by hand. It cannot be rebuilt from the paper, so
  ``classifier`` takes a model of your own, or a function, that decides from the
  eleven features of Table 1 (``FEATURES``); without it every candidate with a storm
  is part of the arc, which overestimates the arc where the rear flank has high ZDR.
- The tracking of the storm objects from scan to scan, and the program with
  its window for the user. Call the function for every volume.

Choices that the paper leaves open (radarx's):

- Directions are bearings in degrees clockwise from north that the vector points
  to, for the FFD vector, the storm motion and the gradient. The grid has ``x``
  to the east and ``y`` to the north, in m.
- ZDR is masked where the reflectivity is not above 20 dBZ, where there is no gradient
  direction. (Section 3b of the paper says that the algorithm removes these pixels.)
- Regions are 8-connected sets of grid cells, and areas are cell counts times the cell
  area, where the paper contours the field.
- The paper smooths the reflectivity for the storm objects without giving the
  width: ``smooth`` (Gaussian, in grid cells) is off by default.
- The mean direction of the gradient of a candidate is the circular mean of the signed
  angles to the FFD vector.
- In a region of more than 300 km2 without a 50 dBZ region of the minimum area, the
  centroid of the 45 dBZ region is the storm object.
- The separation angle is the angle, from 0 to 180 degrees, between the separation
  vector and the storm motion.

Fields to prepare first
-----------------------
The paper grids the lowest tilt of the WSR-88D with 493 m spacing and a Barnes analysis whose
radius grows with range (:func:`radarx.grid.grid_cones`), and calibrates ZDR
with the dry snow aggregates above the freezing level
(:func:`radarx.retrieve.zdr_offset`; its mean offset is 0.3 dB, and the
3.25 dB threshold depends on it). The algorithm is for supercells: the leading edge
of a squall line and high ZDR under developing updrafts are often detected as arcs.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = ["zdr_arc"]

__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

import numpy as np
import xarray as xr

from .._provenance import provenance
from .._registry import accessor_method
from .._sweepwise import dask_support
from ._fieldtools import DBZ_NAMES, KDP_NAMES, RHOHV_NAMES, ZDR_NAMES, find_field

#: Features of a candidate in the order of Table 1 of Wilson and Van Den Broeke (2021);
#: the array that ``classifier`` receives has one column per name.
FEATURES = (
    "area",
    "storm_distance",
    "mean_zdr",
    "max_zdr",
    "mean_rhohv",
    "mean_kdp",
    "mean_dbz",
    "mean_gradient_direction",
    "mean_gradient",
    "angle_to_ffd",
    "storm_vector_x",
    "storm_vector_y",
)

_VERTICAL_NAMES = ("z", "altitude", "height", "level")
_EIGHT = np.ones((3, 3), dtype=int)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _wrap(angle):
    """Angle in degrees wrapped to (-180, 180]."""
    return 180.0 - np.mod(180.0 - np.asarray(angle, dtype=float), 360.0)


def _bearing(dx, dy):
    """Bearing in degrees clockwise from north of the vector (dx east, dy north)."""
    return np.mod(np.degrees(np.arctan2(dx, dy)), 360.0)


def _label(mask):
    from scipy import ndimage

    return ndimage.label(mask, structure=_EIGHT)


def _sums(labels, n, values):
    """Sum of ``values`` over each label 1..n (zeros for an empty set)."""
    return np.bincount(labels.ravel(), weights=values.ravel(), minlength=n + 1)[1:]


def _centroids(labels, n, xx, yy):
    count = np.maximum(_sums(labels, n, np.ones(labels.shape)), 1.0)
    return _sums(labels, n, xx) / count, _sums(labels, n, yy) / count


def _surface(obj, level):
    """The 2-D dataset of the grid at the lowest level (or the nearest to ``level``)."""
    if isinstance(obj, xr.DataArray):
        raise TypeError("obj must be an xarray.Dataset holding the fields")
    for name in _VERTICAL_NAMES:
        if name in obj.dims:
            values = obj[name].values
            k = int(
                np.argmin(values) if level is None else np.argmin(abs(values - level))
            )
            obj = obj.isel({name: k})
            break
    extra = [d for d in obj.dims if d not in ("y", "x")]
    if extra or not {"y", "x"} <= set(obj.dims):
        raise ValueError(
            f"the fields need the dimensions y and x only (found {tuple(obj.dims)}): "
            "select one time or call the function for every volume"
        )
    return obj


def _array(ds, name, candidates, what, required=True):
    chosen = find_field(ds, name, candidates, what, required=required)
    if chosen is None:
        return None
    return np.asarray(ds[chosen].transpose("y", "x").values, dtype=float)


def _spacing(ds):
    out = []
    for d in ("x", "y"):
        if d not in ds.coords or ds[d].size < 2:
            raise ValueError(f"the grid needs a coordinate {d!r} (m)")
        out.append(float(np.median(np.abs(np.diff(ds[d].values)))))
    return out


def _storm_objects(
    dbz, xx, yy, cell_km2, storm_dbz, core_dbz, min_area, max_area, smooth
):
    """Centroids (m) and areas (km2) of the storm objects."""
    field = dbz
    if smooth:
        from scipy import ndimage

        filled = np.where(
            np.isfinite(dbz), dbz, np.nanmin(dbz) if np.isfinite(dbz).any() else 0.0
        )
        field = ndimage.gaussian_filter(filled, float(smooth))
        field = np.where(np.isfinite(dbz), field, np.nan)
    with np.errstate(invalid="ignore"):
        labels, n = _label(field >= storm_dbz)
    if n == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    area = _sums(labels, n, np.ones(labels.shape)) * cell_km2
    cx, cy = _centroids(labels, n, xx, yy)
    out_x, out_y, out_a = [], [], []
    for i in range(n):
        if area[i] <= min_area:
            continue
        if area[i] > max_area:
            with np.errstate(invalid="ignore"):
                inner, m = _label((labels == i + 1) & (field >= core_dbz))
            if m:
                a = _sums(inner, m, np.ones(inner.shape)) * cell_km2
                px, py = _centroids(inner, m, xx, yy)
                keep = a >= min_area
                if keep.any():
                    out_x += list(px[keep])
                    out_y += list(py[keep])
                    out_a += list(a[keep])
                    continue
        out_x.append(cx[i])
        out_y.append(cy[i])
        out_a.append(area[i])
    return np.asarray(out_x), np.asarray(out_y), np.asarray(out_a)


def _closest(px, py, sx, sy):
    """Index of the closest storm object and the distance (m) for every point."""
    if sx.size == 0:
        return np.full(px.shape, -1), np.full(px.shape, np.inf)
    d = np.hypot(px[:, None] - sx[None, :], py[:, None] - sy[None, :])
    k = np.argmin(d, axis=1)
    return k, d[np.arange(px.size), k]


def _selected(classifier, features):
    if classifier is None:
        return np.ones(features.shape[0], dtype=bool)
    if features.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    predict = getattr(classifier, "predict", classifier)
    out = np.asarray(predict(features))
    if out.shape != (features.shape[0],):
        raise ValueError("classifier must return one value per candidate")
    return out.astype(bool)


# --------------------------------------------------------------------------
# the algorithm
# --------------------------------------------------------------------------


@dask_support(
    "computes",
    "the objects (regions, storms) are found from the whole grid of one volume.",
)
@provenance(
    "ZDR arc and KDP-ZDR separation algorithm of Wilson and Van Den Broeke (2021)"
)
def zdr_arc(
    obj,
    zdr=None,
    dbz=None,
    rhohv=None,
    kdp=None,
    *,
    ffd_direction,
    storm_motion=None,
    level=None,
    zdr_threshold=3.25,
    kdp_threshold=1.5,
    min_rhohv=0.6,
    min_dbz=20.0,
    max_gradient_angle=120.0,
    storm_dbz=45.0,
    core_dbz=50.0,
    min_storm_area=20.0,
    max_storm_area=300.0,
    max_distance=30.0,
    foot_dbz=35.0,
    foot_distance=15.0,
    smooth=None,
    classifier=None,
):
    """
    ZDR arc and KDP-ZDR separation of the supercells of a grid.

    See :mod:`radarx.retrieve.zdr_arc` for the algorithm of Wilson and Van Den Broeke
    (2021), what the function leaves out and the choices that are radarx's.

    Parameters
    ----------
    obj : xarray.Dataset
        Fields on a grid with the coordinates ``x`` (east) and ``y`` (north) in m. A
        vertical dimension (``z``, ``altitude``, ``height`` or ``level``) is reduced to
        its lowest level, or the one nearest to ``level``.
    zdr, dbz, rhohv, kdp : str, optional
        Names of the differential reflectivity (dB), the reflectivity (dBZ), the
        copolar correlation coefficient and the specific differential phase
        (degrees km-1). Default: the first usual name of each. ``rhohv`` and
        ``kdp`` are optional: without ``rhohv`` the correlation coefficient is not
        used, and without ``kdp`` there is no KDP foot and no separation.
    ffd_direction : float
        Direction of the forward-flank downdraft vector, perpendicular to the
        forward flank and pointing into the storm, as a bearing in degrees
        (clockwise from north).
    storm_motion : float, optional
        Direction the storm moves to (bearing, degrees), for the separation angle.
    level : float, optional
        Height (m) of the grid level to use. Default: the lowest.
    zdr_threshold : float, optional
        ZDR (dB) of the arc regions. Default 3.25.
    kdp_threshold : float, optional
        KDP (degrees km-1) of the KDP foot regions. Default 1.5.
    min_rhohv : float, optional
        Correlation coefficient below which ZDR is masked. Default 0.6.
    min_dbz : float, optional
        Reflectivity (dBZ) above which the gradient is computed and ZDR counts.
        Default 20.
    max_gradient_angle : float, optional
        Largest angle (degrees) between the reflectivity gradient and the FFD vector
        for a grid point on the inflow side. Default 120.
    storm_dbz, core_dbz : float, optional
        Reflectivity (dBZ) of the storm objects and of the cores inside large
        ones. Default 45 and 50.
    min_storm_area, max_storm_area : float, optional
        A storm object needs more than ``min_storm_area`` km2; a region above
        ``max_storm_area`` km2 is split into its cores. Default 20 and 300.
    max_distance : float, optional
        Largest distance (km) from a candidate to its storm object. Default 30.
    foot_dbz, foot_distance : float, optional
        Reflectivity (dBZ) below which KDP is masked and the largest distance
        (km) from a KDP region to its storm object. Default 35 and 15.
    smooth : float, optional
        Width (grid cells) of a Gaussian filter of the reflectivity for the storm
        objects. Default: none.
    classifier : callable or object with ``predict``, optional
        Decides which candidates belong to the arc: it receives an array of the
        ``FEATURES`` (one row per candidate) and returns a boolean per candidate.
        Default: every candidate that has a storm object.

    Returns
    -------
    xarray.Dataset
        Dimension ``candidate``: ``candidate_area`` (km2), ``candidate_x`` and
        ``candidate_y`` (m), the other features of Table 1, ``candidate_storm`` (index
        of the storm, -1 for none) and ``candidate_arc`` (bool). Dimension ``storm``:
        ``storm_x``, ``storm_y`` (m), ``storm_area`` (km2), ``arc_area`` (km2),
        ``arc_mean_zdr``, ``arc_top10_zdr`` (dB), ``arc_x``, ``arc_y`` (m),
        ``foot_area`` (km2), ``foot_x``, ``foot_y`` (m), ``separation_length``
        (km) and ``separation_angle`` (degrees). On the grid: ``ZDR_ARC_CANDIDATE``
        (the region number, 0 for none), ``ZDR_ARC_STORM`` and ``KDP_FOOT_STORM`` (the
        storm index of the arc and foot cells, -1 elsewhere).

    Raises
    ------
    KeyError
        If a required field is not found.
    ValueError
        If the fields have other dimensions than ``y`` and ``x`` (after reducing the
        vertical one), or ``classifier`` does not return one value per candidate.

    References
    ----------
    Wilson, M. B., and M. S. Van Den Broeke, 2021: An automated Python algorithm to
    quantify ZDR arc and KDP-ZDR separation signatures in supercells. *J. Atmos.
    Oceanic Technol.*, **38**, 371-386, https://doi.org/10.1175/JTECH-D-20-0056.1

    Examples
    --------
    >>> res = radarx.retrieve.zdr_arc(grid, ffd_direction=20.0, storm_motion=60.0)  # doctest: +SKIP
    >>> res.separation_angle  # doctest: +SKIP
    """
    ds = _surface(obj, level)
    zdr_a = _array(ds, zdr, ZDR_NAMES, "differential reflectivity")
    dbz_a = _array(ds, dbz, DBZ_NAMES, "reflectivity")
    rho_a = (
        None
        if rhohv is None
        else _array(ds, rhohv, RHOHV_NAMES, "correlation coefficient")
    )
    kdp_a = (
        None
        if kdp is None
        else _array(ds, kdp, KDP_NAMES, "specific differential phase")
    )
    dx, dy = _spacing(ds)
    xx, yy = np.meshgrid(ds["x"].values.astype(float), ds["y"].values.astype(float))
    cell = dx * dy / 1e6  # km2

    # 1. inflow side: the direction of the reflectivity gradient against the FFD vector
    with np.errstate(invalid="ignore"):
        gy, gx = np.gradient(dbz_a, dy / 1e3, dx / 1e3)  # dBZ km-1
        magnitude = np.hypot(gx, gy)
        has_gradient = (dbz_a > min_dbz) & np.isfinite(magnitude) & (magnitude > 0.0)
        direction = _bearing(gx, gy)
        relative = _wrap(direction - ffd_direction)
        inflow = has_gradient & (np.abs(relative) <= max_gradient_angle)
        keep = inflow
        if rho_a is not None:
            keep = keep & (rho_a >= min_rhohv)
        zdr_masked = np.where(keep & np.isfinite(zdr_a), zdr_a, np.nan)

        # 2. candidate arc objects
        labels, n = _label(zdr_masked >= zdr_threshold)
    cand_x, cand_y = _centroids(labels, n, xx, yy)
    cand_area = _sums(labels, n, np.ones(labels.shape)) * cell
    count = np.maximum(_sums(labels, n, np.ones(labels.shape)), 1.0)

    def mean_of(values):
        if values is None:
            return np.full(n, np.nan)
        finite = np.isfinite(values)
        total = _sums(labels, n, np.where(finite, values, 0.0))
        seen = _sums(labels, n, finite.astype(float))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(seen > 0, total / seen, np.nan)

    mean_zdr = mean_of(zdr_masked)
    mean_dbz = mean_of(dbz_a)
    mean_rho = mean_of(rho_a)
    mean_kdp = mean_of(kdp_a)
    mean_grad = mean_of(np.where(inflow, magnitude, np.nan))
    sin = _sums(labels, n, np.where(inflow, np.sin(np.radians(relative)), 0.0))
    cos = _sums(labels, n, np.where(inflow, np.cos(np.radians(relative)), 0.0))
    mean_dir = np.degrees(np.arctan2(sin / count, cos / count))
    max_zdr = np.full(n, np.nan)
    if n:
        from scipy import ndimage

        max_zdr = np.asarray(
            ndimage.maximum(
                np.where(np.isfinite(zdr_masked), zdr_masked, -np.inf),
                labels,
                np.arange(1, n + 1),
            ),
            dtype=float,
        )

    # 3. storm objects and the association
    sx, sy, sarea = _storm_objects(
        dbz_a, xx, yy, cell, storm_dbz, core_dbz, min_storm_area, max_storm_area, smooth
    )
    storm_of, distance = _closest(cand_x, cand_y, sx, sy)
    near = distance <= max_distance * 1e3
    storm_of = np.where(near, storm_of, -1)
    distance_km = np.where(near, distance / 1e3, np.nan)
    vx = (
        np.where(
            near, cand_x - sx[np.maximum(storm_of, 0)] if sx.size else np.nan, np.nan
        )
        / 1e3
    )
    vy = (
        np.where(
            near, cand_y - sy[np.maximum(storm_of, 0)] if sy.size else np.nan, np.nan
        )
        / 1e3
    )
    angle_ffd = np.where(
        near,
        np.abs(_wrap(_bearing(np.nan_to_num(vx), np.nan_to_num(vy)) - ffd_direction)),
        np.nan,
    )

    # 4. the candidates of the arc
    features = (
        np.column_stack(
            [
                cand_area,
                distance_km,
                mean_zdr,
                max_zdr,
                mean_rho,
                mean_kdp,
                mean_dbz,
                mean_dir,
                mean_grad,
                angle_ffd,
                vx,
                vy,
            ]
        )
        if n
        else np.empty((0, len(FEATURES)))
    )
    chosen = _selected(classifier, features) & (storm_of >= 0)

    ns = sx.size
    arc_storm = np.full(labels.shape, -1)
    for i in np.flatnonzero(chosen):
        arc_storm[labels == i + 1] = storm_of[i]
    out = {
        k: np.full(ns, np.nan)
        for k in (
            "arc_area",
            "arc_mean_zdr",
            "arc_top10_zdr",
            "arc_x",
            "arc_y",
            "foot_area",
            "foot_x",
            "foot_y",
            "separation_length",
            "separation_angle",
        )
    }
    for s in range(ns):
        cells = arc_storm == s
        if not cells.any():
            continue
        values = zdr_masked[cells]
        out["arc_area"][s] = cells.sum() * cell
        out["arc_mean_zdr"][s] = values.mean()
        out["arc_top10_zdr"][s] = np.sort(values)[::-1][:10].mean()
        out["arc_x"][s], out["arc_y"][s] = xx[cells].mean(), yy[cells].mean()

    # 5. KDP foot and the separation vector
    foot_storm = np.full(labels.shape, -1)
    if kdp_a is not None and ns:
        with np.errstate(invalid="ignore"):
            flabels, fn = _label((kdp_a >= kdp_threshold) & (dbz_a >= foot_dbz))
        if fn:
            fx, fy = _centroids(flabels, fn, xx, yy)
            owner, fdist = _closest(fx, fy, sx, sy)
            for j in range(fn):
                if fdist[j] <= foot_distance * 1e3:
                    foot_storm[flabels == j + 1] = owner[j]
        for s in range(ns):
            cells = foot_storm == s
            if not cells.any():
                continue
            out["foot_area"][s] = cells.sum() * cell
            out["foot_x"][s], out["foot_y"][s] = xx[cells].mean(), yy[cells].mean()
        both = np.isfinite(out["arc_x"]) & np.isfinite(out["foot_x"])
        sx_vec = (out["arc_x"] - out["foot_x"]) / 1e3
        sy_vec = (out["arc_y"] - out["foot_y"]) / 1e3
        out["separation_length"][both] = np.hypot(sx_vec, sy_vec)[both]
        if storm_motion is not None:
            angle = np.abs(_wrap(float(storm_motion) - _bearing(sx_vec, sy_vec)))
            out["separation_angle"][both] = angle[both]

    ds_out = xr.Dataset(
        {
            "candidate_area": ("candidate", cand_area),
            "candidate_x": ("candidate", cand_x),
            "candidate_y": ("candidate", cand_y),
            "candidate_storm_distance": ("candidate", distance_km),
            "candidate_mean_zdr": ("candidate", mean_zdr),
            "candidate_max_zdr": ("candidate", max_zdr),
            "candidate_mean_rhohv": ("candidate", mean_rho),
            "candidate_mean_kdp": ("candidate", mean_kdp),
            "candidate_mean_dbz": ("candidate", mean_dbz),
            "candidate_mean_gradient_direction": ("candidate", mean_dir),
            "candidate_mean_gradient": ("candidate", mean_grad),
            "candidate_angle_to_ffd": ("candidate", angle_ffd),
            "candidate_storm_vector_x": ("candidate", vx),
            "candidate_storm_vector_y": ("candidate", vy),
            "candidate_storm": ("candidate", storm_of.astype(int)),
            "candidate_arc": ("candidate", chosen),
            "storm_x": ("storm", sx),
            "storm_y": ("storm", sy),
            "storm_area": ("storm", sarea),
            **{name: ("storm", values) for name, values in out.items()},
            "ZDR_ARC_CANDIDATE": (("y", "x"), labels.astype(int)),
            "ZDR_ARC_STORM": (("y", "x"), arc_storm.astype(int)),
            "KDP_FOOT_STORM": (("y", "x"), foot_storm.astype(int)),
        },
        coords={"y": ds["y"].values, "x": ds["x"].values},
    )
    units = {
        "candidate_area": "km2",
        "candidate_x": "m",
        "candidate_y": "m",
        "candidate_storm_distance": "km",
        "candidate_mean_zdr": "dB",
        "candidate_max_zdr": "dB",
        "candidate_mean_kdp": "degree km-1",
        "candidate_mean_dbz": "dBZ",
        "candidate_mean_gradient_direction": "degree",
        "candidate_mean_gradient": "dBZ km-1",
        "candidate_angle_to_ffd": "degree",
        "candidate_storm_vector_x": "km",
        "candidate_storm_vector_y": "km",
        "storm_x": "m",
        "storm_y": "m",
        "storm_area": "km2",
        "arc_area": "km2",
        "arc_mean_zdr": "dB",
        "arc_top10_zdr": "dB",
        "arc_x": "m",
        "arc_y": "m",
        "foot_area": "km2",
        "foot_x": "m",
        "foot_y": "m",
        "separation_length": "km",
        "separation_angle": "degree",
    }
    for name, unit in units.items():
        ds_out[name].attrs["units"] = unit
    ds_out["separation_angle"].attrs[
        "long_name"
    ] = "Angle between the KDP-ZDR separation vector and the storm motion"
    ds_out["separation_length"].attrs[
        "long_name"
    ] = "Length of the KDP-ZDR separation vector"
    ds_out["arc_area"].attrs["long_name"] = "Area of the ZDR arc"
    ds_out.attrs["ffd_direction"] = float(ffd_direction)
    return ds_out


@accessor_method("dataset", name="zdr_arc", wraps=zdr_arc)
def _zdr_arc_accessor(self):
    """
    ZDR arc and KDP-ZDR separation of the supercells of a grid.

    See :func:`radarx.retrieve.zdr_arc` for the parameters.

    Returns
    -------
    xarray.Dataset
        The candidates, the storms and the maps of the objects.
    """
