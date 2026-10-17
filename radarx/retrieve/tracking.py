#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx Storm Cell Identification and Tracking
=============================================

Storm cells in a sequence of radar grids: where they are, how they move and how they
change, with the statistics of any other field inside every cell.

Algorithm
---------
:func:`identify_cells` finds the cells of every volume in two steps, as tobac does
(Heikenfeld et al. 2019, sections 2.2 and 2.3):

1. Feature detection with several thresholds. The field (for example the composite
   reflectivity) is smoothed if asked, and the regions above each threshold, in
   ascending order, are found. A region of a threshold that holds regions of the next
   higher threshold is replaced by them, so that the features are the regions at the
   highest threshold they reach, and a weak storm is found with its own, lower
   threshold. Regions with fewer than ``min_area`` are dropped at every threshold.
2. Segmentation. The features are the markers of a watershed segmentation of the field
   that floods down to the lowest threshold, so that each feature gets the cell around
   it, and two cells that touch are divided along the saddle between them.

:func:`track_cells` links the cells of consecutive volumes by their overlap, as
PyFLEXTRKR does (Feng et al. 2023, section 2): the cells of the earlier volume are
moved by the storm motion, and a pair of cells is linked where the overlap area is at
least a fraction of the area of one of the two cells (the fraction is taken in both
directions, so that a growing and a shrinking cell are both found). The links are
resolved in this way:

- The links are taken in the order of their overlap area, each cell once: the largest
  overlap continues the track.
- A cell of the later volume that is linked only to a parent that continues
  elsewhere starts a new track that is marked as split from that parent.
- A cell of the earlier volume that is linked only to a cell that another track
  continues ends, and the track is marked as merged into that cell's track.
- Cells without a link start or end a track. Cells that have no overlap at all can be
  linked by distance (``max_speed``) between the position that the motion predicts
  and the later cell, as in the linking of tobac (Heikenfeld et al. 2019, section 2.4).

The storm motion comes from the user (a vector, a dataset such as the one from
:func:`radarx.retrieve.estimate_motion`, or one value per time); without it the cells are
linked where they overlap in place, which needs a time step that is short against the
size of the cells.

:func:`cell_statistics` gives the time series of the statistics (mean, maximum, ...) of
fields on the grid inside every cell: the rotation from :func:`radarx.retrieve.azimuthal_shear`,
the lightning density, the ZDR and KDP columns, the DSD parameters or any other
product on the grid.

What the papers leave open, and what radarx does:

- The thresholds, the smoothing and the sizes are the user's, as in tobac. The markers of
  the watershed are the feature regions themselves, not the single point of the weighted
  mean position that tobac uses, which keeps a marker inside its cell when the region
  is not convex.
- The cell masks are not dilated before the overlap is counted (PyFLEXTRKR dilates them
  for slow volume scans); use the motion for that.
- The fraction of overlap is one of the cells' own areas, not of the union.
- A cell continues the track of the parent with the largest overlap. Splits and merges
  are not decided by the minimum spanning tree of tobac 1.5 (Sokolowsky et al. 2024), which
  needs the whole tracking window, but by the links of two consecutive volumes.
- The position of a cell is the centre of its grid points. Weighted positions can be
  made from the statistics.

Fields to prepare first
-----------------------
Use grids with a regular spacing and the same grid in every volume (for example
:func:`radarx.grid.grid_cones` at 1 km), a time coordinate, and a field in which the
cells stand out, such as the composite reflectivity. Pass the storm motion of
:func:`radarx.retrieve.estimate_motion` when the cells move by more than their size in a
volume interval. A missing value in the field counts as no echo.

Implementation
--------------
Compiled, multithreaded C++ kernels do the labelling, the nested regions, the
watershed, the sums over the cells and the overlap counts of all volumes in one call,
with equivalent NumPy implementations as fallback and test oracle; the links are
resolved in Python. The functions work on :class:`xarray.DataArray` and
:class:`xarray.Dataset` grids with extra leading dimensions and never change their
input.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = ["cell_statistics", "identify_cells", "track_cells"]

__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

import numpy as np
import xarray as xr

from .._provenance import provenance
from .._registry import accessor_method
from .._sweepwise import dask_support
from . import _cells
from ._fieldtools import DBZ_NAMES, choose_engine, detach, find_field

_VERTICAL_NAMES = ("z", "altitude", "height", "level")
_STATS = ("mean", "max", "min", "sum", "count")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _vertical(da):
    for name in _VERTICAL_NAMES:
        if name in da.dims:
            return name
    return None


def _check_horizontal(da, what):
    for d in ("y", "x"):
        if d not in da.dims:
            raise ValueError(
                f"{what} needs the horizontal dimensions y and x, not {da.dims}"
            )


def _spacing(da):
    """Signed grid steps of x and y in m."""
    steps = []
    for d in ("x", "y"):
        if d not in da.coords or da[d].size < 2:
            raise ValueError(
                f"the grid needs a coordinate {d!r} (m) with two values or more"
            )
        steps.append(float(np.median(np.diff(da[d].values))))
    return steps


def _frames(values, lead):
    """The block (..., ny, nx) as frames (n, ny, nx)."""
    return values.reshape((-1,) + values.shape[-2:])


# --------------------------------------------------------------------------
# identification
# --------------------------------------------------------------------------


def _identify_block(
    field,
    thresholds,
    min_size,
    segmentation,
    smooth,
    connectivity,
    use_compiled,
    n_threads,
):
    lead = field.shape[:-2]
    frames = np.ascontiguousarray(_frames(np.asarray(field, dtype=np.float64), lead))
    if smooth:
        filled = np.where(np.isfinite(frames), frames, thresholds[0] - 1.0)
        frames = _cells.smooth(filled, smooth, use_compiled, n_threads)
    markers, _, _ = _cells.features(
        frames, thresholds, min_size, connectivity, use_compiled, n_threads
    )
    with np.errstate(invalid="ignore"):
        mask = np.isfinite(frames) & (frames >= segmentation)
    labels = _cells.watershed(
        frames, markers, mask, connectivity, use_compiled, n_threads
    )
    return labels.reshape(lead + labels.shape[-2:])


@dask_support("lazy")
@provenance(
    "Storm cell identification by multi-threshold features and watershed after tobac"
)
def identify_cells(
    obj,
    field=None,
    *,
    thresholds,
    min_area=10.0,
    segmentation_threshold=None,
    smooth=None,
    connectivity=8,
    n_threads=None,
    engine="auto",
):
    """
    Storm cells of every volume of a grid: multi-threshold features and watershed.

    See :mod:`radarx.retrieve.tracking` for the method (Heikenfeld et al. 2019).

    Parameters
    ----------
    obj : xarray.DataArray or xarray.Dataset
        The field (DataArray) or a dataset with it, on the horizontal dimensions ``y``
        and ``x`` (m) with any leading (time) dimensions. A vertical dimension
        (``z``, ``altitude``, ``height`` or ``level``) is reduced to its maximum, the
        composite.
    field : str, optional
        Name of the field in a dataset. Default: the first of ``DBZH``, ``DBZ``,
        ``reflectivity`` and ``corrected_reflectivity``.
    thresholds : sequence of float
        Thresholds of the field in ascending order, for example ``(30, 40, 50)`` dBZ.
        A cell is found at the highest threshold it reaches.
    min_area : float, optional
        Smallest area (km2) of a region above a threshold. Default 10.
    segmentation_threshold : float, optional
        Value of the field down to which a cell extends. Default: the lowest
        threshold; it cannot be higher.
    smooth : float, optional
        Width (standard deviation, grid cells) of a Gaussian filter that is applied to the
        field first. Default ``None``: none. A missing value counts as a value below
        the lowest threshold.
    connectivity : {4, 8}, optional
        Whether grid points that touch at a corner are connected. Default 8.
    n_threads : int, optional
        Threads for the compiled kernels. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled kernels and
        falls back to NumPy.

    Returns
    -------
    xarray.Dataset
        ``CELL_LABEL`` (the number of the cell at every grid point, 0 for none,
        numbered from 1 in every volume in the raster order of the first point of its
        feature) and ``CELL_COUNT`` (the number of cells per volume).

    Raises
    ------
    ValueError
        For an unknown ``engine``, thresholds that are not ascending, a
        ``segmentation_threshold`` above the lowest threshold, a ``connectivity`` that is
        not 4 or 8 or a field without the dimensions ``y`` and ``x``.
    KeyError
        If the field is not found.

    See Also
    --------
    track_cells, cell_statistics

    References
    ----------
    Heikenfeld, M., P. J. Marinescu, M. Christensen, D. Watson-Parris, F. Senf,
    S. C. van den Heever, and P. Stier, 2019: tobac 1.2: towards a flexible framework for
    tracking and analysis of clouds in diverse datasets. *Geosci. Model Dev.*, **12**,
    4551-4570, https://doi.org/10.5194/gmd-12-4551-2019

    Examples
    --------
    >>> cells = radarx.retrieve.identify_cells(grid, thresholds=(30, 40, 50))  # doctest: +SKIP
    >>> cells.CELL_COUNT  # doctest: +SKIP
    """
    use_compiled = choose_engine(engine, _cells.HAS_COMPILED_KERNEL, "cell")
    connectivity = _cells.check_connectivity(connectivity)
    thr = np.atleast_1d(np.asarray(thresholds, dtype=np.float64))
    if thr.ndim != 1 or thr.size == 0 or np.any(np.diff(thr) <= 0):
        raise ValueError("thresholds must be one or more values in ascending order")
    segmentation = (
        thr[0] if segmentation_threshold is None else float(segmentation_threshold)
    )
    if segmentation > thr[0]:
        raise ValueError("segmentation_threshold cannot be above the lowest threshold")
    if isinstance(obj, xr.Dataset):
        da = obj[find_field(obj, field, DBZ_NAMES, "reflectivity")]
    else:
        da = obj
    _check_horizontal(da, "the field")
    vertical = _vertical(da)
    if vertical is not None:
        da = da.max(vertical, skipna=True)
    dx, dy = _spacing(da)
    min_size = max(1, int(np.ceil(float(min_area) * 1e6 / abs(dx * dy))))

    def block(values):
        return _identify_block(
            values,
            thr,
            min_size,
            segmentation,
            smooth,
            connectivity,
            use_compiled,
            n_threads,
        )

    labels = xr.apply_ufunc(
        block,
        detach(da)[0],
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[np.int32],
        keep_attrs=False,
    ).astype(np.int32)
    out = xr.Dataset(
        {"CELL_LABEL": labels, "CELL_COUNT": labels.max(("y", "x")).astype(np.int32)}
    )
    out["CELL_LABEL"].attrs = {
        "long_name": "Number of the storm cell",
        "comment": (
            f"features at thresholds {', '.join(f'{t:g}' for t in thr)} with regions of at "
            f"least {min_area:g} km2, watershed down to {segmentation:g}"
        ),
    }
    out["CELL_COUNT"].attrs = {"long_name": "Number of storm cells"}
    return out


# --------------------------------------------------------------------------
# motion
# --------------------------------------------------------------------------


def _component(comp, npairs, nt):
    """A motion component as a float array (npairs or 1, ny or 1, nx or 1)."""
    if not isinstance(comp, xr.DataArray):
        return np.full((1, 1, 1), float(comp))
    extra = [d for d in comp.dims if d not in ("time", "y", "x")]
    if extra:
        raise ValueError(
            f"the motion has the dimensions {extra}: use time, y and x only"
        )
    if "time" in comp.dims:
        n = comp.sizes["time"]
        if n == nt:
            comp = comp.isel(time=slice(0, npairs))
        elif n != npairs:
            raise ValueError(
                f"the motion has {n} times for {nt} volumes: give one per volume or per interval"
            )
    order = [d for d in ("time", "y", "x") if d in comp.dims]
    values = np.asarray(comp.transpose(*order).values, dtype=np.float64)
    shape = [
        values.shape[order.index(d)] if d in order else 1 for d in ("time", "y", "x")
    ]
    return values.reshape(shape)


def _motion(motion, npairs, nt):
    """The motion (u, v), each an array (npairs or 1, ny or 1, nx or 1) in m/s."""
    if motion is None:
        return np.zeros((1, 1, 1)), np.zeros((1, 1, 1))
    if isinstance(motion, xr.Dataset):
        if "u" not in motion or "v" not in motion:
            raise ValueError("a motion dataset needs the variables u and v")
        u, v = motion["u"], motion["v"]
    else:
        try:
            u, v = motion
        except (TypeError, ValueError):
            raise TypeError(
                "motion must be (u, v) in m/s or a dataset with u and v"
            ) from None
    return _component(u, npairs, nt), _component(v, npairs, nt)


def _interval(cells, nt, dt):
    """Seconds between consecutive volumes, an array of nt - 1 values."""
    if dt is not None:
        dt = np.atleast_1d(np.asarray(dt, dtype=np.float64))
        if dt.size not in (1, nt - 1):
            raise ValueError(f"dt must give one value per interval ({nt - 1})")
        return np.broadcast_to(dt, (nt - 1,)).copy()
    if "time" not in cells.coords:
        return None
    values = cells["time"].values
    if np.issubdtype(values.dtype, np.datetime64):
        seconds = np.diff(values) / np.timedelta64(1, "s")
    else:
        seconds = np.diff(np.asarray(values, dtype=np.float64))
    return np.asarray(seconds, dtype=np.float64)


# --------------------------------------------------------------------------
# tracking
# --------------------------------------------------------------------------


def _centroids(labels, count, x, y, use_compiled, n_threads, chunk=32):
    """Area (cells) and centre coordinates of every cell: arrays (nt, nlab)."""
    nt = labels.shape[0]
    nlab = int(count.max()) if count.size else 0
    area = np.zeros((nt, nlab))
    cx = np.full((nt, nlab), np.nan)
    cy = np.full((nt, nlab), np.nan)
    if nlab == 0:
        return area, cx, cy
    xx, yy = np.meshgrid(x, y)
    for t0 in range(0, nt, chunk):
        sel = slice(t0, min(nt, t0 + chunk))
        n = labels[sel].shape[0]
        values = np.stack(
            [
                np.broadcast_to(np.ones_like(xx), (n,) + xx.shape),
                np.broadcast_to(xx, (n,) + xx.shape),
                np.broadcast_to(yy, (n,) + yy.shape),
            ]
        )
        total, seen, _, _ = _cells.region_sums(
            labels[sel], values, nlab, use_compiled, n_threads
        )
        area[sel] = seen[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            cx[sel] = np.where(seen[0] > 0, total[1] / seen[0], np.nan)
            cy[sel] = np.where(seen[0] > 0, total[2] / seen[0], np.nan)
    return area, cx, cy


def _shifts(u, v, seconds, dx, dy, ny, nx):
    """Whole-cell shifts (rows, columns), each (npairs, sy, sx), of the motion in every interval."""
    npairs = seconds.size
    rows = np.rint(v * seconds[:, None, None] / dy)
    cols = np.rint(u * seconds[:, None, None] / dx)
    sy = max(rows.shape[1], cols.shape[1])
    sx = max(rows.shape[2], cols.shape[2])
    if sy not in (1, ny) or sx not in (1, nx):
        raise ValueError(
            "a motion on the grid needs the dimensions y and x of the cells"
        )
    rows = np.broadcast_to(rows, (npairs, sy, sx))
    cols = np.broadcast_to(cols, (npairs, sy, sx))
    return rows.astype(np.int32), cols.astype(np.int32)


def _predicted(rows, cols, cx, cy, x0, y0, dx, dy):
    """Positions that the shifts of an interval give to the centres (cx, cy)."""
    sy, sx = rows.shape
    iy = np.clip(np.rint(np.nan_to_num(cy - y0) / dy), 0, None).astype(int)
    ix = np.clip(np.rint(np.nan_to_num(cx - x0) / dx), 0, None).astype(int)
    iy = np.minimum(iy, sy - 1) if sy > 1 else np.zeros_like(iy)
    ix = np.minimum(ix, sx - 1) if sx > 1 else np.zeros_like(ix)
    return cx + cols[iy, ix] * dx, cy + rows[iy, ix] * dy


def _greedy(links, overlap_area):
    """One-to-one continuation: links in the order of their overlap area."""
    order = sorted(links, key=lambda ab: (-overlap_area[ab], ab))
    taken_a, taken_b, chosen = set(), set(), {}
    for a, b in order:
        if a in taken_a or b in taken_b:
            continue
        taken_a.add(a)
        taken_b.add(b)
        chosen[a] = b
    return chosen


def _link_pair(ov, area0, area1, min_overlap, predicted, cx1, cy1, radius):
    """
    The continuations, splits and merges between the cells of two volumes.

    Returns ``continued`` (cell of the first volume -> cell of the second), ``split``
    (new cell -> parent) and ``merged`` (ending cell -> cell it merges into); cells are
    numbered from 0.
    """
    n0, n1 = ov.shape
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.maximum(ov / area0[:, None], ov / area1[None, :])
    linked = (ov > 0) & (frac >= min_overlap)
    links = [(int(a), int(b)) for a, b in zip(*np.nonzero(linked))]
    overlap_area = {ab: ov[ab] for ab in links}
    continued = _greedy(links, overlap_area)
    if radius is not None:
        free_a = [a for a in range(n0) if a not in continued and not linked[a].any()]
        free_b = [
            b
            for b in range(n1)
            if b not in continued.values() and not linked[:, b].any()
        ]
        pairs = []
        for a in free_a:
            for b in free_b:
                d = np.hypot(predicted[0][a] - cx1[b], predicted[1][a] - cy1[b])
                if np.isfinite(d) and d <= radius:
                    pairs.append((d, a, b))
        used_a, used_b = set(), set()
        for d, a, b in sorted(pairs):
            if a in used_a or b in used_b:
                continue
            used_a.add(a)
            used_b.add(b)
            continued[a] = b
    taken_b = set(continued.values())
    split = {}
    for b in range(n1):
        if b in taken_b:
            continue
        parents = [(ov[a, b], a) for a in range(n0) if linked[a, b]]
        if parents:
            split[b] = max(parents, key=lambda p: (p[0], -p[1]))[1]
    merged = {}
    for a in range(n0):
        if a in continued:
            continue
        targets = [(ov[a, b], b) for b in range(n1) if linked[a, b]]
        if targets:
            merged[a] = max(targets, key=lambda p: (p[0], -p[1]))[1]
    return continued, split, merged


@dask_support(
    "computes",
    "a track runs through all volumes, so that the cells of every volume are needed.",
)
@provenance("Storm cell tracking by overlap after PyFLEXTRKR and tobac")
def track_cells(
    cells,
    *,
    motion=None,
    min_overlap=0.3,
    max_speed=None,
    dt=None,
    min_lifetime=1,
    n_threads=None,
    engine="auto",
):
    """
    Track the storm cells of a sequence of volumes by their overlap.

    See :mod:`radarx.retrieve.tracking` for the method (Feng et al. 2023).

    Parameters
    ----------
    cells : xarray.Dataset or xarray.DataArray
        The cell numbers of every volume on ``(time, y, x)``: the result of
        :func:`identify_cells` (``CELL_LABEL``) or any integer labels, 0 for none.
    motion : tuple, xarray.Dataset or None, optional
        Storm motion ``(u, v)`` in m/s (eastward, northward) by which the cells of a
        volume are moved before the overlap is counted: a pair of numbers, or a dataset
        with ``u`` and ``v`` that are scalars, have one value per time (per volume or per
        interval) or a field on ``(y, x)`` (and time), as :func:`estimate_motion` gives.
        Default: none.
    min_overlap : float, optional
        Smallest overlap area, as a fraction of the area of either cell, for a link.
        Default 0.3, the value that Feng et al. (2023) use with advected cells.
    max_speed : float, optional
        Largest speed (m/s) for linking cells that do not overlap: a cell without
        any link is linked to the nearest cell without a link within this speed times
        the interval of the position that the motion predicts. Default ``None``: no
        such links.
    dt : float or sequence of float, optional
        Seconds between volumes, if the cells have no ``time`` coordinate.
    min_lifetime : int, optional
        Shortest track to keep, in volumes. Default 1 (all tracks).
    n_threads : int, optional
        Threads for the compiled kernels. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled kernels and
        falls back to NumPy.

    Returns
    -------
    xarray.Dataset
        ``CELL_ID`` on ``(time, y, x)`` (the track of the cell at a grid point, 0 for
        none) and, on the dimension ``cell`` (the track numbers from 1):
        ``cell_first_time``, ``cell_last_time``, ``cell_lifetime`` (number of
        volumes), ``cell_split_from`` and ``cell_merged_into`` (track numbers, 0 for
        none); on ``(cell, time)``: ``cell_area`` (km2), ``cell_x`` and ``cell_y`` (m,
        the centre of the cell), ``cell_u`` and ``cell_v`` (m/s, from the change of
        position between volumes) - NaN where the track is absent.

    Raises
    ------
    ValueError
        For an unknown ``engine``, cells without the dimensions ``time``, ``y`` and
        ``x``, a motion or a ``dt`` that does not fit the volumes, or no time step
        when the motion or ``max_speed`` needs one.
    TypeError
        If ``motion`` is not a pair or a dataset.

    See Also
    --------
    identify_cells, cell_statistics, estimate_motion

    References
    ----------
    Feng, Z., J. Hardin, H. C. Barnes, J. Li, L. R. Leung, A. Varble, and Z. Zhang, 2023:
    PyFLEXTRKR: a flexible feature tracking Python software for convective cloud
    analysis. *Geosci. Model Dev.*, **16**, 2753-2776,
    https://doi.org/10.5194/gmd-16-2753-2023

    Heikenfeld, M., P. J. Marinescu, M. Christensen, D. Watson-Parris, F. Senf,
    S. C. van den Heever, and P. Stier, 2019: tobac 1.2: towards a flexible framework for
    tracking and analysis of clouds in diverse datasets. *Geosci. Model Dev.*, **12**,
    4551-4570, https://doi.org/10.5194/gmd-12-4551-2019

    Examples
    --------
    >>> tracks = radarx.retrieve.track_cells(cells, motion=(12.0, 5.0))  # doctest: +SKIP
    >>> tracks.cell_lifetime  # doctest: +SKIP
    """
    use_compiled = choose_engine(engine, _cells.HAS_COMPILED_KERNEL, "cell")
    cells = _check_cells(cells, min_overlap)
    labels = np.ascontiguousarray(cells.values, dtype=np.int32)
    nt = labels.shape[0]
    dx, dy = _spacing(cells)
    count = labels.reshape(nt, -1).max(axis=1) if labels.size else np.zeros(nt, int)
    geometry = _Geometry(
        labels,
        count,
        *_centroids(
            labels,
            count,
            np.asarray(cells["x"].values, dtype=np.float64),
            np.asarray(cells["y"].values, dtype=np.float64),
            use_compiled,
            n_threads,
        ),
        dx,
        dy,
        float(cells["x"].values[0]),
        float(cells["y"].values[0]),
    )
    npairs = nt - 1
    seconds = _interval(cells, nt, dt) if npairs else np.zeros(0)
    if npairs and (motion is not None or max_speed is not None) and seconds is None:
        raise ValueError("the cells have no time coordinate: pass dt")
    if seconds is None:
        seconds = np.zeros(npairs)
    u, v = _motion(motion, npairs, nt)
    options = (min_overlap, max_speed, use_compiled, n_threads)
    track_of, split_from, merged_into = _link_all(geometry, u, v, seconds, options)
    return _track_dataset(
        cells,
        geometry,
        seconds,
        track_of,
        split_from,
        merged_into,
        min_lifetime,
        min_overlap,
    )


class _Geometry:
    """Labels, cell counts, areas and centres of the cells of every volume."""

    def __init__(self, labels, count, area, cx, cy, dx, dy, x0, y0):
        self.labels, self.count = labels, count
        self.area, self.cx, self.cy = area, cx, cy
        self.dx, self.dy, self.x0, self.y0 = dx, dy, x0, y0


def _check_cells(cells, min_overlap):
    """The cell numbers as a DataArray on (time, y, x), after the checks of the arguments."""
    if isinstance(cells, xr.Dataset):
        if "CELL_LABEL" not in cells:
            raise ValueError("the dataset needs the variable CELL_LABEL")
        cells = cells["CELL_LABEL"]
    for d in ("time", "y", "x"):
        if d not in cells.dims:
            raise ValueError(
                f"the cells need the dimensions time, y and x, not {cells.dims}"
            )
    if cells.ndim != 3:
        raise ValueError(
            f"the cells need the dimensions time, y and x only, not {cells.dims}"
        )
    if not 0.0 < min_overlap <= 1.0:
        raise ValueError("min_overlap must be in (0, 1]")
    return cells.transpose("time", "y", "x")


def _link_all(g, u, v, seconds, options):
    """
    Link the cells of all consecutive volumes.

    Returns the track number of every cell of every volume, the track that each new
    track split from and the track that each ending track merged into.
    """
    min_overlap, max_speed, use_compiled, n_threads = options
    nt, ny, nx = g.labels.shape
    npairs = nt - 1
    track_of = [np.arange(1, g.count[0] + 1)] if nt else []
    split_from, merged_into = {}, {}
    if not npairs:
        return track_of, split_from, merged_into
    rows, cols = _shifts(u, v, seconds, g.dx, g.dy, ny, nx)
    counts = _cells.overlap(
        g.labels[:-1],
        g.labels[1:],
        g.count[:-1],
        g.count[1:],
        rows,
        cols,
        use_compiled,
        n_threads,
    )
    offsets = np.concatenate(
        [[0], np.cumsum(g.count[:-1].astype(np.int64) * g.count[1:])]
    )
    next_id = int(g.count[0]) + 1
    for p in range(npairs):
        n0, n1 = int(g.count[p]), int(g.count[p + 1])
        links = ({}, {}, {})
        if n0 and n1:
            ov = counts[offsets[p] : offsets[p + 1]].reshape(n0, n1)
            predicted, radius = None, None
            if max_speed is not None:
                predicted = _predicted(
                    rows[p], cols[p], g.cx[p, :n0], g.cy[p, :n0], g.x0, g.y0, g.dx, g.dy
                )
                radius = float(max_speed) * seconds[p]
            links = _link_pair(
                ov,
                np.maximum(g.area[p, :n0], 1.0),
                np.maximum(g.area[p + 1, :n1], 1.0),
                min_overlap,
                predicted,
                g.cx[p + 1, :n1],
                g.cy[p + 1, :n1],
                radius,
            )
        ids, next_id = _new_numbers(track_of[p], n1, next_id, links, split_from)
        for a, b in links[2].items():
            merged_into[int(track_of[p][a])] = int(ids[b])
        track_of.append(ids)
    return track_of, split_from, merged_into


def _new_numbers(previous, n1, next_id, links, split_from):
    """Track numbers of the n1 cells of a volume; new tracks get the numbers from next_id."""
    continued, split, _ = links
    ids = np.zeros(n1, dtype=np.int64)
    for a, b in continued.items():
        ids[b] = previous[a]
    for b in range(n1):
        if ids[b] == 0:
            ids[b] = next_id
            next_id += 1
            if b in split:
                split_from[int(ids[b])] = int(previous[split[b]])
    return ids, next_id


def _lifetimes(track_of, ntrack):
    """First and last volume and number of volumes of every track (index 0 unused)."""
    first = np.full(ntrack + 1, -1)
    last = np.full(ntrack + 1, -1)
    lifetime = np.zeros(ntrack + 1, dtype=np.int64)
    for t, ids in enumerate(track_of):
        ids = np.asarray(ids, dtype=np.int64)
        new = ids[first[ids] < 0]
        first[new] = t
        last[ids] = t
        lifetime[ids] += 1
    return first, last, lifetime


def _track_series(g, track_of, renumber, nkeep, cell_km2):
    """The map of the track numbers and area and centre of every track per volume."""
    nt = g.labels.shape[0]
    cell_id = np.zeros(g.labels.shape, dtype=np.int32)
    area_out = np.full((nkeep, nt), np.nan)
    x_out = np.full((nkeep, nt), np.nan)
    y_out = np.full((nkeep, nt), np.nan)
    for t, ids in enumerate(track_of):
        lut = np.zeros(int(g.count[t]) + 1, dtype=np.int32)
        lut[1:] = renumber[np.asarray(ids, dtype=np.int64)]
        cell_id[t] = lut[g.labels[t]]
        present = np.flatnonzero(lut[1:])
        k = lut[1:][present] - 1
        area_out[k, t] = g.area[t, present] * cell_km2
        x_out[k, t] = g.cx[t, present]
        y_out[k, t] = g.cy[t, present]
    return cell_id, area_out, x_out, y_out


_TRACK_ATTRS = {
    "CELL_ID": {"long_name": "Number of the tracked storm cell"},
    "cell_lifetime": {"long_name": "Number of volumes of the track", "units": "1"},
    "cell_split_from": {"long_name": "Track that the cell split from, 0 for none"},
    "cell_merged_into": {"long_name": "Track that the cell merged into, 0 for none"},
    "cell_area": {"long_name": "Area of the cell", "units": "km2"},
    "cell_x": {"long_name": "x of the centre of the cell", "units": "m"},
    "cell_y": {"long_name": "y of the centre of the cell", "units": "m"},
    "cell_u": {
        "long_name": "Eastward speed of the centre of the cell",
        "units": "m s-1",
    },
    "cell_v": {
        "long_name": "Northward speed of the centre of the cell",
        "units": "m s-1",
    },
}


def _track_dataset(
    cells, g, seconds, track_of, split_from, merged_into, min_lifetime, min_overlap
):
    """The tracks as a dataset: the map of the tracks and the variables of every track."""
    nt = g.labels.shape[0]
    ntrack = int(max(ids.max() for ids in track_of if ids.size)) if g.count.any() else 0
    first, last, lifetime = _lifetimes(track_of, ntrack)
    keep = np.flatnonzero(lifetime >= max(1, int(min_lifetime)))
    keep = keep[keep > 0]
    renumber = np.zeros(ntrack + 1, dtype=np.int64)
    renumber[keep] = np.arange(1, keep.size + 1)
    nkeep = keep.size
    cell_id, area_out, x_out, y_out = _track_series(
        g, track_of, renumber, nkeep, abs(g.dx * g.dy) / 1e6
    )
    splits = np.array(
        [renumber[split_from.get(int(k), 0)] for k in keep], dtype=np.int32
    )
    merges = np.array(
        [renumber[merged_into.get(int(k), 0)] for k in keep], dtype=np.int32
    )
    u_out = np.full((nkeep, nt), np.nan)
    v_out = np.full((nkeep, nt), np.nan)
    if nt > 1:
        step = np.where(seconds > 0, seconds, np.nan)
        u_out[:, 1:] = np.diff(x_out, axis=1) / step
        v_out[:, 1:] = np.diff(y_out, axis=1) / step
    time = cells["time"]
    out = xr.Dataset(
        {
            "CELL_ID": (("time", "y", "x"), cell_id),
            "cell_first_time": ("cell", time.values[first[keep]]),
            "cell_last_time": ("cell", time.values[last[keep]]),
            "cell_lifetime": ("cell", lifetime[keep].astype(np.int32)),
            "cell_split_from": ("cell", splits),
            "cell_merged_into": ("cell", merges),
            "cell_area": (("cell", "time"), area_out),
            "cell_x": (("cell", "time"), x_out),
            "cell_y": (("cell", "time"), y_out),
            "cell_u": (("cell", "time"), u_out),
            "cell_v": (("cell", "time"), v_out),
        },
        coords={
            "time": time,
            "y": cells["y"],
            "x": cells["x"],
            "cell": np.arange(1, nkeep + 1),
        },
    )
    for name, attrs in _TRACK_ATTRS.items():
        out[name].attrs = dict(attrs)
    out.attrs["comment"] = (
        f"overlap of at least {min_overlap:g} of the area of either cell after the storm motion"
    )
    return out


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


@dask_support(
    "computes",
    "the statistics of a cell are collected over all grid points of the volume.",
)
@provenance("Statistics of fields inside storm cells")
def cell_statistics(
    cells, fields, *, stats=("mean", "max"), n_threads=None, engine="auto"
):
    """
    Time series of statistics of fields inside every storm cell.

    Parameters
    ----------
    cells : xarray.Dataset or xarray.DataArray
        The cell numbers on ``(time, y, x)``: ``CELL_ID`` of :func:`track_cells`
        (the cells are the tracks) or ``CELL_LABEL`` of :func:`identify_cells` (the cells
        are numbered in every volume).
    fields : xarray.DataArray or xarray.Dataset
        The fields on the grid of the cells (``y``, ``x`` and ``time``), for example the
        azimuthal shear, the lightning density, the depth of the ZDR column or a
        DSD parameter. A field with a vertical dimension must be reduced first.
    stats : sequence of str, optional
        Any of ``"mean"``, ``"max"``, ``"min"``, ``"sum"`` and ``"count"`` (the number of
        grid points with a value). Default ``("mean", "max")``.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled kernel and
        falls back to NumPy.

    Returns
    -------
    xarray.Dataset
        One variable ``<NAME>_<stat>`` per field and statistic on ``(cell, time)``,
        NaN where the cell has no grid point with a value at that time.

    Raises
    ------
    ValueError
        For an unknown ``engine`` or statistic, cells or fields without the dimensions
        ``time``, ``y`` and ``x`` or on another grid.

    See Also
    --------
    track_cells, identify_cells

    Examples
    --------
    >>> series = radarx.retrieve.cell_statistics(tracks.CELL_ID, shear, stats=("max",))  # doctest: +SKIP
    """
    use_compiled = choose_engine(engine, _cells.HAS_COMPILED_KERNEL, "cell")
    for stat in stats:
        if stat not in _STATS:
            raise ValueError(f"unknown statistic {stat!r}: use {_STATS}")
    cells = _statistics_cells(cells)
    if isinstance(fields, xr.DataArray):
        fields = fields.to_dataset(name=fields.name or "field")
    labels = np.ascontiguousarray(cells.values, dtype=np.int32)
    nlab = int(labels.max()) if labels.size else 0
    names, arrays = _statistics_fields(fields, cells)
    shape = (nlab, labels.shape[0])
    computed = None
    if names and nlab:
        sums = _cells.region_sums(
            labels, np.stack(arrays), nlab, use_compiled, n_threads
        )
        computed = {stat: _statistic(sums, stat) for stat in stats}
    data = {}
    for k, name in enumerate(names):
        for stat in stats:
            values = np.full(shape, np.nan) if computed is None else computed[stat][k].T
            data[f"{name}_{stat}"] = (("cell", "time"), values)
    out = xr.Dataset(
        data, coords={"cell": np.arange(1, nlab + 1), "time": cells["time"]}
    )
    for name in names:
        unit = fields[name].attrs.get("units")
        for stat in stats:
            var = out[f"{name}_{stat}"]
            var.attrs = {"long_name": f"{stat} of {name} in the cell"}
            if unit is not None and stat != "count":
                var.attrs["units"] = unit
    return out


def _statistics_cells(cells):
    """The cell numbers as a DataArray on (time, y, x)."""
    if isinstance(cells, xr.Dataset):
        names = [n for n in ("CELL_ID", "CELL_LABEL") if n in cells]
        if not names:
            raise ValueError("the dataset needs the variable CELL_ID or CELL_LABEL")
        cells = cells[names[0]]
    for d in ("time", "y", "x"):
        if d not in cells.dims:
            raise ValueError(
                f"the cells need the dimensions time, y and x, not {cells.dims}"
            )
    return cells.transpose("time", "y", "x")


def _statistics_fields(fields, cells):
    """Names and values (time, y, x) of the fields, checked against the grid of the cells."""
    names, arrays = [], []
    for name, da in fields.data_vars.items():
        if set(da.dims) != {"time", "y", "x"}:
            raise ValueError(
                f"{name} has the dimensions {da.dims}: reduce it to (time, y, x) first"
            )
        da = da.transpose("time", "y", "x")
        if any(da[d].size != cells[d].size for d in ("time", "y", "x")):
            raise ValueError(f"{name} is not on the grid of the cells")
        names.append(name)
        arrays.append(np.asarray(da.values, dtype=np.float64))
    return names, arrays


def _statistic(sums, stat):
    """One statistic from the sums, counts, maxima and minima of the labels."""
    total, seen, high, low = sums
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "mean": np.where(seen > 0, total / seen, np.nan),
            "sum": np.where(seen > 0, total, np.nan),
            "max": high,
            "min": low,
            "count": seen,
        }[stat]


# --------------------------------------------------------------------------
# accessors
# --------------------------------------------------------------------------


@accessor_method("dataarray", "dataset", name="identify_cells", wraps=identify_cells)
def _identify_cells_accessor(self):
    """
    Storm cells of every volume of a grid: multi-threshold features and watershed.

    See :func:`radarx.retrieve.identify_cells` for the parameters.

    Returns
    -------
    xarray.Dataset
        ``CELL_LABEL`` and ``CELL_COUNT``.
    """


@accessor_method("dataset", "dataarray", name="track_cells", wraps=track_cells)
def _track_cells_accessor(self):
    """
    Track the storm cells of a sequence of volumes by their overlap.

    See :func:`radarx.retrieve.track_cells` for the parameters.

    Returns
    -------
    xarray.Dataset
        ``CELL_ID`` and the track variables on ``cell``.
    """


@accessor_method("dataset", "dataarray", name="cell_statistics", wraps=cell_statistics)
def _cell_statistics_accessor(self):
    """
    Time series of statistics of fields inside every storm cell.

    See :func:`radarx.retrieve.cell_statistics` for the parameters; the object is the
    cell numbers.

    Returns
    -------
    xarray.Dataset
        One variable per field and statistic on ``(cell, time)``.
    """
