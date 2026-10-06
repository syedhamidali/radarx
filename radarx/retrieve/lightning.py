#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Lightning: Flashes, Gridded Products and Lightning Jumps
========================================================

Total-lightning products from Lightning Mapping Array (LMA) VHF sources, e.g.
read with :func:`radarx.io.read_lma`, on radarx grids and storm cells.

All functions take and return :py:class:`xarray.Dataset` /
:py:class:`xarray.DataArray` objects in the layout of
:func:`radarx.io.read_lma` (sources on ``number_of_events``, flashes on
``number_of_flashes``, linked by ``event_parent_flash_id``), which is also the
CF layout of the xlma-python package.

Flashes
-------
:func:`cluster_flashes` groups sources into flashes by their separation in
space and time (Fuchs et al. 2016): the source positions are converted to
Earth-centred Cartesian coordinates, divided by a distance scale (default
3 km) and the source times by a time scale (default 0.15 s), and two sources
belong to the same flash when their normalized space-time distance is at most
one,

.. math::

    \\left(\\frac{|\\mathbf{x}_i - \\mathbf{x}_j|}{d}\\right)^2 +
    \\left(\\frac{t_i - t_j}{\\tau}\\right)^2 \\le 1 .

Flashes are the connected groups of such links, i.e. the clusters of DBSCAN
with a minimum of one point, as in Fuchs et al. (2016) and the xlma-python
flash sorting. Unlike their streamed processing, no maximum flash duration
is imposed (it only limits memory there). Flashes with few sources are kept
and left out later (``min_sources``), as is customary (Fuchs et al. 2016).
For every flash the start and end time, duration, number of sources,
initiation point (first source), centroid and plan area (convex hull of the
sources seen from above; Bruning and MacGorman 2013) are computed.

Gridded products
----------------
:func:`grid_lightning` counts, in every grid box of a radarx grid (or any
``x``/``y`` grid east and north of an origin, azimuthal equidistant as in
:func:`radarx.grid.grid_cones`) and time interval:

- ``source_density``: VHF sources;
- ``flash_extent_density``: flashes with at least one source in the box
  (Bruning and MacGorman 2013);
- ``flash_initiation_density``: flashes whose first source is in the box.

With heights ``z`` the counts are per 3-D box (lightning on the radar grid);
without, per column.

Storm cells and lightning jumps
-------------------------------
:func:`cell_flash_rate` counts the flashes (by initiation point, or every
cell a flash touches) and the sources by height in every cell of a
time-dependent cell mask, e.g. a tracked-storm segmentation, giving flash
rates per cell and vertical source distributions.
:func:`vertical_source_distribution` gives the height distribution of all
sources and flash initiations.

:func:`lightning_jump` applies the "2σ" lightning jump algorithm of Schultz
et al. (2009), as specified by Schultz et al. (2011, section 2c) and Schultz
et al. (2016):

1. the total flash rate is averaged over 2-min periods;
2. the rate of change ``DFRDT`` is the difference between consecutive
   periods divided by the period (flashes min\\ :sup:`-2`);
3. :math:`\\sigma` is the standard deviation of the five previous ``DFRDT``
   values, and the sigma level is ``DFRDT`` / :math:`\\sigma`;
4. a jump occurs when the sigma level reaches 2 while the flash rate is at
   least 10 flashes min\\ :sup:`-1`, after a spin-up of six periods (five
   ``DFRDT`` values before the current one, 14 min);
5. a jump lasts until the sigma level drops below zero, and jumps starting
   within 6 min of an earlier one are one jump.

The heavy loops (clustering, flash properties, gridding, cell counts) run in
a compiled kernel (``radarx.retrieve._lightning``, multithreaded over sources
and flashes) with an identical NumPy reference implementation as fallback.

References
----------
Bruning, E. C., and D. R. MacGorman, 2013: Theory and observations of controls
on lightning flash size spectra. *J. Atmos. Sci.*, **70** (12), 4012-4029,
https://doi.org/10.1175/JAS-D-12-0289.1

Fuchs, B. R., E. C. Bruning, S. A. Rutledge, L. D. Carey, P. R. Krehbiel,
and W. Rison, 2016: Climatological analyses of LMA data with an open-source
lightning flash-clustering algorithm. *J. Geophys. Res. Atmos.*, **121**
(14), 8625-8648, https://doi.org/10.1002/2015JD024663

Schultz, C. J., W. A. Petersen, and L. D. Carey, 2009: Preliminary
development and evaluation of lightning jump algorithms for the real-time
detection of severe weather. *J. Appl. Meteor. Climatol.*, **48** (12),
2543-2563, https://doi.org/10.1175/2009JAMC2237.1

Schultz, C. J., W. A. Petersen, and L. D. Carey, 2011: Lightning and severe
weather: A comparison between total and cloud-to-ground lightning trends.
*Wea. Forecasting*, **26** (5), 744-755,
https://doi.org/10.1175/WAF-D-10-05026.1

Schultz, E. V., C. J. Schultz, L. D. Carey, D. J. Cecil, and M. Bateman,
2016: Automated storm tracking and the lightning jump algorithm using GOES-R
Geostationary Lightning Mapper (GLM) proxy data. *J. Operational Meteor.*,
**4** (7), 92-107, https://doi.org/10.15191/nwajom.2016.0407

.. autosummary::
   :nosignatures:
   :toctree: generated/

   cluster_flashes
   grid_lightning
   cell_flash_rate
   vertical_source_distribution
   lightning_jump
"""

from __future__ import annotations

__all__ = [
    "cluster_flashes",
    "grid_lightning",
    "cell_flash_rate",
    "vertical_source_distribution",
    "lightning_jump",
]

import re

import numpy as np
import pandas as pd
import xarray as xr

from .._registry import accessor_method

try:
    from . import _lightning

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _lightning = None
    HAS_COMPILED_KERNEL = False

EVENTS = "number_of_events"
FLASHES = "number_of_flashes"
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
# nanoseconds per unit of interval strings such as "5min" or "30s"
_UNITS = {
    "ns": 1.0,
    "us": 1e3,
    "ms": 1e6,
    "s": 1e9,
    "sec": 1e9,
    "min": 60e9,
    "t": 60e9,
    "h": 3600e9,
    "hour": 3600e9,
    "d": 86400e9,
    "day": 86400e9,
}

_FLASH_ATTRS = {
    "flash_id": {"long_name": "Flash identifier", "cf_role": "tree_id"},
    "flash_time_start": {"long_name": "Time of the first source of the flash"},
    "flash_time_end": {"long_name": "Time of the last source of the flash"},
    "flash_duration": {"long_name": "Duration of the flash", "units": "s"},
    "flash_event_count": {"long_name": "Number of sources in the flash", "units": "1"},
    "flash_init_latitude": {
        "standard_name": "latitude",
        "long_name": "Latitude of the first source of the flash",
        "units": "degrees_north",
    },
    "flash_init_longitude": {
        "standard_name": "longitude",
        "long_name": "Longitude of the first source of the flash",
        "units": "degrees_east",
    },
    "flash_init_altitude": {
        "standard_name": "altitude",
        "long_name": "Altitude of the first source of the flash",
        "units": "m",
    },
    "flash_center_latitude": {
        "standard_name": "latitude",
        "long_name": "Mean latitude of the sources of the flash",
        "units": "degrees_north",
    },
    "flash_center_longitude": {
        "standard_name": "longitude",
        "long_name": "Mean longitude of the sources of the flash",
        "units": "degrees_east",
    },
    "flash_center_altitude": {
        "standard_name": "altitude",
        "long_name": "Mean altitude of the sources of the flash",
        "units": "m",
    },
    "flash_area": {
        "long_name": "Plan area of the flash (convex hull of its sources)",
        "units": "km2",
    },
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _use_compiled(engine):
    """Whether to run the compiled kernel for the requested ``engine``."""
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled lightning kernel is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


def _f64(x):
    return np.ascontiguousarray(x, dtype=np.float64)


def _i64(x):
    return np.ascontiguousarray(x, dtype=np.int64)


def _check_events(ds):
    for name in ("event_time", "event_latitude", "event_longitude", "event_altitude"):
        if name not in ds.variables:
            raise ValueError(f"the LMA dataset needs {name!r} (see radarx.io.read_lma)")


def _ecef(lon, lat, alt):
    """WGS84 geodetic to Earth-centred Cartesian coordinates (m)."""
    lam, phi = np.radians(lon), np.radians(lat)
    e2 = WGS84_F * (2.0 - WGS84_F)
    sphi = np.sin(phi)
    n = WGS84_A / np.sqrt(1.0 - e2 * sphi * sphi)
    x = (n + alt) * np.cos(phi) * np.cos(lam)
    y = (n + alt) * np.cos(phi) * np.sin(lam)
    z = (n * (1.0 - e2) + alt) * sphi
    return x, y, z


def _project(lon, lat, latitude, longitude, n_threads=None):
    """
    Azimuthal equidistant x, y (m) about the origin, as radarx grids.

    PROJ releases the GIL, so large inputs are projected in chunks on threads
    (one transformer per chunk); the result does not depend on the chunking.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    import pyproj

    crs = pyproj.CRS.from_dict(
        {"proj": "aeqd", "lat_0": latitude, "lon_0": longitude, "datum": "WGS84"}
    )
    lon = np.ascontiguousarray(lon, dtype=float)
    lat = np.ascontiguousarray(lat, dtype=float)

    def project(sl):
        tr = pyproj.Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)
        return tr.transform(lon[sl], lat[sl])

    n = lon.size
    nt = int(n_threads or os.cpu_count() or 1)
    nchunk = min(4 * nt, max(1, n // 100_000))
    if nt == 1 or nchunk == 1:
        x, y = project(slice(None))
    else:
        b = np.linspace(0, n, nchunk + 1).astype(int)
        with ThreadPoolExecutor(min(nt, nchunk)) as pool:
            parts = list(
                pool.map(project, [slice(i, j) for i, j in zip(b[:-1], b[1:])])
            )
        x = np.concatenate([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _seconds(times, ref):
    """Seconds since ``ref`` (datetime64) as float64."""
    return (np.asarray(times, dtype="datetime64[ns]") - ref).astype(np.int64) * 1e-9


def _edges(centers, name):
    """Bin edges midway between monotonic increasing centres."""
    c = np.asarray(centers, dtype=np.float64)
    if c.ndim != 1 or c.size < 1:
        raise ValueError(f"{name} must be a 1-D coordinate")
    if c.size == 1:
        return np.array([-np.inf, np.inf]) if name == "z" else c + [-0.5, 0.5]
    if np.any(np.diff(c) <= 0):
        raise ValueError(f"{name} must increase monotonically")
    mid = 0.5 * (c[1:] + c[:-1])
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


def _interval(value, name="interval"):
    try:
        if isinstance(value, np.timedelta64):
            td = value.astype("timedelta64[ns]")
        elif isinstance(value, str):
            m = re.fullmatch(r"\s*([-+]?\d*\.?\d+)\s*([a-zA-Z]+)\s*", value)
            if not m or m.group(2).lower() not in _UNITS:
                raise ValueError(value)
            ns = float(m.group(1)) * _UNITS[m.group(2).lower()]
            td = np.timedelta64(int(round(ns)), "ns")
        else:
            td = pd.Timedelta(value).to_timedelta64().astype("timedelta64[ns]")
    except (ValueError, TypeError) as err:
        raise ValueError(f"{name} must be a time interval, not {value!r}") from err
    if td <= np.timedelta64(0, "ns"):
        raise ValueError(f"{name} must be positive")
    return td


def _time_edges(times, interval, time_edges):
    """Time bin edges (datetime64[ns]) covering ``times``."""
    if time_edges is not None:
        edges = np.asarray(time_edges, dtype="datetime64[ns]")
        if (
            edges.ndim != 1
            or edges.size < 2
            or np.any(np.diff(edges) <= np.timedelta64(0, "ns"))
        ):
            raise ValueError("time_edges must be at least two increasing times")
        return edges
    step = _interval(interval)
    t = np.asarray(times, dtype="datetime64[ns]")
    if t.size == 0:
        raise ValueError("no sources to bin in time; give time_edges")
    i0 = t.min().astype(np.int64) // step.astype(np.int64)
    i1 = t.max().astype(np.int64) // step.astype(np.int64) + 1
    return (np.arange(i0, i1 + 1) * step.astype(np.int64)).astype("datetime64[ns]")


def _time_coords(edges):
    centers = edges[:-1] + (edges[1:] - edges[:-1]) // 2
    bounds = np.stack([edges[:-1], edges[1:]], axis=-1)
    return {
        "time": ("time", centers, {"long_name": "Centre of the time interval"}),
        "time_bounds": (("time", "bounds"), bounds),
    }


def _origin(obj, latitude, longitude):
    """Origin of x/y grid coordinates: arguments, then coords or attrs."""
    if latitude is not None and longitude is not None:
        return float(latitude), float(longitude)
    if obj is not None:
        names = obj.coords if isinstance(obj, xr.DataArray) else obj.variables
        for lat, lon in (
            ("latitude", "longitude"),
            ("origin_latitude", "origin_longitude"),
            ("radar_latitude", "radar_longitude"),
        ):
            if lat in names and lon in names:
                return float(np.ravel(obj[lat].values)[0]), float(
                    np.ravel(obj[lon].values)[0]
                )
        if "crs_wkt" in names:
            a = obj["crs_wkt"].attrs
            if "latitude_of_projection_origin" in a:
                return float(a["latitude_of_projection_origin"]), float(
                    a["longitude_of_projection_origin"]
                )
        for lat, lon in (
            ("latitude", "longitude"),
            ("origin_latitude", "origin_longitude"),
        ):
            if lat in obj.attrs and lon in obj.attrs:
                return float(obj.attrs[lat]), float(obj.attrs[lon])
    raise ValueError(
        "the grid origin is unknown; give latitude and longitude (the point at "
        "x = y = 0) or a grid with latitude/longitude coordinates"
    )


# --------------------------------------------------------------------------
# NumPy reference implementations (the kernel follows them)
# --------------------------------------------------------------------------


def _components(n, i, j, labels):
    """Merge ``labels`` with the links (i, j); label = smallest member index."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    rows = np.concatenate([i, np.arange(n)])
    cols = np.concatenate([j, labels])
    graph = coo_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    first = np.full(comp.max() + 1 if n else 0, n, dtype=np.int64)
    np.minimum.at(first, comp, np.arange(n))
    return first[comp]


def _cluster_numpy(x, y, z, t, distance, time):
    n = t.size
    labels = np.arange(n)
    id2, it2 = 1.0 / (distance * distance), 1.0 / (time * time)
    pend_i, pend_j, pending = [], [], 0
    lag = 1
    while lag < n:
        dt = t[lag:] - t[:-lag]
        tt = dt * dt * it2
        near = tt <= 1.0
        if not near.any():
            break
        idx = np.nonzero(near)[0]
        dx = x[idx + lag] - x[idx]
        dy = y[idx + lag] - y[idx]
        dz = z[idx + lag] - z[idx]
        ok = (dx * dx + dy * dy + dz * dz) * id2 + tt[idx] <= 1.0
        i = idx[ok]
        j = i + lag
        keep = labels[i] != labels[j]
        pend_i.append(i[keep])
        pend_j.append(j[keep])
        pending += int(keep.sum())
        if pending > n or lag % 32 == 0:
            labels = _components(
                n, np.concatenate(pend_i), np.concatenate(pend_j), labels
            )
            pend_i, pend_j, pending = [], [], 0
        lag += 1
    if pend_i:
        labels = _components(n, np.concatenate(pend_i), np.concatenate(pend_j), labels)
    # number flashes by their first source
    _, first_pos, inverse = np.unique(labels, return_index=True, return_inverse=True)
    rank = np.empty(first_pos.size, dtype=np.int64)
    rank[np.argsort(first_pos)] = np.arange(first_pos.size)
    return rank[inverse].astype(np.int64)


def _hull_area(px, py):
    from scipy.spatial import ConvexHull, QhullError

    pts = np.unique(np.column_stack([px, py]), axis=0)
    if len(pts) < 3:
        return 0.0
    try:
        return float(ConvexHull(pts).volume)
    except QhullError:  # collinear points
        return 0.0


def _flash_stats_numpy(labels, nflash, x, y):
    ok = (labels >= 0) & (labels < nflash)
    lab = labels[ok]
    idx = np.nonzero(ok)[0]
    count = np.bincount(lab, minlength=nflash).astype(np.int64)
    first = np.full(nflash, np.iinfo(np.int64).max, dtype=np.int64)
    last = np.full(nflash, -1, dtype=np.int64)
    np.minimum.at(first, lab, idx)
    np.maximum.at(last, lab, idx)
    first[count == 0] = -1
    area = np.zeros(nflash)
    order = np.argsort(lab, kind="stable")
    splits = np.cumsum(count)[:-1]
    fin = np.isfinite(x[idx]) & np.isfinite(y[idx])
    for f, members in enumerate(np.split(order, splits)):
        members = members[fin[members]]
        if members.size >= 3:
            area[f] = _hull_area(x[idx[members]], y[idx[members]])
    return count, first, last, area


def _bins(edges, v):
    """Bin index of v in [edges[0], edges[-1]), or -1 (also for NaN)."""
    n = edges.size - 1
    k = np.searchsorted(edges, v, side="right") - 1
    with np.errstate(invalid="ignore"):
        inside = (v >= edges[0]) & (v < edges[-1])
    return np.where(inside & (k >= 0) & (k < n), k, -1)


def _pixels_numpy(x, y, z, t, xe, ye, ze, te):
    nx, ny, nz = xe.size - 1, ye.size - 1, ze.size - 1
    ix, iy, iz, it = _bins(xe, x), _bins(ye, y), _bins(ze, z), _bins(te, t)
    ok = (ix >= 0) & (iy >= 0) & (iz >= 0) & (it >= 0)
    return np.where(ok, ((it * nz + iz) * ny + iy) * nx + ix, -1)


def _grid_numpy(x, y, z, t, labels, first, flash_ok, xe, ye, ze, te):
    shape = (te.size - 1, ze.size - 1, ye.size - 1, xe.size - 1)
    size = int(np.prod(shape))
    pix = _pixels_numpy(x, y, z, t, xe, ye, ze, te)
    src = np.bincount(pix[pix >= 0], minlength=size)
    nflash = first.size
    in_flash = (labels >= 0) & (labels < nflash)
    use = in_flash & (pix >= 0)
    use[use] = flash_ok[labels[use]].astype(bool)
    pairs = np.unique(np.column_stack([labels[use], pix[use]]), axis=0)
    fed = np.bincount(pairs[:, 1], minlength=size) if pairs.size else np.zeros(size)
    good = flash_ok.astype(bool) & (first >= 0)
    fpix = pix[first[good]]
    fid = np.bincount(fpix[fpix >= 0], minlength=size)
    return tuple(a.reshape(shape).astype(np.int32) for a in (src, fed, fid))


def _cells_numpy(
    mask, ncell, xe, ye, frame, x, y, z, t, labels, first, flash_ok, te, ze, extent
):
    nframe, ny, nx = mask.shape
    nt, nz = te.size - 1, ze.size - 1
    ix, iy = _bins(xe, x), _bins(ye, y)
    ok = (frame >= 0) & (frame < nframe) & (ix >= 0) & (iy >= 0)
    cell = np.full(t.size, -1, dtype=np.int64)
    cell[ok] = mask[frame[ok], iy[ok], ix[ok]]
    cell[(cell < 0) | (cell >= ncell)] = -1
    it, iz = _bins(te, t), _bins(ze, z)
    s = (cell >= 0) & (it >= 0) & (iz >= 0)
    sources = np.bincount(
        (cell[s] * nt + it[s]) * nz + iz[s], minlength=ncell * nt * nz
    ).reshape(ncell, nt, nz)
    nflash = first.size
    good = flash_ok.astype(bool) & (first >= 0)
    fbin = np.full(nflash, -1, dtype=np.int64)
    fbin[good] = it[first[good]]
    if not extent:
        fcell = np.full(nflash, -1, dtype=np.int64)
        fcell[good] = cell[first[good]]
        use = (fcell >= 0) & (fbin >= 0)
        keys = fcell[use] * nt + fbin[use]
    else:
        in_flash = (labels >= 0) & (labels < nflash) & (cell >= 0)
        lab = labels[in_flash]
        pairs = np.unique(np.column_stack([lab, cell[in_flash]]), axis=0)
        if pairs.size:
            use = fbin[pairs[:, 0]] >= 0
            keys = pairs[use, 1] * nt + fbin[pairs[use, 0]]
        else:
            keys = np.zeros(0, dtype=np.int64)
    flashes = np.bincount(keys, minlength=ncell * nt).reshape(ncell, nt)
    return flashes.astype(np.int64), sources.astype(np.int64)


# --------------------------------------------------------------------------
# flashes
# --------------------------------------------------------------------------


def _sorted_events(ds):
    """Event variables sorted by time; returns (ds_sorted, order)."""
    t = ds["event_time"].values
    order = np.argsort(t, kind="stable")
    if np.all(order == np.arange(order.size)):
        return ds, order
    return ds.isel({EVENTS: order}), order


def _flash_arrays(ds, use_compiled, n_threads):
    """
    Flash index of every source, initiating source and source count per flash.

    ``ds`` must be sorted by time. Works with flashes from
    :func:`cluster_flashes` and with arbitrary flash ids (xlma-python).
    """
    ids = ds["event_parent_flash_id"].values
    nflash = ds.sizes.get(FLASHES, 0)
    own = (
        "flash_id" in ds.variables
        and ids.dtype.kind == "i"
        and np.array_equal(ds["flash_id"].values, np.arange(nflash))
    )
    if own:  # flashes numbered 0 .. n - 1, as from cluster_flashes
        labels = np.where((ids >= 0) & (ids < nflash), ids, -1).astype(np.int64)
    elif "flash_id" in ds.variables:
        flash_ids = np.asarray(ds["flash_id"].values)
        sorter = np.argsort(flash_ids, kind="stable")
        pos = np.searchsorted(flash_ids, ids, sorter=sorter)
        pos = np.clip(pos, 0, max(flash_ids.size - 1, 0))
        labels = np.where(
            flash_ids.size and (flash_ids[sorter[pos]] == ids), sorter[pos], -1
        ).astype(np.int64)
        nflash = flash_ids.size
    else:
        uniq, labels = np.unique(ids, return_inverse=True)
        labels = labels.astype(np.int64)
        nflash = uniq.size
    zeros = np.zeros(labels.size)
    if use_compiled:
        count, first, _, _ = _lightning.flash_stats(
            _i64(labels), nflash, zeros, zeros, n_threads=n_threads
        )
    else:
        count, first, _, _ = _flash_stats_numpy(labels, nflash, zeros, zeros)
    return labels, np.asarray(first), np.asarray(count)


def _with_flashes(ds, use_compiled, n_threads, distance=3000.0, time=0.15):
    """The dataset (sorted by time) with flashes, clustering if needed."""
    if "event_parent_flash_id" not in ds.variables:
        ds = cluster_flashes(
            ds,
            distance=distance,
            time=time,
            engine="compiled" if use_compiled else "numpy",
            n_threads=n_threads,
        )
    ds, _ = _sorted_events(ds)
    return ds


def cluster_flashes(ds, *, distance=3000.0, time=0.15, engine="auto", n_threads=None):
    """
    Group LMA VHF sources into flashes by their space-time separation.

    Parameters
    ----------
    ds : xarray.Dataset
        LMA sources, e.g. from :func:`radarx.io.read_lma` (filter them by
        chi-square and number of stations first, e.g. ``max_chi2=1``,
        ``min_stations=6``).
    distance : float, optional
        Distance scale :math:`d` in metres. Default 3000 m (Fuchs et al.
        2016 use 3 km for sensitive networks and 6 km for less sensitive
        ones).
    time : float, optional
        Time scale :math:`\\tau` in seconds. Default 0.15 s.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled
        kernel and falls back to NumPy.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.

    Returns
    -------
    xarray.Dataset
        The sources sorted by time with ``event_parent_flash_id`` and the
        flashes on ``number_of_flashes`` (``flash_id``, ``flash_time_start``,
        ``flash_time_end``, ``flash_duration``, ``flash_event_count``,
        ``flash_init_latitude/longitude/altitude``,
        ``flash_center_latitude/longitude/altitude``, ``flash_area``), numbered
        in the order of their first source. Existing flash variables are
        replaced.

    Notes
    -----
    Two sources are linked when
    :math:`(|\\Delta \\mathbf{x}| / d)^2 + (\\Delta t / \\tau)^2 \\le 1`, with
    :math:`\\Delta \\mathbf{x}` the straight-line (Earth-centred Cartesian)
    separation; flashes are the connected groups of linked sources (DBSCAN
    with a minimum of one point).

    References
    ----------
    Fuchs, B. R., E. C. Bruning, S. A. Rutledge, L. D. Carey, P. R.
    Krehbiel, and W. Rison, 2016: Climatological analyses of LMA data with an
    open-source lightning flash-clustering algorithm. *J. Geophys. Res.
    Atmos.*, **121** (14), 8625-8648, https://doi.org/10.1002/2015JD024663

    Bruning, E. C., and D. R. MacGorman, 2013: Theory and observations of
    controls on lightning flash size spectra. *J. Atmos. Sci.*, **70** (12),
    4012-4029, https://doi.org/10.1175/JAS-D-12-0289.1
    """
    use_compiled = _use_compiled(engine)
    _check_events(ds)
    if not (distance > 0 and time > 0):
        raise ValueError("distance and time must be positive")
    old = [v for v in ds.variables if FLASHES in ds[v].dims]
    ds = ds.drop_vars(old + [v for v in ["event_parent_flash_id"] if v in ds.variables])
    ds, _ = _sorted_events(ds)
    lat = ds["event_latitude"].values.astype(float)
    lon = ds["event_longitude"].values.astype(float)
    alt = ds["event_altitude"].values.astype(float)
    times = ds["event_time"].values
    n = times.size
    t = _seconds(times, times[0]) if n else np.zeros(0)
    ex, ey, ez = _ecef(lon, lat, alt)
    if n:
        ex, ey, ez = ex - ex[0], ey - ey[0], ez - ez[0]
    nt = int(n_threads or 0)
    if use_compiled:
        labels = _lightning.cluster(
            _f64(ex), _f64(ey), _f64(ez), _f64(t), float(distance), float(time), nt
        )
    else:
        labels = _cluster_numpy(_f64(ex), _f64(ey), _f64(ez), _f64(t), distance, time)
    labels = np.asarray(labels, dtype=np.int64)
    nflash = int(labels.max()) + 1 if n else 0
    # plan area in a local azimuthal equidistant projection
    if n:
        lat0, lon0 = float(np.nanmean(lat)), float(np.nanmean(lon))
        px, py = _project(lon, lat, lat0, lon0, nt)
    else:
        px = py = np.zeros(0)
    if use_compiled:
        count, first, last, area = _lightning.flash_stats(
            labels, nflash, _f64(px), _f64(py), n_threads=nt
        )
    else:
        count, first, last, area = _flash_stats_numpy(labels, nflash, px, py)
    first, last = np.asarray(first), np.asarray(last)

    def mean(v):
        return np.bincount(labels, weights=v, minlength=nflash) / np.maximum(count, 1)

    start, end = times[first], times[last]
    fl = {
        "flash_time_start": start,
        "flash_time_end": end,
        "flash_duration": (end - start).astype(np.int64) * 1e-9,
        "flash_event_count": np.asarray(count, dtype=np.int64),
        "flash_init_latitude": lat[first],
        "flash_init_longitude": lon[first],
        "flash_init_altitude": alt[first],
        "flash_center_latitude": mean(lat),
        "flash_center_longitude": mean(lon),
        "flash_center_altitude": mean(alt),
        "flash_area": np.asarray(area) * 1e-6,
    }
    out = ds.assign({k: (FLASHES, v, dict(_FLASH_ATTRS[k])) for k, v in fl.items()})
    out = out.assign_coords(
        flash_id=(FLASHES, np.arange(nflash, dtype=np.int64), _FLASH_ATTRS["flash_id"])
    )
    out["event_parent_flash_id"] = (
        EVENTS,
        labels,
        {"long_name": "Flash of the source (flash_id)", "cf_role": "tree_id"},
    )
    out.attrs.update(
        {
            "flash_algorithm_name": "radarx space-time clustering (Fuchs et al. 2016)",
            "flash_distance_separation_threshold": float(distance),
            "flash_time_separation_threshold": float(time),
        }
    )
    return out


# --------------------------------------------------------------------------
# gridding
# --------------------------------------------------------------------------


def _xy_of(grid, x, y):
    if grid is not None:
        if x is None:
            x = grid["x"].values
        if y is None:
            y = grid["y"].values
    if x is None or y is None:
        raise ValueError("give a grid or x and y coordinates")
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _flash_ok(count, min_sources):
    return np.ascontiguousarray(
        np.asarray(count) >= int(min_sources or 1), dtype=np.uint8
    )


def grid_lightning(
    ds,
    grid=None,
    *,
    x=None,
    y=None,
    z=None,
    interval="5min",
    time_edges=None,
    latitude=None,
    longitude=None,
    min_sources=10,
    distance=3000.0,
    time=0.15,
    engine="auto",
    n_threads=None,
):
    """
    Source, flash extent and flash initiation densities on a radarx grid.

    Parameters
    ----------
    ds : xarray.Dataset
        LMA sources (:func:`radarx.io.read_lma`), with or without flashes; if
        it has none, :func:`cluster_flashes` groups them first with
        ``distance`` and ``time``.
    grid : xarray.Dataset, optional
        A radarx grid (e.g. from :func:`radarx.grid.grid_cones`) whose ``x``,
        ``y`` (and, with ``z=True``, ``z``) coordinates and origin
        (``latitude``/``longitude`` coordinates or the ``crs_wkt``
        projection) define the grid.
    x, y : array-like, optional
        Grid box centres east and north of the origin (m, azimuthal
        equidistant), instead of or overriding those of ``grid``.
    z : array-like or bool, optional
        Heights above mean sea level (m) of 3-D grid boxes; ``True`` takes
        ``grid.z``. Default: count whole columns.
    interval : str or timedelta, optional
        Length of the time intervals, aligned to multiples of it. Default
        ``"5min"``.
    time_edges : array-like of datetime64, optional
        Explicit time interval edges instead of ``interval``.
    latitude, longitude : float, optional
        Origin of ``x``/``y``, if the grid does not give it.
    min_sources : int, optional
        Flashes with fewer sources are left out of the flash products.
        Default 10.
    distance, time : float, optional
        Clustering scales if ``ds`` has no flashes (see
        :func:`cluster_flashes`).
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. Default ``"auto"``.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.

    Returns
    -------
    xarray.Dataset
        ``source_density``, ``flash_extent_density`` and
        ``flash_initiation_density`` (counts per grid box and interval) on
        ``(time, [z,] y, x)``, with the time interval centres and
        ``time_bounds``, ``lat``/``lon`` axis coordinates and the origin.

    References
    ----------
    Bruning, E. C., and D. R. MacGorman, 2013: Theory and observations of
    controls on lightning flash size spectra. *J. Atmos. Sci.*, **70** (12),
    4012-4029, https://doi.org/10.1175/JAS-D-12-0289.1

    Examples
    --------
    >>> lma = radarx.io.read_lma(files, max_chi2=1.0, min_stations=6)  # doctest: +SKIP
    >>> fed = grid_lightning(lma, grid, interval="2min")  # doctest: +SKIP
    """
    use_compiled = _use_compiled(engine)
    _check_events(ds)
    nt = int(n_threads or 0)
    x, y = _xy_of(grid, x, y)
    if z is True:
        if grid is None or "z" not in grid.variables:
            raise ValueError("z=True needs a grid with a z coordinate")
        z = grid["z"].values
    lat0, lon0 = _origin(grid, latitude, longitude)
    ds = _with_flashes(ds, use_compiled, nt, distance, time)
    labels, first, count = _flash_arrays(ds, use_compiled, nt)
    ok = _flash_ok(count, min_sources)
    times = ds["event_time"].values
    tedges = _time_edges(times, interval, time_edges)
    ref = tedges[0]
    px, py = _project(
        ds["event_longitude"].values, ds["event_latitude"].values, lat0, lon0, nt
    )
    pz = ds["event_altitude"].values.astype(float)
    xe, ye = _edges(x, "x"), _edges(y, "y")
    ze = np.array([-np.inf, np.inf]) if z is None else _edges(z, "z")
    te = _seconds(tedges, ref)
    args = [
        _f64(px),
        _f64(py),
        _f64(pz),
        _f64(_seconds(times, ref)),
        _i64(labels),
        _i64(first),
        ok,
        _f64(xe),
        _f64(ye),
        _f64(ze),
        _f64(te),
    ]
    if use_compiled:
        src, fed, fid = _lightning.grid(*args, n_threads=nt)
    else:
        src, fed, fid = _grid_numpy(*args)
    dims = ("time", "z", "y", "x")
    sel = (slice(None), 0) if z is None else (slice(None),)
    if z is None:
        dims = ("time", "y", "x")
    step = (tedges[1] - tedges[0]).astype("timedelta64[s]")
    per = (
        f"per grid box per {step}"
        if np.all(np.diff(tedges) == tedges[1] - tedges[0])
        else "per grid box and interval"
    )
    names = {
        "source_density": ("Number of LMA VHF sources", src),
        "flash_extent_density": (
            "Number of flashes with a source in the grid box",
            fed,
        ),
        "flash_initiation_density": (
            "Number of flashes initiated in the grid box",
            fid,
        ),
    }
    data = {
        k: (dims, np.asarray(v)[sel], {"long_name": f"{ln} {per}", "units": "1"})
        for k, (ln, v) in names.items()
    }
    from ..grid.cone import _lonlat_axes

    lon_ax, lat_ax = _lonlat_axes(x, y, lat0, lon0)
    coords = dict(_time_coords(tedges))
    coords.update(
        {
            "y": ("y", y, {"units": "m", "long_name": "distance north of the origin"}),
            "x": ("x", x, {"units": "m", "long_name": "distance east of the origin"}),
            "lat": ("y", lat_ax, {"units": "degrees_north"}),
            "lon": ("x", lon_ax, {"units": "degrees_east"}),
            "latitude": (
                (),
                lat0,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "longitude": (
                (),
                lon0,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
        }
    )
    if z is not None:
        coords["z"] = (
            "z",
            np.asarray(z, dtype=float),
            {"units": "m", "long_name": "height above sea level"},
        )
    out = xr.Dataset(data, coords=coords)
    if grid is not None and "crs_wkt" in grid.variables:
        out = out.assign_coords(crs_wkt=grid["crs_wkt"])
    out.attrs = {
        "source": "VHF Lightning Mapping Array",
        "min_sources_per_flash": int(min_sources or 1),
        "flash_algorithm_name": ds.attrs.get("flash_algorithm_name", ""),
    }
    return out


def vertical_source_distribution(
    ds,
    z,
    *,
    interval=None,
    time_edges=None,
    min_sources=10,
    distance=3000.0,
    time=0.15,
    engine="auto",
    n_threads=None,
):
    """
    Height distribution of LMA sources and flash initiations.

    Parameters
    ----------
    ds : xarray.Dataset
        LMA sources (:func:`radarx.io.read_lma`), with or without flashes.
        Select a region first, e.g. with ``ds.where(..., drop=True)`` on
        ``event_latitude``/``event_longitude``.
    z : array-like
        Height bin centres above mean sea level (m).
    interval : str or timedelta, optional
        Length of the time intervals. Default: one interval spanning all
        sources.
    time_edges : array-like of datetime64, optional
        Explicit time interval edges.
    min_sources, distance, time, engine, n_threads
        As in :func:`grid_lightning`.

    Returns
    -------
    xarray.Dataset
        ``source_count`` and ``flash_initiation_count`` on ``(time, z)``.
    """
    use_compiled = _use_compiled(engine)
    _check_events(ds)
    nt = int(n_threads or 0)
    ds = _with_flashes(ds, use_compiled, nt, distance, time)
    labels, first, count = _flash_arrays(ds, use_compiled, nt)
    ok = _flash_ok(count, min_sources)
    times = ds["event_time"].values
    if interval is None and time_edges is None:
        if times.size == 0:
            raise ValueError("no sources; give time_edges")
        time_edges = np.array([times.min(), times.max() + np.timedelta64(1, "ns")])
    tedges = _time_edges(times, interval, time_edges)
    ref = tedges[0]
    n = times.size
    zero = np.zeros(n)
    inf = np.array([-np.inf, np.inf])
    zc = np.asarray(z, dtype=float)
    args = [
        zero,
        zero,
        _f64(ds["event_altitude"].values),
        _f64(_seconds(times, ref)),
        _i64(labels),
        _i64(first),
        ok,
        inf,
        inf,
        _f64(_edges(zc, "z")),
        _f64(_seconds(tedges, ref)),
    ]
    if use_compiled:
        src, _, fid = _lightning.grid(*args, n_threads=nt)
    else:
        src, _, fid = _grid_numpy(*args)
    coords = dict(_time_coords(tedges))
    coords["z"] = ("z", zc, {"units": "m", "long_name": "height above sea level"})
    return xr.Dataset(
        {
            "source_count": (
                ("time", "z"),
                np.asarray(src)[:, :, 0, 0],
                {"long_name": "Number of LMA VHF sources", "units": "1"},
            ),
            "flash_initiation_count": (
                ("time", "z"),
                np.asarray(fid)[:, :, 0, 0],
                {"long_name": "Number of flashes initiated", "units": "1"},
            ),
        },
        coords=coords,
    )


# --------------------------------------------------------------------------
# cells
# --------------------------------------------------------------------------


def cell_flash_rate(
    ds,
    mask,
    *,
    interval="1min",
    time_edges=None,
    z=None,
    count="initiation",
    max_offset=None,
    background=0,
    latitude=None,
    longitude=None,
    min_sources=10,
    distance=3000.0,
    time=0.15,
    engine="auto",
    n_threads=None,
):
    """
    Flash rates and source height distributions of tracked storm cells.

    Parameters
    ----------
    ds : xarray.Dataset
        LMA sources (:func:`radarx.io.read_lma`), with or without flashes.
    mask : xarray.DataArray
        Integer cell labels on ``(time, y, x)`` (e.g. a tracked-storm
        segmentation on a radarx grid; the same label is the same cell at
        all times), with ``x``/``y`` in metres east and north of the origin.
        ``background`` (and negative or NaN values) mark no cell. Each source
        is attributed to the mask frame nearest in time.
    interval : str or timedelta, optional
        Length of the flash-rate intervals. Default ``"1min"``.
    time_edges : array-like of datetime64, optional
        Explicit interval edges. Default: intervals covering the mask times.
    z : array-like, optional
        Height bin centres (m above mean sea level) for the source counts by
        height (vertical source distribution of every cell).
    count : {"initiation", "extent"}, optional
        Count a flash in the cell of its first source (default) or in every
        cell any of its sources falls in. A flash is counted in the interval
        of its first source.
    max_offset : str or timedelta, optional
        Largest time difference between a source and its mask frame.
        Default: half the median spacing of the mask times.
    background : int, optional
        Label of no cell. Default 0.
    latitude, longitude : float, optional
        Origin of ``x``/``y``, if the mask does not give it (coordinates
        ``latitude``/``longitude``, ``origin_latitude``/``origin_longitude``
        or ``crs_wkt``).
    min_sources, distance, time, engine, n_threads
        As in :func:`grid_lightning`.

    Returns
    -------
    xarray.Dataset
        ``flash_count`` and ``flash_rate`` (flashes min\\ :sup:`-1`) on
        ``(cell, time)``, and with ``z`` the ``source_count`` on
        ``(cell, time, z)``; ``cell`` holds the mask labels.
    """
    use_compiled = _use_compiled(engine)
    _check_events(ds)
    if count not in ("initiation", "extent"):
        raise ValueError("count must be 'initiation' or 'extent'")
    if not isinstance(mask, xr.DataArray) or set(mask.dims) != {"time", "y", "x"}:
        raise ValueError("mask must be a DataArray on (time, y, x)")
    nt = int(n_threads or 0)
    mask = mask.transpose("time", "y", "x")
    lat0, lon0 = _origin(mask, latitude, longitude)
    values = mask.values
    valid = (
        np.isfinite(values) if values.dtype.kind == "f" else np.ones(values.shape, bool)
    )
    filled = np.where(valid, values, background).astype(np.int64)
    valid &= (filled != background) & (filled >= 0)
    cells = np.unique(filled[valid])
    index = np.where(valid, np.searchsorted(cells, filled), -1).astype(np.int32)

    ds = _with_flashes(ds, use_compiled, nt, distance, time)
    labels, first, fcount = _flash_arrays(ds, use_compiled, nt)
    ok = _flash_ok(fcount, min_sources)
    times = ds["event_time"].values
    mtimes = mask["time"].values.astype("datetime64[ns]")
    if mtimes.size > 1 and np.any(np.diff(mtimes) <= np.timedelta64(0, "ns")):
        raise ValueError("mask times must increase")
    if max_offset is None:
        spacing = (
            np.median(np.diff(mtimes)) if mtimes.size > 1 else np.timedelta64(5, "m")
        )
        max_off = (spacing // 2).astype("timedelta64[ns]")
    else:
        max_off = _interval(max_offset, "max_offset")
    if times.size:
        mid = mtimes[:-1] + (mtimes[1:] - mtimes[:-1]) // 2
        frame = np.searchsorted(mid, times, side="right").astype(np.int64)
        frame[np.abs(times - mtimes[frame]) > max_off] = -1
    else:
        frame = np.zeros(0, dtype=np.int64)
    if time_edges is None:
        step = _interval(interval)
        span = np.array(
            [mtimes.min() - max_off, mtimes.max() + max_off - np.timedelta64(1, "ns")]
        )
        tedges = _time_edges(span, step, None)
    else:
        tedges = _time_edges(times, interval, time_edges)
    ref = tedges[0]
    px, py = _project(
        ds["event_longitude"].values, ds["event_latitude"].values, lat0, lon0, nt
    )
    zc = None if z is None else np.asarray(z, dtype=float)
    ze = np.array([-np.inf, np.inf]) if zc is None else _edges(zc, "z")
    args = [
        np.ascontiguousarray(index),
        int(cells.size),
        _f64(_edges(mask["x"].values, "x")),
        _f64(_edges(mask["y"].values, "y")),
        _i64(frame),
        _f64(px),
        _f64(py),
        _f64(ds["event_altitude"].values),
        _f64(_seconds(times, ref)),
        _i64(labels),
        _i64(first),
        ok,
        _f64(_seconds(tedges, ref)),
        _f64(ze),
        count == "extent",
    ]
    if use_compiled:
        flashes, sources = _lightning.cells(*args, n_threads=nt)
    else:
        flashes, sources = _cells_numpy(*args)
    minutes = (np.diff(tedges).astype(np.int64) * 1e-9 / 60.0)[np.newaxis, :]
    coords = dict(_time_coords(tedges))
    coords["cell"] = ("cell", cells, {"long_name": "Cell label of the mask"})
    data = {
        "flash_count": (
            ("cell", "time"),
            np.asarray(flashes),
            {"long_name": "Number of flashes of the cell", "units": "1"},
        ),
        "flash_rate": (
            ("cell", "time"),
            np.asarray(flashes) / minutes,
            {"long_name": "Total flash rate of the cell", "units": "min-1"},
        ),
    }
    if zc is not None:
        coords["z"] = ("z", zc, {"units": "m", "long_name": "height above sea level"})
        data["source_count"] = (
            ("cell", "time", "z"),
            np.asarray(sources),
            {"long_name": "Number of LMA VHF sources of the cell", "units": "1"},
        )
    out = xr.Dataset(data, coords=coords)
    out.attrs = {
        "flash_count_method": count,
        "min_sources_per_flash": int(min_sources or 1),
    }
    return out


# --------------------------------------------------------------------------
# lightning jump
# --------------------------------------------------------------------------


def _jump_series(rate, period_min, sigma, min_rate, history, group, ddof):
    n = rate.size
    dfrdt = np.full(n, np.nan)
    dfrdt[1:] = (rate[1:] - rate[:-1]) / period_min
    level = np.full(n, np.nan)
    jump = np.zeros(n, dtype=bool)
    start = np.zeros(n, dtype=bool)
    active = False
    last_start = -np.inf
    for k in range(history + 1, n):
        prev = dfrdt[k - history : k]
        if np.all(np.isfinite(prev)) and np.isfinite(dfrdt[k]):
            sd = np.std(prev, ddof=ddof)
            if sd > 0:
                level[k] = dfrdt[k] / sd
            elif dfrdt[k] != 0:
                level[k] = np.inf * np.sign(dfrdt[k])
        lv = level[k]
        if active:
            if np.isnan(lv) or lv < 0:
                active = False
            else:
                jump[k] = True
                continue
        if lv >= sigma and rate[k] >= min_rate:
            active = True
            jump[k] = True
            if k - last_start > group:
                start[k] = True
            last_start = k
    return dfrdt, level, jump, start


def lightning_jump(
    flash_rate,
    *,
    period="2min",
    sigma=2.0,
    min_rate=10.0,
    history=5,
    group="6min",
    ddof=1,
):
    """
    Lightning jumps with the "2σ" algorithm of Schultz et al. (2009).

    Parameters
    ----------
    flash_rate : xarray.DataArray
        Total flash rate (flashes min\\ :sup:`-1`) on a regularly spaced
        ``time`` dimension, e.g. ``cell_flash_rate(...).flash_rate``; other
        dimensions (cells) are processed independently.
    period : str or timedelta, optional
        Averaging period of the flash rate. Default ``"2min"``.
    sigma : float, optional
        Sigma level of a jump. Default 2.
    min_rate : float, optional
        Flash rate (flashes min\\ :sup:`-1`) the averaged rate must reach for
        a jump. Default 10.
    history : int, optional
        Number of previous ``DFRDT`` values of the standard deviation.
        Default 5 (10 min, a 14-min spin-up with the current period).
    group : str or timedelta, optional
        Jumps starting within this time of the start of an earlier one are
        not new jumps. Default ``"6min"``.
    ddof : int, optional
        Delta degrees of freedom of the standard deviation. Default 1
        (sample standard deviation).

    Returns
    -------
    xarray.Dataset
        On the averaging periods (``time``: first input time of each
        period): ``flash_rate`` (averaged), ``dfrdt`` (flashes min\\ :sup:`-2`), ``sigma_level``,
        ``jump`` (a jump is in progress) and ``jump_start`` (first period of
        a new jump).

    References
    ----------
    Schultz, C. J., W. A. Petersen, and L. D. Carey, 2009: Preliminary
    development and evaluation of lightning jump algorithms for the
    real-time detection of severe weather. *J. Appl. Meteor. Climatol.*,
    **48** (12), 2543-2563, https://doi.org/10.1175/2009JAMC2237.1

    Schultz, C. J., W. A. Petersen, and L. D. Carey, 2011: Lightning and
    severe weather: A comparison between total and cloud-to-ground lightning
    trends. *Wea. Forecasting*, **26** (5), 744-755,
    https://doi.org/10.1175/WAF-D-10-05026.1

    Schultz, E. V., C. J. Schultz, L. D. Carey, D. J. Cecil, and M. Bateman,
    2016: Automated storm tracking and the lightning jump algorithm using
    GOES-R Geostationary Lightning Mapper (GLM) proxy data. *J. Operational
    Meteor.*, **4** (7), 92-107, https://doi.org/10.15191/nwajom.2016.0407
    """
    if not isinstance(flash_rate, xr.DataArray) or "time" not in flash_rate.dims:
        raise ValueError("flash_rate must be a DataArray with a time dimension")
    if int(history) < 2:
        raise ValueError("history must be at least 2")
    step = _interval(period, "period")
    gap = _interval(group, "group")
    times = flash_rate["time"].values.astype("datetime64[ns]")
    if times.size < 2:
        raise ValueError("flash_rate needs at least two times")
    avg = flash_rate.resample(
        time=pd.Timedelta(step), origin=pd.Timestamp(times[0])
    ).mean()
    period_min = step.astype(np.int64) * 1e-9 / 60.0
    group_n = int(gap.astype(np.int64) // step.astype(np.int64))
    core = avg.transpose(..., "time")
    arr = core.values
    flat = arr.reshape(-1, arr.shape[-1])
    outs = [np.empty(flat.shape, dtype=d) for d in (float, float, bool, bool)]
    for i, series in enumerate(flat):
        res = _jump_series(
            series.astype(float),
            period_min,
            float(sigma),
            float(min_rate),
            int(history),
            group_n,
            ddof,
        )
        for o, r in zip(outs, res):
            o[i] = r
    dims = core.dims
    shape = arr.shape
    attrs = {
        "flash_rate": {
            "long_name": f"Total flash rate averaged over {step.astype('timedelta64[s]')}",
            "units": "min-1",
        },
        "dfrdt": {
            "long_name": "Time rate of change of the total flash rate",
            "units": "min-2",
        },
        "sigma_level": {
            "long_name": "DFRDT divided by the standard deviation of the "
            f"{int(history)} previous DFRDT",
            "units": "1",
        },
        "jump": {"long_name": "Lightning jump in progress"},
        "jump_start": {"long_name": "Start of a lightning jump"},
    }
    data = {"flash_rate": (dims, arr, attrs["flash_rate"])}
    for name, o in zip(("dfrdt", "sigma_level", "jump", "jump_start"), outs):
        data[name] = (dims, o.reshape(shape), attrs[name])
    out = xr.Dataset(data, coords=core.coords)
    out.attrs = {
        "algorithm": "Schultz et al. (2009) sigma lightning jump",
        "sigma": float(sigma),
        "min_rate": float(min_rate),
        "history": int(history),
    }
    return out


# --------------------------------------------------------------------------
# accessors
# --------------------------------------------------------------------------


@accessor_method("dataset", name="cluster_flashes")
def _cluster_flashes_accessor(self, **kwargs):
    """
    Group the LMA sources of this dataset into flashes.

    See :func:`radarx.retrieve.cluster_flashes` for the parameters.

    Returns
    -------
    xarray.Dataset
        The sources with ``event_parent_flash_id`` and the flash variables.
    """
    return cluster_flashes(self.xarray_obj, **kwargs)


@accessor_method("dataset", name="grid_lightning")
def _grid_lightning_accessor(self, grid=None, **kwargs):
    """
    Grid the LMA sources of this dataset (source, flash extent and flash
    initiation densities).

    See :func:`radarx.retrieve.grid_lightning` for the parameters.

    Returns
    -------
    xarray.Dataset
        Gridded lightning products on ``(time, [z,] y, x)``.
    """
    return grid_lightning(self.xarray_obj, grid, **kwargs)


@accessor_method("dataarray", name="lightning_jump")
def _lightning_jump_accessor(self, **kwargs):
    """
    Lightning jumps of this flash-rate time series (Schultz et al. 2009).

    See :func:`radarx.retrieve.lightning_jump` for the parameters.

    Returns
    -------
    xarray.Dataset
        Averaged flash rate, DFRDT, sigma level and jump flags.
    """
    return lightning_jump(self.xarray_obj, **kwargs)
