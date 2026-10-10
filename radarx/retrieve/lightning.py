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
space and time, with the normalization of Fuchs et al. (2016) [1]_ (section
2.1, p. 8628; the weighted Euclidean distance of Mach et al. 2007 [7]_): the
source positions are converted to Earth-centred Cartesian
coordinates, divided by a distance scale (default 3 km) and the source times
by a time scale (default 0.15 s), and two sources belong to the same flash
when their normalized space-time distance is at most one (Fuchs et al. use
:math:`\\epsilon = 1` on the same coordinates),

.. math::

    \\left(\\frac{|\\mathbf{x}_i - \\mathbf{x}_j|}{d}\\right)^2 +
    \\left(\\frac{t_i - t_j}{\\tau}\\right)^2 \\le 1 .

Flashes are the connected groups of such links (single linkage). This is what
DBSCAN [2]_ gives with a minimum number of points :math:`N_{min}` of one or
two, and only then. Fuchs et al. (2016) [1]_ (sections 2.1 and 2.2, pp.
8628-8629) ran the scikit-learn DBSCAN with :math:`\\epsilon = 1` and
:math:`N_{min} = 2` (Alabama, Washington D.C.) and :math:`N_{min} = 10`
(Colorado). A flash then starts from a core source with at least
:math:`N_{min}` sources within :math:`\\epsilon`, and an outer source must lie
near a core source; for :math:`N_{min} > 2` this is stricter than the
dot-to-dot linking used here, which Fuchs et al. contrast with DBSCAN (p.
8628). Singletons and clusters of fewer than :math:`N_{min}` sources ("noise" in
DBSCAN) are kept as small flashes. radarx has no :math:`N_{min}` argument, so
it reproduces the Alabama and D.C. setting but not the Colorado one: on 150
synthetic random-walk flashes (check of issue 178) scikit-learn DBSCAN with
:math:`N_{min}` = 1 and 2 gave the same 622 flashes as radarx, with 3 it gave
707 and with 10 it gave 3607 (counts depend on the synthetic sample; an
independent repeat gave identical partitions for 1 and 2 and more flashes for
3 and 10).

Fuchs et al. (2016) also impose, by default, an "arbitrary 3 s maximum
duration" on the flashes, because their streamed processing clusters a buffer
of twice that length (p. 8629); flashes longer than 3 s can be split by it, so
it is a limit that changes the result, not only a memory saving. radarx imposes
no maximum duration. The sources should be filtered first as in Fuchs et al.
(at least six stations by default and :math:`\\chi^2 \\le 1`, p. 8628, e.g.
with the ``min_stations`` and ``max_chi2`` arguments of
:func:`radarx.io.read_lma`).
Flashes with few sources are kept and left out later (``min_sources``), as
is customary (Fuchs et al. 2016, p. 8629). For every flash the start and end
time, duration, number of sources, initiation point (first source),
centroid (mean of the source latitudes, longitudes and altitudes; a radarx
definition) and plan area (convex hull of the sources seen from above, "the
area enclosed by a rubber band wrapped around the flash viewed from above",
Fuchs et al. 2016, p. 8626, after Bruning and MacGorman 2013 [3]_) are
computed.

Gridded products
----------------
:func:`grid_lightning` counts, in every grid box of a radarx grid (or any
``x``/``y`` grid east and north of an origin, azimuthal equidistant as in
:func:`radarx.grid.grid_cones`) and time interval:

- ``source_density``: VHF sources;
- ``flash_extent_density``: flashes with at least one source in the box (the
  definition used here; the product is attributed to Bruning and MacGorman
  2013 [3]_, whose text is not checked, so the attribution is
  unconfirmed);
- ``flash_initiation_density``: flashes whose first source is in the box.

With heights ``z`` the counts are per 3-D box (lightning on the radar grid);
without, per column. The default ``min_sources=10`` follows Schultz et al.
(2011) [5]_ (section 2b, p. 747), who required "a minimum of 10 VHF source
points" per flash to remove spurious noise points (Fuchs et al. 2016 [1]_
use :math:`N_{min} = 10` for Colorado). The 5-min interval, the box edges
midway between grid centres and the azimuthal equidistant projection are
radarx choices without a source.

Storm cells and lightning jumps
-------------------------------
:func:`cell_flash_rate` counts the flashes (by initiation point, or every
cell a flash touches) and the sources by height in every cell of a
time-dependent cell mask, e.g. a tracked-storm segmentation, giving flash
rates per cell and vertical source distributions.
:func:`vertical_source_distribution` gives the height distribution of all
sources and flash initiations.

:func:`lightning_jump` follows the "2σ" lightning jump algorithm of Schultz
et al. (2009) [4]_ as listed step by step by Schultz et al. (2011, section 2c,
pp. 747-748; appendix, p. 753) [5]_ and Schultz et al. (2016, section 2c,
pp. 97-98) [6]_:

1. the total flash rate is averaged over 2-min periods ([5]_ step (i), [6]_
   step 1);
2. the rate of change ``DFRDT`` is the difference between consecutive
   periods divided by the period (flashes min\\ :sup:`-2`; [4]_ Eq. 3, [5]_
   Eq. A2, [6]_ step 2);
3. :math:`\\sigma` is the standard deviation of the five previous ``DFRDT``
   values ([4]_ p. 2549: the five most recent periods "not including the
   period of interest"), and the sigma level is ``DFRDT`` / :math:`\\sigma`
   ([6]_ step 4);
4. a jump occurs when the sigma level reaches 2 while the averaged flash rate
   is at least 10 flashes min\\ :sup:`-1`, after a spin-up of 14 min: six
   2-min periods give the five ``DFRDT`` values and the seventh is the
   current one ([6]_ step 5);
5. a jump lasts until the sigma level drops below zero ([6]_), and jumps
   starting within 6 min of an earlier one are one jump ([4]_ p. 2550, [6]_
   step 6).

Where the algorithm is not fully specified in the papers, or the papers
disagree, the choices made here are listed in the Notes of
:func:`lightning_jump` (strict or non-strict inequalities, end of a jump,
sample or population standard deviation, zero standard deviation).

The heavy loops (clustering, flash properties, gridding, cell counts) run in
a compiled kernel (``radarx.retrieve._lightning``, multithreaded over sources
and flashes) with an identical NumPy reference implementation as fallback.

References
----------
.. [1] Fuchs, B. R., E. C. Bruning, S. A. Rutledge, L. D. Carey, P. R.
   Krehbiel, and W. Rison, 2016: Climatological analyses of LMA data with an
   open-source lightning flash-clustering algorithm. *J. Geophys. Res.
   Atmos.*, **121** (14), 8625-8648, https://doi.org/10.1002/2015JD024663
.. [2] Ester, M., H.-P. Kriegel, J. Sander, and X. Xu, 1996: A density-based
   algorithm for discovering clusters in large spatial databases with noise.
   *Proc. Second Int. Conf. on Knowledge Discovery and Data Mining (KDD-96)*,
   AAAI Press, 226-231 (no DOI; reference as listed in [1]_).
.. [3] Bruning, E. C., and D. R. MacGorman, 2013: Theory and observations of
   controls on lightning flash size spectra. *J. Atmos. Sci.*, **70** (12),
   4012-4029, https://doi.org/10.1175/JAS-D-12-0289.1
.. [4] Schultz, C. J., W. A. Petersen, and L. D. Carey, 2009: Preliminary
   development and evaluation of lightning jump algorithms for the real-time
   detection of severe weather. *J. Appl. Meteor. Climatol.*, **48** (12),
   2543-2563, https://doi.org/10.1175/2009JAMC2237.1
.. [5] Schultz, C. J., W. A. Petersen, and L. D. Carey, 2011: Lightning and
   severe weather: A comparison between total and cloud-to-ground lightning
   trends. *Wea. Forecasting*, **26** (5), 744-755,
   https://doi.org/10.1175/WAF-D-10-05026.1
.. [6] Schultz, E. V., C. J. Schultz, L. D. Carey, D. J. Cecil, and M.
   Bateman, 2016: Automated storm tracking and the lightning jump algorithm
   using GOES-R Geostationary Lightning Mapper (GLM) proxy data. *J.
   Operational Meteor.*, **4** (7), 92-107,
   https://doi.org/10.15191/nwajom.2016.0407
.. [7] Mach, D. M., H. J. Christian, R. J. Blakeslee, D. J. Boccippio, S. J.
   Goodman, and W. L. Boeck, 2007: Performance assessment of the Optical
   Transient Detector and Lightning Imaging Sensor. *J. Geophys. Res.*,
   **112**, D09210, https://doi.org/10.1029/2006JD007787 (cited by [1]_ for
   the normalization; its content is not checked)

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

from .._provenance import provenance
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
    # Flashes = connected components of the links (dx / d)^2 + (dt / tau)^2 <= 1
    # (single linkage), the DBSCAN clustering (Ester et al. 1996) with
    # eps = 1 on the normalized coordinates and a minimum of 1 or 2 points. It
    # equals the flash sorter of Fuchs et al. (2016, pp. 8628-8629) for their
    # N_min = 2 (Alabama, D.C.), not for their N_min = 10 (Colorado), and it
    # has no maximum flash duration (Fuchs et al. impose 3 s by default, which
    # can split longer flashes; see issue 178). Sources are sorted by time, so
    # source i only needs partners j > i with t_j - t_i <= tau; the loop
    # runs over the index lag j - i and stops when no pair is within tau.
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
    source_keys = (cell[s] * nt + it[s]) * nz + iz[s]
    sources = np.bincount(source_keys, minlength=ncell * nt * nz)
    sources = sources.reshape((ncell, nt, nz))
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
    flashes = np.bincount(keys, minlength=ncell * nt).reshape((ncell, nt))
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


@provenance(
    "Space-time clustering of LMA sources into flashes, after Fuchs et al. (2016)",
    extra_refs=("ester-1996",),
)
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
        Distance scale :math:`d` in metres. Default 3000 m: Fuchs et al.
        (2016) [1]_ (p. 8628) took 3 km for the sensitive Colorado network
        and 6 km for the less sensitive Alabama and D.C. networks, "in
        accordance with values from other algorithms" there.
    time : float, optional
        Time scale :math:`\\tau` in seconds. Default 0.15 s, the 150 ms of
        Fuchs et al. (2016) [1]_ (p. 8628).
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
    :math:`\\Delta \\mathbf{x}` the straight-line (Earth-centred Cartesian,
    WGS84) separation; flashes are the connected groups of linked sources
    (single linkage).

    Relation to Fuchs et al. (2016). Single linkage is the clustering of
    DBSCAN [2]_ with :math:`\\epsilon = 1` on these normalized coordinates and
    a minimum number of points :math:`N_{min}` (scikit-learn ``min_samples``,
    the source itself counted) of 1 or 2, and not for larger :math:`N_{min}`.
    Fuchs et al. [1]_ (sections 2.1 and 2.2, pp. 8628-8629) used
    :math:`N_{min} = 2` for Alabama and D.C. and :math:`N_{min} = 10` for
    Colorado. With :math:`N_{min} = 10` a flash grows only from core sources
    with at least ten sources within :math:`\\epsilon` and an outer source must
    be within :math:`\\epsilon` of a core source, unlike the "simple dot-to-dot
    algorithm" that this function implements; sources of clusters smaller than
    :math:`N_{min}` are kept as small flashes in both. radarx has no
    :math:`N_{min}` argument. Check on 150 synthetic random-walk flashes
    (issue 178), scikit-learn ``DBSCAN(eps=1, min_samples=N_min)`` on the same
    normalized coordinates, noise points turned into one-source flashes:
    ``N_min`` = 1 and 2 give the same partition as this function (622
    flashes), ``N_min`` = 3 gives 707 and ``N_min`` = 10 gives 3607 flashes
    against 622 here, so the Colorado setting of Fuchs et al. is not
    reproduced (the counts depend on the synthetic sample; an independent
    repeat on another sample gave identical partitions for 1 and 2 and more
    flashes for 3 and 10).

    Maximum duration. Fuchs et al. [1]_ (p. 8629) impose "by default an
    arbitrary 3 s maximum duration" on flashes, because they cluster a buffer
    of twice that duration in a stream; a flash longer than 3 s can be split
    by this. It is an imposed limit that changes which sources are grouped.
    No maximum duration is imposed here (flashes can be longer than 3 s).

    The flash centroid is the mean of the source latitudes, longitudes and
    altitudes and the plan area is the convex hull of the sources in an
    azimuthal equidistant projection about their mean position (both radarx
    definitions; Fuchs et al. [1]_, p. 8626, describe the plan area as a
    rubber band around the flash seen from above, after Bruning and
    MacGorman [3]_, not checked against the paper). Fuchs et
    al. measure the normalization from the LMA coordinate centre; here the
    first source is subtracted, which does not change any separation. The
    normalization of the space and time axes is the weighted Euclidean
    distance of Mach et al. [4]_ as used by Fuchs et al. [1]_ (p. 8628; the
    paper of Mach et al. is not checked). The plan-area hull is
    computed with Qhull (Barber et al. [6]_) in the NumPy path and with
    Andrew's monotone chain [5]_ in the compiled kernel; both give the area of
    the same convex polygon.

    References
    ----------
    .. [1] Fuchs, B. R., E. C. Bruning, S. A. Rutledge, L. D. Carey, P. R.
       Krehbiel, and W. Rison, 2016: Climatological analyses of LMA data
       with an open-source lightning flash-clustering algorithm. *J.
       Geophys. Res. Atmos.*, **121** (14), 8625-8648,
       https://doi.org/10.1002/2015JD024663
    .. [2] Ester, M., H.-P. Kriegel, J. Sander, and X. Xu, 1996: A
       density-based algorithm for discovering clusters in large spatial
       databases with noise. *Proc. Second Int. Conf. on Knowledge Discovery
       and Data Mining (KDD-96)*, AAAI Press, 226-231 (no DOI; reference as
       listed in [1]_).
    .. [3] Bruning, E. C., and D. R. MacGorman, 2013: Theory and
       observations of controls on lightning flash size spectra. *J. Atmos.
       Sci.*, **70** (12), 4012-4029, https://doi.org/10.1175/JAS-D-12-0289.1
    .. [4] Mach, D. M., H. J. Christian, R. J. Blakeslee, D. J. Boccippio, S.
       J. Goodman, and W. L. Boeck, 2007: Performance assessment of the
       Optical Transient Detector and Lightning Imaging Sensor. *J. Geophys.
       Res.*, **112**, D09210, https://doi.org/10.1029/2006JD007787
    .. [5] Andrew, A. M., 1979: Another efficient algorithm for convex hulls
       in two dimensions. *Inf. Process. Lett.*, **9** (5), 216-219,
       https://doi.org/10.1016/0020-0190(79)90072-3
    .. [6] Barber, C. B., D. P. Dobkin, and H. Huhdanpaa, 1996: The quickhull
       algorithm for convex hulls. *ACM Trans. Math. Softw.*, **22** (4),
       469-483, https://doi.org/10.1145/235815.235821
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


@provenance("Gridded source, flash extent and flash initiation densities")
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
        Default 10, the minimum number of VHF sources per flash required by
        Schultz et al. (2011) [2]_ (section 2b, p. 747) to remove spurious
        noise points; Fuchs et al. (2016) [1]_ (p. 8629) describe such a
        threshold on the number of points as customary.
    distance, time : float, optional
        Clustering scales if ``ds`` has no flashes (see
        :func:`cluster_flashes`, which also states how the flashes relate
        to Fuchs et al. 2016).
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

    Notes
    -----
    Flash extent density counts the flashes with at least one source in a
    grid box; the product is attributed to Bruning and MacGorman (2013) [3]_,
    not checked against the paper. The
    5-min default interval, the box edges midway between grid centres and
    the azimuthal equidistant projection of the sources about the grid
    origin are radarx choices.

    References
    ----------
    .. [1] Fuchs, B. R., E. C. Bruning, S. A. Rutledge, L. D. Carey, P. R.
       Krehbiel, and W. Rison, 2016: Climatological analyses of LMA data
       with an open-source lightning flash-clustering algorithm. *J.
       Geophys. Res. Atmos.*, **121** (14), 8625-8648,
       https://doi.org/10.1002/2015JD024663
    .. [2] Schultz, C. J., W. A. Petersen, and L. D. Carey, 2011: Lightning
       and severe weather: A comparison between total and cloud-to-ground
       lightning trends. *Wea. Forecasting*, **26** (5), 744-755,
       https://doi.org/10.1175/WAF-D-10-05026.1
    .. [3] Bruning, E. C., and D. R. MacGorman, 2013: Theory and
       observations of controls on lightning flash size spectra. *J. Atmos.
       Sci.*, **70** (12), 4012-4029, https://doi.org/10.1175/JAS-D-12-0289.1

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


@provenance("Height distribution of LMA sources and flash initiations")
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

    Notes
    -----
    Flashes and the ``min_sources`` default are as in :func:`grid_lightning`
    and :func:`cluster_flashes`, with their references.
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


def _cell_index(values, background):
    """Cell labels of a mask and the cell index (or -1) of every mask point."""
    valid = (
        np.isfinite(values) if values.dtype.kind == "f" else np.ones(values.shape, bool)
    )
    filled = np.where(valid, values, background).astype(np.int64)
    valid &= (filled != background) & (filled >= 0)
    cells = np.unique(filled[valid])
    index = np.where(valid, np.searchsorted(cells, filled), -1).astype(np.int32)
    return cells, np.ascontiguousarray(index)


def _mask_frames(times, mtimes, max_offset):
    """Nearest mask frame of every source time (-1 if too far) and the offset."""
    if mtimes.size > 1 and np.any(np.diff(mtimes) <= np.timedelta64(0, "ns")):
        raise ValueError("mask times must increase")
    if max_offset is not None:
        max_off = _interval(max_offset, "max_offset")
    elif mtimes.size > 1:
        max_off = (np.median(np.diff(mtimes)) // 2).astype("timedelta64[ns]")
    else:
        max_off = np.timedelta64(150, "s").astype("timedelta64[ns]")
    if not times.size:
        return np.zeros(0, dtype=np.int64), max_off
    mid = mtimes[:-1] + (mtimes[1:] - mtimes[:-1]) // 2
    frame = np.searchsorted(mid, times, side="right").astype(np.int64)
    frame[np.abs(times - mtimes[frame]) > max_off] = -1
    return frame, max_off


@provenance("Flash rates and source heights of tracked storm cells")
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

    Notes
    -----
    Flashes, ``min_sources`` and the clustering scales are as in
    :func:`cluster_flashes` and :func:`grid_lightning` (Fuchs et al. 2016;
    Schultz et al. 2011, with their references there). The 1-min interval,
    the nearest-frame attribution of sources and the default ``max_offset``
    (half the median mask spacing, or 150 s for a single frame) are radarx
    choices without a published source. Flash rates of tracked cells are
    the input of :func:`lightning_jump` (Schultz et al. 2009, 2011, 2016).
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
    cells, index = _cell_index(mask.values, background)

    ds = _with_flashes(ds, use_compiled, nt, distance, time)
    labels, first, fcount = _flash_arrays(ds, use_compiled, nt)
    ok = _flash_ok(fcount, min_sources)
    times = ds["event_time"].values
    mtimes = mask["time"].values.astype("datetime64[ns]")
    frame, max_off = _mask_frames(times, mtimes, max_offset)
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
        index,
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


def _sigma_level(dfrdt, history, ddof):
    """
    DFRDT over the standard deviation of the ``history`` previous values.

    The standard deviation is the sample one by default (``ddof=1``), a radarx
    choice that the papers (Schultz et al. 2009, 2011, 2016) do not settle.
    With a zero standard deviation the level is infinite with the sign of
    DFRDT (NaN if DFRDT is also zero).
    """
    n = dfrdt.size
    level = np.full(n, np.nan)
    if n <= history + 1:
        return level
    prev = np.lib.stride_tricks.sliding_window_view(dfrdt[:-1], history)[1:]
    cur = dfrdt[history + 1 :]
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.std(prev, axis=-1, ddof=ddof)
        lv = np.where(sd > 0, cur / sd, np.inf * np.sign(cur))
    ok = np.all(np.isfinite(prev), axis=-1) & np.isfinite(cur)
    level[history + 1 :] = np.where(ok & ((sd > 0) | (cur != 0)), lv, np.nan)
    return level


def _jump_series(rate, period_min, sigma, min_rate, history, group, ddof):
    # Conventions (see the Notes of lightning_jump): the trigger
    # uses >= although Schultz et al. (2009, 2011) say "exceeds"; a jump ends
    # when the sigma level is below zero (Schultz et al. 2016, step 5), not at
    # DFRDT <= 0 as in Schultz et al. (2009, p. 2550; 2011, p. 748).
    dfrdt = np.full(rate.size, np.nan)
    dfrdt[1:] = (rate[1:] - rate[:-1]) / period_min
    level = _sigma_level(dfrdt, history, ddof)
    trigger = (level >= sigma) & (rate >= min_rate)
    jump = np.zeros(rate.size, dtype=bool)
    start = np.zeros(rate.size, dtype=bool)
    active, last_start = False, -np.inf
    for k in range(rate.size):
        # a jump continues until the sigma level drops below zero (2016); the
        # 2009 and 2011 papers end it at DFRDT <= 0, i.e. also at level == 0
        active = active and level[k] >= 0
        if not active and trigger[k]:
            active = True
            start[k] = k - last_start > group
            last_start = k
        jump[k] = active
    return dfrdt, level, jump, start


@provenance("Lightning jumps by the 2-sigma algorithm of Schultz et al. (2009)")
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
        Averaging period of the flash rate. Default ``"2min"`` (Schultz et
        al. 2011 [2]_ step (i); 2016 [3]_ step 1).
    sigma : float, optional
        Sigma level of a jump. Default 2 (the "2σ" algorithm; chosen "on a
        trial and error basis" by Schultz et al. 2009 [1]_, p. 2550).
    min_rate : float, optional
        Flash rate (flashes min\\ :sup:`-1`) the averaged rate must reach for
        a jump. Default 10 (Schultz et al. 2009 [1]_, p. 2550; 2011 [2]_ step
        (ii); 2016 [3]_ step 5). Applied to the averaged rate with ``>=``,
        see Notes.
    history : int, optional
        Number of previous ``DFRDT`` values of the standard deviation.
        Default 5 (Schultz et al. 2009 [1]_ p. 2549; 2011 [2]_ steps
        (iii)-(v); 2016 [3]_ step 3). The sigma level is first defined at the
        seventh period, i.e. after the 14-min spin-up of Schultz et al. 2016
        [3]_ (six 2-min periods give the five ``DFRDT`` values, the seventh
        period is the current one).
    group : str or timedelta, optional
        Jumps starting within this time of the start of an earlier one are
        not new jumps. Default ``"6min"`` (Schultz et al. 2009 [1]_ p. 2550;
        2016 [3]_ step 6).
    ddof : int, optional
        Delta degrees of freedom of the standard deviation. Default 1
        (sample standard deviation); a radarx choice, see Notes.

    Returns
    -------
    xarray.Dataset
        On the averaging periods (``time``: first input time of each
        period): ``flash_rate`` (averaged), ``dfrdt`` (flashes min\\ :sup:`-2`),
        ``sigma_level``, ``jump`` (a jump is in progress) and ``jump_start``
        (first period of a new jump).

    Notes
    -----
    The steps are those listed by Schultz et al. (2011) [2]_ (section 2c,
    pp. 747-748, appendix p. 753) and Schultz et al. (2016) [3]_ (section 2c,
    pp. 97-98), after the "2σ" algorithm of Schultz et al. (2009) [1]_
    (pp. 2547-2550). The papers differ in, or leave open, the following
    points, so the conventions here are stated explicitly; they
    only matter for exact ties and for the absolute size of the sigma level.

    * *Trigger.* Schultz et al. (2009, p. 2550; 2011, step (vii)) say a jump
      occurs once ``DFRDT`` "exceeds" the 2σ threshold, and 2016 (step 5) that
      the flash rate "exceeds" 10 flashes min\\ :sup:`-1`, i.e. strict
      inequalities. Schultz et al. 2016 (step 4) also call a "2σ jump" one
      with a sigma level of 2. Here a jump needs ``sigma_level >= sigma`` and
      ``flash_rate >= min_rate``, which differs from the strict reading only
      when the values are exactly equal.
    * *End of a jump.* Schultz et al. (2009, p. 2550; 2011, section 3a,
      p. 748) end a jump once ``DFRDT`` is "less than or equal to 0", whereas
      Schultz et al. (2016, step 5) end it once the sigma level "drops below
      zero". Here the 2016 convention is used: the jump continues while
      ``sigma_level >= 0`` and ends at a negative or undefined (NaN) sigma
      level. For a perfectly constant averaged rate with a non-zero standard
      deviation of the previous values (``DFRDT`` = 0) the jump therefore
      continues here, where the 2009 and 2011 rule would end it.
    * *Grouping.* Jumps "separated by 6 min or fewer" are one jump (2009,
      p. 2550; "jump, no jump, jump in consecutive periods"), and within 6
      min "only the first jump remains" (2016, step 6). Here a start counts
      as a new jump (``jump_start``) only when more than ``group`` has passed
      since the start of the most recent jump, including starts that were
      merged; ``jump`` itself is not merged. This reading of "separated" is
      a radarx choice.
    * *Standard deviation.* The papers say "standard deviation" of the five
      previous ``DFRDT`` values and do not say whether it is the sample or the
      population one. ``ddof=1`` (sample) is a radarx choice; ``ddof=0``
      (population) would multiply every sigma level by
      :math:`\\sqrt{n/(n-1)} = \\sqrt{5/4} \\approx 1.118` for ``history``
      :math:`n = 5`. This cannot be settled from the papers.
    * *Zero standard deviation.* If the previous ``DFRDT`` values are all
      equal (a perfectly linear ramp of the rate), the sigma level is
      infinite with the sign of ``DFRDT``, and NaN when ``DFRDT`` is also
      zero. A positive ``DFRDT`` then triggers (if the rate is high enough),
      as the rule "``DFRDT`` exceeds twice the standard deviation" (zero) of
      Schultz et al. gives.
    * *Averaging.* The flash rate is averaged over ``period`` from the first
      input time on; the 10 flashes min\\ :sup:`-1` activation is applied to
      this average (2011, step (ii)), not to the 1-min rates. The standard
      deviation uses the five values before the current one, "not including
      the period of interest" (2009, p. 2549).

    References
    ----------
    .. [1] Schultz, C. J., W. A. Petersen, and L. D. Carey, 2009: Preliminary
       development and evaluation of lightning jump algorithms for the
       real-time detection of severe weather. *J. Appl. Meteor. Climatol.*,
       **48** (12), 2543-2563, https://doi.org/10.1175/2009JAMC2237.1
    .. [2] Schultz, C. J., W. A. Petersen, and L. D. Carey, 2011: Lightning
       and severe weather: A comparison between total and cloud-to-ground
       lightning trends. *Wea. Forecasting*, **26** (5), 744-755,
       https://doi.org/10.1175/WAF-D-10-05026.1
    .. [3] Schultz, E. V., C. J. Schultz, L. D. Carey, D. J. Cecil, and M.
       Bateman, 2016: Automated storm tracking and the lightning jump
       algorithm using GOES-R Geostationary Lightning Mapper (GLM) proxy
       data. *J. Operational Meteor.*, **4** (7), 92-107,
       https://doi.org/10.15191/nwajom.2016.0407
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
