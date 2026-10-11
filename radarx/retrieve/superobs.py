#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx Super-observations
=========================

Super-observations of reflectivity and radial velocity for data assimilation.

A radar volume has far more gates than a model has cells: a WSR-88D volume
holds several million gates, a convection-permitting model resolves a few
hundred metres to a few kilometres. Assimilating every gate is expensive and
the errors of neighbouring gates are correlated, so the gates are averaged
into *super-observations* (Alpert and Kumar 2007). :func:`superobs` bins the
gates of one or more radars into columns of the model grid and layers of
height and averages them.

Binning
-------

* **Column.** Either a block of ``cells`` x ``cells`` model cells of a model
  grid given as latitude and longitude (each gate belongs to the block of its
  nearest cell), or, without a model grid, a square box of ``box`` km on the
  plane around the radar.
* **Layer.** A layer of the height above sea level, 500 m thick up to 12 km
  by default, or the layer edges of the model.
* **Gate.** The position of a gate follows the 4/3 effective Earth model
  (Doviak and Zrnić 1993, Eqs. 2.28b-d): height, ground distance, and the
  latitude and longitude on the great circle of the ray. A gate counts if it
  is closer than ``max_range`` and farther than ``min_range`` on the ground
  and lies in a layer.

Means and spreads
-----------------

* Reflectivity is averaged in linear units (Z in mm6 m-3) and reported in
  dBZ, so that a few strong gates are not diluted by weak ones. The spread is
  the standard deviation of the gate values in dBZ.
* Radial velocity is averaged as it is (positive away from the radar), with
  its standard deviation as spread. Velocities are not dealiased here; use
  :func:`radarx.retrieve.dealias_velocity` first, or let the innovation check
  of the assimilation system reject folded gates.
* The **radial unit vector** (east, north, up) is averaged over the same gates
  as the velocity, so that an observation operator can project a model wind
  ``(u, v, w)`` of the box with ``u*radial_east + v*radial_north +
  w*radial_up``, which is the average of the projections on the gates for a
  uniform wind. It is the unit vector of the ray at the gate (bearing of the
  great circle there, elevation relative to the local horizontal in the 4/3
  Earth model) and is not normalised after averaging.

Error estimate
--------------

The error variance of a super-observation is ``floor**2 + spread**2 / n``
with the number ``n`` of valid gates. The floor (``error_floor``, 2 m s-1 and
5 dBZ by default, the constants of the WRFDA experiments this module comes
from) stands for the errors that averaging does not remove (observation
operator, model, calibration); the second term is the sampling error of the
mean of ``n`` gates that differ by ``spread``, so that a box with a large
spread and few gates gets a larger error. This is a radarx choice. The
super-ob error model of Alpert and Kumar (2007) could not be checked against
the paper.

Echo-free boxes
---------------

With ``missing="clear"`` (the default) a gate without reflectivity is a gate
without echo, which is how volumes mark it, and a box that has at least
``min_gates`` gates but no echo (fewer than ``min_valid`` gates with a
reflectivity, or a mean below ``min_dbz``) is flagged ``nonprecip``. WRFDA can
use these observations to remove model rain where the radar sees none (see
:func:`radarx.io.write_wrfda_radar`). With ``missing="gap"`` gates without
data are left out and there are no such boxes.

The binning and the sums run in a compiled, multithreaded C++ kernel (rays in
blocks taken from an atomic counter; the result does not depend on the number
of threads); if it is not available an equivalent NumPy implementation is
used.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = ["superobs"]

__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

import numpy as np
import xarray as xr

from .._compat import renamed_arguments
from .._provenance import provenance
from .._registry import accessor_method
from .._sweepwise import dask_support
from ..fundamentals.constants import EARTH_RADIUS, EFFECTIVE_RADIUS_4_3
from ..grid.cone import _sweep_dataset, _sweep_names

try:
    from . import _superobs

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _superobs = None
    HAS_COMPILED_KERNEL = False

_DBZ_NAMES = ("DBZH", "DBZ", "reflectivity", "corrected_reflectivity")
_VEL_NAMES = ("VRADH", "VRAD", "VEL", "velocity", "corrected_velocity")
_RHO_NAMES = ("RHOHV", "RHO")
_LAT_NAMES = ("lat", "latitude", "XLAT", "XLAT_M", "lat2d")
_LON_NAMES = ("lon", "longitude", "XLONG", "XLONG_M", "lon2d")
_TERRAIN_NAMES = ("HGT", "HGT_M", "terrain", "terrain_height", "hgt")

# columns of the table of sums of the kernel (see _superobs.cpp)
(
    N_ALL,
    N_ZS,
    N_Z,
    S_Z,
    S_D,
    S_D2,
    N_VS,
    N_V,
    S_V,
    S_V2,
    S_EX,
    S_EY,
    S_EZ,
    S_LAT,
    S_DLON,
    S_H,
    S_R,
) = range(17)
N_STATS = 17


# --------------------------------------------------------------------------
# engine selection
# --------------------------------------------------------------------------


def _use_compiled(engine):
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled super-observation kernel is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


# --------------------------------------------------------------------------
# geometry shared by the kernel and the reference
# --------------------------------------------------------------------------


def _aeqd(lat, lon, lat_ref, lon_ref, radius):
    """Azimuthal equidistant projection (m) about a reference point (radians)."""
    cl = np.cos(lat)
    dl = lon - lon_ref
    a = cl * np.sin(dl)
    b = np.cos(lat_ref) * np.sin(lat) - np.sin(lat_ref) * cl * np.cos(dl)
    c = np.sin(lat_ref) * np.sin(lat) + np.cos(lat_ref) * cl * np.cos(dl)
    r = np.hypot(a, b)
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(r > 0.0, np.arctan2(r, c) / np.where(r > 0.0, r, 1.0), 1.0)
    return radius * k * a, radius * k * b


class _Locator:
    """Nearest model cell of a point (NumPy reference: a k-d tree)."""

    def __init__(self, lat, lon, tolerance):
        self.shape = lat.shape
        self.lat_ref = float(lat[lat.shape[0] // 2, lat.shape[1] // 2])
        self.lon_ref = float(lon[lat.shape[0] // 2, lat.shape[1] // 2])
        x, y = _aeqd(
            np.radians(lat),
            np.radians(lon),
            np.radians(self.lat_ref),
            np.radians(self.lon_ref),
            EARTH_RADIUS,
        )
        self.x, self.y = x, y
        steps = []
        if x.shape[1] > 1:
            steps.append(np.hypot(np.diff(x, axis=1), np.diff(y, axis=1)).ravel())
        if x.shape[0] > 1:
            steps.append(np.hypot(np.diff(x, axis=0), np.diff(y, axis=0)).ravel())
        if not steps:
            raise ValueError("the model grid needs at least two cells")
        self.spacing = float(np.median(np.concatenate(steps)))
        self.max_dist = float(tolerance) * self.spacing
        self._kernel = None
        self._tree = None

    def kernel(self):
        if self._kernel is None:
            self._kernel = _superobs.GridIndex(self.x, self.y, self.max_dist)
        return self._kernel

    def tree(self):
        if self._tree is None:
            from scipy.spatial import cKDTree

            self._tree = cKDTree(np.c_[self.x.ravel(), self.y.ravel()])
        return self._tree

    def nearest(self, lat, lon, bound=True):
        """Flat index of the nearest cell (-1 beyond ``max_dist``) of points in radians."""
        x, y = _aeqd(
            lat, lon, np.radians(self.lat_ref), np.radians(self.lon_ref), EARTH_RADIUS
        )
        d, k = self.tree().query(
            np.c_[x, y],
            distance_upper_bound=self.max_dist if bound else np.inf,
        )
        return np.where(np.isfinite(d), k, -1)


# --------------------------------------------------------------------------
# binning of one sweep: NumPy reference of the kernel
# --------------------------------------------------------------------------


def _bin_sweep_numpy(
    az,
    el,
    rng,
    dbz,
    vr,
    rho,
    lat0,
    lon0,
    alt0,
    reff,
    rearth,
    max_range,
    min_ground,
    edges,
    clear,
    vmax,
    min_rhohv,
    mode,
    box,
    half,
    cells,
    nbx_blocks,
    locator,
):
    """NumPy implementation of the kernel ``_superobs.bin_sweep`` (same sums)."""
    nray, ngate = az.size, rng.size
    has_z, has_v = dbz.size > 0, vr.size > 0
    zok = np.isfinite(dbz) if has_z else np.zeros((nray, ngate), bool)
    with np.errstate(invalid="ignore"):
        vok = (
            np.isfinite(vr) & (np.abs(vr) <= vmax)
            if has_v
            else np.zeros((nray, ngate), bool)
        )
        if rho.size:
            bad = np.isfinite(rho) & (rho < min_rhohv)
            zok = zok & ~bad
            vok = vok & ~bad
    in_range = (rng > 0.0) & (rng <= max_range)
    keep = in_range[None, :] & (np.ones((nray, 1), bool) if clear else (zok | vok))
    ray, gate = np.nonzero(keep)
    empty = np.zeros(0, np.int64), np.zeros((0, N_STATS))
    if ray.size == 0:
        return empty

    r = rng[gate]
    a, e = np.radians(az[ray]), np.radians(el[ray])
    saz, caz, sel, cel = np.sin(a), np.cos(a), np.sin(e), np.cos(e)
    # site altitude in the Earth radius, as in xradar and radarx.fundamentals.geometry
    ra = reff + alt0
    bd = np.sqrt(r * r + ra * ra + 2.0 * r * ra * sel)
    hgt = bd - ra
    hm = hgt + alt0
    s = reff * np.arcsin(r * cel / bd)
    ok = (hm >= edges[0]) & (hm < edges[-1]) & (s > min_ground)
    ray, gate, r, saz, caz, e, hgt, hm, s = (
        x[ok] for x in (ray, gate, r, saz, caz, e, hgt, hm, s)
    )
    level = np.searchsorted(edges, hm, side="right") - 1

    lat0r, lon0r = np.radians(lat0), np.radians(lon0)
    d = s / rearth
    sd, cd = np.sin(d), np.cos(d)
    sl = np.clip(np.sin(lat0r) * cd + np.cos(lat0r) * sd * caz, -1.0, 1.0)
    lat = np.arcsin(sl)
    dlon = np.arctan2(saz * sd * np.cos(lat0r), cd - np.sin(lat0r) * sl)

    if mode == 0:
        ix = np.floor(s * saz / box).astype(np.int64) + half
        iy = np.floor(s * caz / box).astype(np.int64) + half
        width = 2 * half + 1
        inside = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < width)
        col = iy * width + ix
    else:
        k = locator.nearest(lat, lon0r + dlon)
        inside = k >= 0
        nx = locator.shape[1]
        col = (k // nx // cells) * nbx_blocks + (k % nx) // cells
    sel_ = inside
    ray, gate, r, hm, s, e, lat, dlon, level, col = (
        x[sel_] for x in (ray, gate, r, hm, s, e, lat, dlon, level, col)
    )
    if ray.size == 0:
        return empty
    nlev = edges.size - 1
    bins = col * nlev + level
    uniq, inv = np.unique(bins, return_inverse=True)
    inv = inv.ravel()
    n = uniq.size
    stats = np.zeros((n, N_STATS))

    def add(column, weights):
        stats[:, column] = np.bincount(inv, weights=weights, minlength=n)

    zg = zok[ray, gate]
    vg = vok[ray, gate]
    zv = np.where(zg, dbz[ray, gate] if has_z else 0.0, 0.0)
    vv = np.where(vg, vr[ray, gate] if has_v else 0.0, 0.0)
    one = np.ones(ray.size)
    add(N_ALL, one)
    add(S_LAT, lat / (np.pi / 180.0))
    add(S_DLON, dlon / (np.pi / 180.0))
    add(S_H, hm)
    add(S_R, r)
    if has_z:
        add(N_ZS, one)
        add(N_Z, zg.astype(float))
        add(S_Z, np.where(zg, np.power(10.0, zv / 10.0), 0.0))
        add(S_D, zv)
        add(S_D2, zv * zv)
    if has_v:
        add(N_VS, one)
        add(N_V, vg.astype(float))
        add(S_V, vv)
        add(S_V2, vv * vv)
        aa = np.sin(dlon) * np.cos(lat0r)
        bb = np.cos(lat) * np.sin(lat0r) - np.sin(lat) * np.cos(lat0r) * np.cos(dlon)
        nrm = np.hypot(aa, bb)
        eg = e + s / reff
        ceg = np.cos(eg)
        add(S_EX, np.where(vg, aa / nrm * ceg, 0.0))
        add(S_EY, np.where(vg, -bb / nrm * ceg, 0.0))
        add(S_EZ, np.where(vg, np.sin(eg), 0.0))
    return uniq.astype(np.int64), stats


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


class _Radar:
    """A radar: its site, name, time and the sweeps to read."""

    def __init__(self, name, lat, lon, alt, sweeps, time):
        self.name = name
        self.lat, self.lon, self.alt = lat, lon, alt
        self.sweeps = sweeps  # list of Dataset (lazy)
        self.time = time


def _first(ds, names):
    for name in names:
        if name in ds:
            return name
    return None


def _scalar(ds, name):
    if name not in ds.variables:
        raise KeyError(f"the sweep has no {name!r}; it is needed to place the gates")
    values = np.asarray(ds[name].values, dtype=np.float64)
    if values.size != 1:
        raise ValueError(f"{name!r} must be a single value, not a moving platform")
    return float(values.reshape(()))


def _volume_time(sweeps):
    lo, hi = [], []
    for ds in sweeps:
        if "time" in ds.variables:
            t = ds["time"].values
            if t.size and np.issubdtype(t.dtype, np.datetime64):
                lo.append(np.nanmin(t))
                hi.append(np.nanmax(t))
    if not lo:
        return np.datetime64("NaT", "ns")
    lo, hi = min(lo), max(hi)
    return (lo + (hi - lo) / 2).astype("datetime64[ns]")


def _radar_of(name, sweeps, default):
    first = sweeps[0]
    return _Radar(
        name or default,
        _scalar(first, "latitude"),
        _scalar(first, "longitude"),
        _scalar(first, "altitude") if "altitude" in first.variables else 0.0,
        sweeps,
        _volume_time(sweeps),
    )


def _tree_radar(tree, name, default):
    names = _sweep_names(tree)
    if not names:
        raise ValueError("No sweep groups found in DataTree.")
    sweeps = [_sweep_dataset(tree, n) for n in names]
    if name is None:
        name = tree.attrs.get("instrument_name")
        if name is None:
            try:
                name = tree.root.to_dataset().attrs.get("instrument_name")
            except Exception:  # pragma: no cover
                name = None
    return _radar_of(None if name is None else str(name), sweeps, default)


def _radars(obj):
    """List of :class:`_Radar` from a volume, a sweep, a sequence or a dict."""
    if isinstance(obj, xr.DataTree):
        return [_tree_radar(obj, None, "radar_0")]
    if isinstance(obj, xr.Dataset):
        return [_radar_of(None, [obj], "radar_0")]
    if isinstance(obj, dict):
        items = list(obj.items())
    elif isinstance(obj, (list, tuple)):
        items = [(None, o) for o in obj]
    else:
        raise TypeError(
            "obj must be an xarray.DataTree (volume), an xarray.Dataset (sweep), "
            "or a list or dict of them"
        )
    if not items:
        raise ValueError("no radar data given")
    radars, loose = [], []
    for key, item in items:
        if isinstance(item, xr.DataTree):
            radars.append(_tree_radar(item, key, f"radar_{len(radars)}"))
        elif isinstance(item, xr.Dataset):
            loose.append((key, item))
        else:
            raise TypeError("the items must be xarray.DataTree or xarray.Dataset")
    # bare sweeps of one site belong to one radar
    sites = {}
    for key, ds in loose:
        site = (_scalar(ds, "latitude"), _scalar(ds, "longitude"))
        sites.setdefault(site, (key, []))[1].append(ds)
    for key, sweeps in sites.values():
        radars.append(_radar_of(key, sweeps, f"radar_{len(radars)}"))
    return radars


def _sweep_arrays(ds, dbz, vel, rho, use_rho):
    """Contiguous float64 arrays of a sweep; an empty array for a field the sweep lacks."""

    def pick(given, names):
        if given is not None:
            return given if given in ds else None
        return _first(ds, names)

    dbz_name = pick(dbz, _DBZ_NAMES)
    vel_name = pick(vel, _VEL_NAMES)
    rho_name = pick(rho, _RHO_NAMES) if use_rho else None
    ref = dbz_name or vel_name
    if ref is None:
        return None
    ray_dim = [d for d in ds[ref].dims if d != "range"][0]

    def field(name):
        if name is None or name not in ds:
            return np.zeros((0, 0))
        da = ds[name].transpose(ray_dim, "range")
        return np.ascontiguousarray(np.asarray(da.data), dtype=np.float64)

    z, v, r = field(dbz_name), field(vel_name), field(rho_name)
    az = np.ascontiguousarray(ds["azimuth"].values, dtype=np.float64)
    if "elevation" in ds.variables:
        el = np.broadcast_to(ds["elevation"].values, az.shape)
    else:
        el = np.full(az.shape, float(ds["sweep_fixed_angle"]))
    return (
        az,
        np.ascontiguousarray(el, dtype=np.float64),
        np.ascontiguousarray(ds["range"].values, dtype=np.float64),
        z,
        v,
        r,
    )


def _grid_arrays(grid, terrain):
    """Latitude, longitude (and terrain) 2-D arrays of a model grid description."""

    def squeeze(a):
        a = np.asarray(a, dtype=np.float64)
        while a.ndim > 2 and a.shape[0] == 1:
            a = a[0]
        return a

    if isinstance(grid, xr.Dataset):
        lat_name, lon_name = _first(grid, _LAT_NAMES), _first(grid, _LON_NAMES)
        if lat_name is None or lon_name is None:
            raise KeyError(f"the grid needs latitude and longitude ({_LAT_NAMES})")
        lat, lon = squeeze(grid[lat_name].values), squeeze(grid[lon_name].values)
        if terrain is None:
            name = _first(grid, _TERRAIN_NAMES)
            terrain = None if name is None else grid[name]
    elif isinstance(grid, (tuple, list)) and len(grid) == 2:
        lat, lon = (squeeze(getattr(g, "values", g)) for g in grid)
    else:
        raise TypeError(
            "grid must be a number (box size in km), a (lat, lon) pair or a "
            "Dataset with latitude and longitude"
        )
    if lat.ndim == 1 and lon.ndim == 1:
        lat, lon = np.meshgrid(lat, lon, indexing="ij")
    if lat.ndim != 2 or lat.shape != lon.shape:
        raise ValueError("lat and lon must be 1-D axes or 2-D arrays of one shape")
    if terrain is not None:
        terrain = squeeze(getattr(terrain, "values", terrain))
        if terrain.shape != lat.shape:
            raise ValueError("terrain must have the shape of the model grid")
    return lat, lon, terrain


def _layer_edges(layers, top):
    if np.ndim(layers) == 0:
        dz = float(layers)
        if not (dz > 0 and top > 0):
            raise ValueError("layers and top must be positive")
        n = int(np.ceil(top / dz - 1e-9))
        return dz * np.arange(n + 1, dtype=np.float64)
    edges = np.asarray(layers, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("layer edges must be a strictly increasing 1-D array")
    return edges


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def _merge(parts):
    """Add the tables (bins, stats) of sweeps and blocks."""
    bins = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0, np.int64)
    if bins.size == 0:
        return np.zeros(0, np.int64), np.zeros((0, N_STATS))
    stats = np.concatenate([p[1] for p in parts])
    order = np.argsort(bins, kind="stable")
    bins, stats = bins[order], stats[order]
    first = np.r_[0, np.nonzero(np.diff(bins))[0] + 1]
    return bins[first], np.add.reduceat(stats, first, axis=0)


def _radar_sums(
    radar,
    dbz,
    vel,
    rho,
    min_rhohv,
    geometry,
    edges,
    clear,
    vmax,
    target,
    use,
    n_threads,
):
    for given in (dbz, vel, rho if min_rhohv is not None else None):
        if given is not None and not any(given in ds for ds in radar.sweeps):
            raise KeyError(f"{given!r} is not in any sweep of {radar.name!r}")
    parts, used = [], 0
    for ds in radar.sweeps:
        arrays = _sweep_arrays(ds, dbz, vel, rho, min_rhohv is not None)
        if arrays is None:
            continue
        used += 1
        az, el, rng, z, v, r = arrays
        args = (
            az,
            el,
            rng,
            z,
            v,
            r,
            radar.lat,
            radar.lon,
            radar.alt,
            EFFECTIVE_RADIUS_4_3,
            EARTH_RADIUS,
            geometry["max_range"],
            geometry["min_range"],
            edges,
            clear,
            vmax,
            -np.inf if min_rhohv is None else float(min_rhohv),
            target["mode"],
            target["box"],
            target["half"],
            target["cells"],
            target["nbx"],
        )
        if use:
            index = target["locator"].kernel() if target["mode"] == 1 else None
            lat_ref = target["locator"].lat_ref if target["mode"] == 1 else 0.0
            lon_ref = target["locator"].lon_ref if target["mode"] == 1 else 0.0
            parts.append(
                _superobs.bin_sweep(
                    *args, lat_ref, lon_ref, index, n_threads=int(n_threads or 0)
                )
            )
        else:
            parts.append(_bin_sweep_numpy(*args, target["locator"]))
    if not used:
        raise KeyError(
            f"no sweep of {radar.name!r} has a reflectivity ({_DBZ_NAMES}) or a "
            f"velocity ({_VEL_NAMES}); pass dbz= or vel="
        )
    return _merge(parts)


def _std(n, s1, s2):
    """Sample standard deviation from counts and sums (NaN for fewer than 2)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / n) / (n - 1.0)
    return np.where(n > 1, np.sqrt(np.maximum(var, 0.0)), np.nan)


def _lonwrap(lon):
    return (lon + 180.0) % 360.0 - 180.0


@dask_support("eager", "the number of observations depends on the data.")
@renamed_arguments(vel="field", rho="rhohv", rho_min="min_rhohv")
@provenance(
    "Super-observations of radar gates in model columns and layers, after Alpert and Kumar (2007)",
    extra_refs=("doviak-zrnic-1993",),
)
def superobs(
    obj,
    grid,
    *,
    cells=3,
    layers=500.0,
    top=12000.0,
    dbz=None,
    field=None,
    rhohv=None,
    min_rhohv=None,
    max_range=160e3,
    min_range=3000.0,
    min_gates=6,
    min_valid=3,
    min_dbz=5.0,
    max_velocity=40.0,
    missing="clear",
    error_floor=(2.0, 5.0),
    terrain=None,
    domain_tolerance=1.0,
    n_threads=None,
    engine="auto",
):
    """
    Super-observations of reflectivity and radial velocity on a model grid.

    Bins the gates of one or more radars into columns of ``cells`` x
    ``cells`` model cells (or boxes of a given size) and layers of height and
    averages them, as in the super-obbing of Alpert and Kumar (2007). See
    :mod:`radarx.retrieve.superobs` for the binning, the averages and the error
    estimate.

    Parameters
    ----------
    obj : xarray.DataTree, xarray.Dataset, list or dict
        A volume (DataTree with ``sweep_*`` groups, e.g. from xradar), a single
        sweep (Dataset), or several radars as a list or a dict of names to
        volumes. Bare sweeps in a list that share a site form one radar. Sweeps
        carry ``azimuth``, ``elevation``, ``range`` and the site
        ``latitude``, ``longitude`` and ``altitude``; all sweeps are used.
        Dask arrays are read one sweep at a time and are not computed in place.
    grid : float or tuple or xarray.Dataset
        The columns. A number is the side of square boxes in km on the plane
        around each radar (no model grid, ``cells`` is ignored). A model grid is
        given as a ``(lat, lon)`` pair of 2-D arrays (or 1-D axes) or as a
        Dataset with ``lat``/``lon``, ``latitude``/``longitude`` or WRF
        ``XLAT``/``XLONG`` (a leading ``Time`` dimension of length 1 is dropped);
        a Dataset with ``HGT`` or ``terrain`` also gives ``terrain``.
    cells : int, optional
        Model cells per side of a column. Default 3.
    layers : float or array-like, optional
        Layer thickness in m (default 500) from the ground up to ``top``, or the
        layer edges in m above sea level.
    top : float, optional
        Top of the layers (m above sea level) when ``layers`` is a thickness.
        Default 12000.
    dbz, field, rhohv : str, optional
        Names of the reflectivity (dBZ), the radial velocity (m s-1) and the
        correlation coefficient. By default the first found of ``DBZH``, ``DBZ``,
        ``reflectivity``, ``corrected_reflectivity``; of ``VRADH``, ``VRAD``,
        ``VEL``, ``velocity``, ``corrected_velocity``; and of ``RHOHV``. A sweep
        without a field contributes no gates of that field.
    min_rhohv : float, optional
        Gates with a correlation coefficient below this are removed from both
        fields (non-meteorological echo). Default None (no test). The
        prototype of this function used 0.85.
    max_range : float, optional
        Slant range (m) beyond which gates are left out. Default 160 km.
    min_range : float, optional
        Ground distance (m) within which gates are left out. Default 3 km.
    min_gates : int, optional
        Gates a box needs to give a super-observation. Default 6.
    min_valid : int, optional
        Valid gates of a field a box needs to report it. Default 3.
    min_dbz : float, optional
        Reflectivity (dBZ) of the mean below which a box has no echo. Default
        5.
    max_velocity : float, optional
        Velocities with a larger magnitude (m s-1) are left out. Default 40.
    missing : {"clear", "gap"}, optional
        ``"clear"`` (default): a gate without reflectivity is a gate without
        echo, and boxes without echo are flagged ``nonprecip``. ``"gap"``:
        gates without data are left out and there are no ``nonprecip`` boxes.
    error_floor : tuple of float, optional
        Floor of the error of the velocity (m s-1) and the reflectivity (dBZ).
        Default (2, 5).
    terrain : array-like, optional
        Terrain height (m) of the model grid; adds ``terrain_height``.
    domain_tolerance : float, optional
        A gate is used if its nearest model cell center is closer than this
        many grid spacings. Default 1.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled
        kernel and falls back to NumPy.

    Returns
    -------
    xarray.Dataset
        One entry per box with enough gates, along ``obs``. Coordinates:
        ``radar_index`` (into ``radar``), ``column``, ``column_row``,
        ``column_col`` (row and column of the block of model cells, or of the
        box east and north of the radar), ``level``, ``level_height`` (layer
        centre, m). Along ``radar``: ``radar_name``, ``radar_latitude``,
        ``radar_longitude``, ``radar_altitude`` and ``radar_time`` (middle of
        the volume). ``level_edges`` (m) gives the layers. Variables:

        ``reflectivity`` (dBZ), ``reflectivity_spread``, ``reflectivity_error``, ``reflectivity_count``
            Mean in linear Z, standard deviation of the gate dBZ, error
            estimate, number of valid gates. NaN where the box has no echo.
        ``velocity`` (m s-1), ``velocity_spread``, ``velocity_error``, ``velocity_count``
            Mean radial velocity (positive away from the radar) and the same.
        ``radial_east``, ``radial_north``, ``radial_up``
            Mean unit vector of the ray at the velocity gates.
        ``nonprecip``
            1 where the box has no echo (see ``missing``).
        ``gate_count``
            Gates in the box.
        ``latitude``, ``longitude``, ``height``, ``range``
            Mean position of the gates (height above sea level, slant range).
        ``terrain_height``
            With ``terrain``: terrain of the model cell nearest to the box.

    Raises
    ------
    KeyError
        If neither reflectivity nor velocity is found, or the grid has no
        latitude and longitude.
    ValueError, TypeError
        For unknown options or an input that does not fit.
    ImportError
        If ``engine="compiled"`` and the compiled kernel is not available.

    Notes
    -----
    A sweep of a split cut that holds only one of the fields adds gates of that
    field. All sweeps are used, including repeated cuts of the same elevation
    (SAILS), whose gates are then counted once per cut.

    References
    ----------
    Alpert, J. C., and V. K. Kumar, 2007: Radial wind super-obs from the WSR-88D
    radars in the NCEP operational assimilation system. *Mon. Wea. Rev.*,
    **135** (3), 1090-1109, https://doi.org/10.1175/MWR3324.1
    (cited for super-obs as averages of raw radial winds; the box sizes and the
    error model of this function are radarx choices, not checked against the
    paper).

    Doviak, R. J., and D. S. Zrnić, 1993: *Doppler Radar and Weather
    Observations*, 2nd ed. Academic Press (book, no DOI). Eqs. 2.28b-d.

    Examples
    --------
    >>> so = radarx.retrieve.superobs(volume, (xlat, xlong), cells=3)  # doctest: +SKIP
    >>> so = volume.radarx.superobs(2.0, layers=250.0)  # doctest: +SKIP
    """
    use = _use_compiled(engine)
    if missing not in ("clear", "gap"):
        raise ValueError(f"missing must be 'clear' or 'gap', not {missing!r}")
    if min_gates < 1 or min_valid < 1:
        raise ValueError("min_gates and min_valid must be at least 1")
    if cells < 1 or int(cells) != cells:
        raise ValueError("cells must be a positive integer")
    err_v, err_z = (float(e) for e in error_floor)
    edges = _layer_edges(layers, top)
    radars = _radars(obj)

    target = {"cells": int(cells), "nbx": 0, "box": 0.0, "half": 0, "locator": None}
    terrain_arr = None
    if isinstance(grid, (int, float, np.integer, np.floating)):
        box = float(grid) * 1000.0
        if not box > 0:
            raise ValueError("the box size must be positive")
        target.update(mode=0, box=box, half=int(np.ceil(max_range / box)) + 1)
    else:
        lat, lon, terrain_arr = _grid_arrays(grid, terrain)
        target["locator"] = _Locator(lat, lon, domain_tolerance)
        target["mode"] = 1
        target["nbx"] = -(-lat.shape[1] // int(cells))
        if use:
            target["locator"].kernel()
    geometry = {"max_range": float(max_range), "min_range": float(min_range)}

    tables = []
    for radar in radars:
        tables.append(
            _radar_sums(
                radar,
                dbz,
                field,
                rhohv,
                min_rhohv,
                geometry,
                edges,
                missing == "clear",
                float(max_velocity),
                target,
                use,
                n_threads,
            )
        )

    out = _assemble(
        radars,
        tables,
        edges,
        target,
        terrain_arr,
        dict(
            min_gates=min_gates,
            min_valid=min_valid,
            min_dbz=float(min_dbz),
            clear=missing == "clear",
            err_v=err_v,
            err_z=err_z,
        ),
    )
    out.attrs.update(
        {
            "title": "Super-observations of radar reflectivity and radial velocity",
            "radial_velocity_convention": "positive away from the radar",
            "superobs_columns": (
                f"{target['cells']} x {target['cells']} model cells"
                if target["mode"] == 1
                else f"boxes of {target['box'] / 1000.0:g} km around the radar"
            ),
            "superobs_min_gates": int(min_gates),
            "superobs_min_valid": int(min_valid),
            "superobs_min_dbz": float(min_dbz),
            "superobs_missing": missing,
            "superobs_error_floor": np.array([err_v, err_z]),
            "superobs_error_model": "sqrt(floor**2 + spread**2 / n)",
        }
    )
    return out


def _assemble(radars, tables, edges, target, terrain, opt):
    nlev = edges.size - 1
    pieces = []
    for k, (bins, s) in enumerate(tables):
        pieces.append(_finalize(k, radars[k], bins, s, nlev, target, opt))
    cat = {key: np.concatenate([p[key] for p in pieces]) for key in pieces[0]}
    n = cat["level"].size
    dims = ("obs",)

    def var(name, values, units=None, long_name=None, dtype=None, **attrs):
        arr = np.asarray(values)
        if dtype is not None:
            arr = arr.astype(dtype)
        a = dict(attrs)
        if units:
            a["units"] = units
        if long_name:
            a["long_name"] = long_name
        return xr.Variable(dims, arr, a)

    coords = {
        "radar_index": var(
            "", cat["radar"], long_name="index into radar", dtype=np.int32
        ),
        "column": var("", cat["col"], long_name="column of the box", dtype=np.int64),
        "column_row": var(
            "",
            cat["row"],
            long_name="row of the block of model cells, or box index north of the radar",
            dtype=np.int64,
        ),
        "column_col": var(
            "",
            cat["colx"],
            long_name="column of the block of model cells, or box index east of the radar",
            dtype=np.int64,
        ),
        "level": var("", cat["level"], long_name="layer index", dtype=np.int32),
        "level_height": var(
            "",
            0.5 * (edges[cat["level"]] + edges[cat["level"] + 1]),
            "m",
            "centre of the layer, height above sea level",
        ),
        "radar_name": xr.Variable("radar", np.array([r.name for r in radars])),
        "radar_latitude": xr.Variable(
            "radar", np.array([r.lat for r in radars]), {"units": "degrees_north"}
        ),
        "radar_longitude": xr.Variable(
            "radar", np.array([r.lon for r in radars]), {"units": "degrees_east"}
        ),
        "radar_altitude": xr.Variable(
            "radar", np.array([r.alt for r in radars]), {"units": "m"}
        ),
        "radar_time": xr.Variable(
            "radar",
            np.array([r.time for r in radars], dtype="datetime64[ns]"),
            {"long_name": "middle of the volume"},
        ),
        "level_edges": xr.Variable(
            "level_edge",
            edges,
            {"units": "m", "long_name": "layer edges, height above sea level"},
        ),
    }
    data = {
        "reflectivity": var(
            "", cat["dbz"], "dBZ", "mean reflectivity, averaged in linear Z"
        ),
        "reflectivity_spread": var(
            "", cat["dbz_sd"], "dBZ", "standard deviation of the gate reflectivities"
        ),
        "reflectivity_error": var(
            "", cat["dbz_err"], "dBZ", "error estimate of the reflectivity"
        ),
        "reflectivity_count": var(
            "", cat["nz"], None, "gates with a reflectivity", dtype=np.int32
        ),
        "velocity": var(
            "", cat["vr"], "m s-1", "mean radial velocity, positive away from the radar"
        ),
        "velocity_spread": var(
            "", cat["vr_sd"], "m s-1", "standard deviation of the gate velocities"
        ),
        "velocity_error": var(
            "", cat["vr_err"], "m s-1", "error estimate of the velocity"
        ),
        "velocity_count": var(
            "", cat["nv"], None, "gates with a velocity", dtype=np.int32
        ),
        "radial_east": var(
            "", cat["ex"], "1", "mean east component of the unit vector of the ray"
        ),
        "radial_north": var(
            "", cat["ey"], "1", "mean north component of the unit vector of the ray"
        ),
        "radial_up": var(
            "", cat["ez"], "1", "mean upward component of the unit vector of the ray"
        ),
        "nonprecip": var(
            "",
            cat["nonprecip"],
            None,
            "no echo in a box with enough gates",
            dtype=np.int8,
            flag_values=np.array([0, 1], dtype=np.int8),
            flag_meanings="echo_or_unknown nonprecipitating",
        ),
        "gate_count": var("", cat["n"], None, "gates in the box", dtype=np.int32),
        "latitude": var("", cat["lat"], "degrees_north", "mean latitude of the gates"),
        "longitude": var("", cat["lon"], "degrees_east", "mean longitude of the gates"),
        "height": var("", cat["h"], "m", "mean height of the gates above sea level"),
        "range": var("", cat["r"], "m", "mean slant range of the gates"),
    }
    if terrain is not None and target["locator"] is not None and n:
        k = target["locator"].nearest(
            np.radians(cat["lat"]), np.radians(cat["lon"]), bound=False
        )
        data["terrain_height"] = var(
            "", terrain.ravel()[k], "m", "terrain height of the nearest model cell"
        )
    elif terrain is not None:
        data["terrain_height"] = var("", np.zeros(0), "m", "terrain height")
    return xr.Dataset(data, coords=coords)


def _finalize(k, radar, bins, s, nlev, target, opt):
    """Means, spreads and errors from the sums of one radar."""
    n_all = s[:, N_ALL]
    n_z, n_v = s[:, N_Z], s[:, N_V]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_z = s[:, S_Z] / n_z
        dbz = 10.0 * np.log10(np.where(mean_z > 0, mean_z, np.nan))
        mean_v = s[:, S_V] / n_v
    have = n_all >= opt["min_gates"]
    z_ok = have & (n_z >= opt["min_valid"]) & (dbz >= opt["min_dbz"])
    v_ok = have & (n_v >= opt["min_valid"])
    nonprecip = opt["clear"] & have & (s[:, N_ZS] >= opt["min_gates"]) & ~z_ok
    keep = z_ok | v_ok | nonprecip
    dbz_sd = _std(n_z, s[:, S_D], s[:, S_D2])
    vr_sd = _std(n_v, s[:, S_V], s[:, S_V2])
    with np.errstate(invalid="ignore", divide="ignore"):
        dbz_err = np.sqrt(opt["err_z"] ** 2 + np.nan_to_num(dbz_sd) ** 2 / n_z)
        vr_err = np.sqrt(opt["err_v"] ** 2 + np.nan_to_num(vr_sd) ** 2 / n_v)
        ex, ey, ez = (s[:, c] / n_v for c in (S_EX, S_EY, S_EZ))
        lat = s[:, S_LAT] / n_all
        lon = _lonwrap(radar.lon + s[:, S_DLON] / n_all)
    dbz_err = np.where(nonprecip & ~z_ok, opt["err_z"], dbz_err)
    nan = np.nan

    def pick(a):
        return a[keep]

    col = bins // nlev
    if target["mode"] == 1:
        row, colx = col // target["nbx"], col % target["nbx"]
    else:
        width = 2 * target["half"] + 1
        row, colx = col // width - target["half"], col % width - target["half"]
    out = {
        "radar": np.full(keep.sum(), k),
        "col": pick(col),
        "row": pick(row),
        "colx": pick(colx),
        "level": pick(bins % nlev),
        "dbz": pick(np.where(z_ok, dbz, nan)),
        "dbz_sd": pick(np.where(z_ok, dbz_sd, nan)),
        "dbz_err": pick(np.where(z_ok | nonprecip, dbz_err, nan)),
        "nz": pick(n_z),
        "vr": pick(np.where(v_ok, mean_v, nan)),
        "vr_sd": pick(np.where(v_ok, vr_sd, nan)),
        "vr_err": pick(np.where(v_ok, vr_err, nan)),
        "nv": pick(n_v),
        "ex": pick(np.where(v_ok, ex, nan)),
        "ey": pick(np.where(v_ok, ey, nan)),
        "ez": pick(np.where(v_ok, ez, nan)),
        "nonprecip": pick(nonprecip),
        "n": pick(n_all),
        "lat": pick(lat),
        "lon": pick(lon),
        "h": pick(s[:, S_H] / n_all),
        "r": pick(s[:, S_R] / n_all),
    }
    return out


@accessor_method("dataset", "datatree", name="superobs", wraps=superobs)
def _superobs_accessor(self):
    """
    Super-observations of a volume or sweep on a model grid.

    See :func:`radarx.retrieve.superobs` for the parameters.

    Returns
    -------
    xarray.Dataset
        The super-observations along ``obs``.
    """
