#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx VIL, Echo Tops and Water Content
=======================================

Column products of the reflectivity of a radar volume: vertically integrated
liquid (VIL), VIL density, the echo-top height, and the liquid water content
of rain.

All column products work on the same abstraction: a column is a list of
reflectivity samples at known heights. Three kinds of input give columns.

* **Polar volume** (``xarray.DataTree`` with ``sweep_*`` groups). A column is
  an (azimuth, ground range) position of the lowest sweep. Every other sweep
  contributes the value of its nearest ray and gate, placed at its beam height
  (4/3 effective Earth radius). The result is on the polar grid of the lowest
  sweep.
* **Grid** (``xarray.Dataset`` with ``z``, ``y``, ``x``, e.g. the output of
  :func:`radarx.grid.grid_cones`). A column is an (y, x) position.
* **Quasi-vertical profile** (``xarray.Dataset`` with ``height`` and usually
  ``time``, e.g. the output of :func:`radarx.retrieve.qvp_timeseries`). A
  column is one profile.

VIL
---

VIL is the liquid water content of rain obtained from the reflectivity with
the exponential drop size distribution of Marshall and Palmer, integrated
over height (Greene and Clark 1972). With ``Z`` in mm6 m-3 and the layer
depth ``dh`` in m, the discrete form is

.. math::

    \\mathrm{VIL} = 3.44\\times10^{-6} \\sum_i
    \\left(\\frac{Z_i + Z_{i+1}}{2}\\right)^{4/7} \\Delta h_i
    \\quad [\\mathrm{kg\\,m^{-2}}]

summed over the layers between consecutive samples. The coefficient, the
exponent and the units of the underlying relation ``LWC = 3.44e-6 Z^(4/7)``
(kg m-3) are given by Seo et al. (2020, Eq. 3); the form of the sum above is
the one quoted for Greene and Clark (1972); neither could be checked against
the original paper.

Integration limits. Nothing is assumed above the highest valid sample. Below
the lowest valid sample the column is left out by default, so the VIL is a
lower bound, and ``VIL_LOWER_BOUND`` flags it. With ``fill_below=True`` the
lowest value is extended down to the base height (compare the pseudo-CAPPI).
Between two samples (the tilts of a volume, or a gap of missing levels) the
reflectivity varies linearly with height; a column needs two samples.

Reflectivity above ``dbz_cap`` (default 56 dBZ, a radarx choice that follows
the operational product, where it limits the contribution of hail) is set to
that value; ``dbz_cap=None`` uses the data as they are. Values below
``min_dbz`` count as no echo.

A quasi-vertical profile (QVP) gives the VIL of the azimuthal-mean profile,
not the mean of the VIL of the columns of the volume. Because ``Z**(4/7)`` is
concave (Jensen's inequality), the VIL of a profile that is the mean in
linear Z (the QVP default of :func:`radarx.retrieve.qvp`) is at least the mean
of the column VILs, the more so the larger the azimuthal variance of Z; a
profile that is the mean in dBZ or the median gives a lower value. The QVP
also leaves out gates that fail its quality tests, which raises the mean
further, and it starts at the beam height of the first gate, so its VIL is a
lower bound.

Water content. ``liquid_water_content`` gives the gate-by-gate liquid water
content, with the same power law (``method="zm"``) or from the gamma drop size
distribution of :func:`radarx.retrieve.dsd` (``method="dsd"``). A radar
measures the liquid water content in rain only: in the melting layer the
reflectivity is enhanced by wet snow, and in ice the power law overestimates
the water content, so restrict the products to the rain below the melting
layer (``melting``, ``mask``). Ice water content needs a different method that
is not implemented.

Echo top
--------

The echo top is the height of the highest sample with reflectivity at or above
a threshold (18 dBZ, the WSR-88D echo-top product). Following Lakshmanan et
al. (2013), it is interpolated linearly in dBZ between that sample and the
next higher one (the reflectivity of a higher beam without echo is
``no_echo_dbz``, -14 dBZ as in the paper); with no higher sample the top of the
beam is used. See :func:`echo_top` for the differences from the paper.

VIL density is the VIL divided by the echo-top height above the base
(Amburn and Wolf 1997, not checked against the paper).

The column integrals run in a compiled, multithreaded C++ kernel; if it is not
available an equivalent NumPy implementation is used.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = ["vil", "vil_density", "echo_top", "liquid_water_content"]

__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

import numpy as np
import xarray as xr

from .._registry import accessor_method
from ..fundamentals.constants import EFFECTIVE_RADIUS_4_3
from ..fundamentals.geometry import beam_center_height
from ..grid.cone import _select_sweeps, _sweep_dataset, _sweep_names

try:
    from . import _vil

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _vil = None
    HAS_COMPILED_KERNEL = False

# Greene and Clark (1972) relation LWC [kg m-3] = COEFFICIENT * Z**EXPONENT with
# Z in mm6 m-3 (Seo et al. 2020, Eq. 3).
COEFFICIENT = 3.44e-6
EXPONENT = 4.0 / 7.0

_DBZ_NAMES = ("DBZH", "DBZ", "reflectivity", "corrected_reflectivity")
_VERTICAL_NAMES = ("z", "altitude", "height", "level")
_MAX_GAP = 2.0  # sweeps are matched to a ray within this many ray spacings
_CHUNK = 2_000_000  # samples per kernel call of a polar volume
_BEAMWIDTH = 1.0  # degrees, used when the volume does not give one


# --------------------------------------------------------------------------
# engine selection
# --------------------------------------------------------------------------


def _use_compiled(engine):
    """Whether to run the compiled kernel for the requested ``engine``."""
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled VIL kernel is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


# --------------------------------------------------------------------------
# column products: NumPy reference of the kernel
# --------------------------------------------------------------------------


def _columns_numpy(
    h,
    h_top,
    v,
    ceiling,
    shared,
    dbz_cap,
    min_dbz,
    top_threshold,
    no_echo_dbz,
    fill_below,
    base_height,
    interpolate,
):
    """
    NumPy implementation of the kernel ``_vil.columns`` (same results).

    ``v`` has the shape (levels, columns); ``h`` and ``h_top`` are (levels,)
    when ``shared`` and (levels, columns) otherwise. Returns the array
    (VIL, liquid VIL, lowest height, highest height, echo top) of shape
    (5, columns).
    """
    nk, nc = v.shape
    h = np.broadcast_to(h[:, None], v.shape) if shared else h
    h_top = np.broadcast_to(h_top[:, None], v.shape) if shared else h_top
    valid = np.isfinite(h) & ~np.isnan(v)
    order = np.argsort(np.where(valid, h, np.inf), axis=0, kind="stable")
    hs = np.take_along_axis(h, order, 0)
    ts = np.take_along_axis(h_top, order, 0)
    vs = np.take_along_axis(v, order, 0)
    ok = np.take_along_axis(valid, order, 0)
    n = ok.sum(axis=0)

    cap = np.inf if np.isnan(dbz_cap) else dbz_cap
    floor_dbz = -np.inf if np.isnan(min_dbz) else min_dbz
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        z = np.where(vs < floor_dbz, 0.0, np.power(10.0, np.minimum(vs, cap) / 10.0))
        z = np.where(ok, z, 0.0)
        h0, h1 = hs[:-1], hs[1:]
        z0, z1 = z[:-1], z[1:]
        pair = ok[:-1] & ok[1:]
        dh = np.where(pair, h1 - h0, 0.0)
        layer = np.where(pair, np.power(0.5 * (z0 + z1), EXPONENT) * dh, 0.0)
        total = layer.sum(axis=0)
        has = pair.any(axis=0)

        # liquid-phase VIL: layers cut at the ceiling
        c = ceiling[None, :]
        use = pair & ~np.isnan(c) & (h0 < c)
        dh_safe = np.where(dh > 0, dh, 1.0)
        zc = z0 + (z1 - z0) * (c - h0) / dh_safe
        full = h1 <= c
        part = np.power(0.5 * (z0 + zc), EXPONENT) * (c - h0)
        liquid_layer = np.where(use, np.where(full, layer, part), 0.0)
        total_liquid = liquid_layer.sum(axis=0)
        has_liquid = use.any(axis=0)

        if fill_below and np.isfinite(base_height):
            fill = np.power(z[0], EXPONENT)
            seg = hs[0] - base_height
            add = (n > 0) & (seg > 0)
            total = total + np.where(add, fill * seg, 0.0)
            has = has | add
            ceil_ok = ~np.isnan(ceiling)
            seg_l = np.minimum(hs[0], ceiling) - base_height
            add_l = (n > 0) & ceil_ok & (seg_l > 0)
            total_liquid = total_liquid + np.where(add_l, fill * seg_l, 0.0)
            has_liquid = has_liquid | add_l

    vil = np.where(has, COEFFICIENT * total, np.nan)
    liquid = np.where(has_liquid, COEFFICIENT * total_liquid, np.nan)
    lowest = np.where(n > 0, hs[0], np.nan)
    highest = np.where(
        n > 0, np.take_along_axis(hs, np.maximum(n - 1, 0)[None], 0)[0], np.nan
    )

    # echo top: highest sample at or above the threshold
    with np.errstate(invalid="ignore"):
        above = ok & (vs >= top_threshold)
    has_top = above.any(axis=0)
    b = np.where(has_top, nk - 1 - np.argmax(above[::-1], axis=0), 0)
    hb = np.take_along_axis(hs, b[None], 0)[0]
    tb = np.take_along_axis(ts, b[None], 0)[0]
    vb = np.take_along_axis(vs, b[None], 0)[0]
    nxt = np.minimum(b + 1, nk - 1)
    ha = np.take_along_axis(hs, nxt[None], 0)[0]
    va = np.maximum(np.take_along_axis(vs, nxt[None], 0)[0], no_echo_dbz)
    has_next = (b + 1) < n
    with np.errstate(invalid="ignore", divide="ignore"):
        interpolated = hb + (vb - top_threshold) / (vb - va) * (ha - hb)
    if interpolate:
        top = np.where(has_next, interpolated, tb)
    else:
        top = tb
    top = np.where(has_top, top, np.nan)
    return np.stack([vil, liquid, lowest, highest, top])


def _columns(
    h, h_top, v, ceiling, shared, options, base_height, use_compiled, n_threads
):
    """Run the kernel, or its NumPy reference, on the columns ``v``."""
    args = (
        np.ascontiguousarray(h, dtype=np.float64),
        np.ascontiguousarray(h_top, dtype=np.float64),
        np.ascontiguousarray(v, dtype=np.float64),
        np.ascontiguousarray(ceiling, dtype=np.float64),
        bool(shared),
        float("nan") if options["dbz_cap"] is None else float(options["dbz_cap"]),
        float("nan") if options["min_dbz"] is None else float(options["min_dbz"]),
        float(options["top_threshold"]),
        float(options["no_echo_dbz"]),
        bool(options["fill_below"]),
        float(base_height),
        bool(options["interpolate"]),
    )
    if use_compiled:
        return _vil.columns(*args, n_threads=int(n_threads or 0))
    return _columns_numpy(*args)


# --------------------------------------------------------------------------
# geometry of the polar volume
# --------------------------------------------------------------------------


def _ground_range(rng, elevation):
    """Ground distance (arc on the 4/3 Earth) of the gates of a beam [m]."""
    height = beam_center_height(rng, elevation, 0.0)
    reff = EFFECTIVE_RADIUS_4_3
    cos_e = np.cos(np.deg2rad(elevation))
    return reff * np.arcsin(rng * cos_e / (reff + height))


def _height_at_ground_range(ground, elevation, altitude):
    """
    Height above sea level of the beam of elevation ``elevation`` [degrees] at
    ground distance ``ground`` [m]: ``h = a_e cos(e) / cos(e + s / a_e) - a_e``
    (the inverse of the beam geometry of
    :func:`radarx.fundamentals.geometry.beam_center_height`).
    """
    reff = EFFECTIVE_RADIUS_4_3
    e = np.deg2rad(elevation)
    return altitude + reff * (np.cos(e) / np.cos(e + ground / reff) - 1.0)


def _nearest_ray(sweep_azimuth, azimuth):
    """Index of the ray nearest to each ``azimuth`` and the angular distance."""
    a = np.mod(np.asarray(sweep_azimuth, dtype=np.float64), 360.0)
    order = np.argsort(a, kind="stable")
    a = a[order]
    ext = np.r_[a[-1] - 360.0, a, a[0] + 360.0]
    ray = np.r_[order[-1], order, order[0]]
    az = np.mod(np.asarray(azimuth, dtype=np.float64), 360.0)
    j = np.clip(np.searchsorted(ext, az), 1, ext.size - 1)
    pick = np.where(az - ext[j - 1] <= ext[j] - az, j - 1, j)
    return ray[pick], np.abs(ext[pick] - az)


def _beamwidth(dtree, beamwidth):
    """Vertical beam width [degrees]: the argument, the volume, or 1 degree."""
    if beamwidth is not None:
        return float(beamwidth)
    nodes = [dtree.root.to_dataset()]
    if "radar_parameters" in dtree.children:
        nodes.append(dtree["radar_parameters"].to_dataset())
    for ds in nodes:
        for key in ("radar_beam_width_v", "radar_beam_width_h"):
            if key in ds and np.isfinite(float(ds[key])):
                return float(ds[key])
    return _BEAMWIDTH


class _Sweep:
    """A sweep matched to the (azimuth, ground range) columns of the lowest one."""

    def __init__(self, ds, variable, azimuth, ground):
        ray_dim = [d for d in ds[variable].dims if d != "range"][0]
        self.data = ds[variable].transpose(ray_dim, "range").values
        az = np.asarray(ds["azimuth"].values, dtype=np.float64)
        self.elevation = np.ascontiguousarray(
            np.broadcast_to(ds["elevation"].values, az.shape), dtype=np.float64
        )
        spacing = float(np.median(np.diff(np.sort(np.mod(az, 360.0)))))
        self.ray, distance = _nearest_ray(az, azimuth)
        self.in_azimuth = distance <= _MAX_GAP * max(spacing, 1e-6)
        rng = np.asarray(ds["range"].values, dtype=np.float64)
        g = _ground_range(rng, float(np.nanmedian(self.elevation)))
        edges = 0.5 * (g[1:] + g[:-1])
        self.gate = np.clip(np.searchsorted(edges, ground), 0, rng.size - 1)
        self.covered = (ground >= g[0] - 0.5 * (g[1] - g[0])) & (
            ground <= g[-1] + 0.5 * (g[-1] - g[-2])
        )

    def block(self, rows, clear):
        """Values (len(rows), n_ground) with NaN where the sweep has no data."""
        value = self.data[self.ray[rows][:, None], self.gate[None, :]].astype(
            np.float64
        )
        inside = self.in_azimuth[rows][:, None] & self.covered[None, :]
        if clear:
            value = np.where(inside & np.isnan(value), -np.inf, value)
        return np.where(inside, value, np.nan)


def _polar_columns(dtree, variable, tolerance, beamwidth, clear):
    """
    Template, ground range and a generator of kernel inputs for a volume.

    Yields ``(rows, h, h_top, v)`` blocks of whole rays of the lowest sweep.
    """
    names = _sweep_names(dtree)
    if not names:
        raise ValueError("No sweep groups found in DataTree.")
    sweeps = [_sweep_dataset(dtree, name) for name in names]
    selected = _select_sweeps(sweeps, variable, tolerance)
    if not selected:
        raise KeyError(f"no sweep contains {variable!r}")
    low = selected[0]
    template = low[variable]
    ray_dim = [d for d in template.dims if d != "range"][0]
    template = template.transpose(ray_dim, "range")
    elevation = np.broadcast_to(low["elevation"].values, low["azimuth"].shape).astype(
        np.float64
    )
    ground = _ground_range(
        np.asarray(low["range"].values, dtype=np.float64),
        float(np.nanmedian(elevation)),
    )
    azimuth = np.asarray(low["azimuth"].values, dtype=np.float64)
    matched = [_Sweep(ds, variable, azimuth, ground) for ds in selected]
    altitude = float(low["altitude"]) if "altitude" in low else 0.0
    half = 0.5 * _beamwidth(dtree, beamwidth)
    nk = len(matched)

    def blocks():
        per_ray = max(1, _CHUNK // max(ground.size * nk, 1))
        for start in range(0, azimuth.size, per_ray):
            rows = np.arange(start, min(start + per_ray, azimuth.size))
            h = np.empty((nk, rows.size, ground.size))
            h_top = np.empty_like(h)
            v = np.empty_like(h)
            for k, sw in enumerate(matched):
                el = sw.elevation[sw.ray[rows]][:, None]
                h[k] = _height_at_ground_range(ground[None, :], el, altitude)
                h_top[k] = _height_at_ground_range(ground[None, :], el + half, altitude)
                v[k] = sw.block(rows, clear)
            yield (
                rows,
                h.reshape(nk, -1),
                h_top.reshape(nk, -1),
                v.reshape(nk, -1),
            )

    return template, ground, altitude, blocks


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def _find_dbz(ds, name):
    if name is not None:
        if name not in ds:
            raise KeyError(f"{name!r} is not in the dataset")
        return name
    for cand in _DBZ_NAMES:
        if cand in ds:
            return cand
    raise KeyError(f"none of {_DBZ_NAMES} found; pass the field name")


def _kind_of(obj, kind):
    if kind not in (None, "volume", "grid", "qvp"):
        raise ValueError(f"kind must be 'volume', 'grid' or 'qvp', not {kind!r}")
    if isinstance(obj, xr.DataTree):
        if kind not in (None, "volume"):
            raise ValueError("a DataTree is a polar volume")
        return "volume"
    if not isinstance(obj, xr.Dataset):
        raise TypeError("obj must be an xarray.Dataset or xarray.DataTree")
    if kind == "volume":
        raise ValueError("kind='volume' needs an xarray.DataTree of sweeps")
    if kind is not None:
        return kind
    if {"x", "y"} <= set(obj.dims):
        return "grid"
    if "height" in obj.dims:
        return "qvp"
    raise ValueError(
        "cannot tell the kind of the dataset: a grid has x, y and z dimensions "
        "and a quasi-vertical profile a height dimension; pass kind="
    )


def _vertical_dim(da, kind):
    names = _VERTICAL_NAMES if kind == "grid" else ("height",)
    for name in names:
        if name in da.dims:
            return name
    raise ValueError(f"no vertical dimension ({'/'.join(names)}) in {da.dims}")


def _top_edges(z):
    """Upper edge of each level: halfway to the next level, the last by symmetry."""
    z = np.asarray(z, dtype=np.float64)
    if z.size == 1:
        return z.copy()
    mid = 0.5 * (z[1:] + z[:-1])
    return np.r_[mid, z[-1] + 0.5 * (z[-1] - z[-2])]


def _ceiling(melting, template):
    """Ceiling height [m] on the columns of ``template`` (NaN where none)."""
    if melting is None:
        return np.full(template.shape, np.nan)
    if isinstance(melting, xr.Dataset):
        if "melting_layer_bottom" not in melting:
            raise KeyError(
                "melting must be a height or a melting_layer result with "
                "'melting_layer_bottom'"
            )
        melting = melting["melting_layer_bottom"]
    if isinstance(melting, xr.DataArray):
        try:
            out = melting.broadcast_like(template).transpose(*template.dims)
        except ValueError as err:
            raise ValueError(
                f"melting {melting.dims} does not fit the columns {template.dims}"
            ) from err
        if out.shape != template.shape:
            raise ValueError(
                f"melting {melting.sizes} does not match the columns {template.sizes}"
            )
        return np.asarray(out.values, dtype=np.float64)
    return np.full(template.shape, float(melting))


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def _column_products(
    obj,
    dbz,
    kind,
    options,
    melting,
    base_height,
    missing,
    tolerance,
    beamwidth,
    n_threads,
    engine,
):
    """
    Run the column integrals.

    Returns the 5 products as a list of arrays on ``template`` plus
    ``template`` (a DataArray with the output dims and coordinates) and the
    base height used.
    """
    use_compiled = _use_compiled(engine)
    kind = _kind_of(obj, kind)
    if missing not in (None, "clear", "gap"):
        raise ValueError(f"missing must be 'clear' or 'gap', not {missing!r}")
    clear = (missing == "clear") if missing is not None else kind == "volume"
    if options["top_threshold"] <= options["no_echo_dbz"]:
        raise ValueError("no_echo_dbz must be below the echo-top threshold")

    if kind == "volume":
        name = dbz or _volume_field(obj)
        template, ground, altitude, blocks = _polar_columns(
            obj, name, tolerance, beamwidth, clear
        )
        base = altitude if base_height is None else float(base_height)
        ceiling = _ceiling(melting, template).reshape(-1)
        out = np.full((5, template.size), np.nan)
        ns = ground.size
        for rows, h, h_top, v in blocks():
            cols = (rows[:, None] * ns + np.arange(ns)[None, :]).reshape(-1)
            out[:, cols] = _columns(
                h,
                h_top,
                v,
                ceiling[cols],
                False,
                options,
                base,
                use_compiled,
                n_threads,
            )
        template = template.assign_coords(
            ground_range=(
                "range",
                ground,
                {"long_name": "ground range of the gate", "units": "m"},
            )
        )
        attrs = dict(template.attrs)
    else:
        name = _find_dbz(obj, dbz)
        da = obj[name]
        zdim = _vertical_dim(da, kind)
        da = da.transpose(zdim, ...)
        template = da.isel({zdim: 0}, drop=True)
        z = np.asarray(da[zdim].values, dtype=np.float64)
        site = obj["altitude"] if "altitude" in obj else None
        if base_height is not None:
            base = float(base_height)
        elif site is not None and site.size == 1:
            base = float(site)
        else:
            base = 0.0
        v = da.values.reshape(z.size, -1)
        if clear:
            seen = np.maximum.accumulate(~np.isnan(v), axis=0)
            v = np.where(seen & np.isnan(v), -np.inf, v)
        ceiling = _ceiling(melting, template).reshape(-1)
        out = _columns(
            z,
            _top_edges(z),
            v,
            ceiling,
            True,
            options,
            base,
            use_compiled,
            n_threads,
        )
        attrs = dict(da.attrs)
    products = [out[i].reshape(template.shape) for i in range(5)]
    return products, template, base, attrs


def _volume_field(dtree):
    for name in _sweep_names(dtree):
        ds = _sweep_dataset(dtree, name)
        for cand in _DBZ_NAMES:
            if cand in ds:
                return cand
    raise KeyError(f"none of {_DBZ_NAMES} found in a sweep; pass the field name")


def _options(
    dbz_cap=56.0,
    min_dbz=0.0,
    fill_below=False,
    top_threshold=18.0,
    no_echo_dbz=-14.0,
    interpolate=True,
):
    return {
        "dbz_cap": dbz_cap,
        "min_dbz": min_dbz,
        "fill_below": fill_below,
        "top_threshold": float(top_threshold),
        "no_echo_dbz": float(no_echo_dbz),
        "interpolate": interpolate,
    }


def _wrap(template, data, name, attrs):
    out = template.copy(data=np.asarray(data, dtype=np.float32))
    out.attrs = dict(attrs)
    out.name = name
    return out


def _flag(template, vil, lowest, base, fill_below):
    """1 where the VIL is a lower bound (a layer under the lowest sample is left out)."""
    with np.errstate(invalid="ignore"):
        bound = np.isfinite(vil) & (lowest - base > 1.0) & (not fill_below)
    out = template.copy(data=bound.astype(np.int8))
    out.attrs = {
        "long_name": "VIL is a lower bound",
        "comment": (
            "1: the layer between the base height and the lowest valid sample "
            "is not included"
        ),
        "flag_values": np.array([0, 1], dtype=np.int8),
        "flag_meanings": "complete lower_bound",
    }
    out.name = "VIL_LOWER_BOUND"
    return out


# --------------------------------------------------------------------------
# public functions
# --------------------------------------------------------------------------

_COMMON_PARAMS = """
    dbz : str, optional
        Name of the reflectivity (dBZ). By default the first found of
        ``DBZH``, ``DBZ``, ``reflectivity`` and ``corrected_reflectivity``
        (for a volume, in the first sweep that has one).
    kind : {"volume", "grid", "qvp"}, optional
        Kind of input. By default a DataTree is a polar volume, a Dataset with
        ``x`` and ``y`` is a grid and one with ``height`` a quasi-vertical
        profile.
    missing : {"clear", "gap"}, optional
        What a missing (NaN) value means. ``"clear"``: no echo (the
        reflectivity of the gate is below the noise floor), which is how
        radar volumes mark it; for a grid or a profile only the values above
        the lowest valid level are taken as no echo. ``"gap"``: no data, left
        out and bridged linearly. Default ``"clear"`` for a volume, where a
        gate outside the range of a sweep is always left out, and ``"gap"``
        for a grid or a profile, where a missing value can also be a level
        that no beam reached.
    base_height : float, optional
        Height of the ground (m above sea level) that the column starts at.
        Default: the ``altitude`` of the radar site, else 0.
    tolerance : float, optional
        Volume only: sweeps closer than this in elevation (degrees) count as
        the same cut, and the one that reaches farthest is used. Default 0.1.
    beamwidth : float, optional
        Volume only: vertical beam width in degrees for the top of the highest
        beam. By default the value stored in the volume
        (``radar_beam_width_v``), else 1 degree.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled
        kernel and falls back to NumPy."""


def vil(
    obj,
    dbz=None,
    *,
    dbz_cap=56.0,
    min_dbz=0.0,
    fill_below=False,
    melting=None,
    kind=None,
    missing=None,
    base_height=None,
    tolerance=0.1,
    beamwidth=None,
    n_threads=None,
    engine="auto",
):
    """
    Vertically integrated liquid (VIL) of the reflectivity columns.

    Implements the sum of Greene and Clark (1972) over the layers between
    consecutive samples of a column,
    ``VIL = 3.44e-6 sum[((Z_i + Z_{i+1}) / 2)**(4/7) dh]`` in kg m-2 with
    ``Z`` in mm6 m-3 and ``dh`` in m. See :mod:`radarx.retrieve.vil` for the
    columns of the three kinds of input and the integration limits.

    Not checked against Greene and Clark (1972), which could not be
    obtained. The coefficient, the exponent and the units are confirmed by the
    integral form in Seo et al. (2020, Eq. 3-4); the 56 dBZ cap, the 0 dBZ
    ``min_dbz`` and the linear variation of Z with height between samples are
    radarx choices.

    Parameters
    ----------
    obj : xarray.DataTree or xarray.Dataset
        A polar volume with ``sweep_*`` groups (sweeps without the field are
        skipped), a grid, or a quasi-vertical profile (time series).
    dbz_cap : float or None, optional
        Reflectivity (dBZ) above which values are set to this cap. Default 56
        (the operational product); ``None`` for no cap.
    min_dbz : float or None, optional
        Reflectivity (dBZ) below which a value counts as no echo
        (Z = 0 mm6 m-3). Default 0; ``None`` for no threshold.
    fill_below : bool, optional
        Extend the lowest valid value down to ``base_height`` (like a
        pseudo-CAPPI) instead of leaving that layer out. Default False.
    melting : float or xarray.DataArray or xarray.Dataset, optional
        Height (m above sea level) of the melting level, a number, an array
        that broadcasts against the columns (e.g. the 0 degC height of a
        sounding or ERA5 per time) or the result of
        :func:`radarx.retrieve.melting_layer`, whose ``melting_layer_bottom``
        is used. If given, ``VIL_LIQUID`` is also returned: the VIL below that
        height, with the profile cut by linear interpolation.
    {common}

    Returns
    -------
    xarray.Dataset
        On the horizontal coordinates of the input (the polar grid of the
        lowest sweep for a volume, the profile times for a QVP):

        ``VIL`` (kg m-2)
            The vertically integrated liquid. NaN where the column has fewer
            than two valid samples.
        ``VIL_LIQUID`` (kg m-2)
            Only with ``melting``.
        ``VIL_LOWEST_HEIGHT`` (m)
            Height of the lowest sample used, the lower integration limit.
        ``VIL_LOWER_BOUND``
            1 where the VIL is a lower bound because the layer between
            ``base_height`` and the lowest sample is left out (always with
            ``fill_below=False`` unless that sample is at the base). Nothing
            is assumed above the highest valid sample, which is not flagged.

    Raises
    ------
    KeyError
        If the reflectivity or no sweep is found.
    ValueError
        For unknown options or an input that does not fit.
    ImportError
        If ``engine="compiled"`` and the compiled kernel is not available.

    Notes
    -----
    Near the radar the beams of a volume are far apart in height and the
    lowest one starts at a height well above the ground, so the VIL there is
    low (the "cone of silence"); ``VIL_LOWER_BOUND`` marks it. The VIL of a
    quasi-vertical profile is the VIL of the azimuthal-mean profile. For a
    profile that is the mean in linear Z it is not lower than the mean of the
    VIL of the columns of the same volume, since ``Z**(4/7)`` is concave
    (Jensen's inequality); a mean in dBZ or a median gives a lower value.
    The reflectivity of the melting layer is
    enhanced and the water below it is not what the formula assumes, so use
    ``melting`` to get the liquid part.

    References
    ----------
    Greene, D. R., and R. A. Clark, 1972: Vertically integrated liquid water
    --- a new analysis tool. *Mon. Wea. Rev.*, **100** (7), 548-552,
    https://doi.org/10.1175/1520-0493(1972)100<0548:VILWNA>2.3.CO;2
    (sum over layers and the 3.44e-6 coefficient as quoted by the studies
    below; not checked against the paper).

    Seo, B.-C., W. F. Krajewski, and Y. Qi, 2020: Utility of vertically
    integrated liquid water content for radar-rainfall estimation: Quality
    control and precipitation type classification. *Atmos. Res.*, **236**,
    104800, https://doi.org/10.1016/j.atmosres.2019.104800 (Eq. 3 for
    ``LWC = 3.44e-6 Z**(4/7)`` in kg m-3 with Z in mm6 m-3, derived from the
    exponential distribution of Marshall and Palmer 1948; Eq. 4 for the
    integral over the beam heights).

    Marshall, J. S., and W. McK. Palmer, 1948: The distribution of raindrops
    with size. *J. Meteor.*, **5** (4), 165-166,
    https://doi.org/10.1175/1520-0469(1948)005<0165:TDORWS>2.0.CO;2

    Examples
    --------
    >>> out = radarx.retrieve.vil(grid)  # doctest: +SKIP
    >>> out = dtree.radarx.vil(fill_below=True)  # doctest: +SKIP
    """
    options = _options(dbz_cap, min_dbz, fill_below)
    products, template, base, attrs = _column_products(
        obj,
        dbz,
        kind,
        options,
        melting,
        base_height,
        missing,
        tolerance,
        beamwidth,
        n_threads,
        engine,
    )
    vil_, liquid, lowest, _, _ = products
    cap = "none" if dbz_cap is None else f"{dbz_cap:g} dBZ"
    comment = (
        "Greene and Clark (1972) sum of 3.44e-6 ((Z_i + Z_i+1) / 2)^(4/7) dh; "
        f"reflectivity cap {cap}; values below "
        f"{'no threshold' if min_dbz is None else f'{min_dbz:g} dBZ'} count as "
        f"no echo; the layer below the lowest sample is "
        f"{'filled with its value' if fill_below else 'left out'}"
    )
    out = xr.Dataset(
        {
            "VIL": _wrap(
                template,
                vil_,
                "VIL",
                {
                    "long_name": "vertically integrated liquid",
                    "units": "kg m-2",
                    "comment": comment,
                },
            )
        }
    )
    if melting is not None:
        out["VIL_LIQUID"] = _wrap(
            template,
            liquid,
            "VIL_LIQUID",
            {
                "long_name": "vertically integrated liquid below the melting level",
                "units": "kg m-2",
                "comment": comment,
            },
        )
    out["VIL_LOWEST_HEIGHT"] = _wrap(
        template,
        lowest,
        "VIL_LOWEST_HEIGHT",
        {
            "long_name": "lower integration limit (height of the lowest sample)",
            "units": "m",
            "comment": "height above sea level",
        },
    )
    out["VIL_LOWER_BOUND"] = _flag(template, vil_, lowest, base, fill_below)
    out.attrs["vil_base_height"] = base
    return out


vil.__doc__ = vil.__doc__.replace("{common}", _COMMON_PARAMS.strip("\n"))


def echo_top(
    obj,
    dbz=None,
    *,
    threshold=18.0,
    interpolate=True,
    no_echo_dbz=-14.0,
    kind=None,
    missing=None,
    base_height=None,
    tolerance=0.1,
    beamwidth=None,
    n_threads=None,
    engine="auto",
):
    """
    Echo-top height: the highest reflectivity at or above a threshold.

    Implements the algorithm of Lakshmanan et al. (2013). In each column the
    highest sample with reflectivity ``Z_b >= threshold`` is found. If a
    higher sample exists, with reflectivity ``Z_a`` (set to ``no_echo_dbz`` if
    it is lower, as a beam above the echo without signal), the echo top is
    where the reflectivity, varying linearly in dBZ with height between the
    two samples, equals the threshold: ``h_b + (Z_b - T) / (Z_b - Z_a) (h_a -
    h_b)``. If the highest sample is also the highest in the column, the top
    of its beam is used (traditional algorithm). Heights are above sea level.

    Differences from the paper. (1) Lakshmanan et al. interpolate the
    elevation angle ``theta_T`` of the beam at a fixed range and then take
    the height of that beam. Here the interpolation is linear in the beam
    height at a fixed ground range (volume), or in the level height (grid,
    profile); the two agree to first order for the small elevation steps of a
    volume. (2) The paper's Eq. (1), as printed, ends with ``+ theta_b``,
    which does not reproduce ``theta_b`` for ``Z_T = Z_b`` or ``theta_a`` for
    ``Z_T = Z_a``; linear interpolation as described in the text and in their
    footnote 1, ``theta_a + (Z_T - Z_a)(theta_b - theta_a) / (Z_b - Z_a)``,
    is what is implemented (in height). (3) The top of the highest beam is
    ``beamwidth / 2`` above the beam centre for a volume, and half a level
    spacing for a grid or a profile. (4) A gate without echo inside a sweep
    (``missing="clear"``) is the paper's "below the signal-to-noise cutoff"
    case. Sweeps that do not reach the column are not "higher scans" and are
    ignored.

    Parameters
    ----------
    obj : xarray.DataTree or xarray.Dataset
        A polar volume with ``sweep_*`` groups (sweeps without the field are
        skipped), a grid, or a quasi-vertical profile.
    threshold : float, optional
        Reflectivity (dBZ) of the echo top. Default 18, the WSR-88D echo-top
        product (Lakshmanan et al. 2013).
    interpolate : bool, optional
        Interpolate towards the next higher sample (default). ``False`` gives
        the traditional echo top, the top of the beam of the highest sample
        at or above the threshold.
    no_echo_dbz : float, optional
        Reflectivity assigned to a higher beam without echo. Default -14 dBZ,
        the lowest value reported by the WSR-88D (Lakshmanan et al. 2013).
    {common}

    Returns
    -------
    xarray.DataArray
        ``ECHO_TOP`` (m above sea level); NaN where no sample reaches the
        threshold.

    Raises
    ------
    KeyError
        If the reflectivity or no sweep is found.
    ValueError
        For unknown options, or if ``no_echo_dbz`` is not below ``threshold``.

    References
    ----------
    Lakshmanan, V., K. Hondl, C. K. Potvin, and D. Preignitz, 2013: An
    improved method for estimating radar echo-top height. *Wea. Forecasting*,
    **28** (2), 481-488, https://doi.org/10.1175/WAF-D-12-00084.1 (the
    algorithm: section 2, steps i-iii, Eq. 1 and footnote 1; the 18 dBZ
    threshold and the -14 dBZ value: section 2; interpolation in dBZ rather
    than in Z: p. 483).

    Examples
    --------
    >>> top = radarx.retrieve.echo_top(grid)  # doctest: +SKIP
    >>> top = dtree.radarx.echo_top(threshold=10.0)  # doctest: +SKIP
    """
    options = _options(
        top_threshold=threshold, no_echo_dbz=no_echo_dbz, interpolate=interpolate
    )
    products, template, _, _ = _column_products(
        obj,
        dbz,
        kind,
        options,
        None,
        base_height,
        missing,
        tolerance,
        beamwidth,
        n_threads,
        engine,
    )
    return _wrap(
        template,
        products[4],
        "ECHO_TOP",
        {
            "long_name": f"echo-top height ({threshold:g} dBZ)",
            "units": "m",
            "comment": (
                "height above sea level; "
                + (
                    "interpolated in dBZ between the beams that bracket the "
                    "threshold (Lakshmanan et al. 2013)"
                    if interpolate
                    else "top of the beam of the highest sample at or above "
                    "the threshold"
                )
            ),
        },
    )


echo_top.__doc__ = echo_top.__doc__.replace("{common}", _COMMON_PARAMS.strip("\n"))


def vil_density(
    obj,
    dbz=None,
    *,
    dbz_cap=56.0,
    min_dbz=0.0,
    fill_below=False,
    threshold=18.0,
    kind=None,
    missing=None,
    base_height=None,
    tolerance=0.1,
    beamwidth=None,
    n_threads=None,
    engine="auto",
):
    """
    VIL density: the VIL divided by the height of the echo top.

    ``VIL_DENSITY = 1000 VIL / (H_top - base_height)`` in g m-3, with the VIL
    of :func:`vil` (kg m-2) and the echo top of :func:`echo_top`. VIL density
    was introduced as a hail indicator by Amburn and Wolf (1997); values
    above about 3.5 g m-3 are used as a sign of large hail in the literature
    on the product (not checked against the paper).

    Not checked against Amburn and Wolf (1997), which could not be
    obtained, so the definition of the echo top there (threshold, above ground
    or sea level) is not confirmed. Here the echo top is that of
    :func:`echo_top` (18 dBZ by default) and the height is taken above
    ``base_height``, a radarx choice. Both products use the same options as in
    :func:`vil` and :func:`echo_top`, so the density is a lower bound where
    ``VIL_LOWER_BOUND`` of :func:`vil` is set.

    Parameters
    ----------
    obj : xarray.DataTree or xarray.Dataset
        A polar volume, a grid or a quasi-vertical profile, see :func:`vil`.
    dbz_cap, min_dbz, fill_below : optional
        Options of :func:`vil`.
    threshold : float, optional
        Echo-top threshold in dBZ, see :func:`echo_top`. Default 18.
    {common}

    Returns
    -------
    xarray.DataArray
        ``VIL_DENSITY`` (g m-3); NaN where the VIL or the echo top is missing,
        or the echo top is not above ``base_height``.

    References
    ----------
    Amburn, S. A., and P. L. Wolf, 1997: VIL density as a hail indicator.
    *Wea. Forecasting*, **12** (3), 473-478,
    https://doi.org/10.1175/1520-0434(1997)012<0473:VDAAHI>2.0.CO;2
    (definition as the ratio of VIL to echo top; not checked against the
    paper).

    Examples
    --------
    >>> density = radarx.retrieve.vil_density(grid)  # doctest: +SKIP
    """
    options = _options(dbz_cap, min_dbz, fill_below, top_threshold=threshold)
    products, template, base, _ = _column_products(
        obj,
        dbz,
        kind,
        options,
        None,
        base_height,
        missing,
        tolerance,
        beamwidth,
        n_threads,
        engine,
    )
    vil_, _, _, _, top = products
    depth = top - base
    with np.errstate(invalid="ignore", divide="ignore"):
        density = np.where(depth > 0, 1000.0 * vil_ / depth, np.nan)
    return _wrap(
        template,
        density,
        "VIL_DENSITY",
        {
            "long_name": "VIL density",
            "units": "g m-3",
            "comment": (
                "VIL divided by the echo-top height above the base height "
                f"({threshold:g} dBZ echo top)"
            ),
        },
    )


vil_density.__doc__ = vil_density.__doc__.replace(
    "{common}", _COMMON_PARAMS.strip("\n")
)


# --------------------------------------------------------------------------
# liquid water content
# --------------------------------------------------------------------------

_UNIT_FACTOR = {"g m-3": 1.0, "kg m-3": 1.0e-3}


def _lwc_zm(ds, dbz, mask, factor, units):
    name = _find_dbz(ds, dbz)
    z = np.power(10.0, ds[name].astype(np.float64) / 10.0)
    lwc = 3.44e-3 * factor * z**EXPONENT  # g m-3 times the unit factor
    if mask is not None:
        lwc = lwc.where(mask)
    lwc = lwc.astype(np.float32)
    lwc.name = "LWC"
    lwc.attrs = {
        "long_name": "liquid water content of rain",
        "units": units,
        "comment": (
            "power law LWC = 3.44e-3 Z^(4/7) g m-3 (Z in mm6 m-3), exponential "
            "drop size distribution; valid in rain only, overestimates in ice"
        ),
    }
    return lwc


def _dsd_lwc(lwc, factor, units):
    """The LWC of a DSD retrieval (g m-3) in the requested units, with attributes."""
    lwc = lwc * factor
    lwc.name = "LWC"
    lwc.attrs = {
        "long_name": "liquid water content of rain",
        "units": units,
        "comment": (
            "third moment of the gamma drop size distribution retrieved by "
            "radarx.retrieve.dsd from ZH, ZDR (and KDP); valid in rain only"
        ),
    }
    return lwc


def _lwc_dsd(ds, dbz, factor, units, mask, dsd_kwargs, n_threads, engine):
    from .dsd import dsd as _dsd

    res = _dsd(
        ds, dbzh=dbz, mask=mask, n_threads=n_threads, engine=engine, **dsd_kwargs
    )
    return _dsd_lwc(res["LWC"], factor, units)


def liquid_water_content(
    obj,
    method="zm",
    *,
    dbz=None,
    mask=None,
    units="g m-3",
    n_threads=None,
    engine="auto",
    **dsd_kwargs,
):
    """
    Liquid water content of rain, gate by gate.

    ``method="zm"`` is the power law ``LWC = 3.44e-3 Z**(4/7)`` in g m-3 with
    ``Z`` in mm6 m-3, the relation behind the VIL of Greene and Clark (1972)
    for an exponential drop size distribution (Seo et al. 2020, Eq. 3, who
    give ``3.44e-6 Z**(4/7)`` in kg m-3). ``method="dsd"`` retrieves a gamma
    drop size distribution from ``Z``, ``Z_DR`` (and ``K_DP``) with
    :func:`radarx.retrieve.dsd` and returns its liquid water content, so it
    does not assume a distribution and needs a polarimetric radar.

    The liquid water content from radar is meaningful in rain only, below the
    melting layer. Above it the reflectivity is that of ice and the power law
    overestimates the water content; in the melting layer the reflectivity is
    enhanced. Mask the other gates (``mask``). Ice water content needs
    a different method, which is not implemented.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataTree
        A sweep, a grid, a quasi-vertical profile or any dataset with the
        reflectivity; for ``"dsd"`` also ``ZDR``. A DataTree is processed
        sweep by sweep (sweeps without the fields are skipped).
    method : {"zm", "dsd"}, optional
        Power law (default) or gamma drop size distribution.
    dbz : str, optional
        Name of the reflectivity (dBZ), by default ``DBZH``, ``DBZ``,
        ``reflectivity`` or ``corrected_reflectivity``.
    mask : str, xarray.DataArray or bool xarray.DataTree, optional
        Rain gates (``True``), e.g. ``height < melting level``; the other
        gates are NaN. For a DataTree, a DataTree of masks as in
        :func:`radarx.retrieve.dsd` (for ``"zm"`` a mask per sweep node with
        one boolean variable, or a dict by sweep name).
    units : {"g m-3", "kg m-3"}, optional
        Units of the result. Default ``"g m-3"``.
    n_threads, engine : optional
        For ``"dsd"``, see :func:`radarx.retrieve.dsd`. The power law is a
        single array expression and needs no kernel.
    **dsd_kwargs
        For ``"dsd"``, options of :func:`radarx.retrieve.dsd` (``zdr``,
        ``kdp``, ``band``, ``temperature``, ...). ``method``, ``dbzh`` and
        ``mask`` are set here.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        ``LWC`` with ``units`` on the coordinates of the reflectivity; for a
        DataTree one node per sweep with a ``LWC`` variable.

    Raises
    ------
    KeyError
        If the reflectivity (or for ``"dsd"`` ZDR) is not found.
    ValueError
        For an unknown ``method`` or ``units``, or ``dsd_kwargs`` with
        ``method="zm"``.

    References
    ----------
    Seo, B.-C., W. F. Krajewski, and Y. Qi, 2020: Utility of vertically
    integrated liquid water content for radar-rainfall estimation: Quality
    control and precipitation type classification. *Atmos. Res.*, **236**,
    104800, https://doi.org/10.1016/j.atmosres.2019.104800 (Eq. 3)

    Greene, D. R., and R. A. Clark, 1972: Vertically integrated liquid water
    --- a new analysis tool. *Mon. Wea. Rev.*, **100** (7), 548-552,
    https://doi.org/10.1175/1520-0493(1972)100<0548:VILWNA>2.3.CO;2
    (not checked against the paper)

    Examples
    --------
    >>> lwc = radarx.retrieve.liquid_water_content(sweep)  # doctest: +SKIP
    >>> lwc = sweep.radarx.liquid_water_content("dsd", kdp="KDP")  # doctest: +SKIP
    """
    if method not in ("zm", "dsd"):
        raise ValueError(f"method must be 'zm' or 'dsd', not {method!r}")
    if units not in _UNIT_FACTOR:
        raise ValueError(f"units must be 'g m-3' or 'kg m-3', not {units!r}")
    if method == "zm" and dsd_kwargs:
        raise ValueError(
            f"{sorted(dsd_kwargs)} are options of method='dsd', not of 'zm'"
        )
    factor = _UNIT_FACTOR[units]

    def one(ds, node_mask):
        if method == "zm":
            m = ds[node_mask] if isinstance(node_mask, str) else node_mask
            return _lwc_zm(ds, dbz, m, factor, units)
        return _lwc_dsd(
            ds, dbz, factor, units, node_mask, dsd_kwargs, n_threads, engine
        )

    if not isinstance(obj, xr.DataTree):
        if isinstance(obj, xr.DataArray):
            raise TypeError("obj must be an xarray.Dataset or xarray.DataTree")
        return one(obj, mask)

    if method == "dsd":
        res = _lwc_dsd_tree(
            obj, dbz, factor, units, mask, dsd_kwargs, n_threads, engine
        )
        return res
    nodes = {"/": obj.root.to_dataset(inherit=False)}
    for name in _sweep_names(obj):
        ds = _sweep_dataset(obj, name)
        if all(c not in ds for c in ([dbz] if dbz else _DBZ_NAMES)):
            continue
        nodes[name] = one(ds, _node_mask(mask, name)).to_dataset()
    if len(nodes) == 1:
        raise KeyError("no sweep contains the reflectivity")
    return xr.DataTree.from_dict(nodes)


def _node_mask(mask, name):
    if isinstance(mask, (xr.DataTree, dict)):
        if name not in mask:
            return None
        node = mask[name]
        if isinstance(node, xr.DataTree):
            node = node.to_dataset()
        if isinstance(node, xr.Dataset):
            if len(node.data_vars) != 1:
                raise ValueError(
                    "each mask node must hold exactly one boolean variable"
                )
            node = node[next(iter(node.data_vars))]
        return node
    return mask


def _lwc_dsd_tree(dtree, dbz, factor, units, mask, dsd_kwargs, n_threads, engine):
    from .dsd import dsd as _dsd

    res = _dsd(
        dtree, dbzh=dbz, mask=mask, n_threads=n_threads, engine=engine, **dsd_kwargs
    )
    nodes = {"/": res.root.to_dataset(inherit=False)}
    for name in res.children:
        nodes[name] = _dsd_lwc(res[name].ds["LWC"], factor, units).to_dataset()
    return xr.DataTree.from_dict(nodes)


# --------------------------------------------------------------------------
# accessors
# --------------------------------------------------------------------------


@accessor_method("dataset", "datatree", name="vil")
def _vil_accessor(self, **kwargs):
    """
    Vertically integrated liquid of a volume, grid or profile.

    See :func:`radarx.retrieve.vil` for the parameters.

    Returns
    -------
    xarray.Dataset
        ``VIL`` (kg m-2) with the lower-bound flag and the lowest height.
    """
    return vil(self.xarray_obj, **kwargs)


@accessor_method("dataset", "datatree", name="echo_top")
def _echo_top_accessor(self, **kwargs):
    """
    Echo-top height of a volume, grid or profile.

    See :func:`radarx.retrieve.echo_top` for the parameters.

    Returns
    -------
    xarray.DataArray
        ``ECHO_TOP`` (m above sea level).
    """
    return echo_top(self.xarray_obj, **kwargs)


@accessor_method("dataset", "datatree", name="vil_density")
def _vil_density_accessor(self, **kwargs):
    """
    VIL density of a volume, grid or profile.

    See :func:`radarx.retrieve.vil_density` for the parameters.

    Returns
    -------
    xarray.DataArray
        ``VIL_DENSITY`` (g m-3).
    """
    return vil_density(self.xarray_obj, **kwargs)


@accessor_method("dataset", "datatree", name="liquid_water_content")
def _liquid_water_content_accessor(self, method="zm", **kwargs):
    """
    Liquid water content of rain, gate by gate.

    See :func:`radarx.retrieve.liquid_water_content` for the parameters.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        ``LWC``.
    """
    return liquid_water_content(self.xarray_obj, method, **kwargs)
