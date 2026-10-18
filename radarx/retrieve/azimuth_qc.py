#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx Ray-Angle Quality Control
================================

Find and repair jumps in the ray angles of a sweep: single rays whose
azimuth is wrong, duplicated rays, rays out of scan order and gaps where rays
are missing. Every later step (gridding, shear, dealiasing, Doppler-wind
synthesis) assumes that the angle of a ray is the direction the beam pointed.

Provenance. Antenna pointing errors are a known problem of weather radar and
are monitored with the sun and with radar networks (Huuskonen and Holleman
2007; Holleman et al. 2010; Louf and Protat 2023); those methods estimate a
constant bias of the antenna angles, not the jumps within one sweep treated
here. The detection in this module is radarx's own. It uses the median as
the robust location (Hampel 1974; Leys et al. 2013 argue for the median
absolute deviation over the standard deviation), applied to the angle
increments from ray to ray. The tolerances (4 ray spacings for an outlier, 2.5
for a gap, 0.1 for a duplicate, up to 5 rays on each side) are radarx
choices that were tested on synthetic sweeps and on one sector PPI with
missing rays, not derived from a paper.

For every sweep, the typical angle step from ray to ray ``delta`` is the
median of the steps (signed: its sign is the scan direction). A ray is an
**outlier** when it deviates by more than ``tolerance`` spacings from the
positions that the rays on its left predict and also from those that the
rays on its right predict; the prediction of one neighbour is its angle
extrapolated with ``delta``, and the deviation is the median over up to
``window`` neighbours on that side. Needing both sides keeps a persistent
jump from being reported as a run of outliers. Among the rays that are not
outliers, a step of less than ``duplicate_fraction`` spacings marks a
**duplicate**, a step against the scan direction a **reversal**, and a step
of more than ``gap_factor`` spacings forward a **gap** (the ray before it is
flagged and ``azimuth_missing_rays`` gives the number of missing rays).
Angles that are not finite are **missing**.

Outliers, duplicates and missing angles are repaired by linear interpolation
in the ray number between the nearest good rays (extrapolated with ``delta``
at the ends of the sweep). Gaps and reversals are reported and are not
repaired, since the true angles are not known; ``method="regularize"``
returns the sweep on a regular angle grid with the rays of a gap filled with
missing values.

Limits. A run of more than about half of ``window`` consecutive bad rays is
not told from a jump of the scan; fewer than three good rays at an end of
the sweep before a jump cannot be told from outliers. A scan that does not
move (a stare, ``|delta|`` below 0.01 degrees per ray) is left alone. For an
RHI, the elevation is checked in the same way as the azimuth of a PPI.

A compiled C++ kernel does the work; if it is not available, an equivalent
NumPy implementation is used.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = ["AZIMUTH_FLAGS", "azimuth_flags", "fix_azimuth"]

__doc__ = __doc__.format("\n   ".join(__all__[1:]))

import numpy as np
import xarray as xr

from .._fields import as_sweep
from .._provenance import provenance
from .._registry import accessor_method
from .._sweepwise import dask_support
from ._azimuth_qc_numpy import (
    DUPLICATE,
    GAP,
    HAS_COMPILED_KERNEL,
    MISSING,
    OUTLIER,
    REVERSAL,
    flag_azimuth,
    wrap180,
    wrap360,
)
from ._fieldtools import choose_engine
from ._products import product_tree

#: Bit of every kind of flag in ``azimuth_flag``.
AZIMUTH_FLAGS = {
    "outlier": OUTLIER,
    "duplicate": DUPLICATE,
    "gap": GAP,
    "reversal": REVERSAL,
    "missing": MISSING,
}

_REFERENCES = """
    References
    ----------
    .. [1] Hampel, F. R., 1974: The influence curve and its role in robust
       estimation. *J. Amer. Stat. Assoc.*, **69**, 383-393,
       https://doi.org/10.1080/01621459.1974.10482962
    .. [2] Leys, C., C. Ley, O. Klein, P. Bernard, and L. Licata, 2013:
       Detecting outliers: Do not use standard deviation around the mean, use
       absolute deviation around the median. *J. Exp. Soc. Psychol.*, **49**,
       764-766, https://doi.org/10.1016/j.jesp.2013.03.013
    .. [3] Huuskonen, A., and I. Holleman, 2007: Determining weather radar
       antenna pointing using signals detected from the sun at low antenna
       elevations. *J. Atmos. Oceanic Technol.*, **24**, 476-483,
       https://doi.org/10.1175/jtech1978.1
    .. [4] Holleman, I., A. Huuskonen, M. Kurri, and H. Beekhuis, 2010:
       Operational monitoring of weather radar receiving chain using the sun.
       *J. Atmos. Oceanic Technol.*, **27**, 159-166,
       https://doi.org/10.1175/2009jtecha1213.1
    .. [5] Louf, V., and A. Protat, 2023: Real-time monitoring of weather radar
       network calibration and antenna pointing. *J. Atmos. Oceanic Technol.*,
       **40**, 823-844, https://doi.org/10.1175/jtech-d-22-0118.1
"""

_FLAG_ATTRS = {
    "long_name": "quality flags of the ray angle",
    "flag_masks": np.array(list(AZIMUTH_FLAGS.values()), dtype=np.uint8),
    "flag_meanings": " ".join(AZIMUTH_FLAGS),
    "comment": (
        "outlier: bad angle, repaired; duplicate: same angle as the previous ray, repaired; "
        "gap: rays are missing after this ray, not repaired; reversal: step against the scan "
        "direction, not repaired; missing: the angle is not finite, repaired"
    ),
}


_FILLED_ATTRS = {"long_name": "ray added to fill a gap (no data)"}


def _angle_name(ds):
    """Name of the ray angle of a sweep: the dimension 'azimuth' or 'elevation'."""
    for name in ("azimuth", "elevation"):
        if name in ds.dims:
            return name
    for name in ("azimuth", "elevation"):
        if name in ds.variables and ds[name].ndim == 1:
            return name
    raise ValueError("the sweep needs an 'azimuth' or 'elevation' along its rays")


def _ray_dim(ds, name):
    return ds[name].dims[0]


def _sweeps(obj):
    """``(names, datasets)`` of the sweeps of ``obj`` (a Dataset gives one with name None)."""
    obj = as_sweep(obj)
    if isinstance(obj, xr.Dataset):
        return [None], [obj]
    if hasattr(obj, "children"):
        names, sets = [], []
        for name, node in obj.children.items():
            if not name.startswith("sweep"):
                continue
            try:
                ds = node.to_dataset(inherit="all_coords")
            except (TypeError, ValueError):  # pragma: no cover - older xarray
                ds = node.to_dataset()
            names.append(name)
            sets.append(ds)
        if not sets:
            raise ValueError("The volume has no sweep nodes.")
        return names, sets
    raise TypeError("obj must be an xarray.Dataset sweep or an xarray.DataTree volume")


def _check(tolerance, window, duplicate_fraction, gap_factor):
    if not tolerance > 0:
        raise ValueError("tolerance must be positive")
    if int(window) != window or window < 2:
        raise ValueError("window must be an integer of at least 2")
    if not 0 <= duplicate_fraction < 1:
        raise ValueError("duplicate_fraction must be in [0, 1)")
    if not gap_factor > 1:
        raise ValueError("gap_factor must be larger than 1")


def _flag_sweeps(
    sweeps, tolerance, window, duplicate_fraction, gap_factor, n_threads, engine
):
    _check(tolerance, window, duplicate_fraction, gap_factor)
    use_compiled = choose_engine(engine, HAS_COMPILED_KERNEL, "ray-angle")
    names = [_angle_name(ds) for ds in sweeps]
    angles = [
        np.asarray(ds[n].values, dtype=np.float64) for ds, n in zip(sweeps, names)
    ]
    results = flag_azimuth(
        angles,
        tolerance,
        int(window),
        duplicate_fraction,
        gap_factor,
        use_compiled,
        n_threads,
    )
    return names, angles, results


def _missing_rays(flags, fixed, delta):
    """Number of rays missing after every ray that is followed by a gap."""
    out = np.zeros(flags.size, dtype=np.int16)
    for i in np.flatnonzero(flags & GAP):
        if i + 1 < fixed.size and np.isfinite(delta) and delta != 0:
            steps = wrap180(fixed[i + 1] - fixed[i]) / delta
            out[i] = max(int(np.rint(steps)) - 1, 0)
    return out


def _flag_dataset(ds, name, flags, fixed, delta, params):
    dim = _ray_dim(ds, name)
    coords = {c: ds.coords[c] for c in ds.coords if ds.coords[c].dims == (dim,)}
    attrs = {"angle": name, **params}
    out = xr.Dataset(coords=coords)
    out["azimuth_flag"] = xr.DataArray(flags, dims=(dim,), attrs=_FLAG_ATTRS | attrs)
    out["azimuth_fixed"] = xr.DataArray(
        fixed,
        dims=(dim,),
        attrs={"long_name": f"repaired ray {name}", "units": "degrees", **attrs},
    )
    out["azimuth_missing_rays"] = xr.DataArray(
        _missing_rays(flags, fixed, delta),
        dims=(dim,),
        attrs={"long_name": "number of rays missing after this ray", **attrs},
    )
    out["azimuth_step"] = xr.DataArray(
        float(delta),
        attrs={
            "long_name": "typical angle step from ray to ray (the sign is the scan direction)",
            "units": "degrees",
            **attrs,
        },
    )
    return out


@dask_support(
    "computes", "The ray angles are read into memory; the fields are not touched."
)
@provenance(
    "Ray-angle check: outliers, duplicates, gaps and reversals (Hampel 1974; Leys et al. 2013)"
)
def azimuth_flags(
    obj,
    *,
    tolerance=4.0,
    window=5,
    duplicate_fraction=0.1,
    gap_factor=2.5,
    n_threads=None,
    engine="auto",
):
    """
    Flag jumps in the ray angles of a sweep or volume.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataTree
        A sweep (a Dataset with an ``azimuth`` dimension or coordinate, or an
        ``elevation`` one for an RHI), a single sweep node of a volume, or a
        volume whose ``sweep_*`` nodes are checked one by one.
    tolerance : float, optional
        Deviation, in ray spacings, from the positions that the neighbours
        predict beyond which a ray is an outlier. Default 4.
    window : int, optional
        Number of rays on each side that predict the position of a ray.
        Default 5; at least 2.
    duplicate_fraction : float, optional
        A step to the next good ray of less than this share of a ray spacing
        marks a duplicate. Default 0.1.
    gap_factor : float, optional
        A forward step of more than this many ray spacings marks a gap
        (rays are missing). Default 2.5.
    n_threads : int, optional
        Threads of the compiled kernel (one sweep per thread). Default: all.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation. Default ``"auto"``: the compiled kernel when built.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        On the ray dimension: ``azimuth_flag`` (uint8 bits ``outlier`` 1,
        ``duplicate`` 2, ``gap`` 4, ``reversal`` 8, ``missing`` 16; see
        :data:`AZIMUTH_FLAGS`), ``azimuth_fixed`` (the angle with the bad
        rays repaired, degrees in [0, 360) for the repaired rays) and
        ``azimuth_missing_rays`` (rays missing after a ray with the ``gap``
        flag); ``azimuth_step`` (scalar) is the typical step in degrees. A
        DataTree gives one such node per sweep.

    Notes
    -----
    See the module description for the method and its limits. The check
    needs at least five rays with finite angles; a sweep with fewer is
    returned without flags other than ``missing``.
    {references}
    Examples
    --------
    >>> flags = radarx.retrieve.azimuth_flags(sweep)  # doctest: +SKIP
    >>> flags.azimuth_flag.where(flags.azimuth_flag > 0, drop=True)  # doctest: +SKIP
    """
    names, sets = _sweeps(obj)
    params = {
        "tolerance": float(tolerance),
        "window": int(window),
        "duplicate_fraction": float(duplicate_fraction),
        "gap_factor": float(gap_factor),
    }
    angle_names, _, results = _flag_sweeps(
        sets, tolerance, window, duplicate_fraction, gap_factor, n_threads, engine
    )
    out = [
        _flag_dataset(ds, name, flags, fixed, delta, params)
        for ds, name, (flags, fixed, delta) in zip(sets, angle_names, results)
    ]
    if names[0] is None:
        return out[0]
    return product_tree(as_sweep(obj), dict(zip(names, out)))


def _fill_gaps(ds, name, flags, fixed, delta):
    """The sweep with a ray of missing values inserted for every ray missing in a gap."""
    dim = _ray_dim(ds, name)
    missing = _missing_rays(flags, fixed, delta)
    if not missing.any():
        out = ds.copy()
        out["azimuth_filled"] = xr.DataArray(
            np.zeros(ds.sizes[dim], bool), dims=(dim,), attrs=_FILLED_ATTRS
        )
        return out
    source, angle, filled = [], [], []
    for i in range(fixed.size):
        source.append(i)
        angle.append(fixed[i])
        filled.append(False)
        if missing[i]:
            step = wrap180(fixed[i + 1] - fixed[i]) / (missing[i] + 1)
            for k in range(1, missing[i] + 1):
                source.append(i)
                angle.append(wrap360(fixed[i] + k * step))
                filled.append(True)
    source = np.asarray(source)
    filled = np.asarray(filled)
    out = ds.isel({dim: source})
    mask = xr.DataArray(~filled, dims=dim)
    for var in out.data_vars:
        if dim in out[var].dims and np.issubdtype(out[var].dtype, np.floating):
            out[var] = out[var].where(mask)
    out = out.assign_coords({name: (dim, np.asarray(angle))})
    if (
        "time" in out.coords
        and out["time"].dims == (dim,)
        and np.issubdtype(out["time"].dtype, np.datetime64)
    ):
        t = (
            ds["time"]
            .values.astype("datetime64[ns]")
            .astype("int64")
            .astype(np.float64)
        )
        pos = np.arange(fixed.size, dtype=np.float64)
        frac = pos[source].copy()
        run = np.flatnonzero(filled)
        for (
            r
        ) in run:  # position of an inserted ray between its source ray and the next one
            k = 1
            while r - k >= 0 and filled[r - k]:
                k += 1
            frac[r] = source[r] + k / (missing[source[r]] + 1)
        out = out.assign_coords(
            time=(dim, np.interp(frac, pos, t).astype("int64").astype("datetime64[ns]"))
        )
    out["azimuth_filled"] = xr.DataArray(filled, dims=(dim,), attrs=_FILLED_ATTRS)
    return out


def _regularize(ds, name, flags, fixed, delta, tolerance):
    """The sweep on a regular angle grid; the rays of a gap are filled with NaN."""
    dim = _ray_dim(ds, name)
    if not np.isfinite(delta) or delta == 0:
        return ds
    if np.any(flags & REVERSAL):
        raise ValueError(
            "regularize needs the rays in scan order: the sweep has a step against the scan "
            "direction (flag 'reversal'); sort the rays by time first"
        )
    unwrapped = fixed[0] + np.concatenate([[0.0], np.cumsum(wrap180(np.diff(fixed)))])
    # the grid step is the angle covered per ray when the missing rays are counted in,
    # which keeps an error in the median step from adding up along the sweep
    n_missing = int(_missing_rays(flags, fixed, delta).sum())
    spacing = (unwrapped[-1] - unwrapped[0]) / (fixed.size - 1 + n_missing)
    position = (unwrapped - unwrapped[0]) / spacing  # in ray spacings, increasing
    n_new = int(np.rint(position[-1])) + 1
    grid = np.arange(n_new, dtype=np.float64)
    nearest = np.clip(np.searchsorted(position, grid), 1, position.size - 1)
    nearest = np.where(
        np.abs(position[nearest - 1] - grid) <= np.abs(position[nearest] - grid),
        nearest - 1,
        nearest,
    )
    present = np.abs(position[nearest] - grid) <= tolerance
    out = ds.isel({dim: nearest})
    for var in out.data_vars:
        if dim in out[var].dims and np.issubdtype(out[var].dtype, np.floating):
            out[var] = out[var].where(xr.DataArray(present, dims=dim))
    angle = wrap360(unwrapped[0] + grid * spacing)
    out = out.assign_coords({name: (dim, angle)})
    if (
        "time" in out.coords
        and out["time"].dims == (dim,)
        and np.issubdtype(out["time"].dtype, np.datetime64)
    ):
        t = (
            ds["time"]
            .values.astype("datetime64[ns]")
            .astype("int64")
            .astype(np.float64)
        )
        out = out.assign_coords(
            time=(
                dim,
                np.interp(grid, position, t).astype("int64").astype("datetime64[ns]"),
            )
        )
    out["azimuth_filled"] = xr.DataArray(~present, dims=(dim,), attrs=_FILLED_ATTRS)
    return out


def _fix_sweep(ds, name, flags, fixed, delta, method, regularize_tolerance):
    dim = _ray_dim(ds, name)
    bad = (flags & (OUTLIER | DUPLICATE | MISSING)) > 0
    flag_var = xr.DataArray(flags, dims=(dim,), attrs=_FLAG_ATTRS)
    if method == "drop":
        out = ds.isel({dim: np.flatnonzero(~bad)})
        out["azimuth_flag"] = flag_var.isel({dim: np.flatnonzero(~bad)})
        return out
    if method == "interpolate":
        out = ds.assign_coords({name: (dim, fixed)})
        out["azimuth_raw"] = xr.DataArray(
            ds[name].values,
            dims=(dim,),
            attrs={"long_name": f"ray {name} before the repair", "units": "degrees"},
        )
        out["azimuth_flag"] = flag_var
        return out
    out = ds.assign_coords({name: (dim, fixed)})
    if method == "fill":
        return _fill_gaps(out, name, flags, fixed, delta)
    return _regularize(out, name, flags, fixed, delta, regularize_tolerance)


@dask_support(
    "eager",
    "Rays can be removed or added, so the size of the result depends on the data.",
)
@provenance(
    "Ray-angle repair: interpolation of bad angles, removal or regularization (Hampel 1974; Leys et al. 2013)"
)
def fix_azimuth(
    obj,
    method="interpolate",
    *,
    tolerance=4.0,
    window=5,
    duplicate_fraction=0.1,
    gap_factor=2.5,
    regularize_tolerance=0.5,
    n_threads=None,
    engine="auto",
):
    """
    Repair the ray angles of a sweep or volume.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataTree
        A sweep, a single sweep node, or a volume (see :func:`azimuth_flags`).
    method : {"interpolate", "drop", "fill", "regularize"}, optional
        ``"interpolate"`` (default) replaces the angle of the outliers,
        duplicates and missing angles with the interpolation between the
        nearest good rays and keeps every ray; the angle before the repair is
        kept in ``azimuth_raw``. ``"drop"`` removes those rays. ``"fill"``
        interpolates and then inserts a ray of missing values for every ray
        that is missing in a gap, with an interpolated angle and time, and
        keeps every real ray (``azimuth_filled`` marks the new rays).
        ``"regularize"`` interpolates and then puts the sweep on a regular
        angle grid with the typical step: every grid angle takes the nearest
        ray if it lies within ``regularize_tolerance`` spacings and is
        otherwise filled with missing values. Rays that are closer than a
        spacing (the antenna decelerates at the end of a sector) are merged,
        so the number of real rays can fall. It needs the rays in scan order
        and raises an error for a sweep with a ``reversal``.
    tolerance, window, duplicate_fraction, gap_factor, n_threads, engine
        As in :func:`azimuth_flags`.
    regularize_tolerance : float, optional
        For ``"regularize"``: largest distance, in ray spacings, between a grid
        angle and the ray it takes. Default 0.5.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        The input with the ray angles repaired (``azimuth_flag`` says which
        rays were bad). A DataTree keeps its other nodes.

    Notes
    -----
    Gaps and reversals are not repaired by ``"interpolate"`` and ``"drop"``:
    the true angles of missing rays are not known. See :func:`azimuth_flags`.
    {references}
    Examples
    --------
    >>> fixed = radarx.retrieve.fix_azimuth(volume)  # doctest: +SKIP
    >>> regular = radarx.retrieve.fix_azimuth(sweep, "regularize")  # doctest: +SKIP
    """
    if method not in ("interpolate", "drop", "fill", "regularize"):
        raise ValueError(
            f"method must be 'interpolate', 'drop', 'fill' or 'regularize', not {method!r}"
        )
    if not 0 < regularize_tolerance <= 1:
        raise ValueError("regularize_tolerance must be in (0, 1]")
    source = as_sweep(obj)
    names, sets = _sweeps(source)
    angle_names, _, results = _flag_sweeps(
        sets, tolerance, window, duplicate_fraction, gap_factor, n_threads, engine
    )
    fixed_sets = [
        _fix_sweep(ds, name, flags, fixed, delta, method, regularize_tolerance)
        for ds, name, (flags, fixed, delta) in zip(sets, angle_names, results)
    ]
    if names[0] is None:
        return fixed_sets[0]
    tree = source.copy()
    root_coords = set(source.to_dataset(inherit=False).coords)
    for name, ds in zip(names, fixed_sets):
        tree[name] = ds.drop_vars(
            [c for c in ds.coords if c in root_coords and c not in ds.dims]
        )
    return tree


azimuth_flags.__doc__ = azimuth_flags.__doc__.replace("{references}", _REFERENCES)
fix_azimuth.__doc__ = fix_azimuth.__doc__.replace("{references}", _REFERENCES)

# --------------------------------------------------------------------------
# accessors
# --------------------------------------------------------------------------


@accessor_method("dataset", "datatree", name="azimuth_flags", wraps=azimuth_flags)
def _azimuth_flags_accessor(self):
    """
    Flags for jumps in the ray angles of this sweep or volume.

    See :func:`radarx.retrieve.azimuth_flags` for the parameters.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``azimuth_flag``, ``azimuth_fixed``, ``azimuth_missing_rays`` and ``azimuth_step``.
    """


@accessor_method("dataset", "datatree", name="fix_azimuth", wraps=fix_azimuth)
def _fix_azimuth_accessor(self):
    """
    This sweep or volume with the ray angles repaired.

    See :func:`radarx.retrieve.fix_azimuth` for the parameters.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        The input with repaired ray angles and the flag ``azimuth_flag``.
    """
