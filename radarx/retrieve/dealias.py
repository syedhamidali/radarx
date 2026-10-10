#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Doppler Velocity Dealiasing
===========================

Unfold Doppler radial velocities that exceed the Nyquist velocity.

A pulse Doppler radar measures radial velocity only within the Nyquist
interval :math:`[-V_n, V_n]`; a true velocity :math:`v` is reported as
:math:`v - 2 k V_n` for some integer fold :math:`k`. Dealiasing finds
:math:`k` for every gate.

radarx uses a region-based method that works directly on the polar sweep
grid, where each gate's neighbours are known by index. The method is
radarx's own combination of published ideas, not an implementation of one
paper: the criterion (a least-squares sum of squared velocity jumps between
neighbours, Jing and Wiener 1993 [1]), the reference wind (Eilts and Smith
1990 [2]), the volume continuity (James and Houze 2001 [3]) and the VAD fit
(Browning and Wexler 1968 [4]) are the published concepts; the region graph,
spanning tree, coordinate descent and the gate check are radarx's
implementation, and all thresholds below are radarx choices. The details of the
concepts quoted here are not checked against the papers.

1. **Regions.** Neighbouring gates (along the ray and between adjacent rays)
   whose velocities differ by less than ``threshold * Vn`` are joined with a
   union-find. Inside such a region the field is continuous, so all its gates
   share one fold. Velocity jumps of about :math:`2 V_n` (fold lines) separate
   regions. The default ``threshold=0.3`` is radarx's own choice.
2. **Region graph.** For every pair of touching regions, the number of
   boundary gate pairs and the summed velocity jump across the boundary are
   accumulated. Regions separated by short gaps of empty gates (along or
   across rays) are compared across the gap.
3. **Folds.** Integer folds per region minimise the summed squared velocity
   jump over all region boundaries, weighted by boundary length (the
   least-squares criterion of Jing and Wiener 1993 [1]). A maximum spanning tree
   (unambiguous boundaries first, then by length) gives the start; integer
   coordinate descent then moves single regions and whole blocks of
   consistently joined regions until no move lowers the cost, so groups of
   regions that are offset together are corrected too.
4. **Absolute fold.** The largest group of connected regions is matched to a
   reference velocity where one is available: a wind profile (sounding, VAD
   or model) as in Eilts and Smith (1990) [2], or the already dealiased
   sweep below, the volume continuity of James and Houze (2001) [3].
   Otherwise its mean velocity is brought closest to zero (radarx choice). A
   VAD fit of that group (Browning and Wexler 1968 [4]; the in-sweep fit uses
   20 gates on each side and needs at least 50 gates and an azimuth coverage
   above a minimum, radarx choices) then gives the reference for all other,
   disconnected groups.
   Finally, a gate that differs by more than :math:`V_n` from all of its
   neighbours is moved to the fold closest to their mean.

All decisions use integer arithmetic, so the compiled C++ kernel
(multithreaded, all sweeps in one call) and the NumPy fallback give
identical folds. A modular alternative that is not used here is described by
Louf et al. (2020) [5].

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}

References
----------
.. [1] Jing, Z., and G. Wiener, 1993: Two-dimensional dealiasing of Doppler
   velocities. *J. Atmos. Oceanic Technol.*, **10** (6), 798-808,
   https://doi.org/10.1175/1520-0426(1993)010<0798:TDDODV>2.0.CO;2
.. [2] Eilts, M. D., and S. D. Smith, 1990: Efficient dealiasing of Doppler
   velocities using local environment constraints. *J. Atmos. Oceanic
   Technol.*, **7** (1), 118-128,
   https://doi.org/10.1175/1520-0426(1990)007<0118:EDODVU>2.0.CO;2
.. [3] James, C. N., and R. A. Houze, 2001: A real-time four-dimensional
   Doppler dealiasing scheme. *J. Atmos. Oceanic Technol.*, **18** (10),
   1674-1683, https://doi.org/10.1175/1520-0426(2001)018<1674:ARTFDD>2.0.CO;2
.. [4] Browning, K. A., and R. Wexler, 1968: The determination of kinematic
   properties of a wind field using Doppler radar. *J. Appl. Meteor.*, **7** (1),
   105-113, https://doi.org/10.1175/1520-0450(1968)007<0105:TDOKPO>2.0.CO;2
.. [5] Louf, V., A. Protat, R. C. Jackson, S. M. Collis, and J. Helmus, 2020:
   UNRAVEL: A robust modular velocity dealiasing technique for Doppler radar.
   *J. Atmos. Oceanic Technol.*, **37** (5), 741-758,
   https://doi.org/10.1175/JTECH-D-19-0020.1
"""

from __future__ import annotations

__all__ = ["dealias_velocity"]

__doc__ = __doc__.format("\n   ".join(__all__))

import warnings
from itertools import pairwise

import numpy as np
import xarray as xr

from .._provenance import provenance
from ._products import product_tree

try:
    from . import _dealias

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _dealias = None
    HAS_COMPILED_KERNEL = False

EARTH_RADIUS = 6371000.0
_SCALE = 1 << 20  # fixed point for velocity jumps in units of 2 Vn
# The constants below are radarx's own choices, not values from a paper.
_NYQUIST_TOLERANCE = 1.01  # values beyond this times Vn are flags, not data
_MAX_VOTE = 127  # folds beyond this are not physical
_VEL_Q = 256.0  # velocities in fixed point of 1/256 m/s
_TRIG_Q = 16384.0  # sin/cos of azimuth in fixed point
_VAD_WINDOW = 20  # range gates on each side for the in-sweep VAD fit
_VAD_MIN_GATES = 50
_VAD_MIN_SPREAD = 0.02  # azimuth coverage (determinant; full circle 0.25)
_MIN_NEIGHBOURS = 3
_MAX_RAY_GAP = 10  # empty rays bridged between regions
_GATE_PASSES = 2


@provenance(
    "Region-based velocity dealiasing, radarx's own combination of published concepts"
)
def dealias_velocity(
    radar,
    field="VRADH",
    nyquist_velocity=None,
    *,
    threshold=0.3,
    max_gap=20,
    reference=None,
    wind_profile=None,
    sweep_continuity=True,
    name=None,
    max_iterations=100,
    n_threads=None,
    engine="auto",
    products_only=True,
):
    """
    Dealias (unfold) Doppler radial velocities.

    Parameters
    ----------
    radar : xarray.Dataset or xarray.DataTree
        A PPI sweep on ``(azimuth, range)`` with ``azimuth`` and
        ``elevation`` coordinates, or a volume with ``sweep_*`` groups,
        e.g. from xradar.
    field : str, optional
        Radial velocity field. Default ``"VRADH"``.
    nyquist_velocity : float or dict, optional
        Nyquist velocity in m/s. By default it is read from the sweep's
        ``nyquist_velocity`` variable or coordinate (xradar), or from the
        field's ``nyquist_velocity`` attribute. For a DataTree, a dict maps
        sweep names to values.
    threshold : float, optional
        Neighbouring gates belong to the same region when their velocities
        differ by less than ``threshold`` times the Nyquist velocity.
        Default 0.3.
    max_gap : int, optional
        Regions separated along a ray by up to this many empty gates (e.g.
        range-folded or censored gates) are still compared. Default 20.
        Across rays, gaps of up to 10 rays are bridged.
    reference : xarray.DataArray, optional
        Reference radial velocity on the sweep grid (Dataset input only),
        e.g. from a model. It fixes the absolute fold.
    wind_profile : xarray.Dataset, optional
        Horizontal wind profile with variables ``u`` (eastward) and ``v``
        (northward) in m/s on a ``height`` coordinate (metres above sea
        level), e.g. from a sounding, a VAD or a model. Its radial component
        is used as the reference velocity.
    sweep_continuity : bool, optional
        For a DataTree, process sweeps from the lowest elevation upward and
        use the dealiased sweep below as the reference for the next one
        (gaps are filled from ``wind_profile``). Default True.
    name : str, optional
        Name of the dealiased velocity. Default ``f"{field}_dealiased"``
        (e.g. ``"VRADH_dealiased"``), so that it never replaces the measured
        field when the products are merged into the sweep or volume with
        ``.radarx.assign(products)``.
    max_iterations : int, optional
        Maximum passes of the fold optimisation. Default 100.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled
        kernel and falls back to NumPy; both give identical folds.
    products_only : bool, optional
        ``True`` (default) returns the products only. ``False`` keeps the
        earlier behaviour for one more release and emits a
        ``FutureWarning``: a sweep gives the dealiased DataArray named
        ``field``, and a volume gives a copy of the input tree with the
        dealiased field (by default replacing ``field``) in every sweep.
        Use ``dtree.radarx.assign(dealias_velocity(dtree))`` instead.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        For a sweep, the dealiased velocity ``name`` with the input's
        coordinates and dtype. For a volume, a DataTree with the input's
        root and one node per sweep that has ``field``, holding the
        dealiased velocity ``name`` on the sweep's coordinates. Gates with
        no data, or with values beyond the Nyquist velocity (flag values),
        are NaN.

    Raises
    ------
    ValueError
        If the Nyquist velocity is unknown or not positive.
    ImportError
        If ``engine="compiled"`` and the compiled kernel is not available.

    Notes
    -----
    See the module documentation for the method, which combines the concepts
    of Jing and Wiener (1993) [1], Eilts and Smith (1990) [2], James and
    Houze (2001) [3] and Browning and Wexler (1968) [4] and is not an
    implementation of any one of these papers; the thresholds (``threshold``,
    ``max_gap``, the 10 ray gap, ``max_iterations``) are radarx choices.
    Without any reference, the
    absolute fold of the lowest sweep assumes that its mean radial velocity
    is close to zero, as for a horizontally uniform wind seen all around the
    radar; for echoes that cover only part of the circle, pass a
    ``wind_profile`` (or ``reference``).

    References
    ----------
    .. [1] Jing, Z., and G. Wiener, 1993: Two-dimensional dealiasing of
       Doppler velocities. *J. Atmos. Oceanic Technol.*, **10** (6), 798-808,
       https://doi.org/10.1175/1520-0426(1993)010<0798:TDDODV>2.0.CO;2
    .. [2] Eilts, M. D., and S. D. Smith, 1990: Efficient dealiasing of
       Doppler velocities using local environment constraints. *J. Atmos.
       Oceanic Technol.*, **7** (1), 118-128,
       https://doi.org/10.1175/1520-0426(1990)007<0118:EDODVU>2.0.CO;2
    .. [3] James, C. N., and R. A. Houze, 2001: A real-time four-dimensional
       Doppler dealiasing scheme. *J. Atmos. Oceanic Technol.*, **18** (10),
       1674-1683,
       https://doi.org/10.1175/1520-0426(2001)018<1674:ARTFDD>2.0.CO;2
    .. [4] Browning, K. A., and R. Wexler, 1968: The determination of
       kinematic properties of a wind field using Doppler radar. *J. Appl.
       Meteor.*, **7** (1), 105-113,
       https://doi.org/10.1175/1520-0450(1968)007<0105:TDOKPO>2.0.CO;2

    Examples
    --------
    >>> vr = radarx.retrieve.dealias_velocity(ds, "VRADH", 26.0)  # doctest: +SKIP
    >>> vr.name  # doctest: +SKIP
    'VRADH_dealiased'
    >>> products = dtree.radarx.dealias("VRADH")  # doctest: +SKIP
    >>> dtree = dtree.radarx.assign(products)  # doctest: +SKIP
    """
    if products_only:
        name = name or f"{field}_dealiased"
    else:
        warnings.warn(
            "dealias_velocity(..., products_only=False) returns the input with "
            "the dealiased field added and will be removed in the next "
            "release. Use the default products_only=True and merge the "
            "products with ds.radarx.assign(products) or "
            "dtree.radarx.assign(products).",
            FutureWarning,
            stacklevel=2,
        )
        name = name or field
    compiled = _use_compiled(engine)
    options = {
        "threshold": float(threshold),
        "max_gap": int(max_gap),
        "max_iterations": int(max_iterations),
        "n_threads": int(n_threads or 0),
        "compiled": compiled,
    }
    if isinstance(radar, xr.Dataset):
        sweep = _Sweep(radar, field, _nyquist(radar, field, nyquist_velocity))
        ref = None
        if reference is not None:
            ref = reference.broadcast_like(sweep.da).transpose(sweep.ray, "range")
            ref = ref.values.astype(np.float64)
        if wind_profile is not None:
            ref = _fill(ref, _wind_reference(radar, sweep, wind_profile))
        (solved,) = _region_folds([sweep], **options)
        (folds,) = _absolute_folds([sweep], [solved], [ref], **options)
        return sweep.wrap(folds).rename(name)
    if reference is not None:
        raise ValueError("reference is only supported for a single sweep.")
    return _dealias_tree(
        radar,
        field,
        nyquist_velocity,
        wind_profile=wind_profile,
        sweep_continuity=sweep_continuity,
        name=name,
        products_only=products_only,
        **options,
    )


# --------------------------------------------------------------------------
# xarray layer
# --------------------------------------------------------------------------


def _use_compiled(engine):
    """Whether to run the compiled kernel for the requested ``engine``."""
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled dealiasing kernel is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


def _ray_dim(da):
    dims = [d for d in da.dims if d != "range"]
    if da.ndim != 2 or len(dims) != 1 or "range" not in da.dims:
        raise ValueError(f"{da.name!r} must be 2-D on (azimuth, range).")
    return dims[0]


def _nyquist(ds, field, override, sweep=None):
    """Nyquist velocity of a sweep: user value, xradar metadata or attribute."""
    value = override
    if isinstance(value, dict):
        value = value.get(sweep)
    if value is None and "nyquist_velocity" in ds.variables:
        value = np.nanmedian(np.asarray(ds["nyquist_velocity"].values, dtype=float))
    if value is None:
        value = ds[field].attrs.get("nyquist_velocity")
    if value is None:
        where = f" in {sweep!r}" if sweep else ""
        raise ValueError(
            f"Unknown Nyquist velocity{where}: pass nyquist_velocity=... or add a "
            "'nyquist_velocity' coordinate to the sweep."
        )
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Nyquist velocity must be positive, got {value}.")
    return value


def _ray_links(azimuth, max_gap=2.0):
    """Whether ray ``i`` touches ray ``i + 1`` (the last one: the first one)."""
    az = np.asarray(azimuth, dtype=np.float64)
    if az.size < 2:
        return np.zeros(az.size, dtype=np.uint8)
    step = np.abs((np.roll(az, -1) - az + 180.0) % 360.0 - 180.0)
    spacing = np.median(step[:-1]) if az.size > 2 else step[0]
    return (step <= max_gap * spacing).astype(np.uint8)


class _Sweep:
    """A sweep's velocity as contiguous (ray, range) float64 plus metadata."""

    def __init__(self, ds, field, nyquist):
        self.da = ds[field]
        self.ray = _ray_dim(self.da)
        self.nyquist = nyquist
        vel = np.array(self.da.transpose(self.ray, "range").values, dtype=np.float64)
        # values beyond the Nyquist velocity are flags (e.g. range folded)
        vel[~(np.abs(vel) <= _NYQUIST_TOLERANCE * nyquist)] = np.nan
        self.velocity = np.ascontiguousarray(vel)
        self.links = _ray_links(ds["azimuth"].values)
        az = np.deg2rad(np.asarray(ds["azimuth"].values, dtype=np.float64))
        self.ray_sin = np.ascontiguousarray(np.sin(az))
        self.ray_cos = np.ascontiguousarray(np.cos(az))

    def wrap(self, folds):
        """Dealiased DataArray with the input's coordinates, dims and dtype."""
        da = self.da
        out = self.velocity + 2.0 * self.nyquist * folds
        if da.dims[0] != self.ray:
            out = out.T
        dtype = da.dtype if np.issubdtype(da.dtype, np.floating) else np.float64
        attrs = dict(da.attrs)
        attrs.update(
            {
                "long_name": "Dealiased "
                + da.attrs.get("long_name", "radial velocity"),
                "nyquist_velocity": self.nyquist,
                "comment": "Nyquist folds unfolded with radarx.retrieve."
                "dealias_velocity (region-based, Jing and Wiener 1993)",
            }
        )
        attrs.setdefault("units", "m s-1")
        attrs.setdefault(
            "standard_name", "radial_velocity_of_scatterers_away_from_instrument"
        )
        return da.copy(data=out.astype(dtype, copy=False)).assign_attrs(attrs)


def _beam_height(ds, sweep):
    """Gate heights above sea level (4/3 Earth) on (ray, range)."""
    rng = ds["range"].values.astype(np.float64)[None, :]
    elev = np.broadcast_to(ds["elevation"].values, ds[sweep.ray].shape)
    sin_el = np.sin(np.deg2rad(elev.astype(np.float64)))[:, None]
    alt = float(ds["altitude"]) if "altitude" in ds.variables else 0.0
    r_eff = EARTH_RADIUS * 4.0 / 3.0
    return np.sqrt(rng**2 + r_eff**2 + 2.0 * rng * r_eff * sin_el) - r_eff + alt


def _wind_reference(ds, sweep, wind_profile):
    """Radial component of a ``u``/``v`` wind profile at every gate."""
    height = wind_profile["height"].values.astype(np.float64)
    order = np.argsort(height)
    z = _beam_height(ds, sweep)
    u = np.interp(z, height[order], wind_profile["u"].values[order], np.nan, np.nan)
    v = np.interp(z, height[order], wind_profile["v"].values[order], np.nan, np.nan)
    az = np.deg2rad(ds["azimuth"].values.astype(np.float64))[:, None]
    elev = np.broadcast_to(ds["elevation"].values, ds[sweep.ray].shape)
    cos_el = np.cos(np.deg2rad(elev.astype(np.float64)))[:, None]
    return (u * np.sin(az) + v * np.cos(az)) * cos_el


def _fill(ref, other):
    if ref is None:
        return other
    if other is None:
        return ref
    return np.where(np.isnan(ref), other, ref)


def _nearest_index(source, target, tolerance, period=None):
    """Index of the nearest ``source`` value for each ``target`` (-1 if too far)."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if period is not None:
        source = np.mod(source, period)
        target = np.mod(target, period)
    order = np.argsort(source, kind="stable")
    s = source[order]
    pos = np.searchsorted(s, target)
    lo = np.clip(pos - 1, 0, s.size - 1)
    hi = np.clip(pos, 0, s.size - 1)
    dist_lo = np.abs(target - s[lo])
    dist_hi = np.abs(target - s[hi])
    best = np.where(dist_hi < dist_lo, hi, lo)
    dist = np.minimum(dist_lo, dist_hi)
    if period is not None:  # across 0/360
        wrap_first = np.abs(target - s[0] - period)
        wrap_last = np.abs(target + period - s[-1])
        best = np.where(wrap_first < dist, 0, best)
        dist = np.minimum(dist, wrap_first)
        best = np.where(wrap_last < dist, s.size - 1, best)
        dist = np.minimum(dist, wrap_last)
    return np.where(dist <= tolerance, order[best], -1)


def _sweep_mapping(source, target):
    """Nearest ray and gate of sweep ``source`` for each ray and gate of ``target``."""
    az = np.asarray(target["azimuth"].values, dtype=np.float64)
    step = np.abs((np.diff(az) + 180.0) % 360.0 - 180.0)
    az_step = np.median(step) if step.size else 1.0
    rng = np.asarray(target["range"].values, dtype=np.float64)
    rng_step = np.median(np.diff(rng)) if rng.size > 1 else 1.0
    iray = _nearest_index(source["azimuth"].values, az, 1.5 * az_step, 360.0)
    igate = _nearest_index(source["range"].values, rng, 1.5 * rng_step)
    return iray.astype(np.int32), igate.astype(np.int32)


def _region_folds(sweeps, threshold, max_gap, max_iterations, n_threads, compiled):
    """Steps 1-3 for every sweep: (relative folds, component ids)."""
    if compiled:
        return _dealias.region_folds(
            [s.velocity for s in sweeps],
            [s.links for s in sweeps],
            [s.nyquist for s in sweeps],
            threshold=threshold,
            max_gap=max_gap,
            max_ray_gap=_MAX_RAY_GAP,
            max_iterations=max_iterations,
            n_threads=n_threads,
        )
    return [
        _region_folds_numpy(
            s.velocity, s.links, s.nyquist, threshold, max_gap, max_iterations
        )
        for s in sweeps
    ]


def _absolute_folds(
    sweeps, solved, references, n_threads, compiled, previous=None, mapping=None, **_
):
    """Step 4 for every sweep: final integer folds (optionally chained)."""
    if compiled:
        none = np.zeros(0, dtype=np.int32)
        mapping = [(none, none) if m is None else m for m in mapping or []]
        return _dealias.absolute_folds(
            [s.velocity for s in sweeps],
            [k for k, _ in solved],
            [c for _, c in solved],
            [s.nyquist for s in sweeps],
            references,
            [s.ray_sin for s in sweeps],
            [s.ray_cos for s in sweeps],
            [s.links for s in sweeps],
            vad_window=_VAD_WINDOW,
            gate_passes=_GATE_PASSES,
            previous=previous or [],
            prev_ray=[m[0] for m in mapping] if mapping else [],
            prev_gate=[m[1] for m in mapping] if mapping else [],
            n_threads=n_threads,
        )
    out = []
    for i, (s, (k, c), ref) in enumerate(zip(sweeps, solved, references)):
        if previous and previous[i] >= 0:
            ref = _fill(
                _chained_reference(sweeps[previous[i]], out[previous[i]], *mapping[i]),
                ref,
            )
        out.append(
            _absolute_folds_numpy(
                s.velocity, k, c, s.nyquist, ref, s.ray_sin, s.ray_cos, s.links
            )
        )
    return out


def _chained_reference(source, folds, iray, igate):
    """Dealiased ``source`` sweep at the gates given by the nearest-index mapping."""
    unfolded = source.velocity + 2.0 * source.nyquist * folds
    ref = unfolded[np.clip(iray, 0, None)][:, np.clip(igate, 0, None)]
    ref[iray < 0] = np.nan
    ref[:, igate < 0] = np.nan
    return ref


def _dealias_tree(
    dtree,
    field,
    nyquist_velocity,
    wind_profile,
    sweep_continuity,
    name,
    products_only,
    **options,
):
    """Dealias every sweep of a volume that contains ``field``."""
    from ..grid.cone import _sweep_dataset, _sweep_names

    names = [n for n in _sweep_names(dtree) if field in dtree[n].data_vars]
    if not names:
        raise ValueError(f"No sweep contains {field!r}.")
    datasets = {n: _sweep_dataset(dtree, n) for n in names}
    elevation = {n: float(np.nanmedian(datasets[n]["elevation"].values)) for n in names}
    names.sort(key=lambda n: elevation[n])  # stable: split cuts keep their order
    sweeps = [
        _Sweep(datasets[n], field, _nyquist(datasets[n], field, nyquist_velocity, n))
        for n in names
    ]
    # the expensive part, all sweeps in parallel
    solved = _region_folds(sweeps, **options)
    winds = [
        None if wind_profile is None else _wind_reference(datasets[n], s, wind_profile)
        for n, s in zip(names, sweeps)
    ]
    previous = mapping = None
    if sweep_continuity and len(names) > 1:
        # each sweep's absolute fold follows the dealiased sweep below it
        previous = list(range(-1, len(names) - 1))
        mapping = [None] + [
            _sweep_mapping(datasets[a], datasets[b]) for a, b in pairwise(names)
        ]
    folds = _absolute_folds(
        sweeps, solved, winds, previous=previous, mapping=mapping, **options
    )
    results = [s.wrap(f) for s, f in zip(sweeps, folds)]
    if products_only:
        return product_tree(
            dtree, {n: r.rename(name).to_dataset() for n, r in zip(names, results)}
        )
    out = dtree.copy()
    for n, result in zip(names, results):
        out[f"{n}/{name}"] = result.variable
    return out


# --------------------------------------------------------------------------
# NumPy reference implementation (identical folds to the C++ kernel)
# --------------------------------------------------------------------------


def _round_div(num, den):
    """round(num / den) with halves rounded up (integers, den > 0)."""
    return (2 * num + den) // (2 * den)


def _gate_links(nray, ngate, links):
    """Index pairs of neighbouring gates: along rays and between rays."""
    gate = np.arange(nray * ngate).reshape(nray, ngate)
    rays = np.flatnonzero(links)
    rays = rays[(rays + 1) % nray != rays]
    a = np.concatenate([gate[:, :-1].ravel(), gate[rays].ravel()])
    b = np.concatenate([gate[:, 1:].ravel(), gate[(rays + 1) % nray].ravel()])
    return a, b


def _gap_links(valid, ngate, max_gap):
    """Pairs of valid gates on one ray separated by 1 to ``max_gap`` empty gates."""
    g = np.flatnonzero(valid)
    step = np.diff(g)
    bridge = (step > 1) & (step <= max_gap + 1) & (g[:-1] // ngate == g[1:] // ngate)
    return g[:-1][bridge], g[1:][bridge]


def _ray_gap_links(valid, links, max_ray_gap):
    """Pairs of valid gates at one range separated by 1 to ``max_ray_gap`` empty rays."""
    nray, ngate = valid.shape
    gate = np.arange(nray * ngate).reshape(nray, ngate)
    a, b = [], []
    if nray <= 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    rays = np.arange(nray)
    link = links.astype(bool)
    open_ = valid & link[:, None]  # chain from ray r reaches ray r + d - 1 ... r + d
    for d in range(2, max_ray_gap + 2):
        if d >= nray:
            break
        mid = (rays + d - 1) % nray
        open_ = open_ & ~valid[mid] & link[mid][:, None]
        end = (rays + d) % nray
        hit = open_ & valid[end]
        r, j = np.nonzero(hit)
        a.append(gate[r, j])
        b.append(gate[end[r], j])
    if not a:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(a), np.concatenate(b)


def _region_folds_numpy(vel, links, nyq, threshold, max_gap, max_iterations):
    """Steps 1-3 (NumPy/SciPy): relative fold and component of every gate."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    nray, ngate = vel.shape
    n = nray * ngate
    v = vel.ravel()
    valid = ~np.isnan(v)
    ga, gb = _gate_links(nray, ngate, links)
    both = valid[ga] & valid[gb]
    ga, gb = ga[both], gb[both]

    # 1. regions, numbered by their first gate
    join = np.abs(v[ga] - v[gb]) < threshold * nyq
    graph = coo_matrix((np.ones(join.sum()), (ga[join], gb[join])), shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    first = np.full(comp.max() + 1, n, dtype=np.int64)
    np.minimum.at(first, comp[valid], np.flatnonzero(valid))
    rank = np.argsort(np.argsort(first, kind="stable"), kind="stable")
    label = np.where(valid, rank[comp], -1)
    nreg = int(label.max()) + 1
    size = np.bincount(label[valid], minlength=nreg)

    # 2. region adjacency: boundary length and summed jump (fixed point),
    # also across short gaps along the ray
    gap_a, gap_b = _gap_links(valid, ngate, max_gap)
    ray_a, ray_b = _ray_gap_links(valid.reshape(nray, ngate), links, _MAX_RAY_GAP)
    ga = np.concatenate([ga, gap_a, ray_a])
    gb = np.concatenate([gb, gap_b, ray_b])
    la, lb = label[ga], label[gb]
    cross = la != lb
    ga, gb, la, lb = ga[cross], gb[cross], la[cross], lb[cross]
    d = np.where(la < lb, v[gb] - v[ga], v[ga] - v[gb])
    q = np.floor(d / (2.0 * nyq) * float(_SCALE) + 0.5).astype(np.int64)
    lo = np.minimum(la, lb).astype(np.int64)
    hi = np.maximum(la, lb).astype(np.int64)
    keys, inverse = np.unique(lo * (nreg + 1) + hi, return_inverse=True)
    edge_n = np.bincount(inverse, minlength=keys.size).astype(np.int64)
    edge_s = np.zeros(keys.size, dtype=np.int64)
    np.add.at(edge_s, inverse, q)
    edge_a = keys // (nreg + 1)
    edge_b = keys % (nreg + 1)

    # 3a. maximum spanning tree on boundary length, with fold offsets
    parent = np.arange(nreg)
    pot = np.zeros(nreg, dtype=np.int64)  # fold relative to parent

    def find(x):
        path = []
        while parent[x] != x:
            path.append(x)
            x = parent[x]
        acc = 0
        for node in reversed(path):  # nearest the root first
            acc += pot[node]
            pot[node] = acc
            parent[node] = x
        return x

    # confident boundaries first (mean jump within 0.3 of a whole number of
    # folds), then by length; ambiguous ones only join otherwise separate parts
    m_all = _round_div(edge_s, edge_n * _SCALE)
    ambiguous = 10 * np.abs(edge_s - m_all * edge_n * _SCALE) > 3 * edge_n * _SCALE
    for e in np.lexsort((-edge_n, ambiguous)):
        a, b = int(edge_a[e]), int(edge_b[e])
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        m = _round_div(int(edge_s[e]), int(edge_n[e]) * _SCALE)  # k_a - k_b
        pa = 0 if a == ra else int(pot[a])
        pb = 0 if b == rb else int(pot[b])
        delta = m - pa + pb  # k_ra - k_rb
        if ra < rb:
            parent[rb], pot[rb] = ra, -delta
        else:
            parent[ra], pot[ra] = rb, delta
    comp = np.array([find(r) for r in range(nreg)], dtype=np.int64)
    k = np.where(comp == np.arange(nreg), 0, pot).astype(np.int64)

    # 3b. integer least squares: coordinate descent on regions and blocks
    _icm(nreg, edge_a, edge_b, edge_n, edge_s, size, k, max_iterations)
    for _ in range(max_iterations):
        # joined by a boundary that agrees with the folds within 1/4 fold
        dk = k[edge_a] - k[edge_b]
        consistent = 4 * np.abs(edge_s - dk * edge_n * _SCALE) <= edge_n * _SCALE
        sid, nsup = _first_index_labels(nreg, edge_a[consistent], edge_b[consistent])
        if nsup == nreg:
            break
        ssize = np.bincount(sid, weights=size, minlength=nsup).astype(np.int64)
        A, B = sid[edge_a], sid[edge_b]
        cross = A != B
        t = edge_n * (k[edge_b] - k[edge_a]) * _SCALE + edge_s  # block jump A -> B
        A, B, t, en = A[cross], B[cross], t[cross], edge_n[cross]
        lo, hi = np.minimum(A, B), np.maximum(A, B)
        t = np.where(A < B, t, -t)
        keys, inverse = np.unique(lo * (nsup + 1) + hi, return_inverse=True)
        sn = np.zeros(keys.size, dtype=np.int64)
        ss = np.zeros(keys.size, dtype=np.int64)
        np.add.at(sn, inverse, en)
        np.add.at(ss, inverse, t)
        delta = np.zeros(nsup, dtype=np.int64)
        _icm(
            nsup,
            keys // (nsup + 1),
            keys % (nsup + 1),
            sn,
            ss,
            ssize,
            delta,
            max_iterations,
        )
        if not delta.any():
            break
        k += delta[sid]
        _icm(nreg, edge_a, edge_b, edge_n, edge_s, size, k, max_iterations)

    folds = np.zeros(n, dtype=np.int32)
    comps = np.full(n, -1, dtype=np.int32)
    folds[valid] = k[label[valid]]
    comps[valid] = comp[label[valid]]
    return folds.reshape(nray, ngate), comps.reshape(nray, ngate)


def _first_index_labels(n, a, b):
    """Connected components of ``n`` nodes, numbered by their smallest node."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    graph = coo_matrix((np.ones(a.size), (a, b)), shape=(n, n))
    ncomp, comp = connected_components(graph, directed=False)
    first = np.full(ncomp, n, dtype=np.int64)
    np.minimum.at(first, comp, np.arange(n))
    rank = np.argsort(np.argsort(first, kind="stable"), kind="stable")
    return rank[comp], ncomp


def _icm(nnode, edge_a, edge_b, edge_n, edge_s, size, k, max_iterations):
    """Integer Gauss-Seidel descent on the boundary least squares (in place)."""
    src = np.concatenate([edge_a, edge_b])
    dst = np.concatenate([edge_b, edge_a])
    nbn = np.concatenate([edge_n, edge_n])
    order = np.argsort(src, kind="stable")
    src, dst, nbn = src[order], dst[order], nbn[order]
    start = np.searchsorted(src, np.arange(nnode + 1))
    den = np.zeros(nnode, dtype=np.int64)
    sum_s = np.zeros(nnode, dtype=np.int64)
    np.add.at(den, edge_a, edge_n)
    np.add.at(den, edge_b, edge_n)
    np.add.at(sum_s, edge_a, edge_s)
    np.subtract.at(sum_s, edge_b, edge_s)
    nodes = [r for r in np.argsort(-size, kind="stable") if start[r] < start[r + 1]]
    for _ in range(max_iterations):
        changed = 0
        for r in nodes:
            s0, s1 = start[r], start[r + 1]
            nk = int(np.dot(nbn[s0:s1], k[dst[s0:s1]]))
            kn = _round_div(nk * _SCALE + int(sum_s[r]), int(den[r]) * _SCALE)
            if kn != k[r]:
                k[r] = kn
                changed += 1
        if not changed:
            break


def _quant(x, q):
    return np.floor(x * q + 0.5).astype(np.int64)


def _mode_votes(gc, votes, ncomp):
    """Most common vote per component: (components with votes, their vote)."""
    if not gc.size:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    pairs, counts = np.unique(np.stack([gc, votes], axis=1), axis=0, return_counts=True)
    # most gates, then smaller |vote|, then the negative one
    best = np.lexsort((pairs[:, 1], np.abs(pairs[:, 1]), -counts, pairs[:, 0]))
    pairs = pairs[best]
    head = np.r_[True, pairs[1:, 0] != pairs[:-1, 0]]
    return pairs[head, 0], pairs[head, 1]


def _votes(r, v, k, nyq):
    t = (r - v) / (2.0 * nyq)
    return np.clip(np.floor((t - k) + 0.5), -_MAX_VOTE, _MAX_VOTE).astype(np.int64)


def _zero_mean(gc, gv, gk, nyq, ncomp, shift, want):
    sel = want[gc]
    q = _quant(gv[sel] / (2.0 * nyq), float(_SCALE)) + gk[sel] * _SCALE
    total = np.zeros(ncomp, dtype=np.int64)
    np.add.at(total, gc[sel], q)
    count = np.bincount(gc[sel], minlength=ncomp).astype(np.int64)
    for c in np.flatnonzero(count):
        shift[c] = _round_div(-int(total[c]), int(count[c]) * _SCALE)


def _cross(a, b, c, d):
    return a * b - c * d


def _vad_fit(vel, folds, comp, anchor, rsin, rcos, nyq, window):
    """Per range gate VAD coefficients (a0, a1, a2) of the anchor, or None."""
    ngate = vel.shape[1]
    sel = comp == anchor
    s = np.broadcast_to(_quant(rsin, _TRIG_Q)[:, None], vel.shape)
    c = np.broadcast_to(_quant(rcos, _TRIG_Q)[:, None], vel.shape)
    u = _quant(np.where(sel, vel, 0.0), _VEL_Q) + folds.astype(np.int64) * _quant(
        2.0 * nyq, _VEL_Q
    )
    terms = [np.ones_like(u), s, c, s * s, s * c, c * c, u, u * s, u * c]
    acc = np.zeros((9, ngate + 1), dtype=np.int64)
    for q, term in enumerate(terms):
        acc[q, 1:] = np.cumsum(np.where(sel, term, 0).sum(axis=0))
    j = np.arange(ngate)
    lo = np.maximum(0, j - window)
    hi = np.minimum(ngate, j + window + 1)
    m = (acc[:, hi] - acc[:, lo]).astype(np.float64)
    T, V = _TRIG_Q, _VEL_Q
    n = m[0]
    s, c = m[1] / T, m[2] / T
    ss, sc, cc = m[3] / (T * T), m[4] / (T * T), m[5] / (T * T)
    bv, bs, bc = m[6] / V, m[7] / (V * T), m[8] / (V * T)
    with np.errstate(divide="ignore", invalid="ignore"):
        k1 = _cross(ss, cc, sc, sc)
        k2 = _cross(s, cc, sc, c)
        k3 = _cross(s, sc, ss, c)
        det = _cross(n, k1, s, k2) + c * k3
        good = (n >= _VAD_MIN_GATES) & (det / (n * n * n) >= _VAD_MIN_SPREAD)
        e1 = _cross(bs, cc, sc, bc)
        e2 = _cross(bs, sc, ss, bc)
        det0 = _cross(bv, k1, s, e1) + c * e2
        e3 = _cross(s, bc, bs, c)
        det1 = _cross(n, e1, bv, k2) + c * e3
        e4 = _cross(ss, bc, bs, sc)
        det2 = _cross(n, e4, s, e3) + bv * k3
        coef = np.stack([det0 / det, det1 / det, det2 / det], axis=1)
    fitted = np.flatnonzero(good)
    if not fitted.size:
        return None
    # unfitted gates: nearest fitted gate (ties: the one nearer the radar)
    p = np.searchsorted(fitted, j)
    after = fitted[np.minimum(p, fitted.size - 1)]
    before = fitted[np.maximum(p - 1, 0)]
    src = np.where(
        p == fitted.size,
        fitted[-1],
        np.where(
            (p == 0) | (after == j),
            after,
            np.where(after - j < j - before, after, before),
        ),
    )
    return coef[src]


def _gate_check(vel, folds, comp, links, nyq, passes):
    """Refold gates that differ by more than Vn from all (>= 3) neighbours."""
    nray, ngate = vel.shape
    valid = comp >= 0
    twovn_q = int(_quant(2.0 * nyq, _VEL_Q))
    vn_q = int(_quant(nyq, _VEL_Q))
    vq = np.where(valid, _quant(np.where(valid, vel, 0.0), _VEL_Q), 0)
    rays = np.arange(nray)
    prev_ray = (rays - 1) % nray
    next_ray = (rays + 1) % nray
    use_prev = (links[prev_ray] > 0) & (prev_ray != rays)
    use_next = (
        (links > 0)
        & (next_ray != rays)
        & (next_ray != np.where(use_prev, prev_ray, -1))
    )
    f = folds.astype(np.int64)
    for _ in range(passes):
        u = vq + f * twovn_q
        cnt = np.zeros(vel.shape, dtype=np.int64)
        far = np.zeros(vel.shape, dtype=np.int64)
        total = np.zeros(vel.shape, dtype=np.int64)
        for source, use in ((rays, None), (prev_ray, use_prev), (next_ray, use_next)):
            for dj in (-1, 0, 1):
                if use is None and dj == 0:
                    continue
                w = np.zeros(vel.shape, dtype=np.int64)
                ok = np.zeros(vel.shape, dtype=bool)
                lo, hi = max(0, -dj), min(ngate, ngate - dj)
                w[:, lo:hi] = u[source][:, lo + dj : hi + dj]
                ok[:, lo:hi] = valid[source][:, lo + dj : hi + dj]
                if use is not None:
                    ok &= use[:, None]
                cnt += ok
                total += np.where(ok, w, 0)
                far += ok & (np.abs(w - u) > vn_q)
        change = valid & (cnt >= _MIN_NEIGHBOURS) & (far == cnt)
        num = total - cnt * u
        den = np.maximum(cnt, 1) * twovn_q
        d = np.where(change, (2 * num + den) // (2 * den), 0)
        if not d.any():
            break
        f = f + d
    return f.astype(np.int32)


def _absolute_folds_numpy(vel, k, comp, nyq, ref, rsin, rcos, links):
    """Step 4 (NumPy): anchor, in-sweep VAD reference and gate check."""
    folds = np.zeros(vel.shape, dtype=np.int32)
    valid = comp >= 0
    if not valid.any():
        return folds
    ncomp = int(comp.max()) + 1
    gc = comp[valid].astype(np.int64)
    gk = k[valid].astype(np.int64)
    gv = vel[valid]
    csize = np.bincount(gc, minlength=ncomp)
    anchor = int(np.argmax(csize))
    shift = np.zeros(ncomp, dtype=np.int64)
    ext = None if ref is None else np.asarray(ref, dtype=np.float64)[valid]

    # 1. anchor: reference votes, else zero mean
    voted = np.zeros(ncomp, dtype=bool)
    sel = gc == anchor
    if ext is not None:
        sel &= ~np.isnan(ext)
    if ext is not None and sel.any():
        comps, best = _mode_votes(
            gc[sel], _votes(ext[sel], gv[sel], gk[sel], nyq), ncomp
        )
        shift[comps] = best
        voted[comps] = True
    else:
        want = np.zeros(ncomp, dtype=bool)
        want[anchor] = True
        _zero_mean(gc, gv, gk, nyq, ncomp, shift, want)
    first = np.zeros(vel.shape, dtype=np.int32)
    first[comp == anchor] = k[comp == anchor] + shift[anchor]

    # 2. other components: reference, else the anchor's VAD, else zero mean
    coef = _vad_fit(vel, first, comp, anchor, rsin, rcos, nyq, _VAD_WINDOW)
    refval = (
        np.full(vel.shape, np.nan) if ref is None else np.asarray(ref, dtype=np.float64)
    )
    if coef is not None:
        vad = (coef[None, :, 0] + coef[None, :, 1] * rsin[:, None]) + coef[
            None, :, 2
        ] * rcos[:, None]
        refval = np.where(np.isnan(refval), vad, refval)
    r = refval[valid]
    sel = (gc != anchor) & ~np.isnan(r)
    comps, best = _mode_votes(gc[sel], _votes(r[sel], gv[sel], gk[sel], nyq), ncomp)
    shift[comps] = best
    voted[comps] = True
    want = ~voted
    want[anchor] = False
    _zero_mean(gc, gv, gk, nyq, ncomp, shift, want)
    folds[valid] = gk + shift[gc]

    # 3. gate check
    return _gate_check(vel, folds, comp, links, nyq, _GATE_PASSES)
