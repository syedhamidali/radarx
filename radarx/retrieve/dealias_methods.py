#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Dealiasing Alternatives and the Fold Margin
===========================================

A second dealiasing method, R2D2, and a flag for the gates whose fold the data
do not settle. Both work on the polar sweep grid of :mod:`radarx.retrieve.dealias`,
whose region method and UNRAVEL they complement.

**R2D2.** :func:`dealias_r2d2` follows the Region-based Recursive Doppler
Dealiasing of Feldmann et al. (2020) [1]_, which was designed for the cases where
the shear from gate to gate approaches the Nyquist velocity (mesocyclones,
convergence lines, a low Nyquist velocity). It keeps the single-gate decisions
away from the places where they go wrong:

1. Shear zones. The largest difference of every gate to its eight neighbours is
   computed; gates where it reaches ``shear_threshold`` times the Nyquist velocity
   (0.8 in the paper's examples), and the 5 x 5 gates around them, form the
   buffer zones. Folds and strong shear both end up in them.
2. Regions. The connected gates outside the buffer zones are regions, taken to
   lie inside one Nyquist interval.
3. Guidance. Each region is compared with the dealiased sweep above, the dealiased
   sweep of the previous volume and a first guess (a model, a sounding or a
   VAD). The fold of the region is the one whose mean difference to the guidance
   is smallest. If the guidance fields disagree, or the spread of the
   differences exceeds ``spread_threshold`` times the Nyquist velocity (a fold
   hidden inside the region), the region joins the buffer zones.
4. Buffer zones. They are resolved by 2-D continuity as in James and Houze
   (2001) [2]_: a gate next to a resolved gate takes the fold that brings it
   within a threshold of that gate, in passes along and across the rays in both
   directions with thresholds that relax from 0.3 to 0.9 Nyquist velocities, so
   that the weakest shear is settled first and errors stay in the buffer zone.
   The gates that remain follow the median of the resolved gates in a window.
5. Recursion. The steps repeat on the dealiased field. With fewer folds the
   regions grow and the buffer zones shrink. The mean of the largest difference
   decreases until it reaches a minimum; the iteration with the lowest mean is
   the result.

A volume is processed from the highest sweep downward, each sweep guided by the
one above it.

What is the paper's and what is radarx's:

- Taken from the paper: the steps above, the size 5 x 5 of the buffer, the
  window of 10 x 10 gates (11 x 11 here, to centre it), the rule that the first
  guess decides only if the sweeps above and before hold data at less than 10 %
  of the gates in the first iteration and counts in every later one, and the
  stopping rule on the mean of the largest difference.
- radarx's choices, because the paper does not give them: the thresholds of the
  scanning passes (0.3, 0.5, 0.7, 0.9 Nyquist velocities, four directions each),
  the spread threshold (0.5 Nyquist velocities), the least overlap with a guidance
  field (5 gates) and with resolved gates in the window (3), the number of
  iterations (at most 10, stop after 2 without improvement) and the integer
  fixed point (1/256 m/s) in which every decision is made, so that the compiled
  kernel and the NumPy reference give the same folds.
- Not implemented: the filter of the first step (a clutter algorithm of
  MeteoSwiss; use :func:`radarx.retrieve.echo_mask`), and the first guess from
  the COSMO-1 model and the Swiss VAD. Give the first guess as ``reference`` or
  ``wind_profile``. Without one, the first guess is the region method of
  :func:`radarx.retrieve.dealias_velocity` and decides only the regions that the
  sweep above and the previous volume do not cover.

The paper reports, on forward-operator radar data from a numerical model, 75 %
fewer wrong gates than the operational 4DD scheme (James and Houze 2001) with the
model first guess; those numbers depend on the first guess and the data, and the
notebook shows what R2D2 does here on real volumes next to the other methods.

**Fold margin and spread.** :func:`dealias_ambiguity` says where the fold of the
dealiased velocity is not settled. The margin is the evidence in the field. The
velocity is cut into regions of gates that differ by less than a threshold (0.3
Nyquist velocities, as in the region method); a region has one fold. Shifting a
region by one fold, :math:`2 V_n`, changes the squared velocity differences across
its boundary, the criterion of Jing and Wiener (1993) [3]_. With :math:`d_j =
(v_j - v_i) / 2 V_n` for the :math:`n` boundary pairs of the region, the cost of a
shift by :math:`\\pm 1` fold is :math:`n \\mp 2 \\sum_j d_j`, and the margin is the
smaller of the two, per pair:

.. math::

    m = 1 - \\frac{2}{n} \\left| \\sum_j d_j \\right|

:math:`m = 1` where the neighbouring regions are at the same velocity (the fold is
clear), :math:`m = 0` where the boundary velocities differ by half a Nyquist
interval on average (the next fold fits as well), and :math:`m < 0` where it fits
better. This is a radarx definition on the criterion of the paper, which gives no
such measure. Flags mark a weak margin, a short boundary (the fold rests on few
gate pairs) and a region without boundary (its fold is set by the reference
alone, as for an echo that is disconnected from the rest).

The margin cannot see an error that the field does not show: a block that is one
fold off but joined to the rest through a smooth neck is one region, and the
region method puts such a block one fold off in the lowest sweep of the KBMX volume
of 27 April 2011. That kind of ambiguity shows as disagreement between
dealiasings, so ``compare`` takes other results (UNRAVEL, R2D2, another tilt) and
flags the gates where they choose another fold, with the number of folds as the
spread. The notebook shows both on that volume.

The dual-PRF correction of Joe and May (2003) is not implemented; see
:func:`radarx.retrieve.correct_dual_prf` for that of Holleman and Beekhuis (2003).

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}

References
----------
.. [1] Feldmann, M., C. N. James, M. Boscacci, D. Leuenberger, M. Gabella, U.
   Germann, D. Wolfensberger, and A. Berne, 2020: R2D2: A region-based recursive
   Doppler dealiasing algorithm for operational weather radar. *J. Atmos. Oceanic
   Technol.*, **37** (12), 2341-2356, https://doi.org/10.1175/JTECH-D-20-0054.1
.. [2] James, C. N., and R. A. Houze, 2001: A real-time four-dimensional Doppler
   dealiasing scheme. *J. Atmos. Oceanic Technol.*, **18** (10), 1674-1683,
   https://doi.org/10.1175/1520-0426(2001)018<1674:ARTFDD>2.0.CO;2
.. [3] Jing, Z., and G. Wiener, 1993: Two-dimensional dealiasing of Doppler
   velocities. *J. Atmos. Oceanic Technol.*, **10** (6), 798-808,
   https://doi.org/10.1175/1520-0426(1993)010<0798:TDDODV>2.0.CO;2
"""

from __future__ import annotations

__all__ = ["dealias_r2d2", "dealias_ambiguity"]

__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

import functools

import numpy as np
import xarray as xr

from .._fields import VELOCITY, as_sweep, find_field
from .._provenance import provenance
from .._registry import accessor_method
from .._sweepwise import (
    Output,
    Sweep,
    apply_sweeps,
    dask_support,
    ray_coordinate,
    ray_dimension,
)
from ..fundamentals.geometry import beam_center_height
from . import _dealias_methods_numpy as _spec
from ._products import product_tree

try:
    from . import _dealias_methods as _kernel

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _kernel = None
    HAS_COMPILED_KERNEL = False

# The constants below are radarx's own choices, not values from a paper.
_NYQUIST_TOLERANCE = 1.01  # values beyond this times Vn are flags, not data
_MAX_SPEED = 4000.0  # m/s; larger values are not velocities
_MIN_OVERLAP = 5  # gates a region must share with a guidance field
_WINDOW_MIN = 3  # resolved gates the window needs
_FLAGS = {
    "weak_margin": 1,
    "short_boundary": 2,
    "no_boundary": 4,
    "differs_from_compare": 8,
}


def _use_compiled(engine):
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled dealiasing-methods kernel is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


def _is_full_circle(azimuth):
    """Whether the rays cover the whole circle (the first ray follows the last)."""
    az = np.sort(np.mod(np.asarray(azimuth, dtype=np.float64), 360.0))
    if az.size < 3:
        return False
    typical = np.median(np.diff(az))
    return bool(360.0 - (az[-1] - az[0]) <= 2.0 * typical)


def _quantize(values, nyquist=None):
    """Velocities in steps of 1/256 m/s as int32, MISSING for no data and flags."""
    v = np.asarray(values, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        bad = ~np.isfinite(v) | (np.abs(v) > _MAX_SPEED)
        if nyquist is not None:
            bad |= np.abs(v) > _NYQUIST_TOLERANCE * nyquist
    q = np.rint(np.where(bad, 0.0, v) * _spec.VEL_Q).astype(np.int32)
    q[bad] = _spec.MISSING
    return np.ascontiguousarray(q)


def _nyquist_of(ds, field, override, sweep=None):
    from .dealias import _nyquist

    return _nyquist(ds, field, override, sweep)


def _sweep_nodes(dtree, field):
    from ..grid.cone import _sweep_dataset, _sweep_names

    names = [n for n in _sweep_names(dtree) if field in dtree[n].data_vars]
    return names, {n: _sweep_dataset(dtree, n) for n in names}


def _nearest(source, target, tolerance, period=None):
    """Index of the nearest ``source`` value for each ``target`` (-1 if too far)."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if period is not None:
        source, target = np.mod(source, period), np.mod(target, period)
    order = np.argsort(source, kind="stable")
    s = source[order]
    pos = np.searchsorted(s, target)
    lo, hi = np.clip(pos - 1, 0, s.size - 1), np.clip(pos, 0, s.size - 1)
    dist_lo, dist_hi = np.abs(target - s[lo]), np.abs(target - s[hi])
    best = np.where(dist_hi < dist_lo, hi, lo)
    dist = np.minimum(dist_lo, dist_hi)
    if period is not None:  # across 0 / 360
        around_first = np.abs(target - s[0] - period)
        best = np.where(around_first < dist, 0, best)
        dist = np.minimum(dist, around_first)
        around_last = np.abs(target + period - s[-1])
        best = np.where(around_last < dist, s.size - 1, best)
        dist = np.minimum(dist, around_last)
    return np.where(dist <= tolerance, order[best], -1)


def _on_grid(da, ds, ray):
    """Values of ``da`` (a sweep field with coordinates) on the grid of ``ds``, (ray, range)."""
    if da.ndim != 2:
        raise ValueError("guidance fields must be 2-D sweeps (azimuth, range)")
    src_ray = ray_dimension(da)
    values = np.asarray(da.transpose(src_ray, "range").values, dtype=np.float64)
    if src_ray not in da.coords or "range" not in da.coords:
        if values.shape != (ds.sizes[ray], ds.sizes["range"]):
            raise ValueError(
                f"a guidance field without coordinates must have the shape "
                f"{(ds.sizes[ray], ds.sizes['range'])} of the sweep, not {values.shape}"
            )
        return values
    src_az = np.asarray(da[src_ray].values, dtype=np.float64)
    src_rng = np.asarray(da["range"].values, dtype=np.float64)
    az = np.asarray(ds[ray].values, dtype=np.float64)
    rng = np.asarray(ds["range"].values, dtype=np.float64)
    if src_az.shape == az.shape and src_rng.shape == rng.shape:
        if np.allclose(src_az, az) and np.allclose(src_rng, rng):
            return values
    az_step = np.median(np.abs(np.diff(az))) if az.size > 1 else 1.0
    rng_step = np.median(np.diff(rng)) if rng.size > 1 else 1.0
    iray = _nearest(src_az, az, 1.5 * az_step, 360.0 if ray == "azimuth" else None)
    igate = _nearest(src_rng, rng, 1.5 * rng_step)
    out = values[np.clip(iray, 0, None)][:, np.clip(igate, 0, None)]
    out[iray < 0] = np.nan
    out[:, igate < 0] = np.nan
    return out


def _radial_wind(ds, ray, wind_profile):
    """Radial component of a ``u`` / ``v`` wind profile at every gate, (ray, range)."""
    height = np.asarray(wind_profile["height"].values, dtype=np.float64)
    order = np.argsort(height)
    elevation = np.broadcast_to(
        np.asarray(ds["elevation"].values, dtype=np.float64), ds[ray].shape
    )
    rng = np.asarray(ds["range"].values, dtype=np.float64)[None, :]
    altitude = float(ds["altitude"]) if "altitude" in ds.variables else 0.0
    z = beam_center_height(rng, elevation[:, None], radar_height=altitude)
    u = np.interp(z, height[order], wind_profile["u"].values[order], np.nan, np.nan)
    v = np.interp(z, height[order], wind_profile["v"].values[order], np.nan, np.nan)
    az = np.deg2rad(np.asarray(ds["azimuth"].values, dtype=np.float64))[:, None]
    return (u * np.sin(az) + v * np.cos(az)) * np.cos(np.deg2rad(elevation))[:, None]


def _check_options(
    shear_threshold,
    buffer,
    spread_threshold,
    thresholds,
    window,
    max_iterations,
    patience,
):
    if not shear_threshold > 0:
        raise ValueError("shear_threshold must be positive")
    if int(buffer) != buffer or buffer < 1 or buffer % 2 == 0:
        raise ValueError("buffer must be an odd number of gates")
    if not spread_threshold > 0:
        raise ValueError("spread_threshold must be positive")
    thresholds = tuple(float(t) for t in thresholds)
    if not thresholds or any(not 0 < t <= 1 for t in thresholds):
        raise ValueError("thresholds must be Nyquist fractions in (0, 1]")
    if int(window) != window or window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd number of at least 3")
    for value, what in ((max_iterations, "max_iterations"), (patience, "patience")):
        if int(value) != value or value < 1:
            raise ValueError(f"{what} must be a positive integer")
    return thresholds


def _params(nyquist, options, ref_fallback):
    q = _spec.VEL_Q
    return {
        "beta": int(round(options["shear_threshold"] * nyquist * q)),
        "gamma": float(options["spread_threshold"] * nyquist * q),
        "min_overlap": _MIN_OVERLAP,
        "buffer_half": int(options["buffer"]) // 2,
        "thresholds": [int(round(t * nyquist * q)) for t in options["thresholds"]],
        "window_half": int(options["window"]) // 2,
        "window_min": _WINDOW_MIN,
        "max_iterations": int(options["max_iterations"]),
        "patience": int(options["patience"]),
        "ref_fallback": bool(ref_fallback),
    }


def _r2d2_folds(q, nyquist, guides, wrap, options, ref_fallback, compiled, n_threads):
    nyq2 = int(round(2.0 * nyquist * _spec.VEL_Q))
    p = _params(nyquist, options, ref_fallback)
    if compiled:
        empty = np.zeros((0, 0), dtype=np.int32)
        folds, info = _kernel.r2d2_sweep(
            q,
            empty if guides["ref"] is None else guides["ref"],
            empty if guides["above"] is None else guides["above"],
            empty if guides["prev"] is None else guides["prev"],
            bool(wrap),
            nyq2,
            p["beta"],
            p["gamma"],
            p["min_overlap"],
            p["buffer_half"],
            p["thresholds"],
            p["window_half"],
            p["window_min"],
            p["max_iterations"],
            p["patience"],
            p["ref_fallback"],
            int(n_threads or 0),
        )
        names = ("iterations", "best", "unresolved", "buffer")
        return folds, dict(zip(names, (int(i) for i in info)))
    return _spec.r2d2_sweep(q, nyq2, guides, wrap, p)


def _first_guess(ds, field, ray, nyquist, reference, wind_profile, options):
    """The first guess on (ray, range) and whether it only fills in."""
    if reference is None and wind_profile is None:
        from .dealias import dealias_velocity

        guess = dealias_velocity(
            ds,
            field,
            nyquist,
            n_threads=options["n_threads"],
            engine=options["engine"],
        )
        return _on_grid(guess, ds, ray), True, "region"
    parts, source = [], []
    if reference is not None:
        parts.append(_on_grid(reference, ds, ray))
        source.append("reference")
    if wind_profile is not None:
        parts.append(_radial_wind(ds, ray, wind_profile))
        source.append("wind_profile")
    guess = parts[0]
    for other in parts[1:]:
        guess = np.where(np.isnan(guess), other, guess)
    return guess, False, "+".join(source)


def _previous_for(previous, sweep, name, field):
    """The dealiased velocity of the previous volume for a sweep, or None."""
    if previous is None:
        return None
    candidates = (name, f"{field}_dealiased", field)
    if isinstance(previous, xr.DataArray):
        return previous
    if isinstance(previous, xr.DataTree):
        if sweep is None or sweep not in previous:
            return None
        node = previous[sweep].to_dataset()
    else:
        node = previous
    for candidate in candidates:
        if candidate in node.data_vars:
            return node[candidate]
    return None


def _r2d2_one(ds, field, nyquist, sweep, above, previous, reference, wind_profile, o):
    da = ds[field]
    if da.ndim != 2:
        raise ValueError(
            "dealias_r2d2 needs sweeps on (azimuth, range); a field with extra "
            f"dimensions {da.dims} is not supported"
        )
    ray = ray_dimension(da)
    values = np.asarray(da.transpose(ray, "range").values, dtype=np.float64)
    q = _quantize(values, nyquist)
    guess, fallback, source = _first_guess(
        ds, field, ray, nyquist, reference, wind_profile, o
    )
    guides = {
        "ref": _quantize(guess),
        "above": None if above is None else _quantize(_on_grid(above, ds, ray)),
        "prev": None if previous is None else _quantize(_on_grid(previous, ds, ray)),
    }
    wrap = _is_full_circle(ray_coordinate(ds, ray))
    folds, info = _r2d2_folds(
        q, nyquist, guides, wrap, o, fallback, o["compiled"], o["n_threads"]
    )
    out = np.where(q == _spec.MISSING, np.nan, values + 2.0 * nyquist * folds)
    if da.dims[0] != ray:
        out = out.T
    dtype = da.dtype if np.issubdtype(da.dtype, np.floating) else np.float64
    attrs = dict(da.attrs)
    attrs.update(
        {
            "long_name": "Dealiased " + da.attrs.get("long_name", "radial velocity"),
            "nyquist_velocity": nyquist,
            "comment": "Nyquist folds unfolded with radarx.retrieve.dealias_r2d2 "
            "(R2D2, Feldmann et al. 2020)",
            "dealias_first_guess": source,
            "r2d2_iterations": info["iterations"],
            "r2d2_best_iteration": info["best"],
            "r2d2_unresolved_gates": info["unresolved"],
        }
    )
    attrs.setdefault("units", "m s-1")
    attrs.setdefault(
        "standard_name", "radial_velocity_of_scatterers_away_from_instrument"
    )
    return da.copy(data=out.astype(dtype, copy=False)).assign_attrs(attrs)


@dask_support(
    "computes",
    "each sweep follows the dealiased sweep above it and the recursion needs the whole sweep",
)
@provenance(
    "Region-based recursive Doppler dealiasing (Feldmann et al. 2020); "
    "the thresholds of the scanning passes are radarx choices"
)
def dealias_r2d2(
    obj,
    field=None,
    nyquist_velocity=None,
    *,
    reference=None,
    wind_profile=None,
    previous=None,
    shear_threshold=0.8,
    buffer=5,
    spread_threshold=0.5,
    thresholds=(0.3, 0.5, 0.7, 0.9),
    window=11,
    max_iterations=10,
    patience=2,
    name=None,
    n_threads=None,
    engine="auto",
):
    """
    Dealias Doppler velocities with R2D2, the region-based recursive method.

    The method is described in :mod:`radarx.retrieve.dealias_methods`. It
    separates the gates into regions that lie in one Nyquist interval and shear
    zones (buffers), gives each region the fold that agrees with the guidance
    (the sweep above, the previous volume, a first guess), resolves the shear
    zones by continuity from their resolved neighbours, and repeats on the
    result until the largest gate-to-gate difference stops decreasing.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataTree
        A PPI sweep on ``(azimuth, range)``, a sweep node of a volume, or a
        volume with ``sweep_*`` groups. A volume is processed from the highest
        sweep to the lowest, each sweep guided by the one above it.
    field : str, optional
        Radial velocity field. Default: the first of ``VRADH``, ``VRAD``,
        ``VRADV``, ``VEL``, ``VELOCITY``, ``velocity`` that the sweep holds.
    nyquist_velocity : float or dict, optional
        Nyquist velocity in m s-1, read from the sweep by default, as in
        :func:`radarx.retrieve.dealias_velocity`.
    reference : xarray.DataArray, optional
        First guess of the radial velocity on the sweep grid (a sweep only),
        e.g. from a model; NaN where it is not known.
    wind_profile : xarray.Dataset, optional
        Horizontal wind profile with ``u`` and ``v`` in m s-1 on a ``height``
        coordinate (m above sea level); its radial component is a first guess
        (where ``reference`` has none).
    previous : xarray.DataArray, xarray.Dataset or xarray.DataTree, optional
        The dealiased velocity of the previous volume (the output of this
        function or of :func:`radarx.retrieve.dealias_velocity`): for a sweep a
        DataArray or a Dataset with ``name`` or ``f"{field}_dealiased"``; for a
        volume a DataTree with the same sweep names. Its gates are mapped to the
        nearest gate of the sweep.
    shear_threshold : float, optional
        A gate is in a shear zone when its largest difference to a neighbour
        reaches this many Nyquist velocities. Default 0.8, the paper's example.
    buffer : int, optional
        Edge in gates (odd) of the square around a shear gate that is excluded
        from the regions. Default 5, as in the paper.
    spread_threshold : float, optional
        A region joins the buffer zones when the standard deviation of its
        differences to a guidance field exceeds this many Nyquist velocities
        (it then holds a fold or strong shear). Default 0.5, a radarx choice.
    thresholds : sequence of float, optional
        Thresholds, in Nyquist velocities, of the scanning passes that resolve
        the buffer zones, from strict to relaxed. Default (0.3, 0.5, 0.7, 0.9),
        radarx's choice.
    window : int, optional
        Edge in gates (odd) of the window in which the remaining gates follow
        the median of the resolved ones. Default 11 (10 in the paper).
    max_iterations : int, optional
        Most recursions. Default 10.
    patience : int, optional
        Stop after this many recursions without a lower mean of the largest
        difference. Default 2.
    name : str, optional
        Name of the result. Default ``f"{field}_dealiased"``.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use; both give the same folds.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        The dealiased velocity on the sweep's coordinates; a DataTree with one
        node per sweep that has ``field`` for a volume. NaN where the input has
        no data or a flag value (``|v|`` beyond 1.01 Nyquist velocities). The
        attributes ``dealias_first_guess``, ``r2d2_iterations``,
        ``r2d2_best_iteration`` and ``r2d2_unresolved_gates`` (gates that no
        step resolved, which keep their measured velocity) describe the run.

    Raises
    ------
    ValueError
        If the Nyquist velocity is unknown, an option is out of range or
        ``reference`` is given for a volume.
    ImportError
        If ``engine="compiled"`` and the compiled kernel is not available.

    Notes
    -----
    The first guess decides the fold of a region where the other guidance does not
    reach. The default first guess, the region method, makes that a refinement
    of the region method that treats shear zones differently (a velocity
    couplet is resolved from its surroundings and not shifted with a folded
    region next to it); a first guess from a model or sounding makes R2D2 the
    independent method of the paper, and is advised.

    References
    ----------
    .. [1] Feldmann, M., C. N. James, M. Boscacci, D. Leuenberger, M. Gabella,
       U. Germann, D. Wolfensberger, and A. Berne, 2020: R2D2: A region-based
       recursive Doppler dealiasing algorithm for operational weather radar.
       *J. Atmos. Oceanic Technol.*, **37** (12), 2341-2356,
       https://doi.org/10.1175/JTECH-D-20-0054.1
    .. [2] James, C. N., and R. A. Houze, 2001: A real-time four-dimensional
       Doppler dealiasing scheme. *J. Atmos. Oceanic Technol.*, **18** (10),
       1674-1683,
       https://doi.org/10.1175/1520-0426(2001)018<1674:ARTFDD>2.0.CO;2

    Examples
    --------
    >>> vr = radarx.retrieve.dealias_r2d2(ds, "VRADH", 26.0)  # doctest: +SKIP
    >>> dtree.radarx.assign(radarx.retrieve.dealias_r2d2(dtree, wind_profile=sounding))  # doctest: +SKIP
    """
    thresholds = _check_options(
        shear_threshold,
        buffer,
        spread_threshold,
        thresholds,
        window,
        max_iterations,
        patience,
    )
    compiled = _use_compiled(engine)
    obj = as_sweep(obj)
    field = find_field(obj, VELOCITY, field)
    name = name or f"{field}_dealiased"
    options = {
        "shear_threshold": float(shear_threshold),
        "buffer": int(buffer),
        "spread_threshold": float(spread_threshold),
        "thresholds": thresholds,
        "window": int(window),
        "max_iterations": int(max_iterations),
        "patience": int(patience),
        "n_threads": n_threads,
        "engine": engine,
        "compiled": compiled,
    }
    if isinstance(obj, xr.Dataset):
        ds = obj.assign({field: obj[field].compute()})
        nyquist = _nyquist_of(ds, field, nyquist_velocity)
        prev = _previous_for(previous, None, name, field)
        result = _r2d2_one(
            ds, field, nyquist, None, None, prev, reference, wind_profile, options
        )
        return result.rename(name)
    if reference is not None:
        raise ValueError("reference is only supported for a single sweep.")
    names, datasets = _sweep_nodes(obj, field)
    if not names:
        raise ValueError(f"No sweep contains {field!r}.")
    elevation = {n: float(np.nanmedian(datasets[n]["elevation"].values)) for n in names}
    names.sort(key=lambda n: -elevation[n])  # the highest sweep first
    nodes, above = {}, None
    for n in names:
        ds = datasets[n].assign({field: datasets[n][field].compute()})
        nyquist = _nyquist_of(ds, field, nyquist_velocity, n)
        prev = _previous_for(previous, n, name, field)
        result = _r2d2_one(
            ds, field, nyquist, n, above, prev, None, wind_profile, options
        )
        nodes[n] = result.rename(name).to_dataset()
        above = result
    return product_tree(obj, {n: nodes[n] for n in sorted(nodes)})


@accessor_method("dataset", "datatree", name="dealias_r2d2", wraps=dealias_r2d2)
def _dealias_r2d2_accessor(self):
    """
    Dealias the radial velocity with R2D2.

    See :func:`radarx.retrieve.dealias_r2d2` for the parameters.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        The dealiased velocity.
    """


# --------------------------------------------------------------------------
# Fold margin
# --------------------------------------------------------------------------


def _margin_core(batch, min_margin, min_boundary, threshold, n_threads, compiled):
    """
    Margin, flags, region size (and fold spread) of sweeps given as
    ``((azimuth, nyquist), (v, compare_1, ...))``.
    """
    qs, wraps, thr = [], [], []
    for (azimuth, nyquist), arrays in batch:
        qs.append(_quantize(arrays[0]))
        wraps.append(_is_full_circle(azimuth))
        thr.append(int(round(threshold * nyquist * _spec.VEL_Q)))
    if compiled:
        solved = _kernel.fold_margin(qs, wraps, thr, int(n_threads or 0))
    else:
        solved = [_spec.fold_margin(q, t, w) for q, w, t in zip(qs, wraps, thr)]
    out = []
    for ((azimuth, nyquist), arrays), (region, size, pairs, total) in zip(
        batch, solved
    ):
        n2 = 2.0 * nyquist * _spec.VEL_Q
        has = pairs > 0
        margin = np.where(
            has, 1.0 - 2.0 * np.abs(total) / (np.maximum(pairs, 1) * n2), 0.0
        )
        margin = np.clip(margin, -1.0, 1.0)
        flag = (
            _FLAGS["weak_margin"] * (has & (margin < min_margin))
            + _FLAGS["short_boundary"] * (has & (pairs < min_boundary))
            + _FLAGS["no_boundary"] * (~has)
        ).astype(np.uint8)
        valid = region >= 0
        index = np.where(valid, region, 0)
        gate_flag = np.where(valid, flag[index], 0).astype(np.uint8)
        result = [
            np.where(valid, margin[index], np.nan).astype(np.float32),
            gate_flag,
            np.where(valid, size[index], 0).astype(np.int32),
        ]
        if len(arrays) > 1:
            spread = np.zeros(region.shape, dtype=np.int8)
            v = np.asarray(arrays[0], dtype=np.float64)
            for other in arrays[1:]:
                with np.errstate(invalid="ignore"):
                    steps = np.floor(
                        np.abs(np.asarray(other) - v) / (2.0 * nyquist) + 0.5
                    )
                spread = np.maximum(
                    spread, np.nan_to_num(steps).clip(0, 127).astype(np.int8)
                )
            result[1] = result[1] | (
                (spread > 0) * _FLAGS["differs_from_compare"]
            ).astype(np.uint8)
            result.append(spread)
        out.append(tuple(result))
    return out


def _compare_fields(ds, compare, da):
    """The comparison fields of a sweep as DataArrays on the grid of ``da``."""
    if compare is None:
        return []
    items = [compare] if isinstance(compare, (str, xr.DataArray)) else list(compare)
    fields = []
    for item in items:
        other = ds[item] if isinstance(item, str) else item
        fields.append(other.broadcast_like(da).transpose(*da.dims))
    return fields


def _ambiguity_sweep(ds, field, nyquist, compare, name, o):
    da = ds[field]
    others = _compare_fields(ds, compare, da)
    core = functools.partial(
        _margin_core,
        min_margin=o["min_margin"],
        min_boundary=o["min_boundary"],
        threshold=o["threshold"],
        n_threads=o["n_threads"],
        compiled=o["compiled"],
    )
    azimuth = ray_coordinate(ds, ray_dimension(da))
    outputs = (Output(np.float32), Output(np.uint8), Output(np.int32))
    if others:
        outputs += (Output(np.int8),)
    results = apply_sweeps(
        core, Sweep([da, *others], context=(azimuth, nyquist)), outputs
    )
    margin, flag, size = results[:3]
    out = {
        f"{name}_fold_margin": margin.assign_attrs(
            long_name="margin of the fold of the dealiased velocity",
            units="1",
            valid_range=[-1.0, 1.0],
            comment="1 - 2 |mean difference across the region boundary| / (2 Vn): "
            "1 where the neighbouring regions are at the same velocity, 0 where "
            "the next fold fits as well, below 0 where it fits better "
            "(a radarx definition on the criterion of Jing and Wiener 1993)",
            threshold=o["threshold"],
            nyquist_velocity=nyquist,
        ),
        f"{name}_fold_flag": flag.assign_attrs(
            long_name="flags of the fold of the dealiased velocity",
            flag_masks=np.array(list(_FLAGS.values()), dtype=np.uint8),
            flag_meanings=" ".join(_FLAGS),
            min_margin=o["min_margin"],
            min_boundary=o["min_boundary"],
        ),
        f"{name}_region_size": size.assign_attrs(
            long_name="gates in the region of the gate", units="1"
        ),
    }
    if others:
        out[f"{name}_fold_spread"] = results[3].assign_attrs(
            long_name="largest number of folds by which a comparison field differs",
            units="1",
        )
    return xr.Dataset(out)


@dask_support("lazy")
@provenance(
    "Margin of the fold of dealiased velocity on the criterion of Jing and Wiener 1993, "
    "a radarx definition"
)
def dealias_ambiguity(
    obj,
    field=None,
    nyquist_velocity=None,
    *,
    threshold=0.3,
    min_margin=0.25,
    min_boundary=10,
    compare=None,
    name=None,
    n_threads=None,
    engine="auto",
):
    """
    Where the fold of a dealiased velocity is not settled by the data.

    Two measures of the fold choice. The *margin* is the evidence in the field
    itself. The dealiased velocity is cut into regions of neighbouring gates (along
    and across the rays) that differ by less than ``threshold`` Nyquist
    velocities. For a region, the evidence for its fold is the increase of the
    summed squared velocity differences across its boundary (the least-squares
    criterion of Jing and Wiener 1993 [1]_) when the region is moved to the next
    fold, up or down, the smaller of the two, per boundary pair. With :math:`d_j =
    (v_j - v_i) / 2 V_n` for the :math:`n` pairs of a gate :math:`i` in the
    region and its neighbour :math:`j` outside it,

    .. math::

        m = 1 - \\frac{2}{n} \\left| \\sum_j d_j \\right| .

    :math:`m = 1` where all neighbours are at the velocity of the region,
    :math:`m = 0` where they differ by :math:`V_n` on average, so that the next
    fold fits as well, and :math:`m < 0` where it fits better. This is a radarx
    definition on the criterion of the paper; no paper defines such a measure.

    The *spread* is the disagreement with other dealiasings (``compare``): another
    method, another tilt or another setting. Where the alternatives choose another
    fold, the choice was ambiguous.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataTree
        A sweep or a volume with the dealiased velocity.
    field : str, optional
        Dealiased velocity. Default: the first of ``VRADH_dealiased``,
        ``VRAD_dealiased``, ... (the names :func:`radarx.retrieve.dealias_velocity`
        gives), then ``VRADH``, ``VRAD``, ... that the sweep holds.
    nyquist_velocity : float or dict, optional
        Nyquist velocity (m s-1); by default read from the sweep or from the
        ``nyquist_velocity`` attribute that the dealiasing writes.
    threshold : float, optional
        Neighbouring gates belong to one region when their velocities differ by
        less than this many Nyquist velocities. Default 0.3, as in the region
        method of :func:`radarx.retrieve.dealias_velocity`.
    min_margin : float, optional
        Margin below which a region gets the flag ``weak_margin``. Default 0.25,
        a mean velocity difference of 0.75 Nyquist velocities across the
        boundary; a radarx choice.
    min_boundary : int, optional
        Fewest boundary pairs for a firm fold; fewer give ``short_boundary``.
        Default 10, a radarx choice.
    compare : str, xarray.DataArray or list of them, optional
        Other dealiased fields on the same grid (the name of a field of the sweep,
        which for a volume refers to every sweep, or a DataArray for a sweep): the
        gates where one of them differs by at least half a fold (one Nyquist
        velocity) get the flag ``differs_from_compare``, and ``fold_spread`` is
        the largest number of folds by which they differ.
    name : str, optional
        Prefix of the results. Default: ``field``.
    n_threads : int, optional
        Threads for the compiled kernel (sweeps of a stack in parallel). Default:
        all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use; both give the same values.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``{name}_fold_margin`` (float32, NaN without data), ``{name}_fold_flag``
        (uint8, the sum of ``weak_margin`` = 1, ``short_boundary`` = 2,
        ``no_boundary`` = 4 and ``differs_from_compare`` = 8) and
        ``{name}_region_size`` (int32, the gates of the region of the gate), on
        the sweep's dimensions, and ``{name}_fold_spread`` (int8) with
        ``compare``; a DataTree with one node per sweep for a volume. A region
        without boundary (``no_boundary``) has margin 0: its fold is set by the
        reference alone.

    Notes
    -----
    What the margin says. It finds a region whose neighbours contradict its fold
    (a block one fold off inside a smooth field, with margin -1), regions held
    by few gate pairs and regions without a neighbour, such as an echo that is
    disconnected from the rest; ``region_size`` tells these from specks. It
    cannot find an error that the field does not show: a large block that is one
    fold off but joined to the rest by a smooth neck is one region, and a block
    that is off together with everything it touches has a clear margin. The
    sweep that the region method of :func:`radarx.retrieve.dealias_velocity` puts
    one fold off over a disconnected block (KBMX, 27 April 2011, 0.5 degrees) is
    such a case: the margin of the block is as high as that of the rest, while
    ``compare`` with the UNRAVEL result flags all of its gates. Use ``compare``
    for that kind of ambiguity. Real shear greater than half a Nyquist velocity
    across a boundary also gives a weak margin.

    References
    ----------
    .. [1] Jing, Z., and G. Wiener, 1993: Two-dimensional dealiasing of
       Doppler velocities. *J. Atmos. Oceanic Technol.*, **10** (6), 798-808,
       https://doi.org/10.1175/1520-0426(1993)010<0798:TDDODV>2.0.CO;2

    Examples
    --------
    >>> dea = dtree.radarx.dealias("VRADH")  # doctest: +SKIP
    >>> amb = dtree.radarx.assign(dea).radarx.dealias_ambiguity()  # doctest: +SKIP
    >>> unr = dtree.radarx.dealias("VRADH", method="unravel", name="VRADH_unravel")  # doctest: +SKIP
    >>> amb = dtree.radarx.assign(dea).radarx.assign(unr).radarx.dealias_ambiguity(  # doctest: +SKIP
    ...     compare="VRADH_unravel")
    """
    from .dealias import _DEALIASED

    if not threshold > 0:
        raise ValueError("threshold must be positive")
    if int(min_boundary) != min_boundary or min_boundary < 1:
        raise ValueError("min_boundary must be a positive integer")
    if not -1.0 <= min_margin <= 1.0:
        raise ValueError("min_margin must be between -1 and 1")
    obj = as_sweep(obj)
    field = find_field(obj, _DEALIASED, field)
    options = {
        "threshold": float(threshold),
        "min_margin": float(min_margin),
        "min_boundary": int(min_boundary),
        "n_threads": n_threads,
        "compiled": _use_compiled(engine),
    }
    prefix = name or field
    if isinstance(obj, xr.Dataset):
        nyquist = _nyquist_of(obj, field, nyquist_velocity)
        return _ambiguity_sweep(obj, field, nyquist, compare, prefix, options)
    items = [compare] if isinstance(compare, (str, xr.DataArray)) else compare or []
    if any(not isinstance(c, str) for c in items):
        raise ValueError("compare must be the names of fields for a volume")
    names, datasets = _sweep_nodes(obj, field)
    if not names:
        raise ValueError(f"No sweep contains {field!r}.")
    nodes = {
        n: _ambiguity_sweep(
            datasets[n],
            field,
            _nyquist_of(datasets[n], field, nyquist_velocity, n),
            compare,
            prefix,
            options,
        )
        for n in names
    }
    return product_tree(obj, nodes)


@accessor_method(
    "dataset", "datatree", name="dealias_ambiguity", wraps=dealias_ambiguity
)
def _dealias_ambiguity_accessor(self):
    """
    Margin and flags of the fold of a dealiased velocity.

    See :func:`radarx.retrieve.dealias_ambiguity` for the parameters.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        The fold margin, the flags, the region size and, with ``compare``, the
        fold spread.
    """
