#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

r"""
Eyewall and Rainband Diagnostics of a Tropical Cyclone from One Radar
=====================================================================

Two functions describe the radial structure of a vortex that
:func:`radarx.retrieve.gbvtd` and :func:`radarx.retrieve.tc_center` have
located:

- :func:`tc_ring_stats` gives the azimuthal statistics of any polar field around
  the centre: for every ring radius the mean, the standard deviation, the
  amplitude and phase of the azimuthal wavenumbers 1 to 3, the coverage of the
  ring by data and the fraction of it above a threshold. Applied to the
  reflectivity it gives the radial profile of the azimuthal-mean reflectivity
  of the eyewall and the rainbands and their asymmetry; applied to the
  velocity or to a polarimetric field it gives the same for those.
- :func:`tc_structure` analyses a radial profile (the axisymmetric tangential
  wind ``VT0`` of :func:`gbvtd`, or the ``MEAN`` of :func:`tc_ring_stats`): the
  eyewall maximum, its edges and width, secondary maxima and the moat between
  them, the radii at which the profile falls to given levels (wind radii when
  it is a wind), the asymmetry at the eyewall, and, for the tangential wind,
  the vorticity, the angular velocity and the angular momentum of the
  axisymmetric flow.

What the functions are based on
-------------------------------
The ring fit of :func:`tc_ring_stats` is the harmonic analysis in the azimuth
:math:`\gamma` (clockwise from north, seen from the centre),

.. math::

    f(\gamma) = A_0 + \sum_{n=1}^{3} \left(A_n \cos n\gamma
    + B_n \sin n\gamma\right),

by weighted least squares on the gates of a ring (triangular weight in the
distance from the ring radius times the gate area, as in :func:`gbvtd`). The
amplitude of the wavenumber :math:`n` is :math:`\sqrt{A_n^2 + B_n^2}` and its
phase the azimuth of its first maximum, :math:`\arctan(B_n/A_n)/n`. The mean is
:math:`A_0`, which is the mean over the whole circle even where the data have
gaps. The wavenumber :math:`n` is fitted where the largest gap of the data on
the ring is at most :math:`180/n` degrees (the gaps are measured on bins of one
degree).

The primary eyewall of :func:`tc_structure` is the highest local maximum of
the profile whose prominence reaches ``prominence``; its edges are where the
profile crosses half of the height of the maximum above the higher of the two
bases that bound it (the half-prominence width of a peak). A secondary maximum
outside or inside it, with the moat of weaker values between the two, is the
signature of concentric eyewalls and secondary wind maxima that Willoughby et
al. [1]_ described for hurricanes. The relative vorticity of the axisymmetric
tangential wind :math:`V` is :math:`\zeta = r^{-1}\partial(rV)/\partial r`,
the angular velocity :math:`\Omega = V/r` and the angular momentum
:math:`M = rV` (:math:`+ f r^2/2` for the absolute angular momentum); Kossin
and Eastin [2]_ describe the two regimes of the vorticity profile of the eye
and the eyewall, a maximum at the centre or a ring maximum in the eyewall.
Derivatives are second-order differences on the radii of the retrieval.

Choices that are radarx's own
-----------------------------
The thresholds (the prominence that counts as a maximum, the half-prominence
for the edges, the minimum number of gates, the gap limits of the
wavenumbers) are not values from the papers; the papers describe the
structures, not an algorithm. A ring profile of a few radii that are NaN is
analysed on the radii that have values, without regard to the gaps. The wind
radii are the first radius outside the maximum where the profile falls below
the level; the gale-force and hurricane-force levels of 17.5, 25.7 and 32.9 m
s-1 (34, 50 and 64 kt) are options, no default. The wavenumber-1 and 2
amplitudes of the tangential wind of :func:`gbvtd` are those in the angle of
the ring fit (``VT1_AMPLITUDE``); use :func:`radarx.retrieve.tc_decompose` for
the waves in azimuth. A radar sees an eyewall at a fixed height that rises with
range, so ring profiles are profiles on a cone, not at one height.

Compiled kernel
---------------
The ring statistics run in a multithreaded C++ kernel (``_tc_rings.cpp``: the
gates are sorted by the distance from the centre once and the rings are handed
out to the threads through an atomic counter); a NumPy implementation with the
same steps is used if it is not available. The profile analysis of
:func:`tc_structure` works on tens of values and is NumPy only.

References
----------
.. [1] Willoughby, H. E., J. A. Clos, and M. G. Shoreibah, 1982: Concentric eye
   walls, secondary wind maxima, and the evolution of the hurricane vortex.
   *J. Atmos. Sci.*, **39** (2), 395-411,
   https://doi.org/10.1175/1520-0469(1982)039<0395:CEWSWM>2.0.CO;2

.. [2] Kossin, J. P., and M. D. Eastin, 2001: Two distinct regimes in the
   kinematic and thermodynamic structure of the hurricane eye and eyewall.
   *J. Atmos. Sci.*, **58** (9), 1079-1090,
   https://doi.org/10.1175/1520-0469(2001)058<1079:TDRITK>2.0.CO;2

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = ["tc_ring_stats", "tc_structure"]

__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

import numpy as np
import xarray as xr
from scipy.signal import find_peaks, peak_widths

from .._fields import as_sweep, node_dataset
from .._provenance import provenance
from .._registry import accessor_method
from .._sweepwise import dask_support
from ..grid.cone import _sweep_names
from . import tc_winds as tw
from ._fieldtools import (
    DBZ_NAMES,
    MissingField,
    check_object,
    choose_engine,
    detach,
)
from ._products import product_tree

try:
    from . import _tc_rings

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _tc_rings = None
    HAS_COMPILED_KERNEL = False

_PIVOT_TOLERANCE = 1.0e-10
_BINS = 360
_RING_ROWS = (
    "A0",
    "A1",
    "B1",
    "A2",
    "B2",
    "A3",
    "B3",
    "N_POINTS",
    "DATA_GAP",
    "FIT_ORDER",
    "FIT_RMS",
    "STD",
    "ECHO_FRACTION",
)
_MAX_WAVENUMBER = 3


# --------------------------------------------------------------------------
# NumPy reference of the ring kernel
# --------------------------------------------------------------------------


def _largest_gap(seen):
    """Largest gap (degrees) of the occupied one-degree bins, cyclic."""
    occupied = np.flatnonzero(seen)
    if occupied.size == 0:
        return 360.0
    spacing = np.diff(np.concatenate([occupied, [occupied[0] + _BINS]])) - 1
    return float(spacing.max()) + 1.0


def _rings_numpy(gates, cx, cy, radii, cfg):
    x, y, v, w = gates
    nr = len(radii)
    out = np.full((nr, len(_RING_ROWS)), np.nan)
    d = np.hypot(x - cx, y - cy)
    g = np.arctan2(x - cx, y - cy)
    deg = np.degrees(g)
    bin_of = np.clip(
        np.floor(np.where(deg < 0.0, deg + 360.0, deg)).astype(int), 0, 359
    )
    hw = cfg["half_width"]
    for j, radius in enumerate(radii):
        off = np.abs(d - radius)
        wt = w * (1.0 - off / hw)
        sel = (off < hw) & (wt > 0.0)
        count = int(sel.sum())
        if count < cfg["min_points"]:
            continue
        wt, vv, gg = wt[sel], v[sel], g[sel]
        seen = np.zeros(_BINS, dtype=bool)
        seen[bin_of[sel]] = True
        gap = _largest_gap(seen)
        if gap > cfg["max_gap"]:
            continue
        order = 0
        for n in range(1, cfg["max_order"] + 1):
            if gap <= 180.0 / n:
                order = n
        m = 2 * order + 1
        if count <= m:
            continue
        basis = np.ones((count, m))
        for n in range(1, order + 1):
            basis[:, 2 * n - 1] = np.cos(n * gg)
            basis[:, 2 * n] = np.sin(n * gg)
        normal = (basis * wt[:, None]).T @ basis
        rhs = (basis * wt[:, None]).T @ vv
        try:
            chol = np.linalg.cholesky(normal)
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.diag(chol) ** 2 > _PIVOT_TOLERANCE * np.diag(normal)):
            continue
        k_hat = np.linalg.solve(normal, rhs)
        sw = wt.sum()
        swv = (wt * vv).sum()
        swvv = (wt * vv * vv).sum()
        rss = max(swvv - k_hat @ rhs, 0.0)
        mean = swv / sw
        row = np.full(len(_RING_ROWS), np.nan)
        row[:m] = k_hat
        row[7] = count
        row[8] = gap
        row[9] = order
        row[10] = np.sqrt(rss / sw)
        row[11] = np.sqrt(max(swvv / sw - mean * mean, 0.0))
        thr = cfg["threshold"]
        row[12] = np.nan if np.isnan(thr) else (wt * (vv >= thr)).sum() / sw
        out[j] = row
    return out


def _rings(gates, cx, cy, radii, cfg, compiled, n_threads):
    if compiled:
        return _tc_rings.rings(
            *gates,
            float(cx),
            float(cy),
            np.asarray(radii, dtype=np.float64),
            cfg["half_width"],
            cfg["max_order"],
            cfg["min_points"],
            cfg["max_gap"],
            cfg["threshold"],
            int(n_threads or 0),
        )
    return _rings_numpy(gates, cx, cy, radii, cfg)


def _rings_block(v, x, y, w, *, center, radii, cfg, compiled, n_threads):
    gates = tw._gate_arrays(v, x, y, w)
    nr = len(radii)
    if gates[0].size == 0:
        return np.full((len(_RING_ROWS), nr), np.nan)
    gates = tw._crop(gates, center, float(radii.max()) + cfg["half_width"] + 1.0)
    cx, cy = center
    return _rings(gates, cx, cy, radii, cfg, compiled, n_threads).T


def _stack_field(sweep, block, n_out):
    """Apply a block function over the (ray, range) gates, lazily for dask."""
    ray = sweep.ray_dim
    return xr.apply_ufunc(
        block,
        *detach(sweep.velocity, sweep.x, sweep.y, sweep.w),
        input_core_dims=[[ray, "range"]] * 4,
        output_core_dims=[["quantity", "radius"]],
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={
            "output_sizes": {"quantity": len(_RING_ROWS), "radius": n_out},
            "allow_rechunk": True,
        },
        keep_attrs=False,
    )


def _ring_config(ring_width, max_wavenumber, min_points, max_gap, threshold):
    if not ring_width > 0:
        raise ValueError(f"ring_width must be positive, not {ring_width!r}")
    if int(max_wavenumber) != max_wavenumber or not (
        0 <= max_wavenumber <= _MAX_WAVENUMBER
    ):
        raise ValueError(f"max_wavenumber must be 0, 1, 2 or 3, not {max_wavenumber!r}")
    if int(min_points) != min_points or min_points < 2:
        raise ValueError(
            f"min_points must be an integer of at least 2, not {min_points!r}"
        )
    if not 0 < max_gap <= 360:
        raise ValueError(f"max_gap must be between 0 and 360 degrees, not {max_gap!r}")
    if threshold is not None and not np.isfinite(threshold):
        raise ValueError(f"threshold must be finite, not {threshold!r}")
    return {
        "half_width": 0.5 * float(ring_width),
        "max_order": int(max_wavenumber),
        "min_points": int(min_points),
        "max_gap": float(max_gap),
        "threshold": np.nan if threshold is None else float(threshold),
    }


def _ring_dataset(sweep, out, radii, center, cfg, name, units):
    ds = tw._variables(out, _RING_ROWS, radii)
    coef = {k: ds[k] for k in _RING_ROWS[:7]}
    attrs = {"units": units} if units else {}
    out_ds = xr.Dataset(coords=ds.coords)
    out_ds["MEAN"] = coef["A0"].assign_attrs(
        long_name=f"Azimuthal mean of {name}", **attrs
    )
    out_ds["STD"] = ds["STD"].assign_attrs(
        long_name=f"Azimuthal standard deviation of {name} (gates of the ring)", **attrs
    )
    for n in range(1, cfg["max_order"] + 1):
        a, b = coef[f"A{n}"], coef[f"B{n}"]
        out_ds[f"WAVE{n}_AMPLITUDE"] = np.hypot(a, b).assign_attrs(
            long_name=f"Amplitude of the azimuthal wavenumber {n} of {name}", **attrs
        )
        out_ds[f"WAVE{n}_AZIMUTH"] = (
            (np.degrees(np.arctan2(b, a)) / n) % (360.0 / n)
        ).assign_attrs(
            long_name=f"Azimuth of the first maximum of the wavenumber {n}",
            units="degree",
            comment="clockwise from north, seen from the centre",
        )
    out_ds["N_POINTS"] = ds["N_POINTS"].assign_attrs(
        long_name="Number of gates in the ring", units="1"
    )
    out_ds["DATA_GAP"] = ds["DATA_GAP"].assign_attrs(
        long_name="Largest angular gap of the data on the ring", units="degree"
    )
    out_ds["FIT_ORDER"] = ds["FIT_ORDER"].assign_attrs(
        long_name="Highest wavenumber of the ring fit", units="1"
    )
    out_ds["FIT_RMS"] = ds["FIT_RMS"].assign_attrs(
        long_name=f"Weighted rms residual of the ring fit of {name}", **attrs
    )
    if not np.isnan(cfg["threshold"]):
        out_ds["ECHO_FRACTION"] = ds["ECHO_FRACTION"].assign_attrs(
            long_name=f"Fraction of the ring with {name} at or above the threshold",
            units="1",
            threshold=cfg["threshold"],
        )
    cx, cy = center
    out_ds["CENTER_X"] = xr.DataArray(cx).assign_attrs(
        long_name="Centre, metres east of the radar", units="m"
    )
    out_ds["CENTER_Y"] = xr.DataArray(cy).assign_attrs(
        long_name="Centre, metres north of the radar", units="m"
    )
    if sweep.lat is not None and sweep.lon is not None:
        lat, lon = tw._offset_to_latlon(cx, cy, sweep.lat, sweep.lon)
        out_ds["CENTER_LATITUDE"] = xr.DataArray(lat).assign_attrs(
            units="degrees_north"
        )
        out_ds["CENTER_LONGITUDE"] = xr.DataArray(lon).assign_attrs(
            units="degrees_east"
        )
    out_ds.attrs["ring_field"] = name
    out_ds.attrs["ring_max_wavenumber"] = cfg["max_order"]
    return out_ds


@dask_support("lazy_only")
@provenance(
    "Azimuthal ring statistics (mean and wavenumbers 1 to 3) of a polar field around "
    "a vortex centre",
)
def tc_ring_stats(
    obj,
    center=None,
    *,
    center_latlon=None,
    field=None,
    radii=None,
    min_radius=1000.0,
    max_radius=100000.0,
    radius_step=1000.0,
    ring_width=2000.0,
    max_wavenumber=2,
    min_points=30,
    max_gap=180.0,
    threshold=None,
    k_e=4.0 / 3.0,
    n_threads=None,
    engine="auto",
):
    """
    Azimuthal mean and wavenumbers of a polar field on rings around a centre.

    For every ring radius the gates of a plan position indicator within the
    half width of the ring are fitted with a mean and the azimuthal harmonics
    up to ``max_wavenumber`` (equations in :mod:`radarx.retrieve.tc_diagnostics`).
    Applied to the reflectivity this gives the radial profile of the eyewall
    and the rainbands and their asymmetry, for :func:`tc_structure`.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        A plan position indicator with ``azimuth`` (one value per ray),
        ``range`` (m) and ``elevation`` (per ray, else ``sweep_fixed_angle``)
        and the radar ``altitude`` (``latitude`` and ``longitude`` for
        geographic output); a DataArray is the field itself. For a volume
        every sweep that has the field is processed. Dask-backed input gives a
        lazy result.
    center : tuple of float or xarray.Dataset or xarray.DataTree, optional
        ``(x, y)`` of the centre in m east and north of the radar, or the
        result of :func:`radarx.retrieve.tc_center` (for a volume, its tree).
    center_latlon : tuple of float, optional
        ``(latitude, longitude)`` of the centre instead of ``center``.
    field : str, optional
        Name of the field. Default: the first of ``DBZH``, ``DBZ``,
        ``reflectivity`` and ``corrected_reflectivity``.
    radii : array-like, optional
        Ring radii (m). Default ``min_radius`` to ``max_radius`` in steps of
        ``radius_step``.
    min_radius, max_radius, radius_step : float, optional
        Default radii (m): 1 to 100 km in 1 km steps.
    ring_width : float, optional
        Full width of the triangular weight of a ring (m). Default 2000.
    max_wavenumber : int, optional
        Highest azimuthal wavenumber (0 to 3), reduced where the data have wide
        gaps. Default 2.
    min_points : int, optional
        Fewest gates in a ring. Default 30.
    max_gap : float, optional
        Largest angular gap (degrees) of the data of a ring that is still
        fitted. Default 180.
    threshold : float, optional
        Value (in the units of the field) for ``ECHO_FRACTION``, the fraction
        of the weight of the ring with values at or above it. Default None: no
        ``ECHO_FRACTION``.
    k_e : float, optional
        Effective Earth radius factor of the beam geometry. Default 4/3.
    n_threads : int, optional
        Threads of the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation. ``"auto"`` (default) prefers the compiled kernel.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        On the dimension ``radius`` (m): ``MEAN`` (the azimuthal mean, in the
        units of the field), ``STD`` (the standard deviation of the gates of
        the ring), ``WAVE<n>_AMPLITUDE`` and ``WAVE<n>_AZIMUTH`` (amplitude and
        azimuth in degrees clockwise from north of the first maximum, seen
        from the centre) for the wavenumbers ``n`` up to ``max_wavenumber``,
        ``N_POINTS``, ``DATA_GAP`` (degrees), ``FIT_ORDER`` and ``FIT_RMS``,
        and ``ECHO_FRACTION`` with ``threshold``. Scalars ``CENTER_X`` and
        ``CENTER_Y`` (m) and ``CENTER_LATITUDE`` and ``CENTER_LONGITUDE``.
        A ring without enough gates or with a wide gap is NaN. For a volume a
        DataTree with one node per sweep. The input is not changed.

    Raises
    ------
    ValueError
        For invalid options, a sweep without the geometry, or a centre that is
        not finite or at the radar.
    KeyError
        If no sweep of a volume has the field.
    TypeError
        If ``obj`` is not an xarray object.
    ImportError
        If ``engine="compiled"`` and the kernel is not available.

    Notes
    -----
    Gates are weighted by their area (the ground range), so that the mean is
    the mean over the area of the ring. A mean of reflectivity in dBZ is a
    mean of decibels; convert to linear units first for a mean rain rate or
    mean reflectivity factor.

    References
    ----------
    .. [1] Willoughby, H. E., J. A. Clos, and M. G. Shoreibah, 1982: Concentric
           eye walls, secondary wind maxima, and the evolution of the hurricane
           vortex. *J. Atmos. Sci.*, **39** (2), 395-411,
           https://doi.org/10.1175/1520-0469(1982)039<0395:CEWSWM>2.0.CO;2
    """
    check_object(obj)
    compiled = choose_engine(engine, HAS_COMPILED_KERNEL, "tc_ring_stats")
    radii = tw._radii_of(radii, min_radius, max_radius, radius_step)
    cfg = _ring_config(ring_width, max_wavenumber, min_points, max_gap, threshold)
    if not (np.isfinite(k_e) and k_e > 0):
        raise ValueError(f"k_e must be positive, not {k_e!r}")

    def one(ds, name):
        if isinstance(ds, xr.DataArray):
            label = field or ds.name or "field"
            sweep = tw._Sweep(ds, label, k_e)
        else:
            label = _field_name(ds, field)
            sweep = tw._Sweep(ds, label, k_e)
        cx, cy = tw._center_of(center, center_latlon, sweep, name)

        def block(v, x, y, w):
            return _rings_block(
                v,
                x,
                y,
                w,
                center=(cx, cy),
                radii=radii,
                cfg=cfg,
                compiled=compiled,
                n_threads=n_threads,
            )

        out = _stack_field(sweep, block, radii.size)
        units = sweep.velocity.attrs.get("units")
        return _ring_dataset(sweep, out, radii, (cx, cy), cfg, label, units)

    return _per_sweep_with_field(obj, one)


def _field_name(ds, field):
    if field is not None:
        if field not in ds:
            raise MissingField(f"{field!r} is not in the sweep")
        return field
    for name in DBZ_NAMES:
        if name in ds:
            return name
    raise MissingField(f"no reflectivity found (tried {DBZ_NAMES}); pass its name")


def _per_sweep_with_field(obj, func):
    """Apply ``func(sweep, name)`` to a sweep or to the sweeps of a volume that have the field."""
    obj = as_sweep(obj)
    if not isinstance(obj, xr.DataTree):
        try:
            return func(obj, None)
        except MissingField as err:
            raise KeyError(err.args[0]) from None
    nodes, missing = {}, None
    for name in _sweep_names(obj):
        try:
            nodes[name] = func(node_dataset(obj[name]), name)
        except MissingField as err:
            missing = err
    if not nodes:
        raise KeyError(
            "no sweep has the field"
            + (f" ({missing.args[0]})" if missing is not None else "")
        )
    return product_tree(obj, nodes)


# --------------------------------------------------------------------------
# profile structure
# --------------------------------------------------------------------------


def _vertex(r, y, k):
    """Radius and value of the parabola through three points around index ``k``."""
    if k <= 0 or k >= len(r) - 1:
        return float(r[k]), float(y[k])
    r0, r1, r2 = (float(v) for v in r[k - 1 : k + 2])
    y0, y1, y2 = (float(v) for v in y[k - 1 : k + 2])
    first = (y1 - y0) / (r1 - r0)
    a = ((y2 - y1) / (r2 - r1) - first) / (
        r2 - r0
    )  # y = y0 + first (r - r0) + a (r - r0)(r - r1)
    if not a < 0:
        return r1, y1
    # y[k] is not below its neighbours, so the vertex of this concave parabola
    # lies between them
    vertex = 0.5 * (r0 + r1) - 0.5 * first / a
    return vertex, y0 + first * (vertex - r0) + a * (vertex - r0) * (vertex - r1)


def _outer_crossing(r, y, start, level):
    """Radius beyond index ``start`` where ``y`` first falls below ``level``."""
    for i in range(start, len(r) - 1):
        if y[i] >= level > y[i + 1]:
            frac = (y[i] - level) / (y[i] - y[i + 1])
            return float(r[i] + frac * (r[i + 1] - r[i]))
    return np.nan


def _structure_one(r, y, prominence, levels):
    """Scalar diagnostics of a radial profile (``r`` m, ``y`` values, finite)."""
    nan = np.nan
    out = {
        k: nan
        for k in (
            "PEAK_RADIUS",
            "PEAK_VALUE",
            "INNER_EDGE",
            "OUTER_EDGE",
            "WIDTH",
            "SECONDARY_RADIUS",
            "SECONDARY_VALUE",
            "MOAT_RADIUS",
            "MOAT_VALUE",
        )
    }
    out["N_PEAKS"] = 0
    out["OUTER_LEVEL_RADIUS"] = np.full(len(levels), nan)
    if r.size < 3:
        return out
    prom = 0.1 * float(y.max() - y.min()) if prominence is None else float(prominence)
    peaks, props = find_peaks(y, prominence=max(prom, np.finfo(float).tiny))
    out["N_PEAKS"] = int(peaks.size)
    if peaks.size == 0:
        return out
    main = peaks[int(np.argmax(y[peaks]))]
    out["PEAK_RADIUS"], out["PEAK_VALUE"] = _vertex(r, y, main)
    width = peak_widths(y, [main], rel_height=0.5)
    index = np.arange(r.size)
    left, right = np.interp([width[2][0], width[3][0]], index, r)
    out["INNER_EDGE"], out["OUTER_EDGE"] = float(left), float(right)
    out["WIDTH"] = float(right - left)
    others = peaks[peaks != main]
    if others.size:
        second = others[int(np.argmax(y[others]))]
        out["SECONDARY_RADIUS"], out["SECONDARY_VALUE"] = _vertex(r, y, second)
        lo, hi = sorted((main, second))
        k = lo + int(np.argmin(y[lo : hi + 1]))
        out["MOAT_RADIUS"], out["MOAT_VALUE"] = float(r[k]), float(y[k])
    for j, level in enumerate(levels):
        out["OUTER_LEVEL_RADIUS"][j] = _outer_crossing(r, y, main, level)
    return out


def _at(r, y, radius):
    """Value of a profile at ``radius`` (linear), NaN outside the finite radii."""
    if not np.isfinite(radius) or r.size == 0 or not r[0] <= radius <= r[-1]:
        return np.nan
    return float(np.interp(radius, r, y))


def _derivative(y, r):
    """Derivative of ``y`` on the radii ``r``: second order, also at the ends."""
    return np.gradient(y, r, edge_order=2 if r.size >= 3 else 1)


def _kinematics(radius, vt, vr, f):
    """Vorticity, angular velocity, angular momentum and divergence of the profile."""
    ok = np.isfinite(vt) & (radius > 0)
    out = {
        "VORTICITY": np.full(radius.shape, np.nan),
        "ANGULAR_VELOCITY": np.full(radius.shape, np.nan),
        "RELATIVE_ANGULAR_MOMENTUM": np.full(radius.shape, np.nan),
    }
    if ok.sum() >= 2:
        r, v = radius[ok], vt[ok]
        m = r * v
        out["VORTICITY"][ok] = _derivative(m, r) / r
        out["ANGULAR_VELOCITY"][ok] = v / r
        out["RELATIVE_ANGULAR_MOMENTUM"][ok] = m
    if f is not None:
        out["ABSOLUTE_ANGULAR_MOMENTUM"] = (
            out["RELATIVE_ANGULAR_MOMENTUM"] + 0.5 * f * radius**2
        )
    if vr is not None:
        div = np.full(radius.shape, np.nan)
        ok2 = ok & np.isfinite(vr)
        if ok2.sum() >= 2:
            r = radius[ok2]
            div[ok2] = _derivative(r * vr[ok2], r) / r
        out["DIVERGENCE"] = div
    return out


_KINEMATIC_ATTRS = {
    "VORTICITY": ("Relative vorticity of the axisymmetric tangential wind", "s-1"),
    "ANGULAR_VELOCITY": ("Angular velocity of the axisymmetric tangential wind", "s-1"),
    "RELATIVE_ANGULAR_MOMENTUM": ("Relative angular momentum r V", "m2 s-1"),
    "ABSOLUTE_ANGULAR_MOMENTUM": (
        "Absolute angular momentum r V + f r^2 / 2",
        "m2 s-1",
    ),
    "DIVERGENCE": ("Divergence of the axisymmetric radial wind", "s-1"),
}


def _structure_dataset(ds, variable, prominence, levels, kinematics, latitude):
    ds = ds.compute()
    if "radius" not in ds.dims:
        raise ValueError("the dataset has no 'radius' dimension")
    if variable not in ds:
        raise ValueError(f"{variable!r} is not in the dataset")
    radius = np.asarray(ds["radius"].values, dtype=np.float64)
    values = np.asarray(ds[variable].values, dtype=np.float64)
    if values.shape != radius.shape:
        raise ValueError(f"{variable!r} must be a profile on 'radius'")
    ok = np.isfinite(values)
    r, y = radius[ok], values[ok]
    res = _structure_one(r, y, prominence, levels)
    units = ds[variable].attrs.get("units", "")
    out = xr.Dataset(coords={"radius": ds["radius"]})
    meta = {
        "PEAK_RADIUS": ("Radius of the primary maximum", "m"),
        "PEAK_VALUE": ("Value at the primary maximum", units),
        "INNER_EDGE": (
            "Inner radius where the maximum falls to half its prominence",
            "m",
        ),
        "OUTER_EDGE": (
            "Outer radius where the maximum falls to half its prominence",
            "m",
        ),
        "WIDTH": ("Width of the primary maximum at half its prominence", "m"),
        "SECONDARY_RADIUS": ("Radius of the highest secondary maximum", "m"),
        "SECONDARY_VALUE": ("Value at the highest secondary maximum", units),
        "MOAT_RADIUS": ("Radius of the minimum between the two maxima", "m"),
        "MOAT_VALUE": ("Value at the minimum between the two maxima", units),
        "N_PEAKS": ("Number of maxima that reach the prominence", "1"),
    }
    for name, (long_name, unit) in meta.items():
        out[name] = xr.DataArray(res[name]).assign_attrs(
            long_name=long_name, units=unit
        )
    if levels:
        out["OUTER_LEVEL_RADIUS"] = xr.DataArray(
            res["OUTER_LEVEL_RADIUS"],
            dims="level",
            coords={
                "level": (
                    "level",
                    np.asarray(levels, dtype=np.float64),
                    {"units": units},
                )
            },
            attrs={
                "long_name": "Radius outside the maximum where the profile falls to the level",
                "units": "m",
            },
        )
    # asymmetry of the waves at the primary maximum, relative to the profile there
    peak = res["PEAK_RADIUS"]
    for n in (1, 2, 3):
        for prefix in ("WAVE", "VT"):
            name = f"{prefix}{n}_AMPLITUDE"
            if name in ds and ds[name].dims == ("radius",):
                wave = np.asarray(ds[name].values, dtype=np.float64)
                good = ok & np.isfinite(wave)
                a = _at(radius[good], wave[good], peak)
                p = _at(r, y, peak)
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[f"ASYMMETRY{n}"] = xr.DataArray(
                        a / abs(p) if p else np.nan
                    ).assign_attrs(
                        long_name=f"Wavenumber-{n} amplitude relative to the profile at the maximum",
                        units="1",
                    )
                break
    if kinematics:
        vt = values
        vr = np.asarray(ds["VR0"].values, dtype=np.float64) if "VR0" in ds else None
        lat = latitude
        if lat is None and "CENTER_LATITUDE" in ds:
            lat = float(ds["CENTER_LATITUDE"])
        f = None if lat is None else 2.0 * 7.292115e-5 * np.sin(np.deg2rad(lat))
        for name, arr in _kinematics(radius, vt, vr, f).items():
            long_name, unit = _KINEMATIC_ATTRS[name]
            out[name] = xr.DataArray(
                arr, dims="radius", coords={"radius": ds["radius"]}
            ).assign_attrs(long_name=long_name, units=unit)
    out.attrs.update(structure_variable=variable)
    for name in ("CENTER_X", "CENTER_Y", "CENTER_LATITUDE", "CENTER_LONGITUDE"):
        if name in ds:
            out[name] = ds[name].reset_coords(drop=True)
    return out


@dask_support(
    "eager", "the profile of a ring analysis is short and is analysed in memory."
)
@provenance(
    "Eyewall, secondary maxima, moat and wind radii of a radial profile of a "
    "tropical cyclone, with the vorticity and angular momentum of the tangential wind"
)
def tc_structure(
    result,
    variable=None,
    *,
    prominence=None,
    levels=None,
    kinematics=None,
    latitude=None,
):
    """
    Eyewall, secondary maxima and wind radii of a radial profile of a vortex.

    Parameters
    ----------
    result : xarray.Dataset or xarray.DataTree
        The result of :func:`gbvtd` (profile ``VT0``) or :func:`tc_ring_stats`
        (profile ``MEAN``) on the dimension ``radius``; for a volume, its tree.
        Dask-backed results are computed.
    variable : str, optional
        The profile to analyse. Default ``VT0`` if the dataset has it, else
        ``MEAN``.
    prominence : float, optional
        Smallest prominence (in the units of the profile) of a local maximum
        that counts: the height of the maximum above the higher of the two
        lowest values that separate it from a higher maximum or the end of the
        profile. Default: 10 percent of the range of the profile.
    levels : sequence of float, optional
        Levels (in the units of the profile, e.g. 17.5, 25.7 and 32.9 m s-1)
        for ``OUTER_LEVEL_RADIUS``, the radius outside the primary maximum
        where the profile first falls below the level (linear interpolation).
        Default None: not computed.
    kinematics : bool, optional
        Also give the vorticity, the angular velocity and the angular
        momentum of the profile, as the axisymmetric tangential wind (and the
        divergence of ``VR0`` if the dataset has it). Default: True for
        ``VT0``, else False.
    latitude : float, optional
        Latitude (degrees) for the Coriolis parameter of the absolute angular
        momentum. Default: ``CENTER_LATITUDE`` of the result if it has it;
        without a latitude only the relative angular momentum is given.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        Scalars: ``PEAK_RADIUS`` and ``PEAK_VALUE`` (the highest maximum,
        refined with a parabola; NaN if the profile has no maximum that
        reaches the prominence, for instance at the edge of the radii),
        ``INNER_EDGE``, ``OUTER_EDGE`` and ``WIDTH`` (at half its prominence),
        ``SECONDARY_RADIUS`` and ``SECONDARY_VALUE`` (the highest other
        maximum), ``MOAT_RADIUS`` and ``MOAT_VALUE`` (the minimum between the
        two), ``N_PEAKS``, ``OUTER_LEVEL_RADIUS`` on ``level``,
        ``ASYMMETRY1`` to ``ASYMMETRY3`` (the amplitude of the wavenumber at
        the peak relative to the profile, if the dataset has the amplitudes)
        and the centre of the input. On ``radius`` with ``kinematics``:
        ``VORTICITY``, ``ANGULAR_VELOCITY``, ``RELATIVE_ANGULAR_MOMENTUM``,
        ``ABSOLUTE_ANGULAR_MOMENTUM`` and ``DIVERGENCE``. For a volume a
        DataTree with one node per sweep.

    Raises
    ------
    ValueError
        If the dataset has no ``radius`` profile of ``variable``.
    TypeError
        If ``result`` is not a Dataset or a DataTree.

    References
    ----------
    .. [1] Willoughby, H. E., J. A. Clos, and M. G. Shoreibah, 1982: Concentric
           eye walls, secondary wind maxima, and the evolution of the hurricane
           vortex. *J. Atmos. Sci.*, **39** (2), 395-411,
           https://doi.org/10.1175/1520-0469(1982)039<0395:CEWSWM>2.0.CO;2
    .. [2] Kossin, J. P., and M. D. Eastin, 2001: Two distinct regimes in the
           kinematic and thermodynamic structure of the hurricane eye and
           eyewall. *J. Atmos. Sci.*, **58** (9), 1079-1090,
           https://doi.org/10.1175/1520-0469(2001)058<1079:TDRITK>2.0.CO;2
    """
    if prominence is not None and not (np.isfinite(prominence) and prominence > 0):
        raise ValueError(f"prominence must be positive, not {prominence!r}")
    levels = () if levels is None else tuple(float(v) for v in np.atleast_1d(levels))
    if not all(np.isfinite(levels)):
        raise ValueError("levels must be finite")

    def one(ds):
        name = variable or ("VT0" if "VT0" in ds else "MEAN")
        kin = (name == "VT0") if kinematics is None else bool(kinematics)
        return _structure_dataset(ds, name, prominence, levels, kin, latitude)

    if isinstance(result, xr.DataTree):
        nodes = {name: one(node_dataset(result[name])) for name in _sweep_names(result)}
        if not nodes:
            raise ValueError("the volume has no sweeps")
        return product_tree(result, nodes)
    if not isinstance(result, xr.Dataset):
        raise TypeError("result must be an xarray Dataset or DataTree")
    return one(result)


# --------------------------------------------------------------------------
# accessors
# --------------------------------------------------------------------------


@accessor_method(
    "dataarray", "dataset", "datatree", name="tc_ring_stats", wraps=tc_ring_stats
)
def _tc_rings_accessor(self):
    """
    Azimuthal mean and wavenumbers of a polar field on rings around a centre.

    See :func:`radarx.retrieve.tc_ring_stats` for the parameters.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        The ring statistics on ``radius``.
    """


@accessor_method("dataset", "datatree", name="tc_structure", wraps=tc_structure)
def _tc_structure_accessor(self):
    """
    Eyewall, secondary maxima and wind radii of a radial profile of a vortex.

    See :func:`radarx.retrieve.tc_diagnostics` for the parameters.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        The structure diagnostics of the profile.
    """
