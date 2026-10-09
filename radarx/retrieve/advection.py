# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Advection Correction
====================

Shift gridded radar fields to a common analysis time.

A radar volume takes minutes to collect, and different radars scan at
different times. Precipitation moves in between, so before volumes are
merged, compared or used for multi-Doppler winds, every field has to be
moved to the same time (Gal-Chen 1982). This module

1. estimates the storm motion between two gridded volumes by FFT
   cross-correlation (:func:`estimate_motion`), as a single vector or, with
   ``tile=...``, as a smooth, spatially varying field;
2. moves fields along that motion with a semi-Lagrangian scheme
   (:func:`advect`): every output cell takes the value at its departure
   point, interpolated bilinearly or with cubic convolution, and a validity
   mask is advected with the data so that missing data never spreads;
3. interpolates in time between two volumes by advecting the earlier one
   forward and the later one backward to the target time and blending the
   two (:func:`interpolate_time`).

Provenance. The idea of moving data to a common time along the storm motion is
the frame-of-reference correction of Gal-Chen (1982); tracking the motion by
cross-correlating successive echo patterns goes back to Rinehart and Garvey
(1978); the departure-point scheme is the semi-Lagrangian method reviewed by
Staniforth and Côté (1991); cubic convolution is that of Keys (1981).
The FFT implementation of the correlation, the Hann taper, the Gaussian
high-pass, the parabolic sub-cell refinement and its iterations, the tiling,
the validity-mask handling and the blend of :func:`interpolate_time` are
radarx's own constructions, and so are all defaults. The equations of the
cited papers were not checked (the papers are not on disk), except where
a function states otherwise. Full references are in the docstrings of the
functions.

All functions work on the gridded :class:`xarray.Dataset` returned by
:func:`radarx.grid.grid_radar` or ``dtree.radarx.to_grid()`` (``x`` and ``y``
coordinates in metres, a scalar ``time``), or on a DataArray from it. The
interpolation is done by a compiled C++ kernel; if it is not available an
equivalent NumPy implementation is used.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = ["advect", "estimate_motion", "interpolate_time"]

__doc__ = __doc__.format("\n   ".join(__all__))

import datetime
import warnings
from functools import partial

import numpy as np
import xarray as xr

try:
    from . import _advection

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _advection = None
    HAS_COMPILED_KERNEL = False

_REFLECTIVITY_FIELDS = (
    "DBZH",
    "reflectivity",
    "corrected_reflectivity",
    "DBZ",
    "DBZH_CORR",
    "TH",
)
_ORDERS = {"linear": 1, "cubic": 3}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _use_compiled(engine):
    """Whether to run the compiled kernel for the requested ``engine``."""
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled advection kernel is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


def _step(obj, dim):
    """Signed, uniform grid spacing of coordinate ``dim`` (metres)."""
    if dim not in obj.coords:
        raise ValueError(f"the grid needs a {dim!r} coordinate")
    c = np.asarray(obj[dim].values, dtype=np.float64)
    if c.ndim != 1 or c.size < 2:
        raise ValueError(f"{dim!r} must be a 1-D coordinate with at least 2 points")
    d = np.diff(c)
    if not np.allclose(d, d[0], rtol=1e-6, atol=0.0) or d[0] == 0:
        raise ValueError(f"{dim!r} must be evenly spaced")
    return float(d[0])


def _time_of(obj):
    """The scalar ``time`` of a gridded volume as datetime64[ns], or None."""
    if "time" not in obj.coords and not (
        isinstance(obj, xr.Dataset) and "time" in obj.data_vars
    ):
        return None
    t = obj["time"]
    if t.ndim != 0 or not np.issubdtype(t.dtype, np.datetime64):
        return None
    return t.values.astype("datetime64[ns]")


def _seconds(dt):
    """A time step (number of seconds or timedelta) in seconds."""
    if isinstance(dt, xr.DataArray):
        dt = dt.values
    if isinstance(dt, (np.timedelta64, datetime.timedelta)) or (
        isinstance(dt, np.ndarray) and np.issubdtype(dt.dtype, np.timedelta64)
    ):
        return float(np.timedelta64(dt, "ns") / np.timedelta64(1, "s"))
    return float(dt)


def _field(obj, field):
    """The DataArray to track: ``obj`` itself, or ``field`` of a Dataset."""
    if isinstance(obj, xr.DataArray):
        return obj
    if field is None:
        for name in _REFLECTIVITY_FIELDS:
            if name in obj.data_vars:
                return obj[name]
        raise ValueError("no reflectivity field found: pass field=<name>")
    return obj[field]


def _plane(da, x, y):
    """2-D (y, x) float64 array; other dimensions are reduced by the maximum."""
    other = [d for d in da.dims if d not in (x, y)]
    if x not in da.dims or y not in da.dims:
        raise ValueError(f"the field needs dimensions {y!r} and {x!r}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        values = np.asarray(da.transpose(*other, y, x).values, dtype=np.float64)
        if other:
            values = values.reshape(-1, *values.shape[-2:])
            values = np.nanmax(values, axis=0)  # column maximum (composite)
    return values


def _gridded_vars(ds, x, y):
    """Names of floating-point variables on the horizontal grid."""
    return [
        name
        for name, da in ds.data_vars.items()
        if x in da.dims and y in da.dims and np.issubdtype(da.dtype, np.floating)
    ]


# ---------------------------------------------------------------------------
# motion estimation
# ---------------------------------------------------------------------------


def _hann(n):
    return np.hanning(n) if n > 2 else np.ones(n)


def _prepare(a, mask, floor):
    """Echo above ``floor`` inside ``mask``, tapered by a 2-D Hann window."""
    a = np.where(np.isfinite(a) & mask, a - floor, 0.0)
    np.clip(a, 0.0, None, out=a)
    return a * _hann(a.shape[0])[:, None] * _hann(a.shape[1])[None, :]


def _parabolic(lo, mid, hi):
    """
    Sub-cell offset of the vertex of the parabola through three samples.

    Standard three-point parabolic peak interpolation (elementary algebra, not
    from a specific paper).
    """
    d = lo - 2.0 * mid + hi
    return 0.5 * (lo - hi) / d if d < 0 else 0.0


def _correlate(a, b, mask, floor, sigma, max_shift):
    """
    Displacement (rows, columns) that carries ``a`` onto ``b``, and its quality.

    Plain (not phase-normalised) FFT cross-correlation of the windowed echo
    fields after a Gaussian high-pass ``1 - exp(-2 pi^2 sigma^2 k^2)`` (the
    Fourier transform of one minus a Gaussian of standard deviation
    ``sigma``; a radarx construction, as is the Hann taper); the
    peak is searched within ``max_shift`` cells and refined with a parabola.
    The quality is the normalised cross-correlation at the peak (between -1
    and 1). Returns NaN shifts when there is no echo or the peak lies on the
    edge of the search window.
    """
    from scipy import fft as sfft

    fa = _prepare(a, mask, floor)
    fb = _prepare(b, mask, floor)
    if not (fa.any() and fb.any()):
        return np.nan, np.nan, 0.0
    ny, nx = a.shape
    my, mx = max_shift
    shape = (
        sfft.next_fast_len(ny + my + 1, real=True),
        sfft.next_fast_len(nx + mx + 1, real=True),
    )
    A = sfft.rfft2(fa, s=shape, workers=-1)
    B = sfft.rfft2(fb, s=shape, workers=-1)
    if sigma is not None:
        ky = sfft.fftfreq(shape[0])[:, None]
        kx = sfft.rfftfreq(shape[1])[None, :]
        k2 = (sigma[0] * ky) ** 2 + (sigma[1] * kx) ** 2
        highpass = -np.expm1(-2.0 * np.pi**2 * k2)
        A *= highpass
        B *= highpass
    else:
        A[0, 0] = B[0, 0] = 0.0  # remove the mean
    r = sfft.irfft2(np.conj(A) * B, s=shape, workers=-1)
    # zero-lag energies by Parseval (half spectrum: interior columns twice)
    half = np.full(A.shape[1], 2.0)
    half[0] = 1.0
    if shape[1] % 2 == 0:
        half[-1] = 1.0
    size = shape[0] * shape[1]
    energy_a = (half * (A.real**2 + A.imag**2)).sum() / size
    energy_b = (half * (B.real**2 + B.imag**2)).sum() / size
    rows = np.arange(-my, my + 1) % shape[0]
    cols = np.arange(-mx, mx + 1) % shape[1]
    win = r[np.ix_(rows, cols)]
    jy, jx = np.unravel_index(np.argmax(win), win.shape)
    norm = np.sqrt(energy_a * energy_b)
    quality = float(win[jy, jx] / norm) if norm > 0 else 0.0
    if jy in (0, 2 * my) or jx in (0, 2 * mx):
        return np.nan, np.nan, quality
    dy = jy - my + _parabolic(win[jy - 1, jx], win[jy, jx], win[jy + 1, jx])
    dx = jx - mx + _parabolic(win[jy, jx - 1], win[jy, jx], win[jy, jx + 1])
    return float(dy), float(dx), quality


def _track(
    a, b, mask, floor, sigma, max_shift, rows, cols, iterations, kernel, guess=None
):
    """
    Displacement of the region ``(rows, cols)`` from ``a`` to ``b``.

    The first correlation searches ``max_shift`` cells around ``guess``
    (default: zero). Then ``b`` is shifted back by the estimate (cubic
    interpolation) and the small residual displacement is estimated again,
    ``iterations`` times. This removes the bias toward zero that the fixed
    taper window gives a single correlation.
    """
    a_t, m_t = a[rows, cols], mask[rows, cols]
    oy, ox = a_t.shape
    grid_r = np.arange(rows.start, rows.start + oy, dtype=np.float64)[None, :, None]
    grid_c = np.arange(cols.start, cols.start + ox, dtype=np.float64)[None, None, :]

    def shifted_back(dy, dx):
        src_r = np.broadcast_to(grid_r + dy, (1, oy, ox))
        src_c = np.broadcast_to(grid_c + dx, (1, oy, ox))
        return kernel(b[None], src_r, src_c, 3, 0.5)[0, 0]

    if guess is None:
        dy, dx, quality = _correlate(a_t, b[rows, cols], m_t, floor, sigma, max_shift)
    else:
        ry, rx, quality = _correlate(
            a_t, shifted_back(*guess), m_t, floor, sigma, max_shift
        )
        dy, dx = guess[0] + ry, guess[1] + rx
    if not np.isfinite(dy):
        return dy, dx, quality
    for _ in range(iterations):
        ry, rx, q = _correlate(a_t, shifted_back(dy, dx), m_t, floor, sigma, (2, 2))
        if not np.isfinite(ry):
            break
        dy, dx, quality = dy + ry, dx + rx, q
        if abs(ry) < 1e-3 and abs(rx) < 1e-3:
            break
    return dy, dx, quality


def _tile_starts(n, size, step):
    starts = list(range(0, max(n - size, 0) + 1, step))
    if starts[-1] + size < n:
        starts.append(n - size)
    return np.asarray(starts)


def _tile_motion(a, b, mask, size, overlap, min_quality, guess, **options):
    """
    Per-tile displacements (rows, columns) and tile centres (index units).

    Each tile searches around the domain-wide displacement ``guess`` within
    half its length plus two cells. Tiles with echo on less than 5 % of their
    area at either time, or a weak correlation, are NaN.
    """
    ny, nx = a.shape
    size = (min(size[0], ny), min(size[1], nx))
    step = tuple(max(1, round(s * (1.0 - overlap))) for s in size)
    r0 = _tile_starts(ny, size[0], step[0])
    c0 = _tile_starts(nx, size[1], step[1])
    dy = np.full((r0.size, c0.size), np.nan)
    dx = np.full_like(dy, np.nan)
    shift = tuple(
        int(min(np.ceil(0.5 * abs(g)) + 2, n // 2 - 1)) for g, n in zip(guess, size)
    )
    floor = options["floor"]
    echo_a = np.nan_to_num(a, nan=-np.inf) > floor
    echo_b = np.nan_to_num(b, nan=-np.inf) > floor
    for p, i in enumerate(r0):
        for q, j in enumerate(c0):
            rows, cols = slice(i, i + size[0]), slice(j, j + size[1])
            if min(echo_a[rows, cols].mean(), echo_b[rows, cols].mean()) < 0.05:
                continue
            ty, tx, quality = _track(
                a,
                b,
                mask,
                max_shift=shift,
                rows=rows,
                cols=cols,
                guess=guess,
                **options,
            )
            if quality >= min_quality:
                dy[p, q], dx[p, q] = ty, tx
    centres = (r0 + (size[0] - 1) / 2.0, c0 + (size[1] - 1) / 2.0)
    return dy, dx, centres


def _motion_field(a, b, mask, size, overlap, min_quality, smooth, guess, options):
    """Smooth per-cell displacements (rows, columns) from tiled estimates."""
    ny, nx = a.shape
    if np.isnan(guess[0]):  # no first guess: no tiled motion either
        return np.full((ny, nx), np.nan), np.full((ny, nx), np.nan)
    trows, tcols, centres = _tile_motion(
        a, b, mask, size, overlap, min_quality, guess, **options
    )
    trows = _smooth_fill(trows, guess[0], smooth)
    tcols = _smooth_fill(tcols, guess[1], smooth)
    return _to_grid(trows, centres, ny, nx), _to_grid(tcols, centres, ny, nx)


def _pair_seconds(obj_t0, obj_t1, dt):
    """Time between two volumes in seconds, from ``dt`` or their times."""
    if dt is None:
        t0, t1 = _time_of(obj_t0), _time_of(obj_t1)
        if t0 is None or t1 is None:
            raise ValueError("the volumes have no scalar 'time': pass dt=")
        dt_s = float((t1 - t0) / np.timedelta64(1, "s"))
    else:
        dt_s = _seconds(dt)
    if dt_s == 0 or not np.isfinite(dt_s):
        raise ValueError("the two volumes must be at different times")
    return dt_s


def _smooth_fill(values, fallback, sigma):
    """Fill failed tiles with ``fallback`` and smooth with a Gaussian (tiles)."""
    from scipy.ndimage import gaussian_filter

    filled = np.where(np.isfinite(values), values, fallback)
    if sigma and filled.size > 1:
        filled = gaussian_filter(filled, sigma, mode="nearest")
    return filled


def _to_grid(values, centres, ny, nx):
    """Bilinear interpolation of tile values to every grid cell (clamped)."""
    rows = np.arange(ny, dtype=np.float64)
    cols = np.arange(nx, dtype=np.float64)
    along_x = np.array([np.interp(cols, centres[1], v) for v in values])
    return np.array([np.interp(rows, centres[0], c) for c in along_x.T]).T


def estimate_motion(
    obj_t0,
    obj_t1,
    field=None,
    *,
    dt=None,
    floor=5.0,
    highpass=10e3,
    max_speed=60.0,
    min_quality=0.3,
    iterations=3,
    tile=None,
    overlap=0.5,
    smooth=1.0,
    observed=None,
    x="x",
    y="y",
    engine="auto",
    n_threads=None,
):
    """
    Storm motion between two gridded radar volumes.

    The motion is what is needed to move radar data observed at different
    times to a common analysis time (Gal-Chen 1982 [1]_). The displacement of
    the echo pattern between the two times is found by cross-correlation of
    the two echo patterns, the principle of radar echo tracking by correlation
    introduced by Rinehart and Garvey (1978) [2]_. Their equations and
    parameters were not checked here; the FFT implementation and everything
    below are radarx's own. Before correlating, each field is

    * restricted to the area observed at both times (``observed``), so that
      the fixed edge of the radar coverage does not pin the result to zero;
    * clipped at a reflectivity ``floor`` (no echo counts as zero);
    * tapered with a 2-D Hann window, and high-pass filtered with a Gaussian of
      width ``highpass``, which removes the broad background that carries
      little motion information.

    The correlation peak is searched within ``max_speed`` and refined to a
    fraction of a grid cell with a parabolic fit along each axis. Because the
    taper window stays fixed while the echoes move, a single correlation is
    biased toward zero motion; the later field is therefore shifted back by
    the estimate and the small residual is estimated again (``iterations``
    times), which removes the bias. The normalised cross-correlation at the
    final peak must reach ``min_quality``; otherwise the motion is NaN.

    With ``tile`` set, the estimate is repeated on overlapping tiles, giving a
    spatially varying motion. This is a motivation shared with Shapiro et al.
    (2010) [3]_, [4]_, who treat spatially variable advection by a different,
    variational method; radarx does not implement their method. Each tile
    starts from the domain-wide motion and searches within half of it (plus
    two cells) around it, a coarse-to-fine scheme that keeps small tiles from
    locking onto spurious peaks. Tiles with little echo (less than 5 % of the
    tile at either time) or a weak correlation take the domain-wide motion;
    the tile vectors are then smoothed and interpolated bilinearly to every
    grid cell.

    The defaults (``floor`` 5 dBZ, ``highpass`` 10 km, ``max_speed`` 60 m/s,
    ``min_quality`` 0.3, ``iterations`` 3, ``overlap`` 0.5, ``smooth`` 1 tile)
    and the 5 % echo threshold are radarx choices, not taken from the cited
    papers. The sub-cell refinement is the standard three-point parabola
    through the correlation peak.

    Parameters
    ----------
    obj_t0, obj_t1 : xarray.Dataset or xarray.DataArray
        Gridded volumes (or single fields) at the earlier and later time, with
        evenly spaced ``x`` and ``y`` coordinates in metres, e.g. from
        :func:`radarx.grid.grid_radar`. Extra dimensions (``z``) are reduced
        to the column maximum.
    field : str, optional
        Field to track for Dataset input. Default: the first reflectivity
        field found (``DBZH``, ``reflectivity``, ...).
    dt : float or timedelta, optional
        Time between the two volumes in seconds. Default: the difference of
        their ``time`` values.
    floor : float, optional
        Values below this (dBZ) are treated as no echo. Default 5.
    highpass : float or None, optional
        Width (standard deviation, metres) of the Gaussian whose smoothed
        field is removed before correlating. ``None`` only removes the mean.
        Default 10 km.
    max_speed : float, optional
        Largest motion considered, in m/s. Default 60.
    min_quality : float, optional
        Minimum normalised cross-correlation (-1 to 1) at the peak.
        Default 0.3.
    iterations : int, optional
        Refinement passes after the first estimate. Default 3.
    tile : float, optional
        Tile size in metres for a spatially varying motion. Default: one
        motion vector for the whole domain.
    overlap : float, optional
        Fractional overlap of neighbouring tiles. Default 0.5.
    smooth : float, optional
        Standard deviation (in tiles) of the Gaussian smoothing of the tile
        vectors. Default 1.
    observed : xarray.DataArray of bool, optional
        Cells observed at both times. Default: cells where at least one of
        the two fields is not NaN. Pass it when NaN means "not observed"
        rather than "no echo", e.g. for two different radars.
    x, y : str, optional
        Names of the horizontal coordinates. Default ``"x"`` and ``"y"``.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation of the interpolation used by the refinement passes.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.

    Returns
    -------
    xarray.Dataset
        ``u`` (eastward, along ``x``) and ``v`` (northward, along ``y``) storm
        motion in m/s, scalars or on ``(y, x)`` for tiled motion, and the
        ``quality`` of the domain-wide correlation peak.

    Raises
    ------
    ValueError
        If the time step is unknown or zero, or the grid is not regular.

    See Also
    --------
    advect, interpolate_time

    References
    ----------
    .. [1] Gal-Chen, T., 1982: Errors in fixed and moving frame of references:
       Applications for conventional and Doppler radar analysis. *J. Atmos.
       Sci.*, **39**, 2279-2300,
       https://doi.org/10.1175/1520-0469(1982)039<2279:EIFAMF>2.0.CO;2
    .. [2] Rinehart, R. E., and E. T. Garvey, 1978: Three-dimensional storm
       motion detection by conventional weather radar. *Nature*, **273**,
       287-289, https://doi.org/10.1038/273287a0
    .. [3] Shapiro, A., K. M. Willingham, and C. K. Potvin, 2010: Spatially
       variable advection correction of radar data. Part I: Theoretical
       considerations. *J. Atmos. Sci.*, **67**, 3445-3456,
       https://doi.org/10.1175/2010JAS3465.1
    .. [4] Shapiro, A., K. M. Willingham, and C. K. Potvin, 2010: Spatially
       variable advection correction of radar data. Part II: Test results.
       *J. Atmos. Sci.*, **67**, 3457-3470,
       https://doi.org/10.1175/2010JAS3466.1

    Examples
    --------
    >>> motion = radarx.retrieve.estimate_motion(grid0, grid1)  # doctest: +SKIP
    >>> float(motion.u), float(motion.v)  # doctest: +SKIP
    """
    use_compiled = _use_compiled(engine)
    dt_s = _pair_seconds(obj_t0, obj_t1, dt)
    da0 = _field(obj_t0, field)
    da1 = _field(obj_t1, field)
    dx, dy = _step(da0, x), _step(da0, y)
    a, b = _plane(da0, x, y), _plane(da1, x, y)
    if a.shape != b.shape:
        raise ValueError("the two fields must be on the same grid")
    if observed is None:
        mask = np.isfinite(a) | np.isfinite(b)
    else:
        mask = np.asarray(observed.transpose(y, x).values, dtype=bool)
    ny, nx = a.shape
    max_shift = (
        int(min(np.ceil(max_speed * abs(dt_s) / abs(dy)) + 1, ny - 2)),
        int(min(np.ceil(max_speed * abs(dt_s) / abs(dx)) + 1, nx - 2)),
    )
    sigma = None if highpass is None else (highpass / abs(dy), highpass / abs(dx))
    options = {
        "floor": floor,
        "sigma": sigma,
        "iterations": int(iterations),
        "kernel": partial(_interpolate, use_compiled=use_compiled, n_threads=n_threads),
    }
    whole = (slice(0, ny), slice(0, nx))
    rows, cols, quality = _track(
        a, b, mask, max_shift=max_shift, rows=whole[0], cols=whole[1], **options
    )
    if not quality >= min_quality:
        rows = cols = np.nan
    to_u, to_v = dx / dt_s, dy / dt_s  # grid cells -> m/s
    u_val, v_val = cols * to_u, rows * to_v
    dims, coords = (), {}
    if tile is not None:
        size = (max(4, round(tile / abs(dy))), max(4, round(tile / abs(dx))))
        trows, tcols = _motion_field(
            a, b, mask, size, overlap, min_quality, smooth, (rows, cols), options
        )
        u_val, v_val = tcols * to_u, trows * to_v
        dims = (y, x)
        coords = {
            name: c
            for name, c in da0.coords.items()
            if set(c.dims) <= {x, y} and c.ndim > 0
        }
    attrs = {"units": "m s-1"}
    if np.all(np.isnan(u_val)):
        warnings.warn(
            "no reliable motion found (correlation peak too weak or at the "
            "edge of the search window)",
            RuntimeWarning,
            stacklevel=2,
        )
    out = xr.Dataset(
        {
            "u": (dims, u_val, dict(attrs, long_name="eastward storm motion")),
            "v": (dims, v_val, dict(attrs, long_name="northward storm motion")),
            "quality": (
                (),
                quality,
                {
                    "units": "1",
                    "long_name": "normalised cross-correlation at the peak",
                },
            ),
        },
        coords=coords,
        attrs={"dt": dt_s, "tracked_field": str(da0.name)},
    )
    return out


# ---------------------------------------------------------------------------
# semi-Lagrangian advection
# ---------------------------------------------------------------------------


def _keys(t):
    """
    Keys (1981) cubic convolution weights (a = -1/2) for offsets ``t``.

    Keys, R., 1981, IEEE Trans. Acoust. Speech Signal Process. 29, 1153-1160,
    https://doi.org/10.1109/TASSP.1981.1163711. The piecewise cubic kernel with
    the free parameter a = -1/2 is used; the paper (not on disk) was not
    checked for its equation numbers or for a recommended value of a.
    """
    a = -0.5
    t1, t3, t4 = 1.0 + t, 1.0 - t, 2.0 - t
    return (
        ((a * t1 - 5.0 * a) * t1 + 8.0 * a) * t1 - 4.0 * a,
        ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0,
        ((a + 2.0) * t3 - (a + 3.0)) * t3 * t3 + 1.0,
        ((a * t4 - 5.0 * a) * t4 + 8.0 * a) * t4 - 4.0 * a,
    )


def _advect_plane_set(data, src_row, src_col, order, min_weight):
    """NumPy version for one set of departure points: (nk, oy, ox)."""
    nk, ny, nx = data.shape
    ok_pt = (
        np.isfinite(src_row)
        & np.isfinite(src_col)
        & (src_row > -2.0)
        & (src_col > -2.0)
        & (src_row < ny + 1.0)
        & (src_col < nx + 1.0)
    )
    r = np.where(ok_pt, src_row, 0.0)
    c = np.where(ok_pt, src_col, 0.0)
    fi, fj = np.floor(r), np.floor(c)
    i0, j0 = fi.astype(np.int64), fj.astype(np.int64)
    fr, fc = r - fi, c - fj
    shape = (nk,) + r.shape

    def gather(di, dj):
        ii, jj = i0 + di, j0 + dj
        inside = (ii >= 0) & (ii < ny) & (jj >= 0) & (jj < nx)
        vals = data[:, np.clip(ii, 0, ny - 1), np.clip(jj, 0, nx - 1)]
        vals = vals.astype(np.float64)
        return vals, inside & np.isfinite(vals)

    weights = ((1 - fr) * (1 - fc), (1 - fr) * fc, fr * (1 - fc), fr * fc)
    num = np.zeros(shape)
    den = np.zeros(shape)
    lo = np.full(shape, np.inf)
    hi = np.full(shape, -np.inf)
    for w, (di, dj) in zip(weights, ((0, 0), (0, 1), (1, 0), (1, 1))):
        vals, ok = gather(di, dj)
        num += np.where(ok, w * vals, 0.0)
        den += np.where(ok, w, 0.0)
        lo = np.where(ok, np.minimum(lo, vals), lo)
        hi = np.where(ok, np.maximum(hi, vals), hi)
    defined = (den > 0.0) & (den >= min_weight)
    with np.errstate(invalid="ignore", divide="ignore"):
        res = np.where(defined, num / np.where(defined, den, 1.0), np.nan)
    if order == 3:
        wy, wx = _keys(fr), _keys(fc)
        acc = np.zeros(shape)
        complete = np.ones(shape, dtype=bool)
        for a in range(4):
            row = np.zeros(shape)
            for b in range(4):
                vals, ok = gather(a - 1, b - 1)
                used = (wy[a] != 0.0) & (wx[b] != 0.0)
                complete &= ok | ~used
                row += np.where((wx[b] != 0.0) & ok, wx[b] * vals, 0.0)
            acc += np.where(wy[a] != 0.0, wy[a] * row, 0.0)
        res = np.where(complete & defined, np.minimum(hi, np.maximum(lo, acc)), res)
    return np.where(ok_pt, res, np.nan).astype(data.dtype)


def _advect_numpy(data, src_row, src_col, order, min_weight):
    """NumPy implementation of the compiled kernel (same results)."""
    return np.stack(
        [
            _advect_plane_set(data, r, c, order, min_weight)
            for r, c in zip(src_row, src_col)
        ]
    )


def _interpolate(data, src_row, src_col, order, min_weight, use_compiled, n_threads):
    """
    Planes ``data`` (nk, ny, nx) at departure points (ng, oy, ox).

    Returns an array (ng, nk, oy, ox) of the dtype of ``data`` (float32 or
    float64). All planes and all sets of departure points go to the kernel in
    a single call.
    """
    dtype = np.float32 if data.dtype == np.float32 else np.float64
    data = np.ascontiguousarray(data, dtype=dtype)
    src_row = np.ascontiguousarray(src_row, dtype=np.float64)
    src_col = np.ascontiguousarray(src_col, dtype=np.float64)
    if use_compiled:
        return _advection.advect(
            data, src_row, src_col, order, float(min_weight), int(n_threads or 0)
        )
    return _advect_numpy(data, src_row, src_col, order, float(min_weight))


def _motion_arrays(u, v, x, y, ny, nx):
    """``u`` and ``v`` as floats (uniform) or (ny, nx) arrays."""
    if isinstance(u, xr.Dataset):
        if v is not None:
            raise ValueError("pass either a motion Dataset or u and v")
        u, v = u["u"], u["v"]
    if v is None:
        raise ValueError("v is required")
    out = []
    for m in (u, v):
        if isinstance(m, xr.DataArray) and m.ndim > 0:
            if set(m.dims) != {x, y} or m.sizes[y] != ny or m.sizes[x] != nx:
                raise ValueError(f"spatially varying motion must be on ({y}, {x})")
            m = np.asarray(m.transpose(y, x).values, dtype=np.float64)
        else:
            m = float(m)
        if not np.all(np.isfinite(m)):
            raise ValueError("the motion contains NaN")
        out.append(m)
    return out


def _departure(u, v, dts, dx, dy, ny, nx, use_compiled, n_threads):
    """
    Departure points (fractional rows and columns) for every time step.

    Returns two arrays (len(dts), ny, nx). For a uniform motion the
    trajectories are straight lines. For a spatially varying motion the
    displacement is found with the implicit midpoint iteration
    ``alpha = dt * V(x - alpha / 2)`` of two-time-level semi-Lagrangian
    schemes (Staniforth and Côté 1991, Mon. Wea. Rev. 119, 2206-2223; the
    exact equation was not checked), for all time steps at once. Three
    iterations are a radarx choice.
    """
    dts = np.asarray(dts, dtype=np.float64)[:, None, None]
    rows = np.arange(ny, dtype=np.float64)[None, :, None]
    cols = np.arange(nx, dtype=np.float64)[None, None, :]
    shape = (dts.shape[0], ny, nx)
    if np.ndim(u) == 0 and np.ndim(v) == 0:
        return (
            np.broadcast_to(rows - dts * v / dy, shape),
            np.broadcast_to(cols - dts * u / dx, shape),
        )
    speed = np.stack(
        [np.broadcast_to(v / dy, (ny, nx)), np.broadcast_to(u / dx, (ny, nx))]
    )  # cells per second
    alpha_r, alpha_c = dts * speed[0], dts * speed[1]
    for _ in range(3):
        mid_r = np.clip(rows - 0.5 * alpha_r, 0, ny - 1)
        mid_c = np.clip(cols - 0.5 * alpha_c, 0, nx - 1)
        mid = _interpolate(speed, mid_r, mid_c, 1, 0.0, use_compiled, n_threads)
        alpha_r, alpha_c = dts * mid[:, 0], dts * mid[:, 1]
    return rows - alpha_r, cols - alpha_c


def _advect_fields(
    arrays, src_row, src_col, order, min_weight, use_compiled, n_threads
):
    """
    Advect a list of (..., ny, nx) arrays with shared departure points.

    Arrays of the same dtype are stacked into one kernel call. Returns one
    array (ng, ...) per input.
    """
    results = [None] * len(arrays)
    groups = {}
    for n, arr in enumerate(arrays):
        dtype = np.float32 if arr.dtype == np.float32 else np.float64
        groups.setdefault(dtype, []).append(n)
    for dtype, members in groups.items():
        planes = [arrays[n].reshape(-1, *arrays[n].shape[-2:]) for n in members]
        stack = planes[0] if len(planes) == 1 else np.concatenate(planes)
        moved = _interpolate(
            stack.astype(dtype, copy=False),
            src_row,
            src_col,
            order,
            min_weight,
            use_compiled,
            n_threads,
        )
        start = 0
        for n, p in zip(members, planes):
            part = moved[:, start : start + p.shape[0]]
            start += p.shape[0]
            shape = (moved.shape[0],) + arrays[n].shape
            results[n] = part.reshape(shape).astype(arrays[n].dtype, copy=False)
    return results


def _fields(obj, x, y):
    """Advected variables of ``obj`` as {name: DataArray on (..., y, x)}."""
    if isinstance(obj, xr.DataArray):
        das = {obj.name: obj}
    else:
        das = {name: obj[name] for name in _gridded_vars(obj, x, y)}
    out = {}
    for name, da in das.items():
        other = [d for d in da.dims if d not in (x, y)]
        out[name] = da.transpose(*other, y, x)
    return out


def _set_time(obj, time):
    """Replace the scalar ``time`` of ``obj`` (coordinate or variable)."""
    old = obj["time"]
    new = xr.DataArray(np.datetime64(time, "ns"), attrs=old.attrs)
    if "time" in obj.coords:
        return obj.assign_coords(time=new)
    return obj.assign(time=new)


def advect(
    obj,
    u,
    v=None,
    dt=None,
    *,
    time=None,
    method="linear",
    min_weight=0.5,
    x="x",
    y="y",
    engine="auto",
    n_threads=None,
):
    """
    Move gridded fields along the storm motion by a time step.

    Semi-Lagrangian advection: every output cell takes the value found at its
    departure point ``x - u dt`` (Staniforth and Côté 1991 [2]_). Fields
    observed at one time are thereby moved to another, the frame-of-reference
    correction of Gal-Chen (1982) [1]_. The value at the departure point is
    interpolated bilinearly, or with cubic convolution (Keys 1981 [3]_)
    clipped to the four nearest values so that no new extremes appear.

    The cubic option uses the kernel of Keys (1981) with a = -1/2 (the
    value of the free parameter used here; neither the equation numbers nor a
    recommended value were checked against the paper). The clipping to the four nearest values, the validity mask and
    the default ``min_weight`` of 0.5 are radarx choices.

    Missing data are handled with an advected validity mask: a cell is
    defined only if valid neighbours carry at least ``min_weight`` of the
    bilinear weight, and its value is then their weighted mean. Data never
    spread into empty regions by more than half a cell, and cells whose
    departure point lies outside the grid become NaN. Cubic interpolation is
    used only where all of its 16 neighbours are valid.

    For a spatially varying motion the departure points are found with the
    implicit midpoint iteration of two-time-level semi-Lagrangian schemes
    (Staniforth and Côté 1991 [2]_, who review such schemes; the iteration
    count of 3 used here is a radarx choice). Every level of a 3-D field moves
    with the same horizontal motion, the motion is assumed steady between the
    two times (Gal-Chen 1982 [1]_) and vertical motion is not represented.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        Gridded fields with evenly spaced ``x`` and ``y`` coordinates in
        metres. In a Dataset every floating-point variable on ``(y, x)`` is
        advected; other variables and all coordinates are kept.
    u : float, xarray.DataArray or xarray.Dataset
        Eastward motion (m/s), scalar or on ``(y, x)``; or the Dataset from
        :func:`estimate_motion` (then ``v`` is not given).
    v : float or xarray.DataArray, optional
        Northward motion (m/s), scalar or on ``(y, x)``.
    dt : float or timedelta, optional
        Time step in seconds; negative moves fields back in time.
    time : datetime-like, optional
        Target time instead of ``dt``; needs a scalar ``time`` in ``obj``.
    method : {"linear", "cubic"}, optional
        Interpolation at the departure points. Default ``"linear"``.
    min_weight : float, optional
        Minimum share of the bilinear weight on valid neighbours for a cell to
        be defined. Default 0.5.
    x, y : str, optional
        Names of the horizontal coordinates. Default ``"x"`` and ``"y"``.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled
        kernel and falls back to NumPy.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The advected fields with the same coordinates, dtypes and attributes;
        a scalar ``time`` is moved by ``dt``.

    Raises
    ------
    ValueError
        If neither or both of ``dt`` and ``time`` are given, the motion is
        NaN, or the grid is not regular.
    ImportError
        If ``engine="compiled"`` and the compiled kernel is not available.

    See Also
    --------
    estimate_motion, interpolate_time

    References
    ----------
    .. [1] Gal-Chen, T., 1982: Errors in fixed and moving frame of references:
       Applications for conventional and Doppler radar analysis. *J. Atmos.
       Sci.*, **39**, 2279-2300,
       https://doi.org/10.1175/1520-0469(1982)039<2279:EIFAMF>2.0.CO;2
    .. [2] Staniforth, A., and J. Cote, 1991: Semi-Lagrangian integration
       schemes for atmospheric models - A review. *Mon. Wea. Rev.*, **119**,
       2206-2223,
       https://doi.org/10.1175/1520-0493(1991)119<2206:SLISFA>2.0.CO;2
    .. [3] Keys, R., 1981: Cubic convolution interpolation for digital image
       processing. *IEEE Trans. Acoust. Speech Signal Process.*, **29**,
       1153-1160, https://doi.org/10.1109/TASSP.1981.1163711

    Examples
    --------
    >>> motion = radarx.retrieve.estimate_motion(grid0, grid1)  # doctest: +SKIP
    >>> moved = radarx.retrieve.advect(grid0, motion, dt=120.0)  # doctest: +SKIP
    """
    use_compiled = _use_compiled(engine)
    order = _order(method)
    t0 = _time_of(obj)
    if (dt is None) == (time is None):
        raise ValueError("give exactly one of dt and time")
    if time is not None:
        if t0 is None:
            raise ValueError("obj has no scalar 'time': pass dt=")
        dt_s = float((np.datetime64(time, "ns") - t0) / np.timedelta64(1, "s"))
    else:
        dt_s = _seconds(dt)
    dx, dy = _step(obj, x), _step(obj, y)
    ny, nx = obj.sizes[y], obj.sizes[x]
    u, v = _motion_arrays(u, v, x, y, ny, nx)
    src_row, src_col = _departure(u, v, [dt_s], dx, dy, ny, nx, use_compiled, n_threads)
    fields = _fields(obj, x, y)
    moved = _advect_fields(
        [da.values for da in fields.values()],
        src_row,
        src_col,
        order,
        min_weight,
        use_compiled,
        n_threads,
    )
    if isinstance(obj, xr.DataArray):
        da = next(iter(fields.values()))
        out = da.copy(data=moved[0][0]).transpose(*obj.dims)
    else:
        out = obj.copy()
        for (name, da), arr in zip(fields.items(), moved):
            out[name] = da.copy(data=arr[0]).transpose(*obj[name].dims)
    if t0 is not None:
        out = _set_time(out, t0 + np.timedelta64(round(dt_s * 1e9), "ns"))
    return out


# ---------------------------------------------------------------------------
# time interpolation
# ---------------------------------------------------------------------------


def interpolate_time(
    obj_t0,
    obj_t1,
    times,
    motion=None,
    *,
    field=None,
    method="linear",
    min_weight=0.5,
    x="x",
    y="y",
    engine="auto",
    n_threads=None,
    **motion_kwargs,
):
    """
    Advection-corrected time interpolation between two gridded volumes.

    Following the moving frame of reference of Gal-Chen (1982) [1]_, at every
    target time ``t`` the earlier volume is advected forward by ``t - t0`` and
    the later one backward by ``t1 - t``, both along the storm motion, and the
    two are blended with weights ``1 - f`` and ``f``, where
    ``f = (t - t0) / (t1 - t0)``. Echoes therefore move smoothly between the
    two observations instead of fading out in one place and in at another, as
    with plain linear interpolation. Where only one of the two advected
    fields is defined, it is used alone. The forward-backward blend is
    radarx's own construction. The pysteps library (Pulkkinen et al. 2019
    [2]_) provides advection-based extrapolation for precipitation nowcasting;
    whether and how it blends two volumes this way was not checked against
    that paper.

    Parameters
    ----------
    obj_t0, obj_t1 : xarray.Dataset or xarray.DataArray
        Gridded volumes on the same grid, each with a scalar ``time``.
    times : datetime-like or array-like of datetime-like
        Target times between ``obj_t0.time`` and ``obj_t1.time``.
    motion : xarray.Dataset, optional
        Storm motion from :func:`estimate_motion`. Default: estimated from
        the two volumes.
    field : str, optional
        Field used to estimate the motion (see :func:`estimate_motion`).
    method, min_weight, x, y, engine, n_threads
        See :func:`advect`.
    **motion_kwargs
        Passed to :func:`estimate_motion` when ``motion`` is not given.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The advected fields with a new ``time`` dimension.

    Raises
    ------
    ValueError
        If a volume has no time, a target time lies outside the two volumes,
        or no motion could be estimated.

    See Also
    --------
    estimate_motion, advect

    References
    ----------
    .. [1] Gal-Chen, T., 1982: Errors in fixed and moving frame of references:
       Applications for conventional and Doppler radar analysis. *J. Atmos.
       Sci.*, **39**, 2279-2300,
       https://doi.org/10.1175/1520-0469(1982)039<2279:EIFAMF>2.0.CO;2
    .. [2] Pulkkinen, S., D. Nerini, A. A. Perez Hortal, C. Velasco-Forero,
       A. Seed, U. Germann, and L. Foresti, 2019: Pysteps: an open-source
       Python library for probabilistic precipitation nowcasting (v1.0).
       *Geosci. Model Dev.*, **12**, 4185-4219,
       https://doi.org/10.5194/gmd-12-4185-2019

    Examples
    --------
    >>> import pandas as pd  # doctest: +SKIP
    >>> times = pd.date_range(grid0.time.values, grid1.time.values, freq="60s")
    >>> frames = radarx.retrieve.interpolate_time(grid0, grid1, times)  # doctest: +SKIP
    """
    use_compiled = _use_compiled(engine)
    order = _order(method)
    t0, t1, times = _check_times(obj_t0, obj_t1, times)
    if motion is None:
        motion = estimate_motion(
            obj_t0,
            obj_t1,
            field,
            x=x,
            y=y,
            engine=engine,
            n_threads=n_threads,
            **motion_kwargs,
        )
    dx, dy = _step(obj_t0, x), _step(obj_t0, y)
    ny, nx = obj_t0.sizes[y], obj_t0.sizes[x]
    if obj_t1.sizes[y] != ny or obj_t1.sizes[x] != nx:
        raise ValueError("the two volumes must be on the same grid")
    try:
        u, v = _motion_arrays(motion, None, x, y, ny, nx)
    except ValueError as err:
        raise ValueError(f"no usable motion ({err}): pass motion=") from err
    span = (t1 - t0) / np.timedelta64(1, "s")
    frac = (times - t0) / np.timedelta64(1, "s") / span
    fields0, fields1 = _fields(obj_t0, x, y), _fields(obj_t1, x, y)
    if isinstance(obj_t0, xr.Dataset) and set(fields0) - set(fields1):
        raise ValueError(f"obj_t1 lacks {sorted(set(fields0) - set(fields1))}")

    # every target time of each volume in one kernel call
    moved = []
    for fields, dts in ((fields0, frac * span), (fields1, -(1.0 - frac) * span)):
        src_row, src_col = _departure(
            u, v, dts, dx, dy, ny, nx, use_compiled, n_threads
        )
        arrays = [da.values for da in fields.values()]
        options = (order, min_weight, use_compiled, n_threads)
        moved.append(
            dict(zip(fields, _advect_fields(arrays, src_row, src_col, *options)))
        )
    if isinstance(obj_t0, xr.DataArray):  # names of the two may differ
        moved[1] = dict(zip(moved[0], moved[1].values()))
    time_coord = xr.DataArray(
        times, dims="time", attrs=dict(obj_t0["time"].attrs), name="time"
    )
    blended = {
        name: _blend(moved[0][name], moved[1][name], frac, da)
        for name, da in fields0.items()
    }
    if isinstance(obj_t0, xr.DataArray):
        da = next(iter(blended.values()))
        return da.transpose("time", *obj_t0.dims).assign_coords(time=time_coord)
    out = obj_t0.drop_vars("time", errors="ignore")
    for name, da in blended.items():
        out[name] = da.transpose("time", *obj_t0[name].dims)
    return out.assign_coords(time=time_coord)


def _order(method):
    """Interpolation order (1 or 3) of ``method``."""
    if method not in _ORDERS:
        raise ValueError(f"method must be 'linear' or 'cubic', not {method!r}")
    return _ORDERS[method]


def _check_times(obj_t0, obj_t1, times):
    """Volume times and target times (datetime64[ns]) between them."""
    t0, t1 = _time_of(obj_t0), _time_of(obj_t1)
    if t0 is None or t1 is None:
        raise ValueError("both volumes need a scalar 'time'")
    if t1 <= t0:
        raise ValueError("obj_t1 must be later than obj_t0")
    times = np.atleast_1d(np.asarray(times, dtype="datetime64[ns]"))
    if times.ndim != 1 or np.any(times < t0) or np.any(times > t1):
        raise ValueError("target times must lie between the two volumes")
    return t0, t1, times


def _blend(a, b, frac, template):
    """
    ``(1 - f) a + f b`` where both are defined, else the defined one.

    ``a`` and ``b`` are (time, ...) arrays; the result is a DataArray with the
    dimensions, coordinates and attributes of ``template`` after ``time``.
    """
    f = frac.reshape((-1,) + (1,) * (a.ndim - 1)).astype(a.dtype)
    out = np.where(np.isfinite(a), a, b)
    np.copyto(out, (1 - f) * a + f * b, where=np.isfinite(a) & np.isfinite(b))
    return xr.DataArray(
        out,
        dims=("time",) + template.dims,
        coords={k: c for k, c in template.coords.items() if k != "time"},
        attrs=template.attrs,
        name=template.name,
    )
