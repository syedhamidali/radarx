# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Probabilistic Nowcasting
========================

Forecast a gridded radar reflectivity field minutes to hours ahead, as one
deterministic forecast or as an ensemble.

:func:`nowcast` builds on :func:`radarx.retrieve.estimate_motion` (or
:func:`radarx.retrieve.optical_flow`) and on the semi-Lagrangian
:func:`radarx.retrieve.advect`. Three methods are available:

``"extrapolation"``
    Lagrangian persistence: the last field is advected along the motion
    (Germann and Zawadzki 2002).
``"sprog"``
    Spectral prognosis (Seed 2003): the field is split into a cascade of
    spatial scales, every scale evolves with its own autoregressive model, so
    the small scales, which are unpredictable, fade out first.
``"steps"``
    Short-term ensemble prediction system (Bowler et al. 2006): the same
    cascade with a spatially correlated random innovation at every scale,
    refilling the small scales that S-PROG loses, and optionally a random
    perturbation of the motion, for every member.

:func:`nowcast_model` runs a learned network (an ONNX model, through
:mod:`radarx.ml`) on the last frames in place of these methods.
:func:`exceedance_probability` turns an ensemble into the probability of a
threshold being exceeded. The scores of a forecast are in
:mod:`radarx.retrieve.nowcast_verification`.

Provenance. The decomposition into scales with Gaussian band filters, the
autoregressive model of each scale with Yule-Walker parameters, the noise
filtered with the amplitude spectrum of the observed field, the rain mask,
the probability matching and the perturbation of the motion follow the
descriptions of Seed (2003), Bowler et al. (2006) and Pulkkinen et al.
(2019). The implementation is radarx's own, not derived from the pysteps
code: the cascade, the autoregressive recursion and the innovation are
evaluated in the Fourier domain by a compiled kernel (they are linear and act
on each spectral coefficient alone), and the Eulerian forecast is the
Lagrangian forecast advected once with :func:`advect`, in one step for every
lead time. The band filters (central wavenumbers, widths), the limits on the
autoregressive correlations and all defaults are radarx choices. Equation
numbers of the cited papers are not checked unless a docstring gives one.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = ["nowcast", "nowcast_model", "exceedance_probability"]

__doc__ = __doc__.format("\n   ".join(__all__))

import warnings

import numpy as np
import xarray as xr

from .._dtypes import FIELD_DTYPE
from .._provenance import provenance
from .._registry import accessor_method
from .._sweepwise import dask_support
from ._nowcast_numpy import HAS_COMPILED_KERNEL as _HAS_AR_KERNEL
from ._nowcast_numpy import ar_step, rank_match
from .advection import (
    _departure,
    _field,
    _interpolate,
    _motion_arrays,
    _plane,
    _seconds,
    _step,
    _time_of,
    _use_compiled,
    estimate_motion,
)

HAS_COMPILED_KERNEL = _HAS_AR_KERNEL

_METHODS = ("extrapolation", "sprog", "steps")
# memory of the cascade states of one batch of members (bytes)
_STATE_BYTES = 512e6
# limits of the lag-1 correlation of a level (radarx choice)
_RHO_MAX = 0.999
_RHO_MIN = 0.001


# ---------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------


def _frames(obj):
    """
    The history as a list of objects with one time each.

    ``obj`` is a sequence of gridded volumes or one object with a ``time``
    dimension.
    """
    if isinstance(obj, (xr.Dataset, xr.DataArray)):
        if "time" not in obj.dims:
            raise ValueError("obj needs a time dimension or must be a list of volumes")
        return [obj.isel(time=k) for k in range(obj.sizes["time"])]
    frames = list(obj)
    if not all(isinstance(f, (xr.Dataset, xr.DataArray)) for f in frames):
        raise TypeError("obj must be a list of Datasets or DataArrays")
    return frames


def _history(obj, field, dt, x, y):
    """
    Frames (list of objects), planes (n, ny, nx), last time, mean step and steps.

    The steps are the intervals (s) between the frames; they may differ by up
    to 10 % from their mean (the interval between radar volumes does).
    """
    frames = _frames(obj)
    if len(frames) < 2:
        raise ValueError("at least two frames are needed")
    planes = [_plane(_field(f, field), x, y) for f in frames]
    if any(p.shape != planes[0].shape for p in planes):
        raise ValueError("the frames must be on the same grid")
    times = [_time_of(f) for f in frames]
    if dt is None:
        if any(t is None for t in times):
            raise ValueError("the frames have no scalar 'time': pass dt=")
        steps = np.diff(np.array(times, dtype="datetime64[ns]")) / np.timedelta64(
            1, "s"
        )
        dt_s = float(np.mean(steps))
        if np.any(steps <= 0) or not np.allclose(steps, dt_s, rtol=0.1):
            raise ValueError("the frames must be evenly spaced in time: pass dt=")
    else:
        dt_s = _seconds(dt)
        if not dt_s > 0:
            raise ValueError("dt must be positive")
        steps = np.full(len(frames) - 1, dt_s)
    return frames, np.stack(planes), times[-1], dt_s, steps


def _check_options(
    method, n_steps, n_members, ar_order, n_levels, mask_radius, mask_growth
):
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, not {method!r}")
    if int(n_steps) != n_steps or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    if int(n_members) != n_members or n_members < 1:
        raise ValueError("n_members must be a positive integer")
    if ar_order not in (None, 1, 2):
        raise ValueError("ar_order must be 1 or 2")
    if int(n_levels) != n_levels or n_levels < 3:
        raise ValueError("n_levels must be an integer of at least 3")
    if mask_radius is not None and mask_radius < 0:
        raise ValueError("mask_radius must not be negative")
    if mask_growth < 0:
        raise ValueError("mask_growth must not be negative")


def _perturbation(spec):
    """Validate ``motion_perturbation`` and return the two (a, b, c) triples."""
    if spec is None:
        return None
    try:
        par, perp = spec["parallel"], spec["perpendicular"]
        par, perp = tuple(map(float, par)), tuple(map(float, perp))
    except (KeyError, TypeError, ValueError) as err:
        raise ValueError(
            "motion_perturbation must be {'parallel': (a, b, c), "
            "'perpendicular': (a, b, c)}"
        ) from err
    if len(par) != 3 or len(perp) != 3:
        raise ValueError("the parameters of motion_perturbation are (a, b, c)")
    return par, perp


# ---------------------------------------------------------------------------
# cascade
# ---------------------------------------------------------------------------


def _padded_shape(ny, nx, pad):
    """Shape of the padded domain and the offset of the grid inside it."""
    from scipy import fft as sfft

    py, px = int(np.ceil(pad * ny)), int(np.ceil(pad * nx))
    shape = (
        sfft.next_fast_len(ny + 2 * py, real=True),
        sfft.next_fast_len(nx + 2 * px, real=True),
    )
    return shape, ((shape[0] - ny) // 2, (shape[1] - nx) // 2)


def _band_filters(shape, dy, dx, n_levels):
    """
    Gaussian band filters of the cascade on the half spectrum of ``shape``.

    Level 0 holds the largest scales (and, being the only one that is not zero
    there, the mean), the last level the smallest. The central wavenumbers
    of levels 1 to ``n_levels - 1`` follow a geometric series from a
    wavelength of half the domain to 0.6 of the Nyquist wavenumber; a band is
    a Gaussian in the logarithm of the wavenumber whose full width at half
    maximum is the ratio of the series, and the first and the last band are
    flat beyond their centre. The weights are divided by their sum at every
    wavenumber, so the levels add up to the field exactly. All of this is a
    radarx choice. Returns an array (levels, rows, columns).
    """
    ky = np.fft.fftfreq(shape[0], d=abs(dy))
    kx = np.fft.rfftfreq(shape[1], d=abs(dx))
    k = np.hypot(ky[:, None], kx[None, :])
    log_k = np.log(np.maximum(k, 1e-300))
    extent = min(shape[0] * abs(dy), shape[1] * abs(dx))
    k_low = 2.0 / extent
    k_top = 0.6 * min(0.5 / abs(dy), 0.5 / abs(dx))
    n = n_levels - 1  # levels with a centre
    ratio = (k_top / k_low) ** (1.0 / max(n - 1, 1))
    centres = k_low * ratio ** np.arange(n)
    width = 0.5 * np.log(ratio) / np.sqrt(2.0 * np.log(2.0))  # std of ln k
    w = np.empty((n_levels,) + k.shape)
    w[0] = np.where(
        log_k > np.log(centres[0] / ratio),
        np.exp(-0.5 * ((log_k - np.log(centres[0] / ratio)) / width) ** 2),
        1.0,
    )
    for j, c in enumerate(centres, start=1):
        g = np.exp(-0.5 * ((log_k - np.log(c)) / width) ** 2)
        w[j] = np.where(log_k > np.log(c), 1.0, g) if j == n else g
    w /= w.sum(axis=0)
    return w


def _half_weights(nx):
    """Multiplicity of the columns of a half spectrum in the full one."""
    w = np.full(nx // 2 + 1, 2.0)
    w[0] = 1.0
    if nx % 2 == 0:
        w[-1] = 1.0
    return w


def _inner(a, b, weights, n_pixels):
    """Sum over the pixels of the product of two fields, from their spectra."""
    return float(np.sum(weights * np.real(np.conj(a) * b)) / n_pixels**2)


def _ar_parameters(rho1, rho2, order):
    """
    Yule-Walker parameters of the AR(2) (or AR(1)) model of every level.

    ``rho1`` and ``rho2`` are the lag-1 and lag-2 correlations. They are
    limited to ``[0.001, 0.999]`` for lag 1 and to the stationary range
    ``[2 rho1**2 - 1, rho1]`` for lag 2 (radarx limits), so the process is
    stationary. Returns ``phi1``, ``phi2`` and the standard deviation of the
    innovation that keeps the variance of the process at one,
    ``sqrt(1 - phi1 rho1 - phi2 rho2)``.
    """
    r1 = np.clip(rho1, _RHO_MIN, _RHO_MAX)
    if order == 1:
        phi1, phi2, r2 = r1, np.zeros_like(r1), r1**2
    else:
        r2 = np.clip(rho2, 2.0 * r1**2 - 1.0 + 1e-3, r1)
        phi1 = r1 * (1.0 - r2) / (1.0 - r1**2)
        phi2 = (r2 - r1**2) / (1.0 - r1**2)
    gain = np.sqrt(np.maximum(1.0 - phi1 * r1 - phi2 * r2, 0.0))
    return phi1, phi2, gain


# ---------------------------------------------------------------------------
# forecast in the Lagrangian frame
# ---------------------------------------------------------------------------


def _lagrangian_history(planes, u, v, dx, dy, floor, observed, order, steps, kw):
    """
    Frames moved forward to the time of the last one and set to the floor.

    ``planes`` (n, ny, nx) are transformed fields (NaN: no data). Frame
    ``n - 1 - k`` is advected forward by the time (the sum of the last ``k``
    of ``steps``, s) between it and the last one. Cells that carry no data
    after the move are set to ``floor``.
    """
    n, ny, nx = planes.shape
    dts = np.cumsum(np.asarray(steps)[::-1])
    out = np.empty_like(planes)
    out[-1] = planes[-1]
    if n > 1:
        src_r, src_c = _departure(
            u, v, dts, dx, dy, ny, nx, kw["use_compiled"], kw["n_threads"]
        )
        for k in range(1, n):
            moved = _interpolate(
                planes[n - 1 - k][None],
                src_r[k - 1 : k],
                src_c[k - 1 : k],
                order,
                0.5,
                kw["use_compiled"],
                kw["n_threads"],
            )[0, 0]
            out[n - 1 - k] = moved
    out = np.where(np.isfinite(out), out, floor)
    return np.where(observed[None], out, floor)


def _cascade_setup(lagr, w, offset, shape, floor, order):
    """
    Spectra of the levels, their statistics and the autoregressive parameters.

    ``lagr`` (n, ny, nx): the history in the Lagrangian frame, the last frame
    is the latest. Returns the dictionary used by the forecast loop.
    """
    from scipy import fft as sfft

    n, ny, nx = lagr.shape
    oy, ox = offset
    n_pix = shape[0] * shape[1]
    weights = _half_weights(shape[1])[None, :]
    padded = np.full((n,) + shape, floor)
    padded[:, oy : oy + ny, ox : ox + nx] = lagr
    mean = float(padded[-1].mean())  # added back to the forecast
    spectra = []
    for k in range(n):
        spec = sfft.rfft2(padded[k] - mean, workers=-1)
        spec[0, 0] = 0.0
        spectra.append(spec)
    # standard deviation of every level and of the latest frame, per frame
    n_lev = w.shape[0]
    std = np.zeros((n, n_lev))
    for k in range(n):
        for j in range(n_lev):
            a = w[j] * spectra[k]
            std[k, j] = np.sqrt(_inner(a, a, weights, n_pix))
    # lagged correlations of the levels (Lagrangian frame)
    rho = np.zeros((2, n_lev))
    for j in range(n_lev):
        levels = [w[j] * s for s in spectra]
        for lag in (1, 2):
            pairs = [
                (levels[i], levels[i - lag], std[i, j], std[i - lag, j])
                for i in range(lag, n)
            ]
            vals = [
                _inner(a, b, weights, n_pix) / (sa * sb)
                for a, b, sa, sb in pairs
                if sa > 0 and sb > 0
            ]
            rho[lag - 1, j] = np.mean(vals) if vals else 0.0
    phi1, phi2, gain = _ar_parameters(rho[0], rho[1], order)
    return {
        "spectra": spectra,
        "std": std,
        "rho": rho,
        "phi1": phi1,
        "phi2": phi2,
        "gain": gain,
        "mean": mean,
        "weights": weights,
        "n_pixels": n_pix,
    }


def _initial_states(setup, w, order):
    """Normalised spectra of the levels of the latest and the previous frame."""
    std, spectra = setup["std"], setup["spectra"]
    n_lev = w.shape[0]
    shape = spectra[-1].shape
    z1 = np.zeros((n_lev,) + shape, dtype=np.complex128)
    z2 = np.zeros_like(z1)
    for j in range(n_lev):
        if std[-1, j] > 0:
            z1[j] = w[j] * spectra[-1] / std[-1, j]
        if order == 2 and std[-2, j] > 0:
            z2[j] = w[j] * spectra[-2] / std[-2, j]
    return z1, z2


def _noise_amplitude(latest, offset, shape):
    """
    Amplitude spectrum (half spectrum of ``shape``) of the latest frame.

    The mean of the frame is removed and a 2-D Hann window is applied before
    the transform, as in the short-space spectra of Nerini et al. (2017) and
    Pulkkinen et al. (2019); without it the edges of the grid put energy on
    the two axes of the spectrum and the noise shows stripes along the
    axes. The padding is zero.
    """
    from scipy import fft as sfft

    ny, nx = latest.shape
    oy, ox = offset
    window = np.hanning(ny + 2)[1:-1, None] * np.hanning(nx + 2)[None, 1:-1]
    padded = np.zeros(shape)
    padded[oy : oy + ny, ox : ox + nx] = (latest - latest.mean()) * window
    return np.abs(sfft.rfft2(padded, workers=-1))


def _innovation_filter(setup, w, amplitude):
    """
    ``gain_j / s_j * W_j * H`` for every level.

    ``H`` is the ``amplitude`` spectrum of the latest frame (the nonparametric
    filter of Seed et al. 2013, used by Pulkkinen et al. 2019), ``s_j`` the
    standard deviation that white noise filtered with ``W_j H`` would have, so
    that the innovation has unit variance at every level.
    """
    amplitude = amplitude.copy()
    amplitude[0, 0] = 0.0
    wh = w * amplitude[None]
    weights, n_pix = setup["weights"], setup["n_pixels"]
    scale = np.sqrt(
        np.array([np.sum(weights * wh[j] ** 2) / n_pix for j in range(w.shape[0])])
    )
    factor = np.where(scale > 0, setup["gain"] / np.where(scale > 0, scale, 1.0), 0.0)
    return wh * factor[:, None, None]


def _lagrangian_forecast(
    setup, w, wh, z_init, n_steps, members, offset, grid, pshape, kw, noise
):
    """
    Lagrangian forecast of ``members`` (a list of generators) for all leads.

    Returns an array (member, step, ny, nx) of the forecast fields (the
    level sum plus the mean).
    """
    from scipy import fft as sfft

    ny, nx = grid
    oy, ox = offset
    n_lev = w.shape[0]
    z1_0, z2_0 = z_init
    sigma = setup["std"][-1]
    phi1, phi2 = setup["phi1"], setup["phi2"]
    batch = int(max(1, _STATE_BYTES // (2 * n_lev * z1_0[0].size * 16)))
    out = np.empty((len(members), n_steps, ny, nx), dtype=np.float32)
    for start in range(0, len(members), batch):
        gens = members[start : start + batch]
        nm = len(gens)
        z1 = np.ascontiguousarray(np.broadcast_to(z1_0, (nm,) + z1_0.shape).copy())
        z2 = np.ascontiguousarray(np.broadcast_to(z2_0, (nm,) + z2_0.shape).copy())
        for step in range(n_steps):
            e = None
            if noise:
                white = np.stack([g.standard_normal(pshape) for g in gens])
                e = sfft.rfft2(white, workers=-1)
            spec = ar_step(
                z1, z2, phi1, phi2, sigma, wh, e, kw["use_compiled"], kw["n_threads"]
            )
            field = sfft.irfft2(spec, s=pshape, workers=-1)
            out[start : start + nm, step] = (
                field[:, oy : oy + ny, ox : ox + nx] + setup["mean"]
            )
    return out


def _distance_map(echo, dy, dx):
    """Distance (m) of every cell to the nearest echo cell (inf: no echo)."""
    from scipy.ndimage import distance_transform_edt

    if not echo.any():
        return np.full(echo.shape, np.inf)
    return distance_transform_edt(~echo, sampling=(abs(dy), abs(dx)))


def _match(field, observed, ref_sorted, use_compiled, n_threads):
    """
    Give ``field`` the distribution of ``ref_sorted`` (probability matching).

    The values of ``field`` inside ``observed`` are ranked and replaced by the
    sorted reference values of the same rank (Bowler et al. 2006; Pulkkinen et
    al. 2019 use the same cumulative-distribution matching). ``field`` has
    shape (..., ny, nx); a C-contiguous array is changed in place. The result
    is returned.
    """
    field = np.ascontiguousarray(field)
    ny, nx = field.shape[-2:]
    flat = field.reshape(-1, ny * nx)  # a view of the contiguous array
    cells = np.flatnonzero(observed.ravel())
    rank_match(flat, cells, ref_sorted, use_compiled, n_threads)
    return field


# ---------------------------------------------------------------------------
# the nowcast
# ---------------------------------------------------------------------------


def _laplace(rng, size=None):
    """Zero-mean, unit-variance Laplace samples."""
    return rng.laplace(0.0, 1.0 / np.sqrt(2.0), size)


def _perturbed_motion(rng, u, v, par, perp, leads_min):
    """
    Velocity shifts (len(leads), 2) in m/s of one member (Bowler et al. 2006).

    The shift along the mean motion is ``sigma_par(t) e1`` and the one across
    it ``sigma_perp(t) e2`` where ``e1`` and ``e2`` are Laplace samples of
    unit variance and ``sigma(t) = a t**b + c`` (t in minutes).
    """
    mean_u, mean_v = float(np.mean(u)), float(np.mean(v))
    speed = np.hypot(mean_u, mean_v)
    ex = np.array([mean_u, mean_v]) / speed if speed > 0 else np.array([1.0, 0.0])
    ey = np.array([-ex[1], ex[0]])
    e1, e2 = _laplace(rng, 2)
    sig_par = par[0] * leads_min ** par[1] + par[2]
    sig_perp = perp[0] * leads_min ** perp[1] + perp[2]
    return sig_par[:, None] * e1 * ex[None] + sig_perp[:, None] * e2 * ey[None]


def _eulerian(fields, u, v, leads, dx, dy, kw, shifts):
    """
    Advect the Lagrangian forecast (member, step, ny, nx) to the grid.

    ``leads`` are the lead times in s. Without ``shifts`` the members share
    the departure points of a lead time and go to the kernel together; with
    them (per member and lead, in m/s) every member and lead has its own.
    """
    nm, nt, ny, nx = fields.shape
    out = np.empty(fields.shape, dtype=fields.dtype)
    args = (kw["use_compiled"], kw["n_threads"])
    if shifts is None:
        src_r, src_c = _departure(u, v, leads, dx, dy, ny, nx, *args)
        for i in range(nt):
            out[:, i] = _interpolate(
                fields[:, i], src_r[i : i + 1], src_c[i : i + 1], 1, 0.5, *args
            )[0]
        return out
    for m in range(nm):
        for i in range(nt):
            du, dv = shifts[m][i]
            src_r, src_c = _departure(
                u + du, v + dv, leads[i : i + 1], dx, dy, ny, nx, *args
            )
            out[m, i] = _interpolate(fields[m, i][None], src_r, src_c, 1, 0.5, *args)[
                0, 0
            ]
    return out


@dask_support("computes", "it needs the whole history of the forecast.")
@provenance(
    "Cascade nowcast (Lagrangian persistence, S-PROG, STEPS), radarx's own after "
    "Seed (2003) and Bowler et al. (2006)"
)
def nowcast(
    obj,
    n_steps,
    *,
    method="steps",
    field=None,
    dt=None,
    motion=None,
    observed=None,
    n_members=20,
    ar_order=None,
    n_levels=6,
    threshold=10.0,
    zero_value=None,
    mask_radius=20e3,
    mask_growth=0.0,
    probability_matching=True,
    motion_perturbation=None,
    pad=0.25,
    seed=None,
    x="x",
    y="y",
    n_threads=None,
    engine="auto",
):
    """
    Nowcast of a gridded reflectivity field, deterministic or as an ensemble.

    The forecast is made in the frame of the moving precipitation (the
    Lagrangian frame) and then advected to the grid with :func:`advect`:

    1. The motion is estimated from the last two frames with
       :func:`estimate_motion` (or given as ``motion``). Earlier frames are
       advected forward to the time of the last frame.
    2. Reflectivity is used as it is (dBZ is a logarithmic scale, as the dB
       rain rate of Pulkkinen et al. 2019 [3]_); values below ``threshold``
       and missing values are set to ``zero_value``.
    3. The field is decomposed into ``n_levels`` scales with Gaussian band
       filters in the Fourier domain (Seed 2003 [1]_; Bowler et al. 2006
       [2]_). Each level is normalised to unit variance.
    4. Every level follows an autoregressive process of order 2,
       ``Z(t+1) = phi1 Z(t) + phi2 Z(t-1) + gain e(t)``, whose parameters
       come from the lag-1 and lag-2 correlations of the level in the
       Lagrangian frame through the Yule-Walker equations. The innovation has
       the variance ``1 - phi1 rho1 - phi2 rho2`` so that the process keeps
       unit variance (the factor is the square root of that number; the
       published formula for the noise coefficient has no square root, which
       gives a variance of ``(1 - phi1 rho1 - phi2 rho2)**2``; a simulation in
       the tests shows the unit variance). Scales with low correlation, the
       small ones, decay fast.
    5. ``method="sprog"`` stops here (e(t) = 0). With ``"steps"`` the
       innovation ``e(t)`` is white noise filtered by the amplitude spectrum
       of the latest field (Seed et al. 2013 [4]_ as used by Pulkkinen et al.
       2019 [3]_), so that it has the spatial structure of the observed
       field, and split into the same levels. Each member draws its own
       noise.
    6. The forecast of each lead time is limited to the neighbourhood of the
       observed echo (the rain mask of Pulkkinen et al. 2019 [3]_; cells
       farther than ``mask_radius + mask_growth * lead`` from echo above the
       threshold are set to ``zero_value``), and its distribution is matched
       to that of the last observation (``probability_matching``; Bowler et
       al. 2006 [2]_), which restores the variance the smoothing removes.
    7. With ``motion_perturbation`` each member moves with a perturbed
       motion (Bowler et al. 2006 [2]_): a shift along and across the mean
       motion, drawn from a Laplace distribution of unit variance and scaled
       by ``a t**b + c`` (``t`` in minutes).
    8. The fields of all lead times are advected from the Lagrangian frame to
       the grid in one step: the displacement of a lead time is the motion
       times the lead time, with the implicit midpoint iteration of
       :func:`advect` for a spatially varying motion.

    ``method="extrapolation"`` skips steps 3 to 7 and advects the last field.

    The Fourier transforms are done with :mod:`scipy.fft` (multithreaded);
    the autoregressive step and the sum over the levels of all members run in
    a compiled kernel with a NumPy equivalent.

    The central wavenumbers and widths of the band filters, the limits on the
    correlations (lag 1 in 0.001 to 0.999, lag 2 between ``2 rho1**2 - 1`` and
    ``rho1``), the padding of the domain, the defaults of the rain mask and
    the choice to evaluate the cascade in the Fourier domain are radarx's own.
    The perturbation parameters have no default: Bowler et al. (2006) fitted
    them to the advection errors of a radar network, and radarx cannot fit
    them to yours.

    Parameters
    ----------
    obj : list of xarray.Dataset or xarray.DataArray, or xarray.Dataset
        Gridded volumes (or fields) in time order, the last one is the
        analysis time; or one object with a ``time`` dimension. Evenly spaced
        ``x`` and ``y`` coordinates in metres (e.g. from
        :func:`radarx.grid.grid_radar`). Three frames are needed for the
        autoregressive model of order 2, two for order 1. Extra dimensions
        (``z``) are reduced to the column maximum.
    n_steps : int
        Number of lead times. They are ``dt, 2 dt, ..., n_steps dt``.
    method : {"steps", "sprog", "extrapolation"}, optional
        Default ``"steps"``.
    field : str, optional
        Reflectivity field in dBZ. Default: the first reflectivity field found.
    dt : float or timedelta, optional
        Time between the frames in seconds. Default: from their ``time``.
    motion : xarray.Dataset or tuple, optional
        Storm motion (m/s) from :func:`estimate_motion` or
        :func:`optical_flow`, or a tuple ``(u, v)``. Default: estimated from
        the last two frames with :func:`estimate_motion`.
    observed : xarray.DataArray of bool, optional
        Cells that were observed. Default: all cells, so missing values mean
        no echo. Pass the coverage of the radar when missing values mean not
        observed; the forecast is NaN where it would come from outside.
    n_members : int, optional
        Members of the ensemble (``"steps"``). Default 20.
    ar_order : {1, 2}, optional
        Order of the autoregressive model. Default: 2 with three or more
        frames, else 1.
    n_levels : int, optional
        Number of scales of the cascade (at least 3). Default 6.
    threshold : float, optional
        Reflectivity (dBZ) below which there is no echo. Default 10.
    zero_value : float, optional
        Reflectivity assigned to no echo, also in the result. Default:
        ``threshold - 5``.
    mask_radius : float or None, optional
        Distance (m) from the echo beyond which the forecast is set to
        ``zero_value`` at the first lead time. ``None`` switches the mask
        off. Default 20 km.
    mask_growth : float, optional
        Growth of that distance with lead time (m/s). Default 0.
    probability_matching : bool, optional
        Match the distribution of the forecast to the last observation.
        Default True.
    motion_perturbation : dict, optional
        ``{"parallel": (a, b, c), "perpendicular": (a, b, c)}``, the standard
        deviations (m/s) of the shift of the motion along and across it,
        ``a t**b + c`` with ``t`` in minutes. Default: no perturbation.
    pad : float, optional
        Padding of the domain on each side, as a fraction of its size, for
        the Fourier transforms. Default 0.25.
    seed : int, optional
        Seed of the random numbers. A seed gives the same ensemble for any
        ``n_threads`` and engine.
    x, y : str, optional
        Names of the horizontal coordinates. Default ``"x"`` and ``"y"``.
    n_threads : int, optional
        Threads for the kernels. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation of the interpolation and of the autoregressive step.

    Returns
    -------
    xarray.DataArray
        Reflectivity (dBZ) on ``(member, lead_time, y, x)`` for ``"steps"``,
        ``(lead_time, y, x)`` otherwise, ``float32``, with the coordinate
        ``valid_time`` and the scalar ``time`` of the analysis (when the
        frames have times). No echo is ``zero_value``; NaN is where the
        forecast comes from outside the observed area.

    Raises
    ------
    ValueError
        For unknown options, fewer frames than the order needs, frames on
        different grids or not evenly spaced in time, or a motion with NaN.
    ImportError
        If ``engine="compiled"`` and a kernel is not available.

    See Also
    --------
    exceedance_probability, estimate_motion, optical_flow, advect

    References
    ----------
    .. [1] Seed, A. W., 2003: A dynamic and spatial scaling approach to
       advection forecasting. *J. Appl. Meteor.*, **42**, 381-388,
       https://doi.org/10.1175/1520-0450(2003)042<0381:ADASSA>2.0.CO;2
    .. [2] Bowler, N. E., C. E. Pierce, and A. W. Seed, 2006: STEPS: A
       probabilistic precipitation forecasting scheme which merges an
       extrapolation nowcast with downscaled NWP. *Quart. J. Roy. Meteor.
       Soc.*, **132**, 2127-2155, https://doi.org/10.1256/qj.04.100
    .. [3] Pulkkinen, S., D. Nerini, A. A. Perez Hortal, C. Velasco-Forero,
       A. Seed, U. Germann, and L. Foresti, 2019: Pysteps: an open-source
       Python library for probabilistic precipitation nowcasting (v1.0).
       *Geosci. Model Dev.*, **12**, 4185-4219,
       https://doi.org/10.5194/gmd-12-4185-2019
    .. [4] Seed, A. W., C. E. Pierce, and K. Norman, 2013: Formulation and
       evaluation of a scale decomposition-based stochastic precipitation
       nowcast scheme. *Water Resour. Res.*, **49**, 6624-6641,
       https://doi.org/10.1002/wrcr.20536
    .. [5] Germann, U., and I. Zawadzki, 2002: Scale-dependence of the
       predictability of precipitation from continental radar images. Part I:
       Description of the methodology. *Mon. Wea. Rev.*, **130**, 2859-2873,
       https://doi.org/10.1175/1520-0493(2002)130<2859:SDOTPO>2.0.CO;2

    Examples
    --------
    >>> frames = [grid0, grid1, grid2]  # doctest: +SKIP
    >>> ens = radarx.retrieve.nowcast(frames, 12, seed=1)  # doctest: +SKIP
    >>> prob = radarx.retrieve.exceedance_probability(ens, 35.0)  # doctest: +SKIP
    """
    use_compiled = _use_compiled(engine)
    _check_options(
        method, n_steps, n_members, ar_order, n_levels, mask_radius, mask_growth
    )
    n_steps, n_members = int(n_steps), int(n_members)
    perturbation = _perturbation(motion_perturbation)
    if perturbation is not None and method != "steps":
        raise ValueError("motion_perturbation needs method='steps'")
    frames, planes, t_last, dt_s, steps = _history(obj, field, dt, x, y)
    template = _field(frames[-1], field)
    n_frames, ny, nx = planes.shape
    order = ar_order if ar_order is not None else (2 if n_frames >= 3 else 1)
    if method != "extrapolation" and order == 2 and n_frames < 3:
        raise ValueError("ar_order=2 needs at least three frames")
    dx, dy = _step(template, x), _step(template, y)
    if zero_value is None:
        zero_value = threshold - 5.0
    if not zero_value < threshold:
        raise ValueError("zero_value must be below threshold")
    if observed is None:
        seen = np.ones((ny, nx), dtype=bool)
    else:
        seen = np.asarray(observed.transpose(y, x).values, dtype=bool)
        if seen.shape != (ny, nx):
            raise ValueError("observed must be on the grid of the frames")
    kw = {"dt": dt_s, "use_compiled": use_compiled, "n_threads": n_threads}

    # motion
    if motion is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            est = estimate_motion(
                frames[-2],
                frames[-1],
                field,
                dt=dt_s,
                x=x,
                y=y,
                n_threads=n_threads,
                engine=engine,
            )
        u, v = float(est["u"]), float(est["v"])
        if not (np.isfinite(u) and np.isfinite(v)):
            warnings.warn(
                "no reliable motion found; the forecast is persistence (zero motion)",
                RuntimeWarning,
                stacklevel=2,
            )
            u = v = 0.0
    elif isinstance(motion, tuple):
        u, v = _motion_arrays(motion[0], motion[1], x, y, ny, nx)
    else:
        u, v = _motion_arrays(motion, None, x, y, ny, nx)

    # transformed history; NaN stays NaN only outside the observed area
    data = np.where(np.isfinite(planes) & (planes >= threshold), planes, zero_value)
    data = np.where(seen[None], data, np.nan)
    leads = dt_s * np.arange(1, n_steps + 1)
    if method == "extrapolation":
        src_r, src_c = _departure(u, v, leads, dx, dy, ny, nx, use_compiled, n_threads)
        res = _interpolate(
            data[-1][None], src_r, src_c, 1, 0.5, use_compiled, n_threads
        )
        result = res[:, 0][None]  # (1, step, ny, nx)
    else:
        lagr = _lagrangian_history(data, u, v, dx, dy, zero_value, seen, 3, steps, kw)
        shape, offset = _padded_shape(ny, nx, pad)
        w = _band_filters(shape, dy, dx, int(n_levels))
        setup = _cascade_setup(lagr, w, offset, shape, zero_value, order)
        z_init = _initial_states(setup, w, order)
        noise = method == "steps"
        wh = (
            _innovation_filter(setup, w, _noise_amplitude(lagr[-1], offset, shape))
            if noise
            else np.zeros_like(w)
        )
        n_run = n_members if noise else 1
        children = np.random.SeedSequence(seed).spawn(n_run + 1)
        members = [np.random.default_rng(c) for c in children[1:]]
        lag_fc = _lagrangian_forecast(
            setup, w, wh, z_init, n_steps, members, offset, (ny, nx), shape, kw, noise
        )
        latest = lagr[-1]
        echo = latest >= threshold
        if mask_radius is not None:
            dist = _distance_map(echo & seen, dy, dx)
            radius = mask_radius + mask_growth * leads
            inside = dist[None] <= radius[:, None, None]  # (step, ny, nx)
            lag_fc = np.where(inside[None], lag_fc, zero_value)
        if probability_matching:
            lag_fc = _match(
                lag_fc, seen, np.sort(latest[seen]), use_compiled, n_threads
            )
        lag_fc = np.where(lag_fc >= threshold, lag_fc, zero_value)
        lag_fc = np.where(seen[None, None], lag_fc, np.nan)
        shifts = None
        if perturbation is not None:
            rng = np.random.default_rng(children[0])
            minutes = leads / 60.0
            shifts = [
                _perturbed_motion(rng, u, v, perturbation[0], perturbation[1], minutes)
                for _ in range(n_run)
            ]
        result = _eulerian(lag_fc, u, v, leads, dx, dy, kw, shifts)
    result = np.where(
        np.isfinite(result), np.where(result >= threshold, result, zero_value), np.nan
    )
    return _wrap(
        result, template, t_last, leads, method, n_members, zero_value, threshold, x, y
    )


def _wrap(
    result, template, t_last, leads, method, n_members, zero_value, threshold, x, y
):
    """DataArray with the dimensions, coordinates and attributes of the forecast."""
    lead = (leads * 1e9).round().astype("timedelta64[ns]")
    coords = {
        k: c for k, c in template.coords.items() if set(c.dims) <= {x, y} and c.ndim > 0
    }
    coords["lead_time"] = ("lead_time", lead, {"long_name": "forecast lead time"})
    if t_last is not None:
        coords["time"] = ((), t_last, {"long_name": "analysis time"})
        coords["valid_time"] = (
            "lead_time",
            t_last + lead,
            {"long_name": "forecast valid time"},
        )
    dims = ("lead_time", y, x)
    if method == "steps":
        dims = ("member",) + dims
        coords["member"] = ("member", np.arange(result.shape[0]))
    else:
        result = result[0]
    attrs = {
        k: v
        for k, v in template.attrs.items()
        if k not in ("_FillValue", "valid_min", "valid_max")
    }
    attrs.update(
        units="dBZ",
        long_name="nowcast reflectivity",
        nowcast_method=method,
        no_echo_value=float(zero_value),
        echo_threshold=float(threshold),
    )
    return xr.DataArray(
        result.astype(FIELD_DTYPE),
        dims=dims,
        coords=coords,
        attrs=attrs,
        name=template.name if template.name is not None else "DBZH",
    )


@dask_support("computes", "it feeds the whole history to the network.")
@provenance("Nowcast with a learned ONNX model, radarx's own wrapper")
def nowcast_model(
    obj,
    model,
    n_steps,
    *,
    field=None,
    dt=None,
    n_input=None,
    offset=0.0,
    scale=1.0,
    threshold=10.0,
    zero_value=None,
    recurrent=True,
    x="x",
    y="y",
):
    """
    Nowcast with a learned network run by ONNX Runtime.

    The last ``n_input`` frames are normalised, ``(dBZ - offset) / scale``, and
    stacked as the channels of one input tensor of shape ``(1, n_input, ny,
    nx)``; the model returns the next frames as a tensor ``(1, k, ny, nx)``.
    This is the form of networks such as RainNet (Ayzel et al. 2020 [1]_,
    which predicts the next frame from four) and of the networks that predict
    several frames at once (DGMR, Ravuri et al. 2021 [2]_, uses a different
    input and output layout and needs an adapter to this one). With
    ``recurrent=True`` and a model that returns fewer than ``n_steps`` frames,
    the prediction is appended to the input and the model is run again, as
    RainNet is.

    radarx ships no weights. The network, its licence and its citation come
    from the model registry of :mod:`radarx.ml` (:func:`radarx.ml.load_model`,
    :func:`radarx.ml.register_model`); the attributes ``ml_model``,
    ``ml_model_version``, ``ml_model_licence`` and ``ml_model_citation`` of
    the model are written to the result. The network must accept the grid
    as it is (a dynamic or an equal spatial size): the grid is not cut into
    patches. The wrapper, the normalisation and the recursion are radarx's
    own; the checks of the two papers' models are those of their authors, not
    repeated here.

    Parameters
    ----------
    obj : list of xarray.Dataset or xarray.DataArray, or xarray.Dataset
        Gridded volumes in time order, or one object with a ``time``
        dimension, as for :func:`nowcast`.
    model : radarx.ml.Model or str
        An ONNX model, or the name of a registered one.
    n_steps : int
        Number of lead times, ``dt, 2 dt, ..., n_steps dt``.
    field : str, optional
        Reflectivity field in dBZ. Default: the first reflectivity field found.
    dt : float or timedelta, optional
        Time between the frames in seconds. Default: from their ``time``.
    n_input : int, optional
        Number of frames fed to the model. Default: the size of the second
        dimension of the model input if it is fixed, else all frames.
    offset, scale : float, optional
        Normalisation of the input, ``(dBZ - offset) / scale``, and of the
        output (the inverse). Default 0 and 1.
    threshold : float, optional
        Reflectivity (dBZ) below which there is no echo. Default 10.
    zero_value : float, optional
        Reflectivity assigned to no echo, also in the result. Default:
        ``threshold - 5``.
    recurrent : bool, optional
        Feed the predictions back to get more lead times than the model
        returns. Default True.
    x, y : str, optional
        Names of the horizontal coordinates. Default ``"x"`` and ``"y"``.

    Returns
    -------
    xarray.DataArray
        Reflectivity (dBZ) on ``(lead_time, y, x)``, ``float32``.

    Raises
    ------
    ValueError
        If there are fewer frames than ``n_input``, the output of the model is
        not ``(1, k, ny, nx)``, it returns fewer than ``n_steps`` frames and
        ``recurrent`` is False, or the model has more than one input.
    ImportError
        If ONNX Runtime is not installed (``pip install radarx[ml]``).

    See Also
    --------
    nowcast, radarx.ml.load_model

    References
    ----------
    .. [1] Ayzel, G., T. Scheffer, and M. Heistermann, 2020: RainNet v1.0: a
       convolutional neural network for radar-based precipitation nowcasting.
       *Geosci. Model Dev.*, **13**, 2631-2644,
       https://doi.org/10.5194/gmd-13-2631-2020
    .. [2] Ravuri, S., and Coauthors, 2021: Skilful precipitation nowcasting
       using deep generative models of radar. *Nature*, **597**, 672-677,
       https://doi.org/10.1038/s41586-021-03854-z

    Examples
    --------
    >>> model = radarx.ml.load_model("rainnet")  # doctest: +SKIP
    >>> fc = radarx.retrieve.nowcast_model(frames, model, 6, scale=10)  # doctest: +SKIP
    """
    from ..ml import Model, load_model

    if int(n_steps) != n_steps or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    if not scale > 0:
        raise ValueError("scale must be positive")
    n_steps = int(n_steps)
    net = model if isinstance(model, Model) else load_model(model)
    inputs = net.session.get_inputs()
    if len(inputs) != 1:
        raise ValueError(f"the model has {len(inputs)} inputs, one is needed")
    spec = inputs[0].shape
    frames, planes, t_last, dt_s, _ = _history(obj, field, dt, x, y)
    template = _field(frames[-1], field)
    if n_input is None:
        n_input = (
            spec[1] if len(spec) == 4 and isinstance(spec[1], int) else len(frames)
        )
    if int(n_input) != n_input or n_input < 1:
        raise ValueError("n_input must be a positive integer")
    n_input = int(n_input)
    if len(frames) < n_input:
        raise ValueError(f"the model needs {n_input} frames, {len(frames)} are given")
    if zero_value is None:
        zero_value = threshold - 5.0
    if not zero_value < threshold:
        raise ValueError("zero_value must be below threshold")
    data = np.where(np.isfinite(planes) & (planes >= threshold), planes, zero_value)
    window = ((data[-n_input:] - offset) / scale)[None]  # (1, n_input, ny, nx)
    predicted = []
    while len(predicted) < n_steps:
        out = next(iter(net.run(window.astype(np.float32)).values()))
        if out.ndim != 4 or out.shape[0] != 1 or out.shape[2:] != window.shape[2:]:
            raise ValueError(
                f"the model returned {out.shape}, (1, k, {window.shape[2]}, "
                f"{window.shape[3]}) is needed"
            )
        predicted.extend(out[0].astype(np.float64))
        if len(predicted) < n_steps and not recurrent:
            raise ValueError(
                f"the model returns {out.shape[1]} frames, {n_steps} are needed: "
                "use recurrent=True"
            )
        window = np.concatenate([window, out.astype(np.float64)], axis=1)[:, -n_input:]
    result = np.stack(predicted[:n_steps]) * scale + offset
    result = np.where(result >= threshold, result, zero_value)
    leads = dt_s * np.arange(1, n_steps + 1)
    da = _wrap(
        result[None], template, t_last, leads, "model", 1, zero_value, threshold, x, y
    )
    da.attrs.update(net.attrs)
    return da


@provenance("Ensemble exceedance probability, after Germann and Zawadzki (2004)")
def exceedance_probability(ensemble, threshold, *, member_dim="member"):
    """
    Share of the members of an ensemble at or above a threshold.

    The probability of exceeding a threshold is estimated as the fraction of
    the ensemble members that exceed it (the ensemble probability of Bowler et
    al. 2006 [1]_). A cell is NaN where every member is NaN; NaN members are
    left out of the count.

    Parameters
    ----------
    ensemble : xarray.DataArray
        Forecast with a member dimension, e.g. from :func:`nowcast`.
    threshold : float
        Threshold in the units of the ensemble.
    member_dim : str, optional
        Name of the member dimension. Default ``"member"``.

    Returns
    -------
    xarray.DataArray
        Probability (between 0 and 1) on the dimensions of the ensemble
        without the member dimension.

    Raises
    ------
    ValueError
        If the ensemble has no dimension ``member_dim``.

    See Also
    --------
    nowcast

    References
    ----------
    .. [1] Bowler, N. E., C. E. Pierce, and A. W. Seed, 2006: STEPS: A
       probabilistic precipitation forecasting scheme which merges an
       extrapolation nowcast with downscaled NWP. *Quart. J. Roy. Meteor.
       Soc.*, **132**, 2127-2155, https://doi.org/10.1256/qj.04.100
    .. [2] Germann, U., and I. Zawadzki, 2004: Scale dependence of the
       predictability of precipitation from continental radar images. Part
       II: Probability forecasts. *J. Appl. Meteor.*, **43**, 74-89,
       https://doi.org/10.1175/1520-0450(2004)043<0074:SDOTPO>2.0.CO;2

    Examples
    --------
    >>> prob = radarx.retrieve.exceedance_probability(ens, 35.0)  # doctest: +SKIP
    """
    if member_dim not in ensemble.dims:
        raise ValueError(f"the ensemble has no dimension {member_dim!r}")
    valid = ensemble.notnull()
    hits = ((ensemble >= threshold) & valid).sum(member_dim)
    count = valid.sum(member_dim)
    prob = (hits / count.where(count > 0)).astype(FIELD_DTYPE)
    prob.attrs = {
        "units": "1",
        "long_name": f"probability of exceeding {threshold:g}",
    }
    prob.name = "probability"
    return prob


# --------------------------------------------------------------------------
# accessors
# --------------------------------------------------------------------------


@accessor_method("dataarray", "dataset", name="nowcast", wraps=nowcast)
def _nowcast_accessor(self):
    """
    Nowcast of a gridded reflectivity field with a ``time`` dimension.

    See :func:`radarx.retrieve.nowcast` for the parameters.

    Returns
    -------
    xarray.DataArray
        Reflectivity (dBZ) on ``(member, lead_time, y, x)`` for the ensemble
        method, ``(lead_time, y, x)`` otherwise.
    """


@accessor_method(
    "dataarray", "dataset", name="exceedance_probability", wraps=exceedance_probability
)
def _exceedance_probability_accessor(self):
    """
    Probability that an ensemble forecast reaches a threshold.

    See :func:`radarx.retrieve.exceedance_probability` for the parameters.

    Returns
    -------
    xarray.DataArray
        Probability between 0 and 1.
    """
