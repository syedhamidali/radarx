# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Optical-Flow Motion
===================

Storm motion from a sequence of gridded radar fields by optical flow, as an
alternative to the FFT cross-correlation of
:func:`radarx.retrieve.estimate_motion`. Both give the ``u`` and ``v`` of the
Dataset that :func:`radarx.retrieve.advect` and
:func:`radarx.retrieve.nowcast` take.

:func:`optical_flow` offers two variational methods that use the brightness
constancy ``dI/dt + u dI/dx + v dI/dy = 0`` of the echo pattern:

``"lucas_kanade"``
    The local least squares of Lucas and Kanade (1981): every cell takes the
    motion that best explains the change of the echo in a Gaussian window
    around it.
``"darts"``
    A global least squares in the spirit of DARTS (Ruzanski et al. 2011): the
    flow equation is fitted to all cells and all frames at once, with the
    motion written as a few leading coefficients of a cosine series, which
    gives a smooth field whose scale is set by the number of coefficients.

Both run from coarse to fine on an image pyramid and warp the later field
with the current estimate before each correction, which lets them find
displacements of many cells (Bouguet 2001). The sums of the least squares run
in a multithreaded C++ kernel with an equivalent NumPy implementation.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = ["optical_flow"]

__doc__ = __doc__.format("\n   ".join(__all__))

import warnings

import numpy as np
import xarray as xr

from .._provenance import provenance
from .._registry import accessor_method
from .._sweepwise import dask_support
from ._optical_flow_numpy import (
    HAS_COMPILED_KERNEL,  # noqa: F401  (read by the tests)
    cosine_basis,
    darts_normal,
    lk_solve,
)
from .advection import _field, _interpolate, _step, _use_compiled
from .nowcasting import _history

_METHODS = ("lucas_kanade", "darts")
# largest increment per correction, in cells of the level (radarx choice)
_MAX_INCREMENT = 2.0
# displacement left to the finest level of the pyramid, in cells (radarx choice)
_CELLS_PER_LEVEL = 3.0
# smallest side of the coarsest level, in cells
_MIN_SIDE = 16


def _taps(sigma):
    """Normalised Gaussian taps of standard deviation ``sigma`` (cells)."""
    radius = max(1, int(4.0 * sigma + 0.5))
    k = np.arange(-radius, radius + 1)
    w = np.exp(-0.5 * (k / sigma) ** 2)
    return w / w.sum()


def _pool(a):
    """2 by 2 block mean of arrays (..., ny, nx); an odd last row/column is cut."""
    ny, nx = a.shape[-2] // 2 * 2, a.shape[-1] // 2 * 2
    a = a[..., :ny, :nx]
    return a.reshape(*a.shape[:-2], ny // 2, 2, nx // 2, 2).mean(axis=(-3, -1))


def _upsample(flow, shape):
    """Flow (2, ny, nx) of a coarse level on the next finer ``shape``, in its cells."""
    up = np.repeat(np.repeat(flow, 2, axis=-2), 2, axis=-1) * 2.0
    pad = ((0, 0), (0, shape[0] - up.shape[1]), (0, shape[1] - up.shape[2]))
    return np.pad(up, pad, mode="edge")


def _derivatives(a, b, flow, margin, scales, use_compiled, n_threads):
    """
    Space and time derivatives of the pairs after warping ``b`` by ``flow``.

    ``a`` and ``b`` are (pair, ny, nx), ``flow`` (2, ny, nx) holds the
    displacement in rows and columns, per mean time step, that carries ``a``
    onto ``b``; pair ``k`` is ``scales[k]`` times that far apart. ``b`` is
    sampled at ``x + scales[k] flow`` (cubic convolution). Cells within ``margin``
    cells of the edge of the grid, and cells whose sample lies that close to
    it or outside, get zero derivatives: the smoothing of the echo reaches
    ``margin`` cells further than the data, and what enters or leaves the
    grid is not seen in both frames and would bias the estimate. Returns
    ``ix``, ``iy``, ``it`` and the mask of the cells that count.
    """
    n, ny, nx = a.shape
    rows = np.arange(ny, dtype=np.float64)[None, :, None]
    cols = np.arange(nx, dtype=np.float64)[None, None, :]
    warped = np.empty(a.shape)
    inside = np.empty(a.shape, dtype=bool)
    for k in range(n):
        src_r = rows + scales[k] * flow[0][None]
        src_c = cols + scales[k] * flow[1][None]
        warped[k] = _interpolate(
            b[k : k + 1], src_r, src_c, 3, 0.5, use_compiled, n_threads
        )[0, 0]
        inside[k] = (
            (rows[0] >= margin)
            & (rows[0] <= ny - 1 - margin)
            & (cols[0] >= margin)
            & (cols[0] <= nx - 1 - margin)
            & (src_r[0] >= margin)
            & (src_r[0] <= ny - 1 - margin)
            & (src_c[0] >= margin)
            & (src_c[0] <= nx - 1 - margin)
        )
    valid = np.isfinite(warped) & inside
    warped = np.where(np.isfinite(warped), warped, 0.0)
    pad = np.pad(warped, ((0, 0), (1, 1), (1, 1)), mode="edge")
    ix = 0.5 * (pad[:, 1:-1, 2:] - pad[:, 1:-1, :-2])
    iy = 0.5 * (pad[:, 2:, 1:-1] - pad[:, :-2, 1:-1])
    it = warped - a
    ix *= scales[:, None, None]  # the derivative with respect to the flow
    iy *= scales[:, None, None]
    for arr in (ix, iy, it):
        arr[~valid] = 0.0
    return ix, iy, it, valid


def _lk_increment(ix, iy, it, taps, sigma_s, tol, use_compiled, n_threads):
    """
    Lucas-Kanade correction of the flow, (2, ny, nx), in cells.

    The local solution is kept where the smaller eigenvalue of the structure
    tensor reaches ``tol`` times the largest eigenvalue of the field (the
    aperture problem makes the others unreliable), clipped to
    ``_MAX_INCREMENT``, and spread to the other cells by the normalised
    Gaussian average of standard deviation ``sigma_s`` cells (``None``: one
    weighted mean for the whole field).
    """
    from scipy.ndimage import gaussian_filter

    du, dv, lmin, lmax = lk_solve(ix, iy, it, taps, use_compiled, n_threads)
    top = lmax.max()
    if not top > 0:
        return np.zeros((2,) + du.shape)
    weight = np.where(lmin >= tol * top, lmin, 0.0)
    out = np.empty((2,) + du.shape)
    total = weight.sum()
    for k, d in enumerate((dv, du)):  # rows, then columns
        d = np.clip(d, -_MAX_INCREMENT, _MAX_INCREMENT)
        mean = float((weight * d).sum() / total) if total > 0 else 0.0
        if sigma_s is None:
            out[k] = mean
            continue
        num = gaussian_filter(weight * d, sigma_s, mode="nearest")
        den = gaussian_filter(weight, sigma_s, mode="nearest")
        ok = den > 1e-6 * weight.max() if weight.max() > 0 else np.zeros_like(den, bool)
        out[k] = np.where(ok, num / np.where(ok, den, 1.0), mean)
    return out


def _darts_increment(ix, iy, it, order, ridge, use_compiled, n_threads):
    """
    DARTS correction of the flow, (2, ny, nx), in cells.

    Solves the normal equations of the fit of ``ix u + iy v + it = 0`` with
    ``u`` and ``v`` in the cosine basis of ``order``. A ridge term
    ``ridge * mean(diag) * (1 + p**2 + q**2)`` keeps the higher coefficients
    small (radarx choice).
    """
    ny, nx = ix.shape[1:]
    eq = darts_normal(ix, iy, it, order, use_compiled, n_threads)
    d = eq.shape[0]
    m = d // 2
    a, rhs = eq[:, :d].copy(), eq[:, d]
    if not np.trace(a) > 0:
        return np.zeros((2, ny, nx))
    p = np.repeat(np.arange(order + 1), order + 1)
    q = np.tile(np.arange(order + 1), order + 1)
    penalty = np.tile(1.0 + p**2 + q**2, 2)
    a[np.diag_indices(d)] += ridge * np.trace(a) / d * penalty
    try:
        coef = np.linalg.solve(a, rhs)
    except np.linalg.LinAlgError:  # pragma: no cover - the ridge makes it regular
        return np.zeros((2, ny, nx))
    basis = cosine_basis(order, ny, nx)
    return np.stack(
        [np.tensordot(coef[m:], basis, axes=1), np.tensordot(coef[:m], basis, axes=1)]
    )


def _levels(max_cells, shape, n_levels):
    """Number of pyramid levels."""
    if n_levels is None:
        n_levels = 1 + int(np.ceil(np.log2(max(max_cells / _CELLS_PER_LEVEL, 1.0))))
    limit = 1
    side = min(shape)
    while side // 2 >= _MIN_SIDE:
        side //= 2
        limit += 1
    return int(max(1, min(n_levels, limit)))


@dask_support("computes", "it tracks the whole fields of all frames.")
@provenance(
    "Optical-flow storm motion (Lucas-Kanade, DARTS), radarx's own",
    extra_refs=("lucas-kanade-1981", "bouguet-2001"),
)
def optical_flow(
    obj,
    *,
    method="lucas_kanade",
    field=None,
    dt=None,
    floor=5.0,
    prefilter=2e3,
    window=6e3,
    smooth=20e3,
    min_eigenvalue=1e-2,
    order=2,
    ridge=1e-3,
    uniform=False,
    max_speed=60.0,
    n_levels=None,
    iterations=3,
    x="x",
    y="y",
    n_threads=None,
    engine="auto",
):
    """
    Storm motion from a sequence of gridded fields by optical flow.

    The echo pattern ``I`` is assumed to be carried by the motion, ``I(x, t +
    dt) = I(x - u dt, t)``, which for small displacements is the optical-flow
    constraint ``I_t + u I_x + v I_y = 0``. All pairs of consecutive frames
    share one motion. Two methods solve it:

    ``"lucas_kanade"``
        The local least squares of Lucas and Kanade (1981) [1]_: at every
        cell the constraints of the cells in a Gaussian window (standard
        deviation ``window``) give a 2 by 2 system for ``(u, v)``. Where its
        smaller eigenvalue is below ``min_eigenvalue`` times the largest
        found in the field the cell has no texture along some direction (the
        aperture problem; the corner criterion of Shi and Tomasi 1994 [2]_)
        and takes the average of its neighbours, a Gaussian average of
        standard deviation ``smooth`` that weights each cell by its smaller
        eigenvalue.
    ``"darts"``
        A global least squares in the spirit of DARTS (Ruzanski et al. 2011
        [3]_), which solves the flow equation over the whole domain: the
        constraints of all cells and frames are fitted at once with ``u`` and
        ``v`` as the sum of the ``(order + 1)**2`` leading terms of a cosine
        series. ``order=0`` is one vector for the domain, larger orders allow
        a smoother or sharper variation of the motion. The cosine series
        stands in for the leading Fourier coefficients of the DARTS flow
        equation (the field is not periodic).

    Both start on a coarse image pyramid (2 by 2 block means) and refine the
    estimate on every finer level: the later field is warped with the current
    motion (cubic convolution) and the correction is found from the remaining
    difference, ``iterations`` times per level (Bouguet 2001 [4]_). This
    finds displacements of many cells, ``n_levels`` limits their size.

    The echo is ``max(dBZ - floor, 0)`` smoothed by a Gaussian of standard
    deviation ``prefilter``; missing values count as no echo. This
    preprocessing, the choice of the aperture test and its default, the
    normalised averaging, the limit of the correction per iteration, the
    ridge term and all defaults are radarx's own; Lucas and Kanade (1981)
    give the local least squares, Ruzanski et al. (2011) the idea of the
    global spectral fit. The method of Ruzanski et al. (2011) is not copied:
    only its flow equation is used, and the cosine series replaces the
    discrete Fourier coefficients.

    Parameters
    ----------
    obj : list of xarray.Dataset or xarray.DataArray, or xarray.Dataset
        Gridded volumes in time order (at least two), or one object with a
        ``time`` dimension, with evenly spaced ``x`` and ``y`` in metres.
        Extra dimensions (``z``) are reduced to the column maximum.
    method : {"lucas_kanade", "darts"}, optional
        Default ``"lucas_kanade"``.
    field : str, optional
        Field to track. Default: the first reflectivity field found.
    dt : float or timedelta, optional
        Time between frames in seconds. Default: from their ``time``.
    floor : float, optional
        Reflectivity (dBZ) below which there is no echo. Default 5.
    prefilter : float, optional
        Standard deviation (m) of the Gaussian smoothing of the echo before
        the derivatives are taken. Default 2 km.
    window : float, optional
        Standard deviation (m) of the Lucas-Kanade window. Default 6 km.
    smooth : float, optional
        Standard deviation (m) of the Gaussian average that fills cells
        without texture (Lucas-Kanade). Default 20 km.
    min_eigenvalue : float, optional
        Smaller eigenvalue of the structure tensor, as a share of the
        largest eigenvalue of the field, below which a cell is not trusted
        (Lucas-Kanade). Default 0.01.
    order : int, optional
        Highest order of the cosine series (DARTS). Default 2 (9 terms for
        each component).
    ridge : float, optional
        Weight of the term that keeps the higher coefficients small (DARTS),
        relative to the mean diagonal of the normal matrix. Default 0.001.
    uniform : bool, optional
        One motion vector for the whole domain: the weighted mean over the
        field for Lucas-Kanade, ``order=0`` for DARTS. Default False.
    max_speed : float, optional
        Largest motion expected (m/s); sets the number of pyramid levels when
        ``n_levels`` is not given. Default 60.
    n_levels : int, optional
        Pyramid levels (1: no pyramid). Default: enough for ``max_speed``,
        at most what the grid allows (the coarsest level has at least 16
        cells on its shorter side).
    iterations : int, optional
        Corrections per level. Default 3.
    x, y : str, optional
        Names of the horizontal coordinates. Default ``"x"`` and ``"y"``.
    n_threads : int, optional
        Threads for the kernels. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation of the least squares and of the warping.

    Returns
    -------
    xarray.Dataset
        ``u`` (eastward, along ``x``) and ``v`` (northward, along ``y``)
        motion in m/s, on ``(y, x)`` (scalars for ``uniform=True`` or DARTS
        with ``order=0``), and ``quality``, the share of the change between
        the frames that the final motion explains (0 to 1).

    Raises
    ------
    ValueError
        For an unknown ``method``, fewer than two frames, frames that are not
        evenly spaced in time or on different grids, or invalid options.
    ImportError
        If ``engine="compiled"`` and a kernel is not available.

    See Also
    --------
    estimate_motion, advect, nowcast

    References
    ----------
    .. [1] Lucas, B. D., and T. Kanade, 1981: An iterative image registration
       technique with an application to stereo vision. *Proc. 7th Int. Joint
       Conf. on Artificial Intelligence*, Vancouver, 674-679 (no DOI).
    .. [2] Shi, J., and C. Tomasi, 1994: Good features to track. *Proc. IEEE
       Conf. on Computer Vision and Pattern Recognition*, 593-600,
       https://doi.org/10.1109/CVPR.1994.323794
    .. [3] Ruzanski, E., V. Chandrasekar, and Y. Wang, 2011: The CASA
       nowcasting system. *J. Atmos. Oceanic Technol.*, **28**, 640-655,
       https://doi.org/10.1175/2011JTECHA1496.1
    .. [4] Bouguet, J.-Y., 2001: Pyramidal implementation of the affine Lucas
       Kanade feature tracker: description of the algorithm. Intel
       Corporation Microprocessor Research Labs (no DOI).

    Examples
    --------
    >>> frames = [grid0, grid1, grid2]  # doctest: +SKIP
    >>> motion = radarx.retrieve.optical_flow(frames)  # doctest: +SKIP
    >>> moved = radarx.retrieve.advect(grid2, motion, dt=300.0)  # doctest: +SKIP
    """
    use_compiled = _use_compiled(engine)
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, not {method!r}")
    if int(order) != order or order < 0:
        raise ValueError("order must be a non-negative integer")
    if int(iterations) != iterations or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if n_levels is not None and (int(n_levels) != n_levels or n_levels < 1):
        raise ValueError("n_levels must be a positive integer")
    for name, value in (
        ("prefilter", prefilter),
        ("window", window),
        ("smooth", smooth),
    ):
        if value is None or not value > 0:
            raise ValueError(f"{name} must be positive")
    frames, planes, _, dt_s, steps = _history(obj, field, dt, x, y)
    scales = steps / dt_s
    template = _field(frames[-1], field)
    dx, dy = _step(template, x), _step(template, y)
    n, ny, nx = planes.shape
    echo = np.where(np.isfinite(planes), np.clip(planes - floor, 0.0, None), 0.0)
    if prefilter:
        from scipy.ndimage import gaussian_filter

        echo = gaussian_filter(echo, (0, prefilter / abs(dy), prefilter / abs(dx)))
    a, b = echo[:-1], echo[1:]
    if method == "darts" and uniform:
        order = 0
    max_cells = max_speed * dt_s / min(abs(dx), abs(dy))
    levels = _levels(max_cells, (ny, nx), n_levels)
    pyramid = [(a, b)]
    for _ in range(levels - 1):
        pyramid.append((_pool(pyramid[-1][0]), _pool(pyramid[-1][1])))
    if not (a > 0).any() or not (b > 0).any():
        warnings.warn(
            "no echo above the floor: the motion is zero", RuntimeWarning, stacklevel=2
        )
    flow = np.zeros((2,) + pyramid[-1][0].shape[1:])
    for level in range(levels - 1, -1, -1):
        la, lb = pyramid[level]
        if level < levels - 1:
            flow = _upsample(flow, la.shape[1:])
        scale = 2.0**level
        taps = _taps(max(window / abs(dx) / scale, 1.0))
        sigma_s = None if uniform else max(smooth / abs(dx) / scale, 1.0)
        margin = int(np.ceil(3.0 * prefilter / abs(dx) / scale)) + 1
        for _ in range(iterations):
            ix, iy, it, _ = _derivatives(
                la, lb, flow, margin, scales, use_compiled, n_threads
            )
            if method == "lucas_kanade":
                flow = flow + _lk_increment(
                    ix, iy, it, taps, sigma_s, min_eigenvalue, use_compiled, n_threads
                )
            else:
                flow = flow + _darts_increment(
                    ix, iy, it, int(order), ridge, use_compiled, n_threads
                )
    # quality: the share of the change between the frames that is explained
    margin0 = int(np.ceil(3.0 * prefilter / abs(dx))) + 1
    _, _, it_final, counted = _derivatives(
        a, b, flow, margin0, scales, use_compiled, n_threads
    )
    quality = 0.0
    if (a > 0).any() and counted.any():
        change = float(np.sqrt(np.mean(((b - a) ** 2)[counted])))
        left = float(np.sqrt(np.mean((it_final**2)[counted])))
        quality = float(np.clip(1.0 - left / change, 0.0, 1.0)) if change > 0 else 1.0
    to_u, to_v = dx / dt_s, dy / dt_s  # cells -> m/s
    u_val, v_val = flow[1] * to_u, flow[0] * to_v
    dims, coords = (y, x), {
        k: c for k, c in template.coords.items() if set(c.dims) <= {x, y} and c.ndim > 0
    }
    if uniform or (method == "darts" and order == 0):
        u_val, v_val = float(u_val.mean()), float(v_val.mean())
        dims, coords = (), {}
    attrs = {"units": "m s-1"}
    return xr.Dataset(
        {
            "u": (dims, u_val, dict(attrs, long_name="eastward storm motion")),
            "v": (dims, v_val, dict(attrs, long_name="northward storm motion")),
            "quality": (
                (),
                quality,
                {
                    "units": "1",
                    "long_name": "share of the change explained by the motion",
                },
            ),
        },
        coords=coords,
        attrs={"dt": dt_s, "tracked_field": str(template.name), "flow_method": method},
    )


# --------------------------------------------------------------------------
# accessors
# --------------------------------------------------------------------------


@accessor_method("dataarray", "dataset", name="optical_flow", wraps=optical_flow)
def _optical_flow_accessor(self):
    """
    Storm motion by optical flow from a gridded field with a ``time`` dimension.

    See :func:`radarx.retrieve.optical_flow` for the parameters.

    Returns
    -------
    xarray.Dataset
        ``u`` and ``v`` (m/s) and ``quality``.
    """
