# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Ensembles of the error of a rainfall accumulation
=================================================

A radar rainfall accumulation is one estimate. These functions turn it into an
ensemble of possible truths under an **assumed error model**, so that the
probability of reaching a depth can be asked for (with
:func:`radarx.retrieve.exceedance_probability`) and checked against rain gauges
(with :func:`radarx.retrieve.reliability` and
:func:`radarx.retrieve.brier_score`).

* :func:`qpe_error_ensemble`: members of the multiplicative, lognormal,
  spatially correlated error model.
* :func:`qpe_error_model`: the parameters of that model from pairs of radar and
  gauge accumulations.

The model is in the logarithm of the rain, as in the radar ensemble generator
of Germann et al. (2009), who describe the radar error as the log ratio of
gauge and radar with a spatial structure given by an exponential
semivariogram. The construction here is simpler and radarx's own: one member is

.. math::

    A_m(x) = A(x) \\, \\exp\\left(s\\, w(x)\\, z_m(x) - \\tfrac12 s^2 w(x)^2\\right)

with :math:`s` the standard deviation of :math:`\\ln(\\text{truth} / \\text{estimate})`,
:math:`w` an optional scale per cell, and :math:`z_m` a Gaussian field of unit
variance that is the sum of three independent parts with the variance shares
``(common, spatial, nugget)``: one number for the whole field (a calibration or
Z-R bias is the same everywhere), a field with the exponential correlation
:math:`\\exp(-h/L)` and noise from cell to cell. The last term makes the
factor average to 1: **the ensemble mean is the estimate** (a mean of depths in
mm, never of decibels). A cell with no rain stays dry, a missing cell stays
missing. The spatial part is simulated by circulant embedding of the
covariance (Dietrich and Newsam 1997).

The ensemble represents the error model that is passed in, not the physics of
the rain. It contains no vertical profile, bright-band, overshoot or advection
errors, and its probabilities are not calibrated unless a reliability check
against independent gauges says so.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = ["qpe_error_ensemble", "qpe_error_model"]

__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

import itertools
import math

import numpy as np
import xarray as xr

from .._dtypes import FIELD_DTYPE
from .._provenance import provenance
from .._sweepwise import dask_support

#: minimum number of pairs for the error standard deviation and for the variogram
MIN_PAIRS = 5
MIN_PAIRS_VARIOGRAM = 8


def _spacing(da, dim):
    c = np.asarray(da[dim].values, dtype=np.float64)
    if c.size < 2:
        raise ValueError(f"dimension {dim!r} needs at least 2 coordinates")
    step = np.diff(c)
    if not np.allclose(step, step[0], rtol=1e-3):
        raise ValueError(f"the coordinates {dim!r} must be regularly spaced")
    return abs(float(step[0]))


class _CorrelatedField:
    """Unit-variance Gaussian fields with covariance ``exp(-h / L)`` on a regular grid."""

    def __init__(self, shape, length_cells):
        ny, nx = shape
        self.shape = shape
        my, mx = 2 * ny, 2 * nx
        iy = np.minimum(np.arange(my), my - np.arange(my))
        ix = np.minimum(np.arange(mx), mx - np.arange(mx))
        cov = np.exp(-np.hypot(iy[:, None], ix[None, :]) / max(length_cells, 1e-6))
        lam = np.clip(
            np.fft.fft2(cov).real, 0.0, None
        )  # tiny negative eigenvalues -> 0
        self._amp = np.sqrt(lam / (my * mx))
        self._norm = math.sqrt(lam.sum() / (my * mx))  # standard deviation (about 1)
        self._m = (my, mx)

    def sample(self, rng):
        my, mx = self._m
        w = rng.standard_normal((my, mx)) + 1j * rng.standard_normal((my, mx))
        x = np.fft.fft2(self._amp * w).real
        ny, nx = self.shape
        return x[:ny, :nx] / self._norm


@dask_support("computes", "it simulates correlated fields over the whole grid.")
@provenance("Lognormal error ensemble of a rainfall accumulation")
def qpe_error_ensemble(
    accumulation,
    *,
    sigma_ln,
    correlation_length,
    shares=(0.3, 0.5, 0.2),
    members=30,
    seed=0,
    scale=None,
):
    """
    Ensemble of a rainfall accumulation under a multiplicative lognormal error.

    Member ``k`` is ``accumulation * exp(s w z_k - s**2 w**2 / 2)`` (see the
    module description), so the mean of the members is the accumulation and the
    variance of one member of a constant field ``A`` is
    ``A**2 (exp(s**2 w**2) - 1)``. The members are reproducible: member ``k``
    is drawn from ``numpy.random.SeedSequence(seed).spawn(members)[k]``, so the
    same seed gives the same members and more members do not change the first
    ones.

    Parameters
    ----------
    accumulation : xarray.DataArray
        Depth in mm (or any positive quantity that adds up in time) on a
        regular grid with 2-D dimensions and coordinates in metres (the last
        two dimensions are taken as y and x).
    sigma_ln : float
        Standard deviation ``s`` of ``ln(truth / estimate)`` (0.3 is about
        30 %), between 0 and 3.
    correlation_length : float
        Correlation length ``L`` of the spatially correlated part, in metres.
    shares : sequence of three floats, optional
        Shares of the variance of ``z`` in ``(common, spatial, nugget)``:
        non-negative and adding up to 1. Default ``(0.3, 0.5, 0.2)``, an
        assumption, not a measurement; take them from
        :func:`qpe_error_model` when gauges are available.
    members : int, optional
        Number of members, 2 or more. Default 30.
    seed : int, optional
        Seed of the random numbers. Default 0.
    scale : xarray.DataArray or array_like, optional
        ``w``: scales the error cell by cell (for example the relative
        uncertainty of the estimate divided by its median). Default 1.

    Returns
    -------
    xarray.DataArray
        ``float32`` members on the dimension ``member`` followed by the
        dimensions of ``accumulation``, in the units of ``accumulation``.

    Raises
    ------
    ValueError
        For wrong ``sigma_ln``, ``shares``, ``members``, ``correlation_length``
        or an irregular grid.

    See Also
    --------
    qpe_error_model, exceedance_probability, reliability

    References
    ----------
    .. [1] Germann, U., M. Berenguer, D. Sempere-Torres, and M. Zappa, 2009:
       REAL: Ensemble radar precipitation estimation for hydrology in a
       mountainous region. *Quart. J. Roy. Meteor. Soc.*, **135**, 445-456.
    .. [2] Dietrich, C. R., and G. N. Newsam, 1997: Fast and exact simulation
       of stationary Gaussian processes through circulant embedding of the
       covariance matrix. *SIAM J. Sci. Comput.*, **18**, 1088-1107,
       https://doi.org/10.1137/S1064827592240555

    Examples
    --------
    >>> ens = radarx.retrieve.qpe_error_ensemble(
    ...     acc, sigma_ln=0.35, correlation_length=25e3, members=100
    ... )  # doctest: +SKIP
    >>> p = radarx.retrieve.exceedance_probability(ens, 25.0)  # doctest: +SKIP
    """
    if accumulation.ndim != 2:
        raise ValueError("accumulation must be 2-D (y, x)")
    s = float(sigma_ln)
    if not (math.isfinite(s) and 0.0 < s <= 3.0):
        raise ValueError("sigma_ln must be between 0 and 3")
    sh = np.asarray(shares, dtype=np.float64)
    if (
        sh.shape != (3,)
        or np.any(sh < 0)
        or not math.isclose(float(sh.sum()), 1.0, abs_tol=1e-6)
    ):
        raise ValueError(
            "shares are three numbers >= 0 that add up to 1 (common, spatial, nugget)"
        )
    if int(members) != members or members < 2:
        raise ValueError("members must be a whole number of 2 or more")
    members = int(members)
    length = float(correlation_length)
    if not (math.isfinite(length) and length > 0):
        raise ValueError("correlation_length must be positive (metres)")
    ydim, xdim = accumulation.dims
    dy, dx = _spacing(accumulation, ydim), _spacing(accumulation, xdim)
    base = np.asarray(accumulation.values, dtype=np.float32)
    ny, nx = base.shape
    w = (
        1.0
        if scale is None
        else np.asarray(getattr(scale, "values", scale), dtype=np.float32)
    )
    sig = np.float32(s) * w
    half = np.float32(0.5) * sig * sig
    c, sp, nug = np.sqrt(sh)
    gen = _CorrelatedField((ny, nx), length / ((dx + dy) / 2.0)) if sp > 0 else None
    out = np.empty((members, ny, nx), dtype=FIELD_DTYPE)
    for k, child in enumerate(np.random.SeedSequence(int(seed)).spawn(members)):
        rng = np.random.default_rng(child)
        z = np.float32(c * rng.standard_normal())
        if gen is not None:
            z = z + np.float32(sp) * gen.sample(rng).astype(np.float32)
        if nug > 0:
            z = z + np.float32(nug) * rng.standard_normal((ny, nx)).astype(np.float32)
        out[k] = base * np.exp(sig * z - half)  # 0 stays 0, NaN stays NaN
    res = xr.DataArray(
        out,
        dims=("member", ydim, xdim),
        coords={
            "member": np.arange(members),
            **{d: accumulation[d] for d in accumulation.dims},
        },
        name="accumulation_member",
        attrs={
            "units": accumulation.attrs.get("units", ""),
            "long_name": "ensemble of the accumulation under an assumed multiplicative error",
            "sigma_ln": s,
            "shares": " ".join(f"{v:g}" for v in sh),
            "correlation_length": length,
            "seed": int(seed),
        },
    )
    return res


def _fit_exponential(h, sv, var):
    """Nugget, partial sill and range of an exponential variogram (grid search over the range)."""
    edges = np.quantile(h, np.linspace(0, 1, 9))
    centers, gam = [], []
    for lo, hi in itertools.pairwise(edges):
        m = (h >= lo) & (h <= hi)
        if m.sum() >= 3:
            centers.append(h[m].mean())
            gam.append(sv[m].mean())
    if len(centers) < 3:
        return np.nan, np.nan, np.nan
    centers, gam = np.array(centers), np.array(gam)
    best = None
    for a_try in np.geomspace(centers.min() / 2, centers.max() * 4, 80):
        basis = np.column_stack([np.ones_like(centers), 1.0 - np.exp(-centers / a_try)])
        coef, *_ = np.linalg.lstsq(basis, gam, rcond=None)
        coef = np.clip(coef, [0.0, 1e-9], [10 * var, 10 * var])
        err = float(np.sum((basis @ coef - gam) ** 2))
        if best is None or err < best[0]:
            best = (err, float(coef[0]), float(coef[1]), float(a_try))
    return best[1], best[2], best[3]


@dask_support("computes", "it fits one model to all the gauge pairs.")
@provenance("Error model of a rainfall accumulation from gauges")
def qpe_error_model(estimate, gauge, *, x, y, min_value=1.0):
    """
    Parameters of the lognormal error model from radar-gauge pairs.

    With ``e = ln(gauge / radar)`` over the pairs where both are at least
    ``min_value``, the error standard deviation is the root mean square of
    ``e`` (so a bias between gauges and radar is part of the error), the share
    of the variance that is common to the whole field is the squared mean over
    that (the bias), and the rest is split between a spatially correlated part
    and noise from cell to cell by an exponential variogram of the deviations
    (``nugget + sill (1 - exp(-h / L))``, fitted by a grid search over ``L``).

    Parameters
    ----------
    estimate, gauge : array_like or xarray.DataArray
        Accumulations in mm at the gauges: the radar estimate at each gauge and
        the gauge value, 1-D with the same length.
    x, y : array_like
        Gauge positions in metres (any common projected coordinates).
    min_value : float, optional
        Pairs with either value below this are not used (mm). Default 1.

    Returns
    -------
    xarray.Dataset
        Scalars: ``sigma_ln``, ``mean_ln_ratio``, ``std_ln_ratio``,
        ``share_common``, ``share_spatial``, ``share_nugget``,
        ``correlation_length`` (m) and ``n_pairs``. ``sigma_ln`` is NaN with
        fewer than 5 usable pairs, the shares and the length are NaN with
        fewer than 8 (too few for a variogram): the caller then has to say
        which numbers are assumed.

    See Also
    --------
    qpe_error_ensemble

    References
    ----------
    .. [1] Germann, U., M. Berenguer, D. Sempere-Torres, and M. Zappa, 2009:
       REAL: Ensemble radar precipitation estimation for hydrology in a
       mountainous region. *Quart. J. Roy. Meteor. Soc.*, **135**, 445-456.

    Examples
    --------
    >>> m = radarx.retrieve.qpe_error_model(r, g, x=xg, y=yg)  # doctest: +SKIP
    """
    r = np.asarray(getattr(estimate, "values", estimate), dtype=np.float64).ravel()
    g = np.asarray(getattr(gauge, "values", gauge), dtype=np.float64).ravel()
    xs = np.asarray(getattr(x, "values", x), dtype=np.float64).ravel()
    ys = np.asarray(getattr(y, "values", y), dtype=np.float64).ravel()
    if not (r.size == g.size == xs.size == ys.size):
        raise ValueError("estimate, gauge, x and y must have the same length")
    ok = np.isfinite(r) & np.isfinite(g) & (r >= min_value) & (g >= min_value)
    n = int(ok.sum())
    nan = np.nan
    vals = dict(
        sigma_ln=nan,
        mean_ln_ratio=nan,
        std_ln_ratio=nan,
        share_common=nan,
        share_spatial=nan,
        share_nugget=nan,
        correlation_length=nan,
    )
    if n >= MIN_PAIRS:
        e = np.log(g[ok] / r[ok])
        mu, var = float(e.mean()), float(e.var())
        vals.update(
            sigma_ln=math.sqrt(var + mu**2),
            mean_ln_ratio=mu,
            std_ln_ratio=math.sqrt(var),
        )
        if n >= MIN_PAIRS_VARIOGRAM and var > 0:
            res = e - mu
            px, py = xs[ok], ys[ok]
            d = np.hypot(px[:, None] - px[None, :], py[:, None] - py[None, :])
            iu = np.triu_indices(n, 1)
            c0, c, a = _fit_exponential(
                d[iu], 0.5 * (res[:, None] - res[None, :])[iu] ** 2, var
            )
            if np.isfinite(a):
                common = mu**2 / (var + mu**2)
                nug = c0 / (c0 + c)
                vals.update(
                    share_common=common,
                    share_spatial=(1 - common) * (1 - nug),
                    share_nugget=(1 - common) * nug,
                    correlation_length=a,
                )
        elif (
            n >= MIN_PAIRS_VARIOGRAM
        ):  # no scatter at all: everything is the common bias
            vals.update(share_common=1.0, share_spatial=0.0, share_nugget=0.0)
    units = {"correlation_length": "m", "n_pairs": "1"}
    ds = xr.Dataset(
        {
            **{
                k: ((), np.float32(v), {"units": units.get(k, "1")})
                for k, v in vals.items()
            },
            "n_pairs": ((), np.int64(n), {"units": "1"}),
        },
        attrs={"min_value_mm": float(min_value)},
    )
    return ds
