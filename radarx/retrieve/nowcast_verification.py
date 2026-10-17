# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Nowcast Verification
====================

Scores for deterministic and ensemble forecasts of gridded fields, for the
output of :func:`radarx.retrieve.nowcast` and of any other forecast on the
same kind of grid.

* :func:`fss`: the fractions skill score (Roberts and Lean 2008), a
  neighbourhood score that does not punish a displacement smaller than the
  neighbourhood.
* :func:`crps`: the continuous ranked probability score of an ensemble
  (Matheson and Winkler 1976; Hersbach 2000; Gneiting and Raftery 2007), and
  its fair version (Ferro 2014).
* :func:`categorical_scores`: the scores of the 2 by 2 contingency table of a
  threshold (POD, FAR, CSI, bias, ETS, HSS).
* :func:`brier_score`, :func:`reliability` and :func:`rank_histogram`: the
  Brier score (Brier 1950), the reliability diagram (Murphy and Winkler 1987)
  and the rank histogram (Hamill 2001) of probability forecasts and ensembles.

The fractions skill score and the CRPS run in compiled, multithreaded C++
kernels (with a NumPy equivalent): the fractions come from summed-area tables,
so the cost does not depend on the size of the neighbourhood, and the CRPS of
m members costs O(m log m) per point instead of O(m**2).

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = [
    "fss",
    "crps",
    "categorical_scores",
    "brier_score",
    "reliability",
    "rank_histogram",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np
import xarray as xr

from .._dtypes import FIELD_DTYPE
from .._provenance import provenance
from .._sweepwise import dask_support

try:
    from . import _nowcast_verification

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _nowcast_verification = None
    HAS_COMPILED_KERNEL = False

_EDGES = {"inside": 0, "zero": 1}


def _use_compiled(engine):
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError(  # pragma: no cover - depends on the build
            "the compiled verification kernel is not available"
        )
    return HAS_COMPILED_KERNEL and engine != "numpy"


def _spacing(da, dim):
    """Grid spacing (m) of a coordinate; 1 for a dimension without one."""
    if dim not in da.coords:
        raise ValueError(f"the field needs a {dim!r} coordinate")
    c = np.asarray(da[dim].values, dtype=np.float64)
    if c.size < 2:
        raise ValueError(f"{dim!r} needs at least 2 points")
    d = np.diff(c)
    if not np.allclose(d, d[0], rtol=1e-6, atol=0.0) or d[0] == 0:
        raise ValueError(f"{dim!r} must be evenly spaced")
    return abs(float(d[0]))


# ---------------------------------------------------------------------------
# fractions skill score
# ---------------------------------------------------------------------------


def _integral(a):
    """Summed-area table of arrays (n, ny, nx) with a leading row and column of 0."""
    s = np.zeros((a.shape[0], a.shape[1] + 1, a.shape[2] + 1))
    s[:, 1:, 1:] = a.cumsum(axis=1).cumsum(axis=2)
    return s


def _box_sum(s, hy, hx):
    """Sum over the (clipped) square of half widths ``hy``, ``hx`` of every cell."""
    ny, nx = s.shape[1] - 1, s.shape[2] - 1
    i = np.arange(ny)
    j = np.arange(nx)
    i0, i1 = np.clip(i - hy, 0, ny), np.clip(i + hy + 1, 0, ny)
    j0, j1 = np.clip(j - hx, 0, nx), np.clip(j + hx + 1, 0, nx)
    return (
        s[:, i1][:, :, j1]
        - s[:, i0][:, :, j1]
        - s[:, i1][:, :, j0]
        + s[:, i0][:, :, j0]
    )


def _fss_sums_numpy(fc, ob, threshold, hy, hx, edge):
    """NumPy version of the kernel ``fss_sums``: (n, 4) sums."""
    ok = np.isfinite(fc) & np.isfinite(ob)
    ef = (ok & (fc >= threshold)).astype(np.float64)
    eo = (ok & (ob >= threshold)).astype(np.float64)
    ev = ok.astype(np.float64)
    n_f = _box_sum(_integral(ef), hy, hx)
    n_o = _box_sum(_integral(eo), hy, hx)
    if edge == 0:
        area = _box_sum(_integral(ev), hy, hx)
    else:
        area = np.full(ef.shape, (2.0 * hy + 1.0) * (2.0 * hx + 1.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        pf, po = n_f / area, n_o / area
    pf, po = np.where(ok, pf, 0.0), np.where(ok, po, 0.0)
    return np.stack(
        [
            ((pf - po) ** 2).sum(axis=(1, 2)),
            (pf**2).sum(axis=(1, 2)),
            (po**2).sum(axis=(1, 2)),
            ev.sum(axis=(1, 2)),
        ],
        axis=1,
    )


def _half_width(scale, step):
    """Half width in cells of a square neighbourhood of side ``scale`` (m)."""
    return max(0, int(round((scale / step - 1.0) / 2.0)))


@dask_support("computes", "it needs the whole fields to build the neighbourhoods.")
@provenance("Fractions skill score, after Roberts and Lean (2008)")
def fss(
    forecast,
    observed,
    threshold,
    scale,
    *,
    dim=None,
    edge="inside",
    x="x",
    y="y",
    n_threads=None,
    engine="auto",
):
    """
    Fractions skill score of a forecast for a threshold and a scale.

    For every cell the fraction of cells at or above ``threshold`` in the
    square neighbourhood of side ``scale`` is computed for the forecast
    (``F``) and the observation (``O``). The score compares the fractions
    (Roberts and Lean 2008 [1]_)::

        FSS = 1 - sum (F - O)**2 / (sum F**2 + sum O**2)

    It is 1 for a perfect forecast, 0 when the forecast and the observed
    events never overlap in the neighbourhoods, and rises with the scale for
    a displaced forecast (Ebert 2008 [2]_). The sums run over the cells of the
    grid and over the dimensions in ``dim``: the numerators and denominators
    are added before the division, so a time series is scored as a whole, not
    as the mean of daily scores.

    Cells where the forecast or the observation is NaN are left out. The side
    of the neighbourhood is rounded to an odd number of cells. ``edge``
    decides how the fraction is taken at the border of the grid and next to
    missing cells: ``"inside"`` (default) divides by the number of valid cells
    of the neighbourhood, ``"zero"`` by the whole neighbourhood, so that what
    lies outside counts as no event. The fractions come from a summed-area
    table (a compiled kernel with a NumPy equivalent), so the cost does not
    depend on ``scale``.

    Parameters
    ----------
    forecast, observed : xarray.DataArray
        Fields on ``(y, x)``, possibly with more dimensions (``lead_time``,
        ``member``, ...). ``observed`` is broadcast against ``forecast``.
        Their ``x`` and ``y`` coordinates are evenly spaced, in metres.
    threshold : float
        Event threshold, in the units of the fields.
    scale : float or sequence of float
        Side of the neighbourhood in metres. A sequence adds a ``scale``
        dimension to the result.
    dim : str or sequence of str, optional
        Dimensions (besides ``x`` and ``y``) over which the sums run, e.g.
        ``"time"``. Default: none, one score for each of the other dimensions.
    edge : {"inside", "zero"}, optional
        Treatment of the border. Default ``"inside"``.
    x, y : str, optional
        Names of the horizontal coordinates. Default ``"x"`` and ``"y"``.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use.

    Returns
    -------
    xarray.DataArray
        The score (``float32``, dimensionless, NaN where no cell has an event
        in either field), on the dimensions of ``forecast`` that are not
        reduced, with a ``scale`` dimension for a sequence of scales.

    Raises
    ------
    ValueError
        For an unknown ``edge``, a negative scale, a grid that is not regular
        or fields that do not share their dimensions.
    ImportError
        If ``engine="compiled"`` and the kernel is not available.

    See Also
    --------
    crps, categorical_scores

    References
    ----------
    .. [1] Roberts, N. M., and H. W. Lean, 2008: Scale-selective verification
       of rainfall accumulations from high-resolution forecasts of convective
       events. *Mon. Wea. Rev.*, **136**, 78-97,
       https://doi.org/10.1175/2007MWR2123.1
    .. [2] Ebert, E. E., 2008: Fuzzy verification of high-resolution gridded
       forecasts: a review and proposed framework. *Meteor. Appl.*, **15**,
       51-64, https://doi.org/10.1002/met.25

    Examples
    --------
    >>> score = radarx.retrieve.fss(ens, truth, 35.0, [5e3, 20e3])  # doctest: +SKIP
    """
    use_compiled = _use_compiled(engine)
    if edge not in _EDGES:
        raise ValueError(f"edge must be 'inside' or 'zero', not {edge!r}")
    scales = np.atleast_1d(np.asarray(scale, dtype=np.float64))
    if scales.ndim != 1 or np.any(scales <= 0) or not np.all(np.isfinite(scales)):
        raise ValueError("scale must be positive")
    reduce = [dim] if isinstance(dim, str) else list(dim or [])
    forecast, observed = xr.broadcast(forecast, observed)
    for d in (x, y):
        if d not in forecast.dims:
            raise ValueError(f"the fields need the dimension {d!r}")
    if x in reduce or y in reduce:
        raise ValueError("x and y are always summed: leave them out of dim")
    for d in reduce:
        if d not in forecast.dims:
            raise ValueError(f"the fields have no dimension {d!r}")
    dy, dx = _spacing(forecast, y), _spacing(forecast, x)
    other = [d for d in forecast.dims if d not in (x, y)]
    fc = forecast.transpose(*other, y, x)
    ob = observed.transpose(*other, y, x)
    shape = fc.shape[:-2]
    f3 = np.ascontiguousarray(fc.values.reshape(-1, *fc.shape[-2:]), dtype=np.float64)
    o3 = np.ascontiguousarray(ob.values.reshape(-1, *ob.shape[-2:]), dtype=np.float64)
    keep = [d for d in other if d not in reduce]
    axes = tuple(other.index(d) for d in reduce)
    results = []
    for s in scales:
        hy, hx = _half_width(s, dy), _half_width(s, dx)
        if use_compiled:
            sums = _nowcast_verification.fss_sums(
                f3, o3, float(threshold), hy, hx, _EDGES[edge], int(n_threads or 0)
            )
        else:
            sums = _fss_sums_numpy(f3, o3, float(threshold), hy, hx, _EDGES[edge])
        sums = sums.reshape(*shape, 4)
        if axes:
            sums = sums.sum(axis=axes)
        with np.errstate(invalid="ignore", divide="ignore"):
            denom = sums[..., 1] + sums[..., 2]
            score = np.where(denom > 0, 1.0 - sums[..., 0] / denom, np.nan)
        results.append(score)
    values = np.stack(results, axis=-1) if len(scales) > 1 else results[0]
    coords = {d: forecast[d] for d in keep if d in forecast.coords}
    dims = list(keep)
    if len(scales) > 1:
        dims.append("scale")
        coords["scale"] = ("scale", scales, {"units": "m"})
    out = xr.DataArray(
        np.asarray(values).astype(FIELD_DTYPE),
        dims=dims,
        coords=coords,
        name="fss",
        attrs={
            "units": "1",
            "long_name": "fractions skill score",
            "threshold": float(threshold),
        },
    )
    return out


# ---------------------------------------------------------------------------
# CRPS
# ---------------------------------------------------------------------------


def _crps_numpy(ens, obs, fair):
    """NumPy version of the kernel ``crps_ensemble``; ens (n, m), obs (n)."""
    m = ens.shape[1]
    valid = ~np.isnan(ens)
    c = valid.sum(axis=1)
    x = np.sort(ens, axis=1)  # NaN last
    rank = np.arange(1, m + 1)[None, :]
    xs = np.where(valid.sum(axis=1)[:, None] >= rank, x, 0.0)
    abs_err = np.abs(np.where(np.isnan(x), 0.0, x - obs[:, None])).sum(axis=1)
    spread = ((2.0 * rank - c[:, None] - 1.0) * xs).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = c * (c - 1.0) if fair else c * c.astype(np.float64)
        out = abs_err / c - spread / denom
    bad = np.isnan(obs) | (c == 0) | ((c < 2) if fair else False)
    return np.where(bad, np.nan, out)


@dask_support("computes", "it sorts the members of every point.")
@provenance("Ensemble continuous ranked probability score, after Hersbach (2000)")
def crps(
    forecast,
    observed,
    *,
    member_dim="member",
    fair=False,
    dim=None,
    n_threads=None,
    engine="auto",
):
    """
    Continuous ranked probability score of an ensemble forecast.

    For the empirical distribution of the members ``x_i`` and the observation
    ``y`` the score is the energy form (Gneiting and Raftery 2007 [3]_, who
    show it equals the integral of the squared difference of the forecast and
    the observed distribution functions, the CRPS of Matheson and Winkler
    1976 [1]_ and Hersbach 2000 [2]_)::

        CRPS = mean_i |x_i - y| - 1/(2 m**2) sum_ij |x_i - x_j|

    With the members sorted, the sum over all pairs is a weighted sum of the
    sorted values, so the cost is O(m log m) per point. With ``fair=True``
    the second term is divided by ``2 m (m - 1)`` instead, the score that
    does not depend on the size of the ensemble in expectation (Ferro 2014
    [4]_). A deterministic forecast (no member dimension) has the CRPS ``|x -
    y|``, the absolute error.

    NaN members are left out; a point is NaN where the observation is NaN or
    no member is valid.

    Parameters
    ----------
    forecast : xarray.DataArray
        Ensemble with a dimension ``member_dim``, or a deterministic forecast.
    observed : xarray.DataArray
        Observation, broadcast against the forecast.
    member_dim : str, optional
        Name of the member dimension. Default ``"member"``.
    fair : bool, optional
        Use the fair CRPS. Default False.
    dim : str or sequence of str, optional
        Dimensions to average the score over (NaN left out). Default: none, the
        score of every point.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use.

    Returns
    -------
    xarray.DataArray
        The score (``float32``), in the units of the forecast, on the
        dimensions of the forecast without ``member_dim`` (less ``dim``).

    Raises
    ------
    ValueError
        If ``dim`` names a dimension the forecast does not have, or
        ``fair=True`` for a forecast without members.
    ImportError
        If ``engine="compiled"`` and the kernel is not available.

    See Also
    --------
    fss, brier_score

    References
    ----------
    .. [1] Matheson, J. E., and R. L. Winkler, 1976: Scoring rules for
       continuous probability distributions. *Management Science*, **22**,
       1087-1096, https://doi.org/10.1287/mnsc.22.10.1087
    .. [2] Hersbach, H., 2000: Decomposition of the continuous ranked
       probability score for ensemble prediction systems. *Wea. Forecasting*,
       **15**, 559-570,
       https://doi.org/10.1175/1520-0434(2000)015<0559:DOTCRP>2.0.CO;2
    .. [3] Gneiting, T., and A. E. Raftery, 2007: Strictly proper scoring
       rules, prediction, and estimation. *J. Amer. Stat. Assoc.*, **102**,
       359-378, https://doi.org/10.1198/016214506000001437
    .. [4] Ferro, C. A. T., 2014: Fair scores for ensemble forecasts. *Quart.
       J. Roy. Meteor. Soc.*, **140**, 1917-1923,
       https://doi.org/10.1002/qj.2270

    Examples
    --------
    >>> score = radarx.retrieve.crps(ens, truth, dim=["x", "y"])  # doctest: +SKIP
    """
    use_compiled = _use_compiled(engine)
    reduce = [dim] if isinstance(dim, str) else list(dim or [])
    if member_dim in forecast.dims:
        ref = forecast.isel({member_dim: 0}, drop=True)
        ref, obs = xr.broadcast(ref, observed)
        other = list(ref.dims)
        members = forecast.broadcast_like(ref).transpose(*other, member_dim)
        e2 = np.ascontiguousarray(
            members.values.reshape(-1, members.shape[-1]), dtype=np.float64
        )
        o1 = np.ascontiguousarray(
            obs.transpose(*other).values.reshape(-1), dtype=np.float64
        )
        if use_compiled:
            flat = _nowcast_verification.crps_ensemble(
                e2, o1, int(bool(fair)), int(n_threads or 0)
            )
        else:
            flat = _crps_numpy(e2, o1, bool(fair))
        shape = obs.transpose(*other).shape
        template = obs.transpose(*other)
        values = flat.reshape(shape)
    else:
        if fair:
            raise ValueError("fair=True needs an ensemble")
        fc, obs = xr.broadcast(forecast, observed)
        values = np.abs(fc.values.astype(np.float64) - obs.values)
        template = fc
    for d in reduce:
        if d not in template.dims:
            raise ValueError(f"the forecast has no dimension {d!r}")
    out = xr.DataArray(
        values.astype(FIELD_DTYPE),
        dims=template.dims,
        coords={k: c for k, c in template.coords.items()},
        name="crps",
        attrs={
            "units": forecast.attrs.get("units", ""),
            "long_name": "continuous ranked probability score",
        },
    )
    if not out.attrs["units"]:
        del out.attrs["units"]
    if reduce:
        out = out.mean(reduce, skipna=True, keep_attrs=True)
    return out


# ---------------------------------------------------------------------------
# contingency table, probability scores
# ---------------------------------------------------------------------------


def _reduce_dims(obj, dim):
    dims = [dim] if isinstance(dim, str) else list(dim or [])
    for d in dims:
        if d not in obj.dims:
            raise ValueError(f"the field has no dimension {d!r}")
    return dims


@dask_support("computes", "it counts the events of the whole fields.")
@provenance("Contingency-table scores for a threshold, after Murphy and Winkler (1987)")
def categorical_scores(forecast, observed, threshold, *, dim=None):
    """
    Scores of the 2 by 2 contingency table of a threshold.

    An event is a value at or above ``threshold``. From the counts of hits
    (``a``), false alarms (``b``), misses (``c``) and correct negatives
    (``d``), summed over all dimensions in ``dim`` (Murphy and Winkler 1987
    [1]_ give the framework for such forecast-observation tables)::

        POD  = a / (a + c)               probability of detection
        FAR  = b / (a + b)               false alarm ratio
        CSI  = a / (a + b + c)           critical success index
        bias = (a + b) / (a + c)         frequency bias
        ETS  = (a - a_r) / (a + b + c - a_r),  a_r = (a + b)(a + c) / n
        HSS  = 2 (a d - b c) / ((a + c)(c + d) + (a + b)(b + d))

    These are the standard definitions of the textbooks; a ratio with a zero
    denominator is NaN. Cells where the forecast or the observation is NaN are
    left out. For an ensemble use :func:`exceedance_probability` and
    :func:`brier_score`.

    Parameters
    ----------
    forecast, observed : xarray.DataArray
        Fields; ``observed`` is broadcast against ``forecast``.
    threshold : float
        Event threshold in the units of the fields.
    dim : str or sequence of str, optional
        Dimensions to sum the counts over. Default: all of them.

    Returns
    -------
    xarray.Dataset
        ``hits``, ``false_alarms``, ``misses``, ``correct_negatives`` (counts)
        and ``pod``, ``far``, ``csi``, ``bias``, ``ets``, ``hss``.

    Raises
    ------
    ValueError
        If ``dim`` names a dimension the fields do not have.

    See Also
    --------
    fss, crps

    References
    ----------
    .. [1] Murphy, A. H., and R. L. Winkler, 1987: A general framework for
       forecast verification. *Mon. Wea. Rev.*, **115**, 1330-1338,
       https://doi.org/10.1175/1520-0493(1987)115<1330:AGFFFV>2.0.CO;2

    Examples
    --------
    >>> table = radarx.retrieve.categorical_scores(fc, obs, 35.0)  # doctest: +SKIP
    """
    forecast, observed = xr.broadcast(forecast, observed)
    dims = _reduce_dims(forecast, dim) if dim is not None else list(forecast.dims)
    ok = forecast.notnull() & observed.notnull()
    f, o = (forecast >= threshold) & ok, (observed >= threshold) & ok
    counts = {
        "hits": (f & o).sum(dims),
        "false_alarms": (f & ~o).sum(dims),
        "misses": (~f & o).sum(dims),
        "correct_negatives": (ok & ~f & ~o).sum(dims),
    }
    a, b, c, d = (
        counts[k].astype(np.float64)
        for k in ("hits", "false_alarms", "misses", "correct_negatives")
    )
    n = a + b + c + d

    def ratio(num, den):
        return (num / den.where(den > 0)).astype(FIELD_DTYPE)

    a_r = (a + b) * (a + c) / n.where(n > 0)
    scores = {
        "pod": ratio(a, a + c),
        "far": ratio(b, a + b),
        "csi": ratio(a, a + b + c),
        "bias": ratio(a + b, a + c),
        "ets": ratio(a - a_r, a + b + c - a_r),
        "hss": ratio(2.0 * (a * d - b * c), (a + c) * (c + d) + (a + b) * (b + d)),
    }
    units = {"pod": "1", "far": "1", "csi": "1", "bias": "1", "ets": "1", "hss": "1"}
    names = {
        "pod": "probability of detection",
        "far": "false alarm ratio",
        "csi": "critical success index",
        "bias": "frequency bias",
        "ets": "equitable threat score",
        "hss": "Heidke skill score",
    }
    ds = xr.Dataset(counts)
    for k in counts:
        ds[k].attrs = {"units": "1", "long_name": k.replace("_", " ")}
    for k, v in scores.items():
        ds[k] = v
        ds[k].attrs = {"units": units[k], "long_name": names[k]}
    ds.attrs["threshold"] = float(threshold)
    return ds


@dask_support("computes", "it averages over the whole fields.")
@provenance("Brier score, after Brier (1950)")
def brier_score(probability, event, *, dim=None):
    """
    Brier score of probability forecasts of an event.

    The mean squared difference between the forecast probability ``p`` and
    the outcome ``o`` (1 if the event happened, else 0), Brier (1950) [1]_::

        BS = mean (p - o)**2

    0 is perfect; always forecasting the observed frequency ``f`` gives
    ``f (1 - f)``. NaN values are left out.

    Parameters
    ----------
    probability : xarray.DataArray
        Probabilities between 0 and 1, e.g. from
        :func:`exceedance_probability`.
    event : xarray.DataArray
        Outcome: boolean or 0 and 1 (NaN: missing). Broadcast against
        ``probability``.
    dim : str or sequence of str, optional
        Dimensions to average over. Default: all of them.

    Returns
    -------
    xarray.DataArray
        The score (``float32``, dimensionless).

    Raises
    ------
    ValueError
        If a probability lies outside 0 to 1 or ``dim`` is not a dimension.

    See Also
    --------
    reliability, crps

    References
    ----------
    .. [1] Brier, G. W., 1950: Verification of forecasts expressed in terms
       of probability. *Mon. Wea. Rev.*, **78**, 1-3,
       https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2

    Examples
    --------
    >>> bs = radarx.retrieve.brier_score(prob, truth >= 35.0)  # doctest: +SKIP
    """
    p, o = xr.broadcast(probability, event.astype(np.float64).where(event.notnull()))
    if bool(((p < 0) | (p > 1)).any()):
        raise ValueError("probabilities must lie between 0 and 1")
    dims = _reduce_dims(p, dim) if dim is not None else list(p.dims)
    out = ((p - o) ** 2).mean(dims, skipna=True).astype(FIELD_DTYPE)
    out.name = "brier_score"
    out.attrs = {"units": "1", "long_name": "Brier score"}
    return out


@dask_support("computes", "it counts over the whole fields.")
@provenance("Reliability diagram, after Murphy and Winkler (1987)")
def reliability(probability, event, *, bins=10):
    """
    Reliability diagram: observed frequency for binned forecast probability.

    The pairs of forecast probability and outcome are sorted into ``bins``
    equal intervals of probability, and for each the mean forecast
    probability and the observed relative frequency are given, the
    calibration-refinement view of Murphy and Winkler (1987) [1]_. A reliable
    forecast lies on the diagonal. All dimensions are pooled. NaN pairs are
    left out.

    Parameters
    ----------
    probability : xarray.DataArray
        Probabilities between 0 and 1.
    event : xarray.DataArray
        Outcome: boolean or 0 and 1; broadcast against ``probability``.
    bins : int, optional
        Number of probability intervals. Default 10.

    Returns
    -------
    xarray.Dataset
        On the dimension ``bin``: ``forecast_probability`` (mean probability
        in the bin), ``observed_frequency`` and ``count`` (NaN, NaN and 0 for
        an empty bin); ``bin`` holds the centres of the intervals.

    Raises
    ------
    ValueError
        If ``bins`` is less than 1 or a probability lies outside 0 to 1.

    See Also
    --------
    brier_score

    References
    ----------
    .. [1] Murphy, A. H., and R. L. Winkler, 1987: A general framework for
       forecast verification. *Mon. Wea. Rev.*, **115**, 1330-1338,
       https://doi.org/10.1175/1520-0493(1987)115<1330:AGFFFV>2.0.CO;2

    Examples
    --------
    >>> table = radarx.retrieve.reliability(prob, truth >= 35.0)  # doctest: +SKIP
    """
    if int(bins) != bins or bins < 1:
        raise ValueError("bins must be a positive integer")
    bins = int(bins)
    p, o = xr.broadcast(probability, event.astype(np.float64).where(event.notnull()))
    p, o = p.values.ravel(), o.values.ravel()
    ok = np.isfinite(p) & np.isfinite(o)
    p, o = p[ok], o[ok]
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must lie between 0 and 1")
    index = np.minimum((p * bins).astype(int), bins - 1)
    count = np.bincount(index, minlength=bins)
    sum_p = np.bincount(index, weights=p, minlength=bins)
    sum_o = np.bincount(index, weights=o, minlength=bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_p = np.where(count > 0, sum_p / count, np.nan)
        freq = np.where(count > 0, sum_o / count, np.nan)
    centres = (np.arange(bins) + 0.5) / bins
    return xr.Dataset(
        {
            "forecast_probability": (
                "bin",
                mean_p.astype(FIELD_DTYPE),
                {"units": "1", "long_name": "mean forecast probability"},
            ),
            "observed_frequency": (
                "bin",
                freq.astype(FIELD_DTYPE),
                {"units": "1", "long_name": "observed relative frequency"},
            ),
            "count": ("bin", count, {"units": "1", "long_name": "number of pairs"}),
        },
        coords={"bin": ("bin", centres, {"units": "1"})},
    )


@dask_support("computes", "it ranks the observation among the members everywhere.")
@provenance("Rank histogram, after Hamill (2001)")
def rank_histogram(ensemble, observed, *, member_dim="member", seed=None):
    """
    Rank histogram (Talagrand diagram) of an ensemble.

    The rank of the observation among the sorted members, from 0 (below all)
    to ``m`` (above all), is counted over all points. A flat histogram is
    necessary for a reliable ensemble; a U shape means too little spread, a
    dome too much (Hamill 2001 [1]_). Where the observation equals some members
    (the no-echo value, for one) its rank is drawn at random among the tied
    positions, so that ties do not fill the lowest rank. Points where the
    observation or a member is NaN are left out.

    Parameters
    ----------
    ensemble : xarray.DataArray
        Forecast with a member dimension.
    observed : xarray.DataArray
        Observation, broadcast against the ensemble without the members.
    member_dim : str, optional
        Name of the member dimension. Default ``"member"``.
    seed : int, optional
        Seed of the random numbers for the ties.

    Returns
    -------
    xarray.DataArray
        Counts on the dimension ``rank`` (0 to ``m``).

    Raises
    ------
    ValueError
        If the ensemble has no dimension ``member_dim``.

    See Also
    --------
    crps, reliability

    References
    ----------
    .. [1] Hamill, T. M., 2001: Interpretation of rank histograms for
       verifying ensemble forecasts. *Mon. Wea. Rev.*, **129**, 550-560,
       https://doi.org/10.1175/1520-0493(2001)129<0550:IORHFV>2.0.CO;2

    Examples
    --------
    >>> hist = radarx.retrieve.rank_histogram(ens, truth)  # doctest: +SKIP
    """
    if member_dim not in ensemble.dims:
        raise ValueError(f"the ensemble has no dimension {member_dim!r}")
    ref = ensemble.isel({member_dim: 0}, drop=True)
    obs = xr.broadcast(ref, observed)[1]
    members = ensemble.broadcast_like(ref).transpose(*ref.dims, member_dim)
    e = members.values.reshape(-1, members.shape[-1])
    o = obs.transpose(*ref.dims).values.reshape(-1)
    ok = np.isfinite(o) & np.isfinite(e).all(axis=1)
    e, o = e[ok], o[ok]
    below = (e < o[:, None]).sum(axis=1)
    tied = (e == o[:, None]).sum(axis=1)
    rng = np.random.default_rng(seed)
    rank = below + np.floor(rng.random(o.size) * (tied + 1)).astype(int)
    m = members.shape[-1]
    counts = np.bincount(rank, minlength=m + 1)
    return xr.DataArray(
        counts,
        dims="rank",
        coords={"rank": ("rank", np.arange(m + 1))},
        name="rank_histogram",
        attrs={"units": "1", "long_name": "rank histogram counts"},
    )
