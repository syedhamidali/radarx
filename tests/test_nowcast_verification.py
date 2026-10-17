# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the nowcast verification scores
=========================================

The scores are compared with hand-worked tables, with a brute-force
evaluation of their definitions, with the closed form of the CRPS of a
Gaussian forecast and with ``scipy.ndimage.uniform_filter`` for the fractions.
The compiled kernels are compared with the NumPy implementations.
"""

import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import uniform_filter
from scipy.stats import norm

import radarx  # noqa: F401
from radarx.retrieve import nowcast_verification as nv
from radarx.retrieve.nowcast_verification import (
    brier_score,
    categorical_scores,
    crps,
    fss,
    rank_histogram,
    reliability,
)

from .provenance_helpers import without_provenance

ENGINES = ["numpy"] + (["compiled"] if nv.HAS_COMPILED_KERNEL else [])
needs_kernel = pytest.mark.skipif(
    not nv.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)


def _grid(values, **coords):
    dims = ("t", "y", "x")[-values.ndim :]
    base = {
        "y": np.arange(values.shape[-2]) * 1000.0,
        "x": np.arange(values.shape[-1]) * 1000.0,
    }
    base.update(coords)
    return xr.DataArray(
        values, dims=dims, coords={d: base[d] for d in dims if d in base}
    )


# ---------------------------------------------------------------------------
# fractions skill score
# ---------------------------------------------------------------------------


def test_fss_worked_example():
    """A 1 by 1 cell event displaced by one cell on a 5 by 5 grid."""
    f = np.zeros((5, 5))
    o = np.zeros((5, 5))
    f[2, 2] = 1.0
    o[2, 3] = 1.0
    fc, ob = _grid(f), _grid(o)
    # a 1 cell neighbourhood: no overlap, FSS = 1 - 2 / 2 = 0
    assert float(fss(fc, ob, 0.5, 1e3)) == 0.0
    # 3 by 3 neighbourhood, edges counted as no event: both fractions are 1/9 in
    # the 3 by 4 cells around the pair, they agree on the 3 by 2 cells in
    # common and differ by 1/9 on the 3 + 3 cells that see only one event
    score = fss(fc, ob, 0.5, 3e3, edge="zero")
    sse = 6 * (1 / 9) ** 2
    ref = 9 * 2 * (1 / 9) ** 2
    np.testing.assert_allclose(float(score), 1.0 - sse / (ref + 0.0), rtol=1e-6)
    # a perfect forecast
    assert float(fss(fc, fc, 0.5, 3e3)) == 1.0


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("size", [1, 3, 9])
def test_fss_matches_a_uniform_filter(engine, size):
    rng = np.random.default_rng(size)
    f = rng.random((30, 40)) * 10
    o = rng.random((30, 40)) * 10
    pf = uniform_filter((f >= 5).astype(float), size, mode="constant")
    po = uniform_filter((o >= 5).astype(float), size, mode="constant")
    expect = 1.0 - ((pf - po) ** 2).sum() / ((pf**2).sum() + (po**2).sum())
    got = fss(_grid(f), _grid(o), 5.0, size * 1e3, edge="zero", engine=engine)
    np.testing.assert_allclose(float(got), expect, rtol=1e-6)


@pytest.mark.parametrize("engine", ENGINES)
def test_fss_inside_edge_divides_by_the_cells_inside(engine):
    f = np.ones((4, 4))
    o = np.ones((4, 4))
    o[0, 0] = 0.0
    # with the cells inside, a fraction is 1 wherever the field is all event
    inside = fss(_grid(f), _grid(o), 0.5, 3e3, edge="inside", engine=engine)
    zero = fss(_grid(f), _grid(o), 0.5, 3e3, edge="zero", engine=engine)
    assert 0.0 < float(zero) < 1.0 and 0.0 < float(inside) < 1.0
    assert abs(float(zero) - float(inside)) > 1e-4


@pytest.mark.parametrize("engine", ENGINES)
def test_fss_increases_with_scale_for_a_displaced_forecast(engine):
    f = np.zeros((60, 60))
    o = np.zeros((60, 60))
    f[20:30, 20:30] = 20.0
    o[20:30, 26:36] = 20.0
    scores = fss(_grid(f), _grid(o), 10.0, [1e3, 5e3, 11e3, 25e3], engine=engine)
    assert scores.dims == ("scale",)
    values = scores.values
    assert np.all(np.diff(values) > 0)
    assert values[0] < 0.5 < values[-1]
    np.testing.assert_array_equal(scores["scale"].values, [1e3, 5e3, 11e3, 25e3])


def test_fss_dimensions_and_reduction():
    rng = np.random.default_rng(0)
    f = _grid(rng.random((4, 20, 20)) * 10)
    o = _grid(rng.random((4, 20, 20)) * 10)
    per = fss(f, o, 5.0, 3e3)
    assert per.dims == ("t",) and per.shape == (4,)
    whole = fss(f, o, 5.0, 3e3, dim="t")
    assert whole.dims == ()
    # the sums are added before the division (not the mean of the scores)
    sums = [
        (
            (
                uniform_filter((f.values[k] >= 5) * 1.0, 3, mode="constant")
                - uniform_filter((o.values[k] >= 5) * 1.0, 3, mode="constant")
            )
            ** 2
        ).sum()
        for k in range(4)
    ]
    den = sum(
        (uniform_filter((a.values[k] >= 5) * 1.0, 3, mode="constant") ** 2).sum()
        for a in (f, o)
        for k in range(4)
    )
    expect = 1.0 - sum(sums) / den
    got = fss(f, o, 5.0, 3e3, dim="t", edge="zero")
    np.testing.assert_allclose(float(got), expect, rtol=1e-5)
    # the observation is broadcast over members
    members = xr.concat([f.isel(t=0), f.isel(t=1)], dim="member")
    out = fss(members, o.isel(t=2), 5.0, 3e3)
    assert out.dims == ("member",)


def test_fss_ignores_missing_values_and_reports_no_events():
    f = np.full((10, 10), np.nan)
    o = np.full((10, 10), np.nan)
    assert np.isnan(float(fss(_grid(f), _grid(o), 1.0, 3e3)))
    f = np.zeros((10, 10))
    assert np.isnan(float(fss(_grid(f), _grid(f), 1.0, 3e3)))  # no events at all
    f[5, 5] = 5.0
    g = f.copy()
    g[0, 0] = np.nan  # a missing cell is left out in both
    assert float(fss(_grid(f), _grid(g), 1.0, 3e3)) == 1.0


@needs_kernel
def test_fss_kernel_matches_numpy_and_checks_input():
    from radarx.retrieve import _nowcast_verification as k

    rng = np.random.default_rng(1)
    f = rng.random((5, 25, 30)) * 10
    o = rng.random((5, 25, 30)) * 10
    f[0, 3:6, 4:9] = np.nan
    for edge in (0, 1):
        for hy, hx in [(0, 0), (1, 2), (30, 30)]:
            a = k.fss_sums(f, o, 5.0, hy, hx, edge, 3)
            b = nv._fss_sums_numpy(f, o, 5.0, hy, hx, edge)
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)
    with pytest.raises(ValueError, match="3-D"):
        k.fss_sums(f[0], o[0], 5.0, 1, 1, 0, 1)
    with pytest.raises(ValueError, match="differ in shape"):
        k.fss_sums(f, o[:, :, :20], 5.0, 1, 1, 0, 1)
    with pytest.raises(ValueError, match="negative"):
        k.fss_sums(f, o, 5.0, -1, 1, 0, 1)
    with pytest.raises(ValueError, match="edge"):
        k.fss_sums(f, o, 5.0, 1, 1, 2, 1)


def test_fss_inputs_are_checked():
    f = _grid(np.zeros((10, 10)))
    with pytest.raises(ValueError, match="edge must be"):
        fss(f, f, 1.0, 3e3, edge="wrap")
    with pytest.raises(ValueError, match="scale must be positive"):
        fss(f, f, 1.0, -1.0)
    with pytest.raises(ValueError, match="scale must be positive"):
        fss(f, f, 1.0, np.nan)
    with pytest.raises(ValueError, match="always summed"):
        fss(f, f, 1.0, 3e3, dim="x")
    with pytest.raises(ValueError, match="no dimension"):
        fss(f, f, 1.0, 3e3, dim="t")
    with pytest.raises(ValueError, match="need the dimension"):
        fss(f, f, 1.0, 3e3, x="lon")
    with pytest.raises(ValueError, match="engine must be"):
        fss(f, f, 1.0, 3e3, engine="gpu")
    uneven = f.assign_coords(x=np.r_[0.0, 1.0, 3.0, np.arange(7) + 4.0])
    with pytest.raises(ValueError, match="evenly spaced"):
        fss(uneven, uneven, 1.0, 3e3)
    bare = f.drop_vars("x")
    with pytest.raises(ValueError, match="needs a 'x' coordinate"):
        fss(bare, bare, 1.0, 3e3)
    one = xr.DataArray(
        np.zeros((1, 5)), dims=("y", "x"), coords={"y": [0.0], "x": np.arange(5.0)}
    )
    with pytest.raises(ValueError, match="at least 2"):
        fss(one, one, 1.0, 3e3)
    if not nv.HAS_COMPILED_KERNEL:  # pragma: no cover
        with pytest.raises(ImportError):
            fss(f, f, 1.0, 3e3, engine="compiled")


def test_fss_attributes_and_provenance():
    f = _grid(np.eye(10) * 5)
    out = fss(f, f, 1.0, 3e3)
    assert out.dtype == np.float32
    assert out.attrs["units"] == "1" and out.attrs["threshold"] == 1.0
    assert "10.1175/2007MWR2123.1" in out.attrs["radarx_references"]
    assert without_provenance(out).attrs["long_name"] == "fractions skill score"


# ---------------------------------------------------------------------------
# CRPS
# ---------------------------------------------------------------------------


def _brute_crps(x, y, fair=False):
    m = len(x)
    first = np.abs(x - y).mean()
    second = np.abs(x[:, None] - x[None, :]).sum() / (2 * m * (m - 1 if fair else m))
    return first - second


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("fair", [False, True])
def test_crps_matches_the_definition(engine, fair):
    rng = np.random.default_rng(0)
    ens = rng.normal(size=(7, 30))
    obs = rng.normal(size=30)
    got = crps(
        xr.DataArray(ens, dims=("member", "p")), xr.DataArray(obs, dims="p"),
        fair=fair, engine=engine,
    )  # fmt: skip
    expect = [_brute_crps(ens[:, i], obs[i], fair) for i in range(30)]
    np.testing.assert_allclose(got.values, expect, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("engine", ENGINES)
def test_crps_of_a_gaussian_ensemble(engine):
    """Members at the quantiles of a Gaussian give its closed-form CRPS."""
    m = 500
    quantiles = norm.ppf((np.arange(m) + 0.5) / m)
    ens = np.repeat((2.0 + 1.5 * quantiles)[:, None], 400, axis=1)
    obs = np.random.default_rng(1).normal(0.0, 1.0, size=400)
    got = crps(
        xr.DataArray(ens, dims=("member", "p")), xr.DataArray(obs, dims="p"),
        fair=True, engine=engine,
    )  # fmt: skip
    z = (obs - 2.0) / 1.5
    exact = 1.5 * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    assert np.abs(got.values - exact).max() < 0.01


@pytest.mark.parametrize("engine", ENGINES)
def test_crps_special_cases(engine):
    ens = xr.DataArray(
        [[1.0, 3.0], [2.0, np.nan], [np.nan, np.nan]], dims=("p", "member")
    )
    obs = xr.DataArray([2.0, 5.0, 1.0], dims="p")
    out = crps(ens, obs, engine=engine).values
    # two members: mean(|1-2|, |3-2|) - |1-3| / 4 = 1 - 0.5
    np.testing.assert_allclose(out[0], 0.5)
    np.testing.assert_allclose(out[1], 3.0)  # one valid member: the absolute error
    assert np.isnan(out[2])
    fair = crps(ens, obs, fair=True, engine=engine).values
    # fair: 1 - 2 / 2 = 0 for the same two members
    np.testing.assert_allclose(fair[0], 0.0, atol=1e-7)
    assert np.isnan(fair[1]) and np.isnan(fair[2])
    missing = crps(ens, xr.DataArray([np.nan, 1.0, 1.0], dims="p"), engine=engine)
    assert np.isnan(missing.values[0])


def test_crps_dims_units_and_deterministic():
    rng = np.random.default_rng(0)
    ens = xr.DataArray(
        rng.random((5, 3, 4)), dims=("member", "t", "p"), attrs={"units": "dBZ"}
    )
    obs = xr.DataArray(rng.random((3, 4)), dims=("t", "p"))
    full = crps(ens, obs)
    assert full.dims == ("t", "p") and full.attrs["units"] == "dBZ"
    mean = crps(ens, obs, dim=["t", "p"])
    np.testing.assert_allclose(float(mean), float(full.mean()), rtol=1e-5)
    one = crps(ens, obs, dim="t")
    assert one.dims == ("p",)
    det = crps(ens.isel(member=0, drop=True), obs)
    np.testing.assert_allclose(
        det.values, np.abs(ens.isel(member=0).values - obs.values), rtol=1e-6
    )
    with pytest.raises(ValueError, match="fair=True needs"):
        crps(ens.isel(member=0, drop=True), obs, fair=True)
    with pytest.raises(ValueError, match="no dimension"):
        crps(ens, obs, dim="z")
    # an observation with an extra dimension is broadcast
    obs_t = xr.DataArray(rng.random(2), dims="k")
    assert crps(ens, obs_t).dims[-1] == "k" or "k" in crps(ens, obs_t).dims
    # no units
    plain = crps(xr.DataArray(rng.random((3, 4)), dims=("member", "p")), obs.isel(t=0))
    assert "units" not in plain.attrs
    assert (
        "10.1175/1520-0434(2000)015<0559:DOTCRP>2.0.CO;2"
        in full.attrs["radarx_references"]
    )


@needs_kernel
def test_crps_kernel_matches_numpy_and_checks_input():
    from radarx.retrieve import _nowcast_verification as k

    rng = np.random.default_rng(2)
    ens = rng.normal(size=(100, 9))
    ens[::7, 2] = np.nan
    ens[3] = np.nan
    obs = rng.normal(size=100)
    obs[5] = np.nan
    for fair in (0, 1):
        np.testing.assert_allclose(
            k.crps_ensemble(ens, obs, fair, 2), nv._crps_numpy(ens, obs, bool(fair)),
            rtol=1e-12, equal_nan=True,
        )  # fmt: skip
    with pytest.raises(ValueError, match="2-D"):
        k.crps_ensemble(ens[0], obs, 0, 1)
    with pytest.raises(ValueError, match="one value per point"):
        k.crps_ensemble(ens, obs[:5], 0, 1)


# ---------------------------------------------------------------------------
# contingency table and probability scores
# ---------------------------------------------------------------------------


def test_categorical_scores_worked_example():
    # a = 3 hits, b = 2 false alarms, c = 1 miss, d = 4 correct negatives
    fc = xr.DataArray([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dims="p")
    ob = xr.DataArray([1, 1, 1, 0, 0, 1, 0, 0, 0, 0], dims="p")
    t = categorical_scores(fc, ob, 0.5)
    assert [
        int(t[k]) for k in ("hits", "false_alarms", "misses", "correct_negatives")
    ] == [3, 2, 1, 4]
    a, b, c, d, n = 3.0, 2.0, 1.0, 4.0, 10.0
    a_r = (a + b) * (a + c) / n
    np.testing.assert_allclose(float(t.pod), a / (a + c), rtol=1e-6)
    np.testing.assert_allclose(float(t.far), b / (a + b), rtol=1e-6)
    np.testing.assert_allclose(float(t.csi), a / (a + b + c), rtol=1e-6)
    np.testing.assert_allclose(float(t.bias), (a + b) / (a + c), rtol=1e-6)
    np.testing.assert_allclose(float(t.ets), (a - a_r) / (a + b + c - a_r), rtol=1e-6)
    np.testing.assert_allclose(
        float(t.hss),
        2 * (a * d - b * c) / ((a + c) * (c + d) + (a + b) * (b + d)),
        rtol=1e-6,
    )
    assert t.attrs["threshold"] == 0.5


def test_categorical_scores_dims_missing_and_empty():
    rng = np.random.default_rng(0)
    fc = xr.DataArray(rng.random((3, 50)), dims=("t", "p"))
    ob = xr.DataArray(rng.random((3, 50)), dims=("t", "p"))
    ob[0, 0] = np.nan
    whole = categorical_scores(fc, ob, 0.5)
    per_time = categorical_scores(fc, ob, 0.5, dim="p")
    assert per_time.hits.dims == ("t",)
    assert int(whole.hits) == int(per_time.hits.sum())
    total = sum(
        int(whole[k]) for k in ("hits", "false_alarms", "misses", "correct_negatives")
    )
    assert total == 149  # the missing pair is left out
    none = categorical_scores(fc * 0, ob * 0, 0.5)
    assert np.isnan(float(none.pod)) and np.isnan(float(none.ets))
    with pytest.raises(ValueError, match="no dimension"):
        categorical_scores(fc, ob, 0.5, dim="z")
    assert (
        "10.1175/1520-0493(1987)115<1330:AGFFFV>2.0.CO;2"
        in whole.attrs.get("radarx_references", "")
        or "radarx_references" in whole.hits.attrs
    )


def test_brier_score():
    p = xr.DataArray([0.0, 0.5, 1.0, 0.2], dims="k")
    o = xr.DataArray([0, 1, 1, 0], dims="k")
    np.testing.assert_allclose(
        float(brier_score(p, o)), (0 + 0.25 + 0 + 0.04) / 4, rtol=1e-6
    )
    # constant climatological forecast: f (1 - f)
    f = 0.3
    rng = np.random.default_rng(0)
    events = xr.DataArray(rng.random(20000) < f, dims="k")
    bs = brier_score(
        xr.DataArray(np.full(20000, events.mean().item()), dims="k"), events
    )
    np.testing.assert_allclose(float(bs), f * (1 - f), atol=0.01)
    # missing outcomes are left out; dim keeps the others
    o2 = xr.DataArray([[0, 1, np.nan, 0], [0, 1, 1, 0]], dims=("t", "k"))
    out = brier_score(p, o2, dim="k")
    assert out.dims == ("t",)
    np.testing.assert_allclose(out.values[0], (0 + 0.25 + 0.04) / 3, rtol=1e-6)
    with pytest.raises(ValueError, match="between 0 and 1"):
        brier_score(p + 1.0, o)
    with pytest.raises(ValueError, match="no dimension"):
        brier_score(p, o, dim="z")


def test_reliability_of_a_calibrated_forecast():
    rng = np.random.default_rng(0)
    p = rng.random(40000)
    event = rng.random(40000) < p
    table = reliability(
        xr.DataArray(p, dims="k"), xr.DataArray(event, dims="k"), bins=10
    )
    assert table.sizes["bin"] == 10
    np.testing.assert_allclose(
        table.observed_frequency, table.forecast_probability, atol=0.04
    )
    assert int(table["count"].sum()) == 40000
    np.testing.assert_allclose(table["bin"], (np.arange(10) + 0.5) / 10)
    # empty bins
    sparse = reliability(
        xr.DataArray([0.0, 1.0], dims="k"), xr.DataArray([0, 1], dims="k"), bins=4
    )
    assert np.isnan(sparse.forecast_probability.values[1])
    assert list(sparse["count"].values) == [1, 0, 0, 1]
    assert float(sparse.observed_frequency[3]) == 1.0
    with pytest.raises(ValueError, match="bins"):
        reliability(xr.DataArray([0.5], dims="k"), xr.DataArray([1], dims="k"), bins=0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        reliability(xr.DataArray([1.5], dims="k"), xr.DataArray([1], dims="k"))


def test_rank_histogram():
    rng = np.random.default_rng(0)
    members, n = 9, 30000
    ens = xr.DataArray(rng.normal(size=(members, n)), dims=("member", "p"))
    flat = xr.DataArray(rng.normal(size=n), dims="p")
    hist = rank_histogram(ens, flat, seed=0)
    assert hist.sizes["rank"] == members + 1 and int(hist.sum()) == n
    assert hist.values.std() / hist.values.mean() < 0.1  # reliable: flat
    # under-dispersive: the observation falls outside more often
    wide = xr.DataArray(3.0 * rng.normal(size=n), dims="p")
    u_shape = rank_histogram(ens, wide, seed=0).values
    assert (
        u_shape[0] > 3 * u_shape[members // 2]
        and u_shape[-1] > 3 * u_shape[members // 2]
    )
    # ties are spread over the tied ranks
    tied = xr.DataArray(np.zeros((4, 1000)), dims=("member", "p"))
    spread = rank_histogram(tied, xr.DataArray(np.zeros(1000), dims="p"), seed=1).values
    assert spread.min() > 100
    # missing values are left out
    ens2 = ens.copy()
    ens2[0, :10] = np.nan
    assert int(rank_histogram(ens2, flat).sum()) == n - 10
    assert int(rank_histogram(ens, flat.where(flat.p >= 5)).sum()) == n - 5
    with pytest.raises(ValueError, match="no dimension"):
        rank_histogram(ens, flat, member_dim="e")
