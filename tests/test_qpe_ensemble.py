# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the ensemble of the error of a rainfall accumulation
==============================================================

Known answers: the mean and the variance of a lognormal factor on a constant
field, the correlation of the spatial part at one correlation length, the
seeds, and the recovery of the parameters of a model from gauges that were
drawn from it. The check against gauges uses the scores that radarx already has.
"""

import math

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.retrieve import (
    brier_score,
    exceedance_probability,
    qpe_error_ensemble,
    qpe_error_model,
    reliability,
)
from radarx.retrieve import qpe_ensemble as qe

CELL = 1000.0


def _acc(values, cell=CELL):
    values = np.asarray(values, dtype=np.float32)
    ny, nx = values.shape
    return xr.DataArray(
        values,
        dims=("y", "x"),
        coords={"y": np.arange(ny) * cell, "x": np.arange(nx) * cell},
        attrs={"units": "mm"},
    )


def test_mean_and_variance_of_a_lognormal_factor_on_a_constant_field():
    s = 0.4
    ens = qpe_error_ensemble(
        _acc(np.full((40, 40), 20.0)),
        sigma_ln=s,
        correlation_length=6e3,
        shares=(0.0, 0.7, 0.3),
        members=300,
        seed=7,
    )
    assert ens.dims == ("member", "y", "x") and ens.dtype == np.float32
    assert float(ens.mean()) == pytest.approx(
        20.0, rel=0.02
    )  # the mean is the estimate
    assert float(ens.var()) == pytest.approx(400.0 * (math.exp(s**2) - 1.0), rel=0.08)
    assert float(np.log(ens).std()) == pytest.approx(s, rel=0.04)
    assert ens.attrs["units"] == "mm" and ens.attrs["seed"] == 7


def test_geometric_mean_is_lower_by_exp_minus_half_s_squared():
    # depths are averaged linearly: the mean of the logarithms is lower
    s = 0.5
    ens = qpe_error_ensemble(
        _acc(np.full((30, 30), 10.0)),
        sigma_ln=s,
        correlation_length=4e3,
        shares=(0.0, 0.5, 0.5),
        members=300,
        seed=1,
    )
    geo = float(np.exp(np.log(ens).mean()))
    assert geo == pytest.approx(10.0 * math.exp(-(s**2) / 2), rel=0.02)


def test_dry_cells_stay_dry_and_missing_cells_stay_missing():
    base = np.full((20, 20), 5.0)
    base[:5] = 0.0
    base[5:8] = np.nan
    ens = qpe_error_ensemble(
        _acc(base), sigma_ln=0.4, correlation_length=5e3, members=10
    )
    assert (ens.values[:, :5] == 0).all()
    assert np.isnan(ens.values[:, 5:8]).all() and np.isfinite(ens.values[:, 8:]).all()


def test_spatial_part_has_unit_variance_and_the_exponential_correlation():
    gen = qe._CorrelatedField((48, 48), 6.0)
    rng = np.random.default_rng(3)
    fields = np.stack([gen.sample(rng) for _ in range(300)])
    assert float(fields.var()) == pytest.approx(1.0, rel=0.06)
    lag = 6
    c = float(np.mean(fields[:, :, :-lag] * fields[:, :, lag:]))
    assert c == pytest.approx(math.exp(-1.0), abs=0.04)


def test_scale_changes_the_error_cell_by_cell():
    base = _acc(np.full((40, 40), 20.0))
    w = np.ones((40, 40))
    w[:, 20:] = 3.0
    ens = qpe_error_ensemble(
        base,
        sigma_ln=0.2,
        correlation_length=3e3,
        shares=(0, 0, 1),
        members=200,
        scale=w,
    )
    left, right = (
        np.log(ens.values[:, :, :20]).std(),
        np.log(ens.values[:, :, 20:]).std(),
    )
    assert right == pytest.approx(3.0 * left, rel=0.1)


def test_seeds_are_reproducible_and_members_are_stable():
    base = _acc(np.full((16, 16), 8.0))
    kw = dict(sigma_ln=0.3, correlation_length=4e3)
    a = qpe_error_ensemble(base, members=6, seed=42, **kw)
    b = qpe_error_ensemble(base, members=6, seed=42, **kw)
    c = qpe_error_ensemble(base, members=9, seed=42, **kw)
    d = qpe_error_ensemble(base, members=6, seed=43, **kw)
    assert a.equals(b)
    assert np.array_equal(a.values, c.values[:6])
    assert not np.array_equal(a.values, d.values)


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"sigma_ln": 0.0}, "sigma_ln"),
        ({"sigma_ln": 5.0}, "sigma_ln"),
        ({"shares": (0.5, 0.5, 0.5)}, "add up to 1"),
        ({"members": 1}, "members"),
        ({"correlation_length": -1.0}, "correlation_length"),
    ],
)
def test_wrong_arguments(kw, match):
    args = dict(sigma_ln=0.3, correlation_length=5e3, members=3)
    args.update(kw)
    with pytest.raises(ValueError, match=match):
        qpe_error_ensemble(_acc(np.ones((8, 8))), **args)


def test_irregular_grid_and_wrong_dimension_are_refused():
    a = _acc(np.ones((8, 8)))
    bad = a.assign_coords(x=np.cumsum(np.r_[1, np.arange(1, 8)]) * 1000.0)
    with pytest.raises(ValueError, match="regularly spaced"):
        qpe_error_ensemble(bad, sigma_ln=0.3, correlation_length=5e3)
    with pytest.raises(ValueError, match="2-D"):
        qpe_error_ensemble(a.expand_dims(t=2), sigma_ln=0.3, correlation_length=5e3)


def test_provenance_is_written():
    ens = qpe_error_ensemble(
        _acc(np.ones((8, 8))), sigma_ln=0.3, correlation_length=5e3
    )
    assert "ensemble" in ens.attrs["radarx_method"].lower()
    assert "10.1137/S1064827592240555" in ens.attrs["radarx_references"]


def _pairs(n, sigma, bias=1.0, seed=0):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 1e5, n), rng.uniform(0, 1e5, n)
    radar = rng.uniform(5, 60, n)
    gauge = radar * bias * np.exp(rng.standard_normal(n) * sigma)
    return radar, gauge, x, y


def test_error_model_recovers_sigma_and_reports_the_bias():
    r, g, x, y = _pairs(500, 0.35, seed=1)
    m = qpe_error_model(r, g, x=x, y=y)
    assert float(m.sigma_ln) == pytest.approx(0.35, rel=0.08)
    assert abs(float(m.mean_ln_ratio)) < 0.05 and int(m.n_pairs) == 500
    r, g, x, y = _pairs(500, 0.2, bias=1.3, seed=2)
    m = qpe_error_model(r, g, x=x, y=y)
    assert float(m.sigma_ln) == pytest.approx(math.hypot(0.2, math.log(1.3)), rel=0.08)
    # the bias is the common part of the variance
    assert float(m.share_common) == pytest.approx(
        math.log(1.3) ** 2 / float(m.sigma_ln) ** 2, rel=0.1
    )
    assert float(m.share_common + m.share_spatial + m.share_nugget) == pytest.approx(
        1.0, abs=1e-5
    )


def test_error_model_says_what_it_cannot_give_with_few_pairs():
    r, g, x, y = _pairs(3, 0.3)
    m = qpe_error_model(r, g, x=x, y=y)
    assert np.isnan(float(m.sigma_ln)) and int(m.n_pairs) == 3
    r, g, x, y = _pairs(6, 0.3)
    m = qpe_error_model(r, g, x=x, y=y)
    assert np.isfinite(float(m.sigma_ln)) and np.isnan(float(m.share_common))
    with pytest.raises(ValueError, match="same length"):
        qpe_error_model(r[:3], g, x=x, y=y)


def test_synthetic_case_with_known_truth_is_reliable_and_a_too_narrow_one_is_not():
    """Radar = truth / E with E from a known model; gauges read the truth. The ensemble built with the
    parameters fitted from the gauges gives reliable exceedance probabilities."""
    ny = nx = 60
    yy, xx = np.mgrid[0:ny, 0:nx]
    truth = np.clip(25.0 + 20.0 * np.sin(xx / 9.0) * np.cos(yy / 11.0), 2.0, None)
    err = qpe_error_ensemble(
        _acc(np.ones((ny, nx))),
        sigma_ln=0.35,
        correlation_length=8e3,
        shares=(0.0, 0.8, 0.2),
        members=2,
        seed=123,
    ).values[0]
    radar = _acc(truth / err)
    rng = np.random.default_rng(11)
    gi, gj = rng.integers(2, nx - 2, 600), rng.integers(2, ny - 2, 600)
    gauge = truth[gj, gi]
    m = qpe_error_model(radar.values[gj, gi], gauge, x=gi * CELL, y=gj * CELL)
    sigma = float(m.sigma_ln)
    assert sigma == pytest.approx(0.35, rel=0.25)

    def check(s):
        ens = qpe_error_ensemble(
            radar,
            sigma_ln=s,
            correlation_length=float(m.correlation_length),
            shares=(
                float(m.share_common),
                float(m.share_spatial),
                float(m.share_nugget),
            ),
            members=100,
            seed=3,
        )
        probs, events = [], []
        for t in (15.0, 25.0, 35.0):
            p = exceedance_probability(ens, t)
            probs.append(p.values[gj, gi])
            events.append((gauge >= t).astype(float))
        p = xr.DataArray(np.concatenate(probs))
        o = xr.DataArray(np.concatenate(events))
        table = reliability(p, o, bins=10)
        ok = (table["count"] >= 20).values
        fp, of, w = (
            table["forecast_probability"].values[ok],
            table["observed_frequency"].values[ok],
            table["count"].values[ok],
        )
        xm, ym = np.average(fp, weights=w), np.average(of, weights=w)
        slope = float(np.sum(w * (fp - xm) * (of - ym)) / np.sum(w * (fp - xm) ** 2))
        bs = float(brier_score(p, o))
        base = float(o.mean())
        return slope, 1.0 - bs / (base * (1.0 - base))

    slope, skill = check(sigma)
    assert 0.8 < slope < 1.2 and skill > 0.3
    narrow_slope, _ = check(sigma / 2.0)
    assert narrow_slope < slope - 0.05  # half the spread is overconfident
