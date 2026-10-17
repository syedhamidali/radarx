#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the radar-gauge merging
=================================

The kriging is checked against closed-form results (the weights of a small
system written out as a covariance matrix, the exact reproduction of a linear
relation by the external drift, the exact reproduction of the gauges by
conditional merging), the compiled kernels against the NumPy implementation,
and, when wradlib is installed, against its kriging.
"""
import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.retrieve import gauge as gaugemod
from radarx.retrieve import gauge_adjust, gauge_variogram, radar_at_gauges

from .provenance_helpers import without_provenance

ENGINES = ["numpy"] + (["compiled"] if gaugemod.HAS_COMPILED_KERNEL else [])
compiled_only = pytest.mark.skipif(
    not gaugemod.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)
KRIGING = ["ok", "conditional", "ked"]


def truth_field(x, y):
    """A smooth rain field with two cells (mm)."""
    xx, yy = np.meshgrid(x, y)
    return 6.0 * np.exp(
        -((xx - 30e3) ** 2 + (yy - 24e3) ** 2) / (2 * 9e3**2)
    ) + 3.0 * np.exp(-((xx - 10e3) ** 2 + (yy - 45e3) ** 2) / (2 * 6e3**2))


def case(n=40, ng=25, seed=0, bias=0.7, noise=0.1):
    """A radar field (biased and noisy), the truth and gauges that sample the truth."""
    rng = np.random.default_rng(seed)
    x = np.arange(n) * 1000.0
    y = np.arange(n) * 1000.0
    truth = truth_field(x, y)
    radar = xr.DataArray(
        truth * bias * (1 + noise * rng.standard_normal(truth.shape)),
        dims=("y", "x"),
        coords={"x": ("x", x), "y": ("y", y)},
        attrs={"units": "mm"},
        name="RAIN",
    )
    gx = rng.uniform(0.5e3, (n - 1.5) * 1e3, ng)
    gy = rng.uniform(0.5e3, (n - 1.5) * 1e3, ng)
    values = np.array(
        [truth[int(round(b / 1000)), int(round(a / 1000))] for a, b in zip(gx, gy)]
    )
    gauges = xr.DataArray(
        values,
        dims="station",
        coords={"x": ("station", gx), "y": ("station", gy)},
        attrs={"units": "mm"},
    )
    return radar, gauges, truth


def covariance_ok(gx, gy, values, tx, ty, nugget, psill, rng, model="exponential"):
    """Ordinary kriging in the covariance form, solved densely (independent of the code)."""

    def cov(h):
        if model == "exponential":
            c = psill * np.exp(-3 * h / rng)
        elif model == "gaussian":
            c = psill * np.exp(-3 * (h / rng) ** 2)
        else:  # spherical
            c = psill * (
                1 - 1.5 * np.minimum(h / rng, 1) + 0.5 * np.minimum(h / rng, 1) ** 3
            )
        return np.where(h > 0, c, nugget + psill)

    m = len(gx)
    h = np.hypot(gx[:, None] - gx[None], gy[:, None] - gy[None])
    a = np.ones((m + 1, m + 1))
    a[:m, :m] = cov(h)
    a[m, m] = 0.0
    out, var = [], []
    for x0, y0 in zip(tx, ty):
        b = np.ones(m + 1)
        b[:m] = cov(np.hypot(gx - x0, gy - y0))
        sol = np.linalg.solve(a, b)
        out.append(sol[:m] @ values)
        # variance: C(0) - sum w C_i0 - mu
        var.append(nugget + psill - sol[:m] @ b[:m] - sol[m])
    return np.array(out), np.array(var)


# --------------------------------------------------------------------------
# kriging against closed forms
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("model", ["exponential", "spherical", "gaussian"])
def test_ordinary_kriging_matches_the_covariance_form(engine, model):
    rng = np.random.default_rng(4)
    gx, gy = rng.uniform(0, 30e3, 12), rng.uniform(0, 30e3, 12)
    values = rng.uniform(0, 10, 12)
    tx, ty = rng.uniform(0, 30e3, 9), rng.uniform(0, 30e3, 9)
    gauges = xr.DataArray(values, dims="s", coords={"x": ("s", gx), "y": ("s", gy)})
    target = xr.DataArray(
        np.zeros((3, 3)),
        dims=("y", "x"),
        coords={"x": ("x", tx[:3]), "y": ("y", ty[:3])},
    )
    # nine arbitrary targets: a 2-D layout with 2-D coordinates
    target = xr.DataArray(
        np.ones((1, 9)),
        dims=("a", "b"),
        coords={"x": (("a", "b"), tx[None]), "y": (("a", "b"), ty[None])},
    )
    par = {"model": model, "nugget": 0.4, "partial_sill": 3.0, "range": 20e3}
    res = gauge_adjust(
        target,
        gauges,
        method="ok",
        variogram=par,
        n_neighbours=None,
        engine=engine,
        clip=False,
    )
    want, var = covariance_ok(gx, gy, values, tx, ty, 0.4, 3.0, 20e3, model)
    np.testing.assert_allclose(res.RAIN_ADJUSTED.values.ravel(), want, rtol=1e-5)
    np.testing.assert_allclose(
        res.RAIN_ADJUSTED_STD.values.ravel() ** 2, var, rtol=1e-4, atol=1e-6
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_kriging_is_exact_at_the_gauges(engine):
    gx = np.array([1e3, 7e3, 4e3, 9e3, 2e3, 6e3])
    gy = np.array([2e3, 1e3, 8e3, 6e3, 5e3, 9e3])
    values = np.array([1.0, 4.0, 2.5, 0.0, 3.0, 5.0])
    gauges = xr.DataArray(values, dims="s", coords={"x": ("s", gx), "y": ("s", gy)})
    target = xr.DataArray(
        np.ones((1, 6)),
        dims=("a", "b"),
        coords={"x": (("a", "b"), gx[None]), "y": (("a", "b"), gy[None])},
    )
    res = gauge_adjust(
        target,
        gauges,
        method="ok",
        n_neighbours=None,
        min_pairs=5,
        variogram={
            "model": "spherical",
            "nugget": 0.5,
            "partial_sill": 2.0,
            "range": 8e3,
        },
        engine=engine,
        clip=False,
    )
    np.testing.assert_allclose(res.RAIN_ADJUSTED.values.ravel(), values, atol=1e-8)
    np.testing.assert_allclose(res.RAIN_ADJUSTED_STD.values.ravel(), 0.0, atol=1e-5)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("neighbours", [None, 8])
def test_external_drift_reproduces_a_linear_relation(engine, neighbours):
    radar, gauges, _ = case(ng=30)
    at = radar_at_gauges(radar, gauges, radius=500.0)
    exact = xr.DataArray(2.0 + 1.5 * at.values, dims="station", coords=gauges.coords)
    res = gauge_adjust(
        radar,
        exact,
        method="ked",
        radius=500.0,
        n_neighbours=neighbours,
        engine=engine,
        clip=False,
    )
    np.testing.assert_allclose(
        res.RAIN_ADJUSTED.values, 2.0 + 1.5 * radar.values, rtol=1e-4, atol=1e-4
    )
    assert res.attrs["ked_fallback"] == 0


@pytest.mark.parametrize("engine", ENGINES)
def test_conditional_merging_honours_the_gauges(engine):
    radar, gauges, _ = case(ng=15)
    res = gauge_adjust(
        radar,
        gauges,
        method="conditional",
        radius=400.0,
        n_neighbours=None,
        engine=engine,
        clip=False,
    )
    ix = np.rint(gauges.x.values / 1000).astype(int)
    iy = np.rint(gauges.y.values / 1000).astype(int)
    # the radar value at the nearest pixel of a gauge off the pixel centre is a
    # different pixel from the gauge, so compare against that pixel's definition:
    # merged = gauge + radar(pixel) - kriged radar(pixel)
    merged = res.RAIN_ADJUSTED.values[iy, ix]
    assert np.all(np.isfinite(merged))
    # on pixel centres the merged field equals the gauge exactly
    centred = xr.DataArray(
        gauges.values,
        dims="station",
        coords={"x": ("station", ix * 1000.0), "y": ("station", iy * 1000.0)},
    )
    res = gauge_adjust(
        radar,
        centred,
        method="conditional",
        radius=400.0,
        n_neighbours=None,
        engine=engine,
        clip=False,
    )
    np.testing.assert_allclose(
        res.RAIN_ADJUSTED.values[iy, ix], gauges.values, atol=1e-5
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_constant_gauges_give_a_constant_field(engine):
    radar, gauges, _ = case()
    const = gauges.copy(data=np.full(gauges.shape, 2.5))
    for method in ("ok", "ked"):
        res = gauge_adjust(radar, const, method=method, engine=engine)
        np.testing.assert_allclose(res.RAIN_ADJUSTED.values, 2.5, rtol=1e-6)
        assert np.all(res.RAIN_ADJUSTED_STD.values == 0.0)
    res = gauge_adjust(radar, const, method="conditional", engine=engine, radius=500.0)
    assert np.isfinite(res.RAIN_ADJUSTED.values).all()


@pytest.mark.parametrize("engine", ENGINES)
def test_dry_radar_falls_back_to_ordinary_kriging(engine):
    radar, gauges, _ = case()
    dry = radar.copy(data=np.zeros(radar.shape))
    ked = gauge_adjust(dry, gauges, method="ked", engine=engine, n_neighbours=None)
    ok = gauge_adjust(dry, gauges, method="ok", engine=engine, n_neighbours=None)
    np.testing.assert_allclose(
        ked.RAIN_ADJUSTED.values, ok.RAIN_ADJUSTED.values, rtol=1e-5
    )
    assert ked.attrs["ked_fallback"] == radar.size
    local = gauge_adjust(dry, gauges, method="ked", engine=engine, n_neighbours=6)
    assert local.attrs["ked_fallback"] == radar.size


# --------------------------------------------------------------------------
# the compiled kernel against NumPy
# --------------------------------------------------------------------------


@compiled_only
@pytest.mark.parametrize("method", KRIGING)
@pytest.mark.parametrize("neighbours", [None, 7])
@pytest.mark.parametrize("model", ["linear", "spherical", "exponential", "gaussian"])
def test_engines_agree(method, neighbours, model):
    radar, gauges, _ = case(n=24, ng=18, seed=2)
    kw = {"method": method, "n_neighbours": neighbours, "model": model, "radius": 800.0}
    a = gauge_adjust(radar, gauges, engine="numpy", **kw)
    b = gauge_adjust(radar, gauges, engine="compiled", **kw)
    np.testing.assert_allclose(
        a.RAIN_ADJUSTED.values, b.RAIN_ADJUSTED.values, rtol=2e-5, atol=2e-5
    )
    np.testing.assert_allclose(
        a.RAIN_ADJUSTED_STD.values, b.RAIN_ADJUSTED_STD.values, rtol=1e-4, atol=1e-4
    )


@compiled_only
@pytest.mark.parametrize("method", ["mfb", "rda", "brandes"])
def test_engines_agree_for_the_factors(method):
    radar, gauges, _ = case(n=24, ng=18, seed=5)
    a = gauge_adjust(radar, gauges, method=method, engine="numpy", min_pairs=5)
    b = gauge_adjust(radar, gauges, method=method, engine="compiled", min_pairs=5)
    np.testing.assert_allclose(
        a.RAIN_ADJUSTED.values, b.RAIN_ADJUSTED.values, rtol=1e-5, atol=1e-6
    )


@compiled_only
def test_engines_agree_on_the_variogram():
    rng = np.random.default_rng(1)
    gx, gy = rng.uniform(0, 5e4, 40), rng.uniform(0, 5e4, 40)
    smooth = 3.0 + 2.0 * np.sin(gx / 1.2e4) * np.cos(gy / 1.5e4)
    da = xr.DataArray(
        smooth[None] * np.array([[1.0], [1.5], [0.7]]) + rng.normal(0, 0.2, (3, 40)),
        dims=("time", "s"),
        coords={"x": ("s", gx), "y": ("s", gy)},
    )
    a = gauge_variogram(da, engine="numpy", n_lags=8)
    b = gauge_variogram(da, engine="compiled", n_lags=8)
    for name in ("semivariance", "distance", "pairs"):
        np.testing.assert_allclose(
            a[name].values, b[name].values, rtol=1e-10, err_msg=name
        )
    # the fit amplifies rounding differences of the sums
    for name in ("nugget", "partial_sill", "range"):
        np.testing.assert_allclose(
            a[name].values, b[name].values, rtol=1e-5, atol=1e-8, err_msg=name
        )


@compiled_only
def test_thread_counts_agree():
    radar, gauges, _ = case(n=30, ng=20, seed=6)
    one = gauge_adjust(radar, gauges, method="ked", n_threads=1)
    many = gauge_adjust(radar, gauges, method="ked", n_threads=4)
    xr.testing.assert_allclose(one.RAIN_ADJUSTED, many.RAIN_ADJUSTED)


def test_compiled_engine_without_kernel(monkeypatch):
    monkeypatch.setattr(gaugemod, "HAS_COMPILED_KERNEL", False)
    radar, gauges, _ = case(n=12, ng=8)
    with pytest.raises(ImportError):
        gauge_adjust(radar, gauges, engine="compiled")
    res = gauge_adjust(radar, gauges, method="ked", engine="auto", min_pairs=5)
    assert np.isfinite(res.RAIN_ADJUSTED.values).all()


# --------------------------------------------------------------------------
# the simple methods by hand
# --------------------------------------------------------------------------


def line_case():
    """Four gauges on a radar field of 1 mm on a 10 x 10 grid."""
    x = y = np.arange(10) * 1000.0
    radar = xr.DataArray(
        np.ones((10, 10)),
        dims=("y", "x"),
        coords={"x": x, "y": y},
        attrs={"units": "mm"},
    )
    gauges = xr.DataArray(
        [2.0, 3.0, 1.0, 4.0, 0.0, 2.0],
        dims="s",
        coords={
            "x": ("s", [1e3, 3e3, 5e3, 7e3, 2e3, 8e3]),
            "y": ("s", [1e3, 6e3, 3e3, 8e3, 5e3, 2e3]),
        },
    )
    return radar, gauges


@pytest.mark.parametrize("engine", ENGINES)
def test_mean_field_bias_by_hand(engine):
    radar, gauges = line_case()
    res = gauge_adjust(
        radar, gauges, method="mfb", min_pairs=3, engine=engine, radius=400.0
    )
    # pairs with both values positive: the gauge with 0 mm is left out
    assert float(res.MEAN_FIELD_BIAS) == pytest.approx(12.0 / 5.0)
    np.testing.assert_allclose(res.RAIN_ADJUSTED.values, 12.0 / 5.0, rtol=1e-6)
    assert int(res.GAUGES_USED) == 5
    # with min_value 1 only the gauges above 1 mm count, and the radar (1 mm) must also exceed it
    none = gauge_adjust(
        radar,
        gauges,
        method="mfb",
        min_pairs=2,
        min_value=1.0,
        engine=engine,
        radius=400.0,
    )
    assert int(none.GAUGES_USED) == 0
    np.testing.assert_array_equal(none.RAIN_ADJUSTED.values, radar.values)
    assert np.isnan(float(none.MEAN_FIELD_BIAS))


@pytest.mark.parametrize("engine", ENGINES)
def test_brandes_weights_by_hand(engine):
    radar, gauges = line_case()
    res = gauge_adjust(
        radar, gauges, method="brandes", k=4e6, min_pairs=3, engine=engine, radius=400.0
    )
    gx, gy, gv = gauges.x.values, gauges.y.values, gauges.values
    ok = gv > 0
    for ix, iy in [(0, 0), (5, 5), (9, 9), (3, 7)]:
        d2 = (gx[ok] - ix * 1e3) ** 2 + (gy[ok] - iy * 1e3) ** 2
        w = np.exp(-d2 / 4e6)
        want = (w * gv[ok]).sum() / w.sum()
        assert res.ADJUSTMENT_FACTOR.values[iy, ix] == pytest.approx(want, rel=1e-5)
        assert res.RAIN_ADJUSTED.values[iy, ix] == pytest.approx(want, rel=1e-5)
    # the default k is 1 / (2 density) of the 9 km x 9 km domain
    auto = gauge_adjust(
        radar, gauges, method="brandes", min_pairs=3, engine=engine, radius=400.0
    )
    k = 9e3 * 9e3 / (2 * ok.sum())
    d2 = (gx[ok] - 4e3) ** 2 + (gy[ok] - 4e3) ** 2
    w = np.exp(-d2 / k)
    assert auto.ADJUSTMENT_FACTOR.values[4, 4] == pytest.approx(
        (w * gv[ok]).sum() / w.sum(), rel=1e-5
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_brandes_far_from_the_gauges_does_not_underflow(engine):
    radar, gauges = line_case()
    res = gauge_adjust(
        radar, gauges, method="brandes", k=1e3, min_pairs=3, engine=engine, radius=400.0
    )
    assert np.isfinite(res.ADJUSTMENT_FACTOR.values).all()
    # at the nearest gauge the weight dominates: the factor is that gauge's
    assert res.ADJUSTMENT_FACTOR.values[1, 1] == pytest.approx(2.0, rel=1e-6)


def test_range_dependent_adjustment_recovers_the_polynomial():
    n = 60
    x = y = (np.arange(n) - n // 2) * 1000.0  # radar at the centre
    xx, yy = np.meshgrid(x, y)
    r_km = np.hypot(xx, yy) / 1000.0
    truth = np.full((n, n), 5.0)
    bias = np.exp(-0.001 * r_km**2 + 0.02 * r_km - 0.4)  # G / R
    radar = xr.DataArray(
        truth / bias, dims=("y", "x"), coords={"x": x, "y": y}, attrs={"units": "mm"}
    )
    rng = np.random.default_rng(0)
    ix, iy = rng.integers(0, n, 40), rng.integers(0, n, 40)
    gauges = xr.DataArray(
        np.full(40, 5.0),
        dims="s",
        coords={"x": ("s", x[ix]), "y": ("s", y[iy])},
    )
    for engine in ENGINES:
        res = gauge_adjust(
            radar, gauges, method="rda", radius=100.0, min_pairs=10, engine=engine
        )
        gauge_r = np.hypot(x[ix], y[iy]) / 1000.0
        inside = (r_km >= gauge_r.min()) & (r_km <= gauge_r.max())
        np.testing.assert_allclose(res.RAIN_ADJUSTED.values[inside], 5.0, rtol=1e-4)
        assert int(res.GAUGES_USED) == 40


def test_range_dependent_adjustment_with_too_few_pairs_leaves_the_radar():
    radar, gauges, _ = case(n=20, ng=10)
    res = gauge_adjust(radar, gauges, method="rda", min_value=100.0, radius=800.0)
    assert int(res.GAUGES_USED) == 0
    assert np.isnan(res.ADJUSTMENT_FACTOR.values).all()
    np.testing.assert_allclose(res.RAIN_ADJUSTED.values, radar.values, rtol=1e-6)


# --------------------------------------------------------------------------
# shapes, times, missing data, coordinates
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method", ["mfb", "rda", "brandes", "ok", "conditional", "ked"]
)
def test_series_of_times_is_one_call_and_matches_single_calls(method):
    radar, gauges, truth = case(n=24, ng=16, seed=7)
    scales = np.array([0.5, 1.0, 1.5])
    series = xr.concat([radar * s for s in scales], dim="time").assign_coords(
        time=[0, 1, 2]
    )
    gseries = xr.concat(
        [gauges * s for s in (0.9, 1.0, 1.2)], dim="time"
    ).assign_coords(time=[0, 1, 2])
    gseries[1, 3] = np.nan  # a missing gauge in one step: different gauge set
    both = gauge_adjust(series, gseries, method=method, min_pairs=5, radius=800.0)
    assert both.RAIN_ADJUSTED.dims == series.dims
    assert both.GAUGES_USED.dims == ("time",)
    for t in range(3):
        one = gauge_adjust(
            series.isel(time=t),
            gseries.isel(time=t),
            method=method,
            min_pairs=5,
            radius=800.0,
        )
        np.testing.assert_allclose(
            both.RAIN_ADJUSTED.isel(time=t).values,
            one.RAIN_ADJUSTED.values,
            rtol=1e-4,
            atol=1e-5,
        )
    assert int(both.GAUGES_USED[1]) < int(both.GAUGES_USED[0])


def test_dimension_order_and_coordinates_are_kept():
    radar, gauges, _ = case(n=16, ng=10)
    series = radar.expand_dims(time=[3, 4]).transpose("y", "time", "x")
    gseries = gauges.expand_dims(time=[3, 4])
    res = gauge_adjust(series, gseries, method="mfb", min_pairs=3, radius=800.0)
    assert res.RAIN_ADJUSTED.dims == ("y", "time", "x")
    assert res.RAIN_ADJUSTED.dtype == np.float32
    assert res.RAIN_ADJUSTED.attrs["units"] == "mm"
    np.testing.assert_array_equal(res.time.values, [3, 4])
    np.testing.assert_array_equal(res.x.values, radar.x.values)


def test_input_is_not_changed():
    radar, gauges, _ = case(n=16, ng=10)
    r0, g0 = radar.copy(deep=True), gauges.copy(deep=True)
    gauge_adjust(radar, gauges, method="ked", radius=800.0)
    xr.testing.assert_identical(radar, r0)
    xr.testing.assert_identical(gauges, g0)


@pytest.mark.parametrize("method", ["mfb", "ok", "conditional", "ked", "brandes"])
def test_nan_pixels_stay_nan_and_too_few_gauges_leave_the_radar(method):
    radar, gauges, _ = case(n=20, ng=12)
    holes = radar.copy()
    holes[3:6, 4:9] = np.nan
    res = gauge_adjust(holes, gauges, method=method, radius=800.0)
    assert np.isnan(res.RAIN_ADJUSTED.values[3:6, 4:9]).all()
    assert np.isfinite(res.RAIN_ADJUSTED.values[10:, 10:]).all()
    few = gauges.copy()
    few[3:] = np.nan  # 3 gauges left
    res = gauge_adjust(radar, few, method=method, min_pairs=5, radius=800.0)
    assert int(res.GAUGES_USED) == 0
    np.testing.assert_allclose(
        res.RAIN_ADJUSTED.values, np.maximum(radar.values, 0.0), rtol=1e-6
    )


def test_clip_removes_negative_values():
    radar, gauges, _ = case(n=20, ng=12)
    neg = gauge_adjust(radar, gauges, method="conditional", clip=False, radius=800.0)
    pos = gauge_adjust(radar, gauges, method="conditional", clip=True, radius=800.0)
    assert (pos.RAIN_ADJUSTED.values >= 0).all()
    np.testing.assert_allclose(
        pos.RAIN_ADJUSTED.values, np.maximum(neg.RAIN_ADJUSTED.values, 0.0), atol=1e-6
    )


def test_geographic_gauges_and_a_sweep_layout():
    radar, gauges, _ = case(n=20, ng=12)
    lat0, lon0 = 14.0, 79.0
    import pyproj

    tr = pyproj.Transformer.from_crs(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m",
        "EPSG:4326",
        always_xy=True,
    )
    lon, lat = tr.transform(gauges.x.values, gauges.y.values)
    geo = xr.DataArray(
        gauges.values,
        dims="station",
        coords={"lon": ("station", lon), "lat": ("station", lat)},
    )
    base = gauge_adjust(radar, gauges, method="ked", radius=800.0)
    res = gauge_adjust(radar, geo, method="ked", radius=800.0, site=(lat0, lon0))
    np.testing.assert_allclose(
        res.RAIN_ADJUSTED.values, base.RAIN_ADJUSTED.values, rtol=1e-4, atol=1e-4
    )
    with_site = radar.assign_coords(latitude=lat0, longitude=lon0)
    res = gauge_adjust(with_site, geo, method="mfb", radius=800.0)
    assert np.isfinite(res.RAIN_ADJUSTED.values).all()
    with pytest.raises(ValueError, match="site"):
        gauge_adjust(radar, geo, method="mfb")
    # both in lon/lat
    xx, yy = np.meshgrid(radar.x.values, radar.y.values)
    rlon, rlat = tr.transform(xx, yy)
    radar_geo = xr.DataArray(
        radar.values,
        dims=("y", "x"),
        coords={"lon": (("y", "x"), rlon), "lat": (("y", "x"), rlat)},
        attrs={"units": "mm"},
    )
    res = gauge_adjust(radar_geo, geo, method="ked", radius=800.0)
    np.testing.assert_allclose(
        res.RAIN_ADJUSTED.values, base.RAIN_ADJUSTED.values, rtol=1e-3, atol=1e-3
    )
    with pytest.raises(ValueError, match="geographic"):
        gauge_adjust(radar_geo, gauges, method="mfb", x="lon", y="lat")
    with pytest.raises(ValueError, match="position"):
        gauge_adjust(radar_geo, geo, method="rda", radius=800.0)
    # a sweep: gates with 2-D x and y
    az = np.deg2rad(np.arange(0, 360, 2.0))[:, None]
    rg = (np.arange(1, 40) * 500.0)[None, :]
    sx, sy = rg * np.sin(az), rg * np.cos(az)
    field = truth_field(np.arange(-20e3, 20e3, 1e3), np.arange(-20e3, 20e3, 1e3))
    sweep = xr.DataArray(
        np.exp(-np.hypot(sx, sy) / 1e4) * 3.0,
        dims=("azimuth", "range"),
        coords={"x": (("azimuth", "range"), sx), "y": (("azimuth", "range"), sy)},
        attrs={"units": "mm"},
    )
    assert field.size
    sg = xr.DataArray(
        np.array([2.0, 1.5, 1.0, 2.2, 1.1, 0.9, 1.7]),
        dims="s",
        coords={
            "x": ("s", [2e3, -5e3, 8e3, -9e3, 1e3, 11e3, -3e3]),
            "y": ("s", [3e3, 4e3, -6e3, -2e3, 9e3, 1e3, -12e3]),
        },
    )
    for method in ("mfb", "ked", "ok"):
        res = gauge_adjust(sweep, sg, method=method, min_pairs=5, radius=1500.0)
        assert res.RAIN_ADJUSTED.dims == ("azimuth", "range")
        assert np.isfinite(res.RAIN_ADJUSTED.values).all()


def test_radar_at_gauges_is_the_mean_of_the_pixels_within_the_radius():
    x = y = np.arange(8) * 1000.0
    field = np.arange(64, dtype=float).reshape(8, 8)
    radar = xr.DataArray(
        field, dims=("y", "x"), coords={"x": x, "y": y}, attrs={"units": "mm"}, name="R"
    )
    gauges = xr.DataArray(
        [0.0, 0.0, 0.0],
        dims="s",
        coords={"x": ("s", [3e3, 0.0, 30e3]), "y": ("s", [4e3, 0.0, 30e3])},
    )
    at = radar_at_gauges(radar, gauges)
    assert at.attrs["units"] == "mm"
    # 3 x 3 block centred on (3, 4) km: rows 3..5, columns 2..4
    assert at.values[0] == pytest.approx(field[3:6, 2:5].mean())
    # corner: only the pixels that exist
    assert at.values[1] == pytest.approx(field[0:2, 0:2].mean())
    assert np.isnan(at.values[2])
    one = radar_at_gauges(radar, gauges, radius=100.0)
    assert one.values[0] == field[4, 3]
    # a NaN pixel is left out of the mean
    holey = radar.copy()
    holey[4, 3] = np.nan
    assert radar_at_gauges(holey, gauges).values[0] == pytest.approx(
        np.nanmean(holey.values[3:6, 2:5])
    )
    with pytest.raises(ValueError, match="radius"):
        radar_at_gauges(radar, gauges, radius=-1.0)


# --------------------------------------------------------------------------
# variogram
# --------------------------------------------------------------------------


def brute_variogram(x, y, v, n_lags, max_lag):
    i, j = np.triu_indices(len(x), 1)
    h = np.hypot(x[i] - x[j], y[i] - y[j])
    w = max_lag / n_lags
    b = np.minimum((h // w).astype(int), n_lags - 1)
    ok = h < max_lag
    gam, dist, cnt = (
        np.full(n_lags, np.nan),
        np.full(n_lags, np.nan),
        np.zeros(n_lags, int),
    )
    for k in range(n_lags):
        sel = ok & (b == k)
        cnt[k] = sel.sum()
        if cnt[k]:
            gam[k] = 0.5 * np.mean((v[i[sel]] - v[j[sel]]) ** 2)
            dist[k] = h[sel].mean()
    return dist, gam, cnt


@pytest.mark.parametrize("engine", ENGINES)
def test_empirical_variogram_against_a_brute_force_sum(engine):
    rng = np.random.default_rng(8)
    x, y = rng.uniform(0, 5e4, 40), rng.uniform(0, 5e4, 40)
    v = rng.uniform(0, 8, 40)
    da = xr.DataArray(
        v, dims="s", coords={"x": ("s", x), "y": ("s", y)}, attrs={"units": "mm"}
    )
    res = gauge_variogram(da, n_lags=6, max_lag=3e4, engine=engine)
    dist, gam, cnt = brute_variogram(x, y, v, 6, 3e4)
    np.testing.assert_allclose(res.semivariance.values, gam, rtol=1e-10)
    np.testing.assert_allclose(res.distance.values, dist, rtol=1e-10)
    np.testing.assert_array_equal(res.pairs.values, cnt)
    assert res.attrs["model"] == "exponential"
    assert res.semivariance.attrs["units"] == "(mm)2"
    assert res.partial_sill.attrs["units"] == "(mm)2"
    assert res["range"].attrs["units"] == "m"


@pytest.mark.parametrize("model", ["linear", "spherical", "exponential", "gaussian"])
def test_variogram_fit_recovers_the_model_from_its_own_values(model):
    lag = np.linspace(2e3, 4e4, 14)
    count = np.full(14, 200)
    truth = {
        "linear": (0.3, 2e-4, 0.0),
        "spherical": (0.2, 1.5, 2e4),
        "exponential": (0.2, 1.5, 2e4),
        "gaussian": (0.1, 1.5, 2e4),
    }[model]
    gamma = gaugemod._semivariance(model, lag, *truth)
    nug, ps, rg = gaugemod._fit_variogram(model, lag, gamma, count)
    fitted = gaugemod._semivariance(model, lag, nug, ps, rg)
    np.testing.assert_allclose(fitted, gamma, rtol=2e-3)
    if model != "linear":
        assert rg == pytest.approx(truth[2], rel=0.05)


def test_variogram_fit_degenerate_inputs():
    lag = np.array([1e3, 2e3, 3e3])
    nug, ps, rg = gaugemod._fit_variogram(
        "exponential", lag, np.array([np.nan, np.nan, 1.0]), np.array([0, 0, 5])
    )
    assert ps > 0 and rg > 0
    nug, ps, rg = gaugemod._fit_variogram(
        "linear", lag, np.zeros(3), np.array([5, 5, 5])
    )
    assert rg == 0.0 and ps > 0


def test_external_drift_variogram_is_that_of_the_residuals():
    radar, gauges, _ = case(ng=30)
    at = radar_at_gauges(radar, gauges)
    linear = xr.DataArray(1.0 + 2.0 * at.values, dims="station", coords=gauges.coords)
    res = gauge_variogram(linear, radar=radar, n_lags=6)
    # a perfect regression leaves no semivariance
    assert float(np.nanmax(res.semivariance)) < 1e-18
    plain = gauge_variogram(linear, n_lags=6)
    assert float(np.nanmax(plain.semivariance)) > 1e-3


def test_variogram_with_times_and_a_fixed_variogram_dataset():
    radar, gauges, _ = case(n=24, ng=20)
    gs = xr.concat([gauges, gauges * 1.4], dim="time").assign_coords(time=[0, 1])
    rs = xr.concat([radar, radar * 1.4], dim="time").assign_coords(time=[0, 1])
    vario = gauge_variogram(gs, n_lags=7)
    assert vario.nugget.dims == ("time",)
    assert vario.semivariance.dims == ("time", "lag")
    assert vario.partial_sill.values[1] > vario.partial_sill.values[0]
    fitted = gauge_adjust(rs, gs, method="ok", radius=800.0, n_lags=7)
    fixed = gauge_adjust(rs, gs, method="ok", variogram=vario, radius=800.0)
    np.testing.assert_allclose(
        fixed.RAIN_ADJUSTED.values, fitted.RAIN_ADJUSTED.values, rtol=1e-3, atol=1e-3
    )
    # a missing value gives a different gauge set for that time
    gm = gs.copy()
    gm[0, 0] = np.nan
    v2 = gauge_variogram(gm, n_lags=7)
    assert v2.pairs.values[0].sum() < v2.pairs.values[1].sum()


def test_variogram_errors():
    rng = np.random.default_rng(0)
    da = xr.DataArray(
        rng.uniform(0, 1, 10),
        dims="s",
        coords={
            "x": ("s", rng.uniform(0, 1e4, 10)),
            "y": ("s", rng.uniform(0, 1e4, 10)),
        },
    )
    with pytest.raises(ValueError, match="model"):
        gauge_variogram(da, model="cubic")
    with pytest.raises(ValueError, match="n_lags"):
        gauge_variogram(da, n_lags=2)
    with pytest.raises(ValueError, match="max_lag"):
        gauge_variogram(da, max_lag=-1.0)
    with pytest.raises(ValueError, match="engine"):
        gauge_variogram(da, engine="gpu")
    few = da.copy()
    few[2:] = np.nan
    with pytest.raises(ValueError, match="at least 3"):
        gauge_variogram(few)
    with pytest.raises(TypeError):
        gauge_variogram(da.values)
    with pytest.raises(KeyError):
        gauge_variogram(da.drop_vars(["x", "y"]))
    with pytest.raises(ValueError, match="both x and y"):
        gauge_variogram(da, x="x")
    with pytest.raises(KeyError, match="no coordinate"):
        gauge_variogram(da, x="nope", y="y")


# --------------------------------------------------------------------------
# errors and the accessor
# --------------------------------------------------------------------------


def test_adjust_errors():
    radar, gauges, _ = case(n=16, ng=10)
    with pytest.raises(ValueError, match="method"):
        gauge_adjust(radar, gauges, method="nearest")
    with pytest.raises(ValueError, match="model"):
        gauge_adjust(radar, gauges, model="cubic")
    with pytest.raises(ValueError, match="engine"):
        gauge_adjust(radar, gauges, engine="gpu")
    with pytest.raises(ValueError, match="min_pairs"):
        gauge_adjust(radar, gauges, min_pairs=1)
    with pytest.raises(ValueError, match="n_neighbours"):
        gauge_adjust(radar, gauges, n_neighbours=2)
    with pytest.raises(TypeError, match="radar"):
        gauge_adjust(radar.values, gauges)
    with pytest.raises(TypeError, match="gauges"):
        gauge_adjust(radar, gauges.values)
    with pytest.raises(ValueError, match="no finite"):
        gauge_adjust(radar * np.nan, gauges)
    with pytest.raises(ValueError, match="fewer than"):
        gauge_adjust(radar, gauges.isel(station=slice(0, 3)), min_pairs=5)
    with pytest.raises(ValueError, match="dimension"):
        gauge_adjust(radar.expand_dims(time=2), gauges)
    with pytest.raises(ValueError, match="size"):
        gauge_adjust(radar.expand_dims(time=2), gauges.expand_dims(time=3))
    with pytest.raises(ValueError, match="differs"):
        gauge_adjust(
            radar.expand_dims(time=[0, 1]),
            gauges.expand_dims(time=[0, 2]),
            method="mfb",
        )
    with pytest.raises(KeyError):
        gauge_adjust(radar, gauges.drop_vars(["x", "y"]))
    with pytest.raises(KeyError):
        gauge_adjust(radar.drop_vars(["x", "y"]), gauges)
    with pytest.raises(ValueError, match="1-D"):
        gauge_adjust(
            radar,
            xr.DataArray(
                np.ones((2, 5)),
                dims=("a", "b"),
                coords={
                    "x": (("a", "b"), np.ones((2, 5))),
                    "y": (("a", "b"), np.ones((2, 5))),
                },
            ),
        )
    with pytest.raises(ValueError, match="k must"):
        gauge_adjust(radar, gauges, method="brandes", k=-1.0, radius=800.0)


def test_fixed_variogram_errors():
    radar, gauges, _ = case(n=16, ng=10)
    good = {"model": "exponential", "nugget": 0.1, "partial_sill": 1.0, "range": 1e4}
    gauge_adjust(radar, gauges, method="ok", variogram=good, radius=800.0)
    linear = {"model": "linear", "nugget": 0.1, "slope": 1e-4}
    res = gauge_adjust(radar, gauges, method="ok", variogram=linear, radius=800.0)
    assert res.attrs["variogram_model"] == "linear"
    with pytest.raises(KeyError, match="nugget"):
        gauge_adjust(radar, gauges, method="ok", variogram={"model": "exponential"})
    with pytest.raises(ValueError, match="one of"):
        gauge_adjust(radar, gauges, method="ok", variogram={**good, "model": "cubic"})
    with pytest.raises(ValueError, match="negative"):
        gauge_adjust(radar, gauges, method="ok", variogram={**good, "nugget": -1.0})
    with pytest.raises(ValueError, match="range"):
        gauge_adjust(radar, gauges, method="ok", variogram={**good, "range": 0.0})
    with pytest.raises(TypeError, match="variogram"):
        gauge_adjust(radar, gauges, method="ok", variogram=3.0)
    with pytest.raises(KeyError, match="partial_sill"):
        gauge_adjust(radar, gauges, method="ok", variogram=xr.Dataset({"nugget": 0.0}))


def test_kriging_kernel_argument_errors():
    if not gaugemod.HAS_COMPILED_KERNEL:
        pytest.skip("compiled kernel not built")
    k = gaugemod._gauge
    x = np.array([0.0, 1.0, 2.0, 3.0])
    v = np.arange(4.0).reshape(1, 4)
    none = np.empty((0, 0))
    par = np.array([[0.1, 1.0, 5.0]])
    grp = np.zeros(1, dtype=np.int32)
    with pytest.raises(ValueError, match="model"):
        k.krige(x, x, v, none, x, x, none, 7, par, grp, 0)
    with pytest.raises(ValueError, match="params"):
        k.krige(x, x, v, none, x, x, none, 1, par[:, :2], grp, 0)
    with pytest.raises(ValueError, match="group"):
        k.krige(x, x, v, none, x, x, none, 1, par, np.zeros(2, dtype=np.int32), 0)
    with pytest.raises(ValueError, match="xg"):
        k.krige(x[:3], x, v, none, x, x, none, 1, par, grp, 0)
    with pytest.raises(ValueError, match="same size"):
        k.krige(x, x, v, none, x, x[:2], none, 1, par, grp, 0)
    with pytest.raises(ValueError, match="dimensions"):
        k.krige(x, x, v[0], none, x, x, none, 1, par, grp, 0)
    with pytest.raises(ValueError, match="drift_g"):
        k.krige(x, x, v, np.ones((1, 3)), x, x, np.ones((1, 4)), 1, par, grp, 0)
    with pytest.raises(ValueError, match="own group"):
        k.krige(
            x,
            x,
            np.vstack([v, v]),
            np.ones((2, 4)),
            x,
            x,
            np.ones((2, 4)),
            1,
            np.vstack([par, par]),
            np.zeros(2, dtype=np.int32),
            0,
        )
    # all gauges at one point: the system is singular
    same = np.zeros(4)
    with pytest.raises(RuntimeError, match="singular"):
        k.krige(same, same, v, none, x, x, none, 1, par, grp, 0)
    with pytest.raises(RuntimeError, match="singular"):
        k.krige(same, same, v, none, x, x, none, 1, par, grp, 3)
    with pytest.raises(ValueError, match="k must"):
        k.brandes(x, x, v, x, x, 0.0)
    with pytest.raises(ValueError, match="same size"):
        k.brandes(x, x, v, x, x[:2], 1.0)
    with pytest.raises(ValueError, match="2 dimensions"):
        k.brandes(x, x, v[0], x, x, 1.0)
    with pytest.raises(ValueError, match="2-D"):
        k.mean_field_bias(v[0], v[0], 0.0)
    with pytest.raises(ValueError, match="2 dimensions"):
        k.variogram(x, x, v[0], 3, 1.0)
    with pytest.raises(ValueError, match="x and y"):
        k.variogram(x[:3], x, v, 3, 1.0)
    with pytest.raises(ValueError, match="n_lags"):
        k.variogram(x, x, v, 0, 1.0)
    with pytest.raises(ValueError, match="at least one"):
        k.krige(
            np.empty(0), np.empty(0), np.empty((1, 0)), none, x, x, none, 1, par, grp, 0
        )


def test_the_numpy_kriging_rejects_a_shared_drift_group():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    v = np.vstack([np.arange(4.0)] * 2)
    par = np.tile([0.1, 1.0, 5.0], (2, 1))
    with pytest.raises(ValueError, match="own group"):
        gaugemod._krige_numpy(
            x,
            x,
            v,
            np.ones((2, 4)),
            x,
            x,
            np.ones((2, 4)),
            1,
            par,
            np.zeros(2, dtype=np.int32),
            0,
        )


def test_accessor_and_provenance():
    radar, gauges, _ = case(n=16, ng=10)
    direct = gauge_adjust(radar, gauges, method="mfb", radius=800.0)
    via = radar.radarx.gauge_adjust(gauges, method="mfb", radius=800.0)
    xr.testing.assert_identical(without_provenance(direct), without_provenance(via))
    assert direct.RAIN_ADJUSTED.attrs["radarx_references"]
    assert "10.5194/hess-13-195-2009" in direct.RAIN_ADJUSTED.attrs["radarx_references"]


# --------------------------------------------------------------------------
# independent implementations
# --------------------------------------------------------------------------


def test_against_wradlib_kriging():
    wradlib = pytest.importorskip("wradlib")
    rng = np.random.default_rng(11)
    gx, gy = rng.uniform(0, 3e4, 20), rng.uniform(0, 3e4, 20)
    values = rng.uniform(0, 6, 20)
    tx, ty = rng.uniform(0, 3e4, 30), rng.uniform(0, 3e4, 30)
    target = xr.DataArray(
        np.ones((1, 30)),
        dims=("a", "b"),
        coords={"x": (("a", "b"), tx[None]), "y": (("a", "b"), ty[None])},
    )
    gauges = xr.DataArray(values, dims="s", coords={"x": ("s", gx), "y": ("s", gy)})
    # wradlib's "Exp(a)" has the covariance exp(-h/a): the practical range is 3a
    ok = wradlib.ipol.OrdinaryKriging(
        np.c_[gx, gy], np.c_[tx, ty], cov="0.5 Nug(0) + 2.0 Exp(7000.)", nnearest=20
    )
    theirs = ok(values)
    mine = gauge_adjust(
        target,
        gauges,
        method="ok",
        n_neighbours=None,
        clip=False,
        variogram={
            "model": "exponential",
            "nugget": 0.5,
            "partial_sill": 2.0,
            "range": 21e3,
        },
    )
    np.testing.assert_allclose(
        mine.RAIN_ADJUSTED.values.ravel(),
        np.asarray(theirs).ravel(),
        rtol=1e-4,
        atol=1e-4,
    )


# --------------------------------------------------------------------------
# edge cases
# --------------------------------------------------------------------------


def test_variogram_of_a_series_with_different_gauge_sets_and_empty_times():
    rng = np.random.default_rng(2)
    gx, gy = rng.uniform(0, 5e4, 20), rng.uniform(0, 5e4, 20)
    v = 3.0 + np.sin(gx / 1e4) + rng.normal(0, 0.1, (3, 20))
    da = xr.DataArray(v, dims=("time", "s"), coords={"x": ("s", gx), "y": ("s", gy)})
    da[1, 5:] = np.nan  # five gauges left
    da[2, :] = np.nan  # no gauge left
    res = gauge_variogram(da)
    assert res.semivariance.shape[0] == 3
    assert np.isfinite(res.nugget.values[:2]).all()
    assert np.isnan(res.nugget.values[2])
    assert res.pairs.values[0].sum() > res.pairs.values[1].sum()
    assert res.pairs.values[2].sum() == 0


def test_variogram_of_geographic_gauges_and_bad_coordinates():
    rng = np.random.default_rng(3)
    lon, lat = rng.uniform(77.0, 77.5, 15), rng.uniform(12.0, 12.5, 15)
    da = xr.DataArray(
        rng.uniform(0, 4, 15), dims="s", coords={"lon": ("s", lon), "lat": ("s", lat)}
    )
    res = gauge_variogram(da, n_lags=5)
    assert res.pairs.values.sum() > 0
    assert res.semivariance.dims == ("lag",)
    two_d = xr.DataArray(
        np.ones((2, 4)),
        dims=("a", "b"),
        coords={"x": (("a", "b"), np.ones((2, 4))), "y": (("a", "b"), np.ones((2, 4)))},
    )
    with pytest.raises(ValueError, match="1-D"):
        gauge_variogram(two_d)


def test_conditional_merging_of_constant_fields():
    x = y = np.arange(10) * 1000.0
    radar = xr.DataArray(
        np.full((10, 10), 1.0),
        dims=("y", "x"),
        coords={"x": x, "y": y},
        attrs={"units": "mm"},
    )
    rng = np.random.default_rng(0)
    gauges = xr.DataArray(
        np.full(8, 2.0),
        dims="s",
        coords={
            "x": ("s", rng.uniform(1e3, 8e3, 8)),
            "y": ("s", rng.uniform(1e3, 8e3, 8)),
        },
    )
    res = gauge_adjust(radar, gauges, method="conditional", radius=2000.0)
    np.testing.assert_allclose(res.RAIN_ADJUSTED.values, 2.0)
    assert int(res.GAUGES_USED) == 8


def test_explicit_coordinate_names_and_a_single_pixel_radar():
    radar, gauges, _ = case(n=16, ng=10)
    res = gauge_adjust(radar, gauges, method="mfb", x="x", y="y", radius=800.0)
    assert np.isfinite(res.RAIN_ADJUSTED.values).all()
    one = xr.DataArray(
        [[2.0]], dims=("y", "x"), coords={"x": [5e3], "y": [5e3]}, attrs={"units": "mm"}
    )
    g = xr.DataArray([1.0], dims="s", coords={"x": ("s", [5e3]), "y": ("s", [5e3])})
    assert radar_at_gauges(one, g).values[0] == 2.0


def test_duplicate_gauges_are_refused_for_kriging():
    radar, gauges, _ = case(n=16, ng=10)
    dup = gauges.copy()
    dup.coords["x"][1] = dup.coords["x"][0]
    dup.coords["y"][1] = dup.coords["y"][0]
    for method in ("ok", "ked", "conditional"):
        with pytest.raises(ValueError, match="same position"):
            gauge_adjust(radar, dup, method=method, radius=800.0)
    gauge_adjust(radar, dup, method="mfb", radius=800.0)  # fine for the factors


@pytest.mark.parametrize("compiled", [False, True])
def test_a_singular_system_is_an_error(compiled):
    if compiled and not gaugemod.HAS_COMPILED_KERNEL:
        pytest.skip("compiled kernel not built")
    x = np.array([0.0, 0.0, 1.0, 2.0])  # two gauges at the same point
    y = np.zeros(4)
    v = np.array([[1.0, 2.0, 3.0, 4.0]])
    none = np.empty((0, 0))
    par = np.array([[0.1, 1.0, 5.0]])
    with pytest.raises(RuntimeError, match="singular"):
        gaugemod._krige(
            x,
            y,
            v,
            none,
            np.array([0.5]),
            np.array([0.0]),
            none,
            1,
            par,
            np.zeros(1, dtype=np.int32),
            0,
            compiled,
            1,
        )
    zero = np.array([[0.0, 0.0, 5.0]])
    with pytest.raises(RuntimeError, match="singular"):
        gaugemod._krige(
            x,
            y,
            v,
            none,
            np.array([0.5]),
            np.array([0.0]),
            none,
            1,
            zero,
            np.zeros(1, dtype=np.int32),
            0,
            compiled,
            1,
        )


def test_brandes_with_a_time_without_valid_gauges():
    x = np.array([0.0, 1.0, 2.0])
    ratio = np.array([[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan]])
    out = gaugemod._brandes_numpy(x, x, ratio, np.array([0.5]), np.array([0.5]), 1.0)
    assert np.isfinite(out[0]).all() and np.isnan(out[1]).all()
    if gaugemod.HAS_COMPILED_KERNEL:
        out = gaugemod._gauge.brandes(
            x, x, ratio, np.array([0.5]), np.array([0.5]), 1.0
        )
        assert np.isfinite(out[0]).all() and np.isnan(out[1]).all()


def test_variogram_fit_helpers(monkeypatch):
    lag = np.linspace(1e3, 2e4, 10)
    count = np.full(10, 50)
    gamma = gaugemod._semivariance("exponential", lag, 0.1, 1.0, 1e4)
    assert (
        gaugemod._cressie_cost("exponential", lag, gamma, count, 0.0, 0.0, 1e4)
        == np.inf
    )
    import scipy.optimize

    def fail(*args, **kwargs):
        raise ValueError("no convergence")

    monkeypatch.setattr(scipy.optimize, "least_squares", fail)
    nug, ps, rg = gaugemod._fit_variogram("exponential", lag, gamma, count)
    assert ps > 0 and rg > 0  # the grid start is returned
