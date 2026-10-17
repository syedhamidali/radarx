#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the KDP columns and their metrics
===========================================

Columns with a known KDP profile have a slab integral, a thickness and a top that follow
from the definitions of van Lier-Walqui et al. (2016): the integral of KDP over the slab
above the freezing level, the thickness of the levels with KDP above 0.75 degrees km-1
and the height of the highest such level. The compiled kernels and the NumPy
implementations must give the same values.
"""

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.retrieve import kdp_column, kdp_column_metrics

columns = __import__("radarx.retrieve.columns", fromlist=["x"])

ENGINES = ["numpy"] + (["compiled"] if columns.HAS_COMPILED_KERNEL else [])
compiled_only = pytest.mark.skipif(
    not columns.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)

Z = np.arange(0.0, 12001.0, 250.0)
Y = X = np.arange(41) * 1000.0


def grid(top, base=4000.0, value=1.5, background=0.0, name="KDP"):
    """KDP of ``value`` from ``base`` to the map ``top`` (m) above each point of the grid."""
    z = Z[:, None, None]
    values = np.where((z >= base) & (z < top[None]), value, background)
    return xr.DataArray(
        values,
        dims=("z", "y", "x"),
        coords={"z": Z, "y": Y, "x": X},
        name=name,
    )


def two_columns():
    r1 = np.hypot(Y[:, None] - 12e3, X[None, :] - 12e3)
    r2 = np.hypot(Y[:, None] - 30e3, X[None, :] - 28e3)
    top = np.maximum(
        4000.0 + 3000.0 * np.clip(1 - r1 / 6e3, 0, None),
        4000.0 + 2000.0 * np.clip(1 - r2 / 5e3, 0, None),
    )
    return grid(top)


@pytest.mark.parametrize("engine", ENGINES)
def test_slab_integral_thickness_and_top(engine):
    da = grid(np.full((41, 41), 7000.0))
    out = kdp_column(da, freezing_level=4000.0, engine=engine)
    assert set(out.data_vars) == {
        "KDP_SLAB_INTEGRAL",
        "KDP_SLAB_DEPTH",
        "KDP_COLUMN_TOP",
        "KDP_COLUMN_OBJECT",
    }
    assert (
        out.KDP_SLAB_INTEGRAL.dims == ("y", "x")
        and out.KDP_SLAB_INTEGRAL.dtype == np.float32
    )
    # the slab holds eight levels of 250 m: 1.5 degrees km-1 over 2 km is 3 degrees
    np.testing.assert_allclose(out.KDP_SLAB_INTEGRAL.values, 3.0)
    np.testing.assert_allclose(out.KDP_SLAB_DEPTH.values, 2000.0)
    np.testing.assert_allclose(out.KDP_COLUMN_TOP.values, 6750.0)
    assert out.KDP_SLAB_INTEGRAL.attrs["units"] == "degree"
    assert (
        out.KDP_SLAB_DEPTH.attrs["units"] == "m"
        and out.KDP_COLUMN_TOP.attrs["units"] == "m"
    )
    # a deeper slab, a higher threshold and a different freezing level
    deep = kdp_column(da, freezing_level=4000.0, slab_depth=4000.0, engine=engine)
    np.testing.assert_allclose(deep.KDP_SLAB_INTEGRAL.values, 4.5)
    none = kdp_column(da, freezing_level=4000.0, threshold=2.0, engine=engine)
    assert np.isnan(none.KDP_COLUMN_TOP.values).all()
    np.testing.assert_allclose(none.KDP_SLAB_DEPTH.values, 0.0)
    np.testing.assert_allclose(none.KDP_SLAB_INTEGRAL.values, 3.0)
    higher = kdp_column(da, freezing_level=6000.0, engine=engine)
    np.testing.assert_allclose(higher.KDP_SLAB_DEPTH.values, 1000.0)
    # the top is the highest level above the freezing level, also above the slab
    gap = grid(np.full((41, 41), 7000.0), base=4000.0)
    gap.values[(Z >= 4500) & (Z < 6000)] = 0.0
    out = kdp_column(gap, freezing_level=4000.0, engine=engine)
    np.testing.assert_allclose(out.KDP_COLUMN_TOP.values, 6750.0)
    np.testing.assert_allclose(
        out.KDP_SLAB_DEPTH.values, 500.0 + 0.0 + 0.0
    )  # 4000-4500 only


@pytest.mark.parametrize("engine", ENGINES)
def test_negative_kdp_is_not_integrated_and_missing_values_are_skipped(engine):
    da = grid(np.full((41, 41), 7000.0), value=-1.0)
    out = kdp_column(da, freezing_level=4000.0, engine=engine)
    np.testing.assert_allclose(out.KDP_SLAB_INTEGRAL.values, 0.0)
    assert np.isnan(out.KDP_COLUMN_TOP.values).all()
    missing = da.where(da.z < 4000.0)  # NaN from the freezing level up
    out = kdp_column(missing, freezing_level=4000.0, engine=engine)
    assert (
        np.isnan(out.KDP_SLAB_INTEGRAL.values).all()
        and np.isnan(out.KDP_SLAB_DEPTH.values).all()
    )
    # no freezing level, no column
    out = kdp_column(da, freezing_level=np.nan, engine=engine)
    assert (
        np.isnan(out.KDP_COLUMN_TOP.values).all()
        and int(out.KDP_COLUMN_OBJECT.max()) == 0
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_columns_are_objects_with_their_own_top(engine):
    da = two_columns()
    out = kdp_column(da, freezing_level=4000.0, engine=engine)
    labels = out.KDP_COLUMN_OBJECT.values
    assert set(np.unique(labels)) == {0, 1, 2}
    assert (
        labels[12, 12] != labels[28, 30] and labels[12, 12] > 0 and labels[28, 30] > 0
    )
    metrics = kdp_column_metrics(out)
    assert int(metrics.KDP_COLUMN_COUNT) == 2
    assert float(metrics.KDP_COLUMN_MAX_TOP) == pytest.approx(6750.0, abs=250.0)
    inside = labels > 0
    assert float(metrics.KDP_COLUMN_AREA) == pytest.approx(
        float(inside.sum()), rel=1e-6
    )  # 1 km2 cells
    depth = out.KDP_SLAB_DEPTH.values
    assert float(metrics.KDP_VOLUME) == pytest.approx(
        float(np.nansum(depth)) * 1e6 / 1e9, rel=1e-5
    )
    assert (
        metrics.KDP_COLUMN_AREA.attrs["units"] == "km2"
        and metrics.KDP_VOLUME.attrs["units"] == "km3"
    )
    # a cell area replaces the spacing of the grid
    twice = kdp_column_metrics(out, cell_area=2e6)
    assert float(twice.KDP_COLUMN_AREA) == pytest.approx(
        2 * float(metrics.KDP_COLUMN_AREA)
    )
    # without smoothing, and with a deeper minimum, the same two columns
    raw = kdp_column(
        da, freezing_level=4000.0, smooth=None, min_depth=1000.0, engine=engine
    )
    assert int(raw.KDP_COLUMN_OBJECT.max()) == 2
    # a higher floor leaves out the weaker column (3 degrees in the slab against 4.5)
    weak = kdp_column(
        da,
        freezing_level=4000.0,
        slab_depth=4000.0,
        smooth=None,
        min_depth=5000.0,
        engine=engine,
    )
    assert int(weak.KDP_COLUMN_OBJECT.max()) == 1


@pytest.mark.parametrize("engine", ENGINES)
def test_time_dimension_fields_and_options(engine):
    da = two_columns()
    stack = xr.concat(
        [da, da * 0.5], dim=xr.DataArray([0, 1], dims="time", name="time")
    )
    ds = stack.to_dataset(name="KDP")
    ds["RHOHV"] = xr.full_like(ds.KDP, 0.97)
    ds["RHOHV"][
        :, :, 10:15, 10:15
    ] = 0.5  # removes the first column in the first volume
    ds["DBZH"] = xr.full_like(ds.KDP, 40.0)
    ds["DBZH"][:, :, 25:35, 25:35] = 10.0
    out = kdp_column(ds, rhohv="RHOHV", freezing_level=4000.0, engine=engine)
    assert out.KDP_COLUMN_TOP.dims[0] == "time"
    # the removed grid points do not count
    assert np.isnan(float(out.KDP_SLAB_DEPTH.isel(time=0, y=12, x=12)))
    assert float(out.KDP_SLAB_DEPTH.isel(time=0, y=16, x=12)) > 0.0
    both = kdp_column(
        ds,
        "KDP",
        dbz=None,
        rhohv="RHOHV",
        min_dbz=20.0,
        freezing_level=4000.0,
        engine=engine,
    )
    assert np.isnan(float(both.KDP_SLAB_DEPTH.isel(time=0, y=28, x=30)))
    # half the KDP in the second volume is exactly the threshold, which still counts
    # (the first column is removed by the correlation coefficient in both volumes)
    assert float(out.KDP_COLUMN_TOP.isel(time=1).max()) == 5750.0
    # a freezing level in km as a DataArray, as a field of the dataset, and a unit-less number
    level = xr.DataArray(np.full((41, 41), 4.0), dims=("y", "x"), attrs={"units": "km"})
    a = kdp_column(ds, freezing_level=level, engine=engine)
    b = kdp_column(ds, freezing_level=4000.0, engine=engine)
    xr.testing.assert_equal(a, b)
    ds["T0"] = xr.DataArray(
        np.full((41, 41), 4000.0), dims=("y", "x"), attrs={"units": "m"}
    )
    c = kdp_column(ds, freezing_level="T0", engine=engine)
    xr.testing.assert_equal(c, b)
    # unsorted levels are sorted, the input is not changed
    flipped = ds.isel(z=slice(None, None, -1))
    before = flipped.copy(deep=True)
    d = kdp_column(flipped, freezing_level=4000.0, engine=engine)
    xr.testing.assert_equal(d, b)
    xr.testing.assert_identical(flipped, before)


@compiled_only
def test_engines_agree_on_random_fields():
    rnd = np.random.default_rng(3)
    shape = (2, 12, 20, 24)
    kdp = np.where(rnd.random(shape) < 0.1, np.nan, rnd.normal(1.0, 1.0, shape))
    da = xr.DataArray(
        kdp,
        dims=("time", "z", "y", "x"),
        coords={
            "z": 500.0 + 1000.0 * np.arange(12),
            "y": np.arange(20) * 1e3,
            "x": np.arange(24) * 1e3,
        },
        name="KDP",
    )
    a = kdp_column(da, freezing_level=4300.0, engine="numpy")
    b = kdp_column(da, freezing_level=4300.0, engine="compiled", n_threads=2)
    for name in a:
        np.testing.assert_allclose(
            a[name].values, b[name].values, rtol=1e-6, equal_nan=True
        )


def test_errors():
    da = two_columns()
    with pytest.raises(ValueError, match="engine"):
        kdp_column(da, freezing_level=4000.0, engine="gpu")
    with pytest.raises(ValueError, match="vertical"):
        kdp_column(
            da.isel(z=0, drop=True).rename(y="a").expand_dims(q=2),
            freezing_level=4000.0,
        )
    with pytest.raises(ValueError, match="at least two levels"):
        kdp_column(da.isel(z=[3]), freezing_level=4000.0)
    with pytest.raises(ValueError, match="horizontal"):
        kdp_column(da.rename(x="lon"), freezing_level=4000.0)
    with pytest.raises(KeyError):
        kdp_column(da.to_dataset(name="other"), freezing_level=4000.0)
    with pytest.raises(KeyError, match="NOPE"):
        kdp_column(da.to_dataset(name="KDP"), freezing_level="NOPE")
    with pytest.raises(TypeError, match="freezing_level"):
        kdp_column(da, freezing_level=object())
    with pytest.raises(KeyError, match="warm"):
        kdp_column(da, freezing_level="warm")
    ds = da.to_dataset(name="KDP")
    with pytest.raises(ValueError, match="DataArray"):
        kdp_column(da, rhohv="RHOHV", freezing_level=4000.0)
    ds["DBZH"] = xr.full_like(da.isel(z=0), 30.0)
    with pytest.raises(ValueError, match="vertical"):
        kdp_column(ds, min_dbz=20.0, freezing_level=4000.0)
    out = kdp_column(da, freezing_level=4000.0)
    with pytest.raises(ValueError, match="spacing"):
        kdp_column_metrics(out.drop_vars(["x", "y"]))


def test_accessors_and_dask():
    da = two_columns()
    out = kdp_column(da, freezing_level=4000.0)
    same = da.radarx.kdp_column(freezing_level=4000.0)
    xr.testing.assert_equal(same, out)
    xr.testing.assert_equal(out.radarx.kdp_column_metrics(), kdp_column_metrics(out))
    pytest.importorskip("dask")
    lazy = kdp_column(da.chunk({"y": 10}), freezing_level=4000.0)
    assert all(v.chunks is not None for v in lazy.data_vars.values())
    xr.testing.assert_allclose(lazy.compute(), out)
    metrics = kdp_column_metrics(lazy)
    xr.testing.assert_allclose(metrics.compute(), kdp_column_metrics(out))
