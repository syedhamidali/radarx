#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the ZDR column depth and its metrics
==============================================

Columns with a known ZDR profile have a depth that follows from the definition of
Snyder et al. (2015): the vertically continuous levels with ZDR >= 1 dB from the
first level at or above the freezing level, times the layer thickness. The compiled
kernel and the NumPy implementation must give the same values.
"""

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.retrieve import zdr_column, zdr_column_metrics

columns = __import__("radarx.retrieve.columns", fromlist=["x"])

ENGINES = ["numpy"] + (["compiled"] if columns.HAS_COMPILED_KERNEL else [])
compiled_only = pytest.mark.skipif(
    not columns.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)

Z = np.arange(0.0, 12001.0, 250.0)


def grid(profile, ny=2, nx=3, **coords):
    """A grid whose every column has the ZDR profile ``profile(z)``."""
    values = np.broadcast_to(profile(Z)[:, None, None], (Z.size, ny, nx)).copy()
    return xr.DataArray(
        values,
        dims=("z", "y", "x"),
        coords={
            "z": Z,
            "y": np.arange(ny) * 1000.0,
            "x": np.arange(nx) * 1000.0,
            **coords,
        },
        name="ZDR",
    )


def column(depth_start=4000.0, depth_end=6000.0, base=0.2, inside=2.0):
    def profile(z):
        return np.where((z >= depth_start) & (z < depth_end), inside, base)

    return profile


@pytest.mark.parametrize("engine", ENGINES)
def test_depth_is_the_continuous_run_from_the_freezing_level(engine):
    da = grid(column(4000.0, 6000.0))
    out = zdr_column(da, freezing_level=4000.0, engine=engine)
    assert out.name == "ZDR_COLUMN_DEPTH" and out.attrs["units"] == "m"
    assert out.dims == ("y", "x") and out.dtype == np.float32
    np.testing.assert_allclose(out.values, 2000.0)  # 8 levels of 250 m
    # a freezing level in the middle of the column: only the part above counts
    out = zdr_column(da, freezing_level=5000.0, engine=engine)
    np.testing.assert_allclose(out.values, 1000.0)
    # a freezing level between two levels starts at the next level up
    out = zdr_column(da, freezing_level=4100.0, engine=engine)
    np.testing.assert_allclose(out.values, 1750.0)


@pytest.mark.parametrize("engine", ENGINES)
def test_the_run_must_start_at_the_freezing_level(engine):
    # ZDR is 2 dB from 5 to 6 km but not at the first level above a 4 km freezing level
    da = grid(column(5000.0, 6000.0))
    out = zdr_column(da, freezing_level=4000.0, engine=engine)
    np.testing.assert_allclose(out.values, 0.0)

    # a gap ends the run: 4.0 to 4.5 km, a gap of two levels, then more
    def two_parts(z):
        return np.where(
            ((z >= 4000) & (z < 4500)) | ((z >= 5000) & (z < 6000)), 2.0, 0.0
        )

    out = zdr_column(grid(two_parts), freezing_level=4000.0, engine=engine)
    np.testing.assert_allclose(out.values, 500.0)


@pytest.mark.parametrize("engine", ENGINES)
def test_threshold_is_inclusive_and_can_be_changed(engine):
    da = grid(column(4000.0, 5000.0, inside=1.0))
    np.testing.assert_allclose(
        zdr_column(da, freezing_level=4000.0, engine=engine), 1000.0
    )
    out = zdr_column(da, freezing_level=4000.0, threshold=1.5, engine=engine)
    np.testing.assert_allclose(out.values, 0.0)


@pytest.mark.parametrize("engine", ENGINES)
def test_missing_data(engine):
    da = grid(column(4000.0, 6000.0))
    da[:, 0, 0] = np.nan  # no data in the column: NaN, not 0
    da[(Z >= 4000) & (Z < 5000), 0, 1] = np.nan  # no data at the start: no column
    da[(Z >= 4500) & (Z < 4750), 0, 2] = np.nan  # a hole ends the column
    out = zdr_column(da, freezing_level=4000.0, engine=engine)
    assert np.isnan(out[0, 0]) and out[0, 1] == 0.0 and out[0, 2] == 500.0
    np.testing.assert_allclose(out[1].values, 2000.0)
    # a freezing level above the grid or NaN gives NaN
    assert np.isnan(zdr_column(da, freezing_level=20000.0, engine=engine)).all()
    assert np.isnan(zdr_column(da, freezing_level=np.nan, engine=engine)).all()


@pytest.mark.parametrize("engine", ENGINES)
def test_correlation_coefficient_removes_grid_points(engine):
    da = grid(column(4000.0, 6000.0))
    rho = xr.full_like(da, 0.98)
    rho[(Z >= 4500) & (Z < 4750), :, 0] = 0.7  # a hail spike level in one column
    rho[:, :, 1] = np.nan  # a missing correlation coefficient does not count
    out = zdr_column(da, rhohv=rho, freezing_level=4000.0, engine=engine)
    np.testing.assert_allclose(out[:, 0].values, 500.0)
    np.testing.assert_allclose(out[:, 1].values, 0.0)
    np.testing.assert_allclose(out[:, 2].values, 2000.0)
    # without the field there is no removal; min_rhohv moves the limit
    np.testing.assert_allclose(
        zdr_column(da, freezing_level=4000.0, engine=engine), 2000.0
    )
    out = zdr_column(
        da,
        rhohv=rho.where(rho != 0.7, 0.85),
        freezing_level=4000.0,
        min_rhohv=0.9,
        engine=engine,
    )
    np.testing.assert_allclose(out[:, 0].values, 500.0)


@pytest.mark.parametrize("engine", ENGINES)
def test_freezing_level_as_a_field_and_a_map(engine):
    da = grid(column(4000.0, 6000.0))
    level = xr.DataArray(
        [[4000.0, 5000.0, 6000.0], [4000.0, 4000.0, 4000.0]],
        dims=("y", "x"),
        attrs={"units": "m"},
    )
    out = zdr_column(da, freezing_level=level, engine=engine)
    np.testing.assert_allclose(
        out.values, [[2000.0, 1000.0, 0.0], [2000.0, 2000.0, 2000.0]]
    )
    ds = da.to_dataset().assign(T0=level / 1000.0)
    ds["T0"].attrs["units"] = "km"
    out = ds.radarx.zdr_column(freezing_level="T0", engine=engine)
    np.testing.assert_allclose(
        out.values, [[2000.0, 1000.0, 0.0], [2000.0, 2000.0, 2000.0]]
    )
    with pytest.raises(KeyError, match="NOPE"):
        zdr_column(ds, freezing_level="NOPE")


@pytest.mark.parametrize("engine", ENGINES)
def test_descending_levels_and_extra_dimensions(engine):
    da = grid(column(4000.0, 6000.0))
    flipped = da.isel(z=slice(None, None, -1))
    np.testing.assert_allclose(
        zdr_column(flipped, freezing_level=4000.0, engine=engine), 2000.0
    )
    stack = xr.concat([da, da.where(Z[:, None, None] < 5000.0)], dim="time")
    out = zdr_column(stack, freezing_level=4000.0, engine=engine)
    assert out.dims == ("time", "y", "x")
    np.testing.assert_allclose(out.isel(time=0).values, 2000.0)
    np.testing.assert_allclose(out.isel(time=1).values, 1000.0)


@pytest.mark.parametrize("engine", ENGINES)
def test_input_is_not_changed_and_names_are_found(engine):
    da = grid(column())
    before = da.copy(deep=True)
    zdr_column(da, freezing_level=4000.0, engine=engine)
    xr.testing.assert_identical(da, before)
    ds = da.rename("differential_reflectivity").to_dataset()
    np.testing.assert_allclose(
        zdr_column(ds, freezing_level=4000.0, engine=engine), 2000.0
    )
    ds = da.to_dataset(name="MYZDR")
    np.testing.assert_allclose(
        zdr_column(ds, zdr="MYZDR", freezing_level=4000.0, engine=engine), 2000.0
    )
    with pytest.raises(KeyError):
        zdr_column(ds, freezing_level=4000.0)


@compiled_only
def test_engines_agree_on_random_columns():
    rng = np.random.default_rng(5)
    values = rng.normal(1.0, 1.0, (Z.size, 40, 50))
    values[rng.random(values.shape) < 0.05] = np.nan
    da = xr.DataArray(values, dims=("z", "y", "x"), coords={"z": Z})
    rho = xr.DataArray(
        rng.uniform(0.6, 1.0, values.shape), dims=da.dims, coords=da.coords
    )
    level = xr.DataArray(rng.uniform(2000.0, 8000.0, (40, 50)), dims=("y", "x"))
    level[3, 4] = np.nan
    a = zdr_column(da, rhohv=rho, freezing_level=level, engine="numpy")
    b = zdr_column(da, rhohv=rho, freezing_level=level, engine="compiled", n_threads=3)
    np.testing.assert_array_equal(a.values, b.values)
    assert (
        np.isfinite(a.values).any() and (a.values > 0).any() and (a.values == 0).any()
    )


def test_dask_stays_lazy_and_matches():
    pytest.importorskip("dask")
    da = grid(column(4000.0, 6000.0), ny=6, nx=7)
    lazy = da.chunk({"y": 3, "x": 4, "z": 20})
    out = zdr_column(lazy, freezing_level=4000.0)
    assert hasattr(out.data, "dask")
    np.testing.assert_array_equal(
        out.compute().values, zdr_column(da, freezing_level=4000.0).values
    )


def test_errors():
    da = grid(column())
    with pytest.raises(ValueError, match="engine"):
        zdr_column(da, freezing_level=4000.0, engine="fortran")
    with pytest.raises(ValueError, match="vertical dimension"):
        zdr_column(da.isel(z=0, drop=True).rename(y="a"), freezing_level=4000.0)
    with pytest.raises(ValueError, match="at least two levels"):
        zdr_column(da.isel(z=slice(0, 1)), freezing_level=4000.0)
    with pytest.raises(TypeError, match="freezing_level"):
        zdr_column(da, freezing_level=object())
    with pytest.raises(ValueError, match="DataArray"):
        zdr_column(da, rhohv="RHOHV", freezing_level=4000.0)
    if not columns.HAS_COMPILED_KERNEL:
        with pytest.raises(ImportError):
            zdr_column(da, freezing_level=4000.0, engine="compiled")


def test_provenance_and_accessor():
    da = grid(column())
    out = da.radarx.zdr_column(freezing_level=4000.0)
    assert "10.1175/WAF-D-15-0068.1" in out.attrs["radarx_references"]
    assert out.attrs["radarx_method"].startswith("ZDR column depth")


def test_metrics_follow_the_definitions():
    depth = xr.DataArray(
        [[0.0, 500.0, 1000.0], [2000.0, np.nan, 3500.0]],
        dims=("y", "x"),
        coords={"y": [0.0, 2000.0], "x": [0.0, 2000.0, 4000.0]},
    )
    m = zdr_column_metrics(depth, min_depth=1000.0)
    assert float(m.ZDR_COLUMN_MAX_DEPTH) == 3500.0
    cell = 2000.0 * 2000.0
    assert float(m.ZDR_COLUMN_AREA) == pytest.approx(3 * cell / 1e6)
    assert float(m.ZDR_COLUMN_VOLUME) == pytest.approx(
        (1000 + 2000 + 3500) * cell / 1e9
    )
    assert m.ZDR_COLUMN_AREA.attrs["units"] == "km2"
    m = zdr_column_metrics(depth, min_depth=0.0, cell_area=1.0e6)
    assert float(m.ZDR_COLUMN_AREA) == pytest.approx(5.0)
    # a Dataset with the depth, and a time dimension
    stack = xr.concat([depth, depth * 0.0], dim="time")
    m = zdr_column_metrics(stack.to_dataset(name="ZDR_COLUMN_DEPTH"), min_depth=1000.0)
    assert m.ZDR_COLUMN_AREA.dims == ("time",)
    assert (
        float(m.ZDR_COLUMN_AREA[1]) == 0.0 and float(m.ZDR_COLUMN_MAX_DEPTH[1]) == 0.0
    )
    with pytest.raises(ValueError, match="cell_area"):
        zdr_column_metrics(xr.DataArray(np.ones((2, 2)), dims=("y", "x")))
    with pytest.raises(ValueError, match="two horizontal"):
        zdr_column_metrics(xr.DataArray(np.ones(3), dims=("x",)), cell_area=1.0)


def test_columns_of_a_whole_pipeline_from_a_gridded_profile():
    # a grid whose ZDR column grows with the updraft: the depth is the mean of both maps
    zz = Z[:, None, None]
    yy = np.arange(21)[None, :, None]
    xx = np.arange(21)[None, None, :]
    r = np.hypot(yy - 10, xx - 10)
    top = 4000.0 + 3000.0 * np.clip(1 - r / 6.0, 0, None)  # 7 km at the centre
    values = np.where(zz < top, 2.0, 0.0)
    da = xr.DataArray(
        values,
        dims=("z", "y", "x"),
        coords={"z": Z, "y": np.arange(21) * 1000.0, "x": np.arange(21) * 1000.0},
    )
    out = zdr_column(da, freezing_level=4000.0)
    assert float(out[10, 10]) == 3000.0 and float(out.max()) == 3000.0
    m = zdr_column_metrics(out, min_depth=1000.0)
    expected_area = float((out >= 1000.0).sum()) * 1.0
    assert float(m.ZDR_COLUMN_AREA) == pytest.approx(expected_area)
    assert float(m.ZDR_COLUMN_MAX_DEPTH) == 3000.0


@pytest.mark.parametrize("engine", ENGINES)
def test_minimum_reflectivity_removes_weak_echo(engine):
    da = grid(column(4000.0, 6000.0))
    dbz = xr.full_like(da, 40.0).rename("DBZH")
    dbz[(Z >= 4500) & (Z < 4750), :, 0] = 10.0  # weak echo at one level
    ds = xr.Dataset({"ZDR": da, "DBZH": dbz})
    out = zdr_column(ds, min_dbz=20.0, freezing_level=4000.0, engine=engine)
    np.testing.assert_allclose(out[:, 0].values, 500.0)
    np.testing.assert_allclose(out[:, 1:].values, 2000.0)
    # no condition by default, and the field can be given by name or as a DataArray
    np.testing.assert_allclose(
        zdr_column(ds, freezing_level=4000.0, engine=engine), 2000.0
    )
    ds2 = xr.Dataset({"ZDR": da, "MYDBZ": dbz})
    np.testing.assert_allclose(
        zdr_column(
            ds2, dbz="MYDBZ", min_dbz=20.0, freezing_level=4000.0, engine=engine
        )[:, 0].values,
        500.0,
    )
    with pytest.raises(KeyError):
        zdr_column(da.to_dataset(), min_dbz=20.0, freezing_level=4000.0)
