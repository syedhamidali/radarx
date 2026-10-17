#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the KDP-ZDR separation vector and the size sorting parameter
======================================================================

Synthetic storms with a KDP blob and a ZDR blob at a known offset, moving at a known
speed, have exact separation distances, orientations and storm motions. The adaptive
threshold is the largest multiple of the step that keeps the required number of
points. The compiled kernel and the NumPy implementation must agree.
"""

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.retrieve import storm_size_sorting

sorting = __import__("radarx.retrieve.sorting", fromlist=["x"])

ENGINES = ["numpy"] + (["compiled"] if sorting.HAS_COMPILED_KERNEL else [])
compiled_only = pytest.mark.skipif(
    not sorting.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)

X = Y = np.arange(60) * 500.0
NT, STEP = 5, 300.0


def times(n=NT):
    return np.datetime64("2022-03-30T22:00") + np.arange(n) * np.timedelta64(
        int(STEP), "s"
    )


def blob(cx, cy, amplitude, sigma=2500.0):
    r2 = (X[None, :] - cx) ** 2 + (Y[:, None] - cy) ** 2
    return amplitude * np.exp(-r2 / (2 * sigma**2))


def storm(offset=(4000.0, 0.0), speed=(0.0, 13.3333333), n=NT, rhohv=0.97, extra=False):
    """KDP and ZDR blobs ``offset`` (ZDR minus KDP) apart, moving at ``speed`` m/s."""
    kdp, zdr = [], []
    for i in range(n):
        dx, dy = speed[0] * STEP * i, speed[1] * STEP * i
        kdp_i = blob(8000.0 + dx, 8000.0 + dy, 3.0)
        zdr_i = blob(8000.0 + dx + offset[0], 8000.0 + dy + offset[1], 4.0)
        if extra:  # a second storm elsewhere with the opposite offset
            kdp_i = kdp_i + blob(24000.0, 8000.0, 3.0)
            zdr_i = zdr_i + blob(24000.0 - offset[0], 8000.0 + offset[1], 4.0)
        kdp.append(kdp_i)
        zdr.append(zdr_i)
    shape = (n, len(Y), len(X))
    return xr.Dataset(
        {
            "KDP": (("time", "y", "x"), np.array(kdp)),
            "ZDR": (("time", "y", "x"), np.array(zdr)),
            "RHOHV": (("time", "y", "x"), np.full(shape, rhohv)),
        },
        coords={"time": times(n), "y": Y, "x": X},
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_separation_vector_and_the_motion_from_the_centroids(engine):
    res = storm_size_sorting(storm(), rhohv="RHOHV", engine=engine)
    assert res.sizes["time"] == NT
    # the ZDR blob is 4 km to the east of the KDP blob, the storm moves north
    np.testing.assert_allclose(res.SEPARATION_DISTANCE.values, 4.0, atol=1e-9)
    np.testing.assert_allclose(res.SEPARATION_BEARING.values, 90.0, atol=1e-9)
    np.testing.assert_allclose(res.STORM_MOTION_U.values, 0.0, atol=1e-6)
    np.testing.assert_allclose(res.STORM_MOTION_V.values, 13.3333333, rtol=1e-6)
    np.testing.assert_allclose(res.SEPARATION_ORIENTATION.values, 90.0, atol=1e-6)
    np.testing.assert_allclose(res.SIZE_SORTING_PARAMETER.values, 4.0, atol=1e-6)
    # the thresholds are multiples of the step and the regions have at least min_cells points
    for name in ("KDP_THRESHOLD", "ZDR_THRESHOLD"):
        assert ((res[name].values / 0.5) % 1 == 0).all()
    assert (res.KDP_REGION_AREA.values >= 10 * 0.25).all()
    assert res.SEPARATION_DISTANCE.attrs["units"] == "km"
    assert res.KDP_CENTROID_X.attrs["units"] == "m"
    # the centroids are those of the blobs
    np.testing.assert_allclose(res.KDP_CENTROID_X.values, 8000.0, atol=1e-9)
    np.testing.assert_allclose(res.ZDR_CENTROID_X.values, 12000.0, atol=1e-9)


@pytest.mark.parametrize("engine", ENGINES)
def test_orientation_is_clockwise_from_the_storm_motion(engine):
    # a storm moving east with the ZDR blob to the south is 90 degrees to its right
    res = storm_size_sorting(
        storm(offset=(0.0, -3000.0), speed=(13.3333333, 0.0)), engine=engine
    )
    np.testing.assert_allclose(res.SEPARATION_ORIENTATION.values, 90.0, atol=1e-6)
    np.testing.assert_allclose(res.SEPARATION_BEARING.values, 180.0, atol=1e-9)
    # to the north of the motion it is to the left
    res = storm_size_sorting(
        storm(offset=(0.0, 3000.0), speed=(13.3333333, 0.0)), engine=engine
    )
    np.testing.assert_allclose(res.SEPARATION_ORIENTATION.values, -90.0, atol=1e-6)
    np.testing.assert_allclose(res.SIZE_SORTING_PARAMETER.values, 3.0, atol=1e-6)
    # along the motion the orientation is zero and the parameter too, against it 180 degrees
    res = storm_size_sorting(
        storm(offset=(2000.0, 0.0), speed=(13.3333333, 0.0)), engine=engine
    )
    np.testing.assert_allclose(res.SEPARATION_ORIENTATION.values, 0.0, atol=1e-5)
    np.testing.assert_allclose(res.SIZE_SORTING_PARAMETER.values, 0.0, atol=1e-5)
    res = storm_size_sorting(
        storm(offset=(-2000.0, 0.0), speed=(13.3333333, 0.0)), engine=engine
    )
    np.testing.assert_allclose(abs(res.SEPARATION_ORIENTATION.values), 180.0, atol=1e-5)


@pytest.mark.parametrize("engine", ENGINES)
def test_given_motion_wind_and_regions(engine):
    ds = storm(extra=True)
    # the whole grid has two storms: the regions are the 10 points of the highest values
    # of either, so the box selects the storm
    box = storm_size_sorting(
        ds, rhohv="RHOHV", box=(0.0, 16000.0, 0.0, 30000.0), engine=engine
    )
    np.testing.assert_allclose(box.SEPARATION_DISTANCE.values, 4.0, atol=1e-6)
    far = storm_size_sorting(
        ds,
        box=(18000.0, 29500.0, 0.0, 29500.0),
        motion=(0.0, 13.3333333),
        engine=engine,
    )
    # the second storm has the ZDR blob to the west: 90 degrees to the left
    np.testing.assert_allclose(far.SEPARATION_ORIENTATION.values[0], -90.0, atol=1e-5)
    # a mask has the same effect (also without time dimension)
    mask = (ds.x <= 16000.0) & (ds.y <= 30000.0)
    masked = storm_size_sorting(ds, region=mask, engine=engine)
    xr.testing.assert_allclose(masked.SEPARATION_DISTANCE, box.SEPARATION_DISTANCE)
    # a motion given as (u, v), per time, or as a dataset replaces the fitted one
    given = storm_size_sorting(ds, region=mask, motion=(10.0, 0.0), engine=engine)
    np.testing.assert_allclose(given.STORM_MOTION_U.values, 10.0)
    np.testing.assert_allclose(given.SEPARATION_ORIENTATION.values, 0.0, atol=1e-6)
    per_time = xr.Dataset(
        {"u": ("time", np.zeros(NT)), "v": ("time", np.full(NT, 5.0))}
    )
    again = storm_size_sorting(ds, region=mask, motion=per_time, engine=engine)
    np.testing.assert_allclose(again.SEPARATION_ORIENTATION.values, 90.0, atol=1e-6)
    pair = storm_size_sorting(
        ds,
        region=mask,
        motion=(xr.DataArray(np.zeros(NT), dims="time"), 5.0),
        engine=engine,
    )
    xr.testing.assert_allclose(
        pair.SEPARATION_ORIENTATION, again.SEPARATION_ORIENTATION
    )
    # the storm-relative wind: the wind of the storm motion minus the vector points to the
    # west for a southerly... here: wind (0, 10), motion (0, 13.33): storm-relative wind
    # southward, the vector (east) is at 90 degrees from the direction opposite to it
    wind = storm_size_sorting(ds, region=mask, wind=(0.0, 10.0), engine=engine)
    np.testing.assert_allclose(wind.SORTING_WIND_ANGLE.values, 90.0, atol=1e-5)
    aligned = storm_size_sorting(
        ds, region=mask, wind=(-10.0, 13.3333333), engine=engine
    )  # SR wind to the west
    np.testing.assert_allclose(aligned.SORTING_WIND_ANGLE.values, 0.0, atol=1e-5)
    assert "SORTING_WIND_ANGLE" not in box


@pytest.mark.parametrize("engine", ENGINES)
def test_adaptive_threshold_follows_the_signature_and_not_the_bias(engine):
    ds = storm()
    base = storm_size_sorting(ds, engine=engine)
    # a ZDR bias of a whole step moves the threshold by a step and nothing else
    biased = ds.assign(ZDR=ds.ZDR + 0.5)
    shifted = storm_size_sorting(biased, engine=engine)
    np.testing.assert_allclose(
        shifted.ZDR_THRESHOLD.values, base.ZDR_THRESHOLD.values + 0.5
    )
    np.testing.assert_allclose(
        shifted.ZDR_CENTROID_X.values, base.ZDR_CENTROID_X.values
    )
    # a stronger signature gets a higher threshold
    strong = storm_size_sorting(ds.assign(KDP=ds.KDP * 2), engine=engine)
    assert (strong.KDP_THRESHOLD.values > base.KDP_THRESHOLD.values).all()
    # more required points: a lower threshold and a larger area
    many = storm_size_sorting(ds, min_cells=100, engine=engine)
    assert (many.KDP_THRESHOLD.values < base.KDP_THRESHOLD.values).all()
    assert (many.KDP_REGION_AREA.values > base.KDP_REGION_AREA.values).all()
    # a finer step
    fine = storm_size_sorting(ds, step=0.1, engine=engine)
    assert (fine.KDP_THRESHOLD.values >= base.KDP_THRESHOLD.values - 1e-12).all()
    # the correlation coefficient removes points: below the minimum there is no region
    low = storm_size_sorting(storm(rhohv=0.9), rhohv="RHOHV", engine=engine)
    assert (
        np.isnan(low.SEPARATION_DISTANCE.values).all()
        and np.isnan(low.KDP_THRESHOLD.values).all()
    )
    ok = storm_size_sorting(
        storm(rhohv=0.9), rhohv="RHOHV", min_rhohv=0.85, engine=engine
    )
    assert np.isfinite(ok.SEPARATION_DISTANCE.values).all()
    # fewer points than required, and values that stay below the floor of zero
    assert np.isnan(
        storm_size_sorting(ds, min_cells=10**6, engine=engine).KDP_THRESHOLD.values
    ).all()
    negative = ds.assign(KDP=-ds.KDP - 1.0)
    res = storm_size_sorting(negative, engine=engine)
    assert (
        np.isnan(res.KDP_THRESHOLD.values).all()
        and np.isnan(res.SEPARATION_DISTANCE.values).all()
    )
    assert (res.KDP_REGION_AREA.isnull()).all()


@pytest.mark.parametrize("engine", ENGINES)
def test_grids_without_time_and_with_levels(engine):
    ds = storm()
    one = ds.isel(time=0, drop=True)
    res = storm_size_sorting(one, motion=(0.0, 10.0), engine=engine)
    assert "time" not in res.coords and res.SEPARATION_DISTANCE.shape == (1,)
    np.testing.assert_allclose(res.SEPARATION_ORIENTATION.values, 90.0, atol=1e-6)
    # no motion to fit from one volume
    assert np.isnan(
        storm_size_sorting(one, engine=engine).SEPARATION_ORIENTATION.values
    ).all()
    # the lowest level, or the nearest to the level asked for
    levels = xr.concat(
        [ds, ds.assign(ZDR=ds.ZDR * 0.1)],
        dim=xr.DataArray([500.0, 3000.0], dims="z", name="z"),
    )
    low = storm_size_sorting(levels, engine=engine)
    xr.testing.assert_allclose(
        low.SEPARATION_DISTANCE,
        storm_size_sorting(ds, engine=engine).SEPARATION_DISTANCE,
    )
    high = storm_size_sorting(levels, level=2800.0, engine=engine)
    assert not np.allclose(high.ZDR_THRESHOLD.values, low.ZDR_THRESHOLD.values)
    # a single fitted time gives no motion
    two = storm_size_sorting(ds.isel(time=slice(0, 2)), engine=engine)
    np.testing.assert_allclose(two.STORM_MOTION_V.values, 13.3333333, rtol=1e-6)
    constant = ds.isel(time=[0, 0])
    res = storm_size_sorting(constant, engine=engine)
    assert np.isnan(res.STORM_MOTION_U.values).all() or np.allclose(
        res.STORM_MOTION_U.values, 0.0
    )
    # numbers instead of datetimes as the time coordinate
    numeric = ds.assign_coords(time=np.arange(NT) * STEP)
    np.testing.assert_allclose(
        storm_size_sorting(numeric, engine=engine).STORM_MOTION_V.values,
        13.3333333,
        rtol=1e-6,
    )


@compiled_only
def test_engines_agree_on_noisy_fields():
    rnd = np.random.default_rng(5)
    ds = storm()
    noisy = ds.assign(
        KDP=ds.KDP + rnd.normal(0, 0.3, ds.KDP.shape),
        ZDR=ds.ZDR + rnd.normal(0, 0.4, ds.ZDR.shape),
    )
    noisy["KDP"] = noisy.KDP.where(rnd.random(ds.KDP.shape) > 0.05)
    a = storm_size_sorting(noisy, rhohv="RHOHV", engine="numpy")
    b = storm_size_sorting(noisy, rhohv="RHOHV", engine="compiled", n_threads=2)
    for name in a.data_vars:
        np.testing.assert_allclose(
            a[name].values, b[name].values, rtol=1e-12, equal_nan=True
        )


@pytest.mark.parametrize("engine", ENGINES)
def test_adaptive_threshold_equals_the_search_of_the_paper(engine):
    """Loeffler and Kumjian (2018): the largest threshold in steps of 0.5 with N_G points."""
    rnd = np.random.default_rng(11)
    ds = storm()
    values = np.where(
        rnd.random((NT, len(Y), len(X))) < 0.2,
        np.nan,
        rnd.gamma(2.0, 1.2, (NT, len(Y), len(X))),
    )
    ds = ds.assign(
        KDP=(("time", "y", "x"), values), ZDR=(("time", "y", "x"), values[::-1])
    )
    xx, yy = np.meshgrid(X, Y)
    for field, label in (("KDP", "KDP"), ("ZDR", "ZDR")):
        res = storm_size_sorting(ds, min_cells=25, engine=engine)
        for t in range(NT):
            v = ds[field].values[t]
            best = None
            for threshold in np.arange(0.0, np.nanmax(v) + 0.5, 0.5):
                if (v >= threshold).sum() >= 25:  # NaN compares false
                    best = threshold
            chosen = np.isfinite(v) & (v >= best)
            assert res[f"{label}_THRESHOLD"].values[t] == best
            assert res[f"{label}_CENTROID_X"].values[t] == np.median(xx[chosen])
            assert res[f"{label}_CENTROID_Y"].values[t] == np.median(yy[chosen])


def test_errors():
    ds = storm()
    with pytest.raises(TypeError, match="Dataset"):
        storm_size_sorting(ds.KDP)
    with pytest.raises(ValueError, match="engine"):
        storm_size_sorting(ds, engine="gpu")
    with pytest.raises(ValueError, match="min_cells"):
        storm_size_sorting(ds, min_cells=0)
    with pytest.raises(ValueError, match="step"):
        storm_size_sorting(ds, step=0.0)
    with pytest.raises(KeyError):
        storm_size_sorting(ds.drop_vars("KDP"))
    with pytest.raises(KeyError):
        storm_size_sorting(ds, rhohv="NOPE")
    with pytest.raises(ValueError, match="coordinate"):
        storm_size_sorting(ds.drop_vars("x"))
    with pytest.raises(ValueError, match="only"):
        storm_size_sorting(ds.expand_dims(ensemble=2))
    with pytest.raises(TypeError, match="motion"):
        storm_size_sorting(ds, motion=5.0)
    with pytest.raises(ValueError, match="one value per time"):
        storm_size_sorting(ds, motion=(xr.DataArray(np.zeros(3), dims="time"), 1.0))
    with pytest.raises(TypeError, match="wind"):
        storm_size_sorting(ds, wind=3.0)


def test_accessor_dask_and_the_input_is_unchanged():
    ds = storm()
    before = ds.copy(deep=True)
    res = storm_size_sorting(ds, rhohv="RHOHV")
    xr.testing.assert_identical(ds, before)
    xr.testing.assert_equal(ds.radarx.storm_size_sorting(rhohv="RHOHV"), res)
    pytest.importorskip("dask")
    xr.testing.assert_allclose(
        storm_size_sorting(ds.chunk({"time": 2}), rhohv="RHOHV"), res
    )


def size_sorting_case():
    """The arguments of ``storm_size_sorting`` for the dask test of the functions that compute."""
    return storm_size_sorting, (storm(),), {"rhohv": "RHOHV"}
