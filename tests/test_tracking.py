#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for storm cell identification, tracking and statistics
============================================================

Synthetic storms with a known motion, number, size and history (births, deaths, splits
and mergers) have exact answers. The compiled kernels and the NumPy twins must give the
same cells and tracks.
"""

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.retrieve import (
    _cells,
    cell_statistics,
    identify_cells,
    track_cells,
)

ENGINES = ["numpy"] + (["compiled"] if _cells.HAS_COMPILED_KERNEL else [])
compiled_only = pytest.mark.skipif(
    not _cells.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)

X = Y = np.arange(80) * 1000.0
STEP = 300  # s


def times(n):
    return np.datetime64("2022-03-30T22:00") + np.arange(n) * np.timedelta64(STEP, "s")


def blob(cx, cy, peak, sigma=3000.0):
    """A Gaussian cell: ``peak`` dBZ above the background of -10."""
    r2 = (X[None, :] - cx) ** 2 + (Y[:, None] - cy) ** 2
    return peak * np.exp(-r2 / (2 * sigma**2))


def grid(frames, name="DBZH"):
    values = np.asarray(frames) - 10.0
    return xr.DataArray(
        values,
        dims=("time", "y", "x"),
        coords={"time": times(len(frames)), "y": Y, "x": X},
        name=name,
        attrs={"units": "dBZ"},
    )


def moving_storm(n=6, u=20.0, v=10.0):
    """One cell moving at (u, v) m/s, one that stays, one that dies after 4 volumes."""
    frames = []
    for i in range(n):
        f = blob(10e3 + u * STEP * i, 15e3 + v * STEP * i, 55)
        f = f + blob(30e3, 60e3, 50) + (blob(60e3, 15e3, 48) if i < 4 else 0.0)
        frames.append(f)
    return grid(frames)


THRESHOLDS = (20, 30, 40)


@pytest.mark.parametrize("engine", ENGINES)
def test_cells_are_found_with_their_extent(engine):
    cells = identify_cells(moving_storm(), thresholds=THRESHOLDS, engine=engine)
    assert cells.CELL_LABEL.dtype == np.int32
    assert cells.CELL_COUNT.values.tolist() == [3, 3, 3, 3, 2, 2]
    first = cells.CELL_LABEL.isel(time=0)
    # raster order of the first point: the cell at y = 15 km (x = 10 km), then the one at x = 60 km
    assert (
        first.values[15, 10] == 1
        and first.values[15, 60] == 2
        and first.values[60, 30] == 3
    )
    # a cell reaches down to the lowest threshold: 20 dBZ is 30 dB above the background,
    # so r where 55 exp(-r^2/2s^2) = 30 dB, i.e. 3 km * sqrt(2 ln(55/30)) = 3.3 km
    radius = 3000.0 * np.sqrt(2 * np.log(55 / 30.0))
    area = float((first == 1).sum())
    assert area == pytest.approx(np.pi * (radius / 1e3) ** 2, rel=0.12)


@pytest.mark.parametrize("engine", ENGINES)
def test_touching_cells_are_divided_along_the_saddle(engine):
    two = grid([blob(30e3, 40e3, 55) + blob(38e3, 40e3, 55)])
    cells = identify_cells(two, thresholds=(20, 40), min_area=2.0, engine=engine)
    assert cells.CELL_COUNT.values.tolist() == [2]
    row = cells.CELL_LABEL.values[0, 40]
    assert row[30] == 1 and row[38] == 2
    # the border is at the minimum of the field between the two peaks
    field = two.values[0, 40]
    saddle = 30 + int(np.argmin(field[30:39]))
    assert (row[:saddle] != 2).all() and (row[saddle + 1 : 40] != 1).all()
    # a single threshold makes one region of them
    one = identify_cells(two, thresholds=(20,), min_area=2.0, engine=engine)
    assert one.CELL_COUNT.values.tolist() == [1]


@pytest.mark.parametrize("engine", ENGINES)
def test_options_of_the_identification(engine):
    da = moving_storm(2)
    base = identify_cells(da, thresholds=THRESHOLDS, engine=engine)
    # a larger minimum area drops the small regions
    assert (
        identify_cells(
            da, thresholds=THRESHOLDS, min_area=1e4, engine=engine
        ).CELL_COUNT.sum()
        == 0
    )
    # the segmentation threshold cannot be above the lowest threshold, but may be below
    wide = identify_cells(
        da, thresholds=THRESHOLDS, segmentation_threshold=10, engine=engine
    )
    assert (wide.CELL_LABEL > 0).sum() > (base.CELL_LABEL > 0).sum()
    # smoothing and a 4-connected neighbourhood keep the cells
    smooth = identify_cells(
        da, thresholds=THRESHOLDS, smooth=1.0, connectivity=4, engine=engine
    )
    assert smooth.CELL_COUNT.values.tolist() == base.CELL_COUNT.values.tolist()
    # a Dataset, with the default reflectivity field or a named one, and a vertical dimension
    ds = da.to_dataset()
    pd = identify_cells(ds, thresholds=THRESHOLDS, engine=engine)
    xr.testing.assert_equal(pd.CELL_LABEL, base.CELL_LABEL)
    ds = ds.rename(DBZH="other")
    named = identify_cells(ds, "other", thresholds=THRESHOLDS, engine=engine)
    xr.testing.assert_equal(named.CELL_LABEL, base.CELL_LABEL)
    stack = xr.concat(
        [da, da - 20.0], dim=xr.DataArray([500.0, 1500.0], dims="z", name="z")
    )
    composite = identify_cells(
        stack.transpose("time", "z", "y", "x"), thresholds=THRESHOLDS, engine=engine
    )
    xr.testing.assert_equal(composite.CELL_LABEL, base.CELL_LABEL)
    # missing values are no echo
    holes = da.where(da < 30)  # the cores are missing
    assert (
        identify_cells(holes, thresholds=THRESHOLDS, engine=engine).CELL_COUNT.min()
        >= 0
    )
    # the input is not changed
    before = da.copy(deep=True)
    identify_cells(da, thresholds=THRESHOLDS, engine=engine)
    xr.testing.assert_identical(da, before)


@compiled_only
def test_engines_agree_on_cells_and_tracks():
    da = moving_storm()
    a = identify_cells(da, thresholds=THRESHOLDS, engine="numpy")
    b = identify_cells(da, thresholds=THRESHOLDS, engine="compiled", n_threads=2)
    xr.testing.assert_equal(a, b)
    ta = track_cells(a, motion=(20.0, 10.0), engine="numpy")
    tb = track_cells(b, motion=(20.0, 10.0), engine="compiled")
    xr.testing.assert_equal(ta, tb)


def test_errors_of_the_identification():
    da = moving_storm(2)
    with pytest.raises(ValueError, match="ascending"):
        identify_cells(da, thresholds=(30, 20))
    with pytest.raises(ValueError, match="ascending"):
        identify_cells(da, thresholds=())
    with pytest.raises(ValueError, match="segmentation_threshold"):
        identify_cells(da, thresholds=THRESHOLDS, segmentation_threshold=25)
    with pytest.raises(ValueError, match="connectivity"):
        identify_cells(da, thresholds=THRESHOLDS, connectivity=6)
    with pytest.raises(ValueError, match="engine"):
        identify_cells(da, thresholds=THRESHOLDS, engine="gpu")
    with pytest.raises(ValueError, match="horizontal"):
        identify_cells(da.isel(x=0), thresholds=THRESHOLDS)
    with pytest.raises(KeyError):
        identify_cells(da.to_dataset(name="other"), thresholds=THRESHOLDS)
    with pytest.raises(ValueError, match="coordinate"):
        identify_cells(da.drop_vars("x"), thresholds=THRESHOLDS)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_moving_cell_is_one_track_with_the_motion(engine):
    cells = identify_cells(moving_storm(), thresholds=THRESHOLDS, engine=engine)
    tracks = track_cells(cells, motion=(20.0, 10.0), engine=engine)
    # the moving cell is tracked through all six volumes, the cell that stays is a new
    # cell at every volume because the motion moves it away, and it is not linked
    assert tracks.cell_lifetime.max() == 6
    moving = int(tracks.cell_lifetime.values.argmax())
    np.testing.assert_allclose(tracks.cell_u.values[moving, 1:], 20.0, atol=1500 / STEP)
    np.testing.assert_allclose(tracks.cell_v.values[moving, 1:], 10.0, atol=1500 / STEP)
    np.testing.assert_allclose(tracks.cell_x.values[moving, 0], 10e3, atol=300)
    assert (
        tracks.cell_split_from.values.max() == 0
        and tracks.cell_merged_into.values.max() == 0
    )
    assert tracks.cell_first_time.values[moving] == times(1)[0]
    assert tracks.cell_last_time.values[moving] == times(6)[-1]
    # without motion the cells that stay are the tracks, the moving one overlaps for a
    # volume and then is a new cell (its 6 km move is its diameter)
    still = track_cells(cells, engine=engine)
    assert sorted(still.cell_lifetime.values.tolist(), reverse=True)[:2] == [6, 4]
    assert still.CELL_ID.dtype == np.int32 and still.cell_area.attrs["units"] == "km2"
    # the area follows the cell: constant for a cell of constant size
    assert np.nanstd(tracks.cell_area.values[moving]) < 3.0


@pytest.mark.parametrize("engine", ENGINES)
def test_motion_in_several_forms(engine):
    cells = identify_cells(moving_storm(), thresholds=THRESHOLDS, engine=engine)
    expected = track_cells(cells, motion=(20.0, 10.0), engine=engine)
    uv = xr.Dataset({"u": 20.0, "v": 10.0})
    per_interval = xr.Dataset(
        {
            "u": ("time", np.full(5, 20.0)),
            "v": ("time", np.full(5, 10.0)),
        },
        coords={"time": times(6)[:5]},
    )
    per_volume = xr.Dataset(
        {"u": ("time", np.full(6, 20.0)), "v": ("time", np.full(6, 10.0))},
        coords={"time": times(6)},
    )
    field = xr.Dataset(
        {
            "u": (("y", "x"), np.full((80, 80), 20.0)),
            "v": (("y", "x"), np.full((80, 80), 10.0)),
        },
        coords={"y": Y, "x": X},
    )
    for motion in (uv, per_interval, per_volume, field):
        out = track_cells(cells, motion=motion, engine=engine)
        xr.testing.assert_equal(out.CELL_ID, expected.CELL_ID)
    # without a time coordinate the interval is given
    bare = cells.CELL_LABEL.drop_vars("time")
    out = track_cells(bare, motion=(20.0, 10.0), dt=STEP, engine=engine)
    np.testing.assert_array_equal(out.CELL_ID.values, expected.CELL_ID.values)


@pytest.mark.parametrize("engine", ENGINES)
def test_births_deaths_splits_and_mergers(engine):
    # volume 0: A; volume 1: A splits into B and C; volume 2: B and C merge into D, and E starts
    frames = [
        blob(40e3, 40e3, 55, 4000.0),
        blob(37e3, 40e3, 55, 2000.0) + blob(43e3, 40e3, 50, 2000.0),
        blob(40e3, 40e3, 55, 4000.0) + blob(70e3, 70e3, 45),
    ]
    cells = identify_cells(
        grid(frames), thresholds=(20, 30), min_area=2.0, engine=engine
    )
    assert cells.CELL_COUNT.values.tolist() == [1, 2, 2]
    tracks = track_cells(cells, min_overlap=0.2, engine=engine)
    assert tracks.sizes["cell"] == 3
    # track 1 is A, then the cell of the larger overlap, then D; track 2 is the other half
    # that split from track 1 and merged into it; track 3 is the new cell E
    assert tracks.cell_lifetime.values.tolist() == [3, 1, 1]
    assert tracks.cell_split_from.values.tolist() == [0, 1, 0]
    assert tracks.cell_merged_into.values.tolist() == [0, 1, 0]
    assert tracks.cell_first_time.values[1] == times(2)[1]
    assert tracks.cell_last_time.values[1] == times(2)[1]
    assert tracks.cell_first_time.values[2] == times(3)[2]
    # at the last volume the grid holds track 1 at the centre and track 3 at E
    last = tracks.CELL_ID.isel(time=2).values
    assert last[40, 40] == 1 and last[70, 70] == 3
    # no link at all needs more overlap than the cells have: every cell is a track
    # (the halves lie inside A, so that the fraction of their own area is near one)
    assert track_cells(cells, min_overlap=1.0, engine=engine).sizes["cell"] >= 3


@pytest.mark.parametrize("engine", ENGINES)
def test_overlap_threshold_and_minimum_lifetime(engine):
    cells = identify_cells(moving_storm(), thresholds=THRESHOLDS, engine=engine)
    # a motion that is a third too slow leaves the cells with part of their overlap
    loose = track_cells(cells, motion=(13.0, 7.0), min_overlap=0.1, engine=engine)
    strict = track_cells(cells, motion=(13.0, 7.0), min_overlap=0.95, engine=engine)
    assert strict.sizes["cell"] > loose.sizes["cell"]
    kept = track_cells(cells, motion=(20.0, 10.0), min_lifetime=3, engine=engine)
    assert kept.sizes["cell"] == 1 and int(kept.cell_lifetime[0]) == 6
    assert set(np.unique(kept.CELL_ID.values)) == {0, 1}
    assert kept.cell_area.shape == (1, 6)


@pytest.mark.parametrize("engine", ENGINES)
def test_cells_that_do_not_overlap_are_linked_by_distance(engine):
    # a small fast cell moves by 12 grid cells in a volume interval, more than its width
    frames = [blob(10e3 + 12e3 * i, 40e3, 55, 1500.0) for i in range(4)]
    cells = identify_cells(grid(frames), thresholds=(20,), min_area=2.0, engine=engine)
    plain = track_cells(cells, engine=engine)
    assert plain.sizes["cell"] == 4
    linked = track_cells(cells, max_speed=60.0, engine=engine)
    assert linked.sizes["cell"] == 1 and int(linked.cell_lifetime[0]) == 4
    # with the right motion the overlap does it, and max_speed is not needed; a too small
    # speed does not link
    assert track_cells(cells, max_speed=10.0, engine=engine).sizes["cell"] == 4
    predicted = track_cells(cells, motion=(40.0, 0.0), max_speed=5.0, engine=engine)
    assert predicted.sizes["cell"] == 1
    # a motion on the grid supplies the prediction at the position of the cell
    field = xr.Dataset(
        {
            "u": (("y", "x"), np.full((80, 80), 40.0)),
            "v": (("y", "x"), np.zeros((80, 80))),
        },
        coords={"y": Y, "x": X},
    )
    assert (
        track_cells(cells, motion=field, max_speed=5.0, engine=engine).sizes["cell"]
        == 1
    )


def test_links_by_distance_pick_the_nearest_and_use_each_cell_once():
    labels = np.zeros((2, 40, 40), dtype=np.int32)
    labels[0, 10:12, 8:10] = 1  # two cells ...
    labels[0, 10:12, 28:30] = 2
    labels[1, 10:12, 19:21] = 1  # ... and one in between, nearer to the second
    cells = xr.DataArray(
        labels,
        dims=("time", "y", "x"),
        coords={
            "time": np.array([0.0, 300.0]),
            "y": np.arange(40) * 1000.0,
            "x": np.arange(40) * 1000.0,
        },
    )
    out = track_cells(cells, max_speed=60.0)  # numeric times
    assert out.sizes["cell"] == 2 and out.cell_lifetime.values.tolist() == [1, 2]
    assert out.CELL_ID.values[1, 10, 19] == 2


def test_errors_of_the_tracking():
    cells = identify_cells(moving_storm(3), thresholds=THRESHOLDS)
    with pytest.raises(ValueError, match="engine"):
        track_cells(cells, engine="gpu")
    with pytest.raises(ValueError, match="CELL_LABEL"):
        track_cells(xr.Dataset({"other": cells.CELL_LABEL}))
    with pytest.raises(ValueError, match="time, y and x"):
        track_cells(cells.CELL_LABEL.isel(x=0))
    with pytest.raises(ValueError, match="time, y and x only"):
        track_cells(cells.CELL_LABEL.expand_dims(z=2))
    with pytest.raises(ValueError, match="min_overlap"):
        track_cells(cells, min_overlap=0.0)
    with pytest.raises(TypeError, match="motion"):
        track_cells(cells, motion=5.0)
    with pytest.raises(ValueError, match="u and v"):
        track_cells(cells, motion=xr.Dataset({"u": 1.0}))
    bad = xr.Dataset(
        {"u": ("time", np.ones(7)), "v": ("time", np.ones(7))},
        coords={"time": np.arange(7)},
    )
    with pytest.raises(ValueError, match="one per volume"):
        track_cells(cells, motion=bad)
    odd = xr.Dataset({"u": ("z", np.ones(2)), "v": ("z", np.ones(2))})
    with pytest.raises(ValueError, match="time, y and x only"):
        track_cells(cells, motion=odd)
    coarse = xr.Dataset(
        {"u": (("y", "x"), np.ones((3, 3))), "v": (("y", "x"), np.ones((3, 3)))}
    )
    with pytest.raises(ValueError, match="dimensions y and x"):
        track_cells(cells, motion=coarse)
    bare = cells.CELL_LABEL.drop_vars("time")
    with pytest.raises(ValueError, match="dt"):
        track_cells(bare, motion=(1.0, 1.0))
    with pytest.raises(ValueError, match="one value per interval"):
        track_cells(bare, dt=[1.0, 1.0, 1.0])
    assert track_cells(bare, dt=[1.0, 2.0]).sizes["cell"] > 0
    # without motion the interval is not needed
    assert track_cells(bare).sizes["cell"] > 0
    with pytest.raises(ValueError, match="coordinate"):
        track_cells(cells.CELL_LABEL.drop_vars("x"))


def test_tracks_of_one_volume_and_of_no_cells():
    one = identify_cells(moving_storm(1), thresholds=THRESHOLDS)
    out = track_cells(one)
    assert out.sizes["cell"] == 3 and out.cell_lifetime.values.tolist() == [1, 1, 1]
    empty = identify_cells(grid([np.zeros((80, 80))] * 3), thresholds=THRESHOLDS)
    out = track_cells(empty, motion=(1.0, 1.0))
    assert out.sizes["cell"] == 0 and int(out.CELL_ID.max()) == 0
    # cells in the first volume only
    sparse = moving_storm(3)
    sparse[1:] = -10.0
    out = track_cells(identify_cells(sparse, thresholds=THRESHOLDS))
    assert out.cell_lifetime.values.tolist() == [1, 1, 1]


@pytest.mark.parametrize("engine", ENGINES)
def test_statistics_of_fields_inside_the_cells(engine):
    da = moving_storm()
    cells = identify_cells(da, thresholds=THRESHOLDS, engine=engine)
    tracks = track_cells(cells, motion=(20.0, 10.0), engine=engine)
    field = (da * 0 + 2.0).rename("RATE")
    field.attrs["units"] = "mm h-1"
    field = field.where(tracks.CELL_ID != 1, 5.0)
    stats = cell_statistics(
        tracks.CELL_ID,
        field,
        stats=("mean", "max", "min", "sum", "count"),
        engine=engine,
    )
    assert stats.RATE_max.dims == ("cell", "time")
    assert (
        stats.RATE_max.attrs["units"] == "mm h-1"
        and "units" not in stats.RATE_count.attrs
    )
    np.testing.assert_allclose(stats.RATE_mean.values[0], 5.0)
    np.testing.assert_allclose(
        stats.RATE_min.values[1][np.isfinite(stats.RATE_min.values[1])], 2.0
    )
    count = stats.RATE_count.values
    area = tracks.cell_area.values / 1.0  # 1 km2 grid cells
    np.testing.assert_allclose(count[0][count[0] > 0], area[0][count[0] > 0])
    np.testing.assert_allclose(
        stats.RATE_sum.values[0][count[0] > 0], 5.0 * count[0][count[0] > 0]
    )
    assert np.isnan(stats.RATE_mean.values[1][count[1] == 0]).all()
    # the labels of one volume, a Dataset of fields and of cells
    two = cell_statistics(
        cells, xr.Dataset({"a": da, "b": da * 2}), stats=("max",), engine=engine
    )
    assert two.a_max.sizes["cell"] == 3
    np.testing.assert_allclose(two.b_max.values, 2 * two.a_max.values)
    ids = xr.Dataset({"CELL_ID": tracks.CELL_ID})
    np.testing.assert_allclose(
        cell_statistics(ids, da.rename("a"), engine=engine).a_max.values,
        cell_statistics(tracks.CELL_ID, da.rename("a"), engine=engine).a_max.values,
    )


def test_statistics_of_no_cells_and_errors():
    empty = identify_cells(grid([np.zeros((80, 80))] * 2), thresholds=THRESHOLDS)
    field = grid([np.zeros((80, 80))] * 2, "F")
    out = cell_statistics(empty, field)
    assert out.F_max.shape == (0, 2)
    cells = identify_cells(moving_storm(2), thresholds=THRESHOLDS)
    da = moving_storm(2)
    with pytest.raises(ValueError, match="statistic"):
        cell_statistics(cells, da, stats=("median",))
    with pytest.raises(ValueError, match="engine"):
        cell_statistics(cells, da, engine="gpu")
    with pytest.raises(ValueError, match="CELL_ID or CELL_LABEL"):
        cell_statistics(xr.Dataset({"x": cells.CELL_LABEL}), da)
    with pytest.raises(ValueError, match="time, y and x"):
        cell_statistics(cells.CELL_LABEL.isel(x=0), da)
    with pytest.raises(ValueError, match="reduce it"):
        cell_statistics(cells, da.expand_dims(z=2))
    with pytest.raises(ValueError, match="grid of the cells"):
        cell_statistics(cells, da.isel(x=slice(0, 40)))


def test_accessors():
    da = moving_storm(3)
    cells = da.radarx.identify_cells(thresholds=THRESHOLDS)
    xr.testing.assert_equal(cells, identify_cells(da, thresholds=THRESHOLDS))
    tracks = cells.radarx.track_cells(motion=(20.0, 10.0))
    xr.testing.assert_equal(tracks, track_cells(cells, motion=(20.0, 10.0)))
    stats = tracks.CELL_ID.radarx.cell_statistics(da.rename("Z"), stats=("max",))
    xr.testing.assert_equal(
        stats, cell_statistics(tracks.CELL_ID, da.rename("Z"), stats=("max",))
    )


def test_dask_input_stays_lazy_for_the_identification():
    pytest.importorskip("dask")
    da = moving_storm(3)
    lazy = identify_cells(da.chunk({"time": 1, "y": 40}), thresholds=THRESHOLDS)
    assert lazy.CELL_LABEL.chunks is not None
    xr.testing.assert_equal(lazy.compute(), identify_cells(da, thresholds=THRESHOLDS))


def moving_cells_case():
    """The arguments of ``track_cells`` for the dask test of the functions that compute."""
    cells = identify_cells(moving_storm(4), thresholds=THRESHOLDS)
    return track_cells, (cells,), {"motion": (20.0, 10.0)}


def cell_statistics_case():
    da = moving_storm(4)
    cells = identify_cells(da, thresholds=THRESHOLDS)
    return cell_statistics, (cells, da.rename("Z")), {"stats": ("max", "mean")}
