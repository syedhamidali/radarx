# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the return period, the areal reduction factor and the climatological reference
========================================================================================

Known answers: the logarithmic interpolation worked by hand, the table values
themselves, the flags outside the table, the correction of an area mean by a
factor, and the quantile of a record against ``numpy.quantile``.
"""

import math

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.retrieve import (
    areal_reduction_factor,
    climatological_reference,
    return_period,
)

PERIODS = (2.0, 10.0, 100.0)
TABLE = (40.0, 80.0, 160.0)


def _field(value, shape=(4, 5)):
    return xr.DataArray(
        np.full(shape, value),
        dims=("y", "x"),
        coords={"y": np.arange(shape[0]), "x": np.arange(shape[1])},
    )


def test_log_log_interpolation_known_answer():
    # 60 mm lies between 40 mm (2 y) and 80 mm (10 y): fraction ln(60/40)/ln(80/40) of the way in ln(T)
    want = math.exp(
        math.log(2.0) + math.log(1.5) / math.log(2.0) * (math.log(10.0) - math.log(2.0))
    )
    out = return_period(_field(60.0), PERIODS, TABLE)
    assert float(out.return_period[0, 0]) == pytest.approx(want, rel=1e-6)
    assert int(out["flag"][0, 0]) == 0
    assert (
        out.return_period.attrs["units"] == "year"
        and out.return_period.dtype == np.float32
    )


@pytest.mark.parametrize("depth, years", [(40.0, 2.0), (80.0, 10.0), (160.0, 100.0)])
def test_table_depths_give_the_table_periods(depth, years):
    out = return_period(_field(depth), PERIODS, TABLE)
    assert float(out.return_period[1, 1]) == pytest.approx(years, rel=1e-6)
    assert int(out["flag"][1, 1]) == 0


def test_outside_the_table_is_flagged_not_extrapolated():
    low = return_period(_field(20.0), PERIODS, TABLE)
    assert float(low.return_period[0, 0]) == 2.0 and int(low["flag"][0, 0]) == -1
    high = return_period(_field(500.0), PERIODS, TABLE)
    assert float(high.return_period[0, 0]) == 100.0 and int(high["flag"][0, 0]) == 1


def test_a_table_per_cell_and_missing_values():
    depth = _field(60.0)
    table = xr.DataArray(
        np.stack([np.full((4, 5), v) for v in TABLE]),
        dims=("period", "y", "x"),
        coords={"period": list(PERIODS), "y": depth.y, "x": depth.x},
    )
    table[:, 0, 0] = (
        table[:, 0, 0] * 2.0
    )  # a wetter place: 60 mm is rarer elsewhere, commoner here
    table[1, 1, 1] = np.nan
    depth[2, 2] = np.nan
    out = return_period(depth, PERIODS, table)
    assert float(out.return_period[0, 0]) < float(out.return_period[3, 3])
    assert (
        float(out["flag"][0, 0]) == -1
    )  # 60 mm is below the 2-year depth of 80 mm there
    assert np.isnan(float(out.return_period[1, 1])) and np.isnan(
        float(out.return_period[2, 2])
    )


def test_a_table_that_falls_with_the_period_is_not_a_table():
    out = return_period(_field(60.0), PERIODS, (80.0, 40.0, 160.0))
    assert np.isnan(out.return_period.values).all()


@pytest.mark.parametrize(
    "periods, depths",
    [((10, 2), (80, 40)), ((2,), (40,)), ((2, 10), (40, 80, 160)), ((0, 10), (40, 80))],
)
def test_wrong_tables_are_errors(periods, depths):
    with pytest.raises(ValueError):
        return_period(_field(60.0), periods, depths)


def test_areal_reduction_converts_an_area_mean_to_the_point_depth():
    # an area mean of 40 mm with a factor of 0.8 is the 50 mm point depth: exactly the 2 year depth here
    f = areal_reduction_factor(300.0, [1.0], [0.8])
    assert f == 0.8
    out = return_period(_field(40.0 / f), PERIODS, (50.0, 80.0, 160.0))
    assert float(out.return_period[0, 0]) == pytest.approx(2.0, rel=1e-6)
    # without the factor the same area mean looks less rare
    direct = return_period(_field(40.0), PERIODS, (50.0, 80.0, 160.0))
    assert float(direct["flag"][0, 0]) == -1


def test_areal_reduction_curve_is_linear_in_log_area_and_held_at_the_ends():
    assert areal_reduction_factor(100.0, [10, 1000], [1.0, 0.8]) == pytest.approx(0.9)
    assert areal_reduction_factor(1.0, [10, 1000], [1.0, 0.8]) == pytest.approx(1.0)
    assert areal_reduction_factor(1e6, [10, 1000], [1.0, 0.8]) == pytest.approx(0.8)
    many = areal_reduction_factor(
        np.array([10.0, 100.0, 1000.0]), [10, 1000], [1.0, 0.8]
    )
    assert many == pytest.approx([1.0, 0.9, 0.8])


@pytest.mark.parametrize(
    "areas, factors",
    [
        ([10, 5], [1.0, 0.9]),
        ([10, 100], [1.0]),
        ([10, 100], [1.2, 0.9]),
        ([-1, 5], [1, 0.9]),
        ([10, 100], [0.0, 0.9]),
    ],
)
def test_wrong_factor_tables_are_errors(areas, factors):
    with pytest.raises(ValueError):
        areal_reduction_factor(50.0, areas, factors)


def test_climatological_reference_is_the_quantile_where_there_is_enough_record():
    rng = np.random.default_rng(0)
    rec = xr.DataArray(
        rng.gamma(0.5, 2.0, (2000, 3, 4)).astype("f4"),
        dims=("time", "y", "x"),
        attrs={"units": "mm"},
    )
    ref = climatological_reference(rec, quantile=0.995)
    assert (
        ref.dims == ("y", "x")
        and ref.dtype == np.float32
        and ref.attrs["units"] == "mm"
    )
    assert np.allclose(ref.values, np.quantile(rec.values, 0.995, axis=0), rtol=1e-5)
    # a ratio of 1 is the wettest half percent of hours: a record value above the reference is rare
    share = float((rec > ref).mean())
    assert share == pytest.approx(0.005, abs=0.001)
    short = climatological_reference(rec.isel(time=slice(0, 10)))
    assert np.isnan(short.values).all()
    gappy = rec.copy()
    gappy[:, 0, 0] = np.nan
    assert np.isnan(float(climatological_reference(gappy)[0, 0])) and np.isfinite(
        float(climatological_reference(gappy)[1, 1])
    )


def test_climatological_reference_arguments():
    rec = xr.DataArray(np.ones((50, 2)), dims=("time", "x"))
    with pytest.raises(ValueError, match="quantile"):
        climatological_reference(rec, quantile=0.2)
    with pytest.raises(ValueError, match="dimension"):
        climatological_reference(rec, dim="hour")


def test_provenance_is_written():
    out = return_period(_field(60.0), PERIODS, TABLE)
    assert "return period" in out.attrs["radarx_method"].lower()
    assert "radarx_version" in out.attrs
