#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the comparison of a radar rain rate with IMERG
========================================================

A radar grid of 1 km pixels, 120 km wide, around 30 N, 90 W and an IMERG grid of 0.1 degrees every
30 minutes in the layout of the V07 files (``time``, ``lon``, ``lat``; the time is the start of the
step; -9999.9 is missing). The rates are chosen so that the matched values follow from the
construction: a uniform rate, scans of different rates that represent known parts of a step, and a
rate that grows linearly with longitude.
"""

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx._gridcf import aeqd_crs, lonlat
from radarx.retrieve import match_imerg, match_statistics

SITE = (30.0, -90.0)
T0 = np.datetime64("2024-05-01T00:00:00", "ns")


def minutes(*m):
    return np.array(
        [T0 + np.timedelta64(int(v * 60), "s") for v in m], dtype="datetime64[ns]"
    )


def radar_grid(rates, times, half=60e3, spacing=1e3, with_lonlat=True):
    x = np.arange(-half, half + 1, spacing)
    y = np.arange(-half, half + 1, spacing)
    crs = aeqd_crs(*SITE).to_cf()
    lon, lat = lonlat(crs, x, y)
    rates = np.asarray(rates, dtype=float)
    data = np.broadcast_to(rates.reshape(-1, 1, 1), (len(times), y.size, x.size)).copy()
    da = xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={"time": times, "x": x, "y": y},
        name="RAIN_RATE",
        attrs={"units": "mm h-1"},
    )
    da = da.assign_coords(crs_wkt=xr.DataArray(0, attrs=crs))
    if with_lonlat:
        da = da.assign_coords(lon=(("y", "x"), lon), lat=(("y", "x"), lat))
    return da, lon, lat


def imerg_grid(value=5.0, times=None, order="v07"):
    lat = np.round(np.arange(28.05, 31.95, 0.1), 2)
    lon = np.round(np.arange(-91.95, -88.05, 0.1), 2)
    times = minutes(0, 30, 60) if times is None else times
    data = np.full((len(times), lon.size, lat.size), value)
    da = xr.DataArray(
        data,
        dims=("time", "lon", "lat"),
        coords={"time": times, "lon": lon, "lat": lat},
        name="precipitation",
        attrs={"units": "mm/hr"},
    )
    if order == "descending":
        da = da.isel(lat=slice(None, None, -1)).transpose("time", "lat", "lon")
    return da


def test_uniform_rate_is_the_same_on_the_imerg_grid():
    radar, _, _ = radar_grid([6.0, 6.0, 6.0], minutes(5, 15, 25))
    out = match_imerg(radar, imerg_grid(7.2, minutes(0, 30)))
    assert set(out.data_vars) == {
        "radar",
        "imerg",
        "radar_coverage",
        "radar_time_coverage",
    }
    assert out.radar.dims == ("time", "lat", "lon")
    # the 120 km square holds the cells fully inside it
    inner = out.radar_coverage.values >= 1.0
    assert inner.sum() > 80
    first = out.isel(time=0)
    np.testing.assert_allclose(
        first.radar.values[first.radar_coverage.values >= 0.99], 6.0
    )
    np.testing.assert_allclose(out.imerg.values, 7.2)
    assert out.radar.attrs["units"] == "mm h-1"
    # only the first step has scans: 5, 15 and 25 minutes cover 0-30 min; the second has none
    assert np.isnan(out.radar.values[1]).all()
    np.testing.assert_allclose(out.radar_time_coverage.values, [1.0, 0.0], atol=1e-9)


def test_scans_are_weighted_by_the_time_they_represent():
    """Scans at 0, 20 min: the first represents 0-10 min, the second 10-30 min."""
    radar, *_ = radar_grid([10.0, 0.0], minutes(0, 20))
    out = match_imerg(radar, imerg_grid(1.0, minutes(0)))
    good = out.radar_coverage.values[0] >= 0.99
    np.testing.assert_allclose(out.radar.values[0][good], (10 * 10 + 0 * 20) / 30.0)
    # equal spacing: each scan represents 10 minutes
    radar, *_ = radar_grid([0.0, 6.0, 12.0], minutes(5, 15, 25))
    out = match_imerg(radar, imerg_grid(1.0, minutes(0)))
    np.testing.assert_allclose(
        out.radar.values[0][out.radar_coverage.values[0] >= 0.99], 6.0
    )


def test_scans_outside_the_step_do_not_count():
    radar, *_ = radar_grid([50.0, 4.0, 4.0, 50.0], minutes(-20, 8, 22, 50))
    out = match_imerg(radar, imerg_grid(1.0, minutes(0)), min_time_coverage=0.5)
    good = out.radar_coverage.values[0] >= 0.99
    assert good.any()
    # the middle scans represent 1-15 and 15-36 min, cut to 1-15 and 15-30 minutes
    np.testing.assert_allclose(out.radar.values[0][good], 4.0)


def test_a_step_without_enough_scans_is_missing():
    radar, *_ = radar_grid(
        [6.0], minutes(2)
    )  # one scan, 2 minutes, represents 0-4 min with the default
    out = match_imerg(radar, imerg_grid(1.0, minutes(0)))
    assert np.isnan(out.radar.values).all()
    radar, *_ = radar_grid([6.0, 6.0], minutes(5, 14))
    out = match_imerg(radar, imerg_grid(1.0, minutes(0)), min_time_coverage=0.0)
    assert np.isfinite(out.radar.values).any()


def test_time_label():
    radar, *_ = radar_grid([6.0, 6.0, 6.0], minutes(35, 45, 55))
    imerg = imerg_grid(1.0, minutes(30, 60))
    # the label is the start: the scans fall in the step labelled 30 min
    out = match_imerg(radar, imerg)
    assert (
        np.isfinite(out.radar.values[0]).any() and np.isnan(out.radar.values[1]).all()
    )
    # the label is the end: the scans fall in the step labelled 60 min
    out = match_imerg(radar, imerg, time_label="end")
    assert (
        np.isnan(out.radar.values[0]).all() and np.isfinite(out.radar.values[1]).any()
    )
    # the label is the centre: 30 -> 15-45 min, 60 -> 45-75 min; one scan in the first
    out = match_imerg(radar, imerg, time_label="center", min_time_coverage=0.0)
    assert np.isfinite(out.radar.values[0]).any()


def test_rate_that_grows_with_longitude_is_averaged_over_the_cell():
    radar, lon, lat = radar_grid([0.0, 0.0, 0.0], minutes(5, 15, 25))
    radar = (
        radar + (xr.DataArray(lon, dims=("y", "x")) + 91.0) * 10.0
    )  # mm/h per degree
    out = match_imerg(radar, imerg_grid(1.0, minutes(0)))
    good = out.radar_coverage.values[0] >= 0.99
    lon2d = np.broadcast_to(out.lon.values[None, :], good.shape)
    np.testing.assert_allclose(
        out.radar.values[0][good], (lon2d[good] + 91.0) * 10.0, atol=0.15
    )


def test_cells_that_the_radar_does_not_fill_are_masked():
    radar, *_ = radar_grid([6.0, 6.0, 6.0], minutes(5, 15, 25))
    loose = match_imerg(radar, imerg_grid(1.0, minutes(0)), min_coverage=0.05)
    strict = match_imerg(radar, imerg_grid(1.0, minutes(0)), min_coverage=1.0)
    default = match_imerg(radar, imerg_grid(1.0, minutes(0)))
    n = lambda o: int(np.isfinite(o.radar.values).sum())  # noqa: E731
    assert n(loose) > n(default) > n(strict) > 0
    partial = (default.radar_coverage.values > 0) & (default.radar_coverage.values < 1)
    assert partial.any()
    # coverage is the fraction of the pixels that fit into the cell
    assert default.radar_coverage.values.max() == 1.0
    np.testing.assert_array_equal(
        np.isfinite(default.radar.values), default.radar_coverage.values >= 0.5
    )


def test_missing_pixels_are_outside_the_coverage():
    radar, *_ = radar_grid([6.0, 6.0, 6.0], minutes(5, 15, 25))
    radar = radar.where(radar["x"] < 0)
    out = match_imerg(radar, imerg_grid(1.0, minutes(0)))
    covered = out.radar_coverage.values[0] > 0
    lon = np.broadcast_to(out.lon.values[None, :], covered.shape)
    # the radar sees the west half only (the pixels east of the radar are NaN)
    assert lon[covered].max() < -89.9
    np.testing.assert_allclose(
        out.radar.values[0][np.isfinite(out.radar.values[0])], 6.0
    )


def test_imerg_fill_value_and_layouts():
    radar, *_ = radar_grid([6.0, 6.0, 6.0], minutes(5, 15, 25))
    imerg = imerg_grid(2.0, minutes(0))
    imerg = imerg.where(~((imerg.lon > -90.3) & (imerg.lon < -89.9)), -9999.9)
    out = match_imerg(radar, imerg)
    assert np.isnan(out.imerg.values).any() and np.nanmin(out.imerg.values) == 2.0
    # latitude descending and (time, lat, lon): the result is the same, in ascending order
    other = match_imerg(radar, imerg_grid(2.0, minutes(0), order="descending"))
    np.testing.assert_array_equal(np.diff(other.lat.values) > 0, True)
    ref = match_imerg(radar, imerg_grid(2.0, minutes(0)))
    np.testing.assert_allclose(other.radar.values, ref.radar.values, equal_nan=True)
    assert other.radar.dims == ("time", "lat", "lon")


def test_accumulation_output():
    radar, *_ = radar_grid([6.0, 6.0, 6.0], minutes(5, 15, 25))
    out = match_imerg(radar, imerg_grid(4.0, minutes(0)), output="accumulation")
    good = out.radar_coverage.values[0] >= 0.99
    np.testing.assert_allclose(out.radar.values[0][good], 3.0)
    np.testing.assert_allclose(out.imerg.values, 2.0)
    assert out.radar.attrs["units"] == "mm"
    hourly = match_imerg(
        radar, imerg_grid(4.0, minutes(0)), interval=np.timedelta64(1800, "s")
    )
    assert hourly.attrs["interval_seconds"] == 1800.0
    for spelling in (1800, 1800.0, "1800s", "0.5h", "30 min"):
        same = match_imerg(radar, imerg_grid(4.0, minutes(0)), interval=spelling)
        assert same.attrs["interval_seconds"] == 1800.0


def test_without_time_dimensions():
    radar, *_ = radar_grid([6.0], minutes(5))
    snapshot = radar.isel(time=0, drop=True)
    imerg = imerg_grid(3.0, minutes(0)).isel(time=0, drop=True)
    out = match_imerg(snapshot, imerg)
    assert out.radar.dims == ("lat", "lon") and out.radar_time_coverage.ndim == 0
    good = out.radar_coverage.values >= 0.99
    np.testing.assert_allclose(out.radar.values[good], 6.0)
    one = match_imerg(snapshot, imerg_grid(3.0, minutes(0)))
    assert one.radar.dims == ("time", "lat", "lon")
    with pytest.raises(ValueError, match="no time dimension"):
        match_imerg(snapshot, imerg_grid(3.0, minutes(0, 30)))
    stack = match_imerg(radar, imerg, min_time_coverage=0.0)
    assert stack.radar.dims == ("lat", "lon")


def test_radar_georeferencing_alternatives():
    radar, lon, lat = radar_grid([6.0, 6.0, 6.0], minutes(5, 15, 25), with_lonlat=False)
    from_crs = match_imerg(radar, imerg_grid(1.0, minutes(0)))
    with_coords, *_ = radar_grid([6.0, 6.0, 6.0], minutes(5, 15, 25))
    direct = match_imerg(with_coords, imerg_grid(1.0, minutes(0)))
    np.testing.assert_allclose(
        from_crs.radar.values, direct.radar.values, equal_nan=True
    )
    # a plain longitude/latitude grid (1-D axes)
    lat1 = np.arange(29.5, 30.5, 0.01)
    lon1 = np.arange(-90.5, -89.5, 0.01)
    grid = xr.DataArray(
        np.full((1, lat1.size, lon1.size), 3.0),
        dims=("time", "latitude", "longitude"),
        coords={"time": minutes(15), "latitude": lat1, "longitude": lon1},
    )
    out = match_imerg(grid, imerg_grid(1.0, minutes(0)), min_time_coverage=0.0)
    good = out.radar_coverage.values >= 0.99
    assert good.sum() >= 50
    np.testing.assert_allclose(out.radar.values[good], 3.0)


def test_statistics_of_the_matched_fields():
    radar, *_ = radar_grid([6.0, 6.0, 6.0], minutes(5, 15, 25))
    out = match_imerg(radar, imerg_grid(4.0, minutes(0)))
    s = match_statistics(out.radar, out.imerg, threshold=1.0)
    assert float(s.bias) == pytest.approx(-2.0)
    assert float(s.ratio) == pytest.approx(4.0 / 6.0)
    assert float(s.pod) == 1.0 and float(s.far) == 0.0
    assert int(s["count"]) == int(np.isfinite(out.radar.values).sum())
    assert "10.1007/978-3-030-24568-9_19" in out.attrs["radarx_references"]


def test_accessor():
    radar, *_ = radar_grid([6.0, 6.0, 6.0], minutes(5, 15, 25))
    imerg = imerg_grid(2.0, minutes(0))
    xr.testing.assert_identical(
        radar.radarx.match_imerg(imerg).radar, match_imerg(radar, imerg).radar
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"output": "total"}, "output"),
        ({"time_label": "middle"}, "time_label"),
        ({"min_coverage": 1.5}, "min_coverage"),
        ({"min_time_coverage": -0.1}, "min_time_coverage"),
        ({"interval": "0min"}, "interval"),
        ({"interval": "half an hour"}, "interval"),
    ],
)
def test_invalid_options(kwargs, message):
    radar, *_ = radar_grid([6.0], minutes(5))
    with pytest.raises(ValueError, match=message):
        match_imerg(radar, imerg_grid(1.0, minutes(0)), **kwargs)


def test_invalid_inputs():
    radar, *_ = radar_grid([6.0], minutes(5))
    imerg = imerg_grid(1.0, minutes(0))
    with pytest.raises(TypeError):
        match_imerg(radar.values, imerg)
    with pytest.raises(TypeError):
        match_imerg(radar, imerg.to_dataset())
    with pytest.raises(ValueError, match="dimension"):
        match_imerg(radar, imerg.rename(lat="north"))
    with pytest.raises(ValueError, match="at least two"):
        match_imerg(radar, imerg.isel(lat=[0]))
    irregular = imerg.assign_coords(
        lon=imerg.lon.values + 0.002 * np.arange(imerg.sizes["lon"]) ** 2
    )
    with pytest.raises(ValueError, match="regular"):
        match_imerg(radar, irregular)
    far = imerg.assign_coords(lon=imerg.lon.values + 40.0)
    with pytest.raises(ValueError, match="no radar pixel"):
        match_imerg(radar, far)
    bare = xr.DataArray(np.ones((3, 3)), dims=("a", "b"))
    with pytest.raises(ValueError, match="lon"):
        match_imerg(bare, imerg)
    no_crs = radar.drop_vars(["lon", "lat", "crs_wkt"])
    with pytest.raises(ValueError, match="crs_wkt"):
        match_imerg(no_crs, imerg)
    no_crs_attrs = radar.drop_vars(["lon", "lat"]).assign_coords(
        crs_wkt=xr.DataArray(0)
    )
    with pytest.raises(ValueError, match="crs_wkt"):
        match_imerg(no_crs_attrs, imerg)
