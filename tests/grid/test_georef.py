#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the CF georeferencing of grids (radarx._gridcf)
=========================================================
"""
import numpy as np
import pyproj
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx import _gridcf
from radarx import retrieve as R
from radarx.grid import cone, multi
from tests.grid import test_cone, test_multi
from tests.test_cappi import _make_ppi_sweep

LAT0, LON0 = 45.0, 10.0
GEOD = pyproj.Geod(ellps="WGS84")


def _truth(x, y, lat0, lon0):
    """Geographic position of AEQD points from the geodesic (independent of the CRS)."""
    X, Y = np.meshgrid(x, y)
    az = np.degrees(np.arctan2(X, Y))
    s = np.hypot(X, Y)
    lon, lat, _ = GEOD.fwd(np.full(X.shape, lon0), np.full(X.shape, lat0), az, s)
    return lat, lon


def _error_m(grid, lat0, lon0):
    """Largest distance (m) between lat and lon of the grid and the geodesic."""
    lat, lon = _truth(grid["x"].values, grid["y"].values, lat0, lon0)
    _, _, dist = GEOD.inv(lon, lat, grid["lon"].values, grid["lat"].values)
    return float(np.max(dist))


def _check_cf(out, fields, name="crs_wkt"):
    for field in fields:
        assert out[field].attrs["grid_mapping"] == name
    assert name in out.coords
    assert out[name].attrs["grid_mapping_name"] == "azimuthal_equidistant"
    assert out["lat"].dims == out["lon"].dims == ("y", "x")
    assert out["lat"].attrs["standard_name"] == "latitude"
    assert out["lon"].attrs["units"] == "degrees_east"
    assert out["x"].attrs["standard_name"] == "projection_x_coordinate"
    assert out["y"].attrs["standard_name"] == "projection_y_coordinate"
    assert out["x"].attrs["axis"] == "X" and out["y"].attrs["axis"] == "Y"
    assert out["x"].attrs["units"] == "m"
    assert out["z"].attrs["axis"] == "Z" and out["z"].attrs["positive"] == "up"
    assert out["z"].attrs["units"] == "m"
    assert out["time"].attrs["standard_name"] == "time"


@pytest.fixture(scope="module")
def cones():
    return cone.grid_cones(test_cone._volume(field="random"), "DBZH", **test_cone.GRID)


def test_grid_cones_is_georeferenced(cones):
    _check_cf(cones, ["DBZH"])
    assert cones["z"].attrs["standard_name"] == "altitude"
    # the coordinates the old output had are still there
    for name in ("latitude", "longitude", "altitude"):
        assert name in cones.coords
    assert cones["latitude"].attrs["units"] == "degrees_north"


def test_lat_lon_agree_with_the_geodesic_to_one_centimetre(cones):
    assert _error_m(cones, LAT0, LON0) < 0.01
    crs = pyproj.CRS.from_cf(cones["crs_wkt"].attrs)
    inverse = pyproj.Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)
    lon, lat = inverse.transform(*np.meshgrid(cones.x, cones.y))
    np.testing.assert_allclose(cones["lat"].values, lat, atol=1e-12)
    np.testing.assert_allclose(cones["lon"].values, lon, atol=1e-12)


def test_one_dimensional_axes_were_wrong_away_from_the_origin():
    """The numbers behind issue #265: 1-D lat(y) and lon(x) are off in a corner."""
    for half, minimum in ((60e3, 100.0), (230e3, 3000.0)):
        axis = np.array([-half, 0.0, half])
        lon1, lat1 = cone._lonlat_axes(axis, axis, LAT0, LON0)
        wrong = np.meshgrid(lon1, lat1)
        lat, lon = _truth(axis, axis, LAT0, LON0)
        _, _, dist = GEOD.inv(
            lon[-1, -1], lat[-1, -1], wrong[0][-1, -1], wrong[1][-1, -1]
        )
        assert dist > minimum


def test_cf_xarray_and_rioxarray_find_the_crs(cones):
    cf_xarray = pytest.importorskip("cf_xarray")
    assert cf_xarray is not None
    assert cones.cf.grid_mapping_names == {"azimuthal_equidistant": ["crs_wkt"]}
    assert "latitude" in cones.cf.coordinates and "longitude" in cones.cf.coordinates
    assert set(cones.cf.axes) >= {"X", "Y", "Z"}
    rioxarray = pytest.importorskip("rioxarray")
    assert rioxarray is not None
    assert cones["DBZH"].rio.crs.to_dict()["proj"] == "aeqd"


def test_crs_variable_and_user_crs():
    attrs = _gridcf.crs_variable(LAT0, LON0).attrs
    assert attrs["latitude_of_projection_origin"] == LAT0
    assert attrs["longitude_of_projection_origin"] == LON0
    assert _gridcf.user_crs((LAT0, LON0)).equals(_gridcf.aeqd_crs(LAT0, LON0))
    assert _gridcf.user_crs(32643).to_epsg() == 32643


def test_georeference_keeps_attributes_and_strips_without_a_crs():
    ds = xr.Dataset(
        {"f": (("y", "x"), np.ones((2, 2)), {"grid_mapping": "gone"}), "g": ("t", [1])},
        coords={
            "x": ("x", [0.0, 1.0], {"standard_name": "mine"}),
            "y": [0.0, 1.0],
            "z": ("x", [1.0, 2.0]),
        },
    )
    out = _gridcf.georeference(ds)
    assert "grid_mapping" not in out["f"].attrs and "lat" not in out.coords
    assert out["x"].attrs["standard_name"] == "mine"
    assert out["x"].attrs["axis"] == "X"
    assert "standard_name" not in out["z"].attrs
    # the CRS is only given by the caller; lat and lon can be left out
    mapping = _gridcf.aeqd_crs(0.0, 0.0).to_cf()
    bare = _gridcf.georeference(
        xr.Dataset(
            {"f": (("y", "x"), np.ones((2, 2)))},
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        ),
        mapping,
        name="crs",
        lonlat_=False,
    )
    assert "lat" not in bare.coords and bare["f"].attrs["grid_mapping"] == "crs"


def test_existing_two_dimensional_lat_lon_are_kept_unless_recomputed():
    base = xr.Dataset(
        {"f": (("y", "x"), np.ones((2, 3)))},
        coords={
            "x": [0.0, 1e3, 2e3],
            "y": [0.0, 1e3],
            "lat": (("y", "x"), np.full((2, 3), 5.0)),
            "lon": (("y", "x"), np.full((2, 3), 6.0)),
        },
    )
    mapping = _gridcf.aeqd_crs(10.0, 20.0).to_cf()
    kept = _gridcf.georeference(base.copy(), mapping)
    assert (kept["lat"] == 5.0).all()
    again = _gridcf.georeference(base.copy(), mapping, recompute=True)
    assert abs(float(again["lat"][0, 0]) - 10.0) < 1e-9


def test_input_is_not_changed():
    dtree = test_cone._volume(field="random")
    before = {
        name: dict(dtree[name].to_dataset()["DBZH"].attrs)
        for name in ("sweep_0", "sweep_1")
    }
    root_before = {k: dict(v.attrs) for k, v in dtree.to_dataset().coords.items()}
    cone.grid_cones(dtree, "DBZH", **test_cone.GRID)
    for name, attrs in before.items():
        assert dict(dtree[name].to_dataset()["DBZH"].attrs) == attrs
    assert {
        k: dict(v.attrs) for k, v in dtree.to_dataset().coords.items()
    } == root_before


def test_exporter_recomputes_lat_lon_for_a_given_crs(cones, tmp_path):
    out = radarx.io.write_netcdf_cf(cones, tmp_path / "e.nc", crs=(LAT0, LON0 + 1.0))
    with xr.open_dataset(out) as back:
        assert abs(float(back["lon"][20, 20]) - (LON0 + 1.0)) < 1e-6
    own = radarx.io.write_netcdf_cf(cones, tmp_path / "own.nc")
    with xr.open_dataset(own) as back:
        np.testing.assert_allclose(back["lon"].values, cones["lon"].values, atol=1e-12)


# ---------------------------------------------------------------------------
# multi-radar grid and CAPPI
# ---------------------------------------------------------------------------
def test_grid_radars_is_georeferenced():
    lat, lon = test_multi._site(30e3, 10e3)
    network = [test_multi._volume(*test_multi.ORIGIN), test_multi._volume(lat, lon)]
    out = multi.grid_radars(
        network, **test_multi.GRID, origin=test_multi.ORIGIN, data_vars="DBZH"
    )
    _check_cf(out, ["DBZH", "coverage", "azimuth", "elevation", "range"])
    assert out["z"].attrs["standard_name"] == "altitude"
    assert _error_m(out, *test_multi.ORIGIN) < 0.01
    assert "grid_mapping" not in out["beamwidth"].attrs


def test_create_cappi_has_a_crs_and_two_dimensional_lat_lon():
    vol = xr.DataTree.from_dict(
        {
            f"sweep_{k}": _make_ppi_sweep(el, "2024-01-01T00:00:00")
            for k, el in enumerate((1.0, 3.0, 5.0))
        }
    )
    out = R.create_cappi(vol, height=150.0, fields=["DBZH"], x_res=2000.0, y_res=2000.0)
    assert out["DBZH"].attrs["grid_mapping"] == "crs_wkt"
    assert out["lat"].dims == ("y", "x")
    lat0, lon0 = float(out["latitude"]), float(out["longitude"])
    assert _error_m(out, lat0, lon0) < 0.01
    assert (
        out["x"].attrs["axis"] == "X" and out["time"].attrs["standard_name"] == "time"
    )
    assert out["z"].attrs["axis"] == "Z" and out["z"].ndim == 0
    # without the radar position there is nothing to base a CRS on
    stripped = xr.DataTree.from_dict(
        {
            name: vol[name]
            .to_dataset()
            .drop_vars(["latitude", "longitude"], errors="ignore")
            for name in ("sweep_0", "sweep_1", "sweep_2")
        }
    )
    plain = R.create_cappi(
        stripped, height=150.0, fields=["DBZH"], x_res=2000.0, y_res=2000.0
    )
    assert "crs_wkt" not in plain.coords and "lat" not in plain.coords
    assert "grid_mapping" not in plain["DBZH"].attrs
