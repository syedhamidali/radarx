#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for super-observations and the WRFDA ob.radar writer
==========================================================

The oracle for the binning is written here from the definitions (beam height
and ground distance from the 4/3 Earth formulas of ``radarx.fundamentals``,
the great-circle destination, a pandas group-by), not from the module. The
compiled kernel and the NumPy implementation must agree to rounding. The
WRFDA file is read back with a fixed-width reader written in this file.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.fundamentals.constants import EARTH_RADIUS, EFFECTIVE_RADIUS_4_3
from radarx.fundamentals.geometry import beam_center_height, ground_range
from radarx.io import write_wrfda_radar
from radarx.retrieve import superobs

mod = __import__("radarx.retrieve.superobs", fromlist=["x"])

ENGINES = ["numpy"] + (["compiled"] if mod.HAS_COMPILED_KERNEL else [])
compiled_only = pytest.mark.skipif(
    not mod.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)
LAT0, LON0, ALT0 = 34.5, -88.0, 150.0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def destination(lat0, lon0, az, dist):
    """Great-circle destination (degrees) from the standard formula."""
    la0, lo0, a, d = (
        np.radians(lat0),
        np.radians(lon0),
        np.radians(az),
        dist / EARTH_RADIUS,
    )
    la = np.arcsin(np.sin(la0) * np.cos(d) + np.cos(la0) * np.sin(d) * np.cos(a))
    lo = lo0 + np.arctan2(
        np.sin(a) * np.sin(d) * np.cos(la0), np.cos(d) - np.sin(la0) * np.sin(la)
    )
    return np.degrees(la), (np.degrees(lo) + 540.0) % 360.0 - 180.0


def sweep(
    elevation=1.0,
    nray=90,
    ngate=160,
    seed=0,
    lat=LAT0,
    lon=LON0,
    alt=ALT0,
    echo=0.6,
    velocity=True,
    rho=False,
    first=1000.0,
    step=500.0,
):
    """A synthetic sweep with random reflectivity and velocity, NaN where no echo."""
    rng = np.random.default_rng(seed)
    az = (np.arange(nray) + 0.5) * 360.0 / nray
    gates = first + step * np.arange(ngate)
    dbz = rng.uniform(-10.0, 55.0, (nray, ngate))
    dbz[rng.uniform(size=dbz.shape) > echo] = np.nan
    data = {"DBZH": (("azimuth", "range"), dbz)}
    if velocity:
        vr = rng.uniform(-30.0, 30.0, (nray, ngate))
        vr[np.isnan(dbz)] = np.nan
        vr[rng.uniform(size=vr.shape) > 0.9] = 45.0  # beyond max_velocity
        data["VRADH"] = (("azimuth", "range"), vr)
    if rho:
        r = rng.uniform(0.6, 1.0, (nray, ngate))
        r[rng.uniform(size=r.shape) > 0.8] = np.nan
        data["RHOHV"] = (("azimuth", "range"), r)
    return xr.Dataset(
        data,
        coords={
            "azimuth": az,
            "elevation": ("azimuth", np.full(nray, float(elevation))),
            "range": gates,
            "time": (
                "azimuth",
                np.datetime64("2022-03-31T00:00:00", "ns")
                + (np.arange(nray) * 40e9 / nray).astype("timedelta64[ns]"),
            ),
            "latitude": lat,
            "longitude": lon,
            "altitude": alt,
        },
    )


def volume(elevations=(0.5, 1.5, 3.0), name="TEST", **kwargs):
    tree = xr.DataTree(xr.Dataset(attrs={"instrument_name": name}), name="root")
    for k, el in enumerate(elevations):
        tree[f"sweep_{k}"] = xr.DataTree(sweep(elevation=el, seed=k, **kwargs))
    return tree


def model_grid(n=60, dx=3000.0, lat0=LAT0, lon0=LON0):
    """Regular grid of cell centres about (lat0, lon0) on the plane, as 2-D lat, lon."""
    x = (np.arange(n) - (n - 1) / 2.0) * dx
    xx, yy = np.meshgrid(x, x)
    dist = np.hypot(xx, yy)
    az = np.degrees(np.arctan2(xx, yy))
    return destination(lat0, lon0, az, dist)


def oracle_boxes(ds, box, edges, max_range=160e3, min_range=3000.0, clear=False):
    """Mean fields per (box, layer) of the gates of a sweep, from the definitions."""
    az = ds.azimuth.values[:, None] * np.ones((1, ds.sizes["range"]))
    el = ds.elevation.values[:, None] * np.ones((1, ds.sizes["range"]))
    r = ds.range.values[None, :] * np.ones((ds.sizes["azimuth"], 1))
    z = ds.DBZH.values
    v = ds.VRADH.values
    v = np.where(np.abs(v) > 40, np.nan, v)
    h = beam_center_height(r, el, float(ds.altitude))
    s = ground_range(r, el)
    ok = (r <= max_range) & (s > min_range) & (h >= edges[0]) & (h < edges[-1])
    ok &= clear | np.isfinite(z) | np.isfinite(v)
    x, y = s * np.sin(np.radians(az)), s * np.cos(np.radians(az))
    df = pd.DataFrame(
        {
            "ix": np.floor(x[ok] / box).astype(int),
            "iy": np.floor(y[ok] / box).astype(int),
            "lev": np.searchsorted(edges, h[ok], side="right") - 1,
            "z": np.where(np.isfinite(z[ok]), 10 ** (z[ok] / 10.0), np.nan),
            "d": z[ok],
            "v": v[ok],
            "h": h[ok],
            "r": r[ok],
        }
    )
    g = df.groupby(["iy", "ix", "lev"])
    out = g.agg(
        n=("h", "size"),
        nz=("z", "count"),
        z=("z", "mean"),
        d_sd=("d", "std"),
        nv=("v", "count"),
        v=("v", "mean"),
        v_sd=("v", "std"),
        h=("h", "mean"),
        r=("r", "mean"),
    )
    out["dbz"] = 10 * np.log10(out.z)
    return out


def by_key(so):
    flat = so.drop_dims(["radar", "level_edge"])
    return (
        flat.to_dataframe()
        .reset_index()
        .set_index(["column_row", "column_col", "level"])
    )


# --------------------------------------------------------------------------
# binning against the oracle
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_box_means_match_the_oracle(engine):
    ds = sweep()
    edges = np.arange(0, 12001, 500.0)
    so = superobs(
        ds, 5.0, engine=engine, min_gates=1, min_valid=1, min_dbz=-99.0, missing="gap"
    )
    want = oracle_boxes(ds, 5000.0, edges)
    got = by_key(so)
    assert len(got) == len(want)
    got = got.loc[want.index]
    np.testing.assert_array_equal(got.gate_count, want.n)
    np.testing.assert_array_equal(got.reflectivity_count, want.nz)
    np.testing.assert_array_equal(got.velocity_count, want.nv)
    np.testing.assert_allclose(got.reflectivity, want.dbz, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(got.velocity, want.v, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(
        got.reflectivity_spread, want.d_sd, rtol=1e-8, atol=1e-8, equal_nan=True
    )
    np.testing.assert_allclose(
        got.velocity_spread, want.v_sd, rtol=1e-8, atol=1e-8, equal_nan=True
    )
    np.testing.assert_allclose(got.height, want.h, rtol=1e-12)
    np.testing.assert_allclose(got.range, want.r, rtol=1e-12)


@pytest.mark.parametrize("engine", ENGINES)
def test_reflectivity_is_averaged_in_linear_z(engine):
    ds = sweep(nray=1, ngate=20, echo=1.0, velocity=False)
    ds["DBZH"].values[:] = 0.0
    ds["DBZH"].values[0, :5] = 30.0
    so = superobs(
        ds,
        5000.0,
        layers=[0.0, 20000.0],
        engine=engine,
        min_gates=1,
        min_valid=1,
        min_dbz=-99.0,
        min_range=0.0,
        missing="gap",
    )
    assert so.sizes["obs"] == 1
    want = 10 * np.log10((5 * 1000.0 + 15 * 1.0) / 20.0)
    assert float(so.reflectivity[0]) == pytest.approx(want, rel=1e-12)
    assert float(so.reflectivity[0]) > ds.DBZH.values.mean() + 10  # not the dBZ mean


@pytest.mark.parametrize("engine", ENGINES)
def test_layers_and_range_limits(engine):
    ds = sweep(elevation=2.0)
    edges = np.array([0.0, 2000.0, 4000.0, 8000.0])
    so = superobs(
        ds, 20.0, layers=edges, engine=engine, min_gates=1, min_valid=1, missing="gap"
    )
    assert set(np.unique(so.level)) <= {0, 1, 2}
    assert so.height.min() >= 0 and so.height.max() < 8000.0
    np.testing.assert_array_equal(so.level_edges, edges)
    near = superobs(
        ds,
        20.0,
        engine=engine,
        min_gates=1,
        min_valid=1,
        max_range=40e3,
        missing="gap",
    )
    assert near.range.max() <= 40e3
    far = superobs(
        ds, 20.0, engine=engine, min_gates=1, min_valid=1, min_range=30e3, missing="gap"
    )
    assert (ground_range(far.range.values, 2.0) > 30e3 - 1.0).all()


@pytest.mark.parametrize("engine", ENGINES)
def test_thresholds(engine):
    ds = sweep()
    kw = dict(engine=engine, missing="gap")
    few = superobs(ds, 5.0, min_gates=1, min_valid=1, **kw)
    many = superobs(ds, 5.0, min_gates=10, min_valid=1, **kw)
    assert many.sizes["obs"] < few.sizes["obs"]
    assert (many.gate_count >= 10).all()
    strict = superobs(ds, 5.0, min_gates=1, min_valid=4, **kw)
    assert (strict.reflectivity_count[strict.reflectivity.notnull()] >= 4).all()
    assert (strict.velocity_count[strict.velocity.notnull()] >= 4).all()
    weak = superobs(ds, 5.0, min_gates=1, min_valid=1, min_dbz=30.0, **kw)
    assert (weak.reflectivity.dropna("obs") >= 30.0).all()
    # velocities beyond max_velocity are left out
    assert np.abs(few.velocity).max() <= 40.0
    low = superobs(ds, 5.0, min_gates=1, min_valid=1, max_velocity=10.0, **kw)
    assert np.abs(low.velocity).max() <= 10.0


@pytest.mark.parametrize("engine", ENGINES)
def test_correlation_coefficient_removes_gates(engine):
    ds = sweep(rho=True)
    kw = dict(engine=engine, min_gates=1, min_valid=1, missing="gap")
    plain = superobs(ds, 5.0, **kw)
    cut = superobs(ds, 5.0, rho_min=0.85, **kw)
    assert cut.gate_count.sum() < plain.gate_count.sum()
    manual = ds.copy(deep=True)
    bad = (manual.RHOHV < 0.85).values
    for name in ("DBZH", "VRADH"):
        manual[name].values[bad] = np.nan
    ref = superobs(manual, 5.0, **kw)
    xr.testing.assert_allclose(
        cut.drop_vars("radar_time"), ref.drop_vars("radar_time"), rtol=1e-12
    )


# --------------------------------------------------------------------------
# gate geometry and the radial unit vector
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_positions_and_unit_vector(engine):
    """One gate per box: position and ray direction from independent formulas."""
    az, el = 70.0, 2.0
    ds = sweep(elevation=el, nray=1, ngate=6, echo=1.0, first=60e3, step=20e3)
    ds = ds.assign_coords(azimuth=("azimuth", [az]), elevation=("azimuth", [el]))
    ds["DBZH"][:] = 20.0
    ds["VRADH"][:] = 5.0
    so = superobs(
        ds,
        1.0,
        layers=[0.0, 20000.0],
        engine=engine,
        min_gates=1,
        min_valid=1,
        min_dbz=0.0,
        missing="gap",
    )
    r = ds.range.values
    h = beam_center_height(r, el, ALT0)
    s = ground_range(r, el)
    lat, lon = destination(LAT0, LON0, az, s)
    order = np.argsort(so.range.values)
    np.testing.assert_allclose(so.range.values[order], r, rtol=1e-12)
    np.testing.assert_allclose(so.height.values[order], h, rtol=1e-12)
    np.testing.assert_allclose(so.latitude.values[order], lat, atol=1e-9)
    np.testing.assert_allclose(so.longitude.values[order], lon, atol=1e-9)

    # direction of the ray at the gate from vectors: a straight ray in the 4/3 Earth
    reff = EFFECTIVE_RADIUS_4_3
    for k, gate in enumerate(order):
        # plane of the ray: x along the ground at the radar, y up; radar at (0, reff)
        direction = np.array([np.cos(np.radians(el)), np.sin(np.radians(el))])
        point = np.array([0.0, reff]) + r[k] * direction
        up = point / np.hypot(*point)
        elev_local = np.arcsin(direction @ up)
        # horizontal direction on the sphere: tangent of the great circle (true Earth)
        p0 = _unit(LAT0, LON0)
        p1 = _unit(lat[k], lon[k])
        tangent = np.cross(np.cross(p0, p1), p1)
        tangent /= np.linalg.norm(tangent)
        east = np.cross([0, 0, 1.0], p1)
        east /= np.linalg.norm(east)
        north = np.cross(p1, east)
        want = (
            np.cos(elev_local) * (tangent @ east),
            np.cos(elev_local) * (tangent @ north),
            np.sin(elev_local),
        )
        got = (
            float(so.radial_east[gate]),
            float(so.radial_north[gate]),
            float(so.radial_up[gate]),
        )
        np.testing.assert_allclose(got, want, atol=1e-9)
        assert np.linalg.norm(got) == pytest.approx(1.0, abs=1e-12)


def _unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


# --------------------------------------------------------------------------
# model grid
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_grid_columns_are_blocks_of_nearest_cells(engine):
    ds = sweep(elevation=1.0, ngate=120)
    lat, lon = model_grid(n=60, dx=3000.0)
    so = superobs(
        ds,
        (lat, lon),
        cells=3,
        engine=engine,
        min_gates=1,
        min_valid=1,
        missing="gap",
        domain_tolerance=1.0,
    )
    assert so.sizes["obs"] > 100
    # every box position (mean of the gates) lies in the block of its column,
    # within one cell of the block edge
    nbx = -(-60 // 3)
    assert (so.column == so.column_row * nbx + so.column_col).all()
    assert so.column_row.max() < 20 and so.column_col.max() < 20
    # brute-force nearest cell of the gates of the box with the most gates
    full = so.isel(obs=int(so.gate_count.values.argmax()))
    la, lo = float(full.latitude), float(full.longitude)
    x, y = mod._aeqd(
        np.radians(lat),
        np.radians(lon),
        np.radians(LAT0),
        np.radians(LON0),
        EARTH_RADIUS,
    )
    qx, qy = mod._aeqd(
        np.radians(la), np.radians(lo), np.radians(LAT0), np.radians(LON0), EARTH_RADIUS
    )
    j, i = np.unravel_index(np.argmin(np.hypot(x - qx, y - qy)), lat.shape)
    assert abs(j // 3 - int(full.column_row)) <= 1
    assert abs(i // 3 - int(full.column_col)) <= 1


@compiled_only
def test_nearest_cell_search_agrees_with_brute_force():
    rng = np.random.default_rng(3)
    lat, lon = model_grid(n=31, dx=2000.0)
    # a rotated, slightly irregular grid
    lat = lat + rng.normal(0, 1e-4, lat.shape)
    lon = lon + rng.normal(0, 1e-4, lon.shape)
    loc = mod._Locator(lat, lon, 1.0)
    index = loc.kernel()
    qx = rng.uniform(loc.x.min() - 1500, loc.x.max() + 1500, 2000)
    qy = rng.uniform(loc.y.min() - 1500, loc.y.max() + 1500, 2000)
    d = np.hypot(loc.x.ravel()[None] - qx[:, None], loc.y.ravel()[None] - qy[:, None])
    best = d.argmin(axis=1)
    within = d.min(axis=1) < loc.max_dist * 0.999
    got = np.array([index.nearest(a, b) for a, b in zip(qx, qy)])
    np.testing.assert_array_equal(got[within], best[within])
    outside = d.min(axis=1) > loc.max_dist * 1.001
    assert (got[outside] == -1).all()
    assert index.shape == (31, 31)


def test_grid_description_forms():
    ds = sweep(ngate=100)
    lat, lon = model_grid(n=40, dx=4000.0)
    kw = dict(min_gates=1, min_valid=1, missing="gap", engine="numpy")
    a = superobs(ds, (lat, lon), **kw)
    wrf = xr.Dataset(
        {
            "XLAT": (("Time", "south_north", "west_east"), lat[None]),
            "XLONG": (("Time", "south_north", "west_east"), lon[None]),
            "HGT": (("Time", "south_north", "west_east"), 100.0 + lat[None] * 0),
        }
    )
    b = superobs(ds, wrf, **kw)
    xr.testing.assert_allclose(b.drop_vars("terrain_height"), a)
    np.testing.assert_allclose(b.terrain_height, 100.0)
    c = superobs(ds, (lat, lon), terrain=np.full(lat.shape, 7.0), **kw)
    np.testing.assert_allclose(c.terrain_height, 7.0)
    # 1-D axes form a rectilinear grid
    axes = xr.Dataset({"latitude": ("y", lat[:, 20]), "longitude": ("x", lon[20, :])})
    d = superobs(ds, axes, **kw)
    assert d.sizes["obs"] > 0
    with pytest.raises(KeyError):
        superobs(ds, xr.Dataset({"a": ("x", [1.0])}), **kw)
    with pytest.raises(TypeError):
        superobs(ds, "grid", **kw)
    with pytest.raises(ValueError):
        superobs(ds, (lat, lon[:5]), **kw)
    with pytest.raises(ValueError):
        superobs(ds, (lat[:1, :1], lon[:1, :1]), **kw)


def test_gates_outside_the_domain_are_left_out():
    ds = sweep(ngate=160)
    lat, lon = model_grid(n=20, dx=3000.0)  # 60 km wide, the sweep reaches 80 km
    so = superobs(ds, (lat, lon), min_gates=1, min_valid=1, missing="gap")
    far = destination(LAT0, LON0, 0.0, 45e3)[0]
    assert so.latitude.max() < far + 0.1
    wide = superobs(
        ds, (lat, lon), min_gates=1, min_valid=1, missing="gap", domain_tolerance=1e3
    )
    assert wide.sizes["obs"] > so.sizes["obs"]


# --------------------------------------------------------------------------
# echo-free boxes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_nonprecipitating_boxes(engine):
    ds = sweep(nray=36, ngate=100, echo=0.0)  # no echo at all
    ds["DBZH"].values[:, 40:] = 30.0  # echo beyond 21 km
    ds["VRADH"].values[:] = np.nan
    kw = dict(engine=engine, min_gates=6, min_valid=3, layers=[0.0, 10000.0])
    clear = superobs(ds, 8.0, missing="clear", **kw)
    gap = superobs(ds, 8.0, missing="gap", **kw)
    assert int(clear.nonprecip.sum()) > 0
    assert int(gap.nonprecip.sum()) == 0
    no_echo = clear.where(clear.nonprecip == 1, drop=True)
    assert no_echo.reflectivity.isnull().all()
    assert (no_echo.gate_count >= 6).all()
    np.testing.assert_allclose(no_echo.reflectivity_error, 5.0)
    assert (no_echo.range < 25e3).all()
    echo = clear.where(clear.nonprecip == 0, drop=True)
    assert echo.reflectivity.notnull().all()
    np.testing.assert_allclose(echo.reflectivity, 30.0)
    # an echo too weak is also no echo
    weak = superobs(ds, 8.0, missing="clear", min_dbz=40.0, **kw)
    assert int(weak.nonprecip.sum()) == weak.sizes["obs"]


@pytest.mark.parametrize("engine", ENGINES)
def test_split_cut_without_reflectivity_does_not_make_clear_air(engine):
    ds = sweep(velocity=True, echo=1.0).drop_vars("DBZH")
    so = superobs(ds, 8.0, engine=engine, missing="clear", min_gates=1, min_valid=1)
    assert int(so.nonprecip.sum()) == 0
    assert so.reflectivity.isnull().all()
    assert so.velocity.notnull().any()


# --------------------------------------------------------------------------
# error estimate
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_error_model(engine):
    ds = sweep(echo=1.0)
    kw = dict(engine=engine, min_gates=3, min_valid=3, missing="gap")
    so = superobs(ds, 10.0, error_floor=(1.5, 4.0), **kw)
    ok = so.velocity.notnull()
    want_v = np.sqrt(1.5**2 + so.velocity_spread**2 / so.velocity_count)
    np.testing.assert_allclose(so.velocity_error[ok], want_v[ok], rtol=1e-12)
    ok = so.reflectivity.notnull()
    want_z = np.sqrt(4.0**2 + so.reflectivity_spread**2 / so.reflectivity_count)
    np.testing.assert_allclose(so.reflectivity_error[ok], want_z[ok], rtol=1e-12)
    assert (so.velocity_error[so.velocity.notnull()] >= 1.5).all()
    # a box with more spread has a larger error at equal count
    assert so.velocity_error.max() > 1.5


# --------------------------------------------------------------------------
# engines, threads, inputs
# --------------------------------------------------------------------------


@compiled_only
@pytest.mark.parametrize("mode", ["box", "grid"])
@pytest.mark.parametrize("missing", ["clear", "gap"])
def test_kernel_and_numpy_agree(mode, missing):
    tree = volume(rho=True)
    target = 6.0 if mode == "box" else model_grid(n=70, dx=3000.0)
    kw = dict(missing=missing, rho_min=0.8, min_gates=3, min_valid=2)
    a = superobs(tree, target, engine="compiled", **kw)
    b = superobs(tree, target, engine="numpy", **kw)
    assert a.sizes["obs"] > 50
    for name in a.variables:
        if np.issubdtype(a[name].dtype, np.floating):
            np.testing.assert_allclose(
                a[name].values, b[name].values, rtol=1e-11, atol=1e-11, equal_nan=True
            )
        elif np.issubdtype(a[name].dtype, np.datetime64):
            continue
        else:
            np.testing.assert_array_equal(a[name].values, b[name].values)


@compiled_only
def test_threads_do_not_change_the_result():
    tree = volume()
    one = superobs(tree, 5.0, n_threads=1)
    many = superobs(tree, 5.0, n_threads=7)
    xr.testing.assert_identical(one, many)


def test_engine_option():
    ds = sweep()
    with pytest.raises(ValueError, match="engine"):
        superobs(ds, 5.0, engine="gpu")
    if not mod.HAS_COMPILED_KERNEL:
        with pytest.raises(ImportError):
            superobs(ds, 5.0, engine="compiled")


def test_volume_sweeps_and_several_radars():
    tree = volume(elevations=(0.5, 2.0))
    single = superobs(tree, 8.0)
    assert list(single.radar_name.values) == ["TEST"]
    assert (single.radar_index == 0).all()
    other = volume(elevations=(1.0,), name="OTHER", lat=35.5, lon=-87.0, alt=300.0)
    both = superobs({"A": tree, "B": other}, 8.0)
    assert list(both.radar_name.values) == ["A", "B"]
    assert set(np.unique(both.radar_index)) == {0, 1}
    np.testing.assert_allclose(both.radar_latitude, [LAT0, 35.5])
    np.testing.assert_allclose(both.radar_altitude, [ALT0, 300.0])
    # the first radar alone gives the same observations
    first = both.where(both.radar_index == 0, drop=True)
    assert first.sizes["obs"] == single.sizes["obs"]
    np.testing.assert_allclose(first.reflectivity, single.reflectivity)
    # a list of bare sweeps of one site is one radar, of two sites two radars
    bare = superobs([sweep(1.0), sweep(2.0, seed=1)], 8.0)
    assert bare.sizes["radar"] == 1
    apart = superobs([sweep(1.0), sweep(1.0, lat=36.0)], 8.0)
    assert apart.sizes["radar"] == 2
    # without a name the instrument name of the volume is used
    assert list(superobs([tree, other], 8.0).radar_name.values) == ["TEST", "OTHER"]


def test_sweep_without_fields_and_errors():
    ds = sweep()
    with pytest.raises(KeyError):
        superobs(ds, 5.0, dbz="nope")
    nothing = ds.drop_vars(["DBZH", "VRADH"])
    with pytest.raises(KeyError):
        superobs(nothing, 5.0)
    for bad in (dict(missing="x"), dict(cells=0), dict(min_gates=0)):
        with pytest.raises(ValueError):
            superobs(ds, 5.0, **bad)
    with pytest.raises(ValueError):
        superobs(ds, 5.0, layers=[3.0, 1.0])
    with pytest.raises(ValueError):
        superobs(ds, -1.0)
    with pytest.raises(TypeError):
        superobs(42, 5.0)
    with pytest.raises(ValueError):
        superobs([], 5.0)
    no_site = ds.drop_vars("latitude")
    with pytest.raises(KeyError):
        superobs(no_site, 5.0)


def test_input_is_not_changed_and_dask_stays_lazy():
    dask = pytest.importorskip("dask")  # noqa: F841
    ds = sweep(rho=True)
    lazy = ds.chunk({"azimuth": 30})
    before = lazy.copy(deep=True).compute()
    so = superobs(lazy, 5.0, rho_min=0.8)
    assert so.sizes["obs"] > 0
    assert lazy.DBZH.chunks is not None  # still dask-backed
    assert lazy.VRADH.chunks is not None
    xr.testing.assert_identical(lazy.compute(), before)
    eager = superobs(ds, 5.0, rho_min=0.8)
    xr.testing.assert_identical(so, eager)
    tree = volume()
    keep = tree.copy(deep=True)
    superobs(tree, 5.0)
    xr.testing.assert_identical(
        tree["sweep_0"].to_dataset(), keep["sweep_0"].to_dataset()
    )
    assert "radarx_method" not in tree.attrs


def test_attrs_coordinates_and_provenance():
    tree = volume()
    so = superobs(tree, model_grid(n=50, dx=3000.0), cells=3, layers=250.0, top=3000.0)
    assert so.attrs["radial_velocity_convention"] == "positive away from the radar"
    assert so.attrs["radarx_method"].startswith("Super-observations")
    assert "10.1175/MWR3324.1" in so.attrs["radarx_references"].split()
    assert so.attrs["history"].startswith("superobs(")
    assert so.velocity.attrs["units"] == "m s-1"
    assert so.reflectivity.attrs["units"] == "dBZ"
    assert so.level_edges.size == 13 and so.level_edges[-1] == 3000.0
    assert so.height.max() < 3000.0
    np.testing.assert_allclose(
        so.level_height,
        0.5 * (so.level_edges.values[so.level] + so.level_edges.values[so.level + 1]),
    )
    assert np.issubdtype(so.radar_time.dtype, np.datetime64)
    for name in ("radial_east", "radial_north", "radial_up"):
        ok = so.velocity.notnull()
        assert so[name][ok].notnull().all()
        assert np.abs(so[name][ok]).max() <= 1.0 + 1e-12


def test_accessor_matches_function():
    tree = volume()
    a = tree.radarx.superobs(6.0, layers=250.0)
    b = superobs(tree, 6.0, layers=250.0)
    xr.testing.assert_identical(a, b)
    s = sweep()
    xr.testing.assert_identical(s.radarx.superobs(6.0), superobs(s, 6.0))


# --------------------------------------------------------------------------
# WRFDA ob.radar writer and a fixed-width reader
# --------------------------------------------------------------------------


def read_ob_radar(path):
    """
    Reader of the ob.radar format with the fixed-width fields of WRFDA.

    Header: ``(A14, I3)`` for the radar count (columns 15 to 17), then per
    radar ``(A5, 2X, A12, 2F8.3 ...)``: the name starts at column 8, the
    longitude, latitude and altitude are ``F8.3``, ``F8.3`` and ``F8.1``
    separated by two blanks, the time is ``A19`` followed by two ``I6``;
    per column ``(A12, 3X, A19, 2X, 2(F12.3, 2X), F8.1, 2X, I6)``; per level
    ``(3X, F12.1, 2(F12.3, I4, F12.3, 2X))``.
    """
    with open(path, encoding="ascii") as handle:
        lines = handle.read().split("\n")
    assert lines[-1] == ""
    lines = lines[:-1]
    assert lines[0][:14] == "TOTAL NUMBER =" and lines[0][14:17].strip().isdigit()
    count = int(lines[0][14:17])
    assert len(lines[0]) == 17
    assert lines[1] == "#-----------------#"
    at = 2
    radars = []
    for _ in range(count):
        assert lines[at] == ""
        head = lines[at + 1]
        assert head[:5] == "RADAR" and head[5:7] == "  "
        radar = {
            "name": head[7:19].strip(),
            "lon": float(head[19:27]),
            "lat": float(head[29:37]),
            "alt": float(head[39:47]),
            "time": head[49:68],
            "ncol": int(head[68:74]),
            "nlev": int(head[74:80]),
            "columns": [],
        }
        assert len(head) == 80
        assert lines[at + 2] == "#" + "-" * 79 + "#"
        assert lines[at + 3].startswith("#  MM/DD/YYYY")
        at += 4
        for _ in range(radar["ncol"]):
            head = lines[at]
            assert head[:12] == "FM-128 RADAR" and head[12:15] == "   "
            col = {
                "time": head[15:34],
                "lat": float(head[36:48]),
                "lon": float(head[50:62]),
                "elv": float(head[64:72]),
                "nlev": int(head[74:80]),
                "levels": [],
            }
            assert len(head) == 80 and col["nlev"] == radar["nlev"]
            at += 1
            for _ in range(col["nlev"]):
                row = lines[at]
                assert len(row) == 3 + 12 + 2 * (12 + 4 + 12 + 2) and row[:3] == "   "
                col["levels"].append(
                    {
                        "height": float(row[3:15]),
                        "rv": float(row[15:27]),
                        "rv_qc": int(row[27:31]),
                        "rv_err": float(row[31:43]),
                        "rf": float(row[45:57]),
                        "rf_qc": int(row[57:61]),
                        "rf_err": float(row[61:73]),
                    }
                )
                at += 1
            radar["columns"].append(col)
        radars.append(radar)
    assert at == len(lines)
    return radars


@pytest.fixture
def written(tmp_path):
    tree = volume(elevations=(0.5, 1.5), ngate=140)
    ds = sweep(echo=0.0, ngate=140)
    ds["DBZH"].values[:, 60:] = 25.0
    ds["VRADH"].values[:, 60:] = 3.0
    other = xr.DataTree(xr.Dataset(attrs={"instrument_name": "SECOND"}))
    other["sweep_0"] = xr.DataTree(ds.assign_coords(latitude=LAT0 + 0.4))
    so = superobs(
        {"KAAA": tree, "KBBB": other},
        model_grid(n=60, dx=3000.0),
        layers=500.0,
        top=6000.0,
        terrain=np.full((60, 60), 123.0),
    )
    path = tmp_path / "ob.radar"
    stats = write_wrfda_radar(so, path, time="2022-03-31_00:00:00")
    return so, path, stats


def test_wrfda_round_trip(written):
    so, path, stats = written
    radars = read_ob_radar(path)
    assert len(radars) == 2 == stats["radars"]
    assert [r["name"] for r in radars] == ["KAAA", "KBBB"]
    assert sum(r["ncol"] for r in radars) == stats["columns"]
    first = radars[0]
    assert first["lat"] == pytest.approx(LAT0, abs=5e-4)
    assert first["lon"] == pytest.approx(LON0, abs=5e-4)
    assert first["alt"] == pytest.approx(ALT0, abs=0.05)
    assert first["time"] == "2022-03-31_00:00:00"
    assert first["nlev"] == 12
    centers = 250.0 + 500.0 * np.arange(12)
    n_rv = n_rf = 0
    for k, radar in enumerate(radars):
        sub = so.where(so.radar_index == k, drop=True)
        assert radar["ncol"] == np.unique(sub.column).size
        for col, ref in zip(radar["columns"], np.unique(sub.column)):
            rows = sub.where(sub.column == ref, drop=True)
            assert col["lat"] == pytest.approx(float(rows.latitude.mean()), abs=5e-4)
            assert col["lon"] == pytest.approx(float(rows.longitude.mean()), abs=5e-4)
            assert col["elv"] == pytest.approx(123.0)
            np.testing.assert_allclose([lv["height"] for lv in col["levels"]], centers)
            for lev in col["levels"]:
                k_level = int(round((lev["height"] - 250.0) / 500.0))
                hit = rows.where(rows.level == k_level, drop=True)
                if hit.sizes["obs"] and np.isfinite(float(hit.velocity[0])):
                    assert lev["rv_qc"] == 0
                    assert lev["rv"] == pytest.approx(float(hit.velocity[0]), abs=5e-4)
                    assert lev["rv_err"] == pytest.approx(
                        float(hit.velocity_error[0]), abs=5e-4
                    )
                    n_rv += 1
                else:
                    assert lev["rv"] == -888888.0 and lev["rv_qc"] == -88
                    assert lev["rv_err"] == 2.0
                if hit.sizes["obs"] and int(hit.nonprecip[0]):
                    assert lev["rf"] == -999.99 and lev["rf_qc"] == 0
                    n_rf += 1
                elif hit.sizes["obs"] and np.isfinite(float(hit.reflectivity[0])):
                    assert lev["rf"] == pytest.approx(
                        float(hit.reflectivity[0]), abs=5e-4
                    )
                    assert lev["rf_qc"] == 0
                    n_rf += 1
                else:
                    assert lev["rf"] == -888888.0 and lev["rf_qc"] == -88
                    assert lev["rf_err"] == 5.0
    assert n_rv == stats["velocities"] > 0
    assert n_rf == stats["reflectivities"] > 0
    assert int(so.nonprecip.sum()) > 0


def test_wrfda_nonprecip_option_and_time(tmp_path, written):
    so, _, _ = written
    with_value = tmp_path / "a"
    without = tmp_path / "b"
    write_wrfda_radar(so, with_value, nonprecip=-777.5)
    write_wrfda_radar(so, without, nonprecip=None, missing_error=(1.0, 3.0))
    levels = lambda p: [  # noqa: E731
        lv for r in read_ob_radar(p) for c in r["columns"] for lv in c["levels"]
    ]
    a, b = levels(with_value), levels(without)
    assert any(lv["rf"] == -777.5 for lv in a)
    assert not any(lv["rf"] == -777.5 or lv["rf"] == -999.99 for lv in b)
    assert all(lv["rf_err"] == 3.0 for lv in b if lv["rf_qc"] == -88)
    assert all(lv["rv_err"] == 1.0 for lv in b if lv["rv_qc"] == -88)
    # the time of the volume is used by default
    radars = read_ob_radar(with_value)
    stamp = pd.Timestamp(so.radar_time.values[0]).strftime("%Y-%m-%d_%H:%M:%S")
    assert radars[0]["time"] == stamp


def test_wrfda_without_terrain_and_radars_without_observations(tmp_path):
    tree = volume()
    so = superobs(tree, 8.0)
    path = tmp_path / "ob.radar"
    write_wrfda_radar(so, path)
    radars = read_ob_radar(path)
    assert radars[0]["columns"][0]["elv"] == 0.0
    # a radar without observations is not written
    two = superobs(
        {"A": tree, "B": volume(lat=50.0, lon=10.0)},
        model_grid(n=40, dx=3000.0),
        cells=3,
    )
    assert set(np.unique(two.radar_index)) == {0}
    write_wrfda_radar(two, path)
    assert len(read_ob_radar(path)) == 1


def test_wrfda_errors(tmp_path):
    with pytest.raises(ValueError, match="superobs"):
        write_wrfda_radar(xr.Dataset(), tmp_path / "x")
    so = superobs(volume(), 8.0)
    with pytest.raises(ValueError, match="no observations"):
        write_wrfda_radar(so.isel(obs=slice(0, 0)), tmp_path / "x")
    stamp = so.assign_coords(radar_time=("radar", np.array(["NaT"], "datetime64[ns]")))
    with pytest.raises(ValueError, match="time"):
        write_wrfda_radar(stamp, tmp_path / "x")
    write_wrfda_radar(stamp, tmp_path / "ok", time=np.datetime64("2020-01-02T03:04:05"))
    assert read_ob_radar(tmp_path / "ok")[0]["time"] == "2020-01-02_03:04:05"


# --------------------------------------------------------------------------
# real volume
# --------------------------------------------------------------------------


def test_real_nexrad_volume():
    xd = pytest.importorskip("xradar")
    try:
        from open_radar_data import DATASETS

        path = DATASETS.fetch("KLBB20160601_150025_V06")
    except Exception as err:  # pragma: no cover - needs the network
        pytest.skip(f"test data not available: {err}")
    tree = xd.io.open_nexradlevel2_datatree(path)
    lat, lon = model_grid(n=80, dx=2000.0, lat0=33.65, lon0=-101.81)
    so = superobs(tree, (lat, lon), cells=3, rho_min=0.85, max_range=100e3)
    assert so.sizes["obs"] > 100
    assert np.abs(so.velocity).max() <= 40.0
    assert (so.height >= 0).all() and (so.height < 12000).all()
    assert (so.range <= 100e3).all()
    assert (so.gate_count >= 6).all()
    if mod.HAS_COMPILED_KERNEL:
        ref = superobs(
            tree, (lat, lon), cells=3, rho_min=0.85, max_range=100e3, engine="numpy"
        )
        np.testing.assert_allclose(
            so.reflectivity, ref.reflectivity, rtol=1e-11, equal_nan=True
        )
        np.testing.assert_allclose(
            so.velocity, ref.velocity, rtol=1e-11, equal_nan=True
        )
