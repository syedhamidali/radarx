#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests of the squall-line options of the trajectories and the DLA: time
morphing before and after the wind series, storm motion estimated from the
reflectivity, the lateral-boundary rule and the shipped closure tables."""

import importlib
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401  registers the accessors
from radarx.retrieve import (
    diabatic_lagrangian,
    trajectories,
    ziegler2013_precipitation,
    ziegler2013_profiles,
)
from radarx.retrieve import lagrangian as lg

dl = importlib.import_module("radarx.retrieve.diabatic_lagrangian")

DIMS = ("time", "z", "y", "x")
T0 = np.datetime64("2022-03-30T23:00:00", "ns")
ENGINES = ["auto", "numpy"]


def _winds(
    u, v=0.0, w=0.0, dbz=-10.0, nt=2, step=600.0, nx=41, extra=None, ztop=3000.0
):
    """Analytic winds on a 1-km grid, x and y from 0 to (nx - 1) km."""
    c = np.arange(nx) * 1000.0
    z = np.arange(0.0, ztop + 1.0, 500.0)
    t = np.arange(nt) * step
    tt, zz, yy, xx = np.meshgrid(t, z, c, c, indexing="ij")
    data = {}
    for name, f in {"u": u, "v": v, "w": w, "DBZ": dbz, **(extra or {})}.items():
        val = f(tt, xx, yy, zz) if callable(f) else np.full(tt.shape, float(f))
        data[name] = (DIMS, val)
    times = T0 + (t * 1e9).astype("timedelta64[ns]")
    return xr.Dataset(data, coords={"time": times, "z": z, "y": c, "x": c})


def _point(x, y, z=1000.0):
    return {"x": [x], "y": [y], "z": [z]}


def _sounding():
    h = np.arange(0.0, 12001.0, 50.0)
    t = 300.0 - 0.0065 * h
    p = 1e5 * (t / 300.0) ** (9.80665 / (287.04 * 0.0065))
    q = np.where(h < 1500, 0.012, 0.004)
    return xr.Dataset(
        {
            "pressure": ("height", p),
            "temperature": ("height", t),
            "specific_humidity": ("height", q),
        },
        coords={"height": h},
    )


def _assert_same(a, b):
    for k in ("x", "y", "z", "u", "reflectivity"):
        np.testing.assert_allclose(a[k].values, b[k].values, atol=1e-6, equal_nan=True)
    np.testing.assert_array_equal(a.flags.values, b.flags.values)
    np.testing.assert_array_equal(a.n_points.values, b.n_points.values)


# --------------------------------------------------------------------------
# time morphing (Ziegler 2013b, sect. 2c)
# --------------------------------------------------------------------------

PRECIP = {"min_steps": 0}


def _mask_west(xb):
    return {"ahead": lambda t, x, y, z: (x < xb).astype(float)}


@pytest.mark.parametrize("engine", ENGINES)
def test_extension_reaches_air_older_than_the_winds(engine):
    """Uniform flow: the environment (x < 20 km) is 18 km upstream of the
    start, reachable in 1800 s, but the winds span only 600 s."""
    ds = _winds(10.0, extra=_mask_west(20000.0))
    kw = dict(
        start=_point(38000.0, 20000.0),
        termination="precipitation",
        environment_mask="ahead",
        parameters=PRECIP,
        surface_downdraft=False,
        engine=engine,
    )
    a = trajectories(ds, **kw)
    assert int(a.flags[0]) == lg._np_kernel.MAX_STEPS and not bool(a.environment[0])
    n = int(a.n_points[0])
    np.testing.assert_allclose(float(a.x[0, n - 1]), 38000.0 - 6000.0, atol=1e-6)
    # 1500 s before the first analysis: the parcel reaches x < 20 km
    b = trajectories(ds, extend_before=1500.0, storm_motion=(0.0, 0.0), **kw)
    n = int(b.n_points[0])
    assert int(b.flags[0]) == lg._np_kernel.ENV_DBZ and bool(b.environment[0])
    # it stops at the first point where the interpolated mask reaches 0.5
    xe = float(b.x[0, n - 1])
    np.testing.assert_allclose(xe, 38000.0 - 10.0 * 20.0 * (n - 1), atol=1e-6)
    assert 19300.0 <= xe <= 19500.0
    tend = (b.time[0, n - 1].values - T0) / np.timedelta64(1, "s")
    assert tend < 0.0  # before the first analysis
    assert b.attrs["extend"] == (1500.0, 0.0)
    # extend_after only extends forward trajectories
    c = trajectories(ds, extend_after=1500.0, storm_motion=(0.0, 0.0), **kw)
    assert int(c.n_points[0]) == int(a.n_points[0])
    f = trajectories(
        ds,
        direction="forward",
        start=_point(1000.0, 20000.0),
        extend_after=600.0,
        storm_motion=(0.0, 0.0),
        engine=engine,
        reflectivity=None,
    )
    nf = int(f.n_points[0])
    np.testing.assert_allclose(float(f.x[0, nf - 1]), 1000.0 + 12000.0, atol=1e-6)


@pytest.mark.parametrize("engine", ENGINES)
def test_extension_moves_the_first_analysis_with_the_storm(engine):
    """With storm motion c_x the environment edge of the first analysis
    (x_b = 20 km at t_0 = -600 s) is at 20 km + c_x (t - t_0) at t < t_0. A
    parcel at 38 km - 10 m/s |t| meets it at t = -2400 s for c_x = 5 m/s."""
    ds = _winds(10.0, extra=_mask_west(20000.0))
    kw = dict(
        start=_point(38000.0, 20000.0),
        termination="precipitation",
        environment_mask="ahead",
        parameters=PRECIP,
        surface_downdraft=False,
        storm_motion=(5.0, 0.0),
        engine=engine,
    )
    short = trajectories(ds, extend_before=1500.0, **kw)
    assert not bool(short.environment[0])  # ends at t = -2100 s
    b = trajectories(ds, extend_before=3000.0, **kw)
    n = int(b.n_points[0])
    assert bool(b.environment[0])
    tend = (b.time[0, n - 1].values - T0) / np.timedelta64(1, "s")
    # the interpolated mask reaches 0.5 500 m behind the edge: t = -2500 s
    assert -2500.0 - 20.0 - 1e-6 <= tend <= -2500.0 + 1e-6
    other = trajectories(ds, extend_before=3000.0, **{**kw, "engine": "numpy"})
    _assert_same(b, other)


def test_extension_options_and_warnings():
    ds = _winds(10.0)
    p = _point(30000.0, 20000.0)
    with pytest.warns(UserWarning, match="without storm_motion"):
        a = trajectories(ds, start=p, extend_before=200.0, reflectivity=None)
    # extend_before overrides extend
    b = trajectories(
        ds,
        start=p,
        extend=600.0,
        extend_before=0.0,
        storm_motion=(0.0, 0.0),
        reflectivity=None,
    )
    assert b.attrs["extend"] == (0.0, 600.0)
    assert int(a.n_points[0]) == int(b.n_points[0]) + 10
    with pytest.raises(ValueError, match="extend_before"):
        trajectories(ds, start=p, extend_before=-1.0)


# --------------------------------------------------------------------------
# storm motion estimated from the reflectivity
# --------------------------------------------------------------------------


def _moving_cells(cx, cy):
    def dbz(t, x, y, z):
        out = np.full(t.shape, -10.0)
        for x0, y0 in ((12000.0, 14000.0), (26000.0, 22000.0), (18000.0, 30000.0)):
            r2 = (x - x0 - cx * t) ** 2 + (y - y0 - cy * t) ** 2
            out = np.maximum(out, 50.0 * np.exp(-r2 / (2 * 2500.0**2)) - 5.0)
        return out

    return dbz


def test_storm_motion_estimate():
    ds = _winds(5.0, dbz=_moving_cells(8.0, -3.0), step=300.0)
    p = _point(30000.0, 20000.0)
    tr = trajectories(ds, start=p, storm_motion="estimate", termination=False)
    cx, cy = tr.attrs["storm_motion"]
    np.testing.assert_allclose([cx, cy], [8.0, -3.0], atol=0.5)
    # coordinates evenly spaced only to float32 rounding
    jitter = 0.01 * np.sin(np.arange(ds.sizes["x"]))
    tr3 = trajectories(
        ds.assign_coords(x=ds.x + jitter),
        start=p,
        storm_motion="estimate",
        termination=False,
    )
    np.testing.assert_allclose(tr3.attrs["storm_motion"], (cx, cy), atol=1e-6)
    motion = radarx.retrieve.estimate_motion(ds.isel(time=0), ds.isel(time=-1))
    tr2 = trajectories(ds, start=p, storm_motion=motion, termination=False)
    np.testing.assert_allclose(tr2.attrs["storm_motion"], (cx, cy))
    out = diabatic_lagrangian(
        ds,
        _sounding(),
        precipitation="none",
        storm_motion="estimate",
        levels=[2],
        filter_passes=0,
    )
    np.testing.assert_allclose(out.attrs["storm_motion"], (cx, cy))


def test_storm_motion_errors():
    ds = _winds(5.0, dbz=_moving_cells(8.0, -3.0), step=300.0)
    p = _point(30000.0, 20000.0)
    with pytest.raises(ValueError, match="two or more"):
        trajectories(ds.isel(time=[0]), start=p, storm_motion="estimate", extend=60.0)
    with pytest.raises(ValueError, match="two or more"):
        trajectories(ds, start=p, storm_motion="estimate", reflectivity=None)
    with pytest.raises(ValueError, match="'estimate'"):
        trajectories(ds, start=p, storm_motion="guess")
    with pytest.raises(ValueError, match="'u' and 'v'"):
        trajectories(ds, start=p, storm_motion=xr.Dataset({"u": 1.0}))
    tiled = xr.Dataset({"u": ("x", [1.0, 2.0]), "v": ("x", [0.0, 0.0])})
    with pytest.raises(ValueError, match="single"):
        trajectories(ds, start=p, storm_motion=tiled)
    with pytest.raises(ValueError, match="not finite"):
        trajectories(ds, start=p, storm_motion=(np.nan, 0.0))


# --------------------------------------------------------------------------
# lateral boundaries
# --------------------------------------------------------------------------


def _exit_west(engine, boundary, dbz=30.0, extra=None, **kw):
    """Uniform eastward flow: the backward trajectory from x = 5 km leaves the
    domain through its west edge after 500 s, never meeting the test."""
    ds = _winds(10.0, dbz=dbz, nt=3, extra=extra)
    return trajectories(
        ds,
        start=_point(5000.0, 20000.0),
        termination="precipitation",
        parameters=PRECIP,
        surface_downdraft=False,
        boundary=boundary,
        engine=engine,
        **kw,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_boundary_rules(engine):
    B, BS = lg._np_kernel.BOUNDARY, lg._np_kernel.BOUNDARY_STORM
    cases = [
        ("environment", {}, B),
        (["east", "north", "south"], {}, BS),
        (["west"], {}, B),
        ("West", {}, B),
        ("no_echo", {}, BS),  # 30 dBZ at the boundary point
        ("no_echo", {"dbz": -10.0}, B),
        ("environment_mask", {"extra": _mask_west(3000.0)}, B),
        ("environment_mask", {"extra": _mask_west(-1.0)}, BS),
        ("no_echo", {"extra": _mask_west(3000.0)}, B),
    ]
    for boundary, kw, flag in cases:
        mask = "ahead" if "extra" in kw else None
        tr = _exit_west(engine, boundary, environment_mask=mask, **kw)
        assert int(tr.flags[0]) == flag, boundary
        assert bool(tr.environment[0]) == (flag == B)
        n = int(tr.n_points[0])
        assert float(tr.x[0, n - 1]) - 10.0 * 20.0 < 0.0  # the exit point
        assert tr.attrs["boundary"] == str(boundary)
        if engine == "auto":
            _assert_same(tr, _exit_west("numpy", boundary, environment_mask=mask, **kw))
    assert tr.flags.attrs["flag_masks"][-1] == 128
    assert "lateral_boundary_not_environment" in tr.flags.attrs["flag_meanings"]


@pytest.mark.parametrize("engine", ENGINES)
def test_boundary_of_the_storm_moving_analysis(engine):
    """No wind, storm motion -10 m/s: backward in time the analysis frame
    position x - c_x (t - t_i) moves west, so the parcel leaves the moving
    analysis through its west side while it stays inside the fixed grid."""
    ds = _winds(0.0, dbz=30.0, nt=3)
    kw = dict(
        start=_point(5000.0, 20000.0),
        termination="precipitation",
        parameters=PRECIP,
        surface_downdraft=False,
        storm_motion=(-10.0, 0.0),
        engine=engine,
    )
    for boundary, flag in ((["west"], 4), (["east"], 128), ("environment", 4)):
        tr = trajectories(ds, boundary=boundary, **kw)
        assert int(tr.flags[0]) == flag
        n = int(tr.n_points[0])
        assert float(tr.x[0, n - 1]) == pytest.approx(5000.0)
    # the same through the north edge with a northward storm motion
    kn = {**kw, "storm_motion": (0.0, 10.0), "start": _point(5000.0, 35000.0)}
    tr = trajectories(ds, boundary=["north"], **kn)
    assert int(tr.flags[0]) == 4
    tr = trajectories(ds, boundary=["south"], **kn)
    assert int(tr.flags[0]) == 128
    # during time morphing before the first analysis
    tr = trajectories(
        ds.isel(time=[0]),
        boundary=["east"],
        extend_before=2000.0,
        **{**kw, "storm_motion": (10.0, 0.0), "start": _point(35000.0, 20000.0)},
    )
    assert int(tr.flags[0]) == 4
    assert -600.0 < float(tr.time[0, int(tr.n_points[0]) - 1] - ds.time[0]) / 1e9


def test_boundary_rule_needs_a_termination_test():
    """Forward trajectories and termination=False keep flag 4."""
    ds = _winds(-10.0, dbz=30.0, nt=3)
    tr = trajectories(
        ds,
        start=_point(35000.0, 20000.0),
        termination=False,
        boundary=["east"],
        surface_downdraft=False,
    )
    assert int(tr.flags[0]) == 4
    tr = trajectories(
        ds, start=_point(5000.0, 20000.0), direction="forward", boundary=["east"]
    )
    assert int(tr.flags[0]) == 4


def test_boundary_errors():
    ds = _winds(10.0)
    p = _point(5000.0, 20000.0)
    for bad in ("up", [], ["west", "up"], 3):
        with pytest.raises(ValueError, match="boundary"):
            trajectories(ds, start=p, boundary=bad)


@pytest.mark.parametrize("engine", ENGINES)
def test_dla_boundary_and_extension(engine):
    ds = _winds(10.0, dbz=30.0, nt=3, extra=_mask_west(3000.0))
    kw = dict(
        precipitation="none",
        termination="precipitation",
        parameters=PRECIP,
        levels=[2],
        hole_fill=False,
        filter_passes=0,
        engine=engine,
    )
    a = diabatic_lagrangian(ds, _sounding(), **kw)
    b = diabatic_lagrangian(ds, _sounding(), boundary=["east"], **kw)
    west = a.x < 11500.0  # parcels that leave through the west edge
    assert bool(a.environment.where(west, True).all())
    assert int((b.flags.where(west) == 128).sum()) == int(west.sum()) * ds.sizes["y"]
    assert not bool(b.environment.where(west, False).any())
    assert bool(b.theta.isel(x=west.values).isnull().all())
    assert b.attrs["boundary"] == "['east']"
    # environment_mask: the boundary points at x < 3 km are in the mask
    c = diabatic_lagrangian(
        ds, _sounding(), boundary="environment_mask", environment_mask="ahead", **kw
    )
    np.testing.assert_array_equal(c.flags.values & 128, 0)
    # time morphing reaches air older than the 1200 s of winds
    d = diabatic_lagrangian(
        ds,
        _sounding(),
        boundary=["east"],
        extend_before=3600.0,
        storm_motion=(0.0, 0.0),
        **kw,
    )
    assert float(d.origin_time.min()) < -1200.0
    assert d.attrs["extend"] == (3600.0, 0.0)
    if engine == "auto":
        e = diabatic_lagrangian(
            ds,
            _sounding(),
            boundary=["east"],
            extend_before=3600.0,
            storm_motion=(0.0, 0.0),
            **{**kw, "engine": "numpy"},
        )
        for k in ("theta", "qv", "qc", "origin_x", "origin_time"):
            np.testing.assert_allclose(d[k], e[k], rtol=1e-9, equal_nan=True)
        np.testing.assert_array_equal(d.flags, e.flags)


# --------------------------------------------------------------------------
# shipped regression profiles of the Ziegler (2013a) closure
# --------------------------------------------------------------------------


def test_shipped_profiles_reproduce_the_csv():
    path = (
        Path(dl.__file__).parent / "data" / "ziegler2013_profiles_cm1_squall_line.csv"
    )
    lines = path.read_text().splitlines()
    ncomment = sum(ln.startswith("#") for ln in lines)
    assert ncomment >= 10 and all(ln.startswith("#") for ln in lines[:ncomment])
    table = np.genfromtxt(path, delimiter=",", skip_header=ncomment, names=True)
    pr = ziegler2013_profiles()
    np.testing.assert_array_equal(pr.z_star.values, table["z_star"])
    for k in ("Z0r", "S0_qr", "Z0g", "S0_qg", "S0_n0g"):
        np.testing.assert_array_equal(pr[k].values, table[k])
    assert pr.z_star.size == 101 and pr.z_star[0] == 400.0 and pr.z_star[-1] == 10400
    assert "CM1" in pr.attrs["source"] and "squall line" in pr.attrs["title"]
    assert pr.attrs["n0r_m-4"] == 8.0e6 and pr.attrs["graupel_density_kg_m-3"] == 660
    assert pr.Z0r.attrs["units"] == "dBZ"
    with pytest.raises(ValueError, match="unknown profiles"):
        ziegler2013_profiles("supercell")


def test_closure_uses_the_shipped_profiles_by_default():
    ds = _winds(
        0.0, w=lambda t, x, y, z: 1e-3 * x - 5.0, dbz=45.0, nt=2, nx=21, ztop=7000.0
    )
    _, _, _, base = dl._base_state(_sounding(), ds.z.values, 0.0)
    ref = ziegler2013_precipitation(ds, base, profiles=ziegler2013_profiles())
    with pytest.warns(UserWarning, match="squall-line"):
        a = ziegler2013_precipitation(ds, base)
    with pytest.warns(UserWarning, match="squall-line"):
        b = ziegler2013_precipitation(ds, base, profiles="cm1_squall_line")
    for k in ("qr", "nr", "qg", "ng"):
        np.testing.assert_array_equal(a[k], ref[k])
        np.testing.assert_array_equal(b[k], ref[k])
    assert float(a.qr.max()) > 0 and float(a.qg.max()) > 0
    with pytest.warns(UserWarning, match="squall-line"):
        out = diabatic_lagrangian(
            ds, _sounding(), precipitation="ziegler2013", levels=[1], filter_passes=0
        )
    assert float(out.qr.max()) > 0
