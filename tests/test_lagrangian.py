#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests of radarx.retrieve.lagrangian (gridpoint air trajectories)."""

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401  registers the accessors
from radarx.retrieve import lagrangian as lg
from radarx.retrieve import trajectories

needs_kernel = pytest.mark.skipif(
    not lg.HAS_COMPILED_KERNEL, reason="compiled trajectory kernel not built"
)

DIMS = ("time", "z", "y", "x")
T0 = np.datetime64("2022-03-30T23:00:00", "ns")


def _winds(u, v, w, dbz=30.0, nt=4, step=600.0, c=None, z=None):
    """Dataset of analytic winds u(t, x, y, z) etc. on a 20-km grid."""
    c = np.arange(0.0, 20001.0, 1000.0) if c is None else c
    z = np.arange(0.0, 5001.0, 500.0) if z is None else z
    t = np.arange(nt) * step
    tt, zz, yy, xx = np.meshgrid(t, z, c, c, indexing="ij")
    data = {}
    for name, f in (("u", u), ("v", v), ("w", w), ("DBZ", dbz)):
        val = f(tt, xx, yy, zz) if callable(f) else np.full(tt.shape, float(f))
        data[name] = (DIMS, val)
    times = T0 + (t * 1e9).astype("timedelta64[ns]")
    return xr.Dataset(data, coords={"time": times, "z": z, "y": c, "x": c})


def _rotation(om=1.0e-3, xc=10000.0, yc=10000.0, **kw):
    return _winds(
        lambda t, x, y, z: -om * (y - yc), lambda t, x, y, z: om * (x - xc), 0.0, **kw
    )


def _point(x, y, z):
    return {"x": [x], "y": [y], "z": [z]}


# --------------------------------------------------------------------------
# analytic trajectories
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ["auto", "numpy"])
def test_uniform_flow_is_exact(engine):
    ds = _winds(5.0, -2.0, 0.5)
    tr = trajectories(
        ds,
        time=T0,
        direction="forward",
        start=_point(2000.0, 15000.0, 1000.0),
        engine=engine,
    )
    n = int(tr.n_points[0])
    assert int(tr.flags[0]) == 16  # max_steps: the end of the data
    s = np.arange(n) * 20.0
    np.testing.assert_allclose(tr.x[0, :n], 2000.0 + 5.0 * s, atol=1e-9)
    np.testing.assert_allclose(tr.y[0, :n], 15000.0 - 2.0 * s, atol=1e-9)
    np.testing.assert_allclose(tr.z[0, :n], 1000.0 + 0.5 * s, atol=1e-9)
    assert tr.time[0, n - 1].values == T0 + np.timedelta64(1800, "s")
    assert not bool(tr.environment[0])


@pytest.mark.parametrize("iterations", [1, 3])
def test_solid_body_rotation(iterations):
    om, r0 = 1.0e-3, 5000.0
    ds = _rotation(om)
    tr = trajectories(
        ds,
        start=_point(10000.0 + r0, 10000.0, 1000.0),
        iterations=iterations,
        termination=False,
    )
    n = int(tr.n_points[0])
    s = -np.arange(n) * 20.0
    x = tr.x[0, :n].values - 10000.0
    y = tr.y[0, :n].values - 10000.0
    # radius and angle of the analytic circular trajectory
    assert np.abs(np.hypot(x, y) - r0).max() < 1.0
    ang = np.unwrap(np.arctan2(y, x))
    assert np.abs(ang - om * s).max() < 2e-4


def test_backward_forward_closure():
    # time-dependent, curved flow: rotation plus a growing shear
    def u(t, x, y, z):
        return -1e-3 * (y - 1e4) + 2.0 * np.sin(2e-4 * x) * (1 + t / 1800.0)

    def v(t, x, y, z):
        return 1e-3 * (x - 1e4) + 1.0

    def w(t, x, y, z):
        return 0.5 * np.cos(3e-4 * y)

    ds = _winds(u, v, w)
    start = _point(12000.0, 9000.0, 2000.0)
    back = trajectories(ds, start=start, termination=False)
    n = int(back.n_points[0])
    end = {k: [float(back[k][0, n - 1])] for k in ("x", "y", "z")}
    t_end = back.time[0, n - 1].values
    fwd = trajectories(ds, time=t_end, direction="forward", start=end, max_steps=n - 1)
    m = int(fwd.n_points[0])
    assert m == n
    err = np.hypot(float(fwd.x[0, m - 1]) - 12000.0, float(fwd.y[0, m - 1]) - 9000.0)
    assert err < 25.0  # metres after 30 min, Ziegler's (2013a) choice of dt
    assert abs(float(fwd.z[0, m - 1]) - 2000.0) < 5.0


def test_storm_motion_equivalence():
    """Analyses moving with the storm equal one analysis time-morphed."""
    cx, cy = 10.0, -5.0

    def field(dx, dy):
        return lambda t, x, y, z: 3.0 * np.sin(2e-4 * (x - dx - cx * t)) + 1e-4 * (
            y - dy - cy * t
        )

    ds = _winds(field(0, 0), field(500.0, 0), 0.0, step=200.0, nt=6)
    start = {"x": [9000.0, 12000.0], "y": [9000.0, 11000.0], "z": [1000.0, 2500.0]}
    a = trajectories(ds, start=start, storm_motion=(cx, cy), termination=False)
    one = ds.isel(time=[-1])
    b = trajectories(
        one, start=start, storm_motion=(cx, cy), extend=1000.0, termination=False
    )
    n = int(a.n_points[0])
    assert n == 51
    for k in ("x", "y", "z", "u", "v"):
        np.testing.assert_allclose(a[k][:, :n], b[k][:, :n], atol=1e-6)
    # without storm motion the grid of analyses is fixed: a different path
    c = trajectories(ds, start=start, termination=False)
    assert float(abs(c.x[0, n - 1] - a.x[0, n - 1])) > 1.0


def test_single_analysis_needs_morphing():
    ds = _winds(1.0, 0.0, 0.0).isel(time=0)
    with pytest.raises(ValueError, match="time morphing"):
        trajectories(ds, start=_point(5000.0, 5000.0, 500.0))
    tr = trajectories(
        ds.drop_vars("time"),
        start=_point(15000.0, 5000.0, 500.0),
        storm_motion=(0.0, 0.0),
        extend=200.0,
    )
    assert int(tr.n_points[0]) == 11
    np.testing.assert_allclose(float(tr.x[0, 10]), 15000.0 - 200.0)


# --------------------------------------------------------------------------
# surface parcels, termination
# --------------------------------------------------------------------------


def test_surface_downdraft_equations():
    w = np.zeros((1, 3, 1, 4))
    w[0, 1, 0] = [-4.0, -1.0, 2.0, -4.0]
    dbz = np.zeros((1, 3, 1, 4))
    dbz[0, 0, 0] = [55.0, 45.0, 55.0, 30.0]
    ws = lg.surface_downdraft(w, dbz)
    # Z* = 1, 0.5, 1, 0; w_sfc = max(Z* 0.5 w2, -0.75), 0 for w2 >= 0
    np.testing.assert_allclose(ws[0, 0], [-0.75, -0.25, 0.0, 0.0])


def test_surface_parcels_and_downdraft():
    def w(t, x, y, z):
        return np.where(z > 0, -2.0, 0.0)

    ds = _winds(0.0, 0.0, w, dbz=55.0)
    tr = trajectories(ds, levels=[0], termination=False, max_steps=10)
    assert (tr.start_z == 10.0).all() and (tr.k == 0).all()
    # w at the ground is the parameterised downdraft: max(0.5 * -2, -0.75)
    np.testing.assert_allclose(tr.w[0, 0], -0.75 + (10.0 / 500.0) * (-2.0 + 0.75))
    assert float(tr.z[0, 10]) > 10.0  # surface parcels came from above
    off = trajectories(
        ds, levels=[0], termination=False, max_steps=10, surface_downdraft=False
    )
    np.testing.assert_allclose(off.w[0, 0], 10.0 / 500.0 * -2.0)


def run(ds, start, **kw):
    return trajectories(ds, start=start, surface_downdraft=False, **kw)


def test_termination_flags():
    ds = _winds(0.0, 0.0, 1.0, dbz=-10.0, nt=6)
    tr = run(ds, _point(5000.0, 5000.0, 1000.0))
    assert int(tr.flags[0]) == 1 and int(tr.n_points[0]) == 78
    ds = _winds(0.0, 0.0, 0.2, dbz=40.0, nt=6)
    tr = run(ds, _point(5000.0, 5000.0, 1000.0))
    assert int(tr.flags[0]) == 2 and bool(tr.environment[0])
    ds = _winds(-20.0, 0.0, 1.0, dbz=40.0, nt=6)
    tr = run(ds, _point(17000.0, 5000.0, 1000.0))
    assert int(tr.flags[0]) == 4  # through the eastern boundary
    tr = run(ds, _point(25000.0, 5000.0, 1000.0))
    assert int(tr.flags[0]) == 4 and int(tr.n_points[0]) == 0
    ds = _winds(0.0, 0.0, 1.0, dbz=40.0, nt=2)
    tr = run(ds, _point(5000.0, 5000.0, 1000.0), max_steps=100)
    assert int(tr.flags[0]) == 8 and int(tr.n_points[0]) == 31
    tr = run(ds, _point(5000.0, 5000.0, 1000.0), max_steps=5)
    assert int(tr.flags[0]) == 16
    ds["u"] = ds.u.where(ds.x < 4500.0)
    tr = run(ds, _point(5000.0, 5000.0, 1000.0))
    assert int(tr.flags[0]) == 32 and int(tr.n_points[0]) == 0
    ds = _winds(-5.0, 0.0, 0.0, dbz=40.0, nt=2)
    ds["u"] = ds.u.where(ds.x < 9000.0)
    tr = run(ds, _point(5000.0, 5000.0, 1000.0), engine="numpy")
    assert int(tr.flags[0]) == 32 and int(tr.n_points[0]) > 1


def test_missing_reflectivity():
    ds = _winds(0.0, 0.0, 1.0, nt=6).drop_vars("DBZ")
    with pytest.warns(UserWarning, match="no reflectivity"):
        tr = run(ds, _point(5000.0, 5000.0, 1000.0), max_steps=200)
    assert int(tr.flags[0]) == 8
    ds = _winds(0.0, 0.0, 1.0, dbz=np.nan, nt=6)
    tr = run(ds, _point(5000.0, 5000.0, 1000.0))
    assert int(tr.flags[0]) == 1  # NaN reflectivity is no echo
    tr = run(ds, _point(5000.0, 5000.0, 1000.0), reflectivity=None, max_steps=200)
    assert int(tr.flags[0]) == 8


# --------------------------------------------------------------------------
# engines, output, errors
# --------------------------------------------------------------------------


@needs_kernel
@pytest.mark.parametrize("direction", ["backward", "forward"])
def test_engines_agree(direction):
    def u(t, x, y, z):
        return 8.0 * np.sin(3e-4 * y) + t / 600.0

    def w(t, x, y, z):
        return 3.0 * np.cos(2e-4 * x) * np.sin(np.pi * z / 5000.0)

    def dbz(t, x, y, z):
        return 50.0 - 6e-3 * np.hypot(x - 9e3, y - 1e4)

    ds = _winds(u, lambda t, x, y, z: 4.0 + 1e-4 * x, w, dbz=dbz)
    kw = {
        "direction": direction,
        "storm_motion": (3.0, 2.0),
        "extend": 300.0,
        "levels": [0, 3],
    }
    if direction == "forward":
        kw["time"] = T0 + np.timedelta64(300, "s")
    a = trajectories(ds, engine="compiled", n_threads=3, **kw)
    b = trajectories(ds, engine="numpy", **kw)
    np.testing.assert_array_equal(a.n_points, b.n_points)
    np.testing.assert_array_equal(a.flags, b.flags)
    for k in ("x", "y", "z", "u", "v", "w", "reflectivity"):
        np.testing.assert_allclose(a[k], b[k], atol=1e-7, equal_nan=True)
    assert len(np.unique(a.flags)) > 1


def test_gridpoint_output():
    ds = _rotation(c=np.arange(0.0, 6001.0, 1000.0), z=np.arange(0.0, 1001.0, 500.0))
    tr = ds.radarx.trajectories(time=ds.time[-1])
    assert tr.sizes["trajectory"] == 3 * 7 * 7
    assert set(tr.coords) >= {"k", "j", "i", "start_x", "step", "analysis_time"}
    assert tr.flags.attrs["flag_meanings"].startswith("environment")
    assert tr.attrs["direction"] == "backward"
    assert tr.u.attrs["units"] == "m s-1"
    # NaN past the end of each trajectory
    short = int(np.argmin(tr.n_points.values))
    assert np.isnan(tr.x[short, int(tr.n_points[short]) :]).all()
    assert np.isnat(tr.time[short, -1].values)


def test_errors():
    ds = _winds(1.0, 0.0, 0.0)
    p = _point(5000.0, 5000.0, 500.0)
    with pytest.raises(ValueError, match="direction"):
        trajectories(ds, start=p, direction="up")
    with pytest.raises(ValueError, match="dt"):
        trajectories(ds, start=p, dt=0.0)
    with pytest.raises(ValueError, match="iterations"):
        trajectories(ds, start=p, iterations=-1)
    with pytest.raises(ValueError, match="engine"):
        trajectories(ds, start=p, engine="gpu")
    with pytest.raises(ValueError, match="unknown parameters"):
        trajectories(ds, start=p, parameters={"h0": 1})
    with pytest.raises(ValueError, match="variable 'w'"):
        trajectories(ds.drop_vars("w"), start=p)
    with pytest.raises(TypeError):
        trajectories(ds.u, start=p)
    with pytest.raises(ValueError, match="increasing"):
        trajectories(ds.isel(x=slice(None, None, -1)), start=p)
    with pytest.raises(ValueError, match="reflectivity"):
        trajectories(ds, start=p, reflectivity="ZH")
    with pytest.raises(ValueError, match="same size"):
        trajectories(ds, start={"x": [1.0, 2.0], "y": [1.0], "z": [1.0]})
    with pytest.raises(ValueError, match="extend"):
        trajectories(ds, start=p, extend=-1.0)
    with pytest.raises(ValueError, match="distinct"):
        trajectories(xr.concat([ds.isel(time=0), ds.isel(time=0)], "time"), start=p)
    with pytest.raises(MemoryError):
        trajectories(ds, max_steps=10**7)
    if not lg.HAS_COMPILED_KERNEL:  # pragma: no cover - depends on the build
        with pytest.raises(ImportError):
            trajectories(ds, start=p, engine="compiled")


def test_options_and_coordinates():
    ds = _winds(1.0, 0.0, 0.0)
    p = _point(5000.0, 5000.0, 500.0)
    a = trajectories(
        ds,
        start=p,
        reflectivity="DBZ",
        storm_motion=(1.0, 0.0),
        extend=(60.0, 0.0),
        termination=False,
    )
    assert int(a.n_points[0]) == 94
    with pytest.raises(ValueError, match="coordinate"):
        trajectories(ds.drop_vars("y"), start=p)
