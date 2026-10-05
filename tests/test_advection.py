# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for radarx advection correction
=====================================

Synthetic tests translate Gaussian echo blobs with a known motion, so the
motion and the advected fields have an analytic truth. The real-data test
translates a gridded NEXRAD volume by a known number of cells.
"""

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.retrieve import advection
from radarx.retrieve.advection import advect, estimate_motion, interpolate_time

ENGINES = ["numpy"] + (["compiled"] if advection.HAS_COMPILED_KERNEL else [])
needs_kernel = pytest.mark.skipif(
    not advection.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)
T0 = np.datetime64("2026-01-01T00:00:00", "ns")
X = np.arange(-60e3, 60e3 + 1, 1000.0)
Y = np.arange(-50e3, 50e3 + 1, 1000.0)
BLOBS = [(-20e3, 5e3, 50, 6e3), (15e3, -10e3, 40, 4e3), (0.0, 25e3, 45, 8e3)]


def _blobs(t, u, v, x=X, y=Y, z=None, threshold=1.0, dtype="f4"):
    """Echo blobs moved by (u, v) * t, NaN below ``threshold`` (no echo)."""
    X2, Y2 = np.meshgrid(x, y)
    f = np.zeros_like(X2)
    for x0, y0, amp, s in BLOBS:
        f += amp * np.exp(
            -((X2 - x0 - u * t) ** 2 + (Y2 - y0 - v * t) ** 2) / (2 * s**2)
        )
    if threshold is not None:
        f = np.where(f > threshold, f, np.nan)
    f = f.astype(dtype)
    dims, coords = ("y", "x"), {
        "x": ("x", x, {"units": "m"}),
        "y": ("y", y, {"units": "m"}),
    }
    if z is not None:
        f = np.stack([f * (1 - 0.05 * k) for k in range(len(z))])
        dims, coords = ("z",) + dims, dict(coords, z=("z", z, {"units": "m"}))
    ds = xr.Dataset(
        {"DBZH": (dims, f, {"units": "dBZ", "long_name": "reflectivity"})},
        coords=coords,
        attrs={"instrument_name": "synthetic"},
    )
    time = T0 + np.timedelta64(round(t * 1e9), "ns")
    return ds.assign(time=xr.DataArray(time, attrs={"long_name": "volume time"}))


def _interior(da, margin=5):
    return da.isel(x=slice(margin, -margin), y=slice(margin, -margin))


# ---------------------------------------------------------------------------
# motion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("u, v", [(13.7, -8.3), (-21.2, 4.45), (0.0, 0.0), (3.1, 17.9)])
def test_estimate_motion_recovers_translation(u, v):
    motion = estimate_motion(_blobs(0, u, v), _blobs(300, u, v))
    # sub-pixel: 1 cell = 1000 m / 300 s = 3.3 m/s; require < 0.02 cell
    assert abs(float(motion.u) - u) < 0.07
    assert abs(float(motion.v) - v) < 0.07
    assert float(motion.quality) > 0.95
    assert motion.u.attrs["units"] == "m s-1"
    assert motion.attrs["dt"] == 300.0


def test_estimate_motion_is_unbiased_without_refinement_check():
    """Refinement removes the window bias of a single correlation."""
    d0, d1 = _blobs(0, 13.7, -8.3), _blobs(300, 13.7, -8.3)
    single = estimate_motion(d0, d1, iterations=0)
    refined = estimate_motion(d0, d1)
    assert abs(float(refined.u) - 13.7) < abs(float(single.u) - 13.7)


def test_estimate_motion_dataarray_3d_and_explicit_dt():
    z = np.array([500.0, 1000.0, 1500.0])
    d0, d1 = _blobs(0, 10.0, 5.0, z=z), _blobs(240, 10.0, 5.0, z=z)
    m = estimate_motion(d0["DBZH"], d1["DBZH"], dt=np.timedelta64(240, "s"))
    assert abs(float(m.u) - 10.0) < 0.07 and abs(float(m.v) - 5.0) < 0.07
    m_ds = estimate_motion(d0, d1, "DBZH")
    assert np.isclose(float(m_ds.u), float(m.u))


def test_estimate_motion_descending_y():
    d0, d1 = _blobs(0, 8.0, -6.0), _blobs(300, 8.0, -6.0)
    m = estimate_motion(
        d0.isel(y=slice(None, None, -1)), d1.isel(y=slice(None, None, -1))
    )
    assert abs(float(m.u) - 8.0) < 0.07 and abs(float(m.v) + 6.0) < 0.07


def test_estimate_motion_tiled_uniform():
    u, v = 12.0, -7.0
    m = estimate_motion(_blobs(0, u, v), _blobs(300, u, v), tile=40e3)
    assert m.u.dims == ("y", "x") and m.u.shape == (Y.size, X.size)
    np.testing.assert_allclose(m.u, u, atol=0.6)
    np.testing.assert_allclose(m.v, v, atol=0.6)


def test_estimate_motion_tiled_varying():
    """Two storms moving differently: tiles resolve both motions."""
    x = np.arange(-100e3, 100e3 + 1, 1000.0)
    X2, Y2 = np.meshgrid(x, Y)
    storms = [(-60e3, -10e3, 10.0), (-50e3, 15e3, 10.0)]
    storms += [(50e3, -10e3, 20.0), (60e3, 15e3, 20.0)]

    def two(t):
        f = sum(
            40.0 * np.exp(-((X2 - x0 - u * t) ** 2 + (Y2 - y0) ** 2) / (2 * 5e3**2))
            for x0, y0, u in storms
        )
        time = xr.DataArray(T0 + np.timedelta64(t, "s"))
        return xr.Dataset({"DBZH": (("y", "x"), f)}, coords={"x": x, "y": Y}).assign(
            time=time
        )

    d0, d1 = two(0), two(300)
    m = estimate_motion(d0, d1, tile=60e3, smooth=0)
    assert float(m.u.sel(x=-70e3, y=0)) < float(m.u.sel(x=70e3, y=0)) - 5


def test_estimate_motion_no_signal_is_nan():
    rng = np.random.default_rng(1)
    d0 = _blobs(0, 0, 0).assign(DBZH=(("y", "x"), rng.uniform(0, 60, (Y.size, X.size))))
    d1 = d0.assign(DBZH=(("y", "x"), rng.uniform(0, 60, (Y.size, X.size))))
    d1["time"] = d0.time + np.timedelta64(300, "s")
    with pytest.warns(RuntimeWarning, match="no reliable motion"):
        m = estimate_motion(d0, d1)
    assert np.isnan(float(m.u))
    with pytest.raises(ValueError, match="motion"):
        interpolate_time(d0, d1, d0.time.values, motion=m)


def test_estimate_motion_errors():
    d0 = _blobs(0, 1, 1)
    with pytest.raises(ValueError, match="different times"):
        estimate_motion(d0, d0)
    with pytest.raises(ValueError, match="time"):
        estimate_motion(d0.drop_vars("time"), d0.drop_vars("time"))
    with pytest.raises(ValueError, match="reflectivity"):
        estimate_motion(d0.rename(DBZH="foo"), d0.rename(DBZH="foo"), dt=60)
    with pytest.raises(ValueError, match="evenly spaced"):
        bad = d0.assign_coords(x=X**2)
        estimate_motion(bad, bad, dt=60)


# ---------------------------------------------------------------------------
# advection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("method", ["linear", "cubic"])
def test_advect_integer_shift_is_exact(engine, method):
    d0 = _blobs(0, 0, 0)
    # 3 cells east, 2 cells south in 100 s
    out = advect(d0, 30.0, -20.0, dt=100.0, engine=engine, method=method)
    expected = d0.DBZH.shift(x=3, y=-2)
    xr.testing.assert_identical(out.DBZH, expected)
    assert out.time.values == T0 + np.timedelta64(100, "s")
    assert out.time.attrs == d0.time.attrs


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("method, tol", [("linear", 0.5), ("cubic", 0.15)])
def test_advect_matches_translated_truth(engine, method, tol):
    u, v = 13.7, -8.3
    d0, truth = _blobs(0, u, v, threshold=None), _blobs(300, u, v, threshold=None)
    out = advect(d0, u, v, dt=300, engine=engine, method=method)
    err = np.abs(_interior(out.DBZH, 10) - _interior(truth.DBZH, 10))
    assert float(err.max()) < tol
    assert out.DBZH.dtype == np.float32
    assert out.DBZH.attrs == d0.DBZH.attrs


def test_advect_roundtrip_and_time_argument():
    d0 = _blobs(0, 0, 0, threshold=None)
    target = T0 + np.timedelta64(200, "s")
    fwd = advect(d0, 7.3, 4.1, time=target, method="cubic")
    assert fwd.time.values == target
    back = advect(fwd, 7.3, 4.1, dt=-200.0, method="cubic")
    err = np.abs(_interior(back.DBZH, 6) - _interior(d0.DBZH, 6))
    assert float(err.max()) < 0.5  # two sub-cell cubic interpolations
    assert back.time.values == T0


@pytest.mark.parametrize("engine", ENGINES)
def test_advect_validity_mask(engine):
    """Missing data move with the field and do not spread."""
    d0 = _blobs(0, 0, 0, threshold=None)
    holed = d0.DBZH.values.copy()
    holed[40:50, 50:60] = np.nan
    d0 = d0.assign(DBZH=(("y", "x"), holed, d0.DBZH.attrs))
    out = advect(d0, 25.0, 0.0, dt=100.0, engine=engine)  # 2.5 cells east
    hole = out.DBZH.isnull().values
    # the hole moves 2.5 cells: columns 52..62 (half-cell edges decide)
    assert hole[40:50, 53:62].all()
    assert not hole[40:50, 62:].any() and not hole[40:50, 2:53].any()
    # cells entering from outside the grid are empty
    assert hole[:, :2].all() and not hole[:, 2].any()
    # min_weight=1 keeps only cells with all four neighbours valid
    strict = advect(d0, 25.0, 0.0, dt=100.0, engine=engine, min_weight=1.0)
    assert strict.DBZH.isnull().sum() > out.DBZH.isnull().sum()


@pytest.mark.parametrize("engine", ENGINES)
def test_advect_3d_dataset_keeps_structure(engine):
    z = np.array([500.0, 1000.0])
    d0 = _blobs(0, 0, 0, z=z)
    d0["other"] = ("z", np.array([1.0, 2.0]))
    d0 = d0.assign_coords(lat=("y", Y / 111e3))
    out = advect(d0.transpose("x", "z", "y"), 10.0, 10.0, dt=100.0, engine=engine)
    assert out.DBZH.dims == ("x", "z", "y")
    assert "lat" in out.coords and out.instrument_name == "synthetic"
    xr.testing.assert_identical(out["other"], d0["other"])
    xr.testing.assert_identical(
        out.DBZH.transpose("z", "y", "x"), d0.DBZH.shift(x=1, y=1)
    )


def test_advect_variable_motion_uniform_equals_scalar():
    d0 = _blobs(0, 0, 0)
    u = xr.full_like(d0.DBZH, 11.3, dtype="f8").fillna(11.3)
    v = xr.full_like(d0.DBZH, -6.1, dtype="f8").fillna(-6.1)
    a = advect(d0, u, v, dt=200.0)
    b = advect(d0, 11.3, -6.1, dt=200.0)
    xr.testing.assert_allclose(a.DBZH, b.DBZH, atol=1e-5)


def test_advect_variable_motion_departure_points():
    """Shear flow u = c * y: departure points follow the analytic trajectory."""
    d0 = _blobs(0, 0, 0)
    c = 2e-4  # 1/s
    u = (c * d0.y).broadcast_like(d0.DBZH).transpose("y", "x")
    zero = np.zeros_like(u.values)
    grid = (1000.0, 1000.0, Y.size, X.size)
    rows, cols = advection._departure(u.values, zero, [300.0], *grid, False, None)
    # u depends only on y and v = 0: the trajectory is a straight line
    expected = np.arange(X.size)[None, :] - c * Y[:, None] * 300.0 / 1000.0
    np.testing.assert_allclose(cols[0], expected, atol=1e-12)
    np.testing.assert_allclose(rows[0], np.arange(Y.size)[:, None] + 0 * cols[0])


@pytest.mark.parametrize("method", ["linear", "cubic"])
@pytest.mark.parametrize("dtype", ["f4", "f8"])
@needs_kernel
def test_engines_agree(method, dtype):
    """Compiled and NumPy engines agree to rounding (float32: 1e-5 dBZ)."""
    rng = np.random.default_rng(0)
    data = rng.uniform(-10, 60, (3, 37, 41)).astype(dtype)
    data[rng.random(data.shape) < 0.15] = np.nan
    src_r = np.arange(37.0)[None, :, None] + rng.uniform(-3, 3, (2, 37, 41))
    src_c = np.arange(41.0)[None, None, :] + rng.uniform(-3, 3, (2, 37, 41))
    src_r[0, 0, 0] = np.nan
    order = advection._ORDERS[method]
    a = advection._interpolate(data, src_r, src_c, order, 0.5, True, 3)
    b = advection._interpolate(data, src_r, src_c, order, 0.5, False, None)
    assert a.dtype == b.dtype == np.dtype(dtype) and a.shape == (2, 3, 37, 41)
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-5 if dtype == "f4" else 1e-12)


@needs_kernel
def test_kernel_threads_agree():
    rng = np.random.default_rng(3)
    data = rng.uniform(0, 1, (4, 64, 64)).astype("f4")
    r = np.arange(64.0)[None, :, None] + rng.uniform(-2, 2, (3, 64, 64))
    c = np.arange(64.0)[None, None, :] + rng.uniform(-2, 2, (3, 64, 64))
    one = advection._interpolate(data, r, c, 3, 0.5, True, 1)
    many = advection._interpolate(data, r, c, 3, 0.5, True, 8)
    np.testing.assert_array_equal(one, many)


def test_advect_errors():
    d0 = _blobs(0, 0, 0)
    with pytest.raises(ValueError, match="exactly one"):
        advect(d0, 1.0, 1.0)
    with pytest.raises(ValueError, match="exactly one"):
        advect(d0, 1.0, 1.0, dt=1.0, time=T0)
    with pytest.raises(ValueError, match="method"):
        advect(d0, 1.0, 1.0, dt=1.0, method="quintic")
    with pytest.raises(ValueError, match="engine"):
        advect(d0, 1.0, 1.0, dt=1.0, engine="fortran")
    with pytest.raises(ValueError, match="v is required"):
        advect(d0, 1.0, dt=1.0)
    with pytest.raises(ValueError, match="NaN"):
        advect(d0, np.nan, 1.0, dt=1.0)
    with pytest.raises(ValueError, match="time"):
        advect(d0.drop_vars("time"), 1.0, 1.0, time=T0)
    if not advection.HAS_COMPILED_KERNEL:
        with pytest.raises(ImportError):
            advect(d0, 1.0, 1.0, dt=1.0, engine="compiled")


# ---------------------------------------------------------------------------
# time interpolation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_interpolate_time(engine):
    u, v = 13.7, -8.3
    d0, d1 = _blobs(0, u, v, threshold=None), _blobs(300, u, v, threshold=None)
    times = T0 + np.arange(0, 301, 75).astype("timedelta64[s]")
    out = interpolate_time(d0, d1, times, engine=engine, method="cubic")
    assert out.DBZH.dims == ("time", "y", "x")
    np.testing.assert_array_equal(out.time.values, times)
    assert out.time.attrs == d0.time.attrs and out.DBZH.attrs == d0.DBZH.attrs
    for k, t in enumerate((0, 75, 150, 225, 300)):
        truth = _blobs(t, u, v, threshold=None).DBZH
        err = np.abs(_interior(out.DBZH.isel(time=k), 10) - _interior(truth, 10))
        assert float(err.max()) < 0.3, t  # peaks are 40-50 dBZ
    # plain linear interpolation in time is far worse at the midpoint
    linear = 0.5 * (d0.DBZH + d1.DBZH)
    truth = _blobs(150, u, v, threshold=None).DBZH
    assert float(np.abs(_interior(linear, 10) - _interior(truth, 10)).max()) > 5


def test_interpolate_time_dataarray_and_errors():
    u, v = 10.0, 0.0
    d0, d1 = _blobs(0, u, v), _blobs(300, u, v)
    motion = estimate_motion(d0, d1)
    da0 = d0.DBZH.assign_coords(time=d0.time)
    da1 = d1.DBZH.assign_coords(time=d1.time)
    out = interpolate_time(da0, da1, T0 + np.timedelta64(150, "s"), motion=motion)
    assert out.dims == ("time", "y", "x") and out.sizes["time"] == 1
    with pytest.raises(ValueError, match="between"):
        interpolate_time(d0, d1, T0 + np.timedelta64(400, "s"), motion=motion)
    with pytest.raises(ValueError, match="later"):
        interpolate_time(d1, d0, T0, motion=motion)


def test_accessors():
    u, v = 13.7, -8.3
    d0, d1 = _blobs(0, u, v), _blobs(300, u, v)
    motion = d0.radarx.estimate_motion(d1)
    assert abs(float(motion.u) - u) < 0.07
    xr.testing.assert_identical(
        d0.radarx.advect(motion, dt=60.0), advect(d0, motion, dt=60.0)
    )
    frames = d0.radarx.interpolate_time(d1, [T0], motion=motion)
    assert frames.sizes["time"] == 1
    da = d0.DBZH.assign_coords(time=d0.time)
    xr.testing.assert_identical(da.radarx.advect(u, v, 60.0), advect(da, u, v, 60.0))


# ---------------------------------------------------------------------------
# real data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nexrad_grid():
    xd = pytest.importorskip("xradar")
    from open_radar_data import DATASETS

    file = DATASETS.fetch("KLBB20160601_150025_V06")
    dtree = xd.io.open_nexradlevel2_datatree(file, sweep=[0, 1, 2, 3])
    return dtree.radarx.to_grid(
        ["DBZH"],
        x_lim=(-120e3, 120e3),
        y_lim=(-120e3, 120e3),
        z_lim=(500, 3000),
        x_step=1000,
        y_step=1000,
        z_step=500,
    )


def test_nexrad_known_translation(nexrad_grid):
    """A real volume translated by a known number of cells."""
    g0 = nexrad_grid
    dt = 360.0
    shift_x, shift_y = 4, -3  # cells (1 km) in 6 minutes
    moved = g0.DBZH.shift(x=shift_x, y=shift_y)
    g1 = g0.assign(DBZH=moved, time=g0.time + np.timedelta64(int(dt), "s"))
    m = estimate_motion(g0, g1)
    assert float(m.quality) > 0.9
    assert abs(float(m.u) - shift_x * 1000 / dt) < 0.05
    assert abs(float(m.v) - shift_y * 1000 / dt) < 0.05
    back = advect(g1, m, dt=-dt)
    inner = (slice(10, -10),) * 2
    a = back.DBZH.isel(x=inner[0], y=inner[1])
    b = g0.DBZH.isel(x=inner[0], y=inner[1])
    both = a.notnull() & b.notnull()
    assert float(np.abs(a - b).where(both).max()) < 1.0
    assert float(both.sum()) > 0.95 * float(b.notnull().sum())


# ---------------------------------------------------------------------------
# input handling and edge cases
# ---------------------------------------------------------------------------


def test_time_step_types():
    import pandas as pd

    d0, d1 = _blobs(0, 10.0, 0.0), _blobs(300, 10.0, 0.0)
    ref = advect(d0, 10.0, 0.0, dt=120.0)
    for dt in (
        np.timedelta64(120, "s"),
        pd.Timedelta(seconds=120),
        xr.DataArray(np.timedelta64(120, "s")),
    ):
        xr.testing.assert_identical(advect(d0, 10.0, 0.0, dt=dt), ref)
    m = estimate_motion(d0, d1, dt=pd.Timedelta(seconds=300))
    assert abs(float(m.u) - 10.0) < 0.07


def test_non_datetime_time_is_left_alone():
    d0 = _blobs(0, 0, 0).assign(time=xr.DataArray(5.0))
    out = advect(d0, 10.0, 0.0, dt=100.0)
    assert float(out.time) == 5.0


def test_grid_errors():
    d0, d1 = _blobs(0, 1, 1), _blobs(300, 1, 1)
    motion = xr.Dataset({"u": 1.0, "v": 1.0})
    with pytest.raises(ValueError, match="'x' coordinate"):
        advect(d0.drop_vars("x"), 1.0, 1.0, dt=1.0)
    with pytest.raises(ValueError, match="at least 2 points"):
        advect(d0.isel(x=[0]), 1.0, 1.0, dt=1.0)
    with pytest.raises(ValueError, match="dimensions"):
        off_grid = d0.DBZH.rename(x="col").assign_coords(x=("col", X))
        estimate_motion(off_grid, off_grid, dt=60)
    with pytest.raises(ValueError, match="same grid"):
        estimate_motion(d0, d1.isel(x=slice(1, None)))
    with pytest.raises(ValueError, match="same grid"):
        interpolate_time(d0, d1.isel(x=slice(1, None)), T0, motion=motion)
    with pytest.raises(ValueError, match="lacks"):
        interpolate_time(d0, d1.rename(DBZH="other"), T0, motion=motion)
    with pytest.raises(ValueError, match="scalar 'time'"):
        interpolate_time(d0.drop_vars("time"), d1, T0, motion=motion)


def test_motion_argument_errors():
    d0 = _blobs(0, 0, 0)
    motion = xr.Dataset({"u": 1.0, "v": 1.0})
    with pytest.raises(ValueError, match="either"):
        advect(d0, motion, 1.0, dt=1.0)
    bad = xr.DataArray(np.ones((3, 3)), dims=("y", "x"))
    with pytest.raises(ValueError, match="spatially varying"):
        advect(d0, bad, bad, dt=1.0)


def test_estimate_motion_observed_mask_and_no_highpass():
    u, v = 13.7, -8.3
    d0, d1 = _blobs(0, u, v), _blobs(300, u, v)
    observed = xr.ones_like(d0.DBZH, dtype=bool)
    m = estimate_motion(d0, d1, observed=observed, highpass=None)
    assert abs(float(m.u) - u) < 0.1 and abs(float(m.v) - v) < 0.1


def test_estimate_motion_empty_and_too_fast():
    d0 = _blobs(0, 0, 0)
    empty = d0.assign(DBZH=xr.full_like(d0.DBZH, np.nan))
    later = empty.assign(time=d0.time + np.timedelta64(60, "s"))
    with pytest.warns(RuntimeWarning, match="no reliable motion"):
        m = estimate_motion(empty, later)
    assert np.isnan(float(m.u)) and float(m.quality) == 0.0
    # faster than max_speed: the peak lies on the edge of the search window,
    # and without a domain-wide first guess there is no tiled motion either
    d1 = _blobs(300, 30.0, 0.0)
    with pytest.warns(RuntimeWarning, match="no reliable motion"):
        m = estimate_motion(d0, d1, max_speed=10.0, tile=40e3)
    assert m.u.dims == ("y", "x") and bool(m.u.isnull().all())


def test_track_keeps_estimate_when_refinement_fails():
    a = np.zeros((40, 40))
    a[18:22, 10:14] = 30.0
    b = np.zeros((40, 40))
    b[18:22, 13:17] = 30.0
    mask = np.ones(a.shape, dtype=bool)
    whole = slice(0, 40)

    def no_echo(*args, **kwargs):  # an interpolation that loses all echo
        return np.zeros((1, 1, 40, 40))

    dy, dx, q = advection._track(
        a, b, mask, 5.0, None, (8, 8), whole, whole, 3, no_echo
    )
    assert dy == pytest.approx(0.0, abs=0.2) and dx == pytest.approx(3.0, abs=0.2)
    # no echo at all: NaN straight away
    dy, dx, q = advection._track(
        a * 0, b, mask, 5.0, None, (8, 8), whole, whole, 3, no_echo
    )
    assert np.isnan(dy) and q == 0.0


def test_use_compiled_without_kernel(monkeypatch):
    monkeypatch.setattr(advection, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError):
        advection._use_compiled("compiled")
    assert advection._use_compiled("auto") is False
