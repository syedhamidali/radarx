# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the optical-flow motion estimates
===========================================

Synthetic echo textures are moved by a known uniform or sheared displacement,
so the Lucas-Kanade and DARTS estimates have an analytic truth. The compiled
kernels are compared with the NumPy implementations.
"""

import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import gaussian_filter, map_coordinates

import radarx  # noqa: F401
from radarx.retrieve import motion
from radarx.retrieve._optical_flow_numpy import (
    cosine_basis,
    darts_normal,
    darts_normal_numpy,
    lk_solve,
    lk_solve_numpy,
)
from radarx.retrieve.advection import advect, estimate_motion
from radarx.retrieve.motion import optical_flow

ENGINES = ["numpy"] + (["compiled"] if motion.HAS_COMPILED_KERNEL else [])
needs_kernel = pytest.mark.skipif(
    not motion.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)
T0 = np.datetime64("2026-01-01T00:00:00", "ns")
N = 80
DX = 1000.0
DT = 300.0


def _texture(seed=0, size=260):
    rng = np.random.default_rng(seed)
    big = gaussian_filter(rng.standard_normal((size, size)), 5, mode="wrap")
    big += 0.5 * gaussian_filter(rng.standard_normal((size, size)), 2, mode="wrap")
    return 30.0 + 12.0 * (big - big.mean()) / big.std()


BIG = _texture()


def _sample(rows, cols):
    """The texture at fractional positions, NaN below 10 dBZ."""
    f = map_coordinates(BIG, [rows, cols], order=3, mode="wrap")
    return np.where(f >= 10.0, f, np.nan)


def _frames(shift_x, shift_y, count=3, shear=0.0):
    """Frames of the pattern moved by (shift_x, shift_y) cells per step.

    ``shear`` adds ``shear * (row - N / 2)`` cells per step to the column shift.
    """
    rows, cols = np.mgrid[0:N, 0:N].astype(float)
    out = []
    for k in range(count):
        sx = k * (shift_x + shear * (rows - N / 2))
        f = _sample(rows - k * shift_y + 90, cols - sx + 90)
        ds = xr.Dataset(
            {"DBZH": (("y", "x"), f.astype("f4"), {"units": "dBZ"})},
            coords={
                "x": ("x", np.arange(N) * DX, {"units": "m"}),
                "y": ("y", np.arange(N) * DX, {"units": "m"}),
            },
        )
        out.append(ds.assign(time=xr.DataArray(T0 + np.timedelta64(int(DT * k), "s"))))
    return out


def _speed(cells):
    return cells * DX / DT


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("method", ["lucas_kanade", "darts"])
@pytest.mark.parametrize("cells", [(1.3, -0.7), (-2.4, 1.1), (0.0, 0.0), (3.6, 2.9)])
def test_uniform_translation(method, engine, cells):
    sx, sy = cells
    out = optical_flow(_frames(sx, sy), method=method, uniform=True, engine=engine)
    assert out.u.dims == () and out.v.dims == ()
    assert abs(float(out.u) - _speed(sx)) < 0.15
    assert abs(float(out.v) - _speed(sy)) < 0.15
    assert float(out.quality) > 0.85
    assert out.u.attrs["units"] == "m s-1"
    assert out.attrs["dt"] == DT and out.attrs["flow_method"] == method


@pytest.mark.parametrize("method", ["lucas_kanade", "darts"])
def test_dense_field_of_a_uniform_translation(method):
    out = optical_flow(_frames(1.5, -1.0), method=method)
    assert out.u.dims == ("y", "x")
    inner = (slice(10, -10),) * 2
    assert abs(float(out.u[inner].mean()) - _speed(1.5)) < 0.15
    assert abs(float(out.v[inner].mean()) - _speed(-1.0)) < 0.15
    assert float(out.u[inner].std()) < 0.8
    np.testing.assert_array_equal(out["x"].values, np.arange(N) * DX)


def test_agrees_with_estimate_motion():
    frames = _frames(2.0, 1.0)
    fft = estimate_motion(frames[1], frames[2])
    for method in ("lucas_kanade", "darts"):
        flow = optical_flow(frames, method=method, uniform=True)
        assert abs(float(flow.u) - float(fft.u)) < 0.15
        assert abs(float(flow.v) - float(fft.v)) < 0.15


def test_large_displacement_needs_the_pyramid():
    frames = _frames(7.0, -5.0)  # 23 and 17 m/s
    one = optical_flow(frames, uniform=True, n_levels=1)
    many = optical_flow(frames, uniform=True)
    err_one = abs(float(one.u) - _speed(7.0)) + abs(float(one.v) - _speed(-5.0))
    err_many = abs(float(many.u) - _speed(7.0)) + abs(float(many.v) - _speed(-5.0))
    assert err_many < 0.3
    assert err_many < err_one


@pytest.mark.parametrize("method", ["lucas_kanade", "darts"])
def test_sheared_motion_is_followed(method):
    # u grows by 0.8 cells per step over the 80 rows
    frames = _frames(0.0, 0.0, shear=0.01)
    kw = {"order": 3, "smooth": 12e3} if method == "darts" else {"smooth": 12e3}
    out = optical_flow(frames, method=method, **kw)
    rows = np.arange(N)
    truth = _speed(0.01 * (rows - N / 2))
    mean_u = out.u.isel(x=slice(10, -10)).mean("x").values
    inner = slice(12, -12)
    corr = np.corrcoef(mean_u[inner], truth[inner])[0, 1]
    assert corr > 0.9
    slope = np.polyfit(rows[inner], mean_u[inner], 1)[0]
    true_slope = np.polyfit(rows[inner], truth[inner], 1)[0]
    assert abs(slope - true_slope) < 0.5 * abs(true_slope)


def test_result_is_used_by_advect():
    frames = _frames(2.0, 1.0)
    flow = optical_flow(frames)
    back = advect(frames[2], flow, dt=-DT)
    a, b = back["DBZH"].values[15:-15, 15:-15], frames[1]["DBZH"].values[15:-15, 15:-15]
    both = np.isfinite(a) & np.isfinite(b)
    assert both.mean() > 0.4
    assert np.abs(a - b)[both].mean() < 2.0


def test_more_than_two_frames_are_used():
    frames = _frames(1.2, 0.6, count=5)
    out = optical_flow(frames, uniform=True)
    assert abs(float(out.u) - _speed(1.2)) < 0.1
    pair = optical_flow(frames[:2], uniform=True)
    assert abs(float(pair.u) - _speed(1.2)) < 0.2


def test_inputs_dataarray_time_dimension_and_accessor():
    frames = _frames(1.0, 1.0)
    ref = optical_flow(frames, uniform=True)
    arrays = [f["DBZH"].assign_coords(time=f["time"]) for f in frames]
    from_da = optical_flow(arrays, uniform=True)
    np.testing.assert_allclose(float(from_da.u), float(ref.u))
    stacked = xr.concat(arrays, dim="time")
    np.testing.assert_allclose(
        float(optical_flow(stacked, uniform=True).u), float(ref.u)
    )
    via = stacked.radarx.optical_flow(uniform=True)
    np.testing.assert_allclose(float(via.u), float(ref.u))
    bare = [f.drop_vars("time") for f in frames]
    np.testing.assert_allclose(
        float(optical_flow(bare, dt=DT, uniform=True).u), float(ref.u)
    )


def test_texture_less_direction():
    """Stripes along x: only the motion across them (v) is defined."""
    rows, cols = np.mgrid[0:N, 0:N].astype(float)
    frames = []
    for k in range(3):
        f = 30.0 + 15.0 * np.sin(2 * np.pi * (rows - 1.0 * k) / 20.0)
        ds = xr.Dataset(
            {"DBZH": (("y", "x"), f)},
            coords={"x": np.arange(N) * DX, "y": np.arange(N) * DX},
        )
        frames.append(ds.assign(time=T0 + np.timedelta64(int(DT * k), "s")))
    # the global fit finds the defined component and leaves the other at zero
    out = optical_flow(frames, method="darts", order=0, n_levels=1)
    assert abs(float(out.v) - _speed(1.0)) < 0.3
    assert abs(float(out.u)) < 0.3
    # the local fit trusts no cell (aperture problem): zero motion, no failure
    local = optical_flow(frames, uniform=True, n_levels=1)
    assert float(local.u) == 0.0 and float(local.v) == 0.0


def test_no_echo_warns_and_gives_zero():
    frames = [f.assign(DBZH=f.DBZH * np.nan) for f in _frames(1.0, 1.0)]
    with pytest.warns(RuntimeWarning, match="no echo"):
        out = optical_flow(frames, uniform=True)
    assert float(out.u) == 0.0 and float(out.v) == 0.0 and float(out.quality) == 0.0
    with pytest.warns(RuntimeWarning, match="no echo"):
        out = optical_flow(frames, method="darts")
    assert float(out.u.mean()) == 0.0


def test_pyramid_helpers():
    a = np.arange(36.0).reshape(1, 6, 6)
    pooled = motion._pool(a)
    assert pooled.shape == (1, 3, 3) and pooled[0, 0, 0] == a[0, :2, :2].mean()
    assert motion._pool(np.zeros((2, 7, 9))).shape == (2, 3, 4)
    up = motion._upsample(np.ones((2, 3, 3)), (7, 7))
    assert up.shape == (2, 7, 7) and np.all(up == 2.0)
    taps = motion._taps(2.0)
    assert taps.sum() == pytest.approx(1.0) and len(taps) % 2 == 1
    assert motion._levels(60.0, (80, 80), None) <= 3
    assert motion._levels(1.0, (80, 80), None) == 1
    assert motion._levels(1000.0, (40, 40), None) == 2  # limited by the grid
    assert motion._levels(60.0, (80, 80), 2) == 2


@needs_kernel
def test_lk_kernel_matches_numpy_and_checks_input():
    from radarx.retrieve import _optical_flow as k

    rng = np.random.default_rng(0)
    ix, iy, it = (rng.standard_normal((3, 30, 40)) for _ in range(3))
    taps = motion._taps(2.0)
    a = lk_solve(ix, iy, it, taps, True, 3)
    b = lk_solve_numpy(ix, iy, it, taps)
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-10)
    # a singular tensor gives no increment
    flat = np.zeros((1, 10, 10))
    out = lk_solve(flat, flat, flat + 1.0, motion._taps(1.0), True, 1)
    assert np.all(out[:2] == 0.0)
    with pytest.raises(ValueError, match="3-D"):
        k.lk_solve(ix[0], iy[0], it[0], taps, 1)
    with pytest.raises(ValueError, match="differ in shape"):
        k.lk_solve(ix, iy[:, :5], it, taps, 1)
    with pytest.raises(ValueError, match="odd length"):
        k.lk_solve(ix, iy, it, np.ones(4) / 4, 1)


@needs_kernel
@pytest.mark.parametrize("order", [0, 1, 3])
def test_darts_kernel_matches_numpy_and_checks_input(order):
    from radarx.retrieve import _optical_flow as k

    rng = np.random.default_rng(order)
    ix, iy, it = (rng.standard_normal((2, 20, 25)) for _ in range(3))
    ix[0, :5] = 0.0
    iy[0, :5] = 0.0
    a = darts_normal(ix, iy, it, order, True, 4)
    b = darts_normal_numpy(ix, iy, it, order)
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)
    assert a.shape == (2 * (order + 1) ** 2, 2 * (order + 1) ** 2 + 1)
    np.testing.assert_allclose(a[:, :-1], a[:, :-1].T)
    with pytest.raises(ValueError, match="negative"):
        k.darts_normal(ix, iy, it, -1, 1)
    with pytest.raises(ValueError, match="3-D"):
        k.darts_normal(ix[0], iy[0], it[0], 1, 1)


def test_cosine_basis():
    basis = cosine_basis(2, 10, 12)
    assert basis.shape == (9, 10, 12)
    assert np.all(basis[0] == 1.0)
    # orthogonal over the grid
    flat = basis.reshape(9, -1)
    gram = flat @ flat.T
    np.testing.assert_allclose(gram - np.diag(np.diag(gram)), 0.0, atol=1e-9)


def test_darts_order_zero_is_one_vector():
    out = optical_flow(_frames(1.0, 2.0), method="darts", order=0)
    assert out.u.dims == ()
    assert abs(float(out.v) - _speed(2.0)) < 0.2


def test_corrections_without_gradient_are_zero():
    zeros = np.zeros((1, 12, 12))
    out = motion._darts_increment(zeros, zeros, zeros, 1, 1e-3, False, 1)
    assert np.all(out == 0.0)
    out = motion._lk_increment(
        zeros, zeros, zeros, motion._taps(1.0), 2.0, 0.01, False, 1
    )
    assert np.all(out == 0.0)


def test_options_are_checked():
    frames = _frames(1.0, 1.0)
    for kwargs, message in [
        ({"method": "farneback"}, "method must be"),
        ({"order": -1}, "order"),
        ({"order": 1.5}, "order"),
        ({"iterations": 0}, "iterations"),
        ({"n_levels": 0}, "n_levels"),
        ({"window": 0.0}, "window must be positive"),
        ({"smooth": None}, "smooth must be positive"),
        ({"prefilter": -1.0}, "prefilter must be positive"),
        ({"engine": "gpu"}, "engine must be"),
    ]:
        with pytest.raises(ValueError, match=message):
            optical_flow(frames, **kwargs)
    with pytest.raises(ValueError, match="at least two"):
        optical_flow(frames[:1])
    if not motion.HAS_COMPILED_KERNEL:  # pragma: no cover
        with pytest.raises(ImportError):
            optical_flow(frames, engine="compiled")


@needs_kernel
def test_engines_agree_on_the_whole_estimate():
    frames = _frames(1.7, -0.9)
    for method in ("lucas_kanade", "darts"):
        a = optical_flow(frames, method=method, engine="compiled")
        b = optical_flow(frames, method=method, engine="numpy")
        np.testing.assert_allclose(a.u.values, b.u.values, atol=1e-6)
        np.testing.assert_allclose(a.v.values, b.v.values, atol=1e-6)


def test_provenance():
    out = optical_flow(_frames(1.0, 1.0), uniform=True)
    assert "10.1175/2011JTECHA1496.1" in out.attrs["radarx_references"]
    assert "10.1109/CVPR.1994.323794" in out.attrs["radarx_references"]


def test_input_is_not_changed():
    frames = _frames(1.0, 1.0)
    before = [f.copy(deep=True) for f in frames]
    optical_flow(frames)
    for a, b in zip(frames, before):
        xr.testing.assert_identical(a, b)


def test_frames_with_different_intervals():
    """Pairs 280 s and 320 s apart share one motion."""
    times = [0.0, 280.0, 600.0]
    rows, cols = np.mgrid[0:N, 0:N].astype(float)
    frames = []
    for t in times:
        f = _sample(rows - 0.6 * t / DT + 90, cols - 2.0 * t / DT + 90)
        ds = xr.Dataset(
            {"DBZH": (("y", "x"), f.astype("f4"))},
            coords={"x": np.arange(N) * DX, "y": np.arange(N) * DX},
        )
        frames.append(ds.assign(time=T0 + np.timedelta64(int(t), "s")))
    out = optical_flow(frames, uniform=True)
    # per mean interval of 300 s
    assert abs(float(out.u) - _speed(2.0)) < 0.15
    assert abs(float(out.v) - _speed(0.6)) < 0.15
