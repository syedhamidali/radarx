#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for the UNRAVEL dealiasing method (``dealias_velocity(method="unravel")``)."""

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401  registers the accessors
from radarx.retrieve import _unravel_numpy as spec
from radarx.retrieve import dealias, dealias_velocity

from .test_dealias import synthetic_sweep, volume

needs_kernel = pytest.mark.skipif(
    not dealias.HAS_UNRAVEL_KERNEL, reason="compiled UNRAVEL kernel not built"
)
ENGINES = ["numpy", pytest.param("compiled", marks=needs_kernel)]
D = spec.DEFAULTS


def geometry(az, rng):
    x = rng[None, :] * np.sin(np.radians(az))[:, None]
    y = rng[None, :] * np.cos(np.radians(az))[:, None]
    return x, y


def run_spec(vel, vn, az, rng, wrap=True, **params):
    x, y = geometry(az, rng)
    return spec.unravel_sweep(vel, vn, az, rng, x, y, wrap, params)


def run_kernel(vel, vn, az, rng, wrap=True):
    x, y = geometry(az, rng)
    ((out, flag),) = dealias._unravel.unravel(
        [vel],
        [vn],
        [az],
        [rng],
        [x],
        [y],
        [wrap],
        D["alpha"],
        D["alpha_first"],
        D["alpha_pre"],
        D["z_cutoff"],
        D["pre_window"],
        D["beta"],
        list(D["windows"]),
        [tuple(b) for b in D["boxes"]],
        D["closest_half"],
        tuple(D["check_box"]),
        0,
    )
    return out, flag


def uniform_wind(nray=72, ngate=50, speed=17.0, nyquist=10.0, direction=40.0):
    """A uniform wind seen by the radar, folded; a clean test of the whole driver."""
    az = np.arange(nray) * 360.0 / nray + 2.5
    rng = 1000.0 + 500.0 * np.arange(ngate)
    truth = (
        speed * np.cos(np.radians(az - direction))[:, None] * np.ones(ngate)[None, :]
    )
    folded = (truth + nyquist) % (2 * nyquist) - nyquist
    return folded, truth, az, rng


# --------------------------------------------------------------------------
# the building blocks
# --------------------------------------------------------------------------


def test_fold_and_accept():
    assert spec.fold_to(24.0, -5.0, 10.0) == 15.0
    assert spec.fold_to(0.0, 12.0, 10.0) == -8.0
    assert spec.fold_to(3.0, 3.5, 10.0) == 3.5
    assert (
        spec.accept(25.0, -5.0, 10.0, 0.8) == 15.0
        or spec.accept(25.0, -5.0, 10.0, 0.8) is None
    )
    assert spec.accept(14.0, -5.0, 10.0, 0.8) == 15.0
    assert spec.accept(14.0, -5.0, 10.0, 0.05) is None
    assert spec.median([3.0, 1.0, 2.0]) == 2.0
    assert spec.median([4.0, 1.0, 2.0, 3.0]) == 2.5
    assert spec.mean([1.0, 2.0, 6.0]) == 3.0


def test_preprocessing_unfolds_a_spike_and_drops_noise():
    vel = np.full((4, 30), 5.0)
    vel[1, 10] = 5.0 - 20.0  # 20 = 2 Vn below its neighbours
    vel[2, 5] = 5.0 + 3.0 * 2.0 * 10.0 + 4.0  # far from any fold
    out = spec.preprocess(vel, 10.0, 0.3, 2.0, 10)
    assert out[1, 10] == pytest.approx(5.0)
    assert np.isnan(out[2, 5])
    assert (out[0] == 5.0).all()
    # a constant ray has zero deviation: nothing changes (MAD = 0)
    assert np.array_equal(
        spec.preprocess(np.full((2, 20), 7.0), 10.0, 0.3, 2.0, 10),
        np.full((2, 20), 7.0),
    )
    # all-missing rays are left alone
    nan = np.full((2, 12), np.nan)
    assert np.isnan(spec.preprocess(nan, 10.0, 0.3, 2.0, 10)).all()


def test_reference_radials_sit_where_the_wind_is_perpendicular():
    folded, truth, az, rng = uniform_wind(speed=8.0, nyquist=10.0, direction=0.0)
    r1, r2 = spec.reference_radials(folded, az, 10.0, 0.6, True)
    # the mean |v| is lowest at 90 and 270 degrees
    assert {round(az[r1] / 90.0), round(az[r2] / 90.0)} == {1, 3}
    # one separated by at least 120 degrees from the other
    assert spec.circ_diff(az[r1], az[r2], True) >= 120.0
    # an empty sweep and a single ray with data
    assert spec.reference_radials(np.full((5, 5), np.nan), az[:5], 10.0, 0.6, True) == (
        0,
        0,
    )
    one = np.full((6, 8), np.nan)
    one[3] = 1.0
    assert spec.reference_radials(one, np.arange(6) * 60.0, 10.0, 0.6, True) == (3, 3)
    # a sector: the references are minima 120 degrees apart or the same ray
    sector = np.abs(np.linspace(-10, 10, 40))[:, None] * np.ones((40, 6))
    a, b = spec.reference_radials(sector, np.linspace(0.0, 39.0, 40), 20.0, 0.6, False)
    assert a == b


def test_reference_radials_fall_back_without_quiet_rays():
    # every ray holds gates near the Nyquist velocity: the condition is dropped
    vel = np.full((12, 20), 9.0)
    vel[0] = 7.0
    r1, _ = spec.reference_radials(vel, np.arange(12) * 30.0, 10.0, 0.6, True)
    assert r1 == 0
    # one ray with many gates, all near the Nyquist velocity, and rays with a
    # single gate: the condition on the number of gates is dropped
    sparse = np.full((12, 30), np.nan)
    sparse[:, 0] = 1.0
    sparse[2, 0] = 0.1
    sparse[5, :] = 9.0
    assert spec.reference_radials(sparse, np.arange(12) * 30.0, 10.0, 0.6, True)[0] == 2


def test_sectors_cover_every_ray_once_in_a_clean_field():
    """Without aliasing every gate is accepted in the first pass, in any sector layout."""
    vel = np.ones((40, 20)) * 2.0
    for ref1, ref2 in [(3, 23), (3, 3), (30, 5), (0, 39)]:
        out, flag = spec.init_flags(vel, ref1, ref2, 10.0, 0.6, 0.6, True)
        spec.first_pass(vel, out, flag, ref1, ref2, 10.0, 0.6, True)
        assert (flag > 0).all(), (ref1, ref2)
    for ref1, ref2 in [(3, 23), (3, 3), (30, 5), (0, 39)]:
        out, flag = spec.init_flags(vel, ref1, ref2, 10.0, 0.6, 0.6, False)
        spec.first_pass(vel, out, flag, ref1, ref2, 10.0, 0.6, False)
        assert (flag > 0).all(), (ref1, ref2)


def test_modules_unfold_a_patch_from_its_surroundings():
    base = np.full((30, 40), 4.0)
    patch = base.copy()
    patch[10:13, 15:20] -= 20.0  # folded once for Vn = 10
    for name, call in {
        "range": lambda v, o, f: spec.range_pass(v, o, f, 10.0, 0.8, 6),
        "azimuth": lambda v, o, f: spec.azimuth_pass(v, o, f, 0, 10.0, 0.8, 6, True),
        "box": lambda v, o, f: spec.box_pass(v, o, f, 10.0, 0.8, (9, 9), True),
    }.items():
        out = patch.copy()
        flag = np.zeros(patch.shape, dtype=np.int8)
        flag[patch == 4.0] = 1
        # the unprocessed gates are the patch; the others are the references
        call(patch, out, flag)
        assert (flag > 0).all(), name
        assert np.allclose(out, 4.0), name
        assert (flag[10:13, 15:20] == 2).all(), name


def test_closest_reference_reaches_an_island():
    vel = np.full((24, 30), 3.0)
    vel[5:8, 25:28] = np.nan
    vel[5:8, 10:13] = 3.0 + 20.0 - 40.0  # an aliased island by itself
    az = np.arange(24) * 15.0
    rng = 1000.0 + 500.0 * np.arange(30)
    x, y = geometry(az, rng)
    out = vel.copy()
    flag = np.where(np.isnan(vel), -3, 1).astype(np.int8)
    flag[5:8, 10:13] = 0
    spec.closest_pass(vel, out, flag, x, y, 10.0, 2, True)
    assert (flag[5:8, 10:13] > 0).all()
    assert np.allclose(out[5:8, 10:13], 3.0)
    # nothing is processed: no reference, nothing happens
    none = np.zeros((4, 4), dtype=np.int8)
    spec.closest_pass(
        np.ones((4, 4)),
        np.ones((4, 4)),
        none,
        np.ones((4, 4)),
        np.ones((4, 4)),
        10.0,
        2,
        True,
    )
    assert (none == 0).all()


def test_checks_correct_a_wrong_fold():
    vel = np.tile(np.linspace(0.0, 6.0, 40), (20, 1))
    out = vel.copy()
    out[7, 20] += 20.0  # wrong by one fold for Vn = 10
    flag = np.ones(vel.shape, dtype=np.int8)
    rng = np.linspace(1000.0, 20000.0, 40)
    fixed = out.copy()
    spec.radial_check(fixed, flag, rng, 10.0, 0.8)
    assert fixed[7, 20] == pytest.approx(vel[7, 20], abs=1e-6)
    boxed = out.copy()
    spec.box_check(boxed, flag, 10.0, 0.8, (9, 9), True)
    assert boxed[7, 20] == pytest.approx(vel[7, 20], abs=1e-6)
    # a ray with too few gates, a degenerate range axis, nothing to compare with
    short = out.copy()
    spec.radial_check(short, flag, rng, 10.0, 0.8, min_gates=100)
    assert np.array_equal(short, out)
    same_range = out.copy()
    spec.radial_check(same_range, flag, np.ones(40), 10.0, 0.8)
    assert np.array_equal(same_range, out)
    alone = np.array([[5.0]])
    spec.box_check(alone, np.ones((1, 1), dtype=np.int8), 10.0, 0.8, (3, 3), False)
    assert alone[0, 0] == 5.0


def test_check_3d_uses_the_sweep_below():
    below = np.full((12, 20), 4.0)
    above = np.full((12, 20), 4.0)
    above[3:6, 5:9] += 20.0
    flag = np.ones(above.shape, dtype=np.int8)
    iray = np.broadcast_to(np.arange(12)[:, None], above.shape).astype(np.int32)
    igate = np.broadcast_to(np.arange(20)[None, :], above.shape).astype(np.int32)
    new = above.copy()
    spec.check_3d(new, flag, below, flag, iray, igate, 10.0, 0.8, (3, 2), True)
    assert np.allclose(new, 4.0)
    # gates with no gate below are left alone
    none = above.copy()
    spec.check_3d(none, flag, below, flag, -iray - 1, igate, 10.0, 0.8, (3, 2), True)
    assert np.array_equal(none, above)
    # a sector (no wrap) with the patch at its edge, and a lower sweep without data there
    edge = above.copy()
    spec.check_3d(edge, flag, below, flag, iray, igate, 10.0, 0.8, (3, 2), False)
    assert np.allclose(edge, 4.0)
    empty = above.copy()
    spec.check_3d(empty, flag, below, 0 * flag, iray, igate, 10.0, 0.8, (3, 2), False)
    assert np.array_equal(empty, above)


# --------------------------------------------------------------------------
# the driver and the kernel
# --------------------------------------------------------------------------


def test_driver_unfolds_a_uniform_wind():
    folded, truth, az, rng = uniform_wind()
    out, flag = run_spec(folded, 10.0, az, rng)
    assert (flag > 0).all()
    assert np.allclose(out, truth, atol=1e-9)
    assert (flag == 2).sum() == (np.abs(truth) > 10).sum()
    # a sector with the same wind
    out, flag = run_spec(folded[:30], 10.0, az[:30], rng, wrap=False)
    assert (flag > 0).all() and np.allclose(out, truth[:30], atol=1e-9)


@needs_kernel
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(nyquist=10.0, noise=1.0, gaps=0.05, nray=72, ngate=60),
        dict(nyquist=8.0, noise=2.0, gaps=0.3, nray=48, ngate=40, first_azimuth=200.0),
        dict(nyquist=30.0, noise=1.0, gaps=0.0, nray=36, ngate=30),
    ],
)
@pytest.mark.parametrize("wrap", [True, False])
def test_kernel_is_identical_to_the_specification(kwargs, wrap):
    ds, _ = synthetic_sweep(**kwargs)
    vel = ds["VRADH"].values.astype(float)
    az, rng = ds.azimuth.values, ds.range.values
    if not wrap:
        vel, az = vel[:30], az[:30]
    ref, ref_flag = run_spec(vel, kwargs["nyquist"], az, rng, wrap)
    out, flag = run_kernel(vel, kwargs["nyquist"], az, rng, wrap)
    assert np.array_equal(flag, ref_flag)
    np.testing.assert_array_equal(out, ref)


@needs_kernel
def test_kernel_inputs_are_checked():
    k = dealias._unravel
    vel = np.zeros((6, 5))
    az, rng = np.arange(6) * 60.0, np.arange(5) * 500.0 + 500.0
    x, y = geometry(az, rng)
    args = (
        D["alpha"],
        D["alpha_first"],
        D["alpha_pre"],
        D["z_cutoff"],
        D["pre_window"],
        D["beta"],
        list(D["windows"]),
        [tuple(b) for b in D["boxes"]],
        D["closest_half"],
        tuple(D["check_box"]),
        1,
    )
    with pytest.raises(ValueError, match="one array of each kind"):
        k.unravel([vel], [10.0, 10.0], [az], [rng], [x], [y], [True], *args)
    with pytest.raises(ValueError, match="2-D"):
        k.unravel([vel[0]], [10.0], [az], [rng], [x], [y], [True], *args)
    with pytest.raises(ValueError, match="do not match"):
        k.unravel([vel], [10.0], [az[:3]], [rng], [x], [y], [True], *args)
    with pytest.raises(ValueError, match="positive"):
        k.unravel([vel], [0.0], [az], [rng], [x], [y], [True], *args)
    with pytest.raises(ValueError, match="must match"):
        k.check_3d(
            vel,
            np.ones((6, 5), np.int8),
            vel,
            np.ones((2, 2), np.int8),
            np.zeros((6, 5), np.int32),
            np.zeros((6, 5), np.int32),
            10.0,
            0.8,
            3,
            2,
            True,
        )
    with pytest.raises(ValueError, match="2-D"):
        k.check_3d(
            vel[0],
            np.ones(5, np.int8),
            vel,
            np.ones((6, 5), np.int8),
            np.zeros(5, np.int32),
            np.zeros(5, np.int32),
            10.0,
            0.8,
            3,
            2,
            True,
        )


@needs_kernel
def test_threads_do_not_change_the_result():
    tree, _ = volume(elevations=(0.5, 1.5, 3.0), nray=36, ngate=40)
    a = dealias_velocity(tree, "VRADH", method="unravel", n_threads=1)
    b = dealias_velocity(tree, "VRADH", method="unravel", n_threads=4)
    xr.testing.assert_identical(a, b)


# --------------------------------------------------------------------------
# dealias_velocity(method="unravel")
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sweep_and_truth():
    return synthetic_sweep(nyquist=10.0, noise=1.0, gaps=0.05, nray=72, ngate=60)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_sweep_is_dealiased(sweep_and_truth, engine):
    ds, truth = sweep_and_truth
    out = dealias_velocity(ds, "VRADH", method="unravel", engine=engine)
    assert isinstance(out, xr.DataArray) and out.name == "VRADH_dealiased"
    assert out.attrs["dealias_anchor"] == "unravel"
    assert "UNRAVEL" in out.attrs["comment"]
    assert out.attrs["nyquist_velocity"] == 10.0
    ok = np.isfinite(out.values)
    assert 0.8 < ok.mean() <= np.isfinite(ds.VRADH.values).mean()
    assert np.mean(np.abs(out.values - truth)[ok] < 10.0) > 0.99
    # the output is the input shifted by whole intervals, where it has a value
    shift = (out.values - ds.VRADH.values)[ok] / 20.0
    np.testing.assert_allclose(shift, np.round(shift), atol=1e-6)


def test_engines_agree_through_the_api(sweep_and_truth):
    if not dealias.HAS_UNRAVEL_KERNEL:
        pytest.skip("compiled kernel not built")
    ds, _ = sweep_and_truth
    a = dealias_velocity(ds, "VRADH", method="unravel", engine="numpy")
    b = dealias_velocity(ds, "VRADH", method="unravel", engine="compiled")
    np.testing.assert_array_equal(a.values, b.values)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_volume_with_and_without_the_3d_check(engine):
    tree, truths = volume(elevations=(0.5, 1.5, 3.0), nray=36, ngate=40)
    three = dealias_velocity(tree, "VRADH", method="unravel", engine=engine)
    plain = dealias_velocity(
        tree, "VRADH", method="unravel", sweep_continuity=False, engine=engine
    )
    assert (
        set(three.children) == set(plain.children) == {"sweep_0", "sweep_1", "sweep_2"}
    )
    for name in three.children:
        node = three[name].to_dataset()["VRADH_dealiased"]
        ok = np.isfinite(node.values)
        assert np.mean(np.abs(node.values - truths[name])[ok] < 10.0) > 0.95
        # the 2-D result of a sweep is what the sweep alone gives
        alone = dealias_velocity(
            tree[name].to_dataset(inherit="all_coords"),
            "VRADH",
            method="unravel",
            engine=engine,
        )
        np.testing.assert_array_equal(
            plain[name].to_dataset()["VRADH_dealiased"].values, alone.values
        )
    assert (
        plain["sweep_0"].to_dataset()["VRADH_dealiased"].attrs["dealias_anchor"]
        == "unravel"
    )


def test_the_3d_check_repairs_a_sweep_that_is_off_by_a_fold():
    tree, truths = volume(elevations=(0.5, 1.5), nray=36, ngate=40)
    sweep = tree["sweep_1"].to_dataset(inherit="all_coords")
    # a block of the upper sweep unfolded wrongly by a corrupted interval cannot be seen
    # by the sweep alone; the check against the sweep below fixes isolated disagreements
    out_alone = dealias_velocity(sweep, "VRADH", method="unravel", engine="numpy")
    out_tree = dealias_velocity(tree, "VRADH", method="unravel", engine="numpy")
    node = out_tree["sweep_1"].to_dataset()["VRADH_dealiased"]
    err_alone = np.mean(
        np.abs(out_alone.values - truths["sweep_1"])[np.isfinite(out_alone.values)]
        > 10.0
    )
    err_tree = np.mean(
        np.abs(node.values - truths["sweep_1"])[np.isfinite(node.values)] > 10.0
    )
    assert err_tree <= err_alone + 1e-12


def test_options_and_variants(sweep_and_truth):
    ds, _ = sweep_and_truth
    base = dealias_velocity(ds, "VRADH", method="unravel", engine="numpy")
    strict = dealias_velocity(ds, "VRADH", method="unravel", alpha=0.6, engine="numpy")
    assert np.isfinite(strict.values).sum() > 0
    named = dealias_velocity(ds, "VRADH", method="unravel", name="U", engine="numpy")
    assert named.name == "U"
    flipped = dealias_velocity(
        ds.transpose("range", "azimuth"), "VRADH", method="unravel", engine="numpy"
    )
    assert flipped.dims == ("range", "azimuth")
    np.testing.assert_array_equal(
        flipped.transpose("azimuth", "range").values, base.values
    )
    both = dealias_velocity(
        ds, "VRADH", method="unravel", confidence=True, engine="numpy"
    )
    assert set(both.data_vars) == {"VRADH_dealiased", "VRADH_dealiased_confidence"}
    np.testing.assert_array_equal(both["VRADH_dealiased"].values, base.values)
    acc = ds.radarx.dealias("VRADH", method="unravel", engine="numpy")
    np.testing.assert_array_equal(acc.values, base.values)
    # the region method is still the default
    region = dealias_velocity(ds, "VRADH", engine="numpy")
    assert "region-based" in region.attrs["comment"]


def test_references_and_provenance(sweep_and_truth):
    ds, _ = sweep_and_truth
    out = dealias_velocity(ds, "VRADH", method="unravel", engine="numpy")
    assert "10.1175/JTECH-D-19-0020.1" in out.attrs["radarx_references"]
    assert "method='unravel'" in out.attrs["history"]


def test_input_is_not_changed(sweep_and_truth):
    ds, _ = sweep_and_truth
    before = ds.copy(deep=True)
    dealias_velocity(ds, "VRADH", method="unravel", engine="numpy")
    xr.testing.assert_identical(ds, before)


def test_errors(sweep_and_truth, monkeypatch):
    ds, _ = sweep_and_truth
    with pytest.raises(ValueError, match="method must be"):
        dealias_velocity(ds, "VRADH", method="zhang")
    with pytest.raises(ValueError, match="needs no reference"):
        dealias_velocity(ds, "VRADH", method="unravel", wind_profile=xr.Dataset())
    with pytest.raises(ValueError, match="needs no reference"):
        dealias_velocity(ds, "VRADH", method="unravel", reference=ds.VRADH)
    with pytest.raises(ValueError, match="products_only"):
        dealias_velocity(ds, "VRADH", method="unravel", products_only=False)
    with pytest.raises(ValueError, match="alpha"):
        dealias_velocity(ds, "VRADH", method="unravel", alpha=0.0)
    monkeypatch.setattr(dealias, "HAS_UNRAVEL_KERNEL", False)
    with pytest.raises(ImportError, match="UNRAVEL"):
        dealias_velocity(ds, "VRADH", method="unravel", engine="compiled")
    out = dealias_velocity(ds, "VRADH", method="unravel")  # falls back to Python
    assert np.isfinite(out.values).any()


def test_a_gate_that_no_fold_brings_to_its_neighbours_is_dropped():
    vel = np.full((24, 20), 5.0)
    vel[6, 8] = -3.0  # 8 m/s below its neighbours; the next fold, 17, is 12 above them
    vel[6, 9] = (
        -9.0
    )  # the other negative gate of the window: its mean, -6, is 3 m/s from both
    az = np.arange(24) * 15.0
    ds = xr.Dataset(
        {"VRADH": (("azimuth", "range"), vel)},
        coords={
            "azimuth": az,
            "range": 1000.0 + 500.0 * np.arange(20),
            "elevation": ("azimuth", np.full(24, 0.5)),
            "nyquist_velocity": 10.0,
        },
    )
    out = dealias_velocity(ds, "VRADH", method="unravel", engine="numpy")
    assert np.isnan(out.values[6, 8:10]).all()
    rest = np.delete(out.values.ravel(), [6 * 20 + 8, 6 * 20 + 9])
    assert np.allclose(rest, 5.0)


def islands():
    """A strip of echo around the radar and two islands that the early passes cannot reach."""
    nray, ngate = 90, 70
    az = np.arange(nray) * 4.0 + 1.0
    rng = 1000.0 + 500.0 * np.arange(ngate)
    truth = 12.0 * np.cos(np.radians(az - 30.0))[:, None] * np.ones(ngate)[None, :]
    folded = (truth + 10.0) % 20.0 - 10.0
    vel = np.full((nray, ngate), np.nan)
    vel[:, :10] = folded[:, :10]  # the strip
    vel[40:46, 25:31] = folded[40:46, 25:31]  # reached by the box of (40, 10)
    vel[0:5, 55:62] = folded[0:5, 55:62]  # beyond every box: the closest reference
    return vel, truth, az, rng


@pytest.mark.parametrize("wrap", [True, False])
def test_boxes_and_the_closest_reference_reach_the_islands(wrap):
    vel, truth, az, rng = islands()
    if not wrap:
        vel, truth, az = vel[:60], truth[:60], az[:60]
    out, flag = run_spec(vel, 10.0, az, rng, wrap)
    ok = np.isfinite(vel)
    assert (flag[ok] > 0).all()
    assert np.allclose(out[ok], truth[ok], atol=1e-9)


@needs_kernel
@pytest.mark.parametrize("wrap", [True, False])
def test_kernel_is_identical_on_the_islands(wrap):
    vel, truth, az, rng = islands()
    if not wrap:
        vel, truth, az = vel[:60], truth[:60], az[:60]
    ref, ref_flag = run_spec(vel, 10.0, az, rng, wrap)
    out, flag = run_kernel(vel, 10.0, az, rng, wrap)
    assert np.array_equal(flag, ref_flag)
    np.testing.assert_array_equal(out, ref)


@needs_kernel
def test_kernel_3d_check_is_identical_to_the_specification():
    rnd = np.random.default_rng(4)
    above = rnd.normal(0.0, 6.0, (30, 24))
    below = rnd.normal(0.0, 6.0, (28, 20))
    f_above = np.where(rnd.random(above.shape) < 0.2, -3, 1).astype(np.int8)
    f_below = np.where(rnd.random(below.shape) < 0.2, -3, 1).astype(np.int8)
    iray = np.broadcast_to(
        np.clip(np.arange(30) - 1, -1, 27)[:, None], above.shape
    ).astype(np.int32)
    igate = np.broadcast_to(np.arange(24)[None, :] - 2, above.shape).astype(np.int32)
    for wrap in (True, False):
        want = above.copy()
        spec.check_3d(
            want, f_above, below, f_below, iray, igate, 10.0, 0.8, (4, 2), wrap
        )
        got = dealias._unravel.check_3d(
            above, f_above, below, f_below, iray, igate, 10.0, 0.8, 4, 2, wrap
        )
        np.testing.assert_array_equal(got, want)
