#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the speed limit of ``dealias_velocity`` (``max_speed``)
=================================================================

A vortex core with peaks of about 0.95 Vn that joins a region which really is
folded is shifted with that region by the fold optimisation. ``max_speed``
undoes the shift of the gates that end up beyond the speed the storm can have.
"""
import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401  registers the accessors
from radarx.retrieve import dealias, dealias_velocity

from .test_dealias import synthetic_sweep, volume

needs_kernel = pytest.mark.skipif(
    not dealias.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)
ENGINES = ["numpy", pytest.param("compiled", marks=needs_kernel)]
NYQUIST = 30.0
CENTRE = (60, 50)
RADII = (4.0, 6.0)


def vortex_field(rim=28.5, core=-12.0, env=-28.5, patch=-34.0, floor=-12.0):
    """
    True velocity (m/s) of an outbound pocket in an inbound flow.

    The flow is 0.95 Vn towards the radar. A folded patch of -34 m/s (1.13 Vn)
    lies at the top of the pocket, whose rim goes up to +28.5 m/s (0.95 Vn) on
    three sides and falls to ``floor`` on the fourth, and whose velocity
    decreases to ``core`` at the centre. Measured, the patch is at +26 m/s and
    touches the rim without a step, so the pocket and the patch form one
    region of the segmentation.
    """
    ray, gate = np.meshgrid(np.arange(120), np.arange(100), indexing="ij")

    def box(lo, hi, x, width=1.5):
        return 0.5 * (np.tanh((x - lo) / width) - np.tanh((x - hi) / width))

    u = np.full(ray.shape, env) + (patch - env) * box(54, 66, ray) * box(38, 46, gate)
    rho = np.hypot((ray - CENTRE[0]) / RADII[0], (gate - CENTRE[1]) / RADII[1])
    after = 0.5 * (1 - np.tanh((ray - CENTRE[0] - 1.0) / 0.6))
    local_rim = floor + (rim - floor) * after
    bowl = core + (local_rim - core) * np.minimum(rho, 1.0) ** 2
    return np.where(rho <= 1.0, bowl, u), rho <= 1.0


def vortex_sweep():
    """The aliased sweep, the true velocity, the gates of the pocket and of the patch."""
    truth, pocket = vortex_field()
    measured = (truth + NYQUIST) % (2 * NYQUIST) - NYQUIST
    azimuth = (np.arange(120) + 0.5) * 3.0
    ds = xr.Dataset(
        {"VRADH": (("azimuth", "range"), measured, {"units": "m s-1"})},
        coords={
            "azimuth": ("azimuth", azimuth, {"units": "degrees"}),
            "range": ("range", 2000.0 + 250.0 * np.arange(100), {"units": "m"}),
            "elevation": ("azimuth", np.full(120, 0.5)),
            "nyquist_velocity": ((), NYQUIST),
        },
    )
    return ds, truth, pocket, (truth < -NYQUIST) & ~pocket


@pytest.fixture(scope="module")
def vortex():
    return vortex_sweep()


@pytest.mark.parametrize("engine", ENGINES)
def test_the_default_shifts_the_core_with_the_folded_patch(vortex, engine):
    ds, truth, pocket, patch = vortex
    assert truth[pocket].min() > -NYQUIST and truth[pocket].max() < NYQUIST
    out = dealias_velocity(ds, engine=engine).values
    # the core of the pocket goes to speeds of 2.4 Vn; the patch is unfolded
    assert out[pocket].min() < -1.9 * NYQUIST
    assert np.all(np.abs(out - truth)[patch] < 1.0)
    assert np.count_nonzero(np.abs(out - truth) > NYQUIST) > 60


@pytest.mark.parametrize("engine", ENGINES)
def test_max_speed_keeps_the_core_and_the_real_fold(vortex, engine):
    ds, truth, pocket, patch = vortex
    out = dealias_velocity(ds, max_speed=1.2 * NYQUIST, engine=engine).values
    error = np.abs(out - truth)
    assert np.all(error[patch] < 1.0)  # the fold line of the patch survives
    assert np.count_nonzero(error[pocket] > 1.0) <= 8
    assert np.count_nonzero(error > NYQUIST) <= 6
    assert abs(out[pocket].min() - truth[pocket].min()) < 25.0
    assert abs(out[pocket].max() - truth[pocket].max()) < 10.0
    # the measured peaks are 28.5 m/s: the core stays within a few m/s of them
    assert out[pocket].min() > -1.2 * NYQUIST - 0.5


@pytest.mark.parametrize("engine", ENGINES)
def test_a_looser_limit_undoes_less(vortex, engine):
    ds, truth, pocket, _ = vortex
    wrong = [
        np.count_nonzero(
            np.abs(dealias_velocity(ds, max_speed=limit, engine=engine).values - truth)
            > NYQUIST
        )
        for limit in (1.2 * NYQUIST, 1.5 * NYQUIST, 2.5 * NYQUIST, None)
    ]
    assert wrong[0] <= wrong[1] < wrong[2] <= wrong[3]
    assert wrong[0] < wrong[3] / 5


def test_the_engines_agree_on_the_vortex(vortex):
    if not dealias.HAS_COMPILED_KERNEL:
        pytest.skip("compiled kernel not built")
    ds = vortex[0]
    for limit in (1.1 * NYQUIST, 1.2 * NYQUIST, 1.5 * NYQUIST, 2.0 * NYQUIST):
        a = dealias_velocity(ds, max_speed=limit, engine="numpy")
        b = dealias_velocity(ds, max_speed=limit, engine="compiled")
        np.testing.assert_array_equal(a.values, b.values)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_limit_above_the_winds_changes_nothing(engine):
    ds, _ = synthetic_sweep(nyquist=10.0, noise=1.0, ngate=120)
    plain = dealias_velocity(ds, engine=engine)
    limited = dealias_velocity(ds, max_speed=500.0, engine=engine)
    np.testing.assert_array_equal(plain.values, limited.values)


def noisy_sweep():
    return synthetic_sweep(nyquist=7.0, noise=2.0, gaps=0.15, ngate=200, nray=180)[0]


def test_engines_agree_on_a_noisy_field_with_a_limit():
    if not dealias.HAS_COMPILED_KERNEL:
        pytest.skip("compiled kernel not built")
    ds = noisy_sweep()
    plain = dealias_velocity(ds, engine="numpy")
    for limit in (12.0, 16.0, 24.0):
        a = dealias_velocity(ds, max_speed=limit, engine="numpy")
        b = dealias_velocity(ds, max_speed=limit, engine="compiled")
        np.testing.assert_array_equal(a.values, b.values)
    # the limit acts: gates beyond 12 m/s move
    moved = dealias_velocity(ds, max_speed=12.0, engine="numpy")
    assert np.count_nonzero(moved.values != plain.values) > 0


@pytest.mark.parametrize("engine", ENGINES)
def test_volume_with_and_without_sweep_continuity(engine):
    tree, _ = volume(nyquist=7.0, noise=2.0, ngate=160, nray=120)
    for continuity in (True, False):
        out = dealias_velocity(
            tree, max_speed=14.0, sweep_continuity=continuity, engine=engine
        )
        plain = dealias_velocity(tree, sweep_continuity=continuity, engine=engine)
        for name in ("sweep_0", "sweep_1", "sweep_2"):
            got = out[name]["VRADH_dealiased"].values
            ref = plain[name]["VRADH_dealiased"].values
            # the limit only moves gates back by whole intervals
            shift = (got - ref)[np.isfinite(got)] / 14.0
            np.testing.assert_allclose(shift, np.round(shift), atol=1e-4)
            assert np.count_nonzero(shift) > 0  # the limit acts in every sweep
        if engine == "numpy":
            continue
        other = dealias_velocity(
            tree,
            max_speed=14.0,
            sweep_continuity=continuity,
            engine="numpy",
        )
        for name in ("sweep_0", "sweep_1", "sweep_2"):
            np.testing.assert_array_equal(
                out[name]["VRADH_dealiased"].values,
                other[name]["VRADH_dealiased"].values,
            )


def test_lazy_sweeps_get_the_same_limit(vortex):
    pytest.importorskip("dask")
    ds = vortex[0]
    eager = dealias_velocity(ds, max_speed=1.2 * NYQUIST)
    lazy = dealias_velocity(ds.chunk(), max_speed=1.2 * NYQUIST)
    np.testing.assert_array_equal(lazy.values, eager.values)


def test_a_stack_of_sweeps_gets_the_limit_in_every_sweep(vortex):
    ds = vortex[0]
    stack = xr.concat([ds, ds], "time").assign_coords(time=[0, 1])
    out = dealias_velocity(stack, max_speed=1.2 * NYQUIST)
    single = dealias_velocity(ds, max_speed=1.2 * NYQUIST)
    assert out.dims[0] == "time"
    for k in (0, 1):
        np.testing.assert_array_equal(out.values[k], single.values)


def test_accessor_passes_the_limit_on(vortex):
    ds = vortex[0]
    got = ds.radarx.dealias(max_speed=1.2 * NYQUIST)
    expected = dealias_velocity(ds, max_speed=1.2 * NYQUIST)
    np.testing.assert_array_equal(got.values, expected.values)


def test_errors(vortex):
    ds = vortex[0]
    for bad in (0.0, -5.0, float("nan")):
        with pytest.raises(ValueError, match="max_speed must be positive"):
            dealias_velocity(ds, max_speed=bad)
    with pytest.raises(ValueError, match="only used by method='region'"):
        dealias_velocity(ds, method="unravel", max_speed=40.0)


def test_cluster_sizes_follow_the_rays_around():
    mask = np.zeros((4, 5), dtype=bool)
    mask[3, 1] = mask[0, 2] = mask[0, 3] = True  # ray 3 touches ray 0
    mask[2, 4] = True
    valid = np.ones((4, 5), dtype=bool)
    around = np.ones(4, dtype=np.uint8)
    size = dealias._cluster_sizes(mask, valid, around)
    assert size[3, 1] == size[0, 2] == size[0, 3] == 3
    assert size[2, 4] == 1
    assert size[~mask].sum() == 0
    open_ray = np.array([1, 1, 1, 0], dtype=np.uint8)  # the last ray has no successor
    size = dealias._cluster_sizes(mask, valid, open_ray)
    assert size[3, 1] == 1 and size[0, 2] == size[0, 3] == 2
