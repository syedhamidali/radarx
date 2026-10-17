#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the warning of ``dealias_velocity`` about shifted vortex cores
========================================================================

The region method can shift the core of a velocity couplet by one fold together
with a folded region next to it. The sweep alone cannot tell this from a strong
wind, so the result is not changed; a compact pocket of one fold with speeds
beyond two Nyquist velocities is reported, and ``max_speed`` undoes it.
"""
import gzip
import shutil
import warnings

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401  registers the accessors
from radarx.retrieve import dealias, dealias_velocity

from .test_dealias import synthetic_sweep, volume
from .test_dealias_vortex import ENGINES, NYQUIST, vortex_sweep

POCKET = "compact pocket"


class Stub:
    """The parts of a sweep that the pocket search reads."""

    def __init__(self, velocity, nyquist, links=None):
        self.velocity = np.asarray(velocity, dtype=float)
        self.nyquist = nyquist
        nray = self.velocity.shape[0]
        self.links = np.ones(nray, dtype=np.uint8) if links is None else links


def pockets(blocks, nyquist=30.0, shape=(40, 40), velocity=None, links=None):
    """
    Pocket search on a sweep with the given blocks of folds.

    The measured velocity is 20 m/s with the sign of the fold unless
    ``velocity`` is given.
    """
    folds = np.zeros(shape, dtype=np.int32)
    for (rows, cols), fold in blocks:
        folds[rows, cols] = fold
    measured = 20.0 * np.sign(folds) if velocity is None else np.full(shape, velocity)
    sweep = Stub(measured, nyquist, links)
    return dealias._fast_pockets(sweep, folds)


@pytest.fixture(scope="module")
def vortex():
    return vortex_sweep()


@pytest.mark.parametrize("engine", ENGINES)
def test_the_shifted_core_is_reported(vortex, engine):
    ds, truth, pocket, _ = vortex
    with pytest.warns(UserWarning, match=POCKET) as record:
        out = dealias_velocity(ds, engine=engine).values
    assert len(record) == 1
    message = str(record[0].message)
    assert "1 compact pocket(s)" in message and "max_speed" in message
    # the warning points at the user's call
    assert record[0].filename == __file__
    # the speeds it names are those of the result
    assert f"up to {np.abs(out[pocket]).max():.0f} m/s" in message
    assert np.abs(out).max() > 2 * NYQUIST


@pytest.mark.parametrize("engine", ENGINES)
def test_a_speed_limit_leaves_nothing_to_report(vortex, engine):
    ds = vortex[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        dealias_velocity(ds, max_speed=1.2 * NYQUIST, engine=engine)


def test_the_values_are_not_changed_by_the_check(vortex):
    ds = vortex[0]
    with pytest.warns(UserWarning, match=POCKET):
        reported = dealias_velocity(ds, engine="numpy")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = dealias_velocity(ds, max_speed=1e6, engine="numpy")
    np.testing.assert_array_equal(reported.values, again.values)


def test_the_volume_path_reports_it_too(vortex):
    ds = vortex[0]
    dtree = xr.DataTree.from_dict({"/": xr.Dataset(), "sweep_0": ds})
    with pytest.warns(UserWarning, match=POCKET):
        dealias_velocity(dtree)


@pytest.mark.parametrize("engine", ENGINES)
def test_ordinary_sweeps_are_not_reported(engine):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        for nyquist in (10.0, 30.0):
            ds, _ = synthetic_sweep(nyquist=nyquist, seed=3)
            dealias_velocity(ds, engine=engine)
        tree, _ = volume(nyquist=7.0, noise=2.0, ngate=160, nray=120)
        dealias_velocity(tree, engine=engine)


def test_a_pocket_beyond_the_speed_is_counted():
    # 4 x 4 gates of 20 m/s folded by +1 (Vn 30 m/s): 80 m/s, beyond 60 m/s
    count, gates, peak, limit = pockets([((slice(5, 9), slice(5, 9)), 1)])
    assert (count, gates) == (1, 16)
    assert peak == pytest.approx(80.0) and limit == pytest.approx(60.0)
    # the other sign: -20 m/s folded by -1
    count, gates, peak, _ = pockets([((slice(5, 9), slice(5, 9)), -1)])
    assert (count, gates) == (1, 16) and peak == pytest.approx(80.0)


def test_two_pockets_and_neighbours_of_the_other_fold():
    two = [((slice(2, 6), slice(2, 6)), 1), ((slice(20, 24), slice(20, 24)), 1)]
    assert pockets(two)[:2] == (2, 32)
    # a block of +1 beside a block of -1 is not one pocket: 8 gates each
    beside = [((slice(5, 7), slice(5, 9)), 1), ((slice(7, 9), slice(5, 9)), -1)]
    assert pockets(beside)[:2] == (0, 0)


def test_what_is_not_reported():
    block = (slice(5, 9), slice(5, 9))
    # too few gates beyond the speed (9)
    assert pockets([((slice(5, 8), slice(5, 8)), 1)])[:2] == (0, 0)
    # a pocket of more than 200 gates is a plateau of a real fold
    assert pockets([((slice(0, 20), slice(0, 20)), 1)])[:2] == (0, 0)
    # two folds: a wind that fast is folded everywhere
    assert pockets([(block, 2)])[:2] == (0, 0)
    # below the speed: 20 m/s folded by +1 is 80 m/s, 0 m/s is 60 m/s, not beyond
    assert pockets([(block, 1)], velocity=0.0)[:2] == (0, 0)
    # a small Nyquist velocity: 2 Vn is 20 m/s, but 50 m/s is the least reported
    count, _, _, limit = pockets([(block, 1)], nyquist=10.0, velocity=15.0)
    assert count == 0 and limit == pytest.approx(50.0)
    count, gates, peak, _ = pockets([(block, 1)], nyquist=10.0, velocity=35.0)
    assert (count, gates) == (1, 16) and peak == pytest.approx(55.0)


def test_a_pocket_across_the_first_and_last_ray():
    first = (slice(0, 1), slice(5, 11))
    last = (slice(39, 40), slice(5, 11))
    blocks = [(first, 1), (last, 1)]
    # the last ray touches the first one: 12 gates in one pocket
    assert pockets(blocks)[:2] == (1, 12)
    # an open sweep (the last ray does not touch the first): 6 gates each
    open_ray = np.ones(40, dtype=np.uint8)
    open_ray[-1] = 0
    assert pockets(blocks, links=open_ray)[:2] == (0, 0)


def test_cluster_labels_join_equal_values_only():
    valid = np.ones((6, 6), dtype=bool)
    mask = np.ones((6, 6), dtype=bool)
    same = np.zeros((6, 6), dtype=int)
    same[:, 3:] = 1
    links = np.ones(6, dtype=np.uint8)
    _, count = dealias._cluster_labels(mask, valid, links)
    assert count == 1
    label, count = dealias._cluster_labels(mask, valid, links, same=same)
    assert count == 2 and label[0, 0] != label[0, 5]
    sizes = dealias._cluster_sizes(mask, valid, links)
    assert np.all(sizes == 36)


# --------------------------------------------------------------------------
# real data: the Tuscaloosa-Birmingham tornado, KBMX 27 April 2011
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kbmx_sweep(tmp_path_factory):
    """0.5 degree Doppler cut of KBMX, 22:19:45 UTC, from the AWS archive."""
    import xradar as xd

    from radarx.io.aws_data import download_file

    folder = tmp_path_factory.mktemp("kbmx")
    try:
        packed = download_file(
            "unidata-nexrad-level2",
            "2011/04/27/KBMX/KBMX20110427_221945_V03.gz",
            str(folder),
        )
    except Exception as err:  # noqa: BLE001  # pragma: no cover - network
        pytest.skip(f"NEXRAD data not available: {err}")
    path = str(folder / "KBMX20110427_221945_V03")
    with gzip.open(packed) as src, open(path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    ds = xd.io.open_nexradlevel2_datatree(path)["sweep_1"].to_dataset()
    ds["VRADH"] = ds["VRADH"].where(ds["VRADH"] > -63.9)  # no-data codes
    return ds


def couplet_box(ds):
    """Gates within 5 degrees and 5 km of the couplet (azimuth 276, 64 km)."""
    azimuth = np.abs((ds["azimuth"].values - 276.0 + 180.0) % 360.0 - 180.0) < 5.0
    distance = np.abs(ds["range"].values - 64000.0) < 5000.0
    return azimuth[:, None] & distance[None, :]


def test_kbmx_couplet_is_reported_and_limited(kbmx_sweep):
    ds = kbmx_sweep
    box = couplet_box(ds)
    measured = ds["VRADH"].values
    assert np.nanmin(measured[box]) <= -29.5 and np.nanmax(measured[box]) >= 29.5
    with pytest.warns(UserWarning, match=POCKET) as record:
        default = dealias_velocity(ds, nyquist_velocity=29.97)
    assert "1 compact pocket(s)" in str(record[0].message)
    assert "up to 88 m/s" in str(record[0].message)
    # the default shifts the core of the couplet to -88 m/s
    assert np.nanmin(default.values[box]) < -85.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        limited = dealias_velocity(ds, nyquist_velocity=29.97, max_speed=36.0)
    peak = limited.values[box]
    assert -37.0 < np.nanmin(peak) < -34.0 and np.nanmax(peak) > 25.0
    # both engines give the same
    if dealias.HAS_COMPILED_KERNEL:
        numpy = dealias_velocity(
            ds, nyquist_velocity=29.97, max_speed=36.0, engine="numpy"
        )
        np.testing.assert_array_equal(numpy.values, limited.values)
