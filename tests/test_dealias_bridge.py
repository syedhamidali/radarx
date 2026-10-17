#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Region dealiasing where a thin bridge joins two blocks that are a fold apart."""

import gzip
import shutil

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401  registers the accessors
from radarx.retrieve import dealias, dealias_velocity

from .test_dealias import nexrad_volume

needs_kernel = pytest.mark.skipif(
    not dealias.HAS_COMPILED_KERNEL, reason="compiled dealiasing kernel not built"
)
ENGINES = ["numpy", pytest.param("compiled", marks=needs_kernel)]

NYQUIST = 30.0
WEST, BLOCK = 20.0, 35.0  # true velocities (m/s) west and east of the fold line
LINE = 100  # first gate of the block
NECK = slice(170, 172)  # the two rays where the bridge is
NECK_GATES = slice(96, 104)


def bridged_sweep(neck=True, patch=True, seed=3):
    """
    Two blocks of one echo, 15 m/s apart in truth, joined only by a short neck.

    The block east of gate 100 is at 35 m/s, which the Nyquist velocity of
    30 m/s folds to -25 m/s; the west block is at 20 m/s. Along the fold line
    the folded velocities jump by 45 m/s, 0.75 of the Nyquist interval, so the
    boundary says that the east block is one fold above the west one. At the
    neck, two rays wide, the folded velocity passes through zero in steps of
    6.4 m/s (smaller than 0.3 Vn) the way it does in a vortex, so a
    segmentation that follows every joined gate puts both blocks into one
    region. A hole and a small echo (``patch``) apart from the rest have no core of their own.
    Returns the sweep, the true velocity and the mask of the neck gates, which
    no fold decides.
    """
    rnd = np.random.default_rng(seed)
    nray, ngate = 360, 200
    azimuth = np.arange(nray) + 0.5
    rng = 1000.0 + 250.0 * np.arange(ngate)
    truth = np.where(np.arange(ngate) < LINE, WEST, BLOCK)[None, :] * np.ones((nray, 1))
    truth = truth + 0.4 * rnd.standard_normal(truth.shape)
    measured = (truth + NYQUIST) % (2 * NYQUIST) - NYQUIST
    in_neck = np.zeros(truth.shape, dtype=bool)
    if neck:
        ramp = np.linspace(
            WEST, BLOCK - 2 * NYQUIST, NECK_GATES.stop - NECK_GATES.start
        )
        measured[NECK, NECK_GATES] = ramp
        in_neck[NECK, NECK_GATES] = True
    if patch:
        measured[10:16, 20:28] = np.nan  # a hole in the echo
        keep = measured[210:216, 30:38].copy()
        measured[190:241, :] = np.nan  # a gap of rays
        measured[210:216, 30:38] = keep  # leaves a small echo of its own
    ds = xr.Dataset(
        {"VRADH": (("azimuth", "range"), measured, {"units": "m s-1"})},
        coords={
            "azimuth": ("azimuth", azimuth, {"units": "degrees"}),
            "range": ("range", rng, {"units": "meters"}),
            "elevation": ("azimuth", np.full(nray, 0.5), {"units": "degrees"}),
            "nyquist_velocity": ((), NYQUIST, {"units": "m s-1"}),
        },
    )
    return ds, truth, in_neck


def share_right(out, truth, ignore):
    keep = np.isfinite(out.values) & ~ignore
    return float(np.mean(np.abs(out.values - truth)[keep] < NYQUIST))


@pytest.mark.parametrize("engine", ENGINES)
def test_a_thin_bridge_does_not_make_one_region_of_two_blocks(engine):
    ds, truth, neck = bridged_sweep()
    reference = xr.DataArray(
        np.where(neck, np.nan, truth), dims=("azimuth", "range"), coords=ds.coords
    )
    out = dealias_velocity(ds, reference=reference, engine=engine)
    assert share_right(out, truth, neck) > 0.999
    # the east block really is the folded one
    east = np.s_[:, LINE + 2 :]
    ok = np.isfinite(out.values[east]) & ~neck[east]
    np.testing.assert_allclose(out.values[east][ok], truth[east][ok], atol=3.0)


@pytest.mark.parametrize("engine", ENGINES)
def test_the_blocks_are_a_fold_apart_without_any_reference(engine):
    # the boundary alone decides the fold of the east block relative to the west one
    ds, _, neck = bridged_sweep()
    out = dealias_velocity(ds, engine=engine).values
    west = np.nanmedian(out[:, 10:90])
    east = np.nanmedian(out[:, LINE + 10 :])
    assert east - west == pytest.approx(BLOCK - WEST, abs=1.0)


def test_the_neck_decides_nothing_without_the_core_rule(monkeypatch):
    # with every joined gate used the two blocks are one region and the
    # boundary between them is never looked at: the block stays one fold off
    monkeypatch.setattr(dealias, "_CORE_JOINS", 0)
    ds, truth, neck = bridged_sweep()
    reference = xr.DataArray(
        np.where(neck, np.nan, truth), dims=("azimuth", "range"), coords=ds.coords
    )
    out = dealias_velocity(ds, reference=reference, engine="numpy")
    assert share_right(out, truth, neck) < 0.99


@needs_kernel
def test_the_engines_give_the_same_regions_and_folds():
    ds, _, _ = bridged_sweep()
    sweep = dealias._Sweep(ds, "VRADH", NYQUIST)
    options = dict(
        threshold=0.3, max_gap=20, max_iterations=100, n_threads=2, compiled=False
    )
    numpy_k, numpy_c = dealias._region_folds([sweep], **options)[0]
    options["compiled"] = True
    kernel_k, kernel_c = dealias._region_folds([sweep], **options)[0]
    np.testing.assert_array_equal(kernel_k, numpy_k)
    np.testing.assert_array_equal(kernel_c, numpy_c)
    # the echo and the small echo apart from it are two components; the blocks
    # of the echo are a fold apart
    assert np.unique(numpy_c[numpy_c >= 0]).size == 2
    assert numpy_k[0, 150] - numpy_k[0, 50] == 1


@pytest.mark.parametrize("engine", ENGINES)
def test_a_sweep_without_a_large_core_is_segmented_as_before(engine, monkeypatch):
    # too small to hold a core of 1000 gates: the core rule changes nothing
    ds, truth, neck = bridged_sweep()
    window = dict(azimuth=slice(165, 175), range=slice(80, 120))
    reference = xr.DataArray(
        np.where(neck, np.nan, truth), dims=("azimuth", "range"), coords=ds.coords
    ).isel(**window)
    out = dealias_velocity(ds.isel(**window), reference=reference, engine=engine)
    monkeypatch.setattr(dealias, "_CORE_JOINS", 0)
    plain = dealias_velocity(ds.isel(**window), reference=reference, engine="numpy")
    np.testing.assert_array_equal(out.values, plain.values)


def test_segments_helper_without_core_rule_and_without_seeds():
    v = np.array([0.0, 1.0, 2.0, np.nan, 9.0, 10.0, 30.0])
    ja, jb = np.array([0, 1, 4]), np.array([1, 2, 5])
    plain = dealias._segments(v.size, ~np.isnan(v), v, ja, jb, 0)
    assert plain[0] == plain[1] == plain[2] and plain[4] == plain[5]
    assert len({plain[0], plain[4], plain[6]}) == 3
    # no core of 1000 gates: the same regions
    cored = dealias._segments(v.size, ~np.isnan(v), v, ja, jb, 4)
    keep = ~np.isnan(v)
    same = cored[keep][:, None] == cored[keep][None, :]
    assert (same == (plain[keep][:, None] == plain[keep][None, :])).all()


# --------------------------------------------------------------------------
# real data: the lowest Doppler cuts of KBMX, 27 April 2011 22:19 UTC (issue 294)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kbmx2011(tmp_path_factory):
    from radarx.io.aws_data import download_file

    out = tmp_path_factory.mktemp("kbmx2011")
    try:
        packed = download_file(
            "unidata-nexrad-level2",
            "2011/04/27/KBMX/KBMX20110427_221945_V03.gz",
            str(out),
        )
    except Exception as err:  # noqa: BLE001  # pragma: no cover - network
        pytest.skip(f"NEXRAD data not available: {err}")
    path = str(out / "KBMX20110427_221945_V03")
    with gzip.open(packed) as src, open(path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return nexrad_volume(path)


def one_fold_apart(a, b, nyquist):
    """Gates where two dealiased fields differ by more than the Nyquist velocity."""
    both = np.isfinite(a) & np.isfinite(b)
    return int(np.count_nonzero(np.abs(a - b)[both] > nyquist))


def block_north_of_the_radar(da):
    """Mask of the block of issue 294: x 0 to 69 km, y 52 to 133 km from the radar."""
    az = np.deg2rad(da["azimuth"].values)[:, None]
    rng = da["range"].values[None, :]
    x, y = rng * np.sin(az) / 1e3, rng * np.cos(az) / 1e3
    return (x > 0) & (x < 69) & (y > 52) & (y < 133)


@needs_kernel
def test_kbmx_block_of_the_lowest_cut_follows_the_cuts_above(kbmx2011):
    region = dealias_velocity(kbmx2011, "VRADH")
    unravel = dealias_velocity(kbmx2011, "VRADH", method="unravel")
    nyquist = float(kbmx2011["sweep_1"]["nyquist_velocity"])
    for name in ("sweep_1", "sweep_3"):
        a = region[name]["VRADH_dealiased"]
        b = unravel[name]["VRADH_dealiased"].values
        box = block_north_of_the_radar(a)
        both = box & np.isfinite(a.values) & np.isfinite(b)
        assert both.sum() > 10000
        # before: every gate of the block was 2 Vn away from UNRAVEL and from the cuts above
        assert np.mean(np.abs(a.values - b)[both] < nyquist) > 0.97
    first = one_fold_apart(
        region["sweep_1"]["VRADH_dealiased"].values,
        unravel["sweep_1"]["VRADH_dealiased"].values,
        nyquist,
    )
    assert first < 12000  # 29 539 gates before
