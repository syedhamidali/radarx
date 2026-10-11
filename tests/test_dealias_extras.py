#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for the dual-PRF correction and the dealiasing confidence."""

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401  registers the accessors
from radarx.retrieve import (
    correct_dual_prf,
    dealias,
    dealias_confidence,
    dealias_velocity,
)

from .test_dealias import synthetic_sweep, volume

needs_kernel = pytest.mark.skipif(
    not dealias.HAS_COMPILED_KERNEL, reason="compiled dealiasing kernel not built"
)
ENGINES = ["numpy", pytest.param("compiled", marks=needs_kernel)]

VH, VL = 16.0, 12.0  # high and low PRF Nyquist velocities: unfolding factor 3
N = VL / (VH - VL)
VLH = VH * VL / (VH - VL)  # extended interval, 48 m/s


def dual_prf_sweep(sigma=1.0, seed=0, nray=720, ngate=240, gaps=0.0):
    """
    Dual-PRF velocities of a smooth wind field, made as the radar processor does.

    Even rays use the high PRF, odd rays the low PRF. Each ray measures the
    true velocity plus noise folded into the Nyquist interval of its PRF; the
    primary estimate ``(N + 1) v_l - N v_h`` of a ray and its predecessor says
    which interval the folded velocity of the ray belongs to (Holleman and
    Beekhuis 2003, Eqs. 3-5). Returns the sweep and the noise-free field.
    """
    rnd = np.random.default_rng(seed)
    az = np.arange(nray) * 360.0 / nray + 0.25
    rng = 1000.0 + 250.0 * np.arange(ngate)
    a = np.radians(az)[:, None]
    truth = 28.0 * np.sin(a + 0.4) * (0.5 + 0.5 * rng[None, :] / rng[-1])
    truth = truth + 5.0 * np.sin(rng[None, :] / 8e3)

    def fold(v, vu):
        return (v + vu) % (2 * vu) - vu

    high = np.arange(nray) % 2 == 0
    vu = np.where(high, VH, VL)[:, None]
    noisy = truth + sigma * rnd.standard_normal(truth.shape)
    measured = fold(noisy, vu)
    prev = np.roll(measured, 1, axis=0)
    v_h = np.where(high[:, None], measured, prev)
    v_l = np.where(high[:, None], prev, measured)
    primary = fold((N + 1) * v_l - N * v_h, VLH)
    m = np.round((primary - measured) / (2 * vu))
    unfolded = measured + 2 * vu * m
    if gaps:
        unfolded[rnd.random(unfolded.shape) < gaps] = np.nan
    ds = xr.Dataset(
        {
            "VRADH": (
                ("azimuth", "range"),
                unfolded.astype(np.float32),
                {"units": "m s-1"},
            )
        },
        coords={
            "azimuth": ("azimuth", az, {"units": "degrees"}),
            "range": ("range", rng, {"units": "m"}),
            "elevation": ("azimuth", np.full(nray, 0.5)),
        },
    )
    return ds, truth, noisy


def outlier_fraction(values, noisy):
    ok = np.isfinite(values)
    return np.mean(np.abs(values - noisy)[ok] > 5.0)


@pytest.fixture(scope="module")
def dual():
    return dual_prf_sweep()


# --------------------------------------------------------------------------
# local median
# --------------------------------------------------------------------------


def brute_median(vel, wrap, half_start, half_max, min_points, centre):
    nray, ngate = vel.shape
    out = np.full(vel.shape, np.nan)
    if wrap:
        half_max = min(half_max, (nray - 1) // 2)
    for i in range(nray):
        for g in range(ngate):
            if np.isnan(vel[i, g]):
                continue
            for h in range(half_start, half_max + 1):
                vals = []
                for di in range(-h, h + 1):
                    q = i + di
                    if wrap:
                        q %= nray
                    elif not 0 <= q < nray:
                        continue
                    for dg in range(-h, h + 1):
                        k = g + dg
                        if not 0 <= k < ngate or (di == 0 and dg == 0 and not centre):
                            continue
                        if np.isfinite(vel[q, k]):
                            vals.append(vel[q, k])
                if len(vals) >= min_points:
                    out[i, g] = np.median(vals)
                    break
    return out


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("wrap", [True, False])
@pytest.mark.parametrize(
    "half_start, half_max, min_points, centre",
    [(1, 1, 3, False), (1, 3, 9, True), (0, 2, 4, True), (2, 3, 6, False)],
)
def test_local_median_matches_brute_force(
    engine, wrap, half_start, half_max, min_points, centre
):
    rnd = np.random.default_rng(5)
    vel = rnd.normal(0.0, 5.0, (17, 23))
    vel[rnd.random(vel.shape) < 0.35] = np.nan
    vel[3:7, 4:9] = np.nan
    got = dealias._local_median(
        vel, wrap, half_start, half_max, min_points, centre, None, engine == "compiled"
    )
    want = brute_median(vel, wrap, half_start, half_max, min_points, centre)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12, equal_nan=True)


@needs_kernel
def test_local_median_kernel_errors():
    k = dealias._dealias
    with pytest.raises(ValueError, match="2-D"):
        k.local_median(np.zeros(5), True, 1, 1, 3, False, 1)
    with pytest.raises(ValueError, match="half_start"):
        k.local_median(np.zeros((4, 4)), True, 2, 1, 3, False, 1)
    with pytest.raises(ValueError, match="min_points"):
        k.local_median(np.zeros((4, 4)), True, 1, 1, 0, False, 1)


def test_full_circle_detection():
    assert dealias._is_full_circle(np.arange(0.0, 360.0, 1.0))
    assert dealias._is_full_circle(np.mod(np.arange(0.0, 360.0, 1.0) + 200.0, 360.0))
    assert not dealias._is_full_circle(np.arange(0.0, 90.0, 1.0))
    assert not dealias._is_full_circle(np.array([0.0, 1.0]))


# --------------------------------------------------------------------------
# dual-PRF correction
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_dual_prf_removes_the_outliers(dual, engine):
    ds, truth, noisy = dual
    before = outlier_fraction(ds["VRADH"].values.astype(float), noisy)
    assert 0.003 < before < 0.03  # about 1 percent, as in the paper
    out = correct_dual_prf(ds, nyquist_high=VH, nyquist_low=VL, engine=engine)
    fixed = out["VRADH_dualprf"]
    after = outlier_fraction(fixed.values.astype(float), noisy)
    assert after < before / 25.0  # the paper's first pass alone gains 25
    # unchanged gates are unchanged, the others moved by multiples of 2 Vu
    shift = out["VRADH_dualprf_shift"].values
    vu = np.where(np.arange(ds.sizes["azimuth"]) % 2 == 0, VH, VL)[:, None]
    np.testing.assert_allclose(
        fixed.values, ds["VRADH"].values + 2 * vu * shift, rtol=0, atol=2e-5
    )
    moved = shift != 0
    assert moved.sum() >= 0.9 * before * moved.size
    assert fixed.attrs["outliers_corrected"] == moved.sum()
    assert fixed.dtype == np.float32 and out["VRADH_dualprf_shift"].dtype == np.int32
    # what was a good gate is not touched: the share of changed gates is the share of outliers
    assert moved.mean() < 2.5 * before


@pytest.mark.parametrize("passes", [1, 2, 3])
def test_dual_prf_more_passes_help(dual, passes):
    ds, _, noisy = dual
    one = correct_dual_prf(
        ds, nyquist_high=VH, nyquist_low=VL, passes=1, engine="numpy"
    )
    many = correct_dual_prf(
        ds, nyquist_high=VH, nyquist_low=VL, passes=passes, engine="numpy"
    )
    f1 = outlier_fraction(one["VRADH_dualprf"].values.astype(float), noisy)
    fn = outlier_fraction(many["VRADH_dualprf"].values.astype(float), noisy)
    assert fn <= f1 + 1e-12
    assert many["VRADH_dualprf"].attrs["passes"] == passes


def test_dual_prf_engines_identical(dual):
    if not dealias.HAS_COMPILED_KERNEL:
        pytest.skip("compiled kernel not built")
    ds, *_ = dual_prf_sweep(sigma=2.0, seed=3, gaps=0.2)
    a = correct_dual_prf(ds, nyquist_high=VH, nyquist_low=VL, engine="numpy")
    b = correct_dual_prf(ds, nyquist_high=VH, nyquist_low=VL, engine="compiled")
    for name in a.data_vars:
        np.testing.assert_array_equal(a[name].values, b[name].values)


@pytest.mark.parametrize("engine", ENGINES)
def test_dual_prf_prf_labels(dual, engine):
    ds, _, noisy = dual
    n = ds.sizes["azimuth"]
    right = correct_dual_prf(ds, nyquist_high=VH, nyquist_low=VL, engine=engine)
    # the same labels given as an array, a variable and as a DataArray
    flags = np.arange(n) % 2 == 0
    for prf in (flags, flags.astype(int), xr.DataArray(flags, dims="azimuth")):
        got = correct_dual_prf(
            ds, nyquist_high=VH, nyquist_low=VL, prf=prf, engine=engine
        )
        np.testing.assert_array_equal(
            got["VRADH_dualprf"].values, right["VRADH_dualprf"].values
        )
    labelled = ds.assign(prf_flag=("azimuth", flags))
    got = correct_dual_prf(
        labelled, nyquist_high=VH, nyquist_low=VL, prf="prf_flag", engine=engine
    )
    np.testing.assert_array_equal(
        got["VRADH_dualprf"].values, right["VRADH_dualprf"].values
    )
    # the opposite order is the same as first_ray="low"
    swapped = correct_dual_prf(
        ds, nyquist_high=VH, nyquist_low=VL, prf=~flags, engine=engine
    )
    low = correct_dual_prf(
        ds, nyquist_high=VH, nyquist_low=VL, first_ray="low", engine=engine
    )
    np.testing.assert_array_equal(
        swapped["VRADH_dualprf"].values, low["VRADH_dualprf"].values
    )
    # with the wrong labels the thresholds are wrong and the correction is worse
    f_right = outlier_fraction(right["VRADH_dualprf"].values.astype(float), noisy)
    f_wrong = outlier_fraction(low["VRADH_dualprf"].values.astype(float), noisy)
    assert f_wrong > f_right


@pytest.mark.parametrize("engine", ENGINES)
def test_dual_prf_window_grows_over_missing_gates(engine):
    """A gate whose 3 x 3 window is half empty uses a larger window."""
    az = np.arange(0.0, 360.0, 1.0)
    rng = 1000.0 + 250.0 * np.arange(60)
    vel = np.full((az.size, rng.size), 4.0)
    rnd = np.random.default_rng(1)
    holes = rnd.random(vel.shape) < 0.55
    vel[holes] = np.nan
    i, j = 100, 30
    vel[i, j] = 4.0 + 2 * VH  # a high-PRF ray (even index) off by one interval
    ds = xr.Dataset(
        {"VRADH": (("azimuth", "range"), vel)},
        coords={
            "azimuth": az,
            "range": rng,
            "elevation": ("azimuth", np.full(az.size, 0.5)),
        },
    )
    ring = vel[i - 1 : i + 2, j - 1 : j + 2]
    assert np.isfinite(ring).sum() < 9  # the 3 x 3 window is too empty
    narrow = correct_dual_prf(
        ds, nyquist_high=VH, nyquist_low=VL, max_window=3, engine=engine
    )
    assert narrow["VRADH_dualprf_shift"].values[i, j] == 0
    wide = correct_dual_prf(
        ds, nyquist_high=VH, nyquist_low=VL, max_window=9, engine=engine
    )
    assert wide["VRADH_dualprf"].values[i, j] == pytest.approx(4.0)
    assert wide["VRADH_dualprf_shift"].values[i, j] == -1


@pytest.mark.parametrize("engine", ENGINES)
def test_dual_prf_correction_is_the_multiple_of_the_ray_interval(engine):
    """Low-PRF rays move by 2 * 12 and high-PRF rays by 2 * 16 m/s."""
    az = np.arange(0.0, 360.0, 1.0)
    vel = np.full((az.size, 40), 3.0)
    vel[10, 20] += 2 * VH  # even ray: high PRF
    vel[11, 20] += 2 * VL * 2  # odd ray: low PRF, two intervals
    vel[200, 5] = np.nan
    ds = xr.Dataset(
        {"VRADH": (("azimuth", "range"), vel)},
        coords={
            "azimuth": az,
            "range": 1000.0 + 250.0 * np.arange(40),
            "elevation": ("azimuth", np.full(az.size, 0.5)),
        },
    )
    out = correct_dual_prf(ds, nyquist_high=VH, nyquist_low=VL, engine=engine)
    assert out["VRADH_dualprf_shift"].values[10, 20] == -1
    assert out["VRADH_dualprf_shift"].values[11, 20] == -2
    np.testing.assert_allclose(out["VRADH_dualprf"].values[[10, 11], 20], 3.0)
    assert np.isnan(out["VRADH_dualprf"].values[200, 5])  # no data stays no data
    assert (out["VRADH_dualprf_shift"].values != 0).sum() == 2


def test_dual_prf_does_not_wrap_a_sector():
    """The first and last ray of a sector are not neighbours."""
    az = np.arange(0.0, 60.0, 1.0)
    vel = np.full((az.size, 30), 3.0)
    vel[0, 10] += 2 * VH
    ds = xr.Dataset(
        {"VRADH": (("azimuth", "range"), vel)},
        coords={
            "azimuth": az,
            "range": 1000.0 + 250.0 * np.arange(30),
            "elevation": ("azimuth", np.full(az.size, 0.5)),
        },
    )
    out = correct_dual_prf(ds, nyquist_high=VH, nyquist_low=VL, engine="numpy")
    # the corner gate sees 6 valid gates in its 3 x 3 window: grown to 5 x 5 (12) and fixed
    assert out["VRADH_dualprf"].values[0, 10] == pytest.approx(3.0)


def test_dual_prf_volume_dict_nyquist_and_dims():
    ds, *_ = dual_prf_sweep(nray=360, ngate=120)
    flipped = ds.transpose("range", "azimuth")
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "sweep_0": ds,
            "sweep_1": flipped,
            "sweep_2": ds.drop_vars("VRADH").assign(OTHER=ds.VRADH),
        }
    )
    out = correct_dual_prf(
        tree,
        nyquist_high={"sweep_0": VH, "sweep_1": VH},
        nyquist_low={"sweep_0": VL, "sweep_1": VL},
        engine="numpy",
    )
    assert set(out.children) == {"sweep_0", "sweep_1"}
    a = out["sweep_0"].to_dataset()["VRADH_dualprf"]
    b = out["sweep_1"].to_dataset()["VRADH_dualprf"]
    assert b.dims == ("range", "azimuth")
    np.testing.assert_array_equal(a.values, b.transpose("azimuth", "range").values)
    # the accessor
    acc = tree.radarx.correct_dual_prf(
        nyquist_high={"sweep_0": VH, "sweep_1": VH},
        nyquist_low={"sweep_0": VL, "sweep_1": VL},
        engine="numpy",
    )
    xr.testing.assert_identical(acc, out)
    sweep_acc = ds.radarx.correct_dual_prf(nyquist_high=VH, nyquist_low=VL)
    assert "VRADH_dualprf" in sweep_acc


def test_dual_prf_attributes_and_provenance(dual):
    ds, *_ = dual
    out = correct_dual_prf(ds, nyquist_high=VH, nyquist_low=VL, name="V")
    attrs = out["V"].attrs
    assert attrs["nyquist_extended"] == pytest.approx(VLH)
    assert attrs["unfolding_factor"] == pytest.approx(3.0)
    assert attrs["min_points"] == 9 and attrs["max_window"] == 9
    assert attrs["units"] == "m s-1"
    assert "V_shift" in out
    assert "10.1175/1520-0426(2003)20<443:AACODP>2.0.CO;2" in attrs["radarx_references"]


def test_dual_prf_input_unchanged(dual):
    ds, *_ = dual
    before = ds.copy(deep=True)
    correct_dual_prf(ds, nyquist_high=VH, nyquist_low=VL, engine="numpy")
    xr.testing.assert_identical(ds, before)


def test_dual_prf_errors(dual):
    ds, *_ = dual
    ok = {"nyquist_high": VH, "nyquist_low": VL}
    with pytest.raises(ValueError, match="larger"):
        correct_dual_prf(ds, nyquist_high=VL, nyquist_low=VH)
    with pytest.raises(ValueError, match="positive"):
        correct_dual_prf(ds, nyquist_high=VH, nyquist_low=-1.0)
    with pytest.raises(ValueError, match="nyquist_low is needed"):
        correct_dual_prf(ds, nyquist_high=VH, nyquist_low=None)
    with pytest.raises(ValueError, match="passes"):
        correct_dual_prf(ds, passes=0, **ok)
    with pytest.raises(ValueError, match="min_points"):
        correct_dual_prf(ds, min_points=0, **ok)
    with pytest.raises(ValueError, match="max_window"):
        correct_dual_prf(ds, max_window=4, **ok)
    with pytest.raises(ValueError, match="first_ray"):
        correct_dual_prf(ds, first_ray="middle", **ok)
    with pytest.raises(ValueError, match="one value per ray"):
        correct_dual_prf(ds, prf=np.ones(5), **ok)
    with pytest.raises(ValueError, match="engine"):
        correct_dual_prf(ds, engine="fortran", **ok)
    with pytest.raises(KeyError):
        correct_dual_prf(ds, "nope", **ok)
    tree = xr.DataTree.from_dict({"/": xr.Dataset(), "sweep_0": ds})
    with pytest.raises(ValueError, match="No sweep contains"):
        correct_dual_prf(tree, "VEL", **ok)
    with pytest.raises(ValueError, match="nyquist_high is needed for 'sweep_0'"):
        correct_dual_prf(tree, nyquist_high={}, nyquist_low=VL)


def test_dual_prf_then_dealias_with_the_extended_interval(dual):
    """The output goes on to the region-based dealiasing with the extended Nyquist velocity."""
    ds, _, noisy = dual
    fixed = correct_dual_prf(ds, nyquist_high=VH, nyquist_low=VL, engine="numpy")
    again = dealias_velocity(
        fixed, "VRADH_dualprf", nyquist_velocity=VLH, engine="numpy"
    )
    ok = np.isfinite(again.values)
    assert np.mean(np.abs(again.values - noisy)[ok] < 4.0) > 0.99


# --------------------------------------------------------------------------
# confidence
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def solved():
    ds, truth = synthetic_sweep(nyquist=10.0, noise=1.0, gaps=0.05, nray=180, ngate=200)
    return ds, truth, dealias_velocity(ds, "VRADH", engine="numpy")


@pytest.mark.parametrize("engine", ENGINES)
def test_confidence_definition_on_a_small_field(engine):
    vn = 10.0
    vel = np.zeros((8, 8))
    vel[4, 4] = 12.0  # farther than Vn from all eight neighbours
    vel[1, 6] = 4.0  # nearer than Vn
    vel[6, 6] = np.nan
    ds = xr.Dataset(
        {"V": (("azimuth", "range"), vel, {"nyquist_velocity": vn})},
        coords={"azimuth": np.arange(8) * 45.0, "range": 500.0 * (1 + np.arange(8))},
    )
    conf = dealias_confidence(ds, "V", engine=engine)
    assert conf.name == "V_confidence" and conf.dtype == np.float32
    assert conf.values[4, 4] == 0.0
    # a neighbour of the spike has one of its eight neighbours too far away
    assert conf.values[4, 5] == pytest.approx(0.875)
    assert conf.values[1, 6] == 1.0
    assert np.isnan(conf.values[6, 6])
    # the gate next to the missing one has seven valid neighbours
    assert conf.values[6, 5] == 1.0
    # a smaller threshold counts the 4 m/s step as disagreement
    strict = dealias_confidence(ds, "V", threshold=0.3, engine=engine)
    assert strict.values[1, 6] == 0.0 and strict.values[1, 5] == pytest.approx(0.875)
    # a corner gate has five neighbours (the rays wrap around): enough for 5, not for 6
    assert np.isfinite(conf.values[0, 0])
    five = dealias_confidence(ds, "V", min_neighbours=5, engine=engine)
    six = dealias_confidence(ds, "V", min_neighbours=6, engine=engine)
    assert np.isfinite(five.values[0, 0]) and np.isnan(six.values[0, 0])
    # a wider window
    wide = dealias_confidence(ds, "V", window=2, min_neighbours=10, engine=engine)
    assert wide.values[4, 4] == 0.0
    assert conf.attrs["units"] == "1" and conf.attrs["window"].startswith("3 x 3")
    assert conf.attrs["nyquist_velocity"] == vn and conf.attrs["threshold"] == 1.0
    assert "radarx definition" in conf.attrs["comment"]


def test_confidence_engines_identical(solved):
    if not dealias.HAS_COMPILED_KERNEL:
        pytest.skip("compiled kernel not built")
    ds, _, dea = solved
    both = ds.assign(VRADH_dealiased=dea)
    a = dealias_confidence(both, engine="numpy")
    b = dealias_confidence(both, engine="compiled")
    np.testing.assert_array_equal(a.values, b.values)
    w = dealias_confidence(both, window=2, min_neighbours=8, engine="numpy")
    v = dealias_confidence(both, window=2, min_neighbours=8, engine="compiled")
    np.testing.assert_array_equal(w.values, v.values)


def test_confidence_marks_ambiguous_folds(solved):
    ds, truth, dea = solved
    both = ds.assign(VRADH_dealiased=dea)
    conf = dealias_confidence(both)
    assert conf.name == "VRADH_dealiased_confidence"
    ok = np.isfinite(conf.values)
    assert ok.mean() > 0.8
    assert np.median(conf.values[ok]) > 0.8
    # a gate moved by one fold interval is flagged
    wrong = dea.copy(deep=True)
    wrong.values[50, 100] += 2 * 10.0
    wrong.values[90, 40] -= 2 * 10.0
    c2 = dealias_confidence(both.assign(VRADH_dealiased=wrong))
    assert c2.values[50, 100] == 0.0 and c2.values[90, 40] == 0.0
    assert conf.values[50, 100] > 0.2  # the fixed one was fine
    # confidence is lower on gates that are wrong than on gates that are right
    err = np.abs(dea.values - truth) > 10.0
    valid = ok & np.isfinite(dea.values)
    if err[valid].any():
        assert conf.values[err & valid].mean() < conf.values[~err & valid].mean()


@pytest.mark.parametrize("engine", ENGINES)
def test_confidence_of_a_wrongly_folded_region_is_low_only_at_its_edge(engine):
    """A region that is folded as a whole looks continuous inside: the measure is local."""
    vel = np.full((60, 60), 2.0)
    vel[20:40, 20:40] += 20.0  # region off by one interval for Vn = 10
    ds = xr.Dataset(
        {"V": (("azimuth", "range"), vel, {"nyquist_velocity": 10.0})},
        coords={"azimuth": np.arange(60) * 6.0, "range": 500.0 * (1 + np.arange(60))},
    )
    conf = dealias_confidence(ds, "V", engine=engine).values
    assert (conf[22:38, 22:38] == 1.0).all()
    assert (conf[:18, :18] == 1.0).all()
    # both sides of a straight fold line: three of eight neighbours disagree
    assert conf[20, 30] == pytest.approx(5 / 8) and conf[19, 30] == pytest.approx(5 / 8)
    assert conf[30, 20] == pytest.approx(5 / 8) and conf[30, 19] == pytest.approx(5 / 8)
    assert conf[20, 20] == pytest.approx(3 / 8)  # the corner of the region


def test_dealias_velocity_with_confidence_option(solved):
    ds, _, dea = solved
    out = dealias_velocity(ds, "VRADH", confidence=True, engine="numpy")
    assert isinstance(out, xr.Dataset)
    assert set(out.data_vars) == {"VRADH_dealiased", "VRADH_dealiased_confidence"}
    np.testing.assert_array_equal(out["VRADH_dealiased"].values, dea.values)
    ref = dealias_confidence(ds.assign(VRADH_dealiased=dea), engine="numpy")
    np.testing.assert_array_equal(out["VRADH_dealiased_confidence"].values, ref.values)
    assert out["VRADH_dealiased_confidence"].dims == out["VRADH_dealiased"].dims
    # named, and on a sweep stored range first
    flipped = ds.transpose("range", "azimuth")
    named = dealias_velocity(
        flipped, "VRADH", confidence=True, name="U", engine="numpy"
    )
    assert named["U_confidence"].dims == ("range", "azimuth")
    np.testing.assert_array_equal(
        named["U_confidence"].transpose("azimuth", "range").values,
        out["VRADH_dealiased_confidence"].values,
    )
    # the default is unchanged
    plain = dealias_velocity(ds, "VRADH", engine="numpy")
    assert isinstance(plain, xr.DataArray)
    with pytest.raises(ValueError, match="products_only"):
        dealias_velocity(ds, "VRADH", confidence=True, products_only=False)


def test_dealias_velocity_with_confidence_on_a_volume():
    tree, _ = volume(elevations=(0.5, 1.5), nray=90, ngate=120)
    out = dealias_velocity(tree, "VRADH", confidence=True, engine="numpy")
    for name in ("sweep_0", "sweep_1"):
        node = out[name].to_dataset()
        assert set(node.data_vars) == {"VRADH_dealiased", "VRADH_dealiased_confidence"}
        assert np.nanmax(node["VRADH_dealiased_confidence"].values) <= 1.0
    # the same through the accessor, and the confidence of the merged volume
    acc = tree.radarx.dealias("VRADH", confidence=True, engine="numpy")
    xr.testing.assert_identical(acc, out)
    merged = tree.radarx.assign(out)
    again = dealias_confidence(merged, "VRADH_dealiased", engine="numpy")
    np.testing.assert_array_equal(
        again["sweep_0"].to_dataset()["VRADH_dealiased_confidence"].values,
        out["sweep_0"].to_dataset()["VRADH_dealiased_confidence"].values,
    )
    acc2 = merged.radarx.dealias_confidence(engine="numpy")
    xr.testing.assert_identical(acc2, dealias_confidence(merged, engine="numpy"))


def test_confidence_errors(solved):
    ds, _, dea = solved
    both = ds.assign(VRADH_dealiased=dea)
    with pytest.raises(ValueError, match="window"):
        dealias_confidence(both, window=0)
    with pytest.raises(ValueError, match="min_neighbours"):
        dealias_confidence(both, min_neighbours=9)
    with pytest.raises(ValueError, match="min_neighbours"):
        dealias_confidence(both, min_neighbours=0)
    with pytest.raises(ValueError, match="threshold"):
        dealias_confidence(both, threshold=0.0)
    with pytest.raises(ValueError, match="engine"):
        dealias_confidence(both, engine="fortran")
    with pytest.raises(ValueError, match="Nyquist"):
        dealias_confidence(ds.drop_vars("nyquist_velocity"), "VRADH")
    tree = xr.DataTree.from_dict({"/": xr.Dataset(), "sweep_0": ds})
    with pytest.raises(ValueError, match="No sweep contains"):
        dealias_confidence(tree, "VRADH_dealiased")
