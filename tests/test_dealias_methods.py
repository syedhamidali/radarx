#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for R2D2 dealiasing and the fold margin (``radarx.retrieve.dealias_methods``)."""

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401  registers the accessors
from radarx.retrieve import _dealias_methods_numpy as spec
from radarx.retrieve import (
    dealias_ambiguity,
    dealias_methods,
    dealias_r2d2,
    dealias_velocity,
)

from .test_dealias import (
    correct_fraction,
    nexrad_volume,
    residual_jumps,
    synthetic_sweep,
    volume,
    wind_profile,
)

needs_kernel = pytest.mark.skipif(
    not dealias_methods.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)
ENGINES = ["numpy", pytest.param("compiled", marks=needs_kernel)]
Q = spec.VEL_Q


def folded_field(rnd, nray, ngate, nyquist, noise=0.5, gaps=0.05, scale=2.6):
    """A smooth radial velocity field up to ``scale`` Nyquist velocities, folded."""
    az = np.linspace(0, 2 * np.pi, nray, endpoint=False)[:, None]
    rg = np.linspace(0, 1, ngate)[None, :]
    truth = scale * nyquist * np.sin(az + 0.7) * (0.3 + 0.7 * rg) + 3 * np.cos(6 * rg)
    measured = truth + noise * rnd.standard_normal(truth.shape)
    folded = (measured + nyquist) % (2 * nyquist) - nyquist
    folded[rnd.random(truth.shape) < gaps] = np.nan
    return folded, truth


def sweep_of(values, nyquist, nray=None, name="VRADH", elevation=0.5):
    nray, ngate = values.shape
    return xr.Dataset(
        {name: (("azimuth", "range"), values, {"units": "m s-1"})},
        coords={
            "azimuth": ("azimuth", np.arange(nray) * 360.0 / nray + 0.25),
            "range": ("range", 1000.0 + 250.0 * np.arange(ngate)),
            "elevation": ("azimuth", np.full(nray, elevation)),
            "nyquist_velocity": ((), nyquist),
        },
    )


def params(nyquist, **change):
    p = {
        "beta": int(0.8 * nyquist * Q),
        "gamma": 0.5 * nyquist * Q,
        "min_overlap": 5,
        "buffer_half": 2,
        "thresholds": [int(s * nyquist * Q) for s in (0.3, 0.5, 0.7, 0.9)],
        "window_half": 5,
        "window_min": 3,
        "max_iterations": 10,
        "patience": 2,
        "ref_fallback": False,
    }
    p.update(change)
    return p


def run_both(vel, nyquist, guides, wrap, **change):
    """Folds of the NumPy specification and of the kernel, and the infos."""
    q = dealias_methods._quantize(vel, nyquist)
    g = {
        k: None if v is None else dealias_methods._quantize(v)
        for k, v in guides.items()
    }
    p = params(nyquist, **change)
    nyq2 = int(round(2 * nyquist * Q))
    f_spec, i_spec = spec.r2d2_sweep(q, nyq2, g, wrap, p)
    out = [(f_spec, i_spec)]
    if dealias_methods.HAS_COMPILED_KERNEL:
        empty = np.zeros((0, 0), dtype=np.int32)
        f_k, info = dealias_methods._kernel.r2d2_sweep(
            q,
            empty if g["ref"] is None else g["ref"],
            empty if g["above"] is None else g["above"],
            empty if g["prev"] is None else g["prev"],
            wrap,
            nyq2,
            p["beta"],
            p["gamma"],
            p["min_overlap"],
            p["buffer_half"],
            p["thresholds"],
            p["window_half"],
            p["window_min"],
            p["max_iterations"],
            p["patience"],
            p["ref_fallback"],
            2,
        )
        out.append(
            (f_k, dict(zip(("iterations", "best", "unresolved", "buffer"), info)))
        )
    return out


# --------------------------------------------------------------------------
# R2D2
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "nyquist, noise, minimum",
    [(30.0, 1.0, 1.0), (15.0, 1.0, 1.0), (10.0, 1.0, 1.0), (7.0, 1.0, 0.999)],
)
def test_synthetic_accuracy_with_the_default_first_guess(
    engine, nyquist, noise, minimum
):
    ds, truth = synthetic_sweep(nyquist=nyquist, noise=noise)
    out = dealias_r2d2(ds, "VRADH", engine=engine)
    assert correct_fraction(out, ds, truth) >= minimum
    assert out.name == "VRADH_dealiased"
    assert out.attrs["dealias_first_guess"] == "region"


@pytest.mark.parametrize("engine", ENGINES)
def test_a_reference_and_a_wind_profile_are_first_guesses(engine):
    ds, truth = synthetic_sweep(nyquist=10.0, noise=1.0)
    ref = xr.DataArray(truth + 2.0, dims=("azimuth", "range"))
    out = dealias_r2d2(ds, "VRADH", reference=ref, engine=engine)
    assert correct_fraction(out, ds, truth) > 0.999
    assert out.attrs["dealias_first_guess"] == "reference"
    wp = dealias_r2d2(ds, "VRADH", wind_profile=wind_profile(), engine=engine)
    assert wp.attrs["dealias_first_guess"] == "wind_profile"
    # the profile is a uniform wind: far from the truth near the vortex, but the
    # first guess only has to be within a Nyquist interval of the regions
    both = dealias_r2d2(
        ds, "VRADH", reference=ref, wind_profile=wind_profile(), engine=engine
    )
    np.testing.assert_array_equal(both.values, out.values)
    assert both.attrs["dealias_first_guess"] == "reference+wind_profile"


@pytest.mark.parametrize("engine", ENGINES)
def test_a_reference_with_other_coordinates_is_mapped(engine):
    ds, truth = synthetic_sweep(nyquist=10.0, noise=1.0, nray=180, ngate=200)
    half = ds.isel(azimuth=slice(0, None, 2))
    ref = xr.DataArray(
        truth[::2],
        dims=("azimuth", "range"),
        coords={"azimuth": half.azimuth.values, "range": half.range.values},
    )
    out = dealias_r2d2(ds, "VRADH", reference=ref, engine=engine)
    assert correct_fraction(out, ds, truth) > 0.999
    with pytest.raises(ValueError, match="without coordinates"):
        dealias_r2d2(
            ds, "VRADH", reference=xr.DataArray(truth[:5], dims=("azimuth", "range"))
        )
    with pytest.raises(ValueError, match="2-D"):
        dealias_r2d2(ds, "VRADH", reference=ref.expand_dims(time=2))


def test_engines_identical_on_the_synthetic_sweeps():
    if not dealias_methods.HAS_COMPILED_KERNEL:
        pytest.skip("compiled kernel not built")
    for nyquist, noise in ((10.0, 1.0), (7.0, 2.0)):
        ds, _ = synthetic_sweep(nyquist=nyquist, noise=noise, nray=180, ngate=200)
        a = dealias_r2d2(ds, "VRADH", engine="numpy")
        b = dealias_r2d2(ds, "VRADH", engine="compiled")
        np.testing.assert_array_equal(a.values, b.values)
        for key in ("r2d2_iterations", "r2d2_best_iteration", "r2d2_unresolved_gates"):
            assert a.attrs[key] == b.attrs[key]


@pytest.mark.parametrize("wrap", [True, False])
@pytest.mark.parametrize("fallback", [True, False])
@pytest.mark.parametrize("seed", range(6))
def test_kernel_and_specification_agree_with_all_guidance(seed, fallback, wrap):
    rnd = np.random.default_rng(seed)
    nyquist = float(rnd.choice([8.0, 12.5, 20.0]))
    nray, ngate = (60, 70) if seed % 2 else (45, 90)
    vel, truth = folded_field(rnd, nray, ngate, nyquist, noise=1.0 + seed % 3)
    guides = {
        "ref": truth + rnd.normal(0, 0.4 * nyquist, truth.shape),
        "above": np.where(rnd.random(truth.shape) < 0.5, truth, np.nan),
        "prev": None if seed % 3 else truth + rnd.normal(0, 2.0, truth.shape),
    }
    if seed == 5:
        guides["above"] = None
    results = run_both(vel, nyquist, guides, wrap, ref_fallback=fallback)
    for folds, info in results[1:]:
        np.testing.assert_array_equal(folds, results[0][0])
        assert info == results[0][1]


def test_kernel_and_specification_agree_without_guidance_and_on_tiny_sweeps():
    rnd = np.random.default_rng(1)
    vel, _ = folded_field(rnd, 40, 50, 10.0)
    none = {"ref": None, "above": None, "prev": None}
    for wrap in (True, False):
        results = run_both(vel, 10.0, none, wrap)
        assert not results[0][0].any()  # nothing to decide the folds with
        for folds, info in results[1:]:
            np.testing.assert_array_equal(folds, results[0][0])
    empty = np.full((12, 10), np.nan)
    for folds, info in run_both(empty, 10.0, none, True):
        assert not folds.any() and info["unresolved"] == 0
    one = np.array([[3.0, np.nan, -8.0, 9.5]])
    ref = {"ref": np.array([[3.0, 0.0, 12.0, -10.5]]), "above": None, "prev": None}
    results = run_both(one, 10.0, ref, False)
    for folds, _ in results[1:]:
        np.testing.assert_array_equal(folds, results[0][0])


def test_thread_count_does_not_change_the_result():
    if not dealias_methods.HAS_COMPILED_KERNEL:
        pytest.skip("compiled kernel not built")
    ds, _ = synthetic_sweep(nyquist=8.0, noise=2.0, nray=240, ngate=300)
    a = dealias_r2d2(ds, "VRADH", n_threads=1)
    b = dealias_r2d2(ds, "VRADH", n_threads=7)
    np.testing.assert_array_equal(a.values, b.values)


def couplet_sweep(nyquist=30.0):
    """The velocity couplet next to a folded patch of the vortex tests of the region method."""
    ray, gate = np.meshgrid(np.arange(120), np.arange(100), indexing="ij")

    def box(lo, hi, x, width=1.5):
        return 0.5 * (np.tanh((x - lo) / width) - np.tanh((x - hi) / width))

    env = np.full(ray.shape, -28.5)
    u = env + (-34.0 - -28.5) * box(54, 66, ray) * box(38, 46, gate)
    rho = np.hypot((ray - 60) / 4.0, (gate - 50) / 6.0)
    after = 0.5 * (1 - np.tanh((ray - 61.0) / 0.6))
    rim = -12.0 + (28.5 + 12.0) * after
    bowl = -12.0 + (rim + 12.0) * np.minimum(rho, 1.0) ** 2
    truth = np.where(rho <= 1.0, bowl, u)
    measured = (truth + nyquist) % (2 * nyquist) - nyquist
    ds = sweep_of(measured, nyquist)
    ds = ds.assign_coords(azimuth=("azimuth", (np.arange(120) + 0.5) * 3.0))
    return ds, truth, rho <= 1.0, (truth < -nyquist) & ~(rho <= 1.0)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_velocity_couplet_next_to_a_fold_is_mostly_kept(engine):
    ds, truth, pocket, patch = couplet_sweep()
    region = dealias_velocity(ds, engine=engine).values
    out = dealias_r2d2(ds, engine=engine).values
    wrong_region = np.count_nonzero(np.abs(region - truth) > 30.0)
    wrong_r2d2 = np.count_nonzero(np.abs(out - truth) > 30.0)
    assert wrong_region > 50
    assert wrong_r2d2 < wrong_region / 3
    assert np.all(np.abs(out - truth)[patch] < 1.0)  # the fold line is resolved
    guess = xr.DataArray(truth + 3.0, dims=("azimuth", "range"))
    given = dealias_r2d2(ds, reference=guess, engine=engine).values
    assert np.count_nonzero(np.abs(given - truth) > 30.0) < wrong_region / 3


@pytest.mark.parametrize("engine", ENGINES)
def test_volume_goes_from_the_top_and_follows_the_sweep_above(engine):
    dtree, truths = volume()
    out = dealias_r2d2(dtree, wind_profile=wind_profile(), engine=engine)
    assert set(out.children) == set(truths)
    for name, truth in truths.items():
        ds = dtree[name].to_dataset()
        da = out[name]["VRADH_dealiased"]
        assert correct_fraction(da, ds, truth) > 0.99
    # a sweep alone, guided by nothing but the profile, is the same as in the volume
    # when the sweep above does not change its decisions
    top = max(truths, key=lambda n: float(dtree[n].elevation.mean()))
    alone = dealias_r2d2(dtree[top], wind_profile=wind_profile(), engine=engine)
    np.testing.assert_array_equal(alone.values, out[top]["VRADH_dealiased"].values)


def test_volume_default_first_guess_and_missing_fields():
    dtree, truths = volume(elevations=(0.5, 1.5), ngate=200)
    dtree["sweep_1"] = dtree["sweep_1"].to_dataset().drop_vars("VRADH")
    out = dealias_r2d2(dtree)
    assert list(out.children) == ["sweep_0"]
    with pytest.raises(ValueError, match="No sweep contains"):
        dealias_r2d2(dtree.drop_nodes(["sweep_0"]), "VRADH")


def test_previous_volume_guides_the_sweeps():
    dtree, truths = volume(elevations=(0.5, 1.5), ngate=200)
    first = dealias_r2d2(dtree)
    again = dealias_r2d2(dtree, previous=first)
    for name, truth in truths.items():
        ds = dtree[name].to_dataset()
        assert correct_fraction(again[name]["VRADH_dealiased"], ds, truth) > 0.99
    # for a sweep: a DataArray, a Dataset holding the field, or a tree
    ds = dtree["sweep_0"].to_dataset()
    prev = first["sweep_0"]["VRADH_dealiased"]
    a = dealias_r2d2(ds, previous=prev)
    b = dealias_r2d2(ds, previous=first["sweep_0"].to_dataset())
    np.testing.assert_array_equal(a.values, b.values)
    c = dealias_r2d2(ds, previous=first)  # a tree cannot name the sweep: ignored
    assert correct_fraction(c, ds, truths["sweep_0"]) > 0.99
    d = dealias_r2d2(dtree, previous=first.drop_nodes(["sweep_1"]))
    assert set(d.children) == {"sweep_0", "sweep_1"}
    # the previous volume is mapped to the nearest gate
    coarse = first["sweep_0"].to_dataset().isel(azimuth=slice(0, None, 2))
    assert (
        correct_fraction(
            dealias_r2d2(ds, previous=coarse["VRADH_dealiased"]),
            ds,
            truths["sweep_0"],
        )
        > 0.99
    )


def test_the_first_guess_fills_in_where_the_other_guidance_is_missing():
    rnd = np.random.default_rng(3)
    vel, truth = folded_field(rnd, 60, 80, 10.0)
    above = np.where(np.arange(80)[None, :] < 40, truth, np.nan)
    above = np.broadcast_to(above, truth.shape).copy()
    guides = {"ref": truth, "above": above, "prev": None}
    for wrap in (True, False):
        for fallback in (True, False):
            for folds, _ in run_both(vel, 10.0, guides, wrap, ref_fallback=fallback):
                out = np.where(np.isnan(vel), np.nan, vel + 20.0 * folds)
                ok = np.isfinite(out)
                assert np.mean(np.abs(out - truth)[ok] > 10.0) < 0.01


def test_options_are_checked():
    ds, _ = synthetic_sweep(nyquist=10.0, nray=36, ngate=40)
    for bad in (
        {"shear_threshold": 0},
        {"buffer": 4},
        {"buffer": 0},
        {"spread_threshold": -1},
        {"thresholds": ()},
        {"thresholds": (0.3, 1.5)},
        {"window": 10},
        {"window": 1},
        {"max_iterations": 0},
        {"patience": 0.5},
    ):
        with pytest.raises(ValueError):
            dealias_r2d2(ds, **bad)
    with pytest.raises(ValueError, match="engine"):
        dealias_r2d2(ds, engine="gpu")
    with pytest.raises(ValueError, match="Nyquist"):
        dealias_r2d2(ds.drop_vars("nyquist_velocity"))
    with pytest.raises(KeyError):
        dealias_r2d2(ds, "NOT_A_FIELD")
    dtree, _ = volume(elevations=(0.5, 1.5), ngate=40)
    with pytest.raises(ValueError, match="only supported for a single sweep"):
        dealias_r2d2(
            dtree, reference=xr.DataArray(np.zeros((360, 40)), dims=("a", "b"))
        )
    stack = xr.concat([ds, ds], "time")
    with pytest.raises(ValueError, match="needs sweeps on"):
        dealias_r2d2(stack)


def test_compiled_engine_without_the_kernel_is_an_error(monkeypatch):
    monkeypatch.setattr(dealias_methods, "HAS_COMPILED_KERNEL", False)
    ds, _ = synthetic_sweep(nyquist=10.0, nray=36, ngate=40)
    with pytest.raises(ImportError, match="kernel"):
        dealias_r2d2(ds, engine="compiled")
    with pytest.raises(ImportError, match="kernel"):
        dealias_ambiguity(ds, "VRADH", 10.0, engine="compiled")
    out = dealias_r2d2(ds, engine="auto")  # falls back to the NumPy reference
    assert np.isfinite(out.values).any()


def test_dealias_velocity_dispatches_method_r2d2():
    ds, truth = synthetic_sweep(nyquist=10.0, noise=1.0, nray=180, ngate=200)
    direct = dealias_r2d2(ds, "VRADH", wind_profile=wind_profile())
    via = dealias_velocity(ds, "VRADH", method="r2d2", wind_profile=wind_profile())
    xr.testing.assert_equal(direct, via)
    assert direct.attrs["nyquist_velocity"] == via.attrs["nyquist_velocity"]
    assert "R2D2" in via.attrs["comment"] and "Feldmann" in via.attrs["radarx_method"]
    named = dealias_velocity(ds, "VRADH", method="r2d2", name="v")
    assert named.name == "v"
    with pytest.raises(ValueError, match="confidence"):
        dealias_velocity(ds, "VRADH", method="r2d2", confidence=True)
    with pytest.raises(ValueError, match="products_only"):
        dealias_velocity(ds, "VRADH", method="r2d2", products_only=False)
    with pytest.raises(ValueError, match="method must be"):
        dealias_velocity(ds, "VRADH", method="nope")


def test_accessors_and_output_metadata():
    ds, truth = synthetic_sweep(nyquist=10.0, noise=1.0, nray=120, ngate=100)
    out = ds.radarx.dealias_r2d2("VRADH")
    xr.testing.assert_identical(out, dealias_r2d2(ds, "VRADH"))
    assert out.attrs["nyquist_velocity"] == 10.0
    assert out.attrs["units"] == "m s-1"
    assert "Feldmann" in out.attrs["comment"]
    assert out.attrs["radarx_references"].startswith("10.1175/JTECH-D-20-0054.1")
    assert out.dtype == ds.VRADH.dtype
    dtree, _ = volume(elevations=(0.5, 1.5), ngate=60)
    tree_out = dtree.radarx.dealias_r2d2()
    assert set(tree_out.children) == {"sweep_0", "sweep_1"}
    merged = dtree.radarx.assign(tree_out)
    assert "VRADH_dealiased" in merged["sweep_0"].data_vars
    # a sweep node of a volume is a sweep; an integer field gives floats
    node = dealias_r2d2(dtree["sweep_0"])
    assert node.dims == ("azimuth", "range")
    ints = ds.assign(VRADH=(ds.VRADH.dims, np.nan_to_num(ds.VRADH.values).astype("i2")))
    assert dealias_r2d2(ints, "VRADH").dtype == np.float64
    # extra gates: transposed fields keep their dimension order
    t = ds.assign(VRADH=ds.VRADH.transpose("range", "azimuth"))
    assert dealias_r2d2(t, "VRADH").dims == ("range", "azimuth")


def test_flags_and_gaps_are_not_data():
    ds, _ = synthetic_sweep(nyquist=10.0, noise=1.0, nray=90, ngate=80)
    v = ds.VRADH.values.copy()
    v[3, 4] = -63.5  # a flag beyond the Nyquist velocity
    v[5, 6] = np.inf
    out = dealias_r2d2(ds.assign(VRADH=(ds.VRADH.dims, v)), "VRADH")
    assert np.isnan(out.values[3, 4]) and np.isnan(out.values[5, 6])
    empty = ds.assign(VRADH=(ds.VRADH.dims, np.full(v.shape, np.nan)))
    assert np.isnan(dealias_r2d2(empty, "VRADH").values).all()


# --------------------------------------------------------------------------
# fold margin
# --------------------------------------------------------------------------


def two_regions(jump, nyquist=10.0, nray=8, ngate=10):
    """Left half at 0, right half ``jump`` Nyquist velocities higher."""
    v = np.zeros((nray, ngate))
    v[:, ngate // 2 :] = jump * nyquist
    return sweep_of(v, nyquist, name="VRADH_dealiased")


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("jump", [0.0, 0.5, 1.0, 1.5, -1.2])
def test_the_margin_of_two_regions_is_one_minus_the_jump(engine, jump):
    # d = jump / 2 for every boundary pair of both regions, m = 1 - |jump|
    ds = two_regions(jump)
    if jump == 0.0:
        ds["VRADH_dealiased"].values[:, 5:] = 0.0
    out = dealias_ambiguity(ds, nyquist_velocity=10.0, engine=engine)
    margin = out["VRADH_dealiased_fold_margin"].values
    if jump == 0.0:  # one region only: no boundary
        assert np.all(out["VRADH_dealiased_fold_flag"].values == 4)
        assert np.all(margin == 0.0)
        return
    expected = np.clip(1.0 - abs(jump), -1.0, 1.0)
    np.testing.assert_allclose(margin, expected, atol=1e-6)
    flags = out["VRADH_dealiased_fold_flag"].values
    pairs = 8  # boundary pairs of each region
    assert pairs < 10  # short boundary with the default min_boundary
    weak = expected < 0.25
    assert np.all((flags & 1).astype(bool) == weak)
    assert np.all((flags & 2) == 2)
    assert np.all(out["VRADH_dealiased_region_size"].values == 40)


@pytest.mark.parametrize("engine", ENGINES)
def test_an_island_and_a_block_one_fold_off(engine):
    ds, truth = synthetic_sweep(nyquist=10.0, noise=1.0, nray=360, ngate=400)
    good = dealias_velocity(ds, "VRADH").values
    v = good.copy()
    v[:, 380:] = np.nan
    v[200:230, 100:140] += 20.0  # connected to the field, one fold off
    v[40:70, 240:270] = np.nan  # an island, one fold off, with a gap around it
    v[52:58, 252:258] = truth[52:58, 252:258] + 20.0
    da = ds.VRADH.copy(data=v).rename("VRADH_dealiased")
    out = dealias_ambiguity(da.to_dataset(), nyquist_velocity=10.0, engine=engine)
    margin = out["VRADH_dealiased_fold_margin"].values
    flags = out["VRADH_dealiased_fold_flag"].values
    size = out["VRADH_dealiased_region_size"].values
    assert np.nanmean(margin[205:225, 105:135]) < -0.5
    block = flags[210:220, 110:130]
    assert np.mean((block & 1) == 1) > 0.9  # but for gates alone in a gap
    assert np.mean((flags[52:58, 252:258] & 4) == 4) == 1.0
    assert np.all(margin[52:58, 252:258] == 0.0)
    assert np.all(size[52:58, 252:258] == 36)
    assert np.nanmedian(margin) > 0.4
    assert np.isnan(margin[:, 380:]).all() and np.all(flags[:, 380:] == 0)
    assert np.all(size[:, 380:] == 0)
    assert margin.dtype == np.float32 and flags.dtype == np.uint8


@pytest.mark.parametrize("engine", ENGINES)
def test_compare_flags_the_gates_that_differ_by_half_a_fold(engine):
    ds = two_regions(0.4)
    other = ds["VRADH_dealiased"].copy()
    other.values[2, 1] += 10.0
    other.values[3, 2] += 9.9
    third = ds["VRADH_dealiased"].copy()
    third.values[4, 3] -= 31.0  # 1.55 folds: rounds to 2
    ds = ds.assign(OTHER=other, THIRD=third)
    for compare in ("OTHER", other, ["OTHER"], [other]):
        out = dealias_ambiguity(
            ds, "VRADH_dealiased", 10.0, compare=compare, engine=engine
        )
        flags = out["VRADH_dealiased_fold_flag"].values
        spread = out["VRADH_dealiased_fold_spread"].values
        assert flags[2, 1] & 8 and not flags[3, 2] & 8
        assert np.count_nonzero(flags & 8) == 1
        assert spread[2, 1] == 1 and spread.sum() == 1 and spread.dtype == np.int8
    out = dealias_ambiguity(
        ds, "VRADH_dealiased", 10.0, compare=["OTHER", "THIRD"], engine=engine
    )
    spread = out["VRADH_dealiased_fold_spread"].values
    assert spread[2, 1] == 1 and spread[4, 3] == 2 and spread.sum() == 3
    assert np.count_nonzero(out["VRADH_dealiased_fold_flag"].values & 8) == 2
    none = dealias_ambiguity(ds, "VRADH_dealiased", 10.0, engine=engine)
    assert "VRADH_dealiased_fold_spread" not in none


def test_engines_identical_on_random_fields():
    if not dealias_methods.HAS_COMPILED_KERNEL:
        pytest.skip("compiled kernel not built")
    rnd = np.random.default_rng(5)
    for nray, ngate, nyquist in ((90, 70, 10.0), (360, 100, 25.0), (2, 30, 8.0)):
        vel, truth = folded_field(rnd, nray, ngate, nyquist, noise=1.5, gaps=0.15)
        other = truth + rnd.normal(0, 3, truth.shape)
        dealiased = np.where(np.isnan(vel), np.nan, truth)
        ds = sweep_of(dealiased, nyquist, name="VD").assign(
            OT=(("azimuth", "range"), other), O2=(("azimuth", "range"), other + 14.0)
        )
        compare = ["OT", "O2"]
        a = dealias_ambiguity(ds, "VD", compare=compare, engine="numpy")
        b = dealias_ambiguity(ds, "VD", compare=compare, engine="compiled", n_threads=3)
        xr.testing.assert_identical(a, b)


def test_a_full_circle_wraps_the_regions_around_north():
    v = np.zeros((36, 20))
    v[[0, 1, 34, 35], 10:] = 8.0
    out = dealias_ambiguity(sweep_of(v, 10.0, name="VD"), "VD", 10.0)
    size = out["VD_region_size"].values
    assert size[0, 12] == size[35, 12] == 40  # one region across the seam
    assert size[10, 12] == 36 * 20 - 40


def test_a_sector_does_not_wrap_the_regions():
    v2 = np.zeros((36, 20))
    v2[[0, 1, 34, 35], 10:] = 8.0
    ds = sweep_of(v2, 10.0, name="VD").isel(azimuth=slice(0, 30))  # not a full circle
    ds = ds.assign(VD=ds.VD.copy(data=np.zeros((30, 20))))
    ds["VD"].values[[0, 29], 10:] = 8.0
    out = dealias_ambiguity(ds, "VD", 10.0)
    assert out["VD_region_size"].values[0, 12] == 10
    assert out["VD_region_size"].values[29, 12] == 10


def test_ambiguity_default_field_volume_and_options():
    dtree, truths = volume(elevations=(0.5, 1.5), ngate=120)
    dea = dealias_velocity(dtree)
    merged = dtree.radarx.assign(dea)
    out = merged.radarx.dealias_ambiguity()
    assert set(out.children) == {"sweep_0", "sweep_1"}
    assert "VRADH_dealiased_fold_margin" in out["sweep_0"].data_vars
    named = dealias_ambiguity(merged, name="amb", compare="VRADH")
    assert "amb_fold_flag" in named["sweep_1"].data_vars
    flag = out["sweep_0"]["VRADH_dealiased_fold_flag"]
    assert flag.attrs["flag_meanings"].split() == [
        "weak_margin",
        "short_boundary",
        "no_boundary",
        "differs_from_compare",
    ]
    assert list(flag.attrs["flag_masks"]) == [1, 2, 4, 8]
    sweep = merged["sweep_0"].to_dataset()
    one, node = (
        dealias_ambiguity(sweep, name="amb"),
        dealias_ambiguity(merged["sweep_0"], name="amb"),
    )
    for key in one.data_vars:
        np.testing.assert_array_equal(one[key].values, node[key].values)
    with pytest.raises(ValueError, match="names of fields"):
        dealias_ambiguity(merged, compare=sweep["VRADH"])
    for bad in (
        {"threshold": 0},
        {"min_boundary": 0},
        {"min_boundary": 2.5},
        {"min_margin": 2},
    ):
        with pytest.raises(ValueError):
            dealias_ambiguity(sweep, **bad)
    with pytest.raises(ValueError, match="No sweep contains"):
        dealias_ambiguity(merged, "VELOCITY_NOT_THERE")


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kgwx(tmp_path_factory):
    """Strongly aliased low-level jet (KGWX, 30 March 2022), from AWS."""
    from radarx.io.aws_data import download_file

    try:
        path = download_file(
            "unidata-nexrad-level2",
            "2022/03/30/KGWX/KGWX20220330_234639_V06",
            str(tmp_path_factory.mktemp("nexrad")),
        )
    except Exception as err:  # noqa: BLE001  # pragma: no cover - network
        pytest.skip(f"NEXRAD data not available: {err}")
    dtree = nexrad_volume(path)
    for name in dtree.children:
        ds = dtree[name].to_dataset()
        if "VRADH" in ds:
            dtree[name] = ds.assign(VRADH=ds["VRADH"].where(ds["VRADH"] > -63.9))
    return dtree


def test_nexrad_volume_r2d2_and_the_margin(kgwx):
    dtree = kgwx
    names = [n for n in dtree.children if "VRADH" in dtree[n].data_vars]
    out = dealias_r2d2(dtree)
    assert [n for n in out.children] == sorted(names)
    region = dealias_velocity(dtree)
    for name in names:
        vn = float(dtree[name]["nyquist_velocity"])
        ours = out[name]["VRADH_dealiased"].values
        theirs = region[name]["VRADH_dealiased"].values
        ok = np.isfinite(ours) & np.isfinite(theirs)
        # the same folds as the region method on almost every gate
        assert np.mean(np.abs(ours - theirs)[ok] > vn) < 0.02, name
        assert residual_jumps(ours, vn) < 0.005, name
    merged = dtree.radarx.assign(out)
    amb = dealias_ambiguity(merged, "VRADH_dealiased")
    for name in names[:4]:  # the high sweeps are often one region
        flag = amb[name]["VRADH_dealiased_fold_flag"].values
        margin = amb[name]["VRADH_dealiased_fold_margin"].values
        valid = np.isfinite(margin)
        assert np.mean(margin[valid] > 0.25) > 0.8
        assert np.mean(flag[valid] & 1 == 1) < 0.1
