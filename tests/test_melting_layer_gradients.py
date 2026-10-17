#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the melting layer from profile gradients
==================================================

``melting_layer_gradient`` follows the steps of Allabakash et al. (2019), Sect.
2.2. The synthetic profiles are built from Gaussian anomalies whose positions
decide which step gives the top; every result is compared with a plain loop that
states the steps of the paper and uses ``numpy.gradient``. The compiled kernel and
the NumPy implementation must agree, also with gaps, irregular heights and extra
dimensions.
"""

import importlib

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.retrieve import (
    mask_above_melting_layer,
    melting_layer_gradient,
    melting_layer_rhi,
    rhi_qvp,
)

from .provenance_helpers import without_provenance
from .test_qvp_variants import ALT, make_sweep

mg = importlib.import_module("radarx.retrieve.melting_layer_gradients")

ENGINES = ["numpy"] + (["compiled"] if mg.HAS_COMPILED_KERNEL else [])
H = np.arange(0.0, 8000.0, 100.0)


def bump(centre, width, h=H):
    return np.exp(-(((h - centre) / width) ** 2))


def profile(z_centre=3800.0, zdr_centre=3900.0, rho_centre=3800.0, h=H):
    """Z, ZDR and rhohv with an anomaly each, at the given heights."""
    return (
        25.0 + 10.0 * bump(z_centre, 300.0, h),
        0.4 + 1.5 * bump(zdr_centre, 250.0, h),
        0.99 - 0.15 * bump(rho_centre, 250.0, h),
    )


def dataset(z, zdr, rho, h=H, dims=("height",), **coords):
    return xr.Dataset(
        {
            "DBZH": (dims, z, {"units": "dBZ"}),
            "ZDR": (dims, zdr, {"units": "dB"}),
            "RHOHV": (dims, rho, {"units": "1"}),
        },
        coords={"height": h, **coords},
    )


def oracle(h, z, zdr, rho):
    """The steps of the paper in plain Python: (bottom, top, step) or None."""
    az, ad, ar = (np.abs(x / np.max(x)) for x in (z, zdr, rho))
    products = [az * (1 - ar), ad * (1 - ar), az * ad * (1 - ar)]
    grads = [np.gradient(p, h) for p in products]

    def together(indices):
        for i in range(3):
            for j in range(i + 1, 3):
                if indices[i] == indices[j]:
                    return indices[i]
        return None

    bottom = together([int(np.argmax(g)) for g in grads])
    c = together([int(np.argmin(g)) for g in grads])
    if bottom is None or c is None:
        return None
    zmin = int(np.argmin(np.gradient(az, h)))
    rmax = int(np.argmax(np.gradient(ar, h)))
    if zmin == c and rmax == c:
        top, step = c, 3
    elif zmin > c or rmax > c:
        top, step = max(zmin, rmax), 4
    else:
        top, step = c + 1, 5
    if top >= h.size or h[top] <= h[bottom]:
        return None
    return h[bottom], h[top], step


# the three steps, a profile without a layer
CASES = {
    "step3": ((3800.0, 3900.0, 3800.0), (3700.0, 4000.0, 3)),
    "step4": ((3800.0, 3900.0, 4150.0), (3900.0, 4300.0, 4)),
    "step5": ((3800.0, 3800.0, 4400.0), (4200.0, 4700.0, 5)),
    "none": ((3800.0, 3900.0, 4400.0), None),
}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", sorted(CASES))
def test_the_steps_of_the_paper(engine, name):
    centres, expected = CASES[name]
    z, zdr, rho = profile(*centres)
    out = melting_layer_gradient(dataset(z, zdr, rho), engine=engine)
    ref = oracle(H, z, zdr, rho)
    assert ref == expected
    if expected is None:
        assert out.melting_layer_flag == 0 and out.melting_layer_top_step == 0
        assert np.isnan(float(out.melting_layer_top))
        assert np.isnan(float(out.melting_layer_bottom))
        assert np.isnan(float(out.melting_layer_thickness))
        return
    bottom, top, step = expected
    assert float(out.melting_layer_bottom) == bottom
    assert float(out.melting_layer_top) == top
    assert float(out.melting_layer_thickness) == top - bottom
    assert out.melting_layer_flag == 1 and out.melting_layer_top_step == step
    assert 0.0 < top - bottom < 1000.0


@pytest.mark.parametrize("engine", ENGINES)
def test_the_layer_does_not_depend_on_the_scale_of_the_height_or_the_values(engine):
    z, zdr, rho = profile(3800.0, 3900.0, 4150.0)
    a = melting_layer_gradient(dataset(z, zdr, rho), engine=engine)
    # the normalisation removes the scale of the variables
    b = melting_layer_gradient(dataset(2.0 * z, 3.0 * zdr, rho), engine=engine)
    assert float(a.melting_layer_top) == float(b.melting_layer_top)
    assert float(a.melting_layer_bottom) == float(b.melting_layer_bottom)
    # the heights are metres above sea level, whatever their spacing
    c = melting_layer_gradient(dataset(z, zdr, rho, h=H + 500.0), engine=engine)
    assert float(c.melting_layer_top) == float(a.melting_layer_top) + 500.0


def noisy(seed, n=60, nan=0.0):
    r = np.random.default_rng(seed)
    z = 20.0 + r.normal(0.0, 3.0, n)
    zdr = 0.5 + np.abs(r.normal(0.0, 0.5, n))
    rho = 0.98 - np.abs(r.normal(0.0, 0.03, n))
    for x in (z, zdr, rho):
        x[r.random(n) < nan] = np.nan
    return z, zdr, rho


@pytest.mark.parametrize("seed", range(8))
def test_random_profiles_agree_with_the_loop(seed):
    h = np.arange(0.0, 6000.0, 100.0)
    z, zdr, rho = noisy(seed, h.size)
    ref = oracle(h, z, zdr, rho)
    for engine in ENGINES:
        out = melting_layer_gradient(dataset(z, zdr, rho, h=h), engine=engine)
        if ref is None:
            assert out.melting_layer_flag in (0, 2)
        else:
            assert (
                float(out.melting_layer_bottom),
                float(out.melting_layer_top),
                int(out.melting_layer_top_step),
            ) == ref


@pytest.mark.skipif(not mg.HAS_COMPILED_KERNEL, reason="no compiled kernel")
@pytest.mark.parametrize("nan", [0.0, 0.1])
def test_the_engines_agree_on_many_profiles(nan):
    h = np.cumsum(np.r_[0.0, np.random.default_rng(0).uniform(50.0, 150.0, 79)])
    arrays = [noisy(s, h.size, nan) for s in range(300)]
    ds = dataset(
        *(np.stack([a[k] for a in arrays]) for k in range(3)),
        h=h,
        dims=("time", "height"),
        time=np.arange(300),
    )
    a = melting_layer_gradient(ds, engine="numpy")
    b = melting_layer_gradient(ds, engine="compiled")
    c = melting_layer_gradient(ds, engine="compiled", n_threads=1)
    xr.testing.assert_identical(without_provenance(a), without_provenance(b))
    xr.testing.assert_identical(without_provenance(b), without_provenance(c))
    flags = np.bincount(a.melting_layer_flag.values, minlength=3)
    # all three outcomes occur in noise
    assert (flags > 0).all()
    assert (
        a.melting_layer_top_step.values[a.melting_layer_flag.values == 1] >= 3
    ).all()


def test_flags_and_heights_are_consistent():
    z, zdr, rho = (np.stack([noisy(s, 60)[k] for s in range(200)]) for k in range(3))
    h = np.arange(0.0, 6000.0, 100.0)
    out = melting_layer_gradient(dataset(z, zdr, rho, h=h, dims=("p", "height")))
    flag = out.melting_layer_flag.values
    found = flag == 1
    assert (flag == 2).any() and found.any()
    top, bottom = out.melting_layer_top.values, out.melting_layer_bottom.values
    assert (top[found] > bottom[found]).all()
    assert np.isnan(top[~found]).all() and np.isnan(bottom[~found]).all()
    assert (out.melting_layer_top_step.values[~found] == 0).all()
    assert out.melting_layer_flag.attrs["flag_meanings"] == "none detected inconsistent"
    assert out.melting_layer_top.attrs["units"] == "m"


def test_profiles_without_a_layer():
    z, zdr, rho = profile()
    flat = np.full_like(z, 20.0)
    for engine in ENGINES:
        # no gradients, so all heights coincide at the first gate and the top at the same
        # gate gives a layer that does not exist
        out = melting_layer_gradient(dataset(flat, flat, flat), engine=engine)
        assert out.melting_layer_flag in (0, 2)
        # the maximum of the profile is not positive
        out = melting_layer_gradient(dataset(z - 100.0, zdr, rho), engine=engine)
        assert out.melting_layer_flag == 0
        out = melting_layer_gradient(dataset(z, zdr - 100.0, rho), engine=engine)
        assert out.melting_layer_flag == 0
        out = melting_layer_gradient(dataset(z, zdr, rho - 100.0), engine=engine)
        assert out.melting_layer_flag == 0
        # a variable without a value
        out = melting_layer_gradient(dataset(z, zdr * np.nan, rho), engine=engine)
        assert out.melting_layer_flag == 0
        out = melting_layer_gradient(dataset(z * np.nan, zdr, rho), engine=engine)
        assert out.melting_layer_flag == 0


@pytest.mark.parametrize("engine", ENGINES)
def test_the_top_cannot_be_above_the_last_gate_and_inconsistent_layers(engine):
    h = np.arange(0.0, 6000.0, 100.0)
    # the minima coincide at the last gate and the step 5 would need a gate above it
    z, zdr, rho = noisy(0, h.size)
    assert oracle(h, z, zdr, rho) is None
    out = melting_layer_gradient(dataset(z, zdr, rho, h=h), engine=engine)
    assert out.melting_layer_flag == 0
    # the top is not above the bottom: flagged, no heights
    z, zdr, rho = noisy(12, h.size)
    assert oracle(h, z, zdr, rho) is None
    out = melting_layer_gradient(dataset(z, zdr, rho, h=h), engine=engine)
    assert out.melting_layer_flag == 2
    assert np.isnan(float(out.melting_layer_top))


@pytest.mark.parametrize("engine", ENGINES)
def test_height_range_crops_the_profile(engine):
    z, zdr, rho = profile(3800.0, 3900.0, 4150.0)
    # noise at the echo top: its gradient of Z is the highest minimum, which step 4
    # takes for the top of the layer
    z[75:] += 20.0 * np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    ds = dataset(z, zdr, rho)
    full = melting_layer_gradient(ds, engine=engine)
    cropped = melting_layer_gradient(ds, height_range=(1000.0, 6000.0), engine=engine)
    assert float(cropped.melting_layer_top) < 6000.0
    assert (
        3000.0 < float(cropped.melting_layer_bottom) < float(cropped.melting_layer_top)
    )
    assert cropped.melting_layer_flag == 1
    assert float(full.melting_layer_top) == 7500.0
    assert float(cropped.melting_layer_top) == 4300.0


@pytest.mark.parametrize("engine", ENGINES)
def test_extra_dimensions_names_and_coordinates(engine):
    z, zdr, rho = profile(3800.0, 3900.0, 4150.0)
    stack = lambda x: np.stack([x, x])  # noqa: E731
    ds = dataset(
        stack(z),
        stack(zdr),
        stack(rho),
        dims=("time", "height"),
        time=np.array(["2026-01-01T00:00", "2026-01-01T00:05"], dtype="datetime64[ns]"),
        azimuth=109.0,
    )
    ds = ds.rename({"DBZH": "reflectivity", "ZDR": "differential_reflectivity"})
    ds["RHOHV"].attrs["standard_name"] = "radar_correlation_coefficient_hv"
    ds = ds.rename({"RHOHV": "other"})
    out = melting_layer_gradient(ds, engine=engine)
    assert out.melting_layer_top.dims == ("time",)
    assert out.melting_layer_top.values.tolist() == [4300.0, 4300.0]
    assert float(out.azimuth) == 109.0
    assert "height" not in out.dims
    # explicit names
    ds2 = ds.rename({"reflectivity": "Z"})
    with pytest.raises(ValueError, match="needs a dbz profile"):
        melting_layer_gradient(ds2, engine=engine)
    again = melting_layer_gradient(ds2, dbz="Z", engine=engine)
    xr.testing.assert_identical(
        without_provenance(again).drop_attrs(), without_provenance(out).drop_attrs()
    )
    with pytest.raises(ValueError, match="'nothing' not found"):
        melting_layer_gradient(ds2, dbz="nothing", engine=engine)


def test_a_dask_profile_is_computed():
    pytest.importorskip("dask")
    z, zdr, rho = profile(3800.0, 3900.0, 4150.0)
    ds = dataset(z, zdr, rho)
    lazy = melting_layer_gradient(ds.chunk({"height": 20}))
    xr.testing.assert_identical(
        without_provenance(lazy), without_provenance(melting_layer_gradient(ds))
    )


def test_errors():
    z, zdr, rho = profile()
    ds = dataset(z, zdr, rho)
    with pytest.raises(ValueError, match="Dataset with a 'height' dimension"):
        melting_layer_gradient(ds.DBZH)
    with pytest.raises(ValueError, match="Dataset with a 'height' dimension"):
        melting_layer_gradient(ds.rename({"height": "z"}))
    with pytest.raises(ValueError, match="needs a rhohv profile"):
        melting_layer_gradient(ds.drop_vars("RHOHV"))
    with pytest.raises(ValueError, match="needs a zdr profile"):
        melting_layer_gradient(ds.drop_vars("ZDR"))
    with pytest.raises(ValueError, match="strictly ascending"):
        melting_layer_gradient(ds.isel(height=slice(None, None, -1)))
    with pytest.raises(ValueError, match="at least three heights"):
        melting_layer_gradient(ds.isel(height=slice(0, 2)))
    with pytest.raises(ValueError, match="at least three heights"):
        melting_layer_gradient(ds, height_range=(1000.0, 1100.0))
    with pytest.raises(ValueError, match="height_range"):
        melting_layer_gradient(ds, height_range=(3000.0, 2000.0))
    with pytest.raises(ValueError, match="height_range"):
        melting_layer_gradient(ds, height_range=(1000.0,))
    with pytest.raises(ValueError, match="engine must be"):
        melting_layer_gradient(ds, engine="gpu")


def test_without_the_compiled_kernel(monkeypatch):
    z, zdr, rho = profile(3800.0, 3900.0, 4150.0)
    ds = dataset(z, zdr, rho)
    monkeypatch.setattr(mg, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError, match="compiled"):
        melting_layer_gradient(ds, engine="compiled")
    out = melting_layer_gradient(ds)  # "auto" falls back to NumPy
    assert float(out.melting_layer_top) == 4300.0


@pytest.mark.skipif(not mg.HAS_COMPILED_KERNEL, reason="no compiled kernel")
def test_kernel_errors():
    kernel = mg._kernel
    h = np.arange(5.0)
    a = np.ones((2, 5))
    with pytest.raises(ValueError, match="2-D"):
        kernel.melting_layer_gradient(a[0], a, a, h)
    with pytest.raises(ValueError, match="shapes do not match"):
        kernel.melting_layer_gradient(a, a[:1], a, h)
    with pytest.raises(ValueError, match="one value per gate"):
        kernel.melting_layer_gradient(a, a, a, h[:3])
    with pytest.raises(ValueError, match="at least three"):
        kernel.melting_layer_gradient(a[:, :2], a[:, :2], a[:, :2], h[:2])
    with pytest.raises(ValueError, match="ascending"):
        kernel.melting_layer_gradient(a, a, a, h[::-1].copy())
    bottom, top, step = kernel.melting_layer_gradient(a[:0], a[:0], a[:0], h)
    assert bottom.shape == top.shape == step.shape == (0,)


# --------------------------------------------------------------------------
# RHI scans
# --------------------------------------------------------------------------

RHI_ELEVATIONS = np.arange(1.0, 61.0, 0.25)
RHI_RANGE = np.arange(125.0, 30_000.0, 125.0)


def ml_rhi(time="2026-01-01T00:00", z_centre=3800.0, rho_centre=4150.0):
    """An RHI whose gates hold the profile of the melting layer at their height."""
    funcs = {
        "DBZH": lambda h: 25.0 + 10.0 * bump(z_centre, 300.0, h),
        "ZDR": lambda h: 0.4 + 1.5 * bump(3900.0, 250.0, h),
        "RHOHV": lambda h: 0.99 - 0.15 * bump(rho_centre, 250.0, h),
    }
    sweep = make_sweep(
        0.0,
        funcs,
        nray=RHI_ELEVATIONS.size,
        rng=RHI_RANGE,
        elevations=RHI_ELEVATIONS,
        time=time,
    )
    return sweep.assign_coords(azimuth=("azimuth", np.full(RHI_ELEVATIONS.size, 109.0)))


@pytest.mark.parametrize("engine", ENGINES)
def test_rhi_layer_is_the_layer_of_its_profile(engine):
    rhi = ml_rhi()
    out = melting_layer_rhi(rhi, engine=engine)
    profile_ = rhi_qvp(rhi, engine=engine)
    ref = melting_layer_gradient(profile_, engine=engine)
    xr.testing.assert_identical(without_provenance(out), without_provenance(ref))
    assert out.melting_layer_flag == 1
    # the profile is binned in 100 m, the layer of the synthetic anomalies is found
    # within a bin or two of the layer of the profile on the exact heights
    assert abs(float(out.melting_layer_bottom) - 3900.0) <= 200.0
    assert abs(float(out.melting_layer_top) - 4300.0) <= 200.0
    assert float(out.azimuth) == 109.0
    assert out.melting_layer_top.dims == ()


def test_rhi_series_volume_options_and_accessors():
    first, second = ml_rhi(), ml_rhi("2026-01-01T00:05", z_centre=4000.0)
    series = melting_layer_rhi([first, second])
    assert series.melting_layer_top.dims == ("time",) and series.time.size == 2
    assert series.melting_layer_flag.values.tolist() == [1, 1]
    one = melting_layer_rhi(second)
    assert float(one.melting_layer_top) == float(series.melting_layer_top[1])
    # a volume of RHI sweeps: the sweep named is used
    root = xr.Dataset(coords={"latitude": 45.0, "longitude": 10.0, "altitude": ALT})
    drop = ["latitude", "longitude", "altitude"]
    tree = xr.DataTree.from_dict(
        {
            "/": root,
            "sweep_0": first.drop_vars(drop),
            "sweep_1": second.drop_vars(drop),
        }
    )
    xr.testing.assert_allclose(
        without_provenance(melting_layer_rhi(tree)),
        without_provenance(melting_layer_rhi(first)),
    )
    c = melting_layer_rhi(tree, sweep=1)
    assert float(c.melting_layer_top) == float(one.melting_layer_top)
    # a bin of 200 m
    coarse = melting_layer_rhi(first, dz=200.0)
    assert float(coarse.melting_layer_top) % 200.0 == 0.0
    # the detection can be limited in height
    limited = melting_layer_rhi(first, height_range=(2000.0, 6000.0))
    assert limited.melting_layer_flag == 1
    # accessors
    xr.testing.assert_identical(
        first.radarx.melting_layer_rhi(), melting_layer_rhi(first)
    )
    xr.testing.assert_identical(
        tree.radarx.melting_layer_rhi(), melting_layer_rhi(tree)
    )
    profiles = rhi_qvp(first)
    xr.testing.assert_identical(
        profiles.radarx.melting_layer_gradient(), melting_layer_gradient(profiles)
    )


def test_the_bottom_caps_rain_products():
    z, zdr, rho = profile(3800.0, 3900.0, 4150.0)
    layer = melting_layer_gradient(dataset(z, zdr, rho))
    rain = xr.DataArray(
        np.ones((2, 4)),
        dims=("azimuth", "range"),
        coords={
            "elevation": ("azimuth", [0.5, 6.0]),
            "range": ("range", [20e3, 40e3, 60e3, 80e3]),
            "altitude": ALT,
        },
    )
    capped = mask_above_melting_layer(rain, layer)
    assert capped.isnull().sum() > 0 and capped.notnull().sum() > 0


def test_provenance_and_references():
    from radarx._provenance import load_registry

    z, zdr, rho = profile(3800.0, 3900.0, 4150.0)
    registry = load_registry()
    for out in (
        melting_layer_gradient(dataset(z, zdr, rho)),
        melting_layer_rhi(ml_rhi()),
    ):
        refs = out.attrs["radarx_references"].split()
        assert "10.3390/rs11232848" in refs and all(r in registry for r in refs)
        assert "Allabakash" in out.attrs["radarx_method"] and out.attrs["history"]
    assert registry["10.1002/qj.2672"]["doi"] == "10.1002/qj.2672"
    assert (
        "10.1002/qj.2672"
        in melting_layer_gradient(dataset(z, zdr, rho))
        .attrs["radarx_references"]
        .split()
    )


# --------------------------------------------------------------------------
# the search window of step 4
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_top_window_limits_the_search_of_step_4(engine):
    z, zdr, rho = profile(3800.0, 3900.0, 4150.0)
    z[75:] += 20.0 * np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    ds = dataset(z, zdr, rho)
    whole = melting_layer_gradient(ds, engine=engine)
    assert float(whole.melting_layer_top) == 7500.0
    for window in (500.0, 1000.0, 3000.0):
        out = melting_layer_gradient(ds, top_window=window, engine=engine)
        assert float(out.melting_layer_top) == 4300.0
        assert float(out.melting_layer_bottom) == 3900.0
    # a window as wide as the profile is the paper's algorithm
    wide = melting_layer_gradient(ds, top_window=1e6, engine=engine)
    xr.testing.assert_identical(without_provenance(wide), without_provenance(whole))
    # no gate but the one of step 4 within the window: step 3 or 5
    narrow = melting_layer_gradient(ds, top_window=0.0, engine=engine)
    assert narrow.melting_layer_top_step in (3, 5)


@pytest.mark.skipif(not mg.HAS_COMPILED_KERNEL, reason="no compiled kernel")
@pytest.mark.parametrize("window", [0.0, 150.0, 500.0, 2000.0])
def test_top_window_engines_agree(window):
    h = np.arange(0.0, 6000.0, 100.0)
    arrays = [noisy(s, h.size, 0.05) for s in range(200)]
    ds = dataset(
        *(np.stack([a[k] for a in arrays]) for k in range(3)),
        h=h,
        dims=("time", "height"),
        time=np.arange(200),
    )
    a = melting_layer_gradient(ds, top_window=window, engine="numpy")
    b = melting_layer_gradient(ds, top_window=window, engine="compiled")
    xr.testing.assert_identical(without_provenance(a), without_provenance(b))


def test_top_window_errors_and_rhi():
    z, zdr, rho = profile()
    with pytest.raises(ValueError, match="top_window"):
        melting_layer_gradient(dataset(z, zdr, rho), top_window=-1.0)
    with pytest.raises(ValueError, match="top_window"):
        melting_layer_gradient(dataset(z, zdr, rho), top_window=np.nan)
    if mg.HAS_COMPILED_KERNEL:
        a = np.ones((1, 5))
        with pytest.raises(ValueError, match="window"):
            mg._kernel.melting_layer_gradient(a, a, a, np.arange(5.0), -1.0)
    rhi = ml_rhi()
    out = melting_layer_rhi(rhi, top_window=500.0)
    ref = melting_layer_gradient(rhi_qvp(rhi), top_window=500.0)
    xr.testing.assert_identical(without_provenance(out), without_provenance(ref))


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------


def test_real_qvps_of_the_armor_radar():
    """C band, convective rain, 11 April 2008: the layer is near the 0 deg C level."""
    xd = pytest.importorskip("xradar")
    from open_radar_data import DATASETS

    try:
        names = sorted(n for n in DATASETS.registry if n.startswith("RAW_NA_000_125_"))
        volumes = [xd.io.open_iris_datatree(DATASETS.fetch(n)) for n in names[:8]]
    except Exception as err:  # pragma: no cover - offline
        pytest.skip(f"no ARMOR data: {err}")
    tqvp = radarx.retrieve.qvp_timeseries(
        volumes, ["DBZH", "ZDR", "RHOHV"], elevation=12.0
    ).compute()
    # the gradients of the native 26 m profile are noise; average it in 100 m bins
    edges = np.arange(0.0, 16000.0, 100.0)
    binned = (
        tqvp[["DBZH", "ZDR", "RHOHV"]]
        .groupby_bins("height", edges)
        .mean()
        .rename({"height_bins": "height"})
        .transpose("time", "height")
    )
    binned["height"] = [interval.mid for interval in binned.height.values]
    a = melting_layer_gradient(
        binned, height_range=(500.0, 8000.0), top_window=500.0, engine="numpy"
    )
    if mg.HAS_COMPILED_KERNEL:
        b = melting_layer_gradient(
            binned, height_range=(500.0, 8000.0), top_window=500.0, engine="compiled"
        )
        xr.testing.assert_identical(without_provenance(a), without_provenance(b))
    found = a.melting_layer_flag.values == 1
    assert found.sum() >= 3
    # the 0 deg C level of the BMX sounding of 00 UTC 12 April 2008 is at 3.86 km
    top = a.melting_layer_top.values[found]
    bottom = a.melting_layer_bottom.values[found]
    assert ((top > 3000.0) & (top < 4500.0)).all()
    assert ((bottom > 2300.0) & (bottom < top)).all()
