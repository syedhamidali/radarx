#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Sweep-level functions on stacks of sweeps and on dask arrays
============================================================

One table of functions: each must give the same result for a sweep, for a
``(time, azimuth, range)`` stack and for the chunked stack, return dask arrays
for dask input, keep its input, attributes and provenance, and process the
sweeps of a volume with dask arrays.
"""

import functools
import importlib
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx._sweepwise import DASK_SUPPORT
from radarx.retrieve import (
    apply_mask,
    azimuthal_shear,
    correct_attenuation,
    correct_dual_prf,
    correct_rhohv_noise,
    correct_zdr_noise,
    dealias_ambiguity,
    dealias_confidence,
    dealias_velocity,
    dsd,
    echo_mask,
    echo_top,
    estimate_kdp,
    hid,
    identify_cells,
    kdp_column,
    kdp_column_metrics,
    liquid_water_content,
    llsd,
    noise_level,
    phidp_offset,
    qvp,
    radial_divergence,
    rain_rate,
    rain_rate_a,
    rain_rate_kdp,
    rain_rate_z,
    rain_rate_z_zdr,
    snr,
    vil,
    vil_density,
    z_offset,
    zdr_column,
    zdr_column_metrics,
    zdr_offset,
)
from radarx.retrieve import dealias as dealias_mod
from radarx.retrieve import kdp as kdp_mod
from radarx.retrieve import qc as qc_mod
from radarx.retrieve import shear as shear_mod

from . import (
    test_dealias,
    test_dealias_extras,
    test_hid,
    test_kdp,
    test_qc,
    test_qvp,
    test_shear,
)
from .sweepwise_helpers import CHUNKS, check_stack, check_volume, stacked, volume_of

dealias_methods_mod = importlib.import_module("radarx.retrieve.dealias_methods")
dsd_mod = importlib.import_module("radarx.retrieve.dsd")
hid_mod = importlib.import_module("radarx.retrieve.hid")
vil_mod = importlib.import_module("radarx.retrieve.vil")
columns_mod = importlib.import_module("radarx.retrieve.columns")
tracking_cells = importlib.import_module("radarx.retrieve._cells")
noise_mod = importlib.import_module("radarx.retrieve.noise")
attenuation_mod = importlib.import_module("radarx.retrieve.attenuation")
calibration_mod = importlib.import_module("radarx.retrieve.calibration")
qpe_mod = importlib.import_module("radarx.retrieve.qpe")
qvp_mod = importlib.import_module("radarx.retrieve.vertical_profiles")


def qc_sweep(k):
    return test_qc._sweep(seed=k, nray=24, ng=400)[0]


def kdp_sweep(k):
    return test_kdp._sweep(nray=16, ng=300, seed=k)[0]


def shear_sweep(k):
    az = test_shear.jittered_azimuths(180)
    rng = test_shear.RANGE[:160]
    velocity, *_ = test_shear.rankine(az, rng)
    return test_shear.sweep(velocity * (1 + 0.3 * k), az, rng)


def alias_sweep(k):
    return test_dealias.synthetic_sweep(seed=k, nray=120, ngate=200)[0]


def small_alias_sweep(k):
    return test_dealias.synthetic_sweep(seed=k, nray=36, ngate=40)[0]


def polar_sweep(k):
    """A sweep with the fields of the polarimetric products."""
    ds = test_qc._sweep(seed=k, nray=24, ng=400)[0]
    rnd = np.random.default_rng(k)
    dims = ds.DBZH.dims
    ds["KDP"] = (dims, rnd.uniform(0.0, 2.0, ds.DBZH.shape), {"units": "degree km-1"})
    ds["AH"] = (dims, rnd.uniform(0.0, 0.05, ds.DBZH.shape), {"units": "dB km-1"})
    return ds


def rain_sweep(k):
    rnd = np.random.default_rng(k)
    shape = (16, 50)
    dims = ("azimuth", "range")
    return xr.Dataset(
        {
            "DBZH": (dims, rnd.uniform(15, 50, shape), {"units": "dBZ"}),
            "ZDR": (dims, rnd.uniform(0.2, 3.0, shape), {"units": "dB"}),
            "KDP": (
                dims,
                np.where(rnd.random(shape) < 0.1, np.nan, rnd.uniform(0, 2, shape)),
            ),
            "RAIN": (dims, rnd.random(shape) > 0.2),
        },
        coords={
            "azimuth": np.arange(16) * 22.5,
            "range": 1000.0 + 250.0 * np.arange(50),
        },
        attrs={"frequency": 3.0e9},
    )


GRID_CHUNKS = {"time": 1, "z": 5, "y": 2}


def qvp_sweep(k):
    return test_qvp._sweep(nray=60, seed=k, time=f"2026-01-01T0{k}:00")[0]


def column_grid(k):
    rnd = np.random.default_rng(k)
    shape = (12, 5, 4)
    dbz = np.where(rnd.random(shape) < 0.1, np.nan, rnd.uniform(-10, 60, shape))
    return xr.Dataset(
        {"DBZH": (("z", "y", "x"), dbz, {"units": "dBZ"})},
        coords={
            "z": 500.0 + 1000.0 * np.arange(12),
            "y": np.arange(5) * 1000.0,
            "x": np.arange(4) * 1000.0,
            "altitude": ((), 100.0),
        },
    )


@functools.wraps(zdr_column_metrics)
def metrics_without_engine(obj, engine=None, **kwargs):
    """The harness passes an engine; the metrics are plain xarray reductions."""
    return zdr_column_metrics(obj, **kwargs)


def zdr_grid(k):
    rnd = np.random.default_rng(k)
    shape = (12, 5, 4)
    zdr = np.where(rnd.random(shape) < 0.05, np.nan, rnd.uniform(0.0, 2.5, shape))
    return xr.Dataset(
        {
            "ZDR": (("z", "y", "x"), zdr),
            "RHOHV": (("z", "y", "x"), rnd.uniform(0.7, 1.0, shape)),
        },
        coords={
            "z": 500.0 + 1000.0 * np.arange(12),
            "y": np.arange(5) * 1000.0,
            "x": np.arange(4) * 1000.0,
        },
    )


def column_depth_map(k):
    rnd = np.random.default_rng(k)
    depth = np.where(rnd.random((5, 4)) < 0.1, np.nan, rnd.uniform(0.0, 4000.0, (5, 4)))
    return xr.Dataset(
        {"ZDR_COLUMN_DEPTH": (("y", "x"), depth)},
        coords={"y": np.arange(5) * 1000.0, "x": np.arange(4) * 1000.0},
    )


def kdp_grid(k):
    rnd = np.random.default_rng(k)
    shape = (12, 5, 4)
    kdp = np.where(rnd.random(shape) < 0.05, np.nan, rnd.uniform(0.0, 3.0, shape))
    return xr.Dataset(
        {
            "KDP": (("z", "y", "x"), kdp),
            "RHOHV": (("z", "y", "x"), rnd.uniform(0.7, 1.0, shape)),
        },
        coords={
            "z": 500.0 + 1000.0 * np.arange(12),
            "y": np.arange(5) * 1000.0,
            "x": np.arange(4) * 1000.0,
        },
    )


def kdp_column_map(k):
    return kdp_column(kdp_grid(k), freezing_level=4000.0, smooth=None, min_depth=100.0)


@functools.wraps(kdp_column_metrics)
def kdp_metrics_without_engine(obj, engine=None, **kwargs):
    """The harness passes an engine; the metrics are plain xarray reductions."""
    return kdp_column_metrics(obj, **kwargs)


def reflectivity_map(k):
    rnd = np.random.default_rng(k)
    field = np.where(
        rnd.random((20, 24)) < 0.1, np.nan, rnd.uniform(0.0, 50.0, (20, 24))
    )
    return xr.Dataset(
        {"DBZH": (("y", "x"), field, {"units": "dBZ"})},
        coords={"y": np.arange(20) * 1000.0, "x": np.arange(24) * 1000.0},
    )


def hid_sweep(k):
    return test_hid._random_volume(seed=k, n=1, nray=24, ng=60)[0]


def dual_prf_sweep(k):
    return test_dealias_extras.dual_prf_sweep(seed=k, nray=90, ngate=60)[0]


class Case(NamedTuple):
    name: str
    func: Callable
    kernel: bool  # whether the compiled kernel is built
    make: Callable  # make(k) is the sweep Dataset number k
    kwargs: dict
    ignore: tuple  # attributes that only a sweep in memory has
    chunks: dict  # chunks of the dask stack
    dim: str  # name of the dimension of the stack
    volume: bool  # whether a volume of sweeps is a valid input


def case(
    name,
    func,
    kernel,
    make,
    ignore=(),
    chunks=None,
    volume=True,
    dim="time",
    **kwargs,
):
    return Case(name, func, kernel, make, kwargs, ignore, chunks or CHUNKS, dim, volume)


CASES = [
    case("echo_mask", echo_mask, qc_mod.HAS_COMPILED_KERNEL, qc_sweep),
    case(
        "echo_mask-krause",
        echo_mask,
        qc_mod.HAS_COMPILED_KERNEL,
        qc_sweep,
        method="krause2016",
    ),
    case(
        "kdp-hubbert",
        estimate_kdp,
        kdp_mod.HAS_COMPILED_KERNEL,
        kdp_sweep,
        ("phase_noise", "filter_window", "kdp_reference", "phidp_sign", "comment"),
    ),
    case(
        "kdp-vulpiani",
        estimate_kdp,
        kdp_mod.HAS_COMPILED_KERNEL,
        kdp_sweep,
        ("phase_noise", "filter_window", "kdp_reference"),
        method="vulpiani",
        phidp_sign=1,
    ),
    case("apply_mask", apply_mask, qc_mod.HAS_COMPILED_KERNEL, qc_sweep),
    case(
        "azimuthal_shear",
        azimuthal_shear,
        shear_mod.HAS_COMPILED_KERNEL,
        shear_sweep,
        field="VRADH",
    ),
    case(
        "radial_divergence",
        radial_divergence,
        shear_mod.HAS_COMPILED_KERNEL,
        shear_sweep,
        field="VRADH",
    ),
    case("liquid_water_content", liquid_water_content, False, polar_sweep),
    case(
        "snr",
        snr,
        noise_mod.HAS_COMPILED_KERNEL,
        polar_sweep,
        ("noise_level",),
    ),
    case("noise_level", noise_level, noise_mod.HAS_COMPILED_KERNEL, polar_sweep),
    case(
        "correct_rhohv_noise",
        correct_rhohv_noise,
        noise_mod.HAS_COMPILED_KERNEL,
        polar_sweep,
        ("noise_level",),
    ),
    case(
        "correct_zdr_noise",
        correct_zdr_noise,
        noise_mod.HAS_COMPILED_KERNEL,
        polar_sweep,
        ("noise_level",),
    ),
    case(
        "correct_attenuation",
        correct_attenuation,
        attenuation_mod.HAS_COMPILED_KERNEL,
        polar_sweep,
    ),
    case(
        "zdr_offset",
        zdr_offset,
        calibration_mod.HAS_COMPILED_KERNEL,
        polar_sweep,
        volume=False,
    ),
    case(
        "z_offset",
        z_offset,
        calibration_mod.HAS_COMPILED_KERNEL,
        polar_sweep,
        volume=False,
    ),
    case(
        "phidp_offset",
        phidp_offset,
        calibration_mod.HAS_COMPILED_KERNEL,
        polar_sweep,
        volume=False,
    ),
    case("rain_rate_z", rain_rate_z, qpe_mod.HAS_COMPILED_KERNEL, polar_sweep),
    case("rain_rate_kdp", rain_rate_kdp, qpe_mod.HAS_COMPILED_KERNEL, polar_sweep),
    case("rain_rate_z_zdr", rain_rate_z_zdr, qpe_mod.HAS_COMPILED_KERNEL, polar_sweep),
    case("rain_rate_a", rain_rate_a, qpe_mod.HAS_COMPILED_KERNEL, polar_sweep),
    case("rain_rate", rain_rate, qpe_mod.HAS_COMPILED_KERNEL, polar_sweep),
    case("llsd", llsd, shear_mod.HAS_COMPILED_KERNEL, shear_sweep, field="VRADH"),
    case(
        "dealias",
        dealias_velocity,
        dealias_mod.HAS_COMPILED_KERNEL,
        alias_sweep,
        ("dealias_anchor",),
        field="VRADH",
        sweep_continuity=False,
    ),
    case(
        "dealias-confidence",
        dealias_velocity,
        dealias_mod.HAS_COMPILED_KERNEL,
        alias_sweep,
        ("dealias_anchor",),
        field="VRADH",
        confidence=True,
        sweep_continuity=False,
    ),
    case(
        "dealias-unravel",
        dealias_velocity,
        dealias_mod.HAS_UNRAVEL_KERNEL,
        small_alias_sweep,
        ("dealias_anchor",),
        field="VRADH",
        method="unravel",
        sweep_continuity=False,
    ),
    case(
        "dealias_confidence",
        dealias_confidence,
        dealias_mod.HAS_COMPILED_KERNEL,
        alias_sweep,
        field="VRADH",
    ),
    case(
        "dealias_ambiguity",
        dealias_ambiguity,
        dealias_methods_mod.HAS_COMPILED_KERNEL,
        alias_sweep,
        field="VRADH",
    ),
    case(
        "vil",
        vil,
        vil_mod.HAS_COMPILED_KERNEL,
        column_grid,
        chunks=GRID_CHUNKS,
        volume=False,
        missing="clear",
    ),
    case(
        "vil-melting",
        vil,
        vil_mod.HAS_COMPILED_KERNEL,
        column_grid,
        chunks=GRID_CHUNKS,
        volume=False,
        melting=4000.0,
        fill_below=True,
    ),
    case(
        "zdr_column",
        zdr_column,
        columns_mod.HAS_COMPILED_KERNEL,
        zdr_grid,
        chunks=GRID_CHUNKS,
        volume=False,
        rhohv="RHOHV",
        freezing_level=4000.0,
    ),
    case(
        "zdr_column_metrics",
        metrics_without_engine,
        True,
        column_depth_map,
        chunks={"time": 1, "y": 2},
        volume=False,
        min_depth=1000.0,
    ),
    case(
        "kdp_column",
        kdp_column,
        columns_mod.HAS_COMPILED_KERNEL,
        kdp_grid,
        chunks=GRID_CHUNKS,
        volume=False,
        rhohv="RHOHV",
        freezing_level=4000.0,
        smooth=None,
        min_depth=100.0,
    ),
    case(
        "kdp_column_metrics",
        kdp_metrics_without_engine,
        True,
        kdp_column_map,
        chunks={"time": 1, "y": 2},
        volume=False,
    ),
    case(
        "identify_cells",
        identify_cells,
        tracking_cells.HAS_COMPILED_KERNEL,
        reflectivity_map,
        chunks={"time": 1, "y": 10},
        volume=False,
        thresholds=(20.0, 35.0),
        min_area=1.0,
    ),
    case(
        "echo_top",
        echo_top,
        vil_mod.HAS_COMPILED_KERNEL,
        column_grid,
        chunks=GRID_CHUNKS,
        volume=False,
    ),
    case(
        "vil_density",
        vil_density,
        vil_mod.HAS_COMPILED_KERNEL,
        column_grid,
        chunks=GRID_CHUNKS,
        volume=False,
    ),
    case(
        "qvp",
        qvp,
        qvp_mod.HAS_COMPILED_KERNEL,
        qvp_sweep,
        chunks={"volume": 1, "azimuth": 17},
        volume=False,
        dim="volume",
        counts=True,
    ),
    case(
        "qvp-phase",
        qvp,
        qvp_mod.HAS_COMPILED_KERNEL,
        qvp_sweep,
        chunks={"volume": 1, "azimuth": 17},
        volume=False,
        dim="volume",
        data_vars=["DBZH", "ZDR", "PHIDP"],
        elevation_correction=True,
    ),
    case("dsd", dsd, dsd_mod.HAS_COMPILED_KERNEL, rain_sweep, band="S"),
    case(
        "dsd-kdp-mask",
        dsd,
        dsd_mod.HAS_COMPILED_KERNEL,
        rain_sweep,
        method="normalized",
        band="S",
        kdp="KDP",
        mask="RAIN",
    ),
    case(
        "hid",
        hid,
        hid_mod.HAS_COMPILED_KERNEL,
        hid_sweep,
        band="C",
        method="dolan",
        temperature="TEMP",
        mask="MET",
        blockage="PBB",
        scores=True,
    ),
    case(
        "dual_prf",
        correct_dual_prf,
        dealias_mod.HAS_COMPILED_KERNEL,
        dual_prf_sweep,
        ("outliers_corrected",),
        nyquist_high=test_dealias_extras.VH,
        nyquist_low=test_dealias_extras.VL,
    ),
]


def case_params(volume=False):
    for c in CASES:
        if volume and not c.volume:
            continue
        for engine in ("numpy", "compiled"):
            skip = pytest.mark.skipif(
                engine == "compiled" and not c.kernel,
                reason="compiled kernel not built",
            )
            yield pytest.param(
                c, {**c.kwargs, "engine": engine}, id=f"{c.name}-{engine}", marks=skip
            )


@pytest.mark.parametrize("c, kwargs", case_params())
def test_stack(c, kwargs):
    check_stack(c.func, c.make, ignore_attrs=c.ignore, dim=c.dim, **kwargs)


@pytest.mark.parametrize("c, kwargs", case_params())
def test_dask_stack(c, kwargs):
    pytest.importorskip("dask")
    check_stack(
        c.func,
        c.make,
        chunks=c.chunks,
        ignore_attrs=c.ignore,
        dim=c.dim,
        dask=True,
        **kwargs,
    )


@pytest.mark.parametrize("c, kwargs", case_params(volume=True))
def test_volume_with_dask_sweeps(c, kwargs):
    pytest.importorskip("dask")
    check_volume(c.func, c.make, **kwargs)


def test_every_lazy_function_has_a_case():
    declared = {k for k, (kind, _) in DASK_SUPPORT.items() if kind == "lazy"}
    tabled = {
        f"{c.func.__module__.replace('radarx.', '')}.{c.func.__name__}" for c in CASES
    }
    assert declared <= tabled, sorted(declared - tabled)


@pytest.mark.parametrize("chunked", [False, True])
def test_dealias_options_on_a_stack(chunked):
    wind = test_dealias.wind_profile()
    members = [alias_sweep(k) for k in range(2)]
    ref = xr.DataArray(np.full(members[0].VRADH.shape, 3.0), dims=("azimuth", "range"))
    options = {"field": "VRADH", "wind_profile": wind, "reference": ref}
    expected = stacked([dealias_velocity(m, **options) for m in members], "volume")
    stack = stacked(members, "volume")
    if chunked:
        pytest.importorskip("dask")
        stack = stack.chunk({"volume": 1, "azimuth": 30})
    out = dealias_velocity(stack, **options)
    assert (out.chunks is not None) is chunked
    xr.testing.assert_allclose(out.compute().transpose(*expected.dims), expected)
    different = stacked(members, "volume").assign_coords(
        elevation=(("volume", "azimuth"), np.full((2, 120), 0.5))
    )
    with pytest.raises(ValueError, match="same elevation"):
        dealias_velocity(different, field="VRADH", wind_profile=wind)


def test_dealias_continuity_needs_sweeps_without_extra_dimensions():
    stack = stacked([alias_sweep(0), alias_sweep(1)], "volume")
    stack_high = stack.assign_coords(elevation=stack.elevation + 1.0)
    tree = volume_of([stack, stack_high])
    with pytest.raises(ValueError, match="sweep_continuity=False"):
        dealias_velocity(tree, field="VRADH")
    out = dealias_velocity(tree, field="VRADH", sweep_continuity=False)
    assert out["sweep_1"]["VRADH_dealiased"].dims[0] == "volume"
