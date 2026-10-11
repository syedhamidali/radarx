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

import importlib

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.retrieve import (
    correct_dual_prf,
    dealias_confidence,
    dealias_velocity,
    dsd,
    echo_mask,
    estimate_kdp,
    hid,
    llsd,
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
    test_shear,
)
from .sweepwise_helpers import check_stack, check_volume

dask = pytest.importorskip("dask")
dsd_mod = importlib.import_module("radarx.retrieve.dsd")
hid_mod = importlib.import_module("radarx.retrieve.hid")


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


def hid_sweep(k):
    return test_hid._random_volume(seed=k, n=1, nray=24, ng=60)[0]


def dual_prf_sweep(k):
    return test_dealias_extras.dual_prf_sweep(seed=k, nray=90, ngate=60)[0]


def case(name, func, kernel, make, ignore=(), **kwargs):
    return name, func, kernel, make, kwargs, ignore


# (id, function, whether the compiled kernel is built, make(k), keyword
# arguments, attributes that only a sweep in memory has)
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


def case_params():
    for name, func, kernel, make, kwargs, ignore in CASES:
        for engine in ("numpy", "compiled"):
            skip = pytest.mark.skipif(
                engine == "compiled" and not kernel,
                reason="compiled kernel not built",
            )
            yield pytest.param(
                func,
                make,
                {**kwargs, "engine": engine},
                ignore,
                id=f"{name}-{engine}",
                marks=skip,
            )


@pytest.mark.parametrize("func, make, kwargs, ignore", case_params())
def test_stack_and_dask(func, make, kwargs, ignore):
    check_stack(func, make, ignore_attrs=ignore, **kwargs)


@pytest.mark.parametrize("func, make, kwargs, ignore", case_params())
def test_volume_with_dask_sweeps(func, make, kwargs, ignore):
    check_volume(func, make, **kwargs)
