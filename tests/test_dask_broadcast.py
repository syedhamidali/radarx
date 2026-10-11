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

import pytest

import radarx  # noqa: F401
from radarx.retrieve import echo_mask, estimate_kdp, llsd
from radarx.retrieve import kdp as kdp_mod
from radarx.retrieve import qc as qc_mod
from radarx.retrieve import shear as shear_mod

from . import test_kdp, test_qc, test_shear
from .sweepwise_helpers import check_stack, check_volume

dask = pytest.importorskip("dask")


def qc_sweep(k):
    return test_qc._sweep(seed=k, nray=24, ng=400)[0]


def kdp_sweep(k):
    return test_kdp._sweep(nray=16, ng=300, seed=k)[0]


def shear_sweep(k):
    az = test_shear.jittered_azimuths(180)
    rng = test_shear.RANGE[:160]
    velocity, *_ = test_shear.rankine(az, rng)
    return test_shear.sweep(velocity * (1 + 0.3 * k), az, rng)


# (id, function, module with the kernel, make(k), keyword arguments, attributes
# that only a sweep in memory has)
CASES = [
    ("echo_mask", echo_mask, qc_mod, qc_sweep, {}, ()),
    (
        "echo_mask-krause",
        echo_mask,
        qc_mod,
        qc_sweep,
        {"method": "krause2016"},
        (),
    ),
    (
        "kdp-hubbert",
        estimate_kdp,
        kdp_mod,
        kdp_sweep,
        {},
        ("phase_noise", "filter_window", "kdp_reference", "phidp_sign", "comment"),
    ),
    (
        "kdp-vulpiani",
        estimate_kdp,
        kdp_mod,
        kdp_sweep,
        {"method": "vulpiani", "phidp_sign": 1},
        ("phase_noise", "filter_window", "kdp_reference"),
    ),
    ("llsd", llsd, shear_mod, shear_sweep, {"field": "VRADH"}, ()),
]


def case_params():
    for name, func, module, make, kwargs, ignore in CASES:
        for engine in ("numpy", "compiled"):
            skip = pytest.mark.skipif(
                engine == "compiled" and not module.HAS_COMPILED_KERNEL,
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
