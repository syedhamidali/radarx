#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Functions that compute dask arrays
==================================

Functions that need a whole volume or all times take dask input, compute it
and give the result of the same call on arrays in memory. Each case builds
its call from the fixtures of the tests of the function.
"""

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx import grid, retrieve
from radarx._sweepwise import DASK_SUPPORT

from . import (
    test_advection,
    test_cappi,
    test_dealias,
    test_hail,
    test_hid,
    test_qc,
    test_qvp,
    test_shear_layer,
    test_wind_profile,
)

pytest.importorskip("dask")


def chunked(obj):
    """The same object with every variable in two chunks along each dimension."""
    if isinstance(obj, xr.DataTree):
        return obj.map_over_datasets(chunked)
    if isinstance(obj, (xr.Dataset, xr.DataArray)):
        return obj.chunk({d: max(1, n // 2) for d, n in obj.sizes.items()})
    if isinstance(obj, (list, tuple)):
        return type(obj)(chunked(o) for o in obj)
    return obj


def has_dask(obj):
    if isinstance(obj, (list, tuple)):
        return any(has_dask(o) for o in obj)
    if isinstance(obj, xr.DataTree):
        return any(has_dask(n.to_dataset(inherit=False)) for n in obj.subtree)
    if isinstance(obj, xr.DataArray):
        return obj.chunks is not None
    if isinstance(obj, xr.Dataset):
        return any(v.chunks is not None for v in obj.variables.values())
    return False


def vad():
    return retrieve.vad_profile, (test_wind_profile.vad_sweep(),), {}


def cappi():
    volume = xr.DataTree.from_dict(
        {
            f"sweep_{k}": test_cappi._make_ppi_sweep(el, "2024-01-01T00:00:00")
            for k, el in enumerate((1.0, 3.0, 5.0))
        }
    )
    kwargs = {"height": 150.0, "data_vars": ["DBZH"], "x_res": 1000.0, "y_res": 1000.0}
    return retrieve.create_cappi, (volume,), kwargs


def motion():
    t0, t1 = test_advection._blobs(0.0, 10.0, 5.0), test_advection._blobs(60, 10, 5)
    return retrieve.estimate_motion, (t0, t1), {}


def advect():
    t0 = test_advection._blobs(0.0, 10.0, 5.0)
    return retrieve.advect, (t0, 10.0, 5.0), {"dt": 30.0}


def layer_max():
    kwargs = {"field": "VRADH", "layer": (0.0, 6000.0), "nodata": None}
    return (
        retrieve.azimuthal_shear_layer_max,
        (test_shear_layer.column_volume(),),
        kwargs,
    )


def vil_volume():
    return retrieve.vil, (test_shear_layer.column_volume(),), {"dbz": "DBZH"}


def echo_top_volume():
    return retrieve.echo_top, (test_shear_layer.column_volume(),), {"dbz": "DBZH"}


def sheared_sweep():
    sweep = test_shear_layer.column_sweep(0.5)
    return retrieve.rotation_couplets, (sweep,), {}


def qvp_series():
    volumes = [test_qvp._volume(seed=k, nray=60) for k in range(2)]
    return retrieve.qvp_timeseries, (volumes,), {}


def rd_qvp():
    return retrieve.rd_qvp, (test_qvp._volume(nray=60),), {}


def hail_grid():
    dbz = test_hail.hail_storm_dbz(
        *np.meshgrid(
            np.arange(0, 80e3, 8e3),
            np.arange(0, 50e3, 10e3),
            np.arange(500, 14e3, 1000.0),
            indexing="ij",
        )
    )
    ds = xr.Dataset(
        {"DBZH": (("x", "y", "z"), dbz, {"units": "dBZ"})},
        coords={
            "x": np.arange(0, 80e3, 8e3),
            "y": np.arange(0, 50e3, 10e3),
            "z": np.arange(500, 14e3, 1000.0),
        },
    )
    return retrieve.shi, (ds,), {"freezing_level": 4200.0, "minus20_level": 7500.0}


def dealias_volume():
    volume, _ = test_dealias.volume(nray=60, ngate=120)
    return retrieve.dealias_velocity, (volume,), {"field": "VRADH"}


def winter_hid():
    grid = test_hid._winter_grid(300)
    kwargs = {"band": "C", "method": "thompson", "ml_gates": (50, 200)}
    return retrieve.hid, (grid, "T"), kwargs


def split_cuts():
    return retrieve.echo_mask, (test_qc._volume(),), {"split_cuts": True}


CASES = {
    "dealias-continuity": dealias_volume,
    "hid-thompson": winter_hid,
    "echo_mask-split-cuts": split_cuts,
    "vad_profile": vad,
    "create_cappi": cappi,
    "estimate_motion": motion,
    "advect": advect,
    "azimuthal_shear_layer_max": layer_max,
    "vil-volume": vil_volume,
    "echo_top-volume": echo_top_volume,
    "rotation_couplets": sheared_sweep,
    "qvp_timeseries": qvp_series,
    "rd_qvp": rd_qvp,
    "shi": hail_grid,
}


@pytest.mark.parametrize("name", CASES)
def test_dask_input_is_computed(name):
    func, args, kwargs = CASES[name]()
    expected = func(*args, **kwargs)
    dask_args = chunked(args)
    assert any(has_dask(a) for a in dask_args)
    result = func(*dask_args, **kwargs)
    if hasattr(result, "compute"):
        result = result.compute()
    xr.testing.assert_allclose(result, expected)


def test_every_declared_function_exists():
    for key in DASK_SUPPORT:
        module, name = key.rsplit(".", 1)
        assert module.split(".")[0] in ("retrieve", "grid"), key
        package = retrieve if module.startswith("retrieve") else grid
        assert callable(getattr(package, name)), key
