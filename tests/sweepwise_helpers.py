#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Checks shared by the tests of sweep-level functions on stacks and dask arrays.

A function that takes a sweep must give the same result for the sweep, for a
``(time, azimuth, range)`` stack of sweeps and for the stack as dask arrays,
leave its input as it is, return dask arrays for dask input, and keep the
attributes and the provenance.
"""

import numpy as np
import xarray as xr

from radarx._provenance import read_provenance

CHUNKS = {"time": 1, "azimuth": 7}


def stacked(members, dim="time"):
    """Concatenate Datasets or DataArrays along a new leading dimension."""
    return xr.concat(members, dim).assign_coords({dim: np.arange(len(members))})


def data_arrays(obj):
    """The DataArrays of a DataArray or of the data variables of a Dataset."""
    return [obj] if isinstance(obj, xr.DataArray) else list(obj.data_vars.values())


def assert_lazy(obj):
    """Every data variable of ``obj`` is a dask array."""
    for da in data_arrays(obj):
        assert da.chunks is not None, f"{da.name} is not a dask array"


def assert_same_attrs(result, reference, ignore=()):
    """Attributes of every variable, except the keys in ``ignore``."""
    if isinstance(result, xr.DataArray):
        result, reference = result.to_dataset(), reference.to_dataset()
    assert set(result.data_vars) == set(reference.data_vars)
    for name, da in result.data_vars.items():
        keep = {k: v for k, v in da.attrs.items() if k not in ignore}
        want = {k: v for k, v in reference[name].attrs.items() if k not in ignore}
        assert keep.keys() == want.keys(), name
        for key in keep:
            assert np.array_equal(keep[key], want[key]), (name, key)


def check_stack(func, make, n=3, chunks=None, ignore_attrs=(), **kwargs):
    """
    The result for a stack and for dask arrays equals that of the sweeps.

    Parameters
    ----------
    func : callable
        ``func(sweep, **kwargs)`` returning a DataArray or Dataset.
    make : callable
        ``make(k)`` gives the sweep Dataset number ``k``; the sweeps differ.
    n : int
        Number of sweeps in the stack.
    chunks : dict, optional
        Chunks of the dask stack. Default: one sweep per chunk and the
        azimuth in pieces, so that the sweep dimensions must be rechunked.
    ignore_attrs : sequence of str
        Attributes that only the result for a sweep in memory has.
    """
    members = [make(k) for k in range(n)]
    single = [func(m, **kwargs) for m in members]
    expected = stacked(single)
    stack = stacked(members)
    before = stack.copy(deep=True)

    out = func(stack, **kwargs)
    xr.testing.assert_allclose(out, expected)
    assert_same_attrs(out, single[0], ignore_attrs)
    xr.testing.assert_identical(stack, before)

    dask_stack = stack.chunk(CHUNKS if chunks is None else chunks)
    lazy = func(dask_stack, **kwargs)
    assert_lazy(lazy)
    assert_same_attrs(lazy, single[0], ignore_attrs)
    assert read_provenance(lazy) == read_provenance(single[0])
    xr.testing.assert_allclose(lazy.compute(), expected)
    assert all(da.chunks is not None for da in data_arrays(dask_stack))
    xr.testing.assert_identical(dask_stack.compute(), before)
    return out, lazy


def volume_of(members):
    """A DataTree with the sweeps as ``sweep_0``, ``sweep_1``, ..."""
    nodes = {"/": xr.Dataset(attrs={"Conventions": "ODIM_H5/V2_2"})}
    nodes.update({f"/sweep_{k}": m for k, m in enumerate(members)})
    return xr.DataTree.from_dict(nodes)


def check_volume(func, make, n=2, **kwargs):
    """A volume with dask sweeps gives lazy sweeps with the values of the sweeps."""
    members = [make(k) for k in range(n)]
    tree = volume_of(members)
    expected = func(tree, **kwargs)
    lazy_tree = volume_of([m.chunk({"azimuth": 7}) for m in members])
    lazy = func(lazy_tree, **kwargs)
    for node in lazy.subtree:
        if node is lazy or not node.has_data:
            continue
        assert_lazy(node.to_dataset(inherit=False))
    xr.testing.assert_allclose(lazy.compute(), expected)
    return expected, lazy
