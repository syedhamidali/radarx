#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Helpers shared by the noise and calibration modules.

Field lookup, the choice between the compiled kernel and NumPy, the mapping
of a function over the sweeps of a DataTree and the lazy (dask) application
of a block function to DataArrays.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from ..grid.cone import _sweep_dataset, _sweep_names

DBZ_NAMES = ("DBZH", "DBZ", "reflectivity", "corrected_reflectivity")
ZDR_NAMES = ("ZDR", "differential_reflectivity", "corrected_differential_reflectivity")
RHOHV_NAMES = ("RHOHV", "cross_correlation_ratio", "RHO")
PHIDP_NAMES = ("PHIDP", "differential_phase", "UPHIDP")
KDP_NAMES = ("KDP", "specific_differential_phase")
SNR_NAMES = ("SNR", "SNRH", "signal_to_noise_ratio")


class MissingField(KeyError):
    """A field the function needs is not in a sweep (the sweep is skipped)."""


def find_field(ds, name, candidates, what, required=True):
    """
    Name of a variable of ``ds``: ``name`` if given, else the first candidate.

    Returns None if the field is optional and absent.
    """
    if name is not None:
        if name not in ds:
            raise MissingField(f"{name!r} is not in the dataset")
        return name
    for cand in candidates:
        if cand in ds:
            return cand
    if required:
        raise MissingField(f"no {what} found (tried {candidates}); pass its name")
    return None


def primary_field(ds, value, candidates, what):
    """The field a function is about: ``obj`` itself if it is a DataArray."""
    if isinstance(value, xr.DataArray):
        return value
    if isinstance(ds, xr.DataArray):
        return ds
    return ds[find_field(ds, value, candidates, what)]


def choose_engine(engine, has_kernel, label):
    """Whether to run the compiled kernel for the requested ``engine``."""
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not has_kernel:
        raise ImportError(f"the compiled {label} kernel is not available")
    return has_kernel and engine != "numpy"


def node_arg(arg, name):
    """The part of an argument that belongs to sweep ``name``.

    A DataTree (or a dict) holds one entry per sweep; any other value is
    used for every sweep.
    """
    if isinstance(arg, (xr.DataTree, dict)):
        if name not in arg:
            raise MissingField(f"no entry for {name!r}")
        node = arg[name]
        return node.to_dataset() if isinstance(node, xr.DataTree) else node
    return arg


def map_sweeps(dtree, func, what):
    """
    Apply ``func(dataset, name)`` to the sweeps of a volume.

    Returns a DataTree with the root of the input and one node per sweep that
    has the fields. Sweeps without them are skipped.
    """
    nodes = {"/": dtree.root.to_dataset(inherit=False)}
    missing = None
    for name in _sweep_names(dtree):
        try:
            res = func(_sweep_dataset(dtree, name), name)
        except MissingField as err:
            missing = err
            continue
        nodes[name] = res.to_dataset() if isinstance(res, xr.DataArray) else res
    if len(nodes) == 1:
        raise KeyError(
            f"no sweep has the fields {what} needs"
            + (f" ({missing.args[0]})" if missing is not None else "")
        )
    return xr.DataTree.from_dict(nodes)


def check_object(obj):
    if not isinstance(obj, (xr.DataArray, xr.Dataset, xr.DataTree)):
        raise TypeError("obj must be an xarray DataArray, Dataset or DataTree")


def ray_dimension(da):
    """The dimension of a field along which the rays are numbered."""
    if "range" not in da.dims:
        raise ValueError(f"the field needs a 'range' dimension, not {da.dims}")
    others = [d for d in da.dims if d != "range"]
    if not others:
        raise ValueError("the field needs a ray dimension besides 'range'")
    for cand in ("azimuth", "time"):
        if cand in others:
            return cand
    return others[0]


def range_meters(da):
    """The range coordinate of a field in m, as a DataArray."""
    if "range" not in da.coords:
        raise ValueError("the field has no 'range' coordinate")
    return da["range"].astype(np.float64)


def broadcast_block(*blocks):
    """Broadcast the blocks of an element-wise function to flat float64 arrays."""
    arrays = np.broadcast_arrays(*blocks)
    shape = arrays[0].shape
    flat = [np.ascontiguousarray(a, dtype=np.float64).reshape(-1) for a in arrays]
    return shape, flat


def elementwise(func, arrays, dtype=np.float32):
    """
    Apply a flat element-wise function to DataArrays, lazily for dask arrays.

    ``func`` takes one flat float64 array per input and returns a flat array.
    The inputs are aligned and broadcast against each other by xarray.
    """

    def block(*blocks):
        shape, flat = broadcast_block(*blocks)
        return func(*flat).reshape(shape).astype(dtype)

    return xr.apply_ufunc(
        block,
        *arrays,
        dask="parallelized",
        output_dtypes=[dtype],
        keep_attrs=False,
    )


def attrs_of(da, **new):
    """Attributes of a result: the standard name of the input kept if given."""
    out = {}
    if da is not None and "standard_name" in da.attrs and "standard_name" not in new:
        out["standard_name"] = da.attrs["standard_name"]
    out.update(new)
    return out
