#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Floating point precision of results
===================================

One rule for every product of radarx: a floating point *field* that a
function returns (a variable on gates, grid cells, profile levels or
couplets) is ``float32``, whatever the precision of the input. The kernels
and the NumPy fallbacks compute in ``float64`` and round once, at the end, so
both engines give the same values to ``float32`` rounding. Coordinates keep
their precision (``float64`` for positions and times). Integer, boolean and
flag variables keep their integer type.
"""

import numpy as np

__all__ = ["FIELD_DTYPE", "as_field", "fields_as_float32"]

FIELD_DTYPE = np.dtype("float32")


def as_field(values):
    """
    Round floating point values to the precision of radarx fields.

    Parameters
    ----------
    values : numpy.ndarray, dask.array.Array or xarray.DataArray
        Values to convert. Integer and boolean values are returned as they
        are.

    Returns
    -------
    same type as ``values``
        ``float32`` values (no copy if they already are).
    """
    if np.dtype(values.dtype).kind != "f":
        return values
    return values.astype(FIELD_DTYPE, copy=False)


def fields_as_float32(ds):
    """
    Round the floating point data variables of a Dataset to ``float32``.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with the products of a function.

    Returns
    -------
    xarray.Dataset
        Dataset whose floating point data variables are ``float32``;
        attributes and coordinates are kept.
    """
    return ds.assign({name: as_field(da) for name, da in ds.data_vars.items()})
