#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Finding fields and sweeps
=========================

Helpers shared by the retrievals: the candidate names of a field, a search
for the first candidate a sweep or volume holds, :func:`as_sweep`, which
lets a single sweep node of a volume (``dtree["sweep_0"]``) stand for a sweep,
and :func:`align_like`, which matches a mask to a field by label.
"""

from __future__ import annotations

import xarray as xr

__all__ = ["VELOCITY", "align_like", "as_sweep", "find_field"]

# Radial velocity, in the order of preference: xradar's ``VRADH`` first, then
# ``VRAD`` (single polarisation), the vertical channel, and other spellings.
VELOCITY = (
    "VRADH",
    "VRAD",
    "VRADV",
    "VEL",
    "VELOCITY",
    "velocity",
    "radial_velocity",
)


def as_sweep(obj):
    """
    Return a sweep node as a Dataset.

    A DataTree node without ``sweep*`` children (a leaf such as
    ``dtree["sweep_0"]``) is one sweep. It is returned as a Dataset that
    includes the coordinates it inherits from its parents (the radar site),
    so that it is processed as a single sweep. Datasets and volumes are
    returned unchanged.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataTree
        Input of a retrieval.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
    """
    if not isinstance(obj, xr.DataTree):
        return obj
    if any(name.startswith("sweep") for name in obj.children):
        return obj
    try:
        return obj.to_dataset(inherit="all_coords")
    except (TypeError, ValueError):  # xarray without "all_coords"
        return obj.to_dataset()


def _holds(node, name):
    return name in node.data_vars or name in node.coords


def find_field(obj, candidates, name=None):
    """
    Name of a field in a sweep or in the sweeps of a volume.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataTree
        A sweep, or a volume whose ``sweep*`` nodes are searched.
    candidates : tuple of str
        Names to try, in order of preference. The first one that the sweep (or
        any sweep of the volume) holds is returned.
    name : str, optional
        An explicit name. It is returned as it is, so that a missing field
        is reported where it is used.

    Returns
    -------
    str

    Raises
    ------
    KeyError
        If none of the candidates is found.
    TypeError
        If ``obj`` is not a Dataset or a DataTree and no name is given.
    """
    if name is not None:
        return name
    if not isinstance(obj, (xr.Dataset, xr.DataTree)):
        raise TypeError(
            f"expected an xarray.Dataset sweep or DataTree, not {type(obj).__name__}"
        )
    if isinstance(obj, xr.DataTree):
        nodes = [n for k, n in obj.children.items() if k.startswith("sweep")]
        nodes = nodes or [obj]
    else:
        nodes = [obj]
    for cand in candidates:
        if any(_holds(node, cand) for node in nodes):
            return cand
    raise KeyError(f"none of {tuple(candidates)} found; pass the field name")


def align_like(arr, like, what="the array"):
    """
    Put ``arr`` on the labels of ``like`` and raise if that is not possible.

    For every dimension that both objects index, the labels of ``like`` must
    all be in ``arr``. ``arr`` is then reordered (or cut) to the labels of
    ``like``, as ``reindex_like`` does, so that a mask whose rays come in a
    different order still selects the same rays. Labels of ``like`` that
    ``arr`` lacks (it would be filled with NaN) and duplicated labels raise
    an error. Dimensions without labels are matched by size by the caller.

    Parameters
    ----------
    arr : xarray.DataArray
        The mask or field to align.
    like : xarray.DataArray
        The field whose labels are the reference.
    what : str, optional
        Name of ``arr`` for the error message.

    Returns
    -------
    xarray.DataArray

    Raises
    ------
    ValueError
        If the labels of a shared dimension do not match.
    """
    for dim in arr.dims:
        if dim not in like.dims or dim not in arr.indexes or dim not in like.indexes:
            continue
        have, want = arr.indexes[dim], like.indexes[dim]
        if have.equals(want):
            continue
        if have.is_unique and want.is_unique and want.isin(have).all():
            arr = arr.reindex({dim: want})
        else:
            raise ValueError(
                f"{what} does not match the shape or the labels of {dim!r} of "
                f"the field ({len(have)} labels against {len(want)}; every "
                "label of the field must be in it)"
            )
    return arr
