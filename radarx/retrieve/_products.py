#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Helpers shared by the retrievals: product volumes and merging products.

Every retrieval returns its products only. For a sweep that is a Dataset (or
DataArray) of products on the sweep's coordinates; for a volume, a DataTree
with the root of the input and one node of products per processed sweep.
:func:`assign_products` adds such products to the input again.
"""

from __future__ import annotations

import xarray as xr


def product_tree(obj, nodes):
    """
    DataTree of products: the input's own root dataset plus one node per sweep.

    Parameters
    ----------
    obj : xarray.DataTree
        The input volume.
    nodes : dict
        Sweep path -> Dataset of products.
    """
    root = obj.to_dataset(inherit=False)
    tree = {"/": root}
    for path, ds in nodes.items():
        # coordinates the root holds (e.g. the radar site) are inherited
        tree[path] = ds.drop_vars(
            [c for c in ds.coords if c in root.coords and c not in ds.indexes]
        )
    return xr.DataTree.from_dict(tree)


def _as_dataset(products):
    if isinstance(products, xr.DataArray):
        if products.name is None:
            raise ValueError("a DataArray of products needs a name")
        return products.to_dataset()
    if isinstance(products, xr.Dataset):
        return products
    raise TypeError(
        "products must be an xarray.Dataset or a named xarray.DataArray, "
        f"not {type(products).__name__}"
    )


def assign_dataset(ds, products, inherited=None):
    """
    Add the product variables of ``products`` to the sweep ``ds``.

    Products are aligned to the sweep's indexes (a left join: gates the
    products do not cover are NaN). Coordinates the sweep already has, or
    inherits (``inherited``: the sweep with the coordinates of its parents),
    are taken from the sweep; product variables replace sweep variables of
    the same name.
    """
    products = _as_dataset(products)
    full = ds if inherited is None else inherited
    duplicate = [
        name
        for name in products.coords
        if name in full.variables and name not in products.indexes
    ]
    products = products.drop_vars(duplicate)
    if any(name in full.indexes for name in products.indexes):
        _, products = xr.align(full, products, join="left", copy=False)
    return ds.assign(dict(products.data_vars))


def assign_products(target, products):
    """
    Merge retrieval products into a sweep or a volume.

    Parameters
    ----------
    target : xarray.Dataset or xarray.DataTree
        The sweep or volume the products were computed from.
    products : xarray.Dataset, xarray.DataArray or xarray.DataTree
        Products for a sweep (Dataset or named DataArray) or, for a volume,
        a DataTree of products per sweep, as returned by the retrievals.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        A copy of ``target`` with the product variables added to the
        matching sweeps. The root of a product tree is not merged.

    Raises
    ------
    KeyError
        If a product node has no matching node in ``target``.
    TypeError
        For mismatching input types.
    """
    if isinstance(target, xr.Dataset):
        if isinstance(products, xr.DataTree):
            raise TypeError("products for a Dataset must be a Dataset or DataArray")
        return assign_dataset(target, products)
    if not isinstance(products, xr.DataTree):
        raise TypeError("products for a DataTree must be a DataTree of products")
    out = target.copy()
    for node in products.subtree:
        if node is products or not node.has_data:
            continue
        path = node.relative_to(products)
        node_ds = node.to_dataset(inherit=False)
        if not node_ds.data_vars:
            continue
        try:
            dest = out[path]
        except KeyError:
            dest = None
        if not isinstance(dest, xr.DataTree):
            raise KeyError(f"{path!r} of the products is not in the volume")
        dest.dataset = assign_dataset(
            dest.to_dataset(inherit=False), node_ds, dest.to_dataset()
        )
    return out
