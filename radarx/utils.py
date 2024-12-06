#!/usr/bin/env python
# Copyright (c) 2024, radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx Utils
============

This sub-module contains utilitiy functions for various purposes.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = [
    "get_geocoords",
    "cartesian_to_geographic_aeqd",
    "find_multidim_vars",
    "combine_nexrad_sweeps",
]

__doc__ = __doc__.format("\n   ".join(__all__))

from pyproj import CRS, Transformer
import xradar as xd
import xarray as xr
import numpy as np
from xarray import DataTree
import warnings


def get_geocoords(ds):
    """
    Converts Cartesian coordinates (x, y, z) in a radar dataset to geographic
    coordinates (longitude, latitude, altitude) using CRS transformation.

    Parameters
    ----------
    ds : xarray.Dataset
        Radar dataset with Cartesian coordinates.

    Returns
    -------
    xarray.Dataset
        Dataset with added 'lon', 'lat', and 'alt' coordinates and their attributes.
    """
    # Convert the dataset to georeferenced coordinates
    ds = ds.xradar.georeference()
    # Define source and target coordinate reference systems (CRS)
    src_crs = ds.xradar.get_crs()
    trg_crs = CRS.from_user_input(4326)  # EPSG:4326 (WGS 84)
    # Create a transformer for coordinate conversion
    transformer = Transformer.from_crs(src_crs, trg_crs)
    # Transform x, y, z coordinates to latitude, longitude, and altitude
    trg_y, trg_x, trg_z = transformer.transform(ds.x, ds.y, ds.z)
    # Assign new coordinates with appropriate attributes
    ds = ds.assign_coords(
        {
            "lon": (ds.x.dims, trg_x, xd.model.get_longitude_attrs()),
            "lat": (ds.y.dims, trg_y, xd.model.get_latitude_attrs()),
            "alt": (ds.z.dims, trg_z, xd.model.get_altitude_attrs()),
        }
    )
    return ds


def cartesian_to_geographic_aeqd(x, y, lon_0, lat_0, earth_radius):
    """
    Transform Cartesian coordinates (x, y) to geographic
    coordinates (latitude, longitude) using the Azimuthal
    Equidistant (AEQD) map projection.

    Parameters
    ----------
    x, y : array-like
        Cartesian coordinates in the same units as the Earth's
        radius, typically meters.
    lon_0, lat_0 : float
        Longitude and latitude, in degrees, of the center of the
        projection.
    earth_radius : float
        Radius of the Earth in meters.
    Returns
    -------
    lon, lat : array
        Longitude and latitude of Cartesian coordinates in degrees.
    Notes
    -----
    The calculations follow the AEQD projection equations, where the
    Earth's radius is used to define the distance metric.
    """
    # Ensure x and y are at least 1D arrays
    x = np.atleast_1d(np.asarray(x))
    y = np.atleast_1d(np.asarray(y))

    # Convert reference latitude and longitude to radians
    lat_0_rad = np.deg2rad(lat_0)
    lon_0_rad = np.deg2rad(lon_0)

    # Calculate distance (rho) and angular distance (c)
    rho = np.sqrt(x**2 + y**2)
    c = rho / earth_radius

    # Suppress warnings for potential division by zero
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lat_rad = np.arcsin(
            np.cos(c) * np.sin(lat_0_rad) + (y * np.sin(c) * np.cos(lat_0_rad) / rho)
        )

    # Convert latitude to degrees and handle edge cases where rho == 0
    lat_deg = np.rad2deg(lat_rad)
    lat_deg[rho == 0] = lat_0

    # Calculate longitude in radians
    x1 = x * np.sin(c)
    x2 = rho * np.cos(lat_0_rad) * np.cos(c) - y * np.sin(lat_0_rad) * np.sin(c)
    lon_rad = lon_0_rad + np.arctan2(x1, x2)

    # Convert longitude to degrees and normalize to [-180, 180]
    lon_deg = np.rad2deg(lon_rad)
    lon_deg = (lon_deg + 180) % 360 - 180  # Normalize longitude

    return lon_deg, lat_deg


def find_multidim_vars(ds, ndim=2):
    """
    Find variables in an xarray Dataset that have more than one dimension.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset.

    Returns
    -------
    list
        A list of variable names with more than one dimension.
    """
    return [var for var in ds.data_vars if ds[var].ndim == ndim]


def combine_nexrad_sweeps(dtree):
    """
    Combine radar sweeps with identical fixed angles into a new `DataTree`,
    aligning and merging datasets where necessary.

    Parameters
    ----------
    dtree : xarray.DataTree
        Input DataTree containing radar sweeps as child nodes. Each sweep
        is expected to be a dataset with a `sweep_fixed_angle` coordinate.

    Returns
    -------
    xarray.DataTree
        A new DataTree with combined sweeps. Sweeps with identical fixed
        angles are merged, and the resulting sweeps are re-indexed in order.

    Notes
    -----
    - Sweeps with identical fixed angles are grouped and merged. The sweep
      with the largest range is used as the primary dataset, and smaller
      sweeps are aligned and merged into it.
    - If variable names conflict (e.g., `DBZH`), variables from secondary
      datasets are renamed (e.g., `DBZH_SHORT`) to avoid overwrites.
    - Sweeps are re-assigned with new sequential sweep numbers in the output
      DataTree.
    - Nodes that are not sweeps are retained in the new DataTree.

    Examples
    --------
    Combine sweeps in a radar DataTree:

    >>> combined_dtree = combine_sweeps(radar_dtree)

    """
    dtree_copy = dtree.copy(deep=True)
    sweep_groups = {}

    # Group sweeps by their fixed angle
    for swp in dtree.match("sweep_*"):
        ds = dtree[swp].to_dataset()
        angle = ds.sweep_fixed_angle.values.item()
        if angle not in sweep_groups:
            sweep_groups[angle] = []
        sweep_groups[angle].append((swp, ds))

    # Create a new DataTree to store combined sweeps in sorted order
    new_dtree = DataTree()

    # Process each group with matching angles
    for angle, sweeps in sweep_groups.items():
        if len(sweeps) == 1:  # pragma: no cover
            # Add single sweep nodes directly to the new DataTree
            swp, ds = sweeps[0]
            new_dtree[swp] = DataTree(ds)
            continue

        # Sort sweeps by range dimension size in descending order
        sweeps = sorted(sweeps, key=lambda x: x[1].sizes["range"], reverse=True)

        # The largest range dataset is the primary dataset
        primary_swp, primary_ds = sweeps[0]

        for secondary_swp, secondary_ds in sweeps[1:]:  # Process smaller range datasets
            # Align secondary dataset to the primary dataset's range
            aligned_secondary_ds = secondary_ds.reindex_like(
                primary_ds, method="nearest"
            )

            # Rename variables in the secondary dataset to avoid overwrites
            if "DBZH" in aligned_secondary_ds:
                aligned_secondary_ds = aligned_secondary_ds.rename(
                    {"DBZH": "DBZH_SHORT"}
                )

            # Merge aligned secondary dataset into the primary dataset
            primary_ds = xr.merge([primary_ds, aligned_secondary_ds], compat="override")

        # Add the combined dataset to the new DataTree
        new_dtree[primary_swp] = DataTree(primary_ds)

    # Reassign sweep numbers in sorted order
    sweeps_sorted = sorted(
        [key for key in new_dtree.groups if key.startswith("/sweep_")],
        key=lambda k: int(k.split("_")[-1]),
    )

    ordered_dtree = DataTree(dataset=dtree_copy.root.to_dataset())
    for new_index, swp in enumerate(sweeps_sorted):
        # Update the sweep_number and reassign the node to maintain order
        ds = new_dtree[swp].to_dataset()
        ds = ds.assign_coords(sweep_number=new_index)
        ordered_dtree[f"/sweep_{new_index}"] = DataTree(ds)

    # Copy other nodes (non-sweeps) back to the new DataTree
    for key in dtree.groups:
        if not key.startswith("/sweep_") and key != "/":  # pragma: no cover
            ordered_dtree[key] = dtree_copy[key]
    ordered_dtree.attrs = dtree.attrs
    return ordered_dtree
