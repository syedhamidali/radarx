#!/usr/bin/env python
# Copyright (c) 2024, radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx Grid
============

This sub-module contains functions necessary to grid the radar data.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = [
    "stack_data",
    "make_3d_grid",
    "grid_radar",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np
import xarray as xr

try:  # pragma: no cover
    from fastbarnes import interpolation
    from fastbarnes.interpolation import get_half_kernel_size

    FASTBARNES_AVAILABLE = True
except ImportError:  # pragma: no cover
    FASTBARNES_AVAILABLE = False

from ..utils import get_geocoords, find_multidim_vars  #  noqa


def stack_data(dtree, data_vars=None, geo=False):
    """
    Stack data from a radar DataTree into a single xarray Dataset.

    Parameters
    ----------
    dtree : xradar.DataTree
        The input DataTree containing radar data.
    data_vars : list of str or None, optional
        A list of variables to stack. If None, all variables with more
        than one dimension are included.
    geo : bool, optional
        If True, converts coordinates to geographic (lon, lat, alt).

    Returns
    -------
    xarray.Dataset
        A stacked dataset with the specified or all multidimensional variables.
    """
    dtree = dtree.xradar.georeference()
    if geo:
        dtree = dtree.xradar.map_over_sweeps(get_geocoords)
        x, y, z = "lon", "lat", "alt"
    else:
        x, y, z = "x", "y", "z"

    swp_list = []
    data_vars_dict = {}

    # Loop through sweeps
    for swp in dtree.match("sweep_*"):
        ds = dtree[swp].to_dataset()

        # Find variables to include if data_vars is None
        if data_vars is None:  # pragma
            vars_to_stack = find_multidim_vars(ds, ndim=2)
        else:  # pragma: no cover
            vars_to_stack = data_vars

        # Stack coordinates
        xyz = (
            xr.concat(
                [
                    ds.coords[x].reset_coords(drop=True),
                    ds.coords[y].reset_coords(drop=True),
                    ds.coords[z].reset_coords(drop=True),
                ],
                "xyz",
            )
            .stack(npoints=("azimuth", "range"))
            .transpose(..., "xyz")
        )
        swp_list.append(xyz)

        # Stack data for each variable
        for v in vars_to_stack:
            if v not in data_vars_dict:
                data_vars_dict[v] = []
            data = ds[v].stack(npoints=("azimuth", "range"))
            data_vars_dict[v].append(data)

    # Combine stacked data
    xyz = xr.concat(swp_list, "npoints")
    dataset = xr.Dataset(coords={"xyz": xyz})

    for v, data_list in data_vars_dict.items():
        dataset[v] = xr.concat(data_list, "npoints")

    return dataset


def make_3d_grid(
    ds,
    x_lim=(-200e3, 200e3),
    y_lim=(-200e3, 200e3),
    x_step=1000,
    y_step=1000,
    z_lim=(0, 10e3),
    z_step=250,
):
    """
    Generate a 3D Cartesian grid and transform its coordinates to geographic
    latitude, longitude, and altitude using the dataset's CRS.

    Parameters
    ----------
    ds : xarray.Dataset
        Radar dataset with Cartesian coordinates.
    x_lim : tuple of float, optional
        The range of x-coords (m) in Cartesian space. Default is (-200e3, 200e3).
    y_lim : tuple of float, optional
        The range of y-coords (m) in Cartesian space. Default is (-200e3, 200e3).
    x_step : int, optional
        Step size (m) for x-coordinates. Default is 1000.
    y_step : int, optional
        Step size (meters) for y-coordinates. Default is 1000.
    z_lim : tuple of float, optional
        The range of z-coords (m) in Cartesian space. Default is (0, 10e3).
    z_step : int, optional
        Step size (meters) for z-coordinates. Default is 250.

    Returns
    -------
    tuple
        lat : numpy.ndarray
            Latitude values of the transformed grid.
        lon : numpy.ndarray
            Longitude values of the transformed grid.
        x : numpy.ndarray
            Cartesian x-coordinates (meters).
        y : numpy.ndarray
            Cartesian y-coordinates (meters).
        z : numpy.ndarray
            Cartesian z-coordinates (meters, altitude).
        trg_crs : pyproj.CRS
            Target geographic coordinate reference system (CRS).

    Notes
    -----
    - The function uses the Azimuthal Equidistant projection (AEQD)
    defined in the radar dataset.
    - The transformation ensures compatibility between Cartesian and
    geographic coordinates.
    - This function assumes the dataset is compatible with `xradar`
    and has a valid CRS.

    """

    # Create Cartesian grid arrays
    x = np.arange(x_lim[0], x_lim[1] + x_step, x_step)
    y = np.arange(y_lim[0], y_lim[1] + y_step, y_step)
    z = np.arange(z_lim[0], z_lim[1] + z_step, z_step)

    from pyproj import CRS, Transformer

    # Convert the dataset to georeferenced coordinates
    ds = ds.xradar.georeference()
    # Define source and target coordinate reference systems (CRS)
    src_crs = ds.xradar.get_crs()
    trg_crs = CRS.from_user_input(4326)  # EPSG:4326 (WGS 84)
    # Create a transformer for coordinate conversion
    transformer = Transformer.from_crs(src_crs, trg_crs)
    # Transform x, y, z coordinates to latitude, longitude, and altitude
    lat, lon = transformer.transform(x, y)
    return lat, lon, x, y, z, trg_crs


def grid_radar(
    dtree,
    data_vars=None,
    pseudo_cappi=True,
    x_lim=(-100e3, 100e3),
    y_lim=(-100e3, 100e3),
    z_lim=(0, 10e3),
    x_step=1000,
    y_step=1000,
    z_step=250,
    x_smth=0.2,
    y_smth=0.2,
    z_smth=1,
):
    """
    Interpolate radar data to a 3D grid and optionally create a pseudo-CAPPI.

    Parameters
    ----------
    dtree : xradar.DataTree
        Input radar DataTree containing radar sweeps.
    data_vars : list of str, optional
        List of variables to interpolate. If None, all variables in the dataset
        are used. Defaults to None.
    pseudo_cappi : bool, optional
        If True, extrapolates data to lower altitudes to create a pseudo-CAPPI.
        Defaults to True.
    x_lim : tuple of float, optional
        Range of x-coordinates (meters) for the Cartesian grid. Defaults to
        (-100e3, 100e3).
    y_lim : tuple of float, optional
        Range of y-coordinates (meters) for the Cartesian grid. Defaults to
        (-100e3, 100e3).
    z_lim : tuple of float, optional
        Range of z-coordinates (meters) for the Cartesian grid. Defaults to
        (0, 16e3).
    x_step : int, optional
        Grid resolution in the x-direction (meters). Defaults to 500.
    y_step : int, optional
        Grid resolution in the y-direction (meters). Defaults to 500.
    z_step : int, optional
        Grid resolution in the z-direction (meters). Defaults to 250.
    x_smth : float, optional
        Smoothing factor for the x-dimension. Defaults to 0.2.
    y_smth : float, optional
        Smoothing factor for the y-dimension. Defaults to 0.2.
    z_smth : float, optional
        Smoothing factor for the z-dimension. Defaults to 1.

    Returns
    -------
    xarray.Dataset
        Interpolated dataset with the specified variables and 3D grid.
        Includes longitude and latitude coordinates for the grid.

    Notes
    -----
    - The pseudo-CAPPI is created by extrapolating data from higher altitudes
      to fill missing values at lower altitudes.
    - Interpolation is performed using Barnes interpolation.

    """
    if not FASTBARNES_AVAILABLE:  # pragma: no cover
        raise ImportError(
            "The 'fastbarnes' package is required for this function. "
            "Install it via 'pip install fast-barnes-py'."
        )  # pragma: no cover
    ds = stack_data(dtree, data_vars=data_vars, geo=True)
    lat, lon, trgx, trgy, z, trg_crs = make_3d_grid(
        dtree["sweep_0"].to_dataset(),
        x_lim=x_lim,
        y_lim=y_lim,
        x_step=x_step,
        y_step=y_step,
        z_lim=z_lim,
        z_step=z_step,
    )

    x = lon
    y = lat

    x0 = np.asarray([x.min(), y.min(), z.min()], dtype=np.float64)
    xstep = np.diff(x).mean()  # Longitude step in degrees
    ystep = np.diff(y).mean()  # Latitude step in degrees

    size = (len(x), len(y), len(z))

    # Adjust sigma to match grid steps
    sigma_lon = x_smth * xstep  # Smoothing across 2 longitude grid steps
    sigma_lat = y_smth * ystep  # Smoothing across 2 latitude grid steps
    sigma_alt = z_smth * z_step  # Smoothing across 2 altitude grid steps
    sigma = np.array([sigma_lon, sigma_lat, sigma_alt], dtype=np.float64)

    ds_out_fast = xr.Dataset({"z": ("z", z), "y": ("y", y), "x": ("x", x)})

    # Perform Barnes interpolation
    if data_vars is None:  # pragma: no cover
        data_vars = list(ds.data_vars)

    for var in data_vars:
        data_orig = ds[var].values.flatten()
        x_flat = ds["xyz"][:, 0].values.flatten()  # Longitude
        y_flat = ds["xyz"][:, 1].values.flatten()  # Latitude
        z_flat = ds["xyz"][:, 2].values.flatten()  # Altitude

        mask = np.isfinite(data_orig)
        if np.sum(mask) == 0:
            print(f"No valid data points for variable {var}. Skipping...")
            continue

        xyz_data = np.vstack((x_flat[mask], y_flat[mask], z_flat[mask])).T
        data_values = data_orig[mask]

        # Validate kernel size
        kernel_size = (
            2 * get_half_kernel_size(sigma, [xstep, ystep, z_step], num_iter=4) + 1
        )
        if any(kernel_size > np.array(size)):  # pragma: no cover
            print(
                f"Kernel size {kernel_size} exceeds grid size {size}. Skipping..."
            )  # pragma: no cover
            continue

        # Perform interpolation
        field = interpolation.barnes(
            xyz_data,
            data_values,
            sigma,
            x0,
            [xstep, ystep, z_step],
            size,
            max_dist=4,
        )
        if pseudo_cappi:
            # Create pseudo-CAPPI by extrapolating to lower levels
            pseudo_cappi_field = field.copy()
            for z_idx in range(
                len(z) - 3, -1, -1
            ):  # Iterate from higher levels to lower levels
                upper_level = pseudo_cappi_field[
                    z_idx + 1, :, :
                ]  # Data from the level above
                current_level = pseudo_cappi_field[z_idx, :, :]  # Current level
                mask = np.isnan(current_level)  # Identify missing values at this level
                pseudo_cappi_field[z_idx, mask] = upper_level[
                    mask
                ]  # Fill missing with the level above

            # Add pseudo-CAPPI field to the output dataset
            ds_out_fast[var] = (("z", "y", "x"), pseudo_cappi_field)
        else:  # pragma: no cover
            ds_out_fast[var] = (("z", "y", "x"), field)
    # Assign metadata
    ds_out_fast["time"] = ds.time.mean()
    ds_out_fast = ds_out_fast.rename({"x": "lon", "y": "lat"})
    ds_out_fast["x"] = xr.DataArray(trgx, dims="lon")
    ds_out_fast["y"] = xr.DataArray(trgy, dims="lat")
    ds_out_fast = ds_out_fast.set_coords(["x", "y"])
    ds_out_fast = ds_out_fast.swap_dims({"lon": "x", "lat": "y"})
    ds_out_fast.attrs = dtree.attrs
    ds_out_fast.attrs["radar_name"] = ds_out_fast.attrs.get("instrument_name", "")
    return ds_out_fast
