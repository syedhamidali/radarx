#!/usr/bin/env python
# Copyright (c) 2024, radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx Grid
============

This sub-module contains functions necessary to grid the radar data.

``grid_radar(method="cone")`` is the radarx cone gridding of
:mod:`radarx.grid.cone`; ``grid_radar(method="barnes")`` is the Barnes (1964)
objective analysis, evaluated with the fast approximation of Zürcher (2023)
(see the References of :func:`grid_radar`). :func:`stack_data` and
:func:`make_3d_grid` are bookkeeping (stacking gates, building the target
grid); they implement no published algorithm.

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

from .._compat import keyword_only_after, renamed_arguments
from .._provenance import provenance
from .._sweepwise import dask_support
from ..utils import find_multidim_vars, get_geocoords  #  noqa


def _sweep_dataset(dtree, name):
    """
    Return a sweep node as a Dataset including the radar site coordinates.

    Newer xarray only inherits indexed coordinates by default, so the site
    ``latitude``/``longitude``/``altitude`` stored on the root node are lost
    unless ``inherit="all_coords"`` is requested.

    Parameters
    ----------
    dtree : xarray.DataTree
        Radar volume.
    name : str
        Sweep group name, e.g. ``"sweep_0"``.

    Returns
    -------
    xarray.Dataset
        Sweep dataset with inherited coordinates.
    """
    try:
        return dtree[name].to_dataset(inherit="all_coords")
    except (TypeError, ValueError):  # xarray without "all_coords"
        return dtree[name].to_dataset()


@dask_support("computes", "it stacks the sweeps of the volume into one Dataset.")
@renamed_arguments(dtree="obj")
@provenance("Stacking of georeferenced gates into points (no published method)")
def stack_data(obj, data_vars=None, geo=False):
    """
    Stack data from a radar DataTree into a single xarray Dataset.

    Parameters
    ----------
    obj : xradar.DataTree
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

    Notes
    -----
    Bookkeeping only (no published method): the gates are georeferenced with
    xradar (4/3 effective Earth radius model) and stacked into one point
    dimension.
    """
    obj = obj.xradar.georeference()
    if geo:
        obj = obj.xradar.map_over_sweeps(get_geocoords)
        x, y, z = "lon", "lat", "alt"
    else:
        x, y, z = "x", "y", "z"

    swp_list = []
    data_vars_dict = {}

    # Loop through sweeps
    for swp in obj.match("sweep_*"):
        ds = _sweep_dataset(obj, swp)

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


@renamed_arguments(ds="obj")
def make_3d_grid(
    obj,
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
    obj : xarray.Dataset
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
    - ``lon`` is the longitude along ``y = 0`` and ``lat`` the latitude along
      ``x = 0``, so the two axes may have different lengths. Because meridians
      converge, these axes are not the lon/lat of every cell; the cell-wise
      values are 2-D (``grid_radar(method="barnes")`` returns them).
    - The transformation ensures compatibility between Cartesian and
      geographic coordinates.
    - This function assumes the dataset is compatible with `xradar`
      and has a valid CRS.
    - The defaults (+-200 km, 1 km, 0 to 10 km at 250 m) are radarx choices,
      not taken from a publication.
    - The azimuthal equidistant projection and its WGS84 inverse are provided
      by PROJ (through pyproj); no published method is implemented here.

    """

    # Create Cartesian grid arrays
    x = np.arange(x_lim[0], x_lim[1] + x_step, x_step)
    y = np.arange(y_lim[0], y_lim[1] + y_step, y_step)
    z = np.arange(z_lim[0], z_lim[1] + z_step, z_step)

    transformer, trg_crs = _lonlat_transformer(obj)
    # Longitude along y = 0 and latitude along x = 0: x and y are axes of the
    # grid, not paired points, so they may differ in length.
    lon, _ = transformer.transform(x, np.zeros_like(x))
    _, lat = transformer.transform(np.zeros_like(y), y)
    return np.asarray(lat), np.asarray(lon), x, y, z, trg_crs


def _lonlat_transformer(ds):
    """
    Transformer from the grid projection of a radar dataset to lon/lat.

    Returns the ``(x, y) -> (lon, lat)`` transformer and the target CRS
    (EPSG:4326).
    """
    from pyproj import CRS, Transformer

    # Convert the dataset to georeferenced coordinates
    ds = ds.xradar.georeference()
    # Define source and target coordinate reference systems (CRS)
    src_crs = ds.xradar.get_crs()
    trg_crs = CRS.from_user_input(4326)  # EPSG:4326 (WGS 84)
    return Transformer.from_crs(src_crs, trg_crs, always_xy=True), trg_crs


@dask_support("computes", "the result is interpolated from the gates of all sweeps.")
@renamed_arguments(dtree="obj")
@keyword_only_after("n_threads", "fill_depth", "fill_velocity", "k_e")
@provenance("Radar volume gridding by cone interpolation or Barnes analysis")
def grid_radar(
    obj,
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
    method="cone",
    *,
    fill_depth=500.0,
    fill_velocity=False,
    k_e=4.0 / 3.0,
    n_threads=None,
):
    """
    Interpolate radar data to a 3D grid and optionally create a pseudo-CAPPI.

    Two methods are available. ``"cone"`` (default) interpolates within each
    sweep and then between sweeps (see :func:`radarx.grid.grid_cones`): it
    needs no smoothing parameters, keeps the native resolution and is much
    faster. ``"barnes"`` uses Barnes (1964) [1]_ objective analysis via the
    optional fast-barnes-py package (Python < 3.13 only), which implements the
    fast approximate Barnes interpolation of Zürcher (2023) [2]_.

    Parameters
    ----------
    obj : xradar.DataTree
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
        (0, 10e3).
    x_step : int, optional
        Grid resolution in the x-direction (meters). Defaults to 1000.
    y_step : int, optional
        Grid resolution in the y-direction (meters). Defaults to 1000.
    z_step : int, optional
        Grid resolution in the z-direction (meters). Defaults to 250.
    x_smth : float, optional
        Smoothing factor for the x-dimension: the Gaussian width of the Barnes
        kernel as a fraction of ``x_step``. Defaults to 0.2.
    y_smth : float, optional
        Smoothing factor for the y-dimension (fraction of ``y_step``).
        Defaults to 0.2.
    z_smth : float, optional
        Smoothing factor for the z-dimension (fraction of ``z_step``).
        Defaults to 1.
        ``x_smth``, ``y_smth`` and ``z_smth`` apply to ``method="barnes"`` only.
        The grid defaults and the three smoothing factors are radarx choices,
        not taken from Barnes (1964) or Zürcher (2023). The settings passed to
        fast-barnes-py (``max_dist=4`` and ``num_iter=4``) are radarx choices
        too; they are not taken from Zürcher (2023).
    method : {"cone", "barnes"}, optional
        Interpolation method. Default ``"cone"``.
    fill_depth : float, optional
        With ``method="cone"`` and ``pseudo_cappi=True``, the lowest sweep
        fills at most this many metres below itself and never below the radar
        altitude. Default 500 m, a radarx choice. At 150 km the lowest beam is
        about 2.5 km up, and the earlier unlimited fill copied it to the
        ground; ``fill_depth=numpy.inf`` restores that.
    fill_velocity : bool, optional
        With ``method="cone"``, also fill radial velocity below the lowest
        sweep. Default ``False``, because a radial velocity is a projection on
        the beam direction, which changes with height. ``True`` restores the
        earlier behaviour.
    k_e : float, optional
        Effective Earth radius factor of the beam height (Doviak and Zrnic
        1993, Eq. 2.28). Default 4/3. See
        :func:`radarx.fundamentals.geometry.k_e_from_refractivity`. Used by
        ``method="cone"``; the Barnes path georeferences with xradar's 4/3.
    n_threads : int, optional
        Threads for ``method="cone"``. Default: all cores.

    Returns
    -------
    xarray.Dataset
        Interpolated dataset with the specified variables and 3D grid.
        Includes longitude and latitude coordinates for the grid and the
        mean time of the volume as a scalar ``time`` coordinate, so grids of
        several volumes stack with ``xr.concat(grids, "time")``. With
        ``method="barnes"`` ``lat`` and ``lon`` are 2-D on ``(y, x)``, the
        position of every cell of the metric grid.

    Notes
    -----
    - With ``method="barnes"`` the pseudo-CAPPI is created by copying the value
      of the level above into every missing (NaN) cell of a level, working
      downward from the third-highest level to the lowest one (the two
      highest levels are not filled). This fills any NaN cell at any of these
      levels, including gaps inside a column, from the level above, not only
      the cells below the lowest echo; it is a radarx convention, not a
      published algorithm.
    - With ``method="cone"``, ``pseudo_cappi`` fills levels below the lowest
      sweep with that sweep's value; other cells not bracketed by two sweeps
      stay empty. The cone method is radarx's own: its closest published
      relative is the vertical interpolation (VI) scheme of Zhang et al. (2005)
      [3]_ (their Eqs. 5-7, a linear interpolation in elevation angle between
      the two tilts around the grid cell, nearest neighbour in azimuth and
      range). It differs in interpolating bilinearly within each sweep and
      linearly in beam height (not in elevation angle) between sweeps.
    - With ``method="barnes"``, interpolation uses Barnes objective analysis
      on the metric azimuthal equidistant grid (gate positions and the grid
      are both in metres from the radar), so the labelled ``x`` and ``y`` are
      the true cell positions. Earlier versions ran it on a regular lon/lat
      lattice built from the axes, which displaced cells by up to about
      1 km at 100 km from the radar.

    References
    ----------
    .. [1] Barnes, S. L., 1964: A technique for maximizing details in numerical
       weather map analysis. *J. Appl. Meteor.*, **3**, 396-409,
       https://doi.org/10.1175/1520-0450(1964)003<0396:ATFMDI>2.0.CO;2
    .. [2] Zürcher, B. K., 2023: Fast approximate Barnes interpolation:
       illustrated by Python-Numba implementation fast-barnes-py v1.0.
       *Geosci. Model Dev.*, **16**, 1697-1711,
       https://doi.org/10.5194/gmd-16-1697-2023
    .. [3] Zhang, J., K. Howard, and J. J. Gourley, 2005: Constructing
       three-dimensional multiple-radar reflectivity mosaics: Examples of
       convective storms and stratiform rain echoes. *J. Atmos. Oceanic
       Technol.*, **22**, 30-42, https://doi.org/10.1175/JTECH-1689.1

    """
    if method == "cone":
        from .cone import grid_cones

        x = np.arange(x_lim[0], x_lim[1] + x_step, x_step)
        y = np.arange(y_lim[0], y_lim[1] + y_step, y_step)
        z = np.arange(z_lim[0], z_lim[1] + z_step, z_step)
        return grid_cones(
            obj,
            data_vars,
            x,
            y,
            z,
            fill_below=pseudo_cappi,
            fill_depth=fill_depth,
            fill_velocity=fill_velocity,
            k_e=k_e,
            n_threads=n_threads,
        )
    if method != "barnes":
        raise ValueError(f"method must be 'cone' or 'barnes', not {method!r}")
    # fast-barnes-py is optional (it supports Python < 3.13 only), so import it
    # only when the Barnes method is requested
    try:
        from fastbarnes import interpolation
        from fastbarnes.interpolation import get_half_kernel_size
    except ImportError as err:
        raise ImportError(
            "grid_radar(method='barnes') needs the optional 'fast-barnes-py' "
            "package (Python < 3.13 only): 'pip install fast-barnes-py'. "
            "The default method='cone' has no extra dependencies."
        ) from err
    ds = stack_data(obj, data_vars=data_vars, geo=False)
    sweep0 = _sweep_dataset(obj, "sweep_0")
    _, _, x, y, z, _ = make_3d_grid(
        sweep0,
        x_lim=x_lim,
        y_lim=y_lim,
        x_step=x_step,
        y_step=y_step,
        z_lim=z_lim,
        z_step=z_step,
    )

    # The Barnes lattice is the metric grid itself: gates (x, y, z) and the
    # target are in metres in the same azimuthal equidistant projection.
    x0 = np.asarray([x.min(), y.min(), z.min()], dtype=np.float64)
    xstep = np.diff(x).mean()
    ystep = np.diff(y).mean()

    size = (len(x), len(y), len(z))

    # Adjust sigma to match grid steps
    sigma_x = x_smth * xstep
    sigma_y = y_smth * ystep
    sigma_alt = z_smth * z_step
    sigma = np.array([sigma_x, sigma_y, sigma_alt], dtype=np.float64)

    ds_out_fast = xr.Dataset({"z": ("z", z), "y": ("y", y), "x": ("x", x)})

    # Perform Barnes interpolation
    if data_vars is None:  # pragma: no cover
        data_vars = list(ds.data_vars)

    for var in data_vars:
        data_orig = ds[var].values.flatten()
        x_flat = ds["xyz"][:, 0].values.flatten()  # x east of the radar [m]
        y_flat = ds["xyz"][:, 1].values.flatten()  # y north of the radar [m]
        z_flat = ds["xyz"][:, 2].values.flatten()  # altitude [m]

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
    transformer, _ = _lonlat_transformer(sweep0)
    lon, lat = transformer.transform(*np.meshgrid(x, y))
    ds_out_fast = ds_out_fast.assign_coords(
        time=ds["time"].mean().variable,
        lat=(("y", "x"), np.asarray(lat), {"units": "degrees_north"}),
        lon=(("y", "x"), np.asarray(lon), {"units": "degrees_east"}),
    )
    ds_out_fast.attrs = dict(obj.attrs)
    ds_out_fast.attrs["radar_name"] = ds_out_fast.attrs.get("instrument_name", "")
    return ds_out_fast
