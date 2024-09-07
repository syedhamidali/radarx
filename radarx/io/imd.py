#!/usr/bin/env python
# Copyright (c) 2024, radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
IMD Data Reader
===============

This sub-module provides functionality to read and process single radar files
from the Indian Meteorological Department (IMD), returning a quasi-CF-Radial
xarray Dataset.

Example:

    import radarx as rx
    dtree = rx.io.read_sweep(filename)
    dtree = rx.io.read_volume(filename)

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}

Author:
    Hamid Ali Syed
    Email: hamidsyed37@gmail.com
"""


import itertools

import numpy as np
import xarray as xr
from datatree import DataTree

__all__ = ["read_sweep", "read_volume"]


def read_sweep(file):
    """
    Reads a single radar file from IMD and processes it into a quasi-CF-Radial dataset.

    Parameters
    ----------
    file : str or Path
        Path to the radar file.

    Returns
    -------
    ds : xarray.Dataset
        Quasi-CF-Radial dataset with renamed coordinates, variables, and calculated fields.
    """
    ds = xr.open_dataset(file, engine="netcdf4")

    # Rename dimensions and variables for consistency with CF-Radial format
    ds = ds.rename_dims({"radial": "azimuth", "bin": "range"})
    ds = ds.rename_vars(
        {
            "radialAzim": "azimuth",
            "radialElev": "elevation",
            "elevationAngle": "fixed_angle",
            "radialTime": "time",
            "nyquist": "nyquist_velocity",
            "unambigRange": "unambiguous_range",
        }
    )
    ds = ds.set_coords(["azimuth", "elevation"])

    # Rename site-related coordinates
    site_coords = {"siteLat": "latitude", "siteLon": "longitude", "siteAlt": "altitude"}
    ds = ds.rename_vars(site_coords)

    # Add ray gate spacing variable
    ds["ray_gate_spacing"] = xr.DataArray(
        data=np.repeat(ds.gateSize.values, ds.azimuth.size),
        dims=("azimuth",),
        attrs={"long_name": "gate_spacing_for_ray", "units": "meters"},
    )

    # Rename moments for radar variables
    moments = {
        "T": "DBT",  # Total power
        "Z": "DBZ",  # Reflectivity
        "V": "VEL",  # Velocity
        "W": "WIDTH",  # Spectrum width
    }
    # Goes First in order
    ds = ds.rename_vars(moments)

    # Compute range and assign to dataset, Goes 2nd
    ds = ds.pipe(_compute_range)

    # Compute angle resolution, Goes 3rd
    ds = ds.pipe(_angle_resolution)

    # Determine scan type and mode, Goes 4th
    ds = ds.pipe(_scantype)

    # Time coverage handling, Goes 5th
    ds = ds.pipe(_time_coverage)

    # Handle sweep numbers, Goes 6th
    ds = ds.pipe(_sweep_number)

    # Assign start_end_ray_index, Goes 6th
    ds = ds.pipe(_assign_sweep_metadata)

    ds = ds.pipe(_assign_metadata)

    ds = ds.swap_dims({"azimuth": "time"})

    return ds


def read_volume(files):
    """
    Reads multiple files and creates a volume scan data.

    Parameters
    ----------
    files : list of files or list of lists

    Returns
    -------
    DataTree of the volumes
    """
    # Check if there are nested lists and merge them if needed
    if any(isinstance(i, list) for i in files):
        files = _merge_file_lists(files)

    # Determine volume groups from files
    grouped_dataset_list = _determine_volumes(files)

    # Create a list to store the volumes
    volume_datasets = []

    # Iterate over each group of datasets (which represent a volume)
    for datasets in grouped_dataset_list:
        vol = create_volume(datasets)  # Custom function to create the volume
        if isinstance(vol, xr.Dataset):
            volume_datasets.append(vol)  # Only add if it's an xarray.Dataset
        del vol  # Cleanup after each volume is processed

    # Construct the DataTree from the volume datasets (ensure that we pass datasets, not a list)
    volumes = DataTree.from_dict(
        {f"volume_{i}": volume for i, volume in enumerate(volume_datasets)}
    )

    return volumes


def _merge_file_lists(file_lists):
    """
    Merges internal lists of files into a single list.

    Parameters
    ----------
    file_lists : list of lists
        A list containing multiple internal lists of file paths.

    Returns
    -------
    merged_list : list
        A single list containing all the file paths from the internal lists.
    """
    # Use itertools.chain to flatten the list of lists
    merged_list = list(itertools.chain(*file_lists))

    return merged_list


def _compute_range(ds):
    """
    Computes the range to each gate and adds it to the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with computed range coordinates.
    """
    # Using ds.sizes instead of ds.dims to avoid future warning
    gate_count = ds.sizes["range"]

    # Ensure proper conversion to avoid deprecation warning
    ranges = np.arange(
        ds["firstGateRange"].values,
        gate_count * ds["gateSize"].values + ds["firstGateRange"].values,
        ds["gateSize"].astype(int).values,  # Ensure proper integer type conversion
    )

    ds.coords["range"] = xr.DataArray(
        ranges,
        dims=("range",),
        attrs={
            "standard_name": "range_to_center_of_measurement_volume",
            "long_name": "Range from instrument to center of gate",
            "units": "meters",
            "spacing_is_constant": "true",
            "meters_to_center_of_first_gate": ds["firstGateRange"].values,
            "meters_between_gates": ds["gateSize"].values,
        },
    )
    return ds


def _angle_resolution(ds):
    """
    Adds angular resolution for each ray to the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with angular resolution variable.
    """
    ds = ds.drop_dims("sweep", errors="ignore")
    ds["ray_angle_res"] = xr.DataArray(
        data=[ds["angleResolution"]],
        dims=("sweep",),
        attrs={"long_name": "angular_resolution_between_rays", "units": "degrees"},
    )
    ds = ds.drop_vars("angleResolution", errors="ignore")
    return ds


def _scantype(ds):
    """
    Determines the scan type based on the scanType variable.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with scan type and sweep mode variables.
    """
    scan_type_lookup = {
        4: (b"azimuth_surveillance", b"ppi"),
        7: (b"elevation_surveillance", b"rhi"),
        1: (b"sector", b"ppi_sector"),
        2: (b"manual_rhi", b"rhi_sector"),
        0: (b"unknown", b"unknown"),
    }

    # Extract the scalar value from the array
    scan_type = int(ds["scanType"].astype(int).values)

    # Get sweep_mode and scan_type_name from lookup dictionary
    sweep_mode, scan_type_name = scan_type_lookup.get(
        scan_type, (b"unknown", b"unknown")
    )

    ds["sweep_mode"] = xr.DataArray(data=[sweep_mode], dims=("sweep"))
    ds["scan_type"] = xr.DataArray(data=[scan_type_name], dims=("sweep"))
    #     ds = ds.drop_vars("scanType", errors="ignore")
    ds["fixed_angle"] = xr.DataArray(data=[ds["fixed_angle"]], dims=("sweep",))
    return ds


def _time_coverage(ds):
    """
    Processes time coverage start and end times.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with time coverage start and end variables.
    """
    ds = ds.rename_vars({"esStartTime": "time_coverage_start"})
    ds["time_coverage_start"] = xr.DataArray(
        data=str(
            ds.time_coverage_start.dt.strftime("%Y-%m-%dT%H:%M:%SZ").values
        ).encode("utf-8"),
        attrs={"standard_name": "data_volume_start_time_utc"},
    )

    ds["time_coverage_end"] = xr.DataArray(
        data=str(ds.time.max().dt.strftime("%Y-%m-%dT%H:%M:%SZ").values).encode(
            "utf-8"
        ),
        attrs={"standard_name": "data_volume_end_time_utc"},
    )
    return ds


def _sweep_number(ds):
    """
    Adds sweep number information to the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with sweep number variable.
    """
    sweep_num = ds["elevationNumber"].astype(int).values
    ds["sweep_number"] = xr.DataArray(
        data=[sweep_num],
        dims="sweep",
        attrs={"long_name": "sweep_index_number_0_based", "units": ""},
    )
    ds = ds.drop_vars("elevationNumber")
    return ds


def _assign_sweep_metadata(ds):
    """
    Assigns and updates the sweep metadata after reading multiple files.

    Parameters
    ----------
    ds : xarray.Dataset
        Combined dataset with multiple sweeps.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with updated sweep metadata.
    """
    num_sweeps = ds.sizes["sweep"]
    sweep_start = np.arange(0, num_sweeps * ds.sizes["azimuth"], ds.sizes["azimuth"])
    sweep_end = sweep_start + ds.sizes["azimuth"] - 1

    ds["sweep_start_ray_index"] = xr.DataArray(
        sweep_start,
        dims="sweep",
        attrs={"long_name": "Index of the first ray in the sweep"},
    )
    ds["sweep_end_ray_index"] = xr.DataArray(
        sweep_end,
        dims="sweep",
        attrs={"long_name": "Index of the last ray in the sweep"},
    )

    # Update sweep number if not done already
    if "sweep_number" not in ds:
        ds["sweep_number"] = xr.DataArray(
            np.arange(num_sweeps), dims="sweep", attrs={"long_name": "Sweep number"}
        )
    return ds


def _find_variables_by_coords(ds, required_dims):
    """
    Finds variables in the dataset that have the specified dimensions.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset.

    required_dims : set
        A set of required dimensions (e.g., {"range", "azimuth"}).

    Returns
    -------
    matching_vars : list
        List of variable names that have the specified dimensions.

    Example
    -------
    Assuming `ds` is your xarray.Dataset and you want to find variables
    with both "range" and "azimuth" dimensions
    vars_with_rng_and_az = find_variables_by_coords(ds, {"range", "azimuth"})
    """
    matching_vars = []

    for var_name, var_data in ds.data_vars.items():
        var_dims = set(var_data.dims)
        if required_dims.issubset(var_dims):
            matching_vars.append(var_name)

    return matching_vars


def _assign_metadata(ds):
    field_names = _find_variables_by_coords(ds, {"azimuth", "range"})
    field_names = ", ".join(field_names)
    ds.attrs = {
        "Conventions": "CF/Radial instrument_parameters",
        "version": "1.3",
        "title": "",
        "institution": "India Meteorological Department",
        "references": "",
        "source": "",
        "comment": "im/radarx",
        "instrument_name": "",
        "history": "",
        "field_names": field_names,
    }

    return ds


def _determine_nsweeps(files):
    """
    Determine the number of sweeps (nsweeps) and group files
    accordingly into sweep_groups.
    If the nsweeps cannot be determined from metadata, the
    function will detect where the sweep elevations start
    decreasing to identify the start of a new volume.
    """
    import re

    import xarray as xr

    def _natural_sort_key(s, _re=re.compile(r"(\d+)")):
        return [int(t) if i & 1 else t.lower() for i, t in enumerate(_re.split(s))]

    try:
        # Check if files exist and open the first file to determine nsweeps
        if len(files):
            file = files[0]
        else:
            return []

        try:
            ds = read_sweep(file)
        except OSError:
            ds = xr.open_dataset(file, engine="netcdf4")

        # Try to retrieve nsweeps from the dataset
        try:
            nsweeps = ds.attrs["nsweeps"]
        except KeyError:
            nsweeps = ds.sizes["sweep"]
        except KeyError:
            nsweeps = ds["sweep"].size
        except KeyError:
            nsweeps = ds["fixed_angle"].size

        # Sort the files naturally and group them by nsweeps
        files = sorted(files, key=_natural_sort_key)
        sweep_groups = []
        for i in range(0, len(files), nsweeps):
            sweep_groups.append(files[i : i + nsweeps])

        del ds

        return sweep_groups

    except Exception:
        # Fallback method if nsweeps cannot be determined from metadata
        sweep_groups = []
        sweep_times = []
        sweep_elevs = []
        files = sorted(files, key=_natural_sort_key)

        # Collect sweep times and elevations
        for file in files:
            ds = xr.open_dataset(file, engine="netcdf4")
            if "esStartTime" in ds:
                estime = (
                    ds["esStartTime"].dt.strftime("%Y-%m-%dT%H:%M:%S").values.item()
                )
            elif "time_coverage_start" in ds:
                estime = (
                    ds["time_coverage_start"]
                    .dt.strftime("%Y-%m-%dT%H:%M:%S")
                    .values.item()
                )

            if "elevationAngle" in ds:
                ele_ang = ds["elevationAngle"].values.item()
            elif "fixed_angle" in ds:
                ele_ang = ds["fixed_angle"].values.item()
            sweep_times.append(estime)
            sweep_elevs.append(ele_ang)
            del ds

        # Determine sweep groups based on elevation increases/decreases
        current_group = []
        nsweeps = 0

        for i in range(1, len(sweep_elevs)):
            current_group.append(files[i - 1])

            # Check if the elevation decreases, indicating the start of a new volume
            if sweep_elevs[i] < sweep_elevs[i - 1]:
                nsweeps = len(current_group)
                sweep_groups.append(current_group)
                current_group = []

        # Append the last group
        current_group.append(files[-1])
        sweep_groups.append(current_group)

        return sweep_groups


def _determine_volumes(files):
    """Determine Volumes"""
    swp_lists = _determine_nsweeps(files)
    volumes = []
    for i, swp_list in enumerate(swp_lists):
        dataset = []
        for file in swp_list:
            try:
                ds = read_sweep(file)
            except OSError:
                ds = xr.open_dataset(file)
            dataset.append(ds)
            del ds
        volumes.append(dataset)
        del dataset
    return volumes


def create_volume(dataset_list):
    """
    Concatenate a list of datasets along the 'time' and 'sweep' dimensions.
    Non-time-varying variables will be retained as single values.
    Also updates `time_coverage_end` based on the maximum time value.

    You have to give the list of sweeps which are already organized/sorted,
    for that you can either make it manually, or use `_determine_volumes` function

    Parameters
    ----------
    dataset_list : list of xarray.Dataset
        List of datasets representing individual sweeps.

    Returns
    -------
    xarray.Dataset
        Concatenated dataset with proper time and sweep dimensions.
    """

    # Identify variables that should be concatenated along the sweep dimension
    sweep_vars = [
        "fixed_angle",
        "ray_angle_res",
        "sweep_mode",
        "scan_type",
        "sweep_number",
        "sweep_start_ray_index",
        "sweep_end_ray_index",
    ]

    # Identify variables that are constant across sweeps and should not be concatenated
    constant_vars = [
        "time_coverage_start",
        "latitude",
        "longitude",
        "altitude",
        "firstGateRange",
        "gateSize",
        "nyquist_velocity",
        "unambiguous_range",
        "calibConst",
        "radarConst",
        "beamWidthHori",
        "pulseWidth",
        "bandWidth",
        "filterDop",
        "azimuthSpeed",
        "highPRF",
        "lowPRF",
        "dwellTime",
        "waveLength",
        "calI0",
        "calNoise",
        "groundHeight",
        "meltHeight",
        "scanType",
        "logNoise",
        "linNoise",
        "inphaseOffset",
        "quadratureOffset",
        "logSlope",
        "logFilter",
        "filterPntClt",
        "filterThreshold",
        "sampleNum",
        "SQIThresh",
        "LOGThresh",
        "SIGThresh",
        "CSRThresh",
        "DBTThreshFlag",
        "DBZThreshFlag",
        "VELThreshFlag",
        "WIDThreshFlag",
    ]

    # Step 1: Concatenate the datasets along the time dimension
    time_concat = xr.concat(
        [ds.drop_vars(sweep_vars, errors="ignore") for ds in dataset_list], dim="time"
    )

    # Step 2: Handle sweep-specific variables (metadata)
    sweep_datasets = []
    for i, ds in enumerate(dataset_list):
        sweep_ds = {}
        for var in sweep_vars:
            if var in ds:
                # Expand dims to add a 'sweep' dimension if needed
                if "sweep" not in ds[var].dims:
                    sweep_ds[var] = ds[var].expand_dims("sweep")
                else:
                    sweep_ds[var] = ds[var]

        # Update sweep start and end ray indices
        sweep_start = i * ds.sizes["time"]
        sweep_end = sweep_start + ds.sizes["time"] - 1

        sweep_ds["sweep_start_ray_index"] = xr.DataArray(
            np.array([sweep_start]),
            dims="sweep",
            attrs={"long_name": "Index of the first ray in the sweep"},
        )
        sweep_ds["sweep_end_ray_index"] = xr.DataArray(
            np.array([sweep_end]),
            dims="sweep",
            attrs={"long_name": "Index of the last ray in the sweep"},
        )

        # Add the updated sweep metadata dataset to the list
        sweep_datasets.append(xr.Dataset(sweep_ds))

    # Step 3: Concatenate the sweep-specific datasets along the sweep dimension
    sweep_concat = xr.concat(sweep_datasets, dim="sweep")

    # Step 4: Merge the time-concatenated data, sweep-concatenated metadata, and constant variables
    combined_ds = xr.merge([time_concat, sweep_concat])

    # Step 5: Assign constant variables (they do not need to be concatenated)
    for var in constant_vars:
        if var in dataset_list[0]:
            combined_ds[var] = dataset_list[0][var]

    # Step 6: Update the 'time_coverage_end' variable with the maximum time value
    combined_ds["time_coverage_end"] = xr.DataArray(
        data=str(
            combined_ds.time.max().dt.strftime("%Y-%m-%dT%H:%M:%SZ").values
        ).encode("utf-8"),
        attrs={"standard_name": "data_volume_end_time_utc"},
    )

    return combined_ds
