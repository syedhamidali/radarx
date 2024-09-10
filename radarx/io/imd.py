#!/usr/bin/env python
# Copyright (c) 2024, radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
IMD Reader
==========

This sub-module provides functionality to read and process single radar files
from the Indian Meteorological Department (IMD), returning a quasi-CF-Radial
xarray Dataset.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

import logging
import itertools
import numpy as np
import xarray as xr
from datatree import DataTree

logger = logging.getLogger(__name__)

__all__ = [
    "read_sweep",
    "read_volume",
]


def read_sweep(file):
    """
    Read and process a single radar file from the Indian Meteorological Department (IMD).

    This function reads radar data and returns it as a quasi-CF-Radial `xarray.Dataset`,
    with coordinates and variables properly renamed and calculated fields added as necessary.

    Parameters
    ----------
    file : str or pathlib.Path
        The file path to the radar data file.

    Returns
    -------
    xarray.Dataset
        A processed dataset in quasi-CF-Radial format, with variables and coordinates
        appropriately renamed and additional fields calculated.

    Examples
    --------
    >>> import radarx as rx
    >>> ds = rx.io.read_sweep("path/to/radar/file")

    See Also
    --------
    read_volume : Reads and processes radar volume data from multiple sweeps.
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
    ds = ds.rename_vars({k: v for k, v in site_coords.items() if k in ds})

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
    # # Rename only the variables that exist in the dataset
    ds = ds.rename_vars({k: v for k, v in moments.items() if k in ds})

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

    # Assign start_end_ray_index, Goes 7th
    ds = ds.pipe(_assign_sweep_metadata)

    ds = ds.pipe(_assign_metadata)

    ds = ds.swap_dims({"azimuth": "time"})

    return ds


def read_volume(files):
    """
    Read and process multiple radar files to create a volume scan dataset.

    This function reads a list of radar files (or a list of lists, in the case of multi-sweep data)
    and returns a `DataTree` object containing the processed volume scan data.

    Parameters
    ----------
    files : list of str or pathlib.Path or list of list of str or pathlib.Path
        A list of radar file paths or a list of lists, where each sublist represents
        a separate sweep in the radar volume scan.

    Returns
    -------
    DataTree
        A `DataTree` object containing the volume scan data, organized by sweep.

    Examples
    --------
    >>> import radarx as rx
    >>> files = ["sweep1.nc", "sweep2.nc", "sweep3.nc"]
    >>> volume_data = rx.io.read_volume(files)

    See Also
    --------
    read_sweep : Read and process a single radar sweep file.
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

    Parameters
    ----------
    files : list of str
        List of file paths corresponding to radar sweep files.

    Returns
    -------
    sweep_groups : list of list of str
        List of sweep groups where each group contains radar sweep files
        corresponding to a complete volume.

    Notes
    -----
    The function first attempts to retrieve the number of sweeps (nsweeps)
    from the metadata of the files. If it cannot be determined, it uses
    the elevation angles to detect when the sweep elevations decrease,
    which indicates the start of a new volume.
    """
    import re

    def _natural_sort_key(s, _re=re.compile(r"(\d+)")):
        return [int(t) if i & 1 else t.lower() for i, t in enumerate(_re.split(s))]

    try:
        # Check if files exist and open the first file to determine nsweeps
        if len(files):
            file = files[0]
        else:
            logger.warning("No files provided.")
            return []

        ds = xr.open_dataset(file, engine="netcdf4")

        # Try to retrieve nsweeps from the dataset
        try:
            nsweeps = (
                ds.attrs.get("nsweeps")
                or ds.sizes.get("sweep")
                or ds["sweep"].size
                or ds["fixed_angle"].size
            )
        except KeyError:
            logger.error("Could not determine nsweeps from dataset metadata.")
            raise ValueError("Could not determine nsweeps from dataset metadata.")

        # Sort the files naturally and group them by nsweeps
        files = sorted(files, key=_natural_sort_key)
        sweep_groups = [files[i : i + nsweeps] for i in range(0, len(files), nsweeps)]

        del ds
        logger.info(
            f"Successfully grouped files into {len(sweep_groups)} sweep groups."
        )
        return sweep_groups

    except Exception as e:
        # Fallback method if nsweeps cannot be determined from metadata
        logger.warning(f"Failed to determine nsweeps from metadata: {e}")
        sweep_groups = []
        sweep_times = []
        sweep_elevs = []
        files = sorted(files, key=_natural_sort_key)

        # Collect sweep times and elevations
        for file in files:
            ds = xr.open_dataset(file, engine="netcdf4")
            estime = ds.get("esStartTime", ds.get("time_coverage_start", None))
            if estime is not None:
                estime = estime.dt.strftime("%Y-%m-%dT%H:%M:%S").values.item()

            ele_ang = ds.get("elevationAngle", ds.get("fixed_angle", None))
            if ele_ang is not None:
                ele_ang = ele_ang.values.item()

            sweep_times.append(estime)
            sweep_elevs.append(ele_ang if ele_ang is not None else float("nan"))
            del ds

        # Determine sweep groups based on elevation increases/decreases
        current_group = []
        for i in range(1, len(sweep_elevs)):
            current_group.append(files[i - 1])

            # Check if the elevation decreases, indicating the start of a new volume
            if (
                not (np.isnan(sweep_elevs[i]) or np.isnan(sweep_elevs[i - 1]))
                and sweep_elevs[i] < sweep_elevs[i - 1]
            ):
                logger.info(f"Detected volume change at file index {i}.")
                sweep_groups.append(current_group)
                current_group = []

        # Append the last group
        current_group.append(files[-1])
        sweep_groups.append(current_group)
        logger.info(f"Total number of sweep groups determined: {len(sweep_groups)}")

        return sweep_groups


def _determine_volumes(files):
    """
    Determine radar volumes from files.

    Parameters
    ----------
    files : list of str
        List of file paths corresponding to radar sweep files.

    Returns
    -------
    volumes : list of list of xarray.Dataset
        A list of radar volumes, where each volume is a list of xarray.Dataset objects.
        Each dataset corresponds to a radar sweep.

    Notes
    -----
    This function attempts to read radar sweep files using the `read_sweep` function.
    If `read_sweep` raises an exception, the function falls back to `xarray.open_dataset`
    for opening the file.
    """
    # Determine sweep lists based on the number of sweeps
    swp_lists = _determine_nsweeps(files)
    volumes = []

    for swp_list in swp_lists:
        volume = []
        for file in swp_list:
            try:
                # Attempt to read sweep using custom `read_sweep`
                ds = read_sweep(file)
            except (OSError, ValueError, KeyError) as e:
                # Log the specific exception with the filename and fall back to `open_dataset`
                logger.warning(
                    f"Failed to read sweep from {file} with read_sweep. Error: {e}. Falling back to open_dataset."
                )
                try:
                    ds = xr.open_dataset(file)
                except Exception as open_error:
                    logger.error(
                        f"Failed to open {file} with open_dataset. Error: {open_error}"
                    )
                    raise open_error  # Raise the exception if fallback also fails

            volume.append(ds)
        volumes.append(volume)

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
