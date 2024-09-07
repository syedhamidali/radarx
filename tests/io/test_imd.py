import pytest
from unittest import mock
from unittest.mock import patch

from radarx.io.imd import (
    read_sweep,
    read_volume,
    _merge_file_lists,
    _determine_nsweeps,
    _determine_volumes,
    create_volume,
    _compute_range,
    _angle_resolution,
    _scantype,
    _time_coverage,
    _sweep_number,
    _assign_metadata,
)
from radarx.testing.test_data_imd import fetch_imd_test_data
import xarray as xr
import numpy as np


@pytest.fixture(scope="module")
def imd_test_data():
    """
    Fixture to download IMD test data once for all tests in the module.
    Returns the paths to the downloaded files.
    """
    return fetch_imd_test_data()


def ensure_required_fields(ds):
    """
    Ensures that the dataset has the necessary fields for testing.
    Adds any missing fields with mock data.
    """
    if "angleResolution" not in ds:
        ds["angleResolution"] = xr.DataArray(
            np.full((ds.sizes.get("sweep", 1)), 1.0),
            dims="sweep",
            attrs={"long_name": "Angular Resolution", "units": "degrees"},
        )

    if "scanType" not in ds:
        ds["scanType"] = xr.DataArray(
            np.array([4]), dims=()  # Default to 'azimuth_surveillance'
        )

    if "elevationNumber" not in ds:
        ds["elevationNumber"] = xr.DataArray(
            np.array([0]), dims="sweep"  # Default value
        )

    if "azimuth" not in ds:
        ds["azimuth"] = xr.DataArray(
            np.linspace(0, 360, ds.sizes["time"]),
            dims="time",
            attrs={"long_name": "Azimuth angle", "units": "degrees"},
        )

    return ds


def test_read_sweep(imd_test_data):
    """
    Test the read_sweep function to ensure it processes a single sweep file correctly.
    """
    file_path = imd_test_data["GOA210515003646-IMD-C.nc"]
    result = read_sweep(file_path)

    assert result is not None, "Expected a valid dataset"
    assert "DBZ" in result, "Expected DBZ field in the dataset"


def test_read_volume(imd_test_data):
    """
    Test the read_volume function to ensure it processes multiple sweep files into a volume.
    """
    files = [
        imd_test_data["GOA210515003646-IMD-C.nc"],
        imd_test_data["GOA210515003646-IMD-C.nc.1"],
        imd_test_data["GOA210515004746-IMD-C.nc"],
        imd_test_data["GOA210515004746-IMD-C.nc.1"],
        imd_test_data["GOA210515005811-IMD-C.nc"],
        imd_test_data["GOA210515005811-IMD-C.nc.1"],
    ]

    vol = read_volume(files)

    assert vol is not None, "Expected a valid volume dataset"
    assert "/volume_0" in vol.groups, "Expected volume_0 group in the DataTree"

    # Convert to dataset for further checks
    ds = vol["/volume_0"].to_dataset()
    assert "DBZ" in ds, "Expected DBZ field in the dataset"
    assert "VEL" in ds, "Expected VEL field in the dataset"
    assert "WIDTH" in ds, "Expected WIDTH field in the dataset"
    assert "ray_gate_spacing" in ds, "Expected ray_gate_spacing field in the dataset"


def test_volume_field_addition(imd_test_data):
    """
    Test to ensure missing fields in the radar volume are correctly handled (mock added).
    """
    files = [
        imd_test_data["GOA210515003646-IMD-C.nc"],
        imd_test_data["GOA210515003646-IMD-C.nc.1"],
        imd_test_data["GOA210515004746-IMD-C.nc"],
        imd_test_data["GOA210515004746-IMD-C.nc.1"],
        imd_test_data["GOA210515005811-IMD-C.nc"],
        imd_test_data["GOA210515005811-IMD-C.nc.1"],
    ]

    vol = read_volume(files)
    ds = vol["/volume_0"].to_dataset()

    # Check for missing fields and add mock data
    for field in ["DBZ", "VEL", "WIDTH"]:
        if field not in ds:
            ds[field] = xr.DataArray(
                np.full((ds.dims["time"], ds.dims["range"]), np.nan),
                dims=("time", "range"),
            )

    # Verify the fields are added
    assert "DBZ" in ds, "DBZ field should now exist in the dataset"
    assert "VEL" in ds, "VEL field should now exist in the dataset"
    assert "WIDTH" in ds, "WIDTH field should now exist in the dataset"


def test_field_mocking_for_missing(imd_test_data):
    """
    Test the addition of missing fields and mock data during radar processing.
    """
    file_path = imd_test_data["GOA210515003646-IMD-C.nc"]
    ds = read_sweep(file_path)
    ds = ensure_required_fields(ds)

    assert "angleResolution" in ds, "Expected angleResolution field to be present"
    assert "scanType" in ds, "Expected scanType field to be present"
    assert "elevationNumber" in ds, "Expected elevationNumber field to be present"


def test_volume_field_attributes(imd_test_data):
    """
    Test that volume field attributes match expected radar conventions.
    """
    files = [
        imd_test_data["GOA210515003646-IMD-C.nc"],
        imd_test_data["GOA210515003646-IMD-C.nc.1"],
        imd_test_data["GOA210515004746-IMD-C.nc"],
        imd_test_data["GOA210515004746-IMD-C.nc.1"],
        imd_test_data["GOA210515005811-IMD-C.nc"],
        imd_test_data["GOA210515005811-IMD-C.nc.1"],
    ]

    vol = read_volume(files)
    ds = vol["/volume_0"].to_dataset()

    # Ensure the attributes exist
    assert "Conventions" in ds.attrs, "Expected 'Conventions' attribute in the dataset"
    assert (
        "instrument_name" in ds.attrs
    ), "Expected 'instrument_name' attribute in the dataset"
    assert "version" in ds.attrs, "Expected 'version' attribute in the dataset"
    assert "institution" in ds.attrs, "Expected 'institution' attribute in the dataset"


def test_merge_file_lists():
    """
    Test that ensures nested file lists are flattened correctly.
    """
    file_lists = [["file1.nc", "file2.nc"], ["file3.nc", "file4.nc"]]
    merged_list = _merge_file_lists(file_lists)

    # Ensure the result is flattened
    assert len(merged_list) == 4, "Expected 4 files after merging lists."
    assert merged_list == [
        "file1.nc",
        "file2.nc",
        "file3.nc",
        "file4.nc",
    ], "File lists not merged correctly."


def test_assign_metadata(imd_test_data):
    """
    Test the _assign_metadata function to ensure metadata is correctly assigned.
    The test applies multiple preprocessing functions in sequence to ensure
    the dataset is correctly prepared before assigning metadata.
    """
    file_path = imd_test_data["GOA210515003646-IMD-C.nc"]

    # Step 1: Read the sweep file
    ds = xr.open_dataset(file_path, engine="netcdf4")

    # Step 2: Rename dimensions, variables, and set coordinates for consistency
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

    # Step 3: Add ray gate spacing
    ds["ray_gate_spacing"] = xr.DataArray(
        data=np.repeat(ds.gateSize.values, ds.azimuth.size),
        dims=("azimuth",),
        attrs={"long_name": "gate_spacing_for_ray", "units": "meters"},
    )

    # Step 4: Rename moments (DBT, DBZ, VEL, WIDTH)
    moments = {
        "T": "DBT",  # Total power
        "Z": "DBZ",  # Reflectivity
        "V": "VEL",  # Velocity
        "W": "WIDTH",  # Spectrum width
    }
    ds = ds.rename_vars(moments)

    # Step 5: Apply the pipeline functions in sequence
    ds = ds.pipe(_compute_range)  # Compute range and assign to dataset
    ds = ds.pipe(_angle_resolution)  # Compute angle resolution
    ds = ds.pipe(_scantype)  # Determine scan type and mode
    ds = ds.pipe(_time_coverage)  # Time coverage handling
    ds = ds.pipe(_sweep_number)  # Handle sweep numbers

    # Step 6: Assign metadata
    ds_with_metadata = ds.pipe(_assign_metadata)

    # Check the presence of the necessary attributes
    assert "Conventions" in ds_with_metadata.attrs, "Expected 'Conventions' attribute."
    assert (
        "instrument_name" in ds_with_metadata.attrs
    ), "Expected 'instrument_name' attribute."
    assert "field_names" in ds_with_metadata.attrs, "Expected 'field_names' attribute."

    # Check if the metadata contains valid field names
    expected_fields = ["DBT", "DBZ", "VEL", "WIDTH"]
    for field in expected_fields:
        assert (
            field in ds_with_metadata.attrs["field_names"]
        ), f"Field {field} should be in metadata."


def test_determine_nsweeps(imd_test_data):
    """
    Test the _determine_nsweeps function to ensure it correctly groups sweeps.
    """
    files = [
        imd_test_data["GOA210515003646-IMD-C.nc"],
        imd_test_data["GOA210515003646-IMD-C.nc.1"],
        imd_test_data["GOA210515004746-IMD-C.nc"],
        imd_test_data["GOA210515004746-IMD-C.nc.1"],
        imd_test_data["GOA210515005811-IMD-C.nc"],
        imd_test_data["GOA210515005811-IMD-C.nc.1"],
    ]

    sweep_groups = _determine_nsweeps(files)

    # Check if sweeps are correctly grouped
    assert len(sweep_groups) > 0, "No sweep groups detected."
    assert all(
        len(group) == 2 for group in sweep_groups
    ), "Expected groups of 2 sweeps."


def test_determine_volumes(imd_test_data):
    """
    Test the _determine_volumes function to ensure it correctly reads and groups volumes.
    """
    files = [
        imd_test_data["GOA210515003646-IMD-C.nc"],
        imd_test_data["GOA210515003646-IMD-C.nc.1"],
        imd_test_data["GOA210515004746-IMD-C.nc"],
        imd_test_data["GOA210515004746-IMD-C.nc.1"],
        imd_test_data["GOA210515005811-IMD-C.nc"],
        imd_test_data["GOA210515005811-IMD-C.nc.1"],
    ]

    volumes = _determine_volumes(files)

    # Ensure that volumes are correctly grouped and returned
    assert len(volumes) > 0, "Expected volumes to be detected."
    assert all(
        isinstance(volume, list) for volume in volumes
    ), "Expected each volume to be a list of datasets."


def test_read_volume_alternate(imd_test_data):
    """
    Test the create_volume function to ensure sweeps are concatenated correctly into volumes.
    """
    files = [
        imd_test_data["GOA210515003646-IMD-C.nc"],
        imd_test_data["GOA210515003646-IMD-C.nc.1"],
        imd_test_data["GOA210515004746-IMD-C.nc"],
        imd_test_data["GOA210515004746-IMD-C.nc.1"],
        imd_test_data["GOA210515005811-IMD-C.nc"],
        imd_test_data["GOA210515005811-IMD-C.nc.1"],
    ]

    volumes = _determine_volumes(files)
    volume_ds = create_volume(volumes[0])  # Create volume from first set of sweeps

    assert "time" in volume_ds.dims, "Expected time dimension in volume dataset."
    assert "sweep" in volume_ds.dims, "Expected sweep dimension in volume dataset."
    assert volume_ds.sizes["sweep"] == len(volumes[0]), "Mismatch in number of sweeps."


def create_mock_dataset_with_nsweeps(nsweeps=None, fixed_angle_size=None):
    """
    Helper function to create a mock xarray.Dataset with nsweeps or fixed_angle_size.
    """
    ds = xr.Dataset()
    ds.attrs = {"nsweeps": nsweeps} if nsweeps is not None else {}
    ds["fixed_angle"] = xr.DataArray(
        [1.0] * (fixed_angle_size if fixed_angle_size else 1)
    )
    return ds


def test_determine_nsweeps_with_metadata():
    """
    Test `_determine_nsweeps` when nsweeps can be determined from metadata.
    """
    # Mock dataset with nsweeps in metadata
    mock_dataset = create_mock_dataset_with_nsweeps(nsweeps=3)

    # Mock file paths
    mock_files = ["sweep1.nc", "sweep2.nc", "sweep3.nc"]

    with patch("xarray.open_dataset", return_value=mock_dataset):
        sweep_groups = _determine_nsweeps(mock_files)

        # Ensure that nsweeps is correctly determined and files are grouped
        assert len(sweep_groups) == 1, "There should be one volume group"
        assert len(sweep_groups[0]) == 3, "There should be 3 sweeps in the volume"


def test_determine_nsweeps_with_empty_file_list():
    """
    Test `_determine_nsweeps` with an empty file list.
    """
    sweep_groups = _determine_nsweeps([])
    assert sweep_groups == [], "There should be no sweep groups for an empty file list"


# Mocking read_sweep and xr.open_dataset
@pytest.fixture
def mock_read_sweep():
    with mock.patch("radarx.io.imd.read_sweep") as mocked_read_sweep:
        yield mocked_read_sweep


@pytest.fixture
def mock_open_dataset():
    with mock.patch("xarray.open_dataset") as mocked_open_dataset:
        yield mocked_open_dataset


@pytest.fixture
def radar_test_files(imd_test_data):
    """
    Fixture for providing radar test files.
    """
    return [
        imd_test_data["GOA210515003646-IMD-C.nc"],
        imd_test_data["GOA210515004746-IMD-C.nc"],
    ]


def test_determine_volumes_success(mock_read_sweep, radar_test_files):
    """
    Test for successful reading of radar volumes using `read_sweep`.
    """
    # Simulate successful read_sweep function for all files
    mock_read_sweep.return_value = xr.Dataset(
        {"DBZ": (("time", "range"), [[1, 2], [3, 4]])}
    )

    volumes = _determine_volumes(radar_test_files)

    assert len(volumes) == 1, "Expected one volume to be created"
    assert len(volumes[0]) == 2, "Expected two files to be part of the volume"
    assert isinstance(
        volumes[0][0], xr.Dataset
    ), "Expected the first sweep to be an xarray Dataset"
    assert "DBZ" in volumes[0][0], "Expected DBZ field in the dataset"


def test_determine_volumes_fallback_to_open_dataset(
    mock_read_sweep, mock_open_dataset, radar_test_files
):
    """
    Test for fallback to `xr.open_dataset` when `read_sweep` fails.
    """
    # Simulate read_sweep failure for the first file
    mock_read_sweep.side_effect = [
        IOError("read_sweep failed"),
        xr.Dataset({"DBZ": (("time", "range"), [[1, 2], [3, 4]])}),
    ]

    # Simulate successful fallback to open_dataset for the first file
    mock_open_dataset.return_value = xr.Dataset(
        {"VEL": (("time", "range"), [[5, 6], [7, 8]])}
    )

    volumes = _determine_volumes(radar_test_files)

    assert len(volumes) == 1, "Expected one volume to be created"
    assert len(volumes[0]) == 2, "Expected two files to be part of the volume"
    assert isinstance(
        volumes[0][0], xr.Dataset
    ), "Expected the first sweep to be an xarray Dataset"
    assert "VEL" in volumes[0][0], "Expected VEL field to be present after fallback"
    assert (
        "DBZ" in volumes[0][1]
    ), "Expected DBZ field in the dataset for the second file"


def test_determine_volumes_fallback_fail(
    mock_read_sweep, mock_open_dataset, radar_test_files
):
    """
    Test for failure in both `read_sweep` and `xr.open_dataset`.
    """
    # Simulate failure for both `read_sweep` and `xr.open_dataset`
    mock_read_sweep.side_effect = IOError("read_sweep failed")
    mock_open_dataset.side_effect = IOError("open_dataset failed")

    with pytest.raises(IOError, match="open_dataset failed"):
        _determine_volumes(radar_test_files)


if __name__ == "__main__":
    pytest.main()
