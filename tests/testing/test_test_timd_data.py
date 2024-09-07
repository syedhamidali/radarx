import pytest
import logging
from radarx.testing.test_data_imd import fetch_imd_test_data
from pathlib import Path


# Set up a fixture for capturing logs
@pytest.fixture
def capture_logs(caplog):
    caplog.set_level(logging.INFO)
    return caplog


def test_fetch_imd_test_data(capture_logs):
    """
    Test the fetch_imd_test_data function to ensure all files are downloaded and
    logged correctly.
    """
    # Call the function to fetch data
    downloaded_files = fetch_imd_test_data()

    # Assert that all expected files were downloaded
    expected_files = [
        "GOA210515003646-IMD-C.nc",
        "GOA210515003646-IMD-C.nc.1",
        "GOA210515004746-IMD-C.nc",
        "GOA210515004746-IMD-C.nc.1",
        "GOA210515005811-IMD-C.nc",
        "GOA210515005811-IMD-C.nc.1",
    ]
    for file_name in expected_files:
        assert (
            file_name in downloaded_files
        ), f"{file_name} should be in the downloaded files."

    # Verify that logging correctly recorded the download messages
    for record in capture_logs.records:
        assert (
            "Successfully downloaded" in record.message
        ), "Log should contain successful download messages."

    # Ensure no errors were logged
    assert not any(
        record.levelname == "ERROR" for record in capture_logs.records
    ), "No error logs should be present."


def test_fetch_imd_test_data_file_paths():
    """
    Test to ensure that the fetch_imd_test_data function returns valid file paths.
    """
    downloaded_files = fetch_imd_test_data()

    for file_name, file_path in downloaded_files.items():
        file_path_obj = Path(file_path)  # Convert to Path object
        assert (
            file_path_obj.exists()
        ), f"The file {file_name} should exist at {file_path_obj}"


if __name__ == "__main__":
    pytest.main()
