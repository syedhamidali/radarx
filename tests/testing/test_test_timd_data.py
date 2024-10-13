import pytest
import logging
from radarx.testing.test_data_imd import fetch_imd_test_data, display_fetched_files
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


def test_fetch_imd_test_data_error_handling(mocker, capture_logs):
    """
    Test to ensure that the fetch_imd_test_data function handles errors properly.
    """
    # Mock pooch's fetch method to raise an exception
    mocker.patch("pooch.Pooch.fetch", side_effect=Exception("Download failed"))

    # Call the function and assert that it raises an exception
    with pytest.raises(Exception, match="Download failed"):
        fetch_imd_test_data()

    # Check that an error was logged
    assert any(
        "Failed to download" in record.message for record in capture_logs.records
    ), "An error log should be present when a download fails."


def test_display_fetched_files(mocker, capture_logs):
    """
    Test the display_fetched_files function to ensure that it logs the downloaded file paths.
    """
    # Mock fetch_imd_test_data to return sample data
    mock_files = {
        "GOA210515003646-IMD-C.nc": "/path/to/GOA210515003646-IMD-C.nc",
        "GOA210515003646-IMD-C.nc.1": "/path/to/GOA210515003646-IMD-C.nc.1",
    }
    mocker.patch(
        "radarx.testing.test_data_imd.fetch_imd_test_data", return_value=mock_files
    )

    # Call the display_fetched_files function
    display_fetched_files()

    # Verify that the file paths were logged
    for file_name, file_path in mock_files.items():
        assert any(
            f"Local path for {file_name}: {file_path}" in record.message
            for record in capture_logs.records
        ), f"Log should contain the path for {file_name}"


if __name__ == "__main__":
    pytest.main()
