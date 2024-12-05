import pytest
import os
from radarx.io.aws_data import AWS_BUCKETS, list_available_files, download_file


# Fixture for creating temporary directories for file downloads
@pytest.fixture
def temp_download_dir(tmp_path):
    """Fixture for creating a temporary download directory."""
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    return download_dir


def test_list_available_files_mrms():
    """Test the list_available_files function for MRMS bucket."""
    year = "2022"
    month = "03"
    day = "30"
    hour = "23"
    minute = "46"
    prod = "ReflectivityAtLowestAltitude_00.50"
    prefix = (
        f"CONUS/{prod}/{year}{month}{day}/MRMS_{prod}_{year}{month}{day}-{hour}{minute}"
    )

    files = list_available_files(AWS_BUCKETS["MRMS"], prefix)

    # Check that the function returns a list
    assert isinstance(files, list), "The returned value should be a list."

    # If there are files, ensure they contain the correct product and date
    if files:
        assert any(
            f"{year}{month}{day}" in f for f in files
        ), "Files do not match the specified date."
        assert any(
            prod in f for f in files
        ), "Files do not match the specified product."


def test_download_file_mrms(temp_download_dir):
    """Test the download_file function for MRMS bucket."""
    # Example file for testing; replace with a valid file if needed
    file_key = "CONUS/ReflectivityAtLowestAltitude_00.50/20220330/MRMS_ReflectivityAtLowestAltitude_00.50_20220330-234629.grib2.gz"

    # Use the MRMS bucket
    bucket = AWS_BUCKETS["MRMS"]

    # Download the file
    downloaded_file = download_file(bucket, file_key, str(temp_download_dir))

    # Check if the file was downloaded
    if downloaded_file:
        assert os.path.exists(downloaded_file), "The downloaded file does not exist."
        assert downloaded_file.endswith(
            ".grib2.gz"
        ), "Downloaded file extension is incorrect."
    else:
        pytest.skip(f"File not found or download failed for {file_key}")


def test_list_available_files_invalid_prefix():
    """Test the list_available_files function with an invalid prefix."""
    invalid_prefix = "nonexistent/path/"

    files = list_available_files(AWS_BUCKETS["MRMS"], invalid_prefix)

    # Ensure no files are returned for an invalid prefix
    assert files == [], "Files should not be returned for an invalid prefix."


def test_download_file_invalid_key(temp_download_dir):
    """Test the download_file function with an invalid file key."""
    invalid_file_key = "nonexistent/path/file.txt"

    # Use the MRMS bucket
    bucket = AWS_BUCKETS["MRMS"]

    # Try to download the file
    downloaded_file = download_file(bucket, invalid_file_key, str(temp_download_dir))

    # Ensure the function handles the invalid key gracefully
    assert (
        downloaded_file is None
    ), "Download should return None for an invalid file key."
