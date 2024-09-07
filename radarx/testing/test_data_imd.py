import logging
import pooch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_imd_test_data():
    """
    Fetches IMD radar data files using pooch.
    Downloads files from the specified remote URL if they are not cached locally.
    Also ensures that the files match their expected checksums.

    Returns:
        dict: A dictionary containing the local paths of the downloaded files.
    """
    # Define the base URL for the IMD data
    base_url = "https://github.com/syedhamidali/pyscancf_examples/raw/main/data/goa_c/"

    # Define the files and their corresponding MD5 checksums
    registry = {
        "GOA210515003646-IMD-C.nc": "md5:5d4219ead7a340efec670d1c2a2c83ee",
        "GOA210515003646-IMD-C.nc.1": "md5:b132df3d23b632981773bd9f423597ee",
        "GOA210515004746-IMD-C.nc": "md5:ecdbfa3d69f826f91243740e3e6d1b1d",
        "GOA210515004746-IMD-C.nc.1": "md5:32c29c64aca3fe5e07756ac3894ace7a",
        "GOA210515005811-IMD-C.nc": "md5:45f692bbb9be9e5b3464bab7b52731f4",
        "GOA210515005811-IMD-C.nc.1": "md5:32504fa5035f61e49a384b6f9f515d7c",
    }

    # Create a pooch object with cache path and registry
    pooch_obj = pooch.create(
        path=pooch.os_cache("radarx_data"),  # Local cache directory
        base_url=base_url,  # Base URL for the remote files
        registry=registry,  # Registry of files and their MD5 checksums
    )

    # Fetch files and store paths in a dictionary
    downloaded_files = {}
    for file_name in registry:
        try:
            file_path = pooch_obj.fetch(
                file_name
            )  # Fetch the file (download if not cached)
            logger.info(f"Successfully downloaded: {file_name} to {file_path}")
            downloaded_files[file_name] = file_path
        except Exception as e:
            logger.error(f"Failed to download {file_name}: {e}")
            raise

    return downloaded_files


if __name__ == "__main__":
    # Fetch the IMD data and print the paths of downloaded files
    files = fetch_imd_test_data()
    for file_name, file_path in files.items():
        logger.info(f"Local path for {file_name}: {file_path}")
