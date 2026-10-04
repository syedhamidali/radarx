---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3
  name: python3
  language: python
---

# AWS Radar Data

+++

Demonstrates listing and downloading files from AWS radar data buckets.

```{code-cell} ipython3
import os
from radarx.io.aws_data import list_available_files, download_file
import pandas as pd
from itables import init_notebook_mode, show
```

## Set up download directory

```{code-cell} ipython3
DOWNLOAD_DIR = "./downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
```

```{code-cell} ipython3
def display_files(bucket_name, prefix):
    """Utility function to list available files in a bucket and print them."""
    print(f"Searching for files in bucket: {bucket_name}, prefix: {prefix}")
    files = list_available_files(bucket_name, prefix)
    if files:
        print(f"Available files: {files[:5]}")  # Display first 5 files for brevity
    else:
        print("No files found.")
    return files
```

## NEXRAD Level II

```{code-cell} ipython3
bucket_name = "unidata-nexrad-level2"
prefix = "2016/10/06/KAMX/"
files = display_files(bucket_name, prefix)

if files:
    file_to_download = files[0]
    print(f"Downloading: {file_to_download}")
    local_path = download_file(bucket_name, file_to_download, DOWNLOAD_DIR)
    print(f"Downloaded: {local_path}")
```

## NEXRAD Real-Time

```{code-cell} ipython3
bucket_name = "unidata-nexrad-level2-chunks"
prefix = "KAMX/457/"
files = display_files(bucket_name, prefix)

if files:
    file_to_download = files[0]
    print(f"Downloading: {file_to_download}")
    local_path = download_file(bucket_name, file_to_download, DOWNLOAD_DIR)
    print(f"Downloaded: {local_path}")
```

## NEXRAD Level III

```{code-cell} ipython3
bucket_name = "unidata-nexrad-level3"
prefix = "AMX_N0Q/AMX_N0Q_2016_10_06_17_00"
files = display_files(bucket_name, prefix)

if files:
    file_to_download = files[0]
    print(f"Downloading: {file_to_download}")
    local_path = download_file(bucket_name, file_to_download, DOWNLOAD_DIR)
    print(f"Downloaded: {local_path}")
```

## MRMS Data

```{code-cell} ipython3
bucket_name = "noaa-mrms-pds"
prefix = "CONUS/ReflectivityAtLowestAltitude_00.50/20220330/"
files = display_files(bucket_name, prefix)

if files:
    file_to_download = files[0]
    print(f"Downloading: {file_to_download}")
    local_path = download_file(bucket_name, file_to_download, DOWNLOAD_DIR)
    print(f"Downloaded: {local_path}")
```

```{code-cell} ipython3
init_notebook_mode(all_interactive=True)
```

## Config

```{code-cell} ipython3
# Define the configuration table as a dictionary
data = {
    "Function Name": [
        "list_available_files",
        "download_file",
        "list_and_filter_files",
        "get_nexrad_file",
        "get_mrms_file",
        "download_latest_file",
        "get_file_metadata",
        "get_available_prefixes",
        "download_files_in_batch",
        "list_all_files",
    ],
    "Purpose": [
        "Lists available files in a specified S3 bucket based on the given prefix.",
        "Downloads a file from the specified S3 bucket to a local directory.",
        "Combines `list_available_files` and filters results based on additional date criteria.",
        "High-level function to download NEXRAD Level II or Level III files based on location and time.",
        "High-level function to download MRMS files for a specific product and time.",
        "Downloads the most recent file for a specific product.",
        "Retrieves metadata for a specific file in the S3 bucket.",
        "Lists all prefixes (subfolders) available in the S3 bucket.",
        "Downloads multiple files from a list of file keys.",
        "Recursively lists all files in the bucket for a given prefix.",
    ],
    "Supported Products": [
        "unidata-nexrad-level2, unidata-nexrad-level2-chunks, unidata-nexrad-level3, noaa-mrms-pds",
        "Same as above",
        "Same as above",
        "unidata-nexrad-level2, unidata-nexrad-level3",
        "noaa-mrms-pds",
        "unidata-nexrad-level2-chunks, noaa-mrms-pds",
        "All buckets",
        "All buckets",
        "All buckets",
        "All buckets",
    ],
    "Key Parameters": [
        "`bucket` (str): S3 bucket name  <br> `prefix` (str): Prefix to filter files",
        "`bucket` (str): S3 bucket name <br> `file_key` (str): File path <br> `save_dir` (str): Local directory for saving",
        "`bucket` (str): S3 bucket name <br> `prefix` (str): Prefix to filter files <br> `date` (str): Date filter in `YYYYMMDD` format",
        "`location` (str): Radar site code <br> `date` (str): Date in `YYYYMMDD` format <br> `time` (str): Time in `HHMM` format",
        "`product` (str): MRMS product name <br> `date` (str): Date in `YYYYMMDD` format <br> `time` (str): Time in `HHMM` format",
        "`bucket` (str): S3 bucket name <br> `prefix` (str): Prefix to filter files",
        "`bucket` (str): S3 bucket name <br> `file_key` (str): File path",
        "`bucket` (str): S3 bucket name <br> `prefix` (str, optional): Filter for specific subdirectories",
        "`bucket` (str): S3 bucket name <br> `file_keys` (list): List of file paths <br> `save_dir` (str): Local directory for saving",
        "`bucket` (str): S3 bucket name <br> `prefix` (str, optional): Filter for specific paths",
    ],
    "Output": [
        "List of file keys",
        "Downloads file to the specified directory",
        "Filtered list of file keys",
        "Downloads file to `./downloads/nexrad_level2` or `nexrad_level3` directories",
        "Downloads file to `./downloads/mrms` directory",
        "Downloads file to `./downloads/latest` directory",
        "Metadata (e.g., size, last modified date)",
        "List of available prefixes",
        "Downloads files to specified directory",
        "Full list of file keys",
    ],
    "Notes": [
        "Use this to search for files based on time/location prefixes.",
        "Ensure the `save_dir` exists before calling this function.",
        "Use this to narrow down file searches for specific dates.",
        "Automates searching and downloading of specific radar files.",
        "Use this for MRMS-specific file downloads, e.g., Reflectivity, QPE.",
        "Useful for real-time monitoring applications.",
        "Use this for debugging or to retrieve additional information about a file before downloading.",
        "Use this to explore bucket contents or verify structure for new products.",
        "Ensure a valid list of file keys is passed.",
        "Useful for large-scale data exploration; might be slow for buckets with massive datasets.",
    ],
}

# Convert to a pandas DataFrame
df = pd.DataFrame(data)
```

```{code-cell} ipython3
show(df)  # scroll right to see the full list
```
