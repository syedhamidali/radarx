#!/usr/bin/env python
# Copyright (c) 2024-2025, Radarx Developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
AWS Radar Data Utilities
=========================

This module provides utilities to interact with AWS S3 buckets
containing radar data. It supports listing available files in
specific paths and downloading them locally.

Supported Buckets
-----------------

- **NEXRAD Level II Archive**: ``noaa-nexrad-level2``
- **NEXRAD Level II Real-Time**: ``unidata-nexrad-level2-chunks``
- **NEXRAD Level III Real-Time**: ``unidata-nexrad-level3``
- **MRMS Data**: ``noaa-mrms-pds``

Usage Examples
--------------

1. List available files::

       from radarx.io.aws_data import list_available_files
       files = list_available_files('noaa-nexrad-level2', '2016/10/06/KAMX/')
       print(files)

2. Download a file::

       from radarx.io.aws_data import download_file
       file_path = download_file('noaa-nexrad-level2',
           '2016/10/06/KAMX/KAMX20161006_170414_V06', './downloads')
       print('File downloaded to', file_path)

This sub-module contains functions necessary to grid the radar data.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = [
    "get_s3_client",
    "list_available_files",
    "download_file",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import boto3
import botocore
import os

# AWS Buckets for different radar data
AWS_BUCKETS = {
    "NEXRAD_ARCHIVE": "noaa-nexrad-level2",
    "NEXRAD_REALTIME": "unidata-nexrad-level2-chunks",
    "NEXRAD_LEVEL3": "unidata-nexrad-level3",
    "MRMS": "noaa-mrms-pds",
}


def get_s3_client(anonymous=True):
    """
    Create an S3 client.

    Parameters
    ----------
    anonymous : bool, optional
        If True, creates an anonymous S3 client. Default is True.

    Returns
    -------
    boto3.client
        Configured S3 client.

    Examples
    --------
    Create an anonymous client:

        >>> s3 = get_s3_client()

    Create an authenticated client:

        >>> s3 = get_s3_client(anonymous=False)
    """
    if anonymous:
        session = boto3.session.Session()
        return session.client(
            service_name="s3",
            config=botocore.client.Config(signature_version=botocore.UNSIGNED),
        )
    else:
        return boto3.client("s3")  # pragma: no cover


def list_available_files(bucket, prefix, anonymous=True):
    """
    List files in an S3 bucket with the given prefix.

    Parameters
    ----------
    bucket : str
        Name of the AWS S3 bucket.
    prefix : str
        Prefix path in the bucket to search for files.
    anonymous : bool, optional
        If True, uses anonymous access. Default is True.

    Returns
    -------
    list
        List of file paths available under the prefix.

    Examples
    --------
    List files in the NEXRAD Level II archive bucket:

        >>> files = list_available_files("noaa-nexrad-level2",
            "2016/10/06/KAMX/")
        >>> print(files)

    List files in the MRMS bucket:

        >>> files = list_available_files("noaa-mrms-pds",
            "CONUS/ReflectivityAtLowestAltitude_00.50/")
        >>> print(files)
    """
    s3 = get_s3_client(anonymous=anonymous)
    try:  # pragma: no cover
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        files = [item["Key"] for item in response.get("Contents", [])]
        return files
    except botocore.exceptions.ClientError as e:  # pragma: no cover
        print(f"Failed to list files: {e}")  # pragma: no cover
        return []  # pragma: no cover


def download_file(bucket, file_key, save_dir, anonymous=True):
    """
    Download a file from S3.

    Parameters
    ----------
    bucket : str
        Name of the AWS S3 bucket.
    file_key : str
        Key of the file to download.
    save_dir : str
        Directory to save the downloaded file.
    anonymous : bool, optional
        If True, uses anonymous access. Default is True.

    Returns
    -------
    str
        Path to the downloaded file.

    Examples
    --------
    Download a NEXRAD Level II file:

        >>> file_path = download_file(
        ...     "noaa-nexrad-level2",
        ...     "2016/10/06/KAMX/KAMX20161006_170414_V06",
        ...     "./downloads"
        ... )
        >>> print(f"File downloaded to: {file_path}")

    Download an MRMS file:

        >>> file_path = download_file(
        ...     "noaa-mrms-pds",
        ...     "CONUS/ReflectivityAtLowestAltitude_00.50/2016/10/06/" +
                "ReflectivityAtLowestAltitude_00.50_20161006-1700.grib2.gz",
        ...     "./downloads"
        ... )
        >>> print(f"File downloaded to: {file_path}")
    """
    s3 = get_s3_client(anonymous=anonymous)
    os.makedirs(save_dir, exist_ok=True)
    local_file = os.path.join(save_dir, os.path.basename(file_key))
    try:
        s3.download_file(bucket, file_key, local_file)
        print(f"Downloaded: {local_file}")  # pragma: no cover
        return local_file  # pragma: no cover
    except botocore.exceptions.ClientError as e:
        print(f"Failed to download {file_key}: {e}")
        return None
