#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Test configuration shared by the whole suite."""

# open-radar-data files that several test modules read, and the one the QVP
# time-series test starts from
SHARED_DATA = (
    "KLBB20160601_150025_V06",
    "swx_20120520_0641.nc",
    "corcsapr2cmacppiM1.c1.20181111.030003.nc",
    "RAW_NA_000_125_20080411181219",
)
RETRIES = 5


def pytest_configure(config):
    """Make the test-data downloads robust.

    The files come from GitHub, whose raw endpoint answers 504 (gateway
    time-out) when many workers ask at once, and on Windows two workers that
    fetch the same missing file at the same time fail (the temporary download
    cannot replace a file another worker has just created or still holds open).
    Every process therefore retries failed downloads, and the pytest-xdist
    controller fetches the shared files once, before the workers start. A
    download that still fails is left to the tests that need the file.
    """
    try:
        from open_radar_data import DATASETS
    except ImportError:
        return
    DATASETS.retry_if_failed = RETRIES
    if hasattr(config, "workerinput"):  # an xdist worker
        return
    for name in SHARED_DATA:
        try:
            DATASETS.fetch(name)
        except Exception:  # offline or server error: the tests report it
            pass
    try:
        from radarx.testing.test_data_imd import fetch_imd_test_data

        fetch_imd_test_data()
    except Exception:  # offline or server error: the tests report it
        pass
