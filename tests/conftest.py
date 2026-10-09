#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Test configuration shared by the whole suite."""

import sys

# open-radar-data files that several test modules read
SHARED_DATA = (
    "KLBB20160601_150025_V06",
    "swx_20120520_0641.nc",
    "corcsapr2cmacppiM1.c1.20181111.030003.nc",
)


def pytest_configure(config):
    """Download the shared test files once, before the xdist workers start.

    On Windows two workers that fetch the same missing file at the same time
    fail (the temporary download cannot replace a file another worker has just
    created or still holds open). Fetching them here, in the controller, avoids
    the race; a failed download is left to the tests that need the file.
    """
    if sys.platform != "win32" or hasattr(config, "workerinput"):
        return
    try:
        from open_radar_data import DATASETS
    except ImportError:
        return
    for name in SHARED_DATA:
        try:
            DATASETS.fetch(name)
        except Exception:  # offline or server error: the tests report it
            pass
