#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Download the test and notebook data once, before pytest starts.

The default URLs (github.com/<repo>/raw/...) answer 504 (gateway time-out) when
many pytest-xdist workers ask at once; raw.githubusercontent.com serves the
same files quickly. The files are stored in the usual cache folders and
checked against the registered hashes, so the tests find them there. A failed
download is reported but does not stop the job: the tests that need the file
report it.
"""

import sys

import pooch
from open_radar_data import DATASETS

RAW = "https://raw.githubusercontent.com/"
PREFIXES = (
    "KLBB20160601_150025_V06",
    "swx_20120520_0641.nc",
    "corcsapr2cmacppiM1.c1.20181111.030003.nc",
    "RAW_NA_000_125_20080411",
    "IMD/JPR220822135253-IMD-B.nc",
)
IMD_BASE = RAW + "syedhamidali/pyscancf_examples/main/data/goa_c/"


def main():
    failed = []
    DATASETS.base_url = RAW + "openradar/open-radar-data/main/data/"
    DATASETS.retry_if_failed = 5
    for name in sorted(n for n in DATASETS.registry if n.startswith(PREFIXES)):
        try:
            DATASETS.fetch(name)
        except Exception as err:
            failed.append(f"{name}: {err}")
    # the deprecated IMD reader's test files
    from radarx.testing import test_data_imd

    original = pooch.create

    def create(*args, **kwargs):
        kwargs["base_url"] = IMD_BASE
        return original(*args, **kwargs)

    try:
        test_data_imd.pooch.create = create
        test_data_imd.fetch_imd_test_data()
    except Exception as err:
        failed.append(f"IMD test data: {err}")
    finally:
        test_data_imd.pooch.create = original
    n = len(failed)
    print(f"test data fetched, {n} failure{'s' * (n != 1)}")
    for line in failed:
        print("  ", line)
    sys.exit(0)


if __name__ == "__main__":
    main()
