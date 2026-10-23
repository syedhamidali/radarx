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

from open_radar_data import DATASETS

RAW = "https://raw.githubusercontent.com/"
PREFIXES = (
    "KLBB20160601_150025_V06",
    "swx_20120520_0641.nc",
    "corcsapr2cmacppiM1.c1.20181111.030003.nc",
    "RAW_NA_000_125_20080411",
    "71_20181220_060628.pvol.h5",
    "IMD/JPR220822135253-IMD-B.nc",
)


def main():
    failed = []
    DATASETS.base_url = RAW + "openradar/open-radar-data/main/data/"
    DATASETS.retry_if_failed = 5
    for name in sorted(n for n in DATASETS.registry if n.startswith(PREFIXES)):
        try:
            DATASETS.fetch(name)
        except Exception as err:
            failed.append(f"{name}: {err}")
    # the upstream weights of the ML models, converted once so that parallel tests
    # and notebooks do not convert them at the same time
    try:
        from radarx.retrieve import _onnx_models

        for model in ("tornet-baseline-v1", "mistnet-nexrad"):
            _onnx_models.onnx_path(model)
    except Exception as err:
        failed.append(f"ONNX models: {err}")
    n = len(failed)
    print(f"test data fetched, {n} failure{'s' * (n != 1)}")
    for line in failed:
        print("  ", line)
    sys.exit(0)


if __name__ == "__main__":
    main()
