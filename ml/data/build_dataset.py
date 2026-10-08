#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.
"""
Build radar ML training data from NEXRAD Level II with radarx labels.

Examples
--------
List the volumes and splits of a config without processing anything::

    python ml/data/build_dataset.py ml/data/configs/demo.yaml --dry-run

Build the demo set (20 volumes) with 8 worker processes::

    python ml/data/build_dataset.py ml/data/configs/demo.yaml -o demo_out -w 8

See ``ml/README.md`` for the dataset schemas.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mldata.build import build  # noqa: E402
from mldata.cases import (  # noqa: E402
    assign_splits,
    check_no_leakage,
    expand_cases,
    load_config,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="case-selection YAML config")
    parser.add_argument("-o", "--out", default=None, help="output folder")
    parser.add_argument(
        "-c",
        "--cache",
        default=str(Path.home() / ".cache" / "radarx-ml" / "nexrad"),
        help="download cache of the NEXRAD files",
    )
    parser.add_argument("-w", "--workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="list volumes only")
    parser.add_argument("--keep-parts", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.dry_run:
        assign_splits(cfg)
        check_no_leakage(
            cfg["cases"],
            cfg["splits"]["min_gap_hours"],
            cfg["splits"]["holdout_radars"],
        )
        records = expand_cases(cfg)
        for c in cfg["cases"]:
            n = sum(r["case"] == c["name"] for r in records)
            print(f"{c['split']:5s} {c['name']:40s} {c['radar']} {n:4d} volumes")
        print(f"total: {len(records)} volumes")
        return 0
    out = args.out or f"{cfg['name']}_data"
    manifest = build(
        cfg, out, cache_dir=args.cache, workers=args.workers, keep_parts=args.keep_parts
    )
    print(json.dumps({"samples": manifest["samples"], **manifest["timing_seconds"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
