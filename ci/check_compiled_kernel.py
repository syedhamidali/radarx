#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Fail if radarx was installed without any of its compiled kernels.

The extensions are optional at build time (radarx falls back to NumPy), so a
broken compiler setup would otherwise go unnoticed in wheels and CI. Every
``radarx/**/_*.cpp`` source is a kernel; this imports each one from the
installed package. Their results are checked against the NumPy engines in the
test suite.
"""

import importlib
import sys
from pathlib import Path

import radarx

# kernel sources from the checkout (CI) or, failing that, the installed package
root = Path(__file__).resolve().parents[1] / "radarx"
if not root.is_dir():
    root = Path(radarx.__file__).parent
sources = sorted(root.rglob("_*.cpp"))
if not sources:
    sys.exit("no kernel sources found")

missing = []
for source in sources:
    name = ".".join(("radarx",) + source.relative_to(root).with_suffix("").parts)
    try:
        importlib.import_module(name)
    except ImportError as err:
        missing.append(f"{name}: {err}")
    else:
        print(f"compiled kernel {name} OK")

if missing:
    sys.exit("compiled kernels missing:\n  " + "\n  ".join(missing))
