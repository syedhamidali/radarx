#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Build the optional compiled kernels.

Project metadata lives in pyproject.toml. Every ``radarx/**/_*.cpp`` file is a
pybind11 kernel and is built as the extension module of the same name (e.g.
``radarx/retrieve/_kdp.cpp`` -> ``radarx.retrieve._kdp``), so new kernels need
no change here. The extensions are optional: if they cannot be compiled,
radarx still installs and uses a NumPy implementation.
"""

import os
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


class ParallelBuildExt(build_ext):
    """Build the extensions in parallel (one kernel per job)."""

    def finalize_options(self):
        super().finalize_options()
        # ``parallel`` is the -j option of setuptools' build_ext
        if getattr(self, "parallel", None) is None:
            jobs = os.environ.get("RADARX_BUILD_JOBS")
            self.parallel = int(jobs) if jobs else os.cpu_count() or 1


extra_compile_args = []
if sys.platform != "win32":
    extra_compile_args = ["-O3"]

kernels = sorted(Path("radarx").rglob("_*.cpp"))

setup(
    ext_modules=[
        Pybind11Extension(
            ".".join(source.with_suffix("").parts),
            [source.as_posix()],
            cxx_std=17,
            extra_compile_args=extra_compile_args,
            optional=True,
        )
        for source in kernels
    ],
    cmdclass={"build_ext": ParallelBuildExt},
)
