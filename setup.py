#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Build the optional compiled kernels.

Project metadata lives in pyproject.toml. The extensions are optional: if they
cannot be compiled, radarx still installs and uses a NumPy implementation.
"""

import os
import sys

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

setup(
    ext_modules=[
        Pybind11Extension(
            "radarx.grid._cone",
            ["radarx/grid/_cone.cpp"],
            cxx_std=17,
            extra_compile_args=extra_compile_args,
            optional=True,
        ),
        Pybind11Extension(
            "radarx.retrieve._shear",
            ["radarx/retrieve/_shear.cpp"],
            cxx_std=17,
            extra_compile_args=extra_compile_args,
            optional=True,
        ),
        Pybind11Extension(
            "radarx.retrieve._dealias",
            ["radarx/retrieve/_dealias.cpp"],
            cxx_std=17,
            extra_compile_args=extra_compile_args,
            optional=True,
        ),
        Pybind11Extension(
            "radarx.retrieve._advection",
            ["radarx/retrieve/_advection.cpp"],
            cxx_std=17,
            extra_compile_args=extra_compile_args,
            optional=True,
        ),
        Pybind11Extension(
            "radarx.retrieve._kdp",
            ["radarx/retrieve/_kdp.cpp"],
            cxx_std=17,
            extra_compile_args=extra_compile_args,
            optional=True,
        ),
        Pybind11Extension(
            "radarx.retrieve._qvp",
            ["radarx/retrieve/_qvp.cpp"],
            cxx_std=17,
            extra_compile_args=extra_compile_args,
            optional=True,
        ),
        Pybind11Extension(
            "radarx.io._sounding",
            ["radarx/io/_sounding.cpp"],
            cxx_std=17,
            extra_compile_args=extra_compile_args,
            optional=True,
        ),
    ],
    cmdclass={"build_ext": ParallelBuildExt},
)
