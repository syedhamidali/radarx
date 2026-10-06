#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Build the optional compiled kernels (cone gridding, advection).

Project metadata lives in pyproject.toml. The extensions are optional: if they
cannot be compiled, radarx still installs and uses a NumPy implementation.
"""

import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

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
        Pybind11Extension(
            "radarx.retrieve._dsd",
            ["radarx/retrieve/_dsd.cpp"],
            cxx_std=17,
            extra_compile_args=extra_compile_args,
            optional=True,
        ),
    ],
    cmdclass={"build_ext": build_ext},
)
