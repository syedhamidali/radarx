#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Build the optional compiled cone-gridding kernel.

Project metadata lives in pyproject.toml. The extension is optional: if it
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
    ],
    cmdclass={"build_ext": build_ext},
)
