#!/usr/bin/env python
# Copyright (c) 2024, Hamid Ali Syed.
# Distributed under the MIT License. See LICENSE for more info.

"""
Xarray Backends
===============

.. toctree::
    :maxdepth: 4

.. automodule:: radarx.io.imd
"""

from .imd import read_sweep, read_volume  # noqa

__all__ = [s for s in dir() if not s.startswith("_")]
