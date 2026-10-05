#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Deferred imports that keep ``import radarx`` fast."""

import importlib


class LazyModule:
    """
    Stand-in for a module that is imported on first attribute access.

    Parameters
    ----------
    name : str
        Full module name, e.g. ``"matplotlib.pyplot"``.
    """

    def __init__(self, name):
        self._name = name
        self._module = None

    def __getattr__(self, attr):
        if self._module is None:
            self._module = importlib.import_module(self._name)
        return getattr(self._module, attr)


def register_radar_cmaps():
    """Register the cmweather radar colormaps (e.g. ``ChaseSpectral``)."""
    import cmweather  # noqa: F401
