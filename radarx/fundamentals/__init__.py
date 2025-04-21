#!/usr/bin/env python
# Copyright (c) 2024-2025, Radarx Developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radar Fundmentals
=================

This module adapts and extends concepts and calculations inspired by the CSU Radar Tools and wradlib packages. We acknowledge their foundational contributions to radar meteorology software development.

.. toctree::
    :maxdepth: 4

.. automodule:: radarx.fundamentals.attenuation
.. automodule:: radarx.fundamentals.beam
.. automodule:: radarx.fundamentals.common
.. automodule:: radarx.fundamentals.constants
.. automodule:: radarx.fundamentals.doppler
.. automodule:: radarx.fundamentals.geometry
.. automodule:: radarx.fundamentals.power
.. automodule:: radarx.fundamentals.principles
.. automodule:: radarx.fundamentals.reflectivity
.. automodule:: radarx.fundamentals.scattering
.. automodule:: radarx.fundamentals.system
.. automodule:: radarx.fundamentals.timing
.. automodule:: radarx.fundamentals.variables

"""

from .attenuation import *  # noqa
from .beam import *  # noqa
from .common import *  # noqa
from .constants import *  # noqa
from .doppler import *  # noqa
from .geometry import *  # noqa
from .power import *  # noqa
from .principles import *  # noqa
from .reflectivity import *  # noqa
from .scattering import *  # noqa
from .system import *  # noqa
from .timing import *  # noqa
from .variables import *  # noqa

__all__ = [s for s in dir() if not s.startswith("_")]
