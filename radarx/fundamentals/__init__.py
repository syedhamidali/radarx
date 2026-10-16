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
.. automodule:: radarx.fundamentals.refraction
.. automodule:: radarx.fundamentals.scattering
.. automodule:: radarx.fundamentals.system
.. automodule:: radarx.fundamentals.tmatrix
.. automodule:: radarx.fundamentals.timing
.. automodule:: radarx.fundamentals.variables

"""

from . import principles, scattering
from .attenuation import *  # noqa
from .beam import *  # noqa
from .common import *  # noqa
from .constants import *  # noqa
from .doppler import *  # noqa
from .geometry import *  # noqa
from .power import *  # noqa
from .principles import *  # noqa
from .reflectivity import *  # noqa
from .refraction import *  # noqa
from .scattering import *  # noqa
from .system import *  # noqa
from .timing import *  # noqa
from .tmatrix import *  # noqa
from .variables import *  # noqa

# Four names are defined in two modules with different meaning:
#   absorption_coefficient, scattering_coefficient, extinction_coefficient:
#       attenuation (diameter, cross section in m^2, m = n - jk) and
#       scattering (radius, efficiency, any sign of k);
#   doppler_frequency_shift: doppler (frequency, vr) and principles
#       (v_radial, wavelength).
# The star imports above let the later module win silently. The convention is
# the namespace: these names are not exported here, use ``attenuation.<name>``,
# ``scattering.<name>``, ``doppler.<name>`` or ``principles.<name>``. The plain
# names still resolve to what they used to be, with a DeprecationWarning, and
# are removed after the next release.
_AMBIGUOUS = {
    "absorption_coefficient": ("scattering", scattering.absorption_coefficient),
    "scattering_coefficient": ("scattering", scattering.scattering_coefficient),
    "extinction_coefficient": ("scattering", scattering.extinction_coefficient),
    "doppler_frequency_shift": ("principles", principles.doppler_frequency_shift),
}
for _name in _AMBIGUOUS:
    globals().pop(_name, None)
del _name


def __getattr__(name):
    if name in _AMBIGUOUS:
        import warnings

        module, func = _AMBIGUOUS[name]
        warnings.warn(
            f"radarx.fundamentals.{name} is ambiguous and is deprecated; it is "
            f"the function of radarx.fundamentals.{module}. Use "
            f"radarx.fundamentals.{module}.{name} (or the function of the other "
            "module that defines this name) explicitly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return func
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [s for s in dir() if not s.startswith("_")]
