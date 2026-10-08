#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx Retrieval
================

Every public module of this package (a name without a leading underscore) is
imported here and its public functions are re-exported, so a new retrieval
module needs no change in this file. The reference page groups the modules by
topic (``_SECTIONS`` below); a module not listed there appears under "Other".

.. toctree::
    :maxdepth: 3

"""

import importlib
import pkgutil

_SECTIONS = {
    "Quality control and preprocessing": ["qc", "dealias", "kdp", "advection"],
    "Products and profiles": ["cappi", "vertical_profiles"],
    "Wind retrieval and kinematics": ["single_doppler", "multidoppler", "shear"],
    "Precipitation microphysics": [
        "hid",
        "dsd",
        "dsd_bayes",
        "disdrometer",
        "evaporation",
    ],
    "Thermodynamics and cold pools": [
        "lagrangian",
        "diabatic_lagrangian",
        "coldpool",
        "wind_profile",
    ],
    "Convective hazards": ["lightning", "tornado", "biology"],
}

_modules = sorted(
    info.name
    for info in pkgutil.iter_modules(__path__)
    if not info.name.startswith("_")
)
__all__ = []
for _name in _modules:
    _module = importlib.import_module(f".{_name}", __name__)
    _public = getattr(
        _module, "__all__", [n for n in dir(_module) if not n.startswith("_")]
    )
    for _attr in _public:
        globals()[_attr] = getattr(_module, _attr)
        if _attr not in __all__:
            __all__.append(_attr)

_listed = {m for mods in _SECTIONS.values() for m in mods}
for _title, _names in {
    **_SECTIONS,
    "Other": [m for m in _modules if m not in _listed],
}.items():
    _names = [m for m in _names if m in _modules]
    if _names:
        __doc__ += f"\n{_title}\n{'-' * len(_title)}\n"
        for _name in _names:
            __doc__ += f"\n.. automodule:: {__name__}.{_name}\n"

del _name, _module, _public, _attr, _title, _names, _listed
