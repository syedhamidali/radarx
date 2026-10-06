#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx Retrieval
================

Every public module of this package (a name without a leading underscore) is
imported here and its public functions are re-exported, so a new retrieval
module needs no change in this file.

.. toctree::
    :maxdepth: 3

"""

import importlib
import pkgutil

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
    __doc__ += f"\n.. automodule:: {__name__}.{_name}\n"

del _name, _module, _public, _attr
