#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Registry of ``.radarx`` accessor methods.

A feature module adds its accessor methods where the feature lives instead of
editing :mod:`radarx.accessors`::

    from .._registry import accessor_method

    @accessor_method("dataset", "datatree", name="echo_mask")
    def _echo_mask_accessor(self, **kwargs):
        # the docstring of this function is shown on ``ds.radarx.echo_mask``
        return echo_mask(self.xarray_obj, **kwargs)

``self`` is the accessor; ``self.xarray_obj`` is the wrapped xarray object.
:mod:`radarx.accessors` attaches every registered method to the accessor
classes when it is imported.
"""

KINDS = ("dataarray", "dataset", "datatree")

_methods = {kind: {} for kind in KINDS}


def accessor_method(*kinds, name=None):
    """Register a function as a ``.radarx`` method for the given object kinds."""
    unknown = set(kinds) - set(KINDS)
    if not kinds or unknown:
        raise ValueError(f"kinds must be some of {KINDS}, not {kinds!r}")

    def register(func):
        method = name or func.__name__
        for kind in kinds:
            if method in _methods[kind]:
                raise ValueError(f"accessor method {method!r} is already registered")
            _methods[kind][method] = func
        return func

    return register


def registered(kind):
    """Methods registered for one object kind, as ``{name: function}``."""
    return dict(_methods[kind])
