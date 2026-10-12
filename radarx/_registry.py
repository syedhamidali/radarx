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

A method that only passes its arguments on to a function names the function
with ``wraps``. The method then has the signature of that function without
the first argument, so that ``help`` and tab completion show the same names,
defaults and keyword-only markers; the body of the registered function is
not used, only its docstring::

    @accessor_method("dataset", name="vil", wraps=vil)
    def _vil_accessor(self):
        '''Vertically integrated liquid of this volume.'''
"""

import inspect

KINDS = ("dataarray", "dataset", "datatree")

_methods = {kind: {} for kind in KINDS}


def method_signature(func, lead=()):
    """
    Signature of an accessor method that calls ``func``.

    Parameters
    ----------
    func : callable
        Function whose first argument is the xarray object.
    lead : sequence of inspect.Parameter, optional
        Parameters of the method that come before the others, for a method
        that takes more than the wrapped object (a second volume, say).

    Returns
    -------
    inspect.Signature
        ``self``, ``lead`` and the parameters of ``func`` after its first.
    """
    parameters = list(inspect.signature(func).parameters.values())
    first = inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)
    return inspect.Signature([first, *lead] + parameters[1:])


def forward(func, stub=None):
    """
    Accessor method that calls ``func`` on the wrapped object.

    Parameters
    ----------
    func : callable
        Function whose first argument is the xarray object.
    stub : callable, optional
        Function that provides the name, module and docstring of the method.
        Default: ``func`` itself.

    Returns
    -------
    callable
        ``method(self, *args, **kwargs)`` returning
        ``func(self.xarray_obj, *args, **kwargs)`` with the signature of
        ``func`` (first argument ``self``).
    """

    stub = func if stub is None else stub

    def method(self, *args, **kwargs):
        return func(self.xarray_obj, *args, **kwargs)

    method.__name__ = stub.__name__
    method.__qualname__ = stub.__qualname__
    method.__doc__ = stub.__doc__ or func.__doc__
    method.__module__ = stub.__module__
    method.__signature__ = method_signature(func)
    method.__radarx_target__ = func
    return method


def accessor_method(*kinds, name=None, wraps=None):
    """
    Register a function as a ``.radarx`` method for the given object kinds.

    Parameters
    ----------
    *kinds : {"dataarray", "dataset", "datatree"}
        Object kinds that get the method.
    name : str, optional
        Name of the method. Default: the name of the function.
    wraps : callable, optional
        Function the method calls with the wrapped object as first argument;
        the method takes its signature. The decorated function then only
        provides the docstring.
    """
    unknown = set(kinds) - set(KINDS)
    if not kinds or unknown:
        raise ValueError(f"kinds must be some of {KINDS}, not {kinds!r}")

    def register(func):
        method = name or func.__name__
        if wraps is not None:
            func = forward(wraps, func)
            func.__name__ = func.__qualname__ = method
        for kind in kinds:
            if method in _methods[kind]:
                raise ValueError(f"accessor method {method!r} is already registered")
            _methods[kind][method] = func
        return func

    return register


def accessor_for(func, *kinds, name=None):
    """
    Register ``func`` as a ``.radarx`` method with its own signature and docstring.

    Parameters
    ----------
    func : callable
        Function whose first argument is the xarray object.
    *kinds : {"dataarray", "dataset", "datatree"}
        Object kinds that get the method.
    name : str, optional
        Name of the method. Default: the name of ``func``.

    Returns
    -------
    callable
        The method.
    """
    return accessor_method(*kinds, name=name or func.__name__, wraps=func)(func)


def registered(kind):
    """Methods registered for one object kind, as ``{name: function}``."""
    return dict(_methods[kind])
