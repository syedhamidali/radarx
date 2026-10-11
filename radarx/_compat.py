#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Deprecated argument names
=========================

radarx uses one name per concept in every function: ``obj`` for the first
argument, ``dbz`` for reflectivity, ``field`` for the radial velocity,
``min_<quantity>`` for thresholds, and ``n_threads`` before ``engine`` (both
keyword-only). The names that were used before keep working for one release:
:func:`renamed_arguments` maps an old keyword to the new one and warns with a
``DeprecationWarning`` that names both.

Positional use is not affected by a rename, because the position of an
argument does not change.
"""

import functools
import inspect
import warnings

__all__ = ["renamed_arguments", "keyword_only_after"]

# (function, old name, new name) of every alias, for the tests and the docs
RENAMED = []


def _message(function, old, new):
    return (
        f"{function}(): the argument {old!r} is deprecated and will be removed "
        f"in a future release; use {new!r}"
    )


def translate_keywords(function, kwargs, renames):
    """
    Replace the deprecated keywords of ``kwargs`` by the new names.

    Parameters
    ----------
    function : str
        Name of the function, for the messages.
    kwargs : dict
        Keyword arguments of the call; changed in place.
    renames : dict
        Maps an old name to the new one.

    Raises
    ------
    TypeError
        If both the old and the new name are given.
    """
    for old, new in renames.items():
        if old not in kwargs:
            continue
        if new in kwargs:
            raise TypeError(
                f"{function}() got the argument {new!r} and its deprecated "
                f"name {old!r}; give only {new!r}"
            )
        warnings.warn(_message(function, old, new), DeprecationWarning, stacklevel=3)
        kwargs[new] = kwargs.pop(old)
    return kwargs


def renamed_arguments(**renames):
    """
    Decorator that accepts the old names of renamed arguments.

    Parameters
    ----------
    **renames
        ``old_name="new_name"`` for every renamed argument.

    Returns
    -------
    callable
        A decorator. The decorated function keeps its signature, which has
        the new names only. A call with an old name warns
        (``DeprecationWarning``) and uses the value for the new name.

    Examples
    --------
    >>> @renamed_arguments(dbzh="dbz")
    ... def f(obj, dbz=None):
    ...     return dbz
    >>> f(None, dbz=3)
    3
    """

    def decorate(func):
        parameters = inspect.signature(func).parameters
        for old, new in renames.items():
            if old in parameters or new not in parameters:
                raise ValueError(
                    f"{func.__name__}: cannot rename {old!r} to {new!r} "
                    f"for parameters {list(parameters)}"
                )
            RENAMED.append((f"{func.__module__}.{func.__name__}", old, new))

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if renames.keys() & kwargs.keys():
                translate_keywords(func.__name__, kwargs, renames)
            return func(*args, **kwargs)

        wrapper.__radarx_renames__ = dict(renames)
        return wrapper

    return decorate


def keyword_only_after(*names):
    """
    Decorator for arguments that became keyword-only.

    Positional arguments beyond the positional parameters of the function are
    assigned to ``names``, in order, with a ``DeprecationWarning``.

    Parameters
    ----------
    *names
        Names of the arguments that follow the last positional parameter, in
        the order they had when they were positional.

    Returns
    -------
    callable
        A decorator.
    """

    def decorate(func):
        n_positional = sum(
            p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            for p in inspect.signature(func).parameters.values()
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) > n_positional:
                extra = args[n_positional:][: len(names)]
                args = args[:n_positional] + args[n_positional + len(extra) :]
                given = names[: len(extra)]
                clash = [n for n in given if n in kwargs]
                if clash:
                    raise TypeError(
                        f"{func.__name__}() got multiple values for {clash[0]!r}"
                    )
                warnings.warn(
                    f"{func.__name__}(): passing {', '.join(given)} as positional "
                    "arguments is deprecated and will be removed in a future "
                    "release; pass them as keywords",
                    DeprecationWarning,
                    stacklevel=2,
                )
                kwargs.update(zip(given, extra))
            return func(*args, **kwargs)

        return wrapper

    return decorate
