#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Method provenance
=================

Every product function of :mod:`radarx.retrieve` and :mod:`radarx.grid` that
returns an xarray object writes four attributes on it:

``radarx_method``
    Short name of the method. Several methods, oldest first, are joined with
    `` | `` when a result was derived from an earlier radarx result.
``radarx_references``
    DOIs (or the keys of references without a DOI) of the papers the
    docstring cites, separated by a space. A DOI can contain ``;``, so that is
    no separator. The attribute is one string and survives NetCDF.
``radarx_version``
    The radarx version that wrote the result.
``history``
    One line per call, ``function(parameters)``, without a time stamp. The
    parameters are those that differ from the defaults.

:func:`radarx.cite` turns these attributes into citations.
"""

import functools
import inspect
import json
import re
import threading
from pathlib import Path

import xarray as xr

try:  # pragma: no cover
    from xarray import DataTree
except ImportError:  # pragma: no cover
    from datatree import DataTree

__all__ = ["set_provenance", "provenance", "read_provenance"]

METHOD = "radarx_method"
REFERENCES = "radarx_references"
VERSION = "radarx_version"
HISTORY = "history"

_METHOD_SEP = " | "
_SKIP_PARAMS = {"engine", "n_threads"}
_REGISTRY_FILE = Path(__file__).parent / "data" / "references.json"
_state = threading.local()
_registry_cache = {}

# Public functions that are decorated, by qualified name.
DECORATED = {}


def _version():
    from . import __version__

    return __version__


def load_registry():
    """Return the reference registry (DOI or key -> citation fields)."""
    if not _registry_cache:
        with open(_REGISTRY_FILE, encoding="utf-8") as f:
            _registry_cache.update(json.load(f))
    return _registry_cache


_SECTION_END = re.compile(r"\n[ \t]*[A-Z][A-Za-z ]+\n[ \t]*-{3,}")


def docstring_dois(doc):
    """DOIs listed in the References section of a numpydoc docstring."""
    if not doc:
        return []
    start = re.search(r"\n[ \t]*References[ \t]*\n[ \t]*-{3,}[ \t]*\n", doc)
    if start is None:
        return []
    text = doc[start.end() :]
    end = _SECTION_END.search(text)
    if end is not None:
        text = text[: end.start()]
    # a DOI that was wrapped after a "/", "(" or "-" continues on the next line
    text = re.sub(r"(doi\.org/[^\s]*[/(\-])\n[ \t]*", r"\1", text)
    dois = []
    for doi in re.findall(r"doi\.org/(\S+)", text):
        doi = doi.rstrip(".,;")
        if doi not in dois:
            dois.append(doi)
    return dois


def _split(value):
    return (value or "").split()


def _nodes(obj):
    """Objects that carry the attributes: the sweeps of a tree, not its root."""
    if isinstance(obj, DataTree):
        nodes = list(obj.subtree)
        return nodes[1:] if len(nodes) > 1 else nodes
    return [obj]


def _format_param(value):
    if isinstance(value, (bool, int, float, str)):
        return repr(value)
    if (
        isinstance(value, (tuple, list))
        and 0 < len(value) <= 4
        and all(isinstance(v, (bool, int, float, str)) for v in value)
    ):
        return repr(tuple(value))
    return None


def _is_default(signature, name, value):
    """Is ``value`` missing or equal to the default of the parameter?"""
    if value is None:
        return True
    default = signature.parameters[name].default
    if default is inspect.Parameter.empty:
        return False
    try:
        return bool(type(default) is type(value) and default == value)
    except Exception:  # pragma: no cover
        return False


def _history_line(function, params):
    items = []
    for key, value in (params or {}).items():
        text = _format_param(value)
        if text is not None:
            items.append(f"{key}={text}")
    return f"{function}({', '.join(items)})"


def _write(node, method, refs, line):
    attrs = node.attrs
    methods = [m for m in (attrs.get(METHOD) or "").split(_METHOD_SEP) if m]
    if method not in methods:
        methods.append(method)
    references = _split(attrs.get(REFERENCES))
    references += [r for r in refs if r not in references]
    attrs[METHOD] = _METHOD_SEP.join(methods)
    attrs[REFERENCES] = " ".join(references)
    attrs[VERSION] = str(_version())
    attrs[HISTORY] = f"{attrs[HISTORY]}\n{line}" if attrs.get(HISTORY) else line


def set_provenance(obj, method, refs, params=None, function=None):
    """
    Record the method that produced an xarray object.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        Result to annotate, in place. For a DataTree the sweeps are annotated,
        not the root (the root is the input's root).
    method : str
        Short name of the method. Say so where radarx differs from the
        paper it cites.
    refs : sequence of str
        DOIs, or registry keys for references without a DOI.
    params : dict, optional
        Key parameters for the history line. Only numbers, strings, booleans
        and short tuples of them are written. ``None`` is left out.
    function : str, optional
        Name of the calling function for the history line. Default: ``method``.

    Returns
    -------
    obj
        The same object.
    """
    line = _history_line(function or method, params)
    for node in _nodes(obj):
        _write(node, method, refs, line)
    return obj


def _is_xarray(obj):
    return isinstance(obj, (xr.DataArray, xr.Dataset, DataTree))


def _annotate(result, args, method, refs, params, function):
    """Annotate the xarray objects in a result (one level of containers)."""
    if _is_xarray(result):
        if any(result is a for a in args):
            result = result.copy(deep=False)
        return set_provenance(result, method, refs, params, function)
    if isinstance(result, (tuple, list)):
        items = [_annotate(r, args, method, refs, params, function) for r in result]
        return type(result)(items) if isinstance(result, tuple) else items
    if isinstance(result, dict):
        return {
            k: _annotate(v, args, method, refs, params, function)
            for k, v in result.items()
        }
    return result


class _Info:
    """
    Method and references of a decorated function.

    The references are read from the docstring when they are first needed,
    because some modules fill in the docstring after the function is defined.
    """

    def __init__(self, func, method, extra_refs):
        self.func = func
        self.method = method
        self.extra = list(extra_refs)
        self._doc = None
        self._refs = []

    @property
    def references(self):
        doc = self.func.__doc__
        if doc is not self._doc:
            self._doc = doc
            self._refs = docstring_dois(inspect.cleandoc(doc or "")) + self.extra
        return list(self._refs)

    def __getitem__(self, key):
        return getattr(self, key)


def provenance(method, extra_refs=()):
    """
    Decorator that records the method on the xarray results of a function.

    The references are the DOIs in the References section of the function's
    docstring, followed by ``extra_refs`` (registry keys of references
    without a DOI). Only the outermost call writes the attributes, so
    functions that call each other leave one entry. Results that are not
    xarray objects are returned as they are.

    Parameters
    ----------
    method : str
        Short name of the method.
    extra_refs : sequence of str, optional
        Registry keys of references the docstring cites without a DOI.
    """

    def decorate(func):
        signature = inspect.signature(func)
        module = func.__module__.replace("radarx.retrieve.", "").replace(
            "radarx.grid.", ""
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nested = getattr(_state, "depth", 0) > 0
            _state.depth = getattr(_state, "depth", 0) + 1
            try:
                result = func(*args, **kwargs)
            finally:
                _state.depth -= 1
            if nested:
                return result
            try:
                bound = signature.bind_partial(*args, **kwargs).arguments
            except TypeError:  # pragma: no cover
                bound = {}
            params = {
                k: v
                for k, v in bound.items()
                if k not in _SKIP_PARAMS and not _is_default(signature, k, v)
            }
            return _annotate(
                result,
                list(args) + list(kwargs.values()),
                method,
                info.references,
                params,
                func.__name__,
            )

        info = _Info(wrapper, method, extra_refs)
        wrapper.__radarx_provenance__ = info
        DECORATED[f"{module}.{func.__name__}"] = wrapper
        return wrapper

    return decorate


def read_provenance(obj):
    """
    Provenance written on an xarray object.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        Object to read. Every node of a DataTree (root included) and every
        data variable is read.

    Returns
    -------
    list of dict
        One dict per annotated node with ``method`` (list of str),
        ``references`` (list of str), ``version`` (str) and ``history``
        (list of str).
    """
    found = []
    holders = []
    for node in _nodes(obj):
        holders.append(node.attrs)
        if not isinstance(node, xr.DataArray):
            holders.extend(v.attrs for v in node.data_vars.values())
    for attrs in holders:
        if METHOD not in attrs and REFERENCES not in attrs:
            continue
        found.append(
            {
                "method": [
                    m for m in str(attrs.get(METHOD, "")).split(_METHOD_SEP) if m
                ],
                "references": _split(str(attrs.get(REFERENCES, ""))),
                "version": str(attrs.get(VERSION, "")),
                "history": [h for h in str(attrs.get(HISTORY, "")).split("\n") if h],
            }
        )
    return found
