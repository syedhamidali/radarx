#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Sweep-level functions over extra dimensions and dask arrays
===========================================================

A sweep-level function works on one sweep, a ``(azimuth, range)`` plane. This
module lets such a function run on a stack of sweeps with extra leading
dimensions (``time``, ``volume``, ...) and on dask-backed fields without
loading them, with one mechanism for all functions:

* The function writes its work as a *core*: a function of a list of sweeps
  (NumPy arrays) that returns a list of results (NumPy arrays). The core is
  the code that existed before, with the kernel call and the NumPy fallback.
* :func:`apply_sweeps` runs the core. A plain sweep (2-D, NumPy) is passed to
  the core directly, so it costs nothing more than before. A stack is split
  into its sweeps, which go to the core in one call per block; a dask array
  is handled with :func:`xarray.apply_ufunc` (``dask="parallelized"``), which
  rechunks the sweep dimensions into one chunk and keeps the extra
  dimensions chunked as given.
* :func:`apply_sweeps_many` does so for the sweeps of a volume: plain sweeps
  go to the core together (one kernel call over all of them), the others one
  by one.
* :class:`Gates` and :func:`apply_gates` do the same for functions that work
  gate by gate on any dimensions (a sweep, a grid, a profile), and along one
  dimension (the vertical of a column integral).

Functions that cannot be lazy (they need a whole volume or all times, or
return tables) say so with :func:`dask_support`, which adds the rule to
their docstring and records it in :data:`DASK_SUPPORT`.

The module does not import dask.
"""

from __future__ import annotations

import re
import textwrap

import numpy as np
import xarray as xr

__all__ = [
    "DASK_SUPPORT",
    "apply_gates",
    "apply_gates_many",
    "apply_sweeps",
    "apply_sweeps_many",
    "dask_support",
    "extra_dims",
    "Gates",
    "is_dask",
    "is_plain",
    "Output",
    "ray_coordinate",
    "ray_dimension",
    "Sweep",
]

RANGE = "range"
RAY_DIMS = ("azimuth", "elevation")

# kinds of result and the dimensions they have besides the extra ones
KINDS = ("sweep", "ray", "range", "scalar", "gate")


def is_dask(obj):
    """Whether a DataArray or Variable is backed by a dask array."""
    return getattr(obj, "chunks", None) is not None


def ray_dimension(da):
    """
    The dimension of a field along which the rays are numbered.

    ``azimuth`` for a PPI, ``elevation`` for an RHI. A 2-D field with
    another name has the dimension that is not ``range``.

    Raises
    ------
    ValueError
        If the field has no ``range`` dimension or the ray dimension cannot
        be told.
    """
    others = [d for d in da.dims if d != RANGE]
    for name in RAY_DIMS:
        if name in others and RANGE in da.dims:
            return name
    if RANGE in da.dims and len(others) == 1:
        return others[0]
    raise ValueError(
        f"{da.name!r} has the dimensions {da.dims}: a sweep field needs "
        "'range' and 'azimuth' (PPI) or 'elevation' (RHI), and may have "
        "extra dimensions in front"
    )


def ray_coordinate(ds, ray_dim):
    """
    The coordinate of the rays (``azimuth`` or ``elevation``) as float64.

    Raises
    ------
    ValueError
        If the coordinate has other dimensions than the rays, as for a stack
        of sweeps with different azimuths.
    """
    coord = ds[ray_dim]
    if coord.dims != (ray_dim,):
        raise ValueError(
            f"{ray_dim!r} must have the dimension {ray_dim!r} only, not "
            f"{coord.dims}: the sweeps of a stack need the same rays"
        )
    return np.asarray(coord.values, dtype=np.float64)


def extra_dims(da):
    """The dimensions of a field besides its rays and ``range``."""
    ray = ray_dimension(da)
    return tuple(d for d in da.dims if d not in (ray, RANGE))


def is_plain(da):
    """Whether a field is one NumPy sweep: 2-D and not dask."""
    return da.ndim == 2 and not is_dask(da)


class Output:
    """
    One result of a core.

    Parameters
    ----------
    dtype : numpy.dtype or str
        Type of the result.
    kind : {"sweep", "ray", "range", "scalar", "gate"}
        Shape of the result of one sweep: ``(ray, range)``, ``(ray,)``,
        ``(range,)`` or a number. The extra dimensions of the input are
        added in front. ``"gate"`` is the shape of the fields of a
        :class:`Gates` input.
    new_dims : dict, optional
        Name and length of dimensions that the result has after the others
        (the classes of a score, say). The core returns them last.
    """

    def __init__(self, dtype, kind="sweep", new_dims=None):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
        self.dtype = np.dtype(dtype)
        self.kind = kind
        self.new_dims = dict(new_dims or {})

    def core_dims(self, ray):
        """Dimensions of the result of one sweep."""
        base = {
            "sweep": (ray, RANGE),
            "ray": (ray,),
            "range": (RANGE,),
            "scalar": (),
            "gate": (),
        }[self.kind]
        return base + tuple(self.new_dims)

    def __repr__(self):
        return f"Output({self.dtype}, {self.kind!r})"


def _values_as(variable, dims):
    """The values of a Variable with its dimensions in the order ``dims``."""
    if variable.dims != dims:
        variable = variable.transpose(*dims)
    return variable.values


class Sweep:
    """
    The inputs of a core for one sweep or stack of sweeps.

    Parameters
    ----------
    fields : sequence of xarray.DataArray or None
        Fields on ``(..., ray, range)``; ``None`` entries are passed on to
        the core as ``None``. All fields share the same grid. The first
        field that is not None gives the coordinates of the results.
    context : object, optional
        Passed to the core with every sweep of this stack (names, sizes,
        window lengths: anything the core needs besides the fields).
    """

    sliced = True

    def __init__(self, fields, context=None):
        self.fields = list(fields)
        self.context = context
        present = [f for f in self.fields if f is not None]
        if not present:
            raise ValueError("a sweep needs at least one field")
        self.reference = present[0]
        self.ray = ray_dimension(self.reference)
        self.extra = extra_dims(self.reference)
        self.core_dims = (self.ray, RANGE)
        self.dims = self.reference.dims
        for field in present[1:]:
            if set(field.dims) - set(self.reference.dims) or (
                ray_dimension(field) != self.ray
            ):
                raise ValueError(
                    f"{field.name!r} has the dimensions {field.dims}, not "
                    f"those of {self.reference.name!r} {self.reference.dims}"
                )

    @property
    def plain(self):
        """Whether all fields are 2-D NumPy arrays."""
        return all(is_plain(f) for f in self.fields if f is not None)

    def input_core_dims(self, fields):
        """Core dimensions of each of ``fields``."""
        return [list(self.core_dims)] * len(fields)

    def arrays(self):
        """The fields of a plain sweep as ``(ray, range)`` NumPy arrays."""
        want = (self.ray, RANGE)
        return tuple(
            None if f is None else _values_as(f.variable, want) for f in self.fields
        )


def _share_grid(fields, core_dim):
    """Whether the fields have the same dimensions and equal indexes."""
    first = fields[0]
    for field in fields[1:]:
        if field.dims != first.dims:
            return False
        for dim, index in field.indexes.items():
            if dim != core_dim and not index.equals(first.indexes.get(dim, index)):
                return False
    return True


def _dims_first(da, dims):
    """``da`` with the dimensions ``dims`` first, the others after them."""
    if da.dims[: len(dims)] == dims:
        return da
    return da.transpose(*dims, ...)


class Gates:
    """
    The inputs of a gate-wise core: fields broadcast against each other.

    Unlike a :class:`Sweep`, the core sees all dimensions of the fields at
    once, so the fields may be a sweep, a grid, a profile or a stack of them.

    Parameters
    ----------
    fields : sequence of xarray.DataArray or None
        The fields; the first one that is not None gives the dimensions,
        coordinates and order of the results. The others are broadcast to it
        (aligned by their coordinates) and must not have other dimensions.
    context : object, optional
        Passed to the core.
    core_dim : str, optional
        A dimension the core works along (the vertical of a column
        integral). The fields that have it are passed with it last; the
        results do not have it. Default: none.
    """

    sliced = False
    ray = None

    def __init__(self, fields, context=None, core_dim=None):
        present = [f for f in fields if f is not None]
        if not present:
            raise ValueError("a gate-wise function needs at least one field")
        self.core_dim = core_dim
        self.dims = tuple(d for d in present[0].dims if d != core_dim)
        if not _share_grid(present, core_dim):
            exclude = [] if core_dim is None else [core_dim]
            broadcast = iter(xr.broadcast(*present, exclude=exclude))
            fields = [None if f is None else next(broadcast) for f in fields]
        self.fields = [None if f is None else _dims_first(f, self.dims) for f in fields]
        self.reference = next(f for f in self.fields if f is not None)
        self.context = context

    @property
    def plain(self):
        """Whether no field is a dask array."""
        return not any(is_dask(f) for f in self.fields if f is not None)

    def arrays(self):
        """The broadcast fields as NumPy arrays."""
        return tuple(None if f is None else f.values for f in self.fields)

    def input_core_dims(self, fields):
        """Core dimensions of each of ``fields``."""
        return [[self.core_dim] if self.core_dim in f.dims else [] for f in fields]


def _as_dataarray(variable, sweep, name=None):
    """A result Variable on the coordinates of the reference field."""
    dims = set(variable.dims)
    coords = {
        key: coord
        for key, coord in sweep.reference.coords.variables.items()
        if set(coord.dims) <= dims
    }
    return xr.DataArray(variable, coords=coords, name=name)


def _on_reference(sweep, value, core):
    """The reference field with ``value`` (on the dimensions ``core``) as data."""
    ref = sweep.reference
    order = [core.index(d) for d in ref.dims]
    out = ref.copy(deep=False, data=np.transpose(value, order))
    out.attrs = {}
    out.name = None
    return out


def _plain_results(sweep, results, outputs):
    """DataArrays of the result of one plain sweep."""
    out = []
    for value, spec in zip(results, outputs):
        value = np.asarray(value, dtype=spec.dtype)
        core = spec.core_dims(sweep.ray)
        if spec.kind == "gate":
            core = sweep.dims + core
        if set(core) == set(sweep.reference.dims):
            out.append(_on_reference(sweep, value, core))
            continue
        da = _as_dataarray(xr.Variable(core, value), sweep)
        if spec.kind in ("sweep", "gate"):
            da = da.transpose(*sweep.dims, *spec.new_dims)
        out.append(da)
    return out


def _take(array, index):
    """The sweep ``index`` of a block; axes of length 1 are broadcast."""
    return array[tuple(0 if n == 1 else i for n, i in zip(array.shape, index))]


def _block(core, context, outputs, present, n_fields, one_by_one):
    """The function that ``apply_ufunc`` applies to blocks of the fields."""

    def run(*blocks):
        ndim = max(b.ndim for b in blocks)
        blocks = [b.reshape((1,) * (ndim - b.ndim) + b.shape) for b in blocks]
        lead = np.broadcast_shapes(*(b.shape[:-2] for b in blocks))
        nray, ngate = blocks[0].shape[-2:]
        shapes = {
            "sweep": (nray, ngate),
            "ray": (nray,),
            "range": (ngate,),
            "scalar": (),
        }
        results = [np.empty(lead + shapes[o.kind], dtype=o.dtype) for o in outputs]
        indices = list(np.ndindex(*lead))
        batch = []
        for index in indices:
            arrays = [None] * n_fields
            for slot, block in zip(present, blocks):
                arrays[slot] = _take(block, index)
            batch.append((context, tuple(arrays)))
        groups = [[item] for item in batch] if one_by_one else [batch]
        values = [res for group in groups if group for res in core(group)]
        for index, res in zip(indices, values):
            for dest, value in zip(results, res):
                dest[index] = value
        return tuple(results) if len(results) > 1 else results[0]

    return run


def _whole_block(core, context, outputs, present, n_fields):
    """The function that ``apply_ufunc`` applies to blocks of gate fields."""

    def run(*blocks):
        arrays = [None] * n_fields
        for slot, block in zip(present, blocks):
            arrays[slot] = block
        (res,) = core([(context, tuple(arrays))])
        results = tuple(np.asarray(v, dtype=o.dtype) for v, o in zip(res, outputs))
        return results if len(results) > 1 else results[0]

    return run


def _lazy_results(core, sweep, outputs, one_by_one):
    """Results of a stack or a dask field, as DataArrays."""
    present = [k for k, f in enumerate(sweep.fields) if f is not None]
    ray = sweep.ray
    if sweep.sliced:
        func = _block(
            core, sweep.context, outputs, present, len(sweep.fields), one_by_one
        )
    else:
        func = _whole_block(core, sweep.context, outputs, present, len(sweep.fields))
    result = xr.apply_ufunc(
        func,
        *(sweep.fields[k].variable for k in present),
        input_core_dims=sweep.input_core_dims([sweep.fields[k] for k in present]),
        output_core_dims=[list(o.core_dims(ray)) for o in outputs],
        dask="parallelized",
        output_dtypes=[o.dtype for o in outputs],
        dask_gufunc_kwargs={
            "allow_rechunk": True,
            "output_sizes": {k: v for o in outputs for k, v in o.new_dims.items()},
        },
        keep_attrs=False,
    )
    variables = result if isinstance(result, tuple) else (result,)
    out = []
    for var, spec in zip(variables, outputs):
        da = _as_dataarray(var, sweep)
        if spec.kind in ("sweep", "gate"):
            order = [d for d in sweep.dims if d in da.dims]
            da = da.transpose(*order, *spec.new_dims)
        out.append(da)
    return out


def apply_sweeps(core, sweep, outputs, one_by_one=False, raw=False):
    """
    Run a core on a sweep, a stack of sweeps or a dask-backed field.

    Parameters
    ----------
    core : callable
        ``core(batch)`` with ``batch`` a list of ``(context, arrays)``, one
        entry per sweep, where ``arrays`` is the tuple of the ``(ray, range)``
        NumPy arrays of the fields (None where a field is None). It returns
        a list with, for every sweep, the tuple of result arrays in the
        order of ``outputs``.
    sweep : Sweep
        The fields and the context.
    outputs : sequence of Output
        Type and shape of the results.
    one_by_one : bool, optional
        Call the core with one sweep at a time for a stack or dask field, so
        that a decision the core takes over its batch (the sign of the phase,
        say) does not depend on the chunks. Default False.
    raw : bool, optional
        Return the arrays without the DataArrays around them, for a caller
        that builds its own (the dimensions are those of the DataArrays).
        Default False.

    Returns
    -------
    list of xarray.DataArray
        One DataArray per output, on the coordinates of the first field. The
        data are dask arrays if a field is, else NumPy arrays. Names and
        attributes are for the caller to set.
    """
    if sweep.plain:
        (res,) = core([(sweep.context, sweep.arrays())])
        if raw:
            return [np.asarray(v, dtype=o.dtype) for v, o in zip(res, outputs)]
        return _plain_results(sweep, res, outputs)
    results = _lazy_results(core, sweep, outputs, one_by_one)
    return [da.data for da in results] if raw else results


def apply_sweeps_many(core, sweeps, outputs, one_by_one=False):
    """
    Run a core on the sweeps of a volume.

    The plain sweeps go to the core in one call (one kernel call over all of
    them); the others one by one, each lazy if its fields are.

    Parameters
    ----------
    core, outputs, one_by_one
        As for :func:`apply_sweeps`.
    sweeps : sequence of Sweep

    Returns
    -------
    list of list of xarray.DataArray
        For every sweep, one DataArray per output.
    """
    results = [None] * len(sweeps)
    plain = [k for k, sw in enumerate(sweeps) if sw.plain]
    if plain:
        batch = [(sweeps[k].context, sweeps[k].arrays()) for k in plain]
        for k, res in zip(plain, core(batch)):
            results[k] = _plain_results(sweeps[k], res, outputs)
    for k, sw in enumerate(sweeps):
        if results[k] is None:
            results[k] = _lazy_results(core, sw, outputs, one_by_one)
    return results


# a gate-wise core is run the same way
apply_gates = apply_sweeps
apply_gates_many = apply_sweeps_many


# --------------------------------------------------------------------------
# documentation of the behaviour on dask arrays
# --------------------------------------------------------------------------

#: Behaviour on dask input of the functions that declare it, by qualified name.
DASK_SUPPORT = {}

LAZY = "lazy"
LAZY_ONLY = "lazy_only"
COMPUTES = "computes"
EAGER = "eager"

_LAZY_NOTE = (
    "Dask and extra dimensions. The fields may have extra leading dimensions "
    "(for example ``time`` or ``volume``): each sweep is processed on its own "
    "and the result has the same extra dimensions. Dask arrays stay lazy: the "
    "result is a dask array and nothing is computed until it is (the sweep "
    "dimensions are rechunked into one chunk)."
)
_LAZY_ONLY_NOTE = (
    "Dask. Dask arrays stay lazy: the result is a dask array and nothing is "
    "computed until it is."
)
_COMPUTES_NOTE = (
    "Dask. This function computes the dask array: {why} The input is loaded "
    "into memory and the result is held in memory."
)
_HEADER = re.compile(r"^( *)([A-Z][A-Za-z ]*)\n\1-{3,}$", re.MULTILINE)
_PLACEHOLDER = re.compile(r"^ *\{\w+\} *$", re.MULTILINE)
_BEFORE = ("References", "Examples", "See Also")


def _add_note(doc, text):
    """
    ``doc`` with the paragraph ``text`` at the end of its Notes section.

    A line that is only a placeholder (``{references}``), which a module
    fills in later, ends the section as a header does.
    """
    headers = list(_HEADER.finditer(doc))
    indent = headers[0].group(1) if headers else "    "
    paragraph = textwrap.fill(
        text, width=79, initial_indent=indent, subsequent_indent=indent
    )
    marks = [(h.start(), h.group(2)) for h in headers]
    marks += [(m.start(), "{}") for m in _PLACEHOLDER.finditer(doc)]
    marks.sort()
    for k, (start, name) in enumerate(marks):
        if name == "Notes":
            end = marks[k + 1][0] if k + 1 < len(marks) else len(doc)
            body = doc[:end].rstrip("\n ")
            return body + "\n\n" + paragraph + "\n\n" + doc[end:].lstrip("\n")
    section = f"{indent}Notes\n{indent}-----\n{paragraph}\n\n"
    for start, name in marks:
        if name in _BEFORE or name == "{}":
            return doc[:start] + section + doc[start:]
    return doc.rstrip("\n ") + "\n\n" + section.rstrip("\n") + "\n" + indent


def dask_support(kind, why=""):
    """
    Decorator that states how a function treats dask arrays.

    The rule goes into the Notes of the docstring and into
    :data:`DASK_SUPPORT`, which the documentation and the tests read.

    Parameters
    ----------
    kind : {"lazy", "lazy_only", "computes", "eager"}
        ``"lazy"``: the result is a dask array for dask input and the
        function takes extra leading dimensions. ``"lazy_only"``: the same,
        without the statement about extra dimensions. ``"computes"``: the
        function needs a whole volume or all times and computes dask input.
        ``"eager"``: the result cannot be lazy (its size depends on the
        data) and the function computes dask input.
    why : str, optional
        For ``"computes"``: why the function needs the data in memory, one
        sentence. For the other kinds: a sentence on an exception that is
        added to the paragraph.

    Returns
    -------
    callable
        A decorator that returns the function itself with the longer
        docstring.
    """
    if kind not in (LAZY, LAZY_ONLY, COMPUTES, EAGER):
        raise ValueError(
            f"kind must be 'lazy', 'lazy_only', 'computes' or 'eager', not {kind!r}"
        )
    if kind in (COMPUTES, EAGER) and not why:
        raise ValueError("a function that computes dask input needs a reason")

    def decorate(func):
        text = {
            LAZY: _LAZY_NOTE,
            LAZY_ONLY: _LAZY_ONLY_NOTE,
            COMPUTES: _COMPUTES_NOTE.format(why=why),
            EAGER: _COMPUTES_NOTE.format(why=why),
        }[kind]
        if kind in (LAZY, LAZY_ONLY) and why:
            text = f"{text} {why}"
        func.__doc__ = _add_note(func.__doc__ or "", text)
        module = func.__module__.replace("radarx.", "")
        DASK_SUPPORT[f"{module}.{func.__name__}"] = (kind, why)
        return func

    return decorate
