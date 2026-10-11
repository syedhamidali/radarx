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

The module does not import dask.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = [
    "Output",
    "Sweep",
    "apply_sweeps",
    "apply_sweeps_many",
    "extra_dims",
    "is_dask",
    "is_plain",
    "ray_coordinate",
    "ray_dimension",
]

RANGE = "range"
RAY_DIMS = ("azimuth", "elevation")

# kinds of result and the dimensions they have besides the extra ones
KINDS = ("sweep", "ray", "range", "scalar")


def is_dask(obj):
    """Whether an xarray object (or a Variable) is backed by a dask array."""
    return getattr(obj, "chunks", None) is not None and not isinstance(obj, dict)


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
    kind : {"sweep", "ray", "range", "scalar"}
        Shape of the result of one sweep: ``(ray, range)``, ``(ray,)``,
        ``(range,)`` or a number. The extra dimensions of the input are
        added in front.
    """

    def __init__(self, dtype, kind="sweep"):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
        self.dtype = np.dtype(dtype)
        self.kind = kind

    def core_dims(self, ray):
        """Dimensions of the result of one sweep."""
        return {
            "sweep": (ray, RANGE),
            "ray": (ray,),
            "range": (RANGE,),
            "scalar": (),
        }[self.kind]

    def __repr__(self):
        return f"Output({self.dtype}, {self.kind!r})"


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

    def __init__(self, fields, context=None):
        self.fields = list(fields)
        self.context = context
        present = [f for f in self.fields if f is not None]
        if not present:
            raise ValueError("a sweep needs at least one field")
        self.reference = present[0]
        self.ray = ray_dimension(self.reference)
        self.extra = extra_dims(self.reference)
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

    def arrays(self):
        """The fields of a plain sweep as ``(ray, range)`` NumPy arrays."""
        return tuple(
            None if f is None else f.transpose(self.ray, RANGE).values
            for f in self.fields
        )


def _as_dataarray(variable, sweep, name=None):
    """A result Variable on the coordinates of the reference field."""
    ref = sweep.reference
    coords = {
        key: coord
        for key, coord in ref.coords.items()
        if set(coord.dims) <= set(variable.dims)
    }
    return xr.DataArray(variable, coords=coords, name=name)


def _plain_results(sweep, results, outputs):
    """DataArrays of the result of one plain sweep."""
    out = []
    for value, spec in zip(results, outputs):
        dims = spec.core_dims(sweep.ray)
        var = xr.Variable(dims, np.asarray(value, dtype=spec.dtype))
        da = _as_dataarray(var, sweep)
        if spec.kind == "sweep":
            da = da.transpose(*sweep.reference.dims)
        out.append(da)
    return out


def _take(array, index):
    """The sweep ``index`` of a block; axes of length 1 are broadcast."""
    return array[tuple(0 if n == 1 else i for n, i in zip(array.shape, index))]


def _block(core, context, outputs, present, n_fields, one_by_one):
    """The function that ``apply_ufunc`` applies to blocks of the fields."""

    def run(*blocks):
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


def _lazy_results(core, sweep, outputs, one_by_one):
    """Results of a stack or a dask field, as DataArrays."""
    present = [k for k, f in enumerate(sweep.fields) if f is not None]
    ray = sweep.ray
    func = _block(core, sweep.context, outputs, present, len(sweep.fields), one_by_one)
    result = xr.apply_ufunc(
        func,
        *(sweep.fields[k].variable for k in present),
        input_core_dims=[[ray, RANGE]] * len(present),
        output_core_dims=[list(o.core_dims(ray)) for o in outputs],
        dask="parallelized",
        output_dtypes=[o.dtype for o in outputs],
        dask_gufunc_kwargs={"allow_rechunk": True},
        keep_attrs=False,
    )
    variables = result if isinstance(result, tuple) else (result,)
    out = []
    for var, spec in zip(variables, outputs):
        da = _as_dataarray(var, sweep)
        if spec.kind == "sweep":
            order = [d for d in sweep.reference.dims if d in da.dims]
            da = da.transpose(*order)
        out.append(da)
    return out


def apply_sweeps(core, sweep, outputs, one_by_one=False):
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

    Returns
    -------
    list of xarray.DataArray
        One DataArray per output, on the coordinates of the first field. The
        data are dask arrays if a field is, else NumPy arrays. Names and
        attributes are for the caller to set.
    """
    if sweep.plain:
        (res,) = core([(sweep.context, sweep.arrays())])
        return _plain_results(sweep, res, outputs)
    return _lazy_results(core, sweep, outputs, one_by_one)


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
