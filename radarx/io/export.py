#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx Export to Files
======================

Write radarx results to files that other tools read.

Volumes (``xarray.DataTree`` of sweeps, with the fields radarx derived) go to
CfRadial1, CfRadial2 and ODIM_H5 through the writers of :mod:`xradar`, which do
the format work. Gridded Datasets go to CF-compliant NetCDF, and any radarx
result goes to Zarr, through :meth:`xarray.Dataset.to_netcdf` and
:meth:`xarray.Dataset.to_zarr`. What radarx adds around them is a preparation
of a copy of the object (the input is never changed, and the arrays are
shared with it, not copied, apart from the booleans of an ODIM_H5 file):

* attributes that the file cannot store are converted (``None`` is dropped,
  booleans become ``"true"``/``"false"`` in NetCDF, lists and numpy values
  become what the format accepts, dictionaries become JSON strings);
* ``_FillValue`` is moved from the attributes to the encoding of floating
  variables and removed from integer variables (a ``_FillValue`` attribute on
  the HID class codes would read class 0 back as NaN);
* stale encodings (chunk layout, source file, time units) are dropped, and
  the integer packing of a floating variable (``scale_factor``, ``add_offset``
  and an integer ``dtype``, as xradar readers set it) is dropped by default, so
  that NaN and values outside the packed range are not written as garbage;
* the provenance (``radarx_method``, ``radarx_references``, ``radarx_version``
  and ``history``) is kept where the format can hold it (see below).

:func:`write_netcdf_cf` also adds the CF georeferencing of a grid: a
``grid_mapping`` variable with the ``crs_wkt`` radarx writes, ``grid_mapping``
on every field, ``standard_name``, ``units`` and ``axis`` on the coordinates,
2-D ``lat`` and ``lon``, and bounds of the axes.

The functions are named ``write_*`` because :func:`radarx.io.to_cfradial2`
already names the deprecated IMD converter; the ``.radarx`` accessors are
``to_cfradial1``, ``to_cfradial2``, ``to_odim``, ``to_zarr`` and
``to_netcdf_cf``. The writers are not decorated with the provenance decorator,
like :func:`radarx.io.write_wrfda_radar`: they return the path, not a result.

What is written where
---------------------

A volume has the root attributes, the sweeps (each with attributes, fields,
and the coordinates ``azimuth``, ``elevation``, ``time`` and ``range``) and
the georeferenced ``x``, ``y`` and ``z`` that ``dtree.xradar.georeference()``
adds. The derived fields of radarx (``KDP``, ``PHIDP_processed``,
``ECHO_CLASS``, ``METEO_SCORE``, ``METEO_MASK``, ``HID``, ...) are fields on
the ray by gate grid like the measured ones. What each format keeps was
checked by reading the files back (``tests/test_export.py``):

* **Zarr**: everything, identical on reading back, with booleans and ``None``
  in attributes and lazy dask arrays.
* **CfRadial2**: the root attributes and the fields with a ``range``
  dimension, with their attributes. The sweep attributes, the variables
  without a ``range`` dimension (``PHIDP_OFFSET``) and ``x``, ``y`` and ``z``
  are dropped by the xradar writer, and the ray dimension becomes ``time``.
  The provenance of the sweeps is merged onto the root attributes.
* **CfRadial1**: one dataset along ``time`` with all sweeps stacked, the
  fields and the per-ray variables, no sweep attributes (provenance merged
  onto the global attributes), no ``x``, ``y`` and ``z``.
* **ODIM_H5**: the fields on the ray by gate grid under their name as
  ``what/quantity``; attributes of fields, sweeps and root are not stored,
  only the ``radarx_*`` provenance and ``history``, in ``how`` groups.
* **NetCDF (CF)** and **Zarr** for grids: all variables; the CF additions
  of :func:`write_netcdf_cf`.

Booleans in attributes are written as ``"true"`` and ``"false"`` to NetCDF
(CfRadial1, CfRadial2, CF NetCDF) and as numbers to ODIM_H5 (variables).

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = [
    "write_cfradial1",
    "write_cfradial2",
    "write_odim",
    "write_zarr",
    "write_netcdf_cf",
]

__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

import json
import os
from pathlib import Path

import numpy as np
import xarray as xr

from .._gridcf import CRS_NAMES, aeqd_crs, georeference, user_crs
from .._provenance import HISTORY, METHOD, REFERENCES, VERSION
from .._registry import accessor_method

try:  # pragma: no cover
    from xarray import DataTree
except ImportError:  # pragma: no cover
    from datatree import DataTree

#: attributes radarx writes on results, and that the writers keep
PROVENANCE = (METHOD, REFERENCES, VERSION, HISTORY)
_METHOD_SEP = " | "

#: encoding keys that describe the layout of the source and not the data
_LAYOUT = (
    "chunks",
    "preferred_chunks",
    "chunksizes",
    "original_shape",
    "source",
    "contiguous",
    "shards",
    "group",
)
#: encoding keys of a variable that is stored as integers
_PACKING = ("dtype", "scale_factor", "add_offset", "_FillValue")
#: encoding keys of a time variable that a new file should choose itself
_TIME = ("units", "calendar", "dtype", "_FillValue")

_TARGETS = ("netcdf", "zarr")


# ---------------------------------------------------------------------------
# attributes
# ---------------------------------------------------------------------------
class _Drop:
    """Marker for an attribute that is left out of the file."""


_DROP = _Drop()


def _is_placeholder(value):
    """Is this the ``"None"`` string that readers write for an unknown value?"""
    return isinstance(value, str) and value == "None"


def _is_number(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _clean_value(value, target):
    """
    One attribute value in a form the file accepts, or ``None`` to drop it.

    ``target`` is ``"netcdf"`` (also CfRadial1 and CfRadial2) or ``"zarr"``,
    which keeps ``None``, booleans and lists as JSON holds them. The marker
    ``_DROP`` stands for an attribute that the file cannot store at all.
    """
    if value is None:
        return _DROP if target == "netcdf" else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value) if target == "zarr" else str(bool(value)).lower()
    if isinstance(value, np.str_):
        return str(value)
    if isinstance(value, np.floating) and value.dtype.itemsize < 4:
        return np.float32(value)
    if isinstance(value, (str, bytes, np.generic, int, float)):
        if isinstance(value, np.generic) and value.dtype.kind in "mMOU":
            return str(value)
        return value
    if isinstance(value, np.ndarray):
        return _clean_array(value, target)
    if isinstance(value, (list, tuple)):
        return _clean_sequence(value, target)
    if isinstance(value, dict):
        return json.dumps(value, default=_json_default, sort_keys=True)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _clean_array(value, target):
    if value.dtype.kind in "mMOUS" or value.ndim == 0:
        return _clean_sequence(value.ravel().tolist(), target)
    value = value.ravel()
    if value.dtype.kind == "b":
        return value if target == "zarr" else value.astype("int8")
    if value.dtype.kind == "f" and value.dtype.itemsize < 4:
        return value.astype("float32")
    return value


def _clean_sequence(items, target):
    items = [_clean_value(v, target) for v in items]
    items = [v for v in items if v is not _DROP]
    if all(isinstance(v, str) for v in items):
        return items if items else ""
    if all(_is_number(v) for v in items):
        return np.asarray(items)
    if target == "zarr":
        return json.loads(json.dumps(items, default=_json_default))
    return json.dumps(items, default=_json_default)


def _clean_attrs(attrs, target):
    """A new dict with the attributes in a form that ``target`` can store."""
    out = {}
    for key, value in attrs.items():
        value = _clean_value(value, target)
        if value is not _DROP:
            out[str(key)] = value
    return out


# ---------------------------------------------------------------------------
# variables
# ---------------------------------------------------------------------------
def _fix_variable(var, target, packing):
    """Prepare the attributes and the encoding of one variable, in place."""
    attrs = _clean_attrs(var.attrs, target)
    encoding = {k: v for k, v in var.encoding.items() if k not in _LAYOUT}
    kind = var.dtype.kind
    if "_FillValue" in attrs:
        fill = attrs.pop("_FillValue")
        if kind == "f":
            encoding.setdefault("_FillValue", fill)
    if kind == "f" and packing == "drop":
        stored = encoding.get("dtype")
        if stored is not None and np.dtype(stored).kind in "iu":
            for key in _PACKING:
                encoding.pop(key, None)
    elif kind in "mM":
        for key in _TIME:
            encoding.pop(key, None)
    var.attrs = attrs
    var.encoding = encoding


def _check_packing(packing):
    if packing not in ("drop", "keep"):
        raise ValueError(f"packing must be 'drop' or 'keep', not {packing!r}")


def _prepare_dataset(ds, target, packing):
    """Shallow copy of a Dataset with attributes and encodings made writable."""
    out = ds.copy(deep=False)
    for var in out.variables.values():
        _fix_variable(var, target, packing)
    out.attrs = _clean_attrs(ds.attrs, target)
    return out


def _prepare_tree(dtree, target, packing):
    """Shallow copy of a DataTree with attributes and encodings made writable."""
    out = dtree.copy(deep=False)
    for node in out.subtree:
        node.attrs = _clean_attrs(node.attrs, target)
        for var in node.variables.values():
            _fix_variable(var, target, packing)
    return out


def _as_dataset(obj):
    if isinstance(obj, xr.DataArray):
        return obj.to_dataset(name=obj.name or "data")
    return obj


def _path(path):
    return Path(os.fspath(path))


def _require_tree(dtree, name):
    if not isinstance(dtree, DataTree):
        raise TypeError(
            f"{name} writes a radar volume: pass an xarray.DataTree with sweep_* "
            f"nodes, not {type(dtree).__name__}"
        )
    if not list(dtree.match("sweep_*")):
        raise ValueError(f"{name} needs sweep_* nodes in the DataTree")


def _xradar_writer(name):
    try:
        import xradar.io as xio
    except ImportError as err:  # pragma: no cover - xradar is a dependency
        raise ImportError(f"{name} needs xradar") from err
    writer = getattr(xio, name, None)
    if writer is None:
        raise ImportError(f"this xradar has no {name}; install xradar >= 0.11")
    return writer


# ---------------------------------------------------------------------------
# provenance in the global attributes (CfRadial1 has no sweep attributes)
# ---------------------------------------------------------------------------
def _fold_provenance(tree):
    """
    Move the provenance of the sweeps onto the root attributes, in place.

    The CfRadial1 writer of xradar keeps the global attributes of the root and
    drops those of the sweeps; the variables keep theirs.
    """
    sources = [node.attrs for node in tree.subtree if node is not tree]
    root = tree.attrs
    methods = [m for m in str(root.get(METHOD, "")).split(_METHOD_SEP) if m]
    references = str(root.get(REFERENCES, "")).split()
    lines = [
        line
        for line in str(root.get(HISTORY, "")).split("\n")
        if line and line != "None"
    ]
    version = root.get(VERSION)
    for attrs in sources:
        for method in str(attrs.get(METHOD, "")).split(_METHOD_SEP):
            if method and method not in methods:
                methods.append(method)
        for ref in str(attrs.get(REFERENCES, "")).split():
            if ref not in references:
                references.append(ref)
        for line in str(attrs.get(HISTORY, "")).split("\n"):
            if line and line != "None" and line not in lines:
                lines.append(line)
        version = attrs.get(VERSION, version)
    if methods:
        root[METHOD] = _METHOD_SEP.join(methods)
        root[REFERENCES] = " ".join(references)
        root[VERSION] = str(version)
    # xradar appends to the history, so the key has to exist
    root[HISTORY] = "\n".join(lines)


# ---------------------------------------------------------------------------
# CfRadial1, CfRadial2, ODIM_H5
# ---------------------------------------------------------------------------
def _harmonize_fields(tree):
    """
    Give every sweep of ``tree`` the same non-scalar variables, in place.

    The CfRadial1 writer of xradar stacks the sweeps along ``time``. It fails
    with a merge error when the sweeps hold different variables, as the
    surveillance and the Doppler cuts of a NEXRAD volume do (and the derived
    fields radarx computes for some sweeps only). A variable that a sweep
    lacks is added as NaN (single precision, lazy when the other fields are
    dask arrays), with the attributes it has in the sweeps that hold it.
    """
    nodes = [tree[name] for name in tree.match("sweep_*")]
    sweeps = [node.to_dataset(inherit=False) for node in nodes]
    known = {}
    for ds in sweeps:
        for name, var in ds.data_vars.items():
            if var.ndim:
                known.setdefault(name, var)
    for node, ds in zip(nodes, sweeps):
        arrays = [v for v in ds.data_vars.values() if v.ndim]
        if not arrays:
            continue
        reference = max(arrays, key=lambda v: v.ndim)
        for name, var in known.items():
            if name in ds.data_vars or not set(var.dims) <= set(reference.dims):
                continue
            extra = {d: 0 for d in reference.dims if d not in var.dims}
            empty = xr.full_like(reference.isel(extra, drop=True), np.nan, "float32")
            empty.attrs = dict(var.attrs)
            node[name] = empty


def _common_variable_attrs(tree):
    """
    Keep, for each variable of several sweeps, only the attributes all sweeps agree on.

    The CfRadial1 writer of xradar stacks the sweeps along ``time`` and merges the
    attributes of each variable; it stops when a key has different values in
    different sweeps (for example the ``phase_noise`` that ``estimate_kdp`` records
    per sweep). CfRadial1 has one set of attributes per variable, so a value that
    differs between the sweeps cannot be written, and it is dropped.
    """
    nodes = [tree[name] for name in tree.match("sweep_*")]
    seen = {}
    for node in nodes:
        for name, var in node.to_dataset(inherit=False).data_vars.items():
            seen.setdefault(name, []).append(var.attrs)
    for node in nodes:
        for name in node.to_dataset(inherit=False).data_vars:
            attrs_list = seen[name]
            if len(attrs_list) < 2:
                continue
            first = attrs_list[0]
            common = {
                key
                for key in first
                if all(
                    key in other and _same(first[key], other[key])
                    for other in attrs_list[1:]
                )
            }
            node[name].attrs = {
                k: v for k, v in node[name].attrs.items() if k in common
            }


def _same(a, b):
    """Whether two attribute values are equal, for arrays too."""
    try:
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    except (TypeError, ValueError):
        return a == b


def write_cfradial1(dtree, path, *, calibs=True, packing="drop"):
    """
    Write a radar volume as a CfRadial1 NetCDF file.

    The file is written by :func:`xradar.io.to_cfradial1`, which stacks the
    sweeps along one ``time`` dimension (padded to the longest range), as the
    format requires. radarx prepares a copy of ``dtree`` first (see
    :mod:`radarx.io.export`).

    CfRadial1 cannot hold everything of a radarx volume:

    * It has no sweep groups, so the attributes of a sweep are lost. The
      provenance (``radarx_method``, ``radarx_references``, ``radarx_version``
      and ``history``) of all sweeps is merged onto the global attributes
      first, so that :func:`radarx.cite` still finds it. The attributes of
      the variables are written as they are.
    * The georeferenced ``x``, ``y`` and ``z`` are not written, and
      ``sweep_fixed_angle`` is called ``fixed_angle``.
    * Sweeps with different numbers of gates are padded with NaN by xradar.
      A field that only some sweeps hold (the Doppler cuts of a NEXRAD volume
      have no ``KDP``) is NaN in the others, and an integer field becomes
      floating point; xradar stops with a merge error when the sweeps hold
      different fields, so radarx adds the missing ones as NaN first.
    * The derived fields are written as ordinary variables of ``(time,
      range)``.
    * Booleans in attributes become the strings ``"true"`` and ``"false"``,
      ``None`` is dropped, and xradar appends ``": xradar vX CfRadial1
      export"`` to ``history``.
    * The read-back volume has a ``sweep`` dimension and is rebuilt by
      :func:`xradar.io.open_cfradial1_datatree`; it is equal to the input in
      the values of the fields but not identical in attributes and
      coordinates.

    This was checked by reading the files back; the CfRadial specification
    (https://ncar.github.io/CfRadial/) was not used to validate the output.

    Parameters
    ----------
    dtree : xarray.DataTree
        Radar volume with ``sweep_*`` nodes.
    path : str or os.PathLike
        File to write. An existing file is replaced.
    calibs : bool, optional
        Write the calibration parameters of ``radar_calibration``.
    packing : {"drop", "keep"}, optional
        ``"drop"`` (default) writes floating fields as floats. ``"keep"`` keeps
        the integer packing (``scale_factor``, ``add_offset``) that the reader
        of the original file set, which makes the file smaller but rounds the
        values and writes NaN as a number unless a ``_FillValue`` is set.

    Returns
    -------
    pathlib.Path
        The file.

    Raises
    ------
    TypeError, ValueError
        If ``dtree`` is not a DataTree with sweeps, or ``packing`` is unknown.

    See Also
    --------
    radarx.io.write_cfradial2, radarx.io.write_odim
    """
    _require_tree(dtree, "write_cfradial1")
    _check_packing(packing)
    writer = _xradar_writer("to_cfradial1")
    prepared = _prepare_tree(dtree, "netcdf", packing)
    _fold_provenance(prepared)
    # CfRadial1 has no sweep attributes; xradar 0.11 stops when they differ
    for node in prepared.subtree:
        if node is not prepared:
            node.attrs = {}
    # the "None" placeholders of the readers are no values; xradar 0.11 keeps them
    prepared.attrs = {k: v for k, v in prepared.attrs.items() if not _is_placeholder(v)}
    _harmonize_fields(prepared)
    _common_variable_attrs(prepared)
    writer(prepared, os.fspath(path), calibs=calibs)
    return _path(path)


def write_cfradial2(dtree, path, *, engine=None, packing="drop"):
    """
    Write a radar volume as a CfRadial2 NetCDF file.

    The file is written by :func:`xradar.io.to_cfradial2`: the root and every
    sweep are NetCDF groups. radarx prepares a copy of ``dtree`` first (see
    :mod:`radarx.io.export`), which xradar 0.11 also needs because its writer
    changes the tree it is given.

    xradar conforms every sweep to the CfRadial2 layout, which drops content
    (found by reading the files back):

    * the attributes of the sweeps. The provenance of the sweeps is merged onto
      the root attributes first, so that :func:`radarx.cite` still finds it;
      the attributes of the fields are kept;
    * the variables that have no ``range`` dimension, such as the per-ray
      ``PHIDP_OFFSET`` of :func:`radarx.retrieve.estimate_kdp`, and the
      georeferenced ``x``, ``y`` and ``z``;
    * the ray dimension: ``azimuth`` becomes ``time`` (the values are the
      same).

    Also lost in an exact round trip: booleans in attributes become the
    strings ``"true"`` and ``"false"``, ``None`` is dropped, the integer
    packing of a field is replaced by floats (``packing="drop"``), and newer
    xradar appends ``": xradar vX CfRadial2 export"`` to the root ``history``.
    Use :func:`write_zarr` when nothing may be lost.

    Parameters
    ----------
    dtree : xarray.DataTree
        Radar volume with ``sweep_*`` nodes.
    path : str or os.PathLike
        File to write. An existing file is replaced.
    engine : {"netcdf4", "h5netcdf"}, optional
        NetCDF engine. Default: the first that is installed.
    packing : {"drop", "keep"}, optional
        See :func:`write_cfradial1`.

    Returns
    -------
    pathlib.Path
        The file.

    See Also
    --------
    radarx.io.write_cfradial1, radarx.io.write_odim
    """
    _require_tree(dtree, "write_cfradial2")
    _check_packing(packing)
    writer = _xradar_writer("to_cfradial2")
    prepared = _prepare_tree(dtree, "netcdf", packing)
    _fold_provenance(prepared)
    # xradar 0.11 does not set these
    prepared.attrs = {**prepared.attrs, "Conventions": "Cf/Radial", "version": "2.0"}
    writer(prepared, os.fspath(path), engine=engine)
    return _path(path)


def _moments(ds):
    """Names of the variables on the ray by gate grid that ODIM_H5 stores."""
    dim0 = "elevation" if str(ds.get("sweep_mode", "")) == "rhi" else "azimuth"
    return [n for n, v in ds.data_vars.items() if {dim0, "range"} == set(v.dims)]


def _odim_tree(dtree, fields, packing):
    """The copy of the volume that the ODIM writer gets."""
    out = _prepare_tree(dtree, "netcdf", packing)
    groups = list(out.match("sweep_*"))
    for name in groups:
        node = out[name]
        for moment in _moments(node.to_dataset(inherit=False)):
            if fields is not None and moment not in fields:
                del node[moment]
                continue
            var = node[moment]
            if var.dtype.kind == "b":
                converted = var.astype("uint8")
                converted.attrs = dict(var.attrs)
                node[moment] = converted
            # xradar 0.11 reads the stored type from the encoding
            variable = node.variables[moment]
            variable.encoding.setdefault("dtype", variable.dtype)
    return out, groups


def _provenance_of(attrs):
    """The provenance attributes that have a value (readers write "None")."""
    return {
        k: attrs[k] for k in PROVENANCE if str(attrs.get(k, "")) not in ("", "None")
    }


def _write_how(parent, attrs):
    """Write attributes to ``parent/how`` (created when there is something)."""
    if not attrs:
        return
    group = parent.require_group("how")
    for key, value in attrs.items():
        group.attrs.create(key, np.bytes_(str(value).encode("utf-8")))


def _odim_provenance(path, tree, groups):
    """Add the provenance to the ``how`` groups of the ODIM_H5 file."""
    import h5py

    with h5py.File(os.fspath(path), "a") as h5:
        _write_how(h5, _provenance_of(tree.attrs))
        for index, name in enumerate(groups, start=1):
            node = tree[name]
            holder = h5[f"dataset{index}"]
            _write_how(holder, _provenance_of(node.attrs))
            for data in [g for g in holder.values() if "what" in g]:
                quantity = bytes(data["what"].attrs["quantity"]).decode("utf-8")
                if quantity in node.data_vars:
                    _write_how(data, _provenance_of(node[quantity].attrs))


def write_odim(
    dtree,
    path,
    *,
    source=None,
    fields=None,
    optional_how=True,
    compression="gzip",
    compression_opts=6,
    packing="keep",
):
    """
    Write a radar volume as an ODIM_H5 file.

    The file is written by :func:`xradar.io.to_odim`: one ``datasetN`` per
    sweep and one ``dataM`` with ``what/quantity``, ``gain``, ``offset``,
    ``nodata`` and ``undetect`` per field on the ray by gate grid. The
    provenance is added afterwards: ``radarx_method``, ``radarx_references``,
    ``radarx_version`` and ``history`` are written as attributes of
    ``/how``, ``/datasetN/how`` and ``/datasetN/dataM/how``. ODIM_H5 2.4.1
    (section 4.4) allows additional attributes in ``how`` groups; the
    placement in ``dataM/how`` follows the dataset-specific metadata groups of
    the specification and was not checked with another ODIM reader.

    ODIM_H5 cannot hold everything of a radarx volume:

    * ``what/quantity`` is the name of the variable. Table 16 of ODIM_H5
      2.4.1 lists the standard quantities (``DBZH``, ``ZDR``, ``RHOHV``,
      ``PHIDP``, ``KDP``, ``VRADH``, ...). Names radarx derives that are not
      in the table (``PHIDP_processed``, ``ECHO_CLASS``, ``HID``,
      ``METEO_SCORE``) are written under their own name, which readers other
      than xradar may skip or reject; ``fields`` writes a subset.
    * Only variables on the ``(azimuth, range)`` (``(elevation, range)`` for
      RHI) grid are stored. Per-ray fields (``PHIDP_OFFSET``), scalars,
      strings, the georeferenced ``x``, ``y``, ``z`` and the calibration are
      lost.
    * The attributes of the variables (``units``, ``long_name``,
      ``standard_name``) and of the sweeps and the root are lost, except the
      provenance and the radar identifiers of ``source``.
    * Booleans (``METEO_MASK``) are written as ``uint8`` and read back as
      numbers.
    * Fields are stored in the type of their encoding. Floats without an
      encoding are stored as floats with ``nodata`` and ``undetect`` set to
      the largest value of the type (infinity for floats), which the xradar
      reader turns back into NaN; a float field cannot say "no echo" and
      "never measured" apart. For an integer field without an encoding
      (``ECHO_CLASS``, ``HID``) the largest value of the type (127 for
      ``int8``) is ``nodata``: a class code of that value would read back as
      NaN, and the field comes back as floats.
    * With ``optional_how=True`` (the default here) the angles and times of
      every ray are written, and xradar reads them back; without them the
      reader rebuilds a regular azimuth grid and the exact ray angles and
      times are lost.

    The fields read back as floats scaled with ``gain`` and ``offset``. Because
    of this a read-back volume is equal in the values of the fields, but not
    identical.

    Parameters
    ----------
    dtree : xarray.DataTree
        Radar volume with ``sweep_*`` nodes, the root variables
        ``time_coverage_start``, ``time_coverage_end``, ``longitude``,
        ``latitude`` and ``altitude`` and per sweep ``sweep_fixed_angle`` and
        ``sweep_mode``.
    path : str or os.PathLike
        File to write. An existing file is replaced.
    source : str, optional
        Radar identifiers for ``what/source``, as ``TYP:VALUE`` pairs separated
        by commas (ODIM_H5 2.4.1, Table 3), for example ``"NOD:fianj"``. The
        specification makes ``NOD`` mandatory for single-site data; xradar
        accepts ``NOD``, ``RAD`` or ``WMO``. Newer xradar versions build the
        string from the root attributes of a volume that was read from ODIM
        when it is not given; xradar 0.11 needs it.
    fields : sequence of str, optional
        Names of the fields to write. Default: all fields on the grid.
    optional_how : bool, optional
        Write the optional ray angles and times to ``how``. Default True.
    compression, compression_opts : optional
        HDF5 compression of the data, ``"gzip"`` and 6 by default.
    packing : {"keep", "drop"}, optional
        ``"keep"`` (default) writes a field with an integer packing in its
        encoding as integers with ``gain`` and ``offset``, as ODIM_H5 volumes
        usually are, ``"drop"`` writes floats.

    Returns
    -------
    pathlib.Path
        The file.

    Raises
    ------
    ValueError
        If the volume has no radar identifier for ``source``, or a name in
        ``fields`` is not a field on the grid of any sweep.

    See Also
    --------
    radarx.io.write_cfradial1, radarx.io.write_cfradial2
    """
    _require_tree(dtree, "write_odim")
    _check_packing(packing)
    writer = _xradar_writer("to_odim")
    if fields is not None:
        fields = list(fields)
        known = {
            m
            for name in dtree.match("sweep_*")
            for m in _moments(dtree[name].to_dataset(inherit=False))
        }
        unknown = [f for f in fields if f not in known]
        if unknown:
            raise ValueError(
                f"{unknown} are not fields on the ray by gate grid of the sweeps"
            )
    prepared, groups = _odim_tree(dtree, fields, packing)
    writer(
        prepared,
        os.fspath(path),
        source=source,
        optional_how=optional_how,
        compression=compression,
        compression_opts=compression_opts,
    )
    _odim_provenance(path, prepared, groups)
    return _path(path)


# ---------------------------------------------------------------------------
# grids: CF georeferencing
# ---------------------------------------------------------------------------
_CRS_NAMES = CRS_NAMES


def _find_crs(ds, crs):
    """The grid mapping attributes of the grid, from ``crs`` or the dataset."""
    if crs is False:
        return None
    if crs is not None:
        return user_crs(crs).to_cf()
    for name in _CRS_NAMES:
        if name in ds.variables and "crs_wkt" in ds[name].attrs:
            return dict(ds[name].attrs)
    for var in ds.data_vars.values():
        name = var.attrs.get("grid_mapping") or var.encoding.get("grid_mapping")
        if name in ds.variables:
            return dict(ds[name].attrs)
    if "latitude" in ds.variables and "longitude" in ds.variables:
        if ds["latitude"].ndim == 0 and ds["longitude"].ndim == 0:
            return aeqd_crs(float(ds["latitude"]), float(ds["longitude"])).to_cf()
    return None


def _edges(values):
    """The cell edges of a 1-D axis (the midpoints, extended at both ends)."""
    values = np.asarray(values, "f8")
    mid = 0.5 * (values[1:] + values[:-1])
    first = values[0] - (mid[0] - values[0])
    last = values[-1] + (values[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def _bounds(ds, name):
    """Add ``<name>_bnds`` for a 1-D axis of more than one point."""
    values = ds[name].values
    edges = _edges(values)
    ds[f"{name}_bnds"] = xr.DataArray(
        np.stack([edges[:-1], edges[1:]], axis=1),
        dims=(name, "bnds"),
        attrs={"long_name": f"cell bounds of {name}"},
    )
    ds[f"{name}_bnds"].encoding["_FillValue"] = None
    ds[name].attrs["bounds"] = f"{name}_bnds"


def _cf_grid(ds, crs, bounds, lonlat):
    """Copy of a gridded Dataset with CF georeferencing."""
    if "x" not in ds.dims or "y" not in ds.dims:
        raise ValueError("write_netcdf_cf needs a grid with x and y dimensions")
    out = _prepare_dataset(ds, "netcdf", "drop")
    mapping = _find_crs(ds, crs)
    out = out.drop_vars([n for n in _CRS_NAMES if n in out.variables])
    out = georeference(
        out,
        mapping,
        name="crs",
        recompute=crs is not None,
        lonlat_=lonlat,
    )
    if mapping is not None:
        out = out.reset_coords("crs")
    if bounds:
        for name in ("x", "y", "z"):
            if name in out.dims and out.sizes[name] > 1 and out[name].ndim == 1:
                _bounds(out, name)
    # the readers write "None" for the attributes they do not know
    out.attrs = {k: v for k, v in out.attrs.items() if not _is_placeholder(v)}
    if HISTORY in out.attrs:
        lines = str(out.attrs[HISTORY]).split("\n")
        out.attrs[HISTORY] = "\n".join(line for line in lines if line != "None")
    conventions = str(out.attrs.get("Conventions", ""))
    if "CF-" not in conventions:
        out.attrs["Conventions"] = f"{conventions} CF-1.8".strip()
    return out


def write_netcdf_cf(grid, path, *, crs=None, bounds=True, lonlat=True, **kwargs):
    """
    Write a gridded result as a CF-compliant NetCDF file.

    For the Datasets of :func:`radarx.grid.grid_cones`,
    :func:`radarx.grid.grid_radars`, :func:`radarx.retrieve.create_cappi` and
    the products of the retrieve functions that are on an ``x``, ``y`` grid.
    The object is not changed; a copy is prepared (see
    :mod:`radarx.io.export`) and gets:

    * a ``crs`` variable with the grid mapping attributes (``grid_mapping_name``,
      ``crs_wkt``, ...) taken from the ``crs_wkt`` of the grid, and
      ``grid_mapping = "crs"`` on every field with ``x`` and ``y``. The
      ``crs_wkt`` of radarx grids is a scalar coordinate, which would be listed
      in the ``coordinates`` attribute of every field; a data variable is the
      CF convention;
    * ``standard_name`` (``projection_x_coordinate``,
      ``projection_y_coordinate``, ``altitude`` for ``z`` in metres above sea
      level, ``time``), ``units``, ``axis`` and ``positive`` on the
      coordinates, where the grid has none (the grids of radarx already have
      them, see :mod:`radarx._gridcf`);
    * 2-D ``lat(y, x)`` and ``lon(y, x)`` from the CRS when the grid has none
      or 1-D ``lat(y)`` and ``lon(x)``, which are wrong away from the origin of
      an azimuthal equidistant grid (about 4 km at 230 km range, issue #265),
      or when ``crs`` is given;
    * ``x_bnds``, ``y_bnds`` and ``z_bnds`` (``bounds=True``);
    * ``Conventions = "CF-1.8"`` (added to the ones the grid has) and the
      NetCDF-safe attributes of :mod:`radarx.io.export`, without the ``"None"``
      placeholders that readers write. The ``radarx_*`` provenance and ``history`` are
      kept on the Dataset and on the fields.

    A grid without a CRS (``create_cappi``) gets one from its scalar
    ``latitude`` and ``longitude`` coordinates, as an azimuthal equidistant
    projection centred there. That is how xradar georeferences the sweeps,
    but it is an assumption; pass ``crs=`` to state the CRS or ``crs=False`` for
    none. Without a CRS no ``grid_mapping`` is written (one the fields carry is
    removed) and ``lat`` and ``lon`` stay as they are.

    The values of the fields are written as they are. Reading the file back
    with ``xarray.open_dataset`` gives the same fields, attributes and
    coordinates; the added ``crs`` and bounds variables are new, and 1-D
    ``lat`` and ``lon`` are 2-D. The files of a ``grid_cones`` grid and of a
    CAPPI pass the CF checker 4.1.0 against CF-1.8 without errors; it warns
    when a field has neither ``standard_name`` nor ``long_name``, which
    radarx does not invent.

    Parameters
    ----------
    grid : xarray.Dataset or xarray.DataArray
        Gridded result with ``x`` and ``y`` dimensions.
    path : str or os.PathLike
        File to write. An existing file is replaced.
    crs : pyproj.CRS, str, tuple or False, optional
        The CRS of ``x`` and ``y``: anything ``pyproj.CRS.from_user_input``
        takes, a ``(latitude, longitude)`` tuple for an azimuthal equidistant
        grid centred there, or ``False`` for no CRS. Default: the grid's own.
    bounds : bool, optional
        Write the cell bounds of ``x``, ``y`` and ``z``. Default True.
    lonlat : bool, optional
        Write 2-D ``lat`` and ``lon`` computed from the CRS where the grid has
        none or 1-D ones. Default True.
    **kwargs
        Passed to :meth:`xarray.Dataset.to_netcdf` (``engine``,
        ``encoding``, ...).

    Returns
    -------
    pathlib.Path
        The file.

    Raises
    ------
    ValueError
        If the grid has no ``x`` and ``y`` dimensions.

    See Also
    --------
    radarx.io.write_zarr
    """
    ds = _as_dataset(grid)
    if not isinstance(ds, xr.Dataset):
        raise TypeError("write_netcdf_cf writes a Dataset or a DataArray")
    out = _cf_grid(ds, crs, bounds, lonlat)
    out.to_netcdf(os.fspath(path), **kwargs)
    return _path(path)


# ---------------------------------------------------------------------------
# Zarr
# ---------------------------------------------------------------------------
def write_zarr(obj, path, *, cf=False, packing="drop", **kwargs):
    """
    Write a radarx result to a Zarr store.

    Works for a DataTree volume (every sweep is a group), a Dataset and a
    DataArray, and keeps dask arrays lazy: they are written chunk by chunk.
    The store is written by :meth:`xarray.Dataset.to_zarr` or
    :meth:`xarray.DataTree.to_zarr` from a prepared copy (see
    :mod:`radarx.io.export`). Zarr keeps booleans, ``None`` and lists in
    attributes, so the volume reads back identical, with the provenance on the
    nodes and the variables, and the integer packing of the source encoding
    replaced by floats (``packing="drop"``).

    The zarr package version 3 needs xarray 2025.1 or newer; with xarray
    2024.10 install ``zarr<3`` (writing failed with zarr 3 there, also with
    ``zarr_format=2``).

    Parameters
    ----------
    obj : xarray.DataTree, xarray.Dataset or xarray.DataArray
        Result to write.
    path : str or os.PathLike
        Store to write. An existing store is replaced (``mode="w"``).
    cf : bool, optional
        For a gridded Dataset: add the CF georeferencing of
        :func:`write_netcdf_cf`. Default False.
    packing : {"drop", "keep"}, optional
        See :func:`write_cfradial1`.
    **kwargs
        Passed to ``to_zarr`` (``zarr_format``, ``consolidated``,
        ``encoding``, ...).

    Returns
    -------
    pathlib.Path
        The store.

    See Also
    --------
    radarx.io.write_netcdf_cf
    """
    _check_packing(packing)
    kwargs.setdefault("mode", "w")
    if isinstance(obj, DataTree):
        prepared = _prepare_tree(obj, "zarr", packing)
    else:
        ds = _as_dataset(obj)
        if cf:
            prepared = _cf_grid(ds, None, True, True)
            prepared.attrs = _clean_attrs(prepared.attrs, "zarr")
        else:
            prepared = _prepare_dataset(ds, "zarr", packing)
    prepared.to_zarr(os.fspath(path), **kwargs)
    return _path(path)


# ---------------------------------------------------------------------------
# accessor methods
# ---------------------------------------------------------------------------
@accessor_method("datatree", name="to_cfradial1")
def _to_cfradial1(self, path, **kwargs):
    """
    Write the volume as a CfRadial1 file.

    See :func:`radarx.io.write_cfradial1` for the parameters and for what the
    format cannot hold.

    Returns
    -------
    pathlib.Path
        The file.
    """
    return write_cfradial1(self.xarray_obj, path, **kwargs)


@accessor_method("datatree", name="to_cfradial2")
def _to_cfradial2(self, path, **kwargs):
    """
    Write the volume as a CfRadial2 file.

    See :func:`radarx.io.write_cfradial2`.

    Returns
    -------
    pathlib.Path
        The file.
    """
    return write_cfradial2(self.xarray_obj, path, **kwargs)


@accessor_method("datatree", name="to_odim")
def _to_odim(self, path, **kwargs):
    """
    Write the volume as an ODIM_H5 file.

    See :func:`radarx.io.write_odim` for the parameters and for what the
    format cannot hold.

    Returns
    -------
    pathlib.Path
        The file.
    """
    return write_odim(self.xarray_obj, path, **kwargs)


@accessor_method("dataarray", "dataset", "datatree", name="to_zarr")
def _to_zarr(self, path, **kwargs):
    """
    Write the object to a Zarr store.

    See :func:`radarx.io.write_zarr`.

    Returns
    -------
    pathlib.Path
        The store.
    """
    return write_zarr(self.xarray_obj, path, **kwargs)


@accessor_method("dataarray", "dataset", name="to_netcdf_cf")
def _to_netcdf_cf(self, path, **kwargs):
    """
    Write the grid as a CF-compliant NetCDF file.

    See :func:`radarx.io.write_netcdf_cf`.

    Returns
    -------
    pathlib.Path
        The file.
    """
    return write_netcdf_cf(self.xarray_obj, path, **kwargs)
