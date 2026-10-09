#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Polar patches and normalisation for machine-learning models.

Convolutional networks are trained and run on fixed-size tiles. A PPI sweep
is cut into patches of ``n_azimuth`` rays by ``n_range`` gates directly in
radar (polar) coordinates, so no information is lost to interpolation onto a
Cartesian grid. Along azimuth the sweep is periodic: patches that run past
the last ray continue at the first one, so features crossing north are seen
whole and the patch grid needs no padding in that direction. Along range,
patches are laid out from the first gate and the last patch ends at the last
gate; with ``fill_value`` a patch larger than the sweep is padded.

Model outputs on the patches are put back together with
:func:`reassemble`, which averages the overlapping patches with a window
that falls off towards the patch edges (cosine or linear), as is usual for
tiled inference: the predictions near a tile border, where the network sees
the least context, get the smallest weight. Both windows are evaluated at
pixel centres, ``w(i) = sin^2(pi (i + 1/2) / n)`` (cosine) and
``w(i) = 1 - |2 (i + 1/2) / n - 1|`` (linear), so they never vanish and sum
to one for patches overlapping by half their size. Where all patches agree
on a gate, that value is returned as is, so reassembling unchanged patches
gives back the input bit for bit.

A compiled C++ kernel extracts and reassembles the patches of all sweeps of
a volume in one multithreaded call; an equivalent NumPy implementation is
used when the kernel is not available (and as the test oracle).
"""

from __future__ import annotations

__all__ = ["PatchIndex", "polar_patches", "reassemble", "normalize", "denormalize"]

from dataclasses import dataclass, field

import numpy as np
import xarray as xr

try:
    from . import _patches

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _patches = None
    HAS_COMPILED_KERNEL = False

BLENDS = ("cosine", "linear", "mean")
_NORMALIZATION_ATTRS = (
    "ml_normalization",
    "ml_offset",
    "ml_scale",
    "ml_units",
    "ml_standard_name",
)


def _use_compiled(engine):
    """Whether to run the compiled kernel for the requested ``engine``."""
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled patch kernel is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


@dataclass
class PatchIndex:
    """
    Where every patch of :func:`polar_patches` came from.

    Attributes
    ----------
    table : numpy.ndarray
        ``(n_patches, 3)`` integers: sweep number, first ray and first gate of
        each patch.
    size : tuple of int
        Patch size ``(n_azimuth, n_range)``.
    shapes : list of tuple of int
        ``(n_rays, n_gates)`` of each sweep.
    wrap_azimuth : bool
        Whether patches wrap around from the last ray to the first.
    lead_dims : tuple of str
        Dimensions between the patch axis and the two patch dimensions (for a
        Dataset, ``"variable"`` holds the stacked fields).
    lead_shape : tuple of int
        Sizes of ``lead_dims``.
    lead_coords : dict
        Coordinates of ``lead_dims`` that have one (for example the variable
        names).
    dims : tuple of str
        Names of the azimuth and range dimensions.
    templates : list of xarray.Dataset or None
        Per sweep, the coordinates on ``dims`` given back to the reassembled
        result (``None`` for NumPy input).
    paths : list of str or None
        DataTree paths of the sweeps (``None`` unless the input was a DataTree).
    kind : str
        Type of the input: ``"ndarray"``, ``"dataarray"``, ``"dataset"`` or
        ``"datatree"``.
    name : str or None
        Name of an input DataArray.
    root : xarray.Dataset or None
        Root dataset of an input DataTree.
    """

    table: np.ndarray
    size: tuple
    shapes: list
    wrap_azimuth: bool = True
    lead_dims: tuple = ()
    lead_shape: tuple = ()
    lead_coords: dict = field(default_factory=dict)
    dims: tuple = ("azimuth", "range")
    templates: list | None = None
    paths: list | None = None
    kind: str = "ndarray"
    name: str | None = None
    root: xr.Dataset | None = None

    def __len__(self):
        return len(self.table)

    @property
    def shape(self):
        """``(n_rays, n_gates)`` of the sweep (single-sweep input only)."""
        if len(self.shapes) != 1:
            raise ValueError("the index holds several sweeps; use .shapes")
        return self.shapes[0]


# --------------------------------------------------------------------------
# patch layout and blending windows
# --------------------------------------------------------------------------


def _starts(n, size, stride, wrap):
    """First ray (gate) of every patch along one dimension of length ``n``."""
    if wrap:
        return np.arange(0, n, stride, dtype=np.int64)
    if n <= size:
        return np.zeros(1, dtype=np.int64)
    starts = np.arange(0, n - size + 1, stride, dtype=np.int64)
    if starts[-1] + size < n:  # the last patch ends at the last gate
        starts = np.append(starts, n - size)
    return starts


def _window(n, blend):
    """Blending weights of ``n`` pixels (positive everywhere)."""
    x = (np.arange(n) + 0.5) / n
    if blend == "cosine":
        return np.sin(np.pi * x) ** 2
    if blend == "linear":
        return 1.0 - np.abs(2.0 * x - 1.0)
    return np.ones(n)


def _pair(value, name):
    if np.ndim(value) == 0:
        value = (value, value)
    value = tuple(int(v) for v in value)
    if len(value) != 2 or min(value) < 1:
        raise ValueError(f"{name} must be a positive int or a pair of them")
    return value


# --------------------------------------------------------------------------
# NumPy reference implementation of the kernel
# --------------------------------------------------------------------------


def _extract_numpy(data, table, h, w, wrap, fill):
    """Patches ``(n, L, h, w)`` of the sweeps ``data[k]`` of shape ``(L, A, R)``."""
    L = data[0].shape[0]
    out = np.full((len(table), L, h, w), fill, dtype=data[0].dtype)
    for k, sweep in enumerate(data):
        sel = np.flatnonzero(table[:, 0] == k)
        if not sel.size:
            continue  # pragma: no cover
        nray, ngate = sweep.shape[1:]
        rays = table[sel, 1, None] + np.arange(h)
        gates = table[sel, 2, None] + np.arange(w)
        if wrap:
            rays %= nray
        ok = (rays >= 0) & (rays < nray)
        ok = ok[:, :, None] & ((gates >= 0) & (gates < ngate))[:, None, :]
        values = sweep[
            :,
            np.clip(rays, 0, nray - 1)[:, :, None],
            np.clip(gates, 0, ngate - 1)[:, None, :],
        ]  # (L, n, h, w)
        out[sel] = np.where(ok, values, fill).transpose(1, 0, 2, 3)
    return out


def _reassemble_numpy(patches, table, shapes, wa, wr, wrap):
    """Weighted mean of the overlapping patches (scattered patch by patch)."""
    n, L, h, w = patches.shape
    weight = wa[:, None] * wr[None, :]
    results = []
    for k, (nray, ngate) in enumerate(shapes):
        num = np.zeros((L, nray, ngate))
        den = np.zeros((L, nray, ngate))
        lo = np.full((L, nray, ngate), np.inf)
        hi = np.full((L, nray, ngate), -np.inf)
        for p in np.flatnonzero(table[:, 0] == k):
            _, a0, r0 = table[p]
            j0, j1 = max(0, -r0), min(w, ngate - r0)
            if j1 <= j0:
                continue  # pragma: no cover
            rays = a0 + np.arange(h)
            if wrap:
                rays %= nray
            rows = np.flatnonzero((rays >= 0) & (rays < nray))
            # chunks of at most nray rows hold no ray twice
            for c in range(0, rows.size, nray):
                i = rows[c : c + nray]
                values = patches[p][:, i, j0:j1].astype(np.float64)
                ok = ~np.isnan(values)
                wt = np.where(ok, weight[i, j0:j1], 0.0)
                dst = (slice(None), rays[i], slice(r0 + j0, r0 + j1))
                num[dst] += np.where(ok, wt * values, 0.0)
                den[dst] += wt
                lo[dst] = np.minimum(lo[dst], np.where(ok, values, np.inf))
                hi[dst] = np.maximum(hi[dst], np.where(ok, values, -np.inf))
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
        # equal values everywhere a gate is covered: that value, exactly
        out = np.where((den > 0) & (lo == hi), lo, out)
        results.append(out.astype(patches.dtype))
    return results


# --------------------------------------------------------------------------
# xarray layer
# --------------------------------------------------------------------------


def _template(obj, dims):
    """Coordinates of ``obj`` that only depend on the sweep dimensions."""
    return xr.Dataset(
        coords={
            name: coord.variable
            for name, coord in obj.coords.items()
            if set(coord.dims) <= set(dims)
        }
    )


def _sweep_array(ds, variables, dims):
    """Fields of a sweep stacked on a ``variable`` dimension, sweep dims last."""
    da = ds[list(variables)].to_array("variable")
    return da.transpose(..., *dims)


def _has_sweep_dims(obj, dims):
    return all(d in obj.dims for d in dims)


def _split(obj, variables, dims):
    """Arrays ``(lead..., A, R)`` of every sweep, plus the PatchIndex fields."""
    info = {"dims": tuple(dims), "templates": None, "paths": None, "name": None}
    if isinstance(obj, xr.DataTree):
        sweeps, paths, templates = [], [], []
        for node in obj.subtree:
            if not node.has_data or not _has_sweep_dims(node, dims):
                continue
            ds = node.to_dataset(inherit=False)
            names = variables
            if names is None:
                names = [
                    v for v, da in ds.data_vars.items() if _has_sweep_dims(da, dims)
                ]
                if not names:
                    continue
                variables = names  # later sweeps need the fields of the first
            if any(v not in ds.data_vars for v in names):
                continue  # sweeps lacking a field are skipped
            sweeps.append(_sweep_array(ds, names, dims))
            paths.append(node.path)
            templates.append(_template(ds, dims))
        if not sweeps:
            raise ValueError(f"no sweep of the DataTree holds the fields {variables}")
        root = obj.to_dataset(inherit=False)
        info.update(kind="datatree", paths=paths, templates=templates, root=root)
    elif isinstance(obj, xr.Dataset):
        if variables is None:
            variables = [
                v for v, da in obj.data_vars.items() if _has_sweep_dims(da, dims)
            ]
        if not variables:
            raise ValueError(f"the Dataset has no field with dimensions {dims}")
        sweeps = [_sweep_array(obj, variables, dims)]
        info.update(kind="dataset", templates=[_template(obj, dims)])
    elif isinstance(obj, xr.DataArray):
        if not _has_sweep_dims(obj, dims):
            raise ValueError(
                f"the DataArray needs the dimensions {dims}, has {obj.dims}"
            )
        sweeps = [obj.transpose(..., *dims)]
        info.update(kind="dataarray", templates=[_template(obj, dims)], name=obj.name)
    else:
        arr = np.asarray(obj)
        if arr.ndim < 2:
            raise ValueError(
                "a NumPy array needs at least two dimensions (rays, gates)"
            )
        lead = tuple(f"dim_{i}" for i in range(arr.ndim - 2))
        sweeps = [xr.DataArray(arr, dims=lead + tuple(dims))]
        info.update(kind="ndarray")

    lead_dims = sweeps[0].dims[:-2]
    lead_coords = {d: sweeps[0][d].values for d in lead_dims if d in sweeps[0].coords}
    for s in sweeps[1:]:
        if s.dims[:-2] != lead_dims or s.shape[:-2] != sweeps[0].shape[:-2]:
            raise ValueError("all sweeps need the same fields and leading dimensions")
    info.update(
        lead_dims=lead_dims, lead_shape=sweeps[0].shape[:-2], lead_coords=lead_coords
    )
    return [s.values for s in sweeps], info


def polar_patches(
    obj,
    size=(64, 64),
    stride=None,
    wrap_azimuth=True,
    *,
    variables=None,
    dims=("azimuth", "range"),
    fill_value=np.nan,
    engine="auto",
    n_threads=None,
):
    """
    Cut sweeps into fixed-size polar patches for a machine-learning model.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset, xarray.DataTree or numpy.ndarray
        A sweep (DataArray or Dataset with ``azimuth`` and ``range``
        dimensions), a volume (DataTree; all its sweeps are cut in one call) or
        an array whose last two axes are rays and gates. Rays must be in
        azimuth order (as recorded, the first ray may be at any azimuth). For
        a Dataset or DataTree, the fields are stacked on a ``variable``
        dimension (the model's channels).
    size : int or tuple of int, default (64, 64)
        Patch size ``(n_azimuth, n_range)`` in rays and gates.
    stride : int or tuple of int, optional
        Step between patches in rays and gates. Defaults to half the patch
        size, the overlap the blending windows of :func:`reassemble` are made
        for; ``stride=size`` gives patches that do not overlap.
    wrap_azimuth : bool, default True
        Let patches continue from the last ray to the first (PPI sweeps). Use
        ``False`` for sectors and RHIs.
    variables : list of str, optional
        Fields of a Dataset or DataTree (default: every field on ``dims``;
        for a DataTree, those of its first sweep). Sweeps lacking one of them
        are skipped.
    dims : tuple of str, default ("azimuth", "range")
        Names of the ray and gate dimensions.
    fill_value : float, default NaN
        Value of patch pixels outside the sweep (only when a patch is larger
        than the sweep, or along azimuth without wrapping).
    engine : {"auto", "compiled", "numpy"}, default "auto"
        Use the compiled kernel when available, or force one implementation.
    n_threads : int, optional
        Threads of the compiled kernel (default: all cores).

    Returns
    -------
    patches : numpy.ndarray
        ``(n_patches, *lead, n_azimuth, n_range)``; ``lead`` are the other
        dimensions of the input (``variable`` for a Dataset). float64 data
        stays float64, everything else is float32.
    index : PatchIndex
        Where each patch came from; pass it to :func:`reassemble`.

    Examples
    --------
    >>> patches, index = polar_patches(sweep[["DBZH", "ZDR"]], size=(64, 128))
    >>> prediction = model.run({"x": patches})["y"]
    >>> result = reassemble(prediction, index)
    """
    h, w = _pair(size, "size")
    sh, sw = _pair(
        stride if stride is not None else (max(1, h // 2), max(1, w // 2)), "stride"
    )
    arrays, info = _split(obj, variables, dims)
    dtype = np.float64 if all(a.dtype == np.float64 for a in arrays) else np.float32
    lead_shape = arrays[0].shape[:-2]
    data = [
        np.ascontiguousarray(a, dtype=dtype).reshape((-1,) + a.shape[-2:])
        for a in arrays
    ]
    shapes = [tuple(int(n) for n in a.shape[-2:]) for a in data]

    rows = []
    for k, (nray, ngate) in enumerate(shapes):
        ra = _starts(nray, h, sh, wrap_azimuth)
        rg = _starts(ngate, w, sw, False)
        aa, gg = np.meshgrid(ra, rg, indexing="ij")
        rows.append(np.column_stack([np.full(aa.size, k), aa.ravel(), gg.ravel()]))
    table = np.ascontiguousarray(np.concatenate(rows), dtype=np.int64)

    if _use_compiled(engine):
        kernel = getattr(_patches, f"extract_{np.dtype(dtype).name}")
        patches = kernel(
            data,
            table,
            h,
            w,
            bool(wrap_azimuth),
            float(fill_value),
            int(n_threads or 0),
        )
    else:
        patches = _extract_numpy(data, table, h, w, bool(wrap_azimuth), fill_value)
    patches = patches.reshape((len(table),) + lead_shape + (h, w))
    index = PatchIndex(
        table=table,
        size=(h, w),
        shapes=shapes,
        wrap_azimuth=bool(wrap_azimuth),
        **info,
    )
    return patches, index


def _wrap_result(arr, index, k, name):
    """DataArray of sweep ``k`` with the input's coordinates."""
    dims = index.dims
    lead_shape = arr.shape[:-2]
    coords = dict(index.templates[k].coords)
    if lead_shape == index.lead_shape:
        lead_dims = index.lead_dims
        coords.update(index.lead_coords)
    elif len(lead_shape) == 1:
        lead_dims = ("channel",)
    else:
        lead_dims = tuple(f"channel_{i}" for i in range(len(lead_shape)))
    return xr.DataArray(arr, dims=lead_dims + tuple(dims), coords=coords, name=name)


def _as_dataset(da, index, name):
    """Dataset of a reassembled sweep: the input fields again, if they match."""
    if "variable" in da.dims and da.shape[:-2] == index.lead_shape:
        names = [str(v) for v in index.lead_coords["variable"]]
        return xr.Dataset(
            {v: da.isel(variable=i, drop=True) for i, v in enumerate(names)}
        ).drop_vars("variable", errors="ignore")
    return da.rename(name).to_dataset()


def reassemble(
    patches,
    index,
    shape=None,
    blend="cosine",
    *,
    name=None,
    attrs=None,
    engine="auto",
    n_threads=None,
):
    """
    Put patches (or model outputs on them) back together into sweeps.

    Every gate is the weighted mean of the patch pixels covering it, with a
    separable blending window over each patch. NaN pixels are skipped; a gate
    no finite pixel covers is NaN. Unchanged patches give back the input.

    Parameters
    ----------
    patches : numpy.ndarray
        ``(n_patches, *lead, n_azimuth, n_range)``, in the order of
        :func:`polar_patches`. ``lead`` may differ from the input (a model
        with other output channels than input channels).
    index : PatchIndex
        The index returned by :func:`polar_patches`.
    shape : tuple of int, optional
        ``(n_rays, n_gates)`` of the output; defaults to the sweep shape in
        ``index`` (single sweep only).
    blend : {"cosine", "linear", "mean"}, default "cosine"
        Blending window: squared sine (Hann), triangular, or uniform weights.
    name : str, optional
        Name of the result when it is not the input's fields
        (default ``"prediction"``).
    attrs : dict, optional
        Attributes for the result variables (for example
        :attr:`radarx.ml.Model.attrs`).
    engine : {"auto", "compiled", "numpy"}, default "auto"
        Use the compiled kernel when available, or force one implementation.
    n_threads : int, optional
        Threads of the compiled kernel (default: all cores).

    Returns
    -------
    numpy.ndarray, xarray.DataArray, xarray.Dataset or xarray.DataTree
        The same kind of object as the input of :func:`polar_patches`, with
        its coordinates. A Dataset (DataTree) gets its fields back when the
        patches have one channel per field; otherwise the result is one
        variable ``name`` with a ``channel`` dimension.
    """
    if blend not in BLENDS:
        raise ValueError(f"blend must be one of {BLENDS}, not {blend!r}")
    patches = np.asarray(patches)
    h, w = index.size
    if (
        patches.ndim < 3
        or patches.shape[0] != len(index)
        or patches.shape[-2:] != (h, w)
    ):
        raise ValueError(
            f"patches must be (n_patches={len(index)}, ..., {h}, {w}), not {patches.shape}"
        )
    shapes = index.shapes
    if shape is not None:
        if len(shapes) != 1:
            raise ValueError("shape can only be given for a single sweep")
        shapes = [tuple(int(n) for n in shape)]
        if index.kind != "ndarray" and shapes != index.shapes:
            raise ValueError("shape must match the sweep the patches came from")
    dtype = np.float64 if patches.dtype == np.float64 else np.float32
    lead_shape = patches.shape[1:-2]
    flat = np.ascontiguousarray(patches, dtype=dtype).reshape((len(index), -1, h, w))
    wa, wr = _window(h, blend), _window(w, blend)
    if _use_compiled(engine):
        kernel = getattr(_patches, f"reassemble_{np.dtype(dtype).name}")
        out = kernel(
            flat,
            index.table,
            shapes,
            wa,
            wr,
            bool(index.wrap_azimuth),
            int(n_threads or 0),
        )
    else:
        out = _reassemble_numpy(flat, index.table, shapes, wa, wr, index.wrap_azimuth)
    out = [o.reshape(lead_shape + o.shape[-2:]) for o in out]

    if index.kind == "ndarray":
        return out[0]
    name = name or ("prediction" if index.kind != "dataarray" else index.name)
    results = []
    for k, arr in enumerate(out):
        da = _wrap_result(arr, index, k, name)
        if attrs:
            da.attrs.update(attrs)
        results.append(da)
    if index.kind == "dataarray":
        return results[0]
    datasets = [_as_dataset(da, index, name) for da in results]
    if index.kind == "dataset":
        return datasets[0]
    from ..retrieve._products import product_tree

    root = xr.DataTree(index.root)
    return product_tree(root, dict(zip(index.paths, datasets)))


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------


def _statistics(values, method):
    with np.errstate(all="ignore"):
        if method == "zscore":
            return float(np.nanmean(values)), float(np.nanstd(values))
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        return lo, hi - lo


def _normalize_array(da, method, offset, scale, fill_value, clip):
    values = da.values if isinstance(da, xr.DataArray) else np.asarray(da)
    if offset is None or scale is None:
        if not np.isfinite(values).any():
            raise ValueError("cannot compute normalisation statistics of all-NaN data")
        o, s = _statistics(values, method)
        offset = o if offset is None else offset
        scale = s if scale is None else scale
    if not scale or not np.isfinite(scale):
        scale = 1.0  # constant field
    out = (da - offset) / scale
    if clip is not None:
        out = np.clip(out, *clip)
    if fill_value is not None:
        out = (
            out.fillna(fill_value)
            if isinstance(out, xr.DataArray)
            else np.where(np.isnan(out), fill_value, out)
        )
    if isinstance(out, xr.DataArray):
        attrs = {
            k: v for k, v in da.attrs.items() if k not in ("units", "standard_name")
        }
        attrs.update(
            ml_normalization=method,
            ml_offset=float(offset),
            ml_scale=float(scale),
        )
        if "units" in da.attrs:
            attrs["ml_units"] = da.attrs["units"]
        if "standard_name" in da.attrs:
            attrs["ml_standard_name"] = da.attrs["standard_name"]
        out.attrs = attrs
    return out


def _per_variable(value, name):
    return value.get(name) if isinstance(value, dict) else value


def normalize(
    obj, method="zscore", *, offset=None, scale=None, fill_value=None, clip=None
):
    """
    Scale fields for a model: ``(x - offset) / scale``.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or numpy.ndarray
        Fields to normalise (each variable of a Dataset on its own).
    method : {"zscore", "minmax"}, default "zscore"
        Statistics used when ``offset`` or ``scale`` is not given: the mean
        and standard deviation, or the minimum and range (to [0, 1]). NaN is
        ignored. Models are normally trained with fixed values; pass them.
    offset, scale : float or dict, optional
        Fixed offset and scale (per variable name for a Dataset).
    fill_value : float, optional
        Replace NaN (no echo, no data) after scaling, e.g. with 0.
    clip : tuple of float, optional
        Clip the scaled values to ``(low, high)``.

    Returns
    -------
    Same type as ``obj``. DataArrays get the attributes ``ml_normalization``,
    ``ml_offset`` and ``ml_scale`` (and ``ml_units``), which
    :func:`denormalize` uses to undo the scaling.
    """
    if method not in ("zscore", "minmax"):
        raise ValueError(f"method must be 'zscore' or 'minmax', not {method!r}")
    if isinstance(obj, xr.Dataset):
        return obj.assign(
            {
                v: _normalize_array(
                    da,
                    method,
                    _per_variable(offset, v),
                    _per_variable(scale, v),
                    fill_value,
                    clip,
                )
                for v, da in obj.data_vars.items()
            }
        )
    return _normalize_array(obj, method, offset, scale, fill_value, clip)


def _denormalize_array(da, offset, scale):
    attrs = getattr(da, "attrs", {})
    offset = attrs.get("ml_offset") if offset is None else offset
    scale = attrs.get("ml_scale") if scale is None else scale
    if offset is None or scale is None:
        raise ValueError(
            "offset and scale are needed (no ml_offset/ml_scale attributes)"
        )
    out = da * scale + offset
    if isinstance(out, xr.DataArray):
        new = {k: v for k, v in attrs.items() if k not in _NORMALIZATION_ATTRS}
        if "ml_units" in attrs:
            new["units"] = attrs["ml_units"]
        if "ml_standard_name" in attrs:
            new["standard_name"] = attrs["ml_standard_name"]
        out.attrs = new
    return out


def denormalize(obj, *, offset=None, scale=None):
    """
    Undo :func:`normalize`: ``x * scale + offset``.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or numpy.ndarray
        Normalised fields (or model outputs in normalised units).
    offset, scale : float or dict, optional
        Default to the ``ml_offset`` and ``ml_scale`` attributes written by
        :func:`normalize`; needed for NumPy arrays.

    Returns
    -------
    Same type as ``obj``, with the original units restored.
    """
    if isinstance(obj, xr.Dataset):
        return obj.assign(
            {
                v: _denormalize_array(
                    da, _per_variable(offset, v), _per_variable(scale, v)
                )
                for v, da in obj.data_vars.items()
            }
        )
    return _denormalize_array(obj, offset, scale)
