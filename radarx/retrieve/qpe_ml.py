#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx Machine-Learning Rain Rate
=================================

Rain rate from the polarimetric variables, the height and the temperature
with a small network (a multilayer perceptron) or any other regression model
exported to ONNX, run through :mod:`radarx.ml`. It follows the machine
learning layer of radarx: the model is named in the registry
(``radarx/ml/models.toml`` or :func:`radarx.ml.register_model`) together with
its licence and citation, the weights are downloaded on first use and never
shipped in the package, and the function takes and returns xarray objects.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}

Model contract
--------------
The ONNX model takes one float32 input ``features`` of shape ``(N, F)`` and
returns float32 ``rain_rate`` of shape ``(N,)`` in mm h-1 (the mean of the
predictive distribution, so that accumulations are not biased low, as they are
by a median of a skewed distribution) and, optionally, ``rain_rate_lower`` and
``rain_rate_upper`` (one standard deviation below and above in the logarithm
of the rain rate, i.e. the 16th and 84th percentile of a log-normal
distribution). The registry entry of the model lists the ``features``, the
names of the ``F`` columns in order, from

==============  ==========================================================
``dbz``         horizontal reflectivity (dBZ)
``zdr``         differential reflectivity (dB)
``kdp``         specific differential phase (degree km-1)
``rhohv``       correlation coefficient (1)
``height``      height of the gate above the radar (km)
``temperature`` air temperature (degrees Celsius)
``wavelength``  radar wavelength (cm), from ``band``
==============  ==========================================================

so a model trained on gauges (in the way of the RainForest random forest of
Wolfensberger et al. 2021, which uses the reflectivity, the differential phase
and the correlation coefficient, the temperature and the height of the gate) and a model trained on
simulated drop size distributions use the same function. The input scaling
and any transformation of the variables belong to the network graph, so that
the function passes the physical values.

A gate is estimated if all inputs are finite and the ``mask`` allows it;
the network is not run on the other gates (most gates of a volume), which
get NaN. The estimate is only as good as the training data of the model:
see the ``description`` of its registry entry for the regime, bands and
variables it was trained on, and do not use it outside them. Correct the
fields for attenuation and calibration before the estimate (see
:mod:`radarx.retrieve.qpe`).

Implementation
--------------
Collecting the gates that are estimated and writing the outputs back is done
in a compiled, multithreaded C++ kernel; an equivalent NumPy implementation
is the fallback and the test oracle.
"""

from __future__ import annotations

__all__ = ["rain_rate_ml"]

__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

import numpy as np
import xarray as xr

from .._fields import node_dataset
from .._provenance import provenance
from .._registry import accessor_method
from .._sweepwise import dask_support
from .qpe import _is_tree, _sweep_names

try:
    from . import _qpe_ml

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _qpe_ml = None
    HAS_COMPILED_KERNEL = False

FEATURES = ("dbz", "zdr", "kdp", "rhohv", "height", "temperature", "wavelength")
#: Radar wavelengths (cm) of the scattering tables of :mod:`radarx.retrieve.dsd`.
WAVELENGTH = {"S": 10.70687, "C": 5.35344, "X": 3.18928}
DEFAULT_MODEL = "radarx-qpe-v1"
_CHUNK = 1 << 20  # gates per call of the network
_NAMES = {
    "dbz": ("DBZH", "DBZ", "reflectivity", "corrected_reflectivity"),
    "zdr": (
        "ZDR",
        "differential_reflectivity",
        "corrected_differential_reflectivity",
    ),
    "kdp": (
        "KDP",
        "specific_differential_phase",
        "corrected_specific_differential_phase",
    ),
    "rhohv": ("RHOHV", "cross_correlation_ratio", "RHO"),
    "temperature": ("TEMPERATURE", "temperature"),
}
_WHAT = {
    "dbz": "reflectivity",
    "zdr": "differential reflectivity",
    "kdp": "specific differential phase",
    "rhohv": "correlation coefficient",
    "height": "height of the gates",
    "temperature": "temperature",
}
_OUTPUTS = {
    "rain_rate": ("RAIN_RATE_ML", "rain rate from a machine-learning model"),
    "rain_rate_lower": (
        "RAIN_RATE_ML_LOWER",
        "rain rate from a machine-learning model, one standard deviation below",
    ),
    "rain_rate_upper": (
        "RAIN_RATE_ML_UPPER",
        "rain rate from a machine-learning model, one standard deviation above",
    ),
}
_KELVIN = ("K", "kelvin", "Kelvin", "degK")


def _use_compiled(engine):
    """Whether to run the compiled kernel for the requested ``engine``."""
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled ML QPE kernel is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


# --------------------------------------------------------------------------
# gathering and scattering the gates (compiled kernel or NumPy)
# --------------------------------------------------------------------------


def _compact(columns, mask, use_compiled, n_threads):
    """Feature matrix ``(m, F)`` and flat positions of the gates that are estimated."""
    if use_compiled:
        return _qpe_ml.compact(
            [np.ascontiguousarray(c, dtype=np.float32) for c in columns],
            (
                np.empty(0, dtype=np.uint8)
                if mask is None
                else np.ascontiguousarray(mask, dtype=np.uint8)
            ),
            int(n_threads or 0),
        )
    stack = np.stack([np.asarray(c, dtype=np.float32) for c in columns], axis=1)
    valid = np.isfinite(stack).all(axis=1)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    index = np.flatnonzero(valid).astype(np.int64)
    return np.ascontiguousarray(stack[index]), index


def _scatter(index, values, n, use_compiled, n_threads):
    """``(K, n)`` array with ``values`` (m, K) at the positions ``index``, NaN elsewhere."""
    if use_compiled:
        return _qpe_ml.scatter(index, values, int(n), int(n_threads or 0))
    out = np.full((values.shape[1], n), np.nan, dtype=np.float32)
    out[:, index] = values.T
    return out


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------


def _model_features(model, info):
    features = info.get("features", getattr(model, "features", None))
    if features is None:
        raise ValueError(
            "the model does not list its input 'features' (a registry field or "
            "an attribute of the model)"
        )
    if isinstance(features, str):
        features = [f.strip() for f in features.split(",") if f.strip()]
    features = list(features)
    unknown = [f for f in features if f not in FEATURES]
    if unknown or not features:
        raise ValueError(
            f"the features of the model must be some of {FEATURES}, not {features}"
        )
    return features


def _qpe_model(model):
    """The model object, its registry entry and its features."""
    if model is None:
        model = DEFAULT_MODEL
    if isinstance(model, str):
        from radarx import ml

        listed = ml.list_models()
        names = (
            listed.keys() if isinstance(listed, dict) else [m["name"] for m in listed]
        )
        if model not in names:
            raise ValueError(
                f"the rain rate model {model!r} is not registered. The weights of "
                "the radarx rain rate network are not distributed yet; train one "
                "with the scripts in ml/models/qpe of the radarx repository and "
                "register it with radarx.ml.register_model(..., features=[...]), "
                "or pass a loaded model as model=..."
            )
        loaded = ml.load_model(model)
        info = dict(getattr(loaded, "info", None) or {})
        return loaded, info, _model_features(loaded, info)
    if not callable(getattr(model, "run", None)):
        raise TypeError(
            "model must be a registered model name or a loaded radarx.ml "
            f"model with a run() method, not {type(model).__name__}"
        )
    info = dict(getattr(model, "info", None) or {})
    return model, info, _model_features(model, info)


def _model_attrs(model, info):
    attrs = getattr(model, "attrs", None)
    if isinstance(attrs, dict):
        return dict(attrs)
    out = {
        "ml_model": str(info.get("name", type(model).__name__)),
        "ml_model_version": str(info.get("version", "unknown")),
        "ml_model_licence": str(info.get("licence", "unknown")),
    }
    if info.get("citation"):
        out["ml_model_citation"] = str(info["citation"])
    return out


def _run(model, matrix):
    """Run the network over chunks of gates; the outputs by name, each ``(m,)``."""
    parts = {}
    for start in range(0, matrix.shape[0], _CHUNK):
        res = model.run({"features": matrix[start : start + _CHUNK]})
        if "rain_rate" not in res:
            raise KeyError("the model returned no 'rain_rate' output")
        for key in _OUTPUTS:
            if key in res:
                parts.setdefault(key, []).append(
                    np.asarray(res[key], dtype=np.float32).reshape(-1)
                )
    if not parts:  # no gate to estimate
        return {"rain_rate": np.empty(0, dtype=np.float32)}
    return {k: np.concatenate(v) for k, v in parts.items()}


# --------------------------------------------------------------------------
# the fields
# --------------------------------------------------------------------------


def _to_celsius(da):
    units = da.attrs.get("units")
    if units in _KELVIN:
        return da - 273.15
    if units is None and da.size and not hasattr(da.data, "dask"):
        values = da.values
        if np.isfinite(values).any() and np.nanmedian(values) > 100.0:
            return da - 273.15
    return da


def _height(ds, spec):
    """Height of the gates above the radar in km, or None."""
    if isinstance(spec, xr.DataArray):
        da = spec
    elif isinstance(spec, str):
        if ds is None or spec not in ds:
            raise KeyError(f"{spec!r} is not in the dataset")
        da = ds[spec]
    elif spec is None:
        if ds is None or "z" not in ds.coords:
            return None
        da = ds["z"]
        if "altitude" in ds.coords:  # z is above sea level
            da = da - ds["altitude"]
    else:
        return float(spec) / 1000.0
    return da / 1000.0 if da.attrs.get("units", "m") != "km" else da


def _plain_number(value):
    return isinstance(value, (int, float, np.integer, np.floating))


def _gather(ds, features, specs, wavelength, template=None):
    """The DataArray (or number) of every feature the model needs."""
    fields = {}
    for key in features:
        spec = specs.get(key)
        if key == "wavelength":
            fields[key] = float(wavelength)
        elif key == "height":
            value = _height(ds, spec)
            if value is None:
                raise KeyError(
                    "the model needs the height of the gates: pass height= (m "
                    "above the radar) or give the sweep a z coordinate"
                )
            fields[key] = value
        elif isinstance(spec, xr.DataArray):
            fields[key] = _to_celsius(spec) if key == "temperature" else spec
        elif _plain_number(spec) and key == "temperature":
            fields[key] = float(spec)
        else:
            if ds is None:
                raise ValueError(
                    f"the {_WHAT[key]} must be passed as a DataArray when obj "
                    "is a DataArray"
                )
            if spec is not None:
                if spec not in ds:
                    raise KeyError(f"{spec!r} is not in the dataset")
                da = ds[spec]
            else:
                for cand in _NAMES[key]:
                    if cand in ds:
                        da = ds[cand]
                        break
                else:
                    raise KeyError(
                        f"no {_WHAT[key]} found among {_NAMES[key]}; pass the field name"
                    )
            fields[key] = _to_celsius(da) if key == "temperature" else da
    return fields


def _estimate(fields, features, mask, min_dbz, model, info, use_compiled, n_threads):
    """The Dataset of rain rates on the grid of the fields."""
    named = {k: v for k, v in fields.items() if isinstance(v, xr.DataArray)}
    if not named:
        raise ValueError("at least one feature must be a field")
    if mask is not None:
        named["mask"] = mask
    aligned = dict(zip(named, xr.broadcast(*named.values())))
    template = next(iter(aligned.values()))
    dims = template.dims

    def flat(key):
        value = aligned.get(key, fields.get(key))
        if isinstance(value, xr.DataArray):
            return np.asarray(value.transpose(*dims).values, dtype=np.float32).reshape(
                -1
            )
        return np.full(template.size, value, dtype=np.float32)

    columns = [flat(k) for k in features]
    flat_mask = None if mask is None else flat("mask") > 0.5
    if min_dbz is not None:
        if "dbz" not in fields:
            raise ValueError("min_dbz needs the reflectivity as a feature of the model")
        with np.errstate(invalid="ignore"):
            keep = flat("dbz") >= min_dbz
        flat_mask = keep if flat_mask is None else flat_mask & keep
    matrix, index = _compact(columns, flat_mask, use_compiled, n_threads)
    outputs = _run(model, matrix)
    keys = [k for k in _OUTPUTS if k in outputs]
    values = np.stack([outputs[k] for k in keys], axis=1)
    if values.shape[0] != index.size:
        raise ValueError(
            f"the model returned {values.shape[0]} values for {index.size} gates"
        )
    values = np.maximum(values, 0.0).astype(np.float32)
    grid = _scatter(index, values, template.size, use_compiled, n_threads)
    attrs = _model_attrs(model, info)
    data = {}
    for j, key in enumerate(keys):
        name, long_name = _OUTPUTS[key]
        data[name] = xr.DataArray(
            grid[j].reshape(template.shape),
            dims=dims,
            coords=template.coords,
            attrs={
                "standard_name": "rainfall_rate",
                "long_name": long_name,
                "units": "mm h-1",
                **attrs,
            },
        )
    return xr.Dataset(data)


@dask_support("computes", "the network runs on all gates of a sweep at once.")
@provenance("Rain rate from a machine-learning model")
def rain_rate_ml(
    obj,
    *,
    model=None,
    band="S",
    dbz=None,
    zdr=None,
    kdp=None,
    rhohv=None,
    height=None,
    temperature=None,
    mask=None,
    min_dbz=None,
    wavelength=None,
    n_threads=None,
    engine="auto",
):
    """
    Rain rate from a machine-learning model.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataTree
        A sweep or grid with the fields the model needs (by default
        ``DBZH``, ``ZDR``, ``KDP``, ``RHOHV`` and, for ``height``, the
        coordinates ``z`` and ``altitude``), or a volume with ``sweep_*``
        groups (sweeps without the fields are skipped).
    model : str or radarx.ml.Model, optional
        The name of a model of the registry of :mod:`radarx.ml` or a loaded
        model. Default ``"radarx-qpe-v1"``, which is not distributed yet: train
        one with ``ml/models/qpe`` of the radarx repository and register it
        with :func:`radarx.ml.register_model`.
    band : {"S", "C", "X"}, optional
        Radar band, which gives the wavelength feature of models that use it.
        Default ``"S"``.
    dbz, zdr, kdp, rhohv : str or xarray.DataArray, optional
        Names of the fields, or the fields. By default ``DBZH``, ``ZDR``,
        ``KDP`` and ``RHOHV`` (and the other names of
        :func:`radarx.retrieve.rain_rate_z`).
    height : str, xarray.DataArray or float, optional
        Height of the gates above the radar (m, a number or a field; km if the
        units attribute of the field says so), for models that use it. Default: the coordinate ``z`` minus the
        radar ``altitude``, if there is one.
    temperature : str, xarray.DataArray or float, optional
        Air temperature (degrees Celsius; Kelvin is detected from the units
        or the values) for models that use it, for example the field
        ``TEMPERATURE`` of the sounding accessor.
    mask : xarray.DataArray, optional
        Boolean field on the grid of the fields: only gates where it is True
        are estimated. Use the rain classes of :func:`radarx.retrieve.hid` or
        the result of :func:`radarx.retrieve.echo_mask`. Default: all gates.
    min_dbz : float, optional
        Gates with a lower reflectivity are not estimated (NaN). Default:
        none. The training range of the model is in its registry entry.
    wavelength : float, optional
        Radar wavelength (cm) instead of the one of ``band``.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled
        kernel and falls back to NumPy.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``RAIN_RATE_ML`` (mm h-1, float32), and ``RAIN_RATE_ML_LOWER`` and
        ``RAIN_RATE_ML_UPPER`` (one standard deviation below and above in
        the logarithm of the rain rate) if the model gives them, on the
        coordinates of the fields, NaN where a gate is not estimated; for a
        DataTree one node per sweep. The attributes ``ml_model``,
        ``ml_model_version``, ``ml_model_licence`` and ``ml_model_citation``
        record the model. The input is not changed.

    Raises
    ------
    KeyError
        If a field the model needs, or a sweep that has all of them, is not
        found, or the model returns no ``rain_rate``.
    ValueError
        For an unknown ``band`` or ``engine``, a model without ``features``
        or with unknown ones, or a model name that is not registered.
    TypeError
        If ``obj`` is not a Dataset or DataTree, or ``model`` is neither a
        name nor an object with a ``run`` method.
    ImportError
        For a model name when ONNX Runtime is not installed (``pip install
        radarx[ml]``), or ``engine="compiled"`` without the kernel.

    Notes
    -----
    The network and what it was trained on are described by the model, not
    by this function. A model trained on simulated drop size distributions
    (``ml/models/qpe``) is a multivariate rain-rate relation that weights
    :math:`Z_H`, :math:`Z_{DR}` and :math:`K_{DP}` by their noise and gives
    an uncertainty from the natural variability of the drop size
    distribution; it does not correct for attenuation, calibration or the
    vertical profile of reflectivity. A model trained on gauges (in the
    way of RainForest) learns these effects for the radar and the region it was
    trained on. The gauge adjustment of :func:`radarx.retrieve.gauge_adjust`
    applies to either.

    References
    ----------
    Wolfensberger, D., M. Gabella, M. Boscacci, U. Germann, and A. Berne,
    2021: RainForest: a random forest algorithm for quantitative
    precipitation estimation over Switzerland. *Atmos. Meas. Tech.*, **14**
    (4), 3169-3193, https://doi.org/10.5194/amt-14-3169-2021

    Vulpiani, G., S. Giangrande, and F. S. Marzano, 2009: Rainfall
    estimation from polarimetric S-band radar measurements: Validation of a
    neural network approach. *J. Appl. Meteor. Climatol.*, **48** (10),
    2022-2036, https://doi.org/10.1175/2009JAMC2172.1

    Examples
    --------
    >>> ml.register_model("my-qpe", "qpe.onnx", None, "MIT", "citation",
    ...                   features=["dbz", "zdr", "kdp", "rhohv"])  # doctest: +SKIP
    >>> rr = radarx.retrieve.rain_rate_ml(sweep, model="my-qpe")  # doctest: +SKIP
    >>> rr.RAIN_RATE_ML.attrs["ml_model"]  # doctest: +SKIP
    'my-qpe'
    """
    use_compiled = _use_compiled(engine)
    if band not in WAVELENGTH:
        raise ValueError(f"band must be one of {tuple(WAVELENGTH)}, not {band!r}")
    if not isinstance(obj, (xr.Dataset, xr.DataTree)):
        raise TypeError("obj must be an xarray.Dataset or xarray.DataTree")
    loaded, info, features = _qpe_model(model)
    wave = WAVELENGTH[band] if wavelength is None else float(wavelength)
    specs = {
        "dbz": dbz,
        "zdr": zdr,
        "kdp": kdp,
        "rhohv": rhohv,
        "height": height,
        "temperature": temperature,
    }

    def one(ds):
        fields = _gather(ds, features, specs, wave)
        return _estimate(
            fields, features, mask, min_dbz, loaded, info, use_compiled, n_threads
        )

    if isinstance(obj, xr.Dataset):
        return one(obj)
    if not _is_tree(obj):  # pragma: no cover - xr.DataTree is the only tree
        raise TypeError("obj must be an xarray.Dataset or xarray.DataTree")
    nodes_out = {"/": obj.root.to_dataset(inherit=False)}
    for sweep in _sweep_names(obj):
        try:
            nodes_out[sweep] = one(node_dataset(obj[sweep]))
        except KeyError:
            continue
    if len(nodes_out) == 1:
        raise KeyError(f"no sweep contains the fields {features}")
    return xr.DataTree.from_dict(nodes_out)


@accessor_method("dataset", "datatree", name="rain_rate_ml", wraps=rain_rate_ml)
def _rain_rate_ml_accessor(self):
    """
    Rain rate from a machine-learning model.

    See :func:`radarx.retrieve.rain_rate_ml` for the parameters.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``RAIN_RATE_ML`` (mm h-1) and, if the model gives them, its lower and
        upper bounds.
    """
