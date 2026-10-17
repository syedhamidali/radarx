#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Helpers shared by the Level I modules.

Lookup of the IQ variable and of the radar constants (wavelength and pulse
repetition time), the broadcast of arrays against each other for a kernel that
works on one series (or one spectrum) at a time, and the lazy application of a
block function with ``xarray.apply_ufunc``.
"""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr

from ..retrieve import _fieldtools as ft

SPEED_OF_LIGHT = 299792458.0  # m s-1

IQ_NAMES = ("IQ", "iq", "IQH", "IQV")
SPECTRUM_NAMES = ("SPECTRUM", "spectrum", "SPECTRUMH", "SPECTRUMV")
PULSE = "pulse"
VELOCITY = "velocity"
RANGE = "range"


def pulse_dimension(da, dim):
    """Check the pulse dimension of an IQ array."""
    if dim not in da.dims:
        raise ValueError(f"the IQ array has the dimensions {da.dims}, no {dim!r}")
    if da.sizes[dim] < 3:
        raise ValueError("an IQ time series needs at least 3 pulses")
    return dim


def check_complex(da):
    if np.dtype(da.dtype).kind != "c":
        raise ValueError(
            "the IQ samples must be complex; build them from I and Q with "
            "radarx.level1.as_iq"
        )


def _constant(values, name):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"{name} is not finite")
    if np.any(finite != finite.flat[0]):
        raise ValueError(f"{name} varies over the data; pass a single value")
    return float(finite.flat[0])


def _positive(value, name):
    value = float(value)
    if not (np.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be positive, not {value}")
    return value


def radar_constants(da, ds, wavelength, prt):
    """
    Wavelength (m) and pulse repetition time (s) of an IQ array.

    The argument is used if given; otherwise the attributes of the IQ array
    (``wavelength``, ``prt``), then the variables or attributes of the sweep
    (``wavelength``, ``radar_frequency`` in Hz, ``prt``).
    """
    if wavelength is None:
        for holder in (
            da.attrs,
            getattr(ds, "attrs", {}),
            getattr(ds, "data_vars", {}),
        ):
            if "wavelength" in holder:
                wavelength = _constant(holder["wavelength"], "wavelength")
                break
        else:
            for holder in (
                da.attrs,
                getattr(ds, "attrs", {}),
                getattr(ds, "data_vars", {}),
            ):
                if "radar_frequency" in holder:
                    freq = _constant(holder["radar_frequency"], "radar_frequency")
                    wavelength = SPEED_OF_LIGHT / freq
                    break
    if prt is None:
        for holder in (
            da.attrs,
            getattr(ds, "attrs", {}),
            getattr(ds, "data_vars", {}),
        ):
            if "prt" in holder:
                prt = _constant(holder["prt"], "prt")
                break
    if wavelength is None:
        raise ValueError(
            "the wavelength is not known: pass wavelength (m) or put it in the "
            "attributes of the IQ array"
        )
    if prt is None:
        raise ValueError(
            "the pulse repetition time is not known: pass prt (s) or put it in "
            "the attributes of the IQ array"
        )
    return _positive(wavelength, "wavelength"), _positive(prt, "prt")


def iq_array(obj, name):
    """The complex IQ DataArray of a DataArray or Dataset, and its Dataset."""
    if isinstance(obj, xr.DataArray):
        return obj, None
    key = ft.find_field(obj, name, IQ_NAMES, "IQ time series")
    return obj[key], obj


def spectrum_array(obj, name):
    """The spectrum DataArray of a DataArray or Dataset."""
    if isinstance(obj, xr.DataArray):
        return obj
    return obj[ft.find_field(obj, name, SPECTRUM_NAMES, "Doppler spectrum")]


def velocity_axis(da, dim=VELOCITY):
    """Check the velocity axis of a spectrum and return its bin width (m s-1)."""
    if dim not in da.dims:
        raise ValueError(f"the spectrum has the dimensions {da.dims}, no {dim!r}")
    k = da.sizes[dim]
    if k < 3:
        raise ValueError("a spectrum needs at least 3 bins")
    if dim not in da.coords:
        raise ValueError(f"the spectrum has no {dim!r} coordinate")
    v = np.asarray(da[dim].values, dtype=np.float64)
    dv = np.diff(v)
    if not (np.all(dv > 0) and np.allclose(dv, dv[0], rtol=1e-6)):
        raise ValueError(f"the {dim!r} coordinate must increase in equal steps")
    if not np.isclose(v[k // 2], 0.0, atol=1e-6 * dv[0]):
        raise ValueError(f"bin {k // 2} of {dim!r} must be the zero velocity")
    return float(dv[0])


def flatten(blocks, ncores):
    """
    Broadcast blocks over their leading dimensions and flatten those.

    Block ``i`` has ``ncores[i]`` core dimensions last. Returns the leading
    shape and the blocks reshaped to ``(n, *core)``.
    """
    leads = [b.shape[: b.ndim - nc] for b, nc in zip(blocks, ncores)]
    lead = np.broadcast_shapes(*leads)
    flat = []
    for b, nc in zip(blocks, ncores):
        core = b.shape[b.ndim - nc :]
        flat.append(np.broadcast_to(b, lead + core).reshape((-1,) + core))
    return lead, flat


def map_series(func, arrays, in_dims, out_dims, out_sizes, dtypes):
    """
    Apply a block function to DataArrays, lazily for dask arrays.

    ``func`` takes one NumPy block per array, the core dimensions last, and
    returns one array per output with its core dimensions last. The core
    dimensions of an input may be chunked: they are rechunked into one chunk.
    """
    with warnings.catch_warnings():
        # dask runs the block function on empty arrays to learn the result type;
        # for complex input that falls back to a complex stand-in cast to the
        # real output type, which warns without consequence
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        return xr.apply_ufunc(
            func,
            *ft.detach(*arrays),
            input_core_dims=in_dims,
            output_core_dims=out_dims,
            dask="parallelized",
            output_dtypes=dtypes,
            dask_gufunc_kwargs={"output_sizes": out_sizes, "allow_rechunk": True},
            keep_attrs=False,
        )


def node_value(arg, name):
    """The part of an argument that belongs to sweep ``name`` of a volume."""
    return ft.node_arg(arg, name) if name is not None else arg


def db(power):
    """Power in dB; NaN where the power is not positive."""
    power = np.asarray(power, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(
            power > 0.0, 10.0 * np.log10(np.where(power > 0.0, power, 1.0)), np.nan
        )
