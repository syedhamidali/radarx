#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx IQ Time Series
=====================

Level I data are the complex voltage samples of the receiver, the in-phase
(I) and quadrature (Q) components, of every gate for the pulses of a dwell
(one ray). radarx holds them in a complex :class:`xarray.DataArray` named
``IQ`` with a ``pulse`` dimension (slow time, one sample per transmitted
pulse), usually a ``range`` dimension and a ray dimension (``azimuth``). Two
numbers describe the sampling and are kept in the attributes of the array: the
radar ``wavelength`` (m) and the pulse repetition time ``prt`` (s). With them
the Doppler frequency :math:`f_d` of a gate becomes a radial velocity

.. math::

    v = -\\frac{\\lambda f_d}{2}, \\qquad v_a = \\frac{\\lambda}{4\\,T_s},

positive for motion away from the radar (Doviak and Zrnic 1993, Eq. 3.30), with
the Nyquist velocity :math:`v_a` for the pulse repetition time :math:`T_s`.

The functions here build such an array from raw files or from I and Q arrays
and simulate time series with known moments, which is how the processing of
:mod:`radarx.level1` is tested.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = ["as_iq", "read_iq", "simulate_iq"]

__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

import os

import numpy as np
import xarray as xr

from ._common import PULSE, RANGE

DEFAULT_DIMS = ("azimuth", PULSE, RANGE)


def _complex_dtype(*arrays):
    """complex128 for double precision input, complex64 otherwise."""
    wide = False
    for a in arrays:
        dt = np.dtype(a.dtype)
        wide = (
            wide
            or (dt.kind == "f" and dt.itemsize >= 8)
            or (dt.kind == "c" and dt.itemsize >= 16)
        )
    return np.complex128 if wide else np.complex64


def _label(da, wavelength, prt, range_start, gate_spacing, name):
    """Names, coordinates and attributes of an IQ array."""
    da = da.rename(name)
    if PULSE in da.dims and PULSE not in da.coords:
        da = da.assign_coords({PULSE: np.arange(da.sizes[PULSE])})
    if gate_spacing is not None:
        if RANGE not in da.dims:
            raise ValueError(
                "gate_spacing is given but the data have no 'range' dimension"
            )
        da = da.assign_coords(
            {
                RANGE: (
                    RANGE,
                    float(range_start)
                    + float(gate_spacing) * np.arange(da.sizes[RANGE]),
                    {"long_name": "range from the radar", "units": "m"},
                )
            }
        )
    attrs = dict(da.attrs)
    attrs.setdefault("long_name", "complex IQ voltage samples (I + jQ)")
    if wavelength is not None:
        attrs["wavelength"] = float(wavelength)
    if prt is not None:
        attrs["prt"] = float(prt)
    da.attrs = attrs
    return da


def as_iq(
    i,
    q=None,
    *,
    dims=DEFAULT_DIMS,
    scale=1.0,
    wavelength=None,
    prt=None,
    range_start=0.0,
    gate_spacing=None,
    name="IQ",
):
    """
    Complex IQ array from the in-phase and quadrature samples.

    Parameters
    ----------
    i : array_like or xarray.DataArray
        The in-phase samples, or the complex samples if ``q`` is not given.
    q : array_like or xarray.DataArray, optional
        The quadrature samples, with the shape of ``i``.
    dims : sequence of str, optional
        Dimension names of NumPy input, in the order of the axes. The
        default ``("azimuth", "pulse", "range")`` is for a 3-D array; use
        ``("pulse", "range")`` or ``("pulse",)`` for fewer axes. The dimensions
        of a DataArray are kept.
    scale : float, optional
        Factor for integer counts (for example ``1 / 32768`` for 16 bit
        samples). Default 1.
    wavelength : float, optional
        Radar wavelength (m), stored in the attributes.
    prt : float, optional
        Pulse repetition time (s), stored in the attributes.
    range_start : float, optional
        Range of the first gate (m), used with ``gate_spacing``. Default 0.
    gate_spacing : float, optional
        Distance between the gates (m). With it the array gets a ``range``
        coordinate; without it, none.
    name : str, optional
        Name of the array. Default ``"IQ"``.

    Returns
    -------
    xarray.DataArray
        Complex samples, ``complex64`` (``complex128`` for double precision
        input), with a ``pulse`` coordinate and the ``wavelength`` and
        ``prt`` attributes.

    Raises
    ------
    ValueError
        If ``i`` and ``q`` differ in shape, the number of ``dims`` differs
        from the number of axes, or the data are not 1-D, 2-D or 3-D without
        dimension names.
    """
    if isinstance(i, xr.DataArray):
        if q is None:
            data = i
        else:
            if not isinstance(q, xr.DataArray):
                q = xr.DataArray(np.asarray(q), dims=i.dims, coords=i.coords)
            if i.shape != q.shape:
                raise ValueError("i and q must have the same shape")
            data = i + 1j * q.transpose(*i.dims)
        dtype = _complex_dtype(i)
        data = (data * scale).astype(dtype) if scale != 1.0 else data.astype(dtype)
        da = data
    else:
        i = np.asarray(i)
        if q is not None:
            q = np.asarray(q)
            if i.shape != q.shape:
                raise ValueError("i and q must have the same shape")
            values = i.astype(np.float64) + 1j * q.astype(np.float64)
        else:
            if i.dtype.kind != "c":
                raise ValueError("pass q, or complex samples as i")
            values = i
        dtype = _complex_dtype(i, *([q] if q is not None else []))
        if scale != 1.0:
            values = values * scale
        dims = tuple(dims)
        if len(dims) != values.ndim:
            raise ValueError(
                f"the data have {values.ndim} axes but {len(dims)} dimension names "
                f"were given: {dims}"
            )
        da = xr.DataArray(values.astype(dtype), dims=dims)
    return _label(da, wavelength, prt, range_start, gate_spacing, name)


_NETCDF = (".nc", ".nc4", ".cdf", ".netcdf")


def read_iq(
    path,
    *,
    shape=None,
    dtype="int16",
    layout="interleaved",
    offset=0,
    variables=("I", "Q"),
    **kwargs,
):
    """
    Read an IQ time series file.

    Three kinds of file are read:

    * a NumPy file (``.npy``) with a complex array, or a real array whose last
      axis has length 2 (I, Q);
    * a NetCDF file (``.nc``) or Zarr store (``.zarr``) with the variables
      ``variables`` (default ``I`` and ``Q``) or one complex variable ``IQ``;
      the dimension names of the file are kept and the data stay lazy for Zarr;
    * any other file is raw binary: ``shape`` samples of real ``dtype`` values
      after ``offset`` bytes, either interleaved (``I0 Q0 I1 Q1 ...``) or
      planar (all I, then all Q), the samples in C order of ``shape``.

    Parameters
    ----------
    path : str or os.PathLike
        File or Zarr directory.
    shape : tuple of int, optional
        Shape of the complex array of a raw file, for example
        ``(n_rays, n_pulses, n_gates)``. Required for raw files.
    dtype : str or numpy.dtype, optional
        Sample type of a raw file, such as ``"int16"``, ``">i2"`` or
        ``"float32"``. Default ``"int16"``.
    layout : {"interleaved", "planar"}, optional
        Order of the I and Q samples in a raw file. Default ``"interleaved"``.
    offset : int, optional
        Bytes to skip at the start of a raw file (a header). Default 0.
    variables : tuple of str, optional
        Names of the in-phase and quadrature variables of a NetCDF or Zarr
        file.
    **kwargs
        ``dims``, ``scale``, ``wavelength``, ``prt``, ``range_start``,
        ``gate_spacing`` and ``name`` as for :func:`as_iq`.

    Returns
    -------
    xarray.DataArray
        The complex samples, see :func:`as_iq`.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If a raw file is read without ``shape``, is shorter than ``shape``
        needs, or ``layout`` is unknown; or a NetCDF/Zarr file lacks the
        variables.
    """
    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    lower = path.lower().rstrip("/\\")
    if lower.endswith(".npy"):
        data = np.load(path)
        if data.dtype.kind != "c":
            if data.ndim < 1 or data.shape[-1] != 2:
                raise ValueError(
                    "a real .npy array needs a last axis of length 2 (I, Q)"
                )
            return as_iq(data[..., 0], data[..., 1], **_dims(kwargs, data.ndim - 1))
        return as_iq(data, **_dims(kwargs, data.ndim))
    if lower.endswith(_NETCDF) or lower.endswith(".zarr"):
        ds = xr.open_zarr(path) if lower.endswith(".zarr") else xr.open_dataset(path)
        names = {"IQ", "iq"} & set(ds.data_vars)
        if names:
            return as_iq(ds[sorted(names)[0]], **kwargs)
        if not all(v in ds.data_vars for v in variables):
            raise ValueError(f"{path} has no variables {tuple(variables)} or 'IQ'")
        return as_iq(ds[variables[0]], ds[variables[1]], **kwargs)
    if shape is None:
        raise ValueError(
            "a raw IQ file needs its shape, for example shape=(360, 64, 500)"
        )
    if layout not in ("interleaved", "planar"):
        raise ValueError(f"layout must be 'interleaved' or 'planar', not {layout!r}")
    shape = tuple(int(n) for n in shape)
    count = 2 * int(np.prod(shape))
    values = np.fromfile(path, dtype=np.dtype(dtype), count=count, offset=int(offset))
    if values.size < count:
        raise ValueError(
            f"{path} holds {values.size} samples after the offset but shape "
            f"{shape} needs {count}"
        )
    if layout == "interleaved":
        values = values.reshape(shape + (2,))
        re, im = values[..., 0], values[..., 1]
    else:
        values = values.reshape((2,) + shape)
        re, im = values[0], values[1]
    return as_iq(re, im, **_dims(kwargs, len(shape)))


def _dims(kwargs, ndim):
    """Default dimension names for ``ndim`` axes unless ``dims`` is given."""
    out = dict(kwargs)
    if "dims" not in out:
        names = {1: (PULSE,), 2: (PULSE, RANGE), 3: DEFAULT_DIMS}
        if ndim not in names:
            raise ValueError("pass dims for data that are not 1-D, 2-D or 3-D")
        out["dims"] = names[ndim]
    return out


def _as_param(value, dims, name):
    """A simulation parameter as a DataArray."""
    if isinstance(value, xr.DataArray):
        return value
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 0:
        return xr.DataArray(value)
    if dims is not None:
        if len(dims) != value.ndim:
            raise ValueError(
                f"dims {tuple(dims)} do not fit the {value.ndim}-D parameter {name}"
            )
        return xr.DataArray(value, dims=tuple(dims))
    defaults = {1: (RANGE,), 2: ("azimuth", RANGE)}
    if value.ndim not in defaults:
        raise ValueError(f"pass dims for the {value.ndim}-D parameter {name}")
    return xr.DataArray(value, dims=defaults[value.ndim])


def _gaussian_series(rng, velocity, width, power, n_pulses, wavelength, prt):
    """
    Complex Gaussian time series with a Gaussian Doppler spectrum.

    Zrnic's (1975) method: the amplitudes of the DFT bins are the square root
    of the power spectrum times complex Gaussian numbers, the series is the
    inverse DFT. ``velocity``, ``width`` and ``power`` are 1-D arrays, one value
    per series; the result is ``(n, n_pulses)``.
    """
    u = np.arange(n_pulses) / n_pulses  # frequency in cycles per sample
    u_d = -2.0 * velocity * prt / wavelength  # Doppler shift, v = -lambda f / 2
    sig = np.maximum(2.0 * width * prt / wavelength, 0.1 / n_pulses)
    spec = np.zeros((velocity.size, n_pulses))
    for alias in range(-3, 4):
        spec += np.exp(-0.5 * ((u[None, :] - u_d[:, None] - alias) / sig[:, None]) ** 2)
    p_bin = power[:, None] * spec / spec.sum(axis=1, keepdims=True)
    zeta = (
        rng.standard_normal((velocity.size, n_pulses))
        + 1j * rng.standard_normal((velocity.size, n_pulses))
    ) / np.sqrt(2.0)
    return n_pulses * np.fft.ifft(np.sqrt(p_bin) * zeta, axis=1)


def simulate_iq(
    velocity,
    width=1.0,
    power=1.0,
    *,
    noise=0.0,
    clutter_power=0.0,
    clutter_width=0.3,
    n_pulses=64,
    wavelength=0.1,
    prt=1e-3,
    dims=None,
    seed=None,
):
    """
    Simulate IQ time series with a Gaussian Doppler spectrum of known moments.

    Every gate gets a complex Gaussian series whose power spectrum is a
    Gaussian of mean velocity ``velocity`` and standard deviation ``width``,
    folded into the Nyquist interval, plus white noise and optionally ground
    clutter, a second Gaussian at zero velocity. The method is that of Zrnic
    (1975): the amplitudes of the DFT bins are the square root of the power
    spectrum times complex Gaussian numbers, and the series is the inverse
    DFT. The signal power of a gate is ``power`` on average, so the moments of
    the simulated series are known (the mean power, the velocity and the width
    that :func:`~radarx.level1.pulse_pair_moments` and
    :func:`~radarx.level1.spectral_moments` should find, with the sampling error
    of the dwell).

    Parameters
    ----------
    velocity : float, array_like or xarray.DataArray
        Mean radial velocity (m s-1), positive away from the radar. Values
        outside the Nyquist interval are aliased by the sampling, as in a
        radar. An array gives one series per element.
    width : float, array_like or xarray.DataArray, optional
        Spectrum width, the standard deviation of the velocity spectrum
        (m s-1). Widths below a tenth of a spectral bin are raised to that
        value. Default 1.
    power : float, array_like or xarray.DataArray, optional
        Signal power, in the squared units of the samples. Default 1.
    noise : float, array_like or xarray.DataArray, optional
        Power of the white receiver noise per sample. Default 0.
    clutter_power : float, array_like or xarray.DataArray, optional
        Power of the ground clutter (zero mean velocity). Default 0.
    clutter_width : float, optional
        Spectrum width of the clutter (m s-1). Default 0.3.
    n_pulses : int, optional
        Pulses per series. Default 64.
    wavelength : float, optional
        Radar wavelength (m). Default 0.1 (S band).
    prt : float, optional
        Pulse repetition time (s). Default 1e-3. The Nyquist velocity is
        ``wavelength / (4 prt)``.
    dims : sequence of str, optional
        Dimension names of NumPy parameters of 1 or 2 axes; the defaults are
        ``("range",)`` and ``("azimuth", "range")``. DataArray parameters bring
        their own.
    seed : int or numpy.random.Generator, optional
        Seed of the random numbers.

    Returns
    -------
    xarray.DataArray
        Complex ``complex64`` samples with the dimensions of the parameters
        and a ``pulse`` dimension (before ``range`` if there is one), named
        ``IQ``, with the ``wavelength`` and ``prt`` attributes. This function
        runs in NumPy only: it generates test data and is not a processing
        step.

    Raises
    ------
    ValueError
        If ``n_pulses`` is below 3, the wavelength or the pulse repetition
        time is not positive, a power is negative, or a NumPy parameter needs
        ``dims``.

    References
    ----------
    Zrnić, D. S., 1975: Simulation of weatherlike Doppler spectra and
    signals. *J. Appl. Meteor.*, **14** (4), 619-620,
    https://doi.org/10.1175/1520-0450(1975)014<0619:SOWDSA>2.0.CO;2

    Examples
    --------
    >>> iq = radarx.level1.simulate_iq(
    ...     velocity=[5.0, -8.0], width=1.5, power=100.0, noise=1.0, seed=1
    ... )  # doctest: +SKIP
    """
    if n_pulses < 3:
        raise ValueError("n_pulses must be at least 3")
    if not (wavelength > 0 and prt > 0):
        raise ValueError("wavelength and prt must be positive")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    params = {
        "velocity": velocity,
        "width": width,
        "power": power,
        "noise": noise,
        "clutter_power": clutter_power,
    }
    arrays = {k: _as_param(v, dims, k) for k, v in params.items()}
    for key in ("power", "noise", "clutter_power"):
        if np.any(arrays[key].values < 0):
            raise ValueError(f"{key} must not be negative")
    if np.any(arrays["width"].values < 0):
        raise ValueError("width must not be negative")
    names = list(arrays)
    broadcast = xr.broadcast(*arrays.values())
    arrays = dict(zip(names, broadcast))
    template = arrays["velocity"]
    shape = template.shape
    flat = {
        k: np.asarray(v.values, dtype=np.float64).reshape(-1) for k, v in arrays.items()
    }
    x = _gaussian_series(
        rng, flat["velocity"], flat["width"], flat["power"], n_pulses, wavelength, prt
    )
    if np.any(flat["clutter_power"] > 0):
        x = x + _gaussian_series(
            rng,
            np.zeros_like(flat["velocity"]),
            np.full_like(flat["velocity"], float(clutter_width)),
            flat["clutter_power"],
            n_pulses,
            wavelength,
            prt,
        )
    if np.any(flat["noise"] > 0):
        x = x + np.sqrt(flat["noise"] / 2.0)[:, None] * (
            rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
        )
    da = xr.DataArray(
        x.reshape(shape + (n_pulses,)).astype(np.complex64),
        dims=template.dims + (PULSE,),
        coords=template.coords,
    )
    if RANGE in da.dims:
        order = [d for d in da.dims if d not in (PULSE, RANGE)] + [PULSE, RANGE]
        da = da.transpose(*order)
    return _label(da, wavelength, prt, 0.0, None, "IQ")
