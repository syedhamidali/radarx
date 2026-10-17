#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx Ground Clutter Filtering
===============================

Ground clutter is a strong echo with a narrow spectrum centred on zero
velocity (a width of 1 m s-1 or less); it hides the weather echo at low radial
velocities and biases the moments towards zero. Two filters remove it, one in
the time domain and one in the frequency domain.

Regression filter
-----------------
A polynomial of degree :math:`p` is fitted by least squares to the I and Q
samples of each gate over the dwell and subtracted (Torres and Zrnic 1999).
Clutter changes slowly during the dwell, so a low-order polynomial describes it,
while weather with a velocity away from zero oscillates and is left. With
:math:`Q` the :math:`M\\times(p+1)` matrix of orthonormal polynomials (the
discrete Legendre polynomials on the :math:`M` pulses) the filtered series is

.. math::

    y = x - Q\\,(Q^{\\mathsf T} x),

a high-pass filter with a null at zero velocity and a stop band that widens
with :math:`p`. Degree 0 removes the mean (the DC line), degree 1 also a linear
trend, and so on. The filter acts on the full series, so its output is a series
that goes on to :func:`~radarx.level1.doppler_spectra` or
:func:`~radarx.level1.pulse_pair_moments`.

Spectral notch filter
---------------------
In the spectrum the clutter occupies the bins next to zero velocity. A spectral
filter removes the DC line and one or two adjacent coefficients before the
moments are estimated (Doviak and Zrnic 1993, Section 7.10), and replaces the
removed bins, which would otherwise leave a hole that biases the weather
moments. ``fill`` chooses the replacement:

``"interpolate"`` (default)
    A line between the bins on both sides of the notch. On simulated spectra
    of one periodogram it gave the smallest velocity error of the three
    (the Gaussian fit of a single noisy periodogram is less stable).
``"gaussian"``
    The weather part of the spectrum is taken to be a Gaussian. Its peak
    outside the notch is fitted with the weighted least squares fit of the
    logarithm to a parabola of Guo (2011), with the squared power as weight,
    which is a closed-form solution, and the notch bins take the fitted values
    plus the noise level. If fewer than 3 bins can be fitted or the parabola
    has no maximum, the interpolation is used. radarx's choice; it needs
    spectra averaged over several periodograms (``segments``) to be stable.
``"noise"``
    The noise level.

The noise level that the filter needs comes from the objective method of
Hildebrand and Sekhon (1974) unless it is given. The leakage of the clutter
through the window must stay below the weather for the notch to be enough, so
use a window with low sidelobes (``"blackman"``) when the clutter is much
stronger than the weather; the notch cannot remove clutter that the window
has spread over more bins than the notch holds.

Both filters can be applied to selected gates only (``where``), for example
the gates flagged in a clutter map or by a clutter detection; the others are
returned unchanged.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = ["regression_filter", "notch_filter"]

__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

import numpy as np
import xarray as xr

from .._dtypes import FIELD_DTYPE
from .._provenance import provenance
from ..retrieve import _fieldtools as ft
from . import spectra as sp
from ._common import (
    PULSE,
    VELOCITY,
    check_complex,
    db,
    flatten,
    iq_array,
    map_series,
    node_value,
    pulse_dimension,
)

FILLS = {"noise": 0, "interpolate": 1, "gaussian": 2}


def polynomial_basis(order, length):
    """
    Orthonormal polynomials of degree 0 to ``order`` on ``length`` pulses.

    The discrete Legendre polynomials: a QR factorisation of the Legendre
    Vandermonde matrix on the pulses mapped to [-1, 1]. Returns the
    ``(length, order + 1)`` matrix whose columns are orthonormal.
    """
    t = np.linspace(-1.0, 1.0, length)
    q, _ = np.linalg.qr(np.polynomial.legendre.legvander(t, order))
    return np.ascontiguousarray(q)


def _mask(where, name):
    if where is None:
        return None
    where = node_value(where, name)
    if not isinstance(where, xr.DataArray):
        where = xr.DataArray(bool(where))
    return where.astype(bool)


@provenance(
    "Polynomial regression ground clutter filter of IQ time series",
    extra_refs=("doviak-zrnic-1993",),
)
def regression_filter(
    obj,
    iq=None,
    *,
    dim=PULSE,
    order=2,
    where=None,
    n_threads=None,
    engine="auto",
):
    """
    Remove ground clutter from IQ time series with a polynomial regression filter.

    A polynomial of degree ``order`` is fitted to the samples of every gate
    and subtracted; see :mod:`radarx.level1.clutter`.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        The complex IQ samples with a pulse dimension, a sweep that holds
        them, or a volume (sweeps without IQ are skipped).
    iq : str, optional
        Name of the IQ variable of a sweep. Default: ``IQ``, ``iq``, ``IQH``
        or ``IQV``.
    dim : str, optional
        The pulse dimension. Default ``"pulse"``.
    order : int, optional
        Degree of the polynomial, 0 (the mean) up to the number of pulses
        minus 2. The notch widens with the degree, and so does the loss of
        weather signal near zero velocity. Default 2 (radarx choice).
    where : xarray.DataArray of bool, optional
        Filter only the gates where this is True (a clutter map or flag, with
        the dimensions of the gates). The other gates are returned
        unchanged. Default: all gates.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` prefers the compiled kernel and
        falls back to NumPy.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``IQ``, the filtered complex samples with the dimensions of the input,
        and ``CLUTTER_POWER`` (dB), the mean power that was removed, in the
        squared units of the samples (NaN where nothing was removed).

    Raises
    ------
    KeyError
        If no IQ variable is found.
    ValueError
        If the samples are not complex, ``order`` is negative or leaves fewer
        than 2 degrees of freedom, or the pulse dimension is missing.
    ImportError
        If ``engine="compiled"`` and the compiled kernels are not available.

    Notes
    -----
    The filter removes weather signal too: the weather echo with a velocity
    within about ``(order + 1)`` spectral bins of zero is partly removed.
    The degree needed grows with the number of pulses and with the width of
    the clutter, because the polynomial has to follow the variation of the
    clutter over the dwell. In simulated gates with clutter 30 dB
    stronger than the weather and a width of 0.2 m/s, degree 3 suffices for 64
    pulses and degree 6 for 128.
    Spectra of the filtered series have a hole near zero velocity; use
    :func:`notch_filter` on the spectra to fill it.

    References
    ----------
    Torres, S. M., and D. S. Zrnić, 1999: Ground clutter canceling with a
    regression filter. *J. Atmos. Oceanic Technol.*, **16** (10), 1364-1372,
    https://doi.org/10.1175/1520-0426(1999)016<1364:GCCWAR>2.0.CO;2

    Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
    Observations*, 2nd ed. Academic Press (Section 7.10).

    Examples
    --------
    >>> clean = radarx.level1.regression_filter(iq, order=1)  # doctest: +SKIP
    >>> clean_iq = clean["IQ"]  # doctest: +SKIP
    """
    ft.check_object(obj)
    use_compiled = sp.engines(engine)[0]
    kernel = sp.spectra_kernel(use_compiled)
    threads = int(n_threads or 0)
    order = int(order)
    if order < 0:
        raise ValueError("order must not be negative")

    def one(ds, name=None):
        da, parent = iq_array(ds, iq)
        check_complex(da)
        pulse_dimension(da, dim)
        m = da.sizes[dim]
        if order > m - 2:
            raise ValueError(
                f"order {order} leaves fewer than 2 degrees of freedom for {m} pulses"
            )
        basis = polynomial_basis(order, m)
        mask = _mask(where, name)
        arrays, dims_in = [da], [[dim]]
        if mask is not None:
            arrays.append(mask)
            dims_in.append([])

        def block(*blocks):
            lead, flat = flatten(list(blocks), [1] + [0] * (len(blocks) - 1))
            x = np.ascontiguousarray(flat[0], dtype=np.complex128)
            y, removed = kernel.regression_filter(x, basis, n_threads=threads)
            if len(flat) > 1:
                use = flat[1].reshape(-1).astype(bool)
                y = np.where(use[:, None], y, x)
                removed = np.where(use, removed, 0.0)
            return (
                y.reshape(lead + (m,)).astype(da.dtype),
                db(removed).reshape(lead).astype(FIELD_DTYPE),
            )

        out_iq, removed = map_series(
            block, arrays, dims_in, [[dim], []], {}, [da.dtype, FIELD_DTYPE]
        )
        out_iq = out_iq.transpose(*da.dims)
        out_iq.attrs = dict(da.attrs)
        removed.attrs = {
            "long_name": "ground clutter power removed by the regression filter",
            "units": "dB",
            "comment": "mean power in the squared units of the IQ samples",
        }
        return xr.Dataset({"IQ": out_iq, "CLUTTER_POWER": removed})

    if isinstance(obj, xr.DataTree):
        return ft.map_sweeps(obj, one, "regression_filter")
    return one(obj)


@provenance(
    "Spectral notch ground clutter filter with interpolated fill",
    extra_refs=("doviak-zrnic-1993",),
)
def notch_filter(
    obj,
    spectrum=None,
    *,
    dim=VELOCITY,
    half_width=1,
    fill="interpolate",
    noise=None,
    where=None,
    n_sigma=1.0,
    n_average=None,
    min_bins=None,
    n_threads=None,
    engine="auto",
):
    """
    Remove ground clutter from Doppler spectra with a notch at zero velocity.

    The bins within ``half_width`` of the zero velocity bin are replaced; see
    :mod:`radarx.level1.clutter`.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        The spectra (a DataArray with a velocity dimension), a Dataset with
        ``SPECTRUM`` from :func:`~radarx.level1.doppler_spectra`, or a volume
        (sweeps without spectra are skipped).
    spectrum : str, optional
        Name of the spectrum variable. Default: ``SPECTRUM``.
    dim : str, optional
        The velocity dimension. Default ``"velocity"``.
    half_width : int, optional
        Bins on each side of zero velocity that are replaced, so the notch
        has ``2 half_width + 1`` bins. Default 1 (the DC line and one
        adjacent bin on each side).
    fill : {"interpolate", "gaussian", "noise"}, optional
        What replaces the notch bins. Default ``"interpolate"``.
    noise : float, xarray.DataArray or xarray.Dataset, optional
        Total noise power of the series (linear; a DataArray with
        ``units="dB"`` or a Dataset with ``NOISE_POWER`` is read in dB), as
        for :func:`~radarx.level1.spectral_moments`. Default: estimated for
        every spectrum.
    where : xarray.DataArray of bool, optional
        Filter only the gates where this is True. The others are returned
        unchanged. Default: all gates.
    n_sigma : float, optional
        Threshold above the noise level, in standard deviations of the noise
        estimate, for the bins that the Gaussian fit uses. Default 1.
    n_average : float, optional
        Periodograms averaged in a spectrum. By default the attribute of the
        spectra, or 1.
    min_bins : int, optional
        Fewest bins of the noise estimate. Default ``max(3, bins // 8)``.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``SPECTRUM``, the filtered spectra with the attributes of the input,
        and ``CLUTTER_POWER`` (dB), the power that was removed from the notch
        bins (NaN where it is not positive).

    Raises
    ------
    KeyError
        If no spectrum is found.
    ValueError
        For an unknown ``fill``, a negative ``half_width`` or ``n_sigma``, or
        a velocity axis that is not uniform with a zero bin.
    ImportError
        If ``engine="compiled"`` and the compiled kernels are not available.

    References
    ----------
    Guo, H., 2011: A simple algorithm for fitting a Gaussian function [DSP
    tips and tricks]. *IEEE Signal Process. Mag.*, **28** (5), 134-137,
    https://doi.org/10.1109/MSP.2011.941846

    Hildebrand, P. H., and R. S. Sekhon, 1974: Objective determination of the
    noise level in Doppler spectra. *J. Appl. Meteor.*, **13** (7), 808-811,
    https://doi.org/10.1175/1520-0450(1974)013<0808:ODOTNL>2.0.CO;2

    Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
    Observations*, 2nd ed. Academic Press (Section 7.10).

    Examples
    --------
    >>> spectra = radarx.level1.doppler_spectra(iq, window="blackman")  # doctest: +SKIP
    >>> clean = radarx.level1.notch_filter(spectra, half_width=2)  # doctest: +SKIP
    """
    ft.check_object(obj)
    if fill not in FILLS:
        raise ValueError(f"fill must be one of {tuple(FILLS)}, not {fill!r}")
    if half_width < 0:
        raise ValueError("half_width must not be negative")
    if not n_sigma >= 0.0:
        raise ValueError("n_sigma must not be negative")
    use_compiled = sp.engines(engine)[1]
    kernel = sp.moments_kernel(use_compiled)
    threads = int(n_threads or 0)
    lin = sp._noise_linear(noise)

    def one(ds, name=None):
        spec, parent, dv, k = sp._spectrum_fields(ds, spectrum, dim)
        navg = sp._n_average(spec, parent, n_average)
        mb = sp._min_bins(min_bins, k)
        nz = None if lin is None else node_value(lin, name)
        mask = _mask(where, name)
        arrays, dims_in = [spec], [[dim]]
        if nz is not None:
            arrays.append(nz)
            dims_in.append([])
        has_noise = nz is not None
        if mask is not None:
            arrays.append(mask)
            dims_in.append([])

        def block(*blocks):
            lead, flat = flatten(list(blocks), [1] + [0] * (len(blocks) - 1))
            s2 = np.ascontiguousarray(flat[0], dtype=np.float64)
            pos = 1
            if has_noise:
                per_bin = (
                    np.ascontiguousarray(flat[pos].reshape(-1), dtype=np.float64) / k
                )
                pos += 1
            else:
                per_bin = kernel.noise_rows(s2, navg, mb, n_threads=threads)
            out, removed = kernel.notch(
                s2,
                per_bin,
                navg,
                float(n_sigma),
                int(half_width),
                FILLS[fill],
                n_threads=threads,
            )
            if mask is not None:
                use = flat[pos].reshape(-1).astype(bool)
                out = np.where(use[:, None], out, s2)
                removed = np.where(use, removed, np.nan)
            return (
                out.reshape(lead + (k,)).astype(FIELD_DTYPE),
                db(removed).reshape(lead).astype(FIELD_DTYPE),
            )

        filtered, removed = map_series(
            block, arrays, dims_in, [[dim], []], {}, [FIELD_DTYPE, FIELD_DTYPE]
        )
        filtered = filtered.transpose(*spec.dims)
        filtered.attrs = dict(spec.attrs)
        removed.attrs = {
            "long_name": "ground clutter power removed by the notch filter",
            "units": "dB",
            "comment": "in the squared units of the IQ samples",
        }
        out = xr.Dataset({"SPECTRUM": filtered, "CLUTTER_POWER": removed})
        if parent is not None:
            out.attrs = dict(parent.attrs)
        return out

    if isinstance(obj, xr.DataTree):
        return ft.map_sweeps(obj, one, "notch_filter")
    return one(obj)
