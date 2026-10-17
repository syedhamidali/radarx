#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx Doppler Spectra and Spectral Moments
===========================================

From the IQ time series of a gate to the Doppler spectrum and the radar
moments: signal power, mean radial velocity, spectrum width, and the skewness
and kurtosis of the spectrum. Two estimators are offered, the one that works
on the spectrum and the one that works on the autocorrelation of the
time series.

Doppler spectrum
----------------
For a series :math:`x_n` of :math:`L` pulses (:math:`T_s` apart) and a window
:math:`w_n` the power spectrum is the windowed periodogram

.. math::

    P_k = \\frac{\\left|\\sum_{n=0}^{L-1} w_n x_n e^{-2\\pi i kn/L}\\right|^2}
    {L \\sum_n w_n^2},

normalised (Parseval) so that the bins add up to the mean power of the series.
The Doppler frequency of bin :math:`k` is :math:`f = k/(L T_s)` and the radial
velocity :math:`v = -\\lambda f/2` (positive away from the radar), so the
spectrum is returned on the velocity axis :math:`v_j = (j - L/2)\\,\\Delta v`
with :math:`\\Delta v = \\lambda/(2 L T_s)`, from :math:`-v_a` up to
:math:`v_a - \\Delta v`. The windows are the periodic rectangular, Hann,
Hamming and Blackman windows (Harris 1978); a window lowers the leakage of
strong echoes (ground clutter) into the other bins at the price of a wider
spectral resolution. With ``segments`` larger than 1 the dwell is cut into
consecutive blocks whose periodograms are averaged, which lowers the
variance of the estimate by the number of blocks and widens the bins. The FFT
is a radix-2 transform, or the chirp-z transform of Bluestein (1970) if the
length is not a power of two.

Noise
-----
The objective method of Hildebrand and Sekhon (1974) finds the receiver noise
level of a spectrum. The spectral estimates of a noise-only spectrum, averaged
over :math:`n_a` independent periodograms, have :math:`\\mathrm{mean}^2/
\\mathrm{variance} = n_a`. The bins are sorted; the strongest are dropped one at
a time until the remaining ones satisfy that, and their mean is the noise
level per bin. radarx requires at least ``min_bins`` bins (default
:math:`\\max(3, L/8)`) and takes the population variance.

Spectral moments
----------------
The moments are those of the part of the spectrum that is signal. A bin is
signal where :math:`P_k > N(1 + n_\\sigma/\\sqrt{n_a})`, the noise level plus
``n_sigma`` standard deviations of the noise estimate (radarx choice, default
1). With ``region="peak"`` the bins are the contiguous run around the
strongest bin, which excludes spikes and secondary peaks; with
``region="all"`` every bin above the threshold is used. The weights are
:math:`P_k - N` (the noise is subtracted), and the bins are counted from the
strongest one, circularly, so that a spectrum that straddles the Nyquist
velocity is not split in two (Doviak and Zrnic 1993, Eq. 6.24: the sum is
taken around the strongest coefficient and not over the principal interval,
which removes the aliasing bias for symmetric spectra). With :math:`\\Delta_k`
the offset of a bin from the strongest bin, :math:`m_1 = \\sum w_k \\Delta_k /
\\sum w_k` and the central moments :math:`m_i`:

.. math::

    S = \\sum_k (P_k - N), \\quad
    v = (k_0 - L/2 + m_1)\\,\\Delta v, \\quad
    \\sigma_v = \\Delta v\\sqrt{m_2 - m_w},

and the skewness :math:`m_3/m_2^{3/2}` and kurtosis :math:`m_4/m_2^2` (3 for a
Gaussian). The mean velocity is returned in the Nyquist interval
:math:`[-v_a, v_a)`. A window smooths the spectrum with its own power response,
which adds the variance :math:`m_w` to the observed one (in squared bins; the
variance of the response is :math:`\\sum_n (w_{n+1} - w_n)^2 / (4\\pi^2 \\sum_n
w_n^2)` in squared cycles per sample, 1/3 bin\\ :sup:`2` for the Hann window);
``window_correction=True`` (default) takes it off. The rectangular window is not
corrected. The signal-to-noise ratio is :math:`S` divided by the total noise
power :math:`L N`.

Pulse pair moments
------------------
In the time domain the autocorrelation of the series at lag :math:`m`,
:math:`R_m = \\frac{1}{M-m}\\sum_n x_n^* x_{n+m}`, gives the moments without a
Fourier transform (Doviak and Zrnic 1993, Section 6.4.1 and 6.5.1):

.. math::

    v = -\\frac{\\lambda}{4\\pi T_s}\\arg R_1, \\qquad
    \\sigma_v = \\frac{\\lambda}{2\\sqrt{2}\\,\\pi T_s}
    \\sqrt{\\ln\\frac{S}{|R_1|}}\\;\\mathrm{sgn}\\left(\\ln\\frac{S}{|R_1|}\\right),

with the signal power :math:`S = R_0 - N` (Eq. 6.18, 6.19, 6.27 and 6.28), and
for the width from the lags 1 and 2 (Srivastava et al. 1979, Doviak and Zrnic
1993 Eq. 6.32) :math:`\\sigma_v = \\lambda/(2\\sqrt{6}\\,\\pi T_s)\\,
\\sqrt{|\\ln|R_1/R_2||}\\,\\mathrm{sgn}(\\ln|R_1/R_2|)`. Both widths assume a
Gaussian spectrum, and a negative argument of the root, possible at narrow
widths and low SNR, gives zero width. The mean velocity is unbiased by
aliasing of a symmetric spectrum and by white noise, which does not change the
phase of :math:`R_1`; the width and power need the noise power.

Dealiasing with the spectra
---------------------------
A velocity from one pulse repetition time is known only modulo :math:`2v_a`.
:func:`dealias_spectra` takes the mean velocity of the spectral peak, which is
not biased by a spectrum that straddles the Nyquist velocity, and brings it to
the Nyquist interval around a reference velocity (a model wind or a
profile) or, without one, around the last unfolded gate of the ray.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = [
    "doppler_spectra",
    "spectral_noise",
    "spectral_moments",
    "pulse_pair_moments",
    "dealias_spectra",
]

__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

import numpy as np
import xarray as xr

from .._dtypes import FIELD_DTYPE
from .._provenance import provenance
from ..retrieve import _fieldtools as ft
from . import _moments_numpy, _spectra_numpy
from ._common import (
    PULSE,
    RANGE,
    VELOCITY,
    check_complex,
    db,
    flatten,
    iq_array,
    map_series,
    node_value,
    pulse_dimension,
    radar_constants,
    spectrum_array,
    velocity_axis,
)

try:
    from . import _spectra

    HAS_COMPILED_SPECTRA = True
except ImportError:  # pragma: no cover - depends on the build
    _spectra = None
    HAS_COMPILED_SPECTRA = False

try:
    from . import _moments

    HAS_COMPILED_MOMENTS = True
except ImportError:  # pragma: no cover - depends on the build
    _moments = None
    HAS_COMPILED_MOMENTS = False

HAS_COMPILED_KERNEL = HAS_COMPILED_SPECTRA and HAS_COMPILED_MOMENTS

WINDOWS = ("rect", "hann", "hamming", "blackman")
REGIONS = {"peak": 0, "all": 1}
WIDTH_METHODS = ("r0r1", "r1r2")


def engines(engine):
    """Which kernels to use for the requested ``engine``: (spectra, moments)."""
    ft.choose_engine(engine, HAS_COMPILED_KERNEL, "level1")
    return (
        HAS_COMPILED_SPECTRA and engine != "numpy",
        HAS_COMPILED_MOMENTS and engine != "numpy",
    )


def spectra_kernel(use_compiled):
    return _spectra if use_compiled else _spectra_numpy


def moments_kernel(use_compiled):
    return _moments if use_compiled else _moments_numpy


def make_window(name, length):
    """
    Periodic window of ``length`` samples and its variance in squared bins.

    The variance is that of the power response of the window, a quantity of the
    smoothing that the window applies to the spectrum; zero for ``"rect"``,
    which is not corrected.
    """
    if name not in WINDOWS:
        raise ValueError(f"window must be one of {WINDOWS}, not {name!r}")
    x = 2.0 * np.pi * np.arange(length) / length
    if name == "rect":
        return np.ones(length), 0.0
    if name == "hann":
        w = 0.5 - 0.5 * np.cos(x)
    elif name == "hamming":
        w = 0.54 - 0.46 * np.cos(x)
    else:
        w = 0.42 - 0.5 * np.cos(x) + 0.08 * np.cos(2.0 * x)
    dw = np.roll(w, -1) - w
    var = float(np.sum(dw * dw) / (4.0 * np.pi**2 * np.sum(w * w)) * length**2)
    return w, var


def _threads(n_threads):
    return int(n_threads or 0)


def _noise_linear(noise, name=None):
    """A noise argument as linear power: number, DataArray or None."""
    if noise is None:
        return None
    if isinstance(noise, xr.Dataset):
        if "NOISE_POWER" not in noise:
            raise ValueError("a noise Dataset needs the variable NOISE_POWER")
        noise = noise["NOISE_POWER"]
    if isinstance(noise, xr.DataArray):
        if noise.attrs.get("units") == "dB":
            return 10.0 ** (0.1 * noise.astype(np.float64))
        return noise.astype(np.float64)
    value = float(noise)
    if not (np.isfinite(value) and value >= 0.0):
        raise ValueError(f"noise must be a non-negative power, not {noise!r}")
    return xr.DataArray(value)


def _check_options(region, n_sigma):
    if region not in REGIONS:
        raise ValueError(f"region must be 'peak' or 'all', not {region!r}")
    if not n_sigma >= 0.0:
        raise ValueError("n_sigma must not be negative")


def _n_average(spec, parent, n_average):
    if n_average is None:
        n_average = spec.attrs.get("n_average")
        if n_average is None and parent is not None:
            n_average = parent.attrs.get("n_average")
    n_average = 1.0 if n_average is None else float(n_average)
    if not n_average >= 1.0:
        raise ValueError("n_average must be at least 1")
    return n_average


def _window_variance(spec, parent, correction):
    if not correction:
        return 0.0
    var = spec.attrs.get("window_variance_bins2")
    if var is None and parent is not None:
        var = parent.attrs.get("window_variance_bins2")
    return 0.0 if var is None else float(var)


def _min_bins(min_bins, k):
    if min_bins is None:
        min_bins = max(3, k // 8)
    min_bins = int(min_bins)
    if not 1 <= min_bins <= k:
        raise ValueError(f"min_bins must be between 1 and the {k} bins")
    return min_bins


def _channel(channel):
    if channel not in ("H", "V"):
        raise ValueError(f"channel must be 'H' or 'V', not {channel!r}")
    return channel


# --------------------------------------------------------------------------
# Doppler spectra
# --------------------------------------------------------------------------


def _spectra_block(window, n_seg, use_compiled, n_threads):
    kernel = spectra_kernel(use_compiled)

    def block(x):
        lead, (flat,) = flatten([x], [1])
        flat = np.ascontiguousarray(flat, dtype=np.complex128)
        out = kernel.power_spectra(flat, window, n_seg, n_threads=n_threads)
        return out.reshape(lead + (out.shape[-1],)).astype(FIELD_DTYPE)

    return block


@provenance(
    "Windowed periodogram Doppler spectrum of IQ time series",
    extra_refs=("doviak-zrnic-1993",),
)
def doppler_spectra(
    obj,
    iq=None,
    *,
    dim=PULSE,
    window="hann",
    segments=1,
    wavelength=None,
    prt=None,
    n_threads=None,
    engine="auto",
):
    """
    Doppler power spectra of IQ time series.

    The windowed periodogram of the pulses of every gate on the velocity axis
    (positive away from the radar), normalised so that the bins add up to the
    mean power; see :mod:`radarx.level1.spectra`.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        The complex IQ samples with a pulse dimension (a DataArray), a sweep
        that holds them, or a volume (sweep by sweep; sweeps without IQ are
        skipped).
    iq : str, optional
        Name of the IQ variable of a sweep. By default the first of ``IQ``,
        ``iq``, ``IQH`` and ``IQV``.
    dim : str, optional
        The pulse (slow time) dimension. Default ``"pulse"``.
    window : {"hann", "rect", "hamming", "blackman"}, optional
        Window of the periodogram. Default ``"hann"`` (radarx choice; the
        von Hann window is the one of Doviak and Zrnic 1993, Fig. 7.31).
    segments : int, optional
        Number of consecutive blocks of ``M // segments`` pulses whose
        periodograms are averaged. The pulses after the last full block are
        not used. Default 1.
    wavelength : float, optional
        Radar wavelength (m). By default the ``wavelength`` attribute of the
        IQ array, or the sweep's ``wavelength`` or ``radar_frequency``.
    prt : float, optional
        Pulse repetition time (s). By default the ``prt`` attribute or
        variable.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` prefers the compiled kernel and
        falls back to NumPy.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``SPECTRUM``, the power per bin in the squared units of the samples,
        with the dimensions of the IQ array without the pulse dimension and a
        ``velocity`` dimension (ascending, m s-1, bin ``L // 2`` is zero).
        The attributes hold ``wavelength``, ``prt``, ``nyquist_velocity``,
        ``window``, ``n_average`` (``segments``) and
        ``window_variance_bins2``, which the other functions read.

    Raises
    ------
    KeyError
        If a sweep has no IQ variable (a volume where no sweep has one).
    ValueError
        If the samples are not complex, the pulse dimension is missing, there
        are fewer than 3 pulses per segment, or the wavelength or pulse
        repetition time is not known.
    ImportError
        If ``engine="compiled"`` and the compiled kernels are not available.

    Notes
    -----
    The radar moments are estimated from the spectra with
    :func:`spectral_moments`. Strong ground clutter leaks into the bins
    around zero velocity through the window; a window with lower sidelobes
    (``"blackman"``) lowers the leakage.

    References
    ----------
    Harris, F., 1978: On the use of windows for harmonic analysis with the
    discrete Fourier transform. *Proc. IEEE*, **66** (1), 51-83,
    https://doi.org/10.1109/PROC.1978.10837

    Bluestein, L., 1970: A linear filtering approach to the computation of
    discrete Fourier transform. *IEEE Trans. Audio Electroacoust.*, **18** (4),
    451-455, https://doi.org/10.1109/TAU.1970.1162132

    Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
    Observations*, 2nd ed. Academic Press (Chapters 5 and 6).

    Examples
    --------
    >>> spectra = radarx.level1.doppler_spectra(iq, window="blackman")  # doctest: +SKIP
    """
    ft.check_object(obj)
    use_compiled = engines(engine)[0]
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {WINDOWS}, not {window!r}")
    segments = int(segments)
    if segments < 1:
        raise ValueError("segments must be at least 1")

    def one(ds, _name=None):
        da, parent = iq_array(ds, iq)
        check_complex(da)
        pulse_dimension(da, dim)
        wl, tr = radar_constants(da, parent, wavelength, prt)
        length = da.sizes[dim] // segments
        if length < 3:
            raise ValueError(
                f"{da.sizes[dim]} pulses in {segments} segments leave fewer than 3 per segment"
            )
        win, win_var = make_window(window, length)
        block = _spectra_block(win, segments, use_compiled, _threads(n_threads))
        out = map_series(
            block, [da], [[dim]], [[VELOCITY]], {VELOCITY: length}, [FIELD_DTYPE]
        )
        dv = wl / (2.0 * length * tr)
        out = out.assign_coords(
            {
                VELOCITY: (
                    VELOCITY,
                    (np.arange(length) - length // 2) * dv,
                    {
                        "long_name": "radial velocity of the spectral bin",
                        "units": "m s-1",
                        "positive": "away from the radar",
                    },
                )
            }
        )
        out.name = "SPECTRUM"
        attrs = {
            "wavelength": wl,
            "prt": tr,
            "nyquist_velocity": wl / (4.0 * tr),
            "window": window,
            "n_average": float(segments),
            "window_variance_bins2": win_var,
        }
        out.attrs = {
            "long_name": "Doppler power spectrum",
            "units": "1",
            "comment": "power per bin in the squared units of the IQ samples; the bins add up to the mean power",
            **attrs,
        }
        return xr.Dataset({"SPECTRUM": out}, attrs=attrs)

    if isinstance(obj, xr.DataTree):
        return ft.map_sweeps(obj, one, "doppler_spectra")
    return one(obj)


# --------------------------------------------------------------------------
# Noise and moments of spectra
# --------------------------------------------------------------------------


def _spectrum_fields(obj, spectrum, dim):
    """The spectrum DataArray, its Dataset, the bin width and the bin count."""
    spec = spectrum_array(obj, spectrum)
    parent = None if isinstance(obj, xr.DataArray) else obj
    dv = velocity_axis(spec, dim)
    return spec, parent, dv, spec.sizes[dim]


@provenance(
    "Objective noise level of Doppler spectra (Hildebrand-Sekhon)",
    extra_refs=("doviak-zrnic-1993",),
)
def spectral_noise(
    obj,
    spectrum=None,
    *,
    dim=VELOCITY,
    n_average=None,
    min_bins=None,
    n_threads=None,
    engine="auto",
):
    """
    Noise power of Doppler spectra by the objective method of Hildebrand and Sekhon.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        The spectra (a DataArray with a velocity dimension), a Dataset with
        ``SPECTRUM`` as returned by :func:`doppler_spectra`, or a volume of
        such sweeps (sweeps without spectra are skipped).
    spectrum : str, optional
        Name of the spectrum variable. Default: ``SPECTRUM``.
    dim : str, optional
        The velocity dimension. Default ``"velocity"``.
    n_average : float, optional
        Number of independent periodograms averaged in a spectrum. By default
        the ``n_average`` attribute written by :func:`doppler_spectra`, or 1.
    min_bins : int, optional
        Fewest bins that make the noise. Default ``max(3, bins // 8)``.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``NOISE_POWER`` in dB (``units="dB"``), the total noise power of the
        series (the noise per bin times the number of bins), in the squared
        units of the IQ samples. It can be passed as ``noise`` to
        :func:`spectral_moments` and :func:`pulse_pair_moments`. NaN for a
        spectrum with a non-finite bin.

    Raises
    ------
    KeyError
        If no spectrum is found.
    ValueError
        For an invalid velocity axis, ``n_average`` or ``min_bins``.
    ImportError
        If ``engine="compiled"`` and the compiled kernels are not available.

    References
    ----------
    Hildebrand, P. H., and R. S. Sekhon, 1974: Objective determination of the
    noise level in Doppler spectra. *J. Appl. Meteor.*, **13** (7), 808-811,
    https://doi.org/10.1175/1520-0450(1974)013<0808:ODOTNL>2.0.CO;2

    Examples
    --------
    >>> noise = radarx.level1.spectral_noise(spectra)  # doctest: +SKIP
    """
    ft.check_object(obj)
    use_compiled = engines(engine)[1]
    kernel = moments_kernel(use_compiled)
    threads = _threads(n_threads)

    def one(ds, _name=None):
        spec, parent, _dv, k = _spectrum_fields(ds, spectrum, dim)
        navg = _n_average(spec, parent, n_average)
        mb = _min_bins(min_bins, k)

        def block(s):
            lead, (flat,) = flatten([s], [1])
            noise = kernel.noise_rows(
                np.ascontiguousarray(flat, dtype=np.float64),
                navg,
                mb,
                n_threads=threads,
            )
            return db(noise * k).reshape(lead).astype(FIELD_DTYPE)

        out = map_series(block, [spec], [[dim]], [[]], {}, [FIELD_DTYPE])
        out.name = "NOISE_POWER"
        out.attrs = {
            "long_name": "noise power of the Doppler spectrum",
            "units": "dB",
            "comment": "total over the bins, in the squared units of the IQ samples",
        }
        return xr.Dataset({"NOISE_POWER": out})

    if isinstance(obj, xr.DataTree):
        return ft.map_sweeps(obj, one, "spectral_noise")
    return one(obj)


def _moments_fields(
    spec,
    parent,
    dv,
    k,
    noise,
    region,
    n_sigma,
    n_average,
    min_bins,
    correction,
    min_snr,
    channel,
    use_compiled,
    n_threads,
    dim,
):
    """The moments of the spectra of one sweep as a dict of DataArrays."""
    navg = _n_average(spec, parent, n_average)
    mb = _min_bins(min_bins, k)
    win_var = _window_variance(spec, parent, correction)
    va = 0.5 * k * dv
    mode = REGIONS[region]
    kernel = moments_kernel(use_compiled)
    arrays = [spec]
    dims_in = [[dim]]
    if noise is not None:
        arrays.append(noise)
        dims_in.append([])

    def block(*blocks):
        lead, flat = flatten(list(blocks), [1] + [0] * (len(blocks) - 1))
        s2 = np.ascontiguousarray(flat[0], dtype=np.float64)
        if noise is None:
            per_bin = kernel.noise_rows(s2, navg, mb, n_threads=n_threads)
        else:
            per_bin = np.ascontiguousarray(flat[1].reshape(-1), dtype=np.float64) / k
        res = kernel.moments(
            s2, per_bin, navg, float(n_sigma), mode, dv, win_var, n_threads=n_threads
        )
        total_noise = res[:, 0] * k
        power = res[:, 1]
        vel = ((res[:, 2] + va) % (2.0 * va)) - va
        with np.errstate(invalid="ignore", divide="ignore"):
            snr = db(power / total_noise)
        width, skew, kurt = res[:, 3], res[:, 4], res[:, 5]
        if min_snr is not None:
            weak = ~(snr >= min_snr)
            vel, width, skew, kurt = (
                np.where(weak, np.nan, a) for a in (vel, width, skew, kurt)
            )
        outs = (db(power), db(total_noise), snr, vel, width, skew, kurt)
        return tuple(o.reshape(lead).astype(FIELD_DTYPE) for o in outs)

    res = map_series(block, arrays, dims_in, [[]] * 7, {}, [FIELD_DTYPE] * 7)
    names = (
        "SIGNAL_POWER",
        "NOISE_POWER",
        "SNR",
        f"VRAD{channel}",
        f"WRAD{channel}",
        "SKEW",
        "KURT",
    )
    return dict(zip(names, res)), va


_MOMENT_ATTRS = {
    "SIGNAL_POWER": ("signal power of the spectrum", "dB", None),
    "NOISE_POWER": ("noise power of the spectrum", "dB", None),
    "SNR": ("signal-to-noise ratio", "dB", "signal_to_noise_ratio"),
    "SKEW": ("skewness of the Doppler spectrum", "1", None),
    "KURT": ("kurtosis of the Doppler spectrum", "1", None),
}


def _moment_dataset(fields, va, channel, spec):
    ds = xr.Dataset(fields)
    for name, da in ds.data_vars.items():
        if name in _MOMENT_ATTRS:
            long_name, units, standard = _MOMENT_ATTRS[name]
            da.attrs = {"long_name": long_name, "units": units}
            if standard:
                da.attrs["standard_name"] = standard
        elif name.startswith("VRAD"):
            da.attrs = {
                "long_name": "radial velocity from the Doppler spectrum",
                "standard_name": "radial_velocity_of_scatterers_away_from_instrument",
                "units": "m s-1",
                "positive": "away from the radar",
                "valid_range": [-va, va],
            }
        else:
            da.attrs = {
                "long_name": "spectrum width from the Doppler spectrum",
                "units": "m s-1",
            }
    SKEW_KURT = ("SKEW", "KURT")
    for name in SKEW_KURT:
        ds[name].attrs["comment"] = (
            "kurtosis of a Gaussian is 3"
            if name == "KURT"
            else "third standardised moment"
        )
    ds.attrs = {"nyquist_velocity": va}
    return ds


@provenance(
    "Spectral moments of Doppler spectra (noise-subtracted peak moments)",
    extra_refs=("doviak-zrnic-1993",),
)
def spectral_moments(
    obj,
    spectrum=None,
    *,
    dim=VELOCITY,
    noise=None,
    region="peak",
    n_sigma=1.0,
    n_average=None,
    min_bins=None,
    window_correction=True,
    min_snr=None,
    channel="H",
    n_threads=None,
    engine="auto",
):
    """
    Radar moments from Doppler spectra: power, velocity, width, skewness, kurtosis.

    The noise level of every spectrum is found with the method of Hildebrand
    and Sekhon (or given), the bins above the noise form the signal, and the
    moments of the noise-subtracted signal around the strongest bin give the
    power, the mean velocity, the spectrum width, and the skewness and
    kurtosis. See :mod:`radarx.level1.spectra` for the definitions and the
    radarx choices.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        The spectra (a DataArray with a velocity dimension), a Dataset with
        ``SPECTRUM`` from :func:`doppler_spectra`, or a volume of such sweeps
        (sweeps without spectra are skipped).
    spectrum : str, optional
        Name of the spectrum variable. Default: ``SPECTRUM``.
    dim : str, optional
        The velocity dimension. Default ``"velocity"``.
    noise : float, xarray.DataArray or xarray.Dataset, optional
        Total noise power of the series, in the units of the signal power
        (linear), as a number or an array that broadcasts against the
        spectra. A DataArray with ``units="dB"`` or a Dataset with
        ``NOISE_POWER`` (from :func:`spectral_noise`) is read in dB. Default
        None: the noise of every spectrum is estimated.
    region : {"peak", "all"}, optional
        The bins of the moments: the contiguous run above the threshold
        around the strongest bin (default), or all bins above the threshold.
    n_sigma : float, optional
        The threshold above the noise level, in standard deviations of the
        noise estimate (the noise divided by the square root of the number of
        averaged periodograms). Default 1.
    n_average : float, optional
        Periodograms averaged in a spectrum. By default the attribute of the
        spectra, or 1.
    min_bins : int, optional
        Fewest bins of the noise estimate. Default ``max(3, bins // 8)``.
    window_correction : bool, optional
        Take the variance that the window adds to the spectrum off the
        width. Default True; the rectangular window is not corrected.
    min_snr : float, optional
        Signal-to-noise ratio (dB) below which the velocity, width, skewness
        and kurtosis are NaN. A noise-only spectrum has a strongest bin that
        passes the threshold of the bins by chance and gives a random
        velocity; this removes those. Default None: no limit.
    channel : {"H", "V"}, optional
        Polarisation channel of the names ``VRADH``/``WRADH`` (``V`` gives
        ``VRADV``/``WRADV``). Default ``"H"``.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``SIGNAL_POWER`` and ``NOISE_POWER`` (dB, in the squared units of the
        IQ samples), ``SNR`` (dB), ``VRADH`` (m s-1, in the Nyquist interval,
        positive away from the radar), ``WRADH`` (m s-1), ``SKEW`` and
        ``KURT``. NaN where the spectrum has no bin above the threshold or a
        non-finite bin (``SIGNAL_POWER`` and the moments), and the moments
        where the SNR is below ``min_snr``; a single bin has zero width and no
        skewness or kurtosis.

    Raises
    ------
    KeyError
        If no spectrum is found.
    ValueError
        For unknown ``region`` or ``channel``, a negative ``n_sigma`` or
        ``noise``, or a velocity axis that is not uniform with a zero bin.
    ImportError
        If ``engine="compiled"`` and the compiled kernels are not available.

    Notes
    -----
    A spectrum that straddles the Nyquist velocity gives the unbiased mean
    velocity within the Nyquist interval; a mean velocity beyond it is
    aliased and is brought back by :func:`dealias_spectra`.

    References
    ----------
    Hildebrand, P. H., and R. S. Sekhon, 1974: Objective determination of the
    noise level in Doppler spectra. *J. Appl. Meteor.*, **13** (7), 808-811,
    https://doi.org/10.1175/1520-0450(1974)013<0808:ODOTNL>2.0.CO;2

    Harris, F., 1978: On the use of windows for harmonic analysis with the
    discrete Fourier transform. *Proc. IEEE*, **66** (1), 51-83,
    https://doi.org/10.1109/PROC.1978.10837

    Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
    Observations*, 2nd ed. Academic Press (Section 6.4.2, Eq. 6.24).

    Examples
    --------
    >>> moments = radarx.level1.spectral_moments(spectra)  # doctest: +SKIP
    >>> moments = radarx.level1.spectral_moments(spectra, region="all", n_sigma=3)  # doctest: +SKIP
    """
    ft.check_object(obj)
    _check_options(region, n_sigma)
    channel = _channel(channel)
    use_compiled = engines(engine)[1]
    threads = _threads(n_threads)
    lin = _noise_linear(noise)

    def one(ds, name=None):
        spec, parent, dv, k = _spectrum_fields(ds, spectrum, dim)
        nz = None if lin is None else node_value(lin, name)
        fields, va = _moments_fields(
            spec,
            parent,
            dv,
            k,
            nz,
            region,
            n_sigma,
            n_average,
            min_bins,
            window_correction,
            min_snr,
            channel,
            use_compiled,
            threads,
            dim,
        )
        return _moment_dataset(fields, va, channel, spec)

    if isinstance(obj, xr.DataTree):
        return ft.map_sweeps(obj, one, "spectral_moments")
    return one(obj)


# --------------------------------------------------------------------------
# Pulse pair moments
# --------------------------------------------------------------------------


def _pulse_pair_block(
    wl, tr, noise_mode, method, min_bins, use_spec, use_mom, n_threads
):
    spectra = spectra_kernel(use_spec)
    moments = moments_kernel(use_mom)

    def block(*blocks):
        lead, flat = flatten(list(blocks), [1] + [0] * (len(blocks) - 1))
        x = np.ascontiguousarray(flat[0], dtype=np.complex128)
        m = x.shape[1]
        sums = spectra.lag_sums(x, n_threads=n_threads)
        p0, r1, r2 = (
            sums[:, 0],
            sums[:, 1] + 1j * sums[:, 2],
            sums[:, 3] + 1j * sums[:, 4],
        )
        if noise_mode == "spectral":
            win, _ = make_window("hann", m)
            spec = spectra.power_spectra(x, win, 1, n_threads=n_threads)
            mb = _min_bins(min_bins, m)
            noise = moments.noise_rows(spec, 1.0, mb, n_threads=n_threads) * m
        elif noise_mode == "given":
            noise = np.ascontiguousarray(flat[1].reshape(-1), dtype=np.float64)
        else:
            noise = None
        with np.errstate(invalid="ignore", divide="ignore"):
            sig = p0 if noise is None else p0 - noise
            sig = np.where(sig > 0.0, sig, np.nan)
            vel = -wl / (4.0 * np.pi * tr) * np.angle(r1)
            if method == "r0r1":
                ratio = np.log(sig / np.abs(r1))
                scale = wl / (2.0 * np.sqrt(2.0) * np.pi * tr)
            else:
                ratio = np.log(np.abs(r1) / np.abs(r2))
                scale = wl / (2.0 * np.sqrt(6.0) * np.pi * tr)
            width = scale * np.sqrt(np.abs(ratio)) * np.where(ratio >= 0.0, 1.0, 0.0)
            width = np.where(np.isfinite(ratio), width, np.nan)
        outs = [db(sig), vel, width]
        if noise is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                outs += [db(noise), db(sig / noise)]
        return tuple(o.reshape(lead).astype(FIELD_DTYPE) for o in outs)

    return block


@provenance(
    "Pulse pair moments of IQ time series",
    extra_refs=("doviak-zrnic-1993",),
)
def pulse_pair_moments(
    obj,
    iq=None,
    *,
    dim=PULSE,
    noise=None,
    width_method="r0r1",
    wavelength=None,
    prt=None,
    channel="H",
    n_threads=None,
    engine="auto",
):
    """
    Radar moments from the autocorrelation of IQ time series (pulse pair).

    The mean velocity is the phase of the lag-1 autocorrelation, the spectrum
    width comes from the ratio of the power to the lag-1 magnitude or of the
    lag-1 to the lag-2 magnitudes, assuming a Gaussian spectrum; see
    :mod:`radarx.level1.spectra`. This needs no Fourier transform and is the
    estimator used when a spectrum is not computed, but it cannot separate
    clutter from weather by their spectra.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        The complex IQ samples, a sweep that holds them, or a volume.
    iq : str, optional
        Name of the IQ variable of a sweep. Default: ``IQ``, ``iq``, ``IQH``
        or ``IQV``.
    dim : str, optional
        The pulse dimension. Default ``"pulse"``.
    noise : float, xarray.DataArray, xarray.Dataset or "spectral", optional
        Noise power per sample, linear, in the units of the signal power, as
        a number or an array that broadcasts against the gates (a DataArray
        with ``units="dB"`` or a Dataset with ``NOISE_POWER`` is read in
        dB). ``"spectral"`` finds it for every series with the method of
        Hildebrand and Sekhon on the Hann-windowed periodogram. Default None:
        no noise is subtracted, and ``SIGNAL_POWER`` is the total power.
    width_method : {"r0r1", "r1r2"}, optional
        The spectrum width from the lags 0 and 1 (default; needs the noise
        power for a good result) or from the lags 1 and 2 (independent of the
        noise).
    wavelength : float, optional
        Radar wavelength (m). By default the attribute of the IQ array, or
        the sweep's ``wavelength`` or ``radar_frequency``.
    prt : float, optional
        Pulse repetition time (s). By default the attribute or variable.
    channel : {"H", "V"}, optional
        Polarisation channel of the names ``VRADH``/``WRADH``. Default
        ``"H"``.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``SIGNAL_POWER`` (dB), ``VRADH`` (m s-1, in the Nyquist interval,
        positive away from the radar) and ``WRADH`` (m s-1); with ``noise``
        also ``NOISE_POWER`` and ``SNR`` (dB). The width is zero where the
        estimate is negative (narrow spectra at low SNR) and NaN where the
        signal power is not positive.

    Raises
    ------
    KeyError
        If no IQ variable is found.
    ValueError
        For unknown ``width_method``, ``noise`` or ``channel``, samples that
        are not complex, or unknown wavelength or pulse repetition time.
    ImportError
        If ``engine="compiled"`` and the compiled kernels are not available.

    References
    ----------
    Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
    Observations*, 2nd ed. Academic Press (Sections 6.4.1 and 6.5.1,
    Eq. 6.18, 6.19, 6.27, 6.28 and 6.32).

    Srivastava, R. C., A. R. Jameson, and P. H. Hildebrand, 1979:
    Time-domain computation of mean and variance of Doppler spectra.
    *J. Appl. Meteor.*, **18** (2), 189-194,
    https://doi.org/10.1175/1520-0450(1979)018<0189:TDCOMA>2.0.CO;2

    Zrnić, D. S., 1977: Spectral moment estimates from correlated pulse
    pairs. *IEEE Trans. Aerosp. Electron. Syst.*, **AES-13** (4), 344-354,
    https://doi.org/10.1109/TAES.1977.308467

    Hildebrand, P. H., and R. S. Sekhon, 1974: Objective determination of the
    noise level in Doppler spectra. *J. Appl. Meteor.*, **13** (7), 808-811,
    https://doi.org/10.1175/1520-0450(1974)013<0808:ODOTNL>2.0.CO;2

    Examples
    --------
    >>> moments = radarx.level1.pulse_pair_moments(iq, noise="spectral")  # doctest: +SKIP
    """
    ft.check_object(obj)
    channel = _channel(channel)
    if width_method not in WIDTH_METHODS:
        raise ValueError(
            f"width_method must be one of {WIDTH_METHODS}, not {width_method!r}"
        )
    if isinstance(noise, str):
        if noise != "spectral":
            raise ValueError("noise must be a power, an array, or 'spectral'")
        lin, mode = None, "spectral"
    else:
        lin = _noise_linear(noise)
        mode = "none" if lin is None else "given"
    use_spec, use_mom = engines(engine)
    threads = _threads(n_threads)

    def one(ds, name=None):
        da, parent = iq_array(ds, iq)
        check_complex(da)
        pulse_dimension(da, dim)
        wl, tr = radar_constants(da, parent, wavelength, prt)
        arrays, dims_in = [da], [[dim]]
        if mode == "given":
            arrays.append(node_value(lin, name))
            dims_in.append([])
        block = _pulse_pair_block(
            wl, tr, mode, width_method, None, use_spec, use_mom, threads
        )
        n_out = 5 if mode != "none" else 3
        res = map_series(
            block, arrays, dims_in, [[]] * n_out, {}, [FIELD_DTYPE] * n_out
        )
        names = [
            "SIGNAL_POWER",
            f"VRAD{channel}",
            f"WRAD{channel}",
            "NOISE_POWER",
            "SNR",
        ]
        ds_out = xr.Dataset(dict(zip(names, res)))
        va = wl / (4.0 * tr)
        ds_out["SIGNAL_POWER"].attrs = {
            "long_name": (
                "total power of the IQ time series"
                if mode == "none"
                else "signal power of the IQ time series (noise subtracted)"
            ),
            "units": "dB",
        }
        ds_out[f"VRAD{channel}"].attrs = {
            "long_name": "radial velocity from the pulse pair",
            "standard_name": "radial_velocity_of_scatterers_away_from_instrument",
            "units": "m s-1",
            "positive": "away from the radar",
            "valid_range": [-va, va],
        }
        ds_out[f"WRAD{channel}"].attrs = {
            "long_name": "spectrum width from the pulse pair",
            "units": "m s-1",
        }
        if mode != "none":
            ds_out["NOISE_POWER"].attrs = {
                "long_name": "noise power of the IQ time series",
                "units": "dB",
            }
            ds_out["SNR"].attrs = {
                "long_name": "signal-to-noise ratio",
                "standard_name": "signal_to_noise_ratio",
                "units": "dB",
            }
        ds_out.attrs = {"nyquist_velocity": va}
        return ds_out

    if isinstance(obj, xr.DataTree):
        return ft.map_sweeps(obj, one, "pulse_pair_moments")
    return one(obj)


# --------------------------------------------------------------------------
# Dealiasing with the spectra
# --------------------------------------------------------------------------


@provenance(
    "Dealiasing of Doppler spectra by radial continuity or a reference velocity",
    extra_refs=("doviak-zrnic-1993",),
)
def dealias_spectra(
    obj,
    reference=None,
    *,
    spectrum=None,
    dim=VELOCITY,
    max_gap=3,
    noise=None,
    region="peak",
    n_sigma=1.0,
    n_average=None,
    min_bins=None,
    window_correction=True,
    min_snr=3.0,
    channel="H",
    n_threads=None,
    engine="auto",
):
    """
    Unfolded mean velocity of Doppler spectra.

    The mean velocity of the spectral peak, taken around the strongest bin so
    that a spectrum that straddles the Nyquist velocity is not split (see
    :func:`spectral_moments`), is known only within the Nyquist interval.
    This brings it to the interval around a reference velocity, or, where
    there is no reference, around the last unfolded gate of the ray.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        The spectra, as for :func:`spectral_moments`. Without a reference they
        need a ``range`` dimension.
    reference : float, xarray.DataArray, optional
        Reference radial velocity (m s-1) of every gate, such as the radial
        component of a model or profile wind. A number or an array that
        broadcasts against the gates. A gate with a reference is unfolded
        towards it; a gate without (NaN) follows its neighbour along the
        ray. For a volume, a DataTree or dict gives one reference per sweep.
    spectrum : str, optional
        Name of the spectrum variable. Default: ``SPECTRUM``.
    dim : str, optional
        The velocity dimension. Default ``"velocity"``.
    max_gap : int, optional
        Number of gates without a valid velocity that still link two gates
        in the continuity along the ray; beyond it the velocity restarts in
        the Nyquist interval. Default 3.
    min_snr : float, optional
        Gates with a lower signal-to-noise ratio (dB) are not trusted: their
        velocity is NaN and they do not take part in the continuity along the
        ray. Default 3 dB (radarx choice); a noise spectrum gives a random
        velocity that would carry a wrong fold to the next gate. None keeps
        all gates.
    noise, region, n_sigma, n_average, min_bins, window_correction, channel :
        As in :func:`spectral_moments`.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``VRADH`` (m s-1), the unfolded velocity, and ``FOLD`` (``int8``), the
        number of Nyquist intervals (``2 v_a`` each) that were added.

    Raises
    ------
    KeyError
        If no spectrum is found.
    ValueError
        If there is no reference and the spectra have no ``range``
        dimension, ``max_gap`` is negative, or an option is invalid.
    ImportError
        If ``engine="compiled"`` and the compiled kernels are not available.

    Notes
    -----
    Without a reference the first valid gate of a ray stays in the Nyquist
    interval, so continuity carries a wrong first gate along the ray. A
    reference from a wind profile or a model removes that dependence.

    References
    ----------
    Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
    Observations*, 2nd ed. Academic Press (Section 6.4.2: the spectral mean
    around the strongest coefficient).

    Examples
    --------
    >>> unfolded = radarx.level1.dealias_spectra(spectra)  # doctest: +SKIP
    >>> unfolded = radarx.level1.dealias_spectra(spectra, reference=wind_radial)  # doctest: +SKIP
    """
    ft.check_object(obj)
    _check_options(region, n_sigma)
    channel = _channel(channel)
    if max_gap < 0:
        raise ValueError("max_gap must not be negative")
    use_compiled = engines(engine)[1]
    threads = _threads(n_threads)
    lin = _noise_linear(noise)
    kernel = moments_kernel(use_compiled)

    def one(ds, name=None):
        spec, parent, dv, k = _spectrum_fields(ds, spectrum, dim)
        if RANGE not in spec.dims:
            raise ValueError("dealias_spectra needs spectra with a 'range' dimension")
        nz = None if lin is None else node_value(lin, name)
        fields, va = _moments_fields(
            spec,
            parent,
            dv,
            k,
            nz,
            region,
            n_sigma,
            n_average,
            min_bins,
            window_correction,
            min_snr,
            channel,
            use_compiled,
            threads,
            dim,
        )
        vel = fields[f"VRAD{channel}"]
        ref = node_value(reference, name) if reference is not None else None
        arrays, dims_in = [vel], [[RANGE]]
        if ref is not None:
            if isinstance(ref, xr.Dataset):
                raise TypeError("reference must be a number or a DataArray")
            ref = ref if isinstance(ref, xr.DataArray) else xr.DataArray(float(ref))
            arrays.append(xr.broadcast(ref.astype(np.float64), vel)[0])
            dims_in.append([RANGE])

        def block(*blocks):
            lead, flat = flatten(list(blocks), [1] * len(blocks))
            v2 = np.ascontiguousarray(flat[0], dtype=np.float64)
            r2 = (
                np.ascontiguousarray(flat[1], dtype=np.float64)
                if len(flat) > 1
                else np.empty((0, 0))
            )
            out, fold = kernel.unfold(v2, r2, va, int(max_gap), n_threads=threads)
            shape = lead + (v2.shape[1],)
            return out.reshape(shape).astype(FIELD_DTYPE), fold.reshape(shape).astype(
                np.int8
            )

        unfolded, fold = map_series(
            block, arrays, dims_in, [[RANGE], [RANGE]], {}, [FIELD_DTYPE, np.int8]
        )
        unfolded = unfolded.transpose(*vel.dims)
        fold = fold.transpose(*vel.dims)
        unfolded.attrs = {
            "long_name": "dealiased radial velocity from the Doppler spectrum",
            "standard_name": "radial_velocity_of_scatterers_away_from_instrument",
            "units": "m s-1",
            "positive": "away from the radar",
        }
        fold.attrs = {
            "long_name": "number of Nyquist intervals added to the velocity",
            "units": "1",
            "comment": f"each interval is 2 x {va:g} m s-1",
        }
        return xr.Dataset(
            {f"VRAD{channel}": unfolded, "FOLD": fold}, attrs={"nyquist_velocity": va}
        )

    if isinstance(obj, xr.DataTree):
        return ft.map_sweeps(obj, one, "dealias_spectra")
    return one(obj)
