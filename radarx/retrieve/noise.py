#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx Noise Level, SNR and Noise Correction
============================================

Gate-wise corrections that matter at long range, where the signal is weak:
estimate the noise level of a sweep, turn it into a signal-to-noise ratio
(SNR) for every gate, and correct the copolar correlation coefficient
:math:`\\rho_{hv}` and the differential reflectivity :math:`Z_{DR}` for the
noise in the two receiver channels.

Signal model
------------
The horizontal (H) and vertical (V) channels measure the signal plus
independent noise of powers :math:`N_h` and :math:`N_v`. With the signal
powers :math:`P_h`, :math:`P_v` (the noise-free powers) the SNR of the two
channels is :math:`s_h = P_h/N_h` and :math:`s_v = P_v/N_v`. The noise adds
power to both channels and none to the cross-correlation, so

.. math::

    \\rho_{hv}^{m} = \\frac{\\rho_{hv}}
    {\\sqrt{(1 + 1/s_h)(1 + 1/s_v)}},\\qquad
    Z_{DR}^{m} = Z_{DR}\\,\\frac{1 + 1/s_h}{1 + 1/s_v}
    \\quad (Z_{DR}\\ \\text{linear}),

with :math:`s_v = (N_h/N_v)\\, s_h / Z_{DR}`, because :math:`Z_{DR} = P_h/P_v`.
Solved for the signal, :math:`\\rho_{hv} = \\rho_{hv}^m \\sqrt{(1 + 1/s_h)(1 +
Z_{DR}/(\\alpha s_h))}` with :math:`\\alpha = N_h/N_v` is the form of the
correlation coefficient correction of Gourley et al. (2006, Eq. 6, as
implemented in Py-ART; not checked against the paper), and
:math:`Z_{DR} = Z_{DR}^m / (1 + 1/s_h - Z_{DR}^m/(\\alpha s_h))` follows from
the second relation (derived here, not taken from a paper). Without
:math:`Z_{DR}` the two channels are taken to have the same SNR, which gives
:math:`\\rho_{hv} = \\rho_{hv}^m (1 + 1/s_h)`. :math:`\\Phi_{DP}` is not
biased by uncorrelated noise (the noise leaves the phase of the
cross-correlation unchanged); it only gets noisier.

Apply these corrections only to moments in which the noise has not been
subtracted. A radar processor that subtracts the noise from the powers before
it forms :math:`Z_{DR}` and :math:`\\rho_{hv}` has already done a part of
this, and a second correction over-corrects.

SNR
---
The noise power received from a gate at range :math:`r` is the same for all
ranges, but the reflectivity that gives the same power as the noise grows with
:math:`r^2` (the weather radar equation, Doviak and Zrnic 1993). The noise
level of a sweep is therefore one number, the equivalent reflectivity of the
noise at 1 km, :math:`Z_N` in dBZ, and

.. math::

    \\mathrm{SNR}\\ [\\mathrm{dB}] = Z - Z_N - 20 \\log_{10}(r/1\\,\\mathrm{km}).

With ``signal_includes_noise=True`` the reflectivity still contains the noise
power, and the noise is taken off in linear units first.

Noise level
-----------
``"hildebrand"`` (default)
    The objective method of Hildebrand and Sekhon (1974), used on the
    range-corrected power of all gates, as is done for radar reflectivity by
    Kollias et al. (2014, not checked against the paper; the method is
    explained in the abstract as based on the statistical properties of
    receiver noise). The noise power estimated from :math:`n_s` independent
    samples has :math:`\\mathrm{mean}^2/\\mathrm{variance} = n_s`. The gate
    powers (reflectivity corrected to 1 km, in linear units) are sorted; the
    strongest are dropped one at a time until the remaining gates have
    :math:`\\mathrm{mean}^2/\\mathrm{variance} \\ge n_s`, and their mean is
    the noise. This needs data in which gates without signal are kept, that
    is, reflectivity that is not thresholded on SNR. radarx chooses the
    population variance, ``n_samples=64`` as the default and a minimum of
    ``min_count`` gates.
``"floor"``
    For data from which the gates without signal have been removed, such as
    the NEXRAD level 2 moments, no gate holds the noise itself. The lowest
    reflectivity that is reported still follows the noise: a low percentile
    (``percentile``, default 1) of the range-corrected reflectivity of all
    reported gates, minus the SNR of the threshold (``snr_threshold``, in dB,
    0 by default). With the default this is the lowest reportable
    reflectivity at 1 km, an upper bound of the noise level. This method is
    radarx's own.

The corrections run in a compiled, multithreaded C++ kernel; if it is not
available an equivalent NumPy implementation is used. All functions accept a
DataArray, a Dataset (a sweep) or a DataTree (a volume, sweep by sweep) and
keep dask arrays lazy where the computation is element-wise.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = ["noise_level", "snr", "correct_rhohv_noise", "correct_zdr_noise"]

__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

import numpy as np
import xarray as xr

from .._provenance import provenance
from .._registry import accessor_method
from . import _fieldtools as ft

try:
    from . import _noise

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _noise = None
    HAS_COMPILED_KERNEL = False

_NOISE_NAMES = ("NOISE_LEVEL",)
_METHODS = {"hildebrand": 0, "floor": 1}


def _use_compiled(engine):
    return ft.choose_engine(engine, HAS_COMPILED_KERNEL, "noise")


# --------------------------------------------------------------------------
# NumPy reference of the kernels (same results, same order of operations)
# --------------------------------------------------------------------------


def _from_db(x):
    return np.power(10.0, 0.1 * x)


def _to_db(x):
    return 10.0 * np.log10(x)


def _snr_numpy(dbz, noise, range_m, includes_noise):
    with np.errstate(invalid="ignore", divide="ignore"):
        s = dbz - noise - 20.0 * np.log10(range_m / 1000.0)
        if includes_noise:
            lin = _from_db(s) - 1.0
            s = np.where(lin > 0.0, _to_db(np.where(lin > 0.0, lin, 1.0)), np.nan)
        return np.where(range_m > 0.0, s, np.nan)


def _zdr_denoised(zdr, snr_db, alpha_db):
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        s = _from_db(snr_db)
        alpha = _from_db(alpha_db)
        zm = _from_db(zdr)
        denom = 1.0 + 1.0 / s - zm / (alpha * s)
        good = denom > 0.0
        return np.where(good, _to_db(zm / np.where(good, denom, 1.0)), np.nan)


def _correct_zdr_numpy(zdr, snr_db, alpha_db, min_snr):
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(zdr) & np.isfinite(snr_db) & (snr_db >= min_snr)
    return np.where(ok, _zdr_denoised(zdr, snr_db, alpha_db), np.nan)


def _correct_rhohv_numpy(rhohv, snr_db, zdr, alpha_db, zdr_has_noise, min_snr, clip):
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        ok = np.isfinite(rhohv) & np.isfinite(snr_db) & (snr_db >= min_snr)
        s = _from_db(snr_db)
        if zdr is None:
            ratio = 1.0
        else:
            z = _zdr_denoised(zdr, snr_db, alpha_db) if zdr_has_noise else zdr
            z = np.where(np.isfinite(zdr), z, np.nan)
            ratio = _from_db(z) / _from_db(alpha_db)
            ok = ok & np.isfinite(ratio)
        out = rhohv * np.sqrt((1.0 + 1.0 / s) * (1.0 + ratio / s))
        if clip:
            out = np.where(out > 1.0, 1.0, out)
        return np.where(ok, out, np.nan)


def _noise_rows_numpy(dbz, range_db, method, n_samples, min_count, percentile, offset):
    rows, nsamp = dbz.shape
    nr = range_db.size
    corrected = dbz - np.tile(range_db, nsamp // nr)[None, :]
    out = np.full((rows, 3), np.nan)
    out[:, 1] = 0.0
    for r in range(rows):
        v = corrected[r]
        v = v[~np.isnan(v)]
        n = v.size
        out[r, 2] = n
        if n < max(min_count, 1):
            continue
        if method == 1:
            out[r, 0] = np.percentile(v, percentile) - offset
            out[r, 1] = n
            continue
        p = np.sort(_from_db(v))
        cs1 = np.cumsum(p)
        cs2 = np.cumsum(p * p)
        k = np.arange(1, n + 1, dtype=np.float64)
        mean = cs1 / k
        var = cs2 / k - mean * mean
        good = np.flatnonzero(var * n_samples <= mean * mean)
        good = good[good + 1 >= max(min_count, 1)]
        if good.size:
            last = good[-1]
            out[r, 0] = _to_db(mean[last])
            out[r, 1] = last + 1
    return out


# --------------------------------------------------------------------------
# kernel dispatch on flat arrays
# --------------------------------------------------------------------------


def _snr_flat(dbz, noise, range_m, includes_noise, use_compiled, n_threads):
    if use_compiled:
        return _noise.snr_db(
            dbz, noise, range_m, bool(includes_noise), n_threads=int(n_threads or 0)
        )
    return _snr_numpy(dbz, noise, range_m, includes_noise)


def _rhohv_flat(args, use_compiled, n_threads):
    rhohv, snr_db, zdr, alpha_db, zdr_has_noise, min_snr, clip = args
    if use_compiled:
        return _noise.correct_rhohv(
            rhohv,
            snr_db,
            np.empty(0) if zdr is None else zdr,
            float(alpha_db),
            bool(zdr_has_noise),
            float(min_snr),
            bool(clip),
            n_threads=int(n_threads or 0),
        )
    return _correct_rhohv_numpy(*args)


def _zdr_flat(zdr, snr_db, alpha_db, min_snr, use_compiled, n_threads):
    if use_compiled:
        return _noise.correct_zdr(
            zdr, snr_db, float(alpha_db), float(min_snr), n_threads=int(n_threads or 0)
        )
    return _correct_zdr_numpy(zdr, snr_db, alpha_db, min_snr)


# --------------------------------------------------------------------------
# noise level
# --------------------------------------------------------------------------


def _run_noise(dbz, dims, range_db, method, options, use_compiled, n_threads):
    """Noise level of ``dbz`` over the dimensions ``dims`` -> DataArray (..., stat)."""
    method_id = _METHODS[method]

    def block(values):
        values = np.asarray(values, dtype=np.float64)
        lead = values.shape[: values.ndim - len(dims)]
        rows = int(np.prod(lead, dtype=np.int64)) if lead else 1
        flat = np.ascontiguousarray(values.reshape(rows, -1))
        args = (
            flat,
            range_db,
            method_id,
            float(options["n_samples"]),
            int(options["min_count"]),
            float(options["percentile"]),
            float(options["offset"]),
        )
        if use_compiled:
            res = _noise.noise_rows(*args, n_threads=int(n_threads or 0))
        else:
            res = _noise_rows_numpy(*args)
        return res.reshape(lead + (3,))

    return xr.apply_ufunc(
        block,
        dbz,
        input_core_dims=[list(dims)],
        output_core_dims=[["noise_statistic"]],
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={
            "output_sizes": {"noise_statistic": 3},
            "allow_rechunk": True,
        },
    )


def _noise_dataset(
    dbz,
    method,
    by,
    n_samples,
    percentile,
    snr_threshold,
    min_count,
    range_limits,
    n_threads,
    use_compiled,
):
    if method not in _METHODS:
        raise ValueError(f"method must be 'hildebrand' or 'floor', not {method!r}")
    if by not in ("sweep", "ray"):
        raise ValueError(f"by must be 'sweep' or 'ray', not {by!r}")
    if method == "hildebrand" and not n_samples > 0:
        raise ValueError("n_samples must be positive")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100")
    ray = ft.ray_dimension(dbz)
    rng = ft.range_meters(dbz)
    if range_limits is not None:
        lo, hi = range_limits
        keep = (rng >= 1000.0 * lo) & (rng <= 1000.0 * hi)
        dbz = dbz.where(keep)
    with np.errstate(invalid="ignore", divide="ignore"):
        range_db = 20.0 * np.log10(np.asarray(rng.values, dtype=np.float64) / 1000.0)
    range_db = np.where(np.isfinite(range_db), range_db, np.nan)
    dims = (ray, "range") if by == "sweep" else ("range",)
    # the kernel wants the samples of a row laid out (ray, range) with range last
    dbz = dbz.transpose(..., *dims)
    options = {
        "n_samples": n_samples,
        "min_count": min_count,
        "percentile": percentile,
        "offset": snr_threshold,
    }
    stat = _run_noise(dbz, dims, range_db, method, options, use_compiled, n_threads)
    level = stat.isel(noise_statistic=0, drop=True).astype(np.float32)
    used = stat.isel(noise_statistic=1, drop=True).astype(np.int32)
    comment = (
        "Hildebrand and Sekhon (1974) objective noise level of the range-"
        f"corrected power with {n_samples:g} independent samples"
        if method == "hildebrand"
        else f"{percentile:g}th percentile of the range-corrected reflectivity "
        f"minus a threshold SNR of {snr_threshold:g} dB"
    )
    out = xr.Dataset(
        {
            "NOISE_LEVEL": level.assign_attrs(
                long_name="noise level, equivalent reflectivity at 1 km range",
                units="dBZ",
                comment=comment,
            ),
            "NOISE_SAMPLES": used.assign_attrs(
                long_name="number of gates used for the noise level",
                units="1",
            ),
        }
    )
    out.attrs["noise_method"] = method
    out.attrs["noise_by"] = by
    return out


@provenance(
    "Noise level of the reflectivity of a sweep",
    extra_refs=("doviak-zrnic-1993",),
)
def noise_level(
    obj,
    dbz=None,
    *,
    method="hildebrand",
    by="sweep",
    n_samples=64.0,
    percentile=1.0,
    snr_threshold=0.0,
    min_count=50,
    range_limits=None,
    n_threads=None,
    engine="auto",
):
    """
    Noise level of a sweep as the equivalent reflectivity of the noise at 1 km.

    The receiver noise has the same power at every range, so in
    reflectivity it grows as ``20 log10(r)``. The range-corrected
    reflectivity ``Z - 20 log10(r / km)`` of the gates that hold noise only is
    constant and equal to the noise level :math:`Z_N`. Two methods find it; see
    :mod:`radarx.retrieve.noise` for the details and the radarx choices.

    ``"hildebrand"`` applies the objective noise determination of Hildebrand
    and Sekhon (1974) to the gates (reflectivity that still holds gates
    without signal is needed). ``"floor"`` takes a low percentile of the
    range-corrected reflectivity of the reported gates, for data that were
    thresholded on SNR.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        The reflectivity (a DataArray with ``range`` and ray dimensions), a
        sweep, or a volume of sweeps (processed sweep by sweep; sweeps
        without the field are skipped).
    dbz : str, optional
        Name of the reflectivity. By default the first of ``DBZH``, ``DBZ``,
        ``reflectivity`` and ``corrected_reflectivity``.
    method : {"hildebrand", "floor"}, optional
        Noise determination, default ``"hildebrand"``.
    by : {"sweep", "ray"}, optional
        One noise level for the whole sweep (default) or one per ray.
    n_samples : float, optional
        ``"hildebrand"``: number of independent samples averaged in a gate
        (pulses per ray for a gate power). A value that is too small accepts
        signal as noise (biased high); one that is too large rejects noise
        (biased low), and NaN results if no set of gates is steady enough.
        Default 64 (radarx choice).
    percentile : float, optional
        ``"floor"``: the percentile of the range-corrected reflectivity,
        0 to 100. Default 1.
    snr_threshold : float, optional
        ``"floor"``: the SNR (dB) of the lowest reported gate, subtracted from
        the percentile. Default 0.
    min_count : int, optional
        Fewest gates for an estimate; NaN otherwise. Default 50.
    range_limits : tuple of float, optional
        Use only gates between these ranges (km), for example the far ranges
        where there is little signal.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` prefers the compiled kernel and
        falls back to NumPy.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``NOISE_LEVEL`` (dBZ, at 1 km) and ``NOISE_SAMPLES`` (gates used),
        with the dimension of the rays if ``by="ray"``; a DataTree with one
        such node per sweep for a volume.

    Raises
    ------
    KeyError
        If the reflectivity is not found.
    ValueError
        For unknown options or a field without ``range`` and ray dimensions.
    ImportError
        If ``engine="compiled"`` and the compiled kernel is not available.

    Notes
    -----
    The noise level is a property of the receiver and the signal processing,
    so it changes little from one sweep to the next, except between waveforms
    (a long pulse for surveillance and a short one for Doppler cuts have
    different noise levels). The estimate is only as good as the data: with
    ``"hildebrand"`` the sweep must contain gates without signal.

    References
    ----------
    Hildebrand, P. H., and R. S. Sekhon, 1974: Objective determination of the
    noise level in Doppler spectra. *J. Appl. Meteor.*, **13** (7), 808-811,
    https://doi.org/10.1175/1520-0450(1974)013<0808:ODOTNL>2.0.CO;2

    Kollias, P., I. Jo, P. Borque, A. Tatarevic, K. Lamer, N. Bharadwaj, K.
    Widener, K. Johnson, and E. E. Clothiaux, 2014: Scanning ARM cloud
    radars. Part II: Data quality control and processing. *J. Atmos. Oceanic
    Technol.*, **31** (3), 583-598,
    https://doi.org/10.1175/JTECH-D-13-00045.1
    (range-corrected reflectivity and the noise-statistics criterion; not
    checked against the paper).

    Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
    Observations*, 2nd ed. Academic Press (the r-squared dependence of the
    reflectivity of the noise).

    Examples
    --------
    >>> noise = radarx.retrieve.noise_level(sweep)  # doctest: +SKIP
    >>> noise = dtree.radarx.noise_level(method="floor")  # doctest: +SKIP
    """
    ft.check_object(obj)
    use_compiled = _use_compiled(engine)

    def one(ds, _name=None):
        if isinstance(ds, xr.DataArray):
            da = ds
        else:
            da = ds[ft.find_field(ds, dbz, ft.DBZ_NAMES, "reflectivity")]
        return _noise_dataset(
            da,
            method,
            by,
            n_samples,
            percentile,
            snr_threshold,
            min_count,
            range_limits,
            n_threads,
            use_compiled,
        )

    if isinstance(obj, xr.DataTree):
        return ft.map_sweeps(obj, one, "noise_level")
    return one(obj)


# --------------------------------------------------------------------------
# SNR
# --------------------------------------------------------------------------


def _noise_value(noise, ds, dbz_da, noise_options, name, use_compiled, n_threads):
    """The noise level (dBZ at 1 km) as a number or DataArray for one sweep."""
    noise = ft.node_arg(noise, name) if name is not None else noise
    if noise is None:
        opts = dict(noise_options or {})
        opts.setdefault("n_threads", n_threads)
        opts.setdefault("engine", "compiled" if use_compiled else "numpy")
        est = noise_level(dbz_da, **opts)
        return est["NOISE_LEVEL"]
    if isinstance(noise, xr.Dataset):
        key = ft.find_field(noise, None, _NOISE_NAMES, "noise level")
        return noise[key]
    if isinstance(noise, xr.DataArray):
        return noise
    return float(noise)


def _snr_dataarray(
    dbz_da, noise, includes_noise, noise_options, name, use_compiled, n_threads
):
    level = _noise_value(
        noise, None, dbz_da, noise_options, name, use_compiled, n_threads
    )
    level_da = (
        level if isinstance(level, xr.DataArray) else xr.DataArray(np.float64(level))
    )
    rng = ft.range_meters(dbz_da)

    def func(z, n, r):
        return _snr_flat(z, n, r, includes_noise, use_compiled, n_threads)

    out = ft.elementwise(func, [dbz_da, level_da.astype(np.float64), rng])
    out = out.transpose(*dbz_da.dims, ...)
    out.name = "SNR"
    out.attrs = {
        "long_name": "signal-to-noise ratio",
        "standard_name": "signal_to_noise_ratio",
        "units": "dB",
        "comment": (
            "reflectivity minus the noise level at 1 km minus 20 log10(range / km)"
            + (
                "; the noise power is taken off the signal first"
                if includes_noise
                else ""
            )
        ),
    }
    if not isinstance(level, xr.DataArray) or level.ndim == 0:
        try:
            out.attrs["noise_level"] = float(level)
        except TypeError:  # dask scalar
            pass
    return out


@provenance(
    "Signal-to-noise ratio from the reflectivity and the noise level",
    extra_refs=("doviak-zrnic-1993",),
)
def snr(
    obj,
    noise=None,
    dbz=None,
    *,
    signal_includes_noise=False,
    noise_options=None,
    n_threads=None,
    engine="auto",
):
    """
    Signal-to-noise ratio of every gate from the reflectivity.

    ``SNR = Z - Z_N - 20 log10(r / km)`` in dB, with the noise level
    :math:`Z_N` (dBZ at 1 km) of :func:`noise_level`. See
    :mod:`radarx.retrieve.noise`.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        The reflectivity (DataArray), a sweep or a volume.
    noise : float, xarray.DataArray, xarray.Dataset or xarray.DataTree, optional
        Noise level (dBZ at 1 km): a number, an array that broadcasts against
        the reflectivity (for example one value per ray), the result of
        :func:`noise_level` (a DataTree for a volume). By default it is
        estimated from the reflectivity itself with ``noise_options``.
    dbz : str, optional
        Name of the reflectivity, see :func:`noise_level`.
    signal_includes_noise : bool, optional
        The reflectivity still contains the noise power (no noise
        subtraction). Gates where it is not above the noise are NaN.
        Default False.
    noise_options : dict, optional
        Options of :func:`noise_level` when the noise is estimated (``method``,
        ``by``, ``n_samples``, ...).
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        ``SNR`` (dB) on the coordinates of the reflectivity, float32; for a
        volume a DataTree with one node per sweep. Dask arrays stay lazy.

    Raises
    ------
    KeyError
        If the reflectivity is not found.
    ValueError
        For unknown options.

    References
    ----------
    Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
    Observations*, 2nd ed. Academic Press (SNR of a weather echo and the
    equivalent reflectivity of the noise; not checked against the book).

    Hildebrand, P. H., and R. S. Sekhon, 1974: Objective determination of the
    noise level in Doppler spectra. *J. Appl. Meteor.*, **13** (7), 808-811,
    https://doi.org/10.1175/1520-0450(1974)013<0808:ODOTNL>2.0.CO;2
    (the default noise level).

    Examples
    --------
    >>> s = radarx.retrieve.snr(sweep, noise=-40.0)  # doctest: +SKIP
    >>> s = dtree.radarx.snr()  # doctest: +SKIP
    """
    ft.check_object(obj)
    use_compiled = _use_compiled(engine)

    def one(ds, name=None):
        da = (
            ds
            if isinstance(ds, xr.DataArray)
            else ds[ft.find_field(ds, dbz, ft.DBZ_NAMES, "reflectivity")]
        )
        return _snr_dataarray(
            da,
            noise,
            signal_includes_noise,
            noise_options,
            name,
            use_compiled,
            n_threads,
        )

    if isinstance(obj, xr.DataTree):
        return ft.map_sweeps(obj, one, "snr")
    return one(obj)


# --------------------------------------------------------------------------
# corrections
# --------------------------------------------------------------------------


def _field(ds, value, candidates, what, required=True):
    """A DataArray from a name, a DataArray or the default candidates."""
    if isinstance(value, xr.DataArray):
        return value
    if isinstance(ds, xr.DataArray):
        if required:
            raise ValueError(f"with a DataArray as obj, pass {what} as a DataArray")
        return None
    name = ft.find_field(ds, value, candidates, what, required=required)
    return None if name is None else ds[name]


def _resolve_snr(
    ds,
    snr_arg,
    dbz,
    noise,
    includes_noise,
    noise_options,
    name,
    use_compiled,
    n_threads,
):
    """SNR (dB) of a sweep from a field, a name, or the reflectivity and the noise."""
    snr_arg = ft.node_arg(snr_arg, name) if name is not None else snr_arg
    if isinstance(snr_arg, xr.DataArray):
        return snr_arg
    if isinstance(snr_arg, xr.Dataset):
        return snr_arg[ft.find_field(snr_arg, None, ft.SNR_NAMES, "SNR")]
    if isinstance(ds, xr.Dataset):
        have = ft.find_field(ds, snr_arg, ft.SNR_NAMES, "SNR", required=False)
        if have is not None and noise is None:
            return ds[have]
        if snr_arg is not None:
            raise ft.MissingField(f"{snr_arg!r} is not in the dataset")
        da = ds[ft.find_field(ds, dbz, ft.DBZ_NAMES, "reflectivity")]
    elif isinstance(dbz, xr.DataArray):
        da = dbz
    else:
        raise ValueError(
            "with a DataArray as obj, pass snr as a DataArray, or dbz as a "
            "DataArray together with the noise"
        )
    return _snr_dataarray(
        da, noise, includes_noise, noise_options, name, use_compiled, n_threads
    )


def _mark(out, name, long_name, units, like, comment):
    out.name = name
    out.attrs = ft.attrs_of(
        like,
        long_name=long_name,
        units=units,
        comment=comment,
    )
    return out


def _min_snr_text(min_snr):
    return f"gates with an SNR below {min_snr:g} dB are NaN"


@provenance("Correlation coefficient corrected for noise")
def correct_rhohv_noise(
    obj,
    rhohv=None,
    *,
    snr=None,
    zdr=None,
    dbz=None,
    noise=None,
    noise_options=None,
    noise_ratio=0.0,
    zdr_includes_noise=False,
    signal_includes_noise=False,
    min_snr=0.0,
    clip=True,
    n_threads=None,
    engine="auto",
):
    """
    Copolar correlation coefficient corrected for the noise of the channels.

    ``rho = rho_m sqrt((1 + 1/s_h)(1 + Z_DR / (alpha s_h)))`` with the linear
    SNR of the horizontal channel ``s_h``, the linear ``Z_DR`` and the ratio
    of the noise powers ``alpha = N_h / N_v`` (Gourley et al. 2006, Eq. 6, as
    implemented in Py-ART). Without ``zdr`` the channels get the same SNR and
    the factor is ``1 + 1/s_h``. See :mod:`radarx.retrieve.noise`.

    The correction is only valid for a moment in which the noise has not been
    subtracted. Radarx chooses to set gates with an SNR below ``min_snr`` to
    NaN and to limit the result to 1 (``clip``).

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        The correlation coefficient (DataArray), a sweep or a volume.
    rhohv : str or xarray.DataArray, optional
        Name of the correlation coefficient, by default ``RHOHV``. Only for a
        Dataset or DataTree; a DataArray ``obj`` is the field.
    snr : str, xarray.DataArray, xarray.Dataset or xarray.DataTree, optional
        The SNR of the horizontal channel in dB (a field name, an array, the
        result of :func:`snr`). By default the field ``SNR`` if there is one
        (and ``noise`` is not given), else it is computed from the
        reflectivity (``dbz``) and the noise level.
    zdr : bool, str or xarray.DataArray, optional
        The differential reflectivity (dB) that gives the SNR of the vertical
        channel: ``True`` for the default field, a name or an array. By
        default it is not used and the two channels have the same SNR.
    dbz : str or xarray.DataArray, optional
        Reflectivity for the SNR, see :func:`snr`.
    noise, noise_options : optional
        Noise level (dBZ at 1 km) and the options for estimating it, see
        :func:`snr`.
    noise_ratio : float, optional
        ``N_h - N_v`` in dB, the difference of the noise powers of the two
        channels. Default 0.
    zdr_includes_noise : bool, optional
        ``zdr`` is the measured differential reflectivity, which holds the
        noise; it is corrected first with :func:`correct_zdr_noise`. Default
        False (it is used as given).
    signal_includes_noise : bool, optional
        See :func:`snr`. Default False.
    min_snr : float, optional
        SNR (dB) below which the gate is NaN. Default 0: below it the
        correction is larger than the signal.
    clip : bool, optional
        Limit the result to 1. Default True.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        The corrected ``RHOHV`` on the coordinates of the input, float32;
        a DataTree with one node per sweep for a volume. Dask arrays stay
        lazy.

    Raises
    ------
    KeyError
        If a field is not found.
    ValueError
        If the SNR cannot be built from the arguments.

    References
    ----------
    Gourley, J. J., P. Tabary, and J. Parent du Chatelet, 2006: Data quality of
    the Meteo-France C-band polarimetric radar. *J. Atmos. Oceanic Technol.*,
    **23** (10), 1340-1356, https://doi.org/10.1175/JTECH1912.1
    (corrections for the noise of the two channels; Eq. 6 as implemented in
    Py-ART, not checked against the paper).

    Examples
    --------
    >>> rho = radarx.retrieve.correct_rhohv_noise(sweep, noise=-40.0)  # doctest: +SKIP
    >>> rho = dtree.radarx.correct_rhohv_noise(zdr=True)  # doctest: +SKIP
    """
    ft.check_object(obj)
    use_compiled = _use_compiled(engine)

    def one(ds, name=None):
        rho = ft.primary_field(ds, rhohv, ft.RHOHV_NAMES, "correlation coefficient")
        s = _resolve_snr(
            ds,
            snr,
            dbz,
            noise,
            signal_includes_noise,
            noise_options,
            name,
            use_compiled,
            n_threads,
        )
        z = None
        if zdr is not None and zdr is not False:
            z = _field(ds, None if zdr is True else zdr, ft.ZDR_NAMES, "ZDR")

        def func(*flat):
            args = (
                flat[0],
                flat[1],
                flat[2] if z is not None else None,
                noise_ratio,
                zdr_includes_noise,
                min_snr,
                clip,
            )
            return _rhohv_flat(args, use_compiled, n_threads)

        arrays = [rho, s] + ([z] if z is not None else [])
        out = ft.elementwise(func, arrays).transpose(*rho.dims, ...)
        comment = (
            "corrected for the noise of the two channels (Gourley et al. 2006); "
            + _min_snr_text(min_snr)
            + ("; limited to 1" if clip else "")
        )
        return _mark(
            out,
            "RHOHV",
            "cross correlation ratio (RHOHV) corrected for noise",
            "1",
            rho,
            comment,
        )

    if isinstance(obj, xr.DataArray):
        return one(obj)
    if isinstance(obj, xr.DataTree):
        return ft.map_sweeps(obj, one, "correct_rhohv_noise")
    return one(obj)


@provenance("Differential reflectivity corrected for noise")
def correct_zdr_noise(
    obj,
    zdr=None,
    *,
    snr=None,
    dbz=None,
    noise=None,
    noise_options=None,
    noise_ratio=0.0,
    signal_includes_noise=False,
    min_snr=0.0,
    n_threads=None,
    engine="auto",
):
    """
    Differential reflectivity corrected for the noise of the channels.

    The noise adds power to both channels, which pulls the measured ratio
    towards 1. With the signal-to-noise ratio ``s_h`` of the horizontal
    channel and the ratio of the noise powers ``alpha = N_h / N_v``, the
    measured linear ratio is ``Z_DR (1 + 1/s_h) / (1 + Z_DR / (alpha s_h))``,
    so ``Z_DR = Z_m / (1 + 1/s_h - Z_m / (alpha s_h))``. The relation is
    derived here from the signal model of Gourley et al. (2006), who also
    describe a correction for the noise of the two channels (not checked
    against the paper). Gates where the denominator is not positive (the SNR is
    too low for the measured ratio to be reached by any signal) are NaN, as
    are gates below ``min_snr``. See :mod:`radarx.retrieve.noise`.

    Apply it only to a moment in which the noise has not been subtracted.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        The differential reflectivity (DataArray), a sweep or a volume.
    zdr : str or xarray.DataArray, optional
        Name of the differential reflectivity, by default ``ZDR``.
    snr, dbz, noise, noise_options, noise_ratio, signal_includes_noise : optional
        As in :func:`correct_rhohv_noise`.
    min_snr : float, optional
        SNR (dB) below which the gate is NaN. Default 0.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        The corrected ``ZDR`` (dB) on the coordinates of the input, float32;
        a DataTree with one node per sweep for a volume. Dask arrays stay
        lazy.

    Raises
    ------
    KeyError
        If a field is not found.
    ValueError
        If the SNR cannot be built from the arguments.

    References
    ----------
    Gourley, J. J., P. Tabary, and J. Parent du Chatelet, 2006: Data quality of
    the Meteo-France C-band polarimetric radar. *J. Atmos. Oceanic Technol.*,
    **23** (10), 1340-1356, https://doi.org/10.1175/JTECH1912.1

    Examples
    --------
    >>> zdr = radarx.retrieve.correct_zdr_noise(sweep, noise=-40.0)  # doctest: +SKIP
    >>> zdr = dtree.radarx.correct_zdr_noise()  # doctest: +SKIP
    """
    ft.check_object(obj)
    use_compiled = _use_compiled(engine)

    def one(ds, name=None):
        z = ft.primary_field(ds, zdr, ft.ZDR_NAMES, "differential reflectivity")
        s = _resolve_snr(
            ds,
            snr,
            dbz,
            noise,
            signal_includes_noise,
            noise_options,
            name,
            use_compiled,
            n_threads,
        )

        def func(zv, sv):
            return _zdr_flat(zv, sv, noise_ratio, min_snr, use_compiled, n_threads)

        out = ft.elementwise(func, [z, s]).transpose(*z.dims, ...)
        return _mark(
            out,
            "ZDR",
            "differential reflectivity corrected for noise",
            "dB",
            z,
            "corrected for the noise of the two channels; " + _min_snr_text(min_snr),
        )

    if isinstance(obj, xr.DataTree):
        return ft.map_sweeps(obj, one, "correct_zdr_noise")
    return one(obj)


# --------------------------------------------------------------------------
# accessors
# --------------------------------------------------------------------------


@accessor_method("dataarray", "dataset", "datatree", name="noise_level")
def _noise_level_accessor(self, **kwargs):
    """
    Noise level of a sweep or volume as the reflectivity of the noise at 1 km.

    See :func:`radarx.retrieve.noise_level` for the parameters.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``NOISE_LEVEL`` (dBZ) and ``NOISE_SAMPLES``.
    """
    return noise_level(self.xarray_obj, **kwargs)


@accessor_method("dataarray", "dataset", "datatree", name="snr")
def _snr_accessor(self, **kwargs):
    """
    Signal-to-noise ratio of every gate.

    See :func:`radarx.retrieve.snr` for the parameters.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        ``SNR`` (dB).
    """
    return snr(self.xarray_obj, **kwargs)


@accessor_method("dataarray", "dataset", "datatree", name="correct_rhohv_noise")
def _correct_rhohv_noise_accessor(self, **kwargs):
    """
    Copolar correlation coefficient corrected for noise.

    See :func:`radarx.retrieve.correct_rhohv_noise` for the parameters.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        The corrected ``RHOHV``.
    """
    return correct_rhohv_noise(self.xarray_obj, **kwargs)


@accessor_method("dataarray", "dataset", "datatree", name="correct_zdr_noise")
def _correct_zdr_noise_accessor(self, **kwargs):
    """
    Differential reflectivity corrected for noise.

    See :func:`radarx.retrieve.correct_zdr_noise` for the parameters.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        The corrected ``ZDR`` (dB).
    """
    return correct_zdr_noise(self.xarray_obj, **kwargs)
