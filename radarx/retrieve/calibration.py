#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx Calibration of ZDR, Z and the System PhiDP
=================================================

Calibration is the largest uncertainty in dual-polarization products. Three
functions estimate the offsets from the data themselves, by the methods
that Ryzhkov et al. (2005) review, and report them with their uncertainty. They
never change the data: the offset is the measured value minus the true value,
so the corrected field is ``field - offset``.

Differential reflectivity (:func:`zdr_offset`)
    The offset is the mean (or median) of :math:`Z_{DR}` minus a reference
    over gates that are known to have that reference:

    ``"vertical"``
        Vertical pointing (a "birdbath" scan) in rain or snow. The drops and
        crystals fall with their symmetry axis along the beam, so the
        intrinsic :math:`Z_{DR}` is 0 dB whatever their size. This is the
        reference of the vertical method in Ryzhkov et al. (2005).
    ``"light_rain"``
        Light rain, where the drops are small and nearly spherical and the
        intrinsic :math:`Z_{DR}` is a small value that depends weakly on the
        reflectivity (Ryzhkov et al. 2005 use light rain probed at high
        elevation angles). The reference is the :math:`Z_{DR}` of an
        exponential drop size distribution with the intercept
        :math:`N_0 = 8000` mm\\ :sup:`-1` m\\ :sup:`-3` (Marshall and Palmer
        1948) at the same reflectivity, computed from the radarx T-matrix
        tables of :func:`radarx.retrieve.dsd` (about 0.3 to 0.4 dB at 20 to
        22 dBZ at S band). Both the DSD and the use of the tables are radarx
        choices; Ryzhkov et al. use a table derived from disdrometer data,
        which radarx does not have.
    ``"dry_snow"``
        Dry aggregated snow, whose intrinsic :math:`Z_{DR}` is close to 0 dB
        (Ryzhkov et al. 2005 use it at high elevation). The reference is a
        constant, 0.2 dB (``reference``), a radarx choice within the 0 to
        0.5 dB range quoted for dry aggregates. The gates must be selected
        with ``mask``, for example above the melting layer.

    Ryzhkov et al. (2005) adapt the method to elevations below the zenith.
    For a spheroidal scatterer in the Rayleigh regime with its axis vertical, a
    beam at elevation :math:`\\theta` sees the vertical polarization
    through the polarizability :math:`\\alpha_a \\sin^2\\theta + \\alpha_b
    \\cos^2\\theta` and the horizontal one through :math:`\\alpha_a`, so the
    reference :math:`Z_{DR}^0` at 0 degrees becomes
    :math:`Z_{DR}(\\theta) = Z_{DR}^0 - 20\\log_{10}(10^{Z_{DR}^0/20}
    \\sin^2\\theta + \\cos^2\\theta)` (dB). radarx derives this relation
    itself (no canting, Rayleigh scattering) and applies it with the elevation
    of the data.

Reflectivity (:func:`z_offset`)
    The calibration of Z rests on the self-consistency of :math:`Z`, :math:`Z_{DR}` and :math:`K_{DP}` in rain: the
    three are not independent, because for the drop size distributions of rain they are
    tied by the shape of the drops (Gorgucci et al. 1992). The path-integrated
    :math:`K_{DP}` does not depend on the radar calibration, so the Z that the
    other two imply can be compared with the measured one (Ryzhkov et al.
    2005; Gourley et al. 2009 automate the comparison). As in Ryzhkov et al.
    (2005), the gates are averaged in classes of 1 dB of Z first, which
    lowers the noise of :math:`K_{DP}` so much that light rain can be used.
    From the class means of :math:`Z_{DR}` and :math:`K_{DP}` the expected Z is
    :math:`10\\log_{10}(K_{DP}/g(Z_{DR}))`, with :math:`g` the ratio of
    :math:`K_{DP}` to the linear Z, and the offset is the median (or the
    weighted mean) of the measured minus the expected Z over the classes. The
    relation :math:`g` is not one of the papers': radarx computes it from its
    own T-matrix tables for the constrained gamma drop size distributions of
    :func:`radarx.retrieve.dsd` (the shape of the DSD follows from its slope
    through the relation of Cao et al. 2008). A :math:`Z_{DR}` offset shifts the
    result, which is why ``zdr_offset`` is an argument: the two offsets cannot
    be separated by the self-consistency alone.

System differential phase (:func:`phidp_offset`)
    The measured :math:`\\Phi_{DP}` has a system offset, the phase difference of
    the receivers, that has to be taken off before :math:`K_{DP}` or the
    attenuation correction (Gourley et al. 2006 describe a correction of the
    system offset). It is the phase near the radar, where the propagation
    phase is still zero: the circular mean over the first valid gates of each
    ray, summarized over the rays by a circular median.

The statistics run in a compiled, multithreaded C++ kernel (one task per row:
a sweep, a profile or a ray); if it is not available an equivalent NumPy
implementation is used. The functions accept a DataArray, a Dataset or a
DataTree and keep dask arrays lazy.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = ["zdr_offset", "z_offset", "phidp_offset"]

__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

import functools
import importlib

import numpy as np
import xarray as xr

from .._compat import renamed_arguments
from .._provenance import provenance
from .._registry import accessor_method
from . import _fieldtools as ft

try:
    from . import _calibration

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _calibration = None
    HAS_COMPILED_KERNEL = False

_STAT = "calibration_statistic"
_METHOD_RANGE = {
    "vertical": (-np.inf, np.inf),
    "light_rain": (20.0, 25.0),
    "dry_snow": (0.0, 30.0),
}
_DRY_SNOW_ZDR = 0.2  # dB, radarx choice
_MP_N0 = 8000.0  # mm-1 m-3, Marshall and Palmer (1948)


def _use_compiled(engine):
    return ft.choose_engine(engine, HAS_COMPILED_KERNEL, "calibration")


def _dsd_module():
    return importlib.import_module("radarx.retrieve.dsd")


# --------------------------------------------------------------------------
# reference curves from the radarx scattering tables
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=16)
def _light_rain_curve(band, temperature):
    """Intrinsic ZDR (dB) of Marshall-Palmer rain against dBZ, increasing in dBZ."""
    lam = np.geomspace(20.0, 1.0, 400)  # mm-1, decreasing: Z increases
    zh, zv, _ = _dsd_module()._gamma_integrals(band, temperature, _MP_N0, 0.0, lam)
    z = 10.0 * np.log10(zh)
    zdr = 10.0 * np.log10(zh / zv)
    return z, zdr


@functools.lru_cache(maxsize=16)
def _consistency_curve(band, temperature, coefficients):
    """ZDR (dB, increasing) and log10(KDP / Z) of the constrained gamma family."""
    tab = _dsd_module()._lookup_table(
        band, temperature, "constrained", coefficients, 3.0
    )
    return np.array(tab["zdr"]), np.array(tab["logk"] - tab["logz"])


def _band_of(obj, ds, band):
    """Radar band: as given, else from the sweep, else from the volume (VCP)."""
    dsd = _dsd_module()
    if band is not None:
        dsd._check_band(band)
        return band
    return dsd._infer_band([o for o in (ds, obj) if o is not None])


# --------------------------------------------------------------------------
# NumPy reference of the kernels
# --------------------------------------------------------------------------


def _median(v):
    return float(np.median(v))


def _offset_stats_numpy(
    zdr, dbz, rhohv, snr, mask, elev, ref_z, ref_v, z_min, z_max, min_rhohv, min_snr
):
    rows, n = zdr.shape
    out = np.full((rows, 5), np.nan)
    out[:, 0] = 0.0
    table = ref_v.size > 1
    for r in range(rows):
        with np.errstate(invalid="ignore"):
            ok = np.isfinite(zdr[r])
            if rhohv.size:
                ok &= rhohv[r] >= min_rhohv
            if snr.size:
                ok &= snr[r] >= min_snr
            if mask.size:
                ok &= mask[r] != 0.0
            ref = np.full(n, ref_v[0])
            if dbz.size:
                z = dbz[r]
                ok &= np.isfinite(z) & (z >= z_min) & (z <= z_max)
                if table:
                    ref = np.interp(z, ref_z, ref_v)
            theta = np.deg2rad(elev[r])
            if np.isfinite(theta):
                ref = ref - 20.0 * np.log10(
                    np.power(10.0, 0.05 * ref) * np.sin(theta) ** 2 + np.cos(theta) ** 2
                )
        k = int(ok.sum())
        out[r, 0] = k
        if k:
            dev = (zdr[r] - ref)[ok]
            out[r, 1] = dev.mean()
            out[r, 2] = dev.std(ddof=1) if k > 1 else np.nan
            out[r, 3] = _median(dev)
            out[r, 4] = ref[ok].mean()
    return out


def _consistency_stats_numpy(
    z,
    zdr,
    kdp,
    rhohv,
    snr,
    mask,
    gz,
    glg,
    zdr_offset,
    z_min,
    z_max,
    bin_width,
    min_rhohv,
    min_snr,
    min_kdp,
    min_bin_count,
):
    rows, n = z.shape
    nb = int(np.ceil((z_max - z_min) / bin_width))
    out = np.full((rows, 5), np.nan)
    out[:, 0] = 0.0
    out[:, 1] = 0.0
    for r in range(rows):
        with np.errstate(invalid="ignore"):
            ok = np.isfinite(z[r]) & np.isfinite(zdr[r]) & np.isfinite(kdp[r])
            ok &= (z[r] >= z_min) & (z[r] < z_max)
            if rhohv.size:
                ok &= rhohv[r] >= min_rhohv
            if snr.size:
                ok &= snr[r] >= min_snr
            if mask.size:
                ok &= mask[r] != 0.0
        b = np.minimum(nb - 1, np.floor((z[r][ok] - z_min) / bin_width).astype(int))
        cnt = np.bincount(b, minlength=nb).astype(float)
        sz = np.bincount(b, weights=z[r][ok], minlength=nb)
        sd = np.bincount(b, weights=zdr[r][ok], minlength=nb)
        sk = np.bincount(b, weights=kdp[r][ok], minlength=nb)
        with np.errstate(invalid="ignore", divide="ignore"):
            mz, md, mk = sz / cnt, sd / cnt - zdr_offset, sk / cnt
            use = (cnt >= max(min_bin_count, 1)) & (mk >= min_kdp) & (mk > 0.0)
            use &= (md >= gz[0]) & (md <= gz[-1])
            lg = np.interp(md, gz, glg)
            dev = mz - 10.0 * (np.log10(mk) - lg)
        dev, w = dev[use], cnt[use]
        out[r, 0] = w.sum()
        out[r, 1] = dev.size
        if dev.size:
            out[r, 2] = _median(dev)
            out[r, 3] = (w * dev).sum() / w.sum()
            out[r, 4] = dev.std(ddof=1) if dev.size > 1 else np.nan
    return out


def _phidp_rays_numpy(
    phidp, rhohv, snr, mask, first_gate, n_use, min_rhohv, min_snr, min_count
):
    rows, n = phidp.shape
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(phidp)
        ok &= (np.arange(n) >= max(first_gate, 0))[None, :]
        if rhohv.size:
            ok &= rhohv >= min_rhohv
        if snr.size:
            ok &= snr >= min_snr
        if mask.size:
            ok &= mask != 0.0
    ok &= np.cumsum(ok, axis=1) <= n_use
    a = np.deg2rad(np.where(ok, phidp, 0.0))
    sc = np.where(ok, np.cos(a), 0.0).sum(axis=1)
    ss = np.where(ok, np.sin(a), 0.0).sum(axis=1)
    k = ok.sum(axis=1)
    good = k >= max(min_count, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        offset = np.where(good, np.rad2deg(np.arctan2(ss, sc)), np.nan)
        length = np.where(good, np.hypot(sc, ss) / np.maximum(k, 1), np.nan)
    return np.stack([offset, k.astype(float), length], axis=1)


# --------------------------------------------------------------------------
# applying a row kernel to xarray objects
# --------------------------------------------------------------------------


def _empty():
    return np.empty(0)


def _row_apply(fields, leads, dims, nout, func):
    """
    Apply ``func`` to the rows of fields reduced over ``dims``.

    ``fields`` is a dict of DataArrays or None that share ``dims``; ``leads``
    is a list of DataArrays without ``dims`` (one value per row). ``func``
    receives the fields as (rows, samples) float64 arrays (size 0 for the
    missing ones) and the leads as (rows,) arrays, and returns (rows, nout).
    Returns a DataArray with the dimension ``calibration_statistic`` last.
    """
    names = [k for k, v in fields.items() if v is not None]
    arrays = [fields[k] for k in names]
    ncore = len(dims)

    def block(*blocks):
        fb = dict(zip(names, blocks[: len(names)]))
        lb = list(blocks[len(names) :])
        core = blocks[0].shape[blocks[0].ndim - ncore :]
        lead_shape = np.broadcast_shapes(
            *[b.shape[: b.ndim - ncore] for b in fb.values()],
            *[b.shape for b in lb],
        )
        rows = int(np.prod(lead_shape, dtype=np.int64)) if lead_shape else 1
        flat = {}
        for k in fields:
            if k in fb:
                full = np.broadcast_to(
                    np.asarray(fb[k], dtype=np.float64), lead_shape + core
                )
                flat[k] = np.ascontiguousarray(full.reshape(rows, -1))
            else:
                flat[k] = _empty().reshape(0, 0)
        lv = [
            np.ascontiguousarray(
                np.broadcast_to(np.asarray(b, dtype=np.float64), lead_shape).reshape(
                    rows
                )
            )
            for b in lb
        ]
        res = func(flat, lv)
        return res.reshape(lead_shape + (nout,))

    return xr.apply_ufunc(
        block,
        *ft.detach(*arrays, *leads),
        input_core_dims=[list(dims)] * len(arrays) + [[]] * len(leads),
        output_core_dims=[[_STAT]],
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {_STAT: nout}, "allow_rechunk": True},
    )


def _empty2d(a):
    """Kernel argument: the array, or an empty one when it is not used."""
    return np.empty(0) if a.size == 0 else a


def _reduce_dims(da, dim):
    if dim is not None:
        dims = [dim] if isinstance(dim, str) else list(dim)
        missing = [d for d in dims if d not in da.dims]
        if missing:
            raise ValueError(f"{missing} are not dimensions of {da.dims}")
        return tuple(d for d in da.dims if d in dims)
    if "range" in da.dims:
        return (ft.ray_dimension(da), "range")
    if "height" in da.dims:
        return ("height",)
    return tuple(da.dims)


def _same_grid(da, like):
    if da is None:
        return None
    return da.broadcast_like(like).transpose(*like.dims)


def _elevation_lead(ds, elevation, like, dims):
    """Elevation (degrees) per row as a DataArray without the reduced dims."""
    if elevation is None:
        elevation = (
            ds["elevation"]
            if isinstance(ds, xr.Dataset) and "elevation" in ds
            else np.nan
        )
    if not isinstance(elevation, xr.DataArray):
        return xr.DataArray(float(elevation))
    present = [d for d in dims if d in elevation.dims]
    return elevation.mean(dim=present) if present else elevation


def _optional_field(ds, value, candidates, what, like=None):
    """A DataArray from a name, a DataArray, or None (not used)."""
    if value is None or value is False:
        return None
    if isinstance(value, xr.DataArray):
        return value
    if isinstance(ds, xr.DataArray):
        raise ValueError(f"with a DataArray as obj, pass {what} as a DataArray")
    return ds[ft.find_field(ds, None if value is True else value, candidates, what)]


# --------------------------------------------------------------------------
# ZDR offset
# --------------------------------------------------------------------------


def _check_statistic(statistic):
    if statistic not in ("median", "mean"):
        raise ValueError(f"statistic must be 'median' or 'mean', not {statistic!r}")


@renamed_arguments(rhohv_min="min_rhohv", snr_min="min_snr")
@provenance("Differential reflectivity offset from natural targets")
def zdr_offset(
    obj,
    method="light_rain",
    *,
    zdr=None,
    dbz=None,
    rhohv=None,
    snr=None,
    mask=None,
    elevation=None,
    reference=None,
    z_range=None,
    min_rhohv=0.98,
    min_snr=None,
    statistic="median",
    min_count=100,
    dim=None,
    band=None,
    temperature=20.0,
    n_threads=None,
    engine="auto",
):
    """
    Offset of the differential reflectivity from vertical pointing, light rain or dry snow.

    The offset is the median (or mean) of ``ZDR - reference`` over the gates
    that pass the selection (``rhohv`` at least ``min_rhohv``, reflectivity
    within ``z_range``, SNR at least ``min_snr``, ``mask`` true). The offset
    is the measured value minus the true one, so ``ZDR - offset`` is the
    corrected differential reflectivity. See :mod:`radarx.retrieve.calibration`
    for the three methods, the reference values, the elevation dependence and
    the radarx choices.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        ZDR (DataArray), a sweep, a quasi-vertical profile (with the
        dimension ``height``) or a volume (sweep by sweep).
    method : {"light_rain", "vertical", "dry_snow"}, optional
        Natural target, default ``"light_rain"``.
    zdr, dbz, rhohv, snr : str or xarray.DataArray, optional
        Fields (names in a Dataset, or arrays). ``dbz`` is needed for the
        reflectivity limits and for ``"light_rain"``; ``rhohv`` and ``snr``
        are used when given, ``rhohv`` also when found under its default
        name. Defaults: ``ZDR``, ``DBZH``, ``RHOHV``.
    mask : xarray.DataArray, optional
        Boolean array that selects the gates (for example the rain below the
        melting layer, or the snow above it). It is required in practice for
        ``"dry_snow"``.
    elevation : float or xarray.DataArray, optional
        Elevation angle in degrees. By default the ``elevation`` variable of
        the sweep; with none the elevation dependence is not applied.
    reference : float, optional
        Intrinsic ZDR (dB) at 0 degrees elevation of the target. Default: 0
        for ``"vertical"``, 0.2 for ``"dry_snow"``, and for
        ``"light_rain"`` the Marshall-Palmer value at the reflectivity of each
        gate. A number replaces it with a constant.
    z_range : tuple of float, optional
        Reflectivity limits (dBZ) of the target. Default: no limits for
        ``"vertical"``, (20, 25) for ``"light_rain"`` and (0, 30) for
        ``"dry_snow"`` (radarx choices).
    min_rhohv : float, optional
        Smallest correlation coefficient. Default 0.98.
    min_snr : float, optional
        Smallest SNR in dB; the SNR field is only used with this.
    statistic : {"median", "mean"}, optional
        The statistic that is reported as the offset. Default ``"median"``,
        which is robust to the outliers that remain in a selection.
    min_count : int, optional
        Fewest gates for an offset; NaN below it. Default 100.
    dim : str or sequence of str, optional
        Dimensions to reduce. Default: the rays and range of a sweep, the
        height of a profile, else all. A quasi-vertical profile with ``time``
        gives one offset per time.
    band : {"S", "C", "X"}, optional
        Radar band for the ``"light_rain"`` reference. Default: from the data.
    temperature : float, optional
        Temperature (degrees Celsius) of the scattering tables. Default 20.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``ZDR_OFFSET`` (dB), ``ZDR_OFFSET_MEAN``, ``ZDR_OFFSET_MEDIAN``,
        ``ZDR_OFFSET_STD`` (the spread of the selected gates),
        ``ZDR_OFFSET_SE`` (the standard error of the mean) and
        ``ZDR_OFFSET_COUNT`` (gates), with the dimensions that were not
        reduced; a DataTree with one node per sweep for a volume.

    Raises
    ------
    KeyError
        If a field is not found.
    ValueError
        For unknown options or a missing reflectivity.
    ImportError
        If ``engine="compiled"`` and the compiled kernel is not available.

    References
    ----------
    Ryzhkov, A. V., S. E. Giangrande, V. M. Melnikov, and T. J. Schuur, 2005:
    Calibration issues of dual-polarization radar measurements. *J. Atmos.
    Oceanic Technol.*, **22** (8), 1138-1155,
    https://doi.org/10.1175/JTECH1772.1
    (the use of vertical pointing, light rain and dry aggregated snow, and the
    adaptation to other elevations; their reference values and selection
    limits are not reproduced, see above).

    Marshall, J. S., and W. McK. Palmer, 1948: The distribution of raindrops
    with size. *J. Meteor.*, **5** (4), 165-166,
    https://doi.org/10.1175/1520-0469(1948)005<0165:TDORWS>2.0.CO;2

    Examples
    --------
    >>> off = radarx.retrieve.zdr_offset(sweep)  # doctest: +SKIP
    >>> off = dtree.radarx.zdr_offset("light_rain", min_snr=15)  # doctest: +SKIP
    >>> corrected = sweep.ZDR - float(off.ZDR_OFFSET)  # doctest: +SKIP
    """
    ft.check_object(obj)
    if method not in _METHOD_RANGE:
        raise ValueError(
            f"method must be 'light_rain', 'vertical' or 'dry_snow', not {method!r}"
        )
    _check_statistic(statistic)
    use_compiled = _use_compiled(engine)
    limits = _METHOD_RANGE[method] if z_range is None else tuple(z_range)
    if len(limits) != 2 or not limits[0] < limits[1]:
        raise ValueError("z_range must be (low, high) with low < high")

    def one(ds, name=None):
        z = ft.primary_field(ds, zdr, ft.ZDR_NAMES, "ZDR")
        dims = _reduce_dims(z, dim)
        need_dbz = (
            method == "light_rain" or np.isfinite(limits[0]) or np.isfinite(limits[1])
        )
        if dbz is not None or need_dbz:
            refl = _optional_field(
                ds, dbz if dbz is not None else True, ft.DBZ_NAMES, "reflectivity"
            )
        else:
            refl = None
        if method == "light_rain" and refl is None:
            raise ValueError("the light_rain method needs the reflectivity")
        rho = _rho_field(ds, rhohv)
        s = (
            _optional_field(ds, True if snr is None else snr, ft.SNR_NAMES, "SNR")
            if min_snr is not None
            else None
        )
        fields = {
            "zdr": z,
            "dbz": _same_grid(refl, z),
            "rhohv": _same_grid(rho, z),
            "snr": _same_grid(s, z),
            "mask": (
                _same_grid(mask.astype(np.float64), z) if mask is not None else None
            ),
        }
        if method == "light_rain" and reference is None:
            b = _band_of(obj, ds, band)
            ref_z, ref_v = _light_rain_curve(b, float(temperature))
            ref_text = f"Marshall-Palmer rain, {b} band, {temperature:g} degC"
        else:
            const = (
                float(reference)
                if reference is not None
                else (0.0 if method == "vertical" else _DRY_SNOW_ZDR)
            )
            ref_z, ref_v = np.zeros(1), np.full(1, const)
            ref_text = f"constant {const:g} dB at 0 degrees elevation"
        lead = _elevation_lead(ds, elevation, z, dims)

        def func(flat, lv):
            return _run_offset(
                flat,
                lv[0],
                ref_z,
                ref_v,
                limits,
                min_rhohv,
                min_snr,
                use_compiled,
                n_threads,
            )

        res = _row_apply(fields, [lead], dims, 5, func)
        return _offset_dataset(
            res, statistic, min_count, method, ref_text, z, limits, min_rhohv
        )

    if isinstance(obj, xr.DataTree):
        return ft.map_sweeps(obj, one, "zdr_offset")
    return one(obj)


def _rho_field(ds, rhohv):
    """The correlation coefficient: as given, or the default field if present."""
    if rhohv is not None and rhohv is not True:
        return _optional_field(ds, rhohv, ft.RHOHV_NAMES, "correlation coefficient")
    if isinstance(ds, xr.Dataset):
        name = ft.find_field(
            ds, None, ft.RHOHV_NAMES, "correlation coefficient", required=False
        )
        return None if name is None else ds[name]
    return None


def _run_offset(
    flat, elev, ref_z, ref_v, limits, min_rhohv, min_snr, use_compiled, n_threads
):
    args = (
        flat["zdr"],
        _empty2d(flat["dbz"]),
        _empty2d(flat["rhohv"]),
        _empty2d(flat["snr"]),
        _empty2d(flat["mask"]),
        elev,
        ref_z,
        ref_v,
        float(limits[0]),
        float(limits[1]),
        float(min_rhohv),
        float(min_snr) if min_snr is not None else -np.inf,
    )
    if use_compiled:
        return _calibration.offset_stats(*args, n_threads=int(n_threads or 0))
    return _offset_stats_numpy(*args)


def _offset_dataset(
    res, statistic, min_count, method, ref_text, like, limits, min_rhohv
):
    def part(i):
        return res.isel({_STAT: i}, drop=True)

    count, mean, std, median = part(0), part(1), part(2), part(3)
    main = mean if statistic == "mean" else median
    enough = count >= max(min_count, 1)
    comment = (
        f"{statistic} of ZDR minus the reference over {method} gates "
        f"(reference: {ref_text}; RHOHV >= {min_rhohv:g}); subtract it from ZDR"
    )
    se = std / np.sqrt(count.where(count > 0))
    attrs = {"units": "dB"}
    out = xr.Dataset(
        {
            "ZDR_OFFSET": main.where(enough)
            .astype(np.float32)
            .assign_attrs(
                long_name="differential reflectivity offset",
                comment=comment,
                **attrs,
            ),
            "ZDR_OFFSET_MEAN": mean.astype(np.float32).assign_attrs(
                long_name="mean of ZDR minus the reference", **attrs
            ),
            "ZDR_OFFSET_MEDIAN": median.astype(np.float32).assign_attrs(
                long_name="median of ZDR minus the reference", **attrs
            ),
            "ZDR_OFFSET_STD": std.astype(np.float32).assign_attrs(
                long_name="standard deviation of ZDR minus the reference", **attrs
            ),
            "ZDR_OFFSET_SE": se.astype(np.float32).assign_attrs(
                long_name="standard error of the mean offset", **attrs
            ),
            "ZDR_OFFSET_COUNT": count.astype(np.int64).assign_attrs(
                long_name="number of gates used", units="1"
            ),
        }
    )
    out.attrs["zdr_offset_method"] = method
    out.attrs["zdr_offset_reference"] = ref_text
    return out


# --------------------------------------------------------------------------
# Z offset from the self-consistency of Z, ZDR and KDP
# --------------------------------------------------------------------------


def _run_consistency(flat, curve, params, use_compiled, n_threads):
    gz, glg = curve
    args = (
        flat["dbz"],
        flat["zdr"],
        flat["kdp"],
        _empty2d(flat["rhohv"]),
        _empty2d(flat["snr"]),
        _empty2d(flat["mask"]),
        gz,
        glg,
        float(params["zdr_offset"]),
        float(params["z_min"]),
        float(params["z_max"]),
        float(params["bin_width"]),
        float(params["rhohv_min"]),
        float(params["snr_min"]),
        float(params["kdp_min"]),
        int(params["min_bin_count"]),
    )
    if use_compiled:
        return _calibration.consistency_stats(*args, n_threads=int(n_threads or 0))
    return _consistency_stats_numpy(*args)


@renamed_arguments(rhohv_min="min_rhohv", snr_min="min_snr", kdp_min="min_kdp")
@provenance("Reflectivity offset from the self-consistency of Z, ZDR and KDP")
def z_offset(
    obj,
    *,
    dbz=None,
    zdr=None,
    kdp=None,
    rhohv=None,
    snr=None,
    mask=None,
    zdr_offset=0.0,
    z_range=(20.0, 45.0),
    bin_width=1.0,
    min_rhohv=0.98,
    min_snr=None,
    min_kdp=0.05,
    min_bin_count=100,
    statistic="median",
    dim=None,
    band=None,
    temperature=20.0,
    mu_lambda="cao2008",
    n_threads=None,
    engine="auto",
):
    """
    Offset of the reflectivity from the self-consistency of Z, ZDR and KDP in rain.

    In classes of ``bin_width`` dB of the measured reflectivity, the mean
    ``ZDR`` (minus ``zdr_offset``) and ``KDP`` give the reflectivity that rain
    must have, ``10 log10(KDP / g(ZDR))``, where ``g`` is the ratio of ``KDP``
    to the linear reflectivity as a function of ``ZDR``. The offset is the
    median (or the gate-weighted mean) over the classes of the measured minus
    that reflectivity; subtract it from the reflectivity. See
    :mod:`radarx.retrieve.calibration` for the method and its limits.

    Use gates in rain only (``mask``), with the attenuation corrected (the
    consistency relation is for the unattenuated reflectivity; the path
    integral of ``KDP`` is the reference, which is why the specific
    attenuation must be removed from ``Z`` first) and a ``ZDR`` that is
    already calibrated.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        The reflectivity (DataArray; then ``zdr`` and ``kdp`` must be arrays),
        a sweep, a profile or a volume.
    dbz, zdr, kdp, rhohv, snr : str or xarray.DataArray, optional
        Fields. Defaults ``DBZH``, ``ZDR``, ``KDP`` (degree km-1, for example
        from :func:`radarx.retrieve.estimate_kdp`) and ``RHOHV``.
    mask : xarray.DataArray, optional
        Boolean array of the rain gates (below the melting layer).
    zdr_offset : float, optional
        Offset of ZDR in dB, from :func:`zdr_offset`. Default 0.
    z_range : tuple of float, optional
        Limits of the reflectivity classes (dBZ). Default (20, 45); above
        that hail and resonance effects make the relations unreliable
        (radarx choice).
    bin_width : float, optional
        Width of a reflectivity class in dB. Default 1, as in Ryzhkov et al.
        (2005).
    min_rhohv : float, optional
        Smallest correlation coefficient. Default 0.98.
    min_snr : float, optional
        Smallest SNR in dB; the SNR field is only used with this.
    min_kdp : float, optional
        Smallest class mean of KDP (degree km-1) to be used. Default 0.05
        (S band); take about twice as much at C band and four times at X
        band.
    min_bin_count : int, optional
        Fewest gates in a class. Default 100.
    statistic : {"median", "mean"}, optional
        Median over the classes (default) or the weighted mean.
    dim : str or sequence of str, optional
        Dimensions to reduce, see :func:`zdr_offset`.
    band, temperature : optional
        Band and temperature of the scattering tables, see :func:`zdr_offset`.
    mu_lambda : str or sequence of float, optional
        Relation between the shape and the slope of the drop size
        distribution, see :func:`radarx.retrieve.dsd`. Default
        ``"cao2008"``.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``Z_OFFSET`` (dB), ``Z_OFFSET_MEDIAN``, ``Z_OFFSET_MEAN``,
        ``Z_OFFSET_STD`` (the spread of the class offsets), ``Z_OFFSET_COUNT``
        (gates) and ``Z_OFFSET_BINS`` (classes), with the dimensions that
        were not reduced.

    Raises
    ------
    KeyError
        If a field is not found.
    ValueError
        For unknown options.
    ImportError
        If ``engine="compiled"`` and the compiled kernel is not available.

    References
    ----------
    Gorgucci, E., G. Scarchilli, and V. Chandrasekar, 1992: Calibration of
    radars using polarimetric techniques. *IEEE Trans. Geosci. Remote Sens.*,
    **30** (5), 853-858, https://doi.org/10.1109/36.175319
    (the self-consistency of Z, ZDR and KDP for calibration; the paper could
    not be read, the title and the use are from the citing literature).

    Ryzhkov, A. V., S. E. Giangrande, V. M. Melnikov, and T. J. Schuur, 2005:
    Calibration issues of dual-polarization radar measurements. *J. Atmos.
    Oceanic Technol.*, **22** (8), 1138-1155,
    https://doi.org/10.1175/JTECH1772.1
    (averaging in classes of 1 dB of Z; the relations of the paper are not used).

    Gourley, J. J., A. J. Illingworth, and P. Tabary, 2009: Absolute
    calibration of radar reflectivity using redundancy of the polarization
    observations and implied constraints on drop shapes. *J. Atmos. Oceanic
    Technol.*, **26** (4), 689-703, https://doi.org/10.1175/2008JTECHA1152.1

    Cao, Q., G. Zhang, E. Brandes, T. Schuur, A. Ryzhkov, and K. Ikeda, 2008:
    Analysis of video disdrometer and polarimetric radar data to characterize
    rain microphysics in Oklahoma. *J. Appl. Meteor. Climatol.*, **47** (8),
    2238-2255, https://doi.org/10.1175/2008JAMC1732.1
    (the shape-slope relation of the drop size distribution).

    Examples
    --------
    >>> off = radarx.retrieve.z_offset(sweep, mask=rain)  # doctest: +SKIP
    >>> z_cal = sweep.DBZH - float(off.Z_OFFSET)  # doctest: +SKIP
    """
    ft.check_object(obj)
    _check_statistic(statistic)
    use_compiled = _use_compiled(engine)
    if not bin_width > 0 or not z_range[1] > z_range[0]:
        raise ValueError("bin_width must be positive and z_range increasing")
    _, coefficients = _dsd_module()._relation(mu_lambda)

    def one(ds, name=None):
        refl = ft.primary_field(ds, dbz, ft.DBZ_NAMES, "reflectivity")
        zd = _optional_field(ds, zdr if zdr is not None else True, ft.ZDR_NAMES, "ZDR")
        kd = _optional_field(ds, kdp if kdp is not None else True, ft.KDP_NAMES, "KDP")
        dims = _reduce_dims(refl, dim)
        rho = _rho_field(ds, rhohv)
        s = (
            _optional_field(ds, True if snr is None else snr, ft.SNR_NAMES, "SNR")
            if min_snr is not None
            else None
        )
        fields = {
            "dbz": refl,
            "zdr": _same_grid(zd, refl),
            "kdp": _same_grid(kd, refl),
            "rhohv": _same_grid(rho, refl),
            "snr": _same_grid(s, refl),
            "mask": (
                _same_grid(mask.astype(np.float64), refl) if mask is not None else None
            ),
        }
        b = _band_of(obj, ds, band)
        curve = _consistency_curve(b, float(temperature), tuple(coefficients))
        params = {
            "zdr_offset": zdr_offset,
            "z_min": z_range[0],
            "z_max": z_range[1],
            "bin_width": bin_width,
            "rhohv_min": min_rhohv,
            "snr_min": -np.inf if min_snr is None else min_snr,
            "kdp_min": min_kdp,
            "min_bin_count": min_bin_count,
        }

        def func(flat, _lv):
            return _run_consistency(flat, curve, params, use_compiled, n_threads)

        res = _row_apply(fields, [], dims, 5, func)
        return _consistency_dataset(res, statistic, b, temperature, zdr_offset)

    if isinstance(obj, xr.DataTree):
        return ft.map_sweeps(obj, one, "z_offset")
    return one(obj)


def _consistency_dataset(res, statistic, band, temperature, zdr_offset):
    def part(i):
        return res.isel({_STAT: i}, drop=True)

    gates, bins, median, mean, std = (part(i) for i in range(5))
    main = mean if statistic == "mean" else median
    attrs = {"units": "dB"}
    out = xr.Dataset(
        {
            "Z_OFFSET": main.astype(np.float32).assign_attrs(
                long_name="reflectivity offset from self-consistency",
                comment=(
                    f"{statistic} over reflectivity classes of the measured minus "
                    "the expected reflectivity of rain from KDP and ZDR; subtract it "
                    "from the reflectivity"
                ),
                **attrs,
            ),
            "Z_OFFSET_MEDIAN": median.astype(np.float32).assign_attrs(
                long_name="median of the class offsets", **attrs
            ),
            "Z_OFFSET_MEAN": mean.astype(np.float32).assign_attrs(
                long_name="gate-weighted mean of the class offsets", **attrs
            ),
            "Z_OFFSET_STD": std.astype(np.float32).assign_attrs(
                long_name="standard deviation of the class offsets", **attrs
            ),
            "Z_OFFSET_COUNT": gates.astype(np.int64).assign_attrs(
                long_name="number of gates in the classes used", units="1"
            ),
            "Z_OFFSET_BINS": bins.astype(np.int32).assign_attrs(
                long_name="number of reflectivity classes used", units="1"
            ),
        }
    )
    out.attrs["z_offset_band"] = band
    out.attrs["z_offset_temperature"] = float(temperature)
    out.attrs["z_offset_zdr_offset"] = float(zdr_offset)
    return out


# --------------------------------------------------------------------------
# system PhiDP
# --------------------------------------------------------------------------


def _circular_median(angles):
    """Circular median in degrees of the finite values of the last axis."""
    a = np.asarray(angles, dtype=np.float64)
    out = np.full(a.shape[:-1] + (3,), np.nan)
    flat = a.reshape(-1, a.shape[-1])
    res = out.reshape(-1, 3)
    for i, row in enumerate(flat):
        v = row[np.isfinite(row)]
        res[i, 2] = v.size
        if not v.size:
            continue
        r = np.deg2rad(v)
        mean = np.arctan2(np.sin(r).mean(), np.cos(r).mean())
        d = np.rad2deg(np.angle(np.exp(1j * (r - mean))))
        med = np.median(d)
        res[i, 0] = _wrap(np.rad2deg(mean) + med)
        res[i, 1] = 1.4826 * np.median(np.abs(d - med))
    return out


def _wrap(angle):
    """Angle in degrees wrapped to (-180, 180]."""
    return 180.0 - np.mod(180.0 - np.asarray(angle), 360.0)


@renamed_arguments(rhohv_min="min_rhohv", snr_min="min_snr")
@provenance("System differential phase offset from the first gates of each ray")
def phidp_offset(
    obj,
    *,
    phidp=None,
    rhohv=None,
    snr=None,
    mask=None,
    min_range=1.0,
    n_gates=20,
    min_gates=5,
    min_rhohv=0.9,
    min_snr=None,
    n_threads=None,
    engine="auto",
):
    """
    System differential phase of a sweep from the first valid gates of each ray.

    Close to the radar the propagation phase is still zero, so the measured
    ``PHIDP`` there is the system offset. For each ray the circular mean of the
    first ``n_gates`` valid gates beyond ``min_range`` is taken (the angles are
    averaged on the circle, so a phase that wraps at 0/360 degrees is no
    problem); the offset of the sweep is the circular median of the rays,
    which is not affected by rays that start in precipitation or clutter that
    changes the phase. The system offset is the quantity that has to be
    subtracted from ``PHIDP`` before attenuation correction or ``KDP``
    (Gourley et al. 2006). The thresholds are radarx choices.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        ``PHIDP`` (DataArray), a sweep or a volume.
    phidp, rhohv, snr : str or xarray.DataArray, optional
        Fields. Defaults ``PHIDP`` (also ``UPHIDP``) and ``RHOHV``; the SNR
        only with ``min_snr``.
    mask : xarray.DataArray, optional
        Boolean array of the gates that may be used.
    min_range : float, optional
        Gates closer than this (km) are not used (near-field and the first
        gates of the receiver recovery). Default 1.
    n_gates : int, optional
        Number of valid gates averaged in a ray. Default 20.
    min_gates : int, optional
        Fewest valid gates in a ray; the ray is NaN otherwise. Default 5.
    min_rhohv : float, optional
        Smallest correlation coefficient. Default 0.9.
    min_snr : float, optional
        Smallest SNR in dB.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``SYSTEM_PHIDP`` (degrees, in (-180, 180]), the offset of the sweep;
        ``SYSTEM_PHIDP_SPREAD``, the scatter of the rays (1.4826 times the
        median absolute deviation, degrees); ``SYSTEM_PHIDP_RAYS``, the number
        of rays used; ``SYSTEM_PHIDP_RAY``, the offset of each ray;
        ``SYSTEM_PHIDP_GATES``, the gates used in each ray and
        ``SYSTEM_PHIDP_CONCENTRATION``, the mean resultant length of each ray
        (1: all gates agree). For a volume a DataTree with one node per sweep.

    Raises
    ------
    KeyError
        If a field is not found.
    ValueError
        For unknown options or an input without ``range``.
    ImportError
        If ``engine="compiled"`` and the compiled kernel is not available.

    References
    ----------
    Gourley, J. J., P. Tabary, and J. Parent du Chatelet, 2006: Data quality of
    the Meteo-France C-band polarimetric radar. *J. Atmos. Oceanic Technol.*,
    **23** (10), 1340-1356, https://doi.org/10.1175/JTECH1912.1
    (a correction of the system offset in the differential propagation phase
    is part of the paper; its procedure is not reproduced).

    Examples
    --------
    >>> off = radarx.retrieve.phidp_offset(sweep)  # doctest: +SKIP
    >>> phidp = sweep.PHIDP - float(off.SYSTEM_PHIDP)  # doctest: +SKIP
    """
    ft.check_object(obj)
    use_compiled = _use_compiled(engine)

    def one(ds, name=None):
        p = ft.primary_field(ds, phidp, ft.PHIDP_NAMES, "PHIDP")
        ray = ft.ray_dimension(p)
        rng_m = np.asarray(ft.range_meters(p).values)
        first = int(np.searchsorted(rng_m, 1000.0 * min_range))
        rho = _rho_field(ds, rhohv)
        s = (
            _optional_field(ds, True if snr is None else snr, ft.SNR_NAMES, "SNR")
            if min_snr is not None
            else None
        )
        fields = {
            "phidp": p.transpose(..., "range"),
            "rhohv": _same_grid(rho, p),
            "snr": _same_grid(s, p),
            "mask": (
                _same_grid(mask.astype(np.float64), p) if mask is not None else None
            ),
        }
        fields = {
            k: (
                v.transpose(*fields["phidp"].dims)
                if v is not None and k != "phidp"
                else v
            )
            for k, v in fields.items()
        }

        def func(flat, _lv):
            args = (
                flat["phidp"],
                _empty2d(flat["rhohv"]),
                _empty2d(flat["snr"]),
                _empty2d(flat["mask"]),
                first,
                int(n_gates),
                float(min_rhohv),
                -np.inf if min_snr is None else float(min_snr),
                int(min_gates),
            )
            if use_compiled:
                return _calibration.phidp_rays(*args, n_threads=int(n_threads or 0))
            return _phidp_rays_numpy(*args)

        res = _row_apply(fields, [], ("range",), 3, func)
        return _phidp_dataset(res, ray, first, n_gates)

    if isinstance(obj, xr.DataTree):
        return ft.map_sweeps(obj, one, "phidp_offset")
    return one(obj)


def _phidp_dataset(res, ray, first, n_gates):
    per_ray = res.isel({_STAT: 0}, drop=True)
    gates = res.isel({_STAT: 1}, drop=True)
    length = res.isel({_STAT: 2}, drop=True)
    summary = xr.apply_ufunc(
        _circular_median,
        per_ray,
        input_core_dims=[[ray]],
        output_core_dims=[["summary"]],
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {"summary": 3}, "allow_rechunk": True},
    )
    deg = {"units": "degree"}
    out = xr.Dataset(
        {
            "SYSTEM_PHIDP": summary.isel(summary=0, drop=True)
            .astype(np.float32)
            .assign_attrs(
                long_name="system differential phase offset",
                comment=(
                    "circular median over the rays of the circular mean of the first "
                    f"{n_gates} valid gates from gate {first}; subtract it from PHIDP"
                ),
                **deg,
            ),
            "SYSTEM_PHIDP_SPREAD": summary.isel(summary=1, drop=True)
            .astype(np.float32)
            .assign_attrs(
                long_name="scatter of the system differential phase between rays",
                comment="1.4826 times the median absolute deviation",
                **deg,
            ),
            "SYSTEM_PHIDP_RAYS": summary.isel(summary=2, drop=True)
            .astype(np.int32)
            .assign_attrs(
                long_name="number of rays with a system differential phase", units="1"
            ),
            "SYSTEM_PHIDP_RAY": per_ray.astype(np.float32).assign_attrs(
                long_name="system differential phase of the ray", **deg
            ),
            "SYSTEM_PHIDP_GATES": gates.astype(np.int32).assign_attrs(
                long_name="number of gates used in the ray", units="1"
            ),
            "SYSTEM_PHIDP_CONCENTRATION": length.astype(np.float32).assign_attrs(
                long_name="mean resultant length of the phase of the gates of the ray",
                units="1",
            ),
        }
    )
    return out


# --------------------------------------------------------------------------
# accessors
# --------------------------------------------------------------------------


@accessor_method(
    "dataarray", "dataset", "datatree", name="zdr_offset", wraps=zdr_offset
)
def _zdr_offset_accessor(self):
    """
    Offset of ZDR from vertical pointing, light rain or dry snow.

    See :func:`radarx.retrieve.zdr_offset` for the parameters.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``ZDR_OFFSET`` (dB) with its statistics.
    """


@accessor_method("dataarray", "dataset", "datatree", name="z_offset", wraps=z_offset)
def _z_offset_accessor(self):
    """
    Offset of the reflectivity from the self-consistency of Z, ZDR and KDP.

    See :func:`radarx.retrieve.z_offset` for the parameters.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``Z_OFFSET`` (dB) with its statistics.
    """


@accessor_method(
    "dataarray", "dataset", "datatree", name="phidp_offset", wraps=phidp_offset
)
def _phidp_offset_accessor(self):
    """
    System differential phase of a sweep.

    See :func:`radarx.retrieve.phidp_offset` for the parameters.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``SYSTEM_PHIDP`` (degrees) and the offset of every ray.
    """
