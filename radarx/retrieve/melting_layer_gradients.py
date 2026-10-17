#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Melting Layer from Profile Gradients
====================================

The melting layer shows in a vertical profile of the polarimetric variables as a
ρhv minimum with maxima of Z and ZDR. Allabakash et al. (2019) [1]_ find its bottom
and top from the gradients of the profiles of Z, ZDR and ρhv and of their
combinations ``Z (1 - ρhv)``, ``ZDR (1 - ρhv)`` and ``Z ZDR (1 - ρhv)``; the
rules refine the algorithm of Wolfensberger et al. (2016) [2]_, which uses the
gradients of Z and ρhv on RHI scans. :func:`melting_layer_gradient` applies the
rules to any profile on ``height`` (a QVP, a range-defined QVP, a columnar
profile or the profile of an RHI), and :func:`melting_layer_rhi` builds the
profile of an RHI scan with :func:`radarx.retrieve.rhi_qvp` first, with the
settings of the paper.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}

References
----------
.. [1] Allabakash, S., S. Lim, and B.-J. Jang, 2019: Melting layer detection and
   characterization based on range height indicator–quasi vertical profiles.
   *Remote Sens.*, **11** (23), 2848, https://doi.org/10.3390/rs11232848
.. [2] Wolfensberger, D., D. Scipion, and A. Berne, 2016: Detection and
   characterization of the melting layer based on polarimetric radar scans.
   *Quart. J. Roy. Meteor. Soc.*, **142** (S1), 108–124,
   https://doi.org/10.1002/qj.2672
"""

from __future__ import annotations

__all__ = ["melting_layer_gradient", "melting_layer_rhi"]

__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np
import xarray as xr

from .._dtypes import fields_as_float32
from .._provenance import provenance
from .._registry import accessor_method
from .._sweepwise import dask_support
from .vertical_profiles import (
    _DBZ_NAMES,
    _DBZ_STANDARD,
    _RHOHV_NAMES,
    _RHOHV_STANDARD,
    _ZDR_NAMES,
    _ZDR_STANDARD,
    _resolve_field,
    rhi_qvp,
)

try:
    from . import _melting_layer_gradients as _kernel

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _kernel = None
    HAS_COMPILED_KERNEL = False

FLAGS = {"none": 0, "detected": 1, "inconsistent": 2}
# the step of the paper that gave the top: 3 (the gradients agree), 4 (the gradient
# of Z or of rhohv is higher) or 5 (the next gate above)
STEPS = (3, 4, 5)


def _use_compiled(engine):
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled melting-layer kernel is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


def _gradient(f, h):
    """
    Derivative along the last axis: second-order central differences inside, first
    order at the ends, on the (possibly irregular) heights ``h``.

    The same arithmetic as the compiled kernel, in the same order.
    """
    g = np.empty_like(f)
    dh = np.diff(h)
    g[:, 0] = (f[:, 1] - f[:, 0]) / dh[0]
    g[:, -1] = (f[:, -1] - f[:, -2]) / dh[-1]
    hd, hs = dh[:-1], dh[1:]
    g[:, 1:-1] = (
        hs * hs * (f[:, 2:] - f[:, 1:-1]) + hd * hd * (f[:, 1:-1] - f[:, :-2])
    ) / (hs * hd * (hd + hs))
    return g


def _extreme(g, maximum):
    """Index of the first maximum (or minimum) that is not NaN, -1 if all are."""
    nan = np.isnan(g)
    filled = np.where(nan, -np.inf if maximum else np.inf, g)
    index = filled.argmax(axis=1) if maximum else filled.argmin(axis=1)
    return np.where(nan.all(axis=1), -1, index)


def _coincident(a, b, c):
    """The index that at least two of the three indices share, -1 if none."""
    out = np.full(a.shape, -1)
    out = np.where((b >= 0) & (b == c), b, out)
    return np.where((a >= 0) & ((a == b) | (a == c)), a, out)


def _normalised(x):
    """``|x / max(x)|`` per profile, and whether the maximum is positive."""
    nan = np.isnan(x)
    peak = np.where(nan, -np.inf, x).max(axis=1)
    ok = peak > 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(x / np.where(ok, peak, 1.0)[:, None]), ok


def _detect_numpy(zh, zdr, rho, h, window=np.inf):
    """NumPy implementation of the compiled kernel (same results)."""
    az, ok_z = _normalised(zh)
    ad, ok_d = _normalised(zdr)
    ar, ok_r = _normalised(rho)
    ok = ok_z & ok_d & ok_r
    c = 1.0 - ar
    q = [_gradient(p, h) for p in (az * c, ad * c, az * ad * c)]
    bottom = _coincident(*(_extreme(g, True) for g in q))
    cmin = _coincident(*(_extreme(g, False) for g in q))
    gz, gr = _gradient(az, h), _gradient(ar, h)
    if np.isfinite(window):
        # only the gates within ``window`` of the minima of the products
        far = np.abs(h[None, :] - h[np.maximum(cmin, 0)][:, None]) > window
        gz, gr = np.where(far, np.nan, gz), np.where(far, np.nan, gr)
    zmin = _extreme(gz, False)
    rmax = _extreme(gr, True)
    agree = (zmin == cmin) & (rmax == cmin)
    higher = np.maximum(zmin, rmax)
    top = np.where(agree, cmin, np.where(higher > cmin, higher, cmin + 1))
    step = np.where(agree, 3, np.where(higher > cmin, 4, 5))
    found = ok & (bottom >= 0) & (cmin >= 0) & (top < h.size)
    return (
        np.where(found, bottom, -1).astype(np.int32),
        np.where(found, top, -1).astype(np.int32),
        np.where(found, step, 0).astype(np.int32),
    )


def _wrap(values, dims, coords, long_name, **attrs):
    return xr.DataArray(
        values,
        dims=dims,
        coords=coords,
        attrs={"long_name": long_name, **attrs},
    )


@dask_support("computes", "the layer is found from the whole profile.")
@provenance(
    "Melting layer from the gradients of Z, ZDR and rhohv profiles after "
    "Allabakash et al. (2019)"
)
def melting_layer_gradient(
    profiles,
    *,
    dbz="auto",
    rhohv="auto",
    zdr="auto",
    height_range=None,
    top_window=None,
    n_threads=None,
    engine="auto",
):
    """
    Melting-layer top and bottom from the gradients of a vertical profile.

    This is the algorithm of Allabakash et al. (2019) [1]_ (Sect. 2.2, Fig. 2),
    who apply it to the profile of RHI scans of an X-band radar and refine the
    method of Wolfensberger et al. (2016) [2]_ that uses Z and ρhv only. For
    every profile (and every time):

    1. Z (dBZ), ZDR (dB) and ρhv are divided by their maximum in the profile
       and the absolute value is taken (``AN(x)``).
    2. The products ``AN(Z) (1 - AN(ρhv))``, ``AN(ZDR) (1 - AN(ρhv))`` and
       ``AN(Z) AN(ZDR) (1 - AN(ρhv))`` are formed, and the gradients of these,
       of ``AN(Z)`` and of ``AN(ρhv)`` are computed along the height.
    3. The bottom is the height at which the gradient maximum of at least two
       of the three products fall together.
    4. The top starts from the height ``c`` at which the gradient minimum of at
       least two of the products fall together. If it is also the height of
       the minimum gradient of ``AN(Z)`` and of the maximum gradient of
       ``AN(ρhv)`` it is the top. Otherwise, if one of these two heights is
       above ``c``, the higher one is the top (ZDR is low in the ice, so the
       ZDR peak lies below the Z and ρhv signature, and the top has to be
       corrected). Otherwise the top is the gate above ``c``.

    No layer is reported where the maxima or the minima of the products do not
    fall together.

    Parameters
    ----------
    profiles : xarray.Dataset
        Profiles on an ascending ``height`` dimension (m above sea level) with Z,
        ZDR and ρhv, e.g. from :func:`radarx.retrieve.qvp`,
        :func:`radarx.retrieve.qvp_timeseries` or :func:`radarx.retrieve.rhi_qvp`.
        Other dimensions (time, azimuth) are kept.
    dbz, rhohv, zdr : str, optional
        Variable names; ``"auto"`` (default) looks for common names and the CF
        standard names.
    height_range : tuple of float, optional
        Only the gates between these heights (m above sea level) are used, also
        for the normalisation. Default: the whole profile (as in the paper).
        Use it to keep the noise at the echo top and the clutter near the ground
        out of the gradients.
    top_window : float, optional
        Radarx's addition, not in the paper: the minimum gradient of Z and the
        maximum gradient of ρhv of step 4 are searched only within this
        distance (m) of the height ``c`` of step 4, instead of in the whole
        profile. In snow Z falls with height and its steepest decrease can lie
        km above the layer, which the whole-profile search takes for the top
        (see Notes). Default None: the whole profile, as in the paper.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use.

    Returns
    -------
    xarray.Dataset
        ``melting_layer_top``, ``melting_layer_bottom`` and
        ``melting_layer_thickness`` in m (the heights above sea level; NaN where
        there is no layer), ``melting_layer_flag`` (0 none, 1 detected, 2
        inconsistent: the top is not above the bottom, heights are NaN) and
        ``melting_layer_top_step`` (the step of the paper that gave the top: 3, 4
        or 5; 0 where there is no layer), on the dimensions of ``profiles``
        except ``height``. The bottom is the variable that
        :func:`radarx.retrieve.mask_above_melting_layer` takes.

    See Also
    --------
    melting_layer_rhi, radarx.retrieve.melting_layer, radarx.retrieve.rhi_qvp

    Notes
    -----
    The paper states the algorithm in words and a flow chart; where it does not
    fix a detail, the choice is radarx's:

    - The gradient is the derivative along the height in metres (second-order
      central differences, as :func:`numpy.gradient`). The paper does not say
      whether it is taken per gate or per metre; the heights of the extremes do
      not depend on it for regular heights.
    - The products use the normalised variables (the order of the flow chart:
      normalise, combine, take gradients). Normalising the products again does
      not move their extremes.
    - Gradient extremes "coincide" where they are at the same gate, and the top
      of step 3 is "consistent" where the same gate holds both gradient
      extremes. The first gate is taken where several gates have the same value.
    - Z is in dBZ and ZDR in dB, as in Fig. 3 of the paper, where the normalised
      Z lies between 0.7 and 1. Negative values enter through their absolute
      values, as the paper defines (``AN``); a profile whose maximum of Z,
      ZDR or ρhv is not positive has no layer.
    - A layer whose top is not above its bottom is flagged (2) and has no
      heights; the paper does not discuss this.
    - Without coinciding minima there is no top, so no layer, although the flow
      chart of the paper only shows the absence for the bottom.
    - The time consistency of :func:`radarx.retrieve.melting_layer` is not
      applied and neither are the quality limits on the heights that the paper
      does not give.
    - ``top_window`` is radarx's addition (default off, the paper's search of the
      whole profile).

    The paper validates the layer of an X-band RHI QVP against the height of the
    0 °C wet-bulb temperature of MERRA reanalysis (correlation 0.96, Sect. 4).
    radarx has no RHI with the three variables and cannot repeat that. On
    QVPs (profiles of 100 m bins, ``height_range=(500, 8000)``,
    ``top_window=500``) the layer of this function was compared with
    :func:`radarx.retrieve.melting_layer` (``"onset"``). Behind the KGWX squall
    line of 31 March 2022 (S band, stratiform rain, 4 volumes between 00:12 and
    01:05 UTC) the top is on average 113 m below the onset top (standard
    deviation 107 m), 60 m below the ERA5 wet-bulb 0 °C height of the radar
    (3.51 km, median top 3.45 km) and the bottom 212 m above the onset bottom.
    On ARMOR (C band, convective rain, 11 April 2008, 15 of 25 profiles have a
    layer) the median top is 268 m below the onset top (correlation 0.04, the
    standard deviation is 355 m), 62 m above the BMX wet-bulb 0 °C height and 209 m
    below its 0 °C height, and the bottom is 411 m above the onset bottom. Without
    ``top_window`` the tops lie 1 to 3 km too high in all four KGWX profiles and in about
    a third of the ARMOR profiles: the minimum gradient of Z of step 4 is searched
    in the whole profile, and in snow Z decreases with height, so its steepest
    decrease is usually far above the layer. Without ``height_range`` and without
    averaging a native 26 m QVP into 100 m bins the gradients are dominated by
    noise at the echo top and at the ground. The method is made for averaged
    profiles of stratiform precipitation; use it with the window on QVPs.

    References
    ----------
    .. [1] Allabakash, S., S. Lim, and B.-J. Jang, 2019: Melting layer detection
       and characterization based on range height indicator–quasi vertical
       profiles. *Remote Sens.*, **11** (23), 2848,
       https://doi.org/10.3390/rs11232848
    .. [2] Wolfensberger, D., D. Scipion, and A. Berne, 2016: Detection and
       characterization of the melting layer based on polarimetric radar scans.
       *Quart. J. Roy. Meteor. Soc.*, **142** (S1), 108–124,
       https://doi.org/10.1002/qj.2672

    Examples
    --------
    >>> profile = radarx.retrieve.rhi_qvp(rhi)  # doctest: +SKIP
    >>> layer = radarx.retrieve.melting_layer_gradient(profile)  # doctest: +SKIP
    """
    use_compiled = _use_compiled(engine)
    names = _field_names(profiles, dbz, rhohv, zdr)
    h = np.asarray(profiles["height"].values, dtype=np.float64)
    if np.any(np.diff(h) <= 0):
        raise ValueError("height must be strictly ascending.")
    keep = _height_selection(h, height_range)
    window = _search_window(top_window)
    template = profiles[names["dbz"]]
    other = [d for d in template.dims if d != "height"]
    shape = tuple(template.sizes[d] for d in other)

    def as2d(name):
        da = profiles[name].transpose(*other, "height")
        values = da.values.reshape(-1, h.size)[:, keep]
        return np.ascontiguousarray(values, np.float64)

    zh, zd, rho = (as2d(names[k]) for k in ("dbz", "zdr", "rhohv"))
    hk = np.ascontiguousarray(h[keep])
    if use_compiled:
        found = _kernel.melting_layer_gradient(
            zh, zd, rho, hk, window, n_threads=int(n_threads or 0)
        )
    else:
        found = _detect_numpy(zh, zd, rho, hk, window)
    layer = _layer(hk, *found)
    coords = {n: c for n, c in template.coords.items() if "height" not in c.dims}
    return fields_as_float32(_layer_dataset(layer, other, shape, coords))


def _field_names(profiles, dbz, rhohv, zdr):
    """Names of the Z, rhohv and ZDR profiles of a Dataset with ``height``."""
    if not isinstance(profiles, xr.Dataset) or "height" not in profiles.dims:
        raise ValueError("profiles must be a Dataset with a 'height' dimension.")
    names = {}
    for key, value, cands, std in (
        ("dbz", dbz, _DBZ_NAMES, _DBZ_STANDARD),
        ("rhohv", rhohv, _RHOHV_NAMES, _RHOHV_STANDARD),
        ("zdr", zdr, _ZDR_NAMES, _ZDR_STANDARD),
    ):
        names[key] = _resolve_field(profiles, value, cands, std, key)
        if names[key] is None:
            raise ValueError(f"melting_layer_gradient needs a {key} profile.")
    return names


def _height_selection(h, height_range):
    """Boolean mask of the heights used."""
    keep = np.ones(h.size, dtype=bool)
    if height_range is not None:
        if len(height_range) != 2 or not height_range[0] < height_range[1]:
            raise ValueError("height_range must be (lowest, highest) in metres.")
        keep = (h >= height_range[0]) & (h <= height_range[1])
    if keep.sum() < 3:
        raise ValueError("a profile needs at least three heights.")
    return keep


def _search_window(top_window):
    """The window of step 4 in metres (infinite for ``None``)."""
    if top_window is None:
        return np.inf
    window = float(top_window)
    if not window >= 0.0:
        raise ValueError("top_window must not be negative.")
    return window


def _layer(h, bottom, top, step):
    """Heights, flag and step from the gate indices of the detection."""
    found = bottom >= 0
    top_h = np.where(found, h[np.where(found, top, 0)], np.nan)
    bottom_h = np.where(found, h[np.where(found, bottom, 0)], np.nan)
    consistent = found & (top_h > bottom_h)
    flag = np.where(consistent, FLAGS["detected"], FLAGS["none"])
    flag = np.where(found & ~consistent, FLAGS["inconsistent"], flag)
    return {
        "top": np.where(consistent, top_h, np.nan),
        "bottom": np.where(consistent, bottom_h, np.nan),
        "flag": flag.astype(np.int8),
        "step": np.where(consistent, step, 0).astype(np.int8),
    }


def _layer_dataset(layer, dims, shape, coords):
    """Dataset of the layer variables on ``dims``."""
    height = dict(standard_name="altitude", units="m")
    top, bottom = layer["top"], layer["bottom"]

    def wrap(values, long_name, **attrs):
        return _wrap(values.reshape(shape), dims, coords, long_name, **attrs)

    out = xr.Dataset(
        {
            "melting_layer_top": wrap(
                top, "melting layer top above sea level", **height
            ),
            "melting_layer_bottom": wrap(
                bottom, "melting layer bottom above sea level", **height
            ),
            "melting_layer_thickness": wrap(
                top - bottom, "thickness of the melting layer", units="m"
            ),
            "melting_layer_flag": wrap(
                layer["flag"],
                "melting layer detection flag",
                flag_values=np.array(list(FLAGS.values()), dtype=np.int8),
                flag_meanings=" ".join(FLAGS),
            ),
            "melting_layer_top_step": wrap(
                layer["step"],
                "step of the algorithm that gave the melting layer top",
                comment="3: the gradients agree; 4: the gradient of Z or of rhohv "
                "gives a higher top; 5: the gate above the agreeing minima; 0: no layer",
            ),
        }
    )
    out.attrs["melting_layer_method"] = "gradients of Z, ZDR and rhohv"
    return out


@dask_support("computes", "the profile averages over all gates of the scan.")
@provenance(
    "Melting layer from the gradients of the profile of an RHI scan after "
    "Allabakash et al. (2019)"
)
def melting_layer_rhi(
    rhi,
    *,
    sweep=0,
    rmin=6000.0,
    rmax=20000.0,
    max_elevation=45.0,
    min_rhohv=0.6,
    min_dbz=0.0,
    dz=100.0,
    dbz="auto",
    rhohv="auto",
    zdr="auto",
    height_range=None,
    top_window=None,
    n_threads=None,
    engine="auto",
):
    """
    Melting-layer top and bottom of an RHI scan.

    The range of the gates of the RHI is turned into height and the gates of all
    elevations within a height bin are averaged (the range-height-indicator
    quasi-vertical profile of Allabakash et al. 2019 [1]_, see
    :func:`radarx.retrieve.rhi_qvp`). The gradients of the profile give the layer
    (:func:`melting_layer_gradient`). A sequence of RHIs gives a time series.

    Parameters
    ----------
    rhi : xarray.Dataset, xarray.DataTree or sequence of them
        An RHI sweep, a volume of RHI sweeps (then ``sweep`` is used) or a
        sequence of them in time order, as for :func:`radarx.retrieve.rhi_qvp`.
    sweep : int or str, optional
        The sweep of a volume to use. Default 0.
    rmin, rmax : float, optional
        Range limits (m) of the gates used. Default 6000 and 20000 (the paper).
    max_elevation : float, optional
        Only rays below this elevation (degrees) are used. Default 45 (the paper).
    min_rhohv, min_dbz : float or None, optional
        Gates with ``rhohv > min_rhohv`` and ``Z >= min_dbz`` are used. Default
        0.6 and 0 dBZ (the paper).
    dz : float, optional
        Height of the bins (m). Default 100 (the paper).
    dbz, rhohv, zdr : str, optional
        Variable names; ``"auto"`` (default) looks for common names.
    height_range : tuple of float, optional
        Heights (m above sea level) used by the detection, as in
        :func:`melting_layer_gradient`.
    top_window : float, optional
        Search window of step 4 (m), as in :func:`melting_layer_gradient`.
    n_threads : int, optional
        Threads for the compiled kernels. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use.

    Returns
    -------
    xarray.Dataset
        The variables of :func:`melting_layer_gradient` at the azimuth of the RHI
        (and the time of the scan).

    See Also
    --------
    melting_layer_gradient, radarx.retrieve.rhi_qvp

    Notes
    -----
    The settings of the profile are those of the paper (X-band RHIs of 5 min
    spacing, an azimuth of 109 degrees); the profile has the radar's beam
    broadening above 6 km range, which the paper limits by ``rmax`` and
    ``max_elevation`` and does not correct. See :func:`melting_layer_gradient`
    for the algorithm and for what it leaves open.

    References
    ----------
    .. [1] Allabakash, S., S. Lim, and B.-J. Jang, 2019: Melting layer detection
       and characterization based on range height indicator–quasi vertical
       profiles. *Remote Sens.*, **11** (23), 2848,
       https://doi.org/10.3390/rs11232848

    Examples
    --------
    >>> layer = radarx.retrieve.melting_layer_rhi(rhi)  # doctest: +SKIP
    """
    profiles = rhi_qvp(
        rhi,
        sweep=sweep,
        rmin=rmin,
        rmax=rmax,
        max_elevation=max_elevation,
        min_rhohv=min_rhohv,
        min_dbz=min_dbz,
        rhohv=rhohv,
        dbz=dbz,
        dz=dz,
        n_threads=n_threads,
        engine=engine,
    )
    return melting_layer_gradient(
        profiles,
        dbz=dbz,
        rhohv=rhohv,
        zdr=zdr,
        height_range=height_range,
        top_window=top_window,
        n_threads=n_threads,
        engine=engine,
    )


@accessor_method("dataset", name="melting_layer_gradient", wraps=melting_layer_gradient)
def _melting_layer_gradient_accessor(self):
    """
    Melting-layer top and bottom from the gradients of these profiles.

    Returns
    -------
    xarray.Dataset
        ``melting_layer_top``, ``melting_layer_bottom``,
        ``melting_layer_thickness``, ``melting_layer_flag`` and
        ``melting_layer_top_step``.

    See Also
    --------
    radarx.retrieve.melting_layer_gradient
    """


@accessor_method(
    "dataset", "datatree", name="melting_layer_rhi", wraps=melting_layer_rhi
)
def _melting_layer_rhi_accessor(self):
    """
    Melting-layer top and bottom of this RHI scan.

    Returns
    -------
    xarray.Dataset
        ``melting_layer_top``, ``melting_layer_bottom``,
        ``melting_layer_thickness``, ``melting_layer_flag`` and
        ``melting_layer_top_step``.

    See Also
    --------
    radarx.retrieve.melting_layer_rhi
    """
