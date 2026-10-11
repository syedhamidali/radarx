#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx Azimuthal Shear and Radial Divergence
============================================

Azimuthal shear and radial divergence of the Doppler velocity with the
linear least-squares derivative (LLSD) technique (Smith and Elmore 2004;
Miller et al. 2013; Mahalik et al. 2019).

Provenance. The local linear model ``v = a + b s + c dr`` fitted by weighted
least squares in a window that keeps a nearly constant width in metres, its
reading as half the vertical vorticity and half the horizontal divergence for
a symmetric wind field, and the default window (3 range gates of 250 m deep,
about 2500 m wide) are those of the conference preprint of Smith and Elmore
(2004), which was read in full (7 pages, the copy at
``ams.confex.com/ams/pdfpapers/81827.pdf``; earlier notes of radarx say that
only the abstract was read). Not taken from it: the exact solution of the
2 x 2 normal equations for windows that are not symmetric (Smith and Elmore
assume symmetric windows and weights, which makes the normal equations
diagonal), the Gaussian weights, the ``min_valid_fraction`` and the minimum of
three valid gates, which are radarx choices. Smith and Elmore pass the
velocities through a 3 x 3 median filter first; that is the option
``median_filter=3`` (off by default, so that the default numbers do not
change). The preprint does not say how missing gates enter the median; the
rule here (the gate must be valid and the window must hold more than half of
its gates) is a radarx choice. Miller et al. (2013) and Mahalik et al. (2019)
use the LLSD derivatives; nothing else in this module is attributed to them.

:func:`azimuthal_shear_layer_max` computes the layer-maximum azimuthal shear
of the MRMS system (0 to 2 km and 3 to 6 km above ground level) from a
volume, after the description in Smith et al. (2016), which was read in full:
"the maximum value of azimuthal shear is computed for two vertical layers
once every 2 min: 0-2 and 3-6 km above ground level (AGL). At each tilt, the
azimuthal shear values that are not in or near a storm cell (as defined by
reflectivity QC greater than or equal to 20 dBZ within 5 km of the data
point) are removed prior to taking the maximum in each layer at each range."
Miller et al. (2013) could not be read (no open copy was found), so nothing
in the function is attributed to it. How the maximum is located on a grid
(the nearest ray and gate of every tilt at the ground range and azimuth of a
reference tilt, beam heights from the 4/3 Earth model of
:mod:`radarx.fundamentals.geometry`) is a radarx choice; the MRMS system works
on its own 3-D grid with several radars.

At every gate, the radial velocity of the gates in a window of fixed
physical size is fitted by weighted least squares with the plane

.. math::

    v \\approx a + b\\,s + c\\,\\Delta r, \\qquad
    s = r\\,\\Delta\\theta, \\quad \\Delta r = r - r_0,

where :math:`\\Delta\\theta` is the azimuth difference to the centre ray and
:math:`r` the range of each gate. The slope :math:`b` is the azimuthal shear
:math:`\\partial v / \\partial s` and :math:`c` the radial divergence
:math:`\\partial v / \\partial r`, both in s⁻¹. For a solid-body vortex the
azimuthal shear is half the vertical vorticity, and for axisymmetric
convergence the radial divergence is half the horizontal divergence.

Because the window has a fixed size in metres, the number of rays in it
shrinks with range. The measured ray azimuths are used (no assumed spacing),
and the window wraps around north. A compiled C++ kernel does the work; if it
is not available, an equivalent NumPy implementation is used.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = [
    "azimuthal_shear",
    "azimuthal_shear_layer_max",
    "llsd",
    "radial_divergence",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import warnings

import numpy as np
import xarray as xr

from .._compat import renamed_arguments
from .._fields import VELOCITY, as_sweep, find_field
from .._provenance import provenance
from .._registry import accessor_for
from ._products import product_tree

try:
    from . import _shear

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _shear = None
    HAS_COMPILED_KERNEL = False

_REFERENCES = """
    References
    ----------
    .. [1] Smith, T. M., and K. L. Elmore, 2004: The use of radial velocity
       derivative to diagnose rotation and divergence. *Preprints, 11th Conf.
       on Aviation, Range, and Aerospace Meteorology*, Hyannis, MA, Amer.
       Meteor. Soc., P5.6 (conference preprint, no DOI),
       https://ams.confex.com/ams/pdfpapers/81827.pdf
    .. [2] Miller, M. L., V. Lakshmanan, and T. M. Smith, 2013: An automated
       method for depicting mesocyclone paths and intensities. *Wea.
       Forecasting*, **28**, 570-585,
       https://doi.org/10.1175/WAF-D-12-00065.1
    .. [3] Mahalik, M. C., B. R. Smith, K. L. Elmore, D. M. Kingfield, K. L.
       Ortega, and T. M. Smith, 2019: Estimates of gradients in radar moments
       using a linear least squares derivative technique. *Wea. Forecasting*,
       **34**, 415-434, https://doi.org/10.1175/WAF-D-18-0095.1
"""

_ATTRS = {
    "azimuthal_shear": {
        "long_name": "azimuthal shear of radial velocity",
        "units": "s-1",
        "comment": "linear least-squares derivative dv/ds along the arc s = r*dtheta",
    },
    "radial_divergence": {
        "long_name": "radial divergence of radial velocity",
        "units": "s-1",
        "comment": "linear least-squares derivative dv/dr along the beam",
    },
}


def _use_compiled(engine):
    """Whether to run the compiled kernel for the requested ``engine``."""
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled LLSD kernel is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


def _wrap(angle):
    """Angle wrapped to [-pi, pi)."""
    return np.mod(angle + np.pi, 2.0 * np.pi) - np.pi


def _median_numpy(vel, size):
    """
    ``size`` x ``size`` median filter of azimuth-sorted velocities (NumPy).

    Same rule as the compiled kernel: rays wrap around north, a gate keeps a
    value only if it is valid itself and its window (cut at the first and last
    gate) holds more than half of ``size * size`` valid gates; the value is
    the median of the valid gates of the window.
    """
    nray, ngate = vel.shape
    h = size // 2
    nr = min(2 * h + 1, nray)
    need = size * size // 2 + 1
    padded = np.full((nray, ngate + 2 * h), np.nan)
    padded[:, h : h + ngate] = vel
    stack = [
        np.roll(padded, -(d - (nr - 1) // 2), axis=0)[:, k : k + ngate]
        for d in range(nr)
        for k in range(2 * h + 1)
    ]
    window = np.stack(stack)
    count = np.isfinite(window).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(window, axis=0)
    return np.where(np.isfinite(vel) & (count >= need), med, np.nan)


def _llsd_numpy(
    data,
    azimuth,
    rng,
    window_range,
    window_azimuth,
    gaussian,
    min_frac,
    median_size=0,
):
    """
    NumPy implementation of the compiled kernel (same results).

    Sums every (ray, gate) pair of the window directly instead of using
    cumulative sums, so it doubles as an independent test oracle. Gaussian
    weights are ``exp(-2 d^2 / h^2)`` with ``h`` the half window, i.e. a
    standard deviation of ``h / 2`` (a quarter of the window), a radarx choice.
    The 2 x 2 normal equations are solved after removing the weighted means;
    Smith and Elmore (2004) obtain the diagonal system only for symmetric
    windows and weights. The thresholds in ``good`` (at least 3 valid gates,
    ``min_frac`` of the window, determinant above 1e-10 of the product of the
    variances) are radarx choices.
    """
    nray, ngate = data.shape
    if np.any(np.diff(rng) <= 0):
        raise ValueError("range must increase")
    order = np.argsort(np.mod(azimuth, 360.0), kind="stable")
    az = np.radians(np.mod(azimuth, 360.0))[order]
    vel = data[order]
    if median_size:
        vel = _median_numpy(vel, median_size)
    az_step = np.median(np.diff(az))
    r_step = np.median(np.diff(rng))

    half_r = max(0.5 * window_range, 1.5 * r_step)
    with np.errstate(divide="ignore"):
        ha = np.where(rng > 0, 0.5 * window_azimuth / rng, np.pi)
    ha = np.maximum(np.minimum(ha, 0.5 * np.pi), 1.5 * az_step)
    k0 = np.searchsorted(rng, rng - half_r, side="left")
    k1 = np.searchsorted(rng, rng + half_r, side="right") - 1
    n_gates = (k1 - k0 + 1).astype(float)
    gate = np.arange(ngate)
    kmax = int(max((gate - k0).max(), (k1 - gate).max()))

    names = ("n_total", "n_valid", "w", "ws", "wr", "wss", "wsr", "wrr", "wv")
    names += ("wvs", "wvr")
    S = {name: np.zeros((nray, ngate)) for name in names}
    for d in range(nray):
        nb = np.roll(np.arange(nray), -d)  # neighbour of each centre ray
        dt = _wrap(az[nb] - az)
        if d and np.abs(dt).min() > ha[0]:
            continue  # this offset is outside every window
        # windows shrink with range: only gates up to the widest need it
        g_end = int(np.searchsorted(-ha, -np.abs(dt).min(), side="right"))
        if d == 0:
            g_end = ngate
        sl = slice(0, g_end)
        inside = np.abs(dt)[:, None] <= ha[None, sl]
        S["n_total"][:, sl] += np.where(inside, n_gates[None, sl], 0.0)
        vnb = vel[nb]
        wa = np.exp(-2.0 * dt[:, None] ** 2 / ha[None, sl] ** 2) if gaussian else 1.0
        for j in range(-kmax, kmax + 1):
            k = gate[sl] + j
            use = (k >= k0[sl]) & (k <= k1[sl])
            kc = np.clip(k, 0, ngate - 1)
            x = vnb[:, kc]
            ok = inside & use[None, :] & np.isfinite(x)
            if not ok.any():
                continue
            dr = rng[kc] - rng[sl]
            s = rng[kc][None, :] * dt[:, None]
            w = wa * np.exp(-2.0 * dr**2 / half_r**2)[None, :] if gaussian else 1.0
            w = np.where(ok, w, 0.0)
            x = np.where(ok, x, 0.0)
            S["n_valid"][:, sl] += ok
            S["w"][:, sl] += w
            S["ws"][:, sl] += w * s
            S["wr"][:, sl] += w * dr
            S["wss"][:, sl] += w * s * s
            S["wsr"][:, sl] += w * s * dr
            S["wrr"][:, sl] += w * dr * dr
            S["wv"][:, sl] += w * x
            S["wvs"][:, sl] += w * x * s
            S["wvr"][:, sl] += w * x * dr

    with np.errstate(invalid="ignore", divide="ignore"):
        ms = S["ws"] / S["w"]
        mr = S["wr"] / S["w"]
        css = S["wss"] - S["ws"] * ms
        crr = S["wrr"] - S["wr"] * mr
        csr = S["wsr"] - S["ws"] * mr
        cvs = S["wvs"] - S["wv"] * ms
        cvr = S["wvr"] - S["wv"] * mr
        det = css * crr - csr * csr
        shear = (cvs * crr - cvr * csr) / det
        div = (cvr * css - cvs * csr) / det
        good = (
            np.isfinite(vel)
            & (S["n_valid"] >= 3)
            & (S["n_valid"] >= min_frac * S["n_total"])
            & (S["w"] > 0)
            & (css > 0)
            & (crr > 0)
            & (det > 1e-10 * css * crr)
        )
    out = np.full((2, nray, ngate), np.nan, dtype=np.float32)
    out[0, order] = np.where(good, shear, np.nan)
    out[1, order] = np.where(good, div, np.nan)
    return out


def _as_gate_mask(mask, ds, da):
    """Boolean DataArray on the field's dimensions from a name or array."""
    if isinstance(mask, str):
        mask = ds[mask]
    elif not isinstance(mask, xr.DataArray):
        mask = xr.DataArray(np.asarray(mask), dims=da.dims)
    return mask.astype(bool)


def _sweep_arrays(ds, field, mask, exclude, floor):
    """Velocity on (ray, range) and contiguous float64 arrays of one sweep."""
    if field not in ds:
        raise KeyError(f"{field!r} is not in the sweep")
    da = ds[field]
    if "range" not in da.dims or da.ndim != 2:
        raise ValueError(f"{field!r} must be 2-D with a 'range' dimension")
    ray_dim = da.dims[0] if da.dims[1] == "range" else da.dims[1]
    da_t = da.transpose(ray_dim, "range")
    if floor is not None:
        da_t = da_t.where(~(da_t <= floor))
    if mask is not None:
        da_t = da_t.where(_as_gate_mask(mask, ds, da))
    if exclude is not None:
        da_t = da_t.where(~_as_gate_mask(exclude, ds, da))
    data = np.ascontiguousarray(da_t.values, dtype=np.float64)
    azimuth = np.ascontiguousarray(ds["azimuth"].values, dtype=np.float64)
    rng = np.ascontiguousarray(ds["range"].values, dtype=np.float64)
    if azimuth.shape != (data.shape[0],):
        raise ValueError("the sweep needs one azimuth per ray")
    return da, da_t, (data, azimuth, rng)


def _median_size(median_filter):
    """Edge of the median window (0 = off) from ``None``, ``False``, ``True`` or an odd int."""
    if median_filter is None or median_filter is False:
        return 0
    if median_filter is True:
        return 3
    if (
        not isinstance(median_filter, (int, np.integer))
        or median_filter < 1
        or median_filter % 2 == 0
    ):
        raise ValueError(
            "median_filter must be None, True (3 x 3) or an odd window size, "
            f"not {median_filter!r}"
        )
    return int(median_filter)


def _nodata_floor(ds, field, nodata, root_attrs):
    """No-data floor of ``field`` (NEXRAD Level II flag codes), or None."""
    from .qc import _NEXRAD_FIELD_NODATA, _is_nexrad

    if nodata is None or not (nodata == "nexrad" or _is_nexrad(ds, root_attrs)):
        return None
    return _NEXRAD_FIELD_NODATA.get(field)


def _llsd_sweeps(
    sweeps,
    field,
    window,
    weights,
    min_valid_fraction,
    mask,
    exclude,
    nodata,
    root_attrs,
    n_threads,
    engine,
    median_filter=None,
):
    """LLSD on a list of sweep Datasets in one kernel call; one Dataset each."""
    median_size = _median_size(median_filter)
    prepared = [
        _sweep_arrays(
            ds, field, mask, exclude, _nodata_floor(ds, field, nodata, root_attrs)
        )
        for ds in sweeps
    ]
    window_range, window_azimuth = (float(w) for w in window)
    gaussian = weights == "gaussian"
    options = (window_range, window_azimuth, gaussian)
    if _use_compiled(engine):
        data, azimuth, rng = (list(a) for a in zip(*(p[2] for p in prepared)))
        outs = _shear.llsd(
            data,
            azimuth,
            rng,
            *options,
            min_valid_fraction=float(min_valid_fraction),
            n_threads=int(n_threads or 0),
            median_size=median_size,
        )
    else:
        outs = [
            _llsd_numpy(*p[2], *options, float(min_valid_fraction), median_size)
            for p in prepared
        ]

    method = {
        "method": "linear least-squares derivative (LLSD)",
        "source_field": field,
        "window_range_m": window_range,
        "window_azimuth_m": window_azimuth,
        "weights": weights,
        "min_valid_fraction": float(min_valid_fraction),
    }
    if median_size:
        method["median_filter_size"] = median_size
    results = []
    for (da, da_t, _), out in zip(prepared, outs):
        result = xr.Dataset(coords=da_t.coords)
        for k, name in enumerate(("azimuthal_shear", "radial_divergence")):
            result[name] = xr.DataArray(
                out[k], dims=da_t.dims, coords=da_t.coords, attrs=_ATTRS[name] | method
            ).transpose(*da.dims)
        results.append(result)
    return results


@provenance(
    "Azimuthal shear and radial divergence by LLSD (Smith and Elmore 2004)",
    extra_refs=("smith-elmore-2004",),
)
def llsd(
    obj,
    field=None,
    window=(750.0, 2500.0),
    *,
    weights="uniform",
    min_valid_fraction=0.5,
    mask=None,
    exclude=None,
    nodata="auto",
    median_filter=None,
    n_threads=None,
    engine="auto",
):
    """
    Azimuthal shear and radial divergence by linear least-squares derivatives.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataTree
        A PPI sweep with ``azimuth`` and ``range`` coordinates (e.g. from
        xradar), a single sweep node of a volume (``dtree["sweep_0"]``), or a
        volume whose ``sweep_*`` nodes are processed one by one.
    field : str, optional
        Radial velocity field in m s⁻¹. Default: the first of ``VRADH``,
        ``VRAD``, ``VRADV``, ``VEL``, ``VELOCITY``, ``velocity`` that the
        sweep holds. Velocities must be dealiased (or free of aliasing); folds
        produce spurious shear.
    window : tuple of float, optional
        Window size ``(range_m, azimuth_m)`` in metres: its length along the
        beam and its arc length across it. Default ``(750, 2500)``. The window
        covers at least one neighbouring gate and ray on each side and at
        most ±90° in azimuth.
    weights : {"uniform", "gaussian"}, optional
        Uniform weights (default) or a Gaussian with a standard deviation of a
        quarter of the window in each direction.
    min_valid_fraction : float, optional
        Minimum fraction of the gates in the window that must hold valid data.
        Default 0.5. The centre gate must always be valid.
    mask : str or xarray.DataArray, optional
        Gates to use (``True`` = keep, as in :func:`radarx.retrieve.hid`,
        :func:`radarx.retrieve.dsd` and :func:`radarx.retrieve.echo_mask`),
        e.g. the output of ``echo_mask``, or the name of such a variable.
        Other gates are left out of the fit. Before this version ``True``
        meant excluded; use ``exclude`` for that.
    exclude : str or xarray.DataArray, optional
        Gates to leave out of the fit (``True`` = excluded). Can be combined
        with ``mask``.
    nodata : {"auto", "nexrad", None}, optional
        Flag values that are not data. ``"auto"`` (default) treats the
        NEXRAD Level II codes decoded by xradar (velocity at or below
        -63.9 m/s, the floor ``echo_mask`` and ``apply_mask`` use) as
        missing for NEXRAD volumes; ``"nexrad"`` does it for any input,
        None never.
    median_filter : None, True or int, optional
        Median filter of the velocities before the fit, as in Smith and Elmore
        (2004) [1]_ (a 3 x 3 filter). ``None`` (default) or ``False`` applies
        none, ``True`` a 3 x 3 window and an odd integer a square window of
        that edge, in rays and gates (rays wrap around north). A gate keeps a
        value only if it is valid itself and the window holds more than half
        of its gates; the value is the median of the valid ones. The
        treatment of missing gates is a radarx choice: the preprint does not
        describe it. A median filter smooths the isolated velocity spikes
        that give spurious shear, and also lowers the peak shear of small
        vortices.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled
        kernel and falls back to NumPy.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``azimuthal_shear`` and ``radial_divergence`` in s⁻¹ (float32) on the
        sweep's dimensions and coordinates; a DataTree with one such node per
        sweep that has ``field`` and the root of the input if a DataTree was
        given. Merge them into the input with ``.radarx.assign(products)``.

    Notes
    -----
    The plane :math:`v = a + b\\,s + c\\,\\Delta r` with :math:`s = r\\,\\Delta\\theta`
    is fitted to the window around every gate. Its slopes are the azimuthal
    shear :math:`b` and radial divergence :math:`c`. The window holds the rays
    with :math:`r_0 |\\Delta\\theta| \\le` ``azimuth_m / 2`` and the gates with
    :math:`|r - r_0| \\le` ``range_m / 2``, so the number of rays changes
    with range. Measured ray azimuths are used and the window wraps around
    north. With uniform weights the kernel uses cumulative sums along range,
    so each gate costs one lookup per ray in its window.

    For solid-body rotation the azimuthal shear is half the vertical
    vorticity; for axisymmetric convergence the radial divergence is half the
    horizontal divergence. Smith and Elmore (2004) [1]_ state this as an
    approximation that holds for a symmetric wind field (mesocyclone, symmetric
    downburst) and breaks down for asymmetric features such as gust fronts. For
    solid-body rotation :math:`v_\\theta = \\Omega r` one gets :math:`\\partial v /
    \\partial s = \\Omega = \\zeta / 2`, which follows from elementary geometry.

    What is taken from the literature. The fitted plane, the arc coordinate
    :math:`s = r\\,\\Delta\\theta`, the weighted least-squares fit and the
    fixed-width window whose number of rays shrinks with range are those of
    Smith and Elmore (2004) [1]_. Their kernel is 3 range gates deep and about
    2500 m wide (also 5000 and 8000 m), with at least 3 radials; the radarx
    defaults ``window = (750, 2500)`` are 3 gates of 250 m and 2500 m, the
    smallest kernel of that paper. The window was set in metres, so for other
    gate spacings the depth is not 3 gates. Miller et al. (2013) [2]_ and
    Mahalik et al. (2019) [3]_ apply and evaluate the LLSD derivatives; their
    window sizes and weights are not used here. The Gaussian weights (a
    standard deviation of a quarter of the window), ``min_valid_fraction``
    (0.5) and the requirement of at least 3 valid gates are radarx choices.
    {references}
    Examples
    --------
    >>> out = radarx.retrieve.llsd(sweep, "VRADH")  # doctest: +SKIP
    >>> out["azimuthal_shear"].plot()  # doctest: +SKIP
    """
    if weights not in ("uniform", "gaussian"):
        raise ValueError(f"weights must be 'uniform' or 'gaussian', not {weights!r}")
    if len(window) != 2 or min(window) <= 0:
        raise ValueError("window must be two positive lengths (range_m, azimuth_m)")
    options = {
        "window": window,
        "weights": weights,
        "min_valid_fraction": min_valid_fraction,
        "mask": mask,
        "exclude": exclude,
        "nodata": nodata,
        "median_filter": median_filter,
        "n_threads": n_threads,
        "engine": engine,
    }
    _median_size(median_filter)
    if nodata not in ("auto", "nexrad", None):
        raise ValueError(f"nodata must be 'auto', 'nexrad' or None, not {nodata!r}")
    obj = as_sweep(obj)
    field = find_field(obj, VELOCITY, field)
    if isinstance(obj, xr.Dataset):
        return _llsd_sweeps([obj], field, root_attrs=None, **options)[0]
    if hasattr(obj, "children"):  # DataTree: all sweeps in one kernel call
        names, sweeps = [], []
        for name, node in obj.children.items():
            if name.startswith("sweep") and field in node.data_vars:
                try:
                    ds = node.to_dataset(inherit="all_coords")
                except (TypeError, ValueError):  # xarray without "all_coords"
                    ds = node.to_dataset()
                names.append(name)
                sweeps.append(ds)
        if not sweeps:
            raise ValueError(f"No sweep contains {field!r}.")
        results = _llsd_sweeps(sweeps, field, root_attrs=obj.root.attrs, **options)
        return product_tree(obj, dict(zip(names, results)))
    raise TypeError("llsd needs an xarray.Dataset sweep or an xarray.DataTree")


llsd.__doc__ = llsd.__doc__.replace("{references}", _REFERENCES)


@provenance(
    "Azimuthal shear by LLSD (Smith and Elmore 2004)", extra_refs=("smith-elmore-2004",)
)
def azimuthal_shear(obj, field=None, window=(750.0, 2500.0), **kwargs):
    """
    Azimuthal shear (s⁻¹) of the radial velocity by LLSD.

    Same parameters as :func:`llsd`.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        ``azimuthal_shear`` on the sweep's dimensions and coordinates (a
        DataTree of sweeps and the input root if a DataTree was given).

    See Also
    --------
    llsd

    Notes
    -----
    Method: the LLSD of Smith and Elmore (2004) [1]_, as used by Miller et al.
    (2013) [2]_ and Mahalik et al. (2019) [3]_; see :func:`llsd` for what is
    taken from them.
    {references}"""
    return _select(llsd(obj, field, window, **kwargs), "azimuthal_shear")


@provenance(
    "Radial divergence by LLSD (Smith and Elmore 2004)",
    extra_refs=("smith-elmore-2004",),
)
def radial_divergence(obj, field=None, window=(750.0, 2500.0), **kwargs):
    """
    Radial divergence (s⁻¹) of the radial velocity by LLSD.

    Same parameters as :func:`llsd`.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        ``radial_divergence`` on the sweep's dimensions and coordinates (a
        DataTree of sweeps and the input root if a DataTree was given).

    See Also
    --------
    llsd

    Notes
    -----
    Method: the LLSD of Smith and Elmore (2004) [1]_, as used by Miller et al.
    (2013) [2]_ and Mahalik et al. (2019) [3]_; see :func:`llsd` for what is
    taken from them.
    {references}"""
    return _select(llsd(obj, field, window, **kwargs), "radial_divergence")


for _func in (azimuthal_shear, radial_divergence):
    _func.__doc__ = _func.__doc__.replace("{references}", _REFERENCES)


def _select(result, name):
    if isinstance(result, xr.Dataset):
        return result[name]
    return result.map_over_datasets(lambda ds: ds[[name]] if name in ds else ds)


# --------------------------------------------------------------------------
# layer maximum over the tilts of a volume (MRMS-style rotation product)
# --------------------------------------------------------------------------

_REFLECTIVITY = ("DBZH", "DBZ", "reflectivity")
_ROTATION_MODE = {"cyclonic": 0, "anticyclonic": 1, "both": 2}


def _tilt_angle(ds):
    """Fixed elevation angle of a sweep in degrees."""
    if "sweep_fixed_angle" in ds.variables:
        return float(np.asarray(ds["sweep_fixed_angle"]).ravel()[0])
    return float(np.nanmedian(np.asarray(ds["elevation"].values, dtype=float)))


def _volume_sweeps(obj):
    """``(name, Dataset)`` of the ``sweep*`` nodes of a volume, in order."""
    out = []
    for name, node in obj.children.items():
        if not name.startswith("sweep"):
            continue
        try:
            ds = node.to_dataset(inherit="all_coords")
        except (TypeError, ValueError):  # pragma: no cover - older xarray
            ds = node.to_dataset()
        out.append((name, ds))
    return out


def _beam_height_above_radar(ground_angle, elevation, radius):
    """Height of the beam centre above the radar at central angle ``ground_angle``."""
    e = np.deg2rad(elevation)
    return radius * (np.cos(e) / np.cos(e + ground_angle) - 1.0)


def _near_mask_numpy(flag, azimuth, rng, distance):
    """
    NumPy implementation of the compiled ``near_mask`` kernel.

    A gate is true if a flagged gate of the same sweep lies within
    ``distance`` metres in the plane of the sweep (the law of cosines on the
    slant ranges and the azimuth difference); the gate itself counts. Rays are
    visited by increasing azimuth offset and each ray contributes the interval
    of its gates inside the circle, counted with cumulative sums.
    """
    nray, ngate = flag.shape
    order = np.argsort(np.mod(azimuth, 360.0), kind="stable")
    ang = np.radians(np.mod(azimuth, 360.0))[order]
    counts = np.zeros((nray, ngate + 1), dtype=np.int64)
    counts[:, 1:] = np.cumsum(flag[order] != 0, axis=1)
    reach = np.where(
        rng > distance, np.arcsin(np.minimum(distance / rng, 1.0)), 2.0 * np.pi
    )
    hit = np.zeros((nray, ngate), dtype=bool)
    centre = np.arange(nray)
    for d in range(nray):
        smallest = np.inf
        for sign in (1,) if d == 0 else (1, -1):
            q = (centre + sign * d) % nray
            dt = _wrap(ang[q] - ang)
            near = float(np.abs(dt).min())
            smallest = min(smallest, near)
            cols = np.flatnonzero(reach >= near)
            if not cols.size:
                continue
            r0 = rng[cols][None, :]
            sn = r0 * np.sin(dt)[:, None]
            half = np.sqrt(np.maximum(distance**2 - sn**2, 0.0))
            mid = r0 * np.cos(dt)[:, None]
            lo = np.searchsorted(rng, mid - half, side="left")
            hi = np.searchsorted(rng, mid + half, side="right")
            c = counts[q]
            n = np.take_along_axis(c, hi, axis=1) - np.take_along_axis(c, lo, axis=1)
            hit[:, cols] |= (sn**2 <= distance**2) & (hi > lo) & (n > 0)
        if not (reach >= smallest).any():
            break
    out = np.zeros((nray, ngate), dtype=bool)
    out[order] = hit
    return out


def _near_mask(flag, azimuth, rng, distance, n_threads, compiled):
    """Boolean (ray, gate) array: a flagged gate lies within ``distance`` metres."""
    if compiled:
        return _shear.near_mask(
            flag.astype(np.float64), azimuth, rng, float(distance), int(n_threads or 0)
        ).astype(bool)
    return _near_mask_numpy(flag, azimuth, rng, float(distance))


def _nearest_ray_numpy(sorted_az, step, target):
    """Index into ``sorted_az`` of the nearest azimuth within 1.5 steps, else -1."""
    n = sorted_az.size
    target = np.mod(target, 360.0)
    hi = np.searchsorted(sorted_az, target, side="left")
    i0, i1 = (hi - 1) % n, hi % n

    def dist(i):
        d = np.abs(sorted_az[i] - target)
        return np.minimum(d, 360.0 - d)

    best = np.where(dist(i0) <= dist(i1), i0, i1)
    return np.where(dist(best) <= 1.5 * step, best, -1)


def _layer_max_numpy(tilts, elevations, phi, ref_azimuth, radius, bottom, top, mode):
    """
    NumPy implementation of the compiled ``layer_max`` kernel.

    ``tilts`` is a list of ``(shear, azimuth, range)`` per tilt. For every
    gate of the reference grid (azimuth ``ref_azimuth``, central angle
    ``phi`` of its ground range) a tilt contributes its value at the nearest
    ray (within 1.5 ray spacings) and nearest gate (within half a gate) when
    its beam centre is between ``bottom`` and ``top`` above the radar. ``mode``
    0 keeps the largest value, 1 the smallest and 2 the largest magnitude.
    """
    nray, ngate = ref_azimuth.size, phi.size
    best = np.full((nray, ngate), np.nan)
    source = np.full((nray, ngate), -1, dtype=np.int16)
    for t, ((shear, azimuth, rng), el) in enumerate(zip(tilts, elevations)):
        e = np.deg2rad(el)
        b = radius * np.cos(e) / np.cos(e + phi)
        height = b - radius
        slant = b * np.sin(phi) / np.cos(e)
        order = np.argsort(np.mod(azimuth, 360.0), kind="stable")
        sorted_az = np.mod(azimuth, 360.0)[order]
        step = np.median(np.diff(sorted_az)) if sorted_az.size > 1 else 360.0
        pos = _nearest_ray_numpy(sorted_az, step, ref_azimuth)
        ray = np.where(pos >= 0, order[np.maximum(pos, 0)], -1)
        # nearest gate of the tilt at the slant range of each reference gate
        hi = np.searchsorted(rng, slant, side="left")
        lo = np.maximum(hi - 1, 0)
        hi = np.minimum(hi, rng.size - 1)
        gate = np.where(slant - rng[lo] <= rng[hi] - slant, lo, hi)
        half = 0.5 * (rng[-1] - rng[0]) / (rng.size - 1) if rng.size > 1 else 0.0
        gate_ok = np.abs(rng[gate] - slant) <= half * (1.0 + 1e-9)
        use_col = (height >= bottom) & (height <= top) & gate_ok
        x = shear[np.maximum(ray, 0)[:, None], gate[None, :]]
        ok = (ray >= 0)[:, None] & use_col[None, :] & np.isfinite(x)
        if mode == 0:
            better = x > best
        elif mode == 1:
            better = x < best
        else:
            better = np.abs(x) > np.abs(best)
        take = ok & (np.isnan(best) | better)
        best = np.where(take, x, best)
        source = np.where(take, np.int16(t), source)
    return best.astype(np.float32), source


def _reflectivity_sweep(ds, sweeps, dbz_name, angle, tolerance=0.2):
    """The sweep itself if it holds ``dbz_name``, else the one nearest in elevation."""
    if dbz_name in ds.data_vars:
        return ds
    best = None
    for _, other in sweeps:
        if dbz_name not in other.data_vars:
            continue
        d = abs(_tilt_angle(other) - angle)
        if d <= tolerance and (best is None or d < best[0]):
            best = (d, other)
    return None if best is None else best[1]


def _storm_mask(ds, sweeps, dbz_name, angle, threshold, distance, n_threads, compiled):
    """
    Gates on the grid of ``ds`` with reflectivity of at least ``threshold``
    within ``distance`` metres. The reflectivity comes from the sweep itself
    or, if it holds none (split cuts), from the sweep at the same elevation
    angle (within 0.2 degrees), mapped to the grid by the nearest ray and gate.
    """
    from .dealias import _sweep_mapping

    src = _reflectivity_sweep(ds, sweeps, dbz_name, angle)
    if src is None:
        raise ValueError(
            f"min_dbz needs {dbz_name!r} in the sweep or in a sweep "
            "within 0.2 degrees of its elevation angle"
        )
    da = src[dbz_name]
    ray_dim = [d for d in da.dims if d != "range"][0]
    dbz = np.asarray(da.transpose(ray_dim, "range").values, dtype=np.float64)
    flag = np.isfinite(dbz) & (dbz >= threshold)
    near = _near_mask(
        flag,
        np.asarray(src["azimuth"].values, dtype=np.float64),
        np.asarray(src["range"].values, dtype=np.float64),
        distance,
        n_threads,
        compiled,
    )
    if src is ds:
        return near
    iray, igate = _sweep_mapping(src, ds)
    out = near[np.clip(iray, 0, None)][:, np.clip(igate, 0, None)]
    out[iray < 0] = False
    out[:, igate < 0] = False
    return out


@renamed_arguments(reflectivity="dbz", min_reflectivity="min_dbz")
@provenance(
    "Layer-maximum azimuthal shear over the tilts of a volume (MRMS-style)",
    extra_refs=("smith-elmore-2004",),
)
def azimuthal_shear_layer_max(
    obj,
    field=None,
    layer=(0.0, 2000.0),
    *,
    reference=None,
    shear=None,
    window=(750.0, 2500.0),
    weights="uniform",
    min_valid_fraction=0.5,
    median_filter=None,
    mask=None,
    exclude=None,
    nodata="auto",
    dbz=None,
    min_dbz=None,
    reflectivity_distance=5000.0,
    rotation="cyclonic",
    n_threads=None,
    engine="auto",
):
    """
    Layer-maximum azimuthal shear over the tilts of a volume.

    The maximum, at every position of a reference tilt, of the LLSD azimuthal
    shear of all tilts whose beam centre is inside a height layer at that
    position; the MRMS products are the layers 0 to 2 km (near-surface
    rotation) and 3 to 6 km (mid-level rotation) above ground level
    (Smith et al. 2016 [1]_).

    Parameters
    ----------
    obj : xarray.DataTree
        A volume with ``sweep_*`` nodes, with velocity dealiased (see
        :func:`radarx.retrieve.dealias_velocity`), or with an azimuthal shear
        field if ``shear`` is given.
    field : str, optional
        Dealiased radial velocity (m s⁻¹). Default: the first of ``VRADH``,
        ``VRAD``, ``VRADV``, ``VEL``, ``VELOCITY``, ``velocity`` that a sweep
        holds. Not needed if ``shear`` is given.
    layer : tuple of float, optional
        Bottom and top of the layer in metres above the radar. Default
        ``(0, 2000)``; use ``(3000, 6000)`` for the mid-level product. The
        MRMS layers are above ground level; the terrain under the beam is not
        known, so the heights are those of the beam centre above the radar
        (4/3 Earth model, see
        :func:`radarx.fundamentals.geometry.beam_height_at_ground_range`),
        which equal heights above ground over flat terrain.
    reference : str, optional
        Name of the sweep whose rays and gates the result is defined on.
        Default: the sweep with the lowest elevation that holds ``field``.
    shear : str, optional
        Name of an azimuthal shear field already in the sweeps (for example
        from ``.radarx.assign(azimuthal_shear(...))``). Default None: the
        shear is computed here with :func:`llsd`.
    window, weights, min_valid_fraction, median_filter, mask, exclude, nodata
        Options of :func:`llsd` for the shear computed here.
    dbz : str, optional
        Reflectivity field for ``min_dbz``. Default: the first of
        ``DBZH``, ``DBZ`` and ``reflectivity`` in the volume.
    min_dbz : float, optional
        If given (MRMS: 20 dBZ), the shear of a tilt is left out at gates with
        no reflectivity of at least this value within
        ``reflectivity_distance``, before the maximum is taken
        (Smith et al. 2016 [1]_). When a sweep holds no reflectivity (the
        split Doppler cuts of NEXRAD), that of the sweep within 0.2 degrees
        of its elevation angle is used. Default None: no reflectivity test.
    reflectivity_distance : float, optional
        Distance (m) in the plane of the sweep. Default 5000, the MRMS value.
    rotation : {"cyclonic", "anticyclonic", "both"}, optional
        ``"cyclonic"`` (default) takes the largest value (positive shear, the
        mesocyclones of the Northern Hemisphere), ``"anticyclonic"`` the
        smallest (most negative) and ``"both"`` the value of largest
        magnitude, with its sign.
    n_threads : int, optional
        Threads for the compiled kernels. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled
        kernel and falls back to NumPy; both give the same values.

    Returns
    -------
    xarray.Dataset
        On the grid and coordinates of the reference sweep:
        ``azimuthal_shear_layer_max`` in s⁻¹ (float32, NaN where no tilt of
        the layer has data) and ``layer_max_elevation``, the elevation angle
        of the tilt that gave the maximum (degrees, NaN where none).

    Notes
    -----
    For each gate of the reference sweep (azimuth :math:`\\theta`, ground
    range :math:`s`), every tilt with elevation :math:`\\alpha` has its beam
    centre at the height of the 4/3 Earth model at that ground range. If that
    height is in the layer, the value of the tilt at the nearest ray (within
    1.5 ray spacings) and nearest gate (within half a gate) at the same
    ground range is a candidate, and the layer maximum is the largest
    candidate. Repeated cuts of one elevation (SAILS) count as further
    tilts. The shear of each tilt is the LLSD of the tilt's own grid, so the
    velocity and the shear are not interpolated between tilts.

    What is taken from the literature, from Smith et al. (2016) [1]_ (read in
    full): the layers 0 to 2 km and 3 to 6 km, the maximum over the tilts "at
    each range", and the removal of shear that is not in or near a storm cell
    (reflectivity of 20 dBZ within 5 km) before the maximum. The MRMS
    products are made on a 3-D grid from many radars and updated every 2
    minutes; this function is the single-radar equivalent on the polar grid
    of one volume, and the choice of the reference grid, the nearest-gate
    sampling and the beam heights are radarx's. The MRMS pre-processing of
    velocity (the dealiasing of Jing and Wiener 1993, and the LLSD window,
    which Smith et al. do not give) is not reproduced; ``median_filter``
    (Smith and Elmore 2004 [2]_) and ``window`` are options. Miller et al.
    (2013, doi 10.1175/WAF-D-12-00065.1) is cited for the LLSD maximum within
    a layer, but the paper could not be read and nothing here is attributed
    to it.

    References
    ----------
    .. [1] Smith, T. M., V. Lakshmanan, G. J. Stumpf, K. L. Ortega, K. Hondl,
       K. Cooper, K. M. Calhoun, D. M. Kingfield, K. L. Manross, R. Toomey,
       and J. Brogden, 2016: Multi-Radar Multi-Sensor (MRMS) severe weather
       and aviation products: Initial operating capabilities. *Bull. Amer.
       Meteor. Soc.*, **97** (9), 1617-1630,
       https://doi.org/10.1175/BAMS-D-14-00173.1
    .. [2] Smith, T. M., and K. L. Elmore, 2004: The use of radial velocity
       derivative to diagnose rotation and divergence. *Preprints, 11th Conf.
       on Aviation, Range, and Aerospace Meteorology*, Hyannis, MA, Amer.
       Meteor. Soc., P5.6 (conference preprint, no DOI),
       https://ams.confex.com/ams/pdfpapers/81827.pdf

    Examples
    --------
    >>> low = radarx.retrieve.azimuthal_shear_layer_max(dtree)  # doctest: +SKIP
    >>> mid = dtree.radarx.azimuthal_shear_layer_max(layer=(3000.0, 6000.0))  # doctest: +SKIP
    """
    from ..fundamentals.constants import EFFECTIVE_RADIUS_4_3
    from ..fundamentals.geometry import ground_range

    if not isinstance(obj, xr.DataTree) or not any(
        n.startswith("sweep") for n in obj.children
    ):
        raise TypeError(
            "azimuthal_shear_layer_max needs a volume (xarray.DataTree with "
            "sweep nodes)"
        )
    if rotation not in _ROTATION_MODE:
        raise ValueError(
            f"rotation must be 'cyclonic', 'anticyclonic' or 'both', not {rotation!r}"
        )
    if len(layer) != 2 or not layer[0] < layer[1]:
        raise ValueError("layer must be (bottom, top) in metres with bottom < top")
    if reflectivity_distance <= 0:
        raise ValueError("reflectivity_distance must be positive")
    median_size = _median_size(median_filter)
    compiled = _use_compiled(engine)
    radius = EFFECTIVE_RADIUS_4_3
    name = shear if shear is not None else find_field(obj, VELOCITY, field)
    sweeps = _volume_sweeps(obj)
    dbz_name = None
    if min_dbz is not None:
        dbz_name = find_field(obj, _REFLECTIVITY, dbz)
    holders = [(n, ds) for n, ds in sweeps if name in ds.data_vars]
    if not holders:
        raise ValueError(f"No sweep contains {name!r}.")
    angles = {n: _tilt_angle(ds) for n, ds in holders}
    if reference is None:
        reference = min(angles, key=angles.get)
    elif reference not in angles:
        raise ValueError(f"reference {reference!r} is not a sweep that holds {name!r}")
    ref_ds = dict(holders)[reference]
    ref_az = np.asarray(ref_ds["azimuth"].values, dtype=np.float64)
    ref_rng = np.asarray(ref_ds["range"].values, dtype=np.float64)
    phi = ground_range(ref_rng, angles[reference], radius) / radius
    # tilts whose beam never reaches the layer within the range of the reference
    ends = np.array([phi.min(), phi.max()])
    use = []
    for n, ds in holders:
        h = _beam_height_above_radar(ends, angles[n], radius)
        if h[1] >= layer[0] and h[0] <= layer[1]:
            use.append((n, ds))
    arrays, elevations = [], []
    if use:
        if shear is None:
            results = _llsd_sweeps(
                [ds for _, ds in use],
                name,
                window,
                weights,
                min_valid_fraction,
                mask,
                exclude,
                nodata,
                obj.root.attrs,
                n_threads,
                engine,
                median_filter,
            )
            fields = [r["azimuthal_shear"] for r in results]
        else:
            fields = [ds[name] for _, ds in use]
        for (n, ds), da in zip(use, fields):
            ray_dim = [d for d in da.dims if d != "range"][0]
            values = np.ascontiguousarray(
                da.transpose(ray_dim, "range").values, dtype=np.float64
            )
            if min_dbz is not None:
                near = _storm_mask(
                    ds,
                    sweeps,
                    dbz_name,
                    angles[n],
                    min_dbz,
                    reflectivity_distance,
                    n_threads,
                    compiled,
                )
                values = np.where(near, values, np.nan)
            arrays.append(
                (
                    values,
                    np.ascontiguousarray(ds["azimuth"].values, dtype=np.float64),
                    np.ascontiguousarray(ds["range"].values, dtype=np.float64),
                )
            )
            elevations.append(angles[n])
    mode = _ROTATION_MODE[rotation]
    if not arrays:
        value = np.full((ref_az.size, ref_rng.size), np.nan, dtype=np.float32)
        source = np.full(value.shape, -1, dtype=np.int16)
    elif compiled:
        value, source = _shear.layer_max(
            [a[0] for a in arrays],
            [a[1] for a in arrays],
            [a[2] for a in arrays],
            elevations,
            phi,
            ref_az,
            radius,
            float(layer[0]),
            float(layer[1]),
            mode,
            int(n_threads or 0),
        )
    else:
        value, source = _layer_max_numpy(
            arrays,
            elevations,
            phi,
            ref_az,
            radius,
            float(layer[0]),
            float(layer[1]),
            mode,
        )
    elev = np.append(np.asarray(elevations, dtype=np.float64), np.nan)[
        source.astype(np.int64)
    ]
    da_ref = ref_ds[name]
    ray_dim = [d for d in da_ref.dims if d != "range"][0]
    template = da_ref.transpose(ray_dim, "range")
    attrs = {
        "method": "layer maximum of the LLSD azimuthal shear over the tilts of a volume",
        "layer_bottom_m": float(layer[0]),
        "layer_top_m": float(layer[1]),
        "reference_sweep": reference,
        "rotation": rotation,
        "tilts_used": len(arrays),
        "source_field": name,
    }
    if min_dbz is not None:
        attrs["min_reflectivity_dbz"] = float(min_dbz)
        attrs["reflectivity_distance_m"] = float(reflectivity_distance)
    if shear is None:
        attrs["window_range_m"] = float(window[0])
        attrs["window_azimuth_m"] = float(window[1])
        attrs["weights"] = weights
        if median_size:
            attrs["median_filter_size"] = median_size
    out = xr.Dataset(coords=template.coords)
    out["azimuthal_shear_layer_max"] = xr.DataArray(
        value,
        dims=template.dims,
        coords=template.coords,
        attrs={
            "long_name": "layer-maximum azimuthal shear of radial velocity",
            "units": "s-1",
            **attrs,
        },
    ).transpose(*da_ref.dims)
    out["layer_max_elevation"] = xr.DataArray(
        elev.astype(np.float32),
        dims=template.dims,
        coords=template.coords,
        attrs={
            "long_name": "elevation angle of the tilt that gave the layer maximum",
            "units": "degree",
        },
    ).transpose(*da_ref.dims)
    return out


accessor_for(azimuthal_shear_layer_max, "datatree")
