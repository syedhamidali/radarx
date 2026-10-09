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
about 2500 m wide) are those of the extended abstract of Smith and Elmore
(2004). Not taken from it: the exact solution of the
2 x 2 normal equations for windows that are not symmetric (Smith and Elmore
assume symmetric windows and weights, which makes the normal equations
diagonal), the Gaussian weights, the ``min_valid_fraction`` and the minimum of
three valid gates, which are radarx choices; Smith and Elmore also apply a
3 x 3 median filter first, which radarx does not. Miller et al. (2013) and
Mahalik et al. (2019) use the LLSD derivatives; nothing else in this module
is attributed to them.

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

__all__ = ["azimuthal_shear", "llsd", "radial_divergence"]

__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np
import xarray as xr

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
       Meteor. Soc., P5.6 (conference extended abstract, no DOI),
       https://ams.confex.com/ams/11aram22sls/techprogram/paper_81827.htm
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


def _llsd_numpy(data, azimuth, rng, window_range, window_azimuth, gaussian, min_frac):
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


def _sweep_arrays(ds, field, mask):
    """Velocity on (ray, range) and contiguous float64 arrays of one sweep."""
    if field not in ds:
        raise KeyError(f"{field!r} is not in the sweep")
    da = ds[field]
    if "range" not in da.dims or da.ndim != 2:
        raise ValueError(f"{field!r} must be 2-D with a 'range' dimension")
    ray_dim = da.dims[0] if da.dims[1] == "range" else da.dims[1]
    da_t = da.transpose(ray_dim, "range")
    if mask is not None:
        if isinstance(mask, str):
            mask = ds[mask]
        elif not isinstance(mask, xr.DataArray):
            mask = xr.DataArray(np.asarray(mask), dims=da.dims)
        da_t = da_t.where(~mask.astype(bool))
    data = np.ascontiguousarray(da_t.values, dtype=np.float64)
    azimuth = np.ascontiguousarray(ds["azimuth"].values, dtype=np.float64)
    rng = np.ascontiguousarray(ds["range"].values, dtype=np.float64)
    if azimuth.shape != (data.shape[0],):
        raise ValueError("the sweep needs one azimuth per ray")
    return da, da_t, (data, azimuth, rng)


def _llsd_sweeps(
    sweeps, field, window, weights, min_valid_fraction, mask, n_threads, engine
):
    """LLSD on a list of sweep Datasets in one kernel call; one Dataset each."""
    prepared = [_sweep_arrays(ds, field, mask) for ds in sweeps]
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
        )
    else:
        outs = [
            _llsd_numpy(*p[2], *options, float(min_valid_fraction)) for p in prepared
        ]

    method = {
        "method": "linear least-squares derivative (LLSD)",
        "source_field": field,
        "window_range_m": window_range,
        "window_azimuth_m": window_azimuth,
        "weights": weights,
        "min_valid_fraction": float(min_valid_fraction),
    }
    results = []
    for (da, da_t, _), out in zip(prepared, outs):
        result = xr.Dataset(coords=da_t.coords)
        for k, name in enumerate(("azimuthal_shear", "radial_divergence")):
            result[name] = xr.DataArray(
                out[k], dims=da_t.dims, coords=da_t.coords, attrs=_ATTRS[name] | method
            ).transpose(*da.dims)
        results.append(result)
    return results


def llsd(
    obj,
    field="VRADH",
    window=(750.0, 2500.0),
    *,
    weights="uniform",
    min_valid_fraction=0.5,
    mask=None,
    n_threads=None,
    engine="auto",
):
    """
    Azimuthal shear and radial divergence by linear least-squares derivatives.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataTree
        A PPI sweep with ``azimuth`` and ``range`` coordinates (e.g. from
        xradar), or a volume whose ``sweep_*`` nodes are processed one by one.
    field : str, optional
        Radial velocity field in m s⁻¹. Default ``"VRADH"``. Velocities must be
        dealiased (or free of aliasing); folds produce spurious shear.
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
        Gates to leave out of the fit (``True`` = excluded), e.g. a
        reflectivity or quality mask, or the name of such a variable.
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
        "n_threads": n_threads,
        "engine": engine,
    }
    if isinstance(obj, xr.Dataset):
        return _llsd_sweeps([obj], field, **options)[0]
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
        results = _llsd_sweeps(sweeps, field, **options)
        return product_tree(obj, dict(zip(names, results)))
    raise TypeError("llsd needs an xarray.Dataset sweep or an xarray.DataTree")


llsd.__doc__ = llsd.__doc__.replace("{references}", _REFERENCES)


def azimuthal_shear(obj, field="VRADH", window=(750.0, 2500.0), **kwargs):
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


def radial_divergence(obj, field="VRADH", window=(750.0, 2500.0), **kwargs):
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
