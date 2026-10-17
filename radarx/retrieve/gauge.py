#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx Radar-Gauge Merging
==========================

Rain gauges measure the rainfall at a point accurately, a radar measures it
everywhere with errors that vary in space (calibration, the relation between
reflectivity and rain rate, range, beam height, blockage). The functions
here combine the two on the radar grid with the classical methods compared by
Goudenhoofdt and Delobbe (2009):

``"mfb"``
    Mean field bias: the radar field is multiplied by the factor
    :math:`\\sum G_i / \\sum R_i` of the gauge-radar pairs (Eq. 1 of the
    paper, Smith and Krajewski 1991).
``"rda"``
    Range-dependent adjustment: :math:`\\ln(G/R)` of the pairs is fitted by a
    second-order polynomial of the distance to the radar and the radar field is
    multiplied by the exponential of the fit (Eq. 2 of the paper).
``"brandes"``
    Brandes (1975) spatial adjustment: the factors :math:`G_i/R_i` of the
    pairs are interpolated with the weights :math:`\\exp(-d_i^2/k)` and
    :math:`k = (2\\delta)^{-1}`, with the gauge density :math:`\\delta`
    (Eqs. 3-4 of the paper).
``"ok"``
    Ordinary kriging of the gauges alone (the added value of the radar is
    the difference to the methods that use it).
``"conditional"``
    Conditional merging of Sinclair and Pegram (2005), "kriging with
    radar-based error correction" in Goudenhoofdt and Delobbe (2009): the
    radar field minus the kriging of the radar values at the gauge sites,
    added to the kriging of the gauges. The merged field equals the gauge at
    the gauge sites and has the spatial structure of the radar.
``"ked"``
    Kriging with an external drift (Wackernagel 2003, ch. 22): the mean of
    the rain field is a linear function of the radar field, estimated
    together with the interpolation. This was the best method of
    Goudenhoofdt and Delobbe (2009) for every month, distance and
    network density.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}

Implementation
--------------
The empirical variogram, the kriging systems and the Brandes weights run
in a compiled, multithreaded C++ kernel; an equivalent NumPy implementation
is the fallback and the test oracle. The kriging systems are solved by LU
decomposition: with all gauges as neighbours the matrix is factorized once
and every grid cell needs one back substitution, and the weights are
shared by all times when the variogram is the same; with a limited number of
neighbours each cell solves its own small system. All times (and any other
leading dimension) of a field are processed in one call.

What is radarx's choice and not in the papers
---------------------------------------------
* Variogram. The model variogram is fitted automatically by the weighted
  least squares of Cressie (1985), minimizing :math:`\\sum_j N_j
  (\\hat\\gamma_j/\\gamma(h_j)-1)^2` over the lags :math:`j` of the empirical
  variogram (Matheron 1963), with ``"linear"`` (the model of Goudenhoofdt and
  Delobbe 2009), ``"spherical"``, ``"exponential"`` (default) or ``"gaussian"``
  models, each with a nugget. The parameters are fitted for every time. For
  ``"ked"`` the variogram is that of the residuals of the regression of the
  gauges on the radar values at the gauges, which is common practice
  (Goovaerts 2000); for ``"conditional"`` and ``"ok"`` it is that of the gauge
  values (or of the radar values at the gauges when the gauges are constant).
  The practical range of the exponential and gaussian models is the
  distance at which the model reaches 95 % of the sill.
* The radar value at a gauge is the mean of the radar pixels within
  ``radius`` of the gauge, by default within 1.5 pixel spacings (3 x 3
  pixels on a regular grid, as in Goudenhoofdt and Delobbe 2009).
* The kriging systems use the 20 nearest gauges by default (as the paper);
  ``n_neighbours=None`` uses all.
* If the drift is constant among the gauges of a neighbourhood (for example
  no rain in the radar field) the drift system is singular, and ordinary
  kriging is used there instead. The number of such cells is stored in the
  attributes of the result.
* Negative merged values are set to zero (``clip``).
"""

from __future__ import annotations

__all__ = ["radar_at_gauges", "gauge_variogram", "gauge_adjust"]

__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

import numpy as np
import xarray as xr

from .._provenance import provenance
from .._registry import accessor_method
from .._sweepwise import dask_support

try:
    from . import _gauge

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _gauge = None
    HAS_COMPILED_KERNEL = False

METHODS = ("mfb", "rda", "brandes", "ok", "conditional", "ked")
MODELS = ("linear", "spherical", "exponential", "gaussian")
_MODEL_CODE = {name: i for i, name in enumerate(MODELS)}
_GEOGRAPHIC = {"lon", "lat", "longitude", "latitude"}
_COORDS = (("x", "y"), ("lon", "lat"), ("longitude", "latitude"))
_DRIFT_TOLERANCE = 1e-9
_CHUNK = 2048

# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------


def _use_compiled(engine):
    """Whether to run the compiled kernel for the requested ``engine``."""
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled gauge kernel is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


# --------------------------------------------------------------------------
# variogram models (NumPy)
# --------------------------------------------------------------------------


def _semivariance(model, h, nugget, psill, rng):
    """Model variogram at distances ``h`` (0 at h = 0, as in the kriging systems)."""
    h = np.asarray(h, dtype=np.float64)
    if model == "linear":
        g = nugget + psill * h
    elif model == "spherical":
        x = np.minimum(h / rng, 1.0)
        g = nugget + psill * (1.5 * x - 0.5 * x**3)
    elif model == "exponential":
        g = nugget + psill * (1.0 - np.exp(-3.0 * h / rng))
    else:
        g = nugget + psill * (1.0 - np.exp(-3.0 * (h / rng) ** 2))
    return np.where(h > 0.0, g, 0.0)


def _shape(model, h, rng):
    """Variogram shape per unit partial sill."""
    return _semivariance(model, h, 0.0, 1.0, rng)


def _nnls2(basis, target, weight):
    """Non-negative least squares of ``target ~ c0 + c1 basis`` with weights."""
    from scipy.optimize import nnls

    sw = np.sqrt(weight)
    a = np.stack([np.ones_like(basis), basis], axis=1) * sw[:, None]
    coef, _ = nnls(a, target * sw)
    return coef


def _cressie_cost(model, h, g, n, nugget, psill, rng):
    m = _semivariance(model, h, nugget, psill, rng)
    if np.any(m <= 0.0):
        return np.inf
    return float(np.sum(n * (g / m - 1.0) ** 2))


def _fit_variogram(model, lag, gamma, count):
    """
    Nugget, partial sill and range of ``model`` fitted to an empirical
    variogram by the weighted least squares of Cressie (1985).

    The model is linear in the nugget and the partial sill: for each range
    of a coarse grid they come from a non-negative least squares with the
    weights :math:`N_j / \\hat\\gamma_j^2` (the first-order form of Cressie's
    criterion), the best range starts a bounded least-squares fit of the
    criterion itself. The linear model is fitted by the same weights,
    iterated.
    """
    from scipy.optimize import least_squares

    ok = (count > 0) & np.isfinite(gamma) & (lag > 0)
    h, g, n = lag[ok], gamma[ok], count[ok].astype(np.float64)
    if h.size < 3 or not np.any(g > 0):
        top = float(np.nanmax(gamma)) if np.any(np.isfinite(gamma)) else 1.0
        top = top if top > 0 else 1.0
        span = float(np.nanmax(lag)) if lag.size else 1.0
        return 0.0, top, (0.0 if model == "linear" else span / 3.0)
    gmax = float(g.max())
    hmax = float(h.max())
    floor = 1e-6 * gmax
    w0 = n / np.maximum(g, floor) ** 2
    if model == "linear":
        w = w0
        c = np.array([0.0, gmax / hmax])
        for _ in range(4):
            c = _nnls2(h, g, w)
            m = np.maximum(c[0] + c[1] * h, floor)
            w = n / m**2
        return float(c[0]), float(max(c[1], 1e-12 * gmax / hmax)), 0.0
    best = None
    for rng in np.geomspace(0.1 * hmax, 3.0 * hmax, 14):
        c = _nnls2(_shape(model, h, rng), g, w0)
        nug, ps = float(c[0]), float(max(c[1], floor))
        cost = _cressie_cost(model, h, g, n, nug, ps, rng)
        if best is None or cost < best[0]:
            best = (cost, nug, ps, rng)
    _, nug, ps, rng = best

    def resid(p):
        m = _semivariance(model, h, p[0], p[1], p[2])
        return np.sqrt(n) * (g / np.maximum(m, 1e-300) - 1.0)

    lo = [0.0, floor, 0.05 * hmax]
    hi = [2.0 * gmax, 10.0 * gmax, 10.0 * hmax]
    x0 = np.clip([nug, ps, rng], lo, hi)
    try:
        sol = least_squares(resid, x0, bounds=(lo, hi), max_nfev=100)
        cost = _cressie_cost(model, h, g, n, *sol.x)
        if cost <= _cressie_cost(model, h, g, n, *x0):
            return float(sol.x[0]), float(sol.x[1]), float(sol.x[2])
    except (ValueError, FloatingPointError):
        pass
    return float(x0[0]), float(x0[1]), float(x0[2])


# --------------------------------------------------------------------------
# geometry and the xarray layer
# --------------------------------------------------------------------------


def _coord_names(obj, x, y, what):
    if (x is None) != (y is None):
        raise ValueError("pass both x and y, or neither")
    if x is not None:
        for name in (x, y):
            if name not in obj.coords:
                raise KeyError(f"{what} has no coordinate {name!r}")
        return x, y
    for nx, ny in _COORDS:
        if nx in obj.coords and ny in obj.coords:
            return nx, ny
    raise KeyError(
        f"{what} needs coordinates x and y (m) or lon and lat (degrees); "
        f"it has {sorted(map(str, obj.coords))}"
    )


def _is_geographic(obj, name):
    units = str(obj.coords[name].attrs.get("units", ""))
    return name in _GEOGRAPHIC or units.startswith("degree")


def _site_of(radar, site):
    if site is not None:
        lat, lon = (float(v) for v in site)
        return lat, lon
    for source in (radar.coords, radar.attrs):
        if "latitude" in source and "longitude" in source:
            try:
                return float(np.asarray(source["latitude"])), float(
                    np.asarray(source["longitude"])
                )
            except (TypeError, ValueError):  # pragma: no cover
                continue
    return None


def _project(lat0, lon0):
    import pyproj

    return pyproj.Transformer.from_crs(
        "EPSG:4326",
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs",
        always_xy=True,
    )


class _Setup:
    """The radar field and the gauges as NumPy arrays in one metric frame."""


def _prepare(radar, gauges, x, y, site):
    if not isinstance(radar, xr.DataArray):
        raise TypeError("radar must be an xarray.DataArray")
    if not isinstance(gauges, xr.DataArray):
        raise TypeError("gauges must be an xarray.DataArray")
    rxn, ryn = _coord_names(radar, x, y, "radar")
    if x is not None and x in gauges.coords and y in gauges.coords:
        gxn, gyn = x, y
    else:
        # the gauges use the same kind of coordinates as the radar when possible
        order = [(rxn, ryn)] + [p for p in _COORDS if p != (rxn, ryn)]
        for nx, ny in order:
            if nx in gauges.coords and ny in gauges.coords:
                gxn, gyn = nx, ny
                break
        else:
            raise KeyError(
                "gauges need coordinates x and y (m) or lon and lat (degrees); "
                f"they have {sorted(map(str, gauges.coords))}"
            )
    cx, cy = radar.coords[rxn], radar.coords[ryn]
    gx, gy = gauges.coords[gxn], gauges.coords[gyn]
    if gx.ndim != 1 or gy.ndim != 1 or gx.dims != gy.dims:
        raise ValueError(
            "the gauge coordinates must be 1-D along one station dimension"
        )
    station = gx.dims[0]
    spatial = [d for d in radar.dims if d in set(cx.dims) | set(cy.dims)]
    extra = [d for d in radar.dims if d not in spatial]
    if set(gauges.dims) - {station} != set(extra):
        raise ValueError(
            f"the gauges have the dimensions {list(gauges.dims)}; besides the "
            f"station dimension {station!r} they must have the non-spatial "
            f"dimensions of the radar field {extra}"
        )
    for d in extra:
        if radar.sizes[d] != gauges.sizes[d]:
            raise ValueError(f"dimension {d!r} has different sizes in radar and gauges")
        if d in radar.coords and d in gauges.coords:
            if not np.array_equal(radar[d].values, gauges[d].values):
                raise ValueError(
                    f"the coordinate {d!r} differs between radar and gauges"
                )
    r_geo, g_geo = _is_geographic(radar, rxn), _is_geographic(gauges, gxn)
    base = radar.isel({d: 0 for d in extra}, drop=True) if extra else radar
    pxr = np.asarray(
        cx.broadcast_like(base).transpose(*spatial).values, dtype=np.float64
    ).reshape(-1)
    pyr = np.asarray(
        cy.broadcast_like(base).transpose(*spatial).values, dtype=np.float64
    ).reshape(-1)
    gxv = np.asarray(gx.values, dtype=np.float64)
    gyv = np.asarray(gy.values, dtype=np.float64)
    where = _site_of(radar, site)
    if r_geo and not g_geo:
        raise ValueError("the radar has geographic coordinates, the gauges metric ones")
    if g_geo and not r_geo:
        if where is None:
            raise ValueError(
                "the gauges are given by longitude and latitude and the radar "
                "by x and y: pass site=(latitude, longitude) of the radar"
            )
        tr = _project(*where)
        gxv, gyv = tr.transform(gxv, gyv)
        origin = (0.0, 0.0)
    elif r_geo and g_geo:
        centre = where or (float(np.nanmean(pyr)), float(np.nanmean(pxr)))
        tr = _project(*centre)
        pxr, pyr = tr.transform(pxr, pyr)
        gxv, gyv = tr.transform(gxv, gyv)
        origin = (0.0, 0.0) if where is not None else None
    else:
        origin = (0.0, 0.0)
    s = _Setup()
    s.radar = radar
    s.spatial = spatial
    s.extra = extra
    s.station = station
    s.xr, s.yr = np.asarray(pxr), np.asarray(pyr)
    s.xg, s.yg = np.asarray(gxv), np.asarray(gyv)
    s.origin = origin
    t = radar.transpose(*extra, *spatial)
    s.template = t
    s.nt = int(np.prod([radar.sizes[d] for d in extra], dtype=np.int64)) if extra else 1
    s.npix = int(np.prod([radar.sizes[d] for d in spatial], dtype=np.int64))
    s.R = np.asarray(t.values, dtype=np.float64).reshape(s.nt, s.npix)
    g = gauges.transpose(*extra, station)
    s.G = np.asarray(g.values, dtype=np.float64).reshape(s.nt, -1)
    s.ng = s.G.shape[1]
    s.extra_shape = tuple(radar.sizes[d] for d in extra)
    return s


def _extra_array(s, values, name, attrs=None):
    arr = np.asarray(values).reshape(s.extra_shape)
    coords = {d: s.radar.coords[d] for d in s.extra if d in s.radar.coords}
    return xr.DataArray(arr, dims=s.extra, coords=coords, name=name, attrs=attrs or {})


def _field(s, values, name, attrs=None):
    arr = np.asarray(values).reshape(s.template.shape)
    out = xr.DataArray(
        arr,
        dims=s.template.dims,
        coords=s.template.coords,
        name=name,
        attrs=attrs or {},
    )
    return out.transpose(*s.radar.dims)


# --------------------------------------------------------------------------
# sampling the radar at the gauges
# --------------------------------------------------------------------------


def _spacing(xp, yp):
    """Typical distance between neighbouring pixels."""
    from scipy.spatial import cKDTree

    ok = np.isfinite(xp) & np.isfinite(yp)
    pts = np.stack([xp[ok], yp[ok]], axis=1)
    if len(pts) < 2:
        return 1.0
    tree = cKDTree(pts)
    sample = pts[:: max(1, len(pts) // 2000)]
    d, _ = tree.query(sample, k=2)
    d = d[:, 1]
    d = d[d > 0]
    return float(np.median(d)) if d.size else 1.0


def _sample(s, radius):
    """Radar values at the gauges, the mean of the pixels within ``radius``."""
    from scipy.spatial import cKDTree

    ok = np.isfinite(s.xr) & np.isfinite(s.yr)
    pos = np.flatnonzero(ok)
    if radius is None:
        radius = 1.5 * _spacing(s.xr, s.yr)
    if not radius > 0:
        raise ValueError("radius must be positive")
    tree = cKDTree(np.stack([s.xr[ok], s.yr[ok]], axis=1))
    near = tree.query_ball_point(np.stack([s.xg, s.yg], axis=1), r=float(radius))
    width = max((len(v) for v in near), default=0)
    index = np.full((s.ng, max(width, 1)), -1, dtype=np.int64)
    for i, v in enumerate(near):
        index[i, : len(v)] = pos[v]
    out = np.full((s.nt, s.ng), np.nan)
    valid = index >= 0
    safe = np.where(valid, index, 0)
    vals = s.R[:, safe]  # nt, ng, width
    vals = np.where(valid[None], vals, np.nan)
    with np.errstate(all="ignore"):
        finite = np.isfinite(vals)
        n = finite.sum(axis=2)
        total = np.where(finite, vals, 0.0).sum(axis=2)
        out = np.where(n > 0, total / np.maximum(n, 1), np.nan)
    return out


@dask_support("computes", "it samples the whole field at the gauges.")
@provenance("Radar values at rain gauge sites")
def radar_at_gauges(
    obj,
    gauges,
    *,
    radius=None,
    x=None,
    y=None,
    site=None,
):
    """
    Radar values at the sites of the gauges.

    The value at a gauge is the mean of the finite radar pixels within
    ``radius`` of it. The mean over a few pixels limits the effect of wind
    drift and of the small displacement between the radar beam and the gauge
    (Goudenhoofdt and Delobbe 2009 use 3 x 3 pixels of 600 m).

    Parameters
    ----------
    obj : xarray.DataArray
        A rain field on a grid (``y``, ``x``) or on any 2-D layout with 2-D
        coordinates (for example the gates of a sweep with ``x`` and ``y``),
        optionally with leading dimensions such as ``time``.
    gauges : xarray.DataArray
        Gauge values on a station dimension with the coordinates of the
        stations and the same leading dimensions as ``obj``. Only the
        coordinates are used here.
    radius : float, optional
        Radius (m) of the averaging disc. Default 1.5 times the pixel spacing
        (3 x 3 pixels of a regular grid). A site with no finite pixel within
        the radius gets NaN.
    x, y : str, optional
        Names of the coordinates of the positions: ``x`` and ``y`` in metres
        east and north of the radar, or ``lon`` and ``lat`` in degrees.
        Default: ``x``/``y`` if present, otherwise ``lon``/``lat``, then
        ``longitude``/``latitude``.
    site : (float, float), optional
        Latitude and longitude of the radar, needed when the gauges are given
        by longitude and latitude and the radar field by ``x`` and ``y`` and
        the radar does not carry ``latitude`` and ``longitude`` coordinates
        or attributes.

    Returns
    -------
    xarray.DataArray
        The radar values on the dimensions of ``gauges``, in the units of
        ``radar``.

    Raises
    ------
    TypeError
        If ``radar`` or ``gauges`` is not a DataArray.
    KeyError
        If the positions cannot be found.
    ValueError
        If the dimensions of radar and gauges do not match, ``radius`` is not
        positive, or the coordinate systems cannot be matched.

    References
    ----------
    Goudenhoofdt, E., and L. Delobbe, 2009: Evaluation of radar-gauge merging
    methods for quantitative precipitation estimates. *Hydrol. Earth Syst.
    Sci.*, **13** (2), 195-203, https://doi.org/10.5194/hess-13-195-2009

    Examples
    --------
    >>> at_gauges = radar_at_gauges(rain, gauges)  # doctest: +SKIP
    """
    radar = obj
    s = _prepare(radar, gauges, x, y, site)
    values = _sample(s, radius)
    g = gauges.transpose(*s.extra, s.station)
    out = xr.DataArray(
        values.reshape(g.shape),
        dims=g.dims,
        coords=g.coords,
        name=radar.name,
        attrs={k: v for k, v in radar.attrs.items() if k in ("units", "long_name")},
    )
    return out


# --------------------------------------------------------------------------
# empirical variogram and kriging (compiled kernel or NumPy)
# --------------------------------------------------------------------------


def _variogram_arrays(x, y, values, n_lags, max_lag, use_compiled, n_threads):
    """Empirical variogram of ``values`` (n, ng): mean distance, semivariance, pairs."""
    if use_compiled:
        return _gauge.variogram(
            np.ascontiguousarray(x),
            np.ascontiguousarray(y),
            np.ascontiguousarray(values),
            int(n_lags),
            float(max_lag),
            int(n_threads or 0),
        )
    n, ng = values.shape
    i, j = np.triu_indices(ng, 1)
    h = np.hypot(x[i] - x[j], y[i] - y[j])
    keep = h < max_lag
    i, j, h = i[keep], j[keep], h[keep]
    width = max_lag / n_lags
    b = np.minimum((h / width).astype(np.int64), n_lags - 1)
    count = np.bincount(b, minlength=n_lags).astype(np.int64)
    dist = np.bincount(b, weights=h, minlength=n_lags)
    lag = np.where(
        count > 0, dist / np.maximum(count, 1), (np.arange(n_lags) + 0.5) * width
    )
    gamma = np.full((n, n_lags), np.nan)
    for t in range(n):
        sq = (values[t, i] - values[t, j]) ** 2
        sums = np.bincount(b, weights=sq, minlength=n_lags)
        gamma[t] = np.where(count > 0, sums / (2.0 * np.maximum(count, 1)), np.nan)
    return lag, gamma, count


def _drift_degenerate(f):
    """No usable drift: the values are constant to the tolerance."""
    top = float(np.max(np.abs(f))) if f.size else 0.0
    return float(np.ptp(f)) <= _DRIFT_TOLERANCE * top


def _krige_numpy(
    xg, yg, vals, drift_g, xt, yt, drift_t, model, params, group, n_neighbours
):
    """NumPy implementation of the kernel ``_gauge.krige`` (same results)."""
    name = MODELS[model]
    nv, ng = vals.shape
    npix = xt.size
    drift = drift_g.size > 0
    est = np.full((nv, npix), np.nan)
    var = np.full((nv, npix), np.nan)
    fallback = np.zeros(nv, dtype=np.int64)
    local = 0 < n_neighbours < ng
    members = {}
    for t in range(nv):
        members.setdefault(int(group[t]), []).append(t)
    if drift and any(len(m) != 1 for m in members.values()):
        raise ValueError("with a drift every value set needs its own group")
    dgg = np.hypot(xg[:, None] - xg[None, :], yg[:, None] - yg[None, :])
    for mem in members.values():
        first = mem[0]
        p = params[first]
        # the solution does not change if the variogram is divided by a constant
        # (the multipliers and the variance scale with it)
        scale = float(_semivariance(name, dgg, p[0], p[1], p[2]).max())
        if not scale > 0.0:
            raise RuntimeError("the kriging system of the gauges is singular")

        def gam(d):
            return _semivariance(name, d, p[0] / scale, p[1] / scale, p[2])

        ft = drift_t[first] if drift else None
        fg = drift_g[first] if drift else None
        if not local:
            use_drift = drift and not _drift_degenerate(fg)
            m = ng
            n = m + 1 + (1 if use_drift else 0)
            a = np.zeros((n, n))
            a[:m, :m] = gam(dgg)
            a[:m, m] = a[m, :m] = 1.0
            if use_drift:
                f_mean, f_sd = fg.mean(), fg.std()
                a[:m, m + 1] = a[m + 1, :m] = (fg - f_mean) / f_sd
            for start in range(0, npix, _CHUNK):
                sl = slice(start, min(npix, start + _CHUNK))
                d0 = np.hypot(
                    xg[:, None] - xt[sl][None, :], yg[:, None] - yt[sl][None, :]
                )
                b = np.ones((n, d0.shape[1]))
                b[:m] = gam(d0)
                if use_drift:
                    b[m + 1] = (ft[sl] - f_mean) / f_sd
                sol = np.linalg.solve(a, b)
                s2 = np.maximum(np.sum(sol * b, axis=0), 0.0) * scale
                for t in mem:
                    est[t, sl] = vals[t] @ sol[:m]
                    var[t, sl] = s2
            if drift:
                bad = np.isnan(ft)
                est[first, bad] = np.nan
                var[first, bad] = np.nan
                if not use_drift:
                    fallback[first] += int((~bad).sum())
            continue
        m = int(n_neighbours)
        for start in range(0, npix, _CHUNK):
            sl = slice(start, min(npix, start + _CHUNK))
            d2 = (xg[None, :] - xt[sl][:, None]) ** 2 + (
                yg[None, :] - yt[sl][:, None]
            ) ** 2
            idx = np.argsort(d2, axis=1, kind="stable")[:, :m]  # targets, m
            dmat = dgg[idx[:, :, None], idx[:, None, :]]
            d0 = np.sqrt(np.take_along_axis(d2, idx, axis=1))
            if drift:
                fvals = fg[idx]
                top = np.max(np.abs(fvals), axis=1)
                good = np.ptp(fvals, axis=1) > _DRIFT_TOLERANCE * top
                nan_t = np.isnan(ft[sl])
                f_mean = fvals.mean(axis=1)
                f_sd = np.where(good, fvals.std(axis=1), 1.0)
                fnorm = (fvals - f_mean[:, None]) / f_sd[:, None]
                f0n = (ft[sl] - f_mean) / f_sd
            else:
                good = np.zeros(len(idx), dtype=bool)
                nan_t = np.zeros(len(idx), dtype=bool)
            for flag in (False, True):
                sel = (good if flag else ~good) & ~nan_t
                k = int(sel.sum())
                if k == 0:
                    continue
                n = m + 1 + (1 if flag else 0)
                a = np.zeros((k, n, n))
                a[:, :m, :m] = gam(dmat[sel])
                a[:, :m, m] = a[:, m, :m] = 1.0
                b = np.ones((k, n))
                b[:, :m] = gam(d0[sel])
                if flag:
                    a[:, :m, m + 1] = a[:, m + 1, :m] = fnorm[sel]
                    b[:, m + 1] = f0n[sel]
                sol = np.linalg.solve(a, b[:, :, None])[:, :, 0]
                s2 = np.maximum(np.sum(sol * b, axis=1), 0.0) * scale
                pos = np.flatnonzero(sel) + start
                for t in mem:
                    est[t, pos] = np.sum(sol[:, :m] * vals[t][idx[sel]], axis=1)
                    var[t, pos] = s2
                if drift and not flag:
                    fallback[first] += k
    return est, var, fallback


def _krige(
    xg,
    yg,
    vals,
    drift_g,
    xt,
    yt,
    drift_t,
    model,
    params,
    group,
    n_neighbours,
    use_compiled,
    n_threads,
):
    """Kriging of ``vals`` (n, ng) at the targets: estimate, variance, fallbacks."""
    if use_compiled:
        return _gauge.krige(
            np.ascontiguousarray(xg),
            np.ascontiguousarray(yg),
            np.ascontiguousarray(vals),
            np.ascontiguousarray(drift_g),
            np.ascontiguousarray(xt),
            np.ascontiguousarray(yt),
            np.ascontiguousarray(drift_t),
            int(model),
            np.ascontiguousarray(params),
            np.ascontiguousarray(group, dtype=np.int32),
            int(n_neighbours or 0),
            int(n_threads or 0),
        )
    try:
        return _krige_numpy(
            xg,
            yg,
            vals,
            drift_g,
            xt,
            yt,
            drift_t,
            model,
            params,
            group,
            n_neighbours or 0,
        )
    except np.linalg.LinAlgError as err:
        raise RuntimeError("the kriging system of the gauges is singular") from err


def _brandes_numpy(xg, yg, ratio, xt, yt, k):
    nv, ng = ratio.shape
    out = np.full((nv, xt.size), np.nan)
    for start in range(0, xt.size, _CHUNK):
        sl = slice(start, min(xt.size, start + _CHUNK))
        d2 = (xg[None, :] - xt[sl][:, None]) ** 2 + (yg[None, :] - yt[sl][:, None]) ** 2
        for t in range(nv):
            r = ratio[t]
            good = np.isfinite(r)
            if not good.any():
                continue
            dmin = d2[:, good].min(axis=1, keepdims=True)
            w = np.where(
                good[None], np.exp(-np.where(good[None], d2 - dmin, 0.0) / k), 0.0
            )
            out[t, sl] = (w * np.where(good, r, 0.0)[None]).sum(axis=1) / w.sum(axis=1)
    return out


def _mfb_numpy(radar, gauge, min_value):
    pair = (radar > min_value) & (gauge > min_value)
    n = pair.sum(axis=1)
    sr = np.where(pair, radar, 0.0).sum(axis=1)
    sg = np.where(pair, gauge, 0.0).sum(axis=1)
    with np.errstate(all="ignore"):
        factor = np.where(n > 0, sg / sr, np.nan)
    return factor, n.astype(np.int64)


# --------------------------------------------------------------------------
# variogram parameters of the rows
# --------------------------------------------------------------------------


def _row_groups(mask):
    """Rows of ``mask`` that share the same valid gauges: ``[(rows, columns)]``."""
    uniq, inverse = np.unique(mask, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    return [
        (np.flatnonzero(inverse == i), np.flatnonzero(u)) for i, u in enumerate(uniq)
    ]


def _default_lags(ng, n_lags):
    if n_lags is not None:
        if int(n_lags) < 3:
            raise ValueError("n_lags must be at least 3")
        return int(n_lags)
    pairs = ng * (ng - 1) // 2
    return int(min(15, max(5, pairs // 8)))


def _max_lag(x, y, max_lag):
    if max_lag is not None:
        if not max_lag > 0:
            raise ValueError("max_lag must be positive")
        return float(max_lag)
    i, j = np.triu_indices(len(x), 1)
    top = float(np.hypot(x[i] - x[j], y[i] - y[j]).max())
    return 0.5 * top


def _fit_rows(model, x, y, values, n_lags, max_lag, use_compiled, n_threads):
    """Fit the variogram of every row of ``values`` (n, ng); returns (n, 3)."""
    n, ng = values.shape
    nl = _default_lags(ng, n_lags)
    ml = _max_lag(x, y, max_lag)
    lag, gamma, count = _variogram_arrays(x, y, values, nl, ml, use_compiled, n_threads)
    out = np.zeros((n, 3))
    for t in range(n):
        out[t] = _fit_variogram(model, lag, gamma[t], count)
    return out, lag, gamma, count


def _fixed_variogram(variogram, model, s_extra_shape, nt):
    """Parameters of a given variogram as ``(model, (nt, 3))``."""
    if isinstance(variogram, xr.Dataset):
        name = variogram.attrs.get("model", model)
        names = {"nugget": "nugget", "psill": "partial_sill", "range": "range"}
        for key in names.values():
            if key not in variogram:
                raise KeyError(f"the variogram Dataset has no variable {key!r}")
        arrays = [
            np.asarray(variogram[names[k]].values, dtype=np.float64) for k in names
        ]
        arrays = [
            np.broadcast_to(a, s_extra_shape if s_extra_shape else ()).reshape(-1)
            for a in arrays
        ]
        params = np.stack([np.broadcast_to(a, (nt,)) for a in arrays], axis=1)
    elif isinstance(variogram, dict):
        name = variogram.get("model", model)
        key = "slope" if name == "linear" and "slope" in variogram else "partial_sill"
        try:
            vals = [
                float(variogram["nugget"]),
                float(variogram[key]),
                float(variogram.get("range", 0.0)),
            ]
        except KeyError as err:
            raise KeyError(
                "a variogram dict needs 'nugget' and 'partial_sill' "
                "(or 'slope' for the linear model) and 'range'"
            ) from err
        params = np.tile(np.array(vals), (nt, 1))
    else:
        raise TypeError(
            "variogram must be None, a dict or the Dataset of gauge_variogram"
        )
    if name not in MODELS:
        raise ValueError(f"the variogram model must be one of {MODELS}, not {name!r}")
    if np.any(params[:, 0] < 0) or np.any(params[:, 1] < 0):
        raise ValueError("the nugget and the partial sill must not be negative")
    if name != "linear" and np.any(params[:, 2] <= 0):
        raise ValueError("the range of the variogram must be positive")
    return name, params


def _group_ids(params):
    """Value sets with the same parameters share one group."""
    _, inverse = np.unique(params, axis=0, return_inverse=True)
    return np.asarray(inverse).reshape(-1).astype(np.int32)


def _ols_residual(g, r):
    """Residual of the regression ``g ~ a + b r`` (rows)."""
    out = np.empty_like(g)
    for t in range(g.shape[0]):
        a = np.stack([np.ones(g.shape[1]), r[t]], axis=1)
        coef, *_ = np.linalg.lstsq(a, g[t], rcond=None)
        out[t] = g[t] - a @ coef
    return out


# --------------------------------------------------------------------------
# public: variogram
# --------------------------------------------------------------------------


def _check_model(model):
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, not {model!r}")


@dask_support("computes", "it fits the variogram to all gauges of every time.")
@provenance("Empirical variogram of rain gauges and Cressie weighted least squares fit")
def gauge_variogram(
    gauges,
    *,
    radar=None,
    model="exponential",
    n_lags=None,
    max_lag=None,
    radius=None,
    x=None,
    y=None,
    site=None,
    n_threads=None,
    engine="auto",
):
    """
    Empirical and fitted variogram of the gauge values.

    The semivariance :math:`\\hat\\gamma(h) = \\frac{1}{2N(h)}\\sum
    (z_i - z_j)^2` over the pairs of gauges at distance :math:`h` (Matheron
    1963) in equally wide lag bins, and an isotropic model with a nugget fitted
    by the weighted least squares of Cressie (1985). The fit is repeated for
    every time of the gauge series, with the gauges that have a value then.

    Parameters
    ----------
    gauges : xarray.DataArray
        Gauge values on a station dimension with the coordinates of the
        stations (``x``, ``y`` in m or ``lon``, ``lat`` in degrees), and
        optionally leading dimensions such as ``time``. Stations with a NaN
        value are left out for that time.
    radar : xarray.DataArray, optional
        A rain field on a grid. If given, the variogram is that of the
        residuals of the regression of the gauge values on the radar values at
        the gauges, which is the variogram of kriging with an external drift
        (``method="ked"`` of :func:`gauge_adjust`).
    model : {"linear", "spherical", "exponential", "gaussian"}, optional
        Variogram model. Default ``"exponential"``. The practical range of the
        exponential and gaussian models is where they reach 95 % of the sill.
    n_lags : int, optional
        Number of lag bins, at least 3. Default 15, fewer for few gauges.
    max_lag : float, optional
        Largest distance (m). Default half the largest distance between
        gauges.
    radius : float, optional
        With ``radar``: averaging radius (m), see :func:`radar_at_gauges`.
    x, y, site : optional
        See :func:`radar_at_gauges`.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled
        kernel and falls back to NumPy.

    Returns
    -------
    xarray.Dataset
        On the dimension ``lag`` (the bin centres in m) and the leading
        dimensions of ``gauges``: ``semivariance``, ``distance`` (mean
        distance of the pairs of the bin, m), ``pairs``, ``fitted`` (the model
        at ``distance``), and ``nugget``, ``partial_sill`` (the units of
        ``gauges`` squared; the slope in units per m for the linear model)
        and ``range`` (m; 0 for the linear model) on the leading dimensions.
        The attribute ``model`` names the model. The Dataset can be passed to
        :func:`gauge_adjust` as ``variogram``.

    Raises
    ------
    ValueError
        For an unknown ``model`` or ``engine``, ``n_lags`` below 3, a
        non-positive ``max_lag``, or fewer than 3 gauges with a value.

    References
    ----------
    Matheron, G., 1963: Principles of geostatistics. *Econ. Geol.*, **58**
    (8), 1246-1266, https://doi.org/10.2113/gsecongeo.58.8.1246

    Cressie, N., 1985: Fitting variogram models by weighted least squares.
    *J. Int. Assoc. Math. Geol.*, **17** (5), 563-586,
    https://doi.org/10.1007/BF01032109

    Goovaerts, P., 2000: Geostatistical approaches for incorporating
    elevation into the spatial interpolation of rainfall. *J. Hydrol.*,
    **228** (1-2), 113-129, https://doi.org/10.1016/S0022-1694(00)00144-X

    Examples
    --------
    >>> vario = gauge_variogram(gauges)  # doctest: +SKIP
    >>> adjusted = gauge_adjust(rain, gauges, method="ok", variogram=vario)  # doctest: +SKIP
    """
    _check_model(model)
    use_compiled = _use_compiled(engine)
    if radar is None:
        s = _prepare_gauges_only(gauges, x, y)
    else:
        s = _prepare(radar, gauges, x, y, site)
    mask = np.isfinite(s.G)
    values = s.G
    if radar is not None:
        at = _sample(s, radius)
        mask &= np.isfinite(at)
    results = []
    groups = _row_groups(mask)
    n_lags = _default_lags(max(cols.size for _, cols in groups), n_lags)
    for rows, cols in groups:
        if cols.size < 3:
            continue  # the variogram of these times stays NaN
        v = values[np.ix_(rows, cols)]
        if radar is not None:
            v = _ols_residual(v, at[np.ix_(rows, cols)])
        params, lag, gamma, count = _fit_rows(
            model, s.xg[cols], s.yg[cols], v, n_lags, max_lag, use_compiled, n_threads
        )
        results.append((rows, params, lag, gamma, count))
    if not results:
        raise ValueError("at least 3 gauges with a value are needed for a variogram")
    n_l = n_lags
    nt = s.nt
    semi = np.full((nt, n_l), np.nan)
    dist = np.full((nt, n_l), np.nan)
    pairs = np.zeros((nt, n_l), dtype=np.int64)
    fit = np.full((nt, 3), np.nan)
    for rows, params, lag, gamma, count in results:
        semi[rows] = gamma
        dist[rows] = lag
        pairs[rows] = count
        fit[rows] = params
    fitted = np.full_like(semi, np.nan)
    for t in range(nt):
        if np.isfinite(fit[t, 0]):
            fitted[t] = _semivariance(model, dist[t], *fit[t])
    shape = s.extra_shape
    dims = tuple(s.extra) + ("lag",)
    centre = np.nanmean(dist, axis=0)
    coords = {"lag": ("lag", centre, {"units": "m", "long_name": "lag bin centre"})}
    for d in s.extra:
        if d in gauges.coords:
            coords[d] = gauges.coords[d]
    ds = xr.Dataset(
        {
            "semivariance": (dims, semi.reshape(shape + (n_l,))),
            "distance": (
                dims,
                dist.reshape(shape + (n_l,)),
                {"units": "m", "long_name": "mean distance of the pairs of the bin"},
            ),
            "pairs": (dims, pairs.reshape(shape + (n_l,))),
            "fitted": (dims, fitted.reshape(shape + (n_l,))),
            "nugget": (tuple(s.extra), fit[:, 0].reshape(shape)),
            "partial_sill": (tuple(s.extra), fit[:, 1].reshape(shape)),
            "range": (tuple(s.extra), fit[:, 2].reshape(shape), {"units": "m"}),
        },
        coords=coords,
        attrs={"model": model},
    )
    unit = str(gauges.attrs.get("units", ""))
    if unit:
        for name in ("semivariance", "fitted", "nugget", "partial_sill"):
            ds[name].attrs["units"] = (
                f"({unit})2" if name != "partial_sill" or model != "linear" else unit
            )
    ds["semivariance"].attrs["long_name"] = "empirical semivariance"
    ds["fitted"].attrs["long_name"] = f"fitted {model} variogram"
    return ds


def _prepare_gauges_only(gauges, x, y):
    """Setup with the gauges only (no radar field)."""
    if not isinstance(gauges, xr.DataArray):
        raise TypeError("gauges must be an xarray.DataArray")
    gxn, gyn = _coord_names(gauges, x, y, "gauges")
    gx, gy = gauges.coords[gxn], gauges.coords[gyn]
    if gx.ndim != 1 or gy.ndim != 1 or gx.dims != gy.dims:
        raise ValueError(
            "the gauge coordinates must be 1-D along one station dimension"
        )
    station = gx.dims[0]
    extra = [d for d in gauges.dims if d != station]
    xv = np.asarray(gx.values, dtype=np.float64)
    yv = np.asarray(gy.values, dtype=np.float64)
    if _is_geographic(gauges, gxn):
        tr = _project(float(np.nanmean(yv)), float(np.nanmean(xv)))
        xv, yv = tr.transform(xv, yv)
    s = _Setup()
    s.extra = extra
    s.station = station
    s.xg, s.yg = np.asarray(xv), np.asarray(yv)
    g = gauges.transpose(*extra, station)
    s.nt = (
        int(np.prod([gauges.sizes[d] for d in extra], dtype=np.int64)) if extra else 1
    )
    s.G = np.asarray(g.values, dtype=np.float64).reshape(s.nt, -1)
    s.ng = s.G.shape[1]
    s.extra_shape = tuple(gauges.sizes[d] for d in extra)
    return s


# --------------------------------------------------------------------------
# public: gauge_adjust
# --------------------------------------------------------------------------


def _check_distinct(xg, yg):
    """Kriging needs one gauge per position."""
    pos = np.round(np.stack([xg, yg], axis=1), 3)
    _, first, counts = np.unique(pos, axis=0, return_index=True, return_counts=True)
    if np.any(counts > 1):
        raise ValueError(
            "two gauges have the same position (station index "
            f"{int(first[counts > 1][0])}): merge or remove the duplicate"
        )


def _check_method(method):
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, not {method!r}")


def _mean_field_bias(rg, g, min_value, use_compiled, n_threads):
    if use_compiled:
        return _gauge.mean_field_bias(
            np.ascontiguousarray(rg),
            np.ascontiguousarray(g),
            float(min_value),
            int(n_threads or 0),
        )
    return _mfb_numpy(rg, g, min_value)


def _brandes(xg, yg, ratio, xt, yt, k, use_compiled, n_threads):
    if use_compiled:
        return _gauge.brandes(
            np.ascontiguousarray(xg),
            np.ascontiguousarray(yg),
            np.ascontiguousarray(ratio),
            np.ascontiguousarray(xt),
            np.ascontiguousarray(yt),
            float(k),
            int(n_threads or 0),
        )
    return _brandes_numpy(xg, yg, ratio, xt, yt, k)


def _range_adjustment(s, rg, g, min_value, min_pairs):
    """Range-dependent factor field of Goudenhoofdt and Delobbe (2009), Eq. 2."""
    if s.origin is None:
        raise ValueError(
            "method='rda' needs the position of the radar: pass site=(latitude, longitude)"
        )
    ox, oy = s.origin
    dist_t = np.hypot(s.xr - ox, s.yr - oy) / 1000.0
    dist_g = np.hypot(s.xg - ox, s.yg - oy) / 1000.0
    nt = rg.shape[0]
    factor = np.full((nt, s.npix), np.nan)
    pairs = np.zeros(nt, dtype=np.int64)
    for t in range(nt):
        ok = (rg[t] > min_value) & (g[t] > min_value)
        n = int(ok.sum())
        if n < max(min_pairs, 3) or np.unique(dist_g[ok]).size < 3:
            continue
        coef = np.polyfit(dist_g[ok], np.log(g[t, ok] / rg[t, ok]), 2)
        lo, hi = dist_g[ok].min(), dist_g[ok].max()
        factor[t] = np.exp(np.polyval(coef, np.clip(dist_t, lo, hi)))
        pairs[t] = n
    return factor, pairs


def _kriged(s, method, G, Rg, R, rows, cols, name, params, nn, use_compiled, n_threads):
    """Kriging of the rows ``rows`` with the gauges ``cols``: estimate, std, fallbacks."""
    model = _MODEL_CODE[name]
    xs, ys = s.xg[cols], s.yg[cols]
    gs = G[np.ix_(rows, cols)]
    none = np.empty((0, 0))
    if method == "ok":
        groups = _group_ids(params)
        est, var, _ = _krige(
            xs,
            ys,
            gs,
            none,
            s.xr,
            s.yr,
            none,
            model,
            params,
            groups,
            nn,
            use_compiled,
            n_threads,
        )
        return est, np.sqrt(var), 0
    rs = Rg[np.ix_(rows, cols)]
    if method == "ked":
        groups = np.arange(len(rows), dtype=np.int32)
        est, var, fb = _krige(
            xs,
            ys,
            gs,
            rs,
            s.xr,
            s.yr,
            np.ascontiguousarray(R[rows]),
            model,
            params,
            groups,
            nn,
            use_compiled,
            n_threads,
        )
        return est, np.sqrt(var), int(fb.sum())
    n = len(rows)
    groups = _group_ids(params)
    est, var, _ = _krige(
        xs,
        ys,
        np.vstack([gs, rs]),
        none,
        s.xr,
        s.yr,
        none,
        model,
        np.vstack([params, params]),
        np.concatenate([groups, groups]),
        nn,
        use_compiled,
        n_threads,
    )
    merged = R[rows] - est[n:] + est[:n]
    return merged, np.sqrt(var[:n]), 0


def _adjust_mfb(s, Rg, min_value, min_pairs, use_compiled, n_threads):
    factor, pairs = _mean_field_bias(Rg, s.G, min_value, use_compiled, n_threads)
    apply = pairs >= min_pairs
    out = np.where(apply[:, None], s.R * np.where(apply, factor, 1.0)[:, None], s.R)
    extra = {
        "MEAN_FIELD_BIAS": _extra_array(
            s,
            np.where(apply, factor, np.nan).astype(np.float32),
            "MEAN_FIELD_BIAS",
            {
                "long_name": "mean field bias, sum of gauge over sum of radar values",
                "units": "1",
            },
        )
    }
    return out, np.where(apply, pairs, 0), None, extra


def _adjust_rda(s, Rg, min_value, min_pairs):
    factor, pairs = _range_adjustment(s, Rg, s.G, min_value, min_pairs)
    apply = pairs > 0
    out = np.where(apply[:, None], s.R * np.where(apply[:, None], factor, 1.0), s.R)
    extra = {
        "ADJUSTMENT_FACTOR": _field(
            s,
            np.where(apply[:, None], factor, np.nan).astype(np.float32),
            "ADJUSTMENT_FACTOR",
            {"long_name": "range-dependent adjustment factor", "units": "1"},
        )
    }
    return out, pairs, None, extra


def _adjust_brandes(s, Rg, min_value, min_pairs, k, use_compiled, n_threads):
    G = s.G
    ratio = np.full_like(G, np.nan)
    pair = (Rg > min_value) & (G > min_value)
    ratio[pair] = G[pair] / Rg[pair]
    n_pair = pair.sum(axis=1)
    factor = np.full((s.nt, s.npix), np.nan)
    finite = np.isfinite(s.xr) & np.isfinite(s.yr)
    area = float(np.ptp(s.xr[finite]) * np.ptp(s.yr[finite])) if finite.any() else 0.0
    for n in np.unique(n_pair[n_pair >= min_pairs]):
        rows = np.flatnonzero(n_pair == n)
        kk = float(k) if k is not None else area / (2.0 * float(n))
        if not kk > 0:
            raise ValueError("k must be positive (the radar pixels have no extent)")
        factor[rows] = _brandes(
            s.xg, s.yg, ratio[rows], s.xr, s.yr, kk, use_compiled, n_threads
        )
    apply = np.isfinite(factor).any(axis=1)
    out = np.where(apply[:, None], s.R * np.where(apply[:, None], factor, 1.0), s.R)
    extra = {
        "ADJUSTMENT_FACTOR": _field(
            s,
            factor.astype(np.float32),
            "ADJUSTMENT_FACTOR",
            {"long_name": "Brandes spatial adjustment factor", "units": "1"},
        )
    }
    return out, np.where(apply, n_pair, 0), None, extra


def _constant_rows(s, method, rows, cols, gs, rs, shortcut, out, std, used):
    """Rows whose gauges (and drift) are constant: the field is its own estimate."""
    r = rows[shortcut]
    if method == "conditional":
        out[r] = s.R[r] + (gs[shortcut, 0] - rs[shortcut, 0])[:, None]
    else:
        out[r] = gs[shortcut, 0][:, None]
    std[r] = 0.0
    used[r] = cols.size


def _fit_for_method(s, method, model, cols, gs, rs, const_g, todo, opts):
    """Variogram parameters of the rows ``todo`` of one group of gauges."""
    if method == "ked":
        vals = _ols_residual(gs[todo], rs[todo])
    elif method == "conditional":
        vals = np.where(const_g[todo, None], rs[todo], gs[todo])
    else:
        vals = gs[todo]
    return _fit_rows(model, s.xg[cols], s.yg[cols], vals, *opts)[0]


def _adjust_kriging(
    s,
    Rg,
    method,
    variogram,
    model,
    n_lags,
    max_lag,
    nn,
    min_pairs,
    use_compiled,
    n_threads,
):
    G, R = s.G, s.R
    out, used = R.copy(), np.zeros(s.nt, dtype=np.int64)
    fixed = None
    if variogram is not None:
        fixed = _fixed_variogram(variogram, model, s.extra_shape, s.nt)
        model = fixed[0]
    mask = np.isfinite(G)
    if method != "ok":
        mask &= np.isfinite(Rg)
    std = np.full((s.nt, s.npix), np.nan)
    fallbacks = 0
    for rows, cols in _row_groups(mask):
        if cols.size < min_pairs:
            continue
        gs = G[np.ix_(rows, cols)]
        rs = Rg[np.ix_(rows, cols)] if method != "ok" else None
        const_g = np.ptp(gs, axis=1) == 0.0
        shortcut = const_g
        if method == "conditional":
            shortcut = const_g & (np.ptp(rs, axis=1) == 0.0)
        if shortcut.any():
            _constant_rows(s, method, rows, cols, gs, rs, shortcut, out, std, used)
        todo = ~shortcut
        if not todo.any():
            continue
        r = rows[todo]
        if fixed is not None:
            name, params = fixed[0], fixed[1][r]
        else:
            opts = (n_lags, max_lag, use_compiled, n_threads)
            params = _fit_for_method(
                s, method, model, cols, gs, rs, const_g, todo, opts
            )
            name = model
        est, sd, fb = _kriged(
            s, method, G, Rg, R, r, cols, name, params, nn, use_compiled, n_threads
        )
        out[r] = est
        std[r] = sd
        used[r] = cols.size
        fallbacks += fb
    out = np.where(np.isfinite(R), out, np.nan)
    std = np.where(np.isfinite(R), std, np.nan)
    attrs = {"variogram_model": model, "n_neighbours": 0 if nn is None else nn}
    if method == "ked":
        attrs["ked_fallback"] = int(fallbacks)
    return out, used, std, {}, attrs


@dask_support("computes", "kriging needs every gauge and the whole field of each time.")
@provenance(
    "Radar-gauge merging: mean field bias, range-dependent adjustment, Brandes, kriging"
)
def gauge_adjust(
    obj,
    gauges,
    *,
    method="ked",
    variogram=None,
    model="exponential",
    n_lags=None,
    max_lag=None,
    n_neighbours=20,
    radius=None,
    min_value=0.0,
    min_pairs=5,
    clip=True,
    k=None,
    x=None,
    y=None,
    site=None,
    n_threads=None,
    engine="auto",
):
    """
    Adjust a radar rain field to rain gauges.

    Parameters
    ----------
    obj : xarray.DataArray
        The radar rain field (an accumulation or a rate) on a grid ``(y, x)``
        or on any layout with 2-D coordinates, optionally with leading
        dimensions such as ``time``. Radar and gauges must have the same
        units. Pixels that are NaN stay NaN.
    gauges : xarray.DataArray
        The gauge values on a station dimension with the coordinates of the
        stations (``x``, ``y`` in metres east and north of the radar, or
        ``lon``, ``lat`` in degrees), and the same leading dimensions as
        ``radar``. NaN marks a missing value; a time with fewer than
        ``min_pairs`` usable gauges returns the radar field unchanged.
    method : {"mfb", "rda", "brandes", "ok", "conditional", "ked"}, optional
        Default ``"ked"``, the best method of Goudenhoofdt and Delobbe (2009).
        See :mod:`radarx.retrieve.gauge` for the methods.
    variogram : dict or xarray.Dataset, optional
        Kriging methods only. A fixed variogram: ``{"model": ..., "nugget":
        ..., "partial_sill": ..., "range": ...}`` (``"slope"`` replaces
        ``"partial_sill"`` for the linear model), or the result of
        :func:`gauge_variogram`. By default the variogram is fitted to the
        gauges for every time. A fixed variogram makes the kriging weights
        the same for all times, which is much faster for long series.
    model : {"linear", "spherical", "exponential", "gaussian"}, optional
        Variogram model of the automatic fit. Default ``"exponential"``.
    n_lags, max_lag : optional
        Lags of the empirical variogram, see :func:`gauge_variogram`.
    n_neighbours : int or None, optional
        Kriging methods only: number of nearest gauges used at each pixel, at
        least 3 (default 20, as Goudenhoofdt and Delobbe 2009). ``None``: all
        gauges, with one matrix factorization for all pixels.
    radius : float, optional
        Radius (m) over which the radar is averaged at the gauges, see
        :func:`radar_at_gauges`.
    min_value : float, optional
        Pairs of gauge and radar value count only if both exceed this value
        (``"mfb"``, ``"rda"``, ``"brandes"``). Default 0: both positive.
        Goudenhoofdt and Delobbe (2009) use 1 mm for daily accumulations.
    min_pairs : int, optional
        Fewest pairs (``"mfb"``, ``"rda"``, ``"brandes"``) or gauges (kriging
        methods) for which a time is adjusted, at least 2. Default 5. The
        paper requires 10 pairs per day.
    clip : bool, optional
        Set negative results to zero. Default ``True``.
    k : float, optional
        ``"brandes"`` only: the smoothing parameter of the weights
        :math:`\\exp(-d^2/k)` (m2). Default :math:`1/(2\\delta)` with the
        density :math:`\\delta` of the pairs per area of the bounding box of
        the radar pixels.
    x, y, site : optional
        See :func:`radar_at_gauges`. ``"rda"`` needs the position of the radar:
        the origin of ``x`` and ``y``, or ``site`` for geographic coordinates.
    n_threads : int, optional
        Threads for the compiled kernels. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled
        kernel and falls back to NumPy.

    Returns
    -------
    xarray.Dataset
        ``RAIN_ADJUSTED`` (float32, the units and dimensions of ``radar``);
        ``GAUGES_USED`` (number of pairs or gauges used per time, 0 where the
        radar field is returned unchanged); ``RAIN_ADJUSTED_STD`` (kriging
        standard deviation, in the units of ``radar``; ``"ok"``,
        ``"conditional"``, ``"ked"``); ``MEAN_FIELD_BIAS`` (``"mfb"``);
        ``ADJUSTMENT_FACTOR`` (the factor field of ``"rda"`` and
        ``"brandes"``). The attributes name the method and, for kriging, the
        variogram model and the number of pixels where the drift was
        constant and ordinary kriging was used instead (``ked_fallback``).

    Raises
    ------
    TypeError
        If ``radar`` or ``gauges`` is not a DataArray.
    ValueError
        For an unknown ``method``, ``model`` or ``engine``, ``min_pairs`` below
        2, ``n_neighbours`` below 3, coordinates or dimensions that do not
        match, a radar field without finite values, or ``"rda"`` without the
        position of the radar.

    Notes
    -----
    The adjusted field of ``"ok"`` is the gauge-only interpolation on the
    pixels where the radar is finite. ``"ked"`` and ``"conditional"`` need
    the radar value at the gauges, so a gauge outside the radar field (or
    without a finite value within ``radius``) is left out.

    The methods are those of Goudenhoofdt and Delobbe (2009) without the
    static local bias correction. How radarx fits the variogram, which
    gauges it uses and what it does for a constant drift is listed in
    :mod:`radarx.retrieve.gauge`; the methods are not reproduced against the
    data of the paper (not available), but against closed-form kriging
    solutions and independent solvers (see the tests).

    References
    ----------
    Goudenhoofdt, E., and L. Delobbe, 2009: Evaluation of radar-gauge merging
    methods for quantitative precipitation estimates. *Hydrol. Earth Syst.
    Sci.*, **13** (2), 195-203, https://doi.org/10.5194/hess-13-195-2009

    Sinclair, S., and G. Pegram, 2005: Combining radar and rain gauge
    rainfall estimates using conditional merging. *Atmos. Sci. Lett.*, **6**
    (1), 19-22, https://doi.org/10.1002/asl.85

    Brandes, E. A., 1975: Optimizing rainfall estimates with the aid of
    radar. *J. Appl. Meteor.*, **14** (7), 1339-1345,
    https://doi.org/10.1175/1520-0450(1975)014<1339:OREWTA>2.0.CO;2

    Smith, J. A., and W. F. Krajewski, 1991: Estimation of the mean field
    bias of radar rainfall estimates. *J. Appl. Meteor.*, **30** (4),
    397-412, https://doi.org/10.1175/1520-0450(1991)030<0397:EOTMFB>2.0.CO;2

    Wackernagel, H., 2003: *Multivariate Geostatistics*. 3rd ed. Springer,
    https://doi.org/10.1007/978-3-662-05294-5

    Cressie, N., 1985: Fitting variogram models by weighted least squares.
    *J. Int. Assoc. Math. Geol.*, **17** (5), 563-586,
    https://doi.org/10.1007/BF01032109

    Examples
    --------
    >>> merged = gauge_adjust(rain, gauges, method="ked")  # doctest: +SKIP
    >>> merged.RAIN_ADJUSTED.plot()  # doctest: +SKIP
    >>> merged = rain.radarx.gauge_adjust(gauges, method="conditional")  # doctest: +SKIP
    """
    radar = obj
    _check_method(method)
    _check_model(model)
    use_compiled = _use_compiled(engine)
    if int(min_pairs) < 2:
        raise ValueError("min_pairs must be at least 2")
    if n_neighbours is not None and int(n_neighbours) < 3:
        raise ValueError("n_neighbours must be at least 3, or None for all gauges")
    s = _prepare(radar, gauges, x, y, site)
    if not np.isfinite(s.R).any():
        raise ValueError("the radar field has no finite value")
    if s.ng < int(min_pairs):
        raise ValueError(f"there are {s.ng} gauges, fewer than min_pairs={min_pairs}")
    nn = None if n_neighbours is None else int(n_neighbours)
    mp = int(min_pairs)
    attrs = {"method": method}
    Rg = _sample(s, radius) if method != "ok" else None
    if method in ("ok", "conditional", "ked"):
        _check_distinct(s.xg, s.yg)
    if method == "mfb":
        out, used, std, extra = _adjust_mfb(
            s, Rg, min_value, mp, use_compiled, n_threads
        )
    elif method == "rda":
        out, used, std, extra = _adjust_rda(s, Rg, min_value, mp)
    elif method == "brandes":
        out, used, std, extra = _adjust_brandes(
            s, Rg, min_value, mp, k, use_compiled, n_threads
        )
    else:
        out, used, std, extra, more = _adjust_kriging(
            s,
            Rg,
            method,
            variogram,
            model,
            n_lags,
            max_lag,
            nn,
            mp,
            use_compiled,
            n_threads,
        )
        attrs.update(more)
    if clip:
        out = np.where(out < 0.0, 0.0, out)
    units = radar.attrs.get("units")
    field_attrs = {
        "long_name": "radar rain adjusted to rain gauges",
        "comment": f"method {method}",
    }
    if units:
        field_attrs["units"] = units
    data = {
        "RAIN_ADJUSTED": _field(
            s, out.astype(np.float32), "RAIN_ADJUSTED", field_attrs
        ),
        "GAUGES_USED": _extra_array(
            s,
            used.astype(np.int32),
            "GAUGES_USED",
            {"long_name": "number of gauges or pairs used, 0 if not adjusted"},
        ),
    }
    if std is not None:
        sd_attrs = {"long_name": "kriging standard deviation"}
        if units:
            sd_attrs["units"] = units
        data["RAIN_ADJUSTED_STD"] = _field(
            s, std.astype(np.float32), "RAIN_ADJUSTED_STD", sd_attrs
        )
    data.update(extra)
    return xr.Dataset(data, attrs=attrs)


@accessor_method("dataarray", name="gauge_adjust", wraps=gauge_adjust)
def _gauge_adjust_accessor(self):
    """
    Adjust this radar rain field to rain gauges.

    See :func:`radarx.retrieve.gauge_adjust` for the parameters.

    Returns
    -------
    xarray.Dataset
        ``RAIN_ADJUSTED`` and the other products of the method.
    """
