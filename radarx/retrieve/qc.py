#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Non-Meteorological Echo Filtering
=================================

Separate meteorological echo (rain, snow, hail, ice) from non-meteorological
echo (insects, birds, ground clutter, anomalous propagation, sea clutter,
second-trip echo, noise) gate by gate, before KDP, hydrometeor
classification, rain rates or gridding.

The classification follows the fuzzy-logic approach of Gourley et al. (2007)
and Krause (2016): local features of the polarimetric variables and of the
reflectivity are mapped to a membership in [0, 1] ("how meteorological"),
and the weighted mean of the memberships is compared with a threshold. Every
feature is computed along the ray over a window of ``window`` km centred on
the gate:

``rhohv``
    Mean copolar correlation coefficient. Rain, snow and ice have
    :math:`\\rho_{hv} > 0.95`; biological scatterers and clutter much lower
    values (Gourley et al. 2007; Park et al. 2009; Tang et al. 2014). Values
    above 1, which occur only at low signal-to-noise ratio, are reflected
    about 1 (1.05 counts as 0.95).
``zdr``
    Mean differential reflectivity. Insects and birds have :math:`Z_{DR}` of
    several dB up to more than 10 dB at low reflectivity, rarely seen in
    precipitation (Park et al. 2009; Tang et al. 2014).
``zdr_texture``, ``phidp_texture``
    Standard deviation of :math:`Z_{DR}` and circular standard deviation of
    :math:`\\Phi_{DP}`. Both are small in precipitation, where the
    scatterers in neighbouring gates are alike, and large for clutter,
    biological echo and noise (Gourley et al. 2007; Krause 2016).
``dbz_texture``
    Mean squared difference of the reflectivity of adjacent gates
    (:math:`\\mathrm{dB}^2`), large for ground clutter and anomalous
    propagation (Steiner and Smith 2002).
``spin``
    Share (%) of gates at which the reflectivity gradient along the ray
    changes sign with jumps of at least ``spin_threshold`` dB on both sides,
    the "spin change" of Steiner and Smith (2002), large for clutter.

Each membership is a trapezoid ``(a, b, c, d)``: 0 below ``a`` and above
``d``, 1 between ``b`` and ``c``, linear in between (``limits``). Features of
missing fields are left out of the weighted mean, so the method works with
any subset of the polarimetric variables (with reflectivity alone only the
texture features remain). The score is then averaged over the 3 x 3
neighbouring gates (adjacent rays and gates), which makes the decision
spatially coherent, and gates scoring at least ``threshold`` are
meteorological. Finally, connected meteorological regions (8-connected,
across north in full sweeps) smaller than ``min_size`` gates are removed as
speckle, as are gates with fewer than half of the window holding echo.

Gates without signal are not classified: missing values, a signal-to-noise
ratio below ``snr_min`` (if an SNR field is given), and the no-data codes
of NEXRAD Level II data as decoded by xradar (reflectivity
:math:`\\leq` -32 dBZ, :math:`Z_{DR} \\leq` -12.9 dB,
:math:`\\rho_{hv} \\leq` 0.21, :math:`\\Phi_{DP} < 0`; ``nodata``).

In a NEXRAD volume the Doppler cuts of the split-cut elevations hold no
polarimetric variables. By default (``split_cuts=True``) their gates take
the class of the nearest gate of the surveillance cut at the same elevation.

A compiled C++ kernel computes the features with running (prefix) sums along
each ray, so every gate costs the same regardless of the window, and
processes all rays of all sweeps of a volume in one multithreaded call; the
speckle filter runs per sweep in parallel. An equivalent NumPy implementation
is the fallback.

References
----------
Gourley, J. J., P. Tabary, and J. Parent du Chatelet, 2007: A fuzzy logic
algorithm for the separation of precipitating from nonprecipitating echoes
using polarimetric radar observations. *J. Atmos. Oceanic Technol.*,
**24** (8), 1439-1451, https://doi.org/10.1175/JTECH2035.1

Krause, J. M., 2016: A simple algorithm to discriminate between
meteorological and nonmeteorological radar echoes. *J. Atmos. Oceanic
Technol.*, **33** (9), 1875-1885, https://doi.org/10.1175/JTECH-D-15-0239.1

Park, H. S., A. V. Ryzhkov, D. S. Zrnić, and K.-E. Kim, 2009: The
hydrometeor classification algorithm for the polarimetric WSR-88D:
Description and application to an MCS. *Wea. Forecasting*, **24** (3),
730-748, https://doi.org/10.1175/2008WAF2222205.1

Steiner, M., and J. A. Smith, 2002: Use of three-dimensional reflectivity
structure for automated detection and removal of nonprecipitating echoes in
radar data. *J. Atmos. Oceanic Technol.*, **19** (5), 673-686,
https://doi.org/10.1175/1520-0426(2002)019<0673:UOTDRS>2.0.CO;2

Tang, L., J. Zhang, C. Langston, J. Krause, K. Howard, and V. Lakshmanan,
2014: A physically based precipitation-nonprecipitation radar echo
classifier using polarimetric and environmental data in a real-time
national system. *Wea. Forecasting*, **29** (5), 1106-1119,
https://doi.org/10.1175/WAF-D-13-00072.1

.. autosummary::
   :nosignatures:
   :toctree: generated/

   echo_mask
   apply_mask
"""

from __future__ import annotations

__all__ = ["echo_mask", "apply_mask"]

import numpy as np
import xarray as xr

from .dealias import _ray_links, _sweep_mapping
from .kdp import _DBZH_NAMES, _PHIDP_NAMES, _RHOHV_NAMES, _find, _gate_spacing

try:
    from . import _qc

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _qc = None
    HAS_COMPILED_KERNEL = False

_ZDR_NAMES = (
    "ZDR",
    "differential_reflectivity",
    "corrected_differential_reflectivity",
)
_SNR_NAMES = ("SNRH", "SNR", "signal_to_noise_ratio")

#: Echo classes of ``ECHO_CLASS``.
CLASSES = {0: "no_echo", 1: "meteorological", 2: "non_meteorological", 3: "speckle"}

#: Features in the order of the kernel.
FEATURES = (
    "rhohv",
    "zdr",
    "zdr_texture",
    "phidp_texture",
    "dbz_texture",
    "spin",
)

_INF = np.inf

#: Default trapezoid membership ``(a, b, c, d)`` of every feature.
DEFAULT_LIMITS = {
    "rhohv": (0.80, 0.95, _INF, _INF),
    "zdr": (-4.0, -2.0, 3.0, 6.0),
    "zdr_texture": (-_INF, -_INF, 1.0, 2.5),
    "phidp_texture": (-_INF, -_INF, 12.0, 30.0),
    "dbz_texture": (-_INF, -_INF, 30.0, 70.0),
    "spin": (-_INF, -_INF, 30.0, 60.0),
}

#: Default weight of every feature.
DEFAULT_WEIGHTS = {
    "rhohv": 1.0,
    "zdr": 1.0,
    "zdr_texture": 1.0,
    "phidp_texture": 1.0,
    "dbz_texture": 0.25,
    "spin": 0.25,
}

#: No-data floors of NEXRAD Level II data as decoded by xradar: values at or
#: below these are flags (below threshold, range folded), not data.
NEXRAD_NODATA = {
    "dbzh": -32.0,
    "zdr": -12.9,
    "rhohv": 0.21,
    "phidp": -0.1,
}
#: Field floors used by :func:`apply_mask` for NEXRAD data.
_NEXRAD_FIELD_NODATA = {
    "DBZH": -32.0,
    "ZDR": -12.9,
    "RHOHV": 0.21,
    "PHIDP": -0.1,
    "VRADH": -63.9,
}

_ROLES = ("dbzh", "zdr", "rhohv", "phidp")


# --------------------------------------------------------------------------
# NumPy reference implementation (same steps and order as the C++ kernel)
# --------------------------------------------------------------------------


def _prefix(a):
    """Prefix sums along the last axis with a leading zero (sequential)."""
    out = np.zeros(a.shape[:-1] + (a.shape[-1] + 1,))
    np.cumsum(a, axis=-1, out=out[..., 1:])
    return out


def _span(prefix, lo, hi):
    """Window sums ``prefix[hi] - prefix[lo]`` (zero where ``hi <= lo``)."""
    rows = np.arange(prefix.shape[0])[:, None]
    return np.where(hi > lo, prefix[rows, np.maximum(hi, lo)] - prefix[rows, lo], 0.0)


def _trapezoid(x, limits):
    """Trapezoidal membership of ``x`` for ``limits = (a, b, c, d)``."""
    a, b, c, d = limits
    with np.errstate(invalid="ignore", divide="ignore"):
        up = (x - a) / (b - a)
        down = (d - x) / (d - c)
    m = np.where(x < b, up, np.where(x > c, down, 1.0))
    return np.where((x <= a) | (x >= d), 0.0, m)


def _valid(x, floor, echo):
    if x is None:
        return None
    with np.errstate(invalid="ignore"):
        return echo & np.isfinite(x) & (x > floor)


def _features_numpy(sweep, p):
    """Raw score (float32) and the gates with too little echo."""
    z = sweep["dbzh"]
    nray, ng = z.shape
    h = sweep["h"]
    floors = p["floors"]
    with np.errstate(invalid="ignore"):
        echo = np.isfinite(z) & (z > floors[0])
        if sweep["snr"] is not None:
            echo &= sweep["snr"] >= p["snr_min"]
    g = np.arange(ng)
    lo = np.broadcast_to(np.maximum(g - h, 0), (nray, ng))
    hi = np.broadcast_to(np.minimum(g + h, ng - 1) + 1, (nray, ng))
    num = np.zeros((nray, ng))
    den = np.zeros((nray, ng))

    def add(name, value, defined):
        k = FEATURES.index(name)
        w = p["weights"][k]
        if w <= 0:
            return
        m = _trapezoid(np.where(defined, value, 0.0), p["limits"][k])
        num[:] = num + np.where(defined, w * m, 0.0)
        den[:] = den + np.where(defined, w, 0.0)

    ne = _span(_prefix(echo.astype(np.float64)), lo, hi)

    # polarimetric means and textures
    for role, k in (("rhohv", 2), ("zdr", 1)):
        x = sweep[role]
        ok = _valid(x, floors[k], echo)
        if ok is None:
            continue
        v = np.where(ok, x, 0.0)
        if role == "rhohv":
            v = np.where(v > 1.0, 2.0 - v, v)
        n = _span(_prefix(ok.astype(np.float64)), lo, hi)
        s1 = _span(_prefix(v), lo, hi)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = s1 / n
        add(role, mean, n >= 1)
        if role == "zdr":
            s2 = _span(_prefix(v * v), lo, hi)
            with np.errstate(invalid="ignore", divide="ignore"):
                var = s2 / n - mean * mean
            add("zdr_texture", np.sqrt(np.maximum(var, 0.0)), n >= 3)
    phi = sweep["phidp"]
    ok = _valid(phi, floors[3], echo)
    if ok is not None:
        rad = np.deg2rad(np.where(ok, phi, 0.0))
        n = _span(_prefix(ok.astype(np.float64)), lo, hi)
        c = _span(_prefix(np.where(ok, np.cos(rad), 0.0)), lo, hi)
        s = _span(_prefix(np.where(ok, np.sin(rad), 0.0)), lo, hi)
        with np.errstate(invalid="ignore", divide="ignore"):
            r2 = np.clip((c * c + s * s) / (n * n), 1e-300, 1.0)
            sd = np.sqrt(-np.log(r2)) * (180.0 / np.pi)
        add("phidp_texture", sd, n >= 3)

    # reflectivity texture: pairs (g, g + 1) and spin triples centred on g
    zz = np.where(echo, z, 0.0)
    pair = np.zeros((nray, ng))
    pair[:, :-1] = echo[:, :-1] & echo[:, 1:]
    dz = np.zeros((nray, ng))
    dz[:, :-1] = zz[:, 1:] - zz[:, :-1]
    dz2 = np.where(pair > 0, dz * dz, 0.0)
    npair = _span(_prefix(pair), lo, hi - 1)
    tdbz = _span(_prefix(dz2), lo, hi - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        add("dbz_texture", tdbz / npair, npair >= 1)
    trip = np.zeros((nray, ng))
    trip[:, 1:-1] = pair[:, :-2] * pair[:, 1:-1]
    t = p["spin_threshold"]
    d1 = np.zeros((nray, ng))
    d2 = np.zeros((nray, ng))
    d1[:, 1:-1] = dz[:, :-2]
    d2[:, 1:-1] = dz[:, 1:-1]
    flip = (trip > 0) & (d1 * d2 < 0) & (np.abs(d1) >= t) & (np.abs(d2) >= t)
    ntrip = _span(_prefix(trip), lo + 1, hi - 1)
    nspin = _span(_prefix(flip.astype(np.float64)), lo + 1, hi - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        add("spin", 100.0 * nspin / ntrip, ntrip >= 1)

    isolated = echo & (ne < h + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = num / den
    raw = np.where(echo & ~isolated & (den > 0), raw, np.nan).astype(np.float32)
    return raw, echo


def _neighbour_rays(links):
    """Previous and next linked ray of every ray (-1 if none)."""
    nray = links.size
    rays = np.arange(nray)
    nxt = np.where(links.astype(bool), (rays + 1) % nray, -1)
    prv = np.where(links[(rays - 1) % nray].astype(bool), (rays - 1) % nray, -1)
    nxt = np.where(nxt == rays, -1, nxt)
    prv = np.where((prv == rays) | (prv == nxt), -1, prv)
    return prv, nxt


def _smooth_numpy(raw, links):
    """Mean of the finite raw scores over the 3 x 3 neighbouring gates."""
    nray, ng = raw.shape
    prv, nxt = _neighbour_rays(links)
    r = raw.astype(np.float64)
    total = np.zeros((nray, ng))
    count = np.zeros((nray, ng))
    for nb in (prv, np.arange(nray), nxt):
        has = nb >= 0
        rows = r[np.where(has, nb, 0)]
        rows = np.where(has[:, None], rows, np.nan)
        for dg in (-1, 0, 1):
            shifted = np.full((nray, ng), np.nan)
            if dg < 0:
                shifted[:, 1:] = rows[:, :-1]
            elif dg > 0:
                shifted[:, :-1] = rows[:, 1:]
            else:
                shifted = rows
            fin = np.isfinite(shifted)
            total = total + np.where(fin, shifted, 0.0)
            count = count + fin
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / count
    return np.where(np.isfinite(raw), out, np.nan).astype(np.float32)


def _components(met, links):
    """Size of the 8-connected region of every gate of ``met`` (0 elsewhere)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    nray, ng = met.shape
    n = nray * ng
    gate = np.arange(n).reshape(nray, ng)
    _, nxt = _neighbour_rays(links)
    a = [gate[:, :-1].ravel()]
    b = [gate[:, 1:].ravel()]
    rays = np.flatnonzero(nxt >= 0)
    for dg in (-1, 0, 1):
        src = gate[rays][:, max(0, -dg) : ng - max(0, dg)]
        dst = gate[nxt[rays]][:, max(0, dg) : ng - max(0, -dg)]
        a.append(src.ravel())
        b.append(dst.ravel())
    a = np.concatenate(a)
    b = np.concatenate(b)
    flat = met.ravel()
    keep = flat[a] & flat[b]
    graph = coo_matrix((np.ones(keep.sum()), (a[keep], b[keep])), shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    size = np.bincount(comp[flat], minlength=comp.max() + 1)
    return np.where(flat, size[comp], 0).reshape(nray, ng)


def _classify_numpy(sweep, p):
    """Score and class of one sweep (NumPy reference)."""
    raw, echo = _features_numpy(sweep, p)
    score = _smooth_numpy(raw, sweep["links"])
    cls = np.where(echo, 3, 0).astype(np.int8)
    fin = np.isfinite(score)
    meteo = score >= np.float32(p["threshold"])  # compared in float32, as the kernel
    cls[fin & meteo] = 1
    cls[fin & ~meteo] = 2
    if p["min_size"] > 1:
        size = _components(cls == 1, sweep["links"])
        cls[(cls == 1) & (size < p["min_size"])] = 3
    return score, cls


def _classify_compiled(sweeps, p, n_threads):
    """Run the C++ kernel on all sweeps at once (one pool of rays)."""
    return _qc.classify(
        [sw["dbzh"] for sw in sweeps],
        [sw["zdr"] for sw in sweeps],
        [sw["rhohv"] for sw in sweeps],
        [sw["phidp"] for sw in sweeps],
        [sw["snr"] for sw in sweeps],
        [sw["links"] for sw in sweeps],
        [sw["h"] for sw in sweeps],
        np.asarray(p["floors"], dtype=np.float64),
        float(p["snr_min"]),
        np.asarray(p["limits"], dtype=np.float64),
        np.asarray(p["weights"], dtype=np.float64),
        float(p["spin_threshold"]),
        float(p["threshold"]),
        int(p["min_size"]),
        int(n_threads or 0),
    )


# --------------------------------------------------------------------------
# xarray layer
# --------------------------------------------------------------------------


def _use_compiled(engine):
    """Whether to run the compiled kernel for the requested ``engine``."""
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled echo classification kernel is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


def _is_nexrad(ds, root_attrs=None):
    """NEXRAD Level II data as decoded by xradar."""
    if "waveform_type" in ds.attrs:
        return True
    scan = (root_attrs or {}).get("scan_name", ds.attrs.get("scan_name", ""))
    return isinstance(scan, str) and scan.upper().startswith("VCP")


def _floors(nodata, nexrad):
    """No-data floors of (dbzh, zdr, rhohv, phidp)."""
    if isinstance(nodata, str):
        if nodata not in ("auto", "nexrad"):
            raise ValueError(
                f"nodata must be 'auto', 'nexrad', None or a dict, not {nodata!r}"
            )
        table = NEXRAD_NODATA if (nodata == "nexrad" or nexrad) else {}
    elif nodata is None:
        table = {}
    elif isinstance(nodata, dict):
        unknown = set(nodata) - set(_ROLES)
        if unknown:
            raise ValueError(f"unknown nodata keys {sorted(unknown)}; use {_ROLES}")
        table = nodata
    else:
        raise ValueError(
            f"nodata must be 'auto', 'nexrad', None or a dict, not {nodata!r}"
        )
    return [float(table.get(role, -_INF)) for role in _ROLES]


def _options(limits, weights):
    """Membership limits (6 x 4) and weights (6) in kernel order."""
    lim = dict(DEFAULT_LIMITS)
    wts = dict(DEFAULT_WEIGHTS)
    for given, table, kind in ((limits, lim, "limits"), (weights, wts, "weights")):
        given = given or {}
        unknown = set(given) - set(FEATURES)
        if unknown:
            raise ValueError(f"unknown {kind} keys {sorted(unknown)}; use {FEATURES}")
        table.update(given)
    out = np.array([[float(v) for v in lim[k]] for k in FEATURES], dtype=np.float64)
    if out.shape != (len(FEATURES), 4) or np.any(out[:, 1:] < out[:, :-1]):
        raise ValueError(
            "each membership needs four non-decreasing limits (a, b, c, d)"
        )
    w = np.array([float(wts[k]) for k in FEATURES], dtype=np.float64)
    if np.any(w < 0) or not np.any(w > 0):
        raise ValueError("weights must be non-negative and not all zero")
    return out, w


def _prepare_sweep(ds, fields, window):
    """Field names and contiguous (ray, range) float64 arrays of one sweep."""
    dbzh, zdr, rhohv, phidp, snr = fields
    names = {
        "dbzh": _find(ds, dbzh, _DBZH_NAMES, True),
        "zdr": _find(ds, zdr, _ZDR_NAMES, zdr is not None),
        "rhohv": _find(ds, rhohv, _RHOHV_NAMES, rhohv is not None),
        "phidp": _find(ds, phidp, _PHIDP_NAMES, phidp is not None),
        "snr": _find(ds, snr, _SNR_NAMES, snr is not None),
    }
    da = ds[names["dbzh"]]
    if "range" not in da.dims or da.ndim != 2:
        raise ValueError(f"{names['dbzh']!r} must be 2-D with a 'range' dimension")
    ray_dim = da.dims[0] if da.dims[1] == "range" else da.dims[1]
    dr = _gate_spacing(ds["range"].values)

    def arr(name):
        if name is None:
            return None
        values = ds[name].transpose(ray_dim, "range").values
        return np.ascontiguousarray(values, dtype=np.float64)

    if "azimuth" in ds.coords and ray_dim == "azimuth":
        links = _ray_links(ds["azimuth"].values)
    else:
        links = np.ones(da.sizes[ray_dim], dtype=np.uint8)
        links[-1] = 0  # RHI or unknown geometry: no wrap
    sweep = {role: arr(name) for role, name in names.items()}
    sweep.update(
        {
            "ds": ds,
            "names": names,
            "ray_dim": ray_dim,
            "links": np.ascontiguousarray(links, dtype=np.uint8),
            "h": max(1, round(0.5 * float(window) / dr)),
        }
    )
    return sweep


def _wrap_sweep(sweep, score, cls, threshold, comment=None):
    """Score, class and mask on the input sweep coordinates."""
    da = sweep["ds"][sweep["names"]["dbzh"]]
    dims = da.dims
    if dims[0] != sweep["ray_dim"]:  # kernel output is (ray, range)
        score, cls = score.T, cls.T
    used = ", ".join(n for n in sweep["names"].values() if n)
    flags = np.array(sorted(CLASSES), dtype=np.int8)
    out = xr.Dataset(
        {
            "ECHO_CLASS": (
                dims,
                cls,
                {
                    "long_name": "Echo classification",
                    "flag_values": flags,
                    "flag_meanings": " ".join(CLASSES[int(k)] for k in flags),
                    "source_fields": used,
                },
            ),
            "METEO_SCORE": (
                dims,
                score,
                {
                    "long_name": "Meteorological echo score",
                    "units": "1",
                    "comment": "weighted mean fuzzy membership, 3 x 3 average; "
                    f"meteorological at >= {threshold}",
                },
            ),
            "METEO_MASK": (
                dims,
                cls == 1,
                {
                    "long_name": "Meteorological echo mask",
                    "comment": "True where the gate holds meteorological echo",
                },
            ),
        },
        coords=dict(da.coords),
    )
    if comment:
        for name in ("ECHO_CLASS", "METEO_SCORE"):
            out[name].attrs["split_cut_source"] = (
                f"gates with echo in both cuts classified in {comment} "
                "(same elevation, polarimetric)"
            )
    return out


def _run(sweeps, params, n_threads, use_compiled):
    if use_compiled:
        scores, classes = _classify_compiled(sweeps, params, n_threads)
        return list(zip(scores, classes))
    return [_classify_numpy(sw, params) for sw in sweeps]


def _polarimetric(sweep):
    return any(sweep[r] is not None for r in ("zdr", "rhohv", "phidp"))


def _borrow_split_cuts(sweeps, results, tolerance=0.1):
    """Class of Doppler cuts from the polarimetric cut at the same elevation."""
    angles = [_fixed_angle(sw["ds"]) for sw in sweeps]
    comments = [None] * len(sweeps)
    for i, sw in enumerate(sweeps):
        if _polarimetric(sw) or angles[i] is None or sw["ray_dim"] != "azimuth":
            continue
        cands = [
            j
            for j, other in enumerate(sweeps)
            if _polarimetric(other)
            and angles[j] is not None
            and other["ray_dim"] == "azimuth"
            and abs(angles[j] - angles[i]) <= tolerance
        ]
        if not cands:
            continue
        j = min(cands, key=lambda k: abs(k - i))
        iray, igate = _sweep_mapping(sweeps[j]["ds"], sw["ds"])
        src_score, src_cls = results[j]
        score, cls = results[i]
        # nearest source gate of every gate (-1 outside the source sweep)
        ngs = src_cls.shape[1]
        flat = np.where(
            (iray[:, None] >= 0) & (igate[None, :] >= 0),
            iray[:, None].astype(np.int64) * ngs + igate[None, :],
            -1,
        )
        has = (cls != 0) & (flat >= 0)
        src = flat[has]
        take_cls = src_cls.ravel()[src]
        keep = take_cls != 0
        cls = cls.copy()
        score = score.copy()
        idx = np.flatnonzero(has)[keep]
        cls.ravel()[idx] = take_cls[keep]
        score.ravel()[idx] = src_score.ravel()[src[keep]]
        results[i] = (score, cls)
        comments[i] = sweeps[j]["name"]
    return results, comments


def _fixed_angle(ds):
    for name in ("sweep_fixed_angle", "fixed_angle"):
        if name in ds.variables:
            return float(np.asarray(ds[name].values).ravel()[0])
    if "elevation" in ds.coords:
        return float(np.nanmedian(ds["elevation"].values))
    return None  # pragma: no cover - xradar sweeps have one of the above


def echo_mask(
    obj,
    dbzh=None,
    zdr=None,
    rhohv=None,
    phidp=None,
    snr=None,
    *,
    window=1.5,
    threshold=0.6,
    limits=None,
    weights=None,
    spin_threshold=2.0,
    min_size=10,
    snr_min=3.0,
    nodata="auto",
    split_cuts=True,
    n_threads=None,
    engine="auto",
):
    """
    Classify every gate as meteorological or non-meteorological echo.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataTree
        A sweep on ``(azimuth, range)`` (or ``(elevation, range)``), or a
        volume with ``sweep_*`` groups, e.g. from xradar. Range gates must be
        uniformly spaced; ``range`` is in metres.
    dbzh, zdr, rhohv, phidp, snr : str, optional
        Field names of the reflectivity (dBZ, required), differential
        reflectivity (dB), copolar correlation coefficient, differential
        phase (degrees) and signal-to-noise ratio (dB). By default the first
        name found is used: ``DBZH``, ``DBZ``, ``reflectivity``,
        ``corrected_reflectivity``; ``ZDR``, ``differential_reflectivity``,
        ``corrected_differential_reflectivity``; ``RHOHV``,
        ``cross_correlation_ratio``, ``copol_correlation_coeff``; ``UPHIDP``,
        ``PHIDP``, ``uncorrected_differential_phase``, ``differential_phase``,
        ``PHI``; ``SNRH``, ``SNR``, ``signal_to_noise_ratio``. Missing
        optional fields are left out of the classification.
    window : float, optional
        Length (km) of the window along the ray for the features. Default 1.5.
    threshold : float, optional
        Gates whose score (weighted mean membership, averaged over 3 x 3
        gates) is at least this are meteorological. Default 0.6.
    limits : dict, optional
        Trapezoid ``(a, b, c, d)`` of any of the features ``"rhohv"``,
        ``"zdr"`` (dB), ``"zdr_texture"`` (dB), ``"phidp_texture"``
        (degrees), ``"dbz_texture"`` (dB²), ``"spin"`` (%): membership 0
        below ``a`` and above ``d``, 1 between ``b`` and ``c``. Defaults in
        :data:`radarx.retrieve.qc.DEFAULT_LIMITS`.
    weights : dict, optional
        Weights of the features (0 drops one). Defaults in
        :data:`radarx.retrieve.qc.DEFAULT_WEIGHTS`.
    spin_threshold : float, optional
        Smallest reflectivity jump (dB) counted in the spin feature.
        Default 2.
    min_size : int, optional
        Connected meteorological regions with fewer gates are speckle.
        Default 10; 1 or less disables the filter.
    snr_min : float, optional
        Gates with an SNR below this (dB) have no echo, if an SNR field is
        used. Default 3.
    nodata : {"auto", "nexrad"}, dict or None, optional
        No-data floors: values at or below them are flags, not data.
        ``"auto"`` (default) uses the NEXRAD Level II codes as decoded by
        xradar (:data:`radarx.retrieve.qc.NEXRAD_NODATA`) for NEXRAD data
        and none otherwise; a dict sets floors for ``"dbzh"``, ``"zdr"``,
        ``"rhohv"`` and ``"phidp"``. NaN is always missing.
    split_cuts : bool, optional
        In a volume, sweeps without polarimetric fields (the Doppler cuts of
        NEXRAD split cuts) take the class of the nearest gate of the
        polarimetric sweep at the same elevation where both have echo.
        Default True.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled
        kernel and falls back to NumPy.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        For a sweep, a Dataset on the input coordinates with ``ECHO_CLASS``
        (int8: 0 no echo, 1 meteorological, 2 non-meteorological, 3
        speckle), ``METEO_SCORE`` (float32, NaN without echo) and
        ``METEO_MASK`` (True for meteorological echo). For a volume, a
        DataTree with one such node per sweep that has reflectivity, and the
        root of the input.

    Raises
    ------
    KeyError
        If a requested field is missing, or no sweep has reflectivity.
    ValueError
        For invalid options or non-uniform range gates.
    ImportError
        If ``engine="compiled"`` and the compiled kernel is not available.

    References
    ----------
    Gourley, J. J., P. Tabary, and J. Parent du Chatelet, 2007: A fuzzy
    logic algorithm for the separation of precipitating from
    nonprecipitating echoes using polarimetric radar observations. *J.
    Atmos. Oceanic Technol.*, **24** (8), 1439-1451,
    https://doi.org/10.1175/JTECH2035.1

    Krause, J. M., 2016: A simple algorithm to discriminate between
    meteorological and nonmeteorological radar echoes. *J. Atmos. Oceanic
    Technol.*, **33** (9), 1875-1885, https://doi.org/10.1175/JTECH-D-15-0239.1

    Park, H. S., A. V. Ryzhkov, D. S. Zrnić, and K.-E. Kim, 2009: The
    hydrometeor classification algorithm for the polarimetric WSR-88D:
    Description and application to an MCS. *Wea. Forecasting*, **24** (3),
    730-748, https://doi.org/10.1175/2008WAF2222205.1

    Steiner, M., and J. A. Smith, 2002: Use of three-dimensional
    reflectivity structure for automated detection and removal of
    nonprecipitating echoes in radar data. *J. Atmos. Oceanic Technol.*,
    **19** (5), 673-686,
    https://doi.org/10.1175/1520-0426(2002)019<0673:UOTDRS>2.0.CO;2

    Tang, L., J. Zhang, C. Langston, J. Krause, K. Howard, and V.
    Lakshmanan, 2014: A physically based precipitation-nonprecipitation
    radar echo classifier using polarimetric and environmental data in a
    real-time national system. *Wea. Forecasting*, **29** (5), 1106-1119,
    https://doi.org/10.1175/WAF-D-13-00072.1

    Examples
    --------
    >>> qc = radarx.retrieve.echo_mask(dtree)  # doctest: +SKIP
    >>> clean = dtree.radarx.apply_mask(qc)  # doctest: +SKIP
    """
    if not window > 0:
        raise ValueError("window must be positive")
    lim, wts = _options(limits, weights)
    use_compiled = _use_compiled(engine)
    fields = (dbzh, zdr, rhohv, phidp, snr)

    def params(nexrad):
        return {
            "floors": _floors(nodata, nexrad),
            "snr_min": float(snr_min),
            "limits": lim,
            "weights": wts,
            "spin_threshold": float(spin_threshold),
            "threshold": float(threshold),
            "min_size": int(min_size),
        }

    if isinstance(obj, xr.Dataset):
        sweep = _prepare_sweep(obj, fields, window)
        p = params(_is_nexrad(obj))
        score, cls = _run([sweep], p, n_threads, use_compiled)[0]
        return _wrap_sweep(sweep, score, cls, threshold)

    from ..grid.cone import _sweep_names

    names = [
        name
        for name in _sweep_names(obj)
        if _find(obj[name].to_dataset(), dbzh, _DBZH_NAMES, False) is not None
    ]
    if not names:
        raise KeyError("no sweep contains a reflectivity field")
    sweeps = []
    for name in names:
        ds = obj[name].to_dataset(inherit=False)
        sweep = _prepare_sweep(ds, _fields_for(ds, fields), window)
        sweep["name"] = name
        sweeps.append(sweep)
    p = params(_is_nexrad(sweeps[0]["ds"], obj.root.attrs))
    results = _run(sweeps, p, n_threads, use_compiled)
    comments = [None] * len(sweeps)
    if split_cuts:
        results, comments = _borrow_split_cuts(sweeps, results)
    nodes = {"/": obj.root.to_dataset(inherit=False)}
    for sweep, (score, cls), comment in zip(sweeps, results, comments):
        nodes[sweep["name"]] = _wrap_sweep(sweep, score, cls, threshold, comment)
    return xr.DataTree.from_dict(nodes)


def _keep_array(mask, da, name):
    """Boolean (ray, range) array of ``mask`` in the dims order of ``da``."""
    if isinstance(mask, xr.Dataset):
        if "METEO_MASK" not in mask:
            raise KeyError(f"no METEO_MASK for {name!r}; pass the output of echo_mask")
        mask = mask["METEO_MASK"]
    if not isinstance(mask, xr.DataArray):
        raise TypeError("mask must be an xarray DataArray, Dataset or DataTree")
    dims = [d for d in da.dims if d in mask.dims]
    if mask.ndim != 2 or len(dims) != 2:
        raise ValueError(f"the mask does not match the dimensions of {name!r}")
    keep = mask.transpose(*dims)
    if keep.shape != tuple(da.sizes[d] for d in dims):
        raise ValueError(f"the mask does not match the shape of {name!r}")
    return keep.values.astype(bool)


def _mask_fields(ds, mask, fields, floors):
    """Masked copies of ``fields`` of a sweep, as a dict of DataArrays."""
    ray_dims = [d for d in ("azimuth", "elevation") if d in ds.dims]
    if fields is None:
        fields = [
            name
            for name, da in ds.data_vars.items()
            if da.ndim == 2
            and "range" in da.dims
            and any(d in da.dims for d in ray_dims)
            and np.issubdtype(da.dtype, np.floating)
            and name not in ("METEO_SCORE",)
        ]
    elif isinstance(fields, str):
        fields = [fields]
    out = {}
    for name in fields:
        if name not in ds:
            raise KeyError(f"{name!r} is not in the dataset")
        da = ds[name]
        keep = _keep_array(mask, da, name)
        floor = floors.get(name)
        if floor is not None:
            with np.errstate(invalid="ignore"):
                keep = keep & (da.values > floor)
        out[name] = da.where(xr.DataArray(keep, dims=da.dims)).assign_attrs(da.attrs)
    return out


def apply_mask(obj, mask=None, fields=None, *, nodata="auto", **kwargs):
    """
    Set non-meteorological gates to NaN.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataTree
        A sweep or a volume with ``sweep_*`` groups.
    mask : xarray.Dataset, xarray.DataTree or xarray.DataArray, optional
        The output of :func:`echo_mask` for ``obj`` (or its boolean
        ``METEO_MASK``, True for gates to keep). By default it is computed
        with :func:`echo_mask` and ``**kwargs``.
    fields : str or list of str, optional
        Fields to mask. Default: all floating point fields on
        ``(azimuth, range)`` (or ``(elevation, range)``). In a volume,
        sweeps without a field or without a mask are left unchanged.
    nodata : {"auto", "nexrad"} or None, optional
        ``"auto"`` (default) also masks the no-data codes of NEXRAD Level II
        fields decoded by xradar in NEXRAD data (``DBZH`` <= -32,
        ``ZDR`` <= -12.9, ``RHOHV`` <= 0.21, ``PHIDP`` < 0,
        ``VRADH`` <= -63.9), so that every masked field holds NaN at gates
        without data; ``"nexrad"`` always does, None never does. Also passed
        to :func:`echo_mask`.
    **kwargs
        Options of :func:`echo_mask` when ``mask`` is not given.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        A copy of ``obj`` with the masked fields (attributes kept).

    Examples
    --------
    >>> clean = radarx.retrieve.apply_mask(dtree, fields=["DBZH", "ZDR"])  # doctest: +SKIP
    >>> clean = dtree.radarx.apply_mask()  # doctest: +SKIP
    """
    if nodata not in ("auto", "nexrad", None):
        raise ValueError(f"nodata must be 'auto', 'nexrad' or None, not {nodata!r}")
    if mask is None:
        mask = echo_mask(obj, nodata=nodata, **kwargs)
    elif kwargs:
        raise TypeError("echo_mask options are only used when mask is not given")

    def floors(ds, root_attrs=None):
        nexrad = nodata == "nexrad" or (nodata == "auto" and _is_nexrad(ds, root_attrs))
        return _NEXRAD_FIELD_NODATA if nexrad else {}

    if isinstance(obj, xr.Dataset):
        if isinstance(mask, xr.DataTree):
            raise TypeError("a Dataset needs a Dataset or DataArray mask")
        return obj.assign(_mask_fields(obj, mask, fields, floors(obj)))

    if not isinstance(mask, xr.DataTree):
        raise TypeError("a DataTree needs the DataTree output of echo_mask")
    out = obj.copy()
    for name in mask.children:
        if name not in obj.children:
            continue
        ds = obj[name].to_dataset(inherit=False)
        wanted = fields
        if fields is not None:
            wanted = [fields] if isinstance(fields, str) else fields
            wanted = [f for f in wanted if f in ds]
        masked = _mask_fields(
            ds, mask[name].to_dataset(), wanted, floors(ds, obj.root.attrs)
        )
        for field, da in masked.items():
            out[f"{name}/{field}"] = da.variable
    return out


def _fields_for(ds, fields):
    """Optional fields that a sweep of a volume lacks are left out."""
    return (fields[0],) + tuple(f if f is None or f in ds else None for f in fields[1:])
