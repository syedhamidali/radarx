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

Two methods are available (option ``method`` of :func:`echo_mask`).

``method="radarx"`` (default) is a weighted fuzzy-logic scheme in the manner of
Gourley et al. (2007) [1] and Krause (2016) [2]: local features of the
polarimetric variables and of the reflectivity are mapped to a membership in
[0, 1] ("how meteorological"), and the weighted mean of the memberships is
compared with a threshold. It is not an implementation of either paper.
Only the form of the aggregation, :math:`A = \\sum_j W_j P_j / \\sum_j W_j`
(Krause 2016 [2], p. 1876), and the idea that the variability of
:math:`Z_{DR}` and :math:`\\Phi_{DP}` is small in precipitation (both papers)
are taken from them. The membership corners, weights, threshold, window,
smoothing and speckle filter are radarx's own choices; see the Notes of
:func:`echo_mask` for the number-by-number comparison with the two papers.

``method="krause2016"`` is the MetSignal algorithm of Krause (2016) [2]:
the six memberships of its Fig. 2, the weights of its Table 1, standard
deviations over nine range bins, the threshold of 0.8 (warm season) or 0.7
(cold season) and the post-processing rules of its Sect. 2. The Notes of
:func:`echo_mask` list what the paper specifies and what radarx had to choose.
The method of Gourley et al. (2007) [1] is not implemented, because its
membership functions are density estimates of C-band data of one radar that
the paper shows in a figure only (see the Notes).

For ``method="radarx"``, every feature is computed along the ray over a
window of ``window`` km centred on the gate:

``rhohv``
    Mean copolar correlation coefficient. Precipitation has a mode of
    :math:`\\rho_{hv} = 0.97` in the C-band data of Gourley et al. (2007)
    [1] (p. 1443, Fig. 3a), who suppress the precipitation class below 0.7
    (their Table 1, p. 1446); biological scatterers and clutter have much
    lower values. Values above 1, which occur only at low signal-to-noise
    ratio, are reflected about 1 (1.05 counts as 0.95; radarx choice).
``zdr``
    Mean differential reflectivity. The biological-scatterer class of Park
    et al. (2009) [3] has :math:`Z_{DR}` between 0 and 12 dB, with full
    membership from 2 to 10 dB (their Table 1, p. 733); Tang et al. (2014)
    [5] describe the same behaviour (not checked against the paper). Krause (2016) [2] leaves the mean :math:`Z_{DR}` out
    "because of possible radar miscalibration" (p. 1880); radarx uses it
    with a trapezoid of its own.
``zdr_texture``, ``phidp_texture``
    Standard deviation of :math:`Z_{DR}` and circular standard deviation of
    :math:`\\Phi_{DP}` along the ray. Both are small in precipitation, where
    the scatterers in neighbouring gates are alike, and large for clutter,
    biological echo and noise. Krause (2016) [2] uses the standard
    deviations over nine range bins along the radial (p. 1876); Gourley et
    al. (2007) [1] use the root-mean-square difference to the 3 x 3
    neighbouring gates instead (their Eq. 1, p. 1441).
``dbz_texture``
    Mean squared difference of the reflectivity of adjacent gates
    (:math:`\\mathrm{dB}^2`), large for ground clutter and anomalous
    propagation. After the reflectivity texture of Steiner and Smith (2002)
    [4] (not checked against the paper). The corners and the weight are
    radarx's own.
``spin``
    Share (%) of gates at which the reflectivity gradient along the ray
    changes sign with jumps of at least ``spin_threshold`` dB on both sides,
    the "spin change" of Steiner and Smith (2002) [4], large for clutter
    (definition and 2 dB threshold not checked against the paper). The
    corners and the weight are radarx's own.

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
ratio below ``min_snr`` (if an SNR field is given), and the no-data codes
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
.. [1] Gourley, J. J., P. Tabary, and J. Parent du Chatelet, 2007: A fuzzy
   logic algorithm for the separation of precipitating from nonprecipitating
   echoes using polarimetric radar observations. *J. Atmos. Oceanic
   Technol.*, **24** (8), 1439-1451, https://doi.org/10.1175/JTECH2035.1
.. [2] Krause, J. M., 2016: A simple algorithm to discriminate between
   meteorological and nonmeteorological radar echoes. *J. Atmos. Oceanic
   Technol.*, **33** (9), 1875-1885,
   https://doi.org/10.1175/JTECH-D-15-0239.1
.. [3] Park, H. S., A. V. Ryzhkov, D. S. Zrnić, and K.-E. Kim, 2009: The
   hydrometeor classification algorithm for the polarimetric WSR-88D:
   Description and application to an MCS. *Wea. Forecasting*, **24** (3),
   730-748, https://doi.org/10.1175/2008WAF2222205.1
.. [4] Steiner, M., and J. A. Smith, 2002: Use of three-dimensional
   reflectivity structure for automated detection and removal of
   nonprecipitating echoes in radar data. *J. Atmos. Oceanic Technol.*,
   **19** (5), 673-686,
   https://doi.org/10.1175/1520-0426(2002)019<0673:UOTDRS>2.0.CO;2
.. [5] Tang, L., J. Zhang, C. Langston, J. Krause, K. Howard, and V.
   Lakshmanan, 2014: A physically based precipitation-nonprecipitation radar
   echo classifier using polarimetric and environmental data in a real-time
   national system. *Wea. Forecasting*, **29** (5), 1106-1119,
   https://doi.org/10.1175/WAF-D-13-00072.1

.. autosummary::
   :nosignatures:
   :toctree: generated/

   echo_mask
   apply_mask
"""

from __future__ import annotations

import functools

__all__ = ["echo_mask", "apply_mask"]

import numpy as np
import xarray as xr

from .._compat import renamed_arguments
from .._fields import VELOCITY, align_like, as_sweep
from .._provenance import provenance
from .._sweepwise import (
    Output,
    Sweep,
    apply_sweeps_many,
    dask_support,
    ray_coordinate,
    ray_dimension,
)
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

#: Values of the ``method`` option of :func:`echo_mask`.
METHODS = ("radarx", "krause2016")

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

#: Default trapezoid membership ``(a, b, c, d)`` of every feature. These are
#: radarx's own choices, not values of Gourley et al. (2007) or Krause (2016):
#: the nearest paper values are Krause (2016), Fig. 2 (p. 1877): rhohv 0 at
#: 0.75 and 1 at 0.90, SD(ZDR) 1 up to 1 dB and 0 at 2 dB, SD(PhiDP) 1 up to
#: 10 deg and 0 at 20 deg. Only the SD(ZDR) full-membership limit (1 dB)
#: coincides. The ``zdr`` trapezoid has no counterpart in either paper (Krause
#: excludes the mean ZDR, p. 1880); ``dbz_texture`` and ``spin`` follow the
#: features of Steiner and Smith (2002) with corners of radarx (not checked
#: against the paper).
DEFAULT_LIMITS = {
    "rhohv": (0.80, 0.95, _INF, _INF),
    "zdr": (-4.0, -2.0, 3.0, 6.0),
    "zdr_texture": (-_INF, -_INF, 1.0, 2.5),
    "phidp_texture": (-_INF, -_INF, 12.0, 30.0),
    "dbz_texture": (-_INF, -_INF, 30.0, 70.0),
    "spin": (-_INF, -_INF, 30.0, 60.0),
}

#: Default weight of every feature: radarx's own choice. Krause (2016), Table 1
#: (p. 1876), uses 1.0 for Zh, rhohv, V and SD(rhohv) and 2.0 for SD(PhiDP) and
#: SD(ZDR); Gourley et al. (2007), Eqs. 3-4 (p. 1444), derive weights from the
#: overlap areas 0.243, 0.203 and 0.081 of rhohv, texture(ZDR) and
#: texture(PhiDP), i.e. 1 : 1.2 : 3.0. The 0.25 of the two Steiner and Smith
#: (2002) features is not from any paper.
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
#: The no-data code of the radial velocity of NEXRAD Level II data (method
#: ``"krause2016"`` only).
_NEXRAD_VEL_FLOOR = -63.9


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


def _wrap180(d):
    """Differences wrapped into [-180, 180) degrees."""
    return d - 360.0 * np.floor(d / 360.0 + 0.5)


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
    if ok is not None and p.get("phidp_detrended", False):
        # noise std = rms(second difference of the wrapped phase) / sqrt(6)
        okq = np.zeros(phi.shape, dtype=bool)
        q = np.zeros(phi.shape)
        if ng >= 3:
            x = np.where(ok, phi, 0.0)
            d2 = _wrap180(
                _wrap180(x[:, 2:] - x[:, 1:-1]) - _wrap180(x[:, 1:-1] - x[:, :-2])
            )
            okq[:, 1:-1] = ok[:, :-2] & ok[:, 1:-1] & ok[:, 2:]
            q[:, 1:-1] = np.where(okq[:, 1:-1], d2 * d2, 0.0)
        n = _span(_prefix(okq.astype(np.float64)), lo, hi)
        sq = _span(_prefix(q), lo, hi)
        with np.errstate(invalid="ignore", divide="ignore"):
            sd = np.sqrt(sq / n / 6.0)
        add("phidp_texture", sd, n >= 3)
    elif ok is not None:
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
        bool(p.get("phidp_detrended", False)),
        int(n_threads or 0),
    )


# --------------------------------------------------------------------------
# MetSignal of Krause (2016), NumPy reference (same steps as classify_krause)
# --------------------------------------------------------------------------

#: Membership corners ``(a, b, c, d)`` of the MetSignal algorithm, read from
#: Fig. 2 of Krause (2016) (p. 1877). The velocity membership is 1 minus the
#: inverted trapezoid (p. 1876).
KRAUSE_LIMITS = {
    "dbzh": (10.0, 30.0, _INF, _INF),
    "rhohv": (0.75, 0.90, _INF, _INF),
    "vrad": (-1.5, -1.0, 1.0, 1.5),
    "phidp_sd": (-_INF, -_INF, 10.0, 20.0),
    "zdr_sd": (-_INF, -_INF, 1.0, 2.0),
    "rhohv_sd": (-_INF, -_INF, 0.03, 0.04),
}
#: Weights of the MetSignal algorithm, Table 1 of Krause (2016) (p. 1876).
KRAUSE_WEIGHTS = {
    "dbzh": 1.0,
    "rhohv": 1.0,
    "vrad": 1.0,
    "phidp_sd": 2.0,
    "zdr_sd": 2.0,
    "rhohv_sd": 1.0,
}
#: Post-processing of Krause (2016) (pp. 1875-1876): gates with abs(ZDR) above
#: ``zdr_max`` (dB) or rhohv below ``rhohv_min`` are non-meteorological, gates
#: whose reflectivity at 3 km (previous volume) is at least ``override_dbz``
#: are meteorological.
KRAUSE_RULES = {"zdr_max": 4.5, "rhohv_min": 0.65, "override_dbz": 11.0}
#: Thresholds of the aggregation value (p. 1876): warm and cold season.
KRAUSE_THRESHOLDS = {"warm": 0.8, "cold": 0.7}
#: Half length of the window of the standard deviations: nine range bins
#: centred on the gate (p. 1876).
KRAUSE_HALF_WINDOW = 4


def _krause_numpy(sweep, p):
    """Aggregation value and class of one sweep (NumPy reference)."""
    z = sweep["dbzh"]
    nray, ng = z.shape
    floors = p["floors"]
    with np.errstate(invalid="ignore"):
        echo = np.isfinite(z) & (z > floors[0])
        if sweep["snr"] is not None:
            echo &= sweep["snr"] >= p["snr_min"]
    g = np.arange(ng)
    lo = np.broadcast_to(np.maximum(g - KRAUSE_HALF_WINDOW, 0), (nray, ng))
    hi = np.broadcast_to(np.minimum(g + KRAUSE_HALF_WINDOW, ng - 1) + 1, (nray, ng))
    num = np.zeros((nray, ng))
    den = np.zeros((nray, ng))

    def add(name, value, defined):
        w = KRAUSE_WEIGHTS[name]
        m = _trapezoid(np.where(defined, value, 0.0), KRAUSE_LIMITS[name])
        num[:] = num + np.where(defined, w * m, 0.0)
        den[:] = den + np.where(defined, w, 0.0)

    add("dbzh", z, echo)
    rho, zdr, phi, vel = sweep["rhohv"], sweep["zdr"], sweep["phidp"], sweep["vrad"]
    ok_rho = _valid(rho, floors[2], echo)
    ok_zdr = _valid(zdr, floors[1], echo)
    ok_phi = _valid(phi, floors[3], echo)
    ok_vel = _valid(vel, floors[4], echo)
    if ok_rho is not None:
        add("rhohv", rho, ok_rho)
    if ok_vel is not None:
        m = 1.0 - _trapezoid(np.where(ok_vel, vel, 0.0), KRAUSE_LIMITS["vrad"])
        num[:] = num + np.where(ok_vel, KRAUSE_WEIGHTS["vrad"] * m, 0.0)
        den[:] = den + np.where(ok_vel, KRAUSE_WEIGHTS["vrad"], 0.0)
    if ok_phi is not None:
        rad = np.deg2rad(np.where(ok_phi, phi, 0.0))
        n = _span(_prefix(ok_phi.astype(np.float64)), lo, hi)
        c = _span(_prefix(np.where(ok_phi, np.cos(rad), 0.0)), lo, hi)
        s = _span(_prefix(np.where(ok_phi, np.sin(rad), 0.0)), lo, hi)
        with np.errstate(invalid="ignore", divide="ignore"):
            r2 = np.clip((c * c + s * s) / (n * n), 1e-300, 1.0)
            sd = np.sqrt(-np.log(r2)) * (180.0 / np.pi)
        add("phidp_sd", sd, n >= 3)
    for name, x, ok in (("zdr_sd", zdr, ok_zdr), ("rhohv_sd", rho, ok_rho)):
        if ok is None:
            continue
        v = np.where(ok, x, 0.0)
        n = _span(_prefix(ok.astype(np.float64)), lo, hi)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = _span(_prefix(v), lo, hi) / n
            var = _span(_prefix(v * v), lo, hi) / n - mean * mean
        add(name, np.sqrt(np.maximum(var, 0.0)), n >= 3)
    with np.errstate(invalid="ignore", divide="ignore"):
        agg = (num / den).astype(np.float32)
    agg = np.where(echo, agg, np.float32(np.nan))
    meteo = agg >= np.float32(p["threshold"])
    if ok_zdr is not None:
        meteo &= ~(
            ok_zdr & (np.abs(np.where(ok_zdr, zdr, 0.0)) > KRAUSE_RULES["zdr_max"])
        )
    if ok_rho is not None:
        meteo &= ~(ok_rho & (np.where(ok_rho, rho, 1.0) < KRAUSE_RULES["rhohv_min"]))
    if sweep["zh3"] is not None:
        with np.errstate(invalid="ignore"):
            meteo |= np.isfinite(sweep["zh3"]) & (
                sweep["zh3"] >= KRAUSE_RULES["override_dbz"]
            )
    cls = np.where(echo, np.where(meteo, 1, 2), 0).astype(np.int8)
    if p["min_size"] > 1:
        size = _components(cls == 1, sweep["links"])
        cls[(cls == 1) & (size < p["min_size"])] = 3
    return agg, cls


def _krause_compiled(sweeps, p, n_threads):
    """Run the C++ kernel of the MetSignal algorithm on all sweeps at once."""
    return _qc.classify_krause(
        [sw["dbzh"] for sw in sweeps],
        [sw["zdr"] for sw in sweeps],
        [sw["rhohv"] for sw in sweeps],
        [sw["phidp"] for sw in sweeps],
        [sw["vrad"] for sw in sweeps],
        [sw["zh3"] for sw in sweeps],
        [sw["snr"] for sw in sweeps],
        [sw["links"] for sw in sweeps],
        np.asarray(p["floors"], dtype=np.float64),
        float(p["snr_min"]),
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


def _floors(nodata, nexrad, roles=_ROLES):
    """No-data floors of ``roles`` ((dbzh, zdr, rhohv, phidp), plus vrad)."""
    if isinstance(nodata, str):
        if nodata not in ("auto", "nexrad"):
            raise ValueError(
                f"nodata must be 'auto', 'nexrad', None or a dict, not {nodata!r}"
            )
        table = NEXRAD_NODATA if (nodata == "nexrad" or nexrad) else {}
        if table:
            table = {**table, "vrad": _NEXRAD_VEL_FLOOR}
    elif nodata is None:
        table = {}
    elif isinstance(nodata, dict):
        unknown = set(nodata) - set(roles)
        if unknown:
            raise ValueError(f"unknown nodata keys {sorted(unknown)}; use {roles}")
        table = nodata
    else:
        raise ValueError(
            f"nodata must be 'auto', 'nexrad', None or a dict, not {nodata!r}"
        )
    return [float(table.get(role, -_INF)) for role in roles]


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


_ALL_ROLES = ("dbzh", "zdr", "rhohv", "phidp", "snr", "vrad", "zh3")


def _prepare_sweep(ds, fields, window):
    """Field names, the fields on (..., ray, range) and the ray links of a sweep."""
    dbzh, zdr, rhohv, phidp, snr = fields[:5]
    names = {
        "dbzh": _find(ds, dbzh, _DBZH_NAMES, True),
        "zdr": _find(ds, zdr, _ZDR_NAMES, zdr is not None),
        "rhohv": _find(ds, rhohv, _RHOHV_NAMES, rhohv is not None),
        "phidp": _find(ds, phidp, _PHIDP_NAMES, phidp is not None),
        "snr": _find(ds, snr, _SNR_NAMES, snr is not None),
    }
    if len(fields) > 5:  # method "krause2016": radial velocity, Zh at 3 km
        vrad, zh3 = fields[5:]
        names["vrad"] = _find(ds, vrad, VELOCITY, vrad is not None)
        names["zh3"] = _find(ds, zh3, (), zh3 is not None)
    da = ds[names["dbzh"]]
    ray_dim = ray_dimension(da)
    dr = _gate_spacing(ds["range"].values)
    if "azimuth" in ds.coords and ray_dim == "azimuth":
        links = _ray_links(ray_coordinate(ds, ray_dim))
    else:
        links = np.ones(da.sizes[ray_dim], dtype=np.uint8)
        links[-1] = 0  # RHI or unknown geometry: no wrap
    return {
        "ds": ds,
        "names": names,
        "ray_dim": ray_dim,
        "fields": [
            None if names.get(role) is None else ds[names[role]] for role in _ALL_ROLES
        ],
        "links": np.ascontiguousarray(links, dtype=np.uint8),
        "h": max(1, round(0.5 * float(window) / dr)),
    }


def _polarimetric(sweep):
    return any(sweep["names"].get(r) is not None for r in ("zdr", "rhohv", "phidp"))


def _wrap_sweep(sweep, score, cls, threshold, comment=None, method="radarx"):
    """Score, class and mask on the input sweep coordinates."""
    da = sweep["ds"][sweep["names"]["dbzh"]]
    dims = da.dims
    score, cls = score.data, cls.data
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
                    "comment": (
                        "weighted mean fuzzy membership, 3 x 3 average; "
                        f"meteorological at >= {threshold}"
                        if method == "radarx"
                        else "aggregation value A of Krause (2016), the weighted "
                        "mean of the memberships; meteorological at "
                        f"A >= {threshold} after the post-processing rules"
                    ),
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


def _classify_batch(sweeps, params, n_threads, use_compiled):
    if params.get("method") == "krause2016":
        if use_compiled:
            scores, classes = _krause_compiled(sweeps, params, n_threads)
            return list(zip(scores, classes))
        return [_krause_numpy(sw, params) for sw in sweeps]
    if use_compiled:
        scores, classes = _classify_compiled(sweeps, params, n_threads)
        return list(zip(scores, classes))
    return [_classify_numpy(sw, params) for sw in sweeps]


def _core(batch, params, n_threads, use_compiled):
    """Score and class of sweeps given as ``((links, h), fields)``."""
    sweeps = []
    for (links, h), arrays in batch:
        sweep = {
            role: None if a is None else np.ascontiguousarray(a, dtype=np.float64)
            for role, a in zip(_ALL_ROLES, arrays)
        }
        sweeps.append({**sweep, "links": links, "h": h})
    return _classify_batch(sweeps, params, n_threads, use_compiled)


def _run(sweeps, params, n_threads, use_compiled):
    """Score and class (DataArrays) of every prepared sweep."""
    core = functools.partial(
        _core, params=params, n_threads=n_threads, use_compiled=use_compiled
    )
    return apply_sweeps_many(
        core,
        [Sweep(sw["fields"], context=(sw["links"], sw["h"])) for sw in sweeps],
        (Output(np.float32), Output(np.int8)),
    )


def _borrow_split_cuts(sweeps, results, tolerance=0.1):
    """Class of Doppler cuts from the polarimetric cut at the same elevation.

    The classes of the two cuts are combined, so both are computed (dask
    arrays become NumPy arrays); cuts with extra dimensions are left as they
    are.
    """
    angles = [_fixed_angle(sw["ds"]) for sw in sweeps]
    comments = [None] * len(sweeps)
    for i, sw in enumerate(sweeps):
        if (
            _polarimetric(sw)
            or angles[i] is None
            or sw["ray_dim"] != "azimuth"
            or results[i][1].ndim != 2
        ):
            continue
        cands = [
            j
            for j, other in enumerate(sweeps)
            if _polarimetric(other)
            and angles[j] is not None
            and other["ray_dim"] == "azimuth"
            and results[j][1].ndim == 2
            and abs(angles[j] - angles[i]) <= tolerance
        ]
        if not cands:
            continue
        j = min(cands, key=lambda k: abs(k - i))
        iray, igate = _sweep_mapping(sweeps[j]["ds"], sw["ds"])
        src_score, src_cls = (_ray_range(r, sweeps[j]) for r in results[j])
        score, cls = (_ray_range(r, sw) for r in results[i])
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
        results[i] = tuple(
            r.copy(data=v if r.dims[0] == sw["ray_dim"] else v.T)
            for r, v in zip(results[i], (score, cls))
        )
        comments[i] = sweeps[j]["name"]
    return results, comments


def _ray_range(da, sweep):
    """The values of a result on (ray, range) as a NumPy array."""
    return da.transpose(sweep["ray_dim"], "range").values


def _fixed_angle(ds):
    for name in ("sweep_fixed_angle", "fixed_angle"):
        if name in ds.variables:
            return float(np.asarray(ds[name].values).ravel()[0])
    if "elevation" in ds.coords:
        return float(np.nanmedian(ds["elevation"].values))
    return None  # pragma: no cover - xradar sweeps have one of the above


@dask_support(
    "lazy",
    "With ``split_cuts=True`` a Doppler cut of a volume takes its classes from the polarimetric cut at the same elevation, so these two cuts are computed.",
)
@renamed_arguments(dbzh="dbz", vrad="field", snr_min="min_snr")
@provenance(
    "Fuzzy-logic echo score in the manner of Gourley 2007 and Krause 2016, own memberships "
    "and weights (method='krause2016': the MetSignal algorithm of Krause 2016)"
)
def echo_mask(
    obj,
    dbz=None,
    zdr=None,
    rhohv=None,
    phidp=None,
    snr=None,
    *,
    method="radarx",
    field=None,
    zh_3km=None,
    window=None,
    threshold=None,
    limits=None,
    weights=None,
    spin_threshold=2.0,
    min_size=None,
    min_snr=3.0,
    phidp_texture="circular",
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
        A sweep on ``(azimuth, range)`` (or ``(elevation, range)``), a single
        sweep node (``dtree["sweep_0"]``), or a volume with ``sweep_*`` groups,
        e.g. from xradar. Range gates must be uniformly spaced; ``range`` is
        in metres.
    dbz, zdr, rhohv, phidp, snr : str, optional
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
    method : {"radarx", "krause2016"}, optional
        ``"radarx"`` (default) is the scheme described above, in the manner of
        Gourley et al. (2007) [1] and Krause (2016) [2] with memberships and
        weights of radarx. ``"krause2016"`` is the MetSignal algorithm of
        Krause (2016) [2] with the memberships of its Fig. 2, the weights of
        its Table 1, nine-gate standard deviations and its post-processing
        rules (see Notes). With ``"krause2016"``, ``limits``, ``weights`` and
        ``window`` are not accepted, ``spin_threshold`` and
        ``phidp_texture`` are not used, and the score ``METEO_SCORE`` is the
        aggregation value A of the paper, not averaged over 3 x 3 gates.
    field : str, optional
        Radial velocity field (m/s) of ``method="krause2016"``. By default the
        first of ``VRADH``, ``VRAD``, ``VRADV``, ``VEL``, ``VELOCITY``,
        ``velocity``, ``radial_velocity`` that a sweep holds. Where it is
        missing, as in the surveillance cut of a NEXRAD split cut, the
        velocity term is left out of the weighted mean. Not valid for
        ``method="radarx"``, which does not use velocity.
    zh_3km : str, optional
        Name of a field of the sweeps with the reflectivity (dBZ) at 3 km
        height of the previous volume scan on the same gates, for the
        override rule of ``method="krause2016"`` (gates with at least 11 dBZ
        are meteorological). Krause (2016) builds it as a constant-altitude
        PPI from the previous volume (p. 1882); radarx does not compute it.
        Not valid for ``method="radarx"``.
    window : float, optional
        Length (km) of the window along the ray for the features. Default 1.5
        (radarx choice; Krause 2016 [2] uses nine range bins, p. 1876).
        Only for ``method="radarx"``; ``method="krause2016"`` always uses
        nine range bins.
    threshold : float, optional
        Gates whose score are at least this are meteorological. Default for
        ``method="radarx"``: 0.6 (the weighted mean membership averaged over
        3 x 3 gates), radarx's own choice, lower than the 0.8 (warm season)
        and 0.7 (cold season) of Krause (2016) [2] (p. 1876), whose
        memberships and weights differ (see Notes). Default for
        ``method="krause2016"``: 0.8, the warm-season value; pass 0.7 for the
        cold season.
    limits : dict, optional
        Trapezoid ``(a, b, c, d)`` of any of the features ``"rhohv"``,
        ``"zdr"`` (dB), ``"zdr_texture"`` (dB), ``"phidp_texture"``
        (degrees), ``"dbz_texture"`` (dB²), ``"spin"`` (%): membership 0
        below ``a`` and above ``d``, 1 between ``b`` and ``c``. Defaults in
        :data:`radarx.retrieve.qc.DEFAULT_LIMITS` (radarx's own, see Notes).
    weights : dict, optional
        Weights of the features (0 drops one). Defaults in
        :data:`radarx.retrieve.qc.DEFAULT_WEIGHTS` (radarx's own, see Notes).
    spin_threshold : float, optional
        Smallest reflectivity jump (dB) counted in the spin feature.
        Default 2 (radarx choice; not checked against Steiner and Smith
        2002 [4]).
    min_size : int, optional
        Connected meteorological regions with fewer gates are speckle.
        Default 10 for ``method="radarx"`` (radarx choice, not from the cited
        papers) and 1 for ``method="krause2016"``, which has no speckle
        filter; 1 or less disables the filter.
    phidp_texture : {"circular", "detrended"}, optional
        ``"circular"`` (default) is the circular standard deviation of the
        phase along the ray, which includes the rain ramp (about 0.58
        degrees per degree/km of KDP over a 2 km window). ``"detrended"``
        uses the noise standard deviation from the second differences of the
        wrapped phase (their rms divided by the square root of 6), which is
        zero for a ramp of any slope; see ``texture_method`` of
        :func:`radarx.retrieve.estimate_kdp`. On synthetic rain with KDP 5 to
        30 degrees/km both classify all gates as meteorological. Both rules
        are radarx choices.
    min_snr : float, optional
        Gates with an SNR below this (dB) have no echo, if an SNR field is
        used. Default 3 (radarx choice).
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

    Notes
    -----
    The scheme is not an implementation of Gourley et al. (2007) [1] or
    Krause (2016) [2]. Only the form of the weighted-mean aggregation
    (Krause 2016 [2], p. 1876) and the use of the variability of
    :math:`Z_{DR}` and :math:`\\Phi_{DP}` as indicators of non-meteorological
    echo come from them. Which numbers are theirs and which are not (the
    paper values are read from the figures and tables on the pages given;
    ``-`` means not used):

    .. list-table::
       :header-rows: 1
       :widths: 14 20 9 33 24

       * - feature
         - radarx ``DEFAULT_LIMITS`` ``(a, b, c, d)``
         - radarx weight
         - Krause (2016) [2], Fig. 2 (p. 1877), weights in Table 1 (p. 1876)
         - Gourley et al. (2007) [1]
       * - ``rhohv``
         - 0.80, 0.95, inf, inf
         - 1.0
         - 0 at 0.75, 1 at 0.90; weight 1.0
         - Gaussian-kernel density estimate (Fig. 3, Eq. 2), no trapezoid
       * - ``zdr``
         - -4, -2, 3, 6 dB
         - 1.0
         - not used (mean :math:`Z_{DR}` excluded, p. 1880)
         - not used
       * - ``zdr_texture``
         - 1 up to 1.0 dB, 0 at 2.5 dB
         - 1.0
         - SD(ZDR): 1 up to 1 dB, 0 at 2 dB; weight 2.0
         - texture of ZDR (Eq. 1), density estimate
       * - ``phidp_texture``
         - 1 up to 12 deg, 0 at 30 deg
         - 1.0
         - SD(PhiDP): 1 up to 10 deg, 0 at 20 deg; weight 2.0
         - texture of PhiDP (Eq. 1), density estimate
       * - ``dbz_texture``
         - 1 up to 30 dB2, 0 at 70 dB2
         - 0.25
         - not used
         - not used
       * - ``spin``
         - 1 up to 30 %, 0 at 60 %
         - 0.25
         - not used
         - not used
       * - not computed by radarx
         - -
         - -
         - Z_h: 0 at 10, 1 at 30 dBZ (weight 1.0); V: 1 for abs(V) >= 1.5 m/s,
           0 for abs(V) <= 1.0 m/s (weight 1.0); SD(rhohv): 1 up to 0.03, 0 at
           0.04 (weight 1.0)
         - pulse-to-pulse variability of Z and radial velocity only as
           suppression thresholds (Table 1, p. 1446)
       * - threshold
         - 0.6 (on the 3 x 3 averaged score)
         - -
         - 0.8 (warm season), 0.7 (cold season), p. 1876
         - none: the class with the largest aggregation value wins (p. 1445)

    The only corner shared with a paper is the full membership of SD(ZDR)
    up to 1 dB (Krause 2016 [2]). All other corners, all weights, the
    threshold, the 1.5 km window (Krause: nine range bins), the
    3 x 3 averaging of the score, the ``min_size`` speckle filter, ``min_snr``
    (3 dB), ``spin_threshold`` (2 dB), the NEXRAD no-data floors and the
    reflection of rhohv above 1 are radarx's own choices and are not taken
    from these papers. They have not been tuned against the published
    algorithms or a labelled data set.

    Behavioural differences of ``method="radarx"`` from the two papers. They
    are documented and not changed, so that existing results keep their
    numbers; ``method="krause2016"`` has none of the differences with Krause
    (2016) listed below except the choices named under "Method
    ``krause2016``":

    - Weights. All polarimetric features have weight 1.0, the same as
      rhohv. Krause (2016) gives SD(PhiDP) and SD(ZDR) twice the weight of
      rhohv because they discriminate best (Table 1, p. 1876), and Gourley
      et al. (2007) derive weights from the overlap areas 0.243 (rhohv),
      0.203 (texture of ZDR) and 0.081 (texture of PhiDP) that make the
      PhiDP texture count three times rhohv (Eq. 3 and 4, p. 1444).
    - *Threshold.* 0.6 is more permissive than either of the 0.8 and 0.7 of
      Krause (2016), who needed 0.8 to limit false detections owing to
      birds (p. 1882). Gourley et al. (2007) do not use a threshold at all.
    - *Missing rules.* Krause's post-processing rules are not applied: gates
      with :math:`|Z_{DR}| > 4.5` dB or :math:`\\rho_{hv} < 0.65` are
      non-meteorological, and gates with :math:`Z_h > 11` dBZ at 3 km height
      in the previous volume are meteorological (pp. 1875-1876). Radial
      velocity, :math:`Z_h` and SD(:math:`\\rho_{hv}`) are not features, so
      stationary clutter is not penalised by its velocity. A mean
      :math:`Z_{DR}` of 5 dB still has a membership of 0.33 here.
    - *Texture and aggregation of Gourley et al.* The texture is computed
      along the ray over ``window`` instead of as the root-mean-square
      difference to the 3 x 3 neighbours with the range correction of their
      Eq. 1; the memberships are trapezoids instead of density estimates;
      there is one meteorological score, not the maximum over the classes
      precipitation, ground clutter and clear air (their Eq. 3); their
      suppression thresholds (Table 1, p. 1446) and their despeckling of
      precipitation pixels on a 1 km grid (p. 1446) have no counterpart.

    Method ``krause2016``. From the paper (Krause 2016 [2], Sect. 2, Table 1,
    Fig. 2, Sect. 4): the inputs :math:`Z_h`, :math:`\\rho_{hv}`, radial
    velocity V, SD(:math:`\\Phi_{DP}`), SD(:math:`Z_{DR}`) and
    SD(:math:`\\rho_{hv}`), the standard deviations over nine range bins
    centred on the gate; the weights 1, 1, 1, 2, 2, 1; the single-sided
    trapezoid memberships :math:`Z_h` 0 at 10 and 1 at 30 dBZ,
    :math:`\\rho_{hv}` 0 at 0.75 and 1 at 0.90, SD(:math:`\\Phi_{DP}`) 1 up to
    10 and 0 at 20 degrees, SD(:math:`Z_{DR}`) 1 up to 1 and 0 at 2 dB,
    SD(:math:`\\rho_{hv}`) 1 up to 0.03 and 0 at 0.04, and the velocity
    membership 1 minus the inverted trapezoid that is 1 between -1.0 and
    1.0 m/s and 0 beyond :math:`\\pm` 1.5 m/s (the corners are read from the
    dashed lines of Fig. 2 against its axes, the text gives none); the
    aggregation
    :math:`A = \\sum_j W_j P_j / \\sum_j W_j` compared with 0.8 (warm season)
    or 0.7 (cold season); the mean :math:`Z_{DR}` is not a term; then
    gates with :math:`|Z_{DR}| > 4.5` dB or :math:`\\rho_{hv} < 0.65` are
    non-meteorological, and gates whose :math:`Z_h` at 3 km height in the
    previous volume exceeds 11 dBZ are meteorological. The paper does not
    state, and radarx chooses: the standard deviation is the population
    standard deviation of the gates with echo and data in the window (the
    circular standard deviation for :math:`\\Phi_{DP}`, which equals the
    ordinary one for small spread and is safe at the wrap of the phase), at
    least three such gates are needed (else the term is left out), the window
    is cut at the ends of the ray, a field that is missing at a gate drops
    its term from the weighted mean, the comparisons are ``A >= threshold``
    and ``Zh_3km >= 11``, the override is applied after the two rules, the
    rules use the value of the gate itself, and no speckle filter is applied
    (``min_size`` defaults to 1). The paper gives neither the gate spacing of
    its data nor how the 3 km field is built beyond "the elevation closest to
    the constant altitude of 3 km" of the previous volume, so ``zh_3km`` must
    be built by the caller. The standard deviations depend on the gate
    spacing: over nine gates of 250 m a phase ramp of rain alone gives
    SD(:math:`\\Phi_{DP}`) of 1.3 degrees per degree/km of KDP, so that the
    membership starts to fall at a KDP of 7.7 degrees/km. A volume of NEXRAD data has no velocity in the
    surveillance cut of a split cut, so that cut is classified without the
    velocity term (weights 7 instead of 8).

    Gourley et al. (2007) [1] is not implemented as a method. Its three-class
    classifier rests on membership functions that are Gaussian-kernel density
    estimates of C-band data of one radar (Trappes, Fig. 3 and Eq. 2)
    published as plots only; the text
    gives the modes (:math:`\\rho_{hv}` 0.97, 0.92 and 0.42 for precipitation,
    ground clutter and clear air; texture of :math:`Z_{DR}` 0.4, 5.2 and 2.5
    dB; texture of :math:`\\Phi_{DP}` 4, 125 and 35 degrees), the overlap
    areas of the weights (0.243, 0.203, 0.081 giving 1 : 1.20 : 3.00 by
    Eq. 4) and one membership example, not the curves. The slope of the
    range correction of the texture (Eq. 1, Fig. 2) and the pulse-to-pulse
    variability of Z of Table 1 are not given either. Curves digitised
    from the figure would be an invention at S band. The method is built to
    be refitted for each radar and wavelength.

    References
    ----------
    .. [1] Gourley, J. J., P. Tabary, and J. Parent du Chatelet, 2007: A
       fuzzy logic algorithm for the separation of precipitating from
       nonprecipitating echoes using polarimetric radar observations. *J.
       Atmos. Oceanic Technol.*, **24** (8), 1439-1451,
       https://doi.org/10.1175/JTECH2035.1
    .. [2] Krause, J. M., 2016: A simple algorithm to discriminate between
       meteorological and nonmeteorological radar echoes. *J. Atmos. Oceanic
       Technol.*, **33** (9), 1875-1885,
       https://doi.org/10.1175/JTECH-D-15-0239.1
    .. [3] Park, H. S., A. V. Ryzhkov, D. S. Zrnić, and K.-E. Kim, 2009: The
       hydrometeor classification algorithm for the polarimetric WSR-88D:
       Description and application to an MCS. *Wea. Forecasting*, **24** (3),
       730-748, https://doi.org/10.1175/2008WAF2222205.1
    .. [4] Steiner, M., and J. A. Smith, 2002: Use of three-dimensional
       reflectivity structure for automated detection and removal of
       nonprecipitating echoes in radar data. *J. Atmos. Oceanic Technol.*,
       **19** (5), 673-686,
       https://doi.org/10.1175/1520-0426(2002)019<0673:UOTDRS>2.0.CO;2
    .. [5] Tang, L., J. Zhang, C. Langston, J. Krause, K. Howard, and V.
       Lakshmanan, 2014: A physically based precipitation-nonprecipitation
       radar echo classifier using polarimetric and environmental data in a
       real-time national system. *Wea. Forecasting*, **29** (5), 1106-1119,
       https://doi.org/10.1175/WAF-D-13-00072.1

    Examples
    --------
    >>> qc = radarx.retrieve.echo_mask(dtree)  # doctest: +SKIP
    >>> clean = dtree.radarx.apply_mask(qc)  # doctest: +SKIP

    The MetSignal algorithm of Krause (2016), with the reflectivity at 3 km of
    the previous volume as a field ``ZH3KM`` of the sweep:

    >>> qc = radarx.retrieve.echo_mask(
    ...     sweep, method="krause2016", zh_3km="ZH3KM"
    ... )  # doctest: +SKIP
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, not {method!r}")
    krause = method == "krause2016"
    if krause:
        taken = {"limits": limits, "weights": weights, "window": window}
        given = [k for k, v in taken.items() if v is not None]
        if given:
            raise ValueError(
                f"method='krause2016' uses the membership functions, weights and "
                f"nine-gate window of the paper; {given} are options of "
                "method='radarx'"
            )
    elif field is not None or zh_3km is not None:
        raise ValueError("vrad and zh_3km are used by method='krause2016' only")
    if threshold is None:
        threshold = KRAUSE_THRESHOLDS["warm"] if krause else 0.6
    if min_size is None:
        min_size = 1 if krause else 10
    if window is None:
        window = 1.5
    if not window > 0:
        raise ValueError("window must be positive")
    if phidp_texture not in ("circular", "detrended"):
        raise ValueError(
            f"phidp_texture must be 'circular' or 'detrended', not {phidp_texture!r}"
        )
    lim, wts = _options(limits, weights)
    use_compiled = _use_compiled(engine)
    fields = (dbz, zdr, rhohv, phidp, snr)
    if krause:
        fields += (field, zh_3km)
    obj = as_sweep(obj)

    def params(nexrad):
        if krause:
            return {
                "method": method,
                "floors": _floors(nodata, nexrad, _ROLES + ("vrad",)),
                "snr_min": float(min_snr),
                "threshold": float(threshold),
                "min_size": int(min_size),
            }
        return {
            "method": method,
            "floors": _floors(nodata, nexrad),
            "snr_min": float(min_snr),
            "limits": lim,
            "weights": wts,
            "spin_threshold": float(spin_threshold),
            "threshold": float(threshold),
            "min_size": int(min_size),
            "phidp_detrended": phidp_texture == "detrended",
        }

    if isinstance(obj, xr.Dataset):
        sweep = _prepare_sweep(obj, fields, window)
        p = params(_is_nexrad(obj))
        score, cls = _run([sweep], p, n_threads, use_compiled)[0]
        return _wrap_sweep(sweep, score, cls, threshold, method=method)

    from ..grid.cone import _sweep_names

    names = [
        name
        for name in _sweep_names(obj)
        if _find(obj[name].to_dataset(), dbz, _DBZH_NAMES, False) is not None
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
        nodes[sweep["name"]] = _wrap_sweep(
            sweep, score, cls, threshold, comment, method
        )
    return xr.DataTree.from_dict(nodes)


def _keep_array(mask, da, name):
    """Boolean DataArray of ``mask`` on the dimensions of ``da`` (lazy)."""
    if isinstance(mask, xr.Dataset):
        if "METEO_MASK" not in mask:
            raise KeyError(f"no METEO_MASK for {name!r}; pass the output of echo_mask")
        mask = mask["METEO_MASK"]
    if not isinstance(mask, xr.DataArray):
        raise TypeError("mask must be an xarray DataArray, Dataset or DataTree")
    dims = [d for d in da.dims if d in mask.dims]
    if len(dims) != mask.ndim or "range" not in dims or len(dims) < 2:
        raise ValueError(f"the mask does not match the dimensions of {name!r}")
    keep = align_like(mask, da, f"the mask of {name!r}").transpose(*dims)
    if keep.shape != tuple(da.sizes[d] for d in dims):
        raise ValueError(f"the mask does not match the shape of {name!r}")
    return keep.astype(bool)


def _mask_fields(ds, mask, data_vars, floors):
    """Masked copies of ``data_vars`` of a sweep, as a dict of DataArrays."""
    ray_dims = [d for d in ("azimuth", "elevation") if d in ds.dims]
    if data_vars is None:
        data_vars = [
            name
            for name, da in ds.data_vars.items()
            if da.ndim >= 2
            and "range" in da.dims
            and any(d in da.dims for d in ray_dims)
            and np.issubdtype(da.dtype, np.floating)
            and name not in ("METEO_SCORE",)
        ]
    elif isinstance(data_vars, str):
        data_vars = [data_vars]
    out = {}
    for name in data_vars:
        if name not in ds:
            raise KeyError(f"{name!r} is not in the dataset")
        da = ds[name]
        keep = _keep_array(mask, da, name)
        floor = floors.get(name)
        if floor is not None:
            with np.errstate(invalid="ignore"):
                keep = keep & (da > floor)
        out[name] = da.where(keep).assign_attrs(da.attrs)
    return out


@dask_support("lazy")
@renamed_arguments(fields="data_vars")
@provenance("Masking of non-meteorological gates with the radarx echo classification")
def apply_mask(obj, mask=None, data_vars=None, *, nodata="auto", **kwargs):
    """
    Set non-meteorological gates to NaN.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataTree
        A sweep, a single sweep node (``dtree["sweep_0"]``) or a volume with
        ``sweep_*`` groups.
    mask : xarray.Dataset, xarray.DataTree or xarray.DataArray, optional
        The output of :func:`echo_mask` for ``obj`` (or its boolean
        ``METEO_MASK``, True for gates to keep). By default it is computed
        with :func:`echo_mask` and ``**kwargs``.
    data_vars : str or list of str, optional
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

    References
    ----------
    No published method is implemented here: the function applies the mask
    of :func:`echo_mask`, whose Notes and References state how that
    classification relates to Gourley et al. (2007), Krause (2016), Park et
    al. (2009) and Steiner and Smith (2002). The NEXRAD no-data codes it
    masks are those decoded by xradar, not values from a paper.

    Examples
    --------
    >>> clean = radarx.retrieve.apply_mask(dtree, data_vars=["DBZH", "ZDR"])  # doctest: +SKIP
    >>> clean = dtree.radarx.apply_mask()  # doctest: +SKIP
    """
    if nodata not in ("auto", "nexrad", None):
        raise ValueError(f"nodata must be 'auto', 'nexrad' or None, not {nodata!r}")
    node = isinstance(obj, xr.DataTree)
    obj = as_sweep(obj)
    if node and isinstance(mask, xr.DataTree):
        mask = as_sweep(mask)  # the node of a mask volume
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
        return obj.assign(_mask_fields(obj, mask, data_vars, floors(obj)))

    if not isinstance(mask, xr.DataTree):
        raise TypeError("a DataTree needs the DataTree output of echo_mask")
    out = obj.copy()
    for name in mask.children:
        if name not in obj.children:
            continue
        ds = obj[name].to_dataset(inherit=False)
        wanted = data_vars
        if data_vars is not None:
            wanted = [data_vars] if isinstance(data_vars, str) else data_vars
            wanted = [f for f in wanted if f in ds]
        masked = _mask_fields(
            ds, mask[name].to_dataset(), wanted, floors(ds, obj.root.attrs)
        )
        for field, da in masked.items():
            out[f"{name}/{field}"] = da.variable
    return out


def _fields_for(ds, data_vars):
    """Optional fields that a sweep of a volume lacks are left out."""
    return (data_vars[0],) + tuple(
        f if f is None or f in ds else None for f in data_vars[1:]
    )
