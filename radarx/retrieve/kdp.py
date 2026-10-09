#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Differential Phase Processing and KDP
=====================================

Process the measured differential phase :math:`\\Psi_{DP}` of a sweep into a
smooth propagation phase :math:`\\Phi_{DP}` and estimate the specific
differential phase :math:`K_{DP} = \\frac{1}{2}\\, d\\Phi_{DP}/dr`.

Every ray is processed independently along range:

1. **Masking.** Gates count as meteorological where :math:`\\Psi_{DP}` is
   finite, :math:`\\rho_{hv}` reaches ``rhohv_min`` (default 0.85, the level
   below which Park et al. 2009 [3] consider the data contaminated by
   non-meteorological scatterers, p. 736) and the texture of
   :math:`\\Psi_{DP}` (its circular standard deviation over
   ``texture_window``, default 2 km, radarx choice) is at most
   ``texture_max`` (default 20 degrees, radarx choice). More than half of the
   texture window must be valid, so isolated gates are dropped. The circular
   statistic is computed from running sums of :math:`\\cos\\Psi_{DP}` and
   :math:`\\sin\\Psi_{DP}`, so it is not affected by phase folding.
2. **Sign convention.** Some systems record a phase that decreases with
   range in rain. The direction is detected from the phase differences of
   adjacent rain gates of all rays (wrapped, so folding does not matter) and
   the phase is multiplied by -1 if it decreases significantly
   (``phidp_sign``). This step, the thresholds of the detection and the
   offset and unfolding steps below are radarx's own and not taken from a
   paper.
3. **System offset.** The system differential phase is the circular mean of
   the first ``n_offset`` valid gates, either of each ray or pooled over the
   whole sweep (``offset="sweep"``, the default, which is robust to rays
   that start inside precipitation).
4. **Unfolding.** Starting from the system offset, every valid gate is moved
   by a multiple of 360° to the value closest to the mean of the previous
   five unfolded gates, and the offset is subtracted.
5. **Range filtering** with one of the ``method`` choices below. Masked
   gates are bridged by linear interpolation in range.
6. **KDP** is half the least-squares slope of the processed
   :math:`\\Phi_{DP}` over a window whose length adapts to the reflectivity:
   ``kdp_window[0]`` where :math:`Z_H` is at least ``z_threshold`` (heavy
   rain, where :math:`K_{DP}` changes quickly) and ``kdp_window[1]``
   elsewhere. The two window lengths and the 40 dBZ switch are those of the
   WSR-88D algorithm described by Park et al. (2009) [3] (Sect. 2a, p. 732: a
   lightly filtered :math:`K_{DP}` from 9 gates, 2 km at 0.25 km spacing, if
   :math:`Z > 40` dBZ, a heavily filtered one from 25 gates, 6 km, otherwise;
   they cite Ryzhkov and Zrnic 1996, not checked); radarx applies the short
   window at :math:`Z \\geq 40` dBZ. The trade-off between resolution and noise
   is discussed by Wang and Chandrasekar (2009) [2] (not checked against the
   paper). It is given at valid gates whose window
   holds at least ``min_valid_fraction`` valid gates.

Methods
-------
``"hubbert"`` (default)
    Iterative filtering after Hubbert and Bringi (1995) [1]. The profile is
    low-pass filtered; gates whose measurement departs from the filtered
    profile by more than ``delta_threshold`` (backscatter phase
    :math:`\\delta`, noise spikes) and masked gates are replaced by the
    filtered values, and the procedure is repeated ``n_iter`` times. The
    iteration on a monotonically increasing :math:`\\Phi_{DP}` with a
    backscatter phase :math:`\\delta` superposed is the one described by
    Bringi and Chandrasekar (2001) [5] (Sect. 6.6.1, pp. 369-372), who show a
    20th-order finite-impulse-response low-pass filter (their Fig. 6.32b)
    and 1 to 13 iterations (Fig. 6.34d). The low-pass filter
    here is three passes of a moving average of length ``filter_window`` (a
    cubic B-spline kernel, close to a Gaussian), each computed with running
    sums in O(N); it is radarx's substitute for the published filter and not
    the published coefficients. The defaults ``n_iter=10``,
    ``delta_threshold=4`` degrees and ``filter_window=2`` km are radarx's
    own (not checked against Hubbert and Bringi 1995 [1]).
``"vulpiani"``
    Iterative :math:`K_{DP}` estimation in the manner of Vulpiani et al.
    (2012) [4]: :math:`K_{DP}` is estimated from :math:`\\Phi_{DP}`, values
    outside ``kdp_bounds`` are set to zero, :math:`\\Phi_{DP}` is rebuilt by
    integrating :math:`2 K_{DP}` in range, and the two steps are repeated
    ``n_iter`` times. The integration constant is the least-squares fit of
    the rebuilt profile to the measured gates. This description is not
    checked against the paper. The defaults ``n_iter=4`` and
    ``kdp_bounds=(-2, 20)`` degrees/km are radarx's own.
``"monotone"``
    Monotone :math:`\\Phi_{DP}` as assumed by Maesaka et al. (2012) [6] for
    rain below the melting layer: the least-squares non-decreasing fit to the
    valid gates (pool-adjacent-violators algorithm, O(N)), smoothed with the
    same low-pass filter. :math:`K_{DP}` is then never negative. Maesaka et
    al. solve a variational problem under this constraint; the monotone fit
    used here is radarx's own simplification that enforces the same
    assumption in a single O(N) pass (not checked against the
    conference paper). Use it only
    for rain; hail or ice above the melting layer can have negative
    :math:`K_{DP}`.
``"ml"``
    A physics-constrained neural network (a one-dimensional convolutional
    network over range, run with ONNX Runtime through :mod:`radarx.ml`,
    ``pip install radarx[ml]``). Steps 1-4 are the same as above; the network
    receives the range derivative of the unfolded phase, the gate mask,
    :math:`\\rho_{hv}` and :math:`Z_H`, and returns :math:`K_{DP}`, the
    backscatter phase :math:`\\delta` and the standard deviation of
    :math:`K_{DP}`. It was trained on rays simulated from the T-matrix
    scattering tables of :mod:`radarx.retrieve.dsd` (no published method;
    radarx's own model) with a loss that,
    besides the error against the simulated truth, requires
    :math:`\\Psi_{DP} = 2\\int K_{DP}\\,dr + \\delta + \\Phi_{DP}^0`
    at valid gates, penalises negative :math:`K_{DP}` in rain and rough
    profiles (see ``ml/models/kdp`` in the radarx repository). The processed
    :math:`\\Phi_{DP}` is :math:`2\\int K_{DP}\\,dr` plus the constant that
    fits it best to :math:`\\Psi_{DP} - \\delta`, so it is consistent with
    :math:`K_{DP}` by construction. The trained weights are not distributed
    yet: register a model with :func:`radarx.ml.register_model` or pass a
    loaded model as ``model``.

The work is done by a compiled C++ kernel, multithreaded over rays; if it is
not available, an equivalent NumPy implementation is used.

References
----------
.. [1] Hubbert, J., and V. N. Bringi, 1995: An iterative filtering technique
   for the analysis of copolar differential phase and dual-frequency radar
   measurements. *J. Atmos. Oceanic Technol.*, **12** (3), 643-648,
   https://doi.org/10.1175/1520-0426(1995)012<0643:AIFTFT>2.0.CO;2
.. [2] Wang, Y., and V. Chandrasekar, 2009: Algorithm for estimation of the
   specific differential phase. *J. Atmos. Oceanic Technol.*, **26** (12),
   2565-2578, https://doi.org/10.1175/2009JTECHA1358.1
.. [3] Park, H. S., A. V. Ryzhkov, D. S. Zrnić, and K.-E. Kim, 2009: The
   hydrometeor classification algorithm for the polarimetric WSR-88D:
   Description and application to an MCS. *Wea. Forecasting*, **24** (3),
   730-748, https://doi.org/10.1175/2008WAF2222205.1
.. [4] Vulpiani, G., M. Montopoli, L. Delli Passeri, A. G. Gioia, P.
   Giordano, and F. S. Marzano, 2012: On the use of dual-polarized C-band
   radar for operational rainfall retrieval in mountainous areas. *J. Appl.
   Meteor. Climatol.*, **51** (2), 405-425,
   https://doi.org/10.1175/JAMC-D-10-05024.1
.. [5] Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler
   Weather Radar: Principles and Applications*. Cambridge University Press,
   https://doi.org/10.1017/CBO9780511541094
.. [6] Maesaka, T., K. Iwanami, and M. Maki, 2012: Non-negative KDP
   estimation by monotone increasing PhiDP assumption below melting layer.
   *Proc. Seventh European Conf. on Radar in Meteorology and Hydrology
   (ERAD 2012)*, Toulouse, France (conference paper, no DOI).

.. autosummary::
   :nosignatures:
   :toctree: generated/

   estimate_kdp
"""

from __future__ import annotations

__all__ = ["estimate_kdp"]

import numpy as np
import xarray as xr

from ._products import product_tree

try:
    from . import _kdp

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _kdp = None
    HAS_COMPILED_KERNEL = False

METHODS = {"hubbert": 0, "vulpiani": 1, "monotone": 2}
_DEFAULT_N_ITER = {"hubbert": 10, "vulpiani": 4, "monotone": 0}
# The two constants below are radarx's own choices, not from a paper.
_UNFOLD_MEMORY = 5  # unfolding reference: mean of this many previous gates
_MIN_VALID = 3  # rays with fewer valid gates are left empty


# --------------------------------------------------------------------------
# NumPy reference implementation (same steps and order as the C++ kernel)
# --------------------------------------------------------------------------


def _prefix(a):
    """Prefix sums along the last axis with a leading zero."""
    out = np.zeros(a.shape[:-1] + (a.shape[-1] + 1,))
    np.cumsum(a, axis=-1, out=out[..., 1:])
    return out


def _window_sums(prefix, h, n):
    """Sums over the window [g - h, g + h] (clipped) for every gate g."""
    g = np.arange(n)
    lo = np.maximum(g - h, 0)
    hi = np.minimum(g + h, n - 1) + 1
    return prefix[..., hi] - prefix[..., lo], lo, hi


def _valid_mask(phi, rho, rhohv_min, htex, tex_limit):
    """Meteorological gates: finite phase, rhohv and phase texture tests."""
    base = np.isfinite(phi)
    if rho is not None:
        with np.errstate(invalid="ignore"):
            base &= rho >= rhohv_min
    rad = np.where(base, np.deg2rad(np.where(base, phi, 0.0)), 0.0)
    c = np.where(base, np.cos(rad), 0.0)
    s = np.where(base, np.sin(rad), 0.0)
    ng = phi.shape[-1]
    sc, _, _ = _window_sums(_prefix(c), htex, ng)
    ss, _, _ = _window_sums(_prefix(s), htex, ng)
    sn, _, _ = _window_sums(_prefix(base.astype(np.float64)), htex, ng)
    # circular std <= limit  <=>  R^2 >= exp(-limit^2), R = |sum| / n
    ok = (sn >= htex + 1) & (sc * sc + ss * ss >= sn * sn * tex_limit)
    return base & ok, c, s


def _offsets(valid, c, s, n_offset, mode, value):
    """Per-ray system phase offset in degrees."""
    nr = valid.shape[0]
    rank = np.cumsum(valid, axis=1)
    first = valid & (rank <= n_offset)
    cr = np.where(first, c, 0.0).sum(axis=1)
    sr = np.where(first, s, 0.0).sum(axis=1)
    has = first.any(axis=1)
    if mode == "fixed":
        return np.full(nr, float(value))
    sweep = np.rad2deg(np.arctan2(sr.sum(), cr.sum())) if has.any() else 0.0
    if mode == "sweep":
        return np.full(nr, sweep)
    return np.where(has, np.rad2deg(np.arctan2(sr, cr)), sweep)


def _wrap180(d):
    """Differences wrapped into [-180, 180) degrees."""
    return d - 360.0 * np.floor(d / 360.0 + 0.5)


_SIGN_Z = 30.0  # rain gates for the sign test: Z >= 30 dBZ
_SIGN_RHO = 0.95  # and rhohv >= 0.95
_SIGN_SIGMAS = 3.0  # required significance of the sign test


def _sign_stats(phi, valid, rho, z):
    """
    Wrapped phase differences of adjacent rain gates: (sum, sum of squares,
    number of pairs, number of runs of pairs).
    """
    rain = valid.copy()
    with np.errstate(invalid="ignore"):
        if rho is not None:
            rain &= rho >= _SIGN_RHO
        if z is not None:
            rain &= z >= _SIGN_Z
    pair = rain[:, 1:] & rain[:, :-1]
    d = np.where(pair, _wrap180(phi[:, 1:] - phi[:, :-1]), 0.0)
    runs = pair[:, 0].sum() + (pair[:, 1:] & ~pair[:, :-1]).sum()
    return d.sum(), (d * d).sum(), int(pair.sum()), int(runs)


def _decide_sign(stats, phidp_sign):
    """
    +1, or -1 if the phase decreases in rain with a significance of
    ``_SIGN_SIGMAS``. Within a run of rain gates the differences telescope,
    so the noise of their sum grows with the number of runs.
    """
    if phidp_sign != 0:
        return phidp_sign
    total, total2, pairs, runs = (sum(col) for col in zip(*stats))
    if pairs == 0:
        return 1
    noise = np.sqrt(runs * total2 / pairs)
    return -1 if total < -_SIGN_SIGMAS * noise else 1


def _unfold(phi, valid, offset):
    """Unfold valid gates relative to the running reference, minus offset."""
    nr, ng = phi.shape
    buf = np.repeat(offset[:, None], _UNFOLD_MEMORY, axis=1)
    pos = np.zeros(nr, dtype=np.int64)
    rows = np.arange(nr)
    out = np.full(phi.shape, np.nan)
    for g in range(ng):
        v = valid[:, g]
        if not v.any():
            continue
        ref = buf.sum(axis=1) / _UNFOLD_MEMORY
        x = phi[:, g]
        u = x + 360.0 * np.floor((ref - x) / 360.0 + 0.5)
        out[v, g] = u[v] - offset[v]
        buf[rows[v], pos[v]] = u[v]
        pos[v] = (pos[v] + 1) % _UNFOLD_MEMORY
    return out


def _fill(x, valid):
    """Linear interpolation across masked gates; constant beyond the ends."""
    nr, ng = x.shape
    g = np.arange(ng)
    out = np.full(x.shape, np.nan)
    for i in range(nr):
        idx = g[valid[i]]
        if idx.size >= _MIN_VALID:
            out[i] = np.interp(g, idx, x[i, idx])
    return out


def _boxcar(y, h):
    """Moving average of length 2h+1 with odd reflection at both ends."""
    ng = y.shape[-1]
    k = np.minimum(np.arange(1, h + 1), ng - 1)
    left = 2.0 * y[:, :1] - y[:, k[::-1]]
    right = 2.0 * y[:, -1:] - y[:, ng - 1 - k]
    p = np.concatenate([left, y, right], axis=1)
    S = _prefix(p)
    return (S[:, 2 * h + 1 :] - S[:, :ng]) / (2 * h + 1)


def _smooth(y, h):
    """Low-pass filter: three moving-average passes (cubic B-spline)."""
    for _ in range(3):
        y = _boxcar(y, h)
    return y


def _slope(y, hgate, dr):
    """Least-squares slope per gate over [g - h, g + h] (clipped), per km."""
    ng = y.shape[-1]
    g = np.arange(ng, dtype=np.float64)
    lo = np.maximum(np.arange(ng) - hgate, 0)
    hi = np.minimum(np.arange(ng) + hgate, ng - 1) + 1
    P1 = _prefix(y)
    P2 = _prefix(y * g)
    rows = np.arange(y.shape[0])[:, None]
    sy = P1[rows, hi] - P1[rows, lo]
    sgy = P2[rows, hi] - P2[rows, lo]
    n = (hi - lo).astype(np.float64)
    gbar = 0.5 * (lo + hi - 1)
    den = n * (n * n - 1.0) / 12.0
    return (sgy - gbar * sy) / den / dr


def _isotonic(y):
    """Non-decreasing least-squares fit (pool-adjacent-violators)."""
    mean = []
    count = []
    for v in y:
        mean.append(v)
        count.append(1)
        while len(mean) > 1 and mean[-2] > mean[-1]:
            n = count[-2] + count[-1]
            m = (mean[-2] * count[-2] + mean[-1] * count[-1]) / n
            mean.pop()
            count.pop()
            mean[-1] = m
            count[-1] = n
    return np.repeat(mean, count)


def _mask_numpy(phi, rho, z, p):
    """Gate mask, phasors and sign-test statistics of one sweep."""
    tex_limit = np.exp(-np.deg2rad(p["texture_max"]) ** 2)
    valid, c, s = _valid_mask(phi, rho, p["rhohv_min"], p["htex"], tex_limit)
    return valid, c, s, _sign_stats(phi, valid, rho, z)


def _process_numpy(phi, mask, z, dr, p, sign):
    """NumPy implementation of the compiled kernel (same results)."""
    valid, c, s, _ = mask
    valid = valid.copy()
    offset = _offsets(valid, c, s, p["n_offset"], p["offset_mode"], p["offset"])
    x = _unfold(sign * phi, valid, sign * offset)
    good = valid.sum(axis=1) >= _MIN_VALID
    valid &= good[:, None]
    y0 = _fill(x, valid)
    ng = phi.shape[1]
    if z is not None:
        with np.errstate(invalid="ignore"):
            heavy = z >= p["z_threshold"]
    else:
        heavy = np.zeros(phi.shape, dtype=bool)
    hk = np.where(heavy, p["hk_short"], p["hk_long"])

    def kdp_of(y):
        ks = 0.5 * _slope(y, p["hk_short"], dr)
        kl = 0.5 * _slope(y, p["hk_long"], dr)
        return np.where(hk == p["hk_short"], ks, kl)

    method = p["method"]
    y0g = np.where(good[:, None], y0, 0.0)
    if method == METHODS["hubbert"]:
        y = y0g.copy()
        for _ in range(p["n_iter"]):
            f = _smooth(y, p["hf"])
            keep = valid & (np.abs(y0g - f) <= p["delta_threshold"])
            y = np.where(keep, y0g, f)
        out = _smooth(y, p["hf"])
        kdp = kdp_of(out)
    elif method == METHODS["vulpiani"]:
        out = y0g.copy()
        kdp = np.zeros_like(out)
        kmin, kmax = p["kdp_min"], p["kdp_max"]
        for _ in range(max(p["n_iter"], 1)):
            kdp = kdp_of(out)
            kdp = np.where((kdp < kmin) | (kdp > kmax), 0.0, kdp)
            steps = dr * (kdp[:, :-1] + kdp[:, 1:])
            out = np.concatenate(
                [out[:, :1], out[:, :1] + np.cumsum(steps, axis=1)], axis=1
            )
        # integration constant: least-squares fit to the measured gates
        n = np.maximum(valid.sum(axis=1), 1)
        shift = np.where(valid, x - out, 0.0).sum(axis=1) / n
        out = out + shift[:, None]
    else:
        mono = np.zeros_like(y0g)
        gates = np.arange(ng)
        for i in np.flatnonzero(good):
            idx = gates[valid[i]]
            fit = _isotonic(x[i, idx])
            mono[i] = np.interp(gates, idx, fit)
        out = _smooth(mono, p["hf"])
        kdp = kdp_of(out)
    out = np.where(good[:, None], out, np.nan)
    # KDP only at valid gates whose window holds enough valid gates
    nv = _prefix(valid.astype(np.float64))
    enough = np.zeros(valid.shape, dtype=bool)
    for h in (p["hk_short"], p["hk_long"]):
        lo = np.maximum(np.arange(ng) - h, 0)
        hi = np.minimum(np.arange(ng) + h, ng - 1) + 1
        ok = nv[:, hi] - nv[:, lo] >= p["min_valid_fraction"] * (hi - lo)
        enough = np.where(hk == h, ok, enough)
    kdp = np.where(valid & enough, kdp, np.nan)
    return out, kdp, offset


def _process_compiled(sweeps, params, n_threads):
    """Run the C++ kernel on all sweeps at once (one pool of rays)."""
    return _kdp.process_phidp(
        [sw["phi"] for sw in sweeps],
        [sw["rho"] for sw in sweeps],
        [sw["z"] for sw in sweeps],
        [sw["dr"] for sw in sweeps],
        [sw["htex"] for sw in sweeps],
        [sw["hf"] for sw in sweeps],
        [sw["hk_short"] for sw in sweeps],
        [sw["hk_long"] for sw in sweeps],
        params["method"],
        params["rhohv_min"],
        float(np.exp(-np.deg2rad(params["texture_max"]) ** 2)),
        params["n_offset"],
        {"sweep": 0, "ray": 1, "fixed": 2}[params["offset_mode"]],
        float(params["offset"]),
        params["n_iter"],
        params["delta_threshold"],
        params["z_threshold"],
        params["kdp_min"],
        params["kdp_max"],
        params["min_valid_fraction"],
        params["phidp_sign"],
        int(n_threads or 0),
    )


# --------------------------------------------------------------------------
# machine-learning method
# --------------------------------------------------------------------------

DEFAULT_ML_MODEL = "radarx-kdp-v1"
ML_FEATURES = ("dphi", "valid", "rhohv", "has_rhohv", "dbzh", "has_dbzh", "dr")
_ML_CHUNK = 512  # rays per network call


def _ml_features(phi, rho, z, dr, params, sign, mask=None):
    """
    Network inputs of one sweep, from the same masking, sign, offset and
    unfolding steps as the other methods.

    Returns the features (ray, feature, range) as float32 in the order of
    ``ML_FEATURES``, the unfolded phase without offset (NaN at masked gates),
    the valid-gate mask, the rays with enough valid gates and the offsets.
    """
    if mask is None:
        mask = _mask_numpy(phi, rho, z, params)
    valid = mask[0].copy()
    offset = _offsets(
        valid,
        mask[1],
        mask[2],
        params["n_offset"],
        params["offset_mode"],
        params["offset"],
    )
    psi = _unfold(sign * phi, valid, sign * offset)
    good = valid.sum(axis=1) >= _MIN_VALID
    valid &= good[:, None]
    psi = np.where(valid, psi, np.nan)
    filled = np.where(good[:, None], _fill(psi, valid), 0.0)
    nr, ng = phi.shape
    feats = np.zeros((nr, len(ML_FEATURES), ng), dtype=np.float32)
    # half the range derivative (degrees/km), scaled by 0.1
    feats[:, 0] = 0.05 * np.gradient(filled, dr, axis=1)
    feats[:, 1] = valid
    if rho is not None:
        ok = np.isfinite(rho)
        feats[:, 2] = np.where(ok, np.clip(np.where(ok, rho, 0.0), 0.0, 1.0), 0.0)
        feats[:, 3] = ok
    if z is not None:
        ok = np.isfinite(z)
        feats[:, 4] = (
            np.where(ok, np.clip(np.where(ok, z, 0.0), -30.0, 80.0), 0.0) / 50.0
        )
        feats[:, 5] = ok
    feats[:, 6] = dr
    return feats, psi, valid, good, offset


def _ml_phase(kdp, delta, psi, valid, good, dr):
    """
    Processed phase consistent with KDP: twice its range integral plus the
    constant that fits it best (least squares) to ``psi - delta``.
    """
    steps = dr * (kdp[:, :-1] + kdp[:, 1:])  # 2 * trapezoid of KDP
    phi = np.zeros_like(kdp)
    phi[:, 1:] = np.cumsum(steps, axis=1)
    n = np.maximum(valid.sum(axis=1), 1)
    shift = np.where(valid, psi - delta - phi, 0.0).sum(axis=1) / n
    phi = phi + shift[:, None]
    return np.where(good[:, None], phi, np.nan)


def _ml_model(model):
    """The model object for ``method="ml"``: a name or a loaded model."""
    if model is None:
        model = DEFAULT_ML_MODEL
    if not isinstance(model, str):
        if not callable(getattr(model, "run", None)):
            raise TypeError(
                "model must be a registered model name or a loaded radarx.ml "
                f"model with a run() method, not {type(model).__name__}"
            )
        return model, getattr(model, "info", None) or {}, None
    try:
        from radarx import ml
    except ImportError as err:  # radarx.ml or onnxruntime missing
        raise ImportError(
            "method='ml' needs radarx.ml and ONNX Runtime: pip install radarx[ml]"
        ) from err
    listed = ml.list_models()
    names = listed.keys() if isinstance(listed, dict) else [m["name"] for m in listed]
    if model not in names:
        raise ValueError(
            f"the KDP model {model!r} is not registered. The weights of the "
            "radarx KDP network are not distributed yet; train one with the "
            "scripts in ml/models/kdp of the radarx repository and register it "
            "with radarx.ml.register_model(...), or pass a loaded model as "
            "model=..."
        )
    loaded = ml.load_model(model)
    return loaded, getattr(loaded, "info", None) or {}, model


def _ml_run(model, feats):
    """Run the network over chunks of rays: KDP, delta and KDP std."""
    outs = {"kdp": [], "delta": [], "kdp_std": []}
    for start in range(0, feats.shape[0], _ML_CHUNK):
        res = model.run({"features": feats[start : start + _ML_CHUNK]})
        for key in outs:
            if key not in res:
                raise KeyError(f"the KDP model returned no {key!r} output")
            outs[key].append(np.asarray(res[key], dtype=np.float64))
    shape = feats[:, 0].shape
    return [
        np.concatenate(v, axis=0).reshape(shape) if v else np.zeros(shape)
        for v in outs.values()
    ]


def _run_ml(datasets, fields, params, model):
    """``method="ml"`` for a list of sweep datasets."""
    model, info, name = _ml_model(model)
    sweeps = [_prepare_sweep(ds, fields, params) for ds in datasets]
    masks = [
        _mask_numpy(sw["phi"], sw["rho"], sw["z"], {**params, **sw}) for sw in sweeps
    ]
    sign = _decide_sign([m[3] for m in masks], params["phidp_sign"])
    ml_attrs = {
        "ml_model": str(info.get("name", name or type(model).__name__)),
        "ml_model_version": str(info.get("version", "unknown")),
        "ml_model_licence": str(info.get("licence", "unknown")),
    }
    results = []
    for sw, mask in zip(sweeps, masks):
        p = {**params, **sw}
        feats, psi, valid, good, offset = _ml_features(
            sw["phi"], sw["rho"], sw["z"], sw["dr"], p, sign, mask
        )
        kdp, delta, std = _ml_run(model, feats)
        phi = _ml_phase(kdp, delta, psi, valid, good, sw["dr"])
        keep = valid & good[:, None]
        out = _wrap_sweep(
            sw,
            phi,
            np.where(keep, kdp, np.nan),
            offset,
            "ml",
            sign,
        )
        dims = out.KDP.dims
        out["PHIDP_BACKSCATTER"] = (
            dims,
            _transpose_like(np.where(keep, delta, np.nan), sw, dims),
            {
                "long_name": "Backscatter differential phase",
                "units": "degrees",
                "comment": "backscatter phase delta estimated by the network",
            },
        )
        out["KDP_UNCERTAINTY"] = (
            dims,
            _transpose_like(np.where(keep, std, np.nan), sw, dims),
            {
                "long_name": "Standard deviation of the specific differential phase",
                "units": "degrees/km",
                "comment": "predicted by the network",
            },
        )
        for var in ("PHIDP_processed", "KDP", "PHIDP_BACKSCATTER", "KDP_UNCERTAINTY"):
            out[var].attrs.update(ml_attrs)
        out["PHIDP_processed"].attrs["comment"] = (
            f"{sw['name']}{' multiplied by -1,' if sign < 0 else ''} twice the "
            "range integral of the network KDP, fitted to the measured phase "
            "minus the backscatter phase (method='ml')"
        )
        results.append(out)
    return results


def _transpose_like(values, sweep, dims):
    """(ray, range) values in the dimension order ``dims`` of the output."""
    return values if dims[0] == sweep["ray_dim"] else values.T


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
        raise ImportError("the compiled KDP kernel is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


def _half_gates(length_km, dr):
    """Half window in gates for a window ``length_km`` long (at least 1)."""
    return max(1, round(0.5 * float(length_km) / dr))


def _gate_spacing(rng):
    """Uniform gate spacing in km (range in metres)."""
    rng = np.asarray(rng, dtype=np.float64)
    if rng.size < 2:
        raise ValueError("need at least two range gates")
    step = np.diff(rng)
    dr = float(np.median(step))
    if dr <= 0 or np.max(np.abs(step - dr)) > 0.01 * dr:
        raise ValueError("KDP estimation needs uniformly spaced range gates")
    return dr / 1000.0


def _find(ds, name, candidates, required):
    if name is not None:
        if name in ds:
            return name
        if required:
            raise KeyError(f"{name!r} is not in the dataset")
        return None  # a volume sweep without this field is skipped
    for cand in candidates:
        if cand in ds:
            return cand
    if required:
        raise KeyError(f"none of {candidates} found; pass the field name")
    return None


# Raw (unprocessed) phase first: processed fields may already be filtered.
_PHIDP_NAMES = (
    "UPHIDP",
    "PHIDP",
    "uncorrected_differential_phase",
    "differential_phase",
    "PHI",
)
_RHOHV_NAMES = ("RHOHV", "cross_correlation_ratio", "copol_correlation_coeff")
_DBZH_NAMES = ("DBZH", "DBZ", "reflectivity", "corrected_reflectivity")


def _prepare_sweep(ds, fields, params):
    """Field names, contiguous (ray, range) arrays and windows of one sweep."""
    phidp, rhohv, dbzh = fields
    phidp = _find(ds, phidp, _PHIDP_NAMES, True)
    rhohv = _find(ds, rhohv, _RHOHV_NAMES, rhohv is not None)
    dbzh = _find(ds, dbzh, _DBZH_NAMES, dbzh is not None)
    da = ds[phidp]
    if "range" not in da.dims or da.ndim != 2:
        raise ValueError(f"{phidp!r} must be 2-D with a 'range' dimension")
    ray_dim = da.dims[0] if da.dims[1] == "range" else da.dims[1]
    dr = _gate_spacing(ds["range"].values)

    def arr(name):
        if name is None:
            return None
        values = ds[name].transpose(ray_dim, "range").values
        return np.ascontiguousarray(values, dtype=np.float64)

    short, long_ = params["kdp_window"]
    sweep = {
        "ds": ds,
        "name": phidp,
        "fields": (phidp, rhohv, dbzh),
        "ray_dim": ray_dim,
        "phi": arr(phidp),
        "rho": arr(rhohv),
        "z": arr(dbzh),
        "dr": dr,
        "htex": _half_gates(params["texture_window"], dr),
        "hf": _half_gates(params["filter_window"], dr),
        "hk_short": _half_gates(short, dr),
        "hk_long": _half_gates(long_, dr),
    }
    if sweep["z"] is None:
        sweep["hk_short"] = sweep["hk_long"]
    return sweep


def _wrap_sweep(sweep, out, kdp, offset, method, sign):
    """Processed PHIDP, KDP and the offset on the input sweep coordinates."""
    da = sweep["ds"][sweep["name"]]
    ray_dim = sweep["ray_dim"]
    dims = (ray_dim, "range")
    phidp, rhohv, dbzh = sweep["fields"]
    flipped = " multiplied by -1 (phase decreasing in range)," if sign < 0 else ""
    phi_attrs = {
        "standard_name": "radar_differential_phase_hv",
        "long_name": "Processed differential phase HV",
        "units": "degrees",
        "comment": (
            f"{phidp}{flipped} with the system offset removed, unfolded and "
            f"range filtered (method={method!r})"
        ),
        "source_fields": ", ".join(f for f in (phidp, rhohv, dbzh) if f),
        "phidp_sign": int(sign),
    }
    kdp_attrs = {
        "standard_name": "radar_specific_differential_phase_hv",
        "long_name": "Specific differential phase HV",
        "units": "degrees/km",
        "comment": "half the range derivative of the processed PHIDP",
    }
    off_attrs = {
        "long_name": "System differential phase offset",
        "units": "degrees",
        "comment": f"in the convention of {phidp}",
    }
    out_ds = xr.Dataset(
        {
            "PHIDP_processed": (dims, out, phi_attrs),
            "KDP": (dims, kdp, kdp_attrs),
            "PHIDP_OFFSET": ((ray_dim,), offset, off_attrs),
        },
        coords=dict(da.coords),
    )
    for name in ("PHIDP_processed", "KDP"):
        out_ds[name] = out_ds[name].transpose(*da.dims)
    return out_ds


def _run(datasets, fields, params, n_threads, use_compiled):
    """Process a list of sweep datasets; all rays in one kernel call."""
    sweeps = [_prepare_sweep(ds, fields, params) for ds in datasets]
    if use_compiled:
        outs, kdps, offsets, sign = _process_compiled(sweeps, params, n_threads)
        results = zip(outs, kdps, offsets)
    else:
        masks = [
            _mask_numpy(sw["phi"], sw["rho"], sw["z"], {**params, **sw})
            for sw in sweeps
        ]
        sign = _decide_sign([m[3] for m in masks], params["phidp_sign"])
        results = [
            _process_numpy(sw["phi"], m, sw["z"], sw["dr"], {**params, **sw}, sign)
            for sw, m in zip(sweeps, masks)
        ]
    return [
        _wrap_sweep(sw, out, kdp, offset, params["method_name"], sign)
        for sw, (out, kdp, offset) in zip(sweeps, results)
    ]


def estimate_kdp(
    obj,
    phidp=None,
    rhohv=None,
    dbzh=None,
    *,
    method="hubbert",
    rhohv_min=0.85,
    texture_window=2.0,
    texture_max=20.0,
    n_offset=10,
    offset="sweep",
    filter_window=2.0,
    n_iter=None,
    delta_threshold=4.0,
    kdp_window=(2.0, 6.0),
    z_threshold=40.0,
    kdp_bounds=(-2.0, 20.0),
    min_valid_fraction=0.5,
    phidp_sign="auto",
    n_threads=None,
    engine="auto",
    model=None,
):
    """
    Process the differential phase and estimate KDP for a sweep or a volume.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataTree
        A sweep on ``(azimuth, range)`` (or ``(elevation, range)``), or a
        volume with ``sweep_*`` groups, e.g. from xradar. Range gates must be
        uniformly spaced; ``range`` is in metres.
    phidp, rhohv, dbzh : str, optional
        Names of the measured (raw) differential phase (degrees), the
        copolar correlation coefficient and the reflectivity (dBZ). By
        default the first name found is used: ``UPHIDP``, ``PHIDP``,
        ``uncorrected_differential_phase``, ``differential_phase``, ``PHI``;
        ``RHOHV``, ``cross_correlation_ratio``, ``copol_correlation_coeff``;
        ``DBZH``, ``DBZ``, ``reflectivity``, ``corrected_reflectivity``.
        Files often hold several phase fields (raw, unfolded, corrected);
        pass the raw one explicitly when in doubt. The fields used are listed
        in the ``source_fields`` attribute of the output ``PHIDP_processed``. Without
        :math:`\\rho_{hv}` only the texture test masks gates; without
        reflectivity the long KDP window is used everywhere.
    method : {"hubbert", "vulpiani", "monotone", "ml"}, optional
        Range filtering method, see :mod:`radarx.retrieve.kdp`.
        Default ``"hubbert"``. ``"ml"`` runs a neural network and needs
        ``pip install radarx[ml]`` and a registered model (see ``model``).
    rhohv_min : float, optional
        Minimum :math:`\\rho_{hv}` of meteorological gates. Default 0.85, the
        level below which Park et al. (2009) [3] consider the data
        contaminated by non-meteorological scatterers (p. 736).
    texture_window : float, optional
        Length (km) of the window for the phase texture. Default 2 (radarx
        choice).
    texture_max : float, optional
        Maximum circular standard deviation (degrees) of the phase within the
        texture window. Default 20 (radarx choice).
    n_offset : int, optional
        Number of first valid gates used for the system offset. Default 10
        (radarx choice).
    offset : {"sweep", "ray"} or float, optional
        ``"sweep"`` (default) pools the first gates of all rays, ``"ray"``
        estimates one offset per ray, a number is used as the offset in
        degrees.
    filter_window : float, optional
        Length (km) of each of the three moving-average passes of the
        low-pass filter (``"hubbert"`` and ``"monotone"``). Default 2 (radarx
        choice; the filter is a substitute for the FIR filter of Hubbert and
        Bringi 1995 [1]); the combined kernel has a standard deviation of
        about half this length.
    n_iter : int, optional
        Iterations: default 10 for ``"hubbert"`` and 4 for ``"vulpiani"``
        (radarx choices, not checked against the papers [1], [4]).
    delta_threshold : float, optional
        ``"hubbert"``: gates departing from the filtered profile by more
        than this (degrees) are replaced by the filtered value. Default 4
        (radarx choice, not checked against Hubbert and Bringi 1995 [1]).
    kdp_window : (float, float), optional
        Length (km) of the least-squares KDP window where the reflectivity
        is at least / below ``z_threshold``. Default ``(2, 6)``: the 9 and 25
        gates (at 0.25 km spacing) of Park et al. (2009) [3], p. 732.
    z_threshold : float, optional
        Reflectivity (dBZ) above which the short window is used. Default 40,
        as in Park et al. (2009) [3], p. 732 (they use ``Z > 40``, radarx
        ``Z >= 40``).
    kdp_bounds : (float, float), optional
        ``"vulpiani"``: KDP values (degrees/km) outside these bounds are set
        to zero in each iteration. Default ``(-2, 20)`` (radarx choice).
    min_valid_fraction : float, optional
        KDP is only given at gates where at least this share of the gates in
        the KDP window is valid, which removes unreliable values at echo
        edges and next to masked gates. Default 0.5 (radarx choice).
    phidp_sign : {"auto", 1, -1}, optional
        Sign convention of the input phase. Some systems record a phase that
        decreases with range in rain. ``"auto"`` (default) multiplies the
        phase by -1 when it decreases with range in rain (Z >= 30 dBZ,
        rhohv >= 0.95): the sum of the phase differences of adjacent rain
        gates (wrapped into [-180, 180), so folding does not matter), pooled
        over all rays of all sweeps, must be negative by more than three
        times its noise. The value used is stored in the ``phidp_sign``
        attribute of the output ``PHIDP_processed``.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled
        kernel and falls back to NumPy. Not used by ``"ml"``.
    model : str or radarx.ml.Model, optional
        ``method="ml"`` only: the name of a model registered with
        :mod:`radarx.ml` (default ``"radarx-kdp-v1"``) or a loaded model.
        Its ONNX graph takes ``features`` (float32, ray x feature x range)
        and returns ``kdp``, ``delta`` and ``kdp_std`` (ray x range).

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        For a sweep, a Dataset with the processed differential phase
        ``PHIDP_processed`` (degrees, increasing with range in rain; masked gates are
        bridged and gates beyond the first and last valid gate hold the end
        values), ``KDP`` (degrees/km; NaN at non-meteorological gates) and
        the system offset ``PHIDP_OFFSET`` per ray (in the convention of the
        input phase), on the input coordinates. ``method="ml"`` adds the
        backscatter phase ``PHIDP_BACKSCATTER`` and the standard deviation
        ``KDP_UNCERTAINTY`` of KDP, and the attributes ``ml_model``,
        ``ml_model_version`` and ``ml_model_licence``. For a volume, a DataTree
        with one such node per sweep that has the differential phase, and the
        root of the input. Merge the products into the input with
        ``ds.radarx.assign(products)`` or ``dtree.radarx.assign(products)``.

    Raises
    ------
    KeyError
        If a requested field is missing.
    ValueError
        For unknown options or non-uniform range gates, or for
        ``method="ml"`` with a model name that is not registered.
    ImportError
        If ``engine="compiled"`` and the compiled kernel is not available,
        or for ``method="ml"`` without :mod:`radarx.ml` and ONNX Runtime.

    Notes
    -----
    Which parts follow the references: the iterative filtering idea [1], [5],
    the iterative KDP idea [4] and the monotone assumption [6] are the
    published concepts; their implementation details and all default values
    marked "radarx choice" above are not taken from the papers. The windows
    and the 40 dBZ switch are those of Park et al. (2009) [3]. Wang and
    Chandrasekar (2009) [2] is cited for the resolution/noise trade-off of
    the KDP window; its algorithm is not implemented.

    References
    ----------
    .. [1] Hubbert, J., and V. N. Bringi, 1995: An iterative filtering
       technique for the analysis of copolar differential phase and
       dual-frequency radar measurements. *J. Atmos. Oceanic Technol.*,
       **12** (3), 643-648,
       https://doi.org/10.1175/1520-0426(1995)012<0643:AIFTFT>2.0.CO;2
    .. [2] Wang, Y., and V. Chandrasekar, 2009: Algorithm for estimation of
       the specific differential phase. *J. Atmos. Oceanic Technol.*, **26**
       (12), 2565-2578, https://doi.org/10.1175/2009JTECHA1358.1
    .. [3] Park, H. S., A. V. Ryzhkov, D. S. Zrnić, and K.-E. Kim, 2009: The
       hydrometeor classification algorithm for the polarimetric WSR-88D:
       Description and application to an MCS. *Wea. Forecasting*, **24** (3),
       730-748, https://doi.org/10.1175/2008WAF2222205.1
    .. [4] Vulpiani, G., M. Montopoli, L. Delli Passeri, A. G. Gioia, P.
       Giordano, and F. S. Marzano, 2012: On the use of dual-polarized
       C-band radar for operational rainfall retrieval in mountainous areas.
       *J. Appl. Meteor. Climatol.*, **51** (2), 405-425,
       https://doi.org/10.1175/JAMC-D-10-05024.1
    .. [5] Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler
       Weather Radar: Principles and Applications*. Cambridge University
       Press, https://doi.org/10.1017/CBO9780511541094
    .. [6] Maesaka, T., K. Iwanami, and M. Maki, 2012: Non-negative KDP
       estimation by monotone increasing PhiDP assumption below melting
       layer. *Proc. Seventh European Conf. on Radar in Meteorology and
       Hydrology (ERAD 2012)*, Toulouse, France (conference paper, no DOI).

    Examples
    --------
    >>> out = radarx.retrieve.estimate_kdp(dtree["sweep_0"].ds)  # doctest: +SKIP
    >>> out = dtree.radarx.kdp(method="vulpiani")  # doctest: +SKIP
    >>> out = estimate_kdp(sweep, method="ml", model="my-kdp")  # doctest: +SKIP
    """
    if method not in METHODS and method != "ml":
        raise ValueError(
            f"method must be one of {sorted([*METHODS, 'ml'])}, not {method!r}"
        )
    if isinstance(offset, str):
        if offset not in ("sweep", "ray"):
            raise ValueError(
                f"offset must be 'sweep', 'ray' or a number, not {offset!r}"
            )
        offset_mode, offset_value = offset, 0.0
    else:
        offset_mode, offset_value = "fixed", float(offset)
    if isinstance(phidp_sign, str):
        if phidp_sign != "auto":
            raise ValueError(f"phidp_sign must be 'auto', 1 or -1, not {phidp_sign!r}")
        sign_value = 0
    elif phidp_sign in (1, -1):
        sign_value = int(phidp_sign)
    else:
        raise ValueError(f"phidp_sign must be 'auto', 1 or -1, not {phidp_sign!r}")
    if n_iter is None:
        n_iter = _DEFAULT_N_ITER.get(method, 0)
    use_compiled = _use_compiled(engine)
    params = {
        "method": METHODS.get(method, -1),
        "method_name": method,
        "rhohv_min": float(rhohv_min),
        "texture_window": texture_window,
        "texture_max": float(texture_max),
        "n_offset": int(n_offset),
        "offset_mode": offset_mode,
        "offset": offset_value,
        "filter_window": filter_window,
        "n_iter": int(n_iter),
        "delta_threshold": float(delta_threshold),
        "kdp_window": tuple(kdp_window),
        "z_threshold": float(z_threshold),
        "kdp_min": float(kdp_bounds[0]),
        "kdp_max": float(kdp_bounds[1]),
        "min_valid_fraction": float(min_valid_fraction),
        "phidp_sign": sign_value,
    }
    fields = (phidp, rhohv, dbzh)
    if method == "ml":

        def run(datasets):
            return _run_ml(datasets, fields, params, model)

    else:

        def run(datasets):
            return _run(datasets, fields, params, n_threads, use_compiled)

    if isinstance(obj, xr.Dataset):
        return run([obj])[0]

    names = [
        name
        for name in obj.children
        if name.startswith("sweep")
        and _find(obj[name].to_dataset(), phidp, _PHIDP_NAMES, False) is not None
    ]
    if not names:
        raise KeyError("no sweep contains a differential phase field")
    datasets = [obj[name].to_dataset(inherit=False) for name in names]
    results = run(datasets)
    return product_tree(obj, dict(zip(names, results)))
