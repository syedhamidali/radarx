#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Bayesian Drop Size Distribution Retrieval
=========================================

Posterior distribution of the parameters of a normalized gamma raindrop size
distribution (DSD; Testud et al. 2001; Bringi et al. 2002)

.. math::

    N(D) = N_w f(\\mu) \\left(\\frac{D}{D_m}\\right)^{\\mu}
    \\exp\\left(-(4 + \\mu) \\frac{D}{D_m}\\right)

at every radar gate, given :math:`Z_H`, :math:`Z_{DR}` and, where available,
:math:`K_{DP}` and the specific attenuation :math:`A_H`, with per-gate
uncertainty. The deterministic retrievals of :func:`radarx.retrieve.dsd`
fix the shape :math:`\\mu` (or tie it to :math:`\\Lambda`) and return one
DSD; here the shape is uncertain, the measurements are noisy and biased, and
the result is a posterior distribution (Rodgers 2000) summarised by its mean,
standard deviation, maximum (MAP) and quantiles (credible intervals).

State and forward model
-----------------------
The state is :math:`x = (t, D_m, \\mu)` with :math:`t = \\log_{10} N_w`.
For every :math:`(D_m, \\mu)` the radar variables follow from the T-matrix
single-drop scattering tables of :mod:`radarx.retrieve.dsd` (drops up to
8 mm) and are linear in :math:`N_w`:

.. math::

    Z_H = 10 t + L(D_m, \\mu),\\quad Z_{DR} = z(D_m, \\mu),\\quad
    K_{DP} = 10^t k(D_m, \\mu),\\quad A_H = 10^t a(D_m, \\mu).

:math:`(D_m, \\mu)` are discretized on a fixed grid (:math:`D_m` from 0.4 to
4.4 mm in steps of 0.05 mm, :math:`\\mu` from -0.5 to 12 in steps of 0.5);
:math:`t` is continuous. The rain rate (fall speed of Atlas et al. 1973) and
the liquid water content are :math:`10^t` times a function of
:math:`(D_m, \\mu)` (closed-form moments of the untruncated DSD, as in
:func:`radarx.retrieve.dsd`).

Measurement errors
------------------
Independent Gaussian errors on :math:`Z_H` and :math:`Z_{DR}` (dB) whose
variance is the sum of random noise and of an unknown calibration bias (for
one gate a bias of unknown sign is indistinguishable from noise, so its
variance adds), and on :math:`K_{DP}` and :math:`A_H` with an absolute and a
relative part (``errors``; the defaults are listed in :data:`ERRORS`). The
same variances also absorb forward-model errors (drop shapes, canting,
temperature).

Priors
------
:math:`p(t, D_m, \\mu) = p(D_m, \\mu)\\, \\mathcal{N}(t; m(D_m, \\mu),
s(D_m, \\mu)^2)`, a distribution on the grid times a Gaussian in
:math:`\\log_{10} N_w` whose mean and spread may depend on the node
(:func:`dsd_prior`):

``"generic"``
    A weakly informative prior from published disdrometer climatologies:
    :math:`D_m \\sim \\mathcal{N}(1.7, 0.6^2)` mm and
    :math:`\\log_{10} N_w \\sim \\mathcal{N}(3.75, 0.85^2)`, whose central
    95 % covers the :math:`D_m` of 1-2.75 mm and :math:`\\log_{10} N_w` of
    2-5.5 of stratiform, maritime-like and continental-like convective rain
    in Bringi et al. (2003); the shape is centred on the
    :math:`\\mu`-:math:`\\Lambda` relation of Cao et al. (2008) with a
    standard deviation of 2 (the constrained-gamma method is the limit of a
    vanishing spread).
``"perils2022"``
    Learned from 1-min OTT Parsivel2 DSDs of the Portable In situ
    Precipitation Stations (PIPS) in the PERiLS 2022 field campaign (northern
    Mississippi and Alabama, quasi-linear convective systems), fitted by the
    2-4-6 method of moments (Cao and Zhang 2009) after matching every size
    bin to the radar beam along its fall trajectory (size sorting and wind
    drift). A kernel density estimate on the grid with the conditional
    mean and spread of :math:`\\log_{10} N_w`, mixed with 1 % of the generic
    prior so that no DSD is impossible.
an :class:`xarray.Dataset`
    From :func:`dsd_prior`, e.g. learned from any disdrometer data set.

Inference
---------
The posterior is evaluated on the grid. For every node the conditional
posterior in :math:`t` is integrated by the Laplace method around its mode
(found by Gauss-Newton iterations; exact in one step without
:math:`K_{DP}` and :math:`A_H`, when it is Gaussian), giving the posterior
weight of the node and a Gaussian in :math:`t`; the posterior is the mixture
over the nodes. Nodes whose closed-form evidence from :math:`Z_H`,
:math:`Z_{DR}` and the prior is more than ``prune`` nats below the best are
skipped. Marginals of :math:`D_m` and :math:`\\mu` are piecewise constant
over the grid cells; those of :math:`\\log_{10} N_w`, :math:`\\log_{10} R`
and :math:`\\log_{10} W` are Gaussian mixtures, whose quantiles are found by
safeguarded Newton iterations on the mixture CDF. The log evidence
:math:`\\log p(y)` and the misfit :math:`\\chi^2` of the MAP state flag
gates that do not look like rain (hail, melting layer, clutter).

A compiled C++ kernel (``radarx.retrieve._dsd_bayes``) evaluates all gates
of all sweeps in one multithreaded call, with an identical NumPy reference
implementation as fallback and test oracle.

References
----------
Testud, J., S. Oury, R. A. Black, P. Amayenc, and X. Dou, 2001: The concept
of "normalized" distribution to describe raindrop spectra: A tool for cloud
physics and cloud remote sensing. *J. Appl. Meteor.*, **40** (6), 1118-1140,
https://doi.org/10.1175/1520-0450(2001)040<1118:TCONDT>2.0.CO;2

Bringi, V. N., G.-J. Huang, V. Chandrasekar, and E. Gorgucci, 2002: A
methodology for estimating the parameters of a gamma raindrop size
distribution model from polarimetric radar data: Application to a
squall-line event from the TRMM/Brazil campaign. *J. Atmos. Oceanic
Technol.*, **19** (5), 633-645,
https://doi.org/10.1175/1520-0426(2002)019<0633:AMFETP>2.0.CO;2

Bringi, V. N., V. Chandrasekar, J. Hubbert, E. Gorgucci, W. L. Randeu, and
M. Schoenhuber, 2003: Raindrop size distribution in different climatic
regimes from disdrometer and dual-polarized radar analysis. *J. Atmos.
Sci.*, **60** (2), 354-365,
https://doi.org/10.1175/1520-0469(2003)060<0354:RSDIDC>2.0.CO;2

Cao, Q., G. Zhang, E. Brandes, T. Schuur, A. Ryzhkov, and K. Ikeda, 2008:
Analysis of video disdrometer and polarimetric radar data to characterize
rain microphysics in Oklahoma. *J. Appl. Meteor. Climatol.*, **47** (8),
2238-2255, https://doi.org/10.1175/2008JAMC1732.1

Cao, Q., and G. Zhang, 2009: Errors in estimating raindrop size distribution
parameters employing disdrometer and simulated raindrop spectra. *J. Appl.
Meteor. Climatol.*, **48** (2), 406-425,
https://doi.org/10.1175/2008JAMC2026.1

Cao, Q., G. Zhang, and M. Xue, 2013: A variational approach for retrieving
raindrop size distribution from polarimetric radar measurements in the
presence of attenuation. *J. Appl. Meteor. Climatol.*, **52** (1), 169-185,
https://doi.org/10.1175/JAMC-D-12-0101.1

Rodgers, C. D., 2000: *Inverse Methods for Atmospheric Sounding: Theory and
Practice*. Series on Atmospheric, Oceanic and Planetary Physics, Vol. 2,
World Scientific, 238 pp., https://doi.org/10.1142/3171

Atlas, D., R. C. Srivastava, and R. S. Sekhon, 1973: Doppler radar
characteristics of precipitation at vertical incidence. *Rev. Geophys.*,
**11** (1), 1-35, https://doi.org/10.1029/RG011i001p00001

.. autosummary::
   :nosignatures:
   :toctree: generated/

   dsd_bayesian
   dsd_prior
   forward_grid
"""

from __future__ import annotations

__all__ = ["dsd_bayesian", "dsd_prior", "forward_grid"]

import functools
from importlib import resources

import numpy as np
import xarray as xr
from scipy.special import erfc, ndtri

from .._registry import accessor_method
from .dsd import (
    _DBZH_NAMES,
    _OUT_ATTRS,
    _ZDR_NAMES,
    MU_LAMBDA,
    _as_mask,
    _check_band,
    _f_mu,
    _find,
    _gamma_integrals,
    _infer_band,
    _kdp_for,
)

try:
    from . import _dsd_bayes

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _dsd_bayes = None
    HAS_COMPILED_KERNEL = False

# grid of (Dm, mu)
DM_AXIS = (0.4, 0.05, 81)  # first, step, size: 0.4-4.4 mm
MU_AXIS = (-0.5, 0.5, 26)  # -0.5-12

#: Default measurement error model: standard deviations of the random noise
#: and of the calibration bias of Z_H (dB) and Z_DR (dB), and the absolute
#: (degrees/km, dB/km) and relative parts of the K_DP and A_H errors.
ERRORS = {
    "zh": 1.0,
    "zh_bias": 1.0,
    "zdr": 0.2,
    "zdr_bias": 0.1,
    "kdp": 0.3,
    "kdp_rel": 0.1,
    "ah": 0.01,
    "ah_rel": 0.2,
}
PRIORS = ("generic", "perils2022")
_FIELDS = ("L", "ZDR", "KDP", "AH", "LOGR", "LOGW", "DM", "MU", "LOGP", "PMEAN", "PSD")
_N_FIXED = 16
_MAX_NEWTON = 8
_MAX_QUANTILE = 40
_ZMAX = 8.5  # normal tables on [-8.5, 8.5]
_ZRES = 256.0  # points per unit
_WEIGHT_MIN = 1e-9
_LN10 = np.log(10.0)
_LOG2PI = np.log(2.0 * np.pi)

# generic prior (see the module docstring)
_GENERIC = {"dm": (1.7, 0.6), "log10_nw": (3.75, 0.85), "mu_sd": 2.0}


def _axes():
    dm = DM_AXIS[0] + DM_AXIS[1] * np.arange(DM_AXIS[2])
    mu = MU_AXIS[0] + MU_AXIS[1] * np.arange(MU_AXIS[2])
    return dm, mu


# --------------------------------------------------------------------------
# forward model
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=16)
def _forward(band, temperature):
    """Forward model per unit Nw on the (Dm, mu) grid, as (n_dm, n_mu) arrays."""
    dm, mu = _axes()
    d2, m2 = np.meshgrid(dm, mu, indexing="ij")
    lam = (4.0 + m2) / d2
    n0 = _f_mu(m2) * d2 ** (-m2)  # N0 for Nw = 1
    zh, zv, kdp, ah = _gamma_integrals_ah(band, temperature, n0, m2, lam)
    m3 = 6.0 * d2**4 / 256.0  # third moment for Nw = 1
    rate = 6.0e-4 * np.pi * m3 * (9.65 - 10.3 * (lam / (lam + 0.6)) ** (m2 + 4.0))
    lwc = np.pi / 6.0 * 1.0e-3 * m3
    out = {
        "L": 10.0 * np.log10(zh),
        "ZDR": 10.0 * np.log10(zh / zv),
        "KDP": kdp,
        "AH": ah,
        "LOGR": np.log10(rate),
        "LOGW": np.log10(lwc),
        "DM": d2,
        "MU": m2,
    }
    for v in out.values():
        v.flags.writeable = False
    return out


def _gamma_integrals_ah(band, temperature, n0, mu, lam):
    """Z_H, Z_V, K_DP and A_H of gamma DSDs (A_H from the same table)."""
    from .dsd import _single_drop, _trapezoid_weights

    zh, zv, kdp = _gamma_integrals(band, temperature, n0, mu, lam)
    _, d, data = _single_drop(band, temperature)
    w = _trapezoid_weights(d)
    with np.errstate(over="ignore", under="ignore"):
        nd = n0[..., None] * d ** mu[..., None] * np.exp(-lam[..., None] * d)
    ah = (nd * w) @ data[:, 5]
    return zh, zv, kdp, ah


def forward_grid(band="S", temperature=20.0):
    """
    Forward model of the Bayesian DSD retrieval on its (Dm, mu) grid.

    Parameters
    ----------
    band : {"S", "C", "X"}, optional
        Radar band of the scattering tables. Default ``"S"``.
    temperature : float, optional
        Rain temperature in degrees Celsius (0-30). Default 20.

    Returns
    -------
    xarray.Dataset
        On ``dm`` (mm) and ``mu``, for :math:`N_w = 1` m-3 mm-1: ``DBZH``
        (dBZ), ``ZDR`` (dB), ``KDP`` (degrees/km), ``AH`` (dB/km),
        ``RAIN_RATE`` (mm/h) and ``LWC`` (g/m3). :math:`Z_H` adds
        :math:`10 \\log_{10} N_w`; the others scale with :math:`N_w`, except
        :math:`Z_{DR}`.
    """
    band = _check_band(band)
    f = _forward(band, float(temperature))
    dm, mu = _axes()
    dims = ("dm", "mu")
    return xr.Dataset(
        {
            "DBZH": (dims, f["L"], {"units": "dBZ"}),
            "ZDR": (dims, f["ZDR"], {"units": "dB"}),
            "KDP": (dims, f["KDP"], {"units": "degrees/km"}),
            "AH": (dims, f["AH"], {"units": "dB/km"}),
            "RAIN_RATE": (dims, 10.0 ** f["LOGR"], {"units": "mm h-1"}),
            "LWC": (dims, 10.0 ** f["LOGW"], {"units": "g m-3"}),
        },
        coords={
            "dm": (
                "dm",
                dm,
                {"long_name": "Mass-weighted mean diameter", "units": "mm"},
            ),
            "mu": ("mu", mu, {"long_name": "Shape parameter", "units": "1"}),
        },
        attrs={"band": band, "temperature": float(temperature), "nw": 1.0},
    )


# --------------------------------------------------------------------------
# priors
# --------------------------------------------------------------------------


def _mu_cao(dm):
    """Shape mu of the Cao et al. (2008) mu-Lambda relation at given Dm."""
    c2, c1, c0 = MU_LAMBDA["cao2008"]
    # mu = c2 L^2 + c1 L + c0 with L = (4 + mu) / Dm: fixed-point in mu
    mu = np.full_like(dm, 2.0)
    for _ in range(200):
        lam = (4.0 + mu) / dm
        mu = 0.5 * mu + 0.5 * (c2 * lam**2 + c1 * lam + c0)
    return mu


def _prior_dataset(mass, mean, sd, attrs):
    dm, mu = _axes()
    dims = ("dm", "mu")
    return xr.Dataset(
        {
            "prior_mass": (
                dims,
                mass,
                {
                    "long_name": "Prior probability of the (Dm, mu) grid cell",
                    "units": "1",
                },
            ),
            "log10_nw_mean": (
                dims,
                mean,
                {"long_name": "Prior mean of log10 Nw given (Dm, mu)", "units": "1"},
            ),
            "log10_nw_sd": (
                dims,
                sd,
                {"long_name": "Prior standard deviation of log10 Nw given (Dm, mu)"},
            ),
        },
        coords={"dm": ("dm", dm, {"units": "mm"}), "mu": ("mu", mu, {"units": "1"})},
        attrs=attrs,
    )


def _generic():
    dm, mu = _axes()
    (dmm, dms), (tm, ts), mus = (
        _GENERIC["dm"],
        _GENERIC["log10_nw"],
        _GENERIC["mu_sd"],
    )
    mc = _mu_cao(dm)
    # p(Dm) p(mu | Dm), the latter normalized on the mu axis of each Dm
    p_dm = np.exp(-0.5 * ((dm - dmm) / dms) ** 2)
    p_mu = np.exp(-0.5 * ((mu[None, :] - mc[:, None]) / mus) ** 2)
    mass = p_dm[:, None] * p_mu / p_mu.sum(axis=1, keepdims=True)
    mass /= mass.sum()
    shape = mass.shape
    return _prior_dataset(
        mass,
        np.full(shape, tm),
        np.full(shape, ts),
        {
            "prior": "generic",
            "comment": (
                "Dm ~ N(1.7, 0.6^2) mm, log10 Nw ~ N(3.75, 0.85^2), mu ~ "
                "N(mu_Cao2008(Dm), 2^2): ranges of Bringi et al. (2003), "
                "mu-Lambda relation of Cao et al. (2008)"
            ),
        },
    )


def _kde(dm_s, mu_s, t_s, weights=None, defensive=0.01, bandwidth=None):
    """Kernel density prior on the grid from samples of (Dm, mu, log10 Nw)."""
    dm_s, mu_s, t_s = (np.asarray(a, float).ravel() for a in (dm_s, mu_s, t_s))
    w = np.ones_like(dm_s) if weights is None else np.asarray(weights, float).ravel()
    dm, mu = _axes()
    ok = (
        np.isfinite(dm_s)
        & np.isfinite(mu_s)
        & np.isfinite(t_s)
        & np.isfinite(w)
        & (w > 0)
        & (dm_s >= dm[0] - DM_AXIS[1] / 2)
        & (dm_s <= dm[-1] + DM_AXIS[1] / 2)
        & (mu_s >= mu[0] - MU_AXIS[1] / 2)
        & (mu_s <= mu[-1] + MU_AXIS[1] / 2)
    )
    dm_s, mu_s, t_s, w = dm_s[ok], mu_s[ok], t_s[ok], w[ok]
    if dm_s.size < 10:
        raise ValueError("need at least 10 valid DSDs within the grid to learn a prior")
    w = w / w.sum()
    neff = 1.0 / np.sum(w**2)
    scott = neff ** (-1.0 / 6.0)

    def wsd(x):
        m = np.sum(w * x)
        return np.sqrt(np.sum(w * (x - m) ** 2))

    if bandwidth is None:
        h_dm = max(scott * wsd(dm_s), DM_AXIS[1])
        h_mu = max(scott * wsd(mu_s), MU_AXIS[1])
    else:
        h_dm, h_mu = (float(b) for b in bandwidth)
    h_t = scott * wsd(t_s)
    kd = np.exp(-0.5 * ((dm[:, None] - dm_s[None, :]) / h_dm) ** 2)  # (n_dm, n)
    km = np.exp(-0.5 * ((mu[:, None] - mu_s[None, :]) / h_mu) ** 2)  # (n_mu, n)
    k = kd[:, None, :] * km[None, :, :] * w  # (n_dm, n_mu, n)
    dens = k.sum(-1)
    # conditional mean and spread of log10 Nw, shrunk to the global values
    # with a pseudo-weight of one sample
    t_glob = np.sum(w * t_s)
    v_glob = wsd(t_s) ** 2
    w0 = 1.0 / neff
    mean = (k @ t_s + w0 * t_glob) / (dens + w0)
    var = (k @ t_s**2 + w0 * (v_glob + t_glob**2)) / (dens + w0) - mean**2
    var = np.maximum(var, 0.0) + h_t**2
    mass = dens / dens.sum()
    gen = _generic()
    if defensive > 0:
        gm = gen.prior_mass.values
        gmean = gen.log10_nw_mean.values
        gvar = gen.log10_nw_sd.values**2
        tot = (1.0 - defensive) * mass + defensive * gm
        a = np.where(tot > 0, (1.0 - defensive) * mass / np.where(tot > 0, tot, 1), 0.5)
        m_mix = a * mean + (1 - a) * gmean
        var = a * (var + mean**2) + (1 - a) * (gvar + gmean**2) - m_mix**2
        mean, mass = m_mix, tot
    return _prior_dataset(
        mass,
        mean,
        np.sqrt(var),
        {
            "prior": "learned",
            "n_samples": int(dm_s.size),
            "bandwidth_dm": h_dm,
            "bandwidth_mu": h_mu,
            "bandwidth_log10_nw": h_t,
            "defensive_weight": float(defensive),
        },
    )


@functools.lru_cache(maxsize=4)
def _packaged(name):
    path = resources.files("radarx.retrieve") / "data" / f"dsd_prior_{name}.csv"
    with resources.as_file(path) as p:
        vals = np.loadtxt(p, delimiter=",", comments="#", ndmin=2)
    out = _kde(vals[:, 1], vals[:, 2], vals[:, 0])
    out.attrs["prior"] = name
    return out


def dsd_prior(source="generic", *, weights=None, defensive=0.01, bandwidth=None):
    """
    Prior of the Bayesian DSD retrieval on its (Dm, mu) grid.

    Parameters
    ----------
    source : {"generic", "perils2022"} or xarray.Dataset, optional
        A packaged prior (see :mod:`radarx.retrieve.dsd_bayes`), or DSD
        parameters to learn one from: a Dataset with ``NW`` (m-3 mm-1),
        ``DM`` (mm) and ``MU`` on any dimensions, e.g. the output of
        :func:`radarx.retrieve.fit_gamma_moments` applied to disdrometer
        spectra (ideally matched to the radar beam along the fall
        trajectories). Default ``"generic"``.
    weights : array-like or xarray.DataArray, optional
        Weights of the samples of a learned prior. Default: equal.
    defensive : float, optional
        Weight of the generic prior mixed into a learned one, so that DSDs
        absent from the training data stay possible. Default 0.01.
    bandwidth : (float, float), optional
        Kernel widths in :math:`D_m` (mm) and :math:`\\mu` of a learned
        prior. Default: Scott's rule, at least one grid step.

    Returns
    -------
    xarray.Dataset
        On ``dm`` and ``mu``: ``prior_mass`` (sums to one), and the mean
        ``log10_nw_mean`` and standard deviation ``log10_nw_sd`` of the
        Gaussian prior of :math:`\\log_{10} N_w` at each node. It can be
        passed as ``prior`` to :func:`dsd_bayesian`.

    Examples
    --------
    >>> fits = radarx.retrieve.fit_gamma_moments(nd)  # doctest: +SKIP
    >>> prior = dsd_prior(fits)  # doctest: +SKIP
    """
    if isinstance(source, str):
        if source == "generic":
            return _generic()
        if source in PRIORS:
            return _packaged(source).copy()
        raise ValueError(f"prior must be one of {PRIORS} or a Dataset, not {source!r}")
    if not isinstance(source, xr.Dataset):
        raise TypeError("source must be a prior name or a Dataset with NW, DM and MU")
    missing = [v for v in ("NW", "DM", "MU") if v not in source]
    if missing:
        raise KeyError(f"the dataset lacks {missing}")
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.log10(source["NW"].values)
    w = None if weights is None else np.asarray(getattr(weights, "values", weights))
    return _kde(
        source["DM"].values,
        source["MU"].values,
        t,
        weights=w,
        defensive=defensive,
        bandwidth=bandwidth,
    )


def _check_prior(prior):
    if isinstance(prior, str) or prior is None:
        return dsd_prior(prior or "generic")
    if not isinstance(prior, xr.Dataset):
        raise TypeError("prior must be a prior name or a Dataset from dsd_prior")
    dm, mu = _axes()
    for v in ("prior_mass", "log10_nw_mean", "log10_nw_sd"):
        if v not in prior:
            raise KeyError(f"the prior lacks {v!r}; build it with dsd_prior")
    if prior["prior_mass"].shape != (dm.size, mu.size):
        raise ValueError(
            "the prior must be on the grid of dsd_prior (dims dm, mu of size "
            f"{dm.size}, {mu.size})"
        )
    return prior


def _grid(band, temperature, prior):
    """The (11, n_nodes) grid array of the kernel."""
    f = _forward(band, temperature)
    pm = prior["prior_mass"].transpose("dm", "mu").values
    with np.errstate(divide="ignore"):
        logp = np.where(pm > 0, np.log(np.where(pm > 0, pm, 1.0)), -np.inf)
    fields = dict(f)
    fields["LOGP"] = logp
    fields["PMEAN"] = prior["log10_nw_mean"].transpose("dm", "mu").values
    fields["PSD"] = prior["log10_nw_sd"].transpose("dm", "mu").values
    grid = np.stack([np.asarray(fields[k], float).ravel() for k in _FIELDS])
    bad = np.isfinite(grid[8]) & ~(grid[10] > 0)
    if bad.any():
        raise ValueError("prior standard deviations must be positive")
    # nodes without a usable forward model are impossible
    unusable = ~np.all(np.isfinite(grid[[0, 1, 4, 5]]), axis=0)
    grid[8, unusable] = -np.inf
    return np.ascontiguousarray(grid)


def _errors(errors, prune):
    e = dict(ERRORS)
    if errors:
        unknown = set(errors) - set(ERRORS)
        if unknown:
            raise ValueError(
                f"unknown error terms {sorted(unknown)}; use {list(ERRORS)}"
            )
        e.update({k: float(v) for k, v in errors.items()})
    for k, v in e.items():
        if not (np.isfinite(v) and v >= 0):
            raise ValueError(f"error {k!r} must be finite and non-negative")
    zh2 = e["zh"] ** 2 + e["zh_bias"] ** 2
    zdr2 = e["zdr"] ** 2 + e["zdr_bias"] ** 2
    if not (zh2 > 0 and zdr2 > 0):
        raise ValueError("the Z_H and Z_DR errors must not both be zero")
    vec = [zh2, zdr2, e["kdp"], e["kdp_rel"], e["ah"], e["ah_rel"], float(prune)]
    return e, vec


# --------------------------------------------------------------------------
# NumPy reference implementation (same steps and order as the C++ kernel)
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _normal_tables():
    """Normal CDF and density on [-_ZMAX, _ZMAX], _ZRES points per unit."""
    x = -_ZMAX + np.arange(int(2 * _ZMAX * _ZRES) + 1) / _ZRES
    return 0.5 * erfc(-x / np.sqrt(2.0)), np.exp(-0.5 * x * x - 0.5 * _LOG2PI)


def _normal(z):
    """Normal CDF and density by linear interpolation in the tables."""
    cdf_t, pdf_t = _normal_tables()
    ntab = cdf_t.size
    zc = np.clip(z, -_ZMAX, _ZMAX)
    u = (zc + _ZMAX) * _ZRES
    i = np.minimum(u.astype(np.int64), ntab - 2)
    f = u - i
    cdf = cdf_t[i] + f * (cdf_t[i + 1] - cdf_t[i])
    pdf = pdf_t[i] + f * (pdf_t[i + 1] - pdf_t[i])
    low, high = z <= -_ZMAX, z >= _ZMAX
    cdf = np.where(low, 0.0, np.where(high, 1.0, cdf))
    pdf = np.where(low | high, 0.0, pdf)
    return cdf, pdf


def _mixture_quantile(w, m, s, q, start):
    """
    Quantile q of Gaussian mixtures, one per row (w = 0 for unused nodes), by
    safeguarded Newton iterations from ``start``.
    """
    use = w > 0
    with np.errstate(invalid="ignore"):
        lo = np.min(np.where(use, m - _ZMAX * s, np.inf), axis=1)
        hi = np.max(np.where(use, m + _ZMAX * s, -np.inf), axis=1)
    x = np.minimum(np.maximum(start, lo), hi)
    active = np.ones(x.shape, bool)
    s_safe = np.where(use, s, 1.0)
    m_safe = np.where(use, m, 0.0)
    for _ in range(_MAX_QUANTILE):
        if not active.any():
            break
        z = (x[:, None] - m_safe) / s_safe
        c, p = _normal(z)
        cdf = np.sum(np.where(use, w * c, 0.0), axis=1)
        pdf = np.sum(np.where(use, w * p / s_safe, 0.0), axis=1)
        r = cdf - q
        active &= ~(np.abs(r) < 1e-7)
        hi = np.where(active & (r > 0), x, hi)
        lo = np.where(active & (r <= 0), x, lo)
        with np.errstate(divide="ignore", invalid="ignore"):
            xn = np.where(pdf > 0, x - r / pdf, 0.5 * (lo + hi))
        xn = np.where((xn > lo) & (xn < hi), xn, 0.5 * (lo + hi))
        dx = np.abs(xn - x)
        x = np.where(active, xn, x)
        active &= ~(dx < 1e-7)
    return x


def _cell_quantile(marg, x0, dx, q):
    """Quantile of piecewise-constant densities on a uniform axis (rows)."""
    n = marg.shape[1]
    acc = np.cumsum(marg, axis=1)
    prev = acc - marg
    hit = (acc >= q) & (marg > 0)
    i = np.where(hit.any(axis=1), np.argmax(hit, axis=1), n - 1)
    rows = np.arange(marg.shape[0])
    mi = marg[rows, i]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(hit.any(axis=1), (q - prev[rows, i]) / mi, 1.0)
    return x0 + (i - 0.5 + frac) * dx


def _retrieve_chunk(z, zdr, kdp, ah, grid, ev, quantiles):
    """Posterior summaries for gates that all have valid Z_H and Z_DR."""
    zh2, zdr2, ka, kr, aa, ar, prune = ev
    L, ZD, KK, AA, LR, LW, DMv, MUv, LP, PM, PS = grid
    ng = z.size
    zc, dc = z[:, None], zdr[:, None]
    finite_p = np.isfinite(LP)
    vt = zh2 + 100.0 * PS**2
    with np.errstate(invalid="ignore"):
        c = (
            LP
            - 0.5 * (dc - ZD) ** 2 / zdr2
            - 0.5 * (zc - 10.0 * PM - L) ** 2 / vt
            - 0.5 * np.log(vt)
        )
    c = np.where(finite_p, c, -np.inf)
    cmax = c.max(axis=1)
    keep = c >= (cmax - prune)[:, None]
    ok_gate = np.isfinite(cmax)

    use_k = np.zeros(ng, bool)
    vk = np.zeros(ng)
    kobs = np.zeros(ng)
    if kdp is not None:
        fin = np.isfinite(kdp)
        kobs = np.where(fin, kdp, 0.0)
        vk = ka**2 + (kr * kobs) ** 2
        use_k = fin & (vk > 0)
    use_a = np.zeros(ng, bool)
    va = np.zeros(ng)
    aobs = np.zeros(ng)
    if ah is not None:
        fin = np.isfinite(ah)
        aobs = np.where(fin, ah, 0.0)
        va = aa**2 + (ar * aobs) ** 2
        use_a = fin & (va > 0)
    vk_s = np.where(use_k, vk, 1.0)[:, None]
    va_s = np.where(use_a, va, 1.0)[:, None]
    uk, ua = use_k[:, None], use_a[:, None]
    ko, ao = kobs[:, None], aobs[:, None]

    ip = 1.0 / PS**2
    h0 = 100.0 / zh2 + ip
    t = (10.0 * (zc - L) / zh2 + PM * ip) / h0
    t = np.where(keep, t, 0.0)
    h = np.broadcast_to(h0, t.shape).copy()
    nonlin = (uk | ua) & keep
    if nonlin.any():
        # Gauss-Newton per node, column by column in mu: the offset of the
        # mode from the Gaussian part at the previous mu warm-starts the next
        ndm, nmu = DM_AXIS[2], MU_AXIS[2]
        t0 = t.copy()
        for col in range(nmu):
            j = np.arange(ndm) * nmu + col
            act = nonlin[:, j]
            if not act.any():
                continue
            tc = t0[:, j]
            if col > 0:
                prev = nonlin[:, j - 1]
                tc = np.where(act & prev, tc + (t[:, j - 1] - t0[:, j - 1]), tc)
            Lc, PMc, ipc, h0c = L[j], PM[j], ip[j], h0[j]
            Kc, Ac = KK[j], AA[j]
            active = act.copy()
            for _ in range(_MAX_NEWTON):
                p10 = np.exp(_LN10 * tc)
                grad = -10.0 * (zc - 10.0 * tc - Lc) / zh2 + (tc - PMc) * ipc
                jk = p10 * Kc * _LN10
                ja = p10 * Ac * _LN10
                grad = grad + np.where(uk, (p10 * Kc - ko) * jk / vk_s, 0.0)
                grad = grad + np.where(ua, (p10 * Ac - ao) * ja / va_s, 0.0)
                hh = (
                    h0c
                    + np.where(uk, jk * jk / vk_s, 0.0)
                    + np.where(ua, ja * ja / va_s, 0.0)
                )
                step = np.clip(grad / hh, -1.0, 1.0)
                tc = np.where(active, tc - step, tc)
                active &= ~(np.abs(step) < 1e-8)
                if not active.any():
                    break
            t[:, j] = np.where(act, tc, t[:, j])
        # Hessian at the mode with the second-order residual term, at least a
        # tenth of the Gauss-Newton one
        p10 = np.exp(_LN10 * t)
        jk = p10 * KK * _LN10
        ja = p10 * AA * _LN10
        hg = h0 + np.where(uk, jk * jk / vk_s, 0.0) + np.where(ua, ja * ja / va_s, 0.0)
        hx = (
            h0
            + np.where(uk, (jk * jk + (p10 * KK - ko) * jk * _LN10) / vk_s, 0.0)
            + np.where(ua, (ja * ja + (p10 * AA - ao) * ja * _LN10) / va_s, 0.0)
        )
        h = np.where(nonlin, np.maximum(hx, 0.1 * hg), h)
    p10 = 10.0**t
    f = (
        0.5 * (zc - 10.0 * t - L) ** 2 / zh2
        + 0.5 * (t - PM) ** 2 * ip
        + 0.5 * (dc - ZD) ** 2 / zdr2
        + np.log(PS)
        + np.where(uk, 0.5 * (p10 * KK - ko) ** 2 / vk_s, 0.0)
        + np.where(ua, 0.5 * (p10 * AA - ao) ** 2 / va_s, 0.0)
    )
    peak = np.where(keep, LP - f, -np.inf)
    lw = np.where(keep, peak + 0.5 * (_LOG2PI - np.log(h)), -np.inf)
    lwmax = lw.max(axis=1)
    wt = np.where(keep, np.exp(lw - lwmax[:, None]), 0.0)
    total = wt.sum(axis=1)
    w = wt / total[:, None]
    s = 1.0 / np.sqrt(h)
    m = t
    lognorm = (
        -0.5 * np.log(zh2)
        - 0.5 * np.log(zdr2)
        - _LOG2PI
        + np.where(use_k, -0.5 * (_LOG2PI + np.log(np.where(use_k, vk, 1.0))), 0.0)
        + np.where(use_a, -0.5 * (_LOG2PI + np.log(np.where(use_a, va, 1.0))), 0.0)
    )
    logev = lwmax + np.log(total) + lognorm - 0.5 * _LOG2PI
    nobs = 2.0 + use_k + use_a

    def sdev(m1, m2):
        return np.sqrt(np.maximum(m2 - m1 * m1, 0.0))

    st = np.sum(w * m, 1)
    st2 = np.sum(w * (m * m + s * s), 1)
    sd_ = np.sum(w * DMv, 1)
    sd2 = np.sum(w * DMv**2, 1)
    smu = np.sum(w * MUv, 1)
    smu2 = np.sum(w * MUv**2, 1)
    v = 0.5 * (s * _LN10) ** 2
    with np.errstate(over="ignore"):
        r = np.where(keep, 10.0 ** (m + LR), 0.0)
        ww = np.where(keep, 10.0 ** (m + LW), 0.0)
        ev_, ev4 = np.exp(np.where(keep, v, 0.0)), np.exp(np.where(keep, 4 * v, 0.0))
    sr = np.sum(w * r * ev_, 1)
    sr2 = np.sum(w * r * r * ev4, 1)
    sw = np.sum(w * ww * ev_, 1)
    sw2 = np.sum(w * ww * ww * ev4, 1)
    ndm, nmu = DM_AXIS[2], MU_AXIS[2]
    w3 = w.reshape(ng, ndm, nmu)
    marg_dm = w3.sum(axis=2)
    marg_mu = w3.sum(axis=1)

    nq = len(quantiles)
    out = np.full((_N_FIXED + 5 * nq, ng), np.nan)
    out[0] = st
    out[1] = sdev(st, st2)
    out[2] = sd_
    out[3] = np.sqrt(np.maximum(sd2 - sd_**2, 0.0) + DM_AXIS[1] ** 2 / 12.0)
    out[4] = smu
    out[5] = np.sqrt(np.maximum(smu2 - smu**2, 0.0) + MU_AXIS[1] ** 2 / 12.0)
    out[6] = sr
    out[7] = sdev(sr, sr2)
    out[8] = sw
    out[9] = sdev(sw, sw2)
    jb = np.argmax(peak, axis=1)
    rows = np.arange(ng)
    tb = m[rows, jb]
    out[10] = tb
    out[11] = DMv[jb]
    out[12] = MUv[jb]
    pb = 10.0**tb
    chi2 = (z - 10.0 * tb - L[jb]) ** 2 / zh2 + (zdr - ZD[jb]) ** 2 / zdr2
    chi2 = chi2 + np.where(use_k, (pb * KK[jb] - kobs) ** 2 / np.where(use_k, vk, 1), 0)
    chi2 = chi2 + np.where(use_a, (pb * AA[jb] - aobs) ** 2 / np.where(use_a, va, 1), 0)
    out[13] = chi2
    out[14] = logev
    out[15] = nobs
    wq = np.where(w > _WEIGHT_MIN, w, 0.0)
    lr = np.where(keep, m + LR, 0.0)
    lwc = np.where(keep, m + LW, 0.0)
    slr = np.sum(w * lr, 1)
    slr2 = np.sum(w * (lr * lr + s * s), 1)
    slw = np.sum(w * lwc, 1)
    slw2 = np.sum(w * (lwc * lwc + s * s), 1)
    sdt, sdr, sdw = sdev(st, st2), sdev(slr, slr2), sdev(slw, slw2)
    for i, q in enumerate(quantiles):
        zq = ndtri(q)
        out[_N_FIXED + i] = _mixture_quantile(wq, m, s, q, st + zq * sdt)
        out[_N_FIXED + nq + i] = _cell_quantile(marg_dm, DM_AXIS[0], DM_AXIS[1], q)
        out[_N_FIXED + 2 * nq + i] = _cell_quantile(marg_mu, MU_AXIS[0], MU_AXIS[1], q)
        out[_N_FIXED + 3 * nq + i] = 10.0 ** _mixture_quantile(
            wq, lr, s, q, slr + zq * sdr
        )
        out[_N_FIXED + 4 * nq + i] = 10.0 ** _mixture_quantile(
            wq, lwc, s, q, slw + zq * sdw
        )
    out[:, ~ok_gate] = np.nan
    return out


def _retrieve_numpy(z, zdr, kdp, ah, mask, grid, ev, quantiles, chunk=512):
    """NumPy implementation of the compiled kernel (same results)."""
    nq = len(quantiles)
    out = np.full((_N_FIXED + 5 * nq, z.size), np.nan)
    good = np.isfinite(z) & np.isfinite(zdr)
    if mask is not None:
        good &= mask.astype(bool)
    idx = np.flatnonzero(good)
    for a in range(0, idx.size, chunk):
        sel = idx[a : a + chunk]
        out[:, sel] = _retrieve_chunk(
            z[sel],
            zdr[sel],
            None if kdp is None else kdp[sel],
            None if ah is None else ah[sel],
            grid,
            ev,
            quantiles,
        )
    return out


def _retrieve_compiled(arrays, grid, ev, quantiles, n_threads):
    return _dsd_bayes.retrieve(
        [a[0] for a in arrays],
        [a[1] for a in arrays],
        [a[2] for a in arrays],
        [a[3] for a in arrays],
        [a[4] for a in arrays],
        grid,
        DM_AXIS[2],
        MU_AXIS[2],
        DM_AXIS[0],
        DM_AXIS[1],
        MU_AXIS[0],
        MU_AXIS[1],
        list(ev),
        [float(q) for q in quantiles],
        [float(ndtri(q)) for q in quantiles],
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
        raise ImportError("the compiled Bayesian DSD kernel is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


_AH_NAMES = ("AH", "specific_attenuation")
_Q_NAMES = ("LOG10_NW", "DM", "MU", "RAIN_RATE", "LWC")
_NW_ATTRS = {
    "long_name": "Normalized intercept parameter of the drop size distribution",
    "units": "m-3 mm-1",
}
_LOGNW_ATTRS = {
    "long_name": "log10 of the normalized intercept parameter",
    "units": "1",
}


def _out_attrs():
    a = {
        "LOG10_NW": dict(
            _LOGNW_ATTRS, long_name=_LOGNW_ATTRS["long_name"] + ", posterior mean"
        ),
        "DM": dict(_OUT_ATTRS["DM"]),
        "MU": dict(_OUT_ATTRS["MU"]),
        "RAIN_RATE": dict(_OUT_ATTRS["RAIN_RATE"]),
        "LWC": dict(_OUT_ATTRS["LWC"]),
    }
    for k in ("DM", "MU", "RAIN_RATE", "LWC"):
        a[k]["long_name"] += ", posterior mean"
    return a


def _wrap(item, values, attrs, quantiles):
    """Output Dataset on the coordinates of the reflectivity field."""
    ref = item["ref"]
    shape, dims = ref.shape, ref.dims

    def r(i):
        return values[i].reshape(shape)

    base = _out_attrs()
    dv = {}
    order = (("LOG10_NW", 0), ("DM", 2), ("MU", 4), ("RAIN_RATE", 6), ("LWC", 8))
    for name, i in order:
        dv[name] = (dims, r(i), base[name])
        sd_attrs = {k: v for k, v in base[name].items() if k != "standard_name"}
        sd_attrs["long_name"] = sd_attrs["long_name"].replace(
            "posterior mean", "posterior standard deviation"
        )
        dv[name + "_SD"] = (dims, r(i + 1), sd_attrs)
    with np.errstate(over="ignore"):
        dv["NW"] = (
            dims,
            10.0 ** r(0),
            dict(_NW_ATTRS, long_name=_NW_ATTRS["long_name"] + " (10 ** LOG10_NW)"),
        )
    for name, i in (("LOG10_NW", 10), ("DM", 11), ("MU", 12)):
        a = {k: v for k, v in base[name].items() if k != "standard_name"}
        a["long_name"] = a["long_name"].replace(
            "posterior mean", "maximum a posteriori"
        )
        dv[name + "_MAP"] = (dims, r(i), a)
    dv["MISFIT"] = (
        dims,
        r(13),
        {
            "long_name": "Chi-squared misfit of the maximum a posteriori state",
            "units": "1",
            "comment": "sum of squared standardized residuals over N_OBS inputs",
        },
    )
    dv["LOG_EVIDENCE"] = (
        dims,
        r(14),
        {"long_name": "Log marginal likelihood of the observations", "units": "1"},
    )
    dv["N_OBS"] = (dims, r(15), {"long_name": "Number of inputs used", "units": "1"})
    nq = len(quantiles)
    for k, name in enumerate(_Q_NAMES):
        block = values[_N_FIXED + k * nq : _N_FIXED + (k + 1) * nq]
        a = {kk: v for kk, v in base[name].items() if kk != "standard_name"}
        a["long_name"] = a["long_name"].replace("posterior mean", "posterior quantiles")
        dv[name + "_QUANTILES"] = (
            ("quantile",) + dims,
            block.reshape((nq,) + shape),
            a,
        )
    coords = dict(ref.coords)
    coords["quantile"] = ("quantile", np.asarray(quantiles, float))
    out = xr.Dataset(dv, coords=coords, attrs=dict(attrs))
    out.attrs["source_fields"] = ", ".join(f for f in item["fields"] if f)
    return out


def _prepare(ds, fields, mask, kdp_da, ah_da):
    dbzh, zdr = fields
    zname = _find(ds, dbzh, _DBZH_NAMES, True)
    dname = _find(ds, zdr, _ZDR_NAMES, True)
    ref = ds[zname]

    def flat(da):
        _, da = xr.broadcast(ref, da)
        values = da.transpose(*ref.dims).values.ravel()
        return np.ascontiguousarray(values, dtype=np.float64)

    arrays = (
        flat(ref),
        flat(ds[dname]),
        None if kdp_da is None else flat(kdp_da),
        None if ah_da is None else flat(ah_da),
        _as_mask(mask, ds, ref),
    )
    names = (
        zname,
        dname,
        None if kdp_da is None else (kdp_da.name or "KDP"),
        None if ah_da is None else (ah_da.name or "AH"),
    )
    return {"ref": ref, "fields": names, "arrays": arrays}


def _ah_for(ds, ah):
    if ah is None:
        return None
    if isinstance(ah, xr.DataArray):
        return ah
    name = _find(ds, ah, _AH_NAMES, False)
    return None if name is None else ds[name]


def _mask_for(mask, name):
    if not isinstance(mask, (xr.DataTree, dict)):
        return mask
    node = mask[name] if name in mask else None
    if node is None or isinstance(node, xr.DataArray):
        return node
    node = node.to_dataset() if isinstance(node, xr.DataTree) else node
    if len(node.data_vars) != 1:
        raise ValueError("each mask node must hold exactly one boolean variable")
    return node[next(iter(node.data_vars))]


def dsd_bayesian(
    obj,
    *,
    dbzh=None,
    zdr=None,
    kdp=None,
    ah=None,
    mask=None,
    band=None,
    temperature=20.0,
    prior="generic",
    errors=None,
    quantiles=(0.025, 0.16, 0.5, 0.84, 0.975),
    prune=15.0,
    n_threads=None,
    engine="auto",
):
    """
    Bayesian retrieval of the raindrop size distribution with uncertainty.

    Posterior of :math:`(\\log_{10} N_w, D_m, \\mu)` of a normalized gamma
    DSD at every gate from :math:`Z_H`, :math:`Z_{DR}` and, if given,
    :math:`K_{DP}` and :math:`A_H`, with T-matrix forward operators, a
    measurement-error model and a prior learned from disdrometers. See
    :mod:`radarx.retrieve.dsd_bayes` for the method.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataTree
        A sweep, a grid, a QVP or any dataset holding reflectivity and
        differential reflectivity on the same coordinates, or a volume with
        ``sweep_*`` groups (sweeps without both fields are skipped).
    dbzh, zdr : str, optional
        Names of the reflectivity (dBZ) and differential reflectivity (dB),
        calibrated and corrected for attenuation. By default the first found
        of the names :func:`radarx.retrieve.dsd` looks for.
    kdp : str, xarray.DataArray, xarray.DataTree or "estimate", optional
        Specific differential phase (degrees/km) as for
        :func:`radarx.retrieve.dsd`; used at every gate where it is finite,
        with its error model (no threshold is needed). Default: not used.
    ah : str or xarray.DataArray, optional
        Specific attenuation at horizontal polarization (dB/km), e.g. from
        an attenuation correction. Default: not used.
    mask : str, xarray.DataArray or xarray.DataTree, optional
        Rain gates (``True``); the others are left empty. As for
        :func:`radarx.retrieve.dsd`.
    band : {"S", "C", "X"}, optional
        Radar band; by default from the metadata as for
        :func:`radarx.retrieve.dsd`.
    temperature : float, optional
        Rain temperature in degrees Celsius (0-30). Default 20.
    prior : {"generic", "perils2022"} or xarray.Dataset, optional
        Prior of the DSD parameters (see :func:`dsd_prior`). Default
        ``"generic"``.
    errors : dict, optional
        Standard deviations overriding :data:`ERRORS`: ``zh`` and
        ``zh_bias`` (dB), ``zdr`` and ``zdr_bias`` (dB), ``kdp``
        (degrees/km) and ``kdp_rel``, ``ah`` (dB/km) and ``ah_rel``.
    quantiles : sequence of float, optional
        Posterior quantiles to return, in (0, 1). Default: the medians and
        the bounds of the central 68 % and 95 % credible intervals.
    prune : float, optional
        Grid nodes whose evidence from :math:`Z_H`, :math:`Z_{DR}` and the
        prior is more than ``prune`` nats below the best node are skipped.
        Default 15 (a relative weight below :math:`10^{-6}`).
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. Default ``"auto"``.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        On the coordinates of the reflectivity: the posterior means
        ``LOG10_NW``, ``DM`` (mm), ``MU``, ``RAIN_RATE`` (mm/h) and ``LWC``
        (g/m3), their standard deviations ``*_SD``, their quantiles
        ``*_QUANTILES`` along ``quantile``, ``NW`` = 10 ** ``LOG10_NW``, the
        maximum a posteriori state ``LOG10_NW_MAP``, ``DM_MAP``, ``MU_MAP``,
        its ``MISFIT`` (chi-squared over ``N_OBS`` inputs) and the
        ``LOG_EVIDENCE``. NaN where an input is missing or the mask is
        ``False``. For a volume, a DataTree with one node per sweep.

    Raises
    ------
    KeyError
        If a requested field is missing, or no sweep has the fields.
    ValueError
        For unknown options.
    ImportError
        If ``engine="compiled"`` and the compiled kernel is not available.

    References
    ----------
    Testud, J., S. Oury, R. A. Black, P. Amayenc, and X. Dou, 2001: The
    concept of "normalized" distribution to describe raindrop spectra: A tool
    for cloud physics and cloud remote sensing. *J. Appl. Meteor.*, **40**
    (6), 1118-1140,
    https://doi.org/10.1175/1520-0450(2001)040<1118:TCONDT>2.0.CO;2

    Cao, Q., G. Zhang, and M. Xue, 2013: A variational approach for
    retrieving raindrop size distribution from polarimetric radar
    measurements in the presence of attenuation. *J. Appl. Meteor.
    Climatol.*, **52** (1), 169-185, https://doi.org/10.1175/JAMC-D-12-0101.1

    Rodgers, C. D., 2000: *Inverse Methods for Atmospheric Sounding: Theory
    and Practice*. World Scientific, 238 pp., https://doi.org/10.1142/3171

    Examples
    --------
    >>> post = radarx.retrieve.dsd_bayesian(sweep, kdp="KDP")  # doctest: +SKIP
    >>> post.DM_QUANTILES.sel(quantile=[0.16, 0.84])  # doctest: +SKIP
    """
    use_compiled = _use_compiled(engine)
    quantiles = tuple(float(q) for q in np.atleast_1d(quantiles))
    if not all(0.0 < q < 1.0 for q in quantiles):
        raise ValueError("quantiles must be within (0, 1)")
    if not float(prune) > 0:
        raise ValueError("prune must be positive")
    is_tree = isinstance(obj, xr.DataTree)
    if band is None:
        objs = [obj] if not is_tree else [obj.ds]
        if is_tree:
            objs += [obj[c].ds for c in ("radar_parameters",) if c in obj.children] + [
                obj.root.ds
            ]
        band = _infer_band(objs)
    band = _check_band(band)
    temperature = float(temperature)
    prior_ds = _check_prior(prior)
    errs, ev = _errors(errors, prune)
    grid = _grid(band, temperature, prior_ds)
    attrs = {
        "method": "bayesian",
        "band": band,
        "temperature": temperature,
        "prior": str(prior_ds.attrs.get("prior", "custom")),
        "errors": ", ".join(f"{k}={v:g}" for k, v in errs.items()),
        "comment": (
            "posterior of a normalized gamma drop size distribution on a "
            "(Dm, mu) grid with Laplace integration in log10 Nw, retrieved by "
            "radarx.retrieve.dsd_bayesian"
        ),
    }
    fields = (dbzh, zdr)

    def run(items):
        if use_compiled:
            res = _retrieve_compiled(
                [it["arrays"] for it in items], grid, ev, quantiles, n_threads
            )
        else:
            res = [_retrieve_numpy(*it["arrays"], grid, ev, quantiles) for it in items]
        return [_wrap(it, r, attrs, quantiles) for it, r in zip(items, res)]

    if not is_tree:
        est = None
        if isinstance(kdp, str) and kdp == "estimate":
            from .kdp import estimate_kdp

            est = estimate_kdp(obj)["KDP"]
        kdp_da = _kdp_for(obj, kdp, est)
        if isinstance(kdp, str) and kdp != "estimate" and kdp_da is None:
            raise KeyError(f"{kdp!r} is not in the dataset")
        ah_da = _ah_for(obj, ah)
        if isinstance(ah, str) and ah_da is None:
            raise KeyError(f"{ah!r} is not in the dataset")
        return run([_prepare(obj, fields, mask, kdp_da, ah_da)])[0]

    names = [
        name
        for name in obj.children
        if name.startswith("sweep")
        and _find(obj[name].ds, dbzh, _DBZH_NAMES, False) is not None
        and _find(obj[name].ds, zdr, _ZDR_NAMES, False) is not None
    ]
    if not names:
        raise KeyError("no sweep contains both reflectivity and ZDR")
    est_tree = None
    if isinstance(kdp, str) and kdp == "estimate":
        from .kdp import estimate_kdp

        est_tree = estimate_kdp(obj)
    items = []
    for name in names:
        ds = obj[name].to_dataset(inherit=False)
        est = None
        if est_tree is not None and name in est_tree.children:
            est = est_tree[name].ds["KDP"]
        k = kdp
        if isinstance(kdp, xr.DataTree):
            k = kdp[name].ds["KDP"] if name in kdp.children else None
        a = ah
        if isinstance(ah, xr.DataTree):
            a = ah[name].ds["AH"] if name in ah.children else None
        items.append(
            _prepare(
                ds, fields, _mask_for(mask, name), _kdp_for(ds, k, est), _ah_for(ds, a)
            )
        )
    results = run(items)
    nodes = {"/": obj.root.to_dataset(inherit=False)}
    nodes.update(zip(names, results))
    return xr.DataTree.from_dict(nodes)


@accessor_method("dataset", "datatree", name="dsd_bayesian")
def _dsd_bayesian_accessor(self, **kwargs):
    """
    Bayesian retrieval of the raindrop size distribution with uncertainty.

    Posterior mean, standard deviation, quantiles and maximum of
    :math:`\\log_{10} N_w`, :math:`D_m`, :math:`\\mu`, rain rate and liquid
    water content at every gate (of every sweep of a volume).

    Parameters
    ----------
    **kwargs
        Options of :func:`radarx.retrieve.dsd_bayesian`, e.g. ``kdp``,
        ``mask``, ``prior``, ``errors``, ``band``.

    Returns
    -------
    xarray.Dataset or xarray.DataTree

    See Also
    --------
    radarx.retrieve.dsd_bayesian
    """
    return dsd_bayesian(self.xarray_obj, **kwargs)
