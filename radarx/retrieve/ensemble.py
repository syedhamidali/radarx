#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx Ensembles and Uncertainty of the Retrievals
==================================================

Monte Carlo propagation of the errors of the radar measurements, of the
calibration and of the prior knowledge of the drop size distribution (DSD)
through the retrievals of radarx. The members are copies of the radar data
with random errors; each member goes through the same retrieval, and the
ensemble gives the mean, the standard deviation and percentiles of every
product at every gate.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}

Error model
-----------
*Calibration.* The reflectivity of a member gets one random offset and the
differential reflectivity another, the same for all gates of a sweep, drawn
from zero-mean Gaussians. Member ``2k + 1`` has the offsets of member ``2k``
with the opposite sign (antithetic pairs; the same holds for the prior
changes below), so that a few members do not shift the ensemble as a whole by
the sample mean of a few draws. The defaults are 1 dB and 0.1 dB, the accuracy
Giangrande et al. (2004) give for the calibration of :math:`Z` and
:math:`Z_{DR}` of an unblocked polarimetric WSR-88D from rain and snow
signals, and the accuracy Ryzhkov et al. (2014) ask of :math:`Z` and
:math:`Z_{DR}` for the retrieval of :math:`N_w` (their statement is about
precision; radarx takes both numbers as one standard deviation, its own
choice).

*Measurement noise.* The estimate of a radar variable from :math:`M` pulse
pairs fluctuates. Melnikov and Zrnic (2004) derive the standard deviations
for simultaneous transmission and reception (SHV mode) with the receiver noise
included. With :math:`s_h` and :math:`s_v` the signal-to-noise ratios (linear)
of the two channels (:math:`s_v = s_h/z_{dr}`), :math:`\\rho` the copolar
correlation coefficient, and the number of independent samples

.. math::

    M_I = M\\left[1 + 2\\sum_{m=1}^{M-1}\\left(1 - \\frac{m}{M}\\right)
    e^{-(\\pi\\sigma_{vn} m)^2}\\right]^{-1},\\qquad
    \\sigma_{vn} = \\sigma_v / v_a

for a Gaussian Doppler spectrum of width :math:`\\sigma_v` and Nyquist
velocity :math:`v_a` (their Eq. A7), the variances are

.. math::

    \\mathrm{var}\\,\\frac{\\delta P_h}{S_h} = \\frac{2 s_h + 1}{M s_h^2}
    + \\frac{1}{M_I},\\qquad
    \\mathrm{cov}\\left(\\frac{\\delta P_h}{S_h}, \\frac{\\delta P_v}{S_v}\\right)
    = \\frac{\\rho^2}{M_I}

(Eqs. A6, A8, A12), and, in the same order as the code,

.. math::

    \\mathrm{SD}(Z_{DR}) &= \\frac{10}{\\ln 10}\\left[\\frac{2 s_h + 1}{M s_h^2}
    + \\frac{2 s_v + 1}{M s_v^2} + \\frac{2(1-\\rho^2)}{M_I}\\right]^{1/2}
    \\quad\\text{(dB, Eq. 2.11)}\\\\
    \\mathrm{SD}(\\Phi_{DP}) &= \\frac{1}{\\sqrt{2}\\,\\rho}\\left[
    \\frac{s_h + s_v + 1}{M s_h s_v} + \\frac{1-\\rho^2}{M_I}\\right]^{1/2}
    \\quad\\text{(rad, Eq. 3.5)}\\\\
    \\mathrm{SD}(\\rho) &= \\Big[\\frac{(1-2 s_h)\\rho^2}{4 M s_h^2}
    + \\frac{(1-2 s_v)\\rho^2}{4 M s_v^2} + \\frac{s_h + s_v + 1}{2 M s_h s_v}
    + \\frac{(1-\\rho^2)^2}{2 M_I}\\Big]^{1/2}\\quad\\text{(Eq. 4.8)}\\\\
    \\mathrm{SD}(v) &= \\frac{v_a}{\\sqrt{2}\\,\\pi\\,\\rho_1}\\left[
    \\frac{2 s_h(1-\\rho_1^2) + 1}{(M-1) s_h^2}
    + \\frac{1-\\rho_1^2}{M_{I1}}\\right]^{1/2}\\quad\\text{(Eq. C10)}

with :math:`\\rho_1 = e^{-(\\pi\\sigma_{vn})^2/2}` the correlation of
successive pulses and :math:`M_{I1}` the number of independent samples of
the lag-1 estimate (Eq. C6). The reflectivity in dB has the standard
deviation :math:`(10/\\ln 10)\\,[\\mathrm{var}(\\delta P_h/S_h)]^{1/2}`. The
errors of :math:`Z_H` and :math:`Z_{DR}` are drawn from the joint Gaussian of
the two power errors (:math:`Z_H` error :math:`= x_h`, :math:`Z_{DR}` error
:math:`= x_h - x_v`, in units of :math:`10/\\ln 10` dB), so that their
correlation is the one of the powers; the errors of :math:`\\Phi_{DP}`,
:math:`\\rho_{hv}` and :math:`v` are independent of them and of each other
(radarx choice; the cross-covariances with the powers are not used). The
formulas come from a perturbation analysis to second order; Melnikov and
Zrnic (2004) find them within 10 % of simulations for :math:`s_v \\ge 8` dB
(:math:`Z_{DR}`), 5 dB (:math:`\\Phi_{DP}`) and 9 dB (:math:`\\rho_{hv}`, for
:math:`\\rho_{hv} \\ge 0.95`) and :math:`\\sigma_v \\ge 1` m s-1 (1.5 for the
phase) at S band. radarx limits the inverse SNR to 10 (-10 dB) so that the
errors stay finite below that. The errors of different gates are independent.
The copolar correlation coefficient of a member is limited to [0, 1] (Melnikov
and Zrnic set estimates above 1 to 1).

*DSD prior.* Retrievals that depend on prior knowledge of the DSD get a random
change of it (``prior``): the prefactor of a rain rate relation is multiplied
by :math:`e^{\\sigma z}` (:math:`z` standard normal; the same for all gates of
a sweep, so that the map of the spread is right and an area total is
conservative); the specific attenuation coefficient :math:`\\alpha` and the
differential one :math:`\\beta` of the attenuation correction are scaled the
same way; the shape :math:`\\mu` of the DSD retrieval is shifted; the prior of
the Bayesian retrieval is tilted in :math:`D_m` and :math:`\\mu` and shifted
in :math:`\\log_{10} N_w`. The widths of the first two come from
:func:`relation_scatter`, the scatter of the DSD of the generic prior of
:func:`dsd_prior` about a power law, weighted by the rain each DSD carries as
in Ryzhkov et al. (2014); the others are radarx choices (see ``prior``).

Names of the results
--------------------
For a retrieved variable ``X`` the result holds ``X_mean``, ``X_median``,
``X_std`` (standard deviation over the members, ``n - 1``) and one
variable per percentile, ``X_p05``, ``X_p95`` for the default percentiles
(5 and 95; 2.5 gives ``X_p2.5``). ``X_std`` is the quantity that
:func:`radarx.retrieve.single_doppler_winds` returns as ``u_uncertainty``
(``uncertainty=True``). Class codes (``categorical``) give ``X_mode``, the most
frequent class (the lowest code on ties), and ``X_agreement``, the share of
the members that chose it. ``keep_members=True`` adds ``X_members`` with the
dimension ``member``. A gate with fewer finite members than
``min_fraction`` of the ensemble is NaN. The Bayesian retrieval adds
``X_total_std``, the square root of the mean posterior variance
(``X_SD``) plus the variance of the posterior means over the members.

Validation
----------
On a synthetic S-band sweep of rain cells with a known DSD (40 rays of 500
gates, 94 000 rain gates; 16 observed sweeps drawn with the error model, each
with an ensemble of 32 members built around it), the ratio of the RMS error of
the ensemble median to the RMS spread, for the part that comes from the
measurements and the calibration (``prior=False``, the reference is the
retrieval on the true radar fields), is 1.01 for :math:`R(Z)`, 0.97 for
:math:`R(Z, Z_{DR})`, 0.98 and 0.97 for :math:`D_m` and :math:`\\log_{10} N_w`
of :func:`radarx.retrieve.dsd`, 0.97 and 0.96 for the Bayesian retrieval and 1.01
for the reflectivity corrected for attenuation at C band; the true value lies
between the 5th and 95th percentile in 82 to 86 % of the gates (90 % for a
perfect ensemble; the 16 calibration offsets are the only common draws, so the
count is uncertain by a few percent). For :math:`K_{DP}` the ratio is 0.51 and
for :math:`R(K_{DP})` 0.59: the spread is about twice the error, because
:func:`radarx.retrieve.estimate_kdp` chooses the window of its range filter
from the phase noise, so a member with more noise is smoothed more. Against the
true DSD variable (``prior=True``) the ratios are 0.67 for :math:`R(Z)` (the
prior is broader than the scatter of this sweep), 0.86 for :math:`R(K_{DP})`,
1.13 for :math:`R(Z, Z_{DR})`, 0.70 for :math:`D_m` of ``dsd`` and 1.61 for
:math:`D_m` of the Bayesian retrieval, whose own posterior spread is not part of
the ensemble spread: the RMS of the error divided by the spread of
:math:`D_m` is 1.71, and 0.98 with ``X_total_std``, which puts 90 % of the true
values within 1.645 total standard deviations of the mean (70 % for the
ensemble spread alone). A smaller version of the experiment is in the notebook
``Ensemble_Uncertainty``.

Random numbers
--------------
The numbers are drawn from the counter-based generator Philox-4x64-10
(Salmon et al. 2011): the words for gate :math:`i` of member :math:`m` depend
only on ``seed``, the sweep and :math:`(i, m)`, so that the results do not
depend on the number of threads or on the order of the members, and any
member can be recomputed alone. The compiled kernel and the NumPy
implementation (which reads the words from :class:`numpy.random.Philox`)
draw the same 64-bit words. The Gaussian numbers are made with the Box-Muller
transform; the engines use their own ``log``, ``sin`` and ``cos``, so the
perturbed fields agree to rounding (relative 1e-12), not bit for bit.

Notes
-----
Dask arrays are loaded into memory: the members of a sweep are NumPy arrays.
The retrievals themselves run threaded in C++ over the gates or rays of each
member, so the members run one after the other. The input is never changed.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import xarray as xr

from .._provenance import provenance
from .._registry import accessor_method
from . import _ensemble_numpy as _kernels
from . import qpe as _qpe
from ._ensemble_numpy import HAS_COMPILED_KERNEL, KINDS, make_key, member_normals
from ._fieldtools import (
    DBZ_NAMES,
    PHIDP_NAMES,
    RHOHV_NAMES,
    SNR_NAMES,
    ZDR_NAMES,
    MissingField,
    choose_engine,
    find_field,
    map_sweeps,
)
from .attenuation import attenuation_coefficients, correct_attenuation
from .dsd import _infer_band, _relation
from .dsd import dsd as _dsd
from .dsd_bayes import _check_band, dsd_bayesian, dsd_prior, forward_grid
from .hid import hid as _hid
from .kdp import estimate_kdp
from .noise import snr as _snr

__all__ = [
    "perturb_radar",
    "measurement_error",
    "ensemble_retrieval",
    "ensemble_statistics",
    "relation_scatter",
]
__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

VEL_NAMES = ("VRADH", "VRAD", "velocity", "corrected_velocity")
_NAMES = {
    "dbz": DBZ_NAMES,
    "zdr": ZDR_NAMES,
    "phidp": PHIDP_NAMES,
    "rhohv": RHOHV_NAMES,
    "vel": VEL_NAMES,
}
_CANONICAL = {
    "dbz": "DBZH",
    "zdr": "ZDR",
    "phidp": "PHIDP",
    "rhohv": "RHOHV",
    "vel": "VRADH",
}
_ERROR_UNITS = {
    "dbz": "dB",
    "zdr": "dB",
    "phidp": "degrees",
    "rhohv": "1",
    "vel": "m s-1",
}
_RHO_DEFAULT = 0.99
_RHO_MIN = 0.5
_INV_SNR_MAX = 10.0  # SNR of -10 dB: the formulas are not valid below
_NYQUIST_DEFAULT = 25.0
_NOISE_KEYS = ("n_samples", "snr", "spectrum_width", "nyquist")
_DB = _kernels.DB


def _use_compiled(engine):
    """Whether to run the compiled kernel for the requested ``engine``."""
    return choose_engine(engine, HAS_COMPILED_KERNEL, "ensemble")


# --------------------------------------------------------------------------
# error model of the measurements
# --------------------------------------------------------------------------


def _check_samples(n_samples):
    n_samples = float(n_samples)
    if not n_samples >= 4:
        raise ValueError("n_samples must be at least 4")
    return n_samples


def _lag_sum(n_samples, sigma_vn, upto):
    """``sum_{m=1}^{upto} (1 - m/M) exp(-(pi sigma m)^2)`` for scalar or array sigma."""
    m = np.arange(1, int(upto) + 1, dtype=np.float64)
    weight = 1.0 - m / n_samples
    sigma = np.asarray(sigma_vn, dtype=np.float64)
    if sigma.ndim == 0:
        return float(np.sum(weight * np.exp(-((np.pi * float(sigma) * m) ** 2))))
    grid = np.geomspace(1.0e-3, 1.0, 513)
    table = (weight * np.exp(-((np.pi * grid[:, None] * m) ** 2))).sum(axis=1)
    return np.interp(np.log(np.clip(sigma, 1.0e-3, 1.0)), np.log(grid), table)


def _independent_samples(n_samples, sigma_vn):
    """Independent samples of the power estimate (Eq. A7) and of lag 1 (Eq. C6)."""
    m_i = n_samples / (1.0 + 2.0 * _lag_sum(n_samples, sigma_vn, n_samples - 1))
    m_i1 = (n_samples - 1.0) / (
        1.0 + 2.0 * _lag_sum(n_samples, sigma_vn, n_samples - 2)
    )
    return m_i, m_i1


def _inverse_snr(snr_db, shape):
    """Inverse linear SNR of the horizontal channel (0 without noise)."""
    if snr_db is None:
        return np.zeros(shape)
    with np.errstate(over="ignore"):
        inv = 10.0 ** (-np.asarray(snr_db, dtype=np.float64) / 10.0)
    return np.minimum(inv, _INV_SNR_MAX)


def _phase_sd(inv_h, inv_v, rho, n_samples, m_i):
    """Standard deviation of the differential phase in degrees (Eq. 3.5)."""
    cross = inv_h + inv_v + inv_h * inv_v
    rad = np.sqrt(cross / n_samples + (1.0 - rho**2) / m_i) / (np.sqrt(2.0) * rho)
    return np.degrees(rad)


def _rho_sd(inv_h, inv_v, rho, n_samples, m_i):
    """Standard deviation of the correlation coefficient (Eq. 4.8)."""
    rho2 = rho * rho
    cross = inv_h + inv_v + inv_h * inv_v
    var = (
        rho2 * (inv_h**2 - 2.0 * inv_h) / (4.0 * n_samples)
        + rho2 * (inv_v**2 - 2.0 * inv_v) / (4.0 * n_samples)
        + cross / (2.0 * n_samples)
        + (1.0 - rho2) ** 2 / (2.0 * m_i)
    )
    return np.sqrt(np.maximum(var, 0.0))


def _velocity_sd(inv_h, sigma_vn, nyquist, n_samples, m_i1):
    """Standard deviation of the pulse-pair velocity in m s-1 (Eq. C10)."""
    r1sq = np.exp(-((np.pi * sigma_vn) ** 2))
    bracket = (2.0 * inv_h * (1.0 - r1sq) + inv_h**2) / (n_samples - 1.0)
    bracket = bracket + (1.0 - r1sq) / m_i1
    return nyquist / (np.sqrt(2.0) * np.pi * np.sqrt(r1sq)) * np.sqrt(bracket)


def _error_model(zdr_db, rho, snr_db, n_samples, width, nyquist):
    """
    Standard deviations of the measurement errors and the factors of the draw.

    All inputs are float64 arrays of one shape (``snr_db`` may be None);
    ``width`` and ``nyquist`` are in m s-1. Returns a dict with ``dbz``,
    ``zdr`` (dB), ``phidp`` (degrees), ``rhohv``, ``vel`` (m s-1) and
    ``fac``, the six arrays of the kernel: the Cholesky factors of the two
    power errors (three) and the standard deviations of the phase, the
    correlation coefficient and the velocity.
    """
    shape = np.shape(rho)
    sigma_vn = np.clip(np.asarray(width, dtype=np.float64) / nyquist, 1.0e-3, 1.0)
    m_i, m_i1 = _independent_samples(n_samples, sigma_vn)
    rho = np.clip(rho, _RHO_MIN, 1.0)
    inv_h = _inverse_snr(snr_db, shape)
    inv_v = inv_h * 10.0 ** (np.nan_to_num(zdr_db) / 10.0)
    var_h = (2.0 * inv_h + inv_h**2) / n_samples + 1.0 / m_i
    var_v = (2.0 * inv_v + inv_v**2) / n_samples + 1.0 / m_i
    cov = rho * rho / m_i
    a_h = np.sqrt(var_h)
    b_v = cov / a_h
    d_v = np.sqrt(np.maximum(var_v - b_v**2, 0.0))
    sd = {
        "dbz": _DB * a_h,
        "zdr": _DB * np.sqrt(np.maximum(var_h + var_v - 2.0 * cov, 0.0)),
        "phidp": _phase_sd(inv_h, inv_v, rho, n_samples, m_i),
        "rhohv": _rho_sd(inv_h, inv_v, rho, n_samples, m_i),
        "vel": _velocity_sd(inv_h, sigma_vn, nyquist, n_samples, m_i1),
    }
    sd = {k: np.broadcast_to(v, shape) for k, v in sd.items()}
    fac = [np.broadcast_to(v, shape) for v in (a_h, b_v, d_v)]
    fac += [sd["phidp"], sd["rhohv"], sd["vel"]]
    sd["fac"] = np.ascontiguousarray(np.stack(fac))
    return sd


# --------------------------------------------------------------------------
# fields of a sweep
# --------------------------------------------------------------------------


def _as_dataset(obj):
    if isinstance(obj, xr.DataArray):
        if obj.name is None:
            raise ValueError("a DataArray needs a name that identifies the field")
        return obj.to_dataset()
    if not isinstance(obj, (xr.Dataset, xr.DataTree)):
        raise TypeError("obj must be an xarray DataArray, Dataset or DataTree")
    return obj


def _broadcast(da, ref):
    _, da = xr.broadcast(ref, da)
    return da.transpose(*ref.dims)


def _flat(da, ref):
    values = _broadcast(da, ref).values
    return np.ascontiguousarray(values, dtype=np.float64).reshape(-1)


def _gate_values(value, ds, ref):
    """A number, a field name or a DataArray as a flat float64 array of the gates."""
    if isinstance(value, str):
        if value not in ds:
            raise KeyError(f"{value!r} is not in the dataset")
        return _flat(ds[value], ref)
    if isinstance(value, xr.DataArray):
        return _flat(value, ref)
    return np.full(ref.size, float(value))


def _nyquist_of(ds, nyquist):
    """Nyquist velocity (m s-1): given, from the data or the default."""
    value = _NYQUIST_DEFAULT
    if nyquist is not None:
        value = float(nyquist)
    elif "nyquist_velocity" in ds.variables:
        found = np.asarray(ds["nyquist_velocity"].values, dtype=np.float64)
        found = found[np.isfinite(found)]
        value = float(np.median(found)) if found.size else value
    if not value > 0:
        raise ValueError("nyquist must be positive")
    return value


def _wants_estimate(snr):
    return isinstance(snr, str) and snr == "estimate"


def _noise_options(noise, **given):
    """Options of the noise model, or None without noise."""
    if noise is False or noise is None:
        return None
    options = {"n_samples": 64, "snr": None, "spectrum_width": 3.0, "nyquist": None}
    options.update({k: v for k, v in given.items() if v is not None})
    if isinstance(noise, dict):
        unknown = set(noise) - set(_NOISE_KEYS)
        if unknown:
            raise ValueError(f"unknown noise options {sorted(unknown)}")
        options.update(noise)
    return options


class _Sweep:
    """The fields of one sweep as flat arrays, with their error model."""

    def __init__(self, ds, names, noise):
        self.names = {}
        for kind in KINDS:
            found = find_field(ds, names.get(kind), _NAMES[kind], kind, required=False)
            if found is not None:
                self.names[kind] = found
        if not self.names:
            raise MissingField(f"no radar field found among {_NAMES}")
        self.kinds = [k for k in KINDS if k in self.names]
        self.ref = ds[self.names[self.kinds[0]]]
        self.n = self.ref.size
        self.fields = {k: _broadcast(ds[self.names[k]], self.ref) for k in self.kinds}
        self.values = {k: _flat(self.fields[k], self.ref) for k in self.kinds}
        self.error = None if noise is None else self._model(ds, noise)

    def _snr(self, ds, snr):
        if snr is None:
            snr = find_field(ds, None, SNR_NAMES, "snr", required=False)
        if _wants_estimate(snr):
            snr = _snr(ds, dbz=self.names["dbz"])
        return None if snr is None else _gate_values(snr, ds, self.ref)

    def _model(self, ds, noise):
        if _wants_estimate(noise["snr"]) and "dbz" not in self.names:
            raise MissingField("snr='estimate' needs the reflectivity")
        rho = self.values.get("rhohv", np.full(self.n, _RHO_DEFAULT))
        width = noise["spectrum_width"]
        if not isinstance(width, (int, float)):
            width = _gate_values(width, ds, self.ref)
        return _error_model(
            self.values.get("zdr", np.zeros(self.n)),
            rho,
            self._snr(ds, noise["snr"]),
            _check_samples(noise["n_samples"]),
            width,
            _nyquist_of(ds, noise["nyquist"]),
        )

    def factors(self):
        if self.error is not None:
            return self.error["fac"]
        return np.zeros((6, self.n))


def _field_names(dbz, zdr, phidp, rhohv, field):
    given = {"dbz": dbz, "zdr": zdr, "phidp": phidp, "rhohv": rhohv, "vel": field}
    return {k: v for k, v in given.items() if v is not None}


def _stream(name):
    """Stream of the random numbers of a sweep: its number."""
    tail = name.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def _over_sweeps(obj, func, what):
    """Apply ``func(dataset, stream)`` to a Dataset or to the sweeps of a DataTree."""
    obj = _as_dataset(obj)
    if isinstance(obj, xr.DataTree):
        return map_sweeps(obj, lambda ds, name: func(ds, _stream(name)), what)
    return func(obj, 0)


# --------------------------------------------------------------------------
# perturbed fields
# --------------------------------------------------------------------------


def _check_members(members):
    members = int(members)
    if members < 2:
        raise ValueError("members must be at least 2")
    return members


def _check_calibration(calibration):
    if calibration is None:
        return 0.0, 0.0
    sd = tuple(float(v) for v in calibration)
    if len(sd) != 2 or min(sd) < 0:
        raise ValueError(
            "calibration needs two standard deviations (dBZ and ZDR, dB) >= 0"
        )
    return sd


def _antithetic(key, member, block):
    """Four Gaussian numbers of a member; member ``2k + 1`` gets minus those of ``2k``."""
    sign = 1.0 if member % 2 == 0 else -1.0
    return sign * member_normals(key, member // 2, block)


def _biases(key, m0, count, calibration):
    """Calibration offsets (reflectivity, ZDR) of members ``m0 .. m0 + count - 1``."""
    z = np.array([_antithetic(key, m0 + j, 2)[:2] for j in range(count)])
    return (
        np.ascontiguousarray(z[:, 0] * calibration[0]),
        np.ascontiguousarray(z[:, 1] * calibration[1]),
    )


def _perturb(sw, key, m0, count, calibration, use_compiled, n_threads):
    """Perturbed fields of ``count`` members, shape (count, fields, gates)."""
    bias_dbz, bias_zdr = _biases(key, m0, count, calibration)
    return _kernels.perturb(
        sw.values,
        sw.kinds,
        sw.factors(),
        key,
        m0,
        bias_dbz,
        bias_zdr,
        use_compiled,
        n_threads,
    )


def _out_dtype(da):
    return da.dtype if da.dtype.kind == "f" else np.dtype(np.float32)


def _member_array(sw, kind, values):
    """DataArray with the dimension ``member`` for one perturbed field."""
    da = sw.fields[kind]
    count = values.shape[0]
    shaped = values.reshape((count,) + sw.ref.shape).astype(_out_dtype(da))
    return da.expand_dims(member=np.arange(count)).copy(data=shaped)


def _perturbed_dataset(sw, values):
    return xr.Dataset(
        {
            sw.names[k]: _member_array(sw, k, values[:, s])
            for s, k in enumerate(sw.kinds)
        }
    )


def _perturb_sweep(ds, stream, args):
    sw = _Sweep(ds, args["names"], args["noise"])
    key = make_key(args["seed"], stream)
    values = _perturb(
        sw,
        key,
        0,
        args["members"],
        args["calibration"],
        args["use_compiled"],
        args["n_threads"],
    )
    return _perturbed_dataset(sw, values)


@provenance(
    "Monte Carlo perturbation of the radar fields: calibration offsets and "
    "measurement noise after Melnikov and Zrnic (2004), Philox random numbers",
    extra_refs=("melnikov-2004", "giangrande-2004"),
)
def perturb_radar(
    obj,
    *,
    dbz=None,
    zdr=None,
    phidp=None,
    rhohv=None,
    field=None,
    members=32,
    seed=0,
    calibration=(1.0, 0.1),
    noise=True,
    n_samples=64,
    snr=None,
    spectrum_width=3.0,
    nyquist=None,
    n_threads=None,
    engine="auto",
):
    """
    Copies of the radar fields with random calibration and measurement errors.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        A named field (DataArray), a sweep or grid, or a volume with
        ``sweep_*`` groups (sweeps without any of the fields are skipped).
    dbz, zdr, phidp, rhohv, field : str, optional
        Names of the reflectivity (dBZ), the differential reflectivity (dB),
        the differential phase (degrees), the copolar correlation
        coefficient and the radial velocity (m s-1). By default the first
        name found of ``DBZH``, ``DBZ``, ``reflectivity``,
        ``corrected_reflectivity``; ``ZDR``, ``differential_reflectivity``,
        ``corrected_differential_reflectivity``; ``PHIDP``,
        ``differential_phase``, ``UPHIDP``; ``RHOHV``,
        ``cross_correlation_ratio``, ``RHO``; ``VRADH``, ``VRAD``,
        ``velocity``, ``corrected_velocity``. A field that is not found is
        left out; at least one is needed.
    members : int, optional
        Number of members. Default 32.
    seed : int, optional
        Seed of the random numbers, an integer in [0, 2**64). Default 0.
    calibration : (float, float) or None, optional
        Standard deviations (dB) of the offsets of the reflectivity and the
        differential reflectivity; one offset per member and field, the same
        for every gate. Default ``(1.0, 0.1)``; ``None`` for no offsets.
    noise : bool, optional
        Add the measurement noise of :func:`measurement_error`. Default True.
    n_samples : float, optional
        Number of pulse pairs :math:`M` of an estimate (or the number of
        independent estimates averaged). Default 64, the value of the figures
        of Melnikov and Zrnic (2004). At least 4.
    snr : float, str, xarray.DataArray or "estimate", optional
        Signal-to-noise ratio (dB) of the horizontal channel: a number, the
        name of a field, a DataArray, or ``"estimate"`` to compute it with
        :func:`radarx.retrieve.snr`. Default: the field ``SNR`` (or ``SNRH``,
        ``signal_to_noise_ratio``) if there is one, else no noise from the
        receiver (only the variance of the signal). Gates where it is NaN
        give NaN.
    spectrum_width : float, str or xarray.DataArray, optional
        Spectrum width :math:`\\sigma_v` (m s-1) as a number, a field name or
        a DataArray. Default 3, the value of Figs. 2.7 and 3.6 of Melnikov
        and Zrnic (2004).
    nyquist : float, optional
        Nyquist velocity (m s-1). Default: the median ``nyquist_velocity`` of
        the data, else 25.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled
        kernel and falls back to NumPy.

    Returns
    -------
    xarray.DataArray, xarray.Dataset or xarray.DataTree
        The perturbed fields under their own names, with the extra dimension
        ``member`` (a DataArray for a DataArray input). For a volume a
        DataTree with one such node per sweep. The input is not changed.

    Raises
    ------
    KeyError
        If no field is found.
    ValueError
        For unknown options, ``members`` below 2, ``n_samples`` below 4 or a
        ``seed`` outside the range.
    ImportError
        If ``engine="compiled"`` and the compiled kernel is not available.

    Notes
    -----
    The error model, the random numbers and the limits of the formulas are
    in :mod:`radarx.retrieve.ensemble`. Member ``m`` of a sweep depends only
    on ``seed``, the sweep number and ``m``, so that any member can be
    recomputed alone, and more members extend the ensemble.

    References
    ----------
    Melnikov, V. M., and D. S. Zrnic, 2004: Simultaneous transmission mode
    for the polarimetric WSR-88D: Statistical biases and standard deviations
    of polarimetric variables. NOAA/NSSL report, 84 pp.,
    https://www.nssl.noaa.gov/publications/wsr88d_reports/SHV_statistics.pdf

    Giangrande, S. E., A. V. Ryzhkov, V. M. Melnikov, and J. Krause, 2004:
    Calibration of the polarimetric NEXRAD radar using meteorological signals.
    Preprints, 11th Conf. on Aviation, Range, and Aerospace Meteorology,
    paper P5.15, https://ams.confex.com/ams/11aram22sls/techprogram/paper_81811.htm

    Salmon, J. K., M. A. Moraes, R. O. Dror, and D. E. Shaw, 2011: Parallel
    random numbers: as easy as 1, 2, 3. *Proc. 2011 Int. Conf. for High
    Performance Computing, Networking, Storage and Analysis*, 1-12,
    https://doi.org/10.1145/2063384.2063405

    Examples
    --------
    >>> ens = radarx.retrieve.perturb_radar(sweep, members=16, seed=1)  # doctest: +SKIP
    >>> ens.DBZH.std("member")  # doctest: +SKIP
    """
    args = {
        "names": _field_names(dbz, zdr, phidp, rhohv, field),
        "members": _check_members(members),
        "seed": seed,
        "calibration": _check_calibration(calibration),
        "noise": _noise_options(
            noise,
            n_samples=n_samples,
            snr=snr,
            spectrum_width=spectrum_width,
            nyquist=nyquist,
        ),
        "use_compiled": _use_compiled(engine),
        "n_threads": n_threads,
    }
    out = _over_sweeps(
        obj, lambda ds, stream: _perturb_sweep(ds, stream, args), "perturb_radar"
    )
    return out[obj.name] if isinstance(obj, xr.DataArray) else out


# --------------------------------------------------------------------------
# measurement error
# --------------------------------------------------------------------------


def _error_attrs(name, kind, ref):
    return {
        "long_name": f"standard deviation of the measurement error of {name}",
        "units": _ERROR_UNITS[kind],
    }


def _error_sweep(ds, stream, args):
    sw = _Sweep(ds, args["names"], args["noise"])
    return xr.Dataset(
        {
            f"{sw.names[k]}_error": sw.fields[k]
            .copy(data=sw.error[k].reshape(sw.ref.shape).astype(np.float32))
            .assign_attrs(_error_attrs(sw.names[k], k, sw.ref))
            for k in sw.kinds
        }
    )


@provenance(
    "Standard deviation of the radar measurement errors for simultaneous "
    "transmission, after Melnikov and Zrnic (2004)",
    extra_refs=("melnikov-2004",),
)
def measurement_error(
    obj,
    *,
    dbz=None,
    zdr=None,
    phidp=None,
    rhohv=None,
    field=None,
    n_samples=64,
    snr=None,
    spectrum_width=3.0,
    nyquist=None,
):
    """
    Standard deviation of the estimation error of the radar variables.

    The error of an estimate from ``n_samples`` pulse pairs with the receiver
    noise, for simultaneous transmission (SHV mode); the equations are in
    :mod:`radarx.retrieve.ensemble`.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        A named field (DataArray), a sweep or grid, or a volume with
        ``sweep_*`` groups (sweeps without any of the fields are skipped).
    dbz, zdr, phidp, rhohv, field, n_samples, snr, spectrum_width, nyquist
        As for :func:`perturb_radar`. The correlation coefficient of the
        formulas is the field ``rhohv`` limited to 0.5-1 (0.99 without the
        field), the differential reflectivity is the field ``zdr`` (0 dB
        without it).

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        For each field found ``<name>_error``, the standard deviation of its
        error on the coordinates of the field, in dB (reflectivity and
        differential reflectivity), degrees (differential phase), 1
        (correlation coefficient) and m s-1 (velocity), as float32. The
        input is not changed.

    Raises
    ------
    KeyError
        If no field is found.
    ValueError
        For unknown options or ``n_samples`` below 4.

    Notes
    -----
    The formulas are perturbation results to second order; Melnikov and Zrnic
    (2004) find them within 10 % of simulations for an SNR of the vertical
    channel of at least 5 to 9 dB, depending on the variable (the inverse SNR
    is limited to 10 here to keep the result finite), see
    :mod:`radarx.retrieve.ensemble`.

    References
    ----------
    Melnikov, V. M., and D. S. Zrnic, 2004: Simultaneous transmission mode
    for the polarimetric WSR-88D: Statistical biases and standard deviations
    of polarimetric variables. NOAA/NSSL report, 84 pp.,
    https://www.nssl.noaa.gov/publications/wsr88d_reports/SHV_statistics.pdf

    Examples
    --------
    >>> err = radarx.retrieve.measurement_error(sweep, snr=15)  # doctest: +SKIP
    >>> err.ZDR_error  # doctest: +SKIP
    """
    args = {
        "names": _field_names(dbz, zdr, phidp, rhohv, field),
        "noise": _noise_options(
            True,
            n_samples=n_samples,
            snr=snr,
            spectrum_width=spectrum_width,
            nyquist=nyquist,
        ),
    }
    return _over_sweeps(
        obj, lambda ds, stream: _error_sweep(ds, stream, args), "measurement_error"
    )


# --------------------------------------------------------------------------
# statistics of an ensemble
# --------------------------------------------------------------------------


def _pname(q):
    return f"p{q:02g}"


def _check_percentiles(percentiles):
    q = [float(p) for p in np.atleast_1d(percentiles)]
    if not all(0.0 < p < 100.0 for p in q):
        raise ValueError("percentiles must be within (0, 100)")
    return q


def _min_valid(members, min_fraction):
    if not 0.0 <= float(min_fraction) <= 1.0:
        raise ValueError("min_fraction must be within [0, 1]")
    return max(1, int(math.ceil(float(min_fraction) * members)))


def _stat_attrs(template, name, what, members):
    attrs = {k: v for k, v in template.attrs.items() if k in ("units", "long_name")}
    attrs["long_name"] = f"{what} of {attrs.get('long_name', name)} over the ensemble"
    attrs["ensemble_members"] = int(members)
    return attrs


def _on(template, values):
    return template.copy(data=values.astype(np.float32).reshape(template.shape))


def _stat_variables(name, template, stack, settings):
    """Statistics variables of one product; ``stack`` is (members, *template.shape)."""
    members = stack.shape[0]
    q = settings["percentiles"]
    mean, std, pct = _kernels.statistics(
        stack.reshape(members, -1),
        [50.0] + q,
        settings["min_valid"],
        settings["use_compiled"],
        settings["n_threads"],
    )
    values = {
        f"{name}_mean": (mean, "mean"),
        f"{name}_median": (pct[0], "median"),
        f"{name}_std": (std, "standard deviation"),
    }
    for k, p in enumerate(q):
        values[f"{name}_{_pname(p)}"] = (pct[k + 1], f"{p:g}th percentile")
    return {
        key: _on(template, v).assign_attrs(_stat_attrs(template, name, what, members))
        for key, (v, what) in values.items()
    }


def _class_variables(name, template, stack, settings):
    """Mode and agreement of a product of class codes."""
    members = stack.shape[0]
    cls, share = _kernels.mode(
        stack.reshape(members, -1),
        settings["min_valid"],
        settings["use_compiled"],
        settings["n_threads"],
    )
    flags = {k: v for k, v in template.attrs.items() if k.startswith("flag_")}
    mode_attrs = {"ensemble_members": int(members), **flags}
    mode_attrs["long_name"] = f"most frequent class of {name} over the ensemble"
    share_attrs = {
        "long_name": f"share of the members that chose the mode of {name}",
        "units": "1",
        "ensemble_members": int(members),
    }
    return {
        f"{name}_mode": _on(template, cls).assign_attrs(mode_attrs),
        f"{name}_agreement": _on(template, share).assign_attrs(share_attrs),
    }


def _settings(members, percentiles, min_fraction, engine, n_threads):
    return {
        "percentiles": _check_percentiles(percentiles),
        "min_valid": _min_valid(members, min_fraction),
        "use_compiled": _use_compiled(engine),
        "n_threads": n_threads,
    }


def _statistics_sweep(ds, stream, args):
    out = {}
    dim = args["dim"]
    for name, da in ds.data_vars.items():
        if dim not in da.dims:
            continue
        da = da.transpose(dim, ...)
        template = da.isel({dim: 0}, drop=True)
        stack = np.asarray(da.values)
        settings = dict(
            args["settings"], min_valid=_min_valid(stack.shape[0], args["min_fraction"])
        )
        if name in args["categorical"]:
            out.update(_class_variables(name, template, stack, settings))
        else:
            out.update(_stat_variables(name, template, stack, settings))
    if not out:
        raise MissingField(f"no variable has the dimension {dim!r}")
    return xr.Dataset(out)


@provenance("Statistics over the members of an ensemble")
def ensemble_statistics(
    obj,
    *,
    dim="member",
    percentiles=(5, 95),
    categorical=(),
    min_fraction=0.5,
    n_threads=None,
    engine="auto",
):
    """
    Mean, spread and percentiles over the members of an ensemble.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        Members along ``dim``, for example the output of
        :func:`perturb_radar`, or ``<name>_members`` of
        :func:`ensemble_retrieval` (a DataArray needs a name). For a volume
        the sweeps that have the dimension are used. Dask arrays are loaded.
    dim : str, optional
        Name of the member dimension. Default ``"member"``.
    percentiles : sequence of float, optional
        Percentiles (0-100, exclusive) in addition to the median. Default
        ``(5, 95)``.
    categorical : sequence of str, optional
        Names of variables that hold class codes (integers from 0): they get
        the mode and the agreement instead of the statistics.
    min_fraction : float, optional
        Gates where fewer than this fraction of the members is finite are
        NaN. Default 0.5.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. Default ``"auto"``.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        For each variable ``X`` with the dimension: ``X_mean``, ``X_median``,
        ``X_std`` (standard deviation with ``n - 1``) and ``X_p05``,
        ``X_p95`` for the default percentiles (see
        :mod:`radarx.retrieve.ensemble`), or ``X_mode`` and ``X_agreement``
        for class codes, as float32 on the coordinates of ``X`` without
        ``dim``. The percentiles interpolate linearly between the members
        (the default of :func:`numpy.percentile`). The input is not changed.

    Raises
    ------
    KeyError
        If no variable has the dimension.
    ValueError
        For percentiles outside (0, 100), ``min_fraction`` outside [0, 1] or an
        unknown engine.
    ImportError
        If ``engine="compiled"`` and the compiled kernel is not available.

    Examples
    --------
    >>> ens = radarx.retrieve.perturb_radar(sweep, members=16)  # doctest: +SKIP
    >>> stats = radarx.retrieve.ensemble_statistics(ens)  # doctest: +SKIP
    """
    args = {
        "dim": dim,
        "categorical": tuple(categorical),
        "min_fraction": min_fraction,
        "settings": _settings(2, percentiles, min_fraction, engine, n_threads),
    }
    return _over_sweeps(
        obj,
        lambda ds, stream: _statistics_sweep(ds, stream, args),
        "ensemble_statistics",
    )


# --------------------------------------------------------------------------
# prior knowledge of the DSD
# --------------------------------------------------------------------------

_RELATIONS = ("z", "kdp", "z_zdr", "a")
_PRIOR_KEYS = ("relation", "alpha", "mu", "tilt", "log10_nw")
_TILT_SD = 0.5
_LOG10_NW_SD = 0.3
_SCATTER_TEMPERATURE = 20.0


def _weighted_fit(y_values, columns, weights):
    """Weighted least squares of ``ln y`` on ``ln`` of the columns: (sd, coefficients)."""
    ok = np.isfinite(y_values) & (y_values > 0) & (weights > 0)
    for c in columns:
        ok &= np.isfinite(c) & (c > 0)
    w = weights[ok] / weights[ok].sum()
    y = np.log(y_values[ok])
    design = np.stack([np.ones(ok.sum())] + [np.log(c[ok]) for c in columns], axis=1)
    root = np.sqrt(w)
    beta = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)[0]
    resid = y - design @ beta
    return float(np.sqrt(np.sum(w * resid * resid))), beta


def _scatter_samples(prior, band, temperature):
    """DSDs of the prior with the radar variables, the rain rate and the rain weights."""
    prior = prior.transpose("dm", "mu")
    fwd = forward_grid(band, temperature)
    nodes, weights = np.polynomial.hermite_e.hermegauss(9)
    weights = weights / weights.sum()
    lognw = (
        prior["log10_nw_mean"].values[..., None]
        + prior["log10_nw_sd"].values[..., None] * nodes
    )
    nw = 10.0**lognw
    rate = fwd["RAIN_RATE"].values[..., None] * nw
    return {
        "z": 10.0 ** ((fwd["DBZH"].values[..., None] + 10.0 * lognw) / 10.0),
        "kdp": fwd["KDP"].values[..., None] * nw,
        "zdr": 10.0 ** (fwd["ZDR"].values[..., None] / 10.0) * np.ones_like(nw),
        "a": fwd["AH"].values[..., None] * nw,
        "rate": rate,
        "weight": prior["prior_mass"].values[..., None] * weights * rate,
    }


def _scatter_table(prior, band, temperature):
    """Scatter and fit of the four relations and of ``A/KDP`` under a prior."""
    s = _scatter_samples(prior, band, temperature)
    columns = {
        "z": [s["z"]],
        "kdp": [s["kdp"]],
        "z_zdr": [s["z"], s["zdr"]],
        "a": [s["a"]],
    }
    fits = [_weighted_fit(s["rate"], columns[k], s["weight"]) for k in _RELATIONS]
    return fits, *_alpha_statistics(s)


def _alpha_statistics(s):
    """Rain-weighted standard deviation of ``ln(A/KDP)`` and its geometric mean."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = s["a"] / s["kdp"]
    ok = np.isfinite(ratio) & (ratio > 0) & (s["kdp"] > 0)
    w = s["weight"][ok] / s["weight"][ok].sum()
    log_ratio = np.log(ratio[ok])
    mean = float(np.sum(w * log_ratio))
    return float(np.sqrt(np.sum(w * (log_ratio - mean) ** 2))), float(np.exp(mean))


@functools.lru_cache(maxsize=16)
def _named_scatter(name, band, temperature):
    return _scatter_table(dsd_prior(name), band, temperature)


def _prior_mu_sd(prior):
    """Standard deviation of the shape parameter under a prior."""
    prior = prior.transpose("dm", "mu")
    p = prior["prior_mass"].sum("dm")
    p = p / p.sum()
    mean = float((p * prior["mu"]).sum())
    return float(np.sqrt((p * (prior["mu"] - mean) ** 2).sum()))


@provenance("Scatter of rain rate relations about a power law under a DSD prior")
def relation_scatter(band="S", temperature=20.0, prior="generic"):
    """
    Scatter of the rain rate relations caused by the variability of the DSD.

    Draws the drop size distributions of a prior of :func:`dsd_prior` with
    their radar variables from the forward model of :func:`forward_grid`,
    fits each power law :math:`R(Z)`, :math:`R(K_{DP})`,
    :math:`R(Z, Z_{DR})` and :math:`R(A)` to them by least squares in
    :math:`\\ln R`, and returns the standard deviation of the residual. Each
    DSD is weighted by the rain rate it carries and its probability, as the
    optimal relations of Ryzhkov et al. (2014) are fitted. The same is done
    for the ratio :math:`\\alpha = A_H/K_{DP}`, whose variability limits the
    attenuation correction from the differential phase (Ryzhkov et al.
    2014, Sect. 5). The numbers are the defaults of the DSD part of the
    ensemble (``prior`` of :func:`ensemble_retrieval`).

    Parameters
    ----------
    band : {"S", "C", "X"}, optional
        Radar band of the forward model. Default ``"S"``.
    temperature : float, optional
        Rain temperature in degrees Celsius (0-30). Default 20.
    prior : {"generic", "perils2022"} or xarray.Dataset, optional
        The DSD prior, a name or the output of :func:`dsd_prior`. Default
        ``"generic"``, a radarx choice broad around the statistics of Bringi
        et al. (2003).

    Returns
    -------
    xarray.Dataset
        On ``relation`` (``z``, ``kdp``, ``z_zdr``, ``a``): ``ln_sd``, the
        standard deviation of :math:`\\ln R` about the fitted relation (0.3
        is a factor 1.35), the prefactor ``prefactor`` (mm h-1 in the units
        of the radar variable) and the exponent ``exponent`` of the fit, and
        ``zdr_exponent`` of the linear :math:`Z_{DR}` (``z_zdr`` only);
        and the scalars ``alpha`` (geometric mean of :math:`A_H/K_{DP}`,
        dB per degree) and ``alpha_ln_sd`` (standard deviation of its
        logarithm).

    Raises
    ------
    ValueError
        For an unknown band or prior name.

    Notes
    -----
    The result is the variability of the DSDs of the prior and the
    simplified forward model of :func:`forward_grid` (normalized gamma DSD,
    horizontal incidence); it is not the error of a particular relation
    against data, and not the bias of the default coefficients of
    :func:`rain_rate_z` and the others against the fit.

    References
    ----------
    Ryzhkov, A., M. Diederich, P. Zhang, and C. Simmer, 2014: Potential
    utilization of specific attenuation for rainfall estimation, mitigation of
    partial beam blockage, and radar networking. *J. Atmos. Oceanic
    Technol.*, **31** (3), 599-619,
    https://doi.org/10.1175/JTECH-D-13-00038.1

    Bringi, V. N., V. Chandrasekar, J. Hubbert, E. Gorgucci, W. L. Randeu, and
    M. Schoenhuber, 2003: Raindrop size distribution in different climatic
    regimes from disdrometer and dual-polarized radar analysis. *J. Atmos.
    Sci.*, **60** (2), 354-365,
    https://doi.org/10.1175/1520-0469(2003)060<0354:RSDIDC>2.0.CO;2

    Examples
    --------
    >>> sc = radarx.retrieve.relation_scatter("S")  # doctest: +SKIP
    >>> sc.ln_sd.sel(relation="z")  # doctest: +SKIP
    """
    band = _check_band(band)
    if isinstance(prior, str):
        fits, a_sd, a_mean = _named_scatter(prior, band, float(temperature))
    else:
        fits, a_sd, a_mean = _scatter_table(prior, band, float(temperature))
    dims = ("relation",)
    return xr.Dataset(
        {
            "ln_sd": (
                dims,
                [f[0] for f in fits],
                {"long_name": "standard deviation of ln R about the fit"},
            ),
            "prefactor": (dims, [float(np.exp(f[1][0])) for f in fits]),
            "exponent": (dims, [float(f[1][1]) for f in fits]),
            "zdr_exponent": (
                dims,
                [float(f[1][2]) if len(f[1]) > 2 else np.nan for f in fits],
            ),
            "alpha": ((), a_mean, {"units": "dB degree-1"}),
            "alpha_ln_sd": ((), a_sd, {"long_name": "standard deviation of ln(A/KDP)"}),
        },
        coords={"relation": list(_RELATIONS)},
        attrs={"band": band, "temperature": float(temperature)},
    )


# --------------------------------------------------------------------------
# retrievals
# --------------------------------------------------------------------------


def _check_prior(prior):
    """Standard deviations given by the user, or None for no change of the prior."""
    if prior is None or prior is False:
        return None
    if prior is True:
        return {}
    if not isinstance(prior, dict):
        raise TypeError("prior must be a bool or a dict of standard deviations")
    unknown = set(prior) - set(_PRIOR_KEYS)
    if unknown:
        raise ValueError(f"unknown prior entries {sorted(unknown)}; use {_PRIOR_KEYS}")
    if min(prior.values(), default=0.0) < 0:
        raise ValueError("the standard deviations of the prior must not be negative")
    return {k: float(v) for k, v in prior.items()}


_RATE_RELATION = {
    "rain_rate_z": "z",
    "rain_rate_kdp": "kdp",
    "rain_rate_z_zdr": "z_zdr",
    "rain_rate_a": "a",
}


def _prior_widths(user, run, ds):
    """Standard deviations of the prior perturbations that this retrieval uses."""
    widths = dict.fromkeys(_PRIOR_KEYS, 0.0)
    if user is None:
        return widths
    name = run.name
    if name in _RATE_RELATION:
        fits = _named_scatter("generic", run.band_for(ds), _SCATTER_TEMPERATURE)[0]
        default = fits[_RELATIONS.index(_RATE_RELATION[name])][0]
        widths["relation"] = user.get("relation", default)
    if run.att is not None:
        default = _named_scatter("generic", run.band_for(ds), _SCATTER_TEMPERATURE)[1]
        widths["alpha"] = user.get("alpha", default)
    if name == "dsd":
        widths["mu"] = user.get("mu", _prior_mu_sd(dsd_prior("generic")))
    if name == "dsd_bayesian":
        widths["tilt"] = user.get("tilt", _TILT_SD)
        widths["log10_nw"] = user.get("log10_nw", _LOG10_NW_SD)
    return widths


def _prior_draws(key, member, widths):
    """Random changes of the prior of one member, already scaled."""
    n3 = _antithetic(key, member, 3)
    n4 = _antithetic(key, member, 4)
    return {
        "relation": widths["relation"] * n3[0],
        "alpha": widths["alpha"] * n3[1],
        "mu": widths["mu"] * n3[2],
        "tilt_dm": widths["tilt"] * n4[0],
        "tilt_mu": widths["tilt"] * n4[1],
        "nw": widths["log10_nw"] * n4[2],
    }


class _Run:
    """The settings of one ensemble retrieval."""

    def __init__(self, **settings):
        self.__dict__.update(settings)

    def band_for(self, ds):
        if self.band is not None:
            return self.band
        if self.name in _RATE_RELATION and self.att is None:
            return "S"
        return _infer_band([ds])

    def band_kw(self):
        return {} if self.band is None else {"band": self.band}

    def call(self, func, ds, options=None, **extra):
        """Call ``func`` on a member with the options and the engine settings."""
        kwargs = dict(self.options if options is None else options)
        kwargs.update(extra)
        if self.callable:
            return func(ds, **kwargs)
        return func(ds, **kwargs, n_threads=self.n_threads, engine=self.engine)


def _with_kdp(ds, run):
    """The member with ``KDP`` estimated from its differential phase (if it has one)."""
    if "PHIDP" not in ds:
        return ds
    products = estimate_kdp(
        ds, **run.kdp_options, n_threads=run.n_threads, engine=run.engine
    )
    return ds.assign(dict(products.data_vars))


def _attenuation(ds, run, shift):
    """Products of the attenuation correction with the coefficients scaled by ``shift``."""
    band = run.band_for(ds)
    kwargs = dict(run.att)
    coefficients = attenuation_coefficients(
        band, kwargs.get("coefficients", "bringi1990")
    )
    factor = math.exp(shift)
    for key in ("alpha", "beta"):
        kwargs[key] = kwargs.get(key, coefficients[key]) * factor
    kwargs.setdefault("b", coefficients["b"])
    return correct_attenuation(
        ds,
        band=band,
        kdp_options=run.kdp_options or None,
        n_threads=run.n_threads,
        engine=run.engine,
        **kwargs,
    )


def _with_attenuation(ds, run, shift):
    """The member with the corrected reflectivity and differential reflectivity."""
    products = _attenuation(ds, run, shift)
    ds = ds.assign(DBZH=products["DBZH_corrected"], AH=products["AH"])
    if "ZDR_corrected" in products:
        ds = ds.assign(ZDR=products["ZDR_corrected"])
    return ds


def _chain_kdp(ds, run, draws):
    return run.call(estimate_kdp, ds)


def _chain_attenuation(ds, run, draws):
    return _attenuation(ds, run, draws["alpha"])


def _rate_chain(relation, function_name):
    def chain(ds, run, draws):
        func = getattr(_qpe, function_name)
        if relation == "kdp":
            ds = _with_kdp(ds, run)
        if run.att is not None:
            ds = _with_attenuation(ds, run, draws["alpha"])
        rate = run.call(func, ds, **run.band_kw())
        factor = math.exp(draws["relation"])
        return rate.copy(data=(rate.values * factor).astype(rate.dtype)).to_dataset()

    return chain


def _shifted_mu(options, shift):
    """Options of the DSD retrieval with the shape parameter shifted by ``shift``."""
    options = dict(options)
    if options.get("method", "constrained") == "normalized":
        options["mu"] = max(float(options.get("mu", 3.0)) + shift, -0.9)
    else:
        c2, c1, c0 = _relation(options.get("mu_lambda", "cao2008"))[1]
        options["mu_lambda"] = (c2, c1, c0 + shift)
    return options


def _chain_dsd(ds, run, draws):
    if run.att is not None:
        ds = _with_attenuation(ds, run, draws["alpha"])
    options = _shifted_mu(run.options, draws["mu"])
    res = run.call(_dsd, ds, options=options, **run.band_kw())
    with np.errstate(divide="ignore", invalid="ignore"):
        res["LOG10_NW"] = np.log10(res["NW"]).assign_attrs(
            long_name="log10 of the normalized intercept", units="log10(m-3 mm-1)"
        )
    return res


def _tilted_prior(prior, draws):
    """The prior with its mass tilted in ``D_m`` and ``mu`` and ``log10 N_w`` shifted."""
    prior = prior.transpose("dm", "mu").copy(deep=True)
    mass = prior["prior_mass"]
    out = {}
    for axis, tilt in (("dm", draws["tilt_dm"]), ("mu", draws["tilt_mu"])):
        marginal = mass.sum("mu" if axis == "dm" else "dm")
        mean = float((marginal * prior[axis]).sum())
        sd = float(np.sqrt((marginal * (prior[axis] - mean) ** 2).sum()))
        out[axis] = tilt * (prior[axis] - mean) / sd
    tilted = mass * np.exp(out["dm"] + out["mu"])
    prior["prior_mass"] = tilted / tilted.sum()
    prior["log10_nw_mean"] = prior["log10_nw_mean"] + draws["nw"]
    return prior


def _chain_bayes(ds, run, draws):
    if run.att is not None:
        ds = _with_attenuation(ds, run, draws["alpha"])
    options = dict(run.options)
    options.setdefault("quantiles", (0.5,))  # only the means and SDs are used
    prior = options.get("prior", "generic")
    if isinstance(prior, str):
        prior = dsd_prior(prior)
    if any(draws[k] for k in ("tilt_dm", "tilt_mu", "nw")):
        options["prior"] = _tilted_prior(prior, draws)
    return run.call(dsd_bayesian, ds, options=options, **run.band_kw())


def _chain_hid(ds, run, draws):
    return run.call(_hid, ds, scores=False)


def _chain_callable(ds, run, draws):
    return run.call(run.func, ds)


#: retrieval name -> (chain, variables returned by default, class-code variables)
_CHAINS = {
    "kdp": (_chain_kdp, ("KDP",), ()),
    "attenuation": (
        _chain_attenuation,
        ("DBZH_corrected", "ZDR_corrected", "AH"),
        (),
    ),
    "rain_rate_z": (_rate_chain("z", "rain_rate_z"), ("RAIN_RATE_Z",), ()),
    "rain_rate_kdp": (_rate_chain("kdp", "rain_rate_kdp"), ("RAIN_RATE_KDP",), ()),
    "rain_rate_z_zdr": (
        _rate_chain("z_zdr", "rain_rate_z_zdr"),
        ("RAIN_RATE_Z_ZDR",),
        (),
    ),
    "rain_rate_a": (_rate_chain("a", "rain_rate_a"), ("RAIN_RATE_A",), ()),
    "dsd": (_chain_dsd, ("DM", "LOG10_NW", "MU", "RAIN_RATE", "LWC"), ()),
    "dsd_bayesian": (
        _chain_bayes,
        tuple(
            f"{v}{suffix}"
            for suffix in ("", "_SD")
            for v in ("LOG10_NW", "DM", "MU", "RAIN_RATE", "LWC")
        ),
        (),
    ),
    "hid": (_chain_hid, ("HID",), ("HID",)),
}
RETRIEVALS = tuple(_CHAINS)


def _attenuation_options(attenuation, retrieval):
    """Options of the attenuation step, or None when there is no such step."""
    if attenuation is None or attenuation is False:
        return (
            {"method": "zphi"} if retrieval in ("attenuation", "rain_rate_a") else None
        )
    if attenuation is True:
        return {"method": "zphi"}
    if isinstance(attenuation, str):
        return {"method": attenuation}
    if isinstance(attenuation, dict):
        return dict(attenuation)
    raise TypeError("attenuation must be None, a method name or a dict of options")


def _member_dataset(ds, sw, values):
    """A member: the sweep with the perturbed fields under their standard names."""
    drop = [n for names in _NAMES.values() for n in names if n in ds]
    member = ds.drop_vars(drop)
    fields = {}
    for s, kind in enumerate(sw.kinds):
        da = sw.fields[kind]
        shaped = values[s].reshape(da.shape).astype(_out_dtype(da))
        fields[_CANONICAL[kind]] = da.copy(data=shaped)
    return member.assign(fields)


def _select(result, run):
    """The variables of a member's result that go into the statistics."""
    ds = result.to_dataset() if isinstance(result, xr.DataArray) else result
    if run.variables is not None:
        return ds[list(run.variables)]
    names = [v for v in run.defaults if v in ds] or list(ds.data_vars)
    return ds[names]


def _store(stacks, templates, result, member, members):
    for name, da in result.data_vars.items():
        if name not in stacks:
            stacks[name] = np.full((members,) + da.shape, np.nan, dtype=np.float32)
            templates[name] = da
        stacks[name][member] = da.values


def _ensemble_attrs(run, widths):
    parts = [f"{run.members} members, seed {run.seed}"]
    if run.calibration != (0.0, 0.0):
        parts.append(
            f"calibration offsets SD {run.calibration[0]:g} dB (Z), {run.calibration[1]:g} dB (ZDR)"
        )
    if run.noise is not None:
        parts.append(f"measurement noise of {run.noise['n_samples']:g} samples")
    active = {k: v for k, v in widths.items() if v}
    if active:
        parts.append("prior SD " + ", ".join(f"{k} {v:.3g}" for k, v in active.items()))
    return {
        "ensemble_members": int(run.members),
        "ensemble_seed": int(run.seed),
        "ensemble_method": "; ".join(parts),
    }


def _total_std(name, stacks, spread, settings):
    """Posterior and ensemble spread of a Bayesian product combined (total variance)."""
    sd = stacks[f"{name}_SD"].astype(np.float64)
    shape = sd.shape[1:]
    mean_var = _kernels.statistics(
        (sd**2).reshape(sd.shape[0], -1),
        [],
        settings["min_valid"],
        settings["use_compiled"],
        settings["n_threads"],
    )[0]
    total = np.sqrt(mean_var.reshape(shape) + np.asarray(spread, dtype=np.float64) ** 2)
    return total.astype(np.float32)


def _assemble(stacks, templates, run, widths):
    settings = {
        "percentiles": run.percentiles,
        "min_valid": _min_valid(run.members, run.min_fraction),
        "use_compiled": run.use_compiled,
        "n_threads": run.n_threads,
    }
    out = {}
    for name, stack in stacks.items():
        companion = run.posterior and name.endswith("_SD") and name[:-3] in stacks
        if companion:
            continue
        if name in run.categorical:
            out.update(_class_variables(name, templates[name], stack, settings))
        else:
            out.update(_stat_variables(name, templates[name], stack, settings))
        if run.posterior and f"{name}_SD" in stacks:
            out[f"{name}_total_std"] = (
                templates[name]
                .copy(
                    data=_total_std(name, stacks, out[f"{name}_std"].values, settings)
                )
                .assign_attrs(
                    _stat_attrs(
                        templates[name], name, "total standard deviation", run.members
                    )
                )
            )
        if run.keep_members:
            out[f"{name}_members"] = (
                templates[name]
                .expand_dims(member=np.arange(run.members))
                .copy(data=stack)
            )
    return xr.Dataset(out, attrs=_ensemble_attrs(run, widths))


def _ensemble_sweep(ds, stream, run):
    sw = _Sweep(ds, run.names, run.noise)
    key = make_key(run.seed, stream)
    widths = _prior_widths(run.prior, run, ds)
    stacks, templates = {}, {}
    for m in range(run.members):
        values = _perturb(
            sw, key, m, 1, run.calibration, run.use_compiled, run.n_threads
        )
        member = _member_dataset(ds, sw, values[0])
        result = run.chain(member, run, _prior_draws(key, m, widths))
        _store(stacks, templates, _select(result, run), m, run.members)
    return _assemble(stacks, templates, run, widths)


def _resolve_retrieval(retrieval):
    if callable(retrieval):
        return (
            _chain_callable,
            (),
            (),
            retrieval,
            getattr(retrieval, "__name__", "callable"),
        )
    if retrieval not in _CHAINS:
        raise ValueError(
            f"retrieval must be one of {RETRIEVALS} or a function, not {retrieval!r}"
        )
    chain, defaults, classes = _CHAINS[retrieval]
    return chain, defaults, classes, None, retrieval


@provenance(
    "Monte Carlo ensemble of a retrieval over calibration, measurement noise "
    "and DSD prior (error model of Melnikov and Zrnic, 2004)",
    extra_refs=("melnikov-2004", "giangrande-2004"),
)
def ensemble_retrieval(
    obj,
    retrieval,
    *,
    dbz=None,
    zdr=None,
    phidp=None,
    rhohv=None,
    field=None,
    members=32,
    seed=0,
    calibration=(1.0, 0.1),
    noise=True,
    prior=True,
    percentiles=(5, 95),
    variables=None,
    categorical=None,
    keep_members=False,
    min_fraction=0.5,
    band=None,
    attenuation=None,
    options=None,
    n_threads=None,
    engine="auto",
):
    """
    Run a retrieval on an ensemble of perturbed radar data.

    Every member is a copy of the radar fields with random calibration offsets
    and measurement noise (:func:`perturb_radar`) and, if ``prior`` is set,
    with a random change of the DSD prior of the retrieval. The retrievals
    of radarx run on each member, and the result holds the statistics of the
    products over the members, see :mod:`radarx.retrieve.ensemble`.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        A sweep, a grid or a volume with ``sweep_*`` groups (sweeps without
        a radar field are skipped), or a named field (DataArray). The fields
        are the raw measurements: the reflectivity and differential
        reflectivity as measured (they are corrected for attenuation inside
        the ensemble if ``attenuation`` is set), the differential phase
        before any processing and the correlation coefficient.
    retrieval : str or callable
        What runs on each member. One of ``"kdp"`` (:func:`estimate_kdp`
        from the perturbed phase, product ``KDP``), ``"attenuation"``
        (:func:`correct_attenuation`: ``DBZH_corrected``, ``ZDR_corrected``,
        ``AH``), ``"rain_rate_z"``, ``"rain_rate_kdp"`` (with ``KDP``
        estimated from the member's phase), ``"rain_rate_z_zdr"``,
        ``"rain_rate_a"`` (with ``AH`` from the attenuation correction),
        ``"dsd"`` (:func:`dsd`: ``DM``, ``LOG10_NW``, ``MU``,
        ``RAIN_RATE``, ``LWC``), ``"dsd_bayesian"`` (the posterior means of
        :func:`dsd_bayesian`) and ``"hid"`` (:func:`hid`: class codes
        ``HID``). Or a function ``f(dataset, **options)`` that takes a sweep
        with the perturbed fields under the names ``DBZH``, ``ZDR``,
        ``PHIDP``, ``RHOHV`` and ``VRADH`` and returns a Dataset or
        DataArray.
    dbz, zdr, phidp, rhohv, field : str, optional
        Names of the fields, as for :func:`perturb_radar`.
    members : int, optional
        Number of members. Default 32.
    seed : int, optional
        Seed of the random numbers. Default 0.
    calibration : (float, float) or None, optional
        Standard deviations (dB) of the calibration offsets of the
        reflectivity and the differential reflectivity. Default
        ``(1.0, 0.1)``.
    noise : bool or dict, optional
        Add the measurement noise. A dict sets ``n_samples``, ``snr``,
        ``spectrum_width`` and ``nyquist`` as for :func:`perturb_radar`.
        Default True.
    prior : bool or dict, optional
        Change the DSD prior of the retrieval for each member. True uses
        the defaults, a dict overrides the standard deviations ``relation``
        (the log of the rain rate prefactor, from :func:`relation_scatter`),
        ``alpha`` (the log of the attenuation coefficients, from
        :func:`relation_scatter`), ``mu`` (shape of the ``"dsd"`` retrieval;
        the standard deviation of :math:`\\mu` under the generic prior),
        ``tilt`` (0.5; tilt of the prior of ``"dsd_bayesian"`` in standard
        deviations of :math:`D_m` and :math:`\\mu`) and ``log10_nw`` (0.3;
        shift of the prior mean of :math:`\\log_{10} N_w`); the last two are
        radarx choices. False or None for no change. Only the entries that
        belong to the retrieval act. The draws are common to all gates of a
        sweep. Not used by ``"kdp"``, ``"hid"`` or a function.
    percentiles : sequence of float, optional
        Percentiles (0-100, exclusive) besides the median. Default ``(5, 95)``.
    variables : sequence of str, optional
        Variables of the retrieval to keep. Default: those listed for
        ``retrieval`` (the data variables of the result for a function).
    categorical : sequence of str, optional
        Variables that hold class codes (integers from 0): they get the
        mode and the agreement. Default: ``("HID",)`` for ``"hid"``, none
        otherwise.
    keep_members : bool, optional
        Also return every member of every product as ``<name>_members`` with
        the dimension ``member``. Default False.
    min_fraction : float, optional
        Gates where fewer than this fraction of the members is finite are NaN.
        Default 0.5.
    band : {"S", "C", "X"}, optional
        Radar band for the retrieval and for the attenuation correction. By
        default as the retrieval finds it (``"S"`` for the rain rates).
    attenuation : str or dict, optional
        Correct the reflectivity and differential reflectivity of each member
        for attenuation before the retrieval, with the method (``"zphi"`` or
        ``"linear"``) or a dict of options of :func:`correct_attenuation`.
        Always on for ``"attenuation"`` and ``"rain_rate_a"`` (``"zphi"``).
    options : dict, optional
        Options of the retrieval function (for ``"kdp"`` those of
        :func:`estimate_kdp`; ``kdp_options`` for the phase processing of the
        others; for ``"dsd"``, ``"dsd_bayesian"`` and ``"hid"``
        ``kdp="estimate"`` uses the phase of each member). Do not give the
        names of the fields.
    n_threads : int, optional
        Threads of the kernels. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. Default ``"auto"``.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        For each product ``X``: ``X_mean``, ``X_median``, ``X_std``,
        ``X_p05`` and ``X_p95`` (or ``X_mode`` and ``X_agreement``), float32
        on the coordinates of the product, and ``X_members`` with
        ``keep_members``. The attributes ``ensemble_members``,
        ``ensemble_seed`` and ``ensemble_method`` describe the ensemble. For
        a volume, a DataTree with one such node per sweep and the root of the
        input. The input is not changed.

    Raises
    ------
    KeyError
        If no sweep has a radar field, or a requested variable is not
        returned by the retrieval.
    ValueError
        For an unknown retrieval, option or engine, or ``members`` below 2.
    ImportError
        If ``engine="compiled"`` and the compiled kernel is not available.

    Notes
    -----
    The members run one after the other; each runs threaded in C++ over its
    gates or rays. The error model, the draws and their limits are in
    :mod:`radarx.retrieve.ensemble`. ``"dsd_bayesian"`` models errors of its
    own (``errors``) inside the posterior: the spread of its posterior means
    over the ensemble comes on top of the posterior standard deviation and is
    not a replacement for it.

    References
    ----------
    Melnikov, V. M., and D. S. Zrnic, 2004: Simultaneous transmission mode
    for the polarimetric WSR-88D: Statistical biases and standard deviations
    of polarimetric variables. NOAA/NSSL report, 84 pp.,
    https://www.nssl.noaa.gov/publications/wsr88d_reports/SHV_statistics.pdf

    Giangrande, S. E., A. V. Ryzhkov, V. M. Melnikov, and J. Krause, 2004:
    Calibration of the polarimetric NEXRAD radar using meteorological signals.
    Preprints, 11th Conf. on Aviation, Range, and Aerospace Meteorology,
    paper P5.15, https://ams.confex.com/ams/11aram22sls/techprogram/paper_81811.htm

    Ryzhkov, A., M. Diederich, P. Zhang, and C. Simmer, 2014: Potential
    utilization of specific attenuation for rainfall estimation, mitigation of
    partial beam blockage, and radar networking. *J. Atmos. Oceanic
    Technol.*, **31** (3), 599-619,
    https://doi.org/10.1175/JTECH-D-13-00038.1

    Salmon, J. K., M. A. Moraes, R. O. Dror, and D. E. Shaw, 2011: Parallel
    random numbers: as easy as 1, 2, 3. *Proc. 2011 Int. Conf. for High
    Performance Computing, Networking, Storage and Analysis*, 1-12,
    https://doi.org/10.1145/2063384.2063405

    Examples
    --------
    >>> ens = radarx.retrieve.ensemble_retrieval(
    ...     sweep, "rain_rate_kdp", members=32, seed=1
    ... )  # doctest: +SKIP
    >>> ens.RAIN_RATE_KDP_std  # doctest: +SKIP
    """
    chain, defaults, classes, func, name = _resolve_retrieval(retrieval)
    options = dict(options or {})
    run = _Run(
        chain=chain,
        defaults=defaults,
        func=func,
        callable=func is not None,
        name=name,
        names=_field_names(dbz, zdr, phidp, rhohv, field),
        members=_check_members(members),
        seed=seed,
        calibration=_check_calibration(calibration),
        noise=_noise_options(noise),
        prior=_check_prior(prior),
        percentiles=_check_percentiles(percentiles),
        variables=None if variables is None else tuple(variables),
        posterior=name == "dsd_bayesian" and variables is None,
        categorical=tuple(classes if categorical is None else categorical),
        keep_members=bool(keep_members),
        min_fraction=min_fraction,
        band=None if band is None else str(band).upper(),
        att=_attenuation_options(attenuation, name),
        kdp_options=dict(options.pop("kdp_options", None) or {}),
        options=options,
        n_threads=n_threads,
        engine=engine,
        use_compiled=_use_compiled(engine),
    )
    _min_valid(run.members, run.min_fraction)
    return _over_sweeps(
        obj, lambda ds, stream: _ensemble_sweep(ds, stream, run), "ensemble_retrieval"
    )


# --------------------------------------------------------------------------
# accessors
# --------------------------------------------------------------------------


@accessor_method("dataarray", "dataset", "datatree", name="perturb_radar")
def _perturb_radar_accessor(self, **kwargs):
    """
    Radar fields with random calibration and measurement errors.

    See :func:`radarx.retrieve.perturb_radar` for the parameters.

    Returns
    -------
    xarray.DataArray, xarray.Dataset or xarray.DataTree
        The perturbed fields with the dimension ``member``.
    """
    return perturb_radar(self.xarray_obj, **kwargs)


@accessor_method("dataarray", "dataset", "datatree", name="measurement_error")
def _measurement_error_accessor(self, **kwargs):
    """
    Standard deviation of the measurement errors of the radar variables.

    See :func:`radarx.retrieve.measurement_error` for the parameters.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``<name>_error`` for each field found.
    """
    return measurement_error(self.xarray_obj, **kwargs)


@accessor_method("dataarray", "dataset", "datatree", name="ensemble_retrieval")
def _ensemble_retrieval_accessor(self, retrieval, **kwargs):
    """
    Statistics of a retrieval over perturbed radar data.

    See :func:`radarx.retrieve.ensemble_retrieval` for the parameters.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``<name>_mean``, ``_median``, ``_std`` and the percentiles of each
        product.
    """
    return ensemble_retrieval(self.xarray_obj, retrieval, **kwargs)


@accessor_method("dataarray", "dataset", "datatree", name="ensemble_statistics")
def _ensemble_statistics_accessor(self, **kwargs):
    """
    Mean, spread and percentiles over the members of an ensemble.

    See :func:`radarx.retrieve.ensemble_statistics` for the parameters.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``<name>_mean``, ``_median``, ``_std`` and the percentiles of each
        variable with the member dimension.
    """
    return ensemble_statistics(self.xarray_obj, **kwargs)
