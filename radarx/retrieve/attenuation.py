#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Rain Attenuation and Differential Attenuation Correction
========================================================

Rain attenuates the signal of a radar at S, C and X band and attenuates the
horizontal and the vertical polarisation by different amounts. The measured
reflectivity :math:`Z_H` is too low behind rain, and so is the differential
reflectivity :math:`Z_{DR}`. The differential phase :math:`\\Phi_{DP}` is not
affected, and rain also produces it, so it measures how much rain a beam has
passed through. :func:`correct_attenuation` uses that.

With the specific attenuation :math:`A_H` (one way, dB km-1) and the specific
differential attenuation :math:`A_{DP} = A_H - A_V`, the two-way path
integrated attenuation and differential attenuation at range :math:`r` are

.. math::

    \\mathrm{PIA}(r) = 2 \\int_{r_0}^{r} A_H\\,ds, \\qquad
    \\mathrm{PIDA}(r) = 2 \\int_{r_0}^{r} A_{DP}\\,ds,

and the corrections are :math:`Z_H + \\mathrm{PIA}` and
:math:`Z_{DR} + \\mathrm{PIDA}`. Bringi et al. (1990) found that :math:`A_H`
and :math:`A_{DP}` are proportional to the specific differential phase
:math:`K_{DP}` at S, C and X band, :math:`A_H = \\alpha K_{DP}` and
:math:`A_{DP} = \\beta K_{DP}`, with :math:`\\alpha` and :math:`\\beta` in dB
per degree. Because :math:`\\Phi_{DP} = 2 \\int K_{DP}\\,ds`, the two-way
quantities follow from the phase alone, along a ray, without :math:`K_{DP}`:

.. math::

    \\mathrm{PIA}(r) = \\alpha\\,[\\Phi_{DP}(r) - \\Phi_{DP}(r_0)], \\qquad
    \\mathrm{PIDA}(r) = \\beta\\,[\\Phi_{DP}(r) - \\Phi_{DP}(r_0)].

Methods
-------
``"zphi"`` (default)
    The constrained solution of Testud et al. (2000). The specific attenuation
    is assumed to follow :math:`A_H = a Z^b`, where :math:`Z` is the
    attenuation-free reflectivity (mm\\ :sup:`6` m\\ :sup:`-3`) and :math:`b`
    is a constant. The profile of :math:`Z` is only known with the
    attenuation (the measured reflectivity :math:`Z_m`). The Hitschfeld and
    Bordan (1954) solution for a given :math:`a`,
    :math:`A_H(r) = a Z_m^b / [1 - 0.46\\, a b \\int_{r_0}^{r} Z_m^b ds]`,
    is unstable: the denominator tends to zero, and the correction to
    infinity, when :math:`a` is a little too large, and at S, C and X band
    the retrieved profile depends strongly on :math:`a`. ZPHI fixes :math:`a`
    by requiring that the two-way attenuation along the whole path equals the
    value from the phase, :math:`\\mathrm{PIA}(r_0, r_1) = \\alpha\\,\\Delta
    \\Phi_{DP}`. This gives (Ryzhkov et al. 2014, Eqs. 12-15, who write the
    Testud et al. solution in this form)

    .. math::

        A_H(r) = \\frac{Z_m(r)^b\\, C(b, \\mathrm{PIA})}
        {I(r_0, r_1) + C(b, \\mathrm{PIA})\\, I(r, r_1)}, \\quad
        C = e^{0.23\\, b\\, \\mathrm{PIA}} - 1, \\quad
        I(r, r_1) = 0.46\\, b \\int_r^{r_1} Z_m(s)^b ds,

    which does not depend on :math:`a` and is bounded: the denominator is at
    least :math:`I(r_0, r_1) > 0`. The unconstrained solution is the limit of
    ZPHI for a path attenuation that is not known, so the phase constraint is
    what makes the correction stable. With :math:`r_0` and :math:`r_1` the
    first and last rain gate of the ray, the formula gives the attenuation of
    the whole path; as in Ryzhkov et al. (2014) the ray is not cut into
    segments (the segmentation of Testud et al. 2000, which lets :math:`a`
    change along the ray, is not implemented). The factor of the formulas is
    :math:`0.46 = 0.2 \\ln 10 = 0.4605` and :math:`0.23 = 0.1 \\ln 10`; radarx
    uses the exact values and not the rounded ones.
``"linear"``
    :math:`\\mathrm{PIA}(r) = \\alpha\\,[\\Phi_{DP}(r) - \\Phi_{DP}(r_0)]`
    from the phase at every gate (Bringi et al. 1990). It does not need the
    reflectivity profile and gives the correction at every gate directly, but
    it follows the noise of the phase.

Coefficients
------------
:func:`attenuation_coefficients` gives :math:`\\alpha` and :math:`\\beta` (dB
per degree) and :math:`b` for S, C and X band. The default values of
:math:`\\alpha` and :math:`\\beta` are the mean linear relations of Bringi et
al. (1990, Table 1, gamma drop size distributions, 15 degrees C, 3.0, 5.5 and
10.0 GHz): 0.016 and 0.00367 at S band, 0.054 and 0.0157 at C band, 0.25 and
0.05 at X band. The exponent :math:`b` of :math:`A_H \\propto Z^b` is not in
that paper. radarx uses 0.78 at X band, the value Park et al. (2005, Part I)
give for 15 degrees C (from their abstract), and 0.71 at S band and 0.84 at C
band from its own fit to the T-matrix tables of :func:`radarx.retrieve.scattering_table`
(a log-log regression of :math:`A_H` on :math:`Z` for 15 degrees C, 6000
normalized-gamma drop size distributions with :math:`D_m` of 0.9-2.8 mm,
:math:`\\log_{10} N_w` of 2.8-4.4 and :math:`\\mu` of 0-6, restricted to 15-55
dBZ; its X-band value is 0.80). ``coefficients="tmatrix"`` gives
:math:`\\alpha` and :math:`\\beta` from the same fit (S band 0.016 and 0.0031,
C band 0.084 and 0.021, X band 0.315 and 0.050). They agree with Bringi et al.
(1990) at S band and differ most at C band, where the Bringi et al. value is
smaller by a third; the coefficients depend on the drop size distribution and
on the temperature (Ryzhkov et al. 2014, Fig. 5), and the mean value of
Park et al. (2005, Part I) at X band is 0.254 dB per degree with a range of
0.139-0.335. Because the result scales with :math:`\\alpha`, pass your own
``alpha``, ``beta`` and ``b`` where a better value is known for the radar and
the rain.

The scattering quantities behind the coefficients come from the T-matrix
tables of :func:`radarx.retrieve.scattering_table` (with the
:math:`|K_w|^2 = 0.93` convention of the table), not from the Rayleigh
functions of :mod:`radarx.fundamentals.scattering`, which are for spheres and
give no differential terms.

Choices of radarx
-----------------
The papers leave these open; the defaults are radarx's.

* **Gates of the rain path.** A gate belongs to the path of its ray where the
  reflectivity and the phase are finite, the reflectivity is at least
  ``min_dbz`` (default 0 dBZ), :math:`\\rho_{hv}` (if given) is at least
  ``min_rhohv`` (default 0.9, a little above the 0.85 of
  :func:`radarx.retrieve.estimate_kdp`, which is a limit for the phase and
  not for the reflectivity), the user mask (``mask``) is true and the gate is
  below the melting level (``melting``). :math:`r_0` and :math:`r_1` are the
  first and last of these gates. Gates in between that fail the tests get no
  attenuation of their own but are inside the path.
* **Melting layer.** The relations hold in rain. In the melting layer the
  reflectivity is enhanced and above it the particles are ice, which does not
  attenuate much and does not add to :math:`\\Phi_{DP}` as rain does. Pass the
  height of the melting level (``melting``, metres above sea level, e.g. from
  :func:`radarx.retrieve.melting_layer`) to leave the gates at and above it
  out of the path. There is no cutoff by default, so a ray through ice
  must be masked by the caller. Gates above the path get the attenuation of
  the whole path.
* **Phase.** The phase is used as it is, without further smoothing: the
  correction needs :math:`\\Phi_{DP}` with the system offset removed,
  unfolded and filtered in range, which is what
  :func:`radarx.retrieve.estimate_kdp` returns as ``PHIDP_processed``. When
  ``phidp`` is not given, ``PHIDP_processed`` is taken if the dataset has it,
  and otherwise :func:`radarx.retrieve.estimate_kdp` (its defaults) runs on
  the raw phase. A phase that you pass is assumed to be processed. The span
  :math:`\\Delta\\Phi_{DP}` is the difference of the values at the last
  and the first gate of the path.
* **Specific differential phase.** Neither method needs :math:`K_{DP}`. It is
  not computed for the correction, so a bad :math:`K_{DP}` cannot harm it; the
  method needs the phase instead, and a ray without a usable phase gets no
  correction.
* **Small spans.** A ray gets a correction only if :math:`\\Delta\\Phi_{DP}` is
  at least ``min_dphi`` (default 3 degrees; Ryzhkov et al. (2014) find
  :math:`A_H` robust for spans of 2-4 degrees and more). Other rays have a
  path attenuation of zero and no specific attenuation (``NaN``).
* **Linear method.** The attenuation of a gate is not allowed to fall
  below the attenuation of an earlier gate or below zero (it is the running
  maximum of the non-negative values) and :math:`\\Phi_{DP}` is interpolated
  linearly across gates that are not in the path. :math:`A_H` is derived from
  it by a central difference over three gates.
* **Differential attenuation.** :math:`A_{DP} = (\\beta / \\alpha) A_H` for
  both methods, so that :math:`\\mathrm{PIDA} = (\\beta / \\alpha)\\,
  \\mathrm{PIA}`; the two coefficients and their ratio are constants of the
  band. The ratio of the specific differential attenuation to the specific
  attenuation varies with the drop size distribution (Ryzhkov et al. 2014,
  Table 2: 0.12-0.36 at S, 0.14-0.39 at C and 0.14-0.19 at X band).
* **Beyond the path.** Gates after the last rain gate keep the attenuation of
  the whole path; gates before the first have none.
* **Discretisation.** The gate-integrated form of the ZPHI solution is used,
  :math:`\\int A_H ds = \\ln(F_{\\mathrm{in}} / F_{\\mathrm{out}}) / (0.46 b)`
  over a gate with :math:`F(r) = I(r_0, r_1) + C\\, I(r, r_1)`. The attenuation
  of the whole path is then the constraint to rounding for any gate length.

Gates must be uniformly spaced; the correction does not need the geometry
otherwise. The numerical work runs in a compiled, multithreaded C++ kernel
(over rays); if it is not available an equivalent NumPy implementation is
used. A reflectivity that is a dask array stays lazy.

References
----------
.. [1] Testud, J., E. Le Bouar, E. Obligis, and M. Ali-Mehenni, 2000: The
   rain profiling algorithm applied to polarimetric weather radar. *J.
   Atmos. Oceanic Technol.*, **17** (3), 332-356,
   https://doi.org/10.1175/1520-0426(2000)017<0332:TRPAAT>2.0.CO;2
.. [2] Bringi, V. N., V. Chandrasekar, N. Balakrishnan, and D. S. Zrnic, 1990:
   An examination of propagation effects in rainfall on radar measurements
   at microwave frequencies. *J. Atmos. Oceanic Technol.*, **7** (6), 829-840,
   https://doi.org/10.1175/1520-0426(1990)007<0829:AEOPEI>2.0.CO;2
.. [3] Ryzhkov, A., M. Diederich, P. Zhang, and C. Simmer, 2014: Potential
   utilization of specific attenuation for rainfall estimation, mitigation of
   partial beam blockage, and radar networking. *J. Atmos. Oceanic Technol.*,
   **31** (3), 599-619, https://doi.org/10.1175/JTECH-D-13-00038.1
.. [4] Park, S.-G., V. N. Bringi, V. Chandrasekar, M. Maki, and K. Iwanami,
   2005: Correction of radar reflectivity and differential reflectivity for
   rain attenuation at X band. Part I: Theoretical and empirical basis. *J.
   Atmos. Oceanic Technol.*, **22** (11), 1621-1632,
   https://doi.org/10.1175/JTECH1803.1
.. [5] Hitschfeld, W., and J. Bordan, 1954: Errors inherent in the radar
   measurement of rainfall at attenuating wavelengths. *J. Meteor.*, **11**
   (1), 58-67, https://doi.org/10.1175/1520-0469(1954)011<0058:EIITRM>2.0.CO;2

.. autosummary::
   :nosignatures:
   :toctree: generated/

   correct_attenuation
   attenuation_coefficients
"""

from __future__ import annotations

__all__ = ["correct_attenuation", "attenuation_coefficients"]

import numpy as np
import xarray as xr

from .._compat import renamed_arguments
from .._dtypes import fields_as_float32
from .._provenance import provenance
from .._registry import accessor_method
from ..fundamentals.geometry import beam_center_height
from ._fieldtools import detach
from ._products import product_tree
from .dsd import _infer_band
from .kdp import _PHIDP_NAMES, _RHOHV_NAMES, _gate_spacing, estimate_kdp

try:
    from . import _attenuation

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _attenuation = None
    HAS_COMPILED_KERNEL = False

METHODS = {"zphi": 0, "linear": 1}
BANDS = ("S", "C", "X")

# 0.46 and 0.23 of Testud et al. (2000) are 0.2 ln(10) and 0.1 ln(10)
_LN10_10 = 0.23025850929940457
_KAPPA = 2.0 * _LN10_10
_MAX_EXPONENT = 300.0  # guards exp() for a huge path attenuation

# alpha and beta [dB/deg]: Bringi et al. (1990), Table 1; b: see the module
# docstring.
_COEFFICIENTS = {
    "bringi1990": {
        "S": {"alpha": 0.016, "beta": 0.00367, "b": 0.71},
        "C": {"alpha": 0.054, "beta": 0.0157, "b": 0.84},
        "X": {"alpha": 0.25, "beta": 0.05, "b": 0.78},
    },
    "tmatrix": {
        "S": {"alpha": 0.016, "beta": 0.0031, "b": 0.71},
        "C": {"alpha": 0.084, "beta": 0.021, "b": 0.84},
        "X": {"alpha": 0.315, "beta": 0.050, "b": 0.78},
    },
}

_DBZH_NAMES = ("DBZH", "DBZ", "reflectivity", "corrected_reflectivity")
_ZDR_NAMES = ("ZDR", "differential_reflectivity", "corrected_differential_reflectivity")
_RANGE = "range"


# --------------------------------------------------------------------------
# coefficients
# --------------------------------------------------------------------------


def attenuation_coefficients(band, coefficients="bringi1990"):
    """
    Coefficients of the attenuation relations for a radar band.

    Parameters
    ----------
    band : {"S", "C", "X"}
        Radar band.
    coefficients : {"bringi1990", "tmatrix"}, optional
        ``"bringi1990"`` (default): :math:`\\alpha` and :math:`\\beta` are
        the mean linear relations of Bringi et al. (1990), Table 1. ``"tmatrix"``:
        radarx's own fit to its T-matrix scattering tables. See
        :mod:`radarx.retrieve.attenuation` for the exponent.

    Returns
    -------
    dict
        ``alpha`` (dB/deg, :math:`A_H = \\alpha K_{DP}`), ``beta`` (dB/deg,
        :math:`A_{DP} = \\beta K_{DP}`) and ``b`` (exponent of
        :math:`A_H = a Z^b`).

    Raises
    ------
    ValueError
        For an unknown band or set.

    References
    ----------
    .. [1] Bringi, V. N., V. Chandrasekar, N. Balakrishnan, and D. S. Zrnic,
       1990: An examination of propagation effects in rainfall on radar
       measurements at microwave frequencies. *J. Atmos. Oceanic Technol.*,
       **7** (6), 829-840,
       https://doi.org/10.1175/1520-0426(1990)007<0829:AEOPEI>2.0.CO;2
    .. [2] Park, S.-G., V. N. Bringi, V. Chandrasekar, M. Maki, and K.
       Iwanami, 2005: Correction of radar reflectivity and differential
       reflectivity for rain attenuation at X band. Part I: Theoretical and
       empirical basis. *J. Atmos. Oceanic Technol.*, **22** (11), 1621-1632,
       https://doi.org/10.1175/JTECH1803.1

    Examples
    --------
    >>> attenuation_coefficients("C")["alpha"]
    0.054
    """
    if coefficients not in _COEFFICIENTS:
        raise ValueError(
            f"coefficients must be one of {sorted(_COEFFICIENTS)}, not {coefficients!r}"
        )
    key = str(band).upper()
    if key not in BANDS:
        raise ValueError(f"band must be one of {BANDS}, not {band!r}")
    return dict(_COEFFICIENTS[coefficients][key])


# --------------------------------------------------------------------------
# NumPy reference of the kernel
# --------------------------------------------------------------------------


def _path_numpy(dbz, phidp, valid):
    ok = valid.astype(bool) & np.isfinite(dbz) & np.isfinite(phidp)
    ng = ok.shape[1]
    rows = np.arange(ok.shape[0])
    any_ok = ok.any(axis=1)
    i0 = np.argmax(ok, axis=1)
    i1 = ng - 1 - np.argmax(ok[:, ::-1], axis=1)
    with np.errstate(invalid="ignore"):
        dphi = np.where(any_ok, phidp[rows, i1] - phidp[rows, i0], np.nan)
    return ok, i0, i1, dphi


def _zphi_numpy(dbz, ok, dr, pia_total, b):
    """Gate-integrated ZPHI on rays that have a path (same steps as the kernel)."""
    x = np.minimum(0.5 * _KAPPA * b * pia_total, _MAX_EXPONENT)
    c = np.expm1(x)[:, None]
    scale = _KAPPA * b * dr
    with np.errstate(over="ignore", invalid="ignore"):
        s = np.where(ok, scale * np.exp(b * _LN10_10 * np.where(ok, dbz, 0.0)), 0.0)
        incl = np.cumsum(s[:, ::-1], axis=1)[:, ::-1]
        tail = np.concatenate([incl[:, 1:], np.zeros((s.shape[0], 1))], axis=1)
        total = incl[:, :1]
        ah = np.log1p(c * s / (total + c * tail)) / scale
    inc = np.where(ok, ah, 0.0) * dr
    csum = np.cumsum(inc, axis=1)
    before = np.concatenate([np.zeros((s.shape[0], 1)), csum[:, :-1]], axis=1)
    pia = 2.0 * (before + 0.5 * inc)
    return np.where(ok, ah, np.nan), pia


def _linear_numpy(phidp, ok, i0, i1, dr, alpha, max_pia):
    nr, ng = ok.shape
    ar = np.arange(ng)[None, :]
    p = np.maximum.accumulate(np.where(ok, ar, -1), axis=1)
    q = np.minimum.accumulate(np.where(ok, ar, ng)[:, ::-1], axis=1)[:, ::-1]
    pc = np.clip(p, 0, ng - 1)
    qc = np.clip(q, 0, ng - 1)
    php = np.take_along_axis(phidp, pc, axis=1)
    phq = np.take_along_axis(phidp, qc, axis=1)
    span = (q - p).astype(np.float64)
    weight = np.divide(
        (ar - p).astype(np.float64), span, out=np.zeros(ok.shape), where=span != 0
    )
    phif = np.where(ok, phidp, php + (phq - php) * weight)
    inside = (ar >= i0[:, None]) & (ar <= i1[:, None])
    ref = np.take_along_axis(phidp, i0[:, None], axis=1)
    raw = np.where(inside, np.maximum(alpha * (phif - ref), 0.0), 0.0)
    pia = np.maximum.accumulate(raw, axis=1)
    if np.isfinite(max_pia):
        pia = np.minimum(pia, max_pia)
    lo = np.maximum(ar - 1, i0[:, None])
    hi = np.minimum(ar + 1, i1[:, None])
    gap = (hi - lo).astype(np.float64)
    diff = np.take_along_axis(pia, hi, axis=1) - np.take_along_axis(pia, lo, axis=1)
    ah = np.divide(diff, (2.0 * dr) * gap, out=np.zeros(ok.shape), where=gap != 0)
    return np.where(ok, ah, np.nan), pia


def _correct_numpy(dbz, phidp, valid, dr, method, alpha, b, min_dphi, max_pia):
    """
    NumPy implementation of the kernel ``_attenuation.correct`` (same results).

    ``dbz``, ``phidp`` and ``valid`` have the shape (rays, gates). Returns the
    specific attenuation (rays, gates; NaN outside the rain path), the
    path-integrated attenuation (rays, gates) and the phase span (rays).
    """
    nr, ng = dbz.shape
    ok, i0, i1, dphi = _path_numpy(dbz, phidp, valid)
    ah = np.full((nr, ng), np.nan)
    pia = np.zeros((nr, ng))
    with np.errstate(invalid="ignore"):
        active = np.isfinite(dphi) & (dphi >= min_dphi)
    if not active.any():
        return ah, pia, dphi
    sub = np.flatnonzero(active)
    if method == 0:
        total = alpha * dphi[sub]
        if np.isfinite(max_pia):
            total = np.minimum(total, max_pia)
        ah[sub], pia[sub] = _zphi_numpy(dbz[sub], ok[sub], dr, total, b)
    else:
        ah[sub], pia[sub] = _linear_numpy(
            phidp[sub], ok[sub], i0[sub], i1[sub], dr, alpha, max_pia
        )
        # gates after the path keep the attenuation of the path (running max)
    return ah, pia, dphi


def _use_compiled(engine):
    """Whether to run the compiled kernel for the requested ``engine``."""
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled attenuation kernel is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


def _block(
    dbz, phidp, valid, dr, method, alpha, b, min_dphi, max_pia, n_threads, compiled
):
    """Kernel on arrays whose last axis is the range; any leading axes."""
    shape = dbz.shape
    n = shape[-1]
    z = np.ascontiguousarray(dbz.reshape(-1, n), dtype=np.float64)
    p = np.ascontiguousarray(phidp.reshape(-1, n), dtype=np.float64)
    v = np.ascontiguousarray(valid.reshape(-1, n), dtype=np.uint8)
    if z.shape[0] == 0:
        empty = np.zeros(shape)
        return empty, empty.copy(), np.zeros(shape[:-1])
    cap = np.nan if max_pia is None else float(max_pia)
    if compiled:
        ah, pia, dphi = _attenuation.correct(
            z, p, v, dr, METHODS[method], alpha, b, min_dphi, cap, int(n_threads or 0)
        )
    else:
        ah, pia, dphi = _correct_numpy(
            z, p, v, dr, METHODS[method], alpha, b, min_dphi, cap
        )
    return ah.reshape(shape), pia.reshape(shape), dphi.reshape(shape[:-1])


# --------------------------------------------------------------------------
# xarray layer
# --------------------------------------------------------------------------


def _coefficients(band, coefficients, alpha, beta, b):
    """The coefficients used: user values, the rest from the band."""
    if alpha is None or beta is None or b is None:
        base = attenuation_coefficients(band, coefficients)
    else:
        base = {}
    alpha = base.get("alpha") if alpha is None else float(alpha)
    beta = base.get("beta") if beta is None else float(beta)
    b = base.get("b") if b is None else float(b)
    if not alpha > 0:
        raise ValueError(f"alpha must be positive, not {alpha!r}")
    if beta < 0:
        raise ValueError(f"beta must not be negative, not {beta!r}")
    if not 0 < b <= 2:
        raise ValueError(f"b must be in (0, 2], not {b!r}")
    return alpha, beta, b


def _field(ds, value, candidates, required, what):
    """DataArray from a name, a DataArray, or the first of ``candidates``."""
    if isinstance(value, xr.DataArray):
        return value
    if value is not None:
        if ds is None or value not in ds:
            raise KeyError(f"{value!r} is not in the dataset")
        return ds[value]
    if ds is not None:
        for cand in candidates:
            if cand in ds:
                return ds[cand]
    if required:
        raise KeyError(f"no {what} found among {candidates}; pass its name")
    return None


def _gate_height(da, ds):
    """Beam height (m above sea level) on the gates of ``da``."""
    for src in (da, ds):
        if src is not None and "z" in src.coords and _RANGE in src["z"].dims:
            return src["z"]
    elevation = None
    altitude = 0.0
    for src in (da, ds):
        if src is None:
            continue
        if elevation is None:
            for name in ("elevation", "sweep_fixed_angle"):
                if name in src.coords or name in getattr(src, "data_vars", {}):
                    elevation = src[name]
                    break
        if "altitude" in src.coords or "altitude" in getattr(src, "data_vars", {}):
            altitude = float(np.asarray(src["altitude"]).ravel()[0])
    if elevation is None:
        raise ValueError(
            "melting needs the beam height: give an elevation or a 'z' coordinate"
        )
    return beam_center_height(da[_RANGE], elevation, radar_height=altitude)


def _mask_of(mask, ds):
    if mask is None:
        return None
    return ds[mask] if isinstance(mask, str) else mask


def _rain_gates(dbz, phidp, rhohv, mask, melting, ds, min_dbz, min_rhohv):
    """Boolean field of the gates that can belong to the rain path."""
    valid = np.isfinite(dbz) & np.isfinite(phidp) & (dbz >= min_dbz)
    if rhohv is not None:
        valid = valid & (rhohv >= min_rhohv)
    if mask is not None:
        valid = valid & mask.astype(bool)
    if melting is not None:
        valid = valid & (_gate_height(dbz, ds) < melting)
    return valid


def _attrs(units, long_name):
    return {"long_name": long_name, "units": units}


def _correct_fields(
    dbz,
    phidp,
    zdr,
    rhohv,
    mask,
    melting,
    ds,
    method,
    alpha,
    beta,
    b,
    min_dphi,
    max_pia,
    min_dbz,
    min_rhohv,
    n_threads,
    compiled,
    summary,
):
    """The products of one sweep from resolved fields."""
    if _RANGE not in dbz.dims:
        raise ValueError(f"the reflectivity needs a {_RANGE!r} dimension")
    dr = _gate_spacing(dbz[_RANGE].values)
    phidp = phidp.broadcast_like(dbz) if phidp.dims != dbz.dims else phidp
    valid = _rain_gates(dbz, phidp, rhohv, mask, melting, ds, min_dbz, min_rhohv)

    def run(z, p, v):
        return _block(
            z, p, v, dr, method, alpha, b, min_dphi, max_pia, n_threads, compiled
        )

    ah, pia, dphi = xr.apply_ufunc(
        run,
        *detach(dbz, phidp, valid),
        input_core_dims=[[_RANGE]] * 3,
        output_core_dims=[[_RANGE], [_RANGE], []],
        dask="parallelized",
        output_dtypes=[np.float64] * 3,
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    dims = dbz.dims
    ah = ah.transpose(*dims)
    pia = pia.transpose(*dims)
    gamma = beta / alpha
    pida = gamma * pia
    adp = gamma * ah
    out = {}
    corrected = dbz + pia
    corrected.attrs = {
        "standard_name": "equivalent_reflectivity_factor",
        "long_name": "Reflectivity H corrected for rain attenuation",
        "units": "dBZ",
        "comment": f"{dbz.name or 'reflectivity'} plus PIA ({summary})",
    }
    out["DBZH_corrected"] = corrected
    if zdr is not None:
        zdr = zdr.broadcast_like(dbz) if zdr.dims != dbz.dims else zdr
        zc = zdr.transpose(*dims) + pida
        zc.attrs = {
            "standard_name": "radar_differential_reflectivity_hv",
            "long_name": "Differential reflectivity corrected for differential attenuation",
            "units": "dB",
            "comment": f"{zdr.name or 'ZDR'} plus PIDA ({summary})",
        }
        out["ZDR_corrected"] = zc
    ah.attrs = _attrs(
        "dB km-1", "Specific attenuation H (one way), NaN outside the rain path"
    )
    adp.attrs = _attrs(
        "dB km-1",
        "Specific differential attenuation (one way), NaN outside the rain path",
    )
    pia.attrs = _attrs("dB", "Two-way path-integrated attenuation H at the gate")
    pida.attrs = _attrs(
        "dB", "Two-way path-integrated differential attenuation at the gate"
    )
    dphi.attrs = _attrs("degree", "Differential phase span of the rain path")
    out.update({"AH": ah, "ADP": adp, "PIA": pia, "PIDA": pida, "DELTA_PHIDP": dphi})
    result = xr.Dataset(out)
    result.attrs = {
        "attenuation_method": method,
        "attenuation_alpha": alpha,
        "attenuation_beta": beta,
        "attenuation_b": b,
    }
    return fields_as_float32(result)


def _sweep_phidp(ds, phidp, dbz_name, rhohv_name, kdp_options):
    """A processed phase: the field asked for, PHIDP_processed, or estimated."""
    if isinstance(phidp, xr.DataArray):
        return phidp
    if phidp is not None:
        if phidp not in ds:
            raise KeyError(f"{phidp!r} is not in the dataset")
        return ds[phidp]
    if "PHIDP_processed" in ds:
        return ds["PHIDP_processed"]
    raw = next((n for n in _PHIDP_NAMES if n in ds), None)
    if raw is None:
        return None
    names = {"dbz": dbz_name}
    if rhohv_name is not None:
        names["rhohv"] = rhohv_name
    kdp = estimate_kdp(ds, phidp=raw, **{**names, **dict(kdp_options or {})})
    return kdp["PHIDP_processed"]


def _resolve_sweep(ds, dbzh, zdr, phidp, rhohv, mask, kdp_options, required):
    """Fields of one sweep dataset; None where a field is missing and not required."""
    dbz = _field(ds, dbzh, _DBZH_NAMES, required, "reflectivity")
    if dbz is None:
        return None
    rho = _field(ds, rhohv, _RHOHV_NAMES, False, "correlation coefficient")
    zdr_da = _field(ds, zdr, _ZDR_NAMES, False, "differential reflectivity")
    # names for estimate_kdp: only fields that are in the dataset
    ds_work = ds
    extras = {}
    for da, label in ((dbz, "_radarx_dbzh"), (rho, "_radarx_rhohv")):
        if da is not None and (da.name not in ds or ds[da.name] is not da):
            extras[label] = da
    if extras:
        ds_work = ds.assign({k: v.reset_coords(drop=True) for k, v in extras.items()})
    dbz_name = "_radarx_dbzh" if "_radarx_dbzh" in extras else dbz.name
    rho_name = None
    if rho is not None:
        rho_name = "_radarx_rhohv" if "_radarx_rhohv" in extras else rho.name
    phi = _sweep_phidp(ds_work, phidp, dbz_name, rho_name, kdp_options)
    if phi is None:
        if required:
            raise KeyError(
                "no differential phase found (PHIDP_processed or one of "
                f"{_PHIDP_NAMES}); pass phidp"
            )
        return None
    return dbz, phi, zdr_da, rho, _mask_of(mask, ds)


def _band_of(band, objs):
    if band is not None:
        key = str(band).upper()
        if key not in BANDS:
            raise ValueError(f"band must be one of {BANDS}, not {band!r}")
        return key
    return _infer_band(objs)


@renamed_arguments(dbzh="dbz", dbz_min="min_dbz", rhohv_min="min_rhohv")
@provenance(
    "Rain attenuation and differential attenuation correction from PhiDP "
    "(constrained ZPHI after Testud et al., or the linear PhiDP relation of "
    "Bringi et al.)"
)
def correct_attenuation(
    obj,
    dbz=None,
    zdr=None,
    phidp=None,
    rhohv=None,
    *,
    method="zphi",
    band=None,
    coefficients="bringi1990",
    alpha=None,
    beta=None,
    b=None,
    min_dphi=3.0,
    max_pia=None,
    min_dbz=0.0,
    min_rhohv=0.9,
    melting=None,
    mask=None,
    kdp_options=None,
    n_threads=None,
    engine="auto",
):
    """
    Correct the reflectivity and the differential reflectivity for rain attenuation.

    Uses the path-integrated differential phase, with the constrained ZPHI
    solution of Testud et al. (2000) or the linear relation of Bringi et al.
    (1990). The equations, the coefficients for S, C and X band and the
    choices of radarx are in :mod:`radarx.retrieve.attenuation`.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        A sweep on ``(azimuth, range)`` (or any dimensions with a ``range``
        dimension; ``range`` in metres, uniformly spaced), a volume with
        ``sweep_*`` groups, or the reflectivity of a sweep as a DataArray
        (then give ``phidp``, and ``zdr`` for the differential reflectivity,
        as DataArrays). The reflectivity may be a dask array; the result stays
        lazy.
    dbz, zdr, phidp, rhohv : str or xarray.DataArray, optional
        The reflectivity (dBZ), the differential reflectivity (dB), the
        **processed** differential phase (degrees, increasing with range in
        rain, offset removed, see below) and the copolar correlation
        coefficient: a field name or a DataArray. By default the first name
        found is used: ``DBZH``, ``DBZ``, ``reflectivity``,
        ``corrected_reflectivity``; ``ZDR``, ``differential_reflectivity``,
        ``corrected_differential_reflectivity`` (the differential reflectivity
        is corrected only if there is one); ``RHOHV``,
        ``cross_correlation_ratio``, ``copol_correlation_coeff`` (without
        :math:`\\rho_{hv}` that gate test is skipped). Without ``phidp``,
        ``PHIDP_processed`` is used if the dataset has it (the output of
        :func:`radarx.retrieve.estimate_kdp`, merged with ``assign``);
        otherwise :func:`radarx.retrieve.estimate_kdp` is run on the raw
        phase (``UPHIDP``, ``PHIDP``, ``uncorrected_differential_phase``,
        ``differential_phase``, ``PHI``) with its defaults, or with
        ``kdp_options``. Name the **uncorrected** reflectivity if the file
        holds several.
    method : {"zphi", "linear"}, optional
        Default ``"zphi"``.
    band : {"S", "C", "X"}, optional
        Radar band for the coefficients. By default it is read from the radar
        frequency of the data (S band, with a warning, if there is none); not
        needed when ``alpha``, ``beta`` and ``b`` are all given.
    coefficients : {"bringi1990", "tmatrix"}, optional
        Set of :math:`\\alpha` and :math:`\\beta`, see
        :func:`attenuation_coefficients`. Default ``"bringi1990"``.
    alpha, beta, b : float, optional
        Override the coefficients: :math:`A_H = \\alpha K_{DP}` and
        :math:`A_{DP} = \\beta K_{DP}` in dB per degree (so that
        ``PIA = alpha * delta_phidp``) and the exponent of
        :math:`A_H = a Z^b` (``"zphi"`` only).
    min_dphi : float, optional
        Rays whose phase span is smaller (degrees) get no correction.
        Default 3.
    max_pia : float, optional
        Upper limit (dB) of the path attenuation. Default: none.
    min_dbz : float, optional
        Gates with a lower reflectivity (dBZ) are not rain. Default 0.
    min_rhohv : float, optional
        Gates with a lower correlation coefficient are not rain. Default 0.9.
    melting : float or xarray.DataArray, optional
        Height of the melting level (m above sea level, a number or an
        array that broadcasts to the reflectivity). Gates at or above it are
        left out of the rain path. The beam height is the ``z`` coordinate
        (e.g. after :meth:`xradar.accessors.XradarDataTreeAccessor.georeference`)
        or computed from the elevation and ``altitude`` (4/3 Earth radius).
        Default: no cutoff.
    mask : str or xarray.DataArray, optional
        Boolean field of the gates that may belong to the rain path (e.g. a
        classification), True where allowed. A name refers to a variable of
        the sweep. For a volume, a name.
    kdp_options : dict, optional
        Options of :func:`radarx.retrieve.estimate_kdp` for the case that the
        phase has to be processed.
    n_threads : int, optional
        Threads of the compiled kernel (per call; each dask chunk is one
        call). Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation. ``"auto"`` (default) prefers the compiled kernel.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        On the coordinates of the input: ``DBZH_corrected`` (dBZ),
        ``ZDR_corrected`` (dB, if the differential reflectivity is given),
        the specific attenuation ``AH`` and differential attenuation ``ADP``
        (dB km-1, one way; NaN outside the rain path), the two-way path
        attenuation ``PIA`` and differential attenuation ``PIDA`` (dB) at each
        gate, and ``DELTA_PHIDP``, the phase span of the rain path of each ray
        (degrees). The coefficients used are in the attributes
        ``attenuation_alpha``, ``attenuation_beta`` and ``attenuation_b``. For
        a volume, a DataTree with one such node per sweep that has the fields,
        and the root of the input. Merge it into the input with
        ``dtree.radarx.assign(products)``. The input is not changed.

    Raises
    ------
    KeyError
        If a field is missing.
    ValueError
        For unknown options, non-uniform range gates, or no band.
    ImportError
        If ``engine="compiled"`` and the kernel is not available.

    Notes
    -----
    Use the **uncorrected** reflectivity: a field that an earlier chain
    already corrected would be corrected twice. The differential phase must
    be free of the system offset: a phase that does not start near zero at the
    first rain gate is fine, since the span of the rain path is used, but a
    phase that is folded or noisy is not. The method assumes rain. Hail
    changes the coefficients, and ZPHI spreads an error of the coefficients
    over the whole ray (Ryzhkov et al. 2014).

    References
    ----------
    .. [1] Testud, J., E. Le Bouar, E. Obligis, and M. Ali-Mehenni, 2000: The
       rain profiling algorithm applied to polarimetric weather radar. *J.
       Atmos. Oceanic Technol.*, **17** (3), 332-356,
       https://doi.org/10.1175/1520-0426(2000)017<0332:TRPAAT>2.0.CO;2
    .. [2] Bringi, V. N., V. Chandrasekar, N. Balakrishnan, and D. S. Zrnic,
       1990: An examination of propagation effects in rainfall on radar
       measurements at microwave frequencies. *J. Atmos. Oceanic Technol.*,
       **7** (6), 829-840,
       https://doi.org/10.1175/1520-0426(1990)007<0829:AEOPEI>2.0.CO;2
    .. [3] Ryzhkov, A., M. Diederich, P. Zhang, and C. Simmer, 2014: Potential
       utilization of specific attenuation for rainfall estimation, mitigation
       of partial beam blockage, and radar networking. *J. Atmos. Oceanic
       Technol.*, **31** (3), 599-619,
       https://doi.org/10.1175/JTECH-D-13-00038.1
    .. [4] Park, S.-G., V. N. Bringi, V. Chandrasekar, M. Maki, and K.
       Iwanami, 2005: Correction of radar reflectivity and differential
       reflectivity for rain attenuation at X band. Part I: Theoretical and
       empirical basis. *J. Atmos. Oceanic Technol.*, **22** (11), 1621-1632,
       https://doi.org/10.1175/JTECH1803.1
    .. [5] Hitschfeld, W., and J. Bordan, 1954: Errors inherent in the radar
       measurement of rainfall at attenuating wavelengths. *J. Meteor.*,
       **11** (1), 58-67,
       https://doi.org/10.1175/1520-0469(1954)011<0058:EIITRM>2.0.CO;2

    Examples
    --------
    >>> kdp = sweep.radarx.kdp()  # doctest: +SKIP
    >>> out = sweep.radarx.assign(kdp).radarx.correct_attenuation(band="C")  # doctest: +SKIP
    >>> out = correct_attenuation(sweep, method="linear", band="X")  # doctest: +SKIP
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {sorted(METHODS)}, not {method!r}")
    if not min_dphi >= 0:
        raise ValueError(f"min_dphi must not be negative, not {min_dphi!r}")
    compiled = _use_compiled(engine)

    given = all(v is not None for v in (alpha, beta, b))

    def correct(dbz, phi, zdr_da, rho, node_mask, ds, band_key):
        a, bt, bb = _coefficients(band_key, coefficients, alpha, beta, b)
        text = f"{method}, alpha={a:g} dB/deg, beta={bt:g} dB/deg" + (
            f", b={bb:g}" if method == "zphi" else ""
        )
        return _correct_fields(
            dbz,
            phi,
            zdr_da,
            rho,
            node_mask,
            melting,
            ds,
            method,
            a,
            bt,
            bb,
            float(min_dphi),
            max_pia,
            float(min_dbz),
            float(min_rhohv),
            n_threads,
            compiled,
            text,
        )

    if isinstance(obj, xr.DataTree):
        root = obj.root.to_dataset(inherit=False)
        sources = [root]
        if "radar_parameters" in obj.children:
            sources.append(obj["radar_parameters"].to_dataset(inherit=False))
        resolved = {}
        for name in obj.children:
            if not name.startswith("sweep"):
                continue
            ds = obj[name].to_dataset(inherit=True)
            fields = _resolve_sweep(
                ds, dbz, zdr, phidp, rhohv, mask, kdp_options, False
            )
            if fields is not None:
                resolved[name] = (ds, fields)
        if not resolved:
            raise KeyError("no sweep has the reflectivity and a differential phase")
        band_key = None if given else _band_of(band, sources)
        nodes = {
            name: correct(*fields[:4], fields[4], ds, band_key)
            for name, (ds, fields) in resolved.items()
        }
        return product_tree(obj, nodes)

    if isinstance(obj, xr.DataArray):
        ds = None
        dbz_in = obj
        if not isinstance(phidp, xr.DataArray):
            raise ValueError(
                "for a DataArray of reflectivity, phidp must be a DataArray"
            )
        for label, value in (("zdr", zdr), ("rhohv", rhohv), ("mask", mask)):
            if value is not None and not isinstance(value, xr.DataArray):
                raise ValueError(
                    f"for a DataArray of reflectivity, {label} must be a DataArray"
                )
        band_key = None if given else _band_of(band, [obj])
        return correct(dbz_in, phidp, zdr, rhohv, mask, obj, band_key)

    if not isinstance(obj, xr.Dataset):
        raise TypeError("obj must be an xarray.DataArray, Dataset or DataTree")
    fields = _resolve_sweep(obj, dbz, zdr, phidp, rhohv, mask, kdp_options, True)
    band_key = None if given else _band_of(band, [obj])
    return correct(*fields[:4], fields[4], obj, band_key)


# --------------------------------------------------------------------------
# accessors
# --------------------------------------------------------------------------


@accessor_method(
    "dataarray",
    "dataset",
    "datatree",
    name="correct_attenuation",
    wraps=correct_attenuation,
)
def _correct_attenuation_accessor(self):
    """
    Correct the reflectivity and differential reflectivity for rain attenuation.

    See :func:`radarx.retrieve.correct_attenuation` for the parameters. On a
    DataArray (the reflectivity), pass ``phidp`` and optionally ``zdr`` as
    DataArrays.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``DBZH_corrected``, ``ZDR_corrected``, ``AH``, ``ADP``, ``PIA``,
        ``PIDA`` and ``DELTA_PHIDP``.
    """
