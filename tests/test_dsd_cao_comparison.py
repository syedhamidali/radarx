#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Constrained-gamma retrieval against the polynomial estimators of Cao et al.
(2008, Eqs. 9 and 11), with the T-matrix code as the independent reference.

The tests pin the study of the notebook ``DSD_Cao_Comparison``: where the
offset at Z_DR of 1.5-2 dB comes from and what it is not.
"""

import functools
import importlib

import numpy as np
import pytest
import xarray as xr
from scipy.integrate import cumulative_trapezoid
from scipy.special import gamma, gammaincinv

import radarx  # noqa: F401
from radarx.fundamentals import tmatrix as tm
from radarx.fundamentals.tmatrix import raindrop_scattering
from radarx.retrieve import dsd, terminal_fall_speed

dsdmod = importlib.import_module("radarx.retrieve.dsd")

pytestmark = pytest.mark.skipif(
    not tm.HAS_COMPILED_KERNEL, reason="compiled T-matrix kernel not built"
)

CAO = dsdmod.MU_LAMBDA["cao2008"]
WL = 299.792458 / 2.8  # mm, the S band of the packaged tables
LAM = np.geomspace(0.8, 12.0, 160)
MU = CAO[0] * LAM**2 + CAO[1] * LAM + CAO[2]


def cao_r_over_z(x):
    """Cao et al. (2008) Eq. 9: R / Z_H for Z_H in mm6 m-3."""
    return 10 ** (-0.0363 * x**3 + 0.316 * x**2 - 1.178 * x - 1.964)


def cao_d0(x):
    """Cao et al. (2008) Eq. 11."""
    return 0.0436 * x**3 - 0.216 * x**2 + 1.076 * x + 0.659


@functools.cache
def _drops(canting=7.0, wavelength=WL):
    d = np.arange(0.05, 8.0 + 1e-9, 0.05)
    sc = raindrop_scattering(d, wavelength, canting_std=canting)
    return d, sc.sigma_h.values, sc.sigma_v.values


def _weights(d):
    w = np.zeros_like(d)
    step = np.diff(d)
    w[:-1] += step / 2
    w[1:] += step / 2
    return w


def _radar(d, sh, sv, nd, wavelength=WL):
    k = wavelength**4 / (np.pi**5 * dsdmod.KW2)
    zh = k * (nd * _weights(d)) @ sh
    zv = k * (nd * _weights(d)) @ sv
    return 10 * np.log10(zh), 10 * np.log10(zh / zv)


def _rate(d, nd, law):
    v = terminal_fall_speed(d, law=law).values
    return 6e-4 * np.pi * (nd * _weights(d) * v * d**3).sum(-1)


def _median(d, nd):
    c = cumulative_trapezoid(nd * d**3, d, initial=0.0)
    return np.interp(0.5, c / c[-1], d)


def _family(d, mu=MU, lam=LAM):
    return d[None] ** mu[:, None] * np.exp(-lam[:, None] * d[None])


def _curves(x, canting=7.0, law="atlas1973"):
    """R / Z_H and D0 of the constrained family at Z_DR = x."""
    d, sh, sv = _drops(canting)
    nd = _family(d)
    zh, zdr = _radar(d, sh, sv, nd)
    o = np.argsort(zdr)
    r = _rate(d, nd, law) / 10 ** (zh / 10)
    d0 = np.array([_median(d, n) for n in nd])
    return np.interp(x, zdr[o], r[o]), np.interp(x, zdr[o], d0[o])


def _dsd_at(zdr, dbz=40.0):
    ds = xr.Dataset(
        {
            "DBZH": ("gate", np.full(len(zdr), float(dbz))),
            "ZDR": ("gate", np.asarray(zdr, dtype=float)),
        }
    )
    return dsd(ds, band="S", nw_range=None)


def test_independent_family_matches_the_lookup_table():
    """The T-matrix family and dsd() agree, so dsd() does what the physics says."""
    x = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    r, d0 = _curves(x)
    out = _dsd_at(x)
    np.testing.assert_allclose(out.RAIN_RATE.values, 1e4 * r, rtol=6e-3)
    np.testing.assert_allclose(out.D0.values, d0, rtol=6e-3)


@pytest.mark.parametrize(
    "x, r_range, d0_range",
    [
        (0.5, (0.90, 0.97), (0.97, 1.03)),
        (1.0, (1.05, 1.15), (0.89, 0.96)),
        (1.5, (1.18, 1.28), (0.85, 0.91)),
        (1.75, (1.20, 1.31), (0.83, 0.90)),
        (2.0, (1.19, 1.30), (0.82, 0.89)),
    ],
)
def test_offset_against_cao_polynomials(x, r_range, d0_range):
    """dsd() at 40 dBZ against Cao et al. (2008) Eqs. 9 and 11."""
    out = _dsd_at([x])
    r_ratio = float(out.RAIN_RATE[0]) / (1e4 * cao_r_over_z(x))
    d0_ratio = float(out.D0[0]) / cao_d0(x)
    assert r_range[0] < r_ratio < r_range[1]
    assert d0_range[0] < d0_ratio < d0_range[1]


def test_atlas_negative_part_is_negligible():
    """The part of the Atlas law below 0.109 mm is a few parts in 1e6 of R."""
    out = _dsd_at(np.array([0.5, 1.0, 1.75, 3.0]))
    mu, lam = out.MU.values.astype(float), out.LAMBDA.values.astype(float)
    dd = np.linspace(0.0, 0.109, 20001)
    for m, lm in zip(mu, lam):
        law = 9.65 - 10.3 * np.exp(-0.6 * dd)
        assert law.min() < 0
        part = 6e-4 * np.pi * np.trapezoid(law * dd ** (m + 3) * np.exp(-lm * dd), dd)
        total = (
            6e-4
            * np.pi
            * gamma(m + 4)
            * lm ** (-(m + 4))
            * (9.65 - 10.3 * (lm / (lm + 0.6)) ** (m + 4))
        )
        assert part < 0
        assert abs(part / total) < 1e-5


def test_fall_speed_law_is_not_the_cause():
    """Atlas against the drag-balance law: the rain rate differs by under 0.2 %."""
    x = np.linspace(0.4, 3.0, 12)
    r_atlas, _ = _curves(x, law="atlas1973")
    r_drag, _ = _curves(x, law="drag_balance")
    assert np.max(np.abs(r_drag / r_atlas - 1)) < 2e-3


def test_median_approximation():
    """(3.67 + mu) / Lambda is within 0.3 % of the median of D^3 N(D)."""
    d, sh, sv = _drops()
    _, zdr = _radar(d, sh, sv, _family(d))
    ok = (zdr > 0.3) & (zdr < 3.0)
    exact = gammaincinv(MU + 4, 0.5) / LAM
    approx = (MU + 3.67) / LAM
    assert np.max(np.abs(approx / exact - 1)[ok]) < 3e-3


def test_canting_moves_the_comparison():
    """0, 7 and 10 degrees of canting change R/R_Cao at 1.75 dB by about 8 points."""
    x = np.array([1.75])
    ratios = {}
    for canting in (0.05, 7.0, 10.0):
        r, d0 = _curves(x, canting=canting)
        ratios[canting] = (r[0] / cao_r_over_z(x[0]), d0[0] / cao_d0(x[0]))
    assert ratios[0.05][0] > ratios[7.0][0] > ratios[10.0][0]
    assert 1.30 < ratios[0.05][0] < 1.38
    assert 1.22 < ratios[7.0][0] < 1.30
    assert 1.13 < ratios[10.0][0] < 1.21
    assert ratios[0.05][1] < ratios[7.0][1] < ratios[10.0][1]
    # the offset stays whatever the canting
    assert all(r > 1.1 and d0 < 0.92 for r, d0 in ratios.values())


def test_frequency_and_temperature_do_not_matter():
    x = np.array([1.75])
    base = _curves(x)[0][0]
    d = np.arange(0.05, 8.0 + 1e-9, 0.05)
    for kw in (dict(wavelength=299.792458 / 3.0), dict(temperature=0.0)):
        wl = kw.get("wavelength", WL)
        sc = raindrop_scattering(d, wl, temperature=kw.get("temperature", 20.0))
        zh, zdr = _radar(d, sc.sigma_h.values, sc.sigma_v.values, _family(d), wl)
        o = np.argsort(zdr)
        r = _rate(d, _family(d), "atlas1973") / 10 ** (zh / 10)
        assert abs(np.interp(x[0], zdr[o], r[o]) / base - 1) < 5e-3


@pytest.mark.parametrize(
    "dmax, r_range, d0_range",
    [
        (8.0, (0.995, 1.005), (0.995, 1.005)),
        (6.0, (1.02, 1.10), (0.94, 0.99)),
        (5.0, (1.15, 1.25), (0.84, 0.90)),
    ],
)
def test_tail_truncation_reproduces_the_offset(dmax, r_range, d0_range):
    """
    Spectra on the Cao mu-Lambda line with no drops above 5 mm are retrieved
    with the +19 % in R and -13 % in D0 of the comparison at 1.5-2 dB.
    """
    d, sh, sv = _drops()
    k = d <= dmax + 1e-9
    nd = _family(d[k])
    zh, zdr = _radar(d[k], sh[k], sv[k], nd)
    truth_r = _rate(d[k], nd, "drag_balance")
    truth_d0 = np.array([_median(d[k], n) for n in nd])
    ds = xr.Dataset({"DBZH": ("gate", zh), "ZDR": ("gate", zdr)})
    res = dsd(ds, band="S", nw_range=None)
    band = (zdr > 1.5) & (zdr < 2.0)
    assert band.any()
    r_ratio = np.mean((res.RAIN_RATE.values / truth_r)[band])
    d0_ratio = np.mean((res.D0.values / truth_d0)[band])
    assert r_range[0] < r_ratio < r_range[1]
    assert d0_range[0] < d0_ratio < d0_range[1]


def test_zdr_calibration_error_is_as_large_as_the_bias():
    """A 0.2 dB offset in Z_DR at 1.75 dB: R changes by +20 % / -16 %."""
    out = _dsd_at([1.55, 1.75, 1.95])
    r = out.RAIN_RATE.values.astype(float)
    d0 = out.D0.values.astype(float)
    assert 0.15 < r[0] / r[1] - 1 < 0.25
    assert -0.20 < r[2] / r[1] - 1 < -0.12
    assert 0.04 < abs(d0[2] / d0[1] - 1) < 0.08


@pytest.fixture(scope="module")
def klbb():
    xd = pytest.importorskip("xradar")
    from open_radar_data import DATASETS

    file = DATASETS.fetch("KLBB20160601_150025_V06")
    ds = xd.io.open_nexradlevel2_datatree(file, sweep=[0])["sweep_0"].to_dataset(
        inherit=False
    )
    for var, lim in (("DBZH", -32.0), ("ZDR", -12.9), ("RHOHV", 0.21)):
        ds[var] = ds[var].where(ds[var] > lim)
    return ds


def test_real_nexrad_gates_follow_the_offset(klbb):
    """
    Gates of the KLBB S-band volume: the ratio of the retrieved R and D0 to
    the polynomials is a function of Z_DR alone, equal to the one found with
    the simulated family.
    """
    out = dsd(klbb, band="S")
    rain = (klbb.RHOHV > 0.97) & (klbb.DBZH > 15) & (klbb.ZDR > 0.1)
    z = 10 ** (klbb.DBZH / 10)
    r_cao = z * cao_r_over_z(klbb.ZDR)
    r_ratio = (out.RAIN_RATE / r_cao).where(rain)
    d0_ratio = (out.D0 / cao_d0(klbb.ZDR)).where(rain)
    for lo, hi, r_range, d0_range in (
        (0.4, 0.6, (0.90, 0.97), (0.97, 1.03)),
        (1.5, 2.0, (1.18, 1.30), (0.84, 0.90)),
    ):
        sel = ((klbb.ZDR >= lo) & (klbb.ZDR < hi) & rain).values
        if sel.sum() < 30:
            pytest.skip(f"fewer than 30 rain gates with Z_DR {lo}-{hi} dB")
        assert r_range[0] < float(np.nanmedian(r_ratio.values[sel])) < r_range[1]
        assert d0_range[0] < float(np.nanmedian(d0_ratio.values[sel])) < d0_range[1]
