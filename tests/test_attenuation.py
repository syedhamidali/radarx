#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the attenuation and differential attenuation correction
=================================================================

The truth is forward modelled: rain profiles along a ray (from drop size
distributions through the T-matrix tables of ``radar_from_dsd``, or from a
power law ``A = a Z^b`` that ZPHI solves exactly) are attenuated with the exact
path integrals, the phase is the integral of the true KDP, and the correction
must give the reflectivity back. The compiled kernel and the NumPy
implementation must agree to rounding.
"""

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx._provenance import HISTORY, METHOD, REFERENCES, load_registry
from radarx.retrieve import attenuation as attmod
from radarx.retrieve import attenuation_coefficients, correct_attenuation, estimate_kdp
from radarx.retrieve.dsd import radar_from_dsd

ENGINES = ["numpy"] + (["compiled"] if attmod.HAS_COMPILED_KERNEL else [])
compiled_only = pytest.mark.skipif(
    not attmod.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)
KAPPA = 2 * 0.23025850929940457
DR = 0.25  # gate spacing [km]


# --------------------------------------------------------------------------
# forward models
# --------------------------------------------------------------------------


def cumulative(a, dr=DR):
    """Two-way integral to the gate centre of a one-way specific value."""
    inc = a * dr
    before = np.cumsum(inc, axis=-1) - inc
    return 2.0 * (before + 0.5 * inc)


def dsd_profile(band, ng=360, temperature=15.0):
    """Rain along a ray from gamma drop size distributions (two cells), no attenuation."""
    from scipy.special import gamma as G

    r = (np.arange(ng) + 0.5) * DR
    cell = np.exp(-0.5 * ((r - 30) / 6) ** 2) + 0.7 * np.exp(-0.5 * ((r - 60) / 8) ** 2)
    dm = 1.1 + 1.1 * cell
    nw = 10 ** (3.7 - 0.5 * cell)
    mu = 3.0
    f = 6 / 4**4 * (4 + mu) ** (mu + 4) / G(mu + 4)
    d = np.arange(0.05, 8.0001, 0.05)
    nd = (
        nw[:, None]
        * f
        * (d[None, :] / dm[:, None]) ** mu
        * np.exp(-(4 + mu) * d[None, :] / dm[:, None])
    )
    nd = xr.DataArray(nd, dims=("gate", "diameter"), coords={"diameter": d})
    out = radar_from_dsd(nd, band=band, temperature=temperature)
    return {k: out[k].values for k in ("DBZH", "ZDR", "KDP", "AH", "ADP")}


def attenuate(truth):
    """Measured Z, ZDR and the propagation phase from the true profile."""
    pia = cumulative(truth["AH"])
    pida = cumulative(truth["ADP"])
    return {
        "DBZH": truth["DBZH"] - pia,
        "ZDR": truth["ZDR"] - pida,
        "PHIDP": cumulative(truth["KDP"]),
        "PIA": pia,
        "PIDA": pida,
    }


def power_law_ray(a_at_40=0.3, b=0.84, ng=400, dr=0.05, peak=50.0, width=6.0):
    """A ray with ``A = a Z^b`` exactly; returns (r, Z_true dBZ, A, a)."""
    r = (np.arange(ng) + 0.5) * dr
    dbz = 20.0 + (peak - 20.0) * np.exp(-0.5 * ((r - 0.4 * r[-1]) / width) ** 2)
    a = a_at_40 / 10 ** (0.4 * 10 * b)
    a_h = a * 10 ** (b * dbz / 10)
    return r, dbz, a_h, a


def hitschfeld_bordan(zm_dbz, dr, a, b):
    """Unconstrained solution of Hitschfeld and Bordan, gate-centred."""
    w = 10 ** (b * zm_dbz / 10)
    before = np.cumsum(w * dr) - w * dr
    denom = 1.0 - KAPPA * a * b * (before + 0.5 * w * dr)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, a * w / denom, np.inf), denom


# --------------------------------------------------------------------------
# the kernel against its NumPy reference
# --------------------------------------------------------------------------


def random_rays(seed=0, nr=70, ng=260):
    rnd = np.random.default_rng(seed)
    z = 30 + 12 * rnd.standard_normal((nr, ng))
    phi = np.cumsum(np.abs(rnd.normal(0.3, 0.6, (nr, ng))), axis=1)
    phi += rnd.normal(0, 0.5, (nr, ng))
    valid = rnd.random((nr, ng)) > 0.25
    valid[3] = False  # a ray without rain
    valid[4, :] = False
    valid[4, 100] = True  # a single gate
    z[5, 10:40] = np.nan
    phi[6, 50:60] = np.nan
    phi[7] = -phi[7]  # decreasing phase: no correction
    return z, phi, valid.astype(np.uint8)


@compiled_only
@pytest.mark.parametrize("method", [0, 1])
@pytest.mark.parametrize("max_pia", [np.nan, 6.0])
def test_kernel_agrees_with_numpy(method, max_pia):
    z, phi, valid = random_rays()
    got = attmod._attenuation.correct(
        z, phi, valid, DR, method, 0.07, 0.8, 3.0, max_pia, 0
    )
    ref = attmod._correct_numpy(z, phi, valid, DR, method, 0.07, 0.8, 3.0, max_pia)
    for g, r in zip(got, ref):
        assert np.array_equal(np.isnan(g), np.isnan(r))
        np.testing.assert_allclose(g, r, rtol=1e-10, atol=1e-12)
    assert np.isfinite(got[1]).all()


@compiled_only
def test_kernel_threads_do_not_change_the_result():
    z, phi, valid = random_rays(nr=200)
    args = (z, phi, valid, DR, 0, 0.07, 0.8, 3.0, np.nan)
    one = attmod._attenuation.correct(*args, 1)
    many = attmod._attenuation.correct(*args, 4)
    for a, b in zip(one, many):
        np.testing.assert_array_equal(a, b)


@compiled_only
def test_kernel_rejects_bad_shapes():
    z, phi, valid = random_rays()
    with pytest.raises(ValueError):
        attmod._attenuation.correct(
            z, phi[:, :-1], valid, DR, 0, 0.07, 0.8, 3.0, np.nan
        )
    with pytest.raises(ValueError):
        attmod._attenuation.correct(z, phi, valid, -1.0, 0, 0.07, 0.8, 3.0, np.nan)


# --------------------------------------------------------------------------
# properties of the methods
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_zphi_path_attenuation_is_the_constraint(engine):
    """The gate-integrated solution integrates to alpha * delta phi exactly."""
    z, phi, valid = random_rays()
    ah, pia, dphi = attmod._block(
        z, phi, valid, DR, "zphi", 0.07, 0.8, 3.0, None, 0, engine == "compiled"
    )
    ray = (dphi >= 3.0) & np.isfinite(dphi)
    assert ray.sum() > 30
    total = 2 * DR * np.nansum(ah, axis=1)
    np.testing.assert_allclose(total[ray], 0.07 * dphi[ray], rtol=1e-12)
    assert np.all(pia[ray][:, -1] <= 0.07 * dphi[ray] * (1 + 1e-12))
    # no correction without a path or below the minimum span
    assert np.all(pia[~ray] == 0.0) and np.isnan(ah[~ray]).all()
    assert np.all(np.diff(pia, axis=1) >= -1e-12)  # never decreases
    assert np.nanmin(ah) >= 0.0


@pytest.mark.parametrize("engine", ENGINES)
def test_linear_is_alpha_times_the_phase(engine):
    ng = 120
    phi = np.linspace(0.0, 90.0, ng)[None, :] + 5.0
    z = np.full((1, ng), 35.0)
    valid = np.ones((1, ng), dtype=np.uint8)
    ah, pia, dphi = attmod._block(
        z, phi, valid, DR, "linear", 0.054, 0.8, 3.0, None, 0, engine == "compiled"
    )
    np.testing.assert_allclose(pia[0], 0.054 * (phi[0] - phi[0, 0]), rtol=1e-12)
    np.testing.assert_allclose(dphi, 90.0)
    # AH = alpha KDP, KDP = half the range derivative of the phase
    np.testing.assert_allclose(ah[0], 0.054 * 0.5 * 90.0 / ((ng - 1) * DR), rtol=1e-3)


@pytest.mark.parametrize("engine", ENGINES)
def test_linear_never_decreases_and_bridges_gaps(engine):
    ng = 100
    phi = np.linspace(0.0, 50.0, ng)
    phi[40] -= 8.0  # noise: a dip
    z = np.full(ng, 30.0)
    valid = np.ones(ng, dtype=np.uint8)
    valid[60:70] = 0  # a gap
    phi[60:70] = np.nan  # no phase in the gap
    ah, pia, dphi = attmod._block(
        z[None],
        phi[None],
        valid[None],
        DR,
        "linear",
        0.1,
        0.8,
        3.0,
        None,
        0,
        engine == "compiled",
    )
    assert np.all(np.diff(pia[0]) >= 0)
    assert pia[0, 40] == pia[0, 39]  # the dip is not attenuation
    expect = 0.1 * (np.interp(65, [59, 70], [phi[59], phi[70]]) - phi[0])
    assert pia[0, 65] == pytest.approx(expect, rel=1e-12)
    assert np.isnan(ah[0, 65])  # no specific attenuation inside the gap
    assert pia[0, -1] == pytest.approx(0.1 * dphi[0])


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("method", ["zphi", "linear"])
def test_small_span_and_missing_data(engine, method):
    ng = 80
    z = np.full((3, ng), 30.0)
    phi = np.zeros((3, ng))
    phi[0] = np.linspace(0, 2.0, ng)  # span below min_dphi
    phi[1] = np.nan  # no phase
    phi[2] = np.linspace(0, 20.0, ng)
    valid = np.ones((3, ng), dtype=np.uint8)
    ah, pia, dphi = attmod._block(
        z, phi, valid, DR, method, 0.08, 0.8, 3.0, None, 0, engine == "compiled"
    )
    assert np.all(pia[:2] == 0.0) and np.isnan(ah[:2]).all()
    assert np.isnan(dphi[1])
    assert pia[2, -1] > 1.0


@pytest.mark.parametrize("engine", ENGINES)
def test_max_pia_caps_the_path(engine):
    z, phi, valid = random_rays()
    for method in ("zphi", "linear"):
        _, pia, _ = attmod._block(
            z, phi, valid, DR, method, 0.2, 0.8, 3.0, 5.0, 0, engine == "compiled"
        )
        assert pia.max() <= 5.0 + 1e-9


# --------------------------------------------------------------------------
# validation against forward-modelled truth
# --------------------------------------------------------------------------


def sweep_from(meas, band_range=None):
    ng = meas["DBZH"].size
    return xr.Dataset(
        {
            "DBZH": (("azimuth", "range"), meas["DBZH"][None, :]),
            "ZDR": (("azimuth", "range"), meas["ZDR"][None, :]),
            "PHIDP_processed": (("azimuth", "range"), meas["PHIDP"][None, :]),
            "RHOHV": (("azimuth", "range"), np.full((1, ng), 0.99)),
        },
        coords={
            "azimuth": [10.0],
            "range": ("range", (np.arange(ng) + 0.5) * DR * 1000.0),
            "elevation": ("azimuth", [0.5]),
        },
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_zphi_is_exact_for_a_power_law_rain(engine):
    """A = a Z^b is the model of ZPHI: the correction returns Z to within the gate size."""
    r, dbz_true, a_h, a = power_law_ray(dr=0.05)
    b = 0.84
    pia = cumulative(a_h, 0.05)
    zm = dbz_true - pia
    alpha = 0.08
    phi = pia / alpha  # the phase that goes with this path attenuation
    ds = xr.Dataset(
        {
            "DBZH": (("azimuth", "range"), zm[None]),
            "PHIDP_processed": (("azimuth", "range"), phi[None]),
        },
        coords={"azimuth": [0.0], "range": ("range", r * 1000.0)},
    )
    out = correct_attenuation(
        ds, alpha=alpha, beta=0.02, b=b, min_dbz=-50.0, engine=engine
    )
    err = out.DBZH_corrected.values[0] - dbz_true
    assert pia[-1] > 25.0  # a strong case
    assert np.abs(err).max() < 0.1
    np.testing.assert_allclose(out.AH.values[0], a_h, rtol=0.02, atol=1e-3)


@pytest.mark.parametrize("engine", ENGINES)
def test_unconstrained_solution_is_unstable_and_zphi_is_not(engine):
    """5 % too large a coefficient makes Hitschfeld-Bordan singular; ZPHI stays bounded."""
    r, dbz_true, a_h, a = power_law_ray(dr=0.05)
    b = 0.84
    pia = cumulative(a_h, 0.05)
    zm = dbz_true - pia
    # Hitschfeld and Bordan with the exact coefficient is fine ...
    exact, denom = hitschfeld_bordan(zm, 0.05, a, b)
    assert denom.min() > 0
    assert np.abs(10 * np.log10(exact / a_h)).max() < 0.2
    # ... and diverges for a coefficient 5 % too large: the denominator reaches zero
    wrong, denom = hitschfeld_bordan(zm, 0.05, 1.05 * a, b)
    assert denom.min() <= 0 and np.isinf(wrong).any()
    # ZPHI with a path attenuation that is 5 % too large (phase 5 % too large)
    alpha = 0.08
    ds = xr.Dataset(
        {
            "DBZH": (("azimuth", "range"), zm[None]),
            "PHIDP_processed": (("azimuth", "range"), 1.05 * pia[None] / alpha),
        },
        coords={"azimuth": [0.0], "range": ("range", r * 1000.0)},
    )
    out = correct_attenuation(
        ds, alpha=alpha, beta=0.02, b=b, min_dbz=-50.0, engine=engine
    )
    err = out.DBZH_corrected.values[0] - dbz_true
    assert np.isfinite(out.AH.values[0]).all()
    assert np.abs(err).max() < 3.0
    # the error is bounded by the error of the path attenuation
    assert err[-1] == pytest.approx(0.05 * pia[-1], abs=0.3)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("band", ["S", "C", "X"])
def test_forward_modelled_rain_is_corrected(band, engine):
    """T-matrix rain with changing drop sizes, attenuated and corrected."""
    truth = dsd_profile(band)
    meas = attenuate(truth)
    ds = sweep_from(meas)
    pia_total = meas["PIA"][-1]
    dphi = meas["PHIDP"][-1] - meas["PHIDP"][0]
    # the exact ratio of this rain: the only error is the model A = a Z^b
    exact = pia_total / dphi
    ratio = truth["ADP"].sum() / truth["AH"].sum()
    out = correct_attenuation(
        ds,
        alpha=exact,
        beta=ratio * exact,
        b=attmod._COEFFICIENTS["tmatrix"][band]["b"],
        engine=engine,
    )
    err = out.DBZH_corrected.values[0] - truth["DBZH"]
    derr = out.ZDR_corrected.values[0] - truth["ZDR"]
    uncorrected = meas["DBZH"] - truth["DBZH"]
    # the correction removes most of the bias; the rest is the variation of
    # a along the ray, which ZPHI cannot know
    assert np.abs(err).max() < 0.25 * max(1.0, np.abs(uncorrected).max()) + 0.5
    assert (
        np.abs(derr).max()
        < 0.3 * max(0.1, np.abs(meas["ZDR"] - truth["ZDR"]).max()) + 0.1
    )
    assert (
        abs(out.DBZH_corrected.values[0, -1] - truth["DBZH"][-1])
        < 1.0 + 0.06 * pia_total
    )
    # the linear method with the exact coefficient is exact at the end of the path
    lin = correct_attenuation(
        ds, method="linear", alpha=exact, beta=ratio * exact, b=0.8, engine=engine
    )
    assert (
        abs(lin.DBZH_corrected.values[0, -1] - truth["DBZH"][-1])
        < 0.1 + 0.02 * pia_total
    )


# --------------------------------------------------------------------------
# front end
# --------------------------------------------------------------------------


def test_coefficients():
    for band in "SCX":
        for name in ("bringi1990", "tmatrix"):
            c = attenuation_coefficients(band, name)
            assert set(c) == {"alpha", "beta", "b"}
            assert 0 < c["beta"] < c["alpha"]
    c = attenuation_coefficients("c")
    assert (c["alpha"], c["beta"]) == (0.054, 0.0157)
    assert attenuation_coefficients("X")["alpha"] == 0.25
    c["alpha"] = 99.0  # a copy
    assert attenuation_coefficients("C")["alpha"] == 0.054
    assert (
        attenuation_coefficients("S")["alpha"]
        < attenuation_coefficients("C")["alpha"]
        < attenuation_coefficients("X")["alpha"]
    )
    with pytest.raises(ValueError):
        attenuation_coefficients("K")
    with pytest.raises(ValueError):
        attenuation_coefficients("C", "nope")


def test_the_tmatrix_set_is_the_fit_to_the_scattering_tables():
    """alpha and beta of the 'tmatrix' set are regressions through the origin on KDP."""
    from scipy.special import gamma as G

    rnd = np.random.default_rng(7)
    n = 3000
    mu = rnd.uniform(0, 6, n)
    dm = rnd.uniform(0.9, 2.8, n)
    nw = 10 ** rnd.uniform(2.8, 4.4, n)
    f = 6 / 4**4 * (4 + mu) ** (mu + 4) / G(mu + 4)
    d = np.arange(0.05, 8.0001, 0.05)
    nd = nw[:, None] * f[:, None] * (d / dm[:, None]) ** mu[:, None]
    nd = nd * np.exp(-(4 + mu[:, None]) * d / dm[:, None])
    nd = xr.DataArray(nd, dims=("s", "diameter"), coords={"diameter": d})
    for band in "SCX":
        o = radar_from_dsd(nd, band=band, temperature=15.0)
        dbz = o.DBZH.values
        k = o.KDP.values
        ok = (o.AH.values > 1e-5) & (k > 0.02) & (dbz > 15) & (dbz < 55)
        alpha = (o.AH.values[ok] * k[ok]).sum() / (k[ok] ** 2).sum()
        beta = (o.ADP.values[ok] * k[ok]).sum() / (k[ok] ** 2).sum()
        slope = np.polyfit(np.log(10 ** (dbz[ok] / 10)), np.log(o.AH.values[ok]), 1)[0]
        c = attenuation_coefficients(band, "tmatrix")
        assert alpha == pytest.approx(c["alpha"], rel=0.06)
        assert beta == pytest.approx(c["beta"], rel=0.06)
        assert slope == pytest.approx(c["b"], abs=0.05)


def test_dataset_result_fields_and_attributes():
    meas = attenuate(dsd_profile("C"))
    ds = sweep_from(meas)
    before = ds.copy(deep=True)
    out = correct_attenuation(ds, band="C")
    xr.testing.assert_identical(ds, before)  # the input is not changed
    assert set(out.data_vars) == {
        "DBZH_corrected",
        "ZDR_corrected",
        "AH",
        "ADP",
        "PIA",
        "PIDA",
        "DELTA_PHIDP",
    }
    assert out.DBZH_corrected.dims == ds.DBZH.dims
    for name in ("azimuth", "range", "elevation"):
        xr.testing.assert_equal(out[name], ds[name])
    assert out.DBZH_corrected.attrs["units"] == "dBZ"
    assert out.ZDR_corrected.attrs["units"] == "dB"
    assert out.AH.attrs["units"] == "dB km-1" and out.PIA.attrs["units"] == "dB"
    assert out.DBZH_corrected.attrs["standard_name"] == "equivalent_reflectivity_factor"
    assert out.attrs["attenuation_alpha"] == 0.054
    assert out.attrs["attenuation_beta"] == 0.0157
    # PIDA = (beta / alpha) PIA and corrected = measured + attenuation
    np.testing.assert_allclose(out.PIDA, out.PIA * 0.0157 / 0.054, rtol=1e-6)
    np.testing.assert_allclose(out.DBZH_corrected, ds.DBZH + out.PIA, rtol=1e-6)
    np.testing.assert_allclose(out.ZDR_corrected, ds.ZDR + out.PIDA, rtol=1e-6)
    assert float(out.DELTA_PHIDP[0]) == pytest.approx(
        float(ds.PHIDP_processed[0, -1] - ds.PHIDP_processed[0, 0])
    )
    # provenance
    assert out.attrs[METHOD].startswith("Rain attenuation")
    refs = out.attrs[REFERENCES].split()
    assert "10.1175/1520-0426(2000)017<0332:TRPAAT>2.0.CO;2" in refs
    assert "10.1175/1520-0426(1990)007<0829:AEOPEI>2.0.CO;2" in refs
    registry = load_registry()
    assert all(r in registry for r in refs)
    assert out.attrs[HISTORY].startswith("correct_attenuation(")


def test_overrides_and_validation():
    ds = sweep_from(attenuate(dsd_profile("C")))
    a = correct_attenuation(ds, band="C")
    b = correct_attenuation(ds, band="C", alpha=0.054, beta=0.0157)
    xr.testing.assert_allclose(a.DBZH_corrected, b.DBZH_corrected)
    # twice the coefficient, twice the path attenuation
    c = correct_attenuation(ds, band="C", alpha=0.108, method="linear")
    d = correct_attenuation(ds, band="C", method="linear")
    np.testing.assert_allclose(c.PIA.values[0, -1], 2 * d.PIA.values[0, -1])
    # the coefficient set
    t = correct_attenuation(ds, band="C", coefficients="tmatrix")
    assert t.attrs["attenuation_alpha"] == 0.084
    # no band needed when all coefficients are given
    correct_attenuation(ds, alpha=0.05, beta=0.01, b=0.8)
    with pytest.raises(ValueError, match="method"):
        correct_attenuation(ds, band="C", method="nope")
    with pytest.raises(ValueError, match="band"):
        correct_attenuation(ds, band="K")
    with pytest.raises(ValueError, match="engine"):
        correct_attenuation(ds, band="C", engine="gpu")
    with pytest.raises(ValueError, match="alpha"):
        correct_attenuation(ds, band="C", alpha=-1.0, beta=0.0, b=0.8)
    with pytest.raises(KeyError):
        correct_attenuation(ds.drop_vars("DBZH"), band="C")
    with pytest.raises(KeyError):
        correct_attenuation(ds.drop_vars("PHIDP_processed"), band="C")
    with pytest.raises(TypeError):
        correct_attenuation([1, 2, 3], band="C")


def test_band_is_read_from_the_frequency():
    ds = sweep_from(attenuate(dsd_profile("X")))
    ds["frequency"] = 9.4e9
    out = correct_attenuation(ds)
    assert out.attrs["attenuation_alpha"] == 0.25
    ds["frequency"] = 2.8e9
    assert correct_attenuation(ds).attrs["attenuation_alpha"] == 0.016
    with pytest.warns(UserWarning, match="band"):
        correct_attenuation(ds.drop_vars("frequency"))


def test_without_zdr_and_rhohv():
    ds = sweep_from(attenuate(dsd_profile("C"))).drop_vars(["ZDR", "RHOHV"])
    out = correct_attenuation(ds, band="C")
    assert "ZDR_corrected" not in out
    assert out.DBZH_corrected.notnull().all()


def test_rhohv_mask_and_melting_level():
    meas = attenuate(dsd_profile("C"))
    ds = sweep_from(meas)
    ng = ds.sizes["range"]
    base = correct_attenuation(ds, band="C", method="linear")
    # low rhohv beyond gate 200: those gates leave the path, the last value is held
    rho = ds.RHOHV.copy()
    rho[:, 200:] = 0.5
    cut = correct_attenuation(ds.assign(RHOHV=rho), band="C", method="linear")
    np.testing.assert_allclose(cut.PIA.values[0, 200:], cut.PIA.values[0, 199])
    assert cut.PIA.values[0, -1] < base.PIA.values[0, -1]
    # the melting level: beam height from the elevation, 0.5 degrees
    height = attmod.beam_center_height(ds.range, 0.5)
    level = float(height[250])
    melt = correct_attenuation(ds, band="C", method="linear", melting=level)
    np.testing.assert_allclose(melt.PIA.values[0, 250:], melt.PIA.values[0, 249])
    assert ng > 300
    # a mask
    keep = xr.ones_like(ds.DBZH, dtype=bool)
    keep[:, :20] = False
    masked = correct_attenuation(
        ds.assign(keep=keep), band="C", mask="keep", method="linear"
    )
    np.testing.assert_allclose(masked.PIA.values[0, :20], 0.0)
    assert np.isnan(masked.AH.values[0, :20]).all()
    # a z coordinate is used when there is one
    with_z = ds.assign_coords(z=height.broadcast_like(ds.DBZH).transpose(*ds.DBZH.dims))
    same = correct_attenuation(with_z, band="C", method="linear", melting=level)
    np.testing.assert_allclose(same.PIA.values, melt.PIA.values)


def test_dataarray_input_and_accessors():
    ds = sweep_from(attenuate(dsd_profile("C")))
    full = correct_attenuation(ds, band="C")
    out = correct_attenuation(ds.DBZH, phidp=ds.PHIDP_processed, zdr=ds.ZDR, band="C")
    xr.testing.assert_allclose(out.DBZH_corrected, full.DBZH_corrected)
    xr.testing.assert_allclose(out.ZDR_corrected, full.ZDR_corrected)
    acc = ds.DBZH.radarx.correct_attenuation(phidp=ds.PHIDP_processed, band="C")
    xr.testing.assert_allclose(acc.PIA, full.PIA)
    xr.testing.assert_allclose(ds.radarx.correct_attenuation(band="C").PIA, full.PIA)
    with pytest.raises(ValueError, match="phidp"):
        correct_attenuation(ds.DBZH, band="C")


def test_datatree_volume():
    ds = sweep_from(attenuate(dsd_profile("C")))
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset({"frequency": 5.6e9}),
            "/sweep_0": ds,
            "/sweep_1": ds.drop_vars("PHIDP_processed"),  # no phase: skipped
            "/other": xr.Dataset({"x": 1}),
        }
    )
    out = correct_attenuation(tree)
    assert set(out.children) == {"sweep_0"}
    assert "frequency" in out.to_dataset()  # the root of the input
    xr.testing.assert_allclose(
        out["sweep_0"]["DBZH_corrected"],
        correct_attenuation(ds, band="C").DBZH_corrected,
    )
    assert out["sweep_0"].attrs[METHOD]
    merged = tree.radarx.correct_attenuation(band="C").radarx.assign
    assert callable(merged)
    full = tree.radarx.assign(tree.radarx.correct_attenuation(band="C"))
    assert "DBZH_corrected" in full["sweep_0"].to_dataset()
    with pytest.raises(KeyError):
        correct_attenuation(xr.DataTree.from_dict({"/sweep_0": xr.Dataset({"x": 1})}))


def test_dask_stays_lazy_and_agrees():
    dask = pytest.importorskip("dask")  # noqa: F841
    meas = attenuate(dsd_profile("C"))
    base = sweep_from(meas)
    nray = 12
    ds = xr.concat(
        [base.assign_coords(azimuth=[10.0 + i]) for i in range(nray)], "azimuth"
    )
    ds["DBZH"] = ds.DBZH + xr.DataArray(np.arange(nray) * 0.5, dims="azimuth")
    lazy = ds.chunk({"azimuth": 4, "range": 100})
    out = correct_attenuation(lazy, band="C")
    assert out.DBZH_corrected.chunks is not None
    assert out.AH.chunks is not None and out.DELTA_PHIDP.chunks is not None
    assert out.DELTA_PHIDP.dims == ("azimuth",)
    eager = correct_attenuation(ds, band="C")
    xr.testing.assert_allclose(out.compute(), eager)
    assert lazy.DBZH.chunks is not None  # the input stays lazy


def test_dimension_order_is_kept():
    ds = sweep_from(attenuate(dsd_profile("C")))
    ds = xr.concat([ds, ds.assign_coords(azimuth=[11.0])], "azimuth")
    swapped = ds.transpose("range", "azimuth")
    out = correct_attenuation(swapped, band="C")
    assert out.DBZH_corrected.dims == ("range", "azimuth")
    ref = correct_attenuation(ds, band="C")
    np.testing.assert_allclose(out.PIA.T.values, ref.PIA.values)


def test_raw_phase_is_processed_by_estimate_kdp():
    """Without PHIDP_processed the raw phase goes through estimate_kdp."""
    meas = attenuate(dsd_profile("C", ng=360))
    ds = sweep_from(meas).drop_vars("PHIDP_processed")
    rnd = np.random.default_rng(3)
    nray = 20
    raw = 60.0 + meas["PHIDP"][None, :] + rnd.normal(0, 2.0, (nray, 360))
    ds = xr.concat([ds] * nray, "azimuth").assign_coords(azimuth=np.arange(nray) * 1.0)
    ds["PHIDP"] = (("azimuth", "range"), raw)
    out = correct_attenuation(ds, band="C", method="zphi")
    ref = estimate_kdp(ds, phidp="PHIDP", rhohv="RHOHV", dbz="DBZH")
    both = correct_attenuation(ds.assign(PHIDP_processed=ref.PHIDP_processed), band="C")
    xr.testing.assert_allclose(out.PIA, both.PIA)
    # the span is the span of the true propagation phase, to the noise of the filter
    assert (
        abs(float(out.DELTA_PHIDP.mean()) - (meas["PHIDP"][-1] - meas["PHIDP"][0]))
        < 6.0
    )
    with pytest.raises(KeyError):
        correct_attenuation(ds.drop_vars("PHIDP"), band="C")


def test_engines_give_the_same_dataset():
    ds = sweep_from(attenuate(dsd_profile("X")))
    for method in ("zphi", "linear"):
        a = correct_attenuation(ds, band="X", method=method, engine="numpy")
        if attmod.HAS_COMPILED_KERNEL:
            b = correct_attenuation(ds, band="X", method=method, engine="compiled")
            xr.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)
    if not attmod.HAS_COMPILED_KERNEL:  # pragma: no cover
        with pytest.raises(ImportError):
            correct_attenuation(ds, band="X", engine="compiled")


# --------------------------------------------------------------------------
# argument resolution and error paths
# --------------------------------------------------------------------------


def test_compiled_engine_needs_the_kernel(monkeypatch):
    ds = sweep_from(attenuate(dsd_profile("C")))
    monkeypatch.setattr(attmod, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError, match="compiled attenuation kernel"):
        correct_attenuation(ds, band="C", engine="compiled")
    # the NumPy engine and 'auto' still work without the kernel
    auto = correct_attenuation(ds, band="C", engine="auto")
    ref = correct_attenuation(ds, band="C", engine="numpy")
    xr.testing.assert_allclose(auto, ref)


def test_rays_without_a_rain_path_give_no_correction():
    """Every span below min_dphi: nothing is corrected (NumPy engine)."""
    ng = 60
    z = np.full((4, ng), 30.0)
    phi = np.tile(np.linspace(0.0, 1.0, ng), (4, 1))
    valid = np.ones((4, ng), dtype=np.uint8)
    ah, pia, dphi = attmod._correct_numpy(z, phi, valid, DR, 0, 0.08, 0.8, 3.0, np.nan)
    assert np.isnan(ah).all() and np.all(pia == 0.0)
    np.testing.assert_allclose(dphi, 1.0)


def test_sweep_without_rays_is_empty():
    ds = sweep_from(attenuate(dsd_profile("C")))
    none = ds.isel(azimuth=slice(0, 0))
    out = correct_attenuation(none, band="C")
    assert out.DBZH_corrected.shape == (0, ds.sizes["range"])
    assert out.DELTA_PHIDP.shape == (0,)
    empty = np.zeros((0, 5))
    ah, pia, dphi = attmod._block(
        empty, empty, empty, DR, "zphi", 0.05, 0.8, 3.0, None, 0, False
    )
    assert ah.shape == (0, 5) and pia.shape == (0, 5) and dphi.shape == (0,)


def test_coefficient_ranges_are_checked():
    ds = sweep_from(attenuate(dsd_profile("C")))
    with pytest.raises(ValueError, match="beta must not be negative"):
        correct_attenuation(ds, alpha=0.05, beta=-0.01, b=0.8)
    with pytest.raises(ValueError, match="b must be in"):
        correct_attenuation(ds, alpha=0.05, beta=0.01, b=2.5)
    with pytest.raises(ValueError, match="b must be in"):
        correct_attenuation(ds, alpha=0.05, beta=0.01, b=0.0)
    with pytest.raises(ValueError, match="min_dphi must not be negative"):
        correct_attenuation(ds, band="C", min_dphi=-1.0)


def test_fields_by_name_or_by_array():
    ds = sweep_from(attenuate(dsd_profile("C")))
    ref = correct_attenuation(ds, band="C")
    # DataArrays for the fields of a Dataset
    by_array = correct_attenuation(
        ds, dbz=ds.DBZH, phidp=ds.PHIDP_processed, zdr=ds.ZDR, band="C"
    )
    xr.testing.assert_allclose(by_array.DBZH_corrected, ref.DBZH_corrected)
    xr.testing.assert_allclose(by_array.ZDR_corrected, ref.ZDR_corrected)
    # names of fields of the Dataset
    renamed = ds.rename(PHIDP_processed="PHI_USER", DBZH="Z_USER")
    by_name = correct_attenuation(renamed, dbz="Z_USER", phidp="PHI_USER", band="C")
    np.testing.assert_allclose(by_name.PIA, ref.PIA)
    with pytest.raises(KeyError, match="'PHI_NONE' is not in the dataset"):
        correct_attenuation(ds, phidp="PHI_NONE", band="C")
    with pytest.raises(KeyError, match="'REFL_NONE' is not in the dataset"):
        correct_attenuation(ds, dbz="REFL_NONE", band="C")


def test_dataarray_inputs_must_be_dataarrays():
    ds = sweep_from(attenuate(dsd_profile("C")))
    with pytest.raises(ValueError, match="zdr must be a DataArray"):
        correct_attenuation(ds.DBZH, phidp=ds.PHIDP_processed, zdr="ZDR", band="C")
    with pytest.raises(ValueError, match="mask must be a DataArray"):
        correct_attenuation(ds.DBZH, phidp=ds.PHIDP_processed, mask="keep", band="C")


def test_reflectivity_needs_a_range_dimension():
    ds = sweep_from(attenuate(dsd_profile("C")))
    z = ds.DBZH.rename(range="gate")
    with pytest.raises(ValueError, match="needs a 'range' dimension"):
        correct_attenuation(z, phidp=ds.PHIDP_processed.rename(range="gate"), band="C")


def test_melting_height_adds_the_radar_altitude():
    ds = sweep_from(attenuate(dsd_profile("C")))
    ds = ds.assign_coords(altitude=300.0)
    level = float(attmod.beam_center_height(ds.range, 0.5, radar_height=300.0)[250])
    melt = correct_attenuation(ds, band="C", method="linear", melting=level)
    # gate 250 is at the melting level: from there on the path is held
    np.testing.assert_allclose(melt.PIA.values[0, 250:], melt.PIA.values[0, 249])
    # the same level with the radar at sea level: the beam is lower, more rain gates
    low = correct_attenuation(
        ds.assign_coords(altitude=0.0), band="C", method="linear", melting=level
    )
    assert low.PIA.values[0, -1] > melt.PIA.values[0, -1]
    # neither an elevation nor a z coordinate: the beam height is unknown
    with pytest.raises(ValueError, match="melting needs the beam height"):
        correct_attenuation(ds.drop_vars("elevation"), band="C", melting=level)
    # the dataset can be missing: the height comes from the field itself
    height = attmod._gate_height(ds.DBZH.assign_coords(altitude=300.0), None)
    np.testing.assert_allclose(height.values[250], level)


def test_datatree_reads_the_band_from_radar_parameters():
    ds = sweep_from(attenuate(dsd_profile("X")))
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "/radar_parameters": xr.Dataset({"frequency": 9.4e9}),
            "/sweep_0": ds,
        }
    )
    out = correct_attenuation(tree)
    assert out["sweep_0"].attrs["attenuation_alpha"] == 0.25
