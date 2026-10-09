#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for the Bayesian drop size distribution retrieval."""

import importlib

import numpy as np
import pytest
import xarray as xr
from scipy.integrate import trapezoid

import radarx  # noqa: F401
from radarx.retrieve import (
    dsd,
    dsd_bayesian,
    dsd_prior,
    dsd_spectrum,
    fit_gamma_moments,
    forward_grid,
    radar_from_dsd,
)

bayes = importlib.import_module("radarx.retrieve.dsd_bayes")
dsdmod = importlib.import_module("radarx.retrieve.dsd")

ENGINES = ["numpy"] + (["compiled"] if bayes.HAS_COMPILED_KERNEL else [])
compiled_only = pytest.mark.skipif(
    not bayes.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)
FAST = "compiled" if bayes.HAS_COMPILED_KERNEL else "numpy"


def _draw(prior, n, seed=0, band="S", kdp=False, ah=False, errors=None):
    """Truth drawn from the prior and observations with the error model."""
    rnd = np.random.default_rng(seed)
    fw = forward_grid(band)
    e = dict(bayes.ERRORS, **(errors or {}))
    pm = prior.prior_mass.values.ravel()
    j = rnd.choice(pm.size, n, p=pm)
    t = rnd.normal(
        prior.log10_nw_mean.values.ravel()[j], prior.log10_nw_sd.values.ravel()[j]
    )
    flat = {
        k: fw[k].values.ravel()[j] for k in ("DBZH", "ZDR", "KDP", "AH", "RAIN_RATE")
    }
    data = {
        "DBZH": 10 * t
        + flat["DBZH"]
        + rnd.normal(0, np.hypot(e["zh"], e["zh_bias"]), n),
        "ZDR": flat["ZDR"] + rnd.normal(0, np.hypot(e["zdr"], e["zdr_bias"]), n),
    }
    if kdp:
        k = 10**t * flat["KDP"]
        data["KDP"] = k + rnd.normal(0, 1, n) * np.sqrt(
            e["kdp"] ** 2 + (e["kdp_rel"] * k) ** 2
        )
    if ah:
        a = 10**t * flat["AH"]
        data["AH"] = a + rnd.normal(0, 1, n) * np.sqrt(
            e["ah"] ** 2 + (e["ah_rel"] * a) ** 2
        )
    ds = xr.Dataset(
        {k: ("gate", v) for k, v in data.items()}, coords={"gate": np.arange(n)}
    )
    truth = {
        "LOG10_NW": t,
        "DM": fw.dm.values[j // fw.sizes["mu"]],
        "MU": fw.mu.values[j % fw.sizes["mu"]],
        "RAIN_RATE": 10**t * flat["RAIN_RATE"],
    }
    return ds, truth


# --------------------------------------------------------------------------
# forward model and priors
# --------------------------------------------------------------------------


def test_forward_grid_matches_radar_from_dsd():
    """The grid forward model equals the integration of the spectrum."""
    fw = forward_grid("C", temperature=10.0)
    i, k = 20, 8  # Dm = 1.4 mm, mu = 3.5
    dm, mu = float(fw.dm[i]), float(fw.mu[k])
    lam = (4 + mu) / dm
    n0 = dsdmod._f_mu(mu) * dm ** (-mu) * 1e4
    d = np.linspace(0.0, 8.0, 4001)[1:]
    params = xr.Dataset({"N0": n0, "MU": mu, "LAMBDA": lam})
    nd = dsd_spectrum(params, d)
    sim = radar_from_dsd(nd, band="C", temperature=10.0)
    assert float(sim.DBZH) == pytest.approx(float(fw.DBZH[i, k]) + 40.0, abs=0.02)
    assert float(sim.ZDR) == pytest.approx(float(fw.ZDR[i, k]), abs=0.005)
    assert float(sim.KDP) == pytest.approx(1e4 * float(fw.KDP[i, k]), rel=5e-3)
    assert float(sim.AH) == pytest.approx(1e4 * float(fw.AH[i, k]), rel=5e-3)
    assert float(sim.DM) == pytest.approx(dm, rel=1e-3)
    assert float(sim.LWC) == pytest.approx(1e4 * float(fw.LWC[i, k]), rel=1e-3)
    assert fw.attrs["band"] == "C"


def test_generic_prior():
    p = dsd_prior()
    assert p.prior_mass.dims == ("dm", "mu")
    assert float(p.prior_mass.sum()) == pytest.approx(1.0)
    dm_mean = float((p.prior_mass.sum("mu") * p.dm).sum())
    assert dm_mean == pytest.approx(1.7, abs=0.05)
    assert float(p.log10_nw_mean.mean()) == pytest.approx(3.75)
    # the mode of mu follows the Cao et al. (2008) relation
    i = int(np.argmin(abs(p.dm.values - 2.0)))
    mu_mode = float(p.mu[int(np.argmax(p.prior_mass[i].values))])
    c2, c1, c0 = dsdmod.MU_LAMBDA["cao2008"]
    lam = (4 + mu_mode) / 2.0
    assert abs(c2 * lam**2 + c1 * lam + c0 - mu_mode) < 0.75


def test_learned_prior_from_spectra():
    """A prior learned from fitted gamma DSDs concentrates on their values."""
    rnd = np.random.default_rng(3)
    n = 300
    dm = rnd.normal(1.5, 0.15, n)
    mu = rnd.normal(4.0, 0.7, n)
    nw = 10 ** rnd.normal(3.6, 0.2, n)
    lam = (4 + mu) / dm
    n0 = nw * dsdmod._f_mu(mu) * dm ** (-mu)
    params = xr.Dataset(
        {"N0": ("t", n0), "MU": ("t", mu), "LAMBDA": ("t", lam)},
        coords={"t": np.arange(n)},
    )
    nd = dsd_spectrum(params, np.linspace(0.01, 8, 800))
    fits = fit_gamma_moments(nd)
    prior = dsd_prior(fits, defensive=0.0)
    m = prior.prior_mass
    assert float((m.sum("mu") * m.dm).sum()) == pytest.approx(1.5, abs=0.05)
    assert float((m.sum("dm") * m.mu).sum()) == pytest.approx(4.0, abs=0.3)
    j = np.unravel_index(int(np.argmax(m.values)), m.shape)
    assert float(prior.log10_nw_mean[j]) == pytest.approx(3.6, abs=0.1)
    assert 0.15 < float(prior.log10_nw_sd[j]) < 0.35
    assert prior.attrs["n_samples"] == n
    # the defensive mixture keeps every node possible
    mixed = dsd_prior(fits, weights=np.ones(n), bandwidth=(0.1, 0.5))
    assert (mixed.prior_mass > 0).all()
    assert mixed.attrs["defensive_weight"] == 0.01
    # retrieval with the learned prior on its own DSDs: Dm within 0.1 mm
    radar = radar_from_dsd(nd)
    post = dsd_bayesian(radar.rename(t="gate")[["DBZH", "ZDR"]], band="S", prior=prior)
    assert float(abs(post.DM - fits.DM.values).mean()) < 0.1


def test_prior_errors():
    with pytest.raises(ValueError, match="prior must be"):
        dsd_prior("nope")
    with pytest.raises(TypeError):
        dsd_prior(3)
    with pytest.raises(KeyError):
        dsd_prior(xr.Dataset({"DM": ("t", [1.0])}))
    few = xr.Dataset({k: ("t", np.ones(5)) for k in ("NW", "DM", "MU")})
    with pytest.raises(ValueError, match="at least 10"):
        dsd_prior(few)
    ds = xr.Dataset({"DBZH": ("g", [30.0]), "ZDR": ("g", [1.0])})
    with pytest.raises(TypeError):
        dsd_bayesian(ds, band="S", prior=1)
    with pytest.raises(KeyError, match="prior lacks"):
        dsd_bayesian(ds, band="S", prior=xr.Dataset())
    bad = dsd_prior().isel(dm=slice(0, 10))
    with pytest.raises(ValueError, match="grid of dsd_prior"):
        dsd_bayesian(ds, band="S", prior=bad)
    neg = dsd_prior()
    neg["log10_nw_sd"] = neg.log10_nw_sd * 0.0
    with pytest.raises(ValueError, match="positive"):
        dsd_bayesian(ds, band="S", prior=neg)


def test_packaged_prior():
    p = dsd_prior("perils2022")
    assert p.attrs["prior"] == "perils2022"
    assert float(p.prior_mass.sum()) == pytest.approx(1.0)
    assert p.attrs["n_samples"] > 500
    dm_mean = float((p.prior_mass.sum("mu") * p.dm).sum())
    assert 1.0 < dm_mean < 2.5


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------


def _brute_force(z, zdr, kdp, prior, band="S", errors=None):
    """Posterior by direct quadrature in log10 Nw on every node."""
    e = dict(bayes.ERRORS, **(errors or {}))
    fw = forward_grid(band)
    vz = e["zh"] ** 2 + e["zh_bias"] ** 2
    vd = e["zdr"] ** 2 + e["zdr_bias"] ** 2
    t = np.linspace(-1.0, 8.0, 18001)
    L = fw.DBZH.values.ravel()[:, None]
    ll = (
        -0.5 * (z - 10 * t - L) ** 2 / vz
        - 0.5 * np.log(2 * np.pi * vz)
        - 0.5 * (zdr - fw.ZDR.values.ravel()[:, None]) ** 2 / vd
        - 0.5 * np.log(2 * np.pi * vd)
    )
    if kdp is not None:
        vk = e["kdp"] ** 2 + (e["kdp_rel"] * kdp) ** 2
        ll = ll - 0.5 * (kdp - 10**t * fw.KDP.values.ravel()[:, None]) ** 2 / vk
        ll = ll - 0.5 * np.log(2 * np.pi * vk)
    pm = prior.log10_nw_mean.values.ravel()[:, None]
    ps = prior.log10_nw_sd.values.ravel()[:, None]
    lp = -0.5 * ((t - pm) / ps) ** 2 - np.log(ps * np.sqrt(2 * np.pi))
    with np.errstate(divide="ignore"):
        lmass = np.log(prior.prior_mass.values.ravel())[:, None]
    dens = np.exp(ll + lp + lmass)
    node = trapezoid(dens, t, axis=1)
    ev = node.sum()
    w = node / ev
    tm = (trapezoid(dens * t, t, axis=1)).sum() / ev
    dmv = np.repeat(fw.dm.values, fw.sizes["mu"])
    return {"logev": np.log(ev), "t": tm, "dm": (w * dmv).sum()}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("kdp", [None, 2.5])
def test_laplace_matches_quadrature(engine, kdp):
    """Evidence and means agree with brute-force integration."""
    prior = dsd_prior()
    data = {"DBZH": ("g", [45.0]), "ZDR": ("g", [1.8])}
    if kdp is not None:
        data["KDP"] = ("g", [kdp])
    ds = xr.Dataset(data)
    post = dsd_bayesian(ds, band="S", kdp="KDP" if kdp else None, engine=engine)
    ref = _brute_force(45.0, 1.8, kdp, prior)
    # without KDP the conditional posterior in t is Gaussian: exact
    tol = 1e-4 if kdp is None else 5e-2
    assert float(post.LOG_EVIDENCE[0]) == pytest.approx(ref["logev"], abs=tol)
    assert float(post.LOG10_NW[0]) == pytest.approx(ref["t"], abs=tol)
    assert float(post.DM[0]) == pytest.approx(ref["dm"], abs=tol)
    assert int(post.N_OBS[0]) == (2 if kdp is None else 3)


@pytest.mark.parametrize("kdp,ah", [(False, False), (True, False), (True, True)])
def test_calibrated_uncertainty(kdp, ah):
    """Truth drawn from the prior falls in the 68/95 % intervals as often."""
    prior = dsd_prior()
    n = 3000 if FAST == "compiled" else 300
    ds, truth = _draw(prior, n, seed=5, kdp=kdp, ah=ah)
    post = dsd_bayesian(
        ds, band="S", kdp="KDP" if kdp else None, ah="AH" if ah else None, engine=FAST
    )
    tol = 0.03 if n > 1000 else 0.08
    for name in ("LOG10_NW", "DM", "MU", "RAIN_RATE"):
        q = post[name + "_QUANTILES"]
        x = truth[name]
        c68 = np.mean((x >= q.sel(quantile=0.16)) & (x <= q.sel(quantile=0.84)))
        c95 = np.mean((x >= q.sel(quantile=0.025)) & (x <= q.sel(quantile=0.975)))
        assert c68 == pytest.approx(0.68, abs=tol), name
        assert c95 == pytest.approx(0.95, abs=tol * 0.7), name
    # the posterior mean beats the prior mean
    rmse = np.sqrt(np.mean((post.DM.values - truth["DM"]) ** 2))
    assert rmse < 0.5 * np.std(truth["DM"])


def test_kdp_and_ah_reduce_uncertainty():
    """K_DP and A_H add information on Nw: narrower posteriors."""
    prior = dsd_prior()
    ds, _ = _draw(
        prior, 200, seed=2, kdp=True, ah=True, errors={"kdp": 0.1, "ah": 0.002}
    )
    big = ds.DBZH > 40
    zz = dsd_bayesian(ds.where(big, drop=True), band="S", engine=FAST)
    zk = dsd_bayesian(ds.where(big, drop=True), band="S", kdp="KDP", engine=FAST)
    za = dsd_bayesian(
        ds.where(big, drop=True), band="S", kdp="KDP", ah="AH", engine=FAST
    )
    assert float(zk.LOG10_NW_SD.mean()) < float(zz.LOG10_NW_SD.mean())
    assert float(za.LOG10_NW_SD.mean()) <= float(zk.LOG10_NW_SD.mean())


def test_kdp_makes_nw_immune_to_calibration():
    """With precise KDP a Z_H calibration error barely moves Nw."""
    fw = forward_grid("S")
    i, k = 30, 8  # Dm 1.9 mm, mu 3.5
    t = 3.5
    z = 10 * t + float(fw.DBZH[i, k])
    ds = xr.Dataset(
        {
            "DBZH": ("g", [z, z + 3.0]),
            "ZDR": ("g", [float(fw.ZDR[i, k])] * 2),
            "KDP": ("g", [10**t * float(fw.KDP[i, k])] * 2),
        }
    )
    errors = {"zh_bias": 3.0, "kdp": 0.05, "kdp_rel": 0.02}
    with_k = dsd_bayesian(ds, band="S", kdp="KDP", errors=errors, engine=FAST)
    no_k = dsd_bayesian(ds, band="S", errors=errors, engine=FAST)
    dk = float(with_k.LOG10_NW[1] - with_k.LOG10_NW[0])
    dz = float(no_k.LOG10_NW[1] - no_k.LOG10_NW[0])
    assert abs(dk) < 0.4 * abs(dz)


@compiled_only
@pytest.mark.parametrize("kdp,ah", [(False, False), (True, True)])
def test_engines_agree(kdp, ah):
    ds, _ = _draw(dsd_prior(), 150, seed=7, kdp=kdp, ah=ah)
    # missing inputs and masked gates
    ds["DBZH"][3] = np.nan
    ds["ZDR"][4] = np.nan
    if kdp:
        ds["KDP"][5] = np.nan
    mask = xr.DataArray(np.arange(ds.sizes["gate"]) != 6, dims="gate")
    kw = {"band": "S", "kdp": "KDP" if kdp else None, "ah": "AH" if ah else None}
    a = dsd_bayesian(ds, mask=mask, engine="compiled", n_threads=3, **kw)
    b = dsd_bayesian(ds, mask=mask, engine="numpy", **kw)
    for v in a.data_vars:
        np.testing.assert_allclose(a[v], b[v], rtol=1e-7, atol=1e-7, err_msg=v)
    for g in (3, 4, 6):
        assert np.isnan(a.DM[g]) and np.isnan(a.DM_QUANTILES[:, g]).all()
    if kdp:
        assert int(a.N_OBS[5]) == (3 if ah else 2)
    # thread count does not change results
    c = dsd_bayesian(ds, mask=mask, engine="compiled", n_threads=1, **kw)
    xr.testing.assert_equal(a, c)


def test_outputs_and_attrs():
    ds, _ = _draw(dsd_prior(), 20, seed=1)
    out = dsd_bayesian(ds, band="S", quantiles=(0.1, 0.5, 0.9), engine=FAST)
    for v in ("LOG10_NW", "DM", "MU", "RAIN_RATE", "LWC"):
        assert v + "_SD" in out and v + "_QUANTILES" in out
        q = out[v + "_QUANTILES"]
        assert q.dims == ("quantile", "gate")
        assert (q.diff("quantile") >= 0).all()
        assert (q.sel(quantile=0.1) <= out[v] + 1e-9).mean() > 0.9
    np.testing.assert_allclose(out.NW, 10**out.LOG10_NW)
    assert out.RAIN_RATE.attrs["standard_name"] == "rainfall_rate"
    assert "standard_name" not in out.RAIN_RATE_SD.attrs
    assert out.attrs["method"] == "bayesian" and out.attrs["prior"] == "generic"
    assert "zh=1" in out.attrs["errors"]
    assert out.attrs["source_fields"] == "DBZH, ZDR"
    assert ((out.MU_MAP >= -0.5) & (out.MU_MAP <= 12)).all()
    assert (out.MISFIT >= 0).all() and np.isfinite(out.LOG_EVIDENCE).all()
    # rain rate is consistent with the deterministic retrieval's moments
    det = dsd(ds, "normalized", band="S")
    ratio = (out.RAIN_RATE / det.RAIN_RATE).median()
    assert 0.5 < float(ratio) < 2.0


def test_misfit_flags_non_rain():
    """Hail-like Z_H/Z_DR is far from every rain DSD: low evidence."""
    ds = xr.Dataset({"DBZH": ("g", [40.0, 62.0]), "ZDR": ("g", [1.5, -0.3])})
    out = dsd_bayesian(ds, band="S", engine=FAST)
    assert float(out.MISFIT[1]) > 10 * float(out.MISFIT[0]) + 5
    assert float(out.LOG_EVIDENCE[1]) < float(out.LOG_EVIDENCE[0]) - 5


def test_options_and_errors(monkeypatch):
    ds = xr.Dataset({"DBZH": ("g", [30.0]), "ZDR": ("g", [1.0])})
    with pytest.raises(ValueError, match="engine"):
        dsd_bayesian(ds, band="S", engine="fast")
    with pytest.raises(ValueError, match="quantiles"):
        dsd_bayesian(ds, band="S", quantiles=(0.0, 0.5))
    with pytest.raises(ValueError, match="prune"):
        dsd_bayesian(ds, band="S", prune=0)
    with pytest.raises(ValueError, match="unknown error"):
        dsd_bayesian(ds, band="S", errors={"zdr_noise": 0.1})
    with pytest.raises(ValueError, match="non-negative"):
        dsd_bayesian(ds, band="S", errors={"zdr": -1})
    with pytest.raises(ValueError, match="both be zero"):
        dsd_bayesian(ds, band="S", errors={"zdr": 0, "zdr_bias": 0})
    with pytest.raises(ValueError, match="band"):
        dsd_bayesian(ds, band="K")
    with pytest.raises(KeyError):
        dsd_bayesian(ds, band="S", kdp="KDP")
    with pytest.raises(KeyError):
        dsd_bayesian(ds, band="S", ah="AH")
    with pytest.raises(KeyError):
        dsd_bayesian(ds.rename(ZDR="X"), band="S")
    with pytest.warns(UserWarning, match="band"):
        dsd_bayesian(ds, engine=FAST)
    # a DataArray KDP and AH
    out = dsd_bayesian(
        ds,
        band="S",
        kdp=xr.DataArray([0.5], dims="g"),
        ah=xr.DataArray([0.01], dims="g"),
        engine=FAST,
    )
    assert int(out.N_OBS[0]) == 4
    monkeypatch.setattr(bayes, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError):
        dsd_bayesian(ds, band="S", engine="compiled")
    out = dsd_bayesian(ds, band="S")  # falls back to NumPy
    assert np.isfinite(out.DM[0])


@compiled_only
def test_kernel_argument_checks():
    k = bayes._dsd_bayes
    grid = bayes._grid("S", 20.0, dsd_prior())
    z = [np.array([30.0])]
    args = dict(
        dbzh=z,
        zdr=[np.array([1.0])],
        kdp=[None],
        ah=[None],
        mask=[None],
        grid=grid,
        n_dm=81,
        n_mu=26,
        dm0=0.4,
        ddm=0.05,
        mu0=-0.5,
        dmu=0.5,
        errors=[2.0, 0.05, 0.3, 0.1, 0.01, 0.2, 15.0],
        quantiles=[0.5],
        zq=[0.0],
    )
    k.retrieve(**args)
    for key, val, msg in (
        ("zdr", [], "one entry"),
        ("grid", grid[:5], "grid must"),
        ("errors", [1.0], "seven"),
        ("errors", [0.0, 0.05, 0.3, 0.1, 0.01, 0.2, 15.0], "positive"),
        ("zq", [], "zq"),
        ("quantiles", [1.5], "quantiles"),
        ("zdr", [np.array([1.0, 2.0])], "same size"),
        ("kdp", [np.array([1.0, 2.0])], "shape of dbzh"),
    ):
        bad = dict(args, **{key: val})
        if key == "quantiles":
            bad["zq"] = [0.0]
        with pytest.raises(ValueError, match=msg):
            k.retrieve(**bad)
    g2 = grid.copy()
    g2[10, 0] = 0.0
    with pytest.raises(ValueError, match="prior standard"):
        k.retrieve(**dict(args, grid=g2))
    # no quantiles
    out = k.retrieve(**dict(args, quantiles=[], zq=[]))
    assert out[0].shape == (16, 1)


# --------------------------------------------------------------------------
# volumes and accessors
# --------------------------------------------------------------------------


def _volume():
    sweeps = {}
    for i in range(3):
        ds, _ = _draw(dsd_prior(), 60, seed=10 + i, kdp=True)
        ds = xr.Dataset(
            {k: (("azimuth", "range"), v.values.reshape(6, 10)) for k, v in ds.items()},
            coords={"azimuth": np.arange(6.0), "range": np.arange(10.0) * 250},
        )
        if i == 2:
            ds = ds.drop_vars("ZDR")
        sweeps[f"sweep_{i}"] = ds
    root = xr.Dataset(attrs={"scan_name": "VCP-212"})
    return xr.DataTree.from_dict({"/": root, **sweeps})


def test_volume_and_accessors():
    tree = _volume()
    out = tree.radarx.dsd_bayesian(kdp="KDP", engine=FAST)
    assert sorted(out.children) == ["sweep_0", "sweep_1"]
    assert out["sweep_0"].ds.attrs["band"] == "S"
    one = tree["sweep_1"].to_dataset(inherit=False)
    ref = one.radarx.dsd_bayesian(kdp="KDP", band="S", engine=FAST)
    xr.testing.assert_allclose(out["sweep_1"].to_dataset(inherit=False), ref)
    assert out["sweep_0"].ds.DM.dims == ("azimuth", "range")
    # masks and KDP given as trees
    zero = tree["sweep_0"].to_dataset(inherit=False)
    masks = xr.DataTree.from_dict(
        {
            "sweep_0": xr.Dataset({"rain": zero.DBZH > 30}),
            "sweep_1": xr.Dataset({"rain": one.DBZH > -100}),
        }
    )
    kdps = xr.DataTree.from_dict({"sweep_0": xr.Dataset({"KDP": one.KDP * 0 + 1.0})})
    out2 = dsd_bayesian(tree, kdp=kdps, mask=masks, ah=None, engine=FAST)
    rain = (tree["sweep_0"].ds.DBZH > 30).values
    assert np.isnan(out2["sweep_0"].ds.DM.values[~rain]).all()
    assert (out2["sweep_1"].ds.N_OBS == 2).all()  # no KDP node for sweep_1
    assert (out2["sweep_0"].ds.N_OBS.values[rain] == 3).all()
    ahs = xr.DataTree.from_dict({"sweep_1": xr.Dataset({"AH": one.KDP * 0 + 0.01})})
    out3 = dsd_bayesian(tree, ah=ahs, engine=FAST)
    assert (out3["sweep_1"].ds.N_OBS == 3).all()
    bad = xr.DataTree.from_dict({"sweep_0": xr.Dataset({"a": one.DBZH, "b": one.DBZH})})
    with pytest.raises(ValueError, match="exactly one"):
        dsd_bayesian(tree, mask=bad, engine=FAST)
    with pytest.raises(KeyError, match="no sweep"):
        dsd_bayesian(
            xr.DataTree.from_dict({"sweep_0": xr.Dataset({"X": ("a", [1.0])})}),
            band="S",
        )
    # a mask tree without a node for a sweep leaves that sweep unmasked
    partial = xr.DataTree.from_dict({"sweep_0": xr.Dataset({"rain": zero.DBZH > 30})})
    out4 = dsd_bayesian(tree, mask=partial, engine=FAST)
    assert np.isfinite(out4["sweep_1"].ds.DM).all()


def test_volume_estimated_kdp():
    nray, ng, dr = 8, 120, 250.0
    dm, mu, nw = 1.8, 3.0, 10**3.8
    lam = (4 + mu) / dm
    n0 = nw * dsdmod._f_mu(mu) * dm ** (-mu)
    zh, zv, kdp = dsdmod._gamma_integrals(
        "S", 20.0, np.array(n0), np.array(mu), np.array(lam)
    )
    rng = (np.arange(ng) + 0.5) * dr
    phi = 2.0 * kdp * rng / 1000.0 + 30.0
    shape = (nray, ng)
    ds = xr.Dataset(
        {
            "DBZH": (("azimuth", "range"), np.full(shape, 10 * np.log10(zh))),
            "ZDR": (("azimuth", "range"), np.full(shape, 10 * np.log10(zh / zv))),
            "PHIDP": (("azimuth", "range"), np.broadcast_to(phi, shape).copy()),
            "RHOHV": (("azimuth", "range"), np.full(shape, 0.99)),
        },
        coords={
            "azimuth": np.linspace(0.5, 359.5, nray),
            "range": rng,
            "elevation": ("azimuth", np.full(nray, 0.5)),
        },
    )
    out = dsd_bayesian(ds, band="S", kdp="estimate", engine=FAST)
    mid = out.isel(range=slice(30, 90))
    assert float(mid.N_OBS.mean()) == 3
    assert float(abs(mid.DM - dm).mean()) < 0.15
    tree = xr.DataTree.from_dict({"sweep_0": ds})
    out_t = dsd_bayesian(tree, band="S", kdp="estimate", engine=FAST)
    xr.testing.assert_allclose(out_t["sweep_0"].to_dataset(inherit=False), out)


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def klbb():
    xd = pytest.importorskip("xradar")
    from open_radar_data import DATASETS

    file = DATASETS.fetch("KLBB20160601_150025_V06")
    dtree = xd.io.open_nexradlevel2_datatree(file, sweep=[0])
    ds = dtree["sweep_0"].to_dataset(inherit=False)
    for var, lim in (("DBZH", -32.0), ("ZDR", -12.9), ("RHOHV", 0.21)):
        ds[var] = ds[var].where(ds[var] > lim)
    return ds


def test_real_nexrad_sweep(klbb):
    """S-band NEXRAD rain: plausible posteriors, close to the deterministic DSD."""
    ds = klbb
    rain = (ds.RHOHV > 0.97) & (ds.DBZH > 30) & (ds.DBZH < 50)
    sub = ds.where(rain)
    post = dsd_bayesian(sub, band="S", engine=FAST)
    det = dsd(sub, "normalized", band="S")
    ok = rain.values & np.isfinite(det.DM.values)
    assert ok.sum() > 500
    assert np.isfinite(post.DM.values[ok]).all()
    assert 0.8 < np.median(post.DM.values[ok]) < 2.5
    assert 2.5 < np.median(post.LOG10_NW.values[ok]) < 5.0
    # the deterministic Dm lies within the 95 % interval at most gates
    q = post.DM_QUANTILES
    inside = (det.DM >= q.sel(quantile=0.025) - 0.05) & (
        det.DM <= q.sel(quantile=0.975) + 0.05
    )
    assert inside.values[ok].mean() > 0.8
    # the misfit of rain gates is mostly small (chi-squared with 2 inputs)
    assert np.median(post.MISFIT.values[ok]) < 3.0
