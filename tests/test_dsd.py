#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the drop size distribution retrieval
==============================================
"""
import importlib

import numpy as np
import pytest
import xarray as xr
from scipy.integrate import trapezoid

import radarx  # noqa: F401
from radarx.retrieve import (
    dsd,
    dsd_spectrum,
    fit_gamma_moments,
    parsivel_bins,
    radar_from_dsd,
    scattering_table,
)

dsdmod = importlib.import_module("radarx.retrieve.dsd")

ENGINES = ["numpy"] + (["compiled"] if dsdmod.HAS_COMPILED_KERNEL else [])
compiled_only = pytest.mark.skipif(
    not dsdmod.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)


def _constrained_truth(n=400, relation="cao2008", seed=0):
    rnd = np.random.default_rng(seed)
    c2, c1, c0 = dsdmod.MU_LAMBDA[relation]
    lam = rnd.uniform(1.2, 15.0, n)
    mu = c2 * lam**2 + c1 * lam + c0
    keep = mu > -0.9
    lam, mu = lam[keep], mu[keep]
    nw = 10 ** rnd.uniform(2.0, 5.0, lam.size)
    n0 = nw * dsdmod._f_mu(mu) * ((mu + 4.0) / lam) ** (-mu)
    return n0, mu, lam


def _normalized_truth(n=400, mu=3.0, seed=1):
    rnd = np.random.default_rng(seed)
    dm = rnd.uniform(0.6, 3.2, n)
    nw = 10 ** rnd.uniform(2.0, 4.8, n)
    lam = (4.0 + mu) / dm
    n0 = nw * dsdmod._f_mu(mu) * dm ** (-mu)
    return n0, np.full(n, mu), lam, dm, nw


def _radar(band, n0, mu, lam, temperature=20.0):
    zh, zv, kdp = dsdmod._gamma_integrals(band, temperature, n0, mu, lam)
    return xr.Dataset(
        {
            "DBZH": ("gate", 10 * np.log10(zh), {"units": "dBZ"}),
            "ZDR": ("gate", 10 * np.log10(zh / zv), {"units": "dB"}),
            "KDP": ("gate", kdp, {"units": "degrees/km"}),
        },
        coords={"gate": np.arange(zh.size)},
    )


# --------------------------------------------------------------------------
# scattering tables
# --------------------------------------------------------------------------


@pytest.mark.parametrize("band", ["S", "C", "X"])
def test_scattering_table_rayleigh(band):
    """Small drops scatter like Rayleigh spheres; big drops are oblate."""
    tab = scattering_table(band, 10.0)
    wl = tab.attrs["wavelength"]
    d = tab.diameter.values
    assert tab.attrs["band"] == band
    assert tab.sigma_h.attrs["units"] == "mm2"
    small = d <= 0.5
    rayleigh = np.pi**5 * 0.93 * d[small] ** 6 / wl**4
    np.testing.assert_allclose(tab.sigma_h.values[small], rayleigh, rtol=0.03)
    zdr = 10 * np.log10(tab.sigma_h / tab.sigma_v)
    assert abs(float(zdr.sel(diameter=0.5))) < 0.05
    assert 1.0 < float(zdr.sel(diameter=3.0)) < 2.5
    assert (tab.kdp.sel(diameter=slice(1, 4)) > 0).all()
    assert float(tab.axis_ratio.sel(diameter=4.0)) < 0.8


def test_scattering_table_temperature_interpolation():
    a, b = scattering_table("C", 10.0), scattering_table("C", 20.0)
    mid = scattering_table("C", 15.0)
    np.testing.assert_allclose(mid.kdp, 0.5 * (a.kdp + b.kdp))
    with pytest.raises(ValueError, match="temperature"):
        scattering_table("C", 40.0)
    with pytest.raises(ValueError, match="band"):
        scattering_table("Ka")


# --------------------------------------------------------------------------
# synthetic retrievals
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("band", ["S", "C", "X"])
@pytest.mark.parametrize("relation", ["cao2008", "zhang2001"])
def test_constrained_recovers_gamma(band, relation, engine):
    """Simulated ZH, ZDR of constrained-gamma DSDs give the DSDs back."""
    n0, mu, lam = _constrained_truth(relation=relation)
    ds = _radar(band, n0, mu, lam)
    out = dsd(ds, band=band, mu_lambda=relation, engine=engine)
    np.testing.assert_allclose(out.LAMBDA, lam, rtol=1e-4)
    np.testing.assert_allclose(out.MU, mu, atol=1e-4)
    np.testing.assert_allclose(out.N0, n0, rtol=1e-3)
    np.testing.assert_allclose(out.DM, (mu + 4) / lam, rtol=1e-4)
    np.testing.assert_allclose(out.D0, (mu + 3.67) / lam, rtol=1e-4)
    assert out.RAIN_RATE.attrs["standard_name"] == "rainfall_rate"
    assert out.RAIN_RATE.attrs["units"] == "mm h-1"
    assert out.attrs["method"] == "constrained"
    assert out.attrs["band"] == band
    assert out.attrs["source_fields"] == "DBZH, ZDR"
    assert out.DBZH.dims if "DBZH" in out else out.N0.dims == ("gate",)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("band", ["S", "C", "X"])
@pytest.mark.parametrize("mu", [0.0, 3.0, 6.0])
def test_normalized_recovers_gamma(band, mu, engine):
    n0, mus, lam, dm, nw = _normalized_truth(mu=mu)
    ds = _radar(band, n0, mus, lam)
    out = dsd(ds, "normalized", band=band, mu=mu, engine=engine)
    np.testing.assert_allclose(out.DM, dm, rtol=1e-4)
    np.testing.assert_allclose(out.NW, nw, rtol=1e-3)
    np.testing.assert_allclose(out.MU, mu)
    assert out.attrs["mu"] == mu


def test_closed_form_moments_match_integration():
    """Rain rate, LWC, Dm and Nw against numerical integration of N(D)."""
    n0, mu, lam = _constrained_truth(n=50)
    out = dsd(_radar("S", n0, mu, lam), band="S")
    d = np.linspace(1e-4, 30.0, 300001)
    for i in range(0, mu.size, 7):
        nd = n0[i] * d ** mu[i] * np.exp(-lam[i] * d)
        m3 = trapezoid(nd * d**3, d)
        m4 = trapezoid(nd * d**4, d)
        v = 9.65 - 10.3 * np.exp(-0.6 * d)
        rate = 6e-4 * np.pi * trapezoid(v * nd * d**3, d)
        assert np.isclose(out.LWC[i], np.pi / 6 * 1e-3 * m3, rtol=1e-3)
        assert np.isclose(out.RAIN_RATE[i], rate, rtol=1e-3)
        assert np.isclose(out.DM[i], m4 / m3, rtol=1e-3)
        assert np.isclose(out.NW[i], 256 / 6 * m3 / (m4 / m3) ** 4, rtol=1e-3)
        # D0: half of the water mass below it (Ulbrich's approximation)
        cum = np.cumsum(nd * d**3)
        d0 = d[np.searchsorted(cum, 0.5 * cum[-1])]
        assert abs(float(out.D0[i]) - d0) < 0.03 * d0


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("method", ["constrained", "normalized"])
def test_kdp_intercept_is_immune_to_z_bias(method, engine):
    """With KDP the intercept does not depend on the ZH calibration."""
    if method == "constrained":
        n0, mu, lam = _constrained_truth()
        ds = _radar("S", n0, mu, lam)
    else:
        n0, mus, lam, dm, nw = _normalized_truth()
        ds = _radar("S", n0, mus, lam)
    ref = dsd(ds, method, band="S", engine=engine)
    biased = ds.assign(DBZH=ds.DBZH + 3.0)
    off = dsd(biased, method, band="S", engine=engine)
    on = dsd(biased, method, band="S", kdp="KDP", kdp_min=0.5, engine=engine)
    strong = ds.KDP.values >= 0.5
    assert strong.sum() > 20 and (~strong).sum() > 20
    np.testing.assert_allclose(on.NW.values[strong], ref.NW.values[strong], rtol=1e-3)
    np.testing.assert_allclose(on.NW.values[~strong], off.NW.values[~strong])
    np.testing.assert_allclose(off.NW / ref.NW, 10**0.3, rtol=1e-6)
    assert on.attrs["kdp_min"] == 0.5
    assert on.attrs["source_fields"] == "DBZH, ZDR, KDP"


@compiled_only
@pytest.mark.parametrize("kdp", [None, "KDP"])
@pytest.mark.parametrize("method", ["constrained", "normalized"])
def test_engines_agree(method, kdp):
    """Compiled kernel and NumPy oracle agree to rounding (rtol 1e-10)."""
    rnd = np.random.default_rng(3)
    n = 20000
    ds = xr.Dataset(
        {
            "DBZH": ("gate", rnd.uniform(-10, 65, n)),
            "ZDR": ("gate", rnd.uniform(-1, 5, n)),
            "KDP": ("gate", rnd.uniform(-1, 6, n)),
            "RAIN": ("gate", rnd.uniform(size=n) > 0.1),
        }
    )
    ds["DBZH"][::17] = np.nan
    ds["ZDR"][::19] = np.nan
    ds["KDP"][::23] = np.nan
    for band in ("S", "C", "X"):
        kw = dict(band=band, kdp=kdp, mask="RAIN", temperature=7.5)
        a = dsd(ds, method, engine="compiled", n_threads=3, **kw)
        b = dsd(ds, method, engine="numpy", **kw)
        for name in a.data_vars:
            np.testing.assert_allclose(a[name], b[name], rtol=1e-10, err_msg=name)
        assert np.isnan(a.N0.values[~ds.RAIN.values]).all()
        assert np.isnan(a.N0.values[::17]).all()
        assert np.isfinite(a.N0.values).mean() > 0.3
        if kdp is None:
            assert np.isfinite(b.N0.values).sum() == np.isfinite(a.N0.values).sum()


def test_zdr_outside_table_is_clipped():
    ds = xr.Dataset({"DBZH": ("g", [30.0, 30.0, 30.0]), "ZDR": ("g", [-2, 0.5, 9])})
    out = dsd(ds, band="S", nw_range=None)
    tab = dsdmod._lookup_table(
        "S", 20.0, "constrained", dsdmod.MU_LAMBDA["cao2008"], None
    )
    assert np.isclose(out.LAMBDA[0], tab["lam"][0])
    assert np.isclose(out.LAMBDA[2], tab["lam"][-1])
    assert out.LAMBDA[0] > out.LAMBDA[1] > out.LAMBDA[2]


# --------------------------------------------------------------------------
# xarray layer: grids, masks, volumes, options
# --------------------------------------------------------------------------


def test_grid_and_broadcast_mask():
    """Any dimensions (here a 3-D grid) with a mask that broadcasts."""
    n0, mu, lam = _constrained_truth(n=60)
    sim = _radar("X", n0[:48], mu[:48], lam[:48])
    shape = (2, 4, 6)
    ds = xr.Dataset(
        {
            "reflectivity": (("z", "y", "x"), sim.DBZH.values.reshape(shape)),
            "differential_reflectivity": (
                ("z", "y", "x"),
                sim.ZDR.values.reshape(shape),
            ),
        },
        coords={"z": [500.0, 1000.0], "y": np.arange(4.0), "x": np.arange(6.0)},
    )
    low = xr.DataArray([True, False], dims="z", coords={"z": ds.z})
    out = ds.radarx.dsd(band="X", mask=low)
    assert out.N0.dims == ("z", "y", "x")
    np.testing.assert_allclose(out.z, ds.z)
    np.testing.assert_allclose(out.LAMBDA.isel(z=0).values.ravel(), lam[:24], rtol=1e-4)
    assert np.isnan(out.LAMBDA.isel(z=1)).all()


def _sweep(nray=36, ng=400, dr=250.0, seed=0, zdr=True):
    """Sweep of rain cells with consistent ZH, ZDR, KDP and phase."""
    rnd = np.random.default_rng(seed)
    r = (np.arange(ng) + 0.5) * dr / 1000.0
    dm = 1.0 + 1.2 * np.exp(-0.5 * ((r - 40) / 8) ** 2)
    nw = 10 ** (3.4 + 0.9 * np.exp(-0.5 * ((r - 40) / 10) ** 2))
    mu = 3.0
    lam = (4 + mu) / dm
    n0 = nw * dsdmod._f_mu(mu) * dm ** (-mu)
    zh, zv, kdp = dsdmod._gamma_integrals("S", 20.0, n0, mu, lam)
    phi = 2.0 * np.cumsum(kdp) * dr / 1000.0 + 40.0
    shape = (nray, ng)
    data = {
        "DBZH": np.broadcast_to(10 * np.log10(zh), shape).copy(),
        "PHIDP": phi + rnd.normal(0, 1.0, shape),
        "RHOHV": np.full(shape, 0.99),
    }
    if zdr:
        data["ZDR"] = np.broadcast_to(10 * np.log10(zh / zv), shape).copy()
    ds = xr.Dataset(
        {k: (("azimuth", "range"), v) for k, v in data.items()},
        coords={
            "azimuth": np.linspace(0.5, 359.5, nray),
            "range": (np.arange(ng) + 0.5) * dr,
            "elevation": ("azimuth", np.full(nray, 0.5)),
        },
    )
    return ds, dm, nw, kdp


def _volume(**kw):
    a, *_ = _sweep(seed=0)
    b, *_ = _sweep(seed=1)
    c, *_ = _sweep(seed=2, zdr=False)
    root = xr.Dataset(
        coords={"latitude": 33.9, "longitude": -88.3, "altitude": 140.0},
        attrs={"scan_name": "VCP-212"},
    )
    return xr.DataTree.from_dict({"/": root, "sweep_0": a, "sweep_1": b, "sweep_2": c})


def test_volume_one_kernel_call_and_skips_sweeps():
    tree = _volume()
    out = tree.radarx.dsd("normalized")
    # sweep_2 lacks ZDR and is skipped; WSR-88D VCP: S band, no warning
    assert sorted(out.children) == ["sweep_0", "sweep_1"]
    assert out["sweep_0"].ds.attrs["band"] == "S"
    ref = dsd(tree["sweep_0"].to_dataset(inherit=False), "normalized", band="S")
    xr.testing.assert_allclose(out["sweep_0"].to_dataset(inherit=False), ref)
    assert float(out.ds.latitude) == 33.9


def test_volume_masks_and_kdp_trees():
    tree = _volume()
    masks = xr.DataTree.from_dict(
        {
            "sweep_0": xr.Dataset({"rain": tree["sweep_0"].ds.DBZH > 30}),
        }
    )
    out = dsd(tree, mask=masks, band="S")
    z0 = tree["sweep_0"].ds.DBZH
    assert np.isnan(out["sweep_0"].ds.N0.values[(z0 <= 30).values]).all()
    # sweep_1 has no mask node: all gates retrieved
    assert np.isfinite(out["sweep_1"].ds.N0).all()
    out = dsd(tree, mask={"sweep_1": tree["sweep_1"].ds.DBZH > 30}, band="S")
    assert np.isfinite(out["sweep_0"].ds.N0).all()
    out = dsd(tree, mask="RHOHV", band="S")
    assert np.isfinite(out["sweep_0"].ds.N0).all()
    bad = xr.DataTree.from_dict({"sweep_0": xr.Dataset({"a": z0 > 0, "b": z0 > 0})})
    with pytest.raises(ValueError, match="one boolean"):
        dsd(tree, mask=bad, band="S")
    kdps = tree.radarx.kdp()
    out = dsd(tree, "normalized", kdp=kdps, band="S")
    assert out["sweep_0"].ds.attrs["source_fields"] == "DBZH, ZDR, KDP"
    out = dsd(tree, "normalized", kdp="KDP", band="S")  # field missing: unused
    assert out["sweep_0"].ds.attrs["source_fields"] == "DBZH, ZDR"


@pytest.mark.parametrize("engine", ENGINES)
def test_estimated_kdp_gives_nw(engine):
    """kdp='estimate' runs estimate_kdp; Nw from KDP matches the truth."""
    ds, dm, nw, kdp = _sweep()
    out = dsd(ds, "normalized", kdp="estimate", band="S", engine=engine)
    strong = kdp >= 1.5
    assert strong.sum() > 20
    ratio = out.NW.values[:, strong] / nw[strong]
    assert np.median(np.abs(ratio - 1)) < 0.15
    np.testing.assert_allclose(
        out.DM.values[:, ~strong],
        np.broadcast_to(dm[~strong], out.DM.values[:, ~strong].shape),
        rtol=1e-4,
    )
    tree = _volume()
    out = tree.radarx.dsd("normalized", kdp="estimate", engine=engine)
    assert out["sweep_0"].ds.attrs["kdp_min"] == 1.0


def test_band_inference():
    ds = _radar("C", *_constrained_truth(n=10))
    for freq, band in ((2.8e9, "S"), (5.6e9, "C"), (9.4e9, "X")):
        out = dsd(ds.assign_coords(frequency=freq))
        assert out.attrs["band"] == band
    out = dsd(ds.assign_attrs(frequency=5.5e9))
    assert out.attrs["band"] == "C"
    with pytest.raises(ValueError, match="GHz"):
        dsd(ds.assign_coords(frequency=35e9))
    with pytest.warns(UserWarning, match="assuming S"):
        out = dsd(ds)
    assert out.attrs["band"] == "S"
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "radar_parameters": xr.Dataset({"frequency": 9.4e9}),
            "sweep_0": ds,
        }
    )
    assert dsd(tree)["sweep_0"].ds.attrs["band"] == "X"


def test_options_and_errors(monkeypatch):
    ds = _radar("S", *_constrained_truth(n=10))
    out = dsd(ds, band="s", mu_lambda=(-0.0201, 0.902, -1.718), temperature=0)
    ref = dsd(ds, band="S", temperature=0)
    xr.testing.assert_allclose(out, ref)
    assert out.attrs["mu_lambda"].startswith("custom")
    with pytest.raises(ValueError, match="method"):
        dsd(ds, "power-law", band="S")
    with pytest.raises(ValueError, match="engine"):
        dsd(ds, band="S", engine="gpu")
    with pytest.raises(ValueError, match="mu_lambda"):
        dsd(ds, band="S", mu_lambda="brandes")
    with pytest.raises(ValueError, match="three"):
        dsd(ds, band="S", mu_lambda=(1.0, 2.0))
    with pytest.raises(ValueError, match="no usable"):
        dsd(ds, band="S", mu_lambda=(0.0, 0.0, -5.0))
    with pytest.raises(ValueError, match="larger than -1"):
        dsd(ds, "normalized", band="S", mu=-1.0)
    with pytest.raises(KeyError, match="'Z'"):
        dsd(ds, band="S", dbzh="Z")
    with pytest.raises(KeyError, match="none of"):
        dsd(ds.drop_vars("ZDR"), band="S")
    with pytest.raises(KeyError, match="'K'"):
        dsd(ds, band="S", kdp="K")
    with pytest.raises(KeyError, match="mask"):
        dsd(ds, band="S", mask="rain")
    with pytest.raises(TypeError, match="mask"):
        dsd(ds, band="S", mask=np.ones(10, bool))
    with pytest.raises(KeyError, match="no sweep"):
        dsd(xr.DataTree.from_dict({"sweep_0": ds.drop_vars("ZDR")}), band="S")
    allmasked = dsd(ds, band="S", mask=ds.DBZH > 1000, engine="numpy")
    assert np.isnan(allmasked.N0).all()
    monkeypatch.setattr(dsdmod, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError):
        dsd(ds, band="S", engine="compiled")
    out = dsd(ds, band="S")  # falls back to NumPy
    xr.testing.assert_allclose(
        out, ref.pipe(lambda x: dsd(ds, band="S", engine="numpy"))
    )


@compiled_only
def test_kernel_argument_checks():
    t = dsdmod._lookup_table(
        "S", 20.0, "constrained", dsdmod.MU_LAMBDA["cao2008"], None
    )
    cols = [t[k] for k in dsdmod._TABLE_KEYS]
    z = np.zeros(3)
    k = dsdmod._dsd
    with pytest.raises(ValueError, match="one entry"):
        k.retrieve([z], [z], [], [None], *cols)
    with pytest.raises(ValueError, match="same size"):
        k.retrieve([z], [z], [None], [None], cols[0], cols[1][:5], *cols[2:])
    with pytest.raises(ValueError, match="increasing"):
        k.retrieve([z], [z], [None], [None], cols[0][::-1].copy(), *cols[1:])
    with pytest.raises(ValueError, match="two entries"):
        k.retrieve([z], [z], [None], [None], *(c[:1].copy() for c in cols))
    with pytest.raises(ValueError, match="1-D"):
        k.retrieve([z], [z[:2]], [None], [None], *cols)
    with pytest.raises(ValueError, match="shape of dbzh"):
        k.retrieve([z], [z], [z[:2]], [None], *cols)
    (out,) = k.retrieve([np.array([30.0])], [np.array([1.0])], [None], [None], *cols)
    assert out.shape == (8, 1) and np.isfinite(out).all()


# --------------------------------------------------------------------------
# disdrometer helpers
# --------------------------------------------------------------------------


def test_parsivel_bins():
    bins = parsivel_bins()
    assert bins.sizes["diameter"] == 32
    np.testing.assert_allclose(
        bins.diameter_upper[:-1], bins.diameter_lower[1:], atol=1e-3
    )
    assert float(bins.diameter_upper[-1]) == 26.0
    assert bins.bin_width.attrs["units"] == "mm"


def test_spectrum_fit_and_forward_roundtrip():
    """Rebuild N(D), fit it by moments and simulate radar variables."""
    n0, mu, lam = _constrained_truth(n=30)
    sim = _radar("S", n0, mu, lam)
    out = dsd(sim, band="S")
    # on the scattering-table grid with trapezoid weights: exact forward model
    tab = scattering_table("S", 20.0)
    grid = tab.diameter.assign_coords(
        bin_width=("diameter", dsdmod._trapezoid_weights(tab.diameter.values))
    )
    nd = dsd_spectrum(out, grid)
    assert nd.dims == ("gate", "diameter")
    assert nd.attrs["units"] == "m-3 mm-1"
    fwd = radar_from_dsd(nd, "S", 20.0)
    np.testing.assert_allclose(fwd.DBZH, sim.DBZH, atol=1e-3)
    np.testing.assert_allclose(fwd.ZDR, sim.ZDR, atol=1e-4)
    np.testing.assert_allclose(fwd.KDP, sim.KDP, rtol=1e-3)
    assert ((fwd.RHOHV > 0.9) & (fwd.RHOHV <= 1.0)).all()
    assert (fwd.AH > 0).all() and (fwd.ADP > 0).all()
    assert fwd.DBZH.attrs["units"] == "dBZ"
    # method of moments on a fine grid recovers the gamma parameters
    fine = np.linspace(0.005, 25.0, 5000)
    fit = fit_gamma_moments(dsd_spectrum(out, fine))
    np.testing.assert_allclose(fit.MU, mu, atol=0.02)
    np.testing.assert_allclose(fit.LAMBDA, lam, rtol=0.01)
    np.testing.assert_allclose(fit.RAIN_RATE, out.RAIN_RATE, rtol=0.01)
    # Parsivel bins (default): coarse, but close for moderate drops
    pars = dsd_spectrum(out)
    assert pars.sizes["diameter"] == 32
    fit = fit_gamma_moments(pars)
    moderate = (out.DM > 1.0).values
    np.testing.assert_allclose(fit.DM[moderate], out.DM[moderate], rtol=0.05)
    fwd = radar_from_dsd(pars, "S")
    np.testing.assert_allclose(fwd.DBZH[moderate], sim.DBZH[moderate], atol=1.0)


def test_spectrum_inputs():
    params = xr.Dataset({"N0": 8000.0, "MU": 0.0, "LAMBDA": 2.0})
    nd = dsd_spectrum(params, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(nd, 8000 * np.exp(-2 * np.array([1, 2, 3.0])))
    np.testing.assert_allclose(nd.bin_width, 1.0)
    da = xr.DataArray([1.0], dims="diameter").assign_coords(
        bin_width=("diameter", [0.5])
    )
    assert dsd_spectrum(params, da).bin_width.item() == 0.5
    assert dsd_spectrum(params, [2.0]).bin_width.item() == 1.0
    zero = xr.DataArray(
        np.zeros((2, 5)), dims=("t", "D"), coords={"D": np.arange(1.0, 6.0)}
    )
    fit = fit_gamma_moments(zero, dim="D")
    assert np.isnan(fit.MU).all()
    fwd = radar_from_dsd(zero.rename(D="diameter").isel(t=0) + 100.0)
    assert np.isfinite(fwd.DBZH)


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def klbb():
    xd = pytest.importorskip("xradar")
    from open_radar_data import DATASETS

    file = DATASETS.fetch("KLBB20160601_150025_V06")
    dtree = xd.io.open_nexradlevel2_datatree(file, sweep=[0, 1, 2])
    for name in ("sweep_0", "sweep_1", "sweep_2"):
        ds = dtree[name].to_dataset(inherit=False)
        for var, lim in (("DBZH", -32.0), ("ZDR", -12.9), ("RHOHV", 0.21)):
            if var in ds:
                ds[var] = ds[var].where(ds[var] > lim)
        dtree[name] = ds
    return dtree


def test_real_nexrad_volume(klbb):
    """S-band NEXRAD: plausible DSDs in rain, rain rate close to Z-R."""
    out = klbb.radarx.dsd()  # default call: band from the VCP, no warning
    assert "sweep_0" in out.children
    ds = klbb["sweep_0"].to_dataset()
    res = out["sweep_0"].to_dataset()
    assert res.attrs["band"] == "S"
    assert res.N0.dims == ds.DBZH.dims
    rain = ((ds.RHOHV > 0.97) & (ds.DBZH > 35) & (ds.DBZH < 55)).values
    assert rain.sum() > 500
    d0 = res.D0.values[rain]
    # gates with ZDR near 0 dB at 35-45 dBZ are not rain-like (Nw > 1e6)
    assert np.isfinite(d0).mean() > 0.9
    assert 0.8 < np.nanmedian(d0) < 2.5
    lognw = np.log10(res.NW.values[rain])
    assert 2.0 < np.nanmedian(lognw) < 5.0
    z = 10 ** (ds.DBZH.values[rain] / 10)
    r_zr = (z / 300.0) ** (1 / 1.4)
    ratio = res.RAIN_RATE.values[rain] / r_zr
    assert 0.5 < np.nanmedian(ratio) < 2.0
    if dsdmod.HAS_COMPILED_KERNEL:
        ref = dsd(ds, engine="numpy", band="S")
        np.testing.assert_allclose(res.RAIN_RATE, ref.RAIN_RATE, rtol=1e-10)


@pytest.mark.parametrize("engine", ENGINES)
def test_nw_range_rejects_non_rain(engine):
    """Hail-like ZH/ZDR pairs give implausible Nw and are left empty."""
    ds = xr.Dataset({"DBZH": ("g", [55.0, 40.0, 10.0]), "ZDR": ("g", [0.0, 1.5, 3.0])})
    out = dsd(ds, band="S", engine=engine)
    assert np.isnan(out.RAIN_RATE[0]) and np.isfinite(out.RAIN_RATE[1])
    assert np.isnan(out.N0[2])  # 3 dB at 10 dBZ: far too few drops
    assert out.attrs["nw_range"] == [10.0, 1e6]
    keep = dsd(ds, band="S", nw_range=None, engine=engine)
    assert np.isfinite(keep.RAIN_RATE).all()
    assert keep.attrs["nw_range"] == "none"
