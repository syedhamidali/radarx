#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for differential phase processing and KDP
================================================
"""
import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.retrieve import estimate_kdp
from radarx.retrieve import kdp as kdpmod

ENGINES = ["numpy"] + (["compiled"] if kdpmod.HAS_COMPILED_KERNEL else [])
METHODS = ["hubbert", "vulpiani", "monotone"]
DR = 250.0  # gate spacing [m]
NOISE = 3.0  # phase noise [deg]


def _truth(ng):
    """Known KDP profile (two rain cells) and the propagation phase."""
    r = (np.arange(ng) + 0.5) * DR / 1000.0
    kdp = 3.0 * np.exp(-0.5 * ((r - 60) / 6) ** 2) + np.exp(
        -0.5 * ((r - 140) / 15) ** 2
    )
    phi = 2.0 * np.cumsum(kdp) * DR / 1000.0
    return r, kdp, phi


def _sweep(
    nray=60,
    ng=900,
    offset=150.0,
    delta=8.0,
    wrap=True,
    no_signal_from=200.0,
    seed=0,
):
    """
    Synthetic sweep: offset + 2 * integral of a known KDP + a backscatter
    phase bump + Gaussian noise, folded into [-180, 180), and random phase
    with low rhohv beyond ``no_signal_from`` km.
    """
    rnd = np.random.default_rng(seed)
    r, kdp, phi_true = _truth(ng)
    bump = delta * np.exp(-0.5 * ((r - 62) / 1.0) ** 2)
    phi = offset + phi_true + bump + rnd.normal(0.0, NOISE, (nray, ng))
    if wrap:
        phi = (phi + 180.0) % 360.0 - 180.0
    rho = np.full((nray, ng), 0.98)
    dbz = np.broadcast_to(25 + 10 * np.log10(1 + 30 * kdp), (nray, ng)).copy()
    far = r > no_signal_from
    phi[:, far] = rnd.uniform(-180, 180, (nray, far.sum()))
    rho[:, far] = rnd.uniform(0.2, 0.8, (nray, far.sum()))
    dbz[:, far] = rnd.uniform(-20, 0, (nray, far.sum()))
    ds = xr.Dataset(
        {
            "PHIDP": (("azimuth", "range"), phi, {"units": "degrees"}),
            "RHOHV": (("azimuth", "range"), rho),
            "DBZH": (("azimuth", "range"), dbz),
        },
        coords={
            "azimuth": ("azimuth", np.linspace(0.5, 359.5, nray), {"units": "deg"}),
            "range": ("range", (np.arange(ng) + 0.5) * DR, {"units": "m"}),
            "elevation": ("azimuth", np.full(nray, 0.5)),
        },
    )
    return ds, kdp, phi_true, ~far


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("method", METHODS)
def test_recovers_known_kdp(method, engine):
    ds, kdp_true, phi_true, inside = _sweep()
    out = estimate_kdp(ds, method=method, engine=engine)
    k = out.KDP.values[:, inside]
    err = k - kdp_true[inside]
    assert np.isfinite(k).mean() > 0.99
    assert abs(np.mean(err)) < 0.02  # deg/km
    assert np.sqrt(np.mean(err**2)) < 0.25
    # systematic error (mean over rays) away from the backscatter phase bump;
    # its effect is tested in test_hubbert_suppresses_backscatter_phase
    r = ds.range.values[inside] / 1000.0
    away = (r < 56) | (r > 68)
    assert np.max(np.abs(err.mean(axis=0))[away]) < 0.3
    # system offset recovered through the folding at 180 deg
    np.testing.assert_allclose(out.PHIDP_OFFSET.values, 150.0, atol=1.5)
    phi_err = out.PHIDP_processed.values[:, inside] - phi_true[inside]
    assert np.sqrt(np.mean(phi_err**2)) < 4.0
    # non-meteorological gates are masked
    assert np.isnan(out.KDP.values[:, ~inside]).mean() > 0.95


@pytest.mark.parametrize("engine", ENGINES)
def test_hubbert_suppresses_backscatter_phase(engine):
    """The iterative filter reduces the KDP error caused by a delta bump."""
    ds, kdp_true, *_ = _sweep(delta=10.0)
    near = (ds.range.values / 1000 > 56) & (ds.range.values / 1000 < 68)
    hub = estimate_kdp(ds, engine=engine, method="hubbert")
    plain = estimate_kdp(ds, engine=engine, method="hubbert", n_iter=0)
    e_hub = np.abs((hub.KDP.values - kdp_true).mean(axis=0))[near].max()
    e_plain = np.abs((plain.KDP.values - kdp_true).mean(axis=0))[near].max()
    assert e_hub < 0.8
    assert e_hub < 0.6 * e_plain


@pytest.mark.parametrize("engine", ENGINES)
def test_monotone_kdp_non_negative(engine):
    ds, *_ = _sweep()
    out = estimate_kdp(ds, method="monotone", engine=engine)
    k = out.KDP.values
    assert np.nanmin(k) > -1e-9
    assert np.all(np.diff(out.PHIDP_processed.values, axis=1) > -1e-9)


@pytest.mark.skipif(not kdpmod.HAS_COMPILED_KERNEL, reason="kernel not built")
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("offset", ["sweep", "ray", 12.0])
def test_engines_agree(method, offset):
    """
    Compiled kernel and NumPy reference agree to rounding: the same steps in
    the same order; only summation order differs (atol 1e-6 deg, deg/km).
    """
    ds, *_ = _sweep(nray=40, seed=3)
    ds["PHIDP"][:, 300:310] = np.nan  # gap bridged by interpolation
    ds["RHOHV"][5:10, 400:420] = 0.5  # masked block
    ds["RHOHV"][12, :] = 0.1  # ray without valid gates
    ds["DBZH"][:, ::7] = np.nan
    kw = {"method": method, "offset": offset}
    a = estimate_kdp(ds, engine="numpy", **kw)
    b = estimate_kdp(ds, engine="compiled", **kw)
    for name in ("PHIDP_processed", "KDP", "PHIDP_OFFSET"):
        np.testing.assert_array_equal(np.isnan(a[name]), np.isnan(b[name]))
        # the fields are float32: the engines differ by one rounding step
        np.testing.assert_allclose(a[name], b[name], atol=1e-6, rtol=2e-7)
    assert np.isnan(b.KDP[12]).all()


@pytest.mark.skipif(not kdpmod.HAS_COMPILED_KERNEL, reason="kernel not built")
def test_threads_identical():
    ds, *_ = _sweep(nray=50)
    a = estimate_kdp(ds, n_threads=1)
    b = estimate_kdp(ds, n_threads=8)
    xr.testing.assert_identical(a, b)


@pytest.mark.parametrize("engine", ENGINES)
def test_offset_modes(engine):
    ds, *_ = _sweep(nray=20, wrap=False, offset=40.0)
    fixed = estimate_kdp(ds, offset=40.0, engine=engine)
    np.testing.assert_array_equal(fixed.PHIDP_OFFSET, 40.0)
    ray = estimate_kdp(ds, offset="ray", engine=engine)
    np.testing.assert_allclose(ray.PHIDP_OFFSET, 40.0, atol=3 * NOISE)
    assert ray.PHIDP_OFFSET.std() > 0


@pytest.mark.parametrize("engine", ENGINES)
def test_optional_fields_and_output_layout(engine):
    ds, *_ = _sweep(nray=20)
    ds = ds.rename(PHIDP="UPHIDP").transpose("range", "azimuth")
    out = estimate_kdp(ds.drop_vars(["RHOHV", "DBZH"]), engine=engine)
    assert out.KDP.dims == ("range", "azimuth")
    assert out.PHIDP_OFFSET.dims == ("azimuth",)
    xr.testing.assert_identical(out.azimuth, ds.azimuth)
    assert "elevation" in out.coords
    assert out.KDP.attrs["units"] == "degree km-1"
    assert out.KDP.attrs["standard_name"] == "radar_specific_differential_phase_hv"
    assert out.PHIDP_processed.attrs["units"] == "degree"
    assert out.PHIDP_processed.attrs["standard_name"] == "radar_differential_phase_hv"
    assert np.isfinite(out.KDP).mean() > 0.5


def _volume():
    sweeps = {}
    for i, (nray, ng) in enumerate([(40, 900), (30, 700), (20, 500)]):
        ds, *_ = _sweep(nray=nray, ng=ng, seed=i)
        sweeps[f"sweep_{i}"] = ds
    sweeps["sweep_3"] = xr.Dataset(
        {"DBZH": (("azimuth", "range"), np.zeros((4, 5)))},
        coords={"range": np.arange(5.0)},
    )
    root = xr.Dataset(coords={"latitude": 10.0, "longitude": 20.0, "altitude": 5.0})
    return xr.DataTree.from_dict({"/": root, **sweeps})


@pytest.mark.parametrize("engine", ENGINES)
def test_datatree_batches_all_sweeps(engine):
    dtree = _volume()
    out = estimate_kdp(dtree, engine=engine)
    assert list(out.children) == ["sweep_0", "sweep_1", "sweep_2"]
    assert float(out["latitude"]) == 10.0
    for name in out.children:
        single = estimate_kdp(dtree[name].to_dataset(inherit=False), engine=engine)
        xr.testing.assert_identical(out[name].to_dataset(inherit=False), single)


def test_accessors():
    dtree = _volume()
    ds = dtree["sweep_0"].to_dataset(inherit=False)
    xr.testing.assert_identical(ds.radarx.kdp(), estimate_kdp(ds))
    out = dtree.radarx.kdp(method="vulpiani")
    xr.testing.assert_identical(
        out["sweep_1"].to_dataset(inherit=False),
        estimate_kdp(dtree["sweep_1"].to_dataset(inherit=False), method="vulpiani"),
    )
    merged = dtree.radarx.assign(out)
    for name in out.children:
        assert {"PHIDP", "PHIDP_processed", "KDP"} <= set(merged[name].data_vars)
    assert "KDP" not in dtree["sweep_0"].data_vars


def test_errors():
    ds, *_ = _sweep(nray=5, ng=50)
    with pytest.raises(ValueError, match="method"):
        estimate_kdp(ds, method="nope")
    with pytest.raises(ValueError, match="engine"):
        estimate_kdp(ds, engine="nope")
    with pytest.raises(ValueError, match="offset"):
        estimate_kdp(ds, offset="nope")
    with pytest.raises(KeyError):
        estimate_kdp(ds, phidp="nope")
    with pytest.raises(KeyError):
        estimate_kdp(ds.drop_vars("PHIDP"))
    rng = ds.range.values.copy()
    rng[10:] += 100.0
    with pytest.raises(ValueError, match="uniformly"):
        estimate_kdp(ds.assign_coords(range=rng))
    with pytest.raises(KeyError):
        estimate_kdp(xr.DataTree.from_dict({"sweep_0": ds.drop_vars("PHIDP")}))
    for bad in ("up", 0, 2):
        with pytest.raises(ValueError, match="phidp_sign"):
            estimate_kdp(ds, phidp_sign=bad)


def test_datatree_named_field_skips_sweeps_without_it():
    ds, *_ = _sweep(nray=5, ng=50)
    dtree = xr.DataTree.from_dict(
        {"sweep_0": ds, "sweep_1": ds.drop_vars("PHIDP"), "sweep_2": ds}
    )
    out = estimate_kdp(dtree, phidp="PHIDP")
    assert sorted(out.children) == ["sweep_0", "sweep_2"]
    xr.testing.assert_allclose(out["sweep_0"].to_dataset(), estimate_kdp(ds))
    with pytest.raises(KeyError, match="no sweep"):
        estimate_kdp(dtree, phidp="UPHIDP")


def test_input_checks(monkeypatch):
    ds, *_ = _sweep(nray=5, ng=50)
    with pytest.raises(KeyError, match="RHOHV_X"):
        estimate_kdp(ds, rhohv="RHOHV_X")
    with pytest.raises(ValueError, match="two range gates"):
        estimate_kdp(ds.isel(range=slice(0, 1)))
    with pytest.raises(ValueError, match="dimensions"):
        estimate_kdp(ds.assign(PHIDP=ds.PHIDP.isel(azimuth=0)))
    monkeypatch.setattr(kdpmod, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError, match="compiled KDP kernel"):
        estimate_kdp(ds, engine="compiled")


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("method", METHODS)
def test_reversed_sign_convention(method, engine):
    """A phase that decreases with range (some systems) is detected and flipped."""
    ds, kdp_true, _, inside = _sweep(offset=-40.0)
    normal = estimate_kdp(ds, method=method, engine=engine)
    rev = ds.assign(PHIDP=-ds.PHIDP)
    out = estimate_kdp(rev, method=method, engine=engine)
    assert normal.PHIDP_processed.attrs["phidp_sign"] == 1
    assert out.PHIDP_processed.attrs["phidp_sign"] == -1
    np.testing.assert_allclose(out.PHIDP_OFFSET, -normal.PHIDP_OFFSET, atol=1e-9)
    np.testing.assert_allclose(out.KDP, normal.KDP, atol=1e-6)
    np.testing.assert_allclose(out.PHIDP_processed, normal.PHIDP_processed, atol=1e-6)
    err = out.KDP.values[:, inside] - kdp_true[inside]
    assert np.sqrt(np.mean(err**2)) < 0.25
    # forcing the convention
    forced = estimate_kdp(rev, method=method, engine=engine, phidp_sign=1)
    assert forced.PHIDP_processed.attrs["phidp_sign"] == 1
    # the wrong convention gives negative KDP (zero for the monotone fit)
    assert np.nanmean(forced.KDP.values[:, inside]) < (
        1e-6 if method == "monotone" else -0.1
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_sign_without_rain_stays_positive(engine):
    """Without significant evidence (no rain gates) the phase is not flipped."""
    ds, *_ = _sweep(nray=30)
    ds["DBZH"][:] = 20.0  # no gate qualifies as rain for the sign test
    rev = ds.assign(PHIDP=-ds.PHIDP)
    assert estimate_kdp(rev, engine=engine).PHIDP_processed.attrs["phidp_sign"] == 1
    # without reflectivity all valid gates take part
    out = estimate_kdp(rev.drop_vars("DBZH"), engine=engine)
    assert out.PHIDP_processed.attrs["phidp_sign"] == -1


@pytest.mark.parametrize("engine", ENGINES)
def test_kdp_needs_enough_valid_gates_in_window(engine):
    ds, *_ = _sweep(nray=10)
    # a run of 9 valid gates (2.25 km) between two gaps: it passes the
    # texture test, but fills less than half of the 6 km KDP window
    ds["RHOHV"][:, 200:260] = 0.5
    ds["RHOHV"][:, 269:330] = 0.5
    ds["DBZH"][:] = 20.0  # long window everywhere
    out = estimate_kdp(ds, engine=engine)
    assert np.isnan(out.KDP[:, 255:275]).all()
    assert np.isfinite(out.KDP[:, 150:190]).all()
    loose = estimate_kdp(ds, engine=engine, min_valid_fraction=0.0)
    assert np.isfinite(loose.KDP[:, 260:269]).all()
    assert out.KDP.attrs["units"] == "degree km-1"


@pytest.fixture(scope="module")
def klbb():
    xd = pytest.importorskip("xradar")
    from open_radar_data import DATASETS

    file = DATASETS.fetch("KLBB20160601_150025_V06")
    dtree = xd.io.open_nexradlevel2_datatree(file, sweep=[0, 1, 2])
    return dtree


def test_real_nexrad_volume(klbb):
    """S-band NEXRAD volume: plausible offsets and KDP in rain."""
    out = klbb.radarx.kdp()
    # split cuts without differential phase are skipped
    assert "sweep_0" in out.children
    ds = klbb["sweep_0"].to_dataset()
    res = out["sweep_0"].to_dataset()
    assert res.KDP.dims == ds.PHIDP.dims
    assert res.PHIDP_processed.attrs["phidp_sign"] == 1
    assert res.PHIDP_processed.attrs["source_fields"] == "PHIDP, RHOHV, DBZH"
    offset = float(res.PHIDP_OFFSET[0])
    assert 0.0 < offset < 180.0
    k = res.KDP.values
    z = ds.DBZH.values
    rain = (z > 45) & np.isfinite(k)
    weak = (z > 10) & (z < 25) & np.isfinite(k)
    assert rain.sum() > 500
    # KDP grows with rain intensity and is near zero in weak echo
    assert 0.3 < np.mean(k[rain]) < 3.0
    assert abs(np.median(k[weak])) < 0.2
    # most non-meteorological gates are rejected
    assert np.isfinite(k[ds.RHOHV.values < 0.7]).mean() < 0.05
    if kdpmod.HAS_COMPILED_KERNEL:
        ref = estimate_kdp(ds, engine="numpy")
        np.testing.assert_allclose(res.KDP, ref.KDP, atol=1e-6)


def test_against_pyart_vulpiani(klbb):
    """Validation baseline (only when Py-ART is installed)."""
    pyart = pytest.importorskip("pyart")
    from open_radar_data import DATASETS

    radar = pyart.io.read_nexrad_archive(
        DATASETS.fetch("KLBB20160601_150025_V06")
    ).extract_sweeps([0])
    gf = pyart.filters.GateFilter(radar)
    gf.exclude_below("cross_correlation_ratio", 0.85)
    ref, _ = pyart.retrieve.kdp_vulpiani(
        radar, gatefilter=gf, psidp_field="differential_phase", band="S"
    )
    ref = np.ma.filled(ref["data"].astype(float), np.nan)
    ds = klbb["sweep_0"].to_dataset()
    # match rays by azimuth
    az = radar.azimuth["data"]
    idx = [np.argmin(np.abs((az - a + 180) % 360 - 180)) for a in ds.azimuth.values]
    ref = ref[idx]
    out = estimate_kdp(ds, method="vulpiani")
    k = out.KDP.values
    sel = (ds.DBZH.values > 30) & (ds.RHOHV.values > 0.95)
    sel &= np.isfinite(k) & np.isfinite(ref)
    assert np.corrcoef(k[sel], ref[sel])[0, 1] > 0.45
    assert abs(np.mean(k[sel] - ref[sel])) < 0.2


@pytest.fixture(scope="module")
def csapr2():
    xd = pytest.importorskip("xradar")
    from open_radar_data import DATASETS

    file = DATASETS.fetch("corcsapr2cmacppiM1.c1.20181111.030003.nc")
    dtree = xd.io.open_cfradial1_datatree(file)
    return dtree["sweep_0"].to_dataset(inherit="all_coords").load()


def test_real_csapr2_default_call(csapr2):
    """
    C-band convection whose system offset (about 164 deg) folds the phase.
    The file holds the raw phase twice: ``uncorrected_differential_phase``
    increases with range and ``differential_phase`` (360 deg minus it)
    decreases. The default call must give positive KDP in the cores either
    way, consistent with the radar's own KDP.
    """
    ds = csapr2
    z = ds.reflectivity.values
    sel = (z > 30) & (ds.copol_correlation_coeff.values > 0.95)
    radar_kdp = ds.specific_differential_phase.values
    results = {}
    for phidp, sign in ((None, 1), ("differential_phase", -1)):
        out = estimate_kdp(ds, phidp)
        assert out.PHIDP_processed.attrs["phidp_sign"] == sign
        k = out.KDP.values
        assert np.nanmean(k[z >= 45]) > 1.5
        m = sel & np.isfinite(k) & np.isfinite(radar_kdp)
        assert m.sum() > 10_000
        assert np.corrcoef(k[m], radar_kdp[m])[0, 1] > 0.8
        assert abs(np.mean(k[m] - radar_kdp[m])) < 0.2
        # processed phase increases through the storms
        assert np.nanpercentile(out.PHIDP_processed.values, 99) > 150
        assert np.nanpercentile(out.PHIDP_processed.values, 1) > -30
        results[phidp] = k
    assert out.PHIDP_processed.attrs["source_fields"].startswith("differential_phase")
    a, b = results.values()
    m = np.isfinite(a) & np.isfinite(b)
    assert np.corrcoef(a[m], b[m])[0, 1] > 0.95


# --------------------------------------------------------------------------
# phase texture of rain with high KDP, and band-dependent defaults
# --------------------------------------------------------------------------


def _ramp_sweep(kdp, nray=24, ng=800, junk_from=600, noise=3.0, seed=0):
    """Uniform rain with a constant KDP, then random phase (junk)."""
    rnd = np.random.default_rng(seed)
    r = (np.arange(ng) + 0.5) * DR / 1000.0
    phi = 150.0 + 2.0 * kdp * r + rnd.normal(0.0, noise, (nray, ng))
    phi = (phi + 180.0) % 360.0 - 180.0
    phi[:, junk_from:] = rnd.uniform(-180, 180, (nray, ng - junk_from))
    return xr.Dataset(
        {
            "PHIDP": (("azimuth", "range"), phi),
            "RHOHV": (("azimuth", "range"), np.full((nray, ng), 0.98)),
            "DBZH": (("azimuth", "range"), np.full((nray, ng), 35.0)),
        },
        coords={
            "azimuth": np.linspace(0.5, 359.5, nray),
            "range": (np.arange(ng) + 0.5) * DR,
            "elevation": ("azimuth", np.full(nray, 0.5)),
        },
    )


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("kdp", [5.0, 15.0, 20.0, 30.0])
def test_detrended_texture_keeps_heavy_rain(kdp, engine):
    ds = _ramp_sweep(kdp)
    out = estimate_kdp(ds, texture_method="detrended", phidp_sign=1, engine=engine)
    k = out.KDP.values
    assert np.isfinite(k[:, 20:580]).mean() > 0.99
    assert np.nanmedian(k[:, 30:570]) == pytest.approx(kdp, rel=0.01)
    # random phase with a high rhohv is still rejected
    assert np.isfinite(k[:, 620:]).mean() < 0.01


@pytest.mark.parametrize("engine", ENGINES)
def test_circular_texture_masks_heavy_rain(engine):
    """The default rule includes the rain ramp (see texture_method)."""
    ds = _ramp_sweep(20.0)
    out = estimate_kdp(ds, phidp_sign=1, engine=engine)
    assert np.isfinite(out.KDP.values[:, 20:580]).mean() < 0.05
    out = estimate_kdp(ds, phidp_sign=1, texture_max=60.0, engine=engine)
    assert np.isfinite(out.KDP.values[:, 20:580]).mean() > 0.99


@pytest.mark.skipif(not kdpmod.HAS_COMPILED_KERNEL, reason="compiled kernel not built")
@pytest.mark.parametrize("method", METHODS)
def test_texture_engines_agree(method):
    ds, *_ = _sweep(seed=3)
    a = estimate_kdp(ds, method=method, texture_method="detrended", engine="numpy")
    b = estimate_kdp(ds, method=method, texture_method="detrended", engine="compiled")
    np.testing.assert_array_equal(np.isfinite(a.KDP.values), np.isfinite(b.KDP.values))
    np.testing.assert_allclose(a.KDP.values, b.KDP.values, atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(
        a.PHIDP_processed.values, b.PHIDP_processed.values, atol=1e-6, equal_nan=True
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_band_scales_the_defaults(engine):
    """Vulpiani zeroes KDP above 20 deg/km at S band, not at X band."""
    ds = _ramp_sweep(30.0)
    kw = dict(
        method="vulpiani",
        texture_method="detrended",
        phidp_sign=1,
        engine=engine,
    )
    s_band = estimate_kdp(ds, **kw)
    x_band = estimate_kdp(ds, band="X", **kw)
    assert np.nanmedian(s_band.KDP.values[:, 100:500]) < 10.0
    assert np.nanmedian(x_band.KDP.values[:, 100:500]) == pytest.approx(30.0, rel=0.05)
    # values you pass are used as given
    given = estimate_kdp(ds, band="X", kdp_bounds=(-2.0, 20.0), **kw)
    assert np.nanmedian(given.KDP.values[:, 100:500]) < 10.0
    with pytest.raises(ValueError, match="band"):
        estimate_kdp(ds, band="K")
    with pytest.raises(ValueError, match="texture_method"):
        estimate_kdp(ds, texture_method="x")


# --------------------------------------------------------------------------
# adaptive filter window, backscatter phase and uncertainty
# --------------------------------------------------------------------------


def _cell(kmax, width_km, delta, sigma, nray=60, seed=7, dr=DR):
    """A Gaussian rain cell of known KDP with a narrower backscatter bump."""
    rnd = np.random.default_rng(seed)
    r = (np.arange(int(60e3 / dr)) + 0.5) * dr / 1000.0
    kdp = kmax * np.exp(-0.5 * ((r - 30.0) / width_km) ** 2)
    phi = 2.0 * np.cumsum(kdp) * dr / 1000.0
    bump = delta * np.exp(-0.5 * ((r - 30.0) / (0.6 * width_km)) ** 2)
    bump *= np.sqrt(kdp / kmax)
    psi = 30.0 + phi + bump + rnd.normal(0.0, sigma, (nray, r.size))
    dbz = np.broadcast_to(20.0 + 30.0 * kdp / kmax, psi.shape).copy()
    ds = xr.Dataset(
        {
            "PHIDP": (("azimuth", "range"), psi),
            "RHOHV": (("azimuth", "range"), np.full(psi.shape, 0.98)),
            "DBZH": (("azimuth", "range"), dbz),
        },
        coords={
            "azimuth": np.arange(nray, dtype=float),
            "range": r * 1000.0,
            "elevation": ("azimuth", np.full(nray, 0.5)),
        },
    )
    return ds, kdp, bump


_X_CELL = (12.0, 4.0, 12.0, 3.0)  # Kmax, width (km), delta, noise (degrees)
_OLD = {"filter_window": 2.0, "n_iter": 10}  # the fixed defaults before the change


def _window(out):
    return out.PHIDP_processed.attrs["filter_window"]


@pytest.mark.parametrize("engine", ENGINES)
def test_narrow_x_band_cell_keeps_its_peak(engine):
    """
    The fixed 2 km window and 10 iterations took 17 percent off the peak of
    a narrow X-band cell; the default adapts to the largest KDP of the sweep.
    """
    ds, kdp, _ = _cell(*_X_CELL)
    peak = np.argmax(kdp)
    old = estimate_kdp(ds, engine=engine, **_OLD)
    new = estimate_kdp(ds, engine=engine)
    assert np.nanmean(old.KDP.values[:, peak]) < 0.88 * kdp.max()
    assert np.nanmean(new.KDP.values[:, peak]) == pytest.approx(kdp.max(), rel=0.05)

    def rmse(o):
        return np.sqrt(np.nanmean((o.KDP.values - kdp[None]) ** 2))

    assert rmse(new) < 0.8 * rmse(old)
    assert _window(new) < 2.0
    assert _window(old) == 2.0


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "cell", [(1.5, 8.0, 0.0, 3.0), (5.0, 6.0, 6.0, 3.0)], ids=["S", "C"]
)
def test_adaptive_default_equals_the_old_one_at_low_kdp(cell, engine):
    """S and C-band cells keep the 2 km window, so the numbers do not change."""
    ds, *_ = _cell(*cell)
    new = estimate_kdp(ds, engine=engine)
    old = estimate_kdp(ds, engine=engine, **_OLD)
    assert _window(new) == 2.0
    xr.testing.assert_allclose(new.KDP, old.KDP, atol=0, rtol=0)
    xr.testing.assert_allclose(new.PHIDP_processed, old.PHIDP_processed, atol=0, rtol=0)


@pytest.mark.parametrize("engine", ENGINES)
def test_window_follows_noise_and_gate_spacing(engine):
    """A noisier phase needs more gates, hence a wider window; the cap is 2 km."""
    quiet = estimate_kdp(_cell(12.0, 4.0, 12.0, 1.0)[0], engine=engine)
    noisy = estimate_kdp(_cell(12.0, 4.0, 12.0, 5.0)[0], engine=engine)
    assert _window(quiet) < _window(noisy) <= 2.0
    # the same noise on finer gates averages over more gates in the same km
    fine = estimate_kdp(_cell(12.0, 4.0, 12.0, 3.0, dr=125.0)[0], engine=engine)
    coarse = estimate_kdp(_cell(12.0, 4.0, 12.0, 3.0, dr=500.0)[0], engine=engine)
    assert _window(fine) < _window(coarse) <= 2.0
    # an explicit value is used as given
    cell = _cell(*_X_CELL)[0]
    assert _window(estimate_kdp(cell, engine=engine, filter_window=1.0)) == 1.0
    with pytest.raises(ValueError, match="filter_window"):
        estimate_kdp(cell, filter_window=-1.0)
    with pytest.raises(ValueError, match="noise_fraction"):
        estimate_kdp(cell, noise_fraction=0.0)
    with pytest.raises(ValueError, match="n_iter"):
        estimate_kdp(cell, n_iter=-1)


@pytest.mark.parametrize("engine", ENGINES)
def test_monotone_and_vulpiani_keep_the_fixed_window(engine):
    ds, *_ = _cell(*_X_CELL)
    for method in ("monotone", "vulpiani"):
        assert _window(estimate_kdp(ds, method=method, engine=engine)) == 2.0


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("sigma", [1.0, 3.0, 5.0])
def test_phase_noise_estimate(sigma, engine):
    """The noise comes from second differences: folding and the ramp do not matter."""
    ds, *_ = _cell(8.0, 5.0, 8.0, sigma)
    ds["PHIDP"] = (ds.PHIDP + 180.0) % 360.0 - 180.0
    out = estimate_kdp(ds, engine=engine)
    assert out.PHIDP_processed.attrs["phase_noise"] == pytest.approx(sigma, rel=0.05)


@pytest.mark.skipif(not kdpmod.HAS_COMPILED_KERNEL, reason="kernel not built")
@pytest.mark.parametrize("method", METHODS)
def test_new_outputs_agree_between_engines(method):
    ds, *_ = _sweep(nray=40, seed=3)
    ds["PHIDP"][:, 300:310] = np.nan
    ds["RHOHV"][5:10, 400:420] = 0.5
    ds["RHOHV"][12, :] = 0.1  # ray without valid gates
    ds["DBZH"][:, ::7] = np.nan
    a = estimate_kdp(ds, engine="numpy", method=method)
    b = estimate_kdp(ds, engine="compiled", method=method)
    for name in ("PHIDP_BACKSCATTER", "KDP_UNCERTAINTY"):
        if name not in a:
            assert name not in b
            continue
        np.testing.assert_array_equal(np.isnan(a[name]), np.isnan(b[name]))
        np.testing.assert_allclose(a[name], b[name], atol=1e-6, rtol=0)
    for key in ("phase_noise", "filter_window", "kdp_reference"):
        assert a.PHIDP_processed.attrs[key] == pytest.approx(
            b.PHIDP_processed.attrs[key], rel=1e-9, nan_ok=True
        )
    assert np.isnan(b.KDP_UNCERTAINTY[12]).all()


@pytest.mark.skipif(not kdpmod.HAS_COMPILED_KERNEL, reason="kernel not built")
def test_adaptive_window_agrees_between_engines_on_a_volume():
    """Every sweep of a volume adapts on its own noise and largest KDP."""
    sweeps = {
        "sweep_0": _cell(*_X_CELL, nray=30)[0],
        "sweep_1": _cell(1.5, 8.0, 0.0, 3.0, nray=20, seed=2)[0],
        "sweep_2": _cell(12.0, 4.0, 12.0, 5.0, nray=25, seed=3, dr=125.0)[0],
    }
    tree = xr.DataTree.from_dict(sweeps)
    a = estimate_kdp(tree, engine="numpy")
    b = estimate_kdp(tree, engine="compiled")
    windows = []
    for name in sweeps:
        x, y = a[name].to_dataset(), b[name].to_dataset()
        np.testing.assert_allclose(x.KDP, y.KDP, atol=1e-6)
        np.testing.assert_allclose(x.KDP_UNCERTAINTY, y.KDP_UNCERTAINTY, atol=1e-9)
        for key in ("phase_noise", "filter_window", "kdp_reference"):
            assert x.PHIDP_processed.attrs[key] == pytest.approx(
                y.PHIDP_processed.attrs[key], rel=1e-9
            )
        windows.append(_window(x))
    assert windows[0] < 2.0 and windows[1] == 2.0 and windows[2] < 2.0


@pytest.mark.parametrize("engine", ENGINES)
def test_backscatter_phase_of_the_hubbert_filter(engine):
    """
    PHIDP_BACKSCATTER is the measured minus the processed phase, the part the
    filter removed: it finds a narrow bump (about two thirds of it here), is
    zero elsewhere, and exists for the Hubbert filter only.
    """
    ds, _, _, inside = _sweep(delta=10.0, nray=200)
    r = ds.range.values / 1000.0
    out = estimate_kdp(ds, engine=engine, offset=150.0)
    delta = out.PHIDP_BACKSCATTER
    assert delta.dims == out.KDP.dims
    assert delta.attrs["units"] == "degree"
    measured = (ds.PHIDP.values - 150.0 + 180.0) % 360.0 - 180.0
    resid = measured - out.PHIDP_processed.values
    resid -= 360.0 * np.round(resid / 360.0)
    ok = np.isfinite(delta.values)
    np.testing.assert_allclose(delta.values[ok], resid[ok], atol=1e-4, rtol=1e-6)
    bump = 10.0 * np.exp(-0.5 * ((r - 62.0) / 1.0) ** 2)
    centre = (r > 61.5) & (r < 62.5)
    est = np.nanmean(delta.values[:, centre])
    assert 0.5 * bump[centre].mean() < est < bump[centre].mean()
    assert abs(np.nanmean(delta.values[:, (r > 20) & (r < 50)])) < 0.3
    assert np.isnan(delta.values[:, ~inside]).all()
    for method in ("vulpiani", "monotone"):
        assert "PHIDP_BACKSCATTER" not in estimate_kdp(ds, method=method, engine=engine)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("method", METHODS)
def test_kdp_uncertainty_matches_the_spread_over_rays(method, engine):
    """
    All rays share one truth, so the spread of KDP over the rays is the noise
    of the estimate. The uncertainty is the white noise of the phase passed
    through the filter and the slope; the monotone fit is not linear and
    reduces the noise more than predicted, so its uncertainty is an upper bound.
    """
    ds, *_ = _cell(5.0, 6.0, 6.0, 3.0, nray=300)
    out = estimate_kdp(ds, method=method, engine=engine)
    spread = np.nanstd(out.KDP.values, axis=0)[30:210]
    mean_unc = np.nanmean(out.KDP_UNCERTAINTY.values, axis=0)[30:210]
    ratio = np.median(mean_unc / spread)
    if method == "monotone":
        assert 1.0 < ratio < 1.6
    else:
        assert ratio == pytest.approx(1.0, abs=0.2)
    assert out.KDP_UNCERTAINTY.attrs["units"] == "degree km-1"
    unc = out.KDP_UNCERTAINTY.values
    np.testing.assert_array_equal(np.isnan(unc), np.isnan(out.KDP.values))
    assert (unc[np.isfinite(unc)] > 0).all()


@pytest.mark.parametrize("engine", ENGINES)
def test_uncertainty_grows_with_noise_and_shrinks_with_the_window(engine):
    ds, *_ = _cell(5.0, 6.0, 0.0, 2.0, nray=20)
    noisy, *_ = _cell(5.0, 6.0, 0.0, 4.0, nray=20)

    def unc(d, **kw):
        return np.nanmedian(estimate_kdp(d, engine=engine, **kw).KDP_UNCERTAINTY.values)

    assert unc(noisy, filter_window=2.0) == pytest.approx(
        2.0 * unc(ds, filter_window=2.0), rel=0.1
    )
    assert unc(ds, filter_window=1.0) > unc(ds, filter_window=2.0)
    assert unc(ds, kdp_window=(1.0, 3.0)) > unc(ds, kdp_window=(2.0, 6.0))


@pytest.mark.parametrize("engine", ENGINES)
def test_band_factors_are_parameters(engine):
    """The factors 1, 2 and 3 are radarx choices and can be set."""
    ds = _ramp_sweep(30.0)
    kw = dict(
        method="vulpiani", texture_method="detrended", phidp_sign=1, engine=engine
    )
    assert kdpmod.BAND_FACTORS == {"S": 1.0, "C": 2.0, "X": 3.0}
    c = estimate_kdp(ds, band="C", **kw)
    x2 = estimate_kdp(ds, band="X", band_factors={"X": 2.0}, **kw)
    xr.testing.assert_identical(x2.KDP, c.KDP)
    x3 = estimate_kdp(ds, band="X", **kw)
    assert np.nanmedian(x3.KDP.values[:, 100:500]) == pytest.approx(30.0, rel=0.05)
    # a factor of 1 for X gives the S-band bounds (30 deg/km is zeroed)
    x1 = estimate_kdp(ds, band="X", band_factors={"X": 1.0}, **kw)
    assert np.nanmedian(x1.KDP.values[:, 100:500]) < 10.0
    # a band that is not in the table can be added
    ku = estimate_kdp(ds, band="ku", band_factors={"Ku": 4.0}, **kw)
    assert np.nanmedian(ku.KDP.values[:, 100:500]) == pytest.approx(30.0, rel=0.05)
    with pytest.raises(ValueError, match="band"):
        estimate_kdp(ds, band="Ku")
    with pytest.raises(ValueError, match="positive"):
        estimate_kdp(ds, band="X", band_factors={"X": 0.0})
    assert kdpmod.BAND_FACTORS["X"] == 3.0  # the table is not modified
