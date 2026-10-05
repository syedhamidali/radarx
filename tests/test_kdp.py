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
        np.testing.assert_allclose(a[name], b[name], atol=1e-6, rtol=0)
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
    assert out.KDP.attrs["units"] == "degrees/km"
    assert out.KDP.attrs["standard_name"] == "radar_specific_differential_phase_hv"
    assert out.PHIDP_processed.attrs["units"] == "degrees"
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


def test_input_checks(monkeypatch):
    ds, *_ = _sweep(nray=5, ng=50)
    with pytest.raises(KeyError, match="RHOHV_X"):
        estimate_kdp(ds, rhohv="RHOHV_X")
    with pytest.raises(ValueError, match="two range gates"):
        estimate_kdp(ds.isel(range=slice(0, 1)))
    with pytest.raises(ValueError, match="2-D"):
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
    assert out.KDP.attrs["units"] == "degrees/km"


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
