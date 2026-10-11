#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for non-meteorological echo filtering
===========================================
"""
import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.retrieve import apply_mask, echo_mask
from radarx.retrieve import qc as qcmod

from .provenance_helpers import without_provenance

ENGINES = ["numpy"] + (["compiled"] if qcmod.HAS_COMPILED_KERNEL else [])
DR = 250.0  # gate spacing [m]
NRAY, NG = 360, 400


def _sweep(seed=0, nray=NRAY, ng=NG, az0=0.5):
    """
    Synthetic sweep with known echo types along range:

    - gates 20-159: rain (smooth Z, ZDR ~1 dB, rhohv 0.99, smooth PHIDP),
    - gates 180-259: biological echo (ZDR 4-12 dB, rhohv 0.4-0.9, random
      PHIDP, Z 0-10 dBZ),
    - gates 280-339: ground clutter (Z jumping by 20 dB from gate to gate,
      rhohv 0.5-0.9, noisy ZDR and PHIDP),
    - everything else: no data (NEXRAD code -33 dBZ), except isolated
      single rain-like gates at gate 370 of every 10th ray (speckle).
    """
    rnd = np.random.default_rng(seed)
    g = np.arange(ng)
    z = np.full((nray, ng), -33.0)
    zdr = np.full((nray, ng), -13.0625)
    rho = np.full((nray, ng), 0.2017)
    phi = np.full((nray, ng), -0.705)
    truth = np.zeros((nray, ng), dtype=np.int8)

    rain = (g >= 20) & (g < 160)
    z[:, rain] = 35 + 10 * np.sin(g[rain] / 15.0) + rnd.normal(0, 1, (nray, rain.sum()))
    zdr[:, rain] = 1.0 + rnd.normal(0, 0.3, (nray, rain.sum()))
    rho[:, rain] = np.clip(0.99 + rnd.normal(0, 0.005, (nray, rain.sum())), 0, 1)
    phi[:, rain] = 40 + 0.2 * g[rain] + rnd.normal(0, 2, (nray, rain.sum()))
    truth[:, rain] = 1

    bio = (g >= 180) & (g < 260)
    n = bio.sum()
    z[:, bio] = rnd.uniform(0, 10, (nray, n))
    zdr[:, bio] = rnd.uniform(4, 12, (nray, n))
    rho[:, bio] = rnd.uniform(0.4, 0.9, (nray, n))
    phi[:, bio] = rnd.uniform(0, 360, (nray, n))
    truth[:, bio] = 2

    clutter = (g >= 280) & (g < 340)
    n = clutter.sum()
    z[:, clutter] = 30 + 20 * (g[clutter] % 2) + rnd.normal(0, 3, (nray, n))
    zdr[:, clutter] = rnd.normal(0, 3, (nray, n))
    rho[:, clutter] = rnd.uniform(0.5, 0.9, (nray, n))
    phi[:, clutter] = rnd.uniform(0, 360, (nray, n))
    truth[:, clutter] = 2

    z[::10, 370] = 30.0
    zdr[::10, 370] = 1.0
    rho[::10, 370] = 0.99
    phi[::10, 370] = 60.0
    truth[::10, 370] = 3

    ds = xr.Dataset(
        {
            "DBZH": (("azimuth", "range"), z, {"units": "dBZ"}),
            "ZDR": (("azimuth", "range"), zdr, {"units": "dB"}),
            "RHOHV": (("azimuth", "range"), rho),
            "PHIDP": (("azimuth", "range"), phi, {"units": "degrees"}),
        },
        coords={
            "azimuth": ("azimuth", np.mod(az0 + np.arange(nray) * 360 / nray, 360)),
            "range": ("range", (g + 0.5) * DR, {"units": "m"}),
            "elevation": ("azimuth", np.full(nray, 0.5)),
        },
        attrs={"waveform_type": "contiguous_surveillance"},  # as xradar's NEXRAD
    )
    return ds, truth


def _volume():
    """Polarimetric cut, Doppler cut at the same elevation, higher cut, no Z."""
    surv, _ = _sweep(0)
    surv = surv.assign(sweep_fixed_angle=0.5)
    dop = surv[["DBZH"]].assign(
        VRADH=(("azimuth", "range"), np.full((NRAY, NG), 5.0)),
        sweep_fixed_angle=0.5,
    )
    dop["VRADH"][:, :10] = -64.5  # NEXRAD no-data code
    high, _ = _sweep(1)
    high = high.assign(sweep_fixed_angle=1.5)
    other = xr.Dataset({"X": ("t", np.arange(3.0))})
    root = xr.Dataset(attrs={"scan_name": "VCP-212"})
    return xr.DataTree.from_dict(
        {"/": root, "sweep_0": surv, "sweep_1": dop, "sweep_2": high, "sweep_3": other}
    )


def _share(cls, truth, kind, want):
    sel = truth == kind
    return np.mean(np.isin(cls[sel], want))


@pytest.mark.parametrize("engine", ENGINES)
def test_synthetic_echo_types(engine):
    ds, truth = _sweep()
    out = echo_mask(ds, engine=engine)
    cls = out.ECHO_CLASS.values
    assert _share(cls, truth, 1, [1]) > 0.99
    assert _share(cls, truth, 2, [2, 3]) > 0.98
    assert _share(cls, truth, 3, [3]) == 1.0
    assert _share(cls, truth, 0, [0]) == 1.0  # NEXRAD no-data codes
    np.testing.assert_array_equal(out.METEO_MASK.values, cls == 1)
    assert np.isnan(out.METEO_SCORE.values[truth == 0]).all()


@pytest.mark.parametrize("engine", ENGINES)
def test_each_feature_alone_separates_bio_and_rain(engine):
    """Every polarimetric feature alone tells biological echo from rain."""
    ds, truth = _sweep()
    for name in ("rhohv", "zdr", "zdr_texture", "phidp_texture"):
        weights = {k: float(k == name) for k in qcmod.FEATURES}
        cls = echo_mask(ds, weights=weights, engine=engine).ECHO_CLASS.values
        rain = truth == 1
        bio = truth == 2
        bio[:, 280:] = False
        assert np.mean(cls[rain] == 1) > 0.97, name
        assert np.mean(cls[bio] != 1) > 0.9, name


@pytest.mark.parametrize("engine", ENGINES)
def test_reflectivity_only_detects_clutter(engine):
    ds, truth = _sweep()
    out = echo_mask(ds[["DBZH"]], engine=engine)
    cls = out.ECHO_CLASS.values
    assert _share(cls, truth, 1, [1]) > 0.99
    clutter = np.zeros_like(truth, dtype=bool)
    clutter[:, 285:335] = True
    assert np.mean(cls[clutter] != 1) > 0.95
    assert out.ECHO_CLASS.attrs["source_fields"] == "DBZH"


@pytest.mark.skipif(not qcmod.HAS_COMPILED_KERNEL, reason="kernel not built")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_engines_identical(seed):
    ds, _ = _sweep(seed)
    # a sector scan (gap in azimuth) and values above 1 for rhohv
    ds = ds.isel(azimuth=slice(0, 300))
    ds["RHOHV"] = ds.RHOHV + 0.02 * (seed == 1)
    for kwargs in ({}, {"min_size": 0, "window": 3.0}, {"nodata": None}):
        a = echo_mask(ds, engine="numpy", **kwargs)
        b = echo_mask(ds, engine="compiled", **kwargs)
        np.testing.assert_array_equal(a.ECHO_CLASS.values, b.ECHO_CLASS.values)
        np.testing.assert_array_equal(a.METEO_SCORE.values, b.METEO_SCORE.values)


@pytest.mark.skipif(not qcmod.HAS_COMPILED_KERNEL, reason="kernel not built")
def test_thread_count_does_not_change_result():
    ds, _ = _sweep()
    ref = echo_mask(ds, n_threads=1).ECHO_CLASS.values
    for n in (2, 3, 8):
        np.testing.assert_array_equal(echo_mask(ds, n_threads=n).ECHO_CLASS.values, ref)


@pytest.mark.parametrize("engine", ENGINES)
def test_speckle_size_and_wrap_across_north(engine):
    """A region across north is one region; small regions become speckle."""
    ds, _ = _sweep()
    z = np.full((NRAY, NG), np.nan)
    z[-3:, 50:58] = 30.0  # 3 rays before north ...
    z[:2, 50:58] = 30.0  # ... and 2 after: 40 gates in one region
    z[100:102, 50:56] = 30.0  # 12 gates
    ds = xr.Dataset(
        {"DBZH": (("azimuth", "range"), z)}, coords=ds.coords, attrs=ds.attrs
    )
    cls = echo_mask(ds, min_size=20, engine=engine).ECHO_CLASS.values
    assert (cls[-3:, 50:58] == 1).all() and (cls[:2, 50:58] == 1).all()
    assert (cls[100:102, 50:56] == 3).all()
    # without wrap (a sector sweep) both halves are separate and too small
    sector = ds.assign_coords(azimuth=np.linspace(0.5, 100.0, NRAY))
    cls = echo_mask(sector, min_size=30, engine=engine).ECHO_CLASS.values
    assert (cls[-3:, 50:58] == 3).all() and (cls[:2, 50:58] == 3).all()
    cls = echo_mask(ds, min_size=0, engine=engine).ECHO_CLASS.values
    assert (cls[100:102, 50:56] == 1).all()


@pytest.mark.parametrize("engine", ENGINES)
def test_rhi_and_snr(engine):
    ds, truth = _sweep(nray=60)
    rhi = xr.Dataset(
        {k: (("elevation", "range"), ds[k].values) for k in ds.data_vars},
        coords={"elevation": np.linspace(0, 30, 60), "range": ds.range},
    )
    snr = np.full(rhi.DBZH.shape, 20.0)
    snr[:, 20:40] = 0.0
    rhi["SNRH"] = (("elevation", "range"), snr)
    out = echo_mask(rhi, nodata="nexrad", engine=engine)
    assert out.ECHO_CLASS.dims == ("elevation", "range")
    cls = out.ECHO_CLASS.values
    assert (cls[:, 20:40] == 0).all()
    assert np.mean(cls[:, 60:150] == 1) > 0.99
    # transposed input keeps its dimension order
    t = echo_mask(ds.transpose("range", "azimuth"), engine=engine)
    assert t.ECHO_CLASS.dims == ("range", "azimuth")
    np.testing.assert_array_equal(
        t.ECHO_CLASS.values.T, echo_mask(ds, engine=engine).ECHO_CLASS.values
    )


def test_nodata_options():
    ds, truth = _sweep()
    plain = ds.copy()
    plain.attrs = {}
    # without NEXRAD detection the codes are data: no gate without echo
    cls = echo_mask(plain).ECHO_CLASS.values
    assert (cls > 0).all()
    np.testing.assert_array_equal(
        echo_mask(plain, nodata="nexrad").ECHO_CLASS.values,
        echo_mask(ds).ECHO_CLASS.values,
    )
    cls = echo_mask(plain, nodata={"dbzh": -32.0}).ECHO_CLASS.values
    assert (cls[truth == 0] == 0).all()
    with pytest.raises(ValueError, match="nodata"):
        echo_mask(ds, nodata="bad")
    with pytest.raises(ValueError, match="nodata keys"):
        echo_mask(ds, nodata={"vel": 0})
    with pytest.raises(ValueError, match="nodata"):
        echo_mask(ds, nodata=3)


def test_option_errors(monkeypatch):
    ds, _ = _sweep(nray=20)
    with pytest.raises(ValueError, match="unknown limits"):
        echo_mask(ds, limits={"foo": (0, 1, 2, 3)})
    with pytest.raises(ValueError, match="non-decreasing"):
        echo_mask(ds, limits={"rhohv": (1, 0, 2, 3)})
    with pytest.raises(ValueError, match="weights"):
        echo_mask(ds, weights={k: 0 for k in qcmod.FEATURES})
    with pytest.raises(ValueError, match="window"):
        echo_mask(ds, window=0)
    with pytest.raises(ValueError, match="engine"):
        echo_mask(ds, engine="fortran")
    with pytest.raises(KeyError):
        echo_mask(ds, zdr="nope")
    with pytest.raises(ValueError, match="dimensions"):
        echo_mask(ds.isel(azimuth=0))
    with pytest.raises(ValueError, match="uniformly"):
        echo_mask(ds.isel(range=[0, 1, 5, 6]))
    monkeypatch.setattr(qcmod, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError):
        echo_mask(ds, engine="compiled")


def test_custom_limits_and_weights():
    ds, truth = _sweep()
    # a strict rhohv-only classifier keeps only gates with rhohv near 1
    weights = {k: float(k == "rhohv") for k in qcmod.FEATURES}
    cls = echo_mask(ds, weights=weights, threshold=0.99).ECHO_CLASS.values
    assert _share(cls, truth, 1, [1]) > 0.99
    loose = {"zdr": (-np.inf, -np.inf, np.inf, np.inf)}
    out = echo_mask(ds, limits=loose, weights=weights | {"zdr": 1.0})
    assert out.METEO_SCORE.attrs["comment"].endswith(">= 0.6")


@pytest.mark.parametrize("engine", ENGINES)
def test_volume_and_split_cuts(engine):
    dtree = _volume()
    out = echo_mask(dtree, engine=engine)
    assert set(out.children) == {"sweep_0", "sweep_1", "sweep_2"}
    assert out.attrs == dtree.attrs
    surv = out["sweep_0"].ds.ECHO_CLASS.values
    dop = out["sweep_1"].ds.ECHO_CLASS.values
    # the Doppler cut has no polarimetric fields: classes of the surveillance cut
    np.testing.assert_array_equal(dop, surv)
    assert "sweep_0" in out["sweep_1"].ds.ECHO_CLASS.attrs["split_cut_source"]
    own = echo_mask(dtree, split_cuts=False, engine=engine)
    assert (own["sweep_1"].ds.ECHO_CLASS.values == 1).sum() > (dop == 1).sum()
    # one call for the volume = sweep by sweep
    single = echo_mask(dtree["sweep_2"].to_dataset(), engine=engine)
    np.testing.assert_array_equal(
        out["sweep_2"].ds.ECHO_CLASS.values, single.ECHO_CLASS.values
    )
    with pytest.raises(KeyError, match="reflectivity"):
        echo_mask(xr.DataTree.from_dict({"sweep_0": xr.Dataset({"X": 1.0})}))


def test_split_cut_without_partner():
    """A reflectivity-only sweep at another elevation keeps its own class."""
    surv, _ = _sweep(0)
    dop = surv[["DBZH"]].assign_coords(elevation=("azimuth", np.full(NRAY, 3.0)))
    dtree = xr.DataTree.from_dict({"sweep_0": surv, "sweep_1": dop})
    out = echo_mask(dtree)
    single = echo_mask(dop)
    np.testing.assert_array_equal(
        out["sweep_1"].ds.ECHO_CLASS.values, single.ECHO_CLASS.values
    )
    assert "split_cut_source" not in out["sweep_1"].ds.ECHO_CLASS.attrs


def test_apply_mask_sweep():
    ds, truth = _sweep()
    ds["VRADH"] = (("azimuth", "range"), np.full(ds.DBZH.shape, -64.0))
    ds["VRADH"][:, 100:] = 3.0
    ds["COUNT"] = (("azimuth", "range"), np.ones(ds.DBZH.shape, dtype=np.int32))
    qc = echo_mask(ds)
    out = apply_mask(ds, qc)
    keep = qc.METEO_MASK.values
    for name in ("DBZH", "ZDR", "RHOHV", "PHIDP"):
        np.testing.assert_array_equal(np.isfinite(out[name].values), keep)
        assert out[name].attrs == ds[name].attrs
    # NEXRAD velocity code masked as well
    assert np.isnan(out.VRADH.values[:, :100]).all()
    assert np.isfinite(out.VRADH.values[keep & (np.arange(NG) >= 100)]).all()
    np.testing.assert_array_equal(out.COUNT.values, ds.COUNT.values)  # not float
    # computed on the fly, field subset, boolean DataArray mask
    np.testing.assert_array_equal(apply_mask(ds).DBZH.values, out.DBZH.values)
    sub = apply_mask(ds, qc.METEO_MASK, "ZDR")
    np.testing.assert_array_equal(sub.DBZH.values, ds.DBZH.values)
    np.testing.assert_array_equal(np.isfinite(sub.ZDR.values), keep)
    raw = apply_mask(ds, qc, ["VRADH"], nodata=None)
    assert np.isfinite(raw.VRADH.values[keep]).all()


def test_apply_mask_errors():
    ds, _ = _sweep(nray=20)
    qc = echo_mask(ds)
    with pytest.raises(ValueError, match="nodata"):
        apply_mask(ds, qc, nodata="x")
    with pytest.raises(TypeError, match="only used"):
        apply_mask(ds, qc, window=2.0)
    with pytest.raises(TypeError, match="Dataset or DataArray"):
        apply_mask(ds, xr.DataTree(qc))
    with pytest.raises(TypeError, match="DataArray, Dataset"):
        apply_mask(ds, qc.METEO_MASK.values)
    with pytest.raises(KeyError, match="METEO_MASK"):
        apply_mask(ds, qc.drop_vars("METEO_MASK"))
    with pytest.raises(KeyError, match="nope"):
        apply_mask(ds, qc, "nope")
    with pytest.raises(ValueError, match="dimensions"):
        apply_mask(ds, qc.METEO_MASK.isel(range=0))
    with pytest.raises(ValueError, match="shape"):
        apply_mask(ds, qc.METEO_MASK.isel(range=slice(0, 5)))
    with pytest.raises(TypeError, match="DataTree output"):
        apply_mask(_volume(), qc)


def test_apply_mask_volume_and_accessors():
    dtree = _volume()
    qc = dtree.radarx.echo_mask()
    out = dtree.radarx.apply_mask(qc)
    keep = qc["sweep_1"].ds.METEO_MASK.values
    vel = out["sweep_1"].ds.VRADH.values
    np.testing.assert_array_equal(np.isfinite(vel), keep & (np.arange(NG) >= 10))
    assert without_provenance(out["sweep_3"].ds).identical(dtree["sweep_3"].ds)
    assert out.attrs == dtree.attrs
    # a field present in only some sweeps
    only = apply_mask(dtree, qc, "VRADH")
    np.testing.assert_array_equal(
        only["sweep_0"].ds.DBZH.values, dtree["sweep_0"].ds.DBZH.values
    )
    np.testing.assert_array_equal(only["sweep_1"].ds.VRADH.values, vel)
    # a mask for fewer sweeps than the volume
    part = xr.DataTree.from_dict(
        {"sweep_2": qc["sweep_2"].ds, "sweep_9": qc["sweep_2"].ds}
    )
    some = apply_mask(dtree, part)
    np.testing.assert_array_equal(
        some["sweep_0"].ds.DBZH.values, dtree["sweep_0"].ds.DBZH.values
    )
    ds = dtree["sweep_2"].to_dataset()
    sweep_qc = ds.radarx.echo_mask()
    np.testing.assert_array_equal(
        sweep_qc.ECHO_CLASS.values, qc["sweep_2"].ds.ECHO_CLASS.values
    )
    np.testing.assert_array_equal(
        ds.radarx.apply_mask(data_vars="DBZH").DBZH.values,
        out["sweep_2"].ds.DBZH.values,
    )


# --------------------------------------------------------------------------
# real NEXRAD volumes
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def klbb():
    import xradar as xd
    from open_radar_data import DATASETS

    return xd.io.open_nexradlevel2_datatree(DATASETS.fetch("KLBB20160601_150025_V06"))


@pytest.fixture(scope="module")
def kgwx(tmp_path_factory):
    """Squall line and biological echo (KGWX, 30 March 2022), from AWS."""
    import xradar as xd

    from radarx.io.aws_data import download_file

    try:
        path = download_file(
            "unidata-nexrad-level2",
            "2022/03/30/KGWX/KGWX20220330_234639_V06",
            str(tmp_path_factory.mktemp("nexrad")),
        )
    except Exception as err:  # noqa: BLE001  # pragma: no cover - network
        pytest.skip(f"NEXRAD data not available: {err}")
    return xd.io.open_nexradlevel2_datatree(path)


@pytest.mark.skipif(not qcmod.HAS_COMPILED_KERNEL, reason="kernel not built")
def test_nexrad_engines_identical(klbb):
    a = echo_mask(klbb, engine="numpy")
    b = echo_mask(klbb, engine="compiled")
    for name in a.children:
        np.testing.assert_array_equal(
            a[name].ds.ECHO_CLASS.values, b[name].ds.ECHO_CLASS.values
        )
        np.testing.assert_array_equal(
            a[name].ds.METEO_SCORE.values, b[name].ds.METEO_SCORE.values
        )


def test_kgwx_biological_echo_removed_squall_line_kept(kgwx):
    """Default call: ZDR > 4 dB east of the radar removed, squall line kept."""
    out = echo_mask(kgwx)
    ds = kgwx["sweep_0"].to_dataset()
    cls = out["sweep_0"].ds.ECHO_CLASS.values
    az = np.deg2rad(ds.azimuth.values)[:, None]
    r = ds.range.values[None, :] / 1000.0
    x, y = r * np.sin(az), r * np.cos(az)
    z, zdr = ds.DBZH.values, ds.ZDR.values
    bio = (x > 85) & (x < 130) & (abs(y) < 45) & (zdr > 4) & (z < 20) & (cls > 0)
    squall = (x > -170) & (x < 20) & (z >= 35)
    assert bio.sum() > 3000 and squall.sum() > 100_000
    assert np.mean(cls[bio] != 1) > 0.95
    assert np.mean(cls[squall] == 1) > 0.995
    # NEXRAD no-data codes never count as echo
    assert (cls[z <= -32] == 0).all()


def _ramp_rain(kdp, nray=30, ng=200, seed=0):
    """Rain with a constant KDP (a phase ramp) and phase noise of 3 degrees."""
    rnd = np.random.default_rng(seed)
    r_km = (np.arange(ng) + 0.5) * DR / 1000.0
    phi = (100.0 + 2.0 * kdp * r_km + rnd.normal(0, 3.0, (nray, ng)) + 180) % 360 - 180
    return xr.Dataset(
        {
            "DBZH": (("azimuth", "range"), np.full((nray, ng), 35.0)),
            "PHIDP": (("azimuth", "range"), phi, {"units": "degrees"}),
        },
        coords={
            "azimuth": np.arange(nray) * 360.0 / nray,
            "range": (np.arange(ng) + 0.5) * DR,
            "elevation": ("azimuth", np.full(nray, 0.5)),
        },
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_detrended_phidp_texture_ignores_the_rain_ramp(engine):
    """Only phidp_texture counts: the ramp of KDP 30 lowers the circular score."""
    only = {k: 0.0 for k in qcmod.FEATURES if k != "phidp_texture"}
    scores = {}
    for kdp in (0.0, 30.0):
        ds = _ramp_rain(kdp)
        for rule in ("circular", "detrended"):
            out = echo_mask(
                ds, weights=only, phidp_texture=rule, min_size=1, engine=engine
            )
            scores[kdp, rule] = float(out.METEO_SCORE.isel(range=slice(20, 180)).mean())
    assert scores[0.0, "circular"] == pytest.approx(1.0, abs=0.01)
    assert scores[30.0, "circular"] < 0.8
    assert scores[30.0, "detrended"] == pytest.approx(1.0, abs=0.01)
    assert scores[0.0, "detrended"] == pytest.approx(1.0, abs=0.01)


@pytest.mark.skipif(not qcmod.HAS_COMPILED_KERNEL, reason="compiled kernel not built")
def test_detrended_phidp_texture_engines_agree():
    ds, _ = _sweep(2)
    a = echo_mask(ds, phidp_texture="detrended", engine="numpy")
    b = echo_mask(ds, phidp_texture="detrended", engine="compiled")
    np.testing.assert_allclose(
        a.METEO_SCORE.values, b.METEO_SCORE.values, atol=1e-6, equal_nan=True
    )
    np.testing.assert_array_equal(a.ECHO_CLASS.values, b.ECHO_CLASS.values)
    # random phase (biological echo, clutter) is still rejected
    ds, truth = _sweep(2)
    got = b.ECHO_CLASS.values
    assert (got[truth == 2] == 2).mean() > 0.9


def test_phidp_texture_option_is_checked():
    ds, _ = _sweep(0)
    with pytest.raises(ValueError, match="phidp_texture"):
        echo_mask(ds, phidp_texture="x")


# --------------------------------------------------------------------------
# method="krause2016": MetSignal of Krause (2016, JAOT 33, 1875-1885)
# --------------------------------------------------------------------------


def _krause_sweep(z, zdr=None, rho=None, phi=None, vel=None, zh3=None):
    """Sweep Dataset from (ray, range) arrays; missing fields are left out."""
    nray, ng = z.shape
    dims = ("azimuth", "range")
    data = {"DBZH": (dims, z)}
    for name, arr in (
        ("ZDR", zdr),
        ("RHOHV", rho),
        ("PHIDP", phi),
        ("VRADH", vel),
        ("ZH3KM", zh3),
    ):
        if arr is not None:
            data[name] = (dims, arr)
    return xr.Dataset(
        data,
        coords={
            "azimuth": np.arange(nray) * 360.0 / nray,
            "range": (np.arange(ng) + 0.5) * DR,
            "elevation": ("azimuth", np.full(nray, 0.5)),
        },
    )


def _ideal(nray=6, ng=60):
    """Gates with every MetSignal term at full membership."""
    one = np.ones((nray, ng))
    return dict(z=35 * one, zdr=0.5 * one, rho=0.99 * one, phi=30 * one, vel=5 * one)


def _krause_reference(ds, threshold=0.8, zh3=None):
    """
    Plain-loop implementation of Krause (2016) written from the paper:
    Fig. 2 memberships (interpolation tables), Table 1 weights, nine-gate
    standard deviations along the radial, the aggregation of p. 1876 and the
    post-processing rules of pp. 1875-1876. Returns (A, meteorological).
    """
    z = ds.DBZH.values
    zdr = ds.ZDR.values if "ZDR" in ds else None
    rho = ds.RHOHV.values if "RHOHV" in ds else None
    phi = ds.PHIDP.values if "PHIDP" in ds else None
    vel = ds.VRADH.values if "VRADH" in ds else None
    nray, ng = z.shape
    agg = np.full((nray, ng), np.nan)
    met = np.zeros((nray, ng), dtype=bool)
    for r in range(nray):
        for g in range(ng):
            if not np.isfinite(z[r, g]):
                continue
            lo, hi = max(g - 4, 0), min(g + 4, ng - 1) + 1
            terms = [(1.0, np.interp(z[r, g], [10, 30], [0, 1]))]
            if rho is not None:
                terms.append((1.0, np.interp(rho[r, g], [0.75, 0.9], [0, 1])))
            if vel is not None:
                terms.append(
                    (1.0, 1 - np.interp(vel[r, g], [-1.5, -1, 1, 1.5], [0, 1, 1, 0]))
                )
            # gates without echo are not part of a window; fewer than three
            # gates: no standard deviation
            use = np.isfinite(z[r, lo:hi])
            if use.sum() >= 3 and phi is not None:
                w = np.deg2rad(phi[r, lo:hi][use])
                big_r = np.hypot(np.cos(w).mean(), np.sin(w).mean())
                sd = np.degrees(np.sqrt(-2 * np.log(big_r))) if big_r < 1 else 0.0
                terms.append((2.0, np.interp(sd, [10, 20], [1, 0])))
            if use.sum() >= 3 and zdr is not None:
                sd = np.std(zdr[r, lo:hi][use])
                terms.append((2.0, np.interp(sd, [1, 2], [1, 0])))
            if use.sum() >= 3 and rho is not None:
                sd = np.std(rho[r, lo:hi][use])
                terms.append((1.0, np.interp(sd, [0.03, 0.04], [1, 0])))
            w_sum = sum(w for w, _ in terms)
            a = sum(w * m for w, m in terms) / w_sum
            agg[r, g] = a
            ok = a >= threshold
            if zdr is not None and abs(zdr[r, g]) > 4.5:
                ok = False
            if rho is not None and rho[r, g] < 0.65:
                ok = False
            if zh3 is not None and zh3[r, g] >= 11.0:
                ok = True
            met[r, g] = ok
    return agg, met


def _random_krause_sweep(seed, nray=14, ng=70):
    """Mixed gates: smooth rain, noisy echo, a few missing gates."""
    rnd = np.random.default_rng(seed)
    g = np.arange(ng)
    z = rnd.uniform(-5, 45, (nray, ng))
    smooth = g < 35
    z[:, smooth] = 25 + 5 * np.sin(g[smooth] / 6.0)
    noise = rnd.normal(0, 0.004, (nray, ng))
    rho = np.where(smooth, 0.985 + noise, rnd.uniform(0.5, 1.0, (nray, ng)))
    zdr = np.where(
        smooth, 0.8 + rnd.normal(0, 0.3, (nray, ng)), rnd.normal(0, 3.0, (nray, ng))
    )
    ramp = 40 + 0.3 * g + rnd.normal(0, 3, (nray, ng))
    phi = np.where(smooth, ramp, rnd.uniform(0, 360, (nray, ng)))
    vel = rnd.normal(0, 6, (nray, ng))
    zh3 = np.where(rnd.random((nray, ng)) < 0.05, 15.0, np.nan)
    z[rnd.random((nray, ng)) < 0.05] = np.nan
    return _krause_sweep(z, zdr=zdr, rho=rho, phi=phi, vel=vel, zh3=zh3)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_krause_matches_an_independent_implementation(engine, seed):
    """Memberships, weights, window, aggregation and rules, gate by gate."""
    ds = _random_krause_sweep(seed)
    for threshold, zh3 in ((0.8, None), (0.7, "ZH3KM")):
        out = echo_mask(
            ds,
            method="krause2016",
            field="VRADH",
            zh_3km=zh3,
            threshold=threshold,
            nodata=None,
            engine=engine,
        )
        ref_a, ref_met = _krause_reference(
            ds, threshold, None if zh3 is None else np.nan_to_num(ds.ZH3KM.values)
        )
        np.testing.assert_allclose(
            out.METEO_SCORE.values, ref_a, atol=2e-6, equal_nan=True
        )
        echo = np.isfinite(ds.DBZH.values)
        np.testing.assert_array_equal(out.METEO_MASK.values, ref_met & echo)
        np.testing.assert_array_equal(
            out.ECHO_CLASS.values, np.where(echo, np.where(ref_met, 1, 2), 0)
        )


@pytest.mark.parametrize("engine", ENGINES)
def test_krause_weights_and_memberships_by_hand(engine):
    """Every term at 0.5 or 1 gives A = 6.5 / 8 (Table 1 weights, Fig. 2 corners)."""
    f = _ideal()
    f["z"] = f["z"] * 0 + 20.0  # Zh: 0.5
    f["rho"] = f["rho"] * 0 + 0.825  # rhohv: 0.5
    f["vel"] = f["vel"] * 0 - 1.25  # V: 0.5
    out = echo_mask(
        _krause_sweep(**f),
        method="krause2016",
        field="VRADH",
        nodata=None,
        engine=engine,
    )
    interior = out.METEO_SCORE.values[:, 8:-8]
    np.testing.assert_allclose(interior, 6.5 / 8, atol=1e-6)


@pytest.mark.parametrize("engine", ENGINES)
def test_krause_texture_of_phidp_and_zdr_counts_twice(engine):
    """
    rhohv 0.85 (2/3), SD(ZDR) 1.5 dB (0.5), SD(PhiDP) 25 deg (0), Z 30 dBZ and
    V 2 m/s: A = (1 + 2/3 + 1 + 2 * 0 + 2 * 0.5 + 1) / 8 = 0.583, below both
    thresholds of the paper, so the echo is non-meteorological. With weight 1
    for the two textures it would be 0.667 and meteorological at 0.6.
    """
    f = _ideal(ng=80)
    alt = np.tile([0.0, 1.0], 40)[None, :] * np.ones((6, 1))
    f["z"] = f["z"] * 0 + 30.0
    f["rho"] = f["rho"] * 0 + 0.85
    f["vel"] = f["vel"] * 0 + 2.0
    f["zdr"] = 3.0 * alt  # standard deviation over nine gates 1.49 dB
    f["phi"] = 50.0 * alt  # about 25 degrees
    inner = (slice(None), slice(10, 70))
    for threshold in (None, 0.7):
        out = echo_mask(
            _krause_sweep(**f),
            method="krause2016",
            field="VRADH",
            threshold=threshold,
            nodata=None,
            engine=engine,
        )
        np.testing.assert_allclose(out.METEO_SCORE.values[inner], 0.583, atol=0.01)
        assert (out.ECHO_CLASS.values[inner] == 2).all()


@pytest.mark.parametrize("engine", ENGINES)
def test_krause_post_processing_rules(engine):
    """abs(ZDR) > 4.5 dB and rhohv < 0.65 reject; Zh >= 11 dBZ at 3 km restores."""
    f = _ideal(ng=40)
    f["zdr"][:, 5] = 5.0  # rule 1
    f["zdr"][:, 15] = -4.4  # inside the limit
    f["rho"][:, 25] = 0.60  # rule 2
    kw = dict(method="krause2016", field="VRADH", nodata=None, engine=engine)
    out = echo_mask(_krause_sweep(**f), **kw)
    cls = out.ECHO_CLASS.values
    assert (cls[:, 5] == 2).all() and (cls[:, 25] == 2).all()
    assert (cls[:, 30] == 1).all()
    # with a threshold that every aggregation value reaches, only the rules
    # reject: the gates of rule 1 and rule 2 and nothing else
    low = echo_mask(_krause_sweep(**f), threshold=0.1, **{**kw, "method": "krause2016"})
    assert (low.METEO_SCORE.values >= 0.1).all()
    rejected = np.zeros(f["z"].shape, dtype=bool)
    rejected[:, [5, 25]] = True
    np.testing.assert_array_equal(low.ECHO_CLASS.values == 2, rejected)
    # the override of the previous volume wins over both rules
    zh3 = np.full(f["z"].shape, np.nan)
    zh3[:, 5] = 11.0
    zh3[:, 25] = 10.9
    over = echo_mask(_krause_sweep(**f, zh3=zh3), zh_3km="ZH3KM", **kw)
    assert (over.ECHO_CLASS.values[:, 5] == 1).all()
    assert (over.ECHO_CLASS.values[:, 25] == 2).all()
    # the score is the aggregation value, before the rules
    np.testing.assert_array_equal(over.METEO_SCORE.values, out.METEO_SCORE.values)


@pytest.mark.parametrize("engine", ENGINES)
def test_krause_velocity_term_penalises_stationary_echo(engine):
    f = _ideal()
    kw = dict(method="krause2016", nodata=None, engine=engine)
    mid = (slice(None), slice(8, -8))
    still = echo_mask(_krause_sweep(**f | {"vel": f["vel"] * 0}), field="VRADH", **kw)
    moving = echo_mask(_krause_sweep(**f), field="VRADH", **kw)
    np.testing.assert_allclose(still.METEO_SCORE.values[mid], 7 / 8, atol=1e-6)
    np.testing.assert_allclose(moving.METEO_SCORE.values[mid], 1.0, atol=1e-6)
    cut = echo_mask(
        _krause_sweep(**f | {"vel": f["vel"] * 0}), field="VRADH", threshold=0.9, **kw
    )
    assert (cut.ECHO_CLASS.values[mid] == 2).all()
    # without a velocity field the term is dropped from the weighted mean
    none = echo_mask(_krause_sweep(**{k: v for k, v in f.items() if k != "vel"}), **kw)
    np.testing.assert_allclose(none.METEO_SCORE.values[mid], 1.0, atol=1e-6)


def test_krause_defaults_and_options():
    ds = _krause_sweep(**_ideal())
    out = echo_mask(ds, method="krause2016", field="VRADH", nodata=None)
    assert ">= 0.8" in out.METEO_SCORE.attrs["comment"]
    assert "Krause" in out.METEO_SCORE.attrs["comment"]
    assert out.ECHO_CLASS.attrs["source_fields"] == "DBZH, ZDR, RHOHV, PHIDP, VRADH"
    # no speckle filter unless asked for; a lone meteorological gate stays
    f = _ideal()
    f["z"] = f["z"] * 0 + 5.0
    f["rho"] = f["rho"] * 0 + 0.5
    f["z"][3, 30] = 40.0
    f["rho"][3, 30] = 0.99
    lone = echo_mask(_krause_sweep(**f), method="krause2016", nodata=None)
    assert lone.ECHO_CLASS.values[3, 30] == 1
    speckle = echo_mask(
        _krause_sweep(**f), method="krause2016", nodata=None, min_size=10
    )
    assert speckle.ECHO_CLASS.values[3, 30] == 3
    # a velocity field of another name
    out = echo_mask(ds.rename(VRADH="VEL"), method="krause2016", field="VEL")
    assert "VEL" in out.ECHO_CLASS.attrs["source_fields"]
    # the default scheme is the radarx one
    a = echo_mask(ds)
    b = echo_mask(ds, method="radarx")
    xr.testing.assert_identical(a, b)
    assert b.METEO_SCORE.attrs["comment"].endswith(">= 0.6")


def test_krause_option_errors():
    ds = _krause_sweep(**_ideal())
    with pytest.raises(ValueError, match="method"):
        echo_mask(ds, method="gourley2007")
    for bad in (
        {"limits": {"rhohv": (0, 1, 2, 3)}},
        {"weights": {"rhohv": 2.0}},
        {"window": 3.0},
    ):
        with pytest.raises(ValueError, match="krause2016"):
            echo_mask(ds, method="krause2016", **bad)
    with pytest.raises(ValueError, match="krause2016"):
        echo_mask(ds, field="VRADH")
    with pytest.raises(ValueError, match="krause2016"):
        echo_mask(ds, zh_3km="ZH3KM")
    with pytest.raises(KeyError):
        echo_mask(ds, method="krause2016", field="nope")
    with pytest.raises(KeyError):
        echo_mask(ds, method="krause2016", zh_3km="nope")
    with pytest.raises(ValueError, match="nodata"):
        echo_mask(ds, method="krause2016", nodata={"vel": 0})
    with pytest.raises(ValueError, match="nodata"):
        echo_mask(ds, nodata={"vrad": 0})  # a key of the Krause method only
    echo_mask(ds, method="krause2016", field="VRADH", nodata={"vrad": -50.0})


@pytest.mark.skipif(not qcmod.HAS_COMPILED_KERNEL, reason="kernel not built")
@pytest.mark.parametrize("seed", [3, 4])
def test_krause_engines_identical(seed):
    ds = _random_krause_sweep(seed, nray=40, ng=120).isel(azimuth=slice(0, 33))
    for kwargs in (
        {},
        {"field": "VRADH", "zh_3km": "ZH3KM", "threshold": 0.7},
        {"field": "VRADH", "min_size": 12},
    ):
        a = echo_mask(ds, method="krause2016", engine="numpy", **kwargs)
        b = echo_mask(ds, method="krause2016", engine="compiled", **kwargs)
        np.testing.assert_array_equal(a.ECHO_CLASS.values, b.ECHO_CLASS.values)
        np.testing.assert_array_equal(a.METEO_SCORE.values, b.METEO_SCORE.values)
    one = echo_mask(ds, method="krause2016", n_threads=1, field="VRADH")
    for n in (2, 5):
        many = echo_mask(ds, method="krause2016", n_threads=n, field="VRADH")
        np.testing.assert_array_equal(one.ECHO_CLASS.values, many.ECHO_CLASS.values)


def test_krause_does_not_modify_the_input():
    ds = _random_krause_sweep(5)
    for name in ds.data_vars:
        ds[name].attrs["units"] = "x"
    ds.range.attrs["units"] = "m"
    before = ds.copy(deep=True)
    echo_mask(ds, method="krause2016", field="VRADH", zh_3km="ZH3KM", min_size=3)
    xr.testing.assert_identical(ds, before)
    for name in ds.variables:
        assert ds[name].attrs == before[name].attrs


@pytest.mark.parametrize("engine", ENGINES)
def test_krause_nexrad_codes_are_no_data(engine):
    """The NEXRAD velocity code -64.5 m/s is no data, not a moving echo."""
    f = _ideal()
    f["vel"] = f["vel"] * 0 - 64.5
    ds = _krause_sweep(**f)
    ds.attrs["waveform_type"] = "contiguous_surveillance"
    with_code = echo_mask(ds, method="krause2016", field="VRADH", engine=engine)
    without = echo_mask(ds.drop_vars("VRADH"), method="krause2016", engine=engine)
    np.testing.assert_array_equal(
        with_code.METEO_SCORE.values, without.METEO_SCORE.values
    )
    plain = echo_mask(
        ds, method="krause2016", field="VRADH", nodata=None, engine=engine
    )
    assert np.nanmean(plain.METEO_SCORE.values) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("engine", ENGINES)
def test_krause_volume_and_split_cuts(engine):
    dtree = _volume()
    out = echo_mask(dtree, method="krause2016", field="VRADH", engine=engine)
    surv = out["sweep_0"].ds.ECHO_CLASS.values
    dop = out["sweep_1"].ds.ECHO_CLASS.values
    # the Doppler cut (reflectivity and velocity only) takes the polarimetric class
    both = (surv > 0) & (dop > 0)
    assert both.sum() > 1000
    np.testing.assert_array_equal(dop[both], surv[both])
    assert "sweep_3" not in out.children
    own = echo_mask(
        dtree, method="krause2016", field="VRADH", split_cuts=False, engine=engine
    )
    assert own["sweep_1"].ds.METEO_SCORE.attrs["comment"].startswith("aggregation")


@pytest.mark.skipif(not qcmod.HAS_COMPILED_KERNEL, reason="kernel not built")
def test_krause_nexrad_engines_identical(klbb):
    a = echo_mask(klbb, method="krause2016", engine="numpy")
    b = echo_mask(klbb, method="krause2016", engine="compiled")
    for name in a.children:
        np.testing.assert_array_equal(
            a[name].ds.ECHO_CLASS.values, b[name].ds.ECHO_CLASS.values
        )
        np.testing.assert_array_equal(
            a[name].ds.METEO_SCORE.values, b[name].ds.METEO_SCORE.values
        )


def test_krause_kgwx_removes_biological_echo_and_keeps_the_squall_line(kgwx):
    out = echo_mask(kgwx, method="krause2016")
    ds = kgwx["sweep_0"].to_dataset()
    cls = out["sweep_0"].ds.ECHO_CLASS.values
    az = np.deg2rad(ds.azimuth.values)[:, None]
    r = ds.range.values[None, :] / 1000.0
    x, y = r * np.sin(az), r * np.cos(az)
    z, zdr = ds.DBZH.values, ds.ZDR.values
    bio = (x > 85) & (x < 130) & (abs(y) < 45) & (zdr > 4) & (z < 20) & (cls > 0)
    squall = (x > -170) & (x < 20) & (z >= 35)
    assert np.mean(cls[bio] != 1) > 0.95
    assert np.mean(cls[squall] == 1) > 0.9
    assert (cls[z <= -32] == 0).all()


@pytest.mark.parametrize("engine", ENGINES)
def test_krause_synthetic_echo_types(engine):
    """Rain, biological echo and clutter (described in the paper) and AP."""
    ds, truth = _sweep()
    rnd = np.random.default_rng(7)
    # clutter does not move, rain and insects do (Krause 2016, p. 1881)
    vel = rnd.normal(0, 4, ds.DBZH.shape)
    vel[truth == 2] = rnd.normal(0, 0.3, (truth == 2).sum())
    vel[:, 180:260] = rnd.normal(0, 4, (ds.sizes["azimuth"], 80))  # insects drift
    ds["VRADH"] = (ds.DBZH.dims, vel)
    out = echo_mask(ds, method="krause2016", field="VRADH", engine=engine)
    cls = out.ECHO_CLASS.values
    assert _share(cls, truth, 1, [1]) > 0.99
    assert _share(cls, truth, 2, [2]) > 0.99
    assert _share(cls, truth, 0, [0]) == 1.0
    # anomalous propagation: strong, rhohv 0.88 +- 0.08, SD(PhiDP) 25 deg, noisy ZDR
    nray, ng = 60, 300
    one = np.ones((nray, ng))
    ap = _krause_sweep(
        40 * one + rnd.normal(0, 3, (nray, ng)),
        zdr=rnd.normal(0, 2.5, (nray, ng)),
        rho=np.clip(rnd.normal(0.88, 0.08, (nray, ng)), 0, 1),
        phi=rnd.normal(0, 25, (nray, ng)) % 360,
    )
    for threshold in (0.8, 0.7):
        out = echo_mask(
            ap, method="krause2016", threshold=threshold, nodata=None, engine=engine
        )
        assert np.mean(out.ECHO_CLASS.values == 1) < 0.01
