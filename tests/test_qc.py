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
    with pytest.raises(ValueError, match="2-D"):
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
        ds.radarx.apply_mask(fields="DBZH").DBZH.values,
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
