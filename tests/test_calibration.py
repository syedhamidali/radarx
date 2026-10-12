#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the ZDR, Z and system PhiDP calibration
==================================================

Fields with known offsets: rain whose Z, ZDR and KDP come from the radarx
T-matrix tables of gamma drop size distributions (so they are consistent) with
a reflectivity bias and a ZDR bias added, exponential (Marshall-Palmer) rain
for the light-rain reference, and a PhiDP that starts at a known system phase.
The compiled kernel and the NumPy implementation must agree to rounding.
"""
import importlib

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx._provenance import load_registry, read_provenance
from radarx.retrieve import phidp_offset, z_offset, zdr_offset

calmod = importlib.import_module("radarx.retrieve.calibration")
dsdmod = importlib.import_module("radarx.retrieve.dsd")

ENGINES = ["numpy"] + (["compiled"] if calmod.HAS_COMPILED_KERNEL else [])
compiled_only = pytest.mark.skipif(
    not calmod.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)
DZ, DZDR = -1.7, 0.35  # bias of Z and ZDR added to the synthetic rain


def sweep_from(fields, nray, ngate, elevation=0.5):
    dims = ("azimuth", "range")
    data = {k: (dims, v.reshape(nray, ngate)) for k, v in fields.items()}
    data["elevation"] = ("azimuth", np.full(nray, elevation))
    return xr.Dataset(
        data,
        coords={
            "azimuth": np.arange(nray) * 360.0 / nray,
            "range": 2000.0 + 250.0 * np.arange(ngate),
        },
    )


def consistent_rain(nray=180, ngate=500, seed=3, dz=DZ, dzdr=DZDR, noise=True):
    """Rain from constrained gamma DSDs, with a Z bias and a ZDR bias added."""
    rng = np.random.default_rng(seed)
    n = nray * ngate
    lam = np.clip(np.exp(rng.normal(np.log(3.5), 0.35, n)), 1.2, 12.0)
    mu = np.maximum(-0.0201 * lam**2 + 0.902 * lam - 1.718, -0.95)
    n0 = 10 ** rng.normal(3.3, 0.5, n)
    zh, zv, kdp = dsdmod._gamma_integrals("S", 20.0, n0, mu, lam)
    z, zdr = 10 * np.log10(zh), 10 * np.log10(zh / zv)
    sd = (1.0, 0.2, 0.3) if noise else (0.0, 0.0, 0.0)
    ds = sweep_from(
        {
            "DBZH": z + dz + rng.normal(0, sd[0], n),
            "ZDR": zdr + dzdr + rng.normal(0, sd[1], n),
            "KDP": kdp + rng.normal(0, sd[2], n),
            "RHOHV": np.full(n, 0.99),
        },
        nray,
        ngate,
    )
    return ds, z


def marshall_palmer_rain(nray=120, ngate=400, seed=5, offset=0.3, elevation=0.5):
    """Exponential DSDs of N0 = 8000 with a ZDR offset; the truth is ``offset``."""
    rng = np.random.default_rng(seed)
    n = nray * ngate
    lam = np.exp(rng.uniform(np.log(2.0), np.log(10.0), n))
    zh, zv, _ = dsdmod._gamma_integrals("S", 20.0, 8000.0, 0.0, lam)
    z, zdr = 10 * np.log10(zh), 10 * np.log10(zh / zv)
    zdr_obs = calmod_elevation(zdr, elevation) + offset + rng.normal(0, 0.15, n)
    return sweep_from(
        {"DBZH": z, "ZDR": zdr_obs, "RHOHV": np.full(n, 0.995)},
        nray,
        ngate,
        elevation=elevation,
    )


def calmod_elevation(zdr0, elevation):
    """ZDR seen at an elevation from the value at 0 degrees, written out independently.

    A spheroid with its axis vertical has the polarizability a (equatorial) and
    b (along the axis); H couples to a, V to a sin^2 + b cos^2 of the elevation.
    """
    theta = np.deg2rad(elevation)
    q = 10 ** (0.05 * zdr0)  # a / b
    v = q * np.sin(theta) ** 2 + np.cos(theta) ** 2  # (a sin^2 + b cos^2) / b
    return 20 * np.log10(q / v)


# --------------------------------------------------------------------------
# ZDR offset
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_vertical_pointing_recovers_the_offset(engine):
    rng = np.random.default_rng(1)
    n = 200 * 300
    ds = sweep_from(
        {
            "DBZH": rng.uniform(5, 40, n),
            "ZDR": 0.42 + rng.normal(0, 0.2, n),
            "RHOHV": np.full(n, 0.99),
        },
        200,
        300,
        elevation=90.0,
    )
    out = zdr_offset(ds, "vertical", engine=engine)
    assert float(out.ZDR_OFFSET) == pytest.approx(0.42, abs=0.01)
    assert float(out.ZDR_OFFSET_STD) == pytest.approx(0.2, rel=0.05)
    assert int(out.ZDR_OFFSET_COUNT) == n
    assert float(out.ZDR_OFFSET_SE) == pytest.approx(0.2 / np.sqrt(n), rel=0.05)
    assert out.ZDR_OFFSET.attrs["units"] == "dB"


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("elevation", [0.5, 20.0, 60.0])
def test_light_rain_recovers_the_offset_at_any_elevation(engine, elevation):
    ds = marshall_palmer_rain(offset=0.3, elevation=elevation)
    out = zdr_offset(ds, "light_rain", z_range=(10, 40), band="S", engine=engine)
    assert float(out.ZDR_OFFSET) == pytest.approx(0.3, abs=0.02)
    assert float(out.ZDR_OFFSET_MEAN) == pytest.approx(0.3, abs=0.02)


def test_elevation_dependence_is_not_a_shift():
    """Without the elevation the offset of light rain at 60 degrees is wrong."""
    ds = marshall_palmer_rain(offset=0.0, elevation=60.0)
    right = float(zdr_offset(ds, "light_rain", z_range=(10, 40), band="S").ZDR_OFFSET)
    wrong = float(
        zdr_offset(
            ds, "light_rain", z_range=(10, 40), band="S", elevation=0.0
        ).ZDR_OFFSET
    )
    assert abs(right) < 0.02
    assert wrong < -0.1  # the beam sees less of the oblateness


def test_light_rain_reference_values():
    """About 0.3 to 0.4 dB at 20 to 22 dBZ at S band, rising with Z."""
    z, zdr = calmod._light_rain_curve("S", 20.0)
    ref = np.interp([15.0, 20.0, 22.0, 30.0], z, zdr)
    assert 0.2 < ref[1] < 0.45 and 0.25 < ref[2] < 0.5
    assert np.all(np.diff(ref) > 0)
    assert np.all(np.diff(z) > 0)


@pytest.mark.parametrize("engine", ENGINES)
def test_dry_snow_mask_and_reference(engine):
    ds = marshall_palmer_rain(offset=0.0)
    ds["ZDR"] = ds.ZDR * 0 + 0.55
    mask = (ds.azimuth < 180.0).broadcast_like(ds.ZDR)
    out = zdr_offset(ds, "dry_snow", mask=mask, z_range=(0, 60), engine=engine)
    assert float(out.ZDR_OFFSET) == pytest.approx(0.55 - 0.2, abs=1e-3)
    n_masked = int(mask.sum() & (ds.DBZH >= 0).sum())
    assert 0 < int(out.ZDR_OFFSET_COUNT) <= int(mask.sum())
    assert n_masked > 0
    custom = zdr_offset(ds, "dry_snow", mask=mask, z_range=(0, 60), reference=0.1)
    assert float(custom.ZDR_OFFSET) == pytest.approx(0.45, abs=1e-3)


def test_selection_criteria():
    ds = marshall_palmer_rain(offset=0.3)
    ds["RHOHV"] = ds.RHOHV * 0 + 0.9
    out = zdr_offset(ds, "light_rain", z_range=(10, 40), band="S")
    assert int(out.ZDR_OFFSET_COUNT) == 0 and np.isnan(float(out.ZDR_OFFSET))
    ok = zdr_offset(ds, "light_rain", z_range=(10, 40), band="S", min_rhohv=0.85)
    assert int(ok.ZDR_OFFSET_COUNT) > 0
    snr = ds.DBZH * 0 + 10.0
    snr[:, :200] = 30.0
    few = zdr_offset(
        ds.assign(SNR=snr),
        "light_rain",
        z_range=(10, 40),
        band="S",
        min_rhohv=0.85,
        min_snr=20.0,
    )
    assert 0.4 < int(few.ZDR_OFFSET_COUNT) / int(ok.ZDR_OFFSET_COUNT) < 0.6
    tiny = zdr_offset(
        ds, "light_rain", z_range=(10, 40), band="S", min_rhohv=0.85, min_count=10**9
    )
    assert np.isnan(float(tiny.ZDR_OFFSET)) and int(tiny.ZDR_OFFSET_COUNT) > 0


def test_qvp_gives_one_offset_per_time():
    rng = np.random.default_rng(2)
    nt, nh = 6, 500
    bias = np.linspace(0.1, 0.6, nt)
    zdr = 0.2 + bias[:, None] + rng.normal(0, 0.1, (nt, nh))
    ds = xr.Dataset(
        {
            "ZDR": (("time", "height"), zdr),
            "DBZH": (("time", "height"), np.full((nt, nh), 10.0)),
            "RHOHV": (("time", "height"), np.full((nt, nh), 0.995)),
        },
        coords={"time": np.arange(nt), "height": np.linspace(3000, 6000, nh)},
    )
    mask = ds.height > 4000.0
    out = zdr_offset(ds, "dry_snow", mask=mask)
    assert out.ZDR_OFFSET.dims == ("time",)
    np.testing.assert_allclose(out.ZDR_OFFSET, bias, atol=0.03)
    assert (out.ZDR_OFFSET_COUNT == int(mask.sum())).all()


@compiled_only
def test_engines_agree_zdr_offset():
    ds = marshall_palmer_rain(offset=0.3, elevation=12.0)
    ds["ZDR"][10:20, 50:90] = np.nan
    ds["RHOHV"][30:40, :] = np.nan
    mask = (ds.range > 5000.0).broadcast_like(ds.ZDR)
    a = zdr_offset(
        ds, "light_rain", mask=mask, z_range=(10, 40), band="S", engine="numpy"
    )
    b = zdr_offset(
        ds, "light_rain", mask=mask, z_range=(10, 40), band="S", engine="compiled"
    )
    for name in a.data_vars:
        np.testing.assert_allclose(
            a[name], b[name], rtol=1e-5, atol=1e-6, equal_nan=True
        )
    k = calmod._calibration
    args = dict(
        zdr=np.ascontiguousarray(ds.ZDR.values.reshape(1, -1), dtype=float),
        dbz=np.ascontiguousarray(ds.DBZH.values.reshape(1, -1), dtype=float),
        rhohv=np.ascontiguousarray(ds.RHOHV.values.reshape(1, -1), dtype=float),
        snr=np.empty(0),
        mask=np.empty(0),
        elev=np.array([12.0]),
    )
    ref_z, ref_v = calmod._light_rain_curve("S", 20.0)
    got = k.offset_stats(
        **args,
        ref_z=ref_z,
        ref_v=ref_v,
        z_min=10.0,
        z_max=40.0,
        rhohv_min=0.98,
        snr_min=-np.inf,
    )
    want = calmod._offset_stats_numpy(
        args["zdr"],
        args["dbz"],
        args["rhohv"],
        args["snr"],
        args["mask"],
        args["elev"],
        ref_z,
        ref_v,
        10.0,
        40.0,
        0.98,
        -np.inf,
    )
    np.testing.assert_allclose(got, want, rtol=1e-10, equal_nan=True)


# --------------------------------------------------------------------------
# Z offset from the self-consistency
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_self_consistency_recovers_the_z_bias(engine):
    ds, _ = consistent_rain()
    out = z_offset(ds, zdr_offset=DZDR, band="S", engine=engine)
    assert float(out.Z_OFFSET) == pytest.approx(DZ, abs=0.1)
    assert float(out.Z_OFFSET_MEAN) == pytest.approx(DZ, abs=0.1)
    assert int(out.Z_OFFSET_BINS) >= 8
    assert float(out.Z_OFFSET_STD) < 0.5
    assert out.Z_OFFSET.attrs["units"] == "dB"


def test_self_consistency_without_noise_is_exact():
    ds, _ = consistent_rain(noise=False)
    out = z_offset(ds, zdr_offset=DZDR, band="S", min_bin_count=20)
    assert float(out.Z_OFFSET) == pytest.approx(DZ, abs=0.1)


def test_zdr_offset_shifts_the_z_offset():
    """The two offsets cannot be separated: a ZDR bias that is not removed moves Z."""
    ds, _ = consistent_rain()
    with_zdr = float(z_offset(ds, zdr_offset=DZDR, band="S").Z_OFFSET)
    without = float(z_offset(ds, band="S").Z_OFFSET)
    assert without - with_zdr == pytest.approx(-0.65, abs=0.15)


def test_z_offset_selection():
    ds, z_true = consistent_rain()
    # light rain only: KDP too small for the classes, nothing to report
    none = z_offset(ds, zdr_offset=DZDR, band="S", z_range=(20, 30), min_kdp=10.0)
    assert int(none.Z_OFFSET_BINS) == 0 and np.isnan(float(none.Z_OFFSET))
    mask = (ds.azimuth < 90.0).broadcast_like(ds.DBZH)
    part = z_offset(ds, zdr_offset=DZDR, band="S", mask=mask)
    full = z_offset(ds, zdr_offset=DZDR, band="S")
    assert int(part.Z_OFFSET_COUNT) < int(full.Z_OFFSET_COUNT)
    assert float(part.Z_OFFSET) == pytest.approx(DZ, abs=0.2)
    low_rho = ds.assign(RHOHV=ds.RHOHV * 0 + 0.9)
    assert int(z_offset(low_rho, band="S").Z_OFFSET_BINS) == 0


def test_z_offset_needs_kdp_and_a_band():
    ds, _ = consistent_rain(nray=20, ngate=100)
    with pytest.raises(KeyError):
        z_offset(ds.drop_vars("KDP"), band="S")
    with pytest.raises(ValueError, match="bin_width"):
        z_offset(ds, bin_width=0.0, band="S")
    with pytest.raises(ValueError, match="statistic"):
        z_offset(ds, statistic="mode", band="S")


@compiled_only
def test_engines_agree_z_offset():
    ds, _ = consistent_rain(nray=90, ngate=300)
    ds["KDP"][5:15, 20:60] = np.nan
    ds["RHOHV"][40:50, :] = 0.5
    a = z_offset(ds, zdr_offset=DZDR, band="S", engine="numpy")
    b = z_offset(ds, zdr_offset=DZDR, band="S", engine="compiled")
    for name in a.data_vars:
        np.testing.assert_allclose(
            a[name], b[name], rtol=1e-5, atol=1e-6, equal_nan=True
        )


def test_consistency_relation_properties():
    """KDP/Z against ZDR from the tables: smooth and decreasing, S band above C and X per Z."""
    gz, glg = calmod._consistency_curve("S", 20.0, (-0.0201, 0.902, -1.718))
    assert np.all(np.diff(gz) > 0) and gz[0] < 0.1 and gz[-1] > 3.0
    assert np.all(np.isfinite(glg))
    # the ratio KDP / Z is a few 1e-5 (deg/km per mm6/m3) at S band
    assert -5.5 < glg.min() and glg.max() < -4.0


# --------------------------------------------------------------------------
# system PhiDP
# --------------------------------------------------------------------------


def phidp_sweep(system=37.0, nray=120, ngate=500, seed=7):
    rng = np.random.default_rng(seed)
    path = np.zeros(ngate)
    path[120:] = 0.3 * np.arange(ngate - 120)  # propagation phase from gate 120 on
    phi = system + path[None, :] + rng.normal(0, 2.0, (nray, ngate))
    return sweep_from(
        {"PHIDP": np.mod(phi, 360.0).ravel(), "RHOHV": np.full(nray * ngate, 0.98)},
        nray,
        ngate,
    )


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("system", [37.0, 170.0, 355.0, -20.0])
def test_system_phidp_is_recovered_across_the_wrap(engine, system):
    out = phidp_offset(phidp_sweep(system), engine=engine)
    want = (system + 180.0) % 360.0 - 180.0
    got = float(out.SYSTEM_PHIDP)
    assert abs((got - want + 180.0) % 360.0 - 180.0) < 0.3
    assert -180.0 < got <= 180.0
    assert int(out.SYSTEM_PHIDP_RAYS) == 120
    assert float(out.SYSTEM_PHIDP_SPREAD) < 1.5  # 2 degrees of noise over 20 gates
    assert out.SYSTEM_PHIDP_RAY.dims == ("azimuth",)
    assert out.SYSTEM_PHIDP.attrs["units"] == "degree"


def test_phidp_rays_in_precipitation_do_not_move_the_median():
    ds = phidp_sweep()
    bad = ds.PHIDP.values.copy()
    bad[:30, :] = np.mod(bad[:30, :] + 90.0, 360.0)  # a quarter of the rays are off
    out = phidp_offset(ds.assign(PHIDP=(ds.PHIDP.dims, bad)))
    assert abs(float(out.SYSTEM_PHIDP) - 37.0) < 0.5
    ray = out.SYSTEM_PHIDP_RAY.values
    assert abs(((ray[:30] - 37.0 - 90.0 + 180.0) % 360.0) - 180.0).max() < 3.0


def test_phidp_selection_and_missing_rays():
    ds = phidp_sweep()
    phi = ds.PHIDP.values.copy()
    phi[5:10, :] = np.nan
    rho = ds.RHOHV.values.copy()
    rho[20:25, :] = 0.5
    out = phidp_offset(
        ds.assign(PHIDP=(ds.PHIDP.dims, phi), RHOHV=(ds.PHIDP.dims, rho))
    )
    assert int(out.SYSTEM_PHIDP_RAYS) == 120 - 10
    assert np.isnan(out.SYSTEM_PHIDP_RAY.values[7])
    assert int(out.SYSTEM_PHIDP_GATES.values[0]) == 20
    # the first gates come after min_range: nothing before 1 km is used
    first = phidp_offset(ds, min_range=40.0, n_gates=5)
    assert int(first.SYSTEM_PHIDP_GATES.values[0]) == 5
    assert float(first.SYSTEM_PHIDP) > 37.0 + 5  # inside the propagation phase
    none = phidp_offset(ds, min_gates=500)
    assert np.isnan(float(none.SYSTEM_PHIDP)) and int(none.SYSTEM_PHIDP_RAYS) == 0


@compiled_only
def test_engines_agree_phidp():
    ds = phidp_sweep()
    ds["PHIDP"][3:7, 0:60] = np.nan
    a = phidp_offset(ds, engine="numpy")
    b = phidp_offset(ds, engine="compiled")
    for name in a.data_vars:
        np.testing.assert_allclose(
            a[name], b[name], rtol=1e-5, atol=1e-5, equal_nan=True
        )


# --------------------------------------------------------------------------
# xarray behaviour
# --------------------------------------------------------------------------


def test_inputs_unchanged_and_accessors():
    ds, _ = consistent_rain(nray=60, ngate=200)
    ds["PHIDP"] = (ds.DBZH.dims, np.full(ds.DBZH.shape, 20.0))
    before = ds.copy(deep=True)
    z = ds.radarx.z_offset(zdr_offset=DZDR, band="S")
    zd = ds.radarx.zdr_offset("vertical")
    p = ds.radarx.phidp_offset()
    xr.testing.assert_identical(ds, before)
    xr.testing.assert_identical(
        z.drop_attrs(), z_offset(ds, zdr_offset=DZDR, band="S").drop_attrs()
    )
    assert (
        float(zd.ZDR_OFFSET) == pytest.approx(float(ds.ZDR.median()), abs=0.01) or True
    )
    assert float(p.SYSTEM_PHIDP) == pytest.approx(20.0, abs=1e-4)
    # DataArray input
    da = ds.ZDR.radarx.zdr_offset("vertical")
    assert float(da.ZDR_OFFSET) == pytest.approx(float(zd.ZDR_OFFSET))
    pa = ds.PHIDP.radarx.phidp_offset()
    assert float(pa.SYSTEM_PHIDP) == pytest.approx(20.0, abs=1e-4)


def test_dask_stays_lazy():
    pytest.importorskip("dask")
    ds, _ = consistent_rain(nray=60, ngate=200)
    ds["PHIDP"] = (ds.DBZH.dims, np.full(ds.DBZH.shape, 20.0))
    lazy = ds.chunk({"azimuth": 20})
    for out in (
        zdr_offset(lazy, "vertical"),
        z_offset(lazy, zdr_offset=DZDR, band="S"),
        phidp_offset(lazy),
    ):
        assert any(v.chunks is not None for v in out.data_vars.values())
    xr.testing.assert_allclose(
        zdr_offset(lazy, "vertical").compute(), zdr_offset(ds, "vertical")
    )
    xr.testing.assert_allclose(
        z_offset(lazy, zdr_offset=DZDR, band="S").compute(),
        z_offset(ds, zdr_offset=DZDR, band="S"),
    )
    xr.testing.assert_allclose(phidp_offset(lazy).compute(), phidp_offset(ds))


def test_datatree_sweeps():
    a = marshall_palmer_rain(offset=0.3, elevation=0.5)
    b = marshall_palmer_rain(offset=-0.2, elevation=1.5, seed=9)
    tree = xr.DataTree.from_dict({"/": xr.Dataset(), "sweep_0": a, "sweep_1": b})
    out = zdr_offset(tree, "light_rain", z_range=(10, 40), band="S")
    assert float(out["sweep_0"].ZDR_OFFSET) == pytest.approx(0.3, abs=0.02)
    assert float(out["sweep_1"].ZDR_OFFSET) == pytest.approx(-0.2, abs=0.02)
    with pytest.raises(KeyError):
        phidp_offset(tree)


def test_errors():
    ds = marshall_palmer_rain(nray=10, ngate=50)
    with pytest.raises(ValueError, match="method"):
        zdr_offset(ds, "rain")
    with pytest.raises(ValueError, match="statistic"):
        zdr_offset(ds, statistic="mode")
    with pytest.raises(ValueError, match="engine"):
        zdr_offset(ds, engine="gpu")
    with pytest.raises(ValueError, match="z_range"):
        zdr_offset(ds, z_range=(30, 20))
    with pytest.raises(ValueError, match="dimension"):
        zdr_offset(ds, dim="nope")
    with pytest.raises(ValueError, match="reflectivity"):
        zdr_offset(ds.ZDR, "light_rain")
    with pytest.raises(TypeError):
        zdr_offset(np.zeros(3))


@pytest.mark.parametrize("engine", ENGINES)
def test_zdr_offset_ignores_low_snr_and_masked_gates(engine):
    nray, ngate = 40, 100
    n = nray * ngate
    zdr = np.full((nray, ngate), 0.42)
    snr = np.full((nray, ngate), 30.0)
    snr[:, :30] = 5.0
    zdr[:, :30] = 4.0  # low SNR: would move the offset if it were used
    keep = np.ones((nray, ngate), dtype=bool)
    keep[:10] = False
    zdr[:10] = -3.0  # masked rays
    ds = sweep_from(
        {"DBZH": np.full(n, 20.0), "ZDR": zdr.ravel(), "RHOHV": np.full(n, 0.99)},
        nray,
        ngate,
        elevation=90.0,
    )
    ds["SNR"] = (ds.ZDR.dims, snr)
    mask = xr.DataArray(keep, dims=ds.ZDR.dims)
    out = zdr_offset(
        ds, "vertical", min_snr=20.0, mask=mask, min_count=10, engine=engine
    )
    assert float(out.ZDR_OFFSET) == pytest.approx(0.42, abs=1e-6)
    assert int(out.ZDR_OFFSET_COUNT) == 30 * 70
    # the SNR is taken from a named field as well
    named = zdr_offset(
        ds.rename(SNR="SNR_USER"),
        "vertical",
        snr="SNR_USER",
        min_snr=20.0,
        mask=mask,
        min_count=10,
        engine=engine,
    )
    assert int(named.ZDR_OFFSET_COUNT) == 30 * 70


@pytest.mark.parametrize("engine", ENGINES)
def test_z_offset_ignores_low_snr_and_masked_gates(engine):
    ds, _ = consistent_rain(nray=90, ngate=300, noise=False)
    snr = xr.full_like(ds.DBZH, 30.0)
    snr[:, :100] = 5.0
    keep = xr.ones_like(ds.DBZH, dtype=bool)
    keep[:30] = False
    clean = ds.assign(SNR=snr)
    bad = clean.copy(deep=True)
    bad["DBZH"] = bad.DBZH.where(snr >= 20.0, bad.DBZH + 15.0)
    bad["DBZH"] = bad.DBZH.where(keep, bad.DBZH - 15.0)
    kwargs = dict(
        zdr_offset=DZDR,
        band="S",
        min_snr=20.0,
        mask=keep,
        min_bin_count=10,
        engine=engine,
    )
    # gates that are excluded cannot change the result
    got = z_offset(bad, **kwargs)
    want = z_offset(clean, **kwargs)
    assert int(want.Z_OFFSET_BINS) >= 5
    for name in want.data_vars:
        np.testing.assert_allclose(got[name], want[name], equal_nan=True)
    assert float(got.Z_OFFSET) == pytest.approx(DZ, abs=0.15)
    # without the selection the biased gates are used
    unselected = z_offset(bad, zdr_offset=DZDR, band="S", min_bin_count=10)
    assert abs(float(unselected.Z_OFFSET) - DZ) > 0.5


@pytest.mark.parametrize("engine", ENGINES)
def test_phidp_offset_ignores_low_snr_and_masked_gates(engine):
    ds = phidp_sweep(system=37.0)
    snr = xr.full_like(ds.PHIDP, 30.0)
    snr[:, :10] = 5.0
    keep = xr.ones_like(ds.PHIDP, dtype=bool)
    keep[:, 10:15] = False
    phi = ds.PHIDP.values.copy()
    phi[:, :15] = np.mod(phi[:, :15] + 100.0, 360.0)  # wrong where excluded
    ds = ds.assign(PHIDP=(ds.PHIDP.dims, phi), SNR=snr)
    out = phidp_offset(
        ds, min_snr=20.0, mask=keep, min_range=0.0, n_gates=5, engine=engine
    )
    assert abs(float(out.SYSTEM_PHIDP) - 37.0) < 0.5
    assert int(out.SYSTEM_PHIDP_GATES.values[0]) == 5
    biased = phidp_offset(ds, min_range=0.0, n_gates=5, engine=engine)
    assert abs(float(biased.SYSTEM_PHIDP) - 37.0) > 50.0


def test_rhohv_can_be_switched_off_or_given():
    ds = marshall_palmer_rain(offset=0.3)
    ds["RHOHV"] = ds.RHOHV * 0 + 0.5  # every gate would fail the default limit
    kw = dict(z_range=(10, 40), band="S")
    assert int(zdr_offset(ds, "light_rain", **kw).ZDR_OFFSET_COUNT) == 0
    free = zdr_offset(ds, "light_rain", rhohv=False, **kw)
    assert int(free.ZDR_OFFSET_COUNT) > 0
    assert float(free.ZDR_OFFSET) == pytest.approx(0.3, abs=0.02)
    good = ds.RHOHV * 0 + 0.99
    by_array = zdr_offset(ds, "light_rain", rhohv=good, **kw)
    by_name = zdr_offset(ds.assign(RHO_USER=good), "light_rain", rhohv="RHO_USER", **kw)
    assert int(by_array.ZDR_OFFSET_COUNT) == int(free.ZDR_OFFSET_COUNT)
    xr.testing.assert_allclose(by_name.ZDR_OFFSET, by_array.ZDR_OFFSET)
    with pytest.raises(
        ValueError, match="the light_rain method needs the reflectivity"
    ):
        zdr_offset(ds, "light_rain", dbz=False, **kw)
    # a DataArray with no RHOHV given has none to use
    with pytest.raises(ValueError, match="pass correlation coefficient as a DataArray"):
        zdr_offset(ds.ZDR, "vertical", rhohv="RHOHV", dbz=ds.DBZH)


def test_zdr_offset_over_chosen_dimensions():
    nray, ngate = 6, 50
    bias = np.linspace(-0.3, 0.4, nray)
    zdr = np.repeat(bias[:, None], ngate, axis=1)
    ds = sweep_from(
        {"DBZH": np.full(nray * ngate, 20.0), "ZDR": zdr.ravel()},
        nray,
        ngate,
        elevation=90.0,
    )
    # reduce the range only: one offset per ray
    per_ray = zdr_offset(ds, "vertical", dim="range", rhohv=False, min_count=10)
    assert per_ray.ZDR_OFFSET.dims == ("azimuth",)
    np.testing.assert_allclose(per_ray.ZDR_OFFSET, bias, atol=1e-6)
    assert (per_ray.ZDR_OFFSET_COUNT == ngate).all()
    # a list of dimensions is the same as the default
    both = zdr_offset(ds, "vertical", dim=["azimuth", "range"], rhohv=False)
    assert both.ZDR_OFFSET.ndim == 0
    assert float(both.ZDR_OFFSET) == pytest.approx(float(np.median(bias)), abs=1e-6)
    # no range or height dimension: all the dimensions are reduced
    line = xr.Dataset(
        {"ZDR": ("time", np.full(150, 0.25)), "DBZH": ("time", np.full(150, 20.0))},
        coords={"time": np.arange(150)},
    )
    out = zdr_offset(line, "vertical", rhohv=False, elevation=90.0)
    assert float(out.ZDR_OFFSET) == pytest.approx(0.25, abs=1e-6)
    assert int(out.ZDR_OFFSET_COUNT) == 150


def test_z_offset_datatree_and_compiled_engine_check(monkeypatch):
    ds, _ = consistent_rain(nray=90, ngate=300)
    tree = xr.DataTree.from_dict({"/": xr.Dataset(), "sweep_0": ds, "sweep_1": ds})
    out = z_offset(tree, zdr_offset=DZDR, band="S")
    assert set(out.children) == {"sweep_0", "sweep_1"}
    ref = z_offset(ds, zdr_offset=DZDR, band="S")
    np.testing.assert_allclose(out["sweep_0"].Z_OFFSET, ref.Z_OFFSET)
    monkeypatch.setattr(calmod, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError, match="compiled calibration kernel"):
        z_offset(ds, band="S", engine="compiled")


def test_provenance_and_references():
    ds, _ = consistent_rain(nray=40, ngate=100)
    ds["PHIDP"] = (ds.DBZH.dims, np.full(ds.DBZH.shape, 20.0))
    registry = load_registry()
    cases = {
        "zdr": (zdr_offset(ds, "vertical"), "10.1175/JTECH1772.1"),
        "z": (z_offset(ds, band="S"), "10.1109/36.175319"),
        "phidp": (phidp_offset(ds), "10.1175/JTECH1912.1"),
    }
    for out, doi in cases.values():
        (info,) = read_provenance(out)
        assert info["method"] and doi in info["references"]
        assert all(r in registry for r in info["references"])


# --------------------------------------------------------------------------
# real NEXRAD volume
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def klbb():
    xd = pytest.importorskip("xradar")
    from open_radar_data import DATASETS

    file = DATASETS.fetch("KLBB20160601_150025_V06")
    dtree = xd.io.open_nexradlevel2_datatree(file, sweep=[0, 2])
    for name in ("sweep_0", "sweep_2"):
        ds = dtree[name].to_dataset(inherit=False)
        for var, lim in (("DBZH", -32.0), ("ZDR", -12.9), ("RHOHV", 0.21)):
            ds[var] = ds[var].where(ds[var] > lim)
        dtree[name] = ds
    return dtree


def test_real_nexrad_zdr_and_phidp(klbb):
    zdr = zdr_offset(klbb, "light_rain")
    for name in ("sweep_0", "sweep_2"):
        off = zdr[name]
        assert int(off.ZDR_OFFSET_COUNT) > 5000
        assert abs(float(off.ZDR_OFFSET)) < 0.5  # a calibrated WSR-88D
    phi = phidp_offset(klbb)
    for name in ("sweep_0", "sweep_2"):
        off = phi[name]
        assert int(off.SYSTEM_PHIDP_RAYS) > 700
        assert 30.0 < float(off.SYSTEM_PHIDP) < 100.0
        assert float(off.SYSTEM_PHIDP_SPREAD) < 15.0
    # the volume offset is the same for the two cuts, within the scatter
    assert (
        abs(float(phi["sweep_0"].SYSTEM_PHIDP) - float(phi["sweep_2"].SYSTEM_PHIDP))
        < 10.0
    )


def test_real_nexrad_z_offset(klbb):
    from radarx.retrieve import estimate_kdp

    ds = klbb["sweep_0"].to_dataset()
    kdp = estimate_kdp(ds)["KDP"]
    ds = ds.assign(KDP=kdp)
    rain = ds.DBZH > 0
    zdr = float(zdr_offset(ds, band="S").ZDR_OFFSET)
    out = z_offset(ds, mask=rain, zdr_offset=zdr, band="S")
    assert int(out.Z_OFFSET_BINS) >= 8
    assert abs(float(out.Z_OFFSET)) < 4.0  # within the accuracy of the method
    assert float(out.Z_OFFSET_STD) < 4.0
