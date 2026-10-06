#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for radarx quasi-vertical profiles
========================================

Synthetic sweeps have fields whose azimuthal mean is known exactly (a
harmonic in azimuth averages to zero over evenly spaced rays). The compiled
kernel and the NumPy implementation must agree to float32 rounding (the
input and output type): rtol 1e-6 for means in native units, which only
differ in summation order, and 2e-5 dB for means in linear units, where the
kernel converts dB to linear in single precision. Medians, the melting-layer
anchors and flags agree exactly, and melting-layer heights to 1e-9 m (the
onset heights are interpolated between gates, where the compiler may fuse a
multiply and an add).
"""
import numpy as np
import pytest
import xarray as xr
from xradar.georeference import antenna_to_cartesian

import radarx  # noqa: F401
from radarx.retrieve import melting_layer, qvp, qvp_timeseries
from radarx.retrieve import vertical_profiles as vp

ALT = 250.0
ENGINES = ["numpy"] + (["compiled"] if vp.HAS_COMPILED_KERNEL else [])
RNG = np.arange(125.0, 30_000.0, 250.0)


def _sweep(elevation=15.0, nray=360, seed=0, time="2026-01-01T00:00"):
    """Sweep whose azimuthal means are a(r) (native) and 10 log10 L(r) (dB)."""
    az = (np.arange(nray) + 0.5) * 360.0 / nray
    harmonic = np.cos(np.deg2rad(az))[:, None] * np.ones(RNG.size)
    a = 0.01 * RNG / 250.0  # native-unit mean
    lin = 100.0 * (1.0 + RNG / 30_000.0)  # linear-unit mean of DBZH
    dbz = 10.0 * np.log10(lin * (1.0 + 0.5 * harmonic))
    zdr_lin = 1.5 + 0.0 * RNG
    zdr = 10.0 * np.log10(zdr_lin * (1.0 + 0.3 * harmonic))
    rho = 0.98 + 0.01 * harmonic
    phi = a + 2.0 * harmonic
    el = elevation + 0.02 * np.sin(np.deg2rad(az))
    t = np.datetime64(time, "ns") + (np.arange(nray) * 50).astype("timedelta64[ms]")
    dims = ("azimuth", "range")
    ds = xr.Dataset(
        {
            "DBZH": (dims, dbz.astype(np.float32), {"units": "dBZ"}),
            "ZDR": (dims, zdr.astype(np.float32), {"units": "dB"}),
            "RHOHV": (dims, rho.astype(np.float32), {"units": "1"}),
            "PHIDP": (dims, phi.astype(np.float32), {"units": "degrees"}),
        },
        coords={
            "azimuth": az,
            "range": RNG,
            "elevation": ("azimuth", el),
            "time": ("azimuth", t),
            "latitude": 45.0,
            "longitude": 10.0,
            "altitude": ALT,
        },
    )
    return ds, a, 10.0 * np.log10(lin), 10.0 * np.log10(zdr_lin)


def _volume(elevations=(0.5, 5.0, 15.0), **kwargs):
    sweeps = {
        f"sweep_{k}": _sweep(el, **kwargs)[0].drop_vars(
            ["latitude", "longitude", "altitude"]
        )
        for k, el in enumerate(elevations)
    }
    root = xr.Dataset(coords={"latitude": 45.0, "longitude": 10.0, "altitude": ALT})
    return xr.DataTree.from_dict({"/": root, **sweeps})


@pytest.mark.parametrize("engine", ENGINES)
def test_mean_matches_analytic(engine):
    ds, a, dbz, zdr = _sweep()
    out = qvp(ds, engine=engine)
    # float32 inputs: agreement to float32 resolution
    np.testing.assert_allclose(out["PHIDP"], a, atol=2e-5)
    np.testing.assert_allclose(out["DBZH"], dbz, atol=1e-4)  # linear mean
    np.testing.assert_allclose(out["ZDR"], zdr, atol=1e-4)  # linear mean
    np.testing.assert_allclose(out["RHOHV"], 0.98, atol=1e-6)
    assert out["DBZH"].attrs["cell_methods"] == "azimuth: mean"
    assert out["DBZH"].attrs["units"] == "dBZ"
    assert out["DBZH"].dims == ("height",)


@pytest.mark.parametrize("engine", ENGINES)
def test_db_mean_differs_from_native_mean(engine):
    ds, *_ = _sweep()
    lin = qvp(ds, "DBZH", engine=engine)["DBZH"]
    native = qvp(ds, "DBZH", linear=[], engine=engine)["DBZH"]
    np.testing.assert_allclose(native, ds["DBZH"].mean("azimuth"), atol=1e-4)
    assert float((lin - native).min()) > 0.1  # Jensen: power mean > dB mean


@pytest.mark.parametrize("engine", ENGINES)
def test_height_from_beam_geometry(engine):
    ds, *_ = _sweep(elevation=12.0)
    out = qvp(ds, engine=engine)
    el = float(np.median(ds["elevation"]))
    x, y, z = antenna_to_cartesian(RNG, 0.0, el, site_altitude=ALT)
    np.testing.assert_allclose(out["height"], z)
    np.testing.assert_allclose(out["ground_range"], np.hypot(x, y))
    np.testing.assert_allclose(out["range"], RNG)
    assert float(out["elevation"]) == pytest.approx(el)
    assert out["time"].dtype.kind == "M"
    assert float(out["altitude"]) == ALT


@pytest.mark.parametrize("engine", ENGINES)
def test_quality_filter_excludes_gates(engine):
    ds, *_ = _sweep()
    bad = np.zeros(ds["DBZH"].shape, dtype=bool)
    bad[::4] = True  # every 4th ray is garbage with low rhohv
    for name in ("PHIDP", "DBZH", "ZDR"):
        ds[name] = ds[name].where(~bad, 999.0)
    ds["RHOHV"] = ds["RHOHV"].where(~bad, 0.3)
    out = qvp(ds, engine=engine, counts=True)
    # remaining rays are not evenly spaced, so compare to the masked mean
    good = ~bad[:, 0]
    np.testing.assert_allclose(
        out["PHIDP"], ds["PHIDP"].isel(azimuth=good).mean("azimuth"), atol=1e-4
    )
    assert int(out["PHIDP_count"][0]) == good.sum()
    # without the rhohv test the garbage is averaged in
    assert float(qvp(ds, "PHIDP", min_rhohv=None, engine=engine)["PHIDP"][0]) > 100


@pytest.mark.parametrize("engine", ENGINES)
def test_reflectivity_threshold(engine):
    ds, *_ = _sweep()
    ds["DBZH"] = ds["DBZH"].where(ds["azimuth"] < 180, -20.0)
    out = qvp(ds, ["PHIDP", "DBZH"], counts=True, engine=engine)
    assert int(out["PHIDP_count"][0]) == 180
    assert int(qvp(ds, "PHIDP", min_dbz=None, counts=True)["PHIDP_count"][0]) == 360


@pytest.mark.parametrize("engine", ENGINES)
def test_min_count_and_fraction(engine):
    ds, *_ = _sweep()
    keep = np.zeros(360, dtype=bool)
    keep[:30] = True
    ds["PHIDP"] = ds["PHIDP"].where(xr.DataArray(keep, dims="azimuth"))
    assert np.isfinite(qvp(ds, "PHIDP", engine=engine)["PHIDP"]).all()
    assert np.isnan(qvp(ds, "PHIDP", min_count=31, engine=engine)["PHIDP"]).all()
    assert np.isnan(qvp(ds, "PHIDP", min_fraction=0.1, engine=engine)["PHIDP"]).all()


@pytest.mark.parametrize("engine", ENGINES)
def test_median(engine):
    ds, *_ = _sweep()
    out = qvp(ds, reduction="median", engine=engine)
    np.testing.assert_allclose(
        out["DBZH"], ds["DBZH"].median("azimuth"), rtol=0, atol=1e-6
    )
    assert out["DBZH"].attrs["cell_methods"] == "azimuth: median"


def test_engines_agree_random():
    if not vp.HAS_COMPILED_KERNEL:
        pytest.skip("compiled kernel not built")
    rnd = np.random.default_rng(1)
    ds, *_ = _sweep(nray=361)
    for name in ("DBZH", "ZDR", "PHIDP", "RHOHV"):
        x = ds[name].values + rnd.normal(0, 1, ds[name].shape).astype(np.float32)
        x[rnd.random(x.shape) < 0.3] = np.nan
        ds[name] = (ds[name].dims, x, ds[name].attrs)
    ds["RHOHV"] = ds["RHOHV"] * 0.5 + 0.4
    for reduction in ("mean", "median"):
        kw = {"reduction": reduction, "min_count": 5, "counts": True}
        c = qvp(ds, engine="compiled", **kw)
        n = qvp(ds, engine="numpy", **kw)
        for name in c.data_vars:
            np.testing.assert_allclose(
                c[name], n[name], rtol=1e-6, atol=2e-5, equal_nan=True
            )
    for threads in (1, 3):
        t = qvp(ds, engine="compiled", n_threads=threads)
        xr.testing.assert_identical(t, qvp(ds, engine="compiled"))


@pytest.mark.parametrize("engine", ENGINES)
def test_volume_sweep_selection_and_accessor(engine):
    dt = _volume()
    out = dt.radarx.qvp(engine=engine)
    assert float(out["elevation"]) == pytest.approx(15.0, abs=0.05)
    assert float(qvp(dt, elevation=4.0)["elevation"]) == pytest.approx(5.0, abs=0.05)
    assert float(qvp(dt, sweep=0)["elevation"]) == pytest.approx(0.5, abs=0.05)
    assert float(qvp(dt, sweep="sweep_1")["elevation"]) == pytest.approx(5.0, abs=0.05)
    assert "latitude" in out.coords
    ds, *_ = _sweep()
    xr.testing.assert_identical(ds.radarx.qvp(engine=engine), qvp(ds, engine=engine))


@pytest.mark.parametrize("engine", ENGINES)
def test_timeseries(engine):
    vols = [
        _volume(elevations=(0.5, el), time=f"2026-01-01T00:{m:02d}")
        for m, el in ((0, 15.0), (5, 15.0), (10, 14.0))
    ]
    ts = qvp_timeseries(vols, ["DBZH", "PHIDP"], engine=engine)
    assert ts["DBZH"].dims == ("time", "height")
    assert ts.sizes["time"] == 3
    single = [qvp(v, ["DBZH", "PHIDP"], engine=engine) for v in vols]
    np.testing.assert_allclose(ts["DBZH"][0], single[0]["DBZH"])
    np.testing.assert_allclose(ts["height"], single[0]["height"])
    # the 14 deg sweep is interpolated onto the 15 deg heights
    expect = np.interp(
        ts["height"], single[2]["height"], single[2]["PHIDP"], right=np.nan
    )
    np.testing.assert_allclose(ts["PHIDP"][2], expect, rtol=1e-6)
    np.testing.assert_allclose(ts["elevation"], [15.0, 15.0, 14.0], atol=0.05)
    assert np.all(np.diff(ts["time"].values) > np.timedelta64(0, "s"))
    heights = np.arange(500.0, 5000.0, 100.0)
    ts2 = qvp_timeseries(vols, "PHIDP", heights=heights, engine=engine)
    np.testing.assert_allclose(ts2["height"], heights)


def test_interp_keeps_gaps():
    y = np.array([0.0, 1.0, np.nan, 3.0, 4.0])
    out = vp._interp_nan(np.array([0.5, 1.5, 2.5, 3.5, 5.0]), np.arange(5.0), y)
    np.testing.assert_allclose(out, [0.5, np.nan, np.nan, 3.5, np.nan])


def test_errors():
    ds, *_ = _sweep()
    with pytest.raises(ValueError, match="engine"):
        qvp(ds, engine="fortran")
    with pytest.raises(ValueError, match="reduction"):
        qvp(ds, reduction="mode")
    with pytest.raises(ValueError, match="not found"):
        qvp(ds, "VRADH")
    with pytest.raises(ValueError, match="not found"):
        qvp(ds, rhohv="RHO")
    with pytest.raises(TypeError):
        qvp(ds["DBZH"])
    with pytest.raises(ValueError, match="height"):
        melting_layer(ds)


def test_explicit_fields_and_input_checks(monkeypatch):
    ds, *_ = _sweep()
    xr.testing.assert_allclose(qvp(ds, rhohv="RHOHV", dbz="DBZH"), qvp(ds))
    nat = np.full(ds.sizes["azimuth"], np.datetime64("NaT"), "datetime64[ns]")
    out = qvp(ds.assign_coords(time=("azimuth", nat)))
    assert np.isnat(out["time"].values)
    other = ds["DBZH"].rename(azimuth="ray").drop_vars(["elevation", "time"])
    with pytest.raises(ValueError, match="same rays"):
        qvp(ds.assign(OTHER=other), ["DBZH", "OTHER"])
    monkeypatch.setattr(vp, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError, match="compiled QVP kernel"):
        qvp(ds, engine="compiled")


# ---------------------------------------------------------------------------
# melting layer
# ---------------------------------------------------------------------------


def _ramp(h, lo, hi, width=100.0):
    """1 inside [lo, hi], 0 outside, with linear ramps of ``width`` centred
    on the boundaries (so the half-amplitude points are exactly lo and hi)."""
    up = np.clip((h - lo) / width + 0.5, 0.0, 1.0)
    down = np.clip((hi - h) / width + 0.5, 0.0, 1.0)
    return np.minimum(up, down)


def _melting_profile(bottom=2500.0, top=3100.0, spurious=True):
    """
    QVP of stratiform rain with a melting layer between ``bottom`` and ``top``.

    rhohv drops from 0.99 to 0.93 and ZDR rises from 0.4 to 1.2 dB (peak
    1.4 dB a third of the way up) inside the layer; Z has a bright band. With
    ``spurious``, a deep rhohv dip without ZDR or Z enhancement near the echo
    top (5 km) and a ZDR spike in rain without a rhohv dip (1.5 km) are added;
    neither is a melting layer.
    """
    h = np.arange(300.0, 7000.0, 50.0)
    layer = _ramp(h, bottom, top)
    peak = np.exp(-0.5 * ((h - (bottom + (top - bottom) / 3)) / 60.0) ** 2)
    rho = 0.99 - 0.06 * layer
    zdr = 0.4 + 0.8 * layer + 0.2 * peak
    z = np.where(h < bottom, 30.0, 25.0 - 3.0 * (h - top) / 1000) + 8.0 * layer
    z[h > 5600] = np.nan  # echo top
    if spurious:
        rho = rho - 0.12 * _ramp(h, 4900.0, 5100.0)
        zdr = zdr + 1.6 * _ramp(h, 1450.0, 1550.0)
    return h, z, rho, zdr


def _profiles_dataset(profiles, time=None):
    h = profiles[0][0]
    data = {
        name: (("time", "height"), np.stack([p[k] for p in profiles]))
        for k, name in ((1, "DBZH"), (2, "RHOHV"), (3, "ZDR"))
    }
    if time is None:
        time = np.datetime64("2026-01-01T00:00", "ns") + np.arange(
            len(profiles)
        ) * np.timedelta64(5, "m")
    return xr.Dataset(data, coords={"height": h, "time": time})


@pytest.mark.parametrize("engine", ENGINES)
def test_melting_layer_known_layer(engine):
    prof = _profiles_dataset([_melting_profile()] * 3)
    ml = melting_layer(prof, engine=engine)
    assert ml["melting_layer_top"].dims == ("time",)
    assert ml["melting_layer_top"].attrs["units"] == "m"
    np.testing.assert_allclose(ml["melting_layer_top"], 3100.0, atol=100)
    np.testing.assert_allclose(ml["melting_layer_bottom"], 2500.0, atol=100)
    np.testing.assert_allclose(ml["melting_layer_peak"], 2700.0, atol=50)
    np.testing.assert_array_equal(ml["melting_layer_flag"], 1)
    # a single profile without time works too
    one = melting_layer(prof.isel(time=0), engine=engine)
    assert one["melting_layer_top"].dims == ()
    assert float(one["melting_layer_top"]) == pytest.approx(3100.0, abs=100)


@pytest.mark.parametrize("engine", ENGINES)
def test_melting_layer_rejects_spurious_signatures(engine):
    # only the spurious rhohv dip and ZDR spike: nothing is detected
    h, z, rho, zdr = _melting_profile(spurious=True)
    clean = _melting_profile(spurious=False)
    rho_only = rho - clean[2] + 0.99  # remove the melting layer from rhohv
    zdr_only = zdr - clean[3] + 0.4  # and from ZDR
    z_flat = np.where(np.isnan(z), np.nan, 25.0)
    prof = _profiles_dataset([(h, z_flat, rho_only, zdr_only)])
    ml = melting_layer(prof, engine=engine)
    assert np.isnan(ml["melting_layer_top"]).all()
    np.testing.assert_array_equal(ml["melting_layer_flag"], 0)
    # uniform rain
    flat = (h, z_flat, np.full(h.size, 0.99), np.full(h.size, 0.5))
    assert np.isnan(melting_layer(_profiles_dataset([flat]))["melting_layer_top"]).all()


@pytest.mark.parametrize("engine", ENGINES)
def test_melting_layer_search_range(engine):
    # a second, stronger layer-like signature at 4.5-5 km
    h, z, rho, zdr = _melting_profile(spurious=False)
    upper = _ramp(h, 4500.0, 5000.0)
    rho2, zdr2, z2 = rho - 0.05 * upper, zdr + 1.5 * upper, z + 10 * upper
    prof = _profiles_dataset([(h, z2, rho2, zdr2)])
    ml = melting_layer(prof, engine=engine)
    assert float(ml["melting_layer_top"][0]) == pytest.approx(5000.0, abs=100)
    # a freezing level selects the right one
    ml = melting_layer(prof, freezing_level=3300.0, engine=engine)
    assert float(ml["melting_layer_top"][0]) == pytest.approx(3100.0, abs=100)
    fl = xr.DataArray([3300.0], dims="time")
    ml2 = melting_layer(prof, freezing_level=fl, engine=engine)
    xr.testing.assert_identical(ml, ml2)
    # or the height range
    ml = melting_layer(prof, height_range=(1000.0, 4000.0), engine=engine)
    assert float(ml["melting_layer_top"][0]) == pytest.approx(3100.0, abs=100)


def _gaussian_layer(centre=3000.0, sigma=200.0, background=0.99, dip=0.06):
    """Melting layer whose rhohv dip, ZDR and Z peaks are Gaussians of width
    ``sigma`` centred at ``centre`` (known onset and half-prominence heights)."""
    h = np.arange(300.0, 7000.0, 20.0)
    g = np.exp(-0.5 * ((h - centre) / sigma) ** 2)
    rho = background - dip * g
    zdr = 0.3 + 1.2 * g
    z = 25.0 + 10.0 * g
    z[h > 6000] = np.nan
    return h, z, rho, zdr


@pytest.mark.parametrize("engine", ENGINES)
def test_melting_layer_onset_known_heights(engine):
    # ramps: rhohv = 0.99 - 0.06 * layer is within 10 % of the dip from the
    # background (0.984) where layer = 0.1, i.e. 0.4 ramp widths outside the
    # nominal boundaries
    prof = _profiles_dataset([_melting_profile()])
    ml = melting_layer(prof, engine=engine)
    assert ml.attrs["melting_layer_boundaries"] == "onset"
    assert "Griffin" in ml["melting_layer_top"].attrs["method"]
    assert float(ml["melting_layer_top"][0]) == pytest.approx(3140.0, abs=1)
    assert float(ml["melting_layer_bottom"][0]) == pytest.approx(2460.0, abs=1)
    # Gaussian dip of rhohv, ZDR and Z: the onset (10 % of the dip, exp =
    # 0.1) lies at sqrt(2 ln 10) sigma from the centre, the half prominence
    # at sqrt(2 ln 2) sigma, and rhohv = 0.97 (exp = 1/3) at sqrt(2 ln 3)
    sigma, centre = 200.0, 3000.0
    prof = _profiles_dataset([_gaussian_layer(centre, sigma)])
    onset = melting_layer(prof, engine=engine)
    half = melting_layer(prof, boundaries="half_prominence", engine=engine)
    griffin = melting_layer(prof, rhohv_onset=0.97, engine=engine)
    d_onset = sigma * np.sqrt(2 * np.log(10))
    d_half = sigma * np.sqrt(2 * np.log(2))
    d_griffin = sigma * np.sqrt(2 * np.log(3))
    for ml, d in ((onset, d_onset), (griffin, d_griffin)):
        assert float(ml["melting_layer_top"][0]) == pytest.approx(centre + d, abs=2)
        assert float(ml["melting_layer_bottom"][0]) == pytest.approx(centre - d, abs=2)
    assert "or reaches 0.97" in griffin["melting_layer_top"].attrs["method"]
    # half prominence: last gate above half height (20 m gates)
    assert float(half["melting_layer_top"][0]) == pytest.approx(centre + d_half, abs=20)
    assert float(half["melting_layer_bottom"][0]) == pytest.approx(
        centre - d_half, abs=20
    )
    assert "half the prominence" in half["melting_layer_top"].attrs["method"]
    # a lower background (0.96, dip to 0.90): the relative onset is unchanged,
    # and a fixed rhohv_onset above the background falls back to it
    prof = _profiles_dataset([_gaussian_layer(centre, sigma, background=0.96)])
    for kwargs in ({}, {"rhohv_onset": 0.97}):
        low = melting_layer(prof, engine=engine, **kwargs)
        top = float(low["melting_layer_top"][0])
        assert top == pytest.approx(centre + d_onset, abs=2)
    # onset_fraction is honoured: 50 % of the dip is the half-prominence
    prof = _profiles_dataset([_gaussian_layer(centre, sigma)])
    custom = melting_layer(prof, onset_fraction=0.5, engine=engine)
    assert float(custom["melting_layer_top"][0]) == pytest.approx(
        centre + d_half, abs=2
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_melting_layer_onset_falls_back_to_zdr(engine):
    # rhohv stays at its minimum above the layer: no onset above, so the top
    # is the ZDR half-prominence edge; the bottom is still the rhohv onset
    h, z, rho, zdr = _gaussian_layer()
    rho = np.where(h > 3000.0, rho.min(), rho)
    ml = melting_layer(_profiles_dataset([(h, z, rho, zdr)]), engine=engine)
    half = melting_layer(
        _profiles_dataset([_gaussian_layer()]),
        boundaries="half_prominence",
        engine=engine,
    )
    assert float(ml["melting_layer_top"][0]) == float(half["melting_layer_top"][0])
    d_onset = 200.0 * np.sqrt(2 * np.log(10))
    assert float(ml["melting_layer_bottom"][0]) == pytest.approx(3000 - d_onset, abs=2)


def _environment(freezing_level=3300.0, time=None):
    """Sounding with a 6.5 K/km lapse rate and its 0 degC level at
    ``freezing_level``; dewpoint 2 K below the temperature."""
    height = np.arange(0.0, 12000.0, 100.0)
    temperature = 273.15 - 0.0065 * (height - freezing_level)
    pressure = 101325.0 * np.exp(-height / 8000.0)
    ds = xr.Dataset(
        {
            "temperature": ("height", temperature, {"units": "K"}),
            "dewpoint": ("height", temperature - 2.0, {"units": "K"}),
            "pressure": ("height", pressure, {"units": "Pa"}),
        },
        coords={"height": height},
    )
    if time is not None:
        ds = ds.assign_coords(time=np.datetime64(time, "ns"))
    return ds


@pytest.mark.parametrize("engine", ENGINES)
def test_melting_layer_environment(engine):
    from radarx.io.sounding import wet_bulb_zero_height

    # (the wet-bulb kernels of the two engines agree to about 1e-5 m)

    # a stronger layer-like signature at 4.5-5 km above the real one
    h, z, rho, zdr = _melting_profile(spurious=False)
    upper = _ramp(h, 4500.0, 5000.0)
    rho2, zdr2, z2 = rho - 0.05 * upper, zdr + 1.5 * upper, z + 10 * upper
    prof = _profiles_dataset([(h, z2, rho2, zdr2)] * 2)
    env = _environment(3300.0, "2026-01-01T00:00")
    wbz = float(wet_bulb_zero_height(env))
    assert 2800.0 < wbz < 3300.0
    ml = melting_layer(prof, environment=env, engine=engine)
    top = ml["melting_layer_top"].values
    np.testing.assert_allclose(top, 3140.0, atol=1)
    np.testing.assert_allclose(ml["freezing_level"], 3300.0, atol=1e-6)
    np.testing.assert_allclose(ml["wet_bulb_zero_height"], wbz, atol=1e-3)
    np.testing.assert_allclose(
        ml["melting_layer_top_offset_freezing_level"], top - 3300
    )
    np.testing.assert_allclose(
        ml["melting_layer_top_offset_wet_bulb_zero"], top - wbz, atol=1e-3
    )
    assert ml["freezing_level"].dims == ("time",)
    assert ml["wet_bulb_zero_height"].attrs["units"] == "m"
    # the same from a mapping of reference heights
    same = melting_layer(
        prof,
        environment={"freezing_level": 3300.0, "wet_bulb_zero_height": wbz},
        engine=engine,
    )
    xr.testing.assert_allclose(ml, same)
    # temperature only: 0 degC height, no wet-bulb
    dry = melting_layer(
        prof, environment=env.drop_vars(["dewpoint", "pressure"]), engine=engine
    )
    assert "wet_bulb_zero_height" not in dry
    np.testing.assert_allclose(dry["melting_layer_top"], top)
    # an explicit freezing_level overrides the search range, not the output
    high = melting_layer(prof, environment=env, freezing_level=5000.0, engine=engine)
    np.testing.assert_allclose(high["melting_layer_top"], 5040.0, atol=1)
    np.testing.assert_allclose(high["freezing_level"], 3300.0, atol=1e-6)
    # a given freezing_level alone is reported as the reference
    ref = melting_layer(prof, freezing_level=3300.0, engine=engine)
    assert ref["freezing_level"].attrs["long_name"].startswith("reference")
    assert "wet_bulb_zero_height" not in ref
    # several soundings on time are interpolated to the profile times
    envs = xr.concat(
        [
            _environment(3300.0, "2025-12-31T23:00"),
            _environment(3500.0, "2026-01-01T00:10"),
        ],
        dim="time",
    )
    ts = melting_layer(prof, environment=envs, engine=engine)
    np.testing.assert_allclose(
        ts["freezing_level"], [3300 + 200 * 6 / 7, 3300 + 200 * 13 / 14]
    )
    # a missing reference height falls back to height_range
    nan = melting_layer(prof, environment={"freezing_level": np.nan}, engine=engine)
    np.testing.assert_allclose(nan["melting_layer_top"], 5040.0, atol=1)
    assert np.isnan(nan["melting_layer_top_offset_freezing_level"]).all()
    # environments for a single profile
    one = melting_layer(prof.isel(time=0), environment=envs, engine=engine)
    assert one["freezing_level"].dims == ()
    assert float(one["freezing_level"]) == pytest.approx(3300 + 200 * 6 / 7)
    single = melting_layer(prof, environment=env.expand_dims("time"), engine=engine)
    xr.testing.assert_allclose(single, ml)
    unindexed = xr.DataArray([3300.0, 3400.0], dims="time")
    pos = melting_layer(prof, freezing_level=unindexed, engine=engine)
    np.testing.assert_allclose(pos["freezing_level"], [3300.0, 3400.0])
    with pytest.raises(ValueError, match="several times"):
        melting_layer(
            prof.isel(time=0).drop_vars("time"), freezing_level=envs.temperature[:, 0]
        )


def test_melting_layer_environment_errors():
    prof = _profiles_dataset([_melting_profile()])
    with pytest.raises(ValueError, match="boundaries"):
        melting_layer(prof, boundaries="peak")
    with pytest.raises(TypeError, match="environment"):
        melting_layer(prof, environment=3300.0)
    with pytest.raises(ValueError, match="needs 'freezing_level'"):
        melting_layer(prof, environment={"height": 1.0})
    with pytest.raises(ValueError, match="temperature"):
        melting_layer(prof, environment=_environment().drop_vars("temperature"))
    if vp.HAS_COMPILED_KERNEL:
        h = prof["height"].values
        a = np.zeros((1, h.size))
        with pytest.raises(ValueError, match="boundary"):
            vp._qvp.melting_layer(a, a, a, h, [0.0], [1.0], boundaries=7)


def test_melting_layer_time_consistency():
    good = _melting_profile()
    h = good[0]
    # profile 3: a valid-looking signature 2 km higher; profile 5: no signature
    jump = _melting_profile(bottom=4500.0, top=5100.0)
    jump = (h, np.where(np.isnan(jump[1]), 25.0, jump[1]), *jump[2:])
    none = (h, good[1], np.full(h.size, 0.99), np.full(h.size, 0.4))
    profiles = [good] * 3 + [jump] + [good] + [none] + [good] * 3
    prof = _profiles_dataset(profiles)
    raw = melting_layer(prof, median_window=1, max_fill=0)
    assert float(raw["melting_layer_top"][3]) == pytest.approx(5100.0, abs=100)
    assert np.isnan(raw["melting_layer_top"][5])
    ml = melting_layer(prof)
    np.testing.assert_array_equal(ml["melting_layer_flag"], [1, 1, 1, 3, 1, 3, 1, 1, 1])
    np.testing.assert_allclose(ml["melting_layer_top"], 3100.0, atol=100)
    np.testing.assert_allclose(ml["melting_layer_bottom"], 2500.0, atol=100)
    ml = melting_layer(prof, max_fill=0)
    np.testing.assert_array_equal(ml["melting_layer_flag"], [1, 1, 1, 2, 1, 0, 1, 1, 1])
    assert np.isnan(ml["melting_layer_top"][[3, 5]]).all()
    assert ml["melting_layer_flag"].attrs["flag_meanings"] == (
        "none detected rejected filled"
    )


def test_melting_layer_engines_agree():
    if not vp.HAS_COMPILED_KERNEL:
        pytest.skip("compiled kernel not built")
    rnd = np.random.default_rng(3)
    profiles = []
    for _ in range(200):
        bottom = rnd.uniform(1500, 4000)
        h, z, rho, zdr = _melting_profile(bottom, bottom + rnd.uniform(300, 900))
        z = z + rnd.normal(0, 1.0, h.size)
        rho = rho + rnd.normal(0, 0.005, h.size)
        zdr = zdr + rnd.normal(0, 0.1, h.size)
        for x in (z, rho, zdr):
            x[rnd.random(h.size) < 0.05] = np.nan
        profiles.append((h, z, rho, zdr))
    prof = _profiles_dataset(profiles)
    for kwargs in (
        {},
        {"freezing_level": 3000.0},
        {"boundaries": "half_prominence"},
        {"boundaries": "half_prominence", "edge_fraction": 0.8},
        {"rhohv_onset": 0.95, "onset_fraction": 0.3},
    ):
        c = melting_layer(prof, engine="compiled", **kwargs)
        n = melting_layer(prof, engine="numpy", **kwargs)
        xr.testing.assert_allclose(c, n, rtol=0, atol=1e-9)
        for name in ("melting_layer_peak", "melting_layer_flag"):
            xr.testing.assert_identical(c[name], n[name])
    assert (c["melting_layer_flag"] > 0).sum() > 150
    xr.testing.assert_identical(prof.radarx.melting_layer(), melting_layer(prof))
    single = melting_layer(prof, engine="compiled", n_threads=1)
    xr.testing.assert_identical(single, melting_layer(prof, engine="compiled"))


@pytest.fixture(scope="module")
def armor_volume():
    xd = pytest.importorskip("xradar")
    from open_radar_data import DATASETS

    file = DATASETS.fetch("RAW_NA_000_125_20080411181219")
    return xd.io.open_iris_datatree(file)


@pytest.mark.parametrize("reduction", ["mean", "median"])
def test_real_volume(armor_volume, reduction):
    fields = ["DBZH", "ZDR", "RHOHV", "PHIDP", "KDP"]
    out = qvp(armor_volume, fields, reduction=reduction, engine="numpy")
    assert float(out["elevation"]) == pytest.approx(12.0, abs=0.1)
    assert np.all(np.diff(out["height"]) > 0)
    rho = out["RHOHV"].values
    assert np.nanmin(rho) > 0.6 and np.nanmax(rho) <= 1.0
    assert np.isfinite(out["DBZH"]).sum() > 100
    if vp.HAS_COMPILED_KERNEL:
        c = qvp(armor_volume, fields, reduction=reduction, engine="compiled")
        for name in fields:
            np.testing.assert_allclose(
                c[name], out[name], rtol=1e-6, atol=2e-5, equal_nan=True
            )
    ml = melting_layer(out)
    assert set(ml.data_vars) == {
        "melting_layer_top",
        "melting_layer_bottom",
        "melting_layer_peak",
        "melting_layer_flag",
    }


def test_real_timeseries_melting_layer():
    """
    ARMOR, 11 April 2008: a melting-layer signature between about 2.5 and
    4.5 km in every volume. The BMX radiosonde of 00 UTC 12 April 2008 has
    its 0 °C level at 3859 m and its wet-bulb 0 °C level at 3588 m (IEM
    archive, :func:`radarx.io.sounding.isotherm_height` and
    :func:`radarx.io.sounding.wet_bulb_zero_height`).
    """
    xd = pytest.importorskip("xradar")
    from open_radar_data import DATASETS

    names = sorted(n for n in DATASETS.registry if n.startswith("RAW_NA_000_125_"))
    volumes = [xd.io.open_iris_datatree(DATASETS.fetch(n)) for n in names[::4]]
    tqvp = qvp_timeseries(volumes, ["DBZH", "ZDR", "RHOHV"], elevation=12.0)
    env = {"freezing_level": 3859.0, "wet_bulb_zero_height": 3588.0}
    ml = melting_layer(tqvp, environment=env)
    half = melting_layer(tqvp, boundaries="half_prominence")
    for out in (ml, half):
        ok = out["melting_layer_flag"].isin([1, 3]).values
        assert ok.sum() >= 6
        for name in ("melting_layer_top", "melting_layer_bottom"):
            values = out[name].values[ok]
            assert ((values > 2000) & (values < 5000)).all()
        depth = (out["melting_layer_top"] - out["melting_layer_bottom"])[ok]
        assert ((depth >= 200) & (depth <= 2000)).all()
    # the onset top lies above the half-prominence top, close to 0 degC
    top = float(ml["melting_layer_top"].median())
    assert top > float(half["melting_layer_top"].median()) + 100
    assert abs(top - 3859.0) < 400
    offset = ml["melting_layer_top_offset_freezing_level"]
    np.testing.assert_allclose(offset, ml["melting_layer_top"] - 3859.0)


def test_edge_cases():
    ds, *_ = _sweep()
    # fields found by CF standard name; a test is skipped if disabled
    renamed = ds.rename({"RHOHV": "rho", "DBZH": "z"})
    renamed["rho"].attrs["standard_name"] = "radar_correlation_coefficient_hv"
    out = qvp(renamed, ["PHIDP", "z"])
    assert out["PHIDP"].attrs["qvp_filter"] == "rho > 0.6"
    assert qvp(renamed, "PHIDP", rhohv=None)["PHIDP"].attrs["qvp_filter"] == "none"
    # (range, azimuth) order, a single linear variable name, no time
    flipped = ds.drop_vars("time").transpose("range", "azimuth")
    a = qvp(flipped, ["DBZH", "PHIDP"], linear="DBZH")
    b = qvp(ds, ["DBZH", "PHIDP"])
    np.testing.assert_allclose(a["DBZH"], b["DBZH"], rtol=1e-6)
    assert np.isnat(a["time"].values)
    # time series from one sweep, with counts; empty input
    ts = qvp_timeseries(ds, "PHIDP", counts=True)
    assert ts["PHIDP_count"].dims == ("time", "height")
    with pytest.raises(ValueError, match="No volumes"):
        qvp_timeseries([])
    with pytest.raises(ValueError, match="data variables"):
        qvp(xr.Dataset(coords=ds.coords), min_rhohv=None)
    # sweeps without a site and sweep names without numbers
    dt = _volume()
    tree = xr.DataTree.from_dict(
        {"/": dt.ds, "sweep_a": dt["sweep_0"].ds, "sweep_b": dt["sweep_2"].ds}
    )
    out = qvp(tree)
    assert float(out["elevation"]) == pytest.approx(15.0, abs=0.05)
    assert float(out["altitude"]) == ALT
    with pytest.raises(ValueError, match="not found"):
        qvp(dt, sweep=7)
    with pytest.raises(ValueError, match="No sweep"):
        qvp(xr.DataTree(dt.ds))
    with pytest.raises(ValueError, match="2-D"):
        qvp(ds.assign(bad=("range", RNG)), "bad")
    # melting layer input checks
    prof = _profiles_dataset([_melting_profile()])
    with pytest.raises(ValueError, match="needs a zdr"):
        melting_layer(prof.drop_vars("ZDR"))
    with pytest.raises(ValueError, match="ascending"):
        melting_layer(prof.isel(height=slice(None, None, -1)))
    # numeric time axis, and a profile without echo
    numeric = prof.assign_coords(time=[0.0])
    assert int(melting_layer(numeric)["melting_layer_flag"][0]) == 1
    empty = prof * np.nan
    assert int(melting_layer(empty, engine="numpy")["melting_layer_flag"][0]) == 0
    # a ZDR increase that never falls off is not a layer
    h = prof["height"].values
    ramp = (h, np.full(h.size, 30.0), np.full(h.size, 0.95), 0.5 + h / 1e3)
    flat = melting_layer(_profiles_dataset([ramp]), engine="numpy")
    assert int(flat["melting_layer_flag"][0]) == 0
