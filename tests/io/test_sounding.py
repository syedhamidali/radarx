#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for radarx.io.sounding (parsers, profile helpers, kernels, ERA5)."""

import os
import socket
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from radarx.io import _sounding_numpy as ref
from radarx.io import sounding

DATA = Path(__file__).parent / "data"
ENGINES = ["numpy"] + (["compiled"] if sounding.HAS_COMPILED_KERNEL else [])
compiled = pytest.mark.skipif(
    not sounding.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)


def _online(host, port=443):
    try:
        socket.create_connection((host, port), timeout=5).close()
        return True
    except OSError:
        return False


def network(host):
    return pytest.mark.skipif(not _online(host), reason=f"{host} not reachable")


def synthetic_profile(lapse=0.0065, t0=300.0, n=81, top=20000.0, scale=8000.0):
    """Linear temperature, exponential pressure, linear wind profile."""
    z = np.linspace(0.0, top, n)
    t = t0 - lapse * z
    p = 100000.0 * np.exp(-z / scale)
    u = 2.0 + 1e-3 * z
    v = np.full(n, -3.0)
    return sounding._profile_dataset(
        height=z,
        pressure=p,
        temperature=t,
        dewpoint=t - 5.0,
        u=u,
        v=v,
        time=np.datetime64("2022-03-31T00:00"),
        latitude=33.9,
        longitude=-88.3,
        attrs={"source": "synthetic"},
    )


# ---------------------------------------------------------------------------
# parsers


def test_iem_json():
    ds = sounding.open_sounding_file(DATA / "iem_KJAN_202203310000.json")
    assert ds.attrs["station"] == "KJAN"
    assert ds["time"].values == np.datetime64("2022-03-31T00:00", "ns")
    z = ds["height"].values
    assert np.all(np.diff(z) > 0)
    # 991 hPa level: 91 gpm, 16.0 C, dew point 14.9 C, 170 deg at 5 kt
    lev = ds.sel(height=ds["height"][np.argmin(abs(ds["pressure"].values - 99100))])
    assert float(lev["pressure"]) == pytest.approx(99100.0)
    assert float(lev["temperature"]) == pytest.approx(289.15)
    assert float(lev["dewpoint"]) == pytest.approx(288.05)
    assert float(lev["wind_speed"]) == pytest.approx(5 * 0.514444)
    assert float(lev["wind_direction"]) == pytest.approx(170.0)
    assert float(lev["u"]) == pytest.approx(-5 * 0.514444 * np.sin(np.radians(170)))
    assert float(lev["height"]) == pytest.approx(
        6371008.8 * 91 / (6371008.8 - 91), rel=1e-9
    )
    for name, units in [("pressure", "Pa"), ("temperature", "K"), ("u", "m s-1")]:
        assert ds[name].attrs["units"] == units
    assert ds["relative_humidity"].attrs["units"] == "1"


def test_uwyo_csv():
    ds = sounding.open_sounding_file(
        DATA / "uwyo_72235_2022033100.csv", station="72235"
    )
    assert ds.attrs["launch_time"] == "2022-03-31T00:11:56"
    assert ds.sizes["height"] == 113
    first = ds.isel(height=0)
    assert float(first["temperature"]) == pytest.approx(16.1 + 273.15)
    assert float(first["wind_speed"]) == pytest.approx(2.6)
    assert float(first["latitude"]) == pytest.approx(32.3213)
    # relative humidity derived from T and Td: 16.1 / 15.0 C -> 0.93
    assert float(first["relative_humidity"]) == pytest.approx(0.93, abs=0.01)


def test_igra2_text_time_selection():
    path = DATA / "igra2_USM00072235_20220330-31.txt"
    first = sounding.open_sounding_file(path)
    assert first["time"].values == np.datetime64("2022-03-30T12:00", "ns")
    ds = sounding.open_sounding_file(path, time="2022-03-31T00:00")
    assert ds["time"].values == np.datetime64("2022-03-31T00:00", "ns")
    assert ds.attrs["station"] == "USM00072235"
    assert ds.attrs["launch_time"] == "2022-03-31T00:11"
    surface = ds.isel(height=0)
    assert float(surface["pressure"]) == pytest.approx(99143.0)
    assert float(surface["temperature"]) == pytest.approx(16.1 + 273.15)
    assert float(surface["dewpoint"]) == pytest.approx(16.1 - 1.1 + 273.15)
    assert float(surface["wind_speed"]) == pytest.approx(2.6)
    with pytest.raises(ValueError):
        sounding.open_sounding_file(path, time="2022-03-31T12:00")


def test_igra2_zip(tmp_path):
    import zipfile

    path = tmp_path / "USM00072235-data.txt.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.write(DATA / "igra2_USM00072235_20220330-31.txt", "USM00072235-data.txt")
    ds = sounding.open_sounding_file(path, time="2022-03-31T00")
    assert ds["time"].values == np.datetime64("2022-03-31T00:00", "ns")


def test_sharppy_matches_iem():
    spc = sounding.open_sounding_file(DATA / "sharppy_JAN_220331_0000.txt")
    iem = sounding.open_sounding_file(DATA / "iem_KJAN_202203310000.json")
    assert spc.attrs["station"] == "JAN"
    assert spc["time"].values == np.datetime64("2022-03-31T00:00", "ns")
    common = np.intersect1d(spc["height"], iem["height"])
    assert common.size > 50
    for name in ("temperature", "dewpoint", "u", "v", "pressure"):
        xr.testing.assert_allclose(
            spc[name].sel(height=common).reset_coords(drop=True),
            iem[name].sel(height=common).reset_coords(drop=True),
        )


def test_generic_csv(tmp_path):
    path = tmp_path / "snd.csv"
    path.write_text(
        "# test\npres,hght,tmpc,dwpc,drct,sknt\n"
        "1000,100,20,15,180,10\n850,1500,10,5,270,20\n850,1500,10,5,270,20\n"
        "700,3000,0,-9999,270,30\n500,5600,-15,-25,-9999,-9999\n"
    )
    ds = sounding.open_sounding_file(path)
    assert ds.sizes["height"] == 4  # duplicate level merged
    assert np.isnan(ds["dewpoint"].values[2])
    assert np.isnan(ds["u"].values[3])
    assert float(ds["v"][0]) == pytest.approx(10 * 0.514444)  # southerly
    assert float(ds["u"][1]) == pytest.approx(20 * 0.514444)  # westerly


def test_missing_heights_filled_in_log_pressure():
    ds = sounding._profile_dataset(
        pressure=np.array([100000.0, 85000.0, 70000.0]),
        geopotential_height=np.array([100.0, np.nan, 3000.0]),
        temperature=np.array([290.0, 280.0, 270.0]),
    )
    assert ds.sizes["height"] == 3
    w = np.log(100000 / 85000) / np.log(100000 / 70000)
    assert float(ds["geopotential_height"][1]) == pytest.approx(100 + w * 2900)


def test_observed_sources_agree_on_freezing_level():
    iem = sounding.open_sounding_file(DATA / "iem_KJAN_202203310000.json")
    uwyo = sounding.open_sounding_file(DATA / "uwyo_72235_2022033100.csv")
    igra = sounding.open_sounding_file(
        DATA / "igra2_USM00072235_20220330-31.txt", time="2022-03-31T00"
    )
    levels = [float(sounding.isotherm_height(p)) for p in (iem, uwyo, igra)]
    assert 3500 < levels[0] < 4200
    assert np.ptp(levels) < 150


# ---------------------------------------------------------------------------
# stations


def test_station_lookup():
    near = sounding.nearest_station(33.8967, -88.3289, "2022-03-30", source="iem", n=2)
    assert list(near["iem_id"].values) == ["KBMX", "KJAN"]
    assert near["distance"].values[0] == pytest.approx(164e3, rel=0.02)
    assert sounding._station_id("JAN", "uwyo") == "72235"
    assert sounding._station_id(72235, "igra2") == "USM00072235"
    assert sounding._station_id("USM00072230", "iem") == "KBMX"
    assert sounding.station_list().sizes["station"] > 1000


# ---------------------------------------------------------------------------
# thermodynamics


@pytest.mark.parametrize("engine", ENGINES)
def test_thermodynamics(engine):
    es = sounding.saturation_vapor_pressure(273.15, engine=engine)
    assert float(es) == pytest.approx(611.2)
    assert es.attrs["units"] == "Pa"
    t = xr.DataArray(np.linspace(230, 310, 9), dims="level")
    td = sounding.dewpoint_from_vapor_pressure(
        sounding.saturation_vapor_pressure(t, engine=engine), engine=engine
    )
    np.testing.assert_allclose(td, t, rtol=1e-12)
    assert td.dims == ("level",)
    p = xr.DataArray(np.linspace(100000, 30000, 9), dims="level")
    q = sounding.specific_humidity_from_dewpoint(t - 3, p, engine=engine)
    np.testing.assert_allclose(
        sounding.dewpoint_from_specific_humidity(q, p, engine=engine), t - 3, rtol=1e-10
    )
    rh = sounding.relative_humidity_from_dewpoint(t, t, engine=engine)
    np.testing.assert_allclose(rh, 1.0)
    rho = sounding.air_density(101325.0, 288.15, engine=engine)
    assert float(rho) == pytest.approx(1.2250, abs=1e-4)  # ISA sea level
    h = np.array([0.0, 5000.0, 10000.0, 20000.0])
    z = sounding.geopotential_to_height(h * 9.80665, engine=engine)
    np.testing.assert_allclose(z, 6371008.8 * h / (6371008.8 - h))
    assert float(z[2] - 10000.0) == pytest.approx(15.7, abs=0.1)


@pytest.mark.parametrize("engine", ENGINES)
def test_wet_bulb(engine):
    p = np.array([100000.0, 85000.0, 70000.0, 50000.0])
    t = np.array([303.15, 288.15, 275.15, 255.15])
    td = t - np.array([10.0, 5.0, 2.0, 0.0])
    tw = sounding.wet_bulb_temperature(p, t, td, engine=engine).values
    assert np.all(tw <= t) and np.all(tw >= td)
    assert tw[-1] == pytest.approx(t[-1])  # saturated
    # the isobaric energy balance holds at the solution
    r = ref._mixing_ratio(ref._esat(td), p)
    rs = ref._mixing_ratio(ref._esat(tw), p)
    residual = (ref.CPD + r * ref.CPV) * (t - tw) - ref._latent_heat(tw) * (rs - r)
    np.testing.assert_allclose(residual, 0.0, atol=1e-3)
    # 30 C, Td 20 C at 1000 hPa: about 23 C (psychrometric charts)
    assert tw[0] - 273.15 == pytest.approx(22.9, abs=0.3)


def test_wet_bulb_vs_metpy():
    metpy_calc = pytest.importorskip("metpy.calc")
    from metpy.units import units

    p = np.array([1000.0, 900.0, 800.0, 700.0, 600.0])
    t = np.array([25.0, 18.0, 10.0, 2.0, -6.0])
    td = np.array([18.0, 10.0, 0.0, -8.0, -20.0])
    ours = sounding.wet_bulb_temperature(p * 100, t + 273.15, td + 273.15).values
    theirs = metpy_calc.wet_bulb_temperature(
        p * units.hPa, t * units.degC, td * units.degC
    ).m_as("K")
    # isobaric vs pseudo-adiabatic wet-bulb temperature (Davies-Jones 2008)
    np.testing.assert_allclose(ours, theirs, atol=0.4)


# ---------------------------------------------------------------------------
# profile helpers


@pytest.mark.parametrize("engine", ENGINES)
def test_interpolate_profile_any_shape(engine):
    prof = synthetic_profile()
    heights = xr.DataArray(
        np.random.default_rng(1).uniform(100, 19000, (7, 11)),
        dims=("azimuth", "range"),
        coords={"azimuth": np.arange(7.0)},
    )
    out = sounding.interpolate_profile(prof, heights, engine=engine)
    assert out["temperature"].dims == ("azimuth", "range")
    assert "azimuth" in out.coords
    np.testing.assert_allclose(out["temperature"], 300 - 0.0065 * heights, rtol=1e-12)
    # pressure is interpolated in log: exact for an exponential profile
    np.testing.assert_allclose(
        out["pressure"], 1e5 * np.exp(-heights / 8000), rtol=1e-12
    )
    np.testing.assert_allclose(out["u"], 2 + 1e-3 * heights, rtol=1e-12)
    np.testing.assert_allclose(out["wind_speed"], np.hypot(out["u"], out["v"]))
    assert out["temperature"].attrs["units"] == "K"


@pytest.mark.parametrize("engine", ENGINES)
def test_interpolate_profile_outside_and_gaps(engine):
    prof = synthetic_profile()
    prof["temperature"][10:20] = np.nan  # a gap is bridged linearly
    h = xr.DataArray([-50.0, 3000.0, 25000.0, np.nan], dims="gate")
    out = sounding.interpolate_profile(prof, h, ["temperature"], engine=engine)
    t = out["temperature"].values
    assert np.isnan(t[0]) and np.isnan(t[2]) and np.isnan(t[3])
    assert t[1] == pytest.approx(300 - 0.0065 * 3000)
    out = sounding.interpolate_profile(
        prof, h, ["temperature"], extrapolate=True, engine=engine
    )
    t = out["temperature"].values
    assert t[0] == pytest.approx(300.0)
    assert t[2] == pytest.approx(300 - 0.0065 * 20000)


@pytest.mark.parametrize("engine", ENGINES)
def test_isotherm_heights(engine):
    prof = synthetic_profile()
    for iso in (273.15, 263.15, 253.15):
        z = sounding.isotherm_height(prof, iso, engine=engine)
        assert float(z) == pytest.approx((300 - iso) / 0.0065, rel=1e-12)
    # warm nose: below freezing near the ground, warm layer aloft
    z = np.array([0, 500, 1000, 1500, 2000, 3000.0])
    t = np.array([271, 272, 276, 276, 272, 266.0])
    prof = sounding._profile_dataset(
        height=z, temperature=t, pressure=1e5 * np.exp(-z / 8e3)
    )
    top = float(sounding.isotherm_height(prof, engine=engine))
    low = float(sounding.isotherm_height(prof, which="lowest", engine=engine))
    assert top == pytest.approx(1500 + 500 * (276 - 273.15) / 4)
    assert low == top  # one downward crossing (the ground layer is a rise)
    cold = sounding._profile_dataset(
        height=z, temperature=np.full(6, 260.0), pressure=1e5 * np.exp(-z / 8e3)
    )
    assert np.isnan(float(sounding.isotherm_height(cold, engine=engine)))


@pytest.mark.parametrize("engine", ENGINES)
def test_wet_bulb_zero_height(engine):
    prof = synthetic_profile()
    saturated = prof.assign(dewpoint=prof["temperature"])
    wbz = sounding.wet_bulb_zero_height(saturated, engine=engine)
    assert float(wbz) == pytest.approx((300 - 273.15) / 0.0065, rel=1e-9)
    dry = float(sounding.wet_bulb_zero_height(prof, engine=engine))
    assert dry < float(wbz)


@pytest.mark.parametrize("engine", ENGINES)
def test_mean_wind(engine):
    prof = synthetic_profile()
    mw = sounding.mean_wind(prof, 1000, 7000, engine=engine)
    assert float(mw["u"]) == pytest.approx(2 + 1e-3 * 4000)
    assert float(mw["v"]) == pytest.approx(-3.0)
    assert float(mw["wind_direction"]) == pytest.approx(
        np.degrees(np.arctan2(-6.0, 3.0)) % 360
    )
    agl = sounding.mean_wind(
        prof.assign_coords(height=prof.height + 200),
        0,
        6000,
        above_ground=True,
        engine=engine,
    )
    assert float(agl["u"]) == pytest.approx(2 + 1e-3 * 3000)
    assert np.isnan(float(sounding.mean_wind(prof, 0, 30000, engine=engine)["u"]))


@pytest.mark.parametrize("engine", ENGINES)
def test_column_profiles(engine):
    """Profiles with an extra dimension are handled column by column."""
    prof = xr.concat([synthetic_profile(lapse=g) for g in (0.005, 0.0065, 0.008)], "x")
    z = sounding.isotherm_height(prof, engine=engine)
    np.testing.assert_allclose(z, 26.85 / np.array([0.005, 0.0065, 0.008]))
    h = xr.DataArray(np.array([[1000.0, 2000.0]] * 3), dims=("x", "gate"))
    out = sounding.interpolate_profile(prof, h, ["temperature"], engine=engine)
    np.testing.assert_allclose(out["temperature"][:, 0], 300 - np.array([5, 6.5, 8.0]))


# ---------------------------------------------------------------------------
# compiled vs NumPy


@compiled
def test_engines_agree():
    rng = np.random.default_rng(0)
    k = sounding._sounding
    ncol, nlev, nt = 50, 37, 300
    z = np.sort(rng.uniform(0, 20000, (ncol, nlev)), axis=1)
    vals = rng.normal(size=(3, ncol, nlev))
    vals[0] = 300 - 0.0065 * z + rng.normal(size=z.shape)
    vals[2] = np.exp(-z / 8000) * 1e5
    vals[1][rng.random(vals[1].shape) < 0.2] = np.nan
    target = rng.uniform(-500, 21000, (ncol, nt))
    target[0, :5] = np.nan
    logv = np.array([False, False, True])
    for extrap in (False, True):
        a = k.interp_vertical(z, vals, target, logv, extrap, 0)
        b = ref.interp_vertical(z, vals, target, logv, extrap)
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-14, equal_nan=True)
    one = k.interp_vertical(z[:1], vals[:, :1], target, logv, False, 0)
    np.testing.assert_allclose(
        one,
        ref.interp_vertical(z[:1], vals[:, :1], target, logv),
        rtol=1e-10,
        atol=1e-14,
        equal_nan=True,
    )
    for highest in (True, False):
        np.testing.assert_allclose(
            k.level_crossing(z, vals[0], 273.15, highest, 0),
            ref.level_crossing(z, vals[0], 273.15, highest),
            rtol=1e-10,
            atol=1e-14,
            equal_nan=True,
        )
    np.testing.assert_allclose(
        k.layer_mean(z, vals, 1000.0, 9000.0, 0),
        ref.layer_mean(z, vals, 1000.0, 9000.0),
        rtol=1e-10,
        equal_nan=True,
    )
    fields = rng.normal(size=(2, 4, 5, 6, 7))
    lat, lon = np.linspace(30, 31.25, 6), np.linspace(-90, -88.5, 7)
    qlat, qlon = rng.uniform(29.9, 31.3, 200), rng.uniform(-90.1, -88.4, 200)
    np.testing.assert_allclose(
        k.bilinear_columns(fields, np.array([0.3, 0.7]), lat, lon, qlat, qlon, 0),
        ref.bilinear_columns(fields, np.array([0.3, 0.7]), lat, lon, qlat, qlon),
        rtol=1e-10,
        atol=1e-14,
        equal_nan=True,
    )
    u, v, ang = rng.normal(size=(3, 1000))
    np.testing.assert_allclose(
        k.rotate_wind(u, v, ang * 10, 0), ref.rotate_wind(u, v, ang * 10), rtol=1e-12
    )
    p = rng.uniform(20000, 105000, 5000)
    t = rng.uniform(220, 315, 5000)
    td = t - rng.uniform(0, 30, 5000)
    q = rng.uniform(0, 0.02, 5000)
    for op, args in [
        ("esat", [t]),
        ("dewpoint", [p * 0.01]),
        ("vapor_pressure", [q, p]),
        ("specific_humidity", [p * 0.01, p]),
        ("wet_bulb", [p, t, td]),
        ("height", [p]),
        ("density", [p, t, q]),
    ]:
        np.testing.assert_allclose(
            k.thermo(op, args, 0),
            ref.thermo(op, args),
            rtol=1e-9,
            atol=1e-9,
            err_msg=op,
        )


@compiled
def test_compiled_threads_identical():
    rng = np.random.default_rng(3)
    z = np.sort(rng.uniform(0, 20000, (1, 37)), axis=1)
    vals = rng.normal(size=(2, 1, 37))
    target = rng.uniform(0, 20000, (1, 100000))
    logv = np.zeros(2, dtype=bool)
    a = sounding._sounding.interp_vertical(z, vals, target, logv, False, 1)
    b = sounding._sounding.interp_vertical(z, vals, target, logv, False, 8)
    np.testing.assert_array_equal(a, b)


def test_engine_argument():
    with pytest.raises(ValueError):
        sounding.saturation_vapor_pressure(273.15, engine="fortran")


# ---------------------------------------------------------------------------
# wind rotation and grid backgrounds


def _grid(nx=21, ny=17, nz=12, dx=10e3, lat0=33.8967, lon0=-88.3289):
    from radarx.grid.cone import _crs_wkt

    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dx
    z = np.linspace(200, 12000, nz)
    return xr.Dataset(
        {"DBZH": (("z", "y", "x"), np.zeros((nz, ny, nx), dtype=np.float32))},
        coords={
            "z": z,
            "y": y,
            "x": x,
            "crs_wkt": _crs_wkt(lat0, lon0),
            "time": np.datetime64("2022-03-30T23:46:39", "ns"),
        },
    )


def test_rotation_angle_matches_meridian_convergence():
    import pyproj

    grid = _grid(dx=50e3)
    lon, lat, angle = sounding._grid_lonlat_rotation(grid)
    x = grid["x"].values
    assert np.allclose(angle[:, x == 0], 0.0, atol=1e-6)
    # meridians converge to the pole: true north tilts towards the centre line
    assert np.all(angle[:, x > 0] < 0) and np.all(angle[:, x < 0] > 0)
    crs = sounding._grid_crs(grid)
    conv = pyproj.Proj(crs).get_factors(lon, lat).meridian_convergence
    np.testing.assert_allclose(np.abs(angle), np.abs(conv), atol=2e-3)


@pytest.mark.parametrize("engine", ENGINES)
def test_rotate_wind_known(engine):
    k = sounding._kernel(engine)
    # true north 90 deg clockwise from grid +y: a southerly wind blows along +x
    out = np.asarray(
        k.rotate_wind(
            np.array([0.0, 5.0]), np.array([10.0, 0.0]), np.array([90.0, 90.0]), 0
        )
    )
    np.testing.assert_allclose(out[:, 0], [10.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(out[:, 1], [0.0, -5.0], atol=1e-12)


@pytest.mark.parametrize("engine", ENGINES)
def test_profile_to_grid(engine):
    grid = _grid()
    prof = synthetic_profile()
    bg = sounding.profile_to_grid(prof, grid, engine=engine)
    assert bg["temperature"].dims == ("z", "y", "x")
    assert bg["freezing_level"].dims == ("y", "x")
    np.testing.assert_allclose(
        bg["temperature"].isel(y=3, x=5), 300 - 0.0065 * grid["z"], rtol=1e-6
    )
    np.testing.assert_allclose(bg["freezing_level"], 26.85 / 0.0065)
    speed = np.hypot(bg["u"], bg["v"])
    np.testing.assert_allclose(
        speed, np.hypot(2 + 1e-3 * grid["z"], 3.0).broadcast_like(speed), rtol=1e-5
    )
    centre = bg.sel(x=0.0, y=0.0)
    np.testing.assert_allclose(centre["u"], 2 + 1e-3 * grid["z"], rtol=1e-5)
    np.testing.assert_allclose(centre["v"], -3.0, rtol=1e-5)
    expected_rho = sounding.air_density(
        bg["pressure"], bg["temperature"], bg["specific_humidity"]
    )
    np.testing.assert_allclose(bg["air_density"], expected_rho, rtol=1e-5)
    assert bg["u"].attrs["standard_name"] == "x_wind"
    assert "crs_wkt" in bg.coords


def _synthetic_era5(times, south=33.0, north=35.0, west=-89.5, east=-87.0):
    """ERA5-like fields: T lapse rate, height from level, winds linear in lat/lon."""
    levels = np.array([1000, 925, 850, 700, 500, 300, 200, 100], dtype=float)
    lat = np.arange(south, north + 0.01, 0.25)
    lon = np.arange(west, east + 0.01, 0.25)
    H = 8000.0 * np.log(1000.0 / levels)  # geopotential height of each level
    T, LEV, LAT, LON = np.meshgrid(
        np.arange(len(times)), levels, lat, lon, indexing="ij"
    )
    Hg = 8000.0 * np.log(1000.0 / LEV)
    temp = 300.0 - 0.0065 * Hg + 1.0 * T  # 1 K warmer per time step
    ds = xr.Dataset(
        {
            "geopotential": (("time", "level", "latitude", "longitude"), Hg * 9.80665),
            "temperature": (("time", "level", "latitude", "longitude"), temp),
            "specific_humidity": (
                ("time", "level", "latitude", "longitude"),
                np.full(T.shape, 0.005),
            ),
            "u": (
                ("time", "level", "latitude", "longitude"),
                10.0 + 2.0 * (LON - west),
            ),
            "v": (("time", "level", "latitude", "longitude"), 3.0 * (LAT - south)),
            "omega": (
                ("time", "level", "latitude", "longitude"),
                np.full(T.shape, -0.5),
            ),
        },
        coords={
            "time": np.array(times, dtype="datetime64[ns]"),
            "level": levels,
            "latitude": lat,
            "longitude": lon,
        },
    )
    return ds, H


@pytest.mark.parametrize("engine", ENGINES)
def test_era5_profile_synthetic(monkeypatch, engine):
    seen = {}

    def fake_fields(source, box, times, cache=True):
        seen.update(source=source, box=box, times=times)
        return _synthetic_era5(times)[0]

    monkeypatch.setattr(sounding, "_era5_fields", fake_fields)
    monkeypatch.setattr(sounding, "_cds_available", lambda: False)
    prof = sounding.era5_profile(33.9, -88.33, "2022-03-30T23:45", engine=engine)
    assert seen["source"] == "gcs"
    assert [str(t) for t in seen["times"]] == [
        "2022-03-30T23:00:00",
        "2022-03-31T00:00:00",
    ]
    assert prof.attrs["era5_time_weights"] == "0.2500, 0.7500"
    H = 8000.0 * np.log(1000.0 / np.array([1000, 925, 850, 700, 500, 300, 200, 100.0]))
    z = 6371008.8 * H / (6371008.8 - H)
    np.testing.assert_allclose(prof["height"], z, rtol=1e-9)
    np.testing.assert_allclose(prof["temperature"], 300 - 0.0065 * H + 0.75, rtol=1e-9)
    np.testing.assert_allclose(prof["u"], 10 + 2 * (-88.33 + 89.5), rtol=1e-9)
    np.testing.assert_allclose(prof["v"], 3 * 0.9, rtol=1e-9)
    rho = sounding.air_density(
        prof["pressure"], prof["temperature"], prof["specific_humidity"]
    )
    np.testing.assert_allclose(prof["w"], 0.5 / (rho * 9.80665), rtol=1e-9)
    assert prof["pressure"].values[0] == pytest.approx(100000.0)
    near = sounding.era5_profile(
        33.9, -88.33, "2022-03-30T23:45", time_interpolation="nearest", engine=engine
    )
    assert [str(t) for t in seen["times"]] == ["2022-03-31T00:00:00"]
    np.testing.assert_allclose(near["temperature"], 300 - 0.0065 * H, rtol=1e-9)


@pytest.mark.parametrize("engine", ENGINES)
def test_era5_column_synthetic(monkeypatch, engine):
    monkeypatch.setattr(
        sounding, "_era5_fields", lambda s, b, t, cache=True: _synthetic_era5(t)[0]
    )
    monkeypatch.setattr(sounding, "_cds_available", lambda: False)
    grid = _grid()
    bg = sounding.era5_column(grid, engine=engine)  # grid time 23:46:39
    assert bg["u"].dims == ("z", "y", "x")
    lon, lat, angle = sounding._grid_lonlat_rotation(grid)
    u_e = 10 + 2 * (lon + 89.5)
    v_e = 3 * (lat - 33.0)
    a = np.radians(angle)
    ux = u_e * np.cos(a) + v_e * np.sin(a)
    level = bg.sel(z=grid["z"][3])
    np.testing.assert_allclose(level["u"], ux, rtol=1e-5)
    frz = bg["freezing_level"].values
    w = (46 * 60 + 39) / 3600.0  # time weight of the second hour
    H0 = (300 + w - 273.15) / 0.0065
    np.testing.assert_allclose(frz, 6371008.8 * H0 / (6371008.8 - H0), rtol=1e-4)
    assert bg.attrs["era5_source"] == "gcs"
    assert set(sounding.profile_to_grid(synthetic_profile(), grid).data_vars) == set(
        bg.data_vars
    )


def test_time_bracket():
    times, w = sounding._time_bracket("2022-03-30T23:46:39", 1, "linear")
    assert (
        str(times[0]) == "2022-03-30T23:00:00"
        and str(times[1]) == "2022-03-31T00:00:00"
    )
    np.testing.assert_allclose(w, [1 - 2799 / 3600, 2799 / 3600])
    times, w = sounding._time_bracket("2022-03-30T23:46:39", 6, "linear")
    assert str(times[0]) == "2022-03-30T18:00:00"
    times, w = sounding._time_bracket("2022-03-31T00:00", 6, "linear")
    assert times.size == 1 and w[0] == 1.0
    times, _ = sounding._time_bracket("2022-03-30T23:46", 6, "nearest")
    assert str(times[0]) == "2022-03-31T00:00:00"
    assert sounding._synoptic_time("2022-03-30T23:46") == np.datetime64(
        "2022-03-31T00:00"
    )
    assert sounding._synoptic_time("2022-03-30T17:59") == np.datetime64(
        "2022-03-30T12:00"
    )


# ---------------------------------------------------------------------------
# real radar data and accessors


@pytest.fixture(scope="module")
def nexrad_volume():
    import xradar as xd
    from open_radar_data import DATASETS

    file = DATASETS.fetch("KLBB20160601_150025_V06")
    return xd.io.open_nexradlevel2_datatree(file, sweep=[0, 1]).xradar.georeference()


def test_accessor_interpolate_profile_on_sweep(nexrad_volume):
    sweep = nexrad_volume["sweep_0"].to_dataset()
    prof = synthetic_profile()
    out = sweep.radarx.interpolate_profile(prof)
    assert out["temperature"].dims == sweep["z"].dims
    np.testing.assert_allclose(
        out["temperature"], 300 - 0.0065 * sweep["z"], rtol=1e-12
    )
    assert "DBZH" in out


def test_accessor_sounding(monkeypatch, nexrad_volume):
    calls = {}

    def fake_era5(lat, lon, time, **kwargs):
        calls.update(lat=lat, lon=lon, time=time, **kwargs)
        return synthetic_profile()

    monkeypatch.setattr(sounding, "era5_profile", fake_era5)
    prof = nexrad_volume.radarx.sounding(era5_source="gcs")
    assert calls["lat"] == pytest.approx(33.654, abs=1e-3)  # KLBB
    assert calls["source"] == "gcs"
    assert str(calls["time"]).startswith("2016-06-01T15:00")
    assert prof.attrs["radar_altitude"] > 900

    def fake_read(station, time, source="iem", **kwargs):
        calls.update(station=station, synoptic=time, archive=source)
        return synthetic_profile()

    monkeypatch.setattr(sounding, "read_sounding", fake_read)
    nexrad_volume.radarx.sounding("iem")
    near = sounding.nearest_station(33.654, -101.814, "2016-06-01", source="iem")
    assert calls["station"] == str(near["iem_id"].values[0])
    assert calls["archive"] == "iem"
    assert calls["synoptic"] == np.datetime64("2016-06-01T12:00")
    with pytest.raises(ValueError):
        nexrad_volume.radarx.sounding("model")


def test_grid_background_accessor():
    grid = _grid()
    bg = grid.radarx.background(synthetic_profile())
    assert "freezing_level" in bg


# ---------------------------------------------------------------------------
# network (skipped when offline)


@network("mesonet.agron.iastate.edu")
def test_iem_download(tmp_path, monkeypatch):
    monkeypatch.setenv("RADARX_CACHE_DIR", str(tmp_path))
    ds = sounding.read_sounding("JAN", "2022-03-31T00:00", source="iem")
    assert ds.attrs["station"] == "KJAN"
    assert 3500 < float(sounding.isotherm_height(ds)) < 4200


@network("weather.uwyo.edu")
def test_uwyo_download(tmp_path, monkeypatch):
    monkeypatch.setenv("RADARX_CACHE_DIR", str(tmp_path))
    try:
        ds = sounding.read_sounding(72235, "2022-03-31T00:00", source="uwyo")
    except Exception as err:  # the archive is sometimes overloaded
        pytest.skip(f"University of Wyoming archive unavailable: {err}")
    assert ds.sizes["height"] > 1000
    assert 3500 < float(sounding.isotherm_height(ds)) < 4200


@network("www.ncei.noaa.gov")
def test_igra2_download_current_year(tmp_path, monkeypatch):
    monkeypatch.setenv("RADARX_CACHE_DIR", str(tmp_path))
    import datetime as dt

    day = dt.datetime.now(dt.UTC).date() - dt.timedelta(days=20)
    if day.year != dt.datetime.now(dt.UTC).year:
        pytest.skip("too early in the year for the current-year IGRA2 file")
    try:
        ds = sounding.read_sounding("KJAN", f"{day}T00:00", source="igra2")
    except ValueError as err:
        pytest.skip(f"no IGRA2 sounding: {err}")
    assert ds.attrs["station"] == "USM00072235"
    assert ds.sizes["height"] > 20


@pytest.mark.skipif(not sounding._cds_available(), reason="no CDS credentials")
@network("cds.climate.copernicus.eu")
def test_era5_arco_point(tmp_path, monkeypatch):
    monkeypatch.setenv("RADARX_CACHE_DIR", str(tmp_path))
    prof = sounding.era5_profile(33.8967, -88.3289, "2022-03-31T00:00", source="arco")
    assert prof.sizes["height"] == 13
    assert 3300 < float(sounding.isotherm_height(prof)) < 4000


@pytest.mark.skipif(
    not os.environ.get("RADARX_TEST_ERA5_GCS"),
    reason="reads ~600 MB; set RADARX_TEST_ERA5_GCS=1 to run",
)
@network("storage.googleapis.com")
def test_era5_gcs_point(tmp_path, monkeypatch):
    pytest.importorskip("zarr")
    pytest.importorskip("aiohttp")
    monkeypatch.setenv("RADARX_CACHE_DIR", str(tmp_path))
    prof = sounding.era5_profile(
        33.8967,
        -88.3289,
        "2022-03-31T00:00",
        source="gcs",
        time_interpolation="nearest",
    )
    assert prof.sizes["height"] == 37
    assert 3300 < float(sounding.isotherm_height(prof)) < 4000


# ---------------------------------------------------------------------------
# providers and downloads with mocked network access


def _raw_cds(
    times, lat, lon, levels=(1000.0, 850.0, 500.0, 200.0), level_dim="pressure_level"
):
    """A download as the CDS returns it (cfgrib names, descending latitude)."""
    lev = np.asarray(levels)
    H = 8000.0 * np.log(1000.0 / lev)
    shape = (len(times), lev.size, len(lat), len(lon))
    z = np.broadcast_to(H[None, :, None, None] * 9.80665, shape)
    t = np.broadcast_to((300 - 0.0065 * H)[None, :, None, None], shape)
    dims = ("valid_time", level_dim, "latitude", "longitude")
    return xr.Dataset(
        {
            "z": (dims, z),
            "t": (dims, t),
            "q": (dims, np.full(shape, 0.004)),
            "u": (dims, np.full(shape, 5.0)),
            "v": (dims, np.full(shape, -2.0)),
            "w": (dims, np.full(shape, 0.0)),
        },
        coords={
            "valid_time": np.array(times, dtype="datetime64[ns]"),
            level_dim: lev,
            "latitude": np.asarray(lat, dtype=float),
            "longitude": np.asarray(lon, dtype=float),
            "number": 0,
            "expver": "0001",
        },
    )


def test_era5_cds_request(tmp_path, monkeypatch):
    monkeypatch.setenv("RADARX_CACHE_DIR", str(tmp_path))
    seen = {}

    def fake_retrieve(dataset, request, path):
        seen.update(dataset=dataset, request=request)
        times = [np.datetime64("2022-03-30T23:00"), np.datetime64("2022-03-31T00:00")]
        return _raw_cds(
            times, np.arange(34.5, 33.24, -0.25), np.arange(-89.0, -87.74, 0.25)
        )

    monkeypatch.setattr(sounding, "_cds_retrieve", fake_retrieve)
    prof = sounding.era5_profile(33.9, -88.33, "2022-03-30T23:30", source="cds")
    req = seen["request"]
    assert seen["dataset"] == "reanalysis-era5-pressure-levels"
    assert req["time"] == ["00:00", "23:00"] and req["day"] == ["30", "31"]
    assert len(req["pressure_level"]) == 37
    north, west, south, east = req["area"]
    assert south < 33.9 < north and west < -88.33 < east
    assert prof.attrs["era5_source"] == "cds"
    np.testing.assert_allclose(prof["u"], 5.0)
    # cached: no second request
    seen.clear()
    sounding.era5_profile(33.9, -88.33, "2022-03-30T23:30", source="cds")
    assert not seen


def test_era5_arco_request(tmp_path, monkeypatch):
    monkeypatch.setenv("RADARX_CACHE_DIR", str(tmp_path))
    seen = {}

    def fake_retrieve(dataset, request, path):
        seen.update(dataset=dataset, request=request)
        times = np.arange(
            np.datetime64("2022-03-30T00", "h"),
            np.datetime64("2022-03-31T19", "h"),
            np.timedelta64(6, "h"),
        )
        raw = _raw_cds(times, [34.0], [271.75], level_dim="pressureLevel")
        return raw.isel(latitude=0, longitude=0)  # point: scalar coordinates

    monkeypatch.setattr(sounding, "_cds_retrieve", fake_retrieve)
    monkeypatch.setattr(sounding, "_cds_available", lambda: True)
    prof = sounding.era5_profile(33.9, -88.33, "2022-03-30T23:46", source="arco")
    assert seen["dataset"] == "reanalysis-era5-pressure-levels-timeseries"
    assert seen["request"]["location"] == {"latitude": 33.9, "longitude": -88.33}
    assert seen["request"]["date"] == ["2022-03-30/2022-03-31"]
    assert prof.attrs["era5_times"] == "2022-03-30T18:00:00, 2022-03-31T00:00:00"
    assert prof.attrs["era5_grid_point"] == "34.00N -88.25E (nearest)"
    np.testing.assert_allclose(prof["v"], -2.0)
    # an area request (grid columns)
    sounding._era5_arco(
        (33.0, 34.0, -89.0, -88.0), [np.datetime64("2022-03-31")], tmp_path / "x.nc"
    )
    assert seen["request"]["area"] == [34.0, -89.0, 33.0, -88.0]


def test_era5_gcs_lazy_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("RADARX_CACHE_DIR", str(tmp_path))
    times = np.array(
        ["2022-03-30T23:00", "2022-03-31T00:00", "2022-03-31T01:00"],
        dtype="datetime64[ns]",
    )
    lat = np.arange(45.0, 24.9, -0.25)  # descending, as in the store
    lon = np.arange(0.0, 360.0, 0.25)
    raw = _raw_cds(times, lat, lon, levels=(1000.0, 500.0, 200.0))
    store = xr.Dataset(
        {
            "geopotential": raw["z"],
            "temperature": raw["t"],
            "specific_humidity": raw["q"],
            "u_component_of_wind": raw["u"],
            "v_component_of_wind": raw["v"],
            "vertical_velocity": raw["w"],
        }
    ).rename(valid_time="time", pressure_level="level")
    calls = []

    def fake_open_zarr(url, **kwargs):
        calls.append((url, kwargs))
        return store

    monkeypatch.setattr(xr, "open_zarr", fake_open_zarr)
    prof = sounding.era5_profile(33.9, -88.33, "2022-03-31T00:00", source="gcs")
    assert calls[0][0] == sounding.ARCO_ERA5_ZARR
    assert calls[0][0].startswith("https://storage.googleapis.com/")
    assert prof.sizes["height"] == 3
    np.testing.assert_allclose(prof["temperature"].values[0], 300.0)
    # the cached region serves a nearby grid without reopening the store
    bg = sounding.era5_column(_grid(), source="gcs", time_interpolation="nearest")
    assert len(calls) == 1
    speed = np.hypot(bg["u"], bg["v"]).isel(z=0)
    np.testing.assert_allclose(speed, np.hypot(5, 2), rtol=1e-5)


def test_cds_available(monkeypatch, tmp_path):
    pytest.importorskip("cdsapi")
    monkeypatch.setenv("CDSAPI_KEY", "x")
    assert sounding._cds_available()
    monkeypatch.delenv("CDSAPI_KEY")
    monkeypatch.delenv("ECMWF_DATASTORES_KEY", raising=False)
    monkeypatch.setenv("CDSAPI_RC", str(tmp_path / "missing"))
    monkeypatch.setattr(sounding.Path, "home", lambda: tmp_path)
    assert not sounding._cds_available()
    assert sounding._resolve_source("auto", True) == "gcs"
    monkeypatch.setattr(sounding, "_cds_available", lambda: True)
    assert sounding._resolve_source("auto", True) == "cds"
    assert sounding._resolve_source("auto", False) == "cds"
    assert sounding._resolve_source("arco", True) == "arco"


def test_resolve_source_errors():
    with pytest.raises(ValueError):
        sounding._resolve_source("ncep", True)
    with pytest.raises(ValueError):
        sounding._time_bracket("2022-03-31", 1, "cubic")


@pytest.mark.parametrize(
    "source, sample",
    [("iem", "iem_KJAN_202203310000.json"), ("uwyo", "uwyo_72235_2022033100.csv")],
)
def test_read_sounding_mocked(tmp_path, monkeypatch, source, sample):
    seen = {}

    def fake_download(url, fname, subdir, cache=True):
        seen.update(url=url, fname=fname)
        target = tmp_path / fname
        target.write_bytes((DATA / sample).read_bytes())
        return target

    monkeypatch.setattr(sounding, "_download", fake_download)
    ds = sounding.read_sounding("JAN", "2022-03-31T00:00", source=source)
    assert ds.sizes["height"] > 50
    if source == "iem":
        assert "station=KJAN" in seen["url"] and "ts=202203310000" in seen["url"]
    else:
        assert "id=72235" in seen["url"] and "2022-03-31%2000:00:00" in seen["url"]

    def empty_download(url, fname, subdir, cache=True):
        target = tmp_path / fname
        target.write_text(
            '{"profiles": []}' if source == "iem" else "<html>no data</html>"
        )
        return target

    monkeypatch.setattr(sounding, "_download", empty_download)
    with pytest.raises(ValueError):
        sounding.read_sounding("JAN", "2022-03-31T00:00", source=source)
    assert not (tmp_path / seen["fname"]).exists()  # a bad response is not cached


def test_read_igra2_mocked(tmp_path, monkeypatch):
    import zipfile

    urls = []

    def fake_download(url, fname, subdir, cache=True):
        urls.append(url)
        if "data-y2d" in url:
            raise OSError("404")
        path = tmp_path / fname
        with zipfile.ZipFile(path, "w") as z:
            z.write(DATA / "igra2_USM00072235_20220330-31.txt", "USM00072235-data.txt")
        return path

    monkeypatch.setattr(sounding, "_download", fake_download)
    ds = sounding.read_sounding("KJAN", "2022-03-31T00:00", source="igra2")
    assert ds.attrs["station"] == "USM00072235"
    assert "data-y2d/USM00072235-data-beg2022" in urls[0] and "data-por" in urls[1]
    with pytest.raises(ValueError):
        sounding.read_sounding("KJAN", "2022-04-01T00:00", source="igra2")
    with pytest.raises(ValueError):
        sounding.read_sounding("KJAN", "2022-04-01T00:00", source="ruc")
    with pytest.raises(ValueError):
        sounding.read_sounding("ZZZZZ9", "2022-04-01T00:00", source="igra2")


def test_download_uses_cache_dir(tmp_path, monkeypatch):
    import pooch

    monkeypatch.setenv("RADARX_CACHE_DIR", str(tmp_path))

    def fake_retrieve(url, known_hash, fname, path, progressbar):
        out = Path(path) / fname
        assert not out.exists()  # cache=False removed the old file
        out.write_text("x")
        return str(out)

    monkeypatch.setattr(pooch, "retrieve", fake_retrieve)
    (tmp_path / "soundings" / "iem").mkdir(parents=True)
    (tmp_path / "soundings" / "iem" / "f.json").write_text("old")
    path = sounding._download("https://example.org/f", "f.json", "iem", cache=False)
    assert path == tmp_path / "soundings" / "iem" / "f.json"
    assert path.read_text() == "x"


def test_input_variants_and_errors():
    import datetime as dt

    prof = synthetic_profile()
    t = xr.DataArray(np.datetime64("2022-03-31T00:00"))
    assert sounding._to_datetime64(t) == np.datetime64("2022-03-31T00:00")
    aware = dt.datetime(2022, 3, 31, 1, tzinfo=dt.timezone(dt.timedelta(hours=1)))
    assert sounding._to_datetime64(aware) == np.datetime64("2022-03-31T00:00")
    assert sounding._to_datetime64("2022-03-31T00:00Z") == np.datetime64(
        "2022-03-31T00:00"
    )
    with pytest.raises(ValueError):
        sounding._to_datetime64("NaT")
    # heights from a Dataset (a grid) and from a plain list
    grid = _grid(nz=5)
    out = sounding.interpolate_profile(prof, grid, ["temperature"])
    assert out["temperature"].dims == ("z",)
    out = sounding.interpolate_profile(prof, [1000.0, 2000.0], ["wind_speed"])
    np.testing.assert_allclose(out["u"], [3.0, 4.0])
    with pytest.raises(ValueError):
        sounding.interpolate_profile(prof.rename(height="level"), [1000.0])
    with pytest.raises(ValueError):
        sounding.isotherm_height(prof, which="middle")
    with pytest.raises(ValueError):
        sounding._kernel("numpy").layer_mean(
            np.zeros((1, 2)), np.zeros((1, 1, 2)), 5.0, 1.0
        )
    with pytest.raises(ValueError):
        sounding.open_sounding_file(DATA / "uwyo_72235_2022033100.csv", format="netcdf")
    with pytest.raises(ValueError):
        sounding._thermo("bogus", [1.0], engine="numpy")
    with pytest.raises(ValueError):
        sounding._grid_time(grid.drop_vars("time"))
    # relative humidity only: dew point from RH
    rh = sounding._profile_dataset(
        height=np.array([0.0, 1000.0]),
        pressure=np.array([1e5, 9e4]),
        temperature=np.array([290.0, 284.0]),
        relative_humidity=np.array([1.0, 0.5]),
    )
    assert float(rh["dewpoint"][0]) == pytest.approx(290.0)
    # a grid without crs_wkt uses the radar site
    site_grid = grid.drop_vars("crs_wkt").assign_coords(
        latitude=33.8967, longitude=-88.3289
    )
    lon, lat, _ = sounding._grid_lonlat_rotation(site_grid)
    assert lat[grid.sizes["y"] // 2, grid.sizes["x"] // 2] == pytest.approx(33.8967)


def test_csv_needs_temperature(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("pres,hght\n1000,100\n")
    with pytest.raises(ValueError):
        sounding.open_sounding_file(path)
    path.write_text("p,z,t,u,v\n1000,100,20,1,2\n900,1000,15,3,4\n")
    ds = sounding.open_sounding_file(path, format="csv")
    np.testing.assert_allclose(ds["u"], [1, 3])


def test_volume_time_fallbacks():
    root = xr.Dataset(
        coords={"latitude": 33.9, "longitude": -88.3},
        attrs={"time_coverage_start": "2022-03-30T23:46:39Z"},
    )
    site, t = sounding._volume_site_time(xr.DataTree(root))
    assert t == np.datetime64("2022-03-30T23:46:39")
    assert "altitude" not in site
    with pytest.raises(ValueError):
        sounding._volume_site_time(xr.DataTree(xr.Dataset()))
    no_time = xr.Dataset(coords={"latitude": 1.0, "longitude": 2.0})
    with pytest.raises(ValueError):
        sounding._volume_site_time(xr.DataTree(no_time))


def test_small_branches(tmp_path, monkeypatch):
    import pooch

    # IEM file with a time filter
    path = DATA / "iem_KJAN_202203310000.json"
    ds = sounding.open_sounding_file(path, time="2022-03-31T00:00", station="KJAN")
    assert ds.attrs["station"] == "KJAN"
    with pytest.raises(ValueError):
        sounding.open_sounding_file(path, time="2022-03-31T12:00")
    # identifiers not in the table
    assert sounding._station_id("kxyz", "iem") == "KXYZ"
    assert sounding._station_id("99999", "uwyo") == "99999"
    # default cache directory
    monkeypatch.delenv("RADARX_CACHE_DIR", raising=False)
    monkeypatch.setattr(pooch, "os_cache", lambda name: tmp_path / name)
    assert sounding._cache_dir("x") == tmp_path / "radarx" / "soundings" / "x"
    # a point box is widened to one grid cell
    south, north, west, east = sounding._era5_box(33.0, -88.0, 0.0)
    assert north - south == 0.25 and east - west == 0.25
    # an unusable cached file name is ignored
    (sounding._cache_dir("era5") / "era5_gcs_20220331T000000_bad.nc").write_text("")
    assert sounding._gcs_cached((33, 34, -89, -88), np.datetime64("2022-03-31")) is None
    # column profiles broadcast over heights without the column dimension
    prof = xr.concat([synthetic_profile(lapse=g) for g in (0.005, 0.008)], "x")
    out = sounding.interpolate_profile(prof, xr.DataArray([1000.0], dims="gate"))
    np.testing.assert_allclose(out["temperature"].sel(gate=0), [295.0, 292.0])
    # the compiled engine cannot be forced without the kernel
    monkeypatch.setattr(sounding, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError):
        sounding.saturation_vapor_pressure(273.15, engine="compiled")
    np.testing.assert_allclose(sounding.saturation_vapor_pressure(273.15), 611.2)
