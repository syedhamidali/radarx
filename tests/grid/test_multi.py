#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for multi-radar gridding, merging and network calibration
===============================================================
"""
import numpy as np
import pyproj
import pytest
import xarray as xr
from xradar.georeference import antenna_to_cartesian

import radarx  # noqa: F401
from radarx.grid import cone, multi

ENGINES = ["numpy"] + (["compiled"] if multi.HAS_COMPILED_KERNEL else [])
needs_kernel = pytest.mark.skipif(
    not multi.HAS_COMPILED_KERNEL, reason="compiled kernels not built"
)
ORIGIN = (35.0, -90.0)
ELEVATIONS = (0.5, 1.5, 2.5, 4.0, 6.0, 9.0)
T0 = np.datetime64("2022-03-30T23:46:00", "ns")


def _aeqd(lat, lon):
    return pyproj.Proj(proj="aeqd", lat_0=lat, lon_0=lon, datum="WGS84")


def truth(x, y, z):
    """Smooth analytic field in the grid frame (x, y, z in metres)."""
    return 30.0 + 1e-4 * x - 5e-5 * y - 2e-3 * (z - 2000.0)


def truth_zdr(x, y, z):
    return 1.0 + 5e-6 * x + 1e-5 * y


def _volume(
    lat,
    lon,
    alt=100.0,
    bias=0.0,
    zdr_bias=0.0,
    name="RAD",
    elevations=ELEVATIONS,
    start=T0,
    fields=("DBZH", "ZDR", "VRADH", "RHOHV"),
    max_range=80e3,
):
    """Synthetic volume measuring ``truth`` at each gate's grid position."""
    rng_m = np.arange(250.0, max_range, 500.0)
    azimuth = np.arange(0.5, 360.0, 1.0)
    radar, grid = _aeqd(lat, lon), _aeqd(*ORIGIN)
    sweeps = {}
    for k, el in enumerate(elevations):
        az = np.roll(azimuth, 23 * k)
        gx, gy, gz = antenna_to_cartesian(
            rng_m[None, :], az[:, None], np.full((az.size, 1), el), site_altitude=alt
        )
        glon, glat = radar(gx, gy, inverse=True)
        X, Y = grid(glon, glat)
        data = {
            "DBZH": truth(X, Y, gz) + bias,
            "ZDR": truth_zdr(X, Y, gz) + zdr_bias,
            "VRADH": np.sin(np.deg2rad(az))[:, None] * np.ones_like(gz),
            "RHOHV": np.where(gx > 0, 0.99, 0.5) * np.ones_like(gz),
        }
        units = {"DBZH": "dBZ", "ZDR": "dB", "VRADH": "m/s", "RHOHV": "1"}
        sweeps[f"sweep_{k}"] = xr.Dataset(
            {f: (("azimuth", "range"), data[f], {"units": units[f]}) for f in fields},
            coords={
                "azimuth": az,
                "range": rng_m,
                "elevation": ("azimuth", np.full(az.size, el)),
                "time": ("azimuth", np.full(az.size, start)),
            },
        )
    root = xr.Dataset(coords={"latitude": lat, "longitude": lon, "altitude": alt})
    root.attrs["instrument_name"] = name
    return xr.DataTree.from_dict({"/": root, **sweeps})


GRID = dict(
    x=np.arange(-60e3, 60e3 + 1, 3e3),
    y=np.arange(-50e3, 50e3 + 1, 3e3),
    z=np.arange(1000.0, 5000.0 + 1, 500.0),
)


def _site(x, y):
    lon, lat = _aeqd(*ORIGIN)(x, y, inverse=True)
    return float(lat), float(lon)


def _sites():
    return ORIGIN, _site(50e3, 10e3), _site(-20e3, -40e3)


@pytest.fixture(scope="module")
def network():
    (la1, lo1), (la2, lo2), (la3, lo3) = _sites()
    return [
        _volume(la1, lo1, 100.0, name="AAA"),
        _volume(la2, lo2, 250.0, bias=2.5, zdr_bias=-0.3, name="BBB"),
        _volume(la3, lo3, 50.0, bias=-1.5, zdr_bias=0.2, name="CCC"),
    ]


@pytest.mark.parametrize("engine", ENGINES)
def test_single_radar_reproduces_grid_cones(engine):
    dtree = _volume(*ORIGIN)
    a = multi.grid_radars([dtree], **GRID, data_vars="DBZH", engine=engine)
    b = cone.grid_cones(dtree, "DBZH", **GRID, engine=engine)
    np.testing.assert_allclose(
        a.DBZH.isel(radar=0).values, b.DBZH.values, atol=1e-4, equal_nan=True
    )
    assert float(a.radar_x[0]) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("engine", ENGINES)
def test_every_radar_measures_the_truth(network, engine):
    out = multi.grid_radars(
        network, **GRID, origin=ORIGIN, data_vars=["DBZH", "ZDR"], engine=engine
    )
    assert out.DBZH.dims == ("radar", "z", "y", "x")
    assert list(out.radar.values) == ["AAA", "BBB", "CCC"]
    x, y, z = xr.broadcast(out.x, out.y, out.z)
    expected = truth(x, y, z).transpose("z", "y", "x")
    for r, b in enumerate((0.0, 2.5, -1.5)):
        field = out.DBZH.isel(radar=r)
        ok = np.isfinite(field.values)
        assert ok.mean() > 0.2
        # cone gridding of a linear field: only the beam curvature error
        err = (field - b - expected).values[ok]
        assert np.abs(err).max() < 0.1


def test_radar_positions(network):
    out = multi.grid_radars(network, **GRID, origin=ORIGIN, data_vars="DBZH")
    np.testing.assert_allclose(out.radar_x.values, [0, 50e3, -20e3], atol=1e-3)
    np.testing.assert_allclose(out.radar_y.values, [0, 10e3, -40e3], atol=1e-3)
    np.testing.assert_allclose(out.radar_z.values, [100.0, 250.0, 50.0])
    assert out.time.dims == ("radar",)
    assert out.lat.dims == ("y", "x")
    assert out.attrs["origin_latitude"] == ORIGIN[0]


@pytest.mark.parametrize("engine", ENGINES)
def test_beam_geometry_matches_antenna_coordinates(engine):
    """Range, elevation and azimuth at gates of an off-origin radar."""
    lat, lon = _site(40e3, -30e3)
    alt = 300.0
    radar, grid = _aeqd(lat, lon), _aeqd(*ORIGIN)
    rng = np.array([20e3, 60e3, 120e3])
    az = np.array([30.0, 200.0, 310.0])
    el = np.array([0.5, 3.0, 10.0])
    gx, gy, gz = antenna_to_cartesian(rng, az, el, site_altitude=alt)
    glon, glat = radar(gx, gy, inverse=True)
    X, Y = grid(glon, glat)
    sites = [{"latitude": lat, "longitude": lon}]
    for k in range(3):
        cols = multi._column_geometry(np.array([X[k]]), np.array([Y[k]]), ORIGIN, sites)
        z = np.array([gz[k]])
        if engine == "compiled":
            r_out, el_out = multi._multi.beam_geometry(cols["ground"], z, [alt])
        else:
            r_out, el_out = multi._geometry_numpy(cols["ground"], z, [alt])
        assert float(r_out.ravel()[0]) == pytest.approx(rng[k], abs=0.05)
        assert float(cols["antenna_azimuth"][0].ravel()[0]) == pytest.approx(
            az[k], abs=1e-6
        )
        theta = np.degrees(cols["ground"][0].ravel()[0] / (4 / 3 * multi.EARTH_RADIUS))
        assert float(el_out.ravel()[0]) - theta == pytest.approx(el[k], abs=1e-4)

        # beam direction at the cell: from this gate to the next one in the grid
        nx_, ny_, nz_ = antenna_to_cartesian(
            rng[k] + 50.0, az[k], el[k], site_altitude=alt
        )
        nlon, nlat = radar(nx_, ny_, inverse=True)
        X2, Y2 = grid(nlon, nlat)
        heading = np.degrees(np.arctan2(X2 - X[k], Y2 - Y[k])) % 360
        assert float(cols["azimuth"][0].ravel()[0]) == pytest.approx(heading, abs=0.01)
        # x and y are distances at sea level: scale them to the beam's height
        R = 4 / 3 * multi.EARTH_RADIUS
        stretch = (R + gz[k]) / R
        horizontal = np.hypot(X2 - X[k], Y2 - Y[k]) * stretch
        local = np.degrees(np.arctan2(nz_ - gz[k], horizontal))
        assert float(el_out.ravel()[0]) == pytest.approx(local, abs=0.01)


def test_radial_velocity_projection_is_consistent(network):
    """The documented projection reproduces a radial wind from the geometry."""
    out = multi.grid_radars(network, **GRID, origin=ORIGIN, data_vars="DBZH")
    az = np.deg2rad(out.azimuth)
    el = np.deg2rad(out.elevation)
    unit = xr.concat(
        [np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)], "c"
    )
    np.testing.assert_allclose(((unit**2).sum("c")).values, 1.0, atol=1e-5)
    # the beam points away from the radar
    dx = out.x - out.radar_x
    dy = out.y - out.radar_y
    dot = (np.sin(az) * dx + np.cos(az) * dy) / np.hypot(dx, dy)
    assert float(dot.where(np.hypot(dx, dy) > 5e3).min()) > 0.999


@pytest.mark.parametrize("engine", ENGINES)
def test_network_bias_recovers_offsets(network, engine):
    out = multi.grid_radars(
        network, **GRID, origin=ORIGIN, data_vars=["DBZH", "ZDR"], engine=engine
    )
    bias = multi.network_bias(out, "DBZH", reference="AAA", engine=engine)
    np.testing.assert_allclose(bias.bias.values, [0.0, 2.5, -1.5], atol=0.05)
    assert float(bias.pair_bias.sel(radar="BBB", other="AAA")) == pytest.approx(
        2.5, abs=0.05
    )
    assert float(bias.pair_bias.sel(radar="AAA", other="BBB")) == pytest.approx(
        -2.5, abs=0.05
    )
    assert int(bias.pair_count.sel(radar="AAA", other="BBB")) > 200
    assert np.nanmax(np.abs(bias.pair_residual.values)) < 0.05
    zdr = multi.network_bias(out, "ZDR", reference=1, engine=engine)
    np.testing.assert_allclose(zdr.bias.values, [0.3, 0.0, 0.5], atol=0.02)
    assert zdr.attrs["reference"] == "BBB"
    # restricted to a height range and to similar ranges: still unbiased
    sub = multi.network_bias(
        out, "DBZH", z_range=(1000, 3000), max_range_ratio=1.5, max_range=70e3
    )
    np.testing.assert_allclose(sub.bias.values, [0.0, 2.5, -1.5], atol=0.05)


def test_calibrate_and_merge(network):
    out = multi.grid_radars(
        network,
        **GRID,
        origin=ORIGIN,
        data_vars=["DBZH", "ZDR"],
        calibrate=["DBZH", "ZDR"],
        merge=["DBZH", "ZDR"],
    )
    np.testing.assert_allclose(out.DBZH_bias.values, [0.0, 2.5, -1.5], atol=0.05)
    x, y, z = xr.broadcast(out.x, out.y, out.z)
    expected = truth(x, y, z).transpose("z", "y", "x")
    merged = out.DBZH_merged
    assert merged.dims == ("z", "y", "x")
    ok = np.isfinite(merged.values)
    # the merged area covers more than any single radar
    assert ok.sum() > np.isfinite(out.DBZH.isel(radar=0).values).sum()
    assert np.abs((merged - expected).values[ok]).max() < 0.15
    assert np.isfinite(out.ZDR_merged.values).any()


def test_merge_without_calibration_is_between_radars(network):
    out = multi.grid_radars(network, **GRID, origin=ORIGIN, data_vars="DBZH")
    merged = multi.merge_radars(out, "DBZH")
    lo = out.DBZH.min("radar")
    hi = out.DBZH.max("radar")
    ok = np.isfinite(merged.DBZH.values)
    assert (merged.DBZH.values[ok] >= lo.values[ok] - 1e-4).all()
    assert (merged.DBZH.values[ok] <= hi.values[ok] + 1e-4).all()
    assert (merged.DBZH_weight.values[ok] > 0).all()
    assert "radar" not in merged.dims
    # with a short range scale, the value of the nearest radar dominates
    near = dict(x=0.0, y=10e3, z=1000.0)
    short = multi.merge_radars(out, "DBZH", range_scale=10e3)
    assert np.isfinite(out.DBZH.sel(**near)).all()
    assert float(short.DBZH.sel(**near)) == pytest.approx(
        float(out.DBZH.isel(radar=0).sel(**near)), abs=0.01
    )
    # a bias Dataset is subtracted before merging
    bias = multi.network_bias(out, "DBZH")
    fixed = multi.merge_radars(out, "DBZH", bias=bias, range_scale=None)
    x, y, z = xr.broadcast(out.x, out.y, out.z)
    expected = truth(x, y, z).transpose("z", "y", "x")
    ok = np.isfinite(fixed.DBZH.values)
    assert np.abs((fixed.DBZH - expected).values[ok]).max() < 0.15
    fixed = multi.merge_radars(out, ["DBZH"], bias={"DBZH": bias.bias}, beamwidth=0.9)
    assert np.isfinite(fixed.DBZH.values).any()


def test_time_weight_prefers_the_recent_radar():
    (la1, lo1), (la2, lo2), _ = _sites()
    a = _volume(la1, lo1, name="AAA")
    b = _volume(la2, lo2, bias=10.0, name="BBB", start=T0 + np.timedelta64(600, "s"))
    out = multi.grid_radars([a, b], **GRID, origin=ORIGIN, data_vars="DBZH")
    early = multi.merge_radars(out, "DBZH", time=T0, range_scale=None)
    late = multi.merge_radars(
        out, "DBZH", time=T0 + np.timedelta64(600, "s"), range_scale=None
    )
    both = np.isfinite(out.DBZH).all("radar")
    diff = (late.DBZH - early.DBZH).where(both)
    assert float(diff.mean()) > 5.0


@pytest.mark.parametrize("engine", ENGINES)
def test_advection_to_the_analysis_time(engine):
    (la1, lo1), (la2, lo2), _ = _sites()
    a = _volume(la1, lo1, name="AAA")
    b = _volume(la2, lo2, name="BBB", start=T0 + np.timedelta64(120, "s"))
    kw = dict(origin=ORIGIN, data_vars="DBZH", engine=engine)
    still = multi.grid_radars([a, b], **GRID, **kw)
    moved = multi.grid_radars([a, b], **GRID, **kw, motion=(12.5, 0.0), time=T0)
    # radar BBB is moved back by 120 s * 12.5 m/s = 1500 m = half a cell
    shift = still.DBZH.isel(radar=1).shift(x=-1)
    expected = 0.5 * (still.DBZH.isel(radar=1) + shift)
    got = moved.DBZH.isel(radar=1)
    ok = np.isfinite(got.values) & np.isfinite(expected.values)
    np.testing.assert_allclose(got.values[ok], expected.values[ok], atol=1e-3)
    np.testing.assert_allclose(
        moved.DBZH.isel(radar=0).values, still.DBZH.isel(radar=0).values
    )
    assert moved.attrs["analysis_time"].startswith("2022-03-30T23:46:00")
    # a motion Dataset (as from estimate_motion) works the same way
    motion = xr.Dataset({"u": 12.5, "v": 0.0})
    again = multi.grid_radars([a, b], **GRID, **kw, motion=motion)
    np.testing.assert_allclose(again.DBZH.values, moved.DBZH.values, equal_nan=True)


@needs_kernel
def test_engines_agree(network):
    kw = dict(origin=ORIGIN, data_vars=["DBZH", "ZDR", "VRADH"], merge="DBZH")
    a = multi.grid_radars(network, **GRID, **kw, engine="compiled", fill_below=True)
    b = multi.grid_radars(network, **GRID, **kw, engine="numpy", fill_below=True)
    for name in ("DBZH", "ZDR", "VRADH", "azimuth", "elevation", "range"):
        np.testing.assert_allclose(
            a[name].values, b[name].values, rtol=1e-6, atol=1e-3, equal_nan=True
        )
    np.testing.assert_allclose(
        a.DBZH_merged.values, b.DBZH_merged.values, atol=1e-4, equal_nan=True
    )
    ba = multi.network_bias(a, "DBZH", engine="compiled", max_range_ratio=2.0)
    bb = multi.network_bias(a, "DBZH", engine="numpy", max_range_ratio=2.0)
    np.testing.assert_array_equal(ba.pair_count.values, bb.pair_count.values)
    np.testing.assert_allclose(ba.bias.values, bb.bias.values, atol=1e-9)


@needs_kernel
def test_float32_and_float64_sweeps_agree():
    dtree = _volume(*ORIGIN, fields=("DBZH",))
    single = dtree.copy()
    for name in [n for n in single.children if n.startswith("sweep")]:
        ds = single[name].to_dataset()
        ds["DBZH"] = ds["DBZH"].astype(np.float32)
        single[name] = ds
    kw = dict(origin=ORIGIN, data_vars="DBZH", engine="compiled")
    a = multi.grid_radars([dtree, single], **GRID, **kw)
    np.testing.assert_allclose(
        a.DBZH.isel(radar=0).values, a.DBZH.isel(radar=1).values, atol=1e-4
    )
    c = multi.grid_radars([single], **GRID, origin=ORIGIN, data_vars="DBZH")
    d = multi.grid_radars(
        [single], **GRID, origin=ORIGIN, data_vars="DBZH", engine="numpy"
    )
    np.testing.assert_allclose(c.DBZH.values, d.DBZH.values, atol=1e-4, equal_nan=True)


@needs_kernel
def test_kernels_with_random_data():
    rnd = np.random.default_rng(1)
    values = rnd.normal(0, 3, (3, 4, 50)).astype(np.float32)
    values[rnd.random(values.shape) < 0.3] = np.nan
    ground = [rnd.uniform(0, 150e3, 50) for _ in range(3)]
    z = np.array([500.0, 1000.0, 3000.0, 8000.0])
    args = (
        values,
        ground,
        z,
        [10.0, 200.0, 0.0],
        [[0.5, 1.5, 2.4, 3.4], [0.5, 0.9, 1.3], []],
        [1.0, 0.9, 1.0],
        [0.0, 60.0, -400.0],
        40e3,
        300.0,
    )
    m1, w1 = multi._multi.merge(*args)
    m2, w2 = multi._merge_numpy(*args)
    np.testing.assert_allclose(m1, m2, atol=1e-5, equal_nan=True)
    np.testing.assert_allclose(w1, w2, rtol=1e-5)
    rng = rnd.uniform(1e3, 1e5, (3, 200)).astype(np.float32)
    vals = values.reshape(3, -1)
    for ratio in (0.0, 1.5):
        c1 = multi._multi.pair_histograms(vals, rng, ratio, 0.5, 6.0)
        c2 = multi._pair_histograms_numpy(vals, rng, ratio, 0.5, 6.0)
        for p, q in zip(c1, c2):
            np.testing.assert_allclose(p, q, rtol=1e-12)


def test_beam_weight():
    el = [0.5, 1.5, 2.5]
    w = multi._beam_weight_numpy(np.array([0.5, 1.0, 1.5, 3.5, -0.5, 0.0]), el, 1.0)
    np.testing.assert_allclose(w[[0, 2]], 1.0)
    assert w[1] == pytest.approx(0.005**0.125)  # half way: about 0.5
    assert w[3] == pytest.approx(0.005)  # one beamwidth above the top beam
    assert w[4] == pytest.approx(0.005)
    np.testing.assert_allclose(multi._beam_weight_numpy(np.array([3.0]), [], 1.0), 1)
    np.testing.assert_allclose(multi._beam_weight_numpy(np.array([3.0]), [1.0], 0), 1)


def test_missing_fields_qc_and_range(network):
    (la1, lo1), (la2, lo2), _ = _sites()
    no_zdr = _volume(la2, lo2, name="BBB", fields=("DBZH", "RHOHV"))
    out = multi.grid_radars(
        [network[0], no_zdr], **GRID, origin=ORIGIN, rhohv_min=0.8, max_range=40e3
    )
    assert set(out.data_vars) >= {"DBZH", "ZDR", "VRADH", "RHOHV"}
    assert np.isnan(out.ZDR.isel(radar=1).values).all()
    assert np.isnan(out.VRADH.isel(radar=1).values).all()
    # low RHOHV west of each radar is masked, RHOHV itself is kept
    west = out.DBZH.isel(radar=0).where(out.x < -5e3)
    assert np.isnan(west.values).all()
    assert np.isfinite(out.RHOHV.isel(radar=0).where(out.x < -5e3).values).any()
    assert not (out.DBZH.where(out.range > 40e3).notnull()).any()
    # pairs without overlap give NaN biases for the radar not linked
    bias = multi.network_bias(out, "ZDR")
    assert np.isnan(float(bias.bias[1])) and float(bias.bias[0]) == 0.0


def test_unlinked_radar_and_names():
    (la1, lo1), _, _ = _sites()
    far_lat, far_lon = _site(0.0, 500e3)
    a = _volume(la1, lo1, name="AAA")
    b = _volume(far_lat, far_lon, name="AAA", bias=3.0)
    c = _volume(la1, lo1, bias=1.0)
    c.attrs.pop("instrument_name", None)
    out = multi.grid_radars([a, b, c], **GRID, origin=ORIGIN, data_vars="DBZH")
    assert list(out.radar.values) == ["AAA", "AAA_1", "radar_2"]
    named = multi.grid_radars([a, c], **GRID, data_vars="DBZH", names=["one", "two"])
    assert list(named.radar.values) == ["one", "two"]
    bias = multi.network_bias(out, "DBZH")
    assert np.isnan(float(bias.bias[1]))
    assert float(bias.bias[2]) == pytest.approx(1.0, abs=0.05)
    alone = multi.network_bias(out.isel(radar=[1]), "DBZH")
    assert float(alone.bias[0]) == 0.0


def test_radar_name_fallback():
    dtree = _volume(*ORIGIN)
    dtree.attrs["instrument_name"] = "None"
    assert multi._radar_name(dtree, 3) == "radar_3"
    single = multi.grid_radars(dtree, **GRID, data_vars="DBZH", time=T0)
    assert single.sizes["radar"] == 1
    assert "analysis_time" in single.attrs
    assert np.isnat(multi._volume_time([xr.Dataset()]))
    nat = xr.Dataset({"time": ("t", np.array(["NaT"], dtype="datetime64[ns]"))})
    assert np.isnat(multi._volume_time([nat]))


def test_errors(network, monkeypatch):
    with pytest.raises(ValueError, match="at least one"):
        multi.grid_radars([], **GRID)
    with pytest.raises(ValueError, match="required"):
        multi.grid_radars(network, x=GRID["x"])
    with pytest.raises(ValueError, match="engine must be"):
        multi.grid_radars(network, **GRID, engine="gpu")
    with pytest.raises(ValueError, match="one name per radar"):
        multi.grid_radars(network, **GRID, names=["a"])
    with pytest.raises(ValueError, match="No sweep of any radar"):
        multi.grid_radars(network, **GRID, data_vars="KDP")
    out = multi.grid_radars(network[:2], **GRID, origin=ORIGIN, data_vars="DBZH")
    with pytest.raises(ValueError, match="grid has no"):
        multi.network_bias(out, "KDP")
    with pytest.raises(ValueError, match="unknown reference"):
        multi.network_bias(out, "DBZH", reference="XYZ")
    with pytest.raises(ValueError, match="out of range"):
        multi.network_bias(out, "DBZH", reference=5)
    with pytest.raises(ValueError, match="grid has no"):
        multi.merge_radars(out, "KDP")
    monkeypatch.setattr(multi, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError, match="compiled"):
        multi.grid_radars(network, **GRID, engine="compiled")


def test_accessors(network):
    out = multi.grid_radars(network, **GRID, origin=ORIGIN, data_vars="DBZH")
    bias = out.radarx.network_bias("DBZH", reference="AAA")
    np.testing.assert_allclose(bias.bias.values, [0.0, 2.5, -1.5], atol=0.05)
    merged = out.radarx.merge_radars("DBZH", bias=bias)
    assert merged.DBZH.dims == ("z", "y", "x")
    again = network[0].radarx.grid_radars(network[1:], **GRID, data_vars="DBZH")
    assert list(again.radar.values) == ["AAA", "BBB", "CCC"]
    pair = network[0].radarx.grid_radars(network[1], **GRID, data_vars="DBZH")
    assert list(pair.radar.values) == ["AAA", "BBB"]


def test_integer_fields_and_fields_no_radar_has():
    dtree = _volume(*ORIGIN, fields=("DBZH",))
    for name in [n for n in dtree.children if n.startswith("sweep")]:
        ds = dtree[name].to_dataset()
        ds["CLASS"] = ds["DBZH"].round().astype(np.int16)
        dtree[name] = ds
    out = multi.grid_radars(dtree, **GRID, data_vars=["CLASS", "KDP"])
    assert "KDP" not in out and out["CLASS"].dtype == np.float32
    assert np.isfinite(out["CLASS"].values).any()


@pytest.fixture(scope="module")
def nexrad():
    from open_radar_data import DATASETS
    from xradar.io import open_nexradlevel2_datatree

    path = DATASETS.fetch("KLBB20160601_150025_V06")
    dtree = open_nexradlevel2_datatree(path)
    for name in [n for n in dtree.children if n.startswith("sweep")]:
        ds = dtree[name].to_dataset()
        if "DBZH" in ds:
            ds["DBZH"] = ds["DBZH"].where(ds["DBZH"] > -32)
        dtree[name] = ds
    return dtree


def test_real_nexrad_volume(nexrad):
    """A real volume: the shared grid reproduces single-radar cone gridding."""
    grid = dict(
        x=np.arange(-100e3, 100e3 + 1, 2e3),
        y=np.arange(-100e3, 100e3 + 1, 2e3),
        z=np.arange(1000.0, 8000.0 + 1, 1000.0),
    )
    lat = float(nexrad.root["latitude"])
    lon = float(nexrad.root["longitude"])
    out = multi.grid_radars([nexrad], **grid, data_vars=["DBZH", "VRADH"])
    ref = cone.grid_cones(nexrad, ["DBZH", "VRADH"], **grid)
    for name in ("DBZH", "VRADH"):
        a, b = out[name].isel(radar=0).values, ref[name].values
        # geodesic and planar column positions differ by round-off only, which
        # can flip a handful of cells exactly at the edge of the data
        assert np.mean(np.isnan(a) != np.isnan(b)) < 1e-3
        both = np.isfinite(a) & np.isfinite(b)
        assert np.quantile(np.abs(a - b)[both], 0.999) < 1e-3
    assert np.isfinite(out.DBZH.values).mean() > 0.05
    assert float(out.radar_latitude[0]) == pytest.approx(lat)
    # the same volume seen from a shifted origin lands on the same places
    shifted = multi.grid_radars(
        [nexrad], **grid, data_vars="DBZH", origin=(lat + 0.1, lon), merge="DBZH"
    )
    assert float(shifted.radar_y[0]) == pytest.approx(-11.1e3, abs=100)
    assert np.isfinite(shifted.DBZH_merged.values).mean() > 0.05


@needs_kernel
@pytest.mark.parametrize("origin", [ORIGIN, (65.0, 25.0), (-33.0, 151.0)])
def test_column_geodesics_match_proj(origin):
    """Vincenty geodesics in the kernel against PROJ's azimuthal projections."""
    x = np.arange(-300e3, 300e3 + 1, 20e3)
    y = np.arange(-250e3, 250e3 + 1, 20e3)
    proj = _aeqd(*origin)
    sites = []
    for sx, sy in ((0.0, 0.0), (150e3, -40e3), (-220e3, 180e3), (20e3, 20e3)):
        lon, lat = proj(sx, sy, inverse=True)
        sites.append({"latitude": float(lat), "longitude": float(lon)})
    a = multi._column_geometry(x, y, origin, sites, use_compiled=True)
    b = multi._column_geometry(x, y, origin, sites, use_compiled=False)
    np.testing.assert_allclose(a["lat"], b["lat"], atol=1e-9)
    np.testing.assert_allclose(a["lon"], b["lon"], atol=1e-9)
    np.testing.assert_allclose(a["radar_x"], b["radar_x"])
    for r in range(len(sites)):
        np.testing.assert_allclose(a["ground"][r], b["ground"][r], atol=1e-3)
        away = b["ground"][r] > 1.0
        for key, tol in (("antenna_azimuth", 1e-6), ("azimuth", 2e-3)):
            diff = (a[key][r] - b[key][r] + 180.0) % 360.0 - 180.0
            assert np.abs(diff[away]).max() < tol, key
