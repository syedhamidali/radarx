#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the range-defined, columnar and RHI profiles and the melting layer on PPIs
====================================================================================

The sweeps are built from a known function of height: every gate holds F(h),
h the beam height of the gate, so a profile has to give back F. Sectors,
range limits and quality masks are checked by adding a large offset to the gates
that must not be used, and the intermediate-elevation bands of the columnar
profile by giving every sweep its own constant. The melting layer on PPIs is
checked on rings of known heights. The compiled kernels and the NumPy
implementations must agree.
"""

import warnings

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.retrieve import (
    cvp,
    mask_above_melting_layer,
    melting_layer,
    melting_layer_ppi,
    rd_qvp,
    rhi_qvp,
)
from radarx.retrieve import vertical_profiles as vp

from .provenance_helpers import without_provenance

ALT = 250.0
ENGINES = ["numpy"] + (["compiled"] if vp.HAS_COMPILED_KERNEL else [])
RNG = np.arange(125.0, 100_000.0, 250.0)


def make_sweep(
    elevation,
    funcs,
    nray=90,
    rng=RNG,
    offset=None,
    time="2026-01-01T00:00",
    elevations=None,
    azimuth0=0.0,
    azimuth_span=360.0,
):
    """Sweep whose gates hold ``funcs[name](beam height)`` (+ ``offset(az, r)``)."""
    az = azimuth0 + (np.arange(nray) + 0.5) * azimuth_span / nray
    el = np.full(nray, float(elevation)) if elevations is None else elevations
    from xradar.georeference import antenna_to_cartesian

    h = antenna_to_cartesian(
        rng[None, :], 0.0, el[:, None], earth_radius=vp.EARTH_RADIUS, site_altitude=ALT
    )[2]
    data = {}
    for name, f in funcs.items():
        values = np.asarray(f(h), dtype=np.float64)
        if offset is not None and name in offset:
            values = values + offset[name](
                az[:, None] * np.ones_like(h),
                rng[None, :] * np.ones_like(h),
                el[:, None] * np.ones_like(h),
            )
        units = {"DBZH": "dBZ", "ZDR": "dB", "RHOHV": "1", "KDP": "degrees/km"}.get(
            name, "1"
        )
        data[name] = (("azimuth", "range"), values.astype(np.float32), {"units": units})
    t = np.datetime64(time, "ns") + (np.arange(nray) * 50).astype("timedelta64[ms]")
    return xr.Dataset(
        data,
        coords={
            "azimuth": az,
            "range": rng,
            "elevation": ("azimuth", el),
            "time": ("azimuth", t),
            "latitude": 45.0,
            "longitude": 10.0,
            "altitude": ALT,
        },
    )


def make_volume(elevations, funcs, **kwargs):
    sweeps = {
        f"sweep_{k}": make_sweep(el, funcs, **kwargs).drop_vars(
            ["latitude", "longitude", "altitude"]
        )
        for k, el in enumerate(elevations)
    }
    root = xr.Dataset(coords={"latitude": 45.0, "longitude": 10.0, "altitude": ALT})
    return xr.DataTree.from_dict({"/": root, **sweeps})


LINEAR = {
    "DBZH": lambda h: 20.0 + 0.002 * h,
    "ZDR": lambda h: 1.0 + 0.0 * h,
    "RHOHV": lambda h: 0.98 + 0.0 * h,
}


# --------------------------------------------------------------------------
# averaging in height
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", ["top_hat", "cressman"])
def test_blend_engines_agree_and_ignore_missing(kernel):
    rng = np.random.default_rng(1)
    h = np.sort(rng.uniform(0.0, 5000.0, 4000))
    w = rng.uniform(0.0, 2.0, h.size)
    w[::7] = 0.0  # zero weight: not a sample
    x = rng.normal(10.0, 5.0, (3, h.size))
    x[0, ::5] = np.nan
    grid = np.arange(-200.0, 5300.0, 37.0)
    modes = [0, 1, 0]
    kw = dict(radius=120.0, kernel=kernel, modes=modes, min_samples=3, n_threads=2)
    a = vp._blend(h, w, x, grid, engine="numpy", **kw)
    b = vp._blend(h, w, x, grid, engine="compiled", **kw)
    np.testing.assert_allclose(a[0], b[0], rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_array_equal(a[1], b[1])
    assert np.isnan(a[0][:, 0]).all() and (a[1][:, 0] == 0).all()  # no samples
    # a sample with zero weight or a missing value is not counted
    assert a[1].max() <= (h.size - np.isnan(x[0]).sum())


@pytest.mark.parametrize("engine", ENGINES)
def test_blend_known_values(engine):
    # three samples around 100 m, Cressman radius 100 m: weights (R2-d2)/(R2+d2)
    h = np.array([40.0, 100.0, 160.0, 400.0])
    x = np.array([[1.0, 2.0, 4.0, 100.0], [10.0, 20.0, 30.0, 40.0]])
    out, n = vp._blend(
        h, np.ones(4), x, np.array([100.0]), 100.0, "cressman", [0, 1], 1, 0, engine
    )
    w = np.array([(1e4 - 3600) / (1e4 + 3600), 1.0, (1e4 - 3600) / (1e4 + 3600)])
    assert out[0, 0] == pytest.approx(np.sum(w * [1, 2, 4]) / w.sum())
    lin = 10.0 ** (np.array([10.0, 20.0, 30.0]) / 10.0)
    assert out[1, 0] == pytest.approx(10 * np.log10(np.sum(w * lin) / w.sum()))
    assert n[:, 0].tolist() == [3, 3]
    # the top hat is the plain mean of [z - r, z + r)
    out, n = vp._blend(
        h, np.ones(4), x, np.array([100.0]), 61.0, "top_hat", [0, 0], 1, 0, engine
    )
    assert out[0, 0] == pytest.approx(1.0 / 3.0 * (1 + 2 + 4)) and n[0, 0] == 3
    out, n = vp._blend(
        h, np.ones(4), x, np.array([100.0]), 61.0, "top_hat", [0, 0], 4, 0, engine
    )
    assert np.isnan(out).all() and n[0, 0] == 3


def test_blend_errors():
    h, w, x, g = np.arange(5.0), np.ones(5), np.ones((1, 5)), np.arange(3.0)
    with pytest.raises(ValueError, match="radius"):
        vp._blend(h, w, x, g, 0.0, "top_hat", [0], 1, 0, "numpy")
    if vp.HAS_COMPILED_KERNEL:
        from radarx.retrieve import _qvp

        with pytest.raises(ValueError, match="ascending"):
            _qvp.height_blend(h[::-1].copy(), w, x, g, 1.0, 0, [0], 1)
        with pytest.raises(ValueError, match="kernel"):
            _qvp.height_blend(h, w, x, g, 1.0, 5, [0], 1)
        with pytest.raises(ValueError, match="mode"):
            _qvp.height_blend(h, w, x, g, 1.0, 0, [7], 1)
        with pytest.raises(ValueError, match="match"):
            _qvp.height_blend(h, w, x, g, 1.0, 0, [0, 0], 1)
        with pytest.raises(ValueError, match="1-D"):
            _qvp.height_blend(h, w, x[0], g, 1.0, 0, [0], 1)
    with pytest.raises(ValueError, match="vertical_mean"):
        vp._blend_modes(None, ["DBZH"], None, "median")


# --------------------------------------------------------------------------
# range-defined QVP
# --------------------------------------------------------------------------

ELEVATIONS = (0.5, 2.0, 5.0, 10.0, 15.0)


@pytest.mark.parametrize("engine", ENGINES)
def test_rd_qvp_gives_back_the_profile(engine):
    volume = make_volume(ELEVATIONS, LINEAR)
    out = rd_qvp(volume, power=0.0, engine=engine, counts=True)
    assert out.DBZH.dims == ("height",)
    z = out.height.values
    ok = np.isfinite(out.DBZH.values)
    assert ok.sum() > 100 and z[ok].min() < 500 and z[ok].max() > 10_000
    np.testing.assert_allclose(out.DBZH.values[ok], 20.0 + 0.002 * z[ok], atol=0.1)
    np.testing.assert_allclose(out.RHOHV.values[ok], 0.98, atol=1e-6)
    np.testing.assert_allclose(out.ZDR.values[ok], 1.0, atol=1e-5)
    assert (out.DBZH_count.values[ok] >= 1).all()
    assert out.height.attrs["units"] == "m" and float(out.altitude) == ALT
    assert "weighted mean" in out.DBZH.attrs["cell_methods"]
    assert "ground range <= 50000 m" in out.DBZH.attrs["cell_methods"]
    # a low-elevation sweep gives no warning here
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rd_qvp(volume, ["DBZH"], engine=engine)


def test_rd_qvp_engines_agree():
    volume = make_volume(ELEVATIONS, LINEAR)
    for kw in (dict(), dict(power=1.0, vertical_mean="db"), dict(rmax=20e3, dz=250.0)):
        a = rd_qvp(volume, engine="numpy", **kw)
        for engine in ENGINES:
            b = rd_qvp(volume, engine=engine, **kw)
            xr.testing.assert_allclose(without_provenance(a), without_provenance(b))


@pytest.mark.parametrize("engine", ENGINES)
def test_rd_qvp_range_limit_and_weights(engine):
    # the lowest sweep has 10, the highest 20; at a height both see, the
    # nearer one (the higher sweep) counts more
    funcs = {
        "DBZH": lambda h: 25.0 + 0.0 * h,
        "KDP": lambda h: 1.0 + 0.0 * h,
        "RHOHV": lambda h: 0.98 + 0.0 * h,
    }
    low, high = make_sweep(0.5, funcs), make_sweep(15.0, funcs)
    low["KDP"] = low["KDP"] * 0.0 + 10.0
    high["KDP"] = high["KDP"] * 0.0 + 20.0
    volume = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(
                coords={"latitude": 45.0, "longitude": 10.0, "altitude": ALT}
            ),
            "sweep_0": low.drop_vars(["latitude", "longitude", "altitude"]),
            "sweep_1": high.drop_vars(["latitude", "longitude", "altitude"]),
        }
    )
    heights = np.array([500.0, 2000.0])
    equal = rd_qvp(
        volume, ["KDP"], power=0.0, heights=heights, radius=100.0, engine=engine
    )
    steep = rd_qvp(
        volume, ["KDP"], power=3.0, heights=heights, radius=100.0, engine=engine
    )
    # 500 m: the low sweep is within 50 km (its beam reaches 580 m there); at
    # 2000 m only the high sweep is
    assert 10.0 < equal.KDP.values[0] < steep.KDP.values[0] < 20.0
    assert steep.KDP.values[0] > 19.0
    assert equal.KDP.values[1] == pytest.approx(20.0)
    # without the range limit the low sweep reaches 2000 m at 175 km: not in 100 km
    wide = rd_qvp(
        volume,
        ["KDP"],
        power=0.0,
        rmax=99e3,
        heights=heights,
        radius=100.0,
        engine=engine,
    )
    assert equal.KDP.values[1] == wide.KDP.values[1]
    narrow = rd_qvp(
        volume,
        ["KDP"],
        power=0.0,
        rmax=10e3,
        heights=heights,
        radius=100.0,
        engine=engine,
    )
    assert narrow.KDP.values[0] == pytest.approx(20.0)  # the low sweep is out


def test_rd_qvp_selection_series_and_errors():
    volume = make_volume(ELEVATIONS, LINEAR)
    # a sweep without Z/rhohv (Doppler-only cut) is left out
    velocity = make_sweep(1.0, {"VRADH": lambda h: 0.0 * h}).drop_vars(
        ["latitude", "longitude", "altitude"]
    )
    mixed = volume.copy()
    mixed["sweep_5"] = xr.DataTree(velocity)
    a = rd_qvp(volume, ["DBZH"], engine="numpy")
    b = rd_qvp(mixed, ["DBZH"], engine="numpy")
    xr.testing.assert_allclose(without_provenance(a), without_provenance(b))
    # elevation limits
    only = rd_qvp(
        volume, ["DBZH"], min_elevation=4.0, max_elevation=11.0, engine="numpy"
    )
    assert float(only.height.min()) >= 300
    # a sequence of volumes gives time-height profiles
    later = make_volume(ELEVATIONS, LINEAR, time="2026-01-01T00:05")
    series = rd_qvp([volume, later], ["DBZH"], heights=[1000.0, 2000.0])
    assert series.DBZH.dims == ("time", "height") and series.time.size == 2
    assert series.time.values[1] - series.time.values[0] == np.timedelta64(300, "s")
    single = rd_qvp(volume, ["DBZH"], heights=[1000.0, 2000.0])
    np.testing.assert_allclose(series.DBZH.values[0], single.DBZH.values)
    assert single.time.ndim == 0
    # a single sweep works too
    assert np.isfinite(rd_qvp(make_sweep(5.0, LINEAR), ["DBZH"]).DBZH).sum() > 10
    # elevation correction of KDP
    funcs = {**LINEAR, "KDP": lambda h: 1.0 + 0.0 * h}
    corrected = rd_qvp(make_volume((10.0,), funcs), ["KDP"], elevation_correction=True)
    assert float(corrected.KDP.max()) == pytest.approx(
        1.0 / np.cos(np.deg2rad(10.0)) ** 2, rel=1e-3
    )
    with pytest.raises(ValueError, match="rmax"):
        rd_qvp(volume, rmax=0.0)
    with pytest.raises(ValueError, match="within rmax"):
        rd_qvp(volume, rmax=10.0)
    with pytest.raises(ValueError, match="No volumes"):
        rd_qvp([])
    with pytest.raises(ValueError, match="No sweep has the variables"):
        rd_qvp(volume, ["NOPE"])
    with pytest.raises(ValueError, match="strictly ascending"):
        rd_qvp(volume, heights=[2.0, 1.0])
    with pytest.raises(ValueError, match="dz"):
        rd_qvp(volume, dz=0.0)
    with pytest.raises(TypeError, match="DataTree"):
        rd_qvp(volume.sweep_0.DBZH)
    with pytest.raises(ValueError, match="No sweep groups"):
        rd_qvp(xr.DataTree.from_dict({"/": xr.Dataset()}))
    assert rd_qvp(volume, ["DBZH"], heights=[]).height.size == 0
    # a sweep without times has no time; a root whose site is not a coordinate
    untimed = rd_qvp(make_sweep(5.0, LINEAR).drop_vars("time"), ["DBZH"])
    assert np.isnat(untimed.time.values)
    root = xr.Dataset(
        {"latitude": 45.0, "longitude": 10.0, "altitude": ALT}
    )  # variables: the sweeps do not inherit them
    bare = xr.DataTree.from_dict(
        {
            "/": root,
            "sweep_0": make_sweep(5.0, LINEAR).drop_vars(
                ["latitude", "longitude", "altitude"]
            ),
        }
    )
    assert float(rd_qvp(bare, ["DBZH"]).altitude) == ALT
    empty = vp._height_grid(None, np.inf, -np.inf, 100.0)
    assert empty.size == 0


# --------------------------------------------------------------------------
# columnar vertical profile
# --------------------------------------------------------------------------

CVP_ELEVATIONS = (0.5, 1.5, 2.5, 3.5, 4.5, 6.0, 8.0, 10.0, 12.0, 15.0)


def _outside_sector(az, r, el, centre_az=90.0, centre=30e3, half=10e3, half_az=10.0):
    ground = r * np.cos(np.deg2rad(el))
    d_az = np.abs((az - centre_az + 180.0) % 360.0 - 180.0)
    return 100.0 * ((d_az > half_az) | (np.abs(ground - centre) > half))


@pytest.mark.parametrize("engine", ENGINES)
def test_cvp_uses_only_the_sector(engine):
    funcs = {**LINEAR, "KDP": lambda h: 0.001 * h}
    volume = make_volume(
        CVP_ELEVATIONS,
        funcs,
        nray=360,
        offset={"KDP": _outside_sector, "DBZH": _outside_sector},
    )
    out = cvp(
        volume,
        ["DBZH", "KDP"],
        azimuth=90.0,
        ground_range=30e3,
        engine=engine,
        counts=True,
    )
    z = out.height.values
    ok = np.isfinite(out.KDP.values)
    assert ok.sum() > 100
    # inside the sector the profile is the function, the offset of the rest is not seen
    np.testing.assert_allclose(out.KDP.values[ok], 0.001 * z[ok], atol=0.06)
    np.testing.assert_allclose(out.DBZH.values[ok], 20.0 + 0.002 * z[ok], atol=0.2)
    # the full circle of the same sweep is contaminated
    full = vp.qvp(volume, ["KDP"], elevation=10.0)
    assert np.nanmax(full.KDP.values - 0.001 * full.height.values) > 5
    assert float(out.azimuth) == 90.0 and float(out.ground_range) == 30e3
    assert out.height[1] - out.height[0] == 50.0
    assert "Cressman" in out.KDP.attrs["cell_methods"]


@pytest.mark.parametrize("engine", ENGINES)
def test_cvp_intermediate_elevation_bands(engine):
    # every sweep has its own constant: at the centre, a sweep may only
    # contribute between the beams half-way to its neighbours
    funcs = {
        "DBZH": lambda h: 25.0 + 0.0 * h,
        "RHOHV": lambda h: 0.98 + 0.0 * h,
        "KDP": lambda h: 0.0 * h,
    }
    elevations = (0.5, 1.5, 3.0, 6.0)
    sweeps = {}
    for k, el in enumerate(elevations):
        s = make_sweep(el, funcs, nray=360)
        s["KDP"] = s["KDP"] * 0.0 + el
        sweeps[f"sweep_{k}"] = s.drop_vars(["latitude", "longitude", "altitude"])
    volume = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(
                coords={"latitude": 45.0, "longitude": 10.0, "altitude": ALT}
            ),
            **sweeps,
        }
    )
    out = cvp(
        volume,
        ["KDP"],
        azimuth=0.0,
        ground_range=30e3,
        half_range=10e3,
        engine=engine,
        radius=40.0,
    )
    mid = [0.5 * (a + b) for a, b in zip(elevations[:-1], elevations[1:])]
    edges = [vp._height_at_ground(m, 30e3, ALT) for m in mid]
    z = out.height.values
    for (lo, hi), el in zip(zip([-np.inf, *edges], [*edges, np.inf]), elevations):
        inside = (z > lo + 60) & (z < hi - 60) & np.isfinite(out.KDP.values)
        if inside.any():
            np.testing.assert_allclose(out.KDP.values[inside], el, atol=1e-9)
    # the bands are the ones of the helper
    assert vp._intermediate_band(np.array(elevations), 3.0, 30e3, ALT) == (
        pytest.approx(edges[1]),
        pytest.approx(edges[2]),
    )
    low, high = vp._intermediate_band(np.array(elevations), 0.5, 30e3, ALT)
    assert low == -np.inf and high == pytest.approx(edges[0])
    low, high = vp._intermediate_band(np.array(elevations), 6.0, 30e3, ALT)
    assert low == pytest.approx(edges[2]) and high == np.inf


def test_cvp_engines_agree_and_series():
    funcs = {**LINEAR, "KDP": lambda h: 0.001 * h}
    volume = make_volume(CVP_ELEVATIONS, funcs, nray=180)
    kw = dict(azimuth=45.0, ground_range=40e3, half_azimuth=15.0)
    a = cvp(volume, ["DBZH", "KDP"], engine="numpy", **kw)
    for engine in ENGINES:
        b = cvp(volume, ["DBZH", "KDP"], engine=engine, vertical_mean="db", **kw)
        xr.testing.assert_allclose(without_provenance(a), without_provenance(b))
    lin = cvp(volume, ["DBZH"], vertical_mean="linear", **kw)
    assert np.nanmax(np.abs(lin.DBZH.values - a.DBZH.values)) < 0.05
    later = make_volume(CVP_ELEVATIONS, funcs, nray=180, time="2026-01-01T00:05")
    series = cvp(
        [volume, later], ["KDP"], heights=np.arange(1000.0, 3000.0, 100.0), **kw
    )
    assert series.KDP.dims == ("time", "height")
    np.testing.assert_allclose(series.KDP.values[0], series.KDP.values[1])
    assert series.azimuth.ndim == 0


def test_cvp_errors_and_empty_sector():
    volume = make_volume(CVP_ELEVATIONS, LINEAR, nray=180)
    with pytest.raises(ValueError, match="positive"):
        cvp(volume, azimuth=0.0, ground_range=0.0)
    with pytest.raises(ValueError, match="No sweep has gates in the sector"):
        cvp(volume, azimuth=0.0, ground_range=500e3)
    nofield = volume.copy()
    nofield["sweep_0"] = xr.DataTree(
        volume["sweep_0"].to_dataset().drop_vars("azimuth")
    )
    with pytest.raises(ValueError, match="azimuth"):
        cvp(nofield, ["DBZH"], azimuth=0.0, ground_range=30e3)
    # the helper at a very long ground distance still returns a height
    assert vp._height_at_ground(5.0, 100e3, ALT) > 8000.0


# --------------------------------------------------------------------------
# RHI profile
# --------------------------------------------------------------------------


def make_rhi(offset=None, time="2026-01-01T00:00", elevations=None, rng=None):
    rng = np.arange(125.0, 30_000.0, 250.0) if rng is None else rng
    el = np.arange(1.0, 61.0, 0.5) if elevations is None else elevations
    funcs = {
        "DBZH": lambda h: 20.0 + 0.002 * h,
        "ZDR": lambda h: 1.0 + 0.0 * h,
        "RHOHV": lambda h: 0.98 + 0.0 * h,
    }
    sweep = make_sweep(
        0.0, funcs, nray=el.size, rng=rng, elevations=el, offset=offset, time=time
    )
    sweep = sweep.assign_coords(azimuth=("azimuth", np.full(el.size, 109.0)))
    return sweep


@pytest.mark.parametrize("engine", ENGINES)
def test_rhi_qvp_gives_back_the_profile(engine):
    def dirty(az, r, el):
        bad = (r < 6000.0) | (r > 20000.0) | (el >= 45.0)
        return 100.0 * bad

    rhi = make_rhi(offset={"DBZH": dirty, "ZDR": dirty})
    out = rhi_qvp(rhi, engine=engine, counts=True)
    z = out.height.values
    ok = np.isfinite(out.DBZH.values)
    assert ok.sum() > 20
    # bins of 100 m: the mean of F over the bin (slope 0.002 dB/m, 100 m bins)
    np.testing.assert_allclose(out.DBZH.values[ok], 20.0 + 0.002 * z[ok], atol=0.12)
    np.testing.assert_allclose(out.ZDR.values[ok], 1.0, atol=1e-5)
    assert (np.diff(z) == 100.0).all() and (out.DBZH_count.values[ok] >= 1).all()
    assert float(out.azimuth) == 109.0 and "R-QVP" not in out.attrs["title"]
    assert "range 6000 to 20000 m" in out.DBZH.attrs["qvp_filter"]
    assert "qvp_min_count" not in out.DBZH.attrs


@pytest.mark.parametrize("engine", ENGINES)
def test_rhi_qvp_quality_masks(engine):
    low_rho = lambda az, r, el: np.where(el < 20.0, -0.5, 0.0)  # rhohv 0.48 below 20
    rhi = make_rhi(offset={"RHOHV": low_rho})
    masked = rhi_qvp(rhi, ["DBZH"], engine=engine)
    open_ = rhi_qvp(rhi, ["DBZH"], min_rhohv=None, engine=engine)
    assert np.isfinite(open_.DBZH).sum() > np.isfinite(masked.DBZH).sum()
    # Z >= 0 dBZ keeps a gate at exactly 0 dBZ, rhohv > 0.5 rejects 0.5
    rhi = make_rhi()
    rhi["DBZH"] = rhi["DBZH"] * 0.0
    rhi["RHOHV"] = rhi["RHOHV"] * 0.0 + 0.5
    assert np.isnan(rhi_qvp(rhi, ["DBZH"], min_rhohv=0.5, engine=engine).DBZH).all()
    rhi["RHOHV"] = rhi["RHOHV"] * 0.0 + 0.75
    assert np.isfinite(rhi_qvp(rhi, ["DBZH"], min_rhohv=0.5, engine=engine).DBZH).any()
    assert np.isnan(rhi_qvp(rhi, ["DBZH"], min_dbz=0.1, engine=engine).DBZH).all()


def test_rhi_qvp_series_volume_and_errors():
    rhi = make_rhi()
    later = make_rhi(time="2026-01-01T00:05")
    series = rhi_qvp(
        [rhi, later], ["DBZH", "RHOHV"], heights=np.arange(500.0, 4000.0, 100.0)
    )
    assert series.DBZH.dims == ("time", "height") and series.time.size == 2
    # a volume of RHI sweeps: the sweep named is used
    root = xr.Dataset(coords={"latitude": 45.0, "longitude": 10.0, "altitude": ALT})
    tree = xr.DataTree.from_dict(
        {
            "/": root,
            "sweep_0": rhi.drop_vars(["latitude", "longitude", "altitude"]),
            "sweep_1": later.drop_vars(["latitude", "longitude", "altitude"]),
        }
    )
    a = rhi_qvp(tree, ["DBZH"])
    b = rhi_qvp(rhi, ["DBZH"])
    xr.testing.assert_allclose(without_provenance(a), without_provenance(b))
    c = rhi_qvp(tree, ["DBZH"], sweep=1)
    assert c.time.values > a.time.values
    # the melting layer can be searched in the profile
    prof = rhi_qvp(rhi)
    assert melting_layer(prof)["melting_layer_flag"].dtype == np.int8
    # linear averaging of dB changes the mean of a spread of values
    spread = make_rhi()
    spread["DBZH"] = (
        spread["DBZH"] + 10.0 * np.sin(np.arange(spread.sizes["azimuth"]))[:, None]
    ).assign_attrs(
        units="dBZ"
    )  # arithmetic drops the attributes in older xarray
    lin = rhi_qvp(spread, ["DBZH"], dz=2000.0)
    avg = rhi_qvp(spread, ["DBZH"], dz=2000.0, linear=[])
    assert np.nanmax(lin.DBZH.values - avg.DBZH.values) > 0.5
    with pytest.raises(ValueError, match="No scans"):
        rhi_qvp([])
    bad = rhi.assign_coords(elevation=("azimuth", np.zeros((rhi.sizes["azimuth"],))))
    bad2 = bad.copy()
    bad2["elevation"] = (("azimuth", "range"), np.zeros(rhi.DBZH.shape))
    with pytest.raises(ValueError, match="one value for every ray"):
        rhi_qvp(bad2, ["DBZH"])
    empty = rhi_qvp(rhi, ["DBZH"], rmin=1e6, rmax=2e6)
    assert empty.height.size == 0
    no_az = rhi.drop_vars("azimuth").assign_coords(
        azimuth=("azimuth", np.arange(rhi.sizes["azimuth"], dtype=float))
    )
    assert np.isfinite(rhi_qvp(no_az.drop_vars("azimuth"), ["DBZH"]).DBZH).any()


# --------------------------------------------------------------------------
# melting layer on PPIs
# --------------------------------------------------------------------------

ML_BOTTOM, ML_TOP = 3000.0, 3800.0


def ml_funcs(bottom=ML_BOTTOM, top=ML_TOP):
    inside = lambda h: (h >= bottom) & (h <= top)  # noqa: E731
    return {
        "DBZH": lambda h: np.where(inside(h), 38.0, 28.0),
        "ZDR": lambda h: np.where(inside(h), 1.5, 0.3),
        "RHOHV": lambda h: np.where(inside(h), 0.94, 0.99),
    }


def ring_sweep(elevation, nray=360, tilt=0.0, **kwargs):
    """Rays whose melting layer is ``tilt * cos(azimuth)`` metres higher."""
    az = (np.arange(nray) + 0.5) * 360.0 / nray
    el = np.full(nray, float(elevation))
    from xradar.georeference import antenna_to_cartesian

    h = antenna_to_cartesian(
        RNG[None, :], 0.0, el[:, None], earth_radius=vp.EARTH_RADIUS, site_altitude=ALT
    )[2]
    shift = tilt * np.cos(np.deg2rad(az))[:, None]
    f = ml_funcs(**kwargs)
    data = {
        name: (
            ("azimuth", "range"),
            np.asarray(fn(h - shift), np.float32),
            {"units": u},
        )
        for (name, fn), u in zip(f.items(), ("dBZ", "dB", "1"))
    }
    return xr.Dataset(
        data,
        coords={
            "azimuth": az,
            "range": RNG,
            "elevation": ("azimuth", el),
            "time": (
                "azimuth",
                np.datetime64("2026-01-01T00:00", "ns")
                + (np.arange(nray) * 50).astype("timedelta64[ms]"),
            ),
            "latitude": 45.0,
            "longitude": 10.0,
            "altitude": ALT,
        },
    )


def ring_volume(elevations=(1.0, 5.0, 8.0, 15.0), **kwargs):
    sweeps = {
        f"sweep_{k}": ring_sweep(el, **kwargs).drop_vars(
            ["latitude", "longitude", "altitude"]
        )
        for k, el in enumerate(elevations)
    }
    root = xr.Dataset(coords={"latitude": 45.0, "longitude": 10.0, "altitude": ALT})
    return xr.DataTree.from_dict({"/": root, **sweeps})


@pytest.mark.parametrize("engine", ENGINES)
def test_ppi_melting_layer_known_rings(engine):
    ml = melting_layer_ppi(ring_volume(), engine=engine)
    # 80th and 20th percentile of heights spread uniformly over the layer
    assert float(ml.melting_layer_top_all) == pytest.approx(
        ML_BOTTOM + 0.8 * 800.0, abs=30.0
    )
    assert float(ml.melting_layer_bottom_all) == pytest.approx(
        ML_BOTTOM + 0.2 * 800.0, abs=30.0
    )
    np.testing.assert_allclose(
        ml.melting_layer_top, float(ml.melting_layer_top_all), atol=1e-6
    )
    assert ml.melting_layer_top.dims == ("azimuth",) and ml.azimuth.size == 360
    assert (
        ml.melting_layer_points.dtype == np.int64
        and (ml.melting_layer_points > 20).all()
    )
    assert ml.time.ndim == 0
    # only the sweeps from 4 to 10 degrees are used: the points of two sweeps
    one = melting_layer_ppi(ring_volume(elevations=(5.0,)), engine=engine)
    two = melting_layer_ppi(ring_volume(elevations=(5.0, 8.0)), engine=engine)
    assert int(two.melting_layer_points_all) > 1.5 * int(one.melting_layer_points_all)
    assert int(
        melting_layer_ppi(ring_volume(), engine=engine).melting_layer_points_all
    ) == int(two.melting_layer_points_all)
    # the layer and a layer 600 m higher
    high = melting_layer_ppi(ring_volume(bottom=3600.0, top=4400.0), engine=engine)
    assert float(high.melting_layer_top_all) - float(
        ml.melting_layer_top_all
    ) == pytest.approx(600.0, abs=40.0)


@pytest.mark.parametrize("engine", ENGINES)
def test_ppi_melting_layer_follows_the_azimuth(engine):
    volume = ring_volume(elevations=(5.0, 8.0), tilt=400.0)
    ml = melting_layer_ppi(volume, engine=engine, sector=20.0)
    top = ml.melting_layer_top.values
    # the layer is 400 m higher at azimuth 0 than at 180 (cos), smoothed by the window
    assert top[0] - top[180] == pytest.approx(800.0, abs=80.0)
    assert top[90] == pytest.approx(ML_BOTTOM + 640.0, abs=60.0)
    narrow = melting_layer_ppi(volume, engine=engine, sector=2.0)
    assert narrow.melting_layer_top.values[0] - narrow.melting_layer_top.values[
        180
    ] == pytest.approx(800.0, abs=40.0)
    wide = melting_layer_ppi(volume, engine=engine, sector=360.0)
    assert np.ptp(wide.melting_layer_top.values) < 1e-6
    # the window wraps around north
    assert (
        melting_layer_ppi(
            volume, engine=engine, sector=20.0, azimuth_step=5.0
        ).azimuth.size
        == 72
    )


def test_ppi_melting_layer_engines_agree_on_noise():
    rng = np.random.default_rng(3)
    volume = ring_volume(elevations=(4.0, 6.0, 9.0), tilt=300.0)
    for name in volume.children:
        ds = volume[name].to_dataset()
        for v, s in (("DBZH", 6.0), ("ZDR", 0.4), ("RHOHV", 0.02)):
            ds[v] = ds[v] + rng.normal(0.0, s, ds[v].shape).astype(np.float32)
        ds["DBZH"] = ds["DBZH"].where(rng.uniform(size=ds["DBZH"].shape) > 0.1)
        volume[name] = ds
    a = melting_layer_ppi(volume, engine="numpy", min_points=5)
    assert int(a.melting_layer_points_all) > 100
    for engine in ENGINES:
        b = melting_layer_ppi(volume, engine=engine, min_points=5, n_threads=2)
        xr.testing.assert_allclose(without_provenance(a), without_provenance(b))


def test_ppi_melting_layer_options_series_and_errors():
    volume = ring_volume(elevations=(5.0, 8.0))
    shifted = ring_volume(elevations=(5.0, 8.0), bottom=3200.0, top=4000.0)
    series = melting_layer_ppi([volume, shifted])
    assert (
        series.melting_layer_top.dims == ("time", "azimuth") and series.time.size == 2
    )
    assert series.melting_layer_top_all.dims == ("time",)
    assert float(
        series.melting_layer_top_all[1] - series.melting_layer_top_all[0]
    ) == pytest.approx(200.0, abs=40.0)
    # thresholds and percentiles
    for engine in ENGINES:
        strict = melting_layer_ppi(volume, dbz_range=(40.0, 47.0), engine=engine)
        assert np.isnan(strict.melting_layer_top_all)
        assert int(strict.melting_layer_points_all) == 0
        few = melting_layer_ppi(volume, min_points=10**7, engine=engine)
        assert np.isnan(few.melting_layer_top).all()
        assert int(few.melting_layer_points_all) > 0
    low = melting_layer_ppi(volume, height_max=3500.0)
    assert float(low.melting_layer_top_all) < float(
        melting_layer_ppi(volume).melting_layer_top_all
    )
    quart = melting_layer_ppi(volume, percentiles=(75.0, 25.0))
    assert float(quart.melting_layer_top_all) < float(
        melting_layer_ppi(volume).melting_layer_top_all
    )
    # a sweep given directly is used whatever its elevation, with a warning
    sweep = ring_sweep(5.0)
    direct = melting_layer_ppi(sweep)
    assert float(direct.melting_layer_top_all) > 3000
    with pytest.warns(UserWarning, match="outside the 4 to 10 degrees"):
        melting_layer_ppi(ring_sweep(15.0))
    with pytest.raises(ValueError, match="needs Z, ZDR and rhohv"):
        melting_layer_ppi(sweep.drop_vars("ZDR"))
    with pytest.raises(ValueError, match="No sweep between"):
        melting_layer_ppi(ring_volume(elevations=(1.0, 15.0)))
    # sweeps without the fields are skipped in a volume
    plain = volume.copy()
    plain["sweep_2"] = xr.DataTree(
        ring_sweep(6.0).drop_vars(["latitude", "longitude", "altitude", "ZDR"])
    )
    assert int(melting_layer_ppi(plain).melting_layer_points_all) == int(
        melting_layer_ppi(volume).melting_layer_points_all
    )
    with pytest.raises(ValueError, match="azimuth_step"):
        melting_layer_ppi(volume, azimuth_step=0.0)
    with pytest.raises(ValueError, match="sector"):
        melting_layer_ppi(volume, sector=0.0)
    with pytest.raises(ValueError, match="No volumes"):
        melting_layer_ppi([])
    # the beam height must increase with range
    broken = ring_sweep(5.0)
    broken = broken.assign_coords(elevation=("azimuth", np.full(360, -5.0)))
    with pytest.raises(ValueError, match="increase with range"):
        melting_layer_ppi(broken)
    assert melting_layer_ppi(volume).attrs["radarx_references"]


def test_ppi_layer_on_a_pooled_sweep_matches_the_qvp_layer():
    """A ring at a known height is found by the PPI method and in the QVP."""
    volume = ring_volume(elevations=(5.0, 8.0, 15.0))
    ppi = melting_layer_ppi(volume)
    prof = vp.qvp(volume, ["DBZH", "ZDR", "RHOHV"], elevation=15.0)
    qvp_layer = melting_layer(prof, height_range=(1000.0, 6000.0))
    top_qvp = float(qvp_layer.melting_layer_top)
    # both lie within the layer; they define top and bottom differently
    for value in (float(ppi.melting_layer_top_all), top_qvp):
        assert ML_BOTTOM - 100.0 <= value <= ML_TOP + 200.0


# --------------------------------------------------------------------------
# the cap on rain-only products
# --------------------------------------------------------------------------


def rain_sweep(elevation=2.0, nray=60):
    sweep = ring_sweep(elevation, nray=nray)
    dims = ("azimuth", "range")
    rate = xr.DataArray(
        np.full((nray, RNG.size), 5.0, dtype=np.float32),
        dims=dims,
        attrs={"units": "mm h-1"},
    )
    flag = xr.DataArray(
        np.zeros((nray, RNG.size), dtype=np.int8),
        dims=dims,
        attrs={
            "flag_values": np.array([0, 1, 2, 3], np.int8),
            "flag_meanings": "estimated no_relation_for_class missing_input not_classified",
        },
    )
    relation = xr.DataArray(np.full((nray, RNG.size), 3, dtype=np.int8), dims=dims)
    return sweep.assign(
        RAIN_RATE=rate,
        RAIN_RATE_FLAG=flag,
        RAIN_RATE_RELATION=relation,
        D0=rate * 0.1,
        NW=rate,
    )


def gate_heights(sweep):
    return sweep["range"] * 0 + vp._gate_height(sweep, None, None)


def test_cap_removes_rain_products_above_the_bottom():
    sweep = rain_sweep()
    before = sweep.copy(deep=True)
    out = mask_above_melting_layer(sweep, 2000.0)
    xr.testing.assert_identical(sweep, before)  # the input is not changed
    height = vp._gate_height(sweep, None, None)
    above = (height > 2000.0).values
    assert above.any() and (~above).any()
    for name in ("RAIN_RATE", "D0", "NW"):
        assert np.isnan(out[name].values[above]).all()
        assert np.isfinite(out[name].values[~above]).all()
        assert "melting layer bottom" in out[name].attrs["comment"]
    # other variables are kept
    np.testing.assert_array_equal(out["DBZH"].values, sweep["DBZH"].values)
    # the flag says why, and the relation is none
    assert (out.RAIN_RATE_FLAG.values[above] == 4).all() and (
        out.RAIN_RATE_FLAG.values[~above] == 0
    ).all()
    assert (out.RAIN_RATE_RELATION.values[above] == 0).all() and (
        out.RAIN_RATE_RELATION.values[~above] == 3
    ).all()
    assert "above_melting_layer" in out.RAIN_RATE_FLAG.attrs["flag_meanings"].split()
    assert out.RAIN_RATE_FLAG.attrs["flag_values"].tolist() == [0, 1, 2, 3, 4]
    assert out.RAIN_RATE_FLAG.dtype == np.int8
    # the margin moves the cap
    higher = mask_above_melting_layer(sweep, 2000.0, margin=500.0)
    assert (
        np.isfinite(higher.RAIN_RATE.values).sum()
        > np.isfinite(out.RAIN_RATE.values).sum()
    )
    assert np.isfinite(higher.RAIN_RATE.values[height.values <= 2500.0]).all()
    # the lower edge of a wide beam is lower: gates just above the cap are kept
    wide = mask_above_melting_layer(sweep, 2000.0, beam_width=2.0)
    assert (
        np.isfinite(wide.RAIN_RATE.values).sum()
        > np.isfinite(out.RAIN_RATE.values).sum()
    )


def test_cap_without_a_known_layer_keeps_everything():
    sweep = rain_sweep()
    out = mask_above_melting_layer(sweep, np.nan)
    np.testing.assert_array_equal(out.RAIN_RATE.values, sweep.RAIN_RATE.values)
    assert (out.RAIN_RATE_FLAG == 0).all()


def test_cap_from_the_melting_layer_outputs():
    sweep = rain_sweep(elevation=5.0, nray=360)
    ppi = melting_layer_ppi(ring_volume(elevations=(5.0, 8.0), tilt=400.0))
    out = mask_above_melting_layer(sweep, ppi)  # melting_layer_bottom on azimuth
    height = vp._gate_height(sweep, None, None).values
    kept = np.isfinite(out.RAIN_RATE.values)
    # the cap follows the azimuth: higher where the layer is higher
    top_kept = np.where(kept, height, -np.inf).max(axis=1)
    assert top_kept[0] > top_kept[180] + 300.0
    cap = ppi.melting_layer_bottom.values
    for i in (0, 90, 180, 270):
        assert top_kept[i] <= cap[i] + 60.0 and top_kept[i] >= cap[i] - 60.0
    # a profile layer (from melting_layer), with time, applies to all gates
    prof = vp.qvp(
        ring_volume(elevations=(15.0,)), ["DBZH", "ZDR", "RHOHV"], elevation=15.0
    )
    layer = melting_layer(prof)
    out2 = mask_above_melting_layer(sweep, layer)
    assert (
        np.isfinite(out2.RAIN_RATE.values).any()
        and np.isnan(out2.RAIN_RATE.values).any()
    )
    # DataArray and a plain DataArray cap
    da = mask_above_melting_layer(sweep.RAIN_RATE, xr.DataArray(2000.0))
    assert np.isnan(da.values).any() and da.attrs["units"] == "mm h-1"
    assert sweep.RAIN_RATE.attrs.get("comment") is None


def test_cap_variables_height_dask_volume_and_accessors():
    sweep = rain_sweep()
    one = mask_above_melting_layer(sweep, 2000.0, variables="D0")
    assert np.isnan(one.D0.values).any() and np.isfinite(one.RAIN_RATE.values).all()
    given = mask_above_melting_layer(
        sweep, 2000.0, height=vp._gate_height(sweep, None, None) + 1e6
    )
    assert np.isnan(given.RAIN_RATE.values).all()
    # scalar elevation: the height depends on the range only
    flat = sweep.assign_coords(elevation=3.0)
    h = vp._gate_height(flat, None, None)
    assert h.dims == ("range",)
    # lazy
    lazy = sweep.chunk({"azimuth": 20})
    out = mask_above_melting_layer(lazy, 2000.0)
    assert out.RAIN_RATE.chunks is not None
    xr.testing.assert_equal(
        out.RAIN_RATE.compute(), mask_above_melting_layer(sweep, 2000.0).RAIN_RATE
    )
    # a volume
    root = xr.Dataset(coords={"latitude": 45.0, "longitude": 10.0, "altitude": ALT})
    tree = xr.DataTree.from_dict(
        {
            "/": root,
            "sweep_0": rain_sweep(2.0).drop_vars(["latitude", "longitude", "altitude"]),
            "sweep_1": rain_sweep(4.0).drop_vars(["latitude", "longitude", "altitude"]),
        }
    )
    capped = mask_above_melting_layer(tree, 2000.0)
    assert set(capped.children) == {"sweep_0", "sweep_1"}
    assert np.isnan(capped["sweep_1"]["RAIN_RATE"].values).any()
    # accessors
    xr.testing.assert_identical(
        without_provenance(sweep.radarx.mask_above_melting_layer(2000.0)),
        without_provenance(mask_above_melting_layer(sweep, 2000.0)),
    )
    assert np.isnan(
        sweep.RAIN_RATE.radarx.mask_above_melting_layer(2000.0).values
    ).any()
    assert set(tree.radarx.mask_above_melting_layer(2000.0).children) == {
        "sweep_0",
        "sweep_1",
    }


def test_cap_errors():
    sweep = rain_sweep()
    with pytest.raises(TypeError, match="DataArray, Dataset or DataTree"):
        mask_above_melting_layer(np.zeros(3), 2000.0)
    with pytest.raises(ValueError, match="no 'melting_layer_bottom'|has no"):
        mask_above_melting_layer(sweep, xr.Dataset({"x": 1.0}))
    with pytest.raises(ValueError, match="no \\['NOPE'\\]"):
        mask_above_melting_layer(sweep, 2000.0, variables=["NOPE"])
    with pytest.raises(ValueError, match="elevation and range"):
        mask_above_melting_layer(sweep.drop_vars("elevation"), 2000.0)
    two_d = sweep.copy()
    two_d["elevation"] = (("azimuth", "range"), np.zeros(sweep.DBZH.shape))
    with pytest.raises(ValueError, match="scalar or one value per ray"):
        vp._gate_height(two_d, None, None)
    no_az = sweep.drop_vars("azimuth")
    ppi = melting_layer_ppi(ring_volume(elevations=(5.0,)))
    with pytest.raises(ValueError, match="azimuth coordinate"):
        mask_above_melting_layer(no_az, ppi)


# --------------------------------------------------------------------------
# accessors and provenance
# --------------------------------------------------------------------------


def test_accessors():
    volume = make_volume(ELEVATIONS, LINEAR)
    xr.testing.assert_identical(
        volume.radarx.rd_qvp(["DBZH"]), rd_qvp(volume, ["DBZH"])
    )
    cv = make_volume(CVP_ELEVATIONS, LINEAR, nray=180)
    xr.testing.assert_identical(
        cv.radarx.cvp(["DBZH"], azimuth=45.0, ground_range=40e3),
        cvp(cv, ["DBZH"], azimuth=45.0, ground_range=40e3),
    )
    rhi = make_rhi()
    xr.testing.assert_identical(rhi.radarx.rhi_qvp(["DBZH"]), rhi_qvp(rhi, ["DBZH"]))
    ring = ring_volume(elevations=(5.0, 8.0))
    xr.testing.assert_identical(
        ring.radarx.melting_layer_ppi(), melting_layer_ppi(ring)
    )
    sweep = ring_sweep(5.0)
    xr.testing.assert_identical(
        sweep.radarx.melting_layer_ppi(), melting_layer_ppi(sweep)
    )
    xr.testing.assert_identical(
        make_sweep(5.0, LINEAR).radarx.rd_qvp(["DBZH"]),
        rd_qvp(make_sweep(5.0, LINEAR), ["DBZH"]),
    )


def test_provenance_and_references():
    from radarx._provenance import load_registry

    volume = make_volume(ELEVATIONS, LINEAR)
    registry = load_registry()
    for out, doi in (
        (rd_qvp(volume, ["DBZH"]), "10.1175/WAF-D-17-0054.1"),
        (
            cvp(
                make_volume(CVP_ELEVATIONS, LINEAR, nray=180),
                ["DBZH"],
                azimuth=45.0,
                ground_range=40e3,
            ),
            "10.1175/JTECH-D-20-0011.1",
        ),
        (rhi_qvp(make_rhi(), ["DBZH"]), "10.3390/rs11232848"),
        (melting_layer_ppi(ring_volume(elevations=(5.0,))), "10.1175/2007JAMC1634.1"),
        (mask_above_melting_layer(rain_sweep(), 2000.0), "10.1175/JTECH-D-21-0130.1"),
    ):
        refs = out.attrs["radarx_references"].split()
        assert doi in refs and all(r in registry for r in refs)
        assert out.attrs["radarx_method"] and out.attrs["history"]
    for key in (
        "10.1175/WAF-D-17-0054.1",
        "10.1175/JTECH-D-20-0011.1",
        "10.3390/rs11232848",
        "10.1175/JTECH-D-18-0076.1",
        "10.1175/1520-0493(1959)087<0367:AOOAS>2.0.CO;2",
    ):
        assert registry[key]["doi"] == key


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def armor_volume():
    xd = pytest.importorskip("xradar")
    from open_radar_data import DATASETS

    file = DATASETS.fetch("RAW_NA_000_125_20080411181219")
    return xd.io.open_iris_datatree(file)


def test_real_volume_profiles(armor_volume):
    """ARMOR, C band, 11 April 2008 (stratiform rain with a melting layer near 4 km)."""
    fields = ["DBZH", "ZDR", "RHOHV"]
    rd = rd_qvp(armor_volume, fields, engine="numpy")
    assert np.isfinite(rd.DBZH).sum() > 100 and float(rd.height.min()) < 1000
    if vp.HAS_COMPILED_KERNEL:
        c = rd_qvp(armor_volume, fields, engine="compiled")
        xr.testing.assert_allclose(
            without_provenance(rd), without_provenance(c), rtol=1e-6, atol=2e-5
        )
    # the RD-QVP and the QVP of the same sweeps agree where both are defined
    # (the high sweep sees the same rain as the range-limited mean of all sweeps)
    q = vp.qvp(armor_volume, fields, elevation=12.0)
    both = np.interp(
        q.height, rd.height, rd.DBZH.where(np.isfinite(rd.DBZH), np.nan).values
    )
    mask = (
        np.isfinite(both)
        & np.isfinite(q.DBZH.values)
        & (q.height.values > 2000)
        & (q.height.values < 3000)
    )
    assert mask.sum() > 3 and abs(np.mean(both[mask] - q.DBZH.values[mask])) < 8.0
    cv = cvp(armor_volume, fields, azimuth=180.0, ground_range=40e3)
    assert np.isfinite(cv.DBZH).sum() > 50
    assert np.nanmin(cv.RHOHV.values) > 0.6
    ml = melting_layer_ppi(armor_volume)
    top, bottom = float(ml.melting_layer_top_all), float(ml.melting_layer_bottom_all)
    assert 1000.0 < bottom < top < 5500.0
    assert int(ml.melting_layer_points_all) > 1000
    layer = melting_layer(vp.qvp(armor_volume, fields, elevation=12.0))
    # the layer in the PPIs and in the QVP of the highest sweep are in the same place
    assert abs(top - float(layer.melting_layer_top)) < 1500.0
