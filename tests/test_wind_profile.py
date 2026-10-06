#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for radarx.retrieve.wind_profile (shear, storm motion, SRH, VAD)."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401  (registers the accessors)
from radarx.io import sounding
from radarx.retrieve import coldpool
from radarx.retrieve import wind_profile as wp

ENGINES = ["numpy"] + (["compiled"] if coldpool.HAS_COMPILED_KERNEL else [])
DATA = Path(__file__).parent / "io" / "data"


def straight(top=8000.0, dz=10.0, base=0.0):
    """u = z / 600 (0-10 m/s over 6 km), v = 0."""
    z = np.arange(0.0, top + dz / 2, dz)
    return xr.Dataset(
        {"u": ("height", z / 600.0), "v": ("height", np.zeros_like(z))},
        coords={"height": z + base},
    )


def circle(top=3000.0, radius=10.0):
    """Quarter-circle hodograph, clockwise turning, centre at the origin."""
    z = np.linspace(0.0, top, 3001)
    a = np.pi / 2 * z / top
    return xr.Dataset(
        {"u": ("height", -radius * np.cos(a)), "v": ("height", radius * np.sin(a))},
        coords={"height": z},
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_bulk_shear_and_mean_wind(engine):
    p = straight(base=150.0)
    sh = wp.bulk_shear(p, 0, 3000, normal=90.0, engine=engine)
    assert float(sh.shear_u) == pytest.approx(5.0)
    assert float(sh.shear_v) == pytest.approx(0.0, abs=1e-12)
    assert float(sh.shear_normal) == pytest.approx(5.0)
    assert float(sh.shear_direction) == pytest.approx(270.0)  # points east
    assert sh.shear_u.attrs["units"] == "m s-1"
    mean = wp.layer_mean_wind(p, 0, 6000, engine=engine)
    assert float(mean.u) == pytest.approx(5.0)
    # explicit ground in the units of the heights (m above sea level)
    sh2 = wp.bulk_shear(p, 0, 3000, ground=150.0, engine=engine)
    assert float(sh2.shear_u) == pytest.approx(5.0)
    # layer outside the data
    assert np.isnan(wp.bulk_shear(p, 0, 9000, engine=engine).shear_u)
    zero = wp.layer_mean_wind(p, 1000, 1000, engine=engine)
    assert float(zero.u) == pytest.approx(1000 / 600)


@pytest.mark.parametrize("engine", ENGINES)
def test_bunkers_straight_hodograph(engine):
    rm = wp.bunkers_storm_motion(straight(), engine=engine)
    # mean wind (5, 0); shear along +x: right of it is -y
    assert float(rm.u) == pytest.approx(5.0)
    assert float(rm.v) == pytest.approx(-7.5)
    lm = wp.bunkers_storm_motion(straight(), mover="left", engine=engine)
    assert float(lm.v) == pytest.approx(7.5)
    mean = wp.bunkers_storm_motion(straight(), mover="mean", engine=engine)
    assert float(mean.v) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        wp.bunkers_storm_motion(straight(), mover="middle")


@pytest.mark.parametrize("engine", ENGINES)
def test_srh_analytic(engine):
    # straight hodograph, storm (5, -7.5): SRH = 7.5 * du = 7.5 * 5 = 37.5
    srh = wp.storm_relative_helicity(straight(), "right", 0, 3000, engine=engine)
    assert float(srh) == pytest.approx(37.5, rel=1e-9)
    assert srh.attrs["units"] == "m2 s-2"
    # quarter circle of radius R about the storm motion: SRH = R^2 * pi/2,
    # nearly exact for 3001 levels (chords of a circle)
    srh = wp.storm_relative_helicity(circle(), (0.0, 0.0), 0, 3000, engine=engine)
    assert float(srh) == pytest.approx(100 * np.pi / 2, rel=1e-6)
    # a storm on the hodograph's far side reverses the sign of the straight case
    assert float(
        wp.storm_relative_helicity(straight(), (5.0, 7.5), 0, 3000, engine=engine)
    ) == pytest.approx(-37.5)
    # motion given as a Dataset, per time
    times = xr.concat([straight(), straight()], dim="time")
    motion = xr.Dataset({"u": ("time", [5.0, 5.0]), "v": ("time", [-7.5, 7.5])})
    srh = wp.storm_relative_helicity(times, motion, 0, 3000, engine=engine)
    np.testing.assert_allclose(srh, [37.5, -37.5])
    assert srh.dims == ("time",)


def test_srh_vs_metpy_on_real_sounding():
    mpcalc = pytest.importorskip("metpy.calc")
    from metpy.units import units

    prof = sounding.open_sounding_file(DATA / "uwyo_72235_2022033100.csv")
    prof = prof.where(np.isfinite(prof.u) & np.isfinite(prof.v), drop=True)
    z = prof.height.values - prof.height.values[0]
    for top in (1000.0, 3000.0):
        theirs = mpcalc.storm_relative_helicity(
            z * units.m,
            prof.u.values * units("m/s"),
            prof.v.values * units("m/s"),
            depth=top * units.m,
            storm_u=9.0 * units("m/s"),
            storm_v=-3.0 * units("m/s"),
        )[2].m
        ours = float(wp.storm_relative_helicity(prof, (9.0, -3.0), 0, top))
        assert ours == pytest.approx(theirs, rel=1e-9, abs=1e-6)
    # Bunkers et al. (2000): plain height averages, checked against an
    # independent trapezoidal integration (MetPy weights them by pressure)
    u, v = prof.u.values, prof.v.values

    def mean(lo, hi):
        zz = np.concatenate([[lo], z[(z > lo) & (z < hi)], [hi]])
        return (
            np.trapezoid(np.interp(zz, z, u), zz) / (hi - lo),
            np.trapezoid(np.interp(zz, z, v), zz) / (hi - lo),
        )

    m, low, high = mean(0, 6000), mean(0, 500), mean(5500, 6000)
    su, sv = high[0] - low[0], high[1] - low[1]
    mag = np.hypot(su, sv)
    rm = wp.bunkers_storm_motion(prof)
    assert float(rm.u) == pytest.approx(m[0] + 7.5 * sv / mag, rel=1e-9)
    assert float(rm.v) == pytest.approx(m[1] - 7.5 * su / mag, rel=1e-9)


def test_storm_relative_wind_and_accessors():
    p = straight()
    sr = wp.storm_relative_wind(p, (5.0, 0.0), normal=90.0)
    np.testing.assert_allclose(sr.storm_relative_u, p.u - 5.0)
    np.testing.assert_allclose(sr.storm_relative_normal, p.u - 5.0)
    assert sr.storm_relative_speed.dims == ("height",)
    assert float(p.radarx.bulk_shear(0, 6000).shear_u) == pytest.approx(10.0)
    assert float(p.radarx.bunkers_storm_motion().v) == pytest.approx(-7.5)
    assert float(p.radarx.storm_relative_helicity()) == pytest.approx(37.5)
    with pytest.raises(ValueError, match="'u' and 'v'"):
        wp.bulk_shear(p.drop_vars("v"))
    with pytest.raises(ValueError, match="top"):
        wp.bulk_shear(p, 3000, 1000)


def test_profiler_time_height():
    # (time, height) profiler with gaps: one result per time
    z = np.arange(100.0, 3001.0, 100.0)
    u = np.vstack([z / 300.0, z / 150.0, np.full(z.size, np.nan)])
    ds = xr.Dataset(
        {"u": (("time", "height"), u), "v": (("time", "height"), np.zeros_like(u))},
        coords={"time": np.arange(3).astype("datetime64[h]"), "height": z},
    )
    sh = wp.bulk_shear(ds, 0, 2000)
    np.testing.assert_allclose(sh.shear_u, [2000 / 300, 2000 / 150, np.nan])
    assert sh.shear_u.dims == ("time",)


# ---------------------------------------------------------------------------
# VAD


def vad_sweep(elevation=4.0, u=8.0, v=-3.0, w0=0.5, gaps=False):
    az = np.arange(0.5, 360.0, 1.0)
    rng = np.arange(1000.0, 60001.0, 1000.0)
    a = np.radians(az)[:, None]
    el = np.radians(elevation)
    vr = (u * np.sin(a) + v * np.cos(a)) * np.cos(el) + w0 + 0 * rng
    if gaps:
        vr[:200, :] = np.nan  # less than half a circle valid
    return xr.Dataset(
        {"VRADH": (("azimuth", "range"), vr)},
        coords={
            "azimuth": az,
            "range": rng,
            "elevation": ("azimuth", np.full(az.size, elevation)),
            "altitude": 100.0,
            "latitude": 33.9,
            "longitude": -88.3,
            "time": (
                "azimuth",
                np.full(az.size, np.datetime64("2022-03-31T00:00", "ns")),
            ),
        },
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_vad_synthetic(engine):
    out = wp.vad_profile(vad_sweep(), engine=engine)
    ok = np.isfinite(out.u)
    assert ok.sum() > 10
    np.testing.assert_allclose(out.u[ok], 8.0, rtol=1e-9)
    np.testing.assert_allclose(out.v[ok], -3.0, rtol=1e-9)
    np.testing.assert_allclose(out.vad_rms[ok], 0.0, atol=1e-9)
    assert float(out.wind_direction[ok][0]) == pytest.approx(
        np.degrees(np.arctan2(-8.0, 3.0)) % 360
    )
    assert out.u.attrs["standard_name"] == "eastward_wind"
    assert "time" in out.coords and float(out.altitude) == 100.0
    # too narrow an azimuth sector: rejected
    narrow = wp.vad_profile(vad_sweep(gaps=True), min_spread=0.2, engine=engine)
    assert np.isnan(narrow.u).all()
    near = wp.vad_profile(vad_sweep(), max_range=10000.0, engine=engine)
    assert 0 < int(np.isfinite(near.u).sum()) < int(ok.sum())
    with pytest.raises(KeyError):
        wp.vad_profile(vad_sweep(), max_range=10.0)
    noisy = vad_sweep()
    noisy["VRADH"] = noisy.VRADH + np.random.default_rng(0).normal(
        0, 2, noisy.VRADH.shape
    )
    rej = wp.vad_profile(noisy, max_rms=1.0, engine=engine)
    assert np.isnan(rej.u).all()


def test_vad_datatree_skips_sweeps_without_velocity_and_high_elevations():
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "/sweep_0": vad_sweep(0.5).rename(VRADH="DBZH"),
            "/sweep_1": vad_sweep(4.0),
            "/sweep_2": vad_sweep(60.0),
        }
    )
    out = tree.radarx.vad_profile(height_bins=np.arange(0, 6001.0, 250.0))
    ok = np.isfinite(out.u)
    np.testing.assert_allclose(out.u[ok], 8.0)
    assert out.height.size == 24
    with pytest.raises(KeyError):
        wp.vad_profile(vad_sweep(60.0))
    with pytest.raises(TypeError):
        wp.vad_profile(vad_sweep().VRADH)


def test_vad_real_volume():
    """VAD of the dealiased KLBB volume: smooth, small fit residuals."""
    xd = pytest.importorskip("xradar")
    open_radar_data = pytest.importorskip("open_radar_data")
    from radarx.retrieve import dealias_velocity

    path = open_radar_data.DATASETS.fetch("KLBB20160601_150025_V06")
    tree = xd.io.open_nexradlevel2_datatree(path, sweep=[3])
    sweep = tree["sweep_3"].to_dataset(inherit="all_coords")
    sweep["VRADH"] = sweep.VRADH.where(sweep.VRADH > -63.9)
    from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

    with NEXRADLevel2File(path) as nf:
        h = nf.msg_31_data_header[3]
        nyquist = h["msg_31_data_header"]["RAD"]["nyquist_vel"] / 100.0
    a = wp.vad_profile(sweep)
    corrected = dealias_velocity(sweep, nyquist_velocity=nyquist)
    dz = sweep.assign(VRADH_dealiased=corrected)
    b = wp.vad_profile(dz, "VRADH_dealiased")
    assert np.isfinite(b.u).sum() >= 5
    assert float(b.vad_rms.median()) < 5.0
    assert (b.wind_speed < 60).all() or np.isnan(b.wind_speed).any()
    if "compiled" in ENGINES:
        c = wp.vad_profile(dz, "VRADH_dealiased", engine="numpy")
        np.testing.assert_allclose(b.u, c.u, rtol=1e-9, equal_nan=True)
    assert a.height.attrs["units"] == "m"
