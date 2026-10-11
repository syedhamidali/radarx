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

# numpy 2 renamed trapz to trapezoid; the tests run with both
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

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


def with_pressure(profile, scale_height=8000.0):
    """Add p = 1000 exp(-z / H) hPa to a profile on ``height``."""
    p = 1000.0 * np.exp(-profile.height / scale_height)
    return profile.assign(pressure=p)


@pytest.mark.parametrize("engine", ENGINES)
def test_bunkers_pressure_weighted_means(engine):
    """Issue 233: the pressure-weighted variant, int V dp / (p_bottom - p_top)."""
    h, b = 8000.0, 6000.0
    prof = with_pressure(straight(), h)
    mean_p = (h / 600.0) * (1 - np.exp(-b / h) * (1 + b / h)) / (1 - np.exp(-b / h))
    assert 4.3 < mean_p < 4.5  # against 5.0 for the height-weighted mean
    rm = wp.bunkers_storm_motion(prof, weighting="pressure", engine=engine)
    assert float(rm.u) == pytest.approx(mean_p, rel=2e-3)
    assert float(rm.v) == pytest.approx(-7.5, abs=1e-9)
    assert rm.attrs["weighting"] == "pressure"
    height = wp.bunkers_storm_motion(prof, engine=engine)  # default: unchanged
    assert float(height.u) == pytest.approx(5.0)
    assert height.attrs["weighting"] == "height"
    left = wp.bunkers_storm_motion(
        prof, mover="left", weighting="pressure", engine=engine
    )
    assert float(left.v) == pytest.approx(7.5, abs=1e-9)
    # a column that is a profile of times: one result per time
    times = xr.concat([prof, with_pressure(straight(), 4000.0)], dim="time")
    both = wp.bunkers_storm_motion(times, weighting="pressure", engine=engine)
    assert both.u.dims == ("time",)
    assert float(both.u[0]) == pytest.approx(mean_p, rel=2e-3)
    assert float(both.u[1]) < float(both.u[0])  # thinner air aloft: more weight below
    with pytest.raises(ValueError, match="needs 'pressure'"):
        wp.bunkers_storm_motion(straight(), weighting="pressure", engine=engine)
    with pytest.raises(ValueError, match="weighting"):
        wp.bunkers_storm_motion(prof, weighting="mass")


def test_bunkers_pressure_weighting_engines_agree():
    if not coldpool.HAS_COMPILED_KERNEL:
        pytest.skip("compiled kernel not built")
    prof = with_pressure(straight(dz=50.0))
    a = wp.bunkers_storm_motion(prof, weighting="pressure", engine="numpy")
    b = wp.bunkers_storm_motion(prof, weighting="pressure", engine="compiled")
    np.testing.assert_allclose(a.u, b.u, rtol=1e-12)
    np.testing.assert_allclose(a.v, b.v, rtol=1e-12)


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
            _trapezoid(np.interp(zz, z, u), zz) / (hi - lo),
            _trapezoid(np.interp(zz, z, v), zz) / (hi - lo),
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


def linear_field_sweep(
    u0, v0, ux, uy, vx, vy, elevation, w=0.0, vt=0.0, gaps=0, rng_max=21000.0
):
    """Sweep of a wind field linear in the ground distance (x east, y north)."""
    az = np.arange(0.5, 360.0, 1.0)
    rng = np.arange(1000.0, rng_max, 500.0)
    a = np.radians(az)[:, None]
    el = np.radians(elevation)
    x = rng[None, :] * np.cos(el) * np.sin(a)
    y = rng[None, :] * np.cos(el) * np.cos(a)
    u = u0 + ux * x + uy * y
    v = v0 + vx * x + vy * y
    vr = (u * np.sin(a) + v * np.cos(a)) * np.cos(el) + (w - vt) * np.sin(el)
    if gaps:
        vr[:gaps, :] = np.nan
    return xr.Dataset(
        {"VRADH": (("azimuth", "range"), vr)},
        coords={
            "azimuth": az,
            "range": rng,
            "elevation": ("azimuth", np.full(az.size, float(elevation))),
            "altitude": 100.0,
        },
    )


# (ux, uy, vx, vy): divergence, stretching, shearing and vorticity fields
LINEAR_FIELDS = {
    "uniform": (0.0, 0.0, 0.0, 0.0),
    "divergence": (5e-5, 0.0, 0.0, 5e-5),
    "stretching": (1e-4, 0.0, 0.0, -1e-4),
    "shearing and vorticity": (0.0, 1e-4, 2e-4, 0.0),
    "vorticity": (0.0, -1e-4, 1e-4, 0.0),
    "all terms": (6e-5, -4e-5, 3e-5, 5e-5),
}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", LINEAR_FIELDS)
@pytest.mark.parametrize("elevation", [2.0, 8.0])
def test_vad_divergence_and_deformation_analytic(engine, name, elevation):
    ux, uy, vx, vy = LINEAR_FIELDS[name]
    ds = linear_field_sweep(8.0, -3.0, ux, uy, vx, vy, elevation)
    out = wp.vad_profile(
        ds, height_bins=[0.0, 5000.0], min_elevation=0.5, engine=engine
    )
    # no vertical motion: exact for the linear field
    np.testing.assert_allclose(out.u, 8.0, atol=1e-9)
    np.testing.assert_allclose(out.v, -3.0, atol=1e-9)
    np.testing.assert_allclose(out.divergence, ux + vy, atol=1e-12)
    np.testing.assert_allclose(out.stretching_deformation, ux - vy, atol=1e-12)
    np.testing.assert_allclose(out.shearing_deformation, uy + vx, atol=1e-12)
    np.testing.assert_allclose(out.deformation, np.hypot(ux - vy, uy + vx), atol=1e-12)
    assert out.divergence.attrs["units"] == "s-1"
    assert out.vad_rings_harmonics.item() == out.vad_rings.item() > 0


def test_vad_divergence_vertical_motion_bias_is_documented():
    """w - vt enters the zeroth harmonic as (w - vt) sin(el)."""
    ux, uy, vx, vy = LINEAR_FIELDS["all terms"]
    el = 4.0
    ds = linear_field_sweep(8.0, -3.0, ux, uy, vx, vy, el, w=0.5, vt=3.0)
    out = wp.vad_profile(ds, height_bins=[0.0, 5000.0], min_elevation=0.5)
    # the wind and the deformation do not see it
    np.testing.assert_allclose(out.u, 8.0, atol=1e-9)
    np.testing.assert_allclose(out.stretching_deformation, ux - vy, atol=1e-12)
    np.testing.assert_allclose(out.shearing_deformation, uy + vx, atol=1e-12)
    rng = np.arange(1000.0, 21000.0, 500.0)
    scale = rng * np.cos(np.radians(el)) ** 2
    bias = 2.0 * (0.5 - 3.0) * np.sin(np.radians(el)) / scale
    np.testing.assert_allclose(out.divergence, ux + vy + bias.mean(), rtol=1e-9)


def test_vad_harmonics_engines_agree():
    if "compiled" not in ENGINES:  # pragma: no cover
        pytest.skip("compiled kernel not built")
    ds = linear_field_sweep(8.0, -3.0, *LINEAR_FIELDS["all terms"], 3.0, w=0.4)
    ds["VRADH"] = ds.VRADH + np.random.default_rng(1).normal(0, 1, ds.VRADH.shape)
    a = wp.vad_profile(ds, engine="compiled")
    b = wp.vad_profile(ds, engine="numpy")
    for name in a.data_vars:
        np.testing.assert_allclose(a[name], b[name], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("engine", ENGINES)
def test_vad_second_harmonic_needs_wide_coverage(engine):
    """A sector of 250 degrees keeps the wind but not the deformation."""
    ds = linear_field_sweep(8.0, -3.0, *LINEAR_FIELDS["divergence"], 4.0, gaps=110)
    out = wp.vad_profile(
        ds, height_bins=[0.0, 5000.0], min_elevation=0.5, engine=engine
    )
    np.testing.assert_allclose(out.u, 8.0, atol=1e-9)
    assert np.isnan(out.divergence).all() and np.isnan(out.deformation).all()
    assert out.vad_rings.item() > 0 and out.vad_rings_harmonics.item() == 0
    # a 300 degree sector is enough
    ds = linear_field_sweep(8.0, -3.0, *LINEAR_FIELDS["all terms"], 4.0, gaps=60)
    out = wp.vad_profile(
        ds, height_bins=[0.0, 5000.0], min_elevation=0.5, engine=engine
    )
    ux, uy, vx, vy = LINEAR_FIELDS["all terms"]
    np.testing.assert_allclose(out.divergence, ux + vy, atol=1e-12)


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


# ---------------------------------------------------------------------------
# failure modes reported against MetPy (Unidata/MetPy #596, #788, #178, #794)


@pytest.mark.parametrize("engine", ENGINES)
def test_layer_top_equal_to_last_level_to_rounding(engine):
    # (g + k dz) * 1000 is 9e-13 m below g * 1000 + 6000 for g = 0.347 km; the
    # profile still spans the 0-6 km layer above ground = g * 1000
    g, dz = 0.347, 0.1
    z = (g + np.arange(61) * dz) * 1000.0
    assert z[-1] < g * 1000.0 + 6000.0
    p = xr.Dataset(
        {"u": ("height", np.linspace(0, 10, 61)), "v": ("height", np.zeros(61))},
        coords={"height": z},
    )
    sh = wp.bulk_shear(p, 0, 6000, ground=g * 1000.0, engine=engine)
    assert float(sh.shear_u) == pytest.approx(10.0)
    mw = wp.layer_mean_wind(p, 0, 6000, ground=g * 1000.0, engine=engine)
    assert float(mw.u) == pytest.approx(5.0)
    srh = wp.storm_relative_helicity(
        p, (0.0, 0.0), 0, 6000, ground=g * 1000.0, engine=engine
    )
    assert np.isfinite(float(srh))
    # below the lowest level by rounding, and by a real distance
    sh = wp.bulk_shear(p, 0, 3000, ground=g * 1000.0 - 1e-9, engine=engine)
    assert np.isfinite(float(sh.shear_u))
    sh = wp.bulk_shear(p, 0, 3000, ground=g * 1000.0 - 1.0, engine=engine)
    assert np.isnan(float(sh.shear_u))


@pytest.mark.parametrize("engine", ENGINES)
def test_mean_wind_direction_in_range(engine):
    z = np.arange(0.0, 7001.0, 500.0)
    north = xr.Dataset(
        {
            "u": ("height", np.full(z.size, 1e-17)),
            "v": ("height", np.full(z.size, -5.0)),
        },
        coords={"height": z},
    )
    mw = wp.layer_mean_wind(north, 0, 6000, engine=engine)
    assert float(mw.direction) == 0.0
    calm = north.assign(u=north.u * 0, v=north.v * 0)
    assert np.isnan(float(wp.layer_mean_wind(calm, 0, 6000, engine=engine).direction))
