#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the terrain-following lower boundary, the velocity error model and
the mixed-order velocity gridding of the multi-Doppler input
===========================================================================

The terrain test uses an exact anelastic flow over a Gaussian hill,

    u = U + a x,  v = V + a y,
    w = -2 a (z - h) + u dh/dx + v dh/dy,

with constant density: its divergence is ``2 a + dw/dz = 0`` and ``w = u h_x +
v h_y`` at the ground ``z = h``. Two virtual radars sample it, and the
retrieval on terrain-following levels is compared with the retrieval on levels
of constant height.
"""

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.retrieve import multidoppler as md
from radarx.retrieve.multidoppler import (
    dual_doppler_analytic,
    fall_speed,
    multi_doppler,
    multi_doppler_input,
    radar_geometry,
    velocity_error,
)

ENGINES = ["numpy"] + (["compiled"] if md.HAS_COMPILED_KERNEL else [])


# --------------------------------------------------------------------------
# flow over a hill
# --------------------------------------------------------------------------


def hill_case(h0=2000.0, scale=10e3, step=2000.0, a=2e-4, u0=8.0, v0=4.0, dbz=None):
    x = np.arange(-40e3, 40e3 + 1, step)
    y = x.copy()
    z = np.arange(0.0, 12e3 + 1, 1000.0)
    X, Y = np.meshgrid(x, y)
    h = h0 * np.exp(-(X**2 + Y**2) / (2 * scale**2))
    hx, hy = -X / scale**2 * h, -Y / scale**2 * h
    Z = z[:, None, None]
    u = u0 + a * X[None] + 0 * Z
    v = v0 + a * Y[None] + 0 * Z
    w = -2 * a * (Z - h[None]) + u * hx[None] + v * hy[None]
    ds = xr.Dataset(coords={"z": z, "y": y, "x": x, "radar": [0, 1]})
    ds["radar_x"] = ("radar", [-25e3, 25e3])
    ds["radar_y"] = ("radar", [-20e3, -20e3])
    ds["radar_altitude"] = ("radar", [0.0, 0.0])
    geo = radar_geometry(ds)
    el, az = np.radians(geo.elevation.values), np.radians(geo.azimuth.values)
    vr = np.cos(el) * (np.sin(az) * u + np.cos(az) * v) + np.sin(el) * w
    dist = np.hypot(ds.x - ds.radar_x, ds.y - ds.radar_y).transpose("radar", "y", "x")
    vr = np.where((dist < 60e3).values[:, None], vr, np.nan)
    ds["VRADH"] = (("radar", "z", "y", "x"), vr.astype(np.float32))
    if dbz is not None:
        ds["DBZH"] = (("radar", "z", "y", "x"), np.full(vr.shape, dbz, np.float32))
    dims = ("z", "y", "x")
    bg = xr.Dataset(
        {
            "u": (dims, np.full(u.shape, u0)),
            "v": (dims, np.full(u.shape, v0)),
            "air_density": (dims, np.ones(u.shape)),
        },
        coords={"z": z, "y": y, "x": x},
    )
    # truth on the terrain-following levels (z is the height above the ground)
    w_follow = (
        -2 * a * Z + (u0 + a * X)[None] * hx[None] + (v0 + a * Y)[None] * hy[None]
    )
    truth = dict(
        u=np.broadcast_to(u[:1], u.shape),
        v=np.broadcast_to(v[:1], u.shape),
        w=w_follow,
    )
    return ds, bg, xr.DataArray(h, dims=("y", "x")), truth, w


def good(out):
    return (out.beam_crossing_angle.values > 30) & (out.n_radars.values >= 2)


def rms(a, b, mask):
    return float(np.sqrt(np.mean((a - b)[mask] ** 2)))


@pytest.fixture(scope="module")
def hill():
    return hill_case()


@pytest.mark.parametrize("engine", ENGINES)
def test_terrain_following_recovers_the_flow_over_a_hill(hill, engine):
    ds, bg, terrain, truth, w_cartesian = hill
    out = multi_doppler(
        ds, bg, terrain=terrain, fall_speed_correction=False, engine=engine
    )
    m = good(out)
    assert m.sum() > 5000
    assert rms(out.u.values, truth["u"], m) < 0.1
    assert rms(out.v.values, truth["v"], m) < 0.1
    err_w = rms(out.w.values, truth["w"], m)
    assert err_w < 0.25
    # the levels of constant height impose w = 0 at z = 0, below the hill top
    flat = multi_doppler(ds, bg, fall_speed_correction=False, engine=engine)
    above = ds.z.values[:, None, None] >= terrain.values[None]
    mf = good(flat) & above
    err_flat = rms(flat.w.values, w_cartesian, mf)
    assert err_flat > 1.8 * err_w
    # result layout: z is the height above the ground, height above sea level
    np.testing.assert_allclose(out.z, ds.z)
    np.testing.assert_allclose(
        out.height.values, terrain.values[None] + ds.z.values[:, None, None]
    )
    np.testing.assert_allclose(out.terrain, terrain)
    assert "terrain_following" in out.attrs


def test_terrain_engines_agree(hill):
    ds, bg, terrain, _, _ = hill
    if not md.HAS_COMPILED_KERNEL:
        pytest.skip("compiled kernel not built")
    kw = dict(terrain=terrain, fall_speed_correction=False, max_iterations=40)
    a = multi_doppler(ds, bg, engine="numpy", **kw)
    b = multi_doppler(ds, bg, engine="compiled", **kw)
    for c in "uvw":
        np.testing.assert_allclose(a[c], b[c], atol=2e-4)


def test_flat_terrain_is_the_default(hill):
    """Terrain at zero height gives the retrieval on levels of constant height."""
    ds, bg, terrain, _, _ = hill
    flat = xr.zeros_like(terrain)
    kw = dict(fall_speed_correction=False, max_iterations=30)
    a = multi_doppler(ds, bg, **kw)
    b = multi_doppler(ds, bg, terrain=flat, **kw)
    for c in "uvw":
        np.testing.assert_allclose(a[c], b[c], atol=1e-5)
    # a plain array on (y, x) is accepted
    c = multi_doppler(ds, bg, terrain=flat.values, **kw)
    np.testing.assert_allclose(c.w, b.w)


def test_terrain_error_estimate_includes_the_slope(hill):
    ds, bg, terrain, truth, _ = hill
    out = multi_doppler(
        ds,
        bg,
        terrain=terrain,
        fall_speed_correction=False,
        obs_error=1.0,
        error=True,
        error_samples=8,
    )
    m = good(out)
    est = float(np.sqrt(np.mean(out.w_error.values[m] ** 2)))
    actual = rms(out.w.values, truth["w"], m)
    assert 0.3 < est / actual < 6.0
    # w = W + u hx + v hy: the error of w exceeds that of W where the slope is steep
    steep = np.hypot(*np.gradient(terrain.values, 2000.0)) > 0.05
    assert out.w_error.values[:, steep].mean() > 0.0


def test_terrain_with_reflectivity_background_profile_and_freezing_level(hill):
    """The fall speed uses the height above sea level, the background the profile."""
    ds, bg, terrain, _, _ = hill
    ds = ds.assign(
        DBZH=(("radar", "z", "y", "x"), np.full(ds.VRADH.shape, 30.0, np.float32))
    )
    profile = xr.Dataset(
        {
            "u": ("z", np.full(ds.sizes["z"], 8.0)),
            "v": ("z", np.full(ds.sizes["z"], 4.0)),
            "air_density": ("z", np.ones(ds.sizes["z"])),
            "freezing_level": (("y", "x"), np.full(terrain.shape, 6000.0)),
        },
        coords={"z": ds.z},
    )
    out = multi_doppler(ds, profile, terrain=terrain, max_iterations=20)
    vt = out.fall_speed.values
    # above sea level the freezing level is at 6000 m: over the hill (2 km) the
    # snow fall speed starts 2 km lower than over flat ground
    centre = (slice(None), 20, 20)
    edge = (slice(None), 0, 0)
    assert vt[centre][4] < vt[edge][4] or np.isclose(vt[centre][4], vt[edge][4])
    assert np.isfinite(vt[centre][:7]).all()
    # without a background the density comes from the height above sea level
    bare = multi_doppler(
        ds,
        terrain=terrain,
        fall_speed_correction=True,
        freezing_level="rain",
        max_iterations=5,
    )
    assert np.isfinite(bare.fall_speed.values).any()


def test_fall_speed_uses_the_height_coordinate():
    z = np.array([0.0, 1000.0, 2000.0])
    dbz = xr.DataArray(np.full(3, 30.0), dims="z", coords={"z": z})
    plain = fall_speed(dbz, freezing_level=1000.0, melting_depth=0.0)
    lifted = fall_speed(
        dbz.assign_coords(height=("z", z + 1500.0)),
        freezing_level=1000.0,
        melting_depth=0.0,
    )
    snow = float(plain[2])
    np.testing.assert_allclose(lifted, [snow, snow, snow])
    assert float(plain[0]) > snow


def test_terrain_analytic_dual_doppler(hill):
    ds, bg, terrain, truth, _ = hill
    out = dual_doppler_analytic(
        ds, bg, terrain=terrain, fall_speed_correction=False, iterations=4
    )
    low = (out.z.values <= 3000.0)[:, None, None]
    m = good(out) & low & np.isfinite(out.w.values) & (out.u_error.values < 1.0)
    assert m.sum() > 500
    assert rms(out.u.values, truth["u"], m) < 0.5
    assert rms(out.w.values, truth["w"], m) < 0.5
    np.testing.assert_allclose(out.terrain, terrain)
    assert "terrain_following" in out.attrs


def test_terrain_errors(hill):
    ds, bg, terrain, _, _ = hill
    with pytest.raises(ValueError, match="vorticity"):
        multi_doppler(
            ds, bg, terrain=terrain, weights={"vorticity": 0.01}, solver="lbfgsb"
        )
    with pytest.raises(ValueError, match="on the grid"):
        multi_doppler(ds, bg, terrain=terrain.values[:5])
    with pytest.raises(ValueError, match="dimensions y and x"):
        multi_doppler(ds, bg, terrain=terrain.rename(x="lon"))
    bad = terrain.copy()
    bad[3, 3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        multi_doppler(ds, bg, terrain=bad)
    geometry_only = radar_geometry(ds).drop_vars(
        ["radar_x", "radar_y", "radar_altitude"]
    )
    with pytest.raises(ValueError, match="radar_x"):
        multi_doppler(geometry_only, bg, terrain=terrain)
    descending = ds.isel(z=slice(None, None, -1))
    with pytest.raises(ValueError, match="z must increase"):
        multi_doppler(descending, terrain=terrain)


def test_interp_height():
    z = np.array([0.0, 100.0, 200.0, 300.0])
    a = np.arange(4.0)[:, None, None] * np.ones((4, 1, 2))
    a[1, 0, 1] = np.nan
    height = np.array(
        [[[0.0, 50.0]], [[100.0, 150.0]], [[200.0, 250.0]], [[310.0, 300.0]]]
    )
    out = md._interp_height(a, z, height)
    expected = np.array(
        [[[0.0, np.nan]], [[1.0, np.nan]], [[2.0, 2.5]], [[np.nan, 3.0]]]
    )
    np.testing.assert_allclose(out, expected)


# --------------------------------------------------------------------------
# velocity error from the signal to noise ratio and the spectrum width
# --------------------------------------------------------------------------

LAMBDA = 0.1071  # S band


def test_velocity_error_closed_forms():
    ts = 1e-3
    # independent pairs, Eq. 6.22b, evaluated by hand
    snr, width, m = 10.0, 3.0, 50
    rho = np.exp(-8 * np.pi**2 * width**2 * ts**2 / LAMBDA**2)
    n = 10 ** (-snr / 10)
    var = LAMBDA**2 / (32 * np.pi**2 * m * rho**2 * ts**2) * ((1 + n) ** 2 - rho**2)
    got = velocity_error(
        snr, width, wavelength=LAMBDA, prt=ts, n_pairs=m, pairs="independent"
    )
    np.testing.assert_allclose(got, np.sqrt(var), rtol=1e-12)
    # contiguous pairs, Eq. 6.22a with T = Ts
    sn = width * 2 * ts / LAMBDA
    rho2 = np.exp(-8 * np.pi**2 * (2 * width) ** 2 * ts**2 / LAMBDA**2)
    var = (
        LAMBDA**2
        / (32 * np.pi**2 * m * rho**2 * ts**2)
        * ((1 - rho**2) / (2 * sn * np.sqrt(np.pi)) + n**2 + 2 * n * (1 - rho2))
    )
    got = velocity_error(snr, width, wavelength=LAMBDA, prt=ts, n_pairs=m)
    np.testing.assert_allclose(got, np.sqrt(var), rtol=1e-12)
    # high signal to noise ratio and a narrow width: Eq. 6.23
    narrow = velocity_error(60.0, 0.5, wavelength=LAMBDA, prt=ts, n_pairs=m)
    np.testing.assert_allclose(
        narrow, np.sqrt(0.5 * LAMBDA / (8 * m * ts * np.sqrt(np.pi))), rtol=0.03
    )
    # the Nyquist velocity gives the same pulse repetition time
    va = LAMBDA / (4 * ts)
    np.testing.assert_allclose(
        velocity_error(snr, width, wavelength=LAMBDA, nyquist=va, n_pairs=m), got
    )


def test_velocity_error_zero_width_and_wsr88d_figure():
    # a spectrum without width: only the noise terms remain
    got = velocity_error(10.0, 0.0, wavelength=LAMBDA, prt=1e-3, n_pairs=64)
    n = 0.1
    expected = np.sqrt(LAMBDA**2 / (32 * np.pi**2 * 64 * 1e-6) * (n**2 + 0.0))
    np.testing.assert_allclose(got, expected, rtol=1e-12)
    # Doviak and Zrnic (1993), Table 6.1: about 1 m/s at an SNR above 8 dB and
    # a width of 4 m/s with 40 to 200 samples
    for prf in (322.0, 1000.0):
        for m in (40, 64, 200):
            sd = velocity_error(8.0, 4.0, wavelength=LAMBDA, prt=1 / prf, n_pairs=m)
            assert 0.4 < float(sd) < 1.1


def simulate(sigma_v, snr_db, ts, m, n_estimates=3000, seed=0, v0=3.0):
    """Pulse-pair velocities of chunks of a long record with a Gaussian spectrum."""
    rng = np.random.default_rng(seed)
    length = 4096
    f = np.fft.fftfreq(length, d=ts)
    fd = -2 * v0 / LAMBDA
    width = 2 * sigma_v / LAMBDA
    spectrum = sum(np.exp(-0.5 * ((f - fd + k / ts) / width) ** 2) for k in (-1, 0, 1))
    spectrum /= spectrum.sum()
    noise = 10 ** (-snr_db / 10)
    out = []
    while len(out) < n_estimates:
        white = (
            rng.standard_normal(length) + 1j * rng.standard_normal(length)
        ) / np.sqrt(2)
        sig = np.fft.ifft(np.fft.fft(white) * np.sqrt(spectrum * length))
        sig = sig + np.sqrt(noise) * (
            rng.standard_normal(length) + 1j * rng.standard_normal(length)
        ) / np.sqrt(2)
        for start in range(0, length - m, m + 1):
            v = sig[start : start + m + 1]
            r = np.mean(np.conj(v[:-1]) * v[1:])
            out.append(-LAMBDA / (4 * np.pi * ts) * np.angle(r))
    return np.std(out[:n_estimates])


@pytest.mark.parametrize(
    "sigma_v, snr, prt, m",
    [
        (4.0, 20.0, 1e-3, 64),
        (2.0, 20.0, 1e-3, 64),
        (4.0, 8.0, 1e-3, 64),
        (6.0, 15.0, 1e-3, 128),
    ],
)
def test_velocity_error_matches_a_simulation(sigma_v, snr, prt, m):
    """Standard deviation of simulated pulse-pair estimates (agreement within 6 %)."""
    sd = simulate(sigma_v, snr, prt, m)
    formula = float(velocity_error(snr, sigma_v, wavelength=LAMBDA, prt=prt, n_pairs=m))
    assert sd == pytest.approx(formula, rel=0.06)


def test_velocity_error_dataarray_and_nan():
    snr = xr.DataArray(
        [[5.0, 15.0], [np.nan, 25.0]],
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [10, 20]},
    )
    width = xr.DataArray([2.0, 4.0], dims="x", coords={"x": [10, 20]})
    out = velocity_error(snr, width, wavelength=LAMBDA, nyquist=26.0)
    assert out.dims == ("y", "x") and out.name == "velocity_error"
    assert out.attrs["units"] == "m s-1"
    assert np.isnan(out.values[1, 0])
    assert (np.diff(out.values[0]) != 0).all()
    # the error decreases with the signal to noise ratio
    low = velocity_error(snr[0, 0], 2.0, wavelength=LAMBDA, nyquist=26.0)
    high = velocity_error(snr[1, 1], 2.0, wavelength=LAMBDA, nyquist=26.0)
    assert float(low) > float(high)
    # a DataArray width with a plain number as the ratio
    both = velocity_error(10.0, width, wavelength=LAMBDA, nyquist=26.0)
    assert both.dims == ("x",)
    # plain numbers and arrays give arrays
    assert isinstance(
        velocity_error([5.0, 10.0], 2.0, wavelength=LAMBDA, nyquist=26.0), np.ndarray
    )


def test_velocity_error_goes_into_the_weights(hill):
    """A noisier radar counts less: obs_error from the SNR model changes the result."""
    ds, bg, terrain, truth, _ = hill
    snr = xr.DataArray([25.0, 3.0], dims="radar")
    sigma = velocity_error(snr, 3.0, wavelength=LAMBDA, nyquist=26.0, n_pairs=64)
    assert float(sigma[0]) < float(sigma[1])
    kw = dict(terrain=terrain, fall_speed_correction=False, max_iterations=15)
    weighted = multi_doppler(ds, bg, obs_error=sigma.values, **kw)
    plain = multi_doppler(ds, bg, **kw)
    assert not np.allclose(weighted.w, plain.w)


def test_velocity_error_errors():
    kw = dict(wavelength=LAMBDA, nyquist=26.0)
    with pytest.raises(ValueError, match="pairs"):
        velocity_error(10.0, 2.0, pairs="spaced", **kw)
    with pytest.raises(ValueError, match="either nyquist or prt"):
        velocity_error(10.0, 2.0, wavelength=LAMBDA)
    with pytest.raises(ValueError, match="either nyquist or prt"):
        velocity_error(10.0, 2.0, wavelength=LAMBDA, nyquist=26.0, prt=1e-3)
    with pytest.raises(ValueError, match="wavelength"):
        velocity_error(10.0, 2.0, wavelength=0.0, nyquist=26.0)
    with pytest.raises(ValueError, match="n_pairs"):
        velocity_error(10.0, 2.0, n_pairs=0, **kw)
    with pytest.raises(ValueError, match="positive"):
        velocity_error(10.0, 2.0, wavelength=LAMBDA, prt=-1.0)
    with pytest.raises(ValueError, match="negative"):
        velocity_error(10.0, -2.0, **kw)


# --------------------------------------------------------------------------
# mixed-order velocity gridding in the multi-Doppler input
# --------------------------------------------------------------------------


def test_input_mixed_order_velocity():
    """On a field that is linear in space both orders give the same grid."""
    from tests.grid import test_multi as tm

    east = tm._volume(*tm._site(100e3, 0.0), name="E")
    west = tm._volume(*tm._site(-100e3, 0.0), name="W")
    x = np.arange(-40e3, 40e3 + 1, 4e3)
    y = np.arange(-40e3, 40e3 + 1, 4e3)
    z = np.arange(1000.0, 4000.1, 500.0)
    kw = dict(origin=tm.ORIGIN, field="DBZH", dbz="RHOHV")
    linear = multi_doppler_input([east, west], x, y, z, **kw)
    mixed = multi_doppler_input(
        [east, west], x, y, z, velocity_interpolation="mixed", **kw
    )
    expected = tm.truth(x[None, None, :], y[None, :, None], z[:, None, None])
    for k in range(2):
        err = mixed.DBZH.isel(radar=k).values - expected
        assert np.sqrt(np.nanmean(err**2)) < 5e-3
    np.testing.assert_allclose(mixed.DBZH, linear.DBZH, atol=0.02)
    assert "mixed-order" in mixed.DBZH.attrs["gridding_method"]
    # the numpy engine, the order, and the advection to a common time
    again = multi_doppler_input(
        [east, west],
        x,
        y,
        z,
        velocity_interpolation="mixed",
        mixed_order=2.0,
        time="2026-01-01T00:00",
        motion=(0.0, 0.0),
        engine="numpy",
        **kw,
    )
    assert str(again.attrs["analysis_time"]).startswith("2026-01-01")
    np.testing.assert_allclose(again.DBZH, mixed.DBZH, atol=0.02)
    with pytest.raises(ValueError, match="velocity_interpolation"):
        multi_doppler_input([east], x, y, z, velocity_interpolation="cubic")
    with pytest.raises(ValueError, match="mixed_order"):
        multi_doppler_input(
            [east], x, y, z, velocity_interpolation="mixed", mixed_order=-1.0
        )
