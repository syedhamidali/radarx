#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the tropical-cyclone centre, GBVTD and the decomposition
===================================================================

The truth is an analytic vortex written from the definitions: a modified
Rankine-combined vortex (Eqs. 28 to 31 of Lee et al. 1999) with a uniform mean
wind and wavenumber-1 and 2 tangential asymmetries, projected on the beams with
the exact geometry of a radar at the origin. The retrieval is compared with the
known profiles, the ring fits and the simplex search of the compiled kernel with
the NumPy implementation, the NORM score with Table 1 of Bell and Lee (2012) and
the closure with Eqs. 13 to 27 of Lee et al. (1999). The published numbers of
the analytic experiments of Lee et al. (1999) and Lee and Marks (2000) are
checked in the last section. Where the method has a known error (the geometric
distortion of the wavenumber-1 wind, the aliased cross-beam mean wind, the
dependence on the centre) the test states the size of the error it expects.
"""

import numpy as np
import pytest
import xarray as xr
from xradar.georeference import antenna_to_cartesian

import radarx  # noqa: F401
from radarx._provenance import HISTORY, METHOD, REFERENCES, load_registry
from radarx.fundamentals.constants import EARTH_RADIUS
from radarx.retrieve import (
    gbvtd,
    synthetic_vortex,
    tc_center,
    tc_decompose,
    vortex_profile,
)
from radarx.retrieve import tc_winds as tw

HAS_KERNEL = tw.HAS_COMPILED_KERNEL
ENGINES = ["numpy"] + (["compiled"] if HAS_KERNEL else [])
compiled_only = pytest.mark.skipif(not HAS_KERNEL, reason="compiled kernel not built")

VMAX, RMW, XT = 50.0, 20000.0, 0.6
CENTER = (0.0, 80000.0)
RADII = np.arange(5000.0, 50001.0, 5000.0)


def vortex(**kwargs):
    """A coarse sweep of the vortex (fast for the NumPy engine)."""
    options = {
        "center": CENTER,
        "vmax": VMAX,
        "rmw": RMW,
        "xt": XT,
        "azimuth_step": 1.0,
        "range_step": 500.0,
        "max_range": 130000.0,
    }
    options.update(kwargs)
    return synthetic_vortex(**options)


def small_vortex():
    """A coarse sweep for the generic product tests."""
    return vortex(azimuth_step=4.0, max_range=100000.0, asymmetry={1: (0.2, 45.0)})


def small_tc_center(ds=None):
    return tc_center(
        small_vortex() if ds is None else ds,
        radii=np.array([19000.0, 20000.0, 21000.0]),
        guess_grid=2,
    )


def truth(radii=RADII, radial=None):
    return vortex_profile(radii, VMAX, RMW, XT, radial)


def rms(a):
    return float(np.sqrt(np.nanmean(np.asarray(a) ** 2)))


# --------------------------------------------------------------------------
# the analytic vortex
# --------------------------------------------------------------------------


def test_vortex_profile_values():
    r = np.array([0.0, 10000.0, 20000.0, 40000.0])
    vt, vr = vortex_profile(r, 50.0, 20000.0, 0.6)
    assert np.allclose(vt, [0.0, 25.0, 50.0, 50.0 * 0.5**0.6])
    assert np.all(vr == 0.0)
    vt, vr = vortex_profile(r, 50.0, 20000.0, 1.0, radial=(0.1, 3.0))
    # Eq. 16 with the radii in km: outflow 0.1 sqrt((20 - 10) 10) = 1 m/s inside,
    # inflow -3 sqrt(40 - 20) 20 / 40 = -6.7 m/s outside, zero at the centre
    assert np.allclose(vr, [0.0, 1.0, 0.0, -3.0 * np.sqrt(20.0) * 0.5])
    assert np.allclose(vt[3], 25.0)


def test_synthetic_geometry_matches_xradar():
    ds = vortex(elevation=1.5, site=(20.0, 80.0, 300.0))
    az = np.array([10.5, 200.5])
    rng = ds["range"].values
    x, y, z = antenna_to_cartesian(
        rng[None, :],
        az[:, None],
        1.5,
        site_altitude=300.0,
        effective_radius_fraction=4.0 / 3.0,
    )
    sweep = tw._Sweep(ds, None, 4.0 / 3.0)
    sel = ds.indexes["azimuth"].get_indexer(az)
    assert np.allclose(sweep.x.values[sel], x, atol=1e-6, rtol=0)
    assert np.allclose(sweep.y.values[sel], y, atol=1e-6, rtol=0)
    # the beam height above the radar is xradar's z minus the site altitude
    assert np.allclose(ds["BEAM_HEIGHT"].values[sel], z - 300.0, atol=1e-6)


def test_local_elevation_of_the_beam():
    # the ray is straight in the effective Earth: the angle between it and the
    # local horizontal at the gate is the elevation plus the central angle
    ds = vortex(elevation=2.0, site=(20.0, 80.0, 100.0))
    reff = 4.0 / 3.0 * EARTH_RADIUS
    rng = ds["range"].values
    el = np.deg2rad(2.0)
    point = np.stack([rng * np.cos(el), reff + 100.0 + rng * np.sin(el)])
    n = point / np.hypot(*point)
    ray = np.array([np.cos(el), np.sin(el)])
    local = np.arcsin(ray @ n)
    sweep = tw._Sweep(ds, None, 4.0 / 3.0)
    s = sweep.w.values[0]
    expected = el + s / reff
    assert np.allclose(local, expected, atol=1e-9)
    assert np.allclose(
        sweep.vh.values, ds["VRADH"].values / np.cos(expected)[None, :], equal_nan=True
    )


def test_synthetic_pure_mean_wind_is_its_projection():
    ds = vortex(vmax=0.0, mean_wind=(7.0, -4.0), elevation=3.0)
    sweep = tw._Sweep(ds, None, 4.0 / 3.0)
    az = np.deg2rad(ds["azimuth"].values)[:, None]
    phi = np.deg2rad(3.0) + sweep.w.values / (4.0 / 3.0 * EARTH_RADIUS)
    expected = (7.0 * np.sin(az) - 4.0 * np.cos(az)) * np.cos(phi)
    assert np.allclose(ds["VRADH"].values, expected)
    assert np.allclose(ds["VT_TRUE"], 0.0)


def test_synthetic_vortex_is_cyclonic_and_outflow_is_positive():
    # a vortex 80 km north of the radar: a cyclonic (anticlockwise) wind blows
    # north (away from the radar) east of the centre and south (towards it) west
    # of it
    ds = vortex(vmax=40.0, xt=1.0)
    east = ds["VRADH"].sel(azimuth=14.5, range=82000.0, method="nearest")
    west = ds["VRADH"].sel(azimuth=345.5, range=82000.0, method="nearest")
    assert float(east) > 10.0 and float(west) < -10.0
    # outflow inside the radius of maximum wind is away from the radar on the far side
    out = vortex(vmax=0.0, radial=(0.1, 3.0), xt=1.0)
    far = out["VRADH"].sel(azimuth=0.5, range=90000.0, method="nearest")
    assert float(far) > 0.5


def test_synthetic_options():
    ds = vortex(noise=2.0, seed=4, azimuth_step=2.0, max_range=90000.0)
    clean = vortex(azimuth_step=2.0, max_range=90000.0)
    diff = (ds["VRADH"] - clean["VRADH"]).values
    assert abs(np.nanstd(diff) - 2.0) < 0.05
    again = vortex(noise=2.0, seed=4, azimuth_step=2.0, max_range=90000.0)
    assert np.array_equal(again["VRADH"].values, ds["VRADH"].values, equal_nan=True)
    assert ds.sizes == {"azimuth": 180, "range": 180}
    assert ds.attrs["vortex_vmax"] == VMAX and ds.attrs["mean_wind_u"] == 0.0
    # a blank sector, seen from the vortex centre, leaves the data of the others
    blank = vortex(blank_sectors=[(150.0, 210.0), (350.0, 10.0)])
    sweep = tw._Sweep(blank, None, 4.0 / 3.0)
    gamma = np.rad2deg(np.arctan2(sweep.x.values, sweep.y.values - 80000.0)) % 360
    gamma = np.rad2deg(
        np.arctan2(sweep.x.values - CENTER[0], sweep.y.values - CENTER[1])
    )
    gamma = gamma % 360.0
    vel = blank["VRADH"].values
    inside = ((gamma >= 150) & (gamma <= 210)) | (gamma >= 350) | (gamma <= 10)
    assert np.isnan(vel[inside]).all() and np.isfinite(vel[~inside]).all()
    # the vortex wind decays exponentially with the beam height
    fall = vortex(vmax=40.0, height_scale=10000.0, elevation=2.0)
    flat = vortex(vmax=40.0, elevation=2.0)
    z = fall["BEAM_HEIGHT"].values
    assert np.allclose(
        fall["VT_TRUE"].values, flat["VT_TRUE"].values * np.exp(-z / 1e4)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rmw": 0.0},
        {"vmax": -1.0},
        {"range_step": 0.0},
        {"azimuth_step": -1.0},
        {"max_range": 100.0},
    ],
)
def test_synthetic_errors(kwargs):
    with pytest.raises(ValueError):
        synthetic_vortex(**kwargs)


# --------------------------------------------------------------------------
# kernel and NumPy implementation
# --------------------------------------------------------------------------


def gate_arrays(ds):
    sweep = tw._Sweep(ds, None, 4.0 / 3.0)
    return tw._gate_arrays(
        sweep.vh.values, sweep.x.values, sweep.y.values, sweep.w.values
    )


CFG = {
    "half_width": 1000.0,
    "max_order": 4,
    "min_points": 30,
    "max_gap": 120.0,
    "geometric_limit": False,
    "mean_wind": (0.0, 0.0),
}


@compiled_only
def test_ring_fits_agree_between_kernel_and_numpy():
    ds = vortex(
        asymmetry={1: (0.2, 45.0), 2: (0.1, 100.0)},
        mean_wind=(8.0, -3.0),
        noise=1.5,
        blank_sectors=[(200.0, 280.0)],
    )
    gates = gate_arrays(ds)
    cx = np.array([0.0, 800.0, -1500.0, 3000.0, 0.0])
    cy = np.array([80000.0, 80500.0, 79000.0, 82000.0, 5000.0])
    radii = np.array([500.0, 3000.0, 15000.0, 20000.0, 33000.0, 90000.0])
    a = tw._rings(gates, cx, cy, radii, CFG, True, 2)
    b = tw._rings(gates, cx, cy, radii, CFG, False, None)
    assert np.array_equal(np.isnan(a[0]), np.isnan(b[0]))
    assert np.array_equal(np.isnan(a[1]), np.isnan(b[1]))
    assert np.allclose(a[0], b[0], rtol=1e-8, atol=1e-8, equal_nan=True)
    assert np.allclose(a[1], b[1], rtol=1e-8, atol=1e-8, equal_nan=True)
    # some rings fit, some fail (too few gates, the radar inside the ring, the gap)
    fitted = np.isfinite(a[0][..., 0])
    assert fitted.any() and (~fitted).any()


@compiled_only
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_ring_orders_agree(order):
    ds = vortex(asymmetry={1: (0.2, 45.0)}, noise=1.0)
    gates = gate_arrays(ds)
    cfg = dict(CFG, max_order=order)
    a = tw._rings(gates, [0.0], [80000.0], np.array([20000.0]), cfg, True, 1)
    b = tw._rings(gates, [0.0], [80000.0], np.array([20000.0]), cfg, False, 1)
    assert np.allclose(a[0], b[0], rtol=1e-8, atol=1e-8)
    assert a[1][0, 0, 2] == order == b[1][0, 0, 2]
    assert np.all(a[0][0, 0, 2 * order + 1 :] == 0.0)


@compiled_only
def test_simplex_search_agrees_between_kernel_and_numpy():
    ds = vortex(asymmetry={1: (0.15, 45.0)}, noise=2.0, seed=5, mean_wind=(4.0, 2.0))
    gates = gate_arrays(ds)
    starts = np.array([(-3000.0, 76000.0), (2500.0, 84000.0), (5000.0, 78000.0)])
    radii = np.array([14000.0, 20000.0, 28000.0])
    cfg = dict(CFG, max_order=2, mean_wind=(4.0, 2.0))
    search = {
        "step": 4000.0,
        "tolerance": 50.0,
        "max_iter": 200,
        "limit": 25000.0,
        "anchor": (0.0, 80000.0),
    }
    a = tw._search(gates, starts, radii, cfg, search, True, 3)
    b = tw._search(gates, starts, radii, cfg, search, False, None)
    assert np.allclose(a[..., :3], b[..., :3], rtol=1e-6, atol=1e-3)
    assert np.array_equal(a[..., 3:], b[..., 3:])
    assert (a[..., 4] == 1.0).all()


def test_search_rejects_starts_outside_the_limit_and_without_data():
    ds = vortex(max_range=60000.0)  # no data around the centre 80 km away
    gates = gate_arrays(ds)
    cfg = dict(CFG, max_order=2)
    search = {
        "step": 2000.0,
        "tolerance": 50.0,
        "max_iter": 50,
        "limit": 5000.0,
        "anchor": (0.0, 80000.0),
    }
    for compiled in {False, HAS_KERNEL}:
        out = tw._search(
            gates, [(0.0, 80000.0)], np.array([20000.0]), cfg, search, compiled, 1
        )
        assert np.isnan(out[0, 0, :3]).all() and out[0, 0, 4] == 0.0
        # a centre closer to the radar than the ring radius: the radar is inside
        inner = tw._search(
            gates, [(0.0, 80000.0)], np.array([85000.0]), cfg, search, compiled, 1
        )
        assert np.isnan(inner[0, 0, 0])
        far = dict(search, limit=100.0)
        out = tw._search(
            gates, [(30000.0, 90000.0)], np.array([20000.0]), cfg, far, compiled, 1
        )
        assert np.isnan(out[0, 0, 0])


@compiled_only
def test_kernel_argument_checks():
    k = tw._tc_winds
    x = np.zeros(3)
    with pytest.raises(ValueError, match="same length"):
        k.gbvtd_rings(x, x, x, np.zeros(2), [0.0], [1.0], [1.0], 1000.0)
    with pytest.raises(ValueError, match="1-D"):
        k.gbvtd_rings(x, x, x, x, [0.0], [1.0, 2.0], [1.0], 1000.0)
    with pytest.raises(ValueError, match="positive"):
        k.gbvtd_rings(x, x, x, x, [0.0], [1.0], [1.0], 0.0)
    with pytest.raises(ValueError, match="max_order"):
        k.gbvtd_rings(x, x, x, x, [0.0], [1.0], [1.0], 1000.0, max_order=5)
    with pytest.raises(ValueError, match="same length"):
        k.center_search(x, x, x, np.zeros(2), [[0.0, 0.0]], [1.0], 1000.0)
    with pytest.raises(ValueError, match="shape"):
        k.center_search(x, x, x, x, [0.0, 0.0], [1.0], 1000.0)
    with pytest.raises(ValueError, match="positive"):
        k.center_search(x, x, x, x, [[0.0, 0.0]], [1.0], 0.0)
    with pytest.raises(ValueError, match="max_order"):
        k.center_search(x, x, x, x, [[0.0, 0.0]], [1.0], 1000.0, max_order=-1)
    # no gates at all: every ring fails, nothing crashes
    out = k.gbvtd_rings(
        np.zeros(0),
        np.zeros(0),
        np.zeros(0),
        np.zeros(0),
        [0.0],
        [5000.0],
        [1000.0],
        500.0,
    )
    assert np.isnan(out[0]).all()


# --------------------------------------------------------------------------
# GBVTD against the analytic vortex
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_axisymmetric_vortex_is_retrieved(engine):
    ds = vortex(radial=(0.1, 3.0))
    g = gbvtd(ds, CENTER, radii=RADII, engine=engine)
    vt, vr = truth(radial=(0.1, 3.0))
    away = np.abs(RADII - RMW) > 2000.0  # the kink at the radius of maximum wind
    assert np.abs(g["VT0"].values - vt)[away].max() < 0.2
    assert np.abs(g["VR0"].values - vr)[away].max() < 0.05
    # no mean wind and no asymmetry: the other coefficients vanish
    assert np.abs(g["VT1_AMPLITUDE"].values[away]).max() < 0.3
    assert np.abs(g["VT2_AMPLITUDE"].values[away]).max() < 0.4
    assert np.abs(g["VM_ALONG"].values[away]).max() < 0.1
    assert np.abs(g["VT3_AMPLITUDE"].values[away]).max() < 0.5
    assert (g["FIT_ORDER"] == 4).all()
    fine = gbvtd(ds, CENTER, radii=np.arange(5000.0, 40001.0, 1000.0), engine=engine)
    assert abs(float(fine["RMW"]) - RMW) < 500.0
    assert abs(float(fine["VT_MAX"]) - VMAX) < 1.5


def test_cross_beam_mean_wind_is_aliased_and_removed():
    # a mean wind blowing in the cross-beam direction (east) of 10 m/s: the
    # centre is north of the radar, so V_M sin(theta_T - theta_M) = 10
    ds = vortex(mean_wind=(10.0, 0.0), radial=(0.1, 3.0))
    vt, vr = truth(radial=(0.1, 3.0))
    away = np.abs(RADII - RMW) > 2000.0
    plain = gbvtd(ds, CENTER, radii=RADII)
    fixed = gbvtd(ds, CENTER, radii=RADII, mean_wind=(10.0, 0.0))
    # the cross-beam wind leaks in as V_M sin(alpha_max) = 10 R / R_T
    leak = 10.0 * RADII / CENTER[1]
    assert np.allclose((plain["VT0"].values - vt)[away], leak[away], atol=0.3)
    assert np.abs((fixed["VT0"].values - vt)[away]).max() < 0.2
    # the along-beam component of the mean wind is found from the fit (A0 + A2 + A4)
    assert np.abs(plain["VM_ALONG"].values[away]).max() < 0.2
    # wind along the beam (north): found, and no aliasing into VT0
    ds = vortex(mean_wind=(0.0, -6.0))
    g = gbvtd(ds, CENTER, radii=RADII)
    assert np.allclose(g["VM_ALONG"].values[away], -6.0, atol=0.2)
    assert np.abs((g["VT0"].values - vt)[away]).max() < 0.2
    # a mean wind in any direction, with the centre off the axis
    ds = vortex(center=(30000.0, 70000.0), mean_wind=(-5.0, 7.0))
    g = gbvtd(ds, (30000.0, 70000.0), radii=RADII, mean_wind=(-5.0, 7.0))
    assert np.abs((g["VT0"].values - vt)[away]).max() < 0.3
    along = (30000.0 * -5.0 + 70000.0 * 7.0) / np.hypot(30000.0, 70000.0)
    assert np.allclose(g["VM_ALONG"].values[away], along, atol=0.3)


def test_wavenumber_one_and_two_asymmetries():
    spec = {1: (0.2, 45.0), 2: (0.1, 100.0)}
    ds = vortex(asymmetry=spec)
    g = gbvtd(ds, CENTER, radii=RADII)
    vt, _ = truth()
    away = np.abs(RADII - RMW) > 2000.0
    # the coefficients of the series in psi (the quantities of Table 1 of Lee and
    # Marks 2000) are within 10 percent of VT0 of the amplitudes
    wn1 = g["VT1_AMPLITUDE"].values
    wn2 = g["VT2_AMPLITUDE"].values
    assert np.abs(wn1 - 0.2 * vt)[away].max() < 0.1 * vt[away].min() + 1.5
    assert np.abs(wn2 - 0.1 * vt)[away].max() < 0.1 * vt[away].min() + 2.0
    # the series in psi alone gives a rough position of the maximum
    err = (g["VT1_AZIMUTH"].values - 45.0 + 180.0) % 360.0 - 180.0
    assert np.abs(err[away]).max() < 15.0
    assert np.abs(g["VT0"].values - vt)[away].max() < 2.0
    # the waves of the azimuth (the comparison of Fig. 13 of Lee et al. 1999)
    # are found much better when the series goes up to the order 4
    d = tc_decompose(g)
    amp = d["AZIMUTHAL_AMPLITUDE"].sel(wavenumber=1).values
    phase = d["AZIMUTHAL_PHASE"].sel(wavenumber=1).values
    assert np.abs(amp - 0.2 * vt)[away].max() < 0.5
    assert np.abs(phase - 45.0)[away].max() < 2.0
    amp2 = d["AZIMUTHAL_AMPLITUDE"].sel(wavenumber=2).values
    assert np.abs(amp2 - 0.1 * vt)[away].max() < 1.0
    low = tc_decompose(gbvtd(ds, CENTER, radii=RADII, max_wavenumber=3))
    low_phase = low["AZIMUTHAL_PHASE"].sel(wavenumber=1).values
    assert np.abs(low_phase - 45.0)[away].max() > np.abs(phase - 45.0)[away].max()


def test_wavenumber_three_tangential_wind():
    # Eqs. 26 and 27: V_TS3 = 2 A4 and V_TC3 = -2 B4, found for a ring close to
    # the radar line (sin(alpha_max) = 0.125) to within 5 percent
    ds = vortex(absolute_asymmetry={3: (10.0, 25.0)}, azimuth_step=0.5)
    g = gbvtd(ds, CENTER, radii=np.array([10000.0]), ring_width=1000.0)
    assert abs(float(g["VT3_AMPLITUDE"].values[0]) - 10.0) < 0.6
    assert (g["FIT_ORDER"] == 4).all()
    d = tc_decompose(g)
    assert abs(float(d["AZIMUTHAL_AMPLITUDE"].sel(wavenumber=3).values[0]) - 10.0) < 0.6
    # a fit of the order 3 has no wavenumber 3 and the waves above are zero
    low = gbvtd(ds, CENTER, radii=np.array([10000.0]), max_wavenumber=3)
    assert np.isnan(low["VT3_AMPLITUDE"]).all()
    assert np.allclose(tc_decompose(low)["VT_WAVENUMBER3"], 0.0)


def test_asymmetric_radial_wind_leaks_into_the_wavenumber_one_wind():
    # the closure V_R1 = 0: a wavenumber-1 radial wind of 3 m/s is not found and
    # shows up as a wavenumber-1 tangential wind of a few m/s
    ds = vortex(radial_asymmetry={1: (3.0, 120.0)})
    g = gbvtd(ds, CENTER, radii=RADII)
    away = np.abs(RADII - RMW) > 2000.0
    leak = g["VT1_AMPLITUDE"].values[away]
    assert 1.0 < leak.max() < 6.0
    assert np.abs(g["VR0"].values[away]).max() < 0.7  # the mean radial wind is fine


def test_beam_height_and_noise_have_a_bounded_effect():
    vt, _ = truth()
    away = np.abs(RADII - RMW) > 2000.0
    noisy = gbvtd(vortex(noise=3.0, seed=2), CENTER, radii=RADII)
    assert rms((noisy["VT0"].values - vt)[away]) < 0.3
    # the vortex weakens with height and the beam climbs: the retrieval is the
    # wind at the height of the beam over the ring (the beam height at the centre)
    high = vortex(elevation=4.0, height_scale=20000.0)
    g = gbvtd(high, CENTER, radii=RADII)
    z = float(high["BEAM_HEIGHT"].sel(range=80000.0, method="nearest").isel(azimuth=0))
    ratio = (g["VT0"].values / vt)[away]
    assert np.allclose(ratio, np.exp(-z / 20000.0), atol=0.08)


def test_centre_error_sensitivity():
    spec = {1: (0.2, 45.0)}
    ds = vortex(asymmetry=spec, mean_wind=(10.0, 0.0), noise=1.0, seed=1)
    vt, _ = truth()
    away = np.abs(RADII - RMW) > 2000.0
    radii = np.arange(5000.0, 50001.0, 1000.0)
    away = np.abs(radii - RMW) > 2000.0
    vt, _ = truth(radii)
    errs = {}
    for d in (0.0, 1000.0, 2000.0, 5000.0):
        c = (CENTER[0] + d, CENTER[1])
        g = gbvtd(ds, c, radii=radii, mean_wind=(10.0, 0.0))
        a1 = rms(((g["VT1_AMPLITUDE"].values - 0.2 * vt) / (0.2 * vt))[away])
        errs[d] = (rms(((g["VT0"].values - vt) / vt)[away]), a1)
    # the wavenumber-1 wind is much more sensitive to the centre than the mean
    # tangential wind (Lee and Marks 2000, section 2: the error of a real
    # wavenumber 1 of 10 m/s is 20 percent for 1 km in a storm with a radius of
    # maximum wind of 20 km)
    assert errs[1000.0][1] > 0.14 and errs[1000.0][1] > 1.5 * errs[0.0][1]
    assert errs[2000.0][1] > errs[1000.0][1] * 0.9
    assert errs[5000.0][1] > 0.25
    assert errs[5000.0][0] > errs[0.0][0] and errs[1000.0][0] < 0.06
    assert errs[5000.0][0] < 0.12


def test_data_gaps_reduce_the_fit_order():
    vt, _ = truth()
    # a gap in the data of 40 degrees on the ring seen from the centre, towards
    # the radar (south): order 2; 90 degrees: order 1; 140 degrees: no fit (the
    # sampling of the gates adds some degrees to the gap)
    for width, order in ((40.0, 2), (90.0, 1), (140.0, None)):
        lo, hi = 180.0 - width / 2, 180.0 + width / 2
        ds = vortex(blank_sectors=[(lo, hi)])
        g = gbvtd(ds, CENTER, radii=np.array([20000.0]))
        if order is None:
            assert np.isnan(g["VT0"].values).all()
            assert np.isnan(g["DATA_GAP"]).all()
        else:
            assert (g["FIT_ORDER"].values == order).all()
            assert (g["DATA_GAP"].values >= width - 5.0).all()
            assert np.isfinite(g["VT0"].values).all()
    # the wavenumbers that the order does not resolve are NaN
    ds = vortex(blank_sectors=[(160.0, 200.0)])
    g = gbvtd(ds, CENTER, radii=np.array([20000.0]))
    assert np.isfinite(g["VT1_COS"]).all() and np.isnan(g["VT2_COS"]).all()
    ds = vortex(blank_sectors=[(135.0, 225.0)])
    g = gbvtd(ds, CENTER, radii=np.array([20000.0]))
    assert np.isnan(g["VT1_COS"]).all() and np.isfinite(g["VT0"]).all()
    assert np.isnan(g["VM_ALONG"]).all()
    # max_wavenumber limits the order even for complete data
    g = gbvtd(vortex(), CENTER, radii=np.array([20000.0]), max_wavenumber=1)
    assert (g["FIT_ORDER"] == 1).all()
    # max_gap rejects rings
    g = gbvtd(
        vortex(blank_sectors=[(160.0, 200.0)]),
        CENTER,
        radii=np.array([20000.0]),
        max_gap=30.0,
    )
    assert np.isnan(g["VT0"]).all()


def test_rings_that_cannot_exist_are_nan():
    ds = vortex()
    radii = np.array([5000.0, 20000.0, 80500.0, 90000.0])
    g = gbvtd(ds, CENTER, radii=radii)
    # the radar (80 km from the centre) must be outside the ring
    assert np.isfinite(g["VT0"].values[:2]).all()
    assert np.isnan(g["VT0"].values[2:]).all()
    assert np.isnan(g["VT1_AZIMUTH"].values[3])
    # no data at all
    empty = ds.assign(VRADH=ds["VRADH"] * np.nan)
    g = gbvtd(empty, CENTER, radii=radii)
    assert np.isnan(g["VT0"]).all() and np.isnan(g["RMW"])


def test_radius_of_maximum_wind_is_refined_with_a_parabola():
    radii = np.arange(5000.0, 40001.0, 1000.0)
    ds = vortex(rmw=21300.0)
    g = gbvtd(ds, CENTER, radii=radii)
    assert abs(float(g["RMW"]) - 21300.0) < 500.0
    # the maximum at the edge of the radii is not a maximum
    g = gbvtd(ds, CENTER, radii=np.arange(5000.0, 18001.0, 2500.0))
    assert np.isnan(g["RMW"]) and np.isnan(g["VT_MAX"])
    # peak of an exact parabola, and a flat top
    r = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(tw._peak(r, -((r - 2.4) ** 2) + 5.0), [2.4, 5.0])
    assert np.allclose(tw._peak(r, np.array([1.0, 3.0, 3.0, 1.0]))[1], 3.25)
    assert np.allclose(tw._peak(r, np.array([0.0, 1.0, 1.0, 1.0]))[0], 2.5)
    assert np.isnan(tw._peak(r, np.array([1.0, np.nan, 2.0, 1.0]))).all()
    assert np.isnan(tw._peak(r[:2], np.array([1.0, 2.0]))).all()


# --------------------------------------------------------------------------
# centre finding
# --------------------------------------------------------------------------


# Table 1 of Bell and Lee (2012): Hurricane Danny, KLIX, 1 km, 1802 UTC
TABLE1_RADII = np.arange(10.0, 31.0) * 1000.0
TABLE1_VT = np.array(
    [23.6, 22.3, 27.1, 29.4, 29.8, 32.3, 33.6, 33.2, 31.7, 31.6, 31.4]
    + [31.1, 30.9, 31.2, 31.0, 29.6, 29.2, 28.5, 28.0, 27.6, 27.2]
)
TABLE1_STD = np.array(
    [0.66, 4.90, 1.94, 1.36, 0.86, 0.44, 0.13, 0.74, 0.61, 0.28, 0.36]
    + [0.35, 2.79, 1.30, 1.36, 1.00, 3.19, 0.73, 0.84, 1.16, 2.53]
)
TABLE1_N = np.array(
    [10, 11, 11, 12, 10, 13, 12, 12, 13, 13, 13, 10, 9, 13, 12, 11, 8, 12, 14, 13, 12]
)


def test_norm_score_reproduces_table_1_of_bell_and_lee():
    qualified = tw._qualified(TABLE1_VT, TABLE1_RADII, 1000.0)
    # local maxima at 16 and 23 km: the radii within 1 km of them are kept
    assert list(TABLE1_RADII[qualified] / 1000.0) == [15, 16, 17, 22, 23, 24]
    score = tw._norm_scores(
        TABLE1_VT, TABLE1_STD * 1000.0, TABLE1_N.astype(float), 16, qualified
    )
    # the scores of the best peak (15 to 17 km) are those of the table; the
    # second peak (22 to 24 km) comes out 0.03 to 0.06 below the printed scores
    # (the printed parts do not add up to them with the weights 1, 3, 1), but
    # the ranking is the same
    expected = {15: 0.39, 16: 0.94, 17: 0.37}
    for km, value in expected.items():
        assert abs(score[int(km - 10)] - value) < 0.015, km
    second = {22: 0.16, 23: 0.30, 24: 0.27}
    for km, value in second.items():
        assert 0.0 < value - score[int(km - 10)] < 0.07, km
    assert score[12] < score[13] and score[14] < score[13]
    assert score[13] < score[5] < score[6]
    assert np.isnan(score[~qualified]).all()
    assert np.nanargmax(score) == 6  # 16 km
    # the normalised parts of the table
    parts = np.exp((TABLE1_VT[[5, 7, 12]] - 33.6) / 1.0)
    assert np.allclose(parts, [0.27, 0.64, 0.06], atol=0.04)


def test_qualified_radii_without_a_local_maximum_and_with_gaps():
    radii = np.arange(1.0, 7.0) * 1000.0
    rising = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan])
    assert list(tw._qualified(rising, radii, 1000.0)) == [True] * 5 + [False]
    peaked = np.array([1.0, 2.0, 5.0, 2.0, 1.0, 0.0])
    assert (
        list(tw._qualified(peaked, radii, 1000.0))
        == [False, True, True, True] + [False] * 2
    )
    assert list(tw._qualified(peaked, radii, 0.0)) == [False, False, True] + [False] * 3
    none = np.full(6, np.nan)
    assert not tw._qualified(none, radii, 1000.0).any()
    assert np.isnan(
        tw._norm_scores(none, none, np.zeros(6), 16, np.zeros(6, bool))
    ).all()


def test_consensus_drops_outliers_and_unconverged_searches():
    res = np.zeros((6, 2, 5))
    res[:, :, 4] = 1.0
    res[:, 0, 0] = [10.0, 10.0, 10.0, 10.0, 10.0, 60.0]  # one far outlier
    res[:, 0, 1] = 5.0
    res[:, 0, 2] = [30.0, 30.0, 30.0, 30.0, 30.0, 20.0]
    res[:, 1, 4] = 0.0  # nothing converged at the second radius
    out = tw._consensus(res, np.array([1000.0, 2000.0]), 6)
    assert np.allclose(out[:3, 0], [30.0, 10.0, 5.0])
    assert out[4, 0] == 5 and out[3, 0] == 0.0
    assert np.isnan(out[:4, 1]).all() and out[4, 1] == 0.0
    # identical centres: all are kept
    res[:, 0, 0] = 10.0
    assert tw._consensus(res, np.array([1000.0, 2000.0]), 6)[4, 0] == 6


@pytest.mark.parametrize("engine", ENGINES)
def test_centre_of_the_analytic_vortex_is_found(engine):
    c = (30000.0, 70000.0)
    ds = vortex(
        center=c,
        radial=(0.1, 3.0),
        asymmetry={1: (0.15, 45.0)},
        noise=2.0,
        seed=1,
        azimuth_step=2.0,
    )
    radii = np.array([17000.0, 18000.0, 19000.0, 20000.0, 21000.0, 22000.0, 23000.0])
    res = tc_center(ds, radii=radii, guess_grid=2, engine=engine)
    err = np.hypot(float(res["CENTER_X"]) - c[0], float(res["CENTER_Y"]) - c[1])
    assert err < 700.0
    assert abs(float(res["RMW"]) - RMW) <= 2000.0
    assert abs(float(res["VT_MAX"]) - VMAX) < 3.0
    assert float(res["CENTER_UNCERTAINTY"]) < 1500.0
    assert res["SELECTED"].sum() == 1
    assert float(res["FIRST_GUESS_RMW"]) > 0
    # the geographic position is the offset on the sphere
    lat, lon = tw._offset_to_latlon(
        float(res["CENTER_X"]), float(res["CENTER_Y"]), 20.0, 80.0
    )
    assert np.isclose(float(res["CENTER_LATITUDE"]), lat)
    assert np.isclose(float(res["CENTER_LONGITUDE"]), lon)


@compiled_only
def test_tc_center_agrees_between_engines():
    ds = vortex(
        center=(-20000.0, 75000.0),
        asymmetry={1: (0.1, 200.0)},
        mean_wind=(3.0, -2.0),
        noise=1.5,
        seed=7,
        azimuth_step=2.0,
    )
    kwargs = {"radii": np.arange(16000.0, 25001.0, 1500.0), "guess_grid": 2}
    a = tc_center(ds, engine="compiled", **kwargs)
    b = tc_center(ds, engine="numpy", **kwargs)
    for name in ("CENTER_X", "CENTER_Y", "RMW", "VT_MAX"):
        assert np.isclose(float(a[name]), float(b[name]), rtol=1e-6, atol=1e-3), name
    assert np.allclose(a["VT0"], b["VT0"], rtol=1e-6, atol=1e-6, equal_nan=True)
    assert np.array_equal(a["N_CONVERGING"], b["N_CONVERGING"])


def test_centre_finding_from_the_default_first_guess_and_a_position():
    c = (30000.0, 70000.0)
    ds = vortex(
        center=c, asymmetry={1: (0.15, 45.0)}, noise=2.0, seed=3, azimuth_step=2.0
    )
    res = tc_center(
        ds, min_radius=14000.0, max_radius=26000.0, radius_step=2000.0, engine="auto"
    )
    assert (
        np.hypot(float(res["CENTER_X"]) - c[0], float(res["CENTER_Y"]) - c[1]) < 1000.0
    )
    # the dipole first guess is within a quarter of the vortex size
    assert (
        np.hypot(float(res["FIRST_GUESS_X"]) - c[0], float(res["FIRST_GUESS_Y"]) - c[1])
        < 7000.0
    )
    # a first guess as latitude and longitude, and with the mean wind removed
    lat, lon = tw._offset_to_latlon(c[0] + 3000.0, c[1] - 2000.0, 20.0, 80.0)
    res2 = tc_center(
        ds,
        first_guess_latlon=(float(lat), float(lon)),
        radii=np.arange(15000.0, 26000.0, 2000.0),
        mean_wind=(0.0, 0.0),
        guess_spacing=3000.0,
        search_radius=12000.0,
    )
    assert np.isclose(float(res2["FIRST_GUESS_X"]), c[0] + 3000.0, atol=1.0)
    assert np.isclose(float(res2["FIRST_GUESS_Y"]), c[1] - 2000.0, atol=1.0)
    assert (
        np.hypot(float(res2["CENTER_X"]) - c[0], float(res2["CENTER_Y"]) - c[1])
        < 1000.0
    )
    res3 = tc_center(
        ds, first_guess=(c[0], c[1]), radii=np.array([19000.0, 20000.0, 21000.0])
    )
    assert float(res3["FIRST_GUESS_X"]) == c[0]


def test_centre_finding_without_data_is_nan():
    ds = vortex()
    empty = ds.assign(VRADH=ds["VRADH"] * np.nan)
    res = tc_center(empty, radii=np.array([19000.0, 20000.0]), guess_grid=2)
    assert np.isnan(res["CENTER_X"]) and np.isnan(res["RMW"])
    assert np.isnan(res["CENTER_LATITUDE"]) and res["SELECTED"].sum() == 0
    # a first guess without any data around it: no ring can be fitted
    res = tc_center(
        empty, first_guess=(0.0, 50000.0), radii=np.array([20000.0]), guess_grid=2
    )
    assert np.isnan(res["CENTER_X"])
    # too little data for a dipole
    few = ds.assign(VRADH=ds["VRADH"].where(ds["azimuth"] < 0.1))
    res = tc_center(few, radii=np.array([20000.0]))
    assert np.isnan(res["CENTER_X"]) and np.isnan(res["FIRST_GUESS_X"])
    assert (
        tw._dipole(tw._gate_arrays(np.zeros(5), np.ones(5), np.ones(5), np.ones(5)))
        is None
    )


def test_dipole_midpoint_of_a_vortex():
    ds = vortex(vmax=40.0, azimuth_step=0.5, range_step=250.0)
    gates = gate_arrays(ds)
    x, y, rough = tw._dipole(gates)
    # the extreme winds are on the two sides of the eye; the midpoint is the centre
    # to within a fraction of the radius of maximum wind
    assert np.hypot(x - CENTER[0], y - CENTER[1]) < 0.35 * RMW
    assert 0.5 * RMW < rough < 1.6 * RMW


# --------------------------------------------------------------------------
# decomposition
# --------------------------------------------------------------------------


def test_psi_azimuth_mapping_round_trip():
    theta_t = np.deg2rad(37.0)
    sin_a = np.array([0.1, 0.5, 0.9])[:, None]
    psi_true = np.deg2rad(np.arange(-179.0, 180.0, 13.0))[None, :]
    phi = theta_t + psi_true + np.arcsin(sin_a * np.sin(psi_true))
    psi = tw._psi_of_azimuth(phi, theta_t, sin_a)
    assert np.allclose(psi, np.broadcast_to(psi_true, psi.shape), atol=1e-10)
    # psi is zero on the far side and 180 degrees on the near side of the ring
    assert np.allclose(
        tw._psi_of_azimuth(np.array([[theta_t]]), theta_t, sin_a[:1]), 0.0
    )


def test_decomposition_of_a_retrieval():
    spec = {1: (0.2, 45.0), 2: (0.1, 100.0)}
    ds = vortex(asymmetry=spec, radial=(0.1, 3.0))
    radii = np.arange(5000.0, 50001.0, 2500.0)
    g = gbvtd(ds, CENTER, radii=radii)
    d = tc_decompose(g)
    az = d["azimuth"].values
    assert d.sizes == {"radius": radii.size, "azimuth": 72, "wavenumber": 3}
    assert np.allclose(
        d["VT"],
        d["VT_SYMMETRIC"]
        + d["VT_WAVENUMBER1"]
        + d["VT_WAVENUMBER2"]
        + d["VT_WAVENUMBER3"],
    )
    assert np.allclose(d["VT_SYMMETRIC"].values[:, 0], g["VT0"].values)
    # the winds, the position and the geometry of the polar grid
    speed = np.hypot(d["U"], d["V"])
    assert np.allclose(speed, np.hypot(d["VT"], d["VR"]))
    assert np.allclose(
        np.hypot(d["X"] - CENTER[0], d["Y"] - CENTER[1]).values[:, 3], radii
    )
    # at azimuth 90 (east of the centre) the position is east of it and a cyclonic
    # wind blows north
    j = int(np.where(az == 90.0)[0][0])
    assert np.allclose(d["X"].values[:, j] - CENTER[0], radii)
    assert (d["V"].values[:, j] * d["VT"].values[:, j] > 0).all()
    # the asymmetric part is the truth at the physical azimuth to within the
    # distortion of GBVTD: compare the total tangential wind with the forward model
    vt0, _ = truth(radii)
    inside = np.abs(radii - RMW) > 3000.0
    gamma = az
    expected = vt0[:, None] * (
        1.0
        + 0.2 * np.cos(np.deg2rad(gamma - 45.0))
        + 0.1 * np.cos(2.0 * np.deg2rad(gamma - 100.0))
    )
    assert rms((d["VT"].values - expected)[inside]) < 0.07 * vt0[inside].mean()
    # the asymmetry fraction and the radius of maximum wind per azimuth
    assert 0.05 < float(d["ASYMMETRY"].values[inside].mean()) < 0.25
    assert abs(float(np.nanmedian(d["RMW_AZIMUTH"])) - RMW) < 2000.0
    assert np.isclose(float(d["RMW"]), float(g["RMW"]))
    assert np.isclose(float(d["CENTER_Y"]), CENTER[1])
    # the azimuths can be given
    d2 = tc_decompose(g, azimuth=[0.0, 90.0, 180.0])
    assert d2.sizes["azimuth"] == 3
    assert np.allclose(d2["VT"].values[:, 1], d["VT"].values[:, j])


def test_decomposition_errors():
    g = gbvtd(vortex(), CENTER, radii=RADII)
    with pytest.raises(ValueError, match="azimuth"):
        tc_decompose(g, azimuth=[])
    with pytest.raises(ValueError, match="not a gbvtd result"):
        tc_decompose(g[["VT0"]])
    with pytest.raises(TypeError):
        tc_decompose(g["VT0"])
    plain = g.drop_vars("RMW")
    assert np.isnan(float(tc_decompose(plain)["RMW"]))


# --------------------------------------------------------------------------
# input types, volumes and lazy evaluation
# --------------------------------------------------------------------------


def volume(ds):
    root = xr.Dataset(coords={"latitude": 20.0, "longitude": 80.0, "altitude": 50.0})
    sweep = ds.drop_vars(["latitude", "longitude", "altitude"])
    return xr.DataTree.from_dict(
        {"/": root, "/sweep_0": sweep, "/sweep_1": sweep.copy(deep=True)}
    )


def test_dataarray_dataset_and_volume_inputs():
    ds = vortex(asymmetry={1: (0.2, 45.0)}, azimuth_step=2.0)
    radii = np.array([15000.0, 20000.0, 30000.0])
    a = gbvtd(ds, CENTER, radii=radii)
    b = gbvtd(ds["VRADH"], CENTER, radii=radii)
    assert np.allclose(a["VT0"], b["VT0"])
    tree = volume(ds)
    c = gbvtd(tree, CENTER, radii=radii)
    assert isinstance(c, xr.DataTree)
    assert np.allclose(c["sweep_1"]["VT0"], a["VT0"])
    # one sweep of a volume is processed as a sweep
    one = gbvtd(tree["sweep_0"], CENTER, radii=radii)
    assert np.allclose(one["VT0"], a["VT0"])
    # a centre per sweep: the result of tc_center on the volume
    found = tc_center(tree, radii=np.array([19000.0, 20000.0, 21000.0]), guess_grid=2)
    assert set(found.children) == {"sweep_0", "sweep_1"}
    again = gbvtd(tree, found, radii=radii)
    assert np.isfinite(again["sweep_0"]["VT0"].values).all()
    assert "CENTER_LATITUDE" in found["sweep_0"].ds
    dec = tc_decompose(again)
    assert set(dec.children) == {"sweep_0", "sweep_1"}
    # a dictionary of centres, and a DataTree without sweeps
    d = gbvtd(tree, {"sweep_0": CENTER, "sweep_1": (0.0, 79000.0)}, radii=radii)
    assert not np.allclose(d["sweep_0"]["VT0"], d["sweep_1"]["VT0"])
    with pytest.raises(ValueError, match="no sweeps"):
        tc_decompose(xr.DataTree.from_dict({"/": xr.Dataset()}))
    with pytest.raises(KeyError):
        gbvtd(xr.DataTree.from_dict({"/a": xr.Dataset()}), CENTER, radii=radii)


def test_rays_numbered_by_time_and_fixed_angle_elevation():
    ds = vortex(azimuth_step=2.0)
    n = ds.sizes["azimuth"]
    timed = (
        ds.rename({"azimuth": "time"})
        .assign_coords(azimuth=("time", ds["azimuth"].values))
        .drop_vars("elevation")
    )
    timed = timed.assign_coords(sweep_fixed_angle=0.5)
    radii = np.array([15000.0, 20000.0, 30000.0])
    a = gbvtd(ds, CENTER, radii=radii)
    b = gbvtd(timed, CENTER, radii=radii)
    assert np.allclose(a["VT0"], b["VT0"]) and n == timed.sizes["time"]
    # an elevation per ray without a coordinate is an error, and so is a wrong shape
    with pytest.raises(ValueError, match="elevation"):
        gbvtd(timed.drop_vars("sweep_fixed_angle"), CENTER, radii=radii)
    bad = timed.drop_vars("sweep_fixed_angle").assign_coords(
        elevation=("range", np.full(timed.sizes["range"], 0.5))
    )
    with pytest.raises(ValueError, match="one value per ray"):
        gbvtd(bad, CENTER, radii=radii)


def test_dask_input_is_lazy_and_equal():
    dask = pytest.importorskip("dask")  # noqa: F841
    ds = vortex(asymmetry={1: (0.2, 45.0)}, azimuth_step=2.0)
    lazy = ds.chunk({"azimuth": 30})
    radii = np.array([15000.0, 20000.0, 30000.0])
    g = gbvtd(lazy, CENTER, radii=radii)
    assert g["VT0"].chunks is not None and g["RMW"].chunks is not None
    assert g["VT1_AZIMUTH"].chunks is not None
    e = gbvtd(ds, CENTER, radii=radii)
    assert np.allclose(g["VT0"], e["VT0"])
    found = tc_center(lazy, radii=np.array([19000.0, 20000.0, 21000.0]), guess_grid=2)
    assert found["CENTER_X"].chunks is not None and found["VT0"].chunks is not None
    ref = tc_center(ds, radii=np.array([19000.0, 20000.0, 21000.0]), guess_grid=2)
    assert np.isclose(float(found["CENTER_X"]), float(ref["CENTER_X"]))
    # a lazy centre is computed when it is used as the centre of a retrieval
    g2 = gbvtd(lazy, found, radii=radii)
    assert np.allclose(g2["VT0"], gbvtd(ds, ref, radii=radii)["VT0"])


def test_inputs_are_not_changed():
    ds = vortex(asymmetry={1: (0.2, 45.0)}, azimuth_step=2.0)
    before = ds.copy(deep=True)
    attrs = {k: dict(v.attrs) for k, v in ds.coords.items()}
    radii = np.array([15000.0, 20000.0, 30000.0])
    found = tc_center(ds, radii=np.array([19000.0, 20000.0]), guess_grid=2)
    g = gbvtd(ds, found, radii=radii)
    tc_decompose(g)
    gbvtd(ds["VRADH"], CENTER, radii=radii)
    xr.testing.assert_identical(ds, before)
    assert {k: dict(v.attrs) for k, v in ds.coords.items()} == attrs


def test_accessors():
    ds = vortex(azimuth_step=2.0)
    radii = np.array([15000.0, 20000.0])
    a = ds.radarx.gbvtd(CENTER, radii=radii)
    assert np.allclose(a["VT0"], gbvtd(ds, CENTER, radii=radii)["VT0"])
    b = ds["VRADH"].radarx.tc_center(radii=radii, guess_grid=2)
    assert "CENTER_X" in b
    c = volume(ds).radarx.gbvtd(CENTER, radii=radii)
    assert isinstance(c, xr.DataTree)


# --------------------------------------------------------------------------
# options, errors, provenance
# --------------------------------------------------------------------------


def test_geographic_helpers_round_trip():
    lat, lon = tw._offset_to_latlon(30000.0, -45000.0, 15.5, 73.8)
    x, y = tw._latlon_to_offset(float(lat), float(lon), 15.5, 73.8)
    assert np.allclose([x, y], [30000.0, -45000.0], atol=1e-3)
    # 111.2 km per degree of latitude to the north
    lat, lon = tw._offset_to_latlon(0.0, 111194.9, 0.0, 10.0)
    assert np.isclose(float(lat), 1.0, atol=1e-3) and np.isclose(float(lon), 10.0)


def test_centre_arguments_and_errors():
    ds = vortex(azimuth_step=2.0)
    radii = np.array([15000.0, 20000.0])
    lat, lon = tw._offset_to_latlon(*CENTER, 20.0, 80.0)
    a = gbvtd(ds, center_latlon=(float(lat), float(lon)), radii=radii)
    b = gbvtd(ds, CENTER, radii=radii)
    assert np.allclose(a["VT0"], b["VT0"], atol=0.05)
    assert np.isclose(float(a["CENTER_LATITUDE"]), float(lat))
    with pytest.raises(ValueError, match="not both"):
        gbvtd(ds, CENTER, center_latlon=(1.0, 2.0), radii=radii)
    with pytest.raises(ValueError, match="give the centre"):
        gbvtd(ds, radii=radii)
    with pytest.raises(ValueError, match="centre must be|center must be"):
        gbvtd(ds, "north", radii=radii)
    with pytest.raises(ValueError, match="not finite"):
        gbvtd(ds, (np.nan, 1.0), radii=radii)
    with pytest.raises(ValueError, match="at the radar"):
        gbvtd(ds, (0.0, 0.0), radii=radii)
    with pytest.raises(ValueError, match="CENTER_X"):
        gbvtd(ds, xr.Dataset({"A": 1.0}), radii=radii)
    with pytest.raises(ValueError, match="CENTER_X"):
        gbvtd(ds, volume(ds), radii=radii)
    nosite = ds.drop_vars(["latitude", "longitude"])
    with pytest.raises(ValueError, match="latitude and longitude"):
        gbvtd(nosite, center_latlon=(1.0, 2.0), radii=radii)
    with pytest.raises(ValueError, match="latitude and longitude"):
        tc_center(nosite, first_guess_latlon=(1.0, 2.0), radii=radii)
    with pytest.raises(ValueError, match="not both"):
        tc_center(ds, first_guess=(0.0, 1.0), first_guess_latlon=(1.0, 2.0))
    out = gbvtd(nosite, CENTER, radii=radii)
    assert "CENTER_LATITUDE" not in out
    out = tc_center(nosite, radii=radii, guess_grid=2)
    assert "CENTER_LATITUDE" not in out
    # without the altitude the radar is at sea level
    out = gbvtd(ds.drop_vars("altitude"), CENTER, radii=radii)
    assert np.isfinite(out["VT0"]).all()


@pytest.mark.parametrize(
    "func, kwargs, match",
    [
        (gbvtd, {"radii": []}, "radii"),
        (gbvtd, {"radii": [-1.0]}, "radii"),
        (gbvtd, {"min_radius": 0.0}, "min_radius"),
        (gbvtd, {"radius_step": -1.0}, "radius_step"),
        (gbvtd, {"ring_width": 0.0}, "ring_width"),
        (gbvtd, {"max_wavenumber": 5}, "max_wavenumber"),
        (gbvtd, {"max_wavenumber": 1.5}, "max_wavenumber"),
        (gbvtd, {"min_points": 1}, "min_points"),
        (gbvtd, {"max_gap": 0.0}, "max_gap"),
        (gbvtd, {"mean_wind": (1.0,)}, "mean_wind"),
        (gbvtd, {"mean_wind": (1.0, np.nan)}, "finite"),
        (gbvtd, {"k_e": 0.0}, "k_e"),
        (gbvtd, {"engine": "gpu"}, "engine"),
        (tc_center, {"guess_grid": 0}, "guess_grid"),
        (tc_center, {"guess_spacing": -1.0}, "guess_spacing"),
        (tc_center, {"search_radius": 0.0}, "search_radius"),
        (tc_center, {"inflection_window": -1.0}, "inflection_window"),
        (tc_center, {"selection": "best"}, "selection"),
        (tc_center, {"simplex_radius": 0.0}, "simplex_radius"),
        (tc_center, {"k_e": -1.0}, "k_e"),
        (tc_center, {"engine": "gpu"}, "engine"),
    ],
)
def test_option_errors(func, kwargs, match):
    ds = vortex(azimuth_step=4.0, max_range=100000.0)
    args = (ds, CENTER) if func is gbvtd else (ds,)
    with pytest.raises(ValueError, match=match):
        func(*args, **kwargs)


def test_object_and_geometry_errors():
    ds = vortex(azimuth_step=4.0, max_range=100000.0)
    with pytest.raises(TypeError):
        gbvtd(np.zeros((3, 3)), CENTER)
    with pytest.raises(TypeError):
        tc_center(np.zeros((3, 3)))
    with pytest.raises(KeyError):
        tc_center(ds.rename({"VRADH": "ZZ"}))
    with pytest.raises(ValueError, match="two dimensions"):
        tc_center(ds["VRADH"].expand_dims(time=2))
    with pytest.raises(ValueError, match="range"):
        tc_center(ds["VRADH"].isel(azimuth=0))
    noaz = ds.drop_vars("azimuth").assign_coords(
        azimuth=("range", np.zeros(ds.sizes["range"]))
    )
    with pytest.raises(ValueError, match="one value per ray"):
        tc_center(noaz)
    nothing = xr.Dataset(
        {"VRADH": (("time", "range"), np.zeros((4, 5)))},
        coords={"range": np.arange(1.0, 6.0), "elevation": 0.5},
    )
    with pytest.raises(ValueError, match="azimuth"):
        tc_center(nothing)
    with pytest.raises(ValueError, match="min_radius"):
        gbvtd(ds, CENTER, min_radius=5000.0, max_radius=1000.0)


def test_engine_compiled_without_kernel(monkeypatch):
    monkeypatch.setattr(tw, "HAS_COMPILED_KERNEL", False)
    ds = vortex(azimuth_step=4.0, max_range=100000.0)
    with pytest.raises(ImportError, match="compiled"):
        gbvtd(ds, CENTER, engine="compiled")
    out = gbvtd(ds, CENTER, radii=np.array([15000.0, 20000.0]))
    assert np.isfinite(out["VT0"]).all()


def test_provenance_and_references():
    ds = vortex(azimuth_step=2.0)
    radii = np.array([15000.0, 20000.0, 25000.0])
    results = {
        "synthetic_vortex": ds,
        "tc_center": tc_center(ds, radii=radii, guess_grid=2),
        "gbvtd": gbvtd(ds, CENTER, radii=radii),
        "tc_decompose": tc_decompose(gbvtd(ds, CENTER, radii=radii)),
    }
    registry = load_registry()
    for name, res in results.items():
        assert res.attrs[METHOD] and res.attrs[HISTORY].startswith(name), name
        refs = res.attrs[REFERENCES].split()
        assert refs and all(r in registry for r in refs), name
    assert (
        "10.1175/1520-0493(1999)127<2419:TCKSRF>2.0.CO;2"
        in results["gbvtd"].attrs[REFERENCES]
    )
    assert "10.1175/JAMC-D-11-0167.1" in results["tc_center"].attrs[REFERENCES]
    assert "gbvtd(" in results["gbvtd"].attrs[HISTORY]


# --------------------------------------------------------------------------
# the equations and the numbers of Lee et al. (1999) and Lee and Marks (2000)
# --------------------------------------------------------------------------


def lee_1999_coefficients(vtc, vts, vr0, along, cross, sin_a):
    """``A0, A1, B1, A2, B2, A3, B3, A4, B4`` of Eqs. 13 to 18 (no asymmetric V_R)."""
    ca = np.sqrt(1.0 - sin_a**2)
    c = lambda n: vtc.get(n, 0.0)  # noqa: E731
    s = lambda n: vts.get(n, 0.0)  # noqa: E731
    return np.array(
        [
            0.5 * (along * (1.0 + ca) - s(1)),  # Eq. 13
            0.5 * (2.0 * vr0 - s(2)),  # Eq. 14
            0.5 * (c(2) - 2.0 * c(0) - 2.0 * cross * sin_a),  # Eq. 15
            0.5 * (along * (1.0 - ca) - s(3) + s(1)),  # Eq. 16
            0.5 * (c(3) - c(1)),  # Eq. 18, n = 2
            0.5 * (s(2) - s(4)),  # Eq. 17, n = 3
            0.5 * (c(4) - c(2)),
            0.5 * (s(3) - s(5)),
            0.5 * (c(5) - c(3)),
        ]
    )


def test_closure_inverts_the_fourier_relations_of_lee_et_al_1999():
    vtc = {0: 40.0, 1: 3.0, 2: -2.0, 3: 1.5}
    vts = {1: 4.0, 2: 2.5, 3: -1.2}
    vr0, along, cross, sin_a = -3.0, 5.0, -4.0, 0.4
    coef = lee_1999_coefficients(vtc, vts, vr0, along, cross, sin_a)
    rt = 80000.0
    diag = np.array([100.0, 10.0, 4.0, 0.1])
    # the mean wind (u, v) = (cross, 0) of a centre north of the radar has the
    # cross-beam component `cross` and no component along the beam
    rows = dict(
        zip(
            tw._GBVTD_ROWS,
            tw._derived(coef, diag, 0.0, rt, np.array(sin_a * rt), (cross, 0.0)),
        )
    )
    assert np.isclose(rows["VT0"], vtc[0])
    assert np.isclose(rows["VR0"], vr0)
    assert np.isclose(rows["VM_ALONG"], along)  # Eq. 19
    assert np.isclose(rows["VT1_COS"], vtc[1])  # Eq. 23
    assert np.isclose(rows["VT1_SIN"], vts[1])  # Eq. 22
    assert np.isclose(rows["VT2_COS"], vtc[2])  # Eq. 25
    assert np.isclose(rows["VT2_SIN"], vts[2])  # Eq. 24
    assert np.isclose(rows["VT3_COS"], vtc[3])  # Eq. 27
    assert np.isclose(rows["VT3_SIN"], vts[3])  # Eq. 26
    # without the mean wind the cross-beam wind is aliased into V_T0 (Eq. 20);
    # the sign of B3 in Eq. 20 is minus: it cancels the V_TC2 of B1 (Eqs. 15, 18)
    plain = tw._derived(coef, diag, 0.0, rt, np.array(sin_a * rt), (0.0, 0.0))
    assert np.isclose(plain[0], vtc[0] + cross * sin_a)
    # a fit of a lower order leaves the higher waves out, and a ring that is too
    # close to the radar has none
    low = tw._derived(coef, np.array([100.0, 10.0, 2.0, 0.1]), 0.0, rt, 1000.0, (0, 0))
    assert np.isfinite(low[[0, 1, 2, 3, 8]]).all() and np.isnan(low[4:8]).all()
    near = tw._derived(coef, diag, 0.0, rt, np.array(rt), (0.0, 0.0))
    assert np.isnan(near[:9]).all()
    # the geometric orders: the wavenumbers 1, 2 and 3 up to sin(alpha) 0.7, 0.5, 0.33
    orders = [tw._geometric_order(v) for v in (0.2, 0.33, 0.4, 0.5, 0.6, 0.7, 0.8)]
    assert orders == [4, 4, 3, 3, 2, 2, 1]


def paper_sweep(vmax=50.0, **kwargs):
    """The test vortex of Lee et al. (1999): 50 m/s at 20 km, 60 km from the radar."""
    options = {
        "center": (0.0, 60000.0),
        "vmax": vmax,
        "rmw": 20000.0,
        "elevation": 0.0,
        "azimuth_step": 1.0,
        "range_step": 500.0,
        "max_range": 110000.0,
    }
    options.update(kwargs)
    return synthetic_vortex(**options)


# Table 1 of Lee and Marks (2000): the centre (km) of the analysis, the amplitude
# (m/s) of the wavenumbers 1 and 2 and the mean flow retrieved from the
# axisymmetric vortex, and the radius (km) that contains the highest wavenumber 0
TABLE1_DISPLACED = {
    (1, 60): (2.1, 0.3, 2.4, 20),
    (3, 60): (10.0, 1.1, 6.2, 22),
    (5, 60): (14.6, 2.7, 8.5, 24),
    (0, 62): (6.1, 1.9, 0.0, 21),
    (0, 65): (10.1, 4.8, 0.0, 21),
}


@pytest.mark.parametrize("engine", ENGINES)
def test_displaced_centre_of_table_1_of_lee_and_marks_2000(engine):
    ds = paper_sweep()
    for (x, y), (w1, w2, flow, rmw) in TABLE1_DISPLACED.items():
        g = gbvtd(
            ds,
            (x * 1000.0, y * 1000.0),
            radii=np.array([rmw * 1000.0]),
            ring_width=1000.0,
            engine=engine,
        )
        # the apparent wavenumber 1 and 2 of the series in psi, and the apparent
        # mean flow (zero for a centre displaced along the beam)
        assert abs(float(g["VT1_AMPLITUDE"].values[0]) - w1) < 0.6, (x, y)
        assert abs(float(g["VT2_AMPLITUDE"].values[0]) - w2) < 0.8, (x, y)
        assert abs(float(g["VM_ALONG"].values[0]) - flow) < 0.25, (x, y)


def test_displaced_centre_scales_with_the_intensity():
    # Table 2 of Lee and Marks (2000): the apparent wavenumber 1 is proportional to
    # the maximum wind (1.3, 1.7, 2.1 and 2.6 m/s at 1 km for 30 to 60 m/s)
    amp = {}
    for vmax in (30.0, 60.0):
        g = gbvtd(
            paper_sweep(vmax),
            (3000.0, 60000.0),
            radii=np.array([22000.0]),
            ring_width=1000.0,
        )
        amp[vmax] = float(g["VT1_AMPLITUDE"].values[0])
    assert np.isclose(amp[60.0] / amp[30.0], 2.0, rtol=0.02)
    assert abs(amp[30.0] - 5.9) < 0.6 and abs(amp[60.0] - 11.7) < 1.0


@pytest.mark.parametrize(
    "wavenumber, sin_below, sin_above",
    [(1, 0.6, 0.8), (2, 0.4, 0.55), (3, 0.3, 0.4)],
)
def test_wave_limits_of_lee_et_al_1999(wavenumber, sin_below, sin_above):
    # Section 7: for a 20 percent bias the wavenumbers 1, 2 and 3 are retrieved up
    # to sin(alpha_max) of 0.7, 0.5 and 0.33; the series stops at the wavenumber
    # of the test asymmetry (10 m/s at every radius, the maximum towards the radar)
    ds = paper_sweep(absolute_asymmetry={wavenumber: (10.0, 90.0 - 180.0)})
    radii = np.array([sin_below, sin_above]) * 60000.0
    g = gbvtd(
        ds,
        (0.0, 60000.0),
        radii=radii,
        ring_width=1000.0,
        max_wavenumber=wavenumber + 1,
    )
    d = tc_decompose(g)
    bias = d["AZIMUTHAL_AMPLITUDE"].sel(wavenumber=wavenumber).values / 10.0 - 1.0
    assert abs(bias[0]) < 0.2 < abs(bias[1])


@pytest.mark.parametrize("engine", ENGINES)
def test_geometric_limit_of_the_fourier_order(engine):
    # a centre 80 km from the radar: sin(alpha_max) = 0.25, 0.4, 0.6 and 0.75
    ds = vortex(azimuth_step=1.0, range_step=250.0)
    radii = np.array([20000.0, 32000.0, 48000.0, 60000.0])
    free = gbvtd(ds, CENTER, radii=radii, engine=engine)
    limited = gbvtd(ds, CENTER, radii=radii, engine=engine, geometric_limit=True)
    assert list(free["FIT_ORDER"].values) == [4, 4, 4, 2]  # the far ring has gaps
    assert list(limited["FIT_ORDER"].values) == [4, 3, 2, 1]
    # a ring beyond sin(alpha_max) = 0.7 has the wavenumber 0 only
    assert np.isfinite(limited["VT0"]).all()
    assert np.isnan(limited["VT1_COS"].values[3]) and np.isnan(
        limited["VM_ALONG"].values[3]
    )
    assert np.isfinite(limited["VT1_COS"].values[:3]).all()
    assert np.isfinite(limited["VT2_COS"].values[:2]).all()
    assert np.isnan(limited["VT2_COS"].values[2:]).all()
    assert np.isfinite(limited["VT3_COS"].values[0])
    assert np.isnan(limited["VT3_COS"].values[1:]).all()
    assert limited.attrs["gbvtd_geometric_limit"] == 1


@compiled_only
def test_geometric_limit_agrees_between_kernel_and_numpy():
    ds = vortex(asymmetry={1: (0.2, 45.0)}, noise=1.0, azimuth_step=1.0)
    gates = gate_arrays(ds)
    cfg = dict(CFG, geometric_limit=True)
    radii = np.array([8000.0, 20000.0, 32000.0, 48000.0, 60000.0])
    a = tw._rings(gates, [0.0, 1000.0], [80000.0, 79000.0], radii, cfg, True, 2)
    b = tw._rings(gates, [0.0, 1000.0], [80000.0, 79000.0], radii, cfg, False, None)
    assert np.allclose(a[0], b[0], rtol=1e-8, atol=1e-8, equal_nan=True)
    assert np.array_equal(a[1][..., 2], b[1][..., 2])
    assert list(a[1][0, :, 2]) == [4, 4, 3, 2, 1]


def test_tc_center_with_the_geometric_limit_and_the_simplex_radius():
    c = (30000.0, 70000.0)
    ds = vortex(center=c, azimuth_step=2.0)
    kwargs = {"radii": np.array([19000.0, 20000.0, 21000.0]), "guess_grid": 2}
    base = tc_center(ds, **kwargs)
    other = tc_center(ds, geometric_limit=True, simplex_radius=2500.0, **kwargs)
    for res in (base, other):
        err = np.hypot(float(res["CENTER_X"]) - c[0], float(res["CENTER_Y"]) - c[1])
        assert err < 1000.0
    # the first simplex changes the path of the search, not the vortex it finds
    assert base.attrs["tc_center_geometric_limit"] == 0
    assert other.attrs["tc_center_geometric_limit"] == 1
    assert not np.array_equal(base["N_CONVERGING"], other["N_CONVERGING"])


def test_selection_of_the_ring_by_the_highest_wind_or_the_norm_score():
    c = (30000.0, 70000.0)
    ds = vortex(center=c, asymmetry={1: (0.15, 45.0)}, azimuth_step=2.0)
    radii = np.arange(14000.0, 27001.0, 1000.0)
    by_max = tc_center(ds, radii=radii, guess_grid=2, selection="max_vt")
    by_norm = tc_center(ds, radii=radii, guess_grid=2, selection="norm")
    ok = by_max["N_CONVERGING"].values > 0
    top = float(by_max["VT0"].values[ok].max())
    assert np.isclose(float(by_max["VT_MAX"]), top)
    assert by_max["SELECTED"].sum() == 1
    assert by_max["SELECTED"].values[np.nanargmax(by_max["VT0"].values)] == 1
    best = by_norm["NORM_SCORE"].values
    assert by_norm["SELECTED"].values[np.nanargmax(best)] == 1
    # the rings and their scores do not depend on the selection
    assert np.allclose(by_max["NORM_SCORE"], by_norm["NORM_SCORE"], equal_nan=True)
    assert abs(float(by_max["RMW"]) - RMW) <= 2000.0


def test_synthetic_absolute_asymmetry():
    r = vortex(absolute_asymmetry={1: (10.0, 0.0), 3: (4.0, 30.0)}, azimuth_step=2.0)
    plain = vortex(azimuth_step=2.0)
    sweep = tw._Sweep(r, None, 4.0 / 3.0)
    gamma = np.rad2deg(
        np.arctan2(sweep.x.values - CENTER[0], sweep.y.values - CENTER[1])
    )
    expected = 10.0 * np.cos(np.deg2rad(gamma)) + 4.0 * np.cos(
        3 * np.deg2rad(gamma - 30.0)
    )
    diff = r["VT_TRUE"].values - plain["VT_TRUE"].values
    assert np.allclose(diff, expected)  # the same at every radius, not scaled by VT0
    rel = vortex(asymmetry={1: (0.1, 0.0)}, azimuth_step=2.0)
    assert np.allclose(
        rel["VT_TRUE"].values - plain["VT_TRUE"].values,
        0.1 * plain["VT_TRUE"].values * np.cos(np.deg2rad(gamma)),
    )


def test_azimuthal_waves_of_a_known_wind():
    az = np.arange(0.0, 360.0, 5.0)
    phi = np.deg2rad(90.0 - az)  # mathematical angle
    # wavenumber 1 with the maximum east (azimuth 90), wavenumber 2 with maxima
    # at 20 and 200 degrees, wavenumber 3 absent
    vt = np.stack(
        [
            5.0
            + 3.0 * np.cos(np.deg2rad(az - 90.0))
            + 2.0 * np.cos(2 * np.deg2rad(az - 20.0)),
            7.0 + 0.0 * az,
        ]
    )
    amp, azi = tw._azimuthal_waves(vt, phi)
    assert np.allclose(amp[0], [3.0, 2.0, 0.0], atol=1e-10)
    assert np.allclose(azi[0, :2], [90.0, 20.0])
    assert np.allclose(amp[1], 0.0, atol=1e-10)


def test_first_simplex_of_figure_4_of_lee_and_marks_2000():
    # the initial guess is (-10, 50) km, the first simplex the equilateral triangle
    # of circumradius 4 km around it (one vertex to the north); the maximum
    # axisymmetric winds of the vertices A, B and C, of the reflection of A (R1)
    # and of the expansion (X1) are 21.35, 29.60, 34.15, 42.12 and 44.81 m/s (the
    # interpolation of the paper lowers the winds by up to 1 m/s)
    ds = paper_sweep()
    h = 0.5 * np.sqrt(3.0) * 4.0
    a, b, c = (-10.0 - h, 48.0), (-10.0, 54.0), (-10.0 + h, 48.0)
    mid = (0.5 * (b[0] + c[0]), 0.5 * (b[1] + c[1]))
    r1 = (2.0 * mid[0] - a[0], 2.0 * mid[1] - a[1])
    x1 = (3.0 * mid[0] - 2.0 * a[0], 3.0 * mid[1] - 2.0 * a[1])
    paper = {"A": 21.35, "B": 29.60, "C": 34.15, "R1": 42.12, "X1": 44.81}
    vertex = {"A": a, "B": b, "C": c, "R1": r1, "X1": x1}
    radii = np.arange(15000.0, 25001.0, 1000.0)
    for name, (x, y) in vertex.items():
        g = gbvtd(ds, (x * 1000.0, y * 1000.0), radii=radii, ring_width=2000.0)
        assert abs(float(g["VT0"].max()) - paper[name]) < 1.5, name
    # A is the worst vertex, so it is the one that is reflected and then extended
    assert paper["A"] < paper["B"] < paper["C"] < paper["R1"] < paper["X1"]


# Table 4 of Lee and Marks (2000): the analytic vortex with an asymmetry of 10 m/s
# (the mathematical angle of its maximum in degrees), the error (km) of the best
# centre and the axisymmetric wind (m/s) there
TABLE4 = {
    "axisymmetric": (None, 0.31, 48.9),
    "wavenumber 1 at 270": ({1: 270.0}, 0.65, 51.4),
    "wavenumber 2 at 90": ({2: 90.0}, 0.63, 45.1),
}


def table_4_centre(wave, **kwargs):
    absolute = None if wave is None else {n: (10.0, 90.0 - p) for n, p in wave.items()}
    ds = paper_sweep(absolute_asymmetry=absolute)
    return tc_center(
        ds,
        first_guess=(0.0, 60000.0),
        radii=np.array([19000.0, 20000.0, 21000.0]),
        ring_width=2000.0,
        max_wavenumber=2,
        search_radius=40000.0,
        selection="max_vt",
        **kwargs,
    )


@pytest.mark.parametrize("case", sorted(TABLE4))
def test_centre_finding_of_table_4_of_lee_and_marks_2000(case):
    wave, error, w0 = TABLE4[case]
    # 9 starting points 8 km apart (the paper has 36 points 4 km apart)
    res = table_4_centre(wave, guess_grid=3, guess_spacing=8000.0)
    err = np.hypot(float(res["CENTER_X"]), float(res["CENTER_Y"]) - 60000.0) / 1000.0
    # the centres are within the 1 km that Lee and Marks require and not farther
    # from the truth than theirs (the interpolation of the paper adds error), in
    # the same direction (towards the radar) and with the intensity of the paper
    assert err < 1.0 and err <= error
    assert float(res["CENTER_Y"]) < 60000.0 + 100.0
    assert abs(float(res["VT_MAX"]) - w0) < 1.0
    assert float(res["RMW"]) == 20000.0
    assert float(res["CENTER_UNCERTAINTY"]) < 200.0


@compiled_only
def test_centre_finding_with_the_36_starting_points_and_noise():
    # Table 5 of Lee and Marks (2000): a random error of 5 m/s changes the centres
    # little (mean error 0.35 km against 0.34 km without it)
    errors = []
    for noise in (0.0, 5.0):
        ds = paper_sweep(
            absolute_asymmetry={1: (10.0, 0.0)}, azimuth_step=1.0, noise=noise, seed=3
        )
        res = tc_center(
            ds,
            first_guess=(0.0, 60000.0),
            radii=np.arange(18000.0, 22001.0, 1000.0),
            ring_width=2000.0,
            max_wavenumber=2,
            guess_grid=6,
            guess_spacing=4000.0,
            search_radius=40000.0,
            selection="max_vt",
        )
        errors.append(
            np.hypot(float(res["CENTER_X"]), float(res["CENTER_Y"]) - 60000.0) / 1000.0
        )
    assert max(errors) < 0.7
    assert abs(errors[1] - errors[0]) < 0.5
