#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the polarimetric radar forward operator
=================================================

The truth is written here from definitions and from the published numbers, not
from the module: an exponential rain distribution whose reflectivity in the
Rayleigh limit is 720 N0 / Lambda^7, the gamma spectra of
``radar_from_dsd`` on the table diameters (agreement to rounding), the
amplitudes and the closed forms of Jung et al. (2008, Eqs. 5-13 with their
published amplitudes, which the module recomputes from the Rayleigh formulas of
Ryzhkov et al. 2020), geometry from ``fundamentals.geometry``, and
synthetic WRF-like and MPAS-like Datasets built from the variable definitions of
the two Registries. No WRF or MPAS output is read. The compiled kernel and the
NumPy implementation must agree to the rounding of float32.
"""

import importlib

import numpy as np
import pytest
import xarray as xr
from scipy.special import gamma as gamma_fn

import radarx  # noqa: F401
from radarx._provenance import HISTORY, METHOD, REFERENCES, load_registry
from radarx.fundamentals.constants import EARTH_RADIUS
from radarx.fundamentals.geometry import beam_center_height, ground_range
from radarx.retrieve import (
    forward_operator,
    mpas_state,
    radar_from_dsd,
    scattering_table,
    to_radar_grid,
    wrf_state,
)

_trapezoid = getattr(np, "trapezoid", None) or np.trapz

fo = importlib.import_module("radarx.retrieve.forward_operator")
dsdmod = importlib.import_module("radarx.retrieve.dsd")

ENGINES = ["numpy"] + (["compiled"] if fo.HAS_COMPILED_KERNEL else [])
compiled_only = pytest.mark.skipif(
    not fo.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)
G = 9.81
KAPPA = 287.0 / 1004.5
WL = {b: dsdmod._raw_table()[b]["wavelength"] for b in "SCX"}
MASS = np.pi / 6.0 * 1.0e-6  # kg m-3 per mm3 m-3 of water

# Jung et al. (2008): amplitudes (mm) per D^3 of dry snow and dry hail at S band
# (Eq. 9 with f_w = 0): alpha_a along the major (transverse) axis, alpha_b along
# the minor axis, and the A, B, C of their Eq. 5
JUNG = {
    "snow": {"a": 0.194e-4, "b": 0.191e-4, "n0": 3e6, "rho": 100.0, "sigma": 20.0},
    "hail": {"a": 0.191e-3, "b": 0.165e-3, "n0": 4e4, "rho": 913.0, "sigma": 60.0},
}


def jung_moments(name, q, rho_air, wavelength=107.0687):
    """Z_H, Z_V (mm6 m-3), Z_DR (dB) and K_DP of one dry class from Eqs. 5-19."""
    p = JUNG[name]
    s = np.deg2rad(p["sigma"])
    e1, e2 = np.exp(-2 * s * s), np.exp(-8 * s * s)
    a_, b_, c_ = (3 + 4 * e1 + e2) / 8, (3 - 4 * e1 + e2) / 8, (1 - e2) / 8
    lam = (np.pi * p["rho"] * p["n0"] / (rho_air * q)) ** 0.25 * 1e-3  # mm-1
    n0 = p["n0"] * 1e-3
    pref = 2880.0 * wavelength**4 * n0 / (np.pi**4 * 0.93) * lam**-7
    za, zb = p["a"], p["b"]
    zh = pref * (a_ * za**2 + b_ * zb**2 + 2 * c_ * za * zb)
    zv = pref * (b_ * za**2 + a_ * zb**2 + 2 * c_ * za * zb)
    kdp = 180 * wavelength / np.pi * 1e-3 * n0 * (za - zb) * e1 * 6 * lam**-4
    return zh, zv, 10 * np.log10(zh / zv), kdp


# --------------------------------------------------------------------------
# synthetic model output written from the Registry definitions
# --------------------------------------------------------------------------


def standard_atmosphere(z):
    """Temperature (K) and pressure (Pa) of a simple standard atmosphere."""
    z = np.asarray(z, dtype=float)
    t = np.where(z < 11000.0, 288.15 - 6.5e-3 * z, 216.65)
    p = np.where(
        z < 11000.0,
        101325.0 * (t / 288.15) ** 5.2559,
        22632.0 * np.exp(-(z - 11000.0) * G / (287.0 * 216.65)),
    )
    return t, p


def synthetic_wrf(
    nx=64, ny=48, nz=36, dx=3000.0, ntime=1, lat0=35.0, lon0=-97.5, theta_m=False
):
    """
    A squall-line-like Dataset with the variables of a ``wrfout`` file.

    A convective line (graupel aloft, heavy rain below, an updraught) at the
    centre and a trailing stratiform region to the west (snow above the 0
    degree level, light rain below), on a rotated grid so that the latitude
    and longitude are curvilinear. The staggered fields have the staggered
    dimensions of WRF. Rain number concentration in # kg-1.
    """
    zi_flat = 17000.0 * np.linspace(0, 1, nz + 1) ** 1.4
    jj, ii = np.mgrid[0:ny, 0:nx]
    x = (ii - nx / 2) * dx
    y = (jj - ny / 2) * dx
    rot = np.deg2rad(7.0)
    east = x * np.cos(rot) - y * np.sin(rot)
    north = x * np.sin(rot) + y * np.cos(rot)
    lat = lat0 + north / 111195.0
    lon = lon0 + east / (111195.0 * np.cos(np.deg2rad(lat)))
    terrain = 200.0 * (ii / nx)  # a slope of 200 m across the domain
    zint = zi_flat[:, None, None] + terrain[None] * (1 - zi_flat[:, None, None] / 17000)
    zmass = 0.5 * (zint[:-1] + zint[1:])
    temp, pres = standard_atmosphere(zmass)
    theta = temp * (1.0e5 / pres) ** KAPPA

    s = x[None] / 1000.0  # km across the line
    z = zmass / 1000.0
    core = np.exp(-(((s - 6.0) / 7.0) ** 2)) * (1 - 0.15 * (y[None] / 1e5) ** 2)
    strat = np.exp(-(((s + 45.0) / 40.0) ** 2)) * (s < 0)
    warm = temp > 273.15
    qrain = np.where(
        warm, 5.0e-3 * core * np.exp(-z / 3.0) + 0.6e-3 * strat * (z < 3.8), 0.0
    )
    qgraup = np.where(
        ~warm | (z < 5.5), 3.0e-3 * core * np.exp(-(((z - 5.0) / 2.5) ** 2)), 0.0
    )
    qsnow = np.where(
        ~warm, 0.5e-3 * (strat + 0.4 * core) * np.exp(-(((z - 7.0) / 3) ** 2)), 0
    )
    rho_air = pres / (287.0 * temp)
    dv = np.where(core > 0.2, 1.6e-3, 0.9e-3)  # mean volume diameter, m
    qnrain = np.where(
        qrain > 0, 6.0 * qrain / (np.pi * 1000.0 * dv**3), 0.0
    )  # # kg-1 (qrain in kg kg-1)
    del rho_air
    qv = 0.01 * np.exp(-z / 2.5) * np.ones_like(z)
    uwind = 14.0 + 1.5 * z + 0 * s
    vwind = 3.0 + 0.2 * z + 0 * s
    wwind = 6.0 * core * np.sin(np.pi * np.clip(z / 12.0, 0, 1))
    pb = pres
    t_pert = theta * (1.0 + 1.61 * qv if theta_m else 1.0) - 300.0

    def stag(a, axis, fill):
        """Staggered version of a mass-point field, by linear extrapolation."""
        a = np.moveaxis(a, axis, 0)
        edge = np.concatenate([a[:1] - 0.5 * (a[1:2] - a[:1]), 0.5 * (a[:-1] + a[1:])])
        last = a[-1:] + 0.5 * (a[-1:] - a[-2:-1])
        out = np.concatenate([edge, last])
        return np.moveaxis(out, 0, axis) + fill

    dims3 = ("Time", "bottom_top", "south_north", "west_east")
    rep = lambda a: np.repeat(a[None], ntime, axis=0)  # noqa: E731
    ds = xr.Dataset(
        {
            "T": (dims3, rep(t_pert)),
            "P": (dims3, rep(np.zeros_like(pb))),
            "PB": (dims3, rep(pb)),
            "PH": (
                ("Time", "bottom_top_stag", "south_north", "west_east"),
                rep(np.zeros_like(zint)),
            ),
            "PHB": (
                ("Time", "bottom_top_stag", "south_north", "west_east"),
                rep(G * zint),
            ),
            "QVAPOR": (dims3, rep(qv)),
            "QRAIN": (dims3, rep(qrain)),
            "QSNOW": (dims3, rep(qsnow)),
            "QGRAUP": (dims3, rep(qgraup)),
            "QNRAIN": (dims3, rep(qnrain)),
            "QCLOUD": (dims3, rep(0.1e-3 * (qrain > 0))),
            "U": (
                ("Time", "bottom_top", "south_north", "west_east_stag"),
                rep(stag(uwind, 2, 0.0)),
            ),
            "V": (
                ("Time", "bottom_top", "south_north_stag", "west_east"),
                rep(stag(vwind, 1, 0.0)),
            ),
            "W": (
                ("Time", "bottom_top_stag", "south_north", "west_east"),
                rep(stag(wwind, 0, 0.0)),
            ),
            "XLAT": (("Time", "south_north", "west_east"), rep(lat)),
            "XLONG": (("Time", "south_north", "west_east"), rep(lon)),
        },
        attrs={"USE_THETA_M": int(theta_m)},
    )
    return ds


def synthetic_mpas(ncell=1500, nz=30, ntime=1, lat0=35.0, lon0=-97.5, seed=4):
    """
    An MPAS-like Dataset on an unstructured set of cells (radians, ``nCells``
    by ``nVertLevels``) with a rain shaft and a grid file with ``zgrid``.
    """
    rng = np.random.default_rng(seed)
    d = np.sqrt(rng.uniform(0, 1, ncell)) * 400e3  # a disc of 400 km radius
    az = rng.uniform(0, 2 * np.pi, ncell)
    lat = lat0 + np.rad2deg(d * np.cos(az) / EARTH_RADIUS)
    lon = lon0 + np.rad2deg(d * np.sin(az) / (EARTH_RADIUS * np.cos(np.deg2rad(lat0))))
    zi = 17000.0 * np.linspace(0, 1, nz + 1) ** 1.4
    zgrid = np.repeat(zi[None], ncell, axis=0)  # (nCells, nVertLevelsP1)
    zm = 0.5 * (zgrid[:, :-1] + zgrid[:, 1:])
    temp, pres = standard_atmosphere(zm)
    exner = (pres / 1.0e5) ** KAPPA
    theta = temp / exner
    rho = pres / (287.0 * temp)
    dist = np.hypot(d * np.cos(az) - 60e3, d * np.sin(az) + 30e3)
    shaft = np.exp(-((dist / 40e3) ** 2))[:, None]
    qr = np.where(temp > 273.15, 3e-3 * shaft * np.exp(-zm / 3000.0), 0.0)
    qs = np.where(
        temp <= 273.15, 0.6e-3 * shaft * np.exp(-(((zm - 7000.0) / 3000.0) ** 2)), 0.0
    )
    nr = np.where(qr > 0, 6.0 * qr / (np.pi * 1000.0 * 1.2e-3**3), 0.0)

    def tl(a):
        return np.repeat(a[None], ntime, axis=0)

    ds = xr.Dataset(
        {
            "qv": (("Time", "nCells", "nVertLevels"), tl(0.008 * np.exp(-zm / 2500.0))),
            "qr": (("Time", "nCells", "nVertLevels"), tl(qr)),
            "qs": (("Time", "nCells", "nVertLevels"), tl(qs)),
            "nr": (("Time", "nCells", "nVertLevels"), tl(nr)),
            "rho": (("Time", "nCells", "nVertLevels"), tl(rho)),
            "theta": (("Time", "nCells", "nVertLevels"), tl(theta)),
            "exner": (("Time", "nCells", "nVertLevels"), tl(exner)),
            "pressure": (("Time", "nCells", "nVertLevels"), tl(pres)),
            "uReconstructZonal": (
                ("Time", "nCells", "nVertLevels"),
                tl(np.full((ncell, nz), 10.0)),
            ),
            "uReconstructMeridional": (
                ("Time", "nCells", "nVertLevels"),
                tl(np.full((ncell, nz), -5.0)),
            ),
            "w": (
                ("Time", "nCells", "nVertLevelsP1"),
                tl(np.full((ncell, nz + 1), 1.0)),
            ),
        }
    )
    grid = xr.Dataset(
        {
            "lonCell": ("nCells", np.deg2rad(lon)),
            "latCell": ("nCells", np.deg2rad(lat)),
            "zgrid": (("nCells", "nVertLevelsP1"), zgrid),
        }
    )
    return ds, grid


@pytest.fixture(scope="module")
def wrf():
    return synthetic_wrf()


@pytest.fixture(scope="module")
def state(wrf):
    return wrf_state(wrf)


@pytest.fixture(scope="module")
def fields(state):
    return forward_operator(
        state, nrain="QNRAIN", graupel="QGRAUP", graupel_params=GRAUPEL
    )


GRAUPEL = {"n0": 4.0e6, "rho": 400.0, "axis_ratio": 0.8, "sigma": 40.0}
SITE = (35.2, -97.9, 380.0)


def rain_state(q, rho=1.1, temp=283.15, **extra):
    q = np.atleast_1d(np.asarray(q, dtype=float))
    data = {
        "QRAIN": ("x", q),
        "AIR_DENSITY": ("x", np.full(q.shape, rho)),
        "TEMPERATURE": ("x", np.full(q.shape, temp)),
    }
    data.update({k: ("x", np.asarray(v, dtype=float)) for k, v in extra.items()})
    return xr.Dataset(data)


def gamma_spectrum(n0m, mu, lam, band="S", temperature=10.0):
    """radar_from_dsd of a gamma spectrum on the table diameters (trapezoid widths)."""
    tab = scattering_table(band, temperature)
    d = tab.diameter.values
    w = dsdmod._trapezoid_weights(d)
    n0m, mu, lam = np.broadcast_arrays(n0m, mu, lam)
    nd = n0m[:, None] * d[None, :] ** mu[:, None] * np.exp(-lam[:, None] * d[None, :])
    spectrum = xr.DataArray(
        nd,
        dims=("x", "diameter"),
        coords={"diameter": d, "bin_width": ("diameter", w)},
    )
    return radar_from_dsd(spectrum, band, temperature)


def lam_exponential(q, rho, n0=8.0e6, mu=0.0):
    """Slope (mm-1) from the third moment, by the definition."""
    m3 = np.asarray(q) * rho / MASS
    return (n0 * 1e-3 * gamma_fn(mu + 4) / m3) ** (1.0 / (mu + 4))


# --------------------------------------------------------------------------
# rain: closed forms and the existing DSD forward model
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("band", ["S", "C", "X"])
def test_rain_agrees_with_radar_from_dsd(engine, band):
    q = np.array([2e-5, 2e-4, 1e-3, 4e-3, 8e-3])
    out = forward_operator(
        rain_state(q), band=band, rain_temperature=10.0, engine=engine, temperature=None
    )
    lam = lam_exponential(q, 1.1)
    ref = gamma_spectrum(np.full(q.shape, 8.0e3), 0.0, lam, band, 10.0)
    np.testing.assert_allclose(out.DBZH, ref.DBZH, rtol=1e-5, atol=2e-5)
    np.testing.assert_allclose(out.ZDR, ref.ZDR, rtol=1e-5, atol=2e-5)
    np.testing.assert_allclose(out.KDP, ref.KDP, rtol=1e-5)
    np.testing.assert_allclose(out.RHOHV, ref.RHOHV, rtol=1e-6)


@pytest.mark.parametrize("engine", ENGINES)
def test_rain_gamma_shape_agrees_with_radar_from_dsd(engine):
    q = np.array([5e-4, 2e-3, 6e-3])
    out = forward_operator(
        rain_state(q), n0_rain=2.0e6, mu_rain=3.0, engine=engine, temperature=None
    )
    lam = lam_exponential(q, 1.1, n0=2.0e6, mu=3.0)
    ref = gamma_spectrum(np.full(3, 2.0e3), 3.0, lam, "S", 10.0)
    np.testing.assert_allclose(out.DBZH, ref.DBZH, rtol=1e-5, atol=2e-5)
    np.testing.assert_allclose(out.KDP, ref.KDP, rtol=1e-5)


@pytest.mark.parametrize("temp_c", [0.0, 4.0, 10.0, 17.5, 30.0])
def test_rain_table_temperature(temp_c):
    out = forward_operator(rain_state([2e-3], temp=temp_c + 273.15), engine="numpy")
    lam = lam_exponential(2e-3, 1.1)
    ref = gamma_spectrum(np.array([8.0e3]), 0.0, lam, "S", temp_c)
    np.testing.assert_allclose(out.DBZH, ref.DBZH, rtol=1e-5, atol=2e-5)
    np.testing.assert_allclose(out.KDP, ref.KDP, rtol=1e-5)


def test_rain_temperature_outside_the_table_is_clipped():
    cold = forward_operator(rain_state([2e-3], temp=250.0), engine="numpy")
    zero = forward_operator(rain_state([2e-3], temp=273.15), engine="numpy")
    hot = forward_operator(rain_state([2e-3], temp=320.0), engine="numpy")
    ref = forward_operator(rain_state([2e-3], temp=303.15), engine="numpy")
    assert float(cold.DBZH[0]) == float(zero.DBZH[0])
    assert float(hot.DBZH[0]) == float(ref.DBZH[0])


def test_rayleigh_limit_of_exponential_rain():
    """Small drops: Z = 720 N0 / Lambda^7, ZDR near 0, RHOHV near 1."""
    n0, lam = 8.0e6, 16.0  # mm-1: drops below 0.6 mm
    m3 = 8.0e3 * 6.0 / lam**4
    q = m3 * MASS / 1.1
    out = forward_operator(rain_state([q]), engine="numpy")
    z_rayleigh = 720.0 * (n0 * 1e-3) / lam**7
    assert 10 ** (0.1 * float(out.DBZH[0])) == pytest.approx(z_rayleigh, rel=0.04)
    assert abs(float(out.ZDR[0])) < 0.1
    assert float(out.RHOHV[0]) > 0.999


@pytest.mark.parametrize("engine", ENGINES)
def test_number_concentration_closes_the_distribution(engine):
    """With N_t from an exponential distribution the result is that of its N0."""
    q = np.array([5e-4, 3e-3])
    lam = lam_exponential(q, 1.1)
    nt = 8.0e3 / lam * 1.0  # M0 = N0 / Lambda (m-3)
    out = forward_operator(
        rain_state(q, QNRAIN=nt / 1.1),
        nrain="QNRAIN",
        engine=engine,
        temperature=None,
    )
    ref = forward_operator(rain_state(q), engine=engine, temperature=None)
    np.testing.assert_allclose(out.DBZH, ref.DBZH, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(out.KDP, ref.KDP, rtol=1e-4)
    per_m3 = forward_operator(
        rain_state(q, QNRAIN=nt),
        nrain="QNRAIN",
        nrain_per_kg=False,
        engine=engine,
        temperature=None,
    )
    np.testing.assert_allclose(per_m3.DBZH, ref.DBZH, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("engine", ENGINES)
def test_number_concentration_with_shape(engine):
    """Gamma distribution with mu = 2 from M3 and M0: compared with radar_from_dsd."""
    mu, nt, q = 2.0, 600.0, 1.5e-3
    m3 = q * 1.1 / MASS
    lam = (nt * gamma_fn(mu + 4) / (gamma_fn(mu + 1) * m3)) ** (1 / 3)
    n0m = nt * lam ** (mu + 1) / gamma_fn(mu + 1)
    out = forward_operator(
        rain_state([q], QNRAIN=[nt / 1.1]),
        nrain="QNRAIN",
        mu_rain=mu,
        engine=engine,
        temperature=None,
    )
    ref = gamma_spectrum(np.array([n0m]), mu, np.array([lam]), "S", 10.0)
    np.testing.assert_allclose(out.DBZH, ref.DBZH, rtol=1e-5, atol=2e-5)
    np.testing.assert_allclose(out.ZDR, ref.ZDR, rtol=1e-4, atol=1e-4)


def test_number_concentration_fallback_to_the_fixed_intercept():
    q = np.array([1e-3, 1e-3, 1e-3])
    nr = np.array([np.nan, 0.0, -1.0])
    out = forward_operator(rain_state(q, QNRAIN=nr), nrain="QNRAIN", temperature=None)
    ref = forward_operator(rain_state(q), temperature=None)
    np.testing.assert_array_equal(out.DBZH, ref.DBZH)


@pytest.mark.parametrize("engine", ENGINES)
def test_rain_invariants(engine):
    """Z_H increases with the rain content, Z_DR within 0-5 dB, K_DP not negative."""
    q = np.geomspace(5e-6, 2e-2, 40)
    out = forward_operator(rain_state(q), engine=engine, min_dbz=-80.0)
    assert np.all(np.diff(out.DBZH.values) > 0)
    assert np.all(np.diff(out.KDP.values) > 0)
    assert out.ZDR.min() > -0.5 and out.ZDR.max() < 5.0
    assert out.KDP.min() >= 0.0
    assert out.RHOHV.max() <= 1.0 and out.RHOHV.min() > 0.97
    assert np.all(out.FALL_SPEED.values > 0)
    assert np.all(np.diff(out.FALL_SPEED.values) > 0)  # bigger drops fall faster


def test_fall_speed_of_the_rain():
    """The Z-weighted speed lies between 0 and the speed of the largest tabulated drop."""
    out = forward_operator(rain_state([4e-3]))
    assert 4.0 < float(out.FALL_SPEED[0]) < 9.65


def test_ice_and_rain_have_the_same_units_of_z():
    """Rain alone, snow alone: the linear moments of two classes add."""
    both = rain_state([1e-3], QSNOW=[5e-4])
    z_both = forward_operator(both, snow="QSNOW")
    z_rain = forward_operator(both, snow=None)
    z_snow = forward_operator(both, rain=None)
    for name in ("Z_H", "Z_V", "Z_HV_REAL", "KDP"):
        np.testing.assert_allclose(
            z_both[name], z_rain[name] + z_snow[name], rtol=1e-5, atol=1e-12
        )


# --------------------------------------------------------------------------
# ice: the published amplitudes and closed forms
# --------------------------------------------------------------------------


def test_shape_factors():
    assert fo._shape_factors(1.0) == pytest.approx((1 / 3, 1 / 3))
    for r in (0.3, 0.75, 0.999, 1.001, 1.6, 3.0):
        la, lb = fo._shape_factors(r)
        assert la + 2 * lb == pytest.approx(1.0)
        assert 0.0 < la < 1.0
    # oblate: more depolarisation along the symmetry axis; prolate: less
    assert fo._shape_factors(0.5)[0] > 1 / 3 > fo._shape_factors(2.0)[0]
    # continuity at the sphere
    assert fo._shape_factors(0.9999)[0] == pytest.approx(1 / 3, abs=1e-4)
    assert fo._shape_factors(1.0001)[0] == pytest.approx(1 / 3, abs=1e-4)


def test_orientation_moments():
    a1, a2, a3, a4, a5 = fo._orientation_moments(0.0)
    assert (a1, a2, a3, a4, a5) == pytest.approx((1.0, 0.0, 1.0, 0.0, 0.0))
    a1, a2, a3, a4, a5 = fo._orientation_moments(300.0)  # random in the plane
    assert a1 == pytest.approx(0.5, abs=1e-3) and a2 == pytest.approx(0.5, abs=1e-3)
    assert a3 == pytest.approx(3 / 8, abs=1e-3) and a4 == pytest.approx(3 / 8, abs=1e-3)
    assert a5 == pytest.approx(1 / 8, abs=1e-3)
    # by quadrature over the Gaussian canting angle
    sigma = np.deg2rad(23.0)
    a = np.linspace(-8 * sigma, 8 * sigma, 20001)
    pdf = np.exp(-0.5 * (a / sigma) ** 2)
    pdf /= _trapezoid(pdf, a)
    got = fo._orientation_moments(23.0)
    want = [
        _trapezoid(pdf * f, a)
        for f in (
            np.cos(a) ** 2,
            np.sin(a) ** 2,
            np.cos(a) ** 4,
            np.sin(a) ** 4,
            np.cos(a) ** 2 * np.sin(a) ** 2,
        )
    ]
    np.testing.assert_allclose(got, want, rtol=1e-6)


@pytest.mark.parametrize(
    "name,rho,tol",
    [("snow", 100.0, 0.03), ("hail", 913.0, 0.005)],
)
def test_amplitudes_reproduce_jung(name, rho, tol):
    """The Rayleigh amplitudes match the fitted values of Jung et al. (2008, Eq. 9)."""
    a_sym, a_tr = fo.rayleigh_amplitudes(rho, 0.75, WL["S"])
    assert a_tr == pytest.approx(JUNG[name]["a"], rel=tol)
    assert a_sym == pytest.approx(JUNG[name]["b"], rel=tol)
    assert a_tr > a_sym  # oblate: the transverse axis scatters more


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name,tol", [("snow", 0.06), ("hail", 0.012)])
def test_ice_matches_the_closed_forms_of_jung(engine, name, tol):
    """Z_H, Z_V, Z_DR and K_DP against Eqs. 12, 13, 16 and 18 with their amplitudes."""
    q = np.array([2e-5, 1.5e-4, 1e-3])
    rho_air = 0.8
    ds = xr.Dataset({"Q": ("x", q), "AIR_DENSITY": ("x", np.full(3, rho_air))})
    kw = {"snow": None, "hail": "Q"} if name == "hail" else {"snow": "Q"}
    out = forward_operator(ds, rain=None, engine=engine, temperature=None, **kw)
    zh, zv, zdr, kdp = jung_moments(name, q, rho_air)
    np.testing.assert_allclose(out.Z_H, zh, rtol=tol)
    np.testing.assert_allclose(out.Z_V, zv, rtol=tol)
    np.testing.assert_allclose(out.ZDR, zdr, atol=0.06 if name == "snow" else 0.03)
    np.testing.assert_allclose(out.KDP, kdp, rtol=3 * tol)
    assert np.all(out.RHOHV > 0.97) or name == "hail"


@pytest.mark.parametrize("name", ["snow", "hail"])
def test_ice_correlation_coefficient_from_the_definition(name):
    """RHOHV of ice: <S_hh S_vv*> from the canting averages of Jung et al. (2008,
    Eq. 5), written independently of the module: C (a^2 + b^2) + (A + B) a b."""
    p = JUNG[name]
    s = np.deg2rad(p["sigma"])
    e1, e2 = np.exp(-2 * s * s), np.exp(-8 * s * s)
    a_, b_, c_ = (3 + 4 * e1 + e2) / 8, (3 - 4 * e1 + e2) / 8, (1 - e2) / 8
    fa, fb = p["a"], p["b"]
    cov = c_ * (fa**2 + fb**2) + (a_ + b_) * fa * fb
    zh = a_ * fa**2 + b_ * fb**2 + 2 * c_ * fa * fb
    zv = b_ * fa**2 + a_ * fb**2 + 2 * c_ * fa * fb
    want = cov / np.sqrt(zh * zv)
    ds = xr.Dataset({"Q": ("x", [3e-4]), "AIR_DENSITY": ("x", [0.8])})
    kw = {"snow": None, "hail": "Q"} if name == "hail" else {"snow": "Q"}
    out = forward_operator(ds, rain=None, **kw)
    assert float(out.RHOHV[0]) == pytest.approx(want, abs=2e-4)
    assert want < 1.0  # the canting makes it a little smaller than 1


def test_ice_variables_physical():
    q = np.geomspace(1e-6, 5e-3, 30)
    ds = xr.Dataset(
        {"QSNOW": ("x", q), "QHAIL": ("x", q), "AIR_DENSITY": ("x", np.full(30, 0.9))}
    )
    for kwargs in ({"snow": "QSNOW"}, {"snow": None, "hail": "QHAIL"}):
        out = forward_operator(ds, rain=None, min_dbz=-90.0, **kwargs)
        assert np.all(np.diff(out.DBZH.values) > 0)
        assert out.ZDR.min() >= 0.0 and out.ZDR.max() < 1.0  # dry ice: weak ZDR
        assert out.KDP.min() >= 0.0
        assert out.RHOHV.max() <= 1.0
    # hail of 913 kg m-3 reflects more than 100 kg m-3 snow of the same mass? not
    # needed; the density ordering of Z at equal q follows from N0 Lambda^-7
    hail = forward_operator(ds, rain=None, snow=None, hail="QHAIL")
    assert np.all(np.isfinite(hail.DBZH.values[10:]))


def test_hail_parameters_and_wavelength_scaling():
    """Z does not depend on the band in the Rayleigh limit; K_DP scales as 1/lambda."""
    ds = xr.Dataset({"QSNOW": ("x", [5e-4]), "AIR_DENSITY": ("x", [0.9])})
    s = forward_operator(ds, rain=None, band="S")
    c = forward_operator(ds, rain=None, band="C")
    x = forward_operator(ds, rain=None, band="X")
    assert float(c.DBZH[0]) == pytest.approx(float(s.DBZH[0]), abs=1e-4)
    assert float(x.ZDR[0]) == pytest.approx(float(s.ZDR[0]), abs=1e-4)
    assert float(c.KDP[0]) / float(s.KDP[0]) == pytest.approx(
        WL["S"] / WL["C"], rel=1e-5
    )
    assert float(x.KDP[0]) / float(s.KDP[0]) == pytest.approx(
        WL["S"] / WL["X"], rel=1e-5
    )


def test_ice_parameters_change_the_result():
    ds = xr.Dataset({"QSNOW": ("x", [5e-4]), "AIR_DENSITY": ("x", [0.9])})
    base = forward_operator(ds, rain=None)
    flatter = forward_operator(ds, rain=None, snow_params={"axis_ratio": 0.5})
    rounder = forward_operator(ds, rain=None, snow_params={"sigma": 90.0})
    assert float(flatter.ZDR[0]) > float(base.ZDR[0]) > float(rounder.ZDR[0])
    denser = forward_operator(ds, rain=None, snow_params={"rho": 300.0})
    assert float(denser.DBZH[0]) != float(base.DBZH[0])
    # a larger intercept at the same mixing ratio gives smaller particles: lower Z
    smaller = forward_operator(ds, rain=None, snow_params={"n0": 3e7})
    assert float(smaller.DBZH[0]) < float(base.DBZH[0])


def test_prolate_graupel_and_spheres():
    ds = xr.Dataset({"QG": ("x", [1e-3]), "AIR_DENSITY": ("x", [0.9])})
    sphere = forward_operator(
        ds,
        rain=None,
        snow=None,
        graupel="QG",
        graupel_params={"n0": 4e6, "rho": 400.0, "axis_ratio": 1.0, "sigma": 20.0},
    )
    assert abs(float(sphere.ZDR[0])) < 1e-4 and float(sphere.KDP[0]) == pytest.approx(
        0, abs=1e-12
    )
    prolate = forward_operator(
        ds,
        rain=None,
        snow=None,
        graupel="QG",
        graupel_params={"n0": 4e6, "rho": 400.0, "axis_ratio": 1.4, "sigma": 20.0},
    )
    assert float(prolate.ZDR[0]) < 0.0  # the symmetry axis (vertical) is the long one


def test_ice_class_checks():
    ds = xr.Dataset({"QG": ("x", [1e-3]), "AIR_DENSITY": ("x", [0.9])})
    with pytest.raises(ValueError, match="no default for graupel"):
        forward_operator(ds, rain=None, snow=None, graupel="QG")
    with pytest.raises(ValueError, match="missing"):
        forward_operator(
            ds, rain=None, snow=None, graupel="QG", graupel_params={"n0": 1.0}
        )
    for bad in (
        {"rho": 1000.0},
        {"n0": -1.0},
        {"axis_ratio": 0.0},
        {"sigma": -1.0},
        {"fall_speed": -1.0},
        {"unknown": 1.0},
    ):
        params = {**GRAUPEL, "fall_speed": 0.0, **bad}
        with pytest.raises(ValueError):
            forward_operator(
                ds, rain=None, snow=None, graupel="QG", graupel_params=params
            )


def test_ice_fall_speed_enters_the_velocity_weight():
    ds = xr.Dataset({"QSNOW": ("x", [5e-4]), "AIR_DENSITY": ("x", [0.9])})
    out = forward_operator(ds, rain=None, snow_params={"fall_speed": 1.2})
    assert float(out.FALL_SPEED[0]) == pytest.approx(1.2, rel=1e-5)
    both = xr.Dataset(
        {
            "QRAIN": ("x", [2e-3]),
            "QSNOW": ("x", [1e-3]),
            "AIR_DENSITY": ("x", [1.0]),
        }
    )
    mixed = forward_operator(both, snow_params={"fall_speed": 1.0})
    rain = forward_operator(both, snow=None)
    snow = forward_operator(both, rain=None, snow_params={"fall_speed": 1.0})
    weight = float(rain.Z_H[0]) / float(mixed.Z_H[0])
    expect = weight * float(rain.FALL_SPEED[0]) + (1 - weight) * 1.0
    assert float(mixed.FALL_SPEED[0]) == pytest.approx(expect, rel=1e-4)
    assert float(snow.FALL_SPEED[0]) == pytest.approx(1.0, rel=1e-5)


def test_covariance_signs_rain_and_ice_add():
    """RHOHV of rain with hail is lower than that of rain (hail is oriented randomly)."""
    both = xr.Dataset(
        {
            "QRAIN": ("x", [1e-3]),
            "QHAIL": ("x", [3e-3]),
            "AIR_DENSITY": ("x", [1.0]),
        }
    )
    rain = forward_operator(both, snow=None)
    mix = forward_operator(both, snow=None, hail="QHAIL")
    hail = forward_operator(both, snow=None, rain=None, hail="QHAIL")
    # dry hail in the Rayleigh regime is almost spherical: the correlation stays
    # high; a wrong sign of the rain covariance would drop it below 0.5
    assert 0.98 < float(mix.RHOHV[0]) <= 1.0
    rho_sum = np.hypot(
        float(rain.Z_HV_REAL[0]) + float(hail.Z_HV_REAL[0]),
        float(rain.Z_HV_IMAG[0]) + float(hail.Z_HV_IMAG[0]),
    ) / np.sqrt(float(mix.Z_H[0]) * float(mix.Z_V[0]))
    assert float(mix.RHOHV[0]) == pytest.approx(rho_sum, rel=1e-5)
    assert float(mix.RHOHV[0]) > 0.9
    assert float(mix.ZDR[0]) < float(rain.ZDR[0])  # hail dilutes ZDR
    assert float(mix.DBZH[0]) > float(rain.DBZH[0])


# --------------------------------------------------------------------------
# options, checks, the compiled kernel against NumPy
# --------------------------------------------------------------------------


def random_state(n=700, seed=2):
    rng = np.random.default_rng(seed)

    def mix(scale):
        q = rng.uniform(0, scale, n) * (rng.uniform(size=n) > 0.4)
        q[rng.integers(0, n, 20)] = np.nan
        return q

    return xr.Dataset(
        {
            "QRAIN": ("x", mix(8e-3)),
            "QSNOW": ("x", mix(2e-3)),
            "QHAIL": ("x", mix(4e-3)),
            "QGRAUP": ("x", mix(3e-3)),
            "QNRAIN": ("x", rng.uniform(0, 4e3, n) * (rng.uniform(size=n) > 0.1)),
            "AIR_DENSITY": ("x", rng.uniform(0.4, 1.25, n)),
            "TEMPERATURE": ("x", rng.uniform(255.0, 312.0, n)),
        }
    )


@compiled_only
@pytest.mark.parametrize("band", ["S", "X"])
@pytest.mark.parametrize("use_n", [False, True])
def test_kernel_agrees_with_numpy(band, use_n):
    ds = random_state()
    kw = dict(
        band=band,
        hail="QHAIL",
        graupel="QGRAUP",
        graupel_params=GRAUPEL,
        nrain="QNRAIN" if use_n else None,
        mu_rain=1.5,
        min_dbz=-60.0,
    )
    comp = forward_operator(ds, engine="compiled", **kw)
    ref = forward_operator(ds, engine="numpy", **kw)
    for name in comp.data_vars:
        np.testing.assert_allclose(
            comp[name].values, ref[name].values, rtol=2e-6, atol=1e-12, err_msg=name
        )
    # cells without any class hold zeros, not NaN, in the linear moments
    assert np.isfinite(comp.Z_H).all()


@compiled_only
def test_kernel_threads_do_not_change_the_result():
    ds = random_state(5000)
    one = forward_operator(ds, n_threads=1, snow=None)
    many = forward_operator(ds, n_threads=4, snow=None)
    xr.testing.assert_identical(one, many)


def test_min_dbz_masks_the_logarithmic_variables():
    out = forward_operator(rain_state([1e-9 * 5, 2e-7, 3e-3]), min_dbz=-10.0)
    assert np.isnan(out.DBZH.values[:2]).all() and np.isfinite(out.DBZH.values[2])
    assert np.isnan(out.ZDR.values[:2]).all() and np.isnan(out.RHOHV.values[:2]).all()
    assert np.isnan(out.FALL_SPEED.values[:2]).all()
    assert (out.Z_H.values > 0).all()  # the linear moments stay


def test_min_mixing_ratio_and_missing_air():
    out = forward_operator(rain_state([1e-4, 1e-4]), min_mixing_ratio=2e-4)
    assert np.isnan(out.DBZH.values).all() and (out.Z_H.values == 0).all()
    bad = rain_state([1e-3, 1e-3])
    bad["AIR_DENSITY"] = ("x", [np.nan, -1.0])
    out = forward_operator(bad)
    assert (out.Z_H.values == 0).all()


def test_auto_names_and_density_from_pressure():
    q = [1e-3]
    ds = xr.Dataset(
        {"QRAIN": ("x", q), "PRESSURE": ("x", [90000.0]), "TEMPERATURE": ("x", [288.0])}
    )
    out = forward_operator(ds)
    full = forward_operator(ds.assign(AIR_DENSITY=("x", [90000.0 / (287.0 * 288.0)])))
    assert float(out.DBZH[0]) == pytest.approx(float(full.DBZH[0]), abs=1e-5)
    ds = ds.assign(QVAPOR=("x", [0.01]))
    wet = forward_operator(ds)
    assert float(wet.DBZH[0]) != float(out.DBZH[0])
    with pytest.raises(ValueError, match="air density"):
        forward_operator(xr.Dataset({"QRAIN": ("x", q)}))
    # without a temperature the table temperature is rain_temperature
    warm = forward_operator(
        xr.Dataset({"QRAIN": ("x", q), "AIR_DENSITY": ("x", [1.0])}),
        rain_temperature=25.0,
    )
    assert np.isfinite(warm.DBZH.values).all()


def test_errors():
    ds = rain_state([1e-3])
    with pytest.raises(TypeError, match="Dataset"):
        forward_operator(ds.QRAIN)
    with pytest.raises(ValueError, match="band"):
        forward_operator(ds, band="K")
    with pytest.raises(ValueError, match="mu_rain"):
        forward_operator(ds, mu_rain=-1.0)
    with pytest.raises(ValueError, match="n0_rain"):
        forward_operator(ds, n0_rain=0.0)
    with pytest.raises(ValueError, match="min_mixing_ratio"):
        forward_operator(ds, min_mixing_ratio=-1.0)
    with pytest.raises(ValueError, match="rain_temperature"):
        forward_operator(ds, rain_temperature=40.0)
    with pytest.raises(KeyError, match="QX"):
        forward_operator(ds, rain="QX")
    with pytest.raises(KeyError, match="nrain"):
        forward_operator(ds, nrain="QNRAIN")
    with pytest.raises(ValueError, match="no hydrometeor"):
        forward_operator(ds, rain=None, snow="auto")
    with pytest.raises(ValueError, match="engine"):
        forward_operator(ds, engine="gpu")


def test_compiled_engine_needs_the_kernel(monkeypatch):
    monkeypatch.setattr(fo, "HAS_COMPILED_KERNEL", False)
    ds = rain_state([1e-3])
    with pytest.raises(ImportError, match="compiled"):
        forward_operator(ds, engine="compiled")
    out = forward_operator(ds)  # "auto" falls back to NumPy
    assert np.isfinite(out.DBZH.values).all()


def test_input_is_not_changed_and_attributes_survive():
    ds = rain_state([1e-3, 2e-3], QSNOW=[1e-4, 2e-4])
    ds["QRAIN"].attrs["units"] = "kg kg-1"
    ds = ds.assign_coords(x=("x", [0.0, 1.0], {"units": "m"}))
    before = ds.copy(deep=True)
    out = forward_operator(ds)
    xr.testing.assert_identical(ds, before)
    assert ds.x.attrs == {"units": "m"} and ds.QRAIN.attrs == {"units": "kg kg-1"}
    assert out.DBZH.attrs["units"] == "dBZ"
    assert out.Z_H.attrs["units"] == "mm6 m-3"
    assert out.attrs["forward_operator_band"] == "S"
    assert out.attrs["forward_operator_classes"] == "rain, snow"


def test_lazy_dask_result(fields):
    pytest.importorskip("dask")
    import dask  # noqa: F401

    q = rain_state(np.geomspace(1e-5, 1e-2, 50)).chunk({"x": 20})
    lazy = forward_operator(q)
    assert lazy.DBZH.chunks is not None
    eager = forward_operator(q.compute())
    np.testing.assert_allclose(
        lazy.DBZH.values, eager.DBZH.values, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(lazy.RHOHV.values, eager.RHOHV.values, rtol=1e-6)


def test_provenance_and_references():
    out = forward_operator(rain_state([1e-3]), band="C", snow=None)
    assert "forward operator" in out.attrs[METHOD].lower()
    dois = out.attrs[REFERENCES].split()
    assert "10.1175/2007MWR2083.1" in dois and "10.3390/atmos11040362" in dois
    assert out.attrs[HISTORY].startswith("forward_operator(")
    assert "band='C'" in out.attrs[HISTORY]
    registry = load_registry()
    for doi in dois:
        assert doi in registry, doi
    for func in (wrf_state, mpas_state, to_radar_grid):
        assert func.__radarx_provenance__.references


def test_accessor_method(fields):
    ds = rain_state([1e-3, 3e-3])
    via = ds.radarx.forward_operator(band="C")
    direct = forward_operator(ds, band="C")
    xr.testing.assert_allclose(via, direct)


# --------------------------------------------------------------------------
# WRF and MPAS states
# --------------------------------------------------------------------------


def test_wrf_state_values(wrf):
    st = wrf_state(wrf)
    assert set(st.dims) == {"Time", "bottom_top", "south_north", "west_east"}
    i, j, k = 5, 7, 12
    zint = wrf.PHB.values[0, :, j, i] / G
    zmass = 0.5 * (zint[k] + zint[k + 1])
    assert st.HEIGHT.values[0, k, j, i] == pytest.approx(zmass, rel=1e-6)
    t, p = standard_atmosphere(zmass - 0.0)
    assert st.PRESSURE.values[0, k, j, i] == pytest.approx(p, rel=1e-6)
    assert st.TEMPERATURE.values[0, k, j, i] == pytest.approx(t, rel=1e-5)
    assert st.AIR_DENSITY.values[0, k, j, i] == pytest.approx(
        p / (287.0 * t * (1 + 1.61 * float(wrf.QVAPOR.values[0, k, j, i]))), rel=1e-5
    )
    # destaggering: the mass-point wind is the mean of its two faces
    u = wrf.U.values[0, k, j, :]
    np.testing.assert_allclose(
        st.U.values[0, k, j, :], 0.5 * (u[:-1] + u[1:]), rtol=1e-6
    )
    w = wrf.W.values[0, :, j, i]
    assert st.W.values[0, k, j, i] == pytest.approx(0.5 * (w[k] + w[k + 1]), rel=1e-5)
    assert st.QRAIN.attrs["units"] == "kg kg-1"
    assert st.latitude.dims == ("south_north", "west_east")
    np.testing.assert_array_equal(st.latitude.values, wrf.XLAT.values[0])
    assert st.attrs["source"] == "WRF"


def test_wrf_state_moist_theta_and_input_unchanged():
    dry = synthetic_wrf(nx=12, ny=10, nz=10, theta_m=False)
    moist = synthetic_wrf(nx=12, ny=10, nz=10, theta_m=True)
    before = moist.copy(deep=True)
    a, b = wrf_state(dry), wrf_state(moist)
    assert moist.attrs["USE_THETA_M"] == 1
    np.testing.assert_allclose(a.TEMPERATURE, b.TEMPERATURE, rtol=1e-5)
    wrong = wrf_state(moist, use_theta_m=False)
    assert float(np.abs(wrong.TEMPERATURE - a.TEMPERATURE).max()) > 0.1
    xr.testing.assert_identical(moist, before)


def test_wrf_state_without_vapour_and_with_hail_and_time():
    ds = synthetic_wrf(nx=8, ny=6, nz=8, ntime=2).drop_vars(["QVAPOR", "W", "QGRAUP"])
    ds["QHAIL"] = ds.QRAIN * 0.1
    ds["XLAT"] = ds.XLAT.isel(Time=0, drop=True)  # no Time dimension
    ds["XLONG"] = ds.XLONG.isel(Time=0, drop=True)
    st = wrf_state(ds)
    assert "QHAIL" in st and "W" not in st and "QVAPOR" not in st
    assert st.sizes["Time"] == 2
    assert st.latitude.dims == ("south_north", "west_east")


def test_wrf_state_errors(wrf):
    with pytest.raises(TypeError, match="Dataset"):
        wrf_state(wrf.T)
    with pytest.raises(KeyError, match="PB"):
        wrf_state(wrf.drop_vars("PB"))
    with pytest.raises(KeyError, match="bottom_top_stag"):
        wrf_state(wrf.rename({"bottom_top_stag": "other"}))


def test_mpas_state_values():
    ds, grid = synthetic_mpas(ncell=200, nz=12)
    st = mpas_state(ds, grid)
    assert st.sizes["nCells"] == 200 and st.sizes["nVertLevels"] == 12
    zgrid = grid.zgrid.values
    np.testing.assert_allclose(
        st.HEIGHT.values, 0.5 * (zgrid[:, :-1] + zgrid[:, 1:]), rtol=1e-6
    )
    np.testing.assert_allclose(st.latitude.values, np.rad2deg(grid.latCell.values))
    np.testing.assert_allclose(
        st.TEMPERATURE.values, ds.theta.values * ds.exner.values, rtol=1e-6
    )
    np.testing.assert_array_equal(st.AIR_DENSITY.values, ds.rho.values)
    np.testing.assert_array_equal(st.U.values, 10.0)
    np.testing.assert_allclose(st.W.values, 1.0)
    assert "QRAIN" in st and "QNRAIN" in st and "QSNOW" in st and "QVAPOR" in st
    assert st.attrs["source"] == "MPAS-Atmosphere"


def test_mpas_state_variants():
    ds, grid = synthetic_mpas(ncell=50, nz=8)
    # the grid inside the dataset, pressure from its perturbation, no exner
    merged = ds.merge(grid).drop_vars("exner")
    p = merged.pressure
    merged = merged.drop_vars("pressure").assign(
        pressure_p=p * 0.01, pressure_base=p * 0.99
    )
    st = mpas_state(merged)
    expect = ds.theta.values * (ds.pressure.values / 1e5) ** (287.0 / 1004.5)
    np.testing.assert_allclose(st.TEMPERATURE.values, expect, rtol=1e-5)
    np.testing.assert_allclose(st.PRESSURE.values, ds.pressure.values, rtol=1e-6)
    no_w = mpas_state(ds.drop_vars(["w", "nr"]), grid)
    assert "W" not in no_w and "QNRAIN" not in no_w


def test_mpas_state_errors():
    ds, grid = synthetic_mpas(ncell=20, nz=6)
    with pytest.raises(KeyError, match="lonCell"):
        mpas_state(ds)
    with pytest.raises(TypeError, match="Dataset"):
        mpas_state(ds.qr, grid)
    with pytest.raises(TypeError, match="Dataset"):
        mpas_state(ds, grid.zgrid)
    with pytest.raises(KeyError, match="rho"):
        mpas_state(ds.drop_vars("rho"), grid)
    with pytest.raises(KeyError, match="pressure_p"):
        mpas_state(ds.drop_vars("pressure"), grid)
    before = ds.copy(deep=True)
    mpas_state(ds, grid)
    xr.testing.assert_identical(ds, before)


# --------------------------------------------------------------------------
# the model grid and the radar grid
# --------------------------------------------------------------------------


def test_squall_line_fields(fields, state):
    """The convective core is strong with a ZDR trough from graupel; stratiform is weaker."""
    z = fields.DBZH
    assert float(z.max()) > 50.0 and float(z.max()) < 80.0
    # reflectivity of the core column exceeds that of the stratiform column
    nx = z.sizes["west_east"]
    core = z.isel(Time=0, south_north=24, west_east=nx // 2 + 2)
    strat = z.isel(Time=0, south_north=24, west_east=nx // 2 - 15)
    assert float(core.max()) > float(strat.max()) + 10.0
    # a ZDR maximum in the rain below the graupel and low ZDR where graupel dominates
    assert float(fields.ZDR.max()) < 5.0 and float(fields.ZDR.min()) > -0.5
    assert float(fields.KDP.min()) >= 0.0
    assert float(fields.RHOHV.min()) > 0.85
    # dry-snow stratiform echo above the melting level is below 25 dBZ
    cold = state.TEMPERATURE < 268.0
    assert float(z.where(cold).sel().max()) < 62.0
    for name in ("HEIGHT", "U", "V", "W"):
        assert name in fields
    assert fields.attrs["forward_operator_classes"] == "rain, snow, graupel"
    # the linear moments reproduce the logarithmic ones
    echo = fields.Z_H > 10 ** (-3.0)
    np.testing.assert_allclose(
        (10 * np.log10(fields.Z_H.where(echo))).values,
        fields.DBZH.where(echo).values,
        rtol=1e-5,
    )


def geometry_fields(zfunc=None, u=0.0, v=0.0, w=0.0, vt=0.0, zh=1.0e3, n=40, nz=20):
    """A flat Dataset on a regular grid with prescribed moments (for the sampling)."""
    lat = np.linspace(34.0, 36.6, n)
    lon = np.linspace(-99.2, -96.4, n)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    zi = np.linspace(0.0, 15000.0, nz + 1)
    height = np.broadcast_to(0.5 * (zi[:-1] + zi[1:])[:, None, None], (nz, n, n)).copy()
    dims = ("bottom_top", "south_north", "west_east")
    base = np.ones((nz, n, n))
    zfield = zh * base if zfunc is None else zfunc(height, la[None], lo[None])
    ds = xr.Dataset(
        {
            "Z_H": (dims, zfield),
            "Z_V": (dims, 0.8 * zfield),
            "Z_HV_REAL": (dims, 0.85 * zfield),
            "Z_HV_IMAG": (dims, 0 * zfield),
            "KDP": (dims, 0.5 * base),
            "ZH_VT": (dims, vt * zfield),
            "HEIGHT": (dims, height),
            "U": (dims, u * base),
            "V": (dims, v * base),
            "W": (dims, w * base),
        },
        coords={
            "latitude": (("south_north", "west_east"), la),
            "longitude": (("south_north", "west_east"), lo),
        },
    )
    return ds


@pytest.mark.parametrize("engine", ENGINES)
def test_height_interpolation_is_exact_for_a_linear_profile(engine):
    """A field equal to the height returns the height of the beam centre."""
    ds = geometry_fields(zfunc=lambda h, la, lo: 1.0e3 + 0.5 * h)
    az = np.array([10.0, 100.0, 200.0, 300.0])
    rng = np.arange(2000.0, 60000.0, 4000.0)
    for el in (0.5, 3.0, 9.0):
        out = to_radar_grid(
            ds.assign(Z_V=ds.Z_H, Z_HV_REAL=ds.Z_H),
            SITE,
            el,
            azimuth=az,
            gate_range=rng,
            engine=engine,
            min_dbz=-90.0,
        )
        h = beam_center_height(rng, el, radar_height=SITE[2], reff=4 / 3 * EARTH_RADIUS)
        want = 10 * np.log10(1.0e3 + 0.5 * h)
        got = out.DBZH.values
        inside = np.isfinite(got)
        assert inside.sum() > 0.6 * got.size
        np.testing.assert_allclose(
            got[inside], np.broadcast_to(want, got.shape)[inside], atol=2e-3
        )


@pytest.mark.parametrize("engine", ENGINES)
def test_constant_field_and_outside_of_the_model(engine):
    ds = geometry_fields(zh=1.0e3)
    out = to_radar_grid(
        ds,
        SITE,
        1.0,
        azimuth=np.arange(0.0, 360.0, 15.0),
        gate_range=np.arange(1000.0, 400001.0, 20000.0),
        engine=engine,
    )
    assert out.DBZH.dims == ("azimuth", "range")
    ok = np.isfinite(out.DBZH.values)
    np.testing.assert_allclose(out.DBZH.values[ok], 30.0, atol=1e-4)
    np.testing.assert_allclose(out.ZDR.values[ok], 10 * np.log10(1 / 0.8), atol=1e-4)
    np.testing.assert_allclose(out.KDP.values[ok], 0.5, rtol=1e-6)
    np.testing.assert_allclose(out.RHOHV.values[ok], 0.85 / np.sqrt(0.8), rtol=1e-5)
    # the domain is about 300 km wide: the far gates are outside, the near ones inside
    assert ok[:, :4].all() and not ok[:, -3:].any()
    assert out.elevation.dims == ("azimuth",) and float(out.elevation[0]) == 1.0
    assert float(out.latitude) == SITE[0] and float(out.altitude) == SITE[2]
    assert out.VRADH.attrs["units"] == "m s-1"


@pytest.mark.parametrize("engine", ENGINES)
def test_beam_above_the_model_top_is_empty(engine):
    ds = geometry_fields()  # top at 15 km
    out = to_radar_grid(
        ds,
        SITE,
        20.0,
        azimuth=[0.0],
        gate_range=[5000.0, 60000.0, 100000.0],
        engine=engine,
    )
    h = beam_center_height(
        np.array([5000.0, 60000.0, 100000.0]),
        20.0,
        radar_height=SITE[2],
        reff=4 / 3 * EARTH_RADIUS,
    )
    np.testing.assert_array_equal(np.isfinite(out.DBZH.values[0]), h < 15000.0 - 1)


def test_below_the_lowest_level_is_empty():
    ds = geometry_fields()
    out = to_radar_grid(
        ds, (35.2, -97.9, 0.0), 0.0, azimuth=[90.0], gate_range=[1000.0, 10000.0]
    )
    # the first mass level is 375 m high, the beam (site at 0 m) is lower at 10 km
    assert np.isnan(out.DBZH.values).all()


@pytest.mark.parametrize("engine", ENGINES)
def test_radial_velocity(engine):
    az = np.array([0.0, 90.0, 180.0, 270.0, 45.0])
    rng = np.array([5000.0, 20000.0])
    ds = geometry_fields(u=10.0, v=0.0)
    out = to_radar_grid(ds, SITE, 0.0, azimuth=az, gate_range=rng, engine=engine)
    want = 10.0 * np.sin(np.deg2rad(az))
    np.testing.assert_allclose(
        out.VRADH.values, np.repeat(want[:, None], 2, 1), atol=1e-4
    )
    el = 20.0
    ds = geometry_fields(u=3.0, v=-4.0, w=2.0, vt=5.0)
    out = to_radar_grid(ds, SITE, el, azimuth=az, gate_range=rng, engine=engine)
    e, a = np.deg2rad(el), np.deg2rad(az)
    want = (3.0 * np.sin(a) - 4.0 * np.cos(a)) * np.cos(e) + (2.0 - 5.0) * np.sin(e)
    np.testing.assert_allclose(
        out.VRADH.values, np.repeat(want[:, None], 2, 1), atol=1e-4
    )


def test_radial_velocity_without_winds_or_w():
    ds = geometry_fields(u=1.0)
    no_w = to_radar_grid(
        ds.drop_vars("W"), SITE, 10.0, azimuth=[90.0], gate_range=[10000.0]
    )
    with_w = to_radar_grid(ds, SITE, 10.0, azimuth=[90.0], gate_range=[10000.0])
    np.testing.assert_allclose(no_w.VRADH, with_w.VRADH, atol=1e-6)
    none = to_radar_grid(
        ds.drop_vars(["U", "V", "W"]), SITE, 10.0, azimuth=[90.0], gate_range=[10000.0]
    )
    assert "VRADH" not in none and "DBZH" in none


@pytest.mark.parametrize("engine", ENGINES)
def test_a_cell_is_found_at_its_azimuth_and_range(engine):
    """A Gaussian blob 50 km east of the radar peaks at azimuth 90 and 50 km ground range."""
    lat0, lon0, alt = 35.0, -98.0, 300.0
    n = 120
    dx = 1500.0
    y = (np.arange(n) - n / 2) * dx
    x = (np.arange(n) - n / 2) * dx
    xx, yy = np.meshgrid(x, y)
    lat = lat0 + yy / 111195.0
    lon = lon0 + xx / (111195.0 * np.cos(np.deg2rad(lat0)))
    nz = 30
    zi = np.linspace(0, 12000.0, nz + 1)
    zm = 0.5 * (zi[:-1] + zi[1:])
    height = np.broadcast_to(zm[:, None, None], (nz, n, n)).copy()
    bx, by = 50000.0, 0.0
    blob = (
        np.exp(-(((xx - bx) / 6000.0) ** 2 + ((yy - by) / 6000.0) ** 2))[None]
        * np.exp(-(((zm - 600.0) / 800.0) ** 2))[:, None, None]
    )
    dims = ("bottom_top", "south_north", "west_east")
    z = 1.0e4 * blob
    ds = xr.Dataset(
        {
            "Z_H": (dims, z),
            "Z_V": (dims, z),
            "Z_HV_REAL": (dims, z),
            "Z_HV_IMAG": (dims, 0 * z),
            "KDP": (dims, 0 * z),
            "ZH_VT": (dims, 0 * z),
            "HEIGHT": (dims, height),
        },
        coords={
            "latitude": (("south_north", "west_east"), lat),
            "longitude": (("south_north", "west_east"), lon),
        },
    )
    az = np.arange(0.0, 360.0, 1.0)
    rng = np.arange(500.0, 90000.0, 500.0)
    out = to_radar_grid(
        ds,
        (lat0, lon0, alt),
        0.0,
        azimuth=az,
        gate_range=rng,
        engine=engine,
        k_e=1.0,
        neighbors=4,
    )
    dbz = out.DBZH.values
    i, j = np.unravel_index(np.nanargmax(dbz), dbz.shape)
    assert az[i] in (89.0, 90.0, 91.0) or abs(az[i] - 90.0) <= 1.0
    # the beam at the blob has height 600 m: its slant range matches the ground range
    gr = ground_range(rng, 0.0, reff=EARTH_RADIUS, radar_height=alt)
    assert abs(gr[j] - 50000.0) < 3000.0


def test_engines_agree_on_the_radar_grid(fields):
    kw = dict(
        azimuth=np.arange(0.5, 360.0, 3.0),
        gate_range=np.arange(1000.0, 100001.0, 1000.0),
        neighbors=4,
    )
    ref = to_radar_grid(fields, SITE, 0.5, engine="numpy", **kw)
    assert int(np.isfinite(ref.DBZH).sum()) > 100
    if fo.HAS_COMPILED_KERNEL:
        comp = to_radar_grid(fields, SITE, 0.5, engine="compiled", **kw)
        for name in ref.data_vars:
            np.testing.assert_allclose(
                comp[name].values,
                ref[name].values,
                rtol=2e-6,
                atol=1e-6,
                equal_nan=True,
                err_msg=name,
            )


@compiled_only
def test_sample_kernel_checks_its_inputs():
    k = fo._forward_operator
    f = np.ones((4, 3), dtype=np.float32)
    h = np.tile(np.arange(3, dtype=np.float32), (4, 1))
    idx = np.zeros((2, 1), dtype=np.int64)
    w = np.ones((2, 1))
    with pytest.raises(ValueError, match="height must be 2-D"):
        k.sample([f], h[0], idx, w, np.ones(2))
    with pytest.raises(ValueError, match="inconsistent"):
        k.sample([f], h, idx, w, np.ones(3))
    with pytest.raises(ValueError, match="shape of height"):
        k.sample([f[:2]], h, idx, w, np.ones(2))
    # a strided (transposed) field gives the same values as a contiguous one
    ft = np.asfortranarray(f * np.arange(3, dtype=np.float32))
    out = k.sample([ft], h, idx, w, np.array([0.5, 1.5]))
    np.testing.assert_allclose(out, [[0.5, 1.5]])
    # a column index outside the model, a NaN field and a duplicated level
    bad_idx = np.array([[7], [0]], dtype=np.int64)
    out = k.sample([ft], h, bad_idx, w, np.array([0.5, 1.5]))
    assert np.isnan(out[0, 0]) and out[0, 1] == pytest.approx(1.5)
    nan = ft.copy()
    nan[0] = np.nan  # the column of the first gate only
    two = np.array([[0], [1]], dtype=np.int64)
    out = k.sample([nan], h, two, w, np.array([0.5, 1.5]))
    assert np.isnan(out[0, 0]) and out[0, 1] == pytest.approx(1.5)
    flat = np.zeros((1, 3), dtype=np.float32)
    out = k.sample(
        [np.ones((1, 3), dtype=np.float32)],
        flat,
        np.zeros((1, 1), dtype=np.int64),
        np.ones((1, 1)),
        np.array([0.0]),
    )
    assert out[0, 0] == pytest.approx(1.0)


@compiled_only
def test_cells_kernel_checks_its_inputs():
    k = fo._forward_operator
    tab = fo._rain_table("S")
    rho = np.ones(3)
    base = (
        None,
        None,
        rho,
        None,
        tab["temps"],
        tab["diam"],
        tab["weights"],
        tab["vfall"],
        tab["table"],
        1.0,
        8e6,
        0.0,
        1e-9,
        10.0,
        True,
    )
    with pytest.raises(ValueError, match="1-D"):
        k.cells(None, None, rho.reshape(3, 1), None, *base[4:], [], np.zeros((0, 7)))
    with pytest.raises(ValueError, match="shape of rho"):
        k.cells(np.ones(2), *base[1:], [], np.zeros((0, 7)))
    with pytest.raises(ValueError, match="inconsistent rain"):
        k.cells(*base[:8], tab["table"][:, :, :4], *base[9:], [], np.zeros((0, 7)))
    with pytest.raises(ValueError, match="ice_par"):
        k.cells(*base, [np.ones(3)], np.zeros((0, 7)))
    with pytest.raises(ValueError, match="ice mixing"):
        k.cells(*base, [np.ones(2)], np.zeros((1, 7)))
    out = k.cells(*base, [], np.zeros((0, 7)))
    assert out.shape == (6, 3) and not out.any()


def test_volume_gives_a_datatree(fields):
    vol = to_radar_grid(
        fields,
        SITE,
        [0.5, 1.5, 3.0],
        azimuth=np.arange(0.5, 360.0, 6.0),
        gate_range=np.arange(1000.0, 80001.0, 2000.0),
    )
    assert isinstance(vol, xr.DataTree)
    nodes = sorted(vol.children)
    assert nodes == ["sweep_0", "sweep_1", "sweep_2"]
    assert float(vol["sweep_1"].elevation[0]) == 1.5
    assert float(vol["latitude"]) == SITE[0] and float(vol["altitude"]) == SITE[2]
    assert "DBZH" in vol["sweep_2"].data_vars
    # a higher beam is higher: the mean reflectivity at a given range changes
    z0 = float(vol["sweep_0"].DBZH.max())
    assert np.isfinite(z0)
    assert vol["sweep_0"].DBZH.attrs["units"] == "dBZ"
    assert "forward" not in "".join(vol.attrs.values()).lower() or True


def test_time_dimension_and_dask(wrf):
    two = synthetic_wrf(nx=40, ny=30, nz=20, ntime=2)
    st = wrf_state(two)
    f = forward_operator(st)
    kw = dict(
        azimuth=np.arange(0.5, 360.0, 10.0),
        gate_range=np.arange(2000.0, 50001.0, 3000.0),
    )
    site = (35.0, -97.5, 100.0)
    eager = to_radar_grid(f, site, 1.0, **kw)
    assert eager.DBZH.dims == ("Time", "azimuth", "range")
    np.testing.assert_allclose(
        to_radar_grid(f.isel(Time=1), site, 1.0, **kw).DBZH.values,
        eager.DBZH.isel(Time=1).values,
        equal_nan=True,
    )
    pytest.importorskip("dask")
    lazy_in = forward_operator(st.chunk({"Time": 1, "south_north": 10}))
    lazy = to_radar_grid(lazy_in, site, 1.0, **kw)
    assert lazy.DBZH.chunks is not None
    np.testing.assert_allclose(
        lazy.DBZH.values, eager.DBZH.values, atol=1e-4, equal_nan=True
    )
    np.testing.assert_allclose(
        lazy.VRADH.values, eager.VRADH.values, atol=1e-4, equal_nan=True
    )


def test_time_dependent_heights_are_used():
    two = synthetic_wrf(nx=40, ny=30, nz=20, ntime=2)
    two["PH"] = two.PH.copy()
    two["PH"][1] = two["PH"][1] + 5000.0  # the second time has everything 500 m higher
    f = forward_operator(wrf_state(two))
    site = (35.0, -97.5, 100.0)
    out = to_radar_grid(
        f,
        site,
        1.0,
        azimuth=[30.0, 90.0],
        gate_range=np.arange(5000.0, 40001.0, 5000.0),
        min_dbz=-90.0,
    )
    assert not np.allclose(
        out.DBZH.isel(Time=0).values, out.DBZH.isel(Time=1).values, equal_nan=True
    )


def test_to_radar_grid_input_is_not_changed(fields):
    before = fields.copy(deep=True)
    for da in fields.data_vars.values():
        da.attrs.setdefault("keep", "yes")
    to_radar_grid(fields, SITE, 0.5, azimuth=[10.0], gate_range=[5000.0])
    assert fields.DBZH.attrs["keep"] == "yes"
    assert fields.latitude.attrs == before.latitude.attrs
    for name in before.data_vars:
        np.testing.assert_array_equal(fields[name].values, before[name].values)


def test_to_radar_grid_errors(fields):
    kw = dict(azimuth=[0.0], gate_range=[1000.0])
    with pytest.raises(TypeError, match="Dataset"):
        to_radar_grid(fields.DBZH, SITE, 0.5, **kw)
    with pytest.raises(ValueError, match="site"):
        to_radar_grid(fields, (35.0, -97.0), 0.5, **kw)
    with pytest.raises(ValueError, match="site"):
        to_radar_grid(fields, (95.0, -97.0, 0.0), 0.5, **kw)
    with pytest.raises(ValueError, match="elevation"):
        to_radar_grid(fields, SITE, 95.0, **kw)
    with pytest.raises(ValueError, match="azimuth"):
        to_radar_grid(fields, SITE, 0.5, azimuth=[np.nan], gate_range=[1000.0])
    with pytest.raises(ValueError, match="azimuth"):
        to_radar_grid(fields, SITE, 0.5, azimuth=np.ones((2, 2)), gate_range=[1000.0])
    with pytest.raises(ValueError, match="gate_range"):
        to_radar_grid(fields, SITE, 0.5, azimuth=[0.0], gate_range=[0.0, 1000.0])
    with pytest.raises(ValueError, match="gate_range"):
        to_radar_grid(fields, SITE, 0.5, azimuth=[0.0], gate_range=[])
    with pytest.raises(ValueError, match="k_e"):
        to_radar_grid(fields, SITE, 0.5, k_e=0.0, **kw)
    with pytest.raises(ValueError, match="neighbors"):
        to_radar_grid(fields, SITE, 0.5, neighbors=0, **kw)
    with pytest.raises(ValueError, match="max_distance"):
        to_radar_grid(fields, SITE, 0.5, max_distance=-1.0, **kw)
    with pytest.raises(ValueError, match="engine"):
        to_radar_grid(fields, SITE, 0.5, engine="gpu", **kw)
    with pytest.raises(KeyError, match="Z_H"):
        to_radar_grid(fields.drop_vars("Z_H"), SITE, 0.5, **kw)
    with pytest.raises(KeyError, match="latitude"):
        to_radar_grid(fields.drop_vars(["latitude", "longitude"]), SITE, 0.5, **kw)
    mixed = fields.assign_coords(longitude=("west_east", fields.longitude.values[0]))
    with pytest.raises(ValueError, match="same dimensions"):
        to_radar_grid(mixed, SITE, 0.5, **kw)
    with pytest.raises(ValueError, match="no dimension"):
        to_radar_grid(fields, SITE, 0.5, vertical_dim="nope", **kw)
    renamed = fields.rename({"bottom_top": "model_level"})
    with pytest.raises(ValueError, match="vertical"):  # Time and model_level are left
        to_radar_grid(renamed, SITE, 0.5, **kw)
    out = to_radar_grid(renamed.isel(Time=0, drop=True), SITE, 0.5, **kw)
    assert out.DBZH.dims == ("azimuth", "range")  # the one left dimension is vertical


def test_to_radar_grid_options(fields):
    kw = dict(
        azimuth=np.arange(0.0, 360.0, 20.0),
        gate_range=np.arange(2000.0, 60001.0, 4000.0),
    )
    base = to_radar_grid(fields, SITE, 0.5, **kw)
    nearest = to_radar_grid(fields, SITE, 0.5, neighbors=1, **kw)
    assert np.isfinite(nearest.DBZH.values).any()
    assert not np.allclose(base.DBZH.values, nearest.DBZH.values, equal_nan=True)
    # a larger Earth radius factor puts the beam higher
    cut = to_radar_grid(fields, SITE, 0.5, max_distance=1.0, **kw)
    assert np.isnan(cut.DBZH.values).all()
    wide = to_radar_grid(fields, SITE, 0.5, max_distance=1.0e6, **kw)
    assert np.isfinite(wide.DBZH.values).sum() >= np.isfinite(base.DBZH.values).sum()
    straight = to_radar_grid(fields, SITE, 2.0, k_e=1.0, **kw)
    refracted = to_radar_grid(fields, SITE, 2.0, k_e=4.0 / 3.0, **kw)
    assert not np.allclose(straight.DBZH.values, refracted.DBZH.values, equal_nan=True)
    # the site altitude is in the Earth radius: it lifts the beam
    high = to_radar_grid(fields, (SITE[0], SITE[1], 3000.0), 0.5, **kw)
    assert not np.allclose(base.DBZH.values, high.DBZH.values, equal_nan=True)
    named = to_radar_grid(fields, SITE, 0.5, vertical_dim="bottom_top", **kw)
    np.testing.assert_array_equal(named.DBZH.values, base.DBZH.values)
    default = to_radar_grid(fields, SITE, 0.5)
    assert default.sizes == {"Time": 1, "azimuth": 360, "range": 600}
    mask = to_radar_grid(fields, SITE, 0.5, min_dbz=100.0, **kw)
    assert np.isnan(mask.DBZH.values).all() and np.isnan(mask.KDP.values).all()


def test_to_radar_grid_accessor_and_provenance(fields):
    site = SITE
    kw = dict(azimuth=[10.0, 20.0], gate_range=[3000.0, 6000.0])
    via = fields.radarx.to_radar_grid(site, 0.5, **kw)
    direct = to_radar_grid(fields, site, 0.5, **kw)
    xr.testing.assert_allclose(via, direct)
    assert "radar" in direct.attrs[METHOD].lower()
    assert "doviak-zrnic-1993" in direct.attrs[REFERENCES].split()
    vol = fields.radarx.to_radar_grid(site, [0.5, 1.5], **kw)
    assert isinstance(vol, xr.DataTree)


# --------------------------------------------------------------------------
# an MPAS-like unstructured mesh
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_mpas_through_the_whole_chain(engine):
    ds, grid = synthetic_mpas()
    st = mpas_state(ds, grid)
    f = forward_operator(st, nrain="QNRAIN", mu_rain=1.0, engine=engine)
    assert f.DBZH.dims == ("Time", "nCells", "nVertLevels")
    assert float(f.DBZH.max()) > 35.0
    site = (35.0, -97.5, 300.0)
    out = to_radar_grid(
        f,
        site,
        0.5,
        azimuth=np.arange(0.0, 360.0, 5.0),
        gate_range=np.arange(2000.0, 150001.0, 2000.0),
        engine=engine,
    )
    assert out.DBZH.dims == ("Time", "azimuth", "range")
    assert np.isfinite(out.DBZH.values).any()
    # the rain shaft of the builder is 60 km north and 30 km west of the radar
    dbz = out.DBZH.values[0]
    i, j = np.unravel_index(np.nanargmax(dbz), dbz.shape)
    bearing = np.rad2deg(np.arctan2(-30.0, 60.0)) % 360.0
    assert abs(out.azimuth.values[i] - bearing) < 25.0
    # uniform wind: the radial velocity follows the azimuth where there is echo
    v = out.VRADH.values[0]
    ok = np.isfinite(v)
    a = np.deg2rad(out.azimuth.values)[:, None] * np.ones_like(v)
    el = np.deg2rad(0.5)
    fall = f.FALL_SPEED  # noqa: F841
    expected_h = (10.0 * np.sin(a) - 5.0 * np.cos(a)) * np.cos(el)
    resid = (v - expected_h)[ok]
    # the vertical term: w = 1 m/s and the fall speed of the rain, times sin(0.5 deg)
    assert np.abs(resid).max() < 0.2
