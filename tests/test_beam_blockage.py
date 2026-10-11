#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the beam blockage and the terrain clutter flag
========================================================

The truth is written here from the definitions, not from the module: terrain
with a known height at a known range (a flat plateau, a ring ridge around the
radar), the beam height from xradar's georeferencing, and the blocked part of
the beam cross section by numerical integration of the circle (uniform beam,
Bech et al. 2003, Eq. A2) or of the Gaussian power pattern. The compiled
kernel and the NumPy implementation must agree to rounding.
"""

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS, Geod
from scipy import integrate
from scipy.interpolate import RegularGridInterpolator
from xradar.georeference import antenna_to_cartesian

import radarx  # noqa: F401
from radarx._provenance import HISTORY, METHOD, REFERENCES, load_registry
from radarx.fundamentals.constants import EARTH_RADIUS
from radarx.retrieve import beam_blockage

bbmod = __import__("radarx.retrieve.beam_blockage", fromlist=["x"])

ENGINES = ["numpy"] + (["compiled"] if bbmod.HAS_COMPILED_KERNEL else [])
compiled_only = pytest.mark.skipif(
    not bbmod.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)
LAT0, LON0, ALT0 = 30.0, 78.0, 500.0
BEAM = 1.0  # degrees
AEQD = CRS.from_proj4(
    f"+proj=aeqd +lat_0={LAT0} +lon_0={LON0} +x_0=0 +y_0=0 +R={EARTH_RADIUS}"
)


# --------------------------------------------------------------------------
# fixtures written from the definitions
# --------------------------------------------------------------------------


def make_sweep(
    nray=24, ngate=200, dr=500.0, elevation=0.5, altitude=ALT0, beam=BEAM, ray="azimuth"
):
    """A sweep dataset with the geometry in the coordinates."""
    az = (np.arange(nray) + 0.5) * 360.0 / nray
    rng = dr * (np.arange(ngate) + 1)
    coords = {
        "range": rng,
        "elevation": (ray, np.full(nray, elevation)),
        "latitude": LAT0,
        "longitude": LON0,
        "altitude": altitude,
    }
    # rays numbered by time: the azimuth is a plain coordinate that can hold NaN
    coords["azimuth"] = az if ray == "azimuth" else (ray, az)
    if ray != "azimuth":
        coords[ray] = np.arange(nray)
    ds = xr.Dataset(
        {"DBZH": ((ray, "range"), np.zeros((nray, ngate)), {"units": "dBZ"})},
        coords=coords,
    )
    if beam is not None:
        ds["radar_beam_width_v"] = beam
    return ds


def geometry(rng, elevation, altitude=ALT0, ke=None):
    """Ground distance and beam height of gates from xradar (independent code)."""
    x, y, z = antenna_to_cartesian(
        rng, 0.0, elevation, site_altitude=altitude, effective_radius_fraction=ke
    )
    return np.hypot(x, y), z


def aeqd_dem(func, extent=70e3, res=100.0, crs=True):
    """A DEM in metres around the radar, ``func(distance)`` high, with its CRS."""
    x = np.arange(-extent, extent + res, res)
    dem = xr.DataArray(
        func(np.hypot(x[None, :], x[:, None])),
        dims=("y", "x"),
        coords={"y": x, "x": x},
    )
    if crs:
        dem.attrs["crs"] = AEQD.to_wkt()
    return dem


def geographic_dem(func, half=1.0, res=0.002):  # about 220 m cells
    """A DEM in degrees; ``func`` of the great circle distance to the radar (m)."""
    lat = np.arange(LAT0 - half, LAT0 + half + res / 2, res)
    lon = np.arange(LON0 - half, LON0 + half + res / 2, res)
    la, lo = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    p0, l0 = np.deg2rad(LAT0), np.deg2rad(LON0)
    a = (
        np.sin((la - p0) / 2) ** 2
        + np.cos(p0) * np.cos(la) * np.sin((lo - l0) / 2) ** 2
    )
    distance = 2 * EARTH_RADIUS * np.arcsin(np.sqrt(a))
    return xr.DataArray(
        func(distance),
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
    )


def bell(d0=20e3, half_width=3e3, peak=800.0):
    """Smooth ring ridge: a raised cosine of the distance from the radar."""

    def height(d):
        d = np.asarray(d, dtype=float)
        u = np.clip((d - d0) / half_width, -1.0, 1.0)
        return peak * np.cos(0.5 * np.pi * u) ** 2

    return height


def uniform_truth(y, a):
    """Blocked part of a uniform circle (radius a) below the height y by quadrature."""
    if y <= -a:
        return 0.0
    if y >= a:
        return 1.0
    area, _ = integrate.quad(lambda u: 2.0 * np.sqrt(a * a - u * u), -a, y)
    return area / (np.pi * a * a)


def gaussian_truth(y, a, two_way=False):
    """Part of a 2-D Gaussian power pattern (half power at the radius a) below y."""
    sigma = a / np.sqrt(2.0 * np.log(2.0)) / (np.sqrt(2.0) if two_way else 1.0)
    lim = 14.0 * sigma
    if y <= -lim:
        return 0.0
    top = min(y, lim)
    value, _ = integrate.dblquad(
        lambda u, v: np.exp(-(u * u + v * v) / (2.0 * sigma**2))
        / (2.0 * np.pi * sigma**2),
        -lim,
        top,
        -lim,
        lim,
        epsabs=1e-12,
        epsrel=1e-10,
    )
    return value


def beam_radius(r, beam=BEAM):
    return r * np.deg2rad(beam) / 2.0


def reference_ray(rng, elevation, terrain, truth, altitude=ALT0, beam=BEAM, ke=None):
    """PBB and CBB of one ray from the definitions: terrain(distance) analytic."""
    distance, height = geometry(rng, elevation, altitude, ke)
    pbb = np.array(
        [
            truth(terrain(d) - h, beam_radius(r, beam))
            for d, h, r in zip(distance, height, rng)
        ]
    )
    return pbb, np.maximum.accumulate(pbb), terrain(distance), height


# --------------------------------------------------------------------------
# the formula: grazing, above, below
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("pattern", ["uniform", "gaussian"])
def test_plateau_at_the_beam_centre_grazing_above_and_below(engine, pattern):
    """A plateau at a known height: PBB 0.5 at y = 0, 0 above and 1 below."""
    sweep = make_sweep(nray=4, ngate=60)
    k = 39
    r = float(sweep.range[k])
    _, h = geometry(r, 0.5)
    a = beam_radius(r)
    truth = uniform_truth if pattern == "uniform" else gaussian_truth
    for offset in (-10 * a, -a, -0.5 * a, 0.0, 0.5 * a, a, 10 * a):
        dem = aeqd_dem(lambda d: np.full_like(d, h + offset), extent=40e3, res=500.0)
        out = beam_blockage(sweep, dem, pattern=pattern, engine=engine)
        got = out.PBB.values[:, k]
        np.testing.assert_allclose(got, truth(offset, a), atol=2e-6)
        np.testing.assert_allclose(
            out.TERRAIN_HEIGHT.values[:, k], h + offset, atol=1e-2
        )
    # the three cases of the paper
    grazing = aeqd_dem(lambda d: np.full_like(d, h), extent=40e3, res=500.0)
    out = beam_blockage(sweep, grazing, pattern=pattern, engine=engine)
    np.testing.assert_allclose(out.PBB.values[:, k], 0.5, atol=1e-6)
    assert np.all(out.TERRAIN_CLUTTER.values[:, k] == 1)  # the centre is not above it


@pytest.mark.parametrize("engine", ENGINES)
def test_uniform_beam_is_zero_and_one_at_the_edges_exactly(engine):
    sweep = make_sweep(nray=2, ngate=50)
    k = 30
    r = float(sweep.range[k])
    _, h = geometry(r, 0.5)
    a = beam_radius(r)
    values = {}
    for name, offset in (("above", -a), ("below", a), ("far above", -5 * a)):
        dem = aeqd_dem(lambda d: np.full_like(d, h + offset), extent=40e3, res=500.0)
        out = beam_blockage(sweep, dem, pattern="uniform", engine=engine)
        values[name] = float(out.PBB.values[0, k])
    assert values["above"] == pytest.approx(0.0, abs=1e-6)
    assert values["far above"] == 0.0
    assert values["below"] == pytest.approx(1.0, abs=1e-6)


def test_gaussian_values_at_the_half_power_radius():
    """Terrain at +-a: the normal distribution function at 1.1774 sigma."""
    from scipy.stats import norm

    for two_way, scale in ((False, 1.0), (True, np.sqrt(2.0))):
        z = np.sqrt(2.0 * np.log(2.0)) * scale
        assert gaussian_truth(1.0, 1.0, two_way) == pytest.approx(norm.cdf(z), abs=1e-9)
        assert gaussian_truth(-1.0, 1.0, two_way) == pytest.approx(
            norm.cdf(-z), abs=1e-9
        )
    assert norm.cdf(np.sqrt(2.0 * np.log(2.0))) == pytest.approx(0.8805, abs=1e-4)


@pytest.mark.parametrize("engine", ENGINES)
def test_two_way_pattern_is_narrower(engine):
    sweep = make_sweep(nray=2, ngate=50)
    k = 30
    r = float(sweep.range[k])
    _, h = geometry(r, 0.5)
    a = beam_radius(r)
    dem = aeqd_dem(lambda d: np.full_like(d, h + 0.5 * a), extent=40e3, res=500.0)
    one = beam_blockage(sweep, dem, engine=engine).PBB.values[0, k]
    two = beam_blockage(sweep, dem, two_way=True, engine=engine).PBB.values[0, k]
    np.testing.assert_allclose(one, gaussian_truth(0.5 * a, a), atol=2e-6)
    np.testing.assert_allclose(two, gaussian_truth(0.5 * a, a, two_way=True), atol=2e-6)
    assert two > one > 0.5


# --------------------------------------------------------------------------
# a ring ridge around the radar against an independent evaluation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("pattern", ["uniform", "gaussian"])
def test_ridge_in_metres_against_the_definition(engine, pattern):
    sweep = make_sweep(nray=8, ngate=140, dr=500.0)
    ridge = bell()
    dem = aeqd_dem(ridge)
    out = beam_blockage(sweep, dem, pattern=pattern, engine=engine)
    truth = uniform_truth if pattern == "uniform" else gaussian_truth
    pbb, cbb, terrain, height = reference_ray(sweep.range.values, 0.5, ridge, truth)
    assert 0.3 < pbb.max() < 0.9  # a partial block, not 0 or 1
    for q in range(sweep.sizes["azimuth"]):
        np.testing.assert_allclose(out.PBB.values[q], pbb, atol=3e-3)
        np.testing.assert_allclose(out.CBB.values[q], cbb, atol=3e-3)
        np.testing.assert_allclose(out.TERRAIN_HEIGHT.values[q], terrain, atol=0.5)
    # behind the ridge the terrain is low again but the shadow stays
    behind = sweep.range.values > 30e3
    assert out.PBB.values[0][behind].max() < 1e-2  # the Gaussian tail
    assert out.CBB.values[0][behind].min() > 0.3


@pytest.mark.parametrize("engine", ENGINES)
def test_ridge_in_degrees_against_the_definition(engine):
    sweep = make_sweep(nray=6, ngate=100)
    ridge = bell()
    dem = geographic_dem(ridge, half=0.8, res=0.001)  # about 110 m cells
    out = beam_blockage(sweep, dem, engine=engine)
    pbb, cbb, terrain, _ = reference_ray(sweep.range.values, 0.5, ridge, gaussian_truth)
    for q in range(sweep.sizes["azimuth"]):
        np.testing.assert_allclose(out.PBB.values[q], pbb, atol=1e-2)
        np.testing.assert_allclose(out.CBB.values[q], cbb, atol=1e-2)
        np.testing.assert_allclose(out.TERRAIN_HEIGHT.values[q], terrain, atol=1.0)


@pytest.mark.parametrize("engine", ENGINES)
def test_effective_earth_radius_factor(engine):
    sweep = make_sweep(nray=2, ngate=100)
    ridge = bell(peak=400.0)
    dem = aeqd_dem(ridge)
    for ke in (1.0, 4.0 / 3.0, 1.6):
        out = beam_blockage(sweep, dem, k_e=ke, engine=engine)
        pbb, cbb, _, _ = reference_ray(
            sweep.range.values, 0.5, ridge, gaussian_truth, ke=ke
        )
        np.testing.assert_allclose(out.PBB.values[0], pbb, atol=3e-3)
        np.testing.assert_allclose(out.CBB.values[0], cbb, atol=3e-3)


@pytest.mark.parametrize("engine", ENGINES)
def test_site_altitude_is_part_of_the_earth_radius(engine):
    for altitude in (0.0, 2000.0):
        sweep = make_sweep(nray=2, ngate=100, altitude=altitude)
        ridge = bell(peak=2500.0)
        out = beam_blockage(sweep, aeqd_dem(ridge), engine=engine)
        pbb, _, _, _ = reference_ray(
            sweep.range.values, 0.5, ridge, gaussian_truth, altitude=altitude
        )
        np.testing.assert_allclose(out.PBB.values[0], pbb, atol=3e-3)


@pytest.mark.parametrize("engine", ENGINES)
def test_elevation_and_beamwidth_change_the_blockage(engine):
    ridge = bell(peak=600.0)
    dem = aeqd_dem(ridge)
    cbb = {}
    for el in (0.2, 0.5, 1.0, 2.0):
        sweep = make_sweep(nray=2, ngate=100, elevation=el, beam=1.5)
        out = beam_blockage(sweep, dem, engine=engine)
        _, ref, _, _ = reference_ray(
            sweep.range.values, el, ridge, gaussian_truth, beam=1.5
        )
        np.testing.assert_allclose(out.CBB.values[0], ref, atol=3e-3)
        cbb[el] = out.CBB.values[0, -1]
    assert cbb[0.2] > cbb[0.5] > cbb[1.0] >= cbb[2.0]
    assert cbb[2.0] == pytest.approx(0.0, abs=1e-2)  # the Gaussian tail
    # a wider beam is blocked sooner at the same elevation
    wide = beam_blockage(make_sweep(nray=2, ngate=100, elevation=2.0, beam=4.0), dem)
    assert float(wide.CBB.max()) > cbb[2.0]


# --------------------------------------------------------------------------
# invariants
# --------------------------------------------------------------------------


def random_terrain(seed=1, n=401, half=2.0, scale=900.0):
    """A smooth random DEM in degrees (filtered noise, 0 to about ``scale`` m)."""
    rnd = np.random.default_rng(seed)
    noise = rnd.standard_normal((n, n))
    k = np.fft.fftfreq(n)
    kr = np.hypot(k[:, None], k[None, :])
    field = np.fft.ifft2(np.fft.fft2(noise) * np.exp(-((kr / 0.02) ** 2))).real
    field = (field - field.min()) / (field.max() - field.min())
    lat = np.linspace(LAT0 - half, LAT0 + half, n)
    lon = np.linspace(LON0 - half, LON0 + half, n)
    return xr.DataArray(
        scale * field**2,
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
    )


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("pattern", ["uniform", "gaussian"])
def test_invariants_on_random_terrain(engine, pattern):
    dem = random_terrain(scale=900.0)
    mean = []
    for el in (0.0, 0.3, 1.0):
        sweep = make_sweep(nray=72, ngate=240, dr=500.0, elevation=el, altitude=300.0)
        out = beam_blockage(sweep, dem, pattern=pattern, oversample=2, engine=engine)
        pbb, cbb = out.PBB.values, out.CBB.values
        assert np.isfinite(pbb).all() and np.isfinite(cbb).all()
        assert pbb.min() >= 0.0 and pbb.max() <= 1.0
        assert cbb.min() >= 0.0 and cbb.max() <= 1.0
        assert np.all(cbb >= pbb - 1e-7)
        assert np.all(np.diff(cbb, axis=1) >= -1e-7)
        mean.append(cbb.mean())
        if el == 0.3:  # partial blockage occurs, not only 0 and 1
            assert ((pbb > 0.1) & (pbb < 0.9)).any() and (cbb < 0.05).any()
    assert mean[0] > mean[1] > mean[2]  # a higher beam is less blocked
    # the terrain is the bilinear interpolation of the DEM at the gate position
    geod = Geod(a=EARTH_RADIUS, b=EARTH_RADIUS)
    sweep = make_sweep(nray=36, ngate=100, elevation=0.5, altitude=300.0)
    out = beam_blockage(sweep, dem, pattern=pattern, engine=engine)
    ground, _ = geometry(sweep.range.values, 0.5, 300.0)
    az, s = np.meshgrid(sweep.azimuth.values, ground, indexing="ij")
    lon, lat, _ = geod.fwd(np.full(az.shape, LON0), np.full(az.shape, LAT0), az, s)
    interp = RegularGridInterpolator(
        (dem.latitude.values, dem.longitude.values), dem.values
    )
    expected = interp(np.stack([lat, lon], axis=-1))
    np.testing.assert_allclose(out.TERRAIN_HEIGHT.values, expected, atol=0.05)


@pytest.mark.parametrize("engine", ENGINES)
def test_no_terrain_gives_zero(engine):
    # terrain at sea level, a radar 2 km up: the beam never gets near the ground
    dem = geographic_dem(lambda d: np.zeros_like(d))
    sweep = make_sweep(nray=12, ngate=120, elevation=1.0, altitude=2000.0)
    for pattern in ("uniform", "gaussian"):
        out = beam_blockage(sweep, dem, pattern=pattern, engine=engine)
        assert float(out.PBB.max()) < 1e-9
        assert float(out.CBB.max()) < 1e-9
        assert int(out.TERRAIN_CLUTTER.sum()) == 0
    out = beam_blockage(sweep, dem, pattern="uniform", engine=engine)
    assert float(out.CBB.max()) == 0.0
    # terrain above the beam everywhere: everything is blocked and flagged
    high = geographic_dem(lambda d: np.full_like(d, 9000.0))
    out = beam_blockage(sweep, high, engine=engine)
    assert float(out.PBB.min()) > 1.0 - 1e-9
    assert np.all(out.TERRAIN_CLUTTER.values == 1)


@pytest.mark.parametrize("engine", ENGINES)
def test_cumulative_blockage_keeps_the_shadow_of_the_first_block(engine):
    sweep = make_sweep(nray=4, ngate=120)
    first = bell(d0=15e3, peak=700.0)
    second = bell(d0=40e3, peak=300.0)
    ridge = lambda d: np.maximum(first(d), second(d))  # noqa: E731
    out = beam_blockage(sweep, aeqd_dem(ridge), engine=engine)
    pbb, cbb = out.PBB.values[0], out.CBB.values[0]
    assert cbb.max() > 0.5
    k1 = int(np.argmax(pbb[: np.searchsorted(sweep.range.values, 28e3)]))
    assert cbb[k1:].min() >= pbb[k1] - 1e-6
    assert np.all(np.diff(cbb) >= -1e-7)
    # the second ridge has its own PBB, which stays below the first maximum
    assert pbb[np.searchsorted(sweep.range.values, 40e3) :].max() < cbb.max()


@pytest.mark.parametrize("engine", ENGINES)
def test_oversampling_finds_a_ridge_between_the_gates(engine):
    sweep = make_sweep(nray=2, ngate=40, dr=2000.0)
    ridge = bell(d0=21e3, half_width=400.0, peak=1500.0)
    dem = aeqd_dem(ridge, res=25.0, extent=45e3)
    plain = beam_blockage(sweep, dem, pattern="uniform", engine=engine)
    fine = beam_blockage(sweep, dem, pattern="uniform", oversample=8, engine=engine)
    assert float(plain.CBB.max()) == 0.0
    assert float(fine.CBB.max()) > 0.9
    # the PBB of a gate is still the value at its centre
    np.testing.assert_array_equal(plain.PBB.values, fine.PBB.values)
    # the samples between the gates only add to the cumulative value
    assert np.all(fine.CBB.values >= plain.CBB.values - 1e-7)
    # the first gate is sampled towards the radar too (skipping non-positive ranges)
    first = beam_blockage(sweep, dem, oversample=5, engine=engine)
    assert np.isfinite(first.CBB.values[:, 0]).all()
    assert np.all(first.CBB.values[:, 0] == first.PBB.values[:, 0])


@pytest.mark.parametrize("engine", ENGINES)
def test_terrain_clutter_margin(engine):
    sweep = make_sweep(nray=2, ngate=60)
    k = 39
    r = float(sweep.range[k])
    _, h = geometry(r, 0.5)
    for offset, margin, expect in (
        (-50.0, 0.0, 0),  # 50 m below the beam centre
        (-50.0, 49.0, 0),
        (-50.0, 51.0, 1),
        (30.0, 0.0, 1),  # centre inside the terrain
    ):
        dem = aeqd_dem(lambda d: np.full_like(d, h + offset), extent=40e3, res=500.0)
        out = beam_blockage(sweep, dem, clutter_margin=margin, engine=engine)
        assert out.TERRAIN_CLUTTER.values[0, k] == expect
    flag = out.TERRAIN_CLUTTER
    assert flag.dtype == np.int8 and "_FillValue" not in flag.attrs
    assert list(flag.attrs["flag_values"]) == [0, 1]
    # every gate with a beam centre in the terrain has a PBB of at least 0.5
    dem = aeqd_dem(bell(peak=1200.0))
    out = beam_blockage(sweep, dem, engine=engine)
    clutter = out.TERRAIN_CLUTTER.values == 1
    assert clutter.any() and (out.PBB.values[clutter] >= 0.5 - 1e-6).all()


@pytest.mark.parametrize("engine", ENGINES)
def test_missing_terrain_is_nan_and_the_shadow_continues(engine):
    sweep = make_sweep(nray=4, ngate=130)  # rays at 45, 135, 225 and 315 degrees
    ridge = bell(d0=15e3, peak=900.0)
    dem = aeqd_dem(ridge, extent=40e3)
    holed = dem.copy()
    holed.loc[{"y": slice(17e3, 20.5e3), "x": slice(17e3, 20.5e3)}] = np.nan
    out = beam_blockage(sweep, holed, engine=engine)
    rng = sweep.range.values
    in_hole = (rng > 24.5e3) & (rng < 28.5e3)
    assert np.isnan(out.PBB.values[0, in_hole]).all()
    assert np.isnan(out.TERRAIN_HEIGHT.values[0, in_hole]).all()
    assert (out.TERRAIN_CLUTTER.values[0, in_hole] == 0).all()
    assert np.isfinite(out.CBB.values[0, in_hole]).all()  # keeps the earlier maximum
    assert np.isfinite(out.PBB.values[1, in_hole]).all()  # other rays have terrain
    # beyond the DEM nothing is known; the maximum so far is kept
    beyond = rng > 57e3
    assert beyond.any() and np.isnan(out.PBB.values[:, beyond]).all()
    assert np.isfinite(out.CBB.values[:, beyond]).all()
    # a DEM that starts after the first gates: CBB is NaN until terrain is known
    shifted = xr.DataArray(
        np.full((20, 20), 100.0),
        dims=("y", "x"),
        coords={"y": np.linspace(10e3, 40e3, 20), "x": np.linspace(-40e3, -10e3, 20)},
        attrs=dem.attrs,
    )
    west = beam_blockage(sweep, shifted, engine=engine)
    assert np.isnan(west.CBB.values[:, 0]).all()
    assert (
        np.isfinite(west.CBB.values[3]).sum() > 10
    )  # the ray at 315 degrees crosses it
    assert (west.TERRAIN_CLUTTER.values[np.isnan(west.PBB.values)] == 0).all()
    assert np.isnan(west.PBB.values[0]).all() and np.isnan(west.CBB.values[0]).all()


@pytest.mark.parametrize("engine", ENGINES)
def test_rays_without_a_direction_are_missing(engine):
    sweep = make_sweep(nray=6, ngate=50, ray="time")
    sweep["azimuth"].values[2] = np.nan
    sweep["elevation"].values[4] = np.nan
    out = beam_blockage(sweep, aeqd_dem(bell()), engine=engine)
    for ray in (2, 4):
        assert np.isnan(out.PBB.values[ray]).all()
        assert np.isnan(out.CBB.values[ray]).all()
        assert (out.TERRAIN_CLUTTER.values[ray] == 0).all()
    assert np.isfinite(out.PBB.values[[0, 1, 3, 5]]).all()


# --------------------------------------------------------------------------
# the kernel against its NumPy implementation
# --------------------------------------------------------------------------


def kernel_inputs(geographic, seed=3):
    rnd = np.random.default_rng(seed)
    rng = 300.0 + 400.0 * np.arange(180)
    az = rnd.uniform(0.0, 360.0, 50)
    el = rnd.uniform(-0.5, 3.0, 50)
    az[3], el[7] = np.nan, np.nan
    if geographic:
        dem = random_terrain(seed, n=201, half=1.5)
        ya, xa = dem.latitude.values, dem.longitude.values
        site = (LAT0, LON0)
    else:
        ya = xa = np.arange(-80e3, 80e3 + 1, 800.0)
        field = rnd.uniform(0.0, 1200.0, (ya.size, xa.size))
        dem = xr.DataArray(field, dims=("y", "x"), coords={"y": ya, "x": xa})
        site = (0.0, 0.0)
    values = np.ascontiguousarray(dem.values)
    values[10:14, 20:30] = np.nan
    return rng, az, el, values, ya, xa, site


@compiled_only
@pytest.mark.parametrize("geographic", [True, False])
@pytest.mark.parametrize("pattern", [0, 1, 2])
@pytest.mark.parametrize("oversample", [1, 3])
def test_kernel_agrees_with_numpy(geographic, pattern, oversample):
    rng, az, el, values, ya, xa, site = kernel_inputs(geographic)
    args = (ALT0, 4.0 / 3.0, EARTH_RADIUS, 1.2, pattern, oversample, 30.0)
    got = bbmod._beam_blockage.blockage(
        rng, az, el, values, ya, xa, geographic, site[0], site[1], *args, 0
    )
    geo = {
        "geographic": geographic,
        "site_a": site[0],
        "site_b": site[1],
        "altitude": ALT0,
        "earth_radius": EARTH_RADIUS,
        "lon_center": 0.5 * (xa[0] + xa[-1]),
    }
    ref = bbmod._blockage_numpy(
        rng, az, el, values, ya, xa, geo, 4.0 / 3.0, 1.2, pattern, oversample, 30.0
    )
    for g, r in zip(got, ref):
        assert np.array_equal(np.isnan(g), np.isnan(r))
        np.testing.assert_allclose(g, r, rtol=1e-9, atol=1e-12)
    assert got[1].shape == (az.size, rng.size) and got[3].dtype == np.uint8
    assert np.isnan(got[0]).any() and np.isfinite(got[0]).any()
    assert (got[3] == 1).any() and (got[3] == 0).any()


@compiled_only
def test_kernel_threads_do_not_change_the_result():
    rng, az, el, values, ya, xa, site = kernel_inputs(True)
    az = np.tile(az, 8)
    el = np.tile(el, 8)
    args = (
        values,
        ya,
        xa,
        True,
        site[0],
        site[1],
        ALT0,
        4 / 3,
        EARTH_RADIUS,
        1.0,
        0,
        2,
        0.0,
    )
    one = bbmod._beam_blockage.blockage(rng, az, el, *args, 1)
    many = bbmod._beam_blockage.blockage(rng, az, el, *args, 4)
    for a, b in zip(one, many):
        np.testing.assert_array_equal(a, b)


@compiled_only
def test_kernel_rejects_bad_input():
    rng, az, el, values, ya, xa, site = kernel_inputs(True)
    base = (
        values,
        ya,
        xa,
        True,
        site[0],
        site[1],
        ALT0,
        4 / 3,
        EARTH_RADIUS,
        1.0,
        0,
        1,
        0.0,
    )
    f = bbmod._beam_blockage.blockage
    with pytest.raises(ValueError, match="same length"):
        f(rng, az, el[:-1], *base)
    with pytest.raises(ValueError, match="2-D"):
        f(rng, az, el, values, ya[:-1], xa, *base[3:])
    with pytest.raises(ValueError, match="2-D"):
        f(rng, az, el, values[:1], ya[:1], xa, *base[3:])
    with pytest.raises(ValueError, match="unknown pattern"):
        f(rng, az, el, *base[:9], 1.0, 7, 1, 0.0)
    with pytest.raises(ValueError, match="oversample"):
        f(rng, az, el, *base[:9], 1.0, 0, 0, 0.0)


@pytest.mark.parametrize("engine", ENGINES)
def test_engines_give_the_same_dataset(engine):
    dem = random_terrain()
    sweep = make_sweep(nray=30, ngate=150, elevation=0.4, altitude=200.0)
    ref = beam_blockage(sweep, dem, engine="numpy", oversample=2)
    got = beam_blockage(sweep, dem, engine=engine, oversample=2)
    for name in ref.data_vars:
        np.testing.assert_allclose(
            got[name].values, ref[name].values, rtol=1e-5, atol=1e-6
        )
    assert ref.PBB.dtype == np.float32 and ref.TERRAIN_CLUTTER.dtype == np.int8


def test_compiled_engine_needs_the_kernel(monkeypatch):
    monkeypatch.setattr(bbmod, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError, match="compiled beam blockage kernel"):
        beam_blockage(make_sweep(), aeqd_dem(bell()), engine="compiled")
    out = beam_blockage(
        make_sweep(nray=2, ngate=30), aeqd_dem(bell())
    )  # numpy fallback
    assert "PBB" in out


# --------------------------------------------------------------------------
# DEM formats and coordinates
# --------------------------------------------------------------------------


def test_dem_formats_give_the_same_result():
    sweep = make_sweep(nray=6, ngate=100)
    ridge = bell()
    ref = beam_blockage(sweep, aeqd_dem(ridge, res=50.0))
    # latitude and longitude, names lat and lon, descending latitude, a band dimension
    deg = geographic_dem(ridge)
    renamed = deg.rename(latitude="lat", longitude="lon")
    flipped = deg.isel(latitude=slice(None, None, -1))
    banded = deg.expand_dims(band=[1])
    base = beam_blockage(sweep, deg)
    for dem in (renamed, flipped, banded):
        out = beam_blockage(sweep, dem)
        np.testing.assert_allclose(out.CBB.values, base.CBB.values, atol=1e-6)
    np.testing.assert_allclose(base.CBB.values, ref.CBB.values, atol=2e-2)
    # y and x in degrees with the crs of a rioxarray DataArray (spatial_ref coordinate)
    rio = deg.rename(latitude="y", longitude="x").assign_coords(
        spatial_ref=((), 0, {"crs_wkt": CRS.from_epsg(4326).to_wkt()})
    )
    np.testing.assert_allclose(
        beam_blockage(sweep, rio).CBB.values, base.CBB.values, atol=1e-6
    )
    # a projected DEM with the crs as an argument, a EPSG code, or an attribute
    plain = aeqd_dem(ridge, res=50.0, crs=False)
    for crs in (AEQD, AEQD.to_wkt(), AEQD.to_proj4()):
        out = beam_blockage(sweep, plain, crs=crs)
        np.testing.assert_allclose(out.CBB.values, ref.CBB.values, atol=1e-6)
    with_attr = plain.copy()
    with_attr.attrs["crs"] = AEQD.to_wkt()
    np.testing.assert_allclose(
        beam_blockage(sweep, with_attr).CBB.values, ref.CBB.values, atol=1e-6
    )
    wkt = plain.assign_coords(spatial_ref=((), 0, {"crs_wkt": AEQD.to_wkt()}))
    np.testing.assert_allclose(
        beam_blockage(sweep, wkt).CBB.values, ref.CBB.values, atol=1e-6
    )


def test_dem_in_0_360_longitude_and_western_site():
    # a western-hemisphere radar and a DEM that runs 0 to 360 degrees east
    lon0 = -100.0
    sweep = make_sweep(nray=6, ngate=80)
    sweep["longitude"] = lon0
    ridge = bell()
    base = geographic_dem(ridge).assign_coords(longitude=lambda d: d.longitude)
    west = base.assign_coords(longitude=base.longitude.values - LON0 + lon0)
    east = west.assign_coords(longitude=west.longitude.values % 360.0)
    a = beam_blockage(sweep, west)
    b = beam_blockage(sweep, east)
    np.testing.assert_allclose(a.CBB.values, b.CBB.values, atol=1e-6)
    assert float(a.CBB.max()) > 0.3
    # and the other way round: a site given in 0-360 and a DEM in -180 to 180
    sweep["longitude"] = lon0 % 360.0
    c = beam_blockage(sweep, west)
    np.testing.assert_allclose(a.CBB.values, c.CBB.values, atol=1e-6)


def test_dem_errors():
    sweep = make_sweep(nray=2, ngate=20)
    good = geographic_dem(bell())
    with pytest.raises(TypeError, match="DataArray of terrain"):
        beam_blockage(sweep, good.values)
    with pytest.raises(ValueError, match="two dimensions"):
        beam_blockage(sweep, good.expand_dims(time=[0, 1]))
    with pytest.raises(ValueError, match="latitude and longitude, or y and x"):
        beam_blockage(sweep, good.rename(latitude="a", longitude="b"))
    with pytest.raises(ValueError, match="coordinate reference system"):
        beam_blockage(sweep, aeqd_dem(bell(), crs=False))
    far = good.assign_coords(latitude=good.latitude + 20.0)
    with pytest.raises(ValueError, match="does not reach the radar"):
        beam_blockage(sweep, far)
    with pytest.raises(ValueError, match="does not reach the radar"):
        beam_blockage(
            sweep, aeqd_dem(bell(), extent=1e3).assign_coords(x=lambda d: d.x + 5e5)
        )


def test_dem_is_cut_to_the_reach_of_the_radar():
    # a dask-backed DEM much larger than the radar domain is not loaded as a whole
    pytest.importorskip("dask")
    big = geographic_dem(bell(), half=1.0).chunk({"latitude": 100, "longitude": 100})
    out = beam_blockage(make_sweep(nray=4, ngate=40), big)
    ref = beam_blockage(make_sweep(nray=4, ngate=40), big.compute())
    np.testing.assert_array_equal(out.CBB.values, ref.CBB.values)


# --------------------------------------------------------------------------
# the xarray layer
# --------------------------------------------------------------------------


def test_result_fields_and_attributes():
    sweep = make_sweep(nray=6, ngate=80)
    out = beam_blockage(sweep, geographic_dem(bell()))
    assert set(out.data_vars) == {"PBB", "CBB", "TERRAIN_HEIGHT", "TERRAIN_CLUTTER"}
    for name in out.data_vars:
        assert out[name].dims == ("azimuth", "range")
    for coord in ("azimuth", "range", "elevation", "latitude", "longitude", "altitude"):
        assert coord in out.coords
    assert out.PBB.attrs["units"] == "1" and out.TERRAIN_HEIGHT.attrs["units"] == "m"
    assert out.attrs["beam_blockage_pattern"] == "gaussian"
    assert out.attrs["beam_blockage_beamwidth"] == 1.0
    assert out.attrs["beam_blockage_k_e"] == pytest.approx(4.0 / 3.0)
    two = beam_blockage(sweep, geographic_dem(bell()), two_way=True)
    assert two.attrs["beam_blockage_pattern"] == "gaussian two way"
    # the fields serve as a mask
    kept = sweep["DBZH"].where(out["CBB"] < 0.5)
    assert kept.dims == sweep["DBZH"].dims


def test_validation():
    sweep = make_sweep(nray=2, ngate=20)
    dem = geographic_dem(bell())
    with pytest.raises(TypeError, match="xarray DataArray, Dataset or DataTree"):
        beam_blockage(sweep.DBZH.values, dem)
    with pytest.raises(ValueError, match="pattern must be one of"):
        beam_blockage(sweep, dem, pattern="airy")
    with pytest.raises(ValueError, match="two_way applies"):
        beam_blockage(sweep, dem, pattern="uniform", two_way=True)
    for k_e in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="k_e must be positive"):
            beam_blockage(sweep, dem, k_e=k_e)
    for n in (0, 1.5):
        with pytest.raises(ValueError, match="oversample"):
            beam_blockage(sweep, dem, oversample=n)
    with pytest.raises(ValueError, match="clutter_margin"):
        beam_blockage(sweep, dem, clutter_margin=-1.0)
    with pytest.raises(ValueError, match="engine must be"):
        beam_blockage(sweep, dem, engine="gpu")


def test_beam_width_from_the_data_or_the_argument():
    dem = geographic_dem(bell())
    base = make_sweep(nray=2, ngate=60, beam=None)
    with pytest.raises(ValueError, match="do not give the beam width"):
        beam_blockage(base, dem)
    h_only = base.assign(radar_beam_width_h=1.0)
    v_first = base.assign(radar_beam_width_h=3.0, radar_beam_width_v=1.0)
    ref = beam_blockage(base, dem, beamwidth=1.0)
    np.testing.assert_array_equal(beam_blockage(h_only, dem).CBB, ref.CBB)
    np.testing.assert_array_equal(beam_blockage(v_first, dem).CBB, ref.CBB)
    # the argument wins over the metadata
    np.testing.assert_array_equal(
        beam_blockage(make_sweep(beam=5.0, nray=2, ngate=60), dem, beamwidth=1.0).CBB,
        ref.CBB,
    )
    for bad in (0.0, -1.0, 120.0, np.nan):
        with pytest.raises(ValueError, match="beam width must be between"):
            beam_blockage(base, dem, beamwidth=bad)
    nan_width = base.assign(radar_beam_width_v=np.nan, radar_beam_width_h=1.0)
    np.testing.assert_array_equal(beam_blockage(nan_width, dem).CBB, ref.CBB)


def test_radar_site_from_the_data_or_the_argument():
    dem = geographic_dem(bell())
    sweep = make_sweep(nray=2, ngate=60)
    bare = sweep.drop_vars(["latitude", "longitude", "altitude"])
    with pytest.raises(ValueError, match="radar site"):
        beam_blockage(bare, dem)
    ref = beam_blockage(sweep, dem)
    got = beam_blockage(bare, dem, site=(LAT0, LON0, ALT0))
    np.testing.assert_array_equal(got.CBB, ref.CBB)
    # the argument wins over the data
    moved = beam_blockage(sweep, dem, site=(LAT0, LON0, ALT0 + 3000.0))
    assert float(moved.CBB.max()) < float(ref.CBB.max())


def test_geometry_of_the_sweep_is_checked():
    dem = geographic_dem(bell())
    sweep = make_sweep(nray=4, ngate=30)
    with pytest.raises(ValueError, match="'range' coordinate"):
        beam_blockage(
            sweep.drop_vars("range")
            .assign(range=("range", np.arange(30)))
            .drop_indexes("range")
            .reset_coords("range", drop=True),
            dem,
        )
    with pytest.raises(ValueError, match="'azimuth' coordinate"):
        beam_blockage(sweep.drop_vars("azimuth"), dem)
    two_d = sweep.assign_coords(azimuth=(("azimuth", "range"), np.zeros((4, 30))))
    with pytest.raises(ValueError):
        beam_blockage(two_d, dem)
    for bad in (-1.0, 0.0, np.nan):
        r = sweep.range.values.copy()
        r[3] = bad
        with pytest.raises(ValueError, match="range must be finite and positive"):
            beam_blockage(sweep.assign_coords(range=r), dem)
    with pytest.raises(ValueError, match="'elevation' .* or a 'sweep_fixed_angle'"):
        beam_blockage(sweep.drop_vars("elevation"), dem)
    odd = sweep.assign_coords(elevation=("range", np.zeros(30)))
    with pytest.raises(ValueError, match="scalar or have one value per ray"):
        beam_blockage(odd, dem)


def test_elevation_from_a_scalar_or_the_fixed_angle():
    dem = aeqd_dem(bell())
    sweep = make_sweep(nray=3, ngate=80, elevation=0.7)
    ref = beam_blockage(sweep, dem)
    scalar = sweep.drop_vars("elevation").assign_coords(elevation=0.7)
    fixed = sweep.drop_vars("elevation").assign(sweep_fixed_angle=0.7)
    np.testing.assert_array_equal(beam_blockage(scalar, dem).CBB, ref.CBB)
    np.testing.assert_array_equal(beam_blockage(fixed, dem).CBB, ref.CBB)


def test_dataarray_gives_the_geometry():
    dem = geographic_dem(bell())
    sweep = make_sweep(nray=6, ngate=80)
    ref = beam_blockage(sweep, dem)
    field = sweep["DBZH"]
    out = beam_blockage(field, dem, beamwidth=1.0)
    np.testing.assert_array_equal(out.CBB, ref.CBB)
    assert isinstance(out, xr.Dataset)
    # the DataArray carries the site as scalar coordinates and its own name is no field
    out = field.rename("PBB").radarx.beam_blockage(dem, beamwidth=1.0)
    np.testing.assert_array_equal(out.CBB, ref.CBB)
    assert out.PBB.dims == ("azimuth", "range")


def test_ray_dimension_need_not_be_the_azimuth():
    dem = geographic_dem(bell())
    ref = beam_blockage(make_sweep(nray=6, ngate=80), dem)
    out = beam_blockage(make_sweep(nray=6, ngate=80, ray="time"), dem)
    assert out.CBB.dims == ("time", "range")
    np.testing.assert_array_equal(out.CBB.values, ref.CBB.values)
    assert "azimuth" in out.coords and "time" in out.coords


def volume(nsweep=2):
    sweeps = {}
    for k, el in enumerate((0.5, 1.5)[:nsweep]):
        ds = make_sweep(nray=6, ngate=80, elevation=el, beam=None)
        sweeps[f"sweep_{k}"] = ds.drop_vars(["latitude", "longitude", "altitude"])
    root = xr.Dataset(coords={"latitude": LAT0, "longitude": LON0, "altitude": ALT0})
    params = xr.Dataset({"radar_beam_width_v": 1.0, "radar_beam_width_h": 0.9})
    return xr.DataTree.from_dict({"/": root, "radar_parameters": params, **sweeps})


def test_datatree_volume():
    dem = geographic_dem(bell(peak=1000.0))
    tree = volume()
    out = beam_blockage(tree, dem)
    assert isinstance(out, xr.DataTree)
    assert sorted(out.children) == ["sweep_0", "sweep_1"]
    for k, el in enumerate((0.5, 1.5)):
        sweep = make_sweep(nray=6, ngate=80, elevation=el)
        ref = beam_blockage(sweep, dem)
        node = out[f"sweep_{k}"].to_dataset()
        np.testing.assert_array_equal(node.CBB.values, ref.CBB.values)
        assert "PBB" in node and "TERRAIN_CLUTTER" in node
    # the higher sweep is less blocked
    assert float(out["sweep_0"].CBB.max()) > float(out["sweep_1"].CBB.max())
    assert out.latitude == LAT0  # the root of the input
    # the beam width of the root when there is no radar_parameters group
    root = tree.to_dataset().assign(radar_beam_width_v=1.0)
    no_params = xr.DataTree.from_dict(
        {"/": root, "sweep_0": tree["sweep_0"].to_dataset()}
    )
    np.testing.assert_array_equal(
        beam_blockage(no_params, dem)["sweep_0"].CBB.values, out["sweep_0"].CBB.values
    )
    # a single sweep node is one sweep
    one = beam_blockage(tree["sweep_0"], dem, beamwidth=1.0)
    assert isinstance(one, xr.Dataset)
    np.testing.assert_array_equal(one.CBB.values, out["sweep_0"].CBB.values)
    # the products merge into the volume
    merged = tree.radarx.assign(out)
    assert "CBB" in merged["sweep_1"].data_vars


def test_accessors():
    dem = geographic_dem(bell())
    sweep = make_sweep(nray=6, ngate=80)
    ref = beam_blockage(sweep, dem)
    np.testing.assert_array_equal(sweep.radarx.beam_blockage(dem).CBB, ref.CBB)
    tree = volume()
    out = tree.radarx.beam_blockage(dem)
    assert "CBB" in out["sweep_0"].data_vars


def test_dask_stays_lazy_and_agrees():
    dask = pytest.importorskip("dask")  # noqa: F841
    dem = random_terrain()
    sweep = make_sweep(nray=60, ngate=120, elevation=0.4, altitude=200.0)
    ref = beam_blockage(sweep, dem)
    lazy = sweep.chunk({"azimuth": 20})
    out = beam_blockage(lazy, dem)
    assert out.CBB.chunks is not None
    assert out.CBB.chunks[0] == (20, 20, 20)
    assert out.TERRAIN_CLUTTER.dtype == np.int8
    for name in ref.data_vars:
        np.testing.assert_array_equal(out[name].compute().values, ref[name].values)
    # the DataArray of a lazy field gives a lazy result as well
    out = beam_blockage(lazy["DBZH"], dem, beamwidth=1.0)
    assert out.PBB.chunks is not None
    np.testing.assert_array_equal(out.PBB.compute().values, ref.PBB.values)


def test_inputs_are_not_changed():
    dem = geographic_dem(bell())
    sweep = make_sweep(nray=6, ngate=80)
    sweep["range"].attrs.update(units="m", long_name="range")
    sweep["azimuth"].attrs.update(units="degrees")
    before, dem_before = sweep.copy(deep=True), dem.copy(deep=True)
    out = beam_blockage(sweep, dem)
    xr.testing.assert_identical(sweep, before)
    xr.testing.assert_identical(dem, dem_before)
    for coord in ("azimuth", "range", "elevation"):
        assert out[coord].variable is not sweep[coord].variable
    out.coords["azimuth"].attrs["touched"] = 1
    assert "touched" not in sweep["azimuth"].attrs
    tree = volume()
    before = tree.copy(deep=True)
    beam_blockage(tree, dem)
    xr.testing.assert_identical(tree.to_dataset(), before.to_dataset())
    xr.testing.assert_identical(
        tree["sweep_0"].to_dataset(), before["sweep_0"].to_dataset()
    )


def test_provenance_and_references():
    dem = geographic_dem(bell())
    out = beam_blockage(make_sweep(nray=4, ngate=40), dem)
    refs = out.attrs[REFERENCES].split()
    assert "10.1175/1520-0426(2003)020<0845:TSOSPW>2.0.CO;2" in refs
    assert "10.1175/2008JTECHA1133.1" in refs
    assert "doviak-zrnic-1993" in refs
    registry = load_registry()
    assert all(ref in registry for ref in refs)
    assert registry["10.1175/1520-0426(2003)020<0845:TSOSPW>2.0.CO;2"]["year"] == 2003
    assert "Bech et al. (2003)" in out.attrs[METHOD]
    assert out.attrs[HISTORY].startswith("beam_blockage(")
    assert "k_e" not in out.attrs[HISTORY]  # the default is not repeated
    out = beam_blockage(make_sweep(nray=4, ngate=40), dem, k_e=1.2, oversample=2)
    assert "k_e=1.2" in out.attrs[HISTORY] and "oversample=2" in out.attrs[HISTORY]
    assert out.CBB.attrs["radarx_references"] == out.attrs[REFERENCES]
    tree = beam_blockage(volume(), dem)
    assert METHOD in tree["sweep_0"].attrs
    doc = bbmod.beam_blockage.__doc__
    assert "not checked against the paper" in bbmod.__doc__ and "Bech" in doc


def test_netcdf_round_trip(tmp_path):
    out = beam_blockage(make_sweep(nray=4, ngate=40), geographic_dem(bell()))
    out.to_netcdf(tmp_path / "bb.nc")
    with xr.open_dataset(tmp_path / "bb.nc") as back:
        xr.testing.assert_identical(back.load(), out)
        assert back.TERRAIN_CLUTTER.dtype == np.int8
