#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for the LLSD azimuthal shear and radial divergence."""

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401  registers the accessors
from radarx.retrieve import azimuthal_shear, llsd, radial_divergence
from radarx.retrieve import shear as shear_mod

needs_kernel = pytest.mark.skipif(
    not shear_mod.HAS_COMPILED_KERNEL, reason="compiled LLSD kernel not built"
)
ENGINES = ["numpy", pytest.param("compiled", marks=needs_kernel)]

OMEGA = 0.01  # solid-body angular velocity of the vortex core [s-1]
K = 0.002  # convergence rate of the core [s-1]
CORE = 4000.0  # core radius [m]


def rankine(azimuth, rng, centre=(30e3, 40e3)):
    """
    Radial velocity of a Rankine vortex with Rankine-type convergence.

    Inside the core the flow is solid-body rotation ``OMEGA`` plus uniform
    convergence ``K``; outside, both decay as 1/distance. Returns the radial
    velocity and, inside the core, the analytic azimuthal shear
    ``OMEGA + (U . e_theta) / r`` and radial divergence ``-K`` (NaN outside).
    """
    A, R = np.meshgrid(np.radians(azimuth), rng, indexing="ij")
    x, y = R * np.sin(A), R * np.cos(A)
    dx, dy = x - centre[0], y - centre[1]
    rho = np.maximum(np.hypot(dx, dy), 1e-9)
    inside = rho < CORE
    vt = np.where(inside, OMEGA * rho, OMEGA * CORE**2 / rho)
    vr = np.where(inside, -K * rho, -K * CORE**2 / rho)
    u = -vt * dy / rho + vr * dx / rho
    v = vt * dx / rho + vr * dy / rho
    velocity = u * np.sin(A) + v * np.cos(A)
    # unit vector of increasing azimuth (clockwise from north)
    tangential = u * np.cos(A) - v * np.sin(A)
    true_shear = np.where(inside, OMEGA + tangential / R, np.nan)
    true_div = np.where(inside, -K, np.nan)
    return velocity, true_shear, true_div, np.hypot(dx, dy)


def sweep(velocity, azimuth, rng):
    return xr.Dataset(
        {"VRADH": (("azimuth", "range"), velocity, {"units": "m s-1"})},
        coords={
            "azimuth": ("azimuth", azimuth, {"units": "degrees"}),
            "range": ("range", rng, {"units": "m"}),
            "elevation": ("azimuth", np.full(azimuth.size, 0.5)),
        },
    )


def jittered_azimuths(n=720, seed=0):
    """Unsorted azimuths with non-uniform spacing, starting mid-circle."""
    rng = np.random.default_rng(seed)
    az = np.arange(n) * 360.0 / n + rng.uniform(-0.15, 0.15, n) * 360.0 / n
    return np.mod(az + 123.4, 360.0)


RANGE = np.arange(2125.0, 100e3, 250.0)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("weights", ["uniform", "gaussian"])
def test_rankine_vortex(engine, weights):
    az = jittered_azimuths()
    vel, true_shear, true_div, dist = rankine(az, RANGE)
    out = llsd(sweep(vel, az, RANGE), weights=weights, engine=engine)
    shear = out["azimuthal_shear"].values
    div = out["radial_divergence"].values
    # gates whose whole window lies inside the core
    core = dist < CORE - 1600.0
    assert core.sum() > 100
    np.testing.assert_allclose(shear[core], true_shear[core], atol=0.01 * OMEGA)
    np.testing.assert_allclose(div[core], true_div[core], atol=0.01 * OMEGA)
    # at the vortex centre: shear = half the vorticity, divergence = half of it
    centre = np.unravel_index(np.argmin(dist), dist.shape)
    assert shear[centre] == pytest.approx(OMEGA, rel=0.02)
    assert div[centre] == pytest.approx(-K, abs=0.02 * OMEGA)
    # outside the core (1/distance flow, curved within the window) compare with
    # derivatives of the analytic field by central differences
    eps = 1e-4

    def field(a, r):
        return rankine(a, r)[0]

    num_shear = (
        field(az + np.degrees(eps), RANGE) - field(az - np.degrees(eps), RANGE)
    ) / (2 * eps * RANGE)
    num_div = (field(az, RANGE + 1.0) - field(az, RANGE - 1.0)) / 2.0
    outer = (dist > CORE + 2000.0) & (dist < 6 * CORE)
    np.testing.assert_allclose(shear[outer], num_shear[outer], atol=0.05 * OMEGA)
    np.testing.assert_allclose(div[outer], num_div[outer], atol=0.05 * OMEGA)


@pytest.mark.parametrize("engine", ENGINES)
def test_linear_field_is_exact(engine):
    az = jittered_azimuths(seed=1)
    vel = 3.0 + 0.004 * RANGE[None, :] + 0.0 * az[:, None]
    out = llsd(sweep(vel, az, RANGE), engine=engine)
    np.testing.assert_allclose(out["radial_divergence"], 0.004, rtol=1e-4)
    np.testing.assert_allclose(out["azimuthal_shear"], 0.0, atol=1e-7)


@pytest.mark.parametrize("engine", ENGINES)
def test_wraparound_at_north(engine):
    """A vortex straddling north gives the same result as one at 90 deg."""
    az = np.arange(0.25, 360.0, 0.5)
    rot = 90.0
    vel_n, *_, dist_n = rankine(az, RANGE, centre=(0.0, 50e3))
    out_n = llsd(sweep(vel_n, az, RANGE), engine=engine)
    # the same field rotated by 90 deg: shift the azimuth labels
    out_e = llsd(sweep(vel_n, np.mod(az + rot, 360.0), RANGE), engine=engine)
    np.testing.assert_allclose(
        out_n["azimuthal_shear"].values,
        out_e["azimuthal_shear"].values,
        atol=1e-6 * OMEGA,
    )
    centre = np.unravel_index(np.argmin(dist_n), dist_n.shape)
    assert out_n["azimuthal_shear"].values[centre] == pytest.approx(OMEGA, rel=0.02)


@pytest.mark.parametrize("engine", ENGINES)
def test_missing_data_and_valid_fraction(engine):
    az = jittered_azimuths(seed=2)
    vel, *_ = rankine(az, RANGE)
    holes = np.random.default_rng(3).random(vel.shape) < 0.4
    vel = np.where(holes, np.nan, vel)
    ds = sweep(vel, az, RANGE)
    strict = llsd(ds, min_valid_fraction=0.9, engine=engine)["azimuthal_shear"]
    loose = llsd(ds, min_valid_fraction=0.0, engine=engine)["azimuthal_shear"]
    # missing centre gates are never filled
    assert np.isnan(loose.values[holes]).all()
    assert np.isfinite(loose.values[~holes]).mean() > 0.99
    assert np.isfinite(strict.values).sum() < np.isfinite(loose.values).sum()
    # a mask removes gates like missing data
    ds["far"] = (ds.range > 50e3).broadcast_like(ds["VRADH"])
    for mask in ("far", ds["far"], ds["far"].values):
        masked = llsd(ds, mask=mask, engine=engine)["azimuthal_shear"]
        assert np.isnan(masked.sel(range=slice(50.1e3, None))).all()
        assert np.isfinite(masked.sel(range=slice(None, 45e3))).any()


@needs_kernel
@pytest.mark.parametrize("weights", ["uniform", "gaussian"])
def test_compiled_matches_numpy(weights):
    """Both engines agree to float32 rounding (cumulative vs direct sums)."""
    az = jittered_azimuths(seed=4)
    vel, *_ = rankine(az, RANGE)
    vel = np.where(np.random.default_rng(5).random(vel.shape) < 0.2, np.nan, vel)
    # a sector scan with a gap: rays between 200 and 250 deg are missing
    keep = (az < 200) | (az > 250)
    ds = sweep(vel[keep], az[keep], RANGE)
    options = {"weights": weights, "window": (1000.0, 3000.0)}
    a = llsd(ds, engine="compiled", n_threads=3, **options)
    b = llsd(ds, engine="numpy", **options)
    for name in ("azimuthal_shear", "radial_divergence"):
        np.testing.assert_array_equal(np.isnan(a[name]), np.isnan(b[name]))
        np.testing.assert_allclose(a[name], b[name], rtol=1e-4, atol=1e-8)


def test_xarray_interface():
    az = jittered_azimuths(seed=6)
    vel, *_ = rankine(az, RANGE)
    ds = sweep(vel, az, RANGE).transpose("range", "azimuth")
    shear = ds.radarx.azimuthal_shear("VRADH", window=(750.0, 2500.0))
    div = radial_divergence(ds)
    xr.testing.assert_identical(ds.radarx.radial_divergence("VRADH"), div)
    assert shear.dims == ("range", "azimuth")
    assert shear.attrs["units"] == "s-1"
    assert shear.attrs["window_azimuth_m"] == 2500.0
    assert "elevation" in shear.coords
    xr.testing.assert_identical(shear, azimuthal_shear(ds))
    xr.testing.assert_identical(div, ds.radarx.llsd()["radial_divergence"])

    root = xr.Dataset({"site": 1.0})
    tree = xr.DataTree.from_dict({"/": root, "sweep_0": ds, "sweep_1": ds})
    out = tree.radarx.llsd()
    assert set(out.children) == {"sweep_0", "sweep_1"}
    assert float(out["site"]) == 1.0  # the input root is kept
    merged = tree.radarx.assign(out)
    assert {"VRADH", "azimuthal_shear"} <= set(merged["sweep_1"].data_vars)
    xr.testing.assert_allclose(out["sweep_1"].to_dataset()["azimuthal_shear"], shear)
    only = tree.radarx.azimuthal_shear()
    assert list(only["sweep_0"].data_vars) == ["azimuthal_shear"]


def test_errors():
    az = jittered_azimuths()
    ds = sweep(np.zeros((az.size, RANGE.size)), az, RANGE)
    with pytest.raises(ValueError, match="engine"):
        llsd(ds, engine="fortran")
    with pytest.raises(ValueError, match="weights"):
        llsd(ds, weights="triangle")
    with pytest.raises(ValueError, match="window"):
        llsd(ds, window=(0.0, 1000.0))
    with pytest.raises(KeyError):
        llsd(ds, "VEL")
    with pytest.raises(TypeError):
        llsd(ds["VRADH"])
    with pytest.raises(ValueError, match="range must increase"):
        llsd(ds.isel(range=slice(None, None, -1)), engine="numpy")
    with pytest.raises(ValueError, match="2-D"):
        llsd(ds.isel(range=0, drop=False).expand_dims("time"), engine="numpy")
    with pytest.raises(ValueError, match="one azimuth per ray"):
        bad = xr.Dataset(
            {"VRADH": (("time", "range"), np.zeros((az.size, RANGE.size)))},
            coords={"azimuth": ("other", az[:5]), "range": RANGE},
        )
        llsd(bad, engine="numpy")
    tree = xr.DataTree.from_dict({"/": xr.Dataset(), "sweep_0": ds})
    with pytest.raises(ValueError, match="No sweep contains"):
        llsd(tree, "VEL")


def test_compiled_engine_unavailable(monkeypatch):
    az = jittered_azimuths()
    ds = sweep(np.zeros((az.size, RANGE.size)), az, RANGE)
    monkeypatch.setattr(shear_mod, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError, match="compiled LLSD kernel"):
        llsd(ds, engine="compiled")
    out = llsd(ds, engine="auto")  # falls back to NumPy
    np.testing.assert_allclose(out["azimuthal_shear"], 0.0, atol=1e-12)


def test_datatree_older_xarray(monkeypatch):
    """Without inherit="all_coords" support, plain to_dataset() is used."""
    az = jittered_azimuths()
    vel, *_ = rankine(az, RANGE)
    tree = xr.DataTree.from_dict({"/": xr.Dataset(), "sweep_0": sweep(vel, az, RANGE)})
    original = xr.DataTree.to_dataset

    def old_to_dataset(self, inherit=True):
        if inherit == "all_coords":
            raise TypeError("unsupported")
        return original(self, inherit=inherit)

    monkeypatch.setattr(xr.DataTree, "to_dataset", old_to_dataset)
    out = llsd(tree)
    assert "azimuthal_shear" in out["sweep_0"].data_vars


@pytest.fixture(scope="module")
def nexrad_sweep():
    """0.5 deg velocity cut of a NEXRAD volume (KLBB), with no-data masked."""
    xd = pytest.importorskip("xradar")
    from open_radar_data import DATASETS

    file = DATASETS.fetch("KLBB20160601_150025_V06")
    dtree = xd.io.open_nexradlevel2_datatree(file, sweep=[1])
    ds = dtree["sweep_1"].to_dataset()
    # raw codes 0 (below threshold) and 1 (range folded) decode to <= -64
    ds["VRADH"] = ds["VRADH"].where(ds["VRADH"] > -64.0)
    return ds


@pytest.mark.parametrize("engine", ENGINES)
def test_nexrad_least_squares(nexrad_sweep, engine):
    """LLSD on real data equals an explicit least-squares fit per gate."""
    ds = nexrad_sweep
    out = llsd(ds, "VRADH", engine=engine)
    shear = out["azimuthal_shear"].values
    div = out["radial_divergence"].values
    vel = ds["VRADH"].values
    valid = np.isfinite(vel)
    assert np.isfinite(shear[valid]).mean() > 0.8
    assert np.nanpercentile(np.abs(shear), 99) < 0.05

    az = np.radians(ds["azimuth"].values)
    rng = ds["range"].values
    gen = np.random.default_rng(7)
    rays, gates = np.nonzero(np.isfinite(shear))
    for idx in gen.choice(rays.size, 25, replace=False):
        i, g = rays[idx], gates[idx]
        r0 = rng[g]
        dt = np.mod(az - az[i] + np.pi, 2 * np.pi) - np.pi
        in_az = np.abs(dt) <= max(1250.0 / r0, 1.5 * np.radians(0.5))
        in_r = np.abs(rng - r0) <= 375.0
        q, k = np.nonzero(in_az[:, None] & in_r[None, :] & valid)
        G = np.c_[np.ones(q.size), rng[k] * dt[q], rng[k] - r0]
        coef = np.linalg.lstsq(G, vel[q, k], rcond=None)[0]
        assert shear[i, g] == pytest.approx(coef[1], rel=1e-3, abs=1e-6)
        assert div[i, g] == pytest.approx(coef[2], rel=1e-3, abs=1e-6)


@needs_kernel
def test_nexrad_engines_agree(nexrad_sweep):
    a = llsd(nexrad_sweep, engine="compiled")
    b = llsd(nexrad_sweep, engine="numpy")
    for name in ("azimuthal_shear", "radial_divergence"):
        np.testing.assert_allclose(a[name], b[name], rtol=1e-4, atol=1e-8)
