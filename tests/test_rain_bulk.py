#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests of radarx.retrieve.rain_bulk (two-moment rain shaft)
=========================================================

The shape parameter, the fall speeds of the gamma distribution and the
relaxation of the breakup are checked against the equations of Seifert (2008)
(the fall speeds by numerical integration of the distribution), the rain rate
for conservation, the integration against SciPy, the compiled kernel against
the NumPy twin, and the approach to the equilibrium mean volume diameter.
"""

import importlib

import numpy as np
import pytest
import xarray as xr
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq
from scipy.special import gamma

import radarx  # noqa: F401
from radarx.retrieve import _rain_bulk_numpy as ref
from radarx.retrieve import rain_shaft, rain_shaft_bulk

rb = importlib.import_module("radarx.retrieve.rain_bulk")
ENGINES = ["numpy"] + (["compiled"] if rb.HAS_COMPILED_KERNEL else [])


def top_of(dm_mm, lwc_g):
    """Number concentration (m-3) and rainwater content (g m-3) of a top."""
    dm = dm_mm * 1e-3
    n = lwc_g * 1e-3 / ref.mass_of_mean_drop(dm)
    return xr.Dataset(
        {"number_concentration": float(n), "liquid_water_content": float(lwc_g)}
    )


def test_shape_parameter_of_equation_twenty():
    # mu is 1 at the equilibrium diameter, at most 7 below it and 31 far above it
    assert float(ref.shape_parameter(1.1e-3)) == pytest.approx(1.0)
    assert float(ref.shape_parameter(1e-5)) == pytest.approx(7.0, abs=5e-3)
    assert float(ref.shape_parameter(0.2)) == pytest.approx(31.0)
    dm = np.linspace(1e-4, 0.5e-3, 50)
    mu = ref.shape_parameter(dm)
    assert np.all(mu <= 7.0) and np.all(np.diff(mu) < 0)  # falls toward D_eq
    # both branches
    assert float(ref.shape_parameter(0.9e-3)) == pytest.approx(
        6 * np.tanh(4.0 * (0.9 - 1.1)) ** 2 + 1
    )
    assert float(ref.shape_parameter(2.0e-3)) == pytest.approx(
        30 * np.tanh(1.0 * (2.0 - 1.1)) ** 2 + 1
    )


@pytest.mark.parametrize("dm_mm", [0.4, 1.1, 2.3, 4.0])
def test_fall_speeds_are_moments_of_the_gamma_distribution(dm_mm):
    # Eq. (A10) is the integral of a - b exp(-c D) over the distribution:
    # v_k = int v(D) D^(mu+k) exp(-lam D) dD / int D^(mu+k) exp(-lam D) dD
    dm = dm_mm * 1e-3
    mu = float(ref.shape_parameter(dm))
    lam = float(ref.slope(dm, mu))
    vn, vl = ref.sedimentation(dm)
    for k, expected in ((0, vn), (3, vl)):
        num = quad(
            lambda d: (ref.A_V - ref.B_V * np.exp(-ref.C_V * d))
            * d ** (mu + k)
            * np.exp(-lam * d),
            0,
            (mu + k + 1 + 60.0) / lam,
            epsabs=0,
            epsrel=1e-12,
            limit=200,
        )[0]
        den = gamma(mu + k + 1) / lam ** (mu + k + 1)
        assert num / den == pytest.approx(float(expected), rel=1e-8)
    # the mass of the gamma distribution with the mean volume diameter
    n0 = 1.0
    lw = n0 * np.pi / 6 * 1000 * gamma(mu + 4) / gamma(mu + 1) / lam**3
    assert (6 * lw / (np.pi * 1000 * n0)) ** (1 / 3) == pytest.approx(dm, rel=1e-12)


def test_flux_ratio_decreases_and_inverts():
    dm = np.geomspace(ref.DM_LOW, ref.DM_HIGH, 400)
    r = ref.flux_ratio(dm)
    assert np.all(np.diff(r) < 0)
    for d in (1e-4, 8e-4, 1.1e-3, 3e-3, 1e-2):
        assert ref.mean_diameter_of_flux_ratio(
            float(ref.flux_ratio(d))
        ) == pytest.approx(d, rel=1e-12)


@pytest.mark.parametrize("engine", ENGINES)
def test_equilibrium_stays_and_rain_rate_is_conserved(engine):
    top = top_of(1.1, 2.0)
    out = rain_shaft_bulk(top, depth=[0.0, 1000.0, 5000.0], engine=engine)
    assert np.allclose(
        out.number_concentration, float(top.number_concentration), rtol=1e-9
    )
    assert np.allclose(out.mean_volume_diameter, 1.1, rtol=1e-9)
    # the rain rate does not change with depth for rain that is not in equilibrium
    for dm in (0.5, 2.5, 4.0):
        o = rain_shaft_bulk(top_of(dm, 3.0), depth=[0.0, 800.0, 3000.0], engine=engine)
        assert np.allclose(o.rain_rate, o.rain_rate.isel(depth=0), rtol=1e-9)
        # independent: the mass flux of the gamma distribution at the top
        vl = ref.sedimentation(dm * 1e-3)[1]
        assert float(o.rain_rate.isel(depth=0)) == pytest.approx(3.0e-3 * vl * 3600.0)


@pytest.mark.parametrize("engine", ENGINES)
def test_mean_diameter_relaxes_toward_the_equilibrium(engine):
    for dm0, direction in ((4.0, -1), (2.0, -1), (0.6, 1), (0.35, 1)):
        o = rain_shaft_bulk(
            top_of(dm0, 2.0), depth=np.arange(0.0, 9001.0, 1000.0), engine=engine
        )
        dm = o.mean_volume_diameter.values
        assert np.all(np.diff(dm) * direction >= -1e-12)  # monotone toward D_eq
        assert abs(dm[-1] - 1.1) < abs(dm[0] - 1.1)
        assert (
            np.all(np.sign(dm - 1.1) == np.sign(dm0 - 1.1)) or abs(dm[-1] - 1.1) < 5e-3
        )
    # long shaft: the equilibrium of 1.1 mm
    o = rain_shaft_bulk(top_of(3.0, 4.0), depth=[0.0, 30000.0], engine=engine)
    assert float(o.mean_volume_diameter.isel(depth=1)) == pytest.approx(1.1, abs=5e-3)
    # breakup makes drops (the number grows) above 1.1 mm, coalescence loses them below
    o = rain_shaft_bulk(top_of(3.0, 4.0), depth=[0.0, 2000.0], engine=engine)
    assert o.number_concentration.isel(depth=1) > o.number_concentration.isel(depth=0)
    o = rain_shaft_bulk(top_of(0.6, 1.0), depth=[0.0, 2000.0], engine=engine)
    assert o.number_concentration.isel(depth=1) < o.number_concentration.isel(depth=0)


@pytest.mark.parametrize("dm0,lw,tol", [(2.8e-3, 2.5e-3, 2e-6), (0.2e-3, 0.4e-3, 3e-4)])
@pytest.mark.parametrize("engine", ENGINES)
def test_integration_agrees_with_scipy(engine, dm0, lw, tol):
    # independent integration of Eqs. (A12), (A13) in the number and the water
    # content (SciPy, adaptive), with the closure by root finding in Dm; the
    # drops of 0.2 mm only coalesce (no breakup below 0.3 mm), as they grow
    # past 0.3 mm the breakup term of Eq. (A13) starts, a jump of the source
    # that limits both integrations to first order there (hence the looser tolerance)
    n0 = lw / ref.mass_of_mean_drop(dm0)
    vn0, vl0 = ref.sedimentation(dm0)
    fl = vl0 * lw
    fn0 = vn0 * n0

    def dm_of(fn):
        return brentq(
            lambda d: ref.flux_ratio(d) - fn / fl,
            ref.DM_LOW,
            ref.DM_HIGH,
            xtol=1e-15,
            rtol=1e-14,
        )

    def rhs(z, y):
        dm = dm_of(y[0])
        water = fl / ref.sedimentation(dm)[1]
        n = water / ref.mass_of_mean_drop(dm)
        return [ref.source(n, water, dm)]

    z_out = [0.0, 700.0, 1500.0, 3000.0]
    sol = solve_ivp(rhs, (0, 3000.0), [fn0], t_eval=z_out, rtol=1e-11, atol=1e-6)
    top = top_of(dm0 * 1e3, lw * 1e3)
    out = rain_shaft_bulk(top, depth=z_out, engine=engine)
    dm_ref = [dm_of(f) for f in sol.y[0]]
    assert np.allclose(out.mean_volume_diameter * 1e-3, dm_ref, rtol=tol)
    water = fl / np.array([ref.sedimentation(d)[1] for d in dm_ref])
    assert np.allclose(
        out.number_concentration,
        water / ref.mass_of_mean_drop(np.array(dm_ref)),
        rtol=tol,
    )


def test_step_size_converges():
    top = top_of(3.0, 3.0)
    coarse = rain_shaft_bulk(top, depth=2500.0, step=100.0, cfl=1.0, engine="numpy")
    fine = rain_shaft_bulk(top, depth=2500.0, step=2.0, cfl=0.05, engine="numpy")
    assert float(
        coarse.number_concentration / fine.number_concentration - 1
    ) == pytest.approx(0.0, abs=1e-5)


@pytest.mark.skipif(not rb.HAS_COMPILED_KERNEL, reason="compiled kernel not built")
def test_engines_agree_and_threads_do_not_matter():
    rng = np.random.default_rng(5)
    dm = rng.uniform(0.2, 5.0, 12)
    lw = rng.uniform(0.2, 4.0, 12)
    n = lw * 1e-3 / ref.mass_of_mean_drop(dm * 1e-3)
    ds = xr.Dataset(
        {
            "number_concentration": ("x", n),
            "liquid_water_content": ("x", lw),
        }
    )
    a = rain_shaft_bulk(ds, depth=[0.0, 600.0, 1500.0], engine="compiled", n_threads=1)
    b = rain_shaft_bulk(ds, depth=[0.0, 600.0, 1500.0], engine="compiled", n_threads=4)
    c = rain_shaft_bulk(ds, depth=[0.0, 600.0, 1500.0], engine="numpy")
    for name in a.data_vars:
        assert np.array_equal(a[name].values, b[name].values)
        assert np.allclose(a[name], c[name], rtol=1e-12, atol=0)
    assert a.number_concentration.dims == ("x", "depth")


@pytest.mark.parametrize("engine", ENGINES)
def test_density_profile_scales_the_fall_speed(engine):
    top = top_of(2.5, 2.0)
    still = rain_shaft_bulk(top, depth=2000.0, density_correction=False, engine=engine)
    sea = rain_shaft_bulk(top, depth=2000.0, engine=engine)  # no profile: sea-level air
    assert float(still.number_concentration) == float(sea.number_concentration)
    z = np.linspace(0.0, 6000.0, 61)
    rho = 1.2 * np.exp(-z / 8000.0)
    profile = xr.Dataset({"air_density": ("height", rho)}, coords={"height": z})
    dense = rain_shaft_bulk(
        top, depth=[0.0, 2000.0], profile=profile, top_height=2500.0, engine=engine
    )
    thin = rain_shaft_bulk(
        top, depth=[0.0, 2000.0], profile=profile, top_height=6000.0, engine=engine
    )
    # the rain rate does not change with depth; at the top it is proportional to
    # the density factor (rho0/rho)^0.4 of the fall speed
    assert np.allclose(dense.rain_rate, dense.rain_rate.isel(depth=0), rtol=1e-9)
    assert np.allclose(thin.rain_rate, thin.rain_rate.isel(depth=0), rtol=1e-9)
    ratio = float(thin.rain_rate.isel(depth=0) / dense.rain_rate.isel(depth=0))
    assert ratio == pytest.approx(
        (np.interp(2500.0, z, rho) / np.interp(6000.0, z, rho)) ** 0.4, rel=1e-12
    )
    # a uniform density factor scales the fall speed and the collision rate
    # alike, so the shaft does not change: only the rain rate does
    flat = xr.Dataset(
        {"air_density": ("height", np.full(z.size, 0.8))}, coords={"height": z}
    )
    uniform = rain_shaft_bulk(
        top, depth=[0.0, 2000.0], profile=flat, top_height=3000.0, engine=engine
    )
    plain = rain_shaft_bulk(
        top, depth=[0.0, 2000.0], density_correction=False, engine=engine
    )
    np.testing.assert_allclose(
        uniform.number_concentration, plain.number_concentration, rtol=1e-10
    )
    np.testing.assert_allclose(
        uniform.rain_rate, plain.rain_rate * (1.204 / 0.8) ** 0.4, rtol=1e-12
    )
    with pytest.raises(ValueError, match="top_height"):
        rain_shaft_bulk(top, depth=100.0, profile=profile)
    with pytest.raises(ValueError, match="must cover"):
        rain_shaft_bulk(top, depth=[0.0, 5000.0], profile=profile, top_height=3000.0)


@pytest.mark.parametrize("engine", ENGINES)
def test_nan_and_out_of_range_columns(engine):
    n = xr.DataArray([np.nan, 100.0, 5.0e9, 244.5, -1.0, 100.0], dims="x")
    w = xr.DataArray([1.0, np.nan, 1.0, 2.0, 1.0, 0.0], dims="x")
    out = rain_shaft_bulk(
        xr.Dataset({"number_concentration": n, "liquid_water_content": w}),
        depth=[0.0, 500.0],
        engine=engine,
    )
    ok = out.mean_volume_diameter.notnull().all("depth")
    assert ok.values.tolist() == [False, False, False, True, False, False]
    assert out.sizes == {"x": 6, "depth": 2}


def test_scalar_depth_attrs_and_variables():
    out = rain_shaft_bulk(top_of(2.0, 1.0), depth=1000.0)
    assert out.depth.ndim == 0 and float(out.depth) == 1000.0
    for name in (
        "number_concentration",
        "liquid_water_content",
        "mean_volume_diameter",
        "shape_parameter",
        "slope_parameter",
        "rain_rate",
    ):
        assert "units" in out[name].attrs
    assert out.depth.attrs["units"] == "m"
    # shape and slope follow the mean volume diameter
    dm = float(out.mean_volume_diameter) * 1e-3
    assert float(out.shape_parameter) == pytest.approx(float(ref.shape_parameter(dm)))
    mu = float(out.shape_parameter)
    assert float(out.slope_parameter) * 1e3 == pytest.approx(
        ((mu + 3) * (mu + 2) * (mu + 1)) ** (1 / 3) / dm
    )


def test_same_form_as_rain_shaft_output_and_comparison_with_the_bin_model():
    d = np.geomspace(0.15, 8.0, 50)
    mu, dm0, lw = 3.0, 2.5, 2.0
    lam = ((mu + 3) * (mu + 2) * (mu + 1)) ** (1 / 3) / dm0
    shape = d**mu * np.exp(-lam * d)
    y = np.pi / 6 * 1e-3 * d**3 * shape
    w = float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(d)))
    nd = xr.DataArray(
        shape * lw / w, dims="diameter", coords={"diameter": ("diameter", d)}
    )
    bins = rain_shaft(nd, depth=[0.0, 8000.0])
    top = bins.isel(depth=0)[["number_concentration", "liquid_water_content"]]
    bulk = rain_shaft_bulk(top, depth=[0.0, 8000.0])
    # same rain rate (the bin value is the sum over the pivots), both near the
    # equilibrium diameter at the end
    assert float(bulk.rain_rate.isel(depth=1)) == pytest.approx(
        float(bulk.rain_rate.isel(depth=0))
    )
    dm_bin = (
        6e-3 * bins.liquid_water_content / (np.pi * 1000 * bins.number_concentration)
    ) ** (1 / 3) * 1e3
    assert 0.9 < float(dm_bin.isel(depth=1)) < 1.4
    assert 0.9 < float(bulk.mean_volume_diameter.isel(depth=1)) < 1.4


def test_inputs_are_not_changed_dask_is_lazy_provenance_and_accessor():
    dask = pytest.importorskip("dask.array")
    top = top_of(2.0, 1.0)
    n = xr.DataArray(np.full((3, 4), float(top.number_concentration)), dims=("a", "b"))
    w = xr.DataArray(np.full((3, 4), 1.0), dims=("a", "b"))
    ds = xr.Dataset({"number_concentration": n, "liquid_water_content": w})
    before = ds.copy(deep=True)
    lazy = ds.chunk({"a": 1})
    out = rain_shaft_bulk(lazy, depth=[0.0, 1000.0])
    assert out.number_concentration.chunks is not None
    eager = rain_shaft_bulk(ds, depth=[0.0, 1000.0])
    xr.testing.assert_allclose(out.compute(), eager)
    xr.testing.assert_identical(ds, before)
    via = ds.radarx.rain_shaft_bulk(depth=[0.0, 1000.0])
    xr.testing.assert_allclose(via, eager)
    assert eager.attrs["radarx_references"] == "10.1175/2008JAS2586.1"
    del dask


def test_errors():
    top = top_of(2.0, 1.0)
    with pytest.raises(TypeError, match="Dataset"):
        rain_shaft_bulk(top.number_concentration, depth=100.0)
    with pytest.raises(ValueError, match="liquid_water_content"):
        rain_shaft_bulk(top[["number_concentration"]], depth=100.0)
    with pytest.raises(ValueError, match="number_concentration"):
        rain_shaft_bulk(top[["liquid_water_content"]], depth=100.0)
    with pytest.raises(ValueError, match="ascending"):
        rain_shaft_bulk(top, depth=[500.0, 100.0])
    with pytest.raises(ValueError, match="not negative"):
        rain_shaft_bulk(top, depth=[-1.0, 100.0])
    with pytest.raises(ValueError, match="one-dimensional"):
        rain_shaft_bulk(top, depth=[])
    with pytest.raises(ValueError, match="one-dimensional"):
        rain_shaft_bulk(top, depth=[[1.0]])
    with pytest.raises(ValueError, match="positive"):
        rain_shaft_bulk(top, depth=100.0, step=0.0)
    with pytest.raises(ValueError, match="engine"):
        rain_shaft_bulk(top, depth=100.0, engine="fortran")


@pytest.mark.skipif(not rb.HAS_COMPILED_KERNEL, reason="compiled kernel not built")
def test_kernel_rejects_bad_arguments():
    k = rb._rain_bulk
    ok = dict(
        n0=np.array([200.0]),
        water0=np.array([1e-3]),
        z_out=np.array([0.0, 100.0]),
        dz_max=10.0,
        cfl=0.25,
        zs=np.array([0.0, 1e4]),
        ss=np.array([1.0, 1.0]),
    )
    assert k.evolve(**ok).shape == (2, 1, 3)
    for key, value, match in (
        ("n0", np.ones((1, 1)), "dimensions"),
        ("water0", np.array([1.0, 2.0]), "same size"),
        ("zs", np.array([0.0, 1e4, 2e4]), "match"),
        ("dz_max", 0.0, "positive"),
        ("cfl", -1.0, "positive"),
        ("z_out", np.array([100.0, 0.0]), "ascending"),
        ("zs", np.array([1e4, 0.0]), "ascend"),
        ("ss", np.array([1.0, 0.0]), "positive"),
    ):
        bad = dict(ok)
        bad[key] = value
        with pytest.raises(ValueError, match=match):
            k.evolve(**bad)
    # the helper functions of the module agree with the twin
    assert k.shape_parameter(1.1e-3) == pytest.approx(1.0)
    assert k.sedimentation(2e-3) == pytest.approx(ref.sedimentation(2e-3), rel=1e-13)


def test_missing_kernel(monkeypatch):
    monkeypatch.setattr(rb, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError, match="not available"):
        rain_shaft_bulk(top_of(2.0, 1.0), depth=100.0, engine="compiled")
    out = rain_shaft_bulk(top_of(2.0, 1.0), depth=100.0)  # falls back to NumPy
    assert float(out.number_concentration) > 0
