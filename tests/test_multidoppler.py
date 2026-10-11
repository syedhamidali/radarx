# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the multi-Doppler wind retrieval
==========================================

The cost function and its gradient are checked against finite differences
and between the compiled and NumPy engines. The retrieval is checked on an
analytic three-dimensional flow (a Beltrami flow, Shapiro 1993, divided by
a density profile so that it satisfies the anelastic continuity equation)
sampled by two virtual radars, and on two NEXRAD radars.
"""

import warnings

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
)

ENGINES = ["numpy"] + (["compiled"] if md.HAS_COMPILED_KERNEL else [])
needs_kernel = pytest.mark.skipif(
    not md.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)


# --------------------------------------------------------------------------
# synthetic case
# --------------------------------------------------------------------------


def beltrami(x, y, z, wmax=5.0, lx=40e3, lz=10e3, u0=8.0, v0=4.0):
    """Beltrami flow (Shapiro 1993) as a mass flux, plus a mean wind."""
    k = l_ = 2 * np.pi / lx
    m = np.pi / lz
    lam = np.sqrt(k * k + l_ * l_ + m * m)
    kh2 = k * k + l_ * l_
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    fu = (
        -wmax
        / kh2
        * (
            lam * l_ * np.cos(k * X) * np.sin(l_ * Y) * np.sin(m * Z)
            + m * k * np.sin(k * X) * np.cos(l_ * Y) * np.cos(m * Z)
        )
    )
    fv = (
        wmax
        / kh2
        * (
            lam * k * np.sin(k * X) * np.cos(l_ * Y) * np.sin(m * Z)
            - m * l_ * np.cos(k * X) * np.sin(l_ * Y) * np.cos(m * Z)
        )
    )
    fw = wmax * np.cos(k * X) * np.cos(l_ * Y) * np.sin(m * Z)
    rho = 1.2 * np.exp(-Z / 10000.0)
    scale = 1.2 * np.exp(-z.mean() / 10000.0) / rho
    return u0 + fu * scale, v0 + fv * scale, fw * scale, rho


def synthetic_case(
    step=2000.0, nz=11, noise=0.0, radars=((-25e3, -20e3), (25e3, -20e3)), dbz=None
):
    x = np.arange(-40e3, 40e3 + 1, step)
    y = np.arange(-40e3, 40e3 + 1, step)
    z = np.linspace(0.0, 10e3, nz)
    u, v, w, rho = beltrami(x, y, z)
    ds = xr.Dataset(coords={"z": z, "y": y, "x": x, "radar": np.arange(len(radars))})
    ds["radar_x"] = ("radar", np.array([r[0] for r in radars]))
    ds["radar_y"] = ("radar", np.array([r[1] for r in radars]))
    ds["radar_altitude"] = ("radar", np.zeros(len(radars)))
    geo = radar_geometry(ds)
    el = np.radians(geo.elevation.values)
    az = np.radians(geo.azimuth.values)
    vt = 0.0
    if dbz is not None:
        vt = fall_speed(
            xr.DataArray(np.full(u.shape, dbz), dims=("z", "y", "x"), coords={"z": z}),
            xr.DataArray(rho, dims=("z", "y", "x")),
        ).values
    vr = np.cos(el) * (np.sin(az) * u + np.cos(az) * v) + np.sin(el) * (w - vt)
    vr = vr + noise * np.random.default_rng(1).normal(size=vr.shape)
    dist = np.hypot(ds.x - ds.radar_x, ds.y - ds.radar_y).transpose("radar", "y", "x")
    vr = np.where((dist < 60e3).values[:, None], vr, np.nan)
    ds["VRADH"] = (("radar", "z", "y", "x"), vr.astype(np.float32))
    if dbz is not None:
        ds["DBZH"] = (("radar", "z", "y", "x"), np.full(vr.shape, dbz, np.float32))
    bg = xr.Dataset(
        {
            "u": (("z", "y", "x"), np.full(u.shape, 8.0)),
            "v": (("z", "y", "x"), np.full(u.shape, 4.0)),
            "air_density": (("z", "y", "x"), rho),
        },
        coords={"z": z, "y": y, "x": x},
    )
    truth = xr.Dataset(
        {
            "u": (("z", "y", "x"), u),
            "v": (("z", "y", "x"), v),
            "w": (("z", "y", "x"), w),
        }
    )
    return ds, bg, truth


def rms_error(out, truth, mask):
    return {
        c: float(np.sqrt(np.mean((out[c].values - truth[c].values)[mask] ** 2)))
        for c in "uvw"
    }


# --------------------------------------------------------------------------
# operators, cost function and gradient
# --------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_operator_adjoints(axis):
    rng = np.random.default_rng(axis)
    f = rng.normal(size=(4, 5, 6))
    g = rng.normal(size=(4, 5, 6))
    assert np.isclose(
        np.vdot(md._d1(f, axis, 700.0), g), np.vdot(f, md._d1T(g, axis, 700.0))
    )
    assert np.isclose(np.vdot(md._d2(f, axis), g), np.vdot(f, md._d2T(g, axis)))


def random_problem(seed=0, nz=4, ny=5, nx=6, nr=2):
    rng = np.random.default_rng(seed)
    shape = (nz, ny, nx)
    state = rng.normal(size=(3,) + shape)
    args = (
        rng.normal(size=(nr * 3,) + shape),
        rng.normal(size=(nr,) + shape),
        rng.uniform(size=(nr,) + shape) * (rng.uniform(size=(nr,) + shape) > 0.3),
        rng.uniform(0.5, 1.2, size=shape),
        rng.normal(size=(3,) + shape),
        rng.uniform(size=(3,) + shape),
        (rng.uniform(size=shape) > 0.2).astype(float),
        1000.0,
        900.0,
        500.0,
        0.7,
        0.3,
        0.4,
        0.2,
        0.9,
        3.0,
        -2.0,
        1e-4,
        950.0,
        950.0**2 / 10.0,
    )
    return state, args


def test_gradient_matches_finite_differences():
    state, args = random_problem()
    terms, grad = md._cost_gradient_numpy(state, *args)
    assert terms.shape == (5,) and np.all(terms > 0)
    eps = 1e-6
    fd = np.zeros_like(state)
    for idx in np.ndindex(state.shape):
        sp, sm = state.copy(), state.copy()
        sp[idx] += eps
        sm[idx] -= eps
        fd[idx] = (
            md._cost_gradient_numpy(sp, *args)[0].sum()
            - md._cost_gradient_numpy(sm, *args)[0].sum()
        ) / (2 * eps)
    # central differences of a smooth function: error ~ eps**2
    np.testing.assert_allclose(grad, fd, rtol=0, atol=1e-6 * np.abs(grad).max())


@needs_kernel
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("n_threads", [1, 0])
def test_compiled_matches_numpy(seed, n_threads):
    state, args = random_problem(seed, nz=5, ny=7, nx=9, nr=3)
    t1, g1 = md._cost_gradient_numpy(state, *args)
    t2, g2 = md._multidoppler.cost_gradient(state, *args, n_threads)
    # same arithmetic, different summation order: agreement to rounding
    np.testing.assert_allclose(t2, t1, rtol=1e-12)
    np.testing.assert_allclose(g2, g1, rtol=0, atol=1e-12 * np.abs(g1).max())


@needs_kernel
def test_compiled_threads_identical_gradient():
    state, args = random_problem(3, nz=6, ny=20, nx=30)
    _, g1 = md._multidoppler.cost_gradient(state, *args, 1)
    _, g8 = md._multidoppler.cost_gradient(state, *args, 8)
    np.testing.assert_array_equal(g1, g8)  # gathered per cell: no races


@needs_kernel
def test_compiled_kernel_rejects_bad_shapes():
    state, args = random_problem()
    with pytest.raises(ValueError, match="state"):
        md._multidoppler.cost_gradient(state[:2], *args)
    with pytest.raises(ValueError, match="at least"):
        md._multidoppler.cost_gradient(state[:, :1], *args)
    bad = list(args)
    bad[0] = bad[0][:3]
    with pytest.raises(ValueError, match="coef"):
        md._multidoppler.cost_gradient(state, *bad)
    bad = list(args)
    bad[1] = bad[1][:, :, :2]
    with pytest.raises(ValueError, match="target"):
        md._multidoppler.cost_gradient(state, *bad)


# --------------------------------------------------------------------------
# fall speed and geometry
# --------------------------------------------------------------------------


def test_fall_speed_relations():
    z = np.array([1000.0, 8000.0])
    dbz = xr.DataArray(np.array([40.0, 40.0]), dims="z", coords={"z": z})
    vt = fall_speed(dbz)
    np.testing.assert_allclose(vt, 2.65 * 1e4**0.114)
    assert vt.attrs["units"] == "m s-1"
    vt = fall_speed(dbz, freezing_level=4000.0)
    np.testing.assert_allclose(vt, [2.65 * 1e4**0.114, 0.817 * 1e4**0.063])
    rho = xr.DataArray([1.2, 0.6], dims="z", coords={"z": z})
    vt_rho = fall_speed(dbz, rho, freezing_level=4000.0)
    np.testing.assert_allclose(vt_rho / vt, [1.0, 2.0**0.4])
    assert np.isnan(fall_speed(dbz.where(dbz < 0)).values).all()


def test_beam_angles_match_xradar_geometry():
    from xradar.georeference import antenna_to_cartesian

    rng_ = np.array([20e3, 80e3, 150e3])
    for elevation in (0.5, 4.0, 19.5):
        for azimuth in (30.0, 200.0):
            x, y, z = antenna_to_cartesian(
                rng_, azimuth, elevation, site_altitude=150.0
            )
            az, el = md._beam_angles(x, y, z, 150.0)
            np.testing.assert_allclose(az, azimuth, atol=1e-6)
            # local elevation = antenna elevation + angle travelled around the Earth
            gamma = np.degrees(np.hypot(x, y) / (md.EARTH_RADIUS * 4 / 3))
            np.testing.assert_allclose(el - gamma, elevation, atol=0.01)


def test_radar_geometry_needs_positions():
    ds, _, _ = synthetic_case()
    with pytest.raises(ValueError, match="radar_x"):
        radar_geometry(ds.drop_vars("radar_x"))
    geo = radar_geometry(ds)
    assert geo.azimuth.dims == ("radar", "z", "y", "x")
    # straight above a radar the beam is vertical, at the radar it is undefined
    at = geo.sel(radar=0, x=-26e3, y=-20e3, method="nearest")
    assert float(at.elevation.isel(z=-1)) > 80.0


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def beltrami_case():
    return synthetic_case(noise=0.5)


def lobes(out):
    return (out.beam_crossing_angle > 30).values


@pytest.mark.parametrize("engine", ENGINES)
def test_retrieves_beltrami_flow(beltrami_case, engine):
    ds, bg, truth = beltrami_case
    out = multi_doppler(ds, bg, fall_speed_correction=False, engine=engine)
    err = rms_error(out, truth, lobes(out))
    # analytic truth with 0.5 m/s noise on 2-km grid
    assert err["u"] < 0.5 and err["v"] < 0.5 and err["w"] < 0.6
    assert (
        np.corrcoef(out.w.values[lobes(out)], truth.w.values[lobes(out)])[0, 1] > 0.95
    )
    assert (out.w.isel(z=0) == 0).all()  # lower boundary condition
    assert (
        out.u.attrs["units"] == "m s-1"
        and out.w.attrs["standard_name"] == "upward_air_velocity"
    )
    assert list(out.term.values) == list(md.TERMS)
    assert out.cost_history.sizes["term"] == 5
    assert out.cost_history.sum("term")[-1] < out.cost_history.sum("term")[0]
    assert out.attrs["converged"] == 1
    assert out.vr_residual.dims == ("radar", "z", "y", "x")
    assert float(np.sqrt(np.nanmean(out.vr_residual**2))) < 1.0


@needs_kernel
def test_engines_agree_end_to_end(beltrami_case):
    ds, bg, _ = beltrami_case
    # The kernel and the NumPy cost and gradient agree to 1e-16 (see
    # test_cost_and_gradient_engines_agree). Conjugate gradients stopped at
    # the default tolerance end after a different number of iterations when
    # rounding flips the stopping test (114 against 113 on the finest grid
    # here), which moves the state by 0.03 m/s although the minimum is 1 m/s
    # away in the flat directions of the cost. The engines are compared at a
    # tolerance where the iterates have converged.
    kw = dict(fall_speed_correction=False, tolerance=1e-9, max_iterations=3000)
    a = multi_doppler(ds, bg, engine="compiled", **kw)
    b = multi_doppler(ds, bg, engine="numpy", **kw)
    for c in "uvw":
        np.testing.assert_allclose(a[c], b[c], atol=1e-3)
    np.testing.assert_allclose(float(a.cost.sum()), float(b.cost.sum()), rtol=1e-8)


def test_lbfgsb_and_cg_reach_the_same_minimum(beltrami_case):
    ds, bg, _ = beltrami_case
    cg = multi_doppler(ds, bg, fall_speed_correction=False, tolerance=1e-4, levels=1)
    lb = multi_doppler(
        ds,
        bg,
        fall_speed_correction=False,
        solver="lbfgsb",
        tolerance=1e-12,
        max_iterations=2000,
        levels=1,
    )
    np.testing.assert_allclose(float(cg.cost.sum()), float(lb.cost.sum()), rtol=1e-4)
    # the flat directions of the cost (outside the dual-Doppler lobes) converge last
    diff = (cg.w - lb.w).values[lobes(cg)]
    assert np.sqrt(np.mean(diff**2)) < 0.05


def test_coarse_to_fine_levels(beltrami_case):
    ds, bg, truth = beltrami_case
    out = multi_doppler(ds, bg, fall_speed_correction=False, levels=3)
    assert set(np.unique(out.grid_level.values)) == {0, 1, 2}
    assert rms_error(out, truth, lobes(out))["w"] < 0.6


def test_fall_speed_correction():
    ds, bg, truth = synthetic_case(dbz=45.0)
    # the synthetic fall speed is the rain law at all heights
    out = multi_doppler(ds, bg, freezing_level="rain")
    raw = multi_doppler(ds, bg, fall_speed_correction=False)
    m = lobes(out)
    assert rms_error(out, truth, m)["w"] < 0.5 * rms_error(raw, truth, m)["w"]
    assert float(out.fall_speed.max()) > 7.0


def test_w_boundaries(beltrami_case):
    ds, bg, truth = beltrami_case
    out = multi_doppler(
        ds, bg, fall_speed_correction=False, w_boundary=("bottom", "top")
    )
    assert (out.w.isel(z=0) == 0).all() and (out.w.isel(z=-1) == 0).all()
    # the Beltrami flow has w = 0 at the top: the condition helps
    assert rms_error(out, truth, lobes(out))["w"] < 0.4
    echo = ds.assign(VRADH=ds.VRADH.where(ds.z < 5000))
    echo["DBZH"] = echo.VRADH * 0 + 30.0
    out = multi_doppler(echo, bg, fall_speed_correction=False, w_boundary="echo_top")
    assert (out.w.sel(z=slice(7000, None)) == 0).all()
    assert (out.w.isel(z=0) != 0).any()
    out = multi_doppler(ds, bg, fall_speed_correction=False, w_boundary=None)
    assert (out.w.isel(z=0) != 0).any()


def test_vorticity_constraint(beltrami_case):
    ds, bg, truth = beltrami_case
    out = multi_doppler(
        ds,
        bg,
        fall_speed_correction=False,
        weights={"vorticity": 0.01},
        max_iterations=60,
        levels=1,
    )
    assert float(out.cost.sel(term="vorticity")) > 0
    assert np.isfinite(out.w).all()
    assert out.cost_history.sum("term")[-1] < out.cost_history.sum("term")[0]
    with pytest.raises(ValueError, match="vorticity"):
        multi_doppler(ds, bg, weights={"vorticity": 1.0}, solver="cg")


def test_inputs_without_background_and_as_list(beltrami_case):
    ds, _, truth = beltrami_case
    out = multi_doppler(ds, fall_speed_correction=False)
    assert float(out.cost.sel(term="background")) == 0.0
    parts = [ds.isel(radar=[i]) for i in range(ds.sizes["radar"])]
    out2 = multi_doppler(parts, fall_speed_correction=False)
    np.testing.assert_allclose(out.u, out2.u, atol=1e-5)
    # a single radar with a background still gives a (poorly constrained) wind
    one = multi_doppler(ds.isel(radar=[0]), fall_speed_correction=False)
    assert "beam_crossing_angle" not in one


def test_first_guess(beltrami_case):
    ds, bg, truth = beltrami_case
    out = multi_doppler(ds, bg, fall_speed_correction=False)
    again = multi_doppler(ds, bg, fall_speed_correction=False, first_guess=out)
    assert again.sizes["iteration"] < out.sizes["iteration"]
    np.testing.assert_allclose(again.w, out.w, atol=0.1)


def test_missing_reflectivity_warns(beltrami_case):
    ds, bg, _ = beltrami_case
    with pytest.warns(UserWarning, match="fall speed"):
        multi_doppler(ds, bg, max_iterations=2)


def test_background_with_freezing_level():
    ds, bg, truth = synthetic_case(dbz=40.0)
    bg = bg.assign(
        freezing_level=(("y", "x"), np.full((ds.sizes["y"], ds.sizes["x"]), 4000.0))
    )
    bg["w"] = bg.u * 0.0
    out = multi_doppler(ds, bg, weights={"background_w": 0.01}, max_iterations=5)
    above = out.fall_speed.sel(z=slice(5000, None))
    below = out.fall_speed.sel(z=slice(None, 3000))
    assert float(above.max()) < float(below.min())


def test_errors(beltrami_case, monkeypatch):
    ds, bg, _ = beltrami_case
    with pytest.raises(ValueError, match="engine"):
        multi_doppler(ds, engine="fortran")
    with pytest.raises(ValueError, match="solver"):
        multi_doppler(ds, solver="newton")
    with pytest.raises(ValueError, match="unknown weights"):
        multi_doppler(ds, weights={"mass": 1.0})
    with pytest.raises(ValueError, match="levels"):
        multi_doppler(ds, levels=0)
    with pytest.raises(ValueError, match="radar"):
        multi_doppler(ds.isel(radar=0))
    with pytest.raises(ValueError, match="no 'VEL'"):
        multi_doppler(ds, velocity="VEL")
    with pytest.raises(ValueError, match="at least 3"):
        multi_doppler(ds.isel(x=slice(0, 2)))
    with pytest.raises(ValueError, match="evenly"):
        multi_doppler(ds.isel(x=[0, 1, 3, 4]))
    with pytest.raises(ValueError, match="background"):
        multi_doppler(ds, bg.isel(x=slice(0, 5)))
    with pytest.raises(ValueError, match="w_boundary"):
        multi_doppler(ds, w_boundary="sides")
    monkeypatch.setattr(md, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError):
        multi_doppler(ds, engine="compiled")
    assert not md._use_compiled("auto")


def test_input_needs_time_and_motion():
    with pytest.raises(ValueError, match="both time and motion"):
        multi_doppler_input([], [0.0], [0.0], [0.0], time="2022-03-31")


def test_accessor(beltrami_case):
    ds, bg, _ = beltrami_case
    out = ds.radarx.multi_doppler(bg, fall_speed_correction=False, max_iterations=3)
    assert "w" in out


# --------------------------------------------------------------------------
# real data: KGWX and KBMX, 30 March 2022 squall line
# --------------------------------------------------------------------------


def _nexrad(path):
    import xradar as xd
    from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

    with NEXRADLevel2File(path) as nf:
        nyquist = [
            h["msg_31_data_header"]["RAD"]["nyquist_vel"] / 100.0
            for h in nf.msg_31_data_header
        ]
    dtree = xd.io.open_nexradlevel2_datatree(path)
    names = [n for n in dtree.children if n.startswith("sweep")]
    for i, name in enumerate(names):
        ds = dtree[name].to_dataset()
        if "VRADH" in ds:
            ds["VRADH"] = ds.VRADH.where(ds.VRADH > -63.9)
        if "DBZH" in ds:
            ds["DBZH"] = ds.DBZH.where(ds.DBZH > -32)
        dtree[name] = ds.assign_coords(nyquist_velocity=nyquist[i])
    # dealias returns products only; put the dealiased velocity back as VRADH
    return dtree.radarx.assign(dtree.radarx.dealias("VRADH", name="VRADH"))


@pytest.fixture(scope="module")
def nexrad_pair(tmp_path_factory):
    from radarx.io.aws_data import download_file

    out = str(tmp_path_factory.mktemp("nexrad"))
    try:
        paths = [
            download_file("unidata-nexrad-level2", f"2022/03/30/{key}", out)
            for key in ("KGWX/KGWX20220330_235959_V06", "KBMX/KBMX20220330_235713_V06")
        ]
    except Exception as err:  # noqa: BLE001  # pragma: no cover - network
        pytest.skip(f"NEXRAD data not available: {err}")
    return [_nexrad(p) for p in paths]


def test_real_dual_doppler(nexrad_pair):
    x = np.arange(-100e3, 40e3 + 1, 4000.0)
    y = np.arange(-100e3, 80e3 + 1, 4000.0)
    z = np.arange(1000.0, 11e3 + 1, 1000.0)
    grids = multi_doppler_input(
        nexrad_pair, x, y, z, time="2022-03-31T00:00:00", motion=(15.0, 15.0)
    )
    assert list(grids.radar_name.values) == ["KGWX", "KBMX"]
    assert grids.VRADH.dims == ("radar", "z", "y", "x")
    np.testing.assert_allclose(grids.radar_x[1], 145.4e3, atol=1e3)
    assert (grids.time == np.datetime64("2022-03-31T00:00:00", "ns")).all()
    out = multi_doppler(grids)
    good = ((out.beam_crossing_angle > 30) & (out.n_radars >= 2)).values
    assert good.sum() > 1000
    # the retrieved wind explains both radars' velocities
    assert float(np.sqrt(np.nanmean(out.vr_residual.values[:, good] ** 2))) < 2.5
    assert float(np.abs(out.w.values[good]).max()) < 30.0
    # deep south-westerly flow ahead of the line at mid levels
    mid = out.v.sel(z=5000.0).values[good[4]]
    assert mid.mean() > 10.0


# --------------------------------------------------------------------------
# freezing level, lower boundary, advection of the winds, geodesic gridding
# --------------------------------------------------------------------------


def test_fall_speed_blends_across_the_melting_layer():
    z = np.array([1000.0, 3200.0, 3750.0, 4000.0, 8000.0])
    dbz = xr.DataArray(np.full(5, 30.0), dims="z", coords={"z": z})
    rain, snow = 2.65 * 1e3**0.114, 0.817 * 1e3**0.063
    assert rain > 5.5 and snow < 1.5  # 5.8 and 1.3 m/s at 30 dBZ (issue 221)
    vt = fall_speed(dbz, freezing_level=4000.0)
    # rain below 3500 m, snow from 4000 m, linear in between
    np.testing.assert_allclose(vt, [rain, rain, rain + 0.5 * (snow - rain), snow, snow])
    step = fall_speed(dbz, freezing_level=4000.0, melting_depth=0.0)
    np.testing.assert_allclose(step, [rain, rain, rain, rain, snow])
    # columns without a freezing level crossing: ice (-inf) and rain (inf) everywhere
    np.testing.assert_allclose(fall_speed(dbz, freezing_level=-np.inf), snow)
    np.testing.assert_allclose(fall_speed(dbz, freezing_level=np.inf), rain)


def test_no_freezing_level_warns_and_rain_option_is_silent():
    ds, bg, _ = synthetic_case(dbz=30.0)
    with pytest.warns(RuntimeWarning, match="no freezing level"):
        multi_doppler(ds, bg, max_iterations=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        multi_doppler(ds, bg, freezing_level="rain", max_iterations=2)
        out = multi_doppler(ds, bg, freezing_level=4000.0, max_iterations=2)
    assert float(out.fall_speed.sel(z=10000.0).max()) < 2.0  # snow aloft
    assert float(out.fall_speed.sel(z=0.0).min()) > 5.0  # rain below


def test_freezing_level_from_background_temperature():
    ds, bg, _ = synthetic_case(dbz=30.0)
    t = 288.15 - 6.5e-3 * bg.z  # 0 degC at 2308 m
    bg = bg.assign(temperature=t.broadcast_like(bg.u).transpose("z", "y", "x"))
    level = md._background_freezing_level(bg)
    np.testing.assert_allclose(level, 15.0 / 6.5e-3, rtol=1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = multi_doppler(ds, bg, max_iterations=2)
    assert float(out.fall_speed.sel(z=10000.0).max()) < 2.0
    # all ice and all rain columns
    cold = md._background_freezing_level(bg.assign(temperature=bg.temperature - 40.0))
    warm = md._background_freezing_level(bg.assign(temperature=bg.temperature + 100.0))
    assert (cold == -np.inf).all() and (warm == np.inf).all()


def test_freezing_level_given_and_from_temperature_are_merged():
    ds, bg, _ = synthetic_case(dbz=30.0)
    t = 288.15 - 6.5e-3 * bg.z  # 0 degC at 2307.7 m
    shape = (bg.sizes["y"], bg.sizes["x"])
    given = np.full(shape, 4000.0)
    given[:, 0] = np.nan  # no value in the first column
    bg = bg.assign(
        temperature=t.broadcast_like(bg.u).transpose("z", "y", "x"),
        freezing_level=(("y", "x"), given),
    )
    level = md._background_freezing_level(bg)
    assert level.dims == ("y", "x")
    np.testing.assert_allclose(level.values[:, 1:], 4000.0)  # the given value stays
    np.testing.assert_allclose(level.values[:, 0], 15.0 / 6.5e-3, rtol=1e-3)
    # no gaps in the given level: it is used as it is, the temperature is ignored
    full = bg.assign(freezing_level=(("y", "x"), np.full(shape, 3500.0)))
    np.testing.assert_allclose(md._background_freezing_level(full), 3500.0)
    # a temperature without a vertical dimension cannot give a level
    flat = xr.Dataset({"temperature": (("y", "x"), np.full(shape, 280.0))})
    assert md._background_freezing_level(flat) is None


def test_freezing_level_choices():
    ds, bg, _ = synthetic_case(dbz=30.0)
    with pytest.raises(ValueError, match="freezing_level must be 'auto', 'rain'"):
        multi_doppler(ds, bg, freezing_level="snow", max_iterations=2)
    # None takes the level of the background, like 'auto'
    t = 288.15 - 6.5e-3 * bg.z
    bg = bg.assign(temperature=t.broadcast_like(bg.u).transpose("z", "y", "x"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        by_none = multi_doppler(ds, bg, freezing_level=None, max_iterations=2)
    by_auto = multi_doppler(ds, bg, freezing_level="auto", max_iterations=2)
    np.testing.assert_allclose(by_none.fall_speed, by_auto.fall_speed)
    assert md._freezing_level_choice(None, None) is None
    assert md._freezing_level_choice(3000.0, 1000.0) == 3000.0


def test_no_volumes_and_bad_bottom_velocity():
    with pytest.raises(ValueError, match="no radar volumes"):
        multi_doppler_input([], [0.0], [0.0], [0.0])
    ds, bg, _ = synthetic_case(dbz=30.0)
    with pytest.raises(ValueError, match="w_bottom must be finite"):
        multi_doppler(ds, bg, w_bottom=np.nan, fall_speed_correction=False)


def test_shift_winds_needs_an_observation_time_for_every_radar():
    ds, bg, _, t_a = _translating_case()
    times = ds.time.values.copy()
    times[1] = np.datetime64("NaT")
    with pytest.raises(ValueError, match="a radar has no observation time"):
        multi_doppler(
            ds.assign_coords(time=("radar", times)),
            bg,
            shift_winds=True,
            analysis_time=t_a,
            storm_motion=(10.0, 0.0),
        )


def test_shift_winds_moves_the_reflectivity_with_the_storm():
    ds, _, _, t_a = _translating_case()
    c = (20.0, 10.0)
    x = ds.x.values
    # a field linear in x and y: the semi-Lagrangian step is exact inside the grid
    field = 0.001 * x[None, None, :] + 0.002 * ds.y.values[None, :, None]
    dbz = np.broadcast_to(field, (2, ds.sizes["z"], ds.sizes["y"], ds.sizes["x"]))
    ds = ds.assign(DBZH=(("radar", "z", "y", "x"), dbz.copy()))
    vr = ds.VRADH.values.copy()
    coef = np.zeros((2, 3) + vr.shape[1:])
    out, vr_out, coef_out = md._shift_observations(
        ds,
        vr,
        coef,
        reflectivity="DBZH",
        storm_motion=c,
        analysis_time=t_a,
        use_compiled=False,
        n_threads=None,
    )
    # the radars saw the pattern 180 s before and after the analysis time: the
    # values at the analysis time are those at x - c (t_a - t_k)
    for k, dt in enumerate((180.0, -180.0)):
        want = 0.001 * (x[None, None, :] - c[0] * dt) + 0.002 * (
            ds.y.values[None, :, None] - c[1] * dt
        )
        inner = slice(4, -4)
        np.testing.assert_allclose(
            out.DBZH.values[k][:, inner, inner],
            np.broadcast_to(want, out.DBZH.values[k].shape)[:, inner, inner],
            atol=1e-9,
        )
    assert vr_out.shape == vr.shape
    assert coef_out.shape == coef.shape
    # the input dataset keeps its reflectivity
    np.testing.assert_array_equal(ds.DBZH.values, dbz)


def _anelastic_case(z0, step=3000.0, dz=500.0, nz=21, a=2e-4, scale=1e4):
    """Two radars and the analytic anelastic flow of issue 171."""
    x = np.arange(-30e3, 30e3 + 1, step)
    y = np.arange(-30e3, 30e3 + 1, step)
    z = z0 + dz * np.arange(nz)
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    u, v = 5.0 + a * X, 2.0 + a * Y
    w = -2.0 * a * scale * (np.exp(Z / scale) - 1.0)
    ds = xr.Dataset(coords={"z": z, "y": y, "x": x, "radar": [0, 1]})
    ds["radar_x"] = ("radar", [-25e3, 25e3])
    ds["radar_y"] = ("radar", [-20e3, -20e3])
    ds["radar_altitude"] = ("radar", [0.0, 0.0])
    geo = radar_geometry(ds)
    el, az = np.radians(geo.elevation.values), np.radians(geo.azimuth.values)
    ds["VRADH"] = (
        ("radar", "z", "y", "x"),
        np.cos(el) * (np.sin(az) * u + np.cos(az) * v) + np.sin(el) * w,
    )
    rho = 1.2 * np.exp(-Z / scale)
    bg = xr.Dataset(
        {
            "u": (("z", "y", "x"), np.full(u.shape, 5.0)),
            "v": (("z", "y", "x"), np.full(u.shape, 2.0)),
            "air_density": (("z", "y", "x"), rho),
        },
        coords={"z": z, "y": y, "x": x},
    )
    return ds, bg, w


def test_w_bottom_above_the_ground():
    kw = dict(fall_speed_correction=False, tolerance=1e-6)

    def error(out, w):
        mask = (out.beam_crossing_angle > 30).values
        return float(np.sqrt(np.mean((out.w.values - w)[mask] ** 2)))

    ds, bg, w = _anelastic_case(0.0)
    assert error(multi_doppler(ds, bg, **kw), w) < 0.1  # 0.04 m/s
    ds, bg, w = _anelastic_case(500.0)
    with pytest.warns(RuntimeWarning, match="w = 0 at z = 500"):
        wrong = multi_doppler(ds, bg, **kw)
    assert error(wrong, w) > 0.3  # the true w at 500 m is -0.2 m/s
    w_500 = float(w[0, 0, 0])
    assert w_500 == pytest.approx(-0.205, abs=1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        right = multi_doppler(ds, bg, w_bottom=w_500, **kw)
    assert error(right, w) < 0.1
    np.testing.assert_allclose(right.w.isel(z=0), w_500, atol=1e-6)
    # a field on (y, x) works too, and so does the other solver
    field = xr.DataArray(
        np.full((ds.sizes["y"], ds.sizes["x"]), w_500), dims=("y", "x")
    )
    lb = multi_doppler(
        ds,
        bg,
        w_bottom=field,
        solver="lbfgsb",
        levels=1,
        max_iterations=300,
        fall_speed_correction=False,
    )
    np.testing.assert_allclose(lb.w.isel(z=0), w_500, atol=1e-6)
    with pytest.raises(ValueError, match="w_bottom"):
        multi_doppler(ds, bg, w_bottom=np.zeros((2, 2)), **kw)


def _translating_case(c=(20.0, 10.0), seconds=180.0):
    """Beltrami flow translating with ``c``, observed at two different times."""
    step, nz = 2000.0, 11
    x = np.arange(-40e3, 40e3 + 1, step)
    y = np.arange(-40e3, 40e3 + 1, step)
    z = np.linspace(0.0, 10e3, nz)
    radars = ((-25e3, -20e3), (25e3, -20e3))
    t_a = np.datetime64("2022-03-31T00:00:00", "ns")
    lag = (
        np.array([-seconds, seconds]).astype("timedelta64[s]").astype("timedelta64[ns]")
    )
    ds = xr.Dataset(coords={"z": z, "y": y, "x": x, "radar": [0, 1]})
    ds["radar_x"] = ("radar", [r[0] for r in radars])
    ds["radar_y"] = ("radar", [r[1] for r in radars])
    ds["radar_altitude"] = ("radar", np.zeros(2))
    geo = radar_geometry(ds)
    el, az = np.radians(geo.elevation.values), np.radians(geo.azimuth.values)
    vr = np.empty(el.shape)
    for k, dt in enumerate([-seconds, seconds]):
        u, v, w, _ = beltrami(x - c[0] * dt, y - c[1] * dt, z)  # pattern at t_k
        vr[k] = np.cos(el[k]) * (np.sin(az[k]) * u + np.cos(az[k]) * v)
        vr[k] += np.sin(el[k]) * w
    dist = np.hypot(ds.x - ds.radar_x, ds.y - ds.radar_y).transpose("radar", "y", "x")
    ds["VRADH"] = (
        ("radar", "z", "y", "x"),
        np.where((dist < 60e3).values[:, None], vr, np.nan),
    )
    ds = ds.assign_coords(time=("radar", t_a + lag))
    u, v, w, rho = beltrami(x, y, z)
    bg = xr.Dataset(
        {
            "u": (("z", "y", "x"), np.full(u.shape, 8.0)),
            "v": (("z", "y", "x"), np.full(u.shape, 4.0)),
            "air_density": (("z", "y", "x"), rho),
        },
        coords={"z": z, "y": y, "x": x},
    )
    truth = xr.Dataset({n: (("z", "y", "x"), a) for n, a in zip("uvw", (u, v, w))})
    return ds, bg, truth, t_a


def test_shift_winds_beats_advecting_the_radial_velocity():
    from radarx.retrieve import advect

    ds, bg, truth, t_a = _translating_case()
    c = (20.0, 10.0)
    kw = dict(fall_speed_correction=False, tolerance=1e-6)
    # radial velocity moved to the analysis time as a scalar
    moved = []
    for k in range(2):
        one = ds[["VRADH"]].isel(radar=k, drop=True).assign(time=ds.time.values[k])
        moved.append(advect(one, *c, time=t_a)["VRADH"])
    old = ds.assign(VRADH=xr.concat(moved, "radar").transpose("radar", "z", "y", "x"))
    old = radar_geometry(old.drop_vars("time"))
    scalar = multi_doppler(old, bg, **kw)
    shifted = multi_doppler(
        ds, bg, shift_winds=True, analysis_time=t_a, storm_motion=c, **kw
    )
    mask = lobes(shifted)
    e_old = rms_error(scalar, truth, mask)
    e_new = rms_error(shifted, truth, mask)
    for q in "uvw":
        assert e_new[q] < 0.7 * e_old[q], (q, e_old, e_new)
    assert e_new["u"] < 0.3 and e_new["v"] < 0.5
    # default off: no shift, the data are taken at face value
    plain = multi_doppler(ds, bg, **kw)
    assert rms_error(plain, truth, mask)["u"] > 1.0


def test_shift_winds_errors():
    ds, bg, _, t_a = _translating_case()
    with pytest.raises(ValueError, match="analysis_time"):
        multi_doppler(ds, bg, shift_winds=True, storm_motion=(10.0, 0.0))
    with pytest.raises(ValueError, match="observation time"):
        multi_doppler(
            ds.drop_vars("time"),
            bg,
            shift_winds=True,
            analysis_time=t_a,
            storm_motion=(10.0, 0.0),
        )


def test_input_places_radars_by_geodesic_distance():
    """Issue 170: radars 100 km either side of the origin, 35 degrees north."""
    from tests.grid import test_multi as tm

    east = tm._volume(*tm._site(100e3, 0.0), name="E")
    west = tm._volume(*tm._site(-100e3, 0.0), name="W")
    x = np.arange(-40e3, 40e3 + 1, 4e3)
    y = np.arange(-40e3, 40e3 + 1, 4e3)
    z = np.arange(1000.0, 4000.1, 500.0)
    out = multi_doppler_input(
        [east, west], x, y, z, origin=tm.ORIGIN, velocity="DBZH", reflectivity="RHOHV"
    )
    expected = tm.truth(x[None, None, :], y[None, :, None], z[:, None, None])
    for k in range(2):
        err = out.DBZH.isel(radar=k).values - expected
        assert np.sqrt(np.nanmean(err**2)) < 1e-3  # 0.044 dBZ (400 m) before
    assert out.azimuth.dims == ("radar", "z", "y", "x")
    np.testing.assert_allclose(out.radar_x, [100e3, -100e3], atol=1.0)


# --------------------------------------------------------------------------
# analytic dual-Doppler, O'Brien adjustment, observation weights, error estimate
# --------------------------------------------------------------------------

OBRIEN_DOI = "10.1175/1520-0450(1970)009<0197:ASTTCV>2.0.CO;2"


def _columns(seed=0, nz=6, ny=3, nx=4):
    rng = np.random.default_rng(seed)
    z = np.cumsum(rng.uniform(300.0, 700.0, nz)) - 400.0  # uneven levels
    flux = rng.normal(scale=1e-3, size=(nz, ny, nx))
    rho = 1.2 * np.exp(-z[:, None, None] / 1e4) * rng.uniform(0.9, 1.1, (nz, ny, nx))
    return z, flux, rho, rng.normal(scale=0.2, size=(ny, nx))


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("power", [1.0, 2.0, 1.5])
def test_integrate_mass_engines_agree(mode, power):
    z, flux, rho, wb = _columns()
    flux[2, 1, 1] = np.nan
    ref = md._integrate_mass(flux, rho, z, wb, mode, power, False, None)
    if md.HAS_COMPILED_KERNEL:
        for threads in (1, 4):
            got = md._integrate_mass(flux, rho, z, wb, mode, power, True, threads)
            np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14, equal_nan=True)
    gap = np.isnan(ref[:, 1, 1])
    if mode == 0:  # integrating upward, the gap spoils the levels above it
        assert gap.tolist() == [False, False, True, True, True, True]
    elif mode == 1:
        assert gap.tolist() == [True, True, True, False, False, False]
    else:
        assert gap.all()
    assert np.isfinite(np.delete(ref.reshape(len(z), -1), 5, axis=1)).all()


def test_integrate_mass_follows_the_analytic_anelastic_profile():
    z = np.linspace(0.0, 10e3, 101)
    ztop, scale = 10e3, 10e3
    rho = (1.2 * np.exp(-z / scale))[:, None, None] * np.ones((1, 2, 2))
    amp = 3.0
    mass = amp * 1.2 * np.exp(-z / scale) * np.sin(np.pi * z / ztop)
    w = (mass / (1.2 * np.exp(-z / scale)))[:, None, None] * np.ones((1, 2, 2))
    flux = (
        -amp
        * 1.2
        * np.exp(-z / scale)
        * (-np.sin(np.pi * z / ztop) / scale + np.pi / ztop * np.cos(np.pi * z / ztop))
    )[:, None, None] * np.ones((1, 2, 2))
    zero = np.zeros((2, 2))
    for mode in (0, 1, 2):
        for compiled in (False, True) if md.HAS_COMPILED_KERNEL else (False,):
            got = md._integrate_mass(flux, rho, z, zero, mode, 2.0, compiled, None)
            # trapezoid rule on 100 m levels; w is zero at both ends of the profile
            np.testing.assert_allclose(got, w, atol=2e-3)
    # a bottom value shifts the upward integral by rho(z0) w0 / rho(z)
    shifted = md._integrate_mass(flux, rho, z, zero + 1.0, 0, 2.0, False, None)
    np.testing.assert_allclose(shifted - w, (rho[0] / rho)[:, :, :] * 1.0, atol=2e-3)


@pytest.mark.parametrize("kind,power", [("linear", 2.0), ("constant", 1.0)])
def test_obrien_adjust_zeroes_the_top(kind, power):
    z, _, rho, _ = _columns(1)
    w = np.random.default_rng(2).normal(size=rho.shape)
    w[:, 0, 0] = np.nan  # a column without data stays as it is
    out = md._obrien_adjust(w, rho, z, kind, False, None)
    np.testing.assert_allclose(out[-1][np.isfinite(w[-1])], 0.0, atol=1e-14)
    np.testing.assert_allclose(out[0, 1:], w[0, 1:])  # lowest level unchanged
    s = ((z - z[0]) / (z[-1] - z[0])) ** power
    want = (rho * w - s[:, None, None] * (rho * w)[-1]) / rho
    np.testing.assert_allclose(out[:, 1:], want[:, 1:], rtol=1e-12)
    assert np.isnan(out[:, 0, 0]).all()
    if md.HAS_COMPILED_KERNEL:
        got = md._obrien_adjust(w, rho, z, kind, True, 2)
        np.testing.assert_allclose(got, out, rtol=1e-12, atol=1e-14, equal_nan=True)
    with pytest.raises(ValueError, match="obrien_error"):
        md._obrien_adjust(w, rho, z, "cubic", False, None)


@needs_kernel
def test_new_kernels_reject_bad_shapes():
    z, flux, rho, wb = _columns()
    k = md._multidoppler
    with pytest.raises(ValueError, match="flux_div"):
        k.integrate_mass(flux[0], rho, z, wb, 0)
    with pytest.raises(ValueError, match="two levels"):
        k.integrate_mass(flux[:1], rho[:1], z[:1], wb, 0)
    with pytest.raises(ValueError, match="rho"):
        k.integrate_mass(flux, rho[:2], z, wb, 0)
    with pytest.raises(ValueError, match="mode"):
        k.integrate_mass(flux, rho, z, wb, 3)
    with pytest.raises(ValueError, match="w must be"):
        k.obrien_adjust(flux[0], rho, z)
    with pytest.raises(ValueError, match="two levels"):
        k.obrien_adjust(flux[:1], rho[:1], z[:1])
    with pytest.raises(ValueError, match="z has"):
        k.obrien_adjust(flux, rho, z[:2])
    nr, shape = 2, (3, 4, 5)
    coef = np.zeros((nr * 3,) + shape)
    with pytest.raises(ValueError, match="target"):
        k.horizontal_wind(
            coef, np.zeros(shape), np.zeros((nr,) + shape), np.zeros(shape), 1e-8
        )
    with pytest.raises(ValueError, match="coef"):
        k.horizontal_wind(
            coef[:3],
            np.zeros((nr,) + shape),
            np.zeros((nr,) + shape),
            np.zeros(shape),
            1e-8,
        )


def _uniform_flow_case(u0=8.0, v0=-3.0, step=4000.0, nz=5, seed=None, sigma=0.0):
    ds, _, _ = synthetic_case(step=step, nz=nz)
    geo = radar_geometry(ds)
    el, az = np.radians(geo.elevation.values), np.radians(geo.azimuth.values)
    vr = np.cos(el) * (np.sin(az) * u0 + np.cos(az) * v0)
    if seed is not None:
        vr = vr + sigma * np.random.default_rng(seed).normal(size=vr.shape)
    return geo.assign(VRADH=(("radar", "z", "y", "x"), vr)), geo


@pytest.mark.parametrize("engine", ENGINES)
def test_analytic_horizontal_wind_and_geometric_error(engine):
    ds, geo = _uniform_flow_case()
    sigma = 0.7
    out = dual_doppler_analytic(
        ds,
        fall_speed_correction=False,
        obs_error=sigma,
        iterations=1,
        min_crossing_angle=30.0,
        engine=engine,
    )
    ok = np.isfinite(out.u.values)
    assert ok.sum() > 1500
    # no vertical velocity, no noise: the 2 x 2 system returns the wind exactly
    np.testing.assert_allclose(out.u.values[ok], 8.0, atol=1e-4)
    np.testing.assert_allclose(out.v.values[ok], -3.0, atol=1e-4)
    # only cells seen by two radars at 30 degrees or more are solved
    beta = out.beam_crossing_angle.values
    assert (beta[ok] >= 30.0).all() and (out.n_radars.values[ok] == 2).all()
    assert (~ok[(beta < 30.0)]).all()
    # sigma_u^2 + sigma_v^2 = sigma^2 (cos^2 p1 + cos^2 p2) / (cos^2 p1 cos^2 p2 sin^2 b)
    el = np.radians(geo.elevation.values)
    az = np.radians(geo.azimuth.values)
    c1, c2 = np.cos(el[0]) ** 2, np.cos(el[1]) ** 2
    sin_b = np.sin(az[0] - az[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        want = sigma**2 * (c1 + c2) / (c1 * c2 * sin_b**2)
    got = out.u_error.values**2 + out.v_error.values**2
    np.testing.assert_allclose(got[ok], want[ok], rtol=2e-4)
    # beams crossing at 90 degrees at ground level: sqrt(2) sigma
    right = ok & (np.abs(beta - 90.0) < 1.0) & (el[0] < 0.05) & (el[1] < 0.05)
    assert right.any()
    np.testing.assert_allclose(np.sqrt(got[right]), np.sqrt(2.0) * sigma, rtol=2e-3)
    assert out.u_error.attrs["units"] == "m s-1"


def test_analytic_horizontal_error_with_unequal_radar_errors():
    ds, geo = _uniform_flow_case()
    errors = [0.5, 2.0]
    out = dual_doppler_analytic(
        ds, fall_speed_correction=False, obs_error=errors, iterations=1
    )
    el, az = np.radians(geo.elevation.values), np.radians(geo.azimuth.values)
    a, b = np.cos(el) * np.sin(az), np.cos(el) * np.cos(az)
    wts = 1.0 / np.array(errors)[:, None, None, None] ** 2
    n11, n12, n22 = (wts * a * a).sum(0), (wts * a * b).sum(0), (wts * b * b).sum(0)
    det = n11 * n22 - n12**2
    ok = np.isfinite(out.u.values)
    with np.errstate(divide="ignore", invalid="ignore"):
        var_u, var_v = n22 / det, n11 / det
    np.testing.assert_allclose(out.u_error.values[ok] ** 2, var_u[ok], rtol=2e-4)
    np.testing.assert_allclose(out.v_error.values[ok] ** 2, var_v[ok], rtol=2e-4)
    # the noisier radar makes the error larger than for two radars of 0.5 m/s
    equal = dual_doppler_analytic(
        ds, fall_speed_correction=False, obs_error=0.5, iterations=1
    )
    assert (out.u_error.values[ok] > equal.u_error.values[ok]).all()


@pytest.mark.parametrize("engine", ENGINES)
def test_analytic_solver_retrieves_the_beltrami_flow(engine):
    ds, bg, truth = synthetic_case(noise=0.0)
    obrien = dual_doppler_analytic(
        ds,
        bg,
        fall_speed_correction=False,
        boundary="obrien",
        iterations=6,
        engine=engine,
    )
    good = (obrien.beam_crossing_angle > 30).values & np.isfinite(obrien.w.values)
    err = rms_error(obrien, truth, good)
    # exact radial velocities: u is exact, v and w carry the discretisation
    # error of the coupled horizontal-wind and continuity iteration
    assert err["u"] < 0.05 and err["v"] < 1.0 and err["w"] < 0.8
    assert float(abs(obrien.w.isel(z=-1)).max()) < 1e-6  # w = 0 at the top
    assert (
        obrien.w.isel(z=0).values[np.isfinite(obrien.w.isel(z=0).values)] == 0
    ).all()
    assert obrien.attrs["obrien_error"] == "linear"
    assert obrien.attrs["w_change_last_iteration_rms"] < 0.05
    # the Beltrami flow is zero at both ends: the downward integral works too,
    # the upward one lets the divergence error accumulate with height
    down = dual_doppler_analytic(
        ds,
        bg,
        fall_speed_correction=False,
        boundary="downward",
        iterations=6,
        engine=engine,
    )
    assert float(abs(down.w.isel(z=-1)).max()) == 0.0
    err_down = rms_error(down, truth, good)
    up = dual_doppler_analytic(
        ds, bg, fall_speed_correction=False, iterations=1, engine=engine
    )
    err_up = rms_error(up, truth, good)
    assert err_down["w"] < 1.5 and err_up["w"] > err["w"]
    assert err["w"] < err_up["w"]


@needs_kernel
def test_analytic_solver_engines_agree():
    ds, bg, _ = synthetic_case(noise=0.5)
    a = dual_doppler_analytic(ds, bg, fall_speed_correction=False, boundary="obrien")
    b = dual_doppler_analytic(
        ds, bg, fall_speed_correction=False, boundary="obrien", engine="numpy"
    )
    for c in ("u", "v", "w", "u_error", "v_error"):
        np.testing.assert_allclose(a[c], b[c], rtol=1e-4, atol=1e-5, equal_nan=True)


def test_analytic_solver_is_validated_against_the_variational_solution():
    ds, bg, truth = synthetic_case(noise=0.5)
    var = multi_doppler(
        ds, bg, fall_speed_correction=False, w_boundary=("bottom", "obrien")
    )
    ana = dual_doppler_analytic(ds, bg, fall_speed_correction=False, boundary="obrien")
    good = lobes(var) & np.isfinite(ana.w.values)
    e_var, e_ana = rms_error(var, truth, good), rms_error(ana, truth, good)
    # the variational solution is closer to the truth in every component
    for c in "uvw":
        assert e_var[c] < e_ana[c], (c, e_var, e_ana)
    # and the two agree on the horizontal wind to the noise level
    assert float(np.sqrt(np.mean((var.u.values - ana.u.values)[good] ** 2))) < 1.0


def test_analytic_solver_fall_speed_and_options():
    ds, bg, truth = synthetic_case(dbz=45.0)
    on = dual_doppler_analytic(ds, bg, freezing_level="rain", boundary="obrien")
    off = dual_doppler_analytic(ds, bg, fall_speed_correction=False, boundary="obrien")
    good = (on.beam_crossing_angle > 30).values & np.isfinite(on.w.values)
    assert float(on.fall_speed.max()) > 7.0
    # ignoring the fall speed biases the horizontal wind (5.6 m/s against 0.5 m/s)
    assert rms_error(on, truth, good)["v"] < 0.3 * rms_error(off, truth, good)["v"]
    # w_bottom: a number or a field on (y, x), shifting the upward integral
    base = dual_doppler_analytic(ds, bg, fall_speed_correction=False, iterations=1)
    field = xr.DataArray(np.full((ds.sizes["y"], ds.sizes["x"]), 0.5), dims=("y", "x"))
    moved = dual_doppler_analytic(
        ds, bg, fall_speed_correction=False, iterations=1, w_bottom=field
    )
    ok = np.isfinite(moved.w.values) & np.isfinite(base.w.values)
    assert (moved.w.isel(z=0).values[ok[0]] == 0.5).all()
    assert (moved.w.values[ok] > base.w.values[ok] - 1e-3).mean() > 0.9
    # a single radar gives no wind; a list of grids is concatenated
    one = dual_doppler_analytic(ds.isel(radar=[0]), bg, fall_speed_correction=False)
    assert np.isnan(one.u).all() and "beam_crossing_angle" not in one
    parts = [ds.isel(radar=[i]) for i in range(2)]
    both = dual_doppler_analytic(parts, bg, fall_speed_correction=False, iterations=1)
    np.testing.assert_allclose(both.u, base.u, equal_nan=True)


def test_analytic_solver_errors():
    ds, bg, _ = synthetic_case()
    kw = dict(fall_speed_correction=False)
    with pytest.raises(ValueError, match="boundary"):
        dual_doppler_analytic(ds, boundary="sideways", **kw)
    with pytest.raises(ValueError, match="obrien_error"):
        dual_doppler_analytic(ds, boundary="obrien", obrien_error="cubic", **kw)
    with pytest.raises(ValueError, match="iterations"):
        dual_doppler_analytic(ds, iterations=0, **kw)
    with pytest.raises(ValueError, match="w_bottom must be a number"):
        dual_doppler_analytic(ds, w_bottom=np.zeros((2, 2)), **kw)
    with pytest.raises(ValueError, match="w_bottom must be finite"):
        dual_doppler_analytic(ds, w_bottom=np.nan, **kw)
    with pytest.raises(ValueError, match="engine"):
        dual_doppler_analytic(ds, engine="fortran", **kw)
    with pytest.raises(ValueError, match="positive"):
        dual_doppler_analytic(ds, obs_error=-1.0, **kw)


def test_analytic_accessor_provenance_and_unchanged_input():
    ds, bg, _ = synthetic_case(noise=0.5)
    obs = xr.DataArray(
        np.full(ds.VRADH.shape, 0.5), dims=ds.VRADH.dims, coords=ds.VRADH.coords
    )
    before, before_bg, before_obs = (
        ds.copy(deep=True),
        bg.copy(deep=True),
        obs.copy(deep=True),
    )
    out = ds.radarx.dual_doppler_analytic(
        bg, fall_speed_correction=False, boundary="obrien", obs_error=obs
    )
    xr.testing.assert_identical(ds, before)
    xr.testing.assert_identical(bg, before_bg)
    xr.testing.assert_identical(obs, before_obs)
    # (u and v are variables of the background too, which the decorator treats as inputs)
    assert OBRIEN_DOI in out.w.attrs["radarx_references"]
    assert OBRIEN_DOI in out.attrs["radarx_references"]
    assert out.attrs["radarx_method"].startswith("Analytic dual-Doppler")
    assert "boundary='obrien'" in out.attrs["history"]
    # the O'Brien adjustment of the variational retrieval cites the paper too
    var = ds.radarx.multi_doppler(
        bg,
        fall_speed_correction=False,
        w_boundary=("bottom", "obrien"),
        max_iterations=3,
    )
    assert OBRIEN_DOI in var.attrs["radarx_references"]
    assert var.attrs["obrien_error"] == "linear"


# ---- observation weights --------------------------------------------------


def _prepared(ds, bg=None, **kw):
    weights = dict(md.DEFAULT_WEIGHTS)
    weights["smoothness_vertical"] = weights["smoothness"]
    args = dict(
        velocity="VRADH",
        reflectivity="DBZH",
        use_fall_speed=False,
        weights=weights,
        storm_motion=(0.0, 0.0),
        w_boundary="bottom",
    )
    args.update(kw)
    return md._prepare(md._as_multi_radar(ds, "VRADH"), bg, **args)


@pytest.fixture(scope="module")
def small_case():
    return synthetic_case(step=4000.0, nz=6, noise=0.5)


def test_observation_error_sets_the_weights(small_case):
    ds, bg, _ = small_case
    plain = _prepared(ds, bg)
    seen = plain["weight"] > 0
    np.testing.assert_array_equal(plain["weight"][seen], 1.0)
    two = _prepared(ds, bg, obs_error=2.0)
    np.testing.assert_allclose(two["weight"][seen], 0.25)
    np.testing.assert_allclose(two["co"], 0.25)
    per_radar = _prepared(ds, bg, obs_error=[1.0, 4.0])
    np.testing.assert_allclose(per_radar["weight"][0][seen[0]], 1.0)
    np.testing.assert_allclose(per_radar["weight"][1][seen[1]], 1.0 / 16.0)
    # a DataArray on (z, y, x) or on (radar, z, y, x), and a NaN removes the observation
    field = xr.DataArray(
        np.full((ds.sizes["z"], ds.sizes["y"], ds.sizes["x"]), 0.5),
        dims=("z", "y", "x"),
    )
    np.testing.assert_allclose(_prepared(ds, bg, obs_error=field)["weight"][seen], 4.0)
    full = xr.DataArray(np.full(ds.VRADH.shape, 0.5), dims=ds.VRADH.dims)
    full[1, 2, 3, 4] = np.nan
    dropped = _prepared(ds, bg, obs_error=full)
    assert dropped["weight"][1, 2, 3, 4] == 0.0
    assert dropped["n_obs"][2, 3, 4] == plain["n_obs"][2, 3, 4] - int(seen[1, 2, 3, 4])
    with pytest.raises(ValueError, match="positive"):
        _prepared(ds, bg, obs_error=0.0)
    with pytest.raises(ValueError, match="broadcasts"):
        _prepared(ds, bg, obs_error=np.ones(5))


def test_crossing_angle_weight_is_sin_squared(small_case):
    ds, bg, _ = small_case
    plain = _prepared(ds, bg)
    w = _prepared(ds, bg, crossing_angle_weight=True)
    seen = plain["weight"] > 0
    geo = md._as_multi_radar(ds, "VRADH")
    beta = md._crossing_angle(geo, seen)
    two = np.isfinite(beta)
    want = np.maximum(np.sin(np.radians(beta)) ** 2, md._CROSSING_FLOOR)
    np.testing.assert_allclose(w["weight"][0][two & seen[0]], want[two & seen[0]])
    one = seen[0] & ~two  # cells of one radar keep the weight 1
    assert one.any() and (w["weight"][0][one] == 1.0).all()
    assert w["co"] < 1.0 and w["co"] > 0.0
    # one radar: no crossing angle, nothing changes
    single = _prepared(ds.isel(radar=[0]), bg, crossing_angle_weight=True)
    assert (single["weight"][single["weight"] > 0] == 1.0).all()


def test_defaults_of_multi_doppler_are_unchanged(small_case):
    ds, bg, _ = small_case
    kw = dict(fall_speed_correction=False)
    base = multi_doppler(ds, bg, **kw)
    explicit = multi_doppler(
        ds,
        bg,
        obs_error=None,
        crossing_angle_weight=False,
        obrien_error="linear",
        error=False,
        **kw,
    )
    unit = multi_doppler(ds, bg, obs_error=1.0, **kw)
    for other in (explicit, unit):
        for name in ("u", "v", "w", "vr_residual"):
            np.testing.assert_array_equal(base[name].values, other[name].values)
        # the cost is summed over threads in a varying order
        np.testing.assert_allclose(base.cost.values, other.cost.values, rtol=1e-12)
    assert sorted(base.data_vars) == sorted(
        [
            "u", "v", "w", "fall_speed", "mass_residual", "n_radars",
            "vr_residual", "cost", "cost_history", "beam_crossing_angle",
        ]
    )  # fmt: skip
    assert "obrien_error" not in base.attrs and "error_method" not in base.attrs
    # obs_error = 2 is the observation weight 1/4
    quarter = multi_doppler(ds, bg, weights={"observation": 0.25}, **kw)
    twice = multi_doppler(ds, bg, obs_error=2.0, **kw)
    np.testing.assert_array_equal(quarter.w.values, twice.w.values)


def test_noisy_radar_gets_less_weight():
    ds, bg, truth = synthetic_case(step=4000.0, nz=6, noise=0.0)
    rng = np.random.default_rng(5)
    errors = [0.2, 1.0]  # the weights of the other terms are tuned for ~1 m/s
    sig = np.array(errors)[:, None, None, None]
    noisy = ds.assign(VRADH=ds.VRADH + sig * rng.normal(size=ds.VRADH.shape))
    kw = dict(fall_speed_correction=False)
    flat = multi_doppler(noisy, bg, **kw)
    weighted = multi_doppler(noisy, bg, obs_error=errors, **kw)
    m = lobes(flat)
    e_flat, e_w = rms_error(flat, truth, m), rms_error(weighted, truth, m)
    # trusting the good radar more reduces the error (0.54, 0.83, 0.75 against
    # 0.77, 1.13, 0.99 m/s)
    assert all(e_w[c] < 0.85 * e_flat[c] for c in "uvw"), (e_flat, e_w)


def test_crossing_angle_weight_runs_and_changes_the_solution(small_case):
    ds, bg, truth = small_case
    kw = dict(fall_speed_correction=False)
    plain = multi_doppler(ds, bg, **kw)
    weighted = multi_doppler(ds, bg, crossing_angle_weight=True, **kw)
    assert not np.allclose(plain.w, weighted.w)
    m = lobes(plain)
    # not worse than a factor of 1.5 on the well-crossing cells
    e0, e1 = rms_error(plain, truth, m), rms_error(weighted, truth, m)
    assert all(e1[c] < 1.5 * e0[c] for c in "uvw")
    assert weighted.attrs["converged"] == 1


# ---- O'Brien adjustment of the variational solution -------------------------


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("kind", ["linear", "constant"])
def test_variational_obrien_adjustment(small_case, engine, kind):
    ds, bg, truth = small_case
    kw = dict(fall_speed_correction=False, engine=engine)
    plain = multi_doppler(ds, bg, w_boundary="bottom", **kw)
    adj = multi_doppler(
        ds, bg, w_boundary=("bottom", "obrien"), obrien_error=kind, **kw
    )
    assert float(abs(adj.w.isel(z=-1)).max()) < 1e-6
    assert (adj.w.isel(z=0) == 0).all()
    np.testing.assert_array_equal(adj.u.values, plain.u.values)
    np.testing.assert_array_equal(adj.v.values, plain.v.values)
    rho = bg.air_density.values
    z = ds.z.values
    want = md._obrien_adjust(plain.w.values.astype(float), rho, z, kind, False, None)
    np.testing.assert_allclose(adj.w.values, want, atol=2e-6)
    assert adj.attrs["obrien_error"] == kind
    # the Beltrami flow is zero at the top: the adjustment brings w closer to it
    m = lobes(plain)
    assert rms_error(adj, truth, m)["w"] < rms_error(plain, truth, m)["w"]
    assert float(adj.cost.sum()) > float(plain.cost.sum())  # no longer the minimiser


def test_obrien_option_errors(small_case):
    ds, bg, _ = small_case
    with pytest.raises(ValueError, match="obrien_error"):
        multi_doppler(ds, bg, obrien_error="cubic")
    with pytest.raises(
        ValueError, match="w_boundary items must be bottom, top, echo_top or obrien"
    ):
        multi_doppler(ds, bg, w_boundary="sides")


# ---- error estimate ---------------------------------------------------------


def test_error_estimate_matches_the_inverse_hessian():
    ds, bg, _ = synthetic_case(step=20e3, nz=4, noise=0.5)
    p = _prepared(ds, bg, obs_error=0.5)
    free = p["free"]
    homogeneous = dict(p, target=np.zeros_like(p["target"]), bg=np.zeros_like(p["bg"]))
    n = free.size
    index = np.concatenate([np.arange(2 * n), 2 * n + np.flatnonzero(free.ravel())])
    h = np.empty((index.size, index.size))
    for col, k in enumerate(index):
        x = np.zeros(3 * n)
        x[k] = 1.0
        _, g = md._cost_gradient(homogeneous, x.reshape((3,) + free.shape), False, 0)
        h[:, col] = g.reshape(-1)[index]
    np.testing.assert_allclose(h, h.T, atol=1e-9)
    cov = 2.0 * np.linalg.inv(h)  # (H / 2)^-1 for J = sum of squares
    exact = np.zeros(3 * n)
    exact[index] = np.sqrt(np.diag(cov))
    exact = exact.reshape((3,) + free.shape)
    est = md._posterior_error(p, md.HAS_COMPILED_KERNEL, None, 600, 3, 2000, 1e-6)
    assert (est[2][~free] == 0).all() and (exact[2][~free] == 0).all()
    sel = exact > 0
    # 600 draws: the relative uncertainty of each standard deviation is 3 %
    rel = est[sel] / exact[sel]
    assert abs(np.sqrt(np.mean(est[sel] ** 2) / np.mean(exact[sel] ** 2)) - 1.0) < 0.03
    assert np.percentile(np.abs(rel - 1.0), 90) < 0.12


@pytest.mark.parametrize("engine", ENGINES)
def test_error_estimate_in_multi_doppler(small_case, engine):
    ds, bg, truth = small_case
    kw = dict(fall_speed_correction=False, obs_error=0.5, engine=engine)
    out = multi_doppler(ds, bg, error=True, error_samples=12, **kw)
    plain = multi_doppler(ds, bg, **kw)
    for c in "uvw":
        np.testing.assert_array_equal(out[c].values, plain[c].values)
        assert out[f"{c}_error"].dims == ("z", "y", "x")
        assert out[f"{c}_error"].attrs["units"] == "m s-1"
    assert (out.w_error.isel(z=0) == 0).all()  # fixed w has no error
    assert (out.w_error.isel(z=slice(1, None)) > 0).all()
    assert "Gauss-Markov" in out.attrs["error_method"]
    again = multi_doppler(ds, bg, error=True, error_samples=12, **kw)
    np.testing.assert_array_equal(again.w_error.values, out.w_error.values)
    other = multi_doppler(ds, bg, error=True, error_samples=12, error_seed=1, **kw)
    assert not np.array_equal(other.w_error.values, out.w_error.values)


def test_error_estimate_is_close_to_the_actual_error():
    """Beltrami flow on the 2 km grid with 0.5 m/s noise, beams crossing at > 30 deg."""
    sigma = 0.5
    ds, bg, truth = synthetic_case(noise=sigma)
    out = multi_doppler(
        ds,
        bg,
        fall_speed_correction=False,
        obs_error=sigma,
        error=True,
        error_samples=8,
    )
    m = lobes(out)
    ratio = {}
    for c in "uvw":
        actual = float(np.sqrt(np.mean((out[c].values - truth[c].values)[m] ** 2)))
        estimate = float(np.sqrt(np.mean(out[f"{c}_error"].values[m] ** 2)))
        ratio[c] = estimate / actual
    # 1.4, 1.3 and 1.0 with 32 draws: the estimate includes the spread allowed
    # by the weights of the smoothness and mass terms, the actual error is that
    # of a smooth field
    assert 0.7 < ratio["w"] < 1.6 and 0.7 < ratio["v"] < 2.0 and 0.7 < ratio["u"] < 2.2
    # less accurate where the beams cross at small angles
    err = out.v_error.values
    lo = (out.beam_crossing_angle.values > 30) & (out.beam_crossing_angle.values < 50)
    hi = out.beam_crossing_angle.values > 70
    assert err[lo].mean() > err[hi].mean()


def test_error_estimate_errors(small_case):
    ds, bg, _ = small_case
    with pytest.raises(ValueError, match="error_samples"):
        multi_doppler(ds, bg, error=True, error_samples=1)
    with pytest.raises(ValueError, match="quadratic"):
        multi_doppler(ds, bg, error=True, weights={"vorticity": 0.01}, solver="lbfgsb")
