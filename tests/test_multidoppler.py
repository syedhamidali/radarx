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

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.retrieve import multidoppler as md
from radarx.retrieve.multidoppler import (
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
    a = multi_doppler(ds, bg, fall_speed_correction=False, engine="compiled")
    b = multi_doppler(ds, bg, fall_speed_correction=False, engine="numpy")
    for c in "uvw":
        # rounding differences (summation order) change the CG iterates slightly
        np.testing.assert_allclose(a[c], b[c], atol=1e-2)
    np.testing.assert_allclose(float(a.cost.sum()), float(b.cost.sum()), rtol=1e-6)


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
    out = multi_doppler(ds, bg)
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
