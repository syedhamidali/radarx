#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests of radarx.retrieve.rain_collisions (collisions in a rain shaft)
====================================================================

The coalescence efficiencies, breakup fractions and fragment numbers are
compared with the tables of Low and List (1982a, b); the conservation of mass
of every event and of the rain rate of the shaft is checked to rounding, the
integration against SciPy, and the physical behaviour (coalescence grows the
drops, breakup makes small ones).
"""

import importlib

import numpy as np
import pytest
import xarray as xr
from scipy.integrate import solve_ivp

import radarx  # noqa: F401
from radarx.retrieve import _rain_collisions_numpy as ref
from radarx.retrieve import _raindrop_fall, rain_shaft

rc = importlib.import_module("radarx.retrieve.rain_collisions")
ENGINES = ["numpy"] + (["compiled"] if rc.HAS_COMPILED_KERNEL else [])

# (D_L, D_S) in cm and the collision kinetic energy in J, Table 2 of Part I
PAIRS = [
    (0.18, 0.0395, 3.219e-7),
    (0.40, 0.0395, 8.426e-7),
    (0.44, 0.0395, 8.781e-7),
    (0.18, 0.0715, 8.935e-7),
    (0.18, 0.10, 9.484e-7),
    (0.30, 0.10, 41.00e-7),
    (0.36, 0.10, 53.53e-7),
    (0.46, 0.10, 64.78e-7),
    (0.36, 0.18, 85.50e-7),
    (0.46, 0.18, 124.5e-7),
]
# the numbers calculated in the papers
E_COAL = [
    0.507,
    0.620,
    0.633,
    0.315,
    0.244,
    0.0842,
    0.0,
    0.0,
    0.0,
    0.0,
]  # Part I, Table 3
R_F = [1.0, 1.0, 1.0, 1.0, 0.96, 0.37, 0.31, 0.27, 0.23, 0.18]  # Part II, Table 2
R_S = [0.0, 0.0, 0.0, 0.18, 0.12, 0.53, 0.50, 0.34, 0.62, 0.61]
R_D = [0.0, 0.0, 0.0, 0.0, 0.0, 0.096, 0.19, 0.39, 0.15, 0.21]
F_F = [3.08, 3.08, 3.08, 4.55, 2.48, 6.21, 7.41, 8.44, 6.57, 7.27]  # Part II, Table 1
F_S = [None, None, None, 2.01, 2.17, 5.26, 7.94, 10.74, 9.15, 10.90]
F_D = [None, None, None, None, None, 3.41, 9.72, 14.24, 20.81, 29.70]
F_ALL = [
    2.02,
    1.79,
    1.76,
    3.67,
    2.24,
    5.06,
    8.11,
    11.47,
    10.33,
    14.20,
]  # Part II, Table 3


def pair_values(dl, ds, cke):
    dl_m, ds_m = dl * 1e-2, ds * 1e-2
    st = np.pi * ref.SIGMA * (dl_m**2 + ds_m**2)
    sc = np.pi * ref.SIGMA * (dl_m**3 + ds_m**3) ** (2.0 / 3.0)
    return st, sc, st - sc


def grid(n=24, lo=0.2, hi=7.0):
    d = np.geomspace(lo, hi, n)
    return d, _raindrop_fall.sea_level_speed(d)[0]


def marshall_palmer(d, rate=50.0):
    lam = 4.1 * rate**-0.21
    return xr.DataArray(
        8000.0 * np.exp(-lam * d),
        dims="diameter",
        coords={"diameter": ("diameter", d, {"units": "mm"})},
        name="N",
    )


# --------------------------------------------------------------------------
# the numbers of the paper
# --------------------------------------------------------------------------


def test_coalescence_efficiencies_of_part_one():
    for (dl, ds, cke), expected in zip(PAIRS, E_COAL):
        _, sc, dsig = pair_values(dl, ds, cke)
        assert ref.coalescence_efficiency(dl, ds, cke, sc, dsig) == pytest.approx(
            expected, abs=1.5e-3
        )


def test_breakup_fractions_of_part_two():
    for (dl, ds, cke), rf, rs, rd in zip(PAIRS, R_F, R_S, R_D):
        st, _, _ = pair_values(dl, ds, cke)
        raw = ref.breakup_fractions_raw(cke, st)
        np.testing.assert_allclose(raw, [rf, rs, rd], atol=0.016)
        got = ref.breakup_fractions(cke, st)
        assert sum(got) == pytest.approx(1.0, abs=1e-14)
        # rescaled only where the printed fractions exceed one (pair 4: 1.18)
        if sum(raw) <= 1.0 + 1e-12:
            np.testing.assert_allclose(got, raw, atol=1e-12)
        else:
            np.testing.assert_allclose(got, np.array(raw) / sum(raw), atol=1e-12)
    st = pair_values(0.18, 0.0715, 8.935e-7)[0]
    assert sum(ref.breakup_fractions_raw(8.935e-7, st)) == pytest.approx(1.18, abs=0.02)


def test_mean_fragment_numbers_of_part_two():
    for i, (dl, ds, cke) in enumerate(PAIRS):
        st, _, _ = pair_values(dl, ds, cke)
        ff, fs, fd = ref.mean_numbers(dl, ds, cke, st)
        assert ff == pytest.approx(F_F[i], abs=0.02)
        if F_S[i] is not None:
            assert fs == pytest.approx(F_S[i], abs=0.02)
        if F_D[i] is not None:
            # the paper's energies come from another fall-speed fit
            assert fd == pytest.approx(F_D[i], abs=0.15)


def test_total_fragment_numbers_of_part_two():
    for i, (dl, ds, cke) in enumerate(PAIRS):
        st, sc, dsig = pair_values(dl, ds, cke)
        ff, fs, fd = ref.mean_numbers(dl, ds, cke, st)
        rf, rs, rd = ref.breakup_fractions_raw(cke, st)  # as printed
        e = ref.coalescence_efficiency(dl, ds, cke, sc, dsig)
        total = (rf * ff + rs * fs + rd * fd) * (1.0 - e) + e
        assert total == pytest.approx(F_ALL[i], abs=0.06)


def test_filament_numbers_branches_cross():
    # F_f is the smaller of the two branches, so it is continuous at D_S0
    for dl in (0.18, 0.3, 0.46):
        ds0 = ref._crossing(dl)
        f1, f2 = ref._filament_numbers(dl, ds0)
        assert f1 == pytest.approx(f2, rel=1e-9)
        below = ref.mean_numbers(dl, 0.5 * ds0, 1e-6, 1e-6)[0]
        above = ref.mean_numbers(dl, 2.0 * ds0, 1e-6, 1e-6)[0]
        assert below == pytest.approx(ref._filament_numbers(dl, 0.5 * ds0)[1])
        assert above == pytest.approx(max(2.0, ref._filament_numbers(dl, 2.0 * ds0)[0]))


def test_collision_energies_with_the_radarx_fall_speed():
    # the paper used the Gunn-Kinzer speeds; radarx's law gives CKE within 16 %
    for dl, ds, cke in PAIRS:
        vl = float(_raindrop_fall.sea_level_speed(dl * 10.0)[0])
        vs = float(_raindrop_fall.sea_level_speed(ds * 10.0)[0])
        got = ref.energies(dl, ds, vl, vs)[0]
        assert got == pytest.approx(cke, rel=0.16)
    # CKE of Eq. (3.1) written out for one pair
    cke, st, sc, dsig = ref.energies(0.30, 0.10, 8.0, 4.0)
    d_l, d_s = 3e-3, 1e-3
    assert cke == pytest.approx(
        1000.0 * np.pi / 12.0 * d_l**3 * d_s**3 / (d_l**3 + d_s**3) * 16.0, rel=1e-12
    )
    assert st - sc == pytest.approx(dsig)
    assert st == pytest.approx(np.pi * 7.28e-2 * (d_l**2 + d_s**2))


def test_large_remnants_follow_the_mass():
    # the remnant is the drop the small fragments leave: near D_L for filaments
    for dl, ds, cke in PAIRS:
        rem = ref.remnants(dl, ds, cke)
        assert rem["f"][1] and 0.98 < rem["f"][0] / dl <= 1.0
    # sheets (where they occur) 2.5 to 17 % smaller, disks up to 5.4 % larger
    for (dl, ds, cke), rs in zip(PAIRS, R_S):
        rem = ref.remnants(dl, ds, cke)
        if rs > 0.1:
            assert rem["s"][1] and 0.82 < rem["s"][0] / dl < 0.98
        if rem["d"][1]:
            assert 0.995 < rem["d"][0] / dl < 1.055
    # the disk of the smallest pair has no log-normal solution within the limits
    assert not ref.remnants(0.18, 0.0395, 3.219e-7)["d"][1]


def test_filament_distribution_exists_for_every_pair():
    # the fallback of the other types: the large remnant of a filament takes at
    # least a tenth of the mass for every pair of 0.1 to 10 mm
    sizes = np.geomspace(0.01, 1.0, 45)
    for dl in sizes:
        for ds in sizes[sizes < dl]:
            rem = ref.remnants(dl, ds, 1e-6)
            assert rem["f"][1], (dl, ds)


# --------------------------------------------------------------------------
# the tensors
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("mode", [1, 2])
def test_every_event_conserves_mass(engine, mode):
    kernel = rc._kernel(engine)
    for d in (grid(24)[0], np.array([0.3, 0.5, 0.9, 1.4, 2.2, 3.1, 4.4, 6.0, 7.5])):
        v = _raindrop_fall.sea_level_speed(d)[0]
        k, g, e = kernel.build(d, v, mode, 0)
        cube = d**3
        err = np.abs(np.einsum("ijk,k->ij", g, cube)) / cube.max()
        assert err.max() < 1e-14
        # events exist only where the speeds differ (large drop first)
        assert np.all(g[np.triu_indices(d.size)] == 0.0)
        assert np.all(np.diag(k) == 0.0) and np.allclose(k, k.T)


@pytest.mark.parametrize("engine", ENGINES)
def test_event_tensor_properties(engine):
    kernel = rc._kernel(engine)
    d, v = grid(30)
    k, g, e = kernel.build(d, v, 2, 0)
    n = d.size
    ii, jj = np.tril_indices(n, -1)
    # the geometric sweep-out kernel
    expected = (
        0.25
        * np.pi
        * ((d[:, None] + d[None, :]) * 1e-3) ** 2
        * np.abs(v[:, None] - v[None, :])
    )
    np.testing.assert_allclose(k, expected, rtol=1e-12)
    # fragments (production) are never negative; the two losses are -1 each
    prod = g.copy()
    prod[ii, jj, ii] += 1.0
    prod[ii, jj, jj] += 1.0
    assert prod.min() > -1e-13
    # efficiencies are in [0, 1] and of Part I
    assert e.min() >= 0.0 and e.max() <= 0.78
    dl, ds = d[20] * 0.1, d[10] * 0.1
    cke, st, sc, dsig = ref.energies(dl, ds, v[20], v[10])
    assert e[20, 10] == pytest.approx(ref.coalescence_efficiency(dl, ds, cke, sc, dsig))
    # coalescing pairs: the number of drops falls by one and the mass is put on
    # the two pivots around it; coalescence only, every pair
    k1, g1, e1 = kernel.build(d, v, 1, 0)
    assert np.all(e1[ii, jj][k[ii, jj] > 0] == 1.0)
    inside = (d[ii] ** 3 + d[jj] ** 3) < d[-1] ** 3
    np.testing.assert_allclose(
        g1[ii, jj].sum(axis=1)[inside & (k[ii, jj] > 0)], -1.0, atol=1e-13
    )
    # no interactions: nothing happens
    k0, g0, e0 = kernel.build(d, v, 0, 0)
    assert not g0.any() and not e0.any() and np.allclose(k0, k)


@pytest.mark.parametrize("engine", ENGINES)
def test_pairs_without_speed_difference_do_not_collide(engine):
    kernel = rc._kernel(engine)
    d = np.array([1.0, 2.0, 3.0, 4.0])
    v = np.array([3.0, 5.0, 5.0, 7.0])
    k, g, e = kernel.build(d, v, 2, 0)
    assert k[2, 1] == 0.0 and not g[2, 1].any() and e[2, 1] == 0.0
    assert g[1, 0].any()


def test_engines_build_the_same_tensors():
    if not rc.HAS_COMPILED_KERNEL:
        pytest.skip("compiled kernel not built")
    d, v = grid(20, 0.15, 8.0)
    a = ref.build(d, v, 2)
    b = rc._rain_collisions.build(d, v, 2, 2)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, rtol=1e-11, atol=1e-13)
    a1 = ref.build(d, v, 1)
    b1 = rc._rain_collisions.build(d, v, 1, 0)
    for x, y in zip(a1, b1):
        np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-14)


def test_kernel_rejects_bad_arguments():
    if not rc.HAS_COMPILED_KERNEL:
        pytest.skip("compiled kernel not built")
    kernel = rc._rain_collisions
    d, v = grid(8)
    with pytest.raises(ValueError):
        kernel.build(d[::-1].copy(), v, 2, 0)
    with pytest.raises(ValueError):
        kernel.build(d, v[:-1], 2, 0)
    with pytest.raises(ValueError):
        kernel.build(d, v, 3, 0)
    with pytest.raises(ValueError):
        kernel.build(d[:1], v[:1], 2, 0)
    k, g, e = kernel.build(d, v, 2, 0)
    ok = dict(
        kernel=k,
        gain=g,
        v0=v,
        n0=np.ones((1, 8)),
        z_out=np.array([10.0]),
        dz_max=5.0,
        cfl=0.25,
        zs=np.array([0.0, 100.0]),
        ss=np.array([1.0, 1.0]),
    )
    assert kernel.evolve(**ok).shape == (1, 1, 8)
    bad = [
        dict(v0=np.zeros(8)),
        dict(z_out=np.array([10.0, 5.0])),
        dict(z_out=np.array([-1.0])),
        dict(zs=np.array([0.0, 0.0])),
        dict(ss=np.array([1.0, 0.0])),
        dict(ss=np.array([1.0])),
        dict(dz_max=0.0),
        dict(cfl=0.0),
        dict(n0=np.ones((1, 7))),
        dict(gain=g[:, :, :7].copy()),
        dict(z_out=np.ones((1, 1))),
    ]
    for change in bad:
        with pytest.raises(ValueError):
            kernel.evolve(**{**ok, **change})


# --------------------------------------------------------------------------
# the shaft
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("collisions", ["low_list1982", "coalescence", "none"])
def test_rain_rate_is_conserved(engine, collisions):
    d, _ = grid(20)
    nd = marshall_palmer(d)
    out = rain_shaft(
        nd, depth=[0.0, 500.0, 1500.0, 3000.0], collisions=collisions, engine=engine
    )
    rr = out.rain_rate.values
    np.testing.assert_allclose(rr, rr[0], rtol=1e-11)
    assert rr[0] == pytest.approx(55.0, rel=0.05)  # Marshall-Palmer of 50 mm/h
    assert float(out.dsd.min()) > -1e-9 * float(out.dsd.max())
    if collisions == "none":
        np.testing.assert_allclose(out.dsd.isel(depth=-1), nd, rtol=1e-12)
    else:
        assert not np.allclose(out.dsd.isel(depth=-1), nd, rtol=1e-3)


@pytest.mark.parametrize("engine", ENGINES)
def test_mass_flux_of_the_pivots_is_conserved_exactly(engine):
    d, v = grid(20)
    nd = marshall_palmer(d)
    out = rain_shaft(nd, depth=[0.0, 2000.0], engine=engine)
    width = rc._bin_widths(d)
    flux = (out.dsd * width * v * d**3).sum("diameter")
    assert float(flux.isel(depth=1) / flux.isel(depth=0) - 1.0) == pytest.approx(
        0.0, abs=1e-13
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_integration_agrees_with_scipy(engine):
    d, v0 = grid(16, 0.3, 6.0)
    kernel = rc._kernel(engine)
    k, g, _ = kernel.build(d, v0, 2, 0)
    width = rc._bin_widths(d)
    n0 = marshall_palmer(d, 30.0).values * width
    zs, ss = np.array([0.0, 5000.0]), np.array([1.0, 1.25])
    z_out = np.array([800.0, 2500.0])

    def f(z, flux):
        s = np.interp(z, zs, ss)
        conc = flux / (v0 * s)
        rate = s * np.tril(k, -1) * conc[:, None] * conc[None, :]
        return np.einsum("ij,ijk->k", rate, g)

    sol = solve_ivp(
        f,
        (0.0, z_out[-1]),
        n0 * v0 * ss[0],
        t_eval=z_out,
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
    )
    expected = sol.y.T / (v0 * np.interp(z_out, zs, ss)[:, None])
    got = np.asarray(kernel.evolve(k, g, v0, n0[None, :], z_out, 5.0, 0.25, zs, ss, 0))[
        :, 0, :
    ]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-8 * n0.max())
    # a smaller step changes the result by less
    fine = np.asarray(kernel.evolve(k, g, v0, n0[None, :], z_out, 1.0, 0.1, zs, ss, 0))[
        :, 0, :
    ]
    assert (
        np.abs(fine - expected).max() < np.abs(got - expected).max() + 1e-9 * n0.max()
    )


def test_engines_evolve_the_same():
    if not rc.HAS_COMPILED_KERNEL:
        pytest.skip("compiled kernel not built")
    d, _ = grid(24)
    nd = marshall_palmer(d)
    a = rain_shaft(nd, depth=[1000.0, 3000.0], engine="numpy")
    b = rain_shaft(nd, depth=[1000.0, 3000.0], engine="compiled", n_threads=2)
    np.testing.assert_allclose(a.dsd, b.dsd, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(a.rain_rate, b.rain_rate, rtol=1e-12)


@pytest.mark.parametrize("engine", ENGINES)
def test_coalescence_grows_the_drops_and_breakup_makes_small_ones(engine):
    d, v0 = grid(30)
    nd = marshall_palmer(d, 100.0)
    width = rc._bin_widths(d)
    runs = {
        name: rain_shaft(nd, depth=[0.0, 3000.0], collisions=name, engine=engine).isel(
            depth=1
        )
        for name in ("coalescence", "low_list1982", "none")
    }

    def mean_mass_diameter(out):
        n_k = out.dsd * width
        return float((n_k * d**4).sum() / (n_k * d**3).sum())

    top = mean_mass_diameter(runs["none"])
    # every collision coalesces: fewer, larger drops
    assert (
        runs["coalescence"].number_concentration
        < 0.8 * runs["none"].number_concentration
    )
    assert mean_mass_diameter(runs["coalescence"]) > 1.05 * top
    # breakup produces small drops: more drops than with coalescence only
    assert (
        runs["low_list1982"].number_concentration
        > runs["coalescence"].number_concentration
    )
    assert mean_mass_diameter(runs["low_list1982"]) < mean_mass_diameter(
        runs["coalescence"]
    )
    # drops of several mm are made by coalescence and then break up
    big = d > 4.0
    assert float(runs["coalescence"].dsd.where(big).sum()) > float(
        runs["low_list1982"].dsd.where(big).sum()
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_density_profile_scales_the_fall_speed(engine):
    d, v0 = grid(14, 0.3, 6.0)
    nd = marshall_palmer(d)
    z = np.arange(0.0, 6001.0, 500.0)
    rho = 1.204 * np.exp(-z / 8500.0)
    profile = xr.Dataset({"air_density": ("height", rho)}, coords={"height": z})
    out = rain_shaft(
        nd,
        depth=[0.0, 3000.0],
        collisions="none",
        profile=profile,
        top_height=5000.0,
        engine=engine,
    )
    # no collisions: the flux V N is conserved, N(z) = N(0) s(0) / s(z)
    s = lambda h: (1.204 / (1.204 * np.exp(-h / 8500.0))) ** 0.4
    expected = nd * s(5000.0) / s(2000.0)
    np.testing.assert_allclose(out.dsd.isel(depth=1), expected, rtol=1e-3)
    np.testing.assert_allclose(out.dsd.isel(depth=0), nd, rtol=1e-12)
    np.testing.assert_allclose(out.rain_rate, out.rain_rate[0], rtol=1e-10)
    # without the correction, the profile is not used
    plain = rain_shaft(
        nd, depth=3000.0, collisions="none", density_correction=False, engine=engine
    )
    np.testing.assert_allclose(plain.dsd.isel(depth=0), nd, rtol=1e-12)
    # pressure and temperature (with humidity) give the density; a standard atmosphere
    # is used if only the top is given
    p = 101325.0 * (1.0 - 2.25577e-5 * z) ** 5.25588
    t = 288.15 - 0.0065 * z
    q = 0.008 * np.exp(-z / 2500.0)
    thermo = xr.Dataset(
        {
            "pressure": ("height", p),
            "temperature": ("height", t),
            "specific_humidity": ("height", q),
        },
        coords={"height": z},
    )
    a = rain_shaft(
        nd,
        depth=[0.0, 3000.0],
        collisions="none",
        profile=thermo,
        top_height=5000.0,
        engine=engine,
    )
    dry = thermo.drop_vars("specific_humidity")
    b = rain_shaft(
        nd,
        depth=[0.0, 3000.0],
        collisions="none",
        profile=dry,
        top_height=5000.0,
        engine=engine,
    )
    assert float(a.dsd.isel(depth=1).sum()) != float(b.dsd.isel(depth=1).sum())
    std = rain_shaft(
        nd, depth=[0.0, 3000.0], collisions="none", top_height=5000.0, engine=engine
    )
    assert float(std.dsd.isel(depth=1).sum()) > float(std.dsd.isel(depth=0).sum())
    # collisions with the profile still conserve the rain rate
    col = rain_shaft(
        nd, depth=[0.0, 3000.0], profile=profile, top_height=5000.0, engine=engine
    )
    np.testing.assert_allclose(col.rain_rate, col.rain_rate[0], rtol=1e-10)


def test_columns_nan_and_dims():
    d, _ = grid(14, 0.3, 6.0)
    base = marshall_palmer(d)
    stack = xr.concat([base, 0.5 * base, base * np.nan, -base, 0.0 * base], dim="point")
    stack = (
        stack.assign_coords(point=np.arange(5))
        .expand_dims(time=[0, 1])
        .transpose("time", "diameter", "point")
    )
    out = rain_shaft(stack, depth=[0.0, 1000.0])
    assert out.dsd.dims == ("time", "point", "depth", "diameter")
    assert out.rain_rate.dims == ("time", "point", "depth")
    one = rain_shaft(base, depth=[0.0, 1000.0])
    np.testing.assert_allclose(out.dsd.isel(time=1, point=0), one.dsd)
    assert (
        np.isnan(out.dsd.isel(point=2)).all() and np.isnan(out.dsd.isel(point=3)).all()
    )
    assert (out.dsd.isel(point=4) == 0.0).all()
    np.testing.assert_allclose(
        out.rain_rate.isel(point=1, time=0), 0.5 * one.rain_rate, rtol=1e-6
    )
    # a single depth gives a length-one axis; zero depth returns the input
    zero = rain_shaft(base, depth=0.0)
    assert zero.dsd.dims == ("depth", "diameter") and zero.sizes["depth"] == 1
    np.testing.assert_allclose(zero.dsd.isel(depth=0), base, rtol=1e-12)


def test_inputs_are_not_changed_dask_is_lazy_and_provenance():
    dask = pytest.importorskip("dask.array")
    d, _ = grid(14, 0.3, 6.0)
    base = marshall_palmer(d)
    stack = xr.concat(
        [base * f for f in (1.0, 0.7, 1.4, 0.4)], dim="point"
    ).assign_coords(point=("point", np.arange(4), {"units": "1"}))
    before = stack.copy(deep=True)
    eager = rain_shaft(stack, depth=[500.0, 1500.0])
    xr.testing.assert_identical(stack, before)
    assert stack["diameter"].attrs == {"units": "mm"}
    assert eager.diameter.attrs == {"units": "mm"}
    lazy = rain_shaft(stack.chunk({"point": 2}), depth=[500.0, 1500.0])
    assert isinstance(lazy.dsd.data, dask.Array)
    np.testing.assert_allclose(lazy.dsd.compute(), eager.dsd, rtol=1e-12)
    # provenance of the products
    refs = eager.attrs["radarx_references"].split()
    assert "10.1175/1520-0469(1982)039<1591:CCABOR>2.0.CO;2" in refs
    assert "10.1175/1520-0469(1982)039<1607:CCABOR>2.0.CO;2" in refs
    assert "Low and List" in eager.attrs["radarx_method"]
    assert eager.dsd.attrs["units"] == "m-3 mm-1"
    assert eager.attrs["collisions"] == "low_list1982"
    # the accessor
    via = stack.radarx.rain_shaft(depth=[500.0, 1500.0])
    xr.testing.assert_allclose(via.dsd, eager.dsd)


def test_fall_speed_laws_and_validation():
    d = np.geomspace(0.3, 6.0, 12)
    nd = marshall_palmer(d)
    atlas = rain_shaft(nd, depth=500.0, fall_speed="atlas1973")
    drag = rain_shaft(nd, depth=500.0)
    assert not np.allclose(atlas.dsd, drag.dsd)
    poly = rain_shaft(nd, depth=500.0, fall_speed=("polynomial", 1.0, 2.0, 0.0, -0.05))
    assert np.isfinite(poly.dsd).all()
    abf = rain_shaft(nd, depth=500.0, fall_speed=(5.28695537, 1.22273346, 0.29531149))
    assert np.isfinite(abf.dsd).all()
    with pytest.raises(ValueError, match="positive at every diameter"):
        rain_shaft(
            marshall_palmer(np.geomspace(0.05, 6.0, 12)),
            depth=100.0,
            fall_speed="atlas1973",
        )
    with pytest.raises(TypeError, match="DataArray"):
        rain_shaft(nd.values, depth=1.0)
    with pytest.raises(TypeError, match="DataArray"):
        rain_shaft(nd.to_dataset(), depth=1.0)
    with pytest.raises(ValueError, match="collisions must be"):
        rain_shaft(nd, depth=1.0, collisions="x")
    with pytest.raises(ValueError, match="engine must be"):
        rain_shaft(nd, depth=1.0, engine="gpu")
    with pytest.raises(ValueError, match="'diameter' dimension"):
        rain_shaft(nd.rename(diameter="size"), depth=1.0)
    with pytest.raises(ValueError, match="step and cfl"):
        rain_shaft(nd, depth=1.0, step=0.0)
    with pytest.raises(ValueError, match="3 to 256"):
        rain_shaft(nd.isel(diameter=slice(0, 2)), depth=1.0)
    with pytest.raises(ValueError, match="3 to 256"):
        rain_shaft(marshall_palmer(np.geomspace(0.3, 6.0, 300)), depth=1.0)
    with pytest.raises(ValueError, match="ascending"):
        rain_shaft(nd.isel(diameter=slice(None, None, -1)), depth=1.0)
    with pytest.raises(ValueError, match="depth must be"):
        rain_shaft(nd, depth=[100.0, 50.0])
    with pytest.raises(ValueError, match="depth must be"):
        rain_shaft(nd, depth=-1.0)
    with pytest.raises(ValueError, match="number or a one-dimensional"):
        rain_shaft(nd, depth=[])
    with pytest.raises(ValueError, match="top_height"):
        rain_shaft(
            nd,
            depth=1.0,
            profile=xr.Dataset(
                {"air_density": ("height", [1.2, 1.1])},
                coords={"height": [0.0, 1000.0]},
            ),
        )
    z = np.array([0.0, 1000.0, 2000.0])
    prof = xr.Dataset(
        {"air_density": ("height", [1.2, 1.1, 1.0])}, coords={"height": z}
    )
    with pytest.raises(ValueError, match="must cover the shaft"):
        rain_shaft(nd, depth=2500.0, profile=prof, top_height=2000.0)
    with pytest.raises(ValueError, match="must cover the shaft"):
        rain_shaft(nd, depth=500.0, profile=prof, top_height=2500.0)
    with pytest.raises(ValueError, match="'height' coordinate"):
        rain_shaft(
            nd,
            depth=1.0,
            profile=xr.Dataset({"air_density": ("level", [1.2, 1.1])}),
            top_height=100.0,
        )
    with pytest.raises(ValueError, match="'pressure'"):
        rain_shaft(
            nd,
            depth=1.0,
            profile=xr.Dataset(
                {"temperature": ("height", [280.0, 270.0])},
                coords={"height": [0.0, 1000.0]},
            ),
            top_height=500.0,
        )
    with pytest.raises(ValueError, match="'temperature'"):
        rain_shaft(
            nd,
            depth=1.0,
            profile=xr.Dataset(
                {"pressure": ("height", [1e5, 9e4])}, coords={"height": [0.0, 1000.0]}
            ),
            top_height=500.0,
        )
    with pytest.raises(ValueError, match="at least two levels"):
        rain_shaft(
            nd,
            depth=1.0,
            profile=xr.Dataset(
                {"air_density": ("height", [1.2, np.nan])},
                coords={"height": [0.0, 1000.0]},
            ),
            top_height=500.0,
        )


def test_missing_kernel(monkeypatch):
    d = np.geomspace(0.3, 6.0, 12)
    nd = marshall_palmer(d)
    monkeypatch.setattr(rc, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError, match="compiled rain collision kernel"):
        rain_shaft(nd, depth=100.0, engine="compiled")
    out = rain_shaft(nd, depth=100.0)
    assert np.isfinite(out.dsd).all()
