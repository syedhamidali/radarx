#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the eyewall and rainband diagnostics
==============================================

The truth is a field with a known structure about a known centre: Gaussian
rings in the radius with azimuthal harmonics for ``tc_ring_stats``, and profiles of
known maxima, widths, vorticity and angular momentum (a Rankine vortex, solid
body rotation, a potential vortex) for ``tc_structure``. The compiled kernel and
the NumPy implementation of the ring statistics are compared, and the diagnostics
are applied to the GBVTD retrieval of the analytic vortex of Lee et al. (1999).
"""

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx._provenance import HISTORY, METHOD, REFERENCES, load_registry
from radarx.retrieve import gbvtd, synthetic_vortex, tc_ring_stats, tc_structure
from radarx.retrieve import tc_diagnostics as ts
from radarx.retrieve import tc_winds as tw

HAS_KERNEL = ts.HAS_COMPILED_KERNEL
ENGINES = ["numpy"] + (["compiled"] if HAS_KERNEL else [])
compiled_only = pytest.mark.skipif(not HAS_KERNEL, reason="compiled kernel not built")

CENTER = (30000.0, 40000.0)
RADII = np.arange(5000.0, 90001.0, 2500.0)


def profile(r):
    """An eyewall at 20 km and a rainband at 55 km (dBZ)."""
    return (
        20.0
        + 25.0 * np.exp(-0.5 * ((r - 20e3) / 5e3) ** 2)
        + 12.0 * np.exp(-0.5 * ((r - 55e3) / 6e3) ** 2)
    )


def field(r, g):
    # wavenumber 1 that decays outward, wavenumber 2 everywhere
    return (
        profile(r)
        + 4.0 * np.cos(g - np.radians(60.0)) * np.exp(-r / 50e3)
        + 2.0 * np.cos(2.0 * (g - np.radians(100.0)))
    )


def sector(g, lo, hi):
    """Gates between the azimuths ``lo`` and ``hi`` (degrees) seen from the centre."""
    return (np.degrees(g) % 360.0 > lo) & (np.degrees(g) % 360.0 < hi)


def sweep(func=field, azimuth_step=1.0, max_range=150000.0, range_step=500.0):
    az = np.arange(azimuth_step / 2, 360.0, azimuth_step)
    rng = np.arange(range_step, max_range + 1.0, range_step)
    ds = xr.Dataset(
        {
            "DBZH": (
                ("azimuth", "range"),
                np.zeros((az.size, rng.size)),
                {"units": "dBZ"},
            )
        },
        coords={
            "azimuth": az,
            "range": rng,
            "elevation": ("azimuth", np.full(az.size, 0.5)),
            "altitude": 50.0,
            "latitude": 20.0,
            "longitude": 80.0,
        },
    )
    geometry = tw._Sweep(ds, "DBZH", 4.0 / 3.0)
    r = np.hypot(geometry.x.values - CENTER[0], geometry.y.values - CENTER[1])
    g = np.arctan2(geometry.x.values - CENTER[0], geometry.y.values - CENTER[1])
    ds["DBZH"] = ds["DBZH"].copy(data=func(r, g))
    return ds


def volume(*sweeps):
    root = xr.Dataset(coords={"latitude": 20.0, "longitude": 80.0, "altitude": 50.0})
    nodes = {}
    for i, ds in enumerate(sweeps):
        nodes[f"/sweep_{i}"] = ds.drop_vars(["latitude", "longitude", "altitude"])
    return xr.DataTree.from_dict({"/": root, **nodes})


# --------------------------------------------------------------------------
# ring statistics
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_rings_recover_the_mean_and_the_waves(engine):
    out = tc_ring_stats(sweep(), CENTER, radii=RADII, ring_width=1000.0, engine=engine)
    assert out["MEAN"].dims == ("radius",)
    assert np.allclose(out["MEAN"], profile(RADII), atol=0.05)
    assert np.allclose(out["WAVE1_AMPLITUDE"], 4.0 * np.exp(-RADII / 50e3), atol=0.05)
    assert np.allclose(out["WAVE1_AZIMUTH"], 60.0, atol=1.5)
    assert np.allclose(out["WAVE2_AMPLITUDE"], 2.0, atol=0.05)
    assert np.allclose(out["WAVE2_AZIMUTH"], 100.0, atol=1.0)
    assert "WAVE3_AMPLITUDE" not in out
    assert (out["FIT_ORDER"] == 2).all()
    # the small rings have few gates (the beams are 1 degree apart, 90 km away)
    assert (out["DATA_GAP"] <= 12.0).all()
    assert (out["DATA_GAP"].sel(radius=slice(20000.0, 90000.0)) <= 4.0).all()
    assert (out["N_POINTS"] > 50).all()
    # the residual is the radial gradient of the field across the width of the ring
    assert (out["FIT_RMS"] < 1.5).all()
    # the standard deviation of the gates of a ring: waves of 3.6 and 2 about the mean
    assert np.isclose(
        out["STD"].sel(radius=5000.0), np.sqrt(0.5 * (3.62**2 + 2.0**2)), rtol=0.03
    )
    assert out["MEAN"].attrs["units"] == "dBZ"
    assert float(out["CENTER_X"]) == CENTER[0] and float(out["CENTER_Y"]) == CENTER[1]
    assert np.isfinite(out["CENTER_LATITUDE"]) and np.isfinite(out["CENTER_LONGITUDE"])
    assert out.attrs["ring_field"] == "DBZH"


@pytest.mark.parametrize("engine", ENGINES)
def test_wavenumber_three_and_the_mean_only(engine):
    def f(r, g):
        return 10.0 + 3.0 * np.cos(3.0 * (g - np.radians(20.0)))

    ds = sweep(f)
    radii = RADII[4:12]
    out = tc_ring_stats(
        ds, CENTER, radii=radii, ring_width=1000.0, max_wavenumber=3, engine=engine
    )
    assert np.allclose(out["WAVE3_AMPLITUDE"], 3.0, atol=0.02)
    assert np.allclose(out["WAVE3_AZIMUTH"], 20.0, atol=0.5)
    assert np.allclose(out["WAVE1_AMPLITUDE"], 0.0, atol=0.02)
    only = tc_ring_stats(
        ds, CENTER, radii=radii, ring_width=1000.0, max_wavenumber=0, engine=engine
    )
    assert "WAVE1_AMPLITUDE" not in only and np.allclose(only["MEAN"], 10.0, atol=0.05)
    assert (only["FIT_ORDER"] == 0).all()


@compiled_only
def test_engines_agree_with_gaps_and_a_threshold():
    # a missing sector of 100 degrees (seen from the centre) and scattered missing gates
    ds = sweep(lambda r, g: np.where(sector(g, 200.0, 300.0), np.nan, field(r, g)))
    rng = np.random.default_rng(5)
    data = ds["DBZH"].values.copy()
    data[rng.random(data.shape) < 0.1] = np.nan
    ds["DBZH"] = ds["DBZH"].copy(data=data)
    kw = dict(radii=RADII, ring_width=2000.0, max_wavenumber=3, threshold=30.0)
    a = tc_ring_stats(ds, CENTER, engine="compiled", n_threads=3, **kw)
    b = tc_ring_stats(ds, CENTER, engine="numpy", **kw)
    assert set(a.data_vars) == set(b.data_vars)
    for name in a.data_vars:
        np.testing.assert_allclose(
            a[name].values,
            b[name].values,
            rtol=1e-8,
            atol=1e-8,
            equal_nan=True,
            err_msg=name,
        )
    # the gap is 100 degrees: the wavenumber 1 is fitted, the wavenumber 2 is not
    assert (a["FIT_ORDER"].dropna("radius") == 1).all()
    assert (
        (a["DATA_GAP"].dropna("radius") >= 100.0)
        & (a["DATA_GAP"].dropna("radius") <= 106.0)
    ).all()
    assert a["WAVE2_AMPLITUDE"].isnull().all()
    one = tc_ring_stats(ds, CENTER, engine="compiled", n_threads=1, **kw)
    xr.testing.assert_equal(one["MEAN"], a["MEAN"])


@pytest.mark.parametrize("engine", ENGINES)
def test_gap_limits_and_minimum_of_gates(engine):
    ds = sweep(lambda r, g: np.where(sector(g, 100.0, 250.0), np.nan, field(r, g)))
    kw = dict(radii=RADII[4:10], ring_width=1000.0, engine=engine)
    # a gap of 150 degrees: only the mean and the wavenumber 1 (gap up to 180)
    out = tc_ring_stats(ds, CENTER, **kw)
    assert (out["FIT_ORDER"] == 1).all() and np.allclose(
        out["DATA_GAP"], 150.0, atol=3.0
    )
    # a limit below the gap: the rings are not fitted
    assert tc_ring_stats(ds, CENTER, max_gap=120.0, **kw)["MEAN"].isnull().all()
    # too few gates, none at all
    assert tc_ring_stats(ds, CENTER, min_points=10**6, **kw)["MEAN"].isnull().all()
    far = tc_ring_stats(ds, CENTER, radii=[1.0e6], engine=engine)
    assert far["MEAN"].isnull().all() and far["N_POINTS"].isnull().all()
    empty = ds.assign(DBZH=ds["DBZH"] * np.nan)
    assert tc_ring_stats(empty, CENTER, **kw)["MEAN"].isnull().all()


@pytest.mark.parametrize("engine", ENGINES)
def test_echo_fraction(engine):
    def f(r, g):
        return 30.0 + 10.0 * np.cos(g)

    out = tc_ring_stats(
        sweep(f),
        CENTER,
        radii=RADII[4:12],
        ring_width=1000.0,
        threshold=30.0,
        engine=engine,
    )
    assert np.allclose(out["ECHO_FRACTION"], 0.5, atol=0.02)
    low = tc_ring_stats(
        sweep(f),
        CENTER,
        radii=RADII[4:12],
        ring_width=1000.0,
        threshold=45.0,
        engine=engine,
    )
    assert (low["ECHO_FRACTION"] == 0.0).all()
    assert "ECHO_FRACTION" not in tc_ring_stats(
        sweep(f), CENTER, radii=RADII[:3], engine=engine
    )


def test_centre_options_volume_and_field_selection():
    ds = sweep()
    a = tc_ring_stats(ds, CENTER, radii=RADII[:6])
    ll = tw._offset_to_latlon(CENTER[0], CENTER[1], 20.0, 80.0)
    b = tc_ring_stats(ds, center_latlon=ll, radii=RADII[:6])
    assert np.allclose(a["MEAN"], b["MEAN"], atol=1e-2)
    # a DataArray is the field itself
    c = tc_ring_stats(ds["DBZH"], CENTER, radii=RADII[:6])
    xr.testing.assert_equal(a["MEAN"], c["MEAN"])
    renamed = ds.rename({"DBZH": "refl"})
    assert (
        float(tc_ring_stats(renamed, CENTER, field="refl", radii=RADII[:3])["MEAN"][0])
        > 0
    )
    with pytest.raises(KeyError, match="reflectivity"):
        tc_ring_stats(renamed, CENTER, radii=RADII[:3])
    with pytest.raises(KeyError, match="nothing"):
        tc_ring_stats(ds, CENTER, field="nothing", radii=RADII[:3])
    # a volume: one result per sweep that has the field, the others are skipped
    vol = volume(ds, ds.drop_vars("DBZH").assign(OTHER=ds["DBZH"]), ds)
    tree = vol.radarx.tc_ring_stats(CENTER, radii=RADII[:6])
    assert sorted(tree.children) == ["sweep_0", "sweep_2"]
    assert np.allclose(tree["sweep_0"]["MEAN"], a["MEAN"], atol=1e-6)
    with pytest.raises(KeyError, match="no sweep has the field"):
        volume(ds.drop_vars("DBZH").assign(OTHER=ds["DBZH"])).radarx.tc_ring_stats(
            CENTER
        )
    # one centre per sweep from a dict of centres
    centres = {"sweep_0": CENTER, "sweep_2": (CENTER[0] + 3000.0, CENTER[1])}
    moved = tc_ring_stats(vol, centres, radii=RADII[:6])
    assert np.allclose(moved["sweep_0"]["MEAN"], a["MEAN"], atol=1e-6)
    assert not np.allclose(moved["sweep_2"]["MEAN"], a["MEAN"], atol=1e-3)
    assert float(moved["sweep_2"]["CENTER_X"]) == CENTER[0] + 3000.0
    # a sweep without a centre is skipped
    assert list(tc_ring_stats(vol, {"sweep_0": CENTER}, radii=RADII[:3]).children) == [
        "sweep_0"
    ]
    with pytest.raises(KeyError, match="no entry"):
        tc_ring_stats(vol, {"elsewhere": CENTER}, radii=RADII[:3])


def test_ring_input_errors():
    ds = sweep()
    with pytest.raises(TypeError, match="xarray"):
        tc_ring_stats(np.zeros((3, 3)), CENTER)
    with pytest.raises(ValueError, match="give the centre"):
        tc_ring_stats(ds)
    with pytest.raises(ValueError, match="at the radar"):
        tc_ring_stats(ds, (0.0, 0.0))
    with pytest.raises(ValueError, match="not both"):
        tc_ring_stats(ds, CENTER, center_latlon=(20.0, 80.0))
    for bad, match in (
        ({"ring_width": 0.0}, "ring_width"),
        ({"max_wavenumber": 4}, "max_wavenumber"),
        ({"max_wavenumber": 1.5}, "max_wavenumber"),
        ({"min_points": 1}, "min_points"),
        ({"max_gap": 0.0}, "max_gap"),
        ({"max_gap": 400.0}, "max_gap"),
        ({"threshold": np.nan}, "threshold"),
        ({"k_e": 0.0}, "k_e"),
        ({"radii": [-1.0]}, "radii"),
        ({"engine": "gpu"}, "engine"),
    ):
        with pytest.raises(ValueError, match=match):
            tc_ring_stats(ds, CENTER, **bad)
    with pytest.raises(ValueError, match="radius"):
        tc_ring_stats(ds, CENTER, min_radius=5000.0, max_radius=1000.0)
    nolatlon = ds.drop_vars(["latitude", "longitude"])
    with pytest.raises(ValueError, match="latitude and longitude"):
        tc_ring_stats(nolatlon, center_latlon=(20.3, 80.3))
    plain = tc_ring_stats(nolatlon, CENTER, radii=RADII[:3])
    assert "CENTER_LATITUDE" not in plain
    monkey = ts.HAS_COMPILED_KERNEL
    ts.HAS_COMPILED_KERNEL = False
    try:
        with pytest.raises(ImportError, match="compiled"):
            tc_ring_stats(ds, CENTER, engine="compiled")
        assert np.isfinite(tc_ring_stats(ds, CENTER, radii=RADII[:3])["MEAN"]).all()
    finally:
        ts.HAS_COMPILED_KERNEL = monkey


@compiled_only
def test_kernel_rejects_bad_input():
    k = ts._tc_rings
    a, b = np.zeros(4), np.zeros(3)
    with pytest.raises(ValueError, match="same length"):
        k.rings(a, a, a, b, 0.0, 0.0, a, 1000.0)
    with pytest.raises(ValueError, match="1-D"):
        k.rings(a, a, a, a, 0.0, 0.0, np.zeros((2, 2)), 1000.0)
    with pytest.raises(ValueError, match="half_width"):
        k.rings(a, a, a, a, 0.0, 0.0, a, 0.0)
    with pytest.raises(ValueError, match="max_order"):
        k.rings(a, a, a, a, 0.0, 0.0, a, 1000.0, max_order=4)


def test_rings_dask_inputs_unchanged_and_provenance():
    dask = pytest.importorskip("dask")  # noqa: F841
    ds = sweep(azimuth_step=2.0)
    before = ds.copy(deep=True)
    lazy = ds.chunk({"azimuth": 45})
    out = tc_ring_stats(lazy, CENTER, radii=RADII[:8])
    assert out["MEAN"].chunks is not None
    xr.testing.assert_allclose(
        out["MEAN"], tc_ring_stats(ds, CENTER, radii=RADII[:8])["MEAN"]
    )
    xr.testing.assert_identical(ds, before)
    assert out.attrs[METHOD] and out.attrs[HISTORY].startswith("tc_ring_stats(")
    refs = out.attrs[REFERENCES].split()
    registry = load_registry()
    assert "10.1175/1520-0469(1982)039<0395:CEWSWM>2.0.CO;2" in refs
    assert all(r in registry for r in refs)


# --------------------------------------------------------------------------
# structure of a profile
# --------------------------------------------------------------------------


def radial(values, radius=None, **extra):
    radius = np.arange(1000.0, 100001.0, 1000.0) if radius is None else radius
    data = {
        "MEAN": ("radius", np.asarray(values(radius) if callable(values) else values))
    }
    data.update({k: ("radius", v) for k, v in extra.items()})
    return xr.Dataset(data, coords={"radius": ("radius", radius, {"units": "m"})})


def test_a_gaussian_ring_has_the_expected_width():
    s = 5000.0
    base, amp, r0 = 10.0, 30.0, 40000.0
    ds = radial(lambda r: base + amp * np.exp(-0.5 * ((r - r0) / s) ** 2))
    out = tc_structure(ds)
    assert out["N_PEAKS"] == 1
    assert np.isclose(out["PEAK_RADIUS"], r0, atol=1.0)
    assert np.isclose(out["PEAK_VALUE"], base + amp, atol=0.05)
    half = np.sqrt(2.0 * np.log(2.0)) * s  # the half-prominence width of a Gaussian
    assert np.isclose(out["INNER_EDGE"], r0 - half, rtol=0.01)
    assert np.isclose(out["OUTER_EDGE"], r0 + half, rtol=0.01)
    assert np.isclose(out["WIDTH"], 2.0 * half, rtol=0.01)
    assert np.isnan(out["SECONDARY_RADIUS"]) and np.isnan(out["MOAT_VALUE"])
    assert out["PEAK_VALUE"].attrs["units"] == ""


def test_concentric_maxima_and_the_moat():
    ds = radial(profile)
    out = tc_structure(ds)
    assert out["N_PEAKS"] == 2
    assert np.isclose(out["PEAK_RADIUS"], 20000.0, atol=1.0)
    assert np.isclose(out["SECONDARY_RADIUS"], 55000.0, atol=100.0)
    assert np.isclose(out["SECONDARY_VALUE"], profile(55000.0), atol=0.2)
    # the minimum between the two maxima of the sum of two Gaussians
    r = np.arange(1000.0, 100001.0, 1000.0)
    between = (r > 20000.0) & (r < 55000.0)
    k = np.argmin(np.where(between, profile(r), np.inf))
    assert out["MOAT_RADIUS"] == r[k] and out["MOAT_VALUE"] == profile(r[k])
    # a higher prominence that the band does not reach leaves the eyewall only
    assert tc_structure(ds, prominence=13.0)["N_PEAKS"] == 1
    # when the outer maximum is the highest it is the primary one
    swapped = radial(
        lambda r: 20.0
        + 12.0 * np.exp(-0.5 * ((r - 20e3) / 5e3) ** 2)
        + 25.0 * np.exp(-0.5 * ((r - 55e3) / 6e3) ** 2)
    )
    out2 = tc_structure(swapped)
    assert np.isclose(out2["PEAK_RADIUS"], 55000.0, atol=100.0)
    assert np.isclose(out2["SECONDARY_RADIUS"], 20000.0, atol=100.0)


def test_levels_are_crossed_outside_the_maximum():
    r = np.arange(1000.0, 100001.0, 1000.0)
    wind = np.where(
        r <= 20000.0, 50.0 * r / 20000.0, 50.0 * (1.0 - (r - 20000.0) / 100000.0)
    )
    ds = xr.Dataset(
        {"VT0": ("radius", wind, {"units": "m s-1"})},
        coords={"radius": ("radius", r, {"units": "m"})},
    )
    out = tc_structure(ds, levels=[17.5, 25.7, 32.9, 60.0])
    expected = 20000.0 + 100000.0 * (1.0 - np.array([17.5, 25.7, 32.9]) / 50.0)
    assert np.allclose(out["OUTER_LEVEL_RADIUS"].values[:3], expected, atol=1e-6)
    assert np.isnan(out["OUTER_LEVEL_RADIUS"].values[3])  # above the maximum
    assert list(out["level"].values) == [17.5, 25.7, 32.9, 60.0]
    assert out["level"].attrs["units"] == "m s-1"
    # a profile that does not fall to the level before the last radius
    assert np.isnan(tc_structure(ds, levels=[1.0])["OUTER_LEVEL_RADIUS"]).all()
    # the parabola through the three points of a triangular maximum
    assert np.isclose(out["PEAK_RADIUS"], 20000.0, atol=500.0)


def test_profiles_without_a_maximum_and_with_gaps():
    mono = radial(lambda r: 50.0 - r / 2000.0)
    out = tc_structure(mono)
    assert (
        out["N_PEAKS"] == 0 and np.isnan(out["PEAK_RADIUS"]) and np.isnan(out["WIDTH"])
    )
    short = radial([1.0, 2.0], radius=np.array([1000.0, 2000.0]))
    assert tc_structure(short)["N_PEAKS"] == 0
    flat = radial(np.full(30, 5.0), radius=np.arange(1000.0, 30001.0, 1000.0))
    assert tc_structure(flat)["N_PEAKS"] == 0
    nan = radial(lambda r: np.where(r < 60000.0, profile(r), np.nan))
    nan["MEAN"][5:8] = np.nan
    out = tc_structure(nan)
    assert np.isclose(out["PEAK_RADIUS"], 20000.0, atol=5.0)
    # a maximum at the edge of the radii is not a maximum
    edge = radial(lambda r: r / 1000.0)
    assert tc_structure(edge)["N_PEAKS"] == 0
    # the vertex of a parabola that is not a maximum falls back to the sample
    r = np.arange(1000.0, 6001.0, 1000.0)
    flat_top = radial([0.0, 1.0, 3.0, 3.0, 1.0, 0.0], radius=r)
    assert np.isclose(
        tc_structure(flat_top, prominence=0.5)["PEAK_RADIUS"], 3500.0, atol=600.0
    )


def test_kinematics_of_a_rankine_vortex():
    r = np.arange(500.0, 80001.0, 500.0)
    rmw, vmax = 20000.0, 50.0
    vt = np.where(r <= rmw, vmax * r / rmw, vmax * rmw / r)
    ds = xr.Dataset({"VT0": ("radius", vt)}, coords={"radius": ("radius", r)})
    out = tc_structure(ds)
    omega = vmax / rmw
    inside, outside = (r < 0.9 * rmw), (r > 1.1 * rmw)
    assert np.allclose(out["VORTICITY"][inside], 2.0 * omega, rtol=1e-3)
    assert np.allclose(out["VORTICITY"][outside], 0.0, atol=2e-6)
    assert np.allclose(out["ANGULAR_VELOCITY"][inside], omega)
    assert np.allclose(out["RELATIVE_ANGULAR_MOMENTUM"][outside], vmax * rmw)
    assert "ABSOLUTE_ANGULAR_MOMENTUM" not in out and "DIVERGENCE" not in out
    assert np.isclose(out["PEAK_RADIUS"], rmw, atol=100.0)
    # solid-body rotation has the vorticity 2 Omega at every radius, a potential
    # vortex none
    solid = tc_structure(
        xr.Dataset({"VT0": ("radius", 1e-3 * r)}, coords={"radius": r}), kinematics=True
    )
    assert np.allclose(solid["VORTICITY"], 2e-3, rtol=1e-9)
    potential = tc_structure(
        xr.Dataset({"VT0": ("radius", 1e5 / r)}, coords={"radius": r})
    )
    assert np.allclose(potential["VORTICITY"][1:-1], 0.0, atol=1e-7)
    assert np.allclose(potential["RELATIVE_ANGULAR_MOMENTUM"], 1e5)
    assert "VORTICITY" not in tc_structure(radial(profile), kinematics=False)
    assert "VORTICITY" in tc_structure(radial(profile), kinematics=True)


def test_absolute_angular_momentum_divergence_and_latitude():
    r = np.arange(1000.0, 60001.0, 1000.0)
    vt = 1e-3 * r
    vr = 2e-4 * r
    ds = xr.Dataset(
        {"VT0": ("radius", vt), "VR0": ("radius", vr), "CENTER_LATITUDE": ((), 15.0)},
        coords={"radius": r},
    )
    out = tc_structure(ds)
    f = 2.0 * 7.292115e-5 * np.sin(np.deg2rad(15.0))
    assert np.allclose(out["ABSOLUTE_ANGULAR_MOMENTUM"], r * vt + 0.5 * f * r**2)
    # u_r = c r gives the divergence 2 c
    assert np.allclose(out["DIVERGENCE"], 4e-4, rtol=1e-9)
    other = tc_structure(ds.drop_vars("CENTER_LATITUDE"), latitude=-15.0)
    assert np.allclose(other["ABSOLUTE_ANGULAR_MOMENTUM"], r * vt - 0.5 * f * r**2)
    assert float(out["CENTER_LATITUDE"]) == 15.0


def test_structure_of_the_gbvtd_retrieval_of_the_analytic_vortex():
    # the modified Rankine vortex with the decay exponent 1 is a Rankine vortex
    vmax, rmw, centre = 50.0, 20000.0, (0.0, 80000.0)
    ds = synthetic_vortex(
        center=centre,
        vmax=vmax,
        rmw=rmw,
        xt=1.0,
        azimuth_step=1.0,
        range_step=500.0,
        max_range=130000.0,
        asymmetry={1: (0.2, 45.0)},
    )
    radii = np.arange(4000.0, 45001.0, 1000.0)
    wind = gbvtd(ds, centre, radii=radii)
    out = tc_structure(wind, levels=[25.0])
    assert abs(float(out["PEAK_RADIUS"]) - rmw) < 1500.0
    assert abs(float(out["PEAK_VALUE"]) - vmax) < 2.5
    inside = radii < 0.8 * rmw
    assert np.allclose(out["VORTICITY"][inside], 2.0 * vmax / rmw, rtol=0.15)
    # the outer radius of 25 m/s of a profile vmax rmw / r is 2 rmw
    assert abs(float(out["OUTER_LEVEL_RADIUS"].item()) - 2.0 * rmw) < 2500.0
    # the wavenumber-1 amplitude at the maximum relative to the wind
    assert 0.0 < float(out["ASYMMETRY1"]) < 0.6
    assert out["CENTER_LATITUDE"].notnull()
    assert out["VORTICITY"].attrs["units"] == "s-1"


def test_volume_trees_and_accessor():
    r = np.arange(1000.0, 60001.0, 1000.0)
    ds = xr.Dataset({"MEAN": ("radius", profile(r))}, coords={"radius": r})
    tree = xr.DataTree.from_dict(
        {"/": xr.Dataset(), "/sweep_0": ds, "/sweep_1": ds, "/other": ds}
    )
    out = tc_structure(tree)
    assert sorted(out.children) == ["sweep_0", "sweep_1"]
    assert np.isclose(out["sweep_1"]["PEAK_RADIUS"], 20000.0, atol=5.0)
    with pytest.raises(ValueError, match="no sweeps"):
        tc_structure(xr.DataTree.from_dict({"/": xr.Dataset()}))
    acc = ds.radarx.tc_structure(prominence=3.0)
    assert acc["N_PEAKS"] == 2
    assert tree.radarx.tc_structure()["sweep_0"]["N_PEAKS"] == 2


def test_structure_errors_and_provenance():
    r = np.arange(1000.0, 30001.0, 1000.0)
    ds = xr.Dataset({"MEAN": ("radius", profile(r))}, coords={"radius": r})
    with pytest.raises(TypeError, match="Dataset or DataTree"):
        tc_structure(ds["MEAN"])
    with pytest.raises(ValueError, match="not in the dataset"):
        tc_structure(ds, "VT0")
    with pytest.raises(ValueError, match="not in the dataset"):
        tc_structure(xr.Dataset({"x": ("radius", r)}, coords={"radius": r}))
    with pytest.raises(ValueError, match="radius"):
        tc_structure(xr.Dataset({"MEAN": ("range", r)}))
    with pytest.raises(ValueError, match="profile"):
        tc_structure(
            xr.Dataset(
                {"MEAN": (("radius", "azimuth"), np.zeros((r.size, 2)))},
                coords={"radius": r},
            )
        )
    with pytest.raises(ValueError, match="prominence"):
        tc_structure(ds, prominence=-1.0)
    with pytest.raises(ValueError, match="levels"):
        tc_structure(ds, levels=[np.nan])
    out = tc_structure(ds)
    assert out.attrs[METHOD] and out.attrs[HISTORY].startswith("tc_structure(")
    refs = out.attrs[REFERENCES].split()
    registry = load_registry()
    assert "10.1175/1520-0469(2001)058<1079:TDRITK>2.0.CO;2" in refs
    assert all(r_ in registry for r_ in refs)
    # a lazy profile is computed
    dask = pytest.importorskip("dask")  # noqa: F841
    lazy = tc_structure(ds.chunk({"radius": 10}))
    assert np.isclose(lazy["PEAK_RADIUS"], out["PEAK_RADIUS"])


# --------------------------------------------------------------------------
# defensive branches
# --------------------------------------------------------------------------


def test_largest_gap_helper():
    seen = np.zeros(360, dtype=bool)
    assert ts._largest_gap(seen) == 360.0
    seen[[10, 20]] = True
    assert ts._largest_gap(seen) == 350.0  # the gap across north: 349 empty bins
    seen[:] = True
    assert ts._largest_gap(seen) == 1.0


def test_parabola_vertex_fallbacks():
    r = np.array([1000.0, 2000.0, 3000.0])
    assert ts._vertex(r, np.array([1.0, 2.0, 1.5]), 0) == (1000.0, 1.0)
    # not a maximum (a straight line): the sample itself
    assert ts._vertex(r, np.array([1.0, 2.0, 3.0]), 1) == (2000.0, 2.0)
    # uneven radii: the vertex is still between the neighbours
    far = np.array([1000.0, 1100.0, 9000.0])
    rv, yv = ts._vertex(far, np.array([0.0, 1.0, 0.999]), 1)
    assert 1000.0 < rv < 9000.0 and yv >= 1.0
    rv, yv = ts._vertex(r, np.array([1.0, 2.0, 1.5]), 1)
    assert 2000.0 < rv < 2500.0 and yv > 2.0


def test_numpy_rings_with_few_gates_and_singular_systems(monkeypatch):
    gates = (np.array([1000.0, 1010.0, 1020.0]), np.zeros(3), np.ones(3), np.ones(3))
    cfg = ts._ring_config(500.0, 2, 2, 360.0, None)
    # three gates, ring order 0: the three gates fit one mean
    out = ts._rings_numpy(gates, 0.0, 0.0, np.array([5000.0]), cfg)
    assert np.isnan(out).all()  # no gates at that radius
    out = ts._rings_numpy(gates, 0.0, 0.0, np.array([1010.0]), cfg)
    assert np.isclose(out[0, 0], 1.0) and out[0, 9] == 0.0
    # as many gates as unknowns of a higher order are not fitted
    cfg1 = ts._ring_config(500.0, 1, 2, 360.0, None)
    spread = (
        np.array([1000.0, 0.0, -1000.0]),
        np.array([0.0, 1000.0, 0.0]),
        np.array([1.0, 2.0, 3.0]),
        np.ones(3),
    )
    assert np.isnan(ts._rings_numpy(spread, 0.0, 0.0, np.array([1000.0]), cfg1)).all()

    def singular(_):
        raise np.linalg.LinAlgError("singular")

    big = sweep(azimuth_step=4.0, max_range=60000.0, range_step=1000.0)
    geometry = tw._Sweep(big, "DBZH", 4.0 / 3.0)
    gate = tw._gate_arrays(
        big["DBZH"].values, geometry.x.values, geometry.y.values, geometry.w.values
    )
    cfg2 = ts._ring_config(2000.0, 2, 10, 180.0, None)
    ok = ts._rings_numpy(gate, *CENTER, np.array([20000.0]), cfg2)
    assert np.isfinite(ok[0, 0])
    monkeypatch.setattr(np.linalg, "cholesky", singular)
    assert np.isnan(ts._rings_numpy(gate, *CENTER, np.array([20000.0]), cfg2)).all()
    monkeypatch.setattr(np.linalg, "cholesky", lambda a: np.eye(a.shape[0]) * 1e-12)
    assert np.isnan(ts._rings_numpy(gate, *CENTER, np.array([20000.0]), cfg2)).all()


def test_asymmetry_needs_a_peak():
    r = np.arange(1000.0, 30001.0, 1000.0)
    ds = xr.Dataset(
        {
            "MEAN": ("radius", 50.0 - r / 1000.0),
            "WAVE1_AMPLITUDE": ("radius", np.ones(r.size)),
        },
        coords={"radius": r},
    )
    out = tc_structure(ds)
    assert out["N_PEAKS"] == 0 and np.isnan(out["ASYMMETRY1"])
    # and at a peak inside the radii of the wave
    with_peak = xr.Dataset(
        {
            "MEAN": ("radius", profile(r)),
            "WAVE2_AMPLITUDE": ("radius", np.full(r.size, 4.0)),
        },
        coords={"radius": r},
    )
    got = tc_structure(with_peak)
    assert np.isclose(got["ASYMMETRY2"], 4.0 / profile(20000.0), rtol=1e-3)
