#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests of the ray tracer of radarx.fundamentals.refraction
=========================================================

The tracer is checked against a closed form (a profile whose rays are
logarithmic spirals), against an independent adaptive integration, against the
invariant of Snell's law, against the 4/3 Earth model for the standard
gradient and against the sign of the bending for sub- and super-refractive
profiles. The NumPy twin and the compiled kernel are compared.
"""

import numpy as np
import pytest
import xarray as xr
from scipy.integrate import solve_ivp

import radarx  # noqa: F401
from radarx.fundamentals import _refraction_numpy as ref
from radarx.fundamentals import (
    constants,
    k_e_from_rays,
    k_e_from_refractivity,
    refraction,
    trace_rays,
)
from radarx.fundamentals.geometry import beam_center_height, ground_range

A = constants.EARTH_RADIUS
ENGINES = ["numpy"] + (["compiled"] if refraction.HAS_COMPILED_KERNEL else [])
STANDARD = 1e6 / (4.0 * A / 1000.0)  # N-units per km of the 4/3 Earth, 39.24
HEIGHTS = np.arange(0.0, 30001.0, 500.0)


def linear(gradient, n0=313.0, heights=HEIGHTS):
    """Refractivity decreasing by ``gradient`` N-units per km."""
    return n0 - gradient * heights / 1000.0


@pytest.mark.parametrize("engine", ENGINES)
def test_vacuum_is_the_straight_line_of_the_law_of_cosines(engine):
    rng = np.arange(1e3, 300e3, 1e3)
    for elev in (0.0, 0.5, 3.0, 20.0):
        ray = trace_rays(rng, elev, HEIGHTS, np.zeros_like(HEIGHTS), engine=engine)
        np.testing.assert_allclose(
            ray.beam_height, beam_center_height(rng, elev, reff=A), atol=1e-3
        )
        np.testing.assert_allclose(
            ray.ground_range, ground_range(rng, elev, reff=A), atol=1e-3
        )
        # in a straight line r cos(theta) is constant
        np.testing.assert_allclose(
            (A + ray.beam_height) * np.cos(np.deg2rad(ray.gate_elevation)),
            A * np.cos(np.deg2rad(elev)),
            rtol=1e-12,
        )


@pytest.mark.parametrize("engine", ENGINES)
def test_closed_form_logarithmic_spiral(engine):
    # n r = a (n = 1 at the surface of the Earth) makes cos(theta) constant
    # along the ray: r = a + s sin(theta0) and phi = cot(theta0) ln(r / a)
    h = np.arange(0.0, 8001.0, 500.0)
    n_units = (A / (A + h) - 1.0) * 1e6
    rng = np.arange(1e3, 150e3, 1e3)
    for elev in (0.5, 2.0, 8.0):
        ray = trace_rays(rng, elev, h, n_units, engine=engine)
        e = np.deg2rad(elev)
        r = A + rng * np.sin(e)
        phi = np.log(r / A) / np.tan(e)
        if A + 8000.0 < r.max():
            continue
        np.testing.assert_allclose(ray.beam_height, r - A, atol=0.02)
        np.testing.assert_allclose(ray.ground_range, A * phi, atol=0.01)
        np.testing.assert_allclose(ray.gate_elevation, elev, atol=1e-4)


@pytest.mark.parametrize("engine", ENGINES)
def test_grazing_ray_at_the_trapping_gradient_stays_at_one_height(engine):
    # the same profile, elevation 0: the ray follows the surface of the Earth
    h = np.arange(0.0, 501.0, 10.0)
    n_units = (A / (A + h) - 1.0) * 1e6
    ray = trace_rays(np.arange(1e3, 200e3, 1e3), 0.0, h, n_units, engine=engine)
    # the interpolation of N between the levels is the only error
    np.testing.assert_allclose(ray.beam_height, 0.0, atol=1e-2)
    np.testing.assert_allclose(ray.ground_range, np.arange(1e3, 200e3, 1e3), rtol=1e-8)


def snell_invariant(ray, h, n_units, radar_height, elev):
    """n r cos(theta) at the gates; constant along an exact ray."""
    height = np.asarray(ray.beam_height)
    n = 1.0 + 1e-6 * np.interp(height, h, n_units)
    theta = np.deg2rad(np.asarray(ray.gate_elevation))
    return n * (A + height) * np.cos(theta)


@pytest.mark.parametrize("engine", ENGINES)
def test_snell_invariant_is_conserved(engine):
    rng = np.random.default_rng(3)
    h = np.arange(0.0, 20001.0, 250.0)
    # a smooth profile with a sharp inversion and a moist layer
    n_units = (
        320.0 * np.exp(-h / 8000.0)
        + 35.0 * np.exp(-(((h - 1500.0) / 300.0) ** 2))
        - 40.0 / (1.0 + np.exp(-(h - 2500.0) / 40.0))
        + rng.normal(0.0, 0.02, h.size)
    )
    ranges = np.arange(2e3, 250e3, 2e3)
    for elev in (0.0, 0.3, 1.0, 4.0):
        ray = trace_rays(ranges, elev, h, n_units, radar_height=400.0, engine=engine)
        inv = snell_invariant(ray, h, n_units, 400.0, elev)[0]
        # the invariant of the start of the ray
        n0 = 1.0 + 1e-6 * np.interp(400.0, h, n_units)
        c0 = n0 * (A + 400.0) * np.cos(np.deg2rad(elev))
        assert np.max(np.abs(inv / c0 - 1.0)) < 5e-9


def solve_reference(rng, elev, h, n_units, radar_height):
    """
    The same equations integrated with SciPy (DOP853, rtol 1e-13) layer by
    layer: the gradient is constant in a layer, and the integration stops at
    the level (a terminal event) where it changes.
    """
    n_idx = 1.0 + 1e-6 * n_units
    y = np.array([A + radar_height, 0.0, np.deg2rad(elev)])
    s = 0.0
    out = []
    todo = list(rng)
    while todo:
        direction = np.sign(np.sin(y[2])) or 1.0
        i = int(
            np.clip(
                np.searchsorted(h, y[0] - A + direction * 1e-7, side="right") - 1,
                0,
                h.size - 2,
            )
        )
        dn = (n_idx[i + 1] - n_idx[i]) / (h[i + 1] - h[i])
        n0 = n_idx[i] - dn * h[i]

        def f(t, v, dn=dn, n0=n0):
            r, _, th = v
            c = np.cos(th)
            return [np.sin(th), c / r, c * (1.0 / r + dn / (n0 + dn * (r - A)))]

        def below(t, v, i=i):
            return v[0] - A - h[i]

        def above(t, v, i=i):
            return v[0] - A - h[i + 1]

        below.terminal = above.terminal = True
        below.direction, above.direction = -1, 1
        sol = solve_ivp(
            f,
            (s, todo[-1]),
            y,
            method="DOP853",
            rtol=1e-13,
            atol=1e-10,
            events=[below, above],
            dense_output=True,
        )
        while todo and todo[0] <= sol.t[-1]:
            out.append(sol.sol(todo.pop(0)))
        y, s = sol.y[:, -1], sol.t[-1]
    out = np.array(out)
    return out[:, 0] - A, A * out[:, 1], np.rad2deg(out[:, 2])


@pytest.mark.parametrize("engine", ENGINES)
def test_agrees_with_an_adaptive_integration(engine):
    h = np.arange(0.0, 40001.0, 200.0)
    n_units = 315.0 * np.exp(-h / 7500.0) - 25.0 / (1.0 + np.exp(-(h - 1800.0) / 60.0))
    rng = np.arange(5e3, 250e3, 5e3)
    for elev in (0.2, 1.5, 6.0):
        ray = trace_rays(rng, elev, h, n_units, radar_height=250.0, engine=engine)
        height, ground, angle = solve_reference(rng, elev, h, n_units, 250.0)
        np.testing.assert_allclose(ray.beam_height, height, atol=2e-4)
        np.testing.assert_allclose(ray.ground_range, ground, atol=2e-5)
        np.testing.assert_allclose(ray.gate_elevation, angle, atol=1e-7)


@pytest.mark.parametrize("engine", ENGINES)
def test_the_step_converges(engine):
    h = np.arange(0.0, 20001.0, 200.0)
    n_units = 315.0 * np.exp(-h / 7500.0) - 25.0 / (1.0 + np.exp(-(h - 1800.0) / 60.0))
    rng = np.arange(10e3, 120e3, 10e3)
    best = trace_rays(rng, 1.5, h, n_units, radar_height=250.0, step=5.0, engine=engine)
    # steps end at the levels of the profile, so even a step of several km
    # does not smear the change of the gradient there
    for step, limit in ((3200.0, 0.02), (400.0, 2e-3), (100.0, 1e-4)):
        ray = trace_rays(
            rng, 1.5, h, n_units, radar_height=250.0, step=step, engine=engine
        )
        assert float(np.abs(ray.beam_height - best.beam_height).max()) < limit
        assert float(np.abs(ray.ground_range - best.ground_range).max()) < limit / 10.0
    # a linear profile is integrated exactly whatever the step
    lin = linear(STANDARD, heights=h)
    a = trace_rays(rng, 1.0, h, lin, step=5000.0, engine=engine)
    b = trace_rays(rng, 1.0, h, lin, step=10.0, engine=engine)
    np.testing.assert_allclose(a.beam_height, b.beam_height, atol=1e-5)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("radar_height", [0.0, 1500.0])
def test_standard_gradient_reproduces_the_four_thirds_earth(engine, radar_height):
    n_units = linear(STANDARD)
    rng = np.arange(500.0, 200e3 + 1.0, 500.0)
    reff = 4.0 / 3.0 * A
    # elevation: (largest height difference m, largest relative ground range
    # difference) up to 200 km range; the 4/3 model neglects cos(theta), n and
    # the ratio of the radii in the ground distance
    limits = {
        0.0: (0.3, 1.2e-4),
        0.5: (0.5, 1.6e-4),
        1.0: (0.7, 2.2e-4),
        5.0: (2.0, 5e-4),
    }
    for elev, (dh_max, ds_max) in limits.items():
        ray = trace_rays(
            rng, elev, HEIGHTS, n_units, radar_height=radar_height, engine=engine
        )
        h4 = beam_center_height(rng, elev, radar_height=radar_height, reff=reff)
        s4 = ground_range(rng, elev, reff=reff, radar_height=radar_height)
        assert np.abs(ray.beam_height - h4).max() < dh_max
        assert np.abs(ray.ground_range / s4 - 1.0).max() < ds_max
        # the equivalent k_e of every gate is 4/3
        k = k_e_from_rays(rng, elev, ray.beam_height, radar_height)
        assert float(np.abs(k - 4.0 / 3.0).max()) < 1e-3


@pytest.mark.parametrize("engine", ENGINES)
def test_bending_is_ordered_by_the_gradient(engine):
    rng = np.array([50e3, 100e3, 200e3])
    heights = {}
    ground = {}
    low = HEIGHTS[HEIGHTS <= 12000.0]
    for gradient in (0.0, 20.0, STANDARD, 80.0, 120.0):
        ray = trace_rays(rng, 0.5, low, linear(gradient, heights=low), engine=engine)
        heights[gradient] = ray.beam_height.values
        ground[gradient] = ray.ground_range.values
    order = [0.0, 20.0, STANDARD, 80.0, 120.0]
    for lo, hi in zip(order[:-1], order[1:]):
        # a steeper decrease of N bends the beam down more: lower beam
        assert np.all(heights[hi] < heights[lo])
    # sub-refractive: above the 4/3 Earth; super-refractive: below it
    assert np.all(heights[20.0] > heights[STANDARD])
    assert np.all(heights[80.0] < heights[STANDARD])
    # the beam of a superrefractive atmosphere runs lower, so that it is
    # below the straight line by more than the standard atmosphere is
    straight = beam_center_height(rng, 0.5, reff=A)
    assert np.all(heights[0.0] - straight > -1.0)
    assert np.all(straight - heights[80.0] > straight - heights[STANDARD])


@pytest.mark.parametrize("engine", ENGINES)
def test_equivalent_k_e_of_a_constant_gradient_is_the_effective_radius_factor(engine):
    # the old function and the ray tracer give the same factor
    rng = np.arange(5e3, 150e3, 5e3)
    low = HEIGHTS[HEIGHTS <= 12000.0]
    for gradient in (10.0, STANDARD, 60.0, 100.0):
        n_units = linear(gradient, heights=low)
        ray = trace_rays(rng, 0.5, low, n_units, engine=engine)
        k_old = k_e_from_refractivity(low, n_units, layer=(0.0, 10000.0))
        k_new = k_e_from_rays(rng, 0.5, ray.beam_height)
        assert float(np.abs(k_new.median() / k_old - 1.0)) < 1e-3


@pytest.mark.parametrize("engine", ENGINES)
def test_a_duct_traps_a_ray(engine):
    # a layer between 100 and 200 m where N decreases by 400 N/km, steeper than
    # the trapping gradient (-157 N/km)
    h = np.array([0.0, 100.0, 200.0, 5000.0, 30000.0])
    n_units = np.array([320.0, 316.0, 276.0, 86.0, -714.0])
    ranges = np.arange(2e3, 100e3, 2e3)
    # launched along the layer the ray stays within it, near its upper part
    trapped = trace_rays(ranges, 0.0, h, n_units, radar_height=150.0, engine=engine)
    assert float(trapped.beam_height.min()) > -5.0
    assert float(trapped.beam_height.max()) < 151.0
    # launched a little above the horizontal, still trapped
    slightly = trace_rays(ranges, 0.1, h, n_units, radar_height=120.0, engine=engine)
    assert float(slightly.beam_height.max()) < 130.0
    assert float(slightly.beam_height.min()) > 40.0
    # a steeper ray leaves the duct, and so does one that starts above it
    for elev, h0 in ((0.5, 150.0), (0.0, 250.0)):
        free = trace_rays(ranges, elev, h, n_units, radar_height=h0, engine=engine)
        assert float(free.beam_height[-1]) > 500.0
    # the same ray in the standard atmosphere is not trapped
    flat = trace_rays(
        ranges, 0.0, HEIGHTS, linear(STANDARD), radar_height=150.0, engine=engine
    )
    assert float(flat.beam_height[-1]) > 400.0
    # Snell's invariant holds in the duct
    inv = snell_invariant(trapped, h, n_units, 150.0, 0.0)
    n0 = 1.0 + 1e-6 * np.interp(150.0, h, n_units)
    c0 = n0 * (A + 150.0)
    assert np.max(np.abs(inv / c0 - 1.0)) < 1e-8


@pytest.mark.parametrize("engine", ENGINES)
def test_constant_refractive_index_outside_the_profile(engine):
    # a profile that stops at 1 km: above it the index is constant, so the ray
    # curves less than the 4/3 Earth there
    h = np.array([0.0, 1000.0])
    n_units = np.array([313.0, 313.0 - STANDARD])
    rng = np.arange(5e3, 200e3, 5e3)
    short = trace_rays(rng, 3.0, h, n_units, engine=engine)
    full = trace_rays(rng, 3.0, HEIGHTS, linear(STANDARD), engine=engine)
    assert np.all(short.beam_height[-5:] > full.beam_height[-5:])
    # below the first level the index is constant too
    low = trace_rays(rng, 0.5, h + 500.0, n_units, radar_height=0.0, engine=engine)
    assert np.all(np.isfinite(low.beam_height))


def test_engines_agree():
    if not refraction.HAS_COMPILED_KERNEL:
        pytest.skip("compiled kernel not built")
    h = np.arange(0.0, 20001.0, 300.0)
    n_units = 320.0 * np.exp(-h / 8000.0) - 30.0 / (1.0 + np.exp(-(h - 2000.0) / 50.0))
    rng = np.arange(1e3, 300e3, 1e3)
    elev = xr.DataArray([0.1, 0.5, 1.0, np.nan, 5.0, 12.0], dims="ray")
    a = trace_rays(rng, elev, h, n_units, engine="numpy")
    b = trace_rays(rng, elev, h, n_units, engine="compiled", n_threads=2)
    for name in a.data_vars:
        np.testing.assert_allclose(a[name], b[name], rtol=0, atol=1e-6)
    assert np.isnan(a.beam_height.isel(ray=3)).all()
    # the raw kernels, with the step that does not divide the gate spacing
    one = ref.trace(rng, np.array([0.5]), h, 1.0 + 1e-6 * n_units, 10.0, A, 77.0)
    two = refraction._refraction.trace(
        rng, np.array([0.5]), h, 1.0 + 1e-6 * n_units, 10.0, A, 77.0, 0
    )
    np.testing.assert_allclose(one, two, atol=1e-6)


def test_engine_option_and_missing_kernel(monkeypatch):
    with pytest.raises(ValueError, match="engine must be"):
        trace_rays([1e3], 1.0, HEIGHTS, linear(STANDARD), engine="gpu")
    monkeypatch.setattr(refraction, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError, match="compiled refraction kernel"):
        trace_rays([1e3], 1.0, HEIGHTS, linear(STANDARD), engine="compiled")
    out = trace_rays([1e3, 2e3], 1.0, HEIGHTS, linear(STANDARD), engine="auto")
    assert out.beam_height.shape == (2,)


@pytest.mark.parametrize("engine", ENGINES)
def test_ranges_are_sorted_internally_and_elevations_shared(engine):
    h = HEIGHTS
    n_units = linear(STANDARD)
    rng = np.array([30e3, 10e3, 20e3, 0.0, 10e3])
    ray = trace_rays(rng, [1.0, 1.0, 2.0, 1.0], h, n_units, engine=engine)
    assert ray.beam_height.dims == ("ray", "range")
    srt = trace_rays(np.sort(rng), [1.0, 2.0], h, n_units, engine=engine)
    # gate 3 (range 0) is the radar
    np.testing.assert_allclose(ray.beam_height.values[:, 3], 0.0, atol=1e-9)
    np.testing.assert_allclose(
        ray.beam_height.values[0, 0], srt.beam_height.values[0, 4]
    )
    np.testing.assert_allclose(
        ray.beam_height.values[2, 2], srt.beam_height.values[1, 3]
    )
    np.testing.assert_array_equal(ray.beam_height.values[0], ray.beam_height.values[1])
    np.testing.assert_allclose(
        ray.beam_height.values[:, 1], ray.beam_height.values[:, 4]
    )
    # a scalar elevation and one range
    one = trace_rays(10e3, 1.0, h, n_units, engine=engine)
    assert one.beam_height.dims == ("range",) and one.beam_height.size == 1


@pytest.mark.parametrize("engine", ENGINES)
def test_inputs_are_not_changed_and_dask_is_lazy(engine):
    dask = pytest.importorskip("dask.array")
    az = np.arange(8.0)
    elev = xr.DataArray(
        np.array([0.5, 0.5, 1.0, 1.0, 2.0, 2.0, 0.5, np.nan]),
        dims="azimuth",
        coords={"azimuth": ("azimuth", az, {"units": "degrees"})},
    )
    rng = xr.DataArray(
        np.arange(1e3, 100e3, 1e3),
        dims="range",
        coords={"range": ("range", np.arange(1e3, 100e3, 1e3), {"units": "meters"})},
    )
    h = HEIGHTS.copy()
    n_units = linear(STANDARD)
    h0, n0 = h.copy(), n_units.copy()
    rng_before = rng.copy(deep=True)
    eager = trace_rays(rng, elev, h, n_units, engine=engine)
    assert eager.beam_height.dims == ("azimuth", "range")
    assert eager.azimuth.attrs == {"units": "degrees"}
    assert rng["range"].attrs == {"units": "meters"}
    xr.testing.assert_identical(rng, rng_before)
    np.testing.assert_array_equal(h, h0)
    np.testing.assert_array_equal(n_units, n0)
    lazy = trace_rays(rng, elev.chunk({"azimuth": 3}), h, n_units, engine=engine)
    assert isinstance(lazy.beam_height.data, dask.Array)
    for name in eager.data_vars:
        np.testing.assert_allclose(lazy[name].compute(), eager[name], atol=1e-9)
    assert eager.beam_height.attrs["units"] == "m"
    assert eager.gate_elevation.attrs["units"] == "degree"


def test_provenance_is_written():
    out = trace_rays([1e3, 2e3], 1.0, HEIGHTS, linear(STANDARD))
    assert out.attrs["radarx_references"] == "10.6028/NBS.MONO.92"
    assert "Snell" in out.attrs["radarx_method"]
    assert "trace_rays" in out.attrs["history"]
    assert out.beam_height.attrs["radarx_references"] == "10.6028/NBS.MONO.92"
    k = k_e_from_rays(1e3, 1.0, 20.0)
    assert np.isfinite(k) or np.isnan(k)


def test_input_validation():
    n_units = linear(STANDARD)
    with pytest.raises(ValueError, match="at least two"):
        trace_rays([1e3], 1.0, [0.0], [300.0])
    with pytest.raises(ValueError, match="at least two"):
        trace_rays([1e3], 1.0, [0.0, np.nan], [300.0, 300.0])
    with pytest.raises(ValueError, match="same length"):
        trace_rays([1e3], 1.0, HEIGHTS, n_units[:-1])
    with pytest.raises(ValueError, match="distinct"):
        trace_rays([1e3], 1.0, [0.0, 0.0, 1.0], [300.0, 300.0, 299.0])
    with pytest.raises(ValueError, match="N-units"):
        trace_rays([1e3], 1.0, HEIGHTS, 1.0 + 1e-6 * n_units + 1e6)
    with pytest.raises(ValueError, match="step must be positive"):
        trace_rays([1e3], 1.0, HEIGHTS, n_units, step=0.0)
    with pytest.raises(ValueError, match="earth_radius"):
        trace_rays([1e3], 1.0, HEIGHTS, n_units, earth_radius=-1.0)
    with pytest.raises(ValueError, match="one-dimensional"):
        trace_rays(np.ones((2, 2)), 1.0, HEIGHTS, n_units)
    with pytest.raises(ValueError, match="one-dimensional"):
        trace_rays(
            xr.DataArray(np.ones((2, 2)), dims=("a", "b")), 1.0, HEIGHTS, n_units
        )
    with pytest.raises(ValueError, match="two or more"):
        trace_rays([1e3], np.ones((2, 2)), HEIGHTS, n_units)
    # unsorted profile heights are sorted; NaN levels dropped
    h = np.concatenate([HEIGHTS[::-1], [np.nan]])
    n = np.concatenate([n_units[::-1], [1.0]])
    a = trace_rays([50e3], 1.0, h, n)
    b = trace_rays([50e3], 1.0, HEIGHTS, n_units)
    np.testing.assert_allclose(a.beam_height, b.beam_height)
    # a DataArray range along another dimension is renamed
    c = trace_rays(xr.DataArray([1e3, 2e3], dims="gate"), 1.0, HEIGHTS, n_units)
    assert c.beam_height.dims == ("range",)


@pytest.mark.parametrize("engine", ENGINES)
def test_kernel_rejects_bad_arguments(engine):
    kernel = refraction._engine(engine)
    h = np.array([0.0, 1000.0])
    n = np.array([1.0003, 1.0002])
    with pytest.raises(ValueError):
        kernel.trace(np.array([2e3, 1e3]), np.array([1.0]), h, n, 0.0, A, 100.0, 0)
    with pytest.raises(ValueError):
        kernel.trace(np.array([1e3]), np.array([1.0]), h, n[:1], 0.0, A, 100.0, 0)
    with pytest.raises(ValueError):
        kernel.trace(np.array([1e3]), np.array([1.0]), h, n, 0.0, A, 0.0, 0)
    with pytest.raises(ValueError):
        kernel.trace(np.ones((2, 2)), np.array([1.0]), h, n, 0.0, A, 100.0, 0)


def test_numpy_twin_with_a_single_level_profile():
    out = ref.trace(
        np.array([10e3, 20e3]),
        np.array([1.0]),
        np.array([0.0]),
        np.array([1.0003]),
        0.0,
        A,
    )
    straight = beam_center_height(np.array([10e3, 20e3]), 1.0, reff=A)
    np.testing.assert_allclose(out[0, :, 0], straight, atol=1e-3)


def test_k_e_from_rays_edge_cases():
    # a straight ray in a flat atmosphere has no finite k_e: NaN, as at range 0
    k = k_e_from_rays(
        xr.DataArray([0.0, 10e3], dims="range"),
        1.0,
        xr.DataArray([0.0, 174.5], dims="range"),
    )
    assert np.isnan(k.values[0])
    assert k.attrs["units"] == "1" and k.name == "k_e"
    # the effective radius model reproduces itself
    rng = np.array([20e3, 80e3, 150e3])
    for k_true in (1.1, 4.0 / 3.0, 2.0):
        h = beam_center_height(rng, 1.5, radar_height=700.0, reff=k_true * A)
        np.testing.assert_allclose(k_e_from_rays(rng, 1.5, h, 700.0), k_true, rtol=1e-9)
    # NumPy in, NumPy out
    assert isinstance(k_e_from_rays(rng, 1.5, h, 700.0), np.ndarray)
