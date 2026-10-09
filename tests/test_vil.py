#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for VIL, VIL density, echo tops and liquid water content
===============================================================

Columns with known reflectivity have closed-form VIL and echo tops, written
here from the definitions (the Greene and Clark layer sum, linear
interpolation in dBZ of Lakshmanan et al.), not from the module. The compiled
kernel and the NumPy implementation must agree to rounding (rtol 1e-12).
"""
import numpy as np
import pytest
import xarray as xr
from xradar.georeference import antenna_to_cartesian

import radarx  # noqa: F401
from radarx.retrieve import (
    dsd,
    echo_top,
    liquid_water_content,
    melting_layer,
    vil,
    vil_density,
)

vilmod = __import__("radarx.retrieve.vil", fromlist=["x"])

ENGINES = ["numpy"] + (["compiled"] if vilmod.HAS_COMPILED_KERNEL else [])
compiled_only = pytest.mark.skipif(
    not vilmod.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)
K = 3.44e-6
ALT = 140.0


def z_of(dbz):
    return 10.0 ** (np.asarray(dbz, dtype=float) / 10.0)


def hand_vil(h, dbz, cap=56.0, floor=0.0):
    """Greene and Clark layer sum over consecutive samples, from the definition."""
    h = np.asarray(h, dtype=float)
    dbz = np.asarray(dbz, dtype=float)
    z = np.where(dbz < floor, 0.0, z_of(np.minimum(dbz, cap)))
    total = 0.0
    for i in range(len(h) - 1):
        total += (0.5 * (z[i] + z[i + 1])) ** (4.0 / 7.0) * (h[i + 1] - h[i])
    return K * total


def column(dbz, z=None, **coords):
    """Grid dataset with one (y, x) column of reflectivity."""
    dbz = np.asarray(dbz, dtype=float)
    z = np.arange(dbz.size) * 500.0 if z is None else np.asarray(z, dtype=float)
    return xr.Dataset(
        {"DBZH": (("z", "y", "x"), dbz[:, None, None], {"units": "dBZ"})},
        coords={"z": z, "y": [0.0], "x": [0.0], **coords},
    )


# --------------------------------------------------------------------------
# VIL of analytic columns
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_constant_layer_closed_form(engine):
    z = np.arange(0, 10001.0, 500.0)
    dbz = np.where((z >= 1000) & (z <= 5000), 40.0, np.nan)
    out = vil(column(dbz, z), engine=engine)
    expected = K * (1.0e4) ** (4.0 / 7.0) * 4000.0
    np.testing.assert_allclose(out.VIL.values.squeeze(), expected, rtol=1e-6)
    assert out.VIL.attrs["units"] == "kg m-2"
    assert out.VIL.dtype == np.float32
    assert float(out.VIL_LOWEST_HEIGHT.squeeze()) == 1000.0
    # nothing is assumed below the lowest valid sample: a lower bound
    assert int(out.VIL_LOWER_BOUND.squeeze()) == 1
    # extending the lowest value down to the ground adds that layer
    full = vil(column(dbz, z), fill_below=True, engine=engine)
    np.testing.assert_allclose(
        full.VIL.values.squeeze(), K * (1.0e4) ** (4.0 / 7.0) * 5000.0, rtol=1e-6
    )
    assert int(full.VIL_LOWER_BOUND.squeeze()) == 0
    # base height above the lowest sample: nothing is left out
    assert (
        int(
            vil(
                column(dbz, z), base_height=1000.0, engine=engine
            ).VIL_LOWER_BOUND.squeeze()
        )
        == 0
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_two_layers_and_nonuniform_spacing(engine):
    z = np.array([200.0, 900.0, 2500.0, 3100.0])
    dbz = np.array([20.0, 45.0, 35.0, 10.0])
    out = vil(column(dbz, z), engine=engine)
    np.testing.assert_allclose(out.VIL.values.squeeze(), hand_vil(z, dbz), rtol=1e-6)
    # two layers by hand, to see the (mean Z)^(4/7) form
    z2 = np.array([0.0, 1000.0, 3000.0])
    dbz2 = np.array([30.0, 50.0, 30.0])
    by_hand = K * (
        (0.5 * (1e3 + 1e5)) ** (4 / 7) * 1000.0
        + (0.5 * (1e5 + 1e3)) ** (4 / 7) * 2000.0
    )
    out = vil(column(dbz2, z2), engine=engine)
    np.testing.assert_allclose(out.VIL.values.squeeze(), by_hand, rtol=1e-6)
    # sample order does not matter
    shuffled = column(dbz[::-1], z[::-1])
    np.testing.assert_allclose(
        vil(shuffled, engine=engine).VIL.values.squeeze(), hand_vil(z, dbz), rtol=1e-6
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_reflectivity_cap_and_floor(engine):
    z = np.array([0.0, 1000.0, 2000.0])
    dbz = np.array([60.0, 62.0, 58.0])
    capped = vil(column(dbz, z), engine=engine).VIL.values.squeeze()
    np.testing.assert_allclose(capped, K * z_of(56.0) ** (4 / 7) * 2000.0, rtol=1e-6)
    free = vil(column(dbz, z), dbz_cap=None, engine=engine).VIL.values.squeeze()
    np.testing.assert_allclose(free, hand_vil(z, dbz, cap=np.inf), rtol=1e-6)
    assert free > capped
    # values below min_dbz count as no echo, but still bound the layer
    dbz = np.array([30.0, 5.0, 30.0])
    out = vil(column(dbz, z), min_dbz=10.0, engine=engine).VIL.values.squeeze()
    np.testing.assert_allclose(out, hand_vil(z, dbz, floor=10.0), rtol=1e-6)
    none = vil(column(dbz, z), min_dbz=None, engine=engine).VIL.values.squeeze()
    np.testing.assert_allclose(none, hand_vil(z, dbz, floor=-np.inf), rtol=1e-6)


@pytest.mark.parametrize("engine", ENGINES)
def test_gaps_are_bridged_linearly_in_height(engine):
    z = np.arange(0, 6000.0, 1000.0)
    dbz = np.array([30.0, 40.0, np.nan, np.nan, 35.0, 25.0])
    out = vil(column(dbz, z), engine=engine).VIL.values.squeeze()
    keep = ~np.isnan(dbz)
    # the two bracketing samples are joined by one layer of 3 km
    np.testing.assert_allclose(out, hand_vil(z[keep], dbz[keep]), rtol=1e-6)
    # missing="clear": the gap is no echo (-inf dBZ) instead
    clear = vil(column(dbz, z), missing="clear", engine=engine).VIL.values.squeeze()
    zero = np.where(np.isnan(dbz), -np.inf, dbz)
    np.testing.assert_allclose(clear, hand_vil(z, zero), rtol=1e-6)
    assert clear < out


@pytest.mark.parametrize("engine", ENGINES)
def test_clear_keeps_levels_below_the_lowest_valid_unavailable(engine):
    z = np.arange(0, 5000.0, 1000.0)
    dbz = np.array([np.nan, 30.0, np.nan, 20.0, np.nan])
    out = vil(column(dbz, z), missing="clear", engine=engine)
    ref = hand_vil(z[1:], np.where(np.isnan(dbz[1:]), -np.inf, dbz[1:]))
    np.testing.assert_allclose(out.VIL.values.squeeze(), ref, rtol=1e-6)
    assert float(out.VIL_LOWEST_HEIGHT.squeeze()) == 1000.0


@pytest.mark.parametrize("engine", ENGINES)
def test_too_few_samples(engine):
    z = np.array([0.0, 1000.0, 2000.0])
    only_one = column(np.array([np.nan, 40.0, np.nan]), z)
    assert np.isnan(vil(only_one, engine=engine).VIL.values).all()
    out = vil(only_one, fill_below=True, base_height=0.0, engine=engine)
    np.testing.assert_allclose(
        out.VIL.values.squeeze(), K * z_of(40.0) ** (4 / 7) * 1000.0, rtol=1e-6
    )
    empty = column(np.full(3, np.nan), z)
    out = vil(empty, engine=engine)
    assert (
        np.isnan(out.VIL.values).all() and np.isnan(out.VIL_LOWEST_HEIGHT.values).all()
    )
    assert int(out.VIL_LOWER_BOUND.squeeze()) == 0
    assert np.isnan(echo_top(empty, engine=engine).values).all()


@pytest.mark.parametrize("engine", ENGINES)
def test_liquid_vil_below_the_melting_level(engine):
    z = np.array([500.0, 1500.0, 3000.0, 5000.0])
    dbz = np.array([40.0, 45.0, 30.0, 20.0])
    ds = column(dbz, z)
    # ceiling inside the layer 1500-3000 m: the profile is cut there, Z linear
    c = 2400.0
    zc = z_of(45.0) + (z_of(30.0) - z_of(45.0)) * (c - 1500.0) / 1500.0
    expected = K * (
        (0.5 * (z_of(40.0) + z_of(45.0))) ** (4 / 7) * 1000.0
        + (0.5 * (z_of(45.0) + zc)) ** (4 / 7) * (c - 1500.0)
    )
    out = vil(ds, melting=c, engine=engine)
    np.testing.assert_allclose(out.VIL_LIQUID.values.squeeze(), expected, rtol=1e-6)
    np.testing.assert_allclose(out.VIL.values.squeeze(), hand_vil(z, dbz), rtol=1e-6)
    # a ceiling above the profile gives the full VIL; one at a sample
    above = vil(ds, melting=9000.0, engine=engine)
    np.testing.assert_allclose(above.VIL_LIQUID, above.VIL, rtol=1e-6)
    at_sample = vil(ds, melting=1500.0, engine=engine)
    np.testing.assert_allclose(
        at_sample.VIL_LIQUID.values.squeeze(), hand_vil(z[:2], dbz[:2]), rtol=1e-6
    )
    # a ceiling below the lowest sample: no liquid layer unless filled
    low = vil(ds, melting=300.0, engine=engine)
    assert np.isnan(low.VIL_LIQUID.values).all()
    filled = vil(ds, melting=300.0, fill_below=True, base_height=0.0, engine=engine)
    np.testing.assert_allclose(
        filled.VIL_LIQUID.values.squeeze(), K * z_of(40.0) ** (4 / 7) * 300.0, rtol=1e-6
    )
    # no melting level (NaN) gives no liquid VIL
    none = vil(ds, melting=np.nan, engine=engine)
    assert np.isnan(none.VIL_LIQUID.values).all()
    assert "VIL_LIQUID" not in vil(ds, engine=engine)


def test_melting_forms():
    z = np.arange(0, 6000.0, 500.0)
    t = xr.DataArray(np.array([1000.0, 3000.0, np.nan]), dims="time")
    qvp = xr.Dataset(
        {"DBZH": (("time", "height"), np.tile(np.linspace(40, 10, z.size), (3, 1)))},
        coords={"time": np.arange(3), "height": z, "altitude": 100.0},
    )
    a = vil(qvp, melting=t)
    b = vil(qvp, melting=xr.Dataset({"melting_layer_bottom": t}))
    xr.testing.assert_identical(a, b)
    assert a.VIL_LIQUID.dims == ("time",)
    assert a.VIL_LIQUID[0] < a.VIL_LIQUID[1] < a.VIL[1]
    assert np.isnan(a.VIL_LIQUID[2])
    with pytest.raises(KeyError, match="melting_layer_bottom"):
        vil(qvp, melting=xr.Dataset({"x": t}))
    with pytest.raises(ValueError, match="melting"):
        vil(qvp, melting=xr.DataArray(np.ones(4), dims="time"))
    shifted = xr.DataArray(np.ones(3), dims="time", coords={"time": [5, 6, 7]})
    with pytest.raises(ValueError, match="does not match"):
        vil(qvp, melting=shifted)
    with pytest.raises(ValueError, match="fit the columns"):
        vil(qvp, melting=xr.DataArray(np.ones(3), dims="other"))


@pytest.mark.parametrize("engine", ENGINES)
def test_melting_layer_result(engine):
    """The result of melting_layer is used directly (its bottom is the ceiling)."""
    z = np.arange(500.0, 8000.0, 250.0)
    dbz = np.tile(np.linspace(45, 5, z.size), (2, 1))
    qvp = xr.Dataset(
        {
            "DBZH": (("time", "height"), dbz),
            "ZDR": (("time", "height"), np.full(dbz.shape, 0.5)),
            "RHOHV": (("time", "height"), np.full(dbz.shape, 0.99)),
        },
        coords={"time": np.arange(2), "height": z, "altitude": 100.0},
    )
    ml = xr.Dataset(
        {"melting_layer_bottom": ("time", np.array([3000.0, 3500.0]))},
        coords={"time": np.arange(2)},
    )
    out = vil(qvp, melting=ml, engine=engine)
    ref = [hand_vil(z[z <= c], dbz[i][z <= c]) for i, c in enumerate((3000.0, 3500.0))]
    np.testing.assert_allclose(out.VIL_LIQUID, ref, rtol=2e-3)
    assert melting_layer is not None


# --------------------------------------------------------------------------
# echo tops
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_echo_top_interpolates_in_dbz(engine):
    z = np.array([1000.0, 2000.0, 3500.0, 5000.0])
    dbz = np.array([40.0, 30.0, 22.0, 6.0])
    ds = column(dbz, z)
    # highest sample at or above 18 dBZ is 22 dBZ at 3500 m, the next is 6 dBZ
    expected = 3500.0 + (22.0 - 18.0) / (22.0 - 6.0) * 1500.0
    top = echo_top(ds, engine=engine)
    np.testing.assert_allclose(top.values.squeeze(), expected, rtol=1e-6)
    assert top.attrs["units"] == "m"
    # the threshold on a sample gives that sample's height
    np.testing.assert_allclose(
        echo_top(ds, threshold=22.0, engine=engine).values.squeeze(), 3500.0
    )
    # a higher beam without echo is -14 dBZ
    ds2 = column(np.array([40.0, 30.0, 22.0, -np.inf]), z)
    exp2 = 3500.0 + (22.0 - 18.0) / (22.0 + 14.0) * 1500.0
    np.testing.assert_allclose(echo_top(ds2, engine=engine).values.squeeze(), exp2)
    exp3 = 3500.0 + (22.0 - 18.0) / (22.0 + 20.0) * 1500.0
    np.testing.assert_allclose(
        echo_top(ds2, no_echo_dbz=-20.0, engine=engine).values.squeeze(), exp3
    )
    # the traditional echo top is the top of the beam (half a level above)
    trad = echo_top(ds, interpolate=False, engine=engine).values.squeeze()
    np.testing.assert_allclose(trad, 3500.0 + 0.5 * 1500.0)
    # the highest sample is the echo top: top of its bin
    ds3 = column(np.array([40.0, 30.0, 22.0, 30.0]), z)
    np.testing.assert_allclose(
        echo_top(ds3, engine=engine).values.squeeze(), 5000.0 + 0.5 * 1500.0
    )
    # no sample reaches the threshold
    assert np.isnan(echo_top(column(np.array([5.0, 6.0]), z[:2])).values).all()


@pytest.mark.parametrize("engine", ENGINES)
def test_echo_top_clear_marks_the_level_above_as_no_echo(engine):
    z = np.arange(0, 5000.0, 1000.0)
    dbz = np.array([40.0, 30.0, 20.0, np.nan, np.nan])
    gap = echo_top(column(dbz, z), engine=engine).values.squeeze()
    assert gap == 2000.0 + 0.5 * 1000.0  # the 20 dBZ level is the highest
    clear = echo_top(column(dbz, z), missing="clear", engine=engine).values.squeeze()
    np.testing.assert_allclose(clear, 2000.0 + (20.0 - 18.0) / (20.0 + 14.0) * 1000.0)
    assert clear < gap


@pytest.mark.parametrize("engine", ENGINES)
def test_vil_density_is_vil_over_echo_top(engine):
    z = np.arange(0, 12001.0, 500.0)
    dbz = np.where(z <= 8000, 40.0, -np.inf)
    dbz[0] = np.nan
    ds = column(dbz, z, altitude=100.0)
    v = vil(ds, missing="clear", engine=engine)
    top = echo_top(ds, missing="clear", engine=engine)
    dens = vil_density(ds, missing="clear", engine=engine)
    ref = 1000.0 * float(v.VIL.squeeze()) / (float(top.squeeze()) - 100.0)
    np.testing.assert_allclose(dens.values.squeeze(), ref, rtol=1e-6)
    assert dens.attrs["units"] == "g m-3"
    # echo top at or below the base: no density
    assert np.isnan(vil_density(ds, missing="clear", base_height=1e5).values).all()
    # a different echo-top threshold changes the density
    other = vil_density(ds, missing="clear", threshold=45.0)
    assert np.isnan(other.values).all()


# --------------------------------------------------------------------------
# engines
# --------------------------------------------------------------------------


@compiled_only
@pytest.mark.parametrize("fill", [False, True])
def test_compiled_equals_numpy_on_random_columns(fill):
    rnd = np.random.default_rng(3)
    nk, nc = 14, 500
    h = np.sort(rnd.uniform(0, 15e3, (nk, nc)), axis=0)
    h[:, ::7] = h[::-1, ::7]  # unsorted columns
    h[rnd.random(h.shape) < 0.05] = np.nan
    v = rnd.uniform(-10, 70, (nk, nc))
    v[rnd.random(v.shape) < 0.15] = np.nan
    v[rnd.random(v.shape) < 0.1] = -np.inf
    ceiling = np.where(rnd.random(nc) < 0.7, rnd.uniform(0, 12e3, nc), np.nan)
    h_top = h + 300.0
    opts = vilmod._options(fill_below=fill)
    a = vilmod._columns(h, h_top, v, ceiling, False, opts, 100.0, False, 1)
    b = vilmod._columns(h, h_top, v, ceiling, False, opts, 100.0, True, 3)
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=0, equal_nan=True)
    assert np.isfinite(a[0]).sum() > 100 and np.isfinite(a[1]).sum() > 50
    assert np.isfinite(a[4]).sum() > 100
    # shared heights
    hs = np.sort(rnd.uniform(0, 15e3, nk))
    a = vilmod._columns(hs, hs + 300, v, ceiling, True, opts, 100.0, False, 1)
    b = vilmod._columns(hs, hs + 300, v, ceiling, True, opts, 100.0, True, 2)
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=0, equal_nan=True)
    # no cap and no floor, traditional echo top
    opts = vilmod._options(dbz_cap=None, min_dbz=None, interpolate=False)
    a = vilmod._columns(hs, hs + 300, v, ceiling, True, opts, 100.0, False, 1)
    b = vilmod._columns(hs, hs + 300, v, ceiling, True, opts, 100.0, True, 2)
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=0, equal_nan=True)


@compiled_only
def test_kernel_rejects_bad_shapes():
    kernel = vilmod._vil
    ok = np.zeros((3, 4))
    args = (np.zeros(3), np.zeros(3), ok, np.full(4, np.nan), True)
    tail = (56.0, 0.0, 18.0, -14.0, False, 0.0, True)
    with pytest.raises(ValueError):
        kernel.columns(np.zeros(2), np.zeros(3), ok, np.full(4, np.nan), True, *tail)
    with pytest.raises(ValueError):
        kernel.columns(
            np.zeros((3, 4)), np.zeros(3), ok, np.full(4, np.nan), False, *tail
        )
    with pytest.raises(ValueError):
        kernel.columns(*args[:3], np.full(3, np.nan), True, *tail)
    with pytest.raises(ValueError):
        kernel.columns(*args[:2], np.zeros(4), *args[3:], *tail)


def test_engine_errors(monkeypatch):
    ds = column(np.array([10.0, 20.0, 30.0]))
    with pytest.raises(ValueError, match="engine"):
        vil(ds, engine="fortran")
    monkeypatch.setattr(vilmod, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError):
        vil(ds, engine="compiled")
    assert np.isfinite(vil(ds, engine="auto").VIL).all()


# --------------------------------------------------------------------------
# quasi-vertical profiles
# --------------------------------------------------------------------------


def profile_field(nph=360, nz=24):
    z = np.arange(nz) * 500.0 + 500.0
    phi = np.deg2rad(np.arange(nph) + 0.5)
    return z, phi


@pytest.mark.parametrize("engine", ENGINES)
def test_qvp_equals_columns_for_a_uniform_field(engine):
    z, phi = profile_field()
    prof = 45.0 - 3.0 * (z / 1000.0)  # dBZ decreasing with height
    cols = xr.Dataset(
        {"DBZH": (("z", "y", "x"), np.tile(prof[:, None, None], (1, 3, 2)))},
        coords={"z": z, "y": np.arange(3.0), "x": np.arange(2.0), "altitude": 0.0},
    )
    qvp = xr.Dataset(
        {"DBZH": (("time", "height"), np.tile(prof, (4, 1)))},
        coords={"time": np.arange(4), "height": z, "altitude": 0.0},
    )
    a = vil(cols, engine=engine)
    b = vil(qvp, engine=engine)
    assert b.VIL.dims == ("time",)
    np.testing.assert_allclose(b.VIL, a.VIL.values.ravel()[:1].repeat(4), rtol=1e-6)
    # the profile starts above the ground: the result is a lower bound
    assert (b.VIL_LOWER_BOUND == 1).all()
    assert float(b.VIL_LOWEST_HEIGHT[0]) == z[0]


@pytest.mark.parametrize("engine", ENGINES)
def test_qvp_of_the_mean_profile_differs_from_the_mean_vil(engine):
    """Z^(4/7) is concave: the VIL of the mean Z profile is above the mean VIL.

    The mean is taken in linear Z (the QVP default); a mean of dBZ is lower.
    """
    z, phi = profile_field()
    mean_z = z_of(45.0 - 3.0 * (z / 1000.0))
    factor = 1.0 + 0.9 * np.cos(phi)  # azimuthal variation, mean 1
    field = mean_z[:, None] * factor[None, :]
    cols = xr.Dataset(
        {"DBZH": (("z", "y"), 10.0 * np.log10(field))},
        coords={"z": z, "y": np.arange(phi.size) * 1.0, "altitude": 0.0},
    ).expand_dims(x=[0.0])
    columns = vil(cols, dbz_cap=None, engine=engine).VIL.values
    qvp = xr.Dataset(
        {"DBZH": ("height", 10.0 * np.log10(field.mean(axis=1)))},
        coords={"height": z, "altitude": 0.0},
    )
    single = float(vil(qvp, dbz_cap=None, engine=engine).VIL)
    mean_of_columns = float(columns.mean())
    # Jensen's inequality for the concave power 4/7: f(mean Z) >= mean f(Z)
    assert single > mean_of_columns
    assert 1.0 < single / mean_of_columns < 1.3
    # the profile of the mean in dBZ is below the mean VIL
    qvp_db = qvp.copy(data={"DBZH": 10.0 * np.log10(field).mean(axis=1)})
    assert float(vil(qvp_db, dbz_cap=None, engine=engine).VIL) < mean_of_columns


# --------------------------------------------------------------------------
# polar volumes
# --------------------------------------------------------------------------


def storm_dbz(x, y, z):
    """Smooth cell with a vertical decrease; a field of known value everywhere."""
    r2 = ((x - 40e3) ** 2 + (y - 20e3) ** 2) / (15e3) ** 2
    zz = np.clip(z / 10e3, 0, 1.3)
    dbz = 52.0 * np.exp(-r2) - 30.0 * zz**2 - 5.0
    return np.where(dbz > -10.0, dbz, -20.0)


ELEVATIONS = (0.5, 1.5, 2.5, 3.5, 4.5, 6.0, 8.0, 10.0, 12.5, 15.5, 19.5)


def storm_volume(
    field=storm_dbz, elevations=ELEVATIONS, nray=360, dr=500.0, rmax=120e3, noecho=-5.0
):
    az = (np.arange(nray) + 0.5) * 360.0 / nray
    rng = np.arange(dr * 2, rmax, dr)
    sweeps = {}
    for k, e in enumerate(elevations):
        x, y, z = antenna_to_cartesian(rng[None, :], az[:, None], e, site_altitude=ALT)
        d = field(x, y, z)
        d = np.where(d > noecho, d, np.nan)  # gates without echo are missing
        sweeps[f"sweep_{k}"] = xr.Dataset(
            {"DBZH": (("azimuth", "range"), d.astype(np.float32), {"units": "dBZ"})},
            coords={
                "azimuth": az,
                "range": rng,
                "elevation": ("azimuth", np.full(nray, e)),
            },
        )
    root = xr.Dataset(coords={"latitude": 33.9, "longitude": -88.3, "altitude": ALT})
    return xr.DataTree.from_dict({"/": root, **sweeps})


def ground_distance(rng, elevation):
    x, y, _ = antenna_to_cartesian(rng, 0.0, elevation, site_altitude=ALT)
    return np.hypot(x, y)


def hand_loop_column(tree, iaz, s, cap=56.0, floor=0.0, base_clear=True):
    """Explicit loop over the sweeps: nearest gate at ground range s, beam height."""
    heights, values = [], []
    for name in sorted(tree.children, key=lambda n: int(n.split("_")[1])):
        ds = tree[name].to_dataset(inherit=False)
        el = float(ds.elevation[iaz])
        g = ground_distance(ds.range.values, el)
        if s < g[0] - 0.5 * (g[1] - g[0]) or s > g[-1] + 0.5 * (g[-1] - g[-2]):
            continue
        j = int(np.argmin(np.abs(g - s)))
        value = float(ds.DBZH[iaz, j])
        heights.append(radarx_height(s, el))
        values.append(-np.inf if np.isnan(value) else value)
    return np.array(heights), np.array(values)


def radarx_height(s, el):
    """Beam height at ground range s from the beam geometry, by iteration on r."""
    from radarx.fundamentals.geometry import beam_center_height

    r = s / np.cos(np.deg2rad(el))
    for _ in range(50):
        g = ground_distance(r, el)
        r += (s - g) / np.cos(np.deg2rad(el))
    return float(beam_center_height(r, el, ALT))


@pytest.mark.parametrize("engine", ENGINES)
def test_polar_volume_against_a_hand_loop(engine):
    tree = storm_volume()
    out = vil(tree, engine=engine)
    top = echo_top(tree, engine=engine)
    assert out.VIL.dims == ("azimuth", "range") and "ground_range" in out.coords
    ground = out.ground_range.values
    checked = 0
    for iaz in (10, 55, 100, 160):
        for j in range(40, ground.size, 37):
            h, v = hand_loop_column(tree, iaz, ground[j])
            if h.size < 2:
                continue
            ref = hand_vil(h, v)
            np.testing.assert_allclose(
                out.VIL.values[iaz, j], ref, rtol=2e-4, atol=1e-4
            )
            checked += 1
            hit = np.nonzero(v >= 18.0)[0]
            if hit.size:
                b = hit[-1]
                if b + 1 < h.size:
                    za = max(v[b + 1], -14.0)
                    t = h[b] + (v[b] - 18.0) / (v[b] - za) * (h[b + 1] - h[b])
                    np.testing.assert_allclose(top.values[iaz, j], t, rtol=2e-4)
    assert checked > 10
    # the lowest valid height is the lowest beam of the column
    assert float(out.VIL_LOWEST_HEIGHT.max()) > float(out.VIL_LOWEST_HEIGHT.min())
    assert (out.VIL_LOWER_BOUND.values[np.isfinite(out.VIL.values)] == 1).all()


@compiled_only
def test_polar_volume_engines_agree_and_chunks_cover_the_grid(monkeypatch):
    tree = storm_volume(nray=180)
    monkeypatch.setattr(vilmod, "_CHUNK", 5000)  # several azimuth blocks
    a = vil(tree, melting=3000.0, engine="numpy")
    b = vil(tree, melting=3000.0, engine="compiled", n_threads=2)
    xr.testing.assert_allclose(a, b, rtol=1e-6)
    assert np.isfinite(a.VIL).sum() > 1000
    ea = echo_top(tree, engine="numpy")
    eb = echo_top(tree, engine="compiled")
    xr.testing.assert_allclose(ea, eb)


@pytest.mark.parametrize("engine", ENGINES)
def test_polar_fill_below_and_flag(engine):
    tree = storm_volume()
    low = vil(tree, engine=engine)
    full = vil(tree, fill_below=True, engine=engine)
    ok = np.isfinite(low.VIL.values)
    assert (full.VIL.values[ok] >= low.VIL.values[ok]).all()
    assert (full.VIL_LOWER_BOUND == 0).all()
    echo = ok & (low.VIL.values > 0)
    assert echo.sum() > 1000
    assert (full.VIL.values[echo] > low.VIL.values[echo]).mean() > 0.9


@pytest.mark.parametrize("engine", ENGINES)
def test_polar_vs_gridded_agreement(engine):
    """Cone-gridded VIL and VIL of the polar volume agree to within 10 %.

    The grid is interpolated linearly in dBZ between the cones, the polar
    columns integrate with linear Z between beams, so the grid is a few per
    cent lower where the reflectivity falls quickly with height.
    """
    from scipy.interpolate import RegularGridInterpolator

    from radarx.grid import grid_cones

    tree = storm_volume()
    polar = vil(tree, engine=engine)
    x = y = np.arange(-100e3, 100e3 + 1, 1000.0)
    z = np.arange(250.0, 18e3, 500.0)
    grid = grid_cones(tree, "DBZH", x, y, z, engine=engine)
    gridded = vil(grid, engine=engine)
    assert gridded.VIL.dims == ("y", "x")
    az = np.deg2rad(polar.azimuth.values)[:, None]
    s = polar.ground_range.values[None, :]
    X, Y = s * np.sin(az), s * np.cos(az)
    at = RegularGridInterpolator((y, x), gridded.VIL.values, bounds_error=False)
    from_grid = at(np.stack([Y.ravel(), X.ravel()], -1)).reshape(X.shape)
    direct = polar.VIL.values
    use = (np.hypot(X, Y) > 30e3) & (np.hypot(X, Y) < 90e3) & (direct > 1.0)
    use &= np.isfinite(from_grid)
    assert use.sum() > 500
    rel = (from_grid[use] - direct[use]) / direct[use]
    assert abs(np.median(rel)) < 0.10
    assert np.percentile(np.abs(rel), 95) < 0.12
    # echo tops differ by less than the spacing of the grid levels and beams
    etop_p = echo_top(tree, engine=engine).values
    etop_g = echo_top(grid, missing="clear", engine=engine)
    at = RegularGridInterpolator((y, x), etop_g.values, bounds_error=False)
    tg = at(np.stack([Y.ravel(), X.ravel()], -1)).reshape(X.shape)
    both = use & np.isfinite(tg) & np.isfinite(etop_p)
    assert both.sum() > 300
    assert np.median(np.abs(tg[both] - etop_p[both])) < 1500.0


def test_volume_skips_sweeps_without_the_field_and_errors():
    tree = storm_volume(elevations=(0.5, 1.5, 3.0, 6.0))
    reference = vil(tree)
    other = tree["sweep_2"].to_dataset(inherit=False).rename({"DBZH": "OTHER"})
    nodes = {n: tree[n].to_dataset(inherit=False) for n in tree.children}
    nodes["sweep_2"] = other
    nodes["/"] = tree.root.to_dataset()
    partial = vil(xr.DataTree.from_dict(nodes))
    assert partial.VIL.shape == reference.VIL.shape
    assert not np.allclose(partial.VIL.values, reference.VIL.values, equal_nan=True)
    with pytest.raises(KeyError, match="no sweep contains"):
        vil(tree, "NOPE")
    only_other = xr.DataTree.from_dict(
        {
            "/": tree.root.to_dataset(),
            "sweep_0": tree["sweep_0"].to_dataset(inherit=False).rename({"DBZH": "X"}),
        }
    )
    with pytest.raises(KeyError):
        vil(only_other)
    with pytest.raises(ValueError, match="No sweep groups"):
        vil(xr.DataTree.from_dict({"/": tree.root.to_dataset()}), "DBZH")


def test_volume_options_and_beamwidth():
    tree = storm_volume(elevations=(0.5, 1.5, 3.0))
    a = echo_top(tree, interpolate=False, beamwidth=0.5)
    b = echo_top(tree, interpolate=False, beamwidth=2.0)
    ok = np.isfinite(a.values)
    assert (b.values[ok] > a.values[ok]).all()
    # the beam width is read from the volume when it is stored there
    params = xr.Dataset({"radar_beam_width_v": 2.0})
    nodes = {n: tree[n].to_dataset(inherit=False) for n in tree.children}
    nodes["/"] = tree.root.to_dataset()
    nodes["radar_parameters"] = params
    stored = echo_top(xr.DataTree.from_dict(nodes), interpolate=False)
    np.testing.assert_allclose(stored.values, b.values, equal_nan=True)
    # sweeps with the same elevation: the one that reaches farthest is used
    dup = {n: tree[n].to_dataset(inherit=False) for n in tree.children}
    dup["sweep_3"] = dup["sweep_0"].isel(range=slice(0, 50))
    dup["/"] = tree.root.to_dataset()
    again = vil(xr.DataTree.from_dict(dup))
    np.testing.assert_allclose(again.VIL, vil(tree).VIL, equal_nan=True)


def test_sector_scans_leave_columns_outside_the_azimuth_empty():
    tree = storm_volume(elevations=(0.5, 2.0, 5.0), nray=360)
    nodes = {n: tree[n].to_dataset(inherit=False) for n in tree.children}
    nodes["sweep_2"] = nodes["sweep_2"].isel(azimuth=slice(0, 90))
    nodes["/"] = tree.root.to_dataset()
    sector = echo_top(xr.DataTree.from_dict(nodes), interpolate=False)
    full = echo_top(tree, interpolate=False)
    inside = sector.values[:90]
    np.testing.assert_allclose(inside, full.values[:90], equal_nan=True)
    # outside the sector the highest sweep is missing: the top is the lower beam's
    ok = np.isfinite(full.values[200:]) & np.isfinite(sector.values[200:])
    assert (sector.values[200:][ok] <= full.values[200:][ok]).all()


# --------------------------------------------------------------------------
# input handling
# --------------------------------------------------------------------------


def test_input_errors_and_kinds():
    ds = column(np.array([10.0, 20.0, 30.0]))
    with pytest.raises(TypeError):
        vil(ds.DBZH)
    with pytest.raises(ValueError, match="kind must be"):
        vil(ds, kind="cube")
    with pytest.raises(ValueError, match="polar volume"):
        vil(storm_volume(elevations=(0.5, 1.5)), kind="grid")
    with pytest.raises(ValueError, match="DataTree"):
        vil(ds, kind="volume")
    with pytest.raises(ValueError, match="missing must be"):
        vil(ds, missing="maybe")
    with pytest.raises(ValueError, match="no_echo_dbz"):
        echo_top(ds, no_echo_dbz=20.0)
    with pytest.raises(KeyError, match="is not in the dataset"):
        vil(ds, "NOPE")
    assert vil(ds, "DBZH").VIL.shape == (1, 1)
    with pytest.raises(KeyError, match="pass the field name"):
        vil(ds.rename({"DBZH": "Q"}))
    flat = xr.Dataset({"DBZH": (("a", "b"), np.zeros((2, 2)))})
    with pytest.raises(ValueError, match="cannot tell"):
        vil(flat)
    with pytest.raises(ValueError, match="vertical dimension"):
        vil(ds.rename({"z": "w"}).assign_coords(x=[0.0]), kind="grid")
    # explicit kind on a profile-like dataset, different vertical names
    alt = xr.Dataset(
        {"DBZH": (("altitude", "y", "x"), np.full((3, 1, 1), 30.0))},
        coords={"altitude": [0.0, 500.0, 1000.0], "y": [0.0], "x": [0.0]},
    )
    np.testing.assert_allclose(
        vil(alt).VIL.values.squeeze(), K * z_of(30.0) ** (4 / 7) * 1000.0, rtol=1e-6
    )
    # a single level: the top edge is the level itself
    one = xr.Dataset(
        {"DBZH": (("z", "y", "x"), np.full((1, 1, 1), 30.0))},
        coords={"z": [500.0], "y": [0.0], "x": [0.0]},
    )
    assert float(echo_top(one).squeeze()) == 500.0


def test_site_altitude_sets_the_base():
    ds = column(np.array([30.0, 30.0, 30.0]), altitude=250.0, z=[500.0, 1000.0, 1500.0])
    out = vil(ds)
    assert out.attrs["vil_base_height"] == 250.0
    assert int(out.VIL_LOWER_BOUND.squeeze()) == 1
    filled = vil(ds, fill_below=True)
    np.testing.assert_allclose(
        filled.VIL.values.squeeze(), K * z_of(30.0) ** (4 / 7) * 1250.0, rtol=1e-6
    )
    assert (
        vil(column(np.array([30.0, 30.0]), z=[0.0, 100.0])).attrs["vil_base_height"]
        == 0.0
    )
    # more than one time on a grid keeps the leading dimension
    ds4 = xr.concat([ds, ds], dim=xr.DataArray([0, 1], dims="time", name="time"))
    assert vil(ds4).VIL.dims[0] == "time"


def test_accessors():
    z = np.arange(0, 6000.0, 500.0)
    ds = column(np.full(z.size, 35.0), z)
    xr.testing.assert_identical(ds.radarx.vil(), vil(ds))
    xr.testing.assert_identical(ds.radarx.echo_top(), echo_top(ds))
    xr.testing.assert_identical(ds.radarx.vil_density(), vil_density(ds))
    xr.testing.assert_identical(
        ds.radarx.liquid_water_content(), liquid_water_content(ds)
    )
    tree = storm_volume(elevations=(0.5, 1.5, 3.0))
    xr.testing.assert_identical(tree.radarx.vil(), vil(tree))


# --------------------------------------------------------------------------
# liquid water content
# --------------------------------------------------------------------------


def test_lwc_power_law_closed_form():
    dbz = np.array([10.0, 30.0, 40.0, 50.0])
    ds = xr.Dataset({"DBZH": ("gate", dbz)}, coords={"gate": np.arange(4)})
    lwc = liquid_water_content(ds)
    np.testing.assert_allclose(lwc, 3.44e-3 * z_of(dbz) ** (4 / 7), rtol=1e-6)
    assert lwc.attrs["units"] == "g m-3" and lwc.name == "LWC"
    assert "valid in rain only" in lwc.attrs["comment"]
    kg = liquid_water_content(ds, units="kg m-3")
    np.testing.assert_allclose(kg, lwc * 1e-3, rtol=1e-6)
    assert kg.attrs["units"] == "kg m-3"
    # 40 dBZ gives about 0.66 g m-3 (3.44e-3 * 10^(16/7))
    np.testing.assert_allclose(float(lwc[2]), 0.664, atol=1e-3)
    # the gridded VIL of a column of constant LWC is LWC * depth
    z = np.arange(0.0, 2001.0, 500.0)
    col = column(np.full(z.size, 40.0), z)
    total = float(vil(col, dbz_cap=None).VIL.squeeze())
    np.testing.assert_allclose(total, float(kg[2]) * 2000.0, rtol=1e-6)


def test_lwc_mask_names_and_tree():
    tree = storm_volume(elevations=(0.5, 1.5, 3.0))
    sweep = tree["sweep_0"].to_dataset(inherit=False)
    sweep["RAIN"] = sweep.DBZH > 20
    out = liquid_water_content(sweep, mask="RAIN")
    assert np.isnan(out.values[~sweep.RAIN.values]).all()
    assert np.isfinite(out.values[sweep.RAIN.values]).all()
    out = liquid_water_content(sweep, mask=sweep.DBZH > 30)
    assert np.isnan(out.values[sweep.DBZH.values <= 30]).all()
    res = liquid_water_content(tree)
    assert sorted(res.children) == ["sweep_0", "sweep_1", "sweep_2"]
    assert res["sweep_1"].ds.LWC.attrs["units"] == "g m-3"
    masks = {"sweep_0": tree["sweep_0"].ds.DBZH > 25}
    res = liquid_water_content(tree, mask=masks)
    assert np.isnan(
        res["sweep_0"].ds.LWC.values[tree["sweep_0"].ds.DBZH.values <= 25]
    ).all()
    assert np.isfinite(res["sweep_1"].ds.LWC).sum() > 0
    # a Dataset-valued mask node must hold one variable
    node = xr.DataTree.from_dict(
        {"sweep_0": xr.Dataset({"a": tree["sweep_0"].ds.DBZH > 25})}
    )
    res = liquid_water_content(tree, mask=node)
    assert np.isfinite(res["sweep_0"].ds.LWC).sum() > 0
    bad = xr.DataTree.from_dict(
        {
            "sweep_0": xr.Dataset(
                {"a": tree["sweep_0"].ds.DBZH > 1, "b": tree["sweep_0"].ds.DBZH > 1}
            )
        }
    )
    with pytest.raises(ValueError, match="exactly one"):
        liquid_water_content(tree, mask=bad)
    # sweeps without the field are skipped
    nodes = {n: tree[n].to_dataset(inherit=False) for n in tree.children}
    nodes["sweep_1"] = nodes["sweep_1"].rename({"DBZH": "OTHER"})
    nodes["/"] = tree.root.to_dataset()
    res = liquid_water_content(xr.DataTree.from_dict(nodes))
    assert sorted(res.children) == ["sweep_0", "sweep_2"]
    with pytest.raises(KeyError, match="no sweep"):
        liquid_water_content(
            xr.DataTree.from_dict(
                {"/": tree.root.to_dataset(), "sweep_0": sweep.rename({"DBZH": "Q"})}
            )
        )


def dsd_sweep():
    """Rain with a known gamma DSD in every gate (as in the DSD tests)."""
    mod = __import__("radarx.retrieve.dsd", fromlist=["x"])
    rnd = np.random.default_rng(0)
    dm = rnd.uniform(0.8, 2.5, 60)
    nw = 10 ** rnd.uniform(3.0, 4.2, 60)
    mu = 3.0
    lam = (4 + mu) / dm
    n0 = nw * mod._f_mu(mu) * dm ** (-mu)
    zh, zv, kdp = mod._gamma_integrals("S", 20.0, n0, mu, lam)
    ds = xr.Dataset(
        {
            "DBZH": ("gate", 10 * np.log10(zh), {"units": "dBZ"}),
            "ZDR": ("gate", 10 * np.log10(zh / zv), {"units": "dB"}),
            "KDP": ("gate", kdp),
        },
        coords={"gate": np.arange(60)},
    )
    return ds, nw, dm, mu


@pytest.mark.parametrize("engine", ENGINES)
def test_lwc_dsd_equals_the_dsd_output(engine):
    ds, nw, dm, mu = dsd_sweep()
    lwc = liquid_water_content(ds, "dsd", band="S", engine=engine)
    ref = dsd(ds, band="S", engine=engine)
    np.testing.assert_allclose(lwc, ref.LWC, rtol=1e-12)
    assert lwc.attrs["units"] == "g m-3" and "gamma" in lwc.attrs["comment"]
    kg = liquid_water_content(ds, "dsd", band="S", units="kg m-3", engine=engine)
    np.testing.assert_allclose(kg, ref.LWC * 1e-3, rtol=1e-12)
    with_kdp = liquid_water_content(ds, "dsd", band="S", kdp="KDP", engine=engine)
    assert np.isfinite(with_kdp).all()
    # mask: gates outside are NaN
    m = ds.DBZH > float(ds.DBZH.median())
    masked = liquid_water_content(ds, "dsd", band="S", mask=m, engine=engine)
    assert np.isnan(masked.values[~m.values]).all()
    # and the two methods agree within the spread of real drop size
    # distributions (the power law assumes Marshall-Palmer)
    zm = liquid_water_content(ds)
    ratio = (zm / lwc).values
    assert 0.2 < np.nanmedian(ratio) < 5.0


def test_lwc_dsd_volume_and_errors():
    ds, *_ = dsd_sweep()
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(
                coords={"latitude": 33.9, "longitude": -88.3},
                attrs={"scan_name": "VCP-212"},
            ),
            "sweep_0": ds,
        }
    )
    res = liquid_water_content(tree, "dsd", band="S")
    ref = dsd(ds, band="S")
    np.testing.assert_allclose(res["sweep_0"].ds.LWC, ref.LWC, rtol=1e-12)
    assert res["sweep_0"].ds.LWC.attrs["units"] == "g m-3"
    with pytest.raises(ValueError, match="method"):
        liquid_water_content(ds, "ice")
    with pytest.raises(ValueError, match="units"):
        liquid_water_content(ds, units="mg m-3")
    with pytest.raises(ValueError, match="method='dsd'"):
        liquid_water_content(ds, band="S")
    with pytest.raises(TypeError):
        liquid_water_content(ds.DBZH)
    with pytest.raises(KeyError):
        liquid_water_content(ds, "dsd", band="S", zdr="NOPE")
    with pytest.raises(KeyError):
        liquid_water_content(ds, dbz="NOPE")


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------


def test_real_nexrad_volume():
    """KLBB (open-radar-data): plausible VIL, echo top and VIL density."""
    pytest.importorskip("open_radar_data")
    from open_radar_data import DATASETS

    from .test_dealias import nexrad_volume

    try:
        path = DATASETS.fetch("KLBB20160601_150025_V06")
    except Exception as err:  # pragma: no cover - network
        pytest.skip(f"sample data unavailable: {err}")
    tree = nexrad_volume(path)
    out = vil(tree)
    assert out.VIL.dims == ("azimuth", "range")
    finite = out.VIL.values[np.isfinite(out.VIL.values)]
    assert finite.size > 1000 and finite.min() >= 0.0 and finite.max() < 80.0
    assert (out.VIL_LOWER_BOUND.values[np.isfinite(out.VIL.values)] == 1).all()
    top = echo_top(tree)
    t = top.values[np.isfinite(top.values)]
    assert t.size > 100 and 0.0 < np.nanmax(t) < 25e3
    density = vil_density(tree)
    d = density.values[np.isfinite(density.values)]
    assert d.size > 100 and d.min() >= 0.0 and d.max() < 20.0
    if vilmod.HAS_COMPILED_KERNEL:
        other = vil(tree, engine="numpy")
        xr.testing.assert_allclose(out, other, rtol=1e-6)
