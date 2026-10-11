#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests of radarx.retrieve.beam_path (the ray tracer on the gates of a sweep)."""

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.fundamentals import trace_rays
from radarx.fundamentals.constants import EARTH_RADIUS
from radarx.fundamentals.geometry import beam_center_height
from radarx.io import sounding
from radarx.retrieve import beam_path

from .test_beam_blockage import ALT0, make_sweep
from .test_beam_blockage import volume as make_volume

STANDARD = 1e6 / (4.0 * EARTH_RADIUS / 1000.0)
HEIGHTS = np.arange(0.0, 20001.0, 500.0)


def standard_n(gradient=STANDARD, heights=HEIGHTS):
    return 313.0 - gradient * heights / 1000.0


def moist_profile():
    """A profile with every humidity variable, consistent with each other."""
    z = np.arange(0.0, 15001.0, 500.0)
    p = 101325.0 * (1.0 - 2.25577e-5 * z) ** 5.25588
    t = 288.15 - 0.0065 * z
    t = np.maximum(t, 216.65)
    es = sounding.saturation_vapor_pressure(t).values
    e = 0.6 * es * np.exp(-z / 3000.0)
    q = sounding.specific_humidity_from_dewpoint(
        sounding.dewpoint_from_vapor_pressure(e), p
    ).values
    ds = xr.Dataset(
        {
            "pressure": ("height", p),
            "temperature": ("height", t),
            "vapor_pressure": ("height", e),
            "specific_humidity": ("height", q),
            "dewpoint": ("height", sounding.dewpoint_from_vapor_pressure(e).values),
            "relative_humidity": ("height", e / es),
        },
        coords={"height": z},
    )
    return ds


@pytest.mark.parametrize("engine", ["numpy", "auto"])
def test_sweep_products_match_the_tracer(engine):
    ds = make_sweep(nray=6, ngate=120, dr=1000.0, elevation=1.0)
    out = beam_path(ds, (HEIGHTS, standard_n()), engine=engine)
    ref = trace_rays(
        ds["range"], 1.0, HEIGHTS, standard_n(), radar_height=ALT0, engine=engine
    )
    assert out.beam_height.dims == ("azimuth", "range")
    assert out.beam_height.dtype == np.float32
    for name in ("beam_height", "ground_range", "gate_elevation"):
        np.testing.assert_allclose(out[name].isel(azimuth=0), ref[name], rtol=1e-6)
        np.testing.assert_array_equal(
            out[name].isel(azimuth=0), out[name].isel(azimuth=3)
        )
    # 4/3 Earth: k_e is 4/3 and the height is that of the effective radius model
    np.testing.assert_allclose(out.k_e, 4.0 / 3.0, atol=1e-3)
    h43 = beam_center_height(ds["range"].values, 1.0, ALT0, 4.0 / 3.0 * EARTH_RADIUS)
    np.testing.assert_allclose(out.beam_height.isel(azimuth=0), h43, atol=1.0)
    # the coordinates of the sweep are on the products, with their attributes
    assert set(out.coords) >= {"azimuth", "range", "elevation"}
    assert out.attrs["beam_path_radar_height"] == ALT0
    assert out.beam_height.attrs["units"] == "m"
    assert out.k_e.dims == out.beam_height.dims


def test_elevation_per_ray_and_scalar_sweep_angle():
    ds = make_sweep(nray=4, ngate=60, dr=1000.0)
    ds["elevation"] = ("azimuth", np.array([0.5, 1.0, 2.0, np.nan]))
    out = beam_path(ds, (HEIGHTS, standard_n()))
    assert float(out.beam_height.isel(azimuth=1, range=-1)) > float(
        out.beam_height.isel(azimuth=0, range=-1)
    )
    assert np.isnan(out.beam_height.isel(azimuth=3)).all()
    # sweep_fixed_angle (a scalar) in place of the per-ray elevation, rays by time
    ds = make_sweep(nray=4, ngate=60, dr=1000.0, ray="time").drop_vars("elevation")
    ds = ds.assign_coords(sweep_fixed_angle=1.5)
    out = beam_path(ds, (HEIGHTS, standard_n()))
    assert out.beam_height.dims == ("time", "range")
    ref = trace_rays(ds["range"], 1.5, HEIGHTS, standard_n(), radar_height=ALT0)
    np.testing.assert_allclose(out.beam_height.isel(time=0), ref.beam_height, rtol=1e-6)


def test_dataarray_and_volume_and_accessors():
    ds = make_sweep(nray=6, ngate=60, dr=1000.0)
    prof = (HEIGHTS, standard_n())
    ref = beam_path(ds, prof)
    from_field = beam_path(ds["DBZH"], prof)
    xr.testing.assert_allclose(from_field.beam_height, ref.beam_height)
    np.testing.assert_array_equal(ds.radarx.beam_path(prof)["k_e"], ref["k_e"])
    np.testing.assert_array_equal(
        ds["DBZH"].radarx.beam_path(prof).beam_height, ref.beam_height
    )
    tree = make_volume()
    out = beam_path(tree, prof)
    assert isinstance(out, xr.DataTree)
    assert sorted(out.children) == ["sweep_0", "sweep_1"]
    assert float(out.latitude) == float(tree.latitude)  # the root of the input
    for k, el in enumerate((0.5, 1.5)):
        node = out[f"sweep_{k}"].to_dataset()
        single = make_sweep(nray=6, ngate=80, elevation=el, beam=None)
        np.testing.assert_allclose(
            node.beam_height.isel(azimuth=0),
            beam_path(single, prof).beam_height.isel(azimuth=0),
            rtol=1e-6,
        )
    assert float(out["sweep_1"].beam_height.max()) > float(
        out["sweep_0"].beam_height.max()
    )
    via = tree.radarx.beam_path(prof)
    xr.testing.assert_allclose(via["sweep_0"].to_dataset(), out["sweep_0"].to_dataset())
    # a single sweep node of a volume stands for a sweep
    one = beam_path(tree["sweep_0"], prof)
    assert isinstance(one, xr.Dataset)
    # the products merge into the sweep
    merged = ds.radarx.assign(ref)
    assert "beam_height" in merged and "DBZH" in merged


def test_refraction_bends_the_beam():
    ds = make_sweep(nray=2, ngate=100, dr=2000.0, elevation=0.5)
    k = {}
    for name, gradient in (("sub", 20.0), ("standard", STANDARD), ("super", 80.0)):
        out = beam_path(ds, (HEIGHTS[:25], standard_n(gradient, HEIGHTS[:25])))
        k[name] = float(out.k_e.median())
    assert k["sub"] < k["standard"] < k["super"]
    assert k["standard"] == pytest.approx(4.0 / 3.0, abs=1e-3)
    # 1 / (1 - a dn/dh): the effective radius factor of a constant gradient
    for name, gradient in (("sub", 20.0), ("super", 80.0)):
        expected = 1.0 / (1.0 - EARTH_RADIUS * gradient * 1e-9)
        assert k[name] == pytest.approx(expected, rel=2e-3)


@pytest.mark.parametrize(
    "name", ["vapor_pressure", "specific_humidity", "dewpoint", "relative_humidity"]
)
def test_refractivity_from_the_humidity_variables(name):
    prof = moist_profile()
    full = prof.copy()
    keep = prof[["pressure", "temperature", name]]
    ds = make_sweep(nray=2, ngate=60, dr=2000.0, elevation=0.5)
    ref = beam_path(ds, full)  # vapor_pressure is used first
    out = beam_path(ds, keep)
    np.testing.assert_allclose(out.beam_height, ref.beam_height, atol=0.05)
    n = sounding.refractivity(prof.pressure, prof.temperature, prof.vapor_pressure)
    explicit = beam_path(ds, prof[["pressure"]].assign(refractivity=n))
    np.testing.assert_allclose(explicit.beam_height, ref.beam_height, atol=1e-3)
    # the other formulas change the height by less than a metre
    sw = beam_path(ds, full, form="smith_weintraub")
    assert float(np.abs(sw.beam_height - ref.beam_height).max()) < 1.0
    assert float(np.abs(sw.beam_height - ref.beam_height).max()) > 0.0


def test_profile_errors():
    ds = make_sweep(nray=2, ngate=20, dr=1000.0)
    prof = moist_profile()
    with pytest.raises(ValueError, match="water vapour"):
        beam_path(ds, prof[["pressure", "temperature"]])
    with pytest.raises(ValueError, match="pressure"):
        beam_path(ds, prof[["temperature", "relative_humidity"]])
    with pytest.raises(ValueError, match="temperature"):
        beam_path(ds, prof[["pressure", "relative_humidity"]])
    with pytest.raises(ValueError, match="'height' coordinate"):
        beam_path(ds, xr.Dataset({"refractivity": ("level", standard_n())}))
    with pytest.raises(TypeError, match="pair"):
        beam_path(ds, 5)
    with pytest.raises(TypeError, match="pair"):
        beam_path(ds, "profile")
    with pytest.raises(TypeError, match="xarray"):
        beam_path(np.zeros(3), (HEIGHTS, standard_n()))
    with pytest.raises(ValueError, match="form must be"):
        beam_path(ds, prof[["pressure", "temperature", "vapor_pressure"]], form="x")


def test_sweep_errors():
    prof = (HEIGHTS, standard_n())
    ds = make_sweep(nray=2, ngate=20, dr=1000.0)
    with pytest.raises(ValueError, match="'range' coordinate"):
        beam_path(ds.drop_vars("range"), prof)
    with pytest.raises(ValueError, match="elevation"):
        beam_path(ds.drop_vars("elevation"), prof)
    with pytest.raises(ValueError, match="altitude"):
        beam_path(ds.drop_vars("altitude"), prof)
    out = beam_path(ds.drop_vars("altitude"), prof, radar_height=100.0)
    assert out.attrs["beam_path_radar_height"] == 100.0
    with pytest.raises(ImportError):
        import radarx.fundamentals.refraction as rf

        old = rf.HAS_COMPILED_KERNEL
        rf.HAS_COMPILED_KERNEL = False
        try:
            beam_path(ds, prof, engine="compiled")
        finally:
            rf.HAS_COMPILED_KERNEL = old


def test_dask_sweep_gives_a_lazy_result():
    dask = pytest.importorskip("dask.array")
    ds = make_sweep(nray=24, ngate=40, dr=1000.0, elevation=1.0)
    ds["elevation"] = ("azimuth", np.linspace(0.5, 3.0, 24))
    prof = (HEIGHTS, standard_n())
    eager = beam_path(ds, prof)
    lazy = beam_path(ds.chunk({"azimuth": 8}), prof)
    assert isinstance(lazy.beam_height.data, dask.Array)
    assert lazy.beam_height.data.chunks[0] == (8, 8, 8)
    for name in ("beam_height", "ground_range", "gate_elevation", "k_e"):
        np.testing.assert_allclose(lazy[name].compute(), eager[name], rtol=1e-6)
    # a scalar elevation on a chunked sweep
    ds2 = make_sweep(nray=24, ngate=40, dr=1000.0).drop_vars("elevation")
    ds2 = ds2.assign_coords(sweep_fixed_angle=1.0).chunk({"azimuth": 12})
    lazy2 = beam_path(ds2, prof)
    assert isinstance(lazy2.beam_height.data, dask.Array)
    ref2 = trace_rays(ds["range"], 1.0, HEIGHTS, standard_n(), radar_height=ALT0)
    np.testing.assert_allclose(
        lazy2.beam_height.compute().isel(azimuth=0), ref2.beam_height, rtol=1e-6
    )


def test_provenance_and_input_unchanged():
    ds = make_sweep(nray=4, ngate=20, dr=1000.0)
    before = ds.copy(deep=True)
    prof = moist_profile()
    prof_before = prof.copy(deep=True)
    out = beam_path(ds, prof)
    xr.testing.assert_identical(ds, before)
    xr.testing.assert_identical(prof, prof_before)
    assert out.attrs["radarx_references"] == "10.6028/NBS.MONO.92"
    assert "Snell" in out.attrs["radarx_method"]
    for name in out.data_vars:
        assert out[name].attrs["radarx_references"] == "10.6028/NBS.MONO.92"
    assert "beam_path(" in out.attrs["history"]
