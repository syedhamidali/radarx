#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for Radarx interactive (hvplot) plots
===========================================
"""
import numpy as np
import pytest
import xarray as xr

hv = pytest.importorskip("holoviews")
pytest.importorskip("hvplot")

import radarx  # noqa: E402,F401
from radarx.vis import interactive  # noqa: E402

VAR = "DBZH"


@pytest.fixture
def sweep():
    """Small synthetic PPI sweep without x/y/z coordinates."""
    azimuth = np.arange(0.5, 360, 10.0)
    rng = np.arange(250.0, 30_000.0, 1000.0)
    data = np.random.default_rng(0).uniform(-10, 60, (azimuth.size, rng.size))
    data[0, 0] = 3333.0  # unmasked outlier, as in real files
    ds = xr.Dataset(
        {
            VAR: (("azimuth", "range"), data, {"units": "dBZ"}),
            "VRADH": (("azimuth", "range"), data / 4, {"units": "m/s"}),
        },
        coords={
            "azimuth": azimuth,
            "range": rng,
            "elevation": ("azimuth", np.full(azimuth.size, 0.5)),
            "altitude": 300.0,
            "sweep_fixed_angle": 0.5,
        },
    )
    return ds


@pytest.fixture
def rhi_sweep():
    elevation = np.arange(0.5, 30, 1.0)
    rng = np.arange(250.0, 20_000.0, 1000.0)
    data = np.random.default_rng(1).uniform(0, 50, (elevation.size, rng.size))
    return xr.DataArray(
        data,
        dims=("elevation", "range"),
        coords={
            "elevation": elevation,
            "range": rng,
            "azimuth": ("elevation", np.full(elevation.size, 90.0)),
        },
        name=VAR,
    )


@pytest.fixture
def grid():
    """Small synthetic 3D grid as returned by ``to_grid``."""
    x = np.arange(-20e3, 20e3 + 1, 2e3)
    z = np.arange(0, 5e3 + 1, 1e3)
    data = np.random.default_rng(2).uniform(0, 50, (z.size, x.size, x.size))
    return xr.Dataset(
        {VAR: (("z", "y", "x"), data, {"units": "dBZ"})},
        coords={"z": z, "y": x, "x": x},
    )


@pytest.fixture
def dtree(sweep):
    sweeps = {
        f"sweep_{i}": sweep.assign_coords(
            elevation=sweep.elevation + i, sweep_fixed_angle=0.5 + i
        )
        for i in range(2)
    }
    return xr.DataTree.from_dict({"/": xr.Dataset(), **sweeps})


def _render(plot, backend="bokeh"):
    hv.render(plot, backend=backend)
    return plot


def test_dataarray_default_is_range_azimuth(sweep):
    plot = _render(sweep[VAR].radarx.plot())
    assert isinstance(plot, hv.QuadMesh)
    assert plot.kdims[0].name == "range"


def test_ppi_georeferences_and_uses_km(sweep):
    plot = _render(sweep[VAR].radarx.plot.ppi())
    assert [d.name for d in plot.kdims] == ["x", "y"]
    xs = plot.dimension_values("x")
    assert np.nanmax(np.abs(xs)) < 31  # km, not m
    assert "0.5°" in plot.opts.get().kwargs["title"]


def _color_range(plot):
    mapper = hv.renderer("bokeh").get_plot(plot).handles["color_mapper"]
    return mapper.low, mapper.high


def test_ppi_robust_default_and_explicit_clim(sweep):
    low, high = _color_range(sweep[VAR].radarx.plot.ppi())
    assert high < 100  # outlier of 3333 ignored
    assert low > -11  # not forced symmetric around zero
    assert _color_range(sweep[VAR].radarx.plot.ppi(clim=(0, 60))) == (0, 60)
    low, high = _color_range(sweep["VRADH"].radarx.plot.ppi(symmetric=True))
    assert low == -high


def test_existing_xyz_coords_are_reused(sweep):
    da = sweep[VAR]
    geo = interactive._georeference(da)
    km = interactive._georeference(
        da.assign_coords(x=geo.x * 1e3, y=geo.y * 1e3, z=geo.z * 1e3)
    )
    np.testing.assert_allclose(km.x, geo.x)


def test_mesh_and_centroids(sweep):
    mesh = _render(sweep[VAR].radarx.plot.mesh())
    assert mesh.opts.get().kwargs["line_color"] == "black"
    points = _render(sweep[VAR].radarx.plot.centroids())
    assert isinstance(points, hv.Points)
    assert len(points) == sweep[VAR].size


def test_rhi(rhi_sweep):
    plot = _render(rhi_sweep.radarx.plot.rhi())
    assert [d.name for d in plot.kdims] == ["ground_range", "z"]


def test_dataset_facets_over_variables(sweep):
    layout = _render(sweep.radarx.plot.ppi())
    assert isinstance(layout, hv.Layout)
    assert len(layout) == 2
    single = sweep.radarx.plot.ppi(VAR)
    assert isinstance(single, hv.QuadMesh)


def test_dataset_without_radar_variables_raises():
    with pytest.raises(ValueError, match="No radar variables"):
        xr.Dataset({"a": ("t", [1, 2])}).radarx.plot.ppi()


def test_datatree_facets_over_sweeps(dtree):
    layout = _render(dtree.radarx.plot.ppi(VAR))
    assert len(layout) == 2
    single = dtree.radarx.plot.ppi(VAR, sweeps=1)
    assert "1.5°" in single.opts.get().kwargs["title"]
    assert isinstance(dtree.radarx.plot(VAR, sweeps="sweep_0"), hv.QuadMesh)


def test_grid_cappi_and_max_cappi(grid):
    level = _render(grid.radarx.plot.cappi(VAR, z=2000))
    assert "2.0 km" in level.opts.get().kwargs["title"]
    slider = grid[VAR].radarx.plot.cappi()
    assert isinstance(slider, hv.DynamicMap)
    max_cappi = _render(grid.radarx.plot.max_cappi(VAR))
    assert isinstance(max_cappi, hv.Layout)
    assert len(max_cappi) == 4
    with pytest.raises(ValueError, match="3D grid"):
        grid[VAR].isel(z=0).radarx.plot.max_cappi()


def test_fallthrough_to_hvplot(sweep):
    plot = sweep[VAR].radarx.plot.hist()
    assert plot is not None
    with pytest.raises(AttributeError):
        _ = sweep[VAR].radarx.plot._private


def test_matplotlib_backend_keeps_mpl_backend(sweep, grid):
    import matplotlib as mpl

    before = mpl.get_backend()
    try:
        _render(sweep[VAR].radarx.plot.mesh(backend="matplotlib"), "matplotlib")
        _render(grid.radarx.plot.max_cappi(VAR, backend="matplotlib"), "matplotlib")
        assert mpl.get_backend() == before
    finally:
        interactive._assign_backend("bokeh")


def test_invalid_backend(sweep):
    with pytest.raises(ValueError, match="Unsupported backend"):
        sweep[VAR].radarx.plot.ppi(backend="plotly")


def test_missing_range_raises():
    da = xr.DataArray(np.zeros((2, 2)), dims=("a", "b"), name=VAR)
    with pytest.raises(ValueError, match="range"):
        interactive.hvplot_range_azimuth(da)
    with pytest.raises(ValueError, match="georeference"):
        interactive.hvplot_ppi(da)


def test_imd_volume_from_xradar():
    """IMD data read by xradar's backend plots through the accessor."""
    xd = pytest.importorskip("xradar")
    if not hasattr(xd.io, "open_imd_datatree"):
        pytest.skip("xradar release without the IMD backend")
    from open_radar_data import DATASETS

    files = [DATASETS.fetch(f"IMD/JPR220822135253-IMD-B.nc{s}") for s in ("", ".1")]
    dtree = xd.io.open_imd_datatree(files)
    _render(dtree.radarx.plot.ppi("DBZH"))
    _render(dtree.radarx.plot.cappi("DBZH", height=3000, x_res=4000, y_res=4000))


def test_large_fields_rasterize_by_default(sweep, monkeypatch):
    """Big polar meshes are rasterized (if datashader exists); mesh never is."""
    pytest.importorskip("datashader")
    monkeypatch.setattr(interactive, "RASTERIZE_THRESHOLD", 100)
    da = sweep[VAR]
    assert isinstance(da.radarx.plot.ppi(), hv.DynamicMap)
    assert isinstance(da.radarx.plot.ppi(rasterize=False), hv.QuadMesh)
    assert isinstance(da.radarx.plot.mesh(), hv.QuadMesh)
    _render(da.radarx.plot.ppi())
