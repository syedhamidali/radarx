#!/usr/bin/env python

"""Tests for CAPPI retrieval and basic plotting helpers."""

import warnings

import matplotlib

import numpy as np
import pytest
import xarray as xr

import matplotlib.pyplot as plt

import radarx  # noqa: F401
from radarx.retrieve import create_cappi
from radarx.vis import plot_cappi, plot_ppi

matplotlib.use("Agg")

try:
    from xarray import DataTree
except ImportError:  # pragma: no cover
    from datatree import DataTree

try:
    from xradar.transform.cfradial import to_cfradial2 as xradar_to_cfradial2
except ImportError:  # pragma: no cover
    xradar_to_cfradial2 = None


def _make_ppi_sweep(elevation_deg, time_value):
    azimuth = np.array([0.0, 90.0, 180.0, 270.0], dtype=float)
    ranges = np.array([1000.0, 2000.0, 3000.0], dtype=float)

    azimuth_2d, range_2d = np.meshgrid(np.deg2rad(azimuth), ranges, indexing="ij")
    elevation_rad = np.deg2rad(elevation_deg)

    x = range_2d * np.sin(azimuth_2d)
    y = range_2d * np.cos(azimuth_2d)
    z = 100.0 + range_2d * np.sin(elevation_rad)
    dbzh = elevation_deg * 10.0 + (range_2d / 1000.0) + np.arange(len(azimuth))[:, None]

    return xr.Dataset(
        data_vars={
            "DBZH": (
                ("azimuth", "range"),
                dbzh,
                {"units": "dBZ"},
            ),
            "VRADH": (
                ("azimuth", "range"),
                np.full_like(dbzh, elevation_deg),
                {"units": "m s-1"},
            ),
        },
        coords={
            "azimuth": ("azimuth", azimuth),
            "range": ("range", ranges),
            "x": (("azimuth", "range"), x),
            "y": (("azimuth", "range"), y),
            "z": (("azimuth", "range"), z),
            "time": np.datetime64(time_value),
            "latitude": 28.61,
            "longitude": 77.23,
            "altitude": 100.0,
        },
        attrs={
            "instrument_name": "MockRadar",
            "sweep_mode": "azimuth_surveillance",
        },
    )


def _make_rhi_dataset():
    elevations = np.array([0.5, 2.0, 4.0, 6.0], dtype=float)
    ranges = np.array([1000.0, 2000.0, 3000.0], dtype=float)

    elevation_2d, range_2d = np.meshgrid(np.deg2rad(elevations), ranges, indexing="ij")
    x = range_2d * np.cos(elevation_2d)
    y = np.zeros_like(x)
    z = 100.0 + range_2d * np.sin(elevation_2d)
    dbzh = 15.0 + range_2d / 1000.0 + np.arange(len(elevations))[:, None]

    return xr.Dataset(
        data_vars={
            "DBZH": (
                ("elevation", "range"),
                dbzh,
                {"units": "dBZ"},
            )
        },
        coords={
            "elevation": ("elevation", elevations),
            "range": ("range", ranges),
            "x": (("elevation", "range"), x),
            "y": (("elevation", "range"), y),
            "z": (("elevation", "range"), z),
            "time": np.datetime64("2024-01-01T00:00:00"),
            "latitude": 28.61,
            "longitude": 77.23,
            "altitude": 100.0,
        },
        attrs={
            "instrument_name": "MockRadar",
            "sweep_mode": "rhi",
        },
    )


def _make_gridded_dataset():
    x = np.array([-2000.0, 0.0, 2000.0], dtype=float)
    y = np.array([-2000.0, 0.0, 2000.0], dtype=float)
    z = np.array([1000.0, 2000.0, 3000.0], dtype=float)

    yy, xx = np.meshgrid(y, x, indexing="ij")
    lon = 77.23 + xx / 111_000.0
    lat = 28.61 + yy / 111_000.0

    data = np.stack(
        [
            20.0 + yy / 1000.0 + xx / 2000.0,
            25.0 + yy / 1000.0 + xx / 2000.0,
            30.0 + yy / 1000.0 + xx / 2000.0,
        ],
        axis=0,
    )

    return xr.Dataset(
        data_vars={
            "DBZH": (
                ("z", "y", "x"),
                data,
                {"units": "dBZ"},
            )
        },
        coords={
            "x": ("x", x),
            "y": ("y", y),
            "z": ("z", z),
            "lon": (("y", "x"), lon),
            "lat": (("y", "x"), lat),
            "longitude": 77.23,
            "latitude": 28.61,
            "time": np.datetime64("2024-01-01T00:00:00"),
        },
        attrs={"instrument_name": "MockRadar"},
    )


@pytest.fixture
def synthetic_volume():
    return DataTree.from_dict(
        {
            "sweep_0": _make_ppi_sweep(1.0, "2024-01-01T00:00:00"),
            "sweep_1": _make_ppi_sweep(3.0, "2024-01-01T00:00:00"),
            "sweep_2": _make_ppi_sweep(5.0, "2024-01-01T00:00:00"),
        }
    )


def test_create_cappi_function(synthetic_volume):
    ds = create_cappi(
        synthetic_volume,
        height=150.0,
        fields=["DBZH"],
        x_res=1000.0,
        y_res=1000.0,
    )

    assert "DBZH" in ds.data_vars
    assert ds["DBZH"].dims == ("y", "x")
    assert ds.attrs["product"] == "CAPPI"
    assert float(ds["z"].item()) == 150.0
    assert np.isfinite(ds["DBZH"].values).any()


def test_create_cappi_accessor(synthetic_volume):
    ds = synthetic_volume.radarx.create_cappi(
        height=150.0,
        fields=["DBZH"],
        x_res=1000.0,
        y_res=1000.0,
    )

    assert "DBZH" in ds.data_vars
    assert ds.attrs["product"] == "CAPPI"


def test_create_cappi_custom_cartesian_grid(synthetic_volume):
    ds = create_cappi(
        synthetic_volume,
        height=150.0,
        fields=["DBZH"],
        x=np.array([-1500.0, 0.0, 1500.0]),
        y=np.array([-1000.0, 1000.0]),
    )

    assert ds["DBZH"].shape == (2, 3)
    assert np.allclose(ds["x"].values, [-1500.0, 0.0, 1500.0])
    assert np.allclose(ds["y"].values, [-1000.0, 1000.0])


def test_create_cappi_polar_method(synthetic_volume):
    ds = create_cappi(
        synthetic_volume,
        height=150.0,
        fields=["DBZH"],
        method="polar_vertical_interpolation",
    )

    assert "DBZH" in ds.data_vars
    assert ds["DBZH"].dims == ("time", "range")
    assert ds.attrs["method"] == "polar_vertical_interpolation"
    assert ds.attrs["vertical_interpolation"] == "linear"
    assert "longitude" in ds.coords
    assert "latitude" in ds.coords
    assert "altitude" in ds.coords
    assert "elevation" in ds.coords
    assert float(ds["altitude"].item()) == 150.0
    assert float(ds.attrs["radar_altitude"]) == 100.0
    assert "sweep_number" in ds.data_vars
    assert "fixed_angle" in ds.data_vars


def test_create_cappi_polar_accessor(synthetic_volume):
    ds = synthetic_volume.radarx.create_cappi(
        height=150.0,
        fields=["DBZH"],
        method="polar_vertical_interpolation",
    )

    assert ds["DBZH"].dims == ("time", "range")
    assert ds.attrs["method"] == "polar_vertical_interpolation"


def test_create_cappi_legacy_method_aliases(synthetic_volume):
    ds = create_cappi(
        synthetic_volume,
        height=150.0,
        fields=["DBZH"],
        method="polar",
    )

    assert ds.attrs["method"] == "polar_vertical_interpolation"
    assert ds.attrs["vertical_interpolation"] == "linear"


def test_create_cappi_height_window_legacy_alias(synthetic_volume):
    ds = create_cappi(
        synthetic_volume,
        height=150.0,
        fields=["DBZH"],
        method="pseudo_cappi",
    )

    assert ds.attrs["method"] == "height_window_composite"
    assert bool(ds.attrs["apply_quality_control"]) is False


def test_create_cappi_height_window_composite(synthetic_volume):
    ds = create_cappi(
        synthetic_volume,
        height=150.0,
        fields=["DBZH"],
        method="height_window_composite",
        apply_filter=True,
    )

    assert ds["DBZH"].dims == ("time", "range")
    assert ds.attrs["method"] == "height_window_composite"
    assert bool(ds.attrs["apply_quality_control"]) is True
    assert float(ds["altitude"].item()) == 150.0
    ds_geo = ds.xradar.georeference()
    assert "x" in ds_geo.coords
    assert "y" in ds_geo.coords


def test_create_cappi_height_window_composite_sparse_velocity(synthetic_volume):
    sparse_volume = synthetic_volume.copy(deep=True)
    for sweep in ["sweep_0", "sweep_1", "sweep_2"]:
        ds = sparse_volume[sweep].to_dataset()
        ds["VRADH"] = xr.DataArray(
            np.full_like(ds["VRADH"].values, np.nan),
            dims=ds["VRADH"].dims,
            coords=ds["VRADH"].coords,
            attrs=ds["VRADH"].attrs,
        )
        sparse_volume[sweep].ds = ds

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", RuntimeWarning)
        ds = create_cappi(
            sparse_volume,
            height=150.0,
            fields=["DBZH"],
            method="height_window_composite",
            apply_filter=True,
        )

    assert ds["DBZH"].dims == ("time", "range")
    assert not caught


def test_create_cappi_invalid_method_raises(synthetic_volume):
    with pytest.raises(ValueError, match="Unsupported CAPPI method"):
        create_cappi(synthetic_volume, height=150.0, method="bad_method")


def test_create_cappi_missing_georeference_raises(synthetic_volume):
    ungeoreferenced = DataTree.from_dict(
        {"sweep_0": synthetic_volume["sweep_0"].to_dataset().drop_vars(["x", "y", "z"])}
    )

    with pytest.raises(ValueError, match="Run radar.xradar.georeference"):
        create_cappi(ungeoreferenced, height=150.0, fields=["DBZH"])


def test_create_cappi_no_sweeps_raises():
    empty_tree = DataTree()

    with pytest.raises(ValueError, match="No sweep groups found"):
        create_cappi(
            empty_tree,
            height=150.0,
            method="polar_vertical_interpolation",
            fields=["DBZH"],
        )


def test_create_cappi_polar_out_of_range_raises(synthetic_volume):
    with pytest.raises(ValueError, match="outside available gate heights"):
        create_cappi(
            synthetic_volume,
            height=10000.0,
            fields=["DBZH"],
            method="polar_vertical_interpolation",
        )


def test_create_cappi_polar_georeference_and_cf2(synthetic_volume):
    ds = create_cappi(
        synthetic_volume,
        height=150.0,
        fields=["DBZH"],
        method="polar_vertical_interpolation",
    )

    ds_geo = ds.xradar.georeference()
    assert "x" in ds_geo.coords
    assert "y" in ds_geo.coords

    to_cf2 = getattr(ds.xradar, "to_cfradial2", None)
    if to_cf2 is not None:
        dtree = to_cf2()
    else:
        if xradar_to_cfradial2 is None:
            pytest.skip("Current xradar version does not expose CfRadial2 conversion.")
        dtree = xradar_to_cfradial2(ds)
    assert "/sweep_0" in dtree.groups


def test_plot_ppi_function(synthetic_volume, tmp_path):
    ds = synthetic_volume["sweep_0"].to_dataset()
    ax = plot_ppi(
        ds, "DBZH", title="Test PPI", savedir=str(tmp_path), show_figure=False
    )

    assert hasattr(ax, "figure")
    assert (tmp_path / "Test PPI_MockRadar_20240101000000.png").exists()


def test_plot_ppi_accessor_returns_axis(synthetic_volume, tmp_path):
    ds = synthetic_volume["sweep_0"].to_dataset()
    ax = ds.radarx.plot_ppi("DBZH", savedir=str(tmp_path), show_figure=False)

    assert hasattr(ax, "figure")
    assert ax.get_title() == "PPI DBZH"


def test_plot_ppi_dataarray_unknown_time_and_bytes_name(tmp_path):
    ds = _make_ppi_sweep(1.0, "2024-01-01T00:00:00").drop_vars("time")
    ds.attrs["instrument_name"] = b"ByteRadar"
    ax = plot_ppi(
        ds["DBZH"], None, title="Byte PPI", savedir=str(tmp_path), show_figure=False
    )

    assert hasattr(ax, "figure")
    created_files = list(tmp_path.glob("Byte PPI_*_unknown_time.png"))
    assert len(created_files) == 1


def test_plot_ppi_missing_data_var_raises(synthetic_volume):
    with pytest.raises(ValueError, match="data_var must be provided"):
        plot_ppi(synthetic_volume["sweep_0"].to_dataset(), None, show_figure=False)


def test_plot_rhi_accessor(tmp_path):
    ds = _make_rhi_dataset()
    ax = ds.radarx.plot_rhi(
        "DBZH",
        title="Test RHI",
        savedir=str(tmp_path),
        show_figure=False,
    )

    assert hasattr(ax, "figure")
    assert (tmp_path / "Test RHI_MockRadar_20240101000000.png").exists()


def test_plot_rhi_with_existing_axis(tmp_path):
    ds = _make_rhi_dataset()
    fig, ax = plt.subplots()
    returned_ax = ds.radarx.plot_rhi("DBZH", ax=ax, colorbar=False, show_figure=False)

    assert returned_ax is ax
    plt.close(fig)


def test_plot_cappi_function(synthetic_volume, tmp_path):
    ds = synthetic_volume.radarx.to_cappi(
        height=150.0,
        fields=["DBZH"],
        x_res=1000.0,
        y_res=1000.0,
    )
    ax = plot_cappi(
        ds, "DBZH", title="Test CAPPI", savedir=str(tmp_path), show_figure=False
    )

    assert hasattr(ax, "figure")
    assert (tmp_path / "Test CAPPI_MockRadar_20240101000000.png").exists()


def test_plot_cappi_accessor_returns_axis(synthetic_volume, tmp_path):
    ds = synthetic_volume.radarx.to_cappi(height=150.0, fields=["DBZH"])
    ax = ds.radarx.plot_cappi("DBZH", savedir=str(tmp_path), show_figure=False)

    assert hasattr(ax, "figure")
    assert "CAPPI" in ax.get_title()


def test_plot_cappi_without_z_uses_generic_title(synthetic_volume, tmp_path):
    ds = create_cappi(
        synthetic_volume,
        height=150.0,
        fields=["DBZH"],
        x_res=1000.0,
        y_res=1000.0,
    ).drop_vars("z")
    ax = plot_cappi(ds, "DBZH", savedir=str(tmp_path), show_figure=False)

    assert ax.get_title() == "CAPPI DBZH"


def test_plot_max_cappi_accessor_returns_axis(tmp_path):
    ds = _make_gridded_dataset()
    ax = ds.radarx.plot_max_cappi(
        "DBZH",
        title="Test Max-CAPPI",
        add_map=False,
        show_figure=False,
        savedir=str(tmp_path),
    )

    assert hasattr(ax, "figure")
    assert (tmp_path / "Test Max-CAPPI_MockRadar_20240101000000.png").exists()
