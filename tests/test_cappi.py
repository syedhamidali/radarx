#!/usr/bin/env python

"""Tests for CAPPI retrieval and basic plotting helpers."""

import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

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


# ---------------------------------------------------------------------------
# Coverage-targeted tests for radarx/retrieve/cappi.py helper functions
# ---------------------------------------------------------------------------


def test_create_cappi_auto_detect_fields(synthetic_volume):
    """fields=None exercises _default_cappi_fields auto-detection."""
    ds = create_cappi(synthetic_volume, height=150.0)
    assert "DBZH" in ds.data_vars or "VRADH" in ds.data_vars
    assert ds.attrs["product"] == "CAPPI"


def test_create_cappi_explicit_sweeps(synthetic_volume):
    """Explicit sweeps list exercises the _iter_sweeps sweep-list branch."""
    ds = create_cappi(
        synthetic_volume,
        height=150.0,
        fields=["DBZH"],
        sweeps=["sweep_0", "sweep_1"],
    )
    assert "DBZH" in ds.data_vars


def test_create_cappi_with_vertical_tolerance(synthetic_volume):
    """Explicit vertical_tolerance exercises _resolve_vertical_tolerance non-None path."""
    ds = create_cappi(
        synthetic_volume,
        height=150.0,
        fields=["DBZH"],
        vertical_tolerance=500.0,
    )
    assert ds.attrs["product"] == "CAPPI"


def test_create_cappi_polar_nearest(synthetic_volume):
    """nearest vertical_interpolation exercises the nearest-selection branch."""
    from radarx.retrieve.cappi import _create_cappi_polar_vertical_interpolation

    ds = _create_cappi_polar_vertical_interpolation(
        synthetic_volume,
        height=150.0,
        fields=["DBZH"],
        vertical_interpolation="nearest",
    )
    assert "DBZH" in ds.data_vars
    assert ds.attrs["method"] == "polar_vertical_interpolation"


def test_create_cappi_polar_invalid_interpolation(synthetic_volume):
    """Invalid vertical_interpolation raises a clear ValueError."""
    from radarx.retrieve.cappi import _create_cappi_polar_vertical_interpolation

    with pytest.raises(ValueError, match="vertical_interpolation must be"):
        _create_cappi_polar_vertical_interpolation(
            synthetic_volume,
            height=150.0,
            fields=["DBZH"],
            vertical_interpolation="cubic",
        )


def test_create_cappi_height_is_none_raises(synthetic_volume):
    """height=None raises ValueError for cartesian_idw."""
    with pytest.raises(ValueError, match="height must be provided"):
        create_cappi(synthetic_volume, height=None)


def test_create_cappi_no_sweeps_height_window():
    """Empty DataTree raises ValueError inside _create_cappi_height_window_composite."""
    from radarx.retrieve.cappi import _create_cappi_height_window_composite

    with pytest.raises(ValueError, match="No sweep groups found"):
        _create_cappi_height_window_composite(DataTree(), height=150.0, fields=["DBZH"])


def test_create_cappi_polar_auto_detect_fields(synthetic_volume):
    """fields=None with polar method exercises _default_cappi_fields in that path."""
    ds = create_cappi(
        synthetic_volume,
        height=150.0,
        method="polar_vertical_interpolation",
    )
    assert "DBZH" in ds.data_vars or "VRADH" in ds.data_vars


def test_create_cappi_height_window_auto_detect_fields(synthetic_volume):
    """fields=None with height_window_composite exercises _default_cappi_fields in that path."""
    ds = create_cappi(
        synthetic_volume,
        height=150.0,
        method="height_window_composite",
    )
    assert "DBZH" in ds.data_vars or "VRADH" in ds.data_vars


def test_decode_if_bytes():
    """_decode_if_bytes handles bytes, np.bytes_, and plain values."""
    from radarx.retrieve.cappi import _decode_if_bytes

    assert _decode_if_bytes(b"radar") == "radar"
    assert _decode_if_bytes(np.bytes_(b"site")) == "site"
    assert _decode_if_bytes("already_str") == "already_str"
    assert _decode_if_bytes(42) == 42


def test_infer_field_name_no_match():
    """_infer_field_name returns None when none of the candidates are present."""
    from radarx.retrieve.cappi import _infer_field_name

    ds = xr.Dataset({"DBZH": (("azimuth", "range"), np.zeros((4, 3)))})
    assert _infer_field_name(ds, ["VEL", "VR", "VRAD"]) is None


def test_build_target_grid_raises_on_partial_none():
    """_build_target_grid raises when only one of x/y is None."""
    from radarx.retrieve.cappi import _build_target_grid

    with pytest.raises(ValueError, match="Both x and y must be provided"):
        _build_target_grid(None, np.array([1.0, 2.0]))


def test_default_cappi_fields_excludes_metadata():
    """_default_cappi_fields skips reserved names and non-2D variables."""
    from radarx.retrieve.cappi import _default_cappi_fields

    ds = xr.Dataset(
        {
            "DBZH": (("azimuth", "range"), np.zeros((4, 3))),
            "x": (("azimuth", "range"), np.zeros((4, 3))),
            "time": (("azimuth",), np.zeros(4)),
        }
    )
    fields = _default_cappi_fields(ds)
    assert "DBZH" in fields
    assert "x" not in fields
    assert "time" not in fields


def test_cappi_polar_mismatched_azimuth_grids():
    """Sweeps with different azimuth counts force _interp_to_reference interpolation loop."""
    from radarx.retrieve.cappi import _create_cappi_polar_vertical_interpolation

    sweep_0 = _make_ppi_sweep(1.0, "2024-01-01T00:00:00")

    # Second sweep with 8 azimuths instead of 4
    azimuth2 = np.linspace(0.0, 315.0, 8)
    ranges2 = np.array([1000.0, 2000.0, 3000.0], dtype=float)
    az2_2d, r2_2d = np.meshgrid(np.deg2rad(azimuth2), ranges2, indexing="ij")
    el2 = np.deg2rad(3.0)
    sweep_1 = xr.Dataset(
        data_vars={
            "DBZH": (
                ("azimuth", "range"),
                30.0 + r2_2d / 1000.0,
                {"units": "dBZ"},
            ),
            "VRADH": (
                ("azimuth", "range"),
                np.full_like(r2_2d, 3.0),
                {"units": "m s-1"},
            ),
        },
        coords={
            "azimuth": ("azimuth", azimuth2),
            "range": ("range", ranges2),
            "x": (("azimuth", "range"), r2_2d * np.sin(az2_2d)),
            "y": (("azimuth", "range"), r2_2d * np.cos(az2_2d)),
            "z": (("azimuth", "range"), 100.0 + r2_2d * np.sin(el2)),
            "time": np.datetime64("2024-01-01T00:00:00"),
            "latitude": 28.61,
            "longitude": 77.23,
            "altitude": 100.0,
        },
        attrs={"instrument_name": "MockRadar", "sweep_mode": "azimuth_surveillance"},
    )

    tree = DataTree.from_dict({"sweep_0": sweep_0, "sweep_1": sweep_1})
    ds = _create_cappi_polar_vertical_interpolation(tree, height=150.0, fields=["DBZH"])
    assert "DBZH" in ds.data_vars


def test_get_reference_time_axis_no_time():
    """_get_reference_time_axis returns integer indices when 'time' is absent."""
    from radarx.retrieve.cappi import _get_reference_time_axis

    ds = xr.Dataset({"x": ("a", [1, 2, 3])})
    result = _get_reference_time_axis(ds, 5)
    np.testing.assert_array_equal(result, np.arange(5, dtype=int))


def test_interp_vertical_column_no_valid_points():
    """_interp_vertical_column returns (nan, True) when all values are NaN."""
    from radarx.retrieve.cappi import _interp_vertical_column

    val, masked = _interp_vertical_column([np.nan, np.nan], [np.nan, np.nan], 500.0)
    assert np.isnan(val)
    assert masked is True


def test_interp_vertical_column_single_valid_point():
    """_interp_vertical_column returns (nan, True) when only one valid point."""
    from radarx.retrieve.cappi import _interp_vertical_column

    val, masked = _interp_vertical_column([100.0, np.nan], [5.0, np.nan], 500.0)
    assert np.isnan(val)
    assert masked is True


def test_cappi_reference_metadata_empty_tree():
    """_cappi_reference_metadata returns empty dicts when no sweeps are present."""
    from radarx.retrieve.cappi import _cappi_reference_metadata

    coords, attrs = _cappi_reference_metadata(DataTree())
    assert coords == {}
    assert attrs == {}


def test_create_cappi_polar_explicit_max_vertical_distance(synthetic_volume):
    """Explicit max_vertical_distance exercises the non-None tolerance branch."""
    from radarx.retrieve.cappi import _create_cappi_polar_vertical_interpolation

    ds = _create_cappi_polar_vertical_interpolation(
        synthetic_volume,
        height=150.0,
        fields=["DBZH"],
        max_vertical_distance=500.0,
    )
    assert "DBZH" in ds.data_vars


def test_create_cappi_polar_no_2d_fields_raises():
    """Polar method raises when no 2D azimuth/range fields are available."""
    from radarx.retrieve.cappi import _create_cappi_polar_vertical_interpolation

    ds_1d = xr.Dataset(
        {
            "azimuth": ("azimuth", np.array([0.0, 90.0])),
            "range": ("range", np.array([1000.0, 2000.0])),
            "META": ("azimuth", np.array([1.0, 2.0])),
        }
    )
    tree = DataTree.from_dict({"sweep_0": ds_1d})
    with pytest.raises(ValueError, match="No 2D azimuth/range fields"):
        _create_cappi_polar_vertical_interpolation(tree, height=150.0)


def test_create_cappi_height_window_no_2d_fields_raises():
    """height_window_composite raises when no 2D azimuth/range fields are available."""
    from radarx.retrieve.cappi import _create_cappi_height_window_composite

    ds_1d = xr.Dataset(
        {
            "azimuth": ("azimuth", np.array([0.0, 90.0])),
            "range": ("range", np.array([1000.0, 2000.0])),
            "META": ("azimuth", np.array([1.0, 2.0])),
        }
    )
    tree = DataTree.from_dict({"sweep_0": ds_1d})
    with pytest.raises(ValueError, match="No 2D azimuth/range fields"):
        _create_cappi_height_window_composite(tree, height=150.0)


def _tall_volume(field):
    """
    Eleven sweeps from 0.5 to 19.5 degrees, georeferenced with xradar.

    ``field`` is ``"z"`` (gate height in km) or ``"ground"`` (ground distance
    of the gate in km).
    """
    from xradar.georeference import antenna_to_cartesian

    azimuth = np.arange(0.0, 360.0, 10.0)
    ranges = np.arange(1000.0, 120_001.0, 1000.0)
    sweeps = {}
    for k, elevation in enumerate(np.linspace(0.5, 19.5, 11)):
        x, y, z = antenna_to_cartesian(
            ranges[None, :], azimuth[:, None], np.full((azimuth.size, 1), elevation)
        )
        data = z / 1000.0 if field == "z" else np.hypot(x, y) / 1000.0
        sweeps[f"sweep_{k}"] = xr.Dataset(
            {"DBZH": (("azimuth", "range"), data, {"units": "dBZ"})},
            coords={
                "azimuth": ("azimuth", azimuth),
                "range": ("range", ranges),
                "x": (("azimuth", "range"), x),
                "y": (("azimuth", "range"), y),
                "z": (("azimuth", "range"), z),
                "time": np.datetime64("2024-01-01T00:00:00"),
                "latitude": 28.61,
                "longitude": 77.23,
                "altitude": 0.0,
            },
            attrs={"sweep_mode": "azimuth_surveillance"},
        )
    return DataTree.from_dict(sweeps)


def test_polar_cappi_applies_the_vertical_tolerance():
    """No value where no gate is near the height, however wide the gap (#165)."""
    volume = _tall_volume("z")
    ds = create_cappi(
        volume, 18000.0, method="polar", fields=["DBZH"], vertical_tolerance=100.0
    )
    dbz = ds["DBZH"]
    assert dbz.dims == ("time", "range")
    ground = ds["range"].values
    # at 60 km ground range the 15 and 19.5 degree sweeps are 4.5 km apart and
    # neither is within 100 m of 18 km
    column = dbz.isel(time=0).values[np.argmin(np.abs(ground - 60e3))]
    assert np.isnan(column)
    # a pixel with a gate within the tolerance keeps the interpolated height
    valid = dbz.isel(time=0).to_masked_array().compressed()
    assert valid.size > 0
    np.testing.assert_allclose(valid, 18.0, atol=0.1)
    # the tolerance matters: wider keeps more pixels, infinity restores the
    # interpolation across any gap
    counts = [
        create_cappi(
            volume, 18000.0, method="polar", fields=["DBZH"], vertical_tolerance=tol
        )["DBZH"]
        .to_masked_array()
        .count()
        for tol in (100.0, 1000.0, np.inf)
    ]
    assert counts[0] < counts[1] < counts[2]
    # the default is a limit as well
    default = create_cappi(volume, 18000.0, method="polar", fields=["DBZH"])
    assert 0 < default["DBZH"].to_masked_array().count() < counts[2]


@pytest.mark.parametrize("method", ["polar_vertical_interpolation", "pseudo_cappi"])
def test_polar_and_window_cappi_are_labelled_with_ground_range(method):
    """The value at a pixel comes from the ground distance it is labelled with (#166)."""
    volume = _tall_volume("ground")
    ds = create_cappi(volume, 3000.0, method=method, fields=["DBZH"])
    ground = ds["range"].values / 1000.0
    value = ds["DBZH"].isel(time=0).to_masked_array()
    ok = ~np.ma.getmaskarray(value)
    assert ok.sum() > 20
    # a field equal to the gate's ground distance is reproduced at the label;
    # labelled with the slant range the error was 0.5 km at 10 km, 0.2 km at 20 km
    err = np.abs(value.filled(np.nan) - ground)[ok]
    assert err.max() < 0.05
    assert np.all(np.diff(ds["range"].values) > 0)


def _smooth_velocity_sweep(seed=0):
    rng = np.random.default_rng(seed)
    az = np.arange(360.0)
    r = 250.0 * np.arange(1, 801)
    A, R = np.meshgrid(np.radians(az), r, indexing="ij")
    vel = 15.0 * np.cos(A - 1.0) * (R / 200e3 + 0.3) + rng.normal(0, 0.5, A.shape)
    return xr.Dataset(
        {
            "VRADH": (("azimuth", "range"), vel),
            "DBZH": (("azimuth", "range"), np.full(A.shape, 30.0)),
        },
        coords={"azimuth": az, "range": r},
    )


def test_velocity_texture_filter_keeps_clean_velocity():
    """The default limit is in m/s: smooth velocity is not removed."""
    from radarx.retrieve.cappi import _apply_velocity_texture_gate_filter as gate_filter

    ds = _smooth_velocity_sweep()
    out = gate_filter(ds, "VRADH", "DBZH")
    assert np.isfinite(out["VRADH"].values).mean() > 0.99


def test_velocity_texture_filter_removes_garbage():
    from radarx.retrieve.cappi import _apply_velocity_texture_gate_filter as gate_filter

    ds = _smooth_velocity_sweep()
    rng = np.random.default_rng(1)
    bad = rng.random(ds["VRADH"].shape) < 0.5
    ds["VRADH"] = ds["VRADH"].copy(
        data=np.where(bad, rng.uniform(-25, 25, bad.shape), ds["VRADH"].values)
    )
    out = gate_filter(ds, "VRADH", "DBZH")
    kept = np.isfinite(out["VRADH"].values)
    assert kept[bad].mean() < 0.1
    # the limit can be set (m/s)
    loose = gate_filter(ds, "VRADH", "DBZH", texture_threshold=14.0)
    assert np.isfinite(loose["VRADH"].values)[bad].mean() > 0.5


def test_create_cappi_texture_threshold_is_passed_on(synthetic_volume):
    kw = dict(
        height=150.0,
        fields=["DBZH"],
        method="height_window_composite",
        apply_filter=True,
    )
    default = create_cappi(synthetic_volume, **kw)
    strict = create_cappi(synthetic_volume, texture_threshold=0.0, **kw)
    assert np.isfinite(default["DBZH"].values).any()
    assert not np.isfinite(strict["DBZH"].values).any()


@pytest.mark.parametrize("method", ["cartesian_idw", "polar_vertical_interpolation"])
def test_create_cappi_linear_units(synthetic_volume, method):
    """Averaging in linear units gives a higher dBZ than averaging the dB."""
    kw = dict(height=200.0, method=method, fields=["DBZH"])
    in_db = create_cappi(synthetic_volume, **kw)
    lin = create_cappi(synthetic_volume, linear_units=True, **kw)
    a, b = in_db["DBZH"].values, lin["DBZH"].values
    both = np.isfinite(a) & np.isfinite(b)
    assert both.any()
    assert np.all(b[both] >= a[both] - 1e-9)
    assert (b[both] - a[both]).max() > 0.1
    assert lin.attrs["interpolated_in_linear_units"] == "DBZH"
    assert in_db.attrs["interpolated_in_linear_units"] == ""


@pytest.mark.parametrize("method", ["cartesian_idw", "polar_vertical_interpolation"])
def test_create_cappi_warns_for_velocity(synthetic_volume, method):
    with pytest.warns(UserWarning, match="VRADH"):
        create_cappi(synthetic_volume, height=200.0, method=method)


def test_reference_ground_range():
    from radarx.retrieve.cappi import _reference_ground_range

    az = np.arange(0.0, 360.0, 90.0)
    r = np.array([1000.0, 2000.0, 3000.0])
    ang = np.deg2rad(az)[:, None]
    x, y = r[None, :] * np.sin(ang), r[None, :] * np.cos(ang)
    dims = ("azimuth", "range")
    ref = xr.Dataset(coords={"x": (dims, x), "y": (dims, y)})
    # the median over the rays of the distance from the radar
    np.testing.assert_allclose(_reference_ground_range(ref), r)
    # no x or y: the slant range is kept
    assert _reference_ground_range(ref.drop_vars("y")) is None
    assert _reference_ground_range(xr.Dataset()) is None
    # x and y that are not (ray, gate) fields of the same shape
    flat = xr.Dataset(coords={"x": ("range", r), "y": ("range", r)})
    assert _reference_ground_range(flat) is None
    mismatch = xr.Dataset(
        coords={"x": (dims, x), "y": (("azimuth", "other"), y[:, :2] + 0.0)}
    )
    assert _reference_ground_range(mismatch) is None
    # a distance that does not increase with range, or is not finite
    back = xr.Dataset(coords={"x": (dims, x[:, ::-1]), "y": (dims, y[:, ::-1])})
    assert _reference_ground_range(back) is None
    holes = x.copy()
    holes[:, 1] = np.nan
    gap = xr.Dataset(coords={"x": (dims, holes), "y": (dims, y)})
    with pytest.warns(RuntimeWarning, match="All-NaN"):
        assert _reference_ground_range(gap) is None


def test_onto_ground_range_needs_two_gates_per_ray():
    from radarx.retrieve.cappi import _onto_ground_range

    ground_ref = np.array([1000.0, 2000.0, 3000.0])
    values = np.array([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0], [9.0, 9.0, 9.0]])
    ground = np.array(
        [
            [1000.0, 2000.0, 3000.0],  # the same positions: unchanged
            [1500.0, 2500.0, 3500.0],  # shifted by half a gate: interpolated
            [np.nan, np.nan, 3000.0],  # a single gate with a distance: no row
        ]
    )
    out = _onto_ground_range(values, ground, ground_ref)
    np.testing.assert_allclose(out[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(out[1], [np.nan, 5.5, 6.5])
    assert np.isnan(out[2]).all()
