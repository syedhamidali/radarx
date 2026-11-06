"""Integer class codes must not become masked through stale fill encodings."""

import copy

import numpy as np
import pytest
import xarray as xr

from radarx.io import export


@pytest.mark.parametrize("dtype", ["int8", "uint16"])
@pytest.mark.parametrize("engine", ["netcdf4", "zarr"])
@pytest.mark.parametrize("source", ["attribute", "encoding", "both", "none"])
def test_class_fill_sources_round_trip_without_masking(dtype, engine, source, tmp_path):
    values = np.array([[0, 1, 2], [2, 1, 0]], dtype=dtype)
    ds = xr.Dataset(
        {"HID": (("y", "x"), values)}, coords={"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0]}
    )
    ds.HID.attrs.update(flag_values=np.array([0, 1, 2], dtype=dtype), units="1")
    if source in ("attribute", "both"):
        ds.HID.attrs["_FillValue"] = 0
    if source in ("encoding", "both"):
        ds.HID.encoding["_FillValue"] = 0
    original = ds.copy(deep=True)
    encoding = copy.deepcopy(ds.HID.encoding)
    path = tmp_path / ("classes.nc" if engine == "netcdf4" else "classes.zarr")
    if engine == "netcdf4":
        export.write_netcdf_cf(ds, path, crs=False)
    else:
        export.write_zarr(ds, path)
    with xr.open_dataset(path, engine=engine) as back:
        assert back.HID.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(back.HID, values)
        np.testing.assert_array_equal(back.HID.attrs["flag_values"], [0, 1, 2])
    xr.testing.assert_identical(ds, original)
    assert ds.HID.encoding == encoding


@pytest.mark.parametrize("target", ["netcdf", "zarr"])
@pytest.mark.parametrize("packing", ["drop", "keep"])
def test_integer_fill_removal_retains_float_and_packing_controls(target, packing):
    ds = xr.Dataset(
        {
            "classes": ("x", np.array([0, 1], dtype="int16")),
            "floaty": ("x", [np.nan, 1.5]),
        }
    )
    ds.classes.encoding.update(_FillValue=0, dtype="int16")
    ds.floaty.encoding.update(_FillValue=-9999.0, dtype="int16", scale_factor=0.5)
    out = export._prepare_dataset(ds, target, packing)
    assert "_FillValue" not in out.classes.encoding
    assert out.classes.encoding["dtype"] == "int16"
    if packing == "keep":
        assert out.floaty.encoding == ds.floaty.encoding
    else:
        assert not out.floaty.encoding
    assert ds.classes.encoding["_FillValue"] == 0


@pytest.mark.parametrize("target", ["netcdf", "zarr"])
@pytest.mark.parametrize("packing", ["drop", "keep"])
def test_unpacked_float_fill_is_not_removed(target, packing):
    ds = xr.Dataset({"values": ("x", [np.nan, 1.5])})
    ds["values"].encoding["_FillValue"] = -9999.0
    out = export._prepare_dataset(ds, target, packing)
    assert out["values"].encoding["_FillValue"] == -9999.0
    assert ds["values"].encoding["_FillValue"] == -9999.0
