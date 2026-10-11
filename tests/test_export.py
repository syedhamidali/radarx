#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Export of radarx results to CfRadial1/2, ODIM_H5, Zarr and CF NetCDF
====================================================================

Every format is written and read back with xarray and with xradar. Zarr reads
back identical; the other formats are compared in values and attributes and
the losses of each format are asserted.
"""

import copy
import json
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from packaging.version import Version

import radarx
from radarx import retrieve as R
from radarx._provenance import read_provenance
from radarx.io import export

from .grid.test_cone import GRID
from .grid.test_cone import _volume as cone_volume
from .test_cappi import _make_ppi_sweep

xd = pytest.importorskip("xradar")
h5py = pytest.importorskip("h5py")
pytest.importorskip("netCDF4")

from open_radar_data import DATASETS  # noqa: E402

XRADAR = Version(xd.__version__.split("+")[0].split(".dev")[0])
# xarray before 2025.1 cannot write Zarr format 3, which zarr 3 uses
try:
    import zarr

    NO_ZARR_3 = Version(xr.__version__) < Version("2025.1") and Version(
        zarr.__version__
    ) >= Version("3")
except ImportError:  # pragma: no cover
    zarr = None
    NO_ZARR_3 = False

needs_zarr = pytest.mark.skipif(
    zarr is None or NO_ZARR_3, reason="needs zarr and an xarray that writes it"
)

SOURCE = "NOD:uskbb"
TIME_SEP = " | "


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def make_volume(nsweep=2, nray=36, ngate=40, seed=0, ngates=None):
    """A small PPI volume as xradar readers return it, with ``None`` placeholders."""
    rng = np.random.default_rng(seed)
    sweeps = {}
    for k in range(nsweep):
        if ngates is not None:
            ngate = ngates[k]
        ds = xd.model.create_sweep_dataset(
            shape=(nray, ngate),
            rng=250.0,
            elevation=0.5 + 1.5 * k,
            date_str=f"2024-05-01T12:00:{10 * k:02d}",
            time=0.25,
        )
        ds = (
            ds.drop_vars(["latitude", "longitude", "altitude"])
            .assign_coords(range=ds.range + 2000.0)
            .swap_dims({"time": "azimuth"})
        )
        # rays are never on a regular grid
        ds = ds.assign_coords(azimuth=ds.azimuth + 0.4 * np.sin(np.arange(nray)))
        dbz = rng.uniform(-10, 60, (nray, ngate))
        dbz[:3, :5] = np.nan
        dims = ("azimuth", "range")
        ds["DBZH"] = (
            dims,
            dbz,
            {
                "units": "dBZ",
                "long_name": "Equivalent reflectivity factor H",
                "standard_name": "radar_equivalent_reflectivity_factor_h",
            },
        )
        ds["ZDR"] = (dims, rng.uniform(-1, 4, (nray, ngate)), {"units": "dB"})
        phi = np.cumsum(rng.uniform(0, 2, (nray, ngate)), axis=1)
        ds["PHIDP"] = (dims, phi, {"units": "degrees"})
        ds["RHOHV"] = (dims, rng.uniform(0.9, 1.0, (nray, ngate)), {"units": "1"})
        ds["sweep_mode"] = "azimuth_surveillance"
        ds["sweep_number"] = k
        ds["sweep_fixed_angle"] = 0.5 + 1.5 * k
        ds.attrs = {"sails_cut": False, "super_resolution": 11 + k}
        sweeps[f"sweep_{k}"] = ds
    root = xr.Dataset(
        {
            "volume_number": 0,
            "platform_type": "fixed",
            "instrument_type": "radar",
            "time_coverage_start": "2024-05-01T12:00:00Z",
            "time_coverage_end": "2024-05-01T12:00:20Z",
        },
        coords={"latitude": 33.65, "longitude": -101.81, "altitude": 1029.0},
        attrs={
            "instrument_name": "KTST",
            "Conventions": "None",
            "version": "None",
            "title": "None",
            "history": "None",
            "comment": "test volume",
            "avset_enabled": True,
            "number_elevation_cuts": 11,
        },
    )
    return xr.DataTree.from_dict({"/": root, **sweeps})


def make_product_volume():
    """The volume with KDP, the echo classes and a boolean mask merged in."""
    raw = make_volume()
    vol = raw.radarx.assign(R.estimate_kdp(raw))
    vol = vol.radarx.assign(R.echo_mask(raw))
    return vol


@pytest.fixture(scope="module")
def volume():
    return make_product_volume()


def open_volume(path, engine):
    """Read a radar file with xradar: the xarray entry point, or the older function."""
    try:
        return xr.open_datatree(path, engine=engine)
    except NotImplementedError:  # xradar 0.11 has no DataTree backends
        return getattr(xd.io, f"open_{engine}_datatree")(path)


def snapshot(obj):
    """Everything a writer must leave alone: values, attributes and encodings."""
    if isinstance(obj, xr.DataTree):
        return [
            (node.path, snapshot(node.to_dataset(inherit=False)), dict(node.attrs))
            for node in obj.subtree
        ]
    if isinstance(obj, xr.DataArray):
        obj = obj.to_dataset(name=obj.name or "data")
    return (
        copy.deepcopy(dict(obj.attrs)),
        {
            name: (copy.deepcopy(dict(v.attrs)), copy.deepcopy(dict(v.encoding)))
            for name, v in obj.variables.items()
        },
    )


def assert_unchanged(obj, before, reference):
    assert repr(snapshot(obj)) == repr(before)
    if isinstance(obj, xr.DataTree):
        xr.testing.assert_identical(obj, reference)
    else:
        xr.testing.assert_identical(obj, reference)


# ---------------------------------------------------------------------------
# attributes
# ---------------------------------------------------------------------------
class Stamp:
    def __str__(self):
        return "stamp"


def test_clean_attrs_for_netcdf():
    attrs = {
        "none": None,
        "flag": True,
        "numpy_flag": np.bool_(False),
        "text": "keep",
        "numpy_text": np.str_("kept"),
        "bytes": b"b",
        "int": 3,
        "np_int": np.int16(4),
        "float16": np.float16(1.5),
        "datetime": np.datetime64("2024-01-01T00:00"),
        "stamp": pd.Timestamp("2024-01-02"),
        "dict": {"b": 1, "a": np.float32(2.0), "c": np.arange(2), "d": Stamp()},
        "ints": [1, 2, 3],
        "words": ("a", "b"),
        "empty": [],
        "mixed": [1, "a", None, True],
        "bools": np.array([True, False]),
        "small": np.array([1.0, 2.0], dtype="float16"),
        "grid": np.arange(4).reshape(2, 2),
        "scalar_array": np.array(5),
        "dates": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]"),
        "object": Stamp(),
        7: "numeric key",
    }
    out = export._clean_attrs(attrs, "netcdf")
    assert "none" not in out
    assert out["flag"] == "true" and out["numpy_flag"] == "false"
    assert out["text"] == "keep" and out["numpy_text"] == "kept"
    assert type(out["numpy_text"]) is str
    assert out["bytes"] == b"b" and out["int"] == 3 and out["np_int"] == 4
    assert out["float16"] == np.float32(1.5) and out["float16"].dtype == np.float32
    assert out["datetime"].startswith("2024-01-01T00:00")
    assert out["stamp"] == "2024-01-02T00:00:00"
    assert json.loads(out["dict"]) == {"a": 2.0, "b": 1, "c": [0, 1], "d": "stamp"}
    np.testing.assert_array_equal(out["ints"], [1, 2, 3])
    assert out["words"] == ["a", "b"] and out["empty"] == ""
    assert json.loads(out["mixed"]) == [1, "a", "true"]
    np.testing.assert_array_equal(out["bools"], [1, 0])
    assert out["bools"].dtype == np.int8 and out["small"].dtype == np.float32
    np.testing.assert_array_equal(out["grid"], [0, 1, 2, 3])
    assert out["scalar_array"] == [5] or list(out["scalar_array"]) == [5]
    assert out["dates"] == ["2024-01-01", "2024-01-02"]
    assert out["object"] == "stamp" and out["7"] == "numeric key"
    for value in out.values():
        assert not isinstance(value, (bool, np.bool_))


def test_clean_attrs_for_zarr_keeps_json_types():
    attrs = {
        "none": None,
        "flag": True,
        "bools": np.array([True, False]),
        "mixed": [1, "a", None, True],
        "ints": (1, 2),
        "nested": [[1, 2], [3]],
    }
    out = export._clean_attrs(attrs, "zarr")
    assert out["none"] is None and out["flag"] is True
    assert out["bools"].dtype == bool
    assert out["mixed"] == [1, "a", None, True]
    np.testing.assert_array_equal(out["ints"], [1, 2])
    assert out["nested"] == [[1, 2], [3]]


# ---------------------------------------------------------------------------
# variables
# ---------------------------------------------------------------------------
def variable_cases():
    fill = np.full(6, np.nan)
    fill[2] = 1.5
    dims = ("y", "x")
    return xr.Dataset(
        {
            "floaty": (dims, fill[None], {"_FillValue": -9999.0, "units": "m"}),
            "classes": (
                dims,
                np.array([[0, 1, 2, 0, 1, 2]], dtype="int8"),
                {"_FillValue": 0, "flag_values": np.array([0, 1, 2], dtype="int8")},
            ),
            "packed": (
                dims,
                np.array([[0.5, 1.0, np.nan, 2.0, 2.5, 3.0]]),
                {},
            ),
            "when": (
                dims,
                np.array([["2024-01-01", "2024-01-02", "NaT"] * 2], dtype="M8[ns]"),
                {},
            ),
            "chunked": (dims, np.arange(6.0)[None], {}),
        },
        coords={"x": np.arange(6.0), "y": [0.0]},
        attrs={"title": "t", "note": None},
    )


def with_encodings(ds):
    ds = ds.copy(deep=True)
    ds["packed"].encoding.update(
        dtype=np.dtype("uint8"), scale_factor=0.5, add_offset=0.0, source="/old.nc"
    )
    ds["when"].encoding.update(
        units="days since 1900-01-01", calendar="gregorian", dtype=np.dtype("int16")
    )
    ds["chunked"].encoding.update(chunks=(2,), preferred_chunks={"x": 2}, group=0)
    return ds


def test_fill_value_attribute_is_moved_or_dropped():
    prepared = export._prepare_dataset(variable_cases(), "netcdf", "drop")
    assert "_FillValue" not in prepared["floaty"].attrs
    assert prepared["floaty"].encoding["_FillValue"] == -9999.0
    assert "_FillValue" not in prepared["classes"].attrs
    assert "_FillValue" not in prepared["classes"].encoding
    assert "note" not in prepared.attrs and prepared.attrs["title"] == "t"
    assert prepared["classes"].attrs["flag_values"].dtype == np.int8


@pytest.mark.parametrize("engine", ["netcdf4", "zarr"])
def test_class_codes_without_fill_value_round_trip_as_int8(engine, tmp_path):
    if engine == "zarr" and (zarr is None or NO_ZARR_3):
        pytest.skip("needs zarr and an xarray that writes it")
    ds = variable_cases()
    path = tmp_path / ("a.zarr" if engine == "zarr" else "a.nc")
    if engine == "zarr":
        radarx.io.write_zarr(ds, path)
    else:
        radarx.io.write_netcdf_cf(ds, path, crs=False)
    with xr.open_dataset(path, engine=engine) as back:
        assert back["classes"].dtype == np.int8
        np.testing.assert_array_equal(back["classes"], ds["classes"])
        np.testing.assert_array_equal(
            back["floaty"].isnull().values, ds["floaty"].isnull().values
        )
        np.testing.assert_array_equal(
            back["floaty"].values[~np.isnan(back["floaty"])], [1.5]
        )


def test_stale_encodings_are_dropped_and_packing_follows_the_option():
    ds = with_encodings(variable_cases())
    dropped = export._prepare_dataset(ds, "netcdf", "drop")
    assert dropped["packed"].encoding == {}
    assert dropped["when"].encoding == {}
    assert dropped["chunked"].encoding == {}
    kept = export._prepare_dataset(ds, "netcdf", "keep")
    assert kept["packed"].encoding["dtype"] == np.uint8
    assert "source" not in kept["packed"].encoding
    assert kept["when"].encoding == {}
    # the input still has its encodings
    assert ds["packed"].encoding["dtype"] == np.uint8
    assert ds["chunked"].encoding["chunks"] == (2,)


def test_packing_option_is_checked(tmp_path):
    for call in (
        lambda: radarx.io.write_zarr(variable_cases(), tmp_path / "a", packing="x"),
        lambda: radarx.io.write_cfradial1(make_volume(), tmp_path / "a", packing="x"),
        lambda: radarx.io.write_cfradial2(make_volume(), tmp_path / "a", packing="x"),
        lambda: radarx.io.write_odim(make_volume(), tmp_path / "a", packing="x"),
    ):
        with pytest.raises(ValueError, match="packing"):
            call()


def test_packing_keep_writes_integers_and_drop_writes_floats(tmp_path):
    ds = with_encodings(variable_cases())
    kept, dropped = tmp_path / "kept.nc", tmp_path / "dropped.nc"
    export.write_netcdf_cf(ds, kept, crs=False)
    # write_netcdf_cf always writes floats (packing is a volume and Zarr option)
    with xr.open_dataset(kept, mask_and_scale=False) as raw:
        assert raw["packed"].dtype == np.float64
    ds2 = ds[["packed"]].copy()
    prepared = export._prepare_dataset(ds2, "netcdf", "keep")
    prepared.to_netcdf(dropped)
    with xr.open_dataset(dropped, mask_and_scale=False) as raw:
        assert raw["packed"].dtype == np.uint8


# ---------------------------------------------------------------------------
# Zarr
# ---------------------------------------------------------------------------
@needs_zarr
def test_zarr_volume_round_trip_is_identical(volume, tmp_path):
    before = snapshot(volume)
    # the booleans of the sweeps survive Zarr
    out = radarx.io.write_zarr(volume, tmp_path / "v.zarr")
    assert out == tmp_path / "v.zarr"
    with open_volume(out, "zarr") as back:
        xr.testing.assert_identical(back, volume)
        assert back["sweep_0"].attrs["sails_cut"] is False
        assert back.attrs["avset_enabled"] is True
    assert repr(snapshot(volume)) == repr(before)


@needs_zarr
def test_zarr_keeps_dask_arrays_lazy(volume, tmp_path):
    dask = pytest.importorskip("dask")
    chunked = volume.chunk({"azimuth": 12})
    radarx.io.write_zarr(chunked, tmp_path / "c.zarr")
    assert isinstance(chunked["sweep_0"]["DBZH"].data, dask.array.Array)
    with open_volume(tmp_path / "c.zarr", "zarr") as back:
        xr.testing.assert_identical(back.compute(), volume)


@needs_zarr
def test_zarr_dataset_dataarray_and_accessors(tmp_path):
    ds = variable_cases()
    for var in ds.data_vars.values():
        var.attrs.pop("_FillValue", None)
    ds.attrs["flag"] = True
    ds.attrs["nothing"] = None
    out = ds.radarx.to_zarr(tmp_path / "ds.zarr")
    with xr.open_dataset(out, engine="zarr") as back:
        xr.testing.assert_identical(back, ds)
    da = ds["floaty"]
    out = da.radarx.to_zarr(tmp_path / "da.zarr")
    with xr.open_dataset(out, engine="zarr") as back:
        xr.testing.assert_allclose(back["floaty"], da)
    nameless = xr.DataArray([1.0, 2.0], dims="x")
    out = export.write_zarr(nameless, tmp_path / "n.zarr")
    with xr.open_dataset(out, engine="zarr") as back:
        np.testing.assert_array_equal(back["data"], [1.0, 2.0])
    out = make_volume().radarx.to_zarr(tmp_path / "t.zarr")
    assert out.exists()


@needs_zarr
def test_zarr_overwrites_and_passes_options(tmp_path):
    ds = variable_cases()
    export.write_zarr(ds, tmp_path / "a.zarr")
    export.write_zarr(ds[["chunked"]], tmp_path / "a.zarr", consolidated=False)
    with xr.open_dataset(tmp_path / "a.zarr", engine="zarr") as back:
        assert list(back.data_vars) == ["chunked"]


# ---------------------------------------------------------------------------
# CfRadial1
# ---------------------------------------------------------------------------
def provenance_of(attrs):
    return {k: attrs[k] for k in export.PROVENANCE if k in attrs}


def test_cfradial1_round_trip(volume, tmp_path):
    before = snapshot(volume)
    out = radarx.io.write_cfradial1(volume, tmp_path / "v.nc")
    assert out == tmp_path / "v.nc"
    assert repr(snapshot(volume)) == repr(before)

    with xr.open_dataset(out) as flat:
        # the sweeps are stacked along time, the fields are NetCDF variables
        for name in ("DBZH", "ZDR", "PHIDP", "RHOHV", "KDP", "PHIDP_processed"):
            assert flat[name].dims == ("time", "range")
        assert flat.sizes["sweep"] == 2
        # booleans of the global attributes are "true" and "false", None is gone
        assert flat.attrs["avset_enabled"] == "true"
        assert "title" not in flat.attrs and "version" in flat.attrs
        assert flat.attrs["Conventions"] == "Cf/Radial"
        # the provenance of the sweeps is on the global attributes ...
        root = provenance_of(flat.attrs)
        assert root["radarx_method"].count(TIME_SEP) == 1
        assert root["radarx_version"] == radarx.__version__
        assert "estimate_kdp()" in root["history"]
        assert "echo_mask()" in root["history"]
        assert "CfRadial1 export" in root["history"]
        # ... and on the fields
        assert (
            flat["KDP"].attrs["radarx_method"]
            == volume["sweep_0"]["KDP"].attrs["radarx_method"]
        )
        assert (
            flat["KDP"].attrs["standard_name"] == "radar_specific_differential_phase_hv"
        )
        # a field of the sweeps that has no range dimension is kept as well
        assert flat["PHIDP_OFFSET"].dims == ("time",)
        # sweeps cannot carry attributes
        assert "super_resolution" not in flat.attrs
        assert flat["METEO_MASK"].dtype == bool

    back = open_volume(out, "cfradial1")
    for name in ("sweep_0", "sweep_1"):
        for field in ("DBZH", "ZDR", "KDP", "PHIDP_processed", "ECHO_CLASS"):
            want = volume[name][field].sortby("azimuth")
            got = back[name][field].sortby("azimuth")
            np.testing.assert_allclose(got.values, want.values, equal_nan=True)
        assert back[name]["KDP"].attrs["radarx_references"] == (
            volume[name]["KDP"].attrs["radarx_references"]
        )
    assert back["sweep_0"]["ECHO_CLASS"].dtype == np.int8


def test_cfradial1_pads_sweeps_of_different_length(tmp_path):
    vol = make_volume(ngates=(40, 30))
    out = radarx.io.write_cfradial1(vol, tmp_path / "pad.nc")
    with xr.open_dataset(out) as flat:
        assert flat.sizes["range"] == 40
        second = flat["DBZH"].isel(time=slice(36, None))
        assert second.isel(range=slice(30, None)).isnull().all()
        assert second.isel(range=slice(None, 30)).notnull().any()


def test_cfradial1_with_different_fields_per_sweep(volume, tmp_path):
    # the Doppler cuts of a NEXRAD volume have no polarimetric products
    mixed = volume.copy()
    for name in ("KDP", "ECHO_CLASS", "PHIDP_OFFSET", "ZDR"):
        del mixed["sweep_1"][name]
    before = snapshot(mixed)
    out = radarx.io.write_cfradial1(mixed, tmp_path / "mixed.nc")
    assert repr(snapshot(mixed)) == repr(before)
    assert "KDP" not in mixed["sweep_1"].variables
    with xr.open_dataset(out) as flat:
        second = flat.isel(time=slice(36, None))
        for name in ("KDP", "ECHO_CLASS", "ZDR"):
            assert second[name].isnull().all()
            assert flat[name].isel(time=slice(0, 36)).notnull().any()
        assert flat["ECHO_CLASS"].dtype.kind == "f"
        assert (
            flat["KDP"].attrs["radarx_method"]
            == volume["sweep_0"]["KDP"].attrs["radarx_method"]
        )
        assert flat["DBZH"].isel(time=slice(36, None)).notnull().any()


def test_harmonize_fields_leaves_a_sweep_without_fields_alone():
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "sweep_0": xr.Dataset({"f": (("azimuth", "range"), np.ones((2, 2)))}),
            "sweep_1": xr.Dataset({"s": 1}),
        }
    )
    export._harmonize_fields(tree)
    assert "f" not in tree["sweep_1"].variables


def test_cfradial1_provenance_is_not_lost_for_cite(volume, tmp_path):
    out = radarx.io.write_cfradial1(volume, tmp_path / "v.nc")
    with xr.open_dataset(out) as flat:
        found = read_provenance(flat)
    methods = {m for item in found for m in item["method"]}
    assert any("Differential phase" in m for m in methods)
    assert any("echo" in m.lower() for m in methods)


def test_cfradial1_without_calibration_and_options(volume, tmp_path):
    out = volume.radarx.to_cfradial1(tmp_path / "v.nc", calibs=False)
    assert out.exists()
    assert not volume.attrs.get("radarx_method")


def test_cfradial1_provenance_fold_without_provenance(tmp_path):
    vol = make_volume()
    out = radarx.io.write_cfradial1(vol, tmp_path / "plain.nc")
    with xr.open_dataset(out) as flat:
        assert "radarx_method" not in flat.attrs
        assert (
            flat.attrs["history"].startswith("xradar")
            or "CfRadial1" in flat.attrs["history"]
        )


def test_fold_provenance_merges_root_and_sweeps():
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(
                attrs={
                    "radarx_method": "A",
                    "radarx_references": "r1",
                    "radarx_version": "1",
                    "history": "None",
                }
            ),
            "sweep_0": xr.Dataset(
                attrs={
                    "radarx_method": "A | B",
                    "radarx_references": "r1 r2",
                    "radarx_version": "2",
                    "history": "a()\nb()",
                }
            ),
            "sweep_1": xr.Dataset(
                attrs={
                    "radarx_method": "B",
                    "radarx_references": "r2",
                    "history": "None\nb()",
                }
            ),
        }
    )
    export._fold_provenance(tree)
    assert tree.attrs["radarx_method"] == "A | B"
    assert tree.attrs["radarx_references"] == "r1 r2"
    assert tree.attrs["radarx_version"] == "2"
    assert tree.attrs["history"] == "a()\nb()"


# ---------------------------------------------------------------------------
# CfRadial2
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("engine", [None, "netcdf4", "h5netcdf"])
def test_cfradial2_round_trip(volume, tmp_path, engine):
    if engine == "h5netcdf":
        pytest.importorskip("h5netcdf")
    before = snapshot(volume)
    out = radarx.io.write_cfradial2(volume, tmp_path / "v.nc", engine=engine)
    assert repr(snapshot(volume)) == repr(before)

    with xr.open_datatree(out) as back:
        assert back.attrs["Conventions"] == "Cf/Radial"
        assert back.attrs["version"] == "2.0"
        assert back.attrs["avset_enabled"] == "true"
        root = provenance_of(back.attrs)
        assert root["radarx_version"] == radarx.__version__
        assert "estimate_kdp()" in root["history"]
        for name in ("sweep_0", "sweep_1"):
            node = back[name]
            # xradar's CfRadial2 writer swaps the ray dimension for time
            assert node["KDP"].dims == ("time", "range")
            want = volume[name]["KDP"].sortby("azimuth")
            got = node["KDP"].swap_dims(time="azimuth").sortby("azimuth")
            np.testing.assert_allclose(got.values, want.values, equal_nan=True)
            assert provenance_of(node["KDP"].attrs) == provenance_of(
                volume[name]["KDP"].attrs
            )
            assert node["ECHO_CLASS"].dtype == np.int8
            # lost: variables without a range dimension, and the sweep attributes
            assert "PHIDP_OFFSET" not in node.variables
            assert "super_resolution" not in node.attrs


@pytest.mark.skipif(
    "cfradial2" not in xr.backends.list_engines(),
    reason="xradar 0.11 has no CfRadial2 reader",
)
def test_cfradial2_reads_back_with_xradar(volume, tmp_path):
    out = radarx.io.write_cfradial2(volume, tmp_path / "v.nc")
    back = open_volume(out, "cfradial2")
    assert sorted(back.match("sweep_*")) == ["sweep_0", "sweep_1"]
    np.testing.assert_allclose(
        back["sweep_0"]["DBZH"].sortby("azimuth").values,
        volume["sweep_0"]["DBZH"].sortby("azimuth").values,
        equal_nan=True,
    )
    assert "radarx_method" in back["sweep_0"]["KDP"].attrs


def test_cfradial2_accessor(volume, tmp_path):
    assert volume.radarx.to_cfradial2(tmp_path / "a.nc").exists()


# ---------------------------------------------------------------------------
# ODIM_H5
# ---------------------------------------------------------------------------
def read_how(path, group):
    with h5py.File(path) as f:
        return {
            k: (v.decode() if isinstance(v, bytes) else v)
            for k, v in f[group].attrs.items()
        }


def test_odim_round_trip(volume, tmp_path):
    before = snapshot(volume)
    out = radarx.io.write_odim(volume, tmp_path / "v.h5", source=SOURCE)
    assert repr(snapshot(volume)) == repr(before)

    with h5py.File(out) as f:
        assert f["what"].attrs["object"] == b"PVOL"
        assert f["what"].attrs["source"] == SOURCE.encode()
        names = {
            f["dataset1"][key]["what"].attrs["quantity"].decode(): key
            for key in f["dataset1"]
            if key.startswith("data") and key != "data"
        }
        assert {"DBZH", "ZDR", "PHIDP", "RHOHV", "KDP", "ECHO_CLASS"} <= set(names)
        mask = f["dataset1"][names["METEO_MASK"]]["data"]
        assert mask.dtype == np.uint8
        # the other attributes of the variables are not written
        assert "units" not in f["dataset1"][names["KDP"]]["what"].attrs

    # provenance: root, sweep and field
    root = read_how(out, "how")
    sweep = read_how(out, "dataset1/how")
    assert sweep["radarx_method"].count(TIME_SEP) == 1
    assert sweep["history"] == volume["sweep_0"].attrs["history"]
    assert "radarx_method" not in root and root.get("history") != "None"
    with h5py.File(out) as f:
        kdp = f["dataset1"][names["KDP"]]["how"].attrs
        assert kdp["radarx_method"].decode() == (
            volume["sweep_0"]["KDP"].attrs["radarx_method"]
        )
        assert kdp["radarx_references"].decode() == (
            volume["sweep_0"]["KDP"].attrs["radarx_references"]
        )
        assert "how" not in f["dataset1"][names["DBZH"]]

    back = open_volume(out, "odim")
    for name in ("sweep_0", "sweep_1"):
        for field in ("DBZH", "ZDR", "KDP", "PHIDP_processed", "ECHO_CLASS"):
            np.testing.assert_allclose(
                back[name][field].values, volume[name][field].values, equal_nan=True
            )
        # the angles and times of the rays are kept by optional_how
        np.testing.assert_allclose(
            back[name].azimuth.values, volume[name].azimuth.values
        )
        dt = back[name].time.values - volume[name].time.values
        assert np.abs(dt / np.timedelta64(1, "ms")).max() < 1
        # lost: the per-ray field and every attribute of a field
        assert "PHIDP_OFFSET" not in back[name].variables
        assert "radarx_method" not in back[name]["KDP"].attrs
        assert back[name]["METEO_MASK"].dtype != bool


def test_odim_without_optional_how_loses_ray_angles(volume, tmp_path):
    out = radarx.io.write_odim(
        volume, tmp_path / "a.h5", source=SOURCE, optional_how=False
    )
    back = open_volume(out, "odim")
    # the reader rebuilds a regular grid from nrays and a1gate
    assert not np.allclose(back["sweep_0"].azimuth.values, volume["sweep_0"].azimuth)
    with h5py.File(out) as f:
        assert "startazA" not in f["dataset1/how"].attrs


def test_odim_fields_subset_and_accessor(volume, tmp_path):
    out = volume.radarx.to_odim(
        tmp_path / "s.h5", source=SOURCE, fields=["KDP", "DBZH"]
    )
    with h5py.File(out) as f:
        quantities = sorted(
            f["dataset1"][k]["what"].attrs["quantity"].decode()
            for k in f["dataset1"]
            if k.startswith("data") and k != "data"
        )
    assert quantities == ["DBZH", "KDP"]
    with pytest.raises(ValueError, match="not fields"):
        radarx.io.write_odim(volume, tmp_path / "x.h5", source=SOURCE, fields=["HID"])


def test_odim_packing_keeps_integer_gain_and_offset(tmp_path):
    vol = make_volume()
    for name in ("sweep_0", "sweep_1"):
        for field in ("DBZH", "ZDR"):
            var = vol[name].ds[field]
            var.encoding.update(
                dtype=np.dtype("uint8"),
                scale_factor=0.5,
                add_offset=-33.0,
                _FillValue=255,
            )
    # reflectivity on the 0.5 dB grid of the encoding, so that packing is exact
    for name in ("sweep_0", "sweep_1"):
        vol[name]["DBZH"] = vol[name]["DBZH"].round(0)
        vol[name]["ZDR"] = (vol[name]["ZDR"] * 2).round(0) / 2
        for field in ("DBZH", "ZDR"):
            vol[name].ds[field].encoding.update(
                dtype=np.dtype("uint8"), scale_factor=0.5, add_offset=-33.0
            )
    kept = radarx.io.write_odim(vol, tmp_path / "k.h5", source=SOURCE)
    floats = radarx.io.write_odim(vol, tmp_path / "f.h5", source=SOURCE, packing="drop")
    with h5py.File(kept) as f, h5py.File(floats) as g:
        what = f["dataset1/data1/what"].attrs
        assert f["dataset1/data1/data"].dtype == np.uint8
        assert what["gain"] == 0.5 and what["offset"] == -33.0
        assert g["dataset1/data1/data"].dtype == np.float64
    assert kept.stat().st_size < floats.stat().st_size
    back = open_volume(kept, "odim")
    np.testing.assert_allclose(
        back["sweep_0"]["DBZH"].values,
        vol["sweep_0"]["DBZH"].values,
        equal_nan=True,
    )


def test_odim_needs_a_radar_identifier(tmp_path):
    with pytest.raises(ValueError, match="NOD"):
        radarx.io.write_odim(make_volume(), tmp_path / "x.h5")


def test_odim_source_from_odim_attributes(tmp_path):
    if XRADAR < Version("0.12.1"):
        pytest.skip("xradar infers the source from its ODIM attributes after 0.12.0")
    vol = make_volume()
    vol.attrs["node"] = "uskbb"
    out = radarx.io.write_odim(vol, tmp_path / "n.h5")
    with h5py.File(out) as f:
        assert f["what"].attrs["source"] == b"NOD:uskbb"


# ---------------------------------------------------------------------------
# the volume writers: common checks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "write", [export.write_cfradial1, export.write_cfradial2, export.write_odim]
)
def test_volume_writers_check_their_input(write, tmp_path):
    with pytest.raises(TypeError, match="DataTree"):
        write(xr.Dataset(), tmp_path / "a")
    with pytest.raises(ValueError, match="sweep_"):
        write(xr.DataTree.from_dict({"/": xr.Dataset()}), tmp_path / "a")


def test_missing_xradar_writer(monkeypatch):
    import xradar.io as xio

    monkeypatch.delattr(xio, "to_odim")
    with pytest.raises(ImportError, match="to_odim"):
        export._xradar_writer("to_odim")


def test_real_volume_round_trips(tmp_path):
    """The same checks on a NEXRAD volume read from a file."""
    try:
        path = DATASETS.fetch("KLBB20160601_150025_V06")
    except Exception as err:  # pragma: no cover - offline
        pytest.skip(f"test data not available: {err}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = xd.io.open_nexradlevel2_datatree(path, sweep=[0, 2])
        raw = raw.map_over_datasets(
            lambda ds: ds.sel(range=slice(None, 30e3)) if "range" in ds.dims else ds
        )
        vol = raw.radarx.assign(R.estimate_kdp(raw)).radarx.assign(R.echo_mask(raw))
    before = snapshot(vol)
    radarx.io.write_cfradial1(vol, tmp_path / "r1.nc")
    radarx.io.write_cfradial2(vol, tmp_path / "r2.nc")
    radarx.io.write_odim(vol, tmp_path / "r.h5", source="NOD:uskbb")
    assert repr(snapshot(vol)) == repr(before)
    # NaN in a field that the reader read as packed integers must stay NaN
    for engine, file in (("cfradial1", "r1.nc"), ("odim", "r.h5")):
        back = open_volume(tmp_path / file, engine)
        np.testing.assert_allclose(
            back["sweep_0"]["KDP"].sortby("azimuth").values,
            vol["sweep_0"]["KDP"].sortby("azimuth").values,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            back["sweep_0"]["DBZH"].sortby("azimuth").values,
            vol["sweep_0"]["DBZH"].sortby("azimuth").values,
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# grids
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def grid():
    return radarx.grid.grid_cones(cone_volume(field="random"), "DBZH", **GRID)


@pytest.fixture(scope="module")
def cappi():
    vol = xr.DataTree.from_dict(
        {
            f"sweep_{k}": _make_ppi_sweep(el, "2024-01-01T00:00:00")
            for k, el in enumerate((1.0, 3.0, 5.0))
        }
    )
    return R.create_cappi(
        vol, height=150.0, fields=["DBZH"], x_res=1000.0, y_res=1000.0
    )


def test_grid_netcdf_cf(grid, tmp_path):
    import pyproj

    before = snapshot(grid)
    out = radarx.io.write_netcdf_cf(grid, tmp_path / "g.nc")
    assert repr(snapshot(grid)) == repr(before)

    with xr.open_dataset(out) as back:
        # values, provenance and attributes of the field
        np.testing.assert_array_equal(back["DBZH"].values, grid["DBZH"].values)
        assert back["DBZH"].attrs["units"] == "dBZ"
        assert provenance_of(back["DBZH"].attrs) == provenance_of(grid["DBZH"].attrs)
        assert provenance_of(back.attrs) == provenance_of(grid.attrs)
        assert back.attrs["Conventions"] == "CF-1.8"
        assert "title" not in back.attrs
        # georeferencing
        assert back["DBZH"].attrs["grid_mapping"] == "crs"
        crs = back["crs"]
        assert crs.attrs["grid_mapping_name"] == "azimuthal_equidistant"
        assert "crs" in back.data_vars and "crs_wkt" not in back.variables
        assert pyproj.CRS.from_cf(crs.attrs).equals(
            pyproj.CRS.from_wkt(grid["crs_wkt"].attrs["crs_wkt"])
        )
        assert back["x"].attrs["standard_name"] == "projection_x_coordinate"
        assert back["x"].attrs["axis"] == "X" and back["y"].attrs["axis"] == "Y"
        assert back["z"].attrs["axis"] == "Z" and back["z"].attrs["positive"] == "up"
        assert back["z"].attrs["standard_name"] == "altitude"
        assert back["time"].attrs["standard_name"] == "time"
        assert back["latitude"].attrs["units"] == "degrees_north"
        assert back["altitude"].attrs["units"] == "m"
        assert "_FillValue" not in back["x_bnds"].encoding
        # 2-D lat and lon, equal to the 1-D ones along the axes through the origin
        assert back["lat"].dims == ("y", "x") and back["lon"].dims == ("y", "x")
        ix, iy = grid.sizes["x"] // 2, grid.sizes["y"] // 2
        np.testing.assert_allclose(
            back["lat"].values[:, ix], grid["lat"].values, atol=1e-9
        )
        np.testing.assert_allclose(
            back["lon"].values[iy, :], grid["lon"].values, atol=1e-9
        )
        # and away from them the old 1-D values were wrong by more than 50 m
        corner = (
            abs(back["lat"].values[-1, -1] - grid["lat"].values[-1]) * 111e3
            + abs(back["lon"].values[-1, -1] - grid["lon"].values[-1]) * 79e3
        )
        assert corner > 50.0
        # bounds
        assert back["x"].attrs["bounds"] == "x_bnds"
        np.testing.assert_allclose(back["x_bnds"].values[0], [-41e3, -39e3])
        np.testing.assert_allclose(back["z_bnds"].values[-1], [5875.0, 6125.0])
        assert back["y_bnds"].dims == ("y", "bnds")


def test_grid_conventions_are_added_to_the_existing_ones(grid, tmp_path):
    tagged = grid.assign_attrs(
        Conventions="ACDD-1.3",
        title="None",
        levels=np.array([1, 2]),
        history="None\ngrid_cones()",
    )
    out = radarx.io.write_netcdf_cf(tagged, tmp_path / "t.nc")
    with xr.open_dataset(out) as back:
        assert back.attrs["Conventions"] == "ACDD-1.3 CF-1.8"
        np.testing.assert_array_equal(back.attrs["levels"], [1, 2])
        assert back.attrs["history"] == "grid_cones()"
    cf = grid.assign_attrs(Conventions="CF-1.10")
    out = radarx.io.write_netcdf_cf(cf, tmp_path / "cf.nc")
    with xr.open_dataset(out) as back:
        assert back.attrs["Conventions"] == "CF-1.10"


def test_grid_without_options_and_accessor(grid, tmp_path):
    out = grid.radarx.to_netcdf_cf(tmp_path / "plain.nc", bounds=False, lonlat=False)
    with xr.open_dataset(out) as back:
        assert back["lat"].dims == ("y",) and "x_bnds" not in back.variables
        assert back["DBZH"].attrs["grid_mapping"] == "crs"
    out = grid["DBZH"].radarx.to_netcdf_cf(tmp_path / "da.nc")
    with xr.open_dataset(out) as back:
        np.testing.assert_array_equal(back["DBZH"].values, grid["DBZH"].values)


def test_grid_netcdf_kwargs_and_dask(grid, tmp_path):
    pytest.importorskip("dask")
    out = export.write_netcdf_cf(
        grid.chunk({"z": 5}), tmp_path / "c.nc", encoding={"DBZH": {"zlib": True}}
    )
    with xr.open_dataset(out) as back:
        np.testing.assert_array_equal(back["DBZH"].values, grid["DBZH"].values)
        assert back["DBZH"].encoding["zlib"]


def test_cappi_gets_a_crs_from_the_radar_position(cappi, tmp_path):
    import pyproj

    out = radarx.io.write_netcdf_cf(cappi, tmp_path / "c.nc")
    with xr.open_dataset(out) as back:
        assert back["DBZH"].attrs["grid_mapping"] == "crs"
        crs = pyproj.CRS.from_cf(back["crs"].attrs)
        assert crs.coordinate_operation.params[0].value == pytest.approx(28.61)
        assert back["x"].attrs["axis"] == "X"
        # a scalar z has no bounds and no altitude standard name
        assert "z_bnds" not in back and "standard_name" not in back["z"].attrs
        assert "lat" in back.coords and back["lat"].dims == ("y", "x")
        np.testing.assert_allclose(back["lat"].values[3, 3], 28.61, atol=1e-6)
        assert back["time"].attrs["standard_name"] == "time"


def test_crs_argument(cappi, tmp_path):
    import pyproj

    none = radarx.io.write_netcdf_cf(cappi, tmp_path / "none.nc", crs=False)
    with xr.open_dataset(none) as back:
        assert "crs" not in back.variables
        assert "grid_mapping" not in back["DBZH"].attrs
        assert "lat" not in back.coords
    tup = radarx.io.write_netcdf_cf(cappi, tmp_path / "tup.nc", crs=(10.0, 20.0))
    with xr.open_dataset(tup) as back:
        assert back["crs"].attrs["longitude_of_projection_origin"] == 20.0
    explicit = radarx.io.write_netcdf_cf(
        cappi, tmp_path / "epsg.nc", crs=pyproj.CRS.from_epsg(32643)
    )
    with xr.open_dataset(explicit) as back:
        assert back["crs"].attrs["grid_mapping_name"] == "transverse_mercator"
        assert back["lat"].dims == ("y", "x")


def test_grid_without_crs_is_written_without_a_mapping(tmp_path):
    ds = xr.Dataset(
        {"f": (("y", "x"), np.ones((2, 3)))},
        coords={"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0]},
    )
    out = radarx.io.write_netcdf_cf(ds, tmp_path / "n.nc")
    with xr.open_dataset(out) as back:
        assert "crs" not in back.variables and back["x"].attrs["axis"] == "X"
        np.testing.assert_allclose(back["y_bnds"].values, [[-0.5, 0.5], [0.5, 1.5]])
    one = xr.Dataset(
        {"f": (("y", "x"), np.ones((1, 1)))}, coords={"x": [0.0], "y": [0.0]}
    )
    out = radarx.io.write_netcdf_cf(one, tmp_path / "one.nc")
    with xr.open_dataset(out) as back:
        assert "x_bnds" not in back.variables


def test_grid_mapping_variable_found_through_the_field(tmp_path):
    ds = xr.Dataset(
        {"f": (("y", "x"), np.ones((2, 2)), {"grid_mapping": "proj"})},
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
    )
    import pyproj

    ds["proj"] = xr.DataArray(0, attrs=pyproj.CRS.from_epsg(32643).to_cf())
    out = radarx.io.write_netcdf_cf(ds, tmp_path / "p.nc")
    with xr.open_dataset(out) as back:
        assert back["crs"].attrs["grid_mapping_name"] == "transverse_mercator"
        assert back["f"].attrs["grid_mapping"] == "crs"
        assert "proj" in back.variables  # not one of the names radarx renames


def test_netcdf_cf_errors(tmp_path):
    with pytest.raises(ValueError, match="x and y"):
        export.write_netcdf_cf(xr.Dataset({"f": ("t", [1.0])}), tmp_path / "a.nc")
    with pytest.raises(TypeError, match="Dataset"):
        export.write_netcdf_cf(make_volume(), tmp_path / "a.nc")


@needs_zarr
def test_zarr_with_cf_georeferencing(grid, tmp_path):
    out = radarx.io.write_zarr(grid, tmp_path / "g.zarr", cf=True)
    with xr.open_dataset(out, engine="zarr") as back:
        assert back["DBZH"].attrs["grid_mapping"] == "crs"
        assert back["lat"].dims == ("y", "x")
        assert provenance_of(back.attrs) == provenance_of(grid.attrs)
        np.testing.assert_array_equal(back["DBZH"].values, grid["DBZH"].values)


def test_time_encoding_does_not_depend_on_the_source(grid, tmp_path):
    stale = grid.copy()
    stale["time"].encoding.update(units="hours since 1900-01-01", dtype="int16")
    a = radarx.io.write_netcdf_cf(stale, tmp_path / "a.nc")
    b = radarx.io.write_netcdf_cf(grid, tmp_path / "b.nc")
    with (
        xr.open_dataset(a, decode_times=False) as ta,
        xr.open_dataset(b, decode_times=False) as tb,
    ):
        assert ta["time"].attrs["units"] == tb["time"].attrs["units"]
        assert ta["time"].values == tb["time"].values
    assert stale["time"].encoding["units"] == "hours since 1900-01-01"


# ---------------------------------------------------------------------------
# registry and accessors
# ---------------------------------------------------------------------------
def test_public_names():
    for name in export.__all__:
        assert callable(getattr(radarx.io, name))
    assert set(export.__all__) == {
        "write_cfradial1",
        "write_cfradial2",
        "write_odim",
        "write_zarr",
        "write_netcdf_cf",
    }


def test_accessors_are_registered():
    from radarx.accessors import (
        RadarxDataArrayAccessor,
        RadarxDataSetAccessor,
        RadarxDataTreeAccessor,
    )

    for name in ("to_cfradial1", "to_cfradial2", "to_odim", "to_zarr", "to_netcdf_cf"):
        assert hasattr(RadarxDataTreeAccessor, name) == (name != "to_netcdf_cf")
    assert hasattr(RadarxDataSetAccessor, "to_netcdf_cf")
    assert hasattr(RadarxDataArrayAccessor, "to_zarr")
