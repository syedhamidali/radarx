#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for finding fields, sweep nodes as input and masks aligned by label."""

import numpy as np
import pytest
import xarray as xr

import radarx
from radarx._fields import VELOCITY, align_like, as_sweep, find_field

from .test_dealias import synthetic_sweep
from .test_hid import _random_volume as hid_volume
from .test_kdp import _volume as kdp_volume
from .test_qc import _sweep as qc_sweep
from .test_wind_profile import vad_sweep

R = radarx.retrieve


def _node(volume, name="sweep_0"):
    return volume[name]


def _no_site(ds):
    return ds.drop_vars(["latitude", "longitude", "altitude"], errors="ignore")


# helpers


def test_as_sweep():
    tree = kdp_volume()
    node = tree["sweep_0"]
    ds = as_sweep(node)
    assert isinstance(ds, xr.Dataset)
    assert "latitude" in ds.coords  # the site of the root is kept
    assert as_sweep(tree) is tree
    sweep = node.to_dataset()
    assert as_sweep(sweep) is sweep
    single = xr.DataTree(xr.Dataset({"a": ("x", [1.0])}))
    assert isinstance(as_sweep(single), xr.Dataset)


def test_find_field_candidates_in_order():
    ds = xr.Dataset({"VEL": ("x", [1.0]), "VRAD": ("x", [2.0])})
    assert find_field(ds, VELOCITY) == "VRAD"
    assert find_field(ds, VELOCITY, "VEL") == "VEL"
    assert find_field(ds, ("a", "VEL")) == "VEL"
    with pytest.raises(KeyError, match="none of"):
        find_field(xr.Dataset({"x": 1.0}), VELOCITY)
    # a volume: the first candidate that any sweep holds
    tree = xr.DataTree.from_dict(
        {"sweep_0": ds[["VEL"]], "sweep_1": xr.Dataset({"VRADV": ("x", [1.0])})}
    )
    assert find_field(tree, VELOCITY) == "VRADV"
    with pytest.raises(KeyError):
        find_field(tree, ("DBZH",))


@pytest.mark.parametrize("name", ["VRADH", "VRAD", "VEL", "velocity", "VRADV"])
def test_velocity_names(name):
    assert name in VELOCITY


# a sweep node is a sweep (#266)


def test_kdp_accepts_a_sweep_node():
    tree = kdp_volume()
    ref = R.estimate_kdp(tree["sweep_0"].to_dataset(inherit=False))
    out = R.estimate_kdp(tree["sweep_0"])
    assert isinstance(out, xr.Dataset)
    np.testing.assert_array_equal(out["KDP"].values, ref["KDP"].values)
    xr.testing.assert_identical(_no_site(out), ref)
    out = tree["sweep_1"].radarx.kdp()
    assert isinstance(out, xr.Dataset) and "KDP" in out


def test_echo_mask_and_apply_mask_accept_a_sweep_node():
    ds, _ = qc_sweep()
    tree = xr.DataTree.from_dict({"/": xr.Dataset(), "sweep_0": ds})
    node = tree["sweep_0"]
    ref = R.echo_mask(ds)
    out = R.echo_mask(node)
    assert isinstance(out, xr.Dataset)
    np.testing.assert_array_equal(out["METEO_MASK"].values, ref["METEO_MASK"].values)
    masked = R.apply_mask(node, out)
    expect = R.apply_mask(ds, ref)
    np.testing.assert_array_equal(masked["DBZH"].values, expect["DBZH"].values)
    # the node of a mask volume as the mask, and the mask computed on the fly
    both = R.apply_mask(node, R.echo_mask(tree)["sweep_0"])
    np.testing.assert_array_equal(both["DBZH"].values, expect["DBZH"].values)
    np.testing.assert_array_equal(
        R.apply_mask(node)["DBZH"].values, expect["DBZH"].values
    )


def test_velocity_retrievals_accept_a_sweep_node():
    ds, _ = synthetic_sweep(nray=60, ngate=200)
    tree = xr.DataTree.from_dict({"/": xr.Dataset(), "sweep_0": ds})
    node = tree["sweep_0"]
    dealiased = R.dealias_velocity(node)
    ref = R.dealias_velocity(ds)
    assert isinstance(dealiased, xr.DataArray)
    np.testing.assert_array_equal(dealiased.values, ref.values)
    shear = R.llsd(node)
    assert isinstance(shear, xr.Dataset)
    np.testing.assert_array_equal(
        shear["azimuthal_shear"].values, R.llsd(ds)["azimuthal_shear"].values
    )
    assert isinstance(R.azimuthal_shear(node), xr.DataArray)
    assert isinstance(R.radial_divergence(node), xr.DataArray)
    assert isinstance(node.radarx.llsd(), xr.Dataset)


def test_vad_profile_accepts_a_sweep_node():
    tree = xr.DataTree.from_dict({"/": xr.Dataset(), "sweep_0": vad_sweep()})
    out = R.vad_profile(tree["sweep_0"])
    ref = R.vad_profile(vad_sweep())
    np.testing.assert_array_equal(out["u"].values, ref["u"].values)


def test_a_volume_is_still_a_volume():
    tree = kdp_volume()
    out = R.estimate_kdp(tree)
    assert isinstance(out, xr.DataTree)
    assert {"sweep_0", "sweep_1", "sweep_2"} <= set(out.children)


# the velocity field is found (#268)


def _renamed(name):
    ds, _ = synthetic_sweep(nray=60, ngate=200)
    return ds.rename({"VRADH": name})


@pytest.mark.parametrize("name", ["VRAD", "VEL", "velocity"])
def test_velocity_is_found_by_name_candidates(name):
    ref, _ = synthetic_sweep(nray=60, ngate=200)
    ds = _renamed(name)
    shear = R.llsd(ds)
    assert shear["azimuthal_shear"].attrs["source_field"] == name
    np.testing.assert_array_equal(
        shear["azimuthal_shear"].values, R.llsd(ref)["azimuthal_shear"].values
    )
    assert R.azimuthal_shear(ds).name == "azimuthal_shear"
    assert R.radial_divergence(ds).name == "radial_divergence"
    dealiased = R.dealias_velocity(ds)
    assert dealiased.name == f"{name}_dealiased"
    np.testing.assert_array_equal(dealiased.values, R.dealias_velocity(ref).values)
    assert ds.radarx.llsd()["azimuthal_shear"].attrs["source_field"] == name


def test_velocity_is_found_in_a_volume_and_named_fields_win():
    ds = _renamed("VRAD")
    tree = xr.DataTree.from_dict({"/": xr.Dataset(), "sweep_0": ds, "sweep_1": ds})
    out = R.llsd(tree)
    assert set(out.children) == {"sweep_0", "sweep_1"}
    assert set(R.dealias_velocity(tree).children) == {"sweep_0", "sweep_1"}
    both = ds.assign(VRADH=ds["VRAD"] * 2.0)
    assert R.llsd(both)["azimuthal_shear"].attrs["source_field"] == "VRADH"
    named = R.llsd(both, "VRAD")
    assert named["azimuthal_shear"].attrs["source_field"] == "VRAD"
    with pytest.raises(KeyError, match="none of"):
        R.llsd(ds.rename({"VRAD": "other"}))
    with pytest.raises(KeyError):
        R.llsd(ds, "nope")


def test_vad_profile_finds_vrad():
    ds = vad_sweep().rename({"VRADH": "VRAD"})
    out = R.vad_profile(ds)
    ref = R.vad_profile(vad_sweep())
    np.testing.assert_array_equal(out["u"].values, ref["u"].values)


# masks are matched by label (#263)


def test_align_like():
    a = xr.DataArray(
        np.arange(6).reshape(3, 2),
        dims=("azimuth", "range"),
        coords={"azimuth": [3, 1, 2]},
    )
    like = xr.DataArray(
        np.zeros((3, 2)), dims=("azimuth", "range"), coords={"azimuth": [1, 2, 3]}
    )
    out = align_like(a, like)
    np.testing.assert_array_equal(out["azimuth"].values, [1, 2, 3])
    np.testing.assert_array_equal(out.values, [[2, 3], [4, 5], [0, 1]])
    assert align_like(like, like) is like
    wider = like.reindex(azimuth=[0, 1, 2, 3, 4], fill_value=7.0)
    np.testing.assert_array_equal(align_like(wider, like)["azimuth"].values, [1, 2, 3])
    with pytest.raises(ValueError, match="labels"):
        align_like(like.assign_coords(azimuth=[1, 2, 4]), like)
    with pytest.raises(ValueError, match="labels"):
        align_like(like.isel(azimuth=slice(0, 2)), like)
    dup = like.assign_coords(azimuth=[1, 1, 2])
    with pytest.raises(ValueError, match="labels"):
        align_like(like, dup.copy().assign_coords(azimuth=[1, 1, 2]))
    # dimensions without labels are left to the caller
    plain = xr.DataArray(np.zeros((3, 2)), dims=("azimuth", "range"))
    assert align_like(plain, like) is plain


def test_apply_mask_matches_the_mask_by_azimuth():
    ds, _ = qc_sweep(nray=60)
    # the echoes of the synthetic sweep do not change with azimuth: keep a quarter
    mask = R.echo_mask(ds)["METEO_MASK"] & (ds["azimuth"] < 90.0)
    ref = R.apply_mask(ds, mask)
    flipped = mask.isel(azimuth=slice(None, None, -1))
    assert not np.array_equal(flipped.values, mask.values)
    out = R.apply_mask(ds, flipped)
    np.testing.assert_array_equal(out["DBZH"].values, ref["DBZH"].values)
    # a mask for other azimuths is refused, not applied by position
    shifted = mask.assign_coords(azimuth=mask["azimuth"] + 0.25)
    with pytest.raises(ValueError, match="azimuth"):
        R.apply_mask(ds, shifted)
    with pytest.raises(ValueError, match="azimuth"):
        R.apply_mask(ds, R.echo_mask(ds).assign_coords(azimuth=mask["azimuth"] + 1.0))
    # a mask without labels still matches by position
    bare = xr.DataArray(mask.values, dims=mask.dims)
    np.testing.assert_array_equal(
        R.apply_mask(ds, bare)["DBZH"].values, ref["DBZH"].values
    )


def test_apply_mask_volume_matches_the_mask_by_azimuth():
    ds, _ = qc_sweep(nray=60)
    tree = xr.DataTree.from_dict({"/": xr.Dataset(), "sweep_0": ds})
    qc = R.echo_mask(tree)
    quarter = qc["sweep_0"].to_dataset(inherit=False)
    quarter = quarter.assign(
        METEO_MASK=quarter["METEO_MASK"] & (quarter["azimuth"] < 90.0)
    )
    ref = R.apply_mask(tree, xr.DataTree.from_dict({"sweep_0": quarter}))
    flipped = xr.DataTree.from_dict(
        {"sweep_0": quarter.isel(azimuth=slice(None, None, -1))}
    )
    out = R.apply_mask(tree, flipped)
    np.testing.assert_array_equal(
        out["sweep_0"]["DBZH"].values, ref["sweep_0"]["DBZH"].values
    )
    assert np.isnan(out["sweep_0"]["DBZH"].values[30:]).all()


def test_hid_matches_the_mask_by_azimuth():
    ds = hid_volume(n=1)[0]
    keep = ds["MET"]
    ref = R.hid(ds.drop_vars("MET"), temperature="TEMP", mask=keep)
    flipped = keep.isel(azimuth=slice(None, None, -1))
    out = R.hid(ds.drop_vars("MET"), temperature="TEMP", mask=flipped)
    np.testing.assert_array_equal(out["HID"].values, ref["HID"].values)
    shifted = keep.assign_coords(azimuth=keep["azimuth"] + 0.25)
    with pytest.raises(ValueError, match="azimuth"):
        R.hid(ds.drop_vars("MET"), temperature="TEMP", mask=shifted)
