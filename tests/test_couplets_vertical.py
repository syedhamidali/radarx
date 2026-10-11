#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for the vertical continuity of ``rotation_couplets`` (``min_tilts``)."""

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401  registers the accessors
from radarx.fundamentals.geometry import ground_range
from radarx.retrieve import rotation_couplets
from radarx.retrieve import tornado as tor_mod

AZ = np.arange(0.25, 360.0, 0.5)
RNG = np.arange(2125.0, 60e3, 250.0)


def vortices(centres, angle, vmax=40.0, core=700.0, signs=None):
    """A tilt with Rankine vortices at ``centres`` (x, y in metres)."""
    A, R = np.meshgrid(np.radians(AZ), RNG, indexing="ij")
    x, y = R * np.sin(A), R * np.cos(A)
    u = np.zeros_like(x)
    v = np.zeros_like(x)
    for k, (cx, cy) in enumerate(centres):
        sign = 1.0 if signs is None else signs[k]
        dx, dy = x - cx, y - cy
        rho = np.maximum(np.hypot(dx, dy), 1e-6)
        vt = sign * np.where(rho < core, vmax * rho / core, vmax * core / rho)
        u += -vt * dy / rho
        v += vt * dx / rho
    vr = u * np.sin(A) + v * np.cos(A)
    return xr.Dataset(
        {"VRADH": (("azimuth", "range"), vr)},
        coords={
            "azimuth": AZ,
            "range": RNG,
            "sweep_fixed_angle": angle,
            "elevation": ("azimuth", np.full(AZ.size, angle)),
        },
    )


def volume(layout, signs=None):
    """DataTree from ``{(name): (angle, centres)}``."""
    nodes = {"/": xr.Dataset()}
    for name, (angle, centres) in layout.items():
        nodes[name] = vortices(centres, angle, signs=signs and signs.get(name))
    return xr.DataTree.from_dict(nodes)


def place(distance, bearing_deg):
    b = np.radians(bearing_deg)
    return (distance * np.sin(b), distance * np.cos(b))


COLUMN = place(30e3, 20.0)
OTHER = place(40e3, 200.0)


def test_chain_of_tilts_and_an_isolated_couplet():
    layout = {
        "sweep_0": (0.5, [COLUMN, OTHER]),
        "sweep_1": (1.5, [COLUMN]),
        "sweep_2": (2.5, [COLUMN]),
        "sweep_3": (3.5, [COLUMN]),
    }
    vol = volume(layout)
    plain = rotation_couplets(vol)
    assert "n_tilts" not in plain["sweep_0"].data_vars
    assert plain["sweep_0"].sizes["couplet"] == 2
    out = rotation_couplets(vol, min_tilts=1)
    n0 = out["sweep_0"]["n_tilts"].values
    az0 = out["sweep_0"]["azimuth"].values
    # the vortex at 200 degrees exists on one tilt only
    far = np.abs(az0 - 200.0) < 10.0
    assert far.sum() == 1 and n0[far][0] == 1 and (n0[~far] == 4).all()
    kept = rotation_couplets(vol, min_tilts=3)
    assert kept["sweep_0"].sizes["couplet"] == 1
    assert (kept["sweep_0"]["n_tilts"].values == 4).all()
    for name in ("sweep_1", "sweep_2", "sweep_3"):
        assert kept[name].sizes["couplet"] == 1
    # all couplets of the chain are one circulation
    ids = {int(kept[n]["vortex"].values[0]) for n in kept.children}
    assert len(ids) == 1
    assert kept["sweep_0"].attrs["min_tilts"] == 3
    assert kept["sweep_0"].attrs["link_distance_m"] == 5000.0
    # a chain of 4 does not meet 5
    none = rotation_couplets(vol, min_tilts=5)
    assert all(none[n].sizes["couplet"] == 0 for n in none.children)
    # the strongest first ordering and the other variables are untouched
    assert list(kept["sweep_0"].data_vars)[:4] == [
        "azimuthal_shear",
        "delta_v",
        "rotational_velocity",
        "area",
    ]


def test_missing_tilt_needs_max_skip():
    layout = {
        "sweep_0": (0.5, [COLUMN]),
        "sweep_1": (1.5, [COLUMN]),
        "sweep_2": (2.5, []),
        "sweep_3": (3.5, [COLUMN]),
        "sweep_4": (4.5, [COLUMN]),
    }
    vol = volume(layout)
    two = rotation_couplets(vol, min_tilts=2)
    assert two["sweep_0"].sizes["couplet"] == 1
    three = rotation_couplets(vol, min_tilts=3)
    assert all(three[n].sizes["couplet"] == 0 for n in three.children)
    bridged = rotation_couplets(vol, min_tilts=4, max_skip=1)
    assert all(bridged[n].sizes["couplet"] == 1 for n in ("sweep_0", "sweep_4"))
    assert bridged["sweep_0"]["n_tilts"].values[0] == 4
    assert bridged["sweep_0"].attrs["max_skip"] == 1


def test_link_distance_and_tilting_circulation():
    shift = 3000.0
    layout = {
        f"sweep_{k}": (0.5 + k, [(COLUMN[0] + k * shift, COLUMN[1])]) for k in range(3)
    }
    vol = volume(layout)
    linked = rotation_couplets(vol, min_tilts=3)
    assert all(linked[n].sizes["couplet"] == 1 for n in linked.children)
    # the peaks are 3 km apart: a shorter link distance cuts the chain
    cut = rotation_couplets(vol, min_tilts=3, link_distance=2000.0)
    assert all(cut[n].sizes["couplet"] == 0 for n in cut.children)
    assert (
        rotation_couplets(vol, min_tilts=2, link_distance=2000.0)["sweep_0"].sizes[
            "couplet"
        ]
        == 0
    )


def test_opposite_rotation_is_not_linked():
    x = np.array([0.0, 100.0])
    y = np.zeros(2)
    level = np.array([0, 1])
    assert tor_mod._links(x, y, np.array([1.0, -1.0]), level, 5000.0, 0) == []
    assert tor_mod._links(x, y, np.array([1.0, 1.0]), level, 5000.0, 0) == [(1, 0)]
    layout = {"sweep_0": (0.5, [COLUMN]), "sweep_1": (1.5, [COLUMN])}
    # the strongest couplet of each sweep is the core; the flanks of a vortex
    # have the opposite sign and can touch the core of the other one
    opposite = rotation_couplets(
        volume(layout, signs={"sweep_1": [-1.0]}), rotation="both", min_tilts=1
    )
    assert (
        opposite["sweep_0"]["vortex"].values[0]
        != opposite["sweep_1"]["vortex"].values[0]
    )
    same = rotation_couplets(volume(layout), rotation="both", min_tilts=1)
    assert same["sweep_0"]["vortex"].values[0] == same["sweep_1"]["vortex"].values[0]
    assert same["sweep_0"]["n_tilts"].values[0] == 2


def test_repeated_cuts_of_one_angle_are_one_tilt():
    layout = {
        "sweep_0": (0.5, [COLUMN]),
        "sweep_1": (0.5, [COLUMN]),
        "sweep_2": (1.5, [COLUMN]),
    }
    vol = volume(layout)
    out = rotation_couplets(vol, min_tilts=1)
    assert [
        int(out[n]["n_tilts"].values[0]) for n in ("sweep_0", "sweep_1", "sweep_2")
    ] == [2, 2, 2]
    assert all(
        rotation_couplets(vol, min_tilts=3)[n].sizes["couplet"] == 0
        for n in ("sweep_0", "sweep_1", "sweep_2")
    )
    # both repeated cuts link to the cut above
    assert rotation_couplets(vol, min_tilts=2)["sweep_0"].sizes["couplet"] == 1


def test_default_numbers_do_not_change():
    layout = {"sweep_0": (0.5, [COLUMN]), "sweep_1": (1.5, [COLUMN])}
    vol = volume(layout)
    base = rotation_couplets(vol)
    withkw = rotation_couplets(vol, min_tilts=1)
    for name in base.children:
        for var in base[name].data_vars:
            np.testing.assert_array_equal(
                base[name][var].values, withkw[name][var].values
            )
    assert vol.radarx.rotation_couplets(min_tilts=2)["sweep_1"].sizes["couplet"] == 1


def test_positions_use_the_ground_range_of_each_tilt():
    """The peak of a high tilt is placed at its ground range, not its slant range."""
    layout = {"sweep_0": (0.5, [COLUMN]), "sweep_1": (12.0, [COLUMN])}
    vol = volume(layout)
    out = rotation_couplets(vol, min_tilts=2)
    assert out["sweep_1"].sizes["couplet"] == 1
    # without the correction the slant range of the 12 degree peak differs by
    # 2 percent of 30 km, which is inside the link distance anyway: check the
    # helper itself
    nodes = {n: rotation_couplets(vol)[n].to_dataset() for n in vol.children}
    s = ground_range(nodes["sweep_1"]["range"].values, 12.0)
    assert (s < nodes["sweep_1"]["range"].values).all()
    found = tor_mod._vertical_continuity(nodes, 5000.0, 0)
    assert found["sweep_0"][0] == [2] and found["sweep_1"][0] == [2]


def test_errors():
    vol = volume({"sweep_0": (0.5, [COLUMN]), "sweep_1": (1.5, [COLUMN])})
    with pytest.raises(ValueError, match="min_tilts must be"):
        rotation_couplets(vol, min_tilts=0)
    with pytest.raises(ValueError, match="min_tilts must be"):
        rotation_couplets(vol, min_tilts=1.5)
    with pytest.raises(ValueError, match="link_distance"):
        rotation_couplets(vol, min_tilts=2, link_distance=0.0)
    with pytest.raises(ValueError, match="max_skip"):
        rotation_couplets(vol, min_tilts=2, max_skip=-1)
    with pytest.raises(ValueError, match="needs a volume"):
        rotation_couplets(vol["sweep_0"].to_dataset(), min_tilts=2)


def test_provenance_names_the_algorithm_description():
    vol = volume({"sweep_0": (0.5, [COLUMN]), "sweep_1": (1.5, [COLUMN])})
    out = rotation_couplets(vol, min_tilts=2)
    assert "roc-mesocyclone-detection" in out["sweep_0"].attrs["radarx_references"]
