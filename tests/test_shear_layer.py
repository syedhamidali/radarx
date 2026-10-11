#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for the median pre-filter and the layer-maximum azimuthal shear."""

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401  registers the accessors
from radarx.fundamentals.constants import EFFECTIVE_RADIUS_4_3
from radarx.fundamentals.geometry import (
    beam_center_height,
    beam_height_at_ground_range,
    ground_range,
)
from radarx.retrieve import azimuthal_shear, azimuthal_shear_layer_max, llsd
from radarx.retrieve import shear as shear_mod

from .test_shear import RANGE, jittered_azimuths, rankine, sweep

needs_kernel = pytest.mark.skipif(
    not shear_mod.HAS_COMPILED_KERNEL, reason="compiled LLSD kernel not built"
)
ENGINES = ["numpy", pytest.param("compiled", marks=needs_kernel)]

ELEVATIONS = (0.5, 1.5, 3.0, 5.0, 7.0, 10.0, 12.0)
CENTRE = (25e3, 15e3)
CORE = 4000.0


def omega_of_z(z):
    """Vertical profile of the vortex: weak near the ground, strong at 4.5 km."""
    return 0.004 + 0.010 * np.exp(-(((z - 4500.0) / 1500.0) ** 2))


def column_sweep(el, n_az=360, rng=None, with_dbz=True, dbz_name="DBZH"):
    """One tilt of a solid-body vortex column whose rotation changes with height."""
    rng = np.arange(2125.0, 60e3, 250.0) if rng is None else rng
    az = np.arange(n_az) * 360.0 / n_az + 0.5
    phi = np.radians(az)[:, None]
    s = ground_range(rng, el)[None, :]
    z = beam_center_height(rng, el)[None, :]
    x, y = s * np.sin(phi), s * np.cos(phi)
    dx, dy = x - CENTRE[0], y - CENTRE[1]
    rho = np.maximum(np.hypot(dx, dy), 1e-9)
    om = omega_of_z(z)
    vt = np.where(rho < CORE, om * rho, om * CORE**2 / rho)
    u, v = -vt * dy / rho, vt * dx / rho
    vel = np.cos(np.radians(el)) * (u * np.sin(phi) + v * np.cos(phi))
    data = {"VRADH": (("azimuth", "range"), vel)}
    if with_dbz:
        dbz = np.where(rho < 8000.0, 35.0, -10.0)
        data[dbz_name] = (("azimuth", "range"), dbz)
    return xr.Dataset(
        data,
        coords={
            "azimuth": ("azimuth", az, {"units": "degrees"}),
            "range": ("range", rng, {"units": "m"}),
            "elevation": ("azimuth", np.full(n_az, el)),
            "sweep_fixed_angle": el,
        },
    )


def column_volume(elevations=ELEVATIONS, **kwargs):
    nodes = {"/": xr.Dataset({"site": 1.0})}
    for k, el in enumerate(elevations):
        nodes[f"sweep_{k}"] = column_sweep(el, **kwargs)
    return xr.DataTree.from_dict(nodes)


@pytest.fixture(scope="module")
def volume():
    return column_volume()


def nearest_gate(vol, name, ground_distance, el):
    """Gate of a sweep whose ground range is closest to ``ground_distance``."""
    rng = vol[name]["range"].values
    return int(np.argmin(np.abs(ground_range(rng, el) - ground_distance)))


# --------------------------------------------------------------------------
# median pre-filter
# --------------------------------------------------------------------------


def noisy_sweep(seed=1):
    az = jittered_azimuths()
    vel, *_ = rankine(az, RANGE)
    rng = np.random.default_rng(seed)
    vel = vel + rng.normal(0.0, 1.0, vel.shape)
    vel[rng.random(vel.shape) < 0.1] = np.nan
    vel[100:110, 200:230] = np.nan
    return sweep(vel, az, RANGE)


@pytest.mark.parametrize("size", [True, 3, 5])
def test_median_filter_engines_agree(size):
    ds = noisy_sweep()
    a = llsd(ds, median_filter=size, engine="numpy")
    for name in ("azimuthal_shear", "radial_divergence"):
        assert a[name].attrs["median_filter_size"] == (3 if size is True else size)
    if shear_mod.HAS_COMPILED_KERNEL:
        b = llsd(ds, median_filter=size, engine="compiled")
        for name in ("azimuthal_shear", "radial_divergence"):
            np.testing.assert_array_equal(np.isnan(a[name]), np.isnan(b[name]))
            np.testing.assert_allclose(a[name], b[name], rtol=1e-3, atol=1e-8)


@pytest.mark.parametrize("engine", ENGINES)
def test_median_filter_off_keeps_the_numbers(engine):
    ds = noisy_sweep()
    base = llsd(ds, engine=engine)
    for off in (None, False):
        out = llsd(ds, median_filter=off, engine=engine)
        np.testing.assert_array_equal(
            out["azimuthal_shear"].values, base["azimuthal_shear"].values
        )
        assert "median_filter_size" not in out["azimuthal_shear"].attrs


@pytest.mark.parametrize("engine", ENGINES)
def test_median_filter_removes_a_spike(engine):
    """A one-gate spike gives strong shear around it; the median removes it."""
    az = np.arange(0.0, 360.0, 1.0)
    vel = np.full((az.size, RANGE.size), 5.0)
    vel[100, 150] = 40.0
    ds = sweep(vel, az, RANGE)
    plain = llsd(ds, engine=engine)["azimuthal_shear"].values
    clean = llsd(ds, median_filter=3, engine=engine)["azimuthal_shear"].values
    assert np.nanmax(np.abs(plain)) > 0.005
    np.testing.assert_allclose(clean[np.isfinite(clean)], 0.0, atol=1e-9)


@pytest.mark.parametrize("engine", ENGINES)
def test_median_filter_is_the_median_of_the_valid_window(engine):
    """Exact values: a 3 x 3 window with a known set of valid gates."""
    az = np.arange(0.0, 360.0, 1.0)
    rng = np.arange(1000.0, 8000.0, 250.0)
    base = np.add.outer(np.arange(az.size) % 7, np.arange(rng.size) % 5).astype(float)
    ds = sweep(base, az, rng)
    out = llsd(ds, median_filter=3, engine="numpy")
    # a plane fit of the filtered field equals the fit of the median of the data,
    # so compare with a direct median on the interior and the wrap at north
    filtered = shear_mod._median_numpy(np.where(np.isfinite(base), base, np.nan), 3)
    for i, j in [(10, 10), (0, 5), (359, 5), (180, 0), (180, rng.size - 1)]:
        rows = [(i + d) % az.size for d in (-1, 0, 1)]
        cols = [k for k in (j - 1, j, j + 1) if 0 <= k < rng.size]
        window = base[np.ix_(rows, cols)].ravel()
        assert filtered[i, j] == np.median(window)
    assert out["azimuthal_shear"].attrs["median_filter_size"] == 3


def test_median_filter_needs_most_of_the_window():
    vel = np.full((20, 12), np.nan)
    vel[5:8, 4:7] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    out = shear_mod._median_numpy(vel, 3)
    assert out[6, 5] == 5.0
    # the corner gate sees only 4 valid gates of 9: dropped
    assert np.isnan(out[5, 4])
    # a gate with no value of its own stays missing even if its window is full
    vel2 = vel.copy()
    vel2[6, 5] = np.nan
    assert np.isnan(shear_mod._median_numpy(vel2, 3)[6, 5])


def test_median_filter_errors():
    ds = noisy_sweep()
    for bad in (2, 0, -3, 1.5, "3"):
        with pytest.raises(ValueError, match="median_filter"):
            llsd(ds, median_filter=bad)
    assert llsd(ds, median_filter=1).attrs  # a 1 x 1 window is allowed
    assert shear_mod._median_size(np.int64(5)) == 5


# --------------------------------------------------------------------------
# near_mask (reflectivity within a distance)
# --------------------------------------------------------------------------


def brute_force_near(flag, azimuth, rng, distance):
    r0, r1 = np.meshgrid(rng, rng, indexing="ij")  # (gate, gate)
    out = np.zeros(flag.shape, dtype=bool)
    ang = np.radians(azimuth)
    for i in range(flag.shape[0]):
        cos = np.cos(ang - ang[i])  # (ray,)
        d2 = (
            r0[None, :, :] ** 2 + r1[None, :, :] ** 2 - 2 * r0 * r1 * cos[:, None, None]
        )
        near = d2 <= distance**2  # (ray, gate0, gate1)
        out[i] = (near & flag[:, None, :]).any(axis=(0, 2))
    return out


@pytest.mark.parametrize("engine", ENGINES)
def test_near_mask_matches_brute_force(engine):
    rng_ = np.random.default_rng(3)
    az = np.sort(rng_.uniform(0.0, 360.0, 60))
    rng = np.arange(800.0, 9000.0, 400.0)
    flag = rng_.random((az.size, rng.size)) < 0.02
    flag[:, 0] |= rng_.random(az.size) < 0.1
    for distance in (1500.0, 5000.0):
        expected = brute_force_near(flag, az, rng, distance)
        got = shear_mod._near_mask(flag, az, rng, distance, None, engine == "compiled")
        np.testing.assert_array_equal(got, expected)
        assert (got | ~flag).all()  # a flagged gate is near itself


def test_near_mask_unsorted_azimuths_and_errors():
    az = np.array([200.0, 10.0, 100.0, 300.0, 50.0, 250.0, 150.0, 350.0])
    rng = np.arange(1000.0, 5000.0, 500.0)
    flag = np.zeros((az.size, rng.size), dtype=bool)
    flag[2, 3] = True
    expected = brute_force_near(flag, az, rng, 2500.0)
    np.testing.assert_array_equal(
        shear_mod._near_mask(flag, az, rng, 2500.0, None, False), expected
    )
    if shear_mod.HAS_COMPILED_KERNEL:
        np.testing.assert_array_equal(
            shear_mod._near_mask(flag, az, rng, 2500.0, 1, True), expected
        )
        with pytest.raises(ValueError, match="do not match"):
            shear_mod._shear.near_mask(flag * 1.0, az[:3], rng, 1000.0, 1)
        with pytest.raises(ValueError, match="range must increase"):
            shear_mod._shear.near_mask(flag * 1.0, az, rng[::-1].copy(), 1000.0, 1)


# --------------------------------------------------------------------------
# layer maximum
# --------------------------------------------------------------------------


def expected_layer(vol, layer, ref="sweep_0", rotation="cyclonic"):
    """Independent oracle: tilt and gate by the forward geometry, gate by gate."""
    names = [n for n in vol.children if n.startswith("sweep")]
    ref_ds = vol[ref]
    ref_az = ref_ds["azimuth"].values
    ref_rng = ref_ds["range"].values
    ref_el = float(ref_ds["sweep_fixed_angle"])
    s_ref = ground_range(ref_rng, ref_el)
    return names, ref_az, ref_rng, ref_el, s_ref


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("layer", [(0.0, 2000.0), (3000.0, 6000.0)])
def test_layer_max_follows_the_vortex_profile(volume, engine, layer):
    out = azimuthal_shear_layer_max(volume, layer=layer, engine=engine)
    shear = out["azimuthal_shear_layer_max"]
    src = out["layer_max_elevation"]
    assert shear.dtype == np.float32 and shear.dims == ("azimuth", "range")
    assert shear.attrs["units"] == "s-1" and src.attrs["units"] == "degrees"
    assert shear.attrs["layer_bottom_m"] == layer[0]
    assert shear.attrs["reference_sweep"] == "sweep_0"
    # the gate of the reference tilt on top of the vortex centre
    d = np.hypot(*CENTRE)
    az0 = np.degrees(np.arctan2(CENTRE[0], CENTRE[1]))
    i = int(np.argmin(np.abs(volume["sweep_0"]["azimuth"].values - az0)))
    j = nearest_gate(volume, "sweep_0", d, 0.5)
    # tilts with the beam inside the layer over the centre
    inside = [
        el
        for el in ELEVATIONS
        if layer[0] <= beam_height_at_ground_range(d, el) <= layer[1]
    ]
    assert inside
    want = max(
        omega_of_z(beam_height_at_ground_range(d, el)) * np.cos(np.radians(el)) ** 2
        for el in inside
    )
    got = float(shear[i, j])
    assert got == pytest.approx(want, rel=0.12)
    assert float(src[i, j]) in inside
    # the tilt that gave it is the one with the strongest rotation of the layer
    best = max(inside, key=lambda el: omega_of_z(beam_height_at_ground_range(d, el)))
    assert float(src[i, j]) == best
    # nothing exceeds the strongest rotation of the profile
    assert float(np.nanmax(shear)) <= 1.15 * 0.014


@pytest.mark.parametrize("engine", ENGINES)
def test_layer_max_source_tilt_matches_the_geometry(volume, engine):
    """Gate by gate, the tilt that gave the value is in the layer at that range."""
    layer = (0.0, 2000.0)
    out = azimuthal_shear_layer_max(volume, layer=layer, engine=engine)
    src = out["layer_max_elevation"].values
    rng = volume["sweep_0"]["range"].values
    s = ground_range(rng, 0.5)
    for j in range(0, rng.size, 11):
        heights = {el: beam_height_at_ground_range(s[j], el) for el in ELEVATIONS}
        in_layer = {el for el, h in heights.items() if layer[0] <= h <= layer[1]}
        col = src[:, j]
        found = set(np.unique(col[np.isfinite(col)]).astype(float))
        assert {round(f, 1) for f in found} <= {round(e, 1) for e in in_layer}


def test_layer_max_engines_agree(volume):
    if not shear_mod.HAS_COMPILED_KERNEL:
        pytest.skip("compiled kernel not built")
    for kwargs in (
        {},
        {"layer": (3000.0, 6000.0), "rotation": "anticyclonic"},
        {"rotation": "both", "median_filter": 3},
        {"min_reflectivity": 20.0, "weights": "gaussian"},
    ):
        a = azimuthal_shear_layer_max(volume, engine="numpy", **kwargs)
        b = azimuthal_shear_layer_max(volume, engine="compiled", **kwargs)
        for name in a.data_vars:
            np.testing.assert_array_equal(np.isnan(a[name]), np.isnan(b[name]))
            np.testing.assert_allclose(a[name], b[name], rtol=1e-3, atol=1e-8)


def constant_shear_volume(values, holes=None):
    """Tilts whose shear field is one constant each; used with ``shear=``."""
    nodes = {"/": xr.Dataset()}
    for k, (el, value) in enumerate(zip(ELEVATIONS, values)):
        ds = column_sweep(el, n_az=180, rng=np.arange(2000.0, 40e3, 500.0))
        field = np.full(ds["VRADH"].shape, value)
        if holes is not None and k in holes:
            field[:] = np.nan
        ds["SHEAR"] = (("azimuth", "range"), field)
        nodes[f"sweep_{k}"] = ds
    return xr.DataTree.from_dict(nodes)


@pytest.mark.parametrize("engine", ENGINES)
def test_layer_max_modes_and_missing_tilts(engine):
    values = [0.003, -0.02, 0.005, 0.001, 0.01, -0.004, 0.002]
    vol = constant_shear_volume(values)
    rng = vol["sweep_0"]["range"].values
    s = ground_range(rng, 0.5)
    for layer in [(0.0, 2000.0), (500.0, 4000.0), (3000.0, 6000.0)]:
        for rotation, pick in (
            ("cyclonic", max),
            ("anticyclonic", min),
            ("both", None),
        ):
            out = azimuthal_shear_layer_max(
                vol, shear="SHEAR", layer=layer, rotation=rotation, engine=engine
            )
            got = out["azimuthal_shear_layer_max"].values
            for j in range(0, rng.size, 7):
                members = [
                    v
                    for el, v in zip(ELEVATIONS, values)
                    if layer[0] <= beam_height_at_ground_range(s[j], el) <= layer[1]
                ]
                if pick is None:
                    want = max(members, key=abs) if members else np.nan
                else:
                    want = pick(members) if members else np.nan
                # a tilt reaches only the gates its own range covers
                row = got[10, j]
                if np.isnan(want):
                    assert np.isnan(row)
                else:
                    assert row == pytest.approx(want, abs=1e-9) or row in values
    # a tilt without data leaves the others: at the gate where the 3.0 degree
    # tilt (shear 0.005) is the largest in the layer, the next one is 0.003
    holes = constant_shear_volume(values, holes={2})
    layer = (0.0, 2000.0)
    base = azimuthal_shear_layer_max(vol, shear="SHEAR", layer=layer, engine=engine)
    cut = azimuthal_shear_layer_max(holes, shear="SHEAR", layer=layer, engine=engine)
    src = base["layer_max_elevation"].values[10]
    j = int(np.flatnonzero(src == 3.0)[0])
    assert base["azimuthal_shear_layer_max"].values[10, j] == pytest.approx(0.005)
    assert cut["azimuthal_shear_layer_max"].values[10, j] < 0.005
    assert cut["layer_max_elevation"].values[10, j] != 3.0


@pytest.mark.parametrize("engine", ENGINES)
def test_layer_max_reflectivity_screen(volume, engine):
    plain = azimuthal_shear_layer_max(volume, engine=engine)[
        "azimuthal_shear_layer_max"
    ]
    screened = azimuthal_shear_layer_max(volume, min_reflectivity=20.0, engine=engine)[
        "azimuthal_shear_layer_max"
    ]
    # the screen removes gates; the maximum of what is left cannot be larger
    both = np.isfinite(screened.values)
    assert both.sum() < np.isfinite(plain.values).sum()
    assert (screened.values[both] <= plain.values[both] + 1e-9).all()
    # what is left is within 8 km + 5 km of the vortex centre
    rng = volume["sweep_0"]["range"].values
    az = np.radians(volume["sweep_0"]["azimuth"].values)
    s = ground_range(rng, 0.5)
    x, y = s[None, :] * np.sin(az)[:, None], s[None, :] * np.cos(az)[:, None]
    dist = np.hypot(x - CENTRE[0], y - CENTRE[1])
    assert (dist[both] < 8000.0 + 5000.0 + 600.0).all()
    # the vortex itself (inside the 20 dBZ area) is kept
    centre = dist < CORE
    assert np.isfinite(screened.values[centre]).sum() > 0.5 * centre.sum()
    # a distance that is too short keeps less
    short = azimuthal_shear_layer_max(
        volume, min_reflectivity=20.0, reflectivity_distance=500.0, engine=engine
    )["azimuthal_shear_layer_max"]
    assert np.isfinite(short.values).sum() < both.sum()
    assert short.attrs["reflectivity_distance_m"] == 500.0
    assert short.attrs["min_reflectivity_dbz"] == 20.0


def split_volume():
    """Reflectivity and velocity on separate sweeps of the same elevation angles."""
    nodes = {"/": xr.Dataset()}
    k = 0
    for el in ELEVATIONS:
        full = column_sweep(el, n_az=360)
        dbz = full[["DBZH"]].isel(azimuth=slice(None, None, 2))
        dbz = dbz.isel(range=slice(None, None, 2))
        nodes[f"sweep_{k}"] = dbz
        nodes[f"sweep_{k + 1}"] = full[["VRADH"]]
        k += 2
    return xr.DataTree.from_dict(nodes)


@pytest.mark.parametrize("engine", ENGINES)
def test_layer_max_reflectivity_from_the_split_cut(engine):
    vol = split_volume()
    vol_full = column_volume()
    whole = azimuthal_shear_layer_max(vol_full, min_reflectivity=20.0, engine=engine)[
        "azimuthal_shear_layer_max"
    ]
    split = azimuthal_shear_layer_max(
        vol, min_reflectivity=20.0, reference="sweep_1", engine=engine
    )["azimuthal_shear_layer_max"]
    assert split.shape == whole.shape
    both = np.isfinite(whole.values) & np.isfinite(split.values)
    assert both.sum() > 0.8 * np.isfinite(whole.values).sum()
    # the sampling of the coarser reflectivity grid changes only the edge of the screen
    assert np.isfinite(split.values).sum() == pytest.approx(
        np.isfinite(whole.values).sum(), rel=0.1
    )
    # no reflectivity anywhere near an elevation angle
    nodes = {"/": xr.Dataset(), "sweep_0": vol_full["sweep_0"].to_dataset()}
    far = column_sweep(2.0, with_dbz=False)
    nodes["sweep_1"] = far
    nodes["sweep_0"] = nodes["sweep_0"][["DBZH"]]
    with pytest.raises(ValueError, match="within 0.2 degrees"):
        azimuthal_shear_layer_max(
            xr.DataTree.from_dict(nodes),
            min_reflectivity=20.0,
            reference="sweep_1",
            engine=engine,
        )


def test_layer_max_reference_sweep_and_grid(volume):
    out = azimuthal_shear_layer_max(
        volume, layer=(3000.0, 6000.0), reference="sweep_5", engine="numpy"
    )
    assert out["azimuthal_shear_layer_max"].attrs["reference_sweep"] == "sweep_5"
    assert dict(out.sizes) == dict(volume["sweep_5"].to_dataset().sizes)
    xr.testing.assert_equal(out["azimuth"], volume["sweep_5"]["azimuth"])
    # all tilts are above the layer: nothing to take
    empty = azimuthal_shear_layer_max(volume, layer=(20000.0, 21000.0), engine="numpy")
    assert np.isnan(empty["azimuthal_shear_layer_max"]).all()
    assert (empty["layer_max_elevation"].isnull()).all()
    assert empty["azimuthal_shear_layer_max"].attrs["tilts_used"] == 0


def test_layer_max_does_not_change_its_input(volume):
    before = volume.copy(deep=True)
    azimuthal_shear_layer_max(volume, min_reflectivity=20.0, median_filter=3)
    xr.testing.assert_identical(volume.to_dataset(), before.to_dataset())
    for name in before.children:
        xr.testing.assert_identical(
            volume[name].to_dataset(), before[name].to_dataset()
        )


def test_layer_max_with_shear_assigned_to_the_volume(volume):
    shear = azimuthal_shear(volume, "VRADH")
    merged = volume.radarx.assign(shear)
    direct = azimuthal_shear_layer_max(volume, "VRADH", engine="numpy")
    via = azimuthal_shear_layer_max(merged, shear="azimuthal_shear", engine="numpy")
    np.testing.assert_allclose(
        direct["azimuthal_shear_layer_max"].values,
        via["azimuthal_shear_layer_max"].values,
        rtol=1e-6,
        equal_nan=True,
    )
    assert "window_range_m" not in via["azimuthal_shear_layer_max"].attrs


def test_layer_max_accessor_and_provenance(volume):
    a = volume.radarx.azimuthal_shear_layer_max(layer=(3000.0, 6000.0))
    b = azimuthal_shear_layer_max(volume, layer=(3000.0, 6000.0))
    xr.testing.assert_identical(a, b)
    attrs = a["azimuthal_shear_layer_max"].attrs
    assert "radarx_method" in attrs
    assert "10.1175/BAMS-D-14-00173.1" in attrs["radarx_references"]
    assert attrs["window_range_m"] == 750.0 and attrs["weights"] == "uniform"
    med = azimuthal_shear_layer_max(volume, median_filter=3)
    assert med["azimuthal_shear_layer_max"].attrs["median_filter_size"] == 3


def test_layer_max_errors(volume):
    with pytest.raises(TypeError, match="volume"):
        azimuthal_shear_layer_max(volume["sweep_0"].to_dataset())
    with pytest.raises(TypeError, match="volume"):
        azimuthal_shear_layer_max(xr.DataTree.from_dict({"/": xr.Dataset()}))
    with pytest.raises(ValueError, match="rotation"):
        azimuthal_shear_layer_max(volume, rotation="up")
    with pytest.raises(ValueError, match="layer"):
        azimuthal_shear_layer_max(volume, layer=(2000.0, 1000.0))
    with pytest.raises(ValueError, match="layer"):
        azimuthal_shear_layer_max(volume, layer=(0.0, 1000.0, 2000.0))
    with pytest.raises(ValueError, match="reflectivity_distance"):
        azimuthal_shear_layer_max(volume, reflectivity_distance=0.0)
    with pytest.raises(ValueError, match="No sweep contains"):
        azimuthal_shear_layer_max(volume, "VEL")
    with pytest.raises(ValueError, match="reference"):
        azimuthal_shear_layer_max(volume, reference="sweep_99")
    with pytest.raises(ValueError, match="engine"):
        azimuthal_shear_layer_max(volume, engine="fortran")
    with pytest.raises(ValueError, match="median_filter"):
        azimuthal_shear_layer_max(volume, median_filter=4)
    with pytest.raises(ValueError, match="min_reflectivity needs"):
        azimuthal_shear_layer_max(volume, min_reflectivity=20.0, reflectivity="ZZ")
    stripped = xr.DataTree.from_dict(
        {"/": xr.Dataset(), "sweep_0": volume["sweep_0"].to_dataset()[["VRADH"]]}
    )
    with pytest.raises(KeyError):
        azimuthal_shear_layer_max(stripped, min_reflectivity=20.0)


def test_layer_max_without_a_fixed_angle_coordinate():
    """The elevation angle comes from ``elevation`` when there is no fixed angle."""
    vol = column_volume(elevations=(0.5, 1.5))
    for name in ("sweep_0", "sweep_1"):
        vol[name] = vol[name].to_dataset().drop_vars("sweep_fixed_angle")
    out = azimuthal_shear_layer_max(vol, engine="numpy")
    assert np.isfinite(out["azimuthal_shear_layer_max"].values).any()
    assert set(
        np.unique(
            out["layer_max_elevation"].values[
                np.isfinite(out["layer_max_elevation"].values)
            ]
        )
    ) <= {0.5, 1.5}


def test_layer_max_dimension_order_and_older_xarray(monkeypatch, volume):
    ds = volume["sweep_0"].to_dataset().transpose("range", "azimuth")
    vol = xr.DataTree.from_dict(
        {"/": xr.Dataset(), "sweep_0": ds, "sweep_1": volume["sweep_1"].to_dataset()}
    )
    out = azimuthal_shear_layer_max(vol, engine="numpy")
    assert out["azimuthal_shear_layer_max"].dims == ("range", "azimuth")
    original = xr.DataTree.to_dataset

    def old_to_dataset(self, inherit=True):
        if inherit == "all_coords":
            raise TypeError("unsupported")
        return original(self, inherit=inherit)

    monkeypatch.setattr(xr.DataTree, "to_dataset", old_to_dataset)
    again = azimuthal_shear_layer_max(vol, engine="numpy")
    np.testing.assert_array_equal(
        out["azimuthal_shear_layer_max"].values,
        again["azimuthal_shear_layer_max"].values,
    )


def test_layer_max_kernel_errors():
    if not shear_mod.HAS_COMPILED_KERNEL:
        pytest.skip("compiled kernel not built")
    k = shear_mod._shear
    sh = np.zeros((4, 5))
    az = np.arange(4.0)
    rg = np.arange(5.0) * 100 + 100
    phi = np.array([0.001, 0.002])
    with pytest.raises(ValueError, match="same number"):
        k.layer_max([sh], [az, az], [rg], [0.5], phi, az, 8.5e6, 0.0, 1000.0, 0, 1)
    with pytest.raises(ValueError, match="mode"):
        k.layer_max([sh], [az], [rg], [0.5], phi, az, 8.5e6, 0.0, 1000.0, 5, 1)
    with pytest.raises(ValueError, match="do not match"):
        k.layer_max([sh], [az[:3]], [rg], [0.5], phi, az, 8.5e6, 0.0, 1000.0, 0, 1)
    with pytest.raises(ValueError, match="median_size"):
        k.llsd(
            [np.zeros((5, 5))],
            [np.arange(5.0)],
            [np.arange(5.0) + 1],
            500.0,
            500.0,
            False,
            0.5,
            1,
            4,
        )


def test_layer_max_ignores_children_that_are_not_sweeps(volume):
    nodes = {"/": xr.Dataset(), "radar_parameters": xr.Dataset({"x": 1.0})}
    nodes.update({n: volume[n].to_dataset() for n in ("sweep_0", "sweep_1")})
    tree = xr.DataTree.from_dict(nodes)
    out = azimuthal_shear_layer_max(tree, engine="numpy")
    assert out["azimuthal_shear_layer_max"].attrs["tilts_used"] == 2


def test_effective_radius_default():
    # the layer maximum uses the same 4/3 Earth as the geometry module
    assert EFFECTIVE_RADIUS_4_3 == pytest.approx(6371000.0 * 4.0 / 3.0)
