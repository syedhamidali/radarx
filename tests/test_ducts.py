#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests of radarx.retrieve.ducts (trapping layers of a refractivity profile)
========================================================================

The angles of penetration are compared with the numbers of Bean and Dutton
(1966, Section 4.4.4: 8.3 and 4.5 mrad for layers of 100 and 30 m with a
gradient of -0.5 N-units per m) and with the rays of
:func:`radarx.fundamentals.trace_rays`, which turn over inside the layer below
the angle at the height that Snell's law gives and leave it above the angle.
"""

import importlib
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from scipy.optimize import brentq

import radarx  # noqa: F401
from radarx.fundamentals import trace_rays
from radarx.fundamentals.constants import EARTH_RADIUS
from radarx.io import sounding
from radarx.retrieve import _ducts_numpy as ref
from radarx.retrieve import ducting_layers, modified_refractivity

dc = importlib.import_module("radarx.retrieve.ducts")
ENGINES = ["numpy"] + (["compiled"] if dc.HAS_COMPILED_KERNEL else [])
DATA = Path(__file__).parent / "io" / "data"
A = EARTH_RADIUS


def layer_profile(thick, gradient, n0=330.0, top=3000.0, above=-0.04):
    """N(h): a layer from the ground to ``thick`` (gradient N per m), then normal."""
    h = np.array([0.0, thick, top])
    n = np.array(
        [n0, n0 + gradient * thick, n0 + gradient * thick + above * (top - thick)]
    )
    return xr.DataArray(n, dims="height", coords={"height": h})


def test_trapping_gradient_is_one_over_the_earth_radius():
    # Eq. 4.23: -157 N-units/km; the modified refractivity is flat at 1e6/a
    h = np.array([0.0, 1000.0])
    edge = -1.0e6 / A  # N per m
    for factor, expected in ((0.99, 0), (1.01, 1)):
        n = xr.DataArray(
            [300.0, 300.0 + factor * edge * 1000.0], dims="height", coords={"height": h}
        )
        assert int(ducting_layers(n).layer_count) == expected
    assert 1.0e9 / A == pytest.approx(156.96, abs=0.01)  # N-units per km


@pytest.mark.parametrize("engine", ENGINES)
def test_angles_of_bean_and_dutton(engine):
    # Section 4.4.4: a ground-based layer of 100 m reaches the gradient of
    # 0.5 N-units per m at 8.3 mrad, one of 30 m at 4.5 mrad
    for thick, mrad in ((100.0, 8.3), (30.0, 4.5)):
        out = ducting_layers(layer_profile(thick, -0.5), engine=engine)
        assert int(out.layer_count) == 1
        theta = float(out.trapping_angle.isel(layer=0))
        assert np.radians(theta) * 1e3 == pytest.approx(mrad, abs=0.05)
        # Eq. 4.22: theta_p ~ sqrt(2 dM 1e-6)
        dm = float(out.m_deficit.isel(layer=0))
        assert dm == pytest.approx(thick * (0.5 - 1.0e6 / A))
        assert np.radians(theta) == pytest.approx(np.sqrt(2.0 * dm * 1e-6), rel=2e-3)
        assert float(out.layer_base.isel(layer=0)) == 0.0
        assert float(out.layer_top.isel(layer=0)) == thick
        assert float(out.layer_thickness.isel(layer=0)) == thick
        assert float(out.surface_based.isel(layer=0)) == 1.0


def turning_height(p, theta_deg, h_start):
    """Height where a ray from ``h_start`` turns over, from Snell's law (analytic)."""
    h = p.height.values
    n = p.values

    def nr(z):
        return (1.0 + 1e-6 * np.interp(z, h, n)) * (A + z)

    c = nr(h_start) * np.cos(np.radians(theta_deg))
    return brentq(lambda z: nr(z) - c, h_start, h[1], xtol=1e-9)


@pytest.mark.parametrize("thick,gradient", [(100.0, -0.5), (30.0, -0.5), (200.0, -0.8)])
def test_rays_below_the_angle_are_trapped_and_above_leave(thick, gradient):
    p = layer_profile(thick, gradient)
    out = ducting_layers(p)
    theta = float(out.trapping_angle.isel(layer=0))
    r = np.arange(0.0, 150000.0, 100.0)
    for f in (0.5, 0.9, 0.99):
        res = trace_rays(
            r, f * theta, p.height.values, p.values, radar_height=0.0, step=50.0
        )
        top = float(res.beam_height.max())
        # the ray turns at the height of Snell's law, below the top of the layer
        assert top == pytest.approx(turning_height(p, f * theta, 0.0), abs=0.05)
        assert top < thick
    for f in (1.01, 1.1):
        res = trace_rays(
            r, f * theta, p.height.values, p.values, radar_height=0.0, step=50.0
        )
        assert float(res.beam_height.max()) > thick + 50.0
        assert float(res.beam_height.values[-1]) > thick + 50.0


def test_radar_inside_the_layer_and_elevated_layers():
    # an elevated layer from 500 to 700 m with a radar inside it at 600 m
    h = np.array([0.0, 500.0, 700.0, 3000.0])
    n = np.array([320.0, 310.0, 250.0, 200.0])
    p = xr.DataArray(n, dims="height", coords={"height": h})
    out = ducting_layers(p, radar_height=600.0)
    assert int(out.layer_count) == 1
    assert float(out.surface_based.isel(layer=0)) == 0.0
    base_angle = float(out.trapping_angle.isel(layer=0))
    radar_angle = float(out.radar_trapping_angle.isel(layer=0))
    assert 0.0 < radar_angle < base_angle  # from higher up the layer, less is trapped
    # the same with the Snell constant: n r cos(theta) at the top
    nr_top = (1 + 250e-6) * (A + 700.0)
    nr_radar = (1 + 1e-6 * np.interp(600.0, h, n)) * (A + 600.0)
    assert np.cos(np.radians(radar_angle)) == pytest.approx(
        nr_top / nr_radar, rel=1e-13
    )
    # a ray from the radar at 0.98 of the angle stays in the layer, 1.02 leaves
    r = np.arange(0.0, 100000.0, 100.0)
    below = trace_rays(r, 0.98 * radar_angle, h, n, radar_height=600.0, step=50.0)
    above = trace_rays(r, 1.02 * radar_angle, h, n, radar_height=600.0, step=50.0)
    assert float(below.beam_height.max()) < 700.0
    assert float(above.beam_height.max()) > 700.0
    # a radar outside the layer: no angle; no radar height: no angle
    assert np.isnan(
        float(ducting_layers(p, radar_height=100.0).radar_trapping_angle.isel(layer=0))
    )
    assert np.isnan(float(out.radar_trapping_angle.isel(layer=1)))
    assert np.isnan(float(ducting_layers(p).radar_trapping_angle.isel(layer=0)))


@pytest.mark.parametrize("engine", ENGINES)
def test_several_layers_order_max_layers_and_min_deficit(engine):
    h = np.arange(0.0, 6001.0, 500.0)
    # trapping layers at 0-500, 1500-2000 (weak) and 3000-3500 (strong)
    n = np.array(
        [
            400.0,
            300.0,
            296.0,
            292.0,
            206.0,
            204.0,
            200.0,
            80.0,
            75.0,
            73.0,
            70.0,
            68.0,
            66.0,
        ]
    )
    p = xr.DataArray(n, dims="height", coords={"height": h})
    mm = n + h * 1e6 / A
    expected = []
    i = 0
    while i < h.size - 1:
        if mm[i + 1] < mm[i]:
            b = i
            while i < h.size - 1 and mm[i + 1] < mm[i]:
                i += 1
            expected.append((h[b], h[i], mm[b] - mm[i]))
        else:
            i += 1
    out = ducting_layers(p, max_layers=8, engine=engine)
    assert int(out.layer_count) == len(expected) == 3
    k = len(expected)
    np.testing.assert_allclose(out.layer_base.values[:k], [e[0] for e in expected])
    np.testing.assert_allclose(out.layer_top.values[:k], [e[1] for e in expected])
    np.testing.assert_allclose(out.m_deficit.values[:k], [e[2] for e in expected])
    assert np.isnan(out.layer_base.values[k:]).all()
    assert np.all(np.diff(out.layer_base.values[:k]) > 0)
    # fewer slots than layers: the lowest are kept, the count is complete
    few = ducting_layers(p, max_layers=1, engine=engine)
    assert int(few.layer_count) == k and few.sizes["layer"] == 1
    assert float(few.layer_base.isel(layer=0)) == expected[0][0]
    # min_deficit leaves out the weak layers
    deficits = np.array([e[2] for e in expected])
    cut = float(np.sort(deficits)[-1]) - 1e-9
    strong = ducting_layers(p, max_layers=8, min_deficit=cut, engine=engine)
    assert int(strong.layer_count) == int((deficits > cut).sum())


@pytest.mark.parametrize("engine", ENGINES)
def test_real_soundings_agree_with_a_direct_search(engine):
    for name, kw in (
        ("uwyo_72235_2022033100.csv", {"station": "72235"}),
        ("iem_KJAN_202203310000.json", {}),
    ):
        ds = sounding.open_sounding_file(DATA / name, **kw)
        out = ducting_layers(ds, engine=engine)
        n = dc._refractivity(ds, "bean_dutton", "auto")
        assert np.isfinite(n.values).sum() > 50
        m = modified_refractivity(ds)
        np.testing.assert_allclose(m.values, n.values + ds.height.values * 1e6 / A)
        ok = np.isfinite(n.values)
        h, mv = ds.height.values[ok], m.values[ok]
        found = np.flatnonzero(np.diff(mv) < 0)
        # runs of consecutive decreasing steps
        runs = (
            np.split(found, np.flatnonzero(np.diff(found) > 1) + 1)
            if found.size
            else []
        )
        assert int(out.layer_count) == len(runs)
        for k, run in enumerate(runs[:4]):
            assert float(out.layer_base.isel(layer=k)) == h[run[0]]
            assert float(out.layer_top.isel(layer=k)) == h[run[-1] + 1]
            assert float(out.m_deficit.isel(layer=k)) == pytest.approx(
                mv[run[0]] - mv[run[-1] + 1]
            )
            nb = 1 + 1e-6 * n.values[ok][run[0]]
            nt = 1 + 1e-6 * n.values[ok][run[-1] + 1]
            cos = nt * (A + h[run[-1] + 1]) / (nb * (A + h[run[0]]))
            assert np.cos(
                np.radians(float(out.trapping_angle.isel(layer=k)))
            ) == pytest.approx(cos, rel=1e-12)


@pytest.mark.skipif(not dc.HAS_COMPILED_KERNEL, reason="compiled kernel not built")
def test_engines_agree_and_threads_do_not_matter_on_many_profiles():
    rng = np.random.default_rng(11)
    h = np.arange(0.0, 4001.0, 100.0)
    base = 320.0 - 0.05 * h
    n = base + np.cumsum(rng.normal(0.0, 6.0, (60, h.size)), axis=1)
    n[3, 5] = np.nan
    n[7, :] = np.nan
    n[9, :] = 300.0  # no layer
    ds = xr.DataArray(
        n, dims=("profile", "height"), coords={"height": h}, name="refractivity"
    )
    a = ducting_layers(
        ds, radar_height=1000.0, max_layers=6, engine="compiled", n_threads=1
    )
    b = ducting_layers(
        ds, radar_height=1000.0, max_layers=6, engine="compiled", n_threads=4
    )
    c = ducting_layers(ds, radar_height=1000.0, max_layers=6, engine="numpy")
    for name in a.data_vars:
        np.testing.assert_array_equal(a[name].values, b[name].values)
        np.testing.assert_allclose(
            a[name].values, c[name].values, rtol=1e-9, atol=0, equal_nan=True
        )
    assert a.layer_base.dims == ("profile", "layer")
    assert int(a.layer_count.max()) > 6 or int(a.layer_count.max()) >= 3
    assert int(a.layer_count.isel(profile=9)) == 0
    assert int(a.layer_count.isel(profile=7)) == 0


def test_dataset_inputs_dask_accessor_and_provenance():
    dask = pytest.importorskip("dask.array")
    ds = sounding.open_sounding_file(DATA / "iem_KJAN_202203310000.json")
    before = ds.copy(deep=True)
    out = ducting_layers(ds)
    xr.testing.assert_identical(ds, before)
    assert out.layer_base.attrs["units"] == "m"
    assert out.trapping_angle.attrs["units"] == "degree"
    assert out.attrs["radarx_references"] == "10.6028/NBS.MONO.92"
    via = ds.radarx.ducting_layers()
    xr.testing.assert_identical(via, out)
    # a batch of profiles, lazily
    n = dc._refractivity(ds, "bean_dutton", "auto")
    batch = xr.concat([n, n * 1.001], dim="time")
    eager = ducting_layers(batch)
    lazy = ducting_layers(batch.chunk({"time": 1}))
    assert lazy.layer_base.chunks is not None
    xr.testing.assert_allclose(lazy.compute(), eager)
    del dask
    # N given by pressure, temperature and vapour pressure equals the refractivity variable
    ref_ds = ds.assign(refractivity=n)
    xr.testing.assert_identical(ducting_layers(ref_ds), out)


def test_modified_refractivity_and_inputs():
    h = np.array([0.0, 1000.0, 2000.0])
    n = xr.DataArray(
        [320.0, 281.0, 250.0], dims="height", coords={"height": h}, attrs={"units": "N"}
    )
    m = modified_refractivity(n)
    np.testing.assert_allclose(m.values, n.values + h * 1e6 / A)
    assert m.attrs["units"] == "1" and m.name == "modified_refractivity"
    assert n.attrs == {"units": "N"}
    # another Earth radius
    m2 = modified_refractivity(n, earth_radius=8.0e6)
    np.testing.assert_allclose(m2.values, n.values + h * 1e6 / 8.0e6)
    with pytest.raises(ValueError, match="earth_radius"):
        modified_refractivity(n, earth_radius=0.0)


def test_errors():
    h = np.array([0.0, 100.0, 1000.0])
    n = xr.DataArray([330.0, 280.0, 250.0], dims="height", coords={"height": h})
    with pytest.raises(TypeError, match="xarray"):
        ducting_layers([1.0, 2.0])
    with pytest.raises(ValueError, match="height"):
        ducting_layers(n.rename(height="z"))
    with pytest.raises(ValueError, match="height. coordinate"):
        ducting_layers(xr.DataArray([1.0, 2.0], dims="height"))
    with pytest.raises(ValueError, match="at least two"):
        ducting_layers(n.isel(height=[0]))
    with pytest.raises(ValueError, match="ascending"):
        ducting_layers(n.isel(height=[2, 1, 0]))
    with pytest.raises(ValueError, match="max_layers"):
        ducting_layers(n, max_layers=0)
    with pytest.raises(ValueError, match="min_deficit"):
        ducting_layers(n, min_deficit=-1.0)
    with pytest.raises(ValueError, match="earth_radius"):
        ducting_layers(n, earth_radius=-1.0)
    with pytest.raises(ValueError, match="engine"):
        ducting_layers(n, engine="gpu")
    with pytest.raises(ValueError, match="pressure"):
        ducting_layers(xr.Dataset({"temperature": n}))
    with pytest.raises(ValueError, match="temperature"):
        ducting_layers(xr.Dataset({"pressure": n}))
    with pytest.raises(ValueError, match="water vapour"):
        ducting_layers(xr.Dataset({"pressure": n * 300.0, "temperature": n}))
    # the profile Dataset needs a height coordinate
    with pytest.raises(ValueError, match="height"):
        ducting_layers(xr.Dataset({"refractivity": ("z", n.values)}))


def test_missing_kernel(monkeypatch):
    h = np.array([0.0, 100.0, 1000.0])
    n = xr.DataArray([330.0, 280.0, 250.0], dims="height", coords={"height": h})
    monkeypatch.setattr(dc, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError, match="not available"):
        ducting_layers(n, engine="compiled")
    assert int(ducting_layers(n).layer_count) == 1


@pytest.mark.skipif(not dc.HAS_COMPILED_KERNEL, reason="compiled kernel not built")
def test_kernel_rejects_bad_arguments():
    k = dc._ducts
    h = np.array([0.0, 100.0, 1000.0])
    n = np.array([[330.0, 280.0, 250.0]])
    fields, count = k.trapping_layers(h, n, np.nan, A, 2, 0.0, 1)
    assert fields.shape == (1, 2, 7) and count.tolist() == [1]
    for args, match in (
        ((h[:, None], n, np.nan, A, 2, 0.0), "dimensions"),
        ((h[:2], n, np.nan, A, 2, 0.0), "levels"),
        ((h, n, np.nan, A, 0, 0.0), "max_layers"),
        ((h, n, np.nan, 0.0, 2, 0.0), "earth_radius"),
    ):
        with pytest.raises(ValueError, match=match):
            k.trapping_layers(*args)


def test_reference_rejects_nothing_and_handles_short_profiles():
    out, count = ref.trapping_layers(
        np.array([0.0, 1.0]), np.array([[np.nan, 300.0]]), np.nan, A, 2, 0.0
    )
    assert count.tolist() == [0] and np.isnan(out).all()
