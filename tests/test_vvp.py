#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for volume velocity processing (VVP)
==========================================

The truth is a wind that is linear in position, projected on the beams of a
radar with the geometry of the module (the gate positions are checked against
xradar). The fit returns a linear wind exactly, the compiled kernel and the
NumPy implementation agree, the standard errors of Eq. 9.18 of Doviak and
Zrnic (1993) are the scatter of the estimates in a Monte Carlo experiment, and
their worked example of the divergence of a prestorm boundary layer (section
9.3.4) is reproduced.
"""

import numpy as np
import pytest
import xarray as xr
from xradar.georeference import antenna_to_cartesian

import radarx  # noqa: F401
from radarx._provenance import HISTORY, METHOD, REFERENCES, load_registry
from radarx.fundamentals.constants import EARTH_RADIUS
from radarx.fundamentals.geometry import beam_center_height, ground_range
from radarx.retrieve import vvp, vvp_winds

HAS_KERNEL = vvp.HAS_COMPILED_KERNEL
ENGINES = ["numpy"] + (["compiled"] if HAS_KERNEL else [])
compiled_only = pytest.mark.skipif(not HAS_KERNEL, reason="compiled kernel not built")

ALTITUDE = 100.0
ELEVATIONS = (0.5, 1.5, 3.0, 5.0, 8.0)
LINEAR = {
    "u0": 5.0,
    "ux": 1.0e-4,
    "uy": 2.0e-4,
    "uz": 2.0e-3,
    "v0": -3.0,
    "vx": -0.5e-4,
    "vy": 3.0e-4,
    "vz": -1.0e-3,
    "w0": 0.5,
    "wx": 2.0e-5,
    "wy": -1.0e-5,
    "wz": 4.0e-4,
}


def sweep(elevation, azimuth_step=1.0, max_range=60000.0, range_step=500.0):
    az = np.arange(azimuth_step / 2, 360.0, azimuth_step)
    rng = np.arange(range_step, max_range + 1.0, range_step)
    return xr.Dataset(
        {"VRADH": (("azimuth", "range"), np.zeros((az.size, rng.size)))},
        coords={
            "azimuth": az,
            "range": rng,
            "elevation": ("azimuth", np.full(az.size, float(elevation))),
            "altitude": ALTITUDE,
            "latitude": 33.9,
            "longitude": -88.3,
        },
    )


def wind_of(p):
    """The linear wind about the radar (x, y and the height above sea level)."""

    def wind(x, y, z):
        return (
            p["u0"] + p["ux"] * x + p["uy"] * y + p["uz"] * z,
            p["v0"] + p["vx"] * x + p["vy"] * y + p["vz"] * z,
            p["w0"] + p["wx"] * x + p["wy"] * y + p["wz"] * z,
        )

    return wind


def observed(ds, wind, noise=0.0, fall=0.0, seed=0):
    """The radial velocity of ``wind`` on the gates of the sweep ``ds``."""
    g, _ = vvp._sweep_gates(ds, None, None, 4.0 / 3.0, None, (0.0, 90.0))
    u, v, w = wind(g["x"], g["y"], g["z"])
    vr = u * g["c"] * g["sa"] + v * g["c"] * g["ca"] + (w - fall) * g["sn"]
    if noise:
        vr = vr + noise * np.random.default_rng(seed).standard_normal(vr.size)
    out = ds.copy()
    out["VRADH"] = (("azimuth", "range"), vr.reshape(ds["VRADH"].shape))
    return out


def volume(wind=None, elevations=ELEVATIONS, **kwargs):
    wind = wind or wind_of(LINEAR)
    nodes = {
        f"/sweep_{i}": observed(sweep(e), wind, **kwargs)
        for i, e in enumerate(elevations)
    }
    root = xr.Dataset(
        coords={"latitude": 33.9, "longitude": -88.3, "altitude": ALTITUDE}
    )
    return xr.DataTree.from_dict({"/": root, **nodes})


CENTRES = dict(x=[-20000.0, 0.0, 25000.0], y=[30000.0, 5000.0], z=[1500.0, 3000.0])
SMALL = dict(radius=15000.0, depth=2000.0, min_gates=200, **CENTRES)


def truth(out, p=LINEAR):
    zz, yy, xx = np.meshgrid(
        out["z"].values, out["y"].values, out["x"].values, indexing="ij"
    )
    u = p["u0"] + p["ux"] * xx + p["uy"] * yy + p["uz"] * zz
    v = p["v0"] + p["vx"] * xx + p["vy"] * yy + p["vz"] * zz
    zeta = p["vx"] - p["uy"]
    # modified components (Eq. 9.11): u0' = u0 + y0 zeta / 2, v0' = v0 - x0 zeta / 2
    return u + 0.5 * yy * zeta, v - 0.5 * xx * zeta, xx, yy, zz


# --------------------------------------------------------------------------
# the geometry of the gates
# --------------------------------------------------------------------------


def test_gates_match_xradar_and_the_local_elevation():
    ds = sweep(2.0)
    g, site = vvp._sweep_gates(ds, None, None, 4.0 / 3.0, None, (0.0, 90.0))
    az = ds["azimuth"].values
    rng = ds["range"].values
    x, y, z = antenna_to_cartesian(
        rng[None, :],
        az[:, None],
        2.0,
        site_altitude=ALTITUDE,
        effective_radius_fraction=4.0 / 3.0,
    )
    assert np.allclose(g["x"], x.ravel(), atol=1e-6, rtol=0)
    assert np.allclose(g["y"], y.ravel(), atol=1e-6, rtol=0)
    assert np.allclose(g["z"], np.broadcast_to(z, x.shape).ravel(), atol=1e-6, rtol=0)
    assert site["altitude"] == ALTITUDE
    # the ray is straight in the effective Earth: the angle between it and the
    # local horizontal at the gate is the elevation plus the central angle
    reff = 4.0 / 3.0 * EARTH_RADIUS
    el = np.deg2rad(2.0)
    point = np.stack([rng * np.cos(el), reff + ALTITUDE + rng * np.sin(el)])
    n = point / np.hypot(*point)
    local = np.arcsin(np.array([np.cos(el), np.sin(el)]) @ n)
    first_ray = g["sn"][: rng.size]
    assert np.allclose(first_ray, np.sin(local), atol=1e-9)
    s = ground_range(rng, 2.0, reff=reff, radar_height=ALTITUDE)
    assert np.allclose(np.arcsin(first_ray), np.deg2rad(2.0) + s / reff)
    assert np.allclose(
        g["z"][: rng.size], beam_center_height(rng, 2.0, ALTITUDE, reff=reff)
    )


# --------------------------------------------------------------------------
# a linear wind is returned exactly
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_full_model_returns_a_linear_wind(engine):
    out = vvp_winds(volume(), model="full", engine=engine, **SMALL)
    assert out["u"].dims == ("z", "y", "x")
    assert out.sizes == {"z": 2, "y": 2, "x": 3}
    u, v, xx, yy, zz = truth(out)
    assert np.isfinite(out["u"]).all()
    assert np.allclose(out["u"], u, atol=1e-4)
    assert np.allclose(out["v"], v, atol=1e-4)
    assert np.allclose(
        out["w"],
        LINEAR["w0"] + LINEAR["wx"] * xx + LINEAR["wy"] * yy + LINEAR["wz"] * zz,
        atol=1e-4,
    )
    for name, key in (
        ("du_dx", "ux"),
        ("dv_dy", "vy"),
        ("du_dz", "uz"),
        ("dv_dz", "vz"),
        ("dw_dx", "wx"),
        ("dw_dy", "wy"),
        ("dw_dz", "wz"),
    ):
        assert np.allclose(out[name], LINEAR[key], atol=2e-8), name
    assert np.allclose(out["shear"], LINEAR["uy"] + LINEAR["vx"], atol=2e-8)
    assert np.allclose(out["divergence"], LINEAR["ux"] + LINEAR["vy"], atol=2e-8)
    assert np.allclose(
        out["stretching_deformation"], LINEAR["ux"] - LINEAR["vy"], atol=2e-8
    )
    assert np.allclose(
        out["deformation"],
        np.hypot(LINEAR["ux"] - LINEAR["vy"], LINEAR["uy"] + LINEAR["vx"]),
        atol=2e-8,
    )
    assert np.allclose(out["vvp_rms"], 0.0, atol=1e-4)
    assert (out["vvp_gates"] >= 200).all()
    assert np.allclose(out["divergence_error"], 0.0, atol=1e-7)


@pytest.mark.parametrize("engine", ENGINES)
def test_models_fit_the_terms_they_have(engine):
    # a wind that the model contains is returned exactly; the terms the model
    # omits are not in the result
    p = dict(LINEAR, w0=0.0, wx=0.0, wy=0.0, wz=0.0)
    vol = volume(wind_of(p))
    linear = vvp_winds(vol, model="linear", engine=engine, **SMALL)
    assert "w" not in linear and "dw_dz" not in linear
    assert np.allclose(linear["divergence"], p["ux"] + p["vy"], atol=2e-8)
    u, v, *_ = truth(linear, p)
    assert np.allclose(linear["u"], u, atol=1e-4)
    p2 = dict(p, wz=3.0e-4, w0=0.7)
    lw = vvp_winds(volume(wind_of(p2)), model="linear_w", engine=engine, **SMALL)
    assert np.allclose(lw["dw_dz"], 3.0e-4, atol=2e-8)
    assert "dw_dx" not in lw
    # a uniform wind: one fit of two unknowns
    uni = vvp_winds(
        volume(wind_of(dict.fromkeys(LINEAR, 0.0) | {"u0": 7.0, "v0": -2.0})),
        model="uniform",
        engine=engine,
        **SMALL,
    )
    assert np.allclose(uni["u"], 7.0, atol=1e-4) and np.allclose(
        uni["v"], -2.0, atol=1e-4
    )
    assert "divergence" not in uni and "du_dx" not in uni
    assert uni.attrs["vvp_model"] == "uniform"


@pytest.mark.parametrize("engine", ENGINES)
def test_vorticity_option_gives_the_wind_at_the_centre(engine):
    vol = volume()
    plain = vvp_winds(vol, model="full", engine=engine, **SMALL)
    zeta = LINEAR["vx"] - LINEAR["uy"]
    fixed = vvp_winds(vol, model="full", vorticity=zeta, engine=engine, **SMALL)
    zz, yy, xx = np.meshgrid(fixed["z"], fixed["y"], fixed["x"], indexing="ij")
    u0 = LINEAR["u0"] + LINEAR["ux"] * xx + LINEAR["uy"] * yy + LINEAR["uz"] * zz
    v0 = LINEAR["v0"] + LINEAR["vx"] * xx + LINEAR["vy"] * yy + LINEAR["vz"] * zz
    assert np.allclose(fixed["u"], u0, atol=1e-4)
    assert np.allclose(fixed["v"], v0, atol=1e-4)
    # without it the modified components differ by y0 zeta / 2 and x0 zeta / 2
    assert np.allclose(plain["u"] - fixed["u"], 0.5 * yy * zeta, atol=1e-4)
    assert np.allclose(plain["v"] - fixed["v"], -0.5 * xx * zeta, atol=1e-4)


@pytest.mark.parametrize("compiled", [False] + ([True] if HAS_KERNEL else []))
def test_volume_without_vertical_extent_is_not_fitted(compiled):
    # gates at one height do not determine the vertical shear: the normal
    # equations are singular and the volume is NaN, while a model without the
    # vertical terms is fitted
    gates = sector(30000.0, 60000.0, 0.0)
    gates["z"] = np.full_like(gates["z"], 800.0)
    gates["vr"] = 4.0 * gates["c"] * gates["sa"] + gates["c"] * gates["ca"]
    centre = [np.array([gates[k].mean()]) for k in ("x", "y", "z")]
    for mask, expected in ((0b1001, 4.0), (0b1111111, np.nan)):
        coef, err, extra = vvp._fit(
            gates, *centre, 40000.0, 20000.0, mask, 100, compiled, 1
        )
        assert np.allclose(coef[0, 0], expected, atol=1e-6, equal_nan=True)
        assert np.isnan(extra[0, 2]) == np.isnan(expected)
    # too few gates
    coef, _, extra = vvp._fit(
        gates, *centre, 40000.0, 20000.0, 0b1001, 10**6, compiled, 1
    )
    assert np.isnan(coef).all() and np.isnan(extra).all()


def test_nan_gates_and_empty_volumes():
    vol = volume()
    node = vol["sweep_0"].to_dataset()
    node["VRADH"] = node["VRADH"].where(node["azimuth"] > 180.0)
    vol["sweep_0"] = xr.DataTree(node)
    out = vvp_winds(vol, model="uniform", **SMALL)
    assert np.isfinite(out["u"]).any()
    # a centre far outside the data has no gates
    far = vvp_winds(
        vol, model="uniform", radius=5000.0, depth=500.0, x=[1.0e6], y=[0.0], z=[1000.0]
    )
    assert np.isnan(far["u"]).all() and (far["vvp_gates"] == 0).all()


# --------------------------------------------------------------------------
# engines
# --------------------------------------------------------------------------


@compiled_only
@pytest.mark.parametrize("model", ["uniform", "linear", "linear_w", "full"])
def test_engines_agree(model):
    vol = volume(noise=1.0, seed=3)
    a = vvp_winds(vol, model=model, engine="compiled", n_threads=3, **SMALL)
    b = vvp_winds(vol, model=model, engine="numpy", **SMALL)
    assert set(a.data_vars) == set(b.data_vars)
    for name in a.data_vars:
        np.testing.assert_allclose(
            a[name].values.astype(float),
            b[name].values.astype(float),
            rtol=2e-4,
            atol=1e-7,
            equal_nan=True,
            err_msg=name,
        )


@compiled_only
def test_engines_agree_on_default_grid_and_threads():
    vol = volume(noise=0.5, elevations=(0.5, 2.0, 4.0))
    one = vvp_winds(vol, engine="compiled", n_threads=1, radius=20000.0, min_gates=300)
    many = vvp_winds(vol, engine="compiled", n_threads=4, radius=20000.0, min_gates=300)
    ref = vvp_winds(vol, engine="numpy", radius=20000.0, min_gates=300)
    for name in ("u", "v", "divergence", "vvp_rms"):
        np.testing.assert_array_equal(one[name].values, many[name].values)
        np.testing.assert_allclose(
            one[name], ref[name], rtol=2e-4, atol=1e-7, equal_nan=True
        )
    assert np.isfinite(one["u"]).sum() > 10


@compiled_only
def test_kernel_rejects_bad_input():
    k = vvp._vvp
    a = np.zeros(4)
    c = np.ones(3)
    args = (a, a, a, a, a, a, a, a)
    with pytest.raises(ValueError, match="same length"):
        k.vvp_fit(a, a, a, a, a, a, a, c, c, c, c, 1000.0, 500.0, 3, 1, 1)
    with pytest.raises(ValueError, match="x0, y0 and z0"):
        k.vvp_fit(*args, c, c, a[:2], 1000.0, 500.0, 3, 1, 1)
    with pytest.raises(ValueError, match="positive"):
        k.vvp_fit(*args, c, c, c, 0.0, 500.0, 3, 1, 1)
    with pytest.raises(ValueError, match="mask"):
        k.vvp_fit(*args, c, c, c, 1000.0, 500.0, 0, 1, 1)
    with pytest.raises(ValueError, match="too small"):
        big = np.array([0.0, 1.0e6])
        k.vvp_fit(big, big, big, big, big, big, big, big, c, c, c, 1.0, 1.0, 3, 1, 1)


# --------------------------------------------------------------------------
# errors: Eq. 9.18 and the example of section 9.3.4
# --------------------------------------------------------------------------


def sector(r0, r1, azimuth, width=30.0, elevations=(0.4, 0.8), dr=450.0, daz=1.0):
    """Gates of the analysis sector of the example of Doviak and Zrnic (1993)."""
    reff = 4.0 / 3.0 * EARTH_RADIUS
    az = np.arange(azimuth - width / 2 + daz / 2, azimuth + width / 2, daz)
    rng = np.arange(r0, r1, dr)
    parts = []
    for e in elevations:
        s = ground_range(rng, e, reff=reff)
        z = beam_center_height(rng, e, reff=reff)
        theta = np.radians(e) + s / reff
        a = np.radians(az)[:, None]
        shape = (az.size, rng.size)
        parts.append(
            {
                "x": (s * np.sin(a)).ravel(),
                "y": (s * np.cos(a)).ravel(),
                "z": np.broadcast_to(z, shape).ravel(),
                "c": np.broadcast_to(np.cos(theta), shape).ravel(),
                "sn": np.broadcast_to(np.sin(theta), shape).ravel(),
                "sa": np.broadcast_to(np.sin(a), shape).ravel(),
                "ca": np.broadcast_to(np.cos(a), shape).ravel(),
            }
        )
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def divergence_error(gates, sigma=1.0, indices=(0, 1, 2, 3, 4, 5, 6)):
    """sigma sqrt(C22 + C55 + 2 C25) of Eq. 9.18 for the seven-parameter model."""
    centre = [gates[k].mean() for k in ("x", "y", "z")]
    p = vvp._predictors(gates, *centre, 1.0, 1.0)[:, list(indices)]
    cov = np.linalg.inv(p.T @ p)
    a, b = indices.index(1), indices.index(4)
    return sigma * np.sqrt(cov[a, a] + cov[b, b] + 2.0 * cov[a, b])


def test_divergence_error_of_the_prestorm_example():
    # section 9.3.4: a sector of 30 degrees and 30 km, the sweeps at 0.4 and 0.8
    # degrees, 1 degree and 450 m spacing (about 4000 gates) and an error of 1 m/s
    # of the radial velocity give a standard error of the divergence of
    # 1.6e-5 1/s (the book does not give the range of the sector: the value is
    # reached for a sector that starts between 30 and 40 km)
    errors = {
        r0: divergence_error(sector(r0, r0 + 30000.0, 0.0)) for r0 in (30000.0, 40000.0)
    }
    assert 3900 < sector(30000.0, 60000.0, 0.0)["x"].size < 4100
    assert errors[30000.0] > 1.6e-5 > errors[40000.0]
    assert abs(errors[30000.0] / 1.6e-5 - 1.0) < 0.1
    # the error does not depend on the azimuth of the sector and decreases with range
    assert np.isclose(
        divergence_error(sector(30000.0, 60000.0, 100.0)), errors[30000.0], rtol=1e-6
    )
    assert divergence_error(sector(60000.0, 90000.0, 0.0)) < errors[40000.0]


def test_standard_errors_are_the_scatter_of_the_estimates():
    gates = sector(30000.0, 60000.0, 20.0)
    centre = np.array([gates[k].mean() for k in ("x", "y", "z")])
    wind = wind_of(LINEAR)
    u, v, w = wind(gates["x"], gates["y"], gates["z"])
    vr = u * gates["c"] * gates["sa"] + v * gates["c"] * gates["ca"]
    sigma = 1.0
    rng = np.random.default_rng(11)
    mask = sum(1 << k for k in (0, 1, 2, 3, 4, 5, 6))
    radius, depth = 40000.0, 20000.0
    ests, errs, derr = [], [], []
    for _ in range(300):
        g = dict(gates, vr=vr + sigma * rng.standard_normal(vr.size))
        c, e, x = vvp._fit(
            g, centre[:1], centre[1:2], centre[2:], radius, depth, mask, 100, False, 1
        )
        ests.append(c[0])
        errs.append(e[0])
        derr.append(x[0, 0])
    ests, errs = np.array(ests), np.array(errs)
    for k in (1, 4, 6):
        scatter = ests[:, k].std(ddof=1)
        assert np.isclose(scatter, np.mean(errs[:, k]), rtol=0.15), k
    div = ests[:, 1] + ests[:, 4]
    assert np.isclose(div.std(ddof=1), np.mean(derr), rtol=0.15)
    # and Eq. 9.18 with the known sigma
    assert np.isclose(np.mean(derr), divergence_error(gates, sigma), rtol=0.03)


@compiled_only
def test_kernel_errors_match_numpy_errors():
    gates = sector(30000.0, 60000.0, 20.0)
    centre = [np.array([gates[k].mean()]) for k in ("x", "y", "z")]
    gates["vr"] = np.random.default_rng(2).standard_normal(gates["x"].size)
    mask = sum(1 << k for k in range(11)) & ~(1 << 7)
    a = vvp._fit(gates, *centre, 40000.0, 20000.0, mask, 100, True, 1)
    b = vvp._fit(gates, *centre, 40000.0, 20000.0, mask, 100, False, 1)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-12, equal_nan=True)


# --------------------------------------------------------------------------
# fall speed, selection, inputs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_fall_speed_is_removed(engine):
    p = dict(LINEAR, w0=0.5, wx=0.0, wy=0.0, wz=0.0)
    vol = volume(wind_of(p), fall=3.0)
    kw = dict(model="linear_w", engine=engine, **SMALL)
    with_fall = vvp_winds(vol, fall_speed=3.0, **kw)
    without = vvp_winds(vol, **kw)
    assert np.allclose(with_fall["w"], 0.5, atol=1e-3)
    assert np.allclose(without["w"], -2.5, atol=1e-3)
    # a field of the fall speed
    named = vol.copy(deep=True)
    for name in ("sweep_0", "sweep_1", "sweep_2", "sweep_3", "sweep_4"):
        node = named[name].to_dataset()
        node["VT"] = xr.full_like(node["VRADH"], 3.0)
        named[name] = xr.DataTree(node)
    assert np.allclose(vvp_winds(named, fall_speed="VT", **kw)["w"], 0.5, atol=1e-3)
    with pytest.raises(KeyError, match="fall speed"):
        vvp_winds(vol, fall_speed="nothing", **kw)


def test_selection_by_elevation_range_and_field():
    vol = volume()
    low = vvp_winds(vol, max_elevation=2.0, model="uniform", **SMALL)
    allv = vvp_winds(vol, model="uniform", **SMALL)
    assert low["vvp_gates"].sum() < allv["vvp_gates"].sum()
    near = vvp_winds(vol, max_range=30000.0, model="uniform", **SMALL)
    assert near["vvp_gates"].sum() < allv["vvp_gates"].sum()
    with pytest.raises(KeyError, match="elevation range"):
        vvp_winds(vol, min_elevation=20.0, model="uniform")
    renamed = vol.copy(deep=True)
    for name in list(renamed.children):
        node = renamed[name].to_dataset().rename({"VRADH": "my_velocity"})
        renamed[name] = xr.DataTree(node)
    with pytest.raises(KeyError, match="radial velocity"):
        vvp_winds(renamed, model="uniform")
    named = vvp_winds(renamed, field="my_velocity", model="uniform", **SMALL)
    assert np.allclose(
        named["u"], vvp_winds(vol, model="uniform", **SMALL)["u"], equal_nan=True
    )
    with pytest.raises(KeyError, match="nothing"):
        vvp_winds(vol, field="nothing", model="uniform")
    # sweeps without the field are skipped
    mixed = vol.copy(deep=True)
    mixed["sweep_0"] = xr.DataTree(mixed["sweep_0"].to_dataset().drop_vars("VRADH"))
    skipped = vvp_winds(mixed, model="uniform", **SMALL)
    assert skipped["vvp_gates"].sum() < allv["vvp_gates"].sum()


def test_sweep_node_dataset_and_default_grid():
    vol = volume(elevations=(1.0, 2.0))
    ds = vol["sweep_0"].to_dataset(inherit="all_coords")
    from_node = vvp_winds(
        vol["sweep_0"], model="uniform", radius=20000.0, min_gates=300
    )
    from_ds = vvp_winds(ds, model="uniform", radius=20000.0, min_gates=300)
    xr.testing.assert_equal(from_node["u"], from_ds["u"])
    # the default grid covers the gates with the spacing of the radius
    assert np.allclose(np.diff(from_ds["x"]), 20000.0)
    assert from_ds["x"].min() <= -50000.0 and from_ds["x"].max() >= 50000.0
    assert np.allclose(np.diff(from_ds["z"]), 1000.0)
    assert float(from_ds["latitude"]) == 33.9 and float(from_ds["altitude"]) == ALTITUDE


def test_input_errors():
    vol = volume(elevations=(1.0, 2.0))
    ds = vol["sweep_0"].to_dataset(inherit="all_coords")
    with pytest.raises(TypeError, match="xarray"):
        vvp_winds(np.zeros((3, 3)))
    with pytest.raises(TypeError, match="Dataset sweep"):
        vvp_winds(ds["VRADH"])
    with pytest.raises(ValueError, match="model"):
        vvp_winds(ds, model="quadratic")
    for bad in ({"radius": 0.0}, {"depth": -1.0}, {"k_e": 0.0}, {"radius": np.nan}):
        with pytest.raises(ValueError, match="positive"):
            vvp_winds(ds, **bad)
    with pytest.raises(ValueError, match="min_gates"):
        vvp_winds(ds, min_gates=0)
    with pytest.raises(ValueError, match="min_elevation"):
        vvp_winds(ds, min_elevation=10.0, max_elevation=5.0)
    with pytest.raises(ValueError, match="engine"):
        vvp_winds(ds, engine="gpu")
    with pytest.raises(ValueError, match="1-D"):
        vvp_winds(ds, x=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        vvp_winds(ds, y=[np.nan])
    with pytest.raises(ValueError, match="azimuth"):
        vvp_winds(
            ds.drop_vars("azimuth").assign(
                azimuth=("range", np.zeros(ds.sizes["range"]))
            )
        )
    with pytest.raises(ValueError, match="elevation"):
        vvp_winds(ds.drop_vars("elevation"))
    with pytest.raises(ValueError, match="one value per ray"):
        vvp_winds(ds.assign_coords(elevation=("range", np.ones(ds.sizes["range"]))))
    fixed = ds.drop_vars("elevation").assign_coords(sweep_fixed_angle=1.0)
    assert np.isfinite(vvp_winds(fixed, model="uniform", **SMALL)["u"]).any()
    empty = ds.copy()
    empty["VRADH"] = empty["VRADH"] * np.nan
    with pytest.raises(ValueError, match="no valid"):
        vvp_winds(empty)
    with pytest.raises(ImportError, match="compiled"):
        old = vvp.HAS_COMPILED_KERNEL
        vvp.HAS_COMPILED_KERNEL = False
        try:
            vvp_winds(ds, engine="compiled")
        finally:
            vvp.HAS_COMPILED_KERNEL = old
    three = ds.expand_dims(time=2)
    with pytest.raises(ValueError, match="two dimensions"):
        vvp_winds(three)


def test_numpy_fallback_without_the_kernel(monkeypatch):
    monkeypatch.setattr(vvp, "HAS_COMPILED_KERNEL", False)
    ds = volume(elevations=(1.0, 3.0))
    out = vvp_winds(ds, model="uniform", **SMALL)
    assert np.isfinite(out["u"]).any()


def test_dask_input_and_inputs_unchanged():
    dask = pytest.importorskip("dask")  # noqa: F841
    ds = observed(sweep(1.0), wind_of(LINEAR))
    before = ds.copy(deep=True)
    lazy = ds.chunk({"azimuth": 90})
    a = vvp_winds(lazy, model="uniform", **SMALL)
    b = vvp_winds(ds, model="uniform", **SMALL)
    xr.testing.assert_allclose(a["u"], b["u"])
    xr.testing.assert_identical(ds, before)


def test_output_conventions_and_provenance():
    out = vvp_winds(volume(), **SMALL)
    assert out["u"].dtype == np.float32 and out["vvp_gates"].dtype == np.int32
    assert out["x"].dtype == np.float64
    assert out["u"].attrs["standard_name"] == "eastward_wind"
    assert out["divergence"].attrs["units"] == "s-1"
    assert out["z"].attrs["standard_name"] == "altitude"
    assert out.attrs[METHOD] and out.attrs[HISTORY].startswith("vvp_winds(")
    refs = out.attrs[REFERENCES].split()
    registry = load_registry()
    assert "10.1175/1520-0450(1979)018<0532:OTAOSD>2.0.CO;2" in refs
    assert all(r in registry for r in refs)


def test_accessor():
    vol = volume(elevations=(1.0, 2.0))
    a = vol.radarx.vvp_winds(model="uniform", **SMALL)
    b = vvp_winds(vol, model="uniform", **SMALL)
    xr.testing.assert_equal(a["u"], b["u"])
    c = (
        vol["sweep_0"]
        .to_dataset(inherit="all_coords")
        .radarx.vvp_winds(model="uniform", **SMALL)
    )
    assert "u" in c


def test_numpy_fit_skips_singular_systems(monkeypatch):
    gates = sector(30000.0, 60000.0, 0.0)
    gates["vr"] = gates["c"] * gates["sa"]
    centre = [np.array([gates[k].mean()]) for k in ("x", "y", "z")]
    mask = 0b1001

    def singular(_):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(np.linalg, "cholesky", singular)
    coef, _, extra = vvp._fit(gates, *centre, 40000.0, 20000.0, mask, 100, False, 1)
    assert np.isnan(coef).all() and np.isnan(extra).all()
    monkeypatch.setattr(np.linalg, "cholesky", lambda a: np.eye(a.shape[0]) * 1e-12)
    coef, _, _ = vvp._fit(gates, *centre, 40000.0, 20000.0, mask, 100, False, 1)
    assert np.isnan(coef).all()


def test_sweep_gates_of_a_missing_field():
    ds = sweep(1.0)
    assert vvp._sweep_gates(ds, "nothing", None, 4.0 / 3.0, None, (0.0, 90.0)) is None
    plain = ds.drop_vars("VRADH")
    assert vvp._sweep_gates(plain, None, None, 4.0 / 3.0, None, (0.0, 90.0)) is None


def test_unknown_radar_altitude_is_zero():
    ds = observed(sweep(1.0), wind_of(LINEAR)).assign_coords(altitude=np.nan)
    out = vvp_winds(ds, model="uniform", **SMALL)
    assert "altitude" not in out.coords and np.isfinite(out["u"]).any()
