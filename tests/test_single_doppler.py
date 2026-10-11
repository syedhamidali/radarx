# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the single-Doppler wind retrieval
===========================================

The variational retrieval is checked on an analytic flow (a Beltrami flow,
Shapiro 1993, made anelastic) seen by one virtual radar; the network path
with stand-in models (an object with ``run`` and a tiny random ONNX model),
including tiling, regridding to the network's spacing and the variational
refinement; and the gridding of a real NEXRAD volume.
"""

import os

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.retrieve import single_doppler as sd
from radarx.retrieve.single_doppler import FEATURES, single_doppler_winds

from .test_multidoppler import _nexrad, synthetic_case


@pytest.fixture(scope="module")
def case():
    """Beltrami flow seen by the first of two virtual radars."""
    ds, bg, truth = synthetic_case(step=2000.0, nz=11, noise=0.5, dbz=30.0)
    return ds.isel(radar=[0]), bg, truth


class BackgroundModel:
    """Stand-in network: returns the background wind plus a fixed offset."""

    def __init__(self, offset=(1.0, -2.0, 0.5)):
        self.offset = np.asarray(offset, dtype=np.float32)
        self.calls = []
        self.info = {"name": "stand-in", "version": "7", "licence": "MIT"}

    def run(self, inputs):
        f = inputs["features"]
        self.calls.append(f.shape)
        iu, iv = FEATURES.index("background_u"), FEATURES.index("background_v")
        wind = np.stack(
            [
                f[:, iu] * sd.WIND_SCALE + self.offset[0],
                f[:, iv] * sd.WIND_SCALE + self.offset[1],
                np.full(f[:, 0].shape, self.offset[2]),
            ],
            axis=1,
        )
        return {"wind": wind.astype(np.float32)}


def test_features_layout():
    shape = (3, 4, 5)
    vr = np.full(shape, 15.0)
    vr[0, 0, 0] = np.nan
    coef = np.stack([np.full(shape, c) for c in (0.6, 0.8, 0.0)])
    f = sd._features(
        vr,
        np.full(shape, np.nan),
        coef,
        np.full(shape, 30.0),
        np.zeros(shape),
        np.array([500.0, 1000.0, 1500.0]),
        np.full((4, 5), 75e3),
    )
    assert f.shape == (len(FEATURES),) + shape and f.dtype == np.float32
    assert f[0, 0, 0, 0] == 0 and f[1, 0, 0, 0] == 0 and f[0, 1, 1, 1] == 0.5
    assert not f[3].any() and (f[7] == 1).all() and np.isclose(f[10], 0.5).all()
    np.testing.assert_allclose(f[9][:, 0, 0], [500 / 12000, 1000 / 12000, 0.125])


def test_variational_retrieval(case):
    ds, bg, truth = case
    out = single_doppler_winds(ds, bg)
    assert out.u.dims == ("z", "y", "x")
    assert out.u.attrs["standard_name"] == "x_wind"
    assert out.w.attrs["units"] == "m s-1"
    assert "variational single-Doppler" in out.attrs["method"]
    np.testing.assert_array_equal(out.x, ds.x)
    seen = np.isfinite(ds.VRADH.isel(radar=0).values)
    # the retrieved wind explains the radar's own velocities
    assert float(np.sqrt(np.nanmean(out.vr_residual.values**2))) < 1.0
    # and the flow along the beams; across them it is closer to the truth
    # than the background
    err = np.sqrt(np.mean((out.u.values - truth.u.values)[seen] ** 2))
    err_bg = np.sqrt(np.mean((8.0 - truth.u.values)[seen] ** 2))
    assert err < err_bg


def test_network_without_refinement(case):
    ds, bg, _ = case
    model = BackgroundModel()
    out = single_doppler_winds(ds, bg, model=model, refine=False)
    # the grid has the network's spacing? no: 2 km -> interpolated to 1 km and back
    np.testing.assert_allclose(out.u.values, 9.0, atol=1e-4)
    np.testing.assert_allclose(out.v.values, 2.0, atol=1e-4)
    np.testing.assert_allclose(out.w.values, 0.5, atol=1e-4)
    np.testing.assert_allclose(out.u_network.values, out.u.values)
    assert out.attrs["ml_model"] == "stand-in"
    assert out.attrs["ml_model_version"] == "7"
    assert out.attrs["ml_model_licence"] == "MIT"
    assert out.attrs["method"] == "physics-informed network"
    # inputs reached the network padded to multiples of 4, on its 1 km grid
    for shape in model.calls:
        assert all(n % 4 == 0 for n in shape[2:])
    assert model.calls[0][3] >= 81


def test_network_with_refinement(case):
    ds, bg, truth = case
    out = single_doppler_winds(ds, bg, model=BackgroundModel())
    assert "refined by the variational" in out.attrs["method"]
    assert float(np.sqrt(np.nanmean(out.vr_residual.values**2))) < 1.0
    assert "mass_residual" in out and "u_network" in out
    with pytest.raises(ValueError, match="unknown weights"):
        single_doppler_winds(ds, bg, model=BackgroundModel(), weights={"bad": 1})


def test_network_tiles_and_native_grid():
    """A grid on the network's spacing larger than a tile is run in tiles."""
    x = y = np.arange(0.0, 220e3, 1000.0)
    z = np.arange(500.0, 2500.0, 500.0)
    ds = xr.Dataset(coords={"z": z, "y": y, "x": x})
    ds["radar_x"] = -20e3
    ds["radar_y"] = 10e3
    ds["radar_altitude"] = 100.0
    ds["VRADH"] = (("z", "y", "x"), np.full((len(z), len(y), len(x)), 3.0))
    bg = xr.Dataset({"u": ("z", z / 1000.0), "v": ("z", -z / 1000.0)}, coords={"z": z})
    model = BackgroundModel()
    out = single_doppler_winds(ds, bg, model=model, refine=False)
    assert len(model.calls) == 4  # 2 x 2 tiles
    expected = (z / 1000.0 + 1.0)[:, None, None]
    np.testing.assert_allclose(
        out.u.values, np.broadcast_to(expected, out.u.shape), atol=1e-5
    )
    assert (out.v.values < 0).all()


def test_tiles_cover_the_domain():
    assert sd._tiles(100, 160, 32) == [0]
    starts = sd._tiles(300, 160, 32)
    assert starts[0] == 0 and starts[-1] == 140
    w = sd._blend(300, 0, 160, 300, 32)
    assert w[0] == 1 and w[-1] < 0.1


def test_input_forms(case):
    ds, bg, _ = case
    two, _, _ = synthetic_case(step=2000.0, nz=11)
    a = single_doppler_winds(two, bg, radar=0, model=BackgroundModel(), refine=False)
    no_radar_dim = ds.isel(radar=0)
    b = single_doppler_winds(no_radar_dim, bg, model=BackgroundModel(), refine=False)
    np.testing.assert_allclose(a.u.values, b.u.values)
    with pytest.warns(UserWarning, match="no background"):
        single_doppler_winds(ds)


def test_errors(case):
    ds, bg, _ = case
    two, _, _ = synthetic_case(step=2000.0, nz=11)
    with pytest.raises(ValueError, match="several radars"):
        single_doppler_winds(two, bg)
    with pytest.raises(TypeError, match="xarray"):
        single_doppler_winds(ds.VRADH, bg)
    with pytest.raises(TypeError, match="background"):
        single_doppler_winds(ds, bg.u)
    with pytest.raises(ValueError, match="no 'VEL'"):
        single_doppler_winds(ds, bg, velocity="VEL")
    with pytest.raises(ValueError, match="at least 3"):
        single_doppler_winds(ds.isel(z=[0, 1]), bg.isel(z=[0, 1]))
    with pytest.raises(ValueError, match="x, y and z"):
        single_doppler_winds(xr.DataTree(), bg)


def test_accessor(case):
    ds, bg, _ = case
    out = ds.radarx.single_doppler_winds(bg, model=BackgroundModel(), refine=False)
    np.testing.assert_allclose(out.u.values, 9.0, atol=1e-4)


def _tiny_onnx(path, seed=0):
    """Random 1x1x1 convolution plus the background: a tiny ONNX network."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(seed)
    c = len(FEATURES)
    weight = (0.2 * rng.normal(size=(3, c, 1, 1, 1))).astype(np.float32)
    bias = rng.normal(size=3).astype(np.float32)
    pick = np.zeros((3, c, 1, 1, 1), np.float32)
    pick[0, FEATURES.index("background_u")] = sd.WIND_SCALE
    pick[1, FEATURES.index("background_v")] = sd.WIND_SCALE
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["features", "w", "b"], ["delta"]),
            helper.make_node("Conv", ["features", "pick"], ["bg"]),
            helper.make_node("Add", ["delta", "bg"], ["wind"]),
        ],
        "tiny",
        [
            helper.make_tensor_value_info(
                "features", TensorProto.FLOAT, ["n", c, "z", "y", "x"]
            )
        ],
        [
            helper.make_tensor_value_info(
                "wind", TensorProto.FLOAT, ["n", 3, "z", "y", "x"]
            )
        ],
        [
            numpy_helper.from_array(weight, "w"),
            numpy_helper.from_array(bias, "b"),
            numpy_helper.from_array(pick, "pick"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    for k, v in {
        "name": "tiny-random",
        "version": "0",
        "licence": "MIT",
        "dx": "2000",
        "dy": "2000",
        "dz": "1000",
    }.items():
        e = model.metadata_props.add()
        e.key, e.value = k, v
    onnx.save(model, path)
    return weight, bias


def test_onnx_model(case, tmp_path):
    pytest.importorskip("onnxruntime")
    ds, bg, _ = case
    path = str(tmp_path / "tiny.onnx")
    weight, bias = _tiny_onnx(path)
    out = single_doppler_winds(ds, bg, model=path, refine=False)
    assert out.attrs["ml_model"] in ("tiny-random", "tiny")  # metadata or file name
    # the grid is on the model's spacing (dz 1000 m, dx 2000 m): no regridding
    geo = radarx.retrieve.radar_geometry(ds)
    feats = sd._grid_features(geo, bg, "VRADH", "DBZH")
    expected = (
        np.einsum("qc,czyx->qzyx", weight[:, :, 0, 0, 0], feats)
        + bias[:, None, None, None]
    )
    expected[0] += feats[FEATURES.index("background_u")] * sd.WIND_SCALE
    expected[1] += feats[FEATURES.index("background_v")] * sd.WIND_SCALE
    np.testing.assert_allclose(out.u.values, expected[0], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out.w.values, expected[2], rtol=1e-4, atol=1e-4)
    refined = single_doppler_winds(ds, bg, model=path)
    assert float(np.sqrt(np.nanmean(refined.vr_residual.values**2))) < 1.0


@pytest.fixture(scope="module")
def kgwx(tmp_path_factory):
    from radarx.io.aws_data import download_file

    key = "KGWX/KGWX20220330_235959_V06"
    local = os.path.join(os.environ.get("RADARX_NEXRAD_DIR", ""), os.path.basename(key))
    if os.path.exists(local):
        return _nexrad(local)
    out = str(tmp_path_factory.mktemp("nexrad"))
    try:
        path = download_file("unidata-nexrad-level2", f"2022/03/30/{key}", out)
    except Exception as err:  # noqa: BLE001  # pragma: no cover - network
        pytest.skip(f"NEXRAD data not available: {err}")
    if path is None:  # pragma: no cover - network
        pytest.skip("NEXRAD data not available")
    return _nexrad(path)


def test_real_volume(kgwx):
    x = np.arange(-100e3, 20e3 + 1, 4000.0)
    y = np.arange(-100e3, 60e3 + 1, 4000.0)
    z = np.arange(1000.0, 9e3 + 1, 1000.0)
    bg = xr.Dataset(
        {
            "u": ("z", np.linspace(10.0, 30.0, len(z))),
            "v": ("z", np.linspace(20.0, 25.0, len(z))),
        },
        coords={"z": z},
    )
    out = single_doppler_winds(kgwx, bg, x=x, y=y, z=z)
    assert out.u.shape == (len(z), len(y), len(x))
    seen = np.isfinite(out.vr_residual.values)
    assert seen.sum() > 1000
    assert float(np.sqrt(np.mean(out.vr_residual.values[seen] ** 2))) < 3.0
    assert float(np.abs(out.w.values).max()) < 30.0
    net = kgwx.radarx.single_doppler_winds(
        bg, x=x, y=y, z=z, model=BackgroundModel((0.0, 0.0, 0.0)), refine=False
    )
    np.testing.assert_allclose(net.u.sel(z=1000.0).values, 10.0, atol=1e-4)


def test_registry_name_without_registry(case):
    """A name that is not a file goes to the radarx.ml registry."""
    ds, bg, _ = case
    with pytest.raises((ImportError, KeyError, ValueError, FileNotFoundError)):
        single_doppler_winds(ds, bg, model="no-such-model-name")


# --------------------------------------------------------------------------
# fall speed on the network path
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def uniform_case():
    """A uniform wind (8, 4, 0) m/s in heavy rain (45 dBZ) seen by one radar."""
    from radarx.retrieve import fall_speed, radar_geometry

    ds, bg, _ = synthetic_case(step=2000.0, nz=11, dbz=45.0)
    ds = ds.isel(radar=[0]).copy()
    geo = radar_geometry(ds)
    el = np.radians(geo.elevation.values[0])
    az = np.radians(geo.azimuth.values[0])
    rho = bg.air_density.values
    vt = fall_speed(
        xr.DataArray(
            np.full(rho.shape, 45.0), dims=("z", "y", "x"), coords={"z": ds.z.values}
        ),
        xr.DataArray(rho, dims=("z", "y", "x")),
    ).values
    vr = np.cos(el) * (np.sin(az) * 8.0 + np.cos(az) * 4.0) - np.sin(el) * vt
    vr = np.where(np.isfinite(ds.VRADH.values[0]), vr, np.nan)
    ds["VRADH"] = (("radar", "z", "y", "x"), vr[None].astype(np.float32))
    return ds, bg, np.isfinite(vr)


def _wind_error(out, seen):
    return {
        c: float(np.sqrt(np.mean((out[c].values - t)[seen] ** 2)))
        for c, t in zip("uvw", (8.0, 4.0, 0.0))
    }


def test_fall_speed_in_the_refinement(uniform_case):
    """The refinement corrects the radial velocity; the raw network does not."""
    ds, bg, seen = uniform_case
    model = BackgroundModel((0.0, 0.0, 0.0))  # the true wind, here the background
    on = single_doppler_winds(ds, bg, model=model)
    assert max(_wind_error(on, seen).values()) < 1e-3
    assert "variational refinement" in on.attrs["fall_speed_correction"]
    off = single_doppler_winds(ds, bg, model=model, fall_speed_correction=False)
    err = _wind_error(off, seen)
    # the fall speed of 9 to 13 m/s on beams up to 90 degrees changes the wind
    assert err["u"] > 1.0 and err["v"] > 1.0 and err["w"] > 0.5
    assert off.attrs["fall_speed_correction"].startswith("not applied (fall")
    # the variational retrieval alone does the same
    var = single_doppler_winds(ds, bg)
    assert "applied in the cost function" in var.attrs["fall_speed_correction"]
    no = single_doppler_winds(ds, bg, fall_speed_correction=False)
    assert no.attrs["fall_speed_correction"].startswith("not applied (fall")
    assert np.abs(var.w.values - no.w.values)[seen].max() > 1.0


def test_fall_speed_without_refinement(uniform_case):
    """Without refinement the argument has no effect, and the result says so."""
    ds, bg, seen = uniform_case
    model = BackgroundModel((0.0, 0.0, 0.0))
    a = single_doppler_winds(ds, bg, model=model, refine=False)
    assert "not applied explicitly" in a.attrs["fall_speed_correction"]
    with pytest.warns(UserWarning, match="no effect on a network prediction"):
        b = single_doppler_winds(
            ds, bg, model=model, refine=False, fall_speed_correction=False
        )
    np.testing.assert_array_equal(a.w.values, b.w.values)


# --------------------------------------------------------------------------
# uncertainty
# --------------------------------------------------------------------------


def test_smooth_field_statistics():
    rng = np.random.default_rng(0)
    f = sd._smooth_field((40, 120, 120), (1.0, 6.0, 6.0), rng)
    assert abs(f.std() - 1.0) < 0.08
    # correlated over several cells: neighbours agree far more than random ones
    assert np.corrcoef(f[:, :, :-1].ravel(), f[:, :, 1:].ravel())[0, 1] > 0.9
    flat = sd._smooth_field((20, 30, 30), (0.0, 0.0, 0.0), rng)
    assert abs(flat.std() - 1.0) < 0.1  # no smoothing: white noise


def test_background_error_is_mass_continuous(case):
    ds, bg, _ = case
    ds = ds.isel(radar=0)
    rng = np.random.default_rng(3)
    du, dv = sd._background_error(ds, bg, 3.0, (8000.0, 2000.0), rng)
    assert np.isclose(np.sqrt(0.5 * np.mean(du**2 + dv**2)), 3.0)
    rho = bg.air_density.values
    div = np.gradient(rho * du, 2000.0, axis=2) + np.gradient(rho * dv, 2000.0, axis=1)
    assert np.abs(div[:, 2:-2, 2:-2]).max() < 1e-6 * np.abs(rho * du).max()
    # no density in the background: the standard atmosphere
    du2, _ = sd._background_error(ds, bg.drop_vars("air_density"), 1.0, (8e3, 2e3), rng)
    assert du2.shape == du.shape
    zero = sd._background_error(ds, bg, 0.0, (8000.0, 2000.0), rng)
    assert not np.any(zero[0])


@pytest.fixture(scope="module")
def uncertain(case):
    ds, bg, _ = case
    return single_doppler_winds(
        ds, bg, uncertainty=True, members=6, vr_error=1.0, background_error=3.0
    )


def test_uncertainty_output(case, uncertain):
    ds, bg, _ = case
    plain = single_doppler_winds(ds, bg)
    for c in "uvw":
        # the retrieval itself is the unperturbed one
        np.testing.assert_array_equal(uncertain[c].values, plain[c].values)
        sig = uncertain[f"{c}_uncertainty"]
        assert sig.dims == ("z", "y", "x") and sig.attrs["units"] == "m s-1"
        assert "standard_name" not in sig.attrs
        assert np.isfinite(sig.values).all() and (sig.values >= 0).all()
        assert sig.values.max() > 0
    assert "radarx's own definition" in uncertain.attrs["uncertainty_method"]
    assert "6 retrievals" in uncertain.attrs["uncertainty_method"]
    assert "u_uncertainty" not in plain


def test_uncertainty_is_reproducible_and_leaves_inputs_alone(case):
    ds, bg, _ = case
    ds0, bg0 = ds.copy(deep=True), bg.copy(deep=True)
    kw = dict(uncertainty=True, members=3, vr_error=1.0, background_error=2.0)
    a = single_doppler_winds(ds, bg, seed=4, **kw)
    b = single_doppler_winds(ds, bg, seed=4, **kw)
    c = single_doppler_winds(ds, bg, seed=5, **kw)
    np.testing.assert_array_equal(a.u_uncertainty.values, b.u_uncertainty.values)
    assert not np.array_equal(a.u_uncertainty.values, c.u_uncertainty.values)
    xr.testing.assert_identical(ds, ds0)
    xr.testing.assert_identical(bg, bg0)


def test_uncertainty_follows_the_input_errors(case):
    ds, bg, _ = case
    kw = dict(uncertainty=True, members=4, seed=2)
    none = single_doppler_winds(ds, bg, vr_error=0.0, background_error=0.0, **kw)
    for c in "uvw":
        assert float(none[f"{c}_uncertainty"].max()) == 0.0
    small = single_doppler_winds(ds, bg, vr_error=0.2, background_error=0.0, **kw)
    big = single_doppler_winds(ds, bg, vr_error=2.0, background_error=0.0, **kw)
    assert float(big.w_uncertainty.mean()) > 3 * float(small.w_uncertainty.mean())
    bgd = single_doppler_winds(ds, bg, vr_error=0.0, background_error=4.0, **kw)
    assert float(bgd.u_uncertainty.mean()) > 0


def test_uncertainty_without_background_and_with_a_network(case):
    ds, bg, _ = case
    with pytest.warns(UserWarning, match="no background"):
        out = single_doppler_winds(ds, uncertainty=True, members=2, vr_error=1.0)
    assert float(out.w_uncertainty.max()) > 0
    net = single_doppler_winds(
        ds, bg, model=BackgroundModel(), uncertainty=True, members=3
    )
    assert float(net.u_uncertainty.max()) > 0
    assert "u_network" in net


def test_uncertainty_errors(case):
    ds, bg, _ = case
    with pytest.raises(ValueError, match="at least 2"):
        single_doppler_winds(ds, bg, uncertainty=True, members=1)
    with pytest.raises(ValueError, match="negative"):
        single_doppler_winds(ds, bg, uncertainty=True, vr_error=-1.0)
    with pytest.raises(ValueError, match="correlation"):
        single_doppler_winds(ds, bg, uncertainty=True, correlation=(1000.0,))
    with pytest.raises(ValueError, match="correlation"):
        single_doppler_winds(ds, bg, uncertainty=True, correlation=(0.0, 100.0))


def _random_case(seed, sigma_b=3.0, sigma_v=1.0):
    """Truth = background plus a mass-continuous random wind, noisy velocities."""
    from radarx.retrieve import fall_speed, radar_geometry

    ds, bg, _ = synthetic_case(step=2000.0, nz=11, dbz=30.0)
    ds = ds.isel(radar=[0]).copy()
    rng = np.random.default_rng(seed)
    rho = bg.air_density.values
    shape = rho.shape
    psi = sd._smooth_field(shape, (1.0, 12.5, 12.5), rng)
    pu = -np.gradient(psi, 2000.0, axis=1) / rho
    pv = np.gradient(psi, 2000.0, axis=2) / rho
    scale = sigma_b / np.sqrt(0.5 * np.mean(pu**2 + pv**2))
    u, v = 8.0 + scale * pu, 4.0 + scale * pv
    geo = radar_geometry(ds)
    el = np.radians(geo.elevation.values[0])
    az = np.radians(geo.azimuth.values[0])
    vt = fall_speed(
        xr.DataArray(
            np.full(shape, 30.0), dims=("z", "y", "x"), coords={"z": ds.z.values}
        ),
        xr.DataArray(rho, dims=("z", "y", "x")),
    ).values
    vr = np.cos(el) * (np.sin(az) * u + np.cos(az) * v) - np.sin(el) * vt
    vr = vr + sigma_v * rng.normal(size=shape)
    vr = np.where(np.isfinite(ds.VRADH.values[0]), vr, np.nan)
    ds["VRADH"] = (("radar", "z", "y", "x"), vr[None].astype(np.float32))
    return ds, bg, (u, v, np.zeros(shape)), np.isfinite(vr)


def test_uncertainty_matches_the_error_on_synthetic_fields():
    """
    With weights that agree with the stated errors (background weight
    (vr_error / background_error)**2 of the observation weight), the spread
    and the actual error are of the same size on synthetic fields whose
    background error is the stated one.
    """
    sigma_b, sigma_v = 3.0, 1.0
    ratios = []
    for seed in (5, 6):
        ds, bg, truth, seen = _random_case(seed, sigma_b, sigma_v)
        out = single_doppler_winds(
            ds,
            bg,
            uncertainty=True,
            members=10,
            vr_error=sigma_v,
            background_error=sigma_b,
            weights={"background": (sigma_v / sigma_b) ** 2},
        )
        for c, t in zip("uvw", truth):
            actual = np.sqrt(np.mean((out[c].values - t)[seen] ** 2))
            spread = np.sqrt(np.mean(out[f"{c}_uncertainty"].values[seen] ** 2))
            ratios.append(actual / spread)
    assert 0.6 < np.mean(ratios) < 1.6, ratios
    assert all(0.4 < r < 2.5 for r in ratios), ratios
