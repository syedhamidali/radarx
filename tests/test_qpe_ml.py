#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the machine-learning rain rate
========================================

The weights of the radarx network are not distributed, so the tests use
stand-in models whose rain rate is known in closed form: the power law
``R(Z)`` of :func:`radarx.retrieve.rain_rate_z` as a NumPy model and as a
tiny ONNX graph built in the test (when onnx and onnxruntime are installed).
The result of the whole chain must equal ``rain_rate_z``, which checks the
feature columns, the gathering of the gates, the model call and the writing
back independently of what a network learned.
"""
import sys
import types

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.retrieve import qpe_ml as mlmod
from radarx.retrieve import rain_rate_ml, rain_rate_z

from .provenance_helpers import without_provenance

ENGINES = ["numpy"] + (["compiled"] if mlmod.HAS_COMPILED_KERNEL else [])
compiled_only = pytest.mark.skipif(
    not mlmod.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)
A_Z, B_Z = 0.017, 0.714  # the S-band R(Z) of rain_rate_z


def sweep(n_az=18, n_rng=60, seed=0, altitude=250.0):
    """A sweep with random fields and the coordinates of a georeferenced one."""
    rng = np.random.default_rng(seed)
    dims = ("azimuth", "range")
    shape = (n_az, n_rng)
    dbz = rng.uniform(10, 50, shape)
    dbz[:, 45:] = np.nan  # no echo far out
    ds = xr.Dataset(
        {
            "DBZH": (dims, dbz, {"units": "dBZ"}),
            "ZDR": (dims, rng.uniform(0, 3, shape)),
            "KDP": (dims, rng.uniform(0, 2, shape)),
            "RHOHV": (dims, rng.uniform(0.9, 1.0, shape)),
        },
        coords={
            "azimuth": np.arange(n_az) * 20.0,
            "range": 500.0 + np.arange(n_rng) * 250.0,
            "z": (
                dims,
                np.broadcast_to(altitude + 500.0 + np.arange(n_rng) * 20.0, shape),
            ),
            "altitude": altitude,
        },
    )
    ds["ZDR"] = ds.ZDR.where(np.isfinite(ds.DBZH))
    ds["KDP"] = ds.KDP.where(np.isfinite(ds.DBZH))
    ds["RHOHV"] = ds.RHOHV.where(np.isfinite(ds.DBZH))
    return ds


class PowerLaw:
    """R(Z) as a model: reads the dbz column, ignores the others."""

    def __init__(self, features=("dbz", "zdr", "kdp", "rhohv"), outputs=True):
        self.features = list(features)
        self.info = {
            "name": "power-law",
            "version": "7",
            "licence": "MIT",
            "citation": "Nobody (2000), https://doi.org/10.1000/xyz",
        }
        self.outputs = outputs
        self.calls = []

    def run(self, inputs):
        x = inputs["features"]
        self.calls.append(x.shape)
        assert x.dtype == np.float32
        assert x.shape[1] == len(self.features)
        assert np.isfinite(x).all()
        rr = A_Z * 10.0 ** (0.1 * B_Z * x[:, self.features.index("dbz")])
        out = {"rain_rate": rr.astype(np.float32)}
        if self.outputs:
            out["rain_rate_lower"] = (0.8 * rr).astype(np.float32)
            out["rain_rate_upper"] = (1.25 * rr).astype(np.float32)
        return out


# --------------------------------------------------------------------------
# the chain against the power law
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_stand_in_model_reproduces_the_power_law(engine):
    ds = sweep()
    model = PowerLaw()
    out = rain_rate_ml(ds, model=model, engine=engine)
    ref = rain_rate_z(ds)
    assert set(out.data_vars) == {
        "RAIN_RATE_ML",
        "RAIN_RATE_ML_LOWER",
        "RAIN_RATE_ML_UPPER",
    }
    assert out.RAIN_RATE_ML.dims == ds.DBZH.dims
    assert out.RAIN_RATE_ML.dtype == np.float32
    np.testing.assert_allclose(out.RAIN_RATE_ML.values, ref.values, rtol=1e-5)
    np.testing.assert_allclose(
        out.RAIN_RATE_ML_LOWER.values, 0.8 * ref.values, rtol=1e-5
    )
    np.testing.assert_allclose(
        out.RAIN_RATE_ML_UPPER.values, 1.25 * ref.values, rtol=1e-5
    )
    # only the gates with echo were given to the network
    assert model.calls == [(int(np.isfinite(ds.DBZH.values).sum()), 4)]
    assert np.isnan(out.RAIN_RATE_ML.values[:, 45:]).all()
    attrs = out.RAIN_RATE_ML.attrs
    assert attrs["units"] == "mm h-1"
    assert attrs["standard_name"] == "rainfall_rate"
    assert attrs["ml_model"] == "power-law"
    assert attrs["ml_model_version"] == "7"
    assert attrs["ml_model_licence"] == "MIT"
    assert attrs["ml_model_citation"].endswith("10.1000/xyz")
    np.testing.assert_array_equal(out.azimuth.values, ds.azimuth.values)


@compiled_only
@pytest.mark.parametrize("n_threads", [1, 3, None])
def test_engines_agree(n_threads):
    ds = sweep(n_az=40, n_rng=300, seed=3)
    a = rain_rate_ml(ds, model=PowerLaw(), engine="numpy")
    b = rain_rate_ml(ds, model=PowerLaw(), engine="compiled", n_threads=n_threads)
    xr.testing.assert_identical(without_provenance(a), without_provenance(b))


@compiled_only
def test_kernel_gathers_in_gate_order_across_blocks():
    n = 3 * (1 << 16) + 123  # several blocks
    rng = np.random.default_rng(0)
    cols = [rng.uniform(0, 1, n).astype(np.float32) for _ in range(3)]
    cols[0][rng.random(n) < 0.7] = np.nan
    mask = rng.random(n) < 0.8
    mat, idx = mlmod._compact(cols, mask, True, 4)
    mat0, idx0 = mlmod._compact(cols, mask, False, None)
    np.testing.assert_array_equal(idx, idx0)
    np.testing.assert_array_equal(mat, mat0)
    assert (np.diff(idx) > 0).all()
    values = rng.uniform(size=(idx.size, 2)).astype(np.float32)
    s1 = mlmod._scatter(idx, values, n, True, 4)
    s0 = mlmod._scatter(idx, values, n, False, None)
    np.testing.assert_array_equal(s1, s0)
    assert np.isnan(s1[:, np.setdiff1d(np.arange(n), idx)]).all()


@compiled_only
def test_kernel_argument_errors():
    k = mlmod._qpe_ml
    a = np.zeros(4, dtype=np.float32)
    with pytest.raises(ValueError, match="at least one"):
        k.compact([], np.empty(0, dtype=np.uint8))
    with pytest.raises(ValueError, match="same size"):
        k.compact([a, a[:3]], np.empty(0, dtype=np.uint8))
    with pytest.raises(ValueError, match="mask"):
        k.compact([a], np.zeros(3, dtype=np.uint8))
    with pytest.raises(ValueError, match="shape"):
        k.scatter(np.zeros(2, dtype=np.int64), np.zeros((3, 1), dtype=np.float32), 5)
    with pytest.raises(ValueError, match="negative"):
        k.scatter(np.zeros(0, dtype=np.int64), np.zeros((0, 1), dtype=np.float32), -1)
    with pytest.raises(ValueError, match="out of range"):
        k.scatter(np.array([9], dtype=np.int64), np.zeros((1, 1), dtype=np.float32), 5)


def test_compiled_engine_without_kernel(monkeypatch):
    monkeypatch.setattr(mlmod, "HAS_COMPILED_KERNEL", False)
    ds = sweep(n_az=4, n_rng=50)
    with pytest.raises(ImportError):
        rain_rate_ml(ds, model=PowerLaw(), engine="compiled")
    out = rain_rate_ml(ds, model=PowerLaw())
    assert np.isfinite(out.RAIN_RATE_ML.values).any()


def test_chunks_of_gates(monkeypatch):
    monkeypatch.setattr(mlmod, "_CHUNK", 100)
    ds = sweep()
    model = PowerLaw()
    out = rain_rate_ml(ds, model=model)
    assert len(model.calls) > 3 and all(c[0] <= 100 for c in model.calls)
    ref = rain_rate_z(ds)
    np.testing.assert_allclose(out.RAIN_RATE_ML.values, ref.values, rtol=1e-5)


# --------------------------------------------------------------------------
# fields, masks, options
# --------------------------------------------------------------------------


def test_mask_and_min_dbz():
    ds = sweep()
    keep = ds.DBZH > 30
    out = rain_rate_ml(ds, model=PowerLaw(), mask=keep)
    ref = rain_rate_z(ds).where(keep)
    np.testing.assert_allclose(out.RAIN_RATE_ML.values, ref.values, rtol=1e-5)
    out = rain_rate_ml(ds, model=PowerLaw(), min_dbz=30.0)
    np.testing.assert_allclose(out.RAIN_RATE_ML.values, ref.values, rtol=1e-5)
    both = rain_rate_ml(ds, model=PowerLaw(), min_dbz=40.0, mask=keep)
    assert np.isfinite(both.RAIN_RATE_ML.values).sum() < np.isfinite(ref.values).sum()
    # a mask on a transposed grid is aligned by dimension name
    t = rain_rate_ml(ds, model=PowerLaw(), mask=keep.transpose("range", "azimuth"))
    xr.testing.assert_allclose(t.RAIN_RATE_ML, out.RAIN_RATE_ML.where(keep))


def test_nothing_to_estimate():
    ds = sweep()
    out = rain_rate_ml(ds, model=PowerLaw(), mask=ds.DBZH < -100)
    assert np.isnan(out.RAIN_RATE_ML.values).all()
    assert out.RAIN_RATE_ML.dims == ds.DBZH.dims


def test_field_names_and_dataarrays_as_features():
    ds = sweep().rename({"DBZH": "refl", "KDP": "kdp_ml"})
    with pytest.raises(KeyError, match="reflectivity"):
        rain_rate_ml(ds, model=PowerLaw())
    out = rain_rate_ml(ds, model=PowerLaw(), dbz="refl", kdp="kdp_ml")
    out2 = rain_rate_ml(
        ds.drop_vars("refl"), model=PowerLaw(), dbz=ds["refl"], kdp="kdp_ml"
    )
    xr.testing.assert_identical(without_provenance(out), without_provenance(out2))
    with pytest.raises(KeyError, match="nope"):
        rain_rate_ml(ds, model=PowerLaw(), dbz="nope")


def test_height_wavelength_and_temperature_features():
    ds = sweep()
    seen = {}

    class Probe(PowerLaw):
        def run(self, inputs):
            seen["x"] = inputs["features"].copy()
            return super().run(inputs)

    model = Probe(features=("dbz", "height", "temperature", "wavelength"))
    temperature = xr.full_like(ds.DBZH, 283.15).drop_attrs()  # Kelvin, from the values
    rain_rate_ml(ds, model=model, temperature=temperature, band="C")
    x = seen["x"]
    good = np.isfinite(ds.DBZH.values)
    np.testing.assert_allclose(x[:, 2], 10.0, atol=1e-4)
    np.testing.assert_allclose(x[:, 3], mlmod.WAVELENGTH["C"], rtol=1e-6)
    # height above the radar in km: z - altitude
    want = (ds.z.values - 250.0)[good] / 1000.0
    np.testing.assert_allclose(x[:, 1], want, rtol=1e-5)
    # temperature with units, a number, wavelength override, height as a field in km
    ref = xr.full_like(ds.DBZH, 10.0, dtype=float)
    kel = xr.full_like(ds.DBZH, 283.15).assign_attrs(units="K")
    height_km = (ds.z - 250.0).assign_attrs(units="m") / 1000.0
    height_km.attrs["units"] = "km"
    for kw in (
        {"temperature": 10.0},
        {"temperature": kel},
        {"temperature": ref},
        {"temperature": "TEMP"},
    ):
        data = ds.assign(TEMP=ref) if isinstance(kw["temperature"], str) else ds
        rain_rate_ml(data, model=model, wavelength=3.3, height=height_km, **kw)
        np.testing.assert_allclose(seen["x"][:, 2], 10.0, atol=1e-4)
        np.testing.assert_allclose(seen["x"][:, 3], 3.3, rtol=1e-6)
        np.testing.assert_allclose(seen["x"][:, 1], want, rtol=1e-5)
    rain_rate_ml(ds, model=model, temperature=5.0, height=2000.0)
    np.testing.assert_allclose(seen["x"][:, 1], 2.0, rtol=1e-6)
    named = ds.assign(HGT=ds.z - 250.0)
    rain_rate_ml(named, model=model, temperature=5.0, height="HGT")
    np.testing.assert_allclose(seen["x"][:, 1], want, rtol=1e-5)
    with pytest.raises(KeyError, match="HGT"):
        rain_rate_ml(ds, model=model, temperature=5.0, height="HGT")
    # the height is needed, and the temperature
    with pytest.raises(KeyError, match="height"):
        rain_rate_ml(ds.drop_vars("z"), model=model, temperature=5.0)
    with pytest.raises(KeyError, match="temperature"):
        rain_rate_ml(ds, model=model)
    # a sweep above sea level without the radar altitude: z is the height
    no_alt = ds.drop_vars("altitude")
    rain_rate_ml(no_alt, model=model, temperature=5.0)
    np.testing.assert_allclose(seen["x"][:, 1], ds.z.values[good] / 1000.0, rtol=1e-5)


def test_scalar_only_features_need_a_field():
    ds = sweep()
    model = PowerLaw(features=("height", "wavelength"))
    with pytest.raises(ValueError, match="at least one"):
        rain_rate_ml(ds, model=model, height=1000.0)


def test_dataarray_fields_with_a_dataarray_object_are_refused():
    ds = sweep()
    with pytest.raises(TypeError, match="obj must be"):
        rain_rate_ml(ds.DBZH, model=PowerLaw())


def test_input_is_not_changed():
    ds = sweep()
    before = ds.copy(deep=True)
    rain_rate_ml(ds, model=PowerLaw())
    xr.testing.assert_identical(ds, before)


def test_model_without_uncertainty():
    out = rain_rate_ml(sweep(), model=PowerLaw(outputs=False))
    assert list(out.data_vars) == ["RAIN_RATE_ML"]


def test_negative_outputs_are_set_to_zero():
    class Negative(PowerLaw):
        def run(self, inputs):
            return {"rain_rate": -np.ones(len(inputs["features"]), dtype=np.float32)}

    out = rain_rate_ml(sweep(), model=Negative())
    assert np.nanmax(out.RAIN_RATE_ML.values) == 0.0


# --------------------------------------------------------------------------
# volumes
# --------------------------------------------------------------------------


def test_datatree_skips_sweeps_without_the_fields():
    ds = sweep()
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(coords={"altitude": 250.0}),
            "sweep_0": ds.drop_vars("altitude"),
            "sweep_1": ds.drop_vars(["KDP", "altitude"]),
        }
    )
    out = rain_rate_ml(tree, model=PowerLaw())
    assert list(out.children) == ["sweep_0"]
    single = rain_rate_ml(ds, model=PowerLaw())
    np.testing.assert_allclose(
        out["sweep_0"].to_dataset(inherit=False).RAIN_RATE_ML.values,
        single.RAIN_RATE_ML.values,
    )
    via = tree.radarx.rain_rate_ml(model=PowerLaw())
    assert list(via.children) == ["sweep_0"]
    with pytest.raises(KeyError, match="no sweep"):
        rain_rate_ml(
            xr.DataTree.from_dict({"/": xr.Dataset(), "sweep_0": ds.drop_vars("KDP")}),
            model=PowerLaw(),
        )


def test_dataset_accessor_and_provenance():
    ds = sweep()
    out = ds.radarx.rain_rate_ml(model=PowerLaw())
    direct = rain_rate_ml(ds, model=PowerLaw())
    xr.testing.assert_identical(without_provenance(out), without_provenance(direct))
    refs = out.RAIN_RATE_ML.attrs["radarx_references"]
    assert "10.5194/amt-14-3169-2021" in refs


# --------------------------------------------------------------------------
# the model argument
# --------------------------------------------------------------------------


def fake_ml(monkeypatch, listed, model=None):
    """Install a stand-in radarx.ml module (the real one is optional)."""
    mod = types.ModuleType("radarx.ml")
    loaded = []

    def load_model(name, **kwargs):
        loaded.append(name)
        return model

    mod.list_models = lambda: listed
    mod.load_model = load_model
    monkeypatch.setitem(sys.modules, "radarx.ml", mod)
    monkeypatch.setattr(radarx, "ml", mod, raising=False)
    return loaded


def test_model_not_registered(monkeypatch):
    ds = sweep(n_az=2, n_rng=10)
    fake_ml(monkeypatch, [{"name": "other"}])
    with pytest.raises(ValueError, match="radarx-qpe-v1.*not registered"):
        rain_rate_ml(ds)
    fake_ml(monkeypatch, {})
    with pytest.raises(ValueError, match="'mine' is not registered"):
        rain_rate_ml(ds, model="mine")


@pytest.mark.parametrize("listed", [[{"name": "mine"}], {"mine": {}}])
def test_model_by_name(monkeypatch, listed):
    ds = sweep(n_az=2, n_rng=30)
    loaded = fake_ml(monkeypatch, listed, PowerLaw())
    out = rain_rate_ml(ds, model="mine")
    assert loaded == ["mine"]
    assert out.RAIN_RATE_ML.attrs["ml_model"] == "power-law"


def test_features_from_the_registry_entry_and_errors():
    ds = sweep(n_az=2, n_rng=30)

    class Bare:
        info = {"name": "bare", "features": "dbz, zdr"}

        def run(self, inputs):
            assert inputs["features"].shape[1] == 2
            return {"rain_rate": np.ones(len(inputs["features"]), dtype=np.float32)}

    out = rain_rate_ml(ds, model=Bare())
    assert out.RAIN_RATE_ML.attrs["ml_model"] == "bare"
    assert out.RAIN_RATE_ML.attrs["ml_model_version"] == "unknown"
    assert "ml_model_citation" not in out.RAIN_RATE_ML.attrs

    class NoFeatures:
        def run(self, inputs):  # pragma: no cover - not called
            return {}

    with pytest.raises(ValueError, match="features"):
        rain_rate_ml(ds, model=NoFeatures())
    bad = PowerLaw()
    bad.features = ["dbz", "humidity"]
    with pytest.raises(ValueError, match="humidity"):
        rain_rate_ml(ds, model=bad)
    bad.features = []
    with pytest.raises(ValueError, match="must be some of"):
        rain_rate_ml(ds, model=bad)
    with pytest.raises(TypeError, match="run"):
        rain_rate_ml(ds, model=3)

    class NoRate(PowerLaw):
        def run(self, inputs):
            return {"other": np.zeros(len(inputs["features"]), dtype=np.float32)}

    with pytest.raises(KeyError, match="rain_rate"):
        rain_rate_ml(ds, model=NoRate())

    class Short(PowerLaw):
        def run(self, inputs):
            return {"rain_rate": np.zeros(1, dtype=np.float32)}

    with pytest.raises(ValueError, match="returned 1 values"):
        rain_rate_ml(ds, model=Short())


def test_missing_onnxruntime_is_explained(monkeypatch, tmp_path):
    from radarx import ml

    path = tmp_path / "never_loaded.onnx"
    path.write_bytes(b"not a model")
    ml.register_model(
        "test-qpe-no-runtime", path, None, "MIT", "x", features=["dbz"], overwrite=True
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(ImportError, match=r"radarx\[ml\]"):
        rain_rate_ml(sweep(n_az=2, n_rng=10), model="test-qpe-no-runtime")


def test_option_errors():
    ds = sweep(n_az=2, n_rng=10)
    with pytest.raises(ValueError, match="band"):
        rain_rate_ml(ds, model=PowerLaw(), band="K")
    with pytest.raises(ValueError, match="engine"):
        rain_rate_ml(ds, model=PowerLaw(), engine="gpu")
    with pytest.raises(ValueError, match="min_dbz"):
        rain_rate_ml(ds, model=PowerLaw(features=("zdr",)), min_dbz=10.0)
    with pytest.raises(ValueError, match="DataArray"):
        mlmod._gather(None, ["dbz"], {}, 10.0)


# --------------------------------------------------------------------------
# a real ONNX model through the registry
# --------------------------------------------------------------------------


def power_law_onnx(path, features):
    """R(Z) as an ONNX graph: rain_rate = exp(ln a + 0.1 b ln 10 dbz)."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    n_f = len(features)
    w = np.zeros((n_f, 1), dtype=np.float32)
    w[features.index("dbz"), 0] = 0.1 * B_Z * np.log(10.0)
    inits = [
        helper.make_tensor("w", TensorProto.FLOAT, [n_f, 1], w.ravel().tolist()),
        helper.make_tensor("b", TensorProto.FLOAT, [1], [float(np.log(A_Z))]),
        helper.make_tensor("lo", TensorProto.FLOAT, [], [0.8]),
        helper.make_tensor("hi", TensorProto.FLOAT, [], [1.25]),
        helper.make_tensor("axes", TensorProto.INT64, [1], [1]),
    ]
    nodes = [
        helper.make_node("Gemm", ["features", "w", "b"], ["lin"]),
        helper.make_node("Exp", ["lin"], ["col"]),
        helper.make_node("Squeeze", ["col", "axes"], ["rain_rate"]),
        helper.make_node("Mul", ["rain_rate", "lo"], ["rain_rate_lower"]),
        helper.make_node("Mul", ["rain_rate", "hi"], ["rain_rate_upper"]),
    ]
    graph = helper.make_graph(
        nodes,
        "power_law",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, ["N", n_f])],
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, ["N"])
            for name in ("rain_rate", "rain_rate_lower", "rain_rate_upper")
        ],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def test_onnx_model_registered_with_features(tmp_path):
    pytest.importorskip("onnxruntime")
    from radarx import ml

    features = ["dbz", "zdr", "kdp", "rhohv", "wavelength"]
    path = tmp_path / "power_law.onnx"
    power_law_onnx(path, features)
    ml.register_model(
        "test-qpe-power-law",
        path,
        None,
        "MIT",
        "Test (2026), https://doi.org/10.1000/test",
        version="3",
        task="rainfall-estimation",
        features=features,
        overwrite=True,
    )
    ds = sweep()
    out = rain_rate_ml(ds, model="test-qpe-power-law", band="X")
    ref = rain_rate_z(ds)
    np.testing.assert_allclose(out.RAIN_RATE_ML.values, ref.values, rtol=2e-4)
    np.testing.assert_allclose(
        out.RAIN_RATE_ML_UPPER.values, 1.25 * ref.values, rtol=2e-4
    )
    attrs = out.RAIN_RATE_ML.attrs
    assert attrs["ml_model"] == "test-qpe-power-law"
    assert attrs["ml_model_version"] == "3"
    assert attrs["ml_model_licence"] == "MIT"
    assert attrs["ml_model_citation"].endswith("10.1000/test")
    loaded = ml.load_model("test-qpe-power-law")
    assert loaded.info["features"] == features
    assert loaded.info["task"] == "rainfall-estimation"
    again = rain_rate_ml(ds, model=loaded, band="X")
    xr.testing.assert_identical(without_provenance(out), without_provenance(again))
