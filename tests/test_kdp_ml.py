#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the machine-learning KDP method
=========================================

The trained network is not distributed, so these tests use stand-in models:
a NumPy model whose KDP is the input phase derivative (exact plumbing checks)
and a tiny randomly initialised ONNX graph (when onnx and onnxruntime are
installed).
"""
import sys
import types

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.retrieve import estimate_kdp
from radarx.retrieve import kdp as kdpmod

DR = 250.0


def _sweep(nray=8, ng=400, sign=1, seed=0):
    """Known KDP (one cell), a system offset, noise, folding, a gap."""
    rnd = np.random.default_rng(seed)
    r = (np.arange(ng) + 0.5) * DR / 1000.0
    kdp = 2.0 * np.exp(-0.5 * ((r - 50) / 8) ** 2)
    phi_true = 2.0 * np.cumsum(kdp) * DR / 1000.0
    phi = 120.0 + phi_true + rnd.normal(0.0, 2.0, (nray, ng))
    phi = sign * phi
    phi = (phi + 180.0) % 360.0 - 180.0
    rho = np.full((nray, ng), 0.99)
    dbz = np.broadcast_to(25 + 10 * np.log10(1 + 30 * kdp), (nray, ng)).copy()
    far = r > 90
    phi[:, far] = np.nan
    rho[:, far] = np.nan
    dbz[:, far] = np.nan
    ds = xr.Dataset(
        {
            "PHIDP": (("azimuth", "range"), phi, {"units": "degrees"}),
            "RHOHV": (("azimuth", "range"), rho),
            "DBZH": (("azimuth", "range"), dbz),
        },
        coords={
            "azimuth": ("azimuth", np.linspace(0.5, 359.5, nray)),
            "range": ("range", (np.arange(ng) + 0.5) * DR),
            "elevation": ("azimuth", np.full(nray, 0.5)),
        },
    )
    return ds, kdp, ~far


class DerivativeModel:
    """KDP = the phase-derivative feature (no learning), delta = 0."""

    info = {"name": "derivative", "version": "0", "licence": "MIT"}

    def __init__(self):
        self.calls = []

    def run(self, inputs):
        x = inputs["features"]
        self.calls.append(x.shape)
        assert x.dtype == np.float32
        assert x.shape[1] == len(kdpmod.ML_FEATURES)
        kdp = 10.0 * x[:, 0]
        return {
            "kdp": kdp,
            "delta": np.zeros_like(kdp),
            "kdp_std": np.full_like(kdp, 0.5),
        }


def test_ml_method_with_stand_in_model():
    ds, kdp_true, inside = _sweep()
    model = DerivativeModel()
    out = estimate_kdp(ds, method="ml", model=model)
    assert set(out.data_vars) == {
        "PHIDP_processed",
        "KDP",
        "PHIDP_OFFSET",
        "PHIDP_BACKSCATTER",
        "KDP_UNCERTAINTY",
    }
    assert out.KDP.dims == ds.PHIDP.dims
    assert out.KDP.attrs["ml_model"] == "derivative"
    assert out.KDP.attrs["ml_model_version"] == "0"
    assert out.KDP.attrs["ml_model_licence"] == "MIT"
    assert out.KDP.attrs["units"] == "degrees/km"
    assert out.PHIDP_BACKSCATTER.attrs["units"] == "degrees"
    # NaN outside echo, values inside
    assert np.isnan(out.KDP.values[:, ~inside]).all()
    assert np.isfinite(out.KDP.values[:, inside]).all()
    # the raw derivative is noisy but unbiased: the ray mean recovers KDP
    mean = out.KDP.values[:, inside].mean(axis=0)
    smooth = np.convolve(mean, np.ones(9) / 9, mode="same")
    assert np.abs(smooth - kdp_true[inside])[10:-10].max() < 0.5
    # offset (sweep mode) close to the truth
    np.testing.assert_allclose(out.PHIDP_OFFSET.values, 120.0, atol=3.0)
    # the processed phase is twice the integral of KDP, starting near 0
    phi = out.PHIDP_processed.values
    dphi = np.diff(phi, axis=1) / (DR / 1000.0)
    k = out.KDP.values
    np.testing.assert_allclose(
        dphi[:, inside[1:]][:, :-1],
        (k[:, 1:] + k[:, :-1])[:, inside[1:]][:, :-1],
        atol=1e-9,
    )
    assert abs(np.nanmean(phi[:, :5])) < 5.0
    assert np.nanmax(phi) == pytest.approx(2 * kdp_true.sum() * DR / 1000.0, abs=10)


def test_ml_chunks_rays(monkeypatch):
    monkeypatch.setattr(kdpmod, "_ML_CHUNK", 3)
    ds, *_ = _sweep(nray=8)
    model = DerivativeModel()
    out = estimate_kdp(ds, method="ml", model=model)
    assert [c[0] for c in model.calls] == [3, 3, 2]
    ref = estimate_kdp(ds, method="ml", model=DerivativeModel())
    xr.testing.assert_identical(out, ref)


def test_ml_reversed_sign_and_transposed_input():
    ds, *_ = _sweep(sign=-1)
    out = estimate_kdp(ds, method="ml", model=DerivativeModel())
    assert out.PHIDP_processed.attrs["phidp_sign"] == -1
    assert np.nanmean(out.KDP.values) > 0.3
    tr = estimate_kdp(
        ds.transpose("range", "azimuth"), method="ml", model=DerivativeModel()
    )
    assert tr.KDP.dims == ("range", "azimuth")
    assert tr.PHIDP_BACKSCATTER.dims == ("range", "azimuth")
    np.testing.assert_allclose(tr.KDP.T.values, out.KDP.values, equal_nan=True)
    np.testing.assert_allclose(
        tr.KDP_UNCERTAINTY.T.values, out.KDP_UNCERTAINTY.values, equal_nan=True
    )


def test_ml_without_optional_fields():
    ds, *_ = _sweep()
    model = DerivativeModel()
    out = estimate_kdp(ds.drop_vars(["RHOHV", "DBZH"]), method="ml", model=model)
    assert out.PHIDP_processed.attrs["source_fields"] == "PHIDP"
    assert np.isfinite(out.KDP.values).any()


def test_ml_datatree_skips_sweeps_without_phase():
    ds, *_ = _sweep()
    dtree = xr.DataTree.from_dict(
        {"/": xr.Dataset(), "sweep_0": ds, "sweep_1": ds.drop_vars("PHIDP")}
    )
    out = dtree.radarx.kdp(method="ml", model=DerivativeModel())
    assert list(out.children) == ["sweep_0"]
    single = estimate_kdp(ds, method="ml", model=DerivativeModel())
    xr.testing.assert_identical(out["sweep_0"].to_dataset(inherit=False), single)


def _fake_ml(monkeypatch, listed, model=None):
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


def test_ml_model_not_registered(monkeypatch):
    ds, *_ = _sweep(nray=2, ng=100)
    _fake_ml(monkeypatch, [{"name": "other", "task": "x"}])
    with pytest.raises(ValueError, match="radarx-kdp-v1.*not registered"):
        estimate_kdp(ds, method="ml")
    _fake_ml(monkeypatch, {})
    with pytest.raises(ValueError, match="'mine' is not registered"):
        estimate_kdp(ds, method="ml", model="mine")


@pytest.mark.parametrize("listed", [[{"name": "mine"}], {"mine": {}}])
def test_ml_model_by_name(monkeypatch, listed):
    ds, *_ = _sweep(nray=2, ng=100)

    class Unnamed:
        def run(self, inputs):
            return DerivativeModel().run(inputs)

    loaded = _fake_ml(monkeypatch, listed, Unnamed())
    out = estimate_kdp(ds, method="ml", model="mine")
    assert loaded == ["mine"]
    assert out.KDP.attrs["ml_model"] == "mine"
    assert out.KDP.attrs["ml_model_version"] == "unknown"


def test_ml_needs_radarx_ml(monkeypatch):
    ds, *_ = _sweep(nray=2, ng=100)
    monkeypatch.setitem(sys.modules, "radarx.ml", None)
    monkeypatch.delattr(radarx, "ml", raising=False)
    with pytest.raises(ImportError, match=r"radarx\[ml\]"):
        estimate_kdp(ds, method="ml")


def test_ml_bad_models():
    ds, *_ = _sweep(nray=2, ng=100)
    with pytest.raises(TypeError, match="run"):
        estimate_kdp(ds, method="ml", model=object())

    class Incomplete:
        def run(self, inputs):
            return {"kdp": inputs["features"][:, 0]}

    with pytest.raises(KeyError, match="delta"):
        estimate_kdp(ds, method="ml", model=Incomplete())
    with pytest.raises(ValueError, match="'ml'"):
        estimate_kdp(ds, method="nope")


def test_ml_features_layout():
    ds, *_ = _sweep(nray=3, ng=100)
    params = {
        "rhohv_min": 0.85,
        "texture_max": 20.0,
        "n_offset": 10,
        "offset_mode": "sweep",
        "offset": 0.0,
        "htex": 4,
    }
    phi = ds.PHIDP.values
    feats, psi, valid, good, offset = kdpmod._ml_features(
        phi, ds.RHOHV.values, ds.DBZH.values, 0.25, params, 1
    )
    assert feats.shape == (3, len(kdpmod.ML_FEATURES), 100)
    assert feats.dtype == np.float32
    assert good.all()
    np.testing.assert_array_equal(feats[:, 1], valid)
    assert np.all(feats[:, 6] == 0.25)
    assert np.all(feats[:, 3] == 1) and np.all(feats[:, 5] == 1)
    assert np.isnan(psi[~valid]).all()
    bare = kdpmod._ml_features(phi, None, None, 0.25, params, 1)[0]
    assert np.all(bare[:, 2:6] == 0)
    np.testing.assert_allclose(bare[:, 0], feats[:, 0], atol=1e-6)


class OnnxModel:
    """A minimal ONNX Runtime wrapper with the radarx.ml Model interface."""

    def __init__(self, path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.info = {"name": "tiny-random", "version": "test", "licence": "MIT"}

    def run(self, inputs):
        names = [o.name for o in self.session.get_outputs()]
        return dict(zip(names, self.session.run(names, inputs)))


def _tiny_onnx(path, seed=0):
    """Random 1-D convolution: 7 features -> (kdp, delta, softplus std)."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rnd = np.random.default_rng(seed)
    nf = len(kdpmod.ML_FEATURES)
    w = numpy_helper.from_array(rnd.normal(0, 0.3, (3, nf, 5)).astype(np.float32), "w")
    b = numpy_helper.from_array(np.zeros(3, np.float32), "b")
    one = numpy_helper.from_array(np.array([1], np.int64), "axis1")
    nodes = [
        helper.make_node("Conv", ["features", "w", "b"], ["y"], pads=[2, 2]),
        helper.make_node("Split", ["y"], ["k3", "d3", "s3"], axis=1, num_outputs=3),
        helper.make_node("Squeeze", ["k3", "axis1"], ["kdp"]),
        helper.make_node("Squeeze", ["d3", "axis1"], ["delta"]),
        helper.make_node("Softplus", ["s3"], ["s3p"]),
        helper.make_node("Squeeze", ["s3p", "axis1"], ["kdp_std"]),
    ]
    out = [
        helper.make_tensor_value_info(n, TensorProto.FLOAT, ["ray", "range"])
        for n in ("kdp", "delta", "kdp_std")
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [
            helper.make_tensor_value_info(
                "features", TensorProto.FLOAT, ["ray", nf, "range"]
            )
        ],
        out,
        initializer=[w, b, one],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def test_ml_with_tiny_onnx_model(tmp_path):
    pytest.importorskip("onnxruntime")
    path = _tiny_onnx(str(tmp_path / "tiny.onnx"))
    model = OnnxModel(path)
    ds, _, inside = _sweep(nray=4, ng=300)
    out = estimate_kdp(ds, method="ml", model=model)
    assert out.KDP.attrs["ml_model"] == "tiny-random"
    k = out.KDP.values
    assert np.isfinite(k[:, inside]).all()
    assert np.isnan(k[:, ~inside]).all()
    assert (out.KDP_UNCERTAINTY.values[:, inside] > 0).all()
    assert np.isfinite(out.PHIDP_processed.values).all()
