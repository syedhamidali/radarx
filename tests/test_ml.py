#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for radarx.ml: polar patches, normalisation, ONNX models, registry."""

import hashlib
import sys

import numpy as np
import pytest
import xarray as xr

import radarx.ml as ml
from radarx.ml import model as model_mod
from radarx.ml import patches as patch_mod

compiled = pytest.mark.skipif(
    not patch_mod.HAS_COMPILED_KERNEL, reason="compiled patch kernel not built"
)
ENGINES = ["numpy"] + (["compiled"] if patch_mod.HAS_COMPILED_KERNEL else [])


def sweep(nray=360, ngate=200, seed=0, dtype=np.float32):
    """A sweep with a non-trivial field, NaN gaps and polar coordinates."""
    rng = np.random.default_rng(seed)
    azimuth = (np.arange(nray) * 360.0 / nray + 137.0) % 360.0  # starts at 137 deg
    ranges = 125.0 + 250.0 * np.arange(ngate)
    data = rng.normal(20.0, 10.0, (nray, ngate)).astype(dtype)
    data[rng.random((nray, ngate)) < 0.1] = np.nan
    a, r = np.meshgrid(np.radians(azimuth), ranges, indexing="ij")
    return xr.DataArray(
        data,
        dims=("azimuth", "range"),
        coords={
            "azimuth": azimuth,
            "range": ranges,
            "elevation": ("azimuth", np.full(nray, 0.5)),
            "x": (("azimuth", "range"), r * np.sin(a)),
            "y": (("azimuth", "range"), r * np.cos(a)),
            "sweep_fixed_angle": 0.5,
        },
        name="DBZH",
        attrs={"units": "dBZ", "standard_name": "equivalent_reflectivity_factor"},
    )


# --------------------------------------------------------------------------
# patches
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("blend", ["cosine", "linear", "mean"])
@pytest.mark.parametrize(
    "size, stride, wrap",
    [
        ((32, 64), None, True),
        ((32, 64), (32, 64), True),
        ((40, 48), (7, 13), True),
        ((32, 64), (16, 32), False),
        ((50, 300), None, False),  # patches longer than the sweep (padded)
        ((361, 16), None, True),  # more rays than the sweep (wraps twice)
    ],
)
def test_round_trip_is_exact(engine, blend, size, stride, wrap):
    da = sweep(nray=360, ngate=200)
    patches, index = ml.polar_patches(
        da.values, size, stride, wrap_azimuth=wrap, engine=engine
    )
    assert patches.dtype == np.float32
    assert patches.shape[1:] == size
    out = ml.reassemble(patches, index, blend=blend, engine=engine)
    np.testing.assert_array_equal(out, da.values)


@pytest.mark.parametrize("engine", ENGINES)
def test_round_trip_float64(engine):
    data = sweep(dtype=np.float64).values
    patches, index = ml.polar_patches(data, (24, 40), engine=engine)
    assert patches.dtype == np.float64
    out = ml.reassemble(patches, index, engine=engine)
    np.testing.assert_array_equal(out, data)


@compiled
@pytest.mark.parametrize("wrap", [True, False])
@pytest.mark.parametrize("blend", ["cosine", "linear", "mean"])
def test_engines_agree(wrap, blend):
    data = np.stack([sweep(seed=s).values for s in range(3)])  # (3, A, R)
    a, ia = ml.polar_patches(data, (30, 50), (11, 17), wrap, engine="numpy")
    b, ib = ml.polar_patches(data, (30, 50), (11, 17), wrap, engine="compiled")
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ia.table, ib.table)
    # model output: different values in every patch, so blending matters
    rng = np.random.default_rng(1)
    pred = (a * rng.uniform(0.5, 1.5, (len(a), 1, 1, 1))).astype(np.float32)
    ra = ml.reassemble(pred, ia, blend=blend, engine="numpy")
    rb = ml.reassemble(pred, ia, blend=blend, engine="compiled", n_threads=3)
    # same sums in a different order: float32 results agree to rounding
    np.testing.assert_allclose(ra, rb, rtol=1e-6, equal_nan=True)


def test_azimuth_wrap_and_layout():
    data = np.arange(10 * 6, dtype=np.float32).reshape(10, 6)
    patches, index = ml.polar_patches(data, (4, 4), (4, 2), wrap_azimuth=True)
    # rays: 0, 4, 8 (wraps to rays 8, 9, 0, 1); gates: 0, 2
    np.testing.assert_array_equal(np.unique(index.table[:, 1]), [0, 4, 8])
    np.testing.assert_array_equal(np.unique(index.table[:, 2]), [0, 2])
    last = patches[(index.table[:, 1] == 8) & (index.table[:, 2] == 0)][0]
    np.testing.assert_array_equal(last, data[[8, 9, 0, 1], :4])
    assert index.shape == (10, 6) and len(index) == 6

    patches, index = ml.polar_patches(data, (4, 4), (4, 4), wrap_azimuth=False)
    # the last patch ends at the last ray / gate
    np.testing.assert_array_equal(np.unique(index.table[:, 1]), [0, 4, 6])
    np.testing.assert_array_equal(np.unique(index.table[:, 2]), [0, 2])


@pytest.mark.parametrize("engine", ENGINES)
def test_fill_outside_sweep(engine):
    data = np.ones((5, 3), dtype=np.float32)
    patches, index = ml.polar_patches(
        data, (8, 6), wrap_azimuth=False, fill_value=-1.0, engine=engine
    )
    assert patches.shape == (1, 8, 6)
    np.testing.assert_array_equal(patches[0, :5, :3], 1.0)
    np.testing.assert_array_equal(patches[0, 5:], -1.0)
    np.testing.assert_array_equal(patches[0, :, 3:], -1.0)
    np.testing.assert_array_equal(ml.reassemble(patches, index, engine=engine), data)


def test_windows_sum_to_one_at_half_overlap():
    for blend in ("cosine", "linear"):
        w = patch_mod._window(16, blend)
        assert np.all(w > 0)
        np.testing.assert_allclose(w[:8] + w[8:], 1.0)


@pytest.mark.parametrize("engine", ENGINES)
def test_blending_weights(engine):
    # four rays, patches of two rays at every ray: each ray is covered by the
    # second row of one patch and the first row of the next one
    data = np.zeros((4, 1), dtype=np.float32)
    patches, index = ml.polar_patches(data, (2, 1), (1, 1), engine=engine)
    pred = np.arange(4, dtype=np.float32)[:, None, None] * np.ones_like(patches)
    out = ml.reassemble(pred, index, blend="linear", engine=engine)
    # ray a: patches a-1 (row 1) and a (row 0), equal weights -> mean
    np.testing.assert_allclose(out[:, 0], [1.5, 0.5, 1.5, 2.5])
    # NaN pixels are skipped; a gate without finite pixels is NaN
    pred[0] = np.nan
    out = ml.reassemble(pred, index, engine=engine)
    np.testing.assert_allclose(out[:, 0], [3.0, 1.0, 1.5, 2.5])
    pred[1] = np.nan
    out = ml.reassemble(pred, index, engine=engine)
    assert np.isnan(out[1, 0])


@pytest.mark.parametrize("engine", ENGINES)
def test_dataarray_in_out(engine):
    da = sweep()
    stacked = xr.concat([da, da + 1], dim="time").transpose("time", ...)
    patches, index = ml.polar_patches(stacked, 32, engine=engine)
    assert patches.shape[1:] == (2, 32, 32)
    assert index.kind == "dataarray" and index.lead_dims == ("time",)
    out = ml.reassemble(patches, index, engine=engine)
    xr.testing.assert_identical(out, stacked.drop_attrs())
    # a model output with one channel and model attributes
    pred = patches[:, :1] * 0 + 1
    out = ml.reassemble(pred, index, attrs={"ml_model": "m"}, engine=engine)
    assert out.dims == ("channel", "azimuth", "range")
    assert out.attrs["ml_model"] == "m"
    np.testing.assert_array_equal(out.x, da.x)
    out = ml.reassemble(pred[:, :, None], index)
    assert out.dims == ("channel_0", "channel_1", "azimuth", "range")


def test_dataset_in_out():
    ds = xr.Dataset({"DBZH": sweep(), "ZDR": sweep(seed=1) / 10})
    ds["scalar"] = 1.0
    patches, index = ml.polar_patches(ds, (32, 50))
    assert patches.shape[1:] == (2, 32, 50)
    assert list(index.lead_coords["variable"]) == ["DBZH", "ZDR"]
    out = ml.reassemble(patches, index)
    assert set(out.data_vars) == {"DBZH", "ZDR"}
    np.testing.assert_array_equal(out.DBZH, ds.DBZH)
    np.testing.assert_array_equal(out.ZDR, ds.ZDR.astype(np.float32))
    pred = ml.reassemble(patches[:, :1], index, name="prob", attrs={"units": "1"})
    assert list(pred.data_vars) == ["prob"]
    assert pred.prob.attrs["units"] == "1"
    patches, index = ml.polar_patches(ds, 32, variables=["ZDR"])
    assert patches.shape[1] == 1


def volume():
    """Three sweeps of different shapes; the last one lacks ZDR."""
    nodes = {"/": xr.Dataset(coords={"latitude": 33.9, "longitude": -88.3})}
    for k, (nray, ngate) in enumerate([(360, 120), (720, 90), (360, 60)]):
        ds = xr.Dataset({"DBZH": sweep(nray, ngate, seed=k)})
        if k < 2:
            ds["ZDR"] = sweep(nray, ngate, seed=10 + k) / 10
        nodes[f"sweep_{k}"] = ds
    return xr.DataTree.from_dict(nodes)


@pytest.mark.parametrize("engine", ENGINES)
def test_datatree_in_out(engine):
    tree = volume()
    patches, index = ml.polar_patches(tree, (48, 32), engine=engine)
    assert index.kind == "datatree"
    assert index.paths == ["/sweep_0", "/sweep_1"]  # sweep_2 lacks ZDR
    assert np.unique(index.table[:, 0]).tolist() == [0, 1]
    out = ml.reassemble(patches, index, engine=engine)
    assert isinstance(out, xr.DataTree)
    assert "latitude" in out.coords
    for path in index.paths:
        for v in ("DBZH", "ZDR"):
            np.testing.assert_array_equal(
                out[path][v], tree[path][v].astype(np.float32)
            )
    # only DBZH: all three sweeps
    patches, index = ml.polar_patches(tree, 32, variables=["DBZH"], engine=engine)
    assert len(index.paths) == 3
    with pytest.raises(ValueError, match="several sweeps"):
        index.shape


def test_patch_errors():
    data = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="engine"):
        ml.polar_patches(data, 4, engine="gpu")
    with pytest.raises(ValueError, match="size"):
        ml.polar_patches(data, (4, 0))
    with pytest.raises(ValueError, match="size"):
        ml.polar_patches(data, (4, 4, 4))
    with pytest.raises(ValueError, match="two dimensions"):
        ml.polar_patches(np.zeros(5), 4)
    with pytest.raises(ValueError, match="dimensions"):
        ml.polar_patches(xr.DataArray(data, dims=("x", "y")), 4)
    with pytest.raises(ValueError, match="no field"):
        ml.polar_patches(xr.Dataset({"a": ("x", np.zeros(3))}), 4)
    with pytest.raises(ValueError, match="no sweep"):
        ml.polar_patches(volume(), 4, variables=["KDP"])
    tree = xr.DataTree.from_dict(
        {
            "a": xr.Dataset({"DBZH": sweep(20, 10)}),
            "b": xr.Dataset({"DBZH": xr.concat([sweep(20, 10)] * 2, "time")}),
        }
    )
    with pytest.raises(ValueError, match="same fields"):
        ml.polar_patches(tree, 4)
    patches, index = ml.polar_patches(data, 4)
    with pytest.raises(ValueError, match="blend"):
        ml.reassemble(patches, index, blend="max")
    with pytest.raises(ValueError, match="patches must be"):
        ml.reassemble(patches[:, :2], index)
    np.testing.assert_array_equal(ml.reassemble(patches, index, shape=(10, 10)), data)
    _, index = ml.polar_patches(sweep(20, 10), 4)
    with pytest.raises(ValueError, match="shape must match"):
        ml.reassemble(np.zeros((len(index), 4, 4)), index, shape=(10, 10))
    _, index = ml.polar_patches(volume(), 4)
    with pytest.raises(ValueError, match="single sweep"):
        ml.reassemble(np.zeros((len(index), 2, 4, 4)), index, shape=(10, 10))


def test_compiled_engine_unavailable(monkeypatch):
    monkeypatch.setattr(patch_mod, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError, match="not available"):
        ml.polar_patches(np.zeros((4, 4)), 2, engine="compiled")
    patches, index = ml.polar_patches(np.ones((4, 4)), 2)  # auto -> NumPy
    np.testing.assert_array_equal(ml.reassemble(patches, index), 1.0)


@compiled
def test_kernel_input_checks():
    k = patch_mod._patches
    data = [np.zeros((1, 4, 4), np.float32)]
    table = np.zeros((1, 3), np.int64)
    with pytest.raises(ValueError, match="no sweeps"):
        k.extract_float32([], table, 2, 2, True, 0.0)
    with pytest.raises(ValueError, match="positive"):
        k.extract_float32(data, table, 0, 2, True, 0.0)
    with pytest.raises(ValueError, match="same fields"):
        k.extract_float32(
            data + [np.zeros((2, 4, 4), np.float32)], table, 2, 2, True, 0
        )
    with pytest.raises(ValueError, match="empty"):
        k.extract_float32([np.zeros((1, 0, 4), np.float32)], table, 2, 2, True, 0)
    with pytest.raises(ValueError, match="table must"):
        k.extract_float32(data, np.zeros((1, 2), np.int64), 2, 2, True, 0)
    with pytest.raises(ValueError, match="does not exist"):
        k.extract_float32(data, np.array([[1, 0, 0]]), 2, 2, True, 0)
    p = np.zeros((1, 1, 2, 2), np.float32)
    w = np.ones(2)
    with pytest.raises(ValueError, match="patches must"):
        k.reassemble_float32(p[0], table, [(4, 4)], w, w, True)
    with pytest.raises(ValueError, match="no sweeps"):
        k.reassemble_float32(p, table, [], w, w, True)
    with pytest.raises(ValueError, match="weights"):
        k.reassemble_float32(p, table, [(4, 4)], np.ones(3), w, True)
    with pytest.raises(ValueError, match="number of patches"):
        k.reassemble_float32(p, np.zeros((2, 3), np.int64), [(4, 4)], w, w, True)
    with pytest.raises(ValueError, match="empty"):
        k.reassemble_float32(p, table, [(0, 4)], w, w, True)


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------


def test_normalize_round_trip():
    da = sweep()
    z = ml.normalize(da)
    assert abs(float(z.mean())) < 1e-5 and abs(float(z.std()) - 1) < 1e-5
    assert z.attrs["ml_normalization"] == "zscore"
    assert z.attrs["ml_units"] == "dBZ" and "units" not in z.attrs
    back = ml.denormalize(z.assign_attrs(ml_model="m"))
    assert back.attrs["ml_model"] == "m"  # model attributes are kept
    np.testing.assert_allclose(back, da, atol=1e-5, equal_nan=True)
    assert back.attrs["units"] == "dBZ"
    assert back.attrs["standard_name"] == "equivalent_reflectivity_factor"

    m = ml.normalize(da, "minmax", fill_value=0.0)
    assert float(m.min()) == 0.0 and float(m.max()) == pytest.approx(1.0)
    assert not m.isnull().any()

    fixed = ml.normalize(da, offset=0.0, scale=60.0, clip=(0.0, 1.0))
    assert float(fixed.max()) <= 1.0 and float(fixed.min()) >= 0.0
    assert fixed.attrs["ml_scale"] == 60.0


def test_normalize_dataset_and_arrays():
    ds = xr.Dataset({"DBZH": sweep(), "ZDR": sweep(seed=1) / 10})
    out = ml.normalize(ds, offset={"DBZH": 0.0, "ZDR": 1.0}, scale={"DBZH": 10.0})
    assert out.DBZH.attrs["ml_offset"] == 0.0 and out.DBZH.attrs["ml_scale"] == 10.0
    assert out.ZDR.attrs["ml_offset"] == 1.0  # scale from the data
    back = ml.denormalize(out)
    np.testing.assert_allclose(back.ZDR, ds.ZDR, atol=1e-5, equal_nan=True)

    arr = np.array([1.0, 2.0, np.nan, 3.0])
    z = ml.normalize(arr, fill_value=-9.0)
    assert z[2] == -9.0
    np.testing.assert_allclose(
        ml.denormalize(ml.normalize(arr, offset=2.0, scale=0.5), offset=2.0, scale=0.5),
        arr,
    )
    np.testing.assert_array_equal(ml.normalize(np.full(3, 4.0)), 0.0)  # constant
    with pytest.raises(ValueError, match="offset and scale"):
        ml.denormalize(arr)
    with pytest.raises(ValueError, match="all-NaN"):
        ml.normalize(np.full(3, np.nan))
    with pytest.raises(ValueError, match="method"):
        ml.normalize(arr, "robust")


# --------------------------------------------------------------------------
# models and registry
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch, tmp_path):
    monkeypatch.setattr(model_mod, "_registered", {})
    monkeypatch.setenv("RADARX_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("RADARX_MODEL_REGISTRY", raising=False)


def onnx_model(path, scale=2.0, two_inputs=False):
    """A tiny ONNX network y = relu(scale * x + 1) built with onnx.helper."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 1, None, None])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 1, None, None])
    inits = [
        helper.make_tensor("scale", TensorProto.FLOAT, [], [scale]),
        helper.make_tensor("one", TensorProto.FLOAT, [], [1.0]),
    ]
    nodes = [
        helper.make_node("Mul", ["x", "scale"], ["xs"]),
        helper.make_node("Add", ["xs", "one" if not two_inputs else "z"], ["xa"]),
        helper.make_node("Relu", ["xa"], ["y"]),
    ]
    inputs = [x]
    if two_inputs:
        inputs.append(helper.make_tensor_value_info("z", TensorProto.FLOAT, [1]))
    graph = helper.make_graph(nodes, "tiny", inputs, [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8  # readable by older ONNX Runtime releases too
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def tiny(tmp_path):
    pytest.importorskip("onnxruntime")
    return onnx_model(tmp_path / "tiny.onnx")


def test_register_load_run(tiny):
    path, sha = tiny
    entry = ml.register_model(
        "tiny", path, sha, "MIT", "Doe (2026): Tiny.", task="test", inputs={"x": "f"}
    )
    assert entry["name"] == "tiny" and entry["inputs"] == {"x": "f"}
    names = [m["name"] for m in ml.list_models()]
    assert "tiny" in names
    row = [m for m in ml.list_models() if m["name"] == "tiny"][0]
    assert row["licence"] == "MIT" and row["citation"] == "Doe (2026): Tiny."

    model = ml.load_model("tiny", providers="cpu")
    assert isinstance(model, ml.Model)
    assert model.providers == ["CPUExecutionProvider"]
    assert model.inputs == {"x": "float32[N,1,?,?]"}
    text = repr(model)
    assert "licence:   MIT" in text and "Doe (2026)" in text and "task:" in text
    assert model.attrs == {
        "ml_model": "tiny",
        "ml_model_version": "1",
        "ml_model_licence": "MIT",
        "ml_model_citation": "Doe (2026): Tiny.",
    }
    x = np.linspace(-2, 2, 2 * 9, dtype=np.float64).reshape(2, 1, 3, 3)
    expected = np.maximum(2 * x + 1, 0).astype(np.float32)
    np.testing.assert_allclose(model.run({"x": x})["y"], expected, rtol=1e-6)
    np.testing.assert_allclose(model(x)["y"], expected, rtol=1e-6)  # one input
    np.testing.assert_allclose(model.run(x, batch_size=1)["y"], expected, rtol=1e-6)
    np.testing.assert_allclose(model.run(x, ["y"])["y"], expected, rtol=1e-6)
    with pytest.raises(ValueError, match="missing"):
        model.run({"q": x})
    assert ml.load_model("tiny").providers[-1] == "CPUExecutionProvider"
    assert ml.load_model("tiny", providers=["CPUExecutionProvider"]).run(x)
    with pytest.raises(ValueError, match="not available"):
        ml.load_model("tiny", providers="NoSuchExecutionProvider")


def test_model_on_patches(tiny):
    path, sha = tiny
    ml.register_model("tiny", path, sha, "MIT", "Doe (2026)")
    model = ml.load_model("tiny")
    da = sweep(90, 64).fillna(0.0)
    patches, index = ml.polar_patches(da, 32)
    out = model.run({"x": patches[:, None]})["y"][:, 0]
    result = ml.reassemble(out, index, attrs=model.attrs)
    np.testing.assert_allclose(result, np.maximum(2 * da + 1, 0), rtol=1e-6)
    assert result.attrs["ml_model_licence"] == "MIT"


def test_two_inputs_need_dict(tmp_path):
    pytest.importorskip("onnxruntime")
    path, sha = onnx_model(tmp_path / "two.onnx", two_inputs=True)
    ml.register_model("two", path, sha, "MIT", "x")
    model = ml.load_model("two")
    with pytest.raises(ValueError, match="pass a dict"):
        model.run(np.zeros((1, 1, 2, 2)))
    with pytest.raises(ValueError, match="first dimension"):
        model.run({"x": np.zeros((2, 1, 2, 2)), "z": np.ones(1)}, batch_size=1)


def test_sha256_mismatch_and_registry_errors(tiny, tmp_path):
    path, sha = tiny
    ml.register_model("bad", path, "0" * 64, "MIT", "x")
    with pytest.raises(ValueError, match="does not match"):
        ml.load_model("bad")
    ml.register_model("nofile", tmp_path / "missing.onnx", None, "MIT", "x")
    with pytest.raises(FileNotFoundError):
        ml.load_model("nofile")
    with pytest.raises(ValueError, match="already registered"):
        ml.register_model("bad", path, sha, "MIT", "x")
    ml.register_model("bad", path, f"sha256:{sha.upper()}", "MIT", "x", overwrite=True)
    assert ml.load_model("bad").run(np.zeros((1, 1, 1, 1)))["y"][0, 0, 0, 0] == 1.0
    with pytest.raises(ValueError, match="sha256"):
        ml.register_model("web", "https://example.org/m.onnx", None, "MIT", "x")
    with pytest.raises(KeyError, match="unknown model"):
        ml.load_model("no-such-model")


def test_unregistered_path_warns(tiny):
    path, _ = tiny
    with pytest.warns(UserWarning, match="not in the model registry"):
        model = ml.load_model(path)
    assert model.name == "tiny" and "licence:   unknown" in repr(model)


def test_registry_file_and_download(tiny, tmp_path, monkeypatch):
    path, sha = tiny
    toml = tmp_path / "models.toml"
    toml.write_text(
        "[models.from-file]\n"
        f'url = "{path.as_uri()}"\n'
        f'sha256 = "{sha}"\n'
        'licence = "CC-BY-4.0"\n'
        'citation = "Roe (2026)"\n'
        'version = "3"\n'
        "[models.from-web]\n"
        'url = "https://example.org/tiny.onnx"\n'
        f'sha256 = "{sha}"\n'
        'licence = "MIT"\n'
        'citation = "Roe (2026)"\n'
    )
    monkeypatch.setenv("RADARX_MODEL_REGISTRY", str(toml))
    assert ml.load_model("from-file").version == "3"

    calls = []

    def retrieve(url, known_hash, fname, path):
        calls.append((url, known_hash, fname, path))
        target = path / fname
        target.write_bytes(tiny[0].read_bytes())
        return str(target)

    import pooch

    monkeypatch.setattr(pooch, "retrieve", retrieve)
    model = ml.load_model("from-web")
    assert calls[-1][1] == f"sha256:{sha}"
    assert calls[-1][2] == "from-web-v1.onnx"
    assert calls[-1][3] == tmp_path / "cache" / "models"
    assert model.path.parent == tmp_path / "cache" / "models"
    model = ml.load_model("from-web", cache=False)
    assert model.path is None and not calls[-1][3].exists()  # temporary folder

    toml.write_text('[models.broken]\nurl = "x"\n')
    with pytest.raises(ValueError, match="lacks sha256, licence, citation"):
        ml.list_models()


def test_cache_dir_default(monkeypatch, tmp_path):
    import pooch

    monkeypatch.delenv("RADARX_CACHE_DIR")
    monkeypatch.setattr(pooch, "os_cache", lambda name: tmp_path / name)
    assert model_mod._cache_dir() == tmp_path / "radarx" / "models"


def test_shipped_registry_is_valid():
    for entry in ml.list_models():
        assert entry["licence"] and entry["citation"] and entry["url"]
        assert len(model_mod._registry()[entry["name"]]["sha256"]) == 64


def test_missing_onnxruntime(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(ImportError, match=r"pip install radarx\[ml\]"):
        ml.load_model("anything")


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kgwx(tmp_path_factory):
    """Squall line (KGWX, 30 March 2022), from AWS."""
    import xradar as xd

    from radarx.io.aws_data import download_file

    try:
        path = download_file(
            "unidata-nexrad-level2",
            "2022/03/30/KGWX/KGWX20220330_234639_V06",
            str(tmp_path_factory.mktemp("nexrad")),
        )
    except Exception as err:  # noqa: BLE001  # pragma: no cover - network
        pytest.skip(f"NEXRAD data not available: {err}")
    return xd.io.open_nexradlevel2_datatree(path)


def test_nexrad_volume_round_trip(kgwx):
    patches, index = ml.polar_patches(kgwx, (64, 128), variables=["DBZH", "ZDR"])
    assert len(index.paths) >= 5  # the split-cut Doppler sweeps lack ZDR
    out = ml.reassemble(patches, index)
    for path in index.paths:
        for v in ("DBZH", "ZDR"):
            np.testing.assert_array_equal(out[path][v].values, kgwx[path][v].values)
        np.testing.assert_array_equal(out[path].azimuth, kgwx[path].azimuth)
    if patch_mod.HAS_COMPILED_KERNEL:
        np_patches, _ = ml.polar_patches(
            kgwx, (64, 128), variables=["DBZH", "ZDR"], engine="numpy"
        )
        np.testing.assert_array_equal(np_patches, patches)
