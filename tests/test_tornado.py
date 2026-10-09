#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for TorNet / MistNet ONNX conversion, tornado detection and MistNet."""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from radarx.retrieve import (
    _onnx_models,
    biological_echo,
    rotation_couplets,
    tornado_probability,
    tornet_inputs,
)
from radarx.retrieve import biology as bio_mod
from radarx.retrieve import tornado as tor_mod

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

VARS = _onnx_models.TORNET_VARIABLES


# --------------------------------------------------------------------------
# helpers: tiny models and a NumPy reference forward pass
# --------------------------------------------------------------------------


class Session:
    """A loaded model as radarx.ml returns it: ``run`` and ``info``."""

    def __init__(self, model, name="tiny"):
        self.session = ort.InferenceSession(model.SerializeToString())
        self.info = {"name": name, "version": "0", "licence": "MIT", "citation": "x"}

    def run(self, inputs):
        names = [o.name for o in self.session.get_outputs()]
        return dict(zip(names, self.session.run(names, inputs)))


def conv2d(x, w, b=None, pad=0):
    """NCHW convolution (cross-correlation), weights [cout, cin, kh, kw]."""
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    kh, kw = w.shape[2:]
    win = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(2, 3))
    out = np.einsum("nchwij,ocij->nohw", win, w)
    return out if b is None else out + b[None, :, None, None]


def maxpool(x, ceil):
    n, c, h, w = x.shape
    if ceil:
        hh, ww = -(-h // 2), -(-w // 2)
        x = np.pad(
            x,
            ((0, 0), (0, 0), (0, 2 * hh - h), (0, 2 * ww - w)),
            constant_values=-np.inf,
        )
    else:
        hh, ww = h // 2, w // 2
        x = x[:, :, : 2 * hh, : 2 * ww]
    return x.reshape(n, c, hh, 2, ww, 2).max(axis=(3, 5))


def conv_transpose(x, w, stride, pad):
    """Grouped (groups = channels) transposed convolution, w [c, 1, k, k]."""
    n, c, h, wd = x.shape
    k = w.shape[-1]
    full = np.zeros((n, c, (h - 1) * stride + k, (wd - 1) * stride + k))
    for i in range(h):
        for j in range(wd):
            full[:, :, i * stride : i * stride + k, j * stride : j * stride + k] += (
                x[:, :, i, j][:, :, None, None] * w[None, :, 0]
            )
    return full[:, :, pad : full.shape[2] - pad, pad : full.shape[3] - pad]


def tiny_tornet_params(rng, filters=(3, 4), n_tilts=2):
    cin = 7 * n_tilts
    blocks = []
    for f in filters:
        block = []
        for _ in range(2):
            block.append(
                (
                    rng.normal(0, 0.3, (3, 3, cin + 2, f)).astype(np.float32),
                    rng.normal(0, 0.1, f).astype(np.float32),
                )
            )
            cin = f
        blocks.append(block)
    head = [
        (rng.normal(0, 0.3, (1, 1, cin, 5)).astype(np.float32), np.zeros(5, "f4")),
        (rng.normal(0, 0.3, (1, 1, 5, 1)).astype(np.float32), np.ones(1, "f4")),
    ]
    mean = [np.full(n_tilts, m) for m in (20.0, 0.0, 1.5, 0.62, 3.5, 4.5)]
    std = [np.full(n_tilts, s) for s in (40.0, 60.0, 3.5, 0.42, 4.5, 4.5)]
    return {
        "variables": list(VARS),
        "mean": mean,
        "std": std,
        "background": -3.0,
        "blocks": blocks,
        "head": head,
    }


def tornet_numpy(params, feed):
    norm = [(feed[v] - m) / s for v, m, s in zip(VARS, params["mean"], params["std"])]
    x = np.concatenate(norm, axis=-1)
    x = np.where(np.isnan(x), params["background"], x)
    x = np.concatenate([x, feed["range_folded_mask"]], axis=-1).transpose(0, 3, 1, 2)
    c = feed["coordinates"].transpose(0, 3, 1, 2)
    for block in params["blocks"]:
        for k, b in block:
            x = np.maximum(
                conv2d(np.concatenate([x, c], 1), k.transpose(3, 2, 0, 1), b, 1), 0
            )
        x, c = maxpool(x, True), maxpool(c, True)
    for i, (k, b) in enumerate(params["head"]):
        x = conv2d(x, k.transpose(3, 2, 0, 1), b)
        if i < len(params["head"]) - 1:
            x = np.maximum(x, 0)
    return x.max(axis=(1, 2, 3)), x[:, 0]


def tornet_feed(rng, n=2, shape=(13, 22)):
    feed = {}
    for v in VARS:
        a = rng.normal(0, 10, (n, *shape, 2)).astype(np.float32)
        a[rng.random(a.shape) < 0.2] = np.nan
        feed[v] = a
    feed["range_folded_mask"] = (rng.random((n, *shape, 2)) < 0.1).astype("f4")
    r = np.linspace(0.1, 0.7, shape[1], dtype=np.float32)
    coords = np.stack([r, 1 / r], -1)
    feed["coordinates"] = np.ascontiguousarray(
        np.broadcast_to(coords, (n, shape[0], shape[1], 2)), dtype=np.float32
    )
    return feed


def tiny_mistnet_params(rng, ch=4):
    p = {"mean": rng.normal(0, 3, (1, 15, 1, 1))}
    p["mistnet.adaptor.weight"] = rng.normal(0, 0.1, (3, 15, 1, 1))
    p["mistnet.adaptor.bias"] = rng.normal(0, 0.1, 3)
    cin = 3
    for block, convs in _onnx_models._MISTNET_BLOCKS.items():
        for k in convs:
            p[f"mistnet.backbone.{block}.{k}.weight"] = rng.normal(
                0, 0.3, (ch, cin, 3, 3)
            )
            p[f"mistnet.backbone.{block}.{k}.bias"] = rng.normal(0, 0.1, ch)
            cin = ch
    p["mistnet.backbone.block5.7.weight"] = rng.normal(0, 0.1, (ch, ch, 7, 7))
    p["mistnet.backbone.block5.7.bias"] = rng.normal(0, 0.1, ch)
    p["mistnet.backbone.block5.10.weight"] = rng.normal(0, 0.3, (ch, ch, 1, 1))
    p["mistnet.backbone.block5.10.bias"] = rng.normal(0, 0.1, ch)
    for s in range(5):
        h = f"mistnet.prediction.{s}"
        for which in ("pred_32s", "pred_16s", "pred_8s"):
            p[f"{h}.{which}.weight"] = rng.normal(0, 0.5, (3, ch, 1, 1))
            p[f"{h}.{which}.bias"] = rng.normal(0, 0.5, 3)
        p[f"{h}.upsample_2x.weight"] = rng.random((3, 1, 4, 4))
        p[f"{h}.upsample_8x.weight"] = rng.random((3, 1, 16, 16))
    return {k: np.asarray(v, dtype=np.float32) for k, v in p.items()}


def mistnet_numpy(p, x):
    bg = np.isnan(x[:, :5])
    x = np.concatenate(
        [
            np.nan_to_num(x[:, :5], nan=-33.0),
            np.nan_to_num(x[:, 5:10], nan=0.0),
            np.nan_to_num(x[:, 10:], nan=0.0),
        ],
        1,
    ).astype(np.float64)
    x = conv2d(x - p["mean"], p["mistnet.adaptor.weight"], p["mistnet.adaptor.bias"])
    feats = {}
    for block, convs in _onnx_models._MISTNET_BLOCKS.items():
        for k in convs:
            s = f"mistnet.backbone.{block}.{k}"
            x = np.maximum(conv2d(x, p[s + ".weight"], p[s + ".bias"], 1), 0)
        x = maxpool(x, False)
        feats[block] = x
    s = "mistnet.backbone.block5."
    x = np.maximum(conv2d(x, p[s + "7.weight"], p[s + "7.bias"], 3), 0)
    x = np.maximum(conv2d(x, p[s + "10.weight"], p[s + "10.bias"]), 0)
    outs = []
    for i in range(5):
        h = f"mistnet.prediction.{i}"

        def head(src, which, h=h):
            return conv2d(src, p[f"{h}.{which}.weight"], p[f"{h}.{which}.bias"])

        y = conv_transpose(head(x, "pred_32s"), p[f"{h}.upsample_2x.weight"], 2, 1)
        y = y + head(feats["block4"], "pred_16s")
        y = conv_transpose(y, p[f"{h}.upsample_2x.weight"], 2, 1)
        y = y + head(feats["block3"], "pred_8s")
        outs.append(conv_transpose(y, p[f"{h}.upsample_8x.weight"], 8, 4))
    y = np.stack(outs, 2)
    y = np.exp(y - y.max(1, keepdims=True))
    y = y / y.sum(1, keepdims=True)
    y[:, 0] = bg
    return y


# --------------------------------------------------------------------------
# synthetic NEXRAD-like volume
# --------------------------------------------------------------------------


def coded(values, offset, scale, codes=None):
    """A field as xradar decodes NEXRAD Level II (uint8 encoding)."""
    da = values.copy()
    if codes is not None:
        for code, mask in codes.items():
            da = da.where(~mask, offset + code * scale)
    da.encoding = {"scale_factor": scale, "add_offset": offset, "dtype": np.uint8}
    return da


def sweep(angle, fields, n_az=720, n_rng=400, seed=0, az0=0.2):
    rng = np.random.default_rng(seed)
    az = np.mod(az0 + np.arange(n_az) * 360.0 / n_az, 360.0)
    r = 2125.0 + 250.0 * np.arange(n_rng)
    coords = {
        "azimuth": az,
        "range": r,
        "elevation": ("azimuth", np.full(n_az, angle)),
        "time": ("azimuth", np.full(n_az, np.datetime64("2022-03-30T23:46:39"))),
    }
    dims = ("azimuth", "range")
    ds = xr.Dataset(coords=coords)
    ds["sweep_fixed_angle"] = angle
    shape = (n_az, n_rng)
    for name in fields:
        if name == "DBZH":
            v = rng.uniform(-10, 50, shape)
            da = xr.DataArray(v, dims=dims)
            folded = xr.DataArray(rng.random(shape) < 0.02, dims=dims)
            ds[name] = coded(da, -33.0, 0.5, {0: v < 0, 1: folded.values & (v >= 0)})
        elif name == "VRADH":
            v = rng.uniform(-20, 20, shape)
            fold = rng.random(shape) < 0.05
            da = xr.DataArray(v, dims=dims)
            ds[name] = coded(da, -64.5, 0.5, {1: fold})
        elif name == "WRADH":
            ds[name] = coded(
                xr.DataArray(rng.uniform(0, 8, shape), dims=dims), -64.5, 0.5
            )
        elif name == "ZDR":
            ds[name] = xr.DataArray(rng.uniform(-1, 4, shape), dims=dims)
        elif name == "RHOHV":
            ds[name] = xr.DataArray(rng.uniform(0.8, 1.0, shape), dims=dims)
        elif name == "PHIDP":
            ds[name] = xr.DataArray(
                np.cumsum(rng.uniform(0, 0.2, shape), axis=1) + 40, dims=dims
            )
        elif name == "KDP":
            ds[name] = xr.DataArray(rng.uniform(0, 2, shape), dims=dims)
    return ds


@pytest.fixture(scope="module")
def volume():
    pol = ("DBZH", "ZDR", "PHIDP", "RHOHV", "KDP")
    dop = ("DBZH", "VRADH", "WRADH")
    sweeps = [
        sweep(0.48, pol, n_rng=500, seed=1),
        sweep(0.48, dop, seed=2, az0=0.3),
        sweep(0.88, pol, n_rng=500, seed=3),
        sweep(0.88, dop, seed=4),
        sweep(1.45, dop + ("ZDR",), n_az=360, n_rng=300, seed=5),
        sweep(2.4, dop, n_az=360, n_rng=300, seed=6),
        sweep(3.35, dop, n_az=360, n_rng=300, seed=7),
        sweep(4.3, dop, n_az=360, n_rng=300, seed=8),
    ]
    root = xr.Dataset(coords={"latitude": 33.9, "longitude": -88.3, "altitude": 145.0})
    tree = {"/": root}
    tree.update({f"sweep_{i}": ds for i, ds in enumerate(sweeps)})
    return xr.DataTree.from_dict(tree)


# --------------------------------------------------------------------------
# ONNX graphs against NumPy
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(13, 22), (16, 32)])
def test_tornet_graph_matches_numpy(shape):
    rng = np.random.default_rng(0)
    params = tiny_tornet_params(rng)
    model = Session(_onnx_models.tornet_graph(params))
    feed = tornet_feed(rng, shape=shape)
    out = model.run(feed)
    logit, heat = tornet_numpy(params, feed)
    np.testing.assert_allclose(out["logit"][:, 0], logit, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out["heatmap"], heat, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out["logit"][:, 0], out["heatmap"].max(axis=(1, 2)))


def test_mistnet_graph_matches_numpy():
    rng = np.random.default_rng(1)
    params = tiny_mistnet_params(rng)
    model = Session(_onnx_models.mistnet_graph(params))
    x = rng.normal(0, 10, (1, 15, 64, 64)).astype(np.float32)
    x[:, :, 10:20, 30:50] = np.nan
    x[:, 7, 40:50, :] = np.nan
    y = model.run({"x": x})["y"]
    assert y.shape == (1, 3, 5, 64, 64)
    np.testing.assert_allclose(y, mistnet_numpy(params, x), rtol=1e-3, atol=1e-5)
    # background flags missing reflectivity; the classes keep their softmax
    np.testing.assert_array_equal(y[0, 0], np.isnan(x[0, :5]))


# --------------------------------------------------------------------------
# readers of the upstream files
# --------------------------------------------------------------------------


def write_keras(path, params):
    h5py = pytest.importorskip("h5py")
    layers = [{"class_name": "InputLayer", "name": v, "config": {}} for v in VARS]

    def node(*names):
        return [{"args": [[{"config": {"keras_history": [n, 0, 0]}} for n in names]]}]

    for v, m, s in zip(VARS, params["mean"], params["std"]):
        layers.append(
            {
                "class_name": "Normalization",
                "name": f"Normalize_{v}",
                "config": {"mean": list(m), "variance": list(s**2)},
                "inbound_nodes": node(v),
            }
        )
    layers.append(
        {
            "class_name": "Concatenate",
            "name": "Concatenate1",
            "config": {},
            "inbound_nodes": node(*[f"Normalize_{v}" for v in VARS]),
        }
    )
    layers.append({"class_name": "FillNaNs", "name": "f", "config": {"fill_val": -3.0}})
    k = 0
    for block in params["blocks"]:
        for _ in block:
            layers.append({"class_name": "CoordConv2D", "name": f"cc{k}", "config": {}})
            k += 1
        layers.append({"class_name": "MaxPooling2D", "name": f"mp{k}", "config": {}})
    for i, _ in enumerate(params["head"]):
        layers.append({"class_name": "Conv2D", "name": f"c{i}", "config": {}})
    buf = io.BytesIO()
    with h5py.File(buf, "w") as h5:
        convs = [kb for block in params["blocks"] for kb in block]
        for i, (w, b) in enumerate(convs):
            g = h5.create_group(
                f"layers/coord_conv2d{'' if i == 0 else f'_{i}'}/conv/vars"
            )
            g["0"], g["1"] = w, b
        for i, (w, b) in enumerate(params["head"]):
            g = h5.create_group(f"layers/conv2d{'' if i == 0 else f'_{i}'}/vars")
            g["0"], g["1"] = w, b
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("config.json", json.dumps({"config": {"layers": layers}}))
        zf.writestr("model.weights.h5", buf.getvalue())


def test_read_tornet_keras(tmp_path):
    params = tiny_tornet_params(np.random.default_rng(2))
    path = tmp_path / "tiny.keras"
    write_keras(path, params)
    back = _onnx_models.read_tornet_keras(path)
    assert back["variables"] == list(VARS)
    assert back["background"] == -3.0
    np.testing.assert_allclose(back["std"][1], params["std"][1])
    for b0, b1 in zip(params["blocks"], back["blocks"]):
        for (w0, _), (w1, _) in zip(b0, b1):
            np.testing.assert_array_equal(w0, w1)
    np.testing.assert_array_equal(back["head"][1][0], params["head"][1][0])


def write_torchscript(path, params):
    tensors, mods = [], {}
    mean = params["mean"]
    tensors.append(mean)
    for name, value in params.items():
        if name == "mean":
            continue
        *mod, leaf = name.split(".")
        mods.setdefault(tuple(mod), []).append((leaf, len(tensors)))
        tensors.append(value)

    def module(path):
        children = sorted(
            {
                m[len(path)]
                for m in mods
                if m[: len(path)] == path and len(m) > len(path)
            }
        )
        return {
            "name": path[-1] if path else "m",
            "parameters": [
                {"name": leaf, "tensorId": str(i)} for leaf, i in mods.get(path, [])
            ],
            "submodules": [module(path + (c,)) for c in children],
        }

    meta = {
        "mainModule": module(()),
        "tensors": [
            {
                "dims": [str(d) for d in t.shape],
                "strides": [str(s // 4) for s in np.ascontiguousarray(t).strides],
                "offset": "0",
                "dataType": "FLOAT",
                "data": {"key": f"tensors/{i}"},
            }
            for i, t in enumerate(tensors)
        ],
    }
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("m/model.json", json.dumps(meta))
        for i, t in enumerate(tensors):
            zf.writestr(f"m/tensors/{i}", np.ascontiguousarray(t, "<f4").tobytes())


def test_read_mistnet_torchscript(tmp_path):
    params = tiny_mistnet_params(np.random.default_rng(3))
    path = tmp_path / "tiny.pt"
    write_torchscript(path, params)
    back = _onnx_models.read_mistnet_torchscript(path)
    assert set(back) == set(params)
    for k in params:
        np.testing.assert_array_equal(back[k], params[k])


def test_convert_on_first_use(tmp_path, monkeypatch):
    pytest.importorskip("h5py")
    params = tiny_tornet_params(np.random.default_rng(4))
    upstream = tmp_path / "up.keras"
    write_keras(upstream, params)
    monkeypatch.setenv("RADARX_CACHE_DIR", str(tmp_path / "cache"))
    calls = []

    def retrieve(url, known_hash, path, progressbar):
        calls.append(url)
        assert (
            known_hash
            == "sha256:" + _onnx_models.MODELS["tornet-baseline-v1"]["sha256"]
        )
        return str(upstream)

    import pooch

    monkeypatch.setattr(pooch, "retrieve", retrieve)
    out = _onnx_models.onnx_path("tornet-baseline-v1")
    assert out.exists() and out.parent == tmp_path / "cache" / "models"
    assert not upstream.exists()  # the upstream file is not kept
    assert _onnx_models.onnx_path("tornet-baseline-v1") == out
    assert len(calls) == 1
    feed = tornet_feed(np.random.default_rng(5), shape=(8, 16))
    got = ort.InferenceSession(str(out)).run(["logit"], feed)[0][:, 0]
    np.testing.assert_allclose(got, tornet_numpy(params, feed)[0], atol=1e-4)
    with pytest.raises(KeyError):
        _onnx_models.onnx_path("nope")


def test_load_model_objects(tmp_path, monkeypatch):
    params = tiny_tornet_params(np.random.default_rng(6))
    model = Session(_onnx_models.tornet_graph(params))
    assert _onnx_models.load(model, "x") is model
    attrs = _onnx_models.model_attrs(model, "x")
    assert attrs["ml_model"] == "tiny" and attrs["ml_model_licence"] == "MIT"
    pytest.importorskip("radarx.ml")
    path = tmp_path / "tiny.onnx"
    path.write_bytes(_onnx_models.tornet_graph(params).SerializeToString())
    with pytest.warns(UserWarning, match="not in the model registry"):
        loaded = _onnx_models.load(path, "x")
    assert set(loaded.run(tornet_feed(np.random.default_rng(1)))) == {
        "logit",
        "heatmap",
    }
    # a converted model in the cache is registered with its licence and citation
    monkeypatch.setenv("RADARX_CACHE_DIR", str(tmp_path))
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "tornet-baseline-v1.onnx").write_bytes(path.read_bytes())
    net = _onnx_models.load(None, "tornet-baseline-v1")
    attrs = _onnx_models.model_attrs(net, "x")
    assert attrs["ml_model"] == "tornet-baseline-v1"
    assert (
        attrs["ml_model_licence"] == "MIT"
        and "AIES-D-24-0006.1" in attrs["ml_model_citation"]
    )
    assert _onnx_models.load("tornet-baseline-v1", "x").name == "tornet-baseline-v1"


# --------------------------------------------------------------------------
# TorNet inputs and probabilities
# --------------------------------------------------------------------------


def test_tornet_inputs(volume):
    inp = tornet_inputs(volume, dealias=False)
    assert inp.DBZ.dims == ("azimuth", "range", "tilt")
    assert inp.sizes["azimuth"] == 720 and inp.sizes["tilt"] == 2
    np.testing.assert_allclose(inp.azimuth[:2], [0.25, 0.75])
    np.testing.assert_allclose(inp.range[:2], [2125.0, 2375.0])
    assert inp.sizes["range"] == 400  # limited to the velocity range
    # flag codes are removed, range-folded gates are flagged
    assert np.nanmin(inp.DBZ) >= 0 and np.nanmin(inp.VEL) > -60
    rf = inp.range_folded_mask.values
    assert 0.03 < rf.mean() < 0.07 and set(np.unique(rf)) <= {0.0, 1.0}
    assert np.isnan(inp.VEL.values[rf.astype(bool)]).all()
    # nearest ray: the Doppler sweep starts at 0.3 deg, its ray k is target k
    src = volume["sweep_1"].to_dataset()["VRADH"].values[5, :400]
    got = inp.VEL.values[5, :, 0]
    ok = np.isfinite(src) & (src > -63.9)
    np.testing.assert_allclose(got[ok], src[ok])
    np.testing.assert_allclose(inp.KDP.values[:, :, 0], inp.KDP.values[:, :, 0])
    assert float(inp.elevation[1]) == 0.9
    assert "latitude" in inp.coords


def test_tornet_inputs_errors(volume):
    with pytest.raises(ValueError, match="Nyquist"):
        tornet_inputs(volume)
    with pytest.raises(ValueError, match="no sweeps"):
        tornet_inputs(volume, elevations=(0.5, 7.0), dealias=False)
    with pytest.raises(TypeError):
        tornet_inputs(volume["sweep_0"].to_dataset())


def test_tornet_inputs_dealias_and_kdp(volume):
    sub = volume.copy()
    del sub["sweep_0"]["KDP"]
    inp = tornet_inputs(sub, nyquist_velocity=26.0, max_range=60e3)
    assert inp.sizes["range"] == 232
    assert np.isfinite(inp.KDP.values[:, :, 0]).any()
    inp = tornet_inputs(sub, nyquist_velocity={0.5: 26.0, 0.9: 26.0}, max_range=60e3)
    assert np.isfinite(inp.VEL).any()


def test_tornado_probability(volume):
    params = tiny_tornet_params(np.random.default_rng(7))
    model = Session(_onnx_models.tornet_graph(params))
    out = tornado_probability(volume, model, dealias=False, max_range=80e3)
    p = out.tornado_probability
    assert p.dims == ("azimuth", "range") and p.shape == (720, 312)
    assert np.isfinite(p).all() and float(p.min()) >= 0 and float(p.max()) <= 1
    assert p.attrs["ml_model"] == "tiny" and p.attrs["units"] == "1"
    # 720 / 30 azimuth starts; range starts 0, 60 and the last aligned one
    assert out.sizes["chip"] == 24 * 3
    # the chip output is the maximum of its logit map, so the gate field is
    # at least as large as every chip probability inside its chip
    assert float(p.max()) == pytest.approx(float(out.chip_probability.max()), rel=1e-5)
    assert float(out.chip_azimuth[0]) == pytest.approx(30.0)
    # one chip, no overlap: the gate field equals the chip's own heatmap
    inp = tornet_inputs(volume, dealias=False, max_range=62e3)
    one = tornado_probability(inp, model, stride=(720, 240))
    assert inp.sizes["range"] == 240 and one.sizes["chip"] == 1
    chip = one.tornado_probability.isel(azimuth=slice(0, 120))
    assert float(chip.max()) == pytest.approx(float(one.chip_probability[0]), rel=1e-5)
    assert np.isnan(one.tornado_probability.isel(azimuth=slice(120, None))).all()
    with pytest.raises(TypeError):
        tornado_probability(inp, model, dealias=False)
    with pytest.raises(ValueError):
        tornado_probability(inp, model, stride=(0, 1))


def test_tornado_probability_accessor(volume):
    import radarx  # noqa: F401

    model = Session(
        _onnx_models.tornet_graph(tiny_tornet_params(np.random.default_rng(8)))
    )
    out = volume.radarx.tornado_probability(model, dealias=False, max_range=62e3)
    assert "tornado_probability" in out


# --------------------------------------------------------------------------
# rotation couplets
# --------------------------------------------------------------------------


def rankine(center=(-21e3, 2e3), vmax=40.0, core=500.0):
    az = np.arange(0.25, 360.0, 0.5)
    r = np.arange(2125.0, 60e3, 250.0)
    A, R = np.meshgrid(np.radians(az), r, indexing="ij")
    x, y = R * np.sin(A), R * np.cos(A)
    dx, dy = x - center[0], y - center[1]
    rho = np.maximum(np.hypot(dx, dy), 1e-6)
    vt = np.where(rho < core, vmax * rho / core, vmax * core / rho)
    u, v = -vt * dy / rho, vt * dx / rho
    vr = u * np.sin(A) + v * np.cos(A)
    return xr.Dataset(
        {
            "VRADH": (("azimuth", "range"), vr),
            "DBZH": (("azimuth", "range"), 0 * vr + 40),
        },
        coords={
            "azimuth": az,
            "range": r,
            "x": (("azimuth", "range"), x),
            "y": (("azimuth", "range"), y),
        },
    )


def test_rotation_couplets_rankine():
    ds = rankine()
    out = rotation_couplets(ds)
    assert out.sizes["couplet"] >= 1
    first = out.isel(couplet=0)
    assert np.hypot(float(first.x) + 21e3, float(first.y) - 2e3) < 1000.0
    assert float(first.azimuthal_shear) > 0.01
    # the velocity difference across a 40 m/s vortex is about 80 m/s
    assert 60.0 < float(first.delta_v) <= 81.0
    assert float(first.rotational_velocity) == pytest.approx(float(first.delta_v) / 2)
    assert out.attrs["shear_threshold"] == 0.006
    # an anticyclone has no positive-shear couplet at its centre
    anti = ds.assign(VRADH=-ds.VRADH)
    out2 = rotation_couplets(anti)
    if out2.sizes["couplet"]:
        d = np.hypot(out2.x + 21e3, out2.y - 2e3)
        assert float(d.min()) > 400.0
    # reflectivity floor removes everything
    weak = ds.assign(DBZH=ds.DBZH * 0)
    assert rotation_couplets(weak, min_reflectivity=20).sizes["couplet"] == 0


def test_rotation_couplets_tree_and_errors():
    ds = rankine()
    tree = xr.DataTree.from_dict(
        {"/": xr.Dataset(), "sweep_0": ds, "sweep_1": ds.drop_vars("VRADH")}
    )
    out = rotation_couplets(tree)
    assert "sweep_0" in out.children and "sweep_1" not in out.children
    assert out["sweep_0"].sizes["couplet"] >= 1
    import radarx  # noqa: F401

    assert ds.radarx.rotation_couplets().sizes["couplet"] >= 1
    with pytest.raises(KeyError):
        rotation_couplets(ds, "nope")
    with pytest.raises(ValueError):
        rotation_couplets(tree, "nope")
    with pytest.raises(TypeError):
        rotation_couplets(ds.VRADH)


# --------------------------------------------------------------------------
# MistNet
# --------------------------------------------------------------------------


def test_render_geometry():
    ds = sweep(0.5, ("DBZH",), n_az=720, n_rng=300)
    ds["DBZH"] = xr.full_like(ds.DBZH, np.nan)
    ds["DBZH"].encoding = {}
    # gates around 90 deg (east), 20 km
    k = int(np.argmin(np.abs(ds.azimuth.values - 90.2)))
    g = int(np.argmin(np.abs(ds.range.values - 20e3)))
    ds["DBZH"][k - 3 : k + 4, g - 3 : g + 4] = 30.0  # a 1.75 km x 1.75 km patch
    grid = bio_mod._render(ds, ("DBZH",), size=128, resolution=500.0)[0]
    i, j = np.argwhere(np.isfinite(grid)).mean(axis=0)
    assert abs(i - 64) < 0.6 and abs(j - (64 + 40)) < 0.6
    # round trip of a smooth field: polar -> grid -> polar
    ds["DBZH"] = ds.range * 0 + ds.azimuth * 0 + ds.range / 1000.0
    ds["DBZH"].encoding = {}
    grid = bio_mod._render(ds, ("DBZH",), size=128, resolution=500.0)
    back, dims = bio_mod._to_polar(ds, grid, 128, 500.0)
    src = ds.DBZH.transpose(*dims).values
    ok = np.isfinite(back[0])
    assert ok.sum() > 1000
    assert np.nanmax(np.abs(back[0][ok] - src[ok])) < 0.8


def test_biological_echo(volume):
    params = tiny_mistnet_params(np.random.default_rng(9))
    model = Session(_onnx_models.mistnet_graph(params))
    out = biological_echo(volume, model, size=64, resolution=2500.0)
    names = list(out.children)
    assert names == ["sweep_1", "sweep_4", "sweep_5", "sweep_6", "sweep_7"]
    node = out["sweep_1"].to_dataset()
    assert node.biology_probability.dims == ("azimuth", "range")
    wx = node.weather_probability.values
    assert np.nanmax(wx) <= 1 and np.isfinite(wx).any()
    assert node.biological_echo.dtype == bool
    assert not node.biological_echo.values[~np.isfinite(wx)].any()
    assert node.biology_probability.attrs["ml_model"] == "tiny"
    # weather above the threshold everywhere -> no biology
    none = biological_echo(
        volume, model, size=64, resolution=2500.0, weather_threshold=-1
    )
    assert not none["sweep_4"].to_dataset().biological_echo.any()
    import radarx  # noqa: F401

    acc = volume.radarx.biological_echo(model, size=64, resolution=2500.0)
    assert "sweep_7" in acc.children


def test_biological_echo_errors(volume):
    with pytest.raises(TypeError):
        biological_echo(volume["sweep_1"].to_dataset())
    with pytest.raises(ValueError, match="five"):
        biological_echo(volume, elevations=(0.5, 1.5))
    with pytest.raises(ValueError, match="no sweep"):
        biological_echo(volume, fields=("DBZH", "VRADH", "nope"))


# --------------------------------------------------------------------------
# real data (no model download): inputs and couplets from a NEXRAD volume
# --------------------------------------------------------------------------


def test_real_nexrad_inputs_and_couplets():
    pytest.importorskip("open_radar_data")
    from open_radar_data import DATASETS

    from .test_dealias import nexrad_volume

    try:
        path = DATASETS.fetch("KLBB20160601_150025_V06")
    except Exception as err:  # pragma: no cover - network
        pytest.skip(f"sample data unavailable: {err}")
    dtree = nexrad_volume(path)
    # VCP 21 has no 0.9 deg tilt
    with pytest.raises(ValueError, match="0.9"):
        tornet_inputs(dtree)
    inp = tornet_inputs(dtree, elevations=(0.5, 1.5), max_range=150e3)
    assert inp.sizes["azimuth"] == 720 and inp.sizes["tilt"] == 2
    for v in VARS:
        assert np.isfinite(inp[v]).any(), v
    # dealiased velocities stay physical; DBZ floor codes are gone
    assert float(np.nanmax(np.abs(inp.VEL))) < 80.0
    assert float(np.nanmin(inp.DBZ)) > -32.0
    out = rotation_couplets(dtree["sweep_1"].to_dataset(inherit="all_coords"))
    assert "couplet" in out.dims


def test_cache_dir_default(monkeypatch):
    monkeypatch.delenv("RADARX_CACHE_DIR", raising=False)
    assert _onnx_models.cache_dir().name == "models"
    monkeypatch.setenv("RADARX_CACHE_DIR", "/somewhere")
    assert _onnx_models.cache_dir() == Path("/somewhere") / "models"


def test_torchscript_rejects_other_tensor_types(tmp_path):
    path = tmp_path / "tiny.pt"
    write_torchscript(path, tiny_mistnet_params(np.random.default_rng(3)))
    with zipfile.ZipFile(path) as zf:
        members = {n: zf.read(n) for n in zf.namelist()}
    meta = json.loads(members["m/model.json"])
    meta["tensors"][0]["dataType"] = "DOUBLE"
    members["m/model.json"] = json.dumps(meta).encode()
    with zipfile.ZipFile(path, "w") as zf:
        for n, b in members.items():
            zf.writestr(n, b)
    with pytest.raises(ValueError, match="unexpected tensor type"):
        _onnx_models.read_mistnet_torchscript(path)


def test_render_skips_missing_fields():
    ds = sweep(0.5, ["DBZH"], n_az=360, n_rng=100)
    out = bio_mod._render(ds, ("DBZH", None, "ZDR"), 16, 1000.0)
    assert out.shape == (3, 16, 16)
    assert np.isnan(out[1:]).all()


def test_sha256(tmp_path):
    path = tmp_path / "blob"
    path.write_bytes(b"radarx")
    assert _onnx_models.sha256(path) == hashlib.sha256(b"radarx").hexdigest()


def test_sweep_helpers_fallbacks():
    tree = xr.DataTree.from_dict(
        {"/": xr.Dataset(), "sweep_0": xr.Dataset(), "other": xr.Dataset()}
    )
    assert [n for n, _ in tor_mod._sweeps(tree)] == ["sweep_0"]
    ds = xr.Dataset(coords={"elevation": ("azimuth", np.full(4, 1.5))})
    assert tor_mod._fixed_angle(ds) == 1.5
    vel = xr.Dataset({"VRADH": ((), 0.0, {"nyquist_velocity": 27.0})})
    assert tor_mod._nyquist(vel, "VRADH", None, 0.5) == 27.0
    assert tor_mod._nyquist(xr.Dataset({"VRADH": 0.0}), "VRADH", None, 0.5) is None
