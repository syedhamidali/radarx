#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Pretrained third-party models converted to ONNX on first use.

Neither upstream project publishes ONNX files, so radarx downloads the
original weights from the upstream location (sha256-checked), writes the
network as an ONNX graph and caches it next to the other radarx models. Only
NumPy and the ``onnx`` package (plus ``h5py`` for the Keras file) are needed
for the conversion, no deep-learning framework. Weights are never re-hosted
by radarx.

``tornet-baseline-v1``
    The CNN baseline of the TorNet benchmark (Veillette et al. 2025), Keras 3
    file ``tornado_detector_baseline.keras`` of the Hugging Face repository
    ``tornet-ml/tornado_detector_baseline_v1`` (MIT licence).
``mistnet-nexrad``
    MistNet (Lin et al. 2019), the TorchScript file ``mistnet_nexrad.pt`` of
    the GitHub repository ``adokter/MistNet`` (MIT licence).

The graphs reproduce the upstream forward passes operation by operation;
``ml/models/<name>/verify_onnx.py`` in the radarx repository compares them
with Keras and PyTorch.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np

#: Upstream weights and metadata of the models converted on first use.
MODELS = {
    "tornet-baseline-v1": {
        "url": (
            "https://huggingface.co/tornet-ml/tornado_detector_baseline_v1/"
            "resolve/b4d103aa9b2769d7c9258f84de89f876b0a22787/"
            "tornado_detector_baseline.keras"
        ),
        "sha256": "1b698c122baa8b1ef2e218694ba853426f0ef4996c271fe3a0f2d0734e3a3185",
        "licence": "MIT",
        "citation": (
            "Veillette, M. S., J. M. Kurdzo, P. M. Stepanian, J. Y. N. Cho, "
            "T. Reis, S. Samsi, J. McDonald, and N. Chisler, 2025: A benchmark "
            "dataset for tornado detection and prediction using full-resolution "
            "polarimetric weather radar data. Artif. Intell. Earth Syst., 4 (1), "
            "https://doi.org/10.1175/AIES-D-24-0006.1"
        ),
        "version": "1",
        "task": "tornado-detection",
        "inputs": {
            "DBZ, VEL, KDP, RHOHV, ZDR, WIDTH, range_folded_mask, coordinates": (
                "float32[N, azimuth, range, 2] (0.5 deg x 250 m, 0.5 and 0.9 deg "
                "tilts; coordinates = range and 1/range in 1e5 m)"
            )
        },
        "outputs": {
            "logit": "float32[N, 1]",
            "heatmap": "float32[N, azimuth/16, range/16] logits",
        },
    },
    "mistnet-nexrad": {
        "url": (
            "https://media.githubusercontent.com/media/adokter/MistNet/"
            "908f5c059e4f51d9831e726dbf5a4f314d99e7b5/mistnet_nexrad.pt"
        ),
        "sha256": "f869886499bd94629c8c503ef537f6fc2b7c164d89f6ec85f4bc633f21785ea3",
        "licence": "MIT",
        "citation": (
            "Lin, T.-Y., K. Winner, G. Bernstein, A. Mittal, A. M. Dokter, "
            "K. G. Horton, C. Nilsson, B. M. Van Doren, A. Farnsworth, "
            "F. A. La Sorte, S. Maji, and D. Sheldon, 2019: MistNet: Measuring "
            "historical bird migration in the US using archived weather radar "
            "data and convolutional neural networks. Methods Ecol. Evol., 10 "
            "(11), 1908-1922, https://doi.org/10.1111/2041-210X.13280"
        ),
        "version": "1",
        "task": "biology-weather-segmentation",
        "inputs": {
            "x": (
                "float32[N, 15, 608, 608]: DBZH, VRADH, WRADH (channel = "
                "5 * product + scan) of the 0.5-4.5 deg scans on a 500 m grid"
            )
        },
        "outputs": {"y": "float32[N, 3 (background, biology, weather), 5, 608, 608]"},
    },
}

_OPSET = 17
_IR_VERSION = 8


def _require_onnx():
    try:
        import onnx  # noqa: F401
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Converting the upstream weights to ONNX on first use needs the "
            "'onnx' package: pip install onnx"
        ) from err
    return onnx


def cache_dir():
    """Folder of the radarx model cache (``RADARX_CACHE_DIR`` or pooch's)."""
    root = os.environ.get("RADARX_CACHE_DIR")
    if root:
        return Path(root) / "models"
    import pooch

    return Path(pooch.os_cache("radarx")) / "models"


def sha256(path):
    """SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# small graph builder
# --------------------------------------------------------------------------


class _Graph:
    """Collects ONNX nodes and initializers with unique names."""

    def __init__(self):
        self.onnx = _require_onnx()
        self.nodes, self.inits, self.count = [], [], 0

    def name(self, stem):
        self.count += 1
        return f"{stem}_{self.count}"

    def const(self, value, stem="c", dtype=np.float32):
        name = self.name(stem)
        self.inits.append(
            self.onnx.numpy_helper.from_array(np.asarray(value, dtype=dtype), name)
        )
        return name

    def op(self, op_type, inputs, stem=None, n_out=1, out=None, **attrs):
        outs = (
            [out] if out else [self.name(stem or op_type.lower()) for _ in range(n_out)]
        )
        self.nodes.append(
            self.onnx.helper.make_node(op_type, list(inputs), outs, **attrs)
        )
        return outs[0] if n_out == 1 else outs

    def conv(self, x, weight, bias=None, pads=0, **attrs):
        inputs = [x, self.const(weight, "W")]
        if bias is not None:
            inputs.append(self.const(bias, "B"))
        return self.op("Conv", inputs, pads=[pads] * 4, **attrs)

    def model(self, inputs, outputs, doc):
        helper = self.onnx.helper
        graph = helper.make_graph(self.nodes, "radarx", inputs, outputs, self.inits)
        model = helper.make_model(
            graph,
            producer_name="radarx",
            opset_imports=[helper.make_opsetid("", _OPSET)],
            doc_string=doc,
        )
        model.ir_version = _IR_VERSION
        self.onnx.checker.check_model(model)
        return model


def _value(name, shape):
    onnx = _require_onnx()
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


# --------------------------------------------------------------------------
# TorNet CNN baseline
# --------------------------------------------------------------------------

#: Input variables of the TorNet CNN in the order of its first concatenation.
TORNET_VARIABLES = ("DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH")


def read_tornet_keras(path):
    """
    Weights of the TorNet CNN from its Keras 3 ``.keras`` file.

    Returns a dict with ``variables`` (input order), ``mean`` and ``std`` of
    the normalisation (per variable, one value per tilt), ``background``
    (fill value of missing gates), ``blocks`` (list of lists of
    ``(kernel, bias)`` of the CoordConv layers, Keras layout
    ``[kh, kw, cin, cout]``) and ``head`` (the three 1x1 convolutions).
    """
    try:
        import h5py
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError("Reading the TorNet Keras file needs h5py") from err
    with zipfile.ZipFile(path) as zf:
        config = json.loads(zf.read("config.json"))
        weights = io.BytesIO(zf.read("model.weights.h5"))
    layers = config["config"]["layers"]
    by_name = {layer["name"]: layer for layer in layers}

    def inbound(layer):
        names = []
        for node in layer.get("inbound_nodes", []):
            for arg in node["args"]:
                for item in arg if isinstance(arg, list) else [arg]:
                    names.append(item["config"]["keras_history"][0])
        return names

    concat = by_name["Concatenate1"]
    variables, mean, std = [], [], []
    for norm_name in inbound(concat):
        cfg = by_name[norm_name]["config"]
        variables.append(inbound(by_name[norm_name])[0])
        mean.append(np.asarray(cfg["mean"], dtype=np.float64))
        std.append(np.sqrt(np.asarray(cfg["variance"], dtype=np.float64)))
    fill = [
        layer["config"]["fill_val"] for layer in layers if "fill_val" in layer["config"]
    ]
    background = float(fill[0]) if fill else -3.0
    # the block structure: CoordConv layers between max-pooling layers
    blocks, current = [], []
    for layer in layers:
        if layer["class_name"] == "CoordConv2D":
            current.append(layer["name"])
        elif layer["class_name"] == "MaxPooling2D" and current:
            blocks.append(current)
            current = []
    n_coord = sum(len(b) for b in blocks)
    n_head = sum(1 for layer in layers if layer["class_name"] == "Conv2D")
    with h5py.File(weights, "r") as h5:
        group = h5["layers"]

        def h5name(stem, k):
            return stem if k == 0 else f"{stem}_{k}"

        coord = [
            (
                group[h5name("coord_conv2d", k)]["conv/vars/0"][()],
                group[h5name("coord_conv2d", k)]["conv/vars/1"][()],
            )
            for k in range(n_coord)
        ]
        head = [
            (
                group[h5name("conv2d", k)]["vars/0"][()],
                group[h5name("conv2d", k)]["vars/1"][()],
            )
            for k in range(n_head)
        ]
    it = iter(coord)
    return {
        "variables": variables,
        "mean": mean,
        "std": std,
        "background": background,
        "blocks": [[next(it) for _ in block] for block in blocks],
        "head": head,
    }


def tornet_graph(params):
    """
    ONNX graph of the TorNet CNN (Veillette et al. 2025).

    Inputs are the upstream ones (channels last, ``[N, azimuth, range, 2]``):
    the six radar variables, ``range_folded_mask`` and ``coordinates``.
    Outputs are ``logit`` ``[N, 1]``, the upstream output (maximum of the
    heatmap), and ``heatmap`` ``[N, h, w]``, the logit map on a grid 16 times
    coarser than the input.
    """
    g = _Graph()
    n_tilts = len(params["mean"][0])
    shape = ["N", "azimuth", "range", n_tilts]
    names = list(params["variables"]) + ["range_folded_mask", "coordinates"]
    inputs = [_value(name, shape) for name in names]
    normalized = []
    for name, mean, std in zip(params["variables"], params["mean"], params["std"]):
        diff = g.op("Sub", [name, g.const(mean)])
        normalized.append(g.op("Div", [diff, g.const(std)]))
    x = g.op("Concat", normalized, axis=-1)
    x = g.op(
        "Where",
        [g.op("IsNaN", [x]), g.const(params["background"]), x],
    )
    x = g.op("Concat", [x, "range_folded_mask"], axis=-1)
    to_nchw = {"perm": [0, 3, 1, 2]}
    x = g.op("Transpose", [x], **to_nchw)
    c = g.op("Transpose", ["coordinates"], **to_nchw)
    for block in params["blocks"]:
        for kernel, bias in block:
            xc = g.op("Concat", [x, c], axis=1)
            w = np.ascontiguousarray(kernel.transpose(3, 2, 0, 1))
            x = g.op("Relu", [g.conv(xc, w, bias, pads=kernel.shape[0] // 2)])
        # Keras "same" max pooling: ceil(n / 2) outputs, missing values ignored
        pool = {"kernel_shape": [2, 2], "strides": [2, 2], "ceil_mode": 1}
        x = g.op("MaxPool", [x], **pool)
        c = g.op("MaxPool", [c], **pool)
    for k, (kernel, bias) in enumerate(params["head"]):
        w = np.ascontiguousarray(kernel.transpose(3, 2, 0, 1))
        x = g.conv(x, w, bias)
        if k < len(params["head"]) - 1:
            x = g.op("Relu", [x])
    g.op("Squeeze", [x, g.const([1], dtype=np.int64)], out="heatmap")
    g.op("ReduceMax", [x], axes=[2, 3], keepdims=0, out="logit")
    outputs = [_value("logit", ["N", 1]), _value("heatmap", ["N", "h", "w"])]
    return g.model(inputs, outputs, "TorNet CNN baseline (Veillette et al. 2025)")


# --------------------------------------------------------------------------
# MistNet
# --------------------------------------------------------------------------


def read_mistnet_torchscript(path):
    """
    Weights of MistNet from the legacy TorchScript file ``mistnet_nexrad.pt``.

    The file (PyTorch 1.0 format) holds ``model.json``, which lists every
    parameter with the id of a raw little-endian float32 tensor file. Returns
    ``{dotted parameter name: array}`` plus ``"mean"``, the per-channel offset
    subtracted from the input.
    """
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        prefix = names[0].split("/")[0]
        meta = json.loads(zf.read(f"{prefix}/model.json"))

        def tensor(index):
            t = meta["tensors"][index]
            dims = [int(d) for d in t["dims"]]
            strides = [int(s) for s in t["strides"]]
            if t["dataType"] != "FLOAT":
                raise ValueError(f"unexpected tensor type {t['dataType']}")
            raw = np.frombuffer(zf.read(f"{prefix}/{t['data']['key']}"), "<f4")
            offset = int(t.get("offset", 0))
            return np.lib.stride_tricks.as_strided(
                raw[offset:], dims, [4 * s for s in strides]
            ).copy()

        params = {}

        def walk(module, path):
            for p in module.get("parameters", []):
                params[".".join(path + [p["name"]])] = tensor(int(p["tensorId"]))
            for sub in module.get("submodules", []):
                walk(sub, path + [sub["name"]])

        walk(meta["mainModule"], [])
        # the only constant of the forward pass is the input offset
        params["mean"] = tensor(0)
    return params


#: Convolutions of the MistNet backbone (VGG-16) per block.
_MISTNET_BLOCKS = {
    "block1": ("0", "2"),
    "block2": ("0", "2"),
    "block3": ("0", "2", "4"),
    "block4": ("0", "2", "4"),
    "block5": ("0", "2", "4"),
}


def mistnet_graph(params):
    """
    ONNX graph of MistNet (Lin et al. 2019).

    Input ``x`` ``[N, 15, H, W]``: reflectivity, radial velocity and spectrum
    width of five scans (channel ``5 * product + scan``), NaN where there is
    no data; ``H`` and ``W`` divisible by 32 (608 upstream). Output ``y``
    ``[N, 3, 5, H, W]``: per scan the softmax probabilities of biology
    (index 1) and weather (index 2); index 0 (background) is 1 where the
    reflectivity is missing and 0 elsewhere, as upstream.
    """
    g = _Graph()
    p = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}
    n_scans = 5
    inputs = [_value("x", ["N", 3 * n_scans, "H", "W"])]
    dbz, vel, wid = g.op(
        "Split",
        ["x", g.const([n_scans] * 3, dtype=np.int64)],
        n_out=3,
        axis=1,
    )
    missing = g.op("IsNaN", [dbz])
    filled = [
        g.op("Where", [g.op("IsNaN", [v]), g.const(fill), v])
        for v, fill in ((dbz, -33.0), (vel, 0.0), (wid, 0.0))
    ]
    x = g.op("Concat", filled, axis=1)
    x = g.op("Sub", [x, g.const(p["mean"])])
    x = g.conv(x, p["mistnet.adaptor.weight"], p["mistnet.adaptor.bias"])
    features = {}
    for block, convs in _MISTNET_BLOCKS.items():
        for k in convs:
            stem = f"mistnet.backbone.{block}.{k}"
            x = g.conv(x, p[f"{stem}.weight"], p[f"{stem}.bias"], pads=1)
            x = g.op("Relu", [x])
        x = g.op("MaxPool", [x], kernel_shape=[2, 2], strides=[2, 2])
        features[block] = x
    for k, pads in (("7", 3), ("10", 0)):
        stem = f"mistnet.backbone.block5.{k}"
        x = g.conv(x, p[f"{stem}.weight"], p[f"{stem}.bias"], pads=pads)
        x = g.op("Relu", [x])
    scans = []
    for s in range(n_scans):
        stem = f"mistnet.prediction.{s}"

        def head(src, which, stem=stem):
            return g.conv(src, p[f"{stem}.{which}.weight"], p[f"{stem}.{which}.bias"])

        up2 = g.const(p[f"{stem}.upsample_2x.weight"], "W")
        up8 = g.const(p[f"{stem}.upsample_8x.weight"], "W")
        transposed = {"group": 3, "strides": [2, 2], "pads": [1, 1, 1, 1]}
        y = head(x, "pred_32s")
        y = g.op("ConvTranspose", [y, up2], **transposed)
        y = g.op("Add", [y, head(features["block4"], "pred_16s")])
        y = g.op("ConvTranspose", [y, up2], **transposed)
        y = g.op("Add", [y, head(features["block3"], "pred_8s")])
        y = g.op("ConvTranspose", [y, up8], group=3, strides=[8, 8], pads=[4, 4, 4, 4])
        scans.append(g.op("Unsqueeze", [y, g.const([2], dtype=np.int64)]))
    y = g.op("Softmax", [g.op("Concat", scans, axis=2)], axis=1)
    _, rest = g.op("Split", [y, g.const([1, 2], dtype=np.int64)], n_out=2, axis=1)
    background = g.op("Cast", [missing], to=g.onnx.TensorProto.FLOAT)
    background = g.op("Unsqueeze", [background, g.const([1], dtype=np.int64)])
    g.op("Concat", [background, rest], axis=1, out="y")
    outputs = [_value("y", ["N", 3, n_scans, "H", "W"])]
    return g.model(inputs, outputs, "MistNet (Lin et al. 2019)")


# --------------------------------------------------------------------------
# conversion on first use
# --------------------------------------------------------------------------

_READERS = {
    "tornet-baseline-v1": (read_tornet_keras, tornet_graph),
    "mistnet-nexrad": (read_mistnet_torchscript, mistnet_graph),
}


def onnx_path(name):
    """
    Path of the cached ONNX file of a model, converting it on first use.

    The upstream weights are downloaded with pooch (sha256-checked), the
    ONNX graph is written to the radarx model cache, and the upstream file
    is removed again.
    """
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODELS)}")
    out = cache_dir() / f"{name}.onnx"
    if out.exists():
        return out
    import pooch

    info = MODELS[name]
    source = pooch.retrieve(
        info["url"],
        known_hash=f"sha256:{info['sha256']}",
        path=cache_dir() / "upstream",
        progressbar=False,
    )
    read, build = _READERS[name]
    model = build(read(source))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".onnx.part")
    tmp.write_bytes(model.SerializeToString())
    tmp.replace(out)
    with contextlib.suppress(OSError):
        os.remove(source)
    return out


def load(model, default, providers=None):
    """
    A :class:`radarx.ml.Model` for ``model``.

    ``model`` is a loaded model (anything with ``run``), the name of a model
    converted on first use (``default`` when None), the name of a model in
    the radarx registry, or a path to an ONNX file.
    """
    if model is None:
        model = default
    if hasattr(model, "run"):
        return model
    from .. import ml

    model = os.fspath(model)
    if model in MODELS:
        info = {k: v for k, v in MODELS[model].items() if k not in ("url", "sha256")}
        path = onnx_path(model)
        ml.register_model(model, path, None, overwrite=True, **info)
    return ml.load_model(model, providers=providers)


def model_attrs(model, default):
    """``ml_model*`` attributes of an output (contract of radarx.ml)."""
    attrs = getattr(model, "attrs", None)
    if isinstance(attrs, dict) and "ml_model" in attrs:
        return {k: str(v) for k, v in attrs.items()}
    info = getattr(model, "info", None)
    info = info if isinstance(info, dict) else {}
    name = info.get("name") or getattr(model, "name", None) or default
    return {
        "ml_model": str(name),
        "ml_model_version": str(info.get("version", "")),
        "ml_model_licence": str(info.get("licence", "")),
        "ml_model_citation": str(info.get("citation", "")),
    }
