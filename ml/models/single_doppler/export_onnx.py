"""
Export a trained network to ONNX for ``radarx.retrieve.single_doppler_winds``.

The graph takes ``features`` ``float32[N, C, Z, Y, X]`` (dynamic N, Z, Y, X;
multiples of 4) and returns ``wind`` ``float32[N, 3, Z, Y, X]`` in m s-1. The
grid spacing the network was trained for and the feature version are stored
in the ONNX metadata, which radarx reads.

::

    python export_onnx.py RUN/best.pt single_doppler.onnx [--width 16]
"""

import argparse

import numpy as np
import onnx
import torch
from model import SingleDopplerNet

from radarx.retrieve.single_doppler import FEATURE_VERSION, FEATURES

METADATA = {
    "name": "radarx-single-doppler",
    "task": "single-doppler-winds",
    "version": "0.1",
    "licence": "MIT",
    "dx": "1000",
    "dy": "1000",
    "dz": "500",
    "pad_multiple": "4",
    "feature_version": FEATURE_VERSION,
    "features": ",".join(FEATURES),
}


def export(model, path, metadata=METADATA, shape=(24, 64, 64)):
    model = model.eval().cpu()
    x = torch.zeros((1, len(FEATURES)) + tuple(shape))
    torch.onnx.export(
        model,
        (x,),
        path,
        input_names=["features"],
        output_names=["wind"],
        dynamic_axes={
            "features": {0: "n", 2: "z", 3: "y", 4: "x"},
            "wind": {0: "n", 2: "z", 3: "y", 4: "x"},
        },
        opset_version=17,
        dynamo=False,
    )
    m = onnx.load(path)
    for k, v in metadata.items():
        entry = m.metadata_props.add()
        entry.key, entry.value = k, str(v)
    onnx.save(m, path)


def check(model, path, shape=(24, 72, 88)):
    """Largest difference between PyTorch and ONNX Runtime on random input."""
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    x = rng.normal(size=(1, len(FEATURES)) + shape).astype(np.float32)
    with torch.no_grad():
        ref = model.eval().cpu()(torch.from_numpy(x)).numpy()
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out = sess.run(["wind"], {"features": x})[0]
    return float(np.abs(out - ref).max())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("onnx")
    p.add_argument("--width", type=int, default=16)
    a = p.parse_args()
    net = SingleDopplerNet(width=a.width)
    net.load_state_dict(torch.load(a.checkpoint, map_location="cpu"))
    export(net, a.onnx)
    print("max |onnxruntime - torch| =", check(net, a.onnx))
