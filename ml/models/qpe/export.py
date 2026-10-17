#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Export trained weights to ONNX for ``radarx.retrieve.rain_rate_ml``.

    python export.py runs/sim-v1 runs/sim-v1/radarx-qpe-v1.onnx

The graph has one input ``features`` (float32, N x F, the columns of
``<run>/features.json``) and the outputs ``rain_rate``, ``rain_rate_lower``
and ``rain_rate_upper`` (float32, N); the batch axis is dynamic. The graph is
checked against PyTorch with ONNX Runtime and the SHA-256 of the file and the
registry lines are printed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from model import Exported, QPENet


def export(run, path):
    run = Path(run)
    features = json.loads((run / "features.json").read_text())
    history = json.loads((run / "history.json").read_text())
    net = QPENet(features, tuple(history["args"]["hidden"]))
    net.load_state_dict(
        torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    )
    net.eval()
    x = torch.zeros(2, len(features))
    x[:, 3 if "rhohv" in features else 0] = 0.98
    torch.onnx.export(
        Exported(net).eval(),
        (x,),
        str(path),
        input_names=["features"],
        output_names=["rain_rate", "rain_rate_lower", "rain_rate_upper"],
        dynamic_axes={
            "features": {0: "N"},
            "rain_rate": {0: "N"},
            "rain_rate_lower": {0: "N"},
            "rain_rate_upper": {0: "N"},
        },
        opset_version=17,
        dynamo=False,
    )
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    probe = np.stack(
        [
            {
                "dbz": rng.uniform(10, 55, 500),
                "zdr": rng.uniform(-0.5, 4, 500),
                "kdp": rng.uniform(-0.2, 8, 500),
                "rhohv": rng.uniform(0.9, 1.0, 500),
                "height": rng.uniform(0, 5, 500),
                "temperature": rng.uniform(-10, 30, 500),
                "wavelength": rng.choice([3.19, 5.35, 10.7], 500),
            }[f]
            for f in features
        ],
        axis=1,
    ).astype(np.float32)
    ref = [t.detach().numpy() for t in Exported(net)(torch.from_numpy(probe))]
    got = sess.run(None, {"features": probe})
    for a, b in zip(ref, got):
        np.testing.assert_allclose(a, b, rtol=2e-4, atol=1e-5)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    print(path, "sha256", digest)
    print("features =", json.dumps(features))
    return digest


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2])
