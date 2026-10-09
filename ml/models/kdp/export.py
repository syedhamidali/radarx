#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Export trained weights to ONNX for ``estimate_kdp(..., method="ml")``.

    python export.py runs/full/model.pt runs/full/radarx-kdp.onnx

The graph has one input ``features`` (float32, ray x feature x range) and
three outputs ``kdp``, ``delta`` and ``kdp_std`` (float32, ray x range); the
ray and range axes are dynamic. The SHA-256 of the file is printed for the
radarx.ml registry.
"""

from __future__ import annotations

import hashlib
import sys

import numpy as np
import torch
from model import N_FEATURES, Exported, KDPNet


def export(weights, path):
    net = KDPNet()
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    x = torch.zeros(2, N_FEATURES, 300)
    torch.onnx.export(
        Exported(net).eval(),
        (x,),
        path,
        input_names=["features"],
        output_names=["kdp", "delta", "kdp_std"],
        dynamic_axes={
            "features": {0: "ray", 2: "range"},
            "kdp": {0: "ray", 1: "range"},
            "delta": {0: "ray", 1: "range"},
            "kdp_std": {0: "ray", 1: "range"},
        },
        opset_version=17,
        dynamo=False,
    )
    # check the ONNX graph against PyTorch on another shape
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = torch.randn(3, N_FEATURES, 777)
    ref = [t.detach().numpy() for t in net(x)]
    got = sess.run(None, {"features": x.numpy()})
    for a, b in zip(ref, got):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(path, "sha256", digest)
    return digest


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2])
