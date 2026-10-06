"""
Compare radarx's ONNX conversion of MistNet with the TorchScript original.

radarx converts ``mistnet_nexrad.pt`` (GitHub ``adokter/MistNet``, MIT
licence) to ONNX on first use with NumPy and onnx only
(``radarx/retrieve/_onnx_models.py``). The file uses the legacy TorchScript
format of PyTorch 1.0, which PyTorch 2 no longer loads, so the reference needs
an old PyTorch in a separate environment (never a radarx dependency)::

    uv venv -p 3.10 mistnet-ref
    uv pip install -p mistnet-ref "torch==1.13.1" "numpy<2" onnx onnxruntime

Usage::

    python verify_onnx.py mistnet_nexrad.pt

Feeds a 608 x 608 input with a block of synthetic radar data surrounded by
missing values (NaN) and prints the largest absolute difference of the class
probabilities.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location(
    "onnx_models", HERE.parents[3] / "radarx" / "retrieve" / "_onnx_models.py"
)
onnx_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(onnx_models)


def inputs(seed=0):
    rng = np.random.default_rng(seed)
    x = np.full((1, 15, 608, 608), np.nan, np.float32)
    rows, cols = slice(150, 450), slice(100, 500)
    shape = (1, 5, 300, 400)
    x[:, 0:5, rows, cols] = rng.uniform(-10, 40, shape)
    x[:, 5:10, rows, cols] = rng.uniform(-20, 20, shape)
    x[:, 10:15, rows, cols] = rng.uniform(0, 8, shape)
    x[:, :, 300:320, 300:320] = np.nan  # a hole inside the echo
    return x


def functional_reference(path, x):
    """
    The forward pass of ``code/misnet_v4.py`` (inside the .pt file) written
    with ``torch.nn.functional``, for PyTorch versions that cannot load the
    legacy file.
    """
    import torch
    import torch.nn.functional as F

    p = {
        k: torch.from_numpy(v)
        for k, v in onnx_models.read_mistnet_torchscript(path).items()
    }
    x = torch.from_numpy(x.copy())
    bg = torch.isnan(x[:, 0:5])
    dbz, vel, wid = torch.split(x, 5, 1)
    x = torch.cat(
        [
            torch.nan_to_num(dbz, nan=-33.0),
            torch.nan_to_num(vel, nan=0.0),
            torch.nan_to_num(wid, nan=0.0),
        ],
        1,
    )
    x = x - p["mean"]
    x = F.conv2d(x, p["mistnet.adaptor.weight"], p["mistnet.adaptor.bias"])
    feats = {}
    for block, convs in onnx_models._MISTNET_BLOCKS.items():
        for k in convs:
            s = f"mistnet.backbone.{block}.{k}"
            x = F.relu(F.conv2d(x, p[s + ".weight"], p[s + ".bias"], padding=1))
        x = F.max_pool2d(x, 2, 2)
        feats[block] = x
    x = F.relu(
        F.conv2d(
            x,
            p["mistnet.backbone.block5.7.weight"],
            p["mistnet.backbone.block5.7.bias"],
            padding=3,
        )
    )
    x = F.relu(
        F.conv2d(
            x,
            p["mistnet.backbone.block5.10.weight"],
            p["mistnet.backbone.block5.10.bias"],
        )
    )
    outs = []
    for s in range(5):
        h = f"mistnet.prediction.{s}"

        def conv(src, which, h=h):
            return F.conv2d(src, p[f"{h}.{which}.weight"], p[f"{h}.{which}.bias"])

        up2, up8 = p[f"{h}.upsample_2x.weight"], p[f"{h}.upsample_8x.weight"]
        y = F.conv_transpose2d(conv(x, "pred_32s"), up2, stride=2, padding=1, groups=3)
        y = y + conv(feats["block4"], "pred_16s")
        y = F.conv_transpose2d(y, up2, stride=2, padding=1, groups=3)
        y = y + conv(feats["block3"], "pred_8s")
        y = F.conv_transpose2d(y, up8, stride=8, padding=4, groups=3)
        outs.append(y.unsqueeze(2))
    y = torch.softmax(torch.cat(outs, 2), 1)
    y[:, 0] = bg.float()
    return y.numpy()


def main(path):
    import onnxruntime as ort
    import torch

    x = inputs()
    try:
        module = torch.jit.load(path, map_location="cpu")
        with torch.no_grad():
            ref = module(torch.from_numpy(x.copy())).numpy()
        print(f"reference: TorchScript file, torch {torch.__version__}")
    except RuntimeError as err:  # legacy format not supported by this torch
        print(f"torch {torch.__version__} cannot load the file ({err});")
        print("reference: functional re-implementation of its forward pass")
        with torch.no_grad():
            ref = functional_reference(path, x)
    graph = onnx_models.mistnet_graph(onnx_models.read_mistnet_torchscript(path))
    session = ort.InferenceSession(graph.SerializeToString())
    out = session.run(["y"], {"x": x})[0]
    diff = np.abs(ref - out)
    print(f"output shape {out.shape}; max |difference| = {diff.max():.2e}")
    for c, name in enumerate(("background", "biology", "weather")):
        print(
            f"{name:>10}: mean {ref[:, c].mean():.4f} (torch) {out[:, c].mean():.4f} (onnx)"
        )
    return diff.max()


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) < 1e-4 else 1)
