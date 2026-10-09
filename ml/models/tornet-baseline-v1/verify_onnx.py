"""
Compare radarx's ONNX conversion of the TorNet CNN with the Keras original.

radarx converts ``tornado_detector_baseline.keras`` (Hugging Face
``tornet-ml/tornado_detector_baseline_v1``, MIT licence) to ONNX on first use
with NumPy, h5py and onnx only (``radarx/retrieve/_onnx_models.py``). This
script checks that conversion against Keras 3 itself. It needs, in a separate
environment (never a radarx dependency)::

    pip install keras torch h5py onnx onnxruntime
    pip install git+https://github.com/mit-ll/tornet   # custom Keras layers

Usage::

    KERAS_BACKEND=torch python verify_onnx.py tornado_detector_baseline.keras \
        [tornet_sample.nc ...]

Without TorNet files, random inputs (with NaN gaps) of the TorNet chip size
are used. Prints the largest absolute difference of the logits.
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


def random_inputs(rng, n=4, shape=(120, 240)):
    x = {}
    for k, (lo, hi) in {
        "DBZ": (-20, 60),
        "VEL": (-60, 60),
        "KDP": (-2, 5),
        "RHOHV": (0.2, 1.04),
        "ZDR": (-1, 8),
        "WIDTH": (0, 9),
    }.items():
        v = rng.uniform(lo, hi, (n, *shape, 2)).astype("float32")
        v[rng.random(v.shape) < 0.3] = np.nan
        x[k] = v
    x["range_folded_mask"] = (rng.random((n, *shape, 2)) < 0.05).astype("float32")
    r = np.linspace(0.2, 0.8, shape[1])[None, None, :] * np.ones((n, shape[0], 1))
    x["coordinates"] = np.stack([r, 1 / r], -1).astype("float32")
    return x


def tornet_inputs(files):
    from tornet.data.loader import read_file
    from tornet.data.preprocess import add_coordinates

    batch = []
    for f in files:
        d = read_file(f, n_frames=1)
        add_coordinates(d, include_az=False, tilt_last=True)
        batch.append(d)
    keys = list(onnx_models.TORNET_VARIABLES) + ["range_folded_mask"]
    x = {k: np.concatenate([b[k] for b in batch]).astype("float32") for k in keys}
    x["coordinates"] = np.stack([b["coordinates"] for b in batch]).astype("float32")
    return x


def main(path, files):
    import keras
    import onnxruntime as ort
    import tornet.models.keras.layers  # noqa: F401  registers the custom layers

    model = keras.saving.load_model(path, compile=False)
    graph = onnx_models.tornet_graph(onnx_models.read_tornet_keras(path))
    session = ort.InferenceSession(graph.SerializeToString())
    x = tornet_inputs(files) if files else random_inputs(np.random.default_rng(0))
    ref = np.asarray(model.predict(x, verbose=0)).reshape(-1)
    logit, heatmap = session.run(["logit", "heatmap"], x)
    diff = np.abs(ref - logit.reshape(-1))
    print("keras logits:", np.round(ref, 4))
    print("onnx  logits:", np.round(logit.reshape(-1), 4))
    print(f"max |difference| = {diff.max():.2e}; heatmap shape {heatmap.shape}")
    return diff.max()


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1], sys.argv[2:]) < 1e-3 else 1)
