---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3
  name: python3
  language: python
---

# KDP with a Neural Network: the Workflow

`estimate_kdp(..., method="ml")` replaces the range filter of the classical
estimators by a neural network that reads the unfolded differential phase
along each ray, together with $\rho_{hv}$ and $Z_H$, and returns $K_{DP}$, the
backscatter phase $\delta$ and an uncertainty of $K_{DP}$ per gate. The
pre-processing (masking, sign convention, system offset, unfolding) is the
one of the other methods; only the filtering is learned. The real network of
radarx (a one-dimensional U-Net, `ml/models/kdp` in the repository) is
trained on simulated rays whose $K_{DP}$ and $\delta$ are known exactly. Its
weights are not published yet, so this notebook cannot run it.

This notebook runs the whole workflow on real data with the repository's own
tools and a deliberately small network:

1. build training examples from the rays of a real C-band sweep (CSAPR2,
   ARM, 11 November 2018), with the inputs the network reads and, as the
   target, the $K_{DP}$ computed by the radar processor,
2. train on some azimuth sectors, test on others,
3. export the model to ONNX with exactly the inputs and outputs `radarx.ml`
   expects,
4. run `estimate_kdp(..., method="ml")` and compare with the classical
   estimators on the rays held out from training.

**What the target is.** The target is a reference estimate, not the truth.
There is no measurement of the true $K_{DP}$ in a radar volume. The radar
processor derives it from the same differential phase with its own filter
and quality control, so the network learns to reproduce that filter on gates
that the processor kept as precipitation. Agreement with it measures how well
a filter can be learned from real rays, not the error against the truth. The
classical estimators are compared with the same reference.

**What is real and what is small.** The features (`radarx.retrieve.kdp._ml_features`),
the ONNX interface, the registry and the retrieval are the real ones. The
network is a single linear convolution along range (81 gates of 100 m, 7
input channels, 2 fitted outputs) fitted by least squares in under a second
on a CPU, for one gate spacing and one wavelength band. The real network has
0.43 million parameters, is trained for hours on simulated rays with physics
terms in the loss, and handles all gate spacings and bands. Do not read the
numbers below as the performance of the radarx network.

PyTorch is not needed here. The training code of the real network uses it
(`train.py`, `export.py`); the linear model has a closed-form solution and is
written to ONNX with the `onnx` package.

```{code-cell} ipython3
import hashlib
import time
from pathlib import Path

import cmweather  # noqa: F401  registers the radar colormaps
import matplotlib.pyplot as plt
import numpy as np
import onnx
import xradar as xd
from numpy.lib.stride_tricks import sliding_window_view
from onnx import TensorProto, helper, numpy_helper
from open_radar_data import DATASETS

import radarx
from radarx import ml
from radarx.retrieve import estimate_kdp
from radarx.retrieve.kdp import ML_FEATURES
```

## The radar sweep

The C-band (CSAPR2) PPI of deep convection used in the KDP notebook, from the
open-radar-data collection (ARM data, CC BY 4.0). It has 361 rays of 1100 gates
spaced 100 m. The linear filter is trained for this gate spacing, because it
cannot rescale itself with the spacing as the real network can.

```{code-cell} ipython3
file = DATASETS.fetch("corcsapr2cmacppiM1.c1.20181111.030003.nc")
dtree = xd.io.open_cfradial1_datatree(file).xradar.georeference()
sweep = dtree["sweep_0"].to_dataset()
fields = {
    "phidp": "uncorrected_differential_phase",
    "rhohv": "uncorrected_copol_correlation_coeff",
    "dbzh": "uncorrected_reflectivity_h",
}
dr = float(np.diff(sweep.range.values).mean()) / 1000.0  # km
print(f"gate spacing {dr * 1000:.0f} m, {sweep.sizes['azimuth']} rays x {sweep.sizes['range']} gates")
```

## Training examples from real rays

The network inputs are made by the function that prepares a sweep for
`method="ml"`, so training and inference see identical features (listed in
`ML_FEATURES`). To obtain them without calling the private function, a small
object with the interface of a model records the features that
`estimate_kdp` passes to it. The target is the radar's $K_{DP}$
(`specific_differential_phase`), which exists on the gates the processor kept
as precipitation (19 % of the gates).

```{code-cell} ipython3
class Recorder:
    """Has the interface of a model; stores the features it is given."""

    info = {"name": "recorder"}

    def __init__(self):
        self.chunks = []

    def run(self, inputs):
        x = inputs["features"]
        self.chunks.append(x.copy())
        zero = np.zeros(x.shape[::2], np.float32)
        return {"kdp": zero, "delta": zero, "kdp_std": zero}


recorder = Recorder()
estimate_kdp(sweep, method="ml", model=recorder, **fields)
features = np.concatenate(recorder.chunks)  # ray, feature, range
target = sweep.specific_differential_phase.values
print(ML_FEATURES, features.shape)
```

### Training and test rays

The split is by azimuth: rays from 120° to 240° are held out for testing and
rays outside 118° to 242° are used for training. The 2° gaps keep neighbouring
rays out of both sets. The features of one ray depend only on that ray, so no
gate is seen in training and testing, and the two sectors contain different
storm cells.

```{code-cell} ipython3
az = sweep.azimuth.values
test_rays = (az >= 120) & (az < 240)
train_rays = (az < 118) | (az >= 242)
usable = np.isfinite(target) & (features[:, 1] > 0)  # reference and valid phase
for name, rays in (("train", train_rays), ("test", test_rays)):
    print(f"{name}: {rays.sum()} rays, {usable[rays].sum():,} gates with a reference")
```

## Training

The network sees 81 neighbouring gates of all 7 features and predicts $K_{DP}$
at the centre gate. Because it is linear, the best weights follow from the
normal equations of least squares (with a small ridge term), which replace the
gradient descent of the real network. The $\delta$ output has zero weights:
the processor provides no backscatter phase to learn from.

```{code-cell} ipython3
K = 81  # gates in the filter
C = len(ML_FEATURES)


def design(feats):
    """Sliding windows of the features: (rays x gates, C * K + 1)."""
    padded = np.pad(feats, ((0, 0), (0, 0), (K // 2, K // 2)))
    win = sliding_window_view(padded, K, axis=2)  # ray, feature, gate, tap
    x = win.transpose(0, 2, 1, 3).reshape(-1, C * K)
    return np.concatenate([x, np.ones((len(x), 1), np.float32)], axis=1)


start = time.perf_counter()
x_train = design(features[train_rays])[usable[train_rays].reshape(-1)].astype(np.float64)
y_train = target[train_rays][usable[train_rays]]
w_kdp = np.linalg.solve(x_train.T @ x_train + 0.1 * np.eye(C * K + 1), x_train.T @ y_train)
print(f"{len(y_train):,} gates, trained in {time.perf_counter() - start:.1f} s")

pred = (design(features[test_rays]) @ w_kdp).reshape(test_rays.sum(), -1)
sel = usable[test_rays]
err = pred[sel] - target[test_rays][sel]
sigma = float(np.percentile(np.abs(x_train @ w_kdp - y_train), 68))  # one-sigma equivalent
print(f"held-out sector: RMS difference {np.sqrt(np.mean(err**2)):.2f} deg/km "
      f"(spread of the reference {target[test_rays][sel].std():.2f})")
```

The last output of the network, the $K_{DP}$ uncertainty, is a constant here:
the 68th percentile of the absolute difference from the reference on the
training rays, which is the one-standard-deviation value for a Gaussian error.
The held-out rays show whether it is a fair number.

## Export to ONNX

The ONNX graph has the interface of the real export (`export.py`): one input
`features` (ray, feature, range) and three outputs `kdp`, `delta` and
`kdp_std` (ray, range), with dynamic ray and range axes. A convolution with
zero padding does the filtering; the third output channel has zero weights
and the constant uncertainty as its bias.

```{code-cell} ipython3
w = w_kdp[:-1].reshape(C, K)[None].astype(np.float32)
w = np.concatenate([w, np.zeros((2, C, K), np.float32)])  # delta and kdp_std: zero weights
b = np.array([w_kdp[-1], 0.0, sigma], np.float32)

nodes = [
    helper.make_node("Conv", ["features", "w", "b"], ["y"], pads=[K // 2] * 2),
    helper.make_node("Split", ["y"], ["k", "d", "s"], axis=1),
] + [
    helper.make_node("Squeeze", [src, "axis"], [name])
    for src, name in zip("kds", ("kdp", "delta", "kdp_std"))
]
graph = helper.make_graph(
    nodes,
    "linear-kdp",
    [helper.make_tensor_value_info("features", TensorProto.FLOAT, ["ray", C, "range"])],
    [helper.make_tensor_value_info(n, TensorProto.FLOAT, ["ray", "range"])
     for n in ("kdp", "delta", "kdp_std")],
    [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b"),
     numpy_helper.from_array(np.array([1], np.int64), "axis")],
)
model_proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model_proto.ir_version = 8
onnx.checker.check_model(model_proto)
path = Path("linear_kdp.onnx")
onnx.save(model_proto, path)
sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
print(f"{path} {path.stat().st_size / 1e3:.1f} kB, {C * K + 1} fitted parameters")
```

Registering the file with `radarx.ml` records its hash, licence and
citation; they are copied to the attributes of every output. With the real
network the same call would use the file of its release.

```{code-cell} ipython3
ml.register_model(
    "linear-kdp",
    path,
    sha256,
    licence="MIT",
    citation="radarx documentation example (2026), linear filter trained on one CSAPR2 sweep",
    task="kdp",
)
model = ml.load_model("linear-kdp")
model
```

## Held-out rays: network and classical estimators

The call differs from the classical ones only by `method="ml"` and the model
(which may also be given by its registered name). All estimators run on the
whole sweep; the numbers use only the gates of the held-out sector where the
reference exists and the estimator returns a value.

```{code-cell} ipython3
start = time.perf_counter()
results = {"network": estimate_kdp(sweep, method="ml", model="linear-kdp", **fields)}
print(f"network: {time.perf_counter() - start:.2f} s for {sweep.sizes['azimuth']} rays")
for method in ("hubbert", "vulpiani", "monotone"):
    results[method] = estimate_kdp(sweep, method=method, **fields)

ref_test = target[test_rays]
rows = {}
for name, res in results.items():
    k = res.KDP.values[test_rays]
    ok = np.isfinite(ref_test) & np.isfinite(k)
    d = k[ok] - ref_test[ok]
    rows[name] = (np.sqrt(np.mean(d**2)), d.mean(), np.corrcoef(k[ok], ref_test[ok])[0, 1], ok.sum())
    print(f"{name:9s} RMS difference {rows[name][0]:.2f}, bias {rows[name][1]:+.2f} deg/km, "
          f"r = {rows[name][2]:.3f}, {rows[name][3]:,} gates")

k = results["network"].KDP.values[test_rays]
std = results["network"].KDP_UNCERTAINTY.values[test_rays]
ok = np.isfinite(ref_test) & np.isfinite(k)
print(f"reference within one network uncertainty: {np.mean(np.abs(k[ok] - ref_test[ok]) <= std[ok]):.0%} "
      "(68 % for a Gaussian error)")
print(f"constant uncertainty {np.nanmean(std):.2f} deg/km")
```

The linear network has a smaller difference from the reference than the
classical estimators, which is expected: it was fitted to this reference on
this radar and band, while the classical estimators are used with their default
settings and do not know the reference. A network trained on one sweep says nothing about
other radars, bands or weather; with a single sweep there is also no test on a
different day. The comparison shows that the workflow works and that a filter learned from real
rays can follow the processor's $K_{DP}$.

```{code-cell} ipython3
x, y = sweep.x / 1e3, sweep.y / 1e3
panels = [
    (sweep[fields["dbzh"]], "$Z_H$ (dBZ)", "ChaseSpectral", -10, 65),
    (sweep.specific_differential_phase, "radar processor $K_{DP}$ (reference, °/km)", "turbo", -1, 6),
    (results["hubbert"].KDP, "$K_{DP}$, Hubbert and Bringi (°/km)", "turbo", -1, 6),
    (results["network"].KDP, "$K_{DP}$, linear network (°/km)", "turbo", -1, 6),
]
fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True, layout="constrained")
phi = np.deg2rad(az)
for ax, (da, label, cmap, vmin, vmax) in zip(axes.flat, panels):
    pm = ax.pcolormesh(x, y, da, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(pm, ax=ax, shrink=0.8, label=label)
    for edge in (120, 240):  # borders of the held-out sector
        ax.plot([0, 110 * np.sin(np.deg2rad(edge))], [0, 110 * np.cos(np.deg2rad(edge))],
                color="k", lw=1.2, ls="--")
    ax.set(aspect="equal", xlim=(-110, 110), ylim=(-110, 110))
for ax in axes[1]:
    ax.set_xlabel("east (km)")
for ax in axes[:, 0]:
    ax.set_ylabel("north (km)")
plt.show()
```

The dashed lines bound the held-out sector (from 120° to 240° clockwise from
north, to the south); the network was trained on the rest.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharex=True, sharey=True, layout="constrained")
for ax, name in zip(axes, ("hubbert", "vulpiani", "network")):
    kk = results[name].KDP.values[test_rays]
    ok = np.isfinite(ref_test) & np.isfinite(kk)
    ax.hexbin(ref_test[ok], kk[ok], gridsize=50, bins="log", extent=(-1, 7, -1, 7))
    ax.plot([-1, 7], [-1, 7], color="r", lw=1)
    ax.set(xlabel="radar processor $K_{DP}$ (°/km)", aspect="equal",
           title=f"{name}: RMS {rows[name][0]:.2f}, r = {rows[name][2]:.3f}")
axes[0].set_ylabel("estimate (°/km)")
plt.show()
```

### Along one ray

```{code-cell} ipython3
rays_test = np.flatnonzero(test_rays)
score = np.where(np.isfinite(target[rays_test]), target[rays_test], 0).sum(axis=1)
iray = int(rays_test[np.argmax(score)])
r_km = sweep.range.values / 1e3
fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, layout="constrained")
axes[0].plot(r_km, sweep[fields["dbzh"]].isel(azimuth=iray), color="k", lw=0.8)
axes[0].set(ylabel="$Z_H$ (dBZ)", title=f"azimuth {float(sweep.azimuth[iray]):.1f}° (held out)")
axes[1].plot(r_km, sweep[fields["phidp"]].isel(azimuth=iray), ".", ms=2, color="0.6", label="raw $\\Psi_{DP}$")
axes[1].plot(r_km, results["hubbert"].PHIDP_processed.isel(azimuth=iray), color="C0", label="Hubbert")
axes[1].plot(r_km, results["network"].PHIDP_processed.isel(azimuth=iray), color="C3", label="network")
axes[1].set(ylabel="phase (°)")
axes[1].legend(loc="upper left", frameon=False)
axes[2].plot(r_km, sweep.specific_differential_phase.isel(azimuth=iray), color="k", lw=1, label="radar processor")
axes[2].plot(r_km, results["hubbert"].KDP.isel(azimuth=iray), color="C0", label="Hubbert")
axes[2].plot(r_km, results["network"].KDP.isel(azimuth=iray), color="C3", label="network")
axes[2].set(xlabel="range (km)", ylabel="$K_{DP}$ (°/km)")
axes[2].legend(loc="upper right", frameon=False)
plt.show()
```

Besides `KDP` and `PHIDP_processed`, the ML method returns the backscatter
phase (zero here, by construction of this model) and the uncertainty. The
processed phase is $2\int K_{DP}\,dr$ plus the constant that fits the measured
phase, so phase and $K_{DP}$ are consistent.

## What to take from this

- The pipeline is complete and runs in seconds: build examples from real rays,
  train, export, register, retrieve. A trained model of the real architecture
  goes through the same `ml.register_model` and `estimate_kdp(..., method="ml")`
  calls; only the file changes.
- The target is the radar processor's $K_{DP}$, an estimate and not the truth.
  A network trained on it inherits the processor's filter and its mistakes.
  The real network is trained on simulated rays, where the truth is known.
- The linear filter is specialised to one gate spacing and band and uses a
  constant uncertainty. Its results on the held-out sector show what a
  minimal learned filter does, not what the radarx network achieves.
- To train the real network, run `ml/models/kdp/train.py` and `export.py`
  (see `ml/models/kdp/README.md`); that needs PyTorch and about an hour on a
  laptop GPU.

## References

- Hubbert, J., and V. N. Bringi, 1995: An iterative filtering technique for
  the analysis of copolar differential phase and dual-frequency radar
  measurements. *J. Atmos. Oceanic Technol.*, **12**, 643-648,
  https://doi.org/10.1175/1520-0426(1995)012<0643:AIFTFT>2.0.CO;2
