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
weights are **not published yet**, so this notebook cannot run it.

What it does instead is run the **whole workflow** with the repository's own
tools and a deliberately tiny network:

1. simulate rays with known $K_{DP}$ and $\delta$ (`ml/models/kdp/simulate.py`,
   the generator of the real training set: T-matrix scattering of
   normalized-gamma drop size distributions, Testud et al. 2001, melting layer, hail, noise,
   clutter, folding, offsets),
2. train a model on them,
3. export it to ONNX with exactly the inputs and outputs `radarx.ml` expects,
4. run `estimate_kdp(..., method="ml")` on a real C-band sweep and compare with
   the classical estimators.

**What is real and what is a toy.** The simulator, the feature extraction
(`radarx.retrieve.kdp._ml_features`), the ONNX interface, the registry and the
retrieval are the real ones. The network is a toy: a single linear
convolution along range (25 gates, 7 input channels, 2 fitted outputs, 352
parameters) fitted by least squares in a couple of seconds on a CPU, for one
gate spacing and one wavelength band. The real network has 0.43 million
parameters, is trained for hours with physics terms in the loss and
handles all gate spacings and bands. Do not read the numbers below as the
performance of the radarx network; they show that the pipeline works and what
a minimal learned filter does.

PyTorch is not needed here. The training code of the real network uses it
(`train.py`, `export.py`); the toy model has a closed-form solution and is
written to ONNX with the `onnx` package.

```{code-cell} ipython3
import hashlib
import sys
import time
from pathlib import Path

import cmweather  # noqa: F401  registers the radar colormaps
import matplotlib.pyplot as plt
import numpy as np
import onnx
import xarray as xr
import xradar as xd
from numpy.lib.stride_tricks import sliding_window_view
from onnx import TensorProto, helper, numpy_helper
from open_radar_data import DATASETS

import radarx
from radarx import ml
from radarx.retrieve import estimate_kdp
from radarx.retrieve.kdp import ML_FEATURES

# the training code lives in the repository (not in the installed package):
# find it from the folder the notebook runs in
here = Path.cwd().resolve()
repo = next(p for p in [here, *here.parents] if (p / "ml" / "models" / "kdp").is_dir())
sys.path.insert(0, str(repo / "ml" / "models" / "kdp"))
import simulate
```

## The radar sweep

The same C-band (CSAPR2) PPI of deep convection as in the KDP notebook. The
toy network is trained for the gate spacing of this radar, because a linear
filter cannot rescale itself with the spacing as the real network can.

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

## Simulated rays with known truth

`simulate_batch` returns the network inputs (`features`, in the order of
`ML_FEATURES`), the true $K_{DP}$ and $\delta$, and the valid-gate mask. The
inputs are made by the same function that prepares a real sweep, so training
and inference see identical features. One simulated ray at C band:

```{code-cell} ipython3
print(ML_FEATURES)
demo = simulate.simulate_batch(np.random.default_rng(5), 16, 384, dr=dr, band="C")
# a ray with a long stretch of rain and moderate K_DP
score = demo["echo"].sum(axis=1) * (demo["kdp"].max(axis=1) < 12)
i = int(np.argmax(score))
r = dr * np.arange(384)

fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True, layout="constrained")
axes[0].plot(r, demo["z"][i], color="k", lw=0.8)
axes[0].set(ylabel="$Z_H$ (dBZ)", title="a simulated ray")
axes[1].plot(r, demo["phi"][i], ".", ms=2, color="0.5")
axes[1].set(ylabel="measured $\\Psi_{DP}$ (°)")
axes[2].plot(r, demo["kdp"][i], color="C0", label="true $K_{DP}$ (°/km)")
axes[2].plot(r, demo["delta"][i] / 5, color="C3", label="true $\\delta$ / 5 (°)")
axes[2].set(xlabel="range along the ray (km)", ylabel="truth")
axes[2].legend(loc="upper left", frameon=False)
plt.show()
```

## Training

The network sees 25 neighbouring gates of all 7 features and predicts
$K_{DP}$ and $\delta$ at the centre gate. Because it is linear, the best
weights follow from the normal equations of least squares, accumulated over
batches of simulated rays (valid gates only). That takes the place of the
gradient descent of the real network.

```{code-cell} ipython3
K = 25  # gates in the filter
C = len(ML_FEATURES)


def design(features):
    """Sliding windows of the features: (rays x gates, C * K + 1)."""
    padded = np.pad(features, ((0, 0), (0, 0), (K // 2, K // 2)))
    win = sliding_window_view(padded, K, axis=2)  # ray, feature, gate, tap
    x = win.transpose(0, 2, 1, 3).reshape(-1, C * K)
    return np.concatenate([x, np.ones((len(x), 1), np.float32)], axis=1)


rng = np.random.default_rng(1)
xtx = np.zeros((C * K + 1, C * K + 1))
xty = np.zeros((C * K + 1, 2))
n_gates = 0
start = time.perf_counter()
for _ in range(80):  # 80 batches of 32 rays
    batch = simulate.simulate_batch(rng, 32, 384, dr=dr, band="C")
    use = batch["valid"].reshape(-1)
    x = design(batch["features"])[use].astype(np.float64)
    y = np.stack([batch["kdp"].reshape(-1)[use], batch["delta"].reshape(-1)[use]], axis=1)
    xtx += x.T @ x
    xty += x.T @ y
    n_gates += len(x)
weights = np.linalg.solve(xtx + 1e-3 * np.eye(len(xtx)), xty)
print(f"{n_gates:,} gates of 2560 simulated rays, trained in {time.perf_counter() - start:.1f} s")
```

A held-out batch (new random numbers) gives the error of the fitted filter
for $K_{DP}$ and $\delta$ against the truth. The last output of the network,
the $K_{DP}$ uncertainty, is the toy version of the real one: a constant, the
RMS error on this held-out batch.

```{code-cell} ipython3
held = simulate.simulate_batch(np.random.default_rng(99), 64, 384, dr=dr, band="C")
pred = (design(held["features"]) @ weights).reshape(64, 384, 2)
ok = held["valid"]
rmse_kdp = float(np.sqrt(((pred[..., 0] - held["kdp"]) ** 2)[ok].mean()))
rmse_delta = float(np.sqrt(((pred[..., 1] - held["delta"]) ** 2)[ok].mean()))
print(f"held-out K_DP: RMS error {rmse_kdp:.2f} deg/km (spread of the truth {held['kdp'][ok].std():.2f})")
print(f"held-out delta: RMS error {rmse_delta:.2f} deg (spread of the truth {held['delta'][ok].std():.2f})")
```

## Export to ONNX

The ONNX graph has the interface of the real export (`export.py`): one input
`features` (ray, feature, range) and three outputs `kdp`, `delta` and
`kdp_std` (ray, range), with dynamic ray and range axes. A convolution with
zero padding does the filtering; the third output channel has zero weights
and the constant uncertainty as its bias. (The printed parameter count
includes those zero weights; 352 are fitted.)

```{code-cell} ipython3
w = weights[:-1].T.reshape(2, C, K).astype(np.float32)
w = np.concatenate([w, np.zeros((1, C, K), np.float32)])
b = np.concatenate([weights[-1], [rmse_kdp]]).astype(np.float32)

nodes = [
    helper.make_node("Conv", ["features", "w", "b"], ["y"], pads=[K // 2] * 2),
    helper.make_node("Split", ["y"], ["k", "d", "s"], axis=1),
] + [
    helper.make_node("Squeeze", [src, "axis"], [name])
    for src, name in zip("kds", ("kdp", "delta", "kdp_std"))
]
graph = helper.make_graph(
    nodes,
    "toy-kdp",
    [helper.make_tensor_value_info("features", TensorProto.FLOAT, ["ray", C, "range"])],
    [helper.make_tensor_value_info(n, TensorProto.FLOAT, ["ray", "range"])
     for n in ("kdp", "delta", "kdp_std")],
    [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b"),
     numpy_helper.from_array(np.array([1], np.int64), "axis")],
)
model_proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model_proto.ir_version = 8
onnx.checker.check_model(model_proto)
path = Path("toy_kdp.onnx")
onnx.save(model_proto, path)
sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
print(f"{path} {path.stat().st_size / 1e3:.1f} kB, {w.size + b.size} parameters")
```

Registering the file with `radarx.ml` records its hash, licence and
citation; they are copied to the attributes of every output. With the real
network the same call would use the file of its release.

```{code-cell} ipython3
ml.register_model(
    "toy-kdp",
    path,
    sha256,
    licence="MIT",
    citation="radarx documentation example (2026), toy network",
    task="kdp",
)
model = ml.load_model("toy-kdp")
model
```

## Test on simulated rays: the truth is known

Before the real radar, the classical estimators and the toy network process a
held-out batch of simulated rays as one sweep (`estimate_kdp` on a Dataset
with `azimuth` and `range`). The error is the RMS difference from the true
$K_{DP}$ on gates in precipitation that the estimator keeps.

```{code-cell} ipython3
rays = held["phi"].shape[0]
sim = xr.Dataset(
    {
        "PHIDP": (("azimuth", "range"), held["phi"]),
        "RHOHV": (("azimuth", "range"), held["rho"]),
        "DBZH": (("azimuth", "range"), held["z"]),
    },
    coords={"azimuth": np.arange(rays) * 5.0, "range": (10.0 + dr * np.arange(384)) * 1000.0},
)
names = dict(phidp="PHIDP", rhohv="RHOHV", dbzh="DBZH")
rows = {}
for method in ("hubbert", "vulpiani", "monotone", "ml"):
    extra = dict(model=model) if method == "ml" else {}
    out = estimate_kdp(sim, method=method, offset="ray", **names, **extra)
    k = out.KDP.values
    use = held["echo"] & np.isfinite(k)
    rows[method] = (float(np.sqrt(((k - held["kdp"]) ** 2)[use].mean())), float(np.mean((k - held["kdp"])[use])))
    print(f"{method:9s} RMS error {rows[method][0]:.2f} deg/km, bias {rows[method][1]:+.2f} deg/km")
```

The toy network has the smallest error here, which is expected: it was
trained on exactly this kind of data (same simulator, band and gate
spacing), and the classical estimators are used with their default settings.
It is an in-distribution test and says nothing about other radars; the real
network is evaluated on held-out real volumes in `ml/models/kdp/evaluate.py`.

## The real sweep

On the CSAPR2 sweep the call differs from the classical ones only by
`method="ml"` and the model; the model may also be given by its registered
name.

```{code-cell} ipython3
start = time.perf_counter()
ml_out = estimate_kdp(sweep, method="ml", model="toy-kdp", **fields)
print(f"{time.perf_counter() - start:.1f} s for {sweep.sizes['azimuth']} rays")
classical = estimate_kdp(sweep, method="hubbert", **fields)
print(list(ml_out.data_vars))
print({k: v for k, v in ml_out.KDP.attrs.items() if k.startswith("ml_")})
```

Besides `KDP` and `PHIDP_processed`, the ML method returns the backscatter
phase and the uncertainty. The processed phase is $2\int K_{DP}\,dr$ plus the
constant that fits the measured phase minus $\delta$, so phase and $K_{DP}$ are
consistent.

```{code-cell} ipython3
x, y = sweep.x / 1e3, sweep.y / 1e3
panels = [
    (sweep[fields["dbzh"]], "$Z_H$ (dBZ)", "ChaseSpectral", -10, 65),
    (classical.KDP, "$K_{DP}$, Hubbert and Bringi (°/km)", "turbo", -1, 6),
    (ml_out.KDP, "$K_{DP}$, toy network (°/km)", "turbo", -1, 6),
    (ml_out.KDP - classical.KDP, "network minus Hubbert (°/km)", "RdBu_r", -3, 3),
]
fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True, layout="constrained")
for ax, (da, label, cmap, vmin, vmax) in zip(axes.flat, panels):
    pm = ax.pcolormesh(x, y, da, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(pm, ax=ax, shrink=0.8, label=label)
    ax.set(aspect="equal", xlim=(-110, 110), ylim=(-110, 110))
for ax in axes[1]:
    ax.set_xlabel("east (km)")
for ax in axes[:, 0]:
    ax.set_ylabel("north (km)")
plt.show()
```

### Along one ray

```{code-cell} ipython3
iray = int(np.nanargmax(classical.PHIDP_processed.isel(range=-1).values))
r_km = sweep.range.values / 1e3
fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, layout="constrained")
axes[0].plot(r_km, sweep[fields["dbzh"]].isel(azimuth=iray), color="k", lw=0.8)
axes[0].set(ylabel="$Z_H$ (dBZ)", title=f"azimuth {float(sweep.azimuth[iray]):.1f}°")
axes[1].plot(r_km, sweep[fields["phidp"]].isel(azimuth=iray), ".", ms=2, color="0.6", label="raw $\\Psi_{DP}$")
axes[1].plot(r_km, classical.PHIDP_processed.isel(azimuth=iray), color="C0", label="Hubbert")
axes[1].plot(r_km, ml_out.PHIDP_processed.isel(azimuth=iray), color="C3", label="toy network")
axes[1].set(ylabel="phase (°)")
axes[1].legend(loc="upper left", frameon=False)
axes[2].plot(r_km, classical.KDP.isel(azimuth=iray), color="C0", label="Hubbert")
axes[2].plot(r_km, ml_out.KDP.isel(azimuth=iray), color="C3", label="toy network")
axes[2].fill_between(
    r_km,
    (ml_out.KDP - ml_out.KDP_UNCERTAINTY).isel(azimuth=iray),
    (ml_out.KDP + ml_out.KDP_UNCERTAINTY).isel(azimuth=iray),
    color="C3", alpha=0.2, lw=0, label="± constant toy uncertainty",
)
axes[2].set(xlabel="range (km)", ylabel="$K_{DP}$ (°/km)")
axes[2].legend(loc="upper right", frameon=False)
plt.show()
```

On the real ray the toy network follows the same structure as the classical
filter but is noisier, and where the classical estimate ends (low
$\rho_{hv}$ beyond 65 km) it keeps going with high $K_{DP}$ values, so its
processed phase keeps rising. That is the expected failure of a linear
filter trained on simulated rays only: it has no way to learn the
non-linear rules (what to do in noisy, low-$\rho_{hv}$ gates) that the real
network picks up from hundreds of thousands of rays and from physics terms in its loss.

### Against the radar's own KDP

The file contains the $K_{DP}$ computed by the radar processor, as in the KDP
notebook. It is not the truth either, but a third opinion in rain. The
toy network agrees with it less well than the classical estimator does.

```{code-cell} ipython3
ref = sweep.specific_differential_phase.values
rain = (
    (sweep[fields["dbzh"]].values > 30)
    & (sweep[fields["rhohv"]].values > 0.95)
    & np.isfinite(ref)
)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True, sharey=True, layout="constrained")
for ax, (name, res) in zip(axes, [("Hubbert and Bringi", classical), ("toy network", ml_out)]):
    k = res.KDP.values
    sel = rain & np.isfinite(k)
    ax.hexbin(ref[sel], k[sel], gridsize=60, bins="log", extent=(-1, 7, -1, 7))
    ax.plot([-1, 7], [-1, 7], color="r", lw=1)
    ax.set(xlabel="radar $K_{DP}$ (°/km)", ylabel=f"{name} $K_{{DP}}$ (°/km)",
           title=f"r = {np.corrcoef(ref[sel], k[sel])[0, 1]:.2f}, "
                 f"RMS difference {np.sqrt(np.mean((k[sel] - ref[sel]) ** 2)):.2f}",
           aspect="equal")
plt.show()
```

## What to take from this

- The pipeline is complete and runs in seconds: simulate, train, export,
  register, retrieve. A trained model of the real architecture goes through
  the same `ml.register_model` and `estimate_kdp(..., method="ml")` calls;
  only the file changes.
- The toy filter is linear, specialised to one gate spacing and band, and
  carries a constant uncertainty. Its results on the real sweep show what a
  minimal learned filter does, not what the radarx network achieves.
- To train the real network, run `ml/models/kdp/train.py` and `export.py`
  (see `ml/models/kdp/README.md`); that needs PyTorch and about an hour on a
  laptop GPU.

## References

- Hubbert, J., and V. N. Bringi, 1995: An iterative filtering technique for
  the analysis of copolar differential phase and dual-frequency radar
  measurements. *J. Atmos. Oceanic Technol.*, **12**, 643-648,
  https://doi.org/10.1175/1520-0426(1995)012<0643:AIFTFT>2.0.CO;2
- Testud, J., S. Oury, R. A. Black, P. Amayenc, and X. Dou, 2001: The concept
  of "normalized" distribution to describe raindrop spectra: A tool for cloud
  physics and cloud remote sensing. *J. Appl. Meteor.*, **40**, 1118-1140,
  https://doi.org/10.1175/1520-0450(2001)040<1118:TCONDT>2.0.CO;2
