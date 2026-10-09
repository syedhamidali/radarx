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

# Single-Doppler Winds with a Network: the Workflow

`single_doppler_winds(..., model=...)` lets a neural network predict the
three-dimensional wind from the radial velocity and reflectivity of **one**
radar and a background wind. By default the prediction is then used as the
background of the variational retrieval (`refine=True`), so that the final
wind still fits the observed radial velocities and the anelastic continuity
equation. The real network of radarx (a three-dimensional U-Net with a
domain-mean context branch, trained with a physics loss in
`ml/models/single_doppler`) has **no published weights yet**, so this
notebook cannot run it.

Instead it runs the **whole workflow** with the repository's own tools and a
deliberately tiny network:

1. generate synthetic single-Doppler samples with an exact wind truth
   (`ml/models/single_doppler/synthetic.py`, the generator of the real
   training set),
2. fit a toy network on them,
3. export it to ONNX with the metadata radarx reads (grid spacing, feature
   version),
4. run `single_doppler_winds(..., model=...)` on a real WSR-88D volume (the
   KGWX squall line of the multi-Doppler notebook) and compare with the
   variational retrieval without a network, using the dual-Doppler retrieval
   of KGWX and KBMX as the reference.

**What is real and what is a toy.** The sample generator, the input features
(`radarx.retrieve.single_doppler._features`), the ONNX interface and metadata,
the tiling of large grids, the refinement by the variational cost and the
radar data are the real ones. The network is a toy: the radial-velocity
innovation (observation minus the radial component of the background) is
projected back onto the beam direction inside the ONNX graph, and one small
linear 3-D convolution (4 channels in, kernel 3 × 5 × 5, 903 weights)
spreads and mixes it into $u$, $v$ and $w$. Only the convolution is fitted, by least squares in
seconds on a CPU. It ignores the fall speed of the hydrometeors and has no
non-linear layers, so it cannot learn what the real network is for: using
reflectivity and the continuity equation to infer the cross-beam wind. Do not
read the numbers below as the performance of the radarx network; they show
that the pipeline works and what a minimal learned prior does.

PyTorch is not needed here. The training code of the real network uses it
(`train.py`, `export_onnx.py`); the toy model has a closed-form solution and
is written to ONNX with the `onnx` package.

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
from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

import radarx as rx
from radarx import ml
from radarx.io import sounding
from radarx.io.aws_data import download_file
from radarx.retrieve.single_doppler import FEATURE_VERSION, FEATURES, WIND_SCALE, _features

# the training code lives in the repository (not in the installed package):
# find it from the folder the notebook runs in
here = Path.cwd().resolve()
repo = next(p for p in [here, *here.parents] if (p / "ml" / "models" / "single_doppler").is_dir())
sys.path.insert(0, str(repo / "ml" / "models" / "single_doppler"))
import synthetic

print(FEATURES)
```

## Synthetic samples with a known wind

`synthetic.sample` builds a wind field that satisfies the anelastic
continuity equation (a random vector potential plus a Beltrami flow, Shapiro
1993) on a sheared background, a reflectivity field enhanced in updrafts, and
a virtual radar at a random position that samples the radial velocity
(1 m/s noise) where there is echo. It returns the network inputs, the true
wind and the background with a random error, as from a sounding or ERA5.
One sample, with the radial velocity the radar sees at 5 km:

```{code-cell} ipython3
demo = synthetic.sample(np.random.default_rng(3), ny=64, nx=64)
k = int(np.argmin(np.abs(demo["z"] - 5000.0)))
extent = [demo["x"][0] / 1e3, demo["x"][-1] / 1e3, demo["y"][0] / 1e3, demo["y"][-1] / 1e3]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
panels = [
    (demo["dbz"][k], "reflectivity (dBZ)", "ChaseSpectral", -10, 65),
    (demo["vr"][k], "radial velocity (m/s)", "RdBu_r", -25, 25),
    (demo["truth"][2][k], "true $w$ (m/s)", "RdBu_r", -10, 10),
]
for ax, (field, label, cmap, vmin, vmax) in zip(axes, panels):
    im = ax.imshow(field, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label=label, shrink=0.85)
    ax.set(xlabel="east (km)", ylabel="north (km)", aspect="equal")
rx_, ry_, _ = demo["radar"]
if abs(rx_) < 90e3 and abs(ry_) < 90e3:
    axes[1].plot(rx_ / 1e3, ry_ / 1e3, "k^")
plt.show()
```

## Training

The input of the network is the feature array of `single_doppler_winds` (11
channels on a 1 km × 1 km × 500 m grid). The toy network predicts the
departure of the wind from the background, so with all weights zero it
returns the background, as the real one does at initialisation. The
convolution is fitted to 40 samples of 64 × 64 × 24 cells by accumulating
the normal equations of least squares.

```{code-cell} ipython3
iv, im_, ibx, iby, ibz = (FEATURES.index(n) for n in (
    "radial_velocity", "radial_velocity_mask", "beam_x", "beam_y", "beam_z"))
iu, ivb = FEATURES.index("background_u"), FEATURES.index("background_v")
KZ, KY, KX = 3, 5, 5


def innovation_channels(f):
    """Projection of the radial-velocity innovation on the beam, as in the graph."""
    inn = (f[iv] - f[ibx] * f[iu] - f[iby] * f[ivb]) * f[im_]
    return np.stack([inn * f[ibx], inn * f[iby], inn * f[ibz], f[im_]])


def design(chan):
    padded = np.pad(chan, ((0, 0), (KZ // 2,) * 2, (KY // 2,) * 2, (KX // 2,) * 2))
    win = sliding_window_view(padded, (KZ, KY, KX), axis=(1, 2, 3))
    return win.transpose(1, 2, 3, 0, 4, 5, 6).reshape(-1, chan.shape[0] * KZ * KY * KX)


def wind_departure(s):
    """Truth minus the background (m/s), the training target."""
    return np.stack([s["truth"][0] - s["u_bg"], s["truth"][1] - s["v_bg"], s["truth"][2]])


n_in = 4 * KZ * KY * KX
xtx = np.zeros((n_in + 1, n_in + 1))
xty = np.zeros((n_in + 1, 3))
rng = np.random.default_rng(1)
start = time.perf_counter()
for _ in range(40):
    s = synthetic.sample(rng, ny=64, nx=64)
    f = _features(s["vr"], s["dbz"], s["coef"], s["u_bg"], s["v_bg"], s["z"], s["distance"])
    x = design(innovation_channels(f))
    x = np.concatenate([x, np.ones((len(x), 1), np.float32)], axis=1)
    wgt = (s["weight"] * np.isfinite(s["truth"]).all(0)).reshape(-1, 1)
    y = wind_departure(s).reshape(3, -1).T
    x64 = (x * wgt).astype(np.float64)
    xtx += x64.T @ x64
    xty += x64.T @ (y * wgt)
weights = np.linalg.solve(xtx + 1e-2 * np.eye(len(xtx)), xty)
print(f"trained on 40 samples in {time.perf_counter() - start:.1f} s")
```

## Export to ONNX

The graph holds the whole toy network: it splits the feature channels,
forms the innovation, projects it onto the beam, applies the fitted
convolution and adds the result to the background wind (`u_bg`, `v_bg` times
`WIND_SCALE`). Like the real export (`export_onnx.py`) it takes `features`
`(N, 11, Z, Y, X)` with dynamic $N$, $Z$, $Y$ and $X$ and returns `wind`
`(N, 3, Z, Y, X)` in m/s; the grid spacing the network was trained for and the
feature version go into the ONNX metadata, which radarx reads.

```{code-cell} ipython3
w_conv = weights[:-1].T.reshape(3, 4, KZ, KY, KX).astype(np.float32)
b_conv = weights[-1].astype(np.float32)
c = lambda a, name: numpy_helper.from_array(np.asarray(a, np.float32), name)  # noqa: E731

chan = [f"f{i}" for i in range(len(FEATURES))]
nodes = [helper.make_node("Split", ["features"], chan, axis=1)]
nodes += [
    helper.make_node("Mul", [chan[ibx], chan[iu]], ["t1"]),
    helper.make_node("Mul", [chan[iby], chan[ivb]], ["t2"]),
    helper.make_node("Add", ["t1", "t2"], ["radial_bg"]),
    helper.make_node("Sub", [chan[iv], "radial_bg"], ["raw_innovation"]),
    helper.make_node("Mul", ["raw_innovation", chan[im_]], ["innovation"]),
    helper.make_node("Mul", ["innovation", chan[ibx]], ["p_x"]),
    helper.make_node("Mul", ["innovation", chan[iby]], ["p_y"]),
    helper.make_node("Mul", ["innovation", chan[ibz]], ["p_z"]),
    helper.make_node("Concat", ["p_x", "p_y", "p_z", chan[im_]], ["conv_in"], axis=1),
    helper.make_node("Conv", ["conv_in", "w", "b"], ["delta"], pads=[KZ // 2, KY // 2, KX // 2] * 2),
    helper.make_node("Mul", [chan[iu], "scale"], ["u_bg"]),
    helper.make_node("Mul", [chan[ivb], "scale"], ["v_bg"]),
    helper.make_node("Mul", [chan[iu], "zero"], ["w_bg"]),
    helper.make_node("Concat", ["u_bg", "v_bg", "w_bg"], ["background"], axis=1),
    helper.make_node("Add", ["background", "delta"], ["wind"]),
]
graph = helper.make_graph(
    nodes,
    "toy-single-doppler",
    [helper.make_tensor_value_info("features", TensorProto.FLOAT, ["n", len(FEATURES), "z", "y", "x"])],
    [helper.make_tensor_value_info("wind", TensorProto.FLOAT, ["n", 3, "z", "y", "x"])],
    [c(w_conv, "w"), c(b_conv, "b"), c([WIND_SCALE], "scale"), c([0.0], "zero")],
)
model_proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model_proto.ir_version = 8
for key, value in {
    "name": "toy-single-doppler",
    "task": "single-doppler-winds",
    "version": "0.0",
    "licence": "MIT",
    "dx": "1000",
    "dy": "1000",
    "dz": "500",
    "pad_multiple": "4",
    "feature_version": FEATURE_VERSION,
    "features": ",".join(FEATURES),
}.items():
    entry = model_proto.metadata_props.add()
    entry.key, entry.value = key, value
onnx.checker.check_model(model_proto)
path = Path("toy_single_doppler.onnx")
onnx.save(model_proto, path)
sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
print(f"{path} {path.stat().st_size / 1e3:.1f} kB, {w_conv.size + b_conv.size} weights")
```

Registering the model records its hash, licence and citation, as for any
`radarx.ml` model; `single_doppler_winds` also accepts the path of the file
directly.

```{code-cell} ipython3
ml.register_model(
    "toy-single-doppler",
    path,
    sha256,
    licence="MIT",
    citation="radarx documentation example (2026), toy network",
    task="single-doppler-winds",
)
toy = ml.load_model("toy-single-doppler")
```

## Test on held-out synthetic samples

Here the truth is known. The error of the background (what the network
starts from) and of the toy network is the RMS difference from the true wind
on cells with an observation or echo, over 20 new samples.

```{code-cell} ipython3
def predict(s):
    f = _features(s["vr"], s["dbz"], s["coef"], s["u_bg"], s["v_bg"], s["z"], s["distance"])
    return toy.run({"features": f[None]})["wind"][0]


rng = np.random.default_rng(99)
err = {"background": np.zeros(3), "toy network": np.zeros(3)}
count = 0
for _ in range(20):
    s = synthetic.sample(rng, ny=64, nx=64)
    use = (s["weight"] > 0.5) & np.isfinite(s["truth"]).all(0)
    bg = np.stack([s["u_bg"], s["v_bg"], np.zeros_like(s["u_bg"])])
    for name, field in (("background", bg), ("toy network", predict(s))):
        err[name] += np.array([((field[q] - s["truth"][q]) ** 2)[use].sum() for q in range(3)])
    count += use.sum()
for name, total in err.items():
    u_, v_, w_ = np.sqrt(total / count)
    print(f"{name:12s} RMS error  u {u_:.2f}  v {v_:.2f}  w {w_:.2f} m/s")
```

The toy network corrects the wind along the beams, where the radial velocity
measures it, and is no better than the background across them; it has no
way to estimate the vertical velocity beyond what its small convolution can
infer from the radial innovation.

## The KGWX squall line

On 30 March 2022 at 00 UTC a squall line was just west of the KGWX radar
(Columbus, Mississippi) and 166 km from KBMX (Birmingham, Alabama), as in the
multi-Doppler notebook. Both volumes are dealiased (Nyquist velocity from the
radial headers, no-data codes masked) and gridded onto one 2-km grid with
`multi_doppler_input`. The background (wind, density and freezing level) comes
from the Birmingham radiosonde.

```{code-cell} ipython3
def nexrad_volume(key):
    path = download_file("unidata-nexrad-level2", f"2022/03/30/{key}", ".")
    with NEXRADLevel2File(path) as nf:
        nyquist = [h["msg_31_data_header"]["RAD"]["nyquist_vel"] / 100.0 for h in nf.msg_31_data_header]
    dtree = xd.io.open_nexradlevel2_datatree(path)
    for i, name in enumerate(n for n in dtree.children if n.startswith("sweep")):
        ds = dtree[name].to_dataset()
        if "DBZH" in ds:
            ds["DBZH"] = ds.DBZH.where(ds.DBZH > -32)
        if "VRADH" in ds:
            ds["VRADH"] = ds.VRADH.where(ds.VRADH > -63.9)
        dtree[name] = ds.assign_coords(nyquist_velocity=nyquist[i])
    return dtree


def dealiased(key):
    vol = nexrad_volume(key)
    return vol.radarx.assign(vol.radarx.dealias("VRADH", name="VRADH"))


kgwx = dealiased("KGWX/KGWX20220330_235959_V06")
kbmx = dealiased("KBMX/KBMX20220330_235713_V06")
grids = rx.retrieve.multi_doppler_input(
    [kgwx, kbmx],
    x=np.arange(-120e3, 60e3 + 1, 2000.0),
    y=np.arange(-120e3, 100e3 + 1, 2000.0),
    z=np.arange(500.0, 12e3 + 1, 500.0),
)
profile = sounding.read_sounding("BMX", "2022-03-31T00:00")
background = sounding.profile_to_grid(profile, grids)
```

The two radars scanned about 2 minutes apart and the storm moved in between;
the multi-Doppler notebook corrects for that with an estimated storm motion.
That step is left out here to keep the example short. It affects the reference
and the single-radar retrievals alike.

### The reference and the single-radar retrievals

The reference is the dual-Doppler retrieval of both radars. The single-radar
retrievals use KGWX only (`radar=0`): the variational one without a network
(background and mass continuity only), the toy network alone
(`refine=False`) and the toy network refined by the variational cost
(`refine=True`, the default). The network works on a grid of 1 km × 1 km ×
500 m, so radarx interpolates the 2-km grid to it and back.

```{code-cell} ipython3
runs = {}
for name, call in {
    "dual-Doppler": lambda: grids.radarx.multi_doppler(background),
    "variational": lambda: rx.retrieve.single_doppler_winds(grids, background, radar=0),
    "network": lambda: rx.retrieve.single_doppler_winds(
        grids, background, radar=0, model=toy, refine=False
    ),
    "network + variational": lambda: rx.retrieve.single_doppler_winds(
        grids, background, radar=0, model=toy
    ),
}.items():
    start = time.perf_counter()
    runs[name] = call()
    print(f"{name:22s} {time.perf_counter() - start:5.1f} s")
print({k: v for k, v in runs["network + variational"].attrs.items() if k.startswith("ml_")})
```

### How far from the dual-Doppler wind

The dual-Doppler wind is trusted where both radars observe with a beam
crossing angle above 30 degrees. Inside that area (and where KGWX sees
echo) the RMS difference of each single-radar wind from the reference is:

```{code-cell} ipython3
ref = runs["dual-Doppler"]
good = (ref.beam_crossing_angle > 30) & (ref.n_radars >= 2) & grids.DBZH.isel(radar=0).notnull()
bg_full = background.broadcast_like(ref.u)
table = {"background": {"u": bg_full.u, "v": bg_full.v, "w": xr.zeros_like(ref.w)}}  # no background w
table.update({name: runs[name] for name in ("variational", "network", "network + variational")})
print(f"{good.sum().item():,} cells in the dual-Doppler area")
rms = {}
for name, wind in table.items():
    rms[name] = [float(np.sqrt(((wind[c] - ref[c]) ** 2).where(good).mean())) for c in "uvw"]
    print(f"{name:22s} u {rms[name][0]:5.2f}  v {rms[name][1]:5.2f}  w {rms[name][2]:5.2f} m/s")
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
width = 0.2
for j, (name, vals) in enumerate(rms.items()):
    ax.bar(np.arange(3) + (j - 1.5) * width, vals, width, label=name)
ax.set_xticks(range(3), ["$u$", "$v$", "$w$"])
ax.set(ylabel="RMS difference from dual-Doppler (m/s)")
ax.legend(frameon=False, ncols=2)
plt.show()
```

### Maps at 5 km

```{code-cell} ipython3
height = 5000.0
km = dict(x=grids.x / 1e3, y=grids.y / 1e3)
dbz = grids.DBZH.isel(radar=0).sel(z=height)
fig, axes = plt.subplots(1, 4, figsize=(20, 6), layout="constrained", sharey=True)
for ax, (name, wind) in zip(axes, {"dual-Doppler (reference)": ref, **{k: runs[k] for k in ("variational", "network", "network + variational")}}.items()):
    lev = wind.sel(z=height)
    im = ax.pcolormesh(km["x"], km["y"], lev.w.where(dbz.notnull()), cmap="RdBu_r", vmin=-8, vmax=8)
    sub = lev.isel(x=slice(None, None, 4), y=slice(None, None, 4))
    ax.quiver(sub.x / 1e3, sub.y / 1e3, sub.u, sub.v, scale=800, width=0.002)
    ax.contour(km["x"], km["y"], dbz.fillna(-30), levels=[35, 50], colors="k", linewidths=0.6)
    ax.plot(0, 0, "k^")
    ax.set(title=name, xlim=(-120, 60), ylim=(-120, 100), aspect="equal", xlabel="east (km)")
axes[0].set_ylabel("north (km)")
fig.colorbar(im, ax=axes, shrink=0.6, label="$w$ (m/s), arrows: horizontal wind")
plt.show()
```

## What to take from this

- The pipeline is complete and runs in about a minute: sample, fit, export,
  register, retrieve. A trained model of the real architecture goes through
  the same `model=` argument; only the file changes.
- The toy network is a linear projection of the radial innovation with a
  learned spatial spreading. It cannot represent the cross-beam wind that the
  real network is trained to estimate from reflectivity, the continuity
  equation and the physics loss. In this run the toy prior improves the
  horizontal wind modestly over the variational retrieval and leaves $w$
  about unchanged (the network alone is worse in $w$); that says nothing
  about the real network.
- The reference is itself an analysis with errors, from radars scanning at
  different times, and covers only the dual-Doppler area.
- To train the real network, run `ml/models/single_doppler/train.py` and
  `export_onnx.py` (see `ml/models/single_doppler/README.md`).

## References

- Gao, J., M. Xue, A. Shapiro, and K. K. Droegemeier, 1999: A variational
  method for the analysis of three-dimensional wind fields from two Doppler
  radars. *Mon. Wea. Rev.*, **127**, 2128-2142,
  https://doi.org/10.1175/1520-0493(1999)127<2128:AVMFTA>2.0.CO;2
- Shapiro, A., 1993: The use of an exact solution of the Navier-Stokes
  equations in a validation test of a three-dimensional nonhydrostatic
  numerical model. *Mon. Wea. Rev.*, **121**, 2420-2425,
  https://doi.org/10.1175/1520-0493(1993)121<2420:TUOAES>2.0.CO;2
