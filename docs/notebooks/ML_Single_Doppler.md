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
`ml/models/single_doppler`) has no published weights yet, so this
notebook cannot run it.

This notebook runs the whole workflow on real WSR-88D volumes with the
repository's own tools and a deliberately small network:

1. grid pairs of KGWX and KBMX volumes of the 30 March 2022 squall line
   (Level II data from the `unidata-nexrad-level2` bucket on AWS) and retrieve
   the dual-Doppler wind of each pair,
2. fit a small network that predicts that wind from the KGWX radial velocity
   and reflectivity alone, on two volume pairs,
3. export it to ONNX with the metadata radarx reads (grid spacing, feature
   version),
4. run `single_doppler_winds(..., model=...)` on a later volume pair that was
   not used for training, and compare with the variational single-Doppler
   retrieval without a network.

**What the target is.** The target is a reference estimate, not the truth.
The dual-Doppler retrieval combines two radars that scan about 2 minutes
apart, has its own errors and is trusted only where the beams cross at more
than 30 degrees. The network is trained to reproduce it from one radar, only
on those cells, and compared with it on the held-out pair. Agreement with
the dual-Doppler wind says how well the single-radar methods approach it, not
what their error against the true wind is.

**What is real and what is small.** The radar data, the gridding, the dual-Doppler
retrieval, the input features (`radarx.retrieve.single_doppler._features`),
the ONNX interface and metadata, the refinement by the variational cost and
the comparison are the real ones. The network is small: the radial-velocity
innovation (observation minus the radial component of the background) is
projected back onto the beam direction inside the ONNX graph, and one linear
3-D convolution (4 channels in, kernel 3 × 5 × 5, 903 weights) spreads and mixes it
into $u$, $v$ and $w$. Only the convolution is fitted, by least squares in
seconds on a CPU. It has no non-linear layers, so it cannot learn what the
real network is for: using reflectivity and the continuity equation to infer the
cross-beam wind. Training uses two volume pairs from 13 minutes before the
test pair and earlier; storm structure changes slowly, so the test is
not independent of the training period in the way a different storm would be.
Do not read the numbers below as the performance of the radarx network.

PyTorch is not needed here. The training code of the real network uses it
(`train.py`, `export_onnx.py`); this model has a closed-form solution and
is written to ONNX with the `onnx` package.

```{code-cell} ipython3
import hashlib
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
from radarx.retrieve.single_doppler import (
    FEATURE_VERSION,
    FEATURES,
    WIND_SCALE,
    _full_background,
    _grid_features,
    _single_radar,
)

print(FEATURES)
```

## Radar volume pairs

On 30 March 2022 a squall line crossed the area of the KGWX radar (Columbus,
Mississippi), 166 km from KBMX (Birmingham, Alabama). A volume pair is the
KGWX volume and the KBMX volume closest to it in time. Both volumes are
dealiased (Nyquist velocity from the radial headers, no-data codes masked) and
gridded onto one 2-km grid with `multi_doppler_input`; the grid is the same
for every pair. The background (wind, density and freezing level) comes from the
Birmingham radiosonde of 00 UTC on 31 March.

The two training pairs are at 23:33 and 23:46 UTC; the test pair is at
23:59 UTC. Each volume is a download of about 24 MB, which is cached.

```{code-cell} ipython3
def nexrad_volume(key):
    name = key.split("/")[-1]
    path = Path(name)
    if not path.exists():
        download_file("unidata-nexrad-level2", f"2022/03/30/{key}", ".")
    with NEXRADLevel2File(name) as nf:
        nyquist = [h["msg_31_data_header"]["RAD"]["nyquist_vel"] / 100.0 for h in nf.msg_31_data_header]
    dtree = xd.io.open_nexradlevel2_datatree(name)
    for i, sweep_name in enumerate(n for n in dtree.children if n.startswith("sweep")):
        ds = dtree[sweep_name].to_dataset()
        if "DBZH" in ds:
            ds["DBZH"] = ds.DBZH.where(ds.DBZH > -32)
        if "VRADH" in ds:
            ds["VRADH"] = ds.VRADH.where(ds.VRADH > -63.9)
        dtree[sweep_name] = ds.assign_coords(nyquist_velocity=nyquist[i])
    return dtree


def dealiased(key):
    vol = nexrad_volume(key)
    return vol.radarx.assign(vol.radarx.dealias("VRADH", name="VRADH"))


profile = sounding.read_sounding("BMX", "2022-03-31T00:00")
pairs = {
    "23:33": ("KGWX20220330_233252_V06", "KBMX20220330_233334_V06"),
    "23:46": ("KGWX20220330_234639_V06", "KBMX20220330_234521_V06"),
    "23:59": ("KGWX20220330_235959_V06", "KBMX20220330_235713_V06"),
}


def grid_pair(label):
    """Both radars on the common grid, the background and the dual-Doppler wind."""
    kgwx_key, kbmx_key = pairs[label]
    grids = rx.retrieve.multi_doppler_input(
        [dealiased(f"KGWX/{kgwx_key}"), dealiased(f"KBMX/{kbmx_key}")],
        x=np.arange(-120e3, 60e3 + 1, 2000.0),
        y=np.arange(-120e3, 100e3 + 1, 2000.0),
        z=np.arange(500.0, 12e3 + 1, 500.0),
    )
    background = sounding.profile_to_grid(profile, grids)
    reference = grids.radarx.multi_doppler(background)
    trusted = (
        (reference.beam_crossing_angle > 30)
        & (reference.n_radars >= 2)
        & grids.DBZH.isel(radar=0).notnull()
    )
    return grids, background, reference, trusted


start = time.perf_counter()
data = {label: grid_pair(label) for label in pairs}
print(f"three volume pairs gridded and retrieved in {time.perf_counter() - start:.0f} s")
for label, (_, _, _, trusted) in data.items():
    print(f"{label} UTC: {trusted.sum().item():,} cells with a trusted dual-Doppler wind")
```

The two radars scanned about 2 minutes apart and the storm moved in between;
the multi-Doppler notebook corrects for that with an estimated storm motion.
That step is left out here to keep the example short. It affects the reference
and the single-radar retrievals alike.

## Training examples

For each training pair the network input is the feature array of
`single_doppler_winds` for KGWX alone (11 channels), and the target is the
departure of the dual-Doppler wind from the background. The network predicts
this departure, so with all weights zero it returns the background, as the
real one does at initialisation. Only the trusted cells are used. The
convolution is fitted by least squares.

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


def examples(label):
    """Inputs of the convolution and the wind departure on the trusted cells."""
    grids, background, reference, trusted = data[label]
    one = _single_radar(grids, None, None, None, 0, "VRADH", "DBZH")
    bg = _full_background(background, one)
    f = _grid_features(one, bg, "VRADH", "DBZH")
    dims = ("z", "y", "x")
    target = np.stack([
        (reference.u - bg.u).transpose(*dims).values,
        (reference.v - bg.v).transpose(*dims).values,
        reference.w.transpose(*dims).values,
    ])
    keep = trusted.transpose(*dims).values & np.isfinite(target).all(axis=0)
    x = design(innovation_channels(f))[keep.reshape(-1)]
    return np.concatenate([x, np.ones((len(x), 1), np.float32)], axis=1), target[:, keep].T


train = ("23:33", "23:46")
start = time.perf_counter()
parts = [examples(label) for label in train]
x_train = np.concatenate([p[0] for p in parts]).astype(np.float64)
y_train = np.concatenate([p[1] for p in parts])
weights = np.linalg.solve(x_train.T @ x_train + 1e-2 * np.eye(x_train.shape[1]), x_train.T @ y_train)
fit = np.sqrt(np.mean((x_train @ weights - y_train) ** 2, axis=0))
print(f"{len(y_train):,} cells from {len(train)} volume pairs, trained in {time.perf_counter() - start:.1f} s")
print(f"RMS difference of the fit on the training cells: u {fit[0]:.2f}  v {fit[1]:.2f}  w {fit[2]:.2f} m/s")
```

## Export to ONNX

The graph holds the whole network: it splits the feature channels,
forms the innovation, projects it onto the beam, applies the fitted
convolution and adds the result to the background wind (`u_bg`, `v_bg` times
`WIND_SCALE`). Like the real export (`export_onnx.py`) it takes `features`
`(N, 11, Z, Y, X)` with dynamic $N$, $Z$, $Y$ and $X$ and returns `wind`
`(N, 3, Z, Y, X)` in m/s; the grid spacing the network was trained for (2 km here, the spacing of the grids it is trained on) and the
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
    "linear-single-doppler",
    [helper.make_tensor_value_info("features", TensorProto.FLOAT, ["n", len(FEATURES), "z", "y", "x"])],
    [helper.make_tensor_value_info("wind", TensorProto.FLOAT, ["n", 3, "z", "y", "x"])],
    [c(w_conv, "w"), c(b_conv, "b"), c([WIND_SCALE], "scale"), c([0.0], "zero")],
)
model_proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model_proto.ir_version = 8
for key, value in {
    "name": "linear-single-doppler",
    "task": "single-doppler-winds",
    "version": "0.0",
    "licence": "MIT",
    "dx": "2000",
    "dy": "2000",
    "dz": "500",
    "pad_multiple": "4",
    "feature_version": FEATURE_VERSION,
    "features": ",".join(FEATURES),
}.items():
    entry = model_proto.metadata_props.add()
    entry.key, entry.value = key, value
onnx.checker.check_model(model_proto)
path = Path("linear_single_doppler.onnx")
onnx.save(model_proto, path)
sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
print(f"{path} {path.stat().st_size / 1e3:.1f} kB, {w_conv.size + b_conv.size} weights")
```

Registering the model records its hash, licence and citation, as for any
`radarx.ml` model; `single_doppler_winds` also accepts the path of the file
directly.

```{code-cell} ipython3
ml.register_model(
    "linear-single-doppler",
    path,
    sha256,
    licence="MIT",
    citation="radarx documentation example (2026), linear network trained on KGWX and KBMX volumes",
    task="single-doppler-winds",
)
net = ml.load_model("linear-single-doppler")
```

## The held-out volume pair

The KGWX volume of 23:59 UTC and the KBMX volume of 23:57 UTC were not used
in training. The comparison below uses only KGWX for the single-radar
retrievals (`radar=0`).

```{code-cell} ipython3
grids, background, ref, good = data["23:59"]
```

### The reference and the single-radar retrievals

The reference is the dual-Doppler retrieval of both radars. The single-radar
retrievals use KGWX only: the variational one without a network (background
and mass continuity only), the network alone (`refine=False`) and the network
refined by the variational cost (`refine=True`, the default). The network was
fitted on a 2-km grid and works on that grid, so no interpolation is needed.

```{code-cell} ipython3
runs = {}
for name, call in {
    "dual-Doppler": lambda: ref,
    "variational": lambda: rx.retrieve.single_doppler_winds(grids, background, radar=0),
    "network": lambda: rx.retrieve.single_doppler_winds(
        grids, background, radar=0, model=net, refine=False
    ),
    "network + variational": lambda: rx.retrieve.single_doppler_winds(
        grids, background, radar=0, model=net
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

On the held-out pair the variational retrieval brings the horizontal wind
closer to the dual-Doppler wind than the sounding does (RMS difference of $u$
and $v$ 4.7 and 4.1 m/s against 5.3 and 6.3 m/s). The linear network, which
was fitted to that reference on other volumes, is closer still (3.5 and 2.7
m/s) and the network refined by the variational cost is closest (3.1 and 2.5
m/s). For $w$ the variational retrieval is the closest (1.0 m/s) and the
network adds error (1.3 m/s alone, 1.1 m/s refined). A network trained on the
dual-Doppler wind is expected to agree with it better in the horizontal wind;
this is not evidence that it is nearer to the true wind.

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

- The pipeline is complete: grid real volume pairs, retrieve the reference,
  fit, export, register, retrieve on a volume that was not used. A trained
  model of the real architecture goes through the same `model=` argument; only
  the file changes.
- The target is the dual-Doppler wind of two radars scanning at different
  times, an estimate that is trusted only where the beams cross at more than
  30 degrees. A network trained on it inherits its errors, and the numbers
  above are differences from it, not errors against the true wind. The real
  network is trained on synthetic samples with an exact wind.
- The linear network is a projection of the radial innovation with a learned
  spatial spreading. It cannot represent the cross-beam wind that the real
  network is trained to estimate from reflectivity, the continuity equation
  and the physics loss. The held-out pair is 13 minutes after the last training pair
  in the same storm and the same area, so the test shows generalisation in time, not to
  other storms or radars.
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
