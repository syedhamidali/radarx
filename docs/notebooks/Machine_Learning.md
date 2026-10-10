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

# Machine Learning Models

`radarx.ml` is the common ground of the machine-learning methods in radarx. It
does not train networks; it runs trained ones on radar data:

- **ONNX Runtime** runs the models. ONNX is the exchange format every
  deep-learning framework exports to, and ONNX Runtime is a small inference
  engine with wheels for all platforms, so radarx never needs PyTorch or
  TensorFlow at run time. Install it with `pip install radarx[ml]`. The CPU
  is used by default and a CUDA, ROCm or DirectML GPU when ONNX Runtime has
  one; other providers (e.g. CoreML on a Mac) can be passed explicitly.
- **A model registry**, `radarx/ml/models.toml`, lists every model with the URL
  of its ONNX file, its SHA-256 hash, **licence**, **citation**, version and
  the meaning of its inputs and outputs. Weights are never part of radarx:
  they are downloaded from their authors' release on first use, verified
  against the hash and cached. The licence and citation follow the model
  everywhere: in `repr(model)`, in `list_models()` and in the attributes of
  every output it produces.
- **Polar patches.** Networks work on fixed-size tiles. `polar_patches` cuts
  sweeps into tiles of rays × gates directly in radar coordinates (no
  interpolation), wrapping around north, and `reassemble` puts the model
  outputs back together, blending overlapping tiles with a window that gives
  the tile borders the least weight. A C++ kernel does both for all sweeps of
  a volume in one multithreaded call.
- **xarray in, xarray out.** Methods built on a model live next to their
  physical counterparts (a retrieval in `radarx.retrieve`, an accessor method
  `ds.radarx.<method>`), take sweeps or volumes and return them with
  coordinates and attributes. Training code (PyTorch, export to ONNX) lives
  outside the package, in the `ml/` folder of the repository.

This notebook shows the pieces with a tiny network built right here, so it
needs no download of weights.

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
from onnx import TensorProto, helper

import radarx.ml as ml
from radarx.io.aws_data import download_file
```

## A sweep of the KGWX squall line

The KGWX (Columbus AFB, Mississippi) WSR-88D volume of 30 March 2022, 23:46
UTC, from the NOAA NEXRAD archive on AWS. NEXRAD no-data codes are masked.

```{code-cell} ipython3
file = download_file(
    "unidata-nexrad-level2", "2022/03/30/KGWX/KGWX20220330_234639_V06", "downloads"
)


def mask_no_data(ds):
    lowest = {"DBZH": -32.0, "ZDR": -12.9}
    return ds.assign({v: ds[v].where(ds[v] > x) for v, x in lowest.items() if v in ds})


dtree = xd.io.open_nexradlevel2_datatree(file).map_over_datasets(mask_no_data).load()
sweep = dtree["sweep_0"].to_dataset(inherit="all_coords").xradar.georeference()
print(float(sweep.sweep_fixed_angle), "deg;", dict(sweep.DBZH.sizes))
print("first ray at", float(sweep.azimuth[0]), "deg")
```

## Polar patches

`polar_patches` stacks the fields of a Dataset as channels and returns a NumPy
array `(n_patches, channels, n_azimuth, n_range)` plus an index recording where
each patch came from. The default stride is half the patch size, the overlap
the blending windows are made for. Patches continue across the last ray to the
first, so the rays need only be in azimuth order (here the volume starts at
the azimuth recorded first, not at north).

```{code-cell} ipython3
patches, index = ml.polar_patches(sweep[["DBZH", "ZDR"]], size=(64, 128))
print(patches.shape, patches.dtype)
print(index.table[:5])  # sweep, first ray, first gate
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
ax = axes[0]
ax.pcolormesh(sweep.x / 1e3, sweep.y / 1e3, sweep.DBZH, cmap="ChaseSpectral",
              vmin=-10, vmax=70)
nray = sweep.sizes["azimuth"]
for k in (0, 40, 150):
    _, a0, r0 = index.table[k]
    rays = (a0 + np.arange(64)) % nray
    box = sweep.isel(azimuth=rays, range=slice(r0, r0 + 128))
    outline = np.concatenate([box.x[:, 0], box.x[-1], box.x[::-1, -1], box.x[0, ::-1]])
    outline_y = np.concatenate([box.y[:, 0], box.y[-1], box.y[::-1, -1], box.y[0, ::-1]])
    ax.plot(outline / 1e3, outline_y / 1e3, "k", lw=1.5)
ax.set(xlim=(-230, 230), ylim=(-230, 230), aspect="equal", xlabel="East (km)",
       ylabel="North (km)", title="DBZH and three 64 × 128 patches")
axes[1].imshow(patches[40, 0], origin="lower", aspect="auto", cmap="ChaseSpectral",
               vmin=-10, vmax=70)
axes[1].set(title="patch 40, DBZH channel", xlabel="gate", ylabel="ray")
```

Reassembling the unchanged patches gives back the sweep bit for bit, NaN
included, with its coordinates:

```{code-cell} ipython3
back = ml.reassemble(patches, index)
for name in ("DBZH", "ZDR"):
    same = np.array_equal(back[name].values, sweep[name].values, equal_nan=True)
    print(name, "identical:", same)
back
```

## A tiny ONNX model

A real model comes from the registry (`ml.list_models()`); here we build a
network in a few lines with the ONNX helper functions: a 5 × 5 box filter, a
convolution with fixed weights and zero padding. The filter only shows how a
model file is built, registered and run; it is not a trained model. Any network exported from
PyTorch (`torch.onnx.export`) is loaded the same way.

```{code-cell} ipython3
k = 5
x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 1, None, None])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 1, None, None])
kernel = helper.make_tensor("w", TensorProto.FLOAT, [1, 1, k, k], [1 / k**2] * k**2)
conv = helper.make_node("Conv", ["x", "w"], ["y"], pads=[k // 2] * 4)
graph = helper.make_graph([conv], "box-filter", [x], [y], [kernel])
onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
onnx_model.ir_version = 8
onnx.checker.check_model(onnx_model)
path = Path("box_filter.onnx")
onnx.save(onnx_model, path)
sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
```

Models outside the shipped registry are registered for the session with their
licence and citation (or listed in a TOML file named by the
`RADARX_MODEL_REGISTRY` environment variable). `load_model` checks the hash
and opens an ONNX Runtime session.

```{code-cell} ipython3
ml.register_model(
    "box-filter",
    path,
    sha256,
    licence="MIT",
    citation="radarx documentation example (2026)",
    task="smoothing",
    inputs={"x": "float32[N,1,H,W] normalised reflectivity"},
    outputs={"y": "float32[N,1,H,W] smoothed"},
)
model = ml.load_model("box-filter")
model
```

```{code-cell} ipython3
ml.list_models()
```

## Running a model on a sweep

Models are trained on normalised inputs without NaN: `normalize` scales a
field with fixed (or data-derived) offset and scale, fills NaN and records the
scaling in attributes for `denormalize`. The patches are run through the
network in batches and reassembled; the model's attributes go on the output.

```{code-cell} ipython3
dbz = ml.normalize(sweep.DBZH, offset=0.0, scale=60.0, fill_value=0.0)
patches, index = ml.polar_patches(dbz, (64, 128))
start = time.perf_counter()
out = model.run({"x": patches[:, None]}, batch_size=64)["y"][:, 0]
print(f"{len(patches)} patches in {time.perf_counter() - start:.2f} s")
smoothed = ml.denormalize(
    ml.reassemble(out, index, attrs=model.attrs).assign_attrs(dbz.attrs)
)
smoothed.attrs
```

The convolution pads each patch with zeros, so it is wrong within two gates
of a patch border. With patches that do not overlap and a plain mean
(`blend="mean"`), those errors stay as seams; the cosine blending of
overlapping patches gives the borders almost no weight. The reference is the
same network run on the whole sweep at once, with the rays continued
periodically across north as the patches do.

```{code-cell} ipython3
periodic = np.concatenate([dbz.values[-2:], dbz.values, dbz.values[:2]])
whole = model.run(periodic[None, None])["y"][0, 0, 2:-2] * 60.0
tiles, tiles_index = ml.polar_patches(dbz, (64, 128), stride=(64, 128))
seams = ml.reassemble(model.run(tiles[:, None])["y"][:, 0], tiles_index, blend="mean")
blended = smoothed.values
inner = np.s_[:, 2:-2]  # the whole-sweep run also sees zero padding at its ends
for name, field in (("no overlap", seams.values * 60.0), ("cosine blend", blended)):
    err = np.abs(field - whole)[inner]
    print(f"{name:13s} max |error| {err.max():6.2f} dB, mean {err.mean():.4f} dB")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
panels = [
    (whole, "box filter on the whole sweep (dBZ)", "ChaseSpectral", (-10, 70)),
    (seams.values * 60.0 - whole, "no overlap − whole (dB)", "RdBu_r", (-5, 5)),
    (blended - whole, "cosine blend − whole (dB)", "RdBu_r", (-5, 5)),
]
for ax, (field, title, cmap, (vmin, vmax)) in zip(axes, panels):
    pm = ax.pcolormesh(sweep.x / 1e3, sweep.y / 1e3, field, cmap=cmap,
                       vmin=vmin, vmax=vmax)
    fig.colorbar(pm, ax=ax)
    ax.set(xlim=(-150, 50), ylim=(-100, 100), aspect="equal", title=title,
           xlabel="East (km)", ylabel="North (km)")
```

## A whole volume in one call

On a DataTree all sweeps holding the fields are cut in one call to the
compiled kernel (sweeps lacking a field, like the split-cut Doppler sweeps
without ZDR, are skipped), and `reassemble` returns a DataTree.

```{code-cell} ipython3
for engine in ("numpy", "compiled"):
    start = time.perf_counter()
    vol_patches, vol_index = ml.polar_patches(
        dtree, (64, 128), variables=["DBZH", "ZDR"], engine=engine
    )
    volume = ml.reassemble(vol_patches, vol_index, engine=engine)
    print(f"{engine:8s} {len(vol_patches)} patches from {len(vol_index.paths)} "
          f"sweeps and back in {time.perf_counter() - start:.2f} s")
print(vol_index.paths)
```
