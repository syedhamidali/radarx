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

# Doppler Velocity Dealiasing

+++

A Doppler radar measures radial velocity only within the Nyquist interval
$[-V_n, V_n]$. Faster winds fold over: a true velocity $v$ is reported as
$v - 2kV_n$ for some integer fold $k$. Dealiasing finds $k$ for every gate,
which is needed before any wind retrieval.

`radarx.retrieve.dealias_velocity` (or `dtree.radarx.dealias()`) uses a
region-based method on the polar sweep grid:

1. neighbouring gates with similar velocities are joined into regions
   (union-find), so fold lines separate regions (Jing and Wiener 1993);
2. the integer fold of every region minimises the squared velocity jumps
   across all region boundaries, by coordinate descent on single regions and
   on blocks of regions;
3. the absolute fold is fixed by the sweep below (volume continuity, as in
   James and Houze 2001), an optional wind profile (Eilts and Smith 1990), or
   an in-sweep VAD fit, and a final check refolds isolated gates.

A multithreaded C++ kernel does the work for all sweeps of a volume in one
call; an equivalent NumPy implementation gives identical results.

```{code-cell} ipython3
import time

import matplotlib.pyplot as plt
import numpy as np
import xradar as xd
from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

import radarx  # noqa: F401
from radarx.io.aws_data import download_file
```

## A strongly aliased case

The KGWX (Columbus, Mississippi) volume from 30 March 2022 shows a strong
low-level jet, with radial velocities well beyond the Nyquist velocity of
26.4 m/s.

```{code-cell} ipython3
path = download_file(
    "unidata-nexrad-level2", "2022/03/30/KGWX/KGWX20220330_234639_V06", "."
)
dtree = xd.io.open_nexradlevel2_datatree(path)
```

xradar does not yet expose the Nyquist velocity of NEXRAD data, so we read it
from the radial headers and attach it to each sweep as the
`nyquist_velocity` coordinate, where radarx looks for it. (It can also be
passed as `nyquist_velocity=...`.)

```{code-cell} ipython3
with NEXRADLevel2File(path) as nf:
    nyquist = [
        h["msg_31_data_header"]["RAD"]["nyquist_vel"] / 100.0
        for h in nf.msg_31_data_header
    ]
sweeps = [name for name in dtree.children if name.startswith("sweep")]
for name, value in zip(sweeps, nyquist):
    dtree[name] = dtree[name].to_dataset().assign_coords(nyquist_velocity=value)
dtree = dtree.xradar.georeference()
```

## Dealias the volume

Like every radarx retrieval, `dealias` returns its products only: a DataTree
with the root of the volume and, for each sweep, the dealiased velocity
`VRADH_dealiased` (the name never replaces the measured `VRADH`).
`dtree.radarx.assign` adds the products to the matching sweeps of the volume.

```{code-cell} ipython3
start = time.perf_counter()
products = dtree.radarx.dealias("VRADH")
print(f"dealiased the volume in {time.perf_counter() - start:.2f} s")
dealiased = dtree.radarx.assign(products)
dealiased["sweep_1"]
```

```{code-cell} ipython3
def residual_jumps(values, nyquist):
    """Share of neighbouring gate pairs that differ by more than Vn."""
    bad = total = 0
    for a, b in ((values[:, 1:], values[:, :-1]), (np.roll(values, -1, 0), values)):
        ok = np.isfinite(a) & np.isfinite(b)
        bad += np.count_nonzero(np.abs(a - b)[ok] > nyquist)
        total += np.count_nonzero(ok)
    return bad / total


for name in sweeps:
    ds = dealiased[name].to_dataset()
    if "VRADH" not in ds:
        continue
    vn = float(ds.nyquist_velocity)
    raw = ds.VRADH.where(abs(ds.VRADH) <= vn).values
    print(
        f"{name:9s} elevation {float(ds.elevation.median()):5.2f}  "
        f"jumps > Vn: raw {residual_jumps(raw, vn):.4f}  "
        f"dealiased {residual_jumps(ds.VRADH_dealiased.values, vn):.5f}"
    )
```

## Before and after

Gates without data, or flagged as range folded, are left blank.

```{code-cell} ipython3
def plot_sweep(name, extent):
    ds = dealiased[name].to_dataset().sortby("azimuth")
    vn = float(ds.nyquist_velocity)
    raw = ds.VRADH.where(abs(ds.VRADH) <= vn)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, da, title in zip(
        axes,
        [raw, ds.VRADH_dealiased],
        [f"measured (Nyquist {vn:.1f} m/s)", "dealiased"],
    ):
        mesh = da.assign_coords(x=ds.x / 1e3, y=ds.y / 1e3).plot.pcolormesh(
            x="x", y="y", ax=ax, cmap="RdBu_r", vmin=-60, vmax=60, add_colorbar=False
        )
        ax.set_title(title)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("")
        ax.set_aspect("equal")
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
    axes[0].set_ylabel("y (km)")
    fig.colorbar(mesh, ax=axes, label="radial velocity (m/s)")
    fig.suptitle(f"KGWX {name}, elevation {float(ds.elevation.median()):.1f} deg")
    plt.show()


plot_sweep("sweep_1", 250)
```

```{code-cell} ipython3
plot_sweep("sweep_21", 60)
```

## Reference winds

Without a reference, the absolute fold of the lowest sweep comes from the
assumption that the mean radial velocity of a sweep is close to zero, which
holds for a horizontally uniform wind seen all around the radar. When the
echo covers only part of the circle, pass a wind profile (sounding, VAD or
model) as an `xarray.Dataset` with `u` and `v` on a `height` coordinate:

```{code-cell} ipython3
import xarray as xr

height = np.arange(0.0, 15e3, 500.0)
profile = xr.Dataset(
    {"u": ("height", np.full(height.size, 15.0)), "v": ("height", np.full(height.size, 20.0))},
    coords={"height": height},
)
with_profile = dtree.radarx.dealias("VRADH", wind_profile=profile)
```

## References

- Jing, Z., and G. Wiener, 1993: Two-dimensional dealiasing of Doppler
  velocities. *J. Atmos. Oceanic Technol.*, **10**, 798–808,
  <https://doi.org/10.1175/1520-0426(1993)010<0798:TDDODV>2.0.CO;2>
- Eilts, M. D., and S. D. Smith, 1990: Efficient dealiasing of Doppler
  velocities using local environment constraints. *J. Atmos. Oceanic
  Technol.*, **7**, 118–128,
  <https://doi.org/10.1175/1520-0426(1990)007<0118:EDODVU>2.0.CO;2>
- James, C. N., and R. A. Houze, 2001: A real-time four-dimensional Doppler
  dealiasing scheme. *J. Atmos. Oceanic Technol.*, **18**, 1674–1683,
  <https://doi.org/10.1175/1520-0426(2001)018<1674:ARTFDD>2.0.CO;2>
- Browning, K. A., and R. Wexler, 1968: The determination of kinematic
  properties of a wind field using Doppler radar. *J. Appl. Meteor.*, **7**,
  105–113, <https://doi.org/10.1175/1520-0450(1968)007<0105:TDOKPO>2.0.CO;2>
