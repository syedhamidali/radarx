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

# Non-Meteorological Echo Filtering

Weather radars see more than precipitation: insects and birds, ground and sea
clutter, anomalous propagation, second-trip echo and noise. Such echo must be
removed before KDP, hydrometeor classification, rain rates or gridding.

`radarx.retrieve.echo_mask` (or `.radarx.echo_mask()` on a sweep or a volume)
classifies every gate with a fuzzy-logic score in the spirit of Gourley et al.
(2007) and Krause (2016). Features computed along the ray over a short window:

- the mean $\rho_{hv}$ (precipitation: above 0.95),
- the mean $Z_{DR}$ (insects and birds: several dB up to more than 10 dB),
- the texture (standard deviation) of $Z_{DR}$ and of $\Phi_{DP}$,
- the texture of the reflectivity and its "spin change" (Steiner and Smith
  2002), which pick out ground clutter.

Each feature gives a membership between 0 and 1; their weighted mean is
averaged over 3 x 3 gates and compared with a threshold. Small isolated
regions are removed as speckle. NEXRAD no-data codes are recognised, and the
Doppler cuts of NEXRAD split cuts (no polarimetric data) take the class of
the surveillance cut at the same elevation.

`radarx.retrieve.apply_mask` (or `.radarx.apply_mask()`) then sets the
non-meteorological gates of chosen fields to NaN.

A compiled kernel classifies all gates of all sweeps in one multithreaded
call.

```{code-cell} ipython3
import time

import cmweather  # noqa: F401  radar colormaps
import matplotlib.pyplot as plt
import numpy as np
import xradar as xd
from matplotlib.colors import ListedColormap

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.io.aws_data import download_file
from radarx.retrieve import echo_mask
```

## A squall line with biological echo

The KGWX (Columbus, Mississippi) volume from 30 March 2022 shows a squall line
west of the radar. East of it, ahead of the line, a region of weak echo with
$Z_{DR}$ above 5 dB and low, noisy $\rho_{hv}$: insects or birds.

```{code-cell} ipython3
path = download_file(
    "unidata-nexrad-level2", "2022/03/30/KGWX/KGWX20220330_234639_V06", "."
)
dtree = xd.io.open_nexradlevel2_datatree(path)
dtree.load()
```

```{code-cell} ipython3
start = time.perf_counter()
qc = dtree.radarx.echo_mask()
print(f"{time.perf_counter() - start:.2f} s for the whole volume")
qc["sweep_0"].ds
```

```{code-cell} ipython3
clean = dtree.radarx.apply_mask(qc)
```

## Before and after

```{code-cell} ipython3
def xy(ds):
    az = np.deg2rad(ds.azimuth.values)[:, None]
    r = ds.range.values[None, :] / 1000.0
    return r * np.sin(az), r * np.cos(az)


sweep = dtree["sweep_0"].ds
x, y = xy(sweep)
classes = ListedColormap(["#2a78d6", "#eb6834", "#eda100"])

fig, axs = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)
panels = [
    (sweep.DBZH.where(sweep.DBZH > -32), "DBZH before", dict(vmin=-10, vmax=65, cmap="ChaseSpectral")),
    (sweep.ZDR.where(sweep.DBZH > -32), "ZDR before", dict(vmin=-2, vmax=8, cmap="HomeyerRainbow")),
    (qc["sweep_0"].ds.ECHO_CLASS.where(qc["sweep_0"].ds.ECHO_CLASS > 0), "ECHO_CLASS", dict(vmin=0.5, vmax=3.5, cmap=classes)),
    (clean["sweep_0"].ds.DBZH, "DBZH after apply_mask", dict(vmin=-10, vmax=65, cmap="ChaseSpectral")),
]
for ax, (data, title, kw) in zip(axs.flat, panels):
    pm = ax.pcolormesh(x, y, data, shading="auto", **kw)
    cb = fig.colorbar(pm, ax=ax)
    if title == "ECHO_CLASS":
        cb.set_ticks([1, 2, 3], labels=["meteorological", "non-meteorological", "speckle"])
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(-250, 250)
    ax.set_ylim(-250, 250)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
```

The biological echo east of the radar is removed, the squall line and the
stratiform rain are kept. How much of each echo type is removed on the lowest
sweep:

```{code-cell} ipython3
cls = qc["sweep_0"].ds.ECHO_CLASS.values
z, zdr = sweep.DBZH.values, sweep.ZDR.values
regions = {
    "biological (east, ZDR > 4 dB, Z < 20 dBZ)": (x > 85) & (x < 130) & (abs(y) < 45) & (zdr > 4) & (z < 20),
    "squall line (Z >= 35 dBZ)": (x > -170) & (x < 20) & (z >= 35),
    "light rain west (5-25 dBZ)": (x < 0) & (z > 5) & (z < 25),
}
for name, region in regions.items():
    sel = region & (cls > 0)
    print(f"{name:45s} {sel.sum():7d} gates, removed {100 * np.mean(cls[sel] != 1):5.1f} %")
```

## The score

`METEO_SCORE` is the weighted mean membership; gates scoring at least
`threshold` (default 0.6) are meteorological.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7, 6))
pm = ax.pcolormesh(x, y, qc["sweep_0"].ds.METEO_SCORE, vmin=0, vmax=1, cmap="RdBu", shading="auto")
fig.colorbar(pm, ax=ax, label="METEO_SCORE")
ax.set_aspect("equal")
ax.set_xlim(40, 160)
ax.set_ylim(-60, 60)
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)");
```

## Velocity of the Doppler cut

The Doppler cut of the lowest elevation (`sweep_1`) holds no polarimetric
fields; its gates take the class of the surveillance cut `sweep_0`, so the
radial velocity of the biological echo is masked too.

```{code-cell} ipython3
doppler = dtree["sweep_1"].ds
xd1, yd1 = xy(doppler)
fig, axs = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
for ax, data, title in zip(
    axs,
    [doppler.VRADH.where(doppler.VRADH > -63.9), clean["sweep_1"].ds.VRADH],
    ["VRADH before", "VRADH after apply_mask"],
):
    pm = ax.pcolormesh(xd1, yd1, data, vmin=-40, vmax=40, cmap="balance", shading="auto")
    fig.colorbar(pm, ax=ax, label="m/s")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(-250, 250)
    ax.set_ylim(-250, 250)
```

## Options

- `limits` and `weights` change the membership of each feature
  (`radarx.retrieve.qc.DEFAULT_LIMITS`, `DEFAULT_WEIGHTS`),
- `threshold`, `window` (km), `min_size` (gates) and `spin_threshold` (dB),
- `nodata` sets the no-data floors (NEXRAD codes are detected automatically),
- `snr` / `snr_min` use a signal-to-noise ratio field where available,
- `engine` and `n_threads` choose the implementation.

## References

- Gourley, J. J., P. Tabary, and J. Parent du Chatelet, 2007: A fuzzy logic
  algorithm for the separation of precipitating from nonprecipitating echoes
  using polarimetric radar observations. *J. Atmos. Oceanic Technol.*, **24**,
  1439-1451, https://doi.org/10.1175/JTECH2035.1
- Krause, J. M., 2016: A simple algorithm to discriminate between
  meteorological and nonmeteorological radar echoes. *J. Atmos. Oceanic
  Technol.*, **33**, 1875-1885, https://doi.org/10.1175/JTECH-D-15-0239.1
- Steiner, M., and J. A. Smith, 2002: Use of three-dimensional reflectivity
  structure for automated detection and removal of nonprecipitating echoes in
  radar data. *J. Atmos. Oceanic Technol.*, **19**, 673-686,
  https://doi.org/10.1175/1520-0426(2002)019<0673:UOTDRS>2.0.CO;2
