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

# Lightning Mapping Array: flashes, gridded products and lightning jumps

A Lightning Mapping Array (LMA; [Rison et al. 1999](https://doi.org/10.1029/1999GL010856))
locates the VHF radiation of lightning in three dimensions, typically
thousands of "sources" per flash. radarx reads the LMA source files, groups
the sources into flashes and puts total lightning on radar grids and tracked
storm cells:

- `radarx.io.read_lma` reads the ASCII `.dat(.gz)` source files of the
  `lma_analysis` program into an `xarray.Dataset` (sources on
  `number_of_events`, the xlma-python layout);
- `radarx.retrieve.cluster_flashes` groups sources closer than 3 km and
  0.15 s in normalized space-time distance into flashes
  ([Fuchs et al. 2016](https://doi.org/10.1002/2015JD024663));
- `radarx.retrieve.grid_lightning` counts sources, flash extent and flash
  initiations in the boxes of a radarx grid
  ([Bruning and MacGorman 2013](https://doi.org/10.1175/JAS-D-12-0289.1));
- `radarx.retrieve.cell_flash_rate` gives flash rates and source height
  distributions per cell of a tracked-storm mask;
- `radarx.retrieve.lightning_jump` finds lightning jumps with the "2σ"
  algorithm ([Schultz et al. 2009](https://doi.org/10.1175/2009JAMC2237.1)).

Clustering, gridding and the cell counts run in a compiled, multithreaded
kernel.

```{code-cell} ipython3

import matplotlib.pyplot as plt
import numpy as np
import pooch
import xarray as xr

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.io import read_lma
from radarx.retrieve import (
    cell_flash_rate,
    cluster_flashes,
    grid_lightning,
    lightning_jump,
    vertical_source_distribution,
)
```

## Real LMA data

The xlma-python package ships one minute of West Texas LMA data (one file
per second) as an example. `read_lma` reads several files at once and can
filter the sources by the reduced chi-square of their location and the number
of contributing stations:

```{code-cell} ipython3
pooch.get_logger().setLevel("WARNING")
base = (
    "https://raw.githubusercontent.com/deeplycloudy/xlma-python/"
    "97f8aaa88d8730dad62686d076e8bd74c4007be6/examples/data/"
)
seconds = [1, 2, 4, 5, 6, 7, 11, 12, 13, 14, 15, 18, 19, 20, 22, 23, 24, 25]
files = [
    pooch.retrieve(base + f"WTLMA_231224_0057{s:02d}_0001.dat.gz", known_hash=None)
    for s in seconds
]
lma = read_lma(files, max_chi2=2.0, min_stations=6)
lma
```

```{code-cell} ipython3
flashes = cluster_flashes(lma)
print(flashes.sizes["number_of_flashes"], "flashes;",
      int((flashes.flash_event_count >= 10).sum()), "with at least 10 sources")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
big = flashes.flash_event_count.values[flashes.event_parent_flash_id.values] >= 10
sc = ax[0].scatter(flashes.event_longitude[big], flashes.event_latitude[big], s=2,
                   c=flashes.event_parent_flash_id[big] % 20, cmap="tab20")
ax[0].set(xlabel="longitude", ylabel="latitude", title="sources coloured by flash")
ax[1].scatter(flashes.event_time[big], flashes.event_altitude[big] / 1e3, s=2,
              c=flashes.event_parent_flash_id[big] % 20, cmap="tab20")
ax[1].set(ylabel="altitude (km MSL)", title="time-height");
```

Gridded on 2-km boxes around the network centre, the flash extent density
counts each flash once in every box it touches:

```{code-cell} ipython3
x = np.arange(-150e3, 150.1e3, 2e3)
products = grid_lightning(
    flashes,
    x=x,
    y=x,
    latitude=float(lma.network_center_latitude),
    longitude=float(lma.network_center_longitude),
    interval="1min",
    min_sources=10,
)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
for a, name in zip(ax, ["source_density", "flash_extent_density", "flash_initiation_density"]):
    field = products[name].sum("time")
    field.where(field > 0).plot(ax=a, x="lon", y="lat", cmap="magma_r")
    a.set_title(name.replace("_", " "))
```

## A synthetic storm with a lightning jump

To show the cell products, we simulate a storm cell moving east at
15 m s$^{-1}$ whose flash rate sits near 12 flashes min$^{-1}$ for 20 min and
then jumps to 35 flashes min$^{-1}$; each flash is a cloud of 10–60 sources
1–2 km across between 5 and 12 km. A second, weak cell has 3 flashes
min$^{-1}$.

```{code-cell} ipython3
rng = np.random.default_rng(42)
lat0, lon0 = 33.6, -88.5
start = np.datetime64("2022-03-30T23:00:00", "ns")


def storm(rate_per_min, x0, y0, u, minutes):
    t, lat, lon, alt = [], [], [], []
    for m, rate in zip(range(minutes), rate_per_min):
        for tf in np.sort(rng.uniform(60 * m, 60 * (m + 1), rng.poisson(rate))):
            n = rng.integers(10, 60)
            xf = x0 + u * tf + rng.normal(0, 2e3)
            yf = y0 + rng.normal(0, 2e3)
            t.append(tf + np.sort(rng.uniform(0, 0.3, n)))
            lat.append(lat0 + (yf + rng.normal(0, 700, n)) / 111.2e3)
            lon.append(lon0 + (xf + rng.normal(0, 700, n)) / (111.2e3 * np.cos(np.radians(lat0))))
            alt.append(rng.choice([rng.normal(9e3, 1e3, n), rng.normal(6e3, 600, n)]))
    return [np.concatenate(v) for v in (t, lat, lon, alt)]


minutes = 40
strong = np.r_[rng.normal(12, 3, 22), np.linspace(18, 35, 6), np.full(12, 33.0)]
parts = [storm(strong, -20e3, 0.0, 15.0, minutes), storm(np.full(minutes, 3.0), -20e3, -25e3, 15.0, minutes)]
t, lat, lon, alt = (np.concatenate(v) for v in zip(*parts))
order = np.argsort(t)
sources = xr.Dataset(
    {"event_altitude": ("number_of_events", alt[order])},
    coords={
        "event_time": ("number_of_events", start + (t[order] * 1e9).astype("timedelta64[ns]")),
        "event_latitude": ("number_of_events", lat[order]),
        "event_longitude": ("number_of_events", lon[order]),
    },
)
synthetic = cluster_flashes(sources)
print(synthetic.sizes["number_of_events"], "sources,", synthetic.sizes["number_of_flashes"], "flashes")
```

A tracked-cell mask, here two discs following the cells every 5 min on a
1-km grid centred at the radar (in practice e.g. a tobac segmentation of
gridded reflectivity, with the same label for a cell at all times):

```{code-cell} ipython3
x = np.arange(-60e3, 60.1e3, 1e3)
mtimes = start + np.arange(0, minutes * 60 + 1, 300).astype("timedelta64[s]")
el = (mtimes - start).astype(float) * 1e-9
xx, yy = np.meshgrid(x, x)
mask = np.zeros((mtimes.size, x.size, x.size), dtype=np.int32)
for k, s in enumerate(el):
    mask[k][np.hypot(xx - (-20e3 + 15 * s), yy) < 8e3] = 1
    mask[k][np.hypot(xx - (-20e3 + 15 * s), yy + 25e3) < 8e3] = 2
cells = xr.DataArray(
    mask, dims=("time", "y", "x"),
    coords={"time": mtimes, "y": x, "x": x, "latitude": lat0, "longitude": lon0},
)
minute_edges = start + np.arange(minutes + 1).astype("timedelta64[m]")
rates = cell_flash_rate(
    synthetic, cells, time_edges=minute_edges, z=np.arange(500.0, 16e3, 1000.0)
)
rates
```

```{code-cell} ipython3
jumps = rates.flash_rate.radarx.lightning_jump()

fig, ax = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True,
                       gridspec_kw={"width_ratios": [2, 1]})
for c, color in zip(rates.cell.values, ["k", "tab:blue"]):
    ax[0].plot(rates.time, rates.flash_rate.sel(cell=c), color=color, alpha=0.3)
    j = jumps.sel(cell=c)
    ax[0].plot(j.time, j.flash_rate, color=color, lw=2, label=f"cell {c}")
    for s in j.time.values[j.jump_start.values]:
        ax[0].axvline(s, color="tab:red")
    prof = rates.source_count.sel(cell=c).sum("time")
    ax[1].plot(prof / prof.sum(), rates.z / 1e3, color=color)
ax[0].axhline(10, color="0.5", ls=":")
ax[0].set(ylabel="flashes min$^{-1}$", title="flash rates (thin: 1 min, thick: 2 min) and jumps (red)")
ax[0].legend()
ax[1].set(xlabel="fraction of sources", ylabel="height (km MSL)", title="vertical source distribution");
jumps.sigma_level.sel(cell=1).to_series().round(1).tail(12)
```

The 2σ algorithm needs 14 min of history (six 2-min periods), a flash rate of
at least 10 flashes min$^{-1}$ and a rate of change (DFRDT) at least twice the
standard deviation of the five previous ones; the weak cell never qualifies.

## Lightning on a 3-D grid

With heights, `grid_lightning` gives flash extent density by height on the
same boxes as a radarx grid (e.g. one from `radarx.grid.grid_cones`, whose
origin it reads from the grid), and `vertical_source_distribution` gives the
height distribution of all sources and flash initiations:

```{code-cell} ipython3
grid = xr.Dataset(
    coords={"x": x, "y": x, "z": np.arange(1e3, 15.1e3, 1e3), "latitude": lat0, "longitude": lon0}
)
fed3d = grid_lightning(synthetic, grid, z=True, interval="10min")
profile = vertical_source_distribution(synthetic, grid.z.values, interval="10min")

fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
fed3d.flash_extent_density.isel(time=-1).sel(y=0, method="nearest").plot(ax=ax[0], cmap="magma_r")
ax[0].set_title("flash extent density, last 10 min, y = 0")
profile.source_count.plot(ax=ax[1], x="time", y="z", cmap="viridis")
ax[1].set_title("sources per height and 10 min");
```
