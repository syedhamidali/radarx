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

# Multi-Radar Gridding and Network Calibration

+++

Mosaics and multi-Doppler wind retrievals need several radars on one Cartesian
grid. `radarx.grid.grid_radars` cone-grids every radar onto a shared grid (one
origin, one azimuthal equidistant projection, heights above sea level) and keeps
each radar separately along a `radar` dimension, together with the beam
geometry from every radar to every cell. On top of that grid

* `radarx.grid.network_bias` estimates the relative reflectivity (or ZDR)
  calibration of the radars from their overlap regions and reconciles the pairs
  across the network against a reference radar;
* `radarx.grid.merge_radars` merges the radars into one field with range,
  beam-elevation and time weights.

Fields of radars that scanned at different times can be moved to a common
analysis time along the storm motion (`motion=`, see the advection correction
notebook). The cone gridding of all radars and fields, the geodesics from every
radar to every column, the beam geometry, the merge and the comparison
histograms run in compiled, multithreaded C++ kernels.

Here we use three NEXRAD radars around a squall line in Mississippi on
30 March 2022 (KGWX, KNQA, KDGX), with the next KGWX volume for the storm motion.

**References**

- Zhang, J., K. Howard, and J. J. Gourley, 2005: Constructing three-dimensional
  multiple-radar reflectivity mosaics: Examples of convective storms and
  stratiform rain echoes. *J. Atmos. Oceanic Technol.*, **22**, 30-42,
  https://doi.org/10.1175/JTECH-1689.1
- Lakshmanan, V., T. Smith, K. Hondl, G. J. Stumpf, and A. Witt, 2006: A
  real-time, three-dimensional, rapidly updating, heterogeneous radar merger
  technique for reflectivity, velocity, and derived products. *Wea.
  Forecasting*, **21**, 802-823, https://doi.org/10.1175/WAF942.1
- Seo, B.-C., W. F. Krajewski, and J. A. Smith, 2014: Four-dimensional
  reflectivity data comparison between two ground-based radars: methodology
  and statistical analysis. *Hydrol. Sci. J.*, **59**, 1320-1334,
  https://doi.org/10.1080/02626667.2013.839872
- Vincenty, T., 1975: Direct and inverse solutions of geodesics on the
  ellipsoid with application of nested equations. *Surv. Rev.*, **23**, 88-93,
  https://doi.org/10.1179/sre.1975.23.176.88

```{code-cell} ipython3
import cmweather  # noqa
import fsspec
import matplotlib.pyplot as plt
import numpy as np
import xradar as xd

import radarx as rx
```

## Read the volumes

NEXRAD marks missing data with special codes; we mask them before gridding.

```{code-cell} ipython3
files = {
    "KGWX": "KGWX20220330_234639_V06",
    "KNQA": "KNQA20220330_234905_V06",
    "KDGX": "KDGX20220330_234843_V06",
}
nodata = {"DBZH": -32.0, "VRADH": -63.9, "ZDR": -12.9, "RHOHV": 0.21}


def read(name):
    local_file = fsspec.open_local(
        f"simplecache::s3://unidata-nexrad-level2/2022/03/30/{name[:4]}/{name}",
        s3={"anon": True},
        filecache={"cache_storage": "."},
    )
    dtree = xd.io.open_nexradlevel2_datatree(local_file)
    for sweep in [n for n in dtree.children if n.startswith("sweep")]:
        ds = dtree[sweep].to_dataset()
        ds = ds[[v for v in nodata if v in ds]]
        for field, code in nodata.items():
            if field in ds:
                ds[field] = ds[field].where(ds[field] > code)
        dtree[sweep] = ds
    return dtree.load()


radars = [read(name) for name in files.values()]
```

## Storm motion

The radars scan at different times, so we estimate the storm motion from two
consecutive KGWX volumes and move every radar to the KGWX volume time.

```{code-cell} ipython3
x = np.arange(-260e3, 200e3 + 1, 2000.0)
y = np.arange(-200e3, 220e3 + 1, 2000.0)
z = np.arange(1000.0, 8000.0 + 1, 500.0)

g0 = rx.grid.grid_cones(radars[0], "DBZH", x, y, z)
g1 = rx.grid.grid_cones(read("KGWX20220330_235324_V06"), "DBZH", x, y, z)
motion = rx.retrieve.estimate_motion(g0, g1, tile=100e3)
print(f"mean motion u = {float(motion.u.mean()):.1f} m/s, v = {float(motion.v.mean()):.1f} m/s")
```

## Grid all radars onto one grid

The grid origin is KGWX (the default origin is the first radar). Gates with a
co-polar correlation below 0.9 are dropped, and cells farther than 230 km from a
radar are discarded.

```{code-cell} ipython3
grid = rx.grid.grid_radars(
    radars,
    x,
    y,
    z,
    data_vars=["DBZH", "ZDR", "VRADH"],
    motion=motion,
    rhohv_min=0.9,
    max_range=230e3,
)
grid
```

Every radar keeps its own fields on `(radar, z, y, x)`. `azimuth`, `elevation`
and `range` give the direction of the beam at every cell (in the grid frame) and
the slant range from each radar, `radar_x`, `radar_y`, `radar_z` the radar
positions and `time` each volume time: everything a multi-Doppler retrieval
needs. A radial velocity relates to the wind `(u, v, w)` on the grid axes by
`u sin(az) cos(el) + v cos(az) cos(el) + w sin(el)`.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
grid.azimuth.sel(radar="KNQA").plot(ax=axes[0], cmap="twilight", vmin=0, vmax=360)
grid.elevation.sel(radar="KNQA", z=3000.0).plot(ax=axes[1], vmax=10)
(grid.range.sel(radar="KNQA", z=3000.0) / 1e3).plot(ax=axes[2])
for ax, title in zip(axes, ["beam azimuth", "beam elevation, z = 3 km", "slant range (km), z = 3 km"]):
    ax.set_title(f"KNQA {title}")
    ax.set_aspect("equal")
```

## Relative calibration

The overlap regions are compared below the melting layer (1 to 3 km). The pair
biases (medians of the differences) are reconciled by weighted least squares
with KGWX as the reference.

```{code-cell} ipython3
bias = rx.grid.network_bias(grid, "DBZH", reference="KGWX", z_range=(1000, 3000))
print(bias.bias.to_series().round(2))
bias.pair_bias.round(2).to_pandas()
```

```{code-cell} ipython3
zdr_bias = rx.grid.network_bias(grid, "ZDR", reference="KGWX", z_range=(1000, 3000))
zdr_bias.bias.to_series().round(2)
```

The overlap of KGWX and KDGX before and after removing the bias:

```{code-cell} ipython3
corrected = grid.DBZH - bias.bias
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
for ax, data, label in zip(axes, (grid.DBZH, corrected), ("before", "after")):
    a = data.sel(radar="KGWX", z=slice(1000, 3000)).values.ravel()
    b = data.sel(radar="KDGX", z=slice(1000, 3000)).values.ravel()
    ok = np.isfinite(a) & np.isfinite(b) & (a > 15) & (b > 15)
    ax.hist2d(a[ok], b[ok], bins=np.arange(15, 55, 1), cmin=1, cmap="viridis")
    ax.plot([15, 55], [15, 55], "k--")
    ax.set_title(f"{label}: median KDGX - KGWX = {np.median(b[ok] - a[ok]):+.2f} dB")
    ax.set_xlabel("KGWX DBZH (dBZ)")
    ax.set_ylabel("KDGX DBZH (dBZ)")
```

## Merged CAPPI

`merge_radars` averages the radars with weights that favour the nearer radar,
cells close to a beam axis and the radar observed closest to the analysis time.

```{code-cell} ipython3
merged = rx.grid.merge_radars(grid, "DBZH", bias=bias)
fig, ax = plt.subplots(figsize=(7, 6))
merged.DBZH.sel(z=2000.0).plot(ax=ax, cmap="ChaseSpectral", vmin=-10, vmax=70)
ax.plot(grid.radar_x, grid.radar_y, "k^")
for name, rx_, ry_ in zip(grid.radar.values, grid.radar_x.values, grid.radar_y.values):
    ax.annotate(name, (rx_, ry_), xytext=(3, 3), textcoords="offset points")
ax.set_aspect("equal")
ax.set_title("merged DBZH at 2 km MSL, calibrated to KGWX");
```
