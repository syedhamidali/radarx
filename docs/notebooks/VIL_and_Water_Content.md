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

# VIL, echo tops and water content

Four column products of the reflectivity of a radar volume, in
`radarx.retrieve`:

- `vil`: vertically integrated liquid, the liquid water of rain in a column in
  kg m$^{-2}$ ([Greene and Clark 1972](https://doi.org/10.1175/1520-0493(1972)100<0548:VILWNA>2.3.CO;2)),
  $\mathrm{VIL} = 3.44\times10^{-6}\sum_i \left[(Z_i+Z_{i+1})/2\right]^{4/7}\Delta h_i$
  with $Z$ in mm$^6$ m$^{-3}$ and $\Delta h$ in m;
- `echo_top`: the height of the highest 18 dBZ echo, interpolated in dBZ between
  the beams that bracket the threshold
  ([Lakshmanan et al. 2013](https://doi.org/10.1175/WAF-D-12-00084.1));
- `vil_density`: the VIL divided by the echo top
  ([Amburn and Wolf 1997](https://doi.org/10.1175/1520-0434(1997)012<0473:VDAAHI>2.0.CO;2));
- `liquid_water_content`: the water content of rain at every gate, from the
  same power law or from the gamma drop size distribution of `dsd`.

A column is a list of reflectivity samples at known heights. For a polar
volume (a `DataTree` of sweeps) it is an (azimuth, ground range) position of
the lowest sweep, with one sample from every other sweep at its beam height;
for a grid it is an (y, x) position, and for a quasi-vertical profile (QVP) one
profile. Between two beams the reflectivity varies linearly with height, nothing
is assumed above the highest valid beam, and the layer below the lowest beam is
left out by default, so the VIL is a lower bound (`VIL_LOWER_BOUND`) that
`fill_below=True` turns into an estimate. The integrals run in a compiled,
multithreaded kernel.

```{code-cell} ipython3
import warnings

import cmweather  # noqa: F401  radar colormaps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xradar as xd
from open_radar_data import DATASETS

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.io.aws_data import download_file
from radarx.retrieve import (
    echo_top,
    estimate_kdp,
    liquid_water_content,
    melting_layer,
    qvp_timeseries,
    vil,
    vil_density,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
```

## A squall line seen by KGWX

The KGWX WSR-88D (Columbus, Mississippi) volume of 30 March 2022, 23:59 UTC,
from the open NEXRAD archive on AWS. The no-data codes (DBZH at or below
−32 dBZ) are masked and the range is cut at 150 km. The volume keeps all cuts
of the scan; where an elevation was scanned twice the cut that reaches farther
is used.

```{code-cell} ipython3
key = "2022/03/30/KGWX/KGWX20220330_235959_V06"
path = download_file("unidata-nexrad-level2", key, "data")
drop = ["WRADH", "CCORH", "CFP", "ZDR", "PHIDP", "RHOHV"]
volume = xd.io.open_nexradlevel2_datatree(path, drop_variables=drop)


def prepare(ds):
    """Mask the no-data codes and keep the first 150 km."""
    if "range" not in ds.dims:
        return ds
    ds = ds.sel(range=slice(None, 150e3))
    return ds.assign(DBZH=ds.DBZH.where(ds.DBZH > -32).astype("float32"))


volume = volume.map_over_datasets(prepare)
sweeps = [n for n in volume.children if n.startswith("sweep") and "DBZH" in volume[n].ds]
print(len(sweeps), "sweeps with reflectivity, elevations:",
      " ".join(f"{float(volume[n].sweep_fixed_angle):.1f}" for n in sweeps))
```

## VIL, echo top and VIL density

Each call takes the volume and returns a product on the polar grid of the lowest
sweep. `vil` returns a dataset with the VIL, the height of the lowest beam used
(the lower integration limit) and two flags, `VIL_LOWER_BOUND` (the layer under
the lowest beam is missing) and `VIL_TOP_TRUNCATED` (echo continues above the
highest beam); the reflectivity is capped at 56 dBZ (`dbz_cap`), as in the
operational product.

```{code-cell} ipython3
vil_polar = vil(volume)
top = echo_top(volume)
density = vil_density(volume)
vil_polar
```

The same products are available as methods of the volume,
`volume.radarx.vil()`, `.radarx.echo_top()` and `.radarx.vil_density()`.

```{code-cell} ipython3
low = volume[sweeps[0]].to_dataset()
az = np.deg2rad(vil_polar.azimuth.values)[:, None]
s = vil_polar.ground_range.values[None, :]
x, y = s * np.sin(az) / 1e3, s * np.cos(az) / 1e3
rays = np.argsort(vil_polar.azimuth.values)  # no seam where the scan started

panels = [
    (low.DBZH, "$Z_H$ at 0.5° (dBZ)", dict(cmap="ChaseSpectral", vmin=-10, vmax=70)),
    (vil_polar.VIL, "VIL (kg m$^{-2}$)", dict(cmap="viridis", vmin=0, vmax=30)),
    (top / 1e3, "18 dBZ echo top (km)", dict(cmap="cividis", vmin=0, vmax=16)),
    (density, "VIL density (g m$^{-3}$)", dict(cmap="plasma", vmin=0, vmax=3.5)),
]
fig, axes = plt.subplots(2, 2, figsize=(11, 10), layout="constrained")
for ax, (field, label, style) in zip(axes.flat, panels):
    pm = ax.pcolormesh(x[rays], y[rays], field.values[rays], shading="nearest", **style)
    fig.colorbar(pm, ax=ax, label=label, shrink=0.85)
    ax.set_aspect("equal")
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)
    ax.set_xlabel("east of KGWX (km)")
    ax.set_ylabel("north of KGWX (km)")
fig.suptitle("KGWX, 30 March 2022, 23:59 UTC")
plt.show()
```

The line of convection has a VIL of 10 to 25 kg m$^{-2}$ and echo tops
of 8 to 10 km. The broad, weak VIL elsewhere is clear-air and boundary-layer echo,
which is not removed here (see the echo-QC notebook). The VIL density is
largest in the cores, where the water is concentrated in a shallow column; the
upper end of the colour scale, 3.5 g m$^{-3}$, is the value that Amburn and
Wolf (1997) associate with large hail. Within about 10 km of the radar the
highest beam (19.5°) is only a few km high, so the echo top and the VIL there
are truncated (`VIL_TOP_TRUNCATED`).

### Below the lowest beam

The lowest beam rises from a few hundred metres above the radar at 20 km to
2.5 km at 150 km, and the VIL leaves out the layer below it, so it is a lower bound
(`VIL_LOWER_BOUND`). `fill_below=True` extends the lowest value down to the
radar height, as a pseudo-CAPPI does. The mean VIL of the rays that cross the
line is shown against ground range with and without that layer.

```{code-cell} ipython3
filled = vil(volume, fill_below=True)
line = (vil_polar.VIL > 3).any("range")  # rays that cross a cell
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
ground_km = vil_polar.ground_range / 1e3
for name, product in (("lower bound", vil_polar), ("lowest value filled to the ground", filled)):
    axes[0].plot(ground_km, product.VIL.where(line).mean("azimuth"), label=name)
axes[0].set(xlabel="ground range (km)", ylabel="mean VIL of the cells (kg m$^{-2}$)")
axes[0].legend(frameon=False)
axes[1].plot(ground_km, vil_polar.VIL_LOWEST_HEIGHT.mean("azimuth") / 1e3)
axes[1].set(xlabel="ground range (km)", ylabel="mean height of the lowest beam (km)")
plt.show()
```

## Polar volume and cone grid

The same functions work on a grid (any dataset with `z`, `y` and `x`, such as the
output of `grid_cones`). The comparison below uses a 1 km grid with 250 m
levels. The grid interpolates linearly in dBZ between the cones, the polar
columns integrate $Z^{4/7}$ with $Z$ linear between the beams, so the gridded
VIL is lower where the reflectivity falls quickly with height; the median
difference on this volume is about −16 %. Gridding $Z$ instead of dBZ reduces
it to about −3 %.

```{code-cell} ipython3
from scipy.interpolate import RegularGridInterpolator

from radarx.grid import grid_cones

gx = gy = np.arange(-150e3, 150e3 + 1, 1000.0)
gz = np.arange(250.0, 18e3, 250.0)
grid = grid_cones(volume, "DBZH", gx, gy, gz)
vil_grid = vil(grid)

at = RegularGridInterpolator((gy, gx), vil_grid.VIL.values, bounds_error=False)
north, east = s * np.cos(az), s * np.sin(az)
from_grid = at(np.stack([north.ravel(), east.ravel()], -1)).reshape(north.shape)
reference = vil_polar.VIL.values
use = (s > 20e3) & (s < 120e3) & (reference > 1.0) & np.isfinite(from_grid)
rel = (from_grid[use] - reference[use]) / reference[use]
print(f"{use.sum()} columns: median difference {100 * np.median(rel):.0f} %, "
      f"5-95 % range {100 * np.percentile(rel, 5):.0f} to {100 * np.percentile(rel, 95):.0f} %")

fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
ax.hexbin(reference[use], from_grid[use], gridsize=60, extent=(0, 30, 0, 30), mincnt=1, cmap="Blues", bins="log")
ax.plot([0, 30], [0, 30], "k-", lw=0.8)
ax.set(xlabel="VIL of the polar volume (kg m$^{-2}$)", ylabel="VIL of the cone grid (kg m$^{-2}$)", aspect="equal")
plt.show()
```

## Comparison with NEXRAD Level III and MRMS

Three independent VIL products of the same squall line. Two come from the radar
itself: the Level III digital VIL (`DVL`, product 134, polar 1° by 1 km bins,
256 levels) and the legacy VIL (`NVL`, product 57), a 4 km raster with 16
classes (0, 1, 5, 10, 15, … 70 kg m$^{-2}$). The third is the MRMS VIL (the
`VIL_00.50` product of the `noaa-mrms-pds` bucket on a 0.01° grid), computed
from the three-dimensional mosaic of all radars, so it is not the same data
and its time differs by up to 2 minutes. The volume of 23:59:59 UTC (which
scans until about 00:05) is compared with the Level III products of the same
volume and the MRMS field of 00:02:37 UTC. Level III files are read with MetPy
and the MRMS GRIB2 file with cfgrib. The cap of the operational products is not
documented in the files and cannot be read from them; `dbz_cap` and `min_dbz`
are varied below to see how much they matter.

```{code-cell} ipython3
import gzip
import shutil

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from metpy.io import Level3File
from pyproj import Transformer

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")


def fetch(bucket, key, out):
    s3.download_file(bucket, key, out)
    return out


dvl = Level3File(fetch("unidata-nexrad-level3", "GWX_DVL_2022_03_30_23_59_59", "data/dvl"))
nvl = Level3File(fetch("unidata-nexrad-level3", "GWX_NVL_2022_03_30_23_59_59", "data/nvl"))
print("DVL: product", dvl.prod_desc.prod_code, "| NVL: product", nvl.prod_desc.prod_code,
      "| volume", dvl.metadata["vol_time"], "| NVL grid", np.shape(nvl.sym_block[0][0]["data"]))
```

The comparison is made on the reference grids, in the range 20 to 150 km:
my VIL is averaged over each DVL bin (1° by 1 km), over each 4 km NVL cell and
over each MRMS pixel. `fill_below` is the VIL with the lowest value extended
to the ground.

```{code-cell} ipython3
def scores(mine, ref):
    ok = np.isfinite(mine) & np.isfinite(ref)
    d = (mine - ref)[ok]
    row = {"n": int(ok.sum()), "mean ref": ref[ok].mean(), "mean mine": mine[ok].mean(),
           "bias": d.mean(), "rms": np.sqrt((d**2).mean()), "corr": np.corrcoef(mine[ok], ref[ok])[0, 1]}
    hi = ok & (ref > 10)
    row["bias, ref > 10"] = (mine - ref)[hi].mean()
    return row


gr = vil_polar.ground_range.values[None, :] * np.ones((vil_polar.sizes["azimuth"], 1))
within = (gr > 20e3) & (gr < 150e3)
east, north = gr * np.sin(np.deg2rad(vil_polar.azimuth.values[:, None])), gr * np.cos(np.deg2rad(vil_polar.azimuth.values[:, None]))


def binned(values, ia, ib, shape, reduce="mean"):
    """Mean (or maximum) of the gates that fall in each reference bin."""
    ok = within & np.isfinite(values)
    if reduce == "max":
        out = np.full(shape, -np.inf)
        np.maximum.at(out, (ia[ok], ib[ok]), values[ok])
        return np.where(np.isfinite(out), out, np.nan)
    total, count = np.zeros(shape), np.zeros(shape)
    np.add.at(total, (ia[ok], ib[ok]), values[ok])
    np.add.at(count, (ia[ok], ib[ok]), 1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


variants = {
    "default": vil_polar.VIL.values,
    "fill_below": filled.VIL.values,
    "no cap": vil(volume, dbz_cap=None).VIL.values,
    "min_dbz 18": vil(volume, min_dbz=18.0).VIL.values,
}
```

### Level III digital VIL (DVL)

```{code-cell} ipython3
blk = dvl.sym_block[0][0]
raw = np.array([np.frombuffer(bytes(r), dtype=np.uint8) for r in blk["data"]])
decoded = dvl.map_data(raw)
dvl_vil = np.ma.filled(decoded.astype(float), np.nan)
centre = 0.5 * (np.array(blk["start_az"]) + np.array(blk["end_az"]))
offset = (centre[None, :] - vil_polar.azimuth.values[:, None] + 180) % 360 - 180
radial = np.argmin(np.abs(offset), axis=1)  # DVL radial of every ray of the volume
iaz = radial[:, None] * np.ones_like(gr, dtype=int)
ibin = np.floor(gr / 1e3).astype(int)
rows = {}
for name, values in variants.items():
    rows[f"{name}, bin mean"] = scores(binned(values, iaz, ibin, dvl_vil.shape), dvl_vil)
rows["default, bin maximum"] = scores(binned(variants["default"], iaz, ibin, dvl_vil.shape, "max"), dvl_vil)
rows["fill_below, bin maximum"] = scores(binned(variants["fill_below"], iaz, ibin, dvl_vil.shape, "max"), dvl_vil)
print(f"DVL maximum {np.nanmax(dvl_vil):.1f} kg m-2, radarx maximum {float(vil_polar.VIL.max()):.1f}")
pd.DataFrame(rows).T.round(2)
```

### Level III legacy VIL (NVL)

NVL gives classes, so the comparison is of the class of the cell mean VIL. The
product holds the level thresholds in its header.

```{code-cell} ipython3
levels = np.array([getattr(nvl.prod_desc, f"thr{i}") for i in range(1, 17)], float)
codes = np.array(nvl.sym_block[0][0]["data"])
n = codes.shape[0]
cell = 4e3
ix = np.floor((east + n * cell / 2) / cell).astype(int)
iy = np.floor((n * cell / 2 - north) / cell).astype(int)
inside = within & (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
total, count = np.zeros((n, n)), np.zeros((n, n))
ok = inside & np.isfinite(variants["default"])
np.add.at(total, (iy[ok], ix[ok]), variants["default"][ok])
np.add.at(count, (iy[ok], ix[ok]), 1)
cell_mean = np.where(count > 20, total / np.maximum(count, 1), np.nan)
mine_class = np.where(np.isfinite(cell_mean), np.digitize(cell_mean, levels[1:]), -1)
valid = mine_class >= 0
diff = mine_class[valid] - codes[valid]
print(f"NVL: {n} x {n} cells of {cell / 1e3:.0f} km, classes {levels[1:].astype(int).tolist()} kg m-2; "
      f"{valid.sum()} cells compared")
print(f"same class {100 * np.mean(diff == 0):.0f} %, within one class {100 * np.mean(np.abs(diff) <= 1):.0f} %, "
      f"radarx lower {100 * np.mean(diff < 0):.0f} %, higher {100 * np.mean(diff > 0):.0f} %")
```

### MRMS VIL

```{code-cell} ipython3
gz = fetch("noaa-mrms-pds", "CONUS/VIL_00.50/20220331/MRMS_VIL_00.50_20220331-000237.grib2.gz", "data/mrms.gz")
with gzip.open(gz) as src, open("data/mrms.grib2", "wb") as dst:
    shutil.copyfileobj(src, dst)
mrms = xr.open_dataarray("data/mrms.grib2", engine="cfgrib", backend_kwargs={"indexpath": ""})
lat0, lon0 = float(volume.latitude), float(volume.longitude)
mrms = mrms.sel(latitude=slice(lat0 + 1.6, lat0 - 1.6))
lon = np.where(mrms.longitude > 180, mrms.longitude - 360, mrms.longitude)
mrms = mrms.isel(longitude=np.flatnonzero((lon > lon0 - 1.9) & (lon < lon0 + 1.9)))
lon = np.where(mrms.longitude > 180, mrms.longitude - 360, mrms.longitude)
to_xy = Transformer.from_crs({"proj": "latlong", "datum": "WGS84"},
                             {"proj": "aeqd", "lat_0": lat0, "lon_0": lon0, "datum": "WGS84"}, always_xy=True)
mx, my = to_xy.transform(*np.meshgrid(lon, mrms.latitude.values))
mrms_vil = np.where(mrms.values < 0, np.nan, mrms.values.astype(float))  # -1 marks no data
print(f"MRMS: {mrms.sizes['latitude']} x {mrms.sizes['longitude']} pixels of 0.01°, "
      f"maximum {np.nanmax(mrms_vil):.1f} (units of the product: kg m-2, not stored in the file)")
px, py = np.floor((mx + 150e3) / 1e3).astype(int), np.floor((my + 150e3) / 1e3).astype(int)
cx, cy = np.floor((east + 150e3) / 1e3).astype(int), np.floor((north + 150e3) / 1e3).astype(int)
inb = (px >= 0) & (px < 301) & (py >= 0) & (py < 301)
rows = {}
for name, values in variants.items():
    ok = within & np.isfinite(values) & (cx >= 0) & (cx < 301) & (cy >= 0) & (cy < 301)
    total, count = np.zeros((301, 301)), np.zeros((301, 301))
    np.add.at(total, (cy[ok], cx[ok]), values[ok])
    np.add.at(count, (cy[ok], cx[ok]), 1)
    km = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    at_pixel = np.full(mrms_vil.shape, np.nan)
    at_pixel[inb] = km[py[inb], px[inb]]
    rows[name] = scores(at_pixel, mrms_vil)
    if name == "default":
        mine_at_mrms = at_pixel
pd.DataFrame(rows).T.round(2)
```

```{code-cell} ipython3
fig, axes = plt.subplots(2, 3, figsize=(16, 10), layout="constrained")
style = dict(cmap="viridis", vmin=0, vmax=30)
rays = np.argsort(vil_polar.azimuth.values)
pm = axes[0, 0].pcolormesh(east[rays] / 1e3, north[rays] / 1e3, np.where(within, variants["default"], np.nan)[rays], shading="nearest", **style)
axes[0, 0].set_title("radarx VIL")
ref_polar = np.full(gr.shape, np.nan)
ref_polar[within] = dvl_vil[iaz[within], np.minimum(ibin[within], dvl_vil.shape[1] - 1)]
axes[0, 1].pcolormesh(east[rays] / 1e3, north[rays] / 1e3, ref_polar[rays], shading="nearest", **style)
axes[0, 1].set_title("Level III digital VIL (DVL)")
axes[0, 2].pcolormesh(mx / 1e3, my / 1e3, mrms_vil, shading="nearest", **style)
axes[0, 2].set_title("MRMS VIL")
fig.colorbar(pm, ax=axes[0], label="VIL (kg m$^{-2}$)", shrink=0.8)
for ax in axes[0]:
    ax.set_aspect("equal")
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)
    ax.set_xlabel("east of KGWX (km)")
axes[0, 0].set_ylabel("north of KGWX (km)")
m_dvl = binned(variants["default"], iaz, ibin, dvl_vil.shape)
for ax, (x_, y_, label) in zip(axes[1], [
    (dvl_vil, m_dvl, "DVL (kg m$^{-2}$)"),
    (dvl_vil, binned(variants["fill_below"], iaz, ibin, dvl_vil.shape), "DVL (kg m$^{-2}$), radarx with fill_below"),
    (mrms_vil, mine_at_mrms, "MRMS (kg m$^{-2}$)"),
]):
    ok = np.isfinite(x_) & np.isfinite(y_)
    h = ax.hexbin(x_[ok], y_[ok], gridsize=60, extent=(0, 40, 0, 40), mincnt=1, cmap="Blues", bins="log")
    ax.plot([0, 40], [0, 40], "k-", lw=0.8)
    ax.set(xlabel=label.replace(", radarx with fill_below", ""), ylabel="radarx VIL (kg m$^{-2}$)", aspect="equal")
    if "fill_below" in label:
        ax.set_title("radarx with fill_below")
fig.colorbar(h, ax=axes[1], label="bins", shrink=0.8)
plt.show()
```

What the comparison shows. radarx and the DVL of the same volume have the same
structure (correlation 0.94) but radarx is lower, by about 1 kg m$^{-2}$ on
average and by 6 to 7 kg m$^{-2}$ (40 %) in the cores where the DVL exceeds
10 kg m$^{-2}$. The cap and the reflectivity threshold hardly matter (no cap
changes the mean by 0.02 kg m$^{-2}$). Three things explain part of it. The
layer below the lowest beam: with `fill_below` the bias falls from −1.1 to
−0.7 kg m$^{-2}$. The cell size: the DVL value of a 1° by 1 km bin is closer to the
maximum than to the mean of my 0.5° by 0.25 km gates (the bin maximum gives −0.7
and, with `fill_below`, −0.3 kg m$^{-2}$). What remains, about 20 % in the
cores, is not explained here: the algorithm of the radar product is not
published in the files, and the part of the storm above the highest beam, which
radarx leaves out (`VIL_TOP_TRUNCATED`) and an operational product may extend,
is one candidate. The NVL classes agree with radarx in 72 % of the cells and
within one class in 96 %; where they differ, radarx is the lower class
(27 % of the cells, against 1 % higher), the same direction as for the DVL, and
the classes are 5 kg m$^{-2}$ wide above 5. The MRMS VIL is smoother and lower in the cores: its maximum in this scene is
15.7 kg m$^{-2}$ against 29.5 for radarx, and where the MRMS VIL exceeds
10 kg m$^{-2}$ radarx is higher by 5 kg m$^{-2}$ on average (the mean over all
pixels is 0.2 kg m$^{-2}$ higher, since most pixels are weak echo). The
correlation is 0.64 and `fill_below` makes the difference larger. MRMS is a
mosaic of several radars on a 1 km grid at a different time, so a part of the
difference is the data and the time, and the cap and the integration limits of
its algorithm are not given in the file. Taken together, the two operational
products disagree with each other by more than either does with radarx in
the mean. The comparison shows that the structure of the VIL is robust and its
value in the cores uncertain by tens of percent; it cannot show which product is
right, because none of them is a measurement of the liquid water in the column.

### The concavity of $Z^{4/7}$ in a QVP

The VIL of a QVP is the VIL of the azimuthal-mean profile; for a mean in
linear $Z$ it is not lower than the mean of the VILs of the columns the profile
averages, since $Z^{4/7}$ is concave, and for a mean in dBZ it is lower. In the
same volume, on the grid of 250 m levels from 1.5 to 12 km inside 30 km of the
radar (the ring of a QVP of this volume), both are computed over identical
heights. The mean of the column VILs is 1.8 kg m$^{-2}$ (2764 columns); the VIL
of the profile of the mean $Z$ is 3.7, twice as large, and that of the mean
dBZ is 1.2, a third lower. The ring contains cores and echo-free air, so the
spread of $Z$ is large and so is the effect; in a uniform field the three agree.

```{code-cell} ipython3
ring = (np.hypot(*np.meshgrid(gx, gy)) < 30e3)
inner = grid.sel(z=slice(1500, 12000)).where(xr.DataArray(ring, dims=("y", "x"), coords={"y": gy, "x": gx}))
columns = vil(inner, missing="clear").VIL
mean_z = (10 ** (inner.DBZH / 10)).mean(["y", "x"])
profile = xr.Dataset({"DBZH": 10 * np.log10(mean_z)}).rename(z="height")
mean_db = xr.Dataset({"DBZH": inner.DBZH.mean(["y", "x"])}).rename(z="height")
print(f"mean of the column VILs {float(columns.mean()):.2f} kg m-2 ({int(columns.notnull().sum())} columns)")
print(f"VIL of the mean profile in linear Z: {float(vil(profile, missing='clear').VIL):.2f}")
print(f"VIL of the mean profile in dBZ:      {float(vil(mean_db, missing='clear').VIL):.2f}")
```

## Liquid water content of rain

`liquid_water_content` gives the water content of rain gate by gate. The
default `method="zm"` is the power law behind the VIL formula,
$\mathrm{LWC} = 3.44\times10^{-3}\,Z^{4/7}$ g m$^{-3}$, which assumes the
exponential drop size distribution of Marshall and Palmer. `method="dsd"`
retrieves the gamma drop size distribution of each gate from $Z_H$, $Z_{DR}$
and $K_{DP}$ (`radarx.retrieve.dsd`) and returns its water content. Radar
sees the liquid water of rain only: above the melting layer the reflectivity
is that of ice, where the power law overestimates the water content, and in the
melting layer it is enhanced. The gates below the melting layer are selected
with `mask`. Ice water content needs a different method that radarx does not
provide.

The example is a 0.5° C-band sweep of the ARM CSAPR2 radar in deep convection
(Argentina, CACTI); the file has a sounding temperature at every gate and an
echo classification (rain is class 1), and KDP is estimated from the raw
differential phase. The drop size distribution is retrieved where its
normalizing intercept $N_w$ is between $10^2$ and $10^5$ m$^{-3}$ mm$^{-1}$
(`nw_range`); outside that range the polarimetric variables are not
consistent with rain (hail, for example, gives an enormous number of tiny
drops).

```{code-cell} ipython3
file = DATASETS.fetch("corcsapr2cmacppiM1.c1.20181111.030003.nc")
sweep = xd.io.open_cfradial1_datatree(file).xradar.georeference()["sweep_0"].to_dataset()
kdp = estimate_kdp(
    sweep,
    phidp="uncorrected_differential_phase",
    rhohv="uncorrected_copol_correlation_coeff",
    dbzh="attenuation_corrected_reflectivity_h",
)
sweep = sweep.assign(KDP=kdp.KDP)
fields = dict(
    dbz="attenuation_corrected_reflectivity_h",
    mask=(sweep.gate_id == 1) & (sweep.sounding_temperature > 3.0),  # rain, warmer than +3 °C
)
lwc_zm = liquid_water_content(sweep, "zm", **fields)
lwc_dsd = liquid_water_content(
    sweep, "dsd",
    zdr="attenuation_corrected_differential_reflectivity", kdp="KDP", band="C",
    nw_range=(1e2, 1e5), **fields,
)
print(f"{int(lwc_zm.notnull().sum())} rain gates; median LWC {float(lwc_zm.median()):.2f} (power law) "
      f"and {float(lwc_dsd.median()):.2f} g m-3 (DSD)")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
for ax, (field, label) in zip(axes[:2], [(lwc_zm, "power law"), (lwc_dsd, "gamma DSD")]):
    pm = ax.pcolormesh(sweep.x / 1e3, sweep.y / 1e3, field, cmap="viridis", vmin=0, vmax=6, shading="nearest")
    fig.colorbar(pm, ax=ax, label=f"LWC, {label} (g m$^{{-3}}$)", shrink=0.85)
    ax.set_aspect("equal")
    ax.set_xlim(-20, 110)
    ax.set_ylim(-110, 75)
    ax.set_xlabel("east of CSAPR2 (km)")
axes[0].set_ylabel("north of CSAPR2 (km)")
both = lwc_zm.notnull() & lwc_dsd.notnull()
h = axes[2].hexbin(lwc_zm.values[both], lwc_dsd.values[both], gridsize=60, extent=(0, 6, 0, 6), mincnt=1, cmap="Blues", bins="log")
axes[2].plot([0, 6], [0, 6], "k-", lw=0.8)
axes[2].set(xlabel="LWC, power law (g m$^{-3}$)", ylabel="LWC, gamma DSD (g m$^{-3}$)", aspect="equal")
fig.colorbar(h, ax=axes[2], label="gates", shrink=0.85)
plt.show()
```

The two methods show the same structure. They differ where the drop size
distribution departs from the exponential one that the power law assumes: the
gamma retrieval gives larger water contents in the strongest cores, and
`lwc_dsd` is empty where the polarimetric variables are not consistent with rain.

### Power law against the drop size distribution

The same comparison for the KGWX 0.5° sweep (S band), where the rain gates are
those with $\rho_{hv} \ge 0.97$ and $Z_H \ge 10$ dBZ within 120 km, where the
beam is below 2 km, under the melting layer at about 3 km; KDP is estimated
from the differential phase. The table lists, by power-law water content, the
gates, the mean of each method and the mean and rms difference (DSD minus
power law).

```{code-cell} ipython3
kg = xd.io.open_nexradlevel2_datatree(path, sweep=[0], drop_variables=["WRADH", "CCORH", "CFP"])["sweep_0"].to_dataset()
for name, bad in (("DBZH", -32.0), ("ZDR", -12.9), ("RHOHV", 0.21)):
    kg[name] = kg[name].where(kg[name] > bad)
kg = kg.sel(range=slice(None, 120e3))
kg = kg.assign(KDP=estimate_kdp(kg).KDP)
rain_kg = (kg.RHOHV >= 0.97) & (kg.DBZH >= 10.0)
kg_zm = liquid_water_content(kg, "zm", mask=rain_kg)
kg_dsd = liquid_water_content(kg, "dsd", kdp="KDP", band="S", mask=rain_kg, nw_range=(1e2, 1e5))


def lwc_table(zm, dsd_):
    both = (zm.notnull() & dsd_.notnull()).values
    a, b = zm.values[both], dsd_.values[both]
    rows = {}
    for lo, hi in [(0, 0.1), (0.1, 0.5), (0.5, 1), (1, 3), (3, np.inf), (0, np.inf)]:
        m = (a >= lo) & (a < hi)
        rows["all" if hi == np.inf and lo == 0 else f"{lo:g} to {hi:g}"] = {
            "gates": int(m.sum()), "power law": a[m].mean(), "DSD": b[m].mean(),
            "bias": (b[m] - a[m]).mean(), "rms": np.sqrt(((b[m] - a[m]) ** 2).mean()),
        }
    rows["all"]["corr"] = np.corrcoef(a, b)[0, 1]
    return pd.DataFrame(rows).T.round(2)


print("KGWX 0.5°"); display(lwc_table(kg_zm, kg_dsd))
print("CSAPR2 0.5°"); display(lwc_table(lwc_zm, lwc_dsd))
```

In the KGWX sweep the two methods agree on average (0.44 against
0.39 g m$^{-3}$, correlation 0.79) and the rms difference is 0.5 g m$^{-3}$; in
the CSAPR2 sweep the gamma retrieval is larger by 0.26 g m$^{-3}$ on average
(0.69 against 0.43), by 1.25 g m$^{-3}$ where the power law gives 1 to
3 g m$^{-3}$, and the rms difference is 1.0 g m$^{-3}$. The bins are of the
power-law value, so the signs of the differences at the two ends of the table
partly come from the binning itself.

What this comparison shows and what it does not. It shows how far the two
retrievals differ in rain below the melting layer: each is a model of the drop
size distribution, and they differ where it departs from the exponential one,
which is the usual case. It does not show which one is right, because
neither is compared with a measurement of the water content, such as a
disdrometer or an aircraft probe. The agreement is also not independent
evidence for either: both start from the same $Z_H$, the DSD retrieval only
adds $Z_{DR}$ and $K_{DP}$, so the power law and the drop size distribution
are correlated by construction. The $Z_{DR}$ calibration, the attenuation
correction (C band) and the exclusion by `nw_range` set the DSD result.

## Profiles: VIL of a quasi-vertical profile

`vil` also takes a QVP time series (a `height` dimension, with `time`). The
VIL is that of the azimuthal-mean profile, starting at the lowest height of the
profile, so it is a lower bound. If a melting level is given, `VIL_LIQUID` is
the VIL of the rain below it: pass a height (a number or an array over time,
such as the 0 °C height of a sounding or ERA5) or the result of `melting_layer`,
whose bottom is used. The QVPs are those of the ARMOR C-band radar (Huntsville,
Alabama) on 11 April 2008, 12° elevation, as in the QVP notebook.

```{code-cell} ipython3
names = sorted(n for n in DATASETS.registry if n.startswith("RAW_NA_000_125_"))[::2]
volumes = [xd.io.open_iris_datatree(DATASETS.fetch(n)) for n in names]
tqvp = qvp_timeseries(volumes, ["DBZH", "ZDR", "RHOHV"], elevation=12.0)
ml = melting_layer(tqvp)
vil_qvp = vil(tqvp, melting=ml)
vil_qvp
```

```{code-cell} ipython3
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
pm = tqvp.DBZH.plot(ax=axes[0], x="time", y="height", cmap="ChaseSpectral", vmin=-10, vmax=60, add_colorbar=False)
fig.colorbar(pm, ax=axes[0], label="$Z_H$ (dBZ)", pad=0.01)
axes[0].fill_between(ml.time, ml.melting_layer_bottom, ml.melting_layer_top, color="k", alpha=0.3, label="melting layer")
axes[0].set(ylim=(0, 12e3), ylabel="height (m)", xlabel="", title="ARMOR QVP, 12° elevation, 11 April 2008")
axes[0].legend(loc="upper right")
axes[1].plot(vil_qvp.time, vil_qvp.VIL, label="VIL, whole profile")
axes[1].plot(vil_qvp.time, vil_qvp.VIL_LIQUID, label="VIL below the melting layer")
axes[1].set(ylabel="VIL (kg m$^{-2}$)", xlabel="time (UTC)")
axes[1].legend(frameon=False)
plt.show()
```

The VIL of the whole profile includes the ice above the melting layer, where the
formula for rain is not valid, and the bright band; the liquid VIL stops below
the layer.

## References

- Greene, D. R., and R. A. Clark, 1972: Vertically integrated liquid water: A new
  analysis tool. *Mon. Wea. Rev.*, **100**, 548-552,
  <https://doi.org/10.1175/1520-0493(1972)100<0548:VILWNA>2.3.CO;2>
- Seo, B.-C., W. F. Krajewski, and Y. Qi, 2020: Utility of vertically integrated
  liquid water content for radar-rainfall estimation: Quality control and
  precipitation type classification. *Atmos. Res.*, **236**, 104800,
  <https://doi.org/10.1016/j.atmosres.2019.104800>
- Amburn, S. A., and P. L. Wolf, 1997: VIL density as a hail indicator. *Wea.
  Forecasting*, **12**, 473-478,
  <https://doi.org/10.1175/1520-0434(1997)012<0473:VDAAHI>2.0.CO;2>
- Lakshmanan, V., K. Hondl, C. K. Potvin, and D. Preignitz, 2013: An improved
  method for estimating radar echo-top height. *Wea. Forecasting*, **28**,
  481-488, <https://doi.org/10.1175/WAF-D-12-00084.1>
- Marshall, J. S., and W. McK. Palmer, 1948: The distribution of raindrops with
  size. *J. Meteor.*, **5**, 165-166,
  <https://doi.org/10.1175/1520-0469(1948)005<0165:TDORWS>2.0.CO;2>
