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

# ZDR columns

An updraft carries raindrops above the freezing level, where they freeze slowly.
The drops are large and wet, so a column of differential reflectivity of 1 dB or
more stands above the 0 °C level, and its depth follows the strength of the
updraft ([Snyder et al. 2015](https://doi.org/10.1175/WAF-D-15-0068.1)).

- `zdr_column` gives, at every point of a grid, the depth of the column. It counts
  the levels from the freezing level up while $Z_{DR} \ge 1$ dB and the copolar
  correlation coefficient is at least 0.8.
- `zdr_column_metrics` reduces that map to the maximum depth, the area and the
  volume of the columns.

Both are also methods of the grid, `grid.radarx.zdr_column()` and
`depth.radarx.zdr_column_metrics()`. The first example is a synthetic storm
with a known answer. The second grids the KGWX volume of the squall line of 30
March 2022.

```{code-cell} ipython3
import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.retrieve import zdr_column, zdr_column_metrics

warnings.filterwarnings("ignore", category=RuntimeWarning)
```

## A synthetic updraft

The updraft has its column top 3 km above a freezing level at 4 km in the centre
and falling to the freezing level at 8 km from it. A patch of low $\rho_{hv}$ on
the flank stands for a hail spike.

```{code-cell} ipython3
z = np.arange(0.0, 12001.0, 250.0)
y = x = np.arange(41) * 1000.0
r = np.hypot(y[:, None] - 20e3, x[None, :] - 20e3)
column_top = 4000.0 + 3000.0 * np.clip(1 - r / 8e3, 0, None)
zdr = xr.DataArray(
    np.where(z[:, None, None] < column_top[None], 2.0, 0.2),
    dims=("z", "y", "x"),
    coords={"z": z, "y": y, "x": x},
    name="ZDR",
)
rhohv = xr.full_like(zdr, 0.98)
rhohv[:, 20:24, 28:32] = 0.6  # the hail spike

depth = zdr_column(zdr, rhohv=rhohv, freezing_level=4000.0)
metrics = zdr_column_metrics(depth, min_depth=1000.0)
print(
    f"deepest column {float(metrics.ZDR_COLUMN_MAX_DEPTH):.0f} m, "
    f"area {float(metrics.ZDR_COLUMN_AREA):.0f} km2, "
    f"volume {float(metrics.ZDR_COLUMN_VOLUME):.0f} km3"
)
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
pm = axes[0].pcolormesh(x / 1e3, y / 1e3, depth, cmap="viridis", vmin=0, vmax=3000, shading="auto")
fig.colorbar(pm, ax=axes[0], label="ZDR column depth (m)")
axes[0].plot([20, 20], [0, 40], "w--", lw=0.8)
axes[0].set(xlabel="x (km)", ylabel="y (km)", aspect="equal", title="Depth")
pm = axes[1].pcolormesh(y / 1e3, z / 1e3, zdr.isel(x=20), cmap="turbo", vmin=-0.5, vmax=3, shading="auto")
fig.colorbar(pm, ax=axes[1], label="$Z_{DR}$ (dB)")
axes[1].axhline(4.0, color="w", lw=1)
axes[1].set(xlabel="y (km)", ylabel="height (km)", title="Section along the dashed line")
plt.show()
```

The map shows the depth growing from zero at 8 km from the centre to 3 km in the
middle. The hail spike removes the points of low $\rho_{hv}$ from the column, the
paper's way of keeping the three-body scatter signature out of the product. The
section shows the same column from the side, from the freezing level (white line)
up to the column top.

## The squall line of 30 March 2022 at KGWX

The lowest cuts of a WSR-88D volume of the KGWX radar (Columbus, Mississippi) from
the `open-radar-data` package, gridded at 1 km by 250 m with `grid_cones`. The
no-data codes are masked. The freezing level is not measured here: the 4000 m
used is an assumption for a warm sector in spring, in the vertical reference of the grid
(above sea level). A sounding or a model gives the real value; the next cell shows
how much it matters. If the file cannot be fetched, the notebook ends with the
synthetic result only.

```{code-cell} ipython3
import xradar as xd

from radarx.grid import grid_cones
from open_radar_data import DATASETS

try:
    path = DATASETS.fetch("KGWX20220330_234639_V06")
except Exception as err:  # offline: keep the synthetic result
    path = None
    print("open-radar-data file not reachable:", type(err).__name__)

if path is not None:
    volume = xd.io.open_nexradlevel2_datatree(
        path, drop_variables=["WRADH", "CCORH", "CFP", "PHIDP"]
    )

    def prepare(ds):
        """Mask the no-data codes and keep the first 100 km."""
        if "range" not in ds.dims:
            return ds
        ds = ds.sel(range=slice(None, 100e3))
        for name, limit in (("DBZH", -32.0), ("ZDR", -12.9), ("RHOHV", 0.21)):
            if name in ds:
                ds[name] = ds[name].where(ds[name] > limit).astype("float32")
        return ds

    volume = volume.map_over_datasets(prepare)
    gx = gy = np.arange(-100e3, 100e3 + 1, 1000.0)
    gz = np.arange(250.0, 12001.0, 250.0)
    grid = grid_cones(volume, ["DBZH", "ZDR", "RHOHV"], gx, gy, gz)
    rows = []
    for level in (3500.0, 4000.0, 4500.0):
        d = zdr_column(
            grid.ZDR, rhohv=grid.RHOHV, dbz=grid.DBZH, min_dbz=20.0, freezing_level=level
        )
        m = zdr_column_metrics(d, min_depth=1000.0)
        rows.append((level, float(m.ZDR_COLUMN_MAX_DEPTH), float(m.ZDR_COLUMN_AREA), float(m.ZDR_COLUMN_VOLUME)))
    print("freezing level (m)   deepest (m)   area (km2)   volume (km3)")
    for row in rows:
        print(f"{row[0]:18.0f} {row[1]:13.0f} {row[2]:12.0f} {row[3]:14.0f}")
```

`min_dbz=20` is an addition of radarx: without it the noise of weak echo makes
shallow columns all over the grid, which the paper does not have to deal with
(its data are of a calibrated, quality-controlled network).

```{code-cell} ipython3
if path is not None:
    depth_real = zdr_column(
        grid.ZDR, rhohv=grid.RHOHV, dbz=grid.DBZH, min_dbz=20.0, freezing_level=4000.0
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    pm = axes[0].pcolormesh(gx / 1e3, gy / 1e3, grid.DBZH.sel(z=2000.0, method="nearest"), cmap="Greys", vmin=0, vmax=65, shading="auto")
    fig.colorbar(pm, ax=axes[0], label="$Z_H$ at 2 km (dBZ)")
    pm = axes[1].pcolormesh(gx / 1e3, gy / 1e3, depth_real.where(depth_real >= 1000.0), cmap="viridis", vmin=1000, vmax=4000, shading="auto")
    fig.colorbar(pm, ax=axes[1], label="ZDR column depth of 1 km or more (m)")
    for ax, title in zip(axes, ("Reflectivity", "ZDR columns")):
        ax.set(xlabel="east of KGWX (km)", ylabel="north of KGWX (km)", aspect="equal", title=title)
    plt.show()
```

The columns of 1 km or more are mostly on the line of convection to the west of the
radar, in the south-western cells where the reflectivity is strongest, with a few
single points elsewhere that are noise. The freezing level matters a lot: from
3500 to 4500 m the area of the columns falls from 1031 to 157 km$^2$, so the
absolute depths of a case need a freezing level from a sounding or a model. The
$Z_{DR}$ must also be calibrated, because a bias of 0.5 dB moves the 1 dB
threshold by half its value, and a column is only as deep as the vertical
resolution of the gridded volume, which is poor at long range where the beams are
wide.

A vertical section through one of these columns, together with the KDP column of the same
updraft, is in [ZDR and KDP columns in a vertical section](ZDR_KDP_Column_Sections).

## References

- Snyder, J. C., A. V. Ryzhkov, M. R. Kumjian, A. P. Khain, and J. Picca, 2015: A ZDR Column Detection Algorithm to Examine Convective Storm Updrafts. *Weather and Forecasting*, **30**, 1819-1844, <https://doi.org/10.1175/WAF-D-15-0068.1>
