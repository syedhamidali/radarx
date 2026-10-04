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

# Interactive Radar Plots with hvplot

radarx provides interactive radar plots built on [hvplot](https://hvplot.holoviz.org/) and [HoloViews](https://holoviews.org/), following the accessor roadmap discussed in [openradar/xradar#174](https://github.com/openradar/xradar/issues/174). They are available on any `xarray.DataArray`, `xarray.Dataset` and `xarray.DataTree` through the `.radarx.plot` accessor.

Install the optional plotting dependencies with `pip install radarx[plot]`.

```{code-cell}
import holoviews as hv
import xradar as xd
from open_radar_data import DATASETS

import radarx  # noqa: F401  registers the .radarx accessors

hv.extension("bokeh")
```

## Read a radar volume with xradar

```{code-cell}
filename = DATASETS.fetch("swx_20120520_0641.nc")
dtree = xd.io.open_cfradial1_datatree(filename, sweep=[0, 1, 2, 3, 5, 7])
var = "corrected_reflectivity_horizontal"
sweep = dtree["sweep_0"].to_dataset()
```

## DataArray plots

Calling the accessor directly shows a sweep in its native range-azimuth layout.

```{code-cell}
sweep[var].radarx.plot()
```

A georeferenced plan-position indicator (PPI). Missing `x`/`y` coordinates are computed on the fly; distances are in km.

```{code-cell}
sweep[var].radarx.plot.ppi()
```

The gate mesh shows the outline of every radar gate, which is handy to inspect beam geometry and resolution:

```{code-cell}
sweep[var].isel(azimuth=slice(0, 40), range=slice(0, 80)).radarx.plot.mesh()
```

Gate centroids as points coloured by value:

```{code-cell}
sweep[var].isel(range=slice(0, 150)).radarx.plot.centroids()
```

## Dataset plots

On a Dataset, plots are faceted over variables:

```{code-cell}
sweep.radarx.plot.ppi([var, "mean_doppler_velocity"])
```

## DataTree plots

On a DataTree, plots are faceted over sweeps. Select sweeps by index or name with `sweeps=`.

```{code-cell}
dtree.radarx.plot.ppi(var, sweeps=[0, 2, 4])
```

A CAPPI is retrieved with `create_cappi` and plotted in one step:

```{code-cell}
dtree.radarx.plot.cappi(var, height=2000, x_res=1000, y_res=1000)
```

## Gridded data

Grid the volume to a 3D Cartesian domain, then plot a CAPPI level (or omit `z` for a height slider) and a Max-CAPPI with side projections. With the bokeh backend the axes are linked, so zooming the plan view also zooms the projections.

```{code-cell}
grid = dtree.radarx.to_grid(
    data_vars=[var],
    x_lim=(-50e3, 50e3),
    y_lim=(-50e3, 50e3),
    z_lim=(0, 8e3),
    x_step=1000,
    y_step=1000,
    z_step=500,
)
grid.radarx.plot.cappi(var, z=2000)
```

```{code-cell}
grid.radarx.plot.max_cappi(var)
```

## Customising

All keyword arguments are forwarded to hvplot (`clim`, `cmap`, `frame_width`, `rasterize`, ...). By default the colour range uses the 2nd-98th percentiles to ignore unmasked outliers; pass `clim` to fix it, or `symmetric=True` for velocities. Large sweeps are rasterized with [datashader](https://datashader.org/) by default, which keeps plots responsive; pass `rasterize=False` for full vector output. Use `backend="matplotlib"` for static figures. Any other hvplot method is available too:

```{code-cell}
sweep["mean_doppler_velocity"].radarx.plot.ppi(
    cmap="balance", symmetric=True, frame_width=350
)
```

```{code-cell}
sweep[var].radarx.plot.hist(bins=60)
```
