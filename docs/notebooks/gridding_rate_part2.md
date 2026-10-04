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

# Radarx

```{code-cell} ipython3
import radarx as rx
import xradar as xd
import cmweather  # noqa
import cartopy.crs as ccrs
from datetime import datetime
```

```{code-cell} ipython3
def filter_radar(ds):
    ds = ds.where((ds.reflectivity >= 0) & (ds.reflectivity <= 70))
    return ds
```

## Grid Setup

```{code-cell} ipython3
x_lims = (-300e3, 250e3)
y_lims = (-250e3, 300e3)
z_lims = (0, 15e3)
h_res = 2000
v_res = 500
```

## Load data

```{code-cell} ipython3
filename = "KSGF20180612_083109_V06.nc"
```

```{code-cell} ipython3
dtree = xd.io.open_cfradial1_datatree(filename)
```

```{code-cell} ipython3
dtree = rx.utils.combine_nexrad_sweeps(dtree)
```

```{code-cell} ipython3
dtree = dtree.xradar.map_over_sweeps(filter_radar)
```

```{code-cell} ipython3
dtree = dtree.xradar.georeference()
dtree.groups
```

```{code-cell} ipython3
dtree["sweep_0"]["reflectivity"].plot(
    x="x",
    y="y",
    xlim=(-300e3, 200e3),
    ylim=(-200e3, 300e3),
    cmap="ChaseSpectral",
    levels=range(-10, 70),
)
```

## Radarx

```{code-cell} ipython3
tstart = datetime.now()
ds_rx = dtree.radarx.to_grid(
    data_vars=["reflectivity"],
    pseudo_cappi=False,
    x_lim=x_lims,
    y_lim=y_lims,
    z_lim=z_lims,
    x_step=h_res,
    y_step=h_res,
    z_step=v_res,
    x_smth=0.2,
    y_smth=0.2,
    z_smth=0.6,
)
print("Radarx gridding took:", datetime.now() - tstart)
display(ds_rx)
```

## Plot

```{code-cell} ipython3
ds_rx.radarx.plot_max_cappi(
    data_var="reflectivity",
    vmin=-10,
    vmax=70,
    range_rings=True,
    add_map=True,
    projection=ccrs.PlateCarree(),
    title="Radarx",
)
```

```{code-cell} ipython3

```
