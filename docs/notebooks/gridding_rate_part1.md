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

# Py-ART

```{code-cell} ipython3
import os
import pyart
import radarx as rx  # noqa
import cmweather  # noqa
import cartopy.crs as ccrs
from datetime import datetime
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
file = "s3://unidata-nexrad-level2/2018/06/12/KSGF/KSGF20180612_083109_V06"
radar = pyart.io.read_nexrad_archive(file)
filename = os.path.basename(file) + ".nc"
if not os.path.exists(filename):
    pyart.io.write_cfradial(filename, radar)
else:
    print("Already present")
```

```{code-cell} ipython3
tstart = datetime.now()
# Grid using 11 vertical levels, and 101 horizontal grid cells at a resolution on 1 km
grid = pyart.map.grid_from_radars(
    (radar,),
    grid_shape=(31, 276, 276),
    grid_limits=(
        z_lims,
        y_lims,
        x_lims,
    ),
    fields=["reflectivity"],
)

xg = grid.to_xarray()
print("Py-ART gridding took:", datetime.now() - tstart)
display(xg)
del radar
```

## Plot

+++

## Py-ART

```{code-cell} ipython3
xg.squeeze().radarx.plot_max_cappi(
    data_var="reflectivity",
    vmin=-10,
    vmax=70,
    range_rings=True,
    add_map=True,
    projection=ccrs.PlateCarree(),
)
```

```{code-cell} ipython3

```
