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

# IMD Radar Data

The India Meteorological Department (IMD) publishes radar data as NetCDF4
files with an IRIS-inspired layout, one sweep per file. IMD data is read
natively by [xradar](https://docs.openradarscience.org/projects/xradar/en/latest/notebooks/IMD.html)
(releases after 0.12.0); radarx then grids, retrieves and plots it.

```{note}
The radarx IMD reader (`rx.io.read_sweep`, `rx.io.read_volume`,
`rx.io.to_cfradial2`, `rx.io.to_cfradial2_volumes`) is deprecated and will be
removed in a future release. Use the xradar functions shown below instead.
```

```{code-cell} ipython3
import holoviews as hv
import xarray as xr
import xradar as xd
from open_radar_data import DATASETS

import radarx  # noqa: F401  registers the .radarx accessors

hv.extension("bokeh")
```

## Download

Sample files from the Jaipur S-band radar, from the
[open-radar-data](https://github.com/openradar/open-radar-data) repository.
A full volume consists of ten files, one per sweep.

```{code-cell} ipython3
volume_files = [
    DATASETS.fetch(f"IMD/JPR220822135253-IMD-B.nc{s}")
    for s in ["", ".1", ".2", ".3", ".4", ".5", ".6", ".7", ".8", ".9"]
]
```

## A single sweep

The xarray `imd` backend reads one file into a CfRadial2 sweep. Moments are
renamed to their CfRadial2 names (`Z` to `DBZH`, `V` to `VRADH`, ...).

```{code-cell} ipython3
ds = xr.open_dataset(volume_files[0], engine="imd")
ds
```

Calling the radarx plot accessor shows the sweep in its native range-azimuth
layout:

```{code-cell} ipython3
ds.DBZH.radarx.plot()
```

And as a georeferenced PPI, one panel per moment:

```{code-cell} ipython3
ds.radarx.plot.ppi(["DBZH", "VRADH"])
```

## A volume

`open_imd_datatree` assembles the sweep files into one volume DataTree.

```{code-cell} ipython3
dtree = xd.io.open_imd_datatree(volume_files)
dtree
```

PPIs of the lowest sweeps, one panel per sweep:

```{code-cell} ipython3
dtree.radarx.plot.ppi("DBZH", sweeps=[0, 1, 2, 3])
```

A CAPPI at 3 km, retrieved from the volume and plotted in one step:

```{code-cell} ipython3
dtree.radarx.plot.cappi("DBZH", height=3000, x_res=2000, y_res=2000)
```

## Many volumes

A directory usually holds many volumes back to back. `group_imd_files` splits
a list of files into per-volume lists, and `open_imd_volumes` opens them all
into one DataTree with a `vcp_NN` node per volume.

```{code-cell} ipython3
xd.io.group_imd_files(volume_files)
```

For more reading options (`first_dim`, `min_angle`/`max_angle`, time
filtering, ...), see the
[xradar IMD notebook](https://docs.openradarscience.org/projects/xradar/en/latest/notebooks/IMD.html).
