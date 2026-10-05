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

# Grid Radar -> Plot Max-CAPPI

+++

`dtree.radarx.to_grid()` grids the volume with cone gridding (the default `method="cone"`): it interpolates within each sweep and then between sweeps, so it needs no smoothing parameters. Barnes interpolation (`method="barnes"`) is also available if the optional [fast-barnes-py](https://github.com/MeteoSwiss/fast-barnes-py.git) package is installed (Python < 3.13 only); please cite it if you use it in your research.

```{code-cell} ipython3
import fsspec
import radarx as rx
import xradar as xd
import cmweather  # noqa
import matplotlib.pyplot as plt
from radarx.utils import combine_nexrad_sweeps
```

```{code-cell} ipython3
print(rx.__version__)
```

```{code-cell} ipython3
file = "s3://unidata-nexrad-level2/2022/03/30/KGWX/KGWX20220330_234639_V06"
local_file = fsspec.open_local(
    f"simplecache::s3://{file}",
    s3={"anon": True},
    filecache={"cache_storage": "."},
)
dtree = xd.io.open_nexradlevel2_datatree(local_file)
dtree = combine_nexrad_sweeps(dtree)
dtree = dtree.xradar.georeference()
```

```{code-cell} ipython3
display(dtree.groups)
```

```{code-cell} ipython3
def filter_radar(ds):
    ds = ds.where((ds.DBZH > -10) & (ds.DBZH < 75))
    return ds
```

```{code-cell} ipython3
dtree = dtree.xradar.map_over_sweeps(filter_radar)
```

```{code-cell} ipython3
# Create a figure and axis
fig, ax = plt.subplots(figsize=(7, 5))
dtree["sweep_2"]["DBZH"].plot.contourf(
    x="x",
    y="y",
    levels=range(-10, 75),
    cmap="ChaseSpectral",
    ylim=(-200e3, 300e3),  # Adjust y-axis limits
    xlim=(-200e3, 300e3),  # Adjust x-axis limits
    ax=ax,  # Use the created axis
)
# Set the title
ax.set_title(
    f"{dtree.attrs['instrument_name']} {dtree['sweep_0']['time'].min().values}"
)
# Show the plot
plt.show()
```

```{code-cell} ipython3
%%time
ds = dtree.radarx.to_grid(
    data_vars=["DBZH"],
    pseudo_cappi=True,
    x_lim=(-100000.0, 100000.0),
    y_lim=(-100000.0, 100000.0),
    z_lim=(0, 10000.0),
    x_step=1000,
    y_step=1000,
    z_step=250,
)

ds.radarx.plot_max_cappi("DBZH", cmap="ChaseSpectral", add_slogan=True);
```

```{code-cell} ipython3
%%time
ds2 = dtree.radarx.to_grid(
    data_vars=["DBZH"],
    pseudo_cappi=False,
    x_lim=(-200000.0, 200000.0),
    y_lim=(-200000.0, 200000.0),
    z_lim=(0, 15000.0),
    x_step=1000,
    y_step=1000,
    z_step=250,
)

ds2.radarx.plot_max_cappi("DBZH", cmap="ChaseSpectral", add_slogan=True);
```

```{code-cell} ipython3

```
