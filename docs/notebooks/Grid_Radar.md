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
import numpy as np
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

## Georeferencing of the grid

The grid is georeferenced as CF: the azimuthal equidistant projection centred
on the radar is the scalar `crs_wkt`, every field has `grid_mapping`, and the
2-D `lat(y, x)` and `lon(y, x)` are computed from the projection (a latitude
that depends on `y` only is wrong away from the radar: by 0.6 km in the corner
of a 120 km wide grid and by 9 km in the corner of a 460 km wide one). cf_xarray and rioxarray find
the projection.

```{code-cell} ipython3
print(ds2["DBZH"].attrs["grid_mapping"], ds2["crs_wkt"].attrs["grid_mapping_name"])
print(ds2["lat"].dims, ds2["x"].attrs["standard_name"], ds2["z"].attrs["positive"])
```

## Radius of influence gridding

`grid_cones` has no smoothing length. `dtree.radarx.grid_gates()` is the other
family: every output point is the weighted mean of the gates within a radius of
influence in metres, with the weights of Cressman (1959),
$(R^2-d^2)/(R^2+d^2)$, or Barnes (1964), $\exp(-4d^2/R^2)$, or the nearest gate.
With `beamwidth` the radius grows with the distance from the radar like the
beam, so that the weights reach the gates of the neighbouring beams at any range;
`roi_range_factor` and `roi_height_factor` give other growth rates, the
parameters of the radius functions of Py-ART. Points without a gate in reach
stay empty. A gate is a point in space, so the same call grids a sector scan or
a volume with repeated tilts.

```{code-cell} ipython3
%%time
xg = yg = np.arange(-150e3, 150e3 + 1, 1000.0)
zg = np.arange(500.0, 10e3 + 1, 500.0)
cone_grid = rx.grid.grid_cones(dtree, "DBZH", xg, yg, zg)
cressman = dtree.radarx.grid_gates(["DBZH"], xg, yg, zg, weighting="cressman", roi=1000.0, beamwidth=1.0)
barnes = dtree.radarx.grid_gates(["DBZH"], xg, yg, zg, weighting="barnes", roi=1000.0, beamwidth=1.0)
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
for ax, grid, title in zip(axes, (cone_grid, cressman, barnes), ("cone", "Cressman", "Barnes")):
    dbz2km = grid["DBZH"].sel(z=2000.0)
    dbz2km.where(dbz2km >= 0).plot(ax=ax, x="x", y="y", cmap="ChaseSpectral", vmin=0, vmax=70)
    ax.set(title=f"{title}, z = 2 km", aspect="equal")
```

The cone interpolation leaves cells that no two sweeps enclose empty; the
radius of influence also fills cells up to the radius beyond the lowest and
highest sweep, so more of the grid has a value, at the price of the smoothing:

```{code-cell} ipython3
both = cone_grid["DBZH"].notnull() & cressman["DBZH"].notnull()
rms = float(np.sqrt(((cone_grid["DBZH"] - cressman["DBZH"]).where(both) ** 2).mean()))
print(
    f"cells with a value: cone {float(cone_grid['DBZH'].notnull().mean()):.2f}, "
    f"Cressman {float(cressman['DBZH'].notnull().mean()):.2f}; rms difference {rms:.1f} dB"
)
```

## Beam weighting

`weighting="beam"` weights a gate by the pulse volume instead of by its distance
to the point: a Gaussian that is as wide across the beam as the two-way power
pattern of a beam of half-power width `beamwidth` (Doviak and Zrnic 1993,
Eq. 5.75) and as wide along it as `range_resolution` (default: the gate
spacing). The beams that pass through the point dominate, so a radial velocity,
which is a projection on the beam, is mixed less with gates of other directions,
and sharp edges stay sharp. The radius of influence still decides which gates
take part. Radial velocity of the same volume at 2 km, gridded with the
Cressman weights and with the beam weights, and the difference to the cone grid,
which reproduces the gates (the [beam-weighted gridding notebook](Beam_Weighted_Gridding)
shows it on its own):

```{code-cell} ipython3
def mask_velocity(ds):
    return ds.where(ds.VRADH > -63.9) if "VRADH" in ds else ds

vel_tree = dtree.xradar.map_over_sweeps(mask_velocity)
vel = {
    "cone": rx.grid.grid_cones(vel_tree, "VRADH", xg, yg, zg),
    "Cressman": vel_tree.radarx.grid_gates(["VRADH"], xg, yg, zg, weighting="cressman", roi=1000.0, beamwidth=1.0),
    "beam": vel_tree.radarx.grid_gates(["VRADH"], xg, yg, zg, weighting="beam", roi=1000.0, beamwidth=1.0),
}
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
for ax, (title, grid) in zip(axes, vel.items()):
    v2km = grid["VRADH"].sel(z=2000.0)
    v2km.plot(ax=ax, x="x", y="y", cmap="RdBu_r", vmin=-30, vmax=30)
    ax.set(title=f"{title}, VRADH at 2 km", aspect="equal", xlim=(-80e3, 80e3), ylim=(-80e3, 80e3))
for title in ("Cressman", "beam"):
    both = vel["cone"]["VRADH"].notnull() & vel[title]["VRADH"].notnull()
    diff = (vel[title]["VRADH"] - vel["cone"]["VRADH"]).where(both)
    print(f"{title}: rms difference to the cone grid {float(np.sqrt((diff ** 2).mean())):.1f} m/s")
```

## Vertical section

`grid_rhi` grids onto a vertical plane through the radar at an azimuth, from the
gates of an RHI or from the sweeps of a volume:

```{code-cell} ipython3
section = dtree.radarx.grid_rhi(
    ["DBZH"],
    azimuth=250.0,
    distance=np.arange(2e3, 200e3, 1000.0),
    z=np.arange(250.0, 12e3, 250.0),
    roi=1000.0,
    beamwidth=1.0,
)
fig, ax = plt.subplots(figsize=(9, 3.5))
(section["DBZH"] ).plot(ax=ax, x="distance", y="z", cmap="ChaseSpectral", vmin=-10, vmax=70)
ax.set(title="azimuth 250°");
```

## Repeated tilts and split cuts

A volume can scan an elevation more than once: the split cuts of NEXRAD
(a surveillance and a Doppler cut at the same angle), and the extra low tilts of
SAILS and MESO-SAILS minutes apart. `grid_cones` keeps the cut that reaches
farthest (`repeats="farthest"`, the default). `repeats="mean"` averages the
repeats that have a value at a cell, `"latest"` takes the latest and `"nearest"`
the one closest to the analysis time `time`. With a storm motion `motion=(u, v)`
every repeat is first moved to the analysis time, so that the mean is of the
echo at one place and not of an echo that moved.
