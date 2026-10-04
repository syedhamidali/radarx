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

# Radar Sweeps as Unstructured Grids with uxarray

[uxarray](https://uxarray.readthedocs.io/) works with unstructured grids in the
[UGRID conventions](https://ugrid-conventions.github.io/ugrid-conventions/): a
grid is a set of **nodes** (corner points), **faces** (cells, defined by the
nodes around them) and **edges**, and data lives on the faces.

A radar sweep fits this model naturally. Every **gate becomes one face**, a
quadrilateral whose corners lie halfway to the neighbouring rays and gates.
The faces then describe the real gate footprint, which grows with range. That
gives:

- gate polygons drawn with their true shape on a map,
- the true area of every gate, for area-weighted statistics,
- uxarray's spatial tools (subsetting, remapping) on the native polar geometry.

This notebook follows the discussion in
[openradar/xradar#212](https://github.com/openradar/xradar/issues/212) and
[UXARRAY/uxarray#976](https://github.com/UXARRAY/uxarray/issues/976).
Install uxarray with `pip install uxarray`.

```{code-cell} ipython3
import holoviews as hv
import numpy as np
import xradar as xd
from open_radar_data import DATASETS

import radarx  # noqa: F401  registers the .radarx accessors

hv.extension("bokeh")
```

## Read a sweep

The lowest sweep of a NEXRAD volume from Lubbock, Texas, limited to 150 km.
`inherit="all_coords"` keeps the radar site location from the root of the
DataTree, which the conversion needs.

```{code-cell} ipython3
filename = DATASETS.fetch("KLBB20160601_150025_V06")
dtree = xd.io.open_nexradlevel2_datatree(filename, sweep=[0])
sweep = dtree["sweep_0"].to_dataset(inherit="all_coords").sel(range=slice(0, 150e3))
radar_lon, radar_lat = float(sweep.longitude), float(sweep.latitude)
sweep
```

## Convert to uxarray

`to_uxarray` builds the UGRID grid from the gate corners and attaches the
chosen variables to its faces.

```{code-cell} ipython3
uxds = sweep.radarx.to_uxarray(["DBZH"])
uxds
```

```{code-cell} ipython3
uxds.uxgrid
```

## Gate footprints

Zooming in near the radar shows the individual gate polygons: 0.5° wide in
azimuth and 250 m long in range.

```{code-cell} ipython3
near = uxds["DBZH"].subset.bounding_box(
    lon_bounds=(radar_lon - 0.05, radar_lon + 0.05),
    lat_bounds=(radar_lat - 0.05, radar_lat + 0.05),
)
near.plot.polygons(
    cmap="ChaseSpectral", clim=(-10, 60), line_color="black", line_width=0.2,
    title="Gates near the radar",
)
```

The whole sweep, rasterized so it stays responsive:

```{code-cell} ipython3
uxds["DBZH"].plot.polygons(
    rasterize=True, cmap="ChaseSpectral", clim=(-10, 60), title="DBZH, 0.5° sweep",
)
```

## Gate areas

uxarray computes the area of every face (on the unit sphere, so multiply by
the Earth radius squared). A gate's area grows linearly with range, from a few
thousand square metres near the radar to over 300 000 m² at 150 km:

```{code-cell} ipython3
earth_radius = 6371008.8  # m
area = uxds.uxgrid.face_areas.values * earth_radius**2

n_range = sweep.sizes["range"]
mean_area = area.reshape(-1, n_range).mean(axis=0)
hv.Curve(
    (sweep.range.values / 1e3, mean_area / 1e6), "Range (km)", "Gate area (km²)"
).opts(title="Gate area vs range", width=500)
```

## Area-weighted statistics

Every gate counts once in a plain average, so the many small gates close to
the radar dominate it. Weighting by area gives each square kilometre the same
weight. Reflectivity is averaged in linear units (Z), not in dBZ.

```{code-cell} ipython3
dbz = uxds["DBZH"]
echo = dbz.where(dbz >= 10)  # precipitation echo only
z = 10 ** (echo / 10)  # linear reflectivity

plain = 10 * np.log10(float(z.mean()))
weighted = 10 * np.log10(float(z.weighted_mean()))
heavy_area = area[(dbz >= 40).values].sum() / 1e6

print(f"Mean reflectivity, plain average     : {plain:.1f} dBZ")
print(f"Mean reflectivity, area-weighted     : {weighted:.1f} dBZ")
print(f"Area with reflectivity >= 40 dBZ     : {heavy_area:.0f} km²")
```

## Remap to a regular latitude-longitude grid

uxarray can remap face data onto other grids, for example a regular
latitude-longitude grid for comparison with gridded products or model output.
Cells beyond the radar's maximum range are masked.

```{code-cell} ipython3
lon = np.arange(radar_lon - 1.7, radar_lon + 1.7, 0.02)
lat = np.arange(radar_lat - 1.4, radar_lat + 1.4, 0.02)
regular = uxds["DBZH"].remap.to_rectilinear(lon=lon, lat=lat)

# mask cells farther from the radar than the sweep reaches
lon2, lat2 = np.meshgrid(regular["lon"], regular["lat"])
distance = 6371.0 * np.hypot(
    np.deg2rad(lon2 - radar_lon) * np.cos(np.deg2rad(radar_lat)),
    np.deg2rad(lat2 - radar_lat),
)
regular = regular.where(distance <= sweep.range.values[-1] / 1e3)

regular.hvplot.quadmesh(
    x="lon", y="lat", cmap="ChaseSpectral", clim=(-10, 60),
    title="DBZH on a 0.02° latitude-longitude grid", frame_width=450,
)
```

## Notes

- Faces are ordered by azimuth, then range; each face keeps its `azimuth` and
  `range` as coordinates.
- uxarray's `Grid.validate()` flags faces smaller than a fixed tolerance on the
  unit sphere (about 1e-10, roughly 4000 m²). That suits global model meshes;
  radar gates close to the radar are legitimately smaller.
- A full sweep has one face per gate (here about 430 000), so subsetting and
  remapping take a few seconds the first time, while uxarray builds its
  spatial index.
