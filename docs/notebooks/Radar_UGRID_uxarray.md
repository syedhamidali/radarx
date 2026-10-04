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
The faces describe the real gate footprint, which grows with range.

This notebook follows the discussion in
[openradar/xradar#212](https://github.com/openradar/xradar/issues/212) and
[UXARRAY/uxarray#976](https://github.com/UXARRAY/uxarray/issues/976).
Install uxarray with `pip install radarx[uxarray]`.

## Why an unstructured grid?

A radar does not measure at points. Each gate is a sample volume a fixed
length in range but a fixed *angle* wide, so its footprint grows with distance:
a 0.5° × 250 m gate covers about 4 600 m² at 2 km and 330 000 m² at 150 km.
How the sweep is stored decides whether that geometry is kept.

radarx can hold a sweep in three ways:

| | Native polar arrays<br>(xradar sweep) | Cartesian grid<br>(`to_grid`, CAPPI) | Unstructured grid<br>(`to_uxarray`) |
|---|---|---|---|
| Values | as measured | interpolated | as measured |
| Cell geometry | centres only (`x`, `y` from georeferencing) | regular boxes of a chosen size | exact gate polygons |
| Cell area | not stored, must be derived | the same for every cell | true area of every gate |
| Resolution | native, varies with range | one fixed cell size: coarser than the gates near the radar; far away, one wide gate is repeated over several cells | native, varies with range |
| Spatial selection | by index or manual masks | by coordinate ranges | built-in subsetting (box, circle, nearest) |
| Moving data to other grids | needs extra tools | needs extra tools | built-in remapping (nearest, inverse distance, bilinear) to regular or unstructured grids |
| Size | smallest | depends on the grid | largest (nodes and connectivity) |
| Best for | processing along rays: quality control, attenuation, Doppler | mosaics, CAPPIs, 3D volumes, comparison with gridded products | gate-accurate maps, area statistics, comparison with unstructured model meshes |

So an unstructured grid is worth it when the **footprint or the area of each
gate matters** and the original values should not be interpolated:

- **Exact maps:** gate polygons are drawn with their true shape and size,
  without the smoothing or blockiness of a Cartesian grid.
- **Area statistics:** rain area, area-weighted means and totals over a
  region count each square kilometre once, however many gates cover it.
- **Model comparison:** modern weather and climate models (MPAS, ICON, E3SM)
  run on unstructured meshes; uxarray can remap radar gates onto them, or both
  onto a common grid, without first interpolating the radar to Cartesian.
- **One data model:** radar data then uses the same grid conventions and tools
  as model output, which is what the xradar and uxarray issues aim for.

It is not needed for quick-look plots or for processing along rays; the native
polar sweep and `.radarx.plot` cover those with less memory.

```{code-cell} ipython3
import holoviews as hv
import hvplot.xarray  # noqa: F401
import numpy as np
import pyproj
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

Maps below are drawn in longitude and latitude. One degree of longitude is
shorter than one degree of latitude by a factor `cos(latitude)` (0.83 here), so
each map covers a region that is square in kilometres and uses a square
frame. That keeps one kilometre the same length in both directions.

```{code-cell} ipython3
KM_PER_DEG = 111.32
cos_lat = np.cos(np.deg2rad(radar_lat))


def square_box(half_width_km):
    """Longitude and latitude limits of a square region around the radar."""
    dlat = half_width_km / KM_PER_DEG
    dlon = dlat / cos_lat
    return (radar_lon - dlon, radar_lon + dlon), (radar_lat - dlat, radar_lat + dlat)


map_style = dict(cmap="ChaseSpectral", clim=(-10, 60), frame_width=350, frame_height=350)
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

The 10 km × 10 km around the radar shows the individual gate polygons:
0.5° wide in azimuth and 250 m long in range. The empty centre is the radar's
blind zone; the first gate starts at 2 km.

```{code-cell} ipython3
lon_lim, lat_lim = square_box(5)
near = uxds["DBZH"].subset.bounding_box(lon_bounds=lon_lim, lat_bounds=lat_lim)
near.plot.polygons(
    line_color="black", line_width=0.2, xlim=lon_lim, ylim=lat_lim,
    title="Gates within 5 km of the radar", **map_style,
)
```

The whole sweep, rasterized so it stays responsive:

```{code-cell} ipython3
lon_lim, lat_lim = square_box(150)
uxds["DBZH"].plot.polygons(
    rasterize=True, xlim=lon_lim, ylim=lat_lim, title="DBZH, 0.5° sweep", **map_style
)
```

## The three representations side by side

The same square region drawn three ways, using the same colour scale:

1. **Polar sweep:** the native xradar arrays. Only gate centres are stored;
   the plotting library guesses each gate's edges from its neighbours.
2. **Cartesian grid:** the sweep sampled onto fixed 0.5 km cells (nearest gate).
3. **uxarray:** the stored gate polygons.

```{code-cell} ipython3
aeqd = pyproj.Proj(proj="aeqd", lat_0=radar_lat, lon_0=radar_lon, datum="WGS84")
geo = sweep.xradar.georeference()
gate_lon, gate_lat = aeqd(geo["x"].values, geo["y"].values, inverse=True)
polar = sweep["DBZH"].assign_coords(
    lon=(("azimuth", "range"), gate_lon), lat=(("azimuth", "range"), gate_lat)
)


def compare(azimuth, range_km, half_width_km, cartesian_km=0.5):
    """Plot one square region three ways: polar, Cartesian and uxarray."""
    # square region centred on the given azimuth and range
    cx, cy = (range_km * 1e3 * f(np.deg2rad(azimuth)) for f in (np.sin, np.cos))
    clon, clat = aeqd(cx, cy, inverse=True)
    dlat = half_width_km / KM_PER_DEG
    dlon = dlat / cos_lat
    lon_lim, lat_lim = (clon - dlon, clon + dlon), (clat - dlat, clat + dlat)
    style = dict(
        cmap="ChaseSpectral", clim=(-10, 60), xlim=lon_lim, ylim=lat_lim,
        frame_width=250, frame_height=250, colorbar=False,
        xlabel="Longitude", ylabel="Latitude", xticks=3, yticks=3,
    )

    # 1. native polar sweep: gate centres, edges guessed when plotting.
    # Select rays by their angle relative to the region centre, so the
    # selection never wraps across north (which would join rays far apart).
    reach = np.hypot(half_width_km, half_width_km) * 1.5
    rel = (polar["azimuth"] - azimuth + 180) % 360 - 180
    half_angle = np.rad2deg(np.arctan2(reach, max(range_km - reach, 0.5)))
    local = (
        polar.assign_coords(rel_azimuth=rel)
        .where(np.abs(rel) <= half_angle, drop=True)
        .sortby("rel_azimuth")
        .sel(range=slice(max(range_km - reach, 0) * 1e3, (range_km + reach) * 1e3))
    )
    p_polar = local.hvplot.quadmesh(
        x="lon", y="lat", title="Polar sweep (gate centres)", **style
    )

    # 2. Cartesian grid with a fixed cell size
    step = cartesian_km / KM_PER_DEG
    lon = np.arange(lon_lim[0], lon_lim[1], step / cos_lat)
    lat = np.arange(lat_lim[0], lat_lim[1], step)
    grid = uxds["DBZH"].remap.to_rectilinear(lon=lon, lat=lat)
    p_grid = grid.hvplot.quadmesh(
        x="lon", y="lat", title=f"Cartesian grid ({cartesian_km} km)", **style
    )

    # 3. unstructured grid: exact gate polygons
    # select a slightly larger area so gates cut by the frame are still drawn
    faces = uxds["DBZH"].subset.bounding_box(
        lon_bounds=(lon_lim[0] - dlon, lon_lim[1] + dlon),
        lat_bounds=(lat_lim[0] - dlat, lat_lim[1] + dlat),
    )
    p_ux = faces.plot.polygons(
        line_color="black", line_width=0.1, title="uxarray (gate polygons)",
        **{**style, "colorbar": True, "frame_width": 250},
    )
    return (p_polar + p_grid + p_ux).opts(shared_axes=False).cols(3)
```

**Near the radar (10 km, 6 km × 6 km).** Gates are only about 90 m wide and
250 m long. The 0.5 km Cartesian cells each merge several gates, so the fine
structure of the storm is lost. The polar and uxarray panels keep it.

```{code-cell} ipython3
compare(azimuth=31, range_km=10, half_width_km=3)
```

**Far from the radar (140 km, 16 km × 16 km).** Gates are now about 1.2 km
wide but still 250 m long. The Cartesian grid repeats one gate's value over
neighbouring cells across the beam, yet is too coarse to show the 250 m detail
along the beam. The polar and uxarray panels show the elongated gates as they
were measured.

```{code-cell} ipython3
compare(azimuth=299, range_km=140, half_width_km=8)
```

The polar and uxarray panels look almost the same: for a regular sweep, edges
guessed from neighbouring centres are a good approximation. The difference is
that in the polar sweep those edges exist only in the picture, while the
uxarray grid stores them. That is what makes the gate areas, area-weighted
statistics and remapping below possible.

## Gate areas

uxarray computes the area of every face (on the unit sphere, so multiply by
the Earth radius squared). A gate's area is its length times its width, and
the width grows in proportion to range, so a 0.5° × 250 m gate covers about
4 600 m² at 2 km but 330 000 m² at 150 km.

What matters for statistics is how gates and area are spread over range.
Gates are evenly spaced in range, so the share of gates within a given
distance grows linearly, but the share of area grows with the square of the
distance:

```{code-cell} ipython3
earth_radius = 6371008.8  # m
area = uxds.uxgrid.face_areas.values * earth_radius**2

# cumulative share of gates and of area, ordered by range
gate_range_km = uxds["range"].values / 1e3
order = np.argsort(gate_range_km)
range_sorted = gate_range_km[order]
share_gates = np.arange(1, order.size + 1) / order.size
share_area = np.cumsum(area[order]) / area.sum()

half = np.searchsorted(share_gates, 0.5)
print(
    f"Half of all gates lie within {range_sorted[half]:.0f} km of the radar "
    f"but cover only {100 * share_area[half]:.0f} % of the area."
)

step = max(order.size // 2000, 1)  # thin out the curves for plotting
curve_opts = dict(frame_width=350, frame_height=350, ylim=(0, 1))
(
    hv.Curve(
        (range_sorted[::step], share_gates[::step]), "Range (km)", "Cumulative share",
        label="gates",
    )
    * hv.Curve((range_sorted[::step], share_area[::step]), label="area")
).opts(
    hv.opts.Curve(**curve_opts),
    hv.opts.Overlay(title="Share of gates and of area within a range", legend_position="top_left"),
)
```

So a plain average over gates is dominated by the area close to the radar.

## Area-weighted statistics

Every gate counts once in a plain average, so the many small gates close to
the radar dominate it, as shown above. Weighting by area gives each square
kilometre the same weight. Reflectivity is averaged in linear units (Z), not in dBZ.

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

Here the storm sits close to the radar, where gates are small and numerous, so
the plain average overweights it; the area-weighted value describes the
region as a whole.

## Remap to a regular latitude-longitude grid

uxarray can remap face data onto other grids, for example a regular
latitude-longitude grid for comparison with gridded products or model output.
The longitude spacing is widened by `1 / cos(latitude)` so the cells are about
2.2 km × 2.2 km. Cells beyond the radar's maximum range are masked.

```{code-cell} ipython3
dlat = 0.02
lon_lim, lat_lim = square_box(150)
lon = np.arange(*lon_lim, dlat / cos_lat)
lat = np.arange(*lat_lim, dlat)
regular = uxds["DBZH"].remap.to_rectilinear(lon=lon, lat=lat)

# mask cells farther from the radar than the sweep reaches
lon2, lat2 = np.meshgrid(regular["lon"], regular["lat"])
distance = KM_PER_DEG * np.hypot((lon2 - radar_lon) * cos_lat, lat2 - radar_lat)
regular = regular.where(distance <= sweep.range.values[-1] / 1e3)

regular.hvplot.quadmesh(
    x="lon", y="lat", xlim=lon_lim, ylim=lat_lim,
    title="DBZH on a ~2 km latitude-longitude grid", **map_style,
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
