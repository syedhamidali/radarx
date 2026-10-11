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

# Beam blockage and terrain clutter from a digital elevation model

In mountains the lowest beams of a radar run into the ground. The terrain takes
part of the beam power, so the reflectivity behind it is too low, and a beam
centre inside or at the surface of the terrain returns ground clutter.
`radarx.retrieve.beam_blockage` finds both for every gate of a sweep or a
volume from a digital elevation model (DEM):

- `PBB`, the partial beam blockage: the part of the beam power below the terrain
  at the gate ([Bech et al. 2003](https://doi.org/10.1175/1520-0426(2003)020<0845:TSOSPW>2.0.CO;2),
  Gaussian pattern as in
  [Lang et al. 2009](https://doi.org/10.1175/2008JTECHA1133.1));
- `CBB`, the cumulative beam blockage: the largest `PBB` along the ray up to the
  gate, because terrain shades everything behind it in the same direction;
- `TERRAIN_CLUTTER`, a flag for the gates whose beam centre is at most
  `clutter_margin` metres above the terrain surface;
- `TERRAIN_HEIGHT`, the terrain height below the gate.

The beam follows the effective Earth radius model (`k_e`, default 4/3, with the
site altitude in the Earth radius, as in `radarx.fundamentals.geometry` and
`radarx.grid`). The terrain is read from the DEM with bilinear interpolation
along the ray. A compiled multithreaded kernel (over rays) does the work, with
a NumPy fallback that gives the same numbers; the function accepts a `DataArray`,
a `Dataset` (a sweep) or a `DataTree` (a volume) and never changes its input.
The result is not applied to any other field: use `CBB` as a quality field or as
a mask.

With the terrain height $t$ and the beam centre height $h$ at the gate, $y = t - h$
and the half-power beam radius $a = r\beta/2$ at the range $r$ (beam width
$\beta$), the partial blockage of a uniform circular beam is, after Bech et al.
(2003, Eq. A3),

$$
\mathrm{PBB} = \frac{1}{\pi}\left[\frac{y}{a}\sqrt{1-\frac{y^2}{a^2}} + \arcsin\frac{y}{a} + \frac{\pi}{2}\right],
\qquad -a < y < a,
$$

zero below $-a$ and one above $a$. For the Gaussian beam (the default,
`pattern="gaussian"`) it is $\tfrac12\,\mathrm{erfc}\!\left(-y/(\sqrt2\,\sigma)\right)$
with $\sigma = a/\sqrt{2\ln 2}$. The paper does not write that formula down, so
it is derived here from its description of the Gaussian beam and is not checked
against the paper.

```{code-cell} ipython3
import time
import warnings

import cmweather  # noqa: F401  radar colormaps
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.integrate import quad
from xradar.georeference import antenna_to_cartesian

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.retrieve import beam_blockage

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="The input coordinates to pcolormesh")
plt.rcParams["figure.dpi"] = 110
```

## A digital elevation model

radarx does not download terrain. A DEM is an `xarray.DataArray` of heights in
metres above sea level with `latitude` and `longitude` coordinates in degrees
(or `y` and `x` in a projected system, with the `crs` argument). Tiles of SRTM
or of the Copernicus DEM (GLO-30) can be opened with
[rioxarray](https://corteva.github.io/rioxarray/), which radarx does not need:

```python
import rioxarray

dem = rioxarray.open_rasterio("copernicus_dem_tile.tif").squeeze("band", drop=True)
dem = dem.fillna(0)          # sea cells
out = sweep.radarx.beam_blockage(dem)   # or beam_blockage(sweep, dem)
```

Only the part of the DEM that the radar can reach is read, so a large or
dask-backed DEM is fine. Here, so that the page runs without a download, the
terrain is a synthetic mountain field: random terrain with the power-law
spectrum of real relief (spectral synthesis, amplitude falling as
$k^{-1.15}$, i.e. a spectrum slope of 2.3), raised into one range of high
mountains to the north-east of the radar, which sits on a hill.

```{code-cell} ipython3
LAT, LON = 30.0, 78.0  # a made-up site

def synthetic_dem(half_lat=1.5, half_lon=1.7, res=0.004, seed=11):
    lat = np.arange(LAT - half_lat, LAT + half_lat + res / 2, res)
    lon = np.arange(LON - half_lon, LON + half_lon + res / 2, res)
    ny, nx = lat.size, lon.size
    rnd = np.random.default_rng(seed)
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    k = np.hypot(ky, kx)
    k[0, 0] = 1.0
    spectrum = np.fft.fft2(rnd.standard_normal((ny, nx))) * k ** -1.15
    spectrum[0, 0] = 0.0
    relief = np.fft.ifft2(spectrum).real
    relief = (relief - relief.min()) / (relief.max() - relief.min())
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    # distance in km from the radar, and a broad range to the north-east
    dy, dx = (la - LAT) * 111.2, (lo - LON) * 111.2 * np.cos(np.deg2rad(LAT))
    along = (dx + dy) / np.sqrt(2.0)
    across = (dy - dx) / np.sqrt(2.0)
    range_mask = np.exp(-(((along - 38.0) / 24.0) ** 2) - (across / 55.0) ** 2)
    hill = np.exp(-((dx**2 + dy**2) / 9.0**2))  # the radar stands on a hill
    height = 120.0 + 450.0 * relief**1.6 + 1500.0 * range_mask * relief**0.8 + 250.0 * hill
    return xr.DataArray(
        height, dims=("latitude", "longitude"), coords={"latitude": lat, "longitude": lon},
        name="terrain_height", attrs={"units": "m", "long_name": "terrain height above sea level"},
    )

dem = synthetic_dem()
site_ground = float(dem.interp(latitude=LAT, longitude=LON))
ALTITUDE = site_ground + 20.0  # the antenna on a 20 m tower
print(f"DEM {dem.shape[0]} by {dem.shape[1]} cells, {float(dem.min()):.0f} to {float(dem.max()):.0f} m; "
      f"terrain at the radar {site_ground:.0f} m, antenna at {ALTITUDE:.0f} m")
```

## A volume with three low sweeps

The volume has the structure of an xradar volume: the radar site in the root,
the beam width in `radar_parameters` (which `beam_blockage` reads; pass
`beamwidth=` where the data do not have it) and one sweep per elevation with
`azimuth`, `range` and `elevation`.

```{code-cell} ipython3
def make_volume(elevations=(0.5, 1.0, 2.0), nray=360, ngate=300, dr=500.0):
    az = (np.arange(nray) + 0.5) * 360.0 / nray
    rng = dr * (np.arange(ngate) + 1)
    sweeps = {}
    for k, el in enumerate(elevations):
        sweeps[f"sweep_{k}"] = xr.Dataset(
            {"DBZH": (("azimuth", "range"), np.full((nray, ngate), 35.0, dtype=np.float32), {"units": "dBZ"})},
            coords={"azimuth": az, "range": rng, "elevation": ("azimuth", np.full(nray, el))},
        )
    root = xr.Dataset(coords={"latitude": LAT, "longitude": LON, "altitude": ALTITUDE})
    params = xr.Dataset({"radar_beam_width_v": 1.0, "radar_beam_width_h": 1.0})
    return xr.DataTree.from_dict({"/": root, "radar_parameters": params, **sweeps})

volume = make_volume()
t0 = time.perf_counter()
bb = volume.radarx.beam_blockage(dem)  # a DataTree with a node of products per sweep
print(f"{time.perf_counter() - t0:.2f} s for {len(bb.children)} sweeps of {volume['sweep_0'].DBZH.size} gates")
print(bb["sweep_0"].to_dataset()[["PBB", "CBB", "TERRAIN_HEIGHT", "TERRAIN_CLUTTER"]])
```

The maps show the lowest sweeps in plan view. The cumulative blockage stays
high behind a ridge (the shadow) even where the terrain drops again, and the
clutter flag marks the gates in or at the surface.

```{code-cell} ipython3
def to_xy(sweep_name):
    ds = volume[sweep_name].to_dataset()
    x, y, _ = antenna_to_cartesian(
        ds.range.values[None, :], ds.azimuth.values[:, None], float(ds.elevation[0]), site_altitude=ALTITUDE
    )
    return x / 1e3, y / 1e3

fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), layout="constrained")
lat_km = (dem.latitude.values - LAT) * 111.2
lon_km = (dem.longitude.values - LON) * 111.2 * np.cos(np.deg2rad(LAT))
pm = axes[0].pcolormesh(lon_km, lat_km, dem.values, cmap="terrain", vmin=0, vmax=1600, shading="auto")
fig.colorbar(pm, ax=axes[0], shrink=0.8, label="terrain height (m)")
axes[0].plot(0, 0, "k^", ms=8)
axes[0].set_title("synthetic DEM, radar (triangle)")
for ax, name, el in zip(axes[1:], ("sweep_0", "sweep_1", "sweep_2"), (0.5, 1.0, 2.0)):
    x, y = to_xy(name)
    pm = ax.pcolormesh(x, y, bb[name].CBB.values, cmap="magma_r", vmin=0, vmax=1, shading="auto")
    ax.contour(x, y, bb[name].TERRAIN_CLUTTER.values, levels=[0.5], colors="c", linewidths=0.6)
    ax.set_title(f"CBB at {el}° (cyan: terrain clutter)")
fig.colorbar(pm, ax=axes[1:], shrink=0.8, label="cumulative beam blockage")
for ax in axes:
    ax.set(aspect="equal", xlim=(-150, 150), ylim=(-150, 150), xlabel="east (km)")
axes[0].set_ylabel("north (km)")
plt.show()
for name, el in zip(("sweep_0", "sweep_1", "sweep_2"), (0.5, 1.0, 2.0)):
    cbb = bb[name].CBB
    print(f"{el:>3}°: {float((cbb > 0.5).mean()) * 100:5.1f} % of the gates have CBB > 0.5, "
          f"{float((cbb > 0.1).mean()) * 100:5.1f} % have CBB > 0.1, "
          f"{float(bb[name].TERRAIN_CLUTTER.mean()) * 100:4.1f} % are terrain clutter")
```

## One ray

For the ray that looks across most partial blockage: the terrain under the beam,
the beam centre and its half-power edges, and the partial and cumulative blockage
along the ray.

```{code-cell} ipython3
node = bb["sweep_0"].to_dataset()
partial = ((node.PBB > 0.1) & (node.PBB < 0.9)).sum("range")
ray = int(np.argmax(partial.values))
az = float(node.azimuth[ray])
rng_km = node.range.values / 1e3
x, y, z = antenna_to_cartesian(node.range.values, az, 0.5, site_altitude=ALTITUDE)
ground_km = np.hypot(x, y) / 1e3
half_width = node.range.values * np.deg2rad(1.0) / 2.0  # the half-power radius a = r beta / 2

fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, layout="constrained",
                                  gridspec_kw={"height_ratios": [2, 1]})
top.fill_between(ground_km, 0, node.TERRAIN_HEIGHT[ray], color="0.55", label="terrain")
top.fill_between(ground_km, z - half_width, z + half_width, color="C0", alpha=0.25, label="half-power beam")
top.plot(ground_km, z, color="C0", label="beam centre")
top.set(ylabel="height above sea level (m)", ylim=(0, 4500), title=f"azimuth {az:.1f}°, elevation 0.5°")
top.legend(loc="upper left")
bottom.plot(ground_km, node.PBB[ray], label="PBB")
bottom.plot(ground_km, node.CBB[ray], label="CBB")
flag = node.TERRAIN_CLUTTER[ray].values == 1
bottom.plot(ground_km[flag], np.full(flag.sum(), -0.06), "|", color="k", label="terrain clutter")
bottom.set(xlabel="ground distance (km)", ylabel="blockage", ylim=(-0.12, 1.05))
bottom.legend(loc="center left")
plt.show()
```

## Checks against the definition

Terrain at a known height below a gate gives a known answer. A flat plateau
exactly at the beam centre grazes the beam: PBB is 0.5 for either pattern. At
plus or minus the half-power radius $a$ the uniform beam is blocked 100 % or 0 %
and the Gaussian beam by the normal distribution function at $\pm 1.177\,\sigma$,
88 % and 12 %. The evaluation below integrates the beam cross section
numerically and is independent of the kernel.

```{code-cell} ipython3
def uniform_truth(y, a):
    if y <= -a:
        return 0.0
    if y >= a:
        return 1.0
    return quad(lambda u: 2 * np.sqrt(a * a - u * u), -a, y)[0] / (np.pi * a * a)

def gaussian_truth(y, a):
    sigma = a / np.sqrt(2 * np.log(2))
    return quad(lambda u: np.exp(-u * u / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma), -14 * sigma, min(y, 14 * sigma))[0]

sweep = volume["sweep_0"].to_dataset()
sweep = sweep.assign_coords(latitude=LAT, longitude=LON, altitude=ALTITUDE)
k = 79  # the gate at 40 km
r = float(sweep.range[k])
_, _, z_gate = antenna_to_cartesian(r, 0.0, 0.5, site_altitude=ALTITUDE)
a = r * np.deg2rad(1.0) / 2.0
flat = xr.DataArray(
    np.zeros((5, 5)), dims=("latitude", "longitude"),
    coords={"latitude": np.linspace(LAT - 2, LAT + 2, 5), "longitude": np.linspace(LON - 2, LON + 2, 5)},
)
print(f"gate at {r / 1e3:.0f} km: beam centre {float(z_gate):.0f} m, half-power radius {a:.0f} m")
print(f"{'terrain minus beam centre':>28} | {'uniform':>9} {'truth':>7} | {'Gaussian':>9} {'truth':>7}")
for label, y in (("-3 a", -3 * a), ("-a", -a), ("0 (grazing)", 0.0), ("+a", a), ("+3 a", 3 * a)):
    plateau = flat + float(z_gate) + y
    row = []
    for pattern, truth in (("uniform", uniform_truth), ("gaussian", gaussian_truth)):
        out = beam_blockage(sweep, plateau, pattern=pattern, beamwidth=1.0)
        row += [float(out.PBB[0, k]), truth(y, a)]
    print(f"{label:>28} | {row[0]:9.4f} {row[1]:7.4f} | {row[2]:9.4f} {row[3]:7.4f}")
```

The invariants hold on the random terrain of this page: every `PBB` and `CBB` is
between 0 and 1, `CBB` never decreases along a ray and is never smaller than
`PBB`, and a higher sweep is less blocked.

```{code-cell} ipython3
for name in ("sweep_0", "sweep_1", "sweep_2"):
    n = bb[name].to_dataset()
    pbb, cbb = n.PBB.values, n.CBB.values
    ok = (
        np.isfinite(pbb).all()
        and 0 <= pbb.min() and pbb.max() <= 1
        and 0 <= cbb.min() and cbb.max() <= 1
        and (cbb >= pbb - 1e-7).all()
        and (np.diff(cbb, axis=1) >= -1e-7).all()
    )
    print(name, "invariants hold:", ok, "| mean CBB", round(float(cbb.mean()), 3))
```

## Choices: pattern, refraction, sampling

The pattern, the refractivity gradient (`k_e`) and the sampling along the ray
change the result. Bech et al. (2003) show how strongly the blockage depends on
the vertical refractivity gradient; `radarx.fundamentals.geometry.k_e_from_refractivity`
turns a sounding into `k_e`. `oversample=n` also samples the terrain between the
gates, so that a ridge narrower than the gate spacing cannot slip through.

```{code-cell} ipython3
sweep0 = volume["sweep_0"].to_dataset().assign_coords(latitude=LAT, longitude=LON, altitude=ALTITUDE)
variants = {
    "Gaussian, k_e = 4/3": dict(),
    "uniform circle": dict(pattern="uniform"),
    "Gaussian, two-way": dict(two_way=True),
    "Gaussian, k_e = 1.0": dict(k_e=1.0),
    "Gaussian, k_e = 1.6": dict(k_e=1.6),
    "Gaussian, oversample 4": dict(oversample=4),
}
fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
for label, kwargs in variants.items():
    out = beam_blockage(sweep0, dem, beamwidth=1.0, **kwargs)
    values = out.CBB.values
    print(f"{label:>24}: CBB > 0.5 at {np.mean(values > 0.5) * 100:5.2f} % of the gates, "
          f"mean {np.mean(values):.3f}")
    ax.plot(ground_km, out.CBB.values[ray], label=label)
ax.set(xlabel="ground distance (km)", ylabel="CBB", xlim=(0, 80),
       title=f"cumulative blockage along azimuth {az:.1f}°, elevation 0.5°")
ax.legend(ncols=2)
plt.show()
```

## Use as a mask or a quality field

The blockage is not applied to anything. Pass `CBB` or the clutter flag where a
field is masked, or weight a rainfall estimate by $1 - \mathrm{CBB}$.

```{code-cell} ipython3
sw = volume["sweep_0"].to_dataset().assign_coords(latitude=LAT, longitude=LON, altitude=ALTITUDE)
products = sw.radarx.beam_blockage(dem, beamwidth=1.0)  # the accessor of a sweep
clean = sw.DBZH.where((products.CBB < 0.5) & (products.TERRAIN_CLUTTER == 0))
print(f"{int(clean.notnull().sum())} of {clean.size} gates kept")
print(products.attrs["radarx_method"])
print(products.attrs["radarx_references"])
merged = volume.radarx.assign(bb)  # the products of the volume merged into it
print(list(merged["sweep_0"].data_vars))
```

## References

- Bech, J., B. Codina, J. Lorente, and D. Bebbington, 2003: The sensitivity of single polarization weather radar beam blockage correction to variability in the vertical refractivity gradient. *Journal of Atmospheric and Oceanic Technology*, **20** (6), 845-855, <https://doi.org/10.1175/1520-0426(2003)020<0845:TSOSPW>2.0.CO;2>
- Lang, T. J., S. W. Nesbitt, and L. D. Carey, 2009: On the correction of partial beam blockage in polarimetric radar data. *Journal of Atmospheric and Oceanic Technology*, **26** (5), 943-957, <https://doi.org/10.1175/2008JTECHA1133.1>
- Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather Observations*, 2nd ed. Academic Press.
