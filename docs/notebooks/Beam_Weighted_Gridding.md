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

# Beam-Weighted Gridding

+++

`radarx.grid.grid_gates` and `radarx.grid.grid_rhi` give every grid point a
weighted mean of the gates within a radius of influence. With
`weighting="cressman"` or `"barnes"` the weight depends on the distance between
the gate and the point only. With `weighting="beam"` it depends on where the
point sits in the pulse volume of the gate: a Gaussian that is as wide across
the beam as the two-way power pattern of a beam of half-power width `beamwidth`
and as wide along it as `range_resolution` (Doviak and Zrnic 1993, Eq. 5.75).
The beams that pass through the point dominate, so a radial velocity, which is
a projection on the beam, is mixed less with the gates of other directions.

This notebook grids one volume of KGWX (30 March 2022, a squall line over
Mississippi) three ways and compares the results. Without network access it
falls back to a synthetic volume with a sharp velocity edge.

**References**

- Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
  Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6.

```{code-cell} ipython3
import cmweather  # noqa
import fsspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xradar as xd

import radarx as rx
```

## The volume

The real volume is the first KGWX volume of the squall line, with the NEXRAD
no-data codes masked. The synthetic fallback has a gust front: the wind blows
from the west at 25 m/s on one side of a line and from the east at 15 m/s on the
other, with a reflectivity core along it.

```{code-cell} ipython3
def synthetic_volume():
    ranges = np.arange(250.0, 120e3, 250.0)
    azimuths = np.arange(0.5, 360.0, 1.0)
    sweeps = {}
    for k, elevation in enumerate((0.5, 1.5, 2.5, 3.5, 4.5, 6.0, 8.0, 10.0)):
        gate_x = ranges[None, :] * np.cos(np.radians(elevation)) * np.sin(np.radians(azimuths))[:, None]
        gate_y = ranges[None, :] * np.cos(np.radians(elevation)) * np.cos(np.radians(azimuths))[:, None]
        edge = gate_x - 0.5 * gate_y - 10e3
        wind_u = np.where(edge > 0, -15.0, 25.0)
        radial = wind_u * np.sin(np.radians(azimuths))[:, None] * np.cos(np.radians(elevation))
        dbz = 20.0 + 30.0 * np.exp(-((edge / 8e3) ** 2)) * (gate_y < 60e3)
        sweeps[f"sweep_{k}"] = xr.Dataset(
            {"DBZH": (("azimuth", "range"), dbz), "VRADH": (("azimuth", "range"), radial)},
            coords={
                "azimuth": azimuths,
                "range": ranges,
                "elevation": ("azimuth", np.full(azimuths.size, elevation)),
                "time": ("azimuth", np.full(azimuths.size, np.datetime64("2022-03-30T23:46"))),
            },
        )
    root = xr.Dataset(coords={"latitude": 32.4, "longitude": -90.1, "altitude": 100.0})
    return xr.DataTree.from_dict({"/": root, **sweeps})


def real_volume():
    nodata = {"DBZH": -32.0, "VRADH": -63.9}
    local_file = fsspec.open_local(
        "simplecache::s3://unidata-nexrad-level2/2022/03/30/KGWX/KGWX20220330_234639_V06",
        s3={"anon": True},
        filecache={"cache_storage": "."},
    )
    tree = xd.io.open_nexradlevel2_datatree(local_file)
    for sweep in [n for n in tree.children if n.startswith("sweep")]:
        ds = tree[sweep].to_dataset()
        ds = ds[[v for v in nodata if v in ds]]
        for field, code in nodata.items():
            if field in ds:
                ds[field] = ds[field].where(ds[field] > code)
        tree[sweep] = ds
    return tree.load()


try:
    volume = real_volume()
    source = "KGWX, 30 March 2022 23:46 UTC"
    window = (20e3, 90e3, 10e3, 90e3)
except Exception as error:  # offline
    print(f"no network access ({type(error).__name__}), using a synthetic volume")
    volume = synthetic_volume()
    source = "synthetic volume"
    window = (-30e3, 60e3, -40e3, 50e3)
source
```

## Three weightings

All three use the same radius of influence (1.5 km), so the same gates take
part and only their weights differ. The cone grid (`grid_cones`) reproduces the
gates themselves and is the reference. The window below is a stretch of the
squall line where the velocity field has sharp edges: the gust front, where the
wind turns from towards the radar to away from it, and patches of noisy or
folded velocity behind it.

```{code-cell} ipython3
x = np.arange(-100e3, 100e3 + 1, 1000.0)
y = np.arange(-100e3, 100e3 + 1, 1000.0)
z = np.array([2000.0])

options = dict(roi=1500.0, beamwidth=1.0)
grids = {
    "cone (reference)": rx.grid.grid_cones(volume, "VRADH", x, y, z),
    "Cressman": rx.grid.grid_gates(volume, ["VRADH"], x, y, z, weighting="cressman", **options),
    "Barnes": rx.grid.grid_gates(volume, ["VRADH"], x, y, z, weighting="barnes", **options),
    "beam": rx.grid.grid_gates(volume, ["VRADH"], x, y, z, weighting="beam", **options),
}
reference = grids["cone (reference)"].VRADH.isel(z=0)
for name in ("Cressman", "Barnes", "beam"):
    field = grids[name].VRADH.isel(z=0)
    diff = (field - reference).where(field.notnull() & reference.notnull())
    print(f"{name:9s} rms difference to the cone grid: {float(np.sqrt((diff ** 2).mean())):.2f} m/s")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), layout="constrained", sharey=True)
for ax, (name, grid) in zip(axes, grids.items()):
    im = grid.VRADH.isel(z=0).plot(ax=ax, x="x", y="y", cmap="RdBu_r", vmin=-30, vmax=30, add_colorbar=False)
    ax.set(title=name, aspect="equal", xlim=window[:2], ylim=window[2:], xlabel="x (m)", ylabel="y (m)")
fig.colorbar(im, ax=axes, label="radial velocity at 2 km (m/s)", shrink=0.8)
plt.show()
```

Cressman and Barnes weights fall off with distance in every direction, so a
point between two beams averages gates that look at it from different angles
and the edges round off. The beam weights follow the gates whose pulse volume
contains the point, and the edge keeps its shape and its jump. The price is that
on a smooth field the beam weights follow single beams and are a little noisier
than Cressman.

## A vertical section

`grid_rhi` does the same for a vertical plane through the radar. Here the
reflectivity along an azimuth of 45 degrees, gridded with the beam weights:

```{code-cell} ipython3
distance = np.arange(1000.0, 120e3, 500.0)
height = np.arange(250.0, 10e3, 250.0)
rhi = rx.grid.grid_rhi(
    volume, ["DBZH"], azimuth=45.0, distance=distance, z=height, weighting="beam", roi=1500.0, beamwidth=1.0
)
fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
rhi.DBZH.plot(ax=ax, cmap="ChaseSpectral", vmin=-10, vmax=70, cbar_kwargs={"label": "DBZH (dBZ)"})
ax.set(title=f"{source}: section at 45 degrees, beam weights")
plt.show()
```
