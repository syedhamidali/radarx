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

# Blockage-Weighted Merging of Radars

+++

Where a mountain stands in the beam of a radar, that radar underestimates the
reflectivity behind it, and in a mosaic it should count less than a radar that
sees the storm unobstructed. `radarx.grid.merge_radars` takes the blocked
fraction `b` of the beam (0 to 1) on the grid through `blockage=`. The weight of
a radar in a cell is multiplied by `1 - b`, and a radar is left out of the cell
where `b` reaches `max_blockage` (default 0.5). The fraction is a gridded field:
compute the cumulative beam blockage `CBB` of `radarx.retrieve.beam_blockage`
from a digital elevation model for each volume, add it to the volume and grid it
with the other fields (see the beam blockage notebook).

Here the blockage is synthetic, so that the answer is known. Three NEXRAD
radars around a squall line in Mississippi (KGWX, KNQA, KDGX, 30 March 2022)
are gridded together. A ridge on the side of KDGX that faces KGWX takes up to 80 percent of its
beam in a sector of azimuths. A blocked beam loses power, so KDGX reads
`10 log10(1 - b)` dB too low there. We merge the radars with and without the
blockage weight and compare both with the merge of the unspoiled radars.
Without network access the notebook uses three synthetic radars with the same
setup.

```{code-cell} ipython3
import cmweather  # noqa
import fsspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xradar as xd

import radarx as rx
```

## Three radars on one grid

```{code-cell} ipython3
names = ["KGWX", "KNQA", "KDGX"]
files = {
    "KGWX": "KGWX20220330_234639_V06",
    "KNQA": "KNQA20220330_234905_V06",
    "KDGX": "KDGX20220330_234843_V06",
}


def real_radar(name):
    local_file = fsspec.open_local(
        f"simplecache::s3://unidata-nexrad-level2/2022/03/30/{name}/{files[name]}",
        s3={"anon": True},
        filecache={"cache_storage": "."},
    )
    tree = xd.io.open_nexradlevel2_datatree(local_file)
    for sweep in [n for n in tree.children if n.startswith("sweep")]:
        ds = tree[sweep].to_dataset()[["DBZH"]]
        tree[sweep] = ds.where(ds.DBZH > -32.0)
    return tree.load()


def synthetic_radar(lat, lon):
    """A volume of a storm line at 50 dBZ, the same for all radars."""
    ranges = np.arange(500.0, 150e3, 500.0)
    azimuths = np.arange(0.5, 360.0, 1.0)
    sweeps = {}
    for k, elevation in enumerate((0.5, 1.5, 2.5, 3.5, 4.5, 6.0, 8.0, 10.0, 14.0)):
        c = np.cos(np.radians(elevation))
        east = ranges[None, :] * c * np.sin(np.radians(azimuths))[:, None] + (lon + 90.0) * 94e3
        north = ranges[None, :] * c * np.cos(np.radians(azimuths))[:, None] + (lat - 32.0) * 111.2e3
        dbz = 15.0 + 35.0 * np.exp(-(((east - 20e3 + 0.4 * (north - 20e3)) / 12e3) ** 2)) * np.exp(-(((north - 20e3) / 60e3) ** 2))
        sweeps[f"sweep_{k}"] = xr.Dataset(
            {"DBZH": (("azimuth", "range"), dbz)},
            coords={
                "azimuth": azimuths,
                "range": ranges,
                "elevation": ("azimuth", np.full(azimuths.size, elevation)),
                "time": ("azimuth", np.full(azimuths.size, np.datetime64("2022-03-30T23:47"))),
            },
        )
    root = xr.Dataset(coords={"latitude": lat, "longitude": lon, "altitude": 100.0})
    return xr.DataTree.from_dict({"/": root, **sweeps})


try:
    radars = [real_radar(name) for name in names]
    source = "KGWX, KNQA and KDGX, 30 March 2022"
except Exception as error:  # offline
    print(f"no network access ({type(error).__name__}), using synthetic radars")
    radars = [synthetic_radar(32.4, -90.1), synthetic_radar(32.9, -89.6), synthetic_radar(31.9, -89.6)]
    source = "synthetic radars"

x = np.arange(-160e3, 160e3 + 1, 2000.0)
y = np.arange(-160e3, 160e3 + 1, 2000.0)
z = np.arange(1000.0, 5000.0 + 1, 500.0)
grid = rx.grid.grid_radars(radars, x, y, z, data_vars=["DBZH"], names=names, max_range=230e3)
grid.DBZH.dims
```

## A synthetic ridge

The blocked fraction depends on the azimuth of the beam from KDGX, not on the
height, so it is a field on `(radar, y, x)`: zero for the other radars, a bump
of 0.8 for KDGX in the direction of KGWX (with a width of 20 degrees), so that
KGWX sees what KDGX does not. The reflectivity of KDGX loses the matching power.

```{code-cell} ipython3
kdgx = grid.sel(radar="KDGX")
partner = grid.sel(radar="KGWX")
toward_partner = np.degrees(
    np.arctan2(float(partner.radar_x - kdgx.radar_x), float(partner.radar_y - kdgx.radar_y))
)
offset = (kdgx.azimuth - toward_partner + 180.0) % 360.0 - 180.0
b_kdgx = 0.8 * np.exp(-((offset / 20.0) ** 2))
blockage = xr.concat([xr.zeros_like(b_kdgx) if n != "KDGX" else b_kdgx for n in names], dim="radar")
blockage = blockage.assign_coords(radar=names).transpose("radar", "y", "x")

spoiled = grid.assign(
    CBB=blockage,
    DBZH=grid.DBZH + 10.0 * np.log10(1.0 - blockage),  # broadcast over z
)
```

## Merging

Three merges at the same bias: the unspoiled radars (the answer), the spoiled
radars without a blockage weight, and the spoiled radars with it. The last is
run twice: with the default `max_blockage=0.5`, which drops KDGX wherever more
than half of its beam is blocked, and with `max_blockage=0.95`, which keeps it
with a small weight.

```{code-cell} ipython3
truth = rx.grid.merge_radars(grid, "DBZH")
naive = rx.grid.merge_radars(spoiled, "DBZH")
dropped = rx.grid.merge_radars(spoiled, "DBZH", blockage="CBB")
weighted = rx.grid.merge_radars(spoiled, "DBZH", blockage="CBB", max_blockage=0.95)

# cells in the ridge sector that at least two radars see, scored at the height (1 to 3 km,
# below the melting layer) where there are most of them
ridge = (b_kdgx > 0.3) & (grid.DBZH.notnull().sum("radar") >= 2)
height = float(ridge.sel(z=slice(1000, 3000)).sum(["y", "x"]).idxmax("z"))
scored = ridge.sel(z=height)
for name, merged in (("no blockage weight", naive), ("max_blockage 0.5", dropped), ("max_blockage 0.95", weighted)):
    err = (merged.DBZH - truth.DBZH).sel(z=height).where(scored)
    print(f"{name:20s} in the ridge sector at {height / 1e3:.1f} km: mean error {float(err.mean()):+.2f} dB, "
          f"rms error {float(np.sqrt((err ** 2).mean())):.2f} dB")
```

```{code-cell} ipython3
level = dict(z=height)
xs, ys = scored.x.where(scored.any("y")).dropna("x"), scored.y.where(scored.any("x")).dropna("y")
fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), layout="constrained", sharey=True)
errors = {
    "no blockage weight": naive,
    "max_blockage 0.5": dropped,
    "max_blockage 0.95": weighted,
}
im0 = (b_kdgx.assign_coords(x=b_kdgx.x / 1e3, y=b_kdgx.y / 1e3)).plot(
    ax=axes[0], cmap="Greys", vmin=0, vmax=1, add_colorbar=False
)
axes[0].set_title("blocked fraction of KDGX")
for ax, (title, merged) in zip(axes[1:], errors.items()):
    diff = (merged.DBZH.sel(**level) - truth.DBZH.sel(**level)).where(scored)
    im = diff.assign_coords(x=diff.x / 1e3, y=diff.y / 1e3).plot(
        ax=ax, cmap="RdBu_r", vmin=-6, vmax=6, add_colorbar=False
    )
    ax.set_title(title)
for ax in axes:
    ax.set(aspect="equal", xlabel="east of the origin (km)", ylabel="north of the origin (km)",
           xlim=(float(xs.min()) / 1e3 - 10, float(xs.max()) / 1e3 + 10),
           ylim=(float(ys.min()) / 1e3 - 10, float(ys.max()) / 1e3 + 10))
    ax.plot(grid.radar_x.sel(radar="KDGX") / 1e3, grid.radar_y.sel(radar="KDGX") / 1e3, "k^")
fig.colorbar(im, ax=axes[1:], label=f"merged minus unspoiled merge, DBZH at {height / 1e3:.1f} km (dB)", shrink=0.8)
fig.colorbar(im0, ax=axes[0], label="b", fraction=0.05, pad=0.02)
plt.show()
```

The first panel is the ridge, the others show the cells in the sector that at
least two radars see (where `b` is above 0.3), at the height where there are
most of them. Without the weight, the low KDGX values pull the
mosaic down inside the sector by an amount that grows with `b`. With
`max_blockage=0.5` KDGX drops out of the cells where the beam is more than half
blocked and the mosaic returns to the other radars, with a step at the
edge of the sector where KDGX is switched off. With `max_blockage=0.95` the
weight `1 - b` fades KDGX out smoothly.
