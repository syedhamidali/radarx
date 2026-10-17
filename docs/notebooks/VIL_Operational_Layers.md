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

# VIL layers: default, filled to the ground and operational

The vertically integrated liquid (VIL) of a radar volume depends on what is
assumed below the lowest beam and above the highest one. `vil` has three ways to
handle it:

- the default integrates only between the beams, with the reflectivity varying
  linearly with height. The VIL is a lower bound, and `VIL_LOWER_BOUND` flags the
  columns where the layer below the lowest beam is missing;
- `fill_below=True` extends the lowest value down to the ground, as a pseudo-CAPPI
  does;
- `layers="operational"` uses constant layers, each one from the middle to the beam
  below to the middle to the beam above, the lowest from the ground and the highest
  to the top of its beam, as in the layer depths of the WSR-88D algorithm
  description.

The first section shows the three on a column with a known answer. The second
compares them with the Level III digital VIL (DVL) of the radar. The
[VIL notebook](VIL_and_Water_Content) has the rest of the products and the
comparison with the legacy VIL and MRMS.

```{code-cell} ipython3
import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.retrieve import vil

warnings.filterwarnings("ignore", category=RuntimeWarning)
```

## A column with a known answer

A grid of 1 km columns with levels every 500 m from 1 km up to 12 km. The
reflectivity is 45 dBZ from the ground to 8 km and zero above. The true VIL
of the column is the integral of $3.44\times10^{-6} Z^{4/7}$ over the 8 km
from the ground. The grid starts at 1 km, as the lowest beam of a radar does at
long range, so what happens below 1 km is only assumed.

```{code-cell} ipython3
z = np.arange(1000.0, 12001.0, 500.0)
column = xr.DataArray(
    np.where(z < 8000.0, 45.0, -30.0)[:, None, None] * np.ones((1, 3, 3)),
    dims=("z", "y", "x"),
    coords={"z": z, "y": np.arange(3) * 1e3, "x": np.arange(3) * 1e3},
    name="DBZH",
)
dbz = column.to_dataset()
truth = 3.44e-6 * 10 ** (45.0 * 4 / 7 / 10) * 8000.0

options = {
    "default": {},
    "fill_below": {"fill_below": True},
    "operational": {"layers": "operational"},
}
print(f"true VIL of the 8 km column: {truth:.1f} kg m-2")
for name, kwargs in options.items():
    value = float(vil(dbz, base_height=0.0, **kwargs).VIL.isel(y=1, x=1))
    print(f"{name:12s} {value:5.1f} kg m-2  ({100 * value / truth:3.0f} % of the true value)")
```

The default leaves out the 1 km below the lowest level and gets 85 % of the true
value. `fill_below` gets 98 % and `operational` 97 %. Neither reaches the whole,
because the top of the echo falls between two levels of the grid and each option
treats that layer in its own way: the default and `fill_below` count half the
reflectivity of the last echo level over the layer to the next one, `operational`
stops the layer at the middle of the two. Both assume that the rain below the lowest
level is like the rain at it, which is right for a deep uniform column and not for
rain that evaporates or grows below the beam.

## The KGWX squall line against the Level III digital VIL

The KGWX volume of 30 March 2022, 23:59 UTC, and the DVL (product 134, polar, 1°
by 1 km bins) of the same volume, both from the open NEXRAD archive on AWS. The
radarx VIL is averaged over the gates of each DVL bin (the bin mean) or taken as
their maximum, because the DVL value of a bin is closer to the maximum of the
finer gates than to their mean. If the archive cannot be reached the cell prints a
message and the rest is skipped.

```{code-cell} ipython3
import xradar as xd

from radarx.io.aws_data import download_file
from open_radar_data import DATASETS

try:
    volume_path = DATASETS.fetch("KGWX20220330_235959_V06")
    dvl_path = download_file(
        "unidata-nexrad-level3", "GWX_DVL_2022_03_30_23_59_59", "data"
    )
except Exception as err:  # offline: keep the first section
    volume_path = None
    print("NEXRAD archive not reachable:", type(err).__name__)

if volume_path is not None:
    drop = ["WRADH", "CCORH", "CFP", "ZDR", "PHIDP", "RHOHV"]
    volume = xd.io.open_nexradlevel2_datatree(volume_path, drop_variables=drop)

    def prepare(ds):
        """Mask the no-data codes and keep the first 150 km."""
        if "range" not in ds.dims:
            return ds
        ds = ds.sel(range=slice(None, 150e3))
        return ds.assign(DBZH=ds.DBZH.where(ds.DBZH > -32).astype("float32"))

    volume = volume.map_over_datasets(prepare)
    products = {
        "default": vil(volume),
        "fill_below": vil(volume, fill_below=True),
        "operational": vil(volume, layers="operational"),
    }
    print({name: round(float(p.VIL.max()), 1) for name, p in products.items()})
```

The DVL is read with MetPy. Every ray of the volume is matched to the DVL radial
nearest to its azimuth, and every gate to its 1 km bin.

```{code-cell} ipython3
from metpy.io import Level3File

if volume_path is not None:
    dvl = Level3File(dvl_path)
    block = dvl.sym_block[0][0]
    raw = np.array([np.frombuffer(bytes(r), dtype=np.uint8) for r in block["data"]])
    dvl_vil = np.ma.filled(dvl.map_data(raw).astype(float), np.nan)
    centre = 0.5 * (np.array(block["start_az"]) + np.array(block["end_az"]))

    polar = products["default"]
    azimuth = polar.azimuth.values
    ground = polar.ground_range.values[None, :] * np.ones((azimuth.size, 1))
    offset = (centre[None, :] - azimuth[:, None] + 180) % 360 - 180
    iaz = np.argmin(np.abs(offset), axis=1)[:, None] * np.ones(ground.shape, dtype=int)
    ibin = np.minimum(np.floor(ground / 1e3).astype(int), dvl_vil.shape[1] - 1)
    inside = (ground > 20e3) & (ground < 150e3)

    def on_dvl_bins(values, reduce="mean"):
        """The mean or the maximum of the gates in each DVL bin."""
        ok = inside & np.isfinite(values)
        if reduce == "max":
            out = np.full(dvl_vil.shape, -np.inf)
            np.maximum.at(out, (iaz[ok], ibin[ok]), values[ok])
            return np.where(np.isfinite(out), out, np.nan)
        total, count = np.zeros(dvl_vil.shape), np.zeros(dvl_vil.shape)
        np.add.at(total, (iaz[ok], ibin[ok]), values[ok])
        np.add.at(count, (iaz[ok], ibin[ok]), 1)
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)

    bins = {
        (name, reduce): on_dvl_bins(p.VIL.values, reduce)
        for name, p in products.items()
        for reduce in ("mean", "max")
    }
    range_km = (np.arange(dvl_vil.shape[1]) + 0.5)[None, :] * np.ones((dvl_vil.shape[0], 1))
    edges = [20, 50, 100, 150]
    print("median ratio to the DVL where the DVL exceeds 1 kg m-2")
    print("variant, bin        " + "".join(f"{lo:>4d}-{hi:<4d} km  " for lo, hi in zip(edges[:-1], edges[1:])) + "all")
    for (name, reduce), mine in bins.items():
        ok = np.isfinite(mine) & (dvl_vil > 1.0)
        ratios = [
            np.median(mine[ok & (range_km >= lo) & (range_km < hi)] / dvl_vil[ok & (range_km >= lo) & (range_km < hi)])
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
        overall = np.median(mine[ok] / dvl_vil[ok])
        print(f"{name:12s} {reduce:4s}   " + "".join(f"{r:9.2f}     " for r in ratios) + f"{overall:.2f}")
```

```{code-cell} ipython3
if volume_path is not None:
    colours = {"default": "tab:blue", "fill_below": "tab:orange", "operational": "tab:green"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    edge_grid = np.arange(20, 151, 10)
    mid = 0.5 * (edge_grid[:-1] + edge_grid[1:])
    for (name, reduce), mine in bins.items():
        ok = np.isfinite(mine) & (dvl_vil > 1.0)
        ratio = [
            np.median(mine[ok & (range_km >= lo) & (range_km < hi)] / dvl_vil[ok & (range_km >= lo) & (range_km < hi)])
            for lo, hi in zip(edge_grid[:-1], edge_grid[1:])
        ]
        axes[0].plot(mid, ratio, "-" if reduce == "mean" else "--", color=colours[name], label=f"{name}, bin {reduce}")
    axes[0].axhline(1.0, color="k", lw=0.8)
    axes[0].set(xlabel="range (km)", ylabel="median radarx VIL / DVL", title="Ratio to the DVL")
    axes[0].legend(frameon=False, fontsize=8)
    mine = bins[("operational", "max")]
    ok = np.isfinite(mine) & np.isfinite(dvl_vil)
    hb = axes[1].hexbin(dvl_vil[ok], mine[ok], gridsize=50, extent=(0, 40, 0, 40), mincnt=1, cmap="Blues", bins="log")
    fig.colorbar(hb, ax=axes[1], label="DVL bins", shrink=0.8)
    axes[1].plot([0, 40], [0, 40], "k-", lw=0.8)
    axes[1].set(xlabel="DVL (kg m$^{-2}$)", ylabel="radarx VIL, operational, bin maximum (kg m$^{-2}$)", aspect="equal", title="Bin maximum against the DVL")
    plt.show()
```

The default gets worse with range, where the lowest beam is high: its ratio to the
DVL falls from 0.72 at 20 to 50 km to 0.47 at 100 to 150 km (bin mean). Both
`fill_below` and `layers="operational"` make the ratio flat with range, at 0.78 and
0.75 for the bin mean and 0.93 and 0.88 for the bin maximum. `layers="operational"`
is a little lower than `fill_below` because it has no linear interpolation between the
beams, and it does not close the gap by itself: most of what is left is the cell
size of the comparison, since the DVL bin is closer to the maximum of the finer
gates than to their mean. In the cores, where the DVL exceeds 20 kg m$^{-2}$, radarx
stays below the one-to-one line (right panel, `operational`). The algorithm
of the radar product is not published in the files, so these are ratios to an
operational product and not errors. They are from one volume; the
[VIL notebook](VIL_and_Water_Content) has the legacy VIL and MRMS, and the
docstring of `vil` gives the median ratios over four volumes of the same squall line.

## References

- Greene, D. R., and R. A. Clark, 1972: Vertically Integrated Liquid Water: A New Analysis Tool. *Monthly Weather Review*, **100**, 548-552, <https://doi.org/10.1175/1520-0493(1972)100<0548:VILWNA>2.3.CO;2>
