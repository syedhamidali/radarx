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

# Cell identification, tracking and statistics

Storm cells are found with several thresholds and a watershed (`identify_cells`,
Heikenfeld et al. 2019), linked from volume to volume by their overlap after the storm
motion (`track_cells`, Feng et al. 2023), and any gridded field (rotation, lightning,
columns, DSD) is summarised inside every cell as a time series (`cell_statistics`).
Compiled, multithreaded kernels do the labelling, the watershed, the overlap counts and
the sums over the cells, with NumPy implementations that give the same results. The data
here are synthetic, with known answers.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.retrieve import cell_statistics, identify_cells, track_cells
```

## Storm cells

Three storms on a 1 km grid, 5 min apart, that move with the flow at about 20 m/s to the
east and 5 m/s to the north: one keeps its size, one dies after four volumes, and one splits
into two in the third volume. The multi-threshold detection puts a feature in each core and the watershed
gives each feature the cell around it; the cells of the splitting storm are divided
along the saddle between their maxima.

```{code-cell} ipython3
sx = sy = np.arange(80) * 1000.0
stimes = np.datetime64("2022-03-30T22:00") + np.arange(8) * np.timedelta64(300, "s")


def cell(cx, cy, peak, sigma):
    return peak * np.exp(-((sx[None, :] - cx) ** 2 + (sy[:, None] - cy) ** 2) / (2 * sigma**2))


frames = []
for i in range(8):
    f = cell(10e3 + 6e3 * i, 15e3 + 1.5e3 * i, 55, 3000.0)  # keeps its size
    if i < 4:
        f = f + cell(30e3 + 5.4e3 * i, 50e3 + 0.9e3 * i, 48, 3000.0)  # dies
    if i < 3:
        f = f + cell(15e3 + 6.3e3 * i, 28e3 + 1.8e3 * i, 55, 4000.0)  # one storm
    else:
        d = 1.5e3 * (i - 2)  # two cells that move apart
        cx, cy = 15e3 + 6.3e3 * i, 28e3 + 1.8e3 * i
        f = f + cell(cx - d, cy, 52, 2200.0) + cell(cx + d, cy, 50, 2200.0)
    frames.append(f - 10.0)
dbz = xr.DataArray(
    np.array(frames), dims=("time", "y", "x"), coords={"time": stimes, "y": sy, "x": sx}, name="DBZH",
    attrs={"units": "dBZ"},
)

cells = identify_cells(dbz, thresholds=(20, 30, 40), min_area=4.0)
tracks = track_cells(cells, motion=(20.0, 5.0), min_overlap=0.2)
print("cells per volume:", cells.CELL_COUNT.values.tolist())
print("tracks:", tracks.sizes["cell"], "| lifetimes", tracks.cell_lifetime.values.tolist())
print("split from:", tracks.cell_split_from.values.tolist(), "| merged into:", tracks.cell_merged_into.values.tolist())
```

`track_cells` shifts the cells of a volume by the storm motion and links a pair of cells
if their overlap is at least a fraction (`min_overlap`) of the area of either. The
largest overlap continues the track; a cell with only a continued parent starts a new track
that is marked as split from it, and a cell that is linked only to a cell of another track
ends and is marked as merged into it. Without the motion only cells that overlap in place
are linked, and `max_speed` links cells that do not overlap by their distance from the
predicted position. The motion can come from `estimate_motion` (a vector, or a field on
the grid with `tile=`).

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
for ax, t in zip(axes[:2], (0, 7)):
    pm = ax.pcolormesh(sx / 1e3, sy / 1e3, np.where(tracks.CELL_ID.isel(time=t) > 0, tracks.CELL_ID.isel(time=t), np.nan),
                       cmap="tab10", vmin=1, vmax=10, shading="auto")
    ax.set(xlabel="x (km)", ylabel="y (km)", aspect="equal", title=f"tracks at volume {t}")
for tid in tracks.cell:
    axes[2].plot(tracks.cell_x.sel(cell=tid) / 1e3, tracks.cell_y.sel(cell=tid) / 1e3, "o-", ms=3, label=f"track {int(tid)}")
axes[2].set(xlabel="x (km)", ylabel="y (km)", aspect="equal", title="positions of the centres")
axes[2].legend(frameon=False, fontsize=8)
plt.show()
```

### Statistics inside the cells

`cell_statistics` collects mean, maximum, minimum, sum and count of any field on the same
grid inside every tracked cell, and returns time series on `(cell, time)`. The fields can
be any products on the grid: the azimuthal shear of `llsd` or `azimuthal_shear`, the gridded
lightning density, the depth of the ZDR or KDP column, the DSD parameters. A field with a
vertical dimension must be reduced (for example to its maximum) first. Here a rain-rate-like
field and the peak reflectivity of the cells:

```{code-cell} ipython3
rate = (10 ** (dbz / 10 / 1.6) * 0.0365 ** (1 / 1.6)).rename("RATE")  # a Z-R relation, for the example
rate.attrs["units"] = "mm h-1"
series = cell_statistics(tracks.CELL_ID, xr.Dataset({"DBZH": dbz, "RATE": rate}), stats=("max", "mean", "count"))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
for tid in series.cell:
    axes[0].plot(series.time, series.DBZH_max.sel(cell=tid), "o-", ms=3, label=f"track {int(tid)}")
    axes[1].plot(series.time, series.RATE_mean.sel(cell=tid), "o-", ms=3)
axes[0].set(ylabel="peak reflectivity (dBZ)")
axes[1].set(ylabel="mean rain rate (mm/h)")
axes[0].legend(frameon=False, fontsize=8)
for ax in axes:
    ax.tick_params(axis="x", rotation=30)
plt.show()
```

The mask of one tracked cell (`tracks.CELL_ID == 1`) is a valid `region` of `storm_size_sorting`, so
the separation vector of each storm of a grid can be measured on the grid of its own cell.

## What is not here

- **Tobac.** The identification and the segmentation follow tobac (the multi-threshold
  features and the watershed), without its dependencies. The splits and mergers are
  decided from the links of two consecutive volumes, not from the minimum spanning tree of tobac
  1.5 (Sokolowsky et al. 2024). The overlap of cells is not enlarged by dilation
  (PyFLEXTRKR does that for slow volume scans).
- **TITAN and SCIT.** The cost functions of the operational cell trackers are not
  implemented.
- **Numbers of the papers.** The papers show case studies, not tables that a synthetic case
  can reproduce. The tests check the definitions on cases with known answers.

## References

- Feng, Z., J. Hardin, H. C. Barnes, J. Li, L. R. Leung, A. Varble, and Z. Zhang, 2023: PyFLEXTRKR: a flexible feature tracking Python software for convective cloud analysis. *Geoscientific Model Development*, **16**, 2753-2776, <https://doi.org/10.5194/gmd-16-2753-2023>
- Heikenfeld, M., P. J. Marinescu, M. Christensen, D. Watson-Parris, F. Senf, S. C. van den Heever, and P. Stier, 2019: tobac 1.2: towards a flexible framework for tracking and analysis of clouds in diverse datasets. *Geoscientific Model Development*, **12**, 4551-4570, <https://doi.org/10.5194/gmd-12-4551-2019>
- Sokolowsky, G. A., S. W. Freeman, W. K. Jones, J. Kukulies, F. Senf, P. J. Marinescu, M. Heikenfeld, K. N. Brunner, E. C. Bruning, S. M. Collis, R. C. Jackson, G. R. Leung, N. Pfeifer, B. A. Raut, S. M. Saleeby, P. Stier, and S. C. van den Heever, 2024: tobac v1.5: introducing fast 3D tracking, splits and mergers, and other enhancements for identifying and analysing meteorological phenomena. *Geoscientific Model Development*, **17**, 5309-5330, <https://doi.org/10.5194/gmd-17-5309-2024>
