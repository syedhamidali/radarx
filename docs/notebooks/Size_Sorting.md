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

# Size sorting

The storm-relative wind sorts raindrops by size, so that the regions of high
$K_{DP}$ and high $Z_{DR}$ at low levels of a storm are apart. `storm_size_sorting` measures
the distance and the direction of the vector between them, relative to the storm
motion, after Loeffler and Kumjian (2018). `zdr_arc` measures the same signature of a
supercell with the object-based algorithm of Wilson and Van Den Broeke (2021). A
compiled, multithreaded kernel finds the regions of all volumes, with a NumPy
implementation that gives the same results. The data here are synthetic, with known
answers.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.retrieve import storm_size_sorting
```

The storm moves north at 13 m/s. Its low-level KDP region is the core of the rain, and
the $Z_{DR}$ region lies 4 km to the east, on the side of the storm-relative inflow. The
regions are the largest set of grid points above an adaptive threshold (in steps
of 0.5), with at least `min_cells` points; their centroids are the medians of the
coordinates of the points. The vector from the KDP to the $Z_{DR}$ centroid is
90° to the right of the motion, which is fitted to the centroids over the volumes.

```{code-cell} ipython3
nt, step = 6, 300.0
gx = gy = np.arange(60) * 500.0
times = np.datetime64("2022-03-30T22:00") + np.arange(nt) * np.timedelta64(300, "s")


def gaussian(cx, cy, peak, sigma):
    return peak * np.exp(-((gx[None, :] - cx) ** 2 + (gy[:, None] - cy) ** 2) / (2 * sigma**2))


kdp_t, zdr_t = [], []
for i in range(nt):
    cy = 8000.0 + 13.3333 * step * i
    # the offset between the regions grows and turns for a time, and noise is added
    offset = (4000.0 + 400.0 * np.sin(i), 0.0)
    kdp_t.append(gaussian(8000.0, cy, 3.0, 2500.0))
    zdr_t.append(gaussian(8000.0 + offset[0], cy + offset[1], 4.0, 2500.0))
noise = np.random.default_rng(2)
low = xr.Dataset(
    {
        "KDP": (("time", "y", "x"), np.array(kdp_t) + noise.normal(0, 0.05, (nt, 60, 60))),
        "ZDR": (("time", "y", "x"), np.array(zdr_t) + noise.normal(0, 0.05, (nt, 60, 60))),
        "RHOHV": (("time", "y", "x"), np.full((nt, 60, 60), 0.97)),
    },
    coords={"time": times, "y": gy, "x": gx},
)
sorting = storm_size_sorting(low, rhohv="RHOHV", min_cells=10)
print("thresholds KDP", sorting.KDP_THRESHOLD.values, "ZDR", sorting.ZDR_THRESHOLD.values)
print("distance (km)", np.round(sorting.SEPARATION_DISTANCE.values, 2))
print("orientation (deg clockwise from the storm motion)", np.round(sorting.SEPARATION_ORIENTATION.values, 0))
print(
    f"fitted motion ({float(sorting.STORM_MOTION_U[0]):.1f}, {float(sorting.STORM_MOTION_V[0]):.1f}) m/s;",
    "size sorting parameter (km)", np.round(sorting.SIZE_SORTING_PARAMETER.values, 1),
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
ax = axes[0]
ax.contourf(gx / 1e3, gy / 1e3, low.ZDR.isel(time=2), levels=[1.5, 2.5, 3.5, 4.5], cmap="Blues", alpha=0.6)
ax.contour(gx / 1e3, gy / 1e3, low.KDP.isel(time=2), levels=[1.0, 2.0, 2.8], colors="r", linewidths=0.8)
ax.annotate(
    "",
    xy=(float(sorting.ZDR_CENTROID_X[2]) / 1e3, float(sorting.ZDR_CENTROID_Y[2]) / 1e3),
    xytext=(float(sorting.KDP_CENTROID_X[2]) / 1e3, float(sorting.KDP_CENTROID_Y[2]) / 1e3),
    arrowprops={"arrowstyle": "->", "lw": 2},
)
ax.set(xlabel="x (km)", ylabel="y (km)", aspect="equal", title="ZDR (blue), KDP (red), separation vector")
axes[1].plot(sorting.time, sorting.SEPARATION_DISTANCE, "o-", label="distance (km)")
axes[1].plot(sorting.time, sorting.SIZE_SORTING_PARAMETER, "s-", label="size sorting parameter (km)")
axes[1].legend(frameon=False)
axes[1].tick_params(axis="x", rotation=30)
plt.show()
```

The analysis region is the whole grid by default. For a grid with several storms give a
`box=(xmin, xmax, ymin, ymax)` in metres or a boolean `region` (for example one tracked
cell, see the [cell tracking notebook](Cell_Tracking)); the motion can be given (`motion=(u, v)` or the result of
`estimate_motion`), and `wind=(u, v)` of the mean environmental wind of the sorting
layer returns the angle between the separation vector and the direction opposite to
the storm-relative wind.

## What is not here

- **Learned classification.** Wilson and Van Den Broeke (2021) accept the arc objects
  with a random forest that was trained on hand-labelled candidates; `zdr_arc` takes a
  model of your own. The separation vector of `storm_size_sorting` has no learned step.
- **Numbers of the paper.** The paper shows case studies and distributions of 30 storms, not
  tables that a synthetic case can reproduce. The tests check the definitions on cases with
  known answers.

## References

- Loeffler, S. D., and M. R. Kumjian, 2018: Quantifying the Separation of Enhanced ZDR and KDP Regions in Nonsupercell Tornadic Storms. *Weather and Forecasting*, **33**, 1143-1157, <https://doi.org/10.1175/WAF-D-18-0011.1>
- Wilson, M. B., and M. S. Van Den Broeke, 2021: An Automated Python Algorithm to Quantify ZDR Arc and KDP-ZDR Separation Signatures in Supercells. *Journal of Atmospheric and Oceanic Technology*, **38**, 371-386, <https://doi.org/10.1175/JTECH-D-20-0056.1>
