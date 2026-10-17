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

# KDP columns

Rain that an updraft carries above the freezing level gives positive specific
differential phase there, so that a KDP column is a signature of the updraft next to
the ZDR column of Snyder et al. (2015) (`zdr_column`, section 17 of the
[third part of the workflow](Radar_Workflow_Advanced_2)). `kdp_column` and
`kdp_column_metrics` give the columns, their tops, the KDP volume and their number
after van Lier-Walqui et al. (2016). A compiled, multithreaded kernel scans the
columns, with a NumPy implementation that gives the same results. The data here are
synthetic, with known answers.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.retrieve import kdp_column, kdp_column_metrics, zdr_column
```

The grid holds two updrafts, a strong one and a weaker one, as columns of
$K_{DP}=1.5$ °/km that begin at the freezing level (4 km) and end at 7 and 6 km.
`kdp_column` integrates the positive KDP in a slab above the freezing level, smooths
the integral, takes each maximum as the core of a column and floods outwards with a
watershed. Each column gets the height of the highest level with
$K_{DP}\ge 0.75$ °/km as its top.

```{code-cell} ipython3
z = np.arange(0.0, 12001.0, 250.0)
y = x = np.arange(41) * 1000.0
r1 = np.hypot(y[:, None] - 12e3, x[None, :] - 12e3)
r2 = np.hypot(y[:, None] - 30e3, x[None, :] - 28e3)
tops = np.maximum(
    4000 + 3000 * np.clip(1 - r1 / 6e3, 0, None), 4000 + 2000 * np.clip(1 - r2 / 5e3, 0, None)
)
inside = (z[:, None, None] >= 4000) & (z[:, None, None] < tops[None])
rng = np.random.default_rng(1)
kdp = xr.DataArray(
    np.where(inside, 1.5, 0.0) + rng.normal(0, 0.15, inside.shape),
    dims=("z", "y", "x"),
    coords={"z": z, "y": y, "x": x},
    name="KDP",
)
zdr = xr.DataArray(
    np.where(inside, 2.0, 0.3) + rng.normal(0, 0.2, inside.shape),
    dims=("z", "y", "x"),
    coords={"z": z, "y": y, "x": x},
    name="ZDR",
)

columns = kdp_column(kdp, freezing_level=4000.0)
metrics = kdp_column_metrics(columns)
zcol = zdr_column(zdr, freezing_level=4000.0)
print(
    f"{int(metrics.KDP_COLUMN_COUNT)} KDP columns, highest top {float(metrics.KDP_COLUMN_MAX_TOP):.0f} m, "
    f"area {float(metrics.KDP_COLUMN_AREA):.0f} km2, KDP volume {float(metrics.KDP_VOLUME):.0f} km3"
)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
pm = axes[0].pcolormesh(x / 1e3, y / 1e3, columns.KDP_SLAB_INTEGRAL, cmap="magma", shading="auto")
fig.colorbar(pm, ax=axes[0], label="KDP integral in the slab (°)")
pm = axes[1].pcolormesh(x / 1e3, y / 1e3, columns.KDP_COLUMN_OBJECT, cmap="tab10", vmin=0, vmax=9, shading="auto")
fig.colorbar(pm, ax=axes[1], label="KDP column number")
pm = axes[2].pcolormesh(x / 1e3, y / 1e3, columns.KDP_COLUMN_TOP, cmap="viridis", vmin=4000, vmax=7500, shading="auto")
fig.colorbar(pm, ax=axes[2], label="KDP column top (m)")
axes[2].contour(x / 1e3, y / 1e3, zcol, levels=[1000.0, 2000.0], colors="w", linewidths=0.8)
for ax, title in zip(axes, ("slab integral", "columns", "tops, ZDR column depth at 1 and 2 km")):
    ax.set(xlabel="x (km)", ylabel="y (km)", aspect="equal", title=title)
plt.show()
```

With the freezing level as a field (a sounding or a model) the function takes a
height per grid column. The threshold is 0.75 °/km for S band, 1.5 for C band
(`threshold=`); `rhohv=` and `min_rhohv=` remove grid points of low correlation
coefficient. The paper gives neither the width of the Gaussian filter nor the extent
of a column, so `smooth` (in grid cells) and `min_depth` are choices that you can change.

## What is not here

- **ZDR column rules of the paper.** van Lier-Walqui et al. measure the ZDR height with a
  negative vertical gradient and below the homogeneous freezing level; `zdr_column` uses
  the continuous depth of Snyder et al. (2015).
- **Numbers of the paper.** The paper shows case studies, not tables that a synthetic case
  can reproduce. The tests check the definitions on cases with known answers.

## References

- Snyder, J. C., A. V. Ryzhkov, M. R. Kumjian, A. P. Khain, and J. Picca, 2015: A ZDR Column Detection Algorithm to Examine Convective Storm Updrafts. *Weather and Forecasting*, **30**, 1819-1844, <https://doi.org/10.1175/WAF-D-15-0068.1>
- van Lier-Walqui, M., A. M. Fridlind, A. S. Ackerman, S. Collis, J. Helmus, D. R. MacGorman, K. North, P. Kollias, and D. J. Posselt, 2016: On Polarimetric Radar Signatures of Deep Convection for Model Evaluation: Columns of Specific Differential Phase Observed during MC3E. *Monthly Weather Review*, **144**, 737-758, <https://doi.org/10.1175/MWR-D-15-0100.1>
