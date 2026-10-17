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
---

# ZDR and KDP columns in a vertical section

The maps of `zdr_column` and `kdp_column` show where the columns are; a vertical
section shows what they are: a tongue of high $Z_{DR}$ and of positive $K_{DP}$
that reaches above the freezing level, over the heavy rain below. This notebook cuts
one section through a column of a real squall line.

The data are a volume of the KGWX radar (Columbus, Mississippi) of 30 March 2022,
23:46 UTC, from the [open-radar-data](https://github.com/openradar/open-radar-data)
package (version 0.9.0 or newer). The volume is gridded at 1 km by 500 m within a
box of 80 km. The freezing level is not measured here: the 3.7 km used is an
assumption, in the vertical reference of the grid (above sea level), and a sounding
or a model gives the real value.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import xradar as xd
from open_radar_data import DATASETS

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.retrieve import kdp_column, zdr_column

FREEZING_LEVEL = 3700.0  # m, assumed

try:
    path = DATASETS.fetch("KGWX20220330_234639_V06")
except Exception as err:  # offline: the notebook stops after this cell
    path = None
    print("open-radar-data file not reachable:", type(err).__name__)
```

The no-data codes of the file are masked, $K_{DP}$ is estimated from $\Phi_{DP}$ with
`kdp` (this removes the need of a $K_{DP}$ in the file) and the volume goes onto a
Cartesian grid.

```{code-cell} ipython3
if path is not None:
    volume = xd.io.open_nexradlevel2_datatree(path, drop_variables=["WRADH", "CCORH", "CFP"])

    def prepare(ds):
        """Mask the no-data codes and keep the first 100 km."""
        if "range" not in ds.dims:
            return ds
        ds = ds.sel(range=slice(None, 100e3))
        for name, limit in (("DBZH", -32.0), ("ZDR", -12.9), ("RHOHV", 0.21)):
            if name in ds:
                ds[name] = ds[name].where(ds[name] > limit).astype("float32")
        if "PHIDP" in ds:
            ds["PHIDP"] = ds["PHIDP"].astype("float32")
        return ds

    volume = volume.map_over_datasets(prepare)
    volume = volume.radarx.assign(volume.radarx.kdp(phidp="PHIDP", rhohv="RHOHV", dbz="DBZH"))
    grid = volume.radarx.to_grid(
        ["DBZH", "ZDR", "KDP", "RHOHV"],
        x_lim=(-70e3, 10e3),
        y_lim=(-60e3, 20e3),
        z_lim=(500, 12e3),
        x_step=1000,
        y_step=1000,
        z_step=500,
    )
    print(dict(grid.sizes))
```

`min_dbz=30` leaves out weak echo, where the noise of $Z_{DR}$ and $K_{DP}$ makes
shallow columns (an addition of radarx, the papers use calibrated, quality-controlled
data). The section goes through the deepest $Z_{DR}$ column of the box.

```{code-cell} ipython3
if path is not None:
    zdr_depth = zdr_column(
        grid.ZDR, rhohv=grid.RHOHV, dbz=grid.DBZH, min_dbz=30.0, freezing_level=FREEZING_LEVEL
    )
    kdp_cols = kdp_column(grid, freezing_level=FREEZING_LEVEL, rhohv="RHOHV", min_dbz=30.0)
    iy, ix = np.unravel_index(int(np.nanargmax(zdr_depth.values)), zdr_depth.shape)
    x0, y0 = float(grid.x[ix]), float(grid.y[iy])
    print(
        f"deepest ZDR column: {float(zdr_depth[iy, ix]):.0f} m at x={x0 / 1e3:.0f} km, "
        f"y={y0 / 1e3:.0f} km ({np.hypot(x0, y0) / 1e3:.0f} km from the radar); "
        f"KDP column top there: {float(kdp_cols.KDP_COLUMN_TOP[iy, ix]):.0f} m"
    )
```

## The section

The section is along $x$ at the $y$ of the column (the blue line in the plan view; the
red contour encloses the $Z_{DR}$ columns of 1 km or more). The dashed line is the
freezing level, the black contours are the reflectivity (30, 40 and 50 dBZ), and the
white line in each section is the top of the column: the freezing level plus the depth
of the $Z_{DR}$ column, and the top of the $K_{DP}$ column.

```{code-cell} ipython3
if path is not None:
    sec = grid.isel(y=iy)
    x_km, z_km = sec.x / 1e3, sec.z / 1e3
    zdr_top = (FREEZING_LEVEL + zdr_depth.isel(y=iy)) / 1e3
    kdp_top = kdp_cols.KDP_COLUMN_TOP.isel(y=iy) / 1e3

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    pm = axes[0].pcolormesh(
        grid.x / 1e3, grid.y / 1e3, grid.DBZH.sel(z=2000.0, method="nearest"),
        cmap="Greys", vmin=0, vmax=65, shading="auto",
    )
    fig.colorbar(pm, ax=axes[0], label="$Z_H$ at 2 km (dBZ)")
    axes[0].contour(
        grid.x / 1e3, grid.y / 1e3, zdr_depth.where(zdr_depth >= 1000.0).fillna(0) > 0,
        levels=[0.5], colors="tab:red", linewidths=0.8,
    )
    axes[0].axhline(y0 / 1e3, color="tab:blue", lw=1.2)
    axes[0].set(xlabel="x (km)", ylabel="y (km)", aspect="equal",
                title="plan view")

    for ax, var, label, vmax, top, title in (
        (axes[1], "ZDR", "$Z_{DR}$ (dB)", 3.0, zdr_top, "$Z_{DR}$ column"),
        (axes[2], "KDP", "$K_{DP}$ (°/km)", 3.0, kdp_top, "$K_{DP}$ column"),
    ):
        pm = ax.pcolormesh(x_km, z_km, sec[var], cmap="turbo", vmin=-0.5, vmax=vmax, shading="auto")
        fig.colorbar(pm, ax=ax, label=label)
        ax.contour(x_km, z_km, sec.DBZH, levels=[30, 40, 50], colors="k", linewidths=0.6)
        ax.axhline(FREEZING_LEVEL / 1e3, color="w", ls="--", lw=1.2)
        ax.plot(x_km, top.where(top > FREEZING_LEVEL / 1e3), color="w", lw=2.2)
        ax.set(xlim=(x0 / 1e3 - 35, x0 / 1e3 + 35), ylim=(0, 9), xlabel="x (km)",
               ylabel="height above sea level (km)", title=title)
    plt.show()
```

Below the dashed line the heavy rain of the core has $Z_{DR}$ above 3 dB and $K_{DP}$
of 2 to 3 °/km. Above it, in the same place, there is a tongue of $Z_{DR}$ of 1 dB or
more and of positive $K_{DP}$ up to about 6.5 km: raindrops that the updraft has carried
above the freezing level. The two tops agree within about half a kilometre here (the
highest points of the white lines are 6.7 km for $Z_{DR}$ and 6.5 km for $K_{DP}$), and
the $K_{DP}$ column is the smoother of the two, because $K_{DP}$ is estimated over a path.
This is one column of one volume, so it shows the shape and not a statistic. A column is
only as sharp as the vertical resolution of the gridded volume, 500 m here, which is
poor at long range where the beams are wide, and the depth depends directly on the
assumed freezing level.

## References

- Snyder, J. C., A. V. Ryzhkov, M. R. Kumjian, A. P. Khain, and J. Picca, 2015: A ZDR Column Detection Algorithm to Examine Convective Storm Updrafts. *Weather and Forecasting*, **30**, 1819-1844, <https://doi.org/10.1175/WAF-D-15-0068.1>
- van Lier-Walqui, M., A. M. Fridlind, A. S. Ackerman, S. Collis, J. Helmus, D. R. MacGorman, K. North, P. Kollias, and D. J. Posselt, 2016: On Polarimetric Radar Signatures of Deep Convection for Model Evaluation: Columns of Specific Differential Phase Observed during MC3E. *Monthly Weather Review*, **144**, 737-758, <https://doi.org/10.1175/MWR-D-15-0100.1>
