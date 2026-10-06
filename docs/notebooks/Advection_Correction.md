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

# Advection Correction of Radar Volumes

+++

A radar volume takes several minutes to collect, and different radars scan at
different times. Storms move in between, so before volumes are merged, compared
or used for multi-Doppler winds, every field has to be moved to a common
analysis time (Gal-Chen 1982). radarx does this in three steps:

1. `radarx.retrieve.estimate_motion` finds the storm motion between two gridded
   volumes by FFT cross-correlation (one vector, or a smooth field with `tile=`).
2. `radarx.retrieve.advect` moves gridded fields along that motion with a
   semi-Lagrangian scheme; missing data move with the field and never spread.
3. `radarx.retrieve.interpolate_time` produces advection-corrected frames between
   two volumes by advecting the earlier one forward and the later one backward.

The interpolation runs in a compiled, multithreaded C++ kernel (with a NumPy
fallback). Here we use three consecutive NEXRAD volumes of a squall line in
Mississippi (KGWX, 30 March 2022).

**References**

- Gal-Chen, T., 1982: Errors in fixed and moving frame of references:
  Applications for conventional and Doppler radar analysis. *J. Atmos. Sci.*,
  **39**, 2279-2300, https://doi.org/10.1175/1520-0469(1982)039<2279:EIFAMF>2.0.CO;2
- Shapiro, A., K. M. Willingham, and C. K. Potvin, 2010: Spatially variable
  advection correction of radar data. Part I: Theoretical considerations.
  *J. Atmos. Sci.*, **67**, 3445-3456, https://doi.org/10.1175/2010JAS3465.1
- Shapiro, A., K. M. Willingham, and C. K. Potvin, 2010: Spatially variable
  advection correction of radar data. Part II: Test results. *J. Atmos. Sci.*,
  **67**, 3457-3470, https://doi.org/10.1175/2010JAS3466.1
- Pulkkinen, S., D. Nerini, A. A. Pérez Hortal, C. Velasco-Forero, A. Seed,
  U. Germann, and L. Foresti, 2019: Pysteps: an open-source Python library for
  probabilistic precipitation nowcasting (v1.0). *Geosci. Model Dev.*, **12**,
  4185-4219, https://doi.org/10.5194/gmd-12-4185-2019

```{code-cell} ipython3
import cmweather  # noqa
import fsspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xradar as xd

import radarx as rx
from radarx.utils import combine_nexrad_sweeps
```

## Grid three consecutive volumes

```{code-cell} ipython3
files = [
    "KGWX20220330_234639_V06",
    "KGWX20220330_235324_V06",
    "KGWX20220330_235959_V06",
]


def gridded_volume(name):
    local_file = fsspec.open_local(
        f"simplecache::s3://unidata-nexrad-level2/2022/03/30/KGWX/{name}",
        s3={"anon": True},
        filecache={"cache_storage": "."},
    )
    dtree = combine_nexrad_sweeps(xd.io.open_nexradlevel2_datatree(local_file))
    return dtree.radarx.to_grid(
        ["DBZH"],
        x_lim=(-150e3, 150e3),
        y_lim=(-150e3, 150e3),
        z_lim=(500, 10e3),
        x_step=1000,
        y_step=1000,
        z_step=500,
    )


g0, g1, g2 = (gridded_volume(name) for name in files)
for g in (g0, g1, g2):
    print(g.time.values, g.DBZH.shape)
```

Each grid holds the mean time of its volume as a scalar `time` coordinate, so
the grids stack into a time series directly:

```{code-cell} ipython3
series = xr.concat([g0, g1, g2], "time")
series.DBZH.sizes
```

## Estimate the storm motion

The motion is estimated on the column maximum (composite) reflectivity. The
quality is the normalised cross-correlation at the peak; a weak or missing
peak gives NaN instead of a wrong motion.

```{code-cell} ipython3
motion = rx.retrieve.estimate_motion(g0, g1)
motion
```

```{code-cell} ipython3
u, v = float(motion.u), float(motion.v)
print(f"storm motion: u = {u:.1f} m/s, v = {v:.1f} m/s, quality {float(motion.quality):.2f}")
```

## Move a volume to another time

`advect` shifts every gridded field (here all 20 levels) to a new time. The
first volume, moved to the time of the second one, lines up with it. The black
contour is 50 dBZ in volume 2 on every panel, and the arrow is the estimated
motion. The empty strip at the southern edge is where no data could come from:
missing data move with the field instead of being filled in.

```{code-cell} ipython3
moved = rx.retrieve.advect(g0, motion, time=g1.time.values)


def composite(ds):
    """Column maximum reflectivity, with x and y in km for plotting."""
    da = ds.DBZH.max("z")
    return da.assign_coords(
        x=("x", da.x.values / 1e3, {"long_name": "east of the radar", "units": "km"}),
        y=("y", da.y.values / 1e3, {"long_name": "north of the radar", "units": "km"}),
    )


kw = dict(cmap="ChaseSpectral", vmin=-10, vmax=70, add_colorbar=False)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, layout="constrained")
panels = [
    (composite(g0), f"volume 1 ({str(g0.time.values)[11:19]} UTC)"),
    (composite(moved), "volume 1 moved to the time of volume 2"),
    (composite(g1), f"volume 2 ({str(g1.time.values)[11:19]} UTC)"),
]
for ax, (da, title) in zip(axes, panels):
    im = da.plot(ax=ax, **kw)
    ax.contour(g1.x / 1e3, g1.y / 1e3, composite(g1).fillna(-99), levels=[50], colors="k", linewidths=0.7)
    ax.set_title(title)
    ax.set_aspect("equal")
axes[0].quiver(110, -135, u, v, color="k", scale=250, width=0.008)
fig.colorbar(im, ax=axes, label="column maximum reflectivity (dBZ)", shrink=0.9)
plt.show()
```

## How much does it help?

Two independent checks with the third volume:

* **Forecast**: move volume 2 to the time of volume 3 with the motion estimated
  from volumes 1 and 2, and compare with what volume 3 observed (persistence:
  assume nothing moved).
* **Interpolation**: estimate volume 2 from volumes 1 and 3 with
  `interpolate_time`, and compare with plain linear interpolation in time.

The scores are the mean absolute difference of the composite and the critical
success index (CSI) of the 30 dBZ area.

```{code-cell} ipython3
def score(predicted, observed, threshold=30.0):
    p, o = composite(predicted).values, composite(observed).values
    both = np.isfinite(p) & np.isfinite(o)
    mae = np.abs(p - o)[both].mean()
    pe, oe = np.nan_to_num(p, nan=-99) >= threshold, np.nan_to_num(o, nan=-99) >= threshold
    return mae, (pe & oe).sum() / (pe | oe).sum()


forecast = rx.retrieve.advect(g1, motion, time=g2.time.values)
motion_13 = rx.retrieve.estimate_motion(g0, g2)
between = rx.retrieve.interpolate_time(g0, g2, g1.time.values, motion=motion_13).isel(time=0)
frac = float((g1.time - g0.time) / (g2.time - g0.time))
linear = g0.assign(DBZH=(1 - frac) * g0.DBZH + frac * g2.DBZH)

rows = {
    "forecast: persistence": score(g1, g2),
    "forecast: advected": score(forecast, g2),
    "interpolation: linear in time": score(linear, g1),
    "interpolation: advection-corrected": score(between, g1),
}
for name, (mae, csi) in rows.items():
    print(f"{name:38s} MAE {mae:5.2f} dB   CSI(30 dBZ) {csi:.3f}")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, layout="constrained")
panels = [
    (composite(linear), "linear in time"),
    (composite(between), "advection-corrected (interpolate_time)"),
    (composite(g1), "observed"),
]
for ax, (da, title) in zip(axes, panels):
    im = da.plot(ax=ax, **kw)
    ax.set_title(title)
    ax.set_aspect("equal")
fig.colorbar(im, ax=axes, label="column maximum reflectivity (dBZ)", shrink=0.9)
fig.suptitle(f"Volume 2 ({str(g1.time.values)[11:19]} UTC) reconstructed from volumes 1 and 3")
plt.show()
```

Linear interpolation in time shows two half-strength copies of the squall line;
the advection-corrected frame keeps one sharp line in the right place.

## Frames at a fixed interval

`interpolate_time` returns all frames at once, e.g. one per minute between
two volumes, ready to be merged with other radars at common analysis times.

```{code-cell} ipython3
times = np.arange(
    g0.time.values.astype("datetime64[m]") + np.timedelta64(1, "m"),
    g1.time.values,
    np.timedelta64(1, "m"),
).astype("datetime64[ns]")
frames = rx.retrieve.interpolate_time(g0, g1, times, motion=motion)
frames
```

```{code-cell} ipython3
grid = composite(frames).plot(
    col="time", col_wrap=3, figsize=(13, 8), cbar_kwargs={"label": "dBZ"}, **dict(kw, add_colorbar=True)
)
for ax in grid.axs.flat:
    ax.set_aspect("equal")
plt.show()
```

## Spatially varying motion

With `tile=` the estimate is repeated on overlapping tiles (here 60 km),
starting from the domain-wide motion, then smoothed and interpolated to every
grid cell. `advect` follows such a motion field along curved trajectories.

```{code-cell} ipython3
motion_field = rx.retrieve.estimate_motion(g0, g1, tile=60e3)
step = 20
sub = motion_field.isel(x=slice(None, None, step), y=slice(None, None, step))
fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
im = composite(g1).plot(ax=ax, **kw)
ax.quiver(sub.x / 1e3, sub.y / 1e3, sub.u, sub.v, color="k", scale=600)
ax.set_aspect("equal")
ax.set_title("Tiled storm motion over volume 2")
fig.colorbar(im, ax=ax, label="column maximum reflectivity (dBZ)")
plt.show()
print(f"u from {float(motion_field.u.min()):.1f} to {float(motion_field.u.max()):.1f} m/s")
print(f"v from {float(motion_field.v.min()):.1f} to {float(motion_field.v.max()):.1f} m/s")
print("forecast with tiled motion: MAE {:.2f} dB, CSI {:.3f}".format(
    *score(rx.retrieve.advect(g1, motion_field, time=g2.time.values), g2)
))
```

## Speed

All levels of a volume (and all target times of `interpolate_time`) go to the
compiled kernel in one call, spread over all cores.

```{code-cell} ipython3
%timeit rx.retrieve.estimate_motion(g0, g1)
%timeit rx.retrieve.advect(g0, motion, dt=300.0)
%timeit rx.retrieve.advect(g0, motion, dt=300.0, method="cubic")
```
