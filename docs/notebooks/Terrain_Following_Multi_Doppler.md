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

# Terrain-Following Multi-Doppler Analysis

+++

The vertical velocity of a multi-Doppler analysis comes from the continuity
equation, integrated from a lower boundary where the air cannot go through the
ground. On a grid of constant height that boundary is the lowest level, which
is inside the mountain wherever the ground is higher. `terrain=` in
`radarx.retrieve.multi_doppler` and `radarx.retrieve.dual_doppler_analytic`
moves the levels with the ground: level `k` of a column is at
`terrain + z[k] - z[0]`, so `z[0]` is the ground and `z` of the result is the
height above it. With the vertical velocity relative to these levels,
`W = w - u dh/dx - v dh/dy`, the continuity equation keeps its form, `W = 0` at
the ground is flow along the slope, and `w` is returned. The method is the
terrain-following coordinate of Gal-Chen and Somerville (1975).

This notebook tests it on an exact flow over a Gaussian hill, seen by two
virtual radars, so no data are needed.

**References**

- Gal-Chen, T., and R. C. J. Somerville, 1975: On the use of a coordinate
  transformation for the solution of the Navier-Stokes equations. *J.
  Comput. Phys.*, **17**, 209-228,
  https://doi.org/10.1016/0021-9991(75)90037-6

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import radarx as rx
```

## A flow over a hill

The hill is 2 km high with a scale of 10 km. The wind is `u = U + a x`,
`v = V + a y` at every height, and the vertical velocity
`w = -2 a (z - h) + u dh/dx + v dh/dy` satisfies the continuity equation and
flows along the ground. The radial velocity of the two radars is exact, without
noise, so what is left in the error is the lower boundary.

```{code-cell} ipython3
def hill_flow(h0=2000.0, scale=10e3, step=2000.0, a=2e-4, u0=8.0, v0=4.0):
    x = y = np.arange(-40e3, 40e3 + 1, step)
    z = np.arange(0.0, 12e3 + 1, 1000.0)
    X, Y = np.meshgrid(x, y)
    h = h0 * np.exp(-(X**2 + Y**2) / (2 * scale**2))
    hx, hy = -X / scale**2 * h, -Y / scale**2 * h
    Z = z[:, None, None]
    u, v = u0 + a * X[None] + 0 * Z, v0 + a * Y[None] + 0 * Z
    w = -2 * a * (Z - h[None]) + u * hx[None] + v * hy[None]
    ds = xr.Dataset(
        {"radar_x": ("radar", [-25e3, 25e3]), "radar_y": ("radar", [-20e3, -20e3]),
         "radar_altitude": ("radar", [0.0, 0.0])},
        coords={"radar": [0, 1], "z": z, "y": y, "x": x},
    )
    geo = rx.retrieve.radar_geometry(ds)
    el, az = np.radians(geo.elevation.values), np.radians(geo.azimuth.values)
    vr = np.cos(el) * (np.sin(az) * u + np.cos(az) * v) + np.sin(el) * w
    near = (np.hypot(ds.x - ds.radar_x, ds.y - ds.radar_y) < 60e3).transpose("radar", "y", "x")
    ds["VRADH"] = (("radar", "z", "y", "x"), np.where(near.values[:, None], vr, np.nan))
    dims = ("z", "y", "x")
    bg = xr.Dataset(
        {"u": (dims, np.full(u.shape, u0)), "v": (dims, np.full(u.shape, v0)),
         "air_density": (dims, np.ones(u.shape))},
        coords={"z": z, "y": y, "x": x},
    )
    w_above_ground = -2 * a * Z + (u0 + a * X)[None] * hx[None] + (v0 + a * Y)[None] * hy[None]
    return ds, bg, xr.DataArray(h, dims=("y", "x"), name="terrain"), w, w_above_ground


hill, hill_bg, terrain, w_cartesian, w_following = hill_flow()
```

## The two solvers, with and without the terrain

`w_cartesian` is the truth on levels of constant height, `w_following` the truth
on the terrain-following levels (the second index is the height above the
ground). Each solver runs once with levels of constant height and once with
`terrain=`. Only cells in the lowest 3 km, above the ground, that both radars
see at more than 30 degrees are scored: the analytic solver integrates the
continuity equation upward from the ground, so its error grows with height.

```{code-cell} ipython3
runs = {
    "variational, constant height": rx.retrieve.multi_doppler(hill, hill_bg, fall_speed_correction=False),
    "variational, terrain": rx.retrieve.multi_doppler(hill, hill_bg, terrain=terrain, fall_speed_correction=False),
    "analytic, constant height": rx.retrieve.dual_doppler_analytic(
        hill, hill_bg, fall_speed_correction=False, iterations=4
    ),
    "analytic, terrain": rx.retrieve.dual_doppler_analytic(
        hill, hill_bg, terrain=terrain, fall_speed_correction=False, iterations=4
    ),
}
low = hill.z.values[:, None, None] <= 3000.0
above = (hill.z.values[:, None, None] >= terrain.values[None]) & low
truth_w = {name: (w_following if "terrain" in name else w_cartesian) for name in runs}
masks, errors = {}, {}
for name, result in runs.items():
    good = (result.beam_crossing_angle > 30).values & result.w.notnull().values
    masks[name] = good & (low if "terrain" in name else above)
    errors[name] = result.w.values - truth_w[name]
    rms = np.sqrt(np.mean(errors[name][masks[name]] ** 2))
    print(f"{name:30s} rms error of w: {rms:.3f} m/s")
print(f"rms of the true w in these cells: {np.sqrt(np.mean(w_following[masks['variational, terrain']] ** 2)):.2f} m/s")
```

```{code-cell} ipython3
fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained", sharex=True, sharey=True)
j = hill.sizes["y"] // 2  # the section through the top of the hill
kw = dict(cmap="RdBu_r", vmin=-1.0, vmax=1.0, shading="auto")
order = ["variational, constant height", "variational, terrain", "analytic, constant height", "analytic, terrain"]
for ax, name in zip(axes.ravel(), order):
    result = runs[name]
    height = result.height.isel(y=j) if "terrain" in name else result.z + 0 * result.x
    field = xr.DataArray(errors[name], dims=result.w.dims).where(masks[name]).isel(y=j)
    im = ax.pcolormesh(result.x / 1e3, height / 1e3, field, **kw)
    ax.fill_between(hill.x / 1e3, 0, terrain.isel(y=j) / 1e3, color="0.75")
    ax.set_title(name)
    ax.set_ylim(0, 4.5)
for ax in axes[1]:
    ax.set_xlabel("x (km)")
for ax in axes[:, 0]:
    ax.set_ylabel("height above sea level (km)")
fig.colorbar(im, ax=axes, label="error of w (m/s)", shrink=0.8)
plt.show()
```

The grey area is the hill. With levels of constant height `w = 0` is imposed at the lowest level, which is
inside the hill, and the error spreads up the column and across the hill. On
terrain-following levels the error is that of the numerical derivatives only.
The slopes are centred differences of the terrain on the grid, so terrain that
is much rougher than the grid gives a noisy `w`. The terrain is not used to
mask the beams behind ridges (see `beam_blockage`), and the data are
interpolated linearly in height to the terrain-following levels.
