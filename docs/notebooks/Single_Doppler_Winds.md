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

# Single-Doppler Wind Retrieval

+++

One Doppler radar measures only the wind component along its beams. The
other components have to come from prior knowledge.
`radarx.retrieve.single_doppler_winds` offers two kinds of prior:

- **variational** (default): the cost function of
  `radarx.retrieve.multi_doppler` (Gao et al. 1999) with the observation
  term of the one radar, the anelastic mass continuity equation, smoothness
  and a background wind from a sounding or ERA5. Along the beams the wind
  follows the radar; across them it comes from the background and from mass
  continuity.
- **physics-informed network** (`model=...`): a 3-D convolutional network
  trained on multi-Doppler retrievals of NEXRAD radar pairs and on analytic
  flows, with the same variational cost as a physics loss. It runs through
  ONNX Runtime (`pip install radarx[ml]`), and by default its prediction
  becomes the background of the variational retrieval, so that the result
  still fits the observed radial velocities and the continuity equation.
  The training code and the evaluation are in `ml/models/single_doppler`
  of the radarx repository; trained weights are not published yet.

**References**

- Gao, J., M. Xue, A. Shapiro, and K. K. Droegemeier, 1999: A variational
  method for the analysis of three-dimensional wind fields from two Doppler
  radars. *Mon. Wea. Rev.*, **127**, 2128-2142,
  https://doi.org/10.1175/1520-0493(1999)127<2128:AVMFTA>2.0.CO;2
- Shapiro, A., 1993: The use of an exact solution of the Navier-Stokes
  equations in a validation test of a three-dimensional nonhydrostatic
  numerical model. *Mon. Wea. Rev.*, **121**, 2420-2425,
  https://doi.org/10.1175/1520-0493(1993)121<2420:TUOAES>2.0.CO;2

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import radarx as rx
```

## A known flow seen by one virtual radar

A Beltrami flow (Shapiro 1993) with updrafts and downdrafts of 5 m/s on a
sheared background, divided by a density profile so that it satisfies the
anelastic continuity equation, is sampled by one radar 50 km south-west of
the domain centre (1 m/s noise, out to 80 km).

```{code-cell} ipython3
def beltrami(x, y, z, wmax=5.0, lx=40e3, lz=10e3):
    k = l = 2 * np.pi / lx
    m = np.pi / lz
    lam = np.sqrt(k * k + l * l + m * m)
    kh2 = k * k + l * l
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    fu = -wmax / kh2 * (lam * l * np.cos(k * X) * np.sin(l * Y) * np.sin(m * Z)
                        + m * k * np.sin(k * X) * np.cos(l * Y) * np.cos(m * Z))
    fv = wmax / kh2 * (lam * k * np.sin(k * X) * np.cos(l * Y) * np.sin(m * Z)
                       - m * l * np.cos(k * X) * np.sin(l * Y) * np.cos(m * Z))
    fw = wmax * np.cos(k * X) * np.cos(l * Y) * np.sin(m * Z)
    rho = 1.2 * np.exp(-Z / 10e3)
    scale = 1.2 * np.exp(-z.mean() / 10e3) / rho
    u0 = 5.0 + 1.5e-3 * Z  # westerly shear
    v0 = 3.0 + 0.5e-3 * Z
    return u0 + fu * scale, v0 + fv * scale, fw * scale, rho


x = y = np.arange(-40e3, 40e3 + 1, 1000.0)
z = np.arange(0.0, 10e3 + 1, 500.0)
u, v, w, rho = beltrami(x, y, z)
dims = ("z", "y", "x")
coords = {"z": z, "y": y, "x": x}
truth = xr.Dataset({"u": (dims, u), "v": (dims, v), "w": (dims, w)}, coords=coords)

radar = xr.Dataset(
    {"radar_x": -35e3, "radar_y": -35e3, "radar_altitude": 0.0}, coords=coords
)
radar = rx.retrieve.radar_geometry(radar.expand_dims("radar"))
el, az = np.radians(radar.elevation[0]), np.radians(radar.azimuth[0])
vr = np.cos(el) * (np.sin(az) * truth.u + np.cos(az) * truth.v) + np.sin(el) * truth.w
vr = vr + np.random.default_rng(0).normal(0.0, 1.0, vr.shape)
radar["VRADH"] = vr.where(np.hypot(radar.x + 35e3, radar.y + 35e3) < 80e3).expand_dims("radar")

# a background that knows the shear but not the storm (as from ERA5)
height = truth.z.broadcast_like(truth.u).values
background = xr.Dataset(
    {"u": (dims, 5.0 + 1.5e-3 * height), "v": (dims, 3.0 + 0.5e-3 * height), "air_density": (dims, rho)},
    coords=coords,
)

wind = rx.retrieve.single_doppler_winds(radar, background, fall_speed_correction=False)
wind
```

The retrieved wind fits the radar's radial velocities to within the noise.
Its error against the true flow is smaller than that of the background, but
the cross-beam part of the storm circulation stays largely unknown: this is
the fundamental limit of one radar, and what the network is trained to
reduce.

```{code-cell} ipython3
seen = np.isfinite(radar.VRADH[0])
print(f"RMS radial velocity residual: {float(np.sqrt((wind.vr_residual ** 2).mean())):.2f} m/s")
for c in "uvw":
    err = float(np.sqrt(((wind[c] - truth[c]) ** 2).where(seen).mean()))
    err_bg = float(np.sqrt(((background.get(c, 0.0) - truth[c]) ** 2).where(seen).mean()))
    print(f"{c}: RMS error {err:.2f} m/s (background {err_bg:.2f} m/s)")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
level = dict(z=5000.0)
kw = dict(cmap="RdBu_r", vmin=-6, vmax=6, add_colorbar=False)
truth.w.sel(**level).plot(ax=axes[0], **kw)
wind.w.sel(**level).plot(ax=axes[1], **kw)
im = radar.VRADH[0].sel(**level).plot(ax=axes[2], cmap="RdBu_r", vmin=-30, vmax=30, add_colorbar=False)
sub = wind.sel(**level).isel(x=slice(None, None, 6), y=slice(None, None, 6))
axes[1].quiver(sub.x, sub.y, sub.u, sub.v, scale=500)
for ax, title in zip(axes, ["true w at 5 km", "retrieved w and wind", "radial velocity"]):
    ax.set_title(title)
    ax.set_aspect("equal")
fig.colorbar(im, ax=axes[2], label="m/s")
plt.show()
```

## With a trained network

A network exported to ONNX (see `ml/models/single_doppler/README.md`) is
passed as `model`, as a local file or a name of the `radarx.ml` registry:

```python
wind = rx.retrieve.single_doppler_winds(grid, background, model="single_doppler.onnx")
wind.u_network  # the network's own prediction; wind.u is refined by the cost function
```

A radar volume can be given directly with the grid coordinates; it is then
cone-gridded with `radarx.retrieve.multi_doppler_input`:

```python
wind = volume.radarx.single_doppler_winds(background, x=x, y=y, z=z)
```
