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

# Multi-Doppler Wind Retrieval

+++

A Doppler radar measures only the wind component along its beam. Where two
or more radars look at the same air from different directions, the full
three-dimensional wind can be retrieved. `radarx.retrieve.multi_doppler`
does this with a variational method (Gao et al. 1999): the wind on a
Cartesian grid minimises a cost function made of

- the misfit to every radar's radial velocity, corrected for the fall speed
  of the precipitation (Atlas et al. 1973, with the density correction of
  Foote and du Toit 1969),
- the anelastic mass continuity equation,
- a smoothness penalty,
- the distance to a background wind from a sounding or ERA5,
- optionally the vertical vorticity equation (Shapiro et al. 2009; Potvin et
  al. 2012).

The cost and its exact gradient are computed in one multithreaded pass of a
compiled C++ kernel, and the minimisation (preconditioned conjugate
gradients, first on a coarser grid) takes a few seconds.

The workflow is: dealias each radar (`dealias`), grid all radars onto one
grid and move them to a common time (`multi_doppler_input`, which uses cone
gridding and `advect`), add a background (`radarx.io.sounding`), retrieve.

**References**

- Gao, J., M. Xue, A. Shapiro, and K. K. Droegemeier, 1999: A variational
  method for the analysis of three-dimensional wind fields from two Doppler
  radars. *Mon. Wea. Rev.*, **127**, 2128-2142,
  https://doi.org/10.1175/1520-0493(1999)127<2128:AVMFTA>2.0.CO;2
- Shapiro, A., C. K. Potvin, and J. Gao, 2009: Use of a vertical vorticity
  equation in variational dual-Doppler wind analysis. *J. Atmos. Oceanic
  Technol.*, **26**, 2089-2106, https://doi.org/10.1175/2009JTECHA1256.1
- Potvin, C. K., A. Shapiro, and M. Xue, 2012: Impact of a vertical vorticity
  constraint in variational dual-Doppler wind analysis: Tests with real and
  simulated supercell data. *J. Atmos. Oceanic Technol.*, **29**, 32-49,
  https://doi.org/10.1175/JTECH-D-11-00019.1
- Atlas, D., R. C. Srivastava, and R. S. Sekhon, 1973: Doppler radar
  characteristics of precipitation at vertical incidence. *Rev. Geophys.*,
  **11**, 1-35, https://doi.org/10.1029/RG011i001p00001
- Foote, G. B., and P. S. du Toit, 1969: Terminal velocity of raindrops
  aloft. *J. Appl. Meteor.*, **8**, 249-253,
  https://doi.org/10.1175/1520-0450(1969)008<0249:TVORA>2.0.CO;2
- Shapiro, A., 1993: The use of an exact solution of the Navier-Stokes
  equations in a validation test of a three-dimensional nonhydrostatic
  numerical model. *Mon. Wea. Rev.*, **121**, 2420-2425,
  https://doi.org/10.1175/1520-0493(1993)121<2420:TUOAES>2.0.CO;2

```{code-cell} ipython3
import time

import cmweather  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xradar as xd
from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

import radarx as rx
from radarx.io import sounding
from radarx.io.aws_data import download_file
```

## A known flow seen by two virtual radars

First a test with an exact answer. A Beltrami flow (Shapiro 1993) is an
exact three-dimensional solution of the Navier-Stokes equations with
updrafts and downdrafts of 5 m/s; divided by a density profile it satisfies
the anelastic continuity equation exactly. Two virtual radars 50 km apart
sample its radial velocity (with 1 m/s noise) out to 60 km.

```{code-cell} ipython3
def beltrami(x, y, z, wmax=5.0, lx=40e3, lz=10e3, u0=8.0, v0=4.0):
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
    return u0 + fu * scale, v0 + fv * scale, fw * scale, rho


x = y = np.arange(-40e3, 40e3 + 1, 1000.0)
z = np.arange(0.0, 10e3 + 1, 500.0)
u, v, w, rho = beltrami(x, y, z)
dims = ("z", "y", "x")
truth = xr.Dataset({"u": (dims, u), "v": (dims, v), "w": (dims, w)}, coords={"z": z, "y": y, "x": x})

virtual = xr.Dataset(
    {
        "radar_x": ("radar", [-25e3, 25e3]),
        "radar_y": ("radar", [-20e3, -20e3]),
        "radar_altitude": ("radar", [0.0, 0.0]),
    },
    coords={"radar": [0, 1], "z": z, "y": y, "x": x},
)
virtual = rx.retrieve.radar_geometry(virtual)
el, az = np.radians(virtual.elevation), np.radians(virtual.azimuth)
vr = np.cos(el) * (np.sin(az) * truth.u + np.cos(az) * truth.v) + np.sin(el) * truth.w
vr = vr + np.random.default_rng(0).normal(0.0, 1.0, vr.shape)
in_range = np.hypot(virtual.x - virtual.radar_x, virtual.y - virtual.radar_y) < 60e3
virtual["VRADH"] = vr.where(in_range).transpose("radar", "z", "y", "x")
background = xr.Dataset(
    {"u": (dims, np.full(u.shape, 8.0)), "v": (dims, np.full(u.shape, 4.0)), "air_density": (dims, rho)},
    coords={"z": z, "y": y, "x": x},
)

start = time.perf_counter()
wind = rx.retrieve.multi_doppler(virtual, background, fall_speed_correction=False)
print(f"retrieved {wind.u.size:,} grid cells in {time.perf_counter() - start:.1f} s, "
      f"iterations per grid level (coarse first): {wind.attrs['iterations']}")
lobes = wind.beam_crossing_angle > 30
for c in "uvw":
    err = (wind[c] - truth[c]).where(lobes)
    print(f"{c}: RMS error {float(np.sqrt((err ** 2).mean())):.2f} m/s")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
level = dict(z=5000.0)
kw = dict(cmap="RdBu_r", vmin=-6, vmax=6, add_colorbar=False)
truth.w.sel(**level).plot(ax=axes[0], **kw)
wind.w.sel(**level).where(lobes.sel(**level)).plot(ax=axes[1], **kw)
im = (wind.w - truth.w).sel(**level).where(lobes.sel(**level)).plot(ax=axes[2], **kw)
for ax, title in zip(axes, ["true w at 5 km", "retrieved w (beam crossing > 30 deg)", "retrieved - true"]):
    ax.plot(virtual.radar_x, virtual.radar_y, "k^")
    ax.set_title(title)
    ax.set_aspect("equal")
fig.colorbar(im, ax=axes, label="w (m/s)")
plt.show()
```

The cost terms at every iteration show the coarse-grid and fine-grid
minimisations:

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7, 4))
for term in wind.term.values[:4]:
    ax.semilogy(wind.iteration, wind.cost_history.sel(term=term), label=term)
ax.axvline(int((wind.grid_level > 0).sum()), color="k", lw=0.5)
ax.set_xlabel("iteration (coarse grid, then full grid)")
ax.set_ylabel("cost (m$^2$ s$^{-2}$)")
ax.legend()
plt.show()
```

## A squall line seen by KGWX and KBMX

On 30 March 2022 a squall line crossed Mississippi. At 00 UTC it was just
west of the KGWX radar (Columbus, Mississippi); KBMX (Birmingham, Alabama)
is 166 km to the east-south-east. We dealias both volumes, read the Nyquist
velocity from the radial headers (as in the dealiasing notebook) and mask
the no-data codes.

```{code-cell} ipython3
def nexrad_volume(key):
    path = download_file("unidata-nexrad-level2", f"2022/03/30/{key}", ".")
    with NEXRADLevel2File(path) as nf:
        nyquist = [h["msg_31_data_header"]["RAD"]["nyquist_vel"] / 100.0 for h in nf.msg_31_data_header]
    dtree = xd.io.open_nexradlevel2_datatree(path)
    for i, name in enumerate(n for n in dtree.children if n.startswith("sweep")):
        ds = dtree[name].to_dataset()
        if "DBZH" in ds:
            ds["DBZH"] = ds.DBZH.where(ds.DBZH > -32)
        if "VRADH" in ds:
            ds["VRADH"] = ds.VRADH.where(ds.VRADH > -63.9)
        dtree[name] = ds.assign_coords(nyquist_velocity=nyquist[i])
    return dtree


def dealiased(key):
    """Volume with its radial velocity dealiased in place (as ``VRADH``)."""
    vol = nexrad_volume(key)
    return vol.radarx.assign(vol.radarx.dealias("VRADH", name="VRADH"))


kgwx = dealiased("KGWX/KGWX20220330_235959_V06")
kbmx = dealiased("KBMX/KBMX20220330_235713_V06")
```

The radars scanned at different times, so we estimate the storm motion from
the previous KGWX volume and move both radars to 00:00 UTC while gridding
them onto one 2-km grid centred on KGWX.

```{code-cell} ipython3
previous = nexrad_volume("KGWX/KGWX20220330_235324_V06")
grid_kw = dict(x_lim=(-150e3, 150e3), y_lim=(-150e3, 150e3), z_lim=(1000, 6000), z_step=1000)
motion = rx.retrieve.estimate_motion(previous.radarx.to_grid(["DBZH"], **grid_kw), kgwx.radarx.to_grid(["DBZH"], **grid_kw))
print(f"storm motion u = {float(motion.u):.1f} m/s, v = {float(motion.v):.1f} m/s")

grids = rx.retrieve.multi_doppler_input(
    [kgwx, kbmx],
    x=np.arange(-120e3, 60e3 + 1, 2000.0),
    y=np.arange(-120e3, 100e3 + 1, 2000.0),
    z=np.arange(500.0, 12e3 + 1, 500.0),
    time="2022-03-31T00:00:00",
    motion=motion,
)
grids
```

The background (wind, air density and freezing level) comes from the
Birmingham radiosonde at 00 UTC, spread over the grid. `era5_column` gives
an ERA5 background in the same form.

```{code-cell} ipython3
profile = sounding.read_sounding("BMX", "2022-03-31T00:00")
background = sounding.profile_to_grid(profile, grids)

start = time.perf_counter()
wind = grids.radarx.multi_doppler(background)
print(f"retrieval: {time.perf_counter() - start:.1f} s, iterations {wind.attrs['iterations']}")
print(wind.cost.to_series())
```

The dual-Doppler area is where both radars observe with a beam crossing
angle above 30 degrees. KBMX is far away, so its beams reach the line only
above about 3 km.

```{code-cell} ipython3
good = (wind.beam_crossing_angle > 30) & (wind.n_radars >= 2)
dbz = grids.DBZH.max("radar")
km = dict(x=grids.x / 1e3, y=grids.y / 1e3)

height = 5000.0
level = wind.sel(z=height).where(good.sel(z=height))
sub = level.isel(x=slice(None, None, 3), y=slice(None, None, 3))
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), layout="constrained", sharey=True)
im0 = axes[0].pcolormesh(km["x"], km["y"], dbz.sel(z=height), cmap="ChaseSpectral", vmin=-10, vmax=70)
im1 = axes[1].pcolormesh(km["x"], km["y"], level.w, cmap="RdBu_r", vmin=-8, vmax=8)
axes[1].contour(km["x"], km["y"], dbz.sel(z=height).fillna(-30), levels=[35, 50], colors="k", linewidths=0.6)
for ax, title in zip(axes, ["reflectivity and wind", "w (contours: 35 and 50 dBZ)"]):
    ax.quiver(sub.x / 1e3, sub.y / 1e3, sub.u, sub.v, scale=700, width=0.002)
    ax.plot(grids.radar_x / 1e3, grids.radar_y / 1e3, "k^")
    ax.set_title(f"{height / 1e3:.0f} km: {title}")
    ax.set_xlim(-120, 60)
    ax.set_ylim(-120, 100)
    ax.set_aspect("equal")
    ax.set_xlabel("x (km east of KGWX)")
axes[0].set_ylabel("y (km north of KGWX)")
fig.colorbar(im0, ax=axes[0], label="reflectivity (dBZ)", shrink=0.8)
fig.colorbar(im1, ax=axes[1], label="w (m/s)", shrink=0.8)
plt.show()
```

A vertical cross-section of w across the line:

```{code-cell} ipython3
section = dict(y=-40e3, method="nearest")
fig, ax = plt.subplots(figsize=(10, 4))
im = ax.pcolormesh(wind.x / 1e3, wind.z / 1e3, wind.w.sel(**section).where(good.sel(**section)),
                   cmap="RdBu_r", vmin=-10, vmax=10)
ax.contour(grids.x / 1e3, grids.z / 1e3, dbz.sel(**section).fillna(-30), levels=[20, 40], colors="k", linewidths=0.7)
ax.set_xlabel("x (km)")
ax.set_ylabel("height (km)")
ax.set_title("w across the squall line at y = -40 km (contours: 20 and 40 dBZ)")
fig.colorbar(im, label="w (m/s)")
plt.show()
```

The retrieved wind reproduces both radars' velocities to within the
observation noise:

```{code-cell} ipython3
residual = wind.vr_residual.where(good)
for i, name in enumerate(wind.radar_name.values):
    print(f"{name}: RMS radial velocity residual {float(np.sqrt((residual.isel(radar=i) ** 2).mean())):.2f} m/s")
```
