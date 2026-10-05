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

# Azimuthal Shear and Radial Divergence (LLSD)

Azimuthal shear and radial divergence are the standard Doppler diagnostics of
rotation (mesocyclones, tornadoes) and of convergence or divergence (outflow,
downbursts). radarx computes both with the **linear least-squares derivative**
(LLSD) technique (Smith and Elmore 2004; Miller et al. 2013; Mahalik et al.
2019).

At every gate, the radial velocity $v$ of the gates in a window of fixed
physical size is fitted by least squares with the plane

$$
v \approx a + b\,s + c\,\Delta r, \qquad s = r\,\Delta\theta, \quad \Delta r = r - r_0,
$$

where $\Delta\theta$ is the azimuth difference to the centre ray and $r$ the range
of each gate. The slopes are the **azimuthal shear** $b = \partial v/\partial s$
and the **radial divergence** $c = \partial v/\partial r$, both in s⁻¹. For a
solid-body vortex the azimuthal shear is half the vertical vorticity.

The window is set in metres (default 750 m along the beam and 2.5 km across
it), so it holds many rays close to the radar and few far away. The measured
ray azimuths are used, the window wraps around north, and a minimum fraction of
the window must hold valid data. A compiled C++ kernel processes all sweeps of
a volume in one multithreaded call.

```{code-cell} ipython3
import gzip
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xradar as xd

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.io.aws_data import download_file
```

## A synthetic vortex

A Rankine vortex 50 km from the radar: solid-body rotation with
$\omega = 0.01$ s⁻¹ and convergence of $k = 0.002$ s⁻¹ inside a 4 km core, both
decaying with distance outside. In the core the analytic azimuthal shear is
$\omega$ at the centre and the radial divergence is $-k$ everywhere.

```{code-cell} ipython3
omega, k, core = 0.01, 0.002, 4000.0
azimuth = np.arange(0.25, 360.0, 0.5)
rng = np.arange(2125.0, 100e3, 250.0)
A, R = np.meshgrid(np.radians(azimuth), rng, indexing="ij")
x, y = R * np.sin(A), R * np.cos(A)
dx, dy = x - 30e3, y - 40e3
rho = np.maximum(np.hypot(dx, dy), 1e-9)
vt = np.where(rho < core, omega * rho, omega * core**2 / rho)
vc = np.where(rho < core, -k * rho, -k * core**2 / rho)
u = (-vt * dy + vc * dx) / rho
v = (vt * dx + vc * dy) / rho

vortex = xr.Dataset(
    {"VRADH": (("azimuth", "range"), u * np.sin(A) + v * np.cos(A))},
    coords={"azimuth": azimuth, "range": rng, "x": (("azimuth", "range"), x),
            "y": (("azimuth", "range"), y)},
)
result = vortex.radarx.llsd("VRADH")
centre = np.unravel_index(np.argmin(rho), rho.shape)
print("shear at the centre:", float(result.azimuthal_shear[centre]), "(expected 0.01)")
print("divergence at the centre:", float(result.radial_divergence[centre]), "(expected -0.002)")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
panels = [
    (vortex.VRADH, "Radial velocity (m s⁻¹)", "RdBu_r", 40),
    (result.azimuthal_shear, "Azimuthal shear (s⁻¹)", "RdBu_r", 0.012),
    (result.radial_divergence, "Radial divergence (s⁻¹)", "PuOr", 0.004),
]
for ax, (da, title, cmap, lim) in zip(axes, panels):
    pm = ax.pcolormesh(x / 1e3, y / 1e3, da, cmap=cmap, vmin=-lim, vmax=lim)
    fig.colorbar(pm, ax=ax)
    ax.set(xlim=(15, 45), ylim=(25, 55), aspect="equal", title=title,
           xlabel="East (km)", ylabel="North (km)")
```

## The Moore, Oklahoma, tornado (20 May 2013)

The KTLX WSR-88D was about 20 km from the EF5 Moore tornado. We download the
20:16 UTC volume from the NOAA NEXRAD archive on AWS and use the 0.5° velocity
cut.

```{code-cell} ipython3
key = "2013/05/20/KTLX/KTLX20130520_201643_V06.gz"
gz = Path(download_file("unidata-nexrad-level2", key, "downloads"))
file = gz.with_suffix("")
with gzip.open(gz) as src, open(file, "wb") as dst:
    shutil.copyfileobj(src, dst)

dtree = xd.io.open_nexradlevel2_datatree(str(file))
sweep = dtree["sweep_1"].to_dataset(inherit="all_coords").xradar.georeference()
# NEXRAD no-data codes (below threshold, range folded) decode to <= -64 m/s
sweep["VRADH"] = sweep.VRADH.where(sweep.VRADH > -64.0)
print(float(sweep.sweep_fixed_angle), "deg,", sweep.VRADH.shape)
```

LLSD assumes **dealiased** velocities. These were measured with a Nyquist
velocity of about 26 m s⁻¹, and the strongest winds in the tornado fold over;
folds show up as spurious, very large shear values. Dealias the velocity first
when you need the peak values; the location and structure of the couplet are
clear even here.

```{code-cell} ipython3
shear = sweep.radarx.llsd("VRADH", window=(750.0, 2500.0))
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
panels = [
    (sweep.DBZH, "Reflectivity (dBZ)", "turbo", (-10, 70)),
    (sweep.VRADH, "Radial velocity (m s⁻¹)", "RdBu_r", (-26, 26)),
    (shear.azimuthal_shear, "Azimuthal shear (s⁻¹)", "RdBu_r", (-0.02, 0.02)),
]
for ax, (da, title, cmap, (vmin, vmax)) in zip(axes, panels):
    pm = ax.pcolormesh(sweep.x / 1e3, sweep.y / 1e3, da, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(pm, ax=ax)
    ax.set(xlim=(-30, -10), ylim=(-10, 10), aspect="equal", title=title,
           xlabel="East of KTLX (km)", ylabel="North of KTLX (km)")
fig.suptitle("KTLX 2013-05-20 20:16 UTC, 0.5° PPI")
```

The tornado sits about 21 km west of the radar, at the hook echo and debris
ball. Its cyclonic couplet (outbound to the north, inbound to the south) gives
a compact maximum of positive azimuthal shear.

## A whole volume in one call

On a DataTree every sweep with the field is processed in a single call to the
kernel, and a DataTree of results is returned.

```{code-cell} ipython3
start = time.perf_counter()
volume_shear = dtree.radarx.azimuthal_shear("VRADH")
print(f"{len(volume_shear.children)} sweeps in {time.perf_counter() - start:.2f} s")
volume_shear
```

## References

- Smith, T. M., and K. L. Elmore, 2004: The use of radial velocity derivative to
  diagnose rotation and divergence. *Preprints, 11th Conf. on Aviation, Range,
  and Aerospace Meteorology*, Hyannis, MA, Amer. Meteor. Soc., P5.6.
- Miller, M. L., V. Lakshmanan, and T. M. Smith, 2013: An automated method for
  depicting mesocyclone paths and intensities. *Wea. Forecasting*, **28**,
  570–585, <https://doi.org/10.1175/WAF-D-12-00065.1>
- Mahalik, M. C., B. R. Smith, K. L. Elmore, D. M. Kingfield, K. L. Ortega, and
  T. M. Smith, 2019: Estimates of gradients in radar moments using a linear
  least squares derivative technique. *Wea. Forecasting*, **34**, 415–434,
  <https://doi.org/10.1175/WAF-D-18-0095.1>
