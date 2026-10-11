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
from radarx.fundamentals.geometry import (
    beam_center_height,
    beam_height_at_ground_range,
    ground_range,
)
from radarx.io.aws_data import download_file
from xradar.io.backends.nexrad_level2 import NEXRADLevel2File
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
kernel, and a DataTree of results (with the root of the volume) is returned.
`dtree.radarx.assign` adds the results to the sweeps of the volume.

```{code-cell} ipython3
start = time.perf_counter()
volume_shear = dtree.radarx.azimuthal_shear("VRADH")
print(f"{len(volume_shear.children)} sweeps in {time.perf_counter() - start:.2f} s")
volume_shear
```

```{code-cell} ipython3
dtree.radarx.assign(volume_shear)["sweep_1"]
```

## The median pre-filter of Smith and Elmore (2004)

Smith and Elmore pass the radial velocities through a 3 x 3 median filter before
the fit, which removes isolated bad gates that would give spurious shear.
`median_filter=3` does this (on the sweep's own grid, the rays wrapping around
north); the default `None` applies none, so the numbers of earlier versions
do not change. The preprint does not say how missing gates enter the median:
here a gate keeps a value only if it is valid itself and its window holds more
than half of its gates. Adding 2 m/s of noise to the vortex above, the filter
lowers the scatter of the shear away from the vortex and keeps the peak:

```{code-cell} ipython3
noisy = vortex.assign(VRADH=vortex.VRADH + np.random.default_rng(0).normal(0.0, 2.0, vortex.VRADH.shape))
away = (rho > 20e3) & (R < 80e3)
for label, option in (("no filter", None), ("3 x 3 median", 3)):
    shear_n = noisy.radarx.azimuthal_shear("VRADH", median_filter=option)
    print(f"{label:13s} scatter away from the vortex {float(shear_n.where(away).std()):.5f} s⁻¹, "
          f"at the centre {float(shear_n[centre]):.4f} s⁻¹ (analytic 0.0100)")
```

## Layer-maximum azimuthal shear over the tilts of a volume

The rotation products of the NSSL Multi-Radar Multi-Sensor system (Smith et al.
2016) are not the shear of one tilt but its **maximum over the tilts of a height
layer**: 0 to 2 km above the ground for near-surface rotation and 3 to 6 km for
mid-level rotation, after the shear of gates that are not in or near a storm cell
(reflectivity of at least 20 dBZ within 5 km) is left out.
`azimuthal_shear_layer_max` (`dtree.radarx.azimuthal_shear_layer_max`) computes
it for a volume on the grid of a reference tilt (by default the lowest). For each
gate of that tilt it takes, for every tilt, the value at the nearest ray and gate
at the same ground range, if the beam centre of the tilt is inside the layer there
(4/3 Earth model), and the largest of them. It also returns the elevation angle
of the tilt that gave each value.

A check with a known answer: a column of solid-body rotation whose angular
velocity changes with height, $\omega(z) = 0.004 + 0.010\,e^{-((z-4500)/1500)^2}$
s⁻¹ (so $0.004$ near the ground and $0.014$ at 4.5 km), 29 km from the radar,
scanned at seven elevation angles. The layer maxima follow the profile:

```{code-cell} ipython3
elevations = (0.5, 1.5, 3.0, 5.0, 7.0, 10.0, 12.0)
column_centre = (25e3, 15e3)


def omega_of_z(z):
    return 0.004 + 0.010 * np.exp(-(((z - 4500.0) / 1500.0) ** 2))


az_c = np.arange(360) + 0.5
rng_c = np.arange(2125.0, 60e3, 250.0)
tilts = {"/": xr.Dataset()}
for n, el in enumerate(elevations):
    s_g = ground_range(rng_c, el)[None, :]
    z_g = beam_center_height(rng_c, el)[None, :]
    phi = np.radians(az_c)[:, None]
    px, py = s_g * np.sin(phi) - column_centre[0], s_g * np.cos(phi) - column_centre[1]
    r_c = np.maximum(np.hypot(px, py), 1e-9)
    om = omega_of_z(z_g)
    vt_c = np.where(r_c < core, om * r_c, om * core**2 / r_c)
    vel = np.cos(np.radians(el)) * (-vt_c * py / r_c * np.sin(phi) + vt_c * px / r_c * np.cos(phi))
    tilts[f"sweep_{n}"] = xr.Dataset(
        {"VRADH": (("azimuth", "range"), vel)},
        coords={"azimuth": az_c, "range": rng_c, "elevation": ("azimuth", np.full(az_c.size, el)),
                "sweep_fixed_angle": el},
    )
column = xr.DataTree.from_dict(tilts)

dist = float(np.hypot(*column_centre))
i = int(np.argmin(np.abs(az_c - np.degrees(np.arctan2(*column_centre)))))
j = int(np.argmin(np.abs(ground_range(rng_c, elevations[0]) - dist)))
for layer in ((0.0, 2000.0), (3000.0, 6000.0)):
    out = column.radarx.azimuthal_shear_layer_max(layer=layer)
    inside = [e for e in elevations if layer[0] <= beam_height_at_ground_range(dist, e) <= layer[1]]
    analytic = max(omega_of_z(beam_height_at_ground_range(dist, e)) * np.cos(np.radians(e)) ** 2 for e in inside)
    print(f"layer {layer[0] / 1e3:.0f}-{layer[1] / 1e3:.0f} km: {float(out.azimuthal_shear_layer_max[i, j]):.5f} s⁻¹ "
          f"from the {float(out.layer_max_elevation[i, j]):.1f}° tilt (analytic {analytic:.5f})")
```

On the Moore volume the velocities of all Doppler cuts are dealiased first
(the Nyquist velocity is read from the radial headers); then the 0 to 2 km layer
maximum is taken with the 20 dBZ test, from the reflectivity of the surveillance
cut of the same elevation angle. The tornado is a compact maximum on both the
lowest tilt and the layer maximum.

```{code-cell} ipython3
with NEXRADLevel2File(str(file)) as nf:
    nyquist = [h["msg_31_data_header"]["RAD"]["nyquist_vel"] / 100.0 for h in nf.msg_31_data_header]
for name, value in zip([n for n in dtree.children if n.startswith("sweep")], nyquist):
    ds = dtree[name].to_dataset().assign_coords(nyquist_velocity=value)
    dtree[name] = ds.assign(VRADH=ds.VRADH.where(ds.VRADH > -64.0)) if "VRADH" in ds else ds
dealiased = dtree.radarx.assign(dtree.radarx.dealias("VRADH"))

start = time.perf_counter()
low_layer = dealiased.radarx.azimuthal_shear_layer_max("VRADH_dealiased", layer=(0.0, 2000.0), min_reflectivity=20.0)
mid_layer = dealiased.radarx.azimuthal_shear_layer_max("VRADH_dealiased", layer=(3000.0, 6000.0), min_reflectivity=20.0)
print(f"both layers in {time.perf_counter() - start:.1f} s")
lowest = dealiased.radarx.azimuthal_shear("VRADH_dealiased")["sweep_1"].to_dataset().azimuthal_shear
print(f"peak shear: 0.5° tilt {float(lowest.max()):.3f}, 0-2 km maximum {float(low_layer.azimuthal_shear_layer_max.max()):.3f}, "
      f"3-6 km maximum {float(mid_layer.azimuthal_shear_layer_max.max()):.3f} s⁻¹")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
panels = [
    (lowest, "0.5° tilt"),
    (low_layer.azimuthal_shear_layer_max, "maximum over 0-2 km"),
    (mid_layer.azimuthal_shear_layer_max, "maximum over 3-6 km"),
]
for ax, (da, title) in zip(axes, panels):
    pm = ax.pcolormesh(sweep.x / 1e3, sweep.y / 1e3, da, cmap="RdBu_r", vmin=-0.02, vmax=0.02)
    fig.colorbar(pm, ax=ax)
    ax.set(xlim=(-30, -10), ylim=(-10, 10), aspect="equal", title=title,
           xlabel="East of KTLX (km)", ylabel="North of KTLX (km)")
fig.suptitle("KTLX 2013-05-20 20:16 UTC, azimuthal shear (s⁻¹)")
```

On the KGWX squall line of 30 March 2022 the 0 to 2 km layer maximum was
compared with the operational MRMS 0 to 2 km product (merged from several
radars, sampled at its 0.005° cells within 20 to 100 km of the radar, four
volumes with two MRMS times each). The pointwise correlation over all cells
with data in both is 0.34 (median of the eight times) for the 0.5° tilt alone and
0.42 for the layer maximum, 0.46 with `median_filter=3`; the ratio of the
shear maxima in 3 km blocks (median over the blocks, radarx over MRMS) rises
from 0.60 to 0.92 with the layer maximum (0.71 with the median filter, which
lowers peaks). The MRMS product is a different, merged field a few minutes away in
time, so the agreement is limited by that as well as by the method.

## References

- Smith, T. M., and K. L. Elmore, 2004: The use of radial velocity derivative to
  diagnose rotation and divergence. *Preprints, 11th Conf. on Aviation, Range,
  and Aerospace Meteorology*, Hyannis, MA, Amer. Meteor. Soc., P5.6,
  <https://ams.confex.com/ams/pdfpapers/81827.pdf>
- Smith, T. M., V. Lakshmanan, G. J. Stumpf, K. L. Ortega, K. Hondl, K. Cooper,
  K. M. Calhoun, D. M. Kingfield, K. L. Manross, R. Toomey, and J. Brogden, 2016:
  Multi-Radar Multi-Sensor (MRMS) severe weather and aviation products: Initial
  operating capabilities. *Bull. Amer. Meteor. Soc.*, **97**, 1617-1630,
  <https://doi.org/10.1175/BAMS-D-14-00173.1>
- Miller, M. L., V. Lakshmanan, and T. M. Smith, 2013: An automated method for
  depicting mesocyclone paths and intensities. *Wea. Forecasting*, **28**,
  570–585, <https://doi.org/10.1175/WAF-D-12-00065.1>
- Mahalik, M. C., B. R. Smith, K. L. Elmore, D. M. Kingfield, K. L. Ortega, and
  T. M. Smith, 2019: Estimates of gradients in radar moments using a linear
  least squares derivative technique. *Wea. Forecasting*, **34**, 415–434,
  <https://doi.org/10.1175/WAF-D-18-0095.1>
