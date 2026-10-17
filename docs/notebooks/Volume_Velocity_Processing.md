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

# Volume velocity processing: linear winds from one radar

A single Doppler radar measures the radial velocity only. If the wind is a linear
function of position inside a region, the gates of the region determine the wind at its
centre, its gradients, the divergence and the deformation. This is *volume velocity
processing* (VVP, [Waldteufel and Corbin 1979](https://doi.org/10.1175/1520-0450(1979)018<0532:OTAOSD>2.0.CO;2)),
the generalisation of the velocity-azimuth display (VAD, `vad_profile`) from a circle
around the radar to a volume anywhere in the scan. `vvp_winds` follows the formulation of
Doviak and Zrnic (1993, section 9.3). About the centre $(x_0, y_0, z_0)$ of an analysis volume

$$
\mathbf V = \mathbf V_0 + \frac{\partial\mathbf V}{\partial x}(x-x_0) + \frac{\partial\mathbf V}{\partial y}(y-y_0)
 + \frac{\partial\mathbf V}{\partial z}(z-z_0),
$$

and the radial velocity of a gate at the azimuth $\alpha$ and local elevation $\theta$ is linear in
$u_0', u_x, u_z, v_0', v_y, v_z, u_y+v_x, w_0, w_x, w_y, w_z$ (Eqs. 9.11 and 9.13), with the
modified components $u_0' = u_0 + y_0(v_x-u_y)/2$ and $v_0' = v_0 - x_0(v_x-u_y)/2$. The
vorticity $v_x-u_y$ cannot be separated from the wind at the centre. The unknowns are found
by least squares in cylinders around the nodes of a grid, in a compiled multithreaded kernel
(`_vvp.cpp`) with a NumPy implementation that gives the same numbers.

```{code-cell} ipython3
import time

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xradar as xd
from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

import radarx  # noqa: F401
from radarx.fundamentals.constants import EARTH_RADIUS
from radarx.fundamentals.geometry import beam_center_height, ground_range
from open_radar_data import DATASETS
from radarx.retrieve import vad_profile, vvp, vvp_winds
```

## The error of the divergence: the worked example of Doviak and Zrnic

With the covariance of the estimates $s^2 (P^TP)^{-1}$ (Eq. 9.18) the book works out the
standard error of the divergence $u_x+v_y$ of the seven-parameter model for a sector of
30 degrees and 30 km, sweeps at 0.4 and 0.8 degrees, 1 degree and 450 m spacing (about 4000
gates) and a radial velocity error of 1 m/s, and finds $1.6\times10^{-5}$ s$^{-1}$ (the range of the
sector is not given). The predictors of `vvp_winds` give

```{code-cell} ipython3
def sector(r0, r1, azimuth=0.0, width=30.0, elevations=(0.4, 0.8), dr=450.0, daz=1.0):
    reff = 4.0 / 3.0 * EARTH_RADIUS
    az = np.arange(azimuth - width / 2 + daz / 2, azimuth + width / 2, daz)
    rng = np.arange(r0, r1, dr)
    parts = []
    for e in elevations:
        s = ground_range(rng, e, reff=reff)
        z = beam_center_height(rng, e, reff=reff)
        theta = np.radians(e) + s / reff
        a = np.radians(az)[:, None]
        shape = (az.size, rng.size)
        parts.append({
            "x": (s * np.sin(a)).ravel(), "y": (s * np.cos(a)).ravel(),
            "z": np.broadcast_to(z, shape).ravel(), "c": np.broadcast_to(np.cos(theta), shape).ravel(),
            "sn": np.broadcast_to(np.sin(theta), shape).ravel(),
            "sa": np.broadcast_to(np.sin(a), shape).ravel(), "ca": np.broadcast_to(np.cos(a), shape).ravel(),
        })
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def divergence_error(g, sigma=1.0, idx=(0, 1, 2, 3, 4, 5, 6)):
    centre = [g[k].mean() for k in ("x", "y", "z")]
    p = vvp._predictors(g, *centre, 1.0, 1.0)[:, list(idx)]
    cov = np.linalg.inv(p.T @ p)
    a, b = idx.index(1), idx.index(4)
    return sigma * np.sqrt(cov[a, a] + cov[b, b] + 2 * cov[a, b])


for r0 in (20e3, 30e3, 40e3, 50e3, 70e3):
    g = sector(r0, r0 + 30e3)
    print(f"sector {r0 / 1e3:3.0f} to {r0 / 1e3 + 30:3.0f} km: {g['x'].size} gates, "
          f"error of the divergence {divergence_error(g):.2e} 1/s")
```

The error is $1.7\times10^{-5}$ s$^{-1}$ for a sector that starts at 30 km and falls with
range; the value of the book, $1.6\times10^{-5}$, is reached for a sector centred near 49 km
(the book states the number as "on the order of $10^{-5}$"), and 4020 gates are "about 4000".
The kernel returns the same error for its cylinders as `divergence_error` for the same gates
(tested), and a Monte Carlo experiment reproduces it as the scatter of the estimates.

## A strongly sheared low-level jet: KGWX, 30 March 2022

The Columbus (Mississippi) volume from 30 March 2022, 23:46 UTC, ahead of a squall line, has a
low-level jet of more than 35 m/s. The velocity is dealiased first (the Nyquist velocity is read
from the radial headers, as in the dealiasing notebook).

```{code-cell} ipython3
path = DATASETS.fetch("KGWX20220330_234639_V06")
dtree = xd.io.open_nexradlevel2_datatree(path)
with NEXRADLevel2File(path) as nf:
    nyquist = [h["msg_31_data_header"]["RAD"]["nyquist_vel"] / 100.0 for h in nf.msg_31_data_header]
names = [n for n in dtree.children if n.startswith("sweep")]
nodes = {}
for name, vn in zip(names, nyquist):
    ds = dtree[name].to_dataset(inherit="all_coords").assign_coords(nyquist_velocity=vn)
    if "VRADH" in ds:
        ds["VRADH"] = ds["VRADH"].where(ds["VRADH"] > -63.9)  # no-data code
        nodes["/" + name] = ds
volume = xr.DataTree.from_dict({"/": dtree.to_dataset(inherit=False), **nodes})
products = volume.radarx.dealias("VRADH")
for name in volume.children:
    ds = volume[name].to_dataset()
    ds["VRADH"] = products[name]["VRADH_dealiased"]
    volume[name] = xr.DataTree(ds)
altitude = float(volume.altitude)
print(len(volume.children), "velocity sweeps, radar altitude", altitude, "m")
```

### The default call against the VAD

Over the radar the VVP of a volume of the radius $R$ is a VAD of the gates up to $R$, so
`vad_profile` of the same gates (Browning and Wexler 1968: a Fourier fit of every range ring)
is an independent check of the winds. In layers 200 m deep and with the `"uniform"` model
the two agree to hundredths of a m/s; the default model, with the vertical shear, in layers
1 km deep differs by the shear within the layer.

```{code-cell} ipython3
zc = altitude + np.arange(600.0, 5001.0, 200.0)
kw = dict(radius=20000.0, x=[0.0], y=[0.0], z=zc, min_gates=500)
default = vvp_winds(volume, depth=1000.0, **kw).squeeze()
uniform = vvp_winds(volume, depth=200.0, model="uniform", **kw).squeeze()
vad = vad_profile(volume, max_range=20000.0, min_elevation=0.3,
                  height_bins=np.concatenate([zc - 100.0, [zc[-1] + 100.0]]))
for label, vv in (("default, 1 km layers", default), ("uniform, 200 m layers", uniform)):
    du, dv = (vv[k].values - vad[k].values for k in ("u", "v"))
    print(f"{label:22s} rms difference to the VAD: u {np.sqrt(np.nanmean(du**2)):.3f} m/s, v {np.sqrt(np.nanmean(dv**2)):.3f} m/s")
fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True, layout="constrained")
for ax, k in zip(axes, ("u", "v")):
    ax.plot(vad[k], (zc - altitude) / 1e3, "k", label="VAD, 200 m")
    ax.plot(uniform[k], (zc - altitude) / 1e3, "C1--", label="VVP uniform, 200 m")
    ax.plot(default[k], (zc - altitude) / 1e3, "C0", label="VVP linear, 1 km")
    ax.set(xlabel=f"{k} (m/s)")
axes[0].set(ylabel="height above the radar (km)")
axes[0].legend()
plt.show()
```

### A map of the winds and the divergence

The default model on a grid of 10 km spacing at 1.5 km above the radar, in cylinders of 25 km
radius and 2 km depth, within 60 km of the radar. The compiled kernel and the NumPy
implementation give the same fields. The arrows are the modified components $u_0'$, $v_0'$, which
differ from the wind by $y_0\zeta/2$ and $x_0\zeta/2$: away from the radar they include the
vorticity of the flow.

```{code-cell} ipython3
xg = np.arange(-60e3, 60e3 + 1, 10e3)
grid = dict(radius=25000.0, depth=2000.0, x=xg, y=xg, z=[altitude + 1500.0], min_gates=300)
timing = {}
for engine in ("compiled", "numpy"):
    t0 = time.perf_counter()
    result = vvp_winds(volume, engine=engine, **grid)
    timing[engine] = time.perf_counter() - t0
    if engine == "compiled":
        winds = result
print({k: f"{v:.1f} s" for k, v in timing.items()})
print("engines agree:", bool(np.allclose(winds.u, result.u, atol=1e-3, equal_nan=True)
                              and np.allclose(winds.divergence, result.divergence, atol=1e-7, equal_nan=True)))
w = winds.isel(z=0)
fig, ax = plt.subplots(figsize=(6.2, 5.2), layout="constrained")
m = ax.pcolormesh(w.x / 1e3, w.y / 1e3, w.divergence * 1e4, cmap="RdBu_r", vmin=-10, vmax=10, shading="nearest")
ax.quiver(w.x / 1e3, w.y / 1e3, w.u, w.v, scale=400, width=0.004)
ax.set(aspect="equal", xlabel="east (km)", ylabel="north (km)", title="VVP wind and divergence, 1.5 km")
fig.colorbar(m, label="divergence (1e-4 1/s)")
plt.show()
```

## What to keep in mind

- `u` and `v` are the modified components; they are the wind at the centre only where
  $y_0\zeta/2$ is small. A cyclone with a vorticity of $10^{-3}$ s$^{-1}$ at 50 km has a term of
  25 m/s; give `vorticity=` if it is known.
- The standard errors are the random error of the radial velocity only. The bias of a volume in
  which the wind is not linear, or of a model that omits a term that matters (Eq. 9.17), is
  larger for the divergence than the random error in most data; compare volumes of different
  size.
- The vertical velocity enters only as $(w-v_t)\sin\theta$; give `fall_speed=` to remove the
  fall speed of the hydrometeors.

## References

- Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6.
- Browning, K. A., and R. Wexler, 1968: The determination of kinematic properties of a wind field using Doppler radar. *J. Appl. Meteor.*, **7** (1), 105-113, [doi:10.1175/1520-0450(1968)007<0105:TDOKPO>2.0.CO;2](https://doi.org/10.1175/1520-0450(1968)007<0105:TDOKPO>2.0.CO;2).
- Waldteufel, P., and H. Corbin, 1979: On the analysis of single-Doppler radar data. *J. Appl. Meteor.*, **18** (4), 532-542, [doi:10.1175/1520-0450(1979)018<0532:OTAOSD>2.0.CO;2](https://doi.org/10.1175/1520-0450(1979)018<0532:OTAOSD>2.0.CO;2).
