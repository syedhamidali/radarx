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

# Diabatic Lagrangian Analysis

The diabatic Lagrangian analysis (DLA) of Ziegler (2013a, b) retrieves the
potential temperature, water vapour and cloud water, and so the buoyancy, of
a storm from a time series of 3-D multi-Doppler winds and radar data:

1. `radarx.retrieve.trajectories` follows the air backward in time from every
   grid point of the analysis time (predictor-corrector, 20-s steps, winds
   interpolated in space and time on a grid moving with the storm) until it
   reaches the storm environment;
2. along each trajectory $\theta$, $q_v$ and $q_c$ start from the environment
   (a sounding from `radarx.io.sounding` or a 3-D mesoscale analysis) and are
   integrated forward with saturation adjustment, rain evaporation, cloud
   collection, graupel melting and sublimation and rain freezing (rates of
   Lin et al. 1983), Lagrangian damping and a surface flux;
3. the end values make the fields at the analysis time.

Rain and graupel along the trajectories come from a precipitation closure;
the default uses the radarx DSD retrieval ($Z_H$, $Z_{DR}$) and the
hydrometeor classification. This notebook builds a small synthetic storm, so
it runs anywhere; with real data the winds come from
`radarx.retrieve.multi_doppler` analyses joined along `time`.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.retrieve import diabatic_lagrangian, trajectories
```

## A synthetic storm

A storm moving with $(c_x, c_y) = (10, 5)$ m s$^{-1}$: an updraft core of
15 m s$^{-1}$ above a precipitation-filled downdraft on its east side, with
reflectivity up to 55 dBZ, analysed every 3 min for 45 min on a 1-km grid.
The environment is a moist boundary layer below 1.5 km under drier air.

```{code-cell} ipython3
cx, cy = 10.0, 5.0
x = np.arange(-20e3, 20e3 + 1, 1000.0)
y = np.arange(-20e3, 20e3 + 1, 1000.0)
z = np.arange(0.0, 8001.0, 500.0)
t = np.arange(16) * 180.0
T, Z, Y, X = np.meshgrid(t, z, y, x, indexing="ij")
xs, ys = X - cx * (T - t[-1]), Y - cy * (T - t[-1])  # storm-relative position


def blob(x0, y0, r):
    return np.exp(-((xs - x0) ** 2 + (ys - y0) ** 2) / r**2)


vert = np.sin(np.pi * Z / 10e3)
w = 15.0 * blob(-3e3, 0.0, 4e3) * vert - 6.0 * blob(5e3, 0.0, 5e3) * np.sin(np.pi * Z / 8e3)
u = cx + 4.0 * blob(5e3, 0.0, 6e3) * (Z < 1500)  # outflow near the ground
v = cy + 0.0 * X
dbz = 10.0 + 45.0 * blob(4e3, 0.0, 7e3) * np.exp(-Z / 9e3)
zdr = np.clip(0.2 + (dbz - 20.0) / 15.0, 0.1, 3.5)
dims = ("time", "z", "y", "x")
times = np.datetime64("2022-03-30T23:00", "ns") + (t * 1e9).astype("timedelta64[ns]")
winds = xr.Dataset(
    {"u": (dims, u), "v": (dims, v), "w": (dims, w), "DBZ": (dims, dbz), "ZDR": (dims, zdr)},
    coords={"time": times, "z": z, "y": y, "x": x},
)

h = np.arange(0.0, 12001.0, 50.0)
temp = 300.0 - 0.0065 * h
sounding = xr.Dataset(
    {
        "pressure": ("height", 1e5 * (temp / 300.0) ** (9.80665 / (287.04 * 0.0065))),
        "temperature": ("height", temp),
        "specific_humidity": ("height", np.where(h < 1500.0, 0.014, 0.005)),
        "u": ("height", np.full(h.size, cx)),
        "v": ("height", np.full(h.size, cy)),
    },
    coords={"height": h},
)
```

## Trajectories

Backward trajectories from the ground at the analysis time. Surface parcels
start 10 m above the ground and, in precipitation downdrafts, feel the
parameterised surface downdraft of Ziegler (2013a), so air in the cold pool
comes from aloft.

```{code-cell} ipython3
tr = trajectories(winds, storm_motion=(cx, cy), levels=[0])
print(f"{float(tr.environment.mean()):.0%} of the trajectories reached the environment")
```

## Diabatic Lagrangian analysis

```{code-cell} ipython3
dla = winds.radarx.diabatic_lagrangian(sounding, storm_motion=(cx, cy))
dla[["delta_theta_v", "qc", "qr", "qg"]]
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
sfc = dla.isel(z=0)
pc = axes[0].pcolormesh(x / 1e3, y / 1e3, sfc.delta_theta_v, cmap="RdBu_r", vmin=-4, vmax=4)
axes[0].contour(x / 1e3, y / 1e3, winds.DBZ.isel(time=-1, z=0), [30, 45], colors="k", linewidths=0.8)
pick = (tr.j % 6 == 0) & (tr.i % 6 == 0) & tr.environment
for k in np.flatnonzero(pick.values)[::2]:
    axes[0].plot(tr.x[k] / 1e3, tr.y[k] / 1e3, color="0.3", lw=0.6)
axes[0].set(xlabel="x (km)", ylabel="y (km)", title="surface, backward trajectories", aspect="equal")
fig.colorbar(pc, ax=axes[0], label=r"$\Delta\theta_v$ (K)")
sec = dla.sel(y=0.0)
pc = axes[1].pcolormesh(x / 1e3, z / 1e3, sec.delta_theta_v, cmap="RdBu_r", vmin=-4, vmax=4)
axes[1].contour(x / 1e3, z / 1e3, sec.qc * 1e3, [0.1, 1.0, 3.0], colors="k", linewidths=0.8)
axes[1].set(xlabel="x (km)", ylabel="height (km)", title="y = 0: cloud water (g/kg) contours")
fig.colorbar(pc, ax=axes[1], label=r"$\Delta\theta_v$ (K)")
plt.show()
```

The cold pool under the downdraft and the buoyant cloudy updraft appear from
the winds and reflectivity alone. The accumulated change of $\theta$ by each
process along the trajectories shows what made the cold pool:

```{code-cell} ipython3
cold = sfc.delta_theta_v < -1.0
names = [k for k in dla if k.startswith("dtheta_")]
budget = [float(sfc[k].where(cold).mean()) for k in names]
fig, ax = plt.subplots(figsize=(6, 3), layout="constrained")
ax.barh([k[7:].replace("_", " ") for k in names], budget, color="tab:blue")
ax.axvline(0, color="k", lw=0.8)
ax.set(xlabel=r"mean $\Delta\theta$ in the surface cold pool (K)")
plt.show()
```

## Sensitivity tests

Every process can be switched off; the acronyms of Ziegler (2013a, Table 3)
select the published tests, e.g. no rain evaporation:

```{code-cell} ipython3
rvap = winds.radarx.diabatic_lagrangian(sounding, storm_motion=(cx, cy), processes="RVAP", levels=[0])
print(
    f"surface cold pool minimum: CNTL {float(sfc.delta_theta_v.min()):.1f} K, "
    f"no rain evaporation {float(rvap.delta_theta_v.min()):.1f} K"
)
```

## Squall lines: time morphing and the boundary rule

Ziegler's termination test ends surface trajectories in a long-lived cold
pool after about 26 min, still inside the outflow. For squall lines radarx
offers the "precipitation" termination (outside echo and ahead of the gust
front, here the `ahead` mask, or above the cold pool), time morphing before
the first analysis (`extend_before`, Ziegler 2013b) so that air older than the
wind series can reach the inflow, a storm motion estimated from the
reflectivity, and a stricter lateral-boundary rule (`boundary`): exits
through edges inside the storm get flag 128 and are not environmental. In a
simulated squall line these options raised the recovered surface cold pool
from about 25 % to about 80 % of the truth (see
`radarx.retrieve.diabatic_lagrangian`, "Applying DLA to observed QLCS
cases").

```{code-cell} ipython3
winds["ahead"] = (dims, xs > 12e3)  # ahead of the gust front
qlcs = dict(
    storm_motion="estimate",
    termination="precipitation",
    environment_mask="ahead",
    parameters={"min_steps": 0, "env_dbz": 15.0},
    levels=[0],
)
for label, extra in [
    ("winds only", {}),
    ("+ 45 min of time morphing", {"extend_before": 2700.0}),
    ("+ boundary exits only outside echo", {"extend_before": 2700.0, "boundary": "no_echo"}),
]:
    tr = trajectories(winds, **qlcs, **extra)
    print(
        f"{label:38s} environment {float(tr.environment.mean()):4.0%}, "
        f"flag 128 {float(((tr.flags & 128) > 0).mean()):4.0%}"
    )
print("estimated storm motion (m/s): {:.1f}, {:.1f}".format(*tr.attrs["storm_motion"]))
```

## References

- Ziegler, C. L., 2013a: A diabatic Lagrangian technique for the analysis of
  convective storms. Part I. *J. Atmos. Oceanic Technol.*, **30**, 2248–2265,
  doi:10.1175/JTECH-D-12-00194.1
- Ziegler, C. L., 2013b: Part II: Application to a radar-observed storm.
  *J. Atmos. Oceanic Technol.*, **30**, 2266–2280,
  doi:10.1175/JTECH-D-13-00036.1
- Lin, Y.-L., R. D. Farley, and H. D. Orville, 1983: Bulk parameterization of
  the snow field in a cloud model. *J. Climate Appl. Meteor.*, **22**,
  1065–1092, doi:10.1175/1520-0450(1983)022<1065:BPOTSF>2.0.CO;2
