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

# End-to-End Radar Analysis Workflow

+++

This notebook runs the radarx analysis pipeline on one case, step by step and
in the natural order, so that you can see how the pieces fit together and what
each step adds. Every step takes the output of the previous one, and the data
stay in xarray (`DataTree` volumes, `Dataset` sweeps and grids) all the way
through.

**Case:** the squall line that crossed Mississippi on 30 March 2022, seen by
the KGWX WSR-88D (Columbus, Mississippi) in four consecutive volumes from
23:46 to 00:13 UTC.

| Step | What it adds | radarx call |
|---|---|---|
| 1 | volumes, Nyquist velocity, georeferencing | xradar, `download_file` |
| 2 | environment: ERA5 profile and radiosonde, 0 °C height, wind profile | `dtree.radarx.sounding` |
| 3 | dealiased Doppler velocity | `dtree.radarx.dealias` |
| 4 | processed ΦDP and KDP | `dtree.radarx.kdp` |
| 5 | QVP time series and melting layer | `qvp_timeseries`, `melting_layer` |
| 6 | azimuthal shear and radial divergence | `dtree.radarx.llsd` |
| 7 | 3D grid, Max-CAPPI, interactive view, UGRID | `dtree.radarx.to_grid`, `to_uxarray` |
| 8 | storm motion, common analysis time, time interpolation | `estimate_motion`, `advect`, `interpolate_time` |
| 9 | hydrometeor classes, multi-Doppler winds, DSD, evaporation | *added when the features land* |
| 10 | runtime summary and one `DataTree` with all products | |

All heavy steps run in compiled, multithreaded kernels; the runtimes in the
summary table are for this machine with all cores.

```{code-cell} ipython3
import time
import warnings
from contextlib import contextmanager

import cmweather  # noqa: F401  radar colormaps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xradar as xd
from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.io.aws_data import download_file
from radarx.retrieve import (
    advect,
    estimate_motion,
    interpolate_time,
    melting_layer,
    qvp_timeseries,
)
from radarx.utils import combine_nexrad_sweeps

warnings.filterwarnings("ignore", category=RuntimeWarning)

timings = {}


@contextmanager
def timed(step):
    """Record the wall-clock time of a step."""
    start = time.perf_counter()
    yield
    timings[step] = time.perf_counter() - start
    print(f"{step}: {timings[step]:.2f} s")
```


```{note}
This notebook processes several full NEXRAD volumes. It is executed and tested
in the radarx continuous integration on every change; on the documentation
website it is shown without outputs because it exceeds the website's build
resources. Run it locally to see all figures.
```

## 1. Read the volumes

The four volumes come from the NOAA NEXRAD archive on AWS. xradar reads each
file into a `DataTree` with one node per sweep. Two NEXRAD details matter
for what follows:

- the low elevations are scanned twice, a long-range *surveillance* cut
  (reflectivity and polarimetric fields) and a *Doppler* cut with a higher
  Nyquist velocity (velocity); with SAILS, the 0.5° pair is repeated during
  the volume;
- xradar does not yet expose the Nyquist velocity, and decodes the "below
  threshold" and "range folded" codes as the lowest value of each field's
  scale.

So we read the Nyquist velocity from the radial headers and attach it to each
sweep as the `nyquist_velocity` coordinate (where radarx looks for it), mask
the no-data codes, and georeference the volume.

```{code-cell} ipython3
keys = [
    "2022/03/30/KGWX/KGWX20220330_234639_V06",
    "2022/03/30/KGWX/KGWX20220330_235324_V06",
    "2022/03/30/KGWX/KGWX20220330_235959_V06",
    "2022/03/31/KGWX/KGWX20220331_000624_V06",
]
paths = [download_file("unidata-nexrad-level2", key, "data") for key in keys]

# lowest valid value of each field; NEXRAD codes 0 and 1 (below threshold,
# range folded) decode to values below these
VALID_MIN = {"DBZH": -32.0, "ZDR": -12.9, "RHOHV": 0.21, "PHIDP": 0.0, "VRADH": -63.9}


def read_volume(path):
    dtree = xd.io.open_nexradlevel2_datatree(path)
    with NEXRADLevel2File(path) as nf:
        nyquist = [
            h["msg_31_data_header"]["RAD"]["nyquist_vel"] / 100.0
            for h in nf.msg_31_data_header
        ]
    sweeps = [name for name in dtree.children if name.startswith("sweep")]
    for name, value in zip(sweeps, nyquist):
        ds = dtree[name].to_dataset().load()
        for var, lo in VALID_MIN.items():
            if var in ds:
                ds[var] = ds[var].where(ds[var] >= lo)
        dtree[name] = ds.assign_coords(nyquist_velocity=value)
    return dtree.xradar.georeference()


with timed("1. read, Nyquist, georeference"):
    volumes = [read_volume(path) for path in paths]

for vol in volumes:
    print(
        vol.attrs["instrument_name"],
        str(vol.time_coverage_start.values),
        f"{len(vol.match('sweep_*').children)} sweeps",
    )
```

```{code-cell} ipython3
def ppi(ax, ds, var, extent=200, **kwargs):
    """Plot a sweep field in km around the radar."""
    ds = ds.sortby("azimuth")
    pm = ax.pcolormesh(ds.x / 1e3, ds.y / 1e3, ds[var], **kwargs)
    ax.set_aspect("equal")
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_xlabel("east of KGWX (km)")
    return pm


fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), sharey=True, layout="constrained")
for ax, vol in zip(axes, volumes):
    pm = ppi(ax, vol["sweep_0"].to_dataset(), "DBZH", cmap="ChaseSpectral", vmin=-10, vmax=70)
    ax.set_title(f"{str(vol.time_coverage_start.values)[11:16]} UTC, 0.5°")
axes[0].set_ylabel("north of KGWX (km)")
fig.colorbar(pm, ax=axes, label="reflectivity (dBZ)", shrink=0.9)
plt.show()
```

## 2. Environment

Several later steps need the environment around the radar: a wind profile is
the reference for dealiasing (step 3) and the 0 °C height the reference for
the melting layer (step 5). `dtree.radarx.sounding` reads the ERA5 profile at
the radar site (here from Google's anonymous analysis-ready ERA5 store,
`era5_source="gcs"`; the ECMWF ARCO time series and the Copernicus CDS need an
account) or the nearest observed radiosonde. We take ERA5 at 00 UTC, the
middle of the case, and the nearest 00 UTC sounding.

```{note}
ERA5 data are downloaded the first time (one hour of the global field, which
takes a minute or two) and cached afterwards; the timing below is for a warm
cache.
```

```{code-cell} ipython3
from radarx.io import sounding

with timed("2. environment (ERA5, sounding)"):
    era5 = volumes[0].radarx.sounding(era5_source="gcs", time="2022-03-31T00:00")
    raob = volumes[0].radarx.sounding("iem")
    freezing_level = float(sounding.isotherm_height(era5))
    wet_bulb_zero = float(sounding.wet_bulb_zero_height(era5))
print(f"ERA5 at KGWX: 0 °C at {freezing_level:.0f} m, wet-bulb 0 °C at {wet_bulb_zero:.0f} m")
print(
    f"{raob.attrs['station']} sounding: 0 °C at {float(sounding.isotherm_height(raob)):.0f} m"
)
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True, layout="constrained")
for prof, label, color in [(era5, "ERA5 at KGWX", "C0"), (raob, f"{raob.attrs['station']} radiosonde", "C1")]:
    p = prof.where(prof.height < 12e3, drop=True)
    t, td, ws = (p[v].dropna("height") for v in ("temperature", "dewpoint", "wind_speed"))
    axes[0].plot(t - 273.15, t.height / 1e3, color=color, label=label)
    axes[0].plot(td - 273.15, td.height / 1e3, color=color, ls="--")
    axes[1].plot(ws, ws.height / 1e3, color=color)
    axes[2].plot(p.wind_direction.sel(height=ws.height), ws.height / 1e3, ".", color=color, ms=4)
axes[0].axvline(0, color="k", lw=0.8)
axes[0].axhline(freezing_level / 1e3, color="C0", ls=":", label="ERA5 0 °C height")
axes[0].set(xlabel="temperature (solid), dew point (dashed) (°C)", ylabel="height above sea level (km)")
axes[1].set(xlabel="wind speed (m/s)")
axes[2].set(xlabel="wind direction (°)", xlim=(0, 360), xticks=range(0, 361, 90))
axes[0].legend(loc="lower left")
fig.suptitle("Environment, 31 March 2022 00 UTC")
plt.show()
```

## 3. Dealias the Doppler velocity

The Doppler cuts have Nyquist velocities of 26–33 m/s, and the low-level jet
ahead of the line is stronger, so the velocities fold.
`dtree.radarx.dealias` unfolds all sweeps of a volume in one call of the
compiled kernel. Sweeps are processed from the lowest elevation upward, and
the dealiased sweep below fixes the absolute fold of the next one
(`sweep_continuity=True`, the default). The ERA5 wind profile from step 2
(`u` and `v` on `height`) fixes the fold of the lowest sweep and fills gaps.

```{code-cell} ipython3
with timed("3. dealias velocity"):
    volumes = [
        vol.radarx.assign(vol.radarx.dealias("VRADH", wind_profile=era5))
        for vol in volumes
    ]
```

```{code-cell} ipython3
doppler = volumes[0]["sweep_1"].to_dataset()
vn = float(doppler.nyquist_velocity)
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")
for ax, var, title in zip(
    axes, ["VRADH", "VRADH_dealiased"], [f"measured (Nyquist {vn:.1f} m/s)", "dealiased"]
):
    pm = ppi(ax, doppler, var, extent=150, cmap="balance", vmin=-50, vmax=50)
    ax.set_title(title)
axes[0].set_ylabel("north of KGWX (km)")
fig.colorbar(pm, ax=axes, label="radial velocity (m/s)")
fig.suptitle("KGWX 0.5° Doppler cut, 23:46 UTC")
plt.show()
```

## 4. ΦDP processing and KDP

`dtree.radarx.kdp` masks non-meteorological gates, removes the system offset,
filters the phase in range and estimates KDP by least squares, for all rays of
all sweeps in one call. Like every radarx retrieval it returns only its
products (the processed phase as `PHIDP_processed`, so the raw `PHIDP` is
kept); `radarx.assign` adds them to the volumes.

```{code-cell} ipython3
with timed("4. PhiDP processing and KDP"):
    volumes = [
        vol.radarx.assign(vol.radarx.kdp(phidp="PHIDP", rhohv="RHOHV", dbzh="DBZH"))
        for vol in volumes
    ]
```

```{code-cell} ipython3
surv = volumes[0]["sweep_0"].to_dataset()
panels = [
    ("DBZH", "$Z_H$ (dBZ)", dict(cmap="ChaseSpectral", vmin=-10, vmax=70)),
    ("ZDR", "$Z_{DR}$ (dB)", dict(cmap="HomeyerRainbow", vmin=-1, vmax=5)),
    ("PHIDP_processed", "processed $\\Phi_{DP}$ (°)", dict(cmap="viridis", vmin=0, vmax=150)),
    ("KDP", "$K_{DP}$ (°/km)", dict(cmap="HomeyerRainbow", vmin=-0.5, vmax=4)),
]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), sharey=True, layout="constrained")
for ax, (var, label, style) in zip(axes, panels):
    data = surv if var != "PHIDP_processed" else surv.assign(
        PHIDP_processed=surv.PHIDP_processed.where(surv.KDP.notnull())
    )
    pm = ppi(ax, data, var, extent=150, **style)
    fig.colorbar(pm, ax=ax, label=label, shrink=0.85)
axes[0].set_ylabel("north of KGWX (km)")
fig.suptitle("KGWX 0.5° surveillance cut, 23:46 UTC")
plt.show()
```

## 5. QVP time series and melting layer

The quasi-vertical profiles average the highest sweep (19.5°) of every volume
over azimuth and place each gate at its beam height; `melting_layer` finds the
co-located ρhv minimum and ZDR/Z maxima of the bright band. KDP from step 4 is
profiled along with the measured fields. The ERA5 0 °C height from step 2
narrows the search (`freezing_level=`); the melting layer should lie just
below it, since snow melts over a few hundred metres below the 0 °C level.

```{code-cell} ipython3
with timed("5. QVP and melting layer"):
    tqvp = qvp_timeseries(volumes, ["DBZH", "ZDR", "RHOHV", "KDP"], elevation=19.5)
    ml = melting_layer(tqvp, freezing_level=freezing_level)
print(f"ERA5 0 °C height: {freezing_level:.0f} m")
ml.to_dataframe()[["melting_layer_bottom", "melting_layer_peak", "melting_layer_top", "melting_layer_flag"]]
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True, layout="constrained")
colors = plt.cm.viridis(np.linspace(0, 0.9, tqvp.sizes["time"]))
for ax, (var, label) in zip(
    axes,
    [("DBZH", "$Z_H$ (dBZ)"), ("ZDR", "$Z_{DR}$ (dB)"), ("RHOHV", "$\\rho_{hv}$"), ("KDP", "$K_{DP}$ (°/km)")],
):
    for i, color in enumerate(colors):
        ax.plot(tqvp[var].isel(time=i), tqvp.height / 1e3, color=color,
                label=str(tqvp.time.values[i])[11:16] + " UTC")
    ax.axhspan(float(ml.melting_layer_bottom.mean()) / 1e3, float(ml.melting_layer_top.mean()) / 1e3,
               color="0.5", alpha=0.25, label="melting layer")
    ax.axhline(freezing_level / 1e3, color="k", ls=":", label="ERA5 0 °C")
    ax.set_xlabel(label)
for ax, lim in zip(axes, [(-10, 50), (-0.5, 4), (0.8, 1.0), (-0.2, 0.5)]):
    ax.set_xlim(lim)
axes[0].set_ylim(0, 10)
axes[0].set_ylabel("height above sea level (km)")
axes[0].legend(loc="upper right", fontsize=8)
fig.suptitle("QVPs of the 19.5° sweep, KGWX")
plt.show()
```

The bright band (ZDR and Z maxima, ρhv minimum) is found at 2.7–3.0 km, about
0.5 km below the ERA5 0 °C level and 0.4 km below the one of the Birmingham
radiosonde. The QVP circle (within about 10 km of the radar) lies in the rain
just ahead of the line, where melting and evaporation of the precipitation
cool the air and can lower the 0 °C level, while ERA5 and the sounding
describe the larger-scale environment.

## 6. Azimuthal shear and radial divergence

The linear least-squares derivative (LLSD) of the **dealiased** velocity from
step 3 gives the azimuthal shear (rotation) and the radial divergence
(convergence along the gust front) at every gate, for all sweeps in one call.

```{code-cell} ipython3
with timed("6. azimuthal shear and divergence"):
    volumes = [vol.radarx.assign(vol.radarx.llsd("VRADH_dealiased")) for vol in volumes]
```

```{code-cell} ipython3
doppler = volumes[0]["sweep_1"].to_dataset()
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True, layout="constrained")
for ax, (var, label, style) in zip(
    axes,
    [
        ("VRADH_dealiased", "radial velocity (m/s)", dict(cmap="balance", vmin=-50, vmax=50)),
        ("azimuthal_shear", "azimuthal shear (s⁻¹)", dict(cmap="RdBu_r", vmin=-0.01, vmax=0.01)),
        ("radial_divergence", "radial divergence (s⁻¹)", dict(cmap="PuOr", vmin=-0.01, vmax=0.01)),
    ],
):
    pm = ppi(ax, doppler, var, extent=120, **style)
    fig.colorbar(pm, ax=ax, label=label, shrink=0.85)
axes[0].set_ylabel("north of KGWX (km)")
fig.suptitle("LLSD of the dealiased 0.5° velocity, 23:46 UTC")
plt.show()
```

## 7. Cone gridding, Max-CAPPI, interactive view and UGRID

The surveillance and Doppler cuts at each elevation are first merged into one
sweep (`combine_nexrad_sweeps`), so every sweep holds both the polarimetric
fields and the velocity products. `dtree.radarx.to_grid` then grids
reflectivity, ZDR, KDP, dealiased velocity and azimuthal shear of each volume
onto a 300 km × 300 km × 12 km Cartesian grid with the compiled cone
method.

```{code-cell} ipython3
fields = ["DBZH", "ZDR", "KDP", "VRADH_dealiased", "azimuthal_shear"]
with timed("7. merge cuts and grid"):
    grids = [
        combine_nexrad_sweeps(vol).radarx.to_grid(
            fields,
            x_lim=(-150e3, 150e3),
            y_lim=(-150e3, 150e3),
            z_lim=(500, 12e3),
            x_step=1000,
            y_step=1000,
            z_step=500,
        )
        for vol in volumes
    ]
grids[0]
```

The Max-CAPPI shows the column maximum with its north–south and east–west
projections:

```{code-cell} ipython3
grids[0].radarx.plot_max_cappi("DBZH", cmap="ChaseSpectral", vmin=-10, vmax=70, range_rings=True)
```

CAPPIs of the other gridded fields at 2 km:

```{code-cell} ipython3
level = grids[0].sel(z=2000)
fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), sharey=True, layout="constrained")
for ax, (var, label, style) in zip(
    axes,
    [
        ("ZDR", "$Z_{DR}$ (dB)", dict(cmap="HomeyerRainbow", vmin=-1, vmax=5)),
        ("KDP", "$K_{DP}$ (°/km)", dict(cmap="HomeyerRainbow", vmin=-0.5, vmax=4)),
        ("VRADH_dealiased", "radial velocity (m/s)", dict(cmap="balance", vmin=-50, vmax=50)),
        ("azimuthal_shear", "azimuthal shear (s⁻¹)", dict(cmap="RdBu_r", vmin=-0.01, vmax=0.01)),
    ],
):
    pm = ax.pcolormesh(level.x / 1e3, level.y / 1e3, level[var], **style)
    fig.colorbar(pm, ax=ax, label=label, shrink=0.85)
    ax.set_aspect("equal")
    ax.set_xlabel("east of KGWX (km)")
axes[0].set_ylabel("north of KGWX (km)")
fig.suptitle("2 km CAPPIs, 23:46 UTC volume")
plt.show()
```

The same Max-CAPPI as an interactive hvplot view (zoom and pan are linked
between the plan view and the projections):

```{code-cell} ipython3
import holoviews as hv

hv.extension("bokeh")
grids[0].radarx.plot.max_cappi("DBZH", cmap="ChaseSpectral", clim=(-10, 70))
```

A sweep can also be exported as an unstructured (UGRID) mesh with one face
per gate for uxarray:

```{code-cell} ipython3
with timed("7b. UGRID export (one sweep)"):
    uxds = volumes[0]["sweep_1"].to_dataset().radarx.to_uxarray(["VRADH_dealiased", "azimuthal_shear"])
uxds
```

## 8. Storm motion, common analysis time and time interpolation

The four volumes were collected over 27 minutes, during which the line moved
tens of kilometres. `estimate_motion` tracks the reflectivity pattern between
consecutive grids; `advect` moves every gridded field of each volume to a
common analysis time (here 00:00 UTC), and `interpolate_time` produces
advection-corrected frames between two volumes.

```{code-cell} ipython3
analysis_time = np.datetime64("2022-03-31T00:00:00", "ns")
with timed("8. motion, advection, time interpolation"):
    motions = xr.concat(
        [estimate_motion(a, b) for a, b in zip(grids[:-1], grids[1:])], dim="pair"
    )
    motion = motions.mean("pair")
    common = [advect(g, motion, time=analysis_time) for g in grids]
    frame_times = np.arange(
        grids[1].time.values.astype("datetime64[m]") + np.timedelta64(1, "m"),
        grids[2].time.values,
        np.timedelta64(1, "m"),
    ).astype("datetime64[ns]")
    frames = interpolate_time(grids[1], grids[2], frame_times, motion=motion)
motions.to_dataframe()
```

```{code-cell} ipython3
def composite(ds):
    """Column maximum reflectivity, with x and y in km."""
    da = ds.DBZH.max("z")
    return da.assign_coords(x=da.x / 1e3, y=da.y / 1e3)


fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True, layout="constrained")
for ax, data, title in zip(axes, [grids, common], ["as observed", "advected to 00:00 UTC"]):
    ax.pcolormesh(grids[0].x / 1e3, grids[0].y / 1e3, composite(data[1]), cmap="Greys", vmin=0, vmax=70)
    for g, color in zip(data, colors):
        c = composite(g)
        ax.contour(c.x, c.y, c.fillna(-99), levels=[50], colors=[color], linewidths=1.2)
    ax.set_title(f"50 dBZ contours, {title}")
    ax.set_aspect("equal")
    ax.set_xlabel("east of KGWX (km)")
axes[0].set_ylabel("north of KGWX (km)")
handles = [plt.Line2D([], [], color=c) for c in colors]
labels = [str(g.time.values)[11:16] + " UTC" for g in grids]
axes[1].legend(handles, labels, loc="lower right")
plt.show()
```

Moving all volumes to one time brings the 50 dBZ cores of the four volumes
together, most clearly for the discrete cells east of the line. A single
motion vector cannot follow parts of the storm that move differently; a tiled
motion field (`estimate_motion(..., tile=60e3)`) can.

Advection-corrected frames, one per minute between the second and third
volumes:

```{code-cell} ipython3
fg = composite(frames.isel(time=slice(None, None, 2))).plot(
    x="x", y="y", col="time", cmap="ChaseSpectral", vmin=-10, vmax=70,
    figsize=(16, 4.6), cbar_kwargs={"label": "column maximum reflectivity (dBZ)"},
)
for ax in fg.axs.flat:
    ax.set_aspect("equal")
    ax.set_xlabel("east of KGWX (km)")
fg.axs[0, 0].set_ylabel("north of KGWX (km)")
plt.show()
```

## 9. Further steps

+++

```{note}
These sections are *added when the features land*:

- **Hydrometeor classification** (radarx issue #103), from Z, ZDR, KDP and ρhv
  of step 4 and the melting layer of step 5.
- **Multi-radar gridding and multi-Doppler winds** (#99, #100), combining the
  advection-corrected KGWX grid of step 8 with the neighbouring KCBM radar and
  an ERA5 background.
- **Drop size distribution retrieval** (#96).
- **Evaporative cooling** below cloud base (#97).
```

+++

## 10. Summary

The runtime of each step, for all four volumes:

```{code-cell} ipython3
n = len(volumes)
summary = pd.DataFrame(
    {"total (s)": timings, "per volume (s)": {k: v / n for k, v in timings.items()}}
).round(2)
summary.loc["all steps"] = summary.sum()
summary
```

All products of the case in one `DataTree`: the environment profiles,
the polar volumes with every derived field, the QVPs and melting layer, the grids as observed and at the
common analysis time, the storm motion and the interpolated frames.

```{code-cell} ipython3
def stack(datasets):
    return xr.concat(datasets, dim="time")


nodes = {"/": xr.Dataset(attrs={"title": "KGWX 30-31 March 2022 squall line, radarx products"})}
for i, vol in enumerate(volumes):
    for node in vol.subtree:
        nodes[f"/polar/volume_{i}{node.path if node.path != '/' else ''}"] = node.to_dataset(inherit=False)
nodes["/environment/era5"] = era5
nodes["/environment/radiosonde"] = raob
nodes["/qvp"] = tqvp
nodes["/melting_layer"] = ml
nodes["/grid/observed"] = stack(grids)
nodes["/grid/analysis_time"] = stack(common)
nodes["/grid/frames"] = frames
nodes["/motion"] = motions
products = xr.DataTree.from_dict(nodes)
products
```
