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
the KGWX WSR-88D (Columbus, Mississippi) in two consecutive volumes, 23:53
and 00:00 UTC, and by the KBMX WSR-88D (Birmingham, Alabama) for the
multi-radar steps.

| Step | What it adds | radarx call |
|---|---|---|
| 1 | volumes, Nyquist velocity, georeferencing | xradar, `download_file` |
| 2 | non-meteorological echo removed | `dtree.radarx.echo_mask`, `apply_mask` |
| 3 | environment: ERA5 profile and radiosonde, 0 °C height, wind profile | `dtree.radarx.sounding` |
| 4 | dealiased Doppler velocity | `dtree.radarx.dealias` |
| 5 | processed ΦDP and KDP | `dtree.radarx.kdp` |
| 6 | hydrometeor classes | `dtree.radarx.hid` |
| 7 | rain drop size distribution and rain rate | `ds.radarx.dsd` |
| 8 | QVP time series and melting layer | `qvp_timeseries`, `melting_layer` |
| 9 | azimuthal shear and radial divergence | `dtree.radarx.llsd` |
| 10 | 3D grid, Max-CAPPI, interactive view, UGRID | `dtree.radarx.to_grid`, `to_uxarray` |
| 11 | storm motion, common analysis time, time interpolation | `estimate_motion`, `advect`, `interpolate_time` |
| 12 | multi-radar grid and three-dimensional wind | `grid_radars`, `multi_doppler` |
| 13 | evaporation | *added when the feature lands* |
| 14 | runtime summary and one `DataTree` with all products | |

All heavy steps run in compiled, multithreaded kernels; the runtimes in the
summary table are for the machine that built this page, with all cores.

```{code-cell} ipython3
import gc
import time
import warnings
from contextlib import contextmanager

import cmweather  # noqa: F401  radar colormaps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xradar as xd
from matplotlib.colors import BoundaryNorm, ListedColormap
from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.grid import grid_radars
from radarx.io import sounding
from radarx.io.aws_data import download_file
from radarx.retrieve import (
    advect,
    estimate_motion,
    hid_classes,
    interpolate_time,
    melting_layer,
    multi_doppler_input,
    qvp_timeseries,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="The input coordinates to pcolormesh")

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
This notebook downloads and processes three NEXRAD volumes (about 2 GB of
memory and a few minutes). It is executed and tested in the radarx continuous
integration on every change, and the documentation website shows the outputs
of that run.
```

## 1. Read the volumes

The volumes come from the NOAA NEXRAD archive on AWS. xradar reads each file
into a `DataTree` with one node per sweep. Three NEXRAD details matter for
what follows:

- the low elevations are scanned twice, a long-range *surveillance* cut
  (reflectivity and polarimetric fields) and a *Doppler* cut with a higher
  Nyquist velocity (velocity);
- with SAILS, the 0.5° pair is repeated during the volume. The repeats are
  useful for nowcasting but not here, so we read only the first one
  (`sweep=`), together with the fields we use (`drop_variables=` drops the
  spectrum width and the clutter correction);
- the surveillance cuts reach 460 km, far beyond what we analyse, so we keep
  the first 200 km (`max_range`), and the moments, which are 8- and 16-bit
  codes, and the gate positions in single precision. Together this cuts the
  memory of a volume by a factor of five;
- xradar does not yet expose the Nyquist velocity. We read it from the radial
  headers and attach it to each sweep as the `nyquist_velocity` coordinate,
  where radarx looks for it.

The no-data codes are masked in the next step.

```{code-cell} ipython3
keys = [
    "2022/03/30/KGWX/KGWX20220330_235324_V06",
    "2022/03/30/KGWX/KGWX20220330_235959_V06",
]
paths = [download_file("unidata-nexrad-level2", key, "data") for key in keys]


def read_volume(path, fields=("DBZH", "ZDR", "PHIDP", "RHOHV", "VRADH"), max_range=200e3):
    """Read the fields of a NEXRAD volume without the SAILS repeats."""
    with NEXRADLevel2File(path) as nf:
        nyquist = [
            h["msg_31_data_header"]["RAD"]["nyquist_vel"] / 100.0
            for h in nf.msg_31_data_header
        ]
        # fixed angles of the scan strategy; AVSET may end the volume early
        angles = [cut["elevation_angle"] for cut in nf.msg_5["elevation_data"]]
        angles = angles[: len(nyquist)]
    # keep the first cut (or surveillance/Doppler pair) of every elevation
    keep = []
    for i, angle in enumerate(angles):
        if angle not in angles[:i] or (i - 1 in keep and angles[i - 1] == angle):
            keep.append(i)
    drop = ["WRADH", "CCORH", "CFP"] + [
        v for v in ("DBZH", "ZDR", "PHIDP", "RHOHV", "VRADH") if v not in fields
    ]
    dtree = xd.io.open_nexradlevel2_datatree(path, sweep=keep, drop_variables=drop)
    for name in [n for n in dtree.children if n.startswith("sweep")]:
        ds = dtree[name].to_dataset().sel(range=slice(None, max_range))
        ds = ds.assign({v: ds[v].astype("float32") for v in fields if v in ds})
        index = int(ds.sweep_number)
        dtree[name] = ds.load().assign_coords(nyquist_velocity=nyquist[index])
    # gate positions (m) in single precision too
    return dtree.xradar.georeference().map_over_datasets(
        lambda ds: ds.assign_coords({c: ds[c].astype("float32") for c in "xyz" if c in ds.coords})
    )


with timed("1. read, Nyquist, georeference"):
    raw = [read_volume(path) for path in paths]

for vol in raw:
    sweeps = vol.match("sweep_*").children
    print(
        vol.attrs["instrument_name"],
        str(vol.time_coverage_start.values),
        f"{len(sweeps)} sweeps:",
        " ".join(f"{float(vol[n].sweep_fixed_angle):.1f}" for n in sweeps),
    )
```

## 2. Remove non-meteorological echo

Ahead of the line, insects and birds fill the boundary layer with weak echo
of high $Z_{DR}$ and low $\rho_{hv}$, and ground clutter surrounds the radar.
`dtree.radarx.echo_mask` classifies every gate of a volume with a fuzzy-logic
score (Gourley et al. 2007; Krause 2016); the Doppler cuts, which have no
polarimetric fields, take the class of the surveillance cut at the same
elevation. `dtree.radarx.apply_mask` then sets non-meteorological gates and
the NEXRAD no-data codes to NaN in every field. Everything downstream works on
the cleaned volumes.

```{code-cell} ipython3
with timed("2. echo classification and masking"):
    echo = [vol.radarx.echo_mask() for vol in raw]
    volumes = [vol.radarx.apply_mask(qc) for vol, qc in zip(raw, echo)]
```

```{code-cell} ipython3
def ppi(ax, ds, var, extent=200, **kwargs):
    """Plot a sweep field in km around the radar."""
    rays = np.argsort(ds.azimuth.values)  # no seam where the scan started
    pm = ax.pcolormesh(
        ds.x.values[rays] / 1e3, ds.y.values[rays] / 1e3, ds[var].values[rays], **kwargs
    )
    ax.set_aspect("equal")
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_xlabel("east of KGWX (km)")
    return pm


before = raw[1]["sweep_0"].to_dataset()
before = before.assign(DBZH=before.DBZH.where(before.DBZH > -32))
classes = echo[1]["sweep_0"].to_dataset().ECHO_CLASS
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True, layout="constrained")
pm = ppi(axes[0], before, "DBZH", cmap="ChaseSpectral", vmin=-10, vmax=70)
axes[0].set_title("$Z_H$ as measured")
pc = ppi(
    axes[1],
    classes.where(classes > 0).to_dataset(),
    "ECHO_CLASS",
    cmap=ListedColormap(["#2a78d6", "#eb6834", "#eda100"]),
    vmin=0.5,
    vmax=3.5,
)
axes[1].set_title("echo class")
ppi(axes[2], volumes[1]["sweep_0"].to_dataset(), "DBZH", cmap="ChaseSpectral", vmin=-10, vmax=70)
axes[2].set_title("$Z_H$ after apply_mask")
axes[0].set_ylabel("north of KGWX (km)")
for ax in axes[[0, 2]]:
    fig.colorbar(pm, ax=ax, label="reflectivity (dBZ)", shrink=0.85)
cb = fig.colorbar(pc, ax=axes[1], ticks=[1, 2, 3], shrink=0.85)
cb.ax.set_yticklabels(["meteorological", "non-met.", "speckle"])
fig.suptitle("KGWX 0.5°, 00:00 UTC")
plt.show()

# the raw volumes and the classification are not needed any more
del raw, echo, vol, before, classes
gc.collect();
```

## 3. Environment

Several later steps need the environment around the radar: a wind profile is
the reference for dealiasing (step 4), the temperature at every gate is an
input of the hydrometeor classification (step 6) and the 0 °C height the
reference for the melting layer (step 8). `dtree.radarx.sounding` reads the
ERA5 profile at the radar site (here from Google's anonymous analysis-ready
ERA5 store, `era5_source="gcs"`; the ECMWF ARCO time series and the Copernicus
CDS need an account) or the nearest observed radiosonde. We take ERA5 at
00 UTC and the 00 UTC Birmingham sounding.

```{note}
ERA5 data are downloaded the first time (one hour of the global field, which
takes a minute or two) and cached afterwards; the timing below is for a warm
cache.
```

```{code-cell} ipython3
with timed("3. environment (ERA5, sounding)"):
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

## 4. Dealias the Doppler velocity

The Doppler cuts have Nyquist velocities of 26–33 m/s, and the low-level jet
ahead of the line is stronger, so the velocities fold.
`dtree.radarx.dealias` unfolds all sweeps of a volume in one call of the
compiled kernel. Sweeps are processed from the lowest elevation upward, and
the dealiased sweep below fixes the absolute fold of the next one
(`sweep_continuity=True`, the default). The ERA5 wind profile from step 3
(`u` and `v` on `height`) fixes the fold of the lowest sweep and fills gaps.
Like every radarx retrieval it returns only its products; `radarx.assign`
adds them to the volumes.

```{code-cell} ipython3
with timed("4. dealias velocity"):
    volumes = [
        vol.radarx.assign(vol.radarx.dealias("VRADH", wind_profile=era5))
        for vol in volumes
    ]
```

```{code-cell} ipython3
doppler = volumes[1]["sweep_1"].to_dataset()
vn = float(doppler.nyquist_velocity)
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")
for ax, var, title in zip(
    axes, ["VRADH", "VRADH_dealiased"], [f"measured (Nyquist {vn:.1f} m/s)", "dealiased"]
):
    pm = ppi(ax, doppler, var, extent=150, cmap="balance", vmin=-50, vmax=50)
    ax.set_title(title)
axes[0].set_ylabel("north of KGWX (km)")
fig.colorbar(pm, ax=axes, label="radial velocity (m/s)")
fig.suptitle("KGWX 0.5° Doppler cut, 00:00 UTC")
plt.show()
```

## 5. ΦDP processing and KDP

`dtree.radarx.kdp` removes the system offset, filters the phase in range and
estimates KDP by least squares, for all rays of all sweeps in one call. The
processed phase is returned as `PHIDP_processed`, so the raw `PHIDP` is kept.

```{code-cell} ipython3
with timed("5. PhiDP processing and KDP"):
    volumes = [
        vol.radarx.assign(vol.radarx.kdp(phidp="PHIDP", rhohv="RHOHV", dbzh="DBZH"))
        for vol in volumes
    ]
```

```{code-cell} ipython3
surv = volumes[1]["sweep_0"].to_dataset()
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
fig.suptitle("KGWX 0.5° surveillance cut, 00:00 UTC")
plt.show()
```

## 6. Hydrometeor classification

`dtree.radarx.hid` assigns a hydrometeor class to every gate with the fuzzy
logic of Park et al. (2009) for S band, from $Z_H$, $Z_{DR}$, the KDP of
step 5 and $\rho_{hv}$. The ERA5 temperature profile of step 3 is
interpolated to the gate heights and places the melting layer, below which
only liquid and mixed-phase classes are allowed. The non-meteorological
gates removed in step 2 stay unclassified (class 0).

```{code-cell} ipython3
with timed("6. hydrometeor classification"):
    volumes = [
        vol.radarx.assign(vol.radarx.hid(era5, band="S", scores=False))
        for vol in volumes
    ]
```

```{code-cell} ipython3
park = hid_classes("park", "S")
hid_cmap = ListedColormap(
    ["#2a78d6", "#e87ba4", "#1baf7a", "#eda100", "#4a3aa7", "#008300", "#eb6834", "#e34948"]
)
hid_norm = BoundaryNorm(np.arange(0.5, len(park) + 1.5), hid_cmap.N)
surv = volumes[1]["sweep_0"].to_dataset()
upper = volumes[1]["sweep_9"].to_dataset()  # 2.4°
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True, layout="constrained")
for ax, ds in zip(axes, [surv, upper]):
    pm = ppi(ax, ds.assign(HID=ds.HID.where(ds.HID > 0)), "HID", extent=150, cmap=hid_cmap, norm=hid_norm)
    ax.set_title(f"{float(ds.sweep_fixed_angle):.1f}°")
cb = fig.colorbar(pm, ax=axes, ticks=np.arange(1, len(park) + 1), shrink=0.9)
cb.ax.set_yticklabels([name.replace("_", " ") for _, _, name in park])
axes[0].set_ylabel("north of KGWX (km)")
fig.suptitle("Hydrometeor classes, 00:00 UTC")
plt.show()
```

Rain fills the lowest sweep within about 120 km, with heavy rain along the
leading edge of the line. Farther out, and beyond 50 km on the 2.4° sweep,
the beam reaches the melting layer (wet snow, graupel) and the snow and ice
above it.

+++

## 7. Rain drop size distribution

`ds.radarx.dsd` retrieves the gamma drop size distribution of rain at every
gate from $Z_H$ and $Z_{DR}$, with the constrained-gamma method (Zhang et al.
2001; Cao et al. 2008), and takes the intercept from KDP where it is at
least 1 °/km. The hydrometeor classes of step 6 select the rain gates
(light and moderate rain, heavy rain, big drops); hail, the melting layer
and ice would give meaningless drop sizes. We retrieve the DSD on the
lowest sweep, whose rain is the one that reaches the ground.

```{code-cell} ipython3
rain_codes = [code for code, _, name in park if name in ("big_drops", "light_and_moderate_rain", "heavy_rain")]
with timed("7. DSD retrieval (0.5°)"):
    surv = volumes[1]["sweep_0"].to_dataset()
    rain = surv.HID.isin(rain_codes)
    dsd = surv.radarx.dsd(kdp="KDP", mask=rain, band="S")
dsd
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True, layout="constrained")
for ax, (da, label, style) in zip(
    axes,
    [
        (dsd.D0, "median volume diameter $D_0$ (mm)", dict(cmap="plasma", vmin=0.5, vmax=3)),
        (np.log10(dsd.NW), "$\\log_{10} N_w$ (m$^{-3}$ mm$^{-1}$)", dict(cmap="viridis", vmin=2, vmax=5)),
        (dsd.RAIN_RATE, "rain rate (mm/h)", dict(cmap="turbo", vmin=0, vmax=60)),
    ],
):
    pm = ppi(ax, surv.assign(field=da), "field", extent=150, **style)
    fig.colorbar(pm, ax=ax, label=label, shrink=0.85)
axes[0].set_ylabel("north of KGWX (km)")
fig.suptitle("DSD retrieval on the rain gates, 0.5°, 00:00 UTC")
plt.show()
```

The convective line has large drops ($D_0$ above 2 mm) and the highest rain
rates; the trailing stratiform rain has smaller drops.

+++

## 8. QVP time series and melting layer

The quasi-vertical profiles average the highest sweep (19.5°) of every volume
over azimuth and place each gate at its beam height; `melting_layer` finds the
co-located ρhv minimum and ZDR/Z maxima of the bright band. KDP from step 5 is
profiled along with the measured fields. The ERA5 0 °C height from step 3
narrows the search (`freezing_level=`); the melting layer should lie just
below it, since snow melts over a few hundred metres below the 0 °C level.

```{code-cell} ipython3
with timed("8. QVP and melting layer"):
    tqvp = qvp_timeseries(volumes, ["DBZH", "ZDR", "RHOHV", "KDP"], elevation=19.5)
    ml = melting_layer(tqvp, freezing_level=freezing_level)
print(f"ERA5 0 °C height: {freezing_level:.0f} m")
ml.to_dataframe()[["melting_layer_bottom", "melting_layer_peak", "melting_layer_top", "melting_layer_flag"]]
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True, layout="constrained")
colors = ["C0", "C3"]
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

The bright band (ZDR and Z maxima, ρhv minimum) is found at about 3 km,
roughly 0.5 km below the ERA5 0 °C level. The QVP circle (within about 10 km
of the radar) lies in the rain just ahead of the line, where melting and
evaporation of the precipitation cool the air and can lower the 0 °C level,
while ERA5 and the sounding describe the larger-scale environment.

+++

## 9. Azimuthal shear and radial divergence

The linear least-squares derivative (LLSD) of the **dealiased** velocity from
step 4 gives the azimuthal shear (rotation) and the radial divergence
(convergence along the gust front) at every gate, for all sweeps in one call.

```{code-cell} ipython3
with timed("9. azimuthal shear and divergence"):
    volumes = [vol.radarx.assign(vol.radarx.llsd("VRADH_dealiased")) for vol in volumes]
```

```{code-cell} ipython3
doppler = volumes[1]["sweep_1"].to_dataset()
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
fig.suptitle("LLSD of the dealiased 0.5° velocity, 00:00 UTC")
plt.show()
```

## 10. Cone gridding, Max-CAPPI, interactive view and UGRID

`dtree.radarx.to_grid` grids reflectivity, ZDR, KDP, dealiased velocity and
azimuthal shear of each volume onto a 300 km × 300 km × 12 km Cartesian grid
with the compiled cone method. Of the surveillance and Doppler cuts at each
elevation it takes, for every field, the cut that holds it (and reaches
farthest), so the split cuts need no merging first.

```{code-cell} ipython3
fields = ["DBZH", "ZDR", "KDP", "VRADH_dealiased", "azimuthal_shear"]
grid_kw = dict(
    x_lim=(-150e3, 150e3),
    y_lim=(-150e3, 150e3),
    z_lim=(500, 12e3),
    x_step=1000,
    y_step=1000,
    z_step=500,
)
with timed("10. grid"):
    grids = [vol.radarx.to_grid(fields, **grid_kw) for vol in volumes]
grids[1]
```

The Max-CAPPI shows the column maximum with its north–south and east–west
projections:

```{code-cell} ipython3
grids[1].radarx.plot_max_cappi("DBZH", cmap="ChaseSpectral", vmin=-10, vmax=70, range_rings=True)
```

CAPPIs of the other gridded fields at 2 km:

```{code-cell} ipython3
level = grids[1].sel(z=2000)
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
fig.suptitle("2 km CAPPIs, 00:00 UTC volume")
plt.show()
```

The same Max-CAPPI as an interactive hvplot view (zoom and pan are linked
between the plan view and the projections):

```{code-cell} ipython3
import holoviews as hv

hv.extension("bokeh")
grids[1].radarx.plot.max_cappi("DBZH", cmap="ChaseSpectral", clim=(-10, 70))
```

A sweep can also be exported as an unstructured (UGRID) mesh with one face
per gate for uxarray:

```{code-cell} ipython3
with timed("10b. UGRID export (one sweep)"):
    uxds = volumes[1]["sweep_1"].to_dataset().radarx.to_uxarray(["VRADH_dealiased", "azimuthal_shear"])
uxds
```

## 11. Storm motion, common analysis time and time interpolation

The two volumes are 6.5 minutes apart, during which the line moved several
kilometres. `estimate_motion` tracks the reflectivity pattern between the two
grids; `advect` moves every gridded field of each volume to a common
analysis time (here 00:00 UTC), and `interpolate_time` produces
advection-corrected frames between the two volumes.

```{code-cell} ipython3
analysis_time = np.datetime64("2022-03-31T00:00:00", "ns")
with timed("11. motion, advection, time interpolation"):
    motion = estimate_motion(grids[0], grids[1])
    common = [advect(g, motion, time=analysis_time) for g in grids]
    frame_times = np.arange(
        grids[0].time.values.astype("datetime64[m]") + np.timedelta64(1, "m"),
        grids[1].time.values,
        np.timedelta64(1, "m"),
    ).astype("datetime64[ns]")
    frames = interpolate_time(grids[0][["DBZH"]], grids[1][["DBZH"]], frame_times, motion=motion)
motion.to_pandas()
```

```{code-cell} ipython3
def composite(ds):
    """Column maximum reflectivity, with x and y in km."""
    da = ds.DBZH.max("z")
    return da.assign_coords(x=da.x / 1e3, y=da.y / 1e3)


fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True, layout="constrained")
for ax, data, title in zip(axes, [grids, common], ["as observed", "advected to 00:00 UTC"]):
    ax.pcolormesh(grids[1].x / 1e3, grids[1].y / 1e3, composite(data[1]), cmap="Greys", vmin=0, vmax=70)
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

Moving both volumes to one time brings their 50 dBZ cores together. A single
motion vector cannot follow parts of the storm that move differently; a
tiled motion field (`estimate_motion(..., tile=60e3)`) can.

Advection-corrected frames, one per minute between the two volumes:

```{code-cell} ipython3
fg = composite(frames.isel(time=slice(None, None, 2))).plot(
    x="x", y="y", col="time", cmap="ChaseSpectral", vmin=-10, vmax=70,
    figsize=(13, 4.6), cbar_kwargs={"label": "column maximum reflectivity (dBZ)"},
)
for ax in fg.axs.flat:
    ax.set_aspect("equal")
    ax.set_xlabel("east of KGWX (km)")
fg.axs[0, 0].set_ylabel("north of KGWX (km)")
plt.show()
```

## 12. Multi-radar grid and three-dimensional wind

A second radar looking at the same storm from another direction gives a
second wind component. KBMX (Birmingham, Alabama) is 166 km east-south-east
of KGWX. We read only its reflectivity and velocity, dealias the velocity
with the same ERA5 wind profile, and

- grid both radars onto one shared grid centred on KGWX
  (`radarx.grid.grid_radars`), moving each radar to 00:00 UTC with the storm
  motion of step 11, and merge their reflectivity;
- retrieve the three-dimensional wind with the variational multi-Doppler
  method (`multi_doppler`; Gao et al. 1999), with the ERA5 profile as the
  background.

The [multi-radar gridding](Multi_Radar_Grid) and [multi-Doppler](Multi_Doppler)
notebooks show both in more detail.

From here on only the 00:00 UTC KGWX volume is used, so we release the polar
data of the first one.

```{code-cell} ipython3
kgwx = volumes[1]
del volumes
gc.collect()

with timed("12a. KBMX: read, mask, dealias"):
    path = download_file(
        "unidata-nexrad-level2", "2022/03/30/KBMX/KBMX20220330_235713_V06", "data"
    )
    kbmx = read_volume(path, fields=("DBZH", "VRADH"), max_range=300e3)
    kbmx = kbmx.radarx.apply_mask(kbmx.radarx.echo_mask())
    kbmx = kbmx.radarx.assign(kbmx.radarx.dealias("VRADH", wind_profile=era5))

x = np.arange(-120e3, 60e3 + 1, 2000.0)
y = np.arange(-120e3, 100e3 + 1, 2000.0)
z = np.arange(1000.0, 10e3 + 1, 1000.0)
with timed("12b. multi-radar grid"):
    mosaic = grid_radars(
        [kgwx, kbmx],
        x,
        y,
        z,
        data_vars=["DBZH"],
        names=["KGWX", "KBMX"],
        time=analysis_time,
        motion=motion,
        merge=["DBZH"],
    )
with timed("12c. multi-Doppler wind"):
    md_grid = multi_doppler_input(
        [kgwx, kbmx], x, y, z, velocity="VRADH_dealiased", time=analysis_time, motion=motion
    )
    background = sounding.profile_to_grid(era5, md_grid)
    wind = md_grid.radarx.multi_doppler(background, velocity="VRADH_dealiased")
print(f"iterations per grid level: {wind.attrs['iterations']}")
del kbmx, md_grid, background
gc.collect();
```

```{code-cell} ipython3
good = (wind.beam_crossing_angle > 30) & (wind.n_radars >= 2)
height = 5000.0
level = wind[["u", "v", "w"]].sel(z=height).where(good.sel(z=height))
sub = level.isel(x=slice(None, None, 3), y=slice(None, None, 3))
dbz = mosaic.DBZH_merged.sel(z=height)
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), layout="constrained", sharey=True)
im0 = axes[0].pcolormesh(x / 1e3, y / 1e3, dbz, cmap="ChaseSpectral", vmin=-10, vmax=70)
im1 = axes[1].pcolormesh(x / 1e3, y / 1e3, level.w, cmap="RdBu_r", vmin=-8, vmax=8)
axes[1].contour(x / 1e3, y / 1e3, dbz.fillna(-30), levels=[35, 50], colors="k", linewidths=0.6)
for ax, title in zip(axes, ["merged reflectivity and wind", "w (contours: 35 and 50 dBZ)"]):
    ax.quiver(sub.x / 1e3, sub.y / 1e3, sub.u, sub.v, scale=700, width=0.002)
    ax.plot(mosaic.radar_x / 1e3, mosaic.radar_y / 1e3, "k^")
    ax.set_title(f"{height / 1e3:.0f} km, 00:00 UTC: {title}")
    ax.set_xlim(x[0] / 1e3, x[-1] / 1e3)
    ax.set_ylim(y[0] / 1e3, y[-1] / 1e3)
    ax.set_aspect("equal")
    ax.set_xlabel("east of KGWX (km)")
axes[0].set_ylabel("north of KGWX (km)")
fig.colorbar(im0, ax=axes[0], label="reflectivity (dBZ)", shrink=0.8)
fig.colorbar(im1, ax=axes[1], label="w (m/s)", shrink=0.8)
plt.show()
```

The dual-Doppler area is where both radars observe with a beam crossing
angle above 30°; KBMX is far away, so its beams reach the line only above
about 3 km. Updrafts lie along the leading edge of the line.

+++

## 13. Evaporation

+++

```{note}
A section on evaporative cooling below cloud base (radarx issue #97) will be
added here when the feature lands.
```

+++

## 14. Summary

The runtime of each step:

```{code-cell} ipython3
summary = pd.DataFrame({"time (s)": timings}).round(2)
summary.loc["all steps"] = summary.sum()
summary
```

All products of the case in one `DataTree`: the environment profiles,
the 00:00 UTC polar volume with every derived field, the DSD, the QVPs and melting
layer, the grids as observed and at the common analysis time, the storm
motion, the interpolated frames, the two-radar grid and the wind.

```{code-cell} ipython3
nodes = {"/": xr.Dataset(attrs={"title": "KGWX 30-31 March 2022 squall line, radarx products"})}
for node in kgwx.subtree:
    nodes[f"/polar{node.path if node.path != '/' else ''}"] = node.to_dataset(inherit=False)
nodes["/environment/era5"] = era5
nodes["/environment/radiosonde"] = raob
nodes["/dsd"] = dsd
nodes["/qvp"] = tqvp
nodes["/melting_layer"] = ml
for i, (observed, advected) in enumerate(zip(grids, common)):
    nodes[f"/grid/observed/volume_{i}"] = observed
    nodes[f"/grid/analysis_time/volume_{i}"] = advected
nodes["/grid/frames"] = frames
nodes["/grid/two_radars"] = mosaic
nodes["/motion"] = motion
nodes["/wind"] = wind
products = xr.DataTree.from_dict(nodes)
products
```

## References

- Cao, Q., G. Zhang, E. Brandes, T. Schuur, A. Ryzhkov, and K. Ikeda, 2008:
  Analysis of video disdrometer and polarimetric radar data to characterize
  rain microphysics in Oklahoma. *J. Appl. Meteor. Climatol.*, **47**,
  2238–2255, <https://doi.org/10.1175/2008JAMC1732.1>
- Gao, J., M. Xue, A. Shapiro, and K. K. Droegemeier, 1999: A variational
  method for the analysis of three-dimensional wind fields from two Doppler
  radars. *Mon. Wea. Rev.*, **127**, 2128–2142,
  <https://doi.org/10.1175/1520-0493(1999)127<2128:AVMFTA>2.0.CO;2>
- Gourley, J. J., P. Tabary, and J. Parent du Chatelet, 2007: A fuzzy logic
  algorithm for the separation of precipitating from nonprecipitating echoes
  using polarimetric radar observations. *J. Atmos. Oceanic Technol.*, **24**,
  1439–1451, <https://doi.org/10.1175/JTECH2035.1>
- Krause, J. M., 2016: A simple algorithm to discriminate between
  meteorological and nonmeteorological radar echoes. *J. Atmos. Oceanic
  Technol.*, **33**, 1875–1885, <https://doi.org/10.1175/JTECH-D-15-0239.1>
- Park, H. S., A. V. Ryzhkov, D. S. Zrnić, and K.-E. Kim, 2009: The
  hydrometeor classification algorithm for the polarimetric WSR-88D:
  Description and application to an MCS. *Wea. Forecasting*, **24**,
  730–748, <https://doi.org/10.1175/2008WAF2222205.1>
- Zhang, G., J. Vivekanandan, and E. Brandes, 2001: A method for estimating
  rain rate and drop size distribution from polarimetric radar measurements.
  *IEEE Trans. Geosci. Remote Sens.*, **39**, 830–841,
  <https://doi.org/10.1109/36.917906>
