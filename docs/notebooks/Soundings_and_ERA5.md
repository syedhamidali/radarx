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

# Soundings and ERA5 profiles

+++

Radar retrievals need the environment: the 0 °C level for the melting layer,
temperature and humidity for hydrometeor classification and evaporation, and a
background wind for multi-Doppler winds. `radarx.io.sounding` reads observed
radiosonde soundings and ERA5 reanalysis profiles into one xarray format (a
Dataset on `height` above sea level, CF names and SI units) and puts them on
radar gates or radarx grids.

This example uses the KGWX (Columbus AFB, Mississippi) volume from
30 March 2022 23:46 UTC and the nearest radiosondes, Birmingham AL (BMX) and
Jackson MS (JAN), launched for 00 UTC on 31 March 2022.

```{code-cell} ipython3
import time

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xradar as xd

import radarx  # noqa: F401  (registers the .radarx accessors)
from radarx.io import sounding
from radarx.io.aws_data import download_file
```

## The radar volume

```{code-cell} ipython3
file = download_file(
    "unidata-nexrad-level2", "2022/03/30/KGWX/KGWX20220330_234639_V06", "./downloads"
)
dtree = xd.io.open_nexradlevel2_datatree(file, sweep=[0, 1, 2, 3])
dtree = dtree.xradar.georeference()
site = dtree["sweep_0"].to_dataset(inherit="all_coords")
print(float(site.latitude), float(site.longitude), float(site.altitude))
```

## Nearest radiosonde stations

`nearest_station` searches the station table shipped with radarx (IGRA2
station list merged with the Iowa Environmental Mesonet RAOB network).

```{code-cell} ipython3
near = sounding.nearest_station(
    float(site.latitude), float(site.longitude), "2022-03-31", source="iem", n=3
)
near[["iem_id", "wmo_id", "name", "distance"]].to_dataframe()
```

## Observed soundings

`read_sounding` downloads a sounding from the Iowa Environmental Mesonet
(`source="iem"`, the default), the University of Wyoming archive (`"uwyo"`)
or IGRA2 (`"igra2"`), and caches it. `dtree.radarx.sounding("iem")` picks the
nearest station and the nearest 00/12 UTC launch to the volume time.

```{code-cell} ipython3
bmx = dtree.radarx.sounding("iem")
jan = sounding.read_sounding("KJAN", "2022-03-31T00:00", source="iem")
bmx
```

## ERA5 profiles

`era5_profile` reads ERA5 on pressure levels at a point and time. With
`source="gcs"` it opens Google's analysis-ready ERA5 Zarr store lazily
(anonymous, 37 levels, hourly) and reads only the times needed; the ECMWF
ARCO time series (`source="arco"`, 13 levels, 6-hourly) and the full ERA5 on
the Copernicus CDS (`source="cds"`) need a CDS account. Geopotential is
converted to geometric height, ERA5 is interpolated bilinearly to the point
and linearly in time to the radar volume time. The default (`source="auto"`) is hourly ERA5 on
37 levels: from the CDS when a CDS key is configured, otherwise from Google's
ARCO-ERA5.

```{code-cell} ipython3
t0 = time.perf_counter()
era5_kgwx = dtree.radarx.sounding(era5_source="gcs")  # at 23:46:39 UTC
print(f"ERA5 at KGWX: {time.perf_counter() - t0:.1f} s")
era5_bmx = sounding.era5_profile(33.18, -86.78, "2022-03-31T00:00", source="gcs")
era5_jan = sounding.era5_profile(32.32, -90.08, "2022-03-31T00:00", source="gcs")
era5_kgwx
```

### Temperature, dew point and wind

```{code-cell} ipython3
def plot_profiles(profiles, top=12000):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 6), sharey=True)
    for (label, prof), color in zip(profiles.items(), ["C0", "C1", "C2", "C3"]):
        p = prof.where(prof.height < top, drop=True)
        z = p.height / 1000
        ax1.plot(p.temperature - 273.15, z, color=color, label=label)
        ax1.plot(p.dewpoint - 273.15, z, color=color, ls="--")
        ok = np.isfinite(p.u)
        ax2.plot(p.wind_speed[ok], z[ok], color=color, label=label)
        ax3.plot(p.wind_direction[ok], z[ok], ".", color=color, ms=4)
        zero = float(sounding.isotherm_height(prof))
        ax1.axhline(zero / 1000, color=color, lw=0.8, ls=":")
    ax1.axvline(0, color="k", lw=0.8)
    ax1.set(xlabel="temperature (solid), dew point (dashed) [°C]", ylabel="height [km MSL]")
    ax2.set(xlabel="wind speed [m s$^{-1}$]")
    ax3.set(xlabel="wind direction [°]", xlim=(0, 360), xticks=range(0, 361, 90))
    ax1.legend()
    for ax in (ax1, ax2, ax3):
        ax.grid(alpha=0.3)
    return fig


plot_profiles(
    {"BMX RAOB": bmx, "JAN RAOB": jan, "ERA5 at BMX": era5_bmx, "ERA5 at KGWX 23:46": era5_kgwx}
);
```

### 0 °C, −10 °C, −20 °C and wet-bulb 0 °C heights

```{code-cell} ipython3
rows = []
for label, prof in {
    "BMX RAOB": bmx,
    "ERA5 at BMX": era5_bmx,
    "JAN RAOB": jan,
    "ERA5 at JAN": era5_jan,
    "ERA5 at KGWX": era5_kgwx,
}.items():
    rows.append(
        [label]
        + [float(sounding.isotherm_height(prof, t)) for t in (273.15, 263.15, 253.15)]
        + [float(sounding.wet_bulb_zero_height(prof))]
    )
header = ["profile", "0 °C [m]", "−10 °C [m]", "−20 °C [m]", "wet-bulb 0 °C [m]"]
print(("{:>14}" * 5).format(*header))
for row in rows:
    print("{:>14}".format(row[0]) + ("{:>14.0f}" * 4).format(*row[1:]))
```

### ERA5 against the radiosondes

ERA5 is interpolated to the radiosonde heights between 0.5 and 12 km.

```{code-cell} ipython3
def compare(raob, era5, top=12000):
    obs = raob.where((raob.height > 500) & (raob.height < top), drop=True)
    model = sounding.interpolate_profile(era5, obs.height)
    out = {}
    for name in ("temperature", "dewpoint", "u", "v"):
        d = (model[name] - obs[name]).values
        d = d[np.isfinite(d)]
        out[name] = (d.mean(), np.sqrt((d**2).mean()), d.size)
    return out


for label, raob, era5 in [("BMX", bmx, era5_bmx), ("JAN", jan, era5_jan)]:
    for name, (bias, rmse, n) in compare(raob, era5).items():
        print(f"{label} {name:12s} bias {bias:+5.2f}  rmse {rmse:5.2f}  (n={n})")
```

Temperatures agree to about 1 K. The large wind differences at Jackson are
real: a squall line crossed Jackson shortly before the launch, so the sonde
measured the north-westerly outflow below 2 km while ERA5, which smooths and
slightly delays the line, still has the southerly inflow there.

Mean winds for storm motion and wind retrievals:

```{code-cell} ipython3
for label, prof in {"BMX RAOB": bmx, "ERA5 at KGWX": era5_kgwx}.items():
    mw = sounding.mean_wind(prof, 0, 6000, above_ground=True)
    print(
        f"{label}: 0-6 km mean wind {float(mw.wind_speed):.1f} m/s "
        f"from {float(mw.wind_direction):.0f}°"
    )
```

## Temperature at the radar gates

`ds.radarx.interpolate_profile(profile)` adds the profile variables at the
height of every gate of a georeferenced sweep (or every level of a grid).

```{code-cell} ipython3
sweep = dtree["sweep_1"].to_dataset(inherit="all_coords")
sweep = sweep.radarx.interpolate_profile(era5_kgwx, ["temperature", "pressure"])
sweep[["temperature", "pressure"]]
```

```{code-cell} ipython3
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
sweep.DBZH.where(sweep.DBZH > 0).plot(
    x="x", y="y", ax=ax1, cmap="ChaseSpectral", vmin=0, vmax=70
)
sweep.temperature.plot.contour(
    x="x", y="y", ax=ax1, levels=[253.15, 263.15, 273.15], colors="k"
)
(sweep.temperature - 273.15).plot(x="x", y="y", ax=ax2, cmap="RdBu_r", center=0)
ax1.set_title(f"KGWX {float(sweep.sweep_fixed_angle):.1f}° DBZH, isotherms 0/−10/−20 °C")
ax2.set_title("ERA5 temperature at the gate heights [°C]")
for ax in (ax1, ax2):
    ax.set_aspect("equal")
fig.tight_layout()
```

## A background on a radarx grid

`era5_column` interpolates the ERA5 columns to every cell of a radarx grid
(bilinear in the horizontal, linear in time and height) and rotates the wind to
the grid axes. `profile_to_grid` does the same with one sounding, so both can
feed a wind retrieval interchangeably.

```{code-cell} ipython3
grid = dtree.radarx.to_grid(
    data_vars=["DBZH"],
    x_lim=(-150e3, 150e3),
    y_lim=(-150e3, 150e3),
    z_lim=(500, 12000),
    x_step=2000,
    y_step=2000,
    z_step=500,
)
t0 = time.perf_counter()
background = sounding.era5_column(grid, source="gcs")
print(f"ERA5 columns for {grid.sizes['y'] * grid.sizes['x']} cells: "
      f"{time.perf_counter() - t0:.1f} s")
background
```

```{code-cell} ipython3
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
background.freezing_level.plot(x="x", y="y", ax=ax1, cmap="viridis")
ax1.set_title("ERA5 0 °C level [m MSL]")
level = background.sel(z=3000)
grid.DBZH.max("z").plot(x="x", y="y", ax=ax2, cmap="ChaseSpectral", vmin=0, vmax=70)
s = slice(None, None, 10)
ax2.quiver(
    level.x[s], level.y[s], level.u[s, s], level.v[s, s], scale=600, color="k"
)
ax2.set_title("column max DBZH and ERA5 3 km wind")
for ax in (ax1, ax2):
    ax.set_aspect("equal")
fig.tight_layout()
```
