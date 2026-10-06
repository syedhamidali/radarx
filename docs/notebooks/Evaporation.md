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

# Rain Evaporation and Evaporative Cooling

Rain falling below cloud base into unsaturated air evaporates and cools the
air. `radarx.retrieve.evaporation` estimates both from a gamma drop size
distribution (DSD) retrieved by `radarx.retrieve.dsd` and the temperature,
pressure and humidity of the air from a sounding or ERA5
(`radarx.io.sounding`):

- every drop evaporates by ventilated diffusion of water vapour,
  $dm/dt = 2\pi D f_v (S - 1) / (F_K + F_D)$, with the ventilation
  coefficient $f_v = 0.78 + 0.308 N_{Sc}^{1/3} N_{Re}^{1/2}$ and the
  thermodynamic terms $F_K$, $F_D$ listed by Kumjian and Ryzhkov (2010);
- with the fall speed of Atlas et al. (1973) in the form
  $V = a D^b e^{-fD}$ and the saturation vapour pressure of Buck (1981),
  the integral over $N(D) = N_0 D^\mu e^{-\Lambda D}$ is closed-form;
- the outputs are the evaporation rate (kg kg$^{-1}$ s$^{-1}$), the cooling
  rate $L_v E / c_p$ (K s$^{-1}$ and K h$^{-1}$) and the tendency of the
  reflectivity factor (dB h$^{-1}$).

`radarx.retrieve.integrate_evaporation` follows the temperature and humidity
of the air through a sequence of radar volumes. A compiled kernel computes
every gate, QVP level or grid point of all inputs in one multithreaded call.

```{code-cell} ipython3
import warnings

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xradar as xd

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.io import sounding
from radarx.retrieve import (
    drop_evaporation_rate,
    evaporation,
    melting_layer,
    qvp_timeseries,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
```

## Single drops and Marshall–Palmer rain

Small drops shrink fastest; large drops lose the most mass. For a
Marshall–Palmer DSD at 20 °C and 900 hPa the cooling rate grows with the rain
rate and with the saturation deficit, and vanishes in saturated air:

```{code-cell} ipython3
t, p = 293.15, 9.0e4
rh = xr.DataArray(np.linspace(0.3, 1.0, 36), dims="rh", name="relative humidity")
diam = xr.DataArray([0.5, 1.0, 2.0, 4.0], dims="diameter")
e = rh * sounding.saturation_vapor_pressure(t)
qv = 0.622 * e / (p - 0.378 * e)  # specific humidity
drops = drop_evaporation_rate(diam, t, p, qv)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for d in diam.values:
    rate = drops.diameter_rate.where(diam == d, drop=True).squeeze()
    ax1.plot(rh, -rate * 1e3, label=f"{d} mm")
ax1.set_xlabel("relative humidity")
ax1.set_ylabel("$-dD/dt$ ($\\mu$m s$^{-1}$)")
ax1.legend()
for rate in (1.0, 5.0, 20.0, 50.0):
    mp = xr.Dataset({"N0": 8000.0, "MU": 0.0, "LAMBDA": 4.1 * rate**-0.21})
    out = evaporation(mp, temperature=t, pressure=p, relative_humidity=rh)
    ax2.plot(rh, out.COOLING_RATE_HOURLY, label=f"{rate:g} mm h$^{{-1}}$")
ax2.set_xlabel("relative humidity")
ax2.set_ylabel("cooling rate (K h$^{-1}$)")
ax2.legend(title="Marshall–Palmer")
for ax in (ax1, ax2):
    ax.grid(alpha=0.3)
```

## KGWX squall line, 30–31 March 2022

Quasi-vertical profiles (QVPs) of the 19.5° sweep of the KGWX (Columbus,
Mississippi) WSR-88D about every 10 minutes from 00:00 to 01:00 UTC on 31
March 2022: moderate rain ahead of the squall line, then the convective line
passing over the radar at about 00:30. NEXRAD no-data codes are masked.

```{code-cell} ipython3
fs = fsspec.filesystem("s3", anon=True)
keys = sorted(fs.ls("unidata-nexrad-level2/2022/03/31/KGWX/"))
keys = [k for k in keys if "KGWX20220331_00" in k and k.endswith("_V06")][::2]


def open_volume(key):
    local = fsspec.open_local(
        f"simplecache::s3://{key}", s3={"anon": True}, filecache={"cache_storage": "."}
    )
    dtree = xd.io.open_nexradlevel2_datatree(local)
    for name in [n for n in dtree.children if n.startswith("sweep")]:
        ds = dtree[name].to_dataset(inherit=False)
        for var, lim in (("DBZH", -32.0), ("ZDR", -12.9), ("RHOHV", 0.21)):
            ds[var] = ds[var].where(ds[var] > lim)
        dtree[name] = ds
    return dtree


volumes = [open_volume(k) for k in keys]
tqvp = qvp_timeseries(volumes, ["DBZH", "ZDR", "RHOHV"], elevation=19.5)
tqvp = tqvp.sel(height=slice(None, 6000.0))
```

ERA5 at the radar at 00 UTC gives the temperature, pressure and humidity.
`evaporation` interpolates a profile on `height` to the QVP heights itself.
The cloud base is taken as the lowest level with a relative humidity of
95 % or more:

```{code-cell} ipython3
lat, lon = float(volumes[0].latitude), float(volumes[0].longitude)
era5 = sounding.era5_profile(lat, lon, "2022-03-31T00:00", source="gcs")
rh_qvp = sounding.interpolate_profile(era5, tqvp.height, ["relative_humidity"])
cloud_base = float(rh_qvp.height.where(rh_qvp.relative_humidity >= 0.95).min())
print(f"ERA5 cloud base: {cloud_base:.0f} m above sea level")
```

The DSD is retrieved in rain only: below the bottom of the melting layer
(found by `melting_layer` with the ERA5 profile as a guide), with
$\rho_{hv} \geq 0.97$. The QVP $Z_{DR}$ at 19.5° is a little lower than at
low elevation angles, so the retrieved drops are slightly too small.

```{code-cell} ipython3
ml = melting_layer(tqvp, environment=era5)
bottom = ml.melting_layer_bottom.to_series().ffill().bfill().to_xarray()
rain = (tqvp.height < bottom - 250.0) & (tqvp.RHOHV >= 0.97) & (tqvp.DBZH >= 5.0)
params = tqvp.radarx.dsd(band="S", mask=rain)
rates = params.radarx.evaporation(era5)
rates
```

```{code-cell} ipython3
below = rain & (tqvp.height < cloud_base)
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
kw = dict(x="time", y="height")
tqvp.DBZH.where(tqvp.RHOHV > 0.8).plot(
    ax=axes[0], cmap="turbo", vmin=0, vmax=55, cbar_kwargs={"label": "$Z_H$ (dBZ)"}, **kw
)
rates.COOLING_RATE_HOURLY.where(below).plot(
    ax=axes[1], cmap="Blues", vmin=0, cbar_kwargs={"label": "cooling (K h$^{-1}$)"}, **kw
)
(rates.EVAPORATION_RATE * 3.6e6).where(below).plot(
    ax=axes[2],
    cmap="Purples",
    vmin=0,
    cbar_kwargs={"label": "evaporation (g kg$^{-1}$ h$^{-1}$)"},
    **kw,
)
for ax in axes:
    ax.plot(ml.time, ml.melting_layer_bottom, "kv", ms=4, label="melting layer bottom")
    ax.axhline(cloud_base, color="m", label="ERA5 cloud base (RH 95 %)")
    ax.set_ylim(0, 5000)
    ax.set_xlabel("")
    ax.set_title("")
axes[0].legend(loc="upper left", fontsize=8);
```

```{code-cell} ipython3
cool = rates.COOLING_RATE_HOURLY.where(below)
print(
    f"cooling below cloud base: median {float(cool.median()):.2f} K/h, "
    f"90th percentile {float(cool.quantile(0.9)):.2f} K/h, max {float(cool.max()):.2f} K/h"
)
```

The cooling reaches 10–30 K/h at the lowest levels under the convective line,
where heavy rain falls into air of about 85 % relative humidity, and a few
K/h in the moderate rain ahead of it.

## Cooling and moistening between volumes

`integrate_evaporation` starts from the ERA5 state and lets every QVP level
cool and moisten under the observed rain until the next volume (steps of at
most `max_step` seconds, never beyond saturation). The column is held fixed:
there is no advection, mixing or downdraft, so this is the cooling the air
would feel if it stayed under the rain.

```{code-cell} ipython3
state = params.radarx.integrate_evaporation(era5, max_step=30.0)
fig, ax = plt.subplots(figsize=(10, 3.5))
state.TEMPERATURE_CHANGE.where(tqvp.height < cloud_base).plot(
    ax=ax, cmap="Blues_r", cbar_kwargs={"label": "$\\Delta T$ since 00:00 (K)"}, **kw
)
ax.set_ylim(0, 5000);
```

## References

- Atlas, D., R. C. Srivastava, and R. S. Sekhon, 1973: Doppler radar
  characteristics of precipitation at vertical incidence. *Rev. Geophys.*,
  **11**, 1–35, <https://doi.org/10.1029/RG011i001p00001>
- Buck, A. L., 1981: New equations for computing vapor pressure and
  enhancement factor. *J. Appl. Meteor.*, **20**, 1527–1532,
  <https://doi.org/10.1175/1520-0450(1981)020<1527:NEFCVP>2.0.CO;2>
- Foote, G. B., and P. S. du Toit, 1969: Terminal velocity of raindrops
  aloft. *J. Appl. Meteor.*, **8**, 249–253,
  <https://doi.org/10.1175/1520-0450(1969)008<0249:TVORA>2.0.CO;2>
- Kumjian, M. R., and A. V. Ryzhkov, 2010: The impact of evaporation on
  polarimetric characteristics of rain: Theoretical model and practical
  implications. *J. Appl. Meteor. Climatol.*, **49**, 1247–1267,
  <https://doi.org/10.1175/2010JAMC2243.1>
- Li, X., and R. C. Srivastava, 2001: An analytical solution for raindrop
  evaporation and its application to radar rainfall measurements. *J. Appl.
  Meteor.*, **40**, 1607–1616,
  <https://doi.org/10.1175/1520-0450(2001)040<1607:AASFRE>2.0.CO;2>
