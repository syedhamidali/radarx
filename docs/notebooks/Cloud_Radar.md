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

# Ka and W Band Cloud Radar

A cloud radar at 35 GHz (Ka band) or 94 GHz (W band) sees cloud droplets, drizzle and ice crystals
that a weather radar misses, and loses part of its signal on the way. This notebook shows what radarx
has for these radars: the attenuation by gases, cloud liquid water and rain, the classification of
the echoes, and what two frequencies together say about the size of rain drops and snow particles.

| Step | Functions |
|---|---|
| Attenuation of the signal | `gas_specific_attenuation`, `cloud_liquid_attenuation`, `gas_path_attenuation`, `rain_attenuation_relation`, `rain_path_attenuation` |
| Classification of the echoes | `classify_cloud_targets` |
| Dual-frequency ratio | `dual_frequency_ratio`, `dfr_rain`, `dfr_snow` |

The first three sections use synthetic atmospheres and rain with a known answer. The last one
uses a day of open data of the Cloudnet station at Lindenberg (Germany, 1 May 2023): a 35.2 GHz
and a 94 GHz cloud radar looking up at the same sky, and the Cloudnet products that combine them
with a lidar, a microwave radiometer and a weather model. Cloudnet data are open (CC BY 4.0); see
the references.

```{code-cell} ipython3
import tempfile
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.fundamentals import raindrop_scattering
from radarx.retrieve import (
    classify_cloud_targets,
    cloud_liquid_attenuation,
    dfr_rain,
    dfr_snow,
    dual_frequency_ratio,
    gas_path_attenuation,
    gas_specific_attenuation,
    rain_attenuation_relation,
    rain_path_attenuation,
)
```

## 1. Gases: oxygen and water vapour

`gas_specific_attenuation` evaluates the line-by-line model of Rec. ITU-R P.676-13: 44 oxygen and 35
water vapour lines, with the continuum of dry air and of water vapour. The figure is the attenuation of
the standard atmosphere at sea level (15 °C, 7.5 g m$^{-3}$ of vapour). The radar windows at 35 and 94 GHz
lie between the oxygen band at 60 GHz and the water vapour lines at 22, 183 and 325 GHz.

```{code-cell} ipython3
frequency = xr.DataArray(np.arange(1.0, 350.0, 0.25), dims="frequency", name="frequency")
standard = gas_specific_attenuation(frequency, 1013.25, 288.15, vapor_density=7.5)
dry = gas_specific_attenuation(frequency, 1013.25, 288.15, 0.0)
fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
ax.semilogy(frequency, standard.A_GAS, label="7.5 g m$^{-3}$ of water vapour")
ax.semilogy(frequency, dry.A_GAS, label="dry air")
for f0, name in ((35.0, "Ka"), (94.0, "W")):
    ax.axvline(f0, color="0.5", ls=":")
    ax.text(f0 + 2, 3e-3, name)
ax.set(xlabel="frequency (GHz)", ylabel="specific attenuation (dB km$^{-1}$)", ylim=(2e-3, 1e3))
ax.legend()
for f0 in (35.0, 94.0):
    a = gas_specific_attenuation(f0, 1013.25, 288.15, vapor_density=7.5)
    print(f"{f0:.0f} GHz: {float(a.A_GAS):.3f} dB/km, of which oxygen {float(a.A_OXYGEN):.3f}")
```

## 2. Cloud liquid water

Droplets of cloud water are small against both wavelengths, so the Rayleigh formula of Rec. ITU-R P.840-9
holds. The attenuation per unit of liquid water content grows with frequency and falls with temperature.
A liquid water path of 0.3 kg m$^{-2}$ (a thin stratocumulus) costs a vertically pointing W band radar
about 3 dB two ways, and a Ka band radar a little more than 0.5 dB.

```{code-cell} ipython3
values = np.arange(-30.0, 31.0, 1.0)
celsius = xr.DataArray(values, dims="celsius", coords={"celsius": values})
kl = {f: cloud_liquid_attenuation(f, celsius + 273.15) for f in (35.0, 94.0)}
fig, ax = plt.subplots(figsize=(5.5, 3.8), layout="constrained")
for f, k in kl.items():
    ax.plot(celsius, k, label=f"{f:.0f} GHz")
ax.set(xlabel="temperature (°C)", ylabel="$K_l$ (dB km$^{-1}$ per g m$^{-3}$)")
ax.legend()
lwp = 0.3  # kg m-2
for f, k in kl.items():
    print(f"{f:.0f} GHz: two-way attenuation of 0.3 kg m-2 of water at 0 °C = "
          f"{2 * float(k.sel(celsius=0.0)) * lwp:.2f} dB")
```

## 3. Rain

Rain is the strongest attenuator: at W band the one-way attenuation exceeds 6 dB km$^{-1}$ at
10 mm h$^{-1}$, and drops larger than 1.5 mm are beyond the Rayleigh limit, so that the reflectivity
saturates. `rain_attenuation_relation` fits $k = a Z^b$ to the T-matrix scattering of gamma drop size
distributions at the frequency, with $Z$ the reflectivity that the radar measures there, and
`rain_path_attenuation` applies the Hitschfeld and Bordan solution along the beam. The synthetic
column below has a reflectivity of 30 dBZ and the path attenuation of its own relation; the solution
recovers it from the attenuated signal alone.

```{code-cell} ipython3
relations = {f: rain_attenuation_relation(f) for f in (35.0, 94.0)}
for f, r in relations.items():
    print(f"{f:.0f} GHz: k = {float(r.a):.4g} Z^{float(r.b):.3f}, scatter {float(r.scatter):.2f} decades")

gate = 30.0
rng = 100.0 + gate * np.arange(60)
# W band reflectivity saturates in rain, so the column there is weaker than at Ka band
column_dbz = {35.0: 30.0, 94.0: 10.0}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
for ax, (f, r) in zip(axes, relations.items()):
    a, b = float(r.a), float(r.b)
    true_dbz = xr.DataArray(np.full(rng.size, column_dbz[f]), dims="range", coords={"range": rng})
    k = a * (10 ** (true_dbz / 10)) ** b
    pia = 2 * (k * gate * 1e-3).cumsum("range") - k * gate * 1e-3  # two-way, at the gate centre
    measured = true_dbz - pia
    estimate = rain_path_attenuation(measured, f, max_pia=40.0)
    ax.plot(rng, true_dbz, "k", label="true")
    ax.plot(rng, measured, label="measured")
    ax.plot(rng, measured + estimate, "--", label="corrected")
    ax.set(xlabel="range (m)", ylabel="reflectivity (dBZ)", title=f"{f:.0f} GHz, {column_dbz[f]:.0f} dBZ rain")
    ax.legend()
    print(f"{f:.0f} GHz: two-way attenuation at the end {float(pia[-1]):.1f} dB, "
          f"recovered {float(estimate[-1]):.1f} dB")
```

The correction is exact here because the truth follows the same relation. In real rain the relation
has a spread of about 0.1 decades, and at W band the correction is unstable beyond a few dB
(`max_pia`). Use the Doppler spectra or the dual-frequency ratio below where the rain is heavier.

### Dual-frequency ratio of rain

The ratio of the reflectivities at Ka and W band does not depend on how many drops there are, only on
their size, so a table of the ratio against the mass-weighted mean diameter $D_m$ gives the size, and the
reflectivity then gives the normalized intercept parameter $N_w$. The check below computes the two
reflectivities of gamma distributions of known $D_m$ and $N_w$ with the T-matrix code and retrieves
them again.

```{code-cell} ipython3
from math import lgamma

d = np.arange(0.01, 8.0001, 0.01)
mu, nw = 3.0, 6000.0


def reflectivity(dm, frequency):
    f_mu = 6.0 / 4.0**4 * (mu + 4.0) ** (mu + 4.0) / np.exp(lgamma(mu + 4.0))
    nd = nw * f_mu * (d / dm) ** mu * np.exp(-(4.0 + mu) * d / dm)
    wavelength = 299.792458 / frequency
    drops = raindrop_scattering(d, wavelength, temperature=10.0)
    z = wavelength**4 / (np.pi**5 * 0.93) * (nd * drops.sigma_h.values).sum() * 0.01
    return 10 * np.log10(z)


dm_true = np.array([0.8, 1.2, 1.6, 2.0, 2.4, 2.8])
ka_dbz = xr.DataArray([reflectivity(dm, 35.0) for dm in dm_true], dims="case")
w_dbz = xr.DataArray([reflectivity(dm, 94.0) for dm in dm_true], dims="case")
ratio = dual_frequency_ratio(ka_dbz, w_dbz)
rain = dfr_rain(ratio, ka_dbz, mu=mu)
print("DFR (dB):", ratio.values.round(1))
print("Dm true  :", dm_true)
print("Dm found :", rain.DM.values.round(2))
print("Nw found :", rain.NW.values.round(0))
```

## 4. A day of Ka and W band data

The Lindenberg files are the radar files of the two radars and the Cloudnet products of the same day.
The cell downloads them (about 60 MB) into the temporary directory of the machine and keeps the six
hours from 18 to 24 UTC, in which an ice cloud, melting ice, drizzle and liquid layers pass.

```{code-cell} ipython3
BASE = "https://cloudnet.fmi.fi/api/download/product/"
FILES = {
    "categorize": "231c0919-f399-4fea-aa04-3435754f96b9/20230501_lindenberg_categorize.nc",
    "classification": "9eac625e-a74b-478c-b5f6-ff66aaf6771d/20230501_lindenberg_classification.nc",
    "lwc": "8cfef30c-e47b-4276-ac7c-017a44af45ba/20230501_lindenberg_lwc-scaled-adiabatic.nc",
    "w": "ebe41f86-e584-4bd0-bd44-37a20c488913/20230501_lindenberg_rpg-fmcw-94_70dd0955.nc",
}
cache = Path(tempfile.gettempdir()) / "radarx_cloudnet"
cache.mkdir(exist_ok=True)


def fetch(key, attempts=4):
    """Download a file once (to a temporary name first, with retries) and open it."""
    path = cache / Path(FILES[key]).name
    for attempt in range(attempts):
        if path.exists():
            break
        part = path.with_suffix(".part")
        try:
            urllib.request.urlretrieve(BASE + FILES[key], part)  # noqa: S310
            part.rename(path)
        except OSError:
            part.unlink(missing_ok=True)
            if attempt == attempts - 1:
                raise
    return xr.open_dataset(path, engine="netcdf4")


window = slice("2023-05-01T18:00", "2023-05-01T23:59:59")
cat = fetch("categorize").sel(time=window).load()
cls = fetch("classification").sel(time=window).load()
lwc = fetch("lwc").lwc.sel(time=window).load() * 1e3  # g m-3
w = fetch("w").sel(time=window).load()
print(cat.Z.shape, w.Zh.shape, float(cat.radar_frequency), float(w.radar_frequency))
```

### Gas and liquid water attenuation

The categorization file holds the model profile of temperature, pressure and humidity, and the
attenuation that Cloudnet computes with its own absorption model. The profile of the hour of each
profile of the radar goes into `gas_path_attenuation`.

```{code-cell} ipython3
altitude = float(cat.altitude.mean())


def path_attenuation(section):
    """Gas and cloud liquid attenuation of a section of a radar, with the model profile of its hour."""
    pieces = []
    hours = section.time.values.astype("datetime64[h]")
    for hour in np.unique(hours):
        part = section.isel(time=np.flatnonzero(hours == hour))
        model_time = cat.model_time.sel(model_time=hour + np.timedelta64(30, "m"), method="nearest")
        profile = (
            cat[["temperature", "pressure", "q"]]
            .sel(model_time=model_time)
            .rename(model_height="height", q="specific_humidity")
            .drop_vars("model_time")
        )
        liquid = lwc.reindex(time=part.time, method="nearest", tolerance=np.timedelta64(60, "s")).reindex(
            height=part.height.values, method="nearest", tolerance=40.0
        )
        liquid = xr.DataArray(liquid.values, dims=("time", "range"), coords={"time": part.time, "range": part.range})
        pieces.append(gas_path_attenuation(part, profile, lwc=liquid))
    return xr.concat(pieces, "time")


ka = xr.Dataset(
    {"DBZ": cat.Z.rename(height="range"), "ldr": cat.ldr.rename(height="range")},
    coords={
        "range": ("range", cat.height.values.astype(float) - altitude),
        "height": ("range", cat.height.values.astype(float)),
        "radar_frequency": float(cat.radar_frequency),
    },
)
pia_ka = path_attenuation(ka)

ok = np.isfinite(cat.radar_gas_atten.values) & (cat.height.values[None, :] > 400)
ratio = pia_ka.PIA_GAS.values[ok] / cat.radar_gas_atten.values[ok]
print(f"gas attenuation, radarx / Cloudnet: median {np.median(ratio):.3f}, 1-99 % {np.percentile(ratio, [1, 99]).round(3)}")
fig, ax = plt.subplots(figsize=(5, 4.5), layout="constrained")
ax.plot(cat.radar_gas_atten.values[ok][::400], pia_ka.PIA_GAS.values[ok][::400], ".", ms=2)
ax.plot([0, 0.6], [0, 0.6], "k", lw=0.8)
ax.set(xlabel="Cloudnet gas attenuation (dB, two-way)", ylabel="radarx gas_path_attenuation (dB)")
```

The two absorption models agree to a few per cent (the median ratio is 1.00). The gas attenuation
at 5 km is about 0.3 dB at Ka band and 0.9 dB at W band two ways, and that difference of 0.6 dB is what the
dual-frequency ratio needs corrected. The cloud liquid attenuation of the same profiles
(`PIA_CLOUD_LIQUID`, from the liquid water content of Cloudnet) agrees with the liquid attenuation of the
categorization file to 1 %.

### Classification of the echoes

`classify_cloud_targets` uses the reflectivity, the temperature (here the wet-bulb temperature of the
categorization file, as Cloudnet places the melting layer) and the depolarization ratio. Cloudnet has a
lidar and a model besides the radar, so the comparison below merges its classes into the ones a radar
alone can tell.

```{code-cell} ipython3
section = xr.Dataset({"Z": cat.Z, "ldr": cat.ldr, "Tw": cat.Tw}).rename(height="range").assign_coords(range=ka.range)
classes = classify_cloud_targets(section, "Tw", ldr="ldr")
names = ["no echo", "droplets", "drizzle/rain", "ice", "melting ice", "insects"]
merge = {0: 0, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 4, 7: 4, 8: 0, 9: 5, 10: 5}
reference = np.vectorize(lambda v: merge.get(int(v), 0) if np.isfinite(v) else 0)(cls.target_classification.values)
echo = np.isfinite(cat.Z.values)
print(f"gates with an echo: {echo.sum()}, agreement with Cloudnet: {(classes.values == reference)[echo].mean():.3f}")
for i, name in enumerate(names):
    sel = (reference == i) & echo
    if sel.sum():
        print(f"  Cloudnet {name:13s}: {sel.sum():7d} gates, radarx gives the same for {(classes.values[sel] == i).mean():.2f}")

colors = ["white", "tab:cyan", "tab:blue", "0.7", "tab:orange", "tab:green"]
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

cmap = ListedColormap(colors)
norm = BoundaryNorm(np.arange(-0.5, 6.5), cmap.N)
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
for ax, data, title in ((axes[0], reference, "Cloudnet (radar, lidar and model), merged"), (axes[1], classes.values, "radarx, radar only")):
    ax.pcolormesh(cat.time.values, cat.height.values / 1e3, np.where(echo, data, 0).T, cmap=cmap, norm=norm, shading="auto")
    ax.set(ylabel="height (km)", title=title, ylim=(0, 9))
handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="k", lw=0.3) for c in colors[1:]]
axes[1].legend(handles, names[1:], ncol=5, loc="upper right", fontsize=8)
```

### Dual-frequency ratio and snow

The two radars have different gate spacing and time steps, so `dual_frequency_ratio` takes the nearest
gate of the W band radar within a tolerance. The reflectivities of Cloudnet are calibrated so that cloud
droplets give the same dBZ at both frequencies; `calibration="liquid"` converts the ratio to the
$|K_w|^2 = 0.93$ convention of the retrievals, so that ice particles in the Rayleigh regime give 0 dB.
The gas and cloud liquid attenuation of both radars is added first (the lower radar looks through the same liquid layers).

```{code-cell} ipython3
def section_of(ds, frequency):
    return xr.Dataset(
        {"DBZ": ds.Zh},
        coords={
            "range": ds.range.astype("float64"),
            "height": ("range", ds.height.values.astype("float64")),
            "radar_frequency": frequency,
        },
    )


w_section = section_of(w, float(w.radar_frequency))
pia_w = path_attenuation(w_section)

tolerance = {"time": np.timedelta64(15, "s"), "range": 20.0}
ka_dbz = cat.Z.rename(height="range").assign_coords(range=ka.range)
kwargs = dict(match="nearest", tolerance=tolerance, calibration="liquid", frequencies=(35.5, 94.0))
pia_w_total = pia_w.PIA_GAS + pia_w.PIA_CLOUD_LIQUID
pia_ka_total = pia_ka.PIA_GAS + pia_ka.PIA_CLOUD_LIQUID
pia_w_on_ka = (
    pia_w_total.reindex(time=ka_dbz.time, method="nearest", tolerance=np.timedelta64(15, "s"))
    .reindex(range=ka_dbz.range, method="nearest", tolerance=20.0)
)
dfr_raw = dual_frequency_ratio(ka_dbz, w_section.DBZ, **kwargs)
dfr = dual_frequency_ratio(ka_dbz, w_section.DBZ, pia_ka=pia_ka_total, pia_w=pia_w_on_ka, **kwargs)

ice = classes == 3
for lo, hi in ((-25, -15), (-15, -5), (-5, 5), (5, 15)):
    sel = ice & (ka_dbz >= lo) & (ka_dbz < hi) & np.isfinite(dfr)
    print(f"Ka reflectivity {lo:>3} to {hi:>2} dBZ in ice: {int(sel.sum()):6d} gates, "
          f"DFR without correction {float(dfr_raw.where(sel).median()):5.2f} dB, corrected {float(dfr.where(sel).median()):5.2f} dB")
```

Ice crystals that are small against 3 mm are Rayleigh scatterers at both frequencies, so their ratio
is zero, and it grows with the size of the particles: from about 2 dB in the ice cloud of weak
echo to 7 dB where the Ka reflectivity exceeds 5 dBZ. The attenuation correction removes about 1.4 dB. The
2 dB of the weak echo is more than the Rayleigh value; the relative calibration of two radars is
uncertain by a dB or two, and the particles of weak echo are already 0.3 to 0.5 mm in size.

`dfr_snow` turns the ratio and the Ka reflectivity into a mass-weighted mean particle size and an ice
water content, with soft ice spheres of the mass of Brown and Francis (1995). The ice water content of
Cloudnet (from the reflectivity and temperature only) is a loose reference: the two methods have
different assumptions.

```{code-cell} ipython3
ice_gates = ice & np.isfinite(dfr) & (ka_dbz > -30)
snow = dfr_snow(dfr.where(ice_gates), ka_dbz.where(ice_gates), frequencies=(35.2, 94.0))
print("valid retrievals:", int(np.isfinite(snow.IWC).sum()))
print("Dm (mm): 5, 50, 95 percent", np.nanpercentile(snow.DM.values, [5, 50, 95]).round(2))

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout="constrained")
panels = (
    (ka_dbz, "Ka band reflectivity (dBZ)", "viridis", -40, 10),
    (dfr, "dual-frequency ratio Ka - W (dB)", "magma", -2, 12),
    (snow.DM, "Dm of the ice particles (mm)", "cividis", 0, 3),
)
for ax, (data, title, cmap_name, vmin, vmax) in zip(axes, panels):
    pm = ax.pcolormesh(data.time.values, ka.height.values / 1e3, data.values.T, cmap=cmap_name, vmin=vmin, vmax=vmax, shading="auto")
    fig.colorbar(pm, ax=ax, label=title)
    ax.set(ylabel="height (km)", ylim=(0, 9))
```

The accessors give the same results as the functions:

```{code-cell} ipython3
same = section.radarx.classify_cloud_targets("Tw", ldr="ldr")
print(bool((same == classes).all()))
print(bool(np.allclose(dfr.where(ice_gates).radarx.dfr_snow(ka_dbz.where(ice_gates), frequencies=(35.2, 94.0)).DM, snow.DM, equal_nan=True)))
```

## What is not here

* Hail: the Ka and W band ratio of hail depends on the Mie resonances of the stones and is not
  monotonic in the size, so there is no retrieval.
* The Doppler spectra of the two radars (Tridon and Battaglia 2015) and the triple-frequency methods
  (Kneifel et al. 2015) need spectra and a third radar.
* Supercooled liquid, aerosol and mixtures of droplets and drizzle need a lidar.

## References

- Brown, P. R. A., and P. N. Francis, 1995: Improved measurements of the ice water content in cirrus using a total-water probe. *Journal of Atmospheric and Oceanic Technology*, **12**, 410-414, <https://doi.org/10.1175/1520-0426(1995)012<0410:IMOTIW>2.0.CO;2>
- Hitschfeld, W., and J. Bordan, 1954: Errors inherent in the radar measurement of rainfall at attenuating wavelengths. *Journal of Meteorology*, **11**, 58-67, <https://doi.org/10.1175/1520-0469(1954)011<0058:EIITRM>2.0.CO;2>
- Illingworth, A. J., et al., 2007: Cloudnet: Continuous evaluation of cloud profiles in seven operational models using ground-based observations. *Bulletin of the American Meteorological Society*, **88**, 883-898, <https://doi.org/10.1175/BAMS-88-6-883>
- Kneifel, S., A. von Lerber, J. Tiira, D. Moisseev, P. Kollias, and J. Leinonen, 2015: Observed relations between snowfall microphysics and triple-frequency radar measurements. *Journal of Geophysical Research: Atmospheres*, **120**, 6034-6055, <https://doi.org/10.1002/2015JD023156>
- Liebe, H. J., G. A. Hufford, and T. Manabe, 1991: A model for the complex permittivity of water at frequencies below 1 THz. *International Journal of Infrared and Millimeter Waves*, **12**, 659-675, <https://doi.org/10.1007/BF01008897>
- Rec. ITU-R P.676-13, 2022: Attenuation by atmospheric gases and related effects. <https://www.itu.int/rec/R-REC-P.676>
- Rec. ITU-R P.840-9, 2023: Attenuation due to clouds and fog. <https://www.itu.int/rec/R-REC-P.840>
- Tridon, F., and A. Battaglia, 2015: Dual-frequency radar Doppler spectral retrieval of rain drop size distributions and entangled dynamics variables. *Journal of Geophysical Research: Atmospheres*, **120**, 5585-5601, <https://doi.org/10.1002/2014JD023023>
- Tukiainen, S., E. O'Connor, and A. Korpinen, 2020: CloudnetPy: A Python package for processing cloud remote sensing data. *Journal of Open Source Software*, **5**, 2123, <https://doi.org/10.21105/joss.02123>
- Warren, S. G., and R. E. Brandt, 2008: Optical constants of ice from the ultraviolet to the microwave: A revised compilation. *Journal of Geophysical Research: Atmospheres*, **113**, D14220, <https://doi.org/10.1029/2007JD009744>
- Data: ACTRIS Cloud Remote Sensing Data Centre Unit (CLU), Cloudnet products of Lindenberg (DWD Meteorological Observatory Lindenberg), 1 May 2023, <https://cloudnet.fmi.fi> (CC BY 4.0); the product identifiers are in the `pid` attribute of the files.
