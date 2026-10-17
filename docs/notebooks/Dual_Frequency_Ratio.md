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

# Dual-Frequency Ratio of Ka and W Band Radars

Two radars at different frequencies see the same particles with different reflectivity once the particles are
no longer small against the shorter wavelength. The ratio of the two reflectivities, the dual-frequency ratio
(DFR) $Z_{Ka} - Z_{W}$ in dB, grows with the size of the particles and does not depend on how many there are. This
notebook shows `dual_frequency_ratio`, `dfr_rain` and `dfr_snow`: first for rain, with a forward model of known size
distributions, then for ice cloud and snow on a day of Ka and W band data of the Cloudnet station at Lindenberg
(Germany, 1 May 2023; a 35.2 GHz and a 94.0 GHz radar looking up at the same sky).

```{code-cell} ipython3
import tempfile
import urllib.request
from math import lgamma
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.fundamentals import raindrop_scattering
from radarx.retrieve import (
    classify_cloud_targets,
    dfr_rain,
    dfr_snow,
    dual_frequency_ratio,
    gas_path_attenuation,
)
```

## Rain

The ratio of the reflectivities at Ka and W band does not depend on how many drops there are, only on
their size, so a table of the ratio against the mass-weighted mean diameter $D_m$ gives the size, and the
reflectivity then gives the normalized intercept parameter $N_w$. The check below computes the two
reflectivities of gamma distributions of known $D_m$ and $N_w$ with the T-matrix code and retrieves
them again.

```{code-cell} ipython3
d = np.arange(0.01, 8.0001, 0.01)
nw = 6000.0
# backscatter cross sections of the drops, mm2 (oblate drops, 10 degrees Celsius), once per frequency
sigma = {f: raindrop_scattering(d, 299.792458 / f, temperature=10.0).sigma_h.values for f in (35.0, 94.0)}


def reflectivity(dm, frequency, mu=3.0):
    f_mu = 6.0 / 4.0**4 * (mu + 4.0) ** (mu + 4.0) / np.exp(lgamma(mu + 4.0))
    nd = nw * f_mu * (d / dm) ** mu * np.exp(-(4.0 + mu) * d / dm)
    z = (299.792458 / frequency) ** 4 / (np.pi**5 * 0.93) * (nd * sigma[frequency]).sum() * 0.01
    return 10 * np.log10(z)


dm_true = np.array([0.8, 1.2, 1.6, 2.0, 2.4, 2.8])
ka_dbz = xr.DataArray([reflectivity(dm, 35.0) for dm in dm_true], dims="case")
w_dbz = xr.DataArray([reflectivity(dm, 94.0) for dm in dm_true], dims="case")
ratio = dual_frequency_ratio(ka_dbz, w_dbz)
rain = dfr_rain(ratio, ka_dbz, mu=3.0)
print("DFR (dB):", ratio.values.round(1))
print("Dm true  :", dm_true)
print("Dm found :", rain.DM.values.round(2))
print("Nw found :", rain.NW.values.round(0))
```

The ratio of rain starts at the Rayleigh value of the dielectric factors of water (about 0.6 dB at 10 degrees
Celsius) and saturates above 20 dB at drop sizes of about 3.5 mm, so the retrieval works for mass-weighted mean
diameters between about 0.3 and 3.5 mm. Rain attenuates the W band signal by several dB per kilometre, and the ratio
needs an attenuation correction that is good to a fraction of a dB (see `Cloud_Radar_Attenuation`).

```{code-cell} ipython3
dm_axis = np.arange(0.3, 3.6, 0.1)
fig, ax = plt.subplots(figsize=(5.5, 4), layout="constrained")
for mu in (0.0, 3.0, 6.0):
    ratio_mu = [reflectivity(dm, 35.0, mu) - reflectivity(dm, 94.0, mu) for dm in dm_axis]
    ax.plot(dm_axis, ratio_mu, label=f"$\\mu$ = {mu:.0f}")
ax.set(xlabel="$D_m$ (mm)", ylabel="Ka - W dual-frequency ratio (dB)")
ax.legend()
```

## A day of Ka and W band data

The cell downloads four files (about 55 MB) into the temporary directory of the machine and keeps the six hours from
18 to 24 UTC, in which an ice cloud, melting ice, drizzle and liquid layers pass. The categorization file holds the
35.5 GHz radar, the model profile and the Cloudnet products to compare with; the radar file of the 94 GHz
radar is the second frequency.

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

### Attenuation of the two radars

The signals of the two radars are attenuated by gases and by the liquid layers they look through, and by
different amounts. `gas_path_attenuation` takes the profile of the model of the hour of each profile of the radar and
the liquid water content of Cloudnet (see `Cloud_Radar_Attenuation`).

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

The attenuation of the gases agrees with that of Cloudnet (another absorption model) to a few per cent. The gas
attenuation at 5 km is about 0.3 dB at Ka band and 0.9 dB at W band two ways.

### Dual-frequency ratio and snow

The two radars have different gate spacing and time steps, so `dual_frequency_ratio` takes the nearest gate of the W
band radar within a tolerance. The reflectivities of Cloudnet are calibrated so that cloud droplets give the same
dBZ at both frequencies; `calibration="liquid"` converts the ratio to the $|K_w|^2 = 0.93$ convention of the
retrievals, so that ice particles in the Rayleigh regime give 0 dB. The attenuation of both radars is added first.
Ice gates come from `classify_cloud_targets`.

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

section = xr.Dataset({"Z": cat.Z, "ldr": cat.ldr, "Tw": cat.Tw}).rename(height="range").assign_coords(range=ka.range)
classes = classify_cloud_targets(section, "Tw", ldr="ldr")
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

The accessor gives the same:

```{code-cell} ipython3
again = dfr.where(ice_gates).radarx.dfr_snow(ka_dbz.where(ice_gates), frequencies=(35.2, 94.0))
print(bool(np.allclose(again.DM, snow.DM, equal_nan=True)), bool(np.allclose(dfr.where(ice_gates).radarx.dfr_rain(ka_dbz).DM, dfr_rain(dfr.where(ice_gates), ka_dbz).DM, equal_nan=True)))
```

## What is not here

* Hail: the Ka and W band ratio of hail depends on the Mie resonances of the stones and is not monotonic in the
  size, so there is no retrieval.
* The Doppler spectra of the two radars (Tridon and Battaglia 2015) and the triple-frequency methods (Kneifel et al.
  2015) need spectra and a third radar.

## References

- Brown, P. R. A., and P. N. Francis, 1995: Improved measurements of the ice water content in cirrus using a total-water probe. *Journal of Atmospheric and Oceanic Technology*, **12**, 410-414, <https://doi.org/10.1175/1520-0426(1995)012<0410:IMOTIW>2.0.CO;2>
- Hogan, R. J., A. J. Illingworth, and H. Sauvageot, 2000: Measuring Crystal Size in Cirrus Using 35- and 94-GHz Radars. *Journal of Atmospheric and Oceanic Technology*, **17**, 27-37, <https://doi.org/10.1175/1520-0426(2000)017<0027:MCSICU>2.0.CO;2>
- Matrosov, S. Y., 1998: A Dual-Wavelength Radar Method to Measure Snowfall Rate. *Journal of Applied Meteorology*, **37**, 1510-1521, <https://doi.org/10.1175/1520-0450(1998)037<1510:ADWRMT>2.0.CO;2>
- Illingworth, A. J., et al., 2007: Cloudnet: Continuous evaluation of cloud profiles in seven operational models using ground-based observations. *Bulletin of the American Meteorological Society*, **88**, 883-898, <https://doi.org/10.1175/BAMS-88-6-883>
- Kneifel, S., A. von Lerber, J. Tiira, D. Moisseev, P. Kollias, and J. Leinonen, 2015: Observed relations between snowfall microphysics and triple-frequency radar measurements. *Journal of Geophysical Research: Atmospheres*, **120**, 6034-6055, <https://doi.org/10.1002/2015JD023156>
- Rec. ITU-R P.676-13, 2022: Attenuation by atmospheric gases and related effects. <https://www.itu.int/rec/R-REC-P.676>
- Rec. ITU-R P.840-9, 2023: Attenuation due to clouds and fog. <https://www.itu.int/rec/R-REC-P.840>
- Tridon, F., and A. Battaglia, 2015: Dual-frequency radar Doppler spectral retrieval of rain drop size distributions and entangled dynamics variables. *Journal of Geophysical Research: Atmospheres*, **120**, 5585-5601, <https://doi.org/10.1002/2014JD023023>
- Tukiainen, S., E. O'Connor, and A. Korpinen, 2020: CloudnetPy: A Python package for processing cloud remote sensing data. *Journal of Open Source Software*, **5**, 2123, <https://doi.org/10.21105/joss.02123>
- Warren, S. G., and R. E. Brandt, 2008: Optical constants of ice from the ultraviolet to the microwave: A revised compilation. *Journal of Geophysical Research: Atmospheres*, **113**, D14220, <https://doi.org/10.1029/2007JD009744>
- Data: ACTRIS Cloud Remote Sensing Data Centre Unit (CLU), Cloudnet products of Lindenberg (DWD Meteorological Observatory Lindenberg), 1 May 2023, <https://cloudnet.fmi.fi> (CC BY 4.0); the product identifiers are in the `pid` attribute of the files.
