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

# Cloud Radar Attenuation: Gases, Cloud Liquid Water and Rain

A cloud radar at 35 GHz (Ka band) or 94 GHz (W band) loses part of its signal to oxygen and water
vapour, to cloud liquid water and to rain, and the loss differs between the two bands. This notebook shows
the five functions that give it: `gas_specific_attenuation`, `cloud_liquid_attenuation`,
`gas_path_attenuation`, `rain_attenuation_relation` and `rain_path_attenuation`. The data are synthetic,
with a known answer; the notebook `Dual_Frequency_Ratio` applies them to a day of Ka and W band data.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.retrieve import (
    cloud_liquid_attenuation,
    gas_path_attenuation,
    gas_specific_attenuation,
    rain_attenuation_relation,
    rain_path_attenuation,
)
```

## Gases: oxygen and water vapour

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

## Cloud liquid water

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

### Along the beam

`gas_path_attenuation` interpolates a sounding (or a model profile) to the gates, integrates the specific
attenuation along the beam and returns the two-way attenuation in dB, which is added to the measured
reflectivity. With `lwc` it adds the cloud liquid water. The section below is a vertically pointing radar
with a liquid layer of 0.2 g m$^{-3}$ between 1.5 and 2 km.

```{code-cell} ipython3
height = np.arange(0.0, 12001.0, 250.0)
sounding = xr.Dataset(
    {
        "pressure": ("height", 101325.0 * np.exp(-height / 8000.0)),
        "temperature": ("height", 288.15 - 6.5e-3 * height),
        "specific_humidity": ("height", 0.008 * np.exp(-height / 2500.0)),
    },
    coords={"height": height},
)
gates = np.arange(150.0, 7000.0, 30.0)
section = xr.Dataset(
    {"DBZ": ("range", np.full(gates.size, -10.0))},
    coords={"range": gates - 100.0, "height": ("range", gates)},  # a vertical beam from 100 m
)
liquid = xr.zeros_like(section.DBZ)
liquid[(gates > 1500) & (gates < 2000)] = 0.2
fig, ax = plt.subplots(figsize=(5, 4.2), layout="constrained")
for f in (35.0, 94.0):
    pia = gas_path_attenuation(section, sounding, frequency=f, lwc=liquid)
    ax.plot(pia.PIA_GAS, gates / 1e3, label=f"{f:.0f} GHz, gases")
    ax.plot(pia.PIA_GAS + pia.PIA_CLOUD_LIQUID, gates / 1e3, "--", label=f"{f:.0f} GHz, gases and liquid")
ax.set(xlabel="two-way attenuation (dB)", ylabel="height (km)")
ax.legend()
# the accessor gives the same
same = section.radarx.gas_path_attenuation(sounding, frequency=94.0, lwc=liquid)
print(bool(np.allclose(same.PIA_GAS, pia.PIA_GAS)))
```

## Rain

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

```{code-cell} ipython3
# the accessor gives the same
a, b = float(relations[35.0].a), float(relations[35.0].b)
check = measured.radarx.rain_path_attenuation(35.0, a=a, b=b, max_pia=40.0)
print(bool(np.allclose(check, rain_path_attenuation(measured, 35.0, a=a, b=b, max_pia=40.0))))
```

## References

- Hitschfeld, W., and J. Bordan, 1954: Errors inherent in the radar measurement of rainfall at attenuating wavelengths. *Journal of Meteorology*, **11**, 58-67, <https://doi.org/10.1175/1520-0469(1954)011<0058:EIITRM>2.0.CO;2>
- Liebe, H. J., G. A. Hufford, and T. Manabe, 1991: A model for the complex permittivity of water at frequencies below 1 THz. *International Journal of Infrared and Millimeter Waves*, **12**, 659-675, <https://doi.org/10.1007/BF01008897>
- Rec. ITU-R P.676-13, 2022: Attenuation by atmospheric gases and related effects. <https://www.itu.int/rec/R-REC-P.676>
- Rec. ITU-R P.840-9, 2023: Attenuation due to clouds and fog. <https://www.itu.int/rec/R-REC-P.840>
