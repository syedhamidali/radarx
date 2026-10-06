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

# Hydrometeor Classification

+++

`radarx.retrieve.hid` (or `.radarx.hid()` on a sweep, grid or volume) assigns a
hydrometeor class to every gate with fuzzy logic. For each class a membership
function describes how typical the measured $Z_H$, $Z_{DR}$, $K_{DP}$,
$\rho_{hv}$ and temperature are of that class; the memberships are aggregated
into a score per class and the class with the highest score wins.

| `method` | band | classes | reference |
|---|---|---|---|
| `"park"` (default at S band) | S | dry snow, wet snow, ice crystals, graupel, big drops, rain, heavy rain, rain-hail mixture | Park et al. (2009) |
| `"dolan"` (default at C and X band) | C | drizzle, rain, ice crystals, aggregates, wet snow, vertical ice, low/high-density graupel, hail, big drops | Dolan et al. (2013) |
| `"dolan"` | X, S | drizzle, rain, aggregates, ice crystals, low/high-density graupel, vertical ice | Dolan and Rutledge (2009) |
| `"thompson"` | X, C, S | plates, dendrites, ice crystals, aggregates, wet snow, freezing rain, rain | Thompson et al. (2014) |

The temperature at each gate comes from a sounding or ERA5 profile
(`radarx.io.sounding`) interpolated to the gate heights. The Park et al. (2009)
method uses it to place the melting layer and allows only certain classes
below, inside and above it, accounting for the beam width.

A compiled kernel classifies all gates of a volume in one multithreaded pass.

```{code-cell} ipython3
import time

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xradar as xd
from matplotlib.colors import BoundaryNorm, ListedColormap

import radarx  # noqa: F401  (registers the .radarx accessors)
from radarx.io.aws_data import download_file
from radarx.io.sounding import isotherm_height, wet_bulb_zero_height
from radarx.retrieve import hid_classes
```

## An S-band volume: the KGWX squall line

The KGWX (Columbus AFB, Mississippi) WSR-88D volume of 30 March 2022
23:46 UTC samples a squall line with heavy rain along its leading edge and a
trailing stratiform region. The NEXRAD no-data codes are masked first.

```{code-cell} ipython3
file = download_file(
    "unidata-nexrad-level2", "2022/03/30/KGWX/KGWX20220330_234639_V06", "./downloads"
)
raw = xd.io.open_nexradlevel2_datatree(file)


def mask_no_data(ds):
    limits = {"DBZH": -32.0, "ZDR": -12.9, "RHOHV": 0.21}
    return ds.assign({v: ds[v].where(ds[v] > lim) for v, lim in limits.items() if v in ds})


nodes = {"/": raw.root.to_dataset(inherit=False)}
nodes.update({n: mask_no_data(raw[n].to_dataset(inherit=False)) for n in raw.children})
dtree = xr.DataTree.from_dict(nodes).xradar.georeference()
```

## Temperature from ERA5

`dtree.radarx.sounding()` reads the ERA5 profile at the radar site and volume
time. The 0 °C level is near 3.6 km and the wet-bulb 0 °C level near 3.45 km.

```{code-cell} ipython3
profile = dtree.radarx.sounding(era5_source="gcs")
h0 = float(isotherm_height(profile))
hw = float(wet_bulb_zero_height(profile))
print(f"0 degC: {h0:.0f} m, wet-bulb 0 degC: {hw:.0f} m above sea level")
```

## Classify the volume

KDP is estimated with `radarx.retrieve.estimate_kdp` because the volume has
none. Sweeps without $Z_{DR}$ (the NEXRAD Doppler cuts) are skipped. The
melting layer top is the wet-bulb 0 °C height of the profile and its bottom
500 m lower (`ml_thickness`); `melting_layer=` accepts heights from a QVP
(`radarx.retrieve.melting_layer`) instead.

```{code-cell} ipython3
t0 = time.perf_counter()
out = dtree.radarx.hid(profile, band="S")
print(f"{time.perf_counter() - t0:.2f} s (including KDP and the profile interpolation)")
out["sweep_0"]
```

```{code-cell} ipython3
classes = hid_classes("park", "S")
counts = sum(
    np.bincount(out[n]["HID"].values.ravel(), minlength=len(classes) + 1) for n in out.children
)
for code, abbr, name in classes:
    print(f"{abbr:3s} {name:25s} {100 * counts[code] / counts[1:].sum():5.1f} %")
```

## Plan view of the lowest sweep

```{code-cell} ipython3
COLORS = {
    "drizzle": "#9ec5f4", "rain": "#008300", "light_and_moderate_rain": "#008300",
    "heavy_rain": "#eb6834", "big_drops": "#4a3aa7", "rain_hail_mixture": "#e34948",
    "hail": "#e34948", "wet_snow": "#e87ba4", "dry_snow": "#2a78d6", "aggregates": "#2a78d6",
    "ice_crystals": "#1baf7a", "graupel": "#eda100", "low_density_graupel": "#f2c14e",
    "high_density_graupel": "#c98500", "vertically_aligned_ice": "#5c3b1e",
    "plates": "#5c3b1e", "dendrites": "#1c5cab", "freezing_or_frozen_rain": "#4a3aa7",
    "other": "#7f7f7f",
}  # fmt: skip


def class_colors(classes):
    cmap = ListedColormap([COLORS[name] for _, _, name in classes])
    return cmap, BoundaryNorm(np.arange(0.5, len(classes) + 1.5), cmap.N)


def class_colorbar(mappable, ax, classes):
    cb = plt.colorbar(mappable, ax=ax, ticks=np.arange(1, len(classes) + 1), shrink=0.85)
    cb.ax.set_yticklabels([name.replace("_", " ") for _, _, name in classes], fontsize=8)


cmap, norm = class_colors(classes)
sweep = dtree["sweep_0"].to_dataset()
res = out["sweep_0"].to_dataset()
x, y = sweep.x / 1000, sweep.y / 1000
fig, axs = plt.subplots(1, 3, figsize=(17, 5), sharex=True, sharey=True)
m = axs[0].pcolormesh(x, y, sweep.DBZH, cmap="turbo", vmin=-10, vmax=70, shading="auto")
plt.colorbar(m, ax=axs[0], label="$Z_H$ (dBZ)", shrink=0.85)
m = axs[1].pcolormesh(x, y, res.HID.where(res.HID > 0), cmap=cmap, norm=norm, shading="auto")
class_colorbar(m, axs[1], classes)
m = axs[2].pcolormesh(x, y, res.HID_confidence, cmap="viridis", vmin=0, vmax=1, shading="auto")
plt.colorbar(m, ax=axs[2], label="score of the assigned class", shrink=0.85)
for ax, title in zip(axs, ["reflectivity", "hydrometeor class", "confidence"]):
    ax.set_title(f"0.5°: {title}")
    ax.set_xlim(-150, 60)
    ax.set_ylim(-150, 60)
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
axs[0].set_ylabel("y (km)")
fig.tight_layout()
```

Heavy rain follows the leading edge. Beyond about 100 km the 0.5° beam
reaches the melting layer, and wet snow, graupel and then dry snow appear in
the stratiform region. Weak echo east of the radar with high $Z_{DR}$ is
mostly non-meteorological (insects); remove it with `mask=` (e.g. from an echo
classification) before interpreting it.

## A vertical cross-section

The ray nearest to one azimuth through the line is taken from every sweep and
drawn with its 1° beam width.

```{code-cell} ipython3
def cross_section(ax, azimuth, field, **kwargs):
    for name in out.children:
        sw = dtree[name].to_dataset()
        i = int(np.argmin(np.abs((sw.azimuth.values - azimuth + 180) % 360 - 180)))
        ground = np.hypot(sw.x.values[i], sw.y.values[i]) / 1000
        h = sw.z.values[i] / 1000
        half = sw.range.values * np.sin(np.deg2rad(0.5)) / 1000
        values = field(name)[i]
        m = ax.pcolormesh(
            np.vstack([ground, ground]), np.vstack([h - half, h + half]),
            values[None, :-1], shading="flat", **kwargs,
        )  # fmt: skip
    for level, style in ((h0, "--"), (hw, ":")):
        ax.axhline(level / 1000, color="k", ls=style, lw=1)
    ax.set_ylim(0, 12)
    ax.set_xlim(0, 160)
    ax.set_ylabel("height (km)")
    return m


def hid_field(name):
    c = out[name]["HID"].values.astype(float)
    return np.where(c > 0, c, np.nan)


fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
m = cross_section(axs[0], 215, lambda n: dtree[n]["DBZH"].values,
                  cmap="turbo", vmin=-10, vmax=70)  # fmt: skip
plt.colorbar(m, ax=axs[0], label="$Z_H$ (dBZ)")
m = cross_section(axs[1], 215, hid_field, cmap=cmap, norm=norm)
class_colorbar(m, axs[1], classes)
axs[1].set_xlabel("ground range (km)")
axs[0].set_title("azimuth 215°; dashed: ERA5 0 °C, dotted: wet-bulb 0 °C")
fig.tight_layout()
```

Rain fills the layer below the melting layer, with heavy rain under the
convective line (100-120 km) and a rain-hail mixture in its core. Graupel
reaches 8-9 km in the convective updrafts, wet snow marks the melting layer in
the stratiform region and dry snow and ice crystals fill the upper levels.

## Classes by height

```{code-cell} ipython3
bins = np.arange(0, 12001, 500)
freq = np.zeros((bins.size - 1, len(classes)))
for name in out.children:
    c = out[name]["HID"].values.ravel()
    z = dtree[name]["z"].values.ravel()
    for k in range(len(classes)):
        freq[:, k] += np.histogram(z[c == k + 1], bins)[0]
freq /= np.maximum(freq.sum(axis=1, keepdims=True), 1)

fig, ax = plt.subplots(figsize=(7, 5))
left = np.zeros(bins.size - 1)
for k, (_, _, name) in enumerate(classes):
    ax.barh((bins[:-1] + 250) / 1000, freq[:, k], left=left, height=0.45,
            color=COLORS[name], label=name.replace("_", " "))  # fmt: skip
    left += freq[:, k]
ax.axhline(h0 / 1000, color="k", ls="--", lw=1)
ax.set_xlim(0, 1)
ax.set_xlabel("fraction of classified gates")
ax.set_ylabel("height (km)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, fontsize=8, frameon=False)
fig.tight_layout()
```

## A C-band sweep: CSAPR2

The ARM CSAPR2 C-band radar observed deep convection in Argentina during
CACTI. The file holds a sounding temperature at every gate and an echo
classification (`gate_id`) whose rain, snow and melting gates are used as the
mask. KDP is estimated from the raw differential phase.

```{code-cell} ipython3
from open_radar_data import DATASETS

from radarx.retrieve import estimate_kdp

file = DATASETS.fetch("corcsapr2cmacppiM1.c1.20181111.030003.nc")
sw = xd.io.open_cfradial1_datatree(file).xradar.georeference()["sweep_0"].to_dataset()
kdp = estimate_kdp(
    sw,
    phidp="uncorrected_differential_phase",
    rhohv="uncorrected_copol_correlation_coeff",
    dbzh="uncorrected_reflectivity_h",
)
sw = sw.assign(KDP=kdp.KDP, met=sw.gate_id.isin([1, 2, 4]))
res_c = sw.radarx.hid(
    "sounding_temperature",
    band="C",
    mask="met",
    dbzh="attenuation_corrected_reflectivity_h",
    zdr="attenuation_corrected_differential_reflectivity",
    kdp="KDP",
    rhohv="copol_correlation_coeff",
)
classes_c = hid_classes("dolan", "C")
cmap_c, norm_c = class_colors(classes_c)

fig, axs = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
m = axs[0].pcolormesh(sw.x / 1000, sw.y / 1000, sw.attenuation_corrected_reflectivity_h.where(sw.met),
                      cmap="turbo", vmin=-10, vmax=70, shading="auto")  # fmt: skip
plt.colorbar(m, ax=axs[0], label="$Z_H$ (dBZ)", shrink=0.85)
m = axs[1].pcolormesh(sw.x / 1000, sw.y / 1000, res_c.HID.where(res_c.HID > 0), cmap=cmap_c, norm=norm_c, shading="auto")
class_colorbar(m, axs[1], classes_c)
for ax in axs:
    ax.set_aspect("equal")
    ax.set_xlim(-20, 110)
    ax.set_ylim(-110, 75)
    ax.set_xlabel("x (km)")
axs[0].set_ylabel("y (km)")
fig.tight_layout()
```

At 0.5° the beam stays below the melting layer within 110 km, so drizzle and
rain dominate, with hail and big drops in the strongest cores. On this sweep
the classes agree with CSU_RadarTools' `csu_fhc_summer` (C band) at about 90 %
of the gates.

## References

- Park, H. S., A. V. Ryzhkov, D. S. Zrnić, and K.-E. Kim, 2009: The hydrometeor
  classification algorithm for the polarimetric WSR-88D: Description and
  application to an MCS. *Wea. Forecasting*, **24**, 730-748,
  <https://doi.org/10.1175/2008WAF2222205.1>
- Dolan, B., and S. A. Rutledge, 2009: A theory-based hydrometeor
  identification algorithm for X-band polarimetric radars. *J. Atmos. Oceanic
  Technol.*, **26**, 2071-2088, <https://doi.org/10.1175/2009JTECHA1208.1>
- Dolan, B., S. A. Rutledge, S. Lim, V. Chandrasekar, and M. Thurai, 2013: A
  robust C-band hydrometeor identification algorithm and application to a
  long-term polarimetric radar dataset. *J. Appl. Meteor. Climatol.*, **52**,
  2162-2186, <https://doi.org/10.1175/JAMC-D-12-0275.1>
- Thompson, E. J., S. A. Rutledge, B. Dolan, V. Chandrasekar, and B. L. Cheong,
  2014: A dual-polarization radar hydrometeor classification algorithm for
  winter precipitation. *J. Atmos. Oceanic Technol.*, **31**, 1457-1481,
  <https://doi.org/10.1175/JTECH-D-13-00119.1>
