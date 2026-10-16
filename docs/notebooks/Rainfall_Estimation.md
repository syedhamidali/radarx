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

# Rainfall estimation

Rain rate in `radarx.retrieve` from the polarimetric variables, gate by gate:

- `rain_rate_z`: $R(Z) = a\,Z^b$, the operational reflectivity relation;
- `rain_rate_kdp`: $R(K_{DP}) = a\,|K_{DP}|^b\,\mathrm{sign}(K_{DP})$, which
  does not depend on the radar calibration or on attenuation;
- `rain_rate_z_zdr`: $R(Z, Z_{DR}) = a\,Z^b\,z_{dr}^c$, which corrects $R(Z)$
  for the drop size with the differential reflectivity
  ($z_{dr} = 10^{Z_{DR}/10}$);
- `rain_rate_a`: $R(A) = a(T)\,A^{b(T)}$ from the specific attenuation $A$
  of a method based on the differential phase;
- `rain_rate`: the relation is chosen from the hydrometeor class of
  `radarx.retrieve.hid`, and NaN with a flag where no relation is valid
  (hail, graupel, snow, ice, the melting layer).

The relations are those of [Ryzhkov et al. (2014)](https://doi.org/10.1175/JTECH-D-13-00038.1)
and of the extended abstract of [Giangrande and Ryzhkov (2008)](https://doi.org/10.1175/2008JAMC1753.1).
The coefficients depend on the radar band and on the rain regime they were
fitted for, so they are parameters of the functions (`coefficients=`) with
defaults only where a paper gives them:

| band | $R(Z)$ | $R(K_{DP})$ | $R(Z, Z_{DR})$ | $R(A)$ |
|---|---|---|---|---|
| S | 0.017 $Z^{0.714}$ | 44.0 $K_{DP}^{0.822}$ | 0.0142 $Z^{0.770} z_{dr}^{-1.67}$ | Table 1, 0 to 30 °C |
| C | none | 30.5 $K_{DP}^{0.85}$ | 0.0058 $Z^{0.91} z_{dr}^{-2.09}$ | Table 1, 0 to 30 °C |
| X | 0.029 $Z^{0.67}$ | 16.9 $K_{DP}^{0.801}$ | 0.0039 $Z^{1.07} z_{dr}^{-5.97}$ | Table 1, 0 to 30 °C |

The C-band $R(K_{DP})$ and $R(Z, Z_{DR})$ and the X-band $R(Z, Z_{DR})$ come from the
T-matrix simulations of [Bringi and Chandrasekar (2001)](https://doi.org/10.1017/CBO9780511541094)
(Table 8.1 and Eq. 8.14, at 5.45 and 10 GHz); they are not fitted to a region.
The notebook `TMatrix_Textbook_Examples` reproduces them with the T-matrix code of radarx.

A rain rate is only as good as its inputs. Before the estimate, correct $Z_H$
and $Z_{DR}$ for attenuation (strong at C and X band) and calibrate them,
and take $K_{DP}$ from a processed differential phase
(`radarx.retrieve.estimate_kdp`). For $R(A)$ the specific attenuation
must come from the differential phase, for example $A = \alpha K_{DP}$ or the
ZPHI method, so that it does not depend on the reflectivity calibration.
Where there is no default, the functions stop with an error instead of
guessing.

The relations run in a compiled, multithreaded kernel; every function works
on a `DataArray`, a `Dataset` (a sweep or a grid) and a `DataTree` (a volume),
keeps dask arrays lazy and leaves its input unchanged.

```{code-cell} ipython3
import warnings

import cmweather  # noqa: F401  radar colormaps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xradar as xd
from matplotlib.colors import BoundaryNorm, ListedColormap
from open_radar_data import DATASETS

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.io.aws_data import download_file
from radarx.retrieve import (
    dsd,
    estimate_kdp,
    hid_classes,
    radar_from_dsd,
    rain_rate,
    rain_rate_a,
    rain_rate_kdp,
    rain_rate_z,
    rain_rate_z_zdr,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
```

## Rain of known rate: PERiLS disdrometer drop size distributions

To test the relations, we need rain with a rate we know. radarx ships the
gamma parameters ($N_w$, $D_m$, $\mu$) of 1287 one-minute drop size
distributions measured by the Portable In situ Precipitation Stations
(OTT Parsivel2) of the PERiLS 2022 campaign ([Dawson et al. 2025](https://doi.org/10.26023/HFBG-7W5M-WA00);
[Kosiba et al. 2024](https://doi.org/10.1175/BAMS-D-22-0064.1)), the prior of
`dsd_bayesian`. Each distribution is turned into $Z_H$, $Z_{DR}$, $K_{DP}$
and $A_H$ with the T-matrix scattering tables of `radar_from_dsd`, and into
its rain rate with the fall speeds of Atlas et al. (1973). The relations then
see only the radar variables, and we compare their rain rate with that of the
distribution. The simulated variables have no measurement error, so the
scatter shown is that of the relations (the drop size distributions) alone.

```{code-cell} ipython3
from pathlib import Path

import radarx.retrieve as retrieve
from scipy.special import gammaln

path = Path(retrieve.__file__).parent / "data" / "dsd_prior_perils2022.csv"
pips = pd.read_csv(path, comment="#", header=None, names=["log10_nw", "dm", "mu", "iop"])
d = np.arange(0.05, 8.0001, 0.05)
mu = pips.mu.values[:, None]
shape = 6.0 / 256.0 * np.exp((4 + mu) * np.log(4 + mu) - gammaln(mu + 4))
x = d[None, :] / pips.dm.values[:, None]
nd = xr.DataArray(
    10 ** pips.log10_nw.values[:, None] * shape * x**mu * np.exp(-(4 + mu) * x),
    dims=("minute", "diameter"),
    coords={"diameter": d},
    attrs={"units": "m-3 mm-1"},
)
truth = {band: radar_from_dsd(nd, band=band, temperature=20.0) for band in "SCX"}
print(f"{nd.sizes['minute']} one-minute distributions, rain rate "
      f"{float(truth['S'].RAIN_RATE.min()):.1f} to {float(truth['S'].RAIN_RATE.max()):.0f} mm/h, "
      f"median {float(truth['S'].RAIN_RATE.median()):.1f} mm/h")
```

At S band, all four relations with the defaults of the papers:

```{code-cell} ipython3
s = truth["S"]
estimates = {
    "R(Z)": rain_rate_z(s),
    "R(KDP)": rain_rate_kdp(s),
    "R(Z,ZDR)": rain_rate_z_zdr(s),
    "R(A)": rain_rate_a(s, band="S"),
}
R = s.RAIN_RATE.values


def skill(est, ref):
    """Normalized bias and normalized RMSE of an estimate, and the scatter of the log ratio."""
    ok = np.isfinite(est)
    est, ref = est[ok], ref[ok]
    return (
        100 * (est.sum() - ref.sum()) / ref.sum(),
        100 * np.sqrt(np.mean((est - ref) ** 2)) / ref.mean(),
        np.std(np.log(est / ref)),
    )


rows = []
for lo, hi in [(0.5, 5), (5, 20), (20, np.inf)]:
    sel = (R >= lo) & (R < hi)
    for name, est in estimates.items():
        nb, nrmse, sd = skill(est.values[sel], R[sel])
        rows.append((f"{lo:g} to {hi:g} mm/h" if np.isfinite(hi) else f"above {lo:g} mm/h",
                     sel.sum(), name, nb, nrmse, sd))
table = pd.DataFrame(rows, columns=["rain rate", "n", "relation", "bias (%)", "RMSE (%)", "sd of ln(R_est/R)"])
table.round(2).pivot(index="relation", columns="rain rate", values=["bias (%)", "RMSE (%)"]).round(1)
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), sharex=True, sharey=True, layout="constrained")
for ax, (name, est) in zip(axes, estimates.items()):
    hb = ax.hexbin(R, est.values, gridsize=45, xscale="log", yscale="log", mincnt=1,
                   extent=(np.log10(0.3), np.log10(400), np.log10(0.3), np.log10(400)), cmap="viridis")
    ax.plot([0.3, 400], [0.3, 400], "k-", lw=0.8)
    nb, nrmse, _ = skill(est.values[R >= 5], R[R >= 5])
    ax.set_title(f"{name}: bias {nb:+.0f} %, RMSE {nrmse:.0f} % (R $\\geq$ 5 mm/h)", fontsize=10)
    ax.set_xlabel("rain rate of the distribution (mm/h)")
    ax.set_aspect("equal")
axes[0].set_ylabel("estimated rain rate (mm/h)")
fig.colorbar(hb, ax=axes, label="minutes", shrink=0.8)
fig.suptitle("S band, PERiLS 2022 PIPS drop size distributions")
plt.show()
```

The relations differ in what they know. $R(Z)$ does not know the drop size
and is the noisiest. $R(Z, Z_{DR})$ uses the drop size and is less scattered
than $R(Z)$ below 20 mm/h, with a small bias there, but it is biased low in
heavy rain for these distributions (it was fitted to the drop shapes and
distributions of central Oklahoma). Both stop at about 100 mm/h, because
the reflectivity is capped at 53 dBZ (`z_cap`) to limit hail contamination.
$R(A)$ follows the distributions most closely up to 20 mm/h (RMSE below 10 %
in this simulation, where $A$ is exact, and the real $A$ has the error of its
estimate) and overestimates the heaviest rain. $R(K_{DP})$ is nearly unbiased
at all rain rates and has the smallest scatter in rain above 20 mm/h. Within
each regime the biases are those of the coefficients, which is why they are
parameters.

At X band, where the defaults of $R(Z)$ and $R(K_{DP})$ are the ones used for
the Bonn radar and $R(A)$ is the Table 1 relation:

```{code-cell} ipython3
x_band = truth["X"]
est_x = {
    "R(Z)": rain_rate_z(x_band, band="X"),
    "R(KDP)": rain_rate_kdp(x_band, band="X"),
    "R(A)": rain_rate_a(x_band, band="X"),
}
Rx = x_band.RAIN_RATE.values
pd.DataFrame(
    {name: skill(est.values[Rx >= 5], Rx[Rx >= 5]) for name, est in est_x.items()},
    index=["bias (%)", "RMSE (%)", "sd of ln(R_est/R)"],
).round(2)
```

The temperature changes $R(A)$ considerably, more at S band than at C and X
band. The table of the paper is at 0, 10, 20 and 30 °C and radarx interpolates
linearly between these:

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
for ax, (band, amin, amax) in zip(axes, [("S", 4e-4, 0.01), ("C", 3e-3, 0.1), ("X", 0.02, 0.5)]):
    a_axis = xr.DataArray(np.logspace(np.log10(amin), np.log10(amax), 40), dims="a")
    for temp, color in [(0, "C0"), (10, "C2"), (20, "C1"), (30, "C3")]:
        ax.loglog(a_axis, rain_rate_a(a_axis.rename("AH").to_dataset(), band=band, temperature=float(temp)), color=color, label=f"{temp} °C")
    ax.set(title=f"{band} band", xlabel="specific attenuation $A_h$ (dB/km)")
axes[0].set_ylabel("rain rate (mm/h)")
axes[0].legend()
plt.show()
```

## S band: a squall line seen by KGWX

The KGWX (Columbus, Mississippi) WSR-88D volume of 30 March 2022, 23:46 UTC
(the one of the [hydrometeor classification notebook](Hydrometeor_Classification)).
$Z_{DR}$ comes from the radar as it is, and KDP from `estimate_kdp`. The ERA5
profile places the melting layer for the classification.

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
volume = xr.DataTree.from_dict(nodes).xradar.georeference()
volume = volume.radarx.assign(volume.radarx.kdp(phidp="PHIDP", rhohv="RHOHV", dbz="DBZH"))
profile = volume.radarx.sounding(era5_source="gcs")
volume = volume.radarx.assign(volume.radarx.hid(profile, band="S", scores=False))
rain = volume.radarx.rain_rate()
print("sweeps with a rain rate:", list(rain.children))
```

The rain rate of the classes on the lowest sweep. Gates of a class without a
valid relation (graupel, ice, wet snow) are NaN, and `RAIN_RATE_FLAG` tells
them apart from gates where an input is missing or that are not classified:

```{code-cell} ipython3
sweep = volume["sweep_0"].to_dataset()
blend = rain["sweep_0"].to_dataset()
print(blend.RAIN_RATE_FLAG.attrs["flag_meanings"])
flags = {name: int((blend.RAIN_RATE_FLAG == i).sum())
         for i, name in enumerate(blend.RAIN_RATE_FLAG.attrs["flag_meanings"].split())}
print(flags)
relations = dict(zip(blend.RAIN_RATE_RELATION.attrs["flag_meanings"].split(),
                     np.bincount(blend.RAIN_RATE_RELATION.values.ravel(), minlength=5)))
print("gates by relation:", relations)
```

```{code-cell} ipython3
def ppi(ax, ds, da, extent=130, **kwargs):
    rays = np.argsort(ds.azimuth.values)
    pm = ax.pcolormesh(ds.x.values[rays] / 1e3, ds.y.values[rays] / 1e3, da.values[rays], **kwargs)
    ax.set_aspect("equal")
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_xlabel("east of KGWX (km)")
    return pm


have = blend.RAIN_RATE.notnull()
panels = [
    (sweep.DBZH, "$Z_H$ (dBZ)", dict(cmap="ChaseSpectral", vmin=-10, vmax=70)),
    (blend.RAIN_RATE, "rain rate by class (mm/h)", dict(cmap="turbo", vmin=0, vmax=80)),
    (rain_rate_z(sweep).where(have), "R(Z) on the same gates (mm/h)", dict(cmap="turbo", vmin=0, vmax=80)),
    (rain_rate_kdp(sweep).where(have), "R(KDP) on the same gates (mm/h)", dict(cmap="turbo", vmin=0, vmax=80)),
]
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True, layout="constrained")
for ax, (da, label, style) in zip(axes, panels):
    pm = ppi(ax, sweep, da, **style)
    fig.colorbar(pm, ax=ax, label=label, shrink=0.8)
axes[0].set_ylabel("north of KGWX (km)")
plt.show()
```

The default class table uses $R(Z, Z_{DR})$ in rain and $R(K_{DP})$ in a
rain-hail mixture. Other relations are one argument away; here
$R(K_{DP})$ in heavy rain and the wet-snow rule of Giangrande and Ryzhkov
(0.6 $R(Z)$, fitted to Oklahoma gauges, so off by default):

```{code-cell} ipython3
alt = sweep.radarx.rain_rate(class_relations={"heavy_rain": "kdp", "wet_snow": ("z", 0.6)})
for name, field in (("default", blend), ("heavy rain by KDP and wet snow", alt)):
    r = field.RAIN_RATE
    print(f"{name}: {int(r.notnull().sum())} gates, mean {float(r.mean()):.2f} mm/h")
```

To compare with an independent rain rate, we use the DSD retrieval of
`radarx.retrieve.dsd`, which also works from $Z_H$, $Z_{DR}$ and KDP. The two
share their inputs, so this is a consistency check rather than a validation,
and the disdrometer comparison above is the validation.

```{code-cell} ipython3
rain_codes = [c for c, _, name in hid_classes("park", "S")
              if name in ("big_drops", "light_and_moderate_rain", "heavy_rain")]
near = sweep.HID.isin(rain_codes) & (sweep.range <= 120e3)
retrieval = dsd(sweep, kdp="KDP", mask=near, band="S")
est, ref = blend.RAIN_RATE.where(near).values, retrieval.RAIN_RATE.where(near).values
ok = np.isfinite(est) & np.isfinite(ref) & (ref > 1)
ratio = est[ok] / ref[ok]
print(f"{ok.sum()} gates, median ratio {np.median(ratio):.2f}, 5-95 %: "
      f"{np.percentile(ratio, 5):.2f} to {np.percentile(ratio, 95):.2f}")
fig, ax = plt.subplots(figsize=(5.2, 4.8), layout="constrained")
hb = ax.hexbin(ref[ok], est[ok], gridsize=60, xscale="log", yscale="log", mincnt=1, cmap="viridis")
ax.plot([1, 300], [1, 300], "k-", lw=0.8)
ax.set(xlabel="rain rate of the DSD retrieval (mm/h)", ylabel="rain rate by class (mm/h)", aspect="equal")
fig.colorbar(hb, label="gates")
plt.show()
```

## C band: CSAPR2 with corrected fields

The ARM CSAPR2 C-band radar observed deep convection in Argentina. The file
holds the reflectivity and differential reflectivity corrected for
attenuation, the specific attenuation $A_h$ of the Py-ART CMAC processing, a
sounding temperature at every gate and an echo classification. There is no
verified default for $R(Z)$ at C band, and the function says so ($R(K_{DP})$,
$R(Z, Z_{DR})$ and $R(A)$ have defaults):

```{code-cell} ipython3
file = DATASETS.fetch("corcsapr2cmacppiM1.c1.20181111.030003.nc")
csapr = xd.io.open_cfradial1_datatree(file).xradar.georeference()["sweep_0"].to_dataset()
try:
    rain_rate_z(csapr, dbz="attenuation_corrected_reflectivity_h", band="C")
except ValueError as err:
    print(err)
```

$R(A)$ of the table of Ryzhkov et al. (2014) for C band at 20 °C is the
relation the CMAC product uses (it stores it as `rain_rate_A`), so the two
agree; with the sounding temperature of every gate the coefficients follow the
temperature:

```{code-cell} ipython3
ra20 = rain_rate_a(csapr, a="specific_attenuation", band="C", temperature=20.0)
ra_t = rain_rate_a(csapr, a="specific_attenuation", band="C", temperature="sounding_temperature")
ref = csapr.rain_rate_A
ok = np.isfinite(ra20.values) & np.isfinite(ref.values) & (ref.values > 1)
ratio = ra20.values[ok] / ref.values[ok]
print(f"{ok.sum()} gates with R > 1 mm/h; R(A) at 20 °C over the CMAC rain rate: median {np.median(ratio):.3f}, "
      f"5-95 %: {np.percentile(ratio, 5):.3f} to {np.percentile(ratio, 95):.3f}")
ratio_t = ra_t.values[ok] / ref.values[ok]
print(f"with the gate temperature: median {np.median(ratio_t):.3f}, 5-95 %: "
      f"{np.percentile(ratio_t, 5):.3f} to {np.percentile(ratio_t, 95):.3f}")
```

The blend with the C-band classification of the
[hydrometeor classification notebook](Hydrometeor_Classification) uses
$R(A)$ in the rain classes:

```{code-cell} ipython3
kdp = estimate_kdp(
    csapr,
    phidp="uncorrected_differential_phase",
    rhohv="uncorrected_copol_correlation_coeff",
    dbz="uncorrected_reflectivity_h",
)
csapr = csapr.assign(KDP=kdp.KDP, met=csapr.gate_id.isin([1, 2, 4]))
cls_c = csapr.radarx.hid(
    "sounding_temperature",
    band="C",
    mask="met",
    dbz="attenuation_corrected_reflectivity_h",
    zdr="attenuation_corrected_differential_reflectivity",
    kdp="KDP",
    rhohv="copol_correlation_coeff",
)
blend_c = csapr.assign(HID=cls_c.HID).radarx.rain_rate(
    band="C",
    class_relations={"drizzle": "a", "rain": "a", "big_drops": "a"},
    a="specific_attenuation",
    temperature="sounding_temperature",
)
names = cls_c.HID.attrs["flag_meanings"].split()
print({n: int((cls_c.HID == i + 1).sum()) for i, n in enumerate(names)})
print("flags:", {m: int((blend_c.RAIN_RATE_FLAG == i).sum())
                 for i, m in enumerate(blend_c.RAIN_RATE_FLAG.attrs["flag_meanings"].split())})
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharex=True, sharey=True, layout="constrained")
panels = [
    (csapr.attenuation_corrected_reflectivity_h.where(csapr.met), "$Z_H$ corrected (dBZ)", dict(cmap="turbo", vmin=-10, vmax=70)),
    (blend_c.RAIN_RATE, "rain rate by class, R(A) (mm/h)", dict(cmap="turbo", vmin=0, vmax=100)),
    (blend_c.RAIN_RATE_FLAG.where(blend_c.RAIN_RATE_FLAG < 3), "flag (0 estimated, 1 no relation, 2 input missing)", dict(cmap="Set1", vmin=0, vmax=8)),
]
for ax, (da, label, style) in zip(axes, panels):
    pm = ax.pcolormesh(csapr.x / 1000, csapr.y / 1000, da, shading="auto", **style)
    fig.colorbar(pm, ax=ax, label=label, shrink=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(-20, 110)
    ax.set_ylim(-110, 75)
    ax.set_xlabel("x (km)")
axes[0].set_ylabel("y (km)")
plt.show()
```

## Volumes, dask and the accessors

Every function takes a `DataTree` of sweeps and returns one, skipping the
sweeps that lack a field (the NEXRAD Doppler cuts have no $K_{DP}$). Dask
arrays stay lazy until you compute them, and the accessors call the functions:

```{code-cell} ipython3
r_volume = volume.radarx.rain_rate_kdp()
print("sweeps:", list(r_volume.children))
lazy = sweep[["DBZH", "ZDR", "KDP"]].chunk({"azimuth": 120})
r_lazy = lazy.radarx.rain_rate_z_zdr()
print(type(r_lazy.data).__name__, r_lazy.chunks[0][:3], "...")
print("lazy equals eager:", bool(np.allclose(r_lazy.compute(), rain_rate_z_zdr(sweep), equal_nan=True)))
print(r_lazy.attrs["long_name"], "|", r_lazy.attrs["units"])
print(r_lazy.attrs["radarx_references"])
```

## References

- Atlas, D., R. C. Srivastava, and R. S. Sekhon, 1973: Doppler radar characteristics of precipitation at vertical incidence. *Reviews of Geophysics*, **11**, 1-35, <https://doi.org/10.1029/RG011i001p00001>
- Dawson, D., M. Biggerstaff, and S. Waugh, 2025: PERiLS_2022: Portable In Situ Precipitation Stations (PIPS) Data. Version 1.0. NSF NCAR Earth Observing Laboratory, <https://doi.org/10.26023/HFBG-7W5M-WA00>
- Giangrande, S. E., and A. V. Ryzhkov, 2008: Estimation of Rainfall Based on the Results of Polarimetric Echo Classification. *Journal of Applied Meteorology and Climatology*, **47**, 2445-2462, <https://doi.org/10.1175/2008JAMC1753.1> (the relations and the class table were read in the extended abstract of the same title, 33rd Conference on Radar Meteorology, 2007)
- Kosiba, K. A., and Coauthors, 2024: The Propagation, Evolution, and Rotation in Linear Storms (PERiLS) Project. *Bulletin of the American Meteorological Society*, **105**, E1768-E1799, <https://doi.org/10.1175/BAMS-D-22-0064.1>
- Ryzhkov, A. V., S. E. Giangrande, and T. J. Schuur, 2005: Rainfall Estimation with a Polarimetric Prototype of WSR-88D. *Journal of Applied Meteorology*, **44**, 502-515, <https://doi.org/10.1175/JAM2213.1>
- Ryzhkov, A., M. Diederich, P. Zhang, and C. Simmer, 2014: Potential Utilization of Specific Attenuation for Rainfall Estimation, Mitigation of Partial Beam Blockage, and Radar Networking. *Journal of Atmospheric and Oceanic Technology*, **31**, 599-619, <https://doi.org/10.1175/JTECH-D-13-00038.1>
