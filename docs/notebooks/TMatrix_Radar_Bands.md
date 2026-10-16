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

# T-matrix Scattering Across Radar Bands

Rain looks different to a radar of every wavelength. This notebook uses
`radarx.fundamentals.raindrop_scattering` at the six bands of the previous notebook
(`TMatrix_Validation`) to show what changes between S band and W band, checks the result against
radar observations of rain at S, C and X band, and gives the limits of what a single band can do.

```{code-cell} ipython3
from math import lgamma

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xradar as xd
from open_radar_data import DATASETS

from radarx.fundamentals import raindrop_scattering, water_refractive_index

BANDS = {"S": 107.0, "C": 53.5, "X": 31.9, "Ku": 22.1, "Ka": 8.4, "W": 3.19}  # mm
COLORS = dict(zip(BANDS, plt.cm.viridis(np.linspace(0, 0.95, 6))))
D = np.arange(0.1, 8.001, 0.1)  # mm
DD = 0.1
```

## One drop

The backscatter cross section of a drop follows the Rayleigh law,
$\sigma_b \propto D^6/\lambda^4$, while it is small compared with the wavelength. The figure shows the
cross section divided by the Rayleigh value (computed from the refractive index of water with
$|K_w|^2 = |(m^2-1)/(m^2+2)|^2$) in decibels, and the single-drop differential reflectivity
$10\log_{10}(\sigma_h/\sigma_v)$, for drops with the axis ratio of Brandes et al. (2002) and 7 degrees
of canting.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
for band, wavelength in BANDS.items():
    m = water_refractive_index(wavelength, 20.0)
    k2 = abs((m * m - 1) / (m * m + 2)) ** 2
    t = raindrop_scattering(D, wavelength, temperature=20.0)
    rayleigh = np.pi**5 * k2 * D**6 / wavelength**4
    axes[0].plot(D, 10 * np.log10(t.sigma_h / rayleigh), color=COLORS[band], label=band)
    axes[1].plot(D, 10 * np.log10(t.sigma_h / t.sigma_v), color=COLORS[band], label=band)
    axes[2].plot(D, t.kdp, color=COLORS[band], label=band)
axes[0].set(xlabel="diameter (mm)", ylabel="σ$_h$ / σ$_h$(Rayleigh) (dB)", title="Departure from Rayleigh")
axes[1].set(xlabel="diameter (mm)", ylabel="Z$_{DR}$ of one drop (dB)", title="Differential reflectivity")
axes[2].set(xlabel="diameter (mm)", ylabel="K$_{DP}$ per drop per m$^3$ (° km$^{-1}$)", title="Specific differential phase")
axes[2].set_yscale("symlog", linthresh=1e-4)
axes[0].legend(ncols=2)
for ax in axes:
    ax.grid(alpha=0.3)
```

At S band the cross section stays within 1 dB of the Rayleigh value at every size. At C and X band
the largest drops scatter more than Rayleigh says (by up to 6 dB at C band for 6 to 8 mm and
4.6 dB at X band for about 5 mm, a resonance that also lifts $Z_{DR}$ to 7 dB for a 6 mm drop at C
band, and makes $K_{DP}$ slightly negative for drops of 6.0 to 6.5 mm at C band). At Ku band
the excess is 3.5 dB for drops of 3 to 5 mm, and drops of 7 mm and more fall below Rayleigh. At Ka band
the cross section falls below Rayleigh from about 2.5 mm, with a null near 4.2 mm, and at W band from 1 mm
on, oscillating as the Mie theory predicts. At Ka and W band the differential reflectivity of a
single drop oscillates between -4 and +8 dB, and $K_{DP}$ is negative for drops larger than 2.3 mm (Ka) and
0.8 mm (W), shown on the symmetric logarithmic axis, so the polarimetric variables of the
lower bands lose their meaning there.

## A population of drops

For gamma distributions $N(D) = N_w f(\mu)(D/D_m)^\mu \exp[-(3.67+\mu)D/D_m]$
the integrals of the single-drop table give the radar variables. With $\mu = 3$ and a
concentration $N_w = 8000$ mm$^{-1}$ m$^{-3}$, as functions of the mean drop diameter:

```{code-cell} ipython3
def gamma_dsd(dm, mu=3.0, nw=8000.0):
    f = 6 * (3.67 + mu) ** (mu + 4) / (3.67**4 * np.exp(lgamma(mu + 4)))
    x = D[None, :] / np.atleast_1d(dm)[:, None]
    return nw * f * x**mu * np.exp(-(3.67 + mu) * x)


dm = np.linspace(0.5, 3.5, 31)
N = gamma_dsd(dm)
products = {}
for band, wavelength in BANDS.items():
    m = water_refractive_index(wavelength, 20.0)
    k2 = abs((m * m - 1) / (m * m + 2)) ** 2
    t = raindrop_scattering(D, wavelength, temperature=20.0)
    zh = wavelength**4 / (np.pi**5 * k2) * (N * t.sigma_h.values).sum(axis=1) * DD
    zv = wavelength**4 / (np.pi**5 * k2) * (N * t.sigma_v.values).sum(axis=1) * DD
    products[band] = dict(
        z=10 * np.log10(zh),
        zdr=10 * np.log10(zh / zv),
        kdp=(N * t.kdp.values).sum(axis=1) * DD,
        ah=(N * t.ah.values).sum(axis=1) * DD,
    )

fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), layout="constrained")
for band, p in products.items():
    for ax, key in zip(axes, ("z", "zdr", "kdp", "ah")):
        ax.plot(dm, p[key], color=COLORS[band], label=band)
for ax, label in zip(
    axes,
    ("Z$_H$ (dBZ)", "Z$_{DR}$ (dB)", "K$_{DP}$ (° km$^{-1}$)", "A$_H$ (dB km$^{-1}$)"),
):
    ax.set(xlabel="mean diameter D$_m$ (mm)", ylabel=label)
    ax.grid(alpha=0.3)
axes[2].set_yscale("symlog", linthresh=0.1)
axes[3].set_yscale("log")
axes[0].legend(ncols=2)
```

At fixed $N_w$ and $\mu$ the reflectivity of S, C, X and Ku band is the same to within 3 dB
(the Rayleigh value), and at Ka and W band it falls far below it for the large mean diameters
(14 and 32 dB below at $D_m = 3.5$ mm). $Z_{DR}$ increases with $D_m$ up to Ku band (the resonance
of the large drops makes the C-band value the largest, 4.4 dB at $D_m = 3.5$ mm against 3.0 dB at S band), and
at Ka and W band it stays at 0.6 and 0.1 dB, which is why dual-polarization is of no
use at those bands for the size of drops. $K_{DP}$ increases with the frequency up to
Ku band and is negative at W band and, for $D_m > 2.3$ mm, at Ka band. The attenuation grows by
about a factor of 10 from S to C band and from C to X band.

### The ratio of attenuation to differential phase

The ratio $\alpha = A_H / K_{DP}$ is the coefficient of the attenuation correction based on the
differential phase. It depends on the wavelength, on the temperature of the rain, and a little on
the drop size distribution. For $\mu$ from 0 to 6 and $D_m$ from 1 to 2.5 mm:

```{code-cell} ipython3
rows = []
for band, wavelength in list(BANDS.items())[:4]:
    for temperature in (0.0, 20.0):
        t = raindrop_scattering(D, wavelength, temperature=temperature)
        ratios = []
        for mu in (0.0, 3.0, 6.0):
            for dm_ in (1.0, 1.5, 2.0, 2.5):
                n = gamma_dsd(dm_, mu)[0]
                ratios.append((n * t.ah.values).sum() / (n * t.kdp.values).sum())
        rows.append(
            {
                "band": band,
                "temperature (°C)": temperature,
                "min": min(ratios),
                "median": np.median(ratios),
                "max": max(ratios),
            }
        )
pd.DataFrame(rows).set_index(["band", "temperature (°C)"]).round(3)
```

The medians at 20 degrees Celsius, 0.018 (S), 0.083 (C) and 0.30 (X) dB per degree, are those
of the textbooks (Bringi and Chandrasekar 2001, Table 7.1: 0.017, 0.073 and 0.233 at 2.8, 5.5 and 9.3 GHz;
Ryzhkov et al. 2014 find 0.015 to 0.025 at S band for typical and tropical rain). The cold rain attenuates
about 1.7 times as much at S band. At Ka and W band $K_{DP}$ changes sign and the ratio is meaningless.

### The dual-frequency ratio of GPM

The difference of the reflectivities at Ku and Ka band, the dual-frequency ratio
$\mathrm{DFR} = Z_{Ku} - Z_{Ka}$ (dB), grows with the mean diameter because the drops leave the
Rayleigh regime at Ka band first. It is the quantity that the dual-frequency precipitation radar of GPM uses to
retrieve $D_m$. At 13.6 and 35.5 GHz:

```{code-cell} ipython3
ku = raindrop_scattering(D, 299.792458 / 13.6, temperature=10.0)
ka = raindrop_scattering(D, 299.792458 / 35.5, temperature=10.0)


def z_nadir(table, frequency, dsd):
    wavelength = 299.792458 / frequency
    m = water_refractive_index(wavelength, 10.0)
    k2 = abs((m * m - 1) / (m * m + 2)) ** 2
    return 10 * np.log10(wavelength**4 / (np.pi**5 * k2) * (dsd * table.sigma_h.values).sum(axis=1) * DD)


fig, ax = plt.subplots(figsize=(5.5, 3.6), layout="constrained")
for mu in (0.0, 3.0, 6.0):
    dsd = gamma_dsd(dm, mu)
    ax.plot(dm, z_nadir(ku, 13.6, dsd) - z_nadir(ka, 35.5, dsd), label=f"μ = {mu:g}")
ax.set(xlabel="mean diameter D$_m$ (mm)", ylabel="DFR = Z$_{Ku}$ − Z$_{Ka}$ (dB)")
ax.legend()
ax.grid(alpha=0.3)
```

$N_w$ cancels exactly in the difference, so DFR depends on the size of the drops only. It is slightly
negative (down to -1.3 dB) for $D_m$ below 1.3 mm, where the Mie resonance of the small drops at Ka
band is still lifting $Z_{Ka}$, and grows to 13 dB at 3.5 mm. The shape parameter shifts it
by 2 dB at $D_m = 2$ mm (7 dB for $\mu = 0$, 3 dB for $\mu = 6$), and the shift vanishes for the largest mean diameters.

## Radars of three bands observing the same squall line

The squall line of 30 March 2022 (PERiLS IOP2, Kosiba et al. 2024) was observed by the
WSR-88D of Columbus, Mississippi (KGWX, S band), the C-band Doppler on Wheels COW1 and the X-band
DOW7. The table gives, for gates with $\rho_{hv} > 0.97$ in the lowest sweep (S and C band) or the
lowest kilometre of the analysis grid (X band), the quartiles of $Z_{DR}$ and $K_{DP}$ in 5-dB bins of
$Z_H$ from 15 volumes of KGWX, 16 of COW1 and 105 of DOW7. (The summary is derived from the raw
radar files of the project; the files are not distributed with radarx.)

```{code-cell} ipython3
from io import StringIO

PERILS = pd.read_csv(
    StringIO(
        """radar,band,wavelength_mm,volumes,z_lo_dbz,z_hi_dbz,gates,zdr_p25,zdr_median,zdr_p75,kdp_p25,kdp_median,kdp_p75
KGWX,S,107.0,15,30,35,703127,0.812,1.031,1.25,0.018,0.063,0.11
KGWX,S,107.0,15,35,40,791140,1.062,1.281,1.5,0.07,0.118,0.174
KGWX,S,107.0,15,40,45,188648,1.219,1.469,1.719,0.184,0.328,0.597
KGWX,S,107.0,15,45,50,100889,1.469,1.719,2.0,0.636,0.971,1.385
KGWX,S,107.0,15,50,55,44765,1.656,1.906,2.188,1.257,1.69,2.217
COW1,C,54.0,16,30,35,277661,0.38,0.77,1.15,0.0,0.082,0.244
COW1,C,54.0,16,35,40,115389,0.43,0.85,1.283,0.015,0.25,0.645
COW1,C,54.0,16,40,45,44641,0.498,0.98,1.51,0.462,1.13,2.076
COW1,C,54.0,16,45,50,25433,0.691,1.23,1.83,1.27,2.581,3.844
COW1,C,54.0,16,50,55,4990,0.9,1.44,2.11,1.345,3.088,4.421
DOW7,X,31.9,105,30,35,1482898,0.308,0.643,0.94,0.107,0.222,0.407
DOW7,X,31.9,105,35,40,523555,0.528,0.899,1.154,0.223,0.404,0.876
DOW7,X,31.9,105,40,45,75196,0.522,0.874,1.272,1.239,2.091,3.061
DOW7,X,31.9,105,45,50,12573,0.957,1.229,1.526,2.188,3.341,4.334
DOW7,X,31.9,105,50,55,633,1.313,1.599,1.957,2.074,3.059,4.617"""
    )
)
PERILS.assign(z_bin=lambda d: d.z_lo_dbz.astype(str) + "-" + d.z_hi_dbz.astype(str))[
    ["radar", "band", "z_bin", "gates", "zdr_median", "kdp_p25", "kdp_median", "kdp_p75"]
].set_index(["radar", "band", "z_bin"])
```

The model of the previous sections can now be asked a precise question. Given only the median
$Z_H$ of a bin and its median $Z_{DR}$, a gamma distribution with $\mu = 3$ is fixed
($Z_{DR}$ gives $D_m$, then $Z_H$ gives $N_w$), and the T-matrix computed at the wavelength of the radar
predicts $K_{DP}$. This tests the backscatter at two polarizations (which give $D_m$ and $N_w$)
and the forward scattering ($K_{DP}$) at once, and does not depend on the calibration of
the phase.

```{code-cell} ipython3
def kdp_model(wavelength, mu=3.0, temperature=15.0):
    """Z (dBZ), Z_DR (dB) and K_DP at N_w = 1 for a grid of mean diameters."""
    t = raindrop_scattering(D, wavelength, temperature=temperature)
    m = water_refractive_index(wavelength, temperature)
    k2 = abs((m * m - 1) / (m * m + 2)) ** 2
    dm_grid = np.arange(0.4, 3.61, 0.02)
    n = gamma_dsd(dm_grid, mu, nw=1.0)
    zh = wavelength**4 / (np.pi**5 * k2) * (n * t.sigma_h.values).sum(axis=1) * DD
    zv = wavelength**4 / (np.pi**5 * k2) * (n * t.sigma_v.values).sum(axis=1) * DD
    return dm_grid, 10 * np.log10(zh), 10 * np.log10(zh / zv), (n * t.kdp.values).sum(axis=1) * DD


def predict_kdp(wavelength, z, zdr, **kwargs):
    dm_grid, z_unit, zdr_grid, kdp_unit = kdp_model(wavelength, **kwargs)
    dm_ = np.interp(zdr, zdr_grid, dm_grid)
    scale = 10 ** ((z - np.interp(dm_, dm_grid, z_unit)) / 10)  # N_w
    return scale * np.interp(dm_, dm_grid, kdp_unit)


PERILS["kdp_predicted"] = [
    float(predict_kdp(r.wavelength_mm, 0.5 * (r.z_lo_dbz + r.z_hi_dbz), r.zdr_median))
    for r in PERILS.itertuples()
]
PERILS["ratio"] = PERILS.kdp_median / PERILS.kdp_predicted
PERILS.assign(z_bin=lambda d: d.z_lo_dbz.astype(str) + "-" + d.z_hi_dbz.astype(str)).set_index(
    ["radar", "band", "z_bin"]
)[["kdp_median", "kdp_predicted", "ratio"]].round(2)
```

## Open radar data at S and C band

The same prediction for two volumes of the open-radar-data collection: the S-band radar of
Terrey Hills, Sydney (10.5 cm, three lowest sweeps) and the C-band ARMOR radar of Huntsville,
Alabama (5.5 GHz assumed, lowest sweep of five volumes), with the gates between 10 and 90 km (60 km)
from the radar. These are single storms, not a campaign, and the $Z_{DR}$ calibration of the
volumes is unknown.

```{code-cell} ipython3
def binned_medians(z, zdr, kdp, rho, mask, edges=np.arange(30, 56, 5)):
    ok = mask & np.isfinite(z) & np.isfinite(zdr) & np.isfinite(kdp) & (rho > 0.97)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = ok & (z >= lo) & (z < hi)
        if s.sum() >= 100:
            rows.append((lo, hi, int(s.sum()), np.median(zdr[s]), np.median(kdp[s])))
    return pd.DataFrame(rows, columns=["z_lo_dbz", "z_hi_dbz", "gates", "zdr_median", "kdp_median"])


def stack(sweeps, z, zdr, kdp, rho, limits):
    arrays = []
    for ds in sweeps:
        r = ds.range.values[None, :] / 1e3
        mask = np.broadcast_to((r > limits[0]) & (r < limits[1]), ds[z].shape)
        arrays.append((ds[z].values, ds[zdr].values, ds[kdp].values, ds[rho].values, mask))
    return [np.concatenate([a[i].ravel() for a in arrays]) for i in range(5)]


odim = xd.io.open_odim_datatree(DATASETS.fetch("71_20181220_060628.pvol.h5"))
sydney = binned_medians(
    *stack([odim[f"sweep_{i}"].to_dataset() for i in range(3)], "DBZH", "ZDR", "KDP", "RHOHV", (10, 90))
)
armor_sweeps = []
for name in (
    "RAW_NA_000_125_20080411181219",
    "RAW_NA_000_125_20080411181722",
    "RAW_NA_000_125_20080411182229",
    "RAW_NA_000_125_20080411182732",
    "RAW_NA_000_125_20080411183259",
):
    armor_sweeps.append(xd.io.open_iris_datatree(DATASETS.fetch(name))["sweep_0"].to_dataset())
armor = binned_medians(*stack(armor_sweeps, "DBZH", "ZDR", "KDP", "RHOHV", (10, 60)))

frames = []
for label, frame, wavelength in (("Sydney (S)", sydney, 104.8), ("ARMOR (C)", armor, 54.5)):
    frame = frame.copy()
    frame["kdp_predicted"] = [
        float(predict_kdp(wavelength, 0.5 * (r.z_lo_dbz + r.z_hi_dbz), r.zdr_median))
        for r in frame.itertuples()
    ]
    frame["ratio"] = frame.kdp_median / frame.kdp_predicted
    frame.insert(0, "radar", label)
    frames.append(frame)
OPEN = pd.concat(frames)
OPEN.assign(z_bin=lambda d: d.z_lo_dbz.astype(str) + "-" + d.z_hi_dbz.astype(str)).set_index(
    ["radar", "z_bin"]
)[["gates", "zdr_median", "kdp_median", "kdp_predicted", "ratio"]].round(2)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
styles = {"S": "o", "C": "s", "X": "^"}
for radar, frame in PERILS.groupby("radar"):
    band = frame.band.iloc[0]
    ax.plot(frame.kdp_predicted, frame.kdp_median, styles[band] + "-", label=f"{radar} ({band})")
for radar, frame in OPEN.groupby("radar"):
    ax.plot(frame.kdp_predicted, frame.kdp_median, "x--", label=radar)
ax.plot([0.01, 10], [0.01, 10], "k", lw=1)
ax.fill_between([0.01, 10], [0.005, 5], [0.02, 20], color="0.9", zorder=0, label="factor 2")
ax.set(
    xscale="log",
    yscale="log",
    xlim=(0.02, 10),
    ylim=(0.02, 10),
    xlabel="K$_{DP}$ predicted from median Z$_H$ and Z$_{DR}$ (° km$^{-1}$)",
    ylabel="median K$_{DP}$ observed (° km$^{-1}$)",
)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
```

Five radars at three bands agree with the prediction of the T-matrix model in the bins from 45 to 50 dBZ,
where the observed $K_{DP}$ is well above its noise (about 0.3 ° km$^{-1}$): the observed median is
0.87 to 1.32 times the predicted one (KGWX 1.32, COW1 1.17, DOW7 0.87, Sydney 0.94, ARMOR 1.16).
Below 40 dBZ the observed medians are small and set by the noise and the bias of the
estimate of $K_{DP}$, so they are no test (the ratios range from 0.9 to 4); above 50 dBZ
the observed $K_{DP}$ is lower than predicted (ratios of 0.3 to 0.8), as expected where the
heavy rain contains graupel or hail, the reflectivities at C and X band are attenuated, and the
phase is smoothed over small cores (the data do not say which of these dominates).
Two sensitivities of the prediction, computed next, set the scale of what can be concluded:

```{code-cell} ipython3
sensitivity = []
for r in PERILS[PERILS.z_lo_dbz >= 40].itertuples():
    z_mid = 0.5 * (r.z_lo_dbz + r.z_hi_dbz)
    base = float(predict_kdp(r.wavelength_mm, z_mid, r.zdr_median))
    by_mu = [float(predict_kdp(r.wavelength_mm, z_mid, r.zdr_median, mu=mu)) / base - 1 for mu in (1.0, 5.0)]
    by_zdr = [float(predict_kdp(r.wavelength_mm, z_mid, r.zdr_median + d)) / base - 1 for d in (-0.2, 0.2)]
    sensitivity.append(
        {
            "radar": r.radar,
            "z_bin": f"{r.z_lo_dbz}-{r.z_hi_dbz}",
            "mu = 1": by_mu[0],
            "mu = 5": by_mu[1],
            "Zdr - 0.2 dB": by_zdr[0],
            "Zdr + 0.2 dB": by_zdr[1],
        }
    )
pd.DataFrame(sensitivity).set_index(["radar", "z_bin"]).map("{:+.0%}".format)
```

The shape parameter changes the prediction by at most 1 to 2 %, because $Z_H$ and $Z_{DR}$ already fix
the part of the distribution that $K_{DP}$ measures; an error of 0.2 dB in the calibration of
$Z_{DR}$ changes it by 6 to 15 %, a good part of the departures of 13 to 32 % of the
ratios of the 45 to 50 dBZ bins from 1.

## What this shows

- The kernel gives the polarimetric variables of rain at every band of weather radar with the
  physics of the textbook and the code that the community has used for 30 years, in
  milliseconds per drop and in parallel.
- At S, C and X band the ratios between the variables ($A_H/K_{DP}$, $Z_{DR}$ of a distribution)
  agree with the literature, with resonance of the large drops at C and X band; at Ku band
  the polarimetric variables still work, and at Ka and W band only the reflectivity does (and the
  dual-frequency ratio of Ku and Ka, which retrieves the size of the drops).
- Real volumes are consistent with the model within the limits of their data quality at the three
  bands, but they cannot validate the scattering to better than tens of percent: use the textbook
  and Mie comparisons of the other two notebooks for that.

## References

- Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler Weather Radar*.
  Cambridge University Press, <https://doi.org/10.1017/CBO9780511541094>
- Brandes, E. A., G. Zhang, and J. Vivekanandan, 2002: Experiments in rainfall
  estimation with a polarimetric radar in a subtropical environment. *J. Appl. Meteor.*,
  **41**, 674–685, <https://doi.org/10.1175/1520-0450(2002)041<0674:EIREWA>2.0.CO;2>
- Kosiba, K. A., and Coauthors, 2024: The Propagation, Evolution, and Rotation in Linear
  Storms (PERiLS) Project. *Bull. Amer. Meteor. Soc.*, **105**, E1768–E1799,
  <https://doi.org/10.1175/BAMS-D-22-0064.1>
- Ryzhkov, A., M. Diederich, P. Zhang, and C. Simmer, 2014: Potential utilization of
  specific attenuation for rainfall estimation, mitigation of partial beam blockage, and
  radar networking. *J. Atmos. Oceanic Technol.*, **31**, 599–619,
  <https://doi.org/10.1175/JTECH-D-13-00038.1>
