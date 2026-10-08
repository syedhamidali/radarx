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

# Disdrometers

Laser disdrometers such as the OTT Parsivel count the drops that fall
through a 180 × 30 mm laser sheet in 32 size and 32 fall-speed classes.
radarx reads these spectra into xarray and turns them into drop size
distributions and radar variables:

- `radarx.io.read_parsivel` reads Parsivel telegrams logged by PIPS
  (Portable In situ Precipitation Stations), Campbell Scientific TOA5 files
  or plain text, and `radarx.io.read_pips_netcdf` reads PIPS netCDF files.
  Both return `counts` on `(time, velocity, diameter)` with the station
  position as coordinates;
- `disdrometer_qc` removes particles far from the raindrop fall speed
  (margin fallers, splashing, the strong-wind artifact of Friedrich et al.
  2013), and `raupach_berne_correction` applies the correction of Raupach
  and Berne (2015);
- `number_concentration` gives $N(D)$, `dsd_moments` the bulk quantities,
  `fit_gamma` gamma fits by moments or truncated moments, and
  `radar_from_dsd` the polarimetric radar variables from the T-matrix
  tables of `radarx.retrieve.dsd`;
- `match_radar` pairs the disdrometer with the radar gate above it.

A compiled kernel handles the per-spectrum work (velocity shifts, $N(D)$,
gamma fits) for all spectra at once.

```{code-cell} ipython3
import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xradar as xd
from open_radar_data import DATASETS

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.io import parsivel_classes, read_parsivel
from radarx.retrieve import (
    disdrometer_qc,
    fit_gamma,
    match_radar,
    number_concentration,
    raupach_berne_correction,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
```

## A simulated Parsivel record

To keep this example self-contained, we simulate 30 minutes of 10 s
Parsivel telegrams for rain whose gamma DSD intensifies and then decays.
Drops are drawn from the DSD in the sampling volume of each size class,
given fall speeds scattered around the terminal fall speed, and mixed with
splashing drops (small and fast) and slow, large particles, the artifacts
the quality control should remove. The telegrams are written in the
format a PIPS logger stores.

```{code-cell} ipython3
rng = np.random.default_rng(42)
cls = parsivel_classes()
d, dd = cls.diameter.values, cls.bin_width.values
vlo, vup = cls.velocity_lower.values, cls.velocity_upper.values
area = 180e-6 * (30.0 - d / 2)  # effective sampling area, m2
vt = np.maximum(9.65 - 10.3 * np.exp(-0.6 * d), 0.0)

ntime = 180
t = np.arange(ntime)
lam = 3.5 - 1.6 * np.exp(-(((t - 70) / 30.0) ** 2))  # broader DSD in the core
mu = 1.0 + 2.0 * np.exp(-(((t - 70) / 40.0) ** 2))
n0 = 6000.0 * lam ** (mu + 1) / 3.5 ** (mu + 1) * (0.2 + np.exp(-(((t - 70) / 35.0) ** 2)))

lines = []
for k in range(ntime):
    nd = n0[k] * d ** mu[k] * np.exp(-lam[k] * d)
    nd[d < 0.25] = 0.0
    expected = nd * area * dd * 10.0 * vt  # drops through the beam in 10 s
    counts = np.zeros((32, 32), int)
    for i in np.flatnonzero(expected > 0):
        v = rng.normal(vt[i], 0.08 * vt[i] + 0.2, rng.poisson(expected[i]))
        j = np.clip(np.searchsorted(vup, v), 0, 31)
        np.add.at(counts[:, i], j, 1)
    counts[rng.integers(22, 28), rng.integers(2, 5)] += rng.poisson(15)  # splashing
    counts[rng.integers(0, 4), rng.integers(19, 23)] += rng.poisson(2)  # slow and large
    hms = f"00:{k // 6:02d}:{(k % 6) * 10:02d}"
    head = f"304545;{0:08.3f};0000.00;00.000;00010;14000;{counts.sum():05d};22;11.6;{hms};31.03.2022"
    lines.append(head + ";" + ";".join(f"{c:03d}" for c in counts.ravel()) + ";")
with open("parsivel_telegrams.txt", "w") as f:
    f.write("\n".join(lines))

pars = read_parsivel(
    "parsivel_telegrams.txt", station="SIM", latitude=33.6, longitude=-101.8, altitude=990.0
)
pars
```

The velocity-diameter histogram of all records shows the drops along the
fall-speed curve, the splashing drops above it and the slow, large
particles below it. The relative quality control (default ±60 % of the
terminal fall speed, drops up to 8 mm, the two unmeasured classes removed)
keeps the band around the curve:

```{code-cell} ipython3
qc = disdrometer_qc(pars)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, name, title in zip(axes, ("counts", "counts_qc"), ("raw", "quality controlled")):
    total = qc[name].sum("time").where(lambda x: x > 0)
    ax.pcolormesh(
        np.r_[cls.diameter_lower, cls.diameter_upper[-1]],
        np.r_[cls.velocity_lower, cls.velocity_upper[-1]],
        np.log10(total),
        cmap="viridis",
    )
    ax.plot(d, vt, "r-", lw=1, label="terminal fall speed")
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 12)
    ax.set_xlabel("diameter (mm)")
    ax.set_title(title)
axes[0].set_ylabel("fall speed (m s$^{-1}$)")
axes[0].legend(loc="lower right")
plt.tight_layout()
```

## N(D), moments, gamma fits and radar variables

`ds.radarx.disdrometer()` runs the quality control, computes $N(D)$ and the
bulk quantities, fits gamma DSDs (by default the 2-4-6 method of moments,
`MM246`, and the 2-4-6 truncated moments, `TMM246`) and computes S-band
radar variables:

```{code-cell} ipython3
out = pars.radarx.disdrometer(fits=("MM246", "TMM246", "MM346"), band="S")
out
```

```{code-cell} ipython3
fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
out.ND.where(out.ND > 0).pipe(np.log10).plot(
    ax=axes[0], x="time", y="diameter", cmap="turbo", vmin=0, vmax=4,
    cbar_kwargs={"label": "log$_{10}$ N(D)"},
)
axes[0].set_ylim(0, 6)
out.DBZH.plot(ax=axes[1], label="$Z_H$ (T-matrix)")
out.DBZ_RAYLEIGH.plot(ax=axes[1], ls="--", label="$Z$ (Rayleigh)")
axes[1].set_ylabel("dBZ")
axes[1].legend()
out.ZDR.plot(ax=axes[2])
axes[2].set_ylabel("$Z_{DR}$ (dB)")
truth = (4 + mu) / lam
axes[3].plot(out.time, truth, "k-", label="true $D_m$")
out.DM.plot(ax=axes[3], label="measured $D_m$")
out.DM_TMM246.plot(ax=axes[3], ls="--", label="$D_m$ of TMM246 fit")
axes[3].set_ylabel("$D_m$ (mm)")
axes[3].legend()
for ax in axes:
    ax.set_title("")
    ax.set_xlabel("")
plt.tight_layout()
```

With only a few large drops in a 10 s record, the fitted shapes are noisy;
averaging the spectra over 1 min steadies them. The untruncated 2-4-6 fit
overestimates $\mu$ where the largest drops of the DSD are missing from a
record (sampling); the truncated fit removes this high bias but scatters
more in light rain, and the middle moments (2-3-4) give the smallest errors, as Cao and Zhang (2009)
found with simulated spectra:

```{code-cell} ipython3
nd_1min = out.ND.resample(time="60s").mean()
mu_true = xr.DataArray(mu, dims="time", coords={"time": out.time}).resample(time="60s").mean()
fig, ax = plt.subplots(figsize=(10, 3.5))
mu_true.plot(ax=ax, color="k", lw=2, label="true")
for name, moments, truncated in (
    ("MM246", (2, 4, 6), False),
    ("TMM246", (2, 4, 6), True),
    ("MM234", (2, 3, 4), False),
):
    fit = fit_gamma(nd_1min, moments=moments, truncated=truncated)
    err = float(np.sqrt(((fit.MU - mu_true) ** 2).mean()))
    fit.MU.plot(ax=ax, marker=".", lw=0.8, label=f"{name} (rms error {err:.2f})")
ax.set_ylabel("$\\mu$")
ax.set_title("1-min spectra")
ax.legend()
```

## Raupach and Berne (2015) correction

The correction shifts the velocities of each size class onto the terminal
fall speed, removes implausible particles and scales $N(D)$ with factors
calibrated against a 2D video disdrometer for classes of the rain intensity
reported by the Parsivel (here we set it from the measured spectra). The
factors were trained in southern France and mainly reduce the number of
small drops. As in the paper, the velocities are shifted before the
filter, so on raw counts the splashing drops pull the mean velocity of the
small size classes up and the shift moves those whole classes down. With
`counts="counts_qc"` the correction starts from the quality-controlled
counts instead:

```{code-cell} ipython3
pars["rain_rate_instrument"] = out.RAIN_RATE.fillna(0.0)
qc = disdrometer_qc(pars)
rb_raw = number_concentration(raupach_berne_correction(pars, instrument="parsivel2"))
rb_qc = number_concentration(
    raupach_berne_correction(qc, instrument="parsivel2", counts="counts_qc")
)

fig, ax = plt.subplots(figsize=(6.5, 4))
sel = slice("2022-03-31T00:10", "2022-03-31T00:14")
k = (t >= 60) & (t < 90)
truth = (n0[k, None] * d ** mu[k, None] * np.exp(-lam[k, None] * d)).mean(0)
ax.plot(d, truth, "k-", lw=2, label="true")
out.ND.sel(time=sel).mean("time").plot(ax=ax, label="relative QC", yscale="log")
rb_raw.sel(time=sel).mean("time").plot(ax=ax, label="Raupach and Berne, raw counts")
rb_qc.sel(time=sel).mean("time").plot(ax=ax, label="Raupach and Berne, after QC")
ax.set_xlim(0, 6)
ax.set_ylim(1, 1e4)
ax.set_title("")
ax.set_ylabel("N(D) (m$^{-3}$ mm$^{-1}$)")
ax.legend()
```

## Matching the radar gate above the disdrometer

`match_radar` finds the gate of a sweep nearest above the instrument (or
averages the gates within `radius`), records its time, height and distance,
and averages the disdrometer spectra over a window centred on the radar time
(optionally delayed by the fall time of the drops from the beam,
`delay="fall"`). The radar variables computed from the averaged spectra get
the suffix `_disdrometer`.

Here we place the simulated station under the KLBB radar (Lubbock, Texas),
in a rain shower 25 km north-east of it, and give it the time of that sweep:

```{code-cell} ipython3
file = DATASETS.fetch("KLBB20160601_150025_V06")
klbb = xd.io.open_nexradlevel2_datatree(file, sweep=[0])
sweep = klbb["sweep_0"].to_dataset(inherit=False)
sweep["DBZH"] = sweep.DBZH.where(sweep.DBZH > -32.0)
sweep["ZDR"] = sweep.ZDR.where(sweep.ZDR > -12.9)
klbb["sweep_0"] = sweep
sweep = klbb["sweep_0"].to_dataset()

near = sweep.sel(range=slice(15e3, 40e3))
ia, ir = np.unravel_index(int(near.DBZH.fillna(-99).argmax()), near.DBZH.shape)
az, rng_ = float(near.azimuth[ia]), float(near.range[ir])
el = float(near.elevation[ia])
lat0, lon0 = float(klbb["latitude"]), float(klbb["longitude"])
re = 4 / 3 * 6371000.0
h = np.sqrt(rng_**2 + re**2 + 2 * rng_ * re * np.sin(np.deg2rad(el))) - re
s = re * np.arcsin(rng_ * np.cos(np.deg2rad(el)) / (re + h)) / 6371000.0
p0, a = np.deg2rad(lat0), np.deg2rad(az)
lat = np.rad2deg(np.arcsin(np.sin(p0) * np.cos(s) + np.cos(p0) * np.sin(s) * np.cos(a)))
lon = lon0 + np.rad2deg(np.arctan2(np.sin(a) * np.sin(s) * np.cos(p0),
                                   np.cos(s) - np.sin(p0) * np.sin(np.deg2rad(lat))))

t_radar = sweep.time.values[ia]
station = pars.assign_coords(
    latitude=lat,
    longitude=lon,
    altitude=float(klbb["altitude"]),
    time=pars.time.values - pars.time.values[70] + t_radar,  # rain core under the radar
)
pairs = match_radar(station, klbb, fields=["DBZH", "ZDR"], window="60s", delay="fall")
pairs[["DBZH", "DBZH_disdrometer", "ZDR", "ZDR_disdrometer", "beam_height",
       "height_above_ground", "gate_distance", "delay", "n_records"]]
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6, 5.5))
azr = np.deg2rad(sweep.azimuth.values)[:, None]
rkm = sweep.range.values[None, :] / 1e3
pm = ax.pcolormesh(
    rkm * np.sin(azr),
    rkm * np.cos(azr),
    sweep.DBZH.transpose("azimuth", "range").values,
    cmap="turbo",
    vmin=-10,
    vmax=70,
    shading="auto",
)
ax.plot(rng_ / 1e3 * np.sin(np.deg2rad(az)), rng_ / 1e3 * np.cos(np.deg2rad(az)), "k*", ms=14,
        label="disdrometer")
ax.set_xlim(-60, 60)
ax.set_ylim(-60, 60)
ax.set_aspect("equal")
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
ax.legend()
plt.colorbar(pm, label="$Z_H$ (dBZ)")
```

With real deployments, `match_radar` takes a list of volumes (one per scan)
and returns one row per volume, e.g. to compare the radar $Z_H$ and
$Z_{DR}$ with those of the disdrometer spectra or to evaluate radar DSD
retrievals (`radarx.retrieve.dsd`) at the ground.

## References

- Friedrich, K., S. Higgins, F. J. Masters, and C. R. Lopez, 2013:
  Articulating and stationary PARSIVEL disdrometer measurements in conditions
  with strong winds and heavy rainfall. *J. Atmos. Oceanic Technol.*, **30**,
  2063–2080, https://doi.org/10.1175/JTECH-D-12-00254.1
- Raupach, T. H., and A. Berne, 2015: Correction of raindrop size
  distributions measured by Parsivel disdrometers, using a two-dimensional
  video disdrometer as a reference. *Atmos. Meas. Tech.*, **8**, 343–365,
  https://doi.org/10.5194/amt-8-343-2015
- Ulbrich, C. W., and D. Atlas, 1998: Rainfall microphysics and radar
  properties: Analysis methods for drop size spectra. *J. Appl. Meteor.*,
  **37**, 912–923,
  https://doi.org/10.1175/1520-0450(1998)037<0912:RMARPA>2.0.CO;2
- Cao, Q., and G. Zhang, 2009: Errors in estimating raindrop size
  distribution parameters employing disdrometer and simulated raindrop
  spectra. *J. Appl. Meteor. Climatol.*, **48**, 406–425,
  https://doi.org/10.1175/2008JAMC2026.1
