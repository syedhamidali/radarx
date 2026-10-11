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

# Ensembles and Uncertainty of the Retrievals

A retrieval gives one number per gate. How well is it known? The measurements
have random errors (the estimate of a radar variable from a few dozen pulses
fluctuates), the radar has a calibration error, and the retrievals lean on
assumptions about the raindrop size distribution (DSD). `radarx.retrieve.ensemble_retrieval`
(or `.radarx.ensemble_retrieval()`) propagates all three through the retrievals of radarx by
Monte Carlo: every member is a copy of the radar fields with random errors, the
retrieval runs on each member, and the result holds the mean, the median, the
standard deviation and percentiles of every product at every gate.

- `perturb_radar` makes the members (calibration offsets and measurement noise);
- `measurement_error` gives the standard deviation of the noise, from the number of
  pulses, the signal-to-noise ratio (SNR), the correlation coefficient and the
  spectrum width (Melnikov and Zrnic 2004);
- `ensemble_retrieval` runs `estimate_kdp`, `correct_attenuation`, `dsd`,
  `dsd_bayesian`, `hid` and the four rain rate relations on the members, or any
  function you pass;
- `ensemble_statistics` reduces any array with a `member` dimension;
- `relation_scatter` gives the scatter of the rain rate relations that comes from the DSD
  and sets the default width of the prior part.

The random numbers are counter based (Philox-4x64-10): member $m$ of a sweep
depends only on the seed, the sweep and $m$, not on the number of threads, and the
compiled kernel and the NumPy implementation draw the same 64-bit words. The
statistics run in the same compiled kernel; the retrievals are the existing,
threaded ones.

```{code-cell} ipython3
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xradar as xd
from open_radar_data import DATASETS
from scipy.interpolate import RegularGridInterpolator

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.retrieve import (
    ensemble_retrieval,
    ensemble_statistics,
    forward_grid,
    measurement_error,
    perturb_radar,
    relation_scatter,
    snr,
)
from radarx.retrieve.dsd_bayes import DM_AXIS, MU_AXIS

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="The input coordinates to pcolormesh")
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 100, "font.size": 9.5})
```

## The error of a radar estimate

For simultaneous transmission and reception, Melnikov and Zrnic (2004) derive the
standard deviation of $Z_{DR}$, $\Phi_{DP}$, $\rho_{hv}$ and the pulse-pair velocity from the
number of samples $M$, the SNR of the two channels, $\rho_{hv}$ and the number of
independent samples, which depends on the spectrum width. `measurement_error`
evaluates the formulas on the fields of a sweep. Here is the dependence on SNR for
$M=64$ pulses and uncorrelated samples, with a Monte Carlo of the estimators
(complex Gaussian signals) as a check:

```{code-cell} ipython3
def simulate(m, rho, snr_db, nrep=20000, seed=0):
    """SD of ZDR (dB), PHIDP (degrees) and RHOHV of the SHV estimators, by simulation."""
    rng = np.random.default_rng(seed)
    s = 10 ** (snr_db / 10)
    g = lambda: (rng.standard_normal((nrep, m)) + 1j * rng.standard_normal((nrep, m))) / np.sqrt(2)
    common, other = g(), g()
    h = np.sqrt(s) * common + g()
    v = np.sqrt(s) * (rho * common + np.sqrt(1 - rho**2) * other) + g()
    sh, sv = (abs(h) ** 2).mean(1) - 1, (abs(v) ** 2).mean(1) - 1
    cross = (h * v.conj()).mean(1)
    ok = (sh > 0) & (sv > 0)
    return (
        (10 * np.log10(sh[ok] / sv[ok])).std(),
        np.degrees(np.angle(cross)).std(),
        (abs(cross[ok]) / np.sqrt(sh[ok] * sv[ok])).std(),
    )


snr_db = np.arange(4.0, 31.0, 2.0)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
for rho, color in ((0.99, "#0072B2"), (0.95, "#D55E00")):
    ds = xr.Dataset(
        {
            "DBZH": (("azimuth", "range"), np.full((1, snr_db.size), 30.0)),
            "ZDR": (("azimuth", "range"), np.zeros((1, snr_db.size))),
            "PHIDP": (("azimuth", "range"), np.full((1, snr_db.size), 20.0)),
            "RHOHV": (("azimuth", "range"), np.full((1, snr_db.size), rho)),
            "SNR": (("azimuth", "range"), snr_db[None]),
        }
    )
    # a spectrum width equal to the Nyquist velocity: independent samples, M_I = M
    err = measurement_error(ds, snr="SNR", n_samples=64, spectrum_width=25.0, nyquist=25.0)
    sim = np.array([simulate(64, rho, s, seed=int(s)) for s in snr_db])
    for ax, name, k, unit in zip(axes, ("ZDR", "PHIDP", "RHOHV"), range(3), ("dB", "degrees", "")):
        ax.plot(snr_db, err[f"{name}_error"].values[0], color=color, label=f"formula, $\\rho_{{hv}}$={rho}")
        ax.plot(snr_db, sim[:, k], "o", ms=4, color=color, mfc="none", label="simulation")
        ax.set(xlabel="SNR (dB)", ylabel=f"standard deviation of {name} ({unit})".replace(" ()", ""), yscale="log")
axes[0].legend(frameon=False)
plt.show()
```

For a real spectrum width the number of independent samples is smaller than $M$ and the
reflectivity itself has an error of about $1.2$ dB at $M=64$ and $\sigma_v=3$ m s$^{-1}$:

```{code-cell} ipython3
one = xr.Dataset({k: (("azimuth", "range"), np.full((1, 1), v)) for k, v in
                  dict(DBZH=30.0, ZDR=0.5, PHIDP=20.0, RHOHV=0.99, VRADH=5.0, SNR=20.0).items()})
for sigma_v in (1.0, 3.0, 6.0):
    e = measurement_error(one, snr="SNR", spectrum_width=sigma_v, nyquist=25.0)
    print(f"sigma_v {sigma_v:3.0f} m/s: " + ", ".join(f"{v[:-6]} {float(e[v].squeeze()):.3f} {e[v].attrs['units']}" for v in e.data_vars))
```

## A sweep with a known truth

To see whether the spread of the ensemble is the size of the actual error, we need a
case where the truth is known. The sweep below has rain cells with a prescribed DSD
($D_m$, $\mu$, $N_w$), from which the forward model gives $Z_H$, $Z_{DR}$, $K_{DP}$ and the rain
rate; the differential phase is $2\int K_{DP}\,dr$.

```{code-cell} ipython3
def truth_sweep(nray=24, ngate=300, dr=250.0, seed=3):
    rng = np.random.default_rng(seed)
    az = np.linspace(0.5, 359.5, nray)
    rkm = (np.arange(ngate) + 0.5) * dr / 1000.0
    A, R = np.meshgrid(az, rkm, indexing="ij")
    shape = np.zeros_like(A)
    for _ in range(4):
        a0, r0 = rng.uniform(0, 360), rng.uniform(15, 60)
        da = np.minimum(np.abs(A - a0), 360 - np.abs(A - a0))
        shape += np.exp(-0.5 * (da / 25) ** 2 - 0.5 * ((R - r0) / 9) ** 2) * rng.uniform(0.6, 1)
    shape = np.clip(shape, 0, 1)
    rain = shape > 0.08
    dm = np.clip(0.9 + 1.6 * shape + 0.15 * rng.standard_normal(A.shape), 0.6, 3.2)
    lognw = 3.0 + 1.3 * (1 - shape) + 0.2 * rng.standard_normal(A.shape)
    mu = np.clip(2.0 + 3.0 * rng.random(A.shape), -0.4, 9.0)
    fwd = forward_grid("S", 20.0)
    axes_ = (DM_AXIS[0] + DM_AXIS[1] * np.arange(DM_AXIS[2]), MU_AXIS[0] + MU_AXIS[1] * np.arange(MU_AXIS[2]))
    pts = np.stack([dm.ravel(), mu.ravel()], axis=1)
    on_grid = lambda name: RegularGridInterpolator(axes_, fwd[name].transpose("dm", "mu").values)(pts).reshape(A.shape)
    nw = 10.0**lognw
    kdp = on_grid("KDP") * nw
    fields = {
        "DBZH": np.where(rain, on_grid("DBZH") + 10 * lognw, -5.0),
        "ZDR": np.where(rain, on_grid("ZDR"), 0.0),
        "PHIDP": 2 * np.cumsum(np.where(rain, kdp, 0.0), axis=1) * dr / 1000,
        "RHOHV": np.where(rain, 0.99, 0.6),
    }
    fields["SNR"] = fields["DBZH"] - (-40.0 + 20 * np.log10(R))
    dims = ("azimuth", "range")
    truth = {"KDP": kdp, "DM": dm, "LOG10_NW": lognw, "RAIN_RATE": on_grid("RAIN_RATE") * nw}
    ds = xr.Dataset(
        {k: (dims, v) for k, v in {**fields, **{"T_" + k: np.where(rain, v, np.nan) for k, v in truth.items()}}.items()},
        coords={"azimuth": ("azimuth", az, {"units": "degrees"}),
                "range": ("range", rkm * 1000, {"units": "m"}),
                "elevation": ("azimuth", np.full(nray, 0.5))},
    )
    ds["DBZH"].attrs["units"] = "dBZ"
    return ds, rain


truth, rain = truth_sweep()
radar = truth[["DBZH", "ZDR", "PHIDP", "RHOHV", "SNR"]]
print(f"{int(rain.sum())} rain gates of {rain.size}; "
      f"R up to {float(truth.T_RAIN_RATE.max()):.0f} mm/h, KDP up to {float(truth.T_KDP.max()):.1f} deg/km")
```

The "observed" sweeps are drawn from the truth with the error model (one calibration
offset and the noise of every gate; `perturb_radar` with a seed that the ensembles do
not use). For each observed sweep an ensemble is built around it. Two things are
compared with the ensemble: the retrieval applied to the *true* fields (the part of the
error that comes from the measurements and calibration; no prior) and the true DSD
quantity (which adds the error of the retrieval itself; with the prior).

```{code-cell} ipython3
N_REAL, MEMBERS = 10, 24
observed = perturb_radar(radar, members=N_REAL, seed=1234, snr="SNR")
observed = [observed.isel(member=k, drop=True).assign(SNR=radar.SNR) for k in range(N_REAL)]
quiet = dict(calibration=None, noise=False, prior=False)


def compare(retrieval, variable, truth_name, **kw):
    """Errors of the ensemble median and its spread, without and with the prior."""
    reference = ensemble_retrieval(radar, retrieval, members=2, band="S", **quiet, **kw)[f"{variable}_mean"].values
    parts = {"measurements": ([], [], []), "with prior": ([], [], [])}
    for k, ds in enumerate(observed):
        for label, prior in (("measurements", False), ("with prior", True)):
            ens = ensemble_retrieval(ds, retrieval, members=MEMBERS, seed=100 + k, band="S",
                                     noise={"snr": "SNR"}, prior=prior, **kw)
            ref = reference if label == "measurements" else truth[truth_name].values
            good = rain & np.isfinite(ens[f"{variable}_std"].values) & np.isfinite(ref) & np.isfinite(reference)
            err = (ens[f"{variable}_median"].values - ref)[good]
            lo, hi = ens[f"{variable}_p05"].values[good], ens[f"{variable}_p95"].values[good]
            parts[label][0].append(err)
            parts[label][1].append(ens[f"{variable}_std"].values[good])
            parts[label][2].append((ref[good] >= lo) & (ref[good] <= hi))
    return {k: tuple(np.concatenate(x) for x in v) for k, v in parts.items()}


cases = {
    "KDP": ("kdp", "KDP", "T_KDP", {}),
    "R(Z)": ("rain_rate_z", "RAIN_RATE_Z", "T_RAIN_RATE", {}),
    "R(KDP)": ("rain_rate_kdp", "RAIN_RATE_KDP", "T_RAIN_RATE", {}),
    "R(Z, ZDR)": ("rain_rate_z_zdr", "RAIN_RATE_Z_ZDR", "T_RAIN_RATE", {}),
    "Dm (dsd)": ("dsd", "DM", "T_DM", {}),
    "log10 Nw (dsd)": ("dsd", "LOG10_NW", "T_LOG10_NW", {}),
}
t0 = time.time()
results = {name: compare(*args[:3], **args[3]) for name, args in cases.items()}
print(f"{len(cases)} ensembles x {N_REAL} observed sweeps in {time.time() - t0:.0f} s")
print(f"{'':16s}{'measurements only':>34s}{'with the DSD prior, against the true DSD':>46s}")
print(f"{'':16s}{'RMS error/RMS spread':>24s}{'p05-p95 holds':>14s}{'RMS error/RMS spread':>26s}{'p05-p95 holds':>14s}")
for name, parts in results.items():
    cells = []
    for label in ("measurements", "with prior"):
        err, spread, inside = parts[label]
        cells += [np.sqrt(np.mean(err**2) / np.mean(spread**2)), inside.mean()]
    print(f"{name:16s}{cells[0]:24.2f}{cells[1]:14.0%}{cells[2]:26.2f}{cells[3]:14.0%}")
```

A reliable ensemble has a ratio of 1 and holds the truth between its 5th and 95th
percentile in 90 % of the gates. The figure bins the gates by the spread of the ensemble
and compares it with the RMS error in each bin:

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
for ax, label in zip(axes, ("measurements", "with prior")):
    for (name, parts), color in zip(results.items(), plt.cm.tab10.colors):
        err, spread, _ = parts[label]
        scale = np.sqrt(np.mean(spread**2))  # each product in units of its own RMS spread
        edges = np.quantile(spread, np.linspace(0, 1, 9))
        bins = np.clip(np.digitize(spread, edges[1:-1]), 0, 7)
        s = [np.sqrt(np.mean(spread[bins == b] ** 2)) / scale for b in range(8)]
        e = [np.sqrt(np.mean(err[bins == b] ** 2)) / scale for b in range(8)]
        ax.loglog(s, e, "o-", ms=4, color=color, label=name)
    line = np.geomspace(0.05, 20, 2)
    ax.plot(line, line, "k--", lw=1)
    ax.set(xlabel="RMS ensemble spread (units of the overall RMS spread)", ylabel="RMS error (same units)", title=label)
axes[0].legend(frameon=False, fontsize=8)
plt.show()
```

The measurement part is reliable for the retrievals that act on $Z_H$ and $Z_{DR}$ (the
rain rates, $D_m$, $N_w$). The estimate of $K_{DP}$ comes out with a spread larger than its
error: `estimate_kdp` filters the phase and chooses its window from the noise of the
phase, so the ensemble member that has more noise than the observation is smoothed more.
With the prior the rain rate from $Z_H$ is reliable; the $Z_H$-$Z_{DR}$ relation and $K_{DP}$
have errors that are larger than the DSD scatter of the generic prior (the errors of the
retrieval method itself, here a mismatch between the default relation coefficients and
the forward model that made the truth). `relation_scatter` shows where the prior
widths come from:

```{code-cell} ipython3
scatter = relation_scatter("S")
print(scatter.ln_sd.to_series().round(3).to_dict(), "| A/KDP:", round(float(scatter.alpha), 4),
      "dB/deg, ln-sd", round(float(scatter.alpha_ln_sd), 3))
for band in ("C", "X"):
    other = relation_scatter(band)
    print(band, other.ln_sd.to_series().round(3).to_dict(), "| A/KDP ln-sd", round(float(other.alpha_ln_sd), 3))
```

A standard deviation of $\ln R$ of 0.46 for $R(Z)$ is a factor of 1.6 at one standard deviation:
the DSD alone, with a perfectly calibrated radar, limits $R(Z)$ more than the calibration
($1$ dB is 18 %) and the noise do. $R(Z, Z_{DR})$ and $R(A)$ are much less sensitive,
as Ryzhkov et al. (2014) find.

The Bayesian retrieval has its own uncertainty, the posterior. The spread of the ensemble
of its posterior means is the effect of the errors of the measurements and of the prior on
the answer; the total variance is the mean posterior variance plus the variance of the means.
`ensemble_retrieval` returns it as `<name>_total_std`:

```{code-cell} ipython3
def bayes_scores(n_real=6):
    z_ens, z_tot = [], []
    for k, ds in enumerate(observed[:n_real]):
        ens = ensemble_retrieval(ds, "dsd_bayesian", members=16, seed=100 + k, band="S", noise={"snr": "SNR"})
        good = rain & np.isfinite(ens.DM_total_std.values) & np.isfinite(truth.T_DM.values) & (ens.DM_std.values > 0)
        err = (ens.DM_mean.values - truth.T_DM.values)[good]
        z_ens.append(err / ens.DM_std.values[good])
        z_tot.append(err / ens.DM_total_std.values[good])
    return [np.sqrt(np.mean(np.concatenate(z) ** 2)) for z in (z_ens, z_tot)], ens


(z_ens, z_tot), bayes_ens = bayes_scores()
print(f"Dm error / spread: {z_ens:.2f} with the ensemble spread alone, {z_tot:.2f} with the posterior included "
      f"(1 for a reliable estimate)")
```

## Real data: the S-band volume of Lubbock

The 0.5° sweep of the WSR-88D at Lubbock, Texas (KLBB), of 1 June 2016. The NEXRAD
no-data codes are masked, the SNR comes from the noise level that
`radarx.retrieve.snr` estimates from the reflectivity itself, and the number of pulses is
the default of 64 (the real number depends on the volume coverage pattern). The
Nyquist velocity enters through the spectrum width; this file has no
`nyquist_velocity`, so the default of 25 m s$^{-1}$ is used.

```{code-cell} ipython3
path = DATASETS.fetch("KLBB20160601_150025_V06")
klbb = xd.io.open_nexradlevel2_datatree(path, sweep=[0]).xradar.georeference()
sweep = klbb["sweep_0"].to_dataset()
for var, limit in (("DBZH", -32.0), ("ZDR", -12.9), ("RHOHV", 0.21)):
    sweep[var] = sweep[var].where(sweep[var] > limit)
sweep = sweep.assign(SNR=snr(sweep))
sweep = sweep.isel(range=slice(0, 700))  # the first 175 km
sweep = sweep.assign(TEMP=25.0 - 6.5e-3 * sweep["z"].fillna(0.0))  # a standard-atmosphere temperature (degC)
meteo = (sweep.RHOHV > 0.9) & (sweep.DBZH > 10)  # gates of the maps below
print(f"{sweep.sizes['azimuth']} rays x {sweep.sizes['range']} gates, elevation {float(sweep.elevation.mean()):.2f} deg")

noise = {"snr": "SNR"}
t0 = time.time()
kdp_ens = ensemble_retrieval(sweep, "kdp", members=24, seed=1, noise=noise, keep_members=False)
t_kdp = time.time() - t0
t0 = time.time()
rr_kdp = ensemble_retrieval(sweep, "rain_rate_kdp", members=24, seed=1, noise=noise, band="S")
rr_zdr = ensemble_retrieval(sweep, "rain_rate_z_zdr", members=24, seed=1, noise=noise, band="S")
print(f"KDP: 24 members in {t_kdp:.1f} s; the two rain rates: {time.time() - t0:.1f} s")
```

```{code-cell} ipython3
def show(axes, ens, name, label, vmax, rel_max=1.0):
    """Median, standard deviation and relative standard deviation of a product."""
    x, y = ens["x"] / 1000, ens["y"] / 1000
    median, std = ens[f"{name}_median"].where(meteo), ens[f"{name}_std"].where(meteo)
    panels = [
        (median, f"{label}: median", dict(cmap="turbo", vmin=0, vmax=vmax)),
        (std, f"{label}: standard deviation", dict(cmap="magma_r", vmin=0, vmax=vmax / 3)),
        ((std / median).where(median > 0.5 * vmax / 10), "relative standard deviation", dict(cmap="viridis_r", vmin=0, vmax=rel_max)),
    ]
    for ax, (da, title, style) in zip(axes, panels):
        pm = ax.pcolormesh(x, y, da, shading="auto", **style)
        plt.colorbar(pm, ax=ax, shrink=0.8)
        ax.set(title=title, aspect="equal", xlim=(-110, 110), ylim=(-110, 110))
    return axes


rr_kdp = rr_kdp.assign_coords(x=sweep.x, y=sweep.y)
rr_zdr = rr_zdr.assign_coords(x=sweep.x, y=sweep.y)
kdp_ens = kdp_ens.assign_coords(x=sweep.x, y=sweep.y)
fig, axes = plt.subplots(3, 3, figsize=(14, 13), layout="constrained", sharex=True, sharey=True)
show(axes[0], kdp_ens, "KDP", "$K_{DP}$ (deg/km)", 3.0)
show(axes[1], rr_kdp, "RAIN_RATE_KDP", "R($K_{DP}$) (mm/h)", 60.0)
show(axes[2], rr_zdr, "RAIN_RATE_Z_ZDR", "R($Z_H$, $Z_{DR}$) (mm/h)", 60.0)
plt.show()
```

The spread of $K_{DP}$ is the noise of the phase seen through the filter, about 0.2 deg/km
almost everywhere, so the relative spread of $R(K_{DP})$ is large in light rain, where
$K_{DP}$ is small compared with its noise. The relative spread of $R(Z_H, Z_{DR})$ is nearly
uniform: the calibration offsets of the sweep (common to all gates), the noise and the
DSD prior (a common factor) act on every gate in the same way.

The spread of a product also changes the answer: the ensemble median and the retrieval
of the observed fields differ where the retrieval is nonlinear.

```{code-cell} ipython3
from radarx.retrieve import rain_rate_z_zdr

control = rain_rate_z_zdr(sweep)
member_mean = rr_zdr.RAIN_RATE_Z_ZDR_median
both = (control > 5) & (member_mean > 5) & meteo
ratio = (member_mean / control).where(both)
print(f"gates above 5 mm/h: {int(both.sum())}; median(ensemble median / retrieval on the observed fields) = "
      f"{float(ratio.median()):.3f}, 5th-95th percentile {float(ratio.quantile(0.05)):.2f}-{float(ratio.quantile(0.95)):.2f}")
```

### Drop size distribution and hydrometeor class

```{code-cell} ipython3
dsd_ens = ensemble_retrieval(sweep, "dsd", members=24, seed=2, noise=noise, band="S").assign_coords(x=sweep.x, y=sweep.y)
hid_ens = ensemble_retrieval(
    sweep, "hid", members=24, seed=3, noise=noise, options={"temperature": "TEMP", "band": "S"}
).assign_coords(x=sweep.x, y=sweep.y)
fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), layout="constrained", sharex=True, sharey=True)
x, y = dsd_ens.x / 1000, dsd_ens.y / 1000
panels = [
    (dsd_ens.DM_median.where(meteo), "$D_m$ median (mm)", dict(cmap="turbo", vmin=0.5, vmax=3.5)),
    (dsd_ens.DM_std.where(meteo), "$D_m$ standard deviation (mm)", dict(cmap="magma_r", vmin=0, vmax=1)),
    (hid_ens.HID_mode.where(meteo), "most frequent class (HID code)", dict(cmap="tab20", vmin=0, vmax=19)),
    (hid_ens.HID_agreement.where(meteo), "share of members in that class", dict(cmap="viridis", vmin=0, vmax=1)),
]
for ax, (da, title, style) in zip(axes, panels):
    pm = ax.pcolormesh(x, y, da, shading="auto", **style)
    plt.colorbar(pm, ax=ax, shrink=0.8)
    ax.set(title=title, aspect="equal", xlim=(-110, 110), ylim=(-110, 110))
plt.show()
classified = (hid_ens.HID_mode > 0) & meteo
agree = hid_ens.HID_agreement.where(classified)
print(f"classified gates: {int(classified.sum())}; more than 90 % of the members choose the same class in "
      f"{float((agree > 0.9).sum() / classified.sum()):.0%} of them, all members in {float((agree == 1).sum() / classified.sum()):.0%}")
```

## Attenuation correction at C band

The C-band CSAPR2 sweep of the CACTI campaign (Argentina, 11 November 2018). The
corrected reflectivity of `correct_attenuation` depends on the reflectivity calibration
one for one, and on the coefficient $\alpha$ that links the phase to the attenuation. The
`prior` changes $\alpha$ and $\beta$ together by the scatter of $A_H/K_{DP}$ that
`relation_scatter` gives; this is the part of the spread that grows behind heavy rain.

```{code-cell} ipython3
file = DATASETS.fetch("corcsapr2cmacppiM1.c1.20181111.030003.nc")
csapr = xd.io.open_cfradial1_datatree(file).xradar.georeference()["sweep_0"].to_dataset()
names = dict(
    dbz="uncorrected_reflectivity_h",
    zdr="uncorrected_differential_reflectivity",
    phidp="uncorrected_differential_phase",
    rhohv="uncorrected_copol_correlation_coeff",
)
common = dict(members=24, seed=4, band="C", variables=["DBZH_corrected"], **names)
snr_name = {"snr": "signal_to_noise_ratio_copolar_h"}
att = ensemble_retrieval(csapr, "attenuation", noise=snr_name, **common)
att_alpha = ensemble_retrieval(csapr, "attenuation", calibration=None, noise=False, **common)  # the coefficients only
att_meas = ensemble_retrieval(csapr, "attenuation", noise=snr_name, prior=False, **common)  # calibration and noise only
att = att.assign_coords(x=csapr.x, y=csapr.y)
rain_gate = (csapr["uncorrected_copol_correlation_coeff"] > 0.9) & (csapr["uncorrected_reflectivity_h"] > 10)
fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
x, y = att.x / 1000, att.y / 1000
panels = [
    (att.DBZH_corrected_median.where(rain_gate), "corrected $Z_H$, median (dBZ)", dict(cmap="turbo", vmin=-10, vmax=60)),
    (att.DBZH_corrected_std.where(rain_gate), "corrected $Z_H$, standard deviation (dB)", dict(cmap="magma_r", vmin=0, vmax=4)),
]
for ax, (da, title, style) in zip(axes[:2], panels):
    pm = ax.pcolormesh(x, y, da, shading="auto", **style)
    plt.colorbar(pm, ax=ax, shrink=0.8)
    ax.set(title=title, aspect="equal", xlim=(-20, 110), ylim=(-110, 75))
rng_km = att["range"].values / 1000
for ens, label, color in ((att, "all", "k"), (att_meas, "calibration and noise", "#0072B2"), (att_alpha, "attenuation coefficients", "#D55E00")):
    axes[2].plot(rng_km, ens.DBZH_corrected_std.where(rain_gate).mean("azimuth"), color=color, label=label, lw=1.2)
axes[2].set(xlabel="range (km)", ylabel="mean standard deviation of the corrected $Z_H$ (dB)", title="what makes the spread")
axes[2].legend(frameon=False)
plt.show()
```

The calibration offset and the noise of the gate are the larger part of the spread at every
range (1.3 dB). From 25 km outwards the coefficients add 0.3 to 0.6 dB; the two parts add in
quadrature, so the total grows only from 1.3 to 1.5 dB along the ray.

## Members, statistics and the accessors

`keep_members=True` keeps every member as `<name>_members` with the dimension
`member`; `ensemble_statistics` reduces such an array (or the output of `perturb_radar`)
to the same variables, and the results are ordinary xarray objects:

```{code-cell} ipython3
small = sweep.isel(azimuth=slice(0, 120), range=slice(0, 300))
members = ensemble_retrieval(small, "rain_rate_z_zdr", members=16, seed=5, noise=noise, keep_members=True)
print({k: v for k, v in members.sizes.items()}, "|", members.attrs["ensemble_method"])
stats = ensemble_statistics(members.RAIN_RATE_Z_ZDR_members.rename("RAIN_RATE_Z_ZDR"), percentiles=(10, 90))
print(sorted(stats.data_vars))
print("same median as ensemble_retrieval:",
      bool(np.allclose(stats.RAIN_RATE_Z_ZDR_median, members.RAIN_RATE_Z_ZDR_median, equal_nan=True)))

# the perturbed fields themselves, for your own retrieval
perturbed = small.radarx.perturb_radar(members=8, seed=5, snr="SNR")
print({v: perturbed[v].dims for v in perturbed.data_vars})
err = small.radarx.measurement_error(snr="SNR")
print({v: round(float(err[v].median()), 3) for v in err.data_vars})
total = perturbed.radarx.ensemble_statistics()
print(sorted(v for v in total.data_vars if v.startswith("ZDR")))
```

A function that takes a sweep with the standard names (`DBZH`, `ZDR`, `PHIDP`, `RHOHV`) and
returns a Dataset or a DataArray works the same way, for example the reflectivity
and the differential reflectivity, corrected for a fixed attenuation, to the rain rate by
the user's own relation:

```{code-cell} ipython3
def rate(member, a=0.0365, b=0.6, c=-1.1):
    z = 10 ** (member["DBZH"] / 10)
    return (a * z**b * 10 ** (c * member["ZDR"] / 10)).rename("RATE")


own = small.radarx.ensemble_retrieval(rate, members=16, seed=6, noise=noise, options={"a": 0.0365})
print(float(own.RATE_median.where(own.RATE_median > 1).median()), float((own.RATE_std / own.RATE_median).where(own.RATE_median > 1).median()))
```

A volume goes through sweep by sweep (the sweeps without radar fields are skipped), and
every sweep has its own random numbers:

```{code-cell} ipython3
volume = xd.io.open_nexradlevel2_datatree(path, sweep=[0, 2]).xradar.georeference()
for name in volume.children:
    node = volume[name].to_dataset()
    for var, limit in (("DBZH", -32.0), ("ZDR", -12.9), ("RHOHV", 0.21)):
        node[var] = node[var].where(node[var] > limit)
    volume[name] = node.isel(range=slice(0, 500))
volume_ens = volume.radarx.ensemble_retrieval("rain_rate_z", members=8, seed=7)
print(list(volume_ens.children), {n: float(volume_ens[n].RAIN_RATE_Z_std.mean()) for n in volume_ens.children})
```

## What is and what is not in the spread

- The members carry the errors of the *measurements and the calibration* the way
  Melnikov and Zrnic (2004) model them, and the changes of the *prior* that
  `relation_scatter` and the settings of `prior` describe. The retrieval is the
  published, deterministic one; its own approximations (the adaptive window of `estimate_kdp`,
  the default coefficients of the rain rate relations, the fixed shape of the DSD) are not
  in the spread. In the experiment above they show as an error that is larger than the
  spread of the $Z_H$-$Z_{DR}$ relation and of $K_{DP}$.
- The errors of different gates are independent, except for the calibration offsets and the
  prior changes, which are the same for all gates of a sweep. Gate averages of an
  ensemble member are therefore too smooth in their noise and too coherent in their bias.
- Beam blockage, partial beam filling, non-uniform beam filling, ground clutter and wet radome
  are not modelled, nor the bias of $Z_{DR}$ from the noise at low SNR (use
  `correct_zdr_noise` first), and the formulas of Melnikov and Zrnic are valid for
  $\mathrm{SNR}\ge 5$ dB.
- The spread of a product at a gate is a standard deviation over the members. For a skewed
  product (rain rate) use the percentiles and the median; the mean is higher than the
  retrieval of the observed fields.
- Dask arrays are loaded into memory. The memory is (members, gates) times 4 bytes per
  product.

## References

- Giangrande, S. E., A. V. Ryzhkov, V. M. Melnikov, and J. Krause, 2004: Calibration of the polarimetric NEXRAD radar using meteorological signals. *Preprints, 11th Conference on Aviation, Range, and Aerospace Meteorology*, P5.15, <https://ams.confex.com/ams/11aram22sls/techprogram/paper_81811.htm> (read in the extended abstract: calibration to 1-1.5 dB for $Z$ and 0.1 dB for $Z_{DR}$)
- Melnikov, V. M., and D. S. Zrnic, 2004: Simultaneous transmission mode for the polarimetric WSR-88D: statistical biases and standard deviations of polarimetric variables. NOAA/NSSL report, 84 pp., <https://www.nssl.noaa.gov/publications/wsr88d_reports/SHV_statistics.pdf>
- Ryzhkov, A., M. Diederich, P. Zhang, and C. Simmer, 2014: Potential Utilization of Specific Attenuation for Rainfall Estimation, Mitigation of Partial Beam Blockage, and Radar Networking. *Journal of Atmospheric and Oceanic Technology*, **31**, 599-619, <https://doi.org/10.1175/JTECH-D-13-00038.1>
- Salmon, J. K., M. A. Moraes, R. O. Dror, and D. E. Shaw, 2011: Parallel random numbers: as easy as 1, 2, 3. *Proceedings of the 2011 International Conference for High Performance Computing, Networking, Storage and Analysis*, 1-12, <https://doi.org/10.1145/2063384.2063405>
