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

# Bayesian DSD Retrieval

`radarx.retrieve.dsd` returns one drop size distribution (DSD) per gate: the
shape $\mu$ is fixed or tied to the slope, and $Z_H$, $Z_{DR}$ and $K_{DP}$
are taken as exact. `radarx.retrieve.dsd_bayesian` (or `.radarx.dsd_bayesian()`)
returns the **posterior distribution** of the normalized gamma DSD parameters
$(\log_{10} N_w, D_m, \mu)$ instead, given

- the polarimetric variables $Z_H$, $Z_{DR}$ and, where available, $K_{DP}$
  and the specific attenuation $A_H$;
- forward operators from radarx's T-matrix scattering tables (S, C, X band);
- a measurement-error model: noise and an unknown calibration bias of $Z_H$
  and $Z_{DR}$, absolute and relative errors of $K_{DP}$ and $A_H$;
- a prior learned from disdrometers: `"generic"` (ranges of Bringi et al.
  2003, $\mu$–$\Lambda$ relation of Cao et al. 2008) or `"perils2022"`
  (PERiLS 2022 Parsivel2 DSDs matched to the radar beam along their fall
  trajectories), or one learned from any disdrometer data set with
  `dsd_prior`.

Per gate it returns the posterior mean, standard deviation, quantiles
(credible intervals) and maximum (MAP) of $\log_{10} N_w$, $D_m$, $\mu$, the
rain rate and the liquid water content, plus the misfit and the evidence of
the observations, which flag gates that are not rain. The posterior is
evaluated on a grid in $(D_m, \mu)$ with Laplace integration in
$\log_{10} N_w$, in a multithreaded C++ kernel.

```{code-cell} ipython3
import fsspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xradar as xd

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.retrieve import dsd, dsd_bayesian, dsd_prior, forward_grid
```

## Priors

The prior is a distribution over the $(D_m, \mu)$ grid times a Gaussian in
$\log_{10} N_w$ at each node. The generic prior is broad; the prior learned
from the PERiLS disdrometers is concentrated on the DSDs of quasi-linear
convective systems in the south-eastern United States.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
for ax, name in zip(axes, ("generic", "perils2022")):
    p = dsd_prior(name)
    p.prior_mass.T.plot(ax=ax, cmap="Blues", add_colorbar=False)
    ax.set_title(f"prior {name!r}")
    ax.set_xlabel("$D_m$ (mm)")
    ax.set_ylabel("$\\mu$")
```

## Uncertainty that means what it says

Draw DSDs from the prior, simulate $Z_H$, $Z_{DR}$ and $K_{DP}$ with the
forward model and the default error model, and retrieve them: the true values
fall inside the 68 % and 95 % credible intervals about 68 % and 95 % of the
time.

```{code-cell} ipython3
rng = np.random.default_rng(1)
prior = dsd_prior("generic")
fw = forward_grid("S")
e = radarx.retrieve.dsd_bayes.ERRORS
n = 5000
j = rng.choice(prior.prior_mass.size, n, p=prior.prior_mass.values.ravel())
t = rng.normal(prior.log10_nw_mean.values.ravel()[j], prior.log10_nw_sd.values.ravel()[j])
kdp = 10**t * fw.KDP.values.ravel()[j]
sim = xr.Dataset(
    {
        "DBZH": ("gate", 10 * t + fw.DBZH.values.ravel()[j] + rng.normal(0, np.hypot(e["zh"], e["zh_bias"]), n)),
        "ZDR": ("gate", fw.ZDR.values.ravel()[j] + rng.normal(0, np.hypot(e["zdr"], e["zdr_bias"]), n)),
        "KDP": ("gate", kdp + rng.normal(0, 1, n) * np.hypot(e["kdp"], e["kdp_rel"] * kdp)),
    }
)
truth = {"LOG10_NW": t, "DM": fw.dm.values[j // fw.sizes["mu"]], "MU": fw.mu.values[j % fw.sizes["mu"]]}
post = dsd_bayesian(sim, band="S", kdp="KDP")
for name, x in truth.items():
    q = post[name + "_QUANTILES"]
    c68 = np.mean((x >= q.sel(quantile=0.16)) & (x <= q.sel(quantile=0.84)))
    c95 = np.mean((x >= q.sel(quantile=0.025)) & (x <= q.sel(quantile=0.975)))
    rmse = float(np.sqrt(np.mean((post[name] - x) ** 2)))
    print(f"{name:9s} RMSE {rmse:.2f}   coverage 68 %: {c68:.2f}   95 %: {c95:.2f}")
```

$D_m$ is well constrained by $Z_{DR}$; $\mu$ mostly follows the prior, as
$K_{DP}/Z_H$ hardly depends on it at S band. The deterministic methods hide
this; here it shows up as a wide posterior of $\mu$.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
axes[0].plot(truth["DM"], post.DM, ".", ms=1, alpha=0.4)
axes[0].plot([0.4, 4], [0.4, 4], "k-", lw=0.8)
axes[0].set_xlabel("true $D_m$ (mm)")
axes[0].set_ylabel("posterior mean $D_m$ (mm)")
axes[1].hist(post.MU_SD, 40, alpha=0.7, label="$\\mu$")
axes[1].hist(post.DM_SD * 10, 40, alpha=0.7, label="$D_m$ (x10 mm)")
axes[1].set_xlabel("posterior standard deviation")
axes[1].legend();
```

## A squall line in Mississippi

The lowest sweep of the KGWX WSR-88D volume of 30 March 2022, 23:46 UTC,
with $K_{DP}$ from `estimate_kdp` and the crude rain mask of the
[DSD retrieval notebook](DSD_Retrieval.md).

```{code-cell} ipython3
local_file = fsspec.open_local(
    "simplecache::s3://unidata-nexrad-level2/2022/03/30/KGWX/KGWX20220330_234639_V06",
    s3={"anon": True},
    filecache={"cache_storage": "."},
)
dtree = xd.io.open_nexradlevel2_datatree(local_file, sweep=[0])
dtree = dtree.xradar.georeference()
sweep = dtree["sweep_0"].to_dataset()
for name, lim in (("DBZH", -32.0), ("ZDR", -12.9), ("RHOHV", 0.21)):
    sweep[name] = sweep[name].where(sweep[name] > lim)
sweep["KDP"] = sweep.radarx.kdp()["KDP"]
hail = (sweep.DBZH >= 50) & (sweep.ZDR < 0.5)
rain = (sweep.RHOHV >= 0.97) & (sweep.DBZH >= 5) & ~hail & (sweep.range <= 120e3)
post = sweep.radarx.dsd_bayesian(kdp="KDP", mask=rain, band="S", prior="perils2022")
post[["DM", "DM_SD", "RAIN_RATE", "MISFIT"]]
```

```{code-cell} ipython3
q = post.RAIN_RATE_QUANTILES
fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
x, y = sweep.x / 1e3, sweep.y / 1e3
panels = [
    (post.DM, "posterior mean $D_m$ (mm)", "plasma", 0.5, 3.0),
    (post.DM_SD, "posterior sd of $D_m$ (mm)", "viridis", 0, 0.5),
    (post.RAIN_RATE, "posterior mean rain rate (mm h$^{-1}$)", "turbo", 0, 60),
    (
        (q.sel(quantile=0.975) - q.sel(quantile=0.025)) / post.RAIN_RATE,
        "95 % interval of R / R",
        "magma",
        0,
        2,
    ),
]
for ax, (da, title, cmap, vmin, vmax) in zip(axes.flat, panels):
    pm = ax.pcolormesh(x, y, da, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(pm, ax=ax, shrink=0.8)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(-120, 120)
    ax.set_ylim(-120, 120)
```

In the convective line, where $K_{DP}$ is large, the 95 % interval of the
rain rate shrinks to about half of its value elsewhere: $K_{DP}$ fixes
$N_w$ independently of the $Z_H$ calibration. The posterior mean follows
the deterministic normalized-gamma retrieval ($\mu$ = 3) but is pulled
towards the drop sizes of the prior for the largest $Z_{DR}$; the
uncertainty of $D_m$ is smallest in light rain, where the learned prior
already confines the small drops, and largest at 30-40 dBZ.

```{code-cell} ipython3
det = dsd(sweep, "normalized", kdp="KDP", mask=rain, band="S")
ok = rain.values & np.isfinite(det.DM.values) & np.isfinite(post.DM.values)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
axes[0].hist2d(det.DM.values[ok], post.DM.values[ok], bins=np.linspace(0.4, 3.5, 60), cmin=1, cmap="Blues")
axes[0].plot([0.4, 3.5], [0.4, 3.5], "k-", lw=0.8)
axes[0].set_xlabel("normalized gamma ($\\mu$ = 3) $D_m$ (mm)")
axes[0].set_ylabel("Bayesian posterior mean $D_m$ (mm)")
axes[1].hist2d(sweep.DBZH.values[ok], post.DM_SD.values[ok], bins=[np.linspace(5, 60, 56), np.linspace(0, 0.6, 61)], cmin=1, cmap="Blues")
axes[1].set_xlabel("$Z_H$ (dBZ)")
axes[1].set_ylabel("posterior sd of $D_m$ (mm)");
```

The misfit of the MAP state is a $\chi^2$ with about as many degrees of
freedom as inputs for rain; gates far beyond it (here hail and mixed-phase
echoes that pass the crude mask) are not consistent with any rain DSD.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.hist(post.MISFIT.values[ok], np.linspace(0, 30, 61), log=True)
ax.set_xlabel("misfit $\\chi^2$ (3 inputs)")
print(f"gates with misfit > 16: {np.mean(post.MISFIT.values[ok] > 16):.1%}")
```

## References

- Testud, J., S. Oury, R. A. Black, P. Amayenc, and X. Dou, 2001: The concept
  of "normalized" distribution to describe raindrop spectra: A tool for cloud
  physics and cloud remote sensing. *J. Appl. Meteor.*, **40**, 1118–1140,
  <https://doi.org/10.1175/1520-0450(2001)040<1118:TCONDT>2.0.CO;2>
- Bringi, V. N., V. Chandrasekar, J. Hubbert, E. Gorgucci, W. L. Randeu, and
  M. Schoenhuber, 2003: Raindrop size distribution in different climatic
  regimes from disdrometer and dual-polarized radar analysis. *J. Atmos.
  Sci.*, **60**, 354–365,
  <https://doi.org/10.1175/1520-0469(2003)060<0354:RSDIDC>2.0.CO;2>
- Cao, Q., G. Zhang, E. Brandes, T. Schuur, A. Ryzhkov, and K. Ikeda, 2008:
  Analysis of video disdrometer and polarimetric radar data to characterize
  rain microphysics in Oklahoma. *J. Appl. Meteor. Climatol.*, **47**,
  2238–2255, <https://doi.org/10.1175/2008JAMC1732.1>
- Cao, Q., G. Zhang, and M. Xue, 2013: A variational approach for retrieving
  raindrop size distribution from polarimetric radar measurements in the
  presence of attenuation. *J. Appl. Meteor. Climatol.*, **52**, 169–185,
  <https://doi.org/10.1175/JAMC-D-12-0101.1>
- Rodgers, C. D., 2000: *Inverse Methods for Atmospheric Sounding: Theory and
  Practice*. World Scientific, <https://doi.org/10.1142/3171>
