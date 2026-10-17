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

# Probabilistic Nowcasting of a Squall Line

+++

A nowcast moves the precipitation of the last radar volumes forward in time.
radarx builds it on the motion of `estimate_motion` or `optical_flow` and on the
semi-Lagrangian `advect`, in three steps of increasing detail:

1. **Lagrangian persistence** (`nowcast(..., method="extrapolation")`): the last
   field is carried along the motion (Germann and Zawadzki 2002).
2. **S-PROG** (`method="sprog"`): the field is split into spatial scales, each
   scale follows an autoregressive model of order 2, so the small scales, which
   are not predictable, fade first (Seed 2003).
3. **STEPS** (`method="steps"`): the same cascade with a spatially correlated noise
   at every scale that refills the small scales, for every member of an ensemble
   (Bowler et al. 2006), optionally with a perturbed motion.

`exceedance_probability` turns the ensemble into the probability of a threshold,
and `fss`, `crps`, `categorical_scores`, `brier_score`, `reliability` and
`rank_histogram` score it. The cascade, the noise and the verification run in
compiled, multithreaded C++ kernels with NumPy equivalents. The example is the
KGWX squall line of 30 March 2022: three volumes as history, five as truth.

**References**

- Seed, A. W., 2003: A dynamic and spatial scaling approach to advection
  forecasting. *J. Appl. Meteor.*, **42**, 381-388,
  https://doi.org/10.1175/1520-0450(2003)042<0381:ADASSA>2.0.CO;2
- Bowler, N. E., C. E. Pierce, and A. W. Seed, 2006: STEPS: A probabilistic
  precipitation forecasting scheme which merges an extrapolation nowcast with
  downscaled NWP. *Quart. J. Roy. Meteor. Soc.*, **132**, 2127-2155,
  https://doi.org/10.1256/qj.04.100
- Germann, U., and I. Zawadzki, 2002: Scale-dependence of the predictability of
  precipitation from continental radar images. Part I: Description of the
  methodology. *Mon. Wea. Rev.*, **130**, 2859-2873,
  https://doi.org/10.1175/1520-0493(2002)130<2859:SDOTPO>2.0.CO;2
- Roberts, N. M., and H. W. Lean, 2008: Scale-selective verification of rainfall
  accumulations from high-resolution forecasts of convective events. *Mon. Wea.
  Rev.*, **136**, 78-97, https://doi.org/10.1175/2007MWR2123.1
- Pulkkinen, S., D. Nerini, A. A. Pérez Hortal, C. Velasco-Forero, A. Seed,
  U. Germann, and L. Foresti, 2019: Pysteps: an open-source Python library for
  probabilistic precipitation nowcasting (v1.0). *Geosci. Model Dev.*, **12**,
  4185-4219, https://doi.org/10.5194/gmd-12-4185-2019

```{code-cell} ipython3
import cmweather  # noqa
import fsspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xradar as xd

import radarx as rx
from radarx.utils import combine_nexrad_sweeps
```

## Grid the volumes

Eight consecutive volumes, about 6.7 minutes apart. Each is gridded at 2 km and
reduced to the column maximum (the composite reflectivity). Missing data (no echo
or no coverage) are NaN. The NEXRAD no-data code (below -32 dBZ) is masked.

```{code-cell} ipython3
names = [
    "KGWX20220330_231240_V06",
    "KGWX20220330_231927_V06",
    "KGWX20220330_232552_V06",
    "KGWX20220330_233252_V06",
    "KGWX20220330_233952_V06",
    "KGWX20220330_234639_V06",
    "KGWX20220330_235324_V06",
    "KGWX20220330_235959_V06",
]


def composite(name):
    local_file = fsspec.open_local(
        f"simplecache::s3://unidata-nexrad-level2/2022/03/30/KGWX/{name}",
        s3={"anon": True},
        filecache={"cache_storage": "."},
    )
    dtree = combine_nexrad_sweeps(xd.io.open_nexradlevel2_datatree(local_file))
    grid = dtree.radarx.to_grid(
        ["DBZH"],
        x_lim=(-100e3, 100e3),
        y_lim=(-100e3, 100e3),
        z_lim=(500, 10e3),
        x_step=2000,
        y_step=2000,
        z_step=1000,
    )
    dbz = grid.DBZH.where(grid.DBZH > -32.0).max("z")
    return xr.Dataset({"DBZH": dbz.assign_attrs(units="dBZ")}).assign(time=grid.time)


volumes = [composite(n) for n in names]
history, future = volumes[:3], volumes[3:]
print([str(v.time.values)[11:19] for v in volumes])
```

## Motion

`estimate_motion` correlates two volumes. `optical_flow` fits the optical-flow
constraint over all the history: locally (Lucas-Kanade) or with a smooth global
series (DARTS-type). All three give a `Dataset` with `u` and `v` that `advect` and
`nowcast` accept.

```{code-cell} ipython3
fft = rx.retrieve.estimate_motion(history[1], history[2])
lk = rx.retrieve.optical_flow(history, method="lucas_kanade", uniform=True)
darts = rx.retrieve.optical_flow(history, method="darts", uniform=True)
for name, m in (("FFT correlation", fft), ("Lucas-Kanade", lk), ("DARTS", darts)):
    print(f"{name:16s} u = {float(m.u):5.1f} m/s  v = {float(m.v):5.1f} m/s  quality {float(m.quality):.2f}")
```

## Forecasts

```{code-cell} ipython3
K = len(future)
persistence = history[-1].DBZH.fillna(5.0).where(lambda a: a >= 10.0, 5.0)
extrapolation = rx.retrieve.nowcast(history, K, method="extrapolation", motion=lk)
sprog = rx.retrieve.nowcast(history, K, method="sprog", motion=lk)
ensemble = rx.retrieve.nowcast(history, K, method="steps", n_members=20, motion=lk, seed=1)
print(ensemble.dims, ensemble.shape, "lead times", ensemble.lead_time.values.astype("timedelta64[m]"))
```

```{code-cell} ipython3
truth = xr.concat([f.DBZH.fillna(5.0).where(lambda a: a >= 10.0, 5.0) for f in future], dim="lead_time")
truth = truth.assign_coords(lead_time=ensemble.lead_time.values)
prob = rx.retrieve.exceedance_probability(ensemble, 35.0)

kw = dict(cmap="ChaseSpectral", vmin=-10, vmax=70)
fig, axes = plt.subplots(2, 3, figsize=(14, 9), layout="constrained", sharex=True, sharey=True)
panels = [
    (truth.isel(lead_time=-1), "observed"),
    (extrapolation.isel(lead_time=-1), "extrapolation"),
    (sprog.isel(lead_time=-1), "S-PROG"),
    (ensemble.isel(member=0, lead_time=-1), "STEPS, member 0"),
    (ensemble.isel(member=1, lead_time=-1), "STEPS, member 1"),
]
for ax, (da, title) in zip(axes.flat, panels):
    pm = ax.pcolormesh(da.x / 1e3, da.y / 1e3, da, **kw)
    ax.set_title(title)
    ax.set_aspect("equal")
pp = axes.flat[5].pcolormesh(prob.x / 1e3, prob.y / 1e3, prob.isel(lead_time=-1), cmap="Blues", vmin=0, vmax=1)
axes.flat[5].set_title("probability of 35 dBZ or more")
axes.flat[5].set_aspect("equal")
fig.colorbar(pm, ax=axes[:, :2], label="composite reflectivity (dBZ)", shrink=0.6)
fig.colorbar(pp, ax=axes[:, 2], label="probability", shrink=0.6)
fig.suptitle("33 minutes after the last volume")
plt.show()
```

## Skill

The fractions skill score (FSS) of the 30 dBZ area, by scale for the last lead time
and by lead time for a 10 km neighbourhood.

```{code-cell} ipython3
scales = [2e3, 6e3, 10e3, 20e3, 40e3]
methods = {
    "persistence": persistence.expand_dims(lead_time=truth.lead_time.values),
    "extrapolation": extrapolation,
    "S-PROG": sprog,
}
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
for name, fc in methods.items():
    s = rx.retrieve.fss(fc, truth, 30.0, scales, edge="zero")
    axes[0].plot(np.array(scales) / 1e3, s.isel(lead_time=-1), marker="o", label=name)
    axes[1].plot(s.lead_time.values.astype("timedelta64[m]").astype(float), s.sel(scale=10e3), marker="o", label=name)
s = rx.retrieve.fss(ensemble, truth, 30.0, scales, edge="zero").mean("member")
axes[0].plot(np.array(scales) / 1e3, s.isel(lead_time=-1), marker="o", label="STEPS (mean of members)")
axes[1].plot(s.lead_time.values.astype("timedelta64[m]").astype(float), s.sel(scale=10e3), marker="o", label="STEPS (mean of members)")
axes[0].set(xlabel="neighbourhood (km)", ylabel="FSS, 30 dBZ, last lead time", ylim=(0, 1))
axes[1].set(xlabel="lead time (min)", ylabel="FSS, 30 dBZ, 10 km", ylim=(0, 1))
axes[0].legend()
plt.show()
```

## Probabilistic skill

The CRPS of the ensemble against that of the single forecasts (for which it is the
mean absolute error), the rank histogram, and the reliability of the 35 dBZ
probability.

```{code-cell} ipython3
over = ["y", "x"]
crps_ens = rx.retrieve.crps(ensemble, truth, dim=over)
crps_fair = rx.retrieve.crps(ensemble, truth, fair=True, dim=over)
crps_ext = rx.retrieve.crps(extrapolation, truth, dim=over)
crps_sprog = rx.retrieve.crps(sprog, truth, dim=over)
lead_min = crps_ens.lead_time.values.astype("timedelta64[m]").astype(float)
event = truth >= 35.0
rel = rx.retrieve.reliability(prob, event, bins=5)
hist = rx.retrieve.rank_histogram(ensemble, truth, seed=0)
table = rx.retrieve.categorical_scores(sprog, truth, 35.0, dim=["y", "x", "lead_time"])

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), layout="constrained")
for da, label in ((crps_ens, "STEPS"), (crps_fair, "STEPS, fair"), (crps_ext, "extrapolation"), (crps_sprog, "S-PROG")):
    axes[0].plot(lead_min, da, marker="o", label=label)
axes[0].set(xlabel="lead time (min)", ylabel="CRPS (dBZ)")
axes[0].legend()
axes[1].bar(hist["rank"], hist.values)
axes[1].set(xlabel="rank of the observation", title="rank histogram")
axes[2].plot([0, 1], [0, 1], "k--")
axes[2].plot(rel.forecast_probability, rel.observed_frequency, marker="o")
axes[2].set(xlabel="forecast probability", ylabel="observed frequency", title="reliability, 35 dBZ")
plt.show()
print(f"Brier score of the 35 dBZ probability: {float(rx.retrieve.brier_score(prob, event)):.3f}")
print(f"S-PROG at 35 dBZ: POD {float(table.pod):.2f}, FAR {float(table.far):.2f}, CSI {float(table.csi):.2f}")
```

## Learned models

`nowcast_model` runs a trained network (an ONNX model registered in `radarx.ml`) on the
last frames in place of these methods; see the end of the
[third part of the workflow](Radar_Workflow_Advanced_2). radarx ships no weights.

## What the example shows, and does not

One case with a 2 km grid and three volumes of history cannot rank the methods. In this
case (the figures above) extrapolation scores best in the FSS: the squall line moves as
one piece, the motion is well defined and the later volumes show little growth or decay.
A single STEPS member scores below the extrapolation, as it must: it keeps the small
scales that nobody can predict. The ensemble pays off in the probabilistic scores. Its
CRPS is below the mean absolute error of the single forecasts at every lead time, and the
probability of 35 dBZ is close to reliable, a little too high at the middle
probabilities. Its rank histogram slopes down, so the observation falls more often below
the members than above them: here the ensemble is biased high.

Two features of every forecast are not errors. The cells that would come from outside the
grid are NaN (the white strip at the south), and the fields of S-PROG and of the members are
smoother or noisier than the observation at small scales, which is what the cascade does
with the scales it cannot predict.

A check on the whole sequence of KGWX volumes gridded at 1 km (ten forecast starts, 20
members, 41 minutes) was run once, outside this notebook, against pysteps (version 1.21.5)
with its default settings and the same motion field (the Lucas-Kanade vector of
`optical_flow`). The scores are on the cells that the advected last observation covers.
FSS at 5 km, mean over the six lead times:

| method | 20 dBZ | 30 dBZ | 40 dBZ |
|---|---|---|---|
| persistence | 0.87 | 0.74 | 0.47 |
| radarx extrapolation | 0.94 | 0.88 | 0.63 |
| pysteps extrapolation | 0.95 | 0.90 | 0.67 |
| radarx S-PROG | 0.96 | 0.89 | 0.65 |
| pysteps S-PROG | 0.97 | 0.81 | 0.34 |
| radarx STEPS, mean of members | 0.92 | 0.84 | 0.58 |
| pysteps STEPS, mean of members | 0.87 | 0.75 | 0.48 |

The two extrapolations agree to within 0.02 (they use the same motion and the same scheme).
The ensemble of radarx has the lower CRPS (4.4 against 5.1 dBZ) and Brier score (0.117
against 0.146 for 30 dBZ); its rank histogram is flatter. These are the defaults of both
codes with no tuning of either, the transformation, the rain mask and the probability
matching differ, and the motion perturbation of STEPS is off in both, so the table is a
baseline and not a proof that one is better.
