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

# Non-Meteorological Echo Filtering

Weather radars see more than precipitation: insects and birds, ground and sea
clutter, anomalous propagation, second-trip echo and noise. Such echo must be
removed before KDP, hydrometeor classification, rain rates or gridding.

`radarx.retrieve.echo_mask` (or `.radarx.echo_mask()` on a sweep or a volume)
classifies every gate with a fuzzy-logic score in the manner of Gourley et al.
(2007) and Krause (2016). The default scheme (`method="radarx"`) is radarx's
own: only the form of the weighted aggregation and the idea of using the
texture of $Z_{DR}$ and $\Phi_{DP}$ come from the papers, and its membership
corners, weights and threshold are not theirs. `method="krause2016"` is the
MetSignal algorithm of Krause (2016) with the memberships, weights, threshold
and post-processing rules of the paper (see the section below). The default
scheme computes these features along the ray over a short window:

- the mean $\rho_{hv}$ (precipitation: above 0.95),
- the mean $Z_{DR}$ (insects and birds: several dB up to more than 10 dB),
- the texture (standard deviation) of $Z_{DR}$ and of $\Phi_{DP}$,
- the texture of the reflectivity and its "spin change" (Steiner and Smith
  2002), which pick out ground clutter.

Each feature gives a membership between 0 and 1; their weighted mean is
averaged over 3 x 3 gates and compared with a threshold. Small isolated
regions are removed as speckle. NEXRAD no-data codes are recognised, and the
Doppler cuts of NEXRAD split cuts (no polarimetric data) take the class of
the surveillance cut at the same elevation.

`radarx.retrieve.apply_mask` (or `.radarx.apply_mask()`) then sets the
non-meteorological gates of chosen fields to NaN.

A compiled kernel classifies all gates of all sweeps in one multithreaded
call.

```{code-cell} ipython3
import time

import cmweather  # noqa: F401  radar colormaps
import matplotlib.pyplot as plt
import numpy as np
import xradar as xd
from matplotlib.colors import ListedColormap

import radarx  # noqa: F401  registers the .radarx accessors
from open_radar_data import DATASETS
from radarx.retrieve import echo_mask
```

## A squall line with biological echo

The KGWX (Columbus, Mississippi) volume from 30 March 2022 shows a squall line
west of the radar. East of it, ahead of the line, a region of weak echo with
$Z_{DR}$ above 5 dB and low, noisy $\rho_{hv}$: insects or birds.

```{code-cell} ipython3
path = DATASETS.fetch("KGWX20220330_234639_V06")
dtree = xd.io.open_nexradlevel2_datatree(path)
dtree.load()
```

```{code-cell} ipython3
start = time.perf_counter()
qc = dtree.radarx.echo_mask()
print(f"{time.perf_counter() - start:.2f} s for the whole volume")
qc["sweep_0"].ds
```

```{code-cell} ipython3
clean = dtree.radarx.apply_mask(qc)
```

## Before and after

```{code-cell} ipython3
def xy(ds):
    az = np.deg2rad(ds.azimuth.values)[:, None]
    r = ds.range.values[None, :] / 1000.0
    return r * np.sin(az), r * np.cos(az)


sweep = dtree["sweep_0"].ds
x, y = xy(sweep)
classes = ListedColormap(["#2a78d6", "#eb6834", "#eda100"])

fig, axs = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)
panels = [
    (sweep.DBZH.where(sweep.DBZH > -32), "DBZH before", dict(vmin=-10, vmax=65, cmap="ChaseSpectral")),
    (sweep.ZDR.where(sweep.DBZH > -32), "ZDR before", dict(vmin=-2, vmax=8, cmap="HomeyerRainbow")),
    (qc["sweep_0"].ds.ECHO_CLASS.where(qc["sweep_0"].ds.ECHO_CLASS > 0), "ECHO_CLASS", dict(vmin=0.5, vmax=3.5, cmap=classes)),
    (clean["sweep_0"].ds.DBZH, "DBZH after apply_mask", dict(vmin=-10, vmax=65, cmap="ChaseSpectral")),
]
for ax, (data, title, kw) in zip(axs.flat, panels):
    pm = ax.pcolormesh(x, y, data, shading="auto", **kw)
    cb = fig.colorbar(pm, ax=ax)
    if title == "ECHO_CLASS":
        cb.set_ticks([1, 2, 3], labels=["meteorological", "non-meteorological", "speckle"])
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(-250, 250)
    ax.set_ylim(-250, 250)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
```

The biological echo east of the radar is removed, the squall line and the
stratiform rain are kept. How much of each echo type is removed on the lowest
sweep:

```{code-cell} ipython3
cls = qc["sweep_0"].ds.ECHO_CLASS.values
z, zdr = sweep.DBZH.values, sweep.ZDR.values
regions = {
    "biological (east, ZDR > 4 dB, Z < 20 dBZ)": (x > 85) & (x < 130) & (abs(y) < 45) & (zdr > 4) & (z < 20),
    "squall line (Z >= 35 dBZ)": (x > -170) & (x < 20) & (z >= 35),
    "light rain west (5-25 dBZ)": (x < 0) & (z > 5) & (z < 25),
}
for name, region in regions.items():
    sel = region & (cls > 0)
    print(f"{name:45s} {sel.sum():7d} gates, removed {100 * np.mean(cls[sel] != 1):5.1f} %")
```

## The score

`METEO_SCORE` is the weighted mean membership; gates scoring at least
`threshold` (default 0.6) are meteorological.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7, 6))
pm = ax.pcolormesh(x, y, qc["sweep_0"].ds.METEO_SCORE, vmin=0, vmax=1, cmap="RdBu", shading="auto")
fig.colorbar(pm, ax=ax, label="METEO_SCORE")
ax.set_aspect("equal")
ax.set_xlim(40, 160)
ax.set_ylim(-60, 60)
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)");
```

## Velocity of the Doppler cut

The Doppler cut of the lowest elevation (`sweep_1`) holds no polarimetric
fields; its gates take the class of the surveillance cut `sweep_0`, so the
radial velocity of the biological echo is masked too.

```{code-cell} ipython3
doppler = dtree["sweep_1"].ds
xd1, yd1 = xy(doppler)
fig, axs = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
for ax, data, title in zip(
    axs,
    [doppler.VRADH.where(doppler.VRADH > -63.9), clean["sweep_1"].ds.VRADH],
    ["VRADH before", "VRADH after apply_mask"],
):
    pm = ax.pcolormesh(xd1, yd1, data, vmin=-40, vmax=40, cmap="balance", shading="auto")
    fig.colorbar(pm, ax=ax, label="m/s")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(-250, 250)
    ax.set_ylim(-250, 250)
```

## The MetSignal algorithm of Krause (2016)

`method="krause2016"` implements the algorithm of Krause (2016): the
reflectivity, $\rho_{hv}$, the radial velocity and the standard deviations of
$\Phi_{DP}$, $Z_{DR}$ and $\rho_{hv}$ over nine range bins, the memberships of
the paper's Fig. 2, the weights 1, 1, 1, 2, 2, 1 of its Table 1, a threshold of
0.8 in the warm season and 0.7 in the cold season, and the rules that gates
with $|Z_{DR}| > 4.5$ dB or $\rho_{hv} < 0.65$ are not meteorological while
gates with at least 11 dBZ at 3 km height in the previous volume scan are.
Radarx does not build the 3 km field: pass it as a field of the sweep with
`zh_3km`. Without it, as here, the strict threshold also removes some gates
of the squall line (4 % of the gates with 35 dBZ or more at 0.8); the paper
designs the algorithm to flag meteorological signal only where the evidence is
strong and to add back missed detections with the 3 km rule. The surveillance cut of a NEXRAD split cut holds no velocity, so the
velocity term is left out of its weighted mean.

```{code-cell} ipython3
modes = {
    "radarx (default)": qc,
    "krause2016, threshold 0.8": dtree.radarx.echo_mask(method="krause2016"),
    "krause2016, threshold 0.7": dtree.radarx.echo_mask(method="krause2016", threshold=0.7),
}
echo = (sweep.DBZH > -10).values
print(f"{'':28s}{'removed':>9s}{'biological':>12s}{'squall line':>13s}")
for name, result in modes.items():
    c = result["sweep_0"].ds.ECHO_CLASS.values
    row = [np.mean((c != 1)[echo])] + [
        np.mean(c[region & (c > 0)] != 1) for region in list(regions.values())[:2]
    ]
    print(f"{name:28s}" + "".join(f"{100 * v:11.1f} %" for v in row))
```

The MetSignal score has no 3 x 3 averaging and no speckle filter; `min_size`
adds the radarx filter if wanted. The default scheme is unchanged and removes
less echo.

## The classifier of Gourley et al. (2007)

The [Gourley classifier notebook](Echo_Classifier_Gourley) shows it on its own,
with the options and the numbers against the default.

`method="gourley2007"` follows the structure of the paper. The texture of
$Z_{DR}$ and $\Phi_{DP}$ is the root-mean-square difference between a gate and
its 3 x 3 neighbours (rays times gates). Each of the three variables, $\rho_{hv}$
and the two textures, has a membership in each of three classes (precipitation,
ground clutter, clear air). The aggregation value of a class is the weighted
mean of its memberships, with the weights 1 : 1.2 : 3 that the paper derives from
the overlap of the precipitation and non-precipitation curves. The class with
the largest value wins. The thresholds of the paper's Table 1 set the value of a
class to zero where that class is impossible (for example precipitation below 5
dBZ or with $\rho_{hv} < 0.7$), and a despeckling step turns isolated
precipitation gates into non-meteorological ones and clear-air gates enclosed by
precipitation into precipitation.

The paper shows its membership functions as plots of density estimates from the C-band
radar of Trappes. Radarx tabulates them from that figure
(`radarx.retrieve.qc.GOURLEY_MEMBERSHIPS`); they reproduce the modes and the
numbers quoted in the text, and the textures depend on the gate and beam size, so
`memberships` accepts tables fitted to your radar. The paper corrects the textures
for a linear increase with range beyond 45 km; the slope depends on the radar
and is not applied unless `range_correction` is given.

```{code-cell} ipython3
gq = dtree.radarx.echo_mask(method="gourley2007", field="VRADH")
c_default = qc["sweep_0"].ds.ECHO_CLASS.values
c_gourley = gq["sweep_0"].ds.ECHO_CLASS.values
echo_gates = c_default > 0
met_d, met_g = c_default == 1, c_gourley == 1
diff = np.full(c_default.shape, np.nan)
diff[echo_gates & met_d & met_g] = 0
diff[echo_gates & met_d & ~met_g] = 1
diff[echo_gates & ~met_d & met_g] = 2
diff[echo_gates & ~met_d & ~met_g] = 3

fig, axs = plt.subplots(1, 3, figsize=(17, 5.8), constrained_layout=True)
for ax, c, title in (
    (axs[0], c_default, 'method="radarx" (default)'),
    (axs[1], c_gourley, 'method="gourley2007"'),
):
    ax.pcolormesh(x, y, np.where(c > 0, c, np.nan), vmin=0.5, vmax=3.5, cmap=classes, shading="auto")
    ax.set_title(title)
pm = axs[2].pcolormesh(
    x, y, diff, vmin=-0.5, vmax=3.5,
    cmap=ListedColormap(["#2a78d6", "#d03030", "#eda100", "#c8c8c8"]), shading="auto",
)
cb = fig.colorbar(pm, ax=axs[2])
cb.set_ticks([0, 1, 2, 3], labels=["both meteorological", "only default", "only Gourley", "both not"])
axs[2].set_title("Meteorological echo in the two methods")
for ax in axs:
    ax.set_aspect("equal")
    ax.set_xlim(-250, 250)
    ax.set_ylim(-250, 250)
    ax.set_xlabel("x (km)")
print(f"agreement on {echo_gates.sum()} echo gates: {100 * np.mean((met_d == met_g)[echo_gates]):.1f} %")
for name, region in regions.items():
    sel = region & echo_gates
    print(f"{name:45s} removed: default {100 * np.mean(~met_d[sel]):5.1f} %, Gourley {100 * np.mean(~met_g[sel]):5.1f} %")
```

The two methods agree on about 93 % of the echo gates of this sweep. The
Gourley classifier removes less of the biological echo (86 % against 97 %) and
more of the weak echo around the squall line, where gates with
$\rho_{hv} < 0.7$ or below 5 dBZ cannot be precipitation under Table 1. It also needs the polarimetric
variables: beyond about 300 km the surveillance cut holds no $\rho_{hv}$, $Z_{DR}$ or
$\Phi_{DP}$, so these gates are echo without a class (class 3) and count as removed
in the "light rain west" line, while the default scheme classifies them from the
texture of the reflectivity alone. Radial streaks of low $\rho_{hv}$ in the north
are removed only by the Gourley classifier. `rules={"dbz_min": None}` drops the
reflectivity rule and `rules` changes the other thresholds of Table 1.
The Doppler velocity (`field`) only enters through the ground-clutter threshold, and
the pulse-to-pulse variability of the reflectivity (`zsigma`) needs a field that
NEXRAD Level II data do not hold.

## Options

- `method` chooses `"radarx"` (default), `"krause2016"` or `"gourley2007"`;
  `field` (radial velocity), `zh_3km` and `zsigma` are the extra fields of the
  other two,
- for `"gourley2007"`: `memberships`, `weights`, `rules` (Table 1),
  `range_correction` and `despeckle`,
- `limits` and `weights` change the membership of each feature of the default
  scheme (`radarx.retrieve.qc.DEFAULT_LIMITS`, `DEFAULT_WEIGHTS`),
- `threshold`, `window` (km), `min_size` (gates) and `spin_threshold` (dB),
- `nodata` sets the no-data floors (NEXRAD codes are detected automatically),
- `snr` / `min_snr` use a signal-to-noise ratio field where available,
- `engine` and `n_threads` choose the implementation.

## References

- Gourley, J. J., P. Tabary, and J. Parent du Chatelet, 2007: A fuzzy logic
  algorithm for the separation of precipitating from nonprecipitating echoes
  using polarimetric radar observations. *J. Atmos. Oceanic Technol.*, **24**,
  1439-1451, https://doi.org/10.1175/JTECH2035.1
- Krause, J. M., 2016: A simple algorithm to discriminate between
  meteorological and nonmeteorological radar echoes. *J. Atmos. Oceanic
  Technol.*, **33**, 1875-1885, https://doi.org/10.1175/JTECH-D-15-0239.1
- Steiner, M., and J. A. Smith, 2002: Use of three-dimensional reflectivity
  structure for automated detection and removal of nonprecipitating echoes in
  radar data. *J. Atmos. Oceanic Technol.*, **19**, 673-686,
  https://doi.org/10.1175/1520-0426(2002)019<0673:UOTDRS>2.0.CO;2
