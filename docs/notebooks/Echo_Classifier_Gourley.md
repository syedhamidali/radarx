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

# The Gourley Echo Classifier

`qc.echo_mask` sorts every gate of a sweep into meteorological and
non-meteorological echo. Its default, `method="radarx"`, is a fuzzy score of
radarx's own. `method="gourley2007"` is the three-class classifier of Gourley,
Tabary and Parent du Chatelet (2007): precipitation, ground clutter and clear
air compete, and the class with the largest aggregation value wins. It is opt-in,
so nothing changes unless you ask for it.

This notebook runs both on one volume and shows where they differ. The rest of
the echo filtering is in the [Echo_QC](Echo_QC) notebook.

```{code-cell} ipython3
import time
import warnings

import cmweather  # noqa: F401  radar colormaps
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xradar as xd
from matplotlib.colors import ListedColormap

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.io.aws_data import download_file

warnings.filterwarnings("ignore", message="The input coordinates to pcolormesh")
```

## The data

The case is the squall line of KGWX (Columbus, Mississippi) on 30 March 2022,
with a patch of insects or birds ahead of it (weak echo, $Z_{DR}$ above 5 dB,
low and noisy $\rho_{hv}$). The volume comes from the public NEXRAD archive on
AWS. Without a network connection the notebook builds a small synthetic sweep
with the same three ingredients: rain, biological echo and ground clutter.

```{code-cell} ipython3
def synthetic_volume(seed=0):
    rng = np.random.default_rng(seed)
    az = np.arange(0.5, 360.0, 1.0)
    rng_m = 250.0 * np.arange(1, 801)
    A, R = np.meshgrid(np.deg2rad(az), rng_m / 1000.0, indexing="ij")
    x, y = R * np.sin(A), R * np.cos(A)

    rain = np.exp(-(((x + 60) / 50) ** 2 + ((y - 10) / 80) ** 2))
    bio = np.exp(-(((x - 90) / 25) ** 2 + ((y - 0) / 30) ** 2))
    clutter = np.exp(-(R / 12.0) ** 2) * (rng.random(R.shape) < 0.6)

    dbz = np.full(R.shape, np.nan)
    zdr = np.full(R.shape, np.nan)
    rho = np.full(R.shape, np.nan)
    phi = np.full(R.shape, np.nan)
    m = rain > 0.1
    dbz[m] = 15 + 40 * rain[m] + rng.normal(0, 1.5, m.sum())
    zdr[m] = 0.3 + 1.5 * rain[m] + rng.normal(0, 0.3, m.sum())
    rho[m] = 0.985 + rng.normal(0, 0.006, m.sum())
    phi[m] = 20 + 60 * rain[m] * R[m] / 100 + rng.normal(0, 1.5, m.sum())
    m = bio > 0.15
    dbz[m] = 8 + 12 * bio[m] + rng.normal(0, 2, m.sum())
    zdr[m] = 6 + rng.normal(0, 1.8, m.sum())
    rho[m] = np.clip(0.7 + rng.normal(0, 0.1, m.sum()), 0.3, 1)
    phi[m] = rng.uniform(0, 120, m.sum())
    m = clutter > 0.5
    dbz[m] = 40 + rng.normal(0, 5, m.sum())
    zdr[m] = rng.normal(0, 2.5, m.sum())
    rho[m] = np.clip(0.55 + rng.normal(0, 0.15, m.sum()), 0.2, 1)
    phi[m] = rng.uniform(0, 150, m.sum())
    vel = np.where(m, rng.normal(0, 0.5, R.shape), np.nan)

    def f(a, units):
        return (("azimuth", "range"), a, {"units": units})

    ds = xr.Dataset(
        {"DBZH": f(dbz, "dBZ"), "ZDR": f(zdr, "dB"), "RHOHV": f(rho, ""),
         "PHIDP": f(phi, "degrees"), "VRADH": f(vel, "m s-1")},
        coords={"azimuth": az, "range": rng_m, "elevation": ("azimuth", np.full(az.size, 0.5))},
    )
    return xr.DataTree.from_dict({"/": xr.Dataset(), "/sweep_0": ds}), "synthetic sweep"


def load_volume():
    try:
        path = download_file(
            "unidata-nexrad-level2", "2022/03/30/KGWX/KGWX20220330_234639_V06", "."
        )
        dtree = xd.io.open_nexradlevel2_datatree(path)
        dtree.load()
        return dtree, "KGWX 30 March 2022 23:46 UTC"
    except Exception as err:  # offline or archive unavailable
        print(f"no NEXRAD data ({type(err).__name__}), using the synthetic sweep")
        return synthetic_volume()


dtree, label = load_volume()
print(label)
```

## Both methods on the lowest sweep

`field="VRADH"` hands the Doppler cut to the ground-clutter rule of the
Gourley classifier (clutter has a velocity near zero). Everything else is the
default call.

```{code-cell} ipython3
t0 = time.perf_counter()
default = dtree.radarx.echo_mask()
t1 = time.perf_counter()
gourley = dtree.radarx.echo_mask(method="gourley2007", field="VRADH")
t2 = time.perf_counter()
print(f"default {t1 - t0:.2f} s, gourley2007 {t2 - t1:.2f} s for the volume")

sweep = dtree["sweep_0"].to_dataset()
az = np.deg2rad(sweep.azimuth.values)[:, None]
rg = sweep.range.values[None, :] / 1000.0
x, y = rg * np.sin(az), rg * np.cos(az)

c_d = default["sweep_0"].to_dataset().ECHO_CLASS.values
c_g = gourley["sweep_0"].to_dataset().ECHO_CLASS.values
echo = c_d > 0
met_d, met_g = c_d == 1, c_g == 1
print(f"echo gates {echo.sum()}, the methods agree on {100 * np.mean((met_d == met_g)[echo]):.1f} %")
print(f"meteorological: default {100 * np.mean(met_d[echo]):.1f} %, gourley2007 {100 * np.mean(met_g[echo]):.1f} %")
```

The first two panels give the class of each gate, the third marks where the
methods disagree.

```{code-cell} ipython3
diff = np.full(c_d.shape, np.nan)
diff[echo & met_d & met_g] = 0
diff[echo & met_d & ~met_g] = 1
diff[echo & ~met_d & met_g] = 2
diff[echo & ~met_d & ~met_g] = 3
classes = ListedColormap(["#2a78d6", "#eb6834", "#eda100"])
lim = 250 if label.startswith("KGWX") else 200

fig, axs = plt.subplots(1, 3, figsize=(17, 5.8), constrained_layout=True)
for ax, c, title in ((axs[0], c_d, 'method="radarx" (default)'), (axs[1], c_g, 'method="gourley2007"')):
    pm = ax.pcolormesh(x, y, np.where(c > 0, c, np.nan), vmin=0.5, vmax=3.5, cmap=classes, shading="auto")
    ax.set_title(title)
cb = fig.colorbar(pm, ax=axs[:2], location="bottom", shrink=0.6)
cb.set_ticks([1, 2, 3], labels=["meteorological", "non-meteorological", "unclassified"])
pm = axs[2].pcolormesh(
    x, y, diff, vmin=-0.5, vmax=3.5,
    cmap=ListedColormap(["#2a78d6", "#d03030", "#eda100", "#c8c8c8"]), shading="auto",
)
cb = fig.colorbar(pm, ax=axs[2])
cb.set_ticks([0, 1, 2, 3], labels=["both meteorological", "only default", "only Gourley", "neither"])
axs[2].set_title("Meteorological echo in the two methods")
for ax in axs:
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x (km)")
axs[0].set_ylabel("y (km)")
fig.suptitle(label)
plt.show()
```

On the KGWX volume the lowest sweep has 644 377 echo gates, and the two
methods agree on 93.3 % of them (94.6 % over the whole volume of 8.5 million
gates). The gates they split on are mostly weak echo at the edge of the squall
line, the biological patch, and radial streaks of low $\rho_{hv}$ (north and
southwest) that only the Gourley classifier removes. The default scores 96.4 % of the echo as meteorological,
Gourley 90.9 %.

## Where the biological echo goes

A region-by-region count shows what each method keeps. The Gourley classifier
removes less of the biological echo but a bit more of the weak echo, because
Table 1 of the paper rules out precipitation below 5 dBZ and below
$\rho_{hv} = 0.7$.

```{code-cell} ipython3
z, zdr = (sweep[v].values for v in ("DBZH", "ZDR"))
if label.startswith("KGWX"):
    regions = {
        "biological echo (east)": (x > 85) & (x < 130) & (abs(y) < 45) & (zdr > 4) & (z < 20),
        "squall line (Z >= 35 dBZ)": (x > -170) & (x < 20) & (z >= 35),
    }
else:
    regions = {
        "biological echo (east)": (x > 60) & (x < 120) & (abs(y) < 40) & (z < 25),
        "rain (Z >= 35 dBZ)": (x < 0) & (z >= 35),
        "ground clutter": (np.hypot(x, y) < 20) & (z >= 30),
    }
print(f"{'region':28s}{'gates':>8s}{'removed, default':>19s}{'removed, gourley2007':>23s}")
for name, region in regions.items():
    sel = region & echo
    print(f"{name:28s}{sel.sum():8d}{100 * np.mean(~met_d[sel]):17.1f} %{100 * np.mean(~met_g[sel]):21.1f} %")
```

At KGWX the biological echo is 97.4 % removed by the default and 86.0 % by the
Gourley classifier, and the squall line is removed in 0.0 % and 0.5 %. Beyond
about 300 km the surveillance cut has no polarimetric data. The Gourley
classifier leaves those gates unclassified (class 3), the default classifies
them from the texture of the reflectivity.

## The score

For `gourley2007`, `METEO_SCORE` is the share of the precipitation class in the
sum of the three aggregation values, so a gate is meteorological where the
score is largest for that class and there is no single threshold to tune.

```{code-cell} ipython3
fig, axs = plt.subplots(1, 2, figsize=(12, 5.4), constrained_layout=True)
for ax, res, title in zip(axs, (default, gourley), ("default", "gourley2007")):
    pm = ax.pcolormesh(x, y, res["sweep_0"].to_dataset().METEO_SCORE, vmin=0, vmax=1, cmap="RdBu", shading="auto")
    ax.set_title(f"METEO_SCORE, {title}")
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x (km)")
fig.colorbar(pm, ax=axs, label="METEO_SCORE", shrink=0.8)
plt.show()
```

## The opt-in options

All of these apply to `gourley2007` only and are rejected for the other
methods.

- `rules` replaces thresholds of Table 1 of the paper (`GOURLEY_RULES`); `None`
  switches one off.
- `despeckle=False` skips the neighbour check that reassigns isolated gates.
- `range_correction={"zdr": 0.008, "phidp": 0.05}` removes the linear increase
  of the textures with range beyond 45 km. It is off by default because the
  slopes depend on the radar.
- `memberships` takes tables fitted to your own radar. The ones shipped
  (`GOURLEY_MEMBERSHIPS`) are read from Fig. 3 of the paper, which shows C-band
  data of Trappes.
- `weights` changes the weights of $\rho_{hv}$ and the two textures.

```{code-cell} ipython3
variants = {
    "defaults": {},
    "no 5 dBZ rule": dict(rules={"dbz_min": None}),
    "no despeckling": dict(despeckle=False),
    "range correction": dict(range_correction={"zdr": 0.008, "phidp": 0.05}),
    "rhohv weight 2": dict(weights={"rhohv": 2.0}),
}
print(f"{'variant':20s}{'meteorological':>16s}{'same as defaults':>19s}")
base = None
for name, kwargs in variants.items():
    res = dtree.radarx.echo_mask(method="gourley2007", field="VRADH", **kwargs)
    met = res["sweep_0"].to_dataset().ECHO_CLASS.values == 1
    base = met if base is None else base
    print(f"{name:20s}{100 * np.mean(met[echo]):14.1f} %{100 * np.mean((met == base)[echo]):17.1f} %")
```

## Which one to use

The default is faster (0.35 s against 0.77 s on all cores for the KGWX volume of
8.5 million gates) and keeps weak echo with low $\rho_{hv}$. The Gourley
classifier follows a published scheme, removes radial streaks of low
$\rho_{hv}$ and gates below 5 dBZ, and keeps more high-$Z_{DR}$ biological echo.
Against the CMAC gate classification of a CSAPR2 volume (C band, 223 thousand
gates) it reaches an accuracy of 0.864 against 0.844 for the default.
Switching the default would change the class of 5 to 7 % of the echo gates on
these volumes, which is why it is a choice and not a default.

## References

- Gourley, J. J., P. Tabary, and J. Parent du Chatelet, 2007: A fuzzy logic
  algorithm for the separation of precipitating from nonprecipitating echoes
  using polarimetric radar observations. *J. Atmos. Oceanic Technol.*, **24**,
  1439-1451, https://doi.org/10.1175/JTECH2035.1
