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

# VIL, echo tops and water content

Four column products of the reflectivity of a radar volume, in
`radarx.retrieve`:

- `vil`: vertically integrated liquid, the liquid water of rain in a column in
  kg m$^{-2}$ ([Greene and Clark 1972](https://doi.org/10.1175/1520-0493(1972)100<0548:VILWNA>2.3.CO;2)),
  $\mathrm{VIL} = 3.44\times10^{-6}\sum_i \left[(Z_i+Z_{i+1})/2\right]^{4/7}\Delta h_i$
  with $Z$ in mm$^6$ m$^{-3}$ and $\Delta h$ in m;
- `echo_top`: the height of the highest 18 dBZ echo, interpolated in dBZ between
  the beams that bracket the threshold
  ([Lakshmanan et al. 2013](https://doi.org/10.1175/WAF-D-12-00084.1));
- `vil_density`: the VIL divided by the echo top
  ([Amburn and Wolf 1997](https://doi.org/10.1175/1520-0434(1997)012<0473:VDAAHI>2.0.CO;2));
- `liquid_water_content`: the water content of rain at every gate, from the
  same power law or from the gamma drop size distribution of `dsd`.

A column is a list of reflectivity samples at known heights. For a polar
volume (a `DataTree` of sweeps) it is an (azimuth, ground range) position of
the lowest sweep, with one sample from every other sweep at its beam height;
for a grid it is an (y, x) position, and for a quasi-vertical profile (QVP) one
profile. Between two beams the reflectivity varies linearly with height, nothing
is assumed above the highest valid beam, and the layer below the lowest beam is
left out by default, so the VIL is a lower bound (`VIL_LOWER_BOUND`) that
`fill_below=True` turns into an estimate. The integrals run in a compiled,
multithreaded kernel.

```{code-cell} ipython3
import warnings

import cmweather  # noqa: F401  radar colormaps
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xradar as xd
from open_radar_data import DATASETS

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.io.aws_data import download_file
from radarx.retrieve import (
    echo_top,
    estimate_kdp,
    liquid_water_content,
    melting_layer,
    qvp_timeseries,
    vil,
    vil_density,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
```

## A squall line seen by KGWX

The KGWX WSR-88D (Columbus, Mississippi) volume of 30 March 2022, 23:59 UTC,
from the open NEXRAD archive on AWS. The no-data codes (DBZH at or below
−32 dBZ) are masked and the range is cut at 150 km. The volume keeps all cuts
of the scan; where an elevation was scanned twice the cut that reaches farther
is used.

```{code-cell} ipython3
key = "2022/03/30/KGWX/KGWX20220330_235959_V06"
path = download_file("unidata-nexrad-level2", key, "data")
drop = ["WRADH", "CCORH", "CFP", "ZDR", "PHIDP", "RHOHV"]
volume = xd.io.open_nexradlevel2_datatree(path, drop_variables=drop)


def prepare(ds):
    """Mask the no-data codes and keep the first 150 km."""
    if "range" not in ds.dims:
        return ds
    ds = ds.sel(range=slice(None, 150e3))
    return ds.assign(DBZH=ds.DBZH.where(ds.DBZH > -32).astype("float32"))


volume = volume.map_over_datasets(prepare)
sweeps = [n for n in volume.children if n.startswith("sweep") and "DBZH" in volume[n].ds]
print(len(sweeps), "sweeps with reflectivity, elevations:",
      " ".join(f"{float(volume[n].sweep_fixed_angle):.1f}" for n in sweeps))
```

## VIL, echo top and VIL density

Each call takes the volume and returns a product on the polar grid of the lowest
sweep. `vil` returns a dataset with the VIL, the height of the lowest beam used
(the lower integration limit) and two flags, `VIL_LOWER_BOUND` (the layer under
the lowest beam is missing) and `VIL_TOP_TRUNCATED` (echo continues above the
highest beam); the reflectivity is capped at 56 dBZ (`dbz_cap`), as in the
operational product.

```{code-cell} ipython3
vil_polar = vil(volume)
top = echo_top(volume)
density = vil_density(volume)
vil_polar
```

The same products are available as methods of the volume,
`volume.radarx.vil()`, `.radarx.echo_top()` and `.radarx.vil_density()`.

```{code-cell} ipython3
low = volume[sweeps[0]].to_dataset()
az = np.deg2rad(vil_polar.azimuth.values)[:, None]
s = vil_polar.ground_range.values[None, :]
x, y = s * np.sin(az) / 1e3, s * np.cos(az) / 1e3
rays = np.argsort(vil_polar.azimuth.values)  # no seam where the scan started

panels = [
    (low.DBZH, "$Z_H$ at 0.5° (dBZ)", dict(cmap="ChaseSpectral", vmin=-10, vmax=70)),
    (vil_polar.VIL, "VIL (kg m$^{-2}$)", dict(cmap="viridis", vmin=0, vmax=30)),
    (top / 1e3, "18 dBZ echo top (km)", dict(cmap="cividis", vmin=0, vmax=16)),
    (density, "VIL density (g m$^{-3}$)", dict(cmap="plasma", vmin=0, vmax=3.5)),
]
fig, axes = plt.subplots(2, 2, figsize=(11, 10), layout="constrained")
for ax, (field, label, style) in zip(axes.flat, panels):
    pm = ax.pcolormesh(x[rays], y[rays], field.values[rays], shading="nearest", **style)
    fig.colorbar(pm, ax=ax, label=label, shrink=0.85)
    ax.set_aspect("equal")
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)
    ax.set_xlabel("east of KGWX (km)")
    ax.set_ylabel("north of KGWX (km)")
fig.suptitle("KGWX, 30 March 2022, 23:59 UTC")
plt.show()
```

The line of convection has a VIL of 10 to 25 kg m$^{-2}$ and echo tops
of 8 to 10 km. The broad, weak VIL elsewhere is clear-air and boundary-layer echo,
which is not removed here (see the echo-QC notebook). The VIL density is
largest in the cores, where the water is concentrated in a shallow column; the
upper end of the colour scale, 3.5 g m$^{-3}$, is the value that Amburn and
Wolf (1997) associate with large hail. Within about 10 km of the radar the
highest beam (19.5°) is only a few km high, so the echo top and the VIL there
are truncated (`VIL_TOP_TRUNCATED`).

### Below the lowest beam

The lowest beam rises from a few hundred metres above the radar at 20 km to
2.5 km at 150 km, and the VIL leaves out the layer below it, so it is a lower bound
(`VIL_LOWER_BOUND`). `fill_below=True` extends the lowest value down to the
radar height, as a pseudo-CAPPI does. The mean VIL of the rays that cross the
line is shown against ground range with and without that layer.

```{code-cell} ipython3
filled = vil(volume, fill_below=True)
line = (vil_polar.VIL > 3).any("range")  # rays that cross a cell
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
ground_km = vil_polar.ground_range / 1e3
for name, product in (("lower bound", vil_polar), ("lowest value filled to the ground", filled)):
    axes[0].plot(ground_km, product.VIL.where(line).mean("azimuth"), label=name)
axes[0].set(xlabel="ground range (km)", ylabel="mean VIL of the cells (kg m$^{-2}$)")
axes[0].legend(frameon=False)
axes[1].plot(ground_km, vil_polar.VIL_LOWEST_HEIGHT.mean("azimuth") / 1e3)
axes[1].set(xlabel="ground range (km)", ylabel="mean height of the lowest beam (km)")
plt.show()
```

## Polar volume and cone grid

The same functions work on a grid (any dataset with `z`, `y` and `x`, such as the
output of `grid_cones`). The comparison below uses a 1 km grid with 250 m
levels. The grid interpolates linearly in dBZ between the cones, the polar
columns integrate $Z^{4/7}$ with $Z$ linear between the beams, so the gridded
VIL is lower where the reflectivity falls quickly with height; the median
difference on this volume is about −16 %. Gridding $Z$ instead of dBZ reduces
it to about −3 %.

```{code-cell} ipython3
from scipy.interpolate import RegularGridInterpolator

from radarx.grid import grid_cones

gx = gy = np.arange(-150e3, 150e3 + 1, 1000.0)
gz = np.arange(250.0, 18e3, 250.0)
grid = grid_cones(volume, "DBZH", gx, gy, gz)
vil_grid = vil(grid)

at = RegularGridInterpolator((gy, gx), vil_grid.VIL.values, bounds_error=False)
north, east = s * np.cos(az), s * np.sin(az)
from_grid = at(np.stack([north.ravel(), east.ravel()], -1)).reshape(north.shape)
reference = vil_polar.VIL.values
use = (s > 20e3) & (s < 120e3) & (reference > 1.0) & np.isfinite(from_grid)
rel = (from_grid[use] - reference[use]) / reference[use]
print(f"{use.sum()} columns: median difference {100 * np.median(rel):.0f} %, "
      f"5-95 % range {100 * np.percentile(rel, 5):.0f} to {100 * np.percentile(rel, 95):.0f} %")

fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
ax.hexbin(reference[use], from_grid[use], gridsize=60, extent=(0, 30, 0, 30), mincnt=1, cmap="Blues", bins="log")
ax.plot([0, 30], [0, 30], "k-", lw=0.8)
ax.set(xlabel="VIL of the polar volume (kg m$^{-2}$)", ylabel="VIL of the cone grid (kg m$^{-2}$)", aspect="equal")
plt.show()
```

## Liquid water content of rain

`liquid_water_content` gives the water content of rain gate by gate. The
default `method="zm"` is the power law behind the VIL formula,
$\mathrm{LWC} = 3.44\times10^{-3}\,Z^{4/7}$ g m$^{-3}$, which assumes the
exponential drop size distribution of Marshall and Palmer. `method="dsd"`
retrieves the gamma drop size distribution of each gate from $Z_H$, $Z_{DR}$
and $K_{DP}$ (`radarx.retrieve.dsd`) and returns its water content. Radar
sees the liquid water of rain only: above the melting layer the reflectivity
is that of ice, where the power law overestimates the water content, and in the
melting layer it is enhanced. The gates below the melting layer are selected
with `mask`. Ice water content needs a different method that radarx does not
provide.

The example is a 0.5° C-band sweep of the ARM CSAPR2 radar in deep convection
(Argentina, CACTI); the file has a sounding temperature at every gate and an
echo classification (rain is class 1), and KDP is estimated from the raw
differential phase. The drop size distribution is retrieved where its
normalizing intercept $N_w$ is between $10^2$ and $10^5$ m$^{-3}$ mm$^{-1}$
(`nw_range`); outside that range the polarimetric variables are not
consistent with rain (hail, for example, gives an enormous number of tiny
drops).

```{code-cell} ipython3
file = DATASETS.fetch("corcsapr2cmacppiM1.c1.20181111.030003.nc")
sweep = xd.io.open_cfradial1_datatree(file).xradar.georeference()["sweep_0"].to_dataset()
kdp = estimate_kdp(
    sweep,
    phidp="uncorrected_differential_phase",
    rhohv="uncorrected_copol_correlation_coeff",
    dbzh="attenuation_corrected_reflectivity_h",
)
sweep = sweep.assign(KDP=kdp.KDP)
fields = dict(
    dbz="attenuation_corrected_reflectivity_h",
    mask=(sweep.gate_id == 1) & (sweep.sounding_temperature > 3.0),  # rain, warmer than +3 °C
)
lwc_zm = liquid_water_content(sweep, "zm", **fields)
lwc_dsd = liquid_water_content(
    sweep, "dsd",
    zdr="attenuation_corrected_differential_reflectivity", kdp="KDP", band="C",
    nw_range=(1e2, 1e5), **fields,
)
print(f"{int(lwc_zm.notnull().sum())} rain gates; median LWC {float(lwc_zm.median()):.2f} (power law) "
      f"and {float(lwc_dsd.median()):.2f} g m-3 (DSD)")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
for ax, (field, label) in zip(axes[:2], [(lwc_zm, "power law"), (lwc_dsd, "gamma DSD")]):
    pm = ax.pcolormesh(sweep.x / 1e3, sweep.y / 1e3, field, cmap="viridis", vmin=0, vmax=6, shading="nearest")
    fig.colorbar(pm, ax=ax, label=f"LWC, {label} (g m$^{{-3}}$)", shrink=0.85)
    ax.set_aspect("equal")
    ax.set_xlim(-20, 110)
    ax.set_ylim(-110, 75)
    ax.set_xlabel("east of CSAPR2 (km)")
axes[0].set_ylabel("north of CSAPR2 (km)")
both = lwc_zm.notnull() & lwc_dsd.notnull()
h = axes[2].hexbin(lwc_zm.values[both], lwc_dsd.values[both], gridsize=60, extent=(0, 6, 0, 6), mincnt=1, cmap="Blues", bins="log")
axes[2].plot([0, 6], [0, 6], "k-", lw=0.8)
axes[2].set(xlabel="LWC, power law (g m$^{-3}$)", ylabel="LWC, gamma DSD (g m$^{-3}$)", aspect="equal")
fig.colorbar(h, ax=axes[2], label="gates", shrink=0.85)
plt.show()
```

The two methods show the same structure. They differ where the drop size
distribution departs from the exponential one that the power law assumes: the
gamma retrieval gives larger water contents in the strongest cores (the
99th percentile of the rain gates is 6.2 g m$^{-3}$, against 3.3 g m$^{-3}$
for the power law), and `lwc_dsd` is empty where the polarimetric variables are not consistent with rain.

## Profiles: VIL of a quasi-vertical profile

`vil` also takes a QVP time series (a `height` dimension, with `time`). The
VIL is that of the azimuthal-mean profile, starting at the lowest height of the
profile, so it is a lower bound. If a melting level is given, `VIL_LIQUID` is
the VIL of the rain below it: pass a height (a number or an array over time,
such as the 0 °C height of a sounding or ERA5) or the result of `melting_layer`,
whose bottom is used. The QVPs are those of the ARMOR C-band radar (Huntsville,
Alabama) on 11 April 2008, 12° elevation, as in the QVP notebook.

```{code-cell} ipython3
names = sorted(n for n in DATASETS.registry if n.startswith("RAW_NA_000_125_"))[::2]
volumes = [xd.io.open_iris_datatree(DATASETS.fetch(n)) for n in names]
tqvp = qvp_timeseries(volumes, ["DBZH", "ZDR", "RHOHV"], elevation=12.0)
ml = melting_layer(tqvp)
vil_qvp = vil(tqvp, melting=ml)
vil_qvp
```

```{code-cell} ipython3
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
pm = tqvp.DBZH.plot(ax=axes[0], x="time", y="height", cmap="ChaseSpectral", vmin=-10, vmax=60, add_colorbar=False)
fig.colorbar(pm, ax=axes[0], label="$Z_H$ (dBZ)", pad=0.01)
axes[0].fill_between(ml.time, ml.melting_layer_bottom, ml.melting_layer_top, color="k", alpha=0.3, label="melting layer")
axes[0].set(ylim=(0, 12e3), ylabel="height (m)", xlabel="", title="ARMOR QVP, 12° elevation, 11 April 2008")
axes[0].legend(loc="upper right")
axes[1].plot(vil_qvp.time, vil_qvp.VIL, label="VIL, whole profile")
axes[1].plot(vil_qvp.time, vil_qvp.VIL_LIQUID, label="VIL below the melting layer")
axes[1].set(ylabel="VIL (kg m$^{-2}$)", xlabel="time (UTC)")
axes[1].legend(frameon=False)
plt.show()
```

The VIL of the whole profile includes the ice above the melting layer, where the
formula for rain is not valid, and the bright band; the liquid VIL stops below
the layer.

## References

- Greene, D. R., and R. A. Clark, 1972: Vertically integrated liquid water: A new
  analysis tool. *Mon. Wea. Rev.*, **100**, 548-552,
  <https://doi.org/10.1175/1520-0493(1972)100<0548:VILWNA>2.3.CO;2>
- Seo, B.-C., W. F. Krajewski, and Y. Qi, 2020: Utility of vertically integrated
  liquid water content for radar-rainfall estimation: Quality control and
  precipitation type classification. *Atmos. Res.*, **236**, 104800,
  <https://doi.org/10.1016/j.atmosres.2019.104800>
- Amburn, S. A., and P. L. Wolf, 1997: VIL density as a hail indicator. *Wea.
  Forecasting*, **12**, 473-478,
  <https://doi.org/10.1175/1520-0434(1997)012<0473:VDAAHI>2.0.CO;2>
- Lakshmanan, V., K. Hondl, C. K. Potvin, and D. Preignitz, 2013: An improved
  method for estimating radar echo-top height. *Wea. Forecasting*, **28**,
  481-488, <https://doi.org/10.1175/WAF-D-12-00084.1>
- Marshall, J. S., and W. McK. Palmer, 1948: The distribution of raindrops with
  size. *J. Meteor.*, **5**, 165-166,
  <https://doi.org/10.1175/1520-0469(1948)005<0165:TDORWS>2.0.CO;2>
