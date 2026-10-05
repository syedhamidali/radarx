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

# Quasi-vertical profiles (QVP)

A quasi-vertical profile ([Ryzhkov et al. 2016](https://doi.org/10.1175/JTECH-D-15-0020.1))
is the azimuthal average of a high-elevation PPI: every range gate is reduced
over all rays and placed at its beam height. One volume gives one profile, a
sequence of volumes gives a time–height view of the polarimetric variables.

radarx follows the published method: only gates with ρhv > 0.6 and
Z > −10 dBZ are used, at least 30 valid gates are needed on the circle, and
quantities in dB (Z, ZDR) are averaged in linear units. The reduction runs in
a compiled, multithreaded kernel.

```{code-cell} ipython3
import warnings

import cmweather  # noqa: F401
import matplotlib.pyplot as plt
import xradar as xd
from open_radar_data import DATASETS

import radarx  # noqa: F401
from radarx.retrieve import melting_layer, qvp_timeseries

warnings.filterwarnings("ignore", category=RuntimeWarning)
```

## Data

Twenty-five volumes of the ARMOR C-band polarimetric radar (Huntsville,
Alabama) from 11 April 2008, 18:12–20:20 UTC, about every 5 minutes.

```{code-cell} ipython3
names = sorted(n for n in DATASETS.registry if n.startswith("RAW_NA_000_125_"))
volumes = [xd.io.open_iris_datatree(DATASETS.fetch(n)) for n in names]
volumes[0]
```

## One profile

`dtree.radarx.qvp()` profiles one sweep, by default the highest. The scan
strategy changes during the event (highest tilt 12° to 18°), so we ask for
the sweep closest to 12° everywhere.

```{code-cell} ipython3
fields = ["DBZH", "ZDR", "RHOHV", "PHIDP"]
profile = volumes[0].radarx.qvp(fields, elevation=12.0)
profile
```

`height` is the beam height above sea level (4/3 Earth model, as in
xradar), `ground_range` the radius of the circle each value is averaged over.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharey=True)
for ax, name in zip(axes, fields):
    profile[name].plot(ax=ax, y="height")
    ax.set_title(name)
    ax.set_ylim(0, 12e3)
fig.tight_layout()
```

## Time–height display

`qvp_timeseries` profiles all volumes in a single call of the compiled
kernel and puts them on the heights of the first profile.

```{code-cell} ipython3
tqvp = qvp_timeseries(volumes, fields + ["KDP"], elevation=12.0)
tqvp
```

## Melting layer

In the melting layer ρhv drops while ZDR and Z peak
([Giangrande et al. 2008](https://doi.org/10.1175/2007JAMC1634.1)).
`melting_layer` looks for that co-located signature in each profile: the
largest ZDR peak with a ρhv minimum and a Z maximum within 500 m. ρhv dips
without a ZDR/Z enhancement, or ZDR spikes without a ρhv dip, are ignored.
Top and bottom are where the ZDR and ρhv anomalies have fallen to half
their prominence. Along the time series, detections far from the running
median are rejected and short gaps are filled. `melting_layer_flag` records
what happened to each profile. If a sounding or model gives the 0 °C level,
pass it as `freezing_level` to narrow the search.

```{code-cell} ipython3
ml = melting_layer(tqvp)
ml
```

The melting-layer heights over the whole event, on their own axes so that no
outlier is hidden:

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 3))
ax.fill_between(
    ml.time, ml.melting_layer_bottom, ml.melting_layer_top, alpha=0.3, label="layer"
)
ax.plot(ml.time, ml.melting_layer_peak, "k.-", lw=1, label="ZDR peak")
filled = ml.melting_layer_flag == 3  # rejected as inconsistent, then filled
ax.plot(ml.time[filled], ml.melting_layer_peak[filled], "ro", mfc="none", label="filled")
ax.set_ylabel("height above sea level [m]")
ax.legend(loc="upper left", ncols=3)
ax.set_title("Melting layer, ARMOR 11 April 2008")
fig.tight_layout()
```

```{code-cell} ipython3
plots = {
    "DBZH": dict(cmap="ChaseSpectral", vmin=-10, vmax=60),
    "ZDR": dict(cmap="HomeyerRainbow", vmin=-1, vmax=3),
    "RHOHV": dict(cmap="LangRainbow12", vmin=0.8, vmax=1.0),
    "KDP": dict(cmap="HomeyerRainbow", vmin=-0.5, vmax=2),
}
fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
for ax, (name, style) in zip(axes, plots.items()):
    tqvp[name].plot(ax=ax, x="time", y="height", **style)
    ax.plot(ml.time, ml.melting_layer_top, "k^", ms=4, label="ML top")
    ax.plot(ml.time, ml.melting_layer_bottom, "kv", ms=4, label="ML bottom")
    ax.set_ylim(0, 12e3)  # the profiles reach 27 km; all ML heights are shown above
    ax.set_ylabel("height above sea level [m]")
    ax.set_title(f"QVP of {name}, ARMOR, elevation 12°")
    ax.set_xlabel("")
axes[0].legend(loc="upper right")
fig.tight_layout()
```

## References

- Ryzhkov, A., P. Zhang, H. Reeves, M. Kumjian, T. Tschallener, S. Trömel, and
  C. Simmer, 2016: Quasi-vertical profiles—A new way to look at polarimetric
  radar data. *J. Atmos. Oceanic Technol.*, **33**, 551–562,
  <https://doi.org/10.1175/JTECH-D-15-0020.1>
- Trömel, S., M. R. Kumjian, A. V. Ryzhkov, C. Simmer, and M. Diederich, 2013:
  Backscatter differential phase—Estimation and variability. *J. Appl.
  Meteor. Climatol.*, **52**, 2529–2548, <https://doi.org/10.1175/JAMC-D-13-0124.1>
- Giangrande, S. E., J. M. Krause, and A. V. Ryzhkov, 2008: Automatic
  designation of the melting layer with a polarimetric prototype of the
  WSR-88D radar. *J. Appl. Meteor. Climatol.*, **47**, 1354–1364,
  <https://doi.org/10.1175/2007JAMC1634.1>
