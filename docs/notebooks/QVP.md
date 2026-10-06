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
from radarx.io import sounding
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
Along the time series, detections far from the running median are rejected
and short gaps are filled; `melting_layer_flag` records what happened to
each profile.

### Top and bottom

The ρhv dip is the clearest marker of melting: it starts at the wet-bulb
0 °C level, where snow begins to melt, and ends where the snow has melted
([Ryzhkov and Krause 2022](https://doi.org/10.1175/JTECH-D-21-0130.1)).
By default (`boundaries="onset"`) the top and bottom are where the ρhv
signature begins and ends: going up and down from the ρhv minimum, as
[Griffin et al. (2020)](https://doi.org/10.1175/JAMC-D-19-0128.1) do, the
first height where ρhv is back within 10 % of the dip depth from its
background (`onset_fraction`), interpolated between gates. Griffin et al.
use a fixed background, ρhv ≥ 0.97 in S-band QVPs; `rhohv_onset=0.97`
reproduces their definition. The previous definition,
`boundaries="half_prominence"`, puts the edges where the ZDR and ρhv
anomalies have fallen to half their prominence: the width at half height,
so a thinner layer whose top lies below the onset of melting.

```{code-cell} ipython3
ml = melting_layer(tqvp)
half = melting_layer(tqvp, boundaries="half_prominence")
ml
```

### The environment: 0 °C and wet-bulb 0 °C heights

Pass a sounding or an ERA5 profile from `radarx.io.sounding` as
`environment` to compare the layer with the 0 °C and wet-bulb 0 °C heights.
They are returned with the offsets of the layer top, and the search for the
signature is restricted to −1000 to +500 m around the wet-bulb 0 °C height,
which keeps ρhv dips in other layers out. Here: the Birmingham, Alabama (BMX)
radiosonde of 00 UTC 12 April 2008 from the Iowa Environmental Mesonet.
ERA5 at the radar comes from `volumes[0].radarx.sounding()` (or
`radarx.io.sounding.era5_profile`); several profiles concatenated on `time`
are interpolated to the QVP times.

```{code-cell} ipython3
bmx = sounding.read_sounding("BMX", "2008-04-12T00:00", source="iem")
ml = melting_layer(tqvp, environment=bmx)
ml[["freezing_level", "wet_bulb_zero_height"]].isel(time=0).compute()
```

```{code-cell} ipython3
ok = ml.melting_layer_flag.isin([1, 3])  # detected or filled
for name in ("freezing_level", "wet_bulb_zero"):
    offset = ml[f"melting_layer_top_offset_{name}"].where(ok)
    print(f"top - {name}: median {float(offset.median()):+.0f} m")
```

ARMOR saw convective rain on this day: ρhv stays low above the layer and the
signature is deep and variable, but the onset top follows the 0 °C level of
the sounding (median offset about +150 m), while the half-prominence top is
about 270 m below it. In the stratiform rain behind the KGWX squall line of
30–31 March 2022 (S band, 19.5°, 27 volumes) the onset top is at 3.48 km on
average, 36 m below the ERA5 wet-bulb 0 °C height at the radar (3.51 km)
and 97 m below the ERA5 0 °C height, and 48 m above the 0 °C height of the
BMX sounding (3.43 km); the half-prominence top (3.20 km) is 320 m below the
ERA5 wet-bulb 0 °C height and ρhv ≥ 0.97 gives 3.29 km.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.fill_between(
    ml.time, ml.melting_layer_bottom, ml.melting_layer_top, alpha=0.3, label="onset"
)
ax.plot(half.time, half.melting_layer_top, "k^:", ms=4, lw=1, label="half prominence")
ax.plot(half.time, half.melting_layer_bottom, "kv:", ms=4, lw=1)
ax.plot(ml.time, ml.freezing_level, "C3-", label="BMX 0 °C")
ax.plot(ml.time, ml.wet_bulb_zero_height, "C3--", label="BMX wet-bulb 0 °C")
ax.set_ylabel("height above sea level [m]")
ax.legend(loc="lower left", ncols=4, fontsize=8)
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
    ax.plot(ml.time, ml.freezing_level, "m-", lw=1.5, label="0 °C")
    ax.plot(ml.time, ml.wet_bulb_zero_height, "m--", lw=1.5, label="wet-bulb 0 °C")
    ax.set_ylim(0, 12e3)  # the profiles reach 27 km; all ML heights are shown above
    ax.set_ylabel("height above sea level [m]")
    ax.set_title(f"QVP of {name}, ARMOR, elevation 12°")
    ax.set_xlabel("")
axes[0].legend(loc="upper right", ncols=2)
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
- Griffin, E. M., T. J. Schuur, and A. V. Ryzhkov, 2020: A polarimetric radar
  analysis of ice microphysical processes in melting layers of winter storms
  using S-band quasi-vertical profiles. *J. Appl. Meteor. Climatol.*, **59**,
  751–767, <https://doi.org/10.1175/JAMC-D-19-0128.1>
- Ryzhkov, A., and J. Krause, 2022: New polarimetric radar algorithm for
  melting-layer detection and determination of its height. *J. Atmos.
  Oceanic Technol.*, **39**, 529–543, <https://doi.org/10.1175/JTECH-D-21-0130.1>
