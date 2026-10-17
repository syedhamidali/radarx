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

# Radar-gauge merging

A radar covers the whole area but its rainfall estimate has errors that vary
in space and time: calibration, the relation between reflectivity and rain
rate, the height of the beam, attenuation, blockage. A rain gauge measures
the rain at one point accurately and says nothing about the rest. The
functions of `radarx.retrieve.gauge` combine the two on the radar grid with
the methods that
[Goudenhoofdt and Delobbe (2009)](https://doi.org/10.5194/hess-13-195-2009)
compared on four years of data from Belgium:

| `method` | what it does |
|---|---|
| `"mfb"` | multiplies the radar by the mean field bias $\sum G_i / \sum R_i$ |
| `"rda"` | multiplies it by a second-order polynomial of the range to the radar, fitted to $\ln(G/R)$ |
| `"brandes"` | interpolates the factors $G_i/R_i$ with weights $\exp(-d^2/k)$ ([Brandes 1975](https://doi.org/10.1175/1520-0450(1975)014<1339:OREWTA>2.0.CO;2)) |
| `"ok"` | ordinary kriging of the gauges alone (no radar) |
| `"conditional"` | conditional merging ([Sinclair and Pegram 2005](https://doi.org/10.1002/asl.85)): gauge kriging plus radar minus the kriging of the radar at the gauges |
| `"ked"` | kriging with the radar as an external drift (Wackernagel 2003) |

The heavy parts (the variogram of the gauges, the kriging systems for every
pixel, the Brandes weights) run in a multithreaded C++ kernel; the NumPy
implementation behind `engine="numpy"` gives the same fields.

**The data are synthetic**, so that the truth is known everywhere: a rain
accumulation field with a spatial structure, a radar estimate that is too low
and gets worse with range, and two gauge networks. One network is used for the
merging, the other, independent one, only to score it, as in the paper.

```{code-cell} ipython3
import time

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import radarx
from radarx.retrieve import gauge, gauge_adjust, gauge_variogram, radar_at_gauges

print("compiled kernel:", gauge.HAS_COMPILED_KERNEL)
```

## A rain field, a radar and two gauge networks

The truth is a log-normal field with an exponential correlation of 12 km
(a Gaussian random field made by filtering white noise) scaled to a mean of
about 8 mm. The radar sits in the middle of the 100 km domain. It reads 35 %
too low at short range, loses a further 40 % at 50 km, and has a
multiplicative error of its own with a correlation length of 4 km.

```{code-cell} ipython3
rng = np.random.default_rng(7)
n = 100  # 1 km pixels
x = y = (np.arange(n) + 0.5) * 1000.0


def random_field(rng, length_km, n=n):
    """Unit-variance Gaussian field with an exponential-like correlation."""
    k = np.fft.fftfreq(n, d=1.0)
    kk = np.hypot(*np.meshgrid(k, k))
    spectrum = (1.0 + (2 * np.pi * kk * length_km) ** 2) ** -1.5
    f = np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * np.sqrt(spectrum)).real
    return (f - f.mean()) / f.std()


truth = np.exp(0.6 * random_field(rng, 12.0))
truth *= 8.0 / truth.mean()
rr = np.hypot(*np.meshgrid(x - x.mean(), y - y.mean())) / 1000.0  # km from the radar
gain = 0.65 * np.exp(-rr / 80.0)
radar_field = truth * gain * np.exp(0.35 * random_field(rng, 4.0))
radar = xr.DataArray(
    radar_field,
    dims=("y", "x"),
    coords={"x": ("x", x - x.mean()), "y": ("y", y - y.mean())},  # m from the radar
    attrs={"units": "mm", "long_name": "radar accumulation"},
    name="RAIN",
)
truth = xr.DataArray(truth, dims=radar.dims, coords=radar.coords, attrs={"units": "mm"})


def network(n_gauges, seed, error=0.05):
    """Gauges at random positions; they measure the truth with a 5 % error."""
    r = np.random.default_rng(seed)
    gx = r.uniform(-48e3, 48e3, n_gauges)
    gy = r.uniform(-48e3, 48e3, n_gauges)
    value = truth.interp(x=("station", gx), y=("station", gy)).values
    value = value * (1 + error * r.standard_normal(n_gauges))
    return xr.DataArray(
        value, dims="station", coords={"x": ("station", gx), "y": ("station", gy)},
        attrs={"units": "mm"}, name="GAUGE",
    )


gauges = network(40, 1)  # used for the merging
verification = network(60, 2)  # independent, only for the scores
print(f"truth mean {float(truth.mean()):.1f} mm, radar mean {float(radar.mean()):.1f} mm, "
      f"gauges {gauges.size}, verification gauges {verification.size}")
```

## The variogram

Kriging needs the spatial structure of the field, the variogram:
half the mean squared difference between two gauges at a given distance.
`gauge_variogram` computes it in distance bins and fits a model with a nugget
by the weighted least squares of Cressie (1985). `gauge_adjust` does this
itself for every time; the result can also be passed on as `variogram`.

```{code-cell} ipython3
vario = gauge_variogram(gauges, model="exponential", n_lags=10)
print(f"nugget {float(vario.nugget):.2f} mm2, partial sill {float(vario.partial_sill):.2f} mm2, "
      f"range {float(vario['range']) / 1000:.0f} km")
fig, ax = plt.subplots(figsize=(5, 3.4), layout="constrained")
ax.plot(vario.distance / 1000, vario.semivariance, "o", label="empirical")
h = np.linspace(0, float(vario.distance.max()), 100)
ax.plot(h / 1000, gauge._semivariance("exponential", h, float(vario.nugget), float(vario.partial_sill), float(vario["range"])),
        label="fitted exponential")
ax.set(xlabel="distance (km)", ylabel="semivariance (mm$^2$)")
ax.legend()
plt.show()
```

## All methods, scored on the independent gauges

The scores follow the paper: the mean absolute error, the bias of the total
rain (adjusted over gauge, in %) and the root mean square error at the
verification gauges, with the radar value taken from the nearest pixel.

```{code-cell} ipython3
methods = ["mfb", "rda", "brandes", "ok", "conditional", "ked"]
merged = {}
t0 = time.perf_counter()
for m in methods:
    merged[m] = gauge_adjust(radar, gauges, method=m, radius=1500.0)
print(f"six methods on {n * n} pixels: {time.perf_counter() - t0:.2f} s")


def at_verification(field):
    return field.interp(x=("station", verification.x.values), y=("station", verification.y.values)).values


def scores(field):
    est, ref = at_verification(field), verification.values
    return dict(
        mae=np.abs(est - ref).mean(),
        bias=100 * (est.sum() / ref.sum() - 1),
        rmse=np.sqrt(((est - ref) ** 2).mean()),
    )


table = {"radar": scores(radar)} | {m: scores(merged[m].RAIN_ADJUSTED) for m in methods}
print(f"{'':12s}{'MAE (mm)':>10s}{'bias (%)':>10s}{'RMSE (mm)':>11s}")
for name, s in table.items():
    print(f"{name:12s}{s['mae']:10.2f}{s['bias']:10.1f}{s['rmse']:11.2f}")
```

The mean field bias removes the bias of the radar, the methods that adjust
in space (`"rda"`, `"brandes"`) remove more of the error, and the two kriging
methods that use the radar (`"conditional"`, `"ked"`) do best, which is the
order of the paper (its Table 1: mean absolute error 2.41 mm for the radar,
1.81 for the mean field bias, 1.52 for conditional merging and 1.49 for kriging
with an external drift, from a dense network of 1 gauge per 135 km$^2$; here
1 per 250 km$^2$). The gauges alone (`"ok"`) cannot place the cells between
the gauges, so with this network they hardly improve on the radar.

```{code-cell} ipython3
fig, axes = plt.subplots(2, 4, figsize=(17, 8), layout="constrained", sharex=True, sharey=True)
panels = [
    (truth, "truth"),
    (radar, "radar"),
    (merged["mfb"].RAIN_ADJUSTED, "mean field bias"),
    (merged["ok"].RAIN_ADJUSTED, "ordinary kriging of the gauges"),
    (merged["conditional"].RAIN_ADJUSTED, "conditional merging"),
    (merged["ked"].RAIN_ADJUSTED, "kriging with external drift"),
    (merged["ked"].RAIN_ADJUSTED_STD, "KED kriging standard deviation"),
    ((merged["ked"].RAIN_ADJUSTED - truth), "KED minus truth"),
]
for ax, (da, title) in zip(axes.ravel(), panels):
    diff = "minus" in title
    spread = "standard deviation" in title
    pm = ax.pcolormesh(
        da.x / 1000, da.y / 1000, da,
        cmap="RdBu_r" if diff else "turbo", vmin=-10 if diff else 0,
        vmax=10 if diff else 8 if spread else 30, shading="auto",
    )
    ax.plot(gauges.x / 1000, gauges.y / 1000, "k^", ms=4)
    ax.set(title=title, aspect="equal")
    fig.colorbar(pm, ax=ax, shrink=0.8, label="mm")
plt.show()
```

The kriging standard deviation (seventh panel) is zero at the gauges and
grows with the distance to them. Conditional merging and kriging with an
external drift keep the structure of the radar field between the gauges,
which the mean field bias does too but at the wrong level where the radar
error is not uniform.

## The same through the accessor, and the correct units

`radar.radarx.gauge_adjust(gauges, ...)` is the same function. The result is
a Dataset: the adjusted field (float32, in the units of the radar), the number
of gauges used for each time (0 where the radar is returned unchanged because
there are too few), the kriging standard deviation, and for the factor
methods the factors.

```{code-cell} ipython3
res = radar.radarx.gauge_adjust(gauges, method="ked", radius=1500.0)
print(list(res.data_vars), "gauges used:", int(res.GAUGES_USED))
print({k: v for k, v in res.attrs.items()})
print("same as the function:", bool(np.allclose(res.RAIN_ADJUSTED, merged["ked"].RAIN_ADJUSTED)))
radar_at = radar_at_gauges(radar, gauges, radius=1500.0)
print(f"radar at the gauges: mean {float(radar_at.mean()):.1f} mm against {float(gauges.mean()):.1f} mm measured")
```

## Many times in one call

Real use is a series: hourly or daily fields and gauge records. Leading
dimensions such as `time` are processed in one call; a gauge with a missing
value at a time is left out for that time. With a variogram that is the same
for all times (`variogram=`) the kriging weights are computed once and the
whole series costs little more than one field.

```{code-cell} ipython3
nt = 24
scale = 0.5 + rng.random(nt)  # each hour a different amount of rain
series = xr.concat([radar * s for s in scale], dim="time").assign_coords(time=np.arange(nt))
records = xr.concat([gauges * s * (1 + 0.03 * rng.standard_normal(gauges.size)) for s in scale], dim="time")
records = records.assign_coords(time=np.arange(nt))
records[5, :6] = np.nan  # six gauges did not report in hour 5
fixed = dict(model="exponential", nugget=float(vario.nugget), partial_sill=float(vario.partial_sill), range=float(vario["range"]))
t0 = time.perf_counter()
out = gauge_adjust(series, records, method="conditional", variogram=fixed, radius=1500.0)
print(f"{nt} hours: {time.perf_counter() - t0:.2f} s; gauges used per hour:", out.GAUGES_USED.values)
```

## The density of the network

Goudenhoofdt and Delobbe (2009, Sect. 5) remove gauges from the network and
find that the geostatistical methods gain the most from a dense network and
are the best at every density. The same experiment on this field, with 12
random networks for every number of gauges and the error against the truth on
the whole grid (the truth is known here):

```{code-cell} ipython3
counts = [10, 20, 40, 80, 160]
radar_rmse = float(np.sqrt(((radar - truth) ** 2).mean()))
rmse = {m: np.zeros((len(counts), 12)) for m in ["mfb", "brandes", "ok", "conditional", "ked"]}
for i, count in enumerate(counts):
    for rep in range(12):
        net = network(count, 1000 * count + rep)
        for m in rmse:
            res = gauge_adjust(radar, net, method=m, radius=1500.0)
            rmse[m][i, rep] = float(np.sqrt(((res.RAIN_ADJUSTED - truth) ** 2).mean()))
fig, ax = plt.subplots(figsize=(5.8, 3.8), layout="constrained")
ax.axhline(radar_rmse, color="k", ls=":", label="radar")
for m, v in rmse.items():
    ax.errorbar(counts, v.mean(axis=1), v.std(axis=1), marker="o", capsize=2, label=m)
ax.set(xscale="log", xlabel="number of gauges", ylabel="RMSE against the truth (mm)")
ax.legend()
plt.show()
```

The two kriging methods that use the radar are the best at every density,
the gauges alone (`"ok"`) approach them only with 160 gauges, and the mean field
bias does not improve with more gauges because its error is the spatial
structure of the radar error, which one factor cannot remove.

## What to check before trusting the result

* The adjustment is only as good as the gauges: a gauge under-catches in
  wind and the rain it measures is a point value, so compare accumulations of
  at least an hour, as the paper does with days.
* Correct the radar for attenuation, calibration and clutter first (see the
  [rainfall estimation notebook](Rainfall_Estimation)); merging is the last
  step.
* Outside the convex hull of the gauges the adjusted field is an
  extrapolation: look at `RAIN_ADJUSTED_STD`.
* `"ked"` falls back to ordinary kriging where the radar is constant among
  the nearest gauges, for example when there is no rain in the radar field;
  `res.attrs["ked_fallback"]` counts those pixels.
* The merged field of a kriging method can be negative; `clip=True` sets it
  to zero.

## References

* Goudenhoofdt, E., and L. Delobbe, 2009: Evaluation of radar-gauge merging
  methods for quantitative precipitation estimates. *Hydrol. Earth Syst. Sci.*,
  **13**, 195-203, <https://doi.org/10.5194/hess-13-195-2009>
* Sinclair, S., and G. Pegram, 2005: Combining radar and rain gauge rainfall
  estimates using conditional merging. *Atmos. Sci. Lett.*, **6**, 19-22,
  <https://doi.org/10.1002/asl.85>
* Brandes, E. A., 1975: Optimizing rainfall estimates with the aid of radar.
  *J. Appl. Meteor.*, **14**, 1339-1345,
  <https://doi.org/10.1175/1520-0450(1975)014<1339:OREWTA>2.0.CO;2>
* Smith, J. A., and W. F. Krajewski, 1991: Estimation of the mean field bias of
  radar rainfall estimates. *J. Appl. Meteor.*, **30**, 397-412,
  <https://doi.org/10.1175/1520-0450(1991)030<0397:EOTMFB>2.0.CO;2>
* Wackernagel, H., 2003: *Multivariate Geostatistics*. 3rd ed. Springer,
  <https://doi.org/10.1007/978-3-662-05294-5>
* Cressie, N., 1985: Fitting variogram models by weighted least squares.
  *J. Int. Assoc. Math. Geol.*, **17**, 563-586,
  <https://doi.org/10.1007/BF01032109>
* Matheron, G., 1963: Principles of geostatistics. *Econ. Geol.*, **58**,
  1246-1266, <https://doi.org/10.2113/gsecongeo.58.8.1246>
