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

# Drop Size Distribution Retrieval

Rain drop size distributions (DSDs) are well described by a gamma
distribution $N(D) = N_0 D^{\mu} e^{-\Lambda D}$ (Ulbrich 1983). Its three
parameters are linked to the polarimetric radar variables: the differential
reflectivity $Z_{DR}$ depends only on the shape of the DSD (the drop sizes),
while $Z_H$ and $K_{DP}$ also scale with the number of drops.

`radarx.retrieve.dsd` (or `.radarx.dsd()` on a sweep, a volume, a grid or a
QVP) retrieves the DSD at every gate with lookup tables computed from
T-matrix scattering tables shipped with radarx (S, C and X band, 0-30 °C):

- `method="constrained"` (default): the constrained-gamma DSD of Zhang et al.
  (2001), with the $\mu$–$\Lambda$ relation of Cao et al. (2008);
- `method="normalized"`: the normalized gamma DSD (Testud et al. 2001;
  Bringi et al. 2002) with a fixed shape $\mu$.

With `kdp=...`, the intercept is taken from $K_{DP}$ where it is large enough,
which makes the water content and rain rate independent of the $Z_H$
calibration. The outputs are $N_0$, $N_w$, $D_0$, $D_m$, $\mu$, $\Lambda$,
the rain rate and the liquid water content. A compiled kernel processes all
gates of a volume in one multithreaded call.

```{code-cell} ipython3
import fsspec
import matplotlib.pyplot as plt
import numpy as np
import xradar as xd

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.retrieve import (
    dsd,
    dsd_spectrum,
    fit_gamma_moments,
    radar_from_dsd,
    scattering_table,
)
```

## Scattering tables

Single-drop scattering of oblate, slightly canted raindrops. At S band the
drops are Rayleigh scatterers and $Z_{DR}$ grows smoothly with size; at C band
drops of 5-7 mm resonate.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6, 4))
for band in "SCX":
    tab = scattering_table(band, temperature=20.0)
    zdr = 10 * np.log10(tab.sigma_h / tab.sigma_v)
    ax.plot(tab.diameter, zdr, label=f"{band} band ({tab.attrs['wavelength']:.0f} mm)")
ax.set_xlabel("equal-volume diameter (mm)")
ax.set_ylabel("single-drop $Z_{DR}$ (dB)")
ax.legend()
ax.grid(alpha=0.3);
```

## Scattering by single drops: canting, frequency, temperature

The tables above are fixed to 7 degrees of canting and three frequencies.
`radarx.fundamentals.raindrop_scattering` computes the same quantities for any
wavelength, temperature, axis ratio and canting width with a threaded C++
T-matrix code (a translation of the code of Mishchenko that `pytmatrix` wraps;
the 1920 rows of the packaged table are reproduced to 1e-6), so the assumptions
behind the tables can be changed and their effect measured. At S band (2.8 GHz,
20 degrees Celsius) the width of the canting distribution matters more than the
frequency or the temperature:

```{code-cell} ipython3
from radarx.fundamentals import raindrop_scattering

d = np.arange(0.5, 8.01, 0.25)
wavelength = 299.792458 / 2.8  # mm
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
for std in (0.5, 7.0, 10.0, 20.0):
    out = raindrop_scattering(d, wavelength, canting_std=std)
    axes[0].plot(d, 10 * np.log10(out.sigma_h / out.sigma_v), label=f"canting {std:g}°")
    axes[1].plot(d, out.kdp, label=f"canting {std:g}°")
axes[0].set(xlabel="equal-volume diameter (mm)", ylabel="single-drop $Z_{DR}$ (dB)")
axes[1].set(xlabel="equal-volume diameter (mm)", ylabel="$K_{DP}$ per drop per m$^3$ (° km$^{-1}$)")
axes[0].legend()
for ax in axes:
    ax.grid(alpha=0.3)
```

Going from 7 to 20 degrees lowers $Z_{DR}$ by 0.65 dB for a 4 mm drop and by
1.7 dB for an 8 mm drop, and $K_{DP}$ by 27 % at every size. Between 2.7 and
3.0 GHz the $Z_{DR}$ of drops up to 6 mm changes by less than 0.02 dB, but that
of 8 mm drops by 0.36 dB, and $K_{DP}$, which scales as $1/\lambda$ for small
drops (a factor 1.11), is 26 % larger for 8 mm drops; between 0 and 30 degrees
Celsius the $Z_{DR}$ of 8 mm drops changes by 0.3 dB. The
differences between scattering assumptions in the literature (Kumjian and Ryzhkov 2010
use 10.9 cm and 20 degrees of canting, which they call too wide for typical rainfall, where
10 degrees is more appropriate; here 10.7 cm and 7 degrees) are therefore of the size of the
large-drop signal itself. The tables record their assumptions (the width of the
canting distribution and the frequency are attributes of `scattering_table` and
of the `dsd` and `radar_from_dsd` products), and
`python ci/build_dsd_tables.py output.csv 10` writes a table for another width
in a few seconds.

For gamma DSDs the effect is smaller than for the largest drops, but not small
for the retrieval. The cell below gives the change of $Z_{DR}$ and $K_{DP}$ of
normalized gamma DSDs ($N_w$ = 8000 m$^{-3}$ mm$^{-1}$, S band, 2.8 GHz) when the
canting grows from the 7 degrees of the tables, and the $D_m$ that the table of
7 degrees would retrieve from the $Z_{DR}$ of the more canted rain:

```{code-cell} ipython3
from scipy.integrate import trapezoid
from scipy.special import gamma as gamma_function

print(
    "table:",
    scattering_table("S").attrs["canting_std"],
    "degrees,",
    scattering_table("S").attrs["frequency_ghz"],
    "GHz",
)
diam = scattering_table("S").diameter.values
wavelength = 299.792458 / 2.8


def zdr_kdp(std, mu, dm, nw=8000.0):
    out = raindrop_scattering(diam, wavelength, canting_std=std, ddelt=1e-3)
    f = 6.0 / 256.0 * (4 + mu) ** (mu + 4) / gamma_function(mu + 4)
    nd = nw * f * (diam / dm) ** mu * np.exp(-(4 + mu) * diam / dm)
    zh = trapezoid(nd * out.sigma_h.values, diam)
    zv = trapezoid(nd * out.sigma_v.values, diam)
    return 10 * np.log10(zh / zv), trapezoid(nd * out.kdp.values, diam)


dm_grid = np.linspace(0.5, 4.0, 141)
zdr_7 = {
    mu: np.array([zdr_kdp(7.0, mu, x)[0] for x in dm_grid]) for mu in (3.0, -1.0)
}
print("mu  Dm  canting   dZDR (dB)  dKDP (%)  Dm retrieved with 7 degrees")
for mu in (3.0, -1.0):
    for dm in (1.5, 2.0, 3.0):
        z7, k7 = zdr_kdp(7.0, mu, dm)
        for std in (10.0, 20.0):
            z, k = zdr_kdp(std, mu, dm)
            retrieved = np.interp(z, zdr_7[mu], dm_grid)
            print(
                f"{mu:3.0f} {dm:3.1f} {std:6.0f}   {z - z7:9.2f}  {100 * (k / k7 - 1):8.1f}"
                f"  {retrieved:6.2f}"
            )
```

A table of 7 degrees applied to rain canted by 10 degrees retrieves a $D_m$ 2-4 %
too low for $\mu$ = 3 (3-7 % for $\mu$ = -1); for 20 degrees it is 12-21 % too
low (17-32 %), more than the 6-8 % of an elevation angle of 20 degrees (the last
column). Between 2.7 and 3.0 GHz the $Z_{DR}$ of such DSDs changes by less than
0.06 dB and $K_{DP}$ by -4 to +9 %.

## A squall line in Mississippi

The lowest sweep of the KGWX (Columbus, Mississippi) WSR-88D volume of 30
March 2022, 23:46 UTC. NEXRAD no-data codes are masked. The retrieval assumes
that every gate is rain: hail, the melting layer, biological or ground echo
give meaningless DSDs (hail with $Z_H$ of 55 dBZ and $Z_{DR}$ near 0 dB would
be read as an enormous number of tiny drops). A crude rain mask stands in for
a hydrometeor classification here: $\rho_{hv} \geq 0.97$, $Z_H \geq 5$ dBZ,
$Z_{DR} \leq 3.5$ dB, no hail signature ($Z_H \geq 50$ dBZ with
$Z_{DR} < 0.5$ dB), and within 120 km, well below the melting layer at this
elevation. $Z_{DR}$ is averaged over five gates (1.25 km) in range, which
reduces its noise.

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
sweep["ZDR"] = sweep.ZDR.rolling(range=5, center=True, min_periods=3).mean()
hail = (sweep.DBZH >= 50) & (sweep.ZDR < 0.5)
rain = (
    (sweep.RHOHV >= 0.97)
    & (sweep.DBZH >= 5)
    & (sweep.ZDR <= 3.5)
    & ~hail
    & (sweep.range <= 120e3)
)
```

The radar band is read from the `frequency` metadata when there is one, and a
WSR-88D volume is recognized as S band; a single NEXRAD sweep has neither, so
the band is passed:

```{code-cell} ipython3
out = sweep.radarx.dsd(mask=rain, band="S")
out
```

```{code-cell} ipython3
fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
x, y = sweep.x / 1e3, sweep.y / 1e3
panels = [
    (sweep.DBZH.where(rain), "$Z_H$ (dBZ)", "turbo", 0, 60),
    (out.D0, "$D_0$ (mm)", "plasma", 0.5, 3.0),
    (np.log10(out.NW), "$\\log_{10} N_w$ (m$^{-3}$ mm$^{-1}$)", "viridis", 2, 5),
    (out.RAIN_RATE, "rain rate (mm h$^{-1}$)", "turbo", 0, 60),
]
for ax, (da, title, cmap, vmin, vmax) in zip(axes.flat, panels):
    pm = ax.pcolormesh(x, y, da, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(pm, ax=ax, shrink=0.8)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(-120, 120)
    ax.set_ylim(-120, 120)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
```

The convective line has large drops ($D_0 > 2$ mm) and high rain rates; the
trailing stratiform rain has smaller drops. The retrieved rain rate follows
the WSR-88D convective $Z$–$R$ relation $Z = 300 R^{1.4}$ (Fulton et al. 1998)
on average, but varies around it with the drop sizes:

```{code-cell} ipython3
r_zr = (10 ** (sweep.DBZH / 10) / 300.0) ** (1 / 1.4)
ok = (rain & (out.RAIN_RATE > 0.1)).values
bins = np.logspace(-1, 2.3, 50)
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.hist2d(r_zr.values[ok], out.RAIN_RATE.values[ok], bins=[bins, bins], cmin=1, cmap="Blues")
ax.plot([0.1, 200], [0.1, 200], "k-", lw=0.8)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("$Z = 300 R^{1.4}$ (mm h$^{-1}$)")
ax.set_ylabel("DSD retrieval (mm h$^{-1}$)")
ratio = out.RAIN_RATE.values[ok].sum() / r_zr.values[ok].sum()
ax.set_title(f"accumulated ratio {ratio:.2f}");
```

A few gates in the cores above about 58 dBZ get more than 200 mm/h, where
hail probably mixes with the rain; the crude mask above does not catch it,
a hydrometeor classification would.

## Normalized gamma with $K_{DP}$

With `kdp="estimate"`, $K_{DP}$ is computed by
`radarx.retrieve.estimate_kdp`; where it reaches `min_kdp` (1 °/km by
default), $N_w$ comes from $K_{DP}$ and $Z_{DR}$ instead of $Z_H$.

```{code-cell} ipython3
norm = dsd(sweep, "normalized", kdp="estimate", mask=rain, band="S")
fig, ax = plt.subplots(figsize=(5, 4.5))
ok = (rain & np.isfinite(norm.DM)).values
ax.hist2d(
    norm.DM.values[ok],
    np.log10(norm.NW.values[ok]),
    bins=[np.linspace(0.3, 4, 75), np.linspace(1, 6, 60)],
    cmin=1,
    cmap="magma_r",
)
ax.set_xlabel("$D_m$ (mm)")
ax.set_ylabel("$\\log_{10} N_w$")
ax.set_title(norm.attrs["comment"].split("; ")[-1], fontsize=8);
```

## Back to drop spectra

`dsd_spectrum` rebuilds $N(D)$ on disdrometer size bins (the 32 classes of
the OTT Parsivel by default), for comparisons with disdrometers.
`fit_gamma_moments` fits a gamma DSD to measured spectra by the method of
moments and `radar_from_dsd` simulates the radar variables of any spectrum.
Here for a gate of the convective line with about 50 mm/h:

```{code-cell} ipython3
core = abs(out.RAIN_RATE - 50.0).argmin(...)
gate = out.isel(core)
nd = dsd_spectrum(gate)
print(
    f"convective gate: R = {float(gate.RAIN_RATE):.0f} mm/h, D0 = {float(gate.D0):.2f} mm, "
    f"mu = {float(gate.MU):.2f}"
)
print("fit of the binned spectrum:", {k: round(float(v), 2) for k, v in fit_gamma_moments(nd)[["MU", "LAMBDA", "DM"]].items()})
sim = radar_from_dsd(nd, band="S")
print(
    f"simulated from the Parsivel bins: ZH {float(sim.DBZH):.1f} dBZ "
    f"(observed {float(sweep.DBZH.isel(core)):.1f}), ZDR {float(sim.ZDR):.2f} dB "
    f"(observed {float(sweep.ZDR.isel(core)):.2f})"
)
fig, ax = plt.subplots(figsize=(6, 4))
ax.step(nd.diameter, nd, where="mid")
ax.set_yscale("log")
ax.set_xlim(0, 8)
ax.set_ylim(1e-2, None)
ax.set_xlabel("diameter (mm)")
ax.set_ylabel("$N(D)$ (m$^{-3}$ mm$^{-1}$)");
```

## References

- Ulbrich, C. W., 1983: Natural variations in the analytical form of the
  raindrop size distribution. *J. Climate Appl. Meteor.*, **22**, 1764–1775,
  <https://doi.org/10.1175/1520-0450(1983)022<1764:NVITAF>2.0.CO;2>
- Zhang, G., J. Vivekanandan, and E. Brandes, 2001: A method for estimating
  rain rate and drop size distribution from polarimetric radar measurements.
  *IEEE Trans. Geosci. Remote Sens.*, **39**, 830–841,
  <https://doi.org/10.1109/36.917906>
- Testud, J., S. Oury, R. A. Black, P. Amayenc, and X. Dou, 2001: The concept
  of "normalized" distribution to describe raindrop spectra: A tool for cloud
  physics and cloud remote sensing. *J. Appl. Meteor.*, **40**, 1118–1140,
  <https://doi.org/10.1175/1520-0450(2001)040<1118:TCONDT>2.0.CO;2>
- Bringi, V. N., G.-J. Huang, V. Chandrasekar, and E. Gorgucci, 2002: A
  methodology for estimating the parameters of a gamma raindrop size
  distribution model from polarimetric radar data: Application to a
  squall-line event from the TRMM/Brazil campaign. *J. Atmos. Oceanic
  Technol.*, **19**, 633–645,
  <https://doi.org/10.1175/1520-0426(2002)019<0633:AMFETP>2.0.CO;2>
- Cao, Q., G. Zhang, E. Brandes, T. Schuur, A. Ryzhkov, and K. Ikeda, 2008:
  Analysis of video disdrometer and polarimetric radar data to characterize
  rain microphysics in Oklahoma. *J. Appl. Meteor. Climatol.*, **47**,
  2238–2255, <https://doi.org/10.1175/2008JAMC1732.1>
- Fulton, R. A., J. P. Breidenbach, D.-J. Seo, D. A. Miller, and T.
  O'Bannon, 1998: The WSR-88D rainfall algorithm. *Wea. Forecasting*, **13**,
  377–395, <https://doi.org/10.1175/1520-0434(1998)013<0377:TWRA>2.0.CO;2>
