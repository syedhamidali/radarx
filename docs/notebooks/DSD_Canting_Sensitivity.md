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

# Canting and frequency sensitivity of the DSD scattering tables

The drop size distribution retrieval (`dsd`, see the
[DSD retrieval notebook](DSD_Retrieval)) reads the scattering of single raindrops
from a table. The table is fixed to a width of 7° of the canting angle (the tilt
of the drops from the vertical) and to three frequencies, and it records both as
attributes. The assumptions are not small: the single-drop $Z_{DR}$ and $K_{DP}$
of large drops change with them by as much as the signal of the drops itself.

`radarx.fundamentals.raindrop_scattering` computes the same quantities for any
wavelength, temperature, axis ratio and canting width with a threaded T-matrix code
(the packaged table is reproduced to 1e-6), so the assumptions can be changed and
their effect measured. This notebook does that at S band (2.8 GHz, 20 °C) in three
steps: single drops, gamma DSDs, and the error that a table of 7° makes in the
retrieved median volume diameter $D_m$.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.special import gamma as gamma_function

from radarx.fundamentals import raindrop_scattering
from radarx.retrieve import scattering_table

tab = scattering_table("S")
print(
    "the table of S band assumes",
    tab.attrs["canting_std"], "degrees of canting at",
    tab.attrs["frequency_ghz"], "GHz",
)
```

## Single drops

The cell below gives the $Z_{DR}$ and $K_{DP}$ of one drop of each size for four
widths of the canting distribution. A width of 10° is what Kumjian and Ryzhkov
(2010) consider more appropriate for typical rain, 20° is what they use.

```{code-cell} ipython3
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
plt.show()
```

Going from 7° to 20° lowers $Z_{DR}$ by 0.65 dB for a 4 mm drop and by 1.7 dB for an
8 mm drop, and $K_{DP}$ by 27 % at every size. The frequency matters less: between
2.7 and 3.0 GHz the $Z_{DR}$ of drops up to 6 mm changes by less than 0.02 dB, but
that of 8 mm drops by 0.36 dB. $K_{DP}$ scales as $1/\lambda$ for small drops (a
factor 1.11) and is 26 % larger for 8 mm drops. Between 0 and 30 °C the $Z_{DR}$ of
8 mm drops changes by 0.3 dB.

## Gamma size distributions

For a normalized gamma DSD ($N_w$ = 8000 m$^{-3}$ mm$^{-1}$) the sizes are weighted
and the effect is smaller than for the largest drops. The function below gives
$Z_{DR}$ and $K_{DP}$ of a DSD of shape $\mu$ and median volume diameter $D_m$,
for a canting width and a frequency.

```{code-cell} ipython3
diam = tab.diameter.values


def zdr_kdp(std, mu, dm, ghz=2.8, nw=8000.0):
    out = raindrop_scattering(diam, 299.792458 / ghz, canting_std=std, ddelt=1e-3)
    f = 6.0 / 256.0 * (4 + mu) ** (mu + 4) / gamma_function(mu + 4)
    nd = nw * f * (diam / dm) ** mu * np.exp(-(4 + mu) * diam / dm)
    zh = trapezoid(nd * out.sigma_h.values, diam)
    zv = trapezoid(nd * out.sigma_v.values, diam)
    return 10 * np.log10(zh / zv), trapezoid(nd * out.kdp.values, diam)


print("mu  Dm  canting   dZDR (dB)  dKDP (%)")
for mu in (3.0, -1.0):
    for dm in (1.5, 2.0, 3.0):
        z7, k7 = zdr_kdp(7.0, mu, dm)
        for std in (10.0, 20.0):
            z, k = zdr_kdp(std, mu, dm)
            print(f"{mu:3.0f} {dm:3.1f} {std:6.0f}   {z - z7:9.2f}  {100 * (k / k7 - 1):8.1f}")
```

## Error of the retrieved $D_m$

A retrieval that reads $D_m$ from $Z_{DR}$ with the table of 7° is biased when
the rain is more canted. The cell below takes the $Z_{DR}$ of DSDs with canting of
10° and 20°, and gives the $D_m$ that the table of 7° would retrieve from it. It does
the same for a radar of 2.7 and 3.0 GHz instead of 2.8 GHz, still with 7°.

```{code-cell} ipython3
dm_grid = np.linspace(0.5, 5.0, 91)
zdr_table = {mu: np.array([zdr_kdp(7.0, mu, x)[0] for x in dm_grid]) for mu in (3.0, -1.0)}
show = (dm_grid >= 1.0) & (dm_grid <= 3.5)
cases = (("canting 10°, 2.8 GHz", 10.0, 2.8, "-"), ("canting 20°, 2.8 GHz", 20.0, 2.8, "--"),
         ("7°, 2.7 GHz", 7.0, 2.7, ":"), ("7°, 3.0 GHz", 7.0, 3.0, ":"))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained", sharey=True)
for ax, mu in zip(axes, (3.0, -1.0)):
    for label, std, ghz, style in cases:
        z = np.array([zdr_kdp(std, mu, x, ghz=ghz)[0] for x in dm_grid])
        error = 100 * (np.interp(z, zdr_table[mu], dm_grid) / dm_grid - 1)
        ax.plot(dm_grid[show], error[show], style, label=label)
        print(f"mu {mu:g}, {label}, error at Dm = 1.5, 2, 3 mm:",
              np.round(np.interp([1.5, 2.0, 3.0], dm_grid, error), 1), "%")
    ax.set(xlabel="$D_m$ (mm)", title=f"$\\mu$ = {mu:g}")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("error of the retrieved $D_m$ (%)")
axes[0].legend(frameon=False)
plt.show()
```

For $D_m$ from 1.5 to 3 mm, a table of 7° applied to rain canted by 10° retrieves a
$D_m$ 2 to 4 % too low for $\mu$ = 3 (3 to 7 % for $\mu$ = -1). For 20° it is 14 to
21 % too low (17 to 32 %). A change of the frequency between 2.7 and 3.0 GHz changes
the $Z_{DR}$ of such DSDs by less than 0.06 dB and $K_{DP}$ by -4 to +9 %, and moves
the retrieved $D_m$ by less than 3 %. The canting width is the assumption to choose
with care, and `python ci/build_dsd_tables.py output.csv 10` writes a table for
another width in a few seconds.

## References

- Kumjian, M. R., and A. V. Ryzhkov, 2010: The Impact of Evaporation on Polarimetric Characteristics of Rain: Theoretical Model and Practical Implications. *Journal of Applied Meteorology and Climatology*, **49**, 1247-1267, <https://doi.org/10.1175/2010JAMC2243.1>
