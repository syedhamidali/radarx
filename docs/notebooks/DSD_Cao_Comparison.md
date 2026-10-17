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

# Constrained-Gamma Retrieval against the Cao et al. (2008) Estimators

Cao et al. (2008) fitted polynomials of $R/Z_H$, $W/Z_H$ and $D_0$ against
$Z_{DR}$ (their Eqs. 9 to 11) to Oklahoma 2D video disdrometer spectra that they
had sorted and averaged on a grid of rain rate and $D_0$. They state a bias of the
fits against the disdrometer of 9 % in $R$ and 7 % in $D_0$ (their Table 4)
for $Z_{DR}$ from 0.15 to 4 dB. The constrained-gamma retrieval of
`radarx.retrieve.dsd` with the $\mu$–$\Lambda$ relation of the same paper
(their Eq. 7) agrees with the polynomials within their scatter for $Z_{DR}$ up to
1 dB. At 1.5 to 2 dB it gives a rain rate about 25 % higher and a median volume
diameter about 14 % lower. This notebook separates the causes with the packaged
T-matrix code (`radarx.fundamentals.raindrop_scattering`):

1. the fall speed (the Atlas et al. 1973 law that the closed-form rain rate uses,
   which is negative below 0.109 mm);
2. the approximation $D_0 \approx (3.67 + \mu)/\Lambda$;
3. the scattering assumptions behind $Z_H$ and $Z_{DR}$ (canting, frequency,
   temperature) and the largest drop;
4. the constrained family itself: spectra that follow the $\mu$–$\Lambda$
   relation but have a lighter tail than the family assumes.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.special import gamma

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.fundamentals import raindrop_scattering
from radarx.retrieve import dsd, terminal_fall_speed

CAO = (-0.0201, 0.902, -1.718)  # mu(Lambda), Cao et al. (2008) Eq. 7
WL = 299.792458 / 2.8  # S band at 2.8 GHz, mm


def cao_r_over_z(zdr):
    """R / Z_H (mm h-1 per mm6 m-3), Cao et al. (2008) Eq. 9."""
    x = zdr
    return 10 ** (-0.0363 * x**3 + 0.316 * x**2 - 1.178 * x - 1.964)


def cao_d0(zdr):
    """D0 (mm), Cao et al. (2008) Eq. 11."""
    x = zdr
    return 0.0436 * x**3 - 0.216 * x**2 + 1.076 * x + 0.659
```

## The constrained family from T-matrix scattering

Drops of 0.05 to 8 mm on a 0.05 mm grid, 20 °C, 7 degrees of canting, 2.8 GHz,
the same choices as the packaged table. Every member of the family
$N(D) = N_0 D^{\mu} e^{-\Lambda D}$ with $\mu(\Lambda)$ from Eq. 7 gives one
$Z_{DR}$; $R/Z_H$ and $D_0$ follow from the spectrum alone.

```{code-cell} ipython3
def drops(canting=7.0, dmax=8.0, wavelength=WL, temperature=20.0):
    d = np.arange(0.05, dmax + 1e-9, 0.05)
    sc = raindrop_scattering(
        d, wavelength, temperature=temperature, canting_std=canting
    )
    return d, sc.sigma_h.values, sc.sigma_v.values


def weights(d):
    w = np.zeros_like(d)
    step = np.diff(d)
    w[:-1] += step / 2
    w[1:] += step / 2
    return w


def radar(d, sh, sv, nd, wavelength=WL):
    """Z_H (dBZ, for the intercept in nd) and Z_DR (dB)."""
    k = wavelength**4 / (np.pi**5 * 0.93)
    zh = k * (nd * weights(d)) @ sh
    zv = k * (nd * weights(d)) @ sv
    return 10 * np.log10(zh), 10 * np.log10(zh / zv)


def rain_rate(d, nd, law):
    v = terminal_fall_speed(d, law=law).values
    return 6e-4 * np.pi * (nd * weights(d) * v * d**3).sum(-1)


def median_diameter(d, nd):
    c = cumulative_trapezoid(nd * d**3, d, initial=0.0)
    return np.interp(0.5, c / c[-1], d)


lam = np.geomspace(0.8, 12.0, 300)
mu = CAO[0] * lam**2 + CAO[1] * lam + CAO[2]


def family(d, mu, lam):
    return d[None] ** mu[:, None] * np.exp(-lam[:, None] * d[None])


def curves(x, law="atlas1973", **kw):
    """R / Z_H and D0 of the family at Z_DR = x, relative to Cao et al. (2008)."""
    d, sh, sv = drops(**kw)
    nd = family(d, mu, lam)
    zh, zdr = radar(d, sh, sv, nd, kw.get("wavelength", WL))
    o = np.argsort(zdr)
    r = rain_rate(d, nd, law) / 10 ** (zh / 10)
    d0 = np.array([median_diameter(d, n) for n in nd])
    nan = dict(left=np.nan, right=np.nan)  # no extrapolation beyond the family
    return (
        np.interp(x, zdr[o], r[o], **nan) / cao_r_over_z(x),
        np.interp(x, zdr[o], d0[o], **nan) / cao_d0(x),
    )
```

The retrieval of radarx at 40 dBZ and the family built here agree, which shows
that the packaged lookup table and `dsd` do what the T-matrix simulation says:

```{code-cell} ipython3
x = np.array([0.5, 1.0, 1.5, 1.75, 2.0, 2.5, 3.0])
ds = xr.Dataset(
    {
        "DBZH": (("azimuth", "range"), np.full((1, x.size), 40.0)),
        "ZDR": (("azimuth", "range"), x[None]),
    }
)
out = dsd(ds, band="S", nw_range=None)
r_family, d0_family = curves(x, law="atlas1973")
r_dsd = out.RAIN_RATE.values[0] / (1e4 * cao_r_over_z(x))
d0_dsd = out.D0.values[0] / cao_d0(x)

print("Z_DR (dB)             ", *[f"{v:6.2f}" for v in x])
print("R, radarx / Cao Eq. 9 ", *[f"{v:6.3f}" for v in r_dsd])
print("R, T-matrix / Cao     ", *[f"{v:6.3f}" for v in r_family])
print("D0, radarx / Cao Eq 11", *[f"{v:6.3f}" for v in d0_dsd])
print("D0, T-matrix / Cao    ", *[f"{v:6.3f}" for v in d0_family])
```

## 1. The fall speed

The closed-form rain rate integrates $9.65 - 10.3\,e^{-0.6D}$ over
$0 < D < \infty$, so the part below 0.109 mm, where the law is negative, is
included. That part is the integral of the law from 0 to 0.109 mm; its share of
the rain rate is a few parts in $10^{6}$ at any $Z_{DR}$ (below). Replacing the law by the
drag-balance speed of `radarx.retrieve.terminal_fall_speed`, which is positive
and about 0.7 m s$^{-1}$ at 0.2 mm, changes the rain rate by less than
0.1 %. The fall speed is not the cause.

```{code-cell} ipython3
for zdr_value in (0.5, 1.0, 1.75, 3.0):
    d, sh, sv = drops()
    nd = family(d, mu, lam)
    zh, zdr = radar(d, sh, sv, nd)
    o = np.argsort(zdr)
    m = np.interp(zdr_value, zdr[o], mu[o])
    lm = np.interp(zdr_value, zdr[o], lam[o])
    dd = np.linspace(0.0, 0.109, 20001)
    negative = 6e-4 * np.pi * trapezoid(
        (9.65 - 10.3 * np.exp(-0.6 * dd)) * dd ** (m + 3) * np.exp(-lm * dd), dd
    )
    total = (
        6e-4 * np.pi * gamma(m + 4) * lm ** (-(m + 4))
        * (9.65 - 10.3 * (lm / (lm + 0.6)) ** (m + 4))
    )
    print(
        f"Z_DR {zdr_value:4.2f} dB (mu {m:5.2f}, Lambda {lm:4.2f}): "
        f"share of the rain rate from D < 0.109 mm {negative / total:+.1e}"
    )

r_drag, _ = curves(x, law="drag_balance")
print(
    "\nlargest change of R/Z_H with the drag-balance law:",
    f"{np.max(np.abs(r_drag / r_family - 1)):.2%}",
)
```

## 2. The median volume diameter

`dsd` returns $D_0 = (3.67 + \mu)/\Lambda$, exact for $\mu = 0$. The exact median
of the $D^3 N(D)$ distribution of an untruncated gamma spectrum is
$\Lambda^{-1}$ times the 0.5 quantile of a gamma variable of shape $\mu + 4$.

```{code-cell} ipython3
from scipy.special import gammaincinv

exact = gammaincinv(mu + 4, 0.5) / lam
approx = (mu + 3.67) / lam
d, sh, sv = drops()
zh, zdr = radar(d, sh, sv, family(d, mu, lam))
inside = (zdr > 0.3) & (zdr < 3.0)
err = (approx / exact - 1)[inside]
print(
    f"(3.67 + mu) / Lambda against the exact median for Z_DR 0.3 to 3 dB: "
    f"{err.min():+.2%} to {err.max():+.2%}"
)
```

## 3. Scattering assumptions and the largest drop

The paper does not give the canting, the axis ratio or the frequency used for
the disdrometer $Z_{DR}$. The canting matters; the frequency (2.8 or 3.0 GHz)
and the temperature (0 to 30 °C) do not.

```{code-cell} ipython3
variants = {
    "radarx table (7 deg canting, 2.8 GHz, 20 C, 8 mm)": {},
    "canting 0 deg": dict(canting=0.05),
    "canting 10 deg": dict(canting=10.0),
    "3.0 GHz": dict(wavelength=299.792458 / 3.0),
    "0 C": dict(temperature=0.0),
    "30 C": dict(temperature=30.0),
    "largest drop 6 mm": dict(dmax=6.0),
    "largest drop 10 mm": dict(dmax=10.0),
}
xs = np.array([1.0, 1.5, 1.75, 2.0])
rows = {}
print(f"{'':50s} R / Cao Eq. 9 at Z_DR {xs}     D0 / Cao Eq. 11")
for name, kw in variants.items():
    r, d0 = curves(xs, **kw)
    rows[name] = (r, d0)
    print(f"{name:50s} {np.round(r, 2)}   {np.round(d0, 2)}")
```

Zero canting raises the Cao ratio of $R$ to 1.34 at 1.75 dB, 10 degrees lowers it
to 1.17, so the unstated assumption moves the comparison by about 8 points
either way. The maximum drop is a lever of the same size above 2 dB.
None of these removes the offset at 1.5 to 2 dB.

```{code-cell} ipython3
xx = np.linspace(0.4, 3.0, 40)
ds = xr.Dataset(
    {
        "DBZH": (("azimuth", "range"), np.full((1, xx.size), 40.0)),
        "ZDR": (("azimuth", "range"), xx[None]),
    }
)
out = dsd(ds, band="S", nw_range=None)

fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
axes[0, 0].plot(xx, out.RAIN_RATE.values[0], label="radarx constrained gamma")
axes[0, 0].plot(xx, 1e4 * cao_r_over_z(xx), "k--", label="Cao et al. (2008) Eq. 9")
axes[0, 0].set(ylabel="R at 40 dBZ (mm h$^{-1}$)", yscale="log")
axes[0, 1].plot(xx, out.D0.values[0], label="radarx constrained gamma")
axes[0, 1].plot(xx, cao_d0(xx), "k--", label="Cao et al. (2008) Eq. 11")
axes[0, 1].set(ylabel="$D_0$ (mm)")
styles = {
    "canting 0 deg": "C1",
    "canting 10 deg": "C2",
    "largest drop 6 mm": "C3",
}
for name, color in styles.items():
    r, d0 = curves(xx, **variants[name])
    axes[1, 0].plot(xx, r, color, label=name)
    axes[1, 1].plot(xx, d0, color, label=name)
r_drag, d0_drag = curves(xx, law="drag_balance")
r_base, d0_base = curves(xx)
axes[1, 0].plot(xx, r_base, "C0", label="radarx table")
axes[1, 1].plot(xx, d0_base, "C0", label="radarx table")
axes[1, 0].plot(xx, r_drag, "k:", label="drag-balance fall speed")
axes[1, 1].plot(xx, d0_drag, "k:", label="drag-balance fall speed")
axes[1, 0].set(ylabel="$R / R_{\\mathrm{Cao}}$")
axes[1, 1].set(ylabel="$D_0 / D_{0,\\mathrm{Cao}}$")
for ax in axes.ravel():
    ax.set_xlabel("$Z_{DR}$ (dB)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
for ax in axes[1]:
    ax.axhline(1.0, color="gray", lw=0.8)
```

## 4. The constrained family against spectra with a lighter tail

The constrained family has one spectrum per $Z_{DR}$. The estimators of Cao et
al. (2008) average the measured spectra at each $Z_{DR}$, and so can differ from
the family wherever real spectra do not follow the $\mu$–$\Lambda$ line. The tail
of the spectrum is the most sensitive part: $Z_H$ and $Z_{DR}$ weight the large
drops heavily, and the lookup table of `dsd` ignores drops above 8 mm.

To measure the effect, take spectra that follow Eq. 7 exactly but stop at a
largest diameter of 4, 5, 6 or 8 mm, compute their $Z_H$ and $Z_{DR}$ with the
same T-matrix code, retrieve with `dsd` and compare with the rain rate and
median volume diameter of the truncated spectrum itself.

```{code-cell} ipython3
d, sh, sv = drops()
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
summary = {}
for dmax in (4.0, 5.0, 6.0, 8.0):
    k = d <= dmax + 1e-9
    nd = family(d[k], mu, lam)
    zh, zdr = radar(d[k], sh[k], sv[k], nd)
    truth_r = rain_rate(d[k], nd, "drag_balance")
    truth_d0 = np.array([median_diameter(d[k], n) for n in nd])
    ds = xr.Dataset({"DBZH": ("gate", zh), "ZDR": ("gate", zdr)})
    res = dsd(ds, band="S", nw_range=None)
    o = np.argsort(zdr)
    ratio_r = (res.RAIN_RATE.values / truth_r)[o]
    ratio_d0 = (res.D0.values / truth_d0)[o]
    axes[0].plot(zdr[o], ratio_r, label=f"largest drop {dmax:g} mm")
    axes[1].plot(zdr[o], ratio_d0, label=f"largest drop {dmax:g} mm")
    band = (zdr[o] > 1.5) & (zdr[o] < 2.0)
    summary[dmax] = (
        np.nanmean(ratio_r[band]) if band.any() else np.nan,
        np.nanmean(ratio_d0[band]) if band.any() else np.nan,
        zdr.max(),
    )
axes[0].set(xlabel="$Z_{DR}$ of the spectrum (dB)", ylabel="retrieved / true $R$")
axes[1].set(xlabel="$Z_{DR}$ of the spectrum (dB)", ylabel="retrieved / true $D_0$")
for ax in axes:
    ax.axhline(1.0, color="gray", lw=0.8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

for dmax, (r_bias, d0_bias, zmax) in summary.items():
    print(
        f"largest drop {dmax:g} mm (Z_DR up to {zmax:.2f} dB): "
        f"Z_DR 1.5 to 2 dB, R {round(100 * (r_bias - 1)) + 0:+d} %, "
        f"D0 {round(100 * (d0_bias - 1)) + 0:+d} %"
    )
```

A spectrum on the Oklahoma $\mu$–$\Lambda$ line with no drops above 5 mm is
retrieved with the +19 % in rain rate and -13 % in $D_0$ that the comparison with
the polynomials shows at 1.5 to 2 dB. No error in the fall speed or in $D_0$ is
needed.

## Measured spectra

The same comparison with measured spectra (not run here, the data are not
redistributed): 731 one-minute spectra of the OTT Parsivel stations of the PERiLS
2022 field campaign (Dawson et al. 2025, <https://doi.org/10.26023/HFBG-7W5M-WA00>,
the squall lines of 22 and 30 March 2022 in the southeastern United States, rain rate
above 0.5 mm h$^{-1}$, at least 50 drops, quality controlled). $Z_H$ and $Z_{DR}$ come from the radarx
S-band tables (`radar_from_dsd`), so there is no scattering mismatch, and the
retrieved $R$ and $D_0$ are compared with those of the spectrum itself (mean
of the ratio to the observation):

| retrieval | $Z_{DR}$ < 1 dB | 1 to 1.5 dB | 1.5 to 2 dB | 2 to 4 dB |
|---|---|---|---|---|
| number of minutes | 328 | 180 | 116 | 107 |
| constrained, Cao et al. (2008) Eq. 7 | 1.19, 0.87 | 1.23, 0.84 | 1.20, 0.87 | 1.17, 0.89 |
| constrained, Zhang et al. (2001) | 1.05, 1.00 | 1.13, 0.92 | 1.13, 0.93 | 1.13, 0.93 |
| normalized, $\mu$ = 3 | 1.13, 0.92 | 1.06, 1.00 | 0.98, 1.09 | 0.94, 1.15 |
| Cao et al. (2008) Eqs. 9 and 11 | 1.20, 0.90 | 1.06, 0.93 | 0.96, 1.00 | 0.99, 1.04 |

(each cell: $R$, $D_0$ as retrieved over observed). The offset of the constrained
retrieval is the same as against the polynomials, and the polynomials fitted in
Oklahoma describe these spectra at 1.5 dB and above within 4 %. With the
lookup truncated at 5 mm instead of 8 mm the bias at 1.5 to 2 dB becomes +2 % in
$R$ and -2 % in $D_0$, but it stays at +14 % and -8 % when all $Z_{DR}$ are used, so a smaller
largest drop is not a calibrated fix. Parsivel and 2D video disdrometers
differ, and this is one campaign.

## Calibration

A $Z_{DR}$ offset of 0.2 dB, a normal size of calibration error, has the effect of the
bias itself:

```{code-cell} ipython3
for z0 in (1.0, 1.75, 2.5):
    ds = xr.Dataset(
        {
            "DBZH": (("azimuth", "range"), np.full((1, 3), 40.0)),
            "ZDR": (("azimuth", "range"), np.array([[z0 - 0.2, z0, z0 + 0.2]])),
        }
    )
    o = dsd(ds, band="S", nw_range=None)
    r, d0 = o.RAIN_RATE.values[0], o.D0.values[0]
    print(
        f"Z_DR {z0:4.2f} dB: offset of -0.2 / +0.2 dB changes R by "
        f"{r[0] / r[1] - 1:+.0%} / {r[2] / r[1] - 1:+.0%} and D0 by "
        f"{d0[0] / d0[1] - 1:+.0%} / {d0[2] / d0[1] - 1:+.0%}"
    )
```

## When not to use the constrained method

- $Z_{DR}$ above about 1.2 dB. The family allows one spectrum per $Z_{DR}$ and
  the retrieved $R$ is 20 to 25 % high and $D_0$ 12 to 14 % low at 1.5 to 2 dB
  against the Oklahoma polynomials and against measured spectra from the southeastern
  United States. Treat the $R$ and $D_0$ there as biased. For the rain rate use
  the polynomial of Cao et al. (2008, Eq. 9, valid for $Z_{DR}$ from 0.15 to 4 dB)
  or `radarx.retrieve.rain_rate_z_zdr`, which is within 8 % of the measured
  spectra at 1 to 4 dB in the check above.
- Big-drop regions: $Z_{DR}$ high for the reflectivity, as at the leading edge of
  convection. The adjustment of Cao et al. (2008, Eq. 16) is not implemented.
- Uncalibrated or attenuated $Z_{DR}$ and $Z_H$ (see above): $R$ changes by 16 to
  29 % for a 0.2 dB offset in $Z_{DR}$ (and $D_0$ by 6 to 8 %).
- Non-rain echo: hail, the melting layer, ground clutter. Use `mask`.
- $Z_{DR}$ outside the table (flagged in `ZDR_CLIPPED`).

## References

Cao, Q., G. Zhang, E. Brandes, T. Schuur, A. Ryzhkov, and K. Ikeda, 2008:
Analysis of video disdrometer and polarimetric radar data to characterize rain
microphysics in Oklahoma. *J. Appl. Meteor. Climatol.*, **47**, 2238-2255,
<https://doi.org/10.1175/2008JAMC1732.1>

Dawson, D., M. Biggerstaff, and S. Waugh, 2025: PERiLS_2022: Portable In Situ
Precipitation Stations (PIPS) Data. Version 1.0. NSF NCAR Earth Observing
Laboratory, <https://doi.org/10.26023/HFBG-7W5M-WA00>

Zhang, G., J. Vivekanandan, and E. Brandes, 2001: A method for estimating rain
rate and drop size distribution from polarimetric radar measurements. *IEEE
Trans. Geosci. Remote Sens.*, **39**, 830-841,
<https://doi.org/10.1109/36.917906>
