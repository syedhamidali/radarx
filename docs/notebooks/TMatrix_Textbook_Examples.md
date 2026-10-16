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

# T-matrix Scattering: Textbook Examples

Bringi and Chandrasekar (2001, *Polarimetric Doppler Weather Radar*, Cambridge University
Press) tabulate the results of T-matrix simulations of rain at S, C and X band: the
coefficients of the rain rate and water content estimators (Tables 8.1 to 8.3), the
power laws between specific attenuation and specific differential phase (Table 7.1), and
a few fits that every polarimetric radar user knows (Eqs. 7.14 and 8.14). The book
describes the simulations completely, so they can be repeated with
`radarx.fundamentals.raindrop_scattering` and the numbers compared:

- gamma drop size distributions $N(D) = N_w f(\mu)(D/D_0)^{\mu}\exp[-(3.67+\mu)D/D_0]$
  (Eq. 7.62) with $10^3 \le N_w \le 10^5$ mm$^{-1}$ m$^{-3}$ ($\log N_w$ uniform),
  $0.5 \le D_0 \le 2.5$ mm and $-1 \le \mu \le 5$ (both uniform), a largest drop of 8 mm,
  and rain rates of at most 300 mm h$^{-1}$;
- the axis ratio of Beard and Chuang (Eq. 7.3), a Gaussian canting distribution with zero
  mean and a standard deviation of 7 degrees;
- the refractive index of water at 20 degrees Celsius of Ray (1972) at 3, 5.5 and 10 GHz;
- the rain rate $R = 0.6\pi \times 10^{-3} \int v(D) D^3 N(D)\,dD$ with the terminal velocity
  $v = 9.65 - 10.3\exp(-0.6 D)$ (Eqs. 7.65b and 7.66a), and $|K_w|^2 = 0.93$.

The ensemble of the book is not available, so 20 000 distributions are drawn
here; the coefficients of a regression depend a little on the sample, and
the coefficients of one relation depend on each other, so the rain rates the relations
give are compared as well.

```{code-cell} ipython3
from math import lgamma

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from radarx.fundamentals import raindrop_scattering

D = np.arange(0.05, 8.001, 0.05)  # mm
DD = 0.05


def beard_chuang(d):
    """Vertical-to-horizontal axis ratio, Eq. 7.3 of Bringi and Chandrasekar (2001)."""
    return 1.0048 + 5.7e-4 * d - 2.628e-2 * d**2 + 3.682e-3 * d**3 - 1.677e-4 * d**4


velocity = 9.65 - 10.3 * np.exp(-0.6 * D)  # m s-1
K2 = 0.93

rng = np.random.default_rng(2001)
n = 20000
d0 = rng.uniform(0.5, 2.5, n)
mu = rng.uniform(-1.0, 5.0, n)
log_nw = rng.uniform(3.0, 5.0, n)
f_mu = np.array([6 * (3.67 + m) ** (m + 4) / (3.67**4 * np.exp(lgamma(m + 4))) for m in mu])
x = D[None, :] / d0[:, None]
dsd = (10**log_nw)[:, None] * f_mu[:, None] * x ** mu[:, None] * np.exp(-(3.67 + mu[:, None]) * x)

rain_rate = 0.6 * np.pi * 1e-3 * (dsd * D**3 * velocity).sum(axis=1) * DD  # mm h-1
water = np.pi / 6 * 1e-3 * (dsd * D**3).sum(axis=1) * DD  # g m-3
keep = rain_rate <= 300.0
print(f"{keep.sum()} of {n} distributions have a rain rate of at most 300 mm/h")
```

## Rain rate and water content estimators (Tables 8.1 to 8.3)

For each band the single-drop table is computed once (it takes a fraction of a second)
and integrated over all distributions with a matrix product. The estimators
$R(Z_h, Z_{dr}) = c\,Z_h^a\,\zeta_{dr}^b$, $R(K_{dp}) = c\,K_{dp}^b$,
$R(K_{dp}, Z_{dr}) = c\,K_{dp}^a\,\zeta_{dr}^b$ and $W(Z_h, Z_{dr})$ are fitted by
non-linear regression on the linear values, as the book recommends, with
$\zeta_{dr} = 10^{0.1 Z_{dr}}$.

```{code-cell} ipython3
BANDS = {"S": 3.0, "C": 5.45, "X": 10.0}  # GHz
simulated = {}
for band, frequency in BANDS.items():
    wavelength = 299.792458 / frequency
    table = raindrop_scattering(
        D, wavelength, temperature=20.0, canting_std=7.0, axis_ratio=beard_chuang
    )
    zh = wavelength**4 / (np.pi**5 * K2) * (dsd * table.sigma_h.values).sum(axis=1) * DD
    zv = wavelength**4 / (np.pi**5 * K2) * (dsd * table.sigma_v.values).sum(axis=1) * DD
    simulated[band] = dict(
        zh=zh, zeta=zh / zv, kdp=(dsd * table.kdp.values).sum(axis=1) * DD, wavelength=wavelength
    )


def regress(function, y, guess, *variables):
    parameters, _ = curve_fit(function, variables, y, p0=guess, maxfev=20000)
    return parameters


def power2(variables, c, a, b):
    return c * variables[0] ** a * variables[1] ** b


fits = {}
for band, s in simulated.items():
    k = keep
    kk = keep & (s["kdp"] > 0.05)
    fits[band] = dict(
        r_zh_zdr=regress(power2, rain_rate[k], (6e-3, 0.9, -3.0), s["zh"][k], s["zeta"][k]),
        r_kdp=regress(lambda v, c, b: c * v[0] ** b, rain_rate[kk], (50.0, 0.85), s["kdp"][kk]),
        r_kdp_zdr=regress(power2, rain_rate[kk], (50.0, 0.9, -1.0), s["kdp"][kk], s["zeta"][kk]),
        w_zh_zdr=regress(power2, water[k], (1e-3, 0.9, -3.0), s["zh"][k], s["zeta"][k]),
    )

BOOK = {  # Tables 8.1, 8.2, 8.3 and Eq. 8.14 (3 GHz: 50.7, 0.85)
    "S": dict(r_zh_zdr=(6.7e-3, 0.93, -3.43), r_kdp_zdr=(90.8, 0.93, -1.69), w_zh_zdr=(0.7e-3, 0.89, -4.16), r_kdp=(50.7, 0.85)),
    "C": dict(r_zh_zdr=(5.8e-3, 0.91, -2.09), r_kdp_zdr=(37.9, 0.89, -0.72), w_zh_zdr=(0.6e-3, 0.85, -2.36), r_kdp=(129 / 5.45**0.85, 0.85)),
    "X": dict(r_zh_zdr=(3.9e-3, 1.07, -5.97), r_kdp_zdr=(28.6, 0.95, -1.37), w_zh_zdr=(0.9e-3, 0.95, -6.18), r_kdp=(129 / 10.0**0.85, 0.85)),
}
NAMES = dict(
    r_zh_zdr="R(Zh, Zdr)", r_kdp_zdr="R(Kdp, Zdr)", w_zh_zdr="W(Zh, Zdr)", r_kdp="R(Kdp)"
)
rows = []
for relation in ("r_zh_zdr", "r_kdp_zdr", "w_zh_zdr", "r_kdp"):
    for band in BANDS:
        book, ours = BOOK[band][relation], fits[band][relation]
        row = dict(estimator=NAMES[relation], band=band)
        for label, b, o in zip("cab" if len(book) == 3 else "cb", book, ours):
            row[f"{label} (book)"], row[f"{label} (radarx)"] = b, o
        rows.append(row)
pd.DataFrame(rows).set_index(["estimator", "band"]).map(
    lambda v: "" if pd.isna(v) else (f"{v:.2e}" if abs(v) < 0.1 else f"{v:.2f}")
)
```

The multipliers agree to 11 % and the exponent of $Z_h$ or $K_{dp}$ to 0.03 in all
cases but one, and the exponent of $\zeta_{dr}$ to 0.15 in all cases but two. The exceptions
are the fits at X band of $R(Z_h, Z_{dr})$ and $W(Z_h, Z_{dr})$, where the exponent of
$\zeta_{dr}$ is large (-5 to -6) and the multiplier and the two exponents compensate
each other, so that very different coefficients give nearly the same rain rate. The rain rates, not
the coefficients, are therefore compared next: the bias and the normalized standard error
(the root mean square error divided by the mean) of the rain rate or water content that the
coefficients of the book and the fit to this sample give for the simulated distributions.

```{code-cell} ipython3
def evaluate(relation, band, coefficients):
    s = simulated[band]
    kk = keep & (s["kdp"] > 0.05)
    if relation == "r_zh_zdr":
        estimate, truth, sel = power2((s["zh"], s["zeta"]), *coefficients), rain_rate, keep
    elif relation == "r_kdp_zdr":
        estimate, truth, sel = power2((s["kdp"], s["zeta"]), *coefficients), rain_rate, kk
    elif relation == "r_kdp":
        estimate, truth, sel = coefficients[0] * s["kdp"] ** coefficients[1], rain_rate, kk
    else:
        estimate, truth, sel = power2((s["zh"], s["zeta"]), *coefficients), water, keep
    return (
        100 * np.mean(estimate[sel] - truth[sel]) / np.mean(truth[sel]),
        100 * np.sqrt(np.mean((estimate[sel] - truth[sel]) ** 2)) / np.mean(truth[sel]),
    )


rows = []
for relation in ("r_zh_zdr", "r_kdp_zdr", "w_zh_zdr", "r_kdp"):
    for band in BANDS:
        bias_b, nse_b = evaluate(relation, band, BOOK[band][relation])
        bias_o, nse_o = evaluate(relation, band, fits[band][relation])
        rows.append(
            {
                "estimator": NAMES[relation],
                "band": band,
                "bias % (book)": bias_b,
                "nse % (book)": nse_b,
                "bias % (own fit)": bias_o,
                "nse % (own fit)": nse_o,
            }
        )
pd.DataFrame(rows).set_index(["estimator", "band"]).round(1)
```

The relations of the book, which come from the code of the authors, give the rain rate and
water content that the C++ T-matrix computes with biases between -9 and +9 %, and with
normalized errors within 5 percentage points of those of the relation fitted to this very
sample (10 to 35 %; no relation can do better, because it approximates a
three-parameter family of distributions with two radar variables).

## The drop diameter from $Z_{dr}$ (Eq. 7.14)

The mass-weighted mean diameter and the median volume diameter follow from the
differential reflectivity at S band as $D_m = 1.619\,Z_{dr}^{0.485}$ and
$D_0 = 1.529\,Z_{dr}^{0.467}$ (mm, $Z_{dr}$ in dB). The same ensemble, with the
normalization of the textbook, gives:

```{code-cell} ipython3
s = simulated["S"]
zdr_db = 10 * np.log10(s["zeta"])
dm = (4 + mu) / (3.67 + mu) * d0
ok = zdr_db > 0.05
fit_dm = np.polyfit(np.log(zdr_db[ok]), np.log(dm[ok]), 1)
fit_d0 = np.polyfit(np.log(zdr_db[ok]), np.log(d0[ok]), 1)
print(f"Dm = {np.exp(fit_dm[1]):.3f} Zdr^{fit_dm[0]:.3f}   (book: 1.619 Zdr^0.485)")
print(f"D0 = {np.exp(fit_d0[1]):.3f} Zdr^{fit_d0[0]:.3f}   (book: 1.529 Zdr^0.467)")

zdr = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
frame = pd.DataFrame(
    {
        "Zdr (dB)": zdr,
        "D0 book (mm)": 1.529 * zdr**0.467,
        "D0 radarx (mm)": np.exp(fit_d0[1]) * zdr ** fit_d0[0],
        "Dm book (mm)": 1.619 * zdr**0.485,
        "Dm radarx (mm)": np.exp(fit_dm[1]) * zdr ** fit_dm[0],
    }
).set_index("Zdr (dB)")
frame.round(2)
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
axes[0].plot(zdr_db[ok], d0[ok], ".", ms=1.5, alpha=0.2, label="simulated distributions")
grid = np.linspace(0.1, 4.5, 100)
axes[0].plot(grid, 1.529 * grid**0.467, "r", label="book, Eq. 7.14b")
axes[0].plot(grid, np.exp(fit_d0[1]) * grid ** fit_d0[0], "k--", label="fit to the simulation")
axes[0].set(xlabel="$Z_{dr}$ (dB)", ylabel="$D_0$ (mm)", title="S band")
axes[0].legend(markerscale=6)
sx = simulated["S"]
r_book = power2((sx["zh"][keep], sx["zeta"][keep]), *BOOK["S"]["r_zh_zdr"])
axes[1].loglog(rain_rate[keep], r_book, ".", ms=1.5, alpha=0.2)
axes[1].plot([1e-2, 300], [1e-2, 300], "k")
axes[1].set(
    xlabel="simulated rain rate (mm h$^{-1}$)",
    ylabel="$R(Z_h, Z_{dr})$ of the book (mm h$^{-1}$)",
    title="S band, Table 8.1",
)
for ax in axes:
    ax.grid(alpha=0.3, which="both")
```

## Attenuation and differential attenuation against $K_{dp}$ (Table 7.1)

Bringi and Chandrasekar give temperature-averaged (0 to 30 degrees Celsius) power laws
$A_h = \alpha K_{dp}^b$ and $A_{dp} = \beta K_{dp}^b$ for the attenuation and the
differential attenuation, from the simulations of Jameson (1992) for $\alpha$ and
from gamma distributions for $\beta$. The same fits on the ensemble above, at 0, 10, 20 and
30 degrees Celsius and averaged:

```{code-cell} ipython3
BOOK_71 = {  # GHz: alpha, b, beta, b
    2.8: (0.017, 0.84, 0.003, 1.05),
    5.5: (0.073, 0.99, 0.013, 1.23),
    9.3: (0.233, 1.02, 0.033, 1.15),
    13.8: (0.44, 0.97, 0.054, 1.13),
    19.35: (0.72, 0.99, 0.088, 1.17),
}
rows = []
for frequency, book in BOOK_71.items():
    wavelength = 299.792458 / frequency
    fits_t = []
    for temperature in (0.0, 10.0, 20.0, 30.0):
        t = raindrop_scattering(
            D, wavelength, temperature=temperature, canting_std=7.0, axis_ratio=beard_chuang
        )
        kdp = (dsd * t.kdp.values).sum(axis=1) * DD
        ah = (dsd * t.ah.values).sum(axis=1) * DD
        adp = ah - (dsd * t.av.values).sum(axis=1) * DD
        sel = keep & (kdp > 0.05)
        pa = np.polyfit(np.log(kdp[sel]), np.log(ah[sel]), 1)
        pb = np.polyfit(np.log(kdp[sel]), np.log(adp[sel]), 1)
        fits_t.append((np.exp(pa[1]), pa[0], np.exp(pb[1]), pb[0]))
    mean = np.mean(fits_t, axis=0)
    rows.append(
        {
            "GHz": frequency,
            "alpha (book)": book[0],
            "alpha (radarx)": mean[0],
            "b (book)": book[1],
            "b (radarx)": mean[1],
            "beta (book)": book[2],
            "beta (radarx)": mean[2],
            "b (book) ": book[3],
            "b (radarx) ": mean[3],
        }
    )
pd.DataFrame(rows).set_index("GHz").round(3)
```

The simulated $\alpha$ is 5 to 15 % larger than the book's, which is the difference between
the distributions of Jameson (1992) and the ones used here, and $\beta$ agrees to 15 %.
Both increase with the frequency as the book shows, and the exponent $b$ is close to 1
(0.84 to 1.02) at every frequency.

## Cloud attenuation of small droplets (Doviak and Zrnić, problem 3.8)

For droplets much smaller than the wavelength the attenuation is absorption, and
the specific attenuation depends on the liquid water content $M$ only, not on the
sizes of the droplets (Doviak and Zrnić 1993, problem 3.8b):
$\gamma = 4.343\times10^{3}\,\dfrac{6\pi}{\lambda\,\rho_w}\,\mathrm{Im}(K_w)\,M$ in dB km$^{-1}$
with $K_w = (m^2 - 1)/(m^2 + 2)$ (the book writes $\mathrm{Im}(-K_w)$ because it takes the
imaginary part of $m$ to be negative; radarx uses the positive one). The T-matrix calculation for droplets of 0.02 to 0.2 mm
at 3.2 and 10 cm and 20 degrees Celsius, per g m$^{-3}$ of liquid water:

```{code-cell} ipython3
from radarx.fundamentals import water_refractive_index

diameter = np.array([0.02, 0.05, 0.1, 0.2])
rows = []
for wavelength in (32.0, 100.0):
    m = water_refractive_index(wavelength, 20.0)
    k_w = (m * m - 1) / (m * m + 2)
    analytic = 4.343e3 * 6 * np.pi / (wavelength * 1e-3 * 1000.0) * k_w.imag * 1e-3
    t = raindrop_scattering(diameter, wavelength, temperature=20.0, axis_ratio=1.0, canting_std=0.5)
    # one droplet per m3 attenuates by ah dB/km; a g m-3 holds 1e-3 / (pi/6 D^3 1e-9 kg) droplets
    per_gram = t.ah.values / (np.pi / 6 * diameter**3 * 1e-9 * 1000.0) * 1e-3
    for d, a in zip(diameter, per_gram):
        rows.append(
            {"wavelength (cm)": wavelength / 10, "diameter (mm)": d, "T-matrix (dB/km per g/m3)": a, "Rayleigh formula": analytic}
        )
pd.DataFrame(rows).set_index(["wavelength (cm)", "diameter (mm)"]).round(5)
```

The attenuation per unit of water is the same for all droplet sizes and equal to
the closed form: to 0.1 % for droplets up to 0.05 mm, and to 0.3 % (10 cm) and 2 %
(3.2 cm) for 0.2 mm, where the next term in the size parameter appears.

## References

- Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler Weather Radar:
  Principles and Applications*. Cambridge University Press, 636 pp.,
  <https://doi.org/10.1017/CBO9780511541094>
- Doviak, R. J., and D. S. Zrnić, 1993: *Doppler Radar and Weather Observations*, 2nd ed.
  Academic Press, 562 pp.
- Beard, K. V., and C. Chuang, 1987: A new model for the equilibrium shape of raindrops.
  *J. Atmos. Sci.*, **44**, 1509–1524,
  <https://doi.org/10.1175/1520-0469(1987)044<1509:ANMFTE>2.0.CO;2>
- Jameson, A. R., 1992: The effect of temperature on attenuation correction schemes in
  rain using polarization propagation differential phase shift. *J. Appl. Meteor.*, **31**,
  1106–1118, <https://doi.org/10.1175/1520-0450(1992)031<1106:TEOTOA>2.0.CO;2>
- Ray, P. S., 1972: Broadband complex refractive indices of ice and water. *Appl. Opt.*,
  **11**, 1836–1844, <https://doi.org/10.1364/AO.11.001836>
