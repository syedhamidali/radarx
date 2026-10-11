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

# Attenuation and differential attenuation correction

Rain attenuates the signal of a C- or X-band radar, and at S band in heavy
rain. The measured reflectivity $Z_H$ is too low behind rain, and the
differential reflectivity $Z_{DR}$ too, because the horizontal polarisation is
attenuated more than the vertical. The differential phase $\Phi_{DP}$ is not
attenuated and grows with the rain along the ray, so it measures the path.
`radarx.retrieve.correct_attenuation` (or `.radarx.correct_attenuation()`)
corrects $Z_H$ and $Z_{DR}$ with it, in two ways:

- `method="zphi"` (default): the constrained solution of Testud et al. (2000).
  The specific attenuation is $A_H = aZ^b$, and the coefficient $a$ is fixed by
  requiring that the two-way attenuation of the ray is $\alpha\,\Delta\Phi_{DP}$.
- `method="linear"`: $\mathrm{PIA}(r) = \alpha\,[\Phi_{DP}(r)-\Phi_{DP}(r_0)]$ at
  every gate (Bringi et al. 1990).

The differential attenuation follows with the ratio $\beta/\alpha$ of the
band. The coefficients $\alpha$, $\beta$ (dB per degree) and $b$ depend on the
band and can be set. The module documentation lists every choice radarx makes
where the papers leave it open (gate selection, phase processing, melting
layer, behaviour without $K_{DP}$). The numerical work runs in a compiled,
multithreaded C++ kernel with an equivalent NumPy implementation.

```{code-cell} ipython3
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xradar as xd
from open_radar_data import DATASETS
from scipy.special import gamma as gamma_fn

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.retrieve import attenuation as attmod
from radarx.retrieve import attenuation_coefficients, correct_attenuation, estimate_kdp
from radarx.retrieve.dsd import radar_from_dsd

KAPPA = 2 * 0.23025850929940457  # 0.46 of Testud et al.
```

## Coefficients

`attenuation_coefficients(band)` returns $\alpha$ and $\beta$ (the mean linear
relations of Bringi et al. 1990, Table 1, 15 °C) and the exponent $b$. The
second set, `coefficients="tmatrix"`, is radarx's fit to its own T-matrix
scattering tables (`radarx.retrieve.scattering_table`).

```{code-cell} ipython3
rows = []
for band in "SCX":
    for source in ("bringi1990", "tmatrix"):
        rows.append({"band": band, "set": source, **attenuation_coefficients(band, source)})
pd.DataFrame(rows).set_index(["band", "set"])
```

## A rain ray with known attenuation

The truth is forward modelled. Along a ray, a gamma drop size distribution
changes with range (two cells); `radar_from_dsd` gives $Z_H$, $Z_{DR}$,
$K_{DP}$, $A_H$ and $A_{DP}$ at every gate from the T-matrix tables. The
measured fields are the true ones minus the exact path integrals, and the phase
is the integral of the true $K_{DP}$.

```{code-cell} ipython3
DR = 0.25  # km


def cumulative(a, dr=DR):
    """Two-way integral to the gate centre of a one-way specific value."""
    inc = a * dr
    return 2.0 * (np.cumsum(inc) - inc + 0.5 * inc)


def rain_ray(band, ng=360):
    r = (np.arange(ng) + 0.5) * DR
    cell = np.exp(-0.5 * ((r - 30) / 6) ** 2) + 0.7 * np.exp(-0.5 * ((r - 60) / 8) ** 2)
    dm, nw, mu = 1.1 + 1.1 * cell, 10 ** (3.7 - 0.5 * cell), 3.0
    f = 6 / 4**4 * (4 + mu) ** (mu + 4) / gamma_fn(mu + 4)
    d = np.arange(0.05, 8.0001, 0.05)
    nd = nw[:, None] * f * (d / dm[:, None]) ** mu * np.exp(-(4 + mu) * d / dm[:, None])
    nd = xr.DataArray(nd, dims=("gate", "diameter"), coords={"diameter": d})
    t = radar_from_dsd(nd, band=band, temperature=15.0)
    truth = {k: t[k].values for k in ("DBZH", "ZDR", "KDP", "AH", "ADP")}
    meas = {
        "DBZH": truth["DBZH"] - cumulative(truth["AH"]),
        "ZDR": truth["ZDR"] - cumulative(truth["ADP"]),
        "PHIDP": cumulative(truth["KDP"]),
    }
    ds = xr.Dataset(
        {
            "DBZH": (("azimuth", "range"), meas["DBZH"][None]),
            "ZDR": (("azimuth", "range"), meas["ZDR"][None]),
            "PHIDP_processed": (("azimuth", "range"), meas["PHIDP"][None]),
            "RHOHV": (("azimuth", "range"), np.full((1, ng), 0.99)),
        },
        coords={"azimuth": [0.0], "range": ("range", r * 1000.0)},
    )
    return r, truth, meas, ds


rows = []
rays = {}
for band in "SCX":
    r, truth, meas, ds = rain_ray(band)
    rays[band] = (r, truth, meas, ds)
    row = {"band": band, "true PIA (dB)": cumulative(truth["AH"])[-1], "uncorrected Z error at the end (dB)": meas["DBZH"][-1] - truth["DBZH"][-1]}
    for label, kw in {"bringi1990": {}, "tmatrix": {"coefficients": "tmatrix"}}.items():
        for method in ("zphi", "linear"):
            out = correct_attenuation(ds, band=band, method=method, **kw)
            row[f"{method} {label}"] = float(out.DBZH_corrected[0, -1] - truth["DBZH"][-1])
    rows.append(row)
pd.DataFrame(rows).set_index("band").round(2)
```

Errors at the end of the ray in dB. The error is the error of the path
attenuation, $\alpha\,\Delta\Phi_{DP}$, so it is set by the coefficient: the
Bringi et al. value at C band (0.054) is smaller than the effective value of
this rain (0.082), and the T-matrix value (0.084) is close to it. With the
exact ratio of the ray, which the user would only know from a better
calibration, the ZPHI error comes only from the assumption $A_H=aZ^b$ with a
constant $a$ (it changes along this ray):

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), layout="constrained")
for ax, band in zip(axes, "SCX"):
    r, truth, meas, ds = rays[band]
    pia_true = cumulative(truth["AH"])
    exact = pia_true[-1] / (meas["PHIDP"][-1] - meas["PHIDP"][0])
    ratio = truth["ADP"].sum() / truth["AH"].sum()
    b = attenuation_coefficients(band, "tmatrix")["b"]
    ax.plot(r, truth["DBZH"], "k", lw=2, label="true $Z_H$")
    ax.plot(r, meas["DBZH"], color="0.6", label="measured")
    for method, color in (("zphi", "C0"), ("linear", "C1")):
        out = correct_attenuation(ds, method=method, alpha=exact, beta=ratio * exact, b=b)
        ax.plot(r, out.DBZH_corrected[0], color=color, ls="--", label=f"{method}, exact $\\alpha$")
    out = correct_attenuation(ds, band=band)
    ax.plot(r, out.DBZH_corrected[0], color="C3", lw=0.9, label="zphi, default $\\alpha$")
    ax.set_title(f"{band} band, PIA {pia_true[-1]:.1f} dB")
    ax.set_xlabel("range (km)")
axes[0].set_ylabel("$Z_H$ (dBZ)")
axes[0].legend(frameon=False, loc="lower left")
plt.show()
```

The differential reflectivity is corrected in the same way (the bias of the
measured $Z_{DR}$ at the end of the ray, and what remains after the correction
with the T-matrix set). The ratio $\beta/\alpha$ is a constant of the band, but
it varies with the drop size distribution (0.12 to 0.39 at S, C and X band in
Ryzhkov et al. 2014), and the ratio of this ray is 0.10 to 0.13, so the
differential attenuation is over-corrected at C and X band:

```{code-cell} ipython3
rows = []
for band in "SCX":
    r, truth, meas, ds = rays[band]
    out = correct_attenuation(ds, band=band, coefficients="tmatrix")
    rows.append(
        {
            "band": band,
            "measured minus true ZDR (dB)": meas["ZDR"][-1] - truth["ZDR"][-1],
            "corrected minus true ZDR (dB)": float(out.ZDR_corrected[0, -1]) - truth["ZDR"][-1],
        }
    )
pd.DataFrame(rows).set_index("band").round(3)
```

## Why the phase constraint matters

The unconstrained solution of Hitschfeld and Bordan (1954) corrects the
profile from the measurements alone, with a coefficient $a$ in $A_H=aZ^b$:

$$A_H(r) = \frac{a Z_m^b}{1 - 0.46\,ab\int_{r_0}^{r} Z_m^b\,ds}.$$

For the true $a$ it is exact. The denominator is $e^{-0.23\,b\,\mathrm{PIA}(r)}$,
which is small along a strongly attenuated path, so a coefficient that is only a
little too large makes it cross zero and the correction diverges. The rain below
has $A_H=aZ^b$ exactly, with a path attenuation of 28 dB.

```{code-cell} ipython3
b = 0.84
dr = 0.05
r = (np.arange(400) + 0.5) * dr
z_true = 20 + 30 * np.exp(-0.5 * ((r - 8) / 6) ** 2)
a = 0.3 / 10 ** (0.4 * 10 * b)  # 0.3 dB/km at 40 dBZ
ah_true = a * 10 ** (b * z_true / 10)
pia = cumulative(ah_true, dr)
zm = z_true - pia


def hitschfeld_bordan(zm_dbz, a):
    w = 10 ** (b * zm_dbz / 10)
    denom = 1 - KAPPA * a * b * (np.cumsum(w * dr) - 0.5 * w * dr)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, a * w / denom, np.inf), denom


def zphi(phase_scale):
    ds = xr.Dataset(
        {
            "DBZH": (("azimuth", "range"), zm[None]),
            "PHIDP_processed": (("azimuth", "range"), phase_scale * pia[None] / 0.08),
        },
        coords={"azimuth": [0.0], "range": ("range", r * 1000.0)},
    )
    return correct_attenuation(ds, alpha=0.08, beta=0.02, b=b, dbz_min=-50.0)


fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
axes[0].plot(r, ah_true, "k", lw=2, label="true $A_H$")
for scale, color in ((1.0, "C2"), (1.01, "C3"), (1.05, "C1")):
    hb, den = hitschfeld_bordan(zm, scale * a)
    axes[0].plot(r, np.where(np.isfinite(hb), hb, np.nan), color=color, label=f"Hitschfeld-Bordan, $a\\times${scale}")
axes[0].set_ylim(0, 6)
axes[0].set_xlabel("range (km)")
axes[0].set_ylabel("specific attenuation (dB/km)")
axes[0].legend(frameon=False)
axes[0].set_title("unconstrained: unstable")
for scale, color in ((1.0, "C2"), (1.05, "C1"), (1.2, "C3")):
    out = zphi(scale)
    axes[1].plot(r, out.DBZH_corrected[0] - z_true, color=color, label=f"ZPHI, path attenuation $\\times${scale}")
axes[1].set_xlabel("range (km)")
axes[1].set_ylabel("corrected minus true $Z_H$ (dB)")
axes[1].legend(frameon=False)
axes[1].set_title("constrained: the error is the error of the constraint")
plt.show()
print(f"path attenuation {pia[-1]:.1f} dB")
for scale in (1.01, 1.02, 1.05, 1.10):
    hb, den = hitschfeld_bordan(zm, scale * a)
    print(f"Hitschfeld-Bordan with a x{scale}: denominator reaches {den.min():+.3f} (1.0 at the radar), {int((~np.isfinite(hb)).sum())} of {r.size} gates singular")
for scale in (0.9, 1.05, 1.2):
    out = zphi(scale)
    print(f"ZPHI with the path attenuation x{scale}: error at the end {float(out.DBZH_corrected[0, -1] - z_true[-1]):+.2f} dB")
```

ZPHI keeps the denominator above $I(r_0,r_1)>0$ for any constraint, and its
error is proportional to the error of the path attenuation. The price is that
$\alpha\,\Delta\Phi_{DP}$ must be right.

## A real C-band volume

A C-band (CSAPR2) PPI of deep convection in Argentina during CACTI. The raw
phase is processed with `estimate_kdp`, and the **uncorrected** reflectivity
and differential reflectivity are corrected. The file also holds the fields of
the radar's own correction chain (`attenuation_corrected_reflectivity_h`).

```{code-cell} ipython3
file = DATASETS.fetch("corcsapr2cmacppiM1.c1.20181111.030003.nc")
dtree = xd.io.open_cfradial1_datatree(file).xradar.georeference()
sweep = dtree["sweep_0"].to_dataset()
fields = {
    "phidp": "uncorrected_differential_phase",
    "rhohv": "uncorrected_copol_correlation_coeff",
    "dbzh": "uncorrected_reflectivity_h",
}
kdp = estimate_kdp(sweep, **fields)
sweep = sweep.assign(PHIDP_processed=kdp.PHIDP_processed, KDP=kdp.KDP)
att = {
    "zphi, Bringi et al.": correct_attenuation(
        sweep, dbzh=fields["dbzh"], zdr="uncorrected_differential_reflectivity",
        rhohv=fields["rhohv"], band="C"),
    "zphi, T-matrix": correct_attenuation(
        sweep, dbzh=fields["dbzh"], zdr="uncorrected_differential_reflectivity",
        rhohv=fields["rhohv"], band="C", coefficients="tmatrix"),
    "linear, Bringi et al.": correct_attenuation(
        sweep, dbzh=fields["dbzh"], zdr="uncorrected_differential_reflectivity",
        rhohv=fields["rhohv"], band="C", method="linear"),
}
res = att["zphi, T-matrix"]
print(f"rays with a correction: {int((res.DELTA_PHIDP >= 3).sum())} of {res.sizes['azimuth']}; largest phase span {float(res.DELTA_PHIDP.max()):.0f} deg")
for name, out in att.items():
    end = out.PIA.max("range")
    print(f"{name:22s} largest PIA {float(end.max()):5.1f} dB, largest PIDA {float(out.PIDA.max()):.2f} dB")
res
```

```{code-cell} ipython3
x, y = sweep.x / 1000.0, sweep.y / 1000.0
panels = [
    (sweep[fields["dbzh"]], "uncorrected $Z_H$ (dBZ)", "turbo", -10, 65),
    (res.DBZH_corrected, "corrected $Z_H$ (dBZ)", "turbo", -10, 65),
    (res.PIA, "path attenuation PIA (dB)", "magma_r", 0, 30),
    (res.AH.where(res.AH > 0.01), "specific attenuation $A_H$ (dB/km)", "viridis", 0, 2),
]
fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
for ax, (da, label, cmap, vmin, vmax) in zip(axes.flat, panels):
    pm = ax.pcolormesh(x, y, da, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(pm, ax=ax, shrink=0.8, label=label)
    ax.set_aspect("equal")
    ax.set_xlim(-110, 110)
    ax.set_ylim(-110, 110)
for ax in axes[1]:
    ax.set_xlabel("east (km)")
for ax in axes[:, 0]:
    ax.set_ylabel("north (km)")
fig.tight_layout()
plt.show()
```

Along the ray with the largest phase span:

```{code-cell} ipython3
iray = int(np.nanargmax(res.DELTA_PHIDP.values))
r_km = sweep.range.values / 1000.0
fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
axes[0].plot(r_km, sweep[fields["dbzh"]][iray], color="0.5", label="measured")
for name, color in (("zphi, T-matrix", "C0"), ("zphi, Bringi et al.", "C1")):
    axes[0].plot(r_km, att[name].DBZH_corrected[iray], color=color, label=name)
axes[0].plot(r_km, sweep.attenuation_corrected_reflectivity_h[iray], "k:", label="radar's own chain")
axes[0].set_ylabel("$Z_H$ (dBZ)")
axes[0].legend(frameon=False, ncol=2)
axes[1].plot(r_km, sweep.PHIDP_processed[iray], color="C2")
axes[1].set_ylabel("processed $\\Phi_{DP}$ (°)")
for name, color in (("zphi, T-matrix", "C0"), ("zphi, Bringi et al.", "C1"), ("linear, Bringi et al.", "C4")):
    axes[2].plot(r_km, att[name].PIA[iray], color=color, label=name)
axes[2].plot(r_km, sweep.path_integrated_attenuation[iray], "k:", label="radar's own chain")
axes[2].set_ylabel("PIA (dB)")
axes[2].set_xlabel("range (km)")
axes[2].legend(frameon=False)
axes[0].set_title(f"azimuth {float(sweep.azimuth[iray]):.1f}°, phase span {float(res.DELTA_PHIDP[iray]):.0f}°")
fig.tight_layout()
plt.show()
```

### A check against $K_{DP}$

There is no unattenuated reference for this radar, but $K_{DP}$ is not
attenuated and in rain the reflectivity follows it, $Z\approx f(K_{DP})$. The
relation of C-band rain from the T-matrix tables (normalized gamma
distributions; a scatter of about 1 dB in this range of $Z$) is fitted here.
The bias of $Z_H$ against it should not depend on how much attenuation the beam
has passed through (the PIA of the T-matrix correction; the same gates are used
for all rows).

```{code-cell} ipython3
rnd = np.random.default_rng(11)
n = 6000
mu, dm, nw = rnd.uniform(0, 6, n), rnd.uniform(0.9, 2.8, n), 10 ** rnd.uniform(2.8, 4.4, n)
fm = 6 / 4**4 * (4 + mu) ** (mu + 4) / gamma_fn(mu + 4)
dd = np.arange(0.05, 8.0001, 0.05)
nd = nw[:, None] * fm[:, None] * (dd / dm[:, None]) ** mu[:, None] * np.exp(-(4 + mu[:, None]) * dd / dm[:, None])
t = radar_from_dsd(xr.DataArray(nd, dims=("s", "diameter"), coords={"diameter": dd}), band="C", temperature=20.0)
use = (t.KDP.values > 0.5) & (t.DBZH.values > 30) & (t.DBZH.values < 55)
slope, icpt = np.polyfit(np.log10(t.KDP.values[use]), t.DBZH.values[use], 1)
print(f"C-band rain: Z = {icpt:.1f} + {slope:.1f} log10(KDP) dBZ")

raw = sweep[fields["dbzh"]]
gate = (sweep.KDP > 1.0) & (sweep[fields["rhohv"]] > 0.95) & (raw > 20)
pred = icpt + slope * np.log10(sweep.KDP.where(gate))
edges = [0, 2, 5, 10, 20, 40]
rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = gate & (res.PIA >= lo) & (res.PIA < hi)
    row = {"PIA (dB)": f"{lo}-{hi}", "gates": int(sel.sum()), "uncorrected": float((raw - pred).where(sel).mean())}
    for name, out in att.items():
        row[name] = float((out.DBZH_corrected - pred).where(sel).mean())
    row["radar's own chain"] = float((sweep.attenuation_corrected_reflectivity_h - pred).where(sel).mean())
    rows.append(row)
table = pd.DataFrame(rows).set_index("PIA (dB)")
table.round(1)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7, 4))
for col in table.columns[1:]:
    ax.plot(table.index, table[col], "o-", label=col, color="0.5" if col == "uncorrected" else None)
ax.set_xlabel("path attenuation of the T-matrix correction (dB)")
ax.set_ylabel("mean $Z_H$ minus $Z(K_{DP})$ (dB)")
ax.legend(frameon=False)
plt.show()
```

The uncorrected reflectivity falls by more than 20 dB below the $K_{DP}$
estimate in the most attenuated gates. With the T-matrix coefficients (α = 0.084
dB/deg at C band) the bias no longer depends on the path attenuation. The value
of Bringi et al. (1990), 0.054 dB/deg, corrects roughly two thirds to three
quarters of the bias, as expected from the ratio of the coefficients, and the
correction chain of the radar less than that. This is a consistency check with
a relation that comes from the same scattering tables, not an independent
measurement of the attenuation. The offset of about 3 dB that remains in
the gates with little attenuation is not removed by any correction; it may be
the calibration of this radar or the drop shape of the tables (not
investigated).

## Other inputs

`correct_attenuation` accepts a sweep, a volume (a `DataTree`, one node per
sweep with the fields) or the reflectivity as a `DataArray`, and a dask-backed
reflectivity stays lazy:

```{code-cell} ipython3
kw = dict(dbzh=fields["dbzh"], zdr="uncorrected_differential_reflectivity", rhohv=fields["rhohv"], band="C")
one = sweep.radarx.correct_attenuation(**kw)  # the accessor of a sweep
arr = sweep[fields["dbzh"]].radarx.correct_attenuation(  # and of the reflectivity alone
    phidp=sweep.PHIDP_processed, rhohv=sweep[fields["rhohv"]], band="C"
)
tree = xr.DataTree.from_dict({"/": dtree.root.to_dataset(), "/sweep_0": sweep})
products = tree.radarx.correct_attenuation(**kw)  # a volume: products per sweep
merged = tree.radarx.assign(products)  # next to the measured fields
lazy = sweep.chunk({"azimuth": 90}).radarx.correct_attenuation(**kw)
print("DataArray equals the sweep:", bool(np.allclose(arr.PIA, one.PIA)),
      "| volume equals the sweep:", bool(np.allclose(products["sweep_0"]["PIA"], one.PIA)),
      "| lazy:", lazy.DBZH_corrected.chunks is not None,
      "| lazy equals eager:", bool(np.allclose(lazy.PIA.compute(), one.PIA)))
print(one.attrs["radarx_method"])
```

Without `phidp`, the raw phase is processed by `estimate_kdp` (its defaults, or
`kdp_options`), and `melting` (the melting level in m), `mask` and `rhohv_min`
select the gates of the rain path.

## Speed

The compiled kernel against the NumPy implementation, for one sweep and for a
volume's worth of rays:

```{code-cell} ipython3
rng = np.random.default_rng(0)
rows = []
for nr, ng in ((361, 1100), (3600, 2000)):
    z = 35 + 8 * rng.standard_normal((nr, ng))
    phi = np.cumsum(np.abs(rng.normal(0.2, 0.3, (nr, ng))), axis=1)
    ok = np.ones((nr, ng), dtype=np.uint8)
    row = {"rays x gates": f"{nr} x {ng}"}
    for method, code in (("zphi", 0), ("linear", 1)):
        for name, fn in (
            ("compiled", lambda: attmod._attenuation.correct(z, phi, ok, 0.1, code, 0.08, 0.8, 3.0, np.nan, 0)),
            ("NumPy", lambda: attmod._correct_numpy(z, phi, ok, 0.1, code, 0.08, 0.8, 3.0, np.nan)),
        ):
            if name == "compiled" and not attmod.HAS_COMPILED_KERNEL:
                continue
            fn()
            t0 = time.perf_counter()
            fn()
            row[f"{method} {name} (ms)"] = 1000 * (time.perf_counter() - t0)
    rows.append(row)
pd.DataFrame(rows).set_index("rays x gates").round(1)
```

## References

- Testud, J., E. Le Bouar, E. Obligis, and M. Ali-Mehenni, 2000: The rain
  profiling algorithm applied to polarimetric weather radar. *J. Atmos. Oceanic
  Technol.*, **17** (3), 332–356,
  <https://doi.org/10.1175/1520-0426(2000)017<0332:TRPAAT>2.0.CO;2>
- Bringi, V. N., V. Chandrasekar, N. Balakrishnan, and D. S. Zrnić, 1990: An
  examination of propagation effects in rainfall on radar measurements at
  microwave frequencies. *J. Atmos. Oceanic Technol.*, **7** (6), 829–840,
  <https://doi.org/10.1175/1520-0426(1990)007<0829:AEOPEI>2.0.CO;2>
- Ryzhkov, A., M. Diederich, P. Zhang, and C. Simmer, 2014: Potential
  utilization of specific attenuation for rainfall estimation, mitigation of
  partial beam blockage, and radar networking. *J. Atmos. Oceanic Technol.*,
  **31** (3), 599–619, <https://doi.org/10.1175/JTECH-D-13-00038.1>
- Park, S.-G., V. N. Bringi, V. Chandrasekar, M. Maki, and K. Iwanami, 2005:
  Correction of radar reflectivity and differential reflectivity for rain
  attenuation at X band. Part I: Theoretical and empirical basis. *J. Atmos.
  Oceanic Technol.*, **22** (11), 1621–1632,
  <https://doi.org/10.1175/JTECH1803.1>
- Hitschfeld, W., and J. Bordan, 1954: Errors inherent in the radar measurement
  of rainfall at attenuating wavelengths. *J. Meteor.*, **11** (1), 58–67,
  <https://doi.org/10.1175/1520-0469(1954)011<0058:EIITRM>2.0.CO;2>
