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

# Machine-learning rain rate: the workflow

`rain_rate_ml` gives the rain rate of every gate from a model that reads the
polarimetric variables (and, if the model uses them, the height and the
temperature of the gate) and returns the rain rate with a lower and an upper
bound. The model is an ONNX file that `radarx.ml` runs, listed in the model
registry with its licence, its citation and the `features` it reads. The
radarx network (`ml/models/qpe` in the repository, a multilayer perceptron
trained on simulated drop size distributions with measurement errors, or on
your gauges) is not published yet, so this notebook cannot run it.

This notebook runs the whole workflow with the repository's tools and a
deliberately small model:

1. simulate drop size distributions and their radar variables,
2. fit a model on part of them and test it on the rest,
3. write it to ONNX with the inputs and outputs that `radarx.ml` expects,
4. register it and run `rain_rate_ml` on a real C-band sweep, and compare with
   the power laws of `radarx.retrieve`.

**What is real and what is small.** The feature interface, the ONNX contract,
the registry and the retrieval are the real ones. The model is a power law in
the reflectivity, the differential reflectivity and the logarithm of the
specific differential phase, fitted by least squares in the logarithm of the
rain rate (a generalisation of $R(Z, Z_{DR})$ with a third variable, and a
constant spread), for one band. The radarx network has about 9 000 parameters,
is trained on millions of simulated samples of all three bands, with errors
of calibration and noise added to the inputs, and its spread depends on the
inputs. Do not read the numbers below as the performance of that network.

```{code-cell} ipython3
import hashlib
import tempfile
from pathlib import Path

import cmweather  # noqa: F401  registers the radar colormaps
import matplotlib.pyplot as plt
import numpy as np
import onnx
import xarray as xr
import xradar as xd
from onnx import TensorProto, helper
from open_radar_data import DATASETS
from scipy.special import gamma as gamma_fn

import radarx
from radarx import ml
from radarx.retrieve import dsd_spectrum, radar_from_dsd, rain_rate_kdp, rain_rate_ml, rain_rate_z_zdr
```

## Simulated drop size distributions

A normalized gamma distribution (Testud et al. 2001) has three parameters,
the mass-weighted mean diameter $D_m$, the normalized intercept $N_w$ and
the shape $\mu$. We draw them from wide, partly overlapping ranges of the
stratiform, maritime and continental regimes (the widths are choices of this
notebook, not fitted values), and compute $Z_H$, $Z_{DR}$, $K_{DP}$,
$\rho_{hv}$ and the rain rate with the T-matrix scattering tables of radarx
(`radar_from_dsd`, C band, 15 °C). The inputs get the errors of a measurement:
0.3 dB noise on $Z_H$, 0.1 dB on $Z_{DR}$ and 0.1 degree/km plus 10 % of its
value on $K_{DP}$; the target stays the true rain rate.

```{code-cell} ipython3
rng = np.random.default_rng(3)
n = 40000
dm = np.clip(rng.normal(1.6, 0.5, n), 0.6, 3.5)
log_nw = np.clip(3.9 - 0.8 * (dm - 1.6) + rng.normal(0, 0.5, n), 2.5, 5.2)
mu = np.clip(rng.gamma(2.5, 1.4, n) - 1.0, -1.0, 10.0)
d = np.linspace(0.1, 7.9, 120)
f_mu = 6.0 / 4.0**4 * (4.0 + mu) ** (mu + 4.0) / gamma_fn(mu + 4.0)
nd = (10.0**log_nw * f_mu)[:, None] * (d[None] / dm[:, None]) ** mu[:, None] * np.exp(
    -(4.0 + mu[:, None]) * d[None] / dm[:, None]
)
spectra = xr.DataArray(nd, dims=("sample", "diameter"), coords={"diameter": d})
sim = radar_from_dsd(spectra, band="C", temperature=15.0)
keep = (sim.RAIN_RATE > 0.2) & (sim.RAIN_RATE < 300)
sim = sim.where(keep, drop=True)
m = sim.sizes["sample"]
dbz = sim.DBZH.values + rng.normal(0, 0.3, m)
zdr = sim.ZDR.values + rng.normal(0, 0.1, m)
kdp = sim.KDP.values + rng.normal(0, 0.1, m) + 0.1 * sim.KDP.values * rng.normal(size=m)
rain = sim.RAIN_RATE.values
print(f"{m} samples, rain rate {np.percentile(rain, 5):.1f}-{np.percentile(rain, 95):.1f} mm/h (5-95 %)")
```

## A small model, fitted and tested

The model is $\ln R = w_0 + w_1 Z_H + w_2 Z_{DR} + w_3 \ln(K_{DP} + 0.1)$ with
$Z_H$ in dBZ, $Z_{DR}$ in dB and $K_{DP}$ in degree/km. It is fitted by least
squares on 70 % of the samples. The rain rate that the model returns is the mean of the
log-normal distribution with the standard deviation $\sigma$ of the residuals,
$\exp(\mu + \sigma^2/2)$, which keeps the total rain unbiased, and the bounds are
the 16th and 84th percentile $\exp(\mu \mp \sigma)$. Tested on the other 30 %, against the
C-band relations of `radarx.retrieve`:

```{code-cell} ipython3
def design(dbz, zdr, kdp):
    return np.stack([np.ones_like(dbz), dbz, zdr, np.log(np.maximum(kdp, 0.0) + 0.1)], axis=1)


order = rng.permutation(m)
train, test = order[: int(0.7 * m)], order[int(0.7 * m) :]
w, *_ = np.linalg.lstsq(design(dbz, zdr, kdp)[train], np.log(rain[train]), rcond=None)
sigma = float(np.std(np.log(rain[train]) - design(dbz, zdr, kdp)[train] @ w))
print("weights", np.round(w, 4), f"sigma {sigma:.3f}")

fields = xr.Dataset({"DBZH": ("g", dbz[test]), "ZDR": ("g", zdr[test]), "KDP": ("g", kdp[test])})
mu_test = design(dbz[test], zdr[test], kdp[test]) @ w
estimates = {
    "network-like model": np.exp(mu_test + 0.5 * sigma**2),
    "R(Z, ZDR)": rain_rate_z_zdr(fields, band="C", z_cap=None).values,
    "R(KDP)": rain_rate_kdp(fields, band="C", negative="zero").values,
}
truth = rain[test]
print(f"{'':20s}{'bias (%)':>10s}{'scatter (dB)':>14s}")
for name, est in estimates.items():
    ok = truth >= 1.0
    db = 10 * np.log10(np.maximum(est[ok], 1e-3) / truth[ok])
    print(f"{name:20s}{100 * (est.sum() / truth.sum() - 1):10.1f}{0.5 * (np.percentile(db, 84) - np.percentile(db, 16)):14.2f}")
inside = np.mean((truth >= np.exp(mu_test - sigma)) & (truth <= np.exp(mu_test + sigma)))
print(f"share of the true rain rates between the bounds: {inside:.2f} (0.68 for a log-normal spread)")
```

The scatter is the half distance between the 16th and 84th percentile of
$10\log_{10}(\text{estimate}/\text{truth})$ (Germann et al. 2006). A model
that uses $K_{DP}$ beside $Z_H$ and $Z_{DR}$ weights the three by what they
carry, which a power law of one variable cannot. The C-band relations of
`radarx.retrieve` are the simulations of Bringi and Chandrasekar (2001), not a fit to
these distributions, so the comparison shows what is gained by fitting to the
population of drop size distributions that one expects.

## The ONNX file and the registry

A model for `rain_rate_ml` takes the float32 matrix `features` of shape
(N, F), the columns named in the registry entry, and returns `rain_rate`,
`rain_rate_lower` and `rain_rate_upper`, each of shape (N). Everything that
turns the physical values into the model input belongs in the graph. Here the
graph splits the three columns, takes the logarithm of $K_{DP} + 0.1$, applies
the weights, and exponentiates.

```{code-cell} ipython3
def build_onnx(path, w, sigma):
    f32 = TensorProto.FLOAT
    inits = [
        helper.make_tensor("w", f32, [3, 1], w[1:].astype(np.float32).tolist()),
        helper.make_tensor("b", f32, [1], [float(w[0] + 0.5 * sigma**2)]),
        helper.make_tensor("kdp_floor", f32, [], [0.1]),
        helper.make_tensor("kdp_min", f32, [], [0.0]),
        helper.make_tensor("lo", f32, [], [float(np.exp(-sigma - 0.5 * sigma**2))]),
        helper.make_tensor("hi", f32, [], [float(np.exp(sigma - 0.5 * sigma**2))]),
        helper.make_tensor("axes", TensorProto.INT64, [1], [1]),
    ]
    nodes = [
        helper.make_node("Split", ["features"], ["z", "zd", "k"], axis=1),
        helper.make_node("Max", ["k", "kdp_min"], ["k0"]),
        helper.make_node("Add", ["k0", "kdp_floor"], ["k1"]),
        helper.make_node("Log", ["k1"], ["lk"]),
        helper.make_node("Concat", ["z", "zd", "lk"], ["x"], axis=1),
        helper.make_node("Gemm", ["x", "w", "b"], ["ln_r"]),
        helper.make_node("Exp", ["ln_r"], ["col"]),
        helper.make_node("Squeeze", ["col", "axes"], ["rain_rate"]),
        helper.make_node("Mul", ["rain_rate", "lo"], ["rain_rate_lower"]),
        helper.make_node("Mul", ["rain_rate", "hi"], ["rain_rate_upper"]),
    ]
    graph = helper.make_graph(
        nodes,
        "log_linear_rain_rate",
        [helper.make_tensor_value_info("features", f32, ["N", 3])],
        [helper.make_tensor_value_info(n, f32, ["N"]) for n in ("rain_rate", "rain_rate_lower", "rain_rate_upper")],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


folder = Path(tempfile.mkdtemp())
sha = build_onnx(folder / "c-band-log-linear.onnx", w, sigma)
ml.register_model(
    "c-band-log-linear",
    folder / "c-band-log-linear.onnx",
    sha,
    "MIT",
    "radarx notebook (2026): log-linear rain rate fitted to simulated C-band drop size distributions",
    task="rainfall-estimation",
    features=["dbz", "zdr", "kdp"],
    description="R = exp(w0 + w1 ZH + w2 ZDR + w3 ln(KDP + 0.1)), C band, rain only.",
    overwrite=True,
)
model = ml.load_model("c-band-log-linear")
print(model)
```

The registry entry lists the licence of the weights and the paper to cite.
`rain_rate_ml` writes them to the attributes of its output. For a model that
is used by many people, put the entry in `radarx/ml/models.toml` or in a file
named in `RADARX_MODEL_REGISTRY`; the weights are downloaded on first use,
checked against the hash and never shipped in the radarx package.

## A real C-band sweep

The C-band (CSAPR2) PPI of deep convection used in the other notebooks, from
the open-radar-data collection (ARM data, CC BY 4.0), with the corrected
reflectivity, the corrected differential reflectivity and the specific
differential phase of the radar processor. The rain gates are those that the
processor classifies as rain.

```{code-cell} ipython3
file = DATASETS.fetch("corcsapr2cmacppiM1.c1.20181111.030003.nc")
dtree = xd.io.open_cfradial1_datatree(file).xradar.georeference()
sweep = dtree["sweep_0"].to_dataset()
sweep = sweep.assign(
    DBZH=sweep.corrected_reflectivity,
    ZDR=sweep.corrected_differential_reflectivity,
    KDP=sweep.corrected_specific_diff_phase,
)
rain_gates = sweep.gate_id == 1
ml_rain = rain_rate_ml(sweep, model="c-band-log-linear", band="C", mask=rain_gates)
print(list(ml_rain.data_vars))
print({k: v for k, v in ml_rain.RAIN_RATE_ML.attrs.items() if k.startswith("ml_")})
print(f"{int(ml_rain.RAIN_RATE_ML.notnull().sum())} gates estimated of {int(rain_gates.sum())} rain gates")
```

Gates that have all the inputs and the mask get a rain rate; the network
is not run on the other 70 % of the gates. The result is a Dataset on the
coordinates of the sweep, so it goes into the volume with `.radarx.assign`
like the other products, and the same call is the method `sweep.radarx.rain_rate_ml(...)`.

```{code-cell} ipython3
same = sweep.radarx.rain_rate_ml(model=model, band="C", mask=rain_gates)
print("the method gives the same:", bool(np.allclose(same.RAIN_RATE_ML, ml_rain.RAIN_RATE_ML, equal_nan=True)))
z_zdr = rain_rate_z_zdr(sweep, band="C", z_cap=None)
kdp_rate = rain_rate_kdp(sweep, band="C", negative="zero")
r_ml = ml_rain.RAIN_RATE_ML
estimates_real = {
    "R(KDP)": kdp_rate,
    "rain_rate_A of the processor": sweep.rain_rate_A,
    "ML (this notebook)": r_ml,
}
for name, est in estimates_real.items():
    both = est.notnull() & z_zdr.notnull() & rain_gates & (z_zdr > 1)
    ratio = (est / z_zdr).where(both)
    print(f"{name:30s} median ratio to R(Z, ZDR) {float(ratio.median()):.2f}, 16-84 %: "
          f"{float(ratio.quantile(0.16)):.2f}-{float(ratio.quantile(0.84)):.2f} on {int(both.sum())} gates")
```

There is no truth in a radar sweep. The three estimators that exist already
differ from each other by 20 to 40 % on the median, because the corrected
$Z_{DR}$ of this sweep is noisy (5th to 95th percentile of -3.6 to 2.8 dB in rain)
and the relations weigh the variables differently. The model of this notebook,
trained on simulated distributions only, lies within that spread; this says
nothing about its error. Only gauges (or a disdrometer) can.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained", sharex=True, sharey=True)
for ax, (da, title) in zip(
    axes,
    [
        (r_ml, "ML rain rate (mm/h)"),
        (ml_rain.RAIN_RATE_ML_UPPER / ml_rain.RAIN_RATE_ML_LOWER, "upper over lower bound"),
        (z_zdr.where(rain_gates), "R(Z, ZDR), same gates (mm/h)"),
    ],
):
    ratio = "bound" in title
    pm = ax.pcolormesh(
        sweep.x / 1000, sweep.y / 1000, da,
        cmap="viridis" if ratio else "turbo", vmin=1 if ratio else 0, vmax=8 if ratio else 80, shading="auto",
    )
    ax.set(title=title, aspect="equal", xlim=(-60, 60), ylim=(-60, 60), xlabel="x (km)")
    fig.colorbar(pm, ax=ax, shrink=0.8)
axes[0].set_ylabel("y (km)")
plt.show()
```

The model of this notebook has a constant spread, so the ratio of the bounds
is the same on every gate; the radarx network predicts a spread that depends on
the inputs (larger where only $Z_H$ is available than where $K_{DP}$ and $Z_{DR}$ agree).

## Before and after the estimate

* The model knows what it was trained on. The registry entry says which band,
  rain regimes and variables; a model of simulated rain says nothing about hail,
  the melting layer or snow, so mask the gates (`mask=`, for example the rain
  classes of `radarx.retrieve.hid`) and limit the reflectivity (`min_dbz`).
* Correct $Z_H$ and $Z_{DR}$ for attenuation and calibration first (see the
  [rainfall estimation notebook](Rainfall_Estimation)) and take $K_{DP}$ from a
  processed phase (`radarx.retrieve.estimate_kdp`).
* A model trained on gauges, in the way of the RainForest random forest of
  [Wolfensberger et al. (2021)](https://doi.org/10.5194/amt-14-3169-2021), learns the
  effects of the vertical profile of reflectivity and of the range for its radar and
  region, with the height of the gate and the temperature as inputs (`height`,
  `temperature`); it is registered with the same `features` mechanism.
* Adjust the result to gauges with `radarx.retrieve.gauge_adjust` (see the
  [gauge adjustment notebook](Gauge_Adjustment)); the two steps are independent.

## References

* Testud, J., S. Oury, R. A. Black, P. Amayenc, and X. Dou, 2001: The concept of
  "normalized" distribution to describe raindrop spectra. *J. Appl. Meteor.*, **40**,
  1118-1140, <https://doi.org/10.1175/1520-0450(2001)040<1118:TCONDT>2.0.CO;2>
* Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler Weather Radar*.
  Cambridge University Press, <https://doi.org/10.1017/CBO9780511541094>
* Germann, U., G. Galli, M. Boscacci, and M. Bolliger, 2006: Radar precipitation
  measurement in a mountainous region. *Quart. J. Roy. Meteor. Soc.*, **132**, 1669-1692,
  <https://doi.org/10.1256/qj.05.190>
* Wolfensberger, D., M. Gabella, M. Boscacci, U. Germann, and A. Berne, 2021:
  RainForest: a random forest algorithm for quantitative precipitation estimation
  over Switzerland. *Atmos. Meas. Tech.*, **14**, 3169-3193,
  <https://doi.org/10.5194/amt-14-3169-2021>
* Vulpiani, G., S. Giangrande, and F. S. Marzano, 2009: Rainfall estimation from
  polarimetric S-band radar measurements: Validation of a neural network approach.
  *J. Appl. Meteor. Climatol.*, **48**, 2022-2036,
  <https://doi.org/10.1175/2009JAMC2172.1>
