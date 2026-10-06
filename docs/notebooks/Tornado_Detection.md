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

# Tornado Detection and Biological Echo with Pretrained Networks

radarx runs two published convolutional neural networks with
[ONNX Runtime](https://onnxruntime.ai) (`pip install radarx[ml]`), next to the
physical diagnostics they can be checked against:

- **TorNet** (Veillette et al. 2025): the CNN baseline of the TorNet tornado
  detection benchmark. `radarx.retrieve.tornado_probability` prepares its
  inputs from a WSR-88D volume (the 0.5° and 0.9° tilts of reflectivity,
  dealiased velocity, KDP, $\rho_{hv}$, $Z_{DR}$, spectrum width and
  range-folded gates on 0.5° x 250 m gates), runs the network on chips of
  60° x 60 km that tile the sweep, and returns a tornado probability field.
  `radarx.retrieve.rotation_couplets` finds the physical counterpart: compact
  maxima of the LLSD azimuthal shear above 0.006 s⁻¹, the candidate threshold
  of the NWS Tornado Probability Algorithm (Sandmæl et al. 2023).
- **MistNet** (Lin et al. 2019): segmentation of biological scatterers (birds,
  bats, insects) and precipitation from the reflectivity, velocity and
  spectrum width of the 0.5°-4.5° sweeps, the network used by vol2bird.
  `radarx.retrieve.biological_echo` is compared with the polarimetric
  fuzzy-logic `radarx.retrieve.echo_mask`.

Neither project publishes ONNX files. On first use radarx downloads the
original weights from their upstream location (Hugging Face for TorNet,
GitHub for MistNet; both MIT licensed), checks their SHA-256, writes the
network as an ONNX graph with NumPy and `onnx` (`h5py` for the Keras file)
and caches it. No deep-learning framework is needed. The conversion matches
the Keras and PyTorch originals to within 5e-7 (logits) and 1e-6
(probabilities).

```{code-cell} ipython3
import gzip
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xradar as xd
from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.io.aws_data import download_file
from radarx.retrieve import (
    azimuthal_shear,
    biological_echo,
    dealias_velocity,
    echo_mask,
    rotation_couplets,
    tornado_probability,
    tornet_inputs,
)


def nexrad_volume(key):
    """A NEXRAD Level II volume from AWS with the Nyquist velocity of every sweep."""
    path = Path(download_file("unidata-nexrad-level2", key, "downloads"))
    if path.suffix == ".gz":
        unzipped = path.with_suffix("")
        with gzip.open(path) as src, open(unzipped, "wb") as dst:
            shutil.copyfileobj(src, dst)
        path = unzipped
    with NEXRADLevel2File(str(path)) as nf:
        nyquist = [h["msg_31_data_header"]["RAD"]["nyquist_vel"] / 100.0 for h in nf.msg_31_data_header]
    dtree = xd.io.open_nexradlevel2_datatree(str(path))
    for name, value in zip([n for n in dtree.children if n.startswith("sweep")], nyquist):
        dtree[name] = dtree[name].to_dataset().assign_coords(nyquist_velocity=value)
    return dtree


def polar_xy(ds):
    """Plot coordinates (km) of a field on (azimuth, range)."""
    az = np.radians(ds.azimuth.values)[:, None]
    r = ds.range.values[None, :] / 1e3
    return r * np.sin(az), r * np.cos(az)
```

## The Moore, Oklahoma, tornado (20 May 2013)

The EF5 Moore tornado passed about 20 km west of the KTLX radar. The volume
is not part of TorNet, which starts in September 2013. `tornet_inputs` shows
what the network sees: each tilt takes the polarimetric variables from the
surveillance cut and velocity and spectrum width from the Doppler cut of the
NEXRAD split cut, dealiased with `dealias_velocity`; KDP is estimated with
`estimate_kdp`. Flag codes become NaN and range-folded gates are flagged.

```{code-cell} ipython3
ktlx = nexrad_volume("2013/05/20/KTLX/KTLX20130520_201643_V06.gz")
inputs = tornet_inputs(ktlx, max_range=100e3)
inputs
```

```{code-cell} ipython3
start = time.perf_counter()
tor = tornado_probability(inputs)
print(f"{tor.sizes['chip']} chips in {time.perf_counter() - start:.1f} s")
p = tor.tornado_probability
i, j = np.unravel_index(int(np.nanargmax(p.values)), p.shape)
print(f"highest probability {float(p.max()):.2f} at {float(p.azimuth[i]):.1f}°, {float(p.range[j]) / 1e3:.1f} km")
```

For the physical comparison, the 0.5° velocity is dealiased and the LLSD
azimuthal shear maxima above 0.006 s⁻¹ (with at least 20 dBZ) are listed with
the velocity difference across each couplet.

```{code-cell} ipython3
sweep = ktlx["sweep_1"].to_dataset(inherit="all_coords")
sweep["VRADH"] = dealias_velocity(sweep.assign(VRADH=sweep.VRADH.where(sweep.VRADH > -63.9)))
sweep = sweep.xradar.georeference()
couplets = rotation_couplets(sweep, min_reflectivity=20.0)
couplets.isel(couplet=slice(0, 5)).to_dataframe()[["azimuth", "range", "azimuthal_shear", "delta_v", "area"]]
```

```{code-cell} ipython3
def show(inputs, sweep, tor, couplets, box, title):
    x, y = polar_xy(inputs)
    shear = azimuthal_shear(sweep)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), layout="constrained")
    panels = [
        ("Reflectivity 0.5° (dBZ)", x, y, inputs.DBZ[..., 0], "turbo", (-10, 70)),
        ("Dealiased velocity 0.5° (m s⁻¹)", x, y, inputs.VEL[..., 0], "RdBu_r", (-40, 40)),
        ("LLSD azimuthal shear (s⁻¹), couplets", sweep.x / 1e3, sweep.y / 1e3, shear, "RdBu_r", (-0.02, 0.02)),
        ("TorNet tornado probability", x, y, tor.tornado_probability, "magma", (0, 1)),
    ]
    for ax, (name, px, py, da, cmap, (lo, hi)) in zip(axes, panels):
        pm = ax.pcolormesh(px, py, da, cmap=cmap, vmin=lo, vmax=hi)
        fig.colorbar(pm, ax=ax)
        ax.set(xlim=box[:2], ylim=box[2:], aspect="equal", title=name, xlabel="East (km)", ylabel="North (km)")
    axes[2].plot(couplets.x / 1e3, couplets.y / 1e3, "ko", mfc="none", ms=12)
    fig.suptitle(title)


show(inputs, sweep, tor, couplets, (-60, 20, -40, 40), "KTLX 2013-05-20 20:16 UTC")
```

The network gives a probability near 1 on the hook echo and the debris ball,
and the strongest shear couplet (about 0.036 s⁻¹ with a velocity difference
near 100 m s⁻¹) is at the same place. The probability map is coarse: the
network's last layer has one value per 16 x 16 gates (8° by 4 km), so it
localises the tornado to a few kilometres, not to the couplet.

## The KGWX squall line (30 March 2022)

A quasi-linear convective system crossing Mississippi. Shear maxima above the
threshold line the leading edge; the network singles out a few segments.

```{code-cell} ipython3
kgwx = nexrad_volume("2022/03/30/KGWX/KGWX20220330_234639_V06")
inputs = tornet_inputs(kgwx, max_range=150e3)
tor = tornado_probability(inputs)
sweep = kgwx["sweep_1"].to_dataset(inherit="all_coords")
sweep["VRADH"] = dealias_velocity(sweep.assign(VRADH=sweep.VRADH.where(sweep.VRADH > -63.9)))
sweep = sweep.xradar.georeference()
couplets = rotation_couplets(sweep, min_reflectivity=20.0)
show(inputs, sweep, tor, couplets, (-150, 150, -150, 150), "KGWX 2022-03-30 23:46 UTC")
```

## Accuracy on the TorNet test set

The converted network was evaluated on the TorNet test files of 2013
(Zenodo, CC-BY-4.0), read with TorNet's own loader, against the published
baseline (Veillette et al. 2025, Table 2, mean of five trained CNNs; the
published weights are one of them). The radarx numbers are in the pull
request that added this notebook; the 2013 subset is small (60 tornadic of 573
samples), so it reproduces the published skill only approximately.

## Biological echo: MistNet against echo_mask

On a spring night, migrating birds fill the clear-air volume with weak echo
(10-25 dBZ, $\rho_{hv}$ around 0.65) and the typical "butterfly" pattern of
reflectivity. The squall-line volume above is mostly precipitation.

```{code-cell} ipython3
def compare(dtree, title):
    start = time.perf_counter()
    bio = biological_echo(dtree)
    elapsed = time.perf_counter() - start
    qc = echo_mask(dtree)
    name = list(bio.children)[0]
    b, q = bio[name].to_dataset(), qc[name].to_dataset()
    sw = dtree[name].to_dataset(inherit="all_coords").xradar.georeference()
    dbz = sw.DBZH.where(sw.DBZH > -32)
    echo = dbz.notnull() & b.weather_probability.notnull() & (q.ECHO_CLASS > 0)
    mist_weather = ~b.biological_echo & echo
    qc_weather = q.METEO_MASK & echo
    n = int(echo.sum())
    print(f"MistNet in {elapsed:.1f} s; weather: MistNet {float(mist_weather.sum()) / n:.1%}, "
          f"echo_mask {float(qc_weather.sum()) / n:.1%}, agreement {float((mist_weather == qc_weather).where(echo).mean()):.1%}")
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), layout="constrained")
    x, y = sw.x / 1e3, sw.y / 1e3
    panels = [
        ("Reflectivity 0.5° (dBZ)", dbz, "turbo", (-10, 60)),
        ("MistNet biology probability", b.biology_probability.where(echo), "viridis", (0, 1)),
        ("MistNet: weather (blue) / biology (red)", mist_weather.where(echo), "coolwarm_r", (0, 1)),
        ("echo_mask: meteorological (blue) / other (red)", qc_weather.where(echo), "coolwarm_r", (0, 1)),
    ]
    for ax, (label, da, cmap, (lo, hi)) in zip(axes, panels):
        pm = ax.pcolormesh(x, y, da.astype(float), cmap=cmap, vmin=lo, vmax=hi)
        if ax is axes[0] or ax is axes[1]:
            fig.colorbar(pm, ax=ax)
        ax.set(xlim=(-150, 150), ylim=(-150, 150), aspect="equal", title=label, xlabel="East (km)", ylabel="North (km)")
    fig.suptitle(title)


night = xd.io.open_nexradlevel2_datatree(
    download_file("unidata-nexrad-level2", "2022/04/20/KGWX/KGWX20220420_040533_V06", "downloads")
)
compare(night, "KGWX 2022-04-20 04:05 UTC, bird migration")
```

MistNet labels the whole migration as biology. `echo_mask`, built for
precipitation filtering, keeps about a third of it as meteorological (far
from the radar, where its texture features are smooth), although
$\rho_{hv}$ is well below 0.9 almost everywhere.

```{code-cell} ipython3
compare(kgwx, "KGWX 2022-03-30 23:46 UTC, squall line")
```

In the squall line both methods agree on about 97 % of the echo; both find
the weak biological echo at the eastern edge.

## References

- Veillette, M. S., J. M. Kurdzo, P. M. Stepanian, J. Y. N. Cho, T. Reis,
  S. Samsi, J. McDonald, and N. Chisler, 2025: A benchmark dataset for
  tornado detection and prediction using full-resolution polarimetric weather
  radar data. *Artif. Intell. Earth Syst.*, **4** (1),
  <https://doi.org/10.1175/AIES-D-24-0006.1>
- Sandmæl, T. N., B. R. Smith, A. E. Reinhart, I. M. Schick, M. C. Ake,
  J. G. Madden, R. B. Steeves, S. S. Williams, K. L. Elmore, and T. C. Meyer,
  2023: The Tornado Probability Algorithm: A probabilistic machine learning
  tornadic circulation detection algorithm. *Wea. Forecasting*, **38** (3),
  445–466, <https://doi.org/10.1175/WAF-D-22-0123.1>
- Lin, T.-Y., K. Winner, G. Bernstein, A. Mittal, A. M. Dokter, K. G. Horton,
  C. Nilsson, B. M. Van Doren, A. Farnsworth, F. A. La Sorte, S. Maji, and
  D. Sheldon, 2019: MistNet: Measuring historical bird migration in the US
  using archived weather radar data and convolutional neural networks.
  *Methods Ecol. Evol.*, **10** (11), 1908–1922,
  <https://doi.org/10.1111/2041-210X.13280>
- Mahalik, M. C., B. R. Smith, K. L. Elmore, D. M. Kingfield, K. L. Ortega,
  and T. M. Smith, 2019: Estimates of gradients in radar moments using a
  linear least squares derivative technique. *Wea. Forecasting*, **34**,
  415–434, <https://doi.org/10.1175/WAF-D-18-0095.1>
