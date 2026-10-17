# Machine-learning rain rate

This folder holds the training, export and evaluation code of the networks
used by `radarx.retrieve.rain_rate_ml`. It is not part of the radarx package
and is never imported by it. The trained weights are **not distributed yet**;
`rain_rate_ml` raises a clear error until a model is registered with
`radarx.ml.register_model(...)` or passed as `model=...`.

## What the network does

`rain_rate_ml` passes the polarimetric variables of a gate (and, for models
that use them, the height and the temperature) to an ONNX model and returns
its rain rate and an uncertainty. The model is described in the registry
entry by its `features`, the names of the input columns in order:

| feature | meaning |
|---|---|
| `dbz` | horizontal reflectivity (dBZ) |
| `zdr` | differential reflectivity (dB) |
| `kdp` | specific differential phase (degree km-1) |
| `rhohv` | correlation coefficient |
| `height` | height of the gate above the radar (km) |
| `temperature` | air temperature (degrees Celsius) |
| `wavelength` | radar wavelength (cm) |

Outputs: `rain_rate` (mm/h, the mean of the predicted distribution),
`rain_rate_lower` and `rain_rate_upper` (16th and 84th percentile).

## Model trained on simulated drop size distributions (`radarx-qpe-v1`)

`simulate.py` draws normalized-gamma drop size distributions
(Testud et al. 2001) from a mixture of rain regimes and computes the radar
variables with the single-drop T-matrix tables of radarx
(`radarx.retrieve.dsd`, S, C and X band, 5 to 25 degrees Celsius) and the
rain rate with the fall speed of Atlas et al. (1973), as
`radarx.retrieve.radar_from_dsd` does. The prior widths are radarx choices that
cover the ranges of Dm and log10(Nw) that Bringi et al. (2003) report for
different climatic regimes; they are not fitted to data. The measurement
errors of a real radar are added to the inputs (Z_H and Z_DR calibration bias
and noise, noise on K_DP and rho_hv, `simulate.add_noise`) while the target
stays the true rain rate, so the network learns the rain rate expected for a
noisy measurement and its spread. A new sample set is drawn for every epoch.

The network (`model.py`) is a multilayer perceptron (3 x 64, SiLU) that
predicts the mean and the standard deviation of the logarithm of the rain
rate and is trained with the Gaussian negative log-likelihood of the
logarithm (Nix and Weigend 1994). The feature transformation (inverse
hyperbolic sine of K_DP, -log10(1 - rho_hv), logarithm of the wavelength) and
the standardization are part of the exported graph.

What it is: a multivariate rain-rate relation, valid for rain (use the rain
classes of `radarx.retrieve.hid` as `mask`), that weights Z_H, Z_DR and K_DP by
their noise and gives an uncertainty. What it is not: it knows nothing about
attenuation, calibration errors beyond those simulated, partial beam blockage,
the vertical profile of reflectivity, hail or the melting layer, or the
regime of your region beyond the mixture of the simulation.

### Results on held-out simulated samples

Run `radarx-qpe-v1` (40 epochs of 1 million samples, 3 x 64 units, 9 000
parameters), `evaluate.py` with 200 000 new samples per band and set. Bias of
the total rain, normalized mean absolute error and scatter (dB), network
against the power laws of `radarx.retrieve` and a hybrid (R(KDP) where KDP is at
least 0.3 degree/km at S band, scaled with the wavelength, R(Z, ZDR) elsewhere):

| set, band | network | hybrid | R(Z, ZDR) | R(Z) |
|---|---|---|---|---|
| mixture, S | 0.1 %, 0.21, 1.32 dB | -11.6 %, 0.34, 1.89 dB | -22.1 %, 0.32, 1.67 dB | -10.8 %, 0.37, 2.15 dB |
| mixture, C | -0.1 %, 0.18, 1.19 dB | 3.9 %, 0.28, 1.73 dB | 4.7 %, 0.28, 1.66 dB | none |
| mixture, X | -0.1 %, 0.17, 1.11 dB | -12.6 %, 0.28, 1.68 dB | 5.6 %, 0.36, 1.85 dB | 14.7 %, 0.51, 2.51 dB |

The share of true rain rates between the predicted bounds is 0.69 (0.68 for a
calibrated log-normal spread). One regime at a time (the network has seen them only in the
mixture) the bias of the total is between -12.5 % (tropical) and +10.3 %
(continental) at S band and the scatter between 1.2 and 1.3 dB. The simulated
samples share the T-matrix tables and the drop size distribution family with the
training, so this is a test of the learning, not of the physics or of real
rain.

### Real data

`real_data.py` runs the network on KGWX volumes of the 30 March 2022 squall line
(22:52 to 01:05 UTC, lowest sweep with differential phase, rain gates with
rho_hv of at least 0.95 and Z_H of at least 10 dBZ) and compares hourly
accumulations with ASOS gauges (two hours, 9 stations within 148 km whose
precipitation sensor reports; 17 pairs). Bias of the total / normalized mean
absolute error / correlation: R(Z) 16 % / 0.99 / 0.38, hybrid 48 % / 0.85 / 0.58,
network 21 % / 0.90 / 0.47. With 17 pairs, no attenuation or
vertical-profile correction and a 5-minute volume interval for a line moving at
about 20 m/s, the differences between the three estimators are within the
noise; the test shows that the chain works on real volumes and gives
estimates of the right size, not that the network is better than the power laws.

## Models trained on gauges

`train.py --table pairs.npz --features dbz zdr kdp rhohv height temperature`
trains the same network on a table of features and gauge rain rates, in the
way of the RainForest random forest (Wolfensberger et al. 2021) that learns the
effects of the vertical profile of reflectivity, the range, the visibility and
the regime from gauges. The table is not part of radarx: it needs the gauge
network and radar volumes of your country.

## Run

```bash
pip install -r requirements.txt        # torch, onnx, onnxruntime, matplotlib
pip install -e ../../..                # radarx with its compiled kernels
python train.py --out runs/sim-v1 --epochs 60
python export.py runs/sim-v1 runs/sim-v1/radarx-qpe-v1.onnx
python evaluate.py runs/sim-v1/radarx-qpe-v1.onnx --out runs/sim-v1 --figure
```

On the IITM Arka cluster: `qsub pbs/train_qpe.pbs` (queue `cccr`, 96 cores,
conda environment `radarx_qpe`, see the script for the variables).
`export.py` prints the SHA-256 and the `features` line for the registry:

```python
from radarx import ml
ml.register_model(
    "radarx-qpe-v1", "runs/sim-v1/radarx-qpe-v1.onnx", None, "MIT",
    "radarx developers (2026): radarx-qpe-v1, ...",
    task="rainfall-estimation",
    features=["dbz", "zdr", "kdp", "rhohv", "wavelength"],
)
```

## References

- Testud, J., S. Oury, R. A. Black, P. Amayenc, and X. Dou, 2001: The concept
  of "normalized" distribution to describe raindrop spectra. *J. Appl.
  Meteor.*, **40**, 1118-1140,
  https://doi.org/10.1175/1520-0450(2001)040<1118:TCONDT>2.0.CO;2
- Bringi, V. N., V. Chandrasekar, J. Hubbert, E. Gorgucci, W. L. Randeu, and
  M. Schoenhuber, 2003: Raindrop size distribution in different climatic
  regimes from disdrometer and dual-polarized radar analysis. *J. Atmos.
  Sci.*, **60**, 354-365,
  https://doi.org/10.1175/1520-0469(2003)060<0354:RSDIDC>2.0.CO;2
- Nix, D. A., and A. S. Weigend, 1994: Estimating the mean and variance of the
  target probability distribution. *Proc. IEEE Int. Conf. Neural Networks*,
  55-60, https://doi.org/10.1109/ICNN.1994.374138
- Wolfensberger, D., M. Gabella, M. Boscacci, U. Germann, and A. Berne, 2021:
  RainForest: a random forest algorithm for quantitative precipitation
  estimation over Switzerland. *Atmos. Meas. Tech.*, **14**, 3169-3193,
  https://doi.org/10.5194/amt-14-3169-2021
- Atlas, D., R. C. Srivastava, and R. S. Sekhon, 1973: Doppler radar
  characteristics of precipitation at vertical incidence. *Rev. Geophys.*,
  **11**, 1-35, https://doi.org/10.1029/RG011i001p00001
