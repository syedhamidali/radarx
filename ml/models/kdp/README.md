# Physics-constrained neural network for KDP and backscatter phase

This folder holds the training, export and evaluation code of the network used
by `radarx.retrieve.estimate_kdp(..., method="ml")`. It is not part of the
radarx package and is never imported by it. The trained weights are **not
distributed yet**; `method="ml"` raises a clear error until a model is
registered with `radarx.ml.register_model(...)` or passed as `model=...`.

## Problem

The measured differential phase along a ray is

$$\Psi_{DP}(r) = \Phi_{DP}^{0} + 2\int_0^r K_{DP}(s)\,ds + \delta(r) + \varepsilon(r),$$

folded into a 360° interval, with a system offset $\Phi_{DP}^0$, a backscatter
phase $\delta$ (large drops at C and X band, the melting layer, hail) and noise
$\varepsilon$ that grows at low signal-to-noise ratio and low $\rho_{hv}$. Some
systems record $-\Psi_{DP}$. Classical estimators (Hubbert and Bringi 1995;
Vulpiani et al. 2012; the monotone constraint of Maesaka et al. 2012) trade
range resolution against noise with fixed filters and treat $\delta$ as an
outlier. The network learns this trade-off from simulated rays whose $K_{DP}$
and $\delta$ are known, and returns $K_{DP}$, $\delta$ and an uncertainty
of $K_{DP}$ per gate.

## Method

**Pre-processing (deterministic, shared with the other methods).** Gates are
masked with the $\rho_{hv}$ and circular phase-texture tests of
`estimate_kdp`, the sign convention is detected from the rain gates of the
sweep, the system offset is estimated and the phase is unfolded
(`radarx.retrieve.kdp._ml_features`). The same function featurises the
training rays, so training and inference inputs are identical.

**Inputs** (per gate, `ML_FEATURES`): half the range derivative of the
unfolded, gap-bridged phase (×0.1), the valid-gate mask, $\rho_{hv}$ and a
flag for its presence, $Z_H/50$ and a flag, and the gate spacing (km). The
derivative input makes the network invariant to the offset and folding.

**Network** (`model.py`): a one-dimensional U-Net (Ronneberger et al. 2015)
along range: four stride-2 levels (widths 16-32-48-64-64), residual blocks of
kernel-5 convolutions, dilated residual blocks at 1/16 resolution
(receptive field about ±500 gates) and skip connections; 0.43 M parameters,
fully convolutional, so rays of any length run in one call. Outputs:
$K_{DP}$, $\delta$ and $\sigma_{K_{DP}}$ (softplus).

**Loss** (`model.py`): supervised terms on simulated truth, Huber errors of
$K_{DP}$ and $\delta$ and the Gaussian negative log-likelihood of
$\sigma_{K_{DP}}$ with the mean detached (Nix and Weigend 1994), plus physics
terms in the spirit of physics-informed learning (Raissi et al. 2019) that
need no truth:

- *phase consistency*: $\Psi_{DP} - \hat\delta - 2\int \hat K_{DP}\,dr$ must be
  constant along the valid gates of a ray (least-squares constant per ray,
  Huber in units of 3°);
- *non-negativity*: $\mathrm{ReLU}(-\hat K_{DP})^2$ in rain
  ($\rho_{hv} \ge 0.97$, $Z_H \ge 30$ dBZ), a soft constraint so that
  negative $K_{DP}$ remains possible in ice and hail;
- *smoothness*: squared second difference of $\hat K_{DP}$.

Weights: 1 ($K_{DP}$), 0.5 ($\delta$), 0.1 (NLL), 0.2 (consistency),
1 (non-negativity), 0.05 (smoothness). The ablation `--no-physics` sets the
last three to zero.

**Processed phase.** At inference `PHIDP_processed` is
$2\int \hat K_{DP}\,dr$ plus the constant fitted to $\Psi_{DP} - \hat\delta$,
so phase and $K_{DP}$ are consistent by construction.

## Training data

`simulate.py` builds rays (512 gates; gate spacing 75-500 m; S, C and X band;
water temperature 5-25 °C) from range profiles of normalized-gamma DSDs
(Testud et al. 2001): $D_m$, $\log_{10}N_w$ and $\mu$ follow a stratiform
background plus 0-5 Gaussian or flat-topped convective cells, with correlated
perturbations and size sorting (large $D_m$, low $N_w$) at cell edges. $Z_H$,
$Z_{DR}$, $K_{DP}$, $\delta$ and $\rho_{hv}$ of every gate are integrals over the
DSD of the radarx T-matrix single-drop tables (`radarx.retrieve.dsd`,
Mishchenko and Travis 1998; Brandes et al. 2002 axis ratios), so $K_{DP}$ and
$\delta$ are physically consistent. On top:

| effect | how |
|---|---|
| melting layer (35 % of rays) | Gaussian $\delta$ bump up to 6/9/12° (S/C/X), $\rho_{hv}$ dip, bright band; small $K_{DP}$ in ice beyond |
| hail (20 %) | extra $\delta$ up to 4/15/15°, lower $\rho_{hv}$, reduced $K_{DP}$ in the strongest core |
| vertically aligned ice (5 %) | $K_{DP}$ of −0.1 to −0.5 °/km |
| attenuation | $Z_H$ reduced by 0.016/0.08/0.28 dB per degree of $\Phi_{DP}$ |
| noise | 1.5-5° phase noise, inflated at low SNR and low $\rho_{hv}$ |
| non-meteorological gates | clear-air segments, clutter spikes, near-radar clutter (random phase, low $\rho_{hv}$), no-data gates and gaps |
| system | random offset, folding into [−180, 180) or [0, 360), reversed sign (20 %), missing $\rho_{hv}$ (5 %) or $Z_H$ (10 %) |

Batches are generated on the fly by worker processes into a replay buffer
(500 batches of 64 rays, continuously replaced); every step draws a batch
and a random 384-gate window.

**Real rays (optional fine-tuning).** `real_data.py` cuts 512-gate windows
from NEXRAD volumes (KGWX 30 March 2022, 21:00-22:38 UTC); they enter only
through the physics terms. KGWX 23:46 UTC, KLBB 1 June 2016 and the C-band
CSAPR2 PPI are held out for validation.

## Results

See the tables and figures of the pull request that added this folder;
`evaluate.py` regenerates them (`tables.md`, `metrics.json`, PNG figures).

RESULTS_PLACEHOLDER

## Reproduce

```bash
python -m venv venv && . venv/bin/activate
pip install -e /path/to/radarx -r requirements.txt
cd ml/models/kdp
python train.py --out runs/full --steps 12000                # physics-constrained
python train.py --out runs/nophys --steps 12000 --no-physics --seed 1
python real_data.py real_rays.npz KGWX20220330_2*_V06         # optional
python train.py --out runs/ft --init runs/full/model.pt --real real_rays.npz \
    --steps 2000 --lr 3e-4                                   # optional fine-tuning
python export.py runs/full/model.pt runs/full/radarx-kdp.onnx  # prints the SHA-256
python evaluate.py --out results --model full=runs/full/radarx-kdp.onnx \
    --model nophys=runs/nophys/radarx-kdp.onnx --kgwx KGWX20220330_234639_V06 \
    --klbb --csapr2
```

NEXRAD files: `https://unidata-nexrad-level2.s3.amazonaws.com/2022/03/30/KGWX/`.
Training on an Apple M1 Pro (MPS) takes about an hour per model.

Using a trained model:

```python
from radarx import ml
from radarx.retrieve import estimate_kdp

ml.register_model("radarx-kdp-v1", "radarx-kdp.onnx", sha256="...", licence="MIT",
                  citation="...", task="kdp")
out = estimate_kdp(sweep, method="ml")  # KDP, PHIDP_processed,
                                        # PHIDP_BACKSCATTER, KDP_UNCERTAINTY
```

## References

Brandes, E. A., G. Zhang, and J. Vivekanandan, 2002: Experiments in rainfall
estimation with a polarimetric radar in a subtropical environment. *J. Appl.
Meteor.*, **41**, 674-685, https://doi.org/10.1175/1520-0450(2002)041<0674:EIREWA>2.0.CO;2

Giangrande, S. E., R. McGraw, and L. Lei, 2013: An application of linear
programming to polarimetric radar differential phase processing. *J. Atmos.
Oceanic Technol.*, **30**, 1716-1729, https://doi.org/10.1175/JTECH-D-12-00147.1

Hubbert, J., and V. N. Bringi, 1995: An iterative filtering technique for the
analysis of copolar differential phase and dual-frequency radar measurements.
*J. Atmos. Oceanic Technol.*, **12**, 643-648,
https://doi.org/10.1175/1520-0426(1995)012<0643:AIFTFT>2.0.CO;2

Maesaka, T., K. Iwanami, and M. Maki, 2012: Non-negative KDP estimation by
monotone increasing PhiDP assumption below melting layer. *Proc. Seventh
European Conf. on Radar in Meteorology and Hydrology (ERAD 2012)*, Toulouse,
France (conference paper, no DOI).

Mishchenko, M. I., and L. D. Travis, 1998: Capabilities and limitations of a
current FORTRAN implementation of the T-matrix method for randomly oriented,
rotationally symmetric scatterers. *J. Quant. Spectrosc. Radiat. Transfer*,
**60**, 309-324, https://doi.org/10.1016/S0022-4073(98)00008-9

Nix, D. A., and A. S. Weigend, 1994: Estimating the mean and variance of the
target probability distribution. *Proc. 1994 IEEE Int. Conf. on Neural
Networks (ICNN'94)*, Vol. 1, 55-60, https://doi.org/10.1109/ICNN.1994.374138

Raissi, M., P. Perdikaris, and G. E. Karniadakis, 2019: Physics-informed
neural networks: A deep learning framework for solving forward and inverse
problems involving nonlinear partial differential equations. *J. Comput.
Phys.*, **378**, 686-707, https://doi.org/10.1016/j.jcp.2018.10.045

Ronneberger, O., P. Fischer, and T. Brox, 2015: U-Net: Convolutional networks
for biomedical image segmentation. *Medical Image Computing and
Computer-Assisted Intervention (MICCAI 2015)*, Lecture Notes in Computer
Science, Vol. 9351, 234-241, https://doi.org/10.1007/978-3-319-24574-4_28

Ryzhkov, A. V., S. E. Giangrande, V. M. Melnikov, and T. J. Schuur, 2005:
Calibration issues of dual-polarization radar measurements. *J. Atmos.
Oceanic Technol.*, **22**, 1138-1155, https://doi.org/10.1175/JTECH1772.1

Testud, J., S. Oury, R. A. Black, P. Amayenc, and X. Dou, 2001: The concept
of "normalized" distribution to describe raindrop spectra: A tool for cloud
physics and cloud remote sensing. *J. Appl. Meteor.*, **40**, 1118-1140,
https://doi.org/10.1175/1520-0450(2001)040<1118:TCONDT>2.0.CO;2

Vulpiani, G., M. Montopoli, L. D. Passeri, A. G. Gioia, P. Giordano, and
F. S. Marzano, 2012: On the use of dual-polarized C-band radar for operational
rainfall retrieval in mountainous areas. *J. Appl. Meteor. Climatol.*, **51**,
405-425, https://doi.org/10.1175/JAMC-D-10-05024.1

Wang, Y., and V. Chandrasekar, 2009: Algorithm for estimation of the specific
differential phase. *J. Atmos. Oceanic Technol.*, **26**, 2565-2578,
https://doi.org/10.1175/2009JTECHA1358.1
