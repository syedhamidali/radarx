# Trajectory-matched Bayesian DSD retrieval

Analysis code and results behind `radarx.retrieve.dsd_bayesian` (issue #141):
a per-gate Bayesian retrieval of the normalized gamma drop size distribution
(DSD) whose prior is learned from disdrometer spectra matched to the radar
beam along the drop fall trajectories, validated against the PERiLS 2022
Portable In situ Precipitation Stations (PIPS). These scripts are not part of
the radarx package; they need the PIPS files (not distributed) and radar
volumes.

## Method

**State and forward model.** $x = (\log_{10} N_w, D_m, \mu)$ of
$N(D) = N_w f(\mu) (D/D_m)^\mu \exp[-(4+\mu) D/D_m]$ (Testud et al. 2001).
For each $(D_m, \mu)$ on a grid ($D_m$ 0.4-4.4 mm by 0.05 mm, $\mu$ -0.5-12
by 0.5) the T-matrix scattering tables of radarx (Brandes et al. 2002 axis
ratios, 7° canting) give $Z_H = 10\log_{10} N_w + L$, $Z_{DR}$, and
$K_{DP}$, $A_H$ proportional to $N_w$.

**Error model.** Gaussian errors: $Z_H$ and $Z_{DR}$ with noise plus an
unknown calibration bias (variances add per gate), $K_{DP}$ and $A_H$ with
absolute and relative parts. Defaults: 1 + 1 dB, 0.2 + 0.1 dB, 0.3 °/km +
10 %, 0.01 dB/km + 20 %. They can be learned from radar-disdrometer pairs
(below).

**Priors.** $p(D_m, \mu)\,\mathcal{N}(\log_{10} N_w; m(D_m,\mu), s(D_m,\mu)^2)$:
a generic, weakly informative prior (ranges of Bringi et al. 2003,
$\mu$-$\Lambda$ relation of Cao et al. 2008 with a spread of 2), or a kernel
density estimate from fitted disdrometer DSDs with the conditional mean and
spread of $\log_{10} N_w$, mixed with 1 % of the generic prior.

**Inference.** On every grid node the posterior in $\log_{10} N_w$ is
integrated with the Laplace method (Gauss-Newton mode, exact without
$K_{DP}$/$A_H$); the posterior is the resulting mixture. Nodes more than 15
nats below the best closed-form $Z_H$/$Z_{DR}$ evidence are skipped. The
output is the posterior mean, standard deviation, quantiles and MAP of
$\log_{10} N_w$, $D_m$, $\mu$, $R$ and $W$, the log evidence and the misfit.
A C++ kernel (std::thread, dynamic blocks) runs all gates of a volume in one
call, with a NumPy oracle.

## Trajectory matching (stand-in for #140)

`match.py` replaces the rain trajectory model of #140 until it is merged.
Drops fall from the beam (height $h$ over the probe) at the Atlas et al.
(1973) speed with the $(\rho_0/\rho)^{0.4}$ density correction, drift with
the ERA5 layer-mean wind $\mathbf{u}$, and the echo pattern moves with the
storm motion $\mathbf{c}$ (median of `radarx.retrieve.estimate_motion` over
consecutive volumes; IOP2: 15.8, 23.3 m/s, matching an independent estimate
of 15, 24 m/s). For a radar point at the source of 2-mm drops,
$\mathbf{x}_P - \mathbf{u}\tau_{2}$, each size bin is read from the probe at

$$t(D) = t_{gate} + \tau_D - \frac{\mathbf{u}\cdot\hat{\mathbf{c}}}{|\mathbf{c}|}(\tau_D - \tau_2),$$

the time the drops of that size from the same point of the moving pattern
reach it. This undoes size sorting by fall speed and drift along the storm
motion; cross-track drift is reported but cannot be undone with one probe.
Evaporation, break-up and coalescence below the beam are neglected.

*Naive collocation*: gates over the probe and the 60-s spectrum centred on
the overpass. Radar samples are linear-Z averages of the gates within 1 km
(0.5°, 0.9° and 1.3° sweeps of KGWX; KDP from `estimate_kdp`).

## Data

- PIPS (OTT Parsivel2) from PERiLS 2022: IOP1 (22 March, 4 probes), IOP2
  (30-31 March, 4 probes, 14-23 km from KGWX), IOP3 (5 April, 6 probes); 10-s
  spectra with the quality control of the PIPS processing (strong wind,
  splashing, margin fallers, non-rain), averaged over 60 s. Minutes with
  $R \ge 0.5$ mm/h and at least 50 drops m⁻³ are used.
- Radar: KGWX CfRadial volumes for IOP2 (19 volumes, 23:26-01:25 UTC).
- ERA5 profiles via `radarx.io.sounding.era5_profile` (Google ARCO-ERA5).

Aloft-equivalent samples for the prior: IOP1 355, IOP2 316, IOP3 616
minutes (beam heights 1.1-1.3 km, 0.2-0.35 km and 0.7-0.9 km).
`radarx/retrieve/data/dsd_prior_perils2022.csv` holds all of them (the
`"perils2022"` prior); the validation below uses only IOP1 + IOP3 (971
samples) so that IOP2 stays independent.

## Results: IOP2 against PIPS

Leave-one-IOP-out prior (IOP1 + IOP3). All methods use $Z_H$, $Z_{DR}$ and
$K_{DP}$; constrained gamma (Cao et al. 2008) and normalized gamma ($\mu=3$)
take the intercept from $K_{DP}$ where it is at least 1 °/km.

**Radar minus PIPS-simulated variables.** Naive: $Z_H$ −3.4 ± 4.1 dB,
$Z_{DR}$ −0.03 ± 0.39 dB (161 pairs); trajectory: $Z_H$ −3.0 ± 2.9 dB,
$Z_{DR}$ −0.05 ± 0.41 dB (105 pairs). Trajectory matching cuts the $Z_H$
scatter by 30 %.

**Offset-corrected inputs and learned error model** (leave-one-probe-out:
the $Z_H$/$Z_{DR}$ offsets and the robust spreads of radar minus PIPS for each
probe come from the other three probes):

| pairs | method | $D_m$ bias / RMSE (mm) | $\log_{10}N_w$ bias / RMSE | $\mu$ bias / RMSE | $R$ bias / NRMSE (%) |
|---|---|---|---|---|---|
| naive | constrained gamma | −0.24 / 0.39 | +0.33 / 0.42 | −1.9 / 2.6 | −2 / 68 |
| naive | normalized gamma (μ=3) | +0.03 / 0.31 | −0.02 / 0.28 | +0.3 / 2.2 | −17 / 76 |
| naive | Bayesian, generic prior | −0.16 / 0.33 | +0.23 / 0.34 | −1.1 / 2.1 | −0 / 66 |
| naive | Bayesian, learned prior | −0.02 / 0.28 | +0.04 / 0.29 | +0.2 / 1.8 | −15 / 75 |
| trajectory | constrained gamma | −0.30 / 0.40 | +0.33 / 0.41 | −1.6 / 2.3 | −5 / 52 |
| trajectory | normalized gamma (μ=3) | +0.02 / 0.26 | −0.06 / 0.24 | +0.9 / 2.1 | −20 / 60 |
| trajectory | Bayesian, generic prior | −0.23 / 0.34 | +0.30 / 0.36 | −0.8 / 1.8 | −0 / 50 |
| trajectory | **Bayesian, learned prior** | **−0.07 / 0.24** | +0.11 / 0.29 | +0.5 / **1.7** | −14 / 57 |

Coverage of the 68 / 95 % credible intervals (trajectory pairs, learned
prior and error model): $D_m$ 0.68 / 0.94, $\log_{10} N_w$ 0.71 / 0.98,
$\mu$ 0.65 / 0.93, $R$ 0.48 / 0.81. With the default error model and no
offset correction the intervals are too narrow ($D_m$ 0.53 / 0.83): the
radar-disdrometer mismatch is larger than the radar errors alone.

**As observed** (no offset correction, default error model), trajectory
pairs: $D_m$ RMSE 0.41 (constrained), 0.25 (normalized), 0.36 (Bayesian
generic), 0.27 mm (Bayesian learned); $R$ NRMSE 56, 63, 64, 75 % (naive: 70,
79, 86, 96 %). The $-3$ dB $Z_H$ offset of KGWX relative to the Parsivels
drives the low rain rates.

Findings:

1. Trajectory matching lowers the rain-rate NRMSE of every method by 10-20
   points and the $Z_H$ scatter by 30 %, i.e. a large part of the usual
   radar-disdrometer disagreement is the pairing, not the retrieval.
2. The learned prior gives the best $D_m$ and $\mu$ and removes most of the
   $D_m$ bias of the generic prior and of the constrained-gamma relation
   (both tie small $\mu$ to large drops in a way these QLCS DSDs do not
   follow); with the error model learned from the pairs its credible
   intervals are calibrated for $D_m$, $N_w$ and $\mu$.
3. $\mu$ is weakly constrained by S-band $Z_H$, $Z_{DR}$, $K_{DP}$: its
   posterior spread (≈2) is close to the prior's and to the actual error.
4. Rain-rate intervals are too narrow (0.48 / 0.81): the PIPS sampling error
   at 60 s and the beam-to-ground changes the model neglects (evaporation)
   are not in the error model yet.

Caveats: one IOP, 105 trajectory pairs from 19 volumes (3 sweeps each, so
pairs are correlated); KGWX only. Radar data for IOP1 (KGWX) and IOP3
(KMXX) would allow a cross-IOP validation of the error model; they were not
used here.

Figures (`evaluate.py`): `dm_scatter_IOP2.png`, `dm_timeseries_IOP2.png`,
`radar_minus_pips_IOP2.png`.

## Synthetic truth

DSDs drawn from the generic prior and observations simulated with the error
model are recovered with calibrated intervals (68/95 % coverage within ±0.03
for $\log_{10} N_w$, $D_m$, $\mu$ and $R$, 3000 draws; `tests/test_dsd_bayes.py`
and the `Bayesian_DSD` notebook): $D_m$ RMSE 0.19 mm against a prior spread
of 0.6 mm.

## Reproduce

```
python build_pairs.py IOP2 --radar-dir KGWX_CFRAD_DIR --era5 IOP2.nc --out out
python prior_samples.py --out out/samples.csv --era5-dir ERA5_DIR --motion IOP2=15.83,23.29
python evaluate.py --pairs out --iop IOP2 --samples out/samples.csv --out out/results
python evaluate.py --pairs out --iop IOP2 --samples out/samples.csv --out out/results_cal --calibrate
python benchmark.py KGWX_VOLUME
```

`pips.py` reads the PIPS files from `~/Downloads/MULTIDOPPLER/PIPS_data`
(set `pips.ROOT`). ERA5 profiles: `radarx.io.sounding.era5_profile(lat, lon,
time, source="gcs")` saved to `<IOP>.nc`.

## Not done yet

- Switch `match.py` to the rain trajectory model of #140 once merged
  (per-gate trajectories, evaporation).
- An ONNX variant trained on the same pairs (radarx[ml], #146).
- Radar validation on IOP1/IOP3 and other disdrometer networks.

## References

- Testud, J., S. Oury, R. A. Black, P. Amayenc, and X. Dou, 2001, *J. Appl.
  Meteor.*, **40**, 1118-1140, https://doi.org/10.1175/1520-0450(2001)040<1118:TCONDT>2.0.CO;2
- Brandes, E. A., G. Zhang, and J. Vivekanandan, 2002, *J. Appl. Meteor.*,
  **41**, 674-685, https://doi.org/10.1175/1520-0450(2002)041<0674:EIREWA>2.0.CO;2
- Bringi, V. N., V. Chandrasekar, J. Hubbert, E. Gorgucci, W. L. Randeu, and
  M. Schoenhuber, 2003, *J. Atmos. Sci.*, **60**, 354-365,
  https://doi.org/10.1175/1520-0469(2003)060<0354:RSDIDC>2.0.CO;2
- Cao, Q., G. Zhang, E. Brandes, T. Schuur, A. Ryzhkov, and K. Ikeda, 2008,
  *J. Appl. Meteor. Climatol.*, **47**, 2238-2255, https://doi.org/10.1175/2008JAMC1732.1
- Cao, Q., and G. Zhang, 2009, *J. Appl. Meteor. Climatol.*, **48**, 406-425,
  https://doi.org/10.1175/2008JAMC2026.1
- Cao, Q., G. Zhang, and M. Xue, 2013, *J. Appl. Meteor. Climatol.*, **52**,
  169-185, https://doi.org/10.1175/JAMC-D-12-0101.1
- Kumjian, M. R., and A. V. Ryzhkov, 2012: The impact of size sorting on the
  polarimetric radar variables. *J. Atmos. Sci.*, **69**, 2042-2060,
  https://doi.org/10.1175/JAS-D-11-0125.1
- Rodgers, C. D., 2000: *Inverse Methods for Atmospheric Sounding*. World
  Scientific, https://doi.org/10.1142/3171
- Atlas, D., R. C. Srivastava, and R. S. Sekhon, 1973, *Rev. Geophys.*,
  **11**, 1-35, https://doi.org/10.1029/RG011i001p00001
- Foote, G. B., and P. S. du Toit, 1969, *J. Appl. Meteor.*, **8**, 249-253,
  https://doi.org/10.1175/1520-0450(1969)008<0249:TVORA>2.0.CO;2
