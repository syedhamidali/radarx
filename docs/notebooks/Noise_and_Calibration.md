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

# Noise correction and calibration

Two groups of gate-wise and sweep-wise corrections, in `radarx.retrieve`, that
decide how far the dual-polarization fields can be trusted.

**Noise** (`radarx.retrieve.noise`):

- `noise_level`: the noise level of a sweep, as the reflectivity of the noise at
  1 km, by the objective method of
  [Hildebrand and Sekhon (1974)](https://doi.org/10.1175/1520-0450(1974)013<0808:ODOTNL>2.0.CO;2)
  or from the lowest reflectivity that was reported;
- `snr`: the signal-to-noise ratio of every gate;
- `correct_rhohv_noise` and `correct_zdr_noise`: $\rho_{hv}$ and $Z_{DR}$ corrected
  for the noise in the two receiver channels
  ([Gourley et al. 2006](https://doi.org/10.1175/JTECH1912.1)).

**Calibration** (`radarx.retrieve.calibration`), after
[Ryzhkov et al. (2005)](https://doi.org/10.1175/JTECH1772.1):

- `zdr_offset`: the $Z_{DR}$ offset from vertical pointing, light rain or dry snow;
- `z_offset`: the reflectivity offset from the self-consistency of $Z$, $Z_{DR}$
  and $K_{DP}$ in rain
  ([Gorgucci et al. 1992](https://doi.org/10.1109/36.175319));
- `phidp_offset`: the system differential phase of each ray and of the sweep.

All of them run in compiled, multithreaded kernels with a NumPy fallback that
gives the same numbers, accept a `DataArray`, a `Dataset` (a sweep) or a
`DataTree` (a volume, sweep by sweep), keep dask arrays lazy where the
computation is element-wise, and never change their input. Each result carries
the method and the DOIs of the papers (`radarx_method`, `radarx_references`).
The offsets are the measured value minus the true one, so the corrected field
is `field - offset`.

```{code-cell} ipython3
import importlib
import time
import warnings

import cmweather  # noqa: F401  radar colormaps
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xradar as xd
from open_radar_data import DATASETS

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.retrieve import (
    correct_rhohv_noise,
    correct_zdr_noise,
    estimate_kdp,
    noise_level,
    phidp_offset,
    snr,
    z_offset,
    zdr_offset,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
dsd_module = importlib.import_module("radarx.retrieve.dsd")  # scattering tables
noise_module = importlib.import_module("radarx.retrieve.noise")
calibration_module = importlib.import_module("radarx.retrieve.calibration")
rng = np.random.default_rng(42)
```

## Data

The real data are the S-band NEXRAD volume of Lubbock, Texas (KLBB), of 1 June
2016, from `open_radar_data`: the 0.5° surveillance cut and the 1.5° cut, whose
no-data codes are masked.

```{code-cell} ipython3
path = DATASETS.fetch("KLBB20160601_150025_V06")
klbb = xd.io.open_nexradlevel2_datatree(path, sweep=[0, 2])
for name in ("sweep_0", "sweep_2"):
    ds = klbb[name].to_dataset(inherit=False)
    for var, limit in (("DBZH", -32.0), ("ZDR", -12.9), ("RHOHV", 0.21)):
        ds[var] = ds[var].where(ds[var] > limit)
    klbb[name] = ds
sweep = klbb["sweep_0"].to_dataset()
print(f"{sweep.sizes['azimuth']} rays x {sweep.sizes['range']} gates, "
      f"elevation {float(sweep.elevation.mean()):.2f}°, VCP {klbb.attrs['scan_name']}")
```

## Noise level and SNR

The receiver noise has the same power at every range, so the reflectivity that
gives that power grows as $20\log_{10} r$. In the range-corrected reflectivity
$Z - 20\log_{10}(r/\mathrm{km})$ the noise gates lie on one level, the noise level
$Z_N$, and the SNR is $Z - Z_N - 20\log_{10}(r/\mathrm{km})$.

To see how well the two methods find $Z_N$ we need a truth. The synthetic sweep
below has a rain patch and noise of $-40$ dBZ at 1 km, with each gate power
the average of 64 independent samples (a gamma distribution, as in a radar).

```{code-cell} ipython3
def synthetic_sweep(noise=-40.0, samples=64, threshold=None, weak=False, seed=1):
    """Reflectivity of a rain patch and noise of a known level.

    With ``weak`` there is also echo everywhere, with an SNR from -5 to 25 dB, as
    in a real sweep where most gates hold some weak signal.
    """
    g = np.random.default_rng(seed)
    nray, ngate = 120, 800
    range_m = 2000.0 + 250.0 * np.arange(ngate)
    r_db = 20 * np.log10(range_m / 1000.0)[None]
    signal = np.full((nray, ngate), -np.inf)
    if weak:
        signal[:] = noise + g.uniform(-5.0, 25.0, (nray, ngate)) + r_db
    signal[20:60, 100:400] = 25.0 + 8.0 * g.standard_normal((40, 300))
    power = 10 ** (0.1 * (signal - r_db)) * g.gamma(samples, 1 / samples, (nray, ngate))
    power += 10 ** (0.1 * noise) * g.gamma(samples, 1 / samples, (nray, ngate))
    dbz = 10 * np.log10(power) + r_db
    if threshold is not None:  # remove the gates below an SNR, as many radars do
        dbz = np.where(dbz - noise - r_db >= threshold, dbz, np.nan)
    return xr.Dataset(
        {"DBZH": (("azimuth", "range"), dbz.astype("float32"), {"units": "dBZ"})},
        coords={"azimuth": np.arange(nray) * 3.0, "range": range_m},
    )


raw = synthetic_sweep()
hs = noise_level(raw)
floor = noise_level(raw, method="floor", percentile=0.5)
print(f"truth -40.00 dBZ | Hildebrand-Sekhon {float(hs.NOISE_LEVEL):.2f} "
      f"({int(hs.NOISE_SAMPLES)} of {raw.DBZH.size} gates) | "
      f"0.5 % floor {float(floor.NOISE_LEVEL):.2f} (below the mean of the noise gates)")
```

`method="hildebrand"` needs gates without signal and `n_samples`, the number of
independent samples in a gate; `method="floor"` is for data from which the gates
below an SNR threshold were removed and takes a low percentile of the range-corrected
reflectivity minus the threshold. The figure shows the range-corrected
reflectivity of the synthetic sweep and the two estimates.

```{code-cell} ipython3
thresholded = synthetic_sweep(threshold=3.0, weak=True)
floor3 = noise_level(thresholded, method="floor", percentile=0.5, snr_threshold=3.0)
print(f"thresholded at 3 dB: floor method {float(floor3.NOISE_LEVEL):.2f} dBZ (truth -40.00)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, layout="constrained")
for ax, data, title in (
    (axes[0], raw, "all gates kept"),
    (axes[1], thresholded, "gates below 3 dB SNR removed"),
):
    corrected = data.DBZH - 20 * np.log10(data.range / 1000.0)
    ax.scatter(
        np.broadcast_to(data.range / 1e3, corrected.shape)[::3, ::3],
        corrected.values[::3, ::3],
        s=1,
        alpha=0.3,
        color="0.45",
    )
    ax.axhline(-40, color="k", lw=1, label="truth")
    ax.set(xlabel="range (km)", title=title)
axes[0].axhline(float(hs.NOISE_LEVEL), color="C0", ls="--", label="Hildebrand-Sekhon")
axes[1].axhline(float(floor3.NOISE_LEVEL), color="C1", ls="--", label="floor, threshold 3 dB")
axes[0].set_ylabel("$Z - 20\\log_{10}(r/\\mathrm{km})$ (dBZ)")
for ax in axes:
    ax.legend(frameon=False, loc="upper right")
plt.show()

levels = [-55.0, -40.0, -25.0]
hs_levels = [float(noise_level(synthetic_sweep(noise=n)).NOISE_LEVEL) for n in levels]
print("Hildebrand-Sekhon for noise levels", levels, "->", np.round(hs_levels, 2))
for ns in (1, 16, 64, 256):
    value = float(noise_level(raw, n_samples=ns).NOISE_LEVEL)
    print(f"n_samples={ns:4d}: {value:.2f} dBZ")
```

With the right number of samples the method finds the level to a few hundredths of a
dB; too few samples (1) let some signal into the noise (+0.3 dB), too many
reject noise (−0.7 dB at 256).

### NEXRAD

NEXRAD level 2 holds only the gates above its SNR threshold, so no gate is pure
noise and the floor method is the one to use. The lowest range-corrected
reflectivity follows the noise: about −41 dBZ at 1 km at both cuts.

```{code-cell} ipython3
levels = noise_level(klbb, method="floor")
for name in ("sweep_0", "sweep_2"):
    print(name, f"floor {float(levels[name].NOISE_LEVEL):.2f} dBZ at 1 km, from "
          f"{int(levels[name].NOISE_SAMPLES)} gates")
print("Hildebrand-Sekhon on the same data (it lands on the low edge):",
      round(float(noise_level(sweep).NOISE_LEVEL), 2))

snr_volume = snr(klbb, noise=levels)  # one value per sweep, from the result above
snr_sweep = snr_volume["sweep_0"]["SNR"]
az = np.deg2rad(sweep.azimuth.values)[:, None]
x = sweep.range.values[None, :] * np.sin(az) / 1e3
y = sweep.range.values[None, :] * np.cos(az) / 1e3
fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
for ax, field, label, kw in (
    (axes[0], sweep.DBZH, "$Z_H$ (dBZ)", dict(cmap="ChaseSpectral", vmin=-10, vmax=70)),
    (axes[1], snr_sweep, "SNR (dB)", dict(cmap="viridis", vmin=0, vmax=60)),
):
    pm = ax.pcolormesh(x, y, field.values, shading="nearest", **kw)
    fig.colorbar(pm, ax=ax, label=label, shrink=0.85)
    ax.set(xlim=(-150, 150), ylim=(-150, 150), aspect="equal", xlabel="east of KLBB (km)")
axes[0].set_ylabel("north of KLBB (km)")
plt.show()
```

## Correcting RHOHV and ZDR for noise

Noise adds power to each channel and nothing to the cross-correlation. With the SNR
of the two channels $s_h$ and $s_v$ the measured correlation coefficient and
differential reflectivity (linear) are

$$\rho_{hv}^m = \frac{\rho_{hv}}{\sqrt{(1+1/s_h)(1+1/s_v)}},\qquad
Z_{DR}^m = Z_{DR}\,\frac{1+1/s_h}{1+1/s_v},\qquad s_v = \frac{N_h}{N_v}\frac{s_h}{Z_{DR}}.$$

`correct_rhohv_noise` and `correct_zdr_noise` invert them (the form of the first
one is Eq. 6 of Gourley et al. 2006; the second is derived in the module
documentation). To check them against a truth we simulate what a radar does:
complex Gaussian voltages in the two channels with a known correlation
($\rho_{hv}=0.99$) and $Z_{DR}$ (1.2 dB), noise added in each channel (the noise
power of H is 1.5 dB above that of V), and the moments formed from 400 000
samples.

```{code-cell} ipython3
def simulate(rho, zdr_db, snr_db, alpha_db, n=400_000, seed=0):
    """Measured RHOHV and ZDR from noisy complex Gaussian voltages."""
    g = np.random.default_rng(seed)
    cn = lambda: (g.standard_normal(n) + 1j * g.standard_normal(n)) / np.sqrt(2)
    n_h, n_v = 1.0, 10 ** (-0.1 * alpha_db)
    p_h = n_h * 10 ** (0.1 * snr_db)
    p_v = p_h / 10 ** (0.1 * zdr_db)
    a, b = cn(), cn()
    m_h = np.sqrt(p_h) * a + np.sqrt(n_h) * cn()
    m_v = np.sqrt(p_v) * (rho * a + np.sqrt(1 - rho**2) * b) + np.sqrt(n_v) * cn()
    ph, pv = np.mean(abs(m_h) ** 2), np.mean(abs(m_v) ** 2)
    return abs(np.mean(np.conj(m_h) * m_v)) / np.sqrt(ph * pv), 10 * np.log10(ph / pv)


truth_rho, truth_zdr, alpha = 0.99, 1.2, 1.5
snr_grid = np.arange(-2.0, 31.0, 2.0)
measured = np.array([simulate(truth_rho, truth_zdr, s, alpha, seed=i) for i, s in enumerate(snr_grid)])
sim = xr.Dataset(
    {
        "RHOHV": ("g", measured[:, 0]),
        "ZDR": ("g", measured[:, 1]),
        "SNR": ("g", snr_grid),
    }
)
rho_fixed = correct_rhohv_noise(
    sim, zdr=True, zdr_includes_noise=True, noise_ratio=alpha, min_snr=-10.0
)
zdr_fixed = correct_zdr_noise(sim, noise_ratio=alpha, min_snr=-10.0)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
axes[0].plot(snr_grid, measured[:, 0], "o-", label="measured")
axes[0].plot(snr_grid, rho_fixed, "s-", label="corrected")
axes[0].axhline(truth_rho, color="k", lw=1, label="truth")
axes[0].set(xlabel="SNR of H (dB)", ylabel="$\\rho_{hv}$")
axes[1].plot(snr_grid, measured[:, 1], "o-", label="measured")
axes[1].plot(snr_grid, zdr_fixed, "s-", label="corrected")
axes[1].axhline(truth_zdr, color="k", lw=1, label="truth")
axes[1].set(xlabel="SNR of H (dB)", ylabel="$Z_{DR}$ (dB)")
for ax in axes:
    ax.legend(frameon=False)
plt.show()

use = snr_grid >= 0
print(f"SNR >= 0 dB: largest error of RHOHV {np.abs(measured[use, 0] - truth_rho).max():.3f} "
      f"-> {np.abs(rho_fixed.values[use] - truth_rho).max():.4f} after the correction; "
      f"of ZDR {np.abs(measured[use, 1] - truth_zdr).max():.2f} dB "
      f"-> {np.abs(zdr_fixed.values[use] - truth_zdr).max():.3f} dB")
```

The corrections remove the bias down to the statistical error of the simulation
at every SNR above 0 dB; below that, the correction is larger than the signal and
radarx returns NaN (`min_snr`).

### On a real sweep

The moments of NEXRAD level 2 are already corrected for noise: the mean
$\rho_{hv}$ of rain gates at an SNR of 0 to 5 dB is 0.98, where an uncorrected value
would be about 0.5 (see the simulation above). A second correction therefore
over-corrects, as the table shows: $\rho_{hv}$ goes up to about 1 and is limited
to 1. The correction is for moments that still hold the noise.

```{code-cell} ipython3
rho_klbb = correct_rhohv_noise(sweep, snr=snr_sweep)
zdr_klbb = correct_zdr_noise(sweep, snr=snr_sweep)
bins = np.arange(0, 41, 5)
rows = []
for lo, hi in zip(bins[:-1], bins[1:]):
    sel = (snr_sweep >= lo) & (snr_sweep < hi) & (sweep.DBZH > 5)
    rows.append([lo, hi, int(sel.sum()), float(sweep.RHOHV.where(sel).mean()),
                 float(rho_klbb.where(sel).mean()), float(sweep.ZDR.where(sel).median()),
                 float(zdr_klbb.where(sel).median())])
print("SNR (dB)   gates  RHOHV  corrected  ZDR  corrected")
for lo, hi, n, r0, r1, z0, z1 in rows:
    print(f"{lo:3d}-{hi:<3d} {n:8d}  {r0:.3f}  {r1:.3f}      {z0:5.2f}  {z1:5.2f}")
```

## ZDR calibration

`zdr_offset` takes the gates that are known to have an intrinsic $Z_{DR}$ and
reports the median (or mean) of $Z_{DR}$ minus that reference, with its spread,
standard error and number of gates. The three methods of Ryzhkov et al.
(2005) differ in the target:

- `"vertical"`: pointing at the zenith, the drops fall with their symmetry axis
  along the beam, so $Z_{DR}=0$ for any size;
- `"light_rain"`: the drops are small and almost spherical, so $Z_{DR}$ is a
  small number that depends on $Z$ (radarx takes that of a Marshall-Palmer
  distribution from its T-matrix tables, about 0.34 dB at 20 dBZ at S band);
- `"dry_snow"`: aggregates have an intrinsic $Z_{DR}$ close to 0 (0.2 dB here),
  and the gates are selected with `mask`, for example above the melting layer.

Away from the zenith the reference changes: for a spheroid with its axis
vertical, the vertical polarization sees a mix of the two axes, and the
$Z_{DR}$ of the target falls with elevation to 0 at the zenith. radarx derives
that dependence in the Rayleigh limit and applies it with the `elevation` of
the data.

To test them we make rain with a known offset of +0.30 dB: Marshall-Palmer
drop size distributions with a random slope, their $Z$ and $Z_{DR}$ from the
scattering tables, seen at three elevations, with 0.15 dB of noise.

```{code-cell} ipython3
def marshall_palmer_rain(offset, elevation, nray=120, ngate=400, seed=5):
    g = np.random.default_rng(seed)
    n = nray * ngate
    lam = np.exp(g.uniform(np.log(2.0), np.log(10.0), n))
    zh, zv, _ = dsd_module._gamma_integrals("S", 20.0, 8000.0, 0.0, lam)
    z, zdr0 = 10 * np.log10(zh), 10 * np.log10(zh / zv)
    theta = np.deg2rad(elevation)  # what a beam at this elevation sees
    q = 10 ** (0.05 * zdr0)
    zdr = 20 * np.log10(q / (q * np.sin(theta) ** 2 + np.cos(theta) ** 2))
    shape = (nray, ngate)
    return xr.Dataset(
        {
            "DBZH": (("azimuth", "range"), z.reshape(shape)),
            "ZDR": (("azimuth", "range"), (zdr + offset + g.normal(0, 0.15, n)).reshape(shape)),
            "RHOHV": (("azimuth", "range"), np.full(shape, 0.995)),
            "elevation": ("azimuth", np.full(nray, float(elevation))),
        },
        coords={"azimuth": np.arange(nray) * 3.0, "range": 2000.0 + 250.0 * np.arange(ngate)},
    )


print("elevation  offset (truth 0.30)  standard error  gates")
elevations = [0.5, 20.0, 60.0]
estimates = []
for elevation in elevations:
    rain = marshall_palmer_rain(0.30, elevation)
    out = zdr_offset(rain, "light_rain", z_range=(10, 40), band="S")
    estimates.append(float(out.ZDR_OFFSET))
    print(f"{elevation:7.1f}°  {float(out.ZDR_OFFSET):17.3f}  {float(out.ZDR_OFFSET_SE):14.4f}  "
          f"{int(out.ZDR_OFFSET_COUNT)}")
rain60 = marshall_palmer_rain(0.0, 60.0)
wrong = zdr_offset(rain60, "light_rain", z_range=(10, 40), band="S", elevation=0.0)
print(f"at 60° without the elevation dependence the offset would be {float(wrong.ZDR_OFFSET):.2f} dB")
```

The dry-snow and vertical-pointing methods use the same selection and statistics.
A quasi-vertical profile with a time dimension gives one offset per time. Here
the $Z_{DR}$ of snow above 4 km drifts from +0.1 to +0.6 dB over six profiles:

```{code-cell} ipython3
nt, nh = 6, 400
drift = np.linspace(0.1, 0.6, nt)
height = np.linspace(2000, 7000, nh)
qvp = xr.Dataset(
    {
        "ZDR": (("time", "height"), 0.2 + drift[:, None] + rng.normal(0, 0.1, (nt, nh))),
        "DBZH": (("time", "height"), np.full((nt, nh), 12.0)),
        "RHOHV": (("time", "height"), np.full((nt, nh), 0.995)),
    },
    coords={"time": np.arange(nt), "height": height},
)
snow = zdr_offset(qvp, "dry_snow", mask=qvp.height > 4000.0)
print("true drift :", drift.round(2))
print("dry snow   :", snow.ZDR_OFFSET.values.round(2))
```

On the KLBB sweeps the light-rain method (20 to 25 dBZ, $\rho_{hv}\ge0.98$) gives
offsets within 0.2 dB of zero, as expected of a WSR-88D, with a standard error of a
few thousandths of a dB; the spread of the gates, 0.6 to 0.7 dB, is the
natural variability of $Z_{DR}$ in light rain around the reference.

```{code-cell} ipython3
klbb_zdr = zdr_offset(klbb, "light_rain")
for name in ("sweep_0", "sweep_2"):
    off = klbb_zdr[name]
    print(f"{name}: offset {float(off.ZDR_OFFSET):+.3f} dB, spread {float(off.ZDR_OFFSET_STD):.2f} dB, "
          f"standard error {float(off.ZDR_OFFSET_SE):.4f} dB, {int(off.ZDR_OFFSET_COUNT)} gates")

reference = np.interp(sweep.DBZH, *calibration_module._light_rain_curve("S", 20.0))
sel = (sweep.DBZH >= 20) & (sweep.DBZH <= 25) & (sweep.RHOHV >= 0.98)
dev = (sweep.ZDR - reference).where(sel).values.ravel()
dev = dev[np.isfinite(dev)]
fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
ax.hist(dev, bins=np.arange(-1.5, 1.51, 0.05), color="0.6")
ax.axvline(float(klbb_zdr["sweep_0"].ZDR_OFFSET), color="C3", label="median = offset")
ax.axvline(0, color="k", lw=1)
ax.set(xlabel="$Z_{DR}$ minus the light-rain reference (dB)", ylabel="gates")
ax.legend(frameon=False)
plt.show()
```

## Reflectivity calibration from self-consistency

In rain, $Z$, $Z_{DR}$ and $K_{DP}$ are tied by the shape of the drops, and
$K_{DP}$ does not depend on the radar calibration. `z_offset` averages the gates
in 1 dB classes of $Z$ (Ryzhkov et al. 2005; the averaging lowers the noise of
$K_{DP}$ enough for light rain), computes from the class means of $Z_{DR}$ and
$K_{DP}$ the reflectivity that rain must have, and reports the median over the
classes of the measured minus that reflectivity. The relation is computed from the
radarx T-matrix tables for the constrained gamma distributions of `dsd`, not
taken from the papers.

The test rain below has gamma drop size distributions with random $N_0$ and
slope, a reflectivity bias of −1.7 dB and a $Z_{DR}$ bias of +0.35 dB, with
noise on all three fields (1 dB on $Z$, 0.2 dB on $Z_{DR}$, 0.3 °/km on $K_{DP}$).

```{code-cell} ipython3
def consistent_rain(dz=-1.7, dzdr=0.35, nray=180, ngate=500, seed=3):
    g = np.random.default_rng(seed)
    n = nray * ngate
    lam = np.clip(np.exp(g.normal(np.log(3.5), 0.35, n)), 1.2, 12.0)
    mu = np.maximum(-0.0201 * lam**2 + 0.902 * lam - 1.718, -0.95)  # Cao et al. (2008)
    n0 = 10 ** g.normal(3.3, 0.5, n)
    zh, zv, kdp = dsd_module._gamma_integrals("S", 20.0, n0, mu, lam)
    shape = (nray, ngate)
    fields = {
        "DBZH": 10 * np.log10(zh) + dz + g.normal(0, 1.0, n),
        "ZDR": 10 * np.log10(zh / zv) + dzdr + g.normal(0, 0.2, n),
        "KDP": kdp + g.normal(0, 0.3, n),
        "RHOHV": np.full(n, 0.99),
    }
    return xr.Dataset(
        {k: (("azimuth", "range"), v.reshape(shape)) for k, v in fields.items()},
        coords={"azimuth": np.arange(nray) * 2.0, "range": 2000.0 + 250.0 * np.arange(ngate)},
    )


consistent = consistent_rain()
fixed = z_offset(consistent, zdr_offset=0.35, band="S")
unfixed = z_offset(consistent, band="S")
print(f"truth -1.70 dB | with the ZDR offset removed {float(fixed.Z_OFFSET):+.2f} dB "
      f"(spread of the classes {float(fixed.Z_OFFSET_STD):.2f}, {int(fixed.Z_OFFSET_BINS)} classes) | "
      f"ZDR offset ignored {float(unfixed.Z_OFFSET):+.2f} dB")
```

The two offsets cannot be separated by self-consistency alone: a $Z_{DR}$ bias that
is left in shifts the result (here by −0.65 dB for 0.35 dB of $Z_{DR}$), so
calibrate $Z_{DR}$ first and pass its offset. The classes themselves show the
consistency:

```{code-cell} ipython3
gz, glg = calibration_module._consistency_curve("S", 20.0, (-0.0201, 0.902, -1.718))
z = consistent.DBZH.values.ravel()
zdr = consistent.ZDR.values.ravel() - 0.35
kdp = consistent.KDP.values.ravel()
edges = np.arange(20.0, 46.0, 1.0)
centre, expected, mean_z = [], [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = (z >= lo) & (z < hi)
    mk, md, mz = kdp[sel].mean(), zdr[sel].mean(), z[sel].mean()
    if mk > 0.05:
        centre.append(0.5 * (lo + hi))
        mean_z.append(mz)
        expected.append(10 * (np.log10(mk) - np.interp(md, gz, glg)))
fig, ax = plt.subplots(figsize=(5.2, 4.4), layout="constrained")
ax.plot([20, 50], [20, 50], "k", lw=1, label="1:1")
ax.plot(expected, mean_z, "o", label="class means")
ax.plot([20, 50], [18.3, 48.3], "-", color="C3", lw=0.8, label="1:1 shifted by the true bias")
ax.set(xlabel="$Z$ expected from $K_{DP}$ and $Z_{DR}$ (dBZ)", ylabel="measured $Z$ (dBZ)")
ax.legend(frameon=False)
plt.show()
```

On KLBB, with the $K_{DP}$ of `estimate_kdp`, the rain gates (reflectivity above
0 dBZ) and the $Z_{DR}$ offset of the light-rain method:

```{code-cell} ipython3
kdp_klbb = estimate_kdp(sweep)["KDP"]
with_kdp = sweep.assign(KDP=kdp_klbb)
zoff = z_offset(
    with_kdp,
    mask=with_kdp.DBZH > 0,
    zdr_offset=float(klbb_zdr["sweep_0"].ZDR_OFFSET),
)
print(f"Z offset {float(zoff.Z_OFFSET):+.2f} dB (spread of the classes {float(zoff.Z_OFFSET_STD):.2f} dB, "
      f"{int(zoff.Z_OFFSET_BINS)} classes, {int(zoff.Z_OFFSET_COUNT)} gates)")
```

The classes scatter by about 0.7 dB, so the estimate is good to a few tenths of a
dB when many classes are available. The attenuation of $Z$ is not removed here; at
S band it is small.

## System differential phase

The measured $\Phi_{DP}$ contains the phase difference of the receivers. Near the radar
the propagation phase is still zero, so `phidp_offset` takes, for each ray, the
circular mean of the first valid gates beyond `min_range` and combines the rays with a
circular median. The synthetic sweep below has a system phase of 355° (the angle wraps
at 0/360), noise of 2° and a propagation phase from gate 120 on.

```{code-cell} ipython3
def phidp_sweep(system, nray=120, ngate=500, seed=7):
    g = np.random.default_rng(seed)
    path = np.zeros(ngate)
    path[120:] = 0.3 * np.arange(ngate - 120)
    phi = np.mod(system + path[None, :] + g.normal(0, 2.0, (nray, ngate)), 360.0)
    return xr.Dataset(
        {"PHIDP": (("azimuth", "range"), phi), "RHOHV": (("azimuth", "range"), np.full((nray, ngate), 0.98))},
        coords={"azimuth": np.arange(nray) * 3.0, "range": 2000.0 + 250.0 * np.arange(ngate)},
    )


demo = phidp_sweep(355.0)
res = phidp_offset(demo)
print(f"truth 355° = -5° | sweep {float(res.SYSTEM_PHIDP):+.2f}° (rays scatter {float(res.SYSTEM_PHIDP_SPREAD):.2f}°, "
      f"{int(res.SYSTEM_PHIDP_RAYS)} rays)")
print("on KLBB:")
klbb_phi = phidp_offset(klbb)
for name in ("sweep_0", "sweep_2"):
    off = klbb_phi[name]
    print(f"  {name}: {float(off.SYSTEM_PHIDP):.1f}° (spread {float(off.SYSTEM_PHIDP_SPREAD):.1f}°, "
          f"{int(off.SYSTEM_PHIDP_RAYS)} of {off.sizes['azimuth']} rays)")

ray = klbb_phi["sweep_0"].SYSTEM_PHIDP_RAY
fig, ax = plt.subplots(figsize=(8, 3.4), layout="constrained")
ax.plot(ray.azimuth, ray, ".", ms=3)
ax.axhline(float(klbb_phi["sweep_0"].SYSTEM_PHIDP), color="C3", label="sweep value")
ax.set(xlabel="azimuth (°)", ylabel="system $\\Phi_{DP}$ (°)")
ax.legend(frameon=False)
plt.show()
```

## Accessors, volumes and provenance

Every function is also a method of the `.radarx` accessor of the xarray objects, and
works on a whole volume: a `DataTree` gives a `DataTree` with one node per sweep
that has the fields.

```{code-cell} ipython3
by_accessor = klbb.radarx.zdr_offset()
print("accessor equals function:",
      float(by_accessor["sweep_0"].ZDR_OFFSET) == float(klbb_zdr["sweep_0"].ZDR_OFFSET))
noise_by_accessor = klbb.radarx.noise_level(method="floor")
snr_by_accessor = sweep.DBZH.radarx.snr(noise=float(levels["sweep_0"].NOISE_LEVEL))
phi_by_accessor = sweep.radarx.phidp_offset()
z_by_accessor = with_kdp.radarx.z_offset(mask=with_kdp.DBZH > 0, band="S")
rho_by_accessor = sweep.radarx.correct_rhohv_noise(snr=snr_sweep)
zdr_by_accessor = sweep.radarx.correct_zdr_noise(snr=snr_sweep)
print(klbb_zdr["sweep_0"].attrs["radarx_method"])
print(klbb_zdr["sweep_0"].attrs["radarx_references"])
for line in radarx.cite(klbb_zdr, style="text"):
    print(line)
```

## Speed

The kernels run on all cores. The table compares them with the NumPy
implementation on the full 720 x 1832 gate sweep of KLBB (the noise level in its
Hildebrand-Sekhon form, which sorts a million gates, and the per-ray system phase).

```{code-cell} ipython3
def best_of(func, repeat=3):
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return min(times)


snr_arr = snr_sweep
cases = {
    "noise_level (sweep)": lambda e: noise_level(sweep, engine=e),
    "noise_level (by ray)": lambda e: noise_level(sweep, by="ray", min_count=20, engine=e),
    "snr": lambda e: snr(sweep, noise=-41.0, engine=e).load(),
    "correct_rhohv_noise": lambda e: correct_rhohv_noise(sweep, snr=snr_arr, engine=e).load(),
    "correct_zdr_noise": lambda e: correct_zdr_noise(sweep, snr=snr_arr, engine=e).load(),
    "zdr_offset": lambda e: zdr_offset(sweep, band="S", engine=e),
    "z_offset": lambda e: z_offset(with_kdp, mask=with_kdp.DBZH > 0, band="S", engine=e),
    "phidp_offset": lambda e: phidp_offset(sweep, engine=e),
}
print(f"{'function':24s} {'numpy (ms)':>10s} {'compiled (ms)':>14s} {'speed-up':>9s}")
for label, call in cases.items():
    t_np = best_of(lambda: call("numpy"))
    t_cc = best_of(lambda: call("compiled")) if noise_module.HAS_COMPILED_KERNEL else np.nan
    print(f"{label:24s} {1e3 * t_np:10.1f} {1e3 * t_cc:14.1f} {t_np / t_cc:9.1f}")
```

## References

- Gorgucci, E., G. Scarchilli, and V. Chandrasekar, 1992: Calibration of radars
  using polarimetric techniques. *IEEE Trans. Geosci. Remote Sens.*, **30**,
  853-858, <https://doi.org/10.1109/36.175319>
- Gourley, J. J., P. Tabary, and J. Parent du Chatelet, 2006: Data quality of the
  Meteo-France C-band polarimetric radar. *J. Atmos. Oceanic Technol.*, **23**,
  1340-1356, <https://doi.org/10.1175/JTECH1912.1>
- Gourley, J. J., A. J. Illingworth, and P. Tabary, 2009: Absolute calibration of
  radar reflectivity using redundancy of the polarization observations and implied
  constraints on drop shapes. *J. Atmos. Oceanic Technol.*, **26**, 689-703,
  <https://doi.org/10.1175/2008JTECHA1152.1>
- Hildebrand, P. H., and R. S. Sekhon, 1974: Objective determination of the noise
  level in Doppler spectra. *J. Appl. Meteor.*, **13**, 808-811,
  <https://doi.org/10.1175/1520-0450(1974)013<0808:ODOTNL>2.0.CO;2>
- Kollias, P., and Coauthors, 2014: Scanning ARM cloud radars. Part II: Data
  quality control and processing. *J. Atmos. Oceanic Technol.*, **31**, 583-598,
  <https://doi.org/10.1175/JTECH-D-13-00045.1>
- Ryzhkov, A. V., S. E. Giangrande, V. M. Melnikov, and T. J. Schuur, 2005:
  Calibration issues of dual-polarization radar measurements. *J. Atmos. Oceanic
  Technol.*, **22**, 1138-1155, <https://doi.org/10.1175/JTECH1772.1>
- Marshall, J. S., and W. McK. Palmer, 1948: The distribution of raindrops with
  size. *J. Meteor.*, **5**, 165-166,
  <https://doi.org/10.1175/1520-0469(1948)005<0165:TDORWS>2.0.CO;2>
- Cao, Q., G. Zhang, E. Brandes, T. Schuur, A. Ryzhkov, and K. Ikeda, 2008:
  Analysis of video disdrometer and polarimetric radar data to characterize rain
  microphysics in Oklahoma. *J. Appl. Meteor. Climatol.*, **47**, 2238-2255,
  <https://doi.org/10.1175/2008JAMC1732.1>
