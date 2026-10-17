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

# Level I processing: IQ time series, clutter filtering and Doppler spectra

Level I data are the complex voltage samples (I and Q) that the receiver records
for every gate and every pulse of a dwell. Everything else a radar shows, the
power, the velocity, the spectrum width, is estimated from them, and the way it
is estimated decides what can be kept (the weather under ground clutter) and what
is lost (a velocity beyond the Nyquist interval). `radarx.level1` works on such
time series with xarray objects:

- `read_iq`, `as_iq`: IQ arrays from raw binary, NumPy, NetCDF and Zarr files or
  from I and Q arrays;
- `simulate_iq`: time series of gates with a known mean velocity, spectrum width,
  power, noise and clutter (after
  [Zrnic 1975](https://doi.org/10.1175/1520-0450(1975)014<0619:SOWDSA>2.0.CO;2)),
  the truth for every check below;
- `regression_filter` (polynomial regression filter of
  [Torres and Zrnic 1999](https://doi.org/10.1175/1520-0426(1999)016<1364:GCCWAR>2.0.CO;2))
  and `notch_filter` (a notch at zero velocity in the spectrum): ground clutter;
- `doppler_spectra`: the windowed periodogram on a velocity axis;
- `spectral_noise` and `spectral_moments`: the noise level of every spectrum
  ([Hildebrand and Sekhon 1974](https://doi.org/10.1175/1520-0450(1974)013<0808:ODOTNL>2.0.CO;2))
  and the power, velocity, width, skewness and kurtosis of the signal;
- `pulse_pair_moments`: the same moments from the autocorrelation of the series;
- `dealias_spectra`: the velocity unfolded with the spectra.

The functions take a `DataArray`, a `Dataset` (a sweep) or a `DataTree` (a volume,
sweep by sweep), keep dask arrays lazy, never change their input, and run in
compiled, multithreaded kernels with a NumPy implementation that gives the same
numbers. The convention is that of the radar books: positive radial velocity is
away from the radar, $v = -\lambda f_d/2$, and the Nyquist velocity is
$v_a = \lambda/(4 T_s)$ for the pulse repetition time $T_s$.

```{code-cell} ipython3
import tempfile
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import radarx  # noqa: F401
from radarx import level1

warnings.filterwarnings("ignore", category=RuntimeWarning)
WAVELENGTH, PRT = 0.1, 1e-3  # S band, 1 kHz: Nyquist velocity 25 m/s
NYQ = WAVELENGTH / (4 * PRT)
```

## A dwell with a known answer

`simulate_iq` makes one complex series per gate whose power spectrum is a Gaussian
of the requested mean velocity and width, plus white noise. Five gates with
velocities from -20 to 20 m/s, a width of 2 m/s, a signal 20 dB above the noise:

```{code-cell} ipython3
truth = np.array([-20.0, -10.0, 0.0, 10.0, 20.0])
iq = level1.simulate_iq(
    truth, width=2.0, power=100.0, noise=1.0, n_pulses=128,
    wavelength=WAVELENGTH, prt=PRT, dims=("range",), seed=1,
)
print(iq.dims, iq.shape, iq.dtype, iq.attrs["wavelength"], iq.attrs["prt"])
```

A real file is read with `read_iq`. Here the series are written as 16 bit
integers, interleaved I and Q as many research radars store them, and read back with
the scale of the converter:

```{code-cell} ipython3
by_gate = iq.transpose("range", "pulse")
raw = np.stack([by_gate.values.real, by_gate.values.imag], axis=-1) * 400
tmp = Path(tempfile.mkdtemp()) / "dwell.raw"
tmp.write_bytes(raw.astype("<i2").tobytes())
again = level1.read_iq(
    tmp, shape=by_gate.shape, dims=("range", "pulse"),
    scale=1 / 400, wavelength=WAVELENGTH, prt=PRT,
)
print(f"largest difference after the 16 bit conversion: "
      f"{float(np.abs(again - by_gate).max()):.1e} "
      f"(a sample is up to {float(np.abs(iq).max()):.0f})")
```

## Spectra and moments

`doppler_spectra` forms the periodogram of the pulses of every gate with a window
(Hann by default; the Blackman window has lower sidelobes) on the velocity axis,
normalised so that the bins add up to the mean power. `spectral_moments` finds the
noise level of each spectrum, takes the contiguous bins above it around the
strongest one, subtracts the noise and returns the moments. `pulse_pair_moments`
estimates them from the lag-1 autocorrelation without a spectrum.

```{code-cell} ipython3
spectra = level1.doppler_spectra(iq)
moments = level1.spectral_moments(spectra)
pp = level1.pulse_pair_moments(iq, noise=1.0)
for i, v in enumerate(truth):
    print(f"truth {v:6.1f}  spectral {float(moments.VRADH[i]):6.2f}  pulse pair "
          f"{float(pp.VRADH[i]):6.2f}   width {float(moments.WRADH[i]):.2f} / "
          f"{float(pp.WRADH[i]):.2f}   SNR {float(moments.SNR[i]):.1f} / {float(pp.SNR[i]):.1f} dB")

fig, axes = plt.subplots(1, 5, figsize=(14, 2.8), sharey=True, layout="constrained")
for i, ax in enumerate(axes):
    s = spectra.SPECTRUM.isel(range=i)
    ax.semilogy(spectra.velocity, s, drawstyle="steps-mid", color="0.3")
    ax.axvline(truth[i], color="C3", lw=1, label="truth")
    ax.axvline(float(moments.VRADH[i]), color="C0", ls="--", lw=1, label="spectral mean")
    ax.axhline(10 ** (0.1 * float(moments.NOISE_POWER[i])) / s.sizes["velocity"],
               color="C2", lw=1, label="noise level")
    ax.set(xlabel="velocity (m/s)", ylim=(2e-4, 40), title=f"truth {truth[i]:.0f} m/s")
axes[0].set(ylabel="power per bin")
axes[0].legend(frameon=False, fontsize=7, loc="upper left")
plt.show()
```

The spectrum of a gate at $+20$ m/s is not cut at the edge of the axis. The moments
are summed around the strongest bin (Doviak and Zrnic 1993, Eq. 6.24), so a spectrum
that straddles the Nyquist velocity gives its true mean and width, and a velocity beyond
$\pm v_a$ is returned folded into the interval. How well do the estimators do?
Doviak and Zrnic derive the variance of the mean velocity for a Gaussian spectrum (their
Eq. 6.22a for pulse pairs and 6.26 for the spectral mean); 6000 simulated gates give
the same standard deviation to within 2 %:

```{code-cell} ipython3
truth_many = np.random.default_rng(5).uniform(-8, 8, 6000)
many = level1.simulate_iq(
    truth_many, width=2.0, power=100.0, noise=1.0, n_pulses=128,
    wavelength=WAVELENGTH, prt=PRT, dims=("range",), seed=3,
)
m_pp = level1.pulse_pair_moments(many, noise=1.0)
m_sp = level1.spectral_moments(level1.doppler_spectra(many, window="rect"), region="all", n_sigma=3.0)

sn, ns, M = 2.0 / (2 * NYQ), 10 ** (-2.0), 128
rho = np.exp(-2 * np.pi**2 * sn**2)
v622 = WAVELENGTH**2 / (32 * np.pi**2 * M * rho**2 * PRT**2) * (
    (1 - rho**2) / (2 * sn * np.sqrt(np.pi)) + ns**2 + 2 * ns * (1 - rho**4))
v626 = WAVELENGTH**2 / (4 * M * PRT**2) * (sn / (4 * np.sqrt(np.pi)) + 2 * sn**2 * ns + ns**2 / 12)
print(f"pulse pair: sd {np.std(m_pp.VRADH.values - truth_many):.3f} m/s, Eq. 6.22a {np.sqrt(v622):.3f}")
print(f"spectral:   sd {np.nanstd(m_sp.VRADH.values - truth_many):.3f} m/s, Eq. 6.26  {np.sqrt(v626):.3f}")
print(f"width: pulse pair {np.mean(m_pp.WRADH.values):.3f}, "
      f"spectral {np.nanmean(level1.spectral_moments(level1.doppler_spectra(many)).WRADH.values):.3f} (truth 2)")
```

A single periodogram has the variance of an exponential distribution in every bin, so
at low SNR the spectral estimates are noisier than the pulse pair. Averaging
`segments` periodograms of a longer dwell lowers it by the number of segments
and tells `spectral_moments` how many independent estimates the noise level
rests on. The error of the spectral velocity at an SNR of 10 dB:

```{code-cell} ipython3
truth_10 = np.random.default_rng(7).uniform(-10, 10, 4000)
rows = []
for segments, n_pulses in ((1, 64), (2, 128), (4, 256)):
    x = level1.simulate_iq(truth_10, width=2.0, power=10.0, noise=1.0, n_pulses=n_pulses,
                           wavelength=WAVELENGTH, prt=PRT, dims=("range",), seed=2)
    m = level1.spectral_moments(level1.doppler_spectra(x, segments=segments))
    rows.append((n_pulses, segments, np.nanstd(m.VRADH.values - truth_10), np.nanmean(m.WRADH.values)))
for n_pulses, segments, sd, w in rows:
    print(f"{n_pulses:4d} pulses, {segments} segment(s): sd of velocity {sd:.2f} m/s, width {w:.2f} m/s (truth 2)")
```

## Ground clutter

Ground clutter has zero velocity and a width below 1 m/s, and is 30 to 50 dB stronger
than the weather. Gates with weather of 100 (20 dB over the noise) and velocities from
-12 to 12 m/s get clutter 30 dB stronger than the weather.

```{code-cell} ipython3
n = 600
truth_c = np.linspace(-12, 12, n)
dirty = level1.simulate_iq(
    truth_c, width=1.5, power=100.0, noise=1.0, clutter_power=1e5, clutter_width=0.2,
    n_pulses=128, wavelength=WAVELENGTH, prt=PRT, dims=("range",), seed=2,
)
far = np.abs(truth_c) > 6.0  # weather clear of the clutter notch

def error(velocity):
    return float(np.nanmean(np.abs(velocity.values[far] - truth_c[far])))

print(f"unfiltered pulse pair: mean |error| {error(level1.pulse_pair_moments(dirty, noise=1.0).VRADH):.2f} m/s")
for order in (0, 2, 4, 6):
    clean = level1.regression_filter(dirty, order=order)
    print(f"regression filter, degree {order}: "
          f"{error(level1.pulse_pair_moments(clean['IQ'], noise=1.0).VRADH):.2f} m/s")
```

The regression filter fits a polynomial to the samples of each gate and subtracts it, a
high-pass filter with a null at zero velocity whose stop band widens with the degree. The
clutter varies over the dwell (its width of 0.2 m/s is half a spectral bin of the 128
pulses), so a degree of 6 is needed here; a dwell of 64 pulses needs only 3.
A spectral notch does the same job in the spectrum, and fills the removed bins so
that the moments are not biased; use a window with low sidelobes (Blackman) so
that the clutter does not leak into the bins that are kept.

```{code-cell} ipython3
spec_dirty = level1.doppler_spectra(dirty, window="blackman")
fig, axes = plt.subplots(1, 3, figsize=(13, 3.3), layout="constrained")
g = 500  # a gate with weather at about +8 m/s
axes[0].semilogy(spec_dirty.velocity, spec_dirty.SPECTRUM.isel(range=g), drawstyle="steps-mid", label="with clutter")
for fill, color in (("interpolate", "C1"), ("gaussian", "C2")):
    out = level1.notch_filter(spec_dirty, half_width=3, fill=fill)
    axes[0].semilogy(out.velocity, out.SPECTRUM.isel(range=g), drawstyle="steps-mid", color=color, label=f"notch, {fill}")
    m = level1.spectral_moments(out)
    axes[1].plot(truth_c, m.VRADH - truth_c, ".", ms=2, color=color, label=f"notch, {fill}")
reg = level1.regression_filter(dirty, order=6)
axes[1].plot(truth_c, level1.pulse_pair_moments(reg["IQ"], noise=1.0).VRADH - truth_c, ".", ms=2, color="C3", label="regression, degree 6")
raw_m = level1.spectral_moments(spec_dirty)
axes[2].plot(truth_c, raw_m.VRADH, ".", ms=2, color="0.5", label="unfiltered")
axes[2].plot(truth_c, level1.spectral_moments(level1.notch_filter(spec_dirty, half_width=3)).VRADH, ".", ms=2, color="C1", label="notch")
axes[2].plot(truth_c, truth_c, "k", lw=0.7)
axes[0].set(xlabel="velocity (m/s)", ylabel="power per bin", title=f"spectrum of gate {g}")
axes[1].set(xlabel="true velocity (m/s)", ylabel="error (m/s)", ylim=(-6, 6), title="error of the velocity")
axes[2].set(xlabel="true velocity (m/s)", ylabel="velocity (m/s)", title="spectral velocity")
for ax in axes:
    ax.legend(frameon=False, fontsize=7)
plt.show()
```

Both filters remove weather as well: the echo with a velocity within a few bins of zero is
lost or biased, which is why the error rises near 0 m/s in the figure. `where=` filters only
the gates of a clutter map and leaves the others untouched.

## Dealiasing with the spectra

A velocity from one pulse repetition time is known modulo $2v_a$. A ray of 120 gates whose
velocity grows from 0 to 96 m/s (four Nyquist intervals) is measured at 25 m/s:

```{code-cell} ipython3
ramp = np.linspace(0, 96, 120)
ray = level1.simulate_iq(
    ramp[None, :], width=2.0, power=300.0, noise=1.0, n_pulses=64,
    wavelength=WAVELENGTH, prt=PRT, dims=("azimuth", "range"), seed=5,
)
spec_ray = level1.doppler_spectra(ray)
measured = level1.spectral_moments(spec_ray).VRADH
unfolded = level1.dealias_spectra(spec_ray)
within = np.mean(np.abs(unfolded.VRADH.values[0] - ramp) < 4)
print(f"measured velocities lie in [{float(measured.min()):.1f}, {float(measured.max()):.1f}] m/s; "
      f"unfolded within 4 m/s of the truth at {100 * within:.0f} % of the gates, "
      f"folds {int(unfolded.FOLD.min())} to {int(unfolded.FOLD.max())}")
fig, ax = plt.subplots(figsize=(6, 3.3), layout="constrained")
ax.plot(ramp, measured.values[0], ".", color="0.5", label="from the spectrum (folded)")
ax.plot(ramp, unfolded.VRADH.values[0], ".", color="C0", label="dealias_spectra")
ax.plot(ramp, ramp, "k", lw=0.7, label="truth")
ax.set(xlabel="true velocity (m/s)", ylabel="velocity (m/s)")
ax.legend(frameon=False)
plt.show()
```

The mean velocity of the peak, summed around the strongest bin, is not biased by a
spectrum that straddles $\pm v_a$; `dealias_spectra` brings it to the Nyquist interval around
a reference (a model or profile wind, passed as `reference`) or, without one, around the last
unfolded gate of the ray. A wrong first gate is then carried along the ray, and a gate with
an SNR below `min_snr` (3 dB by default) does not take part.

## A real storm

Open radar data has no IQ time series, so the IQ of a real scene is simulated from the Level II
fields of the 1.5° Doppler cut of the NEXRAD radar of Lubbock, Texas (KLBB), of 1 June 2016:
the mean velocity and the width are those the radar measured, and the SNR comes from the
reflectivity and the range. The chain then has to give back the velocity the radar saw.

```{code-cell} ipython3
import xradar as xd
from open_radar_data import DATASETS

klbb = xd.io.open_nexradlevel2_datatree(DATASETS.fetch("KLBB20160601_150025_V06"), sweep=[1])
cut = klbb["sweep_1"].to_dataset().isel(azimuth=slice(100, 164), range=slice(0, 400))
vel, wid, dbz = cut.VRADH, cut.WRADH, cut.DBZH
valid = (vel > -63.9) & (wid >= 0) & (dbz > -20)
snr_true = (dbz + 42.0 - 20 * np.log10(cut.range / 1000.0)).where(valid)
nyq_real = 6.0  # much shorter than the NEXRAD Nyquist velocity: most of this cut is near -7 m/s and folds
prt_real = 0.1071 / (4 * nyq_real)
iq_real = level1.simulate_iq(
    vel.where(valid, 0.0).values, width=wid.where(valid, 1.0).clip(min=0.3).values,
    power=xr.where(valid, 10 ** (0.1 * snr_true.fillna(0)), 0.0).values, noise=1.0, n_pulses=64,
    wavelength=0.1071, prt=prt_real, dims=("azimuth", "range"), seed=1,
)
iq_real = iq_real.assign_coords(azimuth=cut.azimuth, range=cut.range)
spec_real = level1.doppler_spectra(iq_real)
folded = level1.spectral_moments(spec_real, min_snr=3.0).VRADH
# a coarse "model" wind with 2.5 m/s of error as the reference
rng = np.random.default_rng(3)
ref = xr.DataArray(vel.where(valid).values + rng.normal(0, 2.5, vel.shape), dims=("azimuth", "range"))
restored = level1.dealias_spectra(spec_real, reference=ref).VRADH
alone = level1.dealias_spectra(spec_real).VRADH
narrow = valid & (wid < 2.0) & (snr_true > 8)
err = (restored - vel).where(narrow)
print(f"gates with |v| > {nyq_real:.0f} m/s that fold: "
      f"{100 * float((np.abs(vel) > nyq_real).where(narrow).mean()):.0f} %; "
      f"restored within 4 m/s at {100 * float((np.abs(err) < 4).where(narrow).sum() / narrow.sum()):.0f} % "
      f"of the narrow echoes, rms error {float(np.sqrt((err**2).mean())):.2f} m/s")
print(f"continuity along the ray alone: "
      f"{100 * float((np.abs(alone - vel) < 4).where(narrow).sum() / narrow.sum()):.0f} % within 4 m/s")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True, layout="constrained")
panels = (
    (vel.where(narrow), "Level II velocity (truth)"),
    (folded.where(narrow), f"from the spectra, Nyquist {nyq_real:.0f} m/s"),
    (restored.where(narrow), "dealias_spectra"),
)
for ax, (field, title) in zip(axes, panels):
    pm = ax.pcolormesh(cut.range / 1e3, cut.azimuth, field, cmap="RdBu_r", vmin=-15, vmax=15, shading="auto")
    ax.set(xlabel="range (km)", title=title)
axes[0].set_ylabel("azimuth (°)")
fig.colorbar(pm, ax=axes, label="radial velocity (m/s)")
plt.show()
```

The velocity of this cut is nearly uniform (near -7 m/s), so at a Nyquist velocity of 6 m/s
almost every gate folds. Continuity along the ray alone keeps the folded velocity of the
first gate for the whole ray: it fixes the shape of the field up to a whole number of
Nyquist intervals, and the reference decides which one. Spectra as wide as a third of
the Nyquist interval or more cannot be used either, because the noise level is then not
told apart from the signal, so only the narrower echoes are scored. This is a limit of the
method, not of the data.

## Speed

The kernels run in C++ over the series (one gate of one ray each), release the GIL, and
take the series in blocks from an atomic counter, so a sweep of 360 rays and 1000 gates with
64 pulses (23 million samples) takes a second or two for the whole chain on one core. The table gives the
time of each step of the chain on the machine that built this page.

```{code-cell} ipython3
big = level1.simulate_iq(
    np.random.default_rng(0).uniform(-15, 15, (90, 500)), width=2.0, power=100.0, noise=1.0,
    n_pulses=64, wavelength=WAVELENGTH, prt=PRT, seed=1,
)
spec_big = level1.doppler_spectra(big)

def best(func, repeat=3, **kw):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        func(**kw)
        times.append(time.perf_counter() - t0)
    return min(times)

steps = {
    "doppler_spectra": lambda **kw: level1.doppler_spectra(big, **kw),
    "spectral_moments": lambda **kw: level1.spectral_moments(spec_big, **kw),
    "pulse_pair_moments": lambda **kw: level1.pulse_pair_moments(big, noise=1.0, **kw),
    "regression_filter": lambda **kw: level1.regression_filter(big, **kw),
    "notch_filter": lambda **kw: level1.notch_filter(spec_big, **kw),
}
print(f"{big.size / 1e6:.1f} million samples ({big.shape[0]} rays x {big.shape[2]} gates x {big.shape[1]} pulses)")
print(f"{'':20s} {'NumPy':>8s} {'C++ 1 thread':>13s} {'C++ all cores':>14s} {'speedup':>8s}")
for name, func in steps.items():
    t_np = best(func, engine="numpy")
    t_1 = best(func, engine="compiled", n_threads=1)
    t_all = best(func, engine="compiled")
    print(f"{name:20s} {t_np:7.3f}s {t_1:12.3f}s {t_all:13.3f}s {t_np / t_all:7.0f}x")
```

## References

- Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather Observations*,
  2nd ed. Academic Press.
- Guo, H., 2011: A simple algorithm for fitting a Gaussian function [DSP tips and
  tricks]. *IEEE Signal Process. Mag.*, **28**, 134-137,
  <https://doi.org/10.1109/MSP.2011.941846>
- Harris, F., 1978: On the use of windows for harmonic analysis with the discrete
  Fourier transform. *Proc. IEEE*, **66**, 51-83,
  <https://doi.org/10.1109/PROC.1978.10837>
- Hildebrand, P. H., and R. S. Sekhon, 1974: Objective determination of the noise
  level in Doppler spectra. *J. Appl. Meteor.*, **13**, 808-811,
  <https://doi.org/10.1175/1520-0450(1974)013<0808:ODOTNL>2.0.CO;2>
- Torres, S. M., and D. S. Zrnić, 1999: Ground clutter canceling with a regression
  filter. *J. Atmos. Oceanic Technol.*, **16**, 1364-1372,
  <https://doi.org/10.1175/1520-0426(1999)016<1364:GCCWAR>2.0.CO;2>
- Zrnić, D. S., 1975: Simulation of weatherlike Doppler spectra and signals.
  *J. Appl. Meteor.*, **14**, 619-620,
  <https://doi.org/10.1175/1520-0450(1975)014<0619:SOWDSA>2.0.CO;2>
- Zrnić, D. S., 1977: Spectral moment estimates from correlated pulse pairs. *IEEE
  Trans. Aerosp. Electron. Syst.*, **AES-13**, 344-354,
  <https://doi.org/10.1109/TAES.1977.308467>
- Srivastava, R. C., A. R. Jameson, and P. H. Hildebrand, 1979: Time-domain
  computation of mean and variance of Doppler spectra. *J. Appl. Meteor.*, **18**,
  189-194, <https://doi.org/10.1175/1520-0450(1979)018<0189:TDCOMA>2.0.CO;2>
- Bluestein, L., 1970: A linear filtering approach to the computation of discrete
  Fourier transform. *IEEE Trans. Audio Electroacoust.*, **18**, 451-455,
  <https://doi.org/10.1109/TAU.1970.1162132>
