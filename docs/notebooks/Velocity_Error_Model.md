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

# Radial Velocity Error from the Signal to Noise Ratio

+++

A multi-Doppler analysis weighs every radial velocity by the inverse square of
its error (`obs_error`). The noise of a Doppler velocity depends on how much
signal there is: the signal to noise ratio (SNR), the width of the Doppler
spectrum and the number of pulse pairs behind the estimate. The standard
deviation of the pulse-pair estimator follows from these in closed form (Doviak
and Zrnic 1993, Eqs. 6.22a and b, from Zrnic 1977), and
`radarx.retrieve.velocity_error` evaluates it.

This notebook shows the model, maps it for a KGWX sweep and uses it as
`obs_error` in a retrieval where one of two radars gets weaker with range.
Without network access the sweep is synthetic.

**References**

- Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
  Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6.
- Zrnic, D. S., 1977: Spectral moment estimates from correlated pulse pairs.
  *IEEE Trans. Aerosp. Electron. Syst.*, **AES-13**, 344-354,
  https://doi.org/10.1109/TAES.1977.308467

```{code-cell} ipython3
import warnings

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xradar as xd

import radarx as rx

warnings.filterwarnings("ignore", message="The input coordinates to pcolormesh")
```

## The model

At S band (10.7 cm) with a Nyquist velocity of 26.5 m/s and 64 pairs the error
falls from several m/s at an SNR of 0 dB to a floor set by the spectrum width,
and it grows with the width. Independent pairs (`pairs="independent"`) give
a different curve than pairs that share samples (the default, as on
the WSR-88D).

```{code-cell} ipython3
snr_db = np.linspace(-5, 30, 100)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained", sharey=True)
for width, color in ((2.0, "C0"), (4.0, "C1"), (8.0, "C2")):
    for pairs, style in (("contiguous", "-"), ("independent", "--")):
        sd = rx.retrieve.velocity_error(
            snr_db, width, wavelength=0.1071, nyquist=26.5, n_pairs=64, pairs=pairs
        )
        axes[0].plot(snr_db, sd, style, color=color, label=f"{width:g} m/s, {pairs}")
axes[0].set(xlabel="SNR (dB)", ylabel="standard deviation of the velocity (m/s)", yscale="log",
            title="64 pairs", ylim=(0.2, 20))
axes[0].legend(title="spectrum width", fontsize=8)
for pairs_count, color in ((16, "C3"), (64, "C1"), (256, "C4")):
    sd = rx.retrieve.velocity_error(snr_db, 4.0, wavelength=0.1071, nyquist=26.5, n_pairs=pairs_count)
    axes[1].plot(snr_db, sd, color=color, label=f"{pairs_count} pairs")
axes[1].set(xlabel="SNR (dB)", title="4 m/s width, contiguous pairs")
axes[1].legend()
plt.show()
```

## A KGWX sweep

The first Doppler cut of the KGWX volume of 30 March 2022. The SNR comes from
the reflectivity and the noise floor of the sweep (`noise_level`, `snr`), the
spectrum width is the `WRADH` of the volume, and we assume 40 pulse pairs. The
Nyquist velocity is 26.5 m/s here; use the value of the scan.

```{code-cell} ipython3
def real_sweep():
    local_file = fsspec.open_local(
        "simplecache::s3://unidata-nexrad-level2/2022/03/30/KGWX/KGWX20220330_234639_V06",
        s3={"anon": True},
        filecache={"cache_storage": "."},
    )
    tree = xd.io.open_nexradlevel2_datatree(local_file)
    sweep = tree["sweep_1"].to_dataset()[["DBZH", "WRADH"]].sortby("azimuth").load()
    sweep = sweep.where(sweep > -32.0)
    return sweep.where(sweep.range < 150e3, drop=True), rx.retrieve.noise_level(sweep, method="floor")


def synthetic_sweep():
    rng = np.random.default_rng(3)
    ranges = np.arange(250.0, 150e3, 250.0)
    azimuths = np.arange(0.5, 360.0, 1.0)
    dbz = 52.0 - 22.0 * np.log10(ranges / 1e3)[None, :] + rng.normal(0, 3, (azimuths.size, ranges.size))
    dbz = dbz + 12.0 * np.exp(-(((azimuths[:, None] - 120.0) / 25.0) ** 2))
    width = np.clip(rng.normal(3.0, 1.0, dbz.shape), 0.5, None)
    sweep = xr.Dataset(
        {"DBZH": (("azimuth", "range"), dbz), "WRADH": (("azimuth", "range"), width)},
        coords={"azimuth": azimuths, "range": ranges},
    )
    return sweep, -36.0


try:
    sweep, noise = real_sweep()
    source = "KGWX, 0.5 degrees, 30 March 2022 23:46 UTC"
except Exception as error:  # offline
    print(f"no network access ({type(error).__name__}), using a synthetic sweep")
    sweep, noise = synthetic_sweep()
    source = "synthetic sweep"

snr = rx.retrieve.snr(sweep, noise=noise)
sigma = rx.retrieve.velocity_error(snr, sweep.WRADH, wavelength=0.1071, nyquist=26.5, n_pairs=40)
print(f"median error {float(sigma.median()):.2f} m/s, "
      f"5th to 95th percentile {float(sigma.quantile(0.05)):.2f} to {float(sigma.quantile(0.95)):.2f} m/s")
```

```{code-cell} ipython3
az, rng_ = np.radians(sweep.azimuth.values), sweep.range.values
gate_x = rng_[None, :] * np.sin(az)[:, None] / 1e3
gate_y = rng_[None, :] * np.cos(az)[:, None] / 1e3
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained", sharey=True)
panels = (
    (snr, "SNR (dB)", dict(cmap="viridis", vmin=0, vmax=50)),
    (sweep.WRADH, "spectrum width (m/s)", dict(cmap="magma", vmin=0, vmax=8)),
    (sigma, "velocity error (m/s)", dict(cmap="cividis", vmin=0.5, vmax=3)),
)
for ax, (field, label, opts) in zip(axes, panels):
    im = ax.pcolormesh(gate_x, gate_y, field.values, shading="auto", rasterized=True, **opts)
    fig.colorbar(im, ax=ax, label=label, shrink=0.8)
    ax.set(title=label, aspect="equal", xlabel="east (km)", xlim=(-120, 120), ylim=(-120, 120))
axes[0].set_ylabel("north (km)")
fig.suptitle(source)
plt.show()
```

The error rises in the weak echo at the edge of the storm and with the width in
the convective cores, and it stays near 0.5 to 1 m/s where there is a strong,
narrow signal.

## As the observation error of a retrieval

A flow with a known answer, `u = U + a x`, `v = V + a y`, `w = -2 a z`, seen by
two virtual radars. The radar on the left keeps a strong signal, the one on
the right loses 0.5 dB per km of range, and 16 pulse pairs per gate are
averaged. The noise of each gate is drawn with the standard deviation that
`velocity_error` gives. Three retrievals: the default (every gate with an error
of 1 m/s), the model error as `obs_error`, and one constant error equal to the
median of the model, which separates the size of the error from the
differences between the radars. Four noise realisations:

```{code-cell} ipython3
x = y = np.arange(-40e3, 40e3 + 1, 2000.0)
z = np.arange(0.0, 10e3 + 1, 1000.0)
dims = ("z", "y", "x")
Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
a, u0, v0 = 2e-4, 8.0, 4.0
truth = {"u": u0 + a * X, "v": v0 + a * Y, "w": -2 * a * Z}

flow = xr.Dataset(
    {"radar_x": ("radar", [-25e3, 25e3]), "radar_y": ("radar", [-20e3, -20e3]),
     "radar_altitude": ("radar", [0.0, 0.0])},
    coords={"radar": [0, 1], "z": z, "y": y, "x": x},
)
geo = rx.retrieve.radar_geometry(flow)
el, azim = np.radians(geo.elevation.values), np.radians(geo.azimuth.values)
vr = np.cos(el) * (np.sin(azim) * truth["u"] + np.cos(azim) * truth["v"]) + np.sin(el) * truth["w"]
distance = np.hypot(flow.x - flow.radar_x, flow.y - flow.radar_y).transpose("radar", "y", "x").values[:, None] * np.ones_like(vr)

snr_gate = np.empty_like(vr)
snr_gate[0] = 30.0 - 0.1 * distance[0] / 1e3
snr_gate[1] = 30.0 - 0.5 * distance[1] / 1e3
model = rx.retrieve.velocity_error(snr_gate, 4.0, wavelength=0.1071, nyquist=26.5, n_pairs=16)
model = xr.DataArray(model, dims=("radar", "z", "y", "x"), coords=flow.coords)
background = xr.Dataset(
    {"u": (dims, np.full(Z.shape, u0)), "v": (dims, np.full(Z.shape, v0)), "air_density": (dims, np.ones(Z.shape))},
    coords={"z": z, "y": y, "x": x},
)

settings = {"default (1 m/s)": None, "velocity_error": model, "constant, same median": float(np.nanmedian(model))}
scores = {name: [] for name in settings}
for seed in range(4):
    noisy = vr + np.random.default_rng(seed).normal(size=vr.shape) * model.values
    flow["VRADH"] = (("radar", "z", "y", "x"), np.where(distance < 60e3, noisy, np.nan))
    for name, obs_error in settings.items():
        result = rx.retrieve.multi_doppler(flow, background, fall_speed_correction=False, obs_error=obs_error)
        good = (result.beam_crossing_angle > 30).values
        scores[name].append([np.sqrt(np.mean((result[c].values - truth[c])[good] ** 2)) for c in "uvw"])
for name, rms in scores.items():
    print(f"{name:24s} rms error u, v, w: " + ", ".join(f"{v:.2f}" for v in np.mean(rms, axis=0)) + " m/s")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
for radar, color in ((0, "C0"), (1, "C3")):
    near = distance[radar, 3] < 60e3
    axes[0].plot(distance[radar, 3][near][::40] / 1e3, model.values[radar, 3][near][::40], ".", color=color, label=f"radar {radar}")
axes[0].set(xlabel="range (km)", ylabel="error of the radial velocity (m/s)", title="error of the gates at 3 km")
axes[0].legend()
width = 0.27
for k, (name, rms) in enumerate(scores.items()):
    axes[1].bar(np.arange(3) + (k - 1) * width, np.mean(rms, axis=0), width, label=name)
axes[1].set(xticks=range(3), xticklabels=["u", "v", "w"], ylabel="rms error (m/s)", title="retrieved wind")
axes[1].legend()
plt.show()
```

The error of the weaker radar grows from 1.5 m/s to 2.5 m/s at a range of 60 km. Here most of the gain comes from the size of `obs_error`, not from the
difference between the radars: the constant error does about as well. That size
also sets the weight of the gates against the smoothness and mass terms, whose
defaults assume gates of 1 m/s, and noisier data ask for a smoother analysis.
The model is the noise of the estimate of one gate. It does not contain
aliasing, clutter or the interpolation to the grid.
