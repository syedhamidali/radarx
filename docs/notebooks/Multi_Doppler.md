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

# Multi-Doppler Wind Retrieval

+++

A Doppler radar measures only the wind component along its beam. Where two
or more radars look at the same air from different directions, the full
three-dimensional wind can be retrieved. `radarx.retrieve.multi_doppler`
does this with a variational method (Gao et al. 1999): the wind on a
Cartesian grid minimises a cost function made of

- the misfit to every radar's radial velocity, corrected for the fall speed
  of the precipitation (Atlas et al. 1973, with the density correction of
  Foote and du Toit 1969),
- the anelastic mass continuity equation,
- a smoothness penalty,
- the distance to a background wind from a sounding or ERA5,
- optionally the vertical vorticity equation (Shapiro et al. 2009; Potvin et
  al. 2012).

The cost and its exact gradient are computed in one multithreaded pass of a
compiled C++ kernel, and the minimisation (preconditioned conjugate
gradients, first on a coarser grid) takes a few seconds.

The workflow is: dealias each radar (`dealias`), grid all radars onto one
grid and move them to a common time (`multi_doppler_input`, which uses cone
gridding and `advect`), add a background (`radarx.io.sounding`), retrieve.

**References**

- Gao, J., M. Xue, A. Shapiro, and K. K. Droegemeier, 1999: A variational
  method for the analysis of three-dimensional wind fields from two Doppler
  radars. *Mon. Wea. Rev.*, **127**, 2128-2142,
  https://doi.org/10.1175/1520-0493(1999)127<2128:AVMFTA>2.0.CO;2
- Shapiro, A., C. K. Potvin, and J. Gao, 2009: Use of a vertical vorticity
  equation in variational dual-Doppler wind analysis. *J. Atmos. Oceanic
  Technol.*, **26**, 2089-2106, https://doi.org/10.1175/2009JTECHA1256.1
- Potvin, C. K., A. Shapiro, and M. Xue, 2012: Impact of a vertical vorticity
  constraint in variational dual-Doppler wind analysis: Tests with real and
  simulated supercell data. *J. Atmos. Oceanic Technol.*, **29**, 32-49,
  https://doi.org/10.1175/JTECH-D-11-00019.1
- Atlas, D., R. C. Srivastava, and R. S. Sekhon, 1973: Doppler radar
  characteristics of precipitation at vertical incidence. *Rev. Geophys.*,
  **11**, 1-35, https://doi.org/10.1029/RG011i001p00001
- Foote, G. B., and P. S. du Toit, 1969: Terminal velocity of raindrops
  aloft. *J. Appl. Meteor.*, **8**, 249-253,
  https://doi.org/10.1175/1520-0450(1969)008<0249:TVORA>2.0.CO;2
- O'Brien, J. J., 1970: Alternative solutions to the classical vertical
  velocity problem. *J. Appl. Meteor.*, **9**, 197-203,
  https://doi.org/10.1175/1520-0450(1970)009<0197:ASTTCV>2.0.CO;2
- Collis, S., A. Protat, and K.-S. Chung, 2010: The effect of radial velocity
  gridding artifacts on variationally retrieved vertical velocities. *J.
  Atmos. Oceanic Technol.*, **27**, 1239-1246,
  https://doi.org/10.1175/2010JTECHA1402.1
- Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
  Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6.
- Gal-Chen, T., and R. C. J. Somerville, 1975: On the use of a coordinate
  transformation for the solution of the Navier-Stokes equations. *J.
  Comput. Phys.*, **17**, 209-228,
  https://doi.org/10.1016/0021-9991(75)90037-6
- Shapiro, A., 1993: The use of an exact solution of the Navier-Stokes
  equations in a validation test of a three-dimensional nonhydrostatic
  numerical model. *Mon. Wea. Rev.*, **121**, 2420-2425,
  https://doi.org/10.1175/1520-0493(1993)121<2420:TUOAES>2.0.CO;2

```{code-cell} ipython3
import time

import cmweather  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xradar as xd
from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

import radarx as rx
from radarx.io import sounding
from radarx.io.aws_data import download_file
```

## A known flow seen by two virtual radars

First a test with an exact answer. A Beltrami flow (Shapiro 1993) is an
exact three-dimensional solution of the Navier-Stokes equations with
updrafts and downdrafts of 5 m/s; divided by a density profile it satisfies
the anelastic continuity equation exactly. Two virtual radars 50 km apart
sample its radial velocity (with 1 m/s noise) out to 60 km.

```{code-cell} ipython3
def beltrami(x, y, z, wmax=5.0, lx=40e3, lz=10e3, u0=8.0, v0=4.0):
    k = l = 2 * np.pi / lx
    m = np.pi / lz
    lam = np.sqrt(k * k + l * l + m * m)
    kh2 = k * k + l * l
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    fu = -wmax / kh2 * (lam * l * np.cos(k * X) * np.sin(l * Y) * np.sin(m * Z)
                        + m * k * np.sin(k * X) * np.cos(l * Y) * np.cos(m * Z))
    fv = wmax / kh2 * (lam * k * np.sin(k * X) * np.cos(l * Y) * np.sin(m * Z)
                       - m * l * np.cos(k * X) * np.sin(l * Y) * np.cos(m * Z))
    fw = wmax * np.cos(k * X) * np.cos(l * Y) * np.sin(m * Z)
    rho = 1.2 * np.exp(-Z / 10e3)
    scale = 1.2 * np.exp(-z.mean() / 10e3) / rho
    return u0 + fu * scale, v0 + fv * scale, fw * scale, rho


x = y = np.arange(-40e3, 40e3 + 1, 1000.0)
z = np.arange(0.0, 10e3 + 1, 500.0)
u, v, w, rho = beltrami(x, y, z)
dims = ("z", "y", "x")
truth = xr.Dataset({"u": (dims, u), "v": (dims, v), "w": (dims, w)}, coords={"z": z, "y": y, "x": x})

virtual = xr.Dataset(
    {
        "radar_x": ("radar", [-25e3, 25e3]),
        "radar_y": ("radar", [-20e3, -20e3]),
        "radar_altitude": ("radar", [0.0, 0.0]),
    },
    coords={"radar": [0, 1], "z": z, "y": y, "x": x},
)
virtual = rx.retrieve.radar_geometry(virtual)
el, az = np.radians(virtual.elevation), np.radians(virtual.azimuth)
vr = np.cos(el) * (np.sin(az) * truth.u + np.cos(az) * truth.v) + np.sin(el) * truth.w
vr = vr + np.random.default_rng(0).normal(0.0, 1.0, vr.shape)
in_range = np.hypot(virtual.x - virtual.radar_x, virtual.y - virtual.radar_y) < 60e3
virtual["VRADH"] = vr.where(in_range).transpose("radar", "z", "y", "x")
background = xr.Dataset(
    {"u": (dims, np.full(u.shape, 8.0)), "v": (dims, np.full(u.shape, 4.0)), "air_density": (dims, rho)},
    coords={"z": z, "y": y, "x": x},
)

start = time.perf_counter()
wind = rx.retrieve.multi_doppler(virtual, background, fall_speed_correction=False)
print(f"retrieved {wind.u.size:,} grid cells in {time.perf_counter() - start:.1f} s, "
      f"iterations per grid level (coarse first): {wind.attrs['iterations']}")
lobes = wind.beam_crossing_angle > 30
for c in "uvw":
    err = (wind[c] - truth[c]).where(lobes)
    print(f"{c}: RMS error {float(np.sqrt((err ** 2).mean())):.2f} m/s")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
level = dict(z=5000.0)
kw = dict(cmap="RdBu_r", vmin=-6, vmax=6, add_colorbar=False)
truth.w.sel(**level).plot(ax=axes[0], **kw)
wind.w.sel(**level).where(lobes.sel(**level)).plot(ax=axes[1], **kw)
im = (wind.w - truth.w).sel(**level).where(lobes.sel(**level)).plot(ax=axes[2], **kw)
for ax, title in zip(axes, ["true w at 5 km", "retrieved w (beam crossing > 30 deg)", "retrieved - true"]):
    ax.plot(virtual.radar_x, virtual.radar_y, "k^")
    ax.set_title(title)
    ax.set_aspect("equal")
fig.colorbar(im, ax=axes, label="w (m/s)")
plt.show()
```

The cost terms at every iteration show the coarse-grid and fine-grid
minimisations:

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7, 4))
for term in wind.term.values[:4]:
    ax.semilogy(wind.iteration, wind.cost_history.sel(term=term), label=term)
ax.axvline(int((wind.grid_level > 0).sum()), color="k", lw=0.5)
ax.set_xlabel("iteration (coarse grid, then full grid)")
ax.set_ylabel("cost (m$^2$ s$^{-2}$)")
ax.legend()
plt.show()
```

## Observation errors, error estimates, the upper boundary and the analytic solution

The same virtual radars show the options added to `multi_doppler` and the
classical analytic solution. We take every second grid point (2 km
horizontally, 1 km vertically) to keep the error estimate quick.

```{code-cell} ipython3
coarse = virtual.isel(x=slice(None, None, 2), y=slice(None, None, 2), z=slice(None, None, 2))
coarse_bg = background.isel(x=slice(None, None, 2), y=slice(None, None, 2), z=slice(None, None, 2))
coarse_truth = truth.isel(x=slice(None, None, 2), y=slice(None, None, 2), z=slice(None, None, 2))


def rms(wind, mask):
    return {c: round(float(np.sqrt(((wind[c] - coarse_truth[c]).where(mask) ** 2).mean())), 2) for c in "uvw"}


base = rx.retrieve.multi_doppler(coarse, coarse_bg, fall_speed_correction=False)
crossing = base.beam_crossing_angle > 30
print("default             ", rms(base, crossing))
```

**Observation errors.** `obs_error` is the standard deviation of the radial
velocity error in m/s (one number, one per radar, or a field such as one
computed from the signal to noise ratio of the gate); the weight of an
observation in the cost function is divided by its square. The weights of
the other terms are tuned for errors of about 1 m/s, so the option is meant
for the *relative* quality of the radars and gates. Here the first radar has
an error of 0.2 m/s and the second one of 1 m/s:

```{code-cell} ipython3
cel, caz = np.radians(coarse.elevation), np.radians(coarse.azimuth)
clean = np.cos(cel) * (np.sin(caz) * coarse_truth.u + np.cos(caz) * coarse_truth.v) + np.sin(cel) * coarse_truth.w
errors = xr.DataArray([0.2, 1.0], dims="radar")
rng = np.random.default_rng(1)
noise = xr.DataArray(rng.normal(size=clean.shape), dims=clean.dims)
noisy_vr = (clean + errors * noise).where(coarse.VRADH.notnull())
noisy = coarse.assign(VRADH=noisy_vr.transpose("radar", "z", "y", "x"))
flat = rx.retrieve.multi_doppler(noisy, coarse_bg, fall_speed_correction=False)
weighted = rx.retrieve.multi_doppler(noisy, coarse_bg, fall_speed_correction=False, obs_error=errors.values)
print("equal weights       ", rms(flat, crossing))
print("obs_error=[0.2, 1]  ", rms(weighted, crossing))
```

The weights of the other terms are fixed, so a larger error of all radars
(say 3 m/s) lets the smoothness penalty win over the observations; scale
`obs_error` to about 1 m/s unless that is what you want.

`crossing_angle_weight=True` multiplies the weights by the squared sine of
the beam crossing angle. The normal equations already contain the geometry,
so this is an extra down-weighting (a radarx choice, not from a paper).

**Error estimate.** With `error=True` the result has `u_error`, `v_error`
and `w_error`, the standard deviation of the retrieval error from the
covariance of the cost function (the inverse of half its Hessian, which is
constant because the cost is quadratic). The covariance of a million unknowns
cannot be written down, so radarx estimates its diagonal from random draws,
each one a conjugate-gradient solution (`error_samples`, 8 here; use 32 or
more for a result).

```{code-cell} ipython3
est = rx.retrieve.multi_doppler(coarse, coarse_bg, fall_speed_correction=False,
                                obs_error=1.0, error=True, error_samples=8)
actual = rms(est, crossing)
estimate = {c: round(float(np.sqrt((est[f"{c}_error"].where(crossing) ** 2).mean())), 2) for c in "uvw"}
print("actual rms error    ", actual)
print("estimated std       ", estimate)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
for ax, c in zip(axes, "vw"):
    est[f"{c}_error"].sel(z=5000.0).where(crossing.sel(z=5000.0)).plot(ax=ax, cmap="viridis", cbar_kwargs={"label": "m/s"})
    ax.set_title(f"estimated standard deviation of the error of {c} at 5 km")
    ax.set_aspect("equal")
plt.show()
```

The estimate is the error of the analysis system with its weights: it counts
the noise of the radial velocities and the freedom that the weights of the
smoothness and mass terms leave, not the error of the fall speed, of the
gridding or of the advection, and it does not know that a smooth truth is
easier. On the tests of radarx (two radars, 0.5 m/s noise, 2 km grid) the
estimate is 1.0 to 1.4 times the actual error where the beams cross at more
than 30 degrees (this example: 0.8 to 1.4).

**Upper boundary condition.** The true vertical velocity of this flow is zero
at 10 km. `w_boundary` accepts `"obrien"`: after the minimisation the mass
flux is corrected, in each column, by the part of its value at the top that
grows as the square of the height (`obrien_error="constant"`: linearly), so
that `w = 0` there (O'Brien 1970). The option follows the open ARPS code and
documentation of the adjustment; the equations of the paper itself have not been
compared, see the docstring.

```{code-cell} ipython3
adjusted = rx.retrieve.multi_doppler(coarse, coarse_bg, fall_speed_correction=False,
                                     w_boundary=("bottom", "obrien"))
print("w at the top before:", float(abs(base.w.isel(z=-1)).max()), " after:", float(abs(adjusted.w.isel(z=-1)).max()))
print("bottom only         ", rms(base, crossing))
print("bottom and O'Brien  ", rms(adjusted, crossing))
```

**The analytic solution.** `dual_doppler_analytic` solves the 2 x 2 system of
the two radial velocities in every cell and integrates the continuity
equation upward, downward or upward with the O'Brien adjustment. It is
instant but follows every error of the data, and it gives nothing where the
beams do not cross well. It also returns the geometric error of the
horizontal wind, which grows as one over the sine of the crossing angle.

```{code-cell} ipython3
for boundary in ("upward", "downward", "obrien"):
    ana = rx.retrieve.dual_doppler_analytic(coarse, coarse_bg, fall_speed_correction=False,
                                            boundary=boundary, iterations=6, obs_error=1.0)
    ok = crossing & ana.w.notnull()
    print(f"analytic, {boundary:9s}", rms(ana, ok))
print("variational         ", rms(adjusted, crossing))

fig, ax = plt.subplots(figsize=(6, 4))
beta = ana.beam_crossing_angle.where(ana.u_error.notnull()).values.ravel()
herr = np.hypot(ana.u_error, ana.v_error).values.ravel()
ax.scatter(beta[::40], herr[::40], s=3, label="cells")
b = np.linspace(30, 90, 100)
ax.plot(b, np.sqrt(2) / np.sin(np.radians(b)), "k", label=r"$\sqrt{2}\,\sigma / \sin\beta$ (ground level)")
ax.set_xlabel("beam crossing angle (degrees)")
ax.set_ylabel("error of the horizontal wind (m/s)")
ax.legend()
plt.show()
```

## A lower boundary that follows the terrain

The impermeability condition belongs at the ground. `terrain` (the height of
the ground on the grid, for example from a digital elevation model) makes
`multi_doppler` and `dual_doppler_analytic` work on levels that follow it: the
level `k` of a column is at `terrain + z[k] - z[0]`, so `z[0]` is the ground
and `z` of the result is the height above the terrain. With the vertical
velocity relative to these levels, `W = w - u dh/dx - v dh/dy`, the anelastic
continuity equation keeps its form, `W = 0` at the ground is flow along the
terrain, and `w` is returned (see the Notes of `multi_doppler`).

An exact flow over a Gaussian hill of 2 km height, `u = U + a x`, `v = V + a y`,
`w = -2 a (z - h) + u dh/dx + v dh/dy`, which satisfies the continuity
equation and flows along the ground, seen by the same two virtual radars
(the radial velocity is exact, without noise):

```{code-cell} ipython3
def hill_flow(h0=2000.0, scale=10e3, step=2000.0, a=2e-4, u0=8.0, v0=4.0):
    x = y = np.arange(-40e3, 40e3 + 1, step)
    z = np.arange(0.0, 12e3 + 1, 1000.0)
    X, Y = np.meshgrid(x, y)
    h = h0 * np.exp(-(X**2 + Y**2) / (2 * scale**2))
    hx, hy = -X / scale**2 * h, -Y / scale**2 * h
    Z = z[:, None, None]
    u, v = u0 + a * X[None] + 0 * Z, v0 + a * Y[None] + 0 * Z
    w = -2 * a * (Z - h[None]) + u * hx[None] + v * hy[None]
    ds = xr.Dataset(
        {"radar_x": ("radar", [-25e3, 25e3]), "radar_y": ("radar", [-20e3, -20e3]),
         "radar_altitude": ("radar", [0.0, 0.0])},
        coords={"radar": [0, 1], "z": z, "y": y, "x": x},
    )
    geo = rx.retrieve.radar_geometry(ds)
    el, az = np.radians(geo.elevation.values), np.radians(geo.azimuth.values)
    vr = np.cos(el) * (np.sin(az) * u + np.cos(az) * v) + np.sin(el) * w
    near = (np.hypot(ds.x - ds.radar_x, ds.y - ds.radar_y) < 60e3).transpose("radar", "y", "x")
    ds["VRADH"] = (("radar", "z", "y", "x"), np.where(near.values[:, None], vr, np.nan))
    dims = ("z", "y", "x")
    bg = xr.Dataset(
        {"u": (dims, np.full(u.shape, u0)), "v": (dims, np.full(u.shape, v0)),
         "air_density": (dims, np.ones(u.shape))},
        coords={"z": z, "y": y, "x": x},
    )
    w_above_ground = -2 * a * Z + (u0 + a * X)[None] * hx[None] + (v0 + a * Y)[None] * hy[None]
    return ds, bg, xr.DataArray(h, dims=("y", "x"), name="terrain"), w, w_above_ground


hill, hill_bg, terrain, w_cartesian, w_following = hill_flow()
flat = rx.retrieve.multi_doppler(hill, hill_bg, fall_speed_correction=False)
follow = rx.retrieve.multi_doppler(hill, hill_bg, terrain=terrain, fall_speed_correction=False)

above = (hill.z.values[:, None, None] >= terrain.values[None])
ok_flat = ((flat.beam_crossing_angle > 30) & (flat.n_radars >= 2)).values & above
ok_follow = ((follow.beam_crossing_angle > 30) & (follow.n_radars >= 2)).values
e_flat = np.sqrt(np.mean((flat.w.values - w_cartesian)[ok_flat] ** 2))
e_follow = np.sqrt(np.mean((follow.w.values - w_following)[ok_follow] ** 2))
print(f"RMS error of w above the ground: constant-height levels {e_flat:.2f} m/s, "
      f"terrain-following {e_follow:.2f} m/s (the true w has an RMS of "
      f"{np.sqrt(np.mean(w_following[ok_follow] ** 2)):.1f} m/s)")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained", sharey=True)
j = 20  # y = 0
vmax = 1.0
kw = dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
height_above_sea = follow.height.isel(y=j) / 1e3
im = axes[0].pcolormesh(follow.x / 1e3, height_above_sea,
                        (follow.w - w_following).isel(y=j).where(ok_follow[:, j]), **kw)
axes[0].set_title("terrain-following: error of w")
axes[1].pcolormesh(flat.x / 1e3, flat.z / 1e3,
                   (flat.w - w_cartesian).isel(y=j).where(ok_flat[:, j]), **kw)
axes[1].set_title("constant-height levels: error of w")
axes[2].plot(follow.x / 1e3, terrain.isel(y=j) / 1e3, "k")
axes[2].fill_between(follow.x / 1e3, 0, terrain.isel(y=j) / 1e3, color="0.8")
axes[2].set_title("terrain at y = 0")
for ax in axes:
    ax.set_xlabel("x (km)")
axes[0].set_ylabel("height above sea level (km)")
fig.colorbar(im, ax=axes[:2], label="m/s")
plt.show()
```

Levels of constant height impose `w = 0` at the lowest level, which is
inside the hill; the error then spreads up the column. The slopes are
centred differences of the terrain on the grid, so a terrain much rougher
than the grid gives a noisy `w`. The terrain is not used to mask the beam
blockage behind ridges (see `beam_blockage`), and the data are interpolated
linearly in height to the terrain-following levels.

## Observation errors from the signal to noise ratio

The variance of a radial velocity from the pulse-pair estimator depends on
the signal to noise ratio, the spectrum width and the number of pulse pairs
(Doviak and Zrnic 1993, Eqs. 6.22a and b, from Zrnic 1977). `velocity_error`
evaluates it, and its result goes into `obs_error`:

```{code-cell} ipython3
width = np.linspace(0.5, 8.0, 60)
fig, ax = plt.subplots(figsize=(6.5, 4))
for snr in (0, 5, 10, 20):
    sd = rx.retrieve.velocity_error(snr, width, wavelength=0.1071, nyquist=26.5, n_pairs=64)
    ax.plot(width, sd, label=f"SNR {snr} dB")
ax.set_xlabel("spectrum width (m/s)")
ax.set_ylabel("standard deviation of the velocity (m/s)")
ax.set_title("S band, Nyquist velocity 26.5 m/s, 64 pairs")
ax.legend()
plt.show()
```

The model is checked in the tests against the standard deviation of
simulated pulse-pair estimates (within 6 %) and against the WSR-88D figure in
the book (about 1 m/s at 8 dB, 4 m/s width, 40 to 200 samples). On a grid,
compute `snr` and take the spectrum width from the volume (`WRADH`), both
gridded like the velocity, and pass `obs_error=velocity_error(...)`; where the
width or the signal to noise ratio is missing the error is NaN and the
observation is left out. Only the relative sizes of the errors of the radars
matter for the weights of the other terms (see `obs_error`).

## Mixed-order interpolation of the radial velocity

Where the radial velocity changes quickly with height and the beams are a few
degrees apart, interpolating inside each sweep first and between sweeps last
leaves undulations at the horizontal spacing of the beams, and the mass
continuity equation turns their divergence into spurious vertical velocity
(Collis et al. 2010). `radarx.grid.grid_cones_mixed` blends this result with the
one of the opposite order of interpolation, weighted by the beam elevation
(`order` is the exponent of their Eq. 6), and `multi_doppler_input` has the
option `velocity_interpolation="mixed"`. A 20 m/s jet at 7 km, seen by a volume
of twelve sweeps from 0.5 to 42 degrees:

```{code-cell} ipython3
from xradar.georeference import antenna_to_cartesian


def jet(z):
    return 20.0 * np.exp(-(((z - 7000.0) / 1500.0) ** 2))


ranges = np.arange(250.0, 80e3, 250.0)
azimuths = np.arange(0.5, 360.0, 1.0)
sweeps = {}
for k, elevation in enumerate((0.5, 1.5, 3.0, 5.0, 7.5, 10.0, 13.0, 17.0, 22.0, 28.0, 35.0, 42.0)):
    _, _, gate_z = antenna_to_cartesian(ranges[None, :], azimuths[:, None], elevation, site_altitude=100.0)
    vr = jet(gate_z) * np.cos(np.radians(elevation)) * np.sin(np.radians(azimuths))[:, None]
    sweeps[f"sweep_{k}"] = xr.Dataset(
        {"VRADH": (("azimuth", "range"), vr)},
        coords={"azimuth": azimuths, "range": ranges,
                "elevation": ("azimuth", np.full(azimuths.size, elevation)),
                "time": ("azimuth", np.full(azimuths.size, np.datetime64("2026-01-01T00:00")))},
    )
volume = xr.DataTree.from_dict(
    {"/": xr.Dataset(coords={"latitude": 45.0, "longitude": 10.0, "altitude": 100.0}), **sweeps}
)

gx = np.arange(-60e3, 60e3 + 1, 2e3)
gz = np.arange(1000.0, 13e3, 500.0)
vertical_last = rx.grid.grid_cones(volume, "VRADH", gx, gx, gz).VRADH
blended = rx.grid.grid_cones_mixed(volume, "VRADH", gx, gx, gz).VRADH
horizontal_last = rx.grid.grid_cones_mixed(volume, "VRADH", gx, gx, gz, order=0.0).VRADH

# the exact radial velocity of the jet at the grid points
X, Z = np.meshgrid(gx, gz)
elevation = rx.grid.mixed.antenna_elevation(np.abs(X), Z, 100.0)  # beam elevation of the point
truth_section = jet(Z) * np.cos(np.radians(elevation)) * np.sign(X)
fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained", sharey=True)
for ax, field, title in zip(axes, (vertical_last, blended, horizontal_last),
                            ("vertical last (grid_cones)", "mixed, order 0.5", "horizontal last, order 0")):
    section = field.sel(y=0.0)
    err = section - truth_section
    im = ax.pcolormesh(gx / 1e3, gz / 1e3, err, cmap="RdBu_r", vmin=-4, vmax=4, shading="auto")
    rms = float(np.sqrt(np.nanmean(err.where(np.abs(section.x) > 10e3) ** 2)))
    ax.set_title(f"{title}\nRMS error {rms:.2f} m/s")
    ax.set_xlabel("x (km)")
axes[0].set_ylabel("height (km)")
fig.colorbar(im, ax=axes, label="gridded minus true radial velocity (m/s)")
plt.show()
```

The undulations of the first panel are what the blend lowers; the price is
a smoother field in the horizontal at high elevations, where the weight of
the horizontal-last result grows to one. The exponent 0.5 is the value Collis
et al. chose for their 17 tilts from 0.5 to 42 degrees and should be tuned to
the scan strategy.

## A squall line seen by KGWX and KBMX

On 30 March 2022 a squall line crossed Mississippi. At 00 UTC it was just
west of the KGWX radar (Columbus, Mississippi); KBMX (Birmingham, Alabama)
is 166 km to the east-south-east. We dealias both volumes, read the Nyquist
velocity from the radial headers (as in the dealiasing notebook) and mask
the no-data codes.

```{code-cell} ipython3
def nexrad_volume(key):
    path = download_file("unidata-nexrad-level2", f"2022/03/30/{key}", ".")
    with NEXRADLevel2File(path) as nf:
        nyquist = [h["msg_31_data_header"]["RAD"]["nyquist_vel"] / 100.0 for h in nf.msg_31_data_header]
    dtree = xd.io.open_nexradlevel2_datatree(path)
    for i, name in enumerate(n for n in dtree.children if n.startswith("sweep")):
        ds = dtree[name].to_dataset()
        if "DBZH" in ds:
            ds["DBZH"] = ds.DBZH.where(ds.DBZH > -32)
        if "VRADH" in ds:
            ds["VRADH"] = ds.VRADH.where(ds.VRADH > -63.9)
        dtree[name] = ds.assign_coords(nyquist_velocity=nyquist[i])
    return dtree


def dealiased(key):
    """Volume with its radial velocity dealiased in place (as ``VRADH``)."""
    vol = nexrad_volume(key)
    return vol.radarx.assign(vol.radarx.dealias("VRADH", name="VRADH"))


kgwx = dealiased("KGWX/KGWX20220330_235959_V06")
kbmx = dealiased("KBMX/KBMX20220330_235713_V06")
```

The radars scanned at different times, so we estimate the storm motion from
the previous KGWX volume and move both radars to 00:00 UTC while gridding
them onto one 2-km grid centred on KGWX.

```{code-cell} ipython3
previous = nexrad_volume("KGWX/KGWX20220330_235324_V06")
grid_kw = dict(x_lim=(-150e3, 150e3), y_lim=(-150e3, 150e3), z_lim=(1000, 6000), z_step=1000)
motion = rx.retrieve.estimate_motion(previous.radarx.to_grid(["DBZH"], **grid_kw), kgwx.radarx.to_grid(["DBZH"], **grid_kw))
print(f"storm motion u = {float(motion.u):.1f} m/s, v = {float(motion.v):.1f} m/s")

grids = rx.retrieve.multi_doppler_input(
    [kgwx, kbmx],
    x=np.arange(-120e3, 60e3 + 1, 2000.0),
    y=np.arange(-120e3, 100e3 + 1, 2000.0),
    z=np.arange(500.0, 12e3 + 1, 500.0),
    time="2022-03-31T00:00:00",
    motion=motion,
)
grids
```

The background (wind, air density and freezing level) comes from the
Birmingham radiosonde at 00 UTC, spread over the grid. `era5_column` gives
an ERA5 background in the same form.

```{code-cell} ipython3
profile = sounding.read_sounding("BMX", "2022-03-31T00:00")
background = sounding.profile_to_grid(profile, grids)

start = time.perf_counter()
wind = grids.radarx.multi_doppler(background)
print(f"retrieval: {time.perf_counter() - start:.1f} s, iterations {wind.attrs['iterations']}")
print(wind.cost.to_series())
```

The dual-Doppler area is where both radars observe with a beam crossing
angle above 30 degrees. KBMX is far away, so its beams reach the line only
above about 3 km.

```{code-cell} ipython3
good = (wind.beam_crossing_angle > 30) & (wind.n_radars >= 2)
dbz = grids.DBZH.max("radar")
km = dict(x=grids.x / 1e3, y=grids.y / 1e3)

height = 5000.0
level = wind.sel(z=height).where(good.sel(z=height))
sub = level.isel(x=slice(None, None, 3), y=slice(None, None, 3))
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), layout="constrained", sharey=True)
im0 = axes[0].pcolormesh(km["x"], km["y"], dbz.sel(z=height), cmap="ChaseSpectral", vmin=-10, vmax=70)
im1 = axes[1].pcolormesh(km["x"], km["y"], level.w, cmap="RdBu_r", vmin=-8, vmax=8)
axes[1].contour(km["x"], km["y"], dbz.sel(z=height).fillna(-30), levels=[35, 50], colors="k", linewidths=0.6)
for ax, title in zip(axes, ["reflectivity and wind", "w (contours: 35 and 50 dBZ)"]):
    ax.quiver(sub.x / 1e3, sub.y / 1e3, sub.u, sub.v, scale=700, width=0.002)
    ax.plot(grids.radar_x / 1e3, grids.radar_y / 1e3, "k^")
    ax.set_title(f"{height / 1e3:.0f} km: {title}")
    ax.set_xlim(-120, 60)
    ax.set_ylim(-120, 100)
    ax.set_aspect("equal")
    ax.set_xlabel("x (km east of KGWX)")
axes[0].set_ylabel("y (km north of KGWX)")
fig.colorbar(im0, ax=axes[0], label="reflectivity (dBZ)", shrink=0.8)
fig.colorbar(im1, ax=axes[1], label="w (m/s)", shrink=0.8)
plt.show()
```

A vertical cross-section of w across the line:

```{code-cell} ipython3
section = dict(y=-40e3, method="nearest")
fig, ax = plt.subplots(figsize=(10, 4))
im = ax.pcolormesh(wind.x / 1e3, wind.z / 1e3, wind.w.sel(**section).where(good.sel(**section)),
                   cmap="RdBu_r", vmin=-10, vmax=10)
ax.contour(grids.x / 1e3, grids.z / 1e3, dbz.sel(**section).fillna(-30), levels=[20, 40], colors="k", linewidths=0.7)
ax.set_xlabel("x (km)")
ax.set_ylabel("height (km)")
ax.set_title("w across the squall line at y = -40 km (contours: 20 and 40 dBZ)")
fig.colorbar(im, label="w (m/s)")
plt.show()
```

The retrieved wind reproduces both radars' velocities to within the
observation noise:

```{code-cell} ipython3
residual = wind.vr_residual.where(good)
for i, name in enumerate(wind.radar_name.values):
    print(f"{name}: RMS radial velocity residual {float(np.sqrt((residual.isel(radar=i) ** 2).mean())):.2f} m/s")
```
