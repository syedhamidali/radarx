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

# Beam height and ground range through a refractivity profile

The beam height and ground range of every radar function in radarx follow the
effective Earth radius model: a straight ray over an Earth that is 4/3 as
large, which stands for a refractive index that decreases linearly with height
by 39 N-units per km. That is the mean atmosphere. In a humid layer under dry
air the refractivity falls much faster and the beam bends down, in a dry layer
under humid air it falls more slowly and the beam climbs, and where it falls by
more than 157 N-units per km the beam is trapped in a duct. The
refractivity of a sounding or of reanalysis levels tells which case it is.

`radarx.fundamentals.trace_rays` follows the ray through a profile of the
refractivity $N(h)$ and gives the true beam height and ground range of every
gate. `radarx.retrieve.beam_path` does it for the gates of a sweep or a volume,
and `radarx.fundamentals.k_e_from_rays` turns a traced beam into the effective
Earth radius factor that the gridding, merging and beam blockage functions take.
`radarx.io.sounding.refractivity` computes $N$ from pressure, temperature and
water vapour.

## The ray

The atmosphere is spherically stratified: the refractive index
$n = 1 + 10^{-6} N$ depends only on the distance $r = a + h$ from the centre of
the Earth. A ray obeys Snell's law in polar coordinates,
$n\,r\cos\theta = \mathrm{const}$ (Bean and Dutton 1966, Eq. 3.1), with the
elevation $\theta$ of the ray against the local horizontal. With the path length
$s$ and the central angle $\varphi$,

$$
\frac{dr}{ds} = \sin\theta,\qquad
\frac{d\varphi}{ds} = \frac{\cos\theta}{r},\qquad
\frac{d\theta}{ds} = \cos\theta\left(\frac{1}{r} + \frac{1}{n}\frac{dn}{dh}\right),
$$

the last term being the curvature of the ray of Eq. (1.27) of the same book. A
compiled multithreaded kernel integrates the three equations with the
fourth-order Runge-Kutta method from gate to gate (the steps end at the levels
of the profile, where the gradient changes), with a NumPy twin that gives the
same numbers.

The refractivity is $N = 77.6\,p/T + 3.73\times10^{5}\,e/T^2$ with $p$ and $e$ in
hPa and $T$ in K (Bean and Dutton 1966, Eq. 1.17), accurate to 0.5 % at
frequencies up to 30 GHz. The three-term formula of Smith and Weintraub and the
constants of ITU-R P.453-9 are options.

```{code-cell} ipython3
import time

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.fundamentals import k_e_from_rays, trace_rays
from radarx.fundamentals.constants import EARTH_RADIUS
from radarx.fundamentals.geometry import beam_center_height, ground_range
from radarx.io import refractivity, saturation_vapor_pressure
from radarx.retrieve import beam_path

plt.rcParams.update(
    {
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "font.size": 9.5,
    }
)
OKABE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "black": "#222222"}
```

## Refractivity from a sounding

Pressure, temperature and vapour pressure of a standard atmosphere at 60 %
relative humidity give 319 N-units at the ground and a gradient of $-55$
N-units per km in the lowest kilometre; the 4/3 Earth model stands for
$-39$ N-units per km, and the humidity makes the real gradient steeper.

```{code-cell} ipython3
z = np.arange(0.0, 15001.0, 100.0)
pressure = 101325.0 * (1.0 - 2.25577e-5 * z) ** 5.25588
temperature = np.maximum(288.15 - 0.0065 * z, 216.65)
vapor = 0.6 * saturation_vapor_pressure(temperature).values * np.exp(-z / 1500.0)
N = refractivity(pressure, temperature, vapor).values
print(f"N at the ground {N[0]:.1f}, gradient in the lowest km {N[10] - N[0]:.1f} N-units per km")
for form in ("bean_dutton", "smith_weintraub", "itu"):
    print(f"{form:16s}", np.round(refractivity(pressure[:3], temperature[:3], vapor[:3], form=form).values, 2))
```

## Agreement with the 4/3 Earth model

For the constant gradient of the 4/3 Earth ($-39.2$ N-units per km, that is
$-1/(4a)$) the ray tracer gives the effective Earth radius result. The numbers
are those of the tests: up to 200 km and $1^\circ$ the beam height is within
0.5 m and the ground range within 0.016 %; the difference is what the
effective radius model neglects, the cosine of the elevation and the index in
the curvature of the ray, and the ratio of the radii in the ground distance.

```{code-cell} ipython3
standard_gradient = 1e6 / (4.0 * EARTH_RADIUS / 1000.0)  # 39.24 N per km
h = np.arange(0.0, 30001.0, 500.0)
n_linear = 313.0 - standard_gradient * h / 1000.0
rng = np.arange(1e3, 200e3 + 1.0, 1e3)
rows = []
for elevation in (0.5, 1.0, 5.0):
    ray = trace_rays(rng, elevation, h, n_linear)
    h43 = beam_center_height(rng, elevation, reff=4 / 3 * EARTH_RADIUS)
    s43 = ground_range(rng, elevation, reff=4 / 3 * EARTH_RADIUS)
    k_e = k_e_from_rays(rng, elevation, ray.beam_height)
    rows.append((elevation, float(np.abs(ray.beam_height - h43).max()),
                 float(np.abs(ray.ground_range / s43 - 1).max()), float(k_e.min()), float(k_e.max())))
print("elevation  max |dh| (m)  max |ds|/s   k_e range")
for e, dh, ds, k0, k1 in rows:
    print(f"{e:8.1f} {dh:12.2f} {ds:12.2e}   {k0:.4f} to {k1:.4f}")
```

## Sub-refractive, standard and super-refractive atmospheres, and a duct

Four profiles that differ in the lowest 500 m: the standard atmosphere, a
sub-refractive one (a dry layer under moister air, so that $N$ falls more slowly
or rises), a super-refractive one ($N$ falls by 130 N-units per km near the
ground) and a ducting one (steeper than 157 N-units per km).

```{code-cell} ipython3
z = np.arange(0.0, 12001.0, 50.0)
base = 313.0 * np.exp(-z / 7980.0)   # 39.2 N-units per km at the ground
profiles = {
    "standard": base,
    "sub-refractive": base - 38.0 * np.exp(-z / 250.0),
    "super-refractive": base + 25.0 * np.exp(-z / 250.0),
    "duct": base + 55.0 * np.exp(-z / 170.0),
}
colors = {"standard": OKABE["black"], "sub-refractive": OKABE["orange"],
          "super-refractive": OKABE["blue"], "duct": OKABE["red"]}
ranges = np.arange(1e3, 250e3 + 1.0, 1e3)
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)
for name, n_profile in profiles.items():
    axes[0].plot(n_profile[z <= 1500], z[z <= 1500], color=colors[name], label=name)
    gradient = np.gradient(n_profile, z) * 1000.0
    ray = trace_rays(ranges, 0.5, z, n_profile)
    axes[1].plot(ranges / 1e3, ray.beam_height / 1e3, color=colors[name], label=name)
    print(f"{name:17s} lowest-100-m gradient {gradient[:2].mean():7.1f} N/km,"
          f" beam height at 150 km {float(ray.beam_height.sel(range=150e3)):7.0f} m,"
          f" ground range {float(ray.ground_range.sel(range=150e3)) / 1e3:7.3f} km")
axes[1].plot(ranges / 1e3, beam_center_height(ranges, 0.5, reff=4 / 3 * EARTH_RADIUS) / 1e3, "k:", lw=1,
             label="4/3 Earth")
axes[0].set(xlabel="refractivity N", ylabel="height (m)", title="profiles")
axes[1].set(xlabel="range (km)", ylabel="beam height (km)", title="elevation 0.5°")
axes[1].legend(frameon=False, fontsize=8)
trapping = -1e6 / (EARTH_RADIUS / 1000.0)
for elevation, ls in ((0.0, "-"), (0.25, "--"), (0.6, ":")):
    ray = trace_rays(ranges, elevation, z, profiles["duct"], radar_height=120.0)
    axes[2].plot(ranges / 1e3, ray.beam_height, ls, color=OKABE["red"], label=f"{elevation}°")
axes[2].axhline(0.0, color="0.4", lw=1.0)
axes[2].set(xlabel="range (km)", ylabel="beam height (m)", ylim=(-80, 400),
            title="surface duct (radar at 120 m)")
axes[2].legend(title="elevation", frameon=False, fontsize=8)
plt.show()
print(f"trapping gradient {trapping:.0f} N-units per km")
```

The ray of the sub-refractive atmosphere is higher than that of the standard
one and the super-refractive one is lower. In the duct a ray launched along the
layer keeps near the ground and returns to it (the line is the ground: the
model does not stop the ray there) and a steeper ray leaves.

## The gates of a sweep

`beam_path` puts the tracer on the range gates of a sweep or a volume and
returns the beam height, the ground range, the elevation of the ray at the gate
and the effective Earth radius factor of each gate. The profile is a dataset
from `radarx.io.sounding` (with `refractivity`, or with `pressure`,
`temperature` and a humidity variable) or a pair of arrays.

```{code-cell} ipython3
def make_sweep(elevation, nray=360, ngate=480, dr=500.0, altitude=350.0):
    az = (np.arange(nray) + 0.5) * 360.0 / nray
    return xr.Dataset(
        {"DBZH": (("azimuth", "range"), np.zeros((nray, ngate), dtype="f4"))},
        coords={"azimuth": az, "range": dr * (np.arange(ngate) + 1.0),
                "elevation": ("azimuth", np.full(nray, elevation)),
                "latitude": 19.0, "longitude": 73.0, "altitude": altitude},
    )

sounding = xr.Dataset(
    {"refractivity": ("height", profiles["super-refractive"])}, coords={"height": z}
)
sweep = make_sweep(0.5)
t0 = time.time()
path = beam_path(sweep, sounding)
print(f"{path.beam_height.size:,} gates in {1000 * (time.time() - t0):.0f} ms")
print(path.beam_height.isel(azimuth=0, range=[0, 199, 479]).values.round(1), "m")
print("median k_e of the sweep:", float(path.k_e.median()))
accessor = sweep.radarx.beam_path(sounding)  # the same as a method of the sweep
standard = beam_path(sweep, xr.Dataset({"refractivity": ("height", profiles["standard"])}, coords={"height": z}))
fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
r_km = sweep.range / 1e3
ax.plot(r_km, (path.beam_height - standard.beam_height).isel(azimuth=0), color=OKABE["blue"])
ax.set(xlabel="range (km)", ylabel="beam height, super-refractive minus standard (m)")
plt.show()
```

The median `k_e` of the traced beams is the factor to pass as `k_e` to
`radarx.grid.grid_radar`, the cone and merging functions and
`beam_blockage`, which keep the effective radius model.

The layers that trap the beam are found by `ducting_layers`, see the notebook
`Refractivity_Ducts`.

## Limits

- The atmosphere is horizontally homogeneous: a sea breeze front or an outflow
  that changes the profile along the beam is not represented.
- Between the levels of the profile the refractivity is linear; below the first
  level and above the last one the refractive index is constant, which for the
  top of a sounding is a radarx choice.
- The slant range of the gates is taken as the length of the ray; the apparent
  range of the radar, $\int n\,ds$, differs by at most about 200 m
  (Bean and Dutton 1966).
- A ray is not stopped at the surface. In a duct the beam can dip below the
  ground in the model; whether it would be blocked is the question of
  `beam_blockage`.
- Geometric optics: the diffraction of a beam in a duct is not modelled.

## References

- Bean, B. R., and E. J. Dutton, 1966: *Radio Meteorology*. NBS Monograph 92,
  National Bureau of Standards, <https://doi.org/10.6028/NBS.MONO.92>
- Smith, E. K., and S. Weintraub, 1953: The constants in the equation for
  atmospheric refractive index at radio frequencies. *Proc. IRE*, **41**,
  1035–1037, <https://doi.org/10.1109/JRPROC.1953.274297> (the constants as
  reported by Bean and Dutton; the paper has not been compared)
