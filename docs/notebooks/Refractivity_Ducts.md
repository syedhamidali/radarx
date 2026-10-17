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

# Trapping layers of a refractivity profile

A layer where the refractivity falls faster than 157 N-units per km traps the
radar beam, and the radar sees anomalous propagation (ground clutter far away,
echoes at the wrong place). `radarx.retrieve.ducting_layers` finds such layers
in a sounding or in the profiles of a reanalysis before any beam is traced with
`radarx.retrieve.beam_path` (see the notebook `Beam_Refraction`).

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.fundamentals import trace_rays
from radarx.retrieve import ducting_layers, modified_refractivity

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

## Which layers trap the beam

`radarx.retrieve.ducting_layers` finds the trapping layers of a profile before
any beam is traced. The modified refractivity
$M = N + 10^{6}\,h/a$ (Bean and Dutton 1966, Table 1.9) adds the curvature of
the Earth, so that the trapping gradient of $-10^6/a \approx -157$ N-units per
km is where $M$ decreases with height. For a layer from $h_b$ to $h_t$ the strength
is the M-deficit $\Delta M = M(h_b) - M(h_t)$ and the angle of penetration
(their Eq. 4.17, from Snell's law) is the elevation below which a ray from the
base turns over before it reaches the top, $\cos\theta_p = n_t r_t/(n_b r_b)$,
about $\sqrt{2\,\Delta M\,10^{-6}}$ (their Eq. 4.22). The book gives 8.3 mrad
for a ground-based layer of 100 m with a gradient of -0.5 N-units per m and
4.5 mrad for one of 30 m; the function gives the same. Rays traced with
`trace_rays` at fractions of the angle show the rule: below it the ray turns at
the height that Snell's law gives (inside the layer), above it the ray leaves.
The black line is the ground: the tracer does not stop a ray there, so the
trapped rays dip below it where a real ray would be reflected.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)
ranges = np.arange(0.0, 120e3, 200.0)
for k, (thick, color) in enumerate(((100.0, OKABE["blue"]), (30.0, OKABE["orange"]))):
    h = np.array([0.0, thick, 3000.0])
    n = np.array([330.0, 330.0 - 0.5 * thick, 330.0 - 0.5 * thick - 0.04 * (3000.0 - thick)])
    profile = xr.DataArray(n, dims="height", coords={"height": h})
    layer = ducting_layers(profile)
    angle = float(layer.trapping_angle.isel(layer=0))
    print(f"layer of {thick:.0f} m: M deficit {float(layer.m_deficit.isel(layer=0)):.2f}, trapping angle "
          f"{np.radians(angle) * 1e3:.2f} mrad (sqrt(2 dM 1e-6): "
          f"{np.sqrt(2e-6 * float(layer.m_deficit.isel(layer=0))) * 1e3:.2f} mrad)")
    m = modified_refractivity(profile)
    axes[0].plot(m.values[:2], h[:2], "o-", color=color, label=f"{thick:.0f} m layer")
    for frac, ls in ((0.9, "-"), (1.1, "--")):
        ray = trace_rays(ranges, frac * angle, h, n, radar_height=0.0, step=50.0)
        axes[1 + k].plot(ranges / 1e3, ray.beam_height, ls, color=color, label=f"{frac} of the angle")
    axes[1 + k].axhline(thick, color="0.4", lw=0.8)
    axes[1 + k].axhline(0.0, color="k", lw=0.8)
    axes[1 + k].set(xlabel="range (km)", ylabel="beam height (m)", title=f"rays from the base, {thick:.0f} m layer")
    axes[1 + k].legend(frameon=False, fontsize=8)
axes[0].set(xlabel="modified refractivity M", ylabel="height (m)", title="M decreases in the layer")
axes[0].legend(frameon=False, fontsize=8)
plt.show()
```

A sounding gives its layers in the same way (`ds.radarx.ducting_layers()`, or
`ducting_layers(ds)` for the profiles of a reanalysis on extra dimensions);
`radar_height` adds the angle for a radar inside a layer. The wavelength of
the radar is not taken into account: a thin layer traps only short wavelengths
(Bean and Dutton 1966, Eqs. 4.24 and 4.25).

## References

- Bean, B. R., and E. J. Dutton, 1966: *Radio Meteorology*. NBS Monograph 92,
  National Bureau of Standards, <https://doi.org/10.6028/NBS.MONO.92>
