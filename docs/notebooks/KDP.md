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

# Differential Phase Processing and KDP

The specific differential phase $K_{DP}$ is half the range derivative of the
propagation differential phase $\Phi_{DP}$. It is immune to calibration errors,
attenuation and partial beam blockage, which makes it valuable for rain rate,
drop size distribution retrievals and attenuation correction. The measured
phase $\Psi_{DP}$ is not $\Phi_{DP}$, however: it contains the system offset,
it may be folded into a 360° interval, it is noisy, it is meaningless outside
precipitation, and in large drops it carries a backscatter phase $\delta$.

`radarx.retrieve.estimate_kdp` (or `.radarx.kdp()` on a sweep or a volume)
handles all of this ray by ray:

1. masks non-meteorological gates ($\rho_{hv}$ and the phase texture),
2. detects the sign convention: some systems record a phase that decreases
   with range in rain; it is multiplied by −1 if needed (`phidp_sign`),
3. estimates the system offset from the first valid gates,
4. unfolds the phase and removes the offset,
5. filters the profile in range, by default with the iterative filter of
   Hubbert and Bringi (1995), which suppresses $\delta$,
6. computes $K_{DP}$ by least squares over a window that is shorter in heavy
   rain (high $Z_H$) and longer in light rain, at gates whose window holds
   enough valid gates.

A compiled kernel processes all rays of all sweeps in one call, in parallel.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import xradar as xd
from open_radar_data import DATASETS

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.retrieve import estimate_kdp
```

## Read a sweep

A C-band (CSAPR2) PPI of deep convection in Argentina during the CACTI field
campaign. Like many processed files, it holds several phase fields; KDP must
be computed from the **raw** measured phase:

- `uncorrected_differential_phase`: raw phase in [−180°, 180°),
- `differential_phase`: the same raw phase as 360° minus it, in [0°, 360°),
  so it *decreases* with range,
- `unfolded_differential_phase`, `corrected_differential_phase`, ...:
  products of an earlier processing chain.

By default `estimate_kdp` takes the first of `UPHIDP`, `PHIDP`,
`uncorrected_differential_phase`, `differential_phase`, `PHI` that exists; the
fields actually used are stored in the `source_fields` attribute. Passing the
field names explicitly is safest.

```{code-cell} ipython3
file = DATASETS.fetch("corcsapr2cmacppiM1.c1.20181111.030003.nc")
dtree = xd.io.open_cfradial1_datatree(file).xradar.georeference()
sweep = dtree["sweep_0"].to_dataset()
fields = {
    "phidp": "uncorrected_differential_phase",
    "rhohv": "uncorrected_copol_correlation_coeff",
    "dbzh": "uncorrected_reflectivity_h",
}
```

## Process ΦDP and estimate KDP

```{code-cell} ipython3
out = sweep.radarx.kdp(**fields)
out
```

```{code-cell} ipython3
print(out.PHIDP_processed.attrs["source_fields"], "| sign:", out.PHIDP_processed.attrs["phidp_sign"])
```

The default call picks `uncorrected_differential_phase`. Starting from the
decreasing `differential_phase` instead gives the same KDP, because the sign
convention is detected (`phidp_sign=-1`):

```{code-cell} ipython3
default = estimate_kdp(sweep)
flipped = estimate_kdp(sweep, phidp="differential_phase")
for res in (default, flipped):
    k = res.KDP.where(sweep.reflectivity >= 45)
    print(
        f"{res.PHIDP_processed.attrs['source_fields']:62s} sign {res.PHIDP_processed.attrs['phidp_sign']:+d}"
        f"  mean KDP in cores (Z >= 45 dBZ): {float(k.mean()):.2f} deg/km"
    )
```

The system offset of this radar is close to 180°, so the measured phase folds
from +180° to −180° soon after it enters rain. The processed phase is unfolded,
starts at zero and increases through the storms.

```{code-cell} ipython3
print(f"system offset: {float(out.PHIDP_OFFSET[0]):.1f} degrees")
```

## Along one ray

```{code-cell} ipython3
iray = int(np.nanargmax(out.PHIDP_processed.isel(range=-1).values))
ray = sweep.isel(azimuth=iray)
res = out.isel(azimuth=iray)
r_km = sweep.range.values / 1000.0

fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
axes[0].plot(r_km, ray[fields["dbzh"]], color="k", lw=0.8)
axes[0].set_ylabel("$Z_H$ (dBZ)")
axes[1].plot(r_km, ray[fields["phidp"]], ".", ms=2, color="0.6", label="raw $\\Psi_{DP}$")
axes[1].plot(r_km, res.PHIDP_processed, color="C0", label="processed $\\Phi_{DP}$")
axes[1].set_ylabel("phase (°)")
axes[1].legend(loc="upper left")
for method, color in [("hubbert", "C0"), ("vulpiani", "C1"), ("monotone", "C2")]:
    k = estimate_kdp(ray.expand_dims("azimuth"), method=method, offset=float(out.PHIDP_OFFSET[iray]), **fields)
    axes[2].plot(r_km, k.KDP.squeeze(), color=color, label=method)
axes[2].set_ylabel("$K_{DP}$ (°/km)")
axes[2].set_xlabel("range (km)")
axes[2].legend(loc="upper right")
axes[0].set_title(f"azimuth {float(ray.azimuth):.1f}°")
fig.tight_layout()
```

## PPIs

`KDP` is NaN at non-meteorological gates. The processed `PHIDP` is defined on
every gate of a ray (masked gates are bridged and the end values held, as
needed for attenuation correction); here it is shown on meteorological gates
only.

```{code-cell} ipython3
x = sweep.x / 1000.0
y = sweep.y / 1000.0
panels = [
    (sweep[fields["dbzh"]], "$Z_H$ (dBZ)", "ChaseSpectral", -10, 65),
    (sweep[fields["phidp"]], "raw $\\Psi_{DP}$ (°)", "twilight", -180, 180),
    (out.PHIDP_processed.where(out.KDP.notnull()), "processed $\\Phi_{DP}$ (°)", "viridis", 0, 250),
    (out.KDP, "$K_{DP}$ (°/km)", "turbo", -1, 6),
]
fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
for ax, (da, label, cmap, vmin, vmax) in zip(axes.flat, panels):
    try:
        import cmweather  # noqa: F401
    except ImportError:
        cmap = "turbo" if cmap == "ChaseSpectral" else cmap
    pm = ax.pcolormesh(x, y, da, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(pm, ax=ax, shrink=0.8, label=label)
    ax.set_aspect("equal")
    ax.set_xlim(-110, 110)
    ax.set_ylim(-110, 110)
for ax in axes[1]:
    ax.set_xlabel("east (km)")
for ax in axes[:, 0]:
    ax.set_ylabel("north (km)")
fig.tight_layout()
```

## Comparison with the radar's own KDP

The file also holds the KDP computed by the radar processor
(`specific_differential_phase`).

```{code-cell} ipython3
ref = sweep.specific_differential_phase.values
k = out.KDP.values
sel = (
    (sweep[fields["dbzh"]].values > 30)
    & (sweep[fields["rhohv"]].values > 0.95)
    & np.isfinite(k)
    & np.isfinite(ref)
)
fig, ax = plt.subplots(figsize=(5, 5))
ax.hexbin(ref[sel], k[sel], gridsize=60, bins="log", extent=(-1, 7, -1, 7))
ax.plot([-1, 7], [-1, 7], color="r", lw=1)
ax.set_xlabel("radar KDP (°/km)")
ax.set_ylabel("radarx KDP (°/km)")
ax.set_title(f"r = {np.corrcoef(ref[sel], k[sel])[0, 1]:.2f}")
ax.set_aspect("equal")
```

## A whole volume

On a DataTree, every sweep is processed and all rays of the volume go to the
compiled kernel in a single call.

```{code-cell} ipython3
vol = dtree.radarx.kdp(**fields)
vol
```

## References

- Hubbert, J., and V. N. Bringi, 1995: An iterative filtering technique for
  the analysis of copolar differential phase and dual-frequency radar
  measurements. *J. Atmos. Oceanic Technol.*, **12** (3), 643–648,
  <https://doi.org/10.1175/1520-0426(1995)012<0643:AIFTFT>2.0.CO;2>
- Wang, Y., and V. Chandrasekar, 2009: Algorithm for estimation of the
  specific differential phase. *J. Atmos. Oceanic Technol.*, **26** (12),
  2565–2578, <https://doi.org/10.1175/2009JTECHA1358.1>
- Vulpiani, G., M. Montopoli, L. D. Passeri, A. G. Gioia, P. Giordano, and
  F. S. Marzano, 2012: On the use of dual-polarized C-band radar for
  operational rainfall retrieval in mountainous areas. *J. Appl. Meteor.
  Climatol.*, **51** (2), 405–425, <https://doi.org/10.1175/JAMC-D-10-05024.1>
- Maesaka, T., K. Iwanami, and M. Maki, 2012: Non-negative KDP estimation by
  monotone increasing ΦDP assumption below melting layer. *Proc. Seventh
  European Conf. on Radar in Meteorology and Hydrology (ERAD 2012)*, Toulouse,
  France (conference paper, no DOI).
