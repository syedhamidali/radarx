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

# A two-moment rain shaft

`radarx.retrieve.rain_shaft` evolves a drop size distribution on many bins and
is too expensive for every gate of a volume (see the notebook `Rain_Collisions`).
`radarx.retrieve.rain_shaft_bulk` follows two moments of the rain, the number
concentration and the water content, with the self-collection and breakup of
Seifert (2008, Appendix).

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.retrieve import _rain_bulk_numpy as seifert
from radarx.retrieve import rain_shaft, rain_shaft_bulk

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

## The scheme against the bin model

The drops are a gamma distribution whose shape parameter follows from the mean volume diameter $D_m$ (Eq. 20 of the paper; $\mu = 1$ at the equilibrium diameter of 1.1 mm), self-collection removes drops at the rate $k_{rr} N_r L_r$ and breakup is a relaxation of $D_m$ toward 1.1 mm. The
figure compares it with the bin model `rain_shaft` on the same tops. The
two agree on the direction and the scale of the change and on a mean volume
diameter between 1 and 1.3 mm after a few kilometres, not on the details: the
bin model with the breakup of Low and List first undershoots 1.1 mm and then
settles at about 1.26 mm, the two-moment scheme goes to 1.1 mm without
overshoot, and rain of small drops coalesces faster in the two-moment scheme.
The rain rates differ by a few percent because the two-moment scheme assumes a
gamma distribution with the shape parameter of Eq. (20) for the same water
content and number, which has other mass-weighted fall speeds than the
distribution of the bins; each conserves its own rain rate.

```{code-cell} ipython3
depths = np.arange(0.0, 8001.0, 500.0)
d_bins = np.geomspace(0.15, 8.0, 60)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9), constrained_layout=True)
dm_axis = np.linspace(0.2, 5.0, 300)
ax1.plot(dm_axis, seifert.shape_parameter(dm_axis * 1e-3), color=OKABE["black"])
ax1.axvline(1.1, color="0.5", lw=0.8)
ax1.set(xlabel="mean volume diameter (mm)", ylabel="shape parameter", title="Eq. (20) of Seifert (2008)")
for (dm0, lwc, shape_mu), color in zip(((2.5, 2.0, 3.0), (4.0, 3.0, 6.0), (0.6, 1.0, 3.0)),
                                       (OKABE["blue"], OKABE["orange"], OKABE["green"])):
    lam = ((shape_mu + 3) * (shape_mu + 2) * (shape_mu + 1)) ** (1 / 3) / dm0
    shape = d_bins**shape_mu * np.exp(-lam * d_bins)
    mass = np.pi / 6 * 1e-3 * d_bins**3 * shape
    scale = lwc / np.sum(0.5 * (mass[1:] + mass[:-1]) * np.diff(d_bins))
    top = xr.DataArray(shape * scale, dims="diameter", coords={"diameter": d_bins})
    bins = rain_shaft(top, depth=depths)
    dm_bin = (6e-3 * bins.liquid_water_content / (np.pi * 1000 * bins.number_concentration)) ** (1 / 3) * 1e3
    bulk = rain_shaft_bulk(bins.isel(depth=0)[["number_concentration", "liquid_water_content"]], depth=depths)
    ax2.plot(depths / 1e3, dm_bin, color=color, label=f"bins, top {dm0:g} mm")
    ax2.plot(depths / 1e3, bulk.mean_volume_diameter, "--", color=color)
    print(f"top {dm0:g} mm, {lwc:g} g/m3: after 8 km bins {float(dm_bin[-1]):.2f} mm, bulk "
          f"{float(bulk.mean_volume_diameter[-1]):.2f} mm; rain rate bins {float(bins.rain_rate[-1]):.2f}, "
          f"bulk {float(bulk.rain_rate[-1]):.2f} mm/h")
ax2.axhline(1.1, color="0.5", lw=0.8)
ax2.set(xlabel="distance below the top (km)", ylabel="mean volume diameter (mm)",
        title="solid: bin model (Low and List), dashed: two moments")
ax2.legend(frameon=False, fontsize=8)
plt.show()
```

## References

- Seifert, A., 2008: On the parameterization of evaporation of raindrops as
  simulated by a one-dimensional rainshaft model. *J. Atmos. Sci.*, **65**,
  3608–3619, <https://doi.org/10.1175/2008JAS2586.1>
