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

# ZDR arc and KDP-ZDR separation of supercells

In the forward flank of a supercell the storm-relative wind sorts the raindrops by
size. The big drops fall out near the inflow edge and form an arc of high
differential reflectivity ($Z_{DR}$), and the small drops are carried deeper into
the storm, where the specific differential phase ($K_{DP}$) is enhanced. The
distance between the two, and the angle of that separation to the storm motion,
indicate the low-level shear (Wilson and Van Den Broeke 2021).

`zdr_arc` works on a grid of the lowest level. It finds the arc, the storm and the
$K_{DP}$ foot and returns, for every storm, the arc area, the mean and top-10
$Z_{DR}$ of the arc, and the length and angle of the separation vector. The
inputs are the fields on a grid with `x` to the east and `y` to the north, the
bearing of the forward-flank downdraft (FFD) vector, which points into the storm
perpendicular to the forward flank, and the bearing of the storm motion.

The data here are synthetic, the supercell of the unit tests: a Gaussian core of
reflectivity, a patch of high $Z_{DR}$ on its south (inflow) side, a second patch
of high $Z_{DR}$ on the north side, and a $K_{DP}$ foot inside the core. The
answer follows from the construction, so the numbers can be checked by hand.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.retrieve import zdr_arc

x = y = np.arange(0.0, 60001.0, 500.0)
XX, YY = np.meshgrid(x, y)
core = (XX - 30e3) ** 2 + (YY - 32e3) ** 2
south = (XX >= 26e3) & (XX <= 34e3) & (YY >= 20e3) & (YY <= 23e3)
north = (XX >= 26e3) & (XX <= 34e3) & (YY >= 41e3) & (YY <= 43e3)
supercell = xr.Dataset(
    {
        "DBZH": (("y", "x"), 60.0 * np.exp(-core / (2 * 9e3**2))),
        "ZDR": (("y", "x"), np.where(south | north, 4.0, 0.5)),
        "RHOHV": (("y", "x"), np.full(XX.shape, 0.98)),
        "KDP": (("y", "x"), np.where(np.hypot(XX - 30e3, YY - 29e3) <= 3e3, 2.5, 0.1)),
    },
    coords={"x": x, "y": y},
)
```

## The arc and the separation vector

The forward-flank downdraft points north (0°) and the storm moves east (90°). The
reflectivity gradient points to the core. On the south patch that is north, the
direction of the FFD vector, so the patch is on the inflow side; on the north
patch it points south, opposite to the vector, and the patch is masked. The foot is
7.5 km north of the arc, which is at 90° to the storm motion.

```{code-cell} ipython3
arc = zdr_arc(
    supercell, rhohv="RHOHV", kdp="KDP", ffd_direction=0.0, storm_motion=90.0
)
s = arc.isel(storm=0)
print(f"{arc.sizes['candidate']} candidate of the arc")
print(f"arc area {float(s.arc_area):.1f} km2, mean ZDR {float(s.arc_mean_zdr):.1f} dB, "
      f"top-10 ZDR {float(s.arc_top10_zdr):.1f} dB")
print(f"separation {float(s.separation_length):.1f} km at {float(s.separation_angle):.0f} degrees "
      "to the storm motion")
# the accessor gives the same
same = supercell.radarx.zdr_arc(
    rhohv="RHOHV", kdp="KDP", ffd_direction=0.0, storm_motion=90.0
)
print(bool(same.arc_area[0] == s.arc_area))
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), layout="constrained")
km = lambda a: np.asarray(a) / 1e3
pm = axes[0].pcolormesh(km(x), km(y), supercell.ZDR, cmap="turbo", vmin=0, vmax=4.5, shading="auto")
fig.colorbar(pm, ax=axes[0], label="$Z_{DR}$ (dB)")
axes[0].set_title("Input: $Z_{DR}$ with 20 and 40 dBZ contours")
axes[0].contour(km(x), km(y), supercell.DBZH, levels=[20, 40], colors="k", linewidths=0.8)

axes[1].pcolormesh(km(x), km(y), supercell.DBZH, cmap="Greys", vmin=0, vmax=60, shading="auto")
axes[1].contour(km(x), km(y), arc.ZDR_ARC_STORM >= 0, levels=[0.5], colors="tab:red")
axes[1].contour(km(x), km(y), arc.KDP_FOOT_STORM >= 0, levels=[0.5], colors="tab:green")
axes[1].annotate(
    "",
    xy=(float(s.arc_x) / 1e3, float(s.arc_y) / 1e3),
    xytext=(float(s.foot_x) / 1e3, float(s.foot_y) / 1e3),
    arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2),
)
axes[1].set_title("Result: arc (red), $K_{DP}$ foot (green), separation vector")
for ax in axes:
    ax.set(xlabel="x (km)", ylabel="y (km)", aspect="equal")
plt.show()
```

The red contour is the arc on the south side. The north patch has the same
$Z_{DR}$, but the inflow mask removes it, and the blue arrow is the separation
vector from the centre of the foot to the centre of the arc.

## Which side is the inflow

The FFD vector decides which patch counts. With the vector turned around, the
north patch is on the inflow side and the south one is not:

```{code-cell} ipython3
flipped = zdr_arc(
    supercell, rhohv="RHOHV", kdp="KDP", ffd_direction=180.0, storm_motion=90.0
).isel(storm=0)
print(f"arc at y = {float(flipped.arc_y) / 1e3:.1f} km instead of {float(s.arc_y) / 1e3:.1f} km, "
      f"separation {float(flipped.separation_length):.1f} km at {float(flipped.separation_angle):.0f} degrees")
```

In a real case the FFD vector comes from the storm: perpendicular to the leading
edge of the forward flank, pointing into the storm.

## What the algorithm leaves out

The paper removes the candidates that are not part of the arc with a random
forest trained on hand-labelled objects. That model cannot be rebuilt from the
paper, so `classifier` takes a model of your own or a function of the features
(`radarx.retrieve.zdr_arc.FEATURES`). Without it every candidate near a storm is
part of the arc, which overestimates the arc where the rear flank has high
$Z_{DR}$. The algorithm is for supercells: the leading edge of a squall line
and the high $Z_{DR}$ below developing updrafts are often detected as arcs, and
$Z_{DR}$ must be calibrated, since the threshold of 3.25 dB depends on it.

## References

- Wilson, M. B., and M. S. Van Den Broeke, 2021: An Automated Python Algorithm to Quantify ZDR Arc and KDP-ZDR Separation Signatures in Supercells. *Journal of Atmospheric and Oceanic Technology*, **38**, 371-386, <https://doi.org/10.1175/JTECH-D-20-0056.1>
