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

# Dealiasing Alternatives and the Fold Margin

+++

`dealias_velocity` has two methods, `"region"` (the default) and `"unravel"`. This page adds a
third, R2D2 (`dealias_r2d2`, Feldmann et al. 2020), which was designed for strong shear between
neighbouring gates, and two ways to see where the choice of the fold is uncertain:
`dealias_ambiguity`, which gives the evidence for the fold of every region and flags the gates where
other dealiasings disagree.

The case is the lowest Doppler cut of the KBMX volume of 27 April 2011, 22:19 UTC (Tuscaloosa-Birmingham
tornado), in which the region method puts a large block of echo one fold (2 $V_n$) from the tilts above it.

```{code-cell} ipython3
import gzip
import shutil

import matplotlib.pyplot as plt
import numpy as np
import xradar as xd
from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

import radarx  # noqa: F401
from radarx.io.aws_data import download_file
from radarx.retrieve import dealias_ambiguity, dealias_r2d2, dealias_velocity

gz = download_file(
    "unidata-nexrad-level2", "2011/04/27/KBMX/KBMX20110427_221945_V03.gz", "."
)
path = gz[:-3]
with gzip.open(gz, "rb") as src, open(path, "wb") as dst:
    shutil.copyfileobj(src, dst)

with NEXRADLevel2File(path) as nf:
    nyquist = [
        h["msg_31_data_header"]["RAD"]["nyquist_vel"] / 100.0
        for h in nf.msg_31_data_header
    ]
dtree = xd.io.open_nexradlevel2_datatree(path)
sweeps = [n for n in dtree.children if n.startswith("sweep")]
for name, value in zip(sweeps, nyquist):
    ds = dtree[name].to_dataset().assign_coords(nyquist_velocity=value)
    if "VRADH" in ds:  # no-data codes of the archive (until xradar#457 is released)
        ds["VRADH"] = ds["VRADH"].where(ds["VRADH"] > -63.9)
    dtree[name] = ds
```

## Three dealiasings

R2D2 works from the highest sweep down. It cuts every sweep into regions inside one Nyquist interval
and shear zones, gives each region the fold that agrees with the sweep above (and a first guess), and
resolves the shear zones from their neighbours. Without a `reference` or `wind_profile` its first guess is the
region method, which decides only the regions that the sweep above does not reach.

```{code-cell} ipython3
region = dealias_velocity(dtree)
r2d2 = dealias_r2d2(dtree)
low = dtree["sweep_1"].to_dataset()
unravel = dealias_velocity(low, "VRADH", method="unravel")  # one sweep: UNRAVEL is slow
print(r2d2["sweep_1"].VRADH_dealiased.attrs["r2d2_iterations"], "iterations on sweep_1")
```

```{code-cell} ipython3
def residual_jumps(values, vn):
    """Share of neighbouring gate pairs that differ by more than Vn."""
    bad = total = 0
    for a, b in ((values[:, 1:], values[:, :-1]), (np.roll(values, -1, 0), values)):
        ok = np.isfinite(a) & np.isfinite(b)
        bad += np.count_nonzero(np.abs(a - b)[ok] > vn)
        total += np.count_nonzero(ok)
    return bad / total


def tilt_disagreement(out):
    """Share of gates where a tilt and the one above differ by more than Vn."""
    rows = []
    for a, b in zip(sweeps[:-1], sweeps[1:]):
        if "VRADH" not in dtree[a].data_vars or "VRADH" not in dtree[b].data_vars:
            continue
        va = out[a].VRADH_dealiased.sel(
            azimuth=dtree[b].azimuth, range=dtree[b].range, method="nearest"
        ).values
        vb = out[b].VRADH_dealiased.values
        vn = min(float(dtree[a].nyquist_velocity), float(dtree[b].nyquist_velocity))
        ok = np.isfinite(va) & np.isfinite(vb)
        rows.append(np.mean(np.abs(va[ok] - vb[ok]) > vn))
    return np.mean(rows)


for label, out in (("region", region), ("R2D2", r2d2)):
    jumps = np.mean(
        [
            residual_jumps(out[n].VRADH_dealiased.values, float(dtree[n].nyquist_velocity))
            for n in sweeps
            if "VRADH" in dtree[n].data_vars
        ]
    )
    print(
        f"{label:7s} neighbouring gates > Vn apart: {jumps:.4f}   "
        f"tilts a fold apart: {tilt_disagreement(out):.3f}"
    )
```

The jumps are a property of one sweep; the second number compares each tilt with the one above and is
the check that the lowest sweeps are folded consistently with the rest of the volume.

```{code-cell} ipython3
def panel(ax, da, title, **style):
    order = np.argsort(low.azimuth.values)  # increasing azimuth for pcolormesh
    az = np.deg2rad(low.azimuth.values[order])[:, None]
    rng = low.range.values[None, :] / 1e3
    mesh = ax.pcolormesh(
        rng * np.sin(az), rng * np.cos(az), np.asarray(da)[order], shading="auto", **style
    )
    ax.set(title=title, xlim=(-50, 100), ylim=(0, 150), aspect="equal")
    return mesh


fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True, layout="constrained")
style = dict(cmap="RdBu_r", vmin=-60, vmax=60)
panel(axes[0], low.VRADH.values, "measured", **style)
panel(axes[1], region["sweep_1"].VRADH_dealiased.values, "region", **style)
panel(axes[2], unravel.values, "UNRAVEL", **style)
mesh = panel(axes[3], r2d2["sweep_1"].VRADH_dealiased.values, "R2D2", **style)
fig.colorbar(mesh, ax=axes, label="radial velocity (m/s)")
fig.suptitle("KBMX 0.5° Doppler cut, 27 April 2011 22:19 UTC")
plt.show()
```

## Where the fold is uncertain

`dealias_ambiguity` returns the fold margin of every region of the dealiased velocity: the evidence
for its fold from the gates around it (the least-squares criterion of Jing and Wiener 1993, per
boundary pair), and flags for a weak margin, a short boundary and a region without boundary. The
margin sees contradictions in the field, such as a block one fold off inside a smooth wind:

```{code-cell} ipython3
amb = dealias_ambiguity(
    low.assign(VRADH_dealiased=region["sweep_1"].VRADH_dealiased), "VRADH_dealiased"
)
flag = amb.VRADH_dealiased_fold_flag.values
valid = amb.VRADH_dealiased_region_size.values > 0
for bit, name in ((1, "weak margin"), (2, "short boundary"), (4, "no boundary")):
    print(f"{name:15s} {np.mean((flag[valid] & bit) > 0):.3f} of the gates")
```

It cannot see an error that the field does not show. The block that the region method puts one fold
off is joined to the rest through a smooth passage, so it is one region and its margin is as high as
that of the rest. That kind of ambiguity shows as disagreement between dealiasings: `compare` takes
other results and flags the gates where one of them chooses another fold, with the number of folds as
the spread.

```{code-cell} ipython3
both = low.assign(
    region=region["sweep_1"].VRADH_dealiased,
    unravel=unravel,
    r2d2=r2d2["sweep_1"].VRADH_dealiased,
)
amb = dealias_ambiguity(both, "region", compare=["unravel", "r2d2"])
spread = amb.region_fold_spread.values
print(f"gates where UNRAVEL or R2D2 choose another fold than the region method: "
      f"{np.mean(spread[valid] > 0):.3f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, layout="constrained")
mesh = panel(axes[0], amb.region_fold_margin.values, "fold margin", cmap="RdYlGn", vmin=-1, vmax=1)
fig.colorbar(mesh, ax=axes[0], shrink=0.8)
mesh = panel(axes[1], np.where(valid, spread, np.nan), "folds by which the others differ", cmap="magma_r", vmin=0, vmax=2)
fig.colorbar(mesh, ax=axes[1], shrink=0.8)
mesh = panel(axes[2], np.where(valid, np.log10(np.maximum(amb.region_region_size.values, 1)), np.nan),
             "log10 gates in the region", cmap="viridis")
fig.colorbar(mesh, ax=axes[2], shrink=0.8)
plt.show()
```

## What is the paper's and what is radarx's

R2D2 follows the steps of Feldmann et al. (2020): shear zones with a 5 x 5 buffer, regions between them,
region folds from the guidance, continuity in the shear zones with relaxing thresholds, and the recursion until
the mean of the largest gate-to-gate difference stops decreasing. The thresholds of the scanning passes, the
spread threshold and the stopping counts are radarx's choices, and every decision is made in integers (1/256
m/s), so the compiled kernel and the NumPy reference give the same folds:

```{code-cell} ipython3
a = dealias_r2d2(low, "VRADH", engine="numpy")
b = dealias_r2d2(low, "VRADH", engine="compiled")
print("identical:", bool(np.array_equal(a.values, b.values, equal_nan=True)))
```

A first guess from a model or a sounding makes R2D2 the independent method of the paper:
`dealias_r2d2(dtree, wind_profile=profile)` or `reference=` for a sweep. The dual-PRF correction of Joe and
May (2003) is not implemented; `correct_dual_prf` is that of Holleman and Beekhuis (2003).

## References

Feldmann, M., C. N. James, M. Boscacci, D. Leuenberger, M. Gabella, U. Germann, D. Wolfensberger, and
A. Berne, 2020: R2D2: A region-based recursive Doppler dealiasing algorithm for operational weather radar.
*J. Atmos. Oceanic Technol.*, **37**, 2341-2356, https://doi.org/10.1175/JTECH-D-20-0054.1

Jing, Z., and G. Wiener, 1993: Two-dimensional dealiasing of Doppler velocities. *J. Atmos. Oceanic
Technol.*, **10**, 798-808, https://doi.org/10.1175/1520-0426(1993)010<0798:TDDODV>2.0.CO;2
