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

# Doppler Velocity Dealiasing

+++

A Doppler radar measures radial velocity only within the Nyquist interval
$[-V_n, V_n]$. Faster winds fold over: a true velocity $v$ is reported as
$v - 2kV_n$ for some integer fold $k$. Dealiasing finds $k$ for every gate,
which is needed before any wind retrieval.

`radarx.retrieve.dealias_velocity` (or `dtree.radarx.dealias()`) uses a
region-based method on the polar sweep grid:

1. neighbouring gates with similar velocities are joined into regions
   (union-find), so fold lines separate regions (Jing and Wiener 1993);
2. the integer fold of every region minimises the squared velocity jumps
   across all region boundaries, by coordinate descent on single regions and
   on blocks of regions;
3. the absolute fold is fixed by the sweep below (volume continuity, as in
   James and Houze 2001), an optional wind profile (Eilts and Smith 1990), or
   an in-sweep VAD fit, and a final check refolds isolated gates.

A multithreaded C++ kernel does the work for all sweeps of a volume in one
call; an equivalent NumPy implementation gives identical results.

```{code-cell} ipython3
import time

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xradar as xd
from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

import radarx  # noqa: F401
from radarx.io.aws_data import download_file
```

## A strongly aliased case

The KGWX (Columbus, Mississippi) volume from 30 March 2022 shows a strong
low-level jet, with radial velocities well beyond the Nyquist velocity of
26.4 m/s.

```{code-cell} ipython3
path = download_file(
    "unidata-nexrad-level2", "2022/03/30/KGWX/KGWX20220330_234639_V06", "."
)
dtree = xd.io.open_nexradlevel2_datatree(path)
```

xradar does not yet expose the Nyquist velocity of NEXRAD data, so we read it
from the radial headers and attach it to each sweep as the
`nyquist_velocity` coordinate, where radarx looks for it. (It can also be
passed as `nyquist_velocity=...`.)

```{code-cell} ipython3
with NEXRADLevel2File(path) as nf:
    nyquist = [
        h["msg_31_data_header"]["RAD"]["nyquist_vel"] / 100.0
        for h in nf.msg_31_data_header
    ]
sweeps = [name for name in dtree.children if name.startswith("sweep")]
for name, value in zip(sweeps, nyquist):
    dtree[name] = dtree[name].to_dataset().assign_coords(nyquist_velocity=value)
dtree = dtree.xradar.georeference()
```

## Dealias the volume

Like every radarx retrieval, `dealias` returns its products only: a DataTree
with the root of the volume and, for each sweep, the dealiased velocity
`VRADH_dealiased` (the name never replaces the measured `VRADH`).
`dtree.radarx.assign` adds the products to the matching sweeps of the volume.

```{code-cell} ipython3
start = time.perf_counter()
products = dtree.radarx.dealias("VRADH")
print(f"dealiased the volume in {time.perf_counter() - start:.2f} s")
dealiased = dtree.radarx.assign(products)
dealiased["sweep_1"]
```

```{code-cell} ipython3
def residual_jumps(values, nyquist):
    """Share of neighbouring gate pairs that differ by more than Vn."""
    bad = total = 0
    for a, b in ((values[:, 1:], values[:, :-1]), (np.roll(values, -1, 0), values)):
        ok = np.isfinite(a) & np.isfinite(b)
        bad += np.count_nonzero(np.abs(a - b)[ok] > nyquist)
        total += np.count_nonzero(ok)
    return bad / total


for name in sweeps:
    ds = dealiased[name].to_dataset()
    if "VRADH" not in ds:
        continue
    vn = float(ds.nyquist_velocity)
    raw = ds.VRADH.where(abs(ds.VRADH) <= vn).values
    print(
        f"{name:9s} elevation {float(ds.elevation.median()):5.2f}  "
        f"jumps > Vn: raw {residual_jumps(raw, vn):.4f}  "
        f"dealiased {residual_jumps(ds.VRADH_dealiased.values, vn):.5f}"
    )
```

## Before and after

Gates without data, or flagged as range folded, are left blank.

```{code-cell} ipython3
def plot_sweep(name, extent):
    ds = dealiased[name].to_dataset().sortby("azimuth")
    vn = float(ds.nyquist_velocity)
    raw = ds.VRADH.where(abs(ds.VRADH) <= vn)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, da, title in zip(
        axes,
        [raw, ds.VRADH_dealiased],
        [f"measured (Nyquist {vn:.1f} m/s)", "dealiased"],
    ):
        mesh = da.assign_coords(x=ds.x / 1e3, y=ds.y / 1e3).plot.pcolormesh(
            x="x", y="y", ax=ax, cmap="RdBu_r", vmin=-60, vmax=60, add_colorbar=False
        )
        ax.set_title(title)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("")
        ax.set_aspect("equal")
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
    axes[0].set_ylabel("y (km)")
    fig.colorbar(mesh, ax=axes, label="radial velocity (m/s)")
    fig.suptitle(f"KGWX {name}, elevation {float(ds.elevation.median()):.1f} deg")
    plt.show()


plot_sweep("sweep_1", 250)
```

```{code-cell} ipython3
plot_sweep("sweep_21", 60)
```

## Reference winds

Without a reference, the absolute fold of the lowest sweep comes from the
assumption that the mean radial velocity of a sweep is close to zero, which
holds for a horizontally uniform wind seen all around the radar. When the
echo covers only part of the circle, pass a wind profile (sounding, VAD or
model) as an `xarray.Dataset` with `u` and `v` on a `height` coordinate:

```{code-cell} ipython3
import xarray as xr

height = np.arange(0.0, 15e3, 500.0)
profile = xr.Dataset(
    {"u": ("height", np.full(height.size, 15.0)), "v": ("height", np.full(height.size, 20.0))},
    coords={"height": height},
)
with_profile = dtree.radarx.dealias("VRADH", wind_profile=profile)
```

## Where the fold choice is uncertain

`confidence=True` returns, next to the dealiased velocity, a confidence in the
fold chosen for every gate (also available on its own as
`dtree.radarx.dealias_confidence`, for the result of any dealiasing method).
A gate could have been unfolded to $v \pm 2V_n$ instead. A neighbouring gate is
closer to the chosen value than to the next candidate if it differs by less
than $V_n$, and the confidence is the share of the valid neighbours (the 3 x 3
gates around the gate) that do:

- 1 where all neighbours support the fold, as in a smooth field;
- 5/8 on both sides of a straight fold line, 3/8 at the corner of a region;
- 0 for an isolated gate that is unfolded differently from all its neighbours.

It is a radarx definition (no paper defines a confidence for these algorithms)
and it is computed from the result alone. It marks isolated gates, small
clusters and the lines along which the fold changes; a large region that was
unfolded wrongly as a whole has confidence 1 inside and is flagged only along
its edge. Real shear and turbulence also lower it where the velocity changes by
more than $V_n$ from gate to gate.

```{code-cell} ipython3
with_confidence = dtree.radarx.dealias("VRADH", confidence=True)
node = with_confidence["sweep_1"].to_dataset()
conf = node.VRADH_dealiased_confidence
print(f"gates with data: {int(conf.notnull().sum())}")
for level in (1.0, 0.9, 0.5):
    print(f"confidence >= {level}: {float((conf >= level).sum() / conf.notnull().sum()):.2%}")
print(f"confidence below 0.5: {int((conf < 0.5).sum())} gates, "
      f"of which {int((conf == 0).sum())} disagree with all their neighbours")
```

```{code-cell} ipython3
ds = dealiased["sweep_1"].to_dataset().sortby("azimuth")
conf_sorted = conf.sortby("azimuth")
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, da, title, kw in zip(
    axes,
    [ds.VRADH_dealiased, conf_sorted],
    ["dealiased velocity (m/s)", "confidence of the fold"],
    [dict(cmap="RdBu_r", vmin=-60, vmax=60), dict(cmap="viridis", vmin=0, vmax=1)],
):
    mesh = da.assign_coords(x=ds.x / 1e3, y=ds.y / 1e3).plot.pcolormesh(
        x="x", y="y", ax=ax, add_colorbar=True, **kw
    )
    ax.set(title=title, xlabel="x (km)", ylabel="", aspect="equal", xlim=(-150, 150), ylim=(-150, 150))
axes[0].set_ylabel("y (km)")
plt.show()
```

## Dual-PRF radars

A dual-PRF radar unfolds the velocity of each ray with a noisy estimate and
leaves isolated dealiasing errors of a different kind; they are corrected with
`correct_dual_prf` (Holleman and Beekhuis 2003), shown on synthetic data in
section 13 of the [second workflow notebook](Radar_Workflow_Advanced).

## UNRAVEL

`method="unravel"` runs the UNRAVEL algorithm of Louf et al. (2020), which needs
no reference wind. The paper gives the algorithm as modules that a driver calls
in order: a modified z-score filter that also unfolds the edges of aliased
areas, two reference radials found where the mean $|v|$ is smallest, azimuthal
continuity in four sectors from them with $\alpha = 0.6$, range and azimuthal
continuity and box passes with growing windows against the mean or median of
the gates processed so far, a closest-reference module for what is left, a
least-squares check along each ray and a box check, and for a volume a check
against the sweep below ($\alpha = 0.8$ in the later modules). The module
documentation of `radarx.retrieve.dealias` lists the steps, which values are the
paper's and which radarx chose where the paper is silent. What was not
implemented: the "long range" driver, the box that precedes the gate and the
reflectivity gate filter. A compiled kernel runs the sweeps in parallel; a
Python specification gives identical values.

Because it assumes that the reference radials are not aliased, UNRAVEL is
built for a Nyquist velocity that is larger than the winds seen on the radials
of lowest $|v|$; a field folded more than once at those radials is out of its
reach. On the KGWX volume the two methods agree where both have a value:

```{code-cell} ipython3
start = time.perf_counter()
unraveled = dtree.radarx.assign(dtree.radarx.dealias("VRADH", method="unravel"))
print(f"UNRAVEL dealiased the volume in {time.perf_counter() - start:.1f} s")
for name in ("sweep_1", "sweep_8"):
    a = dealiased[name].to_dataset().VRADH_dealiased
    b = unraveled[name].to_dataset().VRADH_dealiased
    both = a.notnull() & b.notnull()
    print(f"{name}: UNRAVEL keeps {float(b.notnull().sum() / a.notnull().sum()):.1%} of the gates of the "
          f"region method; they agree within 1 m/s on {float((abs(a - b) < 1).where(both).mean()):.2%}")
```

The test of the UNRAVEL paper folds an unaliased volume into a smaller Nyquist
interval and counts the gates that a method does not bring back. Here the
Doppler cuts up to 4°, dealiased at their own Nyquist velocity (about 26 m/s)
by the region method, are folded into ±13.3 m/s, the Nyquist velocity of a
C-band radar, and both methods unfold them again; a gate is an error if it
differs from the reference by more than 1 m/s:

```{code-cell} ipython3
nodes = {"/": dtree.to_dataset()}
for name in sweeps:
    ds = dealiased[name].to_dataset()
    if "VRADH" in ds and float(ds.elevation.median()) < 4.1:
        folded = (ds.VRADH_dealiased + 13.3) % 26.6 - 13.3
        nodes["/" + name] = ds[["VRADH_dealiased"]].assign(VRADH=folded).assign_coords(nyquist_velocity=13.3)
c_band = xr.DataTree.from_dict(nodes)
rows = {}
for label, kwargs in (("region", {}), ("UNRAVEL", dict(method="unravel")),
                      ("UNRAVEL, 2-D only", dict(method="unravel", sweep_continuity=False))):
    start = time.perf_counter()
    result = c_band.radarx.dealias("VRADH", **kwargs)
    errors = total = 0
    for name in result.children:
        got = result[name].to_dataset().VRADH_dealiased
        ref = c_band[name].to_dataset().VRADH_dealiased
        both = got.notnull() & ref.notnull()
        errors += int(((abs(got - ref) > 1.0) & both).sum())
        total += int(both.sum())
    rows[label] = (errors / total, time.perf_counter() - start)
for label, (rate, seconds) in rows.items():
    print(f"{label:18s} error rate {rate:.2%} in {seconds:.1f} s")
```

The numbers depend strongly on the volume. On the four KGWX volumes of this
case, with one cut per elevation (the validation scripts of radarx), the error
rate of this test was 1 to 22 % for the region method, 4 to 39 % for UNRAVEL and
29 to 58 % for UNRAVEL without the check between sweeps; UNRAVEL did better
than the region method on two of the four and worse on the other two, and it
dropped about 4 % of the gates as noise. Its result depends on the absolute fold
of the lowest sweep, which the reference radials fix, and the check between
sweeps carries it upward: when the lowest sweep is right the others are
repaired (from 33 % to 4 % on one volume), when it is wrong they follow it. The
volume above, with its repeated 0.5° cuts, is a fifth sample. The error rate of
the paper (0.2 % on its test) is not reproduced on this case: the winds of the
squall line fold up to four times at 13.3 m/s, which is a harder test than the
paper's, and radarx fixed the details that the paper leaves open without
tuning them to this data.

## References

- Jing, Z., and G. Wiener, 1993: Two-dimensional dealiasing of Doppler
  velocities. *J. Atmos. Oceanic Technol.*, **10**, 798–808,
  <https://doi.org/10.1175/1520-0426(1993)010<0798:TDDODV>2.0.CO;2>
- Eilts, M. D., and S. D. Smith, 1990: Efficient dealiasing of Doppler
  velocities using local environment constraints. *J. Atmos. Oceanic
  Technol.*, **7**, 118–128,
  <https://doi.org/10.1175/1520-0426(1990)007<0118:EDODVU>2.0.CO;2>
- James, C. N., and R. A. Houze, 2001: A real-time four-dimensional Doppler
  dealiasing scheme. *J. Atmos. Oceanic Technol.*, **18**, 1674–1683,
  <https://doi.org/10.1175/1520-0426(2001)018<1674:ARTFDD>2.0.CO;2>
- Holleman, I., and H. Beekhuis, 2003: Analysis and correction of dual PRF
  velocity data. *J. Atmos. Oceanic Technol.*, **20**, 443–453,
  <https://doi.org/10.1175/1520-0426(2003)20<443:AACODP>2.0.CO;2>
- Louf, V., A. Protat, R. C. Jackson, S. M. Collis, and J. Helmus, 2020: UNRAVEL:
  A robust modular velocity dealiasing technique for Doppler radar. *J. Atmos.
  Oceanic Technol.*, **37**, 741–758,
  <https://doi.org/10.1175/JTECH-D-19-0020.1>
- Browning, K. A., and R. Wexler, 1968: The determination of kinematic
  properties of a wind field using Doppler radar. *J. Appl. Meteor.*, **7**,
  105–113, <https://doi.org/10.1175/1520-0450(1968)007<0105:TDOKPO>2.0.CO;2>
