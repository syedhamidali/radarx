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

# Region Dealiasing: Thin Necks and Vortex Cores

The region method of `dealias_velocity` joins gates with similar velocities
into regions and gives every region one fold. Two cases on the lowest Doppler
cut of KBMX (Birmingham, Alabama, 27 April 2011, 22:19 UTC, the day of the
Alabama tornado outbreak) went wrong with it, and both are fixed or reported
in the current version:

1. A thin neck of in-between velocities used to join two blocks that are a
   fold apart into one region, so the fold line between them was never looked
   at and one block stayed one fold off. Only core gates connect regions now.
2. The core of a velocity couplet can still be shifted by a fold together with
   a folded region beside it. A sweep alone cannot tell this from a strong
   wind, so the values stay, but the result is checked and a warning names the
   gates and `max_speed` limits them.

The general method is described in the [Dealiasing](Dealiasing) notebook.

```{code-cell} ipython3
import gzip
import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xradar as xd
from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

import radarx  # noqa: F401
from radarx.io.aws_data import download_file
from radarx.retrieve import dealias_velocity

warnings.filterwarnings("ignore", message="The input coordinates to pcolormesh")
```

## The data

The real case is read from the public NEXRAD archive on AWS: the Doppler cut
at 0.5 degrees of KBMX, Nyquist velocity 29.97 m/s, with the no-data codes
masked. The synthetic sweeps are built here and need no network; without a
connection the notebook shows them only.

```{code-cell} ipython3
def xy(ds):
    az = np.deg2rad(ds.azimuth.values)[:, None]
    r = ds.range.values[None, :] / 1000.0
    return r * np.sin(az), r * np.cos(az)


def sweep(measured, nyquist, nray, azimuth, rng):
    return xr.Dataset(
        {"VRADH": (("azimuth", "range"), measured, {"units": "m s-1"})},
        coords={
            "azimuth": ("azimuth", azimuth),
            "range": ("range", rng),
            "elevation": ("azimuth", np.full(nray, 0.5)),
            "nyquist_velocity": ((), nyquist),
        },
    )


def load_kbmx():
    packed = download_file(
        "unidata-nexrad-level2", "2011/04/27/KBMX/KBMX20110427_221945_V03.gz", "."
    )
    with gzip.open(packed) as src, open("KBMX20110427_221945_V03", "wb") as dst:
        shutil.copyfileobj(src, dst)
    path = "KBMX20110427_221945_V03"
    with NEXRADLevel2File(path) as nf:
        nyquist = [
            h["msg_31_data_header"]["RAD"]["nyquist_vel"] / 100.0
            for h in nf.msg_31_data_header
        ]
    dtree = xd.io.open_nexradlevel2_datatree(path)
    ds = dtree["sweep_1"].to_dataset().assign_coords(nyquist_velocity=nyquist[1])
    ds["VRADH"] = ds.VRADH.where(ds.VRADH > -63.9)  # no-data codes
    return ds


def synthetic_neck():
    # two blocks of one echo, 15 m/s apart in truth, joined by a two-ray neck
    vn, nray, ngate = 30.0, 360, 200
    rnd = np.random.default_rng(3)
    az, rng = np.arange(nray) + 0.5, 1000.0 + 250.0 * np.arange(ngate)
    truth = np.where(np.arange(ngate) < 100, 20.0, 35.0)[None, :] * np.ones((nray, 1))
    truth = truth + 0.4 * rnd.standard_normal(truth.shape)
    measured = (truth + vn) % (2 * vn) - vn
    measured[170:172, 96:104] = np.linspace(20.0, 35.0 - 2 * vn, 8)  # the neck
    return sweep(measured, vn, nray, az, rng), truth


def synthetic_couplet():
    # a vortex core inside a folded patch, as in the Dealiasing notebook
    vn = 30.0
    ray, gate = np.meshgrid(np.arange(120), np.arange(100), indexing="ij")

    def box(lo, hi, v, width=1.5):
        return 0.5 * (np.tanh((v - lo) / width) - np.tanh((v - hi) / width))

    truth = np.full(ray.shape, -28.5) + (-34.0 + 28.5) * box(54, 66, ray) * box(38, 46, gate)
    rho = np.hypot((ray - 60) / 4.0, (gate - 50) / 6.0)
    rim = -12.0 + (28.5 + 12.0) * 0.5 * (1 - np.tanh((ray - 61.0) / 0.6))
    truth = np.where(rho <= 1, -12.0 + (rim + 12.0) * np.minimum(rho, 1.0) ** 2, truth)
    measured = (truth + vn) % (2 * vn) - vn
    ds = sweep(measured, vn, 120, (np.arange(120) + 0.5) * 3.0, 2000.0 + 250.0 * np.arange(100))
    r0, a0 = 2.0 + 0.25 * 50, np.deg2rad(181.5)
    return ds, truth, (r0 * np.sin(a0), r0 * np.cos(a0))


try:
    kbmx = load_kbmx()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unravel = dealias_velocity(kbmx, "VRADH", method="unravel").values
except Exception as err:  # offline or archive unavailable
    print(f"no NEXRAD data ({type(err).__name__}), the real case is skipped")
    kbmx = unravel = None
```

## The thin neck

At a fold line next to a vortex, a neck of two or three gates whose velocities
pass through zero is enough to link two blocks that are a fold apart by a chain
of small steps. They used to end up in one region (57 892 gates on the KBMX
cut). Inside a region the fold is constant by construction, so the fold line
between the blocks was never looked at and one block stayed one fold off.

Now only core gates (joined to all four neighbours, in a core of at least 1000
connected gates) connect regions. Every other gate takes, ring by ring, the
region of the joined neighbour closest in velocity. Two blocks joined by a neck
stay two regions, and the boundary between them fixes their relative fold.

The synthetic sweep below is the test case of this fix (`tests/test_dealias_bridge.py`).
The west block is at 20 m/s and the east block at 35 m/s, which the Nyquist
velocity of 30 m/s folds to -25 m/s. The two rays of the neck pass through zero in
steps below $0.3\,V_n$. The old segmentation made one region of the two blocks
and left the east block one fold off (the test fails under it). Without any
reference the boundary alone decides now:

```{code-cell} ipython3
ds_neck, truth_neck = synthetic_neck()
neck_out = dealias_velocity(ds_neck, "VRADH")
west, east = np.s_[:, 10:90], np.s_[:, 110:]
for label, values in (("measured", ds_neck.VRADH.values), ("dealiased", neck_out.values),
                      ("truth", truth_neck)):
    print(f"{label:10s} median west {np.nanmedian(values[west]):6.1f} m/s, "
          f"east {np.nanmedian(values[east]):6.1f} m/s")
within = np.abs(neck_out.values - truth_neck) < 30.0
print(f"gates within Vn of the truth: {100 * np.mean(within):.1f} %")
```

```{code-cell} ipython3
x, y = xy(ds_neck)
fig, axs = plt.subplots(1, 3, figsize=(14, 4.6), sharex=True, sharey=True, constrained_layout=True)
for ax, (title, values) in zip(axs, (("measured", ds_neck.VRADH.values),
                                     ("dealiased", neck_out.values), ("truth", truth_neck))):
    pm = ax.pcolormesh(x, y, values, cmap="RdBu_r", vmin=-60, vmax=60, shading="auto")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
axs[0].set_ylabel("y (km)")
fig.colorbar(pm, ax=axs, label="radial velocity (m/s)", shrink=0.8)
fig.suptitle("Synthetic sweep, two blocks joined by a thin neck")
plt.show()
```

The east block is one fold above the measurement and 15 m/s above the west block,
as in the truth.

### The same on KBMX

On the 0.5 degree cut of KBMX the block north of the radar (x 0 to 69 km, y 52
to 133 km, dashed) is the one of the issue. The current result is compared with
UNRAVEL, which needs no reference wind.

```{code-cell} ipython3
if kbmx is not None:
    vn = float(kbmx.nyquist_velocity)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the vortex warning is the subject of the next section
        result = dealias_velocity(kbmx, "VRADH").values
    x, y = xy(kbmx)
    block = (x > 0) & (x < 69) & (y > 52) & (y < 133)
    both = np.isfinite(result) & np.isfinite(unravel)
    far = (np.abs(result - unravel) > vn) & both
    print(f"gates more than Vn from UNRAVEL: {far.sum()} of {both.sum()}; "
          f"in the block {100 * (1 - far[block].sum() / both[block].sum()):.1f} % within Vn")
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8), sharex=True, sharey=True, constrained_layout=True)
    for ax, (title, values) in zip(axs, (("measured", kbmx.VRADH.values),
                                         ("region, current", result), ("UNRAVEL", unravel))):
        pm = ax.pcolormesh(x, y, values, cmap="RdBu_r", vmin=-60, vmax=60, shading="auto")
        ax.contour(x, y, block.astype(float), levels=[0.5], colors="k", linewidths=0.7, linestyles="--")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(-110, 100)
        ax.set_ylim(20, 170)
        ax.set_xlabel("x (km)")
    axs[0].set_ylabel("y (km)")
    fig.colorbar(pm, ax=axs, label="radial velocity (m/s)", shrink=0.8)
    fig.suptitle("KBMX 27 April 2011, 0.5 degree cut")
    plt.show()
else:
    print("skipped, no NEXRAD data")
```

With the neck cut the block is one fold above its measurement and agrees with
UNRAVEL: 99.9 % of its gates are within one Nyquist velocity. Before the fix only
1.6 % were, and 29 539 gates of the cut were more than $V_n$ from UNRAVEL against
9 667 now (the numbers of the pull request). What is left is a separate echo at
250 km range whose absolute fold has no support in the rest of the sweep.

The fix changes the default result wherever such a neck occurred. On nine NEXRAD
volumes five were unchanged or changed in fewer than 20 gates, and four changed
in 5 000 to 173 000 gates, never with more residual jumps than before.

## The vortex core and `max_speed`

The second case is on the same cut, at azimuth 276 degrees and 64 km range,
where a compact vortex signature sits next to a folded region. The region
method can shift the whole pocket by one fold, the core included, so the core
reads far above any wind the storm could hold. The sweep cannot tell this from
a strong wind, so the default leaves the values and checks the result: a block
of at most 200 connected gates that the folds shifted by the same single fold,
with 10 or more gates beyond 2 $V_n$ and 50 m/s, raises a `UserWarning`.

```{code-cell} ipython3
if kbmx is not None:
    ds_c, ref_c, name_c = kbmx, unravel, "UNRAVEL"
    cx, cy = 64.0 * np.sin(np.deg2rad(276.0)), 64.0 * np.cos(np.deg2rad(276.0))
    half, label_c = 7.0, "KBMX 27 April 2011, 0.5 degree cut"
else:
    ds_c, ref_c, centre = synthetic_couplet()
    cx, cy = centre
    name_c, half, label_c = "truth", 6.0, "synthetic vortex"
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    default = dealias_velocity(ds_c, "VRADH").values
for w in caught:
    if issubclass(w.category, UserWarning):
        print("UserWarning:", w.message)
```

`max_speed` is the largest radial speed the storm can have. Gates the folds put
beyond it go back by one interval where that lowers their jumps to the less
folded neighbours, and the warning is skipped. 36 m/s is a limit chosen for this
case. The guard is described in the section "Vortex cores and the speed limit"
of the [Dealiasing](Dealiasing) notebook.

```{code-cell} ipython3
limited = dealias_velocity(ds_c, "VRADH", max_speed=36.0).values
x, y = xy(ds_c)
near = (abs(x - cx) < half - 2) & (abs(y - cy) < half - 2)
panels = (
    ("measured", ds_c.VRADH.values),
    ("default", default),
    ("max_speed=36", limited),
    (name_c, ref_c),
)
fig, axs = plt.subplots(1, 4, figsize=(17, 4.2), sharex=True, sharey=True, constrained_layout=True)
for ax, (title, values) in zip(axs, panels):
    pm = ax.pcolormesh(x, y, values, cmap="RdBu_r", vmin=-90, vmax=90, shading="auto")
    shown = values[near & np.isfinite(values)]
    ax.set_title(f"{title}\nmost negative / positive {shown.min():.0f} / {shown.max():.0f} m/s", fontsize=9)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
axs[0].set_ylabel("y (km)")
fig.colorbar(pm, ax=axs, label="radial velocity (m/s)", shrink=0.8)
fig.suptitle(label_c)
plt.show()
```

On KBMX the default gives -88 and +26 m/s in a 10 km box around the couplet; the
measured values are about -30 and +30 m/s. With `max_speed=36` the core is at
-36 and the pocket at +26 m/s, close to UNRAVEL (-37 and +30 m/s). The check
names exactly one pocket on this cut (13 gates, speeds up to 88 m/s) and none
with `max_speed=36`.

## Why the default stays "report"

No single limit separates the KBMX couplet from the strong winds of other cases.
On the KGWX squall line of 30 March 2022 a limit of 1.5 $V_n$ would change 1 136
to 1 263 gates per tilt, and UNRAVEL agrees with the unchanged fold on most of
them. So `max_speed=None` stays the default, the warning tells you when to look,
and you set the limit that fits your storm.
