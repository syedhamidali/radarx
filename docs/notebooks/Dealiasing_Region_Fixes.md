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
from radarx.retrieve import dealias, dealias_velocity

warnings.filterwarnings("ignore", message="The input coordinates to pcolormesh")
```

## The data

The volume is read from the public NEXRAD archive on AWS. The Doppler cut at
0.5 degrees has a Nyquist velocity of 29.97 m/s. Gates without data are masked.
Without a network connection the notebook builds two small synthetic sweeps
with the same two problems and uses the true velocity as the reference where
the real case uses UNRAVEL.

```{code-cell} ipython3
def xy(ds):
    az = np.deg2rad(ds.azimuth.values)[:, None]
    r = ds.range.values[None, :] / 1000.0
    return r * np.sin(az), r * np.cos(az)


def kbmx_cases():
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
    reference = dealias_velocity(ds, "VRADH", method="unravel")
    x, y = xy(ds)
    block = (x > 0) & (x < 69) & (y > 52) & (y < 133)
    neck = dict(
        ds=ds, ref=reference.values, block=block, label="KBMX 27 April 2011, 0.5 degree cut",
        ref_name="UNRAVEL", window=(-110, 100, 20, 170),
    )
    couplet = dict(
        ds=ds, ref=reference.values, label="KBMX 27 April 2011, 0.5 degree cut",
        ref_name="UNRAVEL", window=(-110, 100, 20, 170),
        centre=(64.0 * np.sin(np.deg2rad(276.0)), 64.0 * np.cos(np.deg2rad(276.0))), half=7.0,
    )
    return neck, couplet


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


def synthetic_cases():
    # two blocks of one echo, 15 m/s apart in truth, joined by a two-ray neck
    vn, nray, ngate = 30.0, 360, 200
    rnd = np.random.default_rng(3)
    az, rng = np.arange(nray) + 0.5, 1000.0 + 250.0 * np.arange(ngate)
    truth = np.where(np.arange(ngate) < 100, 20.0, 35.0)[None, :] * np.ones((nray, 1))
    truth = truth + 0.4 * rnd.standard_normal(truth.shape)
    measured = (truth + vn) % (2 * vn) - vn
    measured[170:172, 96:104] = np.linspace(20.0, 35.0 - 2 * vn, 8)  # the neck
    ds = sweep(measured, vn, nray, az, rng)
    x, y = xy(ds)
    neck = dict(
        ds=ds, ref=truth, block=np.hypot(x, y) > 27.0, label="synthetic sweep with a neck",
        ref_name="truth", window=(-51, 51, -51, 51),
    )

    # a vortex core inside a folded patch, as in the Dealiasing notebook
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
    couplet = dict(
        ds=ds, ref=truth, label="synthetic vortex", ref_name="truth",
        window=(-27, 27, -27, 27), centre=(r0 * np.sin(a0), r0 * np.cos(a0)), half=6.0,
    )
    return neck, couplet


try:
    case_neck, case_couplet = kbmx_cases()
except Exception as err:  # offline or archive unavailable
    print(f"no NEXRAD data ({type(err).__name__}), using synthetic sweeps")
    case_neck, case_couplet = synthetic_cases()
print(case_neck["label"])
```

## The thin neck

At a fold line next to a vortex, a neck of two or three gates whose velocities
pass through zero is enough to link the block north of the radar to its western
neighbour by a chain of small steps. Both ended up in one region of 57 892
gates. Inside a region the fold is constant by construction, so the fold line
between the blocks was never looked at.

Now only core gates (joined to all four neighbours, in a core of at least 1000
connected gates) connect regions. Every other gate takes, ring by ring, the
region of the joined neighbour closest in velocity. Two blocks joined by a neck
stay two regions, and the boundary between them fixes their relative fold.

The old segmentation is still in the NumPy engine behind the module constant
`_CORE_JOINS` (0 restores it). The cell sets it to 0 for the "before" panel and
back afterwards; both results come from the NumPy engine.

```{code-cell} ipython3
ds = case_neck["ds"]
vn = float(ds.nyquist_velocity)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # the vortex warning is the subject of the next section
    dealias._CORE_JOINS = 0
    try:
        before = dealias_velocity(ds, "VRADH", engine="numpy").values
    finally:
        dealias._CORE_JOINS = 4
    after = dealias_velocity(ds, "VRADH", engine="numpy").values
ref = case_neck["ref"]


def beyond_vn(a, b, mask=None):
    ok = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        ok &= mask
    return int(np.count_nonzero(np.abs(a - b)[ok] > vn)), int(np.count_nonzero(ok))


ref_name = case_neck["ref_name"]
for label, values in (("before", before), ("after", after)):
    bad, total = beyond_vn(values, ref)
    bad_b, total_b = beyond_vn(values, ref, case_neck["block"])
    print(
        f"{label:6s} gates more than Vn from {ref_name}: {bad:6d} of {total}; "
        f"in the block {100 * (1 - bad_b / total_b):5.1f} % within Vn"
    )
```

```{code-cell} ipython3
x, y = xy(ds)
x0, x1, y0, y1 = case_neck["window"]
fig, axs = plt.subplots(1, 4, figsize=(18, 4.8), sharex=True, sharey=True, constrained_layout=True)
panels = (
    ("measured", ds.VRADH.values),
    ("before: one region", before),
    ("after: two regions", after),
    (ref_name, ref),
)
for ax, (title, values) in zip(axs, panels):
    pm = ax.pcolormesh(x, y, values, cmap="RdBu_r", vmin=-60, vmax=60, shading="auto")
    ax.contour(x, y, case_neck["block"].astype(float), levels=[0.5], colors="k", linewidths=0.7, linestyles="--")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("x (km)")
axs[0].set_ylabel("y (km)")
fig.colorbar(pm, ax=axs, label="radial velocity (m/s)", shrink=0.8)
fig.suptitle(case_neck["label"])
plt.show()
```

The dashed box is the block of the issue. Before, it keeps the folded (blue)
velocities of the measurement; after, it is one fold up and agrees with UNRAVEL.
On the KBMX cut the share of the block within one Nyquist velocity of UNRAVEL
rises from 1.6 % to 99.9 %. Over the whole cut the gates more than $V_n$ from
UNRAVEL drop from 29 539 to 9 667. What is left is a separate echo at 250 km
range whose absolute fold has no support in the rest of the sweep.

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
ds = case_couplet["ds"]
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    default = dealias_velocity(ds, "VRADH").values
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
limited = dealias_velocity(ds, "VRADH", max_speed=36.0).values
ref = case_couplet["ref"]
x, y = xy(ds)
cx, cy = case_couplet["centre"]
half = case_couplet["half"]
near = (abs(x - cx) < half - 2) & (abs(y - cy) < half - 2)
panels = (
    ("measured", ds.VRADH.values),
    ("default", default),
    ("max_speed=36", limited),
    (case_couplet["ref_name"], ref),
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
fig.suptitle(case_couplet["label"])
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
