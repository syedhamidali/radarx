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

# Hail products: SHI, MESH, POSH and hail kinetic energy

The hail detection algorithm of the WSR-88D
([Witt et al. 1998](https://doi.org/10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2))
turns the reflectivity profile of a storm and the heights of the 0 and −20 °C
levels into four hail products. They are in `radarx.retrieve`, on the same
columns as the VIL products (see the VIL notebook):

- `hail_kinetic_energy`: the kinetic energy flux of hail,
  $\dot E = 5\times10^{-6}\,10^{0.084Z}\,W(Z)$ in J m$^{-2}$ s$^{-1}$, a fit of
  Waldvogel et al. (1978) to hailstone spectra as written in Eq. 1 of Witt et al.
  (1998), with a weight $W(Z)$ that rises linearly from 0 at $Z_L=40$ dBZ to 1 at
  $Z_U=50$ dBZ;
- `shi`: the severe hail index, $\mathrm{SHI}=0.1\sum_i W_T(H_i)\,\dot E_i\,\Delta H_i$
  in J m$^{-1}$ s$^{-1}$, the flux integrated over height with a temperature weight
  $W_T$ that rises linearly from 0 at the 0 °C level to 1 at the −20 °C level;
- `mesh`: the maximum expected size of hail, $\mathrm{MESH}=2.54\,\mathrm{SHI}^{0.5}$ in mm;
- `posh`: the probability of severe hail,
  $\mathrm{POSH}=29\ln(\mathrm{SHI}/\mathrm{WT})+50$ in %, limited to 0 to 100, with the
  warning threshold $\mathrm{WT}=\max(57.5H_0-121,\,20)$ ($H_0$ in km above the radar).

`hail_products` returns them together. The column sums run in a compiled,
multithreaded kernel with a NumPy implementation that gives the same numbers.
The equations were read in the paper itself; the Waldvogel et al. (1978) papers
were not, and the probability of hail (POH) and the alternative MESH fits of
Murillo and Homeyer (2019) are not implemented (see the end of the notebook).

```{code-cell} ipython3
import warnings

import cmweather  # noqa: F401  radar colormaps
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xradar as xd

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.grid import grid_cones
from radarx.io import sounding
from radarx.io.aws_data import download_file
from radarx.retrieve import (
    hail_kinetic_energy,
    hail_products,
    mesh,
    posh,
    shi,
    vil,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
```

## The weights and the flux

The two weights and the flux are simple functions. The flux grows by a factor
of 10 for every 12 dBZ and is 3.8 J m$^{-2}$ s$^{-1}$ at 70 dBZ (Fig. 3 of the
paper); the weight $W(Z)$ removes the rain reflectivities. The SHI is large only
where strong reflectivity sits above the 0 °C level, and most of all above the −20 °C
level.

```{code-cell} ipython3
dbz = np.linspace(20, 70, 201)
height = np.linspace(0, 12e3, 241)
h0, hm20 = 4000.0, 7000.0

weight_z = np.clip((dbz - 40.0) / 10.0, 0, 1)
flux = 5e-6 * 10 ** (0.084 * dbz) * weight_z
weight_t = np.clip((height - h0) / (hm20 - h0), 0, 1)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
axes[0].plot(dbz, weight_z)
axes[0].set(xlabel="reflectivity (dBZ)", ylabel="$W(Z)$")
axes[1].plot(dbz, flux)
axes[1].set(xlabel="reflectivity (dBZ)", ylabel="$\\dot E$ (J m$^{-2}$ s$^{-1}$)")
axes[2].plot(weight_t, height / 1e3)
axes[2].set(xlabel="$W_T$ for $H_0$ = 4 km, $H_{m20}$ = 7 km", ylabel="height (km)")
for ax in axes:
    ax.grid(alpha=0.3)
plt.show()
```

## Checks on columns

A column of uniform 55 dBZ from the ground to 12 km on 500 m levels, with the 0 and
−20 °C levels at 4 and 7 km, has a closed-form SHI. The levels from 4.5 to 6.5 km
have $W_T=1/6,\dots,5/6$ and sum to 2.5, the eleven levels from 7 to 12 km count
fully, and every sample stands for 500 m:

```{code-cell} ipython3
z = np.arange(0.0, 12001.0, 500.0)
column = xr.Dataset(
    {"DBZH": (("z", "y", "x"), np.full((z.size, 1, 1), 55.0))},
    coords={"z": z, "y": [0.0], "x": [0.0]},
)
out = hail_products(column, freezing_level=4000.0, minus20_level=7000.0)
by_hand = 0.1 * 5e-6 * 10 ** (0.084 * 55.0) * (2.5 + 11.0) * 500.0
print(f"SHI {float(out.SHI.squeeze()):.3f}  by hand {by_hand:.3f} J m-1 s-1")
print(f"MESH {float(out.MESH.squeeze()):.2f} mm = 2.54 sqrt(SHI) = {2.54 * by_hand**0.5:.2f} mm")
print(f"WT {float(out.SHI_WARNING_THRESHOLD.squeeze()):.0f} = 57.5 * 4 - 121,"
      f" POSH {float(out.POSH.squeeze()):.1f} %")
```

Fig. 4 of Witt et al. (1998) gives the reflectivity of the storm components of one
storm cell for five volume scans, for $H_0=3$ km and $H_{m20}=6$ km, and the SHI
printed above each scan. The heights below were read off the figure by eye
(about 0.1 km), so the numbers should agree to a few percent, not exactly:

```{code-cell} ipython3
fig4 = [  # (height ARL in km, dBZ) of each component, SHI of the paper
    ([(2.68, 52), (5.3, 51)], 22),
    ([(2.55, 56), (5.1, 58), (7.38, 43)], 66),
    ([(2.44, 53), (4.9, 59), (7.14, 43)], 68),
    ([(2.3, 54), (4.72, 63), (6.87, 48)], 134),
    ([(2.28, 67), (4.56, 65), (6.6, 48), (8.6, 39)], 172),
]
print("scan  paper  radarx")
for k, (components, printed) in enumerate(fig4, start=1):
    h = np.array([c[0] for c in components]) * 1e3
    d = np.array([c[1] for c in components], dtype=float)
    col = xr.Dataset(
        {"DBZH": (("z", "y", "x"), d[:, None, None])},
        coords={"z": h, "y": [0.0], "x": [0.0]},
    )
    got = float(shi(col, freezing_level=3000.0, minus20_level=6000.0).squeeze())
    print(f"{k:>4} {printed:>6} {got:>7.1f}")
```

## A squall line seen by KGWX

The KGWX WSR-88D (Columbus, Mississippi) volume of 30 March 2022, 23:59 UTC, the
one of the VIL notebook, with the no-data codes masked and the range cut at 150
km.

```{code-cell} ipython3
key = "2022/03/30/KGWX/KGWX20220330_235959_V06"
path = download_file("unidata-nexrad-level2", key, "data")
drop = ["WRADH", "CCORH", "CFP", "ZDR", "PHIDP", "RHOHV"]
volume = xd.io.open_nexradlevel2_datatree(path, drop_variables=drop)


def prepare(ds):
    """Mask the no-data codes and keep the first 150 km."""
    if "range" not in ds.dims:
        return ds
    ds = ds.sel(range=slice(None, 150e3))
    return ds.assign(DBZH=ds.DBZH.where(ds.DBZH > -32).astype("float32"))


volume = volume.map_over_datasets(prepare)
```

### The 0 and −20 °C levels

The two heights come from the ERA5 profile at the radar and the volume time
(`radarx.io.sounding`; the documentation build reads a small pre-extracted copy of
ERA5, as in the sounding notebook). `isotherm_height` finds where the profile
crosses 273.15 and 253.15 K. A radiosonde from `read_sounding` works the same way,
and the levels can also be given as numbers or as fields.

```{code-cell} ipython3
era5 = volume.radarx.sounding(era5_source="gcs", surface=False)
h0 = float(sounding.isotherm_height(era5, 273.15))
hm20 = float(sounding.isotherm_height(era5, 253.15))
altitude = float(era5.attrs["radar_altitude"])
print(f"0 degC at {h0:.0f} m, -20 degC at {hm20:.0f} m above sea level "
      f"(radar at {altitude:.0f} m): H0 = {(h0 - altitude) / 1e3:.2f} km above the radar")
```

### SHI, MESH, POSH and the kinetic energy flux

`hail_products` takes the volume and the profile and returns the products on the
polar grid of the lowest sweep (azimuth and ground range), with the warning
threshold of POSH, the kinetic energy flux of the lowest beam and a flag
`SHI_TOP_TRUNCATED` for the columns where hail-sized echo continues above the highest
beam. The same products are methods of the volume (`volume.radarx.hail_products`,
`.radarx.shi`, `.radarx.mesh`, `.radarx.posh` and `.radarx.hail_kinetic_energy`).

```{code-cell} ipython3
out = hail_products(volume, profile=era5)
out
```

```{code-cell} ipython3
low = volume["sweep_0"].to_dataset()
az = np.deg2rad(out.azimuth.values)[:, None]
s = out.ground_range.values[None, :]
x, y = s * np.sin(az) / 1e3, s * np.cos(az) / 1e3
rays = np.argsort(out.azimuth.values)  # no seam where the scan started

panels = [
    (low.DBZH, "$Z_H$ at 0.5° (dBZ)", dict(cmap="ChaseSpectral", vmin=-10, vmax=70)),
    (out.SHI.where(out.SHI > 0), "SHI (J m$^{-1}$ s$^{-1}$)", dict(cmap="magma_r", vmin=0, vmax=60)),
    (out.MESH.where(out.MESH > 0), "MESH (mm)", dict(cmap="viridis", vmin=0, vmax=25)),
    (out.POSH.where(out.POSH > 0), "POSH (%)", dict(cmap="cividis", vmin=0, vmax=60)),
]
fig, axes = plt.subplots(2, 2, figsize=(11, 10), layout="constrained")
for ax, (field, label, style) in zip(axes.flat, panels):
    pm = ax.pcolormesh(x[rays], y[rays], field.values[rays], shading="nearest", **style)
    fig.colorbar(pm, ax=ax, label=label, shrink=0.85)
    ax.set_aspect("equal")
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)
    ax.set_xlabel("east of KGWX (km)")
    ax.set_ylabel("north of KGWX (km)")
fig.suptitle("KGWX, 30 March 2022, 23:59 UTC")
plt.show()

positive = out.SHI.values > 0
print(f"columns with SHI > 0: {positive.sum()} of {np.isfinite(out.SHI.values).sum()}")
print(f"maximum SHI {float(out.SHI.max()):.0f} J m-1 s-1, MESH {float(out.MESH.max()):.0f} mm, "
      f"POSH {float(out.POSH.max()):.0f} %")
print(f"POSH above 50 %: {int((out.POSH > 50).sum())} columns")
```

The strong band of the squall line at 0.5° shows up in all three products, but
only a part of it has hail-sized SHI. Because $W(Z)$ is zero at 40 dBZ and the 0 °C
level is at 3.6 km, SHI is nonzero only where echo of 40 dBZ or more reaches
above that height. It is positive in 20 426 of the 426 240 columns with data. Over
most of the line it is below 20 J m$^{-1}$ s$^{-1}$, which gives a MESH of up to
11 mm, and the POSH of nearly all columns is 0 because the warning threshold at
this melting level is 74 J m$^{-1}$ s$^{-1}$. The largest values (SHI 98, MESH 25
mm, POSH 58 %) are in one core 20 km west of the radar; the next panel shows its
column. The numbers are those of the algorithm applied to this volume; no hail
reports were compared with them.

The hail kinetic energy flux is the flux of Eq. 1 at the lowest beam. It is
nonzero where the 0.5° reflectivity is above 40 dBZ and does not need the
freezing level.

```{code-cell} ipython3
flux = hail_kinetic_energy(volume)
fig, ax = plt.subplots(figsize=(5.5, 5), layout="constrained")
pm = ax.pcolormesh(
    x[rays], y[rays], flux.where(flux > 0).values[rays], shading="nearest",
    cmap="magma_r", vmin=0, vmax=0.5,
)
fig.colorbar(pm, ax=ax, label="hail kinetic energy flux (J m$^{-2}$ s$^{-1}$)")
ax.set_aspect("equal")
ax.set_xlim(-150, 150)
ax.set_ylim(-150, 150)
ax.set_xlabel("east of KGWX (km)")
ax.set_ylabel("north of KGWX (km)")
plt.show()
```

### Where the SHI comes from

The columns of the three highest-SHI cells, with the reflectivity at the beam
heights of every sweep and the two levels. SHI is the sum of the flux (the
reflectivity above 40 dBZ) weighted by $W_T$.

```{code-cell} ipython3
from radarx.retrieve.vil import _polar_columns

name = "DBZH"
template, ground, _, blocks = _polar_columns(volume, name, 0.1, None, True)
flat = np.argsort(out.SHI.values.ravel())[::-1]
picked = []
for index in flat:
    i, j = divmod(int(index), out.SHI.shape[1])
    if all(abs(j - jj) > 20 or abs(i - ii) > 20 for ii, jj in picked):
        picked.append((i, j))
    if len(picked) == 3:
        break

fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True, layout="constrained")
for ax, (i, j) in zip(axes, picked):
    for rows, h, _, v in blocks():
        if rows[0] <= i <= rows[-1]:
            col = (i - rows[0]) * ground.size + j
            hh, vv = h[:, col] / 1e3, v[:, col]
            break
    order = np.argsort(hh)
    ax.plot(vv[order], hh[order], "o-")
    ax.axhline(h0 / 1e3, color="C3", ls="--", label="0 °C")
    ax.axhline(hm20 / 1e3, color="C0", ls="--", label="−20 °C")
    ax.axvline(40, color="0.5", lw=0.8)
    ax.axvline(50, color="0.5", lw=0.8)
    ax.set_title(
        f"az {out.azimuth.values[i]:.0f}°, {ground[j] / 1e3:.0f} km\n"
        f"SHI {out.SHI.values[i, j]:.0f}, MESH {out.MESH.values[i, j]:.0f} mm, "
        f"POSH {out.POSH.values[i, j]:.0f} %",
        fontsize=10,
    )
    ax.set_xlabel("reflectivity (dBZ)")
axes[0].set_ylabel("beam height above sea level (km)")
axes[0].legend(frameon=False)
plt.show()
```

### Engines

The NumPy implementation gives the same numbers as the compiled kernel:

```{code-cell} ipython3
import time

t0 = time.perf_counter()
fast = hail_products(volume, profile=era5, engine="compiled")
t1 = time.perf_counter()
slow = hail_products(volume, profile=era5, engine="numpy")
t2 = time.perf_counter()
print(f"compiled {t1 - t0:.2f} s, numpy {t2 - t1:.2f} s, "
      f"largest SHI difference {float(abs(fast.SHI - slow.SHI).max()):.1e}")
```

## Levels that vary in space

The 0 and −20 °C levels can be 2-D fields. On a grid, `era5_column` interpolates ERA5
to every cell and gives the freezing level; `hail_products` takes the whole
background as `profile` and finds both isotherms in every column. Here the
freezing level varies by about 0.9 km across the domain (3.0 to 3.9 km), which changes the SHI
cell by cell (and the warning threshold, a function of the 0 °C height).

```{code-cell} ipython3
gx = gy = np.arange(-150e3, 150e3 + 1, 1000.0)
gz = np.arange(250.0, 18e3, 250.0)
grid = grid_cones(volume, "DBZH", gx, gy, gz)
background = sounding.era5_column(grid, "2022-03-30T23:59:59", source="gcs", surface=False)
hail_grid = hail_products(grid, profile=background)

freezing = background.freezing_level
print(f"freezing level {float(freezing.min()):.0f} to {float(freezing.max()):.0f} m")

gxk, gyk = hail_grid.x.values / 1e3, hail_grid.y.values / 1e3
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
for ax, field, label, style in [
    (axes[0], freezing / 1e3, "0 °C level (km)", dict(cmap="coolwarm")),
    (axes[1], hail_grid.SHI.where(hail_grid.SHI > 0), "SHI (J m$^{-1}$ s$^{-1}$)",
     dict(cmap="magma_r", vmin=0, vmax=60)),
    (axes[2], hail_grid.MESH.where(hail_grid.MESH > 0), "MESH (mm)",
     dict(cmap="viridis", vmin=0, vmax=25)),
]:
    pm = ax.pcolormesh(gxk, gyk, field.values, shading="nearest", **style)
    fig.colorbar(pm, ax=ax, label=label, shrink=0.85)
    ax.set_aspect("equal")
    ax.set_xlabel("east of KGWX (km)")
axes[0].set_ylabel("north of KGWX (km)")
plt.show()
```

The grid interpolates the reflectivity linearly in dBZ between the cones and its
columns have 250 m levels, the polar columns have the beams of the volume, so
the two SHI differ; the line shows in both. The grid values at the polar columns
with SHI above 5 J m$^{-1}$ s$^{-1}$, 20 to 120 km from the radar:

```{code-cell} ipython3
from scipy.interpolate import RegularGridInterpolator

at = RegularGridInterpolator((gy, gx), hail_grid.SHI.values, bounds_error=False)
north, east = s * np.cos(az), s * np.sin(az)
from_grid = at(np.stack([north.ravel(), east.ravel()], -1)).reshape(north.shape)
polar = out.SHI.values
use = (s > 20e3) & (s < 120e3) & (polar > 5.0) & np.isfinite(from_grid)
print(f"{use.sum()} polar columns with SHI > 5: median grid / polar = "
      f"{np.median(from_grid[use] / polar[use]):.2f}, correlation "
      f"{np.corrcoef(from_grid[use], polar[use])[0, 1]:.2f}")
```

## POSH and the warning threshold

POSH is 50 % where SHI equals the warning threshold, which grows with the
melting level (a high 0 °C level leaves less room above it for hail growth). Witt et
al. (1998) fitted it to storm days in Oklahoma and Florida, and a Western Region
technical attachment (Graham and Nelson 1998) reports that the threshold
over-warned for radars far above sea level. `warning_offset` (121) and
`posh_offset` (50) are arguments, so another calibration can be used.

```{code-cell} ipython3
shi_values = np.linspace(1, 400, 400)
fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
for h0_km in (2.0, 3.0, 4.0, 5.0):
    wt = max(57.5 * h0_km - 121.0, 20.0)
    p = np.clip(29 * np.log(shi_values / wt) + 50, 0, 100)
    ax.plot(shi_values, p, label=f"$H_0$ = {h0_km:.0f} km (WT = {wt:.0f})")
ax.set(xlabel="SHI (J m$^{-1}$ s$^{-1}$)", ylabel="POSH (%)")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
plt.show()
```

## What is not here

- **MRMS MESH.** No MRMS MESH or SHI product was available for this event, only
  the merged base reflectivity and azimuthal shear, so the products are not
  compared with the operational MRMS MESH. The NOAA product guide
  (Warning Decision Training Division) gives the same $\dot E$, $W(Z)$, $W_T$, SHI
  and POSH equations as the paper; the equation image on its MESH page reads
  MESH(inch) = $\sqrt{\mathrm{SHI}}$, which differs from Eq. 6 of the paper
  ($2.54\,\mathrm{SHI}^{0.5}$ mm, that is $0.1\,\mathrm{SHI}^{0.5}$ inch) by a factor 10
  and was not resolved. The paper is followed here.
- **POH.** The probability of hail of any size of Witt et al. (1998) is read off a
  figure that comes from Waldvogel et al. (1979); the paper gives no equation and
  the original was not read.
- **Other MESH fits.** Murillo and Homeyer (2019) fitted MESH to many more reports;
  that paper was not read.
- **Rounding.** The operational POSH is rounded to 10 % and MESH to 6.35 mm, the
  values here are not.
- **Cells.** The paper applies the sums to the maximum reflectivity of each
  component of a tracked storm cell; here every column is processed, as MRMS does on
  its grid.
- **Time integral.** The hail kinetic energy of hailpads in J m$^{-2}$ is the
  time integral of the flux over a series of volumes, which is not computed.

The products are diagnostics of the reflectivity profile that Witt et al. (1998)
calibrated on 147 hail reports. They are not a measurement of the hail at the ground.
