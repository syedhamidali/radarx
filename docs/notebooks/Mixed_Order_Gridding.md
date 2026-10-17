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

# Mixed-Order Gridding of Radial Velocity

+++

`radarx.grid.grid_cones` interpolates inside every sweep first and between the
sweeps in height last. Where the radial velocity changes quickly with height and
the sweeps are a few degrees apart, that leaves undulations at the horizontal
spacing of the beams, and a variational wind retrieval turns their divergence
into spurious vertical velocity through the continuity equation. Collis et al.
(2010) grid the data twice, once in each order of interpolation, and blend the
results with a weight that grows with the beam elevation (their Eq. 6).
`radarx.grid.grid_cones_mixed` does this for any field, with the exponent of the
weight as `order`, and `radarx.retrieve.multi_doppler_input` takes it with
`velocity_interpolation="mixed"`.

**References**

- Collis, S., A. Protat, and K.-S. Chung, 2010: The effect of radial velocity
  gridding artifacts on variationally retrieved vertical velocities. *J.
  Atmos. Oceanic Technol.*, **27**, 1239-1246,
  https://doi.org/10.1175/2010JTECHA1402.1

```{code-cell} ipython3
import warnings

import cmweather  # noqa
import fsspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xradar as xd
from xradar.georeference import antenna_to_cartesian

import radarx as rx

warnings.filterwarnings("ignore", message="w_boundary='bottom' sets")
```

## A jet with a known answer

A westerly jet of 20 m/s at 7 km height, the same everywhere, seen by a volume
of twelve sweeps from 0.5 to 42 degrees. The wind has no horizontal variation,
so the exact radial velocity at any grid point is known and so is the error of
the gridding. The sweeps are 1 degree apart in azimuth and 250 m in range.

```{code-cell} ipython3
SWEEPS = (0.5, 1.5, 3.0, 5.0, 7.5, 10.0, 13.0, 17.0, 22.0, 28.0, 35.0, 42.0)


def jet(height):
    return 20.0 * np.exp(-(((height - 7000.0) / 1500.0) ** 2))


def jet_volume(latitude=45.0, longitude=10.0, altitude=100.0):
    ranges = np.arange(250.0, 100e3, 250.0)
    azimuths = np.arange(0.5, 360.0, 1.0)
    sweeps = {}
    for k, elevation in enumerate(SWEEPS):
        _, _, gate_z = antenna_to_cartesian(ranges[None, :], azimuths[:, None], elevation, site_altitude=altitude)
        radial = jet(gate_z) * np.cos(np.radians(elevation)) * np.sin(np.radians(azimuths))[:, None]
        sweeps[f"sweep_{k}"] = xr.Dataset(
            {"VRADH": (("azimuth", "range"), radial), "DBZH": (("azimuth", "range"), np.full(radial.shape, 30.0))},
            coords={
                "azimuth": azimuths,
                "range": ranges,
                "elevation": ("azimuth", np.full(azimuths.size, elevation)),
                "time": ("azimuth", np.full(azimuths.size, np.datetime64("2026-01-01T00:00"))),
            },
        )
    root = xr.Dataset(coords={"latitude": latitude, "longitude": longitude, "altitude": altitude})
    return xr.DataTree.from_dict({"/": root, **sweeps})


volume = jet_volume()
gx = np.arange(-60e3, 60e3 + 1, 2e3)
gz = np.arange(1000.0, 13e3, 500.0)

# the exact radial velocity of the jet at the grid points of the section y = 0
X, Z = np.meshgrid(gx, gz)
beam_elevation = rx.grid.mixed.antenna_elevation(np.abs(X), Z, 100.0)
truth = jet(Z) * np.cos(np.radians(beam_elevation)) * np.sign(X)

grids = {"vertical last (grid_cones)": rx.grid.grid_cones(volume, "VRADH", gx, gx, gz).VRADH}
for order in (0.0, 0.5, 1.0, 2.0):
    grids[f"mixed, order {order:g}"] = rx.grid.grid_cones_mixed(volume, "VRADH", gx, gx, gz, order=order).VRADH


def section_error(field):
    return field.sel(y=0.0) - truth


def far_rms(error):
    return float(np.sqrt(np.nanmean(error.where(np.abs(error.x) > 10e3) ** 2)))


for name, field in grids.items():
    print(f"{name:28s} rms error of the radial velocity: {far_rms(section_error(field)):.2f} m/s")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained", sharey=True)
for ax, name in zip(axes, ("vertical last (grid_cones)", "mixed, order 0.5", "mixed, order 0")):
    error = section_error(grids[name])
    im = ax.pcolormesh(gx / 1e3, gz / 1e3, error, cmap="RdBu_r", vmin=-4, vmax=4, shading="auto")
    ax.set_title(f"{name}\nrms {far_rms(error):.2f} m/s")
    ax.set_xlabel("x (km)")
axes[0].set_ylabel("height (km)")
fig.colorbar(im, ax=axes, label="gridded minus exact radial velocity (m/s)")
plt.show()
```

The undulations of the first panel come from the vertical interpolation between
the sweeps at the top of the jet. Interpolating along the sweeps last (`order=0`)
removes them. The jet has no horizontal structure, so this is the best case for
it; on real data it smooths what varies in the horizontal. The mixed result sits
between the two, with the weight of the horizontal-last grid growing with the
beam elevation.
The exponent 0.5 is the value Collis et al. chose for their 17 tilts from 0.5 to
42 degrees; tune it to the scan strategy.

## Real data

The same comparison on radial velocity from a KGWX volume (30 March 2022),
without a truth: the difference between the mixed and the vertical-last grid at
3 km. Without network access the jet volume stands in.

```{code-cell} ipython3
def real_volume():
    local_file = fsspec.open_local(
        "simplecache::s3://unidata-nexrad-level2/2022/03/30/KGWX/KGWX20220330_234639_V06",
        s3={"anon": True},
        filecache={"cache_storage": "."},
    )
    tree = xd.io.open_nexradlevel2_datatree(local_file)
    for sweep in [n for n in tree.children if n.startswith("sweep")]:
        ds = tree[sweep].to_dataset()
        ds = ds[[v for v in ("VRADH",) if v in ds]]
        tree[sweep] = ds.where(ds.VRADH > -63.9) if "VRADH" in ds else ds
    return tree.load()


try:
    real = real_volume()
    source = "KGWX, 30 March 2022 23:46 UTC"
except Exception as error:  # offline
    print(f"no network access ({type(error).__name__}), using the jet volume")
    real, source = volume, "jet volume"

x = y = np.arange(-100e3, 100e3 + 1, 2000.0)
z = np.array([3000.0])
linear = rx.grid.grid_cones(real, "VRADH", x, y, z).VRADH.isel(z=0)
mixed = rx.grid.grid_cones_mixed(real, "VRADH", x, y, z).VRADH.isel(z=0)
difference = (mixed - linear).where(linear.notnull() & mixed.notnull())
print(f"{source}: rms difference {float(np.sqrt((difference ** 2).mean())):.2f} m/s, "
      f"median absolute difference {float(np.abs(difference).median()):.2f} m/s")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained", sharey=True)
for ax, field, title in zip(axes[:2], (linear, mixed), ("vertical last (grid_cones)", "mixed, order 0.5")):
    im = field.plot(ax=ax, cmap="RdBu_r", vmin=-30, vmax=30, add_colorbar=False)
    ax.set_title(title)
fig.colorbar(im, ax=axes[:2], label="radial velocity at 3 km (m/s)", shrink=0.8)
im = difference.plot(ax=axes[2], cmap="PuOr", vmin=-5, vmax=5, add_colorbar=False)
axes[2].set_title("mixed minus vertical last")
fig.colorbar(im, ax=axes[2], label="m/s", shrink=0.8)
for ax in axes:
    ax.set(aspect="equal")
fig.suptitle(source)
plt.show()
```

## In the wind retrieval

`multi_doppler_input(..., velocity_interpolation="mixed")` grids the velocity of
every radar with `grid_cones_mixed` (`mixed_order` is the exponent) and the
other fields as before. The same jet seen by two radars 60 km apart has zero
vertical velocity everywhere, so the vertical velocity that the dual-Doppler
analysis returns is its error. The grid has 3 km spacing.

```{code-cell} ipython3
east = 60e3 / (111.2e3 * np.cos(np.radians(45.0)))
two = [jet_volume(), jet_volume(longitude=10.0 + east)]
x = y = np.arange(-40e3, 100e3 + 1, 3000.0)
z = np.arange(500.0, 12e3, 500.0)
dims = ("z", "y", "x")
background = xr.Dataset(
    {name: (dims, np.full((z.size, y.size, x.size), value)) for name, value in (("u", 0.0), ("v", 0.0), ("air_density", 1.0))},
    coords={"z": z, "y": y, "x": x},
)
winds = {}
for mode in ("linear", "mixed"):
    inputs = rx.retrieve.multi_doppler_input(two, x, y, z, velocity_interpolation=mode)
    winds[mode] = rx.retrieve.multi_doppler(inputs, background, fall_speed_correction=False)
for mode, wind in winds.items():
    good = ((wind.beam_crossing_angle > 30) & (wind.n_radars >= 2)).values
    print(f"{mode:7s} rms of the vertical velocity (true value 0): {np.sqrt(np.mean(wind.w.values[good] ** 2)):.2f} m/s")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained", sharey=True)
for ax, (mode, wind) in zip(axes, winds.items()):
    good = ((wind.beam_crossing_angle > 30) & (wind.n_radars >= 2))
    section = wind.w.where(good).sel(y=0.0, method="nearest")
    im = ax.pcolormesh(wind.x / 1e3, wind.z / 1e3, section, cmap="RdBu_r", vmin=-3, vmax=3, shading="auto")
    ax.set(title=f"{mode} interpolation", xlabel="x (km)")
axes[0].set_ylabel("height (km)")
fig.colorbar(im, ax=axes, label="retrieved w, true value 0 (m/s)")
plt.show()
```

The retrieved vertical velocity is closer to the true zero with the mixed
interpolation; the gain depends on how fast the wind changes with height and on
the beam spacing, so test it on your scan strategy before you rely on it.
