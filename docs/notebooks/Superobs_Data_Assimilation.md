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

# Super-observations for data assimilation

A WSR-88D volume holds several million gates, a convection-permitting model a
few hundred thousand cells, and the errors of neighbouring gates are
correlated. Assimilation systems therefore work on *super-observations*,
averages of the gates in a box of the model grid
([Alpert and Kumar 2007](https://doi.org/10.1175/MWR3324.1)).
`radarx.retrieve.superobs` bins the gates of one or more radars into columns
of the model grid and layers of height and returns, per box:

- the mean reflectivity, averaged in linear $Z$ and given in dBZ;
- the mean radial velocity, with the mean unit vector of the rays so that an
  observation operator can project the model wind on it;
- the number of gates, the spread of the gate values, and an error estimate
  $\sqrt{\mathrm{floor}^2 + \mathrm{spread}^2/n}$;
- a flag for boxes that have gates but no echo, which WRFDA can use to remove
  model rain.

`radarx.io.write_wrfda_radar` writes the result as the `ob.radar` file that
WRFDA reads. The binning and the sums run in a compiled, multithreaded C++
kernel; a NumPy implementation gives the same numbers.

```{code-cell} ipython3
import time
import warnings

import cmweather  # noqa: F401  radar colormaps
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xradar as xd
from pyproj import Transformer
from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.fundamentals.geometry import beam_center_height, ground_range
from radarx.io import write_wrfda_radar
from radarx.io.aws_data import download_file
from radarx.retrieve import superobs, vad_profile

warnings.filterwarnings("ignore", category=RuntimeWarning)
```

## The squall line seen by KGWX

The KGWX WSR-88D volume of 30 March 2022, 23:59 UTC, from the open NEXRAD
archive on AWS. The no-data codes are masked, the range is cut at 150 km, and
the velocity is dealiased with `dealias` (the Nyquist velocity is read from the
radial headers, as in the [dealiasing notebook](Dealiasing)).

```{code-cell} ipython3
key = "2022/03/30/KGWX/KGWX20220330_235959_V06"
path = download_file("unidata-nexrad-level2", key, "data")
drop = ["WRADH", "CCORH", "CFP", "ZDR", "PHIDP"]
volume = xd.io.open_nexradlevel2_datatree(path, drop_variables=drop)

with NEXRADLevel2File(path) as nf:
    nyquist = [
        h["msg_31_data_header"]["RAD"]["nyquist_vel"] / 100.0
        for h in nf.msg_31_data_header
    ]


def prepare(ds):
    """Mask the no-data codes and keep the first 150 km."""
    if "range" not in ds.dims:
        return ds
    ds = ds.sel(range=slice(None, 150e3))
    ds = ds.assign(DBZH=ds.DBZH.where(ds.DBZH > -32))
    if "VRADH" in ds:
        ds = ds.assign(VRADH=ds.VRADH.where(ds.VRADH > -63.9))
    return ds.load()


volume = volume.map_over_datasets(prepare)
sweeps = [n for n in volume.children if n.startswith("sweep")]
for name, value in zip(sweeps, nyquist):
    volume[name] = volume[name].to_dataset().assign_coords(nyquist_velocity=value)
volume = volume.radarx.assign(volume.radarx.dealias("VRADH"))
print(len(sweeps), "sweeps; velocity in", sum("VRADH" in volume[n].ds for n in sweeps))
```

## A model grid

WRFDA reads a model on its own map projection. Here a Lambert conformal grid
of 1 km cells, 300 km wide, stands for the model; `superobs` needs only the
latitude and longitude of the cell centres (a WRF `wrfinput` file works as it
is, with `XLAT`, `XLONG` and `HGT`).

```{code-cell} ipython3
lcc = Transformer.from_crs(
    "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=33.9 +lon_0=-88.33 +R=6370000 +units=m",
    "EPSG:4326",
    always_xy=True,
)
cell = 1000.0
axis = (np.arange(300) - 149.5) * cell
xx, yy = np.meshgrid(axis, axis)
lon2d, lat2d = lcc.transform(xx, yy)
model = xr.Dataset(
    {
        "XLAT": (("south_north", "west_east"), lat2d),
        "XLONG": (("south_north", "west_east"), lon2d),
        "HGT": (("south_north", "west_east"), np.full(lat2d.shape, 130.0)),
    }
)
print(f"cell centres from {lat2d.min():.2f} to {lat2d.max():.2f} N, "
      f"{lon2d.min():.2f} to {lon2d.max():.2f} E")
```

## Super-observations

Each box is 3 x 3 model cells (3 km) and 500 m deep, from the ground to 12 km
above sea level. A box needs 6 gates, and 3 valid gates for a value of
reflectivity or velocity. Gates with a correlation coefficient below 0.85 are
dropped. The call takes the volume and the model grid and returns a dataset
along `obs`, one entry per box with data.

```{code-cell} ipython3
start = time.perf_counter()
so = superobs(volume, model, cells=3, layers=500.0, top=12000.0,
              vel="VRADH_dealiased", rho_min=0.85, max_range=150e3)
t_compiled = time.perf_counter() - start
so
```

The same call as a method of the volume, and with the NumPy implementation.
The two implementations agree to rounding.

```{code-cell} ipython3
start = time.perf_counter()
so_numpy = volume.radarx.superobs(model, cells=3, vel="VRADH_dealiased",
                                  rho_min=0.85, max_range=150e3, engine="numpy")
t_numpy = time.perf_counter() - start
for name in ("reflectivity", "velocity", "velocity_error", "radial_east", "height"):
    assert np.allclose(so[name], so_numpy[name], rtol=1e-10, equal_nan=True)
print(f"compiled {t_compiled:.2f} s, NumPy {t_numpy:.2f} s; "
      f"{int(so.gate_count.sum()):,} gates in {so.sizes['obs']:,} boxes, "
      f"{int(so.reflectivity.count()):,} with reflectivity, "
      f"{int(so.velocity.count()):,} with velocity, "
      f"{int(so.nonprecip.sum()):,} without echo")
```

### Gates and super-observations

One layer, 2 to 2.5 km above sea level. The gates of all sweeps that fall in
the layer are shown on the left, the super-observations on the right.

```{code-cell} ipython3
radar_lat = float(so.radar_latitude[0])
radar_lon = float(so.radar_longitude[0])
radar_alt = float(so.radar_altitude[0])
to_xy = Transformer.from_crs("EPSG:4326",
    "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=33.9 +lon_0=-88.33 +R=6370000 +units=m", always_xy=True)
layer = int(np.searchsorted(so.level_edges.values, 2000.0, side="right") - 1)
lo, hi = so.level_edges.values[layer:layer + 2]

# gates of the layer, from the sweeps (geometry of the 4/3 Earth)
gx, gy, gz = [], [], []
for name in sweeps:
    ds = volume[name].to_dataset()
    az = np.deg2rad(ds.azimuth.values)[:, None]
    el = ds.elevation.values[:, None]
    r = ds.range.values[None, :]
    h = beam_center_height(r, el, radar_alt)
    s = ground_range(r, el)
    ok = (h >= lo) & (h < hi) & (s > 3e3) & np.isfinite(ds.DBZH.values)
    gx.append((s * np.sin(az) / 1e3)[ok])
    gy.append((s * np.cos(az) / 1e3)[ok])
    gz.append(ds.DBZH.values[ok])
gx, gy, gz = np.concatenate(gx), np.concatenate(gy), np.concatenate(gz)
print(f"{gz.size:,} gates with reflectivity in the layer {lo:.0f} to {hi:.0f} m")
```

```{code-cell} ipython3
in_layer = so.where(so.level == layer, drop=True)
ox, oy = to_xy.transform(in_layer.longitude.values, in_layer.latitude.values)
ox0, oy0 = to_xy.transform(radar_lon, radar_lat)
ox, oy = (ox - ox0) / 1e3, (oy - oy0) / 1e3

fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True, sharey=True, layout="constrained")
style = dict(cmap="ChaseSpectral", vmin=-10, vmax=70)
axes[0].scatter(gx, gy, c=gz, s=1.5, marker=".", rasterized=True, **style)
axes[0].set_title(f"{gz.size:,} gates")
sc = axes[1].scatter(ox, oy, c=in_layer.reflectivity, s=11, marker="s", **style)
axes[1].set_title(f"{in_layer.reflectivity.count().item():,} super-observations of 3 km")
for ax in axes:
    ax.set(xlim=(-150, 150), ylim=(-150, 150), aspect="equal", xlabel="east of KGWX (km)")
axes[0].set_ylabel("north of KGWX (km)")
fig.colorbar(sc, ax=axes, label="reflectivity (dBZ)", shrink=0.8)
fig.suptitle(f"Reflectivity, layer {lo / 1e3:.1f} to {hi / 1e3:.1f} km above sea level")
plt.show()
```

The line of convection is kept, and the super-observations average the
speckle of the gates. Gates farther than 100 km are fewer per box because the
beams are far apart in height there.

The radial velocity of the same layer, as a super-observation per box (the
mean of the dealiased gates, positive away from the radar), next to its error
estimate:

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True, sharey=True, layout="constrained")
ok = in_layer.velocity.notnull().values
sc = axes[0].scatter(ox[ok], oy[ok], c=in_layer.velocity[ok], s=11, marker="s",
                     cmap="balance", vmin=-40, vmax=40)
fig.colorbar(sc, ax=axes[0], label="radial velocity (m s$^{-1}$)", shrink=0.8)
sc = axes[1].scatter(ox[ok], oy[ok], c=in_layer.velocity_error[ok], s=11, marker="s",
                     cmap="viridis", vmin=2, vmax=3)
fig.colorbar(sc, ax=axes[1], label="error estimate (m s$^{-1}$)", shrink=0.8)
for ax in axes:
    ax.set(xlim=(-150, 150), ylim=(-150, 150), aspect="equal", xlabel="east of KGWX (km)")
axes[0].set_ylabel("north of KGWX (km)")
plt.show()
```

## Spread and error

The error of a super-observation is $\sqrt{\sigma_0^2 + s^2/n}$ with the
floor $\sigma_0$ (`error_floor`, 2 m s$^{-1}$ and 5 dBZ), the standard deviation $s$ of the gate
values in the box and the number $n$ of valid gates. Boxes with a large spread
and few gates get the larger errors. This error model is a radarx choice; the
paper does not give one that could be checked.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
v = so.where(so.velocity.notnull(), drop=True)
hb = axes[0].hexbin(v.velocity_count, v.velocity_spread, C=v.velocity_error, reduce_C_function=np.mean,
                    gridsize=40, extent=(3, 40, 0, 8), cmap="viridis", vmin=2, vmax=4)
axes[0].set(xlabel="gates with velocity", ylabel="spread (m s$^{-1}$)")
fig.colorbar(hb, ax=axes[0], label="mean error (m s$^{-1}$)")
z = so.where(so.reflectivity.notnull(), drop=True)
hb = axes[1].hexbin(z.reflectivity_count, z.reflectivity_spread, C=z.reflectivity_error, reduce_C_function=np.mean,
                    gridsize=40, extent=(3, 60, 0, 12), cmap="viridis", vmin=5, vmax=7)
axes[1].set(xlabel="gates with reflectivity", ylabel="spread (dBZ)")
fig.colorbar(hb, ax=axes[1], label="mean error (dBZ)")
plt.show()
```

## The radial unit vector

`radial_east`, `radial_north` and `radial_up` are the mean unit vector of the
rays at the velocity gates of the box. A model wind $(u, v, w)$ is compared with
the super-observation as $u\,e + v\,n + w\,w_\mathrm{up}$, which is the mean of the
projections on the gates for a wind that is uniform in the box. As an example,
the VAD wind profile of the same volume (`vad_profile`, a radar-only wind
profile) stands for the model: it is projected on the super-observations up to
6 km and 100 km from the radar, and the difference is the innovation that an assimilation
system would see.

```{code-cell} ipython3
vad = vad_profile(volume, velocity="VRADH_dealiased", max_rms=4.0, max_range=80e3)
near = so.where(so.velocity.notnull() & (so.range < 100e3) & (so.height < 6000), drop=True)
u = np.interp(near.height, vad.height, vad.u)
v = np.interp(near.height, vad.height, vad.v)
background = u * near.radial_east + v * near.radial_north
innovation = near.velocity - background
fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
ax.hist(innovation, bins=np.arange(-30, 30.1, 1.0), color="0.4")
ax.set(xlabel="super-observation minus projected VAD wind (m s$^{-1}$)", ylabel="boxes")
plt.show()
print(f"{near.sizes['obs']:,} super-observations: median innovation "
      f"{float(innovation.median()):.1f} m/s, rms {float(np.sqrt((innovation**2).mean())):.1f} m/s, "
      f"mean error estimate {float(near.velocity_error.mean()):.1f} m/s")
```

The innovations peak at zero with a core of a few m s$^{-1}$, which shows that
the unit vector projects the wind as intended. The wide tail is not examined here:
it can come from the VAD wind, which is uniform in each ring while the squall line
is not, and from folds that the dealiasing without a wind profile did not resolve.

## Boxes without echo

With `missing="clear"` (the default) a gate without reflectivity is a gate
without echo. A box with at least 6 gates but fewer than 3 gates with
reflectivity, or a mean below 5 dBZ, has `nonprecip = 1`. Assimilating these as
zero-rain observations removes spurious model rain; the figure shows them
(grey) with the echo of the same layer.

```{code-cell} ipython3
dry = in_layer.nonprecip.values.astype(bool)
fig, ax = plt.subplots(figsize=(6.2, 5.6), layout="constrained")
ax.scatter(ox[dry], oy[dry], c="0.75", s=11, marker="s", label="no echo")
sc = ax.scatter(ox[~dry], oy[~dry], c=in_layer.reflectivity[~dry], s=11, marker="s", **style)
fig.colorbar(sc, ax=ax, label="reflectivity (dBZ)", shrink=0.8)
ax.set(xlim=(-150, 150), ylim=(-150, 150), aspect="equal",
       xlabel="east of KGWX (km)", ylabel="north of KGWX (km)")
ax.legend(frameon=False, loc="upper right")
plt.show()
print(f"{int(so.nonprecip.sum()):,} of {so.sizes['obs']:,} boxes have no echo")
```

## The WRFDA file

`write_wrfda_radar` writes one block per radar, one record per column and one
line per layer: the layer-centre height, then the velocity and the
reflectivity, each with a quality flag (0 used, -88 missing) and an error. Boxes
without echo get the non-precipitating value -999.99 (`nonprecip=None` writes them as
missing). The first line holds the number of radars in columns 15 to 17.

```{code-cell} ipython3
stats = write_wrfda_radar(so, "ob.radar", time="2022-03-31_00:00:00")
print(stats)
with open("ob.radar") as f:
    head = [next(f).rstrip("\n") for _ in range(16)]
print("\n".join(head))
```

A fixed-width read of the file recovers the counts and the values:

```{code-cell} ipython3
with open("ob.radar") as f:
    lines = f.read().split("\n")
n_radars = int(lines[0][14:17])
at = next(i for i, line in enumerate(lines) if line.startswith("RADAR"))
columns, levels = int(lines[at][68:74]), int(lines[at][74:80])
body = lines[at + 3 : at + 3 + columns * (levels + 1)]
level_lines = [line for line in body if not line.startswith("FM-128")]
rv = np.array([float(line[15:27]) for line in level_lines])
rf = np.array([float(line[45:57]) for line in level_lines])
print(f"{n_radars} radar, {columns} columns of {levels} levels: "
      f"{(rv > -888888).sum():,} velocities, {(rf > -888888).sum():,} reflectivities "
      f"({(rf == -999.99).sum():,} of them without echo)")
assert (rv > -888888).sum() == stats["velocities"]
assert (rf > -888888).sum() == stats["reflectivities"]
```

## Notes

- The velocity is averaged as given. Dealias it first (as here), or rely on the
  innovation check of the assimilation system to reject folded gates.
- SAILS repeats of the 0.5° cut and the surveillance and Doppler cuts of the same
  elevation are all used; their gates count once per cut.
- The layout of the `ob.radar` file is the one of the writer this function comes
  from, which WRFDA 4.7 read without errors; it was not checked here against the
  WRFDA reader. A DART (`obs_seq`) writer is not included, because its format could
  not be checked against the DART documentation.
- Alpert and Kumar (2007) describe super-obs as averages of raw radial winds.
  The box sizes and the error model here are radarx choices.
