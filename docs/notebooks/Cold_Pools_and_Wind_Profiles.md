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

# Cold pools, wind profiles and baroclinic vorticity

+++

Squall lines live off the balance between the cold pool they produce and the
low-level shear of the air they ingest (Rotunno et al. 1988; Weisman and
Rotunno 2004). radarx has the pieces to measure both sides on xarray objects:

- readers for surface station networks (`radarx.io.read_sticknet`,
  `read_pips`) on `(station, time)` and for profilers (`read_mrr`,
  `read_wind_profiler`) on `(time, height)`;
- wind-profile parameters (`bulk_shear`, `bunkers_storm_motion`,
  `storm_relative_helicity`, `vad_profile`) for soundings, ERA5, profilers
  and radar VADs;
- cold-pool diagnostics (`potential_temperatures`, `cold_pool_perturbation`,
  `cold_pool_intensity`, `rkw_ratio`) and the baroclinic generation of
  horizontal vorticity on grids (`baroclinic_generation`).

This example uses the Birmingham, AL (BMX) radiosonde of 00 UTC 31 March
2022, launched ahead of the squall line of that evening (PERiLS IOP2), the
KGWX radar ahead of the line,
plus small synthetic station and grid data. The PERiLS field data used to
validate these functions are archived at NCAR/EOL: TTU StickNet
([doi:10.26023/93M9-AE8F-SX07](https://doi.org/10.26023/93M9-AE8F-SX07)),
UAH MRR ([doi:10.26023/PB1C-EW31-970C](https://doi.org/10.26023/PB1C-EW31-970C))
and UAH 915-MHz wind profiler
([doi:10.26023/F13E-70W4-5N0J](https://doi.org/10.26023/F13E-70W4-5N0J)).

```{code-cell} ipython3
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xradar as xd

import radarx  # noqa: F401  (registers the .radarx accessors)
from radarx.io import read_sticknet, sounding
from radarx.io.aws_data import download_file
from radarx.retrieve import (
    baroclinic_generation,
    buoyancy,
    bulk_shear,
    bunkers_storm_motion,
    cold_pool_intensity,
    cold_pool_perturbation,
    dealias_velocity,
    potential_temperatures,
    rkw_ratio,
    storm_relative_helicity,
    vad_profile,
)
```

## The environment: shear and helicity from a sounding

Any Dataset with `u` and `v` on a vertical dimension works: here the BMX
sounding, in the warm sector ahead of the line. Layers are counted from the
lowest level with a valid wind.

```{code-cell} ipython3
env = sounding.read_sounding("KBMX", "2022-03-31T00:00", source="iem")
env = env.where(np.isfinite(env.u) & np.isfinite(env.v), drop=True)

line_motion = (8.8, -3.2)  # m/s, squall line moving toward the east-south-east
normal = 110.0  # azimuth of the line normal, toward the inflow (degrees)

shear = bulk_shear(env, 0, 2500, normal=normal)
rm = bunkers_storm_motion(env)
print(f"0-6 km bulk shear: {float(bulk_shear(env, 0, 6000).shear_speed):.1f} m/s")
print(f"0-2.5 km line-normal shear du: {float(shear.shear_normal):.1f} m/s")
print(f"Bunkers right mover: {float(rm.u):.1f}, {float(rm.v):.1f} m/s")
for top in (1000, 3000):
    srh_line = float(storm_relative_helicity(env, line_motion, 0, top))
    srh_rm = float(storm_relative_helicity(env, "right", 0, top))
    print(f"SRH 0-{top // 1000} km: line motion {srh_line:.0f}, Bunkers RM {srh_rm:.0f} m2/s2")
```

```{code-cell} ipython3
low = env.where(env.height - env.height[0] <= 6000, drop=True)
fig, ax = plt.subplots(figsize=(5, 5))
sc = ax.scatter(low.u, low.v, c=(low.height - low.height[0]) / 1e3, s=8, cmap="viridis")
ax.plot(low.u, low.v, "k-", lw=0.5)
ax.plot(*line_motion, "rs", label="line motion")
ax.plot(float(rm.u), float(rm.v), "b^", label="Bunkers right mover")
ax.axhline(0, color="0.7", lw=0.5)
ax.axvline(0, color="0.7", lw=0.5)
ax.set_xlabel("u (m s$^{-1}$)")
ax.set_ylabel("v (m s$^{-1}$)")
ax.set_aspect("equal")
ax.legend(loc="upper left")
fig.colorbar(sc, label="height above ground (km)")
ax.set_title("BMX 00 UTC 31 March 2022, 0-6 km hodograph");
```

## A radar VAD on the same evening

`vad_profile` fits the (dealiased) radial velocities of every range ring
(Browning and Wexler 1968) and averages the ring winds in height bins. The
velocities of a few KGWX sweeps are dealiased first with
`dealias_velocity`, using the Nyquist velocity of the Message 31 headers.

```{code-cell} ipython3
file = download_file(
    "unidata-nexrad-level2", "2022/03/30/KGWX/KGWX20220330_224545_V06", "./downloads"
)
from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

with NEXRADLevel2File(file) as nf:
    nyquist = [
        h["msg_31_data_header"]["RAD"]["nyquist_vel"] / 100.0
        for h in nf.msg_31_data_header
    ]
picked = [9, 13, 15, 19]  # sweeps at about 2.4, 4.0, 6.4 and 10 degrees
tree = xd.io.open_nexradlevel2_datatree(file, sweep=picked)
sweeps = {}
for name, index in zip([n for n in tree.children if n.startswith("sweep")], picked):
    ds = tree[name].to_dataset(inherit="all_coords")
    ds["VRADH"] = ds.VRADH.where(ds.VRADH > -63.9)  # no-data code
    ds["VRADH"] = dealias_velocity(ds, nyquist_velocity=nyquist[index])
    sweeps[f"/{name}"] = ds
volume = xr.DataTree.from_dict({"/": tree.to_dataset(), **sweeps})
vad = volume.radarx.vad_profile(max_rms=4.0, max_range=80e3)
vad[["u", "v", "vad_rings"]].dropna("height").isel(height=slice(0, 60, 10))
```

```{code-cell} ipython3
fig, ax = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
for a, comp in zip(ax, ("u", "v")):
    a.plot(vad[comp], vad.height / 1e3, "k.-", ms=3, label="KGWX VAD 22:45 UTC")
    a.plot(env[comp], env.height / 1e3, "b-", label="BMX 00 UTC (190 km east)")
    a.set_xlabel(f"{comp} (m s$^{{-1}}$)")
ax[0].set_ylim(0, 8)
ax[0].set_ylabel("height above sea level (km)")
ax[0].legend()
print(f"VAD 0-3 km bulk shear: {float(bulk_shear(vad, 0, 3000).shear_speed):.1f} m/s")
```

## Surface stations and the cold pool

Station networks are Datasets on `(station, time)` with the station
coordinates on `station`. A StickNet file has the columns
`Time, T, RH, P, WS, WD`; here two short synthetic files in that format stand
in for the PERiLS archive. The second station feels the gust front 15 minutes
after the first.

```{code-cell} ipython3
folder = Path(tempfile.mkdtemp())
(folder / "IOP2_StickNet_Locations.csv").write_text(
    "ID,Latitude,Longitude,Elevation,Array_Type\n"
    "101A,33.80,-88.70,90.0,Coarse\n102A,33.80,-88.50,80.0,Coarse\n"
)
times = np.arange("2022-03-30T23:00", "2022-03-31T01:00", 1, dtype="datetime64[s]")
minutes = (times - times[0]).astype(float) / 60
for sid, arrival in (("0101A", 50.0), ("0102A", 65.0)):
    cold = np.clip((minutes - arrival) / 5.0, 0, 1)  # 5-min ramp at the gust front
    t = 22.0 - 0.01 * minutes - 7.0 * cold
    rh = 75.0 + 20.0 * cold
    p = 990.0 + 2.5 * cold
    ws = 5.0 + 10.0 * np.exp(-(((minutes - arrival - 3) / 3) ** 2))
    wd = 170.0 + 100.0 * cold
    rows = [
        f"{str(ti).replace('T', ' ')},{a:.2f},{b:.1f},{c:.2f},{d:.1f},{e:.0f}"
        for ti, a, b, c, d, e in zip(times, t, rh, p, ws, wd)
    ]
    (folder / f"{sid}_IOP2_level3.txt").write_text("Time,T,RH,P,WS,WD\n" + "\n".join(rows))

net = read_sticknet(folder, iop=2)
net
```

`cold_pool_perturbation` subtracts a reference state, here the mean of a
pre-storm window at every station, and gives the buoyancy
$B = g\,\Delta\theta_v / \overline{\theta_v}$ (no hydrometeor loading unless
it is given).

```{code-cell} ipython3
pert = cold_pool_perturbation(net, slice("2022-03-30T23:00", "2022-03-30T23:30"))
fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
for s in pert.station.values:
    ax[0].plot(pert.time, pert.virtual_potential_temperature_perturbation.sel(station=s), label=s)
    ax[1].plot(pert.time, pert.pressure_perturbation.sel(station=s) / 100, label=s)
ax[0].set_ylabel(r"$\Delta\theta_v$ (K)")
ax[1].set_ylabel(r"$\Delta p$ (hPa)")
ax[0].legend()
fig.autofmt_xdate()
```

The cold-pool intensity $C = \sqrt{2 \int_0^H (-B)\,dz}$ needs the depth. A
post-storm sounding gives it directly; from the surface alone, the hydrostatic
pressure rise gives $C = \sqrt{2 \Delta p / \rho}$, or an assumed depth gives
$C = \sqrt{-B_s H}$ for a deficit decreasing linearly with height.

```{code-cell} ipython3
from radarx.retrieve import (
    cold_pool_intensity_from_pressure,
    cold_pool_intensity_from_surface,
)

after = pert.sel(time=slice("2022-03-31T00:30", None)).mean("time")
c_p = cold_pool_intensity_from_pressure(after.pressure_perturbation, density=1.15)
c_b = cold_pool_intensity_from_surface(after.buoyancy, depth=2000.0)
xr.Dataset({"C_from_pressure": c_p, "C_from_surface_buoyancy_H2km": c_b}).to_dataframe()
```

## Cold-pool intensity from a sounding and the RKW ratio

A post-storm profile minus the pre-storm one gives the buoyancy profile; here
the BMX sounding is cooled by 7 K at the ground, decreasing to zero at
2.5 km, as an idealised post-line sounding.

```{code-cell} ipython3
pre = env.where(env.height - env.height[0] < 8000, drop=True)
agl = pre.height - pre.height[0]
post = pre.copy()
post["temperature"] = pre.temperature - 7.0 * (1 - agl / 2500.0).clip(min=0)
post["dewpoint"] = np.minimum(post.dewpoint, post.temperature)

b = cold_pool_perturbation(post, pre).buoyancy
cp = cold_pool_intensity(b)
c = float(cp.cold_pool_intensity)
print(f"C = {c:.1f} m/s, depth {float(cp.cold_pool_depth):.0f} m")
print(f"RKW ratio C/du (0-2.5 km line-normal shear): {float(rkw_ratio(c, shear.shear_normal)):.1f}")
```

A ratio well above one, as here, means a cold pool that overwhelms the
low-level shear: the line tilts upshear and is fed by rearward-sloping
ascent (Weisman and Rotunno 2004; Bryan et al. 2006 discuss how sensitive the
ratio is to the choice of depth and shear layer).

## Baroclinic generation of horizontal vorticity on a grid

Horizontal buoyancy gradients generate horizontal vorticity,
$d\xi/dt = \partial B/\partial y$ and $d\eta/dt = -\partial B/\partial x$.
On a grid (e.g. the accumulated evaporative cooling of
`radarx.retrieve.integrate_evaporation` with multi-Doppler winds),
`baroclinic_generation` also splits it into streamwise and crosswise parts
relative to the storm-relative wind. Here a synthetic cold pool under a
uniform southerly storm-relative inflow:

```{code-cell} ipython3
x = np.arange(-40e3, 40001, 1000.0)
y = np.arange(-40e3, 40001, 1000.0)
X, Y = np.meshgrid(x, y)
cooling = -5.0 * np.exp(-((X / 15e3) ** 2 + (Y / 25e3) ** 2))  # K
temperature = xr.DataArray(
    293.0 + cooling, dims=("y", "x"), coords={"y": y, "x": x}, name="temperature"
)
b = buoyancy(temperature, 293.0)
wind_u = xr.full_like(temperature, 8.8)
wind_v = xr.full_like(temperature, 10.0)
gen = baroclinic_generation(b, u=wind_u, v=wind_v, storm_motion=line_motion)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
(gen.horizontal_vorticity_generation * 1e6).plot(ax=ax[0], cmap="magma")
q = 6
ax[0].quiver(
    x[::q], y[::q], gen.vorticity_x_generation[::q, ::q], gen.vorticity_y_generation[::q, ::q],
    color="w",
)
ax[0].set_title(r"$|d\omega_h/dt|$ ($10^{-6}$ s$^{-2}$) and its direction")
(gen.streamwise_vorticity_generation * 1e6).plot(ax=ax[1], cmap="RdBu_r")
ax[1].set_title("streamwise part ($10^{-6}$ s$^{-2}$)")
for a in ax:
    a.set_aspect("equal")
```

The generated vorticity points along the edge of the cold pool, clockwise
around a cold anomaly seen from above (on every edge the sense of the
overturning that carries cold air outward below and warm air up over it);
where the storm-relative inflow runs along that edge the generated vorticity
is streamwise.

## Field data

The readers convert the archived PERiLS formats directly, for example:

```python
from radarx.io import read_mrr, read_pips, read_sticknet, read_wind_profiler

stations = xr.concat(
    [
        read_sticknet("TTU_StickNet/", iop=2),  # *_IOP2_level3.txt + locations
        read_pips("PIPS_data/IOP2_033022/netcdf"),  # conventional_raw_*.nc
    ],
    dim="station",
    join="outer",
)
mrr = read_mrr("RaDAPS_MRR_20220330.nc", latitude=33.5956, longitude=-88.9879, altitude=87.0)
rwp = read_wind_profiler("RADAPS_915_20220330_05.nc", min_qc=1)
shear = bulk_shear(rwp, 250, 2500, ground=rwp.altitude)  # one value per profile
```

## References

- Browning, K. A., and R. Wexler, 1968: The determination of kinematic
  properties of a wind field using Doppler radar. *J. Appl. Meteor.*, **7**,
  105-113, https://doi.org/10.1175/1520-0450(1968)007<0105:TDOKPO>2.0.CO;2
- Bryan, G. H., J. C. Knievel, and M. D. Parker, 2006: A multimodel assessment
  of RKW theory's relevance to squall-line characteristics. *Mon. Wea. Rev.*,
  **134**, 2772-2792, https://doi.org/10.1175/MWR3226.1
- Bunkers, M. J., B. A. Klimowski, J. W. Zeitler, R. L. Thompson, and
  M. L. Weisman, 2000: Predicting supercell motion using a new hodograph
  technique. *Wea. Forecasting*, **15**, 61-79,
  https://doi.org/10.1175/1520-0434(2000)015<0061:PSMUAN>2.0.CO;2
- Davies-Jones, R., 1984: Streamwise vorticity: The origin of updraft rotation
  in supercell storms. *J. Atmos. Sci.*, **41**, 2991-3006,
  https://doi.org/10.1175/1520-0469(1984)041<2991:SVTOOU>2.0.CO;2
- Rotunno, R., J. B. Klemp, and M. L. Weisman, 1988: A theory for strong,
  long-lived squall lines. *J. Atmos. Sci.*, **45**, 463-485,
  https://doi.org/10.1175/1520-0469(1988)045<0463:ATFSLL>2.0.CO;2
- Weisman, M. L., and R. Rotunno, 2004: "A theory for strong long-lived squall
  lines" revisited. *J. Atmos. Sci.*, **61**, 361-382,
  https://doi.org/10.1175/1520-0469(2004)061<0361:ATFSLS>2.0.CO;2
