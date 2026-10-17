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

# Diabatic Lagrangian Analysis of a Real Squall Line

+++

The [diabatic Lagrangian notebook](Diabatic_Lagrangian) runs
`radarx.retrieve.diabatic_lagrangian` on a synthetic storm with known winds.
Here the same function runs on the squall line that crossed Mississippi on
30 March 2022 (PERiLS IOP 2), seen by the KGWX radar at Columbus, from four
volumes between 23:46 and 00:09 UTC, one volume every 6.5 minutes. Everything
is open data: the NEXRAD volumes and the ERA5 reanalysis can be read without
credentials, and the surface stations that serve as the independent check are
the ASOS and AWOS stations of the Iowa Environmental Mesonet.

The run has four steps:

1. dealias the volumes, grid them and retrieve the wind on a 2-km grid,
2. take the environment of the storm from ERA5 and check it against the
   surface stations ahead of the line,
3. run the analysis and plot buoyancy, potential temperature, vapour and cloud
   water,
4. compare the retrieved cooling with the stations.

The result is a demonstration of the workflow and a test of how far
NEXRAD-only winds take the method, not a validation of the method. The text
says where it falls short.

```{code-cell} ipython3
import tempfile
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xradar as xd
from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

import radarx as rx
from radarx.io import sounding
from radarx.io.aws_data import download_file
from radarx.retrieve import diabatic_lagrangian
```

## Winds from KGWX

The four KGWX volumes are read from the public NEXRAD archive on AWS (they are
also in `open-radar-data`). The Nyquist velocity comes from the radial
headers, the no-data codes are masked, and the velocity is dealiased with a
limit of 50 m/s on the radial speed.

```{code-cell} ipython3
keys = [
    "KGWX20220330_234639_V06",
    "KGWX20220330_235324_V06",
    "KGWX20220330_235959_V06",
    "KGWX20220331_000624_V06",
]
cache = tempfile.mkdtemp()


def nexrad_volume(key):
    day = f"{key[4:8]}/{key[8:10]}/{key[10:12]}"
    path = download_file("unidata-nexrad-level2", f"{day}/{key[:4]}/{key}", cache)
    with NEXRADLevel2File(path) as nf:
        nyquist = [h["msg_31_data_header"]["RAD"]["nyquist_vel"] / 100.0 for h in nf.msg_31_data_header]
    tree = xd.io.open_nexradlevel2_datatree(path)
    for i, name in enumerate(n for n in tree.children if n.startswith("sweep")):
        ds = tree[name].to_dataset()
        for field, floor in {"DBZH": -32.0, "VRADH": -63.9, "ZDR": -12.9}.items():
            if field in ds:
                ds[field] = ds[field].where(ds[field] > floor)
        tree[name] = ds.assign_coords(nyquist_velocity=nyquist[i])
    return tree.radarx.assign(tree.radarx.dealias("VRADH", name="VRADH", max_speed=50.0))


volumes = [nexrad_volume(key) for key in keys]
```

The analysis grid is 2 km in the horizontal and 500 m in the vertical,
120 km by 180 km, west of the radar where the line is. The lowest level is the
ground (0 m above sea level, a little below the terrain of 70 to 150 m); the
radar fields and winds of the lowest beam height fill the levels below it.

`multi_doppler` does not need a second radar: with one radar, the wind is the
smoothest field that satisfies mass continuity, reproduces the radial
velocity, and stays close to the ERA5 background wind where the beam says
nothing (the component across the beam). The neighbouring NEXRAD radars do not
help for this line: KBMX is 190 km from it and its lowest beam passes at about
4 km, and a dual-Doppler analysis with KGWX and KBMX, as in the
[multi-Doppler notebook](Multi_Doppler), has no cell with a beam crossing
angle above 30 degrees at 1 km and none near the line at 3 km. The winds of the lowest kilometre, which the
analysis needs most, therefore rest on one Doppler component and the
reanalysis.

```{code-cell} ipython3
x = np.arange(-100e3, 20e3 + 1, 2000.0)
y = np.arange(-100e3, 80e3 + 1, 2000.0)
z = np.arange(0.0, 10e3 + 1, 500.0)
polar_kw = dict(x_lim=(x[0], x[-1]), y_lim=(y[0], y[-1]), z_lim=(z[0], z[-1]), x_step=2000.0, y_step=2000.0, z_step=500.0)

steps, background = [], None
for volume in volumes:
    fields = volume.radarx.to_grid(["DBZH", "ZDR"], **polar_kw).rename(DBZH="DBZ").drop_vars(["lat", "lon"])
    grid = rx.retrieve.multi_doppler_input([volume], x=x, y=y, z=z, fill_below=True)
    if background is None:  # ERA5 at 00 UTC (the hour nearest to all four volumes), on the grid
        background = sounding.era5_column(grid, time="2022-03-31T00:00", source="gcs")
    wind = grid.radarx.multi_doppler(background)
    step = xr.Dataset({"u": wind.u, "v": wind.v, "w": wind.w, "DBZ": fields.DBZ, "ZDR": fields.ZDR})
    step = step.drop_vars("time", errors="ignore").expand_dims(time=[fields.time.values.astype("datetime64[s]").astype("datetime64[ns]")])
    steps.append(step)
winds = xr.concat(steps, "time")
for name in ("DBZ", "ZDR"):  # below the lowest beam
    winds[name].loc[dict(z=0.0)] = winds[name].sel(z=500.0).values
print(winds.time.values.astype("datetime64[s]"))
print(f"RMS radial velocity misfit of the last retrieval: {float(np.sqrt((wind.vr_residual ** 2).mean())):.2f} m/s")
```

## The environment and a check against the stations

The trajectories start from an environmental profile. ERA5 at 00 UTC, at a
point just ahead of the line, gives it from the anonymous Google ARCO store (a
cold read of that store takes minutes; the documentation build and the notebook
tests start from a small pre-extracted subset of it).
Five surface stations in the area report every 20 minutes or better. Ahead of
the gust front, before 00 UTC, they show the air that feeds the line:

```{code-cell} ipython3
profile = sounding.era5_profile(33.5, -88.0, "2022-03-31T00:00", source="gcs")

sites = ["STF", "GTR", "CBM", "M40", "TUP"]
url = (
    "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
    + "&".join(f"station={s}" for s in sites)
    + "&data=tmpf&data=dwpf&data=alti&year1=2022&month1=3&day1=30&year2=2022&month2=3&day2=31"
    + "&hour1=22&hour2=2&tz=Etc/UTC&format=onlycomma&latlon=yes&missing=M&trace=T&direct=no"
    + "&report_type=3&report_type=4"
)
obs = pd.read_csv(urllib.request.urlopen(url), na_values="M", parse_dates=["valid"]).dropna(subset=["tmpf", "dwpf", "alti"])
obs["x"] = (obs.lon - float(winds.longitude)) * 111.2e3 * np.cos(np.radians(float(winds.latitude)))
obs["y"] = (obs.lat - float(winds.latitude)) * 111.2e3

# Station pressure is not reported: 990 hPa scaled with the altimeter setting is
# accurate enough for temperature changes of a few kelvin.
obs["pressure"] = 99000.0 * obs.alti / 29.5
T = (obs.tmpf.values - 32.0) * 5 / 9 + 273.15
Td = (obs.dwpf.values - 32.0) * 5 / 9 + 273.15
q = sounding.specific_humidity_from_dewpoint(xr.DataArray(Td), xr.DataArray(obs.pressure.values)).values
r = q / (1.0 - q)
obs["qv"] = r * 1e3  # g/kg
obs["theta_v"] = T * (1e5 / obs.pressure.values) ** 0.2857 * (1 + r / 0.622) / (1 + r)

prior = obs[(obs.valid >= "2022-03-30 23:30") & (obs.valid < "2022-03-31 00:00") & obs.station.isin(["GTR", "M40", "CBM"])]
era5 = profile.isel(height=0)  # the surface level of the profile
era5_r = float(era5.specific_humidity / (1 - era5.specific_humidity))
era5_thv = float(era5.temperature) * (1e5 / float(era5.pressure)) ** 0.2857 * (1 + era5_r / 0.622) / (1 + era5_r)
print(f"stations ahead of the line, 23:30-00:00 UTC (n = {len(prior)}): theta_v {prior.theta_v.mean():.1f} K, q_v {prior.qv.mean():.1f} g/kg")
print(f"ERA5 surface level at 00 UTC:                            theta_v {era5_thv:.1f} K, q_v {era5_r * 1e3:.1f} g/kg")
```

ERA5 is 2.8 K warmer in $\theta_v$ and 3 g kg$^{-1}$ drier than the stations at the
surface: it does not reproduce the moist boundary layer of this case. The
absolute values of the analysis inherit that offset from the environment, but
the change that the storm makes is much less sensitive to it. The comparison
with the stations below therefore uses that change, the buoyancy relative to
the environment, and not the absolute values.

## Running the analysis

The trajectories use the options for squall lines of the
[synthetic notebook](Diabatic_Lagrangian): a storm motion estimated from the
reflectivity, termination outside the echo and ahead of the line, and 45 min
of time morphing before the first volume, because the wind series covers only
20 min and the inflow air is older than that. Ahead of the line is the region
more than 6 km east of the leading 40-dBZ edge at 1 km, found row by row:

```{code-cell} ipython3
def leading_edge(dbz):
    """Easternmost 40-dBZ cell in every row (m), smoothed along the line."""
    edge = np.full(dbz.sizes["y"], np.nan)
    for j, row in enumerate(dbz.values):
        strong = np.flatnonzero(row >= 40.0)
        if strong.size:
            edge[j] = dbz.x.values[strong.max()]
    edge = np.interp(dbz.y.values, dbz.y.values[np.isfinite(edge)], edge[np.isfinite(edge)])
    return pd.Series(edge).rolling(11, center=True, min_periods=1).median().values


edges = np.stack([leading_edge(winds.DBZ.sel(z=1000.0).isel(time=k)) for k in range(winds.sizes["time"])])
ahead = winds.x > xr.DataArray(edges, dims=("time", "y"), coords={"time": winds.time, "y": winds.y}) + 6e3
winds["ahead"] = ahead.broadcast_like(winds.u).transpose("time", "z", "y", "x")

options = dict(
    storm_motion="estimate",
    termination="precipitation",
    environment_mask="ahead",
    extend_before=2700.0,
    parameters={"min_steps": 0, "env_dbz": 15.0},
)
dla = diabatic_lagrangian(winds, profile, **options)
print(f"analysis time {str(dla.time.values)[:19]} UTC, storm motion {dla.attrs['storm_motion'][0]:.1f}, {dla.attrs['storm_motion'][1]:.1f} m/s")
print(f"{float(dla.environment.sel(z=0).mean()):.0%} of the surface trajectories reached the environment")
```

The precipitation closure takes $Z_H$ and $Z_{DR}$ (used as received, without
a $Z_{DR}$ calibration) and has no hydrometeor classification here, so graupel
appears only above the melting level and where the rain retrieval fails at
40 dBZ or more.

```{code-cell} ipython3
def sites_xy(ax):
    for name, group in obs.groupby("station"):
        ax.plot(group.x.iloc[0] / 1e3, group.y.iloc[0] / 1e3, "k^", ms=5)
        ax.text(group.x.iloc[0] / 1e3 + 1.5, group.y.iloc[0] / 1e3 + 1.5, name, fontsize=8)


# the change of theta and q_v from the environmental profile
dla["delta_theta"] = dla.theta - dla.theta_base
q_profile = profile.specific_humidity / (1.0 - profile.specific_humidity)
dla["delta_qv"] = (dla.qv - xr.DataArray(np.interp(dla.z, profile.height, q_profile), dims="z", coords={"z": dla.z})) * 1e3

sfc = dla.sel(z=0.0)
sec = dla.sel(y=-20e3, method="nearest")
dbz_sfc = winds.DBZ.isel(time=-1).sel(z=1000.0)
dbz_sec = winds.DBZ.isel(time=-1).sel(y=sec.y.values, method="nearest")

panels = [
    ("delta_theta_v", r"$\Delta\theta_v$ (K)", "RdBu_r", 6),
    ("delta_theta", r"$\Delta\theta$ (K)", "RdBu_r", 6),
    ("delta_qv", r"$\Delta q_v$ (g kg$^{-1}$)", "BrBG", 4),
]
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), layout="constrained")
for col, (name, label, cmap, lim) in enumerate(panels):
    top, bottom = axes[0, col], axes[1, col]
    im = top.pcolormesh(dla.x / 1e3, dla.y / 1e3, sfc[name], cmap=cmap, vmin=-lim, vmax=lim)
    top.contour(dla.x / 1e3, dla.y / 1e3, dbz_sfc, [40], colors="k", linewidths=0.8)
    sites_xy(top)
    top.set(title=f"{label.split(' (')[0]} at the ground", xlabel="x (km east of KGWX)", ylabel="y (km north of KGWX)", aspect="equal")
    fig.colorbar(im, ax=top, label=label, shrink=0.8)
    im = bottom.pcolormesh(dla.x / 1e3, dla.z / 1e3, sec[name], cmap=cmap, vmin=-lim, vmax=lim)
    bottom.contour(dla.x / 1e3, dla.z / 1e3, dbz_sec, [20, 40], colors="k", linewidths=0.8)
    bottom.contour(dla.x / 1e3, dla.z / 1e3, sec.qc * 1e3, [0.5, 1.0, 2.0], colors="m", linewidths=1.0)
    bottom.set(title=f"{label.split(' (')[0]}, y = {float(sec.y) / 1e3:.0f} km", xlabel="x (km east of KGWX)", ylabel="height (km)", ylim=(0, 6))
    fig.colorbar(im, ax=bottom, label=label, shrink=0.8)
fig.suptitle("Analysis at 00:09 UTC. Black: 40 dBZ at 1 km (maps), 20 and 40 dBZ (sections). Magenta: cloud water 0.5, 1, 2 g/kg")
plt.show()
print(f"surface: minimum delta theta_v {float(sfc.delta_theta_v.min()):.1f} K, "
      f"maximum delta q_v {float(sfc.delta_qv.max()):.1f} g/kg, "
      f"maximum cloud water {float(dla.qc.max()) * 1e3:.1f} g/kg, "
      f"maximum delta theta_v aloft {float(dla.delta_theta_v.max()):.1f} K")
```

At the ground the analysis cools the air by 3 to 5.5 K in a band that follows
the line and reaches 30 km behind it between y = -30 and 0 km, with the matching
moistening of 1 to 3 g kg$^{-1}$ that rain evaporation brings. The cooling is weak
south of y = -40 km, near STF and GTR. In the section the cold, moist air lies
under the echo below 1 km. The cold layer at 2 km is boundary-layer air that
was lifted into the warm layer above it, which is the cap of the ERA5 profile (for
example the parcel at x = -60 km, 2 km started at 350 m), so it is cold
relative to that layer.

Aloft the analysis is not credible. In the updraft at the leading edge
$\Delta\theta_v$ is above +6 K from 3 km up and reaches +19 K at its maximum
(printed above), which is what an undiluted parcel gets when it condenses all
its vapour. Vertical velocity from one Doppler component is the least
reliable part of the winds, and the analysis has no entrainment. Cloud water
comes out mostly in the cooler air west of the updraft, not in it.

## Against the stations

The station $\theta_v$ change at the surface is the value at the time of the
analysis minus the station's mean before 23:30 UTC, when the gust front had
not reached them. The analysis change is `delta_theta_v` at the
grid point nearest to the station, run for each of the four volumes. Only observations within 7 minutes of the
analysis are used.

```{code-cell} ipython3
before = obs[obs.valid < "2022-03-30 23:30"].groupby("station").theta_v.mean()
rows = []
for t in winds.time.values:
    run = diabatic_lagrangian(winds, profile, time=t, levels=[0], **options).sel(z=0.0)
    for name, group in obs.groupby("station"):
        lag = (group.valid - pd.Timestamp(t)).dt.total_seconds().abs()
        if lag.min() > 420.0 or name not in before:
            continue
        o = group.loc[lag.idxmin()]
        rows.append(
            dict(
                time=pd.Timestamp(t).strftime("%H:%M"),
                station=name,
                observed=o.theta_v - before[name],
                analysis=float(run.delta_theta_v.sel(x=o.x, y=o.y, method="nearest")),
            )
        )
table = pd.DataFrame(rows)
table["error"] = table.analysis - table.observed
table["region"] = np.where(table.observed < -3.0, "cold pool", "ahead of the gust front")
print(table.round(1).to_string(index=False))
for region, part in table.groupby("region"):
    print(f"{region}: n = {len(part)}, bias (analysis - observed) {part.error.mean():+.1f} K, "
          f"RMSE {np.sqrt((part.error ** 2).mean()):.1f} K")
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
for name, group in table.groupby("station"):
    ax.plot(group.observed, group.analysis, "o", label=name)
ax.plot([-9, 1], [-9, 1], "k-", lw=0.8)
ax.set(xlim=(-9, 1), ylim=(-9, 1), aspect="equal", xlabel=r"station $\Delta\theta_v$ (K)", ylabel=r"analysis $\Delta\theta_v$ (K)")
ax.legend()
plt.show()
```

## What this shows and what it does not

- The workflow runs end to end on open data in a few minutes: radar volumes,
  one-radar variational winds, a reanalysis environment, and the
  squall-line options of the analysis.
- Ahead of the gust front the analysis changes agree with the stations to
  1.0 K RMS (n = 7, bias +0.5 K): the cooling there is small and the stations
  are few. In the cold pool at STF (observed -5 to -8 K) the analysis gives
  -0.2 to -2.3 K, far too weak; the gust front reached STF before the first
  volume. Behind the line there is little echo left, and the winds there,
  which would carry the cold air down from the line, come from the background
  and not from the radar. Single-Doppler winds near the ground contain no
  downdraft. The cold air behind a squall line needs low-level multi-Doppler
  winds, such as those of the mobile radars of a field campaign (Ziegler
  2013b).
- The few stations (n above) and the 7-minute matching make the numbers a
  consistency check and not a skill score. The stations ahead of the line
  mostly test the environment, and that is where ERA5 is dry and warm.
- The absolute $\theta_v$ and $q_v$ inherit the ERA5 offsets shown above;
  with a better near-surface environment (stations or a nearby sounding, or
  `mesoscale=` and `observations=` of `diabatic_lagrangian`) the same winds
  give different absolute values and nearly the same changes.

## References

- Ziegler, C. L., 2013a: A diabatic Lagrangian technique for the analysis of
  convective storms. Part I. *J. Atmos. Oceanic Technol.*, **30**, 2248–2265,
  doi:10.1175/JTECH-D-12-00194.1
- Ziegler, C. L., 2013b: Part II: Application to a radar-observed storm.
  *J. Atmos. Oceanic Technol.*, **30**, 2266–2280,
  doi:10.1175/JTECH-D-13-00036.1
- Hersbach, H., and Coauthors, 2020: The ERA5 global reanalysis. *Quart. J.
  Roy. Meteor. Soc.*, **146**, 1999–2049, doi:10.1002/qj.3803
