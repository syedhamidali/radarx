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

# End-to-End Radar Analysis Workflow

+++

This notebook runs the radarx analysis pipeline on one case, step by step and
in the natural order, so that you can see how the pieces fit together and what
each step adds. Every step takes the output of the previous one, and the data
stay in xarray (`DataTree` volumes, `Dataset` sweeps and grids) all the way
through.

**Case:** the squall line that crossed Mississippi on 30 March 2022, seen by
the KGWX WSR-88D (Columbus, Mississippi) in two consecutive volumes, 23:53
and 00:00 UTC, and by the KBMX WSR-88D (Birmingham, Alabama) for the
multi-radar steps.

| Step | What it adds | radarx call |
|---|---|---|
| 1 | volumes, Nyquist velocity, georeferencing | xradar, `download_file` |
| 2 | non-meteorological echo removed | `dtree.radarx.echo_mask`, `apply_mask` |
| 3 | environment: ERA5 profile and radiosonde, 0 °C height, humidity, wind-profile and parcel parameters (CAPE, CIN) | `dtree.radarx.sounding`, `read_sounding`, `bulk_shear` |
| 4 | dealiased Doppler velocity and its confidence, a second method (R2D2) and the fold margin, VAD wind profile | `dtree.radarx.dealias`, `dealias_confidence`, `dealias_r2d2`, `dealias_ambiguity`, `vad_profile` |
| 5 | processed ΦDP and KDP, attenuation correction | `dtree.radarx.kdp`, `correct_attenuation` |
| 6 | hydrometeor classes | `dtree.radarx.hid` |
| 7 | rain drop size distribution and rain rate, with the Bayesian posterior | `ds.radarx.dsd`, `dsd_bayesian` |
| 7c | rain rate from $Z_H$, $Z_{DR}$, $K_{DP}$ and $A$, and from the hydrometeor class | `dtree.radarx.rain_rate`, `rain_rate_z`, `rain_rate_kdp`, `rain_rate_z_zdr`, `rain_rate_a` |
| 8 | QVP time series and melting layer, range-defined and columnar profiles, RHI profile, melting layer on PPIs, rain products capped at the melting layer | `qvp_timeseries`, `melting_layer`, `rd_qvp`, `cvp`, `rhi_qvp`, `melting_layer_ppi`, `mask_above_melting_layer` |
| 9 | azimuthal shear and radial divergence, layer-maximum azimuthal shear | `dtree.radarx.llsd`, `azimuthal_shear_layer_max` |
| 10 | 3D grid, Max-CAPPI, other gridding functions, plotting functions, interactive view, UGRID | `dtree.radarx.to_grid`, `grid_cones`, `plot_ppi`, `to_uxarray` |
| 11 | storm motion, common analysis time, time interpolation | `estimate_motion`, `advect`, `interpolate_time` |
| 12 | multi-radar grid, radar bias, ERA5 on the grid, three-dimensional wind | `grid_radars`, `network_bias`, `multi_doppler` |
| 13 | evaporation and evaporative cooling | `evaporation`, `integrate_evaporation` |
| 14 | VIL, liquid VIL, echo top, VIL density, liquid water content and hail products | `vil`, `echo_top`, `vil_density`, `liquid_water_content`, `hail_products`, `shi`, `mesh`, `posh`, `hail_kinetic_energy` |
| 15 | noise level, SNR, noise-corrected RHOHV and ZDR, ZDR, Z and system ΦDP offsets | `noise_level`, `snr`, `correct_rhohv_noise`, `zdr_offset`, `z_offset`, `phidp_offset` |
| 16 | super-observations of reflectivity and velocity with errors, WRFDA `ob.radar` and DART `obs_seq` files | `superobs`, `write_wrfda_radar`, `write_dart_obs_seq` |
| 17 | runtime summary and one `DataTree` with all products | |
| | function index of the whole package | |

The [second part](Radar_Workflow_Advanced) of this notebook shows the first half of
the rest of radarx on small open or synthetic data: the radar equation and unit helpers,
disdrometers, the Bayesian DSD retrieval, raindrop trajectories, surface stations and cold pools,
and lightning. The [third part](Radar_Workflow_Advanced_2) shows tornado detection, single-Doppler
winds, the diabatic Lagrangian analysis, the machine-learning interface, beam blockage from a
digital elevation model, tropical-cyclone winds from one radar, the forward operator, dual-PRF
correction and ensembles. The *Function index* at the end of this page
lists every function of radarx with the section where it is used.

All heavy steps run in compiled, multithreaded kernels; the runtimes in the
summary table are for the machine that built this page, with all cores.

```{code-cell} ipython3
import gc
import time
import warnings
from contextlib import contextmanager

import cmweather  # noqa: F401  radar colormaps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xradar as xd
from matplotlib.colors import BoundaryNorm, ListedColormap
from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.grid import (
    gate_corners,
    grid_cones,
    grid_gates,
    grid_radar,
    grid_radars,
    grid_rhi,
    make_3d_grid,
    merge_radars,
    network_bias,
    stack_data,
)
from radarx.io import (
    air_density,
    dewpoint_from_specific_humidity,
    dewpoint_from_vapor_pressure,
    era5_column,
    era5_profile,
    geopotential_to_height,
    get_s3_client,
    interpolate_profile,
    list_available_files,
    mean_wind,
    nearest_station,
    open_sounding_file,
    parcel_diagnostics,
    read_sounding,
    relative_humidity_from_dewpoint,
    saturation_vapor_pressure,
    sounding,
    specific_humidity_from_dewpoint,
    station_list,
    wet_bulb_temperature,
    write_dart_obs_seq,
    write_wrfda_radar,
)
from radarx.io.aws_data import download_file
from radarx.retrieve import (
    advect,
    apply_mask,
    attenuation_coefficients,
    azimuthal_shear,
    azimuthal_shear_layer_max,
    bulk_shear,
    bunkers_storm_motion,
    correct_attenuation,
    correct_rhohv_noise,
    correct_zdr_noise,
    create_cappi,
    cvp,
    dealias_ambiguity,
    dealias_confidence,
    dealias_r2d2,
    dealias_velocity,
    dsd_bayesian,
    dsd_prior,
    dual_doppler_analytic,
    echo_mask,
    echo_top,
    estimate_kdp,
    estimate_motion,
    evaporation,
    forward_grid,
    hail_kinetic_energy,
    hail_products,
    hid,
    hid_classes,
    integrate_evaporation,
    interpolate_time,
    layer_mean_wind,
    liquid_water_content,
    llsd,
    mask_above_melting_layer,
    melting_layer,
    melting_layer_ppi,
    mesh,
    multi_doppler_input,
    noise_level,
    phidp_offset,
    posh,
    qvp,
    qvp_timeseries,
    rd_qvp,
    radar_geometry,
    radial_divergence,
    rain_rate,
    rain_rate_a,
    rain_rate_kdp,
    rain_rate_z,
    rain_rate_z_zdr,
    rhi_qvp,
    shi,
    snr,
    storm_relative_helicity,
    storm_relative_wind,
    superobs,
    vad_profile,
    vil,
    vil_density,
    z_offset,
    zdr_offset,
)
from radarx.vis import (
    RadarxDataArrayPlotAccessor,
    RadarxDatasetPlotAccessor,
    RadarxDataTreePlotAccessor,
    hvplot_centroids,
    hvplot_cappi,
    hvplot_max_cappi,
    hvplot_mesh,
    hvplot_ppi,
    hvplot_range_azimuth,
    hvplot_rhi,
    plot_cappi,
    plot_maxcappi,
    plot_ppi,
    plot_rhi,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="The input coordinates to pcolormesh")

timings = {}


@contextmanager
def timed(step):
    """Record the wall-clock time of a step."""
    start = time.perf_counter()
    yield
    timings[step] = time.perf_counter() - start
    print(f"{step}: {timings[step]:.2f} s")
```

```{note}
This notebook downloads and processes three NEXRAD volumes (about 2 GB of
memory and a few minutes). It is executed and tested in the radarx continuous
integration on every change, and the documentation website shows the outputs
of that run.
```

## 1. Read the volumes

The volumes come from the NOAA NEXRAD archive on AWS. xradar reads each file
into a `DataTree` with one node per sweep. Three NEXRAD details matter for
what follows:

- the low elevations are scanned twice, a long-range *surveillance* cut
  (reflectivity and polarimetric fields) and a *Doppler* cut with a higher
  Nyquist velocity (velocity);
- with SAILS, the 0.5° pair is repeated during the volume. The repeats are
  useful for nowcasting but not here, so we read only the first one
  (`sweep=`), together with the fields we use (`drop_variables=` drops the
  spectrum width and the clutter correction);
- the surveillance cuts reach 460 km, far beyond what we analyse, so we keep
  the first 200 km (`max_range`), and the moments, which are 8- and 16-bit
  codes, and the gate positions in single precision. Together this cuts the
  memory of a volume by a factor of five;
- xradar does not yet expose the Nyquist velocity. We read it from the radial
  headers and attach it to each sweep as the `nyquist_velocity` coordinate,
  where radarx looks for it.

The no-data codes are masked in the next step.

```{code-cell} ipython3
keys = [
    "2022/03/30/KGWX/KGWX20220330_235324_V06",
    "2022/03/30/KGWX/KGWX20220330_235959_V06",
]
paths = [download_file("unidata-nexrad-level2", key, "data") for key in keys]


def read_nexrad(path, fields=("DBZH", "ZDR", "PHIDP", "RHOHV", "VRADH"), max_range=200e3):
    """Read the fields of a NEXRAD volume without the SAILS repeats."""
    with NEXRADLevel2File(path) as nf:
        nyquist = [
            h["msg_31_data_header"]["RAD"]["nyquist_vel"] / 100.0
            for h in nf.msg_31_data_header
        ]
        # fixed angles of the scan strategy; AVSET may end the volume early
        angles = [cut["elevation_angle"] for cut in nf.msg_5["elevation_data"]]
        angles = angles[: len(nyquist)]
    # keep the first cut (or surveillance/Doppler pair) of every elevation
    keep = []
    for i, angle in enumerate(angles):
        if angle not in angles[:i] or (i - 1 in keep and angles[i - 1] == angle):
            keep.append(i)
    drop = ["WRADH", "CCORH", "CFP"] + [
        v for v in ("DBZH", "ZDR", "PHIDP", "RHOHV", "VRADH") if v not in fields
    ]
    dtree = xd.io.open_nexradlevel2_datatree(path, sweep=keep, drop_variables=drop)
    for name in [n for n in dtree.children if n.startswith("sweep")]:
        ds = dtree[name].to_dataset().sel(range=slice(None, max_range))
        ds = ds.assign({v: ds[v].astype("float32") for v in fields if v in ds})
        index = int(ds.sweep_number)
        dtree[name] = ds.load().assign_coords(nyquist_velocity=nyquist[index])
    # gate positions (m) in single precision too
    return dtree.xradar.georeference().map_over_datasets(
        lambda ds: ds.assign_coords({c: ds[c].astype("float32") for c in "xyz" if c in ds.coords})
    )


with timed("1. read, Nyquist, georeference"):
    raw = [read_nexrad(path) for path in paths]

for vol in raw:
    sweeps = vol.match("sweep_*").children
    print(
        vol.attrs["instrument_name"],
        str(vol.time_coverage_start.values),
        f"{len(sweeps)} sweeps:",
        " ".join(f"{float(vol[n].sweep_fixed_angle):.1f}" for n in sweeps),
    )
```

Which volumes does the archive hold for this evening? `list_available_files`
lists the keys of the public bucket with the anonymous S3 client of
`get_s3_client`, the same client that `download_file` uses:

```{code-cell} ipython3
client = get_s3_client()
available = list_available_files("unidata-nexrad-level2", "2022/03/30/KGWX/KGWX20220330_23")
print(f"{len(available)} KGWX volumes between 23:00 and 23:59 UTC, the first: {available[0]}")
```

## 2. Remove non-meteorological echo

Ahead of the line, insects and birds fill the boundary layer with weak echo
of high $Z_{DR}$ and low $\rho_{hv}$, and ground clutter surrounds the radar.
`dtree.radarx.echo_mask` classifies every gate of a volume with a fuzzy-logic
score (Gourley et al. 2007; Krause 2016); the Doppler cuts, which have no
polarimetric fields, take the class of the surveillance cut at the same
elevation. `dtree.radarx.apply_mask` then sets non-meteorological gates and
the NEXRAD no-data codes to NaN in every field. Everything downstream works on
the cleaned volumes.

```{code-cell} ipython3
with timed("2. echo classification and masking"):
    echo = [vol.radarx.echo_mask() for vol in raw]
    volumes = [vol.radarx.apply_mask(qc) for vol, qc in zip(raw, echo)]
```

```{code-cell} ipython3
def ppi(ax, ds, var, extent=200, **kwargs):
    """Plot a sweep field in km around the radar."""
    rays = np.argsort(ds.azimuth.values)  # no seam where the scan started
    pm = ax.pcolormesh(
        ds.x.values[rays] / 1e3, ds.y.values[rays] / 1e3, ds[var].values[rays], **kwargs
    )
    ax.set_aspect("equal")
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_xlabel("east of KGWX (km)")
    return pm


before = raw[1]["sweep_0"].to_dataset()
before = before.assign(DBZH=before.DBZH.where(before.DBZH > -32))
classes = echo[1]["sweep_0"].to_dataset().ECHO_CLASS
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True, layout="constrained")
pm = ppi(axes[0], before, "DBZH", cmap="ChaseSpectral", vmin=-10, vmax=70)
axes[0].set_title("$Z_H$ as measured")
pc = ppi(
    axes[1],
    classes.where(classes > 0).to_dataset(),
    "ECHO_CLASS",
    cmap=ListedColormap(["#2a78d6", "#eb6834", "#eda100"]),
    vmin=0.5,
    vmax=3.5,
)
axes[1].set_title("echo class")
ppi(axes[2], volumes[1]["sweep_0"].to_dataset(), "DBZH", cmap="ChaseSpectral", vmin=-10, vmax=70)
axes[2].set_title("$Z_H$ after apply_mask")
axes[0].set_ylabel("north of KGWX (km)")
for ax in axes[[0, 2]]:
    fig.colorbar(pm, ax=ax, label="reflectivity (dBZ)", shrink=0.85)
cb = fig.colorbar(pc, ax=axes[1], ticks=[1, 2, 3], shrink=0.85)
cb.ax.set_yticklabels(["meteorological", "non-met.", "speckle"])
fig.suptitle("KGWX 0.5°, 00:00 UTC")
plt.show()
```

Every accessor method is a thin wrapper of a function of `radarx.retrieve`,
which takes the same volume (or sweep) and the same arguments. `echo_mask` and
`apply_mask` give identical results:

```{code-cell} ipython3
echo_fn = echo_mask(raw[1])
masked_fn = apply_mask(raw[1], echo_fn)
same = np.array_equal(
    masked_fn["sweep_0"].DBZH.values, volumes[1]["sweep_0"].DBZH.values, equal_nan=True
)
print("function and accessor give the same masked reflectivity:", same)

# the raw volumes and the classification are not needed any more
del raw, echo, echo_fn, masked_fn, vol, before, classes
gc.collect();
```

## 3. Environment

Several later steps need the environment around the radar: a wind profile is
the reference for dealiasing (step 4), the temperature at every gate is an
input of the hydrometeor classification (step 6) and the 0 °C height the
reference for the melting layer (step 8). `dtree.radarx.sounding` reads the
ERA5 profile at the radar site (here from Google's anonymous analysis-ready
ERA5 store, `era5_source="gcs"`; the ECMWF ARCO time series and the Copernicus
CDS need an account) or the nearest observed radiosonde. We take ERA5 at
00 UTC and the 00 UTC Birmingham sounding.

```{note}
ERA5 data are downloaded the first time (one hour of the global field, which
takes a minute or two) and cached afterwards; the timing below is for a warm
cache.
```

```{code-cell} ipython3
with timed("3. environment (ERA5, sounding)"):
    era5 = volumes[0].radarx.sounding(era5_source="gcs", time="2022-03-31T00:00")
    raob = volumes[0].radarx.sounding("iem")
    freezing_level = float(sounding.isotherm_height(era5))
    wet_bulb_zero = float(sounding.wet_bulb_zero_height(era5))
print(f"ERA5 at KGWX: 0 °C at {freezing_level:.0f} m, wet-bulb 0 °C at {wet_bulb_zero:.0f} m")
print(
    f"{raob.attrs['station']} sounding: 0 °C at {float(sounding.isotherm_height(raob)):.0f} m"
)
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True, layout="constrained")
for prof, label, color in [(era5, "ERA5 at KGWX", "C0"), (raob, f"{raob.attrs['station']} radiosonde", "C1")]:
    p = prof.where(prof.height < 12e3, drop=True)
    t, td, ws = (p[v].dropna("height") for v in ("temperature", "dewpoint", "wind_speed"))
    axes[0].plot(t - 273.15, t.height / 1e3, color=color, label=label)
    axes[0].plot(td - 273.15, td.height / 1e3, color=color, ls="--")
    axes[1].plot(ws, ws.height / 1e3, color=color)
    axes[2].plot(p.wind_direction.sel(height=ws.height), ws.height / 1e3, ".", color=color, ms=4)
axes[0].axvline(0, color="k", lw=0.8)
axes[0].axhline(freezing_level / 1e3, color="C0", ls=":", label="ERA5 0 °C height")
axes[0].set(xlabel="temperature (solid), dew point (dashed) (°C)", ylabel="height above sea level (km)")
axes[1].set(xlabel="wind speed (m/s)")
axes[2].set(xlabel="wind direction (°)", xlim=(0, 360), xticks=range(0, 361, 90))
axes[0].legend(loc="lower left")
fig.suptitle("Environment, 31 March 2022 00 UTC")
plt.show()
```

### Sounding utilities and wind-profile parameters

`dtree.radarx.sounding` is built from the functions of `radarx.io`, which can
be called on their own. `nearest_station` finds the radiosonde stations around
the radar from the catalogue `station_list`, `read_sounding` downloads one
sounding and `era5_profile` reads the ERA5 profile at a point.
`open_sounding_file` reads a sounding from a local file (here the sounding is
written as CSV and read back; SHARPpy, University of Wyoming, IGRA2 and IEM
files work the same way):

```{code-cell} ipython3
lat, lon = float(volumes[0].latitude), float(volumes[0].longitude)
near = nearest_station(lat, lon, "2022-03-31T00:00", source="iem", n=3)
print(f"{station_list().sizes['station']} stations in the catalogue; nearest to KGWX:")
print(near[["iem_id", "name", "distance"]].to_dataframe().assign(distance=lambda d: (d.distance / 1e3).round()))

sonde = read_sounding("KBMX", "2022-03-31T00:00", source="iem")
era5_point = era5_profile(lat, lon, "2022-03-31T00:00", source="gcs")
table = pd.DataFrame(
    {
        "pressure": raob.pressure / 100,
        "height": raob.geopotential_height,
        "temperature": raob.temperature - 273.15,
        "dewpoint": raob.dewpoint - 273.15,
        "u": raob.u,
        "v": raob.v,
    }
).dropna()
table.to_csv("raob.csv", index=False)
from_file = open_sounding_file("raob.csv", format="csv")
print(
    "read_sounding and era5_profile reproduce the profiles of step 3:",
    bool(sonde.temperature.equals(raob.temperature)),
    bool(np.allclose(era5_point.temperature, era5.temperature)),
)
print(f"0 °C height from the CSV file: {float(sounding.isotherm_height(from_file)):.0f} m")
```

The thermodynamic helpers convert between the humidity variables of a
profile (all take and return arrays, with a compiled kernel behind them):

```{code-cell} ipython3
low = raob.isel(height=slice(1, 60))  # the lowest 6 km
es = saturation_vapor_pressure(low.temperature)
rh = relative_humidity_from_dewpoint(low.temperature, low.dewpoint)
q = specific_humidity_from_dewpoint(low.dewpoint, low.pressure)
td_q = dewpoint_from_specific_humidity(q, low.pressure)
td_e = dewpoint_from_vapor_pressure(saturation_vapor_pressure(low.dewpoint))
rho = air_density(low.pressure, low.temperature, q)
tw = wet_bulb_temperature(low.pressure, low.temperature, low.dewpoint)
z_geo = geopotential_to_height(9.80665 * low.geopotential_height)
print(f"round trips: dew point from q {float(abs(td_q - low.dewpoint).max()):.3f} K, "
      f"from vapour pressure {float(abs(td_e - low.dewpoint).max()):.3f} K; "
      f"surface air density {float(rho.dropna('height')[0]):.3f} kg/m3; "
      f"geopotential to geometric height {float((z_geo - low.geopotential_height).max()):.1f} m at 6 km")
valid = raob.where(np.isfinite(raob.u), drop=True)  # the lowest level has no wind
levels = interpolate_profile(valid, [1000.0, 2000.0, 3000.0], ["temperature", "u", "v"])
print(f"0-6 km layer-mean wind, from mean_wind: {float(mean_wind(valid, 0, 6000).wind_speed):.1f} m/s")
levels
```

The wind-profile parameters of the environment, the inflow of a squall line,
from the sounding and from ERA5: the bulk shear of a layer (`bulk_shear`), the
layer-mean wind (`layer_mean_wind`), the supercell motion of Bunkers et al.
(2000) (`bunkers_storm_motion`), the storm-relative wind (`storm_relative_wind`)
and the storm-relative helicity (`storm_relative_helicity`). The line itself
moves toward the east-south-east; the hodograph shows the 0 to 6 km winds
relative to it:

```{code-cell} ipython3
line_motion = (8.8, -3.2)  # m/s, squall line moving toward the east-south-east
rows = {}
for name, prof in (("BMX sounding", valid), ("ERA5 at KGWX", era5)):
    rm = bunkers_storm_motion(prof)
    rows[name] = {
        "0-6 km shear (m/s)": float(bulk_shear(prof, 0, 6000).shear_speed),
        "0-3 km shear (m/s)": float(bulk_shear(prof, 0, 3000, normal=110.0).shear_normal),
        "0-6 km mean wind (m/s)": float(np.hypot(*(float(layer_mean_wind(prof, 0, 6000)[c]) for c in "uv"))),
        "Bunkers right mover u (m/s)": float(rm.u),
        "SRH 0-3 km, line motion (m2/s2)": float(storm_relative_helicity(prof, line_motion, 0, 3000)),
        "SRH 0-3 km, Bunkers (m2/s2)": float(storm_relative_helicity(prof, "right", 0, 3000)),
    }
display(pd.DataFrame(rows).round(1))
via_accessors = (valid.radarx.bulk_shear(0, 6000).shear_speed, valid.radarx.storm_relative_helicity(line_motion, 0, 3000))
print("the accessors of a sounding agree:",
      bool(np.isclose(float(via_accessors[0]), rows["BMX sounding"]["0-6 km shear (m/s)"])),
      bool(np.isclose(float(via_accessors[1]), rows["BMX sounding"]["SRH 0-3 km, line motion (m2/s2)"])),
      bool(np.isclose(float(valid.radarx.bunkers_storm_motion().u), float(bunkers_storm_motion(valid).u))))

low = valid.where(valid.height - valid.height[0] <= 6000, drop=True)
rel = storm_relative_wind(low, line_motion)
rm = bunkers_storm_motion(valid)
fig, ax = plt.subplots(figsize=(5.2, 5), layout="constrained")
sc = ax.scatter(low.u, low.v, c=(low.height - low.height[0]) / 1e3, s=8, cmap="viridis", zorder=3)
ax.plot(low.u, low.v, "k-", lw=0.5)
ax.plot(*line_motion, "s", color="C3", label="squall line motion")
ax.plot(float(rm.u), float(rm.v), "^", color="C0", label="Bunkers right mover")
ax.axhline(0, color="0.7", lw=0.5)
ax.axvline(0, color="0.7", lw=0.5)
ax.set(xlabel="u (m/s)", ylabel="v (m/s)", aspect="equal", title="BMX hodograph, 0 to 6 km")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
fig.colorbar(sc, label="height above ground (km)")
plt.show()
print(f"storm-relative wind at 1 km: {float(rel.sel(height=low.height[0] + 1000, method='nearest').storm_relative_speed):.1f} m/s")
```

### Parcel thermodynamics: CAPE, CIN, the lifted index and the levels

`parcel_diagnostics` lifts a parcel through a sounding or an ERA5 column and
returns the CAPE and CIN, the lifted index and the pressure and height of the
lifting condensation level (LCL), the level of free convection (LFC) and the
equilibrium level (EL), for a surface-based, a mixed-layer (lowest 100 hPa) or
the most unstable parcel (lowest 300 hPa). The thermodynamics are those of
Bolton (1980) with the virtual-temperature correction (see
`radarx.io.parcel`). The ERA5 profile has its levels below the ground masked
and a surface level of 2 m air and 10 m wind (`surface=True` of `era5_profile`):

```{code-cell} ipython3
rows = {}
for name, prof in (("BMX sounding", valid), ("ERA5 at KGWX", era5)):
    for parcel in ("surface-based", "mixed-layer", "most-unstable"):
        d = parcel_diagnostics(prof, parcel)
        rows[(name, parcel)] = {
            "CAPE (J/kg)": float(d.cape),
            "CIN (J/kg)": float(d.cin),
            "lifted index (K)": float(d.lifted_index),
            "LCL (m)": float(d.lcl_height),
            "LFC (m)": float(d.lfc_height),
            "EL (m)": float(d.el_height),
        }
display(pd.DataFrame(rows).T.round(1))
via_accessor = valid.radarx.parcel_diagnostics("most-unstable")
print("the accessor agrees:",
      bool(np.isclose(float(via_accessor.cape), rows[("BMX sounding", "most-unstable")]["CAPE (J/kg)"])))
```

## 4. Dealias the Doppler velocity

The Doppler cuts have Nyquist velocities of 26–33 m/s, and the low-level jet
ahead of the line is stronger, so the velocities fold.
`dtree.radarx.dealias` unfolds all sweeps of a volume in one call of the
compiled kernel. Sweeps are processed from the lowest elevation upward, and
the dealiased sweep below fixes the absolute fold of the next one
(`sweep_continuity=True`, the default). The ERA5 wind profile from step 3
(`u` and `v` on `height`) fixes the fold of the lowest sweep and fills gaps.
Like every radarx retrieval it returns only its products; `radarx.assign`
adds them to the volumes.

```{code-cell} ipython3
with timed("4. dealias velocity"):
    volumes = [
        vol.radarx.assign(vol.radarx.dealias("VRADH", wind_profile=era5))
        for vol in volumes
    ]
```

```{code-cell} ipython3
doppler = volumes[1]["sweep_1"].to_dataset()
vn = float(doppler.nyquist_velocity)
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")
for ax, var, title in zip(
    axes, ["VRADH", "VRADH_dealiased"], [f"measured (Nyquist {vn:.1f} m/s)", "dealiased"]
):
    pm = ppi(ax, doppler, var, extent=150, cmap="balance", vmin=-50, vmax=50)
    ax.set_title(title)
axes[0].set_ylabel("north of KGWX (km)")
fig.colorbar(pm, ax=axes, label="radial velocity (m/s)")
fig.suptitle("KGWX 0.5° Doppler cut, 00:00 UTC")
plt.show()
```

`radarx.retrieve.dealias_velocity` is the function behind the accessor; it
takes one sweep or a whole volume. On the 0.5° Doppler cut it reproduces the
result above:

```{code-cell} ipython3
doppler_in = volumes[1]["sweep_1"].to_dataset().drop_vars("VRADH_dealiased")
unfolded = dealias_velocity(doppler_in, wind_profile=era5)
print("same unfolded velocity as the accessor:",
      bool(np.array_equal(unfolded.values, volumes[1]["sweep_1"].VRADH_dealiased.values, equal_nan=True)))
```

`dealias_confidence` says, for every gate, how clearly the chosen fold beats
the next one given the neighbouring gates: the share of the valid neighbours
(the 3 x 3 gates around it) that differ from the gate by no more than the
Nyquist velocity, so that they prefer its fold to the next. It is 1 in a
smooth field and 0 for an isolated gate unfolded differently from all its
neighbours, and along a straight fold line it is 5/8 on both sides. It is a
radarx definition, computed from the result alone, and it marks small-scale
disagreement, not a whole region that is unfolded wrongly. `dealias(...,
confidence=True)` returns it with the velocity.

```{code-cell} ipython3
confidence = volumes[1].radarx.dealias_confidence("VRADH_dealiased")
conf = confidence["sweep_1"].to_dataset().VRADH_dealiased_confidence
conf_fn = dealias_confidence(volumes[1]["sweep_1"].to_dataset(), "VRADH_dealiased")
print("accessor and function agree:", bool(np.array_equal(conf.values, conf_fn.values, equal_nan=True)),
      f"| gates with confidence below 0.5: {float((conf < 0.5).sum() / conf.notnull().sum()):.2%}")
fig, ax = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
pm = ppi(ax, doppler.assign(confidence=conf), "confidence", extent=150, cmap="viridis", vmin=0, vmax=1)
fig.colorbar(pm, ax=ax, label="confidence of the fold")
ax.set_title("KGWX 0.5° Doppler cut, 00:00 UTC")
plt.show()
```

`dealias_r2d2` is a second method (Feldmann et al. 2020), built for strong shear
between gates; with the ERA5 profile as first guess it works from the highest sweep
down. `dealias_ambiguity` adds the evidence for the fold of every region of a
dealiased sweep and, with `compare`, flags the gates where another dealiasing
chooses another fold ([Dealiasing methods](Dealiasing_Methods)).

```{code-cell} ipython3
r2d2_volume = volumes[1].radarx.dealias_r2d2("VRADH", wind_profile=era5, name="VRADH_r2d2")
r2d2_sweep = dealias_r2d2(doppler_in, "VRADH", wind_profile=era5)
both = volumes[1].radarx.assign(r2d2_volume)
ambiguity = both.radarx.dealias_ambiguity("VRADH_dealiased", compare="VRADH_r2d2")
fold = dealias_ambiguity(
    volumes[1]["sweep_1"].to_dataset(), "VRADH_dealiased"
).VRADH_dealiased_fold_flag
print(f"R2D2 and region folds differ on "
      f"{float((r2d2_sweep != unfolded).sum() / unfolded.notnull().sum()):.2%} of the gates of the 0.5° cut; "
      f"gates with a weak fold margin: {float(((fold & 1) > 0).sum() / (fold > 0).size):.2%}")
```

The dealiased velocities of the higher sweeps also give the horizontal wind
above the radar. `vad_profile` fits a sinusoid to the velocities of every
range ring (the velocity-azimuth display of Browning and Wexler 1968) and
averages the winds of the rings in height bins. It is the radar-only
counterpart of the sounding of step 3:

```{code-cell} ipython3
with timed("4b. VAD wind profile"):
    vad = vad_profile(volumes[1], field="VRADH_dealiased", max_rms=4.0, max_range=80e3)
vad_acc = volumes[1].radarx.vad_profile(field="VRADH_dealiased", max_rms=4.0, max_range=80e3)
fig, axes = plt.subplots(1, 2, figsize=(8.5, 5), sharey=True, layout="constrained")
for ax, comp in zip(axes, ("u", "v")):
    ax.plot(vad[comp], vad.height / 1e3, ".-", color="k", ms=3, label="KGWX VAD, 00:00 UTC")
    ax.plot(era5[comp], era5.height / 1e3, color="C0", label="ERA5")
    ax.plot(raob[comp], raob.height / 1e3, color="C1", label="BMX radiosonde")
    ax.set(xlabel=f"{comp} (m/s)", xlim=(-30, 40) if comp == "u" else (-20, 40))
axes[0].set(ylim=(0, 8), ylabel="height above sea level (km)")
axes[0].legend(loc="upper left", frameon=False)
plt.show()
print("accessor and function agree:", bool(np.allclose(vad.u, vad_acc.u, equal_nan=True)),
      f"| VAD 0-3 km bulk shear {float(bulk_shear(vad, 0, 3000).shear_speed):.1f} m/s")
```

## 5. ΦDP processing and KDP

`dtree.radarx.kdp` removes the system offset, filters the phase in range and
estimates KDP by least squares, for all rays of all sweeps in one call. The
processed phase is returned as `PHIDP_processed`, so the raw `PHIDP` is kept.

```{code-cell} ipython3
with timed("5. PhiDP processing and KDP"):
    volumes = [
        vol.radarx.assign(vol.radarx.kdp(phidp="PHIDP", rhohv="RHOHV", dbz="DBZH"))
        for vol in volumes
    ]
```

```{code-cell} ipython3
surv = volumes[1]["sweep_0"].to_dataset()
panels = [
    ("DBZH", "$Z_H$ (dBZ)", dict(cmap="ChaseSpectral", vmin=-10, vmax=70)),
    ("ZDR", "$Z_{DR}$ (dB)", dict(cmap="HomeyerRainbow", vmin=-1, vmax=5)),
    ("PHIDP_processed", "processed $\\Phi_{DP}$ (°)", dict(cmap="viridis", vmin=0, vmax=150)),
    ("KDP", "$K_{DP}$ (°/km)", dict(cmap="HomeyerRainbow", vmin=-0.5, vmax=4)),
]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), sharey=True, layout="constrained")
for ax, (var, label, style) in zip(axes, panels):
    data = surv if var != "PHIDP_processed" else surv.assign(
        PHIDP_processed=surv.PHIDP_processed.where(surv.KDP.notnull())
    )
    pm = ppi(ax, data, var, extent=150, **style)
    fig.colorbar(pm, ax=ax, label=label, shrink=0.85)
axes[0].set_ylabel("north of KGWX (km)")
fig.suptitle("KGWX 0.5° surveillance cut, 00:00 UTC")
plt.show()
```

`estimate_kdp` is the function behind the accessor and works on one sweep. It
offers several methods (`method="hubbert"` is the default; see the
[KDP notebook](KDP)):

```{code-cell} ipython3
sweep_in = volumes[1]["sweep_0"].to_dataset()
kdp_fn = estimate_kdp(sweep_in, phidp="PHIDP", rhohv="RHOHV", dbz="DBZH")
print("same KDP as the accessor:", bool(np.allclose(kdp_fn.KDP, sweep_in.KDP, equal_nan=True)))
```

### Attenuation correction

Rain attenuates the signal, at S band a little: the two-way attenuation is
$\alpha\,\Delta\Phi_{DP}$ with $\alpha=0.016$ dB per degree
(Bringi et al. 1990), so 2.4 dB for a phase of 150° and up to 6 dB for the largest spans of this
case. `correct_attenuation`
adds the path attenuation to $Z_H$ and the differential attenuation to $Z_{DR}$
along every ray, from the processed phase of the step above, with the constrained
ZPHI solution of Testud et al. (2000) or the linear relation (`method="linear"`).
The coefficients depend on the band (`attenuation_coefficients`) and can be set.
The [attenuation notebook](Attenuation_Correction) shows the method on forward-modelled rain at S, C and X
band and on a C-band volume.

```{code-cell} ipython3
with timed("5b. Attenuation correction"):
    att = correct_attenuation(surv, band="S")
    att_vol = volumes[1].radarx.correct_attenuation(band="S")  # the sweeps of the volume
att_acc = surv.radarx.correct_attenuation(band="S")  # and the accessor of a sweep
print("accessor agrees:", bool(att_acc.PIA.equals(att.PIA)),
      "| sweeps corrected:", list(att_vol.children),
      "| S band:", attenuation_coefficients("S"))
print(f"largest path attenuation {float(att.PIA.max()):.1f} dB, largest differential attenuation "
      f"{float(att.PIDA.max()):.2f} dB")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True, layout="constrained")
panels = [
    (att.PIA, "path attenuation PIA (dB)", dict(cmap="magma_r", vmin=0, vmax=6)),
    (att.PIDA, "differential attenuation PIDA (dB)", dict(cmap="magma_r", vmin=0, vmax=1.2)),
    (att.DBZH_corrected - surv.DBZH, "$Z_H$ corrected minus measured (dB)", dict(cmap="magma_r", vmin=0, vmax=6)),
]
for ax, (da, label, style) in zip(axes, panels):
    pm = ppi(ax, surv.assign(field=da), "field", extent=150, **style)
    fig.colorbar(pm, ax=ax, label=label, shrink=0.85)
axes[0].set_ylabel("north of KGWX (km)")
fig.suptitle("Rain attenuation of the KGWX 0.5° cut, 00:00 UTC")
plt.show()
```

## 6. Hydrometeor classification

`dtree.radarx.hid` assigns a hydrometeor class to every gate with the fuzzy
logic of Park et al. (2009) for S band, from $Z_H$, $Z_{DR}$, the KDP of
step 5 and $\rho_{hv}$. The ERA5 temperature profile of step 3 is
interpolated to the gate heights and places the melting layer, below which
only liquid and mixed-phase classes are allowed. The non-meteorological
gates removed in step 2 stay unclassified (class 0).

```{code-cell} ipython3
with timed("6. hydrometeor classification"):
    volumes = [
        vol.radarx.assign(vol.radarx.hid(era5, band="S", scores=False))
        for vol in volumes
    ]
```

```{code-cell} ipython3
park = hid_classes("park", "S")
hid_cmap = ListedColormap(
    ["#2a78d6", "#e87ba4", "#1baf7a", "#eda100", "#4a3aa7", "#008300", "#eb6834", "#e34948"]
)
hid_norm = BoundaryNorm(np.arange(0.5, len(park) + 1.5), hid_cmap.N)
surv = volumes[1]["sweep_0"].to_dataset()
upper = volumes[1]["sweep_9"].to_dataset()  # 2.4°
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True, layout="constrained")
for ax, ds in zip(axes, [surv, upper]):
    pm = ppi(ax, ds.assign(HID=ds.HID.where(ds.HID > 0)), "HID", extent=150, cmap=hid_cmap, norm=hid_norm)
    ax.set_title(f"{float(ds.sweep_fixed_angle):.1f}°")
cb = fig.colorbar(pm, ax=axes, ticks=np.arange(1, len(park) + 1), shrink=0.9)
cb.ax.set_yticklabels([name.replace("_", " ") for _, _, name in park])
axes[0].set_ylabel("north of KGWX (km)")
fig.suptitle("Hydrometeor classes, 00:00 UTC")
plt.show()
```

`radarx.retrieve.hid` is the function behind the accessor; a single sweep
with the ERA5 profile gives the same classes:

```{code-cell} ipython3
classes_fn = hid(volumes[1]["sweep_0"].to_dataset(), era5, band="S", scores=False)
print("same classes as the accessor:",
      bool(np.array_equal(classes_fn.HID.values, volumes[1]["sweep_0"].HID.values)))
```

Rain fills the lowest sweep within about 120 km, with heavy rain along the
leading edge of the line. Farther out, and beyond 50 km on the 2.4° sweep,
the beam reaches the melting layer (wet snow, graupel) and the snow and ice
above it.

+++

## 7. Rain drop size distribution

`ds.radarx.dsd` retrieves the gamma drop size distribution of rain at every
gate from $Z_H$ and $Z_{DR}$, with the constrained-gamma method (Zhang et al.
2001; Cao et al. 2008), and takes the intercept from KDP where it is at
least 1 °/km. The hydrometeor classes of step 6 select the rain gates
(light and moderate rain, heavy rain, big drops); hail, the melting layer
and ice would give meaningless drop sizes. We retrieve the DSD on the
lowest sweep, whose rain is the one that reaches the ground.

```{code-cell} ipython3
rain_codes = [code for code, _, name in park if name in ("big_drops", "light_and_moderate_rain", "heavy_rain")]
with timed("7. DSD retrieval (0.5°)"):
    surv = volumes[1]["sweep_0"].to_dataset()
    rain = surv.HID.isin(rain_codes)
    dsd = surv.radarx.dsd(kdp="KDP", mask=rain, band="S")
dsd
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True, layout="constrained")
for ax, (da, label, style) in zip(
    axes,
    [
        (dsd.D0, "median volume diameter $D_0$ (mm)", dict(cmap="plasma", vmin=0.5, vmax=3)),
        (np.log10(dsd.NW), "$\\log_{10} N_w$ (m$^{-3}$ mm$^{-1}$)", dict(cmap="viridis", vmin=2, vmax=5)),
        (dsd.RAIN_RATE, "rain rate (mm/h)", dict(cmap="turbo", vmin=0, vmax=60)),
    ],
):
    pm = ppi(ax, surv.assign(field=da), "field", extent=150, **style)
    fig.colorbar(pm, ax=ax, label=label, shrink=0.85)
axes[0].set_ylabel("north of KGWX (km)")
fig.suptitle("DSD retrieval on the rain gates, 0.5°, 00:00 UTC")
plt.show()
```

The convective line has large drops ($D_0$ above 2 mm) and the highest rain
rates; the trailing stratiform rain has smaller drops.

+++

`radarx.retrieve.dsd_bayesian` returns the posterior distribution of the
normalized gamma parameters instead, with the uncertainty of every gate. It
takes the polarimetric variables with their measurement errors, the forward
model of T-matrix scattering tables (`forward_grid`) and a prior learned from
disdrometers (`dsd_prior`; here the generic prior, with the option of one learned from
PERiLS 2022 Parsivel disdrometers, see the [Bayesian DSD notebook](Bayesian_DSD)). KDP fixes $N_w$ independently of the
$Z_H$ calibration, so the uncertainty of the rain rate is smallest in the
convective line:

```{code-cell} ipython3
rain_near = rain & (surv.range <= 120e3)
prior = dsd_prior("generic")
forward = forward_grid("S")
with timed("7b. Bayesian DSD retrieval (0.5°)"):
    post = dsd_bayesian(surv, kdp="KDP", mask=rain_near, band="S", prior="generic")
det = radarx.retrieve.dsd(surv, "normalized", kdp="KDP", mask=rain_near, band="S")
subset = slice(0, 60)  # the accessor on a few rays
post_acc = surv.isel(azimuth=subset).radarx.dsd_bayesian(
    kdp="KDP", mask=rain_near.isel(azimuth=subset), band="S", prior="generic"
)
print("accessor and function agree:", bool(np.allclose(post_acc.DM, post.DM.isel(azimuth=subset), equal_nan=True)))
q = post.RAIN_RATE_QUANTILES
interval = (q.sel(quantile=0.975) - q.sel(quantile=0.025)) / post.RAIN_RATE
print(f"prior {prior.attrs['prior']!r} on {prior.sizes['dm']} x {prior.sizes['mu']} nodes; "
      f"median 95 % interval of the rain rate: {float(interval.median()):.2f} of its value")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), layout="constrained")
panels = [
    (post.DM, "posterior mean $D_m$ (mm)", dict(cmap="plasma", vmin=0.5, vmax=3.0)),
    (post.DM_SD, "posterior sd of $D_m$ (mm)", dict(cmap="viridis", vmin=0, vmax=0.5)),
    (interval, "95 % interval of the rain rate / rain rate", dict(cmap="magma", vmin=0, vmax=2)),
]
for ax, (da, label, style) in zip(axes[:3], panels):
    pm = ppi(ax, surv.assign(field=da), "field", extent=120, **style)
    fig.colorbar(pm, ax=ax, label=label, shrink=0.85)
axes[0].set_ylabel("north of KGWX (km)")
ok = rain_near.values & np.isfinite(det.DM.values) & np.isfinite(post.DM.values)
h = axes[3].hist2d(det.DM.values[ok], post.DM.values[ok], bins=np.linspace(0.4, 3.5, 60), cmin=1, cmap="Blues")
axes[3].plot([0.4, 3.5], [0.4, 3.5], "k-", lw=0.8)
axes[3].set(xlabel="normalized gamma, $\\mu$ = 3: $D_m$ (mm)", ylabel="posterior mean $D_m$ (mm)", aspect="equal")
fig.colorbar(h[3], ax=axes[3], label="gates", shrink=0.85)
fig.suptitle("Bayesian DSD retrieval, KGWX 0.5°, 00:00 UTC")
plt.show()
```

### Rain rate

`rain_rate` turns the classes of step 6 into a rain rate. Each gate gets the
relation of its class: $R(Z, Z_{DR})$ in rain, $R(K_{DP})$ in a rain-hail
mixture (Giangrande and Ryzhkov 2008), and NaN, with a flag that says why,
where no relation is valid (graupel, snow, ice, the melting layer). The S-band
coefficients of the papers are the defaults. At S band $Z_H$ and $Z_{DR}$ are
only weakly attenuated, so we use them as they are; at C and X band apply an
attenuation correction first (see the
[rainfall estimation notebook](Rainfall_Estimation)). The single relations are
`rain_rate_z`, `rain_rate_kdp`, `rain_rate_z_zdr` and `rain_rate_a`; the last
needs the specific attenuation, which we approximate here with
$A = 0.015\,K_{DP}$ dB km$^{-1}$ (the S-band value of Ryzhkov et al. 2014).

```{code-cell} ipython3
with timed("7c. rain rate"):
    surv = volumes[1]["sweep_0"].to_dataset()
    rain_blend = rain_rate(surv)  # classes from surv.HID
    r_z = rain_rate_z(surv)
    r_kdp = rain_rate_kdp(surv)
    r_zdr = rain_rate_z_zdr(surv)
    r_a = rain_rate_a(surv.assign(AH=0.015 * surv.KDP))
flag_counts = {
    m: int((rain_blend.RAIN_RATE_FLAG == i).sum())
    for i, m in enumerate(rain_blend.RAIN_RATE_FLAG.attrs["flag_meanings"].split())
}
print("gates by flag:", flag_counts)
```

```{code-cell} ipython3
panels = [
    (rain_blend.RAIN_RATE, "rain rate by class (mm/h)", dict(cmap="turbo", vmin=0, vmax=60)),
    (r_z.where(rain_blend.RAIN_RATE.notnull()), "R(Z), same gates (mm/h)", dict(cmap="turbo", vmin=0, vmax=60)),
    (r_kdp.where(rain_blend.RAIN_RATE.notnull()), "R(KDP), same gates (mm/h)", dict(cmap="turbo", vmin=0, vmax=60)),
]
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True, layout="constrained")
for ax, (da, label, style) in zip(axes, panels):
    pm = ppi(ax, surv.assign(field=da), "field", extent=120, **style)
    fig.colorbar(pm, ax=ax, label=label, shrink=0.85)
axes[0].set_ylabel("north of KGWX (km)")
fig.suptitle("Rain rate, KGWX 0.5°, 00:00 UTC")
plt.show()
```

The rain rate of the classes can be compared with the one of the DSD retrieval
of step 7, which uses $Z_H$, $Z_{DR}$ and $K_{DP}$ through a gamma
distribution. The blend uses $Z_{DR}$ on the rain gates and the DSD retrieval
takes the intercept from $K_{DP}$ where it is at least 1 °/km:

```{code-cell} ipython3
both = rain_blend.RAIN_RATE.where(rain_near).values, dsd.RAIN_RATE.where(rain_near).values
ok = np.isfinite(both[0]) & np.isfinite(both[1]) & (both[1] > 1)
ratio = both[0][ok] / both[1][ok]
print(f"{ok.sum()} rain gates; median ratio blend / DSD retrieval {np.median(ratio):.2f}, "
      f"5-95 %: {np.percentile(ratio, 5):.2f}-{np.percentile(ratio, 95):.2f}")
```

The same functions are methods of the volume, `vol.radarx.rain_rate()`, and of a
sweep or a field, for example `surv.radarx.rain_rate_z()`:

```{code-cell} ipython3
same = volumes[1].radarx.rain_rate()["sweep_0"].RAIN_RATE
print("same rain rate as the function:", bool(np.allclose(same, rain_blend.RAIN_RATE, equal_nan=True)))
print("per-field accessors:",
      surv.radarx.rain_rate_z().name, surv.radarx.rain_rate_kdp().name,
      surv.radarx.rain_rate_z_zdr().name, surv.radarx.rain_rate_a(a="KDP", coefficients=(4120.0, 1.03)).name)
```

## 8. QVP time series and melting layer

The quasi-vertical profiles average the highest sweep (19.5°) of every volume
over azimuth and place each gate at its beam height; `melting_layer` finds the
co-located ρhv minimum and ZDR/Z maxima of the bright band. KDP from step 5 is
profiled along with the measured fields. The ERA5 0 °C height from step 3
narrows the search (`freezing_level=`); the melting layer should lie just
below it, since snow melts over a few hundred metres below the 0 °C level.

```{code-cell} ipython3
with timed("8. QVP and melting layer"):
    tqvp = qvp_timeseries(volumes, ["DBZH", "ZDR", "RHOHV", "KDP"], elevation=19.5)
    ml = melting_layer(tqvp, freezing_level=freezing_level)
print(f"ERA5 0 °C height: {freezing_level:.0f} m")
ml.to_dataframe()[["melting_layer_bottom", "melting_layer_peak", "melting_layer_top", "melting_layer_flag"]]
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True, layout="constrained")
colors = ["C0", "C3"]
for ax, (var, label) in zip(
    axes,
    [("DBZH", "$Z_H$ (dBZ)"), ("ZDR", "$Z_{DR}$ (dB)"), ("RHOHV", "$\\rho_{hv}$"), ("KDP", "$K_{DP}$ (°/km)")],
):
    for i, color in enumerate(colors):
        ax.plot(tqvp[var].isel(time=i), tqvp.height / 1e3, color=color,
                label=str(tqvp.time.values[i])[11:16] + " UTC")
    ax.axhspan(float(ml.melting_layer_bottom.mean()) / 1e3, float(ml.melting_layer_top.mean()) / 1e3,
               color="0.5", alpha=0.25, label="melting layer")
    ax.axhline(freezing_level / 1e3, color="k", ls=":", label="ERA5 0 °C")
    ax.set_xlabel(label)
for ax, lim in zip(axes, [(-10, 50), (-0.5, 4), (0.8, 1.0), (-0.2, 0.5)]):
    ax.set_xlim(lim)
axes[0].set_ylim(0, 10)
axes[0].set_ylabel("height above sea level (km)")
axes[0].legend(loc="upper right", fontsize=8)
fig.suptitle("QVPs of the 19.5° sweep, KGWX")
plt.show()
```

The bright band (ZDR and Z maxima, ρhv minimum) is found at about 3 km,
roughly 0.5 km below the ERA5 0 °C level. The QVP circle (within about 10 km
of the radar) lies in the rain just ahead of the line, where melting and
evaporation of the precipitation cool the air and can lower the 0 °C level,
while ERA5 and the sounding describe the larger-scale environment.

+++

A single QVP of a volume comes from `radarx.retrieve.qvp`, the function
that `qvp_timeseries` repeats for every volume:

```{code-cell} ipython3
single = qvp(volumes[1], ["DBZH", "ZDR", "RHOHV"], elevation=19.5)
single_acc = volumes[1].radarx.qvp(["DBZH", "ZDR", "RHOHV"], elevation=19.5)  # the accessor of the volume
ml_acc = tqvp.radarx.melting_layer(freezing_level=freezing_level)  # and of the profiles
print("same profile as the time series:",
      bool(np.allclose(single.DBZH.squeeze(), tqvp.DBZH.isel(time=1), equal_nan=True)),
      bool(np.allclose(single_acc.DBZH, single.DBZH, equal_nan=True)),
      "| melting layer from the accessor:", bool(ml_acc.melting_layer_peak.equals(ml.melting_layer_peak)))
```

+++

### Range-defined and columnar profiles, and the melting layer on PPIs

The QVP needs a high sweep. `rd_qvp` (Tobin and Kumjian 2017) combines the
azimuthal means of all sweeps within 50 km of the radar, so that the profile
reaches the ground, and `cvp` (Murphy et al. 2020) profiles a sector of the
volume (here 20 km by 20 degrees) around a chosen point from the sweeps that
see it. Both put the profile on a regular height grid. `melting_layer_ppi`
finds the melting layer on the PPIs between 4 and 10 degrees, with the
radial-profile rules of Giangrande et al. (2008), as the 80th and 20th
percentile of the heights of its points, for every azimuth.

```{code-cell} ipython3
fields = ["DBZH", "ZDR", "RHOHV"]
with timed("8b. range-defined, columnar and PPI profiles"):
    rd = rd_qvp(volumes[1], fields, rmax=50e3)
    # the azimuth with the strongest echo at 40 km on the lowest sweep
    sweep0 = volumes[1]["sweep_0"].to_dataset()
    az0 = float(sweep0.DBZH.sel(range=slice(35e3, 45e3)).mean("range").idxmax("azimuth"))
    column = cvp(volumes[1], fields, azimuth=az0, ground_range=40e3)
    ml_ppi = melting_layer_ppi(volumes)
rd_acc = volumes[1].radarx.rd_qvp(fields, rmax=50e3)
column_acc = volumes[1].radarx.cvp(fields, azimuth=az0, ground_range=40e3)
ml_ppi_acc = volumes[1].radarx.melting_layer_ppi()
print("accessors agree:", bool(rd_acc.DBZH.equals(rd.DBZH)), bool(column_acc.DBZH.equals(column.DBZH)),
      bool(ml_ppi_acc.melting_layer_top_all.equals(ml_ppi.melting_layer_top_all.isel(time=1))))
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True, layout="constrained")
for ax, name in zip(axes, fields):
    ax.plot(tqvp[name].isel(time=1), tqvp.height / 1e3, label="QVP, 19.5°")
    ax.plot(rd[name], rd.height / 1e3, label="RD-QVP, 50 km")
    ax.plot(column[name], column.height / 1e3, label=f"CVP, {az0:.0f}°, 40 km")
    ax.set_xlabel(name)
axes[0].set_ylim(0, 12)
axes[0].set_ylabel("height above sea level (km)")
axes[0].legend(fontsize=8)
plt.show()
top, bottom = ml_ppi.melting_layer_top_all.isel(time=1), ml_ppi.melting_layer_bottom_all.isel(time=1)
print(f"melting layer in the PPIs, all azimuths: {float(bottom):.0f} to {float(top):.0f} m "
      f"({int(ml_ppi.melting_layer_points_all.isel(time=1))} points); "
      f"top between {float(ml_ppi.melting_layer_top.isel(time=1).min()):.0f} and "
      f"{float(ml_ppi.melting_layer_top.isel(time=1).max()):.0f} m depending on the azimuth")
```

The rain rates of the lowest sweep hold in rain only. `mask_above_melting_layer`
sets them (and the other rain-only products) to NaN above the bottom of the
melting layer, here the azimuth-dependent one of the PPIs, and flags the
removed gates in `RAIN_RATE_FLAG`:

```{code-cell} ipython3
rain_sweep = rain_blend.assign_coords(altitude=float(volumes[1].altitude))
capped = mask_above_melting_layer(rain_sweep, ml_ppi.isel(time=1))
capped_acc = rain_sweep.radarx.mask_above_melting_layer(ml_ppi.isel(time=1), margin=0.0)
removed = int((capped.RAIN_RATE_FLAG == 4).sum())
print(f"{removed} of {int(rain_blend.RAIN_RATE.notnull().sum())} rain rates removed above the melting layer;",
      "accessor agrees:", bool(capped_acc.RAIN_RATE.equals(capped.RAIN_RATE)))
```

A scan in range-height (RHI) form has a profile of its own. `rhi_qvp` converts the
range of every gate to its height and averages the gates of all elevations in 100 m
bins (Allabakash et al. 2019). Here an RHI is simulated by sampling the QVP of
this volume along an RHI geometry, so that the profile it returns can be compared
with the one it came from; `melting_layer` works on the result as on a QVP:

```{code-cell} ipython3
from xradar.georeference import antenna_to_cartesian

rhi_range = np.arange(250.0, 40e3, 250.0)
rhi_elevation = np.arange(1.0, 60.0, 0.5)
_, _, rhi_height = antenna_to_cartesian(
    rhi_range[None, :], 0.0, rhi_elevation[:, None], earth_radius=6371000.0, site_altitude=float(volumes[1].altitude)
)
prof = tqvp.isel(time=1)
rhi = xr.Dataset(
    {
        name: (("azimuth", "range"), np.interp(rhi_height, prof.height, prof[name]).astype("float32"), {"units": unit})
        for name, unit in (("DBZH", "dBZ"), ("ZDR", "dB"), ("RHOHV", "1"))
    },
    coords={
        "azimuth": np.full(rhi_elevation.size, 109.0),
        "range": rhi_range,
        "elevation": ("azimuth", rhi_elevation),
        "altitude": float(volumes[1].altitude),
    },
)
rhi_profile = rhi_qvp(rhi, fields, rmin=2e3, rmax=30e3)
rhi_acc = rhi.radarx.rhi_qvp(fields, rmin=2e3, rmax=30e3)
ok = rhi_profile.DBZH.notnull()
diff = rhi_profile.DBZH - np.interp(rhi_profile.height, prof.height, prof.DBZH)
print(f"RHI profile: {int(ok.sum())} bins of 100 m, RMS difference from the profile it was "
      f"sampled from {float(np.sqrt((diff.where(ok) ** 2).mean())):.2f} dB; accessor agrees:",
      bool(rhi_acc.DBZH.equals(rhi_profile.DBZH)))
melting_layer(rhi_profile)[["melting_layer_bottom", "melting_layer_top"]]
```

## 9. Azimuthal shear and radial divergence

The linear least-squares derivative (LLSD) of the **dealiased** velocity from
step 4 gives the azimuthal shear (rotation) and the radial divergence
(convergence along the gust front) at every gate, for all sweeps in one call.

```{code-cell} ipython3
with timed("9. azimuthal shear and divergence"):
    volumes = [vol.radarx.assign(vol.radarx.llsd("VRADH_dealiased")) for vol in volumes]
```

```{code-cell} ipython3
doppler = volumes[1]["sweep_1"].to_dataset()
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True, layout="constrained")
for ax, (var, label, style) in zip(
    axes,
    [
        ("VRADH_dealiased", "radial velocity (m/s)", dict(cmap="balance", vmin=-50, vmax=50)),
        ("azimuthal_shear", "azimuthal shear (s⁻¹)", dict(cmap="RdBu_r", vmin=-0.01, vmax=0.01)),
        ("radial_divergence", "radial divergence (s⁻¹)", dict(cmap="PuOr", vmin=-0.01, vmax=0.01)),
    ],
):
    pm = ppi(ax, doppler, var, extent=120, **style)
    fig.colorbar(pm, ax=ax, label=label, shrink=0.85)
axes[0].set_ylabel("north of KGWX (km)")
fig.suptitle("LLSD of the dealiased 0.5° velocity, 00:00 UTC")
plt.show()
```

The three products are also available one at a time, from `llsd` (both) or
`azimuthal_shear` and `radial_divergence` (each on its own):

```{code-cell} ipython3
both = llsd(doppler, "VRADH_dealiased")
shear_fn = azimuthal_shear(doppler, "VRADH_dealiased")
div_fn = radial_divergence(doppler, "VRADH_dealiased")
shear_acc = doppler.radarx.azimuthal_shear("VRADH_dealiased")  # the accessors of a sweep
div_acc = doppler.radarx.radial_divergence("VRADH_dealiased")
print("the accessors agree:", bool(np.allclose(shear_acc, shear_fn, equal_nan=True)),
      bool(np.allclose(div_acc, div_fn, equal_nan=True)))
print("the three calls agree:",
      bool(np.allclose(both.azimuthal_shear, shear_fn, equal_nan=True)),
      bool(np.allclose(both.radial_divergence, div_fn, equal_nan=True)))
```

The warning and research products of the NSSL (Smith et al. 2016) are not
the shear of one tilt but its **maximum over the tilts of a height layer**: 0 to
2 km above the ground for near-surface rotation and 3 to 6 km for mid-level
rotation, after the shear of gates that are not in or near a storm cell
(reflectivity of 20 dBZ within 5 km) is left out.
`azimuthal_shear_layer_max` does this for a volume, on the grid of the lowest
tilt: at each position it takes the value of every tilt whose beam centre is in
the layer at that ground range and azimuth (4/3 Earth model), and the largest.
`median_filter=3` adds the 3 x 3 median of the velocity that Smith and Elmore
(2004) apply before the fit. `layer_max_elevation` says which tilt gave each
value.

```{code-cell} ipython3
with timed("9b. layer-maximum azimuthal shear"):
    low_layer = volumes[1].radarx.azimuthal_shear_layer_max(
        "VRADH_dealiased", layer=(0.0, 2000.0), min_dbz=20.0
    )
    mid_layer = azimuthal_shear_layer_max(
        volumes[1], "VRADH_dealiased", layer=(3000.0, 6000.0), min_dbz=20.0, median_filter=3
    )
lowest = volumes[1]["sweep_1"].to_dataset()
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True, layout="constrained")
for ax, (da, label) in zip(
    axes,
    [
        (lowest.azimuthal_shear, "0.5° tilt"),
        (low_layer.azimuthal_shear_layer_max, "maximum over 0-2 km"),
        (mid_layer.azimuthal_shear_layer_max, "maximum over 3-6 km"),
    ],
):
    pm = ppi(ax, lowest.assign(field=da), "field", extent=120, cmap="RdBu_r", vmin=-0.01, vmax=0.01)
    ax.set_title(label)
fig.colorbar(pm, ax=axes, label="azimuthal shear (s⁻¹)", shrink=0.85)
axes[0].set_ylabel("north of KGWX (km)")
plt.show()
used = float(low_layer.layer_max_elevation.notnull().mean())
print(f"0-2 km layer maximum has data at {used:.0%} of the 0.5° gates; tilts used:",
      np.unique(low_layer.layer_max_elevation.values[np.isfinite(low_layer.layer_max_elevation.values)]).round(1).tolist())
```

## 10. Cone gridding, Max-CAPPI, interactive view and UGRID

`dtree.radarx.to_grid` grids reflectivity, ZDR, KDP, dealiased velocity and
azimuthal shear of each volume onto a 300 km × 300 km × 12 km Cartesian grid
with the compiled cone method. Of the surveillance and Doppler cuts at each
elevation it takes, for every field, the cut that holds it (and reaches
farthest), so the split cuts need no merging first.

```{code-cell} ipython3
fields = ["DBZH", "ZDR", "KDP", "VRADH_dealiased", "azimuthal_shear"]
grid_kw = dict(
    x_lim=(-150e3, 150e3),
    y_lim=(-150e3, 150e3),
    z_lim=(500, 12e3),
    x_step=1000,
    y_step=1000,
    z_step=500,
)
with timed("10. grid"):
    grids = [vol.radarx.to_grid(fields, **grid_kw) for vol in volumes]
grids[1]
```

The Max-CAPPI shows the column maximum with its north–south and east–west
projections:

```{code-cell} ipython3
grids[1].radarx.plot_max_cappi("DBZH", cmap="ChaseSpectral", vmin=-10, vmax=70, range_rings=True)
```

CAPPIs of the other gridded fields at 2 km:

```{code-cell} ipython3
level = grids[1].sel(z=2000)
fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), sharey=True, layout="constrained")
for ax, (var, label, style) in zip(
    axes,
    [
        ("ZDR", "$Z_{DR}$ (dB)", dict(cmap="HomeyerRainbow", vmin=-1, vmax=5)),
        ("KDP", "$K_{DP}$ (°/km)", dict(cmap="HomeyerRainbow", vmin=-0.5, vmax=4)),
        ("VRADH_dealiased", "radial velocity (m/s)", dict(cmap="balance", vmin=-50, vmax=50)),
        ("azimuthal_shear", "azimuthal shear (s⁻¹)", dict(cmap="RdBu_r", vmin=-0.01, vmax=0.01)),
    ],
):
    pm = ax.pcolormesh(level.x / 1e3, level.y / 1e3, level[var], **style)
    fig.colorbar(pm, ax=ax, label=label, shrink=0.85)
    ax.set_aspect("equal")
    ax.set_xlabel("east of KGWX (km)")
axes[0].set_ylabel("north of KGWX (km)")
fig.suptitle("2 km CAPPIs, 00:00 UTC volume")
plt.show()
```

The same Max-CAPPI as an interactive hvplot view (zoom and pan are linked
between the plan view and the projections):

```{code-cell} ipython3
import holoviews as hv

hv.extension("bokeh")
grids[1].radarx.plot.max_cappi("DBZH", cmap="ChaseSpectral", clim=(-10, 70))
```

A sweep can also be exported as an unstructured (UGRID) mesh with one face
per gate for uxarray:

```{code-cell} ipython3
with timed("10b. UGRID export (one sweep)"):
    uxds = volumes[1]["sweep_1"].to_dataset().radarx.to_uxarray(["VRADH_dealiased", "azimuthal_shear"])
uxds
```

### Other ways to grid, section and plot

`to_grid` is built from functions that can be used directly. `grid_cones` is
the cone interpolation of a volume on any `x`, `y`, `z` axes, `grid_radar` the
older interface with a pseudo-CAPPI fill, and `make_3d_grid` returns the
axes, the geographic position of every grid column and the projection of a
Cartesian grid centred on a radar. `create_cappi` (also
`dtree.radarx.create_cappi`) extracts a constant-altitude plan view,
`stack_data` stacks the gates of a volume into one table of points and
`gate_corners` gives the geographic corners of every gate of a sweep, for
polygon plots and conservative regridding:

```{code-cell} ipython3
xs = np.arange(-100e3, 100e3 + 1, 2000.0)
zs = np.arange(500.0, 8e3 + 1, 500.0)
low_volume = xr.DataTree.from_dict(
    {"/": volumes[1].to_dataset(), "/sweep_0": volumes[1]["sweep_0"].to_dataset(inherit="all_coords")}
)
with timed("10c. functions of the gridding"):
    cones = grid_cones(volumes[1], ["DBZH"], x=xs, y=xs, z=zs)
    pseudo = grid_radar(
        volumes[1], ["DBZH"], x_lim=(-100e3, 100e3), y_lim=(-100e3, 100e3), z_lim=(500, 8e3),
        x_step=2000, y_step=2000, z_step=500,
    )
    cappi = create_cappi(volumes[1], 2000, data_vars=["DBZH"], x_res=2000, y_res=2000)
    cappi_acc = volumes[1].radarx.create_cappi(2000, data_vars=["DBZH"], x_res=2000, y_res=2000)
    cappi_alias = volumes[1].radarx.to_cappi(2000, data_vars=["DBZH"], x_res=2000, y_res=2000)  # an alias
    points = stack_data(low_volume, ["DBZH"])
lat_axis, lon_axis, x_axis, y_axis, z_axis, crs = make_3d_grid(
    volumes[1]["sweep_0"].to_dataset(), x_lim=(-100e3, 100e3), y_lim=(-100e3, 100e3), x_step=2000,
    y_step=2000, z_lim=(500, 8e3), z_step=500,
)
corners = gate_corners(volumes[1]["sweep_0"].to_dataset().isel(azimuth=slice(0, 3), range=slice(0, 3)))
print(f"cone grid {dict(cones.sizes)}; pseudo-CAPPI grid {dict(pseudo.sizes)}; CAPPI {dict(cappi.sizes)}; "
      f"{points.sizes['npoints']} gates stacked; make_3d_grid axes {z_axis.size} x {y_axis.size} x {x_axis.size}; "
      f"corner arrays {corners[0].shape}")
```

### Gridding with a radius of influence

`grid_cones` has no smoothing length. `grid_gates` is the other way to grid: each
output point is the weighted mean of the gates within a radius of influence in
metres, with the weights of Cressman (1959) or Barnes (1964), or the nearest
gate. The radius can grow with the distance from the radar (`roi_range_factor`),
or follow the beam width (`beamwidth`, in degrees), as the radius functions of
Py-ART do. A gate is a point in space, so the scan does not matter: the same
call grids sector scans, RHIs and volumes that repeat a tilt. `grid_rhi` gives a
vertical section through the radar in the same way. For repeated sweeps of one
elevation, such as the split cuts of NEXRAD or the extra low tilts of SAILS,
`grid_cones(..., repeats="mean")` merges them into one cone, and with `motion`
each repeat is first moved to the analysis time:

```{code-cell} ipython3
zs_section = np.arange(250.0, 10e3, 250.0)
distance = np.arange(2e3, 150e3, 1000.0)
with timed("10d. Radius of influence gridding"):
    cressman = grid_gates(
        volumes[1], ["DBZH"], x=xs, y=xs, z=zs, weighting="cressman", roi=1000.0, beamwidth=1.0
    )
    barnes = volumes[1].radarx.grid_gates(
        ["DBZH"], x=xs, y=xs, z=zs, weighting="barnes", roi=1000.0, beamwidth=1.0
    )
    nearest = grid_gates(volumes[1], ["DBZH"], x=xs, y=xs, z=zs, weighting="nearest", roi=1000.0)
    section = grid_rhi(
        volumes[1], ["DBZH"], azimuth=250.0, distance=distance, z=zs_section, roi=1000.0, beamwidth=1.0
    )
    section_acc = volumes[1].radarx.grid_rhi(
        ["DBZH"], azimuth=250.0, distance=distance, z=zs_section, roi=1000.0, beamwidth=1.0
    )
    merged = grid_cones(volumes[1], ["DBZH"], x=xs, y=xs, z=zs, repeats="mean")
both = cones.DBZH.notnull() & cressman.DBZH.notnull()
rms = float(np.sqrt(((cones.DBZH - cressman.DBZH).where(both) ** 2).mean()))
print(
    f"filled cells: cone {float(cones.DBZH.notnull().mean()):.2f}, "
    f"Cressman {float(cressman.DBZH.notnull().mean()):.2f}, Barnes {float(barnes.DBZH.notnull().mean()):.2f}; "
    f"rms difference cone - Cressman {rms:.1f} dB; section {dict(section.sizes)}; "
    f"split cuts merged, filled cells {float(merged.DBZH.notnull().mean()):.3f}"
)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
for ax, grid, label in zip(axes, (cones, cressman, barnes), ("grid_cones", "Cressman", "Barnes")):
    dbz2km = grid.DBZH.sel(z=2000.0)
    dbz2km.where(dbz2km >= 0).plot(ax=ax, x="x", y="y", cmap="ChaseSpectral", vmin=0, vmax=70)
    ax.set(title=f"{label}, 2 km", aspect="equal")
plt.show()
```

The vertical section of a volume along one azimuth is a pseudo range-height
indicator, the input of `plot_rhi`. With `plot_ppi`, `plot_cappi` and
`plot_maxcappi` the plotting functions of `radarx.vis` draw into Matplotlib
axes, so they combine with any other figure:

```{code-cell} ipython3
rays = []
for name in volumes[1].match("sweep_*").children:
    ds = volumes[1][name].to_dataset()
    if "DBZH" in ds:  # the surveillance cuts
        ray = ds.sel(azimuth=250.0, method="nearest")[["DBZH"]].drop_vars(["elevation", "x", "y", "z"], errors="ignore")
        ray = ray.sel(range=slice(None, 150e3)).interp(range=np.arange(1e3, 150e3, 1e3), method="nearest")  # 250 m or 1 km gates
        rays.append(ray.expand_dims(elevation=[float(ds.sweep_fixed_angle)]))
rhi = xr.concat(rays, dim="elevation", join="inner").sortby("elevation")
rhi = rhi.assign_coords(
    azimuth=("elevation", np.full(rhi.sizes["elevation"], 250.0)),
    latitude=float(volumes[1].latitude), longitude=float(volumes[1].longitude),
    altitude=float(volumes[1].altitude), sweep_mode="rhi",
).xradar.georeference()
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
style = dict(cmap="ChaseSpectral", vmin=-10, vmax=70, show_figure=False)
plot_ppi(volumes[1]["sweep_0"].to_dataset(), "DBZH", ax=axes[0], title="plot_ppi, 0.5°", **style)
plot_cappi(cappi, "DBZH", ax=axes[1], title="plot_cappi, 2 km", **style)
plot_rhi(rhi, "DBZH", ax=axes[2], title="plot_rhi, azimuth 250°", **style)
for ax in axes[:2]:
    ax.set(xlim=(-120, 120), ylim=(-120, 120))  # the axes of the plots are in km
axes[2].set_ylim(0, 15)
fig  # show_figure=False closes the figure, which is still displayed here
```

The same three plots are methods of the accessor of a sweep, a CAPPI and an RHI:

```{code-cell} ipython3
ax_ppi = volumes[1]["sweep_0"].to_dataset().radarx.plot_ppi("DBZH", show_figure=False)
ax_cappi = cappi.radarx.plot_cappi("DBZH", show_figure=False)
ax_rhi = rhi.radarx.plot_rhi("DBZH", show_figure=False)
print([type(ax).__name__ for ax in (ax_ppi, ax_cappi, ax_rhi)])
```

`plot_maxcappi` is the function behind the `plot_max_cappi` accessor method
of the figure above (here without the map, and not displayed again):

```{code-cell} ipython3
ax_max = plot_maxcappi(grids[1], "DBZH", add_map=False, cmap="ChaseSpectral", vmin=-10, vmax=70,
                       range_rings=True, show_figure=False)
print(type(ax_max).__name__)
```

The interactive counterparts of every plotting function return HoloViews
objects (the bokeh or the matplotlib backend; the `hvplot_*` functions are what
`.radarx.plot.ppi()` and the other accessor methods call). Most of them are
rasterized with datashader, so that they stay responsive with a million gates.
A selection on the 00:00 UTC volume; the last line displays a static PPI:

```{code-cell} ipython3
dbz0 = volumes[1]["sweep_0"].to_dataset().DBZH
views = {
    "hvplot_ppi": hvplot_ppi(dbz0, clim=(-10, 70), cmap="ChaseSpectral", frame_width=380),
    "hvplot_range_azimuth": hvplot_range_azimuth(dbz0, clim=(-10, 70), cmap="ChaseSpectral"),
    "hvplot_mesh": hvplot_mesh(dbz0.isel(azimuth=slice(0, 30), range=slice(0, 60))),
    "hvplot_centroids": hvplot_centroids(dbz0.isel(range=slice(0, 120))),
    "hvplot_cappi": hvplot_cappi(grids[1].DBZH, z=2000, cmap="ChaseSpectral", clim=(-10, 70)),
    "hvplot_max_cappi": hvplot_max_cappi(grids[1].DBZH, cmap="ChaseSpectral", clim=(-10, 70)),
    "hvplot_rhi": hvplot_rhi(rhi.DBZH, clim=(-10, 70), cmap="ChaseSpectral"),
}
accessors = {
    "DataArray": isinstance(dbz0.radarx.plot, RadarxDataArrayPlotAccessor),
    "Dataset": isinstance(dbz0.to_dataset().radarx.plot, RadarxDatasetPlotAccessor),
    "DataTree": isinstance(volumes[1].radarx.plot, RadarxDataTreePlotAccessor),
}
print({name: type(view).__name__ for name, view in views.items()})
print("the .radarx.plot accessor classes of DataArray, Dataset and DataTree:", accessors)
# a static version of the PPI of a decimated sweep (every 4th ray, the first 150 gates)
hvplot_ppi(dbz0.isel(azimuth=slice(0, 360, 4), range=slice(0, 150)), rasterize=False,
           clim=(-10, 70), cmap="ChaseSpectral", frame_width=380)
```

## 11. Storm motion, common analysis time and time interpolation

The two volumes are 6.5 minutes apart, during which the line moved several
kilometres. `estimate_motion` tracks the reflectivity pattern between the two
grids; `advect` moves every gridded field of each volume to a common
analysis time (here 00:00 UTC), and `interpolate_time` produces
advection-corrected frames between the two volumes.

```{code-cell} ipython3
analysis_time = np.datetime64("2022-03-31T00:00:00", "ns")
with timed("11. motion, advection, time interpolation"):
    motion = estimate_motion(grids[0], grids[1])
    common = [advect(g, motion, time=analysis_time) for g in grids]
    frame_times = np.arange(
        grids[0].time.values.astype("datetime64[m]") + np.timedelta64(1, "m"),
        grids[1].time.values,
        np.timedelta64(1, "m"),
    ).astype("datetime64[ns]")
    frames = interpolate_time(grids[0][["DBZH"]], grids[1][["DBZH"]], frame_times, motion=motion)
motion_acc = grids[0].radarx.estimate_motion(grids[1])  # the accessors of a grid
common_acc = grids[0].radarx.advect(motion_acc, time=analysis_time)
frames_acc = grids[0][["DBZH"]].radarx.interpolate_time(grids[1][["DBZH"]], frame_times, motion=motion_acc)
print("accessors and functions agree:", bool(np.allclose(motion_acc.to_array(), motion.to_array())),
      bool(np.allclose(common_acc.DBZH, common[0].DBZH, equal_nan=True)),
      bool(np.allclose(frames_acc.DBZH, frames.DBZH, equal_nan=True)))
motion.to_pandas()
```

```{code-cell} ipython3
def composite(ds):
    """Column maximum reflectivity, with x and y in km."""
    da = ds.DBZH.max("z")
    return da.assign_coords(x=da.x / 1e3, y=da.y / 1e3)


fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True, layout="constrained")
for ax, data, title in zip(axes, [grids, common], ["as observed", "advected to 00:00 UTC"]):
    ax.pcolormesh(grids[1].x / 1e3, grids[1].y / 1e3, composite(data[1]), cmap="Greys", vmin=0, vmax=70)
    for g, color in zip(data, colors):
        c = composite(g)
        ax.contour(c.x, c.y, c.fillna(-99), levels=[50], colors=[color], linewidths=1.2)
    ax.set_title(f"50 dBZ contours, {title}")
    ax.set_aspect("equal")
    ax.set_xlabel("east of KGWX (km)")
axes[0].set_ylabel("north of KGWX (km)")
handles = [plt.Line2D([], [], color=c) for c in colors]
labels = [str(g.time.values)[11:16] + " UTC" for g in grids]
axes[1].legend(handles, labels, loc="lower right")
plt.show()
```

Moving both volumes to one time brings their 50 dBZ cores together. A single
motion vector cannot follow parts of the storm that move differently; a
tiled motion field (`estimate_motion(..., tile=60e3)`) can.

Advection-corrected frames, one per minute between the two volumes:

```{code-cell} ipython3
fg = composite(frames.isel(time=slice(None, None, 2))).plot(
    x="x", y="y", col="time", cmap="ChaseSpectral", vmin=-10, vmax=70,
    figsize=(13, 4.6), cbar_kwargs={"label": "column maximum reflectivity (dBZ)"},
)
for ax in fg.axs.flat:
    ax.set_aspect("equal")
    ax.set_xlabel("east of KGWX (km)")
fg.axs[0, 0].set_ylabel("north of KGWX (km)")
plt.show()
```

## 12. Multi-radar grid and three-dimensional wind

A second radar looking at the same storm from another direction gives a
second wind component. KBMX (Birmingham, Alabama) is 166 km east-south-east
of KGWX. We read only its reflectivity and velocity, dealias the velocity
with the same ERA5 wind profile, and

- grid both radars onto one shared grid centred on KGWX
  (`radarx.grid.grid_radars`), moving each radar to 00:00 UTC with the storm
  motion of step 11, and merge their reflectivity;
- retrieve the three-dimensional wind with the variational multi-Doppler
  method (`multi_doppler`; Gao et al. 1999), with the ERA5 profile as the
  background, and with the classical analytic solution of the 2 x 2 system
  of the two radial velocities (`dual_doppler_analytic`) for a quick check.

The [multi-radar gridding](Multi_Radar_Grid) and [multi-Doppler](Multi_Doppler)
notebooks show both in more detail.

From here on only the 00:00 UTC KGWX volume is used, so we release the polar
data of the first one.

```{code-cell} ipython3
kgwx = volumes[1]
del volumes
gc.collect()

with timed("12a. KBMX: read, mask, dealias"):
    path = download_file(
        "unidata-nexrad-level2", "2022/03/30/KBMX/KBMX20220330_235713_V06", "data"
    )
    kbmx = read_nexrad(path, fields=("DBZH", "VRADH"), max_range=300e3)
    kbmx = kbmx.radarx.apply_mask(kbmx.radarx.echo_mask())
    kbmx = kbmx.radarx.assign(kbmx.radarx.dealias("VRADH", wind_profile=era5))

x = np.arange(-120e3, 60e3 + 1, 2000.0)
y = np.arange(-120e3, 100e3 + 1, 2000.0)
z = np.arange(1000.0, 10e3 + 1, 1000.0)
with timed("12b. multi-radar grid"):
    mosaic = kgwx.radarx.grid_radars(
        kbmx,
        x,
        y,
        z,
        data_vars=["DBZH"],
        names=["KGWX", "KBMX"],
        time=analysis_time,
        motion=motion,
        merge=["DBZH"],
    )
with timed("12c. multi-Doppler wind"):
    md_grid = multi_doppler_input(
        [kgwx, kbmx], x, y, z, field="VRADH_dealiased", time=analysis_time, motion=motion
    )
    background = sounding.profile_to_grid(era5, md_grid)
    wind = md_grid.radarx.multi_doppler(background, field="VRADH_dealiased")
    # the classical analytic solution (2 x 2 system per cell, continuity equation upward)
    analytic = md_grid.radarx.dual_doppler_analytic(background, field="VRADH_dealiased")
    analytic_fn = dual_doppler_analytic(md_grid, background, field="VRADH_dealiased")
print(f"iterations per grid level: {wind.attrs['iterations']}")
solved = analytic.u.notnull() & (wind.n_radars >= 2)
print(f"analytic dual-Doppler: {int(solved.sum())} cells with a horizontal wind, "
      f"rms difference to the variational u {float(np.sqrt(((analytic.u - wind.u).where(solved) ** 2).mean())):.1f} m/s; "
      "the accessor and the function agree:", bool(analytic.u.equals(analytic_fn.u)))
mosaic_fn = grid_radars([kgwx, kbmx], x, y, z, data_vars=["DBZH"], names=["KGWX", "KBMX"],
                        time=analysis_time, motion=motion, merge=["DBZH"])
print("the accessor of the first volume and grid_radars give the same mosaic:",
      bool(np.allclose(mosaic.DBZH_merged, mosaic_fn.DBZH_merged, equal_nan=True)))
```

`era5_column` is the three-dimensional counterpart of `era5_profile`: it
interpolates ERA5 to every cell of a grid (here the wind that is used as the
background of the multi-Doppler analysis, rotated to the grid axes), and
`radar_geometry` gives the beam azimuth and elevation of every radar at every
cell, from which the beam-crossing angle of the analysis follows:

```{code-cell} ipython3
with timed("12d. ERA5 on the grid, beam geometry"):
    era5_3d = era5_column(md_grid, time="2022-03-31T00:00", source="gcs")
    era5_acc = md_grid.radarx.background(source="gcs", time="2022-03-31T00:00")  # the accessor of the grid
    flat = md_grid.radarx.background(era5)  # one sounding spread over the grid
    sweep_env = kgwx["sweep_0"].to_dataset().radarx.interpolate_profile(era5, ["temperature"])  # at every gate
    geometry = radar_geometry(md_grid)
print(f"ERA5 wind at 5 km over the grid: u {float(era5_3d.u.sel(z=5000).mean()):.1f} m/s, "
      f"v {float(era5_3d.v.sel(z=5000).mean()):.1f} m/s; "
      f"beam elevation of KBMX at 5 km above KGWX: {float(geometry.elevation.isel(radar=1).sel(z=5000, x=0, y=0, method='nearest')):.1f}°")
print("accessor and era5_column agree:", bool(np.allclose(era5_acc.u, era5_3d.u, equal_nan=True)),
      f"| temperature at the gates of the 0.5° sweep: {float(sweep_env.temperature.mean()):.1f} K")
del kbmx, md_grid, background
gc.collect();
```

Two overlapping radars rarely agree exactly in reflectivity. `network_bias`
estimates the relative calibration of each radar against a reference from the
distribution of the differences in the overlap (Seo et al. 2014), and
`merge_radars` (the merge step of `grid_radars`) combines the radars with
weights that decrease with range and time offset (Zhang et al. 2005), after
removing the bias:

```{code-cell} ipython3
bias = network_bias(mosaic, "DBZH", reference="KGWX", min_count=50)
merged = merge_radars(mosaic, "DBZH", bias=bias, time=analysis_time)
bias_acc = mosaic.radarx.network_bias("DBZH", reference="KGWX", min_count=50)  # the accessors of the mosaic
merged_acc = mosaic.radarx.merge_radars("DBZH", bias=bias_acc, time=analysis_time)
print(f"relative bias of KBMX with respect to KGWX: {float(bias.bias.sel(radar='KBMX')):+.2f} dB "
      f"(median of {int(bias.pair_count.sel(radar='KBMX', other='KGWX'))} gate pairs)")
fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained", sharey=True)
pm = axes[0].pcolormesh(x / 1e3, y / 1e3, merged.DBZH.sel(z=3000), cmap="ChaseSpectral", vmin=-10, vmax=70)
fig.colorbar(pm, ax=axes[0], label="reflectivity (dBZ)", shrink=0.8)
axes[0].set_title("3 km, merged after removing the bias")
diff = merged.DBZH.sel(z=3000) - mosaic.DBZH_merged.sel(z=3000)
pd_ = axes[1].pcolormesh(x / 1e3, y / 1e3, diff, cmap="RdBu_r", vmin=-2, vmax=2)
fig.colorbar(pd_, ax=axes[1], label="change by the bias correction (dB)", shrink=0.8)
axes[1].set_title("3 km, difference to the merge without correction")
for ax in axes:
    ax.set(aspect="equal", xlabel="east of KGWX (km)")
axes[0].set_ylabel("north of KGWX (km)")
plt.show()
```

```{code-cell} ipython3
good = (wind.beam_crossing_angle > 30) & (wind.n_radars >= 2)
height = 5000.0
level = wind[["u", "v", "w"]].sel(z=height).where(good.sel(z=height))
sub = level.isel(x=slice(None, None, 3), y=slice(None, None, 3))
dbz = mosaic.DBZH_merged.sel(z=height)
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), layout="constrained", sharey=True)
im0 = axes[0].pcolormesh(x / 1e3, y / 1e3, dbz, cmap="ChaseSpectral", vmin=-10, vmax=70)
im1 = axes[1].pcolormesh(x / 1e3, y / 1e3, level.w, cmap="RdBu_r", vmin=-8, vmax=8)
axes[1].contour(x / 1e3, y / 1e3, dbz.fillna(-30), levels=[35, 50], colors="k", linewidths=0.6)
for ax, title in zip(axes, ["merged reflectivity and wind", "w (contours: 35 and 50 dBZ)"]):
    ax.quiver(sub.x / 1e3, sub.y / 1e3, sub.u, sub.v, scale=700, width=0.002)
    ax.plot(mosaic.radar_x / 1e3, mosaic.radar_y / 1e3, "k^")
    ax.set_title(f"{height / 1e3:.0f} km, 00:00 UTC: {title}")
    ax.set_xlim(x[0] / 1e3, x[-1] / 1e3)
    ax.set_ylim(y[0] / 1e3, y[-1] / 1e3)
    ax.set_aspect("equal")
    ax.set_xlabel("east of KGWX (km)")
axes[0].set_ylabel("north of KGWX (km)")
fig.colorbar(im0, ax=axes[0], label="reflectivity (dBZ)", shrink=0.8)
fig.colorbar(im1, ax=axes[1], label="w (m/s)", shrink=0.8)
plt.show()
```

The dual-Doppler area is where both radars observe with a beam crossing
angle above 30°; KBMX is far away, so its beams reach the line only above
about 3 km. Updrafts lie along the leading edge of the line.

+++

## 13. Evaporation

Rain that falls below cloud base into unsaturated air evaporates and cools
it. `evaporation` computes the evaporation and cooling rates of a gamma DSD
from the temperature, pressure and humidity of the air, with the ventilated
diffusion of each drop in closed form (Kumjian and Ryzhkov 2010), and
`integrate_evaporation` lets the air under a sequence of DSDs cool and moisten
until the next volume. Here the DSD of the QVPs of step 8 (rain below the
melting layer, $\rho_{hv} \geq 0.97$) is combined with the ERA5 profile of step 3:

```{code-cell} ipython3
with timed("13. evaporation"):
    bottom = ml.melting_layer_bottom.to_series().ffill().bfill().to_xarray()
    rain_qvp = (tqvp.height < bottom - 250.0) & (tqvp.RHOHV >= 0.97) & (tqvp.DBZH >= 5.0)
    dsd_qvp = tqvp.radarx.dsd(band="S", mask=rain_qvp)
    rates = evaporation(dsd_qvp, era5)
    state = integrate_evaporation(dsd_qvp, era5, max_step=30.0)
    rates_acc = dsd_qvp.radarx.evaporation(era5)  # the same from the accessors of the DSD
    state_acc = dsd_qvp.radarx.integrate_evaporation(era5, max_step=30.0)
cloud_base = float(interpolate_profile(era5, tqvp.height, ["relative_humidity"]).pipe(
    lambda p: p.height.where(p.relative_humidity >= 0.95).min()))
below = rain_qvp & (tqvp.height < cloud_base)
print(f"ERA5 cloud base (relative humidity 95 %): {cloud_base:.0f} m above sea level")
print("accessors agree:", bool(np.allclose(rates_acc.COOLING_RATE, rates.COOLING_RATE, equal_nan=True)),
      bool(np.allclose(state_acc.TEMPERATURE_CHANGE, state.TEMPERATURE_CHANGE, equal_nan=True)))
rates[["EVAPORATION_RATE", "COOLING_RATE_HOURLY"]].where(below).max("height").to_dataframe().round(2)
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True, layout="constrained")
for i, color in enumerate(colors):
    label = str(tqvp.time.values[i])[11:16] + " UTC"
    axes[0].plot(tqvp.DBZH.isel(time=i).where(below.isel(time=i)), tqvp.height / 1e3, color=color, label=label)
    axes[1].plot(rates.COOLING_RATE_HOURLY.isel(time=i).where(below.isel(time=i)), tqvp.height / 1e3, color=color)
    axes[2].plot(state.TEMPERATURE_CHANGE.isel(time=i).where(tqvp.height < cloud_base), tqvp.height / 1e3, color=color)
for ax in axes:
    ax.axhline(cloud_base / 1e3, color="m", ls=":")
axes[0].set(ylim=(0, 5), ylabel="height above sea level (km)", xlabel="$Z_H$ in the rain (dBZ)")
axes[1].set(xlabel="cooling rate (K/h)")
axes[2].set(xlabel="temperature change since the first volume (K)")
axes[0].legend(frameon=False, loc="upper right", title="volume")
fig.suptitle("Evaporation below cloud base from the QVPs of the 19.5° sweep (dotted: ERA5 cloud base)")
plt.show()
```

Below cloud base the moderate rain of the QVP circle (within about 10 km of
the radar) cools the air by 1 to 2 K/h, strongest near the ground where the
air is driest, so that the 7 minutes between the two volumes change the
temperature by about 0.2 K. This is the cooling that the air would feel if it
stayed under the rain; the heavier rain of the convective line itself falls
outside the QVP circle.

## 14. VIL, echo tops and water content

Column products of the reflectivity of the volume. `vil` integrates the liquid
water of rain over height with the relation of Greene and Clark (1972),
$\mathrm{VIL} = 3.44\times10^{-6}\sum[(Z_i+Z_{i+1})/2]^{4/7}\Delta h$ in
kg m$^{-2}$, for every (azimuth, ground range) column of the volume: each sweep
contributes its value at the beam height, the layer under the lowest beam is left
out (a lower bound) and nothing is assumed above the highest. `echo_top` is the
height of the highest 18 dBZ echo, interpolated in dBZ between the beams that
bracket the threshold (Lakshmanan et al. 2013), and `vil_density` is the VIL
divided by the echo top (Amburn and Wolf 1997). With a melting level, `vil` also
returns the VIL of the rain below it, here the ERA5 0 °C height of step 3.
`liquid_water_content` gives the water content of rain gate by gate, from the
power law of the VIL formula or from the gamma DSD of step 7; it is meaningful in
rain only, so the rain gates of the hydrometeor classification select it. The
[VIL and water content notebook](VIL_and_Water_Content) shows the products on a
cone grid, on CSAPR2 and in a QVP.

```{code-cell} ipython3
with timed("14. VIL, echo top, VIL density, water content"):
    vil_vol = vil(kgwx, melting=freezing_level)
    top = echo_top(kgwx)
    density = vil_density(kgwx)
    lwc_dsd = liquid_water_content(surv, "dsd", kdp="KDP", mask=rain, band="S")
    lwc_zm = liquid_water_content(surv, mask=rain)
    vil_acc = kgwx.radarx.vil(melting=freezing_level)  # the accessors of the volume
    top_acc = kgwx.radarx.echo_top()
    density_acc = kgwx.radarx.vil_density()
    lwc_acc = surv.radarx.liquid_water_content(mask=rain)  # and of the sweep
vil_qvp = vil(tqvp, melting=ml)  # the VIL of the QVPs, below the melting layer
print("accessors agree:", bool(vil_acc.VIL.equals(vil_vol.VIL)), bool(top_acc.equals(top)),
      bool(density_acc.equals(density)), bool(lwc_acc.equals(lwc_zm)))
print(f"VIL of the volume: maximum {float(vil_vol.VIL.max()):.1f} kg m-2, "
      f"{float(vil_vol.VIL_LIQUID.max()):.1f} below the 0 °C level; "
      f"QVP VIL {vil_qvp.VIL.values.round(2)} kg m-2 (rain: {vil_qvp.VIL_LIQUID.values.round(2)})")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), sharey=True, layout="constrained")
panels = [
    (vil_vol.VIL, "VIL (kg m$^{-2}$)", dict(cmap="viridis", vmin=0, vmax=30)),
    (top / 1e3, "18 dBZ echo top (km)", dict(cmap="cividis", vmin=0, vmax=16)),
    (density, "VIL density (g m$^{-3}$)", dict(cmap="plasma", vmin=0, vmax=3.5)),
    (lwc_dsd, "LWC of the rain, gamma DSD (g m$^{-3}$)", dict(cmap="viridis", vmin=0, vmax=6)),
]
for ax, (da, label, style) in zip(axes, panels):
    pm = ppi(ax, surv.assign(field=da), "field", extent=150, **style)
    fig.colorbar(pm, ax=ax, label=label, shrink=0.85)
axes[0].set_ylabel("north of KGWX (km)")
fig.suptitle("VIL, echo top, VIL density and rain water content, KGWX, 00:00 UTC")
plt.show()
```

The columns of the volume sit at the ground range of the lowest sweep, and a
sweep contributes its value at the height of its beam above that ground range.
The two geometry functions that give them are `radarx.fundamentals.geometry.ground_range`
(slant range to ground range) and `beam_height_at_ground_range`; for a gate of
the 0.5° sweep at 100 km slant range:

```{code-cell} ipython3
from radarx.fundamentals import geometry

ground = geometry.ground_range(100e3, 0.5)
print(f"ground range {ground / 1e3:.2f} km, beam height {geometry.beam_height_at_ground_range(ground, 0.5, 30.0):.0f} m "
      f"(slant-range formula: {geometry.beam_center_height(100e3, 0.5, 30.0):.0f} m)")
```

The VIL of the line is 10 to 25 kg m$^{-2}$ with echo tops of 8 to 10 km;
the water content is for the rain gates of the lowest sweep only. Within about
10 km of the radar the highest beam is only a few km high, so the VIL and
the echo top there are truncated (`VIL_TOP_TRUNCATED`).

### Hail products

The hail detection algorithm of Witt et al. (1998) turns the same columns and the
heights of the 0 and −20 °C levels into the severe hail index `shi`
($\mathrm{SHI}=0.1\sum W_T\,\dot E\,\Delta H$ with the kinetic energy flux
$\dot E=5\times10^{-6}\,10^{0.084Z}\,W(Z)$), the maximum expected hail size `mesh`
($2.54\,\mathrm{SHI}^{0.5}$ mm), the probability of severe hail `posh`
and the hail kinetic energy flux of the lowest beam `hail_kinetic_energy`;
`hail_products` returns them together. The two levels come here from the ERA5
profile of step 3 (`profile=era5`; they can also be numbers or fields). The
[hail notebook](Hail_Products) checks the sums against closed-form columns and
against the worked example of the paper.

```{code-cell} ipython3
with timed("14b. hail products"):
    hail = hail_products(kgwx, profile=era5)
    shi_vol = shi(kgwx, profile=era5)
    mesh_vol = mesh(kgwx, profile=era5)
    posh_vol = posh(kgwx, profile=era5)
    flux_vol = hail_kinetic_energy(kgwx)
    hail_acc = kgwx.radarx.hail_products(profile=era5)  # the accessors of the volume
    shi_acc = kgwx.radarx.shi(profile=era5)
    mesh_acc = kgwx.radarx.mesh(profile=era5)
    posh_acc = kgwx.radarx.posh(profile=era5)
    flux_acc = kgwx.radarx.hail_kinetic_energy()
print("accessors agree:", bool(hail_acc.SHI.equals(hail.SHI)), bool(shi_acc.equals(shi_vol)),
      bool(mesh_acc.equals(mesh_vol)), bool(posh_acc.equals(posh_vol)), bool(flux_acc.equals(flux_vol)))
print(f"SHI above 0 in {int((hail.SHI > 0).sum())} columns, maximum {float(hail.SHI.max()):.0f} J m-1 s-1, "
      f"MESH {float(hail.MESH.max()):.0f} mm, POSH {float(hail.POSH.max()):.0f} %, "
      f"hail kinetic energy flux {float(flux_vol.max()):.2f} J m-2 s-1")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, layout="constrained")
panels = [
    (hail.SHI.where(hail.SHI > 0), "SHI (J m$^{-1}$ s$^{-1}$)", dict(cmap="magma_r", vmin=0, vmax=60)),
    (hail.MESH.where(hail.MESH > 0), "MESH (mm)", dict(cmap="viridis", vmin=0, vmax=25)),
    (hail.POSH.where(hail.POSH > 0), "POSH (%)", dict(cmap="cividis", vmin=0, vmax=60)),
]
for ax, (da, label, style) in zip(axes, panels):
    pm = ppi(ax, surv.assign(field=da), "field", extent=150, **style)
    fig.colorbar(pm, ax=ax, label=label, shrink=0.85)
axes[0].set_ylabel("north of KGWX (km)")
fig.suptitle("Hail products with the ERA5 0 and -20 °C levels, KGWX, 00:00 UTC")
plt.show()
```

SHI counts only echo of 40 dBZ or more above the 0 °C level, so it is positive in
the cores of the line and zero elsewhere; the warning threshold of POSH at this
melting level is about 74 J m$^{-1}$ s$^{-1}$, so POSH stays low.

## 15. Noise and calibration

Three things limit how far the dual-polarization fields can be trusted: the noise
at long range, and the calibration of $Z_{DR}$, $Z$ and $\Phi_{DP}$. The
[noise and calibration notebook](Noise_and_Calibration) tests each function against a known
truth; here they run on the KGWX volume.

`noise_level` gives the noise of a sweep as the reflectivity of the noise at 1 km
(Hildebrand and Sekhon 1974, or the lowest reported reflectivity for NEXRAD data,
which are thresholded on SNR), `snr` the SNR of every gate from it, and
`correct_rhohv_noise` and `correct_zdr_noise` invert the effect of the noise of
the two channels on $\rho_{hv}$ and $Z_{DR}$ (Gourley et al. 2006). The moments
of the WSR-88D are already corrected for noise (the mean $\rho_{hv}$ at an SNR of
0 to 5 dB is close to 1), so on this volume the correction only shows what it does. For the calibration, `zdr_offset` compares $Z_{DR}$
in light rain with its intrinsic value (Ryzhkov et al. 2005), `z_offset` compares the
reflectivity with the one that $Z_{DR}$ and $K_{DP}$ imply in rain (the
self-consistency of Gorgucci et al. 1992), and `phidp_offset` finds the system
$\Phi_{DP}$ of each ray from its first gates. Offsets are the measured value minus
the true one.

```{code-cell} ipython3
with timed("15. noise, SNR, noise correction, calibration offsets"):
    sweep0 = kgwx["sweep_0"].to_dataset()
    noise = noise_level(kgwx, method="floor")
    snr_vol = snr(kgwx, noise=noise)
    rho_corrected = correct_rhohv_noise(kgwx, noise=noise)
    zdr_corrected = correct_zdr_noise(kgwx, noise=noise)
    zdr_cal = zdr_offset(kgwx)
    phi_cal = phidp_offset(kgwx)
    z_cal = z_offset(sweep0, mask=rain, zdr_offset=float(zdr_cal["sweep_0"].ZDR_OFFSET))
    # the accessors of the volume and of the sweep give the same results
    noise_acc = kgwx.radarx.noise_level(method="floor")
    snr_acc = kgwx.radarx.snr(noise=noise)
    rho_acc = kgwx.radarx.correct_rhohv_noise(noise=noise)
    zdr_acc = kgwx.radarx.correct_zdr_noise(noise=noise)
    zdr_cal_acc = kgwx.radarx.zdr_offset()
    phi_cal_acc = kgwx.radarx.phidp_offset()
    z_cal_acc = sweep0.radarx.z_offset(mask=rain, zdr_offset=float(zdr_cal["sweep_0"].ZDR_OFFSET))
print("accessors agree:",
      float(noise_acc["sweep_0"].NOISE_LEVEL) == float(noise["sweep_0"].NOISE_LEVEL),
      bool(snr_acc["sweep_0"].SNR.equals(snr_vol["sweep_0"].SNR)),
      bool(rho_acc["sweep_0"].RHOHV.equals(rho_corrected["sweep_0"].RHOHV)),
      bool(zdr_acc["sweep_0"].ZDR.equals(zdr_corrected["sweep_0"].ZDR)),
      float(zdr_cal_acc["sweep_0"].ZDR_OFFSET) == float(zdr_cal["sweep_0"].ZDR_OFFSET),
      float(phi_cal_acc["sweep_0"].SYSTEM_PHIDP) == float(phi_cal["sweep_0"].SYSTEM_PHIDP),
      float(z_cal_acc.Z_OFFSET) == float(z_cal.Z_OFFSET))
s0 = "sweep_0"
print(f"noise level {float(noise[s0].NOISE_LEVEL):.1f} dBZ at 1 km | "
      f"ZDR offset {float(zdr_cal[s0].ZDR_OFFSET):+.2f} dB ({int(zdr_cal[s0].ZDR_OFFSET_COUNT)} gates) | "
      f"Z offset {float(z_cal.Z_OFFSET):+.2f} dB ({int(z_cal.Z_OFFSET_BINS)} classes) | "
      f"system PhiDP {float(phi_cal[s0].SYSTEM_PHIDP):.1f}° (rays scatter {float(phi_cal[s0].SYSTEM_PHIDP_SPREAD):.1f}°)")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True, layout="constrained")
panels = [
    (snr_vol[s0].SNR, "SNR (dB)", dict(cmap="viridis", vmin=0, vmax=60)),
    (rho_corrected[s0].RHOHV, "$\\rho_{hv}$ corrected for noise", dict(cmap="HomeyerRainbow", vmin=0.8, vmax=1.0)),
    (zdr_corrected[s0].ZDR, "$Z_{DR}$ corrected for noise (dB)", dict(cmap="HomeyerRainbow", vmin=-1, vmax=5)),
]
for ax, (da, label, style) in zip(axes, panels):
    pm = ppi(ax, sweep0.assign(field=da), "field", extent=150, **style)
    fig.colorbar(pm, ax=ax, label=label, shrink=0.85)
axes[0].set_ylabel("north of KGWX (km)")
fig.suptitle("Signal-to-noise ratio and noise-corrected moments, KGWX 0.5°, 00:00 UTC")
plt.show()

ray = phi_cal[s0].SYSTEM_PHIDP_RAY
fig, ax = plt.subplots(figsize=(8, 3.2), layout="constrained")
ax.plot(ray.azimuth, ray, ".", ms=3)
ax.axhline(float(phi_cal[s0].SYSTEM_PHIDP), color="C3", label="sweep value")
ax.set(xlabel="azimuth (°)", ylabel="system $\\Phi_{DP}$ (°)")
ax.legend(frameon=False)
plt.show()
```

## 16. Super-observations for data assimilation

A volume has millions of gates, a model a few hundred thousand cells, and the
errors of neighbouring gates are correlated, so assimilation systems use
averages of the gates in a box, the super-observations of Alpert and Kumar
(2007). `superobs` bins the gates into columns of 3 x 3 cells of a model grid and
layers of 500 m, and returns the mean reflectivity (averaged in linear $Z$), the
mean radial velocity with the mean unit vector of the rays (so that an
operator can project the model wind), the number of gates, the spread and an
error estimate $\sqrt{\mathrm{floor}^2 + \mathrm{spread}^2/n}$, and flags the
boxes that have gates but no echo. `write_wrfda_radar` writes the result as
the `ob.radar` file of WRFDA and `write_dart_obs_seq` as a DART `obs_seq` file. The model here is a Lambert conformal grid of
1 km cells; a WRF file with `XLAT` and `XLONG` is used the same way. The
[super-observation notebook](Superobs_Data_Assimilation) shows the gates and the
super-observations, the errors and the file in more detail.

```{code-cell} ipython3
from pyproj import Transformer

lcc = "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=33.9 +lon_0=-88.33 +R=6370000 +units=m"
axis = (np.arange(300) - 149.5) * 1000.0
lon2d, lat2d = Transformer.from_crs(lcc, "EPSG:4326", always_xy=True).transform(*np.meshgrid(axis, axis))

with timed("16. super-observations, WRFDA and DART files"):
    so = superobs(kgwx, (lat2d, lon2d), cells=3, field="VRADH_dealiased", min_rhohv=0.85, max_range=150e3)
    so_acc = kgwx.radarx.superobs((lat2d, lon2d), cells=3, field="VRADH_dealiased", min_rhohv=0.85, max_range=150e3)
    stats = write_wrfda_radar(so, "ob.radar", time="2022-03-31_00:00:00")
    dart_stats = write_dart_obs_seq(so, "obs_seq.out", time="2022-03-31_00:00:00")
print("accessor agrees:", bool(so.reflectivity.equals(so_acc.reflectivity)), bool(so.velocity.equals(so_acc.velocity)))
print(f"{int(so.gate_count.sum()):,} gates in {so.sizes['obs']:,} boxes: {int(so.reflectivity.count()):,} with "
      f"reflectivity, {int(so.velocity.count()):,} with velocity, {int(so.nonprecip.sum()):,} without echo; "
      f"WRFDA file: {stats['columns']:,} columns; DART file: {dart_stats['observations']:,} records")
```

```{code-cell} ipython3
layer = so.where(so.level == 4, drop=True)  # 2 to 2.5 km above sea level
to_xy = Transformer.from_crs("EPSG:4326", lcc, always_xy=True)
ox, oy = to_xy.transform(layer.longitude.values, layer.latitude.values)
ox0, oy0 = to_xy.transform(float(so.radar_longitude[0]), float(so.radar_latitude[0]))
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True, layout="constrained")
panels = [
    (layer.reflectivity, "reflectivity (dBZ)", dict(cmap="ChaseSpectral", vmin=-10, vmax=70)),
    (layer.velocity, "radial velocity (m s$^{-1}$)", dict(cmap="balance", vmin=-40, vmax=40)),
    (layer.velocity_error, "velocity error estimate (m s$^{-1}$)", dict(cmap="viridis", vmin=2, vmax=3)),
]
for ax, (da, label, style) in zip(axes, panels):
    ok = da.notnull().values
    sc = ax.scatter((ox - ox0)[ok] / 1e3, (oy - oy0)[ok] / 1e3, c=da[ok], s=9, marker="s", **style)
    fig.colorbar(sc, ax=ax, label=label, shrink=0.85)
    ax.set(xlim=(-150, 150), ylim=(-150, 150), aspect="equal", xlabel="east of KGWX (km)")
axes[0].set_ylabel("north of KGWX (km)")
fig.suptitle("Super-observations of 3 km, layer 2 to 2.5 km above sea level, KGWX, 00:00 UTC")
plt.show()
```

The dealiased velocity of step 4 is used as it is. The super-observations keep the
line and the low-level wind and replace millions of gates by tens of thousands of
boxes. Boxes without echo ahead of and behind the line are flagged, and WRFDA can use
them to remove spurious model rain.

## 17. Summary

The runtime of each step:

```{code-cell} ipython3
summary = pd.DataFrame({"time (s)": timings}).round(2)
summary.loc["all steps"] = summary.sum()
summary
```

All products of the case in one `DataTree`: the environment profiles,
the 00:00 UTC polar volume with every derived field, the DSD, the rain rate, the QVPs and melting
layer, the VIL products, the super-observations, the grids as observed and at the common analysis time, the storm
motion, the interpolated frames, the two-radar grid and the wind.

```{code-cell} ipython3
nodes = {"/": xr.Dataset(attrs={"title": "KGWX 30-31 March 2022 squall line, radarx products"})}
for node in kgwx.subtree:
    nodes[f"/polar{node.path if node.path != '/' else ''}"] = node.to_dataset(inherit=False)
nodes["/environment/era5"] = era5
nodes["/environment/radiosonde"] = raob
nodes["/dsd"] = dsd
nodes["/rain_rate"] = rain_blend
nodes["/qvp"] = tqvp
nodes["/melting_layer"] = ml
nodes["/vil"] = vil_vol.assign(ECHO_TOP=top, VIL_DENSITY=density)
nodes["/superobs"] = so
for i, (observed, advected) in enumerate(zip(grids, common)):
    nodes[f"/grid/observed/volume_{i}"] = observed
    nodes[f"/grid/analysis_time/volume_{i}"] = advected
nodes["/grid/frames"] = frames
nodes["/grid/two_radars"] = mosaic
nodes["/motion"] = motion
nodes["/wind"] = wind
products = xr.DataTree.from_dict(nodes)
products
```

## Function index

Every public name of radarx (the `__all__` of `radarx.retrieve`, `radarx.grid`,
`radarx.io`, `radarx.ml`, `radarx.vis`, `radarx.fundamentals`, `radarx.core`),
with its module and the section of this notebook or of the
[second part](Radar_Workflow_Advanced) or the
[third part](Radar_Workflow_Advanced_2) where it is used in an example. The
radarx IMD reader (`read_sweep`, `read_volume`, `to_cfradial2`,
`to_cfradial2_volumes` and the helpers of `radarx.testing`) is deprecated
and shown through xradar in the [IMD notebook](IMD_Radar_Data). A test
(`tests/test_workflow_covers_api.py`) fails when a name is missing from the
notebooks or from this index.

| function | module | section |
|---|---|---|
| `celsius_to_kelvin` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `celsius_to_si` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `degrees_to_radians` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `fahrenheit_to_kelvin` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `fahrenheit_to_si` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `feet_to_meters` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `frequency_to_wavelength` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `ghz_to_hz` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `gigahertz_to_si` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `hz_to_ghz` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `hz_to_mhz` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `kelvin_to_celsius` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `kelvin_to_fahrenheit` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `knots_to_mps` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `kph_to_mps` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `megahertz_to_si` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `meters_to_feet` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `meters_to_miles` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `mhz_to_hz` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `microseconds_to_seconds` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `microseconds_to_si` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `miles_to_meters` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `miles_to_si` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `minutes_to_seconds` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `minutes_to_si` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `mps_to_kph` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `radians_to_degrees` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `seconds_to_microseconds` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `seconds_to_minutes` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `si_to_celsius` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `si_to_fahrenheit` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `si_to_gigahertz` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `si_to_megahertz` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `si_to_microseconds` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `si_to_miles` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `si_to_minutes` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `wavelength_to_frequency` | `radarx.core.conversion` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `C` | `radarx.fundamentals` | [core](Radar_Workflow): 3. Environment |
| `DBZ_TO_Z_FACTOR` | `radarx.fundamentals` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `DIELECTRIC_ICE` | `radarx.fundamentals` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `DIELECTRIC_WATER` | `radarx.fundamentals` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `EARTH_RADIUS` | `radarx.fundamentals` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `EFFECTIVE_RADIUS_4_3` | `radarx.fundamentals` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `K_BOLTZMANN` | `radarx.fundamentals` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `RADAR_BANDS` | `radarx.fundamentals` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `T_STANDARD` | `radarx.fundamentals` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `TYPICAL_BEAMWIDTH` | `radarx.fundamentals` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `TYPICAL_PULSE_WIDTHS` | `radarx.fundamentals` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `Z_TO_DBZ_FACTOR` | `radarx.fundamentals` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `k_complex` | `radarx.fundamentals.attenuation` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `azimuthal_resolution` | `radarx.fundamentals.beam` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `beamwidth_to_radians` | `radarx.fundamentals.beam` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `compute_azimuth_resolution` | `radarx.fundamentals.beam` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `compute_beamwidth` | `radarx.fundamentals.beam` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `compute_volume_resolution` | `radarx.fundamentals.beam` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `volume_resolution` | `radarx.fundamentals.beam` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `dbz_from_z` | `radarx.fundamentals.common` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `dbz_to_z` | `radarx.fundamentals.common` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `ensure_positive` | `radarx.fundamentals.common` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `kilometers_to_meters` | `radarx.fundamentals.common` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `kilometers_to_si` | `radarx.fundamentals.common` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `km_to_m` | `radarx.fundamentals.common` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `km_to_si` | `radarx.fundamentals.common` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `kts_to_mps` | `radarx.fundamentals.common` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `kts_to_si` | `radarx.fundamentals.common` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `linearize_dbz` | `radarx.fundamentals.common` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `m_to_km` | `radarx.fundamentals.common` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `meters_to_kilometers` | `radarx.fundamentals.common` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `mps_to_knots` | `radarx.fundamentals.common` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `mps_to_kts` | `radarx.fundamentals.common` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `si_to_kilometers` | `radarx.fundamentals.common` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `si_to_km` | `radarx.fundamentals.common` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `si_to_kts` | `radarx.fundamentals.common` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `z_to_dbz` | `radarx.fundamentals.common` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `doppler_dilemma` | `radarx.fundamentals.doppler` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `dual_prf_velocity` | `radarx.fundamentals.doppler` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `max_frequency` | `radarx.fundamentals.doppler` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `nyquist_velocity` | `radarx.fundamentals.doppler` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `unambiguous_range` | `radarx.fundamentals.doppler` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `beam_center_height` | `radarx.fundamentals.geometry` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `beam_height_at_ground_range` | `radarx.fundamentals.geometry` | [core](Radar_Workflow): 14. VIL, echo tops and water content |
| `effective_radius` | `radarx.fundamentals.geometry` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `ground_range` | `radarx.fundamentals.geometry` | [core](Radar_Workflow): 14. VIL, echo tops and water content |
| `half_power_radius` | `radarx.fundamentals.geometry` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `k_e_from_refractivity` | `radarx.fundamentals.geometry` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `k_e_from_rays` | `radarx.fundamentals.refraction` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `trace_rays` | `radarx.fundamentals.refraction` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `sample_volume_gaussian` | `radarx.fundamentals.geometry` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `compute_average_power` | `radarx.fundamentals.power` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `compute_min_detectable_signal` | `radarx.fundamentals.power` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `compute_peak_power` | `radarx.fundamentals.power` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `compute_doppler_shift` | `radarx.fundamentals.principles` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `compute_nyquist_velocity` | `radarx.fundamentals.principles` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `compute_range_resolution` | `radarx.fundamentals.principles` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `compute_snr` | `radarx.fundamentals.principles` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `doppler_frequency_shift` | `radarx.fundamentals.principles` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `radar_range` | `radarx.fundamentals.principles` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `range_resolution` | `radarx.fundamentals.principles` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `round_trip_time` | `radarx.fundamentals.principles` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `snr` | `radarx.fundamentals.principles` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `dbz_attenuation_correction` | `radarx.fundamentals.reflectivity` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `hitschfeld_bordan_correction` | `radarx.fundamentals.reflectivity` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `z_to_r_custom` | `radarx.fundamentals.reflectivity` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `z_to_r_marshall_palmer` | `radarx.fundamentals.reflectivity` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `absorption_coefficient` | `radarx.fundamentals.scattering` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `backscatter_cross_section` | `radarx.fundamentals.scattering` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `extinction_coefficient` | `radarx.fundamentals.scattering` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `normalized_backscatter_cross_section` | `radarx.fundamentals.scattering` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `scattering_coefficient` | `radarx.fundamentals.scattering` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `size_parameter` | `radarx.fundamentals.scattering` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `ant_eff_area` | `radarx.fundamentals.system` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `antenna_gain` | `radarx.fundamentals.system` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `frequency` | `radarx.fundamentals.system` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `frequency_from_wavelength` | `radarx.fundamentals.system` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `power_return_target` | `radarx.fundamentals.system` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `pulse_duration` | `radarx.fundamentals.system` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `pulse_duration_from_length` | `radarx.fundamentals.system` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `pulse_length` | `radarx.fundamentals.system` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `pulse_length_from_duration` | `radarx.fundamentals.system` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `radar_const` | `radarx.fundamentals.system` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `radar_equation` | `radarx.fundamentals.system` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `size_param` | `radarx.fundamentals.system` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `solve_peak_power` | `radarx.fundamentals.system` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `wavelength` | `radarx.fundamentals.system` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `wavelength_from_frequency` | `radarx.fundamentals.system` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `compute_blind_range` | `radarx.fundamentals.timing` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `compute_duty_cycle` | `radarx.fundamentals.timing` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `compute_max_unambiguous_range` | `radarx.fundamentals.timing` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `compute_max_unambiguous_velocity` | `radarx.fundamentals.timing` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `compute_prf` | `radarx.fundamentals.timing` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `compute_pulse_repetition_interval` | `radarx.fundamentals.timing` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `axis_ratio_brandes` | `radarx.fundamentals.tmatrix` | [part 3](Radar_Workflow_Advanced_2): 14. Forward operator for WRF and MPAS output |
| `orientation_quadrature` | `radarx.fundamentals.tmatrix` | [part 3](Radar_Workflow_Advanced_2): 14. Forward operator for WRF and MPAS output |
| `raindrop_scattering` | `radarx.fundamentals.tmatrix` | [part 3](Radar_Workflow_Advanced_2): 14. Forward operator for WRF and MPAS output |
| `spheroid_scattering` | `radarx.fundamentals.tmatrix` | [part 3](Radar_Workflow_Advanced_2): 14. Forward operator for WRF and MPAS output |
| `water_refractive_index` | `radarx.fundamentals.tmatrix` | [part 3](Radar_Workflow_Advanced_2): 14. Forward operator for WRF and MPAS output |
| `circular_depolarization_ratio` | `radarx.fundamentals.variables` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `differential_reflectivity` | `radarx.fundamentals.variables` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `linear_depolarization_ratio` | `radarx.fundamentals.variables` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `radial_velocity` | `radarx.fundamentals.variables` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `reflectivity_factor` | `radarx.fundamentals.variables` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `grid_cones` | `radarx.grid.cone` | [core](Radar_Workflow): Other ways to grid, section and plot |
| `grid_gates` | `radarx.grid.gates` | [core](Radar_Workflow): Other ways to grid, section and plot |
| `grid_rhi` | `radarx.grid.gates` | [core](Radar_Workflow): Other ways to grid, section and plot |
| `grid_radar` | `radarx.grid.grid` | [core](Radar_Workflow): Other ways to grid, section and plot |
| `make_3d_grid` | `radarx.grid.grid` | [core](Radar_Workflow): Other ways to grid, section and plot |
| `stack_data` | `radarx.grid.grid` | [core](Radar_Workflow): Other ways to grid, section and plot |
| `grid_radars` | `radarx.grid.multi` | [core](Radar_Workflow): 12. Multi-radar grid and three-dimensional wind |
| `merge_radars` | `radarx.grid.multi` | [core](Radar_Workflow): 12. Multi-radar grid and three-dimensional wind |
| `network_bias` | `radarx.grid.multi` | [core](Radar_Workflow): 12. Multi-radar grid and three-dimensional wind |
| `gate_corners` | `radarx.grid.ugrid` | [core](Radar_Workflow): Other ways to grid, section and plot |
| `to_uxarray` | `radarx.grid.ugrid` | [core](Radar_Workflow): 10. Cone gridding, Max-CAPPI, interactive view and UGRID |
| `download_file` | `radarx.io.aws_data` | [core](Radar_Workflow): 1. Read the volumes |
| `get_s3_client` | `radarx.io.aws_data` | [core](Radar_Workflow): 1. Read the volumes |
| `list_available_files` | `radarx.io.aws_data` | [core](Radar_Workflow): 1. Read the volumes |
| `write_dart_obs_seq` | `radarx.io.dart` | [core](Radar_Workflow): 16. Super-observations for data assimilation |
| `parsivel_classes` | `radarx.io.disdrometer` | [part 2](Radar_Workflow_Advanced): 2. Disdrometers |
| `read_parsivel` | `radarx.io.disdrometer` | [part 2](Radar_Workflow_Advanced): 2. Disdrometers |
| `read_pips_netcdf` | `radarx.io.disdrometer` | [part 2](Radar_Workflow_Advanced): 2. Disdrometers |
| `write_cfradial1` | `radarx.io.export` | [part 3](Radar_Workflow_Advanced_2): 12. Export to files |
| `write_cfradial2` | `radarx.io.export` | [part 3](Radar_Workflow_Advanced_2): 12. Export to files |
| `write_netcdf_cf` | `radarx.io.export` | [part 3](Radar_Workflow_Advanced_2): 12. Export to files |
| `write_odim` | `radarx.io.export` | [part 3](Radar_Workflow_Advanced_2): 12. Export to files |
| `write_zarr` | `radarx.io.export` | [part 3](Radar_Workflow_Advanced_2): 12. Export to files |
| `read_lma` | `radarx.io.lma` | [part 2](Radar_Workflow_Advanced): 6. Lightning mapping |
| `parcel_diagnostics` | `radarx.io.parcel` | [core](Radar_Workflow): Parcel thermodynamics: CAPE, CIN, the lifted index and the levels |
| `read_mrr` | `radarx.io.profiler` | [part 2](Radar_Workflow_Advanced): 5. Surface stations, profilers and cold pools |
| `read_wind_profiler` | `radarx.io.profiler` | [part 2](Radar_Workflow_Advanced): 5. Surface stations, profilers and cold pools |
| `air_density` | `radarx.io.sounding` | [core](Radar_Workflow): Sounding utilities and wind-profile parameters |
| `dewpoint_from_specific_humidity` | `radarx.io.sounding` | [core](Radar_Workflow): Sounding utilities and wind-profile parameters |
| `dewpoint_from_vapor_pressure` | `radarx.io.sounding` | [core](Radar_Workflow): Sounding utilities and wind-profile parameters |
| `era5_column` | `radarx.io.sounding` | [core](Radar_Workflow): 12. Multi-radar grid and three-dimensional wind |
| `era5_profile` | `radarx.io.sounding` | [core](Radar_Workflow): Sounding utilities and wind-profile parameters |
| `geopotential_to_height` | `radarx.io.sounding` | [core](Radar_Workflow): Sounding utilities and wind-profile parameters |
| `interpolate_profile` | `radarx.io.sounding` | [core](Radar_Workflow): Sounding utilities and wind-profile parameters |
| `isotherm_height` | `radarx.io.sounding` | [core](Radar_Workflow): 3. Environment |
| `mean_wind` | `radarx.io.sounding` | [core](Radar_Workflow): Sounding utilities and wind-profile parameters |
| `nearest_station` | `radarx.io.sounding` | [core](Radar_Workflow): Sounding utilities and wind-profile parameters |
| `open_sounding_file` | `radarx.io.sounding` | [core](Radar_Workflow): Sounding utilities and wind-profile parameters |
| `profile_to_grid` | `radarx.io.sounding` | [core](Radar_Workflow): 12. Multi-radar grid and three-dimensional wind |
| `read_sounding` | `radarx.io.sounding` | [core](Radar_Workflow): Sounding utilities and wind-profile parameters |
| `refractivity` | `radarx.io.sounding` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `relative_humidity_from_dewpoint` | `radarx.io.sounding` | [core](Radar_Workflow): Sounding utilities and wind-profile parameters |
| `saturation_vapor_pressure` | `radarx.io.sounding` | [core](Radar_Workflow): Sounding utilities and wind-profile parameters |
| `specific_humidity_from_dewpoint` | `radarx.io.sounding` | [core](Radar_Workflow): Sounding utilities and wind-profile parameters |
| `station_list` | `radarx.io.sounding` | [core](Radar_Workflow): Sounding utilities and wind-profile parameters |
| `wet_bulb_temperature` | `radarx.io.sounding` | [core](Radar_Workflow): Sounding utilities and wind-profile parameters |
| `wet_bulb_zero_height` | `radarx.io.sounding` | [core](Radar_Workflow): 3. Environment |
| `read_pips` | `radarx.io.surface` | [part 2](Radar_Workflow_Advanced): 5. Surface stations, profilers and cold pools |
| `read_sticknet` | `radarx.io.surface` | [part 2](Radar_Workflow_Advanced): 5. Surface stations, profilers and cold pools |
| `read_sticknet_locations` | `radarx.io.surface` | [part 2](Radar_Workflow_Advanced): 5. Surface stations, profilers and cold pools |
| `write_wrfda_radar` | `radarx.io.wrfda` | [core](Radar_Workflow): 16. Super-observations for data assimilation |
| `list_models` | `radarx.ml.model` | [part 3](Radar_Workflow_Advanced_2): 10. Machine-learning plumbing |
| `load_model` | `radarx.ml.model` | [part 3](Radar_Workflow_Advanced_2): 10. Machine-learning plumbing |
| `Model` | `radarx.ml.model` | [part 3](Radar_Workflow_Advanced_2): 10. Machine-learning plumbing |
| `register_model` | `radarx.ml.model` | [part 3](Radar_Workflow_Advanced_2): 10. Machine-learning plumbing |
| `denormalize` | `radarx.ml.patches` | [part 3](Radar_Workflow_Advanced_2): 10. Machine-learning plumbing |
| `normalize` | `radarx.ml.patches` | [part 3](Radar_Workflow_Advanced_2): 10. Machine-learning plumbing |
| `PatchIndex` | `radarx.ml.patches` | [part 3](Radar_Workflow_Advanced_2): 10. Machine-learning plumbing |
| `polar_patches` | `radarx.ml.patches` | [part 3](Radar_Workflow_Advanced_2): 10. Machine-learning plumbing |
| `reassemble` | `radarx.ml.patches` | [part 3](Radar_Workflow_Advanced_2): 10. Machine-learning plumbing |
| `advect` | `radarx.retrieve.advection` | [core](Radar_Workflow): 11. Storm motion, common analysis time and time interpolation |
| `estimate_motion` | `radarx.retrieve.advection` | [core](Radar_Workflow): 11. Storm motion, common analysis time and time interpolation |
| `interpolate_time` | `radarx.retrieve.advection` | [core](Radar_Workflow): 11. Storm motion, common analysis time and time interpolation |
| `biological_echo` | `radarx.retrieve.biology` | [part 3](Radar_Workflow_Advanced_2): 7. Tornado detection and biological echo |
| `correct_attenuation` | `radarx.retrieve.attenuation` | [core](Radar_Workflow): 5. ΦDP processing and KDP |
| `beam_blockage` | `radarx.retrieve.beam_blockage` | [part 3](Radar_Workflow_Advanced_2): 11. Beam blockage and terrain clutter |
| `beam_path` | `radarx.retrieve.beam_path` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `zdr_offset` | `radarx.retrieve.calibration` | [core](Radar_Workflow): 15. Noise and calibration |
| `z_offset` | `radarx.retrieve.calibration` | [core](Radar_Workflow): 15. Noise and calibration |
| `phidp_offset` | `radarx.retrieve.calibration` | [core](Radar_Workflow): 15. Noise and calibration |
| `create_cappi` | `radarx.retrieve.cappi` | [core](Radar_Workflow): Other ways to grid, section and plot |
| `baroclinic_generation` | `radarx.retrieve.coldpool` | [part 2](Radar_Workflow_Advanced): 5. Surface stations, profilers and cold pools |
| `buoyancy` | `radarx.retrieve.coldpool` | [part 2](Radar_Workflow_Advanced): 5. Surface stations, profilers and cold pools |
| `cold_pool_intensity` | `radarx.retrieve.coldpool` | [part 2](Radar_Workflow_Advanced): 5. Surface stations, profilers and cold pools |
| `cold_pool_intensity_from_pressure` | `radarx.retrieve.coldpool` | [part 2](Radar_Workflow_Advanced): 5. Surface stations, profilers and cold pools |
| `cold_pool_intensity_from_surface` | `radarx.retrieve.coldpool` | [part 2](Radar_Workflow_Advanced): 5. Surface stations, profilers and cold pools |
| `cold_pool_perturbation` | `radarx.retrieve.coldpool` | [part 2](Radar_Workflow_Advanced): 5. Surface stations, profilers and cold pools |
| `potential_temperatures` | `radarx.retrieve.coldpool` | [part 2](Radar_Workflow_Advanced): 5. Surface stations, profilers and cold pools |
| `rkw_ratio` | `radarx.retrieve.coldpool` | [part 2](Radar_Workflow_Advanced): 5. Surface stations, profilers and cold pools |
| `zdr_column` | `radarx.retrieve.columns` | [part 3](Radar_Workflow_Advanced_2): 17. ZDR columns |
| `zdr_column_metrics` | `radarx.retrieve.columns` | [part 3](Radar_Workflow_Advanced_2): 17. ZDR columns |
| `dealias_confidence` | `radarx.retrieve.dealias` | [core](Radar_Workflow): 4. Dealias the Doppler velocity |
| `dealias_velocity` | `radarx.retrieve.dealias` | [core](Radar_Workflow): 4. Dealias the Doppler velocity |
| `correct_dual_prf` | `radarx.retrieve.dealias` | [part 3](Radar_Workflow_Advanced_2): 15. Dual-PRF velocity correction |
| `dealias_ambiguity` | `radarx.retrieve.dealias_methods` | [core](Radar_Workflow): 4. Dealias the Doppler velocity |
| `dealias_r2d2` | `radarx.retrieve.dealias_methods` | [core](Radar_Workflow): 4. Dealias the Doppler velocity |
| `diabatic_lagrangian` | `radarx.retrieve.diabatic_lagrangian` | [part 3](Radar_Workflow_Advanced_2): 9. Diabatic Lagrangian analysis |
| `microphysical_rates` | `radarx.retrieve.diabatic_lagrangian` | [part 3](Radar_Workflow_Advanced_2): 9. Diabatic Lagrangian analysis |
| `polarimetric_precipitation` | `radarx.retrieve.diabatic_lagrangian` | [part 3](Radar_Workflow_Advanced_2): 9. Diabatic Lagrangian analysis |
| `ziegler2013_precipitation` | `radarx.retrieve.diabatic_lagrangian` | [part 3](Radar_Workflow_Advanced_2): 9. Diabatic Lagrangian analysis |
| `ziegler2013_profiles` | `radarx.retrieve.diabatic_lagrangian` | [part 3](Radar_Workflow_Advanced_2): 9. Diabatic Lagrangian analysis |
| `disdrometer_qc` | `radarx.retrieve.disdrometer` | [part 2](Radar_Workflow_Advanced): 2. Disdrometers |
| `dsd_moments` | `radarx.retrieve.disdrometer` | [part 2](Radar_Workflow_Advanced): 2. Disdrometers |
| `fit_gamma` | `radarx.retrieve.disdrometer` | [part 2](Radar_Workflow_Advanced): 2. Disdrometers |
| `match_radar` | `radarx.retrieve.disdrometer` | [part 2](Radar_Workflow_Advanced): 2. Disdrometers |
| `number_concentration` | `radarx.retrieve.disdrometer` | [part 2](Radar_Workflow_Advanced): 2. Disdrometers |
| `process_disdrometer` | `radarx.retrieve.disdrometer` | [part 2](Radar_Workflow_Advanced): 2. Disdrometers |
| `radar_at_location` | `radarx.retrieve.disdrometer` | [part 2](Radar_Workflow_Advanced): 2. Disdrometers |
| `raupach_berne_correction` | `radarx.retrieve.disdrometer` | [part 2](Radar_Workflow_Advanced): 2. Disdrometers |
| `terminal_fall_speed` | `radarx.retrieve.disdrometer` | [part 2](Radar_Workflow_Advanced): 2. Disdrometers |
| `dsd` | `radarx.retrieve.dsd` | [core](Radar_Workflow): 7. Rain drop size distribution |
| `dsd_spectrum` | `radarx.retrieve.dsd` | [part 2](Radar_Workflow_Advanced): 3. Bayesian DSD retrieval with uncertainty |
| `fit_gamma_moments` | `radarx.retrieve.dsd` | [part 2](Radar_Workflow_Advanced): 2. Disdrometers |
| `parsivel_bins` | `radarx.retrieve.dsd` | [part 2](Radar_Workflow_Advanced): 2. Disdrometers |
| `radar_from_dsd` | `radarx.retrieve.dsd` | [part 2](Radar_Workflow_Advanced): 2. Disdrometers |
| `scattering_table` | `radarx.retrieve.dsd` | [part 2](Radar_Workflow_Advanced): 2. Disdrometers |
| `dsd_bayesian` | `radarx.retrieve.dsd_bayes` | [core](Radar_Workflow): 7. Rain drop size distribution |
| `dsd_prior` | `radarx.retrieve.dsd_bayes` | [core](Radar_Workflow): 7. Rain drop size distribution |
| `forward_grid` | `radarx.retrieve.dsd_bayes` | [core](Radar_Workflow): 7. Rain drop size distribution |
| `ensemble_retrieval` | `radarx.retrieve.ensemble` | [part 3](Radar_Workflow_Advanced_2): 16. Ensembles and uncertainty of the retrievals |
| `ensemble_statistics` | `radarx.retrieve.ensemble` | [part 3](Radar_Workflow_Advanced_2): 16. Ensembles and uncertainty of the retrievals |
| `measurement_error` | `radarx.retrieve.ensemble` | [part 3](Radar_Workflow_Advanced_2): 16. Ensembles and uncertainty of the retrievals |
| `perturb_radar` | `radarx.retrieve.ensemble` | [part 3](Radar_Workflow_Advanced_2): 16. Ensembles and uncertainty of the retrievals |
| `relation_scatter` | `radarx.retrieve.ensemble` | [part 3](Radar_Workflow_Advanced_2): 16. Ensembles and uncertainty of the retrievals |
| `drop_evaporation_rate` | `radarx.retrieve.evaporation` | [part 2](Radar_Workflow_Advanced): 4. Raindrop trajectories and size sorting |
| `evaporation` | `radarx.retrieve.evaporation` | [core](Radar_Workflow): 13. Evaporation |
| `integrate_evaporation` | `radarx.retrieve.evaporation` | [core](Radar_Workflow): 13. Evaporation |
| `forward_operator` | `radarx.retrieve.forward_operator` | [part 3](Radar_Workflow_Advanced_2): 14. Forward operator for WRF and MPAS output |
| `to_radar_grid` | `radarx.retrieve.forward_operator` | [part 3](Radar_Workflow_Advanced_2): 14. Forward operator for WRF and MPAS output |
| `wrf_state` | `radarx.retrieve.forward_operator` | [part 3](Radar_Workflow_Advanced_2): 14. Forward operator for WRF and MPAS output |
| `mpas_state` | `radarx.retrieve.forward_operator` | [part 3](Radar_Workflow_Advanced_2): 14. Forward operator for WRF and MPAS output |
| `hail_kinetic_energy` | `radarx.retrieve.hail` | [core](Radar_Workflow): 14. VIL, echo tops and water content |
| `hail_products` | `radarx.retrieve.hail` | [core](Radar_Workflow): 14. VIL, echo tops and water content |
| `mesh` | `radarx.retrieve.hail` | [core](Radar_Workflow): 14. VIL, echo tops and water content |
| `posh` | `radarx.retrieve.hail` | [core](Radar_Workflow): 14. VIL, echo tops and water content |
| `shi` | `radarx.retrieve.hail` | [core](Radar_Workflow): 14. VIL, echo tops and water content |
| `hid` | `radarx.retrieve.hid` | [core](Radar_Workflow): 6. Hydrometeor classification |
| `hid_classes` | `radarx.retrieve.hid` | [core](Radar_Workflow): 6. Hydrometeor classification |
| `hail_fall_speed` | `radarx.retrieve.ice_fall` | [part 2](Radar_Workflow_Advanced): 4. Raindrop trajectories and size sorting |
| `melting_snow_fall_speed` | `radarx.retrieve.ice_fall` | [part 2](Radar_Workflow_Advanced): 4. Raindrop trajectories and size sorting |
| `estimate_kdp` | `radarx.retrieve.kdp` | [core](Radar_Workflow): 5. ΦDP processing and KDP |
| `trajectories` | `radarx.retrieve.lagrangian` | [part 3](Radar_Workflow_Advanced_2): 9. Diabatic Lagrangian analysis |
| `cell_flash_rate` | `radarx.retrieve.lightning` | [part 2](Radar_Workflow_Advanced): 6. Lightning mapping |
| `cluster_flashes` | `radarx.retrieve.lightning` | [part 2](Radar_Workflow_Advanced): 6. Lightning mapping |
| `grid_lightning` | `radarx.retrieve.lightning` | [part 2](Radar_Workflow_Advanced): 6. Lightning mapping |
| `lightning_jump` | `radarx.retrieve.lightning` | [part 2](Radar_Workflow_Advanced): 6. Lightning mapping |
| `vertical_source_distribution` | `radarx.retrieve.lightning` | [part 2](Radar_Workflow_Advanced): 6. Lightning mapping |
| `dual_doppler_analytic` | `radarx.retrieve.multidoppler` | [core](Radar_Workflow): 12. Multi-radar grid and three-dimensional wind |
| `fall_speed` | `radarx.retrieve.multidoppler` | [part 3](Radar_Workflow_Advanced_2): 9. Diabatic Lagrangian analysis |
| `multi_doppler` | `radarx.retrieve.multidoppler` | [core](Radar_Workflow): 12. Multi-radar grid and three-dimensional wind |
| `multi_doppler_input` | `radarx.retrieve.multidoppler` | [core](Radar_Workflow): 12. Multi-radar grid and three-dimensional wind |
| `radar_geometry` | `radarx.retrieve.multidoppler` | [core](Radar_Workflow): 12. Multi-radar grid and three-dimensional wind |
| `noise_level` | `radarx.retrieve.noise` | [core](Radar_Workflow): 15. Noise and calibration |
| `snr` | `radarx.retrieve.noise` | [core](Radar_Workflow): 15. Noise and calibration |
| `correct_rhohv_noise` | `radarx.retrieve.noise` | [core](Radar_Workflow): 15. Noise and calibration |
| `correct_zdr_noise` | `radarx.retrieve.noise` | [core](Radar_Workflow): 15. Noise and calibration |
| `apply_mask` | `radarx.retrieve.qc` | [core](Radar_Workflow): 2. Remove non-meteorological echo |
| `echo_mask` | `radarx.retrieve.qc` | [core](Radar_Workflow): 2. Remove non-meteorological echo |
| `rain_rate` | `radarx.retrieve.qpe` | [core](Radar_Workflow): 7. Rain drop size distribution, Rain rate |
| `rain_rate_a` | `radarx.retrieve.qpe` | [core](Radar_Workflow): 7. Rain drop size distribution, Rain rate |
| `rain_rate_kdp` | `radarx.retrieve.qpe` | [core](Radar_Workflow): 7. Rain drop size distribution, Rain rate |
| `rain_rate_z` | `radarx.retrieve.qpe` | [core](Radar_Workflow): 7. Rain drop size distribution, Rain rate |
| `rain_rate_z_zdr` | `radarx.retrieve.qpe` | [core](Radar_Workflow): 7. Rain drop size distribution, Rain rate |
| `rain_shaft` | `radarx.retrieve.rain_collisions` | [part 2](Radar_Workflow_Advanced): 4. Raindrop trajectories and size sorting |
| `rain_source_points` | `radarx.retrieve.rain_trajectories` | [part 2](Radar_Workflow_Advanced): 4. Raindrop trajectories and size sorting |
| `rain_trajectories` | `radarx.retrieve.rain_trajectories` | [part 2](Radar_Workflow_Advanced): 4. Raindrop trajectories and size sorting |
| `size_sorting` | `radarx.retrieve.rain_trajectories` | [part 2](Radar_Workflow_Advanced): 4. Raindrop trajectories and size sorting |
| `surface_dsd` | `radarx.retrieve.rain_trajectories` | [part 2](Radar_Workflow_Advanced): 4. Raindrop trajectories and size sorting |
| `trajectory_matched_times` | `radarx.retrieve.rain_trajectories` | [part 2](Radar_Workflow_Advanced): 4. Raindrop trajectories and size sorting |
| `attenuation_coefficients` | `radarx.retrieve.attenuation` | [core](Radar_Workflow): 5. ΦDP processing and KDP |
| `azimuthal_shear` | `radarx.retrieve.shear` | [core](Radar_Workflow): 9. Azimuthal shear and radial divergence |
| `azimuthal_shear_layer_max` | `radarx.retrieve.shear` | [core](Radar_Workflow): 9. Azimuthal shear and radial divergence |
| `llsd` | `radarx.retrieve.shear` | [core](Radar_Workflow): 9. Azimuthal shear and radial divergence |
| `radial_divergence` | `radarx.retrieve.shear` | [core](Radar_Workflow): 9. Azimuthal shear and radial divergence |
| `single_doppler_winds` | `radarx.retrieve.single_doppler` | [part 3](Radar_Workflow_Advanced_2): 8. Wind from a single Doppler radar |
| `gbvtd` | `radarx.retrieve.tc_winds` | [part 3](Radar_Workflow_Advanced_2): 13. Tropical-cyclone winds from one radar |
| `synthetic_vortex` | `radarx.retrieve.tc_winds` | [part 3](Radar_Workflow_Advanced_2): 13. Tropical-cyclone winds from one radar |
| `tc_center` | `radarx.retrieve.tc_winds` | [part 3](Radar_Workflow_Advanced_2): 13. Tropical-cyclone winds from one radar |
| `tc_decompose` | `radarx.retrieve.tc_winds` | [part 3](Radar_Workflow_Advanced_2): 13. Tropical-cyclone winds from one radar |
| `vortex_profile` | `radarx.retrieve.tc_winds` | [part 3](Radar_Workflow_Advanced_2): 13. Tropical-cyclone winds from one radar |
| `superobs` | `radarx.retrieve.superobs` | [core](Radar_Workflow): 16. Super-observations for data assimilation |
| `rotation_couplets` | `radarx.retrieve.tornado` | [part 3](Radar_Workflow_Advanced_2): 7. Tornado detection and biological echo |
| `tornado_probability` | `radarx.retrieve.tornado` | [part 3](Radar_Workflow_Advanced_2): 7. Tornado detection and biological echo |
| `tornet_inputs` | `radarx.retrieve.tornado` | [part 3](Radar_Workflow_Advanced_2): 7. Tornado detection and biological echo |
| `cvp` | `radarx.retrieve.vertical_profiles` | [core](Radar_Workflow): 8. QVP time series and melting layer |
| `mask_above_melting_layer` | `radarx.retrieve.vertical_profiles` | [core](Radar_Workflow): 8. QVP time series and melting layer |
| `melting_layer` | `radarx.retrieve.vertical_profiles` | [core](Radar_Workflow): 8. QVP time series and melting layer |
| `melting_layer_ppi` | `radarx.retrieve.vertical_profiles` | [core](Radar_Workflow): 8. QVP time series and melting layer |
| `qvp` | `radarx.retrieve.vertical_profiles` | [core](Radar_Workflow): 8. QVP time series and melting layer |
| `qvp_timeseries` | `radarx.retrieve.vertical_profiles` | [core](Radar_Workflow): 8. QVP time series and melting layer |
| `rd_qvp` | `radarx.retrieve.vertical_profiles` | [core](Radar_Workflow): 8. QVP time series and melting layer |
| `rhi_qvp` | `radarx.retrieve.vertical_profiles` | [core](Radar_Workflow): 8. QVP time series and melting layer |
| `echo_top` | `radarx.retrieve.vil` | [core](Radar_Workflow): 14. VIL, echo tops and water content |
| `liquid_water_content` | `radarx.retrieve.vil` | [core](Radar_Workflow): 14. VIL, echo tops and water content |
| `vil` | `radarx.retrieve.vil` | [core](Radar_Workflow): 14. VIL, echo tops and water content |
| `vil_density` | `radarx.retrieve.vil` | [core](Radar_Workflow): 14. VIL, echo tops and water content |
| `bulk_shear` | `radarx.retrieve.wind_profile` | [core](Radar_Workflow): Sounding utilities and wind-profile parameters |
| `bunkers_storm_motion` | `radarx.retrieve.wind_profile` | [core](Radar_Workflow): Sounding utilities and wind-profile parameters |
| `layer_mean_wind` | `radarx.retrieve.wind_profile` | [core](Radar_Workflow): Sounding utilities and wind-profile parameters |
| `storm_relative_helicity` | `radarx.retrieve.wind_profile` | [core](Radar_Workflow): Sounding utilities and wind-profile parameters |
| `storm_relative_wind` | `radarx.retrieve.wind_profile` | [core](Radar_Workflow): Sounding utilities and wind-profile parameters |
| `vad_profile` | `radarx.retrieve.wind_profile` | [core](Radar_Workflow): 4. Dealias the Doppler velocity |
| `zdr_arc` | `radarx.retrieve.zdr_arc` | [part 3](Radar_Workflow_Advanced_2): 18. ZDR arc and KDP-ZDR separation |
| `hvplot_cappi` | `radarx.vis.interactive` | [core](Radar_Workflow): Other ways to grid, section and plot |
| `hvplot_centroids` | `radarx.vis.interactive` | [core](Radar_Workflow): Other ways to grid, section and plot |
| `hvplot_max_cappi` | `radarx.vis.interactive` | [core](Radar_Workflow): Other ways to grid, section and plot |
| `hvplot_mesh` | `radarx.vis.interactive` | [core](Radar_Workflow): Other ways to grid, section and plot |
| `hvplot_ppi` | `radarx.vis.interactive` | [core](Radar_Workflow): Other ways to grid, section and plot |
| `hvplot_range_azimuth` | `radarx.vis.interactive` | [core](Radar_Workflow): Other ways to grid, section and plot |
| `hvplot_rhi` | `radarx.vis.interactive` | [core](Radar_Workflow): Other ways to grid, section and plot |
| `RadarxDataArrayPlotAccessor` | `radarx.vis.interactive` | [core](Radar_Workflow): Other ways to grid, section and plot |
| `RadarxDatasetPlotAccessor` | `radarx.vis.interactive` | [core](Radar_Workflow): Other ways to grid, section and plot |
| `RadarxDataTreePlotAccessor` | `radarx.vis.interactive` | [core](Radar_Workflow): Other ways to grid, section and plot |
| `plot_maxcappi` | `radarx.vis.maxcappi` | [core](Radar_Workflow): Other ways to grid, section and plot |
| `plot_cappi` | `radarx.vis.plots` | [core](Radar_Workflow): Other ways to grid, section and plot |
| `plot_ppi` | `radarx.vis.plots` | [core](Radar_Workflow): Other ways to grid, section and plot |
| `plot_rhi` | `radarx.vis.plots` | [core](Radar_Workflow): Other ways to grid, section and plot |

The methods of the `.radarx` accessors:

| accessor method | section |
|---|---|
| `.radarx.advect` | [core](Radar_Workflow): 11. Storm motion, common analysis time and time interpolation |
| `.radarx.apply_mask` | [core](Radar_Workflow): 2. Remove non-meteorological echo |
| `.radarx.assign` | [core](Radar_Workflow): 4. Dealias the Doppler velocity |
| `.radarx.azimuthal_shear` | [core](Radar_Workflow): 9. Azimuthal shear and radial divergence |
| `.radarx.azimuthal_shear_layer_max` | [core](Radar_Workflow): 9. Azimuthal shear and radial divergence |
| `.radarx.background` | [core](Radar_Workflow): 12. Multi-radar grid and three-dimensional wind |
| `.radarx.baroclinic_generation` | [part 2](Radar_Workflow_Advanced): 5. Surface stations, profilers and cold pools |
| `.radarx.beam_blockage` | [part 3](Radar_Workflow_Advanced_2): 11. Beam blockage and terrain clutter |
| `.radarx.beam_path` | [part 2](Radar_Workflow_Advanced): 1. Radar equation, beam geometry, Doppler and unit helpers |
| `.radarx.biological_echo` | [part 3](Radar_Workflow_Advanced_2): 7. Tornado detection and biological echo |
| `.radarx.bulk_shear` | [core](Radar_Workflow): Sounding utilities and wind-profile parameters |
| `.radarx.bunkers_storm_motion` | [core](Radar_Workflow): Sounding utilities and wind-profile parameters |
| `.radarx.cluster_flashes` | [part 2](Radar_Workflow_Advanced): 6. Lightning mapping |
| `.radarx.cold_pool_intensity` | [part 2](Radar_Workflow_Advanced): 5. Surface stations, profilers and cold pools |
| `.radarx.correct_attenuation` | [core](Radar_Workflow): 5. ΦDP processing and KDP |
| `.radarx.correct_dual_prf` | [part 3](Radar_Workflow_Advanced_2): 15. Dual-PRF velocity correction |
| `.radarx.correct_rhohv_noise` | [core](Radar_Workflow): 15. Noise and calibration |
| `.radarx.correct_zdr_noise` | [core](Radar_Workflow): 15. Noise and calibration |
| `.radarx.create_cappi` | [core](Radar_Workflow): Other ways to grid, section and plot |
| `.radarx.cvp` | [core](Radar_Workflow): 8. QVP time series and melting layer |
| `.radarx.dealias` | [core](Radar_Workflow): 4. Dealias the Doppler velocity |
| `.radarx.dealias_ambiguity` | [core](Radar_Workflow): 4. Dealias the Doppler velocity |
| `.radarx.dealias_confidence` | [core](Radar_Workflow): 4. Dealias the Doppler velocity |
| `.radarx.dealias_r2d2` | [core](Radar_Workflow): 4. Dealias the Doppler velocity |
| `.radarx.diabatic_lagrangian` | [part 3](Radar_Workflow_Advanced_2): 9. Diabatic Lagrangian analysis |
| `.radarx.disdrometer` | [part 2](Radar_Workflow_Advanced): 2. Disdrometers |
| `.radarx.dsd` | [core](Radar_Workflow): 7. Rain drop size distribution |
| `.radarx.dsd_bayesian` | [core](Radar_Workflow): 7. Rain drop size distribution |
| `.radarx.dual_doppler_analytic` | [core](Radar_Workflow): 12. Multi-radar grid and three-dimensional wind |
| `.radarx.echo_mask` | [core](Radar_Workflow): 2. Remove non-meteorological echo |
| `.radarx.echo_top` | [core](Radar_Workflow): 14. VIL, echo tops and water content |
| `.radarx.ensemble_retrieval` | [part 3](Radar_Workflow_Advanced_2): 16. Ensembles and uncertainty of the retrievals |
| `.radarx.ensemble_statistics` | [part 3](Radar_Workflow_Advanced_2): 16. Ensembles and uncertainty of the retrievals |
| `.radarx.estimate_motion` | [core](Radar_Workflow): 11. Storm motion, common analysis time and time interpolation |
| `.radarx.evaporation` | [core](Radar_Workflow): 13. Evaporation |
| `.radarx.forward_operator` | [part 3](Radar_Workflow_Advanced_2): 14. Forward operator for WRF and MPAS output |
| `.radarx.gbvtd` | [part 3](Radar_Workflow_Advanced_2): 13. Tropical-cyclone winds from one radar |
| `.radarx.grid_gates` | [core](Radar_Workflow): Other ways to grid, section and plot |
| `.radarx.grid_lightning` | [part 2](Radar_Workflow_Advanced): 6. Lightning mapping |
| `.radarx.grid_radars` | [core](Radar_Workflow): 12. Multi-radar grid and three-dimensional wind |
| `.radarx.grid_rhi` | [core](Radar_Workflow): Other ways to grid, section and plot |
| `.radarx.hail_fall_speed` | [part 2](Radar_Workflow_Advanced): 4. Raindrop trajectories and size sorting |
| `.radarx.hail_kinetic_energy` | [core](Radar_Workflow): 14. VIL, echo tops and water content |
| `.radarx.hail_products` | [core](Radar_Workflow): 14. VIL, echo tops and water content |
| `.radarx.hid` | [core](Radar_Workflow): 6. Hydrometeor classification |
| `.radarx.integrate_evaporation` | [core](Radar_Workflow): 13. Evaporation |
| `.radarx.interpolate_profile` | [core](Radar_Workflow): 12. Multi-radar grid and three-dimensional wind |
| `.radarx.interpolate_time` | [core](Radar_Workflow): 11. Storm motion, common analysis time and time interpolation |
| `.radarx.kdp` | [core](Radar_Workflow): 5. ΦDP processing and KDP |
| `.radarx.lightning_jump` | [part 2](Radar_Workflow_Advanced): 6. Lightning mapping |
| `.radarx.liquid_water_content` | [core](Radar_Workflow): 14. VIL, echo tops and water content |
| `.radarx.llsd` | [core](Radar_Workflow): 9. Azimuthal shear and radial divergence |
| `.radarx.mask_above_melting_layer` | [core](Radar_Workflow): 8. QVP time series and melting layer |
| `.radarx.measurement_error` | [part 3](Radar_Workflow_Advanced_2): 16. Ensembles and uncertainty of the retrievals |
| `.radarx.melting_layer` | [core](Radar_Workflow): 8. QVP time series and melting layer |
| `.radarx.melting_layer_ppi` | [core](Radar_Workflow): 8. QVP time series and melting layer |
| `.radarx.melting_snow_fall_speed` | [part 2](Radar_Workflow_Advanced): 4. Raindrop trajectories and size sorting |
| `.radarx.merge_radars` | [core](Radar_Workflow): 12. Multi-radar grid and three-dimensional wind |
| `.radarx.mesh` | [core](Radar_Workflow): 14. VIL, echo tops and water content |
| `.radarx.multi_doppler` | [core](Radar_Workflow): 12. Multi-radar grid and three-dimensional wind |
| `.radarx.network_bias` | [core](Radar_Workflow): 12. Multi-radar grid and three-dimensional wind |
| `.radarx.noise_level` | [core](Radar_Workflow): 15. Noise and calibration |
| `.radarx.parcel_diagnostics` | [core](Radar_Workflow): Parcel thermodynamics: CAPE, CIN, the lifted index and the levels |
| `.radarx.perturb_radar` | [part 3](Radar_Workflow_Advanced_2): 16. Ensembles and uncertainty of the retrievals |
| `.radarx.phidp_offset` | [core](Radar_Workflow): 15. Noise and calibration |
| `.radarx.plot` | [core](Radar_Workflow): 10. Cone gridding, Max-CAPPI, interactive view and UGRID |
| `.radarx.plot_cappi` | [core](Radar_Workflow): Other ways to grid, section and plot |
| `.radarx.plot_max_cappi` | [core](Radar_Workflow): 10. Cone gridding, Max-CAPPI, interactive view and UGRID |
| `.radarx.plot_ppi` | [core](Radar_Workflow): Other ways to grid, section and plot |
| `.radarx.plot_rhi` | [core](Radar_Workflow): Other ways to grid, section and plot |
| `.radarx.posh` | [core](Radar_Workflow): 14. VIL, echo tops and water content |
| `.radarx.potential_temperatures` | [part 2](Radar_Workflow_Advanced): 5. Surface stations, profilers and cold pools |
| `.radarx.qvp` | [core](Radar_Workflow): 8. QVP time series and melting layer |
| `.radarx.radial_divergence` | [core](Radar_Workflow): 9. Azimuthal shear and radial divergence |
| `.radarx.rain_rate` | [core](Radar_Workflow): 7. Rain drop size distribution, Rain rate |
| `.radarx.rain_rate_a` | [core](Radar_Workflow): 7. Rain drop size distribution, Rain rate |
| `.radarx.rain_rate_kdp` | [core](Radar_Workflow): 7. Rain drop size distribution, Rain rate |
| `.radarx.rain_rate_z` | [core](Radar_Workflow): 7. Rain drop size distribution, Rain rate |
| `.radarx.rain_rate_z_zdr` | [core](Radar_Workflow): 7. Rain drop size distribution, Rain rate |
| `.radarx.rain_shaft` | [part 2](Radar_Workflow_Advanced): 4. Raindrop trajectories and size sorting |
| `.radarx.rain_trajectories` | [part 2](Radar_Workflow_Advanced): 4. Raindrop trajectories and size sorting |
| `.radarx.rd_qvp` | [core](Radar_Workflow): 8. QVP time series and melting layer |
| `.radarx.rhi_qvp` | [core](Radar_Workflow): 8. QVP time series and melting layer |
| `.radarx.rotation_couplets` | [part 3](Radar_Workflow_Advanced_2): 7. Tornado detection and biological echo |
| `.radarx.shi` | [core](Radar_Workflow): 14. VIL, echo tops and water content |
| `.radarx.single_doppler_winds` | [part 3](Radar_Workflow_Advanced_2): 8. Wind from a single Doppler radar |
| `.radarx.snr` | [core](Radar_Workflow): 15. Noise and calibration |
| `.radarx.sounding` | [core](Radar_Workflow): 3. Environment |
| `.radarx.storm_relative_helicity` | [core](Radar_Workflow): Sounding utilities and wind-profile parameters |
| `.radarx.superobs` | [core](Radar_Workflow): 16. Super-observations for data assimilation |
| `.radarx.to_cappi` | [core](Radar_Workflow): Other ways to grid, section and plot |
| `.radarx.to_cfradial1` | [part 3](Radar_Workflow_Advanced_2): 12. Export to files |
| `.radarx.to_cfradial2` | [part 3](Radar_Workflow_Advanced_2): 12. Export to files |
| `.radarx.to_grid` | [core](Radar_Workflow): 10. Cone gridding, Max-CAPPI, interactive view and UGRID |
| `.radarx.to_netcdf_cf` | [part 3](Radar_Workflow_Advanced_2): 12. Export to files |
| `.radarx.to_odim` | [part 3](Radar_Workflow_Advanced_2): 12. Export to files |
| `.radarx.to_radar_grid` | [part 3](Radar_Workflow_Advanced_2): 14. Forward operator for WRF and MPAS output |
| `.radarx.to_uxarray` | [core](Radar_Workflow): 10. Cone gridding, Max-CAPPI, interactive view and UGRID |
| `.radarx.to_zarr` | [part 3](Radar_Workflow_Advanced_2): 12. Export to files |
| `.radarx.tc_center` | [part 3](Radar_Workflow_Advanced_2): 13. Tropical-cyclone winds from one radar |
| `.radarx.tornado_probability` | [part 3](Radar_Workflow_Advanced_2): 7. Tornado detection and biological echo |
| `.radarx.trajectories` | [part 3](Radar_Workflow_Advanced_2): 9. Diabatic Lagrangian analysis |
| `.radarx.vad_profile` | [core](Radar_Workflow): 4. Dealias the Doppler velocity |
| `.radarx.vil` | [core](Radar_Workflow): 14. VIL, echo tops and water content |
| `.radarx.vil_density` | [core](Radar_Workflow): 14. VIL, echo tops and water content |
| `.radarx.z_offset` | [core](Radar_Workflow): 15. Noise and calibration |
| `.radarx.zdr_arc` | [part 3](Radar_Workflow_Advanced_2): 18. ZDR arc and KDP-ZDR separation |
| `.radarx.zdr_column` | [part 3](Radar_Workflow_Advanced_2): 17. ZDR columns |
| `.radarx.zdr_column_metrics` | [part 3](Radar_Workflow_Advanced_2): 17. ZDR columns |
| `.radarx.zdr_offset` | [core](Radar_Workflow): 15. Noise and calibration |

## References

- Dawson, D., M. Biggerstaff, and S. Waugh, 2025: PERiLS_2022: Portable In Situ Precipitation Stations (PIPS) Data. Version 1.0. NSF NCAR Earth Observing Laboratory, https://doi.org/10.26023/HFBG-7W5M-WA00.
- Kosiba, K. A., and Coauthors, 2024: The Propagation, Evolution, and Rotation in Linear Storms (PERiLS) Project. Bull. Amer. Meteor. Soc., 105, E1768-E1799, https://doi.org/10.1175/BAMS-D-22-0064.1.

- Amburn, S. A., and P. L. Wolf, 1997: VIL Density as a Hail Indicator. *Weather and Forecasting*, **12**, 473-478, <https://doi.org/10.1175/1520-0434(1997)012<0473:VDAAHI>2.0.CO;2>
- Browning, K. A., and R. Wexler, 1968: The Determination of Kinematic Properties of a Wind Field Using Doppler Radar. *Journal of Applied Meteorology*, **7**, 105-113, <https://doi.org/10.1175/1520-0450(1968)007<0105:TDOKPO>2.0.CO;2>
- Bunkers, M. J., B. A. Klimowski, J. W. Zeitler, R. L. Thompson, and M. L. Weisman, 2000: Predicting Supercell Motion Using a New Hodograph Technique. *Weather and Forecasting*, **15**, 61-79, <https://doi.org/10.1175/1520-0434(2000)015<0061:PSMUAN>2.0.CO;2>
- Cao, Q., G. Zhang, E. Brandes, T. Schuur, A. Ryzhkov, and K. Ikeda, 2008: Analysis of Video Disdrometer and Polarimetric Radar Data to Characterize Rain Microphysics in Oklahoma. *Journal of Applied Meteorology and Climatology*, **47**, 2238-2255, <https://doi.org/10.1175/2008JAMC1732.1>
- Gao, J., M. Xue, A. Shapiro, and K. K. Droegemeier, 1999: A Variational Method for the Analysis of Three-Dimensional Wind Fields from Two Doppler Radars. *Monthly Weather Review*, **127**, 2128-2142, <https://doi.org/10.1175/1520-0493(1999)127<2128:AVMFTA>2.0.CO;2>
- Gorgucci, E., G. Scarchilli, and V. Chandrasekar, 1992: Calibration of Radars Using Polarimetric Techniques. *IEEE Transactions on Geoscience and Remote Sensing*, **30**, 853-858, <https://doi.org/10.1109/36.175319>
- Gourley, J. J., P. Tabary, and J. Parent du Chatelet, 2006: Data Quality of the Meteo-France C-Band Polarimetric Radar. *Journal of Atmospheric and Oceanic Technology*, **23**, 1340-1356, <https://doi.org/10.1175/JTECH1912.1>
- Gourley, J. J., P. Tabary, and J. Parent du Chatelet, 2007: A Fuzzy Logic Algorithm for the Separation of Precipitating from Nonprecipitating Echoes Using Polarimetric Radar Observations. *Journal of Atmospheric and Oceanic Technology*, **24**, 1439-1451, <https://doi.org/10.1175/JTECH2035.1>
- Giangrande, S. E., and A. V. Ryzhkov, 2008: Estimation of Rainfall Based on the Results of Polarimetric Echo Classification. *Journal of Applied Meteorology and Climatology*, **47**, 2445-2462, <https://doi.org/10.1175/2008JAMC1753.1>
- Greene, D. R., and R. A. Clark, 1972: Vertically Integrated Liquid Water: A New Analysis Tool. *Monthly Weather Review*, **100**, 548-552, <https://doi.org/10.1175/1520-0493(1972)100<0548:VILWNA>2.3.CO;2>
- Hersbach, H., and Coauthors, 2020: The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, **146**, 1999-2049, <https://doi.org/10.1002/qj.3803>
- Hildebrand, P. H., and R. S. Sekhon, 1974: Objective Determination of the Noise Level in Doppler Spectra. *Journal of Applied Meteorology*, **13**, 808-811, <https://doi.org/10.1175/1520-0450(1974)013<0808:ODOTNL>2.0.CO;2>
- Krause, J. M., 2016: A Simple Algorithm to Discriminate between Meteorological and Nonmeteorological Radar Echoes. *Journal of Atmospheric and Oceanic Technology*, **33**, 1875-1885, <https://doi.org/10.1175/JTECH-D-15-0239.1>
- Kumjian, M. R., and A. V. Ryzhkov, 2010: The Impact of Evaporation on Polarimetric Characteristics of Rain: Theoretical Model and Practical Implications. *Journal of Applied Meteorology and Climatology*, **49**, 1247-1267, <https://doi.org/10.1175/2010JAMC2243.1>
- Lakshmanan, V., K. Hondl, C. K. Potvin, and D. Preignitz, 2013: An Improved Method for Estimating Radar Echo-Top Height. *Weather and Forecasting*, **28**, 481-488, <https://doi.org/10.1175/WAF-D-12-00084.1>
- Park, H. S., A. V. Ryzhkov, D. S. Zrnić, and K.-E. Kim, 2009: The Hydrometeor Classification Algorithm for the Polarimetric WSR-88D: Description and Application to an MCS. *Weather and Forecasting*, **24**, 730-748, <https://doi.org/10.1175/2008WAF2222205.1>
- Ryzhkov, A. V., S. E. Giangrande, V. M. Melnikov, and T. J. Schuur, 2005: Calibration Issues of Dual-Polarization Radar Measurements. *Journal of Atmospheric and Oceanic Technology*, **22**, 1138-1155, <https://doi.org/10.1175/JTECH1772.1>
- Ryzhkov, A., M. Diederich, P. Zhang, and C. Simmer, 2014: Potential Utilization of Specific Attenuation for Rainfall Estimation, Mitigation of Partial Beam Blockage, and Radar Networking. *Journal of Atmospheric and Oceanic Technology*, **31**, 599-619, <https://doi.org/10.1175/JTECH-D-13-00038.1>
- Seo, B.-C., W. F. Krajewski, and J. A. Smith, 2014: Four-dimensional reflectivity data comparison between two ground-based radars: methodology and statistical analysis. *Hydrological Sciences Journal*, **59**, 1320-1334, <https://doi.org/10.1080/02626667.2013.839872>
- Seo, B.-C., W. F. Krajewski, and Y. Qi, 2020: Utility of vertically integrated liquid water content for radar-rainfall estimation: Quality control and precipitation type classification. *Atmospheric Research*, **236**, 104800, <https://doi.org/10.1016/j.atmosres.2019.104800>
- Smith, T. M., and K. L. Elmore, 2004: The use of radial velocity derivative to diagnose rotation and divergence. *Preprints, 11th Conf. on Aviation, Range, and Aerospace Meteorology*, Hyannis, MA, Amer. Meteor. Soc., P5.6, <https://ams.confex.com/ams/pdfpapers/81827.pdf>
- Smith, T. M., V. Lakshmanan, G. J. Stumpf, K. L. Ortega, K. Hondl, K. Cooper, K. M. Calhoun, D. M. Kingfield, K. L. Manross, R. Toomey, and J. Brogden, 2016: Multi-Radar Multi-Sensor (MRMS) Severe Weather and Aviation Products: Initial Operating Capabilities. *Bulletin of the American Meteorological Society*, **97**, 1617-1630, <https://doi.org/10.1175/BAMS-D-14-00173.1>
- Zhang, G., J. Vivekanandan, and E. Brandes, 2001: A method for estimating rain rate and drop size distribution from polarimetric radar measurements. *IEEE Transactions on Geoscience and Remote Sensing*, **39**, 830-841, <https://doi.org/10.1109/36.917906>
- Zhang, J., K. Howard, and J. J. Gourley, 2005: Constructing Three-Dimensional Multiple-Radar Reflectivity Mosaics: Examples of Convective Storms and Stratiform Rain Echoes. *Journal of Atmospheric and Oceanic Technology*, **22**, 30-42, <https://doi.org/10.1175/JTECH-1689.1>
