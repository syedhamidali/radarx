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

# End-to-End Radar Workflow, Part 3: Real Radar Products, Models and Uncertainty

+++

This third part continues the [second part](Radar_Workflow_Advanced) (sections 1
to 6: the radar equation and unit helpers, disdrometers, the Bayesian DSD
retrieval, raindrop trajectories, surface stations and cold pools, lightning)
with the rest of radarx: tornado detection and biological echo on a NEXRAD
volume, single-Doppler winds, the diabatic Lagrangian analysis, machine-learning
plumbing, beam blockage, export to files, tropical-cyclone winds, the forward
operator for WRF and MPAS, dual-PRF correction and ensembles. It follows the same
style as the [core workflow](Radar_Workflow) and uses the same kind of inputs:
small open data sets or synthetic data with a known answer. The notebook is
split in parts so that each one runs in a short time and little memory; the
*Function index* at the end of the core workflow lists where each function is
shown.

| Section | What it shows | radarx calls |
|---|---|---|
| 7 | tornado detection and biological echo | `tornet_inputs`, `tornado_probability`, `biological_echo` |
| 8 | wind from a single Doppler radar | `single_doppler_winds`, `radar_geometry` |
| 9 | diabatic Lagrangian analysis | `trajectories`, `diabatic_lagrangian`, `microphysical_rates` |
| 10 | machine-learning plumbing | `radarx.ml` |
| 11 | beam blockage and terrain clutter from a digital elevation model | `beam_blockage` |
| 12 | export to CfRadial1, CfRadial2, ODIM_H5, Zarr and CF NetCDF | `write_cfradial1`, `write_cfradial2`, `write_odim`, `write_zarr`, `write_netcdf_cf` |
| 13 | tropical-cyclone winds from one radar: centre, GBVTD, decomposition, eyewall and ring diagnostics | `tc_center`, `gbvtd`, `tc_decompose`, `tc_ring_stats`, `tc_structure`, `synthetic_vortex`, `vortex_profile` |
| 14 | polarimetric forward operator for WRF and MPAS model output | `wrf_state`, `mpas_state`, `forward_operator`, `to_radar_grid` |
| 15 | dual-PRF velocity correction | `correct_dual_prf` |
| 16 | ensembles and uncertainty of the retrievals: errors of the measurements, calibration and DSD prior propagated by Monte Carlo | `perturb_radar`, `measurement_error`, `ensemble_retrieval`, `ensemble_statistics`, `relation_scatter` |
| 17 | ZDR column depth, area and volume above the freezing level | `zdr_column`, `zdr_column_metrics` |
| 18 | ZDR arc and KDP-ZDR separation of supercells on a grid | `zdr_arc` |
| 19 | volume matching of a ground radar and the GPM DPR, DPR drop size parameters, comparison with IMERG | `match_dpr`, `match_statistics`, `dpr_parallax`, `dpr_dsd`, `match_imerg` |
| 20 | linear wind, divergence and deformation from one radar in analysis volumes (VVP) | `vvp_winds` |
| 21 | KDP columns, KDP-ZDR size sorting, storm-cell identification, tracking and statistics | `kdp_column`, `kdp_column_metrics`, `storm_size_sorting`, `identify_cells`, `track_cells`, `cell_statistics` |
| 22 | Ka and W band cloud radar: gas, cloud liquid and rain attenuation, echo classification, dual-frequency ratio retrievals | `gas_specific_attenuation`, `cloud_liquid_attenuation`, `gas_path_attenuation`, `rain_attenuation_relation`, `rain_path_attenuation`, `classify_cloud_targets`, `dual_frequency_ratio`, `dfr_rain`, `dfr_snow` |
| 23 | Level I: IQ time series, clutter filtering, Doppler spectra and moments, dealiasing with the spectra | `read_iq`, `as_iq`, `simulate_iq`, `regression_filter`, `notch_filter`, `doppler_spectra`, `spectral_noise`, `spectral_moments`, `pulse_pair_moments`, `dealias_spectra` |
| 24 | nowcasting (extrapolation, S-PROG, STEPS), optical flow and verification scores | `nowcast`, `optical_flow`, `exceedance_probability`, `fss`, `crps`, `categorical_scores`, `brier_score`, `reliability`, `rank_histogram`, `nowcast_model` |
| 25 | rainfall in context: return period from precipitation frequency, areal reduction, climatological reference | `return_period`, `areal_reduction_factor`, `climatological_reference` |

```{code-cell} ipython3
import math
import tempfile
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cmweather  # noqa: F401  radar colormaps
import xradar as xd

import radarx  # noqa: F401  registers the .radarx accessors
import radarx.ml as ml
from radarx.ml import Model, PatchIndex
from radarx.fundamentals import (
    attenuation,
    beam,
    common,
    constants,
    doppler,
    geometry,
    power,
    principles,
    reflectivity,
    refraction,
    scattering,
    system,
    timing,
    variables,
)
from radarx.core import conversion
from radarx.io import (
    parsivel_classes,
    read_lma,
    read_mrr,
    read_parsivel,
    read_pips,
    read_pips_netcdf,
    read_sticknet,
    read_sticknet_locations,
    read_wind_profiler,
    refractivity,
    saturation_vapor_pressure,
    write_cfradial1,
    write_cfradial2,
    write_netcdf_cf,
    write_odim,
    write_zarr,
)
from radarx.retrieve import (
    azimuthal_shear,
    baroclinic_generation,
    beam_blockage,
    beam_path,
    biological_echo,
    bulk_shear,
    buoyancy,
    cell_flash_rate,
    cluster_flashes,
    cold_pool_intensity,
    cold_pool_intensity_from_pressure,
    cold_pool_intensity_from_surface,
    cold_pool_perturbation,
    correct_dual_prf,
    dealias_velocity,
    diabatic_lagrangian,
    disdrometer_qc,
    drop_evaporation_rate,
    dsd,
    dsd_bayesian,
    dsd_moments,
    dsd_prior,
    dsd_spectrum,
    echo_mask,
    ensemble_retrieval,
    ensemble_statistics,
    fall_speed,
    fit_gamma,
    fit_gamma_moments,
    forward_grid,
    forward_operator,
    gbvtd,
    grid_lightning,
    hail_fall_speed,
    lightning_jump,
    match_radar,
    measurement_error,
    melting_snow_fall_speed,
    microphysical_rates,
    mpas_state,
    number_concentration,
    parsivel_bins,
    perturb_radar,
    polarimetric_precipitation,
    potential_temperatures,
    process_disdrometer,
    radar_at_location,
    radar_from_dsd,
    radar_geometry,
    rain_shaft,
    rain_source_points,
    rain_trajectories,
    raupach_berne_correction,
    relation_scatter,
    rkw_ratio,
    rotation_couplets,
    scattering_table,
    single_doppler_winds,
    size_sorting,
    snr,
    surface_dsd,
    synthetic_vortex,
    tc_center,
    tc_decompose,
    tc_ring_stats,
    tc_structure,
    terminal_fall_speed,
    to_radar_grid,
    tornado_probability,
    tornet_inputs,
    trajectories,
    trajectory_matched_times,
    vertical_source_distribution,
    vortex_profile,
    vvp_winds,
    wrf_state,
    ziegler2013_precipitation,
    ziegler2013_profiles,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="The input coordinates to pcolormesh")

plt.rcParams.update(
    {
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 100,
        "savefig.dpi": 100,
        "font.size": 9.5,
    }
)
OKABE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00"}
work = Path(tempfile.mkdtemp())  # scratch folder for the small files written below
```

## 7. Tornado detection and biological echo

Two published convolutional networks run through ONNX Runtime without any
deep-learning framework: TorNet (Veillette et al. 2025) for tornado
detection and MistNet (Lin et al. 2019) for biological scatterers. Their
authors publish the weights under the MIT licence; radarx downloads them on
first use, checks their SHA-256 and converts them to ONNX (see the
[tornado notebook](Tornado_Detection)). The input is the KGWX volume of the
squall line, 23:46 UTC. `tornet_inputs` prepares the network inputs
(dealiased velocity, KDP and the polarimetric fields of the 0.5 and 0.9
degree tilts), `tornado_probability` runs the network, and
`rotation_couplets` finds the physical counterpart, compact maxima of the
linear least-squares derivative (LLSD) azimuthal shear from `azimuthal_shear`;
with `min_tilts` they must also be vertically continuous.

```{code-cell} ipython3
import gzip
import shutil

from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

from radarx.io.aws_data import download_file


def nexrad_volume(key):
    """A NEXRAD Level II volume from AWS with the Nyquist velocity of every sweep."""
    path = Path(download_file("unidata-nexrad-level2", key, str(work)))
    if path.suffix == ".gz":
        unzipped = path.with_suffix("")
        with gzip.open(path) as src, open(unzipped, "wb") as dst:
            shutil.copyfileobj(src, dst)
        path = unzipped
    with NEXRADLevel2File(str(path)) as nf:
        nyquist = [h["msg_31_data_header"]["RAD"]["nyquist_vel"] / 100.0 for h in nf.msg_31_data_header]
    dtree = xd.io.open_nexradlevel2_datatree(str(path))
    for name, value in zip([n for n in dtree.children if n.startswith("sweep")], nyquist):
        dtree[name] = dtree[name].to_dataset().assign_coords(nyquist_velocity=value)
    return dtree


kgwx = nexrad_volume("2022/03/30/KGWX/KGWX20220330_234639_V06")
inputs = tornet_inputs(kgwx, max_range=150e3)
tor = tornado_probability(inputs)
sweep = kgwx["sweep_1"].to_dataset(inherit="all_coords")
sweep["VRADH"] = dealias_velocity(sweep.assign(VRADH=sweep.VRADH.where(sweep.VRADH > -63.9)))
sweep = sweep.xradar.georeference()
shear = azimuthal_shear(sweep)
couplets = rotation_couplets(sweep, min_dbz=20.0)
tor_acc = kgwx.radarx.tornado_probability(max_range=150e3)  # the accessor of the volume runs the same network
couplets_acc = sweep.radarx.rotation_couplets(min_dbz=20.0)
print("accessors agree:", bool(np.allclose(tor_acc.tornado_probability, tor.tornado_probability, equal_nan=True)),
      couplets_acc.sizes["couplet"] == couplets.sizes["couplet"])
print(f"{tor.sizes['chip']} chips; highest tornado probability {float(tor.tornado_probability.max()):.2f}; "
      f"{couplets.sizes['couplet']} rotation couplets above 0.006 1/s")
couplets.isel(couplet=slice(0, 5)).to_dataframe()[["azimuth", "range", "azimuthal_shear", "delta_v"]].round(3)
```

A circulation that is real extends over several tilts. With `min_tilts`, the
chain of couplets through each couplet is followed up and down the Doppler cuts
of the dealiased volume (on the next elevation angle, the nearest couplet of the
same sense of rotation within `link_distance` of the peak, after the association
of shear features across elevation angles in the NSSL Mesocyclone Detection
Algorithm), and only the couplets on a chain of at least that many tilts are
kept. `n_tilts` says how long the chain is:

```{code-cell} ipython3
volume = kgwx.xradar.georeference().radarx.assign(kgwx.radarx.dealias("VRADH"))
every = rotation_couplets(volume)  # uses VRADH_dealiased where a sweep has it
continuous = rotation_couplets(volume, min_tilts=3)
n_every = sum(every[n].sizes["couplet"] for n in every.children)
n_kept = sum(continuous[n].sizes["couplet"] for n in continuous.children)
print(f"{n_every} couplets on {len(every.children)} Doppler cuts; {n_kept} are on a chain of at least 3 tilts")
deepest = max(continuous.children, key=lambda n: continuous[n].sizes["couplet"])
continuous[deepest].to_dataset()[["azimuthal_shear", "delta_v", "n_tilts"]].isel(couplet=slice(0, 4)).to_dataframe().round(3)
```

```{code-cell} ipython3
def polar_xy(ds):
    az = np.radians(ds.azimuth.values)[:, None]
    r = ds.range.values[None, :] / 1e3
    return r * np.sin(az), r * np.cos(az)


px, py = polar_xy(inputs)
fig, axes = plt.subplots(2, 2, figsize=(11, 10), layout="constrained")
panels = [
    ("reflectivity 0.5° (dBZ)", px, py, inputs.DBZ[..., 0], "turbo", (-10, 70)),
    ("dealiased velocity 0.5° (m/s)", px, py, inputs.VEL[..., 0], "RdBu_r", (-40, 40)),
    ("LLSD azimuthal shear (1/s) and couplets", sweep.x / 1e3, sweep.y / 1e3, shear, "RdBu_r", (-0.01, 0.01)),
    ("TorNet tornado probability", px, py, tor.tornado_probability, "magma", (0, 1)),
]
for ax, (name, gx_, gy_, da, cmap, (lo, hi)) in zip(axes.flat, panels):
    pm = ax.pcolormesh(gx_, gy_, da, cmap=cmap, vmin=lo, vmax=hi)
    fig.colorbar(pm, ax=ax, shrink=0.8)
    ax.set(xlim=(-150, 150), ylim=(-150, 150), aspect="equal", title=name)
axes[0, 0].set_ylabel("north of KGWX (km)")
axes[1, 0].set_ylabel("north of KGWX (km)")
axes[1, 0].set_xlabel("east of KGWX (km)")
axes[1, 1].set_xlabel("east of KGWX (km)")
axes[1, 0].plot(couplets.x / 1e3, couplets.y / 1e3, "ko", mfc="none", ms=6, mew=0.8)
plt.show()
```

`biological_echo` runs MistNet on the reflectivity, velocity and spectrum
width of the lowest five tilts. In the squall line it agrees with the
polarimetric fuzzy-logic `echo_mask` on nearly all echo, and finds the weak
biological echo at the edge:

```{code-cell} ipython3
bio = biological_echo(kgwx)
bio_acc = kgwx.radarx.biological_echo()
print("accessor and function agree:",
      bool(np.allclose(bio_acc[list(bio.children)[0]].biology_probability, bio[list(bio.children)[0]].biology_probability, equal_nan=True)))
qc = echo_mask(kgwx)
name = list(bio.children)[0]
b, q = bio[name].to_dataset(), qc[name].to_dataset()
sw = kgwx[name].to_dataset(inherit="all_coords").xradar.georeference()
dbz = sw.DBZH.where(sw.DBZH > -32)
echo = dbz.notnull() & b.weather_probability.notnull() & (q.ECHO_CLASS > 0)
mist_weather = ~b.biological_echo & echo
qc_weather = q.METEO_MASK & echo
print(f"weather: MistNet {float(mist_weather.sum() / echo.sum()):.1%}, echo_mask "
      f"{float(qc_weather.sum() / echo.sum()):.1%}, agreement "
      f"{float((mist_weather == qc_weather).where(echo).mean()):.1%}")
from matplotlib.colors import ListedColormap

# 0: both call it weather, 1: only MistNet, 2: only echo_mask
verdict = xr.where(mist_weather & qc_weather, 0.0, xr.where(mist_weather, 1.0, xr.where(qc_weather, 2.0, 3.0)))
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
panels = [
    ("reflectivity (dBZ)", dbz, "turbo", (-10, 60)),
    ("MistNet biology probability", b.biology_probability.where(echo), "viridis", (0, 1)),
    ("MistNet against echo_mask", verdict.where(echo), ListedColormap(["#cfcfcf", OKABE["blue"], OKABE["orange"], OKABE["red"]]), (-0.5, 3.5)),
]
for ax, (label, da, cmap, (lo, hi)) in zip(axes, panels):
    pm = ax.pcolormesh(sw.x / 1e3, sw.y / 1e3, da, cmap=cmap, vmin=lo, vmax=hi)
    cb = fig.colorbar(pm, ax=ax, shrink=0.8)
    ax.set(xlim=(-150, 150), ylim=(-150, 150), aspect="equal", title=label, xlabel="east of KGWX (km)")
cb.set_ticks([0, 1, 2, 3])
cb.set_ticklabels(["both: weather", "MistNet only", "echo_mask only", "both: not weather"])
axes[0].set_ylabel("north of KGWX (km)")
plt.show()
```

## 8. Wind from a single Doppler radar

One Doppler radar measures only the wind component along its beams.
`single_doppler_winds` fills in the rest with the variational cost function of
the multi-Doppler analysis (Gao et al. 1999), the anelastic mass-continuity
equation and a background wind from a sounding or ERA5. `radar_geometry`
gives the beam azimuth and elevation of every radar at every grid cell. Here
a Beltrami flow (Shapiro 1993) with 5 m/s updrafts on a sheared background is
sampled by one virtual radar 50 km south-west of the domain centre (1 m/s
noise, out to 80 km).

```{code-cell} ipython3
def beltrami(x, y, z, wmax=5.0, lx=40e3, lz=10e3):
    k = l = 2 * np.pi / lx
    m = np.pi / lz
    lam_ = np.sqrt(k * k + l * l + m * m)
    kh2 = k * k + l * l
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    fu = -wmax / kh2 * (lam_ * l * np.cos(k * X) * np.sin(l * Y) * np.sin(m * Z)
                        + m * k * np.sin(k * X) * np.cos(l * Y) * np.cos(m * Z))
    fv = wmax / kh2 * (lam_ * k * np.sin(k * X) * np.cos(l * Y) * np.sin(m * Z)
                       - m * l * np.cos(k * X) * np.sin(l * Y) * np.cos(m * Z))
    fw = wmax * np.cos(k * X) * np.cos(l * Y) * np.sin(m * Z)
    rho = 1.2 * np.exp(-Z / 10e3)
    scale = 1.2 * np.exp(-z.mean() / 10e3) / rho
    return 5.0 + 1.5e-3 * Z + fu * scale, 3.0 + 0.5e-3 * Z + fv * scale, fw * scale, rho


gx = gy = np.arange(-40e3, 40e3 + 1, 2000.0)
gz = np.arange(0, 10e3 + 1, 1000.0)
u, v, w, rho = beltrami(gx, gy, gz)
dims, coords = ("z", "y", "x"), {"z": gz, "y": gy, "x": gx}
truth = xr.Dataset({"u": (dims, u), "v": (dims, v), "w": (dims, w)}, coords=coords)
rad = xr.Dataset({"radar_x": -35e3, "radar_y": -35e3, "radar_altitude": 0.0}, coords=coords)
rad = radar_geometry(rad.expand_dims("radar"))
el, az = np.radians(rad.elevation[0]), np.radians(rad.azimuth[0])
vr = np.cos(el) * (np.sin(az) * truth.u + np.cos(az) * truth.v) + np.sin(el) * truth.w
vr = vr + np.random.default_rng(0).normal(0.0, 1.0, vr.shape)
rad["VRADH"] = vr.where(np.hypot(rad.x + 35e3, rad.y + 35e3) < 80e3).expand_dims("radar")
height_ = truth.z.broadcast_like(truth.u).values
background = xr.Dataset(
    {"u": (dims, 5.0 + 1.5e-3 * height_), "v": (dims, 3.0 + 0.5e-3 * height_), "air_density": (dims, rho)},
    coords=coords,
)
wind_sd = single_doppler_winds(rad, background, fall_speed_correction=False)
wind_acc = rad.radarx.single_doppler_winds(background, fall_speed_correction=False)
print("accessor and function agree:", bool(np.allclose(wind_acc.u, wind_sd.u)))
seen = np.isfinite(rad.VRADH[0])
print(f"RMS radial-velocity residual: {float(np.sqrt((wind_sd.vr_residual ** 2).mean())):.2f} m/s")
for comp in "uvw":
    err = float(np.sqrt(((wind_sd[comp] - truth[comp]) ** 2).where(seen).mean()))
    err_bg = float(np.sqrt(((background.get(comp, 0.0) - truth[comp]) ** 2).where(seen).mean()))
    print(f"{comp}: RMS error {err:.2f} m/s (background {err_bg:.2f} m/s)")
```

With `uncertainty=True` the retrieval is repeated on inputs with random errors
(`vr_error` for the radial velocity, `background_error` for the background wind)
and `u_uncertainty`, `v_uncertainty` and `w_uncertainty` are the standard
deviations of the members. The weights below agree with the stated errors, and
the result records how the fall speed was handled:

```{code-cell} ipython3
wind_u = single_doppler_winds(
    rad, background, fall_speed_correction=False, uncertainty=True, members=6,
    vr_error=1.0, background_error=3.0, weights={"background": (1.0 / 3.0) ** 2},
)
for comp in "uvw":
    actual = float(np.sqrt(((wind_u[comp] - truth[comp]) ** 2).where(seen).mean()))
    spread = float(np.sqrt((wind_u[f"{comp}_uncertainty"] ** 2).where(seen).mean()))
    print(f"{comp}: actual RMS error {actual:.2f} m/s, RMS uncertainty {spread:.2f} m/s")
print(wind_u.attrs["fall_speed_correction"])
```

```{code-cell} ipython3
def km(da):
    return da.assign_coords(x=da.x / 1e3, y=da.y / 1e3)


fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
level = dict(z=5000.0)
kw = dict(cmap="RdBu_r", vmin=-6, vmax=6, add_colorbar=False)
km(truth.w.sel(**level)).plot(ax=axes[0], **kw)
im = km(wind_sd.w.sel(**level)).plot(ax=axes[1], **kw)
sub = km(wind_sd.sel(**level)).isel(x=slice(None, None, 3), y=slice(None, None, 3))
axes[1].quiver(sub.x, sub.y, sub.u, sub.v, scale=400)
im2 = km(rad.VRADH[0].sel(**level)).plot(ax=axes[2], cmap="RdBu_r", vmin=-30, vmax=30, add_colorbar=False)
for ax, title in zip(axes, ["true w at 5 km", "retrieved w and wind", "radial velocity"]):
    ax.set(title=title, aspect="equal", xlabel="x (km)", ylabel="y (km)")
    ax.plot(-35, -35, "k^", ms=8)
fig.colorbar(im, ax=axes[:2], label="w (m/s)", shrink=0.8)
fig.colorbar(im2, ax=axes[2], label="radial velocity (m/s)", shrink=0.8)
plt.show()
```

## 9. Diabatic Lagrangian analysis

The diabatic Lagrangian analysis (DLA; Ziegler 2013a, b) retrieves the
potential temperature, water vapour and cloud water, and so the buoyancy, of a
storm from a time series of three-dimensional multi-Doppler winds and radar
data. `trajectories` follows the air backward in time from every grid point
of the analysis time until it reaches the storm environment, and
`diabatic_lagrangian` integrates the thermodynamic state forward along the
trajectories with saturation adjustment, rain evaporation, cloud collection,
melting and sublimation (Lin et al. 1983). A small synthetic storm moving at
$(10, 5)$ m/s has an updraft core of 15 m/s next to a precipitation-filled
downdraft, reflectivity up to 55 dBZ and a moist boundary layer below 1.5 km.

```{code-cell} ipython3
cx_, cy_ = 10.0, 5.0
x = np.arange(-20e3, 20e3 + 1, 1000.0)
y = np.arange(-20e3, 20e3 + 1, 1000.0)
z = np.arange(0.0, 8001.0, 500.0)
tt = np.arange(16) * 180.0
T, Z, Y, X = np.meshgrid(tt, z, y, x, indexing="ij")
xs, ys = X - cx_ * (T - tt[-1]), Y - cy_ * (T - tt[-1])  # storm-relative position


def blob(x0, y0, r):
    return np.exp(-((xs - x0) ** 2 + (ys - y0) ** 2) / r**2)


w = 15.0 * blob(-3e3, 0.0, 4e3) * np.sin(np.pi * Z / 10e3) - 6.0 * blob(5e3, 0.0, 5e3) * np.sin(np.pi * Z / 8e3)
u = cx_ + 4.0 * blob(5e3, 0.0, 6e3) * (Z < 1500)
dbz = 10.0 + 45.0 * blob(4e3, 0.0, 7e3) * np.exp(-Z / 9e3)
zdr = np.clip(0.2 + (dbz - 20.0) / 15.0, 0.1, 3.5)
dims4 = ("time", "z", "y", "x")
dla_times = np.datetime64("2022-03-30T23:00", "ns") + (tt * 1e9).astype("timedelta64[ns]")
winds = xr.Dataset(
    {"u": (dims4, u), "v": (dims4, cy_ + 0.0 * X), "w": (dims4, w), "DBZ": (dims4, dbz), "ZDR": (dims4, zdr)},
    coords={"time": dla_times, "z": z, "y": y, "x": x},
)
hh = np.arange(0.0, 12001.0, 50.0)
temp = 300.0 - 0.0065 * hh
pressure = 1e5 * (temp / 300.0) ** (9.80665 / (287.04 * 0.0065))
env = xr.Dataset(
    {"pressure": ("height", pressure), "temperature": ("height", temp),
     "specific_humidity": ("height", np.where(hh < 1500.0, 0.014, 0.005)),
     "u": ("height", np.full(hh.size, cx_)), "v": ("height", np.full(hh.size, cy_))},
    coords={"height": hh},
)
tr = trajectories(winds, storm_motion=(cx_, cy_), levels=[0])
print(f"{float(tr.environment.mean()):.0%} of the trajectories from the ground reached the environment")
dla = diabatic_lagrangian(winds, env, storm_motion=(cx_, cy_))
tr_acc = winds.radarx.trajectories(storm_motion=(cx_, cy_), levels=[0])  # accessors of the winds
dla_acc = winds.radarx.diabatic_lagrangian(env, storm_motion=(cx_, cy_))
print("accessors agree:", bool(np.allclose(tr_acc.x, tr.x, equal_nan=True)),
      bool(np.allclose(dla_acc.theta, dla.theta, equal_nan=True)))
sfc = dla.isel(z=0)
sec = dla.sel(y=0.0)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
pc = axes[0].pcolormesh(x / 1e3, y / 1e3, sfc.delta_theta_v, cmap="RdBu_r", vmin=-4, vmax=4)
axes[0].contour(x / 1e3, y / 1e3, winds.DBZ.isel(time=-1, z=0), [30, 45], colors="k", linewidths=0.8)
pick = (tr.j % 6 == 0) & (tr.i % 6 == 0) & tr.environment
for k in np.flatnonzero(pick.values)[::2]:
    axes[0].plot(tr.x[k] / 1e3, tr.y[k] / 1e3, color="0.3", lw=0.6)
axes[0].set(xlabel="x (km)", ylabel="y (km)", title="surface, backward trajectories", aspect="equal")
fig.colorbar(pc, ax=axes[0], label=r"$\Delta\theta_v$ (K)")
pc = axes[1].pcolormesh(x / 1e3, z / 1e3, sec.delta_theta_v, cmap="RdBu_r", vmin=-8, vmax=8)
axes[1].contour(x / 1e3, z / 1e3, sec.qc * 1e3, [0.1, 1.0, 3.0], colors="k", linewidths=0.8)
axes[1].set(xlabel="x (km)", ylabel="height (km)", title="y = 0, contours: cloud water (g/kg)")
fig.colorbar(pc, ax=axes[1], label=r"$\Delta\theta_v$ (K)")
plt.show()
```

The cold pool under the downdraft and the buoyant cloudy updraft appear from
the winds and the reflectivity alone. The precipitation along the trajectories
comes from a closure. The default (`polarimetric_precipitation`) uses the DSD
retrieval from $Z_H$ and $Z_{DR}$, and `ziegler2013_precipitation` the
reflectivity-only closure of Ziegler (2013a) with the regression profiles of
`ziegler2013_profiles`. Both need the base state of the sounding (density and
the melting level):

```{code-cell} ipython3
rho_base = pressure / (287.04 * temp)
base = xr.Dataset(
    {"rho": ("z", np.interp(z, hh, rho_base))}, coords={"z": z},
    attrs={"ground_height": 0.0, "melting_level": (300.0 - 273.15) / 0.0065,
           "minus15_level": (300.0 - 258.15) / 0.0065},
)
last = winds.isel(time=[-1])
pol = polarimetric_precipitation(last, base)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    zig = ziegler2013_precipitation(last, base, profiles=ziegler2013_profiles())
profiles_cm1 = ziegler2013_profiles()
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
for ax, (name, field) in zip(axes[:2], [("polarimetric closure: rain (g/kg)", pol.qr), ("Ziegler (2013a) closure: rain (g/kg)", zig.qr)]):
    pm = ax.pcolormesh(x / 1e3, z / 1e3, field.isel(time=0).sel(y=0.0) * 1e3, cmap="Blues", vmin=0, vmax=8)
    ax.set(title=name, xlabel="x (km)", ylabel="height (km)")
fig.colorbar(pm, ax=axes[:2], label="g/kg", shrink=0.9)
for name_, color in (("Z0r", OKABE["blue"]), ("Z0g", OKABE["orange"])):
    axes[2].plot(profiles_cm1[name_].where(profiles_cm1[name_] < 90), profiles_cm1.z_star / 1e3, color=color,
                 label={"Z0r": "rain", "Z0g": "graupel"}[name_])
axes[2].set(title="regression profiles of the closure", xlabel="reflectivity scale (dBZ)", ylabel="$z^*$ (km)")
axes[2].legend(frameon=False)
plt.show()
```

`fall_speed` gives the reflectivity-weighted terminal fall speed of rain and
ice that the precipitation loading uses, and `microphysical_rates` the
Lin et al. (1983) rates and the resulting temperature and humidity tendencies
for a given state, here for rain of increasing mixing ratio below cloud base:

```{code-cell} ipython3
zc = xr.DataArray(np.arange(0.0, 8001.0, 250.0), dims="z", name="z")
zc = zc.assign_coords(z=zc)
column = 50.0 - 4.0 * zc / 1e3  # a reflectivity profile decreasing with height (dBZ)
density = xr.DataArray(np.interp(zc, hh, rho_base), dims="z", coords={"z": zc.values})
vt_plain = fall_speed(column, freezing_level=4000.0)
vt_dens = fall_speed(column, air_density=density, freezing_level=4000.0)
qr_axis = xr.DataArray(np.linspace(0.2e-3, 6e-3, 8), dims="qr")
rates = microphysical_rates(
    theta=300.0, pressure=90000.0, qv=0.011, qc=0.0, qr=qr_axis, nr=1e4 * xr.ones_like(qr_axis),
    qg=0.0, ng=0.0,
)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
ax1.plot(vt_plain, zc / 1e3, color=OKABE["blue"], label="no density correction")
ax1.plot(vt_dens, zc / 1e3, color=OKABE["orange"], label="with density correction")
ax1.axhline(4.0, color="0.4", ls=":")
ax1.set(xlabel="fall speed (m/s)", ylabel="height (km)")
ax1.legend(frameon=False, loc="upper right")
ax2.plot(qr_axis * 1e3, rates.dtheta_dt * 3600, color=OKABE["red"], label="$d\\theta/dt$ (K/h)")
ax2.set(xlabel="rain mixing ratio (g/kg)", ylabel="$d\\theta/dt$ (K/h)")
plt.show()
```

## 10. Machine-learning plumbing

`radarx.ml` runs trained networks, never trains them. A registry lists models
with their licence and citation (`list_models`, `register_model`,
`load_model`), `normalize` and `denormalize` scale the fields, `polar_patches`
cuts sweeps into tiles in radar coordinates (wrapping around north) and
`reassemble` puts model outputs back together with a blending window. A tiny
network built right here, a 5 by 5 box filter as an ONNX convolution, stands
in for a real one (it only shows how a model file is registered and run) and
the sweep is the real KGWX reflectivity of the tornado section.

```{code-cell} ipython3
import hashlib

import onnx
from onnx import TensorProto, helper

sweep_ml = sweep[["DBZH"]]  # the 0.9 degree KGWX sweep of the tornado section
sweep_ml = sweep_ml.assign(DBZH=sweep_ml.DBZH.where(sweep_ml.DBZH > -32.0))

k = 5
inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 1, None, None])
outp = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 1, None, None])
kernel = helper.make_tensor("w", TensorProto.FLOAT, [1, 1, k, k], [1 / k**2] * k**2)
graph = helper.make_graph([helper.make_node("Conv", ["x", "w"], ["y"], pads=[k // 2] * 4)],
                          "box-filter", [inp], [outp], [kernel])
onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
onnx_model.ir_version = 8
onnx.checker.check_model(onnx_model)
path = work / "box_filter.onnx"
onnx.save(onnx_model, path)
ml.register_model(
    "box-filter", path, hashlib.sha256(path.read_bytes()).hexdigest(), licence="MIT",
    citation="radarx documentation example (2026)", task="smoothing",
    inputs={"x": "float32[N,1,H,W] normalised reflectivity"},
    outputs={"y": "float32[N,1,H,W] smoothed"},
)
model = ml.load_model("box-filter")
print(model, "| a Model:", isinstance(model, Model))
print("models in the registry:", [m["name"] for m in ml.list_models()])
dbz_n = ml.normalize(sweep_ml.DBZH, offset=0.0, scale=60.0, fill_value=0.0)
patches, index = ml.polar_patches(dbz_n, (64, 128))
out = model.run({"x": patches[:, None]}, batch_size=64)["y"][:, 0]
smoothed = ml.denormalize(
    ml.reassemble(out, index, attrs=model.attrs).assign_attrs(dbz_n.attrs),
    mask=sweep_ml.DBZH.isnull(),  # the gates that were filled with 0 come back as NaN
)
print(f"{len(patches)} patches, array shape {patches.shape}; the PatchIndex remembers where each came from:",
      isinstance(index, PatchIndex))
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout="constrained")
for ax, (da, title, cmap, lim) in zip(
    axes,
    [(sweep_ml.DBZH, "KGWX reflectivity (dBZ)", "ChaseSpectral", (-10, 60)),
     (smoothed, "after the network (dBZ)", "ChaseSpectral", (-10, 60)),
     (smoothed - sweep_ml.DBZH, "difference (dB)", "RdBu_r", (-10, 10))],
):
    pm = ax.pcolormesh(sweep_ml.x / 1e3, sweep_ml.y / 1e3, da, cmap=cmap, vmin=lim[0], vmax=lim[1])
    fig.colorbar(pm, ax=ax, shrink=0.8)
    ax.set(title=title, aspect="equal", xlim=(-100, 100), ylim=(-100, 100), xlabel="east (km)")
axes[0].set_ylabel("north (km)")
plt.show()
```

## 11. Beam blockage and terrain clutter

`beam_blockage` takes a digital elevation model (an `xarray.DataArray` of
heights with `latitude` and `longitude`, read with `rioxarray` from SRTM or
Copernicus DEM tiles, which radarx does not download) and returns, for every
gate of a sweep or volume, the partial beam blockage `PBB` (the part of the
beam power below the terrain, Bech et al. 2003, with a Gaussian beam as in Lang
et al. 2009), the cumulative beam blockage `CBB` (the largest `PBB` along the
ray), the terrain height under the gate and a flag for gates whose beam centre
is at the terrain surface. The [beam blockage notebook](Beam_Blockage) tests it
against a numerical integration of the beam cross section. Here a ridge of
800 m, 25 km from the radar, blocks the lowest cut and the shadow stays
behind it.

```{code-cell} ipython3
site_lat, site_lon, site_alt = 30.0, 78.0, 300.0
glat = np.arange(site_lat - 1.0, site_lat + 1.0 + 0.001, 0.005)
glon = np.arange(site_lon - 1.2, site_lon + 1.2 + 0.001, 0.005)
la, lo = np.meshgrid(glat, glon, indexing="ij")
dist_km = np.hypot((la - site_lat) * 111.2, (lo - site_lon) * 111.2 * np.cos(np.deg2rad(site_lat)))
terrain = xr.DataArray(
    150.0 + 800.0 * np.cos(0.5 * np.pi * np.clip((dist_km - 25.0) / 4.0, -1.0, 1.0)) ** 2,
    dims=("latitude", "longitude"), coords={"latitude": glat, "longitude": glon},
)
nray, ngate = 180, 160
bb_sweep = xr.Dataset(
    {"DBZH": (("azimuth", "range"), np.full((nray, ngate), 30.0))},
    coords={"azimuth": (np.arange(nray) + 0.5) * 2.0, "range": 500.0 * (np.arange(ngate) + 1),
            "elevation": ("azimuth", np.full(nray, 0.5)),
            "latitude": site_lat, "longitude": site_lon, "altitude": site_alt},
)
blocked = beam_blockage(bb_sweep, terrain, beamwidth=1.0)  # PBB, CBB, TERRAIN_HEIGHT, TERRAIN_CLUTTER
blocked_acc = bb_sweep.radarx.beam_blockage(terrain, beamwidth=1.0, pattern="uniform")  # the accessor
ray0 = blocked.isel(azimuth=0)
print(f"ray at {float(ray0.azimuth):.0f} deg: PBB peaks at {float(ray0.PBB.max()):.2f}, "
      f"CBB behind the ridge {float(ray0.CBB[-1]):.2f}, "
      f"uniform beam {float(blocked_acc.CBB.isel(azimuth=0)[-1]):.2f}")
kept = bb_sweep.DBZH.where(blocked.CBB < 0.5)  # use CBB as a mask; nothing is applied for you
print(f"{int(kept.notnull().sum())} of {kept.size} gates kept")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
r_km = bb_sweep.range / 1e3
axes[0].fill_between(r_km, 0, ray0.TERRAIN_HEIGHT, color="0.6", label="terrain")
axes[0].plot(r_km, geometry.beam_center_height(bb_sweep.range, 0.5, radar_height=site_alt), label="beam centre")
axes[0].set(xlabel="range (km)", ylabel="height (m)", ylim=(0, 1500)); axes[0].legend()
axes[1].plot(r_km, ray0.PBB, label="PBB")
axes[1].plot(r_km, ray0.CBB, label="CBB")
axes[1].set(xlabel="range (km)", ylabel="blockage", ylim=(-0.02, 1.02)); axes[1].legend()
plt.show()
```

## 12. Export to files

`radarx.io` writes volumes and grids to files that other tools read, with the
provenance attributes (`radarx_method`, `radarx_references`, `radarx_version`,
`history`) kept where the format can hold them: `write_cfradial1`,
`write_cfradial2` and `write_odim` for a `DataTree` volume (through the writers
of xradar), `write_zarr` for any result and `write_netcdf_cf` for a grid, which
also gets the CF georeferencing (`grid_mapping`, `standard_name`, `axis`, 2-D
`lat` and `lon`, bounds). Each has a `.radarx.to_*` method. The
[Exporting results](Exporting_Results) page shows what every format keeps and
loses. Here a small NEXRAD volume with derived fields is written to each format.

```{code-cell} ipython3
from open_radar_data import DATASETS

raw_exp = xd.io.open_nexradlevel2_datatree(DATASETS.fetch("KLBB20160601_150025_V06"), sweep=[0, 2])
raw_exp = raw_exp.map_over_datasets(
    lambda ds: ds.sel(range=slice(None, 60e3)) if "range" in ds.dims else ds
).xradar.georeference()
vol_exp = raw_exp.radarx.assign(raw_exp.radarx.kdp()).radarx.assign(raw_exp.radarx.echo_mask())

write_zarr(vol_exp, work / "volume.zarr")
write_cfradial2(vol_exp, work / "volume_cfradial2.nc")
write_cfradial1(vol_exp, work / "volume_cfradial1.nc")
write_odim(vol_exp, work / "volume.h5", source="NOD:uskbb", data_vars=["DBZH", "ZDR", "KDP"])
vol_exp.radarx.to_zarr(work / "volume_accessor.zarr")
vol_exp.radarx.to_cfradial1(work / "volume_accessor_cf1.nc")
vol_exp.radarx.to_cfradial2(work / "volume_accessor_cf2.nc")
vol_exp.radarx.to_odim(work / "volume_accessor.h5", source="NOD:uskbb")

same = xr.open_datatree(work / "volume.zarr", engine="zarr")
xr.testing.assert_identical(same, vol_exp)
cf1_exp = xr.open_dataset(work / "volume_cfradial1.nc")
print("Zarr reads back identical;", "CfRadial1 holds", sorted(cf1_exp.attrs["radarx_method"].split(" | ")))
```

```{code-cell} ipython3
grid_exp = radarx.grid.grid_cones(
    vol_exp, "DBZH", x=np.arange(-40e3, 40e3 + 1, 2e3), y=np.arange(-40e3, 40e3 + 1, 2e3),
    z=np.arange(500.0, 4000.0, 500.0),
)
write_netcdf_cf(grid_exp, work / "grid.nc")
grid_exp.radarx.to_netcdf_cf(work / "grid_accessor.nc")
grid_exp.radarx.to_zarr(work / "grid.zarr")
with xr.open_dataset(work / "grid.nc") as cf_grid:
    print(cf_grid["DBZH"].attrs["grid_mapping"], cf_grid["lat"].dims, cf_grid.attrs["Conventions"])
```

## 13. Tropical-cyclone winds from one radar

`synthetic_vortex` forward-models a modified Rankine-combined vortex
(Lee et al. 1999) with a mean wind, a wavenumber-1 asymmetry and noise to the
Doppler velocity of one radar 70 km away, and `vortex_profile` gives its
axisymmetric truth. `tc_center` finds the circulation centre and the radius of
maximum wind (the GBVTD-simplex of Lee and Marks 2000 with the consensus of Bell
and Lee 2012), `gbvtd` retrieves the axisymmetric tangential and radial wind and the
wavenumber 1 to 3 asymmetries on rings around it (the ground-based velocity track
display, Lee et al. 1999) and `tc_decompose` splits the result into the symmetric
and asymmetric winds on a polar grid. The
[tropical-cyclone notebook](Tropical_Cyclone_Winds) tests the retrieval against
the vortex, reproduces the numbers of the three papers of the GBVTD series, shows
the sensitivity to the centre (1, 2 and 5 km) and tries the Tauktae (2021) volume
of the Goa radar, which is too far away to be retrieved.

```{code-cell} ipython3
tc_true = (0.0, 70000.0)  # centre, m east and north of the radar
tc_sweep = synthetic_vortex(
    center=tc_true, vmax=45.0, rmw=18000.0, xt=0.6, asymmetry={1: (0.2, 60.0)},
    mean_wind=(6.0, 0.0), noise=1.5, seed=2, azimuth_step=2.0, max_range=120000.0,
)
tc_found = tc_center(tc_sweep, min_radius=10000.0, max_radius=30000.0)
tc_acc = tc_sweep.radarx.tc_center(min_radius=10000.0, max_radius=30000.0)  # the accessor
tc_radii = np.arange(4000.0, 40001.0, 2000.0)
tc_rings = gbvtd(tc_sweep, tc_found, radii=tc_radii, mean_wind=(6.0, 0.0))
tc_rings_acc = tc_sweep.radarx.gbvtd(tc_true, radii=tc_radii, mean_wind=(6.0, 0.0))  # the accessor
tc_vt, _ = vortex_profile(tc_radii, 45.0, 18000.0, 0.6)
tc_polar = tc_decompose(tc_rings, azimuth=np.arange(0.0, 360.0, 10.0))
print(f"centre error {np.hypot(float(tc_found.CENTER_X) - tc_true[0], float(tc_found.CENTER_Y) - tc_true[1]):.0f} m, "
      f"RMW {float(tc_found.RMW) / 1e3:.1f} km (truth 18), maximum {float(tc_found.VT_MAX):.1f} m/s (truth 45)")
print(f"VT0 rms error {float(np.sqrt(np.nanmean((tc_rings.VT0 - tc_vt) ** 2))):.2f} m/s; "
      f"wavenumber-1 amplitude {float(tc_polar.AZIMUTHAL_AMPLITUDE.sel(wavenumber=1, radius=30000.0)):.1f} m/s "
      f"(truth {0.2 * float(tc_vt[13]):.1f}) towards {float(tc_polar.AZIMUTHAL_PHASE.sel(wavenumber=1, radius=30000.0)):.0f} deg (truth 60)")
print("accessors agree:", bool(np.isclose(float(tc_acc.CENTER_X), float(tc_found.CENTER_X))
                           and np.allclose(tc_rings_acc.VT0, gbvtd(tc_sweep, tc_true, radii=tc_radii, mean_wind=(6.0, 0.0)).VT0)))
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
axes[0].plot(tc_radii / 1e3, tc_vt, "k", label="truth")
axes[0].plot(tc_radii / 1e3, tc_rings.VT0, label="GBVTD")
axes[0].set(xlabel="radius (km)", ylabel="axisymmetric tangential wind (m/s)")
axes[0].legend()
im = axes[1].pcolormesh(tc_polar.X / 1e3, tc_polar.Y / 1e3, tc_polar.VT, cmap="viridis", shading="auto")
axes[1].set(aspect="equal", xlabel="east (km)", ylabel="north (km)", title="retrieved tangential wind")
fig.colorbar(im, ax=axes[1], label="m/s")
plt.show()
```

The azimuthal statistics of a field around the centre and the structure of its radial
profile follow: `tc_ring_stats` gives the mean and the wavenumbers of any polar field on
rings (here the true tangential wind of the synthetic vortex, in practice the
reflectivity), and `tc_structure` finds the eyewall maximum, its width and the
vorticity of the retrieved winds.

```{code-cell} ipython3
tc_field = tc_ring_stats(tc_sweep, tc_found, field="VT_TRUE", radii=tc_radii, ring_width=2000.0)
tc_eye = tc_structure(tc_rings, levels=[30.0])
tc_eye_acc = tc_rings.radarx.tc_structure(levels=[30.0])  # the accessor
tc_field_acc = tc_sweep.radarx.tc_ring_stats(tc_found, field="VT_TRUE", radii=tc_radii, ring_width=2000.0)  # the accessor
print(f"maximum {float(tc_eye.PEAK_VALUE):.1f} m/s at {float(tc_eye.PEAK_RADIUS) / 1e3:.1f} km (truth 45 at 18), "
      f"width {float(tc_eye.WIDTH) / 1e3:.1f} km, 30 m/s at {float(tc_eye.OUTER_LEVEL_RADIUS[0]) / 1e3:.0f} km; "
      f"vorticity inside the eyewall {float(tc_eye.VORTICITY.sel(radius=10000.0)) * 1e3:.1f} 1e-3 1/s (truth {2 * 45 / 18000 * 1e3:.1f})")
print("accessors agree:", bool(np.isclose(float(tc_eye_acc.PEAK_RADIUS), float(tc_eye.PEAK_RADIUS))
                           and np.allclose(tc_field_acc.MEAN.values, tc_field.MEAN.values, equal_nan=True)))
```

## 14. Forward operator for WRF and MPAS output

A forward operator turns the microphysics state of a numerical model into the
fields a radar measures, so that a model run is compared with the radar in radar
space. `wrf_state` and `mpas_state` read a WRF `wrfout` Dataset or an MPAS output
Dataset on its unstructured mesh into one standard state; `forward_operator`
computes `DBZH`, `ZDR`, `KDP`, `RHOHV` and the fall speed on the model grid (T-matrix
rain, and Rayleigh dry snow, hail and graupel as in Jung et al. 2008, with the
equations of Ryzhkov et al. 2020); `to_radar_grid` samples them along the beam
(4/3 Earth radius, site altitude in the radius) and adds the radial velocity. The
rain size distribution of a scheme is not encoded: the intercept `n0_rain`, the
shape `mu_rain` or the number concentration is an explicit assumption. The
[forward operator notebook](Forward_Operator) shows a squall line, the checks
against closed forms and `radar_from_dsd`, and the assumptions. No WRF output is
read here: the Datasets are small synthetic ones with the variables of the two
Registries.

```{code-cell} ipython3
def tiny_wrf(nx=40, ny=30, nz=24, dx=4000.0):
    """A rain and snow blob with the variables of a wrfout file (flat terrain)."""
    zi = 16000.0 * np.linspace(0, 1, nz + 1) ** 1.4
    zm = 0.5 * (zi[:-1] + zi[1:])
    t_k = 288.15 - 6.5e-3 * np.minimum(zm, 11000.0)
    p_pa = 101325.0 * (t_k / 288.15) ** 5.2559
    jj, ii = np.mgrid[0:ny, 0:nx]
    lat = 35.0 + (jj - ny / 2) * dx / 111195.0
    lon = -97.5 + (ii - nx / 2) * dx / (111195.0 * np.cos(np.deg2rad(35.0)))
    blob = np.exp(-(((ii - 24) ** 2 + (jj - 15) ** 2) / 40.0))[None]
    q_rain = np.where(t_k[:, None, None] > 273.15, 3e-3 * blob * np.exp(-zm / 3000.0)[:, None, None], 0.0)
    q_snow = np.where(t_k[:, None, None] <= 273.15, 0.8e-3 * blob, 0.0)
    d3 = ("Time", "bottom_top", "south_north", "west_east")
    d3z = ("Time", "bottom_top_stag", "south_north", "west_east")
    full = lambda a: np.broadcast_to(a, (nz, ny, nx))[None]  # noqa: E731
    return xr.Dataset({
        "T": (d3, full((t_k * (1e5 / p_pa) ** (287.0 / 1004.5) - 300.0)[:, None, None])),
        "P": (d3, full(np.zeros((1, 1, 1)))), "PB": (d3, full(p_pa[:, None, None])),
        "PH": (d3z, np.zeros((1, nz + 1, ny, nx))),
        "PHB": (d3z, np.broadcast_to(9.81 * zi[:, None, None], (nz + 1, ny, nx))[None]),
        "QVAPOR": (d3, full(0.008 * np.exp(-zm / 2500.0)[:, None, None])),
        "QRAIN": (d3, q_rain[None]), "QSNOW": (d3, q_snow[None]),
        "QNRAIN": (d3, np.where(q_rain > 0, 6.0 * q_rain / (np.pi * 1000.0 * 1.2e-3**3), 0.0)[None]),
        "U": (("Time", "bottom_top", "south_north", "west_east_stag"), np.full((1, nz, ny, nx + 1), 12.0)),
        "V": (("Time", "bottom_top", "south_north_stag", "west_east"), np.full((1, nz, ny + 1, nx), 3.0)),
        "W": (d3z, np.full((1, nz + 1, ny, nx), 0.5)),
        "XLAT": (("Time", "south_north", "west_east"), lat[None]),
        "XLONG": (("Time", "south_north", "west_east"), lon[None]),
    })


fo_state = wrf_state(tiny_wrf())  # TEMPERATURE, AIR_DENSITY, HEIGHT, winds, mixing ratios
fo_fields = forward_operator(fo_state, nrain="QNRAIN", mu_rain=2.0)  # name the rain assumption
fo_acc = fo_state.radarx.forward_operator(nrain="QNRAIN", mu_rain=2.0, band="C")  # the accessor
fo_sweep = to_radar_grid(fo_fields, (35.0, -98.1, 300.0), 0.5, gate_range=np.arange(500.0, 90001.0, 500.0))
fo_volume = fo_fields.radarx.to_radar_grid((35.0, -98.1, 300.0), [0.5, 2.0], gate_range=np.arange(500.0, 90001.0, 500.0))
print(f"model grid: DBZH up to {float(fo_fields.DBZH.max()):.1f} dBZ, ZDR up to {float(fo_fields.ZDR.max()):.2f} dB, "
      f"KDP up to {float(fo_fields.KDP.max()):.2f} deg/km; C band ZDR {float(fo_acc.ZDR.max()):.2f} dB")
print(f"radar space: DBZH {float(fo_sweep.DBZH.max()):.1f} dBZ, VRADH {float(fo_sweep.VRADH.min()):.1f} to "
      f"{float(fo_sweep.VRADH.max()):.1f} m/s, volume sweeps {sorted(fo_volume.children)}")

rng_mp = np.random.default_rng(0)
n_mp, nz_mp = 300, 20
lat_mp = 35.0 + rng_mp.uniform(-2, 2, n_mp)
lon_mp = -97.5 + rng_mp.uniform(-2, 2, n_mp)
zi_mp = np.repeat((15000.0 * np.linspace(0, 1, nz_mp + 1) ** 1.4)[None], n_mp, 0)
zm_mp = 0.5 * (zi_mp[:, :-1] + zi_mp[:, 1:])
t_mp = 288.15 - 6.5e-3 * np.minimum(zm_mp, 11000.0)
p_mp = 101325.0 * (t_mp / 288.15) ** 5.2559
dims_mp = ("Time", "nCells", "nVertLevels")
mpas = xr.Dataset({
    "qr": (dims_mp, np.where(t_mp > 273.15, 2e-3, 0.0)[None]), "qs": (dims_mp, np.where(t_mp <= 273.15, 4e-4, 0.0)[None]),
    "rho": (dims_mp, (p_mp / (287.0 * t_mp))[None]), "theta": (dims_mp, (t_mp * (1e5 / p_mp) ** (287.0 / 1004.5))[None]),
    "pressure": (dims_mp, p_mp[None]),
    "uReconstructZonal": (dims_mp, np.full((1, n_mp, nz_mp), 8.0)),
    "uReconstructMeridional": (dims_mp, np.full((1, n_mp, nz_mp), 2.0)),
    "w": (("Time", "nCells", "nVertLevelsP1"), np.zeros((1, n_mp, nz_mp + 1))),
})
mpas_grid_ = xr.Dataset({"lonCell": ("nCells", np.deg2rad(lon_mp)), "latCell": ("nCells", np.deg2rad(lat_mp)),
                         "zgrid": (("nCells", "nVertLevelsP1"), zi_mp)})
mp_fields = forward_operator(mpas_state(mpas, mpas_grid_), n0_rain=4.0e6)  # unstructured mesh, fixed intercept
mp_sweep = to_radar_grid(mp_fields, (35.0, -97.5, 200.0), 1.0, azimuth=np.arange(0.5, 360.0, 2.0),
                         gate_range=np.arange(1000.0, 120001.0, 1000.0))
print(f"MPAS-like mesh: DBZH up to {float(mp_fields.DBZH.max()):.1f} dBZ on the cells, "
      f"{int(mp_sweep.DBZH.notnull().sum())} radar gates with echo")
```

### T-matrix scattering of single drops

The scattering tables of the DSD retrievals come from a threaded C++ T-matrix
code. `raindrop_scattering` gives the table quantities of oblate raindrops for any
wavelength, temperature and canting width; `spheroid_scattering` is the
low-level call (the amplitude and phase matrices of spheroids in given
directions, averaged over the orientation quadrature of `orientation_quadrature`);
`water_refractive_index` and `axis_ratio_brandes` are its inputs.

```{code-cell} ipython3
from radarx.fundamentals import (
    axis_ratio_brandes,
    orientation_quadrature,
    raindrop_scattering,
    spheroid_scattering,
    water_refractive_index,
)

wl = 299.792458 / 5.6  # C band, mm
m = water_refractive_index(wl, 20.0)
drops = raindrop_scattering([2.0, 4.0, 6.0], wl, temperature=20.0, canting_std=10.0)
orient = orientation_quadrature(std=10.0, n_alpha=5, n_beta=10)
S, Z, nmax = spheroid_scattering(
    [1.0, 2.0], 1.0 / axis_ratio_brandes([2.0, 4.0]), wl, m,
    [[90, 90, 0, 180], [90, 90, 0, 0]], orient,
)
print(f"m = {m:.3f}; ZDR of 2, 4, 6 mm drops: "
      f"{(10 * np.log10(drops.sigma_h / drops.sigma_v)).values.round(2)} dB; "
      f"S shape {S.shape}, orders {nmax}")
```

## 15. Dual-PRF velocity correction

A dual-PRF radar alternates two pulse repetition frequencies from ray to ray
and unfolds each ray with the difference of the two velocity estimates. The
difference is noisy (five times the noise of one estimate for an unfolding
factor of 3), so some gates (1 % in the C-band data of the paper, 0.5 % below) are unfolded to the wrong Nyquist
interval of its PRF and stands out as an isolated outlier. `correct_dual_prf`
moves the outliers back by multiples of twice the unambiguous velocity of their
ray's PRF, using the local median velocity (Holleman and Beekhuis 2003). The
data are synthetic: a smooth wind field that the "radar processor" measures at
alternating Nyquist velocities of 16 and 12 m/s (extended interval 48 m/s)
with 1 m/s noise:

```{code-cell} ipython3
vh, vl = 16.0, 12.0  # Nyquist velocities of the high and low PRF
factor = vl / (vh - vl)  # unfolding factor 3
vext = vh * vl / (vh - vl)
rnd = np.random.default_rng(2)
az = np.arange(720) * 0.5 + 0.25
rng = 1000.0 + 250.0 * np.arange(240)
truth = 28.0 * np.sin(np.radians(az)[:, None] + 0.4) * (0.5 + 0.5 * rng[None, :] / rng[-1]) + 5.0 * np.sin(rng[None, :] / 8e3)


def fold(v, vu):
    return (v + vu) % (2 * vu) - vu


high = np.arange(az.size) % 2 == 0
vu = np.where(high, vh, vl)[:, None]
noisy = truth + rnd.standard_normal(truth.shape)  # what each ray measures, before folding
measured = fold(noisy, vu)
previous = np.roll(measured, 1, axis=0)
primary = fold((factor + 1) * np.where(high[:, None], previous, measured) - factor * np.where(high[:, None], measured, previous), vext)
radar = measured + 2 * vu * np.round((primary - measured) / (2 * vu))  # the unfolded velocity of the processor
dual = xr.Dataset(
    {"VRADH": (("azimuth", "range"), radar)},
    coords={"azimuth": az, "range": rng, "elevation": ("azimuth", np.full(az.size, 0.5))},
)

fixed = dual.radarx.correct_dual_prf("VRADH", nyquist_high=vh, nyquist_low=vl)
direct = correct_dual_prf(dual, "VRADH", nyquist_high=vh, nyquist_low=vl, passes=1)
outliers = lambda v: float((np.abs(v - noisy) > 5.0).mean())
print(f"outliers: raw {outliers(radar):.2%}, one pass {outliers(direct.VRADH_dualprf.values):.3%}, "
      f"two passes {outliers(fixed.VRADH_dualprf.values):.3%}; "
      f"{int((fixed.VRADH_dualprf_shift != 0).sum())} gates moved, "
      f"{int((fixed.VRADH_dualprf_shift.values[high] != 0).sum())} of them on high-PRF rays")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
x, y = rng * np.sin(np.radians(az))[:, None] / 1e3, rng * np.cos(np.radians(az))[:, None] / 1e3
for ax, (name, field) in zip(axes, [("raw dual-PRF velocity", radar), ("corrected", fixed.VRADH_dualprf.values), ("gates moved", fixed.VRADH_dualprf_shift.values != 0)]):
    pm = ax.pcolormesh(x, y, field, cmap="RdBu_r" if name != "gates moved" else "Greys", vmin=-35 if name != "gates moved" else 0, vmax=35 if name != "gates moved" else 1)
    ax.set(title=name, aspect="equal", xlim=(-30, 30), ylim=(-30, 30), xlabel="east (km)")
plt.show()
```

What it does not do: it does not remove clutter and noise first (the group-size
filter of the paper), and it does not unfold velocities beyond the extended
interval; run `dealias_velocity` with `nyquist_velocity` equal to the extended
interval (48 m/s here) for that.

## 16. Ensembles and uncertainty of the retrievals

Every retrieval above is one number per gate. `ensemble_retrieval` gives the
uncertainty that comes from the random error of the measurements, the calibration and the
prior knowledge of the DSD: it perturbs the radar fields (`perturb_radar`, with the
standard deviations of Melnikov and Zrnic 2004 from `measurement_error`), runs
the retrieval on each member and returns the mean, median, standard deviation and
percentiles (`ensemble_statistics` does the last step for any array with a `member`
dimension). `relation_scatter` is the scatter of the rain rate relations caused by the
DSD that sets the default width of the prior part. The
[ensemble notebook](Ensemble_Uncertainty) tests the spread against a known truth and
shows maps for the S-band KLBB sweep and the C-band CSAPR2 sweep. Here is the
Lubbock sweep again, with the signal-to-noise ratio from the noise level of the sweep:

```{code-cell} ipython3
from open_radar_data import DATASETS

lubbock = xd.io.open_nexradlevel2_datatree(DATASETS.fetch("KLBB20160601_150025_V06"), sweep=[0])
ens_sweep = lubbock["sweep_0"].to_dataset(inherit=False).isel(range=slice(0, 600))
for var, limit in (("DBZH", -32.0), ("ZDR", -12.9), ("RHOHV", 0.21)):
    ens_sweep[var] = ens_sweep[var].where(ens_sweep[var] > limit)
ens_sweep = ens_sweep.assign(SNR=snr(ens_sweep))

error = measurement_error(ens_sweep, snr="SNR")  # standard deviation of the error of each field
rain_gates = (ens_sweep.RHOHV > 0.95) & (ens_sweep.DBZH > 30)
print({v: round(float(error[v].where(rain_gates).median()), 3) for v in error.data_vars})

members = perturb_radar(ens_sweep, members=8, seed=1, snr="SNR")  # the fields with random errors
print(dict(members.DBZH.sizes), "| offset of the reflectivity of each member (dB):",
      np.round((members.DBZH - ens_sweep.DBZH).where(rain_gates).median(["azimuth", "range"]).values, 2))
stats = ensemble_statistics(members)  # mean, median, std and percentiles over the members
print(sorted(v for v in stats.data_vars if v.startswith("ZDR")))

scatter = relation_scatter("S")  # the scatter of ln R about each power law, from the DSD prior
print(scatter.ln_sd.to_series().round(2).to_dict())
ens = ensemble_retrieval(ens_sweep, "rain_rate_z_zdr", members=24, seed=1, noise={"snr": "SNR"}, band="S")
```

The same functions are methods of the accessors, and a volume goes through sweep by
sweep. The retrievals the ensemble can run are listed in the documentation of
`ensemble_retrieval`:

```{code-cell} ipython3
via_accessor = ens_sweep.radarx.ensemble_retrieval("kdp", members=16, seed=1, noise={"snr": "SNR"})
print(float(via_accessor.KDP_median.where(rain_gates).median()), float(via_accessor.KDP_std.where(rain_gates).median()))
print(float(ens_sweep.radarx.measurement_error(snr="SNR").ZDR_error.where(rain_gates).median()))
print(sorted(v for v in ens_sweep.radarx.perturb_radar(members=4, snr="SNR").data_vars))
print(sorted(v for v in members.radarx.ensemble_statistics().data_vars if v.startswith("PHIDP")))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
azi, rng = np.deg2rad(ens_sweep.azimuth.values)[:, None], ens_sweep.range.values[None, :]
x, y = rng * np.sin(azi) / 1e3, rng * np.cos(azi) / 1e3
for ax, (da, title, style) in zip(axes, [
    (ens.RAIN_RATE_Z_ZDR_median, "R($Z_H$, $Z_{DR}$) median (mm/h)", dict(cmap="turbo", vmin=0, vmax=50)),
    (ens.RAIN_RATE_Z_ZDR_std, "standard deviation (mm/h)", dict(cmap="magma_r", vmin=0, vmax=15)),
    (via_accessor.KDP_std, "$K_{DP}$ standard deviation (deg/km)", dict(cmap="magma_r", vmin=0, vmax=0.6)),
]):
    pm = ax.pcolormesh(x, y, da.where(rain_gates), shading="auto", **style)
    plt.colorbar(pm, ax=ax, shrink=0.8)
    ax.set(title=title, aspect="equal", xlim=(-120, 20), ylim=(-30, 110))
plt.show()
```

## 17. ZDR columns

The [ZDR column notebook](ZDR_Column) has this example with a section and a real case.

Drops that an updraft lofts above the freezing level freeze slowly, so a column
of differential reflectivity of 1 dB or more stands above the 0 °C level, and its
depth follows the strength of the updraft (Snyder et al. 2015). `zdr_column`
gives the depth at every point of a grid, counting the levels from the freezing
level up while $Z_{DR} \ge 1$ dB and the copolar correlation coefficient is at least
0.8; `zdr_column_metrics` gives the maximum depth, the area and the volume of
the columns. The data are synthetic: an updraft whose column is 3 km deep in the
centre, with a hail spike of low $\rho_{hv}$ on its flank.

```{code-cell} ipython3
from radarx.retrieve import zdr_column, zdr_column_metrics

zc = np.arange(0.0, 12001.0, 250.0)
yc = xc = np.arange(41) * 1000.0
r = np.hypot(yc[:, None] - 20e3, xc[None, :] - 20e3)
column_top = 4000.0 + 3000.0 * np.clip(1 - r / 8e3, 0, None)  # freezing level 4 km
zdr_grid = xr.DataArray(
    np.where(zc[:, None, None] < column_top[None], 2.0, 0.2),
    dims=("z", "y", "x"),
    coords={"z": zc, "y": yc, "x": xc},
    name="ZDR",
)
rho_grid = xr.full_like(zdr_grid, 0.98)
rho_grid[:, 20:24, 28:32] = 0.6  # a hail spike
depth = zdr_column(zdr_grid, rhohv=rho_grid, freezing_level=4000.0)
metrics = zdr_column_metrics(depth, min_depth=1000.0)
print(f"deepest column {float(metrics.ZDR_COLUMN_MAX_DEPTH):.0f} m, "
      f"area {float(metrics.ZDR_COLUMN_AREA):.0f} km2, volume {float(metrics.ZDR_COLUMN_VOLUME):.0f} km3")
# the accessors give the same
same = zdr_grid.radarx.zdr_column(rhohv=rho_grid, freezing_level=4000.0)
print(bool(np.allclose(same, depth)), float(same.radarx.zdr_column_metrics(min_depth=2000.0).ZDR_COLUMN_AREA))
fig, ax = plt.subplots(figsize=(5, 4.2), layout="constrained")
pm = ax.pcolormesh(xc / 1e3, yc / 1e3, depth, cmap="viridis", vmin=0, vmax=3000, shading="auto")
fig.colorbar(pm, label="ZDR column depth (m)")
ax.set(xlabel="x (km)", ylabel="y (km)", aspect="equal")
```

The hail spike removes the grid points of low $\rho_{hv}$ from the column, the
paper's way of keeping the three-body scatter signature out of the product.
The depth needs a freezing level in the vertical reference of the grid (a
sounding or a model) and a calibrated $Z_{DR}$.

## 18. ZDR arc and KDP-ZDR separation

The [ZDR arc notebook](ZDR_Arc) shows the same storm with its panels and the effect of the FFD vector.

In the forward flank of a supercell the storm-relative wind sorts the drops by size: the
big ones fall out near the inflow edge and form an arc of high $Z_{DR}$, the small ones are
carried deeper into the storm, where $K_{DP}$ is enhanced. The distance between the two, and
its angle to the storm motion, indicate the low-level shear (Wilson and Van Den Broeke 2021).
`zdr_arc` finds the arc, the storm and the $K_{DP}$ foot on a grid of the lowest level and
gives the arc area, the mean and top-10 $Z_{DR}$ of the arc, and the length and angle of the
separation vector. The data are synthetic: a storm with its arc on the south (inflow) side,
a patch of high $Z_{DR}$ on the north side that the inflow mask removes, and a foot in the core.
The forward-flank downdraft vector points north and the storm moves east.

```{code-cell} ipython3
from radarx.retrieve import zdr_arc

xa = ya = np.arange(0.0, 60001.0, 500.0)
XX, YY = np.meshgrid(xa, ya)
core = (XX - 30e3) ** 2 + (YY - 32e3) ** 2
supercell = xr.Dataset(
    {
        "DBZH": (("y", "x"), 60.0 * np.exp(-core / (2 * 9e3**2))),
        "ZDR": (("y", "x"), np.where(
            ((XX >= 26e3) & (XX <= 34e3) & (YY >= 20e3) & (YY <= 23e3))
            | ((XX >= 26e3) & (XX <= 34e3) & (YY >= 41e3) & (YY <= 43e3)), 4.0, 0.5)),
        "RHOHV": (("y", "x"), np.full(XX.shape, 0.98)),
        "KDP": (("y", "x"), np.where(np.hypot(XX - 30e3, YY - 29e3) <= 3e3, 2.5, 0.1)),
    },
    coords={"x": xa, "y": ya},
)
arc = zdr_arc(supercell, rhohv="RHOHV", kdp="KDP", ffd_direction=0.0, storm_motion=90.0)
s = arc.isel(storm=0)
print(f"{arc.sizes['candidate']} candidate of the arc; arc {float(s.arc_area):.1f} km2, mean ZDR {float(s.arc_mean_zdr):.1f} dB, "
      f"top-10 {float(s.arc_top10_zdr):.1f} dB; separation {float(s.separation_length):.1f} km at {float(s.separation_angle):.0f}° to the storm motion")
# the accessor gives the same
print(bool(supercell.radarx.zdr_arc(rhohv="RHOHV", kdp="KDP", ffd_direction=0.0, storm_motion=90.0).arc_area[0] == s.arc_area))
fig, ax = plt.subplots(figsize=(5.2, 4.6), layout="constrained")
ax.pcolormesh(xa / 1e3, ya / 1e3, supercell.DBZH, cmap="Greys", vmin=0, vmax=60, shading="auto")
ax.contour(xa / 1e3, ya / 1e3, supercell.ZDR, levels=[3.25], colors="tab:blue")
ax.contour(xa / 1e3, ya / 1e3, arc.ZDR_ARC_STORM >= 0, levels=[0.5], colors="tab:red")
ax.contour(xa / 1e3, ya / 1e3, arc.KDP_FOOT_STORM >= 0, levels=[0.5], colors="tab:green")
ax.annotate("", xy=(float(s.arc_x) / 1e3, float(s.arc_y) / 1e3), xytext=(float(s.foot_x) / 1e3, float(s.foot_y) / 1e3),
            arrowprops=dict(arrowstyle="->", color="k"))
ax.set(xlabel="x (km)", ylabel="y (km)", aspect="equal", title="ZDR arc (red), KDP foot (green), 3.25 dB contours (blue)")
```

The paper removes the candidates that are not part of the arc with a random forest trained on
hand-labelled objects; `classifier` takes a model of your own or a function of the features
(`radarx.retrieve.zdr_arc.FEATURES`), and without it every candidate near a storm is part of
the arc. The reflectivity gradient mask keeps the north patch out here. The algorithm is for
supercells: the leading edge of a squall line and the high $Z_{DR}$ below developing
updrafts are often detected as arcs, and $Z_{DR}$ must be calibrated.

## 19. Radar and satellite matching

A ground radar and the Dual-frequency Precipitation Radar (DPR) of the GPM core satellite see the
same storm with different beams. `match_dpr` averages both over the volume that they share: the
DPR ray is slanted, so the part at the height $h$ lies $h \tan\theta$ closer to the track than
the position in the file, and the ground radar beam is at the height that its elevation gives
(Schwaller and Morris 2011). `match_statistics` compares the pairs, `dpr_dsd` splits the DSD
parameters of the DPR (Dm and 10 log10 Nw with $\mu = 3$, the conventions of the radarx DSD retrieval,
Seto et al. 2021) and `match_imerg` puts a radar rain rate on the grid and the half hours of IMERG
(Huffman et al. 2020). The files of the DPR and of IMERG need an Earthdata login, so the data are
synthetic: a storm that leans with height. The [Satellite_Matching](Satellite_Matching) notebook
shows the geometry, the DSD check and the IMERG comparison in full.

```{code-cell} ipython3
import pyproj
from xradar.georeference import antenna_to_cartesian

from radarx._gridcf import aeqd_crs, lonlat
from radarx.retrieve import dpr_dsd, dpr_parallax, match_dpr, match_imerg, match_statistics


def sat_storm(x, y, h):
    dbz = 50.0 * np.exp(-((x - 55e3 - 0.5 * h) ** 2 + (y - 5e3) ** 2) / (2 * 9e3**2)) - 3e-3 * np.maximum(h - 8e3, 0.0)
    return np.where(dbz > 5.0, dbz, np.nan)


sat_az = np.arange(0.0, 360.0, 2.0) + 1.0
sat_rng = 125.0 + 250.0 * np.arange(500)
sat_sweeps = {}
for i, el in enumerate((1.0, 4.0, 9.0, 15.0)):
    gx, gy, gz = antenna_to_cartesian(sat_rng[None, :], sat_az[:, None], np.full((sat_az.size, 1), el))
    sat_sweeps[f"/sweep_{i}"] = xr.Dataset(
        {"DBZH": (("azimuth", "range"), sat_storm(gx, gy, gz), {"units": "dBZ"})},
        coords={"azimuth": sat_az, "elevation": ("azimuth", np.full(sat_az.size, el)), "range": sat_rng},
    )
sat_site = xr.Dataset(coords={"latitude": 30.0, "longitude": -90.0, "altitude": 0.0})
sat_volume = xr.DataTree.from_dict({"/": sat_site, **sat_sweeps})

# a DPR swath that observes the same storm along its slanted rays (track 30 km west of the radar)
sat_crs = aeqd_crs(30.0, -90.0)
sat_zenith = (np.arange(49) - 24) * 17.5 / 24.0
sat_xx, sat_yy = np.meshgrid(-30e3 + 407e3 * np.tan(np.radians(sat_zenith)), np.arange(-60e3, 60e3, 5e3))
sat_lon, sat_lat = pyproj.Transformer.from_crs(sat_crs, 4326, always_xy=True).transform(sat_xx, sat_yy)
sat_h = ((175 - np.arange(176)) * 125.0)[None, None, :] * np.cos(np.radians(sat_zenith))[None, :, None]
sat_h = np.broadcast_to(sat_h, (sat_xx.shape[0], 49, 176)).copy()
sat_true_x = sat_xx[..., None] - sat_h * np.tan(np.radians(sat_zenith))[None, :, None]
sat_dpr = xr.Dataset(
    {
        "Longitude": (("scan", "ray"), sat_lon),
        "Latitude": (("scan", "ray"), sat_lat),
        "localZenithAngle": (("scan", "ray"), np.broadcast_to(sat_zenith, sat_lon.shape).copy()),
        "height": (("scan", "ray", "bin"), sat_h),
        "zFactorFinal": (("scan", "ray", "bin"), sat_storm(sat_true_x, sat_yy[..., None], sat_h), {"units": "dBZ"}),
    }
)
print(f"parallax of a 5 km bin on the outer ray: {float(dpr_parallax(5000.0, sat_zenith[-1])):.0f} m")
sat_kwargs = dict(gr_fields="DBZH", dpr_fields="zFactorFinal", min_range=20e3, max_range=100e3)
sat_match = match_dpr(sat_volume, sat_dpr, **sat_kwargs)
sat_flat = sat_dpr.assign(localZenithAngle=sat_dpr.localZenithAngle * 0.0)  # without the parallax
sat_none = sat_volume.radarx.match_dpr(sat_flat, **sat_kwargs)
for label, m in (("with parallax correction", sat_match), ("without", sat_none)):
    ok = m.valid & (m.gr_DBZH > 15) & (m.elevation >= 4.0) & (abs(m.ray - 24) >= 10)
    s = match_statistics(m.gr_DBZH.where(ok), m.dpr_zFactorFinal.where(ok))
    print(f"{label}: {int(s['count'])} pairs, bias {float(s.bias):.2f} dB, RMSE {float(s.rmse):.2f} dB")

# the DSD parameters of the file: 10 log10 Nw and Dm
sat_param = xr.DataArray(np.stack([np.full(sat_h.shape, 34.0), np.full(sat_h.shape, 1.8)], axis=-1), dims=("scan", "ray", "bin", "nDSD"))
print(float(dpr_dsd(sat_param).DM[0, 0, 0]), "mm,", float(dpr_dsd(sat_param).NW[0, 0, 0]), "m-3 mm-1")

# IMERG: a radar rain rate every 10 minutes and the half hour of IMERG that holds them
sat_x = np.arange(-100e3, 100e3 + 1, 1000.0)
sat_glon, sat_glat = lonlat(sat_crs.to_cf(), sat_x, sat_x)
sat_times = np.datetime64("2024-05-01T00:05", "ns") + np.arange(3) * np.timedelta64(10, "m")
sat_rate = xr.DataArray(
    np.stack([np.full((sat_x.size, sat_x.size), r) for r in (3.0, 6.0, 9.0)]),
    dims=("time", "y", "x"),
    coords={"time": sat_times, "x": sat_x, "y": sat_x, "lon": (("y", "x"), sat_glon), "lat": (("y", "x"), sat_glat)},
)
sat_imerg = xr.DataArray(
    np.full((1, 20, 20), 5.0),
    dims=("time", "lon", "lat"),
    coords={"time": np.array(["2024-05-01T00:00"], dtype="datetime64[ns]"), "lon": np.round(np.arange(-90.95, -89.0, 0.1), 2), "lat": np.round(np.arange(29.05, 31.0, 0.1), 2)},
)
sat_both = sat_rate.radarx.match_imerg(sat_imerg)  # the same as match_imerg(sat_rate, sat_imerg)
print(f"{float(sat_both.radar.max()):.1f} mm/h from the three scans (3, 6, 9: 10 minutes each), IMERG {float(sat_both.imerg.max()):.1f}")
```

The storm is known, so the check is exact: with the parallax correction the pairs agree to about
1 dB (RMSE; the two beams do not sample the same volume), without it the RMSE is more than 3 dB
where the beams are high. Three scans of 3, 6 and 9 mm/h that represent 10 minutes each average to
6 mm/h in the half hour. The choices that the papers leave open (the footprint size, the weights, the
fraction of the volume that must have data) are listed in the documentation of
`radarx.retrieve.satellite_match`.
## 20. Volume velocity processing

A single Doppler radar sees only the radial velocity, but if the wind is a linear function of
position inside a region, the gates of the region determine the wind at its centre, its
gradients, the divergence and the deformation (VVP, Waldteufel and Corbin 1979, in the form of
Doviak and Zrnic 1993). `vvp_winds` fits the linear model in cylinders around the nodes of a
grid, from all sweeps of a volume. The data are synthetic: a linear wind with divergence,
shear and vertical shear, seen by five sweeps of a radar.

```{code-cell} ipython3
from radarx.fundamentals.constants import EARTH_RADIUS
from radarx.fundamentals.geometry import beam_center_height, ground_range

vz_az = np.arange(0.5, 360.0, 1.0)
vz_rng = np.arange(500.0, 60001.0, 500.0)
vz_wind = dict(u0=5.0, ux=1e-4, uy=2e-4, uz=2e-3, v0=-3.0, vx=-5e-5, vy=3e-4, vz=-1e-3)
vz_nodes = {}
for i, el in enumerate((0.5, 1.5, 3.0, 5.0, 8.0)):
    sw = xr.Dataset(
        {"VRADH": (("azimuth", "range"), np.zeros((vz_az.size, vz_rng.size)))},
        coords={"azimuth": vz_az, "range": vz_rng, "elevation": ("azimuth", np.full(vz_az.size, el)),
                "altitude": 100.0, "latitude": 33.9, "longitude": -88.3},
    )
    reff = 4.0 / 3.0 * EARTH_RADIUS
    s_ = ground_range(vz_rng[None, :], el, reff=reff, radar_height=100.0)
    z_ = beam_center_height(vz_rng[None, :], el, radar_height=100.0, reff=reff)
    theta = np.deg2rad(el) + s_ / reff  # local elevation of the beam
    sa_, ca_ = np.sin(np.deg2rad(vz_az))[:, None], np.cos(np.deg2rad(vz_az))[:, None]
    x_, y_ = s_ * sa_, s_ * ca_
    u_ = vz_wind["u0"] + vz_wind["ux"] * x_ + vz_wind["uy"] * y_ + vz_wind["uz"] * z_
    v_ = vz_wind["v0"] + vz_wind["vx"] * x_ + vz_wind["vy"] * y_ + vz_wind["vz"] * z_
    sw["VRADH"] = sw["VRADH"].copy(data=np.cos(theta) * (u_ * sa_ + v_ * ca_))
    vz_nodes[f"/sweep_{i}"] = sw
vz_volume = xr.DataTree.from_dict({"/": xr.Dataset(coords={"latitude": 33.9, "longitude": -88.3, "altitude": 100.0}), **vz_nodes})
vvp = vvp_winds(vz_volume, radius=15000.0, depth=2000.0, x=[-20000.0, 0.0, 25000.0], y=[30000.0], z=[1500.0, 3000.0], min_gates=200)
vvp_acc = vz_volume.radarx.vvp_winds(radius=15000.0, depth=2000.0, x=[0.0], y=[30000.0], z=[1500.0], min_gates=200)  # the accessor
print(f"divergence {float(vvp.divergence.mean()) * 1e4:.3f} 1e-4 1/s (truth {(vz_wind['ux'] + vz_wind['vy']) * 1e4:.1f}), "
      f"standard error {float(vvp.divergence_error.max()):.1e} 1/s")
print("accessor agrees:", bool(np.isclose(float(vvp_acc.u[0, 0, 0]), float(vvp.u.sel(x=0.0, y=30000.0, z=1500.0)))))
```

`u` and `v` are the modified components of the model, equal to the wind at the centre only
where the vorticity term is small (`vorticity=` removes it for a known vorticity). The
vorticity itself cannot be retrieved. The [VVP notebook](Volume_Velocity_Processing) checks the
standard error against the worked example of Doviak and Zrnic and the winds against the VAD of a
NEXRAD volume.

## 25. Radar-gauge merging

A radar field has errors that vary in space, a gauge measures one point well. `gauge_adjust`
merges them on the radar grid with the methods compared by Goudenhoofdt and Delobbe (2009): the
mean field bias (`"mfb"`), a range-dependent adjustment (`"rda"`), the Brandes spatial adjustment
(`"brandes"`), ordinary kriging of the gauges (`"ok"`), conditional merging of Sinclair and
Pegram (2005) (`"conditional"`) and kriging with the radar as external drift (`"ked"`). The
variograms are fitted automatically (`gauge_variogram`) and the kriging systems of all pixels
and times run in one multithreaded call. The data are synthetic: a rain accumulation field, a
radar that reads 40 % too low and loses more with range, and 30 gauges that sample the truth.
The [gauge adjustment notebook](Gauge_Adjustment) has the full comparison.

```{code-cell} ipython3
from radarx.retrieve import gauge_adjust, gauge_variogram, radar_at_gauges

gx = gy = (np.arange(60) - 29.5) * 1000.0
GX, GY = np.meshgrid(gx, gy)
rain_truth = 12.0 * np.exp(-((GX - 8e3) ** 2 + (GY + 5e3) ** 2) / (2 * 9e3**2)) + 3.0
radar_acc = xr.DataArray(
    rain_truth * 0.6 * np.exp(-np.hypot(GX, GY) / 90e3)
    * np.exp(0.15 * np.random.default_rng(2).standard_normal(GX.shape)),
    dims=("y", "x"), coords={"x": gx, "y": gy}, attrs={"units": "mm"}, name="RAIN",
)
rg = np.random.default_rng(5)
px, py = rg.uniform(-28e3, 28e3, 30), rg.uniform(-28e3, 28e3, 30)
gauge_obs = xr.DataArray(
    12.0 * np.exp(-((px - 8e3) ** 2 + (py + 5e3) ** 2) / (2 * 9e3**2)) + 3.0,
    dims="station", coords={"x": ("station", px), "y": ("station", py)}, attrs={"units": "mm"},
)
variogram = gauge_variogram(gauge_obs, model="exponential", n_lags=8)
print(f"fitted range {float(variogram['range']) / 1000:.0f} km; radar at the gauges "
      f"{float(radar_at_gauges(radar_acc, gauge_obs).mean()):.1f} mm, gauges {float(gauge_obs.mean()):.1f} mm")
errors = {}
for method in ["mfb", "rda", "brandes", "ok", "conditional", "ked"]:
    merged = gauge_adjust(radar_acc, gauge_obs, method=method)
    errors[method] = float(np.sqrt(((merged.RAIN_ADJUSTED - rain_truth) ** 2).mean()))
print("RMSE against the truth (mm):", {k: round(v, 2) for k, v in errors.items()},
      "radar", round(float(np.sqrt(((radar_acc - rain_truth) ** 2).mean())), 2))
# the accessor gives the same
same = radar_acc.radarx.gauge_adjust(gauge_obs, method="ked")
print(bool(np.allclose(same.RAIN_ADJUSTED, gauge_adjust(radar_acc, gauge_obs, method="ked").RAIN_ADJUSTED)))
```

The kriging methods come closest to the truth here, and conditional merging, which keeps the
structure of the radar field and matches the gauges, is the best; which of them wins depends on the
field and the network (see the notebook). `RAIN_ADJUSTED_STD` is the kriging standard deviation,
zero at the gauges. The radar must be corrected for attenuation, calibration and
clutter first, and the gauge and the radar must have the same units.

## 26. Machine-learning rain rate

`rain_rate_ml` runs a model of the registry (`radarx.ml`) on the polarimetric variables, and the
height and temperature if the model uses them, and returns the rain rate with a lower and an
upper bound. The radarx network is trained on simulated drop size distributions
(`ml/models/qpe`) and its weights are not published yet, so this example registers a model that
is built here: the C-band-like power law $R(Z_H, Z_{DR})$ written as an ONNX graph. The
[machine-learning rain rate notebook](ML_Rainfall_Estimation) fits such a model to simulated
drop size distributions and runs it on a real sweep.

```{code-cell} ipython3
import tempfile
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from radarx import ml
from radarx.retrieve import rain_rate_ml, rain_rate_z_zdr

f32 = TensorProto.FLOAT
a_c, b_c, c_c = 0.0058, 0.91, -2.09  # C-band R(Z, ZDR), Bringi and Chandrasekar (2001)
w_zdr = np.array([[0.1 * np.log(10.0) * b_c], [0.1 * np.log(10.0) * c_c]], dtype=np.float32)
graph = helper.make_graph(
    [
        helper.make_node("Gemm", ["features", "w", "b"], ["ln_r"]),
        helper.make_node("Exp", ["ln_r"], ["col"]),
        helper.make_node("Squeeze", ["col", "axes"], ["rain_rate"]),
    ],
    "power_law",
    [helper.make_tensor_value_info("features", f32, ["N", 2])],
    [helper.make_tensor_value_info("rain_rate", f32, ["N"])],
    [
        helper.make_tensor("w", f32, [2, 1], w_zdr.ravel().tolist()),
        helper.make_tensor("b", f32, [1], [float(np.log(a_c))]),
        helper.make_tensor("axes", TensorProto.INT64, [1], [1]),
    ],
)
onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
onnx_model.ir_version = 8
path = Path(tempfile.mkdtemp()) / "c_band_z_zdr.onnx"
onnx.save(onnx_model, str(path))
ml.register_model("c-band-power-law", path, None, "MIT", "Bringi and Chandrasekar (2001), https://doi.org/10.1017/CBO9780511541094",
                  task="rainfall-estimation", features=["dbz", "zdr"], overwrite=True)

rain_sweep = xr.Dataset(
    {"DBZH": (("azimuth", "range"), np.random.default_rng(1).uniform(15, 50, (36, 80))),
     "ZDR": (("azimuth", "range"), np.random.default_rng(2).uniform(0.2, 3.0, (36, 80)))},
    coords={"azimuth": np.arange(36) * 10.0, "range": 500.0 + np.arange(80) * 250.0},
)
learned = rain_rate_ml(rain_sweep, model="c-band-power-law", band="C", mask=rain_sweep.DBZH > 20)
physical = rain_rate_z_zdr(rain_sweep, band="C", z_cap=None)
print({k: v for k, v in learned.RAIN_RATE_ML.attrs.items() if k.startswith("ml_")})
print("same as the power law of radarx.retrieve where estimated:",
      bool(np.allclose(learned.RAIN_RATE_ML.values[rain_sweep.DBZH.values > 20],
                       physical.values[rain_sweep.DBZH.values > 20], rtol=1e-4)))
# the accessor gives the same
print(bool(np.allclose(rain_sweep.radarx.rain_rate_ml(model="c-band-power-law", band="C", mask=rain_sweep.DBZH > 20).RAIN_RATE_ML,
                       learned.RAIN_RATE_ML, equal_nan=True)))
```

The gates without a finite input or outside the mask are NaN and the network is not run on them. A
model that returns `rain_rate_lower` and `rain_rate_upper` gets them as `RAIN_RATE_ML_LOWER` and
`RAIN_RATE_ML_UPPER`.
## 21. Storm diagnostics: KDP columns, size sorting and cells

Three more diagnostics of a convective storm on a grid (the
[KDP columns](KDP_Columns), [size sorting](Size_Sorting) and [cell tracking](Cell_Tracking) notebooks show them in more detail).
`kdp_column` finds the columns of specific differential phase that an updraft
carries above the freezing level, after van Lier-Walqui et al. (2016): the KDP
integrated in a slab above the freezing level is smoothed and divided into one
column per maximum with a watershed, and every grid point gets the height of the
highest level with $K_{DP}\ge 0.75$ °/km. `storm_size_sorting` measures the separation
of the low-level KDP and $Z_{DR}$ regions that the storm-relative wind causes by
sorting the raindrops by size, after Loeffler and Kumjian (2018): the centroids of
the regions above an adaptive threshold, the distance between them and the angle
of the vector from the KDP to the $Z_{DR}$ centroid to the storm motion.
`identify_cells`, `track_cells` and `cell_statistics` find the storm cells with
several thresholds and a watershed (Heikenfeld et al. 2019), link them from volume
to volume by their overlap after the storm motion (Feng et al. 2023) and collect the
statistics of any gridded field inside every cell. The data are synthetic: two
updrafts, and two storms that move at 20 and 12 m/s, one of which has a $Z_{DR}$
region 4 km east of its KDP region.

```{code-cell} ipython3
from radarx.retrieve import (
    cell_statistics,
    identify_cells,
    kdp_column,
    kdp_column_metrics,
    storm_size_sorting,
    track_cells,
)

# KDP columns: two updrafts, freezing level at 4 km
zk = np.arange(0.0, 12001.0, 250.0)
yk = xk = np.arange(41) * 1000.0
r1 = np.hypot(yk[:, None] - 12e3, xk[None, :] - 12e3)
r2 = np.hypot(yk[:, None] - 30e3, xk[None, :] - 28e3)
tops = np.maximum(4000 + 3000 * np.clip(1 - r1 / 6e3, 0, None), 4000 + 2000 * np.clip(1 - r2 / 5e3, 0, None))
kdp_grid = xr.DataArray(
    np.where((zk[:, None, None] >= 4000) & (zk[:, None, None] < tops[None]), 1.5, 0.0),
    dims=("z", "y", "x"),
    coords={"z": zk, "y": yk, "x": xk},
    name="KDP",
)
columns = kdp_column(kdp_grid, freezing_level=4000.0)
col_metrics = kdp_column_metrics(columns)
print(f"{int(col_metrics.KDP_COLUMN_COUNT)} KDP columns, tops up to {float(col_metrics.KDP_COLUMN_MAX_TOP):.0f} m, "
      f"{float(col_metrics.KDP_COLUMN_AREA):.0f} km2, KDP volume {float(col_metrics.KDP_VOLUME):.0f} km3")
print("accessor agrees:", bool(np.allclose(kdp_grid.radarx.kdp_column(freezing_level=4000.0).KDP_COLUMN_TOP.fillna(-1),
                                           columns.KDP_COLUMN_TOP.fillna(-1))),
      "| metrics accessor:", int(columns.radarx.kdp_column_metrics().KDP_COLUMN_COUNT))

# two storms that move, 5 min apart; the first has its ZDR region 4 km east of its KDP region
nt, step = 6, 300.0
xs = ys = np.arange(80) * 1000.0
times = np.datetime64("2022-03-30T22:00") + np.arange(nt) * np.timedelta64(300, "s")


def blob(cx, cy, peak, sigma):
    return peak * np.exp(-((xs[None, :] - cx) ** 2 + (ys[:, None] - cy) ** 2) / (2 * sigma**2))


dbz, kdp, zdr = [], [], []
for i in range(nt):
    c1 = (10e3 + 20.0 * step * i, 15e3 + 5.0 * step * i)
    c2 = (45e3 + 12.0 * step * i, 55e3 - 4.0 * step * i)
    dbz.append(blob(*c1, 55, 3000.0) + blob(*c2, 50, 3500.0) - 10)
    kdp.append(blob(c1[0] - 1e3, c1[1], 3.0, 2000.0))
    zdr.append(blob(c1[0] + 3e3, c1[1], 4.0, 2000.0))
coords = {"time": times, "y": ys, "x": xs}
storms = xr.Dataset(
    {
        "DBZH": (("time", "y", "x"), np.array(dbz)),
        "KDP": (("time", "y", "x"), np.array(kdp)),
        "ZDR": (("time", "y", "x"), np.array(zdr)),
    },
    coords=coords,
)
sorting = storm_size_sorting(storms, box=(0, 50e3, 0, 40e3))
print("separation distance (km):", np.round(sorting.SEPARATION_DISTANCE.values, 2),
      "| orientation to the storm motion (deg):", np.round(sorting.SEPARATION_ORIENTATION.values, 0))
print(f"fitted storm motion {float(sorting.STORM_MOTION_U[0]):.1f}, {float(sorting.STORM_MOTION_V[0]):.1f} m/s;",
      f"size sorting parameter {float(sorting.SIZE_SORTING_PARAMETER.median()):.1f} km")

cells = identify_cells(storms, thresholds=(20, 30, 40))
tracks = track_cells(cells, motion=(16.0, 2.0), max_speed=30.0)
series = cell_statistics(tracks.CELL_ID, storms[["DBZH", "KDP"]], stats=("max", "mean"))
print("cells per volume:", cells.CELL_COUNT.values.tolist(),
      "| tracks:", tracks.sizes["cell"], "lifetimes:", tracks.cell_lifetime.values.tolist())
print("peak reflectivity of track 1 (dBZ):", np.round(series.DBZH_max.sel(cell=1).values, 0))
# the accessors give the same
same = storms.radarx.storm_size_sorting(box=(0, 50e3, 0, 40e3))
c2 = storms.DBZH.radarx.identify_cells(thresholds=(20, 30, 40))
t2 = c2.radarx.track_cells(motion=(16.0, 2.0), max_speed=30.0)
s2 = t2.CELL_ID.radarx.cell_statistics(storms[["DBZH", "KDP"]], stats=("max", "mean"))
print(bool(np.allclose(same.SEPARATION_DISTANCE, sorting.SEPARATION_DISTANCE)),
      bool(np.array_equal(t2.CELL_ID, tracks.CELL_ID)), bool(np.allclose(s2.DBZH_max, series.DBZH_max, equal_nan=True)))
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
pm = axes[0].pcolormesh(xk / 1e3, yk / 1e3, columns.KDP_COLUMN_TOP, cmap="viridis", vmin=4000, vmax=8000, shading="auto")
fig.colorbar(pm, ax=axes[0], label="KDP column top (m)")
axes[0].contour(xk / 1e3, yk / 1e3, columns.KDP_COLUMN_OBJECT > 0, levels=[0.5], colors="k", linewidths=0.8)
axes[0].set(xlabel="x (km)", ylabel="y (km)", aspect="equal", title="KDP columns")
for tid in tracks.cell:
    axes[1].plot(tracks.cell_x.sel(cell=tid) / 1e3, tracks.cell_y.sel(cell=tid) / 1e3, "o-", ms=3, label=f"track {int(tid)}")
axes[1].set(xlabel="x (km)", ylabel="y (km)", aspect="equal", title="cell tracks")
axes[1].legend(frameon=False, fontsize=8)
plt.show()
```

`kdp_column` and `storm_size_sorting` need the grid of `grid_cones` and a KDP from
`estimate_kdp`; the slab, the smoothing and the sizes of the regions are choices of
the user, as in the papers. `track_cells` takes the storm motion from the user or from
`estimate_motion`.

## 22. Ka and W band cloud radar

A cloud radar at 35 or 94 GHz loses signal to oxygen and water vapour, to cloud liquid water and
to rain, and the loss differs between the two bands. The calls below show every function on a
synthetic vertically pointing section with a known answer; the notebooks `Cloud_Radar_Attenuation`, `Cloud_Radar_Classification` and
`Dual_Frequency_Ratio` go into each step, the last two on a day of Ka and W band data of the Cloudnet
station at Lindenberg compared with Cloudnet. The section is a stratiform cloud with a liquid layer at 1.5 to 2 km, a melting layer
at 2.5 km and ice above.

```{code-cell} ipython3
from radarx.retrieve import (
    classify_cloud_targets,
    cloud_liquid_attenuation,
    dfr_rain,
    dfr_snow,
    dual_frequency_ratio,
    gas_path_attenuation,
    gas_specific_attenuation,
    rain_attenuation_relation,
    rain_path_attenuation,
)

height = np.arange(0.0, 12001.0, 250.0)
sounding = xr.Dataset(
    {
        "pressure": ("height", 101325.0 * np.exp(-height / 8000.0)),
        "temperature": ("height", 288.15 - 6.5e-3 * height),
        "specific_humidity": ("height", 0.008 * np.exp(-height / 2500.0)),
    },
    coords={"height": height},
)
gates = np.arange(150.0, 7000.0, 30.0)
section = xr.Dataset(
    {"DBZ": (("time", "range"), np.full((6, gates.size), -10.0))},
    coords={
        "time": np.arange(6),
        "range": gates - 100.0,
        "height": ("range", gates),  # height above sea level of a vertical beam
        "radar_frequency": 94.0,
    },
)
lwc = xr.zeros_like(section.DBZ)
lwc[:, (gates > 1500) & (gates < 2000)] = 0.2  # g m-3, a liquid layer

# specific attenuation of the gases and of liquid water at the two radar frequencies
for f in (35.0, 94.0):
    gas = gas_specific_attenuation(f, 1013.25, 288.15, vapor_density=7.5)
    print(f"{f:.0f} GHz at sea level: gas {float(gas.A_GAS):.2f} dB/km, "
          f"cloud water {float(cloud_liquid_attenuation(f, 273.15)):.2f} dB/km per g/m3")

# two-way attenuation along the beam, and the accessor
pia = gas_path_attenuation(section, sounding, lwc=lwc)
via = section.radarx.gas_path_attenuation(sounding, lwc=lwc)
print("W band, 7 km: gas", round(float(pia.PIA_GAS.isel(time=0, range=-1)), 2), "dB, liquid layer",
      round(float(pia.PIA_CLOUD_LIQUID.isel(time=0, range=-1)), 2), "dB;", bool(np.allclose(via.PIA_GAS, pia.PIA_GAS)))

# classification from the reflectivity, the depolarization ratio and the temperature
dbz = np.where(gates < 2500.0, 5.0, -5.0)
section["DBZ"] = (("time", "range"), np.tile(dbz, (6, 1)))
section["ldr"] = (("time", "range"), np.full((6, gates.size), -28.0))
section["temperature"] = ("range", np.interp(gates, height, sounding.temperature.values))
classes = classify_cloud_targets(section, "temperature", ldr="ldr")
same = section.radarx.classify_cloud_targets("temperature", ldr="ldr")
print({name: int((classes == code).sum()) for name, code in (("drizzle/rain", 2), ("melting", 4), ("ice", 3))}, bool((same == classes).all()))

# dual-frequency ratio and the retrievals of rain and snow
ratio = dual_frequency_ratio(section.DBZ, section.DBZ - 6.0, pia_ka=0.2, pia_w=1.2)
rain = dfr_rain(ratio, section.DBZ)
snow = dfr_snow(ratio, section.DBZ)
print("DFR", float(ratio.isel(time=0, range=0)), "dB; rain Dm", round(float(rain.DM.isel(time=0, range=0)), 2),
      "mm; snow Dm", round(float(snow.DM.isel(time=0, range=-1)), 2), "mm")
print(bool(np.allclose(ratio.radarx.dfr_rain(section.DBZ).DM, rain.DM, equal_nan=True)),
      bool(np.allclose(ratio.radarx.dfr_snow(section.DBZ).IWC, snow.IWC, equal_nan=True)))

# attenuation by rain: k = a Z^b fitted to T-matrix scattering, and the Hitschfeld-Bordan solution
relation = rain_attenuation_relation(35.0)
print(f"35 GHz: k = {float(relation.a):.4g} Z^{float(relation.b):.2f}")
rain_dbz = xr.DataArray(np.full(60, 30.0), dims="range", coords={"range": 100.0 + 30.0 * np.arange(60)})
k = float(relation.a) * (10 ** (30.0 / 10)) ** float(relation.b)
attenuated = rain_dbz - 2 * k * 0.03 * (np.arange(60) + 0.5)
rain_pia = rain_path_attenuation(attenuated, 35.0, a=float(relation.a), b=float(relation.b), max_pia=30.0)
print("two-way rain attenuation at the end:", round(float(rain_pia[-1]), 1), "dB; accessor",
      bool(np.allclose(attenuated.radarx.rain_path_attenuation(a=float(relation.a), b=float(relation.b), max_pia=30.0), rain_pia)))
```

The attenuation of the gases and the cloud liquid water is a few tenths of a dB at Ka band and
a few dB at W band over a few kilometres, and it must be removed before the two reflectivities are
subtracted. The classification is the radar-only subset of the categories of Cloudnet; the
dual-frequency retrievals assume a size distribution (gamma for rain, exponential for snow) and
return NaN where the ratio saturates.

## 23. Level I: IQ time series, clutter filtering and Doppler spectra

Level I data are the complex samples (I and Q) of every gate and pulse of a dwell, and
`radarx.level1` turns them into the radar moments. The data are synthetic, so that the
answer is known: gates with velocities from -12 to 12 m/s, a width of 1.5 m/s, a signal 20 dB
above the noise and ground clutter 30 dB stronger than the weather at zero velocity. The
[Level I notebook](Level_I_Processing) shows the methods in detail, the textbook check of the
estimator variances and a real storm.

```{code-cell} ipython3
from radarx import level1

nyq = 0.1 / (4 * 1e-3)  # wavelength 10 cm, pulse repetition time 1 ms
v_true = np.linspace(-12, 12, 300)
iq = level1.simulate_iq(
    v_true, width=1.5, power=100.0, noise=1.0, clutter_power=1e5, clutter_width=0.2,
    n_pulses=64, wavelength=0.1, prt=1e-3, dims=("range",), seed=2,
)
# the same series from I and Q counts, as a reader gives them (raw files: read_iq)
counts = (iq.transpose("range", "pulse").values * 20).round()
iq = level1.as_iq(counts.real, counts.imag, scale=1 / 20, dims=("range", "pulse"), wavelength=0.1, prt=1e-3)
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "dwell.raw"
    path.write_bytes(np.stack([counts.real, counts.imag], axis=-1).astype("<i2").tobytes())
    same = level1.read_iq(path, shape=counts.shape, dims=("range", "pulse"), scale=1 / 20, wavelength=0.1, prt=1e-3)
print(bool(np.allclose(same, iq)))

far = np.abs(v_true) > 6.0
filtered = level1.regression_filter(iq, order=3)  # the filter in the time domain
spectra = level1.doppler_spectra(iq, window="blackman")
notched = level1.notch_filter(spectra, half_width=3)  # or in the spectrum
for label, moments in (
    ("with clutter, pulse pair", level1.pulse_pair_moments(iq, noise=1.0)),
    ("regression filter, pulse pair", level1.pulse_pair_moments(filtered["IQ"], noise="spectral")),
    ("notch filter, spectral", level1.spectral_moments(notched, noise=level1.spectral_noise(spectra))),
):
    print(f"{label:30s} mean |error| {float(np.abs(moments.VRADH.values[far] - v_true[far]).mean()):5.2f} m/s")
```

Aliased velocities are brought back with the spectra, by continuity along the ray or towards a
reference wind:

```{code-cell} ipython3
ramp = np.linspace(0, 70, 100)
ray = level1.simulate_iq(ramp[None, :], width=2.0, power=300.0, noise=1.0, n_pulses=64,
                         wavelength=0.1, prt=1e-3, dims=("azimuth", "range"), seed=5)
ray_spectra = level1.doppler_spectra(ray)
unfolded = level1.dealias_spectra(ray_spectra)
with_reference = level1.dealias_spectra(ray_spectra, reference=xr.DataArray(ramp + 5.0, dims="range"))
print(f"unfolded within 4 m/s: {float((abs(unfolded.VRADH - ramp) < 4).mean()):.2f}, "
      f"with a reference {float((abs(with_reference.VRADH - ramp) < 4).mean()):.2f}; "
      f"largest fold {int(unfolded.FOLD.max())}")
```

## 24. Nowcasting and verification

A nowcast moves the precipitation of the last few radar volumes forward in time.
`nowcast` estimates the motion (or takes it from `estimate_motion` or
`optical_flow`), splits the field into spatial scales that each follow an
autoregressive model, adds spatially correlated noise to every scale for an
ensemble (S-PROG and STEPS; Seed 2003, Bowler et al. 2006) and advects the
result with the same semi-Lagrangian scheme as `advect`. `optical_flow` gives
the motion by the Lucas-Kanade or the DARTS-type least squares, and the
verification functions score the result: `fss` (Roberts and Lean 2008), `crps`,
`categorical_scores`, `brier_score`, `reliability` and `rank_histogram`. The data
are synthetic: a random echo pattern that moves with 12 m/s to the east and 5 m/s
to the north and changes a little in time, seen in five volumes five minutes apart.

```{code-cell} ipython3
from scipy.ndimage import gaussian_filter, shift

from radarx.retrieve import (
    brier_score,
    categorical_scores,
    crps,
    exceedance_probability,
    fss,
    nowcast,
    optical_flow,
    rank_histogram,
    reliability,
)

rng = np.random.default_rng(7)
size, step_s = 90, 300.0
pattern = gaussian_filter(rng.standard_normal((260, 260)), 5, mode="wrap")
pattern += 0.5 * gaussian_filter(rng.standard_normal((260, 260)), 2, mode="wrap")
pattern = 30.0 + 12.0 * (pattern - pattern.mean()) / pattern.std()
evolving = gaussian_filter(rng.standard_normal((260, 260)), 4, mode="wrap")


def volume(k):
    """Volume k of the moving, slowly changing pattern (12 m/s east, 5 m/s north)."""
    field = pattern + 2.0 * k * evolving
    moved = shift(field, (5.0 * step_s * k / 1000.0, 12.0 * step_s * k / 1000.0), order=3, mode="wrap")
    moved = moved[60 : 60 + size, 60 : 60 + size]
    grid = np.arange(size) * 1000.0
    return xr.Dataset(
        {"DBZH": (("y", "x"), np.where(moved >= 10.0, moved, np.nan).astype("f4"), {"units": "dBZ"})},
        coords={"x": ("x", grid, {"units": "m"}), "y": ("y", grid, {"units": "m"})},
    ).assign(time=np.datetime64("2026-01-01T00:00:00", "ns") + np.timedelta64(int(step_s * k), "s"))


volumes = [volume(k) for k in range(11)]
history, future = volumes[:5], volumes[5:]
```

```{code-cell} ipython3
flow = optical_flow(history, method="lucas_kanade")
darts = optical_flow(history, method="darts", uniform=True)
print(f"Lucas-Kanade: u = {float(flow.u.mean()):.1f}, v = {float(flow.v.mean()):.1f} m/s; "
      f"DARTS: u = {float(darts.u):.1f}, v = {float(darts.v):.1f} m/s")
ens = nowcast(history, 6, n_members=16, motion=flow, seed=1)  # STEPS, 16 members
det = nowcast(history, 6, method="sprog", motion=flow)
via = xr.concat(history, dim="time").radarx.nowcast(6, method="extrapolation")  # a Dataset with a time dimension
flow_acc = xr.concat(history, dim="time").radarx.optical_flow(uniform=True)  # the accessor of a field
print(ens.dims, ens.shape, float(abs(via.isel(lead_time=0)).mean() > 0))
```

```{code-cell} ipython3
truth = xr.concat([f["DBZH"].fillna(5.0).where(lambda a: a >= 10.0, 5.0) for f in future], dim="lead_time")
truth = truth.assign_coords(lead_time=ens.lead_time.values)
prob = exceedance_probability(ens, 30.0)
prob_acc = ens.radarx.exceedance_probability(30.0)  # the accessor gives the same
print("FSS of 30 dBZ at 5 and 20 km, lead times 5 to 30 min:")
for name, fc in (("S-PROG", det), ("STEPS member", ens.isel(member=0))):
    print(f"  {name:12s}", fss(fc, truth, 30.0, [5e3, 20e3]).mean("lead_time").values.round(2))
print("CRPS of the ensemble:", float(crps(ens, truth, dim=["x", "y", "lead_time"])), "dBZ; fair CRPS:",
      float(crps(ens, truth, fair=True, dim=["x", "y", "lead_time"])))
table = categorical_scores(det, truth, 30.0, dim=["x", "y", "lead_time"])
print(f"S-PROG 30 dBZ: POD {float(table.pod):.2f}, FAR {float(table.far):.2f}, CSI {float(table.csi):.2f}")
event = truth >= 30.0
print(f"Brier score of the 30 dBZ probability {float(brier_score(prob, event)):.3f}")
rel = reliability(prob, event, bins=5)
hist = rank_histogram(ens, truth, seed=0)
print(rel.observed_frequency.values.round(2), hist.values)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), layout="constrained")
axes[0].pcolormesh(truth.x / 1e3, truth.y / 1e3, truth.isel(lead_time=-1), cmap="ChaseSpectral", vmin=0, vmax=60)
axes[0].set_title("observed, +30 min")
pm = axes[1].pcolormesh(prob.x / 1e3, prob.y / 1e3, prob.isel(lead_time=-1), cmap="Blues", vmin=0, vmax=1)
axes[1].set_title("P(Z >= 30 dBZ), +30 min")
fig.colorbar(pm, ax=axes[1], shrink=0.8)
axes[2].bar(hist["rank"], hist.values)
axes[2].set(title="rank histogram", xlabel="rank of the observation")
for ax in axes[:2]:
    ax.set(xlabel="x (km)", ylabel="y (km)", aspect="equal")
```

A single member scores below the deterministic S-PROG forecast, as it should: it keeps the small
scales that cannot be predicted, and an ensemble is judged by the CRPS, the Brier score and the
reliability instead. The synthetic pattern changes less in time than the noise that STEPS adds
(the noise is estimated from the frames), so its rank histogram is a dome, too much spread; the
[nowcasting notebook](Nowcasting) runs the methods on the NEXRAD squall line.

`nowcast_model` runs a trained network on the last frames. radarx ships no weights: a model is
registered in `radarx.ml` with its licence and citation. A toy network built here (it adds
1 dB per step to the last frame) stands in for a real one such as RainNet.

```{code-cell} ipython3
import hashlib

import onnx
from onnx import TensorProto, helper

from radarx.retrieve import nowcast_model

n_in = 3
inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", n_in, None, None])
outp = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 1, None, None])
consts = [
    helper.make_tensor("starts", TensorProto.INT64, [1], [-1]),
    helper.make_tensor("ends", TensorProto.INT64, [1], [np.iinfo(np.int64).max]),
    helper.make_tensor("axes", TensorProto.INT64, [1], [1]),
    helper.make_tensor("step", TensorProto.FLOAT, [], [0.1]),  # 0.1 * 10 dB
]
nodes = [helper.make_node("Slice", ["x", "starts", "ends", "axes"], ["last"]),
         helper.make_node("Add", ["last", "step"], ["y"])]
graph = helper.make_graph(nodes, "drift", [inp], [outp], consts)
toy = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
toy.ir_version = 8
onnx.checker.check_model(toy)
toy_path = work / "drift.onnx"
onnx.save(toy, toy_path)
ml.register_model("drift-toy", toy_path, hashlib.sha256(toy_path.read_bytes()).hexdigest(), licence="MIT",
                  citation="radarx documentation example (2026)", task="nowcast")
learned = nowcast_model(history[-3:], "drift-toy", 3, scale=10.0)
base = history[-1].DBZH.values
print(learned.attrs["ml_model"], learned.dims, float((learned.isel(lead_time=2).values - base)[base >= 10].mean()))
```

## 25. Rainfall in context

A depth of rain means more with its return period and with what is usual at the place. `return_period`
reads the return period of a depth from precipitation frequency estimates (the depth for a few return
periods, as in NOAA Atlas 14, interpolated in the logarithm of both; a depth outside the table is flagged,
not extrapolated). The table describes *point* depths, so the mean over an area has to be turned into a
point depth first with an areal reduction factor of your own (`areal_reduction_factor`; radarx ships no
factors, because published curves depend on the region, the duration and the return period).
`climatological_reference` gives a quantile of a record at each place, for example the 99.5th percentile of
the one-hour totals, to compare with `depth / reference`. None of this is a flood forecast.

```{code-cell} ipython3
from radarx.retrieve import areal_reduction_factor, climatological_reference, return_period

rng = np.random.default_rng(4)
shape = (20, 20)
coords = {"y": np.arange(20) * 1000.0, "x": np.arange(20) * 1000.0}
mean_depth = xr.DataArray(40.0 + 20.0 * rng.random(shape), dims=("y", "x"), coords=coords, attrs={"units": "mm"})

# point depths for 2, 10 and 100 years (a made-up table) and a factor for a 300 km2 basin
arf = areal_reduction_factor(300.0, [10, 1000], [0.99, 0.85])
rp_direct = return_period(mean_depth, [2, 10, 100], [50.0, 80.0, 160.0])
rp_point = return_period(mean_depth / arf, [2, 10, 100], [50.0, 80.0, 160.0])
print(f"factor {arf:.2f}: median return period {float(rp_direct.return_period.median()):.1f} years "
      f"without it, {float(rp_point.return_period.median()):.1f} with it")

hourly = xr.DataArray(rng.gamma(0.4, 3.0, (3000, 20, 20)).astype("f4"), dims=("time", "y", "x"), coords=coords,
                      attrs={"units": "mm"})
reference = climatological_reference(hourly, quantile=0.995)
print(f"reference depth {float(reference.mean()):.1f} mm; an hour of 30 mm is {float((30.0 / reference).mean()):.1f} times it")
```

## References

- Feng, Z., J. Hardin, H. C. Barnes, J. Li, L. R. Leung, A. Varble, and Z. Zhang, 2023: PyFLEXTRKR: a flexible feature tracking Python software for convective cloud analysis. *Geoscientific Model Development*, **16**, 2753-2776, <https://doi.org/10.5194/gmd-16-2753-2023>
- Heikenfeld, M., P. J. Marinescu, M. Christensen, D. Watson-Parris, F. Senf, S. C. van den Heever, and P. Stier, 2019: tobac 1.2: towards a flexible framework for tracking and analysis of clouds in diverse datasets. *Geoscientific Model Development*, **12**, 4551-4570, <https://doi.org/10.5194/gmd-12-4551-2019>
- Holleman, I., and H. Beekhuis, 2003: Analysis and Correction of Dual PRF Velocity Data. *Journal of Atmospheric and Oceanic Technology*, **20**, 443-453, <https://doi.org/10.1175/1520-0426(2003)20<443:AACODP>2.0.CO;2>
- Huffman, G. J., et al., 2020: Integrated Multi-satellitE Retrievals for the Global Precipitation Measurement (GPM) Mission (IMERG). *Satellite Precipitation Measurement*, Springer, 343-353, <https://doi.org/10.1007/978-3-030-24568-9_19>
- Gao, J., M. Xue, A. Shapiro, and K. K. Droegemeier, 1999: A Variational Method for the Analysis of Three-Dimensional Wind Fields from Two Doppler Radars. *Monthly Weather Review*, **127**, 2128-2142, <https://doi.org/10.1175/1520-0493(1999)127<2128:AVMFTA>2.0.CO;2>
- Illingworth, A. J., et al., 2007: Cloudnet: Continuous evaluation of cloud profiles in seven operational models using ground-based observations. *Bulletin of the American Meteorological Society*, **88**, 883-898, <https://doi.org/10.1175/BAMS-88-6-883>
- Jung, Y., G. Zhang, and M. Xue, 2008: Assimilation of Simulated Polarimetric Radar Data for a Convective Storm Using the Ensemble Kalman Filter. Part I: Observation Operators for Reflectivity and Polarimetric Variables. *Monthly Weather Review*, **136**, 2228-2245, <https://doi.org/10.1175/2007MWR2083.1>
- Lin, T.-Y., K. Winner, G. Bernstein, A. Mittal, A. M. Dokter, K. G. Horton, C. Nilsson, B. M. Van Doren, A. Farnsworth, F. A. La Sorte, S. Maji, and D. Sheldon, 2019: MistNet: Measuring historical bird migration in the US using archived weather radar data and convolutional neural networks. *Methods in Ecology and Evolution*, **10**, 1908-1922, <https://doi.org/10.1111/2041-210X.13280>
- Lin, Y.-L., R. D. Farley, and H. D. Orville, 1983: Bulk Parameterization of the Snow Field in a Cloud Model. *Journal of Climate and Applied Meteorology*, **22**, 1065-1092, <https://doi.org/10.1175/1520-0450(1983)022<1065:BPOTSF>2.0.CO;2>
- Loeffler, S. D., and M. R. Kumjian, 2018: Quantifying the Separation of Enhanced ZDR and KDP Regions in Nonsupercell Tornadic Storms. *Weather and Forecasting*, **33**, 1143-1157, <https://doi.org/10.1175/WAF-D-18-0011.1>
- Melnikov, V. M., and D. S. Zrnic, 2004: Simultaneous transmission mode for the polarimetric WSR-88D: statistical biases and standard deviations of polarimetric variables. NOAA/NSSL report, 84 pp., <https://www.nssl.noaa.gov/publications/wsr88d_reports/SHV_statistics.pdf>
- Ryzhkov, A. V., J. Snyder, J. T. Carlin, A. Khain, and M. Pinsky, 2020: What Polarimetric Weather Radars Offer to Cloud Modelers: Forward Radar Operators and Microphysical/Thermodynamic Retrievals. *Atmosphere*, **11**, 362, <https://doi.org/10.3390/atmos11040362>
- Ryzhkov, A., M. Diederich, P. Zhang, and C. Simmer, 2014: Potential Utilization of Specific Attenuation for Rainfall Estimation, Mitigation of Partial Beam Blockage, and Radar Networking. *Journal of Atmospheric and Oceanic Technology*, **31**, 599-619, <https://doi.org/10.1175/JTECH-D-13-00038.1>
- Schwaller, M. R., and K. R. Morris, 2011: A Ground Validation Network for the Global Precipitation Measurement Mission. *Journal of Atmospheric and Oceanic Technology*, **28**, 301-319, <https://doi.org/10.1175/2010JTECHA1403.1>
- Seto, S., T. Iguchi, R. Meneghini, J. Awaka, T. Kubota, T. Masaki, and N. Takahashi, 2021: The Precipitation Rate Retrieval Algorithms for the GPM Dual-frequency Precipitation Radar. *Journal of the Meteorological Society of Japan*, **99**, 205-237, <https://doi.org/10.2151/jmsj.2021-011>
- Snyder, J. C., A. V. Ryzhkov, M. R. Kumjian, A. P. Khain, and J. Picca, 2015: A ZDR Column Detection Algorithm to Examine Convective Storm Updrafts. *Weather and Forecasting*, **30**, 1819-1844, <https://doi.org/10.1175/WAF-D-15-0068.1>
- Testud, J., S. Oury, R. A. Black, P. Amayenc, and X. Dou, 2001: The Concept of "Normalized" Distribution to Describe Raindrop Spectra: A Tool for Cloud Physics and Cloud Remote Sensing. *Journal of Applied Meteorology*, **40**, 1118-1140, <https://doi.org/10.1175/1520-0450(2001)040<1118:TCONDT>2.0.CO;2>
- Wilson, M. B., and M. S. Van Den Broeke, 2021: An Automated Python Algorithm to Quantify ZDR Arc and KDP-ZDR Separation Signatures in Supercells. *Journal of Atmospheric and Oceanic Technology*, **38**, 371-386, <https://doi.org/10.1175/JTECH-D-20-0056.1>
- Shapiro, A., 1993: The Use of an Exact Solution of the Navier-Stokes Equations in a Validation Test of a Three-Dimensional Nonhydrostatic Numerical Model. *Monthly Weather Review*, **121**, 2420-2425, <https://doi.org/10.1175/1520-0493(1993)121<2420:TUOAES>2.0.CO;2>
- van Lier-Walqui, M., A. M. Fridlind, A. S. Ackerman, S. Collis, J. Helmus, D. R. MacGorman, K. North, P. Kollias, and D. J. Posselt, 2016: On Polarimetric Radar Signatures of Deep Convection for Model Evaluation: Columns of Specific Differential Phase Observed during MC3E. *Monthly Weather Review*, **144**, 737-758, <https://doi.org/10.1175/MWR-D-15-0100.1>
- Veillette, M. S., J. M. Kurdzo, P. M. Stepanian, J. Y. N. Cho, T. Reis, S. Samsi, J. McDonald, and N. Chisler, 2025: A Benchmark Dataset for Tornado Detection and Prediction Using Full-Resolution Polarimetric Weather Radar Data. *Artificial Intelligence for the Earth Systems*, **4**, <https://doi.org/10.1175/AIES-D-24-0006.1>
- Ziegler, C. L., 2013a: A Diabatic Lagrangian Technique for the Analysis of Convective Storms. Part I: Description and Validation via an Observing System Simulation Experiment. *Journal of Atmospheric and Oceanic Technology*, **30**, 2248-2265, <https://doi.org/10.1175/JTECH-D-12-00194.1>
- Ziegler, C. L., 2013a: A Diabatic Lagrangian Technique for the Analysis of Convective Storms. Part II: Application to a Radar-Observed Storm. *Journal of Atmospheric and Oceanic Technology*, **30**, 2266-2280, <https://doi.org/10.1175/JTECH-D-13-00036.1>
- Goudenhoofdt, E., and L. Delobbe, 2009: Evaluation of radar-gauge merging methods for quantitative precipitation estimates. *Hydrology and Earth System Sciences*, **13**, 195-203, <https://doi.org/10.5194/hess-13-195-2009>
- Sinclair, S., and G. Pegram, 2005: Combining radar and rain gauge rainfall estimates using conditional merging. *Atmospheric Science Letters*, **6**, 19-22, <https://doi.org/10.1002/asl.85>
- Seed, A. W., 2003: A Dynamic and Spatial Scaling Approach to Advection Forecasting. *Journal of Applied Meteorology*, **42**, 381-388, <https://doi.org/10.1175/1520-0450(2003)042<0381:ADASSA>2.0.CO;2>
- Bowler, N. E., C. E. Pierce, and A. W. Seed, 2006: STEPS: A probabilistic precipitation forecasting scheme which merges an extrapolation nowcast with downscaled NWP. *Quarterly Journal of the Royal Meteorological Society*, **132**, 2127-2155, <https://doi.org/10.1256/qj.04.100>
- Roberts, N. M., and H. W. Lean, 2008: Scale-Selective Verification of Rainfall Accumulations from High-Resolution Forecasts of Convective Events. *Monthly Weather Review*, **136**, 78-97, <https://doi.org/10.1175/2007MWR2123.1>
