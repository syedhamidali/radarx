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

# End-to-End Radar Workflow, Part 2: Beyond the Core Pipeline

+++

The [core workflow](Radar_Workflow) follows one squall line from the raw
NEXRAD volumes to the three-dimensional wind. This second part shows the first
half of the rest of radarx, and the [third part](Radar_Workflow_Advanced_2) the
second half, with the same style of one or two sentences, a call with realistic
arguments and a figure or a printed result for every function. Everything here
is fast: the inputs are either small open data sets or synthetic data with a
known answer, so that nothing depends on field-campaign archives that cannot
be redistributed (the formats of those archives are read from small files
written on the fly).

| Section | What it shows | radarx calls |
|---|---|---|
| 1 | radar equation, beam geometry (4/3 Earth and ray tracing through a refractivity profile), Doppler and unit helpers | `radarx.fundamentals`, `radarx.core`, `refractivity`, `beam_path`, `modified_refractivity`, `ducting_layers` |
| 2 | disdrometer spectra, quality control, gamma fits, radar variables | `radarx.io.read_parsivel`, `disdrometer_qc`, `fit_gamma`, `match_radar` |
| 3 | Bayesian DSD retrieval with uncertainty | `dsd_prior`, `forward_grid`, `dsd_bayesian` |
| 4 | raindrop trajectories and size sorting, collisions in a rain shaft, hail and melting snow fall speeds | `rain_trajectories`, `size_sorting`, `surface_dsd`, `rain_shaft`, `rain_shaft_bulk`, `hail_fall_speed`, `melting_snow_fall_speed` |
| 5 | surface stations, profilers, cold pools | `read_sticknet`, `read_pips`, `cold_pool_intensity` |
| 6 | lightning mapping | `read_lma`, `cluster_flashes`, `lightning_jump` |

```{code-cell} ipython3
import math
import tempfile
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
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
    ducting_layers,
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
    modified_refractivity,
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
    rain_shaft_bulk,
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
    terminal_fall_speed,
    to_radar_grid,
    tornado_probability,
    tornet_inputs,
    trajectories,
    trajectory_matched_times,
    vertical_source_distribution,
    vortex_profile,
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

## 1. Radar equation, beam geometry, Doppler and unit helpers

`radarx.fundamentals` holds the textbook relations of radar meteorology as
plain functions on floats and arrays (Rinehart 2010; Rahman 2019): the radar
equation, pulse and antenna parameters, the Doppler dilemma, beam geometry,
Rayleigh scattering, simple reflectivity relations and the polarimetric
variables. `radarx.core.conversion` collects unit conversions, and
`radarx.fundamentals.constants` the constants. Because there are so many
small functions, they are listed here as one table: each row is a
representative call of a WSR-88D-like S-band radar (3 GHz, 1000 Hz PRF,
1.5 $\mu$s pulse, 8.5 m antenna) and the value it returns.

```{code-cell} ipython3
modules = dict(
    constants=constants, system=system, doppler=doppler, principles=principles,
    timing=timing, power=power, beam=beam, geometry=geometry, reflectivity=reflectivity,
    common=common, scattering=scattering, attenuation=attenuation, variables=variables,
    conversion=conversion,
)
calls = """
system.wavelength_from_frequency(3e9)
system.wavelength(3e9)
system.frequency_from_wavelength(0.1)
system.frequency(0.1)
system.pulse_length_from_duration(1.5e-6)
system.pulse_length(1.5e-6)
system.pulse_duration_from_length(450.0)
system.pulse_duration(450.0)
system.antenna_gain(1e5, 1.0)
system.ant_eff_area(45.0, 0.1)
system.size_param(2e-3, 0.1)
system.radar_equation(750e3, 10**4.5, 10**4.5, 0.1, 1.0, 100e3, 1.0)
system.solve_peak_power(1e-13, 10**4.5, 10**4.5, 0.1, 1.0, 100e3, 1.0)
system.power_return_target(750e3, 45.0, 0.1, 1.0, 100e3)
system.radar_const(750e3, 45.0, 1.5e-6, 0.1, 1.0, 1.0, 1.0, 1.0)
doppler.nyquist_velocity(1000.0, 0.1)
doppler.unambiguous_range(1000.0)
doppler.max_frequency(1000.0)
doppler.doppler_frequency_shift(3e9, 30.0)
doppler.dual_prf_velocity(0.1, 750.0, 1000.0)
doppler.doppler_dilemma(3e8, 0.1)
principles.compute_nyquist_velocity(1000.0, 0.1)
principles.compute_doppler_shift(30.0, 0.1)
principles.doppler_frequency_shift(30.0, 0.1)
principles.compute_range_resolution(1.5e-6)
principles.range_resolution(1.5e-6)
principles.round_trip_time(100e3)
principles.snr(1e-12, 1e-14)
principles.compute_snr(1e-12, 1e6, 290.0)
principles.radar_range(750e3, 10**4.5, 0.1, 1.0, 1.0, 1e-13)
timing.compute_duty_cycle(1000.0, 1.5e-6)
timing.compute_prf(1.5e-6, 1.5e-3)
timing.compute_pulse_repetition_interval(1000.0)
timing.compute_max_unambiguous_range(1000.0)
timing.compute_max_unambiguous_velocity(1000.0, 0.1)
timing.compute_blind_range(1.5e-6)
power.compute_average_power(750e3, 1.5e-3)
power.compute_peak_power(1e3, 50.0)
power.compute_min_detectable_signal(1e6, 290.0)
beam.compute_beamwidth(0.1, 8.5)
beam.beamwidth_to_radians(1.0)
beam.azimuthal_resolution(100e3, 1.0)
beam.compute_azimuth_resolution(100e3, 0.01745)
beam.compute_volume_resolution(100e3, 0.01745, 450.0)
beam.volume_resolution(100e3, 1.0, 1.0, 450.0)
geometry.effective_radius()
geometry.k_e_from_refractivity([0.0, 500.0, 1000.0], [320.0, 301.0, 282.0])
geometry.beam_center_height(100e3, 0.5, 30.0)
geometry.half_power_radius(100e3, 0.5)
geometry.sample_volume_gaussian(100e3, 1.0, 1.0, 450.0)
reflectivity.z_to_r_marshall_palmer(40.0)
reflectivity.z_to_r_custom(40.0, 300.0, 1.4)
reflectivity.dbz_attenuation_correction(40.0)
common.dbz_to_z(40.0)
common.dbz_from_z(1e4)
common.linearize_dbz(40.0)
common.z_to_dbz(1e4)
common.ensure_positive(3.0)
common.km_to_m(1.5)
common.kilometers_to_meters(1.5)
common.kilometers_to_si(1.5)
common.km_to_si(1.5)
common.m_to_km(1500.0)
common.meters_to_kilometers(1500.0)
common.si_to_kilometers(1500.0)
common.si_to_km(1500.0)
common.kts_to_mps(50.0)
common.kts_to_si(50.0)
common.mps_to_kts(25.0)
common.mps_to_knots(25.0)
common.si_to_kts(25.0)
scattering.size_parameter(2e-3, 0.1)
scattering.backscatter_cross_section(2e-3, 0.1)
scattering.normalized_backscatter_cross_section(2e-3, 0.1)
scattering.absorption_coefficient(1e-3, 0.1, 8.0 - 2.0j)
scattering.scattering_coefficient(1e-3, 0.1, 8.0 - 2.0j)
scattering.extinction_coefficient(1e-3, 0.1, 8.0 - 2.0j)
attenuation.absorption_coefficient(2e-3, 0.1, 8.0 - 2.0j)
attenuation.scattering_coefficient(2e-3, 0.1, 8.0 - 2.0j)
attenuation.extinction_coefficient(2e-3, 0.1, 8.0 - 2.0j)
attenuation.k_complex(8.0 - 2.0j)
variables.differential_reflectivity(1e4, 8e3)
variables.linear_depolarization_ratio(1e4, 1e2)
variables.circular_depolarization_ratio(1e4, 1e3)
variables.radial_velocity(200.0, 0.1)
variables.reflectivity_factor(1e-12, 1e-14)
conversion.celsius_to_kelvin(10.0)
conversion.celsius_to_si(10.0)
conversion.kelvin_to_celsius(283.15)
conversion.si_to_celsius(283.15)
conversion.fahrenheit_to_kelvin(50.0)
conversion.fahrenheit_to_si(50.0)
conversion.kelvin_to_fahrenheit(283.15)
conversion.si_to_fahrenheit(283.15)
conversion.degrees_to_radians(180.0)
conversion.radians_to_degrees(np.pi)
conversion.feet_to_meters(10000.0)
conversion.meters_to_feet(3048.0)
conversion.meters_to_miles(10000.0)
conversion.miles_to_meters(10.0)
conversion.miles_to_si(10.0)
conversion.si_to_miles(10000.0)
conversion.meters_to_kilometers(1500.0)
conversion.kilometers_to_meters(1.5)
conversion.kilometers_to_si(1.5)
conversion.si_to_kilometers(1500.0)
conversion.knots_to_mps(50.0)
conversion.mps_to_knots(25.0)
conversion.kph_to_mps(90.0)
conversion.mps_to_kph(25.0)
conversion.frequency_to_wavelength(3e9)
conversion.wavelength_to_frequency(0.1)
conversion.hz_to_mhz(3e9)
conversion.hz_to_ghz(3e9)
conversion.mhz_to_hz(3000.0)
conversion.ghz_to_hz(3.0)
conversion.megahertz_to_si(3000.0)
conversion.gigahertz_to_si(3.0)
conversion.si_to_megahertz(3e9)
conversion.si_to_gigahertz(3e9)
conversion.seconds_to_minutes(600.0)
conversion.minutes_to_seconds(10.0)
conversion.minutes_to_si(10.0)
conversion.si_to_minutes(600.0)
conversion.seconds_to_microseconds(1.5e-6)
conversion.microseconds_to_seconds(1.5)
conversion.microseconds_to_si(1.5)
conversion.si_to_microseconds(1.5e-6)
constants.C
constants.K_BOLTZMANN
constants.T_STANDARD
constants.EARTH_RADIUS
constants.EFFECTIVE_RADIUS_4_3
constants.DIELECTRIC_WATER
constants.DIELECTRIC_ICE
constants.DBZ_TO_Z_FACTOR
constants.Z_TO_DBZ_FACTOR
constants.RADAR_BANDS
constants.TYPICAL_BEAMWIDTH
constants.TYPICAL_PULSE_WIDTHS
reflectivity.hitschfeld_bordan_correction(np.array([30.0, 40.0, 40.0]), np.array([1e3, 2e3, 3e3]), a=1e-4, b=0.8)[-1]
""".strip().splitlines()
rows = []
for line in calls:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # dbz_attenuation_correction
        value = eval(line, dict(modules, np=np))
    text = f"{value:.5g}" if isinstance(value, (float, np.floating)) else str(value)
    rows.append((line.split("(")[0], line if "(" in line else "constant", text))
table = pd.DataFrame(rows, columns=["function", "call", "result"])
print(f"{len(table)} calls from {table['function'].str.split('.').str[0].nunique()} modules")
table.iloc[[0, 11, 15, 16, 19, 31, 43, 46, 53, 72, 83]].to_string(index=False)
```

Two of the many relations in one picture: the Doppler dilemma, the trade-off
between the maximum unambiguous range and the Nyquist velocity as the PRF
changes, and the beam centre height with the 4/3 effective Earth radius
that the radar equation of every later step relies on:

```{code-cell} ipython3
prf = np.linspace(300.0, 1400.0, 200)
elevation = [0.5, 1.5, 4.0]
ranges = np.linspace(1e3, 250e3, 200)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
ax1.plot(prf, doppler.unambiguous_range(prf) / 1e3, color=OKABE["blue"], label="unambiguous range")
ax1.set(xlabel="PRF (Hz)", ylabel="unambiguous range (km)")
twin = ax1.twinx()
twin.plot(prf, doppler.nyquist_velocity(prf, 0.1), color=OKABE["orange"], label="Nyquist velocity")
twin.set_ylabel("Nyquist velocity (m/s)")
twin.grid(False)
twin.spines["right"].set_visible(True)
ax1.legend(handles=[ax1.lines[0], twin.lines[0]], loc="upper center", frameon=False)
for el, color in zip(elevation, [OKABE["blue"], OKABE["orange"], OKABE["green"]]):
    ax2.plot(ranges / 1e3, geometry.beam_center_height(ranges, el, 0.0) / 1e3, color=color,
             label=f"{el}°")
ax2.set(xlabel="range (km)", ylabel="beam centre height (km)")
ax2.legend(title="elevation", frameon=False)
plt.show()
```

### Refraction: the beam through a refractivity profile

The 4/3 Earth stands for a refractivity that decreases by 39 N-units per km.
`refractivity` computes $N$ from pressure, temperature and vapour pressure
(Bean and Dutton 1966, Eq. 1.17), `trace_rays` integrates the ray through the
profile (Snell's law in a spherically stratified atmosphere) and gives the true
beam height and ground range, `k_e_from_rays` the effective Earth radius factor
that reproduces a traced beam, and `beam_path`, also a method of a sweep or a
volume, does it for the gates of a sweep. A moist layer under dry air bends
the beam down; here a sounding with such a layer is compared with the 4/3 Earth:

```{code-cell} ipython3
zr = np.arange(0.0, 12001.0, 100.0)
p_r = 101325.0 * (1.0 - 2.25577e-5 * zr) ** 5.25588
t_r = np.maximum(288.15 - 0.0065 * zr, 216.65)
e_r = 0.55 * saturation_vapor_pressure(t_r).values * np.exp(-zr / 1500.0)
e_r = e_r * (1.0 + 0.5 * np.exp(-zr / 300.0))  # a moist layer at the ground
n_r = refractivity(p_r, t_r, e_r).values
ranges_r = np.arange(2e3, 200e3 + 1.0, 2e3)
ray = refraction.trace_rays(ranges_r, 0.5, zr, n_r, radar_height=100.0)
h43 = geometry.beam_center_height(ranges_r, 0.5, 100.0)
k_e = refraction.k_e_from_rays(ranges_r, 0.5, ray.beam_height, 100.0)
print(f"N at the ground {n_r[0]:.0f}; beam height at 150 km {float(ray.beam_height.sel(range=150e3)):.0f} m "
      f"(4/3 Earth: {geometry.beam_center_height(150e3, 0.5, 100.0):.0f} m); k_e of the gates {float(k_e.median()):.3f}")
sweep_r = xr.Dataset(
    {"DBZH": (("azimuth", "range"), np.zeros((4, ranges_r.size), dtype="f4"))},
    coords={"azimuth": np.arange(4) * 90.0 + 45.0, "range": ranges_r,
            "elevation": ("azimuth", np.full(4, 0.5)), "altitude": 100.0},
)
profile_r = xr.Dataset({"refractivity": ("height", n_r)}, coords={"height": zr})
path = beam_path(sweep_r, profile_r)
print("beam_path, accessor:", float(sweep_r.radarx.beam_path(profile_r).beam_height.isel(azimuth=0, range=-1)),
      "m =", float(path.beam_height.isel(azimuth=0, range=-1)), "m")
```

A layer where the refractivity falls faster than 157 N-units per km traps the
beam. `modified_refractivity` adds the curvature term, so that such a layer is
where it decreases with height, and `ducting_layers` (also a method of a
profile) lists the layers with their strength and the elevation below which a
ray from the base is trapped. The numbers of Bean and Dutton (1966) are a
ground-based layer of 100 m with a gradient of -0.5 N-units per m (8.3 mrad)
and one of 30 m (4.5 mrad); a ray at 0.97 of the angle stays in the layer and
one at 1.03 leaves it:

```{code-cell} ipython3
z_d = np.array([0.0, 30.0, 3000.0])
duct = xr.DataArray([330.0, 315.0, 315.0 - 0.04 * 2970.0], dims="height", coords={"height": z_d})
layers = ducting_layers(duct)
theta = float(layers.trapping_angle.isel(layer=0))
print(f"layer {float(layers.layer_base.isel(layer=0)):g}-{float(layers.layer_top.isel(layer=0)):g} m, "
      f"M deficit {float(layers.m_deficit.isel(layer=0)):.1f}, trapping angle {np.radians(theta) * 1e3:.2f} mrad")
print("M at the levels:", modified_refractivity(duct).values.round(1))
r_d = np.arange(0.0, 100e3, 500.0)
for frac in (0.97, 1.03):
    tr = refraction.trace_rays(r_d, frac * theta, z_d, duct.values, radar_height=0.0)
    print(f"{frac} of the angle: highest point {float(tr.beam_height.max()):.0f} m")
print(duct.to_dataset(name="refractivity").radarx.ducting_layers().layer_count.item(), "layer (accessor)")
```

## 2. Disdrometers

A laser disdrometer such as the OTT Parsivel counts the drops that fall
through a laser sheet in 32 size and 32 fall-speed classes. `parsivel_classes`
(an `xarray.Dataset`) and `parsivel_bins` (a dictionary of arrays) give the
class limits. `read_parsivel` reads the telegrams logged by a PIPS or a
data logger into counts on `(time, velocity, diameter)`. The telegrams here
are simulated: 30 minutes of 10 s records of rain whose gamma DSD intensifies
and decays, mixed with splashing drops (small and fast) and slow, large
particles, the artifacts that the quality control should remove.

```{code-cell} ipython3
rng = np.random.default_rng(42)
cls = parsivel_classes()
bins = parsivel_bins()
print({k: np.shape(v) for k, v in bins.items()})
d, dd = cls.diameter.values, cls.bin_width.values
vup = cls.velocity_upper.values
area = 180e-6 * (30.0 - d / 2)  # effective sampling area, m2
vt = np.maximum(9.65 - 10.3 * np.exp(-0.6 * d), 0.0)

ntime = 180
t = np.arange(ntime)
lam = 3.5 - 1.6 * np.exp(-(((t - 70) / 30.0) ** 2))
mu = 1.0 + 2.0 * np.exp(-(((t - 70) / 40.0) ** 2))
n0 = 6000.0 * lam ** (mu + 1) / 3.5 ** (mu + 1) * (0.2 + np.exp(-(((t - 70) / 35.0) ** 2)))

lines = []
for k in range(ntime):
    nd = n0[k] * d ** mu[k] * np.exp(-lam[k] * d)
    nd[d < 0.25] = 0.0
    expected = nd * area * dd * 10.0 * vt  # drops through the beam in 10 s
    counts = np.zeros((32, 32), int)
    for i in np.flatnonzero(expected > 0):
        v = rng.normal(vt[i], 0.08 * vt[i] + 0.2, rng.poisson(expected[i]))
        j = np.clip(np.searchsorted(vup, v), 0, 31)
        np.add.at(counts[:, i], j, 1)
    counts[rng.integers(22, 28), rng.integers(2, 5)] += rng.poisson(15)  # splashing
    counts[rng.integers(0, 4), rng.integers(19, 23)] += rng.poisson(2)  # slow and large
    hms = f"00:{k // 6:02d}:{(k % 6) * 10:02d}"
    head = f"304545;{0:08.3f};0000.00;00.000;00010;14000;{counts.sum():05d};22;11.6;{hms};31.03.2022"
    lines.append(head + ";" + ";".join(f"{c:03d}" for c in counts.ravel()) + ";")
(work / "parsivel_telegrams.txt").write_text("\n".join(lines))

pars = read_parsivel(
    work / "parsivel_telegrams.txt", station="SIM", latitude=33.6, longitude=-101.8, altitude=990.0
)
pars
```

PIPS netCDF files store the same spectra as `VD_matrix`. `read_pips_netcdf`
reads them; here the first three records of the simulation are written in that
layout and read back, and the counts agree with the telegrams:

```{code-cell} ipython3
vd = pars.counts.isel(time=slice(0, 3)).values.astype(float)
vd[vd == 0] = np.nan
xr.Dataset(
    {
        "VD_matrix": (("time", "fallspeed_bin", "diameter_bin"), vd),
        "pcount": ("time", pars.counts.isel(time=slice(0, 3)).sum(("velocity", "diameter")).values),
        "precipintensity": ("time", [0.0, 0.0, 0.0]),
        "sample_interval": ("time", [10.0, 10.0, 10.0]),
        "windspd": ("time", [2.0, 2.0, 2.0]),
    },
    coords={"time": pars.time.values[:3]},
    attrs={"probe_name": "PIPS_SIM", "location": "(33.6, -101.8, 990.0)", "deployment_name": "SIM"},
).to_netcdf(work / "parsivel_combined_sim.nc")
pips_spectra = read_pips_netcdf(work / "parsivel_combined_sim.nc")
print("same counts as the telegrams:", bool((pips_spectra.counts.values == pars.counts.isel(time=slice(0, 3)).values).all()))
```

The velocity-diameter histogram of all records shows the drops along the
fall-speed curve, the splashing drops above it and the slow, large particles
below it. `disdrometer_qc` keeps the band of 60 % around the terminal fall
speed, drops up to 8 mm and removes the two unmeasured classes:

```{code-cell} ipython3
qc = disdrometer_qc(pars)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
for ax, name, title in zip(axes, ("counts", "counts_qc"), ("raw", "quality controlled")):
    total = qc[name].sum("time").where(lambda x: x > 0)
    pm = ax.pcolormesh(
        np.r_[cls.diameter_lower, cls.diameter_upper[-1]],
        np.r_[cls.velocity_lower, cls.velocity_upper[-1]],
        np.log10(total),
        cmap="viridis",
    )
    ax.plot(d, vt, color=OKABE["red"], lw=1.2, label="terminal fall speed")
    ax.set(xlim=(0, 8), ylim=(0, 12), xlabel="diameter (mm)", title=title)
axes[0].set_ylabel("fall speed (m/s)")
axes[0].legend(loc="lower right", frameon=False)
fig.colorbar(pm, ax=axes, label="log$_{10}$ counts", shrink=0.9)
plt.show()
```

`process_disdrometer` (or `ds.radarx.disdrometer()`) runs the quality control,
`number_concentration` ($N(D)$), `dsd_moments`, gamma fits by moments and
truncated moments (`fit_gamma`, `fit_gamma_moments`) and the S-band radar
variables of `radar_from_dsd` in one call. The individual steps can also be
called alone:

```{code-cell} ipython3
out = process_disdrometer(pars, fits=("MM246", "TMM246"), band="S")
out2 = pars.radarx.disdrometer(fits=("MM246",), band="S")  # the accessor gives the same
nd = number_concentration(qc, counts="counts_qc")
moments = dsd_moments(nd)
fit246 = fit_gamma(nd.resample(time="60s").mean(), moments=(2, 4, 6), truncated=False)
fit_mom = fit_gamma_moments(nd.resample(time="60s").mean())
radar = radar_from_dsd(nd.isel(time=slice(60, 90)).mean("time"), band="S")
print("accessor and function agree:", bool(np.allclose(out.DM, out2.DM, equal_nan=True)))
print("variables of dsd_moments:", list(moments.data_vars))
print(f"radar variables of the core of the rain: Z_H {float(radar.DBZH):.1f} dBZ, "
      f"Z_DR {float(radar.ZDR):.2f} dB, K_DP {float(radar.KDP):.2f} deg/km")
```

```{code-cell} ipython3
fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True, constrained_layout=True)
out.ND.where(out.ND > 0).pipe(np.log10).plot(
    ax=axes[0], x="time", y="diameter", cmap="turbo", vmin=0, vmax=4,
    cbar_kwargs={"label": "log$_{10}$ N(D)"},
)
axes[0].set_ylim(0, 6)
axes[0].set_ylabel("diameter (mm)")
out.DBZH.plot(ax=axes[1], color=OKABE["blue"], label="$Z_H$ (T-matrix)")
out.DBZ_RAYLEIGH.plot(ax=axes[1], color=OKABE["orange"], ls="--", label="$Z$ (Rayleigh)")
axes[1].set_ylabel("reflectivity (dBZ)")
axes[1].legend(frameon=False, loc="upper right")
axes[2].plot(out.time, (4 + mu) / lam, color="k", label="true $D_m$")
out.DM.plot(ax=axes[2], color=OKABE["blue"], label="measured $D_m$")
out.DM_TMM246.plot(ax=axes[2], color=OKABE["red"], ls="--", label="$D_m$ of the TMM246 fit")
axes[2].set_ylabel("$D_m$ (mm)")
axes[2].legend(frameon=False, loc="upper right")
for ax in axes:
    ax.set_title("")
for ax in axes[:2]:
    ax.set_xlabel("")
axes[2].set_xlabel("time")
plt.show()
```

`terminal_fall_speed` is the raindrop fall speed that the quality control and
the corrections below use, and `scattering_table` the T-matrix table of
single-drop scattering amplitudes behind `radar_from_dsd`. The correction of
Raupach and Berne (2015) shifts the velocities of each size class onto the
terminal fall speed and rescales $N(D)$; it needs the intensity reported by
the instrument:

```{code-cell} ipython3
pars["rain_rate_instrument"] = out.RAIN_RATE.fillna(0.0)
rb = number_concentration(
    raupach_berne_correction(disdrometer_qc(pars), instrument="parsivel2", counts="counts_qc")
)
table = scattering_table(band="S")
k = (t >= 60) & (t < 90)
truth = (n0[k, None] * d ** mu[k, None] * np.exp(-lam[k, None] * d)).mean(0)
sel = slice(pars.time.values[60], pars.time.values[89])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
ax1.semilogy(d, truth, color="k", lw=2, label="true")
ax1.semilogy(d, out.ND.sel(time=sel).mean("time"), color=OKABE["blue"], label="relative QC")
ax1.semilogy(d, rb.sel(time=sel).mean("time"), color=OKABE["orange"], label="Raupach and Berne")
ax1.set(xlim=(0, 6), ylim=(1, 1e4), xlabel="diameter (mm)", ylabel="N(D) (m$^{-3}$ mm$^{-1}$)")
ax1.legend(frameon=False)
dgrid = np.linspace(0.5, 6, 100)
ax2.plot(dgrid, terminal_fall_speed(dgrid, 1.2), color=OKABE["blue"], label="1.2 kg/m$^3$ (surface)")
ax2.plot(dgrid, terminal_fall_speed(dgrid, 0.7), color=OKABE["red"], label="0.7 kg/m$^3$ (about 4 km)")
ax2.set(xlabel="diameter (mm)", ylabel="terminal fall speed (m/s)")
ax2.legend(frameon=False)
plt.show()
print("scattering_table variables:", list(table.data_vars)[:6], "...")
```

`match_radar` pairs the spectra with the radar gate above the instrument and
`radar_at_location` reads any radar variable above a ground point. Here the
station is placed under the KLBB radar (Lubbock, Texas, open-radar-data), in a
rain shower 25 km north-east of it, with the time of that sweep:

```{code-cell} ipython3
from open_radar_data import DATASETS

klbb = xd.io.open_nexradlevel2_datatree(DATASETS.fetch("KLBB20160601_150025_V06"), sweep=[0])
sweep = klbb["sweep_0"].to_dataset(inherit=False)
sweep["DBZH"] = sweep.DBZH.where(sweep.DBZH > -32.0)
sweep["ZDR"] = sweep.ZDR.where(sweep.ZDR > -12.9)
klbb["sweep_0"] = sweep
sweep = klbb["sweep_0"].to_dataset()
near = sweep.sel(range=slice(15e3, 40e3))
ia, ir = np.unravel_index(int(np.argmax(near.DBZH.fillna(-99).values)), near.DBZH.shape)
az, rg, el = float(near.azimuth[ia]), float(near.range[ir]), float(near.elevation[ia])
re = 4 / 3 * 6371000.0
h = np.sqrt(rg**2 + re**2 + 2 * rg * re * np.sin(np.deg2rad(el))) - re
s = re * np.arcsin(rg * np.cos(np.deg2rad(el)) / (re + h)) / 6371000.0
lat0, lon0 = float(klbb["latitude"]), float(klbb["longitude"])
p0, a = np.deg2rad(lat0), np.deg2rad(az)
lat = np.rad2deg(np.arcsin(np.sin(p0) * np.cos(s) + np.cos(p0) * np.sin(s) * np.cos(a)))
lon = lon0 + np.rad2deg(np.arctan2(np.sin(a) * np.sin(s) * np.cos(p0),
                                   np.cos(s) - np.sin(p0) * np.sin(np.deg2rad(lat))))
t_radar = sweep.time.values[ia]
station = pars.assign_coords(
    latitude=lat, longitude=lon, altitude=float(klbb["altitude"]),
    time=pars.time.values - pars.time.values[70] + t_radar,
)
pairs = match_radar(station, klbb, data_vars=["DBZH", "ZDR"], window="60s", delay="fall")
above = radar_at_location(klbb, lat, lon, float(klbb["altitude"]), data_vars=["DBZH", "ZDR"])
print(f"radar gate above the station: Z_H {above.DBZH.item():.1f} dBZ, "
      f"Z_DR {above.ZDR.item():.2f} dB at {above.beam_height.item():.0f} m")
pairs[["DBZH", "DBZH_disdrometer", "ZDR", "ZDR_disdrometer", "n_records"]].to_dataframe()
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(5.2, 4.8), constrained_layout=True)
azr = np.deg2rad(sweep.azimuth.values)[:, None]
rkm = sweep.range.values[None, :] / 1e3
pm = ax.pcolormesh(
    rkm * np.sin(azr), rkm * np.cos(azr), sweep.DBZH.transpose("azimuth", "range").values,
    cmap="turbo", vmin=-10, vmax=70, shading="auto",
)
ax.plot(rg / 1e3 * np.sin(np.deg2rad(az)), rg / 1e3 * np.cos(np.deg2rad(az)), "k*", ms=13,
        label="disdrometer")
ax.set(xlim=(-60, 60), ylim=(-60, 60), aspect="equal", xlabel="east of KLBB (km)",
       ylabel="north of KLBB (km)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
fig.colorbar(pm, label="$Z_H$ (dBZ)")
plt.show()
```

## 3. Bayesian DSD retrieval with uncertainty

`dsd` (step 7 of the core workflow) returns one drop size distribution per
gate and treats $Z_H$, $Z_{DR}$ and $K_{DP}$ as exact. `dsd_bayesian` returns
the posterior distribution of the normalized gamma parameters $(\log_{10} N_w,
D_m, \mu)$ instead, with the measurement errors and a prior of disdrometer DSDs.
`dsd_prior` builds that prior on a $(D_m, \mu)$ grid (the `"perils2022"` prior was learned from the PIPS spectra of Dawson et al. 2025), and `forward_grid` is the
forward model, the radar variables of every node of the grid from the T-matrix
tables.

```{code-cell} ipython3
priors = {name: dsd_prior(name) for name in ("generic", "perils2022")}
fw = forward_grid("S")
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
for ax, (name, p) in zip(axes, priors.items()):
    p.prior_mass.T.plot(ax=ax, cmap="Blues", add_colorbar=False)
    ax.set(title=f"prior {name!r}", xlabel="$D_m$ (mm)", ylabel="$\\mu$")
pm = fw.ZDR.T.plot(ax=axes[2], cmap="viridis", add_colorbar=False)
axes[2].set(title="forward model: $Z_{DR}$ (dB)", xlabel="$D_m$ (mm)", ylabel="$\\mu$")
fig.colorbar(pm, ax=axes[2], label="dB")
plt.show()
```

To check that the uncertainty means what it says, draw DSDs from the prior,
simulate $Z_H$, $Z_{DR}$ and $K_{DP}$ with the forward model and the default
error model, and retrieve them: the true values fall inside the 68 % and 95 %
credible intervals about 68 % and 95 % of the time.

```{code-cell} ipython3
from radarx.retrieve import dsd_bayes

rng = np.random.default_rng(1)
prior = priors["generic"]
e = dsd_bayes.ERRORS
n = 3000
j = rng.choice(prior.prior_mass.size, n, p=prior.prior_mass.values.ravel())
lognw = rng.normal(prior.log10_nw_mean.values.ravel()[j], prior.log10_nw_sd.values.ravel()[j])
kdp = 10**lognw * fw.KDP.values.ravel()[j]
sim = xr.Dataset(
    {
        "DBZH": ("gate", 10 * lognw + fw.DBZH.values.ravel()[j]
                 + rng.normal(0, np.hypot(e["zh"], e["zh_bias"]), n)),
        "ZDR": ("gate", fw.ZDR.values.ravel()[j]
                + rng.normal(0, np.hypot(e["zdr"], e["zdr_bias"]), n)),
        "KDP": ("gate", kdp + rng.normal(0, 1, n) * np.hypot(e["kdp"], e["kdp_rel"] * kdp)),
    }
)
truth = {"LOG10_NW": lognw, "DM": fw.dm.values[j // fw.sizes["mu"]], "MU": fw.mu.values[j % fw.sizes["mu"]]}
post = dsd_bayesian(sim, band="S", kdp="KDP")
for name, x in truth.items():
    q = post[name + "_QUANTILES"]
    c68 = np.mean((x >= q.sel(quantile=0.16)) & (x <= q.sel(quantile=0.84)))
    c95 = np.mean((x >= q.sel(quantile=0.025)) & (x <= q.sel(quantile=0.975)))
    print(f"{name:9s} RMSE {float(np.sqrt(np.mean((post[name] - x) ** 2))):.2f}   "
          f"coverage 68 %: {c68:.2f}   95 %: {c95:.2f}")
```

`dsd_spectrum` rebuilds $N(D)$ from gamma parameters, for example from the
Bayesian posterior mean of one gate compared with the truth that generated it:

```{code-cell} ipython3
k = int(np.argmax(sim.DBZH.values))
mean_dsd = post.isel(gate=k)
d_axis = np.linspace(0.3, 7.0, 120)
dm_t, mu_t, nw_t = truth["DM"][k], truth["MU"][k], 10 ** truth["LOG10_NW"][k]
lam_t = (4.0 + mu_t) / dm_t
n0_t = nw_t * 6.0 / 4.0**4 * (4.0 + mu_t) ** (mu_t + 4.0) / math.gamma(mu_t + 4.0) / dm_t**mu_t
params_true = xr.Dataset({"N0": n0_t, "MU": mu_t, "LAMBDA": lam_t})
lam_p = (4.0 + float(mean_dsd.MU)) / float(mean_dsd.DM)
mu_p, dm_p = float(mean_dsd.MU), float(mean_dsd.DM)
n0_p = 10 ** float(mean_dsd.LOG10_NW) * 6.0 / 4.0**4 * (4.0 + mu_p) ** (mu_p + 4.0) / math.gamma(mu_p + 4.0) / dm_p**mu_p
params_post = xr.Dataset({"N0": n0_p, "MU": float(mean_dsd.MU), "LAMBDA": lam_p})
fig, ax = plt.subplots(figsize=(5.5, 3.6), constrained_layout=True)
ax.semilogy(d_axis, dsd_spectrum(params_true, d_axis), color="k", lw=2, label="truth")
ax.semilogy(d_axis, dsd_spectrum(params_post, d_axis), color=OKABE["orange"], label="posterior mean")
ax.set(xlabel="diameter (mm)", ylabel="N(D) (m$^{-3}$ mm$^{-1}$)", ylim=(1e-1, None))
ax.legend(frameon=False)
plt.show()
```

## 4. Raindrop trajectories and size sorting

Large drops fall fast and small drops are blown further by the wind, so the
drop size distribution at the ground differs from the one at the radar beam
(Kumjian and Ryzhkov 2012). `rain_trajectories` follows one drop per source
point and size to the ground in a sheared wind with evaporation. The
environment is a synthetic sounding with 20 m/s of shear over 3 km and dry
air near the ground; `terminal_fall_speed` is the fall speed the drops use
and `drop_evaporation_rate` their mass loss.

```{code-cell} ipython3
def environment(z, u, v):
    temperature = 297.0 - 6.5e-3 * z
    return xr.Dataset(
        {
            "temperature": ("height", temperature),
            "pressure": ("height", 100500.0 * (temperature / 297.0) ** 5.2559),
            "relative_humidity": ("height", 0.97 - 0.42 * np.exp(-z / 1500.0)),
            "u": ("height", u),
            "v": ("height", v),
        },
        coords={"height": z},
    )


z = np.arange(0.0, 6001.0, 50.0)
profile = environment(z, 2.0 + 24.0 * (1.0 - np.exp(-z / 2500.0)), 1.0 + 10.0 * (1.0 - np.exp(-z / 2500.0)))
motion = (6.0, 3.0)  # storm motion (u, v), m/s
diameters = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
traj = rain_trajectories(
    {"x": 0.0, "y": 0.0, "z": 3000.0}, diameters, profile=profile, storm_motion=motion, store_path=2
).squeeze()
sort = size_sorting(traj, 2.0).squeeze()
rate = drop_evaporation_rate(xr.DataArray(diameters, dims="diameter"), 293.15, 9.0e4, 0.008)
state = {1: "reaches the ground", 2: "evaporates"}
pd.DataFrame(
    {
        "fall time (s)": traj.fall_time.values.round(0),
        "mass evaporated (%)": (100 * traj.evaporated_mass_fraction.values).round(1),
        "arrival after the 2 mm drop (s)": sort.arrival_offset.values.round(0),
        "terminal speed (m/s)": terminal_fall_speed(diameters, 1.2).values.round(2),
        "evaporation (-dD/dt, um/s)": (-rate.diameter_rate.values * 1e3).round(2),
        "outcome": [state.get(int(s), "aloft") for s in traj.status.values],
    },
    index=pd.Index(diameters, name="diameter (mm)"),
)
```

The same trajectories from the accessor of a Dataset of source points:

```{code-cell} ipython3
source_ds = xr.Dataset({"x": 0.0, "y": 0.0, "z": 3000.0})
traj_acc = source_ds.radarx.rain_trajectories(
    diameters, profile=profile, storm_motion=motion, store_path=2
).squeeze()
print("accessor and function land at the same place:",
      bool(np.allclose(traj_acc.landing_x, traj.landing_x, equal_nan=True)))
```

A rain cell moving with the storm, observed aloft on a 2 km grid every minute,
is followed to the ground, and `surface_dsd` collects the drops that land on a
surface grid with number-flux conservation. The small drops are blown farther
and arrive later, which raises $D_m$ at the edge of the cell:

```{code-cell} ipython3
from scipy.special import gammaln

x = np.arange(-24e3, 24.1e3, 2e3)
times = np.arange(0.0, 1801.0, 60.0)
sizes = np.arange(0.5, 6.01, 0.25)
width = 0.25
t_, y_, x_ = np.meshgrid(times, x, x, indexing="ij")
core = np.exp(-(((x_ + 12e3 - motion[0] * t_) ** 2 + (y_ + 9e3 - motion[1] * t_) ** 2) / 5e3**2))
dm_src, nw_src, mu0 = 1.8 + 1.2 * core, 8000.0 * core, 3.0
f = 6.0 / 4.0**4 * np.exp((mu0 + 4.0) * np.log(4.0 + mu0) - gammaln(mu0 + 4.0))
s = sizes / dm_src[..., None]
nd_src = xr.DataArray(
    nw_src[..., None] * f * s**mu0 * np.exp(-(4.0 + mu0) * s),
    dims=("time", "y", "x", "diameter"),
    coords={"time": times, "y": x, "x": x, "diameter": sizes},
    name="ND",
    attrs={"units": "m-3 mm-1"},
)
cell = rain_trajectories(
    xr.Dataset({"z": ((), 2500.0)}, coords={"time": times, "y": x, "x": x}),
    sizes, profile=profile, storm_motion=motion, time_step=10.0, max_time=1500.0,
)
sfc = surface_dsd(
    cell, nd_src, x=np.arange(-30e3, 45.1e3, 2e3), y=np.arange(-30e3, 45.1e3, 2e3),
    time=np.arange(0.0, 2700.0, 60.0), duration=60.0,
)
m3 = (nd_src * sizes**3 * width).sum("diameter")
dm_aloft = ((nd_src * sizes**4 * width).sum("diameter") / m3).sel(time=600.0).where(m3.sel(time=600.0) > 1e-3)
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
kw = dict(x="x", y="y", add_colorbar=False, vmin=1.8, vmax=3.0, cmap="viridis")
im = dm_aloft.assign_coords(x=dm_aloft.x / 1e3, y=dm_aloft.y / 1e3).plot(ax=axes[0], **kw)
g = sfc.DM.sel(time=960.0).where(sfc.NT.sel(time=960.0) > 1.0)
g.assign_coords(x=sfc.x / 1e3, y=sfc.y / 1e3).plot(ax=axes[1], **kw)
fig.colorbar(im, ax=axes, location="bottom", shrink=0.6, label="$D_m$ (mm)")
axes[0].set_title("$D_m$ at 2.5 km, t = 10 min")
axes[1].set_title("$D_m$ at the ground, t = 16 min")
for ax in axes:
    ax.set(xlabel="x (km)", ylabel="y (km)", aspect="equal", xlim=(-25, 40), ylim=(-30, 25))
plt.show()
print(f"{float((cell.status == 1).mean()) * 100:.0f} % of the drops reach the ground, "
      f"{float((cell.status == 2).mean()) * 100:.0f} % evaporate")
```

For a disdrometer the question is reversed: where were the drops aloft?
`rain_source_points` integrates backward in time from a site and, given the
DSD aloft, returns the spectrum at the site. `trajectory_matched_times` gives
the time at which drops of each size released from a moving echo pattern
arrive, the construction that pairs a radar gate with disdrometer spectra:

```{code-cell} ipython3
site = {"x": -2e3, "y": -6e3}
arrival = np.arange(900.0, 2401.0, 60.0)
back = rain_source_points(
    xr.Dataset({"x": ((), site["x"]), "y": ((), site["y"]), "z": ((), 0.0)}, coords={"time": arrival}),
    sizes, source_height=2500.0, profile=profile, storm_motion=motion, source_dsd=nd_src, time_step=10.0,
)
fwd = sfc.ND.sel(x=site["x"], y=site["y"], time=arrival)
# a gate at 2.5 km whose 2 mm drops land at the site of the disdrometer
ref = rain_trajectories({"x": 0.0, "y": 0.0, "z": 2500.0}, [2.0], profile=profile,
                        storm_motion=motion, evaporation=False).squeeze()
match_sizes = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
found = trajectory_matched_times(
    {"x": 0.0, "y": 0.0, "z": 2500.0, "time": 0.0},
    {"x": float(ref.landing_x), "y": float(ref.landing_y)},
    match_sizes, storm_motion=motion, profile=profile, evaporation=False,
)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
groups = {"1 to 2 mm": (1.0, 2.0), "2 to 3.5 mm": (2.0, 3.5), "3.5 to 6 mm": (3.5, 6.1)}
for (name, (lo, hi)), c in zip(groups.items(), (OKABE["orange"], OKABE["blue"], OKABE["red"])):
    sel = (sizes >= lo) & (sizes < hi)
    bk = (back.ND.transpose("time", "diameter").isel(diameter=sel) * width).sum("diameter")
    fw_ = (fwd.isel(diameter=sel) * width).sum("diameter")
    ax1.plot(arrival / 60, bk / bk.max(), color=c, label=f"backward, {name}")
    ax1.plot(arrival / 60, fw_ / bk.max(), "o", color=c, ms=3.5)
ax1.set(xlabel="time (min)", ylabel="concentration / maximum (dots: forward)")
ax1.legend(frameon=False, fontsize=8)
ax2.plot(match_sizes, (found.arrival_time.values.ravel() - float(ref.fall_time)) / 60.0, "o-",
         color=OKABE["blue"])
ax2.axhline(0, color="0.4", lw=0.8)
ax2.set(xlabel="drop diameter (mm)", ylabel="arrival relative to the 2 mm drops (min)")
plt.show()
```

### Collisions in a rain shaft, and hail and melting snow

The trajectories cannot let drop sizes collide. `rain_shaft` evolves the DSD of
a column of still air with collisions, coalescence and breakup (the scheme of
Low and List 1982, with the geometric collision kernel) and conserves the rain
rate; with coalescence only, the drops grow and become few, with breakup the
large drops that form break up again:

```{code-cell} ipython3
d_shaft = np.geomspace(0.2, 7.0, 40)
nd_shaft = xr.DataArray(8000.0 * np.exp(-4.1 * 50.0**-0.21 * d_shaft), dims="diameter",
                        coords={"diameter": ("diameter", d_shaft, {"units": "mm"})})
runs = {name: rain_shaft(nd_shaft, depth=[0.0, 2000.0], collisions=name)
        for name in ("none", "coalescence", "low_list1982")}
for name, out in runs.items():
    print(f"{name:13s} drops {float(out.number_concentration.isel(depth=0)):7.0f} -> "
          f"{float(out.number_concentration.isel(depth=1)):7.0f} m-3, rain rate "
          f"{float(out.rain_rate.isel(depth=1)):.3f} mm/h")
via = nd_shaft.radarx.rain_shaft(depth=2000.0)  # the accessor of a DSD
```

`rain_shaft_bulk` is the two-moment counterpart of Seifert (2008): it carries
the number and the water content of the rain, with self-collection and a
breakup that relaxes the mean volume diameter toward 1.1 mm, so that it is
cheap enough for every gate. It takes the output of `rain_shaft` (or any
dataset with `number_concentration` and `liquid_water_content`):

```{code-cell} ipython3
top = runs["low_list1982"].isel(depth=0)[["number_concentration", "liquid_water_content"]]
bulk = rain_shaft_bulk(top, depth=[0.0, 2000.0, 8000.0])
print("mean volume diameter (mm):", bulk.mean_volume_diameter.values.round(2),
      "shape parameter:", bulk.shape_parameter.values.round(1))
via_bulk = top.radarx.rain_shaft_bulk(depth=2000.0)  # the accessor of a dataset
```

Above and in the melting layer the fall speeds are other laws.
`hail_fall_speed` is the size relation of Heymsfield et al. (2020) for the
maximum dimension of a stone, and `melting_snow_fall_speed` mixes the dry-snow
and raindrop speeds with the weight of Frick et al. (2013) (the dry-snow law has
to be given: the one here is an example). Both are also `fall_speed` options of
`rain_trajectories`:

```{code-cell} ipython3
stones = np.array([8.0, 15.0, 30.0, 60.0])
print("hail:", ", ".join(f"{d:g} mm {v:.1f} m/s" for d, v in zip(stones, hail_fall_speed(stones).values)))
print("percentiles at 30 mm:", [float(hail_fall_speed(30.0, f"heymsfield2020_p{p}")) for p in (10, 50, 90)])
print("Milbrandt and Yau (2005) law, 30 mm, at the ground and at 6 km:",
      [round(float(hail_fall_speed(30.0, "milbrandt_yau2005", air_density=rho)), 1) for rho in (1.204, 0.66)])
liquid = xr.DataArray([0.0, 0.3, 0.6, 0.9, 1.0], dims="liquid_fraction")
print("2 mm melting snow:", melting_snow_fall_speed(2.0, liquid, snow=(1.0, 0.3)).values.round(2), "m/s")
print("accessors:", xr.DataArray(stones, dims="d").radarx.hail_fall_speed().values.round(1),
      xr.DataArray([1.0, 2.0], dims="d").radarx.melting_snow_fall_speed(0.5, snow=(1.0, 0.3)).values.round(2))
stone = rain_trajectories({"x": 0.0, "y": 0.0, "z": 4000.0}, stones, profile=profile,
                          storm_motion=motion, fall_speed=("hail", "heymsfield2020")).squeeze()
print("hail landing distance (km):", (np.hypot(stone.landing_x, stone.landing_y).values / 1e3).round(1))
```

## 5. Surface stations, profilers and cold pools

A squall line lives off the balance between its cold pool and the low-level
shear (Rotunno et al. 1988). radarx reads surface station networks and
profilers into xarray and derives the cold pool from them. The field-campaign
archives cannot be redistributed, so small files in the same formats are
written here: two StickNet stations (comma-separated text), two PIPS stations
(netCDF) that feel the gust front 15 minutes apart, a micro rain radar and a
915 MHz wind profiler.

```{code-cell} ipython3
(work / "IOP2_StickNet_Locations.csv").write_text(
    "ID,Latitude,Longitude,Elevation,Array_Type\n101A,33.80,-88.70,90.0,Coarse\n102A,33.80,-88.50,80.0,Coarse\n"
)
times = np.arange("2022-03-30T23:00", "2022-03-31T01:00", 1, dtype="datetime64[s]")
minutes = (times - times[0]).astype(float) / 60


def front(arrival):
    cold = np.clip((minutes - arrival) / 5.0, 0, 1)  # 5-min ramp at the gust front
    return cold, 22.0 - 0.01 * minutes - 7.0 * cold, 75.0 + 20.0 * cold, 990.0 + 2.5 * cold


for sid, arrival in (("0101A", 50.0), ("0102A", 65.0)):
    cold, t_c, rh, p = front(arrival)
    ws = 5.0 + 10.0 * np.exp(-(((minutes - arrival - 3) / 3) ** 2))
    rows = [f"{str(ti).replace('T', ' ')},{a:.2f},{b:.1f},{c:.2f},{d:.1f},{e:.0f}"
            for ti, a, b, c, d, e in zip(times, t_c, rh, p, ws, 170.0 + 100.0 * cold)]
    (work / f"{sid}_IOP2_level3.txt").write_text("Time,T,RH,P,WS,WD\n" + "\n".join(rows))

pips_files = []
for name, loc, arrival in (("PIPS1A", "(33.75, -88.45, 70.7)", 55.0), ("PIPS2A", "(33.70, -88.40, 68.0)", 70.0)):
    cold, t_c, rh, p = front(arrival)
    xr.Dataset(
        {"fasttemp": ("time", t_c), "slowtemp": ("time", t_c + 0.2), "RH": ("time", rh),
         "pressure": ("time", p), "windspd": ("time", np.full(times.size, 4.0)),
         "winddirabs": ("time", np.full(times.size, 180.0))},
        coords={"time": times.astype("datetime64[ns]")},
        attrs={"probe_name": name, "location": loc, "deployment_name": "IOP2_033022"},
    ).to_netcdf(work / f"conventional_raw_{name}.nc")
    pips_files.append(work / f"conventional_raw_{name}.nc")

locations = read_sticknet_locations(work / "IOP2_StickNet_Locations.csv")
print(locations.to_dataframe().to_string() if hasattr(locations, "to_dataframe") else locations)
stick = read_sticknet(work, iop=2)
pips = read_pips(pips_files)
net = xr.concat([stick, pips], dim="station", join="outer")
print("stations:", list(net.station.values), "| variables:", list(net.data_vars)[:7])
```

`potential_temperatures` gives the mixing ratio and the potential, virtual
potential and equivalent potential temperatures of the network, and
`cold_pool_perturbation` subtracts a pre-storm reference (here the mean of the
first half hour) and gives the buoyancy
$B = g\,\Delta\theta_v / \overline{\theta_v}$:

```{code-cell} ipython3
thermo = potential_temperatures(net)
thermo_acc = net.radarx.potential_temperatures()  # the accessor of a station network
print("variables of potential_temperatures:", [v for v in thermo.data_vars if v not in net.data_vars][:5])
pert = cold_pool_perturbation(net, slice("2022-03-30T23:00", "2022-03-30T23:30"))
fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True, constrained_layout=True)
colors = [OKABE["blue"], OKABE["orange"], OKABE["green"], OKABE["red"]]
for s, c in zip(pert.station.values, colors):
    ax[0].plot(pert.time, pert.virtual_potential_temperature_perturbation.sel(station=s), color=c, label=str(s))
    ax[1].plot(pert.time, pert.pressure_perturbation.sel(station=s) / 100, color=c)
ax[0].set_ylabel(r"$\Delta\theta_v$ (K)")
ax[1].set_ylabel(r"$\Delta p$ (hPa)")
ax[0].legend(ncol=4, frameon=False, loc="center left")
fig.autofmt_xdate()
plt.show()
```

The cold-pool intensity $C = \sqrt{2\int_0^H (-B)\,dz}$ needs a buoyancy
profile (`cold_pool_intensity`); from a surface station alone,
`cold_pool_intensity_from_pressure` uses the hydrostatic pressure rise and
`cold_pool_intensity_from_surface` an assumed depth. `rkw_ratio` divides the
intensity by the low-level shear of a profiler (step 3 of the core workflow
uses a sounding for the same purpose). The wind profiler gives the shear and
the Micro Rain Radar the rain above the cold pool:

```{code-cell} ipython3
nt, nh = 3, 12
height = 126.0 + 250.0 * np.arange(nh)
u_prof = np.tile(np.linspace(2.0, 14.0, nh), (nt, 1))
xr.Dataset(
    {"epochTime": ("time", 1648652400.0 + 300 * np.arange(nt)),
     "u": (("time", "height"), u_prof), "v": (("time", "height"), np.zeros((nt, nh))),
     "w": (("time", "height"), np.zeros((nt, nh))), "qcTag": (("time", "height"), np.full((nt, nh), 5.0))},
    coords={"height": height, "latitude": ("latitude", [33.6]), "longitude": ("longitude", [-88.99]),
            "altitude": ("altitude", [87.0])},
    attrs={"System": "RWP"},
).to_netcdf(work / "rwp.nc")
rwp = read_wind_profiler(work / "rwp.nc", min_qc=1)

ng = 15
gates = np.tile(np.arange(1, ng + 1) * 150.0, (6, 1))
mrr_raw = xr.Dataset(
    {"MRR rangegate": (("time", "MRR rangegate"), gates),
     "MRR_Capital_Z": (("time", "MRR rangegate"), np.tile(35.0 - np.arange(ng), (6, 1))),
     "MRR_W": (("time", "MRR rangegate"), np.full((6, ng), 6.0)),
     "MRR_RR": (("time", "MRR rangegate"), np.tile(8.0 - 0.4 * np.arange(ng), (6, 1)))},
    coords={"time": ("time", 1648651350.0 + 60 * np.arange(6))},
    attrs={"system": "MRR"},
)
mrr_raw.time.attrs["units"] = "seconds since 1970-01-01 00:00:00"
mrr_raw.to_netcdf(work / "mrr.nc")
mrr = read_mrr(work / "mrr.nc", latitude=33.6, longitude=-89.0, altitude=87.0)

shear = bulk_shear(rwp, 0, 2500, ground=rwp.altitude + 126.0)
# a post-storm buoyancy profile decreasing linearly from -0.2 m/s2 at the ground to zero at 2.5 km
agl = np.arange(0.0, 6000.0, 50.0)
b_profile = xr.DataArray(-0.2 * np.clip(1 - agl / 2500.0, 0, None), dims="height", coords={"height": agl})
cp = cold_pool_intensity(b_profile)
c = float(cp.cold_pool_intensity)
assert float(b_profile.radarx.cold_pool_intensity(dim="height").cold_pool_intensity) == c
after = pert.sel(time=slice("2022-03-31T00:30", None)).mean("time")
c_p = cold_pool_intensity_from_pressure(after.pressure_perturbation, density=1.15)
c_b = cold_pool_intensity_from_surface(after.buoyancy, depth=2000.0)
du = float(shear.shear_speed.isel(time=0))
print(f"profile: C = {c:.1f} m/s, depth {float(cp.cold_pool_depth):.0f} m; "
      f"RKW ratio C/du = {float(rkw_ratio(c, du)):.2f} (du = {du:.1f} m/s)")
print(f"surface estimates, mean of the stations: from pressure {float(c_p.mean()):.1f} m/s, "
      f"from surface buoyancy {float(c_b.mean()):.1f} m/s")
fig, ax = plt.subplots(1, 3, figsize=(10, 3.8), constrained_layout=True, sharey=True)
ax[0].plot(b_profile, agl / 1e3, color=OKABE["blue"])
ax[0].set(xlabel="buoyancy (m/s$^2$)", ylabel="height above ground (km)", title="cold pool", xticks=[-0.2, -0.1, 0])
ax[1].plot(mrr.DBZ.isel(time=0), mrr.height_agl / 1e3, color=OKABE["orange"])
ax[1].set(xlabel="MRR reflectivity (dBZ)", title="rain above the station")
ax[2].plot(rwp.u.isel(time=0), (rwp.height - rwp.altitude) / 1e3, color=OKABE["green"])
ax[2].set(xlabel="profiler $u$ (m/s)", title="wind")
ax[0].set_ylim(0, 3.5)
plt.show()
```

Horizontal buoyancy gradients generate horizontal vorticity,
$d\xi/dt = \partial B/\partial y$ and $d\eta/dt = -\partial B/\partial x$.
`buoyancy` turns a temperature field into $B$ and `baroclinic_generation`
gives the generation and its streamwise part relative to the storm-relative
wind, here for a synthetic cold pool under a southerly inflow:

```{code-cell} ipython3
gx = np.arange(-40e3, 40001, 1000.0)
X, Y = np.meshgrid(gx, gx)
temperature = xr.DataArray(293.0 - 5.0 * np.exp(-((X / 15e3) ** 2 + (Y / 25e3) ** 2)),
                           dims=("y", "x"), coords={"y": gx, "x": gx}, name="temperature")
b_grid = buoyancy(temperature, 293.0)
gen = baroclinic_generation(
    b_grid, u=xr.full_like(temperature, 8.8), v=xr.full_like(temperature, 10.0), storm_motion=(8.8, -3.2)
)
gen_acc = b_grid.radarx.baroclinic_generation(
    u=xr.full_like(temperature, 8.8), v=xr.full_like(temperature, 10.0), storm_motion=(8.8, -3.2)
)
print("accessor and function agree:",
      bool(np.allclose(gen_acc.horizontal_vorticity_generation, gen.horizontal_vorticity_generation)))
fig, ax = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True, constrained_layout=True)
km = gx / 1e3
pm0 = ax[0].pcolormesh(km, km, gen.horizontal_vorticity_generation * 1e6, cmap="magma")
ax[0].quiver(km[::6], km[::6], gen.vorticity_x_generation[::6, ::6], gen.vorticity_y_generation[::6, ::6], color="w")
pm1 = ax[1].pcolormesh(km, km, gen.streamwise_vorticity_generation * 1e6, cmap="RdBu_r")
fig.colorbar(pm0, ax=ax[0], label="$10^{-6}$ s$^{-2}$")
fig.colorbar(pm1, ax=ax[1], label="$10^{-6}$ s$^{-2}$")
ax[0].set_title("generation of horizontal vorticity")
ax[1].set_title("streamwise part")
for a in ax:
    a.set(aspect="equal", xlabel="x (km)")
ax[0].set_ylabel("y (km)")
plt.show()
```

## 6. Lightning mapping

A Lightning Mapping Array (LMA; Rison et al. 1999) locates the VHF radiation
of lightning in three dimensions, typically thousands of "sources" per flash.
The observed LMA archives cannot be redistributed, so a storm is simulated: a
cell moving east at 15 m/s whose flash rate sits near 12 flashes per minute
for 20 min and then jumps to about 35 per minute, each flash a cloud of 10 to
60 sources 1 to 2 km across between 5 and 12 km, and a weak second cell with 3
flashes per minute. The sources are written in the ASCII format of the
`lma_analysis` program and read back with `read_lma`.

```{code-cell} ipython3
rng = np.random.default_rng(42)
lat0, lon0 = 33.6, -88.5
start = np.datetime64("2022-03-30T23:00:00", "ns")
minutes_total = 40


def storm(rate_per_min, x0, y0, u, n_minutes):
    t, lat, lon, alt = [], [], [], []
    for m, rate in zip(range(n_minutes), rate_per_min):
        for tf in np.sort(rng.uniform(60 * m, 60 * (m + 1), rng.poisson(rate))):
            n = rng.integers(10, 60)
            xf = x0 + u * tf + rng.normal(0, 2e3)
            yf = y0 + rng.normal(0, 2e3)
            t.append(tf + np.sort(rng.uniform(0, 0.3, n)))
            lat.append(lat0 + (yf + rng.normal(0, 700, n)) / 111.2e3)
            lon.append(lon0 + (xf + rng.normal(0, 700, n)) / (111.2e3 * np.cos(np.radians(lat0))))
            alt.append(rng.choice([rng.normal(9e3, 1e3, n), rng.normal(6e3, 600, n)]))
    return [np.concatenate(v) for v in (t, lat, lon, alt)]


strong = np.r_[rng.normal(12, 3, 22), np.linspace(18, 35, 6), np.full(12, 33.0)]
parts = [storm(strong, -20e3, 0.0, 15.0, minutes_total),
         storm(np.full(minutes_total, 3.0), -20e3, -25e3, 15.0, minutes_total)]
t, lat, lon, alt = (np.concatenate(v) for v in zip(*parts))
order = np.argsort(t)
header = f"""Lightning Mapping Array analyzed data
Analysis program: lma_analysis -d 20220330 -t 230000 -s {minutes_total * 60}
Data start time: 03/30/22 23:00:00
Number of seconds analyzed: {minutes_total * 60}
Location: SIMULATED
Coordinate center (lat,lon,alt): {lat0:.7f} {lon0:.7f} 0.00
Coordinate frame: cartesian
Number of stations: 3
Station information: id, name, lat(d), lon(d), alt(m), delay(ns), board_rev, rec_ch
Sta_info: A  Site 1             33.8896328  -89.0188344    83.52  100 52  3
Sta_info: B  Site 2             33.4405831  -88.8309586    74.73  100 52  3
Sta_info: C  Site 3             33.7498089  -88.6912139    56.38  100 52  3
Station data: id, name, win(us), dec_win(us), data_ver, rms_error(ns), sources, %, <P/P_m>, active
Sta_data: A  Site 1              80    12   70   914793  82.6  1.82   A
Sta_data: B  Site 2              80    12   70   218586  19.7  2.15   A
Sta_data: C  Site 3              80    12   70   946493  85.5  1.30   A
Metric file version: 4
Station mask order: CBA
Data: time (UT sec of day), lat, lon, alt(m), reduced chi^2, P(dBW), mask
Data format: 15.9f 12.8f 13.8f 9.2f 6.2f 5.1f 6x
Number of events: {t.size}
*** data ***
"""
rows = "\n".join(
    f"{82800.0 + t[i]:.9f} {lat[i]:.8f} {lon[i]:.8f} {alt[i]:.2f} 1.00 10.0 0x7" for i in order
)
(work / "LYLOUT_220330_230000_2400.dat").write_text(header + rows + "\n")
sources = read_lma(work / "LYLOUT_220330_230000_2400.dat", max_chi2=2.0, min_stations=3)
print(sources.sizes["number_of_events"], "sources read; network centre",
      float(sources.network_center_latitude), float(sources.network_center_longitude))
```

`cluster_flashes` groups sources closer than 3 km and 0.15 s in normalized
space-time distance into flashes (Fuchs et al. 2016):

```{code-cell} ipython3
flashes = cluster_flashes(sources)
assert sources.radarx.cluster_flashes().sizes == flashes.sizes  # the accessor of the sources
print(flashes.sizes["number_of_flashes"], "flashes;",
      int((flashes.flash_event_count >= 10).sum()), "with at least 10 sources")
big = flashes.flash_event_count.values[flashes.event_parent_flash_id.values] >= 10
window = (flashes.event_time.values >= start + np.timedelta64(30, "m")) & (
    flashes.event_time.values < start + np.timedelta64(31, "m"))
big = big & window  # one minute of the storm, for legibility
fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
ax[0].scatter(flashes.event_longitude[big], flashes.event_latitude[big], s=2,
              c=flashes.event_parent_flash_id[big] % 20, cmap="tab20")
ax[0].set(xlabel="longitude", ylabel="latitude", title="sources of 23:30 to 23:31, coloured by flash")
ax[1].scatter(flashes.event_time[big], flashes.event_altitude[big] / 1e3, s=2,
              c=flashes.event_parent_flash_id[big] % 20, cmap="tab20")
ax[1].set(ylabel="altitude (km MSL)", title="time and height")
ax[1].tick_params(axis="x", rotation=30)
plt.show()
```

`grid_lightning` counts sources, flash extent and flash initiations in the
boxes of a radarx grid (Bruning and MacGorman 2013), and
`vertical_source_distribution` gives their height distribution:

```{code-cell} ipython3
gx = np.arange(-60e3, 60.1e3, 1e3)
grid = xr.Dataset(coords={"x": gx, "y": gx, "z": np.arange(1e3, 15.1e3, 1e3), "latitude": lat0, "longitude": lon0})
density = grid_lightning(flashes, grid, interval="10min")
fed3d = grid_lightning(flashes, grid, z=True, interval="10min")
assert flashes.radarx.grid_lightning(grid, interval="10min").sizes == density.sizes  # accessor form
profile_src = vertical_source_distribution(flashes, grid.z.values, interval="10min")
fig, ax = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
fed = density.flash_extent_density.sum("time")
pm = ax[0].pcolormesh(gx / 1e3, gx / 1e3, fed.where(fed > 0), cmap="magma_r")
fig.colorbar(pm, ax=ax[0], label="flashes per box")
ax[0].set(title="flash extent density", xlabel="x (km)", ylabel="y (km)", aspect="equal")
sec = fed3d.flash_extent_density.isel(time=-1).sel(y=0, method="nearest")
pm = ax[1].pcolormesh(gx / 1e3, fed3d.z / 1e3, sec.where(sec > 0), cmap="magma_r")
fig.colorbar(pm, ax=ax[1], label="flashes per box")
ax[1].set(title="last 10 min, y = 0", xlabel="x (km)", ylabel="height (km)")
pm = ax[2].pcolormesh(profile_src.time, profile_src.z / 1e3, profile_src.source_count.T, cmap="viridis")
fig.colorbar(pm, ax=ax[2], label="sources")
ax[2].set(title="sources per height and 10 min", ylabel="height (km)")
fig.autofmt_xdate()
plt.show()
```

`cell_flash_rate` gives flash rates and source height distributions per cell
of a tracked-storm mask, here two discs following the cells every 5 min, and
`lightning_jump` finds jumps with the "2$\sigma$" algorithm of Schultz et al.
(2009):

```{code-cell} ipython3
cx = np.arange(-60e3, 60.1e3, 1e3)
mtimes = start + np.arange(0, minutes_total * 60 + 1, 300).astype("timedelta64[s]")
elapsed = (mtimes - start).astype(float) * 1e-9
xx, yy = np.meshgrid(cx, cx)
mask = np.zeros((mtimes.size, cx.size, cx.size), dtype=np.int32)
for k, sec_ in enumerate(elapsed):
    mask[k][np.hypot(xx - (-20e3 + 15 * sec_), yy) < 8e3] = 1
    mask[k][np.hypot(xx - (-20e3 + 15 * sec_), yy + 25e3) < 8e3] = 2
cells = xr.DataArray(mask, dims=("time", "y", "x"),
                     coords={"time": mtimes, "y": cx, "x": cx, "latitude": lat0, "longitude": lon0})
minute_edges = start + np.arange(minutes_total + 1).astype("timedelta64[m]")
rates = cell_flash_rate(flashes, cells, time_edges=minute_edges, z=np.arange(500.0, 16e3, 1000.0))
jumps = rates.flash_rate.radarx.lightning_jump()
jumps_fn = lightning_jump(rates.flash_rate)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True, gridspec_kw={"width_ratios": [2, 1]})
for c, color, label in zip(rates.cell.values, [OKABE["blue"], OKABE["orange"]], ["strong cell", "weak cell"]):
    ax[0].plot(rates.time, rates.flash_rate.sel(cell=c), color=color, alpha=0.35)
    j = jumps_fn.sel(cell=c)
    ax[0].plot(j.time, j.flash_rate, color=color, lw=2, label=label)
    for s_ in j.time.values[j.jump_start.values]:
        ax[0].axvline(s_, color=OKABE["red"])
    prof = rates.source_count.sel(cell=c).sum("time")
    ax[1].plot(prof / prof.sum(), rates.z / 1e3, color=color)
ax[0].axhline(10, color="0.5", ls=":")
ax[0].set(ylabel="flashes per minute", title="flash rates (thin: 1 min, thick: 2 min), jumps in red")
ax[0].legend(frameon=False, loc="upper left")
ax[1].set(xlabel="fraction of sources", ylabel="height (km MSL)", title="source heights")
fig.autofmt_xdate()
plt.show()
print("accessor and function give the same jumps:", bool((jumps.jump_start == jumps_fn.jump_start).all()))
```

## References

- Dawson, D., M. Biggerstaff, and S. Waugh, 2025: PERiLS_2022: Portable In Situ Precipitation Stations (PIPS) Data. Version 1.0. NSF NCAR Earth Observing Laboratory, https://doi.org/10.26023/HFBG-7W5M-WA00.
- Bean, B. R., and E. J. Dutton, 1966: Radio Meteorology. *NBS Monograph 92*, National Bureau of Standards, <https://doi.org/10.6028/NBS.MONO.92>
- Bruning, E. C., and D. R. MacGorman, 2013: Theory and Observations of Controls on Lightning Flash Size Spectra. *Journal of the Atmospheric Sciences*, **70**, 4012-4029, <https://doi.org/10.1175/JAS-D-12-0289.1>
- Frick, C., A. Seifert, and H. Wernli, 2013: A bulk parametrization of melting snowflakes with explicit liquid water fraction for the COSMO model. *Geoscientific Model Development*, **6**, 1925-1939, <https://doi.org/10.5194/gmd-6-1925-2013>
- Fuchs, B. R., E. C. Bruning, S. A. Rutledge, L. D. Carey, P. R. Krehbiel, and W. Rison, 2016: Climatological analyses of LMA data with an open-source lightning flash-clustering algorithm. *Journal of Geophysical Research: Atmospheres*, **121**, 8625-8648, <https://doi.org/10.1002/2015JD024663>
- Heymsfield, A., M. Szakáll, A. Jost, I. Giammanco, R. Wright, and J. Brimelow, 2020: Corrigendum (to: A comprehensive observational study of graupel and hail terminal velocity, mass flux, and kinetic energy). *Journal of the Atmospheric Sciences*, **77**, 405-412, <https://doi.org/10.1175/JAS-D-19-0185.1>
- Kumjian, M. R., and A. V. Ryzhkov, 2012: The Impact of Size Sorting on the Polarimetric Radar Variables. *Journal of the Atmospheric Sciences*, **69**, 2042-2060, <https://doi.org/10.1175/JAS-D-11-0125.1>
- Low, T. B., and R. List, 1982: Collision, coalescence and breakup of raindrops. Part I: Experimentally established coalescence efficiencies and fragment size distributions in breakup. *Journal of the Atmospheric Sciences*, **39**, 1591-1606, <https://doi.org/10.1175/1520-0469(1982)039<1591:CCABOR>2.0.CO;2>
- Low, T. B., and R. List, 1982: Collision, coalescence and breakup of raindrops. Part II: Parameterization of fragment size distributions. *Journal of the Atmospheric Sciences*, **39**, 1607-1619, <https://doi.org/10.1175/1520-0469(1982)039<1607:CCABOR>2.0.CO;2>
- Rahman, H., 2019: Fundamental Principles of Radar. CRC Press, <https://doi.org/10.1201/9780429279478>
- Raupach, T. H., and A. Berne, 2015: Correction of raindrop size distributions measured by Parsivel disdrometers, using a two-dimensional video disdrometer as a reference. *Atmospheric Measurement Techniques*, **8**, 343-365, <https://doi.org/10.5194/amt-8-343-2015>
- Rison, W., R. J. Thomas, P. R. Krehbiel, T. Hamlin, and J. Harlin, 1999: A GPS-based three-dimensional lightning mapping system: Initial observations in central New Mexico. *Geophysical Research Letters*, **26**, 3573-3576, <https://doi.org/10.1029/1999GL010856>
- Rotunno, R., J. B. Klemp, and M. L. Weisman, 1988: A Theory for Strong, Long-Lived Squall Lines. *Journal of the Atmospheric Sciences*, **45**, 463-485, <https://doi.org/10.1175/1520-0469(1988)045<0463:ATFSLL>2.0.CO;2>
- Ryzhkov, A. V., J. Snyder, J. T. Carlin, A. Khain, and M. Pinsky, 2020: What Polarimetric Weather Radars Offer to Cloud Modelers: Forward Radar Operators and Microphysical/Thermodynamic Retrievals. *Atmosphere*, **11**, 362, <https://doi.org/10.3390/atmos11040362>
- Ryzhkov, A., M. Diederich, P. Zhang, and C. Simmer, 2014: Potential Utilization of Specific Attenuation for Rainfall Estimation, Mitigation of Partial Beam Blockage, and Radar Networking. *Journal of Atmospheric and Oceanic Technology*, **31**, 599-619, <https://doi.org/10.1175/JTECH-D-13-00038.1>
- Schultz, C. J., W. A. Petersen, and L. D. Carey, 2009: Preliminary Development and Evaluation of Lightning Jump Algorithms for the Real-Time Detection of Severe Weather. *Journal of Applied Meteorology and Climatology*, **48**, 2543-2563, <https://doi.org/10.1175/2009JAMC2237.1>
