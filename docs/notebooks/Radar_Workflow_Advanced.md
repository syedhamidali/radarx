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
NEXRAD volumes to the three-dimensional wind. This second part shows the rest
of radarx, with the same style of one or two sentences, a call with realistic
arguments and a figure or a printed result for every function. Everything here
is fast: the inputs are either small open data sets or synthetic data with a
known answer, so that nothing depends on field-campaign archives that cannot
be redistributed (the formats of those archives are read from small files
written on the fly).

| Section | What it shows | radarx calls |
|---|---|---|
| 1 | radar equation, beam geometry, Doppler and unit helpers | `radarx.fundamentals`, `radarx.core` |
| 2 | disdrometer spectra, quality control, gamma fits, radar variables | `radarx.io.read_parsivel`, `disdrometer_qc`, `fit_gamma`, `match_radar` |
| 3 | Bayesian DSD retrieval with uncertainty | `dsd_prior`, `forward_grid`, `dsd_bayesian` |
| 4 | raindrop trajectories and size sorting | `rain_trajectories`, `size_sorting`, `surface_dsd` |
| 5 | surface stations, profilers, cold pools | `read_sticknet`, `read_pips`, `cold_pool_intensity` |
| 6 | lightning mapping | `read_lma`, `cluster_flashes`, `lightning_jump` |
| 7 | tornado detection and biological echo | `tornet_inputs`, `tornado_probability`, `biological_echo` |
| 8 | wind from a single Doppler radar | `single_doppler_winds`, `radar_geometry` |
| 9 | diabatic Lagrangian analysis | `trajectories`, `diabatic_lagrangian`, `microphysical_rates` |
| 10 | machine-learning plumbing | `radarx.ml` |

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
)
from radarx.retrieve import (
    azimuthal_shear,
    baroclinic_generation,
    biological_echo,
    bulk_shear,
    buoyancy,
    cell_flash_rate,
    cluster_flashes,
    cold_pool_intensity,
    cold_pool_intensity_from_pressure,
    cold_pool_intensity_from_surface,
    cold_pool_perturbation,
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
    fall_speed,
    fit_gamma,
    fit_gamma_moments,
    forward_grid,
    grid_lightning,
    lightning_jump,
    match_radar,
    microphysical_rates,
    number_concentration,
    parsivel_bins,
    polarimetric_precipitation,
    potential_temperatures,
    process_disdrometer,
    radar_at_location,
    radar_from_dsd,
    radar_geometry,
    rain_source_points,
    rain_trajectories,
    raupach_berne_correction,
    rkw_ratio,
    rotation_couplets,
    scattering_table,
    single_doppler_winds,
    size_sorting,
    surface_dsd,
    terminal_fall_speed,
    tornado_probability,
    tornet_inputs,
    trajectories,
    trajectory_matched_times,
    vertical_source_distribution,
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
""".strip().splitlines()
rows = []
for line in calls:
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
pairs = match_radar(station, klbb, fields=["DBZH", "ZDR"], window="60s", delay="fall")
above = radar_at_location(klbb, lat, lon, float(klbb["altitude"]), fields=["DBZH", "ZDR"])
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
linear least-squares derivative (LLSD) azimuthal shear from `azimuthal_shear`.

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
couplets = rotation_couplets(sweep, min_reflectivity=20.0)
tor_acc = kgwx.radarx.tornado_probability(max_range=150e3)  # the accessor of the volume runs the same network
couplets_acc = sweep.radarx.rotation_couplets(min_reflectivity=20.0)
print("accessors agree:", bool(np.allclose(tor_acc.tornado_probability, tor.tornado_probability, equal_nan=True)),
      couplets_acc.sizes["couplet"] == couplets.sizes["couplet"])
print(f"{tor.sizes['chip']} chips; highest tornado probability {float(tor.tornado_probability.max()):.2f}; "
      f"{couplets.sizes['couplet']} rotation couplets above 0.006 1/s")
couplets.isel(couplet=slice(0, 5)).to_dataframe()[["azimuth", "range", "azimuthal_shear", "delta_v"]].round(3)
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
smoothed = ml.denormalize(ml.reassemble(out, index, attrs=model.attrs).assign_attrs(dbz_n.attrs))
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

## References

- Dawson, D., M. Biggerstaff, and S. Waugh, 2025: PERiLS_2022: Portable In Situ Precipitation Stations (PIPS) Data. Version 1.0. NSF NCAR Earth Observing Laboratory, https://doi.org/10.26023/HFBG-7W5M-WA00.
- Bruning, E. C., and D. R. MacGorman, 2013: Theory and Observations of Controls on Lightning Flash Size Spectra. *Journal of the Atmospheric Sciences*, **70**, 4012-4029, <https://doi.org/10.1175/JAS-D-12-0289.1>
- Fuchs, B. R., E. C. Bruning, S. A. Rutledge, L. D. Carey, P. R. Krehbiel, and W. Rison, 2016: Climatological analyses of LMA data with an open-source lightning flash-clustering algorithm. *Journal of Geophysical Research: Atmospheres*, **121**, 8625-8648, <https://doi.org/10.1002/2015JD024663>
- Gao, J., M. Xue, A. Shapiro, and K. K. Droegemeier, 1999: A Variational Method for the Analysis of Three-Dimensional Wind Fields from Two Doppler Radars. *Monthly Weather Review*, **127**, 2128-2142, <https://doi.org/10.1175/1520-0493(1999)127<2128:AVMFTA>2.0.CO;2>
- Kumjian, M. R., and A. V. Ryzhkov, 2012: The Impact of Size Sorting on the Polarimetric Radar Variables. *Journal of the Atmospheric Sciences*, **69**, 2042-2060, <https://doi.org/10.1175/JAS-D-11-0125.1>
- Lin, T.-Y., K. Winner, G. Bernstein, A. Mittal, A. M. Dokter, K. G. Horton, C. Nilsson, B. M. Van Doren, A. Farnsworth, F. A. La Sorte, S. Maji, and D. Sheldon, 2019: MistNet: Measuring historical bird migration in the US using archived weather radar data and convolutional neural networks. *Methods in Ecology and Evolution*, **10**, 1908-1922, <https://doi.org/10.1111/2041-210X.13280>
- Lin, Y.-L., R. D. Farley, and H. D. Orville, 1983: Bulk Parameterization of the Snow Field in a Cloud Model. *Journal of Climate and Applied Meteorology*, **22**, 1065-1092, <https://doi.org/10.1175/1520-0450(1983)022<1065:BPOTSF>2.0.CO;2>
- Mahalik, M. C., B. R. Smith, K. L. Elmore, D. M. Kingfield, K. L. Ortega, and T. M. Smith, 2019: Estimates of Gradients in Radar Moments Using a Linear Least Squares Derivative Technique. *Weather and Forecasting*, **34**, 415-434, <https://doi.org/10.1175/WAF-D-18-0095.1>
- Rahman, H., 2019: Fundamental Principles of Radar. CRC Press, <https://doi.org/10.1201/9780429279478>
- Raupach, T. H., and A. Berne, 2015: Correction of raindrop size distributions measured by Parsivel disdrometers, using a two-dimensional video disdrometer as a reference. *Atmospheric Measurement Techniques*, **8**, 343-365, <https://doi.org/10.5194/amt-8-343-2015>
- Rison, W., R. J. Thomas, P. R. Krehbiel, T. Hamlin, and J. Harlin, 1999: A GPS-based three-dimensional lightning mapping system: Initial observations in central New Mexico. *Geophysical Research Letters*, **26**, 3573-3576, <https://doi.org/10.1029/1999GL010856>
- Rotunno, R., J. B. Klemp, and M. L. Weisman, 1988: A Theory for Strong, Long-Lived Squall Lines. *Journal of the Atmospheric Sciences*, **45**, 463-485, <https://doi.org/10.1175/1520-0469(1988)045<0463:ATFSLL>2.0.CO;2>
- Sandmæl, T. N., B. R. Smith, A. E. Reinhart, I. M. Schick, M. C. Ake, J. G. Madden, R. B. Steeves, S. S. Williams, K. L. Elmore, and T. C. Meyer, 2023: The Tornado Probability Algorithm: A Probabilistic Machine Learning Tornadic Circulation Detection Algorithm. *Weather and Forecasting*, **38**, 445-466, <https://doi.org/10.1175/WAF-D-22-0123.1>
- Schultz, C. J., W. A. Petersen, and L. D. Carey, 2009: Preliminary Development and Evaluation of Lightning Jump Algorithms for the Real-Time Detection of Severe Weather. *Journal of Applied Meteorology and Climatology*, **48**, 2543-2563, <https://doi.org/10.1175/2009JAMC2237.1>
- Shapiro, A., 1993: The Use of an Exact Solution of the Navier-Stokes Equations in a Validation Test of a Three-Dimensional Nonhydrostatic Numerical Model. *Monthly Weather Review*, **121**, 2420-2425, <https://doi.org/10.1175/1520-0493(1993)121<2420:TUOAES>2.0.CO;2>
- Veillette, M. S., J. M. Kurdzo, P. M. Stepanian, J. Y. N. Cho, T. Reis, S. Samsi, J. McDonald, and N. Chisler, 2025: A Benchmark Dataset for Tornado Detection and Prediction Using Full-Resolution Polarimetric Weather Radar Data. *Artificial Intelligence for the Earth Systems*, **4**, <https://doi.org/10.1175/AIES-D-24-0006.1>
- Ziegler, C. L., 2013a: A Diabatic Lagrangian Technique for the Analysis of Convective Storms. Part I: Description and Validation via an Observing System Simulation Experiment. *Journal of Atmospheric and Oceanic Technology*, **30**, 2248-2265, <https://doi.org/10.1175/JTECH-D-12-00194.1>
- Ziegler, C. L., 2013a: A Diabatic Lagrangian Technique for the Analysis of Convective Storms. Part II: Application to a Radar-Observed Storm. *Journal of Atmospheric and Oceanic Technology*, **30**, 2266-2280, <https://doi.org/10.1175/JTECH-D-13-00036.1>
