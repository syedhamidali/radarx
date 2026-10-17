# User Guide

In this section we provide material how to use radarx.

Most of the material is distributed as jupyter notebooks to be able to interactively explore radarx functionality.

To use radarx in a project:

```python
import radarx
```

## Conventions

### Argument names

Every function of radarx uses one name per concept, and the accessor methods
(`ds.radarx.kdp`, `dtree.radarx.vil`, ...) have the signature of the function
they call, so `help` and tab completion show the same names.

| Concept | Name |
|---|---|
| radar data (sweep, volume, grid) as first argument | `obj` |
| reflectivity field | `dbz` |
| radial velocity field | `field` |
| differential phase, correlation coefficient, differential reflectivity, specific differential phase | `phidp`, `rhohv`, `zdr`, `kdp` |
| fields to process | `data_vars` |
| lowest accepted value of a quantity | `min_<quantity>` (`min_dbz`, `min_rhohv`, `min_snr`, `min_zdr`, `min_kdp`) |
| half-power beam width (degrees) | `beamwidth` |
| effective Earth radius factor | `k_e` |
| height of the radar | `radar_height` |
| number of threads, kernel implementation | `n_threads`, `engine`, both keyword-only, in this order |

The functions of {mod}`radarx.fundamentals` take plain numbers and arrays and
name them after the quantity, with the unit in the name where it matters
(`elevation_deg`, `range_m`, `beamwidth_rad`).

The names used up to release 0.4 (`dbzh`, `reflectivity`, `velocity`, `vrad`,
`vel`, `rho`, `ds`, `dtree`, `radar`, `volume`, `grid` as first argument,
`fields`, `dbz_min`, `rhohv_min`, `rho_min`, `snr_min`, `zdr_min`, `kdp_min`,
`min_reflectivity`, `beam_width`) still work for one release: the call gives a
`DeprecationWarning` that names the old and the new argument. In
`radarx.grid.grid_radar` the arguments after `method` (`n_threads`,
`fill_depth`, `fill_velocity`, `k_e`) are keyword-only; passing them by
position also warns. `hitschfeld_bordan_correction(use_compiled=False)` is
`engine="numpy"`.

### Precision

A floating point field that a function returns (a variable on gates, grid
cells, profile levels or sweeps) is `float32`, whatever the precision of the
input. The compiled kernels and the NumPy fallbacks compute in `float64` and
round once at the end, so both engines agree to `float32` rounding. These
results keep `float64`:

- coordinates (positions, times, angles),
- tables with one row per object, such as the couplets of `rotation_couplets`,
  and the storm motion of `estimate_motion`,
- scalar diagnostics of a solver (the cost of `multi_doppler`),
- corrected copies of an input field (`dealias_velocity`, `apply_mask`,
  `advect`), which keep the precision of the input.

Integer, boolean and flag variables keep their integer type.

### Units

The `units` attributes written by radarx are in the UDUNITS style: unit
symbols with integer exponents, separated by a space (`m s-1`, `degree km-1`,
`dB km-1`, `m-3 mm-1`, `mm6 m-3`, `kg m-2`, `s-1`, `dBZ`, `degree`, `1`). The
latitude and longitude keep the CF names `degrees_north` and `degrees_east`.
Fields that radarx reads can carry other spellings (xradar writes `degrees`
and `meters`); the functions accept both.

### Dask and extra dimensions

Fields can be dask arrays (a Zarr archive opened with `chunks=`) and can have
extra leading dimensions (`time`, `volume`) in front of `(azimuth, range)`.
The Notes of each docstring say which of these cases the function handles, and
the table below lists them.

A **per-sweep function** (lazy, extra dimensions) processes each
`(azimuth, range)` sweep of a stack on its own and gives a result with the
same extra dimensions. For dask input the result is a dask array and nothing
is computed until it is. The sweep dimensions are rechunked into one chunk;
the extra dimensions keep their chunks, so `chunk({"time": 1})` gives one task
per sweep. The sweeps of a DataTree are processed one by one in the same way.
A sweep in memory takes the code path it always had. Attributes that hold a
number found from the data (the noise level of `snr`, the filter window of
`estimate_kdp`) are written for a sweep in memory only.

```python
ds = xr.open_zarr("sweeps.zarr", chunks={"time": 1})  # (time, azimuth, range)
products = radarx.retrieve.estimate_kdp(ds)           # lazy, one task per sweep
products["KDP"].isel(time=0).compute()
```

A **function that computes** needs a whole volume, all times or all gates (a
gridding, a wind retrieval, a motion estimate). It takes dask input, computes
it and returns the result in memory, and its Notes begin with "This function
computes the dask array". A **function that cannot be lazy** returns a result
whose size depends on the data (couplets, flashes, super-observations); it
computes the dask input in the same way.

Some functions of the first class compute in one case, and say so: a volume
with `sweep_continuity=True` in `dealias_velocity`, `split_cuts=True` in
`echo_mask`, the winter scheme of `hid`, and a polar volume in `vil`,
`vil_density` and `echo_top`. The table covers `radarx.retrieve`, `radarx.grid`
and `radarx.ml`; a function with no row there takes a profile, a table or
coordinates only.

dask is optional: radarx imports it nowhere, xarray does when the data are
dask arrays.

| Behaviour | Module | Functions |
|---|---|---|
| Lazy, extra dimensions | `radarx.retrieve.attenuation` | `correct_attenuation` |
| Lazy, extra dimensions | `radarx.retrieve.calibration` | `phidp_offset`, `z_offset`, `zdr_offset` |
| Lazy, extra dimensions | `radarx.retrieve.columns` | `zdr_column`, `zdr_column_metrics` |
| Lazy, extra dimensions | `radarx.retrieve.dealias` | `correct_dual_prf`, `dealias_confidence`, `dealias_velocity` |
| Lazy, extra dimensions | `radarx.retrieve.dsd` | `dsd` |
| Lazy, extra dimensions | `radarx.retrieve.hid` | `hid` |
| Lazy, extra dimensions | `radarx.retrieve.kdp` | `estimate_kdp` |
| Lazy, extra dimensions | `radarx.retrieve.noise` | `correct_rhohv_noise`, `correct_zdr_noise`, `noise_level`, `snr` |
| Lazy, extra dimensions | `radarx.retrieve.qc` | `apply_mask`, `echo_mask` |
| Lazy, extra dimensions | `radarx.retrieve.qpe` | `rain_rate`, `rain_rate_a`, `rain_rate_kdp`, `rain_rate_z`, `rain_rate_z_zdr` |
| Lazy, extra dimensions | `radarx.retrieve.shear` | `azimuthal_shear`, `llsd`, `radial_divergence` |
| Lazy, extra dimensions | `radarx.retrieve.vertical_profiles` | `qvp` |
| Lazy, extra dimensions | `radarx.retrieve.vil` | `echo_top`, `liquid_water_content`, `vil`, `vil_density` |
| Lazy | `radarx.retrieve.beam_blockage` | `beam_blockage` |
| Lazy | `radarx.retrieve.beam_path` | `beam_path` |
| Lazy | `radarx.retrieve.forward_operator` | `forward_operator`, `to_radar_grid` |
| Lazy | `radarx.retrieve.rain_collisions` | `rain_shaft` |
| Lazy | `radarx.retrieve.tc_winds` | `gbvtd`, `tc_center`, `tc_decompose` |
| Computes | `radarx.grid.cone` | `grid_cones` |
| Computes | `radarx.grid.gates` | `grid_gates`, `grid_rhi` |
| Computes | `radarx.grid.grid` | `grid_radar`, `stack_data` |
| Computes | `radarx.grid.multi` | `grid_radars`, `merge_radars`, `network_bias` |
| Computes | `radarx.grid.ugrid` | `to_uxarray` |
| Computes | `radarx.ml.patches` | `polar_patches` |
| Computes | `radarx.retrieve.advection` | `advect`, `estimate_motion`, `interpolate_time` |
| Computes | `radarx.retrieve.biology` | `biological_echo` |
| Computes | `radarx.retrieve.cappi` | `create_cappi` |
| Computes | `radarx.retrieve.coldpool` | `baroclinic_generation`, `cold_pool_intensity`, `cold_pool_perturbation`, `potential_temperatures` |
| Computes | `radarx.retrieve.diabatic_lagrangian` | `diabatic_lagrangian`, `microphysical_rates`, `polarimetric_precipitation`, `ziegler2013_precipitation` |
| Computes | `radarx.retrieve.disdrometer` | `dsd_moments`, `fit_gamma`, `match_radar`, `number_concentration`, `process_disdrometer`, `radar_at_location`, `raupach_berne_correction` |
| Computes | `radarx.retrieve.dsd` | `dsd_spectrum`, `fit_gamma_moments`, `radar_from_dsd` |
| Computes | `radarx.retrieve.dsd_bayes` | `dsd_bayesian`, `dsd_prior` |
| Computes | `radarx.retrieve.evaporation` | `evaporation`, `integrate_evaporation` |
| Computes | `radarx.retrieve.hail` | `hail_kinetic_energy`, `hail_products`, `mesh`, `posh`, `shi` |
| Computes | `radarx.retrieve.imerg` | `match_imerg` |
| Computes | `radarx.retrieve.lagrangian` | `trajectories` |
| Computes | `radarx.retrieve.lightning` | `grid_lightning`, `vertical_source_distribution` |
| Computes | `radarx.retrieve.multidoppler` | `dual_doppler_analytic`, `multi_doppler`, `multi_doppler_input` |
| Computes | `radarx.retrieve.rain_trajectories` | `rain_trajectories`, `surface_dsd` |
| Computes | `radarx.retrieve.satellite_match` | `match_dpr` |
| Computes | `radarx.retrieve.shear` | `azimuthal_shear_layer_max` |
| Computes | `radarx.retrieve.single_doppler` | `single_doppler_winds` |
| Computes | `radarx.retrieve.tornado` | `tornado_probability`, `tornet_inputs` |
| Computes | `radarx.retrieve.vertical_profiles` | `cvp`, `mask_above_melting_layer`, `melting_layer`, `melting_layer_ppi`, `qvp_timeseries`, `rd_qvp`, `rhi_qvp` |
| Computes | `radarx.retrieve.wind_profile` | `vad_profile` |
| Computes | `radarx.retrieve.zdr_arc` | `zdr_arc` |
| Computes, cannot be lazy | `radarx.retrieve.lightning` | `cell_flash_rate`, `cluster_flashes` |
| Computes, cannot be lazy | `radarx.retrieve.rain_trajectories` | `rain_source_points`, `trajectory_matched_times` |
| Computes, cannot be lazy | `radarx.retrieve.superobs` | `superobs` |
| Computes, cannot be lazy | `radarx.retrieve.tornado` | `rotation_couplets` |

```{toctree}
:maxdepth: 2
:caption: Getting started

notebooks/Radar_Workflow
notebooks/Radar_Workflow_Advanced
notebooks/Radar_Workflow_Advanced_2
notebooks/Grid_Radar
notebooks/aws_data
notebooks/IMD_Radar_Data
notebooks/Interactive_Plots
notebooks/Exporting_Results
```

```{toctree}
:maxdepth: 2
:caption: Quality control and preprocessing

notebooks/Echo_QC
notebooks/Dealiasing
notebooks/KDP
notebooks/Attenuation_Correction
notebooks/Beam_Blockage
notebooks/Beam_Refraction
notebooks/Advection_Correction
notebooks/Superobs_Data_Assimilation
```

```{toctree}
:maxdepth: 2
:caption: Gridding and radar networks

notebooks/Multi_Radar_Grid
notebooks/Radar_UGRID_uxarray
```

```{toctree}
:maxdepth: 2
:caption: Wind retrieval and kinematics

notebooks/Single_Doppler_Winds
notebooks/Tropical_Cyclone_Winds
notebooks/Multi_Doppler
notebooks/Azimuthal_Shear
```

```{toctree}
:maxdepth: 2
:caption: Precipitation microphysics

notebooks/Hydrometeor_Classification
notebooks/DSD_Retrieval
notebooks/TMatrix_Validation
notebooks/TMatrix_Textbook_Examples
notebooks/TMatrix_Radar_Bands
notebooks/Disdrometers
notebooks/QVP
notebooks/Evaporation
notebooks/Rain_Size_Sorting
notebooks/Rain_Collisions
notebooks/Bayesian_DSD
notebooks/Rainfall_Estimation
```

```{toctree}
:maxdepth: 2
:caption: Thermodynamics and cold pools

notebooks/Soundings_and_ERA5
notebooks/Diabatic_Lagrangian
notebooks/Cold_Pools_and_Wind_Profiles
```

```{toctree}
:maxdepth: 2
:caption: Convective hazards

notebooks/Lightning
notebooks/Tornado_Detection
notebooks/Hail_Products
```

```{toctree}
:maxdepth: 2
:caption: Model verification

notebooks/Forward_Operator
```

```{toctree}
:maxdepth: 2
:caption: Machine learning

notebooks/Machine_Learning
notebooks/ML_KDP
notebooks/ML_Single_Doppler
```

```{toctree}
:maxdepth: 2
:caption: Gridding rates (Py-ART, radarx, wradlib)

notebooks/gridding_rate_part1
notebooks/gridding_rate_part2
notebooks/gridding_rate_part3
```

```{toctree}
:maxdepth: 2
:caption: Exercises

notebooks/radarx_fundamentals_examples
```
