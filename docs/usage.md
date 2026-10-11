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

```{toctree}
:maxdepth: 2
:caption: Getting started

notebooks/Radar_Workflow
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
