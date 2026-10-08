# User Guide

In this section we provide material how to use radarx.

Most of the material is distributed as jupyter notebooks to be able to interactively explore radarx functionality.

To use radarx in a project:

```python
import radarx
```

```{toctree}
:maxdepth: 2
:caption: Getting started

notebooks/Radar_Workflow
notebooks/Grid_Radar
notebooks/aws_data
notebooks/IMD_Radar_Data
notebooks/Interactive_Plots
```

```{toctree}
:maxdepth: 2
:caption: Quality control and preprocessing

notebooks/Echo_QC
notebooks/Dealiasing
notebooks/KDP
notebooks/Advection_Correction
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
notebooks/Bayesian_DSD
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
```

```{toctree}
:maxdepth: 2
:caption: Machine learning

notebooks/Machine_Learning
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
