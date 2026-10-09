# Packaged data of radarx.retrieve

`dsd_prior_perils2022.csv` holds the gamma DSD parameters (log10 Nw, Dm, mu) of 1-min
spectra from the Portable In Situ Precipitation Stations (PIPS, OTT Parsivel2) of the
PERiLS 2022 field campaign, matched to the lowest WSR-88D beam along the drop fall
trajectories. The `"perils2022"` prior of `radarx.retrieve.dsd_prior` is learned from it.
The prior was learned from PIPS spectra of the data set below, and no quality control
beyond the processing of the drop size distributions in that data set was applied.
The spectra themselves are not part of radarx.

Data:

Dawson, D., M. Biggerstaff, and S. Waugh, 2025: PERiLS_2022: Portable In Situ Precipitation Stations (PIPS) Data. Version 1.0. NSF NCAR Earth Observing Laboratory, https://doi.org/10.26023/HFBG-7W5M-WA00.

Campaign:

Kosiba, K. A., and Coauthors, 2024: The Propagation, Evolution, and Rotation in Linear Storms (PERiLS) Project. Bull. Amer. Meteor. Soc., 105, E1768-E1799, https://doi.org/10.1175/BAMS-D-22-0064.1.

We thank the PERiLS investigators and the NSF NCAR Earth Observing Laboratory for these data.
The data set is distributed under the UCAR terms of use (https://www.ucar.edu/terms-of-use).
The default prior of radarx is `"generic"`, which does not depend on these data.
