"""
Constants
=========

.. module:: radarx.fundamentals.constants
   :synopsis: Physical and radar-related constants for use in radarx fundamental calculations.

This module contains constants used across radar signal processing and physical modeling, including:
- Physical constants (e.g., speed of light, Boltzmann constant)
- Radar-specific constants (e.g., dielectric factors, beamwidths)

Every constant below states its source in a comment. Values that are
radarx's own choices (nominal band wavelengths, typical beamwidths and pulse
widths, the Earth radius) are labelled as such and are not taken from a
publication. Physical constants come from the SI definitions [1]_ [2]_;
the Earth radius, dielectric factors and WSR-88D values from the textbooks
[3]_ [4]_ [5]_ [6]_ (the comments next to each constant give the equation or
table).

References
----------
.. [1] BIPM, 2019: *The International System of Units (SI)*, 9th ed.
       Bureau International des Poids et Mesures, Sevres,
       https://www.bipm.org/en/publications/si-brochure (document, no DOI).
       Defines c (metre, 1983) and k (2019 revision) by exact values.
.. [2] Newell, D. B., F. Cabiati, J. Fischer, K. Fujii, S. G. Karshenboim,
       H. S. Margolis, E. de Mirandes, P. J. Mohr, F. Nez, K. Pachucki,
       T. J. Quinn, B. N. Taylor, M. Wang, B. M. Wood, and Z. Zhang, 2018:
       The CODATA 2017 values of h, e, k, and N_A for the revision of the
       SI. *Metrologia*, **55** (1), L13-L16,
       https://doi.org/10.1088/1681-7575/aa950a
.. [3] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
       Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
       DOI).
.. [4] Rinehart, R. E., 1991: *Radar for Meteorologists*, 2nd ed. (3rd
       printing 1994). Rinehart Publications, Grand Forks, ND, ISBN
       0-9608700-7-5 (book, no DOI).
.. [5] Fabry, F., 2015: *Radar Meteorology: Principles and Practice*.
       Cambridge University Press, https://doi.org/10.1017/CBO9781107707405
       (book; ISBN 978-1-107-07046-2).
.. [6] Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler
       Weather Radar: Principles and Applications*. Cambridge University
       Press, https://doi.org/10.1017/CBO9780511541094 (book; ISBN
       0-521-62384-7).

.. autosummary::
   :nosignatures:
   :toctree: generated/

    {}
"""

__all__ = [
    "C",
    "DBZ_TO_Z_FACTOR",
    "DIELECTRIC_ICE",
    "DIELECTRIC_WATER",
    "EARTH_RADIUS",
    "EFFECTIVE_RADIUS_4_3",
    "K_BOLTZMANN",
    "RADAR_BANDS",
    "T_STANDARD",
    "TYPICAL_BEAMWIDTH",
    "TYPICAL_PULSE_WIDTHS",
    "Z_TO_DBZ_FACTOR",
]

__doc__ = __doc__.format("\n   ".join(__all__))

# Speed of light in vacuum: exact by the definition of the metre (17th CGPM,
# 1983), BIPM SI Brochure 9th ed. (2019) [1]. (radarx.core.conversion uses
# the rounded 3e8 m/s as its default instead.)
C = 299_792_458  # [m/s]

# Radar band central wavelengths: radarx choice, nominal rounded values (S
# 3 GHz, C 6 GHz, X 10 GHz, K 20 GHz, Ka 35 GHz, W 94 GHz), not the exact
# wavelengths of any particular radar and not taken from a publication. The
# band letter limits of IEEE Std 521 are not used here. "K" at 1.5 cm is
# ambiguous between the K, Ku and Ka sub-bands.
RADAR_BANDS = {
    "S": 0.10,  # [m] ~10 cm
    "C": 0.05,  # [m] ~5 cm
    "X": 0.03,  # [m] ~3 cm
    "K": 0.015,  # [m] ~1.5 cm
    "Ka": 0.0085,  # [m] ~0.85 cm
    "W": 0.0032,  # [m] ~3.2 mm
}

# Typical beamwidth values: radarx choice, rounded one-degree class values
# (the WSR-88D beamwidth is 1 degree, Doviak and Zrnic 1993, Table 3.1, p. 47
# [3]); the other entries are not taken from a publication.
TYPICAL_BEAMWIDTH = {
    "WSR-88D": 1.0,  # [degrees]
    "C-band": 1.0,  # [degrees]
    "X-band": 1.0,  # [degrees]
    "Ka-band": 0.5,  # [degrees]
}

# Typical pulse widths: radarx choice, order-of-magnitude values; not taken
# from a publication. For comparison the WSR-88D pulse widths are 1.57 us and
# 4.57 us (Doviak and Zrnic 1993, Table 3.1, p. 47 [3]).
TYPICAL_PULSE_WIDTHS = {
    "short": 1e-7,  # [s]
    "medium": 5e-7,  # [s]
    "long": 1e-6,  # [s]
}

# Boltzmann constant: exact since the 2019 SI revision, k = 1.380649e-23 J/K
# (CODATA 2017 value, Newell et al. 2018 [2]; BIPM 2019 [1]). D&Z (1993, p. 54)
# quote the rounded 1.38e-23 W s/K [3].
K_BOLTZMANN = 1.380649e-23  # [J/K]

# Mean Earth radius: radarx choice, the conventional round figure 6371 km. The
# books give slightly different values: D&Z (1993) use 6375 km in their
# refractivity figure [3], Rinehart (1991) uses 6374 km [4]. The choice moves
# the 4/3 beam height by only metres at 100 km range.
EARTH_RADIUS = 6371000.0  # [m]
# Effective Earth radius a_e = k_e a with k_e = 4/3 for the standard
# refractivity gradient dn/dh = -1/(4a): D&Z (1993), Eqs. (2.27)-(2.28d),
# pp. 21-22 [3]; Rinehart (1991), Chapter 3 (pp. 42-43) [4].
EFFECTIVE_RADIUS_4_3 = EARTH_RADIUS * 4 / 3  # [m]

# Dielectric factors |K|^2 = |(eps - 1)/(eps + 2)|^2 of the Rayleigh
# backscatter law (not dielectric constants).
# Water 0.93 and solid ice 0.176 (density 920 kg/m3): Fabry (2015), Table 3.1,
# p. 34 [5]. Bringi and Chandrasekar (2001), Section 7.4, text after
# Eq. (7.82) [6]: |K_w|^2 ~ 0.93 and |K_ice|^2 ~ 0.17 at 3 GHz and 0 C.
# Rinehart (1991), p. 66 [4] quotes 0.93 and 0.197 for ice (older value). The
# true |K|^2 of water depends on temperature and wavelength (D&Z 1993 [3],
# Rinehart 1991 [4]); 0.93 is the conventional constant.
DIELECTRIC_WATER = 0.93
DIELECTRIC_ICE = 0.176

# 10/ln(10) = 4.3429...: 10*log10(x) = (10/ln 10)*ln(x). Pure definition of
# the decibel, no source needed. (Its name suggests Z -> dBZ, but it only
# converts natural logarithms to dB.)
Z_TO_DBZ_FACTOR = 10.0 / __import__("numpy").log(10)  # ~4.3429

# ln(10)/10 = 0.2303...: x = exp((ln 10 / 10) * x_dB). Pure definition.
DBZ_TO_Z_FACTOR = __import__("numpy").log(10) / 10.0  # ~0.2303

# Standard system temperature for thermal noise calculations: T0 = 290 K.
# D&Z (1993, p. 56) set the temperatures of radome, line and T/R switch to
# 290 K "which approximates the temperature of the environment" [3]. The
# conventional standard reference temperature T0 = 290 K is not cited to a
# standard.
T_STANDARD = 290.0  # [K]
