"""
Constants
=========

.. module:: radarx.fundamentals.constants
   :synopsis: Physical and radar-related constants for use in radarx fundamental calculations.

This module contains constants used across radar signal processing and physical modeling, including:
- Physical constants (e.g., speed of light, Boltzmann constant)
- Radar-specific constants (e.g., dielectric constants, beamwidths)

.. autosummary::
   :nosignatures:
   :toctree: generated/

   C
   DBZ_TO_Z_FACTOR
   DIELECTRIC_ICE
   DIELECTRIC_WATER
   EARTH_RADIUS
   EFFECTIVE_RADIUS_4_3
   K_BOLTZMANN
   RADAR_BANDS
   T_STANDARD
   TYPICAL_BEAMWIDTH
   TYPICAL_PULSE_WIDTHS
   Z_TO_DBZ_FACTOR
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

# Speed of light in vacuum
C = 299_792_458  # [m/s]

# Radar band central wavelengths
RADAR_BANDS = {
    "S": 0.10,  # [m] ~10 cm
    "C": 0.05,  # [m] ~5 cm
    "X": 0.03,  # [m] ~3 cm
    "K": 0.015,  # [m] ~1.5 cm
    "Ka": 0.0085,  # [m] ~0.85 cm
    "W": 0.0032,  # [m] ~3.2 mm
}

# Typical beamwidth values
TYPICAL_BEAMWIDTH = {
    "WSR-88D": 1.0,  # [degrees]
    "C-band": 1.0,  # [degrees]
    "X-band": 1.0,  # [degrees]
    "Ka-band": 0.5,  # [degrees]
}

# Typical pulse widths
TYPICAL_PULSE_WIDTHS = {
    "short": 1e-7,  # [s]
    "medium": 5e-7,  # [s]
    "long": 1e-6,  # [s]
}

# Boltzmann constant
K_BOLTZMANN = 1.380649e-23  # [J/K]

EARTH_RADIUS = 6371000.0  # [m]
EFFECTIVE_RADIUS_4_3 = EARTH_RADIUS * 4 / 3  # [m]

# Dielectric constants (unitless, for Rayleigh scattering)
DIELECTRIC_WATER = 0.93
DIELECTRIC_ICE = 0.176

# Reflectivity conversion factor (optional)
Z_TO_DBZ_FACTOR = 10.0 / __import__("numpy").log(10)  # ~4.3429

# Inverse reflectivity conversion factor (optional)
DBZ_TO_Z_FACTOR = __import__("numpy").log(10) / 10.0  # ~0.2303

# Standard system temperature for thermal noise calculations
T_STANDARD = 290.0  # [K]
