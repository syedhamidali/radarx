"""
Radarx Conversion
=================

This module provides conversions between metric, imperial, and radar-relevant units.
It is designed for consistent, clear, and SI-compliant conversions.

References
----------
- BIPM SI Brochure: https://www.bipm.org/en/publications/si-brochure/
- Doviak and Zrnić (1993), Doppler Radar and Weather Observations

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = [
    "meters_to_kilometers",
    "kilometers_to_meters",
    "meters_to_miles",
    "miles_to_meters",
    "hz_to_mhz",
    "mhz_to_hz",
    "hz_to_ghz",
    "ghz_to_hz",
    "wavelength_to_frequency",
    "frequency_to_wavelength",
    "microseconds_to_seconds",
    "seconds_to_microseconds",
    "mps_to_kph",
    "kph_to_mps",
    "mps_to_knots",
    "knots_to_mps",
    "degrees_to_radians",
    "radians_to_degrees",
    "celsius_to_kelvin",
    "kelvin_to_celsius",
    "meters_to_feet",
    "feet_to_meters",
    "seconds_to_minutes",
    "minutes_to_seconds",
    "kelvin_to_fahrenheit",
    "fahrenheit_to_kelvin",
    "megahertz_to_si",
    "si_to_megahertz",
    "gigahertz_to_si",
    "si_to_gigahertz",
    "kilometers_to_si",
    "si_to_kilometers",
    "miles_to_si",
    "si_to_miles",
    "microseconds_to_si",
    "si_to_microseconds",
    "minutes_to_si",
    "si_to_minutes",
    "celsius_to_si",
    "si_to_celsius",
    "fahrenheit_to_si",
    "si_to_fahrenheit",
]

__doc__ = __doc__.format("\n   ".join(__all__))

from typing import Union
import numpy as np

Number = Union[float, int, np.ndarray]


# Length Conversions
def meters_to_kilometers(value: Number) -> Number:
    """Convert meters to kilometers."""
    return np.asarray(value) / 1_000


def kilometers_to_meters(value: Number) -> Number:
    """Convert kilometers to meters."""
    return np.asarray(value) * 1_000


def meters_to_miles(value: Number) -> Number:
    """Convert meters to miles."""
    return np.asarray(value) / 1_609.344


def miles_to_meters(value: Number) -> Number:
    """Convert miles to meters."""
    return np.asarray(value) * 1_609.344


# Frequency Conversions
def hz_to_mhz(value: Number) -> Number:
    """Convert Hertz to Megahertz."""
    return np.asarray(value) / 1e6


def mhz_to_hz(value: Number) -> Number:
    """Convert Megahertz to Hertz."""
    return np.asarray(value) * 1e6


def hz_to_ghz(value: Number) -> Number:
    """Convert Hertz to Gigahertz."""
    return np.asarray(value) / 1e9


def ghz_to_hz(value: Number) -> Number:
    """Convert Gigahertz to Hertz."""
    return np.asarray(value) * 1e9


# Wavelength Conversions
def wavelength_to_frequency(wavelength_m: Number, c: float = 3e8) -> Number:
    """Convert wavelength in meters to frequency in Hz."""
    return np.asarray(c) / np.asarray(wavelength_m)


def frequency_to_wavelength(frequency_hz: Number, c: float = 3e8) -> Number:
    """Convert frequency in Hz to wavelength in meters."""
    return np.asarray(c) / np.asarray(frequency_hz)


# Time Conversions
def microseconds_to_seconds(value: Number) -> Number:
    """Convert microseconds to seconds."""
    return np.asarray(value) * 1e-6


def seconds_to_microseconds(value: Number) -> Number:
    """Convert seconds to microseconds."""
    return np.asarray(value) / 1e-6


# Velocity Conversions
def mps_to_kph(value: Number) -> Number:
    """Convert meters per second to kilometers per hour."""
    return np.asarray(value) * 3.6


def kph_to_mps(value: Number) -> Number:
    """Convert kilometers per hour to meters per second."""
    return np.asarray(value) / 3.6


def mps_to_knots(value: Number) -> Number:
    """Convert meters per second to knots."""
    return np.asarray(value) * 1.943844


def knots_to_mps(value: Number) -> Number:
    """Convert knots to meters per second."""
    return np.asarray(value) / 1.943844


# Angular Conversions
def degrees_to_radians(value: Number) -> Number:
    """Convert degrees to radians."""
    return np.radians(value)


def radians_to_degrees(value: Number) -> Number:
    """Convert radians to degrees."""
    return np.degrees(value)


# Temperature Conversions
def celsius_to_kelvin(value: Number) -> Number:
    """Convert Celsius to Kelvin."""
    return np.asarray(value) + 273.15


def kelvin_to_celsius(value: Number) -> Number:
    """Convert Kelvin to Celsius."""
    return np.asarray(value) - 273.15


# Additional Length Conversions
def meters_to_feet(value: Number) -> Number:
    """Convert meters to feet."""
    return np.asarray(value) * 3.28084


def feet_to_meters(value: Number) -> Number:
    """Convert feet to meters."""
    return np.asarray(value) / 3.28084


# Additional Time Conversions
def seconds_to_minutes(value: Number) -> Number:
    """Convert seconds to minutes."""
    return np.asarray(value) / 60


def minutes_to_seconds(value: Number) -> Number:
    """Convert minutes to seconds."""
    return np.asarray(value) * 60


# Additional Temperature Conversions
def kelvin_to_fahrenheit(value: Number) -> Number:
    """Convert Kelvin to Fahrenheit."""
    return (np.asarray(value) - 273.15) * 9 / 5 + 32


def fahrenheit_to_kelvin(value: Number) -> Number:
    """Convert Fahrenheit to Kelvin."""
    return (np.asarray(value) - 32) * 5 / 9 + 273.15


# SI-compliant alias functions for unit conversions

# Frequency aliases
megahertz_to_si = mhz_to_hz
si_to_megahertz = hz_to_mhz
gigahertz_to_si = ghz_to_hz
si_to_gigahertz = hz_to_ghz

# Length aliases
kilometers_to_si = kilometers_to_meters
si_to_kilometers = meters_to_kilometers
miles_to_si = miles_to_meters
si_to_miles = meters_to_miles

# Time aliases
microseconds_to_si = microseconds_to_seconds
si_to_microseconds = seconds_to_microseconds
minutes_to_si = minutes_to_seconds
si_to_minutes = seconds_to_minutes

# Temperature aliases
celsius_to_si = celsius_to_kelvin
si_to_celsius = kelvin_to_celsius
fahrenheit_to_si = fahrenheit_to_kelvin
si_to_fahrenheit = kelvin_to_fahrenheit
