import numpy as np
from radarx.core import conversion


def test_length_conversions():
    assert conversion.meters_to_kilometers(1000) == 1
    assert conversion.kilometers_to_meters(1) == 1000
    assert np.isclose(conversion.meters_to_miles(1609.344), 1)
    assert np.isclose(conversion.miles_to_meters(1), 1609.344)
    assert np.isclose(conversion.meters_to_feet(1), 3.28084)
    assert np.isclose(conversion.feet_to_meters(3.28084), 1)


def test_frequency_conversions():
    assert conversion.hz_to_mhz(1e6) == 1
    assert conversion.mhz_to_hz(1) == 1e6
    assert conversion.hz_to_ghz(1e9) == 1
    assert conversion.ghz_to_hz(1) == 1e9


def test_wavelength_frequency_conversions():
    assert conversion.wavelength_to_frequency(1) == 3e8
    assert conversion.frequency_to_wavelength(3e8) == 1


def test_time_conversions():
    assert conversion.microseconds_to_seconds(1e6) == 1
    assert conversion.seconds_to_microseconds(1) == 1e6
    assert conversion.seconds_to_minutes(120) == 2
    assert conversion.minutes_to_seconds(2) == 120


def test_velocity_conversions():
    assert conversion.mps_to_kph(1) == 3.6
    assert conversion.kph_to_mps(3.6) == 1
    assert np.isclose(conversion.mps_to_knots(1), 1.943844)
    assert np.isclose(conversion.knots_to_mps(1.943844), 1)


def test_angular_conversions():
    assert np.isclose(conversion.degrees_to_radians(180), np.pi)
    assert np.isclose(conversion.radians_to_degrees(np.pi), 180)


def test_temperature_conversions():
    assert conversion.celsius_to_kelvin(0) == 273.15
    assert conversion.kelvin_to_celsius(273.15) == 0
    assert np.isclose(conversion.kelvin_to_fahrenheit(273.15), 32)
    assert np.isclose(conversion.fahrenheit_to_kelvin(32), 273.15)


def test_si_aliases():
    # Frequency
    assert conversion.megahertz_to_si(1) == 1e6
    assert conversion.si_to_megahertz(1e6) == 1
    assert conversion.gigahertz_to_si(1) == 1e9
    assert conversion.si_to_gigahertz(1e9) == 1

    # Length
    assert conversion.kilometers_to_si(1) == 1000
    assert conversion.si_to_kilometers(1000) == 1
    assert np.isclose(conversion.miles_to_si(1), 1609.344)
    assert np.isclose(conversion.si_to_miles(1609.344), 1)

    # Time
    assert conversion.microseconds_to_si(1e6) == 1
    assert conversion.si_to_microseconds(1) == 1e6
    assert conversion.minutes_to_si(2) == 120
    assert conversion.si_to_minutes(120) == 2

    # Temperature
    assert conversion.celsius_to_si(0) == 273.15
    assert conversion.si_to_celsius(273.15) == 0
    assert np.isclose(conversion.fahrenheit_to_si(32), 273.15)
    assert np.isclose(conversion.si_to_fahrenheit(273.15), 32)
