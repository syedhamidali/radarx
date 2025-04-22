import numpy as np
import pytest
from radarx.core import conversion


@pytest.mark.parametrize(
    "func, input_val, expected",
    [
        (conversion.meters_to_kilometers, 1000, 1),
        (conversion.kilometers_to_meters, 1, 1000),
        (conversion.meters_to_miles, 1609.344, 1),
        (conversion.miles_to_meters, 1, 1609.344),
        (conversion.meters_to_feet, 1, 3.28084),
        (conversion.feet_to_meters, 3.28084, 1),
    ],
)
def test_length_conversions(func, input_val, expected):
    assert np.isclose(func(input_val), expected)


@pytest.mark.parametrize(
    "func, input_val, expected",
    [
        (conversion.hz_to_mhz, 1e6, 1),
        (conversion.mhz_to_hz, 1, 1e6),
        (conversion.hz_to_ghz, 1e9, 1),
        (conversion.ghz_to_hz, 1, 1e9),
    ],
)
def test_frequency_conversions(func, input_val, expected):
    assert np.isclose(func(input_val), expected)


@pytest.mark.parametrize(
    "func, input_val, expected",
    [
        (conversion.wavelength_to_frequency, 1, 3e8),
        (conversion.frequency_to_wavelength, 3e8, 1),
    ],
)
def test_wavelength_frequency_conversions(func, input_val, expected):
    assert np.isclose(func(input_val), expected)


@pytest.mark.parametrize(
    "func, input_val, expected",
    [
        (conversion.microseconds_to_seconds, 1e6, 1),
        (conversion.seconds_to_microseconds, 1, 1e6),
        (conversion.seconds_to_minutes, 120, 2),
        (conversion.minutes_to_seconds, 2, 120),
    ],
)
def test_time_conversions(func, input_val, expected):
    assert np.isclose(func(input_val), expected)


@pytest.mark.parametrize(
    "func, input_val, expected",
    [
        (conversion.mps_to_kph, 1, 3.6),
        (conversion.kph_to_mps, 3.6, 1),
        (conversion.mps_to_knots, 1, 1.943844),
        (conversion.knots_to_mps, 1.943844, 1),
    ],
)
def test_velocity_conversions(func, input_val, expected):
    assert np.isclose(func(input_val), expected)


@pytest.mark.parametrize(
    "func, input_val, expected",
    [
        (conversion.degrees_to_radians, 180, np.pi),
        (conversion.radians_to_degrees, np.pi, 180),
    ],
)
def test_angular_conversions(func, input_val, expected):
    assert np.isclose(func(input_val), expected)


@pytest.mark.parametrize(
    "func, input_val, expected",
    [
        (conversion.celsius_to_kelvin, 0, 273.15),
        (conversion.kelvin_to_celsius, 273.15, 0),
        (conversion.kelvin_to_fahrenheit, 273.15, 32),
        (conversion.fahrenheit_to_kelvin, 32, 273.15),
    ],
)
def test_temperature_conversions(func, input_val, expected):
    assert np.isclose(func(input_val), expected)


# SI-compliant alias function tests
@pytest.mark.parametrize(
    "func, input_val, expected",
    [
        # Frequency
        (conversion.megahertz_to_si, 1, 1e6),
        (conversion.si_to_megahertz, 1e6, 1),
        (conversion.gigahertz_to_si, 1, 1e9),
        (conversion.si_to_gigahertz, 1e9, 1),
    ],
)
def test_si_aliases_frequency(func, input_val, expected):
    assert np.isclose(func(input_val), expected)


@pytest.mark.parametrize(
    "func, input_val, expected",
    [
        # Length
        (conversion.kilometers_to_si, 1, 1000),
        (conversion.si_to_kilometers, 1000, 1),
        (conversion.miles_to_si, 1, 1609.344),
        (conversion.si_to_miles, 1609.344, 1),
    ],
)
def test_si_aliases_length(func, input_val, expected):
    assert np.isclose(func(input_val), expected)


@pytest.mark.parametrize(
    "func, input_val, expected",
    [
        # Time
        (conversion.microseconds_to_si, 1e6, 1),
        (conversion.si_to_microseconds, 1, 1e6),
        (conversion.minutes_to_si, 2, 120),
        (conversion.si_to_minutes, 120, 2),
    ],
)
def test_si_aliases_time(func, input_val, expected):
    assert np.isclose(func(input_val), expected)


@pytest.mark.parametrize(
    "func, input_val, expected",
    [
        # Temperature
        (conversion.celsius_to_si, 0, 273.15),
        (conversion.si_to_celsius, 273.15, 0),
        (conversion.fahrenheit_to_si, 32, 273.15),
        (conversion.si_to_fahrenheit, 273.15, 32),
    ],
)
def test_si_aliases_temperature(func, input_val, expected):
    assert np.isclose(func(input_val), expected)
