import numpy as np
from radarx.fundamentals import principles


def test_range_resolution():
    pw = 1e-6  # 1 microsecond
    rr = principles.range_resolution(pw)
    assert np.isclose(rr, principles.C * pw / 2)


def test_snr():
    signal = 1.0
    noise = 0.01
    snr_db = principles.snr(signal, noise)
    assert snr_db > 0


def test_doppler_frequency_shift():
    v_radial = 10.0  # m/s
    wavelength = 0.05  # 5 cm
    freq = principles.doppler_frequency_shift(v_radial, wavelength)
    assert np.isclose(freq, 2 * v_radial / wavelength)


def test_round_trip_time():
    distance = 15000  # meters
    t = principles.round_trip_time(distance)
    assert np.isclose(t, 2 * distance / principles.C)


def test_radar_range():
    # Example values from Rinehart or typical radar systems
    transmit_power = 5000  # W
    gain = 1000  # linear
    wavelength = 0.1  # 10 cm
    rcs = 1  # m^2
    system_loss = 1.5
    min_detectable_power = 1e-13  # W

    r = principles.radar_range(
        transmit_power, gain, wavelength, rcs, system_loss, min_detectable_power
    )

    # Manually compute expected range for assertion
    expected = (
        (transmit_power * gain**2 * wavelength**2 * rcs)
        / ((4 * np.pi) ** 3 * system_loss * min_detectable_power)
    ) ** 0.25
    assert np.isclose(r, expected)


def test_compute_doppler_shift():
    velocity = 15.0  # m/s
    wavelength = 0.1  # 10 cm
    expected_shift = 2 * velocity / wavelength
    assert np.isclose(
        principles.compute_doppler_shift(velocity, wavelength), expected_shift
    )


def test_compute_snr():
    power_received = 1e-10  # W
    noise_bandwidth = 1e6  # Hz
    system_temp = 290  # K
    noise_power = principles.K_BOLTZMANN * system_temp * noise_bandwidth
    expected_snr = power_received / noise_power
    assert np.isclose(
        principles.compute_snr(power_received, noise_bandwidth, system_temp),
        expected_snr,
    )
