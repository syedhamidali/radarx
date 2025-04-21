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
