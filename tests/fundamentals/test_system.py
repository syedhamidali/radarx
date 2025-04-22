import numpy as np
from radarx.fundamentals import system


def test_antenna_gain():
    p1, p2 = 100.0, 1.0
    gain = system.antenna_gain(p1, p2)
    assert np.isclose(gain, 20.0)


def test_frequency_from_wavelength():
    wavelength = 0.03
    freq = system.frequency(wavelength)
    from radarx.fundamentals import constants

    assert np.isclose(freq, constants.C / wavelength)


def test_wavelength_from_frequency():
    freq = 1e9
    wl = system.wavelength(freq)
    from radarx.fundamentals import constants

    assert np.isclose(wl, constants.C / freq)


def test_pulse_length():
    pdur = 1e-6
    length = system.pulse_length(pdur)
    from radarx.fundamentals import constants

    assert np.isclose(length, constants.C * pdur / 2)


def test_pulse_duration():
    length = 150.0
    pdur = system.pulse_duration(length)
    assert np.isclose(pdur, 2 * length / 3e8)


def test_ant_eff_area():
    gain = 45.0
    wl = 0.03
    area = system.ant_eff_area(gain, wl)
    assert area > 0


def test_size_param():
    d = 0.001
    wl = 0.05
    alpha = system.size_param(d, wl)
    assert alpha > 0


def test_power_return_target():
    power_t = 1000.0
    gain = 45.0
    wl = 0.03
    sig = 1e-6
    r = 10000
    p = system.power_return_target(power_t, gain, wl, sig, r)
    assert p > 0


def test_radar_const():
    power_t = 1e6  # 1 MW
    gain = 45.0  # dB
    tau = 1e-6  # pulse width in seconds
    wavelength = 0.05  # 5 cm
    bw_h = 1.0  # degrees
    bw_v = 1.0  # degrees
    aloss = 2.0  # dB
    rloss = 1.0  # dB

    const = system.radar_const(power_t, gain, tau, wavelength, bw_h, bw_v, aloss, rloss)
    assert const > 0


def test_radar_equation():
    power_t = 1000.0
    gain = 1000.0
    wavelength = 0.03
    sigma = 1.0
    r = 10000.0
    received_power = system.radar_equation(power_t, gain, gain, wavelength, sigma, r)
    assert received_power > 0


def test_solve_peak_power():
    received_power = 1e-14
    gain = 1000.0
    wavelength = 0.03
    sigma = 1.0
    r = 100000.0
    peak_power = system.solve_peak_power(
        received_power, gain, gain, wavelength, sigma, r
    )
    assert peak_power > 0
