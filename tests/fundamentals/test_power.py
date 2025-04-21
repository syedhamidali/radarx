import numpy as np
from radarx.fundamentals import power


def test_peak_power():
    v = 1000  # volts
    r = 50  # ohms
    p = power.compute_peak_power(v, r)
    assert np.isclose(p, v**2 / r)


def test_average_power():
    pk = 1000  # watts
    dc = 0.1  # 10% duty cycle
    avg = power.compute_average_power(pk, dc)
    assert np.isclose(avg, pk * dc)


def test_min_detectable_signal():
    bw = 1e6  # 1 MHz
    temp = 290  # system temperature in Kelvin
    s = power.compute_min_detectable_signal(bw, temp)
    assert s > 0
