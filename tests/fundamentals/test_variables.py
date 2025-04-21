import numpy as np
from radarx.fundamentals import variables


def test_reflectivity_factor():
    p_return = 1e-9
    radar_const = 1e6
    range_m = 10000
    dielectric = 0.93
    z = variables.reflectivity_factor(p_return, radar_const, dielectric, range_m)
    assert z > 0


def test_differential_reflectivity():
    zh = 800.0
    zv = 200.0
    zdr = variables.differential_reflectivity(zh, zv)
    assert np.isclose(zdr, 10 * np.log10(zh / zv))


def test_linear_depolarization_ratio():
    zh = 800.0
    zv = 200.0
    ldr = variables.linear_depolarization_ratio(zh, zv)
    assert np.isclose(ldr, 10 * np.log10(zv / zh))


def test_circular_depolarization_ratio():
    zp = 800.0
    zo = 200.0
    cdr = variables.circular_depolarization_ratio(zp, zo)
    assert np.isclose(cdr, 10 * np.log10(zp / zo))


def test_radial_velocity():
    freq = 500.0
    wl = 0.03
    vr = variables.radial_velocity(freq, wl)
    assert np.isclose(vr, freq * wl / 2)
