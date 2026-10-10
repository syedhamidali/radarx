import numpy as np

from radarx.fundamentals import variables


def test_reflectivity_factor():
    p_return = 1e-9
    radar_const = 1e6
    range_m = 10000
    dielectric = 0.93
    z = variables.reflectivity_factor(p_return, radar_const, dielectric, range_m)
    assert z > 0


def test_reflectivity_factor_inverts_the_radar_equation():
    # P_r = C |K|^2 Z / r^2, |K|^2 = 0.93 (the default) is not squared
    radar_const, k2, z_true, r = 7.0e-3, 0.93, 1.0, 20e3
    p_return = radar_const * k2 * z_true / r**2
    assert np.isclose(
        variables.reflectivity_factor(p_return, radar_const, range_m=r), z_true
    )
    assert np.isclose(
        variables.reflectivity_factor(p_return, radar_const, k2, r), z_true
    )


def test_differential_reflectivity():
    zh = 800.0
    zv = 200.0
    zdr = variables.differential_reflectivity(zh, zv)
    assert np.isclose(zdr, 10 * np.log10(zh / zv))


def test_linear_depolarization_ratio():
    # LDR_vh = 10 log10(Z_vh / Z_hh), cross-polar over co-polar
    zhh = 1e4
    zvh = 1e2
    ldr = variables.linear_depolarization_ratio(z_co=zhh, z_cross=zvh)
    assert np.isclose(ldr, -20.0)
    assert np.isclose(variables.linear_depolarization_ratio(zhh, zvh), ldr)


def test_linear_depolarization_ratio_is_not_zdr():
    # the co-polar channels Zh and Zv give ZDR; with Zh as the co-polar and
    # a cross-polar channel 20 dB below it the LDR is -20 dB whatever Zv is
    zh, zv, zvh = 800.0, 200.0, 8.0
    ldr = variables.linear_depolarization_ratio(zh, zvh)
    zdr = variables.differential_reflectivity(zh, zv)
    assert np.isclose(ldr, -20.0)
    assert not np.isclose(ldr, -zdr)


def test_circular_depolarization_ratio():
    # same-sense (weak) channel over opposite-sense (strong) channel
    zp = 2.0
    zo = 200.0
    cdr = variables.circular_depolarization_ratio(zp, zo)
    assert np.isclose(cdr, -20.0)
    assert np.isclose(cdr, 10 * np.log10(zp / zo))


def test_radial_velocity():
    freq = 500.0
    wl = 0.03
    vr = variables.radial_velocity(freq, wl)
    assert np.isclose(vr, freq * wl / 2)
