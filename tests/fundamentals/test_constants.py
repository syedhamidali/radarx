import numpy as np
from radarx.fundamentals import constants


def test_speed_of_light():
    assert np.isclose(constants.C, 3e8, rtol=5e-3)


def test_k_boltzmann():
    assert np.isclose(constants.K_BOLTZMANN, 1.381e-23)


def test_earth_radius():
    assert np.isclose(constants.EARTH_RADIUS, 6371000.0)


def test_effective_radius_4_3():
    assert np.isclose(
        constants.EFFECTIVE_RADIUS_4_3, 4.0 / 3.0 * constants.EARTH_RADIUS
    )


def test_dielectric_values():
    assert constants.DIELECTRIC_WATER > constants.DIELECTRIC_ICE


def test_z_to_dbz_factor():
    assert np.isclose(constants.Z_TO_DBZ_FACTOR, 10.0 / np.log(10))


def test_dbz_to_z_factor():
    assert np.isclose(constants.DBZ_TO_Z_FACTOR, np.log(10) / 10.0)


def test_temperature_standard():
    assert constants.T_STANDARD == 290.0
