import numpy as np
from radarx.fundamentals import reflectivity


def test_z_to_r_marshall_palmer():
    dbz = 40
    rr = reflectivity.z_to_r_marshall_palmer(dbz)
    assert rr > 0


def test_z_to_r_custom():
    dbz = np.array([30, 35, 40])
    rr = reflectivity.z_to_r_custom(dbz, a=250, b=1.5)
    assert rr.shape == dbz.shape
    assert np.all(rr > 0)


def test_dbz_attenuation_correction():
    dbz = np.array([10, 20, 30])
    corrected = reflectivity.dbz_attenuation_correction(dbz)
    assert np.all(corrected > dbz)
