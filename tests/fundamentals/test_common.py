import numpy as np
from radarx.fundamentals import common


def test_linearize_dbz():
    dbz = 30.0
    z = common.linearize_dbz(dbz)
    assert np.isclose(z, 10 ** (dbz / 10.0))


def test_dBz_from_Z():
    z = 1000.0
    dbz = common.dbz_from_z(z)
    assert np.isclose(dbz, 10.0 * np.log10(z))


def test_m_to_km():
    assert common.m_to_km(1000) == 1.0


def test_km_to_m():
    assert common.km_to_m(1.0) == 1000.0


def test_mps_to_kts():
    assert np.isclose(common.mps_to_kts(10.0), 19.4384, atol=1e-4)


def test_kts_to_mps():
    assert np.isclose(common.kts_to_mps(19.4384), 10.0, atol=1e-4)


def test_si_to_kts_and_back():
    assert np.isclose(common.si_to_kts(10.0), 19.4384, atol=1e-4)
    assert np.isclose(common.kts_to_si(19.4384), 10.0, atol=1e-4)


def test_si_to_km_and_back():
    assert np.isclose(common.si_to_km(1000.0), 1.0)
    assert np.isclose(common.km_to_si(1.0), 1000.0)


def test_z_to_dbz_and_back():
    z = 1000.0
    dbz = common.z_to_dbz(z)
    assert np.isclose(dbz, 10 * np.log10(z))
    assert np.isclose(common.dbz_to_z(dbz), z)


def test_meters_kilometers_conversion():
    assert np.isclose(common.meters_to_kilometers(3000), 3.0)
    assert np.isclose(common.kilometers_to_meters(3.0), 3000.0)


def test_ensure_positive_valid():
    assert common.ensure_positive(5.0) == 5.0


def test_ensure_positive_invalid():
    try:
        common.ensure_positive(-1.0)
    except ValueError as e:
        assert str(e) == "value must be positive, got -1.0"
    else:
        assert False, "Expected ValueError for negative input"


def test_si_to_kilometers_units():
    assert np.isclose(common.si_to_kilometers(1000, unit="m"), 1.0)
    assert np.isclose(common.si_to_kilometers(2.0, unit="km"), 2.0)
    try:
        common.si_to_kilometers(100, unit="mi")
    except ValueError as e:
        assert "Unsupported unit" in str(e)
    else:
        assert False, "Expected ValueError for unsupported unit"


def test_kilometers_to_si():
    assert np.isclose(common.kilometers_to_si(3.0), 3000.0)
