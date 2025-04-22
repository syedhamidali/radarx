import numpy as np
from radarx.fundamentals import beam


def test_beamwidth_to_radians():
    bw_deg = 1.0
    bw_rad = beam.beamwidth_to_radians(bw_deg)
    assert np.isclose(bw_rad, np.deg2rad(bw_deg))


def test_azimuthal_resolution():
    range_m = 10000  # 10 km
    bw_deg = 1.0
    res = beam.azimuthal_resolution(range_m, bw_deg)
    assert np.isclose(res, range_m * np.deg2rad(bw_deg))


def test_volume_resolution():
    range_m = 5000
    bw_h = 1.0
    bw_v = 1.0
    pulse_length = 150
    vol = beam.volume_resolution(range_m, bw_h, bw_v, pulse_length)
    bw_h_rad = np.deg2rad(bw_h)
    bw_v_rad = np.deg2rad(bw_v)
    expected = range_m**2 * bw_h_rad * bw_v_rad * pulse_length / (4 * np.log(2))
    assert np.isclose(vol, expected)


def test_compute_beamwidth():
    wavelength = 0.1  # 10 cm
    antenna_diameter = 1.0  # 1 meter
    beamwidth = beam.compute_beamwidth(wavelength, antenna_diameter)
    expected = 1.22 * wavelength / antenna_diameter
    assert np.isclose(beamwidth, expected)


def test_compute_volume_resolution():
    range_m = 10000  # 10 km
    beamwidth_rad = np.deg2rad(1.0)  # 1 degree in radians
    pulse_length = 150  # meters
    vol = beam.compute_volume_resolution(range_m, beamwidth_rad, pulse_length)
    az_res = beam.azimuthal_resolution(range_m, np.rad2deg(beamwidth_rad))
    expected = az_res * az_res * pulse_length
    assert np.isclose(vol, expected)
