import numpy as np

from radarx.fundamentals import beam
from radarx.fundamentals.constants import C


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
    expected = 1.27 * wavelength / antenna_diameter
    assert np.isclose(beamwidth, expected)
    assert np.isclose(
        beam.compute_beamwidth(wavelength, antenna_diameter, factor=1.22),
        1.22 * wavelength / antenna_diameter,
    )


def test_compute_beamwidth_is_closer_to_the_wsr88d_beamwidth():
    # WSR-88D, D = 8.54 m: 0.96 deg at 2.7 GHz and 0.88 deg at 3.0 GHz
    # (published values as quoted in issue #196)
    for freq, published in ((2.7e9, 0.96), (3.0e9, 0.88)):
        wavelength = C / freq
        bw = np.rad2deg(beam.compute_beamwidth(wavelength, 8.54))
        bw_null = np.rad2deg(beam.compute_beamwidth(wavelength, 8.54, factor=1.22))
        assert abs(bw / published - 1) < 0.035
        assert abs(bw_null / published - 1) > 0.05


def test_compute_volume_resolution():
    range_m = 10000  # 10 km
    beamwidth_rad = np.deg2rad(1.0)  # 1 degree in radians
    pulse_length = 150  # meters
    vol = beam.compute_volume_resolution(range_m, beamwidth_rad, pulse_length)
    az_res = beam.azimuthal_resolution(range_m, np.rad2deg(beamwidth_rad))
    expected = az_res * az_res * pulse_length
    assert np.isclose(vol, expected)
