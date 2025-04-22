import numpy as np
from radarx.fundamentals import geometry, constants


def test_effective_radius_default():
    r_eff = geometry.effective_radius()
    assert np.isclose(r_eff, constants.EFFECTIVE_RADIUS_4_3, rtol=0.01)


def test_effective_radius_custom():
    r_eff = geometry.effective_radius(dndh=-40e-6)
    assert r_eff > 0


def test_beam_center_height():
    range_m = 10000  # 10 km
    elev = 1.0  # degree
    h0 = 200.0  # radar height in m
    h = geometry.beam_center_height(range_m, elev, h0)
    assert h > h0


def test_sample_volume_gaussian():
    vol = geometry.sample_volume_gaussian(
        range_m=10000, beamwidth_h_deg=1.0, beamwidth_v_deg=1.0, pulse_length_m=300
    )
    assert vol > 0


def test_half_power_radius():
    r = 5000  # 5 km
    bw = 1.0  # degree
    radius = geometry.half_power_radius(r, bw)
    assert np.isclose(radius, (r * np.deg2rad(bw)) / 2.0)
