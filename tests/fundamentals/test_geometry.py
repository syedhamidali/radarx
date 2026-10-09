import numpy as np
import pytest

from radarx.fundamentals import constants, geometry


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


def test_ground_range_and_beam_height_at_ground_range():
    from xradar.georeference import antenna_to_cartesian

    rng = np.array([2e3, 30e3, 120e3, 300e3])
    for el in (0.5, 4.0, 19.5):
        s = geometry.ground_range(rng, el)
        x, y, z = antenna_to_cartesian(rng, 0.0, el, site_altitude=140.0)
        # xradar projects the arc to x and y slightly differently (2e-5 relative)
        np.testing.assert_allclose(s, np.hypot(x, y), rtol=5e-5)
        h = geometry.beam_height_at_ground_range(s, el, 140.0)
        np.testing.assert_allclose(h, z, atol=0.5)
        np.testing.assert_allclose(h, geometry.beam_center_height(rng, el, 140.0))
    # near the radar the ground range is the horizontal range, the height the rise
    assert np.isclose(geometry.ground_range(1000.0, 0.0), 1000.0, rtol=1e-6)
    assert geometry.beam_height_at_ground_range(0.0, 10.0, 25.0) == pytest.approx(25.0)
    # the Earth radius can be changed
    assert geometry.ground_range(50e3, 1.0, reff=6.371e6) != geometry.ground_range(
        50e3, 1.0
    )
