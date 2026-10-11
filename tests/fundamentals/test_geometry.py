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
        s = geometry.ground_range(rng, el, radar_height=140.0)
        x, y, z = antenna_to_cartesian(rng, 0.0, el, site_altitude=140.0)
        # the same Earth-curvature model as xradar, with the site altitude in
        # the Earth radius: no difference left
        np.testing.assert_allclose(s, np.hypot(x, y), rtol=1e-9)
        h = geometry.beam_height_at_ground_range(s, el, 140.0)
        np.testing.assert_allclose(h, z, atol=1e-6)
        np.testing.assert_allclose(h, geometry.beam_center_height(rng, el, 140.0))
        np.testing.assert_allclose(
            geometry.beam_center_height(rng, el, 140.0), z, atol=1e-6
        )
    # near the radar the ground range is the horizontal range, the height the rise
    assert np.isclose(geometry.ground_range(1000.0, 0.0), 1000.0, rtol=1e-6)
    assert geometry.beam_height_at_ground_range(0.0, 10.0, 25.0) == pytest.approx(25.0)
    # the Earth radius can be changed
    assert geometry.ground_range(50e3, 1.0, reff=6.371e6) != geometry.ground_range(
        50e3, 1.0
    )


def test_site_altitude_is_part_of_the_earth_radius():
    # 200 km, 0.5 deg, 2 km site: sqrt(r^2 + (R + alt)^2 + 2 r (R + alt) sin(e)) - R
    r, el, alt = 200e3, 0.5, 2000.0
    a = geometry.EFFECTIVE_RADIUS_4_3 + alt
    expected = (
        np.sqrt(r**2 + a**2 + 2 * r * a * np.sin(np.deg2rad(el)))
        - geometry.EFFECTIVE_RADIUS_4_3
    )
    assert geometry.beam_center_height(r, el, alt) == pytest.approx(expected)
    # not the old form, which differs by about half a metre here
    old = (
        np.sqrt(
            r**2
            + geometry.EFFECTIVE_RADIUS_4_3**2
            + 2 * r * geometry.EFFECTIVE_RADIUS_4_3 * np.sin(np.deg2rad(el))
        )
        - geometry.EFFECTIVE_RADIUS_4_3
        + alt
    )
    assert abs(geometry.beam_center_height(r, el, alt) - old) > 0.3


def test_geometry_agrees_with_the_gridding_code():
    from radarx.grid.multi import _cell_angles

    for alt in (0.0, 140.0, 2000.0):
        for el in (0.5, 4.0, 19.5):
            rng = np.array([5e3, 60e3, 200e3])
            s = geometry.ground_range(rng, el, radar_height=alt)
            h = geometry.beam_center_height(rng, el, alt)
            for k in range(rng.size):
                slant, _, antenna = _cell_angles(
                    np.array([s[k]]), np.array([h[k]]), alt
                )
                assert slant[0, 0] == pytest.approx(rng[k], abs=1e-3)
                assert antenna[0, 0] == pytest.approx(el, abs=1e-6)


def test_k_e_from_refractivity_standard_atmosphere():
    # -39 N-units per km is the standard gradient: k_e close to 4/3
    height = np.array([0.0, 500.0, 1000.0, 3000.0])
    n = 300.0 - 39e-3 * height
    assert geometry.k_e_from_refractivity(height, n) == pytest.approx(
        geometry.effective_radius(-39e-6) / constants.EARTH_RADIUS
    )
    assert geometry.k_e_from_refractivity(height, n) == pytest.approx(4 / 3, abs=0.01)


def test_k_e_from_refractivity_sub_and_super_refraction():
    height = np.array([0.0, 1000.0])
    sub = geometry.k_e_from_refractivity(height, np.array([300.0, 300.0]))
    sup = geometry.k_e_from_refractivity(height, np.array([300.0, 200.0]))
    assert sub == pytest.approx(1.0, abs=0.01)  # no gradient: true Earth radius
    assert sup > 4 / 3
    # layer and unsorted profile
    n = np.array([280.0, 300.0, 250.0])
    h = np.array([1000.0, 0.0, 3000.0])
    assert geometry.k_e_from_refractivity(h, n, layer=(0.0, 1000.0)) == pytest.approx(
        geometry.k_e_from_refractivity([0.0, 1000.0], [300.0, 280.0])
    )


def test_k_e_from_refractivity_errors():
    with pytest.raises(ValueError, match="two valid levels"):
        geometry.k_e_from_refractivity([0.0], [300.0])
    with pytest.raises(ValueError, match="same length"):
        geometry.k_e_from_refractivity([0.0, 1.0], [300.0])
    with pytest.raises(ValueError, match="positive depth"):
        geometry.k_e_from_refractivity([0, 1000.0], [300, 280.0], layer=(5.0, 5.0))
    with pytest.raises(ValueError, match="does not reach"):
        geometry.k_e_from_refractivity([0, 1000.0], [300, 280.0], layer=(2e3, 3e3))
    with pytest.raises(ValueError, match="trapping"):
        geometry.k_e_from_refractivity([0, 1000.0], [300, 100.0])
