# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Physical checks of the T-matrix kernel at S, C, X, Ku, Ka and W band.

The references are independent of the code: the Mie solution of Bohren and Huffman
(recursions written here), the optical theorem, reciprocity, rotation about the vertical
and the Rayleigh limit with the depolarization factors of spheroids.
"""

from math import lgamma

import numpy as np
import pytest

from radarx.fundamentals import tmatrix as tm

pytestmark = pytest.mark.skipif(
    not tm.HAS_COMPILED_KERNEL, reason="compiled T-matrix kernel not built"
)

BANDS = {"S": 107.0, "C": 53.5, "X": 31.9, "Ku": 22.1, "Ka": 8.4, "W": 3.19}


def mie_coefficients(m, x):
    """Mie coefficients a_n, b_n (Bohren and Huffman, downward recurrence)."""
    nmax = int(round(x + 4.05 * x ** (1 / 3) + 2))
    mx = m * x
    nmx = int(round(max(nmax, abs(mx)) + 16))
    d = np.zeros(nmx + 1, dtype=complex)
    for n in range(nmx, 0, -1):
        d[n - 1] = n / mx - 1.0 / (d[n] + n / mx)
    a = np.zeros(nmax, dtype=complex)
    b = np.zeros(nmax, dtype=complex)
    psi0, psi1 = np.cos(x), np.sin(x)
    chi0, chi1 = -np.sin(x), np.cos(x)
    xi1 = complex(psi1, -chi1)
    for n in range(1, nmax + 1):
        psi = (2 * n - 1) / x * psi1 - psi0
        chi = (2 * n - 1) / x * chi1 - chi0
        xi = complex(psi, -chi)
        a[n - 1] = ((d[n] / m + n / x) * psi - psi1) / ((d[n] / m + n / x) * xi - xi1)
        b[n - 1] = ((d[n] * m + n / x) * psi - psi1) / ((d[n] * m + n / x) * xi - xi1)
        psi0, psi1, chi0, chi1 = psi1, psi, chi1, chi
        xi1 = complex(psi1, -chi1)
    return a, b


def mie_amplitudes(m, x, theta):
    """Bohren-Huffman S1, S2 at the scattering angles ``theta`` (degrees)."""
    a, b = mie_coefficients(m, x)
    mu = np.cos(np.radians(np.atleast_1d(theta)))
    s1 = np.zeros(mu.shape, dtype=complex)
    s2 = np.zeros(mu.shape, dtype=complex)
    pi0, pi1 = np.zeros_like(mu), np.ones_like(mu)
    for n in range(1, len(a) + 1):
        tau = n * mu * pi1 - (n + 1) * pi0
        f = (2 * n + 1) / (n * (n + 1))
        s1 += f * (a[n - 1] * pi1 + b[n - 1] * tau)
        s2 += f * (a[n - 1] * tau + b[n - 1] * pi1)
        pi0, pi1 = pi1, ((2 * n + 1) * mu * pi1 - (n + 1) * pi0) / n
    return s1, s2


def mie_extinction(m, x):
    a, b = mie_coefficients(m, x)
    n = np.arange(1, len(a) + 1)
    return 2 / x**2 * np.sum((2 * n + 1) * (a + b).real)


def sphere_quadrature(n_theta, n_phi):
    """Gauss-Legendre in the polar angle, uniform in azimuth: angles (deg) and weights."""
    nodes, weights = np.polynomial.legendre.leggauss(n_theta)
    theta = np.degrees(np.arccos(nodes))
    phi = (np.arange(n_phi) + 0.5) * 360.0 / n_phi
    t, p = np.meshgrid(theta, phi, indexing="ij")
    w = np.outer(weights, np.full(n_phi, 2 * np.pi / n_phi))
    return t.ravel(), p.ravel(), w.ravel()


@pytest.mark.parametrize("band", BANDS)
def test_spheres_match_the_mie_solution(band):
    wavelength = BANDS[band]
    m = tm.water_refractive_index(wavelength, 10.0)
    k = 2 * np.pi / wavelength
    angles = np.array([0.0, 30.0, 90.0, 150.0, 180.0])
    geometry = np.array([[90.0, 90.0, 0.0, a] for a in angles])
    for diameter in (0.5, 3.0, 8.0):
        radius = diameter / 2
        s, z, _ = tm.spheroid_scattering(
            radius, 1.0, wavelength, m, geometry, [[0.0, 0.0, 1.0]]
        )
        s1, s2 = mie_amplitudes(m, k * radius, angles)
        z11 = 0.5 * (abs(s1) ** 2 + abs(s2) ** 2) / k**2
        z12 = 0.5 * (abs(s2) ** 2 - abs(s1) ** 2) / k**2
        np.testing.assert_allclose(z[0, :, 0, 0], z11, rtol=0, atol=1e-7 * z11.max())
        np.testing.assert_allclose(
            np.abs(z[0, :, 0, 1]), np.abs(z12), rtol=0, atol=1e-7 * z11.max()
        )
        # extinction from the forward amplitude (optical theorem), absorbing sphere
        extinction = 4 * np.pi / k * s[0, 0, 0, 0].imag
        reference = mie_extinction(m, k * radius) * np.pi * radius**2
        assert extinction == pytest.approx(reference, rel=1e-7)


@pytest.mark.parametrize("band", ["X", "Ka", "W"])
def test_spheres_of_every_size_match_mie_including_the_zeros_of_cos_x(band):
    # the scale of the spherical Bessel functions was fixed through cos(x), which lost all
    # accuracy at x = pi/2, 3 pi/2, ... (a drop of half a wavelength: 4.2 mm at 8.4 mm)
    wavelength = BANDS[band]
    m = tm.water_refractive_index(wavelength, 10.0)
    k = 2 * np.pi / wavelength
    x = np.concatenate([np.linspace(0.05, 8.0, 400), np.arange(1, 17) * np.pi / 2])
    x = x[x / k <= 5.0]
    _, z, _ = tm.spheroid_scattering(
        x / k, 1.0, wavelength, m, [[90, 90, 0, 180]], [[0.0, 0.0, 1.0]]
    )
    z = z[:, 0]
    sigma = 2 * np.pi * (z[:, 0, 0] - z[:, 0, 1] - z[:, 1, 0] + z[:, 1, 1])
    mie = np.array(
        [4 * np.pi * abs(mie_amplitudes(m, xi, [180.0])[0][0]) ** 2 / k**2 for xi in x]
    )
    np.testing.assert_allclose(sigma, mie, rtol=0, atol=1e-6 * mie.max())


@pytest.mark.parametrize("band", ["S", "X", "Ka", "W"])
def test_scattering_integral_equals_extinction_without_absorption(band):
    wavelength = BANDS[band]
    m = complex(1.33, 1e-12)
    k = 2 * np.pi / wavelength
    for diameter, ratio in ((1.0, 1.1), (4.0, 1.4), (7.0, 1.8)):
        radius = diameter / 2
        theta, phi, w = sphere_quadrature(int(4 * np.pi * radius / wavelength) + 24, 48)
        geometry = np.vstack(
            [
                [90.0, 90.0, 0.0, 0.0],
                np.column_stack(
                    [np.full_like(theta, 90.0), theta, np.zeros_like(theta), phi]
                ),
            ]
        )
        s, z, _ = tm.spheroid_scattering(
            radius,
            ratio,
            wavelength,
            m,
            geometry,
            [[37.0, 63.0, 1.0]],
            ddelt=1e-4,
            ndgs=4,
        )
        zs = z[0, 1:]
        # the first Stokes parameter of a wave polarized along the first (second) axis
        scattering_first = np.sum(w * (zs[:, 0, 0] + zs[:, 0, 1]))
        scattering_second = np.sum(w * (zs[:, 0, 0] - zs[:, 0, 1]))
        extinction_first = 4 * np.pi / k * s[0, 0, 0, 0].imag
        extinction_second = 4 * np.pi / k * s[0, 0, 1, 1].imag
        assert scattering_first == pytest.approx(extinction_first, rel=1e-4)
        assert scattering_second == pytest.approx(extinction_second, rel=1e-4)


@pytest.mark.parametrize("band", BANDS)
def test_backscatter_is_reciprocal(band):
    wavelength = BANDS[band]
    m = tm.water_refractive_index(wavelength, 10.0)
    rng = np.random.default_rng(3)
    for diameter in (1.0, 4.0, 8.0):
        ratio = 1.0 / float(tm.axis_ratio_brandes(diameter))
        alpha, beta = rng.uniform(0, 360), rng.uniform(0, 180)
        s, _, _ = tm.spheroid_scattering(
            diameter / 2, ratio, wavelength, m, [[90, 90, 0, 180]], [[alpha, beta, 1.0]]
        )
        assert abs(s[0, 0, 0, 1] + s[0, 0, 1, 0]) <= 1e-6 * np.abs(s).max()


@pytest.mark.parametrize("band", ["C", "Ka", "W"])
def test_reciprocity_in_general_directions(band):
    # S(n, n0) and S(-n0, -n) are related by transposition with a sign for the
    # off-diagonal elements (Mishchenko's basis vectors change sign under inversion)
    wavelength = BANDS[band]
    m = tm.water_refractive_index(wavelength, 10.0)
    rng = np.random.default_rng(5)
    for diameter in (1.0, 3.0, 6.0):
        ratio = 1.0 / float(tm.axis_ratio_brandes(diameter))
        alpha, beta = rng.uniform(0, 360), rng.uniform(0, 180)
        t0, t = rng.uniform(10, 170, 2)
        p0, p = rng.uniform(0, 180), rng.uniform(180, 360)
        args = (diameter / 2, ratio, wavelength, m)
        kwargs = dict(ddelt=1e-6, ndgs=8)
        s, _, _ = tm.spheroid_scattering(
            *args, [[t0, t, p0, p]], [[alpha, beta, 1.0]], **kwargs
        )
        s2, _, _ = tm.spheroid_scattering(
            *args,
            [[180 - t, 180 - t0, (p + 180) % 360, (p0 + 180) % 360]],
            [[alpha, beta, 1.0]],
            **kwargs,
        )
        a, b = s[0, 0], s2[0, 0]
        violation = max(
            abs(a[0, 0] - b[0, 0]),
            abs(a[1, 1] - b[1, 1]),
            abs(a[0, 1] + b[1, 0]),
            abs(a[1, 0] + b[0, 1]),
        )
        assert violation <= 5e-6 * np.abs(a).max()


@pytest.mark.parametrize("band", ["S", "Ku", "W"])
def test_turning_particle_and_directions_together_changes_nothing(band):
    wavelength = BANDS[band]
    m = tm.water_refractive_index(wavelength, 10.0)
    for turn in (17.0, 123.0, 300.0):
        base, _, _ = tm.spheroid_scattering(
            2.0, 1.3, wavelength, m, [[70, 100, 20, 200]], [[40.0, 70.0, 1.0]]
        )
        turned, _, _ = tm.spheroid_scattering(
            2.0,
            1.3,
            wavelength,
            m,
            [[70, 100, (20 + turn) % 360, (200 + turn) % 360]],
            [[(40.0 + turn) % 360, 70.0, 1.0]],
        )
        assert np.abs(turned - base).max() <= 1e-6 * np.abs(base).max()


def test_the_azimuth_of_a_vertical_axis_does_not_matter():
    m = tm.water_refractive_index(31.9, 10.0)
    first, _, _ = tm.spheroid_scattering(
        2.0, 1.4, 31.9, m, [[90, 90, 0, 180]], [[0.0, 0.0, 1.0]]
    )
    second, _, _ = tm.spheroid_scattering(
        2.0, 1.4, 31.9, m, [[90, 90, 0, 180]], [[123.0, 0.0, 1.0]]
    )
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-12 * np.abs(first).max())


def depolarization_factors(ratio):
    """Depolarization factors of an oblate spheroid (axis ratio horizontal/vertical > 1)."""
    e2 = ratio**2 - 1.0
    e = np.sqrt(e2)
    vertical = (1 + e2) / e2 * (1 - np.arctan(e) / e)
    return (1 - vertical) / 2, vertical


@pytest.mark.parametrize("band", ["S", "X", "Ku", "W"])
def test_small_spheroids_follow_the_rayleigh_limit(band):
    wavelength = BANDS[band]
    m = tm.water_refractive_index(wavelength, 20.0)
    eps = m * m
    errors = {}
    for ratio in (1.1, 1.5, 2.5):
        for size in (0.02, 0.04):
            radius = size * wavelength / (2 * np.pi)
            _, z, _ = tm.spheroid_scattering(
                radius, ratio, wavelength, m, [[90, 90, 0, 180]], [[0.0, 0.0, 1.0]]
            )
            z = z[0, 0]
            horizontal = 2 * np.pi * (z[0, 0] - z[0, 1] - z[1, 0] + z[1, 1])
            vertical = 2 * np.pi * (z[0, 0] + z[0, 1] + z[1, 0] + z[1, 1])
            lx, lz = depolarization_factors(ratio)
            volume = 4 / 3 * np.pi * radius**3
            k = 2 * np.pi / wavelength
            rayleigh = [
                4
                * np.pi
                * k**4
                * abs(volume * (eps - 1) / (1 + factor * (eps - 1)) / (4 * np.pi)) ** 2
                for factor in (lx, lz)
            ]
            errors[ratio, size] = (
                horizontal / rayleigh[0] - 1,
                vertical / rayleigh[1] - 1,
            )
    for ratio in (1.1, 1.5, 2.5):
        for pair in range(2):
            coarse, fine = errors[ratio, 0.04][pair], errors[ratio, 0.02][pair]
            assert abs(coarse) < 0.01
            # the departure from the Rayleigh limit vanishes with the square of the size
            # (the relative error at the smaller size is dominated by rounding in W band)
            if abs(coarse) > 2e-3:
                assert abs(coarse / fine) == pytest.approx(4.0, rel=0.25)


@pytest.mark.parametrize("band", BANDS)
def test_the_default_accuracy_converges_the_raindrop_quantities(band):
    wavelength = BANDS[band]
    diameter = np.arange(0.2, 8.01, 0.3)
    default = tm.raindrop_scattering(diameter, wavelength)
    reference = tm.raindrop_scattering(diameter, wavelength, ddelt=1e-7, ndgs=8)
    for name in ("sigma_h", "sigma_v", "ah", "av"):
        np.testing.assert_allclose(default[name], reference[name], rtol=2e-3)
    # K_DP crosses zero at short wavelengths: compare to the largest value
    scale = np.abs(reference.kdp).max()
    np.testing.assert_allclose(default.kdp, reference.kdp, rtol=0, atol=2e-3 * scale)


def test_the_accuracy_of_the_fortran_code_is_not_enough_at_w_band():
    diameter = np.arange(5.5, 7.01, 0.1)
    loose = tm.raindrop_scattering(diameter, BANDS["W"], ddelt=1e-3)
    reference = tm.raindrop_scattering(diameter, BANDS["W"], ddelt=1e-7, ndgs=8)
    error = np.abs(loose.sigma_v / reference.sigma_v - 1).max()
    assert error > 0.05


def gamma_integral(table, name, mean_diameter, shape):
    diameter = table.diameter.values
    f = 6 * (3.67 + shape) ** (shape + 4) / (3.67**4 * np.exp(lgamma(shape + 4)))
    x = diameter / mean_diameter
    n = f * x**shape * np.exp(-(3.67 + shape) * x)
    return np.sum(n * table[name].values) * (diameter[1] - diameter[0])


def test_specific_attenuation_over_differential_phase_grows_with_frequency():
    # A_h / K_DP of gamma drop size distributions in rain at 20 degC: about 0.015 to
    # 0.025 dB/deg at S band (Ryzhkov et al. 2014), the familiar 0.08 at C band
    diameter = np.arange(0.1, 8.01, 0.1)
    ratio = {}
    for band, wavelength in (("S", 110.0), ("C", 53.0), ("X", 32.0), ("Ku", 22.04)):
        table = tm.raindrop_scattering(diameter, wavelength, temperature=20.0)
        values = [
            gamma_integral(table, "ah", dm, mu) / gamma_integral(table, "kdp", dm, mu)
            for mu in (0.0, 3.0, 6.0)
            for dm in (1.0, 1.5, 2.0, 2.5)
        ]
        ratio[band] = np.median(values)
    assert 0.012 < ratio["S"] < 0.025
    assert 0.06 < ratio["C"] < 0.11
    assert 0.22 < ratio["X"] < 0.38
    assert ratio["S"] < ratio["C"] < ratio["X"] < ratio["Ku"]
