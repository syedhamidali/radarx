#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests of the T-matrix scattering of raindrops (radarx.fundamentals.tmatrix)."""

import csv
from pathlib import Path

import numpy as np
import pytest

from radarx.fundamentals import tmatrix as tm

needs_kernel = pytest.mark.skipif(
    not tm.HAS_COMPILED_KERNEL, reason="compiled T-matrix kernel not built"
)

DATA = (
    Path(tm.__file__).resolve().parents[1] / "retrieve" / "data" / "dsd_scattering.csv"
)
BANDS = {"S": 299.792458 / 2.8, "C": 299.792458 / 5.6, "X": 299.792458 / 9.4}

# Amplitude matrices of the FORTRAN code of Mishchenko (double precision: pytmatrix 0.3
# with the REAL*4 storage of the T-matrix changed to REAL*8) for particles in a fixed
# orientation: radius, wavelength (both mm), refractive index, axis ratio, ndgs, geometry
# (thet0, thet, phi0, phi), orientation (alpha, beta), S11, S12, S21, S22 and Z11, Z12, Z33.
FORTRAN = [
    (
        (1.5, 107.0, 8.5 + 1.1j, 1.25, 3, (90, 90, 0, 180), (0.0, 0.0)),
        [
            0.009332755934461751 + 3.094836843315653e-05j,
            0j,
            0j,
            -0.012063578967260331 - 7.275165170689584e-05j,
        ],
        [0.00011631826071794582, -2.92169695842061e-05, -0.00011258868974246796],
    ),
    (
        (3.0, 53.5, 7.0 + 2.2j, 1.6, 3, (90, 90, 0, 0), (20.0, 5.0)),
        [
            0.31274859017724405 + 0.10373278901147479j,
            0.005746563995435517 + 0.003519518740847228j,
            0.005746563995435515 + 0.0035195187408472266j,
            0.5046224745592994 + 0.2212471388904965j,
        ],
        [0.20612856524568798, -0.0975109830618143, 0.18081596027801425],
    ),
    (
        (2.2, 31.9, 8.1 + 2.0j, 1.15, 3, (70, 40, 20, 100), (30.0, 70.0)),
        [
            0.288740122486579 + 0.06789682565609398j,
            0.31122940070220045 + 0.10188123671611227j,
            -0.04971008665680038 - 0.15101375032885572j,
            -0.0013630272251588683 + 0.12559254304904377j,
        ],
        [0.11813797687027452, -0.00488089409872328, -0.02272293377253097],
    ),
    (
        (4.0, 31.9, 1.78 + 0.01j, 0.7, 3, (90, 90, 0, 180), (45.0, 60.0)),
        [
            0.8683507985452193 + 0.15706362690011844j,
            -0.06654717051087966 - 0.025215190418176825j,
            0.06654717805271881 + 0.02521519299728786j,
            -0.8955185618886146 - 0.16735768323499162j,
        ],
        [0.8093964228247379, -0.02562999773092728, -0.8089743953442836],
    ),
]


@needs_kernel
@pytest.mark.parametrize("case, s_ref, z_ref", FORTRAN)
def test_fixed_orientation_matches_the_fortran_code(case, s_ref, z_ref):
    radius, wavelength, m, eps, ndgs, geometry, angles = case
    s, z, nmax = tm.spheroid_scattering(
        radius, eps, wavelength, m, geometry, [[*angles, 1.0]], ndgs=ndgs
    )
    scale = np.abs(s_ref).max()
    np.testing.assert_allclose(s[0, 0].ravel(), s_ref, rtol=0, atol=1e-9 * scale)
    np.testing.assert_allclose(
        [z[0, 0, 0, 0], z[0, 0, 0, 1], z[0, 0, 2, 2]],
        z_ref,
        rtol=0,
        atol=1e-9 * abs(z_ref[0]),
    )
    assert nmax[0] >= 5


def read_table():
    rows = [
        r
        for r in csv.reader(line for line in DATA.open() if not line.startswith("#"))
        if r and r[0] != "band"
    ]
    return rows


@needs_kernel
def test_reproduces_the_dsd_table_built_with_pytmatrix():
    # the table of radarx.retrieve.dsd was built with pytmatrix (FORTRAN); every 11th row
    rows = read_table()[::11]
    names = ("sigma_h", "sigma_v", "copol_re", "copol_im", "kdp", "ah", "av")
    groups = {}
    for r in rows:
        groups.setdefault((r[0], float(r[2])), []).append(r)
    for (band, temperature), group in groups.items():
        diameters = np.array([float(r[3]) for r in group])
        got = tm.raindrop_scattering(
            diameters,
            BANDS[band],
            temperature=temperature,
            canting_std=7.0,
            ndgs=4,
            ddelt=1e-3,
        )
        for j, name in enumerate(names):
            want = np.array([float(r[4 + j]) for r in group])
            # the file keeps 7 digits: compare to a millionth of the largest value
            np.testing.assert_allclose(
                got[name].values, want, rtol=1e-5, atol=2e-6 * np.abs(want).max()
            )


@needs_kernel
def test_a_small_sphere_follows_the_rayleigh_law():
    wavelength = 107.0
    m = tm.water_refractive_index(wavelength, 20.0)
    d = 0.5  # mm, size parameter 0.015
    out = tm.raindrop_scattering(
        [d], wavelength, m=m, axis_ratio=1.0, canting_std=7.0, ndgs=2
    )
    k2 = abs((m**2 - 1.0) / (m**2 + 2.0)) ** 2
    rayleigh = np.pi**5 / wavelength**4 * k2 * d**6
    np.testing.assert_allclose(out.sigma_h.values[0], rayleigh, rtol=2e-3)
    np.testing.assert_allclose(out.sigma_h.values[0], out.sigma_v.values[0], rtol=1e-9)
    assert (
        abs(out.kdp.values[0]) < 1e-9
    )  # a sphere does not shift the phase differentially


@needs_kernel
def test_oblate_drops_have_positive_zdr_and_kdp_that_grow_with_size():
    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = tm.raindrop_scattering(d, 299.792458 / 2.8)
    zdr = 10 * np.log10(out.sigma_h / out.sigma_v)
    assert (np.diff(zdr.values) > 0).all() and zdr.values[0] > 0
    assert (out.kdp.values > 0).all() and (np.diff(out.kdp.values) > 0).all()
    assert (out.ah.values >= out.av.values).all()


@needs_kernel
def test_more_canting_lowers_zdr():
    d = np.array([3.0, 5.0])
    wavelength = 299.792458 / 2.8
    zdr = []
    for std in (0.5, 7.0, 20.0):
        out = tm.raindrop_scattering(d, wavelength, canting_std=std)
        zdr.append(10 * np.log10(out.sigma_h / out.sigma_v).values)
    assert (np.diff(np.array(zdr), axis=0) < 0).all()


@needs_kernel
def test_threads_do_not_change_the_result():
    d = np.arange(0.5, 6.1, 0.5)
    one = tm.raindrop_scattering(d, 299.792458 / 5.6, n_threads=1)
    many = tm.raindrop_scattering(d, 299.792458 / 5.6, n_threads=4)
    for name in one.data_vars:
        np.testing.assert_array_equal(one[name].values, many[name].values)


@needs_kernel
def test_a_size_parameter_that_is_too_large_raises():
    with pytest.raises(RuntimeError, match="too large"):
        tm.spheroid_scattering(
            200.0, 1.2, 10.0, 8 + 2j, (90, 90, 0, 180), [[0, 0, 1.0]]
        )


@needs_kernel
def test_invalid_orientation_weights_and_angles():
    with pytest.raises(ValueError, match="weights"):
        tm.spheroid_scattering(1.0, 1.2, 100.0, 8 + 2j, (90, 90, 0, 180), [[0, 0, 0.0]])
    with pytest.raises(RuntimeError, match="angle"):
        tm.spheroid_scattering(
            1.0, 1.2, 100.0, 8 + 2j, (90, 90, 0, 180), [[0, 200.0, 1.0]]
        )


def test_orientation_quadrature_integrates_the_canting_density():
    q = tm.orientation_quadrature(std=10.0, mean=0.0, n_alpha=4, n_beta=10)
    assert q.shape == (40, 3)
    beta = np.unique(q[:, 1])
    assert beta.size == 10 and beta.min() > 0 and beta.max() < 180
    w = q[q[:, 0] == 0.0, 2]
    # the expectation of cos(beta) under the density sin(b) exp(-b^2 / 2 s^2), by dense integration
    b = np.linspace(0, 180, 200001)
    f = np.sin(np.radians(b)) * np.exp(-0.5 * (b / 10.0) ** 2)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    want = trapezoid(np.cos(np.radians(b)) * f, b) / trapezoid(f, b)
    got = np.sum(np.cos(np.radians(q[q[:, 0] == 0.0, 1])) * w) / w.sum()
    np.testing.assert_allclose(got, want, rtol=1e-6)
    uniform = tm.orientation_quadrature(distribution="uniform", n_alpha=1, n_beta=6)
    # random orientation: the mean of cos(beta)^2 is 1/3
    np.testing.assert_allclose(
        np.sum(np.cos(np.radians(uniform[:, 1])) ** 2 * uniform[:, 2])
        / uniform[:, 2].sum(),
        1.0 / 3.0,
        rtol=1e-6,
    )
    with pytest.raises(ValueError):
        tm.orientation_quadrature(distribution="nope")


def test_water_refractive_index_and_axis_ratio():
    m = tm.water_refractive_index(299.792458 / 2.8, 20.0)
    assert m.real > 8.0 and 0.5 < m.imag < 2.5
    assert (
        tm.water_refractive_index(31.9, 0.0).imag
        > tm.water_refractive_index(31.9, 30.0).imag
    )
    r = tm.axis_ratio_brandes([0.1, 3.0, 8.0])
    assert r[0] == pytest.approx(0.9976, abs=1e-3) and (r <= 1.0).all() and r[1] > r[2]


def test_there_is_no_numpy_engine():
    with pytest.raises(ValueError, match="no NumPy version"):
        tm.raindrop_scattering([1.0], 107.0, engine="numpy")
    with pytest.raises(ValueError, match="engine must be"):
        tm.spheroid_scattering(
            1.0, 1.2, 100.0, 8 + 2j, (90, 90, 0, 180), [[0, 0, 1.0]], engine="fortran"
        )


def test_without_the_kernel_the_functions_say_so(monkeypatch):
    monkeypatch.setattr(tm, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError, match="compiled kernel"):
        tm.raindrop_scattering([1.0], 107.0)


def test_a_non_positive_canting_width_raises():
    with pytest.raises(ValueError, match="std must be positive"):
        tm.orientation_quadrature(std=0.0)


@needs_kernel
def test_axis_ratio_can_be_a_function_of_the_diameter():
    d = [1.0, 3.0]
    own = tm.raindrop_scattering(d, 107.0, axis_ratio=lambda x: np.full(len(x), 0.8))
    fixed = tm.raindrop_scattering(d, 107.0, axis_ratio=0.8)
    np.testing.assert_allclose(own.sigma_h, fixed.sigma_h)
    assert (own.sigma_h != tm.raindrop_scattering(d, 107.0).sigma_h).all()
