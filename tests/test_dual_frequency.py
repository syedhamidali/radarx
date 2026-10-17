#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the dual-frequency ratio retrievals
=============================================

The look-up of the retrievals is checked against forward models with known truth:
rain of known size distribution at S and C band (the reflectivity of
:func:`radarx.retrieve.radar_from_dsd`, which has its own scattering tables) and at Ka
and W band (T-matrix scattering of the test), snow with an independent Mie code, and
the Rayleigh limit of the ratio against the dielectric factors of water. The two
engines agree and the error paths raise.
"""

import numpy as np
import pytest
import xarray as xr
from scipy.special import gammaln, spherical_jn, spherical_yn

import radarx  # noqa: F401  registers the accessors
from radarx.fundamentals import tmatrix
from radarx.retrieve import dfr_rain, dfr_snow, dual_frequency_ratio
from radarx.retrieve import dual_frequency as df

# the tables of the retrievals are built with the T-matrix kernel
pytestmark = pytest.mark.skipif(
    not tmatrix.HAS_COMPILED_KERNEL, reason="T-matrix kernel not built"
)
compiled_only = pytest.mark.skipif(
    not df.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)
ENGINES = ["numpy", pytest.param("compiled", marks=compiled_only)]


def normalized_gamma(d, dm, mu, nw):
    f_mu = 6.0 / 4.0**4 * (mu + 4.0) ** (mu + 4.0) / np.exp(gammaln(mu + 4.0))
    x = d / dm
    return nw * f_mu * x**mu * np.exp(-(4.0 + mu) * x)


# ---- the ratio -------------------------------------------------------------------


def two_fields():
    time = np.arange(5)
    rng = np.arange(0.0, 400.0, 100.0)
    ka = xr.DataArray(
        np.arange(20.0).reshape(5, 4),
        dims=("time", "range"),
        coords={"time": time, "range": rng},
        name="DBZ",
    )
    w = ka - 3.0
    return ka, w


def test_the_ratio_is_the_difference_with_the_corrections():
    ka, w = two_fields()
    out = dual_frequency_ratio(ka, w)
    assert out.name == "DFR" and out.attrs["units"] == "dB"
    np.testing.assert_allclose(out, 3.0)
    assert out.dtype == np.float32
    corrected = dual_frequency_ratio(ka, w, pia_ka=0.5, pia_w=xr.full_like(w, 2.5))
    np.testing.assert_allclose(corrected, 3.0 + 0.5 - 2.5)
    assert corrected.dims == ka.dims


def test_exact_matching_keeps_the_common_coordinates():
    ka, w = two_fields()
    out = dual_frequency_ratio(ka, w.isel(time=slice(1, 4), range=slice(0, 3)))
    assert out.sizes == {"time": 3, "range": 3}


def test_nearest_matching_uses_the_tolerance():
    ka, w = two_fields()
    shifted = w.assign_coords(time=w.time + 0.3, range=w.range + 20.0)
    out = dual_frequency_ratio(
        ka, shifted, match="nearest", tolerance={"time": 0.5, "range": 50.0}
    )
    np.testing.assert_allclose(out, 3.0)
    strict = dual_frequency_ratio(ka, shifted, match="nearest", tolerance={"time": 0.2})
    assert np.isnan(strict).all()
    loose = dual_frequency_ratio(ka, shifted, match="nearest")
    assert np.isfinite(loose).all() and loose.dims == ka.dims


def test_interpolated_matching_is_linear():
    ka, w = two_fields()
    w_coarse = w.isel(range=slice(0, None, 2))
    out = dual_frequency_ratio(ka.isel(range=slice(0, 3)), w_coarse, match="interp")
    # the field is linear in time and range here; gates beyond the coarse grid are NaN
    assert out.sizes["range"] == 3
    np.testing.assert_allclose(out.isel(range=[0, 1]), 3.0, atol=1e-5)


def test_ratio_errors():
    ka, w = two_fields()
    with pytest.raises(ValueError, match="match"):
        dual_frequency_ratio(ka, w, match="best")
    with pytest.raises(TypeError, match="dbz_w"):
        dual_frequency_ratio(ka, w.values)
    with pytest.raises(TypeError, match="dbz_ka"):
        dual_frequency_ratio(ka.values, w)


# ---- the look-up -------------------------------------------------------------------


def test_the_look_up_kernel_and_reference_agree_and_handle_the_edges():
    table = np.vstack(
        [[0.0, 1.0, 3.0, 4.0], [0.5, 1.0, 2.0, 5.0], [10.0, 9.0, 7.0, 6.0]]
    )
    x = np.array([-2.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.9, 4.0, 4.1, np.nan, 10.0])
    ref = df._lookup_numpy(x, table, 1.0)
    expected_first = [
        np.nan,
        0.5,
        0.5,
        0.75,
        1.0,
        1.5,
        4.7,
        5.0,
        np.nan,
        np.nan,
        np.nan,
    ]
    np.testing.assert_allclose(ref[0], expected_first)
    np.testing.assert_allclose(ref[1][1], 10.0)
    if df.HAS_COMPILED_KERNEL:
        for threads in (1, 3):
            got = df._lookup(x, table, 1.0, True, threads)
            np.testing.assert_allclose(got, ref, equal_nan=True)
    assert df._lookup(x, table, 1.0, False, None).shape == (2, x.size)


@compiled_only
def test_look_up_kernel_input_checks():
    from radarx.retrieve import _dfr

    lut = np.array([[0.0, 1.0], [1.0, 2.0]])
    with pytest.raises(ValueError, match="1-D"):
        _dfr.lookup(np.ones((2, 2)), lut[0], lut[1:])
    with pytest.raises(ValueError, match="at least 2"):
        _dfr.lookup(np.ones(2), np.ones(1), np.ones((1, 1)))
    with pytest.raises(ValueError, match="shape"):
        _dfr.lookup(np.ones(2), lut[0], np.ones((1, 3)))
    with pytest.raises(ValueError, match="increase"):
        _dfr.lookup(np.ones(2), np.array([1.0, 1.0]), lut[1:])


def test_a_table_needs_a_rising_ratio():
    with pytest.raises(ValueError, match="increase"):
        df._monotonic_segment(np.array([3.0, 2.0, 1.0]))
    keep = df._monotonic_segment(np.array([1.0, 0.5, 0.7, 2.0, 3.0, 2.5]))
    assert keep == slice(1, 5)


# ---- rain ---------------------------------------------------------------------------


def test_rain_ratio_at_small_sizes_is_the_ratio_of_the_dielectric_factors():
    from radarx.fundamentals.tmatrix import water_refractive_index

    def k2(frequency):
        m = water_refractive_index(299.792458 / frequency, 10.0)
        k = ((m**2 - 1) / (m**2 + 2)).real, ((m**2 - 1) / (m**2 + 2)).imag
        return abs(complex(*k)) ** 2

    table = df._rain_table(35.0, 94.0, 10.0, 3.0)
    rayleigh = 10.0 * np.log10(k2(35.0) / k2(94.0))
    # the table starts where the ratio is smallest; drops of 0.3 mm are not Rayleigh at
    # W band, and the models of the permittivity differ a little
    assert table[0, 0] == pytest.approx(rayleigh, abs=0.2)
    assert (np.diff(table[0]) > 0).all() and (np.diff(table[1]) > 0).all()


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("dm", [0.9, 1.5, 2.0, 2.8])
@pytest.mark.parametrize("mu", [0.0, 3.0])
def test_rain_at_ka_and_w_band_recovers_a_known_distribution(dm, mu, engine):
    from radarx.fundamentals.tmatrix import raindrop_scattering

    d = np.arange(0.01, 8.0001, 0.01)
    nw = 6000.0
    nd = normalized_gamma(d, dm, mu, nw)
    z = {}
    for f in (35.0, 94.0):
        drops = raindrop_scattering(d, 299.792458 / f, 10.0, ddelt=1e-5)
        z[f] = 10.0 * np.log10(
            (299.792458 / f) ** 4
            / (np.pi**5 * 0.93)
            * (nd * drops.sigma_h.values).sum()
            * 0.01
        )
    out = dfr_rain(
        xr.DataArray(z[35.0] - z[94.0]),
        xr.DataArray(z[35.0]),
        mu=mu,
        engine=engine,
    )
    lwc = np.pi / 6.0 * 1e-3 * (nd * d**3).sum() * 0.01
    assert float(out.DM) == pytest.approx(dm, abs=0.03)
    assert float(out.NW) == pytest.approx(nw, rel=0.1)
    assert float(out.LWC) == pytest.approx(lwc, rel=0.1)
    assert out.DM.attrs["units"] == "mm" and out.NW.attrs["units"] == "m-3 mm-1"
    assert out.LWC.attrs["units"] == "g m-3"
    assert out.attrs["mu"] == mu


@pytest.mark.parametrize("pair, bands", [((2.8, 5.6), ("S", "C"))])
def test_rain_look_up_against_the_scattering_tables_of_radar_from_dsd(pair, bands):
    from radarx.retrieve import radar_from_dsd

    d = np.arange(0.01, 8.0, 0.01)
    for dm in (0.8, 1.2, 1.6, 2.0):
        nd = xr.DataArray(
            normalized_gamma(d, dm, 3.0, 8000.0),
            dims="diameter",
            coords={"diameter": d},
        )
        low = radar_from_dsd(nd, bands[0], 20.0)
        high = radar_from_dsd(nd, bands[1], 20.0)
        out = dfr_rain(
            xr.DataArray(float(low.DBZH - high.DBZH)),
            xr.DataArray(float(low.DBZH)),
            frequencies=pair,
            temperature=20.0,
        )
        assert float(out.DM) == pytest.approx(dm, abs=0.02)
        assert float(out.NW) == pytest.approx(8000.0, rel=0.03)
        assert float(out.LWC) == pytest.approx(float(low.LWC), rel=0.03)


@pytest.mark.parametrize("engine", ENGINES)
def test_rain_out_of_table_ratios(engine):
    table = df._rain_table(35.0, 94.0, 10.0, 3.0)
    lo, hi = table[0, 0], table[0, -1]
    ratio = xr.DataArray(
        [lo - 2.0, lo - 0.5, lo, 0.5 * (lo + hi), hi, hi + 1.0, np.nan]
    )
    out = dfr_rain(ratio, xr.full_like(ratio, 20.0), engine=engine)
    dm = out.DM.values
    assert np.isnan(dm[[0, 5, 6]]).all()
    assert dm[1] == pytest.approx(table[1, 0])  # clipped to the smallest size...
    assert np.isnan(out.NW.values[1]) and np.isnan(out.LWC.values[1])  # ...without Nw
    assert np.isfinite(out.NW.values[2:5]).all()
    assert (np.diff(dm[2:5]) > 0).all()
    wide = dfr_rain(ratio, xr.full_like(ratio, 20.0), tolerance=3.0, engine=engine)
    assert np.isfinite(wide.DM.values[0])
    strict = dfr_rain(ratio, xr.full_like(ratio, 20.0), tolerance=0.0, engine=engine)
    assert np.isnan(strict.DM.values[1])


@compiled_only
def test_rain_engines_agree_on_a_field_and_with_dask():
    rnd = np.random.default_rng(4)
    ratio = xr.DataArray(rnd.uniform(-1.0, 24.0, (8, 50)), dims=("time", "range"))
    dbz = xr.DataArray(rnd.uniform(-10.0, 30.0, (8, 50)), dims=("time", "range"))
    a = dfr_rain(ratio, dbz, engine="numpy")
    b = dfr_rain(ratio, dbz, engine="compiled", n_threads=2)
    for name in a:
        np.testing.assert_allclose(b[name], a[name], rtol=1e-5, equal_nan=True)
    pytest.importorskip("dask")
    lazy = dfr_rain(ratio.chunk({"time": 4}), dbz.chunk({"time": 4}))
    assert lazy.DM.chunks is not None
    for name in a:
        np.testing.assert_allclose(
            lazy[name].compute(), a[name], rtol=1e-5, equal_nan=True
        )


def test_rain_accessor_and_provenance():
    ratio = xr.DataArray([5.0, 10.0], dims="x")
    dbz = xr.DataArray([20.0, 25.0], dims="x")
    out = ratio.radarx.dfr_rain(dbz, mu=2.0)
    xr.testing.assert_allclose(out, dfr_rain(ratio, dbz, mu=2.0))
    assert "Tridon" in out.attrs["radarx_method"]
    assert "10.1002/2014JD023023" in out.attrs["radarx_references"]
    before = ratio.copy()
    dfr_rain(ratio, dbz)
    xr.testing.assert_identical(ratio, before)


def test_rain_errors():
    ratio = xr.DataArray([5.0], dims="x")
    with pytest.raises(TypeError, match="dfr"):
        dfr_rain(np.array([5.0]), ratio)
    with pytest.raises(TypeError, match="dbz"):
        dfr_rain(ratio, np.array([5.0]))
    with pytest.raises(ValueError, match="tolerance"):
        dfr_rain(ratio, ratio, tolerance=-1.0)
    with pytest.raises(ValueError, match="frequencies"):
        dfr_rain(ratio, ratio, frequencies=(94.0, 35.0))
    with pytest.raises(ValueError, match="frequencies"):
        dfr_rain(ratio, ratio, frequencies=35.0)
    with pytest.raises(ValueError, match="engine"):
        dfr_rain(ratio, ratio, engine="gpu")


def test_the_kernel_is_not_required_for_the_numpy_engine(monkeypatch):
    monkeypatch.setattr(df, "HAS_COMPILED_KERNEL", False)
    ratio = xr.DataArray([5.0, 10.0], dims="x")
    assert np.isfinite(dfr_rain(ratio, ratio + 15.0).DM).all()
    with pytest.raises(ImportError):
        dfr_rain(ratio, ratio, engine="compiled")


# ---- snow -----------------------------------------------------------------------------


def mie_backscatter(x, m):
    """Backscattering cross section over the geometric one of a sphere (Bohren and Huffman)."""
    nmax = int(np.ceil(x + 4.05 * x ** (1 / 3) + 2)) + 5
    n = np.arange(1, nmax + 1)
    z = m * x

    def psi(k, arg):
        return arg * spherical_jn(k, arg)

    def dpsi(k, arg):
        return spherical_jn(k, arg) + arg * spherical_jn(k, arg, derivative=True)

    def xi(k, arg):
        return arg * (spherical_jn(k, arg) + 1j * spherical_yn(k, arg))

    def dxi(k, arg):
        return (spherical_jn(k, arg) + 1j * spherical_yn(k, arg)) + arg * (
            spherical_jn(k, arg, derivative=True)
            + 1j * spherical_yn(k, arg, derivative=True)
        )

    a = (m * psi(n, z) * dpsi(n, x) - psi(n, x) * dpsi(n, z)) / (
        m * psi(n, z) * dxi(n, x) - xi(n, x) * dpsi(n, z)
    )
    b = (psi(n, z) * dpsi(n, x) - m * psi(n, x) * dpsi(n, z)) / (
        psi(n, z) * dxi(n, x) - m * xi(n, x) * dpsi(n, z)
    )
    s = ((2 * n + 1) * (-1.0) ** n * (a - b)).sum()
    return abs(s) ** 2 / x**2  # sigma_b / (pi a^2)


def mie_sigma(d_mm, frequency, coefficients):
    wavelength = 299.792458 / frequency
    eps_ice = df._ice_index(frequency) ** 2
    mass = df._ice_sphere_mass(d_mm, coefficients)
    fraction = mass / (df._ICE_DENSITY * np.pi / 6.0 * (d_mm * 1e-3) ** 3)
    index = np.sqrt(df._maxwell_garnett(eps_ice, fraction))
    index = np.where(index.imag < 0, index.conjugate(), index)
    return np.array(
        [
            np.pi * (di / 2.0) ** 2 * mie_backscatter(np.pi * di / wavelength, mi)
            for di, mi in zip(d_mm, index)
        ]
    )


@pytest.mark.parametrize("frequency", [35.0, 94.0])
def test_soft_sphere_backscatter_agrees_with_a_mie_code(frequency):
    d = np.array([0.2, 0.5, 1.0, 2.0, 3.5, 5.0, 8.0, 12.0])
    got = df._snow_backscatter(d, frequency, df._BROWN_FRANCIS)
    np.testing.assert_allclose(
        got, mie_sigma(d, frequency, df._BROWN_FRANCIS), rtol=2e-4
    )


def test_the_refractive_index_of_ice_and_the_mixing_rule():
    n35 = df._ice_index(35.0)
    n94 = df._ice_index(94.0)
    assert n35.real == pytest.approx(1.786, abs=0.001)
    assert 1e-4 < n35.imag < n94.imag < 1e-2  # absorption grows with frequency
    assert df._ice_index(10.0).imag < n35.imag
    with pytest.raises(ValueError, match="4.9 to 230"):
        df._ice_index(300.0)
    with pytest.raises(ValueError, match="4.9 to 230"):
        df._ice_index(1.0)
    eps = n35**2
    assert df._maxwell_garnett(eps, 0.0) == pytest.approx(1.0)
    assert df._maxwell_garnett(eps, 1.0) == pytest.approx(eps)
    # the low-density limit is linear: 1 + 3 f (eps - 1) / (eps + 2)
    f = 1e-3
    assert df._maxwell_garnett(eps, f) == pytest.approx(
        1.0 + 3.0 * f * (eps - 1.0) / (eps + 2.0), rel=1e-3
    )


def test_soft_sphere_mass_is_capped_at_solid_ice():
    d = np.array([0.01, 0.1, 1.0, 10.0])
    mass = df._ice_sphere_mass(d, df._BROWN_FRANCIS)
    sphere = 917.0 * np.pi / 6.0 * (d * 1e-3) ** 3
    assert mass[0] == pytest.approx(sphere[0])
    assert (mass <= sphere * (1 + 1e-12)).all()
    assert mass[3] == pytest.approx(0.0185 * (0.01) ** 1.9)


def test_snow_table_rises_from_the_rayleigh_limit_to_saturation():
    table = df._snow_table(35.0, 94.0, 0.0, df._BROWN_FRANCIS)
    assert table[0, 0] < 0.5  # ice is a Rayleigh scatterer at both frequencies
    assert 12.0 < table[0, -1] < 22.0
    assert (np.diff(table[0]) > 0).all() and (np.diff(table[1]) > 0).all()


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("slope", [1.5, 3.0, 6.0])
def test_snow_recovers_a_known_exponential_distribution(slope, engine):
    d = np.geomspace(0.05, 40.0, 600)
    weight = np.gradient(d)
    n0 = 5.0
    nd = n0 * np.exp(-slope * d)
    sigma = {f: mie_sigma(d, f, df._BROWN_FRANCIS) for f in (35.0, 94.0)}
    z = {
        f: 10.0
        * np.log10(
            (299.792458 / f) ** 4 / (np.pi**5 * 0.93) * (nd * sigma[f] * weight).sum()
        )
        for f in sigma
    }
    mass = df._ice_sphere_mass(d, df._BROWN_FRANCIS)
    dm = (nd * d * mass * weight).sum() / (nd * mass * weight).sum()
    iwc = (nd * mass * 1e3 * weight).sum()
    out = dfr_snow(
        xr.DataArray(z[35.0] - z[94.0]), xr.DataArray(z[35.0]), engine=engine
    )
    assert float(out.DM) == pytest.approx(dm, rel=0.04)
    assert float(out.IWC) == pytest.approx(iwc, rel=0.12)
    assert out.IWC.attrs["units"] == "g m-3"
    assert out.IWC.attrs["standard_name"] == "mass_concentration_of_cloud_ice_in_air"
    assert out.attrs["mass_size"] == (0.0185, 1.9)


def test_snow_options_change_the_table():
    ratio = xr.DataArray(6.0)
    dbz = xr.DataArray(15.0)
    base = dfr_snow(ratio, dbz)
    gamma = dfr_snow(ratio, dbz, mu=2.0)
    heavier = dfr_snow(ratio, dbz, mass_size=(0.04, 2.0))
    other = dfr_snow(ratio, dbz, frequencies=(35.0, 94.0))
    assert float(gamma.DM) != float(base.DM)
    assert float(heavier.IWC) != float(base.IWC)
    assert float(other.DM) == float(base.DM)


def test_snow_out_of_table_and_errors():
    table = df._snow_table(35.0, 94.0, 0.0, df._BROWN_FRANCIS)
    ratio = xr.DataArray([table[0, 0] - 0.5, table[0, -1] + 1.0, np.nan, 8.0], dims="x")
    out = dfr_snow(ratio, xr.full_like(ratio, 10.0))
    assert out.DM.values[0] == pytest.approx(table[1, 0])
    assert np.isnan(out.IWC.values[0])
    assert np.isnan(out.DM.values[1:3]).all()
    assert np.isfinite(out.IWC.values[3])
    with pytest.raises(TypeError, match="dfr"):
        dfr_snow(np.array([5.0]), ratio)
    with pytest.raises(ValueError, match="frequencies"):
        dfr_snow(ratio, ratio, frequencies=(35.0, 300.0))
    with pytest.raises(ValueError, match="mass_size must"):
        dfr_snow(ratio, ratio, mass_size=0.02)
    with pytest.raises(ValueError, match="mass_size needs"):
        dfr_snow(ratio, ratio, mass_size=(-1.0, 1.9))
    with pytest.raises(ValueError, match="mass_size needs"):
        dfr_snow(ratio, ratio, mass_size=(0.02, 3.5))
    with pytest.raises(ValueError, match="tolerance"):
        dfr_snow(ratio, ratio, tolerance=-0.1)


def test_snow_accessor_dask_and_provenance():
    ratio = xr.DataArray(
        np.linspace(0.5, 12.0, 40).reshape(4, 10), dims=("time", "range")
    )
    dbz = xr.full_like(ratio, 12.0)
    expected = dfr_snow(ratio, dbz)
    via = ratio.radarx.dfr_snow(dbz)
    xr.testing.assert_allclose(via, expected)
    assert "10.1029/2007JD009744" in expected.attrs["radarx_references"]
    assert expected.DM.dtype == np.float32
    pytest.importorskip("dask")
    lazy = dfr_snow(ratio.chunk({"time": 2}), dbz.chunk({"time": 2}))
    assert lazy.IWC.chunks is not None
    xr.testing.assert_allclose(lazy.compute(), expected)
