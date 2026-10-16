#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the rain-rate relations and the hydrometeor-class blend
=================================================================

The relations are checked against values worked out by hand from the
coefficients of the papers, the compiled kernel against the NumPy
implementation (rtol 1e-12), and the whole chain against the rain rate of
known drop size distributions: the PERiLS 2022 Parsivel gamma parameters that
ship with radarx are turned into Z, ZDR, KDP and A with the T-matrix tables of
``radarx.retrieve.dsd`` and into a rain rate with the fall speed of Atlas et
al., and the estimates must return that rain rate within the scatter of the
relations.
"""
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
import radarx.retrieve as retrieve
from radarx.retrieve import (
    hid,
    hid_classes,
    radar_from_dsd,
    rain_rate,
    rain_rate_a,
    rain_rate_kdp,
    rain_rate_z,
    rain_rate_z_zdr,
)

from .provenance_helpers import without_provenance

qpe = __import__("radarx.retrieve.qpe", fromlist=["x"])


def same(a, b):
    """Two results are identical apart from the provenance of the call."""
    xr.testing.assert_identical(without_provenance(a), without_provenance(b))


ENGINES = ["numpy"] + (["compiled"] if qpe.HAS_COMPILED_KERNEL else [])
compiled_only = pytest.mark.skipif(
    not qpe.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)

PARK = {name: code for code, _, name in hid_classes("park", "S")}


def sweep(n_az=24, n_rng=40, seed=0):
    """A sweep with random fields on (azimuth, range) and coordinates."""
    rng = np.random.default_rng(seed)
    dims = ("azimuth", "range")

    def field(lo, hi, **attrs):
        return (dims, rng.uniform(lo, hi, (n_az, n_rng)), attrs)

    return xr.Dataset(
        {
            "DBZH": field(5, 62, units="dBZ"),
            "ZDR": field(-0.8, 3.5, units="dB"),
            "KDP": field(-0.4, 4.0, units="degree km-1"),
            "AH": field(0.0, 0.06, units="dB km-1"),
        },
        coords={
            "azimuth": ("azimuth", np.linspace(0.5, 359.5, n_az)),
            "range": ("range", 500.0 + 250.0 * np.arange(n_rng)),
            "elevation": ("azimuth", np.full(n_az, 0.5)),
        },
    )


def classes(ds, scheme=("park", "S"), seed=1):
    """A ``HID`` array with the flag attributes of ``retrieve.hid``."""
    table = hid_classes(*scheme)
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, len(table) + 1, ds.DBZH.shape).astype(np.int8)
    return xr.DataArray(
        codes,
        dims=ds.DBZH.dims,
        coords=ds.DBZH.coords,
        name="HID",
        attrs={
            "flag_values": np.arange(1, len(table) + 1, dtype=np.int8),
            "flag_meanings": " ".join(name for _, _, name in table),
        },
    )


def z_of(dbz):
    return 10.0 ** (np.asarray(dbz, dtype=float) / 10.0)


# --------------------------------------------------------------------------
# closed forms
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_relations_closed_form(engine):
    ds = xr.Dataset(
        {
            "DBZH": ("g", [20.0, 35.0, 45.0]),
            "ZDR": ("g", [0.5, 1.0, 2.0]),
            "KDP": ("g", [0.2, 1.0, 3.0]),
            "AH": ("g", [0.001, 0.003, 0.01]),
        }
    )
    z = z_of(ds.DBZH.values)
    zdr = z_of(ds.ZDR.values)
    kdp = ds.KDP.values
    a = ds.AH.values
    np.testing.assert_allclose(
        rain_rate_z(ds, engine=engine), 1.70e-2 * z**0.714, rtol=2e-7
    )
    np.testing.assert_allclose(
        rain_rate_kdp(ds, engine=engine), 44.0 * kdp**0.822, rtol=2e-7
    )
    np.testing.assert_allclose(
        rain_rate_z_zdr(ds, engine=engine),
        1.42e-2 * z**0.770 * zdr**-1.67,
        rtol=2e-7,
    )
    np.testing.assert_allclose(
        rain_rate_a(ds, engine=engine), 4.12e3 * a**1.03, rtol=2e-7
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_x_band_defaults(engine):
    ds = xr.Dataset({"DBZH": ("g", [30.0]), "KDP": ("g", [2.0]), "AH": ("g", [0.1])})
    z = z_of(30.0)
    np.testing.assert_allclose(
        rain_rate_z(ds, band="X", engine=engine), 0.029 * z**0.67, rtol=2e-7
    )
    np.testing.assert_allclose(
        rain_rate_kdp(ds, band="X", engine=engine), 16.9 * 2.0**0.801, rtol=2e-7
    )
    np.testing.assert_allclose(
        rain_rate_a(ds, band="X", engine=engine), 43.5 * 0.1**0.79, rtol=2e-7
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_reflectivity_cap(engine):
    ds = xr.Dataset({"DBZH": ("g", [53.0, 60.0, 70.0]), "ZDR": ("g", [1.0] * 3)})
    out = rain_rate_z(ds, engine=engine).values
    assert out[0] == pytest.approx(out[1]) == pytest.approx(out[2])
    free = rain_rate_z(ds, z_cap=None, engine=engine).values
    assert free[2] > free[1] > free[0]
    low = rain_rate_z_zdr(ds, z_cap=40.0, engine=engine).values
    assert low[0] == pytest.approx(low[2])


@pytest.mark.parametrize("engine", ENGINES)
def test_negative_kdp_and_attenuation(engine):
    ds = xr.Dataset({"KDP": ("g", [-1.0, 0.0, 1.0]), "AH": ("g", [-0.01, 0.0, 0.01])})
    kept = rain_rate_kdp(ds, engine=engine).values
    assert kept[0] == pytest.approx(-kept[2]) and kept[1] == 0.0 and kept[2] > 0
    zero = rain_rate_kdp(ds, negative="zero", engine=engine).values
    np.testing.assert_array_equal(zero, [0.0, 0.0, kept[2]])
    ka = rain_rate_a(ds, engine=engine).values
    assert ka[0] == pytest.approx(-ka[2])
    assert rain_rate_a(ds, negative="zero", engine=engine).values[0] == 0.0


@pytest.mark.parametrize("engine", ENGINES)
def test_zdr_min_and_missing_values(engine):
    ds = xr.Dataset(
        {
            "DBZH": ("g", [40.0, 40.0, np.nan, 40.0]),
            "ZDR": ("g", [-1.0, 0.0, 1.0, np.nan]),
        }
    )
    plain = rain_rate_z_zdr(ds, engine=engine).values
    limited = rain_rate_z_zdr(ds, min_zdr=0.0, engine=engine).values
    assert plain[0] > plain[1]
    assert limited[0] == pytest.approx(limited[1])
    assert np.isnan(plain[2]) and np.isnan(plain[3])
    z = rain_rate_z(ds, engine=engine).values
    assert np.isnan(z[2]) and np.isfinite(z[3])


@pytest.mark.parametrize("engine", ENGINES)
def test_a_temperature_interpolation_and_clamping(engine):
    ds = xr.Dataset({"AH": ("g", [0.01] * 5)})
    temps = {0.0: (2.23e3, 1.03), 10.0: (3.10e3, 1.03), 20.0: (4.12e3, 1.03)}
    for t, (p0, p1) in temps.items():
        np.testing.assert_allclose(
            rain_rate_a(ds, temperature=t, engine=engine), p0 * 0.01**p1, rtol=2e-7
        )
    mid = rain_rate_a(ds, temperature=15.0, engine=engine).values[0]
    assert mid == pytest.approx(0.5 * (3.10e3 + 4.12e3) * 0.01**1.03, rel=2e-7)
    # constant outside 0-30 degrees Celsius
    hot = rain_rate_a(ds, temperature=45.0, engine=engine).values[0]
    assert hot == pytest.approx(5.33e3 * 0.01**1.03, rel=2e-7)
    cold = rain_rate_a(ds, temperature=-5.0, engine=engine).values[0]
    assert cold == pytest.approx(2.23e3 * 0.01**1.03, rel=2e-7)
    # vertical polarization and C band
    v = rain_rate_a(ds, temperature=20.0, polarization="v", engine=engine).values[0]
    assert v == pytest.approx(5.51e3 * 0.01**1.06, rel=2e-7)
    c = rain_rate_a(ds, band="C", temperature=20.0, engine=engine).values[0]
    assert c == pytest.approx(294.0 * 0.01**0.89, rel=2e-7)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_temperature_field_and_kelvin(engine):
    ds = xr.Dataset(
        {
            "AH": ("g", [0.01, 0.01, 0.01]),
            "T": ("g", [0.0, 10.0, np.nan], {"units": "degC"}),
            "TK": ("g", [273.15, 283.15, 293.15], {"units": "K"}),
        }
    )
    out = rain_rate_a(ds, temperature="T", engine=engine).values
    assert out[0] == pytest.approx(2.23e3 * 0.01**1.03, rel=2e-7)
    assert out[1] == pytest.approx(3.10e3 * 0.01**1.03, rel=2e-7)
    assert np.isnan(out[2])
    k = rain_rate_a(ds, temperature="TK", engine=engine).values
    assert k[2] == pytest.approx(4.12e3 * 0.01**1.03, rel=2e-7)
    # a fixed relation ignores the temperature
    fixed = rain_rate_a(ds, coefficients=(100.0, 1.0), temperature="T", engine=engine)
    np.testing.assert_allclose(fixed, 1.0, rtol=2e-7)


@pytest.mark.parametrize("engine", ENGINES)
def test_user_coefficients(engine):
    ds = sweep(4, 5)
    out = rain_rate_z(ds, coefficients=(0.0365, 0.625), engine=engine)
    np.testing.assert_allclose(
        out, 0.0365 * np.minimum(z_of(ds.DBZH), z_of(53.0)) ** 0.625, rtol=2e-7
    )
    out = rain_rate_z_zdr(ds, coefficients=(0.01, 0.8, -1.0), engine=engine)
    expected = (
        0.01 * np.minimum(z_of(ds.DBZH), z_of(53.0)) ** 0.8 * z_of(ds.ZDR) ** -1.0
    )
    np.testing.assert_allclose(out, expected, rtol=2e-7)


def test_no_default_is_not_invented():
    ds = sweep(3, 3)
    # there is no C-band Z-R relation that could be verified
    with pytest.raises(ValueError, match="no verified default"):
        rain_rate_z(ds, band="C")
    # R(A) has a table for every band
    for band in qpe.BANDS:
        assert rain_rate_a(ds, band=band).name == "RAIN_RATE_A"
    assert set(qpe.COEFFICIENTS["C"]) == {"kdp", "z_zdr"}
    assert set(qpe.COEFFICIENTS["X"]) == {"z", "kdp", "z_zdr"}


@pytest.mark.parametrize("engine", ENGINES)
def test_textbook_defaults_at_c_and_x_band(engine):
    # Table 8.1 and Eq. 8.14 of Bringi and Chandrasekar (2001): 129 (KDP / f)^0.85 at 5.45 GHz
    assert qpe.COEFFICIENTS["C"]["kdp"][0] == pytest.approx(129 / 5.45**0.85, rel=2e-3)
    ds = sweep(3, 3)
    kdp = ds.KDP.values
    out = rain_rate_kdp(ds, band="C", negative="keep", engine=engine)
    np.testing.assert_allclose(
        out, 30.5 * np.abs(kdp) ** 0.85 * np.sign(kdp), rtol=2e-7
    )
    for band, (a, b, c) in (("C", (5.8e-3, 0.91, -2.09)), ("X", (3.9e-3, 1.07, -5.97))):
        out = rain_rate_z_zdr(ds, band=band, engine=engine)
        expected = a * np.minimum(z_of(ds.DBZH), z_of(53.0)) ** b * z_of(ds.ZDR) ** c
        np.testing.assert_allclose(out, expected, rtol=2e-7)


def simulated_rain(wavelength, n=3000, seed=2001):
    """Z, ZDR, KDP and R of gamma distributions as in Bringi and Chandrasekar (2001), 8.1.1."""
    from math import lgamma

    from radarx.fundamentals import raindrop_scattering

    d = np.arange(0.05, 8.001, 0.05)
    rng = np.random.default_rng(seed)
    d0, mu, lognw = (
        rng.uniform(0.5, 2.5, n),
        rng.uniform(-1.0, 5.0, n),
        rng.uniform(3.0, 5.0, n),
    )
    f = np.array(
        [6 * (3.67 + m) ** (m + 4) / (3.67**4 * np.exp(lgamma(m + 4))) for m in mu]
    )
    x = d[None, :] / d0[:, None]
    dsd = (
        (10**lognw)[:, None]
        * f[:, None]
        * x ** mu[:, None]
        * np.exp(-(3.67 + mu[:, None]) * x)
    )
    beard_chuang = lambda dd: (  # noqa: E731
        1.0048 + 5.7e-4 * dd - 2.628e-2 * dd**2 + 3.682e-3 * dd**3 - 1.677e-4 * dd**4
    )
    t = raindrop_scattering(
        d, wavelength, temperature=20.0, canting_std=7.0, axis_ratio=beard_chuang
    )
    scale = wavelength**4 / (np.pi**5 * 0.93) * 0.05
    zh = scale * (dsd * t.sigma_h.values).sum(axis=1)
    zv = scale * (dsd * t.sigma_v.values).sum(axis=1)
    kdp = (dsd * t.kdp.values).sum(axis=1) * 0.05
    rate = (
        0.6
        * np.pi
        * 1e-3
        * (dsd * d**3 * (9.65 - 10.3 * np.exp(-0.6 * d))).sum(axis=1)
        * 0.05
    )
    return zh, zh / zv, kdp, rate


@pytest.mark.parametrize(
    ("band", "frequency", "relation"),
    [("C", 5.45, "z_zdr"), ("C", 5.45, "kdp"), ("X", 10.0, "z_zdr")],
)
def test_textbook_defaults_reproduce_the_tmatrix_simulation(band, frequency, relation):
    # the defaults are fitted to this simulation: they give its rain rate with a bias of
    # at most 12 % and a normalized error of at most 30 % (the C++ T-matrix of radarx)
    pytest.importorskip("radarx.fundamentals._tmatrix")
    zh, zeta, kdp, rate = simulated_rain(299.792458 / frequency)
    keep = (rate <= 300.0) & (kdp > 0.05)
    coef = qpe.COEFFICIENTS[band][relation]
    if relation == "z_zdr":
        a, b, c = coef
        estimate = a * zh**b * zeta**c
    else:
        estimate = coef[0] * kdp ** coef[1]
    bias = (estimate[keep] - rate[keep]).mean() / rate[keep].mean()
    nse = np.sqrt(((estimate[keep] - rate[keep]) ** 2).mean()) / rate[keep].mean()
    assert abs(bias) < 0.12 and nse < 0.30, (bias, nse)


def test_input_errors():
    ds = sweep(3, 3)
    with pytest.raises(ValueError, match="band"):
        rain_rate_z(ds, band="K")
    with pytest.raises(ValueError, match="engine"):
        rain_rate_z(ds, engine="gpu")
    with pytest.raises(ValueError, match="negative"):
        rain_rate_kdp(ds, negative="abs")
    with pytest.raises(ValueError, match="polarization"):
        rain_rate_a(ds, polarization="x")
    with pytest.raises(ValueError, match="2 numbers"):
        rain_rate_z(ds, coefficients=(1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="sequence"):
        rain_rate_z(ds, coefficients=1.0)
    with pytest.raises(KeyError, match="reflectivity"):
        rain_rate_z(ds.drop_vars("DBZH"))
    with pytest.raises(KeyError, match="nope"):
        rain_rate_z(ds, dbz="nope")
    with pytest.raises(TypeError):
        rain_rate_z(np.zeros(3))
    with pytest.raises(ValueError, match="DataArray"):
        rain_rate_z_zdr(ds.DBZH, zdr="ZDR")
    with pytest.raises(ValueError, match="temperature"):
        rain_rate_a(ds.AH, temperature="T")
    with pytest.raises(TypeError, match="temperature"):
        rain_rate_a(ds, temperature=[1.0, 2.0])
    if not qpe.HAS_COMPILED_KERNEL:
        with pytest.raises(ImportError):
            rain_rate_z(ds, engine="compiled")


# --------------------------------------------------------------------------
# agreement of the compiled kernel and NumPy
# --------------------------------------------------------------------------


def holes(ds, seed=3):
    """Dataset with 10 % NaN in every field."""
    rng = np.random.default_rng(seed)
    out = ds.copy()
    for name in ("DBZH", "ZDR", "KDP", "AH"):
        out[name] = out[name].where(rng.random(out[name].shape) > 0.1)
    return out


@compiled_only
@pytest.mark.parametrize(
    "func, kwargs",
    [
        (rain_rate_z, {}),
        (rain_rate_z, {"z_cap": None}),
        (rain_rate_kdp, {}),
        (rain_rate_kdp, {"negative": "zero", "band": "X"}),
        (rain_rate_z_zdr, {"min_zdr": -0.2}),
        (rain_rate_a, {"temperature": 12.5}),
        (rain_rate_a, {"temperature": 25.0, "band": "C", "polarization": "v"}),
        (rain_rate_a, {"negative": "zero", "band": "X"}),
    ],
)
def test_compiled_equals_numpy(func, kwargs):
    ds = holes(sweep(40, 90))
    a = func(ds, engine="compiled", **kwargs)
    b = func(ds, engine="numpy", **kwargs)
    xr.testing.assert_allclose(a, b, rtol=1e-6, atol=0)  # float32 results
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))


@compiled_only
def test_compiled_equals_numpy_in_double_precision():
    """The kernels themselves agree to rounding (rtol 1e-12), before the float32 cast."""
    rng = np.random.default_rng(5)
    n = 5000
    dbz = rng.uniform(-5, 65, n)
    zdr = rng.uniform(-1, 4, n)
    kdp = rng.uniform(-1, 6, n)
    a = rng.uniform(-0.01, 0.1, n)
    t = rng.uniform(-10, 40, n)
    for v in (dbz, zdr, kdp, a, t):
        v[rng.random(n) < 0.05] = np.nan
    for band in qpe.BANDS:
        rels = ["a"] + [r for r in ("z", "kdp", "z_zdr") if r in qpe.COEFFICIENTS[band]]
        coef, nodes = qpe._coefficients(rels, band, None, "h")
        for negative in ("keep", "zero"):
            setup = qpe._Setup(coef, nodes, 53.0, 0.0, negative)
            for rel in (qpe._CODE[r] for r in rels):
                got = qpe._qpe.relation(
                    rel, dbz, zdr, kdp, a, t, n, setup.coef, setup.a_t,
                    setup.a_p0, setup.a_p1, setup.z_cap, setup.zdr_min,
                    setup.negative, 3,
                )  # fmt: skip
                want = qpe._rate_numpy(rel, dbz, zdr, kdp, a, t, setup)
                np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)
                np.testing.assert_array_equal(np.isnan(got), np.isnan(want))


@compiled_only
def test_blend_kernel_equals_numpy():
    rng = np.random.default_rng(6)
    n = 4000
    cls = rng.integers(0, 12, n).astype(np.int32)  # codes beyond the table too
    dbz = rng.uniform(0, 65, n)
    zdr = rng.uniform(-1, 4, n)
    kdp = rng.uniform(-1, 6, n)
    a = rng.uniform(0, 0.1, n)
    t = rng.uniform(0, 30, n)
    for v in (dbz, zdr, kdp, a, t):
        v[rng.random(n) < 0.1] = np.nan
    lut_rel = np.array([-1, 0, 1, 2, 3, -1, 2, 1], dtype=np.int32)
    lut_fac = np.array([1.0, 1.0, 0.8, 1.0, 0.6, 1.0, 2.8, 1.0])
    coef, nodes = qpe._coefficients(list(qpe.RELATIONS), "S", None, "h")
    setup = qpe._Setup(coef, nodes, 53.0, None, "keep")
    got = qpe._qpe.blend(
        cls, dbz, zdr, kdp, a, t, lut_rel, lut_fac, setup.coef, setup.a_t,
        setup.a_p0, setup.a_p1, setup.z_cap, setup.zdr_min, setup.negative, 4,
    )  # fmt: skip
    want = qpe._blend_numpy(cls, dbz, zdr, kdp, a, t, lut_rel, lut_fac, setup)
    np.testing.assert_allclose(got[0], want[0], rtol=1e-12, atol=0)
    np.testing.assert_array_equal(np.isnan(got[0]), np.isnan(want[0]))
    np.testing.assert_array_equal(got[1], want[1])
    np.testing.assert_array_equal(got[2], want[2])


@compiled_only
def test_kernel_rejects_bad_sizes():
    one = np.full(1, np.nan)
    coef = [1.0] * 7
    nodes = np.zeros(1)
    with pytest.raises(ValueError):
        qpe._qpe.relation(
            0,
            np.zeros(3),
            one,
            one,
            one,
            one,
            5,
            coef,
            nodes,
            nodes + 1,
            nodes + 1,
            53.0,
            np.nan,
            0,
        )
    with pytest.raises(ValueError):
        qpe._qpe.relation(
            0,
            one,
            one,
            one,
            one,
            one,
            1,
            coef[:3],
            nodes,
            nodes + 1,
            nodes + 1,
            53.0,
            np.nan,
            0,
        )
    with pytest.raises(ValueError):
        qpe._qpe.relation(
            9,
            one,
            one,
            one,
            one,
            one,
            1,
            coef,
            nodes,
            nodes + 1,
            nodes + 1,
            53.0,
            np.nan,
            0,
        )
    with pytest.raises(ValueError):
        qpe._qpe.relation(
            0,
            one,
            one,
            one,
            one,
            one,
            1,
            coef,
            nodes,
            nodes[:0],
            nodes[:0],
            53.0,
            np.nan,
            0,
        )


@compiled_only
def test_thread_counts_agree():
    ds = sweep(30, 70)
    one = rain_rate_z_zdr(ds, n_threads=1)
    many = rain_rate_z_zdr(ds, n_threads=7)
    xr.testing.assert_identical(one, many)


# --------------------------------------------------------------------------
# front end: containers, coordinates, laziness, attributes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_attributes_coordinates_and_input_unchanged(engine):
    ds = sweep()
    before = ds.copy(deep=True)
    for func, name in (
        (rain_rate_z, "RAIN_RATE_Z"),
        (rain_rate_kdp, "RAIN_RATE_KDP"),
        (rain_rate_z_zdr, "RAIN_RATE_Z_ZDR"),
        (rain_rate_a, "RAIN_RATE_A"),
    ):
        out = func(ds, engine=engine)
        assert out.name == name
        assert out.dtype == np.float32
        assert out.attrs["units"] == "mm h-1"
        assert out.attrs["standard_name"] == "rainfall_rate"
        assert out.attrs["long_name"] and out.attrs["comment"].startswith("R = ")
        assert out.dims == ("azimuth", "range")
        for coord in ("azimuth", "range", "elevation"):
            xr.testing.assert_equal(out[coord], ds[coord])
        assert out.attrs["radarx_references"]
    xr.testing.assert_identical(ds, before)
    assert "radarx_method" not in ds.attrs


def test_dataarray_input_and_second_field():
    ds = sweep(6, 8)
    out = rain_rate_z(ds.DBZH)
    xr.testing.assert_allclose(out, rain_rate_z(ds), rtol=0)
    zz = rain_rate_z_zdr(ds.DBZH, zdr=ds.ZDR)
    xr.testing.assert_allclose(zz, rain_rate_z_zdr(ds), rtol=0)
    rr = rain_rate_a(ds.AH, temperature=ds.DBZH * 0 + 12.0)
    xr.testing.assert_allclose(rr, rain_rate_a(ds, temperature=12.0), rtol=0)
    assert ds.DBZH.attrs == {"units": "dBZ"}  # the input keeps its attributes


def test_field_names_are_found_or_given():
    ds = sweep(4, 5).rename(
        DBZH="corrected_reflectivity",
        ZDR="corrected_differential_reflectivity",
        KDP="specific_differential_phase",
        AH="specific_attenuation",
    )
    for func in (rain_rate_z, rain_rate_kdp, rain_rate_z_zdr, rain_rate_a):
        assert np.isfinite(func(ds)).all()
    custom = ds.rename(corrected_reflectivity="Z_good")
    assert np.isfinite(rain_rate_z(custom, dbz="Z_good")).all()
    with pytest.raises(KeyError):
        rain_rate_z(custom)


@pytest.mark.parametrize("engine", ENGINES)
def test_dask_stays_lazy_and_matches(engine):
    pytest.importorskip("dask")
    ds = holes(sweep(30, 50))
    lazy = ds.chunk({"azimuth": 7, "range": 20})
    for func, kwargs in (
        (rain_rate_z, {}),
        (rain_rate_kdp, {}),
        (rain_rate_z_zdr, {}),
        (rain_rate_a, {"temperature": lazy.DBZH * 0 + 14.0}),
    ):
        out = func(lazy, engine=engine, **kwargs)
        assert out.chunks is not None, func.__name__
        kw = dict(kwargs)
        if "temperature" in kw:
            kw["temperature"] = ds.DBZH * 0 + 14.0
        xr.testing.assert_allclose(out.compute(), func(ds, engine=engine, **kw), rtol=0)


def volume(names=("sweep_0", "sweep_1", "sweep_2")):
    nodes = {"/": xr.Dataset()}
    for i, name in enumerate(names):
        ds = sweep(12, 20, seed=10 + i)
        if i == 2:
            ds = ds.drop_vars(["ZDR", "AH"])
        nodes[name] = ds
    return xr.DataTree.from_dict(nodes)


def test_datatree_skips_sweeps_without_the_fields():
    vol = volume()
    before = vol.copy(deep=True)
    out = rain_rate_z(vol)
    assert isinstance(out, xr.DataTree)
    assert set(out.children) == {"sweep_0", "sweep_1", "sweep_2"}
    zz = rain_rate_z_zdr(vol)
    assert set(zz.children) == {"sweep_0", "sweep_1"}
    ra = vol.radarx.rain_rate_a()
    assert set(ra.children) == {"sweep_0", "sweep_1"}
    assert float(out["sweep_0"]["RAIN_RATE_Z"].mean()) > 0
    xr.testing.assert_allclose(
        out["sweep_1"]["RAIN_RATE_Z"], rain_rate_z(vol["sweep_1"].to_dataset())
    )
    assert vol.identical(before)
    with pytest.raises(KeyError, match="no sweep"):
        rain_rate_kdp(
            xr.DataTree.from_dict(
                {"/": xr.Dataset(), "sweep_0": sweep().drop_vars("KDP")}
            )
        )


def test_accessors():
    ds = sweep(5, 6)
    xr.testing.assert_identical(ds.radarx.rain_rate_z(), rain_rate_z(ds))
    xr.testing.assert_identical(
        ds.radarx.rain_rate_kdp(negative="zero"), rain_rate_kdp(ds, negative="zero")
    )
    xr.testing.assert_identical(ds.radarx.rain_rate_z_zdr(), rain_rate_z_zdr(ds))
    xr.testing.assert_identical(
        ds.radarx.rain_rate_a(band="X"), rain_rate_a(ds, band="X")
    )
    xr.testing.assert_identical(ds.DBZH.radarx.rain_rate_z(), rain_rate_z(ds.DBZH))
    h = classes(ds)
    full = ds.assign(HID=h)
    xr.testing.assert_identical(full.radarx.rain_rate(), rain_rate(full))
    vol = volume()
    assert vol.radarx.rain_rate_kdp().children


def test_provenance_is_recorded():
    from radarx._provenance import read_provenance

    ds = sweep(3, 3)
    info = read_provenance(rain_rate_z(ds))[0]
    assert "10.1175/JTECH-D-13-00038.1" in info["references"]
    assert info["history"][0].startswith("rain_rate_z(")
    full = ds.assign(HID=classes(ds))
    info = read_provenance(rain_rate(full))[0]
    assert "10.1175/2008JAMC1753.1" in info["references"]
    assert "10.1175/JAM2213.1" in info["references"]


# --------------------------------------------------------------------------
# the class blend
# --------------------------------------------------------------------------


def one_class_dataset(name, scheme=("park", "S")):
    """One gate per class code, with fields that suit every relation."""
    table = hid_classes(*scheme)
    n = len(table)
    ds = xr.Dataset(
        {
            "DBZH": ("g", np.full(n, 40.0)),
            "ZDR": ("g", np.full(n, 1.2)),
            "KDP": ("g", np.full(n, 1.5)),
            "AH": ("g", np.full(n, 0.004)),
        }
    )
    ds["HID"] = (
        "g",
        np.arange(1, n + 1, dtype=np.int8),
        {
            "flag_values": np.arange(1, n + 1, dtype=np.int8),
            "flag_meanings": " ".join(c[2] for c in table),
        },
    )
    return ds


@pytest.mark.parametrize("engine", ENGINES)
def test_default_blend_by_class(engine):
    ds = one_class_dataset("park")
    out = rain_rate(ds, engine=engine)
    names = [c[2] for c in hid_classes("park", "S")]
    zz = float(rain_rate_z_zdr(ds, engine=engine)[0])
    kdp = float(rain_rate_kdp(ds, engine=engine)[0])
    for i, name in enumerate(names):
        r = float(out.RAIN_RATE[i])
        flag = int(out.RAIN_RATE_FLAG[i])
        used = int(out.RAIN_RATE_RELATION[i])
        if name in ("light_and_moderate_rain", "heavy_rain", "big_drops"):
            assert r == pytest.approx(zz) and flag == 0 and used == 3, name
        elif name == "rain_hail_mixture":
            assert r == pytest.approx(kdp) and flag == 0 and used == 2, name
        else:
            assert np.isnan(r) and flag == 1 and used == 0, name
    assert out.RAIN_RATE.attrs["units"] == "mm h-1"
    assert list(out.RAIN_RATE_FLAG.attrs["flag_values"]) == [0, 1, 2, 3]
    assert "no_relation_for_class" in out.RAIN_RATE_FLAG.attrs["flag_meanings"]
    assert "R(Z,ZDR)" in out.RAIN_RATE_RELATION.attrs["comment"]


@pytest.mark.parametrize("engine", ENGINES)
def test_dolan_c_band_classes(engine):
    """Hail and ice have no relation; rain has the C-band relations of the textbook."""
    ds = one_class_dataset("dolan", ("dolan", "C"))
    default = rain_rate(
        ds, band="C", engine=engine
    )  # z_zdr and kdp have C-band defaults
    names = [c[2] for c in hid_classes("dolan", "C")]
    assert np.isfinite(default.RAIN_RATE[names.index("rain")])
    assert np.isnan(float(default.RAIN_RATE[names.index("hail")]))
    out = rain_rate(
        ds,
        band="C",
        class_relations={"drizzle": "a", "rain": "a", "big_drops": "a"},
        engine=engine,
    )
    ra = float(rain_rate_a(ds, band="C", engine=engine)[0])
    for i, name in enumerate(names):
        if name in ("drizzle", "rain", "big_drops"):
            assert float(out.RAIN_RATE[i]) == pytest.approx(ra)
        else:
            assert np.isnan(float(out.RAIN_RATE[i])), name
    assert int(out.RAIN_RATE_FLAG[names.index("hail")]) == 1
    # coefficients make the default relations available at C band
    full = rain_rate(
        ds, band="C", coefficients={"z_zdr": (0.01, 0.8, -1.5)}, engine=engine
    )
    assert np.isfinite(full.RAIN_RATE[names.index("rain")])


@pytest.mark.parametrize("engine", ENGINES)
def test_class_relations_override(engine):
    ds = one_class_dataset("park")
    names = [c[2] for c in hid_classes("park", "S")]
    out = rain_rate(
        ds,
        class_relations={
            "wet_snow": ("z", 0.6),
            "heavy_rain": "kdp",
            "light_and_moderate_rain": None,
            "dry_snow": "a",
        },
        engine=engine,
    )
    z = float(rain_rate_z(ds, engine=engine)[0])
    assert float(out.RAIN_RATE[names.index("wet_snow")]) == pytest.approx(0.6 * z)
    assert int(out.RAIN_RATE_RELATION[names.index("wet_snow")]) == 1
    assert float(out.RAIN_RATE[names.index("heavy_rain")]) == pytest.approx(
        float(rain_rate_kdp(ds, engine=engine)[0])
    )
    assert np.isnan(float(out.RAIN_RATE[names.index("light_and_moderate_rain")]))
    assert float(out.RAIN_RATE[names.index("dry_snow")]) == pytest.approx(
        float(rain_rate_a(ds, engine=engine)[0])
    )
    with pytest.raises(ValueError, match="not one of the classes"):
        rain_rate(ds, class_relations={"hail": "z"})
    with pytest.raises(ValueError, match="unknown relation"):
        rain_rate(ds, class_relations={"wet_snow": "zz"})
    with pytest.raises(ValueError, match="pair"):
        rain_rate(ds, class_relations={"wet_snow": ("z", 1.0, 2.0)})
    with pytest.raises(ValueError, match="negative"):
        rain_rate(ds, class_relations={"wet_snow": ("z", -1.0)})


@pytest.mark.parametrize("engine", ENGINES)
def test_flags_for_missing_inputs_and_unclassified(engine):
    ds = one_class_dataset("park")
    names = [c[2] for c in hid_classes("park", "S")]
    rain = names.index("heavy_rain")
    ds["ZDR"][rain] = np.nan
    ds["HID"][0] = 0  # unclassified
    out = rain_rate(ds, engine=engine)
    assert int(out.RAIN_RATE_FLAG[rain]) == 2 and np.isnan(float(out.RAIN_RATE[rain]))
    assert int(out.RAIN_RATE_FLAG[0]) == 3 and int(out.RAIN_RATE_RELATION[0]) == 0


@pytest.mark.parametrize("engine", ENGINES)
def test_blend_equals_the_single_relations(engine):
    ds = sweep(30, 60)
    ds["HID"] = classes(ds)
    out = rain_rate(ds, engine=engine)
    h = ds.HID
    zz = rain_rate_z_zdr(ds, engine=engine)
    kd = rain_rate_kdp(ds, engine=engine)
    rain = h.isin(
        [PARK["light_and_moderate_rain"], PARK["heavy_rain"], PARK["big_drops"]]
    )
    mixed = h == PARK["rain_hail_mixture"]
    expected = xr.where(rain, zz, xr.where(mixed, kd, np.nan))
    xr.testing.assert_allclose(out.RAIN_RATE, expected.astype("float32"), rtol=0)
    assert int(((out.RAIN_RATE_FLAG == 3) == (h == 0)).all())
    estimated = out.RAIN_RATE_FLAG == 0
    assert bool((estimated == (rain | mixed)).all())


@compiled_only
def test_blend_engines_agree_on_a_sweep():
    ds = holes(sweep(40, 80))
    ds["HID"] = classes(ds)
    kwargs = {
        "class_relations": {"wet_snow": ("z", 0.6), "dry_snow": "a"},
        "temperature": 8.0,
    }
    a = rain_rate(ds, engine="compiled", **kwargs)
    b = rain_rate(ds, engine="numpy", **kwargs)
    xr.testing.assert_allclose(a, b, rtol=1e-6, atol=0)
    for name in ("RAIN_RATE_FLAG", "RAIN_RATE_RELATION"):
        np.testing.assert_array_equal(a[name], b[name])


def test_blend_with_a_given_or_computed_classification():
    ds = sweep(10, 12)
    h = classes(ds)
    given = rain_rate(ds, hid=h)
    via_name = rain_rate(ds.assign(cls=h), hid="cls")
    via_dataset = rain_rate(ds, hid=h.to_dataset())
    same(given, via_name)
    same(given, via_dataset)
    with pytest.raises(KeyError):
        rain_rate(ds, hid="missing")
    with pytest.raises(KeyError, match="HID"):
        rain_rate(ds, hid=xr.Dataset({"x": ("a", [1])}))
    with pytest.raises(TypeError):
        rain_rate(ds, hid=3)
    with pytest.raises(ValueError, match="flag_meanings"):
        rain_rate(ds, hid=h.drop_attrs())
    # no HID anywhere: it is computed by retrieve.hid
    rng = np.random.default_rng(2)
    rain_like = ds.assign(
        DBZH=("azimuth range".split(), rng.uniform(25, 45, ds.DBZH.shape)),
        ZDR=("azimuth range".split(), rng.uniform(0.5, 2.0, ds.DBZH.shape)),
        KDP=("azimuth range".split(), rng.uniform(0.0, 1.0, ds.DBZH.shape)),
        RHOHV=("azimuth range".split(), np.full(ds.DBZH.shape, 0.99)),
    )
    computed = rain_rate(rain_like, hid_kwargs={"scores": False})
    expected = rain_rate(rain_like.assign(HID=hid(rain_like, scores=False).HID))
    same(computed, expected)
    assert (computed.RAIN_RATE_FLAG == 0).any()


def test_blend_dask_and_coordinates():
    pytest.importorskip("dask")
    ds = sweep(20, 30)
    ds["HID"] = classes(ds)
    lazy = ds.chunk({"azimuth": 6})
    out = rain_rate(lazy)
    assert out.RAIN_RATE.chunks is not None and out.RAIN_RATE_FLAG.chunks is not None
    xr.testing.assert_identical(out.compute(), rain_rate(ds))
    for coord in ("azimuth", "range", "elevation"):
        xr.testing.assert_equal(out[coord], ds[coord])


def test_blend_datatree():
    vol = volume()
    nodes = {"/": xr.Dataset()}
    for name in vol.children:
        ds = vol[name].to_dataset()
        nodes[name] = ds.assign(
            HID=classes(ds.drop_vars(["ZDR", "AH"], errors="ignore"))
        )
    tree = xr.DataTree.from_dict(nodes)
    out = rain_rate(tree)
    assert set(out.children) == {"sweep_0", "sweep_1"}  # sweep_2 has no ZDR
    xr.testing.assert_identical(
        out["sweep_0"].to_dataset().drop_attrs(),
        rain_rate(tree["sweep_0"].to_dataset()).drop_attrs(),
    )
    # a classification given as a tree with the same sweeps
    hids = xr.DataTree.from_dict(
        {"/": xr.Dataset(), **{n: tree[n].to_dataset()[["HID"]] for n in vol.children}}
    )
    plain = xr.DataTree.from_dict(
        {"/": xr.Dataset(), **{n: vol[n].to_dataset() for n in vol.children}}
    )
    again = rain_rate(plain, hid=hids)
    xr.testing.assert_identical(
        again["sweep_1"].to_dataset().drop_attrs(),
        out["sweep_1"].to_dataset().drop_attrs(),
    )
    with pytest.raises(KeyError, match="no sweep"):
        rain_rate(plain, hid=xr.DataTree.from_dict({"/": xr.Dataset()}))
    with pytest.raises(TypeError):
        rain_rate(ds_array())


def ds_array():
    return sweep().DBZH


# --------------------------------------------------------------------------
# synthetic rain with a known rain rate
# --------------------------------------------------------------------------


def perils_dsds():
    """Gamma parameters of 1287 PERiLS 2022 PIPS minutes (radarx's packaged prior)."""
    import pandas as pd
    from scipy.special import gammaln

    path = Path(retrieve.__file__).parent / "data" / "dsd_prior_perils2022.csv"
    df = pd.read_csv(
        path, comment="#", header=None, names=["log10_nw", "dm", "mu", "iop"]
    )
    d = np.arange(0.05, 8.0001, 0.05)
    mu = df.mu.values[:, None]
    f = 6.0 / 256.0 * np.exp((4 + mu) * np.log(4 + mu) - gammaln(mu + 4))
    x = d[None, :] / df.dm.values[:, None]
    nd = 10 ** df.log10_nw.values[:, None] * f * x**mu * np.exp(-(4 + mu) * x)
    return xr.DataArray(nd, dims=("sample", "diameter"), coords={"diameter": d})


def skill(estimate, truth):
    """Normalized bias and normalized RMSE of an estimate, over the finite pairs."""
    estimate, truth = np.asarray(estimate, float), np.asarray(truth, float)
    ok = np.isfinite(estimate)
    nb = (estimate[ok].sum() - truth[ok].sum()) / truth[ok].sum()
    nrmse = np.sqrt(np.mean((estimate[ok] - truth[ok]) ** 2)) / truth[ok].mean()
    return nb, nrmse


def test_exponential_rain_of_known_rate():
    """Marshall-Palmer drops of 10 mm/h: R(A) at S band and R(Z) return it closely."""
    d = np.arange(0.05, 8.0001, 0.05)
    lam = 4.1 * 10.0**-0.21
    nd = xr.DataArray(
        (8000.0 * np.exp(-lam * d))[None, :],
        dims=("s", "diameter"),
        coords={"diameter": d},
    )
    for band, expected in (("S", 0.1), ("C", 0.1), ("X", 0.15)):
        out = radar_from_dsd(nd, band=band, temperature=20.0)
        truth = float(out.RAIN_RATE[0])
        est = float(rain_rate_a(out, band=band)[0])
        assert abs(est / truth - 1.0) < expected, (band, est, truth)
    out = radar_from_dsd(nd, band="S", temperature=20.0)
    truth = float(out.RAIN_RATE[0])
    assert abs(float(rain_rate_z(out)[0]) / truth - 1.0) < 0.3
    assert abs(float(rain_rate_kdp(out)[0]) / truth - 1.0) < 0.3


@pytest.mark.parametrize("engine", ENGINES)
def test_perils_dsd_ensemble_s_band(engine):
    pytest.importorskip("scipy")
    nd = perils_dsds()
    out = radar_from_dsd(nd, band="S", temperature=20.0)
    truth = out.RAIN_RATE.values
    heavy = truth >= 5.0
    assert heavy.sum() > 500
    stats = {
        "z": rain_rate_z(out, engine=engine).values,
        "kdp": rain_rate_kdp(out, engine=engine).values,
        "z_zdr": rain_rate_z_zdr(out, engine=engine).values,
        "a": rain_rate_a(out, engine=engine).values,
    }
    s = {k: skill(v[heavy], truth[heavy]) for k, v in stats.items()}
    # biases and scatter of the relations on these DSDs (R >= 5 mm/h)
    assert abs(s["kdp"][0]) < 0.05 and s["kdp"][1] < 0.30
    assert abs(s["z"][0]) < 0.20 and s["z"][1] < 0.70
    assert abs(s["z_zdr"][0]) < 0.30 and s["z_zdr"][1] < 0.80
    assert abs(s["a"][0]) < 0.30 and s["a"][1] < 0.60
    # KDP is the least scattered of the relations in heavy rain
    assert s["kdp"][1] < min(s["z"][1], s["z_zdr"][1], s["a"][1])
    # the median of every ratio is close to one
    for key, est in stats.items():
        assert abs(np.median(est[heavy] / truth[heavy]) - 1.0) < 0.2, key


def test_perils_dsd_ensemble_x_band():
    pytest.importorskip("scipy")
    out = radar_from_dsd(perils_dsds(), band="X", temperature=20.0)
    truth = out.RAIN_RATE.values
    heavy = truth >= 5.0
    nb_k, nrmse_k = skill(rain_rate_kdp(out, band="X").values[heavy], truth[heavy])
    nb_a, nrmse_a = skill(rain_rate_a(out, band="X").values[heavy], truth[heavy])
    assert abs(nb_k) < 0.05 and nrmse_k < 0.30
    assert abs(nb_a) < 0.10 and nrmse_a < 0.35


def test_perils_blend_returns_the_dsd_rain_rate():
    """Rain with a known classification: the blend equals R(Z, ZDR) of the DSD."""
    pytest.importorskip("scipy")
    out = radar_from_dsd(perils_dsds(), band="S", temperature=20.0)
    ds = out[["DBZH", "ZDR", "KDP", "AH"]].assign(
        HID=(
            "sample",
            np.full(out.sizes["sample"], PARK["light_and_moderate_rain"], np.int8),
            {
                "flag_values": np.arange(1, 9, dtype=np.int8),
                "flag_meanings": " ".join(c[2] for c in hid_classes("park", "S")),
            },
        )
    )
    est = rain_rate(ds)
    assert bool((est.RAIN_RATE_FLAG == 0).all())
    truth = out.RAIN_RATE.values
    heavy = truth >= 5.0
    nb, nrmse = skill(est.RAIN_RATE.values[heavy], truth[heavy])
    assert abs(nb) < 0.30 and nrmse < 0.80
    np.testing.assert_allclose(
        est.RAIN_RATE.values, rain_rate_z_zdr(ds).values, rtol=0, atol=0
    )


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------


def test_real_c_band_sweep_reproduces_the_cmac_rain_rate():
    """ARM CSAPR2 (C band): R(A) of the CMAC specific attenuation at 20 degrees Celsius."""
    xd = pytest.importorskip("xradar")
    from open_radar_data import DATASETS

    path = DATASETS.fetch("corcsapr2cmacppiM1.c1.20181111.030003.nc")
    sw = xd.io.open_cfradial1_datatree(path)["sweep_0"].to_dataset()
    before = sw.copy(deep=True)
    est = rain_rate_a(sw, a="specific_attenuation", band="C", temperature=20.0)
    ref = sw["rain_rate_A"]
    ok = np.isfinite(est.values) & np.isfinite(ref.values) & (ref.values > 1.0)
    assert ok.sum() > 10000
    ratio = est.values[ok] / ref.values[ok]
    assert abs(np.median(ratio) - 1.0) < 0.01
    assert np.percentile(ratio, 5) > 0.97 and np.percentile(ratio, 95) < 1.03
    # the temperature of the sounding changes the coefficients
    warm = rain_rate_a(
        sw, a="specific_attenuation", band="C", temperature="sounding_temperature"
    )
    assert np.isfinite(warm).sum() > 0
    assert not np.allclose(warm.values[ok], est.values[ok])
    xr.testing.assert_identical(sw, before)


def test_real_c_band_blend():
    xd = pytest.importorskip("xradar")
    from open_radar_data import DATASETS

    from radarx.retrieve import estimate_kdp

    path = DATASETS.fetch("corcsapr2cmacppiM1.c1.20181111.030003.nc")
    sw = xd.io.open_cfradial1_datatree(path)["sweep_0"].to_dataset()
    kdp = estimate_kdp(
        sw,
        phidp="uncorrected_differential_phase",
        rhohv="uncorrected_copol_correlation_coeff",
        dbz="uncorrected_reflectivity_h",
    )
    sw = sw.assign(KDP=kdp.KDP, met=sw.gate_id.isin([1, 2, 4]))
    classes_c = sw.radarx.hid(
        "sounding_temperature",
        band="C",
        mask="met",
        dbz="attenuation_corrected_reflectivity_h",
        zdr="attenuation_corrected_differential_reflectivity",
        kdp="KDP",
        rhohv="copol_correlation_coeff",
    )
    out = rain_rate(
        sw.assign(HID=classes_c.HID),
        band="C",
        class_relations={"drizzle": "a", "rain": "a", "big_drops": "a"},
        a="specific_attenuation",
        temperature="sounding_temperature",
    )
    flag = out.RAIN_RATE_FLAG.values
    assert (flag == 0).sum() > 1000  # rain gates
    assert (flag == 1).sum() > 100  # gates of classes without a relation (ice, hail)
    assert (flag == 3).sum() > 1000  # not classified
    assert np.isfinite(out.RAIN_RATE.values[flag == 0]).all()
    assert np.isnan(out.RAIN_RATE.values[flag != 0]).all()


def rain_like_sweep(seed):
    """A sweep with rain-like fields and a high correlation coefficient."""
    ds = sweep(10, 12, seed=seed)
    rng = np.random.default_rng(seed)
    dims, shape = ("azimuth", "range"), ds.DBZH.shape
    return ds.assign(
        DBZH=(dims, rng.uniform(25, 45, shape)),
        ZDR=(dims, rng.uniform(0.5, 2.0, shape)),
        KDP=(dims, rng.uniform(0.0, 1.0, shape)),
        RHOHV=(dims, np.full(shape, 0.99)),
    )


def test_blend_datatree_computes_the_classification_once():
    nodes = {
        "/": xr.Dataset(),
        "sweep_0": rain_like_sweep(1),
        "sweep_1": rain_like_sweep(2),
        # no ZDR and RHOHV: the classification has no such sweep
        "sweep_2": rain_like_sweep(3).drop_vars(["ZDR", "RHOHV"]),
    }
    tree = xr.DataTree.from_dict(nodes)
    out = rain_rate(tree, band="S", hid_kwargs={"scores": False})
    assert set(out.children) == {"sweep_0", "sweep_1"}
    classified = hid(tree, band="S", scores=False)
    for name in ("sweep_0", "sweep_1"):
        ds = tree[name].to_dataset().assign(HID=classified[name].to_dataset()["HID"])
        same(out[name].to_dataset(), rain_rate(ds, band="S"))
        assert (out[name]["RAIN_RATE_FLAG"] == 0).any()


def test_blend_coefficients_and_class_field_checks():
    ds = sweep(10, 12)
    h = classes(ds)
    with pytest.raises(ValueError, match=r"unknown relations \['zz'\] in coefficients"):
        rain_rate(ds, hid=h, coefficients={"zz": (300.0, 1.4)})
    short = h.copy()
    short.attrs["flag_values"] = short.attrs["flag_values"][:-1]
    with pytest.raises(ValueError, match="differ in length"):
        rain_rate(ds, hid=short)


@pytest.mark.parametrize("engine", ENGINES)
def test_blend_with_a_float_class_field(engine):
    """NaN in a float class field is class 0 (unclassified)."""
    ds = sweep(10, 12)
    h = classes(ds)
    as_float = h.astype(np.float64)
    as_float.values[2:4, :] = np.nan
    as_int = h.copy()
    as_int.values[2:4, :] = 0
    got = rain_rate(ds, hid=as_float, engine=engine)
    want = rain_rate(ds, hid=as_int, engine=engine)
    same(got, want)
    assert (got.RAIN_RATE_FLAG.values[2:4] != 0).all()


def test_compiled_kernel_is_required_when_asked(monkeypatch):
    ds = sweep(3, 3)
    monkeypatch.setattr(qpe, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError, match="compiled QPE kernel is not available"):
        rain_rate_z(ds, engine="compiled")
    with pytest.raises(ImportError, match="compiled QPE kernel is not available"):
        rain_rate(ds, hid=classes(ds), engine="compiled")
    # the NumPy implementation is used without the kernel
    assert rain_rate_z(ds, engine="auto").name == "RAIN_RATE_Z"
