#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the hail products (SHI, MESH, POSH, hail kinetic energy)
===================================================================

The expected values are written here from the equations of Witt et al. (1998)
(Eqs. 1 to 6) with explicit loops, not from the module. Columns with constant
reflectivity have a closed-form SHI, the weights are checked at their limits, and
the SHI of the five volume scans of the paper's Fig. 4 (H0 = 3 km, Hm20 = 6 km,
reflectivity and height of each storm component read off the figure) is
reproduced. The compiled kernel and the NumPy implementation must agree to
rounding (rtol 1e-12).
"""
import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.retrieve import (
    hail_kinetic_energy,
    hail_products,
    mesh,
    posh,
    shi,
    vil,
)

from .test_vil import ALT, storm_volume

hailmod = __import__("radarx.retrieve.hail", fromlist=["x"])
vilmod = __import__("radarx.retrieve.vil", fromlist=["x"])

ENGINES = ["numpy"] + (["compiled"] if hailmod.HAS_COMPILED_KERNEL else [])
compiled_only = pytest.mark.skipif(
    not hailmod.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)
H0, HM20 = 4000.0, 7000.0


def energy(dbz, zl=40.0, zu=50.0):
    """Eq. 1 of Witt et al. (1998), J m-2 s-1."""
    if not dbz > zl:
        return 0.0
    w = 1.0 if dbz >= zu else (dbz - zl) / (zu - zl)
    return 5e-6 * 10.0 ** (0.084 * dbz) * w


def weight_t(h, h0, hm20):
    """Eq. 2."""
    if h <= h0:
        return 0.0
    if h >= hm20:
        return 1.0
    return (h - h0) / (hm20 - h0)


def hand_shi(h, dbz, h0, hm20, zl=40.0, zu=50.0):
    """Eq. 3 as the discrete sum of the paper, sample by sample."""
    order = np.argsort(h)
    h = np.asarray(h, dtype=float)[order]
    dbz = np.asarray(dbz, dtype=float)[order]
    n = len(h)
    total = 0.0
    for i in range(n):
        if i == 0:
            dh = h[1] - h[0]
            if h[0] > h0:  # base of the storm above H0
                dh = 0.5 * (h[0] + h[1]) - h0
        elif i == n - 1:
            dh = h[n - 1] - h[n - 2]
        else:
            dh = 0.5 * (h[i + 1] - h[i - 1])
        total += weight_t(h[i], h0, hm20) * energy(dbz[i], zl, zu) * dh
    return 0.1 * total


def hand_products(h, dbz, h0, hm20, radar=0.0, off=121.0, poff=50.0):
    """SHI, MESH, POSH and WT of one column by Eqs. 3 to 6."""
    s = hand_shi(h, dbz, h0, hm20)
    wt = max(57.5 * (h0 - radar) / 1000.0 - off, 20.0)
    p = 0.0 if s <= 0 else min(100.0, max(0.0, 29.0 * np.log(s / wt) + poff))
    return s, 2.54 * s**0.5, p, wt


def hail_storm_dbz(x, y, z):
    """A cell with a 60 dBZ core that reaches 10 km, so that SHI is positive."""
    r2 = ((x - 40e3) ** 2 + (y - 20e3) ** 2) / (15e3) ** 2
    zz = np.clip(z / 14e3, 0, 1.3)
    dbz = 66.0 * np.exp(-r2) - 12.0 * zz**2 - 6.0
    return np.where(dbz > -10.0, dbz, -20.0)


def column(dbz, z=None, **coords):
    """Grid dataset with one (y, x) column of reflectivity."""
    dbz = np.asarray(dbz, dtype=float)
    z = np.arange(dbz.size) * 500.0 if z is None else np.asarray(z, dtype=float)
    return xr.Dataset(
        {"DBZH": (("z", "y", "x"), dbz[:, None, None], {"units": "dBZ"})},
        coords={"z": z, "y": [0.0], "x": [0.0], **coords},
    )


def one(da):
    return float(np.asarray(da.values).squeeze())


# --------------------------------------------------------------------------
# analytic columns
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_constant_reflectivity_closed_form(engine):
    """Z >= ZU everywhere on a regular grid: SHI = 0.1 E sum(W_T) d."""
    z = np.arange(0.0, 12001.0, 500.0)
    out = hail_products(
        column(np.full(z.size, 55.0), z),
        freezing_level=4000.0,
        minus20_level=7000.0,
        engine=engine,
    )
    # W_T sums to (0.5 + 1 + ... + 2.5) / 3 = 2.5 for the five levels from 4.5 to
    # 6.5 km, and to 1 for each of the eleven levels from 7 to 12 km
    expected = 0.1 * 5e-6 * 10 ** (0.084 * 55.0) * (2.5 + 11.0) * 500.0
    np.testing.assert_allclose(one(out.SHI), expected, rtol=1e-6)
    np.testing.assert_allclose(one(out.SHI), 140.69, atol=0.01)
    np.testing.assert_allclose(one(out.MESH), 2.54 * np.sqrt(expected), rtol=1e-6)
    np.testing.assert_allclose(one(out.SHI_WARNING_THRESHOLD), 57.5 * 4.0 - 121.0)
    np.testing.assert_allclose(
        one(out.POSH), 29.0 * np.log(expected / 109.0) + 50.0, rtol=1e-5
    )
    np.testing.assert_allclose(one(out.HKE), 5e-6 * 10 ** (0.084 * 55.0), rtol=1e-6)
    assert out.SHI.attrs["units"] == "J m-1 s-1"
    assert out.MESH.attrs["units"] == "mm"
    assert out.POSH.attrs["units"] == "%"
    assert out.HKE.attrs["units"] == "J m-2 s-1"
    assert out.SHI.dtype == np.float32
    # echo continues at the highest sample
    assert int(out.SHI_TOP_TRUNCATED.squeeze()) == 1


@pytest.mark.parametrize("engine", ENGINES)
def test_layer_above_the_minus20_level_closed_form(engine):
    """A 50 dBZ layer entirely above Hm20 has W_T = 1: SHI = 0.1 E (sum of depths)."""
    z = np.arange(0.0, 14001.0, 1000.0)
    dbz = np.where((z >= 8000) & (z <= 11000), 50.0, -np.inf)
    out = shi(
        column(dbz, z), freezing_level=4000.0, minus20_level=7000.0, engine=engine
    )
    # four samples with 1 km depth each (interior (H+1 - H-1)/2), top sample too
    np.testing.assert_allclose(
        one(out), 0.1 * 5e-6 * 10 ** (0.084 * 50.0) * 4 * 1000.0, rtol=1e-6
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_weights_at_their_limits(engine):
    """W(Z) is 0 at ZL, 1 at ZU and linear between; W_T is 0 at H0 and 1 at Hm20."""
    z = np.array([0.0, 4000.0, 5500.0, 7000.0, 8000.0])

    def total(dbz_at):
        dbz = np.array([0.0, 0.0, 0.0, 0.0, dbz_at])
        return one(
            shi(
                column(dbz, z),
                freezing_level=4000.0,
                minus20_level=7000.0,
                engine=engine,
            )
        )

    # only the top sample (8 km, W_T = 1) has echo; dH = 1000 m
    assert total(40.0) == 0.0  # Z = ZL: W = 0
    assert total(35.0) == 0.0
    np.testing.assert_allclose(
        total(45.0), 0.1 * 5e-6 * 10 ** (0.084 * 45) * 0.5 * 1000.0, rtol=1e-6
    )
    np.testing.assert_allclose(
        total(50.0), 0.1 * 5e-6 * 10 ** (0.084 * 50) * 1000.0, rtol=1e-6
    )
    np.testing.assert_allclose(
        total(60.0), 0.1 * 5e-6 * 10 ** (0.084 * 60) * 1000.0, rtol=1e-6
    )
    # sample at the heights of the levels
    dbz = np.array([0.0, 60.0, 60.0, 60.0, 0.0])
    s = one(
        shi(
            column(dbz, z),
            freezing_level=4000.0,
            minus20_level=7000.0,
            engine=engine,
        )
    )
    e = 5e-6 * 10 ** (0.084 * 60)
    # 4 km: W_T = 0; 5.5 km: W_T = 0.5, dH = 1500; 7 km: W_T = 1, dH = 1250
    np.testing.assert_allclose(s, 0.1 * e * (0.5 * 1500.0 + 1.0 * 1250.0), rtol=1e-6)
    # custom limits
    custom = shi(
        column(np.array([0.0, 0.0, 0.0, 0.0, 45.0]), z),
        freezing_level=4000.0,
        minus20_level=7000.0,
        zl=30.0,
        zu=60.0,
        engine=engine,
    )
    np.testing.assert_allclose(
        one(custom), 0.1 * 5e-6 * 10 ** (0.084 * 45) * 0.5 * 1000.0, rtol=1e-6
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_base_above_the_freezing_level(engine):
    """The bottom sample above H0 counts from H0 to the middle with the next one."""
    z = np.array([5000.0, 6000.0, 9000.0])
    dbz = np.array([55.0, 55.0, 55.0])
    out = hail_products(
        column(dbz, z), freezing_level=3000.0, minus20_level=6000.0, engine=engine
    )
    e = 5e-6 * 10 ** (0.084 * 55.0)
    wt0 = (5000.0 - 3000.0) / 3000.0
    expected = (
        0.1 * e * (wt0 * (0.5 * (5000 + 6000) - 3000.0) + 1.0 * 2000.0 + 1.0 * 3000.0)
    )
    np.testing.assert_allclose(one(out.SHI), expected, rtol=1e-6)
    np.testing.assert_allclose(
        one(out.SHI), hand_shi(z, dbz, 3000.0, 6000.0), rtol=1e-6
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_nonuniform_spacing_and_sample_order(engine):
    rng = np.random.default_rng(5)
    z = np.sort(rng.uniform(100.0, 13000.0, 14))
    dbz = rng.uniform(20.0, 66.0, z.size)
    expected = hand_shi(z, dbz, H0, HM20)
    got = shi(column(dbz, z), freezing_level=H0, minus20_level=HM20, engine=engine)
    np.testing.assert_allclose(one(got), expected, rtol=1e-6)
    # the order of the samples does not matter
    shuffled = shi(
        column(dbz[::-1], z[::-1]),
        freezing_level=H0,
        minus20_level=HM20,
        engine=engine,
    )
    np.testing.assert_allclose(one(shuffled), expected, rtol=1e-6)


# --------------------------------------------------------------------------
# Fig. 4 of Witt et al. (1998)
# --------------------------------------------------------------------------

# (height ARL in km, maximum reflectivity in dBZ) of each storm component of the
# five volume scans, read off Fig. 4 of the paper (H0 = 3 km, Hm20 = 6 km), and
# the SHI printed above each scan. The heights are read by eye to about 0.1 km.
FIG4 = [
    ([(2.68, 52), (5.3, 51)], 22.0),
    ([(2.55, 56), (5.1, 58), (7.38, 43)], 66.0),
    ([(2.44, 53), (4.9, 59), (7.14, 43)], 68.0),
    ([(2.3, 54), (4.72, 63), (6.87, 48)], 134.0),
    ([(2.28, 67), (4.56, 65), (6.6, 48), (8.6, 39)], 172.0),
]


@pytest.mark.parametrize("engine", ENGINES)
def test_worked_example_of_the_paper(engine):
    """The SHI of the five volume scans of Fig. 4, from the heights read off it."""
    published, computed = [], []
    for comps, printed in FIG4:
        h = np.array([c[0] for c in comps]) * 1000.0
        d = np.array([c[1] for c in comps], dtype=float)
        got = one(
            shi(
                column(d, h), freezing_level=3000.0, minus20_level=6000.0, engine=engine
            )
        )
        published.append(printed)
        computed.append(got)
    published, computed = np.array(published), np.array(computed)
    # four scans within 4 %, the first within 15 % (one component, 0.1 km reading error)
    np.testing.assert_allclose(computed[1:], published[1:], rtol=0.04)
    np.testing.assert_allclose(computed[0], published[0], rtol=0.15)
    # the SHI rises with the scan number as in the figure
    assert (np.diff(computed) > 0).all()


# --------------------------------------------------------------------------
# MESH, POSH and WT by hand
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_mesh_posh_and_threshold_by_hand(engine):
    """Eqs. 4 to 6 evaluated by hand for several SHI."""
    z = np.arange(0.0, 14001.0, 1000.0)
    for dbz_top, h0 in [(50.0, 4000.0), (60.0, 3000.0), (45.0, 4500.0)]:
        dbz = np.where(z >= h0, dbz_top, -np.inf)
        out = hail_products(
            column(dbz, z),
            freezing_level=h0,
            minus20_level=h0 + 3000.0,
            engine=engine,
        )
        s, m, p, wt = hand_products(z, dbz, h0, h0 + 3000.0)
        np.testing.assert_allclose(one(out.SHI), s, rtol=1e-6)
        np.testing.assert_allclose(one(out.MESH), m, rtol=1e-6)
        np.testing.assert_allclose(one(out.POSH), p, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(one(out.SHI_WARNING_THRESHOLD), wt)
    # MESH at the SHI of the paper's Table 6: 2.54 sqrt(511) = 57.4 mm
    np.testing.assert_allclose(2.54 * np.sqrt(511.0), 57.41, atol=0.01)


@pytest.mark.parametrize("engine", ENGINES)
def test_posh_is_50_at_the_threshold_and_limited(engine):
    z = np.arange(0.0, 14001.0, 1000.0)
    dbz = np.where(z >= 5000.0, 55.0, -np.inf)
    base = hail_products(
        column(dbz, z), freezing_level=5000.0, minus20_level=8000.0, engine=engine
    )
    s = one(base.SHI)
    # choose the offset that makes WT equal SHI: POSH = 50 exactly
    h0_km = 5.0
    off = 57.5 * h0_km - s
    at = posh(
        column(dbz, z),
        freezing_level=5000.0,
        minus20_level=8000.0,
        warning_offset=off,
        engine=engine,
    )
    np.testing.assert_allclose(one(at), 50.0, atol=1e-3)
    # a much higher threshold (offset) gives POSH 0 and a lower one gives 100
    low = posh(
        column(dbz, z),
        freezing_level=5000.0,
        minus20_level=8000.0,
        warning_offset=-5000.0,
        engine=engine,
    )
    assert one(low) == 0.0
    high = posh(
        column(dbz, z),
        freezing_level=5000.0,
        minus20_level=8000.0,
        warning_offset=287.4,  # WT = 20.1: clipped at 100
        posh_offset=90.0,
        engine=engine,
    )
    assert one(high) == 100.0
    # the warning threshold has a floor of 20 J m-1 s-1
    floor = hail_products(
        column(dbz, z), freezing_level=1000.0, minus20_level=4000.0, engine=engine
    )
    assert one(floor.SHI_WARNING_THRESHOLD) == 20.0
    # the western region adjustment: WT = 57.5 H0 - 5.8 and an offset of 30
    west = hail_products(
        column(dbz, z),
        freezing_level=3000.0,
        minus20_level=6000.0,
        warning_offset=5.8,
        posh_offset=30.0,
        engine=engine,
    )
    np.testing.assert_allclose(one(west.SHI_WARNING_THRESHOLD), 57.5 * 3.0 - 5.8)
    np.testing.assert_allclose(
        one(west.POSH),
        min(
            100.0,
            max(
                0.0,
                29 * np.log(one(west.SHI) / (57.5 * 3.0 - 5.8)) + 30.0,
            ),
        ),
        rtol=1e-5,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_threshold_is_above_the_radar(engine):
    """WT uses the 0 degC level above the radar, not above sea level."""
    z = np.arange(0.0, 14001.0, 1000.0)
    dbz = np.full(z.size, 55.0)
    a = hail_products(
        column(dbz, z), freezing_level=4000.0, minus20_level=7000.0, engine=engine
    )
    b = hail_products(
        column(dbz, z),
        freezing_level=4000.0,
        minus20_level=7000.0,
        radar_height=1500.0,
        engine=engine,
    )
    np.testing.assert_allclose(one(a.SHI_WARNING_THRESHOLD), 57.5 * 4.0 - 121.0)
    np.testing.assert_allclose(one(b.SHI_WARNING_THRESHOLD), 57.5 * 2.5 - 121.0)
    np.testing.assert_allclose(one(a.SHI), one(b.SHI))
    # the site altitude of the dataset is the default
    site = column(dbz, z).assign(altitude=1500.0)
    c = hail_products(site, freezing_level=4000.0, minus20_level=7000.0, engine=engine)
    np.testing.assert_allclose(one(c.SHI_WARNING_THRESHOLD), 57.5 * 2.5 - 121.0)


# --------------------------------------------------------------------------
# invariants
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_zero_below_the_freezing_level_and_below_the_threshold(engine):
    z = np.arange(0.0, 14001.0, 500.0)
    # strong echo only below the 0 degC level
    dbz = np.where(z <= 4000.0, 65.0, -np.inf)
    out = hail_products(
        column(dbz, z), freezing_level=H0, minus20_level=HM20, engine=engine
    )
    assert one(out.SHI) == 0.0 and one(out.MESH) == 0.0 and one(out.POSH) == 0.0
    # echo above the freezing level but not above ZL
    weak = np.where(z > 4000.0, 40.0, -np.inf)
    out = hail_products(
        column(weak, z), freezing_level=H0, minus20_level=HM20, engine=engine
    )
    assert one(out.SHI) == 0.0 and one(out.POSH) == 0.0
    # no echo at all
    clear = hail_products(
        column(np.full(z.size, -np.inf), z),
        freezing_level=H0,
        minus20_level=HM20,
        engine=engine,
    )
    assert one(clear.SHI) == 0.0 and one(clear.HKE) == 0.0
    assert int(clear.SHI_TOP_TRUNCATED.squeeze()) == 0


@pytest.mark.parametrize("engine", ENGINES)
def test_monotone_in_reflectivity_and_in_the_levels(engine):
    z = np.arange(0.0, 14001.0, 500.0)
    base = np.where(z > 3000.0, 48.0, -np.inf)
    previous = None
    for bump in range(0, 20, 2):
        d = base + bump
        s = one(shi(column(d, z), freezing_level=H0, minus20_level=HM20, engine=engine))
        if previous is not None:
            assert s > previous
        previous = s
    # a lower freezing level (more of the storm in the hail layer) gives a larger SHI
    d = np.where(z > 2000.0, 55.0, -np.inf)
    values = [
        one(
            shi(
                column(d, z),
                freezing_level=h0,
                minus20_level=h0 + 3000.0,
                engine=engine,
            )
        )
        for h0 in (2000.0, 3000.0, 4000.0, 5000.0, 6000.0)
    ]
    assert (np.diff(values) < 0).all()
    # the same height shift of the samples and of the levels leaves SHI unchanged
    a = one(shi(column(d, z), freezing_level=H0, minus20_level=HM20, engine=engine))
    b = one(
        shi(
            column(d, z + 700.0),
            freezing_level=H0 + 700.0,
            minus20_level=HM20 + 700.0,
            engine=engine,
        )
    )
    np.testing.assert_allclose(a, b, rtol=1e-6)
    # MESH is monotone in SHI
    m = [
        one(
            mesh(
                column(base + b_, z),
                freezing_level=H0,
                minus20_level=HM20,
                engine=engine,
            )
        )
        for b_ in (0.0, 4.0, 8.0, 12.0)
    ]
    assert (np.diff(m) > 0).all()


@pytest.mark.parametrize("engine", ENGINES)
def test_hail_kinetic_energy_of_the_lowest_sample(engine):
    z = np.array([300.0, 1000.0, 3000.0])
    cases = {
        30.0: 0.0,  # below ZL
        40.0: 0.0,
        45.0: 0.5 * 5e-6 * 10 ** (0.084 * 45.0),
        50.0: 5e-6 * 10 ** (0.084 * 50.0),
        70.0: 5e-6 * 10 ** (0.084 * 70.0),
    }
    for dbz, expected in cases.items():
        got = hail_kinetic_energy(column(np.array([dbz, 60.0, 20.0]), z), engine=engine)
        np.testing.assert_allclose(one(got), expected, rtol=1e-6, atol=1e-12)
    # 3.8 J m-2 s-1 at 70 dBZ (Fig. 3 of the paper)
    np.testing.assert_allclose(5e-6 * 10 ** (0.084 * 70.0), 3.79, atol=0.01)
    # the flux is that of 5.0e-6 Z^0.84 with Z in mm6 m-3 (Holleman 2001, Eq. 2.4)
    np.testing.assert_allclose(
        5e-6 * 10 ** (0.084 * 55.0), 5.0e-6 * (10 ** (55.0 / 10)) ** 0.84, rtol=1e-12
    )
    # a gap below the lowest valid sample: the lowest valid one is used
    got = hail_kinetic_energy(column(np.array([np.nan, 55.0, 20.0]), z), engine=engine)
    np.testing.assert_allclose(one(got), 5e-6 * 10 ** (0.084 * 55.0), rtol=1e-6)
    assert hail_kinetic_energy(column(np.array([np.nan, np.nan]))).isnull().all()
    assert got.attrs["units"] == "J m-2 s-1"


@pytest.mark.parametrize("engine", ENGINES)
def test_too_few_samples_and_missing_levels(engine):
    # one sample: no SHI, but a kinetic energy flux
    one_sample = column(np.array([55.0]), np.array([5000.0]))
    out = hail_products(
        one_sample, freezing_level=H0, minus20_level=HM20, engine=engine
    )
    assert np.isnan(out.SHI.values).all() and np.isnan(out.POSH.values).all()
    assert one(out.HKE) > 0
    # all samples missing
    none = column(np.full(4, np.nan))
    out = hail_products(none, freezing_level=H0, minus20_level=HM20, engine=engine)
    assert np.isnan(out.SHI.values).all() and np.isnan(out.HKE.values).all()
    assert int(out.SHI_TOP_TRUNCATED.squeeze()) == 0
    # unknown levels (NaN) give NaN products, not 0
    z = np.arange(0.0, 10001.0, 1000.0)
    d = column(np.full(z.size, 55.0), z)
    for h0, hm in [(np.nan, HM20), (H0, np.nan), (7000.0, 4000.0), (4000.0, 4000.0)]:
        da0 = xr.DataArray(np.full((1, 1), h0), dims=("y", "x"))
        dam = xr.DataArray(np.full((1, 1), hm), dims=("y", "x"))
        out = hail_products(d, freezing_level=da0, minus20_level=dam, engine=engine)
        assert np.isnan(out.SHI.values).all()
        assert np.isnan(out.MESH.values).all() and np.isnan(out.POSH.values).all()
        assert one(out.HKE) > 0


# --------------------------------------------------------------------------
# kernel against the NumPy implementation
# --------------------------------------------------------------------------


@compiled_only
@pytest.mark.parametrize("shared", [True, False])
def test_compiled_equals_numpy_on_random_columns(shared):
    rng = np.random.default_rng(11)
    nk, nc = 12, 700
    v = rng.uniform(-10.0, 72.0, (nk, nc))
    v[rng.random((nk, nc)) < 0.15] = np.nan
    v[rng.random((nk, nc)) < 0.1] = -np.inf
    v[:, 0] = np.nan  # an empty column
    v[1:, 1] = np.nan  # a single sample
    if shared:
        h = np.sort(rng.uniform(100.0, 14000.0, nk))
    else:
        h = rng.uniform(100.0, 14000.0, (nk, nc))
        h[rng.random((nk, nc)) < 0.05] = np.nan
        h[:, 2] = h[0, 2]  # equal heights in one column
    h0 = rng.uniform(1500.0, 5500.0, nc)
    hm20 = h0 + rng.uniform(1500.0, 3500.0, nc)
    h0[3], hm20[4], hm20[5] = np.nan, np.nan, h0[5] - 100.0
    h0[6] = 14500.0  # all samples below the freezing level
    for radar in (0.0, 600.0):
        a = hailmod._columns_numpy(
            h, v, h0, hm20, shared, radar, 40.0, 50.0, 121.0, 50.0
        )
        b = hailmod._hail.columns(
            h, v, h0, hm20, shared, radar, 40.0, 50.0, 121.0, 50.0, n_threads=3
        )
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=0.0, equal_nan=True)
        assert np.isfinite(a[0]).sum() > 300


@compiled_only
def test_kernel_rejects_bad_arguments():
    k = hailmod._hail
    v = np.zeros((3, 4))
    ok = (np.arange(3.0), v, np.zeros(4), np.ones(4))
    args = (True, 0.0, 40.0, 50.0, 121.0, 50.0)
    k.columns(*ok, *args)
    with pytest.raises(ValueError, match="2-D"):
        k.columns(np.arange(3.0), np.zeros(3), np.zeros(4), np.ones(4), *args)
    with pytest.raises(ValueError, match="shared heights"):
        k.columns(np.arange(4.0), v, np.zeros(4), np.ones(4), *args)
    with pytest.raises(ValueError, match="shape of v"):
        k.columns(
            np.zeros((3, 2)),
            v,
            np.zeros(4),
            np.ones(4),
            False,
            0.0,
            40.0,
            50.0,
            121.0,
            50.0,
        )
    with pytest.raises(ValueError, match="one value per column"):
        k.columns(np.arange(3.0), v, np.zeros(3), np.ones(4), *args)
    with pytest.raises(ValueError, match="above zl"):
        k.columns(*ok, True, 0.0, 50.0, 40.0, 121.0, 50.0)
    # no columns
    out = k.columns(np.arange(3.0), np.zeros((3, 0)), np.zeros(0), np.zeros(0), *args)
    assert out.shape == (6, 0)


def test_engine_errors(monkeypatch):
    ds = column(np.full(6, 50.0))
    with pytest.raises(ValueError, match="engine"):
        shi(ds, freezing_level=H0, minus20_level=HM20, engine="gpu")
    monkeypatch.setattr(hailmod, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError, match="hail kernel"):
        shi(ds, freezing_level=H0, minus20_level=HM20, engine="compiled")
    # "auto" falls back to NumPy
    assert np.isfinite(
        shi(ds, freezing_level=1000.0, minus20_level=3000.0, engine="auto").values
    ).all()


# --------------------------------------------------------------------------
# the 0 and -20 degC levels
# --------------------------------------------------------------------------


def standard_profile(t0_height=4200.0, lapse=6.5e-3, nz=160):
    """A sounding with a constant lapse rate and a known 0 degC height."""
    height = np.linspace(0.0, 16000.0, nz)
    temperature = 273.15 - lapse * (height - t0_height)
    return xr.Dataset(
        {"temperature": ("height", temperature)}, coords={"height": height}
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_levels_from_a_profile(engine):
    z = np.arange(0.0, 14001.0, 500.0)
    d = column(np.full(z.size, 52.0), z)
    prof = standard_profile()
    h0 = 4200.0
    hm20 = 4200.0 + 20.0 / 6.5e-3
    got = hail_products(d, profile=prof, engine=engine)
    ref = hail_products(d, freezing_level=h0, minus20_level=hm20, engine=engine)
    np.testing.assert_allclose(got.SHI.values, ref.SHI.values, rtol=1e-4)
    np.testing.assert_allclose(
        one(got.SHI_WARNING_THRESHOLD), 57.5 * 4.2 - 121.0, rtol=1e-4
    )
    # a level that is given wins over the profile
    mixed = shi(d, profile=prof, freezing_level=3000.0, engine=engine)
    ref = shi(d, freezing_level=3000.0, minus20_level=hm20, engine=engine)
    np.testing.assert_allclose(mixed.values, ref.values, rtol=1e-4)
    # an ERA5-column style profile on z
    era = standard_profile().rename({"height": "z"})
    np.testing.assert_allclose(
        shi(d, profile=era, engine=engine).values,
        shi(d, profile=prof, engine=engine).values,
    )
    # a profile per time of a quasi-vertical profile
    times = np.array(["2022-03-30T12", "2022-03-30T13"], dtype="datetime64[ns]")
    qvp = xr.Dataset(
        {"DBZH": (("time", "height"), np.full((2, z.size), 52.0))},
        coords={"time": times, "height": z},
    )
    both = xr.concat([standard_profile(4200.0), standard_profile(3000.0)], dim="time")
    both = both.assign_coords(time=times)
    out = hail_products(qvp, profile=both, engine=engine)
    assert out.SHI.dims == ("time",)
    assert out.SHI.values[1] > out.SHI.values[0]
    np.testing.assert_allclose(
        out.SHI.values[0], one(shi(d, profile=prof, engine=engine)), rtol=1e-4
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_levels_as_two_dimensional_fields(engine):
    """A freezing level per grid cell (e.g. from ERA5) changes the SHI cell by cell."""
    z = np.arange(0.0, 14001.0, 500.0)
    ny, nx = 3, 4
    ds = xr.Dataset(
        {"DBZH": (("z", "y", "x"), np.full((z.size, ny, nx), 55.0))},
        coords={"z": z, "y": np.arange(ny) * 1e3, "x": np.arange(nx) * 1e3},
    )
    h0 = xr.DataArray(
        np.linspace(2500.0, 5500.0, ny * nx).reshape(ny, nx),
        dims=("y", "x"),
        coords={"y": ds.y, "x": ds.x},
        attrs={"units": "m"},
    )
    out = hail_products(ds, freezing_level=h0, minus20_level=h0 + 3000.0, engine=engine)
    assert out.SHI.dims == ("y", "x")
    for j in (0, 5, 11):
        iy, ix = divmod(j, nx)
        ref = hand_products(
            z, np.full(z.size, 55.0), float(h0[iy, ix]), float(h0[iy, ix]) + 3000.0
        )
        np.testing.assert_allclose(out.SHI.values[iy, ix], ref[0], rtol=1e-6)
        np.testing.assert_allclose(
            out.POSH.values[iy, ix], ref[2], rtol=1e-5, atol=1e-4
        )
    assert (np.diff(out.SHI.values.ravel()) < 0).all()
    # the transposed field gives the same result
    t = hail_products(
        ds, freezing_level=h0.T, minus20_level=(h0 + 3000.0).T, engine=engine
    )
    xr.testing.assert_allclose(out.SHI, t.SHI)
    # and the inputs are not changed
    assert h0.attrs == {"units": "m"}
    assert ds.x.attrs == {}


def test_level_errors():
    z = np.arange(0.0, 10001.0, 1000.0)
    d = column(np.full(z.size, 55.0), z)
    with pytest.raises(ValueError, match="0 and -20 degC"):
        shi(d)
    with pytest.raises(ValueError, match="0 and -20 degC"):
        shi(d, freezing_level=4000.0)
    with pytest.raises(ValueError, match="must be above"):
        shi(d, freezing_level=5000.0, minus20_level=4000.0)
    with pytest.raises(TypeError, match="number"):
        shi(d, freezing_level="high", minus20_level=7000.0)
    with pytest.raises(ValueError, match="does not fit"):
        shi(
            d,
            freezing_level=xr.DataArray(np.ones(3), dims="time"),
            minus20_level=7000.0,
        )
    with pytest.raises(ValueError, match="does not match"):
        shi(
            d,
            freezing_level=xr.DataArray(np.ones(3), dims="x", coords={"x": [0, 1, 2]}),
            minus20_level=7000.0,
        )
    with pytest.raises(TypeError, match="Dataset"):
        shi(d, profile=standard_profile()["temperature"])
    with pytest.raises(KeyError, match="temperature"):
        shi(
            d,
            profile=xr.Dataset(
                {"t": ("height", [1.0, 2.0])}, coords={"height": [0, 1]}
            ),
        )
    with pytest.raises(ValueError, match="height"):
        shi(
            d,
            profile=xr.Dataset(
                {"temperature": ("level", [1.0, 2.0])}, coords={"level": [0, 1]}
            ),
        )
    with pytest.raises(ValueError, match="zu"):
        shi(d, freezing_level=4000.0, minus20_level=7000.0, zl=50.0, zu=40.0)
    with pytest.raises(ValueError, match="missing"):
        shi(d, freezing_level=4000.0, minus20_level=7000.0, missing="skip")
    with pytest.raises(KeyError, match="fieldname"):
        shi(d, "fieldname", freezing_level=4000.0, minus20_level=7000.0)


# --------------------------------------------------------------------------
# inputs: grid, profile and polar volume
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_gaps_clear_and_the_dbz_name(engine):
    z = np.arange(0.0, 12001.0, 1000.0)
    dbz = np.full(z.size, 55.0)
    dbz[6] = np.nan  # a missing level inside the column
    d = column(dbz, z).rename({"DBZH": "reflectivity"})
    gap = shi(d, freezing_level=H0, minus20_level=HM20, engine=engine)
    keep = np.isfinite(dbz)
    np.testing.assert_allclose(
        one(gap), hand_shi(z[keep], dbz[keep], H0, HM20), rtol=1e-6
    )
    clear = shi(
        d, freezing_level=H0, minus20_level=HM20, missing="clear", engine=engine
    )
    dbz2 = np.where(np.isfinite(dbz), dbz, -np.inf)
    np.testing.assert_allclose(one(clear), hand_shi(z, dbz2, H0, HM20), rtol=1e-6)
    assert one(clear) < one(gap)
    # explicit field name
    d2 = d.rename({"reflectivity": "myfield"})
    np.testing.assert_allclose(
        one(shi(d2, "myfield", freezing_level=H0, minus20_level=HM20, engine=engine)),
        one(gap),
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_quasi_vertical_profile(engine):
    z = np.arange(0.0, 12001.0, 500.0)
    qvp = xr.Dataset(
        {
            "DBZH": (
                ("time", "height"),
                np.vstack([np.full(z.size, 55.0), np.full(z.size, 45.0)]),
            )
        },
        coords={"time": [0, 1], "height": z},
    )
    out = hail_products(qvp, freezing_level=H0, minus20_level=HM20, engine=engine)
    assert out.SHI.dims == ("time",)
    for i, value in enumerate((55.0, 45.0)):
        np.testing.assert_allclose(
            out.SHI.values[i], hand_shi(z, np.full(z.size, value), H0, HM20), rtol=1e-6
        )


@pytest.mark.parametrize("engine", ENGINES)
def test_polar_volume_against_a_hand_loop(engine):
    from .test_vil import hand_loop_column

    tree = storm_volume(field=hail_storm_dbz)
    out = hail_products(
        tree, freezing_level=3000.0, minus20_level=6000.0, engine=engine
    )
    assert out.SHI.dims == ("azimuth", "range") and "ground_range" in out.coords
    ground = out.ground_range.values
    checked = positive = 0
    # the cell is at azimuth 63 degrees and 45 km from the radar
    for iaz in (10, 58, 62, 64, 68, 75, 160):
        for j in range(40, ground.size, 9):
            h, v = hand_loop_column(tree, iaz, ground[j])
            if h.size < 2:
                continue
            s, m, p, wt = hand_products(h, v, 3000.0, 6000.0, radar=ALT)
            np.testing.assert_allclose(out.SHI.values[iaz, j], s, rtol=2e-4, atol=1e-4)
            np.testing.assert_allclose(out.MESH.values[iaz, j], m, rtol=2e-4, atol=1e-3)
            np.testing.assert_allclose(out.POSH.values[iaz, j], p, rtol=2e-4, atol=1e-3)
            np.testing.assert_allclose(out.SHI_WARNING_THRESHOLD.values[iaz, j], wt)
            checked += 1
            positive += s > 0
    assert checked > 60 and positive > 10
    # the flux of the lowest beam
    lowest = hail_kinetic_energy(tree, engine=engine)
    assert lowest.dims == ("azimuth", "range")
    iaz, j = 62, 85
    h, v = hand_loop_column(tree, iaz, ground[j])
    np.testing.assert_allclose(
        lowest.values[iaz, j], energy(v[np.argmin(h)]), rtol=2e-4, atol=1e-9
    )
    # the products are NaN where the lowest sweep does not reach and finite elsewhere
    assert np.isfinite(out.SHI.values).sum() > 1000


@compiled_only
def test_polar_volume_engines_agree_and_chunks_cover_the_grid(monkeypatch):
    tree = storm_volume(field=hail_storm_dbz, nray=180)
    monkeypatch.setattr(vilmod, "_CHUNK", 5000)  # several azimuth blocks
    field = xr.DataArray(
        np.linspace(2500.0, 4500.0, 180),
        dims="azimuth",
        coords={"azimuth": tree["sweep_0"].azimuth},
    )
    a = hail_products(
        tree, freezing_level=field, minus20_level=field + 3000.0, engine="numpy"
    )
    b = hail_products(
        tree,
        freezing_level=field,
        minus20_level=field + 3000.0,
        engine="compiled",
        n_threads=2,
    )
    xr.testing.assert_allclose(a, b, rtol=1e-6)
    assert (a.SHI.values[np.isfinite(a.SHI.values)] > 0).sum() > 100
    # the azimuthal dependence of the level reaches the output
    flat = hail_products(tree, freezing_level=3500.0, minus20_level=6500.0)
    assert not np.allclose(a.SHI.values, flat.SHI.values, equal_nan=True)


def test_polar_volume_uses_the_same_columns_as_vil():
    """Columns without data are the same ones in the hail and the VIL products."""
    tree = storm_volume(field=hail_storm_dbz, elevations=(0.5, 1.5, 3.0, 6.0))
    out = hail_products(tree, freezing_level=3000.0, minus20_level=6000.0)
    ref = vil(tree)
    assert out.SHI.shape == ref.VIL.shape
    np.testing.assert_array_equal(
        np.isfinite(out.SHI.values), np.isfinite(ref.VIL.values)
    )
    np.testing.assert_allclose(out.ground_range, ref.ground_range)
    # a volume with an explicit radar height and field name
    named = hail_products(
        tree, "DBZH", freezing_level=3000.0, minus20_level=6000.0, radar_height=0.0
    )
    np.testing.assert_allclose(
        named.SHI_WARNING_THRESHOLD.values[np.isfinite(named.SHI.values)],
        57.5 * 3.0 - 121.0,
    )


def test_volume_errors_and_kinds():
    tree = storm_volume(elevations=(0.5, 1.5, 3.0))
    with pytest.raises(KeyError):
        shi(tree, "nofield", freezing_level=3000.0, minus20_level=6000.0)
    with pytest.raises(TypeError):
        shi(np.zeros(3), freezing_level=3000.0, minus20_level=6000.0)
    with pytest.raises(ValueError, match="polar volume"):
        shi(tree, kind="grid", freezing_level=3000.0, minus20_level=6000.0)
    # the sweeps can be treated as separate cuts only when they differ in elevation
    close = storm_volume(elevations=(0.5, 0.52, 3.0))
    out = shi(close, freezing_level=3000.0, minus20_level=6000.0, tolerance=0.1)
    assert out.shape == (360, out.shape[1])


# --------------------------------------------------------------------------
# accessors, provenance, inputs
# --------------------------------------------------------------------------


def test_accessors():
    z = np.arange(0.0, 14001.0, 500.0)
    ds = column(np.full(z.size, 52.0), z)
    kw = {"freezing_level": H0, "minus20_level": HM20}
    xr.testing.assert_identical(ds.radarx.hail_products(**kw), hail_products(ds, **kw))
    xr.testing.assert_identical(ds.radarx.shi(**kw), shi(ds, **kw))
    xr.testing.assert_identical(ds.radarx.mesh(**kw), mesh(ds, **kw))
    xr.testing.assert_identical(ds.radarx.posh(**kw), posh(ds, **kw))
    xr.testing.assert_identical(
        ds.radarx.hail_kinetic_energy(), hail_kinetic_energy(ds)
    )
    tree = storm_volume(elevations=(0.5, 1.5, 3.0))
    xr.testing.assert_identical(
        tree.radarx.shi(**{"freezing_level": 3000.0, "minus20_level": 6000.0}),
        shi(tree, freezing_level=3000.0, minus20_level=6000.0),
    )


def test_provenance_and_the_input_is_not_changed():
    z = np.arange(0.0, 14001.0, 500.0)
    ds = column(np.full(z.size, 52.0), z)
    ds.z.attrs["units"] = "m"
    before = ds.copy(deep=True)
    out = hail_products(ds, freezing_level=H0, minus20_level=HM20)
    xr.testing.assert_identical(ds, before)
    assert ds.z.attrs == {"units": "m"}
    assert (
        "10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2"
        in out.attrs["radarx_references"]
    )
    assert out.SHI.attrs["radarx_method"].startswith("Hail detection algorithm")
    assert "hail_products(" in out.attrs["history"]
    for func in (shi, mesh, posh, hail_kinetic_energy):
        res = func(
            ds,
            **(
                {}
                if func is hail_kinetic_energy
                else {"freezing_level": H0, "minus20_level": HM20}
            ),
        )
        assert (
            "10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2"
            in res.attrs["radarx_references"]
        )
    # fields of the levels keep their attributes
    h0 = xr.DataArray(
        np.full((1, 1), H0), dims=("y", "x"), coords={"y": [0.0], "x": [0.0]}
    )
    h0.y.attrs["units"] = "m"
    shi(ds, freezing_level=h0, minus20_level=h0 + 3000.0)
    assert h0.y.attrs == {"units": "m"}


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------


def test_real_nexrad_volume():
    """KLBB (open-radar-data): plausible hail products on a real volume."""
    pytest.importorskip("open_radar_data")
    from open_radar_data import DATASETS

    from .test_dealias import nexrad_volume

    try:
        path = DATASETS.fetch("KLBB20160601_150025_V06")
    except Exception as err:  # pragma: no cover - network
        pytest.skip(f"sample data unavailable: {err}")
    tree = nexrad_volume(path)
    out = hail_products(tree, freezing_level=4200.0, minus20_level=7200.0)
    assert out.SHI.dims == ("azimuth", "range")
    s = out.SHI.values[np.isfinite(out.SHI.values)]
    assert s.size > 1000 and s.min() >= 0.0 and s.max() < 2000.0
    m = out.MESH.values[np.isfinite(out.MESH.values)]
    np.testing.assert_allclose(m, 2.54 * np.sqrt(s), rtol=1e-5)
    p = out.POSH.values[np.isfinite(out.POSH.values)]
    assert p.min() >= 0.0 and p.max() <= 100.0
    if hailmod.HAS_COMPILED_KERNEL:
        other = hail_products(
            tree, freezing_level=4200.0, minus20_level=7200.0, engine="numpy"
        )
        xr.testing.assert_allclose(out, other, rtol=1e-6)
