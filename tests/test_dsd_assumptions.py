#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Scattering assumptions of the drop size distribution tables
===========================================================

The tables record their canting width and frequencies, and the documented
sensitivity of ZDR and KDP to both (section "Sensitivity to canting and
frequency" of :mod:`radarx.retrieve.dsd`) is reproduced with the T-matrix code.
"""
import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from scipy.integrate import trapezoid
from scipy.special import gamma as gamma_function

import radarx  # noqa: F401
from radarx.fundamentals import tmatrix
from radarx.retrieve import (
    dsd,
    dsd_bayesian,
    forward_grid,
    radar_from_dsd,
    scattering_table,
)

dsdmod = importlib.import_module("radarx.retrieve.dsd")

needs_kernel = pytest.mark.skipif(
    not tmatrix.HAS_COMPILED_KERNEL, reason="compiled T-matrix kernel not built"
)
FREQUENCY = {"S": 2.8, "C": 5.6, "X": 9.4}
LIGHT = 299.792458  # mm GHz


@pytest.mark.parametrize("band", ["S", "C", "X"])
def test_table_records_canting_and_frequency(band):
    tab = scattering_table(band, 20.0)
    assert tab.attrs["canting_std"] == 7.0
    assert tab.attrs["canting_std_units"] == "degree"
    assert tab.attrs["canting"] == "Gaussian, 0 mean, 7 degrees standard deviation"
    assert tab.attrs["frequency_ghz"] == FREQUENCY[band]
    assert tab.attrs["elevation"] == 0.0
    assert tab.attrs["elevation_units"] == "degree"
    # the recorded frequency is the one of the wavelength column of the table
    np.testing.assert_allclose(
        tab.attrs["wavelength"], LIGHT / tab.attrs["frequency_ghz"], rtol=1e-5
    )


def test_assumptions_are_read_from_the_header_of_the_file():
    line = (
        "# canting_std_deg=10; frequency_ghz_S=2.75; frequency_ghz_C=5.6; "
        "frequency_ghz_X=9.4; elevation_deg=0; axis_ratio=Brandes et al. (2002)"
    )
    got = dsdmod._parse_assumptions(line)
    assert got == {
        "canting_std": 10.0,
        "frequency_ghz": {"S": 2.75, "C": 5.6, "X": 9.4},
        "elevation": 0.0,
    }
    assert dsdmod._table_assumptions() == {
        "canting_std": 7.0,
        "frequency_ghz": FREQUENCY,
        "elevation": 0.0,
    }
    with pytest.raises(KeyError):
        dsdmod._parse_assumptions("# canting_std_deg=7")


def _gamma_nd(diameter, dm, mu, nw=8000.0):
    f = 6.0 / 256.0 * (4 + mu) ** (mu + 4) / gamma_function(mu + 4)
    return nw * f * (diameter / dm) ** mu * np.exp(-(4 + mu) * diameter / dm)


def test_products_carry_the_assumptions():
    tab = scattering_table("C", 10.0)
    nd = xr.DataArray(
        np.stack([_gamma_nd(tab.diameter.values, dm, 3.0) for dm in (1.2, 2.0)]),
        dims=("gate", "diameter"),
        coords={"diameter": tab.diameter.values},
    )
    sim = radar_from_dsd(nd, "C", 10.0)
    assert sim.attrs["canting_std"] == 7.0
    assert sim.attrs["frequency_ghz"] == 5.6
    radar = sim[["DBZH", "ZDR", "KDP"]]
    out = dsd(radar, band="C", temperature=10.0, engine="numpy")
    assert out.attrs["canting_std"] == 7.0
    assert out.attrs["frequency_ghz"] == 5.6
    tree = dsd(
        xr.DataTree.from_dict({"/": xr.Dataset(), "sweep_0": radar}),
        band="C",
        temperature=10.0,
        engine="numpy",
    )
    assert tree["sweep_0"].ds.attrs["frequency_ghz"] == 5.6
    assert forward_grid("X").attrs["canting_std"] == 7.0
    assert forward_grid("X").attrs["frequency_ghz"] == 9.4
    bayes = dsd_bayesian(radar, band="C", temperature=10.0, engine="numpy")
    assert bayes.attrs["canting_std"] == 7.0
    assert bayes.attrs["frequency_ghz"] == 5.6


@needs_kernel
@pytest.mark.parametrize("band", ["S", "C", "X"])
def test_table_is_the_tmatrix_result_of_the_recorded_assumptions(band):
    """The recorded canting width and frequency are what produced the numbers."""
    tab = scattering_table(band, 20.0)
    d = np.array([1.0, 2.5, 4.0, 5.5, 8.0])
    out = tmatrix.raindrop_scattering(
        d,
        LIGHT / tab.attrs["frequency_ghz"],
        temperature=20.0,
        canting_std=tab.attrs["canting_std"],
        ndgs=4,
        ddelt=1e-3,
    )
    for name in ("sigma_h", "sigma_v", "kdp"):
        np.testing.assert_allclose(
            out[name].values, tab[name].sel(diameter=d).values, rtol=2e-5
        )


@needs_kernel
@pytest.mark.parametrize("band", ["S", "C", "X"])
def test_kdp_per_drop_scales_with_canting_at_every_size(band):
    # up to 4 mm: at C band the KDP of drops of 6 mm changes sign (resonance)
    d = np.arange(0.5, 4.01, 0.5)
    wl = LIGHT / FREQUENCY[band]
    base = tmatrix.raindrop_scattering(d, wl, canting_std=7.0, ddelt=1e-3).kdp.values
    for std, factor in ((10.0, 0.955), (20.0, 0.731)):
        kdp = tmatrix.raindrop_scattering(d, wl, canting_std=std, ddelt=1e-3).kdp.values
        np.testing.assert_allclose(kdp / base, factor, atol=0.005)


@needs_kernel
def test_single_drop_zdr_change_with_canting():
    """Numbers of the sensitivity section: S band, 20 degrees Celsius."""
    d = np.array([2.0, 4.0, 6.0, 8.0])
    wl = LIGHT / 2.8

    def zdr(std):
        out = tmatrix.raindrop_scattering(d, wl, canting_std=std, ddelt=1e-3)
        return 10 * np.log10(out.sigma_h.values / out.sigma_v.values)

    base = zdr(7.0)
    np.testing.assert_allclose(base, [0.62, 2.29, 3.98, 5.00], atol=0.006)
    np.testing.assert_allclose(
        zdr(10.0) - base, [-0.03, -0.11, -0.21, -0.29], atol=0.006
    )
    np.testing.assert_allclose(
        zdr(20.0) - base, [-0.17, -0.65, -1.20, -1.66], atol=0.006
    )


def _dsd_radar(std, ghz, mu, dm):
    tab = scattering_table("S", 20.0)
    d = tab.diameter.values
    out = tmatrix.raindrop_scattering(
        d, LIGHT / ghz, canting_std=std, ndgs=4, ddelt=1e-3
    )
    nd = _gamma_nd(d, dm, mu)
    k = (LIGHT / ghz) ** 4 / (np.pi**5 * dsdmod.KW2)
    zh = k * trapezoid(nd * out.sigma_h.values, d)
    zv = k * trapezoid(nd * out.sigma_v.values, d)
    return 10 * np.log10(zh), 10 * np.log10(zh / zv), trapezoid(nd * out.kdp.values, d)


@needs_kernel
@pytest.mark.parametrize(
    "mu, dm, dzdr10, dzdr20, dz20",
    [
        (3.0, 1.5, -0.03, -0.20, -0.06),
        (3.0, 2.0, -0.06, -0.35, -0.10),
        (3.0, 3.0, -0.11, -0.67, -0.17),
        (-1.0, 2.0, -0.10, -0.58, -0.15),
        (-1.0, 3.0, -0.14, -0.86, -0.18),
    ],
)
def test_dsd_zdr_change_with_canting(mu, dm, dzdr10, dzdr20, dz20):
    z7, zdr7, k7 = _dsd_radar(7.0, 2.8, mu, dm)
    _, zdr10, k10 = _dsd_radar(10.0, 2.8, mu, dm)
    z20, zdr20, k20 = _dsd_radar(20.0, 2.8, mu, dm)
    assert zdr10 - zdr7 == pytest.approx(dzdr10, abs=0.011)
    assert zdr20 - zdr7 == pytest.approx(dzdr20, abs=0.011)
    assert z20 - z7 == pytest.approx(dz20, abs=0.011)
    assert k10 / k7 - 1 == pytest.approx(-0.045, abs=0.003)
    assert k20 / k7 - 1 == pytest.approx(-0.269, abs=0.003)
    # the table of the packaged file is the 7 degree case
    nd = xr.DataArray(
        _gamma_nd(scattering_table("S").diameter.values, dm, mu),
        dims="diameter",
        coords={"diameter": scattering_table("S").diameter.values},
    )
    sim = radar_from_dsd(nd, "S", 20.0)
    assert float(sim.ZDR) == pytest.approx(zdr7, abs=6e-3)
    assert float(sim.KDP) == pytest.approx(k7, rel=2e-3)


@needs_kernel
def test_frequency_changes_kdp_but_hardly_zdr_at_s_band():
    for mu, dm in ((3.0, 2.0), (-1.0, 3.0)):
        z0, zdr0, k0 = _dsd_radar(7.0, 2.8, mu, dm)
        for ghz, kdp_change in ((2.7, -0.04), (2.75, -0.02), (3.0, 0.08)):
            z, zdr, k = _dsd_radar(7.0, ghz, mu, dm)
            assert abs(zdr - zdr0) < 0.06
            assert abs(z - z0) < 0.18
            assert k / k0 - 1 == pytest.approx(kdp_change, abs=0.015)
        assert abs(_dsd_radar(7.0, 2.75, mu, dm)[1] - zdr0) < 0.012


@needs_kernel
def test_a_table_of_7_degrees_misreads_more_canted_rain():
    """Dm retrieved with the table of 7 degrees from rain canted by 10 and 20 degrees."""
    grid = np.linspace(0.5, 4.0, 141)
    for mu, bounds10, bounds20 in (
        (3.0, (-0.04, -0.019), (-0.215, -0.115)),
        (-1.0, (-0.075, -0.025), (-0.325, -0.165)),
    ):
        zdr7 = np.array([_dsd_radar(7.0, 2.8, mu, x)[1] for x in grid])
        for std, bounds in ((10.0, bounds10), (20.0, bounds20)):
            for dm in (1.5, 2.0, 3.0):
                zdr = _dsd_radar(std, 2.8, mu, dm)[1]
                error = np.interp(zdr, zdr7, grid) / dm - 1
                assert bounds[0] < error < bounds[1]


def _builder():
    path = Path(__file__).resolve().parents[1] / "ci" / "build_dsd_tables.py"
    if not path.exists():
        pytest.skip("ci/build_dsd_tables.py is not part of the installed package")
    spec = importlib.util.spec_from_file_location("build_dsd_tables", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@needs_kernel
def test_table_builder_writes_the_assumptions_and_reproduces_the_file(tmp_path):
    builder = _builder()
    assert builder.FREQUENCIES_GHZ == FREQUENCY
    out = tmp_path / "dsd_scattering.csv"
    builder.main(out)
    packaged = Path(dsdmod.__file__).parent / "data" / "dsd_scattering.csv"
    new = out.read_text().splitlines()
    old = packaged.read_text().splitlines()
    assert new[1] == old[1]  # the line that records canting and frequencies
    assert dsdmod._parse_assumptions(new[1])["canting_std"] == 7.0
    a = np.loadtxt(out, delimiter=",", comments="#", skiprows=5, usecols=range(1, 11))
    b = np.loadtxt(
        packaged, delimiter=",", comments="#", skiprows=5, usecols=range(1, 11)
    )
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-8)


@needs_kernel
def test_table_builder_with_another_canting_width(tmp_path, monkeypatch):
    builder = _builder()
    monkeypatch.setattr(builder, "TEMPERATURES", (20.0,))
    monkeypatch.setattr(builder, "DIAMETERS", np.array([2.0, 4.0]))
    out = tmp_path / "t10.csv"
    builder.main(out, 10)
    lines = out.read_text().splitlines()
    assert dsdmod._parse_assumptions(lines[1])["canting_std"] == 10.0
    rows = [line.split(",") for line in lines[5:]]
    # S band, 4 mm: KDP is 4.5 % lower than with 7 degrees
    s4 = next(r for r in rows if r[0] == "S" and float(r[3]) == 4.0)
    ratio = float(s4[8]) / float(scattering_table("S").kdp.sel(diameter=4.0))
    assert ratio == pytest.approx(0.955, abs=0.003)
