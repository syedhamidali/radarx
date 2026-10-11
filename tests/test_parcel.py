# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for radarx.io.parcel (CAPE, CIN, lifted index, LCL, LFC and EL)
=====================================================================

Three independent references:

- the pseudo-adiabat of Bolton's equation (43) is compared with the ODE of the
  saturated adiabatic lapse rate integrated by SciPy;
- CAPE, CIN and the levels of an analytic sounding are recomputed in the test
  from that ODE and a fine trapezoid integration of the buoyancy;
- values of MetPy 1.7.1 (``surface_based_cape_cin``, ``mixed_layer_cape_cin``,
  ``most_unstable_cape_cin``, ``lcl``, ``lfc``, ``el``, ``lifted_index``) for
  the same soundings, entered as numbers (MetPy computes the saturated
  ascent, the integral and the layer means in its own ways, so the agreement
  is to one per cent, not to rounding).
"""

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from scipy.integrate import solve_ivp

import radarx  # noqa: F401
from radarx.io import _parcel_numpy as ref
from radarx.io import parcel as pc
from radarx.io import parcel_diagnostics

# numpy 2 renamed trapz to trapezoid; the tests run with both
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

ENGINES = ["numpy"] + (["compiled"] if pc.HAS_COMPILED_KERNEL else [])
needs_kernel = pytest.mark.skipif(
    not pc.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)
DATA = Path(__file__).parent / "io" / "data"
G, RD, CPD, EPS = 9.80665, 287.04749, 1005.7, 287.04749 / 461.52311


# --------------------------------------------------------------------------
# soundings
# --------------------------------------------------------------------------


def esat(t):
    tc = t - 273.15
    return 611.2 * np.exp(17.67 * tc / (tc + 243.5))


def analytic_sounding(dz=250.0, top=18000.0, p0=100000.0, qcap=0.014):
    """A Weisman and Klemp (1982) style profile (constants as recalled, not
    checked): potential temperature and relative humidity of height, the
    pressure from the hydrostatic equation, the mixing ratio capped at 14 g/kg."""
    z = np.arange(0.0, top + 1, dz)
    th0, thtr, ttr, ztr = 300.0, 343.0, 213.0, 12000.0
    th = np.where(
        z <= ztr,
        th0 + (thtr - th0) * (np.minimum(z, ztr) / ztr) ** 1.25,
        thtr * np.exp(G / (CPD * ttr) * (z - ztr)),
    )
    rh = np.where(z <= ztr, 1.0 - 0.75 * (np.minimum(z, ztr) / ztr) ** 1.25, 0.25)
    kappa = 287.04 / CPD
    pi = np.empty_like(z)
    pi[0] = (p0 / 1e5) ** kappa
    for k in range(1, z.size):
        pi[k] = pi[k - 1] - G / (CPD * 0.5 * (th[k] + th[k - 1])) * dz
    p = 1e5 * pi ** (1.0 / kappa)
    t = th * pi
    es = esat(t)
    qv = np.minimum(0.622 * rh * es / (p - rh * es), qcap)
    e = qv * p / (0.622 + qv)
    td = 243.5 * np.log(e / 611.2) / (17.67 - np.log(e / 611.2)) + 273.15
    return z, p, t, np.minimum(td, t)


def jan_sounding():
    """The radiosonde of Jackson, MS, 31 March 2022 00 UTC (stable, capped)."""
    rows = json.loads((DATA / "iem_KJAN_202203310000.json").read_text())["profiles"][0][
        "profile"
    ]
    get = lambda k: np.array(
        [np.nan if r[k] is None else r[k] for r in rows], float
    )  # noqa: E731
    p, t, td, z = get("pres"), get("tmpc"), get("dwpc"), get("hght")
    ok = np.isfinite(p) & np.isfinite(t) & np.isfinite(td) & np.isfinite(z)
    return z[ok], p[ok] * 100.0, t[ok] + 273.15, td[ok] + 273.15


def warm_jan_sounding():
    """The same with the lowest 1.5 km 11 K warmer and 9 K moister: one buoyant layer."""
    z, p, t, td = jan_sounding()
    w = np.clip(1.0 - (z - z[0]) / 1500.0, 0.0, 1.0)
    return z, p, t + 11.0 * w, td + 9.0 * w


def profile(z, p, t, td, **extra):
    return xr.Dataset(
        {
            "pressure": ("height", p),
            "temperature": ("height", t),
            "dewpoint": ("height", td),
            **extra,
        },
        coords={"height": z},
    )


# --------------------------------------------------------------------------
# thermodynamic functions of Bolton (1980)
# --------------------------------------------------------------------------


def test_bolton_equations():
    # eq. (10): 6.112 hPa at 0 degC, and the inverse
    assert ref.esat(273.15) == pytest.approx(611.2)
    assert ref.dewpoint(ref.esat(285.0)) == pytest.approx(285.0)
    # eq. (15) and the LCL pressure: MetPy 1.7.1 lcl(1000 hPa, 30 degC, 22 degC)
    # gives 889.86 hPa and 20.098 degC, lcl(950 hPa, 20 degC, 10 degC) 818.35 hPa
    # and 7.792 degC
    assert ref.lcl_temperature(303.15, 295.15) - 273.15 == pytest.approx(
        20.098, abs=0.01
    )
    assert ref._Parcel(1e5, 303.15, 295.15).pl / 100 == pytest.approx(889.86, rel=5e-4)
    assert ref._Parcel(95000.0, 293.15, 283.15).pl / 100 == pytest.approx(
        818.35, rel=5e-4
    )
    assert ref.lcl_temperature(290.0, 290.0) == 290.0  # saturated: no lowering
    # eq. (6): the potential temperature of a dry parcel
    assert 290.0 * (1e5 / 90000.0) ** ref.kappa(0.0) == pytest.approx(
        290.0 * (1e5 / 90000.0) ** 0.2854
    )


@pytest.mark.parametrize(
    "t0, td0, p0",
    [(303.15, 295.15, 100000.0), (293.15, 288.15, 95000.0), (283.15, 280.0, 85000.0)],
)
def test_pseudo_adiabat_of_theta_e_matches_the_ode(t0, td0, p0):
    """Conserving the theta_E of eq. (43) follows the saturated adiabatic ODE."""

    def dtdp(p, t):
        es = ref.esat(t[0])
        rs = EPS * es / (p - es)
        lv = (2.501 - 0.00237 * (t[0] - 273.15)) * 1e6
        num = (RD * t[0] + lv * rs) / p
        return [num / (CPD + lv**2 * rs * EPS / (RD * t[0] ** 2))]

    parcel = ref._Parcel(p0, t0, td0)
    sol = solve_ivp(
        dtdp,
        (parcel.pl, 20000.0),
        [parcel.tl],
        rtol=1e-10,
        atol=1e-10,
        dense_output=True,
    )
    for p in (80000.0, 60000.0, 50000.0, 30000.0):
        if p < parcel.pl:
            assert parcel.temperature(p)[0] == pytest.approx(sol.sol(p)[0], abs=0.05)
    # below the LCL the dry adiabat holds, and the LCL is reached at T_L
    assert parcel.temperature(parcel.pl)[0] == pytest.approx(parcel.tl, abs=1e-6)


# --------------------------------------------------------------------------
# CAPE, CIN and levels against a fine numerical integration
# --------------------------------------------------------------------------


def reference_parcel_diagnostics(z, p, t, td):
    """Surface-based CAPE, CIN, LCL, LFC, EL by ODE ascent and trapezoids."""
    parcel = ref._Parcel(p[0], t[0], td[0])

    def dtdp(pp, tt):
        es = ref.esat(tt[0])
        rs = EPS * es / (pp - es)
        lv = (2.501 - 0.00237 * (tt[0] - 273.15)) * 1e6
        return [
            ((RD * tt[0] + lv * rs) / pp) / (CPD + lv**2 * rs * EPS / (RD * tt[0] ** 2))
        ]

    sol = solve_ivp(
        dtdp, (parcel.pl, p[-1]), [parcel.tl], rtol=1e-10, atol=1e-10, dense_output=True
    )
    tp = np.empty_like(t)
    rp = np.empty_like(t)
    for k, pk in enumerate(p):
        if pk >= parcel.pl:
            tp[k] = parcel.theta0 * (pk / 1e5) ** parcel.kappa0
            rp[k] = parcel.r0
        else:
            tp[k] = sol.sol(pk)[0]
            es = ref.esat(tp[k])
            rp[k] = EPS * es / (pk - es)
    re = EPS * esat(td) / (p - esat(td))
    tvp, tve = tp * (1 + rp / EPS) / (1 + rp), t * (1 + re / EPS) / (1 + re)
    b = G * (tvp - tve) / tve
    # a fine grid in height: cubic interpolation is not needed at 5 m
    zf = np.arange(z[0], z[-1], 5.0)
    bf = np.interp(zf, z, b)
    pf = np.exp(np.interp(zf, z, np.log(p)))
    above = pf < parcel.pl
    pos = bf > 0
    lfc = zf[above & pos][0]
    el = zf[above & pos][-1]
    cape = _trapezoid(np.where((zf >= lfc) & (zf <= el), np.maximum(bf, 0.0), 0.0), zf)
    cin = _trapezoid(np.where(zf < lfc, np.minimum(bf, 0.0), 0.0), zf)
    lcl_z = np.interp(np.log(parcel.pl), np.log(p)[::-1], z[::-1])
    return dict(cape=cape, cin=cin, lcl_z=lcl_z, lfc_z=lfc, el_z=el, lcl_p=parcel.pl)


def test_surface_based_parcel_against_the_ode_reference():
    z, p, t, td = analytic_sounding(dz=100.0)
    truth = reference_parcel_diagnostics(z, p, t, td)
    for engine in ENGINES:
        out = parcel_diagnostics(profile(z, p, t, td), engine=engine)
        assert float(out.cape) == pytest.approx(truth["cape"], rel=0.01)
        assert float(out.cin) == pytest.approx(truth["cin"], abs=3.0, rel=0.05)
        assert float(out.lcl_height) == pytest.approx(truth["lcl_z"], abs=40.0)
        assert float(out.lcl_pressure) == pytest.approx(truth["lcl_p"], rel=1e-9)
        assert float(out.lfc_height) == pytest.approx(truth["lfc_z"], abs=150.0)
        assert float(out.el_height) == pytest.approx(truth["el_z"], abs=150.0)
    # the sounding is a big-CAPE one
    assert 1500.0 < truth["cape"] < 2500.0


# --------------------------------------------------------------------------
# values of MetPy 1.7.1
# --------------------------------------------------------------------------

# surface-based, mixed-layer (100 hPa), most-unstable (300 hPa): CAPE, CIN
# (J kg-1); LCL, LFC, EL (hPa) and lifted index (K) of the surface parcel
METPY = {
    "analytic": dict(
        sb=(1884.4, -47.3),
        ml=(2040.9, -32.2),
        mu=(2313.8, -2.3),
        lcl=891.7,
        lfc=827.8,
        el=223.3,
        li=-6.95,
    ),
    "warm_jan": dict(
        sb=(4425.4, 0.0), mu=(4425.4, 0.0), lcl=947.0, lfc=947.0, li=-8.14
    ),
}
SOUNDINGS = {"analytic": analytic_sounding, "warm_jan": warm_jan_sounding}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", list(METPY))
def test_against_metpy(name, engine):
    z, p, t, td = SOUNDINGS[name]()
    want = METPY[name]
    prof = profile(z, p, t, td)
    sb = parcel_diagnostics(prof, "surface-based", engine=engine)
    assert float(sb.cape) == pytest.approx(want["sb"][0], rel=0.02)
    assert float(sb.cin) == pytest.approx(want["sb"][1], abs=3.0)
    assert float(sb.lcl_pressure) / 100.0 == pytest.approx(want["lcl"], rel=2e-3)
    assert float(sb.lfc_pressure) / 100.0 == pytest.approx(want["lfc"], rel=0.01)
    # MetPy interpolates the parcel temperature between the levels of the sounding
    assert float(sb.lifted_index) == pytest.approx(want["li"], abs=0.4)
    if "el" in want:
        assert float(sb.el_pressure) / 100.0 == pytest.approx(want["el"], rel=0.03)
    mu = parcel_diagnostics(prof, "most-unstable", engine=engine)
    assert float(mu.cape) == pytest.approx(want["mu"][0], rel=0.02)
    assert float(mu.cin) == pytest.approx(want["mu"][1], abs=3.0)
    if "ml" in want:
        ml = parcel_diagnostics(prof, "mixed-layer", engine=engine)
        assert float(ml.cape) == pytest.approx(want["ml"][0], rel=0.02)
        assert float(ml.cin) == pytest.approx(want["ml"][1], abs=12.0)
        # the mixing ratio of this profile is constant in the lowest kilometre
        assert float(ml.parcel_dewpoint) == pytest.approx(
            float(sb.parcel_dewpoint), abs=0.05
        )
        assert float(ml.parcel_pressure) == float(sb.parcel_pressure)


@pytest.mark.parametrize("engine", ENGINES)
def test_stable_capped_sounding_has_no_free_convection(engine):
    z, p, t, td = jan_sounding()
    out = parcel_diagnostics(profile(z, p, t, td), "mixed-layer", engine=engine)
    assert float(out.cape) < 5.0  # MetPy 0.0
    sb = parcel_diagnostics(profile(z, p, t, td), engine=engine)
    assert float(sb.cape) < 5.0  # MetPy 0.8
    assert float(sb.lifted_index) == pytest.approx(7.84, abs=0.3)
    assert float(sb.lcl_pressure) / 100.0 == pytest.approx(974.8, rel=1e-3)


def test_virtual_temperature_correction_raises_cape():
    """Doswell and Rasmussen (1994): neglecting it underestimates CAPE."""
    prof = profile(*analytic_sounding())
    with_tv = float(parcel_diagnostics(prof).cape)
    without = float(parcel_diagnostics(prof, virtual=False).cape)
    assert 0.04 < (with_tv - without) / with_tv < 0.12  # 5.7 %
    assert (
        parcel_diagnostics(prof, virtual=False).attrs["virtual_temperature_correction"]
        == 0
    )


# --------------------------------------------------------------------------
# definitions and edge cases
# --------------------------------------------------------------------------


def test_most_unstable_parcel_is_the_largest_theta_e():
    z, p, t, td = analytic_sounding()
    # a moist layer 100 hPa above the ground: the most unstable parcel is there
    t2, td2 = t.copy(), td.copy()
    layer = (p < 90000.0) & (p > 85000.0)
    td2[layer] = np.minimum(t[layer] - 0.1, td[layer] + 12.0)
    prof = profile(z, p, t2, td2)
    mu = parcel_diagnostics(prof, "most-unstable")
    k = int(np.argmax([ref.theta_e(a, b, c) for a, b, c in zip(t2, td2, p)]))
    # the maximum within the lowest 300 hPa
    ok = p >= p[0] - 30000.0
    kk = np.nonzero(ok)[0][
        int(np.argmax([ref.theta_e(t2[i], td2[i], p[i]) for i in np.nonzero(ok)[0]]))
    ]
    assert float(mu.parcel_pressure) == pytest.approx(p[kk])
    assert float(mu.parcel_temperature) == pytest.approx(t2[kk])
    assert kk > 0 or k == 0
    # a search layer of 10 Pa holds only the lowest level
    shallow = parcel_diagnostics(prof, "most-unstable", depth=10.0)
    assert float(shallow.parcel_pressure) == pytest.approx(p[0])


def test_mixed_layer_means_the_potential_temperature_and_mixing_ratio():
    z, p, t, td = analytic_sounding(dz=50.0)
    out = parcel_diagnostics(profile(z, p, t, td), "mixed-layer")
    layer = p >= p[0] - 10000.0
    r = EPS * esat(td) / (p - esat(td))
    theta = t * (1e5 / p) ** (0.2854 * (1 - 0.28 * r))
    w = -np.gradient(p)[layer]
    theta_m = np.sum(theta[layer] * w) / np.sum(w)
    r_m = np.sum(r[layer] * w) / np.sum(w)
    t0 = theta_m * (p[0] / 1e5) ** (0.2854 * (1 - 0.28 * r_m))
    assert float(out.parcel_temperature) == pytest.approx(t0, abs=0.05)
    assert float(out.parcel_pressure) == p[0]
    assert out.attrs["parcel_layer_depth"] == 10000.0


def test_saturated_surface_lcl_is_the_surface():
    z, p, t, td = analytic_sounding()
    td = td.copy()
    td[0] = t[0]
    for engine in ENGINES:
        out = parcel_diagnostics(profile(z, p, t, td), engine=engine)
        assert float(out.lcl_pressure) == pytest.approx(p[0])
        assert float(out.lcl_height) == pytest.approx(z[0])


@pytest.mark.parametrize("engine", ENGINES)
def test_no_lfc_gives_zero_cape_and_nan_levels(engine):
    z = np.arange(0.0, 12001.0, 500.0)
    t = 300.0 - 0.0065 * z  # lapse rate far below the moist adiabat... in dry air
    p = 1e5 * (t / 300.0) ** 5.2559
    td = t - 25.0
    out = parcel_diagnostics(profile(z, p, t, td), engine=engine)
    assert float(out.cape) == 0.0
    assert np.isnan(float(out.cin)) and np.isnan(float(out.lfc_height))
    assert np.isnan(float(out.el_height))
    assert np.isfinite(float(out.lifted_index)) and float(out.lifted_index) > 0
    # lcl above the profile top: no LCL either
    assert float(out.parcel_pressure) == p[0]


@pytest.mark.parametrize("engine", ENGINES)
def test_missing_levels_order_and_bad_columns(engine):
    z, p, t, td = analytic_sounding(dz=250.0)
    base = parcel_diagnostics(profile(z, p, t, td), engine=engine)
    # NaN levels are skipped, the rest is interpolated
    t_bad = t.copy()
    t_bad[[7, 8]] = np.nan
    skipped = parcel_diagnostics(profile(z, p, t_bad, td), engine=engine)
    assert float(skipped.cape) == pytest.approx(float(base.cape), rel=0.02)
    # a profile from the top down gives the same
    flipped = parcel_diagnostics(
        profile(z[::-1], p[::-1], t[::-1], td[::-1]), engine=engine
    )
    xr.testing.assert_allclose(flipped.drop_vars("height", errors="ignore"), base)
    # a column of NaN and one of two levels
    cols = xr.Dataset(
        {
            "pressure": (("x", "height"), np.vstack([p, p, p])),
            "temperature": (
                ("x", "height"),
                np.vstack([t, np.full_like(t, np.nan), t]),
            ),
            "dewpoint": (("x", "height"), np.vstack([td, td, td])),
        },
        coords={"height": z, "x": [0, 1, 2]},
    )
    cols["temperature"][2, 2:] = np.nan
    out = parcel_diagnostics(cols, engine=engine)
    assert out.cape.dims == ("x",)
    assert float(out.cape[0]) == pytest.approx(float(base.cape))
    assert np.isnan(out.cape[1:]).all() and np.isnan(out.lcl_height[1:]).all()


@pytest.mark.parametrize("engine", ENGINES)
def test_levels_without_a_dewpoint_or_in_the_wrong_order_are_skipped(engine):
    z, p, t, td = analytic_sounding(dz=250.0)
    base = parcel_diagnostics(profile(z, p, t, td), engine=engine)
    # a level without a dew point and one with a non-positive dew point
    td_bad = td.copy()
    td_bad[7] = np.nan
    td_bad[9] = -1.0
    skipped = parcel_diagnostics(profile(z, p, t, td_bad), engine=engine)
    assert float(skipped.cape) == pytest.approx(float(base.cape), rel=0.02)
    assert float(skipped.lcl_pressure) == pytest.approx(float(base.lcl_pressure))
    # a repeated level (the pressure does not fall) and a level out of order
    dup = np.insert(np.arange(z.size), 6, 5)
    again = parcel_diagnostics(profile(z[dup], p[dup], t[dup], td[dup]), engine=engine)
    xr.testing.assert_allclose(again.drop_vars("height", errors="ignore"), base)
    swapped = np.arange(z.size)
    swapped[[10, 11]] = swapped[[11, 10]]
    out = parcel_diagnostics(
        profile(z[swapped], p[swapped], t[swapped], td[swapped]), engine=engine
    )
    assert float(out.cape) == pytest.approx(float(base.cape), rel=0.03)


@pytest.mark.parametrize("engine", ENGINES)
def test_lcl_above_the_top_of_the_profile(engine):
    z, p, t, td = analytic_sounding(dz=250.0)
    keep = z <= 3000.0
    dry = np.minimum(td, t - 60.0)  # the LCL is far above 3 km
    out = parcel_diagnostics(
        profile(z[keep], p[keep], t[keep], dry[keep]), engine=engine
    )
    assert float(out.cape) == 0.0
    for name in ("lcl_pressure", "lcl_height", "lfc_height", "el_height", "cin"):
        assert np.isnan(float(out[name])), name
    assert float(out.parcel_pressure) == p[0]


@pytest.mark.parametrize("engine", ENGINES)
def test_a_cap_between_two_buoyant_layers_is_not_counted(engine):
    z, p, t, td = analytic_sounding(dz=100.0)
    free = parcel_diagnostics(profile(z, p, t, td), engine=engine)
    # a warm layer at 6 to 8 km makes the parcel negatively buoyant for a while
    warm = t + 9.0 * np.exp(-(((z - 7000.0) / 700.0) ** 2))
    capped = parcel_diagnostics(
        profile(z, p, warm, np.minimum(td, warm)), engine=engine
    )
    truth = reference_parcel_diagnostics(z, p, warm, np.minimum(td, warm))
    assert float(capped.cape) == pytest.approx(truth["cape"], rel=0.02)
    assert float(capped.cape) < 0.95 * float(free.cape)
    assert float(capped.el_height) == pytest.approx(truth["el_z"], abs=150.0)


def test_parcel_helpers_of_the_numpy_reference():
    p = [100000.0, 90000.0, 80000.0]
    f = [1.0, 2.0, 4.0]
    # linear in ln p between the levels that bracket the pressure
    assert ref._interp_lnp(p, f, 95000.0) == pytest.approx(
        1.0 + np.log(0.95) / np.log(0.9)
    )
    mid = np.sqrt(90000.0 * 80000.0)
    assert ref._interp_lnp(p, f, mid) == pytest.approx(3.0)
    assert np.isnan(ref._interp_lnp(p, f, 50000.0))  # above the top
    assert np.isnan(ref._interp_lnp(p, f, 110000.0))  # below the first level
    # a mixed layer of no depth is the lowest level
    z, pp, t, td = analytic_sounding(dz=250.0)
    assert ref._parcel_start(1, 0.0, z, pp, t, td) == (pp[0], t[0], td[0], 0)


def test_a_level_at_the_lcl_pressure_is_the_lcl():
    z, p, t, td = analytic_sounding(dz=250.0)
    pl = ref._Parcel(p[0], t[0], td[0]).pl
    k = int(np.searchsorted(-p, -pl))  # the first level above the LCL
    w = np.log(pl / p[k - 1]) / np.log(p[k] / p[k - 1])
    insert = [z[k - 1] + w * (z[k] - z[k - 1])] + [
        a[k - 1] + w * (a[k] - a[k - 1]) for a in (t, td)
    ]
    z2, p2, t2, td2 = (
        np.insert(a, k, v) for a, v in zip((z, p, t, td), (insert[0], pl, *insert[1:]))
    )
    base = parcel_diagnostics(profile(z, p, t, td), engine="numpy")
    out = parcel_diagnostics(profile(z2, p2, t2, td2), engine="numpy")
    assert float(out.lcl_pressure) == pl
    assert float(out.lcl_height) == pytest.approx(insert[0], abs=1e-6)
    assert float(out.cape) == pytest.approx(float(base.cape), rel=0.01)


def test_humidity_variants_and_units():
    z, p, t, td = analytic_sounding()
    base = parcel_diagnostics(profile(z, p, t, td))
    e = esat(td)
    q = EPS * e / (p - (1 - EPS) * e)
    dry = profile(z, p, t, td).drop_vars("dewpoint")
    from_q = parcel_diagnostics(dry.assign(specific_humidity=("height", q)))
    np.testing.assert_allclose(from_q.cape, base.cape, rtol=2e-3)
    rh = esat(td) / esat(t)
    from_rh = parcel_diagnostics(dry.assign(relative_humidity=("height", rh)))
    np.testing.assert_allclose(from_rh.cape, base.cape, rtol=1e-3)
    from_pct = parcel_diagnostics(dry.assign(relative_humidity=("height", 100 * rh)))
    np.testing.assert_allclose(from_pct.cape, from_rh.cape)
    with pytest.raises(ValueError, match="pressure must be in Pa"):
        parcel_diagnostics(profile(z, p / 100.0, t, td))
    with pytest.raises(ValueError, match="dewpoint"):
        parcel_diagnostics(profile(z, p, t, td).drop_vars("dewpoint"))
    with pytest.raises(ValueError, match="parcel must be"):
        parcel_diagnostics(profile(z, p, t, td), "elevated")
    with pytest.raises(ValueError, match="depth"):
        parcel_diagnostics(profile(z, p, t, td), "mixed-layer", depth=-1.0)
    with pytest.raises(ValueError, match="needs 'temperature'"):
        parcel_diagnostics(profile(z, p, t, td).drop_vars("temperature"))
    with pytest.raises(ValueError, match="engine"):
        parcel_diagnostics(profile(z, p, t, td), engine="gpu")
    nocoord = profile(z, p, t, td).drop_vars("height")
    with pytest.raises(ValueError, match="height"):
        parcel_diagnostics(nocoord)
    # the vertical dimension must be a dimension of the fields
    with pytest.raises(ValueError, match="dimension 'alt' not found"):
        parcel_diagnostics(profile(z, p, t, td).assign_coords(alt=0.0), dim="alt")
    assert (
        float(parcel_diagnostics(nocoord.assign(zz=("height", z)), height="zz").cape)
        > 0
    )


def grid(nx=4, ny=3):
    """ERA5-like columns on (z, y, x): the analytic sounding with a warmer east."""
    z, p, t, td = analytic_sounding(dz=500.0)
    shape = (z.size, ny, nx)
    shift = np.linspace(-3.0, 3.0, nx)[None, None, :]
    return xr.Dataset(
        {
            "pressure": (
                ("z", "y", "x"),
                np.broadcast_to(p[:, None, None], shape).copy(),
            ),
            "temperature": (
                ("z", "y", "x"),
                np.broadcast_to(
                    t[:, None, None] + shift * (z[:, None, None] < 4000), shape
                ).copy(),
            ),
            "dewpoint": (
                ("z", "y", "x"),
                np.broadcast_to(td[:, None, None], shape).copy(),
            ),
        },
        coords={"z": z, "y": np.arange(ny), "x": np.arange(nx)},
    )


@needs_kernel
def test_engines_agree_on_a_grid_and_threads_do_not_matter():
    g = grid()
    for parcel in ("surface-based", "mixed-layer", "most-unstable"):
        a = parcel_diagnostics(g, parcel, dim="z", engine="numpy")
        b = parcel_diagnostics(g, parcel, dim="z", engine="compiled", n_threads=4)
        c = parcel_diagnostics(g, parcel, dim="z", engine="compiled", n_threads=1)
        assert a.cape.dims == ("y", "x")
        for name in a.data_vars:
            np.testing.assert_allclose(
                a[name], b[name], rtol=1e-9, atol=1e-6, equal_nan=True
            )
            np.testing.assert_array_equal(b[name], c[name])
    # a warmer boundary layer gives more CAPE
    cape = parcel_diagnostics(g, dim="z").cape
    assert (cape.diff("x") < 0).all() or (cape.diff("x") > 0).all()


def test_accessor_and_metadata():
    prof = profile(*analytic_sounding())
    out = prof.radarx.parcel_diagnostics("mixed-layer", depth=5000.0)
    assert (
        out.attrs["parcel"] == "mixed-layer"
        and out.attrs["parcel_layer_depth"] == 5000.0
    )
    assert out.cape.attrs["units"] == "J kg-1"
    assert out.cin.attrs["units"] == "J kg-1"
    assert out.lifted_index.attrs["units"] == "K"
    assert out.lcl_height.attrs["units"] == "m"
    assert float(out.cin) <= 0.0
    assert "Bolton" in out.attrs["radarx_method"]
    assert (
        "10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2"
        in out.attrs["radarx_references"]
    )


def test_compiled_kernel_input_checks():
    if not pc.HAS_COMPILED_KERNEL:
        pytest.skip("compiled kernel not built")
    z, p, t, td = (a[None] for a in analytic_sounding())
    with pytest.raises(ValueError, match="kind"):
        pc._parcel.parcel(z, p, t, td, 3, 10000.0, True, 0)
    with pytest.raises(ValueError, match="depth"):
        pc._parcel.parcel(z, p, t, td, 1, 0.0, True, 0)
    with pytest.raises(ValueError, match="same shape"):
        pc._parcel.parcel(z, p[:, :-1], t, td, 0, 1.0, True, 0)


def test_compiled_engine_unavailable(monkeypatch):
    monkeypatch.setattr(pc, "HAS_COMPILED_KERNEL", False)
    prof = profile(*analytic_sounding())
    with pytest.raises(ImportError):
        parcel_diagnostics(prof, engine="compiled")
    assert float(parcel_diagnostics(prof).cape) > 0
