#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for ``echo_mask(method="gourley2007")``
=============================================

Gourley, J. J., P. Tabary, and J. Parent du Chatelet, 2007, J. Atmos. Oceanic
Technol., 24, 1439-1451: texture (Eq. 1), aggregation (Eqs. 3 and 4), the
thresholds of Table 1 and the despeckling of Sect. 3.
"""
import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.retrieve import _gourley_numpy as gnp
from radarx.retrieve import _gourley_tables as gtab
from radarx.retrieve import apply_mask, echo_mask
from radarx.retrieve import qc as qcmod

from .test_qc import DR, _sweep, _volume

ENGINES = ["numpy"] + (["compiled"] if qcmod.HAS_COMPILED_GOURLEY else [])
M = "gourley2007"
DIMS = ("azimuth", "range")
NEXRAD_FLOORS = dict(dbzh=-32.0, zdr=-12.9, rhohv=0.21, phidp=-0.1)


def _ds(z, zdr=None, rho=None, phi=None, vel=None, sig=None, snr=None):
    nray, ng = z.shape
    data = {"DBZH": (DIMS, z)}
    for name, a in (
        ("ZDR", zdr),
        ("RHOHV", rho),
        ("PHIDP", phi),
        ("VRADH", vel),
        ("ZSIG", sig),
        ("SNRH", snr),
    ):
        if a is not None:
            data[name] = (DIMS, a)
    return xr.Dataset(
        data,
        coords={
            "azimuth": np.arange(nray) * 360.0 / nray,
            "range": (np.arange(ng) + 0.5) * DR,
            "elevation": ("azimuth", np.full(nray, 0.5)),
        },
    )


def _flat(rho, z=20.0, zdr=0.5, phi=30.0, nray=36, ng=30, **extra):
    """Constant fields: the textures are zero, the paper memberships at 0."""
    one = np.ones((nray, ng))
    return _ds(z * one, zdr=zdr * one, rho=rho * one, phi=phi * one, **extra)


def _tables():
    return gtab.memberships(None)


def _expected_share(rho, tex_zdr=0.0, tex_phi=0.0, suppress=(False, False, False)):
    """Hand calculation of Eqs. 3 and 4 with the default tables."""
    xs, ys = _tables()
    w = gtab.weights(None)
    values = (rho, tex_zdr, tex_phi)
    q = np.zeros(3)
    for k in range(3):
        for c in range(3):
            q[c] += w[k] * np.interp(values[k], xs[k], ys[k][c])
    q /= w.sum()
    q[np.asarray(suppress, dtype=bool)] = 0.0
    return q[0] / q.sum(), q


# --------------------------------------------------------------------------
# the paper's numbers
# --------------------------------------------------------------------------


def test_weights_follow_equation_4():
    w = gtab.DEFAULT_WEIGHTS
    assert sum(w.values()) == pytest.approx(1.0)
    # "the texture of PhiDP has 3 times more weight than rhohv and more than
    # double the weight of the texture of ZDR" (p. 1444)
    assert w["phidp_texture"] / w["rhohv"] == pytest.approx(3.0, abs=0.01)
    assert w["phidp_texture"] / w["zdr_texture"] == pytest.approx(2.5, abs=0.01)
    assert w["zdr_texture"] / w["rhohv"] == pytest.approx(1.2, abs=0.01)
    inv = np.array([1 / 0.243, 1 / 0.203, 1 / 0.081])
    np.testing.assert_allclose(
        [w["rhohv"], w["zdr_texture"], w["phidp_texture"]], inv / inv.sum()
    )
    assert qcmod.GOURLEY_WEIGHTS is gtab.DEFAULT_WEIGHTS


def test_membership_example_of_the_paper():
    """rhohv = 0.97: precipitation 0.149, ground clutter 0.030, clear air 0.007."""
    xs, ys = _tables()
    got = [np.interp(0.97, xs[0], ys[0][c]) for c in range(3)]
    np.testing.assert_allclose(got, [0.149, 0.030, 0.007], atol=1e-12)


@pytest.mark.parametrize(
    "name, modes, tol",
    [
        ("rhohv", (0.97, 0.92, 0.42), (0.02, 0.03, 0.1)),
        ("zdr_texture", (0.4, 5.2, 2.5), (0.15, 0.3, 0.3)),
        ("phidp_texture", (4.0, 125.0, 35.0), (2.5, 10.0, 6.0)),
    ],
)
def test_membership_modes_of_the_paper(name, modes, tol):
    """Modes of Fig. 3 quoted in Sect. 2b (p. 1443)."""
    x, *curves = qcmod.GOURLEY_MEMBERSHIPS[name]
    x = np.asarray(x)
    fine = np.linspace(x[0], x[-1], 20001)
    for curve, mode, t in zip(curves, modes, tol):
        peak = fine[np.argmax(np.interp(fine, x, curve))]
        assert abs(peak - mode) <= t, (name, mode, peak)


def test_texture_is_the_rms_difference_of_equation_1():
    """3 x 3 box: sum of squared differences of the 8 neighbours / 9, root."""
    y = np.zeros((9, 9))
    y[4, 4] = 3.0
    links = np.ones(9, dtype=np.uint8)
    ok = np.ones(y.shape, dtype=bool)
    t = gnp.texture(y, ok, links, False)
    assert t[4, 4] == pytest.approx(np.sqrt(8 * 9.0 / 9.0))
    # a neighbour of the spike: one difference of 3 among nine gates
    assert t[4, 5] == pytest.approx(np.sqrt(9.0 / 9.0))
    assert t[4, 3] == pytest.approx(1.0)
    assert t[3, 5] == pytest.approx(1.0)
    assert t[0, 0] == 0.0
    # a ramp along the ray: differences of +-1 and +-2 on three rays
    ramp = np.tile(np.arange(9.0), (9, 1))
    t = gnp.texture(ramp, ok, links, False)
    assert t[4, 4] == pytest.approx(np.sqrt(3 * (1 + 1) / 9.0))
    # fewer than three neighbours with data: undefined
    ok2 = ok.copy()
    ok2[3:6, 3:6] = False
    ok2[4, 4] = True
    ok2[3, 3] = ok2[3, 4] = True
    assert np.isnan(gnp.texture(ramp, ok2, links, False)[4, 4])
    # the phase difference is wrapped
    phi = np.full((9, 9), 359.0)
    phi[:, 5:] = 1.0
    t = gnp.texture(phi, ok, links, True)
    assert t[4, 4] == pytest.approx(np.sqrt(3 * 4 / 9.0))
    assert gnp.texture(phi, ok, links, False)[4, 4] > 100


# --------------------------------------------------------------------------
# aggregation, rules and despeckling by hand
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("rho", [0.3, 0.5, 0.75, 0.92, 0.97, 0.995])
def test_score_is_the_share_of_the_precipitation_class(engine, rho):
    out = echo_mask(_flat(rho), method=M, nodata=None, engine=engine)
    suppress = (rho < 0.7, False, False)
    share, q = _expected_share(rho, suppress=suppress)
    got = out.METEO_SCORE.values
    np.testing.assert_allclose(got, np.float32(share), rtol=1e-6)
    wins = q[0] > max(q[1], q[2]) and not suppress[0]
    assert (out.ECHO_CLASS.values == (1 if wins else 2)).all()


@pytest.mark.parametrize("engine", ENGINES)
def test_rain_like_gates_win_and_clutter_like_gates_do_not(engine):
    rng = np.random.default_rng(3)
    shape = (60, 80)
    rain = _ds(
        30 + rng.normal(0, 1, shape),
        zdr=0.8 + rng.normal(0, 0.15, shape),
        rho=np.clip(0.99 + rng.normal(0, 0.003, shape), 0, 1),
        phi=40 + 0.3 * np.arange(80) + rng.normal(0, 1.5, shape),
    )
    clutter = _ds(
        30 + rng.normal(0, 10, shape),
        zdr=rng.normal(0, 4, shape),
        rho=rng.uniform(0.4, 0.9, shape),
        phi=rng.uniform(0, 360, shape),
    )
    assert np.mean(echo_mask(rain, method=M, engine=engine).METEO_MASK) > 0.99
    assert np.mean(echo_mask(clutter, method=M, engine=engine).METEO_MASK) < 0.01


@pytest.mark.parametrize("engine", ENGINES)
def test_table_1_thresholds_suppress_precipitation(engine):
    """Each threshold of Table 1 removes a gate that scores as rain."""
    kw = dict(method=M, nodata=None, engine=engine)
    base = echo_mask(_flat(0.99, z=25.0), **kw)
    assert base.METEO_MASK.all()
    # rhohv < 0.7 (rain-like texture, but the correlation is low)
    low_rho = echo_mask(_flat(0.69, z=25.0), **kw)
    assert not low_rho.METEO_MASK.any()
    assert np.isfinite(low_rho.METEO_SCORE.values).all()
    # Z < 5 dBZ
    assert not echo_mask(_flat(0.99, z=4.0), **kw).METEO_MASK.any()
    assert echo_mask(_flat(0.99, z=6.0), **kw).METEO_MASK.all()
    # PhiDP < -40 degrees
    assert not echo_mask(_flat(0.99, z=25.0, phi=-41.0), **kw).METEO_MASK.any()
    assert echo_mask(_flat(0.99, z=25.0, phi=-39.0), **kw).METEO_MASK.all()
    # texture of PhiDP > 100 degrees: alternating phase
    ds = _flat(0.99, z=25.0)
    ds["PHIDP"][:, ::2] = 150.0
    ds["PHIDP"][:, 1::2] = -50.0 + 360.0
    assert not echo_mask(ds, **kw).METEO_MASK.any()
    # the rules can be switched off or changed
    assert echo_mask(
        _flat(0.69, z=25.0), rules={"rhohv_min": None}, **kw
    ).METEO_MASK.all()
    assert echo_mask(_flat(0.99, z=4.0), rules={"dbz_min": None}, **kw).METEO_MASK.all()
    assert echo_mask(_flat(0.99, z=4.0), rules={"dbz_min": 3.0}, **kw).METEO_MASK.all()


@pytest.mark.parametrize("engine", ENGINES)
def test_table_1_thresholds_for_ground_clutter_and_clear_air(engine):
    """Suppressing a class raises the share of the precipitation class."""
    kw = dict(method=M, nodata=None, engine=engine)
    rho = 0.7  # not suppressed; precipitation membership is small
    free = echo_mask(_flat(rho), **kw).METEO_SCORE.values
    np.testing.assert_allclose(free, np.float32(_expected_share(rho)[0]), rtol=1e-6)
    # |V| > 5 m/s: ground clutter suppressed
    fast = _flat(rho, vel=np.full((36, 30), -6.0))
    got = echo_mask(fast, field="VRADH", **kw).METEO_SCORE.values
    np.testing.assert_allclose(
        got, np.float32(_expected_share(rho, suppress=(0, 1, 0))[0]), rtol=1e-6
    )
    slow = _flat(rho, vel=np.full((36, 30), 4.0))
    np.testing.assert_array_equal(
        echo_mask(slow, field="VRADH", **kw).METEO_SCORE.values, free
    )
    # pulse-to-pulse variability of Z > 5 dB: ground clutter suppressed
    sig = _flat(rho, sig=np.full((36, 30), 6.0))
    got = echo_mask(sig, zsigma="ZSIG", **kw).METEO_SCORE.values
    np.testing.assert_allclose(
        got, np.float32(_expected_share(rho, suppress=(0, 1, 0))[0]), rtol=1e-6
    )
    sig5 = _flat(rho, sig=np.full((36, 30), 5.0))
    np.testing.assert_array_equal(
        echo_mask(sig5, zsigma="ZSIG", **kw).METEO_SCORE.values, free
    )
    # Z > 30 dBZ: clear air suppressed
    strong = _flat(rho, z=31.0)
    got = echo_mask(strong, **kw).METEO_SCORE.values
    np.testing.assert_allclose(
        got, np.float32(_expected_share(rho, suppress=(0, 0, 1))[0]), rtol=1e-6
    )
    # both, and the rules of the other classes unchanged
    both = _flat(rho, z=31.0, vel=np.full((36, 30), 9.0))
    got = echo_mask(both, field="VRADH", **kw).METEO_SCORE.values
    np.testing.assert_allclose(
        got, np.float32(_expected_share(rho, suppress=(0, 1, 1))[0]), rtol=1e-6
    )
    off = echo_mask(
        both, field="VRADH", rules={"vrad_max": None, "dbz_max": None}, **kw
    ).METEO_SCORE.values
    np.testing.assert_array_equal(off, free)


def _patch(rho_centre, neighbours, rho_far=0.4, shape=(36, 30)):
    """Constant fields with rhohv rho_far, except the gates of a 3 x 3 box."""
    rho = np.full(shape, rho_far)
    r0, g0 = 10, 10
    rho[r0, g0] = rho_centre
    for dr, dg in neighbours:
        rho[r0 + dr, g0 + dg] = 0.99
    ds = _flat(0.99, nray=shape[0], ng=shape[1])
    ds["RHOHV"] = (DIMS, rho)
    return ds, (r0, g0)


BOX = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("n_precip", range(0, 9))
def test_despeckling_of_isolated_precipitation(engine, n_precip):
    """Fewer than three precipitation neighbours: the gate is reassigned."""
    ds, (r0, g0) = _patch(0.99, BOX[:n_precip])
    on = echo_mask(ds, method=M, nodata=None, engine=engine).ECHO_CLASS.values
    off = echo_mask(
        ds, method=M, nodata=None, despeckle=False, engine=engine
    ).ECHO_CLASS.values
    assert off[r0, g0] == 1
    assert on[r0, g0] == (3 if n_precip < 3 else 1)
    # the neighbours are precipitation gates themselves
    for dr, dg in BOX[:n_precip]:
        assert off[r0 + dr, g0 + dg] == 1


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("n_precip", range(0, 9))
def test_despeckling_of_enclosed_clear_air(engine, n_precip):
    """More than six precipitation neighbours: clear air becomes precipitation."""
    ds, (r0, g0) = _patch(0.4, BOX[:n_precip], rho_far=0.4)
    on = echo_mask(ds, method=M, nodata=None, engine=engine).ECHO_CLASS.values
    off = echo_mask(
        ds, method=M, nodata=None, despeckle=False, engine=engine
    ).ECHO_CLASS.values
    assert off[r0, g0] == 2
    expected = 1 if n_precip > 6 else 2
    # the gates around are precipitation gates that may themselves be isolated
    if n_precip > 6:
        assert on[r0, g0] == expected
    else:
        assert on[r0, g0] == 2


@pytest.mark.parametrize("engine", ENGINES)
def test_ground_clutter_is_not_converted_by_the_despeckling(engine):
    """Only clear-air gates enclosed by precipitation are reassigned."""
    ds, (r0, g0) = _patch(0.4, BOX)
    ds["ZDR"][r0, g0] = 9.0  # large texture of ZDR, PhiDP too: clutter-like
    ds["PHIDP"][r0, g0] = 200.0
    ds["ZDR"][r0, g0 - 6] = 6.0
    cls = echo_mask(ds, method=M, nodata=None, engine=engine).ECHO_CLASS.values
    assert cls[r0, g0 - 6] == 2
    assert cls[r0, g0] in (1, 2)  # decided by the largest class, either is allowed


@pytest.mark.parametrize("engine", ENGINES)
def test_gates_without_polarimetric_data_are_class_3(engine):
    ds = _ds(np.full((36, 30), 20.0), vel=np.full((36, 30), 1.0))
    out = echo_mask(ds, method=M, field="VRADH", nodata=None, engine=engine)
    assert (out.ECHO_CLASS.values == 3).all()
    assert np.isnan(out.METEO_SCORE.values).all()
    assert not out.METEO_MASK.any()


@pytest.mark.parametrize("engine", ENGINES)
def test_a_single_field_is_enough(engine):
    for name in ("rho", "zdr", "phi"):
        f = dict(_flat(0.99, nray=36, ng=30).data_vars)
        keep = {"rho": "RHOHV", "zdr": "ZDR", "phi": "PHIDP"}[name]
        ds = xr.Dataset({"DBZH": f["DBZH"], keep: f[keep]}, coords=_flat(0.99).coords)
        out = echo_mask(ds, method=M, nodata=None, engine=engine)
        assert out.METEO_MASK.all(), name


@pytest.mark.parametrize("engine", ENGINES)
def test_no_echo_and_nexrad_codes(engine):
    ds = _flat(0.99, z=25.0)
    ds["DBZH"][:, :5] = -33.0  # NEXRAD no-data code
    ds["ZDR"][:, 10:15] = -13.0625
    ds["RHOHV"][:, 10:15] = 0.2017
    ds["PHIDP"][:, 10:15] = -0.705
    ds = ds.assign_attrs(waveform_type="contiguous_surveillance")
    out = echo_mask(ds, method=M, engine=engine)
    cls = out.ECHO_CLASS.values
    assert (cls[:, :5] == 0).all()
    assert np.isnan(out.METEO_SCORE.values[:, :5]).all()
    # the polarimetric codes are no data: only DBZH is left, so no class
    assert (cls[:, 10:15] == 3).all()
    assert (cls[:, 5:10] == 1).all()
    snr = np.where(np.arange(30) < 20, 10.0, 0.0) * np.ones((36, 1))
    with_snr = _flat(0.99, z=25.0, snr=snr)
    out = echo_mask(with_snr, method=M, nodata=None, engine=engine)
    assert (out.ECHO_CLASS.values[:, 20:] == 0).all()


# --------------------------------------------------------------------------
# engines against an independent per-gate implementation
# --------------------------------------------------------------------------


def _reference(ds, rules=None, despeckle=True, slopes=(0.0, 0.0), start=45.0, **names):
    """Plain loops over gates, written from the paper's equations."""
    xs, ys = _tables()
    w = gtab.weights(None)
    r = {**gtab.DEFAULT_RULES, **(rules or {})}
    z = ds.DBZH.values
    fld = {k: ds[v].values for k, v in names.items()}
    nray, ng = z.shape
    rng_km = ds.range.values / 1000.0
    az = ds.azimuth.values

    def ring(i):
        return [(i - 1) % nray, i, (i + 1) % nray]

    def echo(i, g):
        return np.isfinite(z[i, g]) and z[i, g] > -32.0

    def has(a, floor, i, g):
        return echo(i, g) and np.isfinite(a[i, g]) and a[i, g] > floor

    def tex(a, floor, i, g, wrap):
        total, n = 0.0, 1
        for q in ring(i):
            for dg in (-1, 0, 1):
                gg = g + dg
                if (q == i and dg == 0) or gg < 0 or gg >= ng:
                    continue
                if has(a, floor, q, gg):
                    d = a[i, g] - a[q, gg]
                    if wrap:
                        d = d - 360 * np.floor(d / 360 + 0.5)
                    total += d * d
                    n += 1
        return np.sqrt(total / n) if n - 1 >= 3 else np.nan

    typ = np.zeros((nray, ng), dtype=int)
    score = np.full((nray, ng), np.nan)
    for i in range(nray):
        for g in range(ng):
            if not echo(i, g):
                continue
            num = np.zeros(3)
            den = 0.0
            sup = [False, False, False]
            terms = []
            if "rho" in fld and has(fld["rho"], 0.21, i, g):
                v = fld["rho"][i, g]
                v = 2 - v if v > 1 else v
                terms.append((0, v))
                sup[0] |= v < r["rhohv_min"]
            if "zdr" in fld and has(fld["zdr"], -12.9, i, g):
                t = tex(fld["zdr"], -12.9, i, g, False)
                if slopes[0] and np.isfinite(t):
                    t = max(t - slopes[0] * max(rng_km[g] - start, 0), 0)
                if np.isfinite(t):
                    terms.append((1, t))
            if "phi" in fld and has(fld["phi"], -0.1, i, g):
                t = tex(fld["phi"], -0.1, i, g, True)
                if slopes[1] and np.isfinite(t):
                    t = max(t - slopes[1] * max(rng_km[g] - start, 0), 0)
                if np.isfinite(t):
                    terms.append((2, t))
                    sup[0] |= t > r["phidp_texture_max"]
                sup[0] |= fld["phi"][i, g] < r["phidp_min"]
            for k, v in terms:
                for c in range(3):
                    num[c] += w[k] * np.interp(v, xs[k], ys[k][c])
                den += w[k]
            sup[0] |= z[i, g] < r["dbz_min"]
            sup[2] |= z[i, g] > r["dbz_max"]
            if "vel" in fld and has(fld["vel"], -63.9, i, g):
                sup[1] |= abs(fld["vel"][i, g]) > r["vrad_max"]
            if "sig" in fld and np.isfinite(fld["sig"][i, g]):
                sup[1] |= fld["sig"][i, g] > r["zsigma_max"]
            if den == 0:
                typ[i, g] = 4
                continue
            q = num / den
            q[sup] = 0.0
            tot = q.sum()
            score[i, g] = q[0] / tot if tot > 0 else 0.0
            typ[i, g] = 1 if q[0] > max(q[1], q[2]) else (2 if q[1] > q[2] else 3)
    cls = np.zeros((nray, ng), dtype=np.int8)
    out = typ.copy()
    if despeckle:
        for i in range(nray):
            for g in range(ng):
                if typ[i, g] not in (1, 3):
                    continue
                n = 0
                for q in ring(i):
                    for dg in (-1, 0, 1):
                        gg = g + dg
                        if (q == i and dg == 0) or gg < 0 or gg >= ng:
                            continue
                        n += typ[q, gg] == 1
                if typ[i, g] == 1 and n < 3:
                    out[i, g] = 5  # speckle
                if typ[i, g] == 3 and n > 6:
                    out[i, g] = 1
    cls[out == 1] = 1
    cls[np.isin(out, (2, 3))] = 2
    cls[np.isin(out, (4, 5))] = 3
    del az
    return score, cls


def _random_sweep(seed, nray=24, ng=60, holes=True):
    rng = np.random.default_rng(seed)
    shape = (nray, ng)
    kind = rng.integers(0, 3, (nray // 4 + 1, ng // 6 + 1)).repeat(4, 0).repeat(6, 1)
    kind = kind[:nray, :ng]
    z = np.where(kind == 0, 28, np.where(kind == 1, 12, 33)) + rng.normal(0, 4, shape)
    zdr = np.where(kind == 0, 0.8, np.where(kind == 1, 3, 1.5)) + rng.normal(
        0, 1, shape
    ) * np.where(kind == 0, 0.2, 1.5)
    rho = np.where(kind == 0, 0.985, np.where(kind == 1, 0.55, 0.8)) + rng.normal(
        0, 0.05, shape
    )
    phi = (
        np.where(kind == 0, 50 + 0.5 * np.arange(ng), rng.uniform(0, 360, shape))
        + rng.normal(0, 3, shape)
    ) % 360
    vel = rng.normal(0, 4, shape)
    sig = np.abs(rng.normal(3, 2, shape))
    if holes:
        z[rng.random(shape) < 0.05] = -33.0
        z[rng.random(shape) < 0.02] = np.nan
        zdr[rng.random(shape) < 0.05] = -13.0625
        rho[rng.random(shape) < 0.05] = 0.2
        phi[rng.random(shape) < 0.05] = np.nan
        vel[rng.random(shape) < 0.05] = -64.0
    return _ds(z, zdr=zdr, rho=rho, phi=phi, vel=vel, sig=sig)


NAMES = dict(zdr="ZDR", rho="RHOHV", phi="PHIDP", vel="VRADH", sig="ZSIG")


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("seed", range(6))
def test_matches_an_independent_implementation(engine, seed):
    ds = _random_sweep(seed)
    ds = ds.assign_attrs(waveform_type="contiguous_surveillance")
    kw = dict(method=M, field="VRADH", zsigma="ZSIG", engine=engine)
    out = echo_mask(ds, **kw)
    ref_score, ref_cls = _reference(ds, **NAMES)
    np.testing.assert_array_equal(out.ECHO_CLASS.values, ref_cls)
    np.testing.assert_allclose(
        out.METEO_SCORE.values, ref_score.astype(np.float32), rtol=1e-6, equal_nan=True
    )
    assert set(np.unique(ref_cls)) >= {0, 1, 2}
    # without despeckling, with other rules and a range correction
    rules = {"rhohv_min": 0.6, "dbz_min": None, "vrad_max": 3.0}
    corr = {"zdr": 0.05, "phidp": 0.5, "start_km": 0.5}
    out = echo_mask(ds, rules=rules, range_correction=corr, despeckle=False, **kw)
    ref_score, ref_cls = _reference(
        ds,
        rules={**rules, "dbz_min": -np.inf},
        despeckle=False,
        slopes=(0.05, 0.5),
        start=0.5,
        **NAMES,
    )
    np.testing.assert_array_equal(out.ECHO_CLASS.values, ref_cls)
    np.testing.assert_allclose(
        out.METEO_SCORE.values, ref_score.astype(np.float32), rtol=1e-6, equal_nan=True
    )


@pytest.mark.skipif(not qcmod.HAS_COMPILED_GOURLEY, reason="kernel not built")
@pytest.mark.parametrize("seed", range(4))
def test_engines_identical_and_thread_count_does_not_matter(seed):
    ds = _random_sweep(seed, nray=90, ng=140).isel(azimuth=slice(0, 71))
    kw = dict(method=M, field="VRADH", zsigma="ZSIG")
    a = echo_mask(ds, engine="numpy", **kw)
    b = echo_mask(ds, engine="compiled", **kw)
    np.testing.assert_array_equal(a.ECHO_CLASS.values, b.ECHO_CLASS.values)
    np.testing.assert_allclose(
        a.METEO_SCORE.values, b.METEO_SCORE.values, rtol=1e-6, equal_nan=True
    )
    one = echo_mask(ds, n_threads=1, **kw)
    for n in (2, 5):
        many = echo_mask(ds, n_threads=n, **kw)
        np.testing.assert_array_equal(one.ECHO_CLASS.values, many.ECHO_CLASS.values)
        np.testing.assert_array_equal(one.METEO_SCORE.values, many.METEO_SCORE.values)


@pytest.mark.skipif(not qcmod.HAS_COMPILED_GOURLEY, reason="kernel not built")
def test_kernel_input_checks():
    from radarx.retrieve import _gourley

    z = np.full((4, 5), 20.0)
    links = np.ones(4, dtype=np.uint8)
    rng = np.arange(5.0)
    xs, ys = _tables()
    args = dict(
        dbzh=[z],
        zdr=[None],
        rhohv=[z * 0 + 0.99],
        phidp=[None],
        vrad=[None],
        zsigma=[None],
        snr=[None],
        links=[links],
        rng_km=[rng],
        floors=np.full(5, -np.inf),
        snr_min=3.0,
        mem_x=xs,
        mem_y=ys,
        weights=gtab.weights(None),
        rules=gtab.rules(None),
        slopes=np.array([0.0, 0.0, 45.0]),
        despeckle=True,
    )
    scores, classes = _gourley.classify(**args)
    assert scores[0].shape == (4, 5) and classes[0].dtype == np.int8
    bad = [
        ("zdr", [z[:2]]),
        ("links", [links[:2]]),
        ("rng_km", [rng[:2]]),
        ("floors", np.zeros(4)),
        ("mem_x", xs[:2]),
        ("mem_y", ys[:2]),
        ("weights", np.ones(2)),
        ("rules", np.ones(3)),
        ("slopes", np.ones(2)),
        ("snr", [None, None]),
        ("dbzh", [np.ones(5)]),
    ]
    for key, value in bad:
        with pytest.raises(ValueError):
            _gourley.classify(**{**args, key: value})
    short = [np.ones((2, 5)), ys[1], ys[2]]
    with pytest.raises(ValueError):
        _gourley.classify(**{**args, "mem_y": [short[0], ys[1], ys[2]]})


# --------------------------------------------------------------------------
# options
# --------------------------------------------------------------------------


def test_option_errors():
    ds = _flat(0.99)
    with pytest.raises(ValueError, match="gourley2007"):
        echo_mask(ds, method=M, threshold=0.5)
    with pytest.raises(ValueError, match="gourley2007"):
        echo_mask(ds, method=M, window=2.0)
    with pytest.raises(ValueError, match="gourley2007"):
        echo_mask(ds, method=M, min_size=5)
    with pytest.raises(ValueError, match="gourley2007"):
        echo_mask(ds, method=M, limits={"rhohv": (0, 1, 2, 3)})
    with pytest.raises(ValueError, match="spin_threshold"):
        echo_mask(ds, method=M, spin_threshold=3.0)
    with pytest.raises(ValueError, match="phidp_texture"):
        echo_mask(ds, method=M, phidp_texture="detrended")
    with pytest.raises(ValueError, match="zh_3km"):
        echo_mask(ds, method=M, zh_3km="X")
    with pytest.raises(ValueError, match="unknown weights"):
        echo_mask(ds, method=M, weights={"zdr": 1.0})
    with pytest.raises(ValueError, match="non-negative"):
        echo_mask(ds, method=M, weights={"rhohv": -1.0})
    with pytest.raises(ValueError, match="non-negative"):
        echo_mask(ds, method=M, weights={k: 0 for k in gtab.VARIABLES})
    with pytest.raises(ValueError, match="unknown rules"):
        echo_mask(ds, method=M, rules={"x": 1})
    with pytest.raises(ValueError, match="unknown range_correction"):
        echo_mask(ds, method=M, range_correction={"x": 1})
    with pytest.raises(ValueError, match="range_correction"):
        echo_mask(ds, method=M, range_correction={"zdr": -1.0})
    with pytest.raises(KeyError):
        echo_mask(ds, method=M, field="nope")
    with pytest.raises(KeyError):
        echo_mask(ds, method=M, zsigma="nope")
    # the other methods reject the options of this one
    for kw in (
        {"memberships": {}},
        {"rules": {}},
        {"range_correction": {}},
        {"despeckle": True},
        {"zsigma": "ZSIG"},
    ):
        with pytest.raises(ValueError, match="gourley2007"):
            echo_mask(ds, **kw)
        with pytest.raises(ValueError, match="gourley2007"):
            echo_mask(ds, method="krause2016", **kw)
    with pytest.raises(ValueError, match="krause2016"):
        echo_mask(ds, field="VRADH")
    with pytest.raises(ValueError, match="method must be"):
        echo_mask(ds, method="gourley")
    with pytest.raises(ValueError, match="engine"):
        echo_mask(ds, method=M, engine="gpu")


@pytest.mark.skipif(qcmod.HAS_COMPILED_GOURLEY, reason="only without the kernel")
def test_compiled_engine_needs_the_kernel():  # pragma: no cover - depends on the build
    with pytest.raises(ImportError):
        echo_mask(_flat(0.99), method=M, engine="compiled")


def test_membership_tables_are_checked():
    ds = _flat(0.99)
    good = ([0, 1, 2], [0.1, 0.2, 0.1], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2])
    echo_mask(ds, method=M, memberships={"rhohv": good})
    for bad in (
        ([0, 1, 2], [0.1, 0.2, 0.1], [0.1, 0.1, 0.1]),  # curve missing
        ([0, 1], [0.1], [0.1, 0.1], [0.2, 0.2]),  # lengths differ
        ([0], [0.1], [0.1], [0.1]),  # one point
        ([0, 0, 1], [0.1] * 3, [0.1] * 3, [0.1] * 3),  # not increasing
        ([0, 1, np.nan], [0.1] * 3, [0.1] * 3, [0.1] * 3),  # x not finite
        ([0, 1, 2], [0.1, -0.2, 0.1], [0.1] * 3, [0.1] * 3),  # negative
        ([0, 1, 2], [0.1, np.inf, 0.1], [0.1] * 3, [0.1] * 3),  # not finite
        (["a", "b"], [1, 1], [1, 1], [1, 1]),  # not numeric
        3.0,  # not a table
    ):
        with pytest.raises(ValueError, match="memberships"):
            echo_mask(ds, method=M, memberships={"rhohv": bad})
    with pytest.raises(ValueError, match="unknown memberships"):
        echo_mask(ds, method=M, memberships={"zdr": good})


@pytest.mark.parametrize("engine", ENGINES)
def test_own_memberships_and_weights_are_used(engine):
    ds = _flat(0.99)
    # a rhohv table that prefers clear air everywhere, with all the weight on it
    table = ([0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0])
    out = echo_mask(
        ds,
        method=M,
        memberships={"rhohv": table},
        weights={"rhohv": 1.0, "zdr_texture": 0.0, "phidp_texture": 0.0},
        nodata=None,
        engine=engine,
    )
    assert (out.ECHO_CLASS.values == 2).all()
    assert (out.METEO_SCORE.values == 0).all()


@pytest.mark.parametrize("engine", ENGINES)
def test_range_correction_lowers_the_textures_beyond_the_start(engine):
    """Beyond start_km the fitted linear increase is subtracted (Fig. 2)."""
    rng = np.random.default_rng(0)
    shape = (36, 400)
    one = np.ones(shape)
    phi = (50 + rng.normal(0, 8.0, shape)) % 360
    zdr = 0.5 + rng.normal(0, 0.5, shape)
    ds = _ds(25 * one, zdr=zdr, rho=0.995 * one, phi=phi)
    kw = dict(method=M, nodata=None, engine=engine)
    plain = echo_mask(ds, **kw).METEO_SCORE.values
    start = {"start_km": 10.0}
    same = echo_mask(ds, range_correction=start, **kw).METEO_SCORE.values
    np.testing.assert_array_equal(same, plain)
    # slopes so large that every texture beyond the start is corrected to zero
    big = {"zdr": 1e3, "phidp": 1e5, "start_km": 10.0}
    corr = echo_mask(ds, range_correction=big, **kw).METEO_SCORE.values
    near = ds.range.values / 1000.0 <= 10.0
    np.testing.assert_array_equal(corr[:, near], plain[:, near])
    expect = np.float32(_expected_share(0.995)[0])
    far = ~near
    ok = np.isfinite(corr[:, far])
    np.testing.assert_allclose(corr[:, far][ok], expect, rtol=1e-6)
    assert np.nanmean(corr[:, far]) > np.nanmean(plain[:, far])


# --------------------------------------------------------------------------
# volumes, split cuts, other methods untouched
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_synthetic_echo_types(engine):
    ds, truth = _sweep()
    out = echo_mask(ds, method=M, engine=engine)
    cls = out.ECHO_CLASS.values
    assert np.mean(cls[truth == 1] == 1) > 0.99
    assert np.mean(cls[truth == 2] != 1) > 0.99
    assert np.mean(cls[truth == 3] == 3) == 1.0
    assert (cls[truth == 0] == 0).all()
    np.testing.assert_array_equal(out.METEO_MASK.values, cls == 1)
    assert out.METEO_SCORE.attrs["comment"].startswith("share of the precipitation")


@pytest.mark.parametrize("engine", ENGINES)
def test_volume_and_split_cuts(engine):
    dtree = _volume()
    out = echo_mask(dtree, method=M, field="VRADH", engine=engine)
    surv = out["sweep_0"].ds.ECHO_CLASS.values
    dop = out["sweep_1"].ds.ECHO_CLASS.values
    both = (surv > 0) & (dop > 0)
    assert both.sum() > 1000
    np.testing.assert_array_equal(dop[both], surv[both])
    assert "sweep_3" not in out.children
    own = echo_mask(dtree, method=M, field="VRADH", split_cuts=False, engine=engine)
    assert (
        own["sweep_1"].ds.ECHO_CLASS.values[own["sweep_1"].ds.ECHO_CLASS > 0] == 3
    ).all()
    # a sweep lacking the velocity field is not an error
    again = echo_mask(dtree, method=M, field="VRADH", engine=engine)
    np.testing.assert_array_equal(
        again["sweep_0"].ds.ECHO_CLASS.values, out["sweep_0"].ds.ECHO_CLASS.values
    )
    masked = apply_mask(dtree, mask=out, data_vars=["DBZH"])
    assert (
        masked["sweep_0"].ds.DBZH.isnull().sum()
        > dtree["sweep_0"].ds.DBZH.isnull().sum()
    )


def test_accessor_and_input_unchanged():
    ds = _random_sweep(1)
    before = ds.copy(deep=True)
    out = ds.radarx.echo_mask(method=M, field="VRADH", zsigma="ZSIG")
    xr.testing.assert_identical(ds, before)
    assert set(out.data_vars) == {"ECHO_CLASS", "METEO_SCORE", "METEO_MASK"}
    assert out.METEO_SCORE.dtype == np.float32 and out.ECHO_CLASS.dtype == np.int8


def test_other_methods_are_unchanged():
    ds, _ = _sweep()
    a = echo_mask(ds)
    b = echo_mask(ds, method="radarx")
    np.testing.assert_array_equal(a.ECHO_CLASS.values, b.ECHO_CLASS.values)
    assert "gourley2007" in qcmod.METHODS


def test_dask_input_matches():
    pytest.importorskip("dask")
    ds = _random_sweep(2, nray=48, ng=50)
    eager = echo_mask(ds, method=M, field="VRADH", zsigma="ZSIG")
    lazy = echo_mask(
        ds.chunk({"azimuth": 24, "range": -1}), method=M, field="VRADH", zsigma="ZSIG"
    )
    np.testing.assert_array_equal(eager.ECHO_CLASS.values, lazy.ECHO_CLASS.values)
    np.testing.assert_array_equal(eager.METEO_SCORE.values, lazy.METEO_SCORE.values)


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def klbb():
    import xradar as xd
    from open_radar_data import DATASETS

    return xd.io.open_nexradlevel2_datatree(DATASETS.fetch("KLBB20160601_150025_V06"))


@pytest.fixture(scope="module")
def kgwx(tmp_path_factory):
    """Squall line and biological echo (KGWX, 30 March 2022), from AWS."""
    import xradar as xd

    from radarx.io.aws_data import download_file

    try:
        path = download_file(
            "unidata-nexrad-level2",
            "2022/03/30/KGWX/KGWX20220330_234639_V06",
            str(tmp_path_factory.mktemp("nexrad")),
        )
    except Exception as err:  # noqa: BLE001  # pragma: no cover - network
        pytest.skip(f"NEXRAD data not available: {err}")
    return xd.io.open_nexradlevel2_datatree(path)


@pytest.fixture(scope="module")
def csapr():
    """C-band CSAPR2 sweep with the CMAC gate classification (ARM)."""
    import xradar as xd
    from open_radar_data import DATASETS

    path = DATASETS.fetch("corcsapr2cmacppiM1.c1.20181111.030003.nc")
    return xd.io.open_cfradial1_datatree(path)["sweep_0"].to_dataset()


@pytest.mark.skipif(not qcmod.HAS_COMPILED_GOURLEY, reason="kernel not built")
def test_nexrad_engines_identical(klbb):
    kw = dict(method=M, field="VRADH")
    a = echo_mask(klbb, engine="numpy", **kw)
    b = echo_mask(klbb, engine="compiled", **kw)
    for name in a.children:
        ca, cb = a[name].ds.ECHO_CLASS.values, b[name].ds.ECHO_CLASS.values
        np.testing.assert_array_equal(ca, cb)
        np.testing.assert_allclose(
            a[name].ds.METEO_SCORE.values,
            b[name].ds.METEO_SCORE.values,
            rtol=1e-6,
            equal_nan=True,
        )


def test_kgwx_biological_echo_removed_squall_line_kept(kgwx):
    out = echo_mask(kgwx, method=M, field="VRADH")
    ds = kgwx["sweep_0"].to_dataset()
    cls = out["sweep_0"].ds.ECHO_CLASS.values
    az = np.deg2rad(ds.azimuth.values)[:, None]
    r = ds.range.values[None, :] / 1000.0
    x, y = r * np.sin(az), r * np.cos(az)
    z, zdr = ds.DBZH.values, ds.ZDR.values
    bio = (x > 85) & (x < 130) & (abs(y) < 45) & (zdr > 4) & (z < 20) & (cls > 0)
    squall = (x > -170) & (x < 20) & (z >= 35)
    assert bio.sum() > 3000 and squall.sum() > 100_000
    assert np.mean(cls[bio] != 1) > 0.8
    assert np.mean(cls[squall] == 1) > 0.99
    assert (cls[z <= -32] == 0).all()
    # the default method keeps about the same gates
    default = echo_mask(kgwx)["sweep_0"].ds.ECHO_CLASS.values
    echo = cls > 0
    assert np.mean((cls == 1)[echo] == (default == 1)[echo]) > 0.9


def test_csapr_agrees_with_the_cmac_gate_classification(csapr):
    """C band (the band of the paper's membership functions), CMAC reference."""
    f = dict(
        dbz="reflectivity",
        zdr="differential_reflectivity",
        rhohv="copol_correlation_coeff",
        phidp="differential_phase",
        snr="signal_to_noise_ratio",
    )
    out = echo_mask(csapr, method=M, field="mean_doppler_velocity", **f)
    gid = csapr.gate_id.values
    ref_met = np.isin(gid, [1, 2, 4])  # rain, snow, melting
    ref_non = np.isin(gid, [0, 3, 5])  # multi-trip, no scatter, clutter
    met = out.METEO_MASK.values
    echo = out.ECHO_CLASS.values > 0
    tp = (met & ref_met).sum()
    fp = (met & ref_non).sum()
    fn = (~met & ref_met & echo).sum()
    tn = (~met & ref_non & echo).sum()
    assert (tp + tn) / (tp + fp + fn + tn) > 0.83
    assert tp / (tp + fn) > 0.8
    assert fp / (tp + fp) < 0.15
