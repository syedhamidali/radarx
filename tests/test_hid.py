#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the fuzzy-logic hydrometeor classification
=====================================================
"""
import importlib

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.retrieve import hid, hid_classes

hidmod = importlib.import_module("radarx.retrieve.hid")

ENGINES = ["numpy"] + (["compiled"] if hidmod.HAS_COMPILED_KERNEL else [])
SCHEMES = [("park", "S"), ("dolan", "C"), ("dolan", "X"), ("dolan", "S")]
SCHEMES_ALL = SCHEMES + [("thompson", b) for b in ("S", "C", "X")]


def _sweep(fields, z=None, rng=None, extra=None):
    """Sweep on (azimuth, range) from 2-D fields, with gate heights ``z``."""
    shape = np.shape(fields["DBZH"])
    nray, ng = shape
    rng = 1000.0 + 250.0 * np.arange(ng) if rng is None else rng
    coords = {"azimuth": np.arange(nray) + 0.5, "range": rng}
    if z is not None:
        coords["z"] = (("azimuth", "range"), np.broadcast_to(z, shape))
    data = {k: (("azimuth", "range"), np.asarray(v, float)) for k, v in fields.items()}
    if extra:
        data.update(extra)
    return xr.Dataset(data, coords=coords)


def _centres(method, band):
    """One gate per class at the centre of its membership functions."""
    classes, t = hidmod._scheme(method, band)
    nc = len(classes)
    vals = np.full((5, nc), np.nan)
    for c in range(nc):
        if t["kind"][c, 0] == hidmod._TRAP:
            z = 0.5 * (t["par"][c, 0, 1] + t["par"][c, 0, 2])
            vals[0, c] = z
            for v in (1, 2, 3):
                p = t["par"][c, v] + [hidmod._zfunc(s, z) for s in t["fsel"][c, v]]
                vals[v, c] = 0.5 * (p[1] + p[2])
            vals[2, c] = 10 ** (vals[2, c] / 10)  # LKdp -> KDP
        else:
            for v in range(5):
                if t["kind"][c, v] == hidmod._BETA:
                    vals[v, c] = t["par"][c, v, 0]
    return classes, vals


@pytest.mark.parametrize("method,band", SCHEMES_ALL)
@pytest.mark.parametrize("engine", ENGINES)
def test_class_centres_are_recovered(method, band, engine):
    """Gates at the centre of a class's membership functions get that class."""
    classes, vals = _centres(method, band)
    nc = len(classes)
    fields = {"DBZH": vals[0][None], "ZDR": vals[1][None], "KDP": vals[2][None]}
    fields["RHOHV"] = vals[3][None]
    temp = None
    if not np.all(np.isnan(vals[4])):
        fields["TEMP"] = np.nan_to_num(vals[4], nan=-10.0)[None]
        temp = "TEMP"
    ds = _sweep(fields, z=np.full((1, nc), 2000.0))
    out = hid(ds, temp, band=band, method=method, engine=engine)
    codes = out.HID.values[0]
    expected = np.arange(1, nc + 1)
    if method == "thompson":
        # too few wet snow gates for a melting layer: the above-melting-layer
        # classes are used everywhere, so only those are recovered here
        above = [i for i, (a, _) in enumerate(classes) if a in ("PL", "DN", "IC", "AG")]
        np.testing.assert_array_equal(codes[above], expected[above])
    else:
        np.testing.assert_array_equal(codes, expected)
        np.testing.assert_allclose(out.HID_confidence.values[0], 1.0, atol=1e-6)


def _random_volume(seed=0, n=3, nray=40, ng=120):
    rnd = np.random.default_rng(seed)
    sweeps = []
    for k in range(n):
        shape = (nray, ng)
        z = rnd.uniform(-10, 70, shape)
        f = {
            "DBZH": np.where(rnd.random(shape) < 0.05, np.nan, z),
            "ZDR": np.where(
                rnd.random(shape) < 0.05, np.nan, rnd.uniform(-2, 6, shape)
            ),
            "KDP": np.where(rnd.random(shape) < 0.1, np.nan, rnd.uniform(-1, 8, shape)),
            "RHOHV": rnd.uniform(0.7, 1.0, shape),
            "TEMP": rnd.uniform(-40, 30, shape),
            "PHIDP_processed": rnd.uniform(0, 300, shape),
            "PBB": rnd.uniform(0, 60, shape),
        }
        h = np.broadcast_to(np.linspace(0, 12000, ng), shape) + 1000 * k
        ds = _sweep(f, z=h).assign(MET=(("azimuth", "range"), rnd.random(shape) > 0.1))
        sweeps.append(ds)
    return sweeps


@pytest.mark.skipif(not hidmod.HAS_COMPILED_KERNEL, reason="compiled kernel not built")
@pytest.mark.parametrize("method,band", SCHEMES_ALL)
def test_engines_agree(method, band):
    """Compiled kernel and NumPy oracle: identical classes, scores to 1e-6."""
    sweeps = _random_volume()
    tree = xr.DataTree.from_dict({f"sweep_{i}": ds for i, ds in enumerate(sweeps)})
    kw = dict(band=band, method=method, mask="MET", blockage="PBB")
    if method == "park":
        kw["melting_layer"] = (3000.0, 3500.0)
    if method == "thompson":
        kw["ml_gates"] = (10, 100)
    a = hid(tree, "TEMP", engine="compiled", **kw)
    b = hid(tree, "TEMP", engine="numpy", **kw)
    for i, ds in enumerate(sweeps):
        x, y = a[f"sweep_{i}"].to_dataset(), b[f"sweep_{i}"].to_dataset()
        np.testing.assert_array_equal(x.HID.values, y.HID.values)
        np.testing.assert_allclose(x.HID_scores, y.HID_scores, atol=1e-6)
        np.testing.assert_allclose(x.HID_confidence, y.HID_confidence, atol=1e-6)
        assert (x.HID.values[~ds.MET.values] == 0).all()
        assert (x.HID.values[ds.MET.values & ds.DBZH.notnull().values] > 0).all()
    if method == "thompson":
        h1 = float(a["sweep_0"]["melting_layer_height"])
        h2 = float(b["sweep_0"]["melting_layer_height"])
        assert h1 == h2


@pytest.mark.skipif(not hidmod.HAS_COMPILED_KERNEL, reason="compiled kernel not built")
def test_thread_count_does_not_change_result():
    ds = _random_volume(n=1, nray=200)[0]
    ref = hid(ds, "TEMP", band="C", n_threads=1)
    for nt in (2, 7, None):
        out = hid(ds, "TEMP", band="C", n_threads=nt)
        np.testing.assert_array_equal(out.HID.values, ref.HID.values)
        np.testing.assert_array_equal(out.HID_scores.values, ref.HID_scores.values)


def _rain_gate(z=45.0, height=1000.0, rng=None, **extra):
    zdr = 0.68 - 4.81e-2 * z + 2.92e-3 * z * z  # Park f2: inside the rain range
    f = {"DBZH": [[z]], "ZDR": [[zdr - 0.3]], "KDP": [[1.0]], "RHOHV": [[0.99]]}
    f.update({k: [[v]] for k, v in extra.items()})
    return _sweep(f, z=[[height]], rng=rng)


@pytest.mark.parametrize("engine", ENGINES)
def test_park_melting_layer_restrictions(engine):
    """No rain above the melting layer top, no dry snow below its bottom."""
    names = {a: c for c, a, _ in hid_classes("park")}
    ml = (3000.0, 3500.0)
    below = hid(_rain_gate(height=1000.0), band="S", melting_layer=ml, engine=engine)
    assert below.HID.item() == names["RA"]
    above = hid(_rain_gate(height=6000.0), band="S", melting_layer=ml, engine=engine)
    assert above.HID.item() not in (names["RA"], names["HR"], names["BD"])
    # dry snow signature below the melting layer is not dry snow
    snow = _sweep(
        {"DBZH": [[25.0]], "ZDR": [[0.15]], "KDP": [[0.1]], "RHOHV": [[0.99]]},
        z=[[6000.0]],
    )
    assert hid(snow, melting_layer=ml, engine=engine).HID.item() == names["DS"]
    low = snow.assign_coords(z=(("azimuth", "range"), [[1000.0]]))
    assert hid(low, melting_layer=ml, engine=engine).HID.item() != names["DS"]
    # the beam is 1 deg wide: at 200 km its upper edge (+1.7 km) reaches into
    # the layer from 1.5 km, where wet snow and graupel become possible
    far = _sweep(
        {"DBZH": [[35.0]], "ZDR": [[1.5]], "KDP": [[0.1]], "RHOHV": [[0.93]]},
        z=[[2000.0]],
        rng=np.array([200e3]),
    )
    near = far.assign_coords(range=[20e3])
    assert hid(far, melting_layer=ml, engine=engine).HID.item() == names["WS"]
    assert hid(near, melting_layer=ml, engine=engine).HID.item() != names["WS"]
    narrow = hid(far, melting_layer=ml, beamwidth=0.1, engine=engine)
    assert narrow.HID.item() != names["WS"]


@pytest.mark.parametrize("engine", ENGINES)
def test_park_hard_thresholds(engine):
    """
    Table 3: a class failing a hard threshold is never assigned; the next
    highest score is taken instead.
    """
    ds = _random_volume(n=1, nray=100)[0]
    out = hid(ds, quality=False, engine=engine)
    _, table = hidmod._scheme("park", "S")
    x = {0: ds.DBZH.values, 1: ds.ZDR.values, 2: ds.KDP.values, 3: ds.RHOHV.values}
    banned = np.zeros(out.HID_scores.shape, dtype=bool)
    for c, v, op, thr, fsel in table["rules"]:
        limit = thr + hidmod._zfunc(fsel, x[0])
        with np.errstate(invalid="ignore"):
            banned[c] |= (x[v] > limit) if op == 0 else (x[v] < limit)
    cls = out.HID.values
    ok = cls > 0
    idx = np.where(ok, cls - 1, 0)
    assert not np.take_along_axis(banned, idx[None], 0)[0][ok].any()
    allowed = np.where(banned, -1.0, out.HID_scores.values)
    np.testing.assert_allclose(
        out.HID_confidence.values[ok], allowed.max(0)[ok], atol=1e-6
    )
    # the rules matter: the unrestricted argmax is suppressed at many gates
    top = np.nanargmax(np.where(ok[None], out.HID_scores.values, 0.0), axis=0)
    assert (top[ok] != idx[ok]).mean() > 0.02


@pytest.mark.parametrize("engine", ENGINES)
def test_park_confidence_vector(engine):
    """PHIDP, low rhohv and beam blockage change the weights (Park et al. 2009)."""
    ds = _sweep({"DBZH": [[30.0]], "ZDR": [[2.5]], "KDP": [[0.3]], "RHOHV": [[1.0]]})

    def scores(d, **kw):
        return hid(d, engine=engine, **kw).HID_scores.values[:, 0, 0]

    off = scores(ds, quality=False)
    np.testing.assert_allclose(scores(ds), off, atol=1e-7)  # Q = 1 at rhohv = 1
    att = ds.assign(PHIDP_processed=(("azimuth", "range"), [[400.0]]))
    assert not np.allclose(scores(att), off)
    np.testing.assert_allclose(scores(att, quality=False), off, atol=1e-7)
    blk = ds.assign(B=(("azimuth", "range"), [[80.0]]))
    assert not np.allclose(scores(blk, blockage="B"), off)
    low = ds.assign(RHOHV=(("azimuth", "range"), [[0.9]]))
    assert not np.allclose(scores(low), scores(low, quality=False))
    # Eq. (23): no rhohv penalty below 0.8
    vlow = ds.assign(RHOHV=(("azimuth", "range"), [[0.7]]))
    np.testing.assert_allclose(scores(vlow), scores(vlow, quality=False), atol=1e-7)


@pytest.mark.parametrize("engine", ENGINES)
def test_dolan_temperature_membership(engine):
    """Rain at -30 degC is not rain; without temperature it is."""
    rn = hid_classes("dolan", "C")[1][0]
    ds = _sweep(
        {"DBZH": [[39.0, 39.0]], "ZDR": [[2.3, 2.3]], "KDP": [[5.5, 5.5]]},
        extra={"RHOHV": (("azimuth", "range"), [[1.0, 1.0]])},
    )
    ds["T"] = (("azimuth", "range"), [[20.0, -30.0]], {"units": "degC"})
    out = hid(ds, "T", band="C", engine=engine)
    assert out.HID.values[0, 0] == rn
    assert out.HID.values[0, 1] != rn
    assert hid(ds, band="C", engine=engine).HID.values[0, 1] == rn


def _profile(t0=293.15, lapse=0.0065, humid=False):
    height = np.arange(0.0, 16000.0, 250.0)
    temp = t0 - lapse * height
    prof = xr.Dataset(
        {"temperature": ("height", temp, {"units": "K"})},
        coords={"height": height},
    )
    if humid:
        prof["dewpoint"] = ("height", temp - 2.0, {"units": "K"})
        prof["pressure"] = ("height", 101325.0 * np.exp(-height / 8000.0))
    return prof


@pytest.mark.parametrize("engine", ENGINES)
def test_temperature_inputs_are_equivalent(engine):
    ds = _random_volume(n=1)[0].drop_vars("TEMP")
    prof = _profile()
    tk = xr.DataArray(
        prof.temperature.interp(height=ds.z).values,
        dims=("azimuth", "range"),
        attrs={"units": "K"},
    )
    ref = hid(ds, prof, band="C", engine=engine)
    for temperature in (
        prof["temperature"],  # DataArray profile on height
        tk,  # per gate, kelvin from units
        (tk - 273.15).assign_attrs(units="degC"),
        tk.assign_attrs(units=None),  # kelvin detected from the values
    ):
        out = hid(ds, temperature, band="C", engine=engine)
        np.testing.assert_array_equal(out.HID.values, ref.HID.values)
    named = hid(ds.assign(T=tk), "T", band="C", engine=engine)
    np.testing.assert_array_equal(named.HID.values, ref.HID.values)


def test_park_melting_layer_from_temperature():
    """Top at the (wet-bulb) 0 degC height, bottom ml_thickness lower."""
    opts = {"n_threads": None, "dbzh": None}
    prof = _profile()
    bottom, top = hidmod._melting_layer_heights(None, prof, 500.0, [], opts)
    assert top == pytest.approx(20.0 / 0.0065, abs=1.0)
    assert bottom == pytest.approx(top - 500.0)
    wet = _profile(humid=True)
    _, top_wet = hidmod._melting_layer_heights(None, wet, 500.0, [], opts)
    assert top_wet < top  # wet-bulb zero is lower than the 0 degC isotherm
    _, top_c = hidmod._melting_layer_heights(
        None, prof.assign(temperature=prof.temperature - 273.15).temperature.assign_attrs(units="degC"),
        500.0, [], opts,
    )  # fmt: skip
    assert top_c == pytest.approx(top, abs=1.0)
    # from a temperature field per gate
    ds = _random_volume(n=1)[0]
    ds["T"] = (("azimuth", "range"), 20.0 - 0.0065 * ds.z.values, {"units": "degC"})
    _, top_g = hidmod._melting_layer_heights(None, "T", 500.0, [ds], opts)
    assert top_g == pytest.approx(top, abs=100.0)
    # from radarx.retrieve.melting_layer output
    ml = xr.Dataset(
        {"melting_layer_bottom": ("time", [2900.0, 3100.0]),
         "melting_layer_top": ("time", [3400.0, 3600.0])}
    )  # fmt: skip
    assert hidmod._melting_layer_heights(ml, None, 500.0, [], opts) == (3000.0, 3500.0)
    assert hidmod._melting_layer_heights(None, None, 500.0, [], opts) is None
    cold = _profile(t0=250.0)
    assert hidmod._melting_layer_heights(None, cold, 500.0, [], opts) is None
    # no wet-bulb crossing (no humidity data): the 0 degC isotherm is used
    dry = wet.assign(dewpoint=wet.dewpoint * np.nan)
    _, top_dry = hidmod._melting_layer_heights(None, dry, 500.0, [], opts)
    assert top_dry == pytest.approx(top, abs=1.0)
    # gates without heights do not contribute
    nz = ds.drop_vars("z")
    assert hidmod._melting_layer_heights(None, "T", 500.0, [nz], opts) is None
    with pytest.raises(ValueError, match="bottom"):
        hidmod._melting_layer_heights((4000.0, 3000.0), None, 500.0, [], opts)
    # the full call uses the profile for the Park restrictions
    rain = _rain_gate(height=6000.0)
    names = {a: c for c, a, _ in hid_classes("park")}
    assert hid(rain).HID.item() == names["RA"]
    assert hid(rain, prof).HID.item() != names["RA"]


def _winter_grid(ws_gates, rain_like=True):
    """Grid (z, x): rain-like gates at 500 m, wet snow at 1500 m, ice above."""
    nx = 400
    zlev = np.array([500.0, 1500.0, 4000.0])
    dbz = np.full((3, nx), np.nan)
    zdr = np.full((3, nx), np.nan)
    rho = np.full((3, nx), np.nan)
    kdp = np.full((3, nx), np.nan)
    dbz[0], zdr[0], rho[0], kdp[0] = 30.0, 0.5, 0.99, 0.1
    dbz[1, :ws_gates], zdr[1, :ws_gates], rho[1, :ws_gates] = 25.0, 2.0, 0.75
    dbz[1, ws_gates:], zdr[1, ws_gates:], rho[1, ws_gates:] = 20.0, 0.3, 0.99
    dbz[2], zdr[2], kdp[2], rho[2] = 17.0, 2.6, 1.0, 0.99  # dendrites
    temp = np.broadcast_to(np.array([5.0, 0.0, -15.0])[:, None], (3, nx))
    return xr.Dataset(
        {
            "DBZH": (("z", "x"), dbz),
            "ZDR": (("z", "x"), zdr),
            "KDP": (("z", "x"), kdp),
            "RHOHV": (("z", "x"), rho),
            "T": (("z", "x"), temp, {"units": "degC"}),
        },
        coords={"z": zlev, "x": np.arange(nx) * 1000.0},
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_thompson_melting_scenarios(engine):
    """Complete, partial and no melting (Thompson et al. 2014)."""
    names = {a: c for c, a, _ in hid_classes("thompson", "C")}
    kw = dict(band="C", method="thompson", ml_gates=(50, 200), engine=engine)
    full = hid(_winter_grid(300), "T", **kw)
    assert float(full.melting_layer_height) == pytest.approx(1500.0, abs=5.0)
    assert "complete melting" in full.HID.attrs["comment"]
    assert (full.HID.values[0] == names["RN"]).all()
    assert (full.HID.values[1, :300] == names["WS"]).all()
    assert (full.HID.values[2] == names["DN"]).all()
    part = hid(_winter_grid(100), "T", **kw)
    assert "partial melting" in part.HID.attrs["comment"]
    assert (part.HID.values[1, :100] == names["WS"]).all()
    assert not np.isin(part.HID.values[0], [names["RN"], names["FZ"]]).any()
    none = hid(_winter_grid(20), "T", **kw)
    assert "no melting" in none.HID.attrs["comment"]
    assert not np.isin(none.HID.values, [names["WS"], names["RN"]]).any()
    # freezing rain below the melting layer when it is below 0 degC
    cold = _winter_grid(300)
    cold["T"] = cold["T"].copy(
        data=np.broadcast_to(np.array([-4.0, 0.0, -15.0])[:, None], cold.T.shape)
    )
    out = hid(cold, "T", **kw)
    assert (out.HID.values[0] == names["FZ"]).all()
    # never the "other" class
    assert not (out.HID.values == names["OT"]).any()


@pytest.mark.parametrize("engine", ENGINES)
def test_thompson_needs_heights(engine):
    ds = _winter_grid(10).drop_vars("z")
    with pytest.raises(ValueError, match="heights"):
        hid(ds, "T", band="C", method="thompson", engine=engine)


@pytest.mark.parametrize("engine", ENGINES)
def test_grid_and_qvp_inputs(engine):
    """Grids (z, y, x) and QVPs (time, height) keep their coordinates."""
    rnd = np.random.default_rng(1)
    shape = (4, 5, 6)
    grid = xr.Dataset(
        {
            v: (("z", "y", "x"), rnd.uniform(lo, hi, shape))
            for v, lo, hi in (
                ("DBZH", 0, 60),
                ("ZDR", 0, 3),
                ("KDP", 0, 2),
                ("RHOHV", 0.9, 1),
            )
        },
        coords={
            "z": [500.0, 2000.0, 5000.0, 9000.0],
            "y": np.arange(5.0),
            "x": np.arange(6.0),
        },
    )
    prof = _profile()
    out = hid(grid, prof, band="X", engine=engine)
    assert out.HID.dims == ("z", "y", "x")
    assert out.HID_scores.dims == ("hid_class", "z", "y", "x")
    np.testing.assert_array_equal(out.y, grid.y)
    tz = prof.temperature.interp(height=grid.z) - 273.15
    t3 = tz.broadcast_like(grid.DBZH).assign_attrs(units="degC")
    ref = hid(grid, t3, band="X", engine=engine)
    np.testing.assert_array_equal(out.HID.values, ref.HID.values)
    qvp = xr.Dataset(
        {
            v: (("time", "height"), grid[v].values.reshape(20, 6))
            for v in ("DBZH", "ZDR", "KDP", "RHOHV")
        },
        coords={"time": np.arange(20), "height": np.linspace(0, 10000, 6)},
    )
    out = hid(qvp, prof, band="S", method="park", engine=engine)
    assert out.HID.dims == ("time", "height")
    single = hid(qvp.isel(time=3), prof, band="S", method="park", engine=engine)
    assert single.HID.dims == ("height",)
    np.testing.assert_array_equal(single.HID.values, out.HID.values[3])
    # the melting layer restricts the classes of the QVP levels too
    names = {a: c for c, a, _ in hid_classes("park")}
    top = out.HID.values[:, out.height.values > 4000]
    assert not np.isin(top, [names["RA"], names["HR"]]).any()


@pytest.mark.parametrize("engine", ENGINES)
def test_polar_heights_from_beam_geometry(engine):
    """Without 'z', gate heights come from range, elevation and altitude."""
    ds = _rain_gate(height=0.0).drop_vars("z")
    ds = ds.assign_coords(elevation=("azimuth", [10.0]), altitude=500.0)
    ds = ds.assign_coords(range=[30e3])
    names = {a: c for c, a, _ in hid_classes("park")}
    # 30 km at 10 deg: about 5.8 km, above the melting layer
    out = hid(ds, melting_layer=(3000.0, 3500.0), engine=engine)
    assert out.HID.item() != names["RA"]
    with pytest.raises(ValueError, match="heights"):
        hid(ds.drop_vars("elevation"), melting_layer=(3000.0, 3500.0), engine=engine)


@pytest.mark.parametrize("engine", ENGINES)
def test_missing_variables_and_mask(engine):
    """NaN reflectivity or masked gates: class 0; other NaNs are left out."""
    ds = _sweep(
        {
            "DBZH": [[np.nan, 39.0, 39.0, 39.0]],
            "ZDR": [[2.3, np.nan, 2.3, 2.3]],
            "KDP": [[5.5, 5.5, np.nan, 5.5]],
            "RHOHV": [[1.0, 1.0, 1.0, np.nan]],
        }
    )
    mask = xr.DataArray([[True, True, True, False]], dims=("azimuth", "range"))
    out = hid(ds, band="C", mask=mask, engine=engine)
    rn = hid_classes("dolan", "C")[1][0]
    np.testing.assert_array_equal(out.HID.values, [[0, rn, rn, 0]])
    assert np.isnan(out.HID_confidence.values[0, [0, 3]]).all()
    assert np.isnan(out.HID_scores.values[:, 0, 0]).all()
    # only reflectivity: Park aggregation over Z alone
    zonly = hid(ds[["DBZH"]], engine=engine)
    assert zonly.HID.values[0, 1] > 0
    noscore = hid(ds, band="C", scores=False, engine=engine)
    assert "HID_scores" not in noscore


def test_output_attributes_and_coords():
    ds = _random_volume(n=1)[0]
    out = hid(ds, "TEMP", band="C")
    assert out.HID.dtype == np.int8
    assert out.HID.attrs["flag_meanings"].split()[1] == "rain"
    np.testing.assert_array_equal(out.HID.attrs["flag_values"], np.arange(1, 11))
    assert "JAMC-D-12-0275.1" in out.HID.attrs["references"]
    assert out.HID_confidence.attrs["units"] == "1"
    assert list(out.hid_class.values) == [a for _, a, _ in hid_classes("dolan", "C")]
    np.testing.assert_array_equal(out.z, ds.z)
    assert out.HID.dims == ds.DBZH.dims
    assert "DBZH, ZDR, KDP, RHOHV" in out.HID.attrs["comment"]
    x = hid(ds, band="X")
    assert "2009JTECHA1208.1" in x.HID.attrs["references"]
    # transposed input keeps its dimension order
    tr = hid(ds.transpose("range", "azimuth"), "TEMP", band="C")
    assert tr.HID.dims == ("azimuth", "range")
    np.testing.assert_array_equal(tr.HID.values, out.HID.values)


def test_hid_classes():
    assert [a for _, a, _ in hid_classes()] == [
        "DS",
        "WS",
        "CR",
        "GR",
        "BD",
        "RA",
        "HR",
        "RH",
    ]
    assert len(hid_classes("auto", "C")) == 10
    assert len(hid_classes("dolan", "X")) == 7
    assert hid_classes("thompson", "C")[0] == (1, "PL", "plates")
    assert hid_classes("auto", "x") == hid_classes("dolan", "X")


def _volume_with_doppler_cut():
    sweeps = _random_volume(n=2)
    doppler = sweeps[0][["DBZH"]]
    root = xr.Dataset(coords={"latitude": 0.0, "longitude": 0.0, "altitude": 0.0})
    return xr.DataTree.from_dict(
        {"/": root, "sweep_0": sweeps[0], "sweep_1": doppler, "sweep_2": sweeps[1]}
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_datatree_skips_sweeps_and_matches_sweeps(engine):
    tree = _volume_with_doppler_cut()
    out = hid(tree, "TEMP", band="C", engine=engine)
    assert set(out.children) == {"sweep_0", "sweep_2"}
    for name in out.children:
        single = hid(tree[name].to_dataset(), "TEMP", band="C", engine=engine)
        np.testing.assert_array_equal(out[name]["HID"].values, single.HID.values)
    prof = _profile()
    out = tree.radarx.hid(prof, band="S")
    assert set(out.children) == {"sweep_0", "sweep_2"}
    sw = tree["sweep_0"].to_dataset().radarx.hid(prof, band="S")
    np.testing.assert_array_equal(out["sweep_0"]["HID"].values, sw.HID.values)


def _phase_sweep():
    """Sweep with a differential phase increasing through rain, no KDP."""
    nray, ng = 30, 400
    r = 0.25 * (np.arange(ng) + 0.5)
    kdp = 2.0 * np.exp(-0.5 * ((r - 50) / 8) ** 2)
    phi = 20.0 + 2 * np.cumsum(kdp) * 0.25
    shape = (nray, ng)
    return _sweep(
        {
            "DBZH": np.broadcast_to(30 + 8 * kdp, shape),
            "ZDR": np.full(shape, 1.5),
            "RHOHV": np.full(shape, 0.99),
            "PHIDP": np.broadcast_to(phi, shape)
            + np.random.default_rng(0).normal(0, 1, shape),
        },
        rng=1000.0 * r,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_kdp_estimated_when_missing(engine):
    from radarx.retrieve import estimate_kdp

    ds = _phase_sweep()
    out = hid(ds, engine=engine)
    assert "KDP" in out.HID.attrs["comment"]
    est = estimate_kdp(ds, engine=engine)
    ref = hid(
        ds.assign(KDP=est.KDP, PHIDP_processed=est.PHIDP_processed), engine=engine
    )
    np.testing.assert_array_equal(out.HID.values, ref.HID.values)
    tree = xr.DataTree.from_dict(
        {
            "sweep_0": ds,
            "sweep_1": ds.assign(KDP=est.KDP),
            "sweep_2": ds.drop_vars("PHIDP"),
        }
    )
    tout = hid(tree, engine=engine)
    np.testing.assert_array_equal(tout["sweep_0"]["HID"].values, ref.HID.values)
    assert "KDP" not in tout["sweep_2"]["HID"].attrs["comment"]


def test_errors(monkeypatch):
    ds = _random_volume(n=1)[0]
    with pytest.raises(ValueError, match="band"):
        hid(ds, band="K")
    with pytest.raises(ValueError, match="method"):
        hid(ds, method="foo")
    with pytest.raises(ValueError, match="S band only"):
        hid(ds, band="C", method="park")
    with pytest.raises(ValueError, match="engine"):
        hid(ds, engine="gpu")
    with pytest.raises(ValueError, match="ml_gates"):
        hid(ds, method="thompson", ml_gates=(100, 10))
    with pytest.raises(KeyError):
        hid(ds, dbzh="nope")
    with pytest.raises(KeyError):
        hid(ds.drop_vars("DBZH"))
    with pytest.raises(KeyError, match="temperature"):
        hid(ds, "nope", band="C")
    with pytest.raises(TypeError):
        hid(ds, temperature=3.0, band="C")
    with pytest.raises(KeyError, match="temperature"):
        hid(
            ds, xr.Dataset({"t": ("height", [1.0])}, coords={"height": [0.0]}), band="C"
        )
    tree = _volume_with_doppler_cut()
    with pytest.raises(TypeError, match="volume"):
        hid(tree, ds.TEMP, band="C")
    with pytest.raises(TypeError, match="volume"):
        hid(tree, mask=ds.MET)
    with pytest.raises(TypeError):
        hid(ds, xr.DataTree())
    with pytest.raises(KeyError, match="no sweep"):
        hid(xr.DataTree.from_dict({"sweep_0": ds[["DBZH"]]}))
    monkeypatch.setattr(hidmod, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError):
        hid(ds, engine="compiled")
    out = hid(ds, "TEMP", band="C")  # falls back to NumPy
    assert out.HID.dtype == np.int8


@pytest.mark.skipif(not hidmod.HAS_COMPILED_KERNEL, reason="compiled kernel not built")
def test_kernel_input_checks():
    classes, t = hidmod._scheme("dolan", "C")
    one = [np.zeros((1, 2))]
    none = [None]
    empty = np.zeros(0, np.int64)

    def call(**over):
        args = dict(
            zh=one, zdr=none, kdp=none, rhohv=none, temperature=none, phidp=none,
            blockage=none, valid=none, height=none, range=none, ml_bottom=none,
            ml_top=none, kind=t["kind"], par=t["par"], fsel=t["fsel"],
            weight=t["weight"], group=t["group"], r_class=empty, r_var=empty,
            r_op=empty, r_fsel=empty, r_thr=np.zeros(0), zone_allowed=None,
            mode=t["mode"], kdp_log=False, quality=False, sin_half_beam=0.0,
            ws_class=-1, ot_class=-1, ml_gates_partial=0, ml_gates_complete=0,
            stats_rmin=0.0, stats_rmax=0.0, hist_lo=0.0, hist_bin=1.0, hist_n=1,
        )  # fmt: skip
        args.update(over)
        return hidmod._hid.classify(**args)

    assert call()[0][0].shape == (1, 2)
    bad = [
        dict(zdr=[None, None]),
        dict(mode=5),
        dict(kind=np.zeros((2, 3), np.int64)),
        dict(par=np.zeros(3)),
        dict(zh=[np.zeros(3)]),
        dict(zdr=[np.zeros(3)]),
        dict(valid=[np.zeros(3, np.uint8)]),
        dict(zone_allowed=np.zeros(3, np.int64)),
        dict(r_class=np.zeros(1, np.int64)),
        dict(
            r_class=np.array([99]),
            r_var=np.array([0]),
            r_op=np.array([0]),
            r_fsel=np.array([0]),
            r_thr=np.zeros(1),
        ),  # fmt: skip
        dict(mode=2),
        dict(mode=2, ws_class=0, ot_class=1, hist_bin=0.0),
        dict(mode=2, ws_class=0, ot_class=1),
        dict(kind=np.full_like(t["kind"], 7)),
        dict(fsel=np.full_like(t["fsel"], 9)),
        dict(par=np.zeros_like(t["par"])),
    ]
    for over in bad:
        with pytest.raises(ValueError):
            call(**over)


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------


def _clean_nexrad(ds):
    """Mask the NEXRAD no-data codes."""
    out = {}
    for v, lim in (("DBZH", -32), ("ZDR", -12.9), ("RHOHV", 0.21)):
        if v in ds:
            out[v] = ds[v].where(ds[v] > lim)
    return ds.assign(out)


@pytest.fixture(scope="module")
def klbb():
    xd = pytest.importorskip("xradar")
    from open_radar_data import DATASETS

    file = DATASETS.fetch("KLBB20160601_150025_V06")
    dtree = xd.io.open_nexradlevel2_datatree(file, sweep=[0, 1, 2, 3, 4, 5])
    nodes = {"/": dtree.root.to_dataset(inherit=False)}
    for name in dtree.children:
        nodes[name] = _clean_nexrad(dtree[name].to_dataset(inherit=False))
    return xr.DataTree.from_dict(nodes).xradar.georeference()


def test_real_nexrad_volume(klbb):
    """S-band volume: rain below the melting layer, snow and ice above it."""
    prof = _profile(t0=303.15)  # 0 degC near 4.7 km (June, west Texas)
    out = klbb.radarx.hid(prof, band="S")
    assert "sweep_1" not in out.children  # Doppler cut without ZDR
    names = {a: c for c, a, _ in hid_classes("park")}
    rain = [names[a] for a in ("BD", "RA", "HR", "RH")]
    ice = [names[a] for a in ("DS", "CR", "GR")]
    low, high, n = [], [], 0
    for name in out.children:
        cls = out[name]["HID"].values
        z = klbb[name]["z"].values
        n += (cls > 0).sum()
        low.append(cls[(cls > 0) & (z < 3000)])
        high.append(cls[(cls > 0) & (z > 6000)])
    low, high = np.concatenate(low), np.concatenate(high)
    assert n > 100_000
    assert np.isin(low, rain).mean() > 0.95
    assert np.isin(high, ice + [names["RH"]]).mean() > 0.95
    if hidmod.HAS_COMPILED_KERNEL:
        ref = hid(klbb["sweep_0"].to_dataset(), prof, band="S", engine="numpy")
        np.testing.assert_array_equal(out["sweep_0"]["HID"].values, ref.HID.values)


@pytest.fixture(scope="module")
def csapr2():
    xd = pytest.importorskip("xradar")
    from open_radar_data import DATASETS

    file = DATASETS.fetch("corcsapr2cmacppiM1.c1.20181111.030003.nc")
    dtree = xd.io.open_cfradial1_datatree(file)
    ds = dtree["sweep_0"].to_dataset(inherit="all_coords").load()
    met = ds.gate_id.isin([1, 2, 4]) & ds.attenuation_corrected_reflectivity_h.notnull()
    return ds.assign(met=met)


CSAPR2_FIELDS = dict(
    dbzh="attenuation_corrected_reflectivity_h",
    zdr="attenuation_corrected_differential_reflectivity",
    rhohv="copol_correlation_coeff",
    kdp="specific_differential_phase",
)


def test_real_csapr2_c_band(csapr2):
    """C-band convection (0.5 deg): drizzle and rain, a little hail and big drops."""
    out = csapr2.radarx.hid(
        "sounding_temperature", band="C", mask="met", **CSAPR2_FIELDS
    )
    names = {a: c for c, a, _ in hid_classes("dolan", "C")}
    cls = out.HID.values[csapr2.met.values]
    assert (cls > 0).all()
    liquid = np.isin(cls, [names[a] for a in ("DZ", "RN", "BD", "HA")])
    assert liquid.mean() > 0.95
    z = csapr2.attenuation_corrected_reflectivity_h.values
    core = out.HID.values[(z > 50) & csapr2.met.values]
    assert (
        np.isin(core, [names["RN"], names["HA"], names["BD"], names["HDG"]]).mean()
        > 0.95
    )
    assert (out.HID.values[~csapr2.met.values] == 0).all()


def test_against_csu_radartools(csapr2):
    """Validation baseline (only when CSU_RadarTools is installed)."""
    csu_fhc = pytest.importorskip("csu_radartools.csu_fhc")
    ds = csapr2
    out = hid(ds, "sounding_temperature", band="C", mask="met", **CSAPR2_FIELDS)
    ref = csu_fhc.csu_fhc_summer(
        dz=ds[CSAPR2_FIELDS["dbzh"]].values,
        zdr=ds[CSAPR2_FIELDS["zdr"]].values,
        rho=ds[CSAPR2_FIELDS["rhohv"]].values,
        kdp=ds[CSAPR2_FIELDS["kdp"]].values,
        T=ds.sounding_temperature.values,
        use_temp=True,
        band="C",
    )
    order = ["drizzle", "rain", "ice_crystals", "aggregates", "wet_snow",
             "vertically_aligned_ice", "low_density_graupel", "high_density_graupel",
             "hail", "big_drops"]  # fmt: skip
    ours = [n for _, _, n in hid_classes("dolan", "C")]
    mapped = np.array([0] + [ours.index(n) + 1 for n in order])[np.asarray(ref, int)]
    m = ds.met.values
    assert np.mean(mapped[m] == out.HID.values[m]) > 0.85
