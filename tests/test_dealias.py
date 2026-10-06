#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for radarx Doppler velocity dealiasing
============================================
"""
import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.retrieve import dealias, dealias_velocity

ENGINES = ["numpy"] + (["compiled"] if dealias.HAS_COMPILED_KERNEL else [])
SITE_ALT = 500.0


def wind(x, y, z):
    """Uniform wind plus a Rankine vortex and a convergence zone (m/s)."""
    u = 12.0 + 0.002 * z
    v = 18.0 + 0.0 * z
    # Rankine vortex 40 km east, 30 km north, radius 6 km, 25 m/s
    dx, dy = x - 40e3, y - 30e3
    r = np.hypot(dx, dy) + 1e-9
    vt = np.where(r < 6e3, 25.0 * r / 6e3, 25.0 * 6e3 / r)
    u = u - vt * dy / r
    v = v + vt * dx / r
    # convergence toward a line 20 km west of the radar
    conv = -10.0 * np.tanh((x + 20e3) / 8e3)
    return u + conv, v


def synthetic_sweep(
    nyquist=10.0,
    noise=1.0,
    gaps=0.05,
    elevation=0.5,
    seed=0,
    nray=360,
    ngate=400,
    first_azimuth=37.5,
):
    """Aliased PPI sweep and its true radial velocity."""
    rnd = np.random.default_rng(seed)
    azimuth = np.mod(first_azimuth + np.arange(nray) * 360.0 / nray, 360.0)
    rng = 125.0 + 250.0 * np.arange(ngate)
    az = np.deg2rad(azimuth)[:, None]
    el = np.deg2rad(elevation)
    r_eff = 6371000.0 * 4.0 / 3.0
    z = np.sqrt(rng**2 + r_eff**2 + 2 * rng * r_eff * np.sin(el)) - r_eff + SITE_ALT
    s = rng * np.cos(el)
    x, y = s * np.sin(az), s * np.cos(az)
    u, v = wind(x, y, z[None, :])
    truth = (u * np.sin(az) + v * np.cos(az)) * np.cos(el)
    measured = truth + noise * rnd.standard_normal(truth.shape)
    aliased = np.mod(measured + nyquist, 2 * nyquist) - nyquist
    missing = rnd.random(truth.shape) < gaps
    missing[100:110, 50:300] = True  # a block of missing data
    missing[:, 380:] = True  # beyond the echo
    aliased[missing] = np.nan
    ds = xr.Dataset(
        {
            "VRADH": (
                ("azimuth", "range"),
                aliased,
                {
                    "standard_name": "radial_velocity_of_scatterers_away_from_instrument_h",
                    "long_name": "Radial velocity of scatterers away from instrument H",
                    "units": "m s-1",
                },
            )
        },
        coords={
            "azimuth": ("azimuth", azimuth, {"units": "degrees"}),
            "range": ("range", rng, {"units": "meters"}),
            "elevation": ("azimuth", np.full(nray, elevation), {"units": "degrees"}),
            "nyquist_velocity": ((), nyquist, {"units": "m s-1"}),
            "altitude": ((), SITE_ALT),
            "latitude": ((), 33.0),
            "longitude": ((), -101.0),
        },
    )
    return ds, truth


def correct_fraction(out, ds, truth):
    """Share of valid gates whose fold is right (error below Vn)."""
    nyq = float(ds["nyquist_velocity"])
    valid = np.isfinite(ds["VRADH"].values)
    return np.mean(np.abs(out.values - truth)[valid] < nyq)


def wind_profile():
    height = np.arange(0.0, 20e3, 250.0)
    return xr.Dataset(
        {
            "u": ("height", 12.0 + 0.002 * height),
            "v": ("height", np.full(height.size, 18.0)),
        },
        coords={"height": height},
    )


def volume(elevations=(0.5, 1.5, 3.0), **kwargs):
    """Synthetic volume (DataTree) of aliased sweeps and their truths."""
    sweeps, truths = {}, {}
    for i, el in enumerate(elevations):
        ds, truth = synthetic_sweep(elevation=el, seed=i, **kwargs)
        sweeps[f"sweep_{i}"] = ds.drop_vars(["latitude", "longitude", "altitude"])
        truths[f"sweep_{i}"] = truth
    root = xr.Dataset(
        coords={"latitude": 33.0, "longitude": -101.0, "altitude": SITE_ALT}
    )
    return xr.DataTree.from_dict({"/": root, **sweeps}), truths


def residual_jumps(values, nyquist):
    """Share of neighbouring valid gate pairs that differ by more than Vn."""
    bad = total = 0
    for a, b in ((values[:, 1:], values[:, :-1]), (np.roll(values, -1, 0), values)):
        ok = np.isfinite(a) & np.isfinite(b)
        bad += np.count_nonzero(np.abs(a - b)[ok] > nyquist)
        total += np.count_nonzero(ok)
    return bad / max(total, 1)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "nyquist, noise, minimum",
    [
        (30.0, 1.0, 1.0),
        (15.0, 1.0, 1.0),
        (10.0, 1.0, 1.0),
        (7.0, 1.0, 1.0),
        (7.0, 2.0, 0.995),
    ],
)
def test_synthetic_accuracy(engine, nyquist, noise, minimum):
    ds, truth = synthetic_sweep(nyquist=nyquist, noise=noise)
    out = dealias_velocity(ds, engine=engine)
    assert correct_fraction(out, ds, truth) >= minimum
    if nyquist <= 10:  # several folds really occur
        assert np.nanmax(np.abs(truth)) > 3 * nyquist


@pytest.mark.skipif(not dealias.HAS_COMPILED_KERNEL, reason="compiled kernel not built")
@pytest.mark.parametrize("nyquist, noise", [(10.0, 1.0), (7.0, 2.0), (7.0, 3.0)])
def test_engines_identical(nyquist, noise):
    ds, _ = synthetic_sweep(nyquist=nyquist, noise=noise, gaps=0.2, ngate=200)
    options = (
        {},
        {"wind_profile": wind_profile()},
        {"reference": ds["VRADH"] * 0 + 5.0},
    )
    for kwargs in options:
        a = dealias_velocity(ds, engine="compiled", **kwargs)
        b = dealias_velocity(ds, engine="numpy", **kwargs)
        np.testing.assert_array_equal(a.values, b.values)
    tree, _ = volume(nyquist=nyquist, noise=noise)
    for continuity in (True, False):
        a = dealias_velocity(tree, engine="compiled", sweep_continuity=continuity)
        b = dealias_velocity(tree, engine="numpy", sweep_continuity=continuity)
        for n in ("sweep_0", "sweep_1", "sweep_2"):
            np.testing.assert_array_equal(
                a[n]["VRADH_dealiased"].values, b[n]["VRADH_dealiased"].values
            )


@pytest.mark.skipif(not dealias.HAS_COMPILED_KERNEL, reason="compiled kernel not built")
def test_thread_count_does_not_change_result():
    tree, _ = volume(nyquist=7.0, noise=2.0)
    a = dealias_velocity(tree, n_threads=1)
    b = dealias_velocity(tree, n_threads=0)
    for n in ("sweep_0", "sweep_1", "sweep_2"):
        np.testing.assert_array_equal(
            a[n]["VRADH_dealiased"].values, b[n]["VRADH_dealiased"].values
        )


@pytest.mark.parametrize("engine", ENGINES)
def test_reference_fixes_absolute_fold(engine):
    # echo only in a sector where the wind blows away from the radar: without
    # a reference the zero-mean assumption picks the wrong fold
    ds, truth = synthetic_sweep(nyquist=10.0, noise=0.5)
    az = xr.DataArray(ds["azimuth"].values, dims="azimuth")
    rng = xr.DataArray(ds["range"].values, dims="range")
    ds["VRADH"] = ds["VRADH"].where((az > 0) & (az < 80) & (rng < 50e3))
    blind = dealias_velocity(ds, engine=engine)
    assert correct_fraction(blind, ds, truth) < 0.5
    out = dealias_velocity(ds, wind_profile=wind_profile(), engine=engine)
    assert correct_fraction(out, ds, truth) == 1.0
    ref = xr.DataArray(truth, dims=("azimuth", "range")) + 3.0
    out = dealias_velocity(ds, reference=ref, engine=engine)
    assert correct_fraction(out, ds, truth) == 1.0


@pytest.mark.parametrize("engine", ENGINES)
def test_volume_continuity(engine):
    tree, truths = volume(nyquist=8.0, noise=1.0)
    out = dealias_velocity(tree, engine=engine)
    for n, truth in truths.items():
        ds = tree[n].to_dataset()
        # products only: the dealiased velocity, never the measured field
        assert list(out[n].data_vars) == ["VRADH_dealiased"]
        assert correct_fraction(out[n]["VRADH_dealiased"], ds, truth) == 1.0
    xr.testing.assert_identical(out.to_dataset(), tree.to_dataset())
    named = dealias_velocity(tree, engine=engine, name="VR")
    assert list(named["sweep_0"].data_vars) == ["VR"]


def test_products_merge_into_the_volume():
    tree, _ = volume()
    products = tree.radarx.dealias()
    merged = tree.radarx.assign(products)
    for n in ("sweep_0", "sweep_1", "sweep_2"):
        assert {"VRADH", "VRADH_dealiased"} <= set(merged[n].data_vars)
        xr.testing.assert_identical(merged[n]["VRADH"], tree[n]["VRADH"])
        np.testing.assert_array_equal(
            merged[n]["VRADH_dealiased"].values, products[n]["VRADH_dealiased"].values
        )
    assert "VRADH_dealiased" not in tree["sweep_0"].data_vars  # input unchanged
    ds, _ = synthetic_sweep()
    vr = dealias_velocity(ds)
    assert vr.name == "VRADH_dealiased"
    merged = ds.radarx.assign(vr)
    xr.testing.assert_identical(merged["VRADH"], ds["VRADH"])
    np.testing.assert_array_equal(merged["VRADH_dealiased"].values, vr.values)


def test_old_return_style_warns():
    ds, _ = synthetic_sweep()
    with pytest.warns(FutureWarning, match="products_only"):
        old = dealias_velocity(ds, products_only=False)
    assert old.name == "VRADH"
    np.testing.assert_array_equal(old.values, dealias_velocity(ds).values)
    tree, _ = volume()
    with pytest.warns(FutureWarning, match="products_only"):
        old = dealias_velocity(tree, products_only=False)
    new = dealias_velocity(tree)
    for n in ("sweep_0", "sweep_1", "sweep_2"):
        # the whole sweep, with the measured field replaced
        assert set(old[n].data_vars) == set(tree[n].data_vars)
        np.testing.assert_array_equal(
            old[n]["VRADH"].values, new[n]["VRADH_dealiased"].values
        )
    with pytest.warns(FutureWarning):
        kept = tree.radarx.dealias(products_only=False, name="VRADH_dealiased")
    np.testing.assert_array_equal(
        kept["sweep_0"]["VRADH"].values, tree["sweep_0"]["VRADH"].values
    )


def test_output_attributes_coords_and_dims():
    ds, _ = synthetic_sweep()
    ds = ds.assign_coords(time=("azimuth", np.arange(ds.sizes["azimuth"])))
    out = dealias_velocity(ds)
    assert out.dims == ds["VRADH"].dims
    assert set(ds["VRADH"].coords) <= set(out.coords)
    assert out.attrs["units"] == "m s-1"
    assert out.attrs["standard_name"].startswith("radial_velocity")
    assert out.attrs["nyquist_velocity"] == 10.0
    assert "Dealiased" in out.attrs["long_name"]
    # transposed input gives the transposed result
    tr = dealias_velocity(ds.transpose("range", "azimuth"))
    assert tr.dims == ("range", "azimuth")
    np.testing.assert_array_equal(tr.values.T, out.values)
    # float32 input stays float32
    assert dealias_velocity(ds.astype({"VRADH": "float32"})).dtype == np.float32


def test_nyquist_sources_and_flags():
    ds, _ = synthetic_sweep()
    meta = dealias_velocity(ds)
    no_meta = ds.drop_vars("nyquist_velocity")
    with pytest.raises(ValueError, match="Nyquist"):
        dealias_velocity(no_meta)
    np.testing.assert_array_equal(
        dealias_velocity(no_meta, nyquist_velocity=10.0), meta
    )
    attr = no_meta.copy()
    attr["VRADH"].attrs["nyquist_velocity"] = 10.0
    np.testing.assert_array_equal(dealias_velocity(attr), meta)
    with pytest.raises(ValueError, match="positive"):
        dealias_velocity(ds, nyquist_velocity=-1.0)
    # values beyond the Nyquist velocity are flags (e.g. range folded)
    flagged = ds.copy(deep=True)
    flagged["VRADH"][5, 5] = -64.5
    assert np.isnan(dealias_velocity(flagged)[5, 5])
    with pytest.raises(ValueError, match="engine"):
        dealias_velocity(ds, engine="fortran")
    tree, _ = volume()
    with pytest.raises(ValueError, match="single sweep"):
        dealias_velocity(tree, reference=ds["VRADH"])
    with pytest.raises(ValueError, match="No sweep"):
        dealias_velocity(tree, "DBZH")


@pytest.mark.parametrize("engine", ENGINES)
def test_empty_and_sector_sweeps(engine):
    ds, _ = synthetic_sweep()
    empty = ds.copy(deep=True)
    empty["VRADH"][:] = np.nan
    assert np.isnan(dealias_velocity(empty, engine=engine)).all()
    # a 90 degree sector scan has no link between its first and last ray
    sector, _ = synthetic_sweep(nray=90, first_azimuth=10.0)
    sector = sector.assign_coords(azimuth=("azimuth", 10.0 + np.arange(90.0)))
    assert dealias._ray_links(sector["azimuth"].values)[-1] == 0
    out = dealias_velocity(sector, engine=engine)
    assert out.shape == sector["VRADH"].shape


def test_accessors():
    ds, _ = synthetic_sweep()
    np.testing.assert_array_equal(ds.radarx.dealias(), dealias_velocity(ds))
    tree, _ = volume()
    out = tree.radarx.dealias("VRADH", sweep_continuity=False)
    expected = dealias_velocity(tree, sweep_continuity=False)
    for n in ("sweep_0", "sweep_1", "sweep_2"):
        np.testing.assert_array_equal(
            out[n]["VRADH_dealiased"].values, expected[n]["VRADH_dealiased"].values
        )


# --------------------------------------------------------------------------
# real NEXRAD volumes
# --------------------------------------------------------------------------


def nexrad_volume(path):
    """NEXRAD Level II volume with each sweep's Nyquist velocity attached."""
    import xradar as xd
    from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

    with NEXRADLevel2File(path) as nf:
        nyquist = [
            h["msg_31_data_header"]["RAD"]["nyquist_vel"] / 100.0
            for h in nf.msg_31_data_header
        ]
    dtree = xd.io.open_nexradlevel2_datatree(path)
    names = [n for n in dtree.children if n.startswith("sweep")]
    for i, name in enumerate(names):
        dtree[name] = (
            dtree[name].to_dataset().assign_coords(nyquist_velocity=nyquist[i])
        )
    return dtree


def compare_with_pyart(path, dtree, out):
    """Per sweep: share of gates with Py-ART's fold, and residual jumps."""
    pyart = pytest.importorskip("pyart")
    radar = pyart.io.read_nexrad_archive(path)
    corrected = pyart.correct.dealias_region_based(radar)["data"]
    names = [n for n in dtree.children if n.startswith("sweep")]
    rows = {}
    for i, name in enumerate(names):
        if "VRADH" not in dtree[name].data_vars:
            continue
        ds = out[name].to_dataset()
        nyquist = float(ds["nyquist_velocity"])
        sl = radar.get_slice(i)
        order = np.argsort(radar.azimuth["data"][sl])
        theirs = np.ma.filled(corrected[sl], np.nan)[order]
        ours = ds["VRADH_dealiased"].values[np.argsort(ds["azimuth"].values)]
        ng = min(theirs.shape[1], ours.shape[1])
        theirs, ours = theirs[:, :ng], ours[:, :ng]
        both = np.isfinite(theirs) & np.isfinite(ours)
        same = np.mean(np.abs(theirs - ours)[both] < 1.0)
        rows[name] = (
            same,
            residual_jumps(ours, nyquist),
            residual_jumps(theirs, nyquist),
        )
    return rows


@pytest.fixture(scope="module")
def klbb():
    from open_radar_data import DATASETS

    path = DATASETS.fetch("KLBB20160601_150025_V06")
    return path, nexrad_volume(path)


@pytest.fixture(scope="module")
def kgwx(tmp_path_factory):
    """Strongly aliased low-level jet (KGWX, 30 March 2022), from AWS."""
    from radarx.io.aws_data import download_file

    try:
        path = download_file(
            "unidata-nexrad-level2",
            "2022/03/30/KGWX/KGWX20220330_234639_V06",
            str(tmp_path_factory.mktemp("nexrad")),
        )
    except Exception as err:  # noqa: BLE001  # pragma: no cover - network
        pytest.skip(f"NEXRAD data not available: {err}")
    return path, nexrad_volume(path)


@pytest.mark.parametrize("case", ["klbb", "kgwx"])
def test_nexrad_engines_and_jumps(case, request):
    _, dtree = request.getfixturevalue(case)
    out = dealias_velocity(dtree)
    for name in dtree.children:
        if "VRADH" not in dtree[name].data_vars:
            continue
        nyquist = float(dtree[name]["nyquist_velocity"])
        raw = dtree[name]["VRADH"].values
        raw = np.where(np.abs(raw) <= 1.01 * nyquist, raw, np.nan)
        ours = out[name]["VRADH_dealiased"].values
        assert residual_jumps(ours, nyquist) <= residual_jumps(raw, nyquist)
        assert residual_jumps(ours, nyquist) < 0.005
    if dealias.HAS_COMPILED_KERNEL and case == "klbb":
        ref = dealias_velocity(dtree, engine="numpy")
        for name in dtree.children:
            if "VRADH" in dtree[name].data_vars:
                np.testing.assert_array_equal(
                    out[name]["VRADH_dealiased"].values,
                    ref[name]["VRADH_dealiased"].values,
                )


@pytest.mark.parametrize("case", ["klbb", "kgwx"])
def test_nexrad_matches_pyart(case, request):
    path, dtree = request.getfixturevalue(case)
    out = dealias_velocity(dtree)
    rows = compare_with_pyart(path, dtree, out)
    assert rows
    for name, (same, ours, theirs) in rows.items():
        # Py-ART's region-based fold almost everywhere, and no more residual
        # jumps above the Nyquist velocity than Py-ART leaves
        assert same > 0.99, (name, same)
        assert ours <= theirs + 2e-4, (name, ours, theirs)
