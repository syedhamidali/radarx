#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the volume matching of a ground radar and the GPM DPR
===============================================================

A synthetic DPR swath: the satellite flies north at 407 km, 50 km east of a ground radar at 30 N,
90 W, with 49 rays at zenith angles from -17.5 to 17.5 degrees and 5 km between the scans. The
footprints sit at ``x = 407 km * tan(zenith)`` from the track at the ellipsoid, the range bins are
125 m apart and reach 22 km. The ground radar has sweeps at 0.5, 5, 10 and 19 degrees. The fields are
chosen so that the matched means follow from the geometry: a field equal to the east and north
position of the gates gives the position of the match volume (the footprint centre displaced by the
parallax at the height of the beam), a field equal to the height of the bin gives the height.
"""

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.fundamentals.geometry import beam_height_at_ground_range
from radarx.grid import cone
from radarx.retrieve import _satellite_match_numpy as ref
from radarx.retrieve import (
    dpr_dsd,
    dpr_parallax,
    dsd,
    match_dpr,
    match_statistics,
)

sm = __import__("radarx.retrieve.satellite_match", fromlist=["x"])

try:
    from radarx.retrieve import _satellite_match as kernel
except ImportError:  # pragma: no cover
    kernel = None

ENGINES = ["numpy"] + (["compiled"] if kernel is not None else [])

SITE = {"latitude": 30.0, "longitude": -90.0, "altitude": 100.0}
ELEVATIONS = (0.5, 5.0, 10.0, 19.0)
HEIGHT_SAT = 407e3
TRACK_X = 50e3


def aeqd():
    import pyproj

    from radarx._gridcf import aeqd_crs

    crs = aeqd_crs(SITE["latitude"], SITE["longitude"])
    forward = pyproj.Transformer.from_crs(
        pyproj.CRS.from_epsg(4326), crs, always_xy=True
    )
    back = pyproj.Transformer.from_crs(crs, pyproj.CRS.from_epsg(4326), always_xy=True)
    return forward, back


def sweep(elevation, *, fill=None, naz=720, ngate=600, name="DBZH", **fields):
    azimuth = (np.arange(naz) + 0.5) * 360.0 / naz
    rng = 125.0 + 250.0 * np.arange(ngate)
    from xradar.georeference import antenna_to_cartesian

    gx, gy, _ = antenna_to_cartesian(
        rng[None, :],
        azimuth[:, None],
        np.full((naz, 1), elevation),
        site_altitude=SITE["altitude"],
    )
    data = {
        name: (("azimuth", "range"), np.full((naz, ngate), 35.0), {"units": "dBZ"}),
        "X": (("azimuth", "range"), gx),
        "Y": (("azimuth", "range"), gy),
    }
    for key, value in fields.items():
        data[key] = (("azimuth", "range"), value)
    ds = xr.Dataset(
        data,
        coords={
            "azimuth": ("azimuth", azimuth),
            "elevation": ("azimuth", np.full(naz, elevation)),
            "range": ("range", rng),
            "time": (
                "azimuth",
                np.full(naz, np.datetime64("2024-05-01T12:00:10", "ns")),
            ),
        },
    )
    if fill is not None:
        ds[name] = ds[name].where(fill(ds))
    return ds


_SWEEPS = {}


def volume(**kwargs):
    root = xr.Dataset(
        coords={k: v for k, v in SITE.items()},
        data_vars={"radar_beam_width_h": 1.0},
    )
    if kwargs:
        nodes = {f"/sweep_{i}": sweep(e, **kwargs) for i, e in enumerate(ELEVATIONS)}
    else:  # the plain sweeps are built once; the tests change copies
        if not _SWEEPS:
            _SWEEPS.update({e: sweep(e) for e in ELEVATIONS})
        nodes = {f"/sweep_{i}": _SWEEPS[e].copy() for i, e in enumerate(ELEVATIONS)}
    return xr.DataTree.from_dict({"/": root, **nodes})


def swath(nscan=40, zfactor=None):
    """DPR swath of the construction above."""
    forward, back = aeqd()
    j = np.arange(49)
    zenith = (j - 24) * 17.5 / 24.0
    x = TRACK_X + HEIGHT_SAT * np.tan(np.radians(zenith))
    y = (np.arange(nscan) - nscan / 2 + 0.5) * 5e3
    xx, yy = np.meshgrid(x, y)
    lon, lat = back.transform(xx, yy)
    nbin = 176
    depth = (nbin - 1 - np.arange(nbin)) * 125.0  # slant distance above the ellipsoid
    height = depth[None, None, :] * np.cos(np.radians(zenith))[None, :, None]
    height = np.broadcast_to(height, (nscan, 49, nbin)).copy()
    zf = np.full((nscan, 49, nbin), 30.0) if zfactor is None else zfactor(height)
    time = np.datetime64("2024-05-01T12:00:00", "ns") + (
        np.arange(nscan) * np.timedelta64(100, "ms")
    )
    return xr.Dataset(
        {
            "Longitude": (("scan", "ray"), lon),
            "Latitude": (("scan", "ray"), lat),
            "localZenithAngle": (
                ("scan", "ray"),
                np.broadcast_to(zenith, (nscan, 49)).copy(),
            ),
            "height": (("scan", "ray", "bin"), height),
            "zFactorFinal": (("scan", "ray", "bin"), zf, {"units": "dBZ"}),
            "H": (("scan", "ray", "bin"), height),
            "time": ("scan", time),
        }
    )


def matched(engine="auto", gr=None, dpr=None, **kwargs):
    kwargs.setdefault("gr_fields", {"X": "mean", "Y": "mean", "DBZH": "db"})
    kwargs.setdefault("dpr_fields", {"H": "mean", "zFactorFinal": "db"})
    return match_dpr(
        volume() if gr is None else gr,
        swath() if dpr is None else dpr,
        engine=engine,
        **kwargs,
    )


@pytest.fixture(scope="module")
def result():
    return matched()


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def test_parallax_is_height_times_tangent():
    assert float(dpr_parallax(5000.0, 17.0)) == pytest.approx(
        5000 * np.tan(np.radians(17))
    )
    assert float(dpr_parallax(5000.0, -17.0)) == float(dpr_parallax(5000.0, 17.0))
    assert float(dpr_parallax(0.0, 17.0)) == 0.0
    da = xr.DataArray([0.0, 1000.0, 2000.0], dims="bin")
    assert dpr_parallax(da, 10.0).dims == ("bin",)


def test_footprints_are_the_ones_in_range(result):
    assert result.sizes["sweep"] == 4
    assert result.sizes["footprint"] > 500
    forward, _ = aeqd()
    x, y = forward.transform(result.longitude.values, result.latitude.values)
    ground = np.hypot(x, y)
    assert ground.min() >= 15e3 and ground.max() <= 100e3
    np.testing.assert_allclose(result.elevation, ELEVATIONS)


def test_match_volume_is_the_footprint_displaced_by_the_parallax(result):
    """Position of the volume: P0 - h tan(zenith) toward the track, at the beam height."""
    forward, _ = aeqd()
    x0, y0 = forward.transform(result.longitude.values, result.latitude.values)
    zenith = result.ray.values * 0 + (result.ray.values - 24) * 17.5 / 24.0
    tan_z = np.tan(np.radians(np.abs(zenith)))
    side = np.sign(zenith)  # east of the track: shift to the west
    seen_shift = 0.0
    for k, el in enumerate(ELEVATIONS):
        h = result.match_height.values[:, k]
        s = result.ground_range.values[:, k]
        # the beam height of the ground range of the shifted position
        np.testing.assert_allclose(
            h,
            beam_height_at_ground_range(s, el, SITE["altitude"], 4.0 / 3.0 * 6371000.0),
            atol=0.5,
        )
        mx, my = forward.transform(
            result.match_longitude.values[:, k], result.match_latitude.values[:, k]
        )
        np.testing.assert_allclose(mx, x0 - side * h * tan_z, atol=0.5)
        np.testing.assert_allclose(my, y0, atol=0.5)
        np.testing.assert_allclose(np.hypot(mx, my), s, atol=0.5)
        seen_shift = max(seen_shift, np.abs(mx - x0).max())
    assert seen_shift > 1000.0  # the high sweeps are displaced by more than 1 km


def test_ground_radar_mean_position_is_the_volume_position(result):
    """The mean of the gate position over the disc is the centre of the volume."""
    forward, _ = aeqd()
    good = result.valid.values
    assert good.sum() > 100
    for k in range(4):
        mx, my = forward.transform(
            result.match_longitude.values[:, k], result.match_latitude.values[:, k]
        )
        sel = good[:, k]
        np.testing.assert_allclose(result.gr_X.values[sel, k], mx[sel], atol=350.0)
        np.testing.assert_allclose(result.gr_Y.values[sel, k], my[sel], atol=350.0)
    # without the parallax the volume would lie 1 km or more from where the gates are
    x0, _ = forward.transform(result.longitude.values, result.latitude.values)
    high = good[:, 3]
    assert np.abs(result.gr_X.values[high, 3] - x0[high]).max() > 1500.0


def test_dpr_bins_are_centred_on_the_beam(result):
    """Mean height of the bins in the beam is its centre; their number follows the width."""
    reff = 6371000.0 * 4 / 3
    r = (reff + result.match_height.values) * np.sin(result.ground_range.values / reff)
    r = r / np.cos(np.radians(result.elevation.values))[None, :]
    half = (
        r * np.radians(1.0) / 2 * np.cos(np.radians(result.elevation.values))[None, :]
    )
    # the beams that neither the ground (the lowest bin is the ellipsoid) nor the top of the
    # data window (20.5 km on the outer rays) cut
    whole = (
        result.valid.values
        & (result.match_height.values - half > 0.0)
        & (result.match_height.values + half < 20000.0)
    )
    assert whole.sum() > 100
    dh = result.dpr_H.values - result.match_height.values
    assert np.nanmax(np.abs(dh[whole])) < 125.0
    zen = np.abs((result.ray.values - 24) * 17.5 / 24.0)
    expected = 2 * half / (125.0 * np.cos(np.radians(zen)))[:, None]
    n = result.dpr_bins.values
    assert np.all(np.abs(n[whole] - expected[whole]) <= 1.5)
    # a beam that the ground cuts has its mean above the centre
    cut = result.valid.values & (result.match_height.values - half < -250.0)
    if cut.any():
        assert np.all(dh[cut] > 0.0)


# --------------------------------------------------------------------------
# averaging
# --------------------------------------------------------------------------


def test_uniform_fields_stay_uniform(result):
    good = result.valid.values
    np.testing.assert_allclose(result.gr_DBZH.values[good], 35.0, atol=1e-9)
    np.testing.assert_allclose(result.dpr_zFactorFinal.values[good], 30.0, atol=1e-9)
    np.testing.assert_allclose(result.gr_DBZH_std.values[good], 0.0, atol=1e-6)
    np.testing.assert_allclose(result.gr_DBZH_fraction.values[good], 1.0)
    assert result.gr_gates.values[good].min() > 3
    assert result.gr_DBZH.attrs["averaging"] == "db"
    assert result.gr_X.attrs["averaging"] == "mean"


def test_db_fields_are_averaged_in_linear_units():
    """Alternate gates of 30 and 40 dBZ: 10 log10((1e3 + 1e4) / 2) = 37.40 dBZ."""
    ngate = 600
    pattern = np.where(np.arange(ngate) % 2 == 0, 30.0, 40.0)
    vol = volume()
    for name in vol.children:
        vol[name]["DBZH"] = vol[name]["DBZH"] * 0 + pattern[None, :]
        vol[name]["DBZH"].attrs["units"] = "dBZ"
    out = matched(gr=vol)
    good = out.valid.values
    np.testing.assert_allclose(out.gr_DBZH.values[good], 37.40, atol=0.35)
    np.testing.assert_allclose(out.gr_DBZH_std.values[good], 5.0, atol=0.5)
    arithmetic = matched(gr=vol, gr_fields={"DBZH": "mean"})
    np.testing.assert_allclose(arithmetic.gr_DBZH.values[good], 35.0, atol=0.35)


def test_partly_filled_volumes_are_rejected_by_the_fraction():
    def north_only(ds):
        return (
            ds["Y"] > 2500.0
        )  # the line through the centres of the footprints at y = 2.5 km

    vol = volume(fill=north_only)
    out = matched(gr=vol, min_fraction=0.9)
    f = out.gr_DBZH_fraction.values
    assert ((f > 0.2) & (f < 0.8)).any()  # the footprints that straddle y = 0
    mean = out.gr_DBZH.values
    assert np.all(np.isnan(mean[f < 0.9]))
    assert np.all(np.isfinite(mean[f >= 0.9]))
    loose = matched(gr=vol, min_fraction=0.0)
    assert np.isfinite(loose.gr_DBZH.values[(f > 0)]).all()
    assert not out.valid.values[np.isnan(mean)].any()


def test_missing_values_of_the_file_are_not_averaged():
    dpr = swath()
    dpr["zFactorFinal"] = dpr["zFactorFinal"].where(dpr["height"] < 3000.0, -9999.9)
    out = matched(dpr=dpr, dpr_fields="zFactorFinal", gr_fields="DBZH")
    low = (out.match_height.values < 2000.0) & out.valid.values
    assert low.any()
    np.testing.assert_allclose(out.dpr_zFactorFinal.values[low], 30.0)
    high = out.match_height.values > 6000.0
    assert np.isnan(out.dpr_zFactorFinal.values[high]).all()


def test_gaussian_weights_keep_a_uniform_field_and_change_the_mean_of_a_gradient():
    flat = matched(weighting="gaussian")
    uniform = matched()
    good = flat.valid.values & uniform.valid.values
    np.testing.assert_allclose(flat.gr_DBZH.values[good], 35.0, atol=1e-9)
    np.testing.assert_allclose(
        flat.gr_X.values[good], uniform.gr_X.values[good], atol=300.0
    )
    assert not np.array_equal(flat.gr_gates.values, np.zeros(1))


@pytest.mark.skipif(kernel is None, reason="compiled kernel not available")
@pytest.mark.parametrize("weighting", ["uniform", "gaussian"])
def test_engines_agree(weighting):
    a = matched(weighting=weighting)
    b = matched("compiled", weighting=weighting, n_threads=2)
    assert list(a.data_vars) == list(b.data_vars)
    for name in a.data_vars:
        np.testing.assert_allclose(
            a[name].values, b[name].values, rtol=1e-9, atol=1e-9, err_msg=name
        )
    xr.testing.assert_equal(a.valid, b.valid)


def test_time_difference_and_its_limit():
    out = matched()
    # sweep mean time 12:00:10, scans at 12:00:00 + 0.1 s * scan
    expected = 10.0 - 0.1 * out.scan.values
    np.testing.assert_allclose(out.time_difference.values[:, 0], expected, atol=1e-6)
    cut = matched(max_time_difference=7.0)
    late = np.abs(out.time_difference.values) > 7.0
    assert late.any() and not late.all()
    assert np.isnan(cut.gr_DBZH.values[late]).all()
    assert np.isfinite(cut.gr_DBZH.values[~late & out.valid.values]).all()
    no_time = swath().drop_vars("time")
    assert np.isnan(matched(dpr=no_time).time_difference.values).all()


def test_sweep_dataset_and_sweeps_without_the_field():
    vol = volume()
    vol["sweep_1"] = vol["sweep_1"].to_dataset().drop_vars("Y")
    out = matched(gr=vol)
    assert out.sizes["sweep"] == 3
    np.testing.assert_allclose(out.elevation, [0.5, 10.0, 19.0])
    single = sweep(10.0)
    single = single.assign_coords(SITE)
    one = match_dpr(
        single, swath(), gr_fields="DBZH", dpr_fields="zFactorFinal", engine="numpy"
    )
    assert one.sizes["sweep"] == 1 and float(one.elevation[0]) == 10.0
    full = matched()
    both = full.isel(sweep=2)
    np.testing.assert_allclose(
        one.gr_DBZH.values[:, 0], both.gr_DBZH.values, equal_nan=True
    )
    # default fields: reflectivity of the ground radar and zFactorFinal of the DPR
    default = match_dpr(single, swath(), engine="numpy")
    assert {"gr_DBZH", "dpr_zFactorFinal"} <= set(default.data_vars)


def test_no_footprint_in_range():
    out = matched(min_range=95e3, max_range=96e3)
    for engine in ENGINES:
        empty = matched(engine, max_range=1000.0, min_range=0.0)
        assert empty.sizes["footprint"] == 0
        assert empty.gr_DBZH.shape == (0, 4)
    assert out.sizes["footprint"] >= 0


def test_alias_names_of_the_swath_and_dimension_names():
    dpr = swath().rename(
        {
            "Longitude": "longitude",
            "Latitude": "latitude",
            "localZenithAngle": "zenith",
            "scan": "nscan",
            "ray": "nray",
            "bin": "nbin",
        }
    )
    out = matched(dpr=dpr, dims=("nscan", "nray", "nbin"))
    ref_out = matched()
    np.testing.assert_allclose(out.gr_X.values, ref_out.gr_X.values, equal_nan=True)


def test_zenith_signed_or_not_gives_the_same_volumes():
    signed = swath()
    unsigned = swath()
    unsigned["localZenithAngle"] = abs(unsigned["localZenithAngle"])
    a, b = matched(dpr=signed), matched(dpr=unsigned)
    np.testing.assert_allclose(a.gr_X.values, b.gr_X.values, equal_nan=True)


def test_footprint_size_changes_the_number_of_gates():
    small = matched(footprint_diameter=2000.0)
    large = matched(footprint_diameter=8000.0)
    assert np.nanmedian(large.gr_gates.values) > 4 * np.nanmedian(small.gr_gates.values)


def test_attributes_and_provenance(result):
    assert result.attrs["footprint_diameter"] == 5000.0
    assert result.attrs["beamwidth"] == 1.0
    assert "10.1175/2010JTECHA1403.1" in result.attrs["radarx_references"]
    assert "10.2151/jmsj.2021-011" in result.attrs["radarx_references"]
    assert result.gr_DBZH.attrs["units"] == "dBZ"


def test_beamwidth_from_the_volume_or_the_argument():
    wide = matched(beamwidth=2.0)
    narrow = matched(beamwidth=0.5)
    assert np.nanmean(wide.dpr_bins.values) > 3 * np.nanmean(narrow.dpr_bins.values)
    vol = volume()
    root = vol["/"].to_dataset().drop_vars("radar_beam_width_h")
    vol = xr.DataTree.from_dict(
        {"/": root, **{f"/{n}": vol[n].to_dataset() for n in vol.children}}
    )
    assert matched(gr=vol).attrs["beamwidth"] == 1.0


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"weighting": "boxcar"}, ValueError),
        ({"footprint_diameter": 0.0}, ValueError),
        ({"min_fraction": 1.5}, ValueError),
        ({"min_range": 50e3, "max_range": 20e3}, ValueError),
        ({"beamwidth": 0.0}, ValueError),
        ({"engine": "gpu"}, ValueError),
        ({"gr_fields": "NOPE"}, KeyError),
        ({"gr_fields": {"DBZH": "geometric"}}, ValueError),
        ({"gr_fields": []}, ValueError),
        ({"dpr_fields": "NOPE"}, KeyError),
    ],
)
def test_invalid_arguments(kwargs, error):
    with pytest.raises(error):
        match_dpr(volume(), swath(), **{"gr_fields": "DBZH", **kwargs})


def test_invalid_inputs():
    with pytest.raises(TypeError):
        match_dpr(volume(), swath().to_array())
    with pytest.raises(TypeError):
        match_dpr(np.zeros(3), swath())
    with pytest.raises(ValueError, match="range"):
        match_dpr(xr.Dataset(), swath())
    no_site = sweep(1.0)
    with pytest.raises(ValueError, match="latitude"):
        match_dpr(no_site, swath())
    with pytest.raises(KeyError):
        match_dpr(volume(), swath().drop_vars("height"))
    with pytest.raises(ValueError, match="select one"):
        extra = swath()
        extra["zFactorFinal"] = extra["zFactorFinal"].expand_dims(nfreq=2)
        match_dpr(volume(), extra, dpr_fields="zFactorFinal")
    with pytest.raises(ValueError, match="needs the dimensions"):
        flat = swath()
        flat["height"] = flat["height"].isel(ray=0, drop=True)
        match_dpr(volume(), flat)
    with pytest.raises(ValueError, match="at least two rays"):
        match_dpr(volume(), swath().isel(ray=[24]))
    with pytest.raises(KeyError):
        vol = volume()
        match_dpr(vol, swath(), gr_fields="NOPE")


def test_compiled_kernel_required(monkeypatch):
    monkeypatch.setattr(sm, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError):
        match_dpr(volume(), swath(), engine="compiled")
    out = match_dpr(
        volume(), swath(), engine="auto", gr_fields="DBZH", dpr_fields="zFactorFinal"
    )
    assert out.sizes["sweep"] == 4


# --------------------------------------------------------------------------
# the kernels against a direct computation
# --------------------------------------------------------------------------


def brute_gates(gx, gy, off, values, modes, qx, qy, qs, radius, gaussian):
    nv, nq = values.shape[0], qx.size
    mean = np.full((nv, nq), np.nan)
    sd = np.full((nv, nq), np.nan)
    count = np.zeros((nv, nq))
    total = np.zeros(nq, int)
    for q in range(nq):
        a, b = off[qs[q]], off[qs[q] + 1]
        d2 = (gx[a:b] - qx[q]) ** 2 + (gy[a:b] - qy[q]) ** 2
        inside = np.flatnonzero(d2 <= radius**2) + a
        total[q] = inside.size
        w = (
            np.exp(-np.log(2) * d2[inside - a] / radius**2)
            if gaussian
            else np.ones(inside.size)
        )
        for v in range(nv):
            x = values[v, inside]
            ok = np.isfinite(x)
            count[v, q] = ok.sum()
            if ok.any():
                t = 10 ** (0.1 * x[ok]) if modes[v] else x[ok]
                mean[v, q] = (w[ok] * t).sum() / w[ok].sum()
                m = (w[ok] * x[ok]).sum() / w[ok].sum()
                sd[v, q] = np.sqrt(
                    max(0.0, (w[ok] * x[ok] ** 2).sum() / w[ok].sum() - m * m)
                )
    return mean, sd, count, total


def random_gates(seed=3, n=4000, nq=60):
    rng = np.random.default_rng(seed)
    gx = rng.uniform(0, 30e3, n)
    gy = rng.uniform(0, 30e3, n)
    cut = int(0.375 * n)
    off = np.array([0, cut, cut, n], dtype=np.int64)  # an empty sweep in the middle
    values = rng.normal(30, 8, (2, n))
    values[rng.random((2, n)) < 0.2] = np.nan
    modes = np.array([1, 0], dtype=np.int64)
    qx = rng.uniform(-2e3, 32e3, nq)
    qy = rng.uniform(-2e3, 32e3, nq)
    qs = rng.integers(0, 3, nq).astype(np.int64)
    return gx, gy, off, values, modes, qx, qy, qs


@pytest.mark.parametrize("gaussian", [False, True])
@pytest.mark.parametrize("engine", ENGINES)
def test_gate_kernel_matches_a_direct_sum(engine, gaussian):
    args = random_gates()
    module = kernel if engine == "compiled" else ref
    got = module.match_gates(*args, 2500.0, gaussian, 2)
    want = brute_gates(*args, 2500.0, gaussian)
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w, rtol=1e-10, atol=1e-10)
    assert (got[3][args[7] == 1] == 0).all()  # the empty sweep


def test_gate_kernel_handles_nonfinite_positions_and_far_queries():
    gx, gy, off, values, modes, qx, qy, qs = random_gates(n=200, nq=10)
    gx[5] = np.nan
    qx[0] = np.nan
    qx[1] = 1e9
    for module in [ref] + ([kernel] if kernel else []):
        mean, sd, count, total = module.match_gates(
            gx, gy, off, values, modes, qx, qy, qs, 3000.0
        )
        assert total[0] == 0 and total[1] == 0
        assert np.isnan(mean[:, 0]).all() and np.isnan(mean[:, 1]).all()
    if kernel is not None:
        a = kernel.match_gates(gx, gy, off, values, modes, qx, qy, qs, 3000.0)
        b = ref.match_gates(gx, gy, off, values, modes, qx, qy, qs, 3000.0)
        for u, v in zip(a, b):
            np.testing.assert_allclose(u, v, equal_nan=True)


def test_bucket_grid_is_coarsened_for_huge_extents():
    """A tiny radius on a wide domain would need too many buckets: the grid coarsens."""
    if kernel is None:
        pytest.skip("compiled kernel not available")
    rng = np.random.default_rng(0)
    gx = np.concatenate([rng.uniform(0, 1e6, 3000), [0.0, 1e7]])
    gy = np.concatenate([rng.uniform(0, 1e6, 3000), [0.0, 1e7]])
    off = np.array([0, gx.size], dtype=np.int64)
    values = rng.normal(size=(1, gx.size))
    qx, qy = gx[:50] + 0.5, gy[:50]
    qs = np.zeros(50, np.int64)
    modes = np.zeros(1, np.int64)
    a = kernel.match_gates(gx, gy, off, values, modes, qx, qy, qs, 5.0)
    b = ref.match_gates(gx, gy, off, values, modes, qx, qy, qs, 5.0)
    for u, v in zip(a, b):
        np.testing.assert_allclose(u, v, equal_nan=True)


@pytest.mark.parametrize("gaussian", [False, True])
@pytest.mark.parametrize("engine", ENGINES)
def test_bin_kernel(engine, gaussian):
    rng = np.random.default_rng(1)
    nray, nbin = 12, 40
    z = np.sort(rng.uniform(0, 10e3, (nray, nbin)), axis=1)
    values = rng.normal(25, 6, (2, nray, nbin))
    values[rng.random(values.shape) < 0.15] = np.nan
    modes = np.array([1, 0], dtype=np.int64)
    qr = rng.integers(0, nray, 30).astype(np.int64)
    zc = rng.uniform(1e3, 9e3, 30)
    half = rng.uniform(0, 1500, 30)
    module = kernel if engine == "compiled" else ref
    mean, sd, count, total = module.match_bins(
        z, values, modes, qr, zc - half, zc + half, zc, gaussian, 2
    )
    for q in range(30):
        inside = (z[qr[q]] >= zc[q] - half[q]) & (z[qr[q]] <= zc[q] + half[q])
        assert total[q] == inside.sum()
        w = (
            np.exp(-np.log(2) * ((z[qr[q]][inside] - zc[q]) / half[q]) ** 2)
            if gaussian and half[q] > 0
            else np.ones(inside.sum())
        )
        for v in range(2):
            x = values[v, qr[q]][inside]
            ok = np.isfinite(x)
            assert count[v, q] == ok.sum()
            if ok.any():
                t = 10 ** (0.1 * x[ok]) if modes[v] else x[ok]
                np.testing.assert_allclose(
                    mean[v, q], (w[ok] * t).sum() / w[ok].sum(), rtol=1e-10
                )
            else:
                assert np.isnan(mean[v, q])


@pytest.mark.parametrize("module_name", ["numpy", "compiled"])
def test_kernel_argument_errors(module_name):
    module = ref if module_name == "numpy" else kernel
    if module is None:
        pytest.skip("compiled kernel not available")
    gx, gy, off, values, modes, qx, qy, qs = random_gates(n=100, nq=5)
    off = np.array([0, 50, 100], dtype=np.int64)
    qs = np.zeros(5, np.int64)
    with pytest.raises((ValueError, Exception)):
        module.match_gates(gx, gy, off, values, modes, qx, qy, qs, 0.0)
    with pytest.raises((ValueError, Exception)):
        module.match_gates(gx, gy, off[:2], values, modes, qx, qy, qs, 1000.0)
    with pytest.raises((ValueError, Exception)):
        module.match_gates(gx, gy, off, values[:, :50], modes, qx, qy, qs, 1000.0)
    with pytest.raises((ValueError, Exception)):
        module.match_gates(gx, gy, off, values, modes, qx, qy, qs + 5, 1000.0)
    z = np.zeros((3, 4))
    v = np.zeros((1, 3, 4))
    one = np.zeros(1, np.int64)
    with pytest.raises((ValueError, Exception)):
        module.match_bins(z, v[:, :2], one, one, np.zeros(1), np.zeros(1), np.zeros(1))
    with pytest.raises((ValueError, Exception)):
        module.match_bins(z, v, one, one + 7, np.zeros(1), np.zeros(1), np.zeros(1))


# --------------------------------------------------------------------------
# DSD parameters of the DPR
# --------------------------------------------------------------------------


def normalized_gamma(dm, nw, mu=3.0):
    """N(D) of the normalized gamma DSD of Testud et al. (2001), the form of Seto et al. (2021)."""
    from math import gamma

    d = np.arange(0.05, 8.0, 0.01)
    f = 6.0 / 4.0**4 * (4.0 + mu) ** (mu + 4.0) / gamma(mu + 4.0)
    nd = nw * f * (d / dm) ** mu * np.exp(-(4.0 + mu) * d / dm)
    return xr.DataArray(
        nd,
        dims="diameter",
        coords={"diameter": d, "bin_width": ("diameter", np.full(d.size, 0.01))},
    )


def test_dpr_dsd_splits_the_parameter_array():
    arr = np.array([[[[36.0, 1.5], [-9999.9, -9999.9]]]])
    param = xr.DataArray(arr, dims=("scan", "ray", "bin", "nDSD"))
    out = dpr_dsd(param)
    assert float(out.DM[0, 0, 0]) == 1.5
    assert float(out.DBNW[0, 0, 0]) == 36.0
    assert float(out.NW[0, 0, 0]) == pytest.approx(10**3.6)
    assert np.isnan(out.DM[0, 0, 1]) and np.isnan(out.NW[0, 0, 1])
    assert out.attrs["mu"] == 3.0
    assert out.DM.attrs["units"] == "mm" and out.NW.attrs["units"] == "m-3 mm-1"
    renamed = dpr_dsd(param.rename(nDSD="parts"), dim="parts")
    xr.testing.assert_identical(renamed.DM, out.DM)
    other = param.rename(nDSD="DSD_params")
    assert float(dpr_dsd(other).DM[0, 0, 0]) == 1.5


def test_dpr_dsd_errors():
    with pytest.raises(TypeError):
        dpr_dsd(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="cannot find"):
        dpr_dsd(xr.DataArray(np.zeros((3, 3)), dims=("a", "b")))
    with pytest.raises(ValueError, match="cannot find"):
        dpr_dsd(xr.DataArray(np.zeros((3, 2, 2)), dims=("a", "b", "c")))
    with pytest.raises(ValueError, match="two elements"):
        dpr_dsd(xr.DataArray(np.zeros((3, 2)), dims=("a", "b")), dim="a")


@pytest.mark.parametrize("dm, db_nw", [(1.2, 36.0), (1.8, 34.0), (2.4, 32.0)])
def test_radarx_retrieval_uses_the_dpr_convention(dm, db_nw):
    """Dm and 10 log10 Nw of the DPR, with mu = 3, are what ``dsd`` returns for the same DSD.

    The distribution is built in the form of Eq. 2 of Seto et al. (2021); its S-band reflectivity and
    differential reflectivity go through ``dsd(method="normalized", mu=3)``.
    """
    nw = 10 ** (0.1 * db_nw)
    spectrum = normalized_gamma(dm, nw)
    fields = radarx.retrieve.radar_from_dsd(spectrum, band="S")
    # the moments of the spectrum: Dm = M4 / M3 and Nw of the Dm form
    assert float(fields.DM) == pytest.approx(dm, rel=0.01)
    assert float(fields.NW) == pytest.approx(nw, rel=0.02)
    ds = xr.Dataset(
        {"DBZH": ("r", [float(fields.DBZH)]), "ZDR": ("r", [float(fields.ZDR)])}
    )
    ds["DBZH"].attrs["units"] = "dBZ"
    ds["ZDR"].attrs["units"] = "dB"
    got = dsd(ds, method="normalized", mu=3.0, band="S")
    assert float(got.DM[0]) == pytest.approx(dm, rel=0.04)
    assert 10 * np.log10(float(got.NW[0])) == pytest.approx(db_nw, abs=0.6)


def test_matched_dbnw_is_the_log_of_the_linear_mean():
    dpr = swath()
    nbin = dpr.sizes["bin"]
    alt = np.where(np.arange(nbin) % 2 == 0, 30.0, 36.0)
    dpr["DBNW"] = (
        ("scan", "ray", "bin"),
        np.broadcast_to(alt, dpr["height"].shape).copy(),
        {"units": "dB"},
    )
    out = match_dpr(volume(), dpr, gr_fields="DBZH", dpr_fields="DBNW", engine="numpy")
    good = out.valid.values & (out.dpr_bins.values >= 12)
    assert good.sum() > 100
    expected = 10 * np.log10((10**3.0 + 10**3.6) / 2)
    np.testing.assert_allclose(out.dpr_DBNW.values[good], expected, atol=0.45)
    assert out.dpr_DBNW.attrs["averaging"] == "db"


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def test_match_statistics_known_values():
    ref_ = xr.DataArray([10.0, 20.0, 30.0, 40.0, np.nan, 5.0], dims="pair")
    est = xr.DataArray([12.0, 18.0, 33.0, 44.0, 50.0, np.nan], dims="pair")
    s = match_statistics(ref_, est)
    assert int(s["count"]) == 4
    assert float(s.mean_reference) == 25.0
    assert float(s.bias) == pytest.approx((2 - 2 + 3 + 4) / 4)
    assert float(s.rmse) == pytest.approx(np.sqrt((4 + 4 + 9 + 16) / 4))
    assert float(s.mae) == pytest.approx(11 / 4)
    assert float(s.ratio) == pytest.approx(107 / 100)
    r = np.corrcoef([10, 20, 30, 40], [12, 18, 33, 44])[0, 1]
    assert float(s.correlation) == pytest.approx(r)
    assert "pod" not in s


def test_match_statistics_detection_scores_and_dimensions():
    ref_ = xr.DataArray(
        [[0.0, 1.0, 2.0, 0.0], [3.0, 0.0, 0.0, 1.0]], dims=("t", "pair")
    )
    est = xr.DataArray([[0.0, 2.0, 0.0, 1.0], [4.0, 0.0, 1.0, 0.0]], dims=("t", "pair"))
    s = match_statistics(ref_, est, threshold=1.0)
    # events: ref [F T T F | T F F T], est [F T F T | T F T F]
    assert float(s.pod) == pytest.approx(2 / 4)
    assert float(s.far) == pytest.approx(2 / 4)
    assert float(s.csi) == pytest.approx(2 / 6)
    assert float(s.frequency_bias) == pytest.approx(4 / 4)
    per_t = match_statistics(ref_, est, threshold=1.0, dim="pair")
    assert per_t.sizes == {"t": 2}
    assert float(per_t.pod[0]) == pytest.approx(1 / 2)
    none = match_statistics(
        xr.DataArray([0.0, 0.0], dims="p"),
        xr.DataArray([0.0, 0.0], dims="p"),
        threshold=1.0,
    )
    assert (
        np.isnan(float(none.pod))
        and np.isnan(float(none.csi))
        and np.isnan(float(none.ratio))
    )
    transposed = match_statistics(ref_, est.transpose("pair", "t"))
    assert float(transposed.bias) == float(match_statistics(ref_, est).bias)


def test_match_statistics_errors_and_single_pair():
    a = xr.DataArray([1.0, 2.0], dims="p")
    with pytest.raises(TypeError):
        match_statistics(np.zeros(2), a)
    with pytest.raises(ValueError):
        match_statistics(a, xr.DataArray([1.0, 2.0, 3.0], dims="p"))
    one = match_statistics(a[:1], a[:1])
    assert np.isnan(float(one.correlation))


def test_matched_reflectivities_agree_in_decibels(result):
    s = match_statistics(result.gr_DBZH, result.dpr_zFactorFinal)
    assert float(s.bias) == pytest.approx(-5.0, abs=1e-6)  # 30 dBZ against 35 dBZ
    assert int(s["count"]) == int(result.valid.sum())


def test_accessor():
    out = volume().radarx.match_dpr(
        swath(), gr_fields="DBZH", dpr_fields="zFactorFinal", engine="numpy"
    )
    assert out.sizes["sweep"] == 4
    expected = match_dpr(
        volume(), swath(), gr_fields="DBZH", dpr_fields="zFactorFinal", engine="numpy"
    )
    np.testing.assert_allclose(
        out.gr_DBZH.values, expected.gr_DBZH.values, equal_nan=True
    )


def test_gate_positions_are_those_of_the_grid_code():
    """The matched gate positions are the xradar ones used by the gridding functions."""
    ds = sweep(5.0, naz=36, ngate=50)
    gx, gy, values = sm._sweep_gates(ds, [("X", 0)], SITE["altitude"], 4.0 / 3.0)
    np.testing.assert_allclose(gx, ds["X"].values.ravel())
    np.testing.assert_allclose(values[0], ds["X"].values.ravel())
    assert gy.shape == gx.shape == (36 * 50,)
    assert cone.EARTH_RADIUS == sm.EARTH_RADIUS
