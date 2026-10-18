#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the ray-angle check and repair
========================================

Synthetic sweeps with known jumps: a sweep with regular angles and a little jitter
has no flags; a bad ray, a run of bad rays, a duplicated ray, rays out of order,
missing rays and a missing angle are found and, where the rule says so, repaired to the
true angle. The compiled kernel and the NumPy implementation must agree.
"""

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.retrieve import AZIMUTH_FLAGS, azimuth_flags, fix_azimuth

azimuth_qc = __import__("radarx.retrieve.azimuth_qc", fromlist=["x"])
numpy_side = __import__("radarx.retrieve._azimuth_qc_numpy", fromlist=["x"])

ENGINES = ["numpy"] + (["compiled"] if azimuth_qc.HAS_COMPILED_KERNEL else [])
compiled_only = pytest.mark.skipif(
    not azimuth_qc.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)
OUTLIER, DUPLICATE, GAP, REVERSAL, MISSING = (
    AZIMUTH_FLAGS[k] for k in ("outlier", "duplicate", "gap", "reversal", "missing")
)
RNG = np.random.default_rng(7)


def angles(n=360, start=0.0, step=1.0, jitter=0.05):
    """Regular ray angles in degrees, wrapped to [0, 360), with a little jitter."""
    return (start + step * np.arange(n) + RNG.normal(0.0, jitter, n)) % 360.0


def sweep(az, name="azimuth", n_range=6):
    """A sweep Dataset with a field and the ray angle ``az`` along ``name``."""
    az = np.asarray(az, dtype=float)
    rng = 500.0 + 250.0 * np.arange(n_range)
    data = np.tile(np.arange(az.size, dtype=np.float32)[:, None], (1, n_range))
    ds = xr.Dataset(
        {"DBZH": ((name, "range"), data)},
        coords={name: az, "range": rng},
    )
    ds["DBZH"].attrs["units"] = "dBZ"
    return ds


def flag_of(az, **kwargs):
    return azimuth_flags(sweep(az), **kwargs).azimuth_flag.values


def where_flag(flags, bit):
    return np.flatnonzero(flags & bit).tolist()


# --------------------------------------------------------------------------
# no false alarms
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("step", [1.0, 0.5, -1.0, -0.5, 1.4])
def test_regular_sweep_has_no_flags(engine, step):
    az = angles(n=int(360 / abs(step)), start=350.0, step=step)  # wraps through north
    assert not flag_of(az, engine=engine).any()


@pytest.mark.parametrize("engine", ENGINES)
def test_sector_scan_with_slow_ends_has_no_flags(engine):
    """Spacing that shrinks to 0.4 at the ends (antenna acceleration) is not a jump."""
    steps = np.concatenate(
        [np.linspace(0.45, 1.0, 12), np.ones(100), np.linspace(1.0, 0.45, 12)]
    )
    az = (
        50.0
        + np.concatenate([[0.0], np.cumsum(steps)])
        + RNG.normal(0, 0.1, steps.size + 1)
    )
    assert not flag_of(az, engine=engine).any()


@pytest.mark.parametrize("engine", ENGINES)
def test_stationary_antenna_is_left_alone(engine):
    az = np.full(40, 123.0) + RNG.normal(0, 0.001, 40)
    assert not flag_of(az, engine=engine).any()


@pytest.mark.parametrize("engine", ENGINES)
def test_too_few_rays_give_no_flags(engine):
    assert not flag_of(angles(4), engine=engine).any()


# --------------------------------------------------------------------------
# what is found
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("jump", [30.0, -45.0, 170.0, 200.0])
def test_single_bad_ray_is_an_outlier_and_is_repaired(engine, jump):
    truth = angles(start=340.0)
    az = truth.copy()
    az[100] = (az[100] + jump) % 360.0
    out = azimuth_flags(sweep(az), engine=engine)
    assert where_flag(out.azimuth_flag.values, OUTLIER) == [100]
    assert where_flag(out.azimuth_flag.values, ~OUTLIER & 255) == []
    error = np.abs((out.azimuth_fixed.values[100] - truth[100] + 180.0) % 360.0 - 180.0)
    assert error < 0.3
    # the other rays keep their angles exactly
    keep = np.arange(truth.size) != 100
    assert np.array_equal(out.azimuth_fixed.values[keep], az[keep])


@pytest.mark.parametrize("engine", ENGINES)
def test_run_of_two_bad_rays_is_found(engine):
    truth = angles()
    az = truth.copy()
    az[200:202] = (az[200:202] - 40.0) % 360.0
    out = azimuth_flags(sweep(az), engine=engine)
    assert where_flag(out.azimuth_flag.values, OUTLIER) == [200, 201]
    assert np.all(np.abs(out.azimuth_fixed.values[200:202] - truth[200:202]) < 0.4)


@pytest.mark.parametrize("engine", ENGINES)
def test_duplicate_ray(engine):
    az = angles()
    az[100] = az[99]
    out = azimuth_flags(sweep(az), engine=engine)
    assert where_flag(out.azimuth_flag.values, DUPLICATE) == [100]
    assert abs(out.azimuth_fixed.values[100] - (az[99] + az[101]) / 2) < 0.2


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("missing", [2, 3, 6])
def test_gap_is_flagged_with_the_number_of_missing_rays(engine, missing):
    az = angles(n=131, start=50.0, jitter=0.05)
    az[120:] += missing  # `missing` rays are missing after ray 119
    out = azimuth_flags(sweep(az), engine=engine)
    assert where_flag(out.azimuth_flag.values, GAP) == [119]
    assert out.azimuth_missing_rays.values[119] == missing
    assert out.azimuth_missing_rays.values.sum() == missing
    assert not (out.azimuth_flag.values & np.uint8(255 - GAP)).any()
    assert np.array_equal(out.azimuth_fixed.values, az)  # a gap is not repaired
    assert out.azimuth_step == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_one_ray_skip_is_not_a_gap(engine):
    az = angles(n=131, start=50.0)
    az[60:] += 1.0
    assert not flag_of(az, engine=engine).any()


@pytest.mark.parametrize("engine", ENGINES)
def test_rays_out_of_order(engine):
    az = angles(n=400)
    az[300:303] = az[300:303][::-1]
    flags = flag_of(az, engine=engine)
    assert where_flag(flags, REVERSAL)
    assert where_flag(flags, GAP)
    assert not (flags & (OUTLIER | DUPLICATE)).any()


@pytest.mark.parametrize("engine", ENGINES)
def test_missing_angle_is_interpolated(engine):
    truth = angles()
    az = truth.copy()
    az[50] = np.nan
    out = azimuth_flags(sweep(az), engine=engine)
    assert where_flag(out.azimuth_flag.values, MISSING) == [50]
    assert abs(out.azimuth_fixed.values[50] - truth[50]) < 0.2


@pytest.mark.parametrize("engine", ENGINES)
def test_bad_ray_at_the_ends_of_the_sweep(engine):
    truth = angles(n=200, start=10.0)
    az = truth.copy()
    az[0] += 25.0
    az[-1] -= 25.0
    out = azimuth_flags(sweep(az), engine=engine)
    assert where_flag(out.azimuth_flag.values, OUTLIER) == [0, 199]
    assert np.all(np.abs(out.azimuth_fixed.values[[0, -1]] - truth[[0, -1]]) < 0.3)


@pytest.mark.parametrize("engine", ENGINES)
def test_elevation_of_an_rhi_is_checked(engine):
    el = 0.5 + 0.25 * np.arange(120) + RNG.normal(0, 0.01, 120)
    el[60] += 5.0
    ds = xr.Dataset(
        {"DBZH": (("elevation", "range"), np.zeros((120, 4), np.float32))},
        coords={"elevation": el, "range": np.arange(4) * 250.0 + 250.0},
    )
    out = azimuth_flags(ds, engine=engine)
    assert where_flag(out.azimuth_flag.values, OUTLIER) == [60]
    assert out.azimuth_fixed.attrs["angle"] == "elevation"


# --------------------------------------------------------------------------
# the two engines
# --------------------------------------------------------------------------


@compiled_only
def test_compiled_kernel_and_numpy_agree_on_random_sweeps():
    rng = np.random.default_rng(11)
    for _ in range(150):
        n = int(rng.integers(8, 700))
        az = angles(
            n, rng.uniform(0, 360), rng.choice([-1, 1]) * rng.choice([0.5, 1.0, 1.4])
        )
        for i in rng.choice(n, size=int(rng.integers(0, 6)), replace=False):
            az[i] += rng.uniform(-60, 60)
        if rng.random() < 0.3:
            az[int(rng.integers(0, n))] = np.nan
        if rng.random() < 0.3:
            az[int(rng.integers(5, max(6, n - 5))) :] += rng.uniform(3, 9)
        a = numpy_side.flag_azimuth_numpy(az, 4.0, 5, 0.1, 2.5)
        b = numpy_side.flag_azimuth([az], 4.0, 5, 0.1, 2.5, True, 2)[0]
        assert np.array_equal(a[0], b[0])
        assert np.allclose(a[1], b[1], atol=1e-9, equal_nan=True)
        assert np.isclose(a[2], b[2], equal_nan=True)


@compiled_only
def test_kernel_handles_many_sweeps_and_threads():
    sweeps = [angles(n=int(n)) for n in (100, 360, 720, 50)]
    one = numpy_side.flag_azimuth(sweeps, 4.0, 5, 0.1, 2.5, True, 1)
    many = numpy_side.flag_azimuth(sweeps, 4.0, 5, 0.1, 2.5, True, 4)
    for x, y in zip(one, many):
        assert np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1])


def test_compiled_engine_error_without_kernel(monkeypatch):
    monkeypatch.setattr(azimuth_qc, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError):
        azimuth_flags(sweep(angles()), engine="compiled")


# --------------------------------------------------------------------------
# repair methods
# --------------------------------------------------------------------------


def test_interpolate_keeps_every_ray_and_the_raw_angle():
    truth = angles()
    az = truth.copy()
    az[100] += 30.0
    fixed = fix_azimuth(sweep(az))
    assert fixed.sizes["azimuth"] == truth.size
    assert abs(fixed.azimuth.values[100] - truth[100]) < 0.3
    assert fixed.azimuth_raw.values[100] == pytest.approx(az[100])
    assert fixed.azimuth_flag.values[100] == OUTLIER
    assert np.array_equal(fixed.DBZH.values, sweep(az).DBZH.values)


def test_fixing_twice_changes_nothing_more():
    az = angles()
    az[[40, 41, 250]] += 25.0
    once = fix_azimuth(sweep(az))
    again = azimuth_flags(sweep(once.azimuth.values))
    assert not again.azimuth_flag.values.any()


def test_drop_removes_the_bad_rays():
    az = angles()
    az[[10, 200]] += 40.0
    az[100] = az[99]
    out = fix_azimuth(sweep(az), "drop")
    assert out.sizes["azimuth"] == az.size - 3
    assert not azimuth_flags(out).azimuth_flag.values.any()
    assert 10.0 not in out.DBZH.values[:, 0]  # the data of ray 10 left with it


def test_fill_inserts_the_missing_rays_and_keeps_the_real_ones():
    az = angles(n=131, start=50.0, jitter=0.05)
    az[120:] += 4.0
    ds = sweep(az)
    ds = ds.assign_coords(
        time=(
            "azimuth",
            np.datetime64("2017-08-25T22:31")
            + np.arange(131) * np.timedelta64(40, "ms"),
        )
    )
    out = fix_azimuth(ds, "fill")
    assert out.sizes["azimuth"] == 135
    filled = out.azimuth_filled.values
    assert filled.sum() == 4 and np.flatnonzero(filled).tolist() == [120, 121, 122, 123]
    assert np.all(np.isnan(out.DBZH.values[filled]))
    assert np.array_equal(out.DBZH.values[~filled], ds.DBZH.values)
    steps = np.diff(out.azimuth.values)
    assert (
        steps.min() > 0.7 and steps.max() < 1.3
    )  # the new rays are spaced like the real ones
    t = out.time.values.astype("datetime64[ns]").astype("int64")
    assert np.all(
        np.diff(t) > 0
    )  # the time of the new rays lies between their neighbours


def test_fill_without_a_gap_adds_the_marker_only():
    out = fix_azimuth(sweep(angles()), "fill")
    assert out.sizes["azimuth"] == 360 and not out.azimuth_filled.any()


def test_regularize_puts_the_sweep_on_a_regular_grid():
    az = angles(n=200, start=20.0, jitter=0.1)
    az[120:] += 3.0
    out = fix_azimuth(sweep(az), "regularize")
    steps = np.diff(out.azimuth.values)
    assert np.allclose(steps, steps[0], atol=1e-6)
    assert out.azimuth_filled.values.sum() == 3
    assert np.all(np.isnan(out.DBZH.values[out.azimuth_filled.values]))


def test_regularize_refuses_a_sweep_out_of_order():
    az = angles(n=400)
    az[300:303] = az[300:303][::-1]
    with pytest.raises(ValueError, match="scan order"):
        fix_azimuth(sweep(az), "regularize")


# --------------------------------------------------------------------------
# objects, accessors, arguments
# --------------------------------------------------------------------------


def volume(*sweeps):
    nodes = {"/": xr.Dataset(attrs={"instrument_name": "test"})}
    for i, ds in enumerate(sweeps):
        nodes[f"/sweep_{i}"] = ds
    return xr.DataTree.from_dict(nodes)


def test_volume_gives_one_node_per_sweep():
    clean, bad = angles(), angles()
    bad[100] += 30.0
    vol = volume(sweep(clean), sweep(bad))
    out = azimuth_flags(vol)
    assert set(out.children) == {"sweep_0", "sweep_1"}
    assert not out["sweep_0"].azimuth_flag.values.any()
    assert out["sweep_1"].azimuth_flag.values[100] == OUTLIER
    repaired = fix_azimuth(vol)
    assert set(repaired.children) == {"sweep_0", "sweep_1"}
    assert abs(repaired["sweep_1"].azimuth.values[100] - clean[100]) < 3.0
    assert repaired.attrs == vol.attrs


def test_volume_with_different_ray_counts_after_fill():
    gap = angles(n=100, start=0.0)
    gap[50:] += 3.0
    out = fix_azimuth(volume(sweep(angles(n=100)), sweep(gap)), "fill")
    assert out["sweep_0"].sizes["azimuth"] == 100
    assert out["sweep_1"].sizes["azimuth"] == 103


def test_single_sweep_node_of_a_volume():
    vol = volume(sweep(angles()))
    out = azimuth_flags(vol["sweep_0"])
    assert isinstance(out, xr.Dataset)


def test_accessors_call_the_functions():
    ds = sweep(angles())
    assert not ds.radarx.azimuth_flags().azimuth_flag.values.any()
    assert ds.radarx.fix_azimuth("drop").sizes["azimuth"] == 360
    vol = volume(ds)
    assert "sweep_0" in vol.radarx.azimuth_flags().children


def test_provenance_names_the_papers():
    out = azimuth_flags(sweep(angles()))
    assert "10.1080/01621459.1974.10482962" in out.attrs["radarx_references"]
    assert "10.1016/j.jesp.2013.03.013" in out.attrs["radarx_references"]
    assert (
        "azimuth" in out.attrs["radarx_method"].lower()
        or "ray" in out.attrs["radarx_method"].lower()
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tolerance": 0},
        {"window": 1},
        {"duplicate_fraction": 1.0},
        {"gap_factor": 1.0},
        {"engine": "gpu"},
    ],
)
def test_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        azimuth_flags(sweep(angles()), **kwargs)


def test_bad_method_and_object():
    with pytest.raises(ValueError, match="method"):
        fix_azimuth(sweep(angles()), "invent")
    with pytest.raises(ValueError, match="regularize_tolerance"):
        fix_azimuth(sweep(angles()), "regularize", regularize_tolerance=0.0)
    with pytest.raises(TypeError):
        azimuth_flags(np.arange(10))
    with pytest.raises(ValueError, match="azimuth"):
        azimuth_flags(xr.Dataset({"x": ("t", np.arange(5.0))}))


# --------------------------------------------------------------------------
# cases for tests/test_dask_computes.py
# --------------------------------------------------------------------------


def gap_sweep():
    """A sweep with two bad rays and a gap of three rays."""
    az = angles(n=120, start=50.0, jitter=0.05)
    az[70:] += 3.0
    az[20] += 30.0
    az[40] = az[39]
    return sweep(az, n_range=8)


def flags_case():
    return azimuth_flags, (gap_sweep(),), {}


def fill_case():
    return fix_azimuth, (gap_sweep(), "fill"), {}


# --------------------------------------------------------------------------
# real data: sector PPIs of the CPOL radar (25 August 2017, 22:31 UTC) with gaps
# --------------------------------------------------------------------------

# Ray azimuths of two sweeps (elevations 2.3 and 23 degrees) of a sector scan from 50 to 180
# degrees with 131 rays. The antenna slows down at the end of the sector (steps of 0.5
# degrees), and rays are missing at one place of each sweep.
CPOL_SWEEP_2 = [
    50.273,
    51.273,
    52.287,
    53.264,
    54.283,
    55.283,
    56.253,
    57.288,
    58.282,
    59.252,
    60.266,
    61.257,
    62.295,
    63.268,
    64.281,
    65.303,
    66.289,
    67.305,
    68.261,
    69.277,
    70.304,
    71.274,
    72.287,
    73.265,
    74.298,
    75.292,
    76.278,
    77.294,
    78.272,
    79.272,
    80.291,
    81.271,
    82.271,
    83.271,
    84.273,
    85.290,
    86.289,
    87.278,
    88.286,
    89.272,
    90.286,
    91.294,
    92.282,
    93.301,
    94.287,
    95.304,
    96.303,
    97.287,
    98.306,
    99.261,
    100.280,
    101.264,
    102.277,
    103.302,
    104.282,
    105.287,
    106.304,
    107.290,
    108.306,
    109.262,
    110.281,
    111.278,
    112.261,
    113.277,
    114.261,
    115.282,
    116.282,
    117.265,
    118.282,
    119.265,
    120.289,
    121.284,
    122.292,
    123.286,
    124.266,
    125.305,
    126.299,
    127.282,
    128.290,
    129.273,
    130.295,
    131.295,
    132.284,
    133.294,
    134.278,
    135.294,
    136.296,
    137.282,
    138.299,
    139.282,
    140.298,
    141.298,
    142.281,
    143.303,
    144.286,
    145.275,
    146.255,
    147.283,
    148.291,
    149.268,
    150.282,
    151.306,
    152.290,
    153.303,
    154.303,
    155.289,
    156.388,
    157.154,
    158.351,
    159.326,
    160.181,
    161.194,
    162.263,
    162.801,
    164.279,
    165.265,
    166.264,
    167.267,
    168.283,
    169.253,
    173.741,
    174.743,
    175.710,
    176.732,
    177.295,
    177.731,
    178.283,
    178.728,
    179.264,
    179.747,
    180.280,
]
CPOL_SWEEP_18 = [
    50.271,
    51.265,
    52.281,
    53.278,
    54.275,
    55.269,
    56.269,
    57.280,
    58.307,
    59.260,
    60.298,
    61.293,
    62.276,
    63.279,
    64.297,
    65.270,
    66.275,
    67.302,
    68.266,
    69.274,
    70.288,
    71.252,
    72.276,
    73.265,
    74.237,
    75.265,
    76.270,
    77.281,
    78.261,
    79.277,
    80.291,
    81.277,
    82.285,
    83.290,
    84.271,
    85.284,
    86.267,
    87.267,
    88.292,
    89.269,
    90.275,
    91.288,
    92.241,
    93.277,
    94.279,
    95.260,
    96.281,
    97.289,
    98.267,
    99.275,
    100.297,
    101.277,
    102.299,
    103.291,
    104.277,
    105.287,
    106.260,
    107.271,
    108.284,
    109.259,
    110.270,
    111.275,
    112.283,
    113.258,
    114.283,
    115.263,
    116.260,
    117.271,
    118.273,
    119.279,
    120.226,
    121.275,
    122.278,
    123.253,
    124.258,
    125.263,
    126.299,
    127.293,
    128.271,
    129.276,
    130.301,
    131.273,
    132.281,
    133.259,
    134.294,
    135.286,
    136.252,
    137.288,
    138.268,
    139.274,
    140.268,
    141.287,
    142.226,
    143.286,
    144.264,
    145.289,
    146.264,
    153.254,
    154.292,
    155.284,
    156.261,
    157.297,
    158.302,
    159.272,
    160.285,
    161.257,
    162.260,
    163.271,
    164.287,
    165.232,
    166.259,
    167.297,
    168.289,
    169.266,
    170.299,
    171.291,
    172.266,
    173.298,
    174.293,
    174.721,
    175.270,
    175.735,
    176.276,
    176.718,
    177.297,
    177.693,
    178.289,
    178.726,
    179.275,
    179.734,
    180.302,
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    ("azimuth", "ray", "missing"), [(CPOL_SWEEP_2, 119, 3), (CPOL_SWEEP_18, 96, 6)]
)
def test_cpol_sector_with_missing_rays(engine, azimuth, ray, missing):
    out = azimuth_flags(sweep(azimuth), engine=engine)
    flags = out.azimuth_flag.values
    # one gap and nothing else: the slow end of the sector is not a jump
    assert np.flatnonzero(flags).tolist() == [ray]
    assert flags[ray] == GAP
    assert out.azimuth_missing_rays.values[ray] == missing
    filled = fix_azimuth(sweep(azimuth), "fill", engine=engine)
    assert filled.sizes["azimuth"] == len(azimuth) + missing
    assert np.diff(filled.azimuth.values).max() < 1.5
    assert (
        not azimuth_flags(filled)
        .azimuth_flag.values[~filled.azimuth_filled.values]
        .any()
    )
