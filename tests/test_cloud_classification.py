#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the target classification of cloud radar echoes
=========================================================

Every rule is checked on gates built to trigger it, the compiled kernel against the
NumPy implementation, the temperature given in its several forms, sweeps and volumes,
and the classification of the Cloudnet product of Lindenberg (1 May 2023, 35.5 GHz),
which uses a lidar and a model besides the radar, on real data.
"""

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401  registers the accessors
from radarx.retrieve import classify_cloud_targets
from radarx.retrieve import cloud_classification as cc

compiled_only = pytest.mark.skipif(
    not cc.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)
ENGINES = ["numpy", pytest.param("compiled", marks=compiled_only)]

NO_ECHO, DROPLETS, DRIZZLE, ICE, MELTING, INSECTS, LARGE_ICE, UNKNOWN = range(8)
K = 273.15


def gates(dbz, celsius, ldr=None, dfr=None):
    n = len(dbz)
    ds = xr.Dataset(
        {
            "DBZ": ("range", np.asarray(dbz, dtype=float)),
            "temperature": ("range", np.asarray(celsius, dtype=float) + K),
        },
        coords={"range": 100.0 + 30.0 * np.arange(n)},
    )
    if ldr is not None:
        ds["ldr"] = ("range", np.asarray(ldr, dtype=float))
    if dfr is not None:
        ds["DFR"] = ("range", np.asarray(dfr, dtype=float))
    return ds


@pytest.mark.parametrize("engine", ENGINES)
def test_the_rules_in_order(engine):
    dbz = [
        np.nan,
        -30.0,
        5.0,
        5.0,
        -30.0,
        -20.1,
        -20.0,
        5.0,
        5.0,
        5.0,
        15.0,
        -30.0,
        5.0,
    ]
    celsius = [
        10.0,
        10.0,
        np.nan,
        1.0,
        20.0,
        20.0,
        20.0,
        20.0,
        -10.0,
        -10.0,
        20.0,
        0.0,
        2.5,
    ]
    ldr = [
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        -15.0,
        np.nan,
        -15.0,
        -15.0,
        np.nan,
        np.nan,
    ]
    dfr = [
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        4.0,
        2.0,
        np.nan,
        np.nan,
        np.nan,
    ]
    ds = gates(dbz, celsius, ldr, dfr)
    out = classify_cloud_targets(ds, ldr="ldr", dfr="DFR", engine=engine)
    expected = [
        NO_ECHO,  # no reflectivity
        DROPLETS,  # warm, below the drizzle threshold, no depolarization
        UNKNOWN,  # no temperature
        MELTING,  # in the melting zone
        DROPLETS,  # warm and weak
        DROPLETS,  # just below the drizzle threshold
        DRIZZLE,  # at the threshold
        INSECTS,  # warm, depolarizing, weak
        LARGE_ICE,  # cold with a large ratio
        ICE,  # cold with a small ratio (the depolarization does not matter below zero)
        DRIZZLE,  # a warm echo of 15 dBZ with depolarization is above the insect limit
        MELTING,  # the lower edge of the melting zone is inside
        DRIZZLE,  # 2.5 degrees C is above the zone of 0 to 2, 5 dBZ
    ]
    np.testing.assert_array_equal(out.values, expected)
    assert out.name == "TARGET_CLASS" and out.dtype == np.int8
    assert list(out.attrs["flag_values"]) == list(range(8))
    assert out.attrs["flag_meanings"].split()[3] == "ice"


def test_the_melting_zone_edges_and_thresholds_are_options():
    ds = gates([0.0, 0.0, 0.0, 0.0], [-0.1, 0.0, 2.0, 2.1])
    default = classify_cloud_targets(ds).values
    np.testing.assert_array_equal(default, [ICE, MELTING, MELTING, DRIZZLE])
    wider = classify_cloud_targets(ds, melting_temperature=(-1.0, 3.0)).values
    np.testing.assert_array_equal(wider, [MELTING] * 4)
    higher = classify_cloud_targets(
        ds, min_drizzle_dbz=5.0, melting_temperature=(-1.0, 1.0)
    ).values
    np.testing.assert_array_equal(higher, [MELTING, MELTING, DROPLETS, DROPLETS])
    quiet = classify_cloud_targets(ds, min_dbz=5.0).values
    assert (quiet == NO_ECHO).all()


def test_insects_and_large_ice_need_their_fields_and_can_be_turned_off():
    ds = gates([0.0, 0.0], [10.0, -10.0], ldr=[-15.0, -15.0], dfr=[5.0, 5.0])
    no_fields = classify_cloud_targets(ds)
    np.testing.assert_array_equal(no_fields.values, [DRIZZLE, ICE])
    with_fields = classify_cloud_targets(ds, ldr="ldr", dfr="DFR")
    np.testing.assert_array_equal(with_fields.values, [INSECTS, LARGE_ICE])
    off = classify_cloud_targets(
        ds, ldr="ldr", dfr="DFR", min_insect_ldr=None, min_large_dfr=None
    )
    np.testing.assert_array_equal(off.values, [DRIZZLE, ICE])
    sample = classify_cloud_targets(
        ds, ldr=ds.ldr, dfr=ds.DFR + 0.0, max_insect_dbz=-10.0
    )
    np.testing.assert_array_equal(sample.values, [DRIZZLE, LARGE_ICE])


@compiled_only
def test_engines_agree_on_random_fields_and_threads():
    rnd = np.random.default_rng(7)
    shape = (30, 200)

    def field(low, high, missing):
        values = rnd.uniform(low, high, shape)
        values[rnd.random(shape) < missing] = np.nan
        return values

    ds = xr.Dataset(
        {
            "DBZ": (("time", "range"), field(-50, 20, 0.1)),
            "temperature": (("time", "range"), field(250, 290, 0.02)),
            "ldr": (("time", "range"), field(-35, -5, 0.3)),
            "dfr": (("time", "range"), field(-1, 12, 0.3)),
        },
        coords={"range": np.arange(200.0)},
    )
    a = classify_cloud_targets(ds, ldr="ldr", dfr="dfr", engine="numpy")
    b = classify_cloud_targets(ds, ldr="ldr", dfr="dfr", engine="compiled", n_threads=3)
    np.testing.assert_array_equal(a, b)
    assert set(np.unique(a)) == set(range(8))
    one = classify_cloud_targets(
        ds, ldr="ldr", dfr="dfr", engine="compiled", n_threads=1
    )
    np.testing.assert_array_equal(one, b)


@compiled_only
def test_kernel_input_checks():
    from radarx.retrieve import _cloud_class

    one = np.ones(3)
    params = np.zeros(7)
    with pytest.raises(ValueError, match="1-D"):
        _cloud_class.classify(np.ones((2, 2)), one, one, one, params)
    with pytest.raises(ValueError, match="same length"):
        _cloud_class.classify(one, np.ones(2), one, one, params)
    with pytest.raises(ValueError, match="7 values"):
        _cloud_class.classify(one, one, one, one, np.zeros(3))


# ---- the temperature -------------------------------------------------------------


def test_temperature_in_celsius_or_kelvin_and_as_a_number():
    ds = gates([0.0, 0.0, 0.0], [-10.0, 1.0, 10.0])
    reference = classify_cloud_targets(ds)
    celsius = xr.DataArray(
        ds.temperature.values - K, dims="range", attrs={"units": "degC"}
    )
    np.testing.assert_array_equal(classify_cloud_targets(ds, celsius), reference)
    guessed = xr.DataArray(
        ds.temperature.values - K, dims="range"
    )  # no units: below 150 is Celsius
    np.testing.assert_array_equal(classify_cloud_targets(ds, guessed), reference)
    kelvin = xr.DataArray(ds.temperature.values, dims="range", attrs={"units": "K"})
    np.testing.assert_array_equal(classify_cloud_targets(ds, kelvin), reference)
    np.testing.assert_array_equal(classify_cloud_targets(ds, "temperature"), reference)
    warm = classify_cloud_targets(ds, 12.0)
    assert (warm == DRIZZLE).all()
    cold = classify_cloud_targets(ds, 261.0)
    assert (cold == ICE).all()
    renamed = ds.rename(temperature="T")
    np.testing.assert_array_equal(classify_cloud_targets(renamed), reference)


def sweep(elevation=90.0, n_gates=120):
    rng = 100.0 + 50.0 * np.arange(n_gates)
    return xr.Dataset(
        {"DBZ": (("azimuth", "range"), np.full((3, n_gates), 0.0))},
        coords={
            "azimuth": [0.0, 120.0, 240.0],
            "range": rng,
            "elevation": ("azimuth", np.full(3, elevation)),
            "altitude": 50.0,
        },
    )


def test_a_temperature_profile_is_put_on_the_gates():
    height = np.arange(0.0, 8001.0, 500.0)
    profile = xr.Dataset(
        {"temperature": ("height", 288.15 - 6.5e-3 * height)}, coords={"height": height}
    )
    ds = sweep()
    out = classify_cloud_targets(ds, profile)
    gate_height = ds.range.values + 50.0  # vertical beam from the altitude of 50 m
    temp = 288.15 - 6.5e-3 * gate_height - K
    expected = np.where(temp < 0.0, ICE, np.where(temp <= 2.0, MELTING, DRIZZLE))
    np.testing.assert_array_equal(out.isel(azimuth=0), expected)
    assert set(np.unique(out)) == {ICE, MELTING, DRIZZLE}
    # a temperature profile in Celsius
    celsius = profile.assign(temperature=("height", profile.temperature.values - K))
    celsius.temperature.attrs["units"] = "degC"
    np.testing.assert_array_equal(classify_cloud_targets(ds, celsius), out)
    # given gate heights
    heights = xr.DataArray(gate_height, dims="range", coords={"range": ds.range})
    np.testing.assert_array_equal(
        classify_cloud_targets(ds, profile, height=heights), out
    )
    # a lower elevation reaches a given temperature at a longer range
    slant = classify_cloud_targets(sweep(elevation=10.0), profile)
    assert int((slant == ICE).sum()) < int((out == ICE).sum())


def test_temperature_errors():
    ds = gates([0.0], [1.0]).drop_vars("temperature")
    with pytest.raises(KeyError, match="no temperature"):
        classify_cloud_targets(ds)
    with pytest.raises(KeyError, match="nope"):
        classify_cloud_targets(ds, "nope")
    profile = xr.Dataset({"temperature": ("level", [280.0, 270.0])})
    with pytest.raises(ValueError, match="height"):
        classify_cloud_targets(sweep(), profile)
    backwards = xr.Dataset(
        {"temperature": ("height", [270.0, 280.0, 290.0])},
        coords={"height": [200.0, 100.0, 0.0]},
    )
    with pytest.raises(ValueError, match="increasing"):
        classify_cloud_targets(sweep(), backwards)
    with pytest.raises(KeyError, match="no temperature"):
        classify_cloud_targets(ds, [1.0])


# ---- fields, objects and options ---------------------------------------------------------


def test_data_array_input_and_the_reflectivity_names():
    ds = gates([0.0, 0.0], [10.0, -10.0])
    out = classify_cloud_targets(ds.DBZ, ds.temperature)
    np.testing.assert_array_equal(out, [DRIZZLE, ICE])
    cloudnet = ds.rename(DBZ="Z")
    np.testing.assert_array_equal(classify_cloud_targets(cloudnet), out)
    other = ds.rename(DBZ="strange")
    with pytest.raises(KeyError, match="no reflectivity"):
        classify_cloud_targets(other)
    np.testing.assert_array_equal(classify_cloud_targets(other, dbz="strange"), out)
    np.testing.assert_array_equal(classify_cloud_targets(other, dbz=other.strange), out)
    with pytest.raises(KeyError, match="nope"):
        classify_cloud_targets(ds, dbz="nope")
    with pytest.raises(ValueError, match="DataArray"):
        classify_cloud_targets(ds.DBZ, ds.temperature, ldr="ldr")
    with pytest.raises(KeyError, match="nope"):
        classify_cloud_targets(ds, ldr="nope")


def test_the_dataset_ldr_may_be_given_by_name_or_array():
    ds = gates([0.0], [10.0], ldr=[-15.0])
    by_name = classify_cloud_targets(ds, ldr="ldr")
    by_array = classify_cloud_targets(ds, ldr=ds.ldr)
    assert int(by_name[0]) == INSECTS and int(by_array[0]) == INSECTS


def test_a_volume_skips_the_sweeps_without_a_reflectivity():
    good = sweep().assign(temperature=("range", np.full(120, 270.0)))
    empty = xr.Dataset({"x": ("t", np.arange(3.0))})
    tree = xr.DataTree.from_dict(
        {"/": xr.Dataset(), "/sweep_0": good, "/sweep_1": empty}
    )
    out = classify_cloud_targets(tree)
    assert list(out.children) == ["sweep_0"]
    expected = classify_cloud_targets(good)
    xr.testing.assert_equal(out["sweep_0"].to_dataset().TARGET_CLASS, expected)
    nothing = xr.DataTree.from_dict({"/": xr.Dataset(), "/sweep_0": empty})
    with pytest.raises(KeyError, match="no sweep"):
        classify_cloud_targets(nothing)


def test_dask_input_stays_lazy_and_gives_the_same_result():
    pytest.importorskip("dask")
    ds = sweep().assign(temperature=("range", np.linspace(285.0, 255.0, 120)))
    expected = classify_cloud_targets(ds)
    lazy = classify_cloud_targets(ds.chunk({"azimuth": 1}))
    assert lazy.chunks is not None
    xr.testing.assert_equal(lazy.compute(), expected)


def test_accessor_provenance_and_the_input_is_unchanged():
    ds = gates([0.0, -30.0, 5.0], [10.0, -10.0, 1.0])
    before = ds.copy(deep=True)
    out = ds.radarx.classify_cloud_targets(ldr=None)
    xr.testing.assert_equal(out, classify_cloud_targets(ds))
    xr.testing.assert_identical(ds, before)
    assert "Cloudnet" in out.attrs["radarx_method"]
    assert "10.1175/BAMS-88-6-883" in out.attrs["radarx_references"]
    via_array = ds.DBZ.radarx.classify_cloud_targets(ds.temperature)
    xr.testing.assert_equal(via_array, out)


def test_option_and_type_errors():
    ds = gates([0.0], [1.0])
    with pytest.raises(ValueError, match="engine"):
        classify_cloud_targets(ds, engine="gpu")
    with pytest.raises(ValueError, match="pair"):
        classify_cloud_targets(ds, melting_temperature=1.0)
    with pytest.raises(ValueError, match="increase"):
        classify_cloud_targets(ds, melting_temperature=(2.0, 0.0))
    with pytest.raises(TypeError, match="obj"):
        classify_cloud_targets(np.zeros(3))


def test_the_kernel_is_not_required_for_the_numpy_engine(monkeypatch):
    monkeypatch.setattr(cc, "HAS_COMPILED_KERNEL", False)
    ds = gates([0.0], [-5.0])
    assert int(classify_cloud_targets(ds)[0]) == ICE
    with pytest.raises(ImportError):
        classify_cloud_targets(ds, engine="compiled")


# ---- real data: the Cloudnet classification of Lindenberg, 1 May 2023 ------------------------

CLOUDNET = {
    "categorize": (
        "https://cloudnet.fmi.fi/api/download/product/231c0919-f399-4fea-aa04-3435754f96b9/"
        "20230501_lindenberg_categorize.nc"
    ),
    "classification": (
        "https://cloudnet.fmi.fi/api/download/product/9eac625e-a74b-478c-b5f6-ff66aaf6771d/"
        "20230501_lindenberg_classification.nc"
    ),
}


@pytest.fixture(scope="module")
def cloudnet(tmp_path_factory):
    """The categorization and classification files of Cloudnet, Lindenberg, 2023-05-01."""
    import urllib.request

    root = tmp_path_factory.mktemp("cloudnet")
    out = {}
    for name, url in CLOUDNET.items():
        error = None
        for _ in range(3):  # the server sometimes ends a transfer early
            try:
                path = root / f"{name}.nc"
                urllib.request.urlretrieve(url, path)  # noqa: S310 - fixed https URLs
                out[name] = xr.open_dataset(path).load()
                break
            except Exception as err:  # offline, or the files are not available
                error = err
        else:
            pytest.skip(f"Cloudnet data unavailable: {error}")
    return out


def test_the_classification_agrees_with_cloudnet_on_real_data(cloudnet):
    cat, cls = cloudnet["categorize"], cloudnet["classification"]
    section = xr.Dataset(
        {
            "Z": cat.Z,
            "ldr": cat.ldr,
            "Tw": cat.Tw,
        }  # the wet-bulb temperature, as Cloudnet
    ).rename(height="range")
    out = classify_cloud_targets(section, "Tw", ldr="ldr")
    # Cloudnet class -> radarx class
    mapping = {
        0: NO_ECHO,
        1: DROPLETS,
        2: DRIZZLE,
        3: DRIZZLE,
        4: ICE,
        5: ICE,
        6: MELTING,
        7: MELTING,
        8: NO_ECHO,
        9: INSECTS,
        10: INSECTS,
    }
    reference = np.vectorize(
        lambda v: mapping.get(int(v), NO_ECHO) if np.isfinite(v) else NO_ECHO
    )(cls.target_classification.values)
    echo = np.isfinite(section.Z.values)
    assert echo.sum() > 100000
    agree = (out.values == reference)[echo]
    assert agree.mean() > 0.88
    # ice, the most frequent class, is found
    ice = (reference == ICE) & echo
    assert ice.sum() > 50000
    assert (out.values[ice] == ICE).mean() > 0.97
    # most of the insects of Cloudnet (it has a lidar and a model) are found too
    insects = (reference == INSECTS) & echo
    assert (out.values[insects] == INSECTS).mean() > 0.7
    # without the depolarization ratio there are no insects, and they are called something else
    blind = classify_cloud_targets(section, "Tw")
    assert not (blind == INSECTS).any()
