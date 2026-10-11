#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the dtype policy and the unit strings of the results
==============================================================

The main functions run on a synthetic polarimetric volume, once with
``float32`` and once with ``float64`` fields. Every floating point variable of
a result must be ``float32`` (see :mod:`radarx._dtypes`), whatever the
precision of the input, and every ``units`` attribute must be written in the
UDUNITS style: a product of unit symbols with integer exponents (``m s-1``,
``degree km-1``, ``dB km-1``, ``m-3 mm-1``, ``kg m-2``).
"""
import ast
import re
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import radarx as rx
from radarx import retrieve as rt
from radarx._dtypes import FIELD_DTYPE, as_field, fields_as_float32

# ---- the unit grammar -------------------------------------------------------

#: unit symbols that radarx writes; the CF names for latitude and longitude
SYMBOLS = {
    "m", "km", "mm", "s", "h", "min", "K", "degC", "Pa", "hPa", "kg", "g", "dB",
    "dBZ", "dBW", "degree", "degrees_north", "degrees_east", "J", "W", "V",
}  # fmt: skip
FACTOR = re.compile(r"^([A-Za-z_]+)(-?\d+)?$")


def valid_units(text):
    """Is ``text`` a UDUNITS product of symbols with integer exponents?"""
    if text in ("1", "%"):
        return True
    if not isinstance(text, str) or not text or text != text.strip():
        return False
    for factor in text.split(" "):
        match = FACTOR.match(factor)
        if match is None or match.group(1) not in SYMBOLS:
            return False
    return True


@pytest.mark.parametrize(
    "text",
    [
        "m s-1", "degree km-1", "dB km-1", "m-3 mm-1", "kg m-2", "s-1", "dBZ",
        "degree", "1", "%", "mm6 m-3", "m2 s-2", "degrees_north", "J m-1 s-1",
        "kg kg-1 s-1", "K h-1", "min-2",
    ],
)  # fmt: skip
def test_unit_grammar_accepts_udunits_strings(text):
    assert valid_units(text)


@pytest.mark.parametrize(
    "text",
    [
        "degrees/km", "dB/km", "degrees", "meters", "unitless", "percent",
        "meters per second", "m/s", "m^2", "m s^-1", "m-3 mm-(1+mu)", "", " m",
        "m  s-1", "mm h-1 ", None,
    ],
)  # fmt: skip
def test_unit_grammar_rejects_other_spellings(text):
    assert not valid_units(text)


# ---- the synthetic volume ---------------------------------------------------


def _sweep(elev, dtype, n_az=72, n_rng=160, seed=0):
    rng = np.random.default_rng(seed)
    az = (np.arange(n_az) + 0.5) * 360.0 / n_az
    r = 500.0 + 250.0 * np.arange(n_rng)
    rr, aa = np.meshgrid(r, az)
    dbz = 35 + 12 * np.sin(rr / 9000.0) * np.cos(np.radians(aa) * 2)
    dbz = dbz + rng.normal(0, 1, rr.shape)
    zdr = 0.8 + 0.03 * (dbz - 30) + rng.normal(0, 0.1, rr.shape)
    rho = np.clip(0.985 + rng.normal(0, 0.004, rr.shape), 0.5, 1.0)
    kdp = np.clip(0.02 * (dbz - 20), 0, None) * 0.1
    phi = np.cumsum(2 * kdp * 0.25, axis=1) + 20 + rng.normal(0, 1, rr.shape)
    nyq = 12.0
    vr = 20 * np.sin(np.radians(aa) - 1.0) * np.cos(np.radians(elev))
    vr = ((vr + rng.normal(0, 0.5, rr.shape) + nyq) % (2 * nyq)) - nyq
    dims = ("azimuth", "range")

    def v(a, units):
        return (dims, a.astype(dtype), {"units": units})

    ds = xr.Dataset(
        {
            "DBZH": v(dbz, "dBZ"),
            "ZDR": v(zdr, "dB"),
            "RHOHV": v(rho, "1"),
            "PHIDP": v(phi, "degree"),
            "VRADH": v(vr, "m s-1"),
            "SNRH": v(dbz - 10 + rng.normal(0, 1, rr.shape), "dB"),
        },
        coords={
            "azimuth": ("azimuth", az, {"units": "degree"}),
            "range": ("range", r, {"units": "m"}),
            "elevation": ("azimuth", np.full(n_az, elev), {"units": "degree"}),
            "time": (
                "azimuth",
                np.datetime64("2024-05-01T12:00:00")
                + (np.arange(n_az) * 20).astype("timedelta64[ms]"),
            ),
            "latitude": 35.0,
            "longitude": -97.0,
            "altitude": 300.0,
        },
    )
    ds["sweep_fixed_angle"] = elev
    return ds


def _volume(dtype):
    nodes = {"/": xr.Dataset({"latitude": 35.0, "longitude": -97.0, "altitude": 300.0})}
    for i, elev in enumerate((0.5, 1.5, 2.5, 3.5)):
        nodes[f"sweep_{i}"] = _sweep(elev, dtype, seed=i)
    return xr.DataTree.from_dict(nodes).xradar.georeference()


def _cases(dtype):
    vol = _volume(dtype)
    sweep = vol["sweep_0"].to_dataset()
    kdp = rt.estimate_kdp(sweep, phidp="PHIDP", rhohv="RHOHV", dbz="DBZH")
    sweep_kdp = sweep.assign(KDP=kdp["KDP"], PHIDP_processed=kdp["PHIDP_processed"])
    grid = rx.grid.grid_radar(
        vol,
        ["DBZH", "ZDR", "RHOHV"],
        z_lim=(0, 8000),
        x_lim=(-60e3, 60e3),
        y_lim=(-60e3, 60e3),
        x_step=3000,
        y_step=3000,
        z_step=1000,
    )
    profile = rt.qvp(vol)
    temperature = xr.DataArray(
        np.full(sweep.DBZH.shape, 10.0), dims=sweep.DBZH.dims, coords=sweep.DBZH.coords
    )
    return {
        "estimate_kdp": lambda: kdp,
        "qvp": lambda: profile,
        "rd_qvp": lambda: rt.rd_qvp(vol),
        "echo_mask": lambda: rt.echo_mask(sweep),
        "dealias_velocity": lambda: rt.dealias_velocity(
            sweep, "VRADH", nyquist_velocity=12.0
        ),
        "llsd": lambda: rt.llsd(sweep, "VRADH"),
        "rain_rate_z": lambda: rt.rain_rate_z(sweep),
        "rain_rate_kdp": lambda: rt.rain_rate_kdp(sweep_kdp),
        "rain_rate_z_zdr": lambda: rt.rain_rate_z_zdr(sweep),
        "snr": lambda: rt.snr(sweep),
        "noise_level": lambda: rt.noise_level(sweep),
        "correct_rhohv_noise": lambda: rt.correct_rhohv_noise(sweep),
        "correct_zdr_noise": lambda: rt.correct_zdr_noise(sweep),
        "melting_layer": lambda: rt.melting_layer(profile),
        "create_cappi": lambda: rt.create_cappi(
            vol, 2000.0, data_vars=["DBZH"], x_res=3000, y_res=3000
        ),
        "grid_radar": lambda: grid,
        "grid_gates": lambda: rx.grid.grid_gates(
            vol,
            ["DBZH"],
            x=np.arange(-30e3, 30e3, 3000.0),
            y=np.arange(-30e3, 30e3, 3000.0),
            z=np.array([1000.0, 2000.0]),
            roi=2000.0,
        ),
        "vil": lambda: rt.vil(grid),
        "echo_top": lambda: rt.echo_top(grid),
        "vil_density": lambda: rt.vil_density(grid),
        "liquid_water_content": lambda: rt.liquid_water_content(grid),
        "dsd": lambda: rt.dsd(sweep_kdp, band="S"),
        "dsd_bayesian": lambda: rt.dsd_bayesian(sweep_kdp, band="S"),
        "hid": lambda: rt.hid(sweep_kdp, temperature=temperature, band="S"),
        "correct_attenuation": lambda: rt.correct_attenuation(
            sweep, phidp="PHIDP", band="S"
        ),
        "zdr_offset": lambda: rt.zdr_offset(sweep),
        "phidp_offset": lambda: rt.phidp_offset(sweep),
        "vad_profile": lambda: rt.vad_profile(vol, "VRADH"),
        "azimuthal_shear_layer_max": lambda: rt.azimuthal_shear_layer_max(vol, "VRADH"),
        "hail_products": lambda: rt.hail_products(
            grid.assign(DBZH=grid.DBZH + 15),
            freezing_level=3500.0,
            minus20_level=6500.0,
        ),
        "rotation_couplets": lambda: rt.rotation_couplets(sweep, "VRADH"),
        "estimate_motion": lambda: rt.estimate_motion(
            grid.isel(z=0).assign_coords(time=np.datetime64("2024-05-01T12:00")),
            grid.isel(z=0).assign_coords(time=np.datetime64("2024-05-01T12:05")),
        ),
    }


#: results that keep float64: one row per object, and the motion of a storm
TABLES = {"rotation_couplets", "estimate_motion"}
#: corrected copies of an input field keep the precision of the input
CORRECTED_INPUT = {"dealias_velocity"}


def _datasets(result):
    if isinstance(result, xr.DataTree):
        node = result["sweep_0"] if "sweep_0" in result.children else result
        return [node.to_dataset()]
    if isinstance(result, xr.DataArray):
        return [result.to_dataset(name=result.name or "result")]
    return [result]


@pytest.fixture(scope="module", params=["float32", "float64"])
def cases(request):
    dtype = np.dtype(request.param)
    return dtype, _cases(dtype)


FUNCTIONS = list(_cases(np.dtype("float32")))


@pytest.mark.parametrize("name", FUNCTIONS)
def test_fields_are_float32_and_units_are_udunits(cases, name):
    dtype, calls = cases
    result = calls[name]()
    expected = dtype if name in CORRECTED_INPUT else FIELD_DTYPE
    checked = 0
    for ds in _datasets(result):
        for var, da in ds.data_vars.items():
            if da.dtype.kind == "f":
                checked += 1
                if name not in TABLES:
                    assert da.dtype == expected, f"{name}.{var} is {da.dtype}"
                assert valid_units(
                    da.attrs.get("units")
                ), f"{name}.{var}: {da.attrs.get('units')!r}"
            elif "units" in da.attrs:
                assert valid_units(da.attrs["units"]), f"{name}.{var}"
    assert checked


def test_as_field_rounds_floats_and_keeps_integers():
    assert as_field(np.ones(3)).dtype == FIELD_DTYPE
    ints = np.arange(3)
    assert as_field(ints) is ints
    flags = xr.DataArray(np.array([True, False]))
    assert as_field(flags) is flags
    single = np.ones(3, dtype=np.float32)
    assert as_field(single) is single


def test_fields_as_float32_keeps_attributes_and_coordinates():
    ds = xr.Dataset(
        {
            "a": ("x", np.arange(3.0), {"units": "m"}),
            "n": ("x", np.arange(3)),
            "b": ("x", np.ones(3, dtype=np.float32)),
        },
        coords={"x": np.array([0.0, 1.0, 2.0])},
        attrs={"title": "t"},
    )
    out = fields_as_float32(ds)
    assert out.a.dtype == FIELD_DTYPE and out.b.dtype == FIELD_DTYPE
    assert out.n.dtype == ds.n.dtype
    assert out.a.attrs == {"units": "m"} and out.attrs == {"title": "t"}
    assert out.x.dtype == np.float64


def test_dask_fields_stay_lazy():
    dask = pytest.importorskip("dask.array")
    da = xr.DataArray(dask.ones(4, chunks=2), dims="x")
    out = as_field(da)
    assert out.dtype == FIELD_DTYPE
    assert out.chunks is not None


# ---- the source ---------------------------------------------------------------

OLD_SPELLINGS = {
    "degrees/km", "dB/km", "deg/km", "unitless", "percent", "meters per second",
    "m-3 mm-(1+mu)", "degrees", "meters",
}  # fmt: skip
#: the IMD reader keeps the CfRadial spellings of the xradar readers; the profile
#: code reads both spellings of an angle unit
READS_OR_MIMICS = {"imd.py", "vertical_profiles.py"}


def _units_values(tree):
    """String constants that are the value of a ``units`` key, keyword or item."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            pairs = zip(node.keys, node.values)
            values = [v for k, v in pairs if getattr(k, "value", None) == "units"]
        elif isinstance(node, ast.keyword) and node.arg == "units":
            values = [node.value]
        elif isinstance(node, ast.Assign):
            subscript = [
                t
                for t in node.targets
                if isinstance(t, ast.Subscript)
                and getattr(t.slice, "value", None) == "units"
            ]
            values = [node.value] if subscript else []
        else:
            continue
        for value in values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                yield node.lineno, value.value


def test_units_written_by_the_source_are_udunits():
    root = Path(rx.__file__).parent
    found = 0
    for path in sorted(root.rglob("*.py")):
        if path.name == "imd.py":
            continue
        for lineno, text in _units_values(ast.parse(path.read_text())):
            found += 1
            assert valid_units(text), f"{path.relative_to(root)}:{lineno}: {text!r}"
    assert found > 300


def test_old_unit_spellings_do_not_appear_in_the_source():
    root = Path(rx.__file__).parent
    for path in sorted(root.rglob("*.py")):
        if path.name in READS_OR_MIMICS:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and node.value in OLD_SPELLINGS:
                pytest.fail(f"{path.relative_to(root)}:{node.lineno}: {node.value!r}")
