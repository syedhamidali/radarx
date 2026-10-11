#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
The input of a radarx function is never changed
===============================================

``xarray.apply_ufunc`` with several DataArray inputs and ``keep_attrs=False``
strips the attributes (``units``, ``long_name`` and the like) from the
coordinates of its first input in xarray 2024.10, the oldest release radarx
supports. The retrievals that combine fields pass shallow copies, so the
sweep of the caller keeps the attributes of ``range`` and ``azimuth``.
"""

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx.retrieve import (
    beam_blockage,
    beam_path,
    correct_attenuation,
    ensemble_retrieval,
    ensemble_statistics,
    gbvtd,
    measurement_error,
    perturb_radar,
    rain_rate,
    rain_rate_a,
    rain_rate_z_zdr,
    snr,
    tc_center,
    z_offset,
)

from .test_beam_blockage import bell, geographic_dem

NAZ, NRNG = 12, 60


def make_sweep():
    """A sweep with attributes on the fields and on every coordinate."""
    rng = np.random.default_rng(3)
    dims = ("azimuth", "range")

    def field(lo, hi, **attrs):
        return (dims, rng.uniform(lo, hi, (NAZ, NRNG)), attrs)

    return xr.Dataset(
        {
            "DBZH": field(15, 45, units="dBZ"),
            "ZDR": field(0.0, 2.5, units="dB"),
            "KDP": field(0.0, 2.0, units="degree km-1"),
            "AH": field(0.0, 0.05, units="dB km-1"),
            "PHIDP": (dims, np.cumsum(rng.uniform(0, 0.4, (NAZ, NRNG)), axis=1)),
            "RHOHV": field(0.95, 1.0, units="1"),
            "TEMP": field(5.0, 20.0, units="degC"),
            "HID": (
                dims,
                rng.integers(1, 4, (NAZ, NRNG)).astype(np.int8),
                {
                    "flag_values": np.arange(1, 4, dtype=np.int8),
                    "flag_meanings": "drizzle rain big_drops",
                },
            ),
            "VRADH": field(-12.0, 12.0, units="m s-1"),
        },
        coords={
            "azimuth": (
                "azimuth",
                np.linspace(0.5, 359.5, NAZ),
                {"long_name": "azimuth angle", "units": "degree", "axis": "azimuth"},
            ),
            "range": (
                "range",
                500.0 + 250.0 * np.arange(NRNG),
                {"long_name": "range", "units": "m", "meters_between_gates": 250.0},
            ),
            "elevation": ("azimuth", np.full(NAZ, 0.5), {"units": "degree"}),
        },
    )


CALLS = {
    "rain_rate_a_temperature_field": lambda ds: rain_rate_a(
        ds, a="AH", band="S", temperature="TEMP"
    ),
    "rain_rate_z_zdr": lambda ds: rain_rate_z_zdr(ds, dbz="DBZH", zdr="ZDR", band="S"),
    "rain_rate_blend": lambda ds: rain_rate(
        ds,
        hid="HID",
        band="S",
        class_relations={"drizzle": "z", "rain": "a", "big_drops": "z_zdr"},
        a="AH",
        temperature="TEMP",
    ),
    "snr": lambda ds: snr(ds, noise=-30.0, dbz="DBZH"),
    "perturb_radar": lambda ds: perturb_radar(ds, members=2),
    "measurement_error": lambda ds: measurement_error(ds, snr=20.0),
    "ensemble_retrieval": lambda ds: ensemble_retrieval(
        ds, "rain_rate_z_zdr", members=2, band="S"
    ),
    "ensemble_retrieval_attenuation": lambda ds: ensemble_retrieval(
        ds, "rain_rate_a", members=2, band="S"
    ),
    "ensemble_statistics": lambda ds: ensemble_statistics(perturb_radar(ds, members=3)),
    "correct_attenuation": lambda ds: correct_attenuation(
        ds, dbz="DBZH", zdr="ZDR", phidp="PHIDP", rhohv="RHOHV", band="S"
    ),
    "z_offset": lambda ds: z_offset(ds, dbz="DBZH", zdr="ZDR", kdp="KDP", band="S"),
    "tc_center": lambda ds: tc_center(
        ds, first_guess=(3000.0, 4000.0), radii=np.array([1500.0, 2000.0]), guess_grid=2
    ),
    "gbvtd": lambda ds: gbvtd(
        ds, (3000.0, 4000.0), radii=np.array([1500.0, 2000.0]), min_points=5
    ),
    "beam_blockage": lambda ds: beam_blockage(
        ds, geographic_dem(bell()), site=(30.0, 78.0, 500.0), beamwidth=1.0
    ),
    "beam_path": lambda ds: beam_path(
        ds,
        (
            np.arange(0.0, 12001.0, 500.0),
            313.0 - 39.2e-3 * np.arange(0.0, 12001.0, 500.0),
        ),
        radar_height=500.0,
    ),
}


@pytest.mark.parametrize("name", sorted(CALLS))
def test_input_sweep_is_unchanged(name):
    ds = make_sweep()
    expected = make_sweep()
    out = CALLS[name](ds)
    for coord in ("azimuth", "range", "elevation"):
        assert ds[coord].attrs == expected[coord].attrs, coord
    xr.testing.assert_identical(ds, expected)
    # the product does not share the coordinate variables of the input either
    if isinstance(out, xr.DataArray):
        for coord in ("azimuth", "range"):
            if coord in out.coords:
                assert out[coord].variable is not ds[coord].variable
