#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Synthetic IQ time series with known moments for the Level I tests."""
import numpy as np
import xarray as xr

from radarx import level1

WAVELENGTH = 0.1  # m, S band
PRT = 1e-3  # s
NYQUIST = WAVELENGTH / (4 * PRT)  # 25 m s-1


def gates(
    n=600,
    vmax=12.0,
    width=2.0,
    snr_db=25.0,
    n_pulses=64,
    clutter=0.0,
    clutter_width=0.2,
    seed=1,
    noise=1.0,
):
    """``n`` gates of one ray with velocities from -vmax to vmax; returns (iq, truth)."""
    truth = np.linspace(-vmax, vmax, n)
    iq = level1.simulate_iq(
        truth,
        width=width,
        power=noise * 10 ** (0.1 * snr_db),
        noise=noise,
        clutter_power=clutter,
        clutter_width=clutter_width,
        n_pulses=n_pulses,
        wavelength=WAVELENGTH,
        prt=PRT,
        dims=("range",),
        seed=seed,
    )
    return iq, truth


def sweep(n_az=4, n_rng=30, n_pulses=32, seed=0, **kwargs):
    """A small sweep of IQ with an azimuth and a range dimension."""
    rng = np.random.default_rng(seed)
    velocity = rng.uniform(-15, 15, (n_az, n_rng))
    iq = level1.simulate_iq(
        velocity,
        width=1.5,
        power=200.0,
        noise=1.0,
        n_pulses=n_pulses,
        wavelength=WAVELENGTH,
        prt=PRT,
        seed=seed,
        **kwargs,
    )
    iq = iq.assign_coords(
        azimuth=("azimuth", np.arange(n_az) * 1.0),
        range=("range", 100.0 + 50 * np.arange(n_rng)),
    )
    return iq, velocity


def volume(iq, names=("sweep_0", "sweep_1")):
    """A DataTree with IQ in the sweeps ``names``; a sweep ``sweep_9`` has none."""
    nodes = {"/": xr.Dataset(attrs={"instrument_name": "synthetic"})}
    for name in names:
        nodes[name] = xr.Dataset({"IQ": iq})
    nodes["sweep_9"] = xr.Dataset({"DBZH": (("azimuth", "range"), np.zeros((2, 3)))})
    return xr.DataTree.from_dict(nodes)
