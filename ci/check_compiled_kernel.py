#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Fail if radarx was installed without its compiled kernels.

The extensions are optional at build time (radarx falls back to NumPy), so a
broken compiler setup would otherwise go unnoticed in wheels and CI.
"""

import sys

import numpy as np

from radarx.grid import cone
from radarx.retrieve import advection, dealias, shear

if not cone.HAS_COMPILED_KERNEL:
    sys.exit("radarx.grid._cone was not built: the compiled kernel is missing")

# a tiny call to make sure the kernel loads and runs
azimuth = np.arange(0.5, 360.0, 1.0)
rng = np.arange(125.0, 10_000.0, 250.0)
data = np.ones((azimuth.size, rng.size))
out = cone._cone.grid_cones(
    np.array([0.0, 2000.0]),
    np.array([0.0, 2000.0]),
    np.array([100.0]),
    [data, data],
    [azimuth, azimuth],
    [np.full(azimuth.size, 0.5), np.full(azimuth.size, 5.0)],
    [rng, rng],
    0.0,
    fill_below=True,
)
if out.shape != (1, 2, 2) or not np.isfinite(out).any():
    sys.exit(f"compiled kernel returned an unexpected result: {out}")
print("compiled cone-gridding kernel OK")

if not shear.HAS_COMPILED_KERNEL:
    sys.exit("radarx.retrieve._shear was not built: the compiled kernel is missing")

# radial velocity growing linearly with range: divergence 1e-3 s-1, no shear
velocity = np.broadcast_to(1e-3 * rng, data.shape)
(out,) = shear._shear.llsd([velocity], [azimuth], [rng], 750.0, 2500.0)
if out.shape != (2,) + data.shape or not np.allclose(out[1], 1e-3, rtol=1e-4):
    sys.exit(f"compiled LLSD kernel returned an unexpected result: {out}")
print("compiled LLSD kernel OK")
if not dealias.HAS_COMPILED_KERNEL:
    sys.exit("radarx.retrieve._dealias was not built: the compiled kernel is missing")
nyquist = 10.0
truth = np.tile(np.linspace(-25.0, 25.0, 50), (36, 1))
aliased = np.mod(truth + nyquist, 2 * nyquist) - nyquist
links = np.ones(36, dtype=np.uint8)
((folds, comps),) = dealias._dealias.region_folds([aliased], [links], [nyquist])
if folds.shape != aliased.shape or (comps < 0).any():
    sys.exit("compiled dealiasing kernel returned an unexpected result")
print("compiled dealiasing kernel OK")
if not advection.HAS_COMPILED_KERNEL:
    sys.exit("radarx.retrieve._advection was not built: the compiled kernel is missing")
planes = np.arange(12.0).reshape(1, 3, 4)
rows = np.broadcast_to(np.arange(3.0)[None, :, None], (1, 3, 4))
cols = np.broadcast_to(np.arange(4.0)[None, None, :] - 1.0, (1, 3, 4))
moved = advection._advection.advect(planes, rows, cols, 1, 0.5, 0)
if moved.shape != (1, 1, 3, 4) or not np.allclose(
    moved[0, 0, :, 1:], planes[0, :, :-1]
):
    sys.exit(f"compiled advection kernel returned an unexpected result: {moved}")
print("compiled advection kernel OK")
from radarx.retrieve import kdp  # noqa: E402

if not kdp.HAS_COMPILED_KERNEL:
    sys.exit("radarx.retrieve._kdp was not built: the compiled kernel is missing")
gates = np.arange(200)
phi = np.tile(30.0 + 0.5 * gates, (3, 1))
phi_out, kdp_out, offset, sign = kdp._kdp.process_phidp(
    [phi],
    [None],
    [None],
    [0.25],
    [4],
    [4],
    [4],
    [12],
    0,
    0.85,
    0.9,
    10,
    0,
    0.0,
    10,
    4.0,
    40.0,
    -2.0,
    20.0,
)
if not np.allclose(kdp_out[0][:, 20:-20], 1.0) or not np.allclose(
    offset[0], 32.25, atol=0.01
):
    sys.exit("compiled KDP kernel returned an unexpected result")
print("compiled KDP kernel OK")

from radarx.retrieve import vertical_profiles as qvp  # noqa: E402

if not qvp.HAS_COMPILED_KERNEL:
    sys.exit("radarx.retrieve._qvp was not built: the compiled kernel is missing")

values, counts = qvp._qvp.azimuthal_reduce(
    [[np.ones((azimuth.size, rng.size), dtype=np.float32)]], [[]], [], [1], [1]
)
if values[0].shape != (1, rng.size) or not np.allclose(values[0], 1.0):
    sys.exit(f"compiled QVP kernel returned an unexpected result: {values[0]}")
print("compiled QVP kernel OK")

from radarx.io import sounding  # noqa: E402

if not sounding.HAS_COMPILED_KERNEL:
    sys.exit("radarx.io._sounding was not built: the compiled kernel is missing")
z = np.array([[0.0, 1000.0, 2000.0]])
values = np.array([[[280.0, 270.0, 260.0]]])
out = sounding._sounding.interp_vertical(
    z, values, np.array([[500.0]]), np.array([False]), False, 0
)
if out.shape != (1, 1, 1) or not np.isclose(out[0, 0, 0], 275.0):
    sys.exit(f"compiled sounding kernel returned an unexpected result: {out}")
print("compiled sounding kernel OK")
