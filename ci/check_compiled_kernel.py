#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Fail if radarx was installed without its compiled kernels.

The extension is optional at build time (radarx falls back to NumPy), so a
broken compiler setup would otherwise go unnoticed in wheels and CI.
"""

import sys

import numpy as np

from radarx.grid import cone
from radarx.retrieve import shear

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
