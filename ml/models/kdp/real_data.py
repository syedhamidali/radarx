#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Real rays for fine-tuning (no truth: physics terms only).

    python real_data.py real_rays.npz KGWX20220330_210004_V06 ...

NEXRAD Level II volumes (e.g. from the public ``unidata-nexrad-level2`` AWS
bucket) are read with xradar; the no-data codes are masked (DBZH <= -32,
ZDR <= -12.9, RHOHV <= 0.21). Every sweep with a differential phase is
featurised with the same function as at inference (sweep-level sign test and
offset), and cut into windows of 512 gates with at least 20 % valid gates.
"""

from __future__ import annotations

import sys

import numpy as np
import xradar as xd

from radarx.retrieve.kdp import (
    _decide_sign,
    _gate_spacing,
    _half_gates,
    _mask_numpy,
    _ml_features,
)

N_GATES = 512
PER_VOLUME = 4000  # windows kept per volume


def nexrad_sweeps(path):
    """(phi, rho, z, dr) of every sweep of a NEXRAD volume with PHIDP."""
    dtree = xd.io.open_nexradlevel2_datatree(path)
    for name in dtree.children:
        if not name.startswith("sweep"):
            continue
        ds = dtree[name].to_dataset()
        if "PHIDP" not in ds or "RHOHV" not in ds:
            continue
        z = ds.DBZH.where(ds.DBZH > -32).values.astype(float)
        rho = ds.RHOHV.where(ds.RHOHV > 0.21).values.astype(float)
        phi = ds.PHIDP.values.astype(float)
        yield phi, rho, z, _gate_spacing(ds.range.values)


def featurise(phi, rho, z, dr):
    params = {
        "rhohv_min": 0.85,
        "texture_max": 20.0,
        "n_offset": 10,
        "offset_mode": "sweep",
        "offset": 0.0,
        "htex": _half_gates(2.0, dr),
    }
    mask = _mask_numpy(phi, rho, z, params)
    sign = _decide_sign([mask[3]], 0)
    feats, psi, valid, good, _ = _ml_features(phi, rho, z, dr, params, sign, mask)
    return feats, np.nan_to_num(psi), valid


def windows(feats, psi, valid, rng, per_ray=2):
    out = []
    ng = feats.shape[-1]
    for i in range(feats.shape[0]):
        for _ in range(per_ray):
            s = rng.integers(0, max(ng - N_GATES, 0) + 1)
            sl = slice(s, s + N_GATES)
            if valid[i, sl].mean() >= 0.2 and valid[i, sl].shape[0] == N_GATES:
                out.append((feats[i, :, sl], psi[i, sl], valid[i, sl]))
    return out


def main(dest, *paths):
    rng = np.random.default_rng(0)
    items = []
    for path in paths:
        new = []
        for phi, rho, z, dr in nexrad_sweeps(path):
            new += windows(*featurise(phi, rho, z, dr), rng, per_ray=1)
        keep = rng.permutation(len(new))[:PER_VOLUME]
        items += [new[k] for k in keep]
        print(path, len(new), len(items), flush=True)
    feats, psi, valid = (np.stack(x) for x in zip(*items))
    np.savez(
        dest,
        features=feats.astype(np.float32),
        psi=psi.astype(np.float32),
        valid=valid.astype(np.float32),
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
