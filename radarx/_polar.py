#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Helpers shared by the modules that read polar volumes (not public API)."""

import numpy as np

DEFAULT_BEAMWIDTH = 1.0  # degrees, used when a volume does not give one


def nearest_ray(azimuth, target, distance=False):
    """
    Index of the ray nearest to every target azimuth (circular).

    Parameters
    ----------
    azimuth : array-like
        Azimuths of the rays [degrees], in any order.
    target : array-like
        Azimuths [degrees] to match.
    distance : bool, optional
        Also return the angular distance [degrees] to the matched ray.

    Returns
    -------
    numpy.ndarray or tuple
        Index into ``azimuth`` (and the distance). A target halfway between
        two rays takes the one at the larger azimuth.
    """
    azimuth = np.mod(np.asarray(azimuth, dtype=np.float64), 360.0)
    order = np.argsort(azimuth, kind="stable")
    az = azimuth[order]
    ext = np.concatenate([az[-1:] - 360.0, az, az[:1] + 360.0])
    target = np.mod(np.asarray(target, dtype=np.float64), 360.0)
    idx = np.clip(np.searchsorted(ext, target), 1, len(ext) - 1)
    left = target - ext[idx - 1] < ext[idx] - target
    pick = np.where(left, idx - 1, idx)
    ray = order[np.mod(pick - 1, len(az))]
    if distance:
        return ray, np.abs(ext[pick] - target)
    return ray


def beam_width(dtree, default=DEFAULT_BEAMWIDTH):
    """
    Vertical beam width [degrees] stored in a volume, else ``default``.

    Looks for ``radar_beam_width_v`` (then ``radar_beam_width_h``) in the root
    and in the ``radar_parameters`` group of the DataTree.
    """
    nodes = [dtree.root.to_dataset()]
    if "radar_parameters" in dtree.children:
        nodes.append(dtree["radar_parameters"].to_dataset())
    for ds in nodes:
        for key in ("radar_beam_width_v", "radar_beam_width_h"):
            if key in ds and np.isfinite(float(ds[key])):
                return float(ds[key])
    return float(default)
