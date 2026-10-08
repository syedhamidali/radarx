#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Biological Echo Segmentation (MistNet)
======================================

Separate biological scatterers (birds, bats, insects) from precipitation with
MistNet (Lin et al. 2019), a fully convolutional network (VGG-16 backbone
with FCN-8s heads) trained on 239 000 WSR-88D volumes labelled with dual-pol
products. It sees reflectivity, radial velocity and spectrum width of the
0.5°, 1.5°, 2.5°, 3.5° and 4.5° sweeps of a volume, each rendered on a
608 x 608 Cartesian grid of 500 m cells centred on the radar (±152 km), and
returns, per sweep and cell, the probabilities of biology and of weather.
As in Lin et al. (2019), a gate is weather when the weather probability of
its sweep, or the mean weather probability of the five sweeps, exceeds 0.45;
all other gates with reflectivity are biological.

This module renders the sweeps exactly as the reference implementation in
vol2bird (Dokter et al. 2011) feeds the network: for each cell centre the
nearest gate on a 4/3 effective earth, NaN where the gate has no data. The
probabilities are mapped back to the polar gates of the five sweeps by
their nearest cell.

The upstream TorchScript weights (MIT licence, GitHub ``adokter/MistNet``)
are converted to ONNX on first use (needs the ``onnx`` package once) and
cached; inference needs ``onnxruntime`` (``pip install radarx[ml]``). The
network was trained on S-band data; use with other wavelengths is untested.

Compare with :func:`radarx.retrieve.echo_mask`, the fuzzy-logic
meteorological / non-meteorological classification from the polarimetric
variables.

References
----------
Lin, T.-Y., K. Winner, G. Bernstein, A. Mittal, A. M. Dokter, K. G. Horton,
C. Nilsson, B. M. Van Doren, A. Farnsworth, F. A. La Sorte, S. Maji, and
D. Sheldon, 2019: MistNet: Measuring historical bird migration in the US
using archived weather radar data and convolutional neural networks.
*Methods Ecol. Evol.*, **10** (11), 1908-1922,
https://doi.org/10.1111/2041-210X.13280

Dokter, A. M., F. Liechti, H. Stark, L. Delobbe, P. Tabary, and I. Holleman,
2011: Bird migration flight altitudes studied by a network of operational
weather radars. *J. R. Soc. Interface*, **8** (54), 30-43,
https://doi.org/10.1098/rsif.2010.0116

.. autosummary::
   :nosignatures:
   :toctree: generated/

   biological_echo
"""

from __future__ import annotations

__all__ = ["biological_echo"]

import numpy as np
import xarray as xr

from .._registry import accessor_method
from . import _onnx_models
from ._products import product_tree
from .tornado import _clean, _first, _fixed_angle, _sweeps

#: Name of the default model, converted on first use.
DEFAULT_MODEL = "mistnet-nexrad"

#: Elevations (degrees) of the five MistNet input sweeps.
ELEVATIONS = (0.5, 1.5, 2.5, 3.5, 4.5)

_EARTH_RADIUS = 6371200.0 * 4.0 / 3.0  # 4/3 effective earth radius, m
_BLEED = 8  # cells at the grid edge that are not mapped back


def _slant_range(distance, elev):
    """Slant range (m) of a ground distance (m) at elevation ``elev`` (rad)."""
    gamma = distance / _EARTH_RADIUS
    beta = np.pi / 2 - elev - gamma
    return _EARTH_RADIUS * np.sin(gamma) / np.sin(beta)


def _ground_distance(rng, elev):
    """Ground distance (m) of a slant range (m) at elevation ``elev`` (rad)."""
    height = (
        np.sqrt(rng**2 + _EARTH_RADIUS**2 + 2 * _EARTH_RADIUS * rng * np.sin(elev))
        - _EARTH_RADIUS
    )
    return _EARTH_RADIUS * np.arcsin(rng * np.cos(elev) / (_EARTH_RADIUS + height))


def _sweep_lookup(ds, field):
    """Values on (ray, range) sorted by azimuth, with azimuths and ranges."""
    da = ds[field]
    ray_dim = [d for d in da.dims if d != "range"][0]
    da = _clean(da)[0].transpose(ray_dim, "range")
    az = np.mod(np.asarray(ds["azimuth"].values, dtype=np.float64), 360.0)
    order = np.argsort(az)
    return da.values[order], az[order], np.asarray(ds["range"].values, np.float64)


def _nearest_index(sorted_az, target):
    ext = np.concatenate([sorted_az[-1:] - 360.0, sorted_az, sorted_az[:1] + 360.0])
    idx = np.clip(np.searchsorted(ext, target), 1, len(ext) - 1)
    left = target - ext[idx - 1] < ext[idx] - target
    return np.mod(np.where(left, idx - 1, idx) - 1, sorted_az.size)


def _render(ds, fields, size, resolution):
    """Fields of a sweep on the MistNet grid (rows north, columns east)."""
    elev = np.radians(_fixed_angle(ds))
    offsets = resolution * (np.arange(size) - size // 2).astype(np.float64)
    north, east = np.meshgrid(offsets, offsets, indexing="ij")
    azimuth = np.mod(np.degrees(np.arctan2(east, north)), 360.0)
    rng = _slant_range(np.hypot(north, east), elev)
    out = np.full((len(fields), size, size), np.nan, dtype=np.float32)
    for k, field in enumerate(fields):
        if field is None or field not in ds:
            continue
        values, az, ranges = _sweep_lookup(ds, field)
        ray = _nearest_index(az, azimuth)
        step = ranges[1] - ranges[0]
        gate = np.rint((rng - ranges[0]) / step).astype(np.int64)
        inside = (gate >= 0) & (gate < ranges.size)
        out[k][inside] = values[ray[inside], gate[inside]]
    return out


def _to_polar(ds, grid, size, resolution):
    """Nearest-cell values of a MistNet grid at the gates of a sweep."""
    da = ds[_first(ds, ("DBZH", "DBZ", "reflectivity"))]
    ray_dim = [d for d in da.dims if d != "range"][0]
    elev = np.radians(_fixed_angle(ds))
    az = np.radians(np.asarray(ds["azimuth"].values, dtype=np.float64))[:, None]
    dist = _ground_distance(np.asarray(ds["range"].values, np.float64), elev)[None, :]
    north, east = dist * np.cos(az), dist * np.sin(az)
    limit = resolution * (size - _BLEED) / 2
    valid = (np.abs(north) <= limit) & (np.abs(east) <= limit)
    i = np.clip(np.rint(north / resolution + size // 2).astype(np.int64), 0, size - 1)
    j = np.clip(np.rint(east / resolution + size // 2).astype(np.int64), 0, size - 1)
    out = np.where(valid, grid[..., i, j], np.nan).astype(np.float32)
    return out, (ray_dim, "range")


def _select(sweeps, elevations, fields):
    chosen = []
    for target in elevations:
        best, best_diff = None, np.inf
        for name, ds in sweeps:
            if not all(f is not None and f in ds for f in fields):
                continue
            diff = abs(_fixed_angle(ds) - target)
            if diff < best_diff - 1e-6:
                best, best_diff = (name, ds), diff
        if best is None:
            raise ValueError(f"no sweep with {fields} for {target}°")
        chosen.append(best)
    return chosen


def biological_echo(
    volume,
    model=None,
    *,
    fields=("DBZH", "VRADH", "WRADH"),
    elevations=ELEVATIONS,
    weather_threshold=0.45,
    size=608,
    resolution=500.0,
    providers=None,
):
    """
    Biology and weather probabilities from MistNet (ONNX).

    Parameters
    ----------
    volume : xarray.DataTree
        A WSR-88D volume with ``sweep_*`` groups (xradar). For each of
        ``elevations`` the sweep nearest in fixed angle that holds all
        ``fields`` is used (the Doppler cut of NEXRAD split cuts).
    model : str or path or radarx.ml.Model, optional
        Default ``"mistnet-nexrad"``, the published MistNet weights converted
        to ONNX on first use; or an ONNX file with the same input and output,
        a registry name or a loaded model.
    fields : tuple of str, optional
        Reflectivity (dBZ), radial velocity and spectrum width (m s⁻¹), as
        measured (velocities need not be dealiased). Default
        ``("DBZH", "VRADH", "WRADH")``.
    elevations : tuple of float, optional
        The five input elevations (degrees). Default 0.5° to 4.5° in 1° steps.
    weather_threshold : float, optional
        Weather probability above which a gate is weather, on its own sweep
        or on the mean of the five sweeps. Default 0.45 (Lin et al. 2019).
    size, resolution : int, float, optional
        Cells per side and cell size (m) of the Cartesian input grid.
        Default 608 and 500 m, the grid MistNet was trained on.
    providers : list of str, optional
        ONNX Runtime execution providers. Default: CPU.

    Returns
    -------
    xarray.DataTree
        The root of the input and, for each of the five sweeps used,
        ``biology_probability`` and ``weather_probability`` (0-1) and
        ``biological_echo`` (bool: reflectivity present and not weather) on
        the sweep's dimensions and coordinates (NaN / False beyond ±150 km).
        Merge them into the volume with ``.radarx.assign(products)``.
        Attributes ``ml_model``, ``ml_model_version``, ``ml_model_licence``
        and ``ml_model_citation``.

    References
    ----------
    Lin, T.-Y., K. Winner, G. Bernstein, A. Mittal, A. M. Dokter,
    K. G. Horton, C. Nilsson, B. M. Van Doren, A. Farnsworth, F. A. La Sorte,
    S. Maji, and D. Sheldon, 2019: MistNet: Measuring historical bird
    migration in the US using archived weather radar data and convolutional
    neural networks. *Methods Ecol. Evol.*, **10** (11), 1908-1922,
    https://doi.org/10.1111/2041-210X.13280

    Examples
    --------
    >>> bio = radarx.retrieve.biological_echo(dtree)  # doctest: +SKIP
    >>> dtree = dtree.radarx.assign(bio)  # doctest: +SKIP
    """
    if not hasattr(volume, "children"):
        raise TypeError("biological_echo needs an xarray.DataTree volume")
    if len(elevations) != 5:
        raise ValueError("MistNet needs exactly five elevations")
    chosen = _select(_sweeps(volume), elevations, fields)
    x = np.concatenate(
        [_render(ds, fields, size, resolution)[None] for _, ds in chosen]
    )  # [scan, product, H, W]
    x = x.transpose(1, 0, 2, 3).reshape(1, 3 * len(chosen), size, size)
    net = _onnx_models.load(model, DEFAULT_MODEL, providers)
    y = np.asarray(net.run({"x": np.ascontiguousarray(x, dtype=np.float32)})["y"])[0]
    weather_mean = y[2].mean(axis=0)
    attrs = _onnx_models.model_attrs(net, DEFAULT_MODEL)
    nodes = {}
    for s, (name, ds) in enumerate(chosen):
        if name in nodes:  # the same sweep nearest to two elevations
            continue  # pragma: no cover
        grid = np.stack([y[1, s], y[2, s], weather_mean])
        polar, dims = _to_polar(ds, grid, size, resolution)
        bio, wx, wx_mean = polar
        dbz = _clean(ds[_first(ds, ("DBZH", "DBZ", "reflectivity"))])[0]
        dbz = dbz.transpose(*dims).values
        weather = (wx > weather_threshold) | (wx_mean > weather_threshold)
        coords = ds[fields[0]].transpose(*dims).coords
        common = {"units": "1", **attrs}
        out = xr.Dataset(
            {
                "biology_probability": (
                    dims,
                    bio,
                    {"long_name": "probability of biological echo (MistNet)", **common},
                ),
                "weather_probability": (
                    dims,
                    wx,
                    {"long_name": "probability of weather echo (MistNet)", **common},
                ),
                "biological_echo": (
                    dims,
                    np.isfinite(dbz) & np.isfinite(wx) & ~weather,
                    {
                        "long_name": "biological echo (MistNet)",
                        "comment": (
                            f"reflectivity present and weather probability of the "
                            f"sweep and of the 5-sweep mean <= {weather_threshold}"
                        ),
                        **attrs,
                    },
                ),
            },
            coords=coords,
        )
        nodes[name] = out.transpose(*ds[fields[0]].dims)
    return product_tree(volume, nodes)


@accessor_method("datatree", name="biological_echo")
def _biological_echo_accessor(self, model=None, **kwargs):
    return biological_echo(self.xarray_obj, model, **kwargs)


_biological_echo_accessor.__doc__ = biological_echo.__doc__
