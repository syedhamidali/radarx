"""
Fixed polar grid: every sweep is put on the same ``(azimuth, range)`` axes.

NEXRAD sweeps differ in the number of rays (720 super-resolution rays at
0.5 degrees, 360 at 1 degree), the ray positions and the number of gates.
Training samples need one shape, so inputs and labels are taken from the
nearest native ray and gate of each fixed bin. Nearest-neighbour selection
(not averaging) keeps class labels valid and inputs and labels consistent:
both come from the same native gate. Bins without a ray or gate within the
tolerance get the fill value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class PolarGrid:
    """Fixed polar axes: ``n_azimuth`` bins over 360 degrees, uniform gates."""

    n_azimuth: int = 360
    first_gate: float = 2125.0
    gate_spacing: float = 250.0
    n_gates: int = 920

    @classmethod
    def from_config(cls, cfg):
        return cls(**{k: cfg[k] for k in cls.__dataclass_fields__ if k in cfg})

    @property
    def azimuth(self):
        """Bin centres in degrees."""
        return (np.arange(self.n_azimuth) + 0.5) * (360.0 / self.n_azimuth)

    @property
    def range(self):
        """Gate centres in metres."""
        return self.first_gate + self.gate_spacing * np.arange(self.n_gates)

    def coords(self):
        return {
            "azimuth": (
                "azimuth",
                self.azimuth.astype("float32"),
                {"units": "degrees", "long_name": "azimuth bin centre"},
            ),
            "range": (
                "range",
                self.range.astype("float32"),
                {"units": "m", "long_name": "range to the gate centre"},
            ),
        }


def ray_index(azimuth, n_azimuth):
    """
    Index of the nearest native ray of every fixed azimuth bin.

    Parameters
    ----------
    azimuth : array-like
        Native ray azimuths (degrees, any order).
    n_azimuth : int
        Number of fixed bins over 360 degrees.

    Returns
    -------
    numpy.ndarray
        Ray index per bin, ``-1`` where no ray lies within half a bin plus
        half the native ray spacing.
    """
    az = np.mod(np.asarray(azimuth, dtype=np.float64), 360.0)
    centres = (np.arange(n_azimuth) + 0.5) * (360.0 / n_azimuth)
    if az.size == 0:
        return np.full(n_azimuth, -1, dtype=np.int64)
    order = np.argsort(az)
    s = az[order]
    if s.size > 1:
        steps = np.diff(np.concatenate([s, s[:1] + 360.0]))
        native = float(np.median(steps))
    else:
        native = 360.0 / n_azimuth
    tol = 0.5 * (360.0 / n_azimuth) + 0.5 * native
    # candidates: the rays just below and above each centre (with wrap)
    ext = np.concatenate([s[-1:] - 360.0, s, s[:1] + 360.0])
    pos = np.searchsorted(ext, centres)
    lo, hi = pos - 1, np.minimum(pos, ext.size - 1)
    d_lo, d_hi = centres - ext[lo], ext[hi] - centres
    pick = np.where(d_hi < d_lo, hi, lo)
    dist = np.minimum(d_lo, d_hi)
    idx = order[(pick - 1) % s.size]
    return np.where(dist <= tol, idx, -1)


def gate_index(rng, grid):
    """
    Index of the native gate at each fixed gate, ``-1`` where there is none.

    A native gate is used when its centre is within half a fixed gate
    spacing of the fixed gate centre.
    """
    r = np.asarray(rng, dtype=np.float64)
    target = grid.range
    if r.size == 0:
        return np.full(target.size, -1, dtype=np.int64)
    pos = np.clip(np.searchsorted(r, target), 1, max(r.size - 1, 1))
    lo = pos - 1
    hi = np.minimum(pos, r.size - 1)
    pick = np.where(np.abs(r[hi] - target) < np.abs(r[lo] - target), hi, lo)
    ok = np.abs(r[pick] - target) <= 0.5 * grid.gate_spacing + 1e-6
    return np.where(ok, pick, -1)


def take(values, rays, gates, fill):
    """Values at ``(rays[i], gates[j])``, ``fill`` where either is ``-1``."""
    values = np.asarray(values)
    out = values[np.clip(rays, 0, None)][:, np.clip(gates, 0, None)]
    bad = (rays < 0)[:, None] | (gates < 0)[None, :]
    if bad.any():
        out = np.where(bad, np.asarray(fill, dtype=out.dtype), out)
    return out


def fill_value(dtype):
    """NaN for floats, False for booleans and 0 for integers (no class)."""
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return np.nan
    if dtype.kind == "b":
        return False
    return 0


def resample_sweep(ds, names, grid, dtypes=None):
    """
    Put variables of a sweep on the fixed polar grid.

    Parameters
    ----------
    ds : xarray.Dataset
        Sweep with ``azimuth`` and ``range`` (metres) coordinates.
    names : sequence of str
        Variables on ``(azimuth, range)`` (in either order) to resample.
    grid : PolarGrid
        Target axes.
    dtypes : dict, optional
        Output dtype per variable; default float32 for floats, else unchanged.

    Returns
    -------
    xarray.Dataset
        The variables on ``(azimuth, range)`` of ``grid``, attributes kept.
    """
    dtypes = dtypes or {}
    ray_dim = [d for d in ds[names[0]].dims if d != "range"][0]
    rays = ray_index(ds["azimuth"].values, grid.n_azimuth)
    gates = gate_index(ds["range"].values, grid)
    out = {}
    for name in names:
        da = ds[name].transpose(ray_dim, "range")
        dtype = np.dtype(dtypes.get(name, da.dtype))
        if name not in dtypes and dtype.kind == "f":
            dtype = np.dtype("float32")
        vals = np.asarray(da.values)
        if vals.dtype.kind == "f" and dtype.kind in "iub":
            vals = np.where(np.isfinite(vals), vals, 0)
        vals = vals.astype(dtype, copy=False)
        out[name] = (
            ("azimuth", "range"),
            take(vals, rays, gates, fill_value(dtype)),
            dict(da.attrs),
        )
    return xr.Dataset(out, coords=grid.coords())
