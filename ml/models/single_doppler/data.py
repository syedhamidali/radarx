"""
Samples for training and evaluation.

A sample is a dict of NumPy arrays on ``(z, y, x)`` (see
``synthetic.sample``): ``vr``, ``dbz``, ``coef``, ``u_bg``, ``v_bg``, ``rho``,
``fall_speed``, ``truth``, ``weight``, ``distance``. Real samples come from
the files written by ``make_dataset.py``: one per radar and analysis time,
with the multi-Doppler wind as the target where both radars observe with a
beam crossing angle above 30 degrees.
"""

import glob
import os

import numpy as np
import xarray as xr
from physics import WEIGHTS

from radarx.retrieve.multidoppler import _beam_angles
from radarx.retrieve.single_doppler import _features


class RealFile:
    """A dataset file kept in memory (float32), giving one sample per radar."""

    def __init__(self, path):
        ds = xr.load_dataset(path)
        dims = ("z", "y", "x")
        f32 = lambda da: da.transpose(..., *dims).values.astype(
            np.float32
        )  # noqa: E731
        self.path = path
        self.x, self.y, self.z = ds.x.values, ds.y.values, ds.z.values
        self.vr = f32(ds.VRADH.transpose("radar", *dims))
        self.dbz = f32(ds.DBZH.transpose("radar", *dims))
        self.u_bg, self.v_bg = f32(ds.u_bg), f32(ds.v_bg)
        self.rho = f32(ds.air_density)
        self.freezing = ds.freezing_level.transpose("y", "x").values.astype(np.float64)
        self.fall_speed = np.nan_to_num(f32(ds.fall_speed))
        self.truth = np.stack([f32(ds[c]) for c in "uvw"])
        self.good = f32((ds.n_radars >= 2) & (ds.beam_crossing_angle > 30))
        self.n_radars = ds.n_radars.transpose(*dims).values
        self.crossing = f32(ds.beam_crossing_angle)
        self.radars = [
            (float(ds.radar_x[k]), float(ds.radar_y[k]), float(ds.radar_altitude[k]))
            for k in range(ds.sizes["radar"])
        ]
        self.names = [str(n) for n in ds.radar_name.values]
        self.attrs = dict(ds.attrs)

    @property
    def shape(self):
        return self.rho.shape

    def sample(self, k, j0=0, i0=0, ny=None, nx=None):
        """Sample of radar ``k`` on the region starting at (j0, i0)."""
        ny = ny or self.shape[1] - j0
        nx = nx or self.shape[2] - i0
        sl = (slice(None), slice(j0, j0 + ny), slice(i0, i0 + nx))
        x, y, z = self.x[i0 : i0 + nx], self.y[j0 : j0 + ny], self.z
        Z_, Y_, X_ = np.meshgrid(z, y, x, indexing="ij")
        rx_, ry_, alt = self.radars[k]
        az, el = _beam_angles(X_ - rx_, Y_ - ry_, Z_, np.full(Z_.shape, alt))
        a_, e_ = np.radians(az), np.radians(el)
        f64 = lambda a: np.asarray(a, dtype=np.float64)  # noqa: E731
        return dict(
            vr=f64(self.vr[k][sl]),
            dbz=f64(self.dbz[k][sl]),
            coef=np.stack(
                [np.cos(e_) * np.sin(a_), np.cos(e_) * np.cos(a_), np.sin(e_)]
            ),
            u_bg=f64(self.u_bg[sl]),
            v_bg=f64(self.v_bg[sl]),
            rho=f64(self.rho[sl]),
            fall_speed=f64(self.fall_speed[sl]),
            truth=f64(self.truth[(slice(None),) + sl]),
            weight=f64(self.good[sl]),
            distance=np.hypot(X_[0] - rx_, Y_[0] - ry_),
            x=x,
            y=y,
            z=z,
            radar=(rx_, ry_, alt),
            freezing_level=self.freezing[j0 : j0 + ny, i0 : i0 + nx],
            name=f"{os.path.basename(self.path)[:-3]}:{self.names[k]}",
        )

    def random_sample(self, rng, size=96, focus=0.8):
        """Random radar and crop, centred on a supervised column with probability ``focus``."""
        k = int(rng.integers(len(self.radars)))
        ny, nx = self.shape[1:]
        cols = self._good_columns()
        if rng.uniform() < focus and len(cols):
            jc, ic = cols[rng.integers(len(cols))]
        else:
            jc, ic = rng.integers(ny), rng.integers(nx)
        j0 = int(np.clip(jc - size // 2, 0, ny - size))
        i0 = int(np.clip(ic - size // 2, 0, nx - size))
        return self.sample(k, j0, i0, size, size)

    def _good_columns(self):
        if not hasattr(self, "_cols"):
            self._cols = np.argwhere(self.good.any(0))
        return self._cols


def real_files(root, split):
    return sorted(glob.glob(os.path.join(root, split, "*.nc")))


_SPATIAL = ("vr", "dbz", "u_bg", "v_bg", "rho", "fall_speed", "weight")


def augment(rng, s):
    """Random symmetry of the square: flips of x and y and the x-y swap."""
    s = dict(s)
    if rng.uniform() < 0.5:  # x -> -x
        for k in _SPATIAL + ("coef", "truth"):
            s[k] = s[k][..., ::-1]
        s["distance"] = s["distance"][:, ::-1]
        s["coef"] = s["coef"] * np.array([-1, 1, 1])[:, None, None, None]
        s["truth"] = s["truth"] * np.array([-1, 1, 1])[:, None, None, None]
        s["u_bg"] = -s["u_bg"]
    if rng.uniform() < 0.5:  # y -> -y
        for k in _SPATIAL + ("coef", "truth"):
            s[k] = s[k][..., ::-1, :]
        s["distance"] = s["distance"][::-1, :]
        s["coef"] = s["coef"] * np.array([1, -1, 1])[:, None, None, None]
        s["truth"] = s["truth"] * np.array([1, -1, 1])[:, None, None, None]
        s["v_bg"] = -s["v_bg"]
    if rng.uniform() < 0.5:  # swap x and y
        for k in _SPATIAL + ("coef", "truth"):
            s[k] = np.swapaxes(s[k], -1, -2)
        s["distance"] = s["distance"].T
        s["coef"] = s["coef"][[1, 0, 2]]
        s["truth"] = s["truth"][[1, 0, 2]]
        s["u_bg"], s["v_bg"] = s["v_bg"], s["u_bg"]
    return {
        k: np.ascontiguousarray(v) if isinstance(v, np.ndarray) else v
        for k, v in s.items()
    }


def tensors(s):
    """Network input and loss arrays (float32) of one sample."""
    feats = _features(
        s["vr"], s["dbz"], s["coef"], s["u_bg"], s["v_bg"], s["z"], s["distance"]
    )
    obs = np.isfinite(s["vr"])
    target = np.where(obs, np.nan_to_num(s["vr"]) + s["coef"][2] * s["fall_speed"], 0.0)
    coef = s["coef"] * obs
    bg = np.stack([s["u_bg"], s["v_bg"], np.zeros_like(s["u_bg"])])
    bgw = np.zeros_like(bg)
    bgw[:2] = WEIGHTS["background"]
    f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)  # noqa: E731
    return dict(
        features=f32(feats),
        coef=f32(coef),
        target=f32(target),
        obs=f32(obs * WEIGHTS["observation"]),
        rho=f32(s["rho"]),
        bg=f32(bg),
        bgw=f32(bgw),
        truth=f32(np.nan_to_num(s["truth"])),
        weight=f32(s["weight"] * np.isfinite(s["truth"]).all(0)),
    )
