"""
Evaluate single-Doppler retrievals against multi-Doppler and synthetic truth.

Methods compared, all from the same single radar:

``background``
    The ERA5 (or, for synthetic samples, the perturbed true) background, w = 0.
``vad``
    Velocity-volume processing: at every height one horizontally uniform
    wind fitted by least squares to the fall-speed corrected radial
    velocities within 5-80 km of the radar (the gridded analogue of the
    velocity-azimuth display, Browning and Wexler 1968; Waldteufel and
    Corbin 1979); the background where too few data; w = 0.
``variational``
    ``radarx.retrieve.single_doppler_winds`` without a network: the radarx
    variational cost with one radar (observation, mass continuity,
    smoothness, background).
``network``
    The network alone (``refine=False``).
``network+var``
    The network as the background of the variational retrieval (default).

Scores: RMSE of u, v, w on the cells where the reference is defined
(multi-Doppler: two radars with a beam crossing angle above 30 degrees;
synthetic: cells observed by the radar), and the same as a function of the
distance from the radar, of the beam elevation and of the angle between the
beam and the wind.

::

    python evaluate.py --data DATA --model single_doppler.onnx --out EVAL
"""

import argparse
import json
import os
import time

import numpy as np
import synthetic
import xarray as xr
from data import RealFile, real_files

from radarx.retrieve.single_doppler import single_doppler_winds

METHODS = ("background", "vad", "variational", "network", "network+var")
RANGE_BINS = np.array([0, 25, 50, 75, 100, 125, 150, 200]) * 1e3
ELEV_BINS = np.array([0, 1, 2, 4, 8, 20, 90])
CROSS_BINS = np.array([0, 15, 30, 45, 60, 75, 90])


def to_xarray(s):
    """One-radar grid and background Datasets of a sample."""
    dims = ("z", "y", "x")
    coords = {"z": s["z"], "y": s["y"], "x": s["x"]}
    rx_, ry_, alt = s["radar"]
    ds = xr.Dataset(
        {
            "VRADH": (("radar",) + dims, s["vr"][None]),
            "DBZH": (("radar",) + dims, s["dbz"][None]),
            "radar_x": ("radar", [rx_]),
            "radar_y": ("radar", [ry_]),
            "radar_altitude": ("radar", [alt]),
        },
        coords=dict(coords, radar=[0]),
    )
    bg = xr.Dataset(
        {
            "u": (dims, s["u_bg"]),
            "v": (dims, s["v_bg"]),
            "air_density": (dims, s["rho"]),
            "freezing_level": (("y", "x"), s["freezing_level"]),
        },
        coords=coords,
    )
    return ds, bg


def vad(s, rmin=5e3, rmax=80e3, min_obs=50):
    """Horizontally uniform wind per level from the radial velocities."""
    obs = (
        np.isfinite(s["vr"])
        & (s["distance"] >= rmin)[None]
        & (s["distance"] <= rmax)[None]
    )
    target = np.nan_to_num(s["vr"]) + s["coef"][2] * s["fall_speed"]
    u = np.array(s["u_bg"], dtype=float)
    v = np.array(s["v_bg"], dtype=float)
    for k in range(u.shape[0]):
        m = obs[k]
        if m.sum() < min_obs:
            continue
        a = np.stack([s["coef"][0][k][m], s["coef"][1][k][m]], axis=1)
        sol, *_ = np.linalg.lstsq(a, target[k][m], rcond=None)
        if np.linalg.cond(a) < 50:
            u[k], v[k] = sol
    return np.stack([u, v, np.zeros_like(u)])


def retrieve(s, model, methods=METHODS):
    out = {}
    if "background" in methods:
        out["background"] = np.stack([s["u_bg"], s["v_bg"], np.zeros_like(s["u_bg"])])
    if "vad" in methods:
        out["vad"] = vad(s)
    ds, bg = to_xarray(s)
    timing = {}
    if "variational" in methods:
        t = time.perf_counter()
        r = single_doppler_winds(ds, bg)
        timing["variational"] = time.perf_counter() - t
        out["variational"] = np.stack([r[c].values for c in "uvw"])
    if model is not None and "network+var" in methods:
        t = time.perf_counter()
        r = single_doppler_winds(ds, bg, model=model)
        timing["network+var"] = time.perf_counter() - t
        out["network+var"] = np.stack([r[c].values for c in "uvw"])
        out["network"] = np.stack([r[f"{c}_network"].values for c in "uvw"])
    return out, timing


def accumulate(acc, key, pred, truth, mask):
    se = ((pred - truth) ** 2)[:, mask].sum(1)
    a = acc.setdefault(key, [np.zeros(3), 0])
    a[0] += se
    a[1] += int(mask.sum())


def binned(acc, name, method, pred, truth, mask, values, bins):
    idx = np.digitize(values, bins) - 1
    for b in range(len(bins) - 1):
        accumulate(acc, (name, method, b), pred, truth, mask & (idx == b))


def geometry(s):
    el = np.degrees(np.arcsin(np.clip(s["coef"][2], -1, 1)))
    beam_az = np.arctan2(s["coef"][0], s["coef"][1])
    wind_dir = np.arctan2(s["truth"][0], s["truth"][1])
    cross = np.degrees(np.abs(np.arcsin(np.sin(wind_dir - beam_az))))
    dist = np.broadcast_to(s["distance"][None], el.shape)
    return dist, el, cross


def evaluate(samples, model, mask_fn):
    acc, timings = {}, []
    for s in samples:
        preds, timing = retrieve(s, model)
        timings.append(timing)
        mask = mask_fn(s)
        dist, el, cross = geometry(s)
        for m, p in preds.items():
            accumulate(acc, ("all", m), p, s["truth"], mask)
            binned(acc, "range", m, p, s["truth"], mask, dist, RANGE_BINS)
            binned(acc, "elevation", m, p, s["truth"], mask, el, ELEV_BINS)
            binned(acc, "cross", m, p, s["truth"], mask, cross, CROSS_BINS)
        print(
            s.get("name", "synthetic"),
            {
                m: np.round(
                    np.sqrt(((p - s["truth"]) ** 2)[:, mask].mean(1)), 2
                ).tolist()
                for m, p in preds.items()
            },
            flush=True,
        )
    res = {}
    for key, (se, n) in acc.items():
        res["|".join(map(str, key))] = dict(rmse=np.sqrt(se / max(n, 1)).tolist(), n=n)
    t = (
        {k: float(np.mean([x[k] for x in timings if k in x])) for k in timings[0]}
        if timings
        else {}
    )
    return res, t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--n-synthetic", type=int, default=24)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = [RealFile(f) for f in real_files(a.data, a.split)]
    real = [f.sample(k) for f in files for k in range(len(f.radars))]
    res_real, t_real = evaluate(real, a.model, lambda s: s["weight"] > 0)
    rng = np.random.default_rng(999)
    syn = [synthetic.sample(rng, ny=96, nx=96) for _ in range(a.n_synthetic)]
    res_syn, t_syn = evaluate(syn, a.model, lambda s: np.isfinite(s["vr"]))
    out = dict(
        real=res_real,
        synthetic=res_syn,
        timing_real=t_real,
        timing_synthetic=t_syn,
        n_real=len(real),
        n_synthetic=len(syn),
    )
    json.dump(out, open(os.path.join(a.out, f"scores_{a.split}.json"), "w"), indent=1)
    for name, res in (("multi-Doppler (real)", res_real), ("synthetic truth", res_syn)):
        print(f"\nRMSE (u, v, w) vs {name}:")
        for m in METHODS:
            r = res.get(f"all|{m}")
            if r:
                print(f"  {m:12s} {np.round(r['rmse'], 2).tolist()}  n={r['n']}")


if __name__ == "__main__":
    main()
