#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Evaluate the exported rain rate model on simulated samples that it did not see.

    python evaluate.py runs/sim-v1/radarx-qpe-v1.onnx --out runs/sim-v1

The model is run through ``radarx.ml`` and ``radarx.retrieve.rain_rate_ml`` is
not needed: the features are passed to the ONNX model directly. It is compared
with the power laws of ``radarx.retrieve`` (R(Z), R(KDP), R(Z, ZDR) of the
band) and with a simple hybrid, R(KDP) where K_DP exceeds a threshold that
scales with the wavelength (0.3 degree/km at S band) and R(Z, ZDR) elsewhere,
on

* ``iid``: samples from the training mixture of regimes with measurement
  errors (new seed);
* ``clean``: the same without measurement errors, which isolates the scatter
  of the drop size distribution;
* ``stratiform``, ``maritime``, ``continental``, ``tropical``: one regime each
  (the model has seen them only in the mixture).

Scores, per band and sample set: the bias of the total rain (sum of the
estimates over the sum of the truth, in %), the normalized mean absolute error
(mean |estimate - truth| over the mean truth), the scatter (half the distance
between the 16th and 84th percentile of 10 log10(estimate/truth), dB, after
Germann et al. 2006) and, for the model, the fraction of the truth inside its
one-sigma bounds (68 % for a calibrated Gaussian in the logarithm). The
rain rates below 1 mm/h are left out of the scatter (as the gauge threshold
of Wolfensberger et al. 2021 is 0.1 mm per 10 minutes).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr
from simulate import BANDS, WAVELENGTH, simulate

from radarx import ml
from radarx.retrieve import rain_rate_kdp, rain_rate_z, rain_rate_z_zdr

SETS = {
    "iid": dict(regimes=None, noise=1.0),
    "clean": dict(regimes=None, noise=0.0),
    "stratiform": dict(regimes=["stratiform"], noise=1.0),
    "maritime": dict(regimes=["maritime"], noise=1.0),
    "continental": dict(regimes=["continental"], noise=1.0),
    "tropical": dict(regimes=["tropical"], noise=1.0),
}


def baselines(band, f):
    """The power-law estimates of the band for the feature matrix ``f``."""
    ds = xr.Dataset(
        {
            "DBZH": ("g", f[:, 0].astype(np.float64)),
            "ZDR": ("g", f[:, 1].astype(np.float64)),
            "KDP": ("g", f[:, 2].astype(np.float64)),
        }
    )
    out = {}
    if band != "C":
        out["R(Z)"] = rain_rate_z(ds, band=band, z_cap=None).values
    out["R(Z,ZDR)"] = rain_rate_z_zdr(ds, band=band, z_cap=None).values
    out["R(KDP)"] = rain_rate_kdp(ds, band=band, negative="zero").values
    thr = 0.3 * WAVELENGTH["S"] / WAVELENGTH[band]
    out["hybrid"] = np.where(ds.KDP.values >= thr, out["R(KDP)"], out["R(Z,ZDR)"])
    return out


def scores(est, truth, lower=None, upper=None):
    est = np.asarray(est, dtype=np.float64)
    ok = np.isfinite(est) & (est > 0)
    big = ok & (truth >= 1.0)
    db = 10.0 * np.log10(est[big] / truth[big])
    out = {
        "bias_percent": float(100.0 * (est[ok].sum() / truth[ok].sum() - 1.0)),
        "nmae": float(np.abs(est[ok] - truth[ok]).sum() / truth[ok].sum()),
        "scatter_db": float(0.5 * (np.percentile(db, 84) - np.percentile(db, 16))),
        "n": int(ok.sum()),
    }
    if lower is not None:
        out["inside_one_sigma"] = float(np.mean((truth >= lower) & (truth <= upper)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", help="the ONNX file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=300_000, help="per band and set")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--figure", action="store_true")
    args = ap.parse_args()
    ml.register_model(
        "evaluated", args.model, None, "MIT", "radarx evaluation", overwrite=True
    )
    net = ml.load_model("evaluated", providers="cpu")
    result = {}
    keep = {}
    for name, cfg in SETS.items():
        for band in BANDS:
            d = simulate(
                args.samples,
                seed=[args.seed, list(SETS).index(name), BANDS.index(band)],
                bands=(band,),
                **cfg,
            )
            truth = d["rain_rate"].astype(np.float64)
            f = d["features"]
            out = net.run({"features": f}, nan="ignore")
            table = {
                "network": scores(
                    out["rain_rate"],
                    truth,
                    out["rain_rate_lower"],
                    out["rain_rate_upper"],
                )
            }
            for k, v in baselines(band, f).items():
                table[k] = scores(v, truth)
            result[f"{name}/{band}"] = table
            if name == "iid":
                keep[band] = (truth, out["rain_rate"], baselines(band, f))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "evaluation.json").write_text(json.dumps(result, indent=1))
    for name in SETS:
        print(f"\n{name}")
        for band in BANDS:
            for method, s in result[f"{name}/{band}"].items():
                extra = (
                    f"  inside 1 sigma {s['inside_one_sigma']:.2f}"
                    if "inside_one_sigma" in s
                    else ""
                )
                print(
                    f"  {band} {method:10s} bias {s['bias_percent']:7.1f} %  "
                    f"NMAE {s['nmae']:.3f}  scatter {s['scatter_db']:.2f} dB{extra}"
                )
    if args.figure:
        figure(keep, Path(args.out) / "evaluation.png")


def figure(keep, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for j, band in enumerate(BANDS):
        truth, net, base = keep[band]
        sel = np.random.default_rng(0).choice(len(truth), 4000, replace=False)
        ax = axes[0, j]
        best = "hybrid"
        ax.loglog(truth[sel], base[best][sel], ".", ms=2, alpha=0.4, label=best)
        ax.loglog(truth[sel], net[sel], ".", ms=2, alpha=0.4, label="network")
        ax.plot([0.1, 400], [0.1, 400], "k", lw=0.8)
        ax.set(
            xlim=(0.5, 400),
            ylim=(0.5, 400),
            title=f"{band} band",
            xlabel="true rain rate (mm/h)",
        )
        if j == 0:
            ax.set_ylabel("estimate (mm/h)")
            ax.legend(markerscale=5)
        ax = axes[1, j]
        for label, est in [(best, base[best]), ("network", net)]:
            ratio = 10 * np.log10(np.maximum(est, 1e-3) / truth)
            ax.hist(
                ratio[truth >= 1],
                bins=np.linspace(-8, 8, 80),
                histtype="step",
                label=label,
                density=True,
            )
        ax.set(xlabel="10 log10(estimate / truth) (dB)")
        if j == 0:
            ax.set_ylabel("density")
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
