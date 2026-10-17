#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Train the rain rate network.

    python train.py --out runs/sim-v1                 # simulated drop size distributions
    python train.py --out runs/gauges --table pairs.npz   # features and gauge rain rates

The simulated mode draws a new set of distributions and measurement errors
for every epoch (``simulate.py``), so the network never sees a sample twice.
The table mode trains on a ``.npz`` with the arrays ``features`` (N, F, in
the order of ``--features``, from the features of ``model.FEATURES``) and ``rain_rate`` (N, mm/h, the gauge rate
at the gate or column) and optionally ``weight`` (N); the first 10 % of the
rows (after a seeded shuffle) are held out for validation. Use it with the
features of a gauge-radar data base of your radar (reflectivity, differential
reflectivity, differential phase, correlation coefficient, height,
temperature) as in Wolfensberger et al. (2021). The simulation has the
features ``dbz zdr kdp rhohv wavelength``.

The weights go to ``<out>/model.pt``, the history to ``<out>/history.json``.
Export with ``export.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from model import QPENet, nll
from simulate import FEATURES, simulate


def epoch_data(args, epoch):
    d = simulate(args.samples, seed=[args.seed, epoch], noise=args.noise)
    x = torch.from_numpy(d["features"])
    y = torch.from_numpy(np.log(d["rain_rate"]))
    return x, y, None


def validation_data(args):
    d = simulate(args.val_samples, seed=[args.seed, 10**6], noise=args.noise)
    return (
        torch.from_numpy(d["features"]),
        torch.from_numpy(np.log(d["rain_rate"])),
    )


def table_data(args):
    z = np.load(args.table)
    x = z["features"].astype(np.float32)
    y = np.log(np.maximum(z["rain_rate"].astype(np.float32), 0.01))
    w = z["weight"].astype(np.float32) if "weight" in z else None
    order = np.random.default_rng(args.seed).permutation(len(x))
    n_val = max(1, len(x) // 10)
    val, tr = order[:n_val], order[n_val:]
    train = (
        torch.from_numpy(x[tr]),
        torch.from_numpy(y[tr]),
        None if w is None else torch.from_numpy(w[tr]),
    )
    return train, (torch.from_numpy(x[val]), torch.from_numpy(y[val]))


def evaluate(net, x, y, batch=262144):
    net.eval()
    total, n = 0.0, 0
    err = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            mu, ls = net(x[i : i + batch])
            total += float(nll(mu, ls, y[i : i + batch])) * len(mu)
            err.append((mu - y[i : i + batch]).numpy())
            n += len(mu)
    e = np.concatenate(err)
    net.train()
    return total / n, float(np.sqrt((e**2).mean())), float(e.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--table")
    ap.add_argument(
        "--features", nargs="+", default=list(FEATURES), help="columns of the table"
    )
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--samples", type=int, default=2_000_000, help="per epoch")
    ap.add_argument("--val-samples", type=int, default=400_000)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64, 64])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    net = QPENet(args.features, tuple(args.hidden))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    steps_per_epoch = math.ceil(args.samples / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1
    )
    if args.table:
        (x_all, y_all, w_all), (xv, yv) = table_data(args)
        if x_all.shape[1] != len(args.features):
            raise SystemExit(
                "the table has a different number of features than --features"
            )
    else:
        if list(args.features) != list(FEATURES):
            raise SystemExit(f"the simulation has the features {list(FEATURES)}")
        xv, yv = validation_data(args)
        net.set_scaling(xv)
    if args.table:
        net.set_scaling(x_all)
    (out / "features.json").write_text(json.dumps(list(args.features)))
    history, best = [], math.inf
    t0 = time.time()
    for epoch in range(args.epochs):
        if not args.table:
            x_all, y_all, w_all = epoch_data(args, epoch)
        perm = torch.randperm(len(x_all))
        loss_sum = 0.0
        for i in range(0, len(perm), args.batch):
            idx = perm[i : i + args.batch]
            mu, ls = net(x_all[idx])
            loss = nll(mu, ls, y_all[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            loss_sum += float(loss) * len(idx)
        val_nll, val_rmse, val_bias = evaluate(net, xv, yv)
        history.append(
            {
                "epoch": epoch,
                "train_nll": loss_sum / len(perm),
                "val_nll": val_nll,
                "val_rmse_ln": val_rmse,
                "val_bias_ln": val_bias,
                "seconds": time.time() - t0,
            }
        )
        print(json.dumps(history[-1]), flush=True)
        if val_nll < best:
            best = val_nll
            torch.save(net.state_dict(), out / "model.pt")
    (out / "history.json").write_text(
        json.dumps({"args": vars(args), "history": history}, indent=1)
    )
    print("best validation NLL", best)


if __name__ == "__main__":
    main()
