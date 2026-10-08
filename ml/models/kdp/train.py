#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Train the KDP network on simulated rays, optionally fine-tune on real rays.

    python train.py --out runs/full --steps 12000
    python train.py --out runs/nophys --steps 12000 --no-physics
    python train.py --out runs/ft --init runs/full/model.pt --real real_rays.npz \\
        --steps 2000 --lr 3e-4

Synthetic batches are generated on the fly by worker processes (an endless
stream) into a replay buffer of ``--buffer`` batches, from which every step
draws a batch and a random window of 384 gates; new batches keep replacing
old ones, so generation and training run at their own speeds without
over-fitting to a fixed set. Real rays have no
truth and contribute only the physics terms. The trained weights are written
to ``<out>/model.pt`` and exported with ``export.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import torch
from model import KDPNet, physics_terms, supervised_terms
from simulate import simulate_batch

N_GATES = 512
CROP = 384


class Synthetic(torch.utils.data.IterableDataset):
    def __init__(self, seed, n_rays):
        self.seed = seed
        self.n_rays = n_rays

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng([self.seed, wid])
        while True:
            b = simulate_batch(rng, self.n_rays, N_GATES)
            yield {
                "features": torch.from_numpy(b["features"]),
                "psi": torch.from_numpy(b["psi"]),
                "valid": torch.from_numpy(b["valid"].astype(np.float32)),
                "kdp": torch.from_numpy(b["kdp"]),
                "delta": torch.from_numpy(b["delta"]),
            }


def weights(args):
    if args.no_physics:
        return {
            "kdp": 1.0,
            "delta": 0.5,
            "nll": 0.1,
            "cons": 0.0,
            "neg": 0.0,
            "rough": 0.0,
        }
    return {
        "kdp": 1.0,
        "delta": 0.5,
        "nll": 0.1,
        "cons": 0.2,
        "neg": 1.0,
        "rough": 0.05,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--rays", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-physics", action="store_true")
    ap.add_argument("--init", default=None, help="weights to start from")
    ap.add_argument("--real", default=None, help="npz of real rays (fine-tuning)")
    ap.add_argument("--real-weight", type=float, default=1.0)
    ap.add_argument("--buffer", type=int, default=500)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = KDPNet().to(device)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=device))
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / 300)
        * 0.5
        * (1 + math.cos(math.pi * s / args.steps)),
    )
    loader = torch.utils.data.DataLoader(
        Synthetic(args.seed, args.rays),
        batch_size=None,
        num_workers=args.workers,
        persistent_workers=True,
        prefetch_factor=4,
    )
    real = None
    if args.real:
        z = np.load(args.real)
        real = {k: torch.from_numpy(z[k]) for k in ("features", "psi", "valid")}
        rng = np.random.default_rng(args.seed)
    w = weights(args)
    log = []
    t0 = time.time()
    buffer = []
    fill = {"n": 0}

    def produce():
        for batch in loader:
            if len(buffer) < args.buffer:
                buffer.append(batch)
            else:
                buffer[np.random.randint(args.buffer)] = batch
            fill["n"] += 1

    threading.Thread(target=produce, daemon=True).start()
    while len(buffer) < min(50, args.buffer):
        time.sleep(1.0)
    pick = np.random.default_rng(args.seed + 1)
    for step in range(args.steps):
        batch = buffer[pick.integers(len(buffer))]
        start = pick.integers(0, N_GATES - CROP + 1)
        b = {k: v[..., start : start + CROP].to(device) for k, v in batch.items()}
        kdp, delta, std = net(b["features"])
        lk, ld, nll = supervised_terms(
            kdp, delta, std, b["kdp"], b["delta"], b["valid"]
        )
        cons, neg, rough = physics_terms(kdp, delta, b["psi"], b["features"])
        terms = {
            "kdp": lk,
            "delta": ld,
            "nll": nll,
            "cons": cons,
            "neg": neg,
            "rough": rough,
        }
        loss = sum(w[k] * v for k, v in terms.items())
        if real is not None:
            idx = rng.integers(0, real["features"].shape[0], args.rays)
            rf = real["features"][idx].to(device)
            rk, rd, _ = net(rf)
            rc, rn, rr = physics_terms(rk, rd, real["psi"][idx].to(device), rf)
            # backscatter phase is rare in rain: mild L1 sparsity
            sparse = (rd.abs() * rf[:, 1]).sum() / rf[:, 1].sum().clamp(min=1)
            rloss = 0.2 * rc + 1.0 * rn + 0.05 * rr + 0.01 * sparse
            loss = loss + args.real_weight * rloss
            terms["real_cons"] = rc
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0:
            rec = {
                "step": step,
                "loss": float(loss),
                "time": time.time() - t0,
                "generated": fill["n"],
            }
            rec.update({k: float(v) for k, v in terms.items()})
            log.append(rec)
            print(json.dumps(rec), flush=True)
        if step % 2000 == 0:
            torch.save(net.state_dict(), out / "model.pt")
    torch.save(net.state_dict(), out / "model.pt")
    (out / "log.json").write_text(
        json.dumps({"args": vars(args), "weights": w, "log": log})
    )


if __name__ == "__main__":
    main()
