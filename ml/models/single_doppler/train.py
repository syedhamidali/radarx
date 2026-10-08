"""
Train the physics-informed single-Doppler network.

Every batch mixes crops of real samples (target: the multi-Doppler wind where
two radars observe with a beam crossing angle above 30 degrees) and synthetic
samples (target: the exact wind). The loss is

    L = sum_c a_c <(pred_c - target_c)^2>_weight + lambda * J(pred) / N

with ``a = (1, 1, 2)`` for ``(u, v, w)`` and ``J`` the radarx variational
cost of the one radar (observation, anelastic mass continuity, smoothness,
background; ``physics.variational_cost``) per grid cell.

::

    python train.py --data DATA --out RUN [--lambda-phys 0.1] [--epochs 40]
"""

import argparse
import json
import os
import time

import numpy as np
import synthetic
import torch
from data import RealFile, augment, real_files, tensors
from model import SingleDopplerNet
from physics import variational_cost
from torch.utils.data import DataLoader, IterableDataset

from radarx.retrieve.single_doppler import _predict

KEYS = ("features", "coef", "target", "obs", "rho", "bg", "bgw", "truth", "weight")
COMPONENT_WEIGHTS = (1.0, 1.0, 2.0)


class SyntheticStream(IterableDataset):
    def __init__(self, seed, size):
        self.seed, self.size = seed, size

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng([self.seed, wid, int(time.time() * 1e3) % 2**31])
        while True:
            s = synthetic.sample(rng, ny=self.size, nx=self.size)
            yield tensors(augment(rng, s))


def stack(items, device):
    return {
        k: torch.from_numpy(np.stack([it[k] for it in items])).to(device) for k in KEYS
    }


def losses(model, b, lam):
    pred = model(b["features"])
    w = b["weight"]
    cw = torch.tensor(COMPONENT_WEIGHTS, device=pred.device).view(1, 3, 1, 1, 1)
    err = ((pred - b["truth"]) ** 2 * cw).sum(1)
    sup = (err * w).sum() / w.sum().clamp(min=1.0)
    terms = variational_cost(
        pred,
        b["coef"],
        b["target"],
        b["obs"],
        b["rho"],
        b["bg"],
        b["bgw"],
        1000.0,
        1000.0,
        500.0,
        reduce="mean",
    )
    phys = sum(t.mean() for t in terms.values())
    return sup + lam * phys, sup, phys, {k: float(t.mean()) for k, t in terms.items()}


class TorchRunner:
    """``run(dict) -> dict`` interface of radarx models around a torch module."""

    def __init__(self, model, device):
        self.model, self.device = model, device

    def run(self, inputs):
        x = torch.from_numpy(next(iter(inputs.values()))).to(self.device)
        with torch.no_grad():
            return {"wind": self.model(x).cpu().numpy()}


def rmse_real(model, files, device):
    """RMSE of u, v, w against the multi-Doppler wind (good cells), all radars."""
    runner = TorchRunner(model, device)
    se, n = np.zeros(3), 0
    for f in files:
        for k in range(len(f.radars)):
            s = f.sample(k)
            feats = tensors(s)["features"]
            wind = _predict(runner, feats, 4)
            m = s["weight"] > 0
            se += ((wind - s["truth"]) ** 2)[:, m].sum(1)
            n += int(m.sum())
    return np.sqrt(se / max(n, 1))


def rmse_synthetic(model, samples, device):
    runner = TorchRunner(model, device)
    se, n = np.zeros(3), 0
    for s in samples:
        wind = _predict(runner, tensors(s)["features"], 4)
        m = np.isfinite(s["vr"])
        se += ((wind - s["truth"]) ** 2)[:, m].sum(1)
        n += int(m.sum())
    return np.sqrt(se / max(n, 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--steps", type=int, default=200, help="steps per epoch")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--crop", type=int, default=64)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lambda-phys", type=float, default=0.1)
    p.add_argument("--real-fraction", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init", default=None, help="checkpoint to start from")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    train = [RealFile(f) for f in real_files(a.data, "train")]
    val = [RealFile(f) for f in real_files(a.data, "val")]
    if not train:
        a.real_fraction = 0.0
    print(
        f"{len(train)} train files, {len(val)} val files, device {device}", flush=True
    )
    val_rng = np.random.default_rng(1234)
    val_syn = [synthetic.sample(val_rng, ny=96, nx=96) for _ in range(12)]

    stream = DataLoader(
        SyntheticStream(a.seed, a.crop),
        batch_size=None,
        num_workers=a.workers,
        persistent_workers=a.workers > 0,
        prefetch_factor=4 if a.workers > 0 else None,
    )
    syn_iter = iter(stream)
    model = SingleDopplerNet(width=a.width)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu"))
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * a.steps, pct_start=0.1
    )
    history, best = [], np.inf
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    for epoch in range(a.epochs):
        model.train()
        t0 = time.perf_counter()
        acc = np.zeros(3)
        for step in range(a.steps):
            items = []
            for _ in range(a.batch):
                if train and rng.uniform() < a.real_fraction:
                    f = train[rng.integers(len(train))]
                    items.append(tensors(augment(rng, f.random_sample(rng, a.crop))))
                else:
                    items.append(next(syn_iter))
            b = stack(items, device)
            loss, sup, phys, _ = losses(model, b, a.lambda_phys)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            acc += [float(loss), float(sup), float(phys)]
        model.eval()
        r_val = rmse_real(model, val, device) if val else np.full(3, np.nan)
        r_syn = rmse_synthetic(model, val_syn, device)
        score = float(np.nanmean(r_val)) if val else float(np.mean(r_syn))
        rec = dict(
            epoch=epoch,
            loss=acc[0] / a.steps,
            supervised=acc[1] / a.steps,
            physics=acc[2] / a.steps,
            val_real_rmse=r_val.tolist(),
            val_synthetic_rmse=r_syn.tolist(),
            seconds=time.perf_counter() - t0,
        )
        history.append(rec)
        print(json.dumps(rec), flush=True)
        torch.save(model.state_dict(), os.path.join(a.out, "last.pt"))
        if score < best:
            best = score
            torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
        json.dump(history, open(os.path.join(a.out, "history.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
