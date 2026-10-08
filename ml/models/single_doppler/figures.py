"""
Figures of the evaluation (``evaluate.py`` scores) and of one test case.

::

    python figures.py --scores EVAL/scores_test.json --data DATA --model M.onnx --history RUN/history.json --out FIGS
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from data import RealFile, real_files  # noqa: E402
from evaluate import CROSS_BINS, ELEV_BINS, METHODS, RANGE_BINS, retrieve  # noqa: E402

COLORS = dict(zip(METHODS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
MARKERS = dict(zip(METHODS, ["o", "s", "^", "D", "v"]))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update(
    {
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def bars(scores, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained", sharey=True)
    for ax, (key, title) in zip(
        axes,
        (
            ("real", "vs multi-Doppler (hold-out case)"),
            ("synthetic", "vs exact truth (synthetic)"),
        ),
    ):
        res = scores[key]
        methods = [m for m in METHODS if f"all|{m}" in res]
        width = 0.8 / len(methods)
        for i, m in enumerate(methods):
            r = res[f"all|{m}"]["rmse"]
            xs = np.arange(3) + (i - (len(methods) - 1) / 2) * width
            b = ax.bar(xs, r, width * 0.92, color=COLORS[m], label=m)
            ax.bar_label(b, fmt="%.1f", fontsize=7, color=MUTED, padding=1)
        ax.set_xticks(range(3), ["u", "v", "w"])
        ax.set_title(f"RMSE {title}", color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("RMSE (m s$^{-1}$)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(out, dpi=130)


def curves(scores, out, key="real"):
    res = scores[key]
    specs = (
        ("range", RANGE_BINS / 1e3, "distance from the radar (km)"),
        ("elevation", ELEV_BINS, "beam elevation (deg)"),
        ("cross", CROSS_BINS, "angle between beam and wind (deg)"),
    )
    fig, axes = plt.subplots(
        2, 3, figsize=(13, 6.5), layout="constrained", sharex="col"
    )
    for col, (name, bins, label) in enumerate(specs):
        centres = 0.5 * (bins[:-1] + bins[1:])
        for row, (q, comp) in enumerate(((0, "horizontal (u, v)"), (2, "w"))):
            ax = axes[row, col]
            for m in METHODS:
                vals = []
                for b in range(len(bins) - 1):
                    r = res.get(f"{name}|{m}|{b}")
                    if r is None or r["n"] < 200:
                        vals.append(np.nan)
                        continue
                    rm = np.asarray(r["rmse"])
                    vals.append(
                        np.sqrt(0.5 * (rm[0] ** 2 + rm[1] ** 2)) if q == 0 else rm[2]
                    )
                if np.isfinite(vals).any():
                    ax.plot(
                        centres,
                        vals,
                        color=COLORS[m],
                        marker=MARKERS[m],
                        ms=5,
                        lw=2,
                        label=m,
                    )
            ax.set_ylabel(f"RMSE {comp} (m s$^{{-1}}$)")
            if row == 1:
                ax.set_xlabel(label)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"Skill vs geometry ({'hold-out case vs multi-Doppler' if key == 'real' else 'synthetic truth'})",
        color=INK,
    )
    fig.savefig(out, dpi=130)


def case_map(path, radar, model, out, height=5000.0):
    f = RealFile(path)
    s = f.sample(radar)
    preds, _ = retrieve(s, model, methods=("variational", "network+var"))
    k = int(np.argmin(np.abs(s["z"] - height)))
    good = s["weight"][k] > 0
    seen = np.isfinite(s["vr"][k])
    km = (s["x"] / 1e3, s["y"] / 1e3)
    panels = [
        ("multi-Doppler (two radars)", s["truth"], good),
        ("variational, one radar", preds["variational"], seen),
        ("network", preds["network"], seen),
        ("network + variational", preds["network+var"], seen),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), layout="constrained", sharey=True)
    for ax, (title, wind, mask) in zip(axes, panels):
        im = ax.pcolormesh(
            *km, np.where(mask, wind[2][k], np.nan), cmap="RdBu_r", vmin=-8, vmax=8
        )
        st = 8
        sub = (slice(None, None, st), slice(None, None, st))
        uu = np.where(mask, wind[0][k], np.nan)[sub]
        vv = np.where(mask, wind[1][k], np.nan)[sub]
        ax.quiver(km[0][sub[1]], km[1][sub[0]], uu, vv, scale=800, width=0.002)
        for j, (rx_, ry_, _) in enumerate(f.radars):
            ax.plot(rx_ / 1e3, ry_ / 1e3, "^", color="k" if j == radar else MUTED, ms=8)
            ax.annotate(
                f.names[j],
                (rx_ / 1e3, ry_ / 1e3),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_title(title, color=INK)
        ax.set_aspect("equal")
        ax.set_xlabel("x (km)")
        ax.grid(False)
    axes[0].set_ylabel("y (km)")
    fig.colorbar(im, ax=axes, label="w (m s$^{-1}$)", shrink=0.85)
    fig.suptitle(
        f"{os.path.basename(path)[:-3]}: {f.names[radar]} alone, {height / 1e3:.0f} km (w and wind)",
        color=INK,
    )
    fig.savefig(out, dpi=110)


def history_plot(hist, out):
    h = json.load(open(hist))
    ep = [r["epoch"] for r in h]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    axes[0].plot(
        ep,
        [r["supervised"] for r in h],
        color=COLORS["background"],
        lw=2,
        label="supervised",
    )
    axes[0].plot(
        ep,
        [r["physics"] for r in h],
        color=COLORS["vad"],
        lw=2,
        label="physics (J per cell)",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].legend(frameon=False)
    for q, c in enumerate("uvw"):
        axes[1].plot(
            ep,
            [r["val_real_rmse"][q] for r in h],
            lw=2,
            color=list(COLORS.values())[q],
            label=f"{c} (validation case)",
        )
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("RMSE vs multi-Doppler (m s$^{-1}$)")
    axes[1].legend(frameon=False)
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--history", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--case", default=None)
    p.add_argument("--radar", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    scores = json.load(open(a.scores))
    bars(scores, os.path.join(a.out, "rmse_bars.png"))
    curves(scores, os.path.join(a.out, "rmse_vs_geometry_real.png"), "real")
    curves(scores, os.path.join(a.out, "rmse_vs_geometry_synthetic.png"), "synthetic")
    if a.history:
        history_plot(a.history, os.path.join(a.out, "training.png"))
    case = a.case or real_files(a.data, "test")[len(real_files(a.data, "test")) // 2]
    case_map(case, a.radar, a.model, os.path.join(a.out, "case_map.png"))
