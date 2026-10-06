#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Evaluate ``estimate_kdp(..., method="ml")`` against the classical methods.

    python evaluate.py --out results --model full=runs/full/radarx-kdp.onnx \\
        --model nophys=runs/nophys/radarx-kdp.onnx \\
        --kgwx KGWX20220330_234639_V06

Synthetic truth (rays from ``simulate.py`` with a fixed seed, not seen in
training): bias, RMSE and coverage per stratum, the step response
(effective resolution), the response to an isolated backscatter phase bump,
and runtime. Real data: the CSAPR2 C-band PPI against the KDP of the ARM
processing, and S-band NEXRAD volumes (KGWX, KLBB) against the KDP expected
from Z_H and Z_DR in rain (self-consistency, relation from the radarx
T-matrix tables). Writes ``metrics.json``, ``tables.md`` and figures.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from simulate import GATE_SPACINGS, rain_variables, simulate_batch

from radarx.retrieve import estimate_kdp

CLASSIC = ("hubbert", "vulpiani", "monotone")


class OnnxModel:
    """ONNX Runtime session with the radarx.ml Model interface."""

    def __init__(self, path, name):
        import onnxruntime as ort

        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.info = {"name": name, "version": "dev", "licence": "MIT"}

    def run(self, inputs):
        names = [o.name for o in self.session.get_outputs()]
        return dict(zip(names, self.session.run(names, inputs)))


def as_dataset(phi, rho, z, dr):
    nray, ng = phi.shape
    data = {"PHIDP": (("azimuth", "range"), phi)}
    if rho is not None:
        data["RHOHV"] = (("azimuth", "range"), rho)
    if z is not None:
        data["DBZH"] = (("azimuth", "range"), z)
    return xr.Dataset(
        data,
        coords={
            "azimuth": np.arange(nray, dtype=float),
            "range": (np.arange(ng) + 0.5) * dr * 1000.0,
        },
    )


def run_methods(ds, models, **kw):
    out = {}
    for m in CLASSIC:
        out[m] = estimate_kdp(ds, method=m, **kw)
    for name, model in models.items():
        out[name] = estimate_kdp(ds, method="ml", model=model, **kw)
    return out


def edge_distance(echo, dr):
    """Distance (km) of every gate to the nearest non-echo gate or ray end."""
    nr, ng = echo.shape
    big = ng + 1
    left = np.zeros((nr, ng))
    right = np.zeros((nr, ng))
    run = np.zeros(nr)
    for g in range(ng):
        run = np.where(echo[:, g], run + 1, 0)
        left[:, g] = run
    run = np.zeros(nr)
    for g in range(ng - 1, -1, -1):
        run = np.where(echo[:, g], run + 1, 0)
        right[:, g] = run
    d = np.minimum(left, right)
    d[~echo] = 0
    return np.minimum(d, big) * dr


# --------------------------------------------------------------------------
# synthetic truth
# --------------------------------------------------------------------------


def synthetic(models, n_batches, n_rays, n_gates, seed=2026):
    rng = np.random.default_rng(seed)
    names = list(CLASSIC) + list(models)
    acc = {m: {"err": [], "delta_err": []} for m in names}
    strata = {k: [] for k in ("kdp", "delta", "edge", "band", "dr", "z")}
    examples = []
    timing = {m: 0.0 for m in names}
    for i in range(n_batches):
        band = "SCX"[i % 3]
        dr = GATE_SPACINGS[i % len(GATE_SPACINGS)]
        b = simulate_batch(rng, n_rays, n_gates, dr=dr, band=band)
        ds = as_dataset(b["phi"], b["rho"], b["z"], dr)
        res = {}
        for m in names:
            t = time.perf_counter()
            if m in CLASSIC:
                res[m] = estimate_kdp(ds, method=m, offset="ray")
            else:
                res[m] = estimate_kdp(ds, method="ml", model=models[m], offset="ray")
            timing[m] += time.perf_counter() - t
        echo = b["echo"]
        for m in names:
            k = res[m].KDP.values
            e = np.where(echo, k - b["kdp"], np.nan)
            acc[m]["err"].append(e[echo])
            if "PHIDP_BACKSCATTER" in res[m]:
                d = res[m].PHIDP_BACKSCATTER.values - b["delta"]
                acc[m]["delta_err"].append(d[echo])
        strata["kdp"].append(b["kdp"][echo])
        strata["delta"].append(b["delta"][echo])
        strata["edge"].append(edge_distance(echo, dr)[echo])
        strata["band"].append(np.full(echo.sum(), band))
        strata["dr"].append(np.full(echo.sum(), dr))
        z = b["z"] if b["z"] is not None else np.full(echo.shape, np.nan)
        strata["z"].append(z[echo])
        if len(examples) < 6 and 0.1 <= dr <= 0.25:
            j = int(np.argmax(np.nanmax(b["delta"], axis=1)))
            examples.append(
                {
                    "band": band,
                    "dr": dr,
                    "phi": b["phi"][j],
                    "z": None if b["z"] is None else b["z"][j],
                    "kdp": b["kdp"][j],
                    "delta": b["delta"][j],
                    "est": {m: res[m].KDP.values[j] for m in names},
                    "delta_est": {
                        m: res[m].PHIDP_BACKSCATTER.values[j]
                        for m in names
                        if "PHIDP_BACKSCATTER" in res[m]
                    },
                }
            )
        print(f"synthetic batch {i + 1}/{n_batches}", flush=True)
    st = {k: np.concatenate(v) for k, v in strata.items()}
    err = {m: np.concatenate(acc[m]["err"]) for m in names}
    derr = {
        m: np.concatenate(acc[m]["delta_err"]) for m in names if acc[m]["delta_err"]
    }
    common = np.logical_and.reduce([np.isfinite(err[m]) for m in names])
    sel = {
        "all": np.ones_like(common),
        "light (KDP < 0.3)": st["kdp"] < 0.3,
        "moderate (0.3-2)": (st["kdp"] >= 0.3) & (st["kdp"] < 2),
        "heavy (KDP >= 2)": st["kdp"] >= 2,
        "delta > 2 deg": st["delta"] > 2,
        "edges (< 1 km)": st["edge"] <= 1.0,
        "S band": st["band"] == "S",
        "C band": st["band"] == "C",
        "X band": st["band"] == "X",
    }
    table = {}
    for m in names:
        rows = {}
        for s, mask in sel.items():
            e = err[m][mask & common]
            cov = np.isfinite(err[m][mask]).mean() if mask.any() else np.nan
            rows[s] = {
                "n": int(e.size),
                "bias": float(np.mean(e)) if e.size else np.nan,
                "rmse": float(np.sqrt(np.mean(e**2))) if e.size else np.nan,
                "mae": float(np.mean(np.abs(e))) if e.size else np.nan,
                "p99_abs": float(np.percentile(np.abs(e), 99)) if e.size else np.nan,
                "coverage": float(cov),
            }
        table[m] = rows
    delta_table = {
        m: {
            "rmse_all": float(np.sqrt(np.nanmean(d**2))),
            "rmse_delta_gt2": float(np.sqrt(np.nanmean(d[st["delta"] > 2] ** 2))),
        }
        for m, d in derr.items()
    }
    gates = sum(v.size for v in strata["kdp"])
    timing = {m: t / gates * 1e6 for m, t in timing.items()}  # us per gate
    return table, delta_table, examples, timing, st, err


# --------------------------------------------------------------------------
# controlled experiments
# --------------------------------------------------------------------------


def controlled_ray(kdp, delta, z, dr, sigma, n_real, rng):
    """Rays with one KDP / delta / Z profile and independent noise."""
    ng = kdp.size
    phi_true = np.concatenate([[0.0], np.cumsum(dr * (kdp[:-1] + kdp[1:]))])
    phi = 40.0 + phi_true + delta + rng.normal(0, sigma, (n_real, ng))
    rho = np.full((n_real, ng), 0.99)
    zz = np.broadcast_to(z, (n_real, ng)).copy()
    return as_dataset(phi, rho, zz, dr)


def step_response(models, dr=0.25, n_real=200, sigma=3.0, seed=1):
    """
    KDP stepping from a low to a high value in mid-ray. The effective
    resolution is the 10-90 % rise distance of the mean estimate.
    """
    rng = np.random.default_rng(seed)
    ng = 400
    r = (np.arange(ng) + 0.5) * dr
    out = {}
    curves = {}
    for label, lo, hi, zlo, zhi in (
        ("heavy (0.5 -> 3 deg/km)", 0.5, 3.0, 38.0, 50.0),
        ("light (0 -> 0.5 deg/km)", 0.0, 0.5, 25.0, 38.0),
    ):
        mid = r[ng // 2]
        kdp = np.where(r < mid, lo, hi)
        z = np.where(r < mid, zlo, zhi)
        ds = controlled_ray(kdp, np.zeros(ng), z, dr, sigma, n_real, rng)
        res = run_methods(ds, models, offset=40.0)
        out[label] = {}
        curves[label] = {"r": r - mid, "truth": kdp}
        for m, o in res.items():
            k = o.KDP.values
            mean = np.nanmean(k, axis=0)
            std = np.nanmean(np.nanstd(k, axis=0)[ng // 2 + 40 : ng - 40])
            frac = (mean - lo) / (hi - lo)
            win = slice(ng // 2 - 80, ng // 2 + 80)
            rr = r[win]
            f = frac[win]
            r10 = rr[np.argmax(f >= 0.1)]
            r90 = rr[np.argmax(f >= 0.9)]
            out[label][m] = {
                "rise_10_90_km": float(r90 - r10),
                "noise_std_plateau": float(std),
                "bias_plateau": float(np.nanmean(mean[ng // 2 + 40 : ng - 40]) - hi),
            }
            curves[label][m] = mean
    return out, curves


def delta_bump(models, dr=0.25, n_real=200, sigma=3.0, seed=2):
    """
    Constant KDP of 1 deg/km with an isolated backscatter phase bump of
    8 degrees (Gaussian, 1 km standard deviation) in mid-ray. Spurious KDP is
    the largest departure of the mean estimate from the truth around it.
    """
    rng = np.random.default_rng(seed)
    ng = 400
    r = (np.arange(ng) + 0.5) * dr
    mid = r[ng // 2]
    kdp = np.full(ng, 1.0)
    out = {}
    curves = {"r": r - mid}
    for amp in (8.0,):
        delta = amp * np.exp(-0.5 * ((r - mid) / 1.0) ** 2)
        z = np.full(ng, 45.0)
        ds = controlled_ray(kdp, delta, z, dr, sigma, n_real, rng)
        res = run_methods(ds, models, offset=40.0)
        win = slice(ng // 2 - 40, ng // 2 + 40)
        for m, o in res.items():
            mean = np.nanmean(o.KDP.values, axis=0)
            err = mean - kdp
            out[m] = {
                "max_spurious_kdp": float(np.nanmax(np.abs(err[win]))),
                "integrated_abs_error_deg": float(np.nansum(np.abs(err[win])) * dr * 2),
            }
            if "PHIDP_BACKSCATTER" in o:
                d = np.nanmean(o.PHIDP_BACKSCATTER.values, axis=0)
                out[m]["delta_peak_estimate"] = float(np.nanmax(d[win]))
                curves[m + "_delta"] = d
            curves[m] = mean
        curves["truth_delta"] = delta
    return out, curves


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------


def self_consistency_relation(band="S", temperature=20.0):
    """
    log10(KDP / Z_lin) as a cubic polynomial of Z_DR (dB), fitted to
    normalized-gamma DSDs (Dm 0.6-3.5 mm, log10 Nw 2-5, mu 0-8) with the
    radarx T-matrix tables.
    """
    dm, lnw, mu = np.meshgrid(
        np.linspace(0.6, 3.5, 60), np.linspace(2, 5, 13), np.linspace(0, 8, 9)
    )
    z, zdr, kdp, _, _ = rain_variables(
        band, temperature, lnw.ravel(), dm.ravel(), mu.ravel()
    )
    ok = (zdr > 0.2) & (zdr < 3.5) & (kdp > 1e-4)
    y = np.log10(kdp[ok] / 10 ** (z[ok] / 10))
    return np.polyfit(zdr[ok], y, 3)


def nexrad_sweep(path, sweep=0):
    import xradar as xd

    dtree = xd.io.open_nexradlevel2_datatree(path, sweep=[sweep])
    ds = dtree[f"sweep_{sweep}"].to_dataset()
    ds["DBZH"] = ds.DBZH.where(ds.DBZH > -32)
    ds["ZDR"] = ds.ZDR.where(ds.ZDR > -12.9)
    ds["RHOHV"] = ds.RHOHV.where(ds.RHOHV > 0.21)
    return ds


def find_phase_sweep(path):
    """Index of the lowest sweep with differential phase."""
    import xradar as xd

    for i in range(4):
        dtree = xd.io.open_nexradlevel2_datatree(path, sweep=[i])
        if "PHIDP" in dtree[f"sweep_{i}"].ds:
            return i
    raise ValueError("no sweep with PHIDP")


def real_nexrad(path, models, coef, label, figdir):
    ds = nexrad_sweep(path, find_phase_sweep(path))
    res = {}
    timing = {}
    for m in list(CLASSIC) + list(models):
        t = time.perf_counter()
        if m in CLASSIC:
            res[m] = estimate_kdp(ds, method=m)
        else:
            res[m] = estimate_kdp(ds, method="ml", model=models[m])
        timing[m] = time.perf_counter() - t
    z = ds.DBZH.values
    zdr = ds.ZDR.values
    rho = ds.RHOHV.values
    r = ds.range.values / 1000.0
    rain = (z >= 35) & (z <= 52) & (rho >= 0.98) & (zdr > 0.2) & (zdr < 3.5)
    rain &= (r < 100)[None, :]
    expect = 10 ** (z / 10) * 10 ** np.polyval(coef, zdr)
    light = (z >= 15) & (z < 25) & (rho >= 0.97) & (r < 100)[None, :]
    out = {}
    for m, o in res.items():
        k = o.KDP.values
        sel = rain & np.isfinite(k)
        sel_l = light & np.isfinite(k)
        out[m] = {
            "n_rain": int(sel.sum()),
            "mean_ratio_est_over_selfcons": float(
                np.mean(k[sel]) / np.mean(expect[sel])
            ),
            "corr_selfcons": float(np.corrcoef(k[sel], expect[sel])[0, 1]),
            "rmse_selfcons": float(np.sqrt(np.mean((k[sel] - expect[sel]) ** 2))),
            "light_rain_std": float(np.std(k[sel_l])),
            "light_rain_mean": float(np.mean(k[sel_l])),
            "negative_frac_rain": float(np.mean(k[sel] < 0)),
            "runtime_s": timing[m],
        }
    ppi_figure(ds, res, f"{label} {path.name}", figdir / f"ppi_{label}.png", "DBZH")
    return out, ds, res


def real_csapr2(models, figdir):
    import xradar as xd
    from open_radar_data import DATASETS

    file = DATASETS.fetch("corcsapr2cmacppiM1.c1.20181111.030003.nc")
    ds = (
        xd.io.open_cfradial1_datatree(file)["sweep_0"]
        .to_dataset(inherit="all_coords")
        .load()
    )
    fields = {
        "phidp": "uncorrected_differential_phase",
        "rhohv": "uncorrected_copol_correlation_coeff",
        "dbzh": "uncorrected_reflectivity_h",
    }
    res = {}
    timing = {}
    for m in list(CLASSIC) + list(models):
        t = time.perf_counter()
        if m in CLASSIC:
            res[m] = estimate_kdp(ds, method=m, **fields)
        else:
            res[m] = estimate_kdp(ds, method="ml", model=models[m], **fields)
        timing[m] = time.perf_counter() - t
    ref = ds.specific_differential_phase.values
    z = ds[fields["dbzh"]].values
    sel0 = (z > 30) & (ds[fields["rhohv"]].values > 0.95) & np.isfinite(ref)
    out = {}
    for m, o in res.items():
        k = o.KDP.values
        sel = sel0 & np.isfinite(k)
        out[m] = {
            "n": int(sel.sum()),
            "corr_radar_kdp": float(np.corrcoef(k[sel], ref[sel])[0, 1]),
            "bias_vs_radar_kdp": float(np.mean(k[sel] - ref[sel])),
            "rmse_vs_radar_kdp": float(np.sqrt(np.mean((k[sel] - ref[sel]) ** 2))),
            "mean_kdp_cores_z45": float(np.nanmean(k[z >= 45])),
            "runtime_s": timing[m],
        }
    fig, axes = plt.subplots(1, len(res), figsize=(3.2 * len(res), 3.4), sharey=True)
    for ax, (m, o) in zip(axes, res.items()):
        k = o.KDP.values
        sel = sel0 & np.isfinite(k)
        ax.hexbin(ref[sel], k[sel], gridsize=50, bins="log", extent=(-1, 8, -1, 8))
        ax.plot([-1, 8], [-1, 8], color="r", lw=0.8)
        ax.set_title(f"{m}\nr={out[m]['corr_radar_kdp']:.3f}", fontsize=9)
        ax.set_xlabel("ARM KDP (deg/km)")
    axes[0].set_ylabel("estimate (deg/km)")
    fig.tight_layout()
    fig.savefig(figdir / "csapr2_vs_radar_kdp.png", dpi=130)
    plt.close(fig)
    ds = ds.assign(DBZH=ds[fields["dbzh"]])
    ppi_figure(
        ds,
        res,
        "CSAPR2 2018-11-11 03:00 UTC",
        figdir / "ppi_csapr2.png",
        "DBZH",
        extent=110,
    )
    return out


def ppi_figure(ds, res, title, path, zname, extent=150):
    az = np.deg2rad(ds.azimuth.values)[:, None]
    r = ds.range.values[None, :] / 1000.0
    x = r * np.sin(az)
    y = r * np.cos(az)
    panels = [(ds[zname], "Z_H (dBZ)", "turbo", -10, 65)]
    for m, o in res.items():
        panels.append((o.KDP, f"KDP {m}", "turbo", -1, 6))
    n = len(panels)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(4 * cols, 3.8 * rows), sharex=True, sharey=True
    )
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax, (da, label, cmap, vmin, vmax) in zip(axes.flat, panels):
        pm = ax.pcolormesh(
            x, y, da.values, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto"
        )
        fig.colorbar(pm, ax=ax, shrink=0.8)
        ax.set_title(label, fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------
# figures and tables
# --------------------------------------------------------------------------


def example_figure(examples, path):
    fig, axes = plt.subplots(
        len(examples), 2, figsize=(13, 2.6 * len(examples)), squeeze=False
    )
    for row, ex in zip(axes, examples):
        r = np.arange(ex["kdp"].size) * ex["dr"]
        row[0].plot(r, ex["phi"], ".", ms=1.5, color="0.5")
        row[0].set_ylabel(f"{ex['band']} band\nraw phase (deg)")
        ax2 = row[0].twinx()
        ax2.plot(r, ex["delta"], color="k", lw=1, label="true delta")
        for m, d in ex["delta_est"].items():
            ax2.plot(r, d, lw=0.8, label=f"delta {m}")
        ax2.set_ylabel("delta (deg)")
        ax2.legend(fontsize=6, loc="upper right")
        row[1].plot(r, ex["kdp"], color="k", lw=1.5, label="truth")
        for m, k in ex["est"].items():
            row[1].plot(r, k, lw=0.8, label=m)
        row[1].set_ylabel("KDP (deg/km)")
        row[1].legend(fontsize=6, ncol=3, loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("range (km)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def curves_figure(step_curves, bump_curves, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (label, c) in zip(axes[:2], step_curves.items()):
        ax.plot(c["r"], c["truth"], color="k", lw=1.5, label="truth")
        for m, v in c.items():
            if m in ("r", "truth"):
                continue
            ax.plot(c["r"], v, lw=1, label=m)
        ax.set_xlim(-8, 8)
        ax.set_title(f"step response, {label}", fontsize=9)
        ax.set_xlabel("range from step (km)")
        ax.set_ylabel("mean KDP (deg/km)")
        ax.legend(fontsize=7)
    ax = axes[2]
    ax.axhline(1.0, color="k", lw=1.5, label="truth")
    for m, v in bump_curves.items():
        if m == "r" or m.endswith("_delta"):
            continue
        ax.plot(bump_curves["r"], v, lw=1, label=m)
    ax.set_xlim(-8, 8)
    ax.set_title("8 deg backscatter phase bump, KDP = 1 deg/km", fontsize=9)
    ax.set_xlabel("range from bump (km)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def md_tables(metrics):
    lines = []
    syn = metrics["synthetic"]
    methods = list(syn)
    strata = list(next(iter(syn.values())))
    for key, label in (("rmse", "RMSE"), ("bias", "bias")):
        lines.append(f"\n**Synthetic truth: KDP {label} (deg/km)**\n")
        lines.append("| stratum | " + " | ".join(methods) + " |")
        lines.append("|---" * (len(methods) + 1) + "|")
        for s in strata:
            vals = " | ".join(f"{syn[m][s][key]:.3f}" for m in methods)
            lines.append(f"| {s} (n={syn[methods[0]][s]['n']}) | {vals} |")
    lines.append("\n**Coverage of echo gates (fraction with an estimate)**\n")
    lines.append("| stratum | " + " | ".join(methods) + " |")
    lines.append("|---" * (len(methods) + 1) + "|")
    for s in strata:
        lines.append(
            f"| {s} | "
            + " | ".join(f"{syn[m][s]['coverage']:.3f}" for m in methods)
            + " |"
        )
    lines.append("\n**Backscatter phase delta (ML only), RMSE (deg)**\n")
    lines.append("| model | all echo gates | true delta > 2 deg |")
    lines.append("|---|---|---|")
    for m, d in metrics["synthetic_delta"].items():
        lines.append(f"| {m} | {d['rmse_all']:.2f} | {d['rmse_delta_gt2']:.2f} |")
    for label, rows in metrics["step"].items():
        lines.append(
            f"\n**Step response, {label}: 10-90 % rise (km), plateau noise std and bias (deg/km)**\n"
        )
        lines.append("| method | rise (km) | noise std | bias |")
        lines.append("|---|---|---|---|")
        for m, v in rows.items():
            lines.append(
                f"| {m} | {v['rise_10_90_km']:.2f} | {v['noise_std_plateau']:.3f} | {v['bias_plateau']:+.3f} |"
            )
    lines.append("\n**Isolated 8 deg backscatter bump (KDP = 1 deg/km)**\n")
    lines.append(
        "| method | max spurious KDP (deg/km) | integrated abs. error (deg) | peak delta estimate (deg) |"
    )
    lines.append("|---|---|---|---|")
    for m, v in metrics["delta_bump"].items():
        pk = v.get("delta_peak_estimate")
        lines.append(
            f"| {m} | {v['max_spurious_kdp']:.3f} | {v['integrated_abs_error_deg']:.2f} | {'' if pk is None else f'{pk:.2f}'} |"
        )
    if "csapr2" in metrics:
        lines.append(
            "\n**CSAPR2 (C band) vs the KDP of the ARM processing (Z > 30 dBZ, rhohv > 0.95)**\n"
        )
        lines.append(
            "| method | n | r | bias | RMSE | mean KDP, Z >= 45 | runtime (s) |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for m, v in metrics["csapr2"].items():
            lines.append(
                f"| {m} | {v['n']} | {v['corr_radar_kdp']:.3f} | {v['bias_vs_radar_kdp']:+.3f} | {v['rmse_vs_radar_kdp']:.3f} | {v['mean_kdp_cores_z45']:.2f} | {v['runtime_s']:.2f} |"
            )
    for key in [k for k in metrics if k.startswith("nexrad_")]:
        lines.append(
            f"\n**{key[7:]}: self-consistency in rain (35-52 dBZ, rhohv >= 0.98, r < 100 km) and light-rain noise (15-25 dBZ)**\n"
        )
        lines.append(
            "| method | n | mean KDP / self-consistent | r | RMSE | light-rain mean | light-rain std | negative in rain | runtime (s) |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for m, v in metrics[key].items():
            lines.append(
                f"| {m} | {v['n_rain']} | {v['mean_ratio_est_over_selfcons']:.3f} | {v['corr_selfcons']:.3f} | {v['rmse_selfcons']:.3f} | "
                f"{v['light_rain_mean']:+.3f} | {v['light_rain_std']:.3f} | {v['negative_frac_rain']:.3f} | {v['runtime_s']:.2f} |"
            )
    lines.append(
        "\n**Runtime on synthetic rays (microseconds per echo gate, one thread for ML via ONNX Runtime defaults)**\n"
    )
    lines.append("| " + " | ".join(metrics["timing_us_per_gate"]) + " |")
    lines.append("|---" * len(metrics["timing_us_per_gate"]) + "|")
    lines.append(
        "| "
        + " | ".join(f"{v:.2f}" for v in metrics["timing_us_per_gate"].values())
        + " |"
    )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", action="append", default=[], help="name=path.onnx")
    ap.add_argument("--batches", type=int, default=48)
    ap.add_argument("--rays", type=int, default=32)
    ap.add_argument("--gates", type=int, default=800)
    ap.add_argument("--kgwx", action="append", default=[])
    ap.add_argument("--klbb", action="store_true")
    ap.add_argument("--csapr2", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = {}
    for spec in args.model:
        name, path = spec.split("=", 1)
        models[f"ml-{name}"] = OnnxModel(path, name)
    metrics = {}
    table, dtable, examples, timing, _, _ = synthetic(
        models, args.batches, args.rays, args.gates
    )
    metrics["synthetic"] = table
    metrics["synthetic_delta"] = dtable
    metrics["timing_us_per_gate"] = timing
    if examples:
        example_figure(examples[:4], out / "synthetic_examples.png")
    metrics["step"], step_curves = step_response(models)
    metrics["delta_bump"], bump_curves = delta_bump(models)
    curves_figure(step_curves, bump_curves, out / "step_and_delta_response.png")
    coef = self_consistency_relation()
    metrics["selfcons_coef_S"] = coef.tolist()
    for path in args.kgwx:
        p = Path(path)
        metrics[f"nexrad_{p.name[:4]} {p.name[4:19]}"], _, _ = real_nexrad(
            p, models, coef, p.name[:4], out
        )
    if args.klbb:
        from open_radar_data import DATASETS

        p = Path(DATASETS.fetch("KLBB20160601_150025_V06"))
        metrics[f"nexrad_KLBB {p.name[4:19]}"], _, _ = real_nexrad(
            p, models, coef, "KLBB", out
        )
    if args.csapr2:
        metrics["csapr2"] = real_csapr2(models, out)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (out / "tables.md").write_text(md_tables(metrics))
    print(md_tables(metrics))


if __name__ == "__main__":
    main()
