"""
Validate DSD retrievals against PIPS on naive and trajectory-matched pairs.

    python evaluate.py --pairs DIR --iop IOP2 --samples samples.csv --out DIR

The learned prior is built from the aloft-equivalent PIPS samples of the
OTHER deployments only (leave-one-IOP-out), so the validation IOP is
independent of it. Methods: constrained gamma (Cao et al. 2008 mu-Lambda),
normalized gamma (mu = 3), both with the intercept from KDP where
KDP >= 1 deg/km, and the Bayesian retrieval with the generic and the learned
prior (Z_H, Z_DR and K_DP).
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from radarx.retrieve import dsd, dsd_bayesian, dsd_prior  # noqa: E402

METHODS = ("constrained", "normalized", "bayes_generic", "bayes_learned")
LABELS = {
    "constrained": "constrained gamma",
    "normalized": "normalized gamma (mu=3)",
    "bayes_generic": "Bayesian, generic prior",
    "bayes_learned": "Bayesian, learned prior",
}


def learned_prior(samples, exclude):
    df = pd.read_csv(samples)
    df = df[(df.kind == "aloft") & (df.iop != exclude)]
    ds = xr.Dataset(
        {
            "NW": ("t", 10**df.log10_nw.values),
            "DM": ("t", df.dm.values),
            "MU": ("t", df.mu.values),
        }
    )
    return dsd_prior(ds), len(df)


def calibration(pairs, ok):
    """
    Leave-one-probe-out radar offsets and error model: for every pair, the
    median and spread of radar minus PIPS-simulated Z_H, Z_DR and K_DP over
    the pairs of the OTHER probes.
    """
    n = pairs.sizes["pair"]
    off = {k: np.zeros(n) for k in ("DBZH", "ZDR")}
    sd = {k: np.zeros(n) for k in ("DBZH", "ZDR", "KDP")}
    probes = pairs.probe.values
    for p in np.unique(probes):
        train = ok & (probes != p)
        test = probes == p
        for k in ("DBZH", "ZDR", "KDP"):
            d = pairs[k].values[train] - pairs[f"PIPS_{k}_SIM"].values[train]
            d = d[np.isfinite(d)]
            med = np.median(d)
            mad = 1.4826 * np.median(np.abs(d - med))  # robust spread
            if k in off:
                off[k][test] = med
            sd[k][test] = mad
    return off, sd


def retrieve(pairs, prior, calibrate=False, ok=None):
    rad = xr.Dataset(
        {k: ("pair", pairs[k].values.astype(float)) for k in ("DBZH", "ZDR", "KDP")},
        coords={"pair": np.arange(pairs.sizes["pair"])},
    )
    errs = [None] * pairs.sizes["pair"]
    if calibrate:
        off, sd = calibration(pairs, ok)
        rad["DBZH"] = rad.DBZH - off["DBZH"]
        rad["ZDR"] = rad.ZDR - off["ZDR"]
        errs = [
            {
                "zh": sd["DBZH"][i],
                "zh_bias": 0.5,
                "zdr": sd["ZDR"][i],
                "zdr_bias": 0.05,
                "kdp": sd["KDP"][i],
                "kdp_rel": 0.1,
            }
            for i in range(pairs.sizes["pair"])
        ]
    out = {}
    for m in ("constrained", "normalized"):
        r = dsd(rad, m, kdp="KDP", band="S", nw_range=None)
        out[m] = {
            "DM": r.DM.values,
            "LOG10_NW": np.log10(r.NW.values),
            "MU": r.MU.values,
            "RAIN_RATE": r.RAIN_RATE.values,
        }
    for name, p in (("bayes_generic", "generic"), ("bayes_learned", prior)):
        if calibrate:  # error model per probe (leave-one-probe-out)
            parts = []
            for p_name in np.unique(pairs.probe.values):
                idx = np.flatnonzero(pairs.probe.values == p_name)
                parts.append(
                    dsd_bayesian(
                        rad.isel(pair=idx),
                        kdp="KDP",
                        band="S",
                        prior=p,
                        errors=errs[idx[0]],
                    )
                )
            r = xr.concat(parts, "pair").sortby("pair")
        else:
            r = dsd_bayesian(rad, kdp="KDP", band="S", prior=p)
        out[name] = {k: r[k].values for k in ("DM", "LOG10_NW", "MU", "RAIN_RATE")}
        out[name]["Q"] = {
            k: r[k + "_QUANTILES"].values for k in ("DM", "LOG10_NW", "MU", "RAIN_RATE")
        }
        out[name]["MISFIT"] = r.MISFIT.values
    return out


def reference(pairs):
    return {
        "DM": pairs.PIPS_DM.values,
        "LOG10_NW": np.log10(pairs.PIPS_NW.values),
        "MU": pairs.PIPS_MU.values,
        "RAIN_RATE": pairs.PIPS_RAIN_RATE.values,
    }


def select(pairs):
    return (
        (pairs.PIPS_RAIN_RATE >= 0.5)
        & (pairs.PIPS_NT >= 50)
        & np.isfinite(pairs.PIPS_MU)
        & (pairs.DBZH >= 10)
        & (pairs.RHOHV >= 0.95)
    ).values


def scores(est, ref, ok):
    s = {}
    for k in ("DM", "LOG10_NW", "MU", "RAIN_RATE"):
        e, r = est[k][ok], ref[k][ok]
        good = np.isfinite(e) & np.isfinite(r)
        e, r = e[good], r[good]
        if k == "RAIN_RATE":
            s[k] = {
                "bias_pct": float(100 * (e.sum() / r.sum() - 1)),
                "nrmse_pct": float(100 * np.sqrt(np.mean((e - r) ** 2)) / r.mean()),
                "corr": float(np.corrcoef(e, r)[0, 1]),
                "n": int(good.sum()),
            }
        else:
            s[k] = {
                "bias": float(np.mean(e - r)),
                "rmse": float(np.sqrt(np.mean((e - r) ** 2))),
                "corr": float(np.corrcoef(e, r)[0, 1]),
                "n": int(good.sum()),
            }
    if "Q" in est:
        cov = {}
        for k in ("DM", "LOG10_NW", "MU", "RAIN_RATE"):
            q, r = est["Q"][k][:, ok], ref[k][ok]
            good = np.isfinite(r) & np.isfinite(q).all(0)
            q, r = q[:, good], r[good]
            cov[k] = {
                "68": float(np.mean((r >= q[1]) & (r <= q[3]))),
                "95": float(np.mean((r >= q[0]) & (r <= q[4]))),
            }
        s["coverage"] = cov
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--iop", default="IOP2")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prior, n_train = learned_prior(a.samples, a.iop)
    print("learned prior from", n_train, "aloft samples (excluding", a.iop + ")")
    results, data = {}, {}
    for kind in ("naive", "traj"):
        pairs = xr.open_dataset(os.path.join(a.pairs, f"pairs_{a.iop}_{kind}.nc"))
        ok = select(pairs)
        est = retrieve(pairs, prior, a.calibrate, ok)
        ref = reference(pairs)
        data[kind] = (pairs, ok, est, ref)
        res = {m: scores(est[m], ref, ok) for m in METHODS}
        res["n_pairs"] = int(ok.sum())
        dz = pairs.DBZH.values[ok] - pairs.PIPS_DBZH_SIM.values[ok]
        dd = pairs.ZDR.values[ok] - pairs.PIPS_ZDR_SIM.values[ok]
        res["radar_minus_pips"] = {
            "DBZH_median": float(np.median(dz)),
            "DBZH_sd": float(np.std(dz)),
            "ZDR_median": float(np.median(dd)),
            "ZDR_sd": float(np.std(dd)),
        }
        results[kind] = res
    with open(os.path.join(a.out, f"metrics_{a.iop}.json"), "w") as f:
        json.dump(results, f, indent=1)
    table(results, os.path.join(a.out, f"metrics_{a.iop}.md"))
    figures(data, a.out, a.iop)
    print(open(os.path.join(a.out, f"metrics_{a.iop}.md")).read())


def table(results, path):
    lines = []
    for kind in ("naive", "traj"):
        r = results[kind]
        rz = r["radar_minus_pips"]
        lines.append(
            f"\n**{'naive collocation' if kind == 'naive' else 'trajectory matching'}** "
            f"({r['n_pairs']} pairs; radar - PIPS: Z_H {rz['DBZH_median']:+.1f} +- {rz['DBZH_sd']:.1f} dB, "
            f"Z_DR {rz['ZDR_median']:+.2f} +- {rz['ZDR_sd']:.2f} dB)\n"
        )
        lines.append(
            "| method | Dm bias / RMSE (mm) | log10 Nw bias / RMSE | mu bias / RMSE | R bias / NRMSE (%) |"
        )
        lines.append("|---|---|---|---|---|")
        for m in METHODS:
            s = r[m]
            lines.append(
                f"| {LABELS[m]} | {s['DM']['bias']:+.2f} / {s['DM']['rmse']:.2f} | "
                f"{s['LOG10_NW']['bias']:+.2f} / {s['LOG10_NW']['rmse']:.2f} | "
                f"{s['MU']['bias']:+.1f} / {s['MU']['rmse']:.1f} | "
                f"{s['RAIN_RATE']['bias_pct']:+.0f} / {s['RAIN_RATE']['nrmse_pct']:.0f} |"
            )
        for m in ("bayes_generic", "bayes_learned"):
            c = r[m]["coverage"]
            lines.append(
                f"\n{LABELS[m]} coverage of the 68 / 95 % intervals: "
                + ", ".join(f"{k} {c[k]['68']:.2f} / {c[k]['95']:.2f}" for k in c)
            )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def figures(data, out, iop):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), constrained_layout=True)
    for row, kind in enumerate(("naive", "traj")):
        pairs, ok, est, ref = data[kind]
        for ax, m in zip(axes[row], METHODS):
            e = est[m]["DM"][ok]
            r = ref["DM"][ok]
            if "Q" in est[m]:
                q = est[m]["Q"]["DM"][:, ok]
                ax.errorbar(
                    r, e, yerr=[e - q[1], q[3] - e], fmt="none", ecolor="0.75", lw=0.6
                )
            ax.plot(r, e, ".", ms=4)
            ax.plot([0.5, 3.5], [0.5, 3.5], "k-", lw=0.8)
            good = np.isfinite(e) & np.isfinite(r)
            rmse = np.sqrt(np.mean((e[good] - r[good]) ** 2))
            ax.set_title(f"{LABELS[m]}\n{kind}: RMSE {rmse:.2f} mm", fontsize=9)
            ax.set_xlim(0.5, 3.5)
            ax.set_ylim(0.5, 3.5)
            ax.set_xlabel("PIPS $D_m$ (mm)")
            ax.set_ylabel("radar $D_m$ (mm)")
    fig.savefig(os.path.join(out, f"dm_scatter_{iop}.png"), dpi=120)
    plt.close(fig)

    # time series of one probe: trajectory pairs with credible intervals
    pairs, ok, est, ref = data["traj"]
    probes = np.unique(pairs.probe.values)
    fig, axes = plt.subplots(
        len(probes),
        1,
        figsize=(10, 2.6 * len(probes)),
        sharex=True,
        constrained_layout=True,
    )
    for ax, p in zip(np.atleast_1d(axes), probes):
        sel = (pairs.probe.values == p) & ok
        t = pairs.gate_time.values[sel]
        o = np.argsort(t)
        q = est["bayes_learned"]["Q"]["DM"][:, sel][:, o]
        ax.fill_between(t[o], q[0], q[4], color="C0", alpha=0.15, label="95 %")
        ax.fill_between(t[o], q[1], q[3], color="C0", alpha=0.3, label="68 %")
        ax.plot(
            t[o],
            est["bayes_learned"]["DM"][sel][o],
            "C0-",
            label="Bayesian (learned prior)",
        )
        ax.plot(
            t[o], est["constrained"]["DM"][sel][o], "C1--", label="constrained gamma"
        )
        ax.plot(t[o], ref["DM"][sel][o], "k.", label="PIPS (aloft-equivalent)")
        ax.set_ylabel("$D_m$ (mm)")
        ax.set_title(p, fontsize=9)
    np.atleast_1d(axes)[0].legend(fontsize=7, ncol=5)
    fig.savefig(os.path.join(out, f"dm_timeseries_{iop}.png"), dpi=120)
    plt.close(fig)

    # radar minus PIPS-simulated Z_H and Z_DR
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for kind, c in (("naive", "C3"), ("traj", "C0")):
        pairs, ok, *_ = data[kind]
        for ax, v, b in zip(
            axes, ("DBZH", "ZDR"), (np.linspace(-15, 15, 41), np.linspace(-2, 2, 41))
        ):
            d = pairs[v].values[ok] - pairs[f"PIPS_{v}_SIM"].values[ok]
            ax.hist(
                d,
                b,
                histtype="step",
                color=c,
                lw=1.5,
                label=f"{kind}: sd {np.std(d):.2f}",
            )
            ax.set_xlabel(f"radar - PIPS {v}")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.savefig(os.path.join(out, f"radar_minus_pips_{iop}.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
