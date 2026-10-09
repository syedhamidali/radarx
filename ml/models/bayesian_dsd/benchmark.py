"""
Timing of the Bayesian DSD retrieval on a NEXRAD sweep.

    python benchmark.py VOLUME [--prior generic]

Rain gates of the lowest sweep (rhohv >= 0.97, Z_H >= 5 dBZ) with K_DP from
radarx.retrieve.estimate_kdp; compiled kernel on 1 thread and on all cores,
and the NumPy engine on a subset; deterministic retrievals for reference.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import radar

from radarx.retrieve import dsd, dsd_bayesian


def best(f, n=3):
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    return min(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("volume")
    ap.add_argument("--prior", default="generic")
    a = ap.parse_args()
    ds = radar.read_sweep(a.volume)
    rain = (ds.RHOHV >= 0.97) & (ds.DBZH >= 5)
    sub = ds[["DBZH", "ZDR", "KDP"]].where(rain)
    n = int(rain.sum())
    print(f"{n} rain gates of {ds.DBZH.size}; {os.cpu_count()} cores")
    kw = {"band": "S", "kdp": "KDP", "prior": a.prior}
    t1 = best(lambda: dsd_bayesian(sub, n_threads=1, **kw), 1)
    ta = best(lambda: dsd_bayesian(sub, **kw))
    print(f"compiled, 1 thread : {t1:.2f} s ({1e6 * t1 / n:.1f} us/gate)")
    print(
        f"compiled, all cores: {ta:.2f} s ({1e6 * ta / n:.1f} us/gate), speed-up {t1 / ta:.1f}"
    )
    small = sub.isel(azimuth=slice(0, 40))
    m = int(rain.isel(azimuth=slice(0, 40)).sum())
    tn = best(lambda: dsd_bayesian(small, engine="numpy", **kw), 1)
    tc = best(lambda: dsd_bayesian(small, n_threads=1, **kw), 1)
    print(
        f"numpy on {m} gates: {tn:.2f} s ({1e6 * tn / m:.0f} us/gate), compiled 1 thread {tc:.3f} s: x{tn / tc:.0f}"
    )
    td = best(lambda: dsd(sub, "constrained", kdp="KDP", band="S"))
    print(f"deterministic constrained gamma: {td:.3f} s")
    _ = np


if __name__ == "__main__":
    main()
