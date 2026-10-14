#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Time the compiled kernels against their NumPy twins on synthetic radar data.

Each benchmark calls one public function with ``engine="numpy"`` and with
``engine="compiled"`` on one thread, on a synthetic sweep or volume of a realistic size
(720 rays of 1000 gates), and reports the median wall time of several repeats and the
speedup. The speedup, not the absolute time, is what is checked, so the result does not
depend on the speed of the runner: the script fails when a kernel is slower than its
NumPy twin (speedup below 0.8, the allowance for timing noise) or slower than ``--min-speedup`` times the speedup recorded
in ``ci/benchmark_baseline.json`` (a 3-fold loss by default). The results are written as
Markdown (also to ``$GITHUB_STEP_SUMMARY``) and as JSON.

Usage: python ci/benchmark_kernels.py [--repeats 5] [--json bench.json] [--update-baseline]
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

import radarx as rx

NRAY, NGATE = 720, 1000
BASELINE = Path(__file__).with_name("benchmark_baseline.json")


def sweep(seed=0, elevation=0.5, nyquist=30.0):
    """A synthetic rainy PPI sweep with every field the benchmarks need."""
    rng = np.random.default_rng(seed)
    azimuth = (np.arange(NRAY) + 0.5) * 360.0 / NRAY
    r = 250.0 + 250.0 * np.arange(NGATE)
    a, g = np.meshgrid(np.radians(azimuth), r / 1e3, indexing="ij")
    # a rain band, a uniform wind that wraps beyond the Nyquist velocity, a rising PHIDP
    echo = np.exp(-(((g - 120.0) / 80.0) ** 2)) * (1.0 + 0.3 * np.sin(3 * a))
    dbz = 8.0 + 40.0 * echo + rng.normal(0, 1.0, echo.shape)
    zdr = 0.3 + 2.0 * echo + rng.normal(0, 0.15, echo.shape)
    rhohv = np.clip(0.99 - 0.02 * rng.random(echo.shape) - 0.4 * (echo < 0.05), 0, 1)
    phidp = (
        10.0
        + np.cumsum(0.04 * np.maximum(echo, 0) * 40.0 * 0.5, axis=1)
        + rng.normal(0, 1.0, echo.shape)
    )
    wind = 45.0 * np.cos(a - 1.0) * (0.3 + echo)
    vrad = ((wind + nyquist) % (2 * nyquist)) - nyquist
    dims = ("azimuth", "range")
    ds = xr.Dataset(
        {
            "DBZH": (dims, dbz.astype("f4")),
            "ZDR": (dims, zdr.astype("f4")),
            "RHOHV": (dims, rhohv.astype("f4")),
            "PHIDP": (dims, phidp.astype("f4")),
            "VRADH": (dims, vrad.astype("f4")),
        },
        coords={
            "azimuth": ("azimuth", azimuth),
            "range": ("range", r),
            "elevation": ("azimuth", np.full(NRAY, elevation)),
            "nyquist_velocity": ((), nyquist),
            "altitude": ((), 100.0),
            "latitude": ((), 33.0),
            "longitude": ((), -88.0),
        },
    )
    for name in ds.data_vars:
        ds[name].attrs["units"] = {
            "DBZH": "dBZ",
            "ZDR": "dB",
            "RHOHV": "1",
            "PHIDP": "degrees",
            "VRADH": "m s-1",
        }[name]
    return ds


def volume():
    """A DataTree of four sweeps for the gridding benchmarks."""
    sweeps = {
        f"sweep_{i}": sweep(i, el).drop_vars(["altitude", "latitude", "longitude"])
        for i, el in enumerate((0.5, 1.5, 2.5, 3.5))
    }
    root = xr.Dataset(coords={"latitude": 33.0, "longitude": -88.0, "altitude": 100.0})
    return xr.DataTree.from_dict({"/": root, **sweeps})


def benchmarks():
    ds, tree = sweep(), volume()
    x = np.arange(-150e3, 150e3 + 1, 5000.0)
    z = np.arange(500.0, 8e3, 1000.0)
    one = dict(n_threads=1)
    return {
        "dealias_velocity (region)": lambda e: rx.retrieve.dealias_velocity(
            ds, "VRADH", nyquist_velocity=30.0, engine=e, **one
        ),
        "estimate_kdp (hubbert)": lambda e: rx.retrieve.estimate_kdp(
            ds, engine=e, **one
        ),
        "echo_mask": lambda e: rx.retrieve.echo_mask(ds, engine=e, **one),
        "llsd (azimuthal shear)": lambda e: rx.retrieve.llsd(
            ds, "VRADH", engine=e, **one
        ),
        "correct_attenuation (zphi)": lambda e: rx.retrieve.correct_attenuation(
            ds, band="S", engine=e, **one
        ),
        "dsd (constrained gamma)": lambda e: rx.retrieve.dsd(
            ds, dbz="DBZH", zdr="ZDR", band="S", engine=e, **one
        ),
        "grid_cones": lambda e: rx.grid.grid_cones(
            tree, "DBZH", x, x, z, engine=e, **one
        ),
        "grid_gates (cressman)": lambda e: rx.grid.grid_gates(
            tree, ["DBZH"], x, x, z, roi=3000.0, engine=e, **one
        ),
    }


def timeit(fn, repeats):
    fn()  # warm up: imports, tables, caches
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn()
        if hasattr(out, "compute"):
            out = out.compute()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--json", default="benchmark.json")
    p.add_argument(
        "--min-speedup",
        type=float,
        default=0.8,
        help="fail below this speedup (timing noise allowance)",
    )
    p.add_argument(
        "--regression",
        type=float,
        default=3.0,
        help="fail when the speedup falls below baseline / this",
    )
    p.add_argument("--update-baseline", action="store_true")
    a = p.parse_args()
    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    rows, failures = [], []
    for name, call in benchmarks().items():
        try:
            t_np = timeit(lambda: call("numpy"), a.repeats)
            t_c = timeit(lambda: call("compiled"), a.repeats)
        except ImportError as err:
            failures.append(f"{name}: compiled kernel missing ({err})")
            continue
        speed = t_np / t_c
        base = baseline.get(name)
        rows.append(
            dict(name=name, numpy_s=t_np, compiled_s=t_c, speedup=speed, baseline=base)
        )
        if speed < a.min_speedup:
            failures.append(
                f"{name}: compiled {speed:.2f}x the speed of NumPy (limit {a.min_speedup})"
            )
        elif base and speed < base / a.regression:
            failures.append(
                f"{name}: speedup {speed:.1f} fell from the baseline {base:.1f} by more than {a.regression}x"
            )
    lines = [
        "| function | NumPy (s) | compiled, 1 thread (s) | speedup | baseline |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        b = "" if r["baseline"] is None else f"{r['baseline']:.1f}"
        lines.append(
            f"| {r['name']} | {r['numpy_s']:.3f} | {r['compiled_s']:.3f} | {r['speedup']:.1f} | {b} |"
        )
    table = "\n".join(lines)
    print(table)
    Path(a.json).write_text(json.dumps(rows, indent=1))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
            fh.write(f"### Kernel benchmark ({NRAY} rays x {NGATE} gates)\n\n{table}\n")
    if a.update_baseline:
        BASELINE.write_text(
            json.dumps({r["name"]: round(r["speedup"], 1) for r in rows}, indent=1)
            + "\n"
        )
        print("baseline written to", BASELINE)
    if failures:
        sys.exit("benchmark failed:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()
