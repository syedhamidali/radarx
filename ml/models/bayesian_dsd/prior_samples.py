"""
Gamma DSD parameters of aloft-equivalent PIPS spectra, for learning priors.

    python prior_samples.py --out samples.csv [--era5-dir DIR] [--motion IOP=u,v ...]

For every deployment the spectra are shifted bin by bin to the height of the
lowest (0.5 deg) beam of the nearest WSR-88D over the probe (see
:func:`match.aloft_spectra`), averaged over 60 s, and fitted (Dm and Nw
from the moments, mu from the 2-4-6 moment fit). Minutes with R < 0.5 mm/h
or fewer than 50 drops m-3 are skipped. The storm motion of each IOP is
given (from build_pairs.py) or taken as the ERA5 0-6 km mean wind; without
either only the fall-speed sorting is undone.
"""

from __future__ import annotations

import argparse
import os

import match
import numpy as np
import pips
import xarray as xr

# WSR-88D sites nearest to the deployments: latitude, longitude, antenna altitude (m)
RADARS = {
    "IOP1": ("KGWX", 33.8967, -88.3289, 179.0),
    "IOP2": ("KGWX", 33.8967, -88.3289, 179.0),
    "IOP3": ("KMXX", 32.5367, -85.7897, 152.0),
}


def beam_height(radar, lat, lon, alt, elevation=0.5):
    """Height of the beam centre over a site (4/3 effective earth radius)."""
    from radar import project

    _, rlat, rlon, ralt = radar
    x, y = project(rlat, rlon, lat, lon)
    r = float(np.hypot(x, y))
    re = 4.0 / 3.0 * 6371e3
    th = np.deg2rad(elevation)
    z = np.sqrt(r * r + re * re + 2 * r * re * np.sin(th)) - re + ralt
    return z - alt, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--era5-dir", default=None)
    ap.add_argument("--motion", nargs="*", default=[])
    ap.add_argument("--iops", nargs="*", default=["IOP1", "IOP2", "IOP3"])
    a = ap.parse_args()
    motions = {
        m.split("=")[0]: [float(v) for v in m.split("=")[1].split(",")]
        for m in a.motion
    }
    rows = []
    for iop in a.iops:
        prof = None
        if a.era5_dir and os.path.exists(os.path.join(a.era5_dir, f"{iop}.nc")):
            prof = xr.open_dataset(os.path.join(a.era5_dir, f"{iop}.nc"))
        for path in pips.files(iop):
            raw, _ = pips.load(path)
            att = raw.attrs
            h, rng = beam_height(
                RADARS[iop], att["latitude"], att["longitude"], att["altitude"]
            )
            u = match.layer_wind(prof, h, att["altitude"])
            if u is None:
                u = np.array([np.nanmean(raw.u), np.nanmean(raw.v)])
            if iop in motions:
                c = np.array(motions[iop])
            elif prof is not None:
                z = prof.height.values - att["altitude"]
                zz = np.linspace(0, 6000, 61)
                c = np.array(
                    [np.interp(zz, z, prof.u).mean(), np.interp(zz, z, prof.v).mean()]
                )
            else:
                c = None  # unknown: undo the fall-speed sorting only
            nd = match.aloft_spectra(raw, h, u, c)
            par = pips.parameters(nd)
            ok = pips.usable(par).values
            surf = pips.parameters(
                raw.ND.resample(time="60s")
                .mean()
                .assign_coords(bin_width=raw.bin_width)
            )
            sok = pips.usable(surf).values
            for kind, p, m in (("aloft", par, ok), ("surface", surf, sok)):
                for i in np.flatnonzero(m):
                    rows.append(
                        (
                            iop,
                            att["probe"],
                            kind,
                            float(np.log10(p.NW[i])),
                            float(p.DM[i]),
                            float(p.MU[i]),
                            float(p.RAIN_RATE[i]),
                        )
                    )
            print(
                iop,
                att["probe"],
                f"h={h:.0f} m r={rng / 1e3:.0f} km",
                "u",
                u.round(1),
                "c",
                None if c is None else np.round(c, 1),
                ok.sum(),
                sok.sum(),
                flush=True,
            )
    with open(a.out, "w") as f:
        f.write("iop,probe,kind,log10_nw,dm,mu,rain_rate\n")
        for r in rows:
            f.write(",".join([r[0], r[1], r[2]] + [f"{v:.5g}" for v in r[3:]]) + "\n")


if __name__ == "__main__":
    main()
