"""
Read PERiLS PIPS (Portable In situ Precipitation Station) Parsivel2 files.

The ``parsivel_combined_*_10s.nc`` files hold 10-s drop spectra
``ND_roqc`` (m-3 mm-1, quality-controlled for strong wind, splashing, margin
fallers and non-rain) on the 32 Parsivel size classes, and the station
meteorology. :func:`load` returns the spectra on 10 s and averaged over
``window`` seconds, with the station location, wind and thermodynamics.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import xarray as xr

from radarx.retrieve import fit_gamma_moments, radar_from_dsd

ROOT = os.path.expanduser("~/Downloads/MULTIDOPPLER/PIPS_data")
DEPLOYMENTS = {
    "IOP1": "IOP1_032222",
    "IOP2": "IOP2_033022",
    "IOP3": "IOP3_040522",
    "IN": "030622_IN_test",
}


def files(iop):
    folder = os.path.join(ROOT, DEPLOYMENTS[iop], "netcdf")
    return sorted(glob.glob(os.path.join(folder, "parsivel_combined_*_10s.nc")))


def _location(ds):
    lat, lon, alt = (float(v) for v in ds.attrs["location"].strip("()").split(","))
    if not 0 < alt < 1000:  # missing station altitude: typical of the deployments
        alt = 100.0
    return lat, lon, alt


def load(path, window=60.0):
    """
    One PIPS deployment: 10-s spectra and ``window``-s averages.

    Returns
    -------
    (xarray.Dataset, xarray.Dataset)
        10-s and averaged data: ``ND`` (time, diameter) with the
        ``bin_width`` coordinate, wind components ``u``, ``v`` (m/s, towards),
        ``T`` (degC), ``p`` (hPa), the fraction of valid 10-s records
        ``valid``; attrs ``probe``, ``latitude``, ``longitude``,
        ``altitude``.
    """
    ds = xr.open_dataset(path)
    nd_name = "ND_roqc" if "ND_roqc" in ds else "ND_qc"
    d = ds["diameter"].values
    width = ds["max_diameter"].values - ds["min_diameter"].values
    nd = ds[nd_name].values.astype(float)
    valid = (
        np.isfinite(ds["pcount"].values)
        if "pcount" in ds
        else np.ones(len(ds.time), bool)
    )
    nd = np.where(np.isfinite(nd), nd, 0.0)
    # wind: speed and direction it blows from (meteorological)
    spd = ds["windspd"].values.astype(float)
    wdir = np.deg2rad(ds["winddirabs"].values.astype(float))
    u = -spd * np.sin(wdir)
    v = -spd * np.cos(wdir)
    lat, lon, alt = _location(ds)
    attrs = {
        "probe": ds.attrs.get("probe_name", os.path.basename(path)),
        "deployment": ds.attrs.get("deployment_name", ""),
        "latitude": lat,
        "longitude": lon,
        "altitude": alt,
    }
    raw = xr.Dataset(
        {
            "ND": (("time", "diameter"), nd),
            "u": ("time", u),
            "v": ("time", v),
            "T": ("time", ds["fasttemp"].values.astype(float)),
            "p": ("time", ds["pressure"].values.astype(float)),
            "valid": ("time", valid.astype(float)),
        },
        coords={
            "time": ds.time.values,
            "diameter": d,
            "bin_width": ("diameter", width),
        },
        attrs=attrs,
    )
    raw = raw.sortby("time")
    raw = raw.isel(time=np.unique(raw.time.values, return_index=True)[1])
    avg = raw.resample(time=f"{int(window)}s", label="left").mean()
    avg = avg.assign_coords(time=avg.time + np.timedelta64(int(window * 500), "ms"))
    avg = avg.assign_coords(bin_width=raw.bin_width)
    avg.attrs = attrs
    return raw, avg


def parameters(nd, band="S", temperature=20.0):
    """
    DSD parameters of spectra: moments (Dm, Nw, R, LWC, simulated radar
    variables) and the 2-4-6 moment gamma fit (mu).
    """
    sim = radar_from_dsd(nd, band=band, temperature=temperature)
    fit = fit_gamma_moments(nd)
    m0 = (nd * nd.bin_width).sum("diameter")
    out = xr.Dataset(
        {
            "DM": sim.DM,
            "NW": sim.NW,
            "MU": fit.MU,
            "RAIN_RATE": sim.RAIN_RATE,
            "LWC": sim.LWC,
            "NT": m0,
            "DBZH_SIM": sim.DBZH,
            "ZDR_SIM": sim.ZDR,
            "KDP_SIM": sim.KDP,
            "AH_SIM": sim.AH,
        }
    )
    return out


def usable(params, rain_min=0.5, nt_min=50.0):
    """Rain minutes with enough drops for a stable moment fit."""
    return (
        (params.RAIN_RATE >= rain_min)
        & (params.NT >= nt_min)
        & np.isfinite(params.MU)
        & np.isfinite(params.DM)
    )
