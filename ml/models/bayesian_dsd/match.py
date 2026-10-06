"""
Radar-disdrometer pairs: naive collocation and trajectory matching.

A stand-in for the rain trajectory model of radarx issue #140 (to be
replaced by it once merged): drops fall from the radar beam (height ``h``
above the disdrometer) at the terminal speed of Atlas et al. (1973),
corrected for air density as (rho0/rho)^0.4 (Foote and du Toit 1969), and
drift with the layer-mean horizontal wind ``u`` (ERA5). The echo pattern is
frozen and moves with the storm motion ``c`` (radarx.retrieve.estimate_motion).

Naive pair
    the radar gates over the disdrometer and the 60-s surface spectrum
    centred on the time the beam passed over it.
Trajectory pair
    the radar gates at the source point ``x_P - u tau_ref`` of reference
    drops (D_ref = 2 mm, fall time ``tau_ref``) and the "aloft-equivalent"
    spectrum: every size bin D is taken from the disdrometer at the time its
    drops from that point of the (moving) pattern reach it,

        t(D) = t_gate + tau_D - u.c_hat (tau_D - tau_ref) / |c|,

    which undoes size sorting by the fall-speed differences and the wind
    drift along the storm motion. The cross-track drift
    ``|u_perp| |tau_D - tau_ref|`` cannot be undone with one disdrometer and
    is reported. Concentrations are scaled by (rho/rho0)^0.4 (constant flux).
    Evaporation and break-up/coalescence below the beam are neglected.
"""

from __future__ import annotations

import numpy as np
import pips as pipsmod
import xarray as xr

RHO0 = 1.204  # kg m-3, sea-level reference of the Atlas et al. fall speed
D_REF = 2.0  # mm


def fall_speed(d, rho=RHO0):
    v = 9.65 - 10.3 * np.exp(-0.6 * np.asarray(d, float))
    return np.maximum(v, 0.1) * (RHO0 / rho) ** 0.4


def layer_wind(profile, h, alt):
    """Mean wind (u, v towards, m/s) between the ground and h above it."""
    if profile is None:
        return None
    z = profile.height.values - alt
    zz = np.linspace(0.0, max(h, 10.0), 50)
    u = np.interp(zz, z, profile.u.values)
    v = np.interp(zz, z, profile.v.values)
    return np.array([u.mean(), v.mean()])


def layer_density(profile, h, alt):
    if profile is None or "pressure" not in profile:
        return RHO0 * np.exp(-(alt + h / 2) / 8500.0)
    z = profile.height.values - alt
    p = np.interp(h / 2, z, profile.pressure.values)
    t = np.interp(h / 2, z, profile.temperature.values)
    return p / (287.05 * t)


def radar_sample(pts, x, y, radius=1000.0, min_gates=3):
    """Mean radar variables of the gates within ``radius`` of (x, y)."""
    d2 = (pts.x.values - x) ** 2 + (pts.y.values - y) ** 2
    sel = d2 <= radius**2
    z = pts.DBZH.values[sel]
    ok = np.isfinite(z) & np.isfinite(pts.ZDR.values[sel])
    if ok.sum() < min_gates:
        return None
    zh = 10 ** (z[ok] / 10)
    zv = zh / 10 ** (pts.ZDR.values[sel][ok] / 10)
    kdp = pts.KDP.values[sel]
    return {
        "DBZH": 10 * np.log10(zh.mean()),
        "ZDR": 10 * np.log10(zh.mean() / zv.mean()),
        "KDP": np.nanmean(kdp) if np.isfinite(kdp).any() else np.nan,
        "RHOHV": np.nanmean(pts.RHOHV.values[sel]),
        "height": float(np.mean(pts.z.values[sel])),
        "gate_time": pts.gate_time.values[sel][0],
        "n_gates": int(ok.sum()),
        "elevation": float(pts.attrs.get("elevation", np.nan)),
    }


def _window(raw, t, half=30.0):
    """10-s spectra within +-half s of t (times as datetime64)."""
    dt = (raw.time.values - t) / np.timedelta64(1, "s")
    return np.abs(dt) <= half


def pairs(raw, pts_list, site_xy, profile, motion, half=30.0):
    """
    Naive and trajectory pairs of one disdrometer with all volumes.

    Returns
    -------
    (xarray.Dataset, xarray.Dataset)
        Naive and trajectory pairs along ``pair``: radar variables and the
        matched spectra ``ND`` (pair, diameter).
    """
    xp, yp = site_xy
    alt = raw.attrs["altitude"]
    d = raw.diameter.values
    c = np.asarray(motion, float)
    cn = np.linalg.norm(c)
    chat = c / cn
    naive, traj = [], []
    for pts in pts_list:
        over = radar_sample(pts, xp, yp)
        if over is None:
            continue
        h = over["height"] - alt  # beam height (z is above sea level) over the site
        rho = layer_density(profile, h, alt)
        tau = h / fall_speed(d, rho)
        tau_ref = h / fall_speed(D_REF, rho)
        u = layer_wind(profile, h, alt)
        if u is None:
            u = np.array([np.nanmean(raw.u), np.nanmean(raw.v)])
        # naive: surface spectrum centred on the overpass
        w = _window(raw, over["gate_time"], half)
        if w.sum() >= 4:
            naive.append(_pair(raw.ND.values[w].mean(0), over, h, 0.0, 0.0))
        # trajectory: radar at the source point, spectrum bin by bin
        xs, ys = xp - u[0] * tau_ref, yp - u[1] * tau_ref
        src = radar_sample(pts, xs, ys)
        if src is None:
            continue
        shift = tau - (u @ chat) * (tau - tau_ref) / cn
        nd = np.full(d.size, np.nan)
        for k in range(d.size):
            wk = _window(
                raw, src["gate_time"] + np.timedelta64(int(shift[k] * 1e3), "ms"), half
            )
            if wk.sum() >= 4:
                nd[k] = raw.ND.values[wk, k].mean()
        if not np.isfinite(nd[d <= 6]).all():
            continue
        nd = np.nan_to_num(nd) * (rho / RHO0) ** 0.4
        cross = abs(u[0] * chat[1] - u[1] * chat[0])
        drift = cross * np.abs(tau - tau_ref)
        # cross-track drift of 1-mm and 4-mm drops relative to 2-mm drops
        dr = float(np.interp(1.0, d, drift) + np.interp(4.0, d, drift))
        traj.append(_pair(nd, src, h, float(tau_ref), dr))
    return _stack(naive, d, raw), _stack(traj, d, raw)


def aloft_spectra(raw, h, u, motion, rho=None, step=60.0, half=30.0):
    """
    Aloft-equivalent spectra at height ``h`` over the disdrometer, every
    ``step`` s, without radar data (for learning priors): the same bin-wise
    time shifts as the trajectory pairs.
    """
    d = raw.diameter.values
    rho = (
        RHO0 * np.exp(-(raw.attrs["altitude"] + h / 2) / 8500.0) if rho is None else rho
    )
    tau = h / fall_speed(d, rho)
    tau_ref = h / fall_speed(D_REF, rho)
    if motion is None:  # no storm motion: fall-speed sorting only
        shift = tau - tau_ref
    else:
        c = np.asarray(motion, float)
        cn = np.linalg.norm(c)
        shift = (tau - tau_ref) * (1.0 - (np.asarray(u, float) @ (c / cn)) / cn)
    times = np.arange(
        raw.time.values[0] + np.timedelta64(300, "s"),
        raw.time.values[-1] - np.timedelta64(300, "s"),
        np.timedelta64(int(step), "s"),
    )
    tt = (raw.time.values - times[0]) / np.timedelta64(1, "s")
    nds = []
    for t in (times - times[0]) / np.timedelta64(1, "s"):
        nd = np.full(d.size, np.nan)
        for k in range(d.size):
            w = np.abs(tt - (t + shift[k])) <= half
            if w.sum() >= 4:
                nd[k] = raw.ND.values[w, k].mean()
        nds.append(nd)
    nd = np.nan_to_num(np.array(nds)) * (rho / RHO0) ** 0.4
    return xr.DataArray(
        nd,
        dims=("time", "diameter"),
        coords={
            "time": times,
            "diameter": d,
            "bin_width": ("diameter", raw.bin_width.values),
        },
    )


def _pair(nd, sample, h, tau_ref, drift):
    out = dict(sample)
    out.update({"ND": nd, "beam_height": h, "tau_ref": tau_ref, "cross_drift": drift})
    return out


def _stack(rows, d, raw):
    if not rows:
        return None
    keys = [k for k in rows[0] if k != "ND"]
    ds = xr.Dataset(
        {k: ("pair", np.array([r[k] for r in rows])) for k in keys}
        | {"ND": (("pair", "diameter"), np.array([r["ND"] for r in rows]))},
        coords={"diameter": d, "bin_width": ("diameter", raw.bin_width.values)},
        attrs=dict(raw.attrs),
    )
    return ds


def with_parameters(ds, band="S"):
    par = pipsmod.parameters(ds.ND, band=band)
    return xr.merge([ds, par.rename({k: "PIPS_" + k for k in par.data_vars})])
