#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
NumPy reference implementation of the echo classifier of Gourley et al. (2007).

Same function, arguments and results as the compiled
``radarx.retrieve._gourley`` module; used when the extension is not built and
as the test oracle. The method is described in
:func:`radarx.retrieve.echo_mask` (``method="gourley2007"``).

Gourley, J. J., P. Tabary, and J. Parent du Chatelet, 2007: A fuzzy logic
algorithm for the separation of precipitating from nonprecipitating echoes
using polarimetric radar observations. J. Atmos. Oceanic Technol., 24 (8),
1439-1451, https://doi.org/10.1175/JTECH2035.1

Steps (equation and page numbers of the paper):

1. texture of ZDR and PhiDP: root-mean-square difference of the gate and its
   3 (ray) x 3 (gate) neighbours (Eq. 1, p. 1441), optionally corrected for
   its linear increase with range beyond 45 km (p. 1442);
2. membership of rhohv and of the two textures in each of the three classes
   (Fig. 3, looked up in tables), aggregation value of every class with the
   weights of Eqs. 3 and 4 (p. 1444);
3. the thresholds of Table 1 (p. 1446) set the aggregation value of the
   suppressed class to zero;
4. the class with the largest aggregation value wins (p. 1445);
5. despeckling over the 8 neighbours (p. 1446).
"""

import numpy as np

# class codes of the internal type map
NONE, PRECIP, CLUTTER, CLEAR, UNCLASSIFIED = 0, 1, 2, 3, 4
# classes of the output, as radarx.retrieve.qc.CLASSES
NO_ECHO, METEO, NONMETEO, SPECKLE = 0, 1, 2, 3


def _wrap180(d):
    return d - 360.0 * np.floor(d / 360.0 + 0.5)


def _neighbour_rays(links):
    from .qc import _neighbour_rays as impl

    return impl(links)


def _shifted(arr, rows, has, dg, fill):
    """Values of the gate ``dg`` away on the neighbouring rays ``rows``."""
    rows_arr = arr[np.where(has, rows, 0)]
    rows_arr = np.where(has[:, None], rows_arr, fill)
    out = np.full(arr.shape, fill, dtype=arr.dtype)
    if dg < 0:
        out[:, 1:] = rows_arr[:, :-1]
    elif dg > 0:
        out[:, :-1] = rows_arr[:, 1:]
    else:
        out = rows_arr
    return out


def texture(y, ok, links, wrap):
    """
    Root-mean-square difference to the neighbouring gates (Eq. 1).

    The sum runs over the gates of the 3 x 3 box that have data (the centre
    adds zero) and is divided by their number, so gates without data do not
    pull the texture towards zero; fewer than three neighbours with data
    leave it undefined (NaN).
    """
    nray, ng = y.shape
    prv, nxt = _neighbour_rays(links)
    own = np.arange(nray)
    yv = np.where(ok, y, 0.0)
    total = np.zeros((nray, ng))
    count = np.zeros((nray, ng))
    for rows in (prv, own, nxt):
        has = rows >= 0
        for dg in (-1, 0, 1):
            if rows is own and dg == 0:
                continue
            nb_ok = _shifted(ok, rows, has, dg, False)
            nb = _shifted(yv, rows, has, dg, 0.0)
            d = yv - nb
            if wrap:
                d = _wrap180(d)
            use = ok & nb_ok
            total = total + np.where(use, d * d, 0.0)
            count = count + use
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.sqrt(total / (count + 1.0))
    return np.where(ok & (count >= 3), t, np.nan)


def _valid(x, floor, echo):
    if x is None:
        return None
    with np.errstate(invalid="ignore"):
        return echo & np.isfinite(x) & (x > floor)


def _membership(x, xs, ys):
    """Linear interpolation of the table, clamped at its ends."""
    return np.interp(x, xs, ys)


def _classify_sweep(sw, floors, snr_min, mem_x, mem_y, weights, rules, slopes):
    z = sw["dbzh"]
    with np.errstate(invalid="ignore"):
        echo = np.isfinite(z) & (z > floors[0])
        if sw["snr"] is not None:
            echo &= sw["snr"] >= snr_min
    nray, ng = z.shape
    ok_zdr = _valid(sw["zdr"], floors[1], echo)
    ok_rho = _valid(sw["rhohv"], floors[2], echo)
    ok_phi = _valid(sw["phidp"], floors[3], echo)
    ok_vel = _valid(sw["vrad"], floors[4], echo)
    ok_sig = _valid(sw["zsigma"], -np.inf, echo)

    rho_min, phi_min, z_precip_min, phitex_max, sig_max, vel_max, z_clear_max = rules
    num = np.zeros((3, nray, ng))
    den = np.zeros((nray, ng))
    suppress_p = np.zeros((nray, ng), dtype=bool)

    def add(k, value, defined):
        nonlocal den
        w = weights[k]
        for c in range(3):
            m = _membership(np.where(defined, value, 0.0), mem_x[k], mem_y[k][c])
            num[c] = num[c] + np.where(defined, w * m, 0.0)
        den = den + np.where(defined, w, 0.0)

    rho_v = None
    if ok_rho is not None:
        rho_v = np.where(ok_rho, sw["rhohv"], 0.0)
        rho_v = np.where(rho_v > 1.0, 2.0 - rho_v, rho_v)
        add(0, rho_v, ok_rho)
        suppress_p |= ok_rho & (rho_v < rho_min)
    if ok_zdr is not None:
        t = texture(sw["zdr"], ok_zdr, sw["links"], False)
        t = _range_corrected(t, sw["rng_km"], slopes[0], slopes[2])
        add(1, np.where(np.isfinite(t), t, 0.0), np.isfinite(t))
    if ok_phi is not None:
        t = texture(sw["phidp"], ok_phi, sw["links"], True)
        t = _range_corrected(t, sw["rng_km"], slopes[1], slopes[2])
        defined = np.isfinite(t)
        add(2, np.where(defined, t, 0.0), defined)
        suppress_p |= defined & (t > phitex_max)
        with np.errstate(invalid="ignore"):
            suppress_p |= ok_phi & (sw["phidp"] < phi_min)

    zz = np.where(echo, z, 0.0)
    suppress_p |= echo & (zz < z_precip_min)
    suppress_g = np.zeros((nray, ng), dtype=bool)
    if ok_sig is not None:
        suppress_g |= ok_sig & (np.where(ok_sig, sw["zsigma"], 0.0) > sig_max)
    if ok_vel is not None:
        suppress_g |= ok_vel & (np.abs(np.where(ok_vel, sw["vrad"], 0.0)) > vel_max)
    suppress_c = echo & (zz > z_clear_max)

    with np.errstate(invalid="ignore", divide="ignore"):
        q = num / den
    q[0] = np.where(suppress_p, 0.0, q[0])
    q[1] = np.where(suppress_g, 0.0, q[1])
    q[2] = np.where(suppress_c, 0.0, q[2])
    total = q[0] + q[1] + q[2]
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(total > 0.0, q[0] / total, 0.0)
    defined = echo & (den > 0)
    score = np.where(defined, share, np.nan).astype(np.float32)
    typ = np.full((nray, ng), NONE, dtype=np.int8)
    typ[echo] = UNCLASSIFIED
    kind = np.where(
        q[0] > np.maximum(q[1], q[2]), PRECIP, np.where(q[1] > q[2], CLUTTER, CLEAR)
    )
    typ[defined] = kind[defined]
    return score, typ


def _range_corrected(t, rng_km, slope, start_km):
    """Texture minus its linear increase beyond ``start_km`` (Fig. 2)."""
    if slope == 0.0:
        return t
    extra = slope * np.maximum(rng_km - start_km, 0.0)
    return np.where(np.isfinite(t), np.maximum(t - extra[None, :], 0.0), t)


def _precip_neighbours(is_p, links):
    """Number of the 8 neighbouring gates of every gate that are precipitation."""
    nray, ng = is_p.shape
    prv, nxt = _neighbour_rays(links)
    own = np.arange(nray)
    count = np.zeros((nray, ng), dtype=np.int64)
    for rows in (prv, own, nxt):
        has = rows >= 0
        for dg in (-1, 0, 1):
            if rows is own and dg == 0:
                continue
            count += _shifted(is_p, rows, has, dg, False)
    return count


def _despeckle(typ, links):
    """Isolated precipitation to clear air, enclosed clear air to precipitation."""
    count = _precip_neighbours(typ == PRECIP, links)
    out = typ.copy()
    out[(typ == PRECIP) & (count < 3)] = CLEAR
    out[(typ == CLEAR) & (count > 6)] = PRECIP
    return out


def _to_class(typ, isolated):
    cls = np.zeros(typ.shape, dtype=np.int8)
    cls[typ == UNCLASSIFIED] = SPECKLE
    cls[(typ == CLUTTER) | (typ == CLEAR)] = NONMETEO
    cls[typ == PRECIP] = METEO
    cls[isolated] = SPECKLE
    return cls


def classify(
    dbzh,
    zdr,
    rhohv,
    phidp,
    vrad,
    zsigma,
    snr,
    links,
    rng_km,
    floors,
    snr_min,
    mem_x,
    mem_y,
    weights,
    rules,
    slopes,
    despeckle,
    n_threads=0,
):
    """Score (float32) and class (int8) of every gate of every sweep."""
    del n_threads  # the reference is single-threaded
    scores, classes = [], []
    for i in range(len(dbzh)):
        sw = {
            "dbzh": dbzh[i],
            "zdr": zdr[i],
            "rhohv": rhohv[i],
            "phidp": phidp[i],
            "vrad": vrad[i],
            "zsigma": zsigma[i],
            "snr": snr[i],
            "links": links[i],
            "rng_km": np.asarray(rng_km[i], dtype=np.float64),
        }
        score, typ = _classify_sweep(
            sw, floors, snr_min, mem_x, mem_y, weights, rules, slopes
        )
        isolated = np.zeros(typ.shape, dtype=bool)
        if despeckle:
            new = _despeckle(typ, links[i])
            isolated = (typ == PRECIP) & (new == CLEAR)
            typ = new
        scores.append(score)
        classes.append(_to_class(typ, isolated))
    return scores, classes
