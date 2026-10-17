#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tables and options of the echo classifier of Gourley et al. (2007).

The membership functions of the paper are density estimates of C-band data of
the Trappes radar that the paper shows in its Fig. 3 (p. 1444) and does not
tabulate. The tables below were read by eye from the figure at the points
listed and are linearly interpolated; they reproduce the modes and the one
membership example given in the text (see :func:`radarx.retrieve.echo_mask`).
They are an approximation of the figure, not the data behind it.

Gourley, J. J., P. Tabary, and J. Parent du Chatelet, 2007: A fuzzy logic
algorithm for the separation of precipitating from nonprecipitating echoes
using polarimetric radar observations. J. Atmos. Oceanic Technol., 24 (8),
1439-1451, https://doi.org/10.1175/JTECH2035.1
"""

import numpy as np

#: Variables of the classifier in the order of the kernel.
VARIABLES = ("rhohv", "zdr_texture", "phidp_texture")

#: Classes of the membership tables.
CLASS_NAMES = ("precipitation", "ground_clutter", "clear_air")

_RHOHV_X = [0, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97]
_RHOHV_X += [0.98, 0.99, 1.0]
_ZDR_X = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 5.2, 6, 7, 8, 10, 12, 15]
_PHI_X = [0, 2, 4, 6, 8, 10, 15, 20, 22, 30, 35, 38, 50, 60, 80, 100, 120, 125, 140]
_PHI_X += [160, 180, 200, 225, 250]

#: Default membership tables ``(x, precipitation, ground_clutter, clear_air)``
#: of rhohv (-), the texture of ZDR (dB) and the texture of PhiDP (degrees),
#: read from Fig. 3 of Gourley et al. (2007). The values are the density of
#: the figure (areas under the curves one, bins of the paper).
DEFAULT_MEMBERSHIPS = {
    "rhohv": (
        _RHOHV_X,
        # precipitation: 0.149 at 0.97 (p. 1445), mode 0.97-0.98 (p. 1443)
        [0, 0, 0, 0, 0, 0, 0, 0, 0.0005, 0.001, 0.005, 0.012, 0.07, 0.149]
        + [0.175, 0.172, 0.145],
        # ground clutter: 0.030 at 0.97, mode 0.92
        [0, 0, 0.0015, 0.003, 0.0045, 0.0055, 0.0075, 0.011, 0.0195, 0.0255, 0.0305]
        + [0.0312, 0.0305, 0.030, 0.0295, 0.0275, 0.025],
        # clear air: 0.007 at 0.97, mode 0.42
        [0, 0, 0.003, 0.0125, 0.0172, 0.0168, 0.0148, 0.0135, 0.0125, 0.011, 0.0095]
        + [0.0090, 0.0078, 0.007, 0.0065, 0.0058, 0.005],
    ),
    "zdr_texture": (
        _ZDR_X,
        # precipitation: mode 0.4-0.5 dB
        [0.057, 0.15, 0.222, 0.19, 0.13, 0.06, 0.03, 0.014, 0.007, 0.002, 0.0006]
        + [0.0005, 0.0002, 0.0001, 0, 0, 0, 0],
        # ground clutter: mode 5.2 dB
        [0.0015, 0.0018, 0.002, 0.003, 0.004, 0.007, 0.0125, 0.019, 0.027, 0.037]
        + [0.0408, 0.0410, 0.0405, 0.037, 0.031, 0.017, 0.0065, 0.0005],
        # clear air: mode 2.5 dB
        [0.004, 0.01, 0.017, 0.025, 0.033, 0.05, 0.062, 0.067, 0.067, 0.045, 0.024]
        + [0.022, 0.013, 0.007, 0.004, 0.0012, 0.0003, 0.0],
    ),
    "phidp_texture": (
        _PHI_X,
        # precipitation: mode 4 deg, 0.0001 at 50 deg
        [0.18, 0.22, 0.24, 0.20, 0.12, 0.075, 0.027, 0.0125, 0.01, 0.0035, 0.002]
        + [0.001, 0.0001, 0.00003, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # ground clutter: mode 125 deg
        [0.0007, 0.0009, 0.0011, 0.0013, 0.0015, 0.0016, 0.0021, 0.0026, 0.0028]
        + [0.0034, 0.0038, 0.0042, 0.0058, 0.0070, 0.0097, 0.0123, 0.0134, 0.0135]
        + [0.0132, 0.0118, 0.0095, 0.0068, 0.0045, 0.0028],
        # clear air: mode 35 deg
        [0.0068, 0.0078, 0.0090, 0.0100, 0.0108, 0.0115, 0.0130, 0.0138, 0.0140]
        + [0.0147, 0.0148, 0.0146, 0.0132, 0.0120, 0.0098, 0.0088, 0.0081, 0.0079]
        + [0.0072, 0.0062, 0.0053, 0.0043, 0.0030, 0.0019],
    ),
}

#: Overlap areas of the precipitation and non-precipitation densities
#: (p. 1444): rhohv, texture of ZDR, texture of PhiDP.
OVERLAP_AREAS = {"rhohv": 0.243, "zdr_texture": 0.203, "phidp_texture": 0.081}


def _weights_from_areas(areas):
    inverse = {k: 1.0 / a for k, a in areas.items()}
    total = sum(inverse.values())
    return {k: v / total for k, v in inverse.items()}


#: Weights of Eq. 4 (p. 1444): the inverse overlap areas, normalised to sum
#: one (the normalisation has no effect on the class); 1 : 1.20 : 3.00.
DEFAULT_WEIGHTS = _weights_from_areas(OVERLAP_AREAS)

_INF = np.inf

#: Thresholds of Table 1 (p. 1446). A class is suppressed (its aggregation
#: value set to zero) where the variable passes the threshold:
#: ``rhohv_min``, ``phidp_min`` (degrees), ``dbz_min`` (dBZ) and
#: ``phidp_texture_max`` (degrees) for precipitation; ``zsigma_max`` (dB,
#: pulse-to-pulse variability of Z) and ``vrad_max`` (m/s, absolute radial
#: velocity) for ground clutter; ``dbz_max`` (dBZ) for clear air.
DEFAULT_RULES = {
    "rhohv_min": 0.7,
    "phidp_min": -40.0,
    "dbz_min": 5.0,
    "phidp_texture_max": 100.0,
    "zsigma_max": 5.0,
    "vrad_max": 5.0,
    "dbz_max": 30.0,
}

#: The value of a rule that never suppresses its class (a rule set to ``None``).
_OFF = {
    "rhohv_min": -_INF,
    "phidp_min": -_INF,
    "dbz_min": -_INF,
    "phidp_texture_max": _INF,
    "zsigma_max": _INF,
    "vrad_max": _INF,
    "dbz_max": _INF,
}

#: Start of the range correction of the textures (km, p. 1442).
RANGE_CORRECTION_START_KM = 45.0


def memberships(given):
    """Tables (x list, y list of (3, n) arrays) of the three variables."""
    given = dict(given or {})
    unknown = set(given) - set(VARIABLES)
    if unknown:
        raise ValueError(f"unknown memberships keys {sorted(unknown)}; use {VARIABLES}")
    xs, ys = [], []
    for name in VARIABLES:
        table = given.get(name, DEFAULT_MEMBERSHIPS[name])
        try:
            x, *curves = (np.asarray(a, dtype=np.float64) for a in table)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"memberships[{name!r}] must be (x, precipitation, ground_clutter, "
                "clear_air) with numeric sequences"
            ) from err
        if len(curves) != 3 or x.ndim != 1 or x.size < 2:
            raise ValueError(
                f"memberships[{name!r}] must be (x, precipitation, ground_clutter, "
                "clear_air), x with at least two values"
            )
        if any(c.shape != x.shape for c in curves):
            raise ValueError(f"memberships[{name!r}]: all curves need the length of x")
        if not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
            raise ValueError(f"memberships[{name!r}]: x must be finite and increasing")
        if any(not np.all(np.isfinite(c)) or np.any(c < 0) for c in curves):
            raise ValueError(f"memberships[{name!r}]: values must be finite and >= 0")
        xs.append(np.ascontiguousarray(x))
        ys.append(np.ascontiguousarray(np.vstack(curves)))
    return xs, ys


def weights(given):
    """Weights of rhohv and the two textures in kernel order."""
    wts = dict(DEFAULT_WEIGHTS)
    given = given or {}
    unknown = set(given) - set(VARIABLES)
    if unknown:
        raise ValueError(f"unknown weights keys {sorted(unknown)}; use {VARIABLES}")
    wts.update(given)
    out = np.array([float(wts[k]) for k in VARIABLES], dtype=np.float64)
    if np.any(out < 0) or not np.any(out > 0):
        raise ValueError("weights must be non-negative and not all zero")
    return out


def rules(given):
    """The seven thresholds of Table 1 in kernel order (None switches one off)."""
    names = list(DEFAULT_RULES)
    given = given or {}
    unknown = set(given) - set(names)
    if unknown:
        raise ValueError(f"unknown rules keys {sorted(unknown)}; use {tuple(names)}")
    table = {**DEFAULT_RULES, **given}
    return np.array(
        [_OFF[k] if table[k] is None else float(table[k]) for k in names],
        dtype=np.float64,
    )


def range_correction(given):
    """Slopes (per km) of the textures of ZDR and PhiDP and the start (km)."""
    given = dict(given or {})
    start = float(given.pop("start_km", RANGE_CORRECTION_START_KM))
    unknown = set(given) - {"zdr", "phidp"}
    if unknown:
        raise ValueError(
            f"unknown range_correction keys {sorted(unknown)}; "
            "use 'zdr', 'phidp' and 'start_km'"
        )
    slopes = [float(given.get("zdr", 0.0)), float(given.get("phidp", 0.0))]
    if any(not np.isfinite(s) or s < 0 for s in slopes) or not np.isfinite(start):
        raise ValueError("range_correction slopes must be finite and non-negative")
    return np.array(slopes + [start], dtype=np.float64)
