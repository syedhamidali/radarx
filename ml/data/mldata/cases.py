"""
Case selection, train/val/test splits and leakage checks.

A *case* is one radar over a time window (an event: a squall line, a
hurricane landfall, a snowstorm). Splits are assigned per **event**, never
per volume or sweep: consecutive volumes and neighbouring radars see the same
storms, so splitting them would leak test storms into training. Every case
has an ``event`` key (default: its start date); all cases of an event share a
split. Radars listed in ``holdout_radars`` are test-only, and so is every
event they take part in. :func:`check_no_leakage` verifies the result: cases
in different splits must be at least ``min_gap_hours`` apart, and held-out
radars must not appear outside ``test``.
"""

from __future__ import annotations

import copy
import hashlib
import zlib
from datetime import datetime, timedelta

import numpy as np

SPLITS = ("train", "val", "test")

#: Defaults of every config key (see ``ml/data/configs/*.yaml``).
DEFAULTS = {
    "name": "radarx-nexrad",
    "seed": 0,
    "max_elevation": 4.0,
    "max_volumes_per_case": None,
    "polar": {
        "n_azimuth": 360,
        "first_gate": 2125.0,
        "gate_spacing": 250.0,
        "n_gates": 920,
    },
    "tasks": {
        "qc": {},
        "kdp": {},
        "hid": {},
        "dealias": {
            "nyquist": [8.0, 20.0],
            "sources": ["observed_unaliased", "radarx_dealiased"],
            "max_jump_fraction": 5e-4,
            "min_gates": 2000,
        },
        "inpaint": {
            "n_sectors": [1, 3],
            "width": [2.0, 20.0],
            "start_range": [2000.0, 60000.0],
            "partial_probability": 0.5,
            "partial_fraction": [0.3, 0.9],
        },
        "nowcast": {
            "grid": {
                "size": 256,
                "spacing": 1000.0,
                "z": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0],
                "max_range": 230000.0,
                "no_echo": -10.0,
            },
            "n_input": 2,
            "n_target": 1,
            "max_gap_minutes": 12.0,
            "motion": {"tile": 64000.0, "floor": 10.0},
        },
        "multidoppler": {"pairs": [], "grid": None},
    },
    "splits": {
        "fractions": {"train": 0.7, "val": 0.15, "test": 0.15},
        "holdout_radars": [],
        "min_gap_hours": 24.0,
    },
    "cases": [],
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def as_datetime(value):
    """Naive UTC datetime from a datetime or an ISO string (``Z`` allowed)."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    text = str(value).strip().replace("Z", "")
    return datetime.fromisoformat(text)


def normalise(cfg):
    """
    Config with defaults filled in and every case completed and checked.

    Tasks set to ``false`` (or ``null``) in the config are removed. Each case
    needs ``name``, ``radar``, ``start`` and ``end``; ``event`` defaults to
    the start date.

    Raises
    ------
    ValueError
        For missing case keys, duplicate names, bad split names or fractions.
    """
    user_tasks = (cfg or {}).get("tasks")
    out = _merge(DEFAULTS, cfg)
    if user_tasks is not None:
        out["tasks"] = {
            k: _merge(DEFAULTS["tasks"].get(k, {}), v if isinstance(v, dict) else {})
            for k, v in user_tasks.items()
            if v not in (False, None)
        }
    fractions = out["splits"]["fractions"]
    if set(fractions) - set(SPLITS) or not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError(f"split fractions must be over {SPLITS} and sum to 1")
    names = set()
    cases = []
    for case in out["cases"]:
        missing = {"name", "radar", "start", "end"} - set(case)
        if missing:
            raise ValueError(f"case {case.get('name', case)} lacks {sorted(missing)}")
        if case["name"] in names:
            raise ValueError(f"duplicate case name {case['name']!r}")
        names.add(case["name"])
        c = dict(case)
        c["radar"] = str(c["radar"]).upper()
        c["start"], c["end"] = as_datetime(c["start"]), as_datetime(c["end"])
        if c["end"] < c["start"]:
            raise ValueError(f"case {c['name']!r} ends before it starts")
        c["event"] = str(c.get("event") or f"{c['start']:%Y-%m-%d}")
        if c.get("split") not in (None, *SPLITS):
            raise ValueError(f"case {c['name']!r}: unknown split {c['split']!r}")
        cases.append(c)
    out["cases"] = cases
    return out


def load_config(path):
    """Read a YAML case-selection config and :func:`normalise` it."""
    import yaml

    with open(path) as f:
        return normalise(yaml.safe_load(f))


def hash_split(event, seed, fractions):
    """Deterministic split of an event from a hash of ``seed`` and ``event``."""
    digest = hashlib.sha256(f"{seed}:{event}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2.0**64
    edge = 0.0
    for name in SPLITS:
        edge += fractions.get(name, 0.0)
        if u < edge:
            return name
    return SPLITS[-1]


def assign_splits(cfg):
    """
    Set ``split`` on every case of a normalised config (in place).

    Order: explicit ``split`` of any case of an event, else ``test`` if a
    held-out radar takes part in the event, else :func:`hash_split`. All
    cases of an event get the same split.

    Returns
    -------
    list of dict
        The cases.
    """
    holdout = {r.upper() for r in cfg["splits"]["holdout_radars"]}
    events = {}
    for c in cfg["cases"]:
        events.setdefault(c["event"], []).append(c)
    for event, cases in events.items():
        explicit = {c["split"] for c in cases if c.get("split")}
        if len(explicit) > 1:
            raise ValueError(f"event {event!r} has cases in splits {sorted(explicit)}")
        if any(c["radar"] in holdout for c in cases):
            if explicit and explicit != {"test"}:
                raise ValueError(
                    f"event {event!r} includes a held-out radar but is "
                    f"assigned to {explicit.pop()!r}"
                )
            split = "test"
        elif explicit:
            split = explicit.pop()
        else:
            split = hash_split(event, cfg["seed"], cfg["splits"]["fractions"])
        for c in cases:
            c["split"] = split
    return cfg["cases"]


def check_no_leakage(cases, min_gap_hours=24.0, holdout_radars=()):
    """
    Verify that the splits cannot share storms.

    Parameters
    ----------
    cases : sequence of dict
        Cases with ``name``, ``radar``, ``start``, ``end`` and ``split``.
    min_gap_hours : float
        Smallest time between two cases of different splits (any radars).
    holdout_radars : sequence of str
        Radars that may only be in ``test``.

    Raises
    ------
    ValueError
        Naming the first offending pair of cases or held-out radar.
    """
    gap = timedelta(hours=float(min_gap_hours))
    holdout = {r.upper() for r in holdout_radars}
    for c in cases:
        if c["radar"] in holdout and c["split"] != "test":
            raise ValueError(
                f"held-out radar {c['radar']} in {c['split']} ({c['name']})"
            )
    ordered = sorted(cases, key=lambda c: c["start"])
    for i, a in enumerate(ordered):
        for b in ordered[i + 1 :]:
            if b["start"] - a["end"] >= gap:
                break
            if a["split"] != b["split"]:
                raise ValueError(
                    f"cases {a['name']!r} ({a['split']}) and {b['name']!r} "
                    f"({b['split']}) are less than {min_gap_hours} h apart"
                )


def expand_cases(cfg, lister=None):
    """
    Volumes of every case, in time order per case.

    Parameters
    ----------
    cfg : dict
        Normalised config whose cases have a ``split``.
    lister : callable, optional
        Passed to :func:`mldata.nexrad.list_volumes` (for tests).

    Returns
    -------
    list of dict
        One record per volume: ``volume_id``, ``key``, ``radar``, ``time``
        (ISO string), ``case``, ``event``, ``split``.
    """
    from .nexrad import list_volumes, parse_key, volume_id

    out = []
    limit = cfg.get("max_volumes_per_case")
    for c in cfg["cases"]:
        keys = list_volumes(c["radar"], c["start"], c["end"], lister=lister)
        n = c.get("max_volumes", limit)
        if n is not None:
            keys = keys[: int(n)]
        for key in keys:
            radar, when = parse_key(key)
            out.append(
                {
                    "volume_id": volume_id(radar, when),
                    "key": key,
                    "radar": radar,
                    "time": when.isoformat(),
                    "case": c["name"],
                    "event": c["event"],
                    "split": c["split"],
                }
            )
    return out


def sample_rng(seed, *parts):
    """Random generator for one sample, independent of processing order."""
    words = [int(seed) & 0xFFFFFFFF]
    words += [zlib.crc32(str(p).encode()) for p in parts]
    return np.random.default_rng(words)
