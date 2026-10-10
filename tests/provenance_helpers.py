#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Helper for tests that compare results without their provenance attributes."""

import re

import xarray as xr

KEYS = ("radarx_method", "radarx_references", "radarx_version")


def without_provenance(obj):
    """Copy of a Dataset or DataArray without the attributes radarx adds.

    Removes ``radarx_method``, ``radarx_references``, ``radarx_version`` and
    the ``function(parameters)`` lines of ``history``, and the three keys of
    the data variables of a Dataset.
    """
    out = obj.copy()
    for key in KEYS:
        out.attrs.pop(key, None)
    if isinstance(out, xr.Dataset):
        for var in out.data_vars.values():
            for key in KEYS:
                var.attrs.pop(key, None)
    lines = [
        line
        for line in str(out.attrs.get("history", "")).split("\n")
        if line and not re.fullmatch(r"\w+\(.*\)", line)
    ]
    if lines:
        out.attrs["history"] = "\n".join(lines)
    else:
        out.attrs.pop("history", None)
    return out
