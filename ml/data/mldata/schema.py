"""
Dataset schemas: every task's variables, dimensions, dtypes and roles.

Each task is written as one Zarr store per split, ``<task>/<split>.zarr``,
with all samples stacked on the ``sample`` dimension. Every variable has a
role:

``input``
    What a model sees.
``label``
    The training target, produced by a radarx retrieval (or known exactly,
    for the artificial folding and blockage).
``baseline``
    radarx's own answer from the inputs, for comparing a trained model with
    the physical method on the same samples.
``meta``
    Per-sample metadata (also as coordinates: ``radar``, ``volume_id``,
    ``time``, ...).

The schemas are the contract with the training code (``ml/models/``); a
change here is a change of the dataset version, :data:`SCHEMA_VERSION`.
"""

from __future__ import annotations

import numpy as np

#: Version of the dataset layout; bump on any change of the schemas.
SCHEMA_VERSION = "1"

POLAR = ("sample", "azimuth", "range")
#: Tasks on the fixed polar grid (one sample per sweep).
POLAR_TASKS = ("qc", "kdp", "hid", "dealias", "inpaint")
GRID2D = ("sample", "y", "x")


def _v(dims, dtype, role, units, long_name, **extra):
    return dict(
        dims=tuple(dims),
        dtype=np.dtype(dtype).str,
        role=role,
        units=units,
        long_name=long_name,
        **extra,
    )


#: Coordinates on ``sample`` shared by all polar tasks.
POLAR_COORDS = {
    "radar": _v(("sample",), "<U4", "meta", "", "ICAO identifier of the radar"),
    "volume_id": _v(("sample",), "<U32", "meta", "", "radar and volume start time"),
    "sweep": _v(("sample",), "int16", "meta", "1", "sweep index in the volume"),
    "elevation": _v(("sample",), "float32", "meta", "degrees", "fixed angle"),
    "time": _v(("sample",), "datetime64[ns]", "meta", "", "sweep start time"),
    "latitude": _v(("sample",), "float32", "meta", "degrees_north", "radar"),
    "longitude": _v(("sample",), "float32", "meta", "degrees_east", "radar"),
    "altitude": _v(("sample",), "float32", "meta", "m", "radar above sea level"),
    "case": _v(("sample",), "<U64", "meta", "", "case name in the config"),
}

_DBZH = ("dBZ", "equivalent reflectivity factor (no-data codes masked)")
_ZDR = ("dB", "differential reflectivity")
_RHOHV = ("1", "copolar correlation coefficient")
_PHIDP = ("degrees", "measured (raw) differential phase")

SCHEMAS = {
    "qc": {
        "description": "Gate-wise echo classification (radarx.retrieve.echo_mask).",
        "variables": {
            "DBZH": _v(POLAR, "float32", "input", *_DBZH),
            "ZDR": _v(POLAR, "float32", "input", *_ZDR),
            "RHOHV": _v(POLAR, "float32", "input", *_RHOHV),
            "PHIDP": _v(POLAR, "float32", "input", *_PHIDP),
            "ECHO_CLASS": _v(
                POLAR,
                "int8",
                "label",
                "1",
                "echo class",
                flag_values=[0, 1, 2, 3],
                flag_meanings="no_echo meteorological non_meteorological speckle",
            ),
            "METEO_SCORE": _v(
                POLAR, "float32", "label", "1", "meteorological membership score"
            ),
        },
    },
    "kdp": {
        "description": "Differential phase processing and KDP "
        "(radarx.retrieve.estimate_kdp).",
        "variables": {
            "PHIDP": _v(POLAR, "float32", "input", *_PHIDP),
            "RHOHV": _v(POLAR, "float32", "input", *_RHOHV),
            "DBZH": _v(POLAR, "float32", "input", *_DBZH),
            "ZDR": _v(POLAR, "float32", "input", *_ZDR),
            "PHIDP_processed": _v(
                POLAR, "float32", "label", "degrees", "processed differential phase"
            ),
            "KDP": _v(
                POLAR, "float32", "label", "degrees/km", "specific differential phase"
            ),
        },
    },
    "hid": {
        "description": "Hydrometeor classification (radarx.retrieve.hid) of "
        "meteorological gates.",
        "variables": {
            "DBZH": _v(POLAR, "float32", "input", *_DBZH),
            "ZDR": _v(POLAR, "float32", "input", *_ZDR),
            "RHOHV": _v(POLAR, "float32", "input", *_RHOHV),
            "KDP": _v(POLAR, "float32", "input", "degrees/km", "KDP from radarx"),
            "TEMPERATURE": _v(
                POLAR, "float32", "input", "degC", "temperature at the gate (NaN: none)"
            ),
            "METEO_MASK": _v(POLAR, "bool", "input", "1", "meteorological echo"),
            "HID": _v(
                POLAR,
                "int8",
                "label",
                "1",
                "hydrometeor class (flag_values/flag_meanings from radarx; 0 none)",
            ),
            "HID_confidence": _v(
                POLAR, "float32", "label", "1", "score of the assigned class"
            ),
        },
    },
    "dealias": {
        "description": "Velocity dealiasing: unaliased velocities folded "
        "artificially into a lower Nyquist interval.",
        "variables": {
            "VRADH_folded": _v(
                POLAR, "float32", "input", "m s-1", "radial velocity folded at nyquist"
            ),
            "DBZH": _v(POLAR, "float32", "input", *_DBZH),
            "nyquist_velocity": _v(
                ("sample",), "float32", "input", "m s-1", "artificial Nyquist velocity"
            ),
            "VRADH": _v(POLAR, "float32", "label", "m s-1", "true radial velocity"),
            "FOLD": _v(
                POLAR, "int8", "label", "1", "fold k: VRADH = folded + 2 k nyquist"
            ),
            "VRADH_radarx": _v(
                POLAR,
                "float32",
                "baseline",
                "m s-1",
                "radarx.retrieve.dealias_velocity of VRADH_folded",
            ),
            "nyquist_velocity_original": _v(
                ("sample",), "float32", "meta", "m s-1", "Nyquist velocity of the scan"
            ),
            "truth_source": _v(
                ("sample",),
                "int8",
                "meta",
                "1",
                "origin of VRADH",
                flag_values=[0, 1],
                flag_meanings="observed_unaliased radarx_dealiased",
            ),
            "folded_fraction": _v(
                ("sample",),
                "float32",
                "meta",
                "1",
                "share of valid gates with FOLD != 0",
            ),
        },
    },
    "inpaint": {
        "description": "Beam-blockage inpainting: artificial total and partial "
        "blockage sectors over quality-controlled reflectivity.",
        "variables": {
            "DBZH_blocked": _v(
                POLAR, "float32", "input", "dBZ", "DBZH with artificial blockage"
            ),
            "BLOCKAGE": _v(
                POLAR, "float32", "input", "1", "blocked fraction of the beam power"
            ),
            "DBZH": _v(
                POLAR, "float32", "label", "dBZ", "quality-controlled reflectivity"
            ),
        },
    },
    "nowcast": {
        "description": "Composite-reflectivity sequences; motion "
        "(radarx.retrieve.estimate_motion) and extrapolation "
        "(radarx.retrieve.advect) from the input frames only.",
        "variables": {
            "DBZH": _v(
                ("sample", "lead", "y", "x"),
                "float32",
                "input",
                "dBZ",
                "column-maximum reflectivity of the input frames (lead <= 0)",
            ),
            "DBZH_target": _v(
                ("sample", "lead_target", "y", "x"),
                "float32",
                "label",
                "dBZ",
                "column-maximum reflectivity of the target frames (lead > 0)",
            ),
            "frame_time": _v(
                ("sample", "lead"), "datetime64[ns]", "meta", "", "volume start time"
            ),
            "target_time": _v(
                ("sample", "lead_target"),
                "datetime64[ns]",
                "meta",
                "",
                "volume start time",
            ),
            "u": _v(GRID2D, "float32", "baseline", "m s-1", "eastward echo motion"),
            "v": _v(GRID2D, "float32", "baseline", "m s-1", "northward echo motion"),
            "DBZH_extrapolated": _v(
                ("sample", "lead_target", "y", "x"),
                "float32",
                "baseline",
                "dBZ",
                "lead-0 frame advected to each target lead",
            ),
        },
    },
    "multidoppler": {
        "description": "Three-dimensional wind from two radars "
        "(radarx.retrieve.multi_doppler) with the gridded inputs.",
        "variables": {
            "VRADH": _v(
                ("sample", "radar_index", "z", "y", "x"),
                "float32",
                "input",
                "m s-1",
                "gridded dealiased radial velocity",
            ),
            "DBZH": _v(
                ("sample", "radar_index", "z", "y", "x"),
                "float32",
                "input",
                "dBZ",
                "gridded reflectivity",
            ),
            "beam_azimuth": _v(
                ("sample", "radar_index", "z", "y", "x"),
                "float32",
                "input",
                "degrees",
                "beam azimuth at the cell",
            ),
            "beam_elevation": _v(
                ("sample", "radar_index", "z", "y", "x"),
                "float32",
                "input",
                "degrees",
                "beam elevation at the cell",
            ),
            "u": _v(("sample", "z", "y", "x"), "float32", "label", "m s-1", "eastward"),
            "v": _v(
                ("sample", "z", "y", "x"), "float32", "label", "m s-1", "northward"
            ),
            "w": _v(("sample", "z", "y", "x"), "float32", "label", "m s-1", "upward"),
            "n_radars": _v(
                ("sample", "z", "y", "x"), "int8", "label", "1", "radars observing"
            ),
            "beam_crossing_angle": _v(
                ("sample", "z", "y", "x"), "float32", "label", "degrees", "crossing"
            ),
        },
    },
}

#: Coordinates on ``sample`` of the gridded tasks.
GRID_COORDS = {
    "radar": POLAR_COORDS["radar"],
    "case": POLAR_COORDS["case"],
    "time": _v(("sample",), "datetime64[ns]", "meta", "", "analysis (lead 0) time"),
}

for _task, _spec in SCHEMAS.items():
    _spec["coords"] = dict(POLAR_COORDS if _task in POLAR_TASKS else GRID_COORDS)
SCHEMAS["multidoppler"]["coords"]["radar"] = _v(
    ("sample", "radar_index"), "<U4", "meta", "", "ICAO identifiers of the radars"
)

#: Task names in build order.
TASKS = tuple(SCHEMAS)


def variables(task, role=None):
    """Names of the variables of ``task`` (with ``role``, if given)."""
    spec = SCHEMAS[task]["variables"]
    return [k for k, v in spec.items() if role is None or v["role"] == role]


def attrs_for(task, name):
    """CF-style attributes of variable ``name`` of ``task``."""
    spec = SCHEMAS[task]["variables"][name]
    attrs = {"long_name": spec["long_name"], "role": spec["role"]}
    if spec["units"]:  # datetimes get their units from the encoding
        attrs["units"] = spec["units"]
    for key in ("flag_values", "flag_meanings"):
        if key in spec:
            attrs[key] = spec[key]
    return attrs


def validate(ds, task):
    """
    Check that ``ds`` follows the schema of ``task``.

    Parameters
    ----------
    ds : xarray.Dataset
        Samples of one task, e.g. from :func:`mldata.open_dataset`.
    task : str
        Task name, one of :data:`TASKS`.

    Raises
    ------
    ValueError
        Listing every missing variable and every dims or dtype mismatch.
    """
    if task not in SCHEMAS:
        raise ValueError(f"unknown task {task!r}; one of {TASKS}")
    problems = []
    spec = dict(SCHEMAS[task]["variables"])
    spec.update(SCHEMAS[task]["coords"])
    for name, s in spec.items():
        if name not in ds.variables:
            problems.append(f"missing {name}")
            continue
        var = ds[name]
        if tuple(var.dims) != s["dims"]:
            problems.append(f"{name}: dims {var.dims} != {s['dims']}")
        kind = np.dtype(s["dtype"]).kind
        if kind == "U":
            if var.dtype.kind not in "UO":
                problems.append(f"{name}: dtype {var.dtype} is not a string")
        elif var.dtype != np.dtype(s["dtype"]):
            problems.append(f"{name}: dtype {var.dtype} != {np.dtype(s['dtype'])}")
    if problems:
        raise ValueError(f"{task}: " + "; ".join(problems))
