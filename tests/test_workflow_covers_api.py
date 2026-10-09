#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""The end-to-end workflow notebooks must show every public radarx function.

``docs/notebooks/Radar_Workflow.md`` (the core pipeline on the KGWX squall
line) and ``docs/notebooks/Radar_Workflow_Advanced.md`` (the rest, on small
open or synthetic data) together call every name of the ``__all__`` of the
public sub-packages and every method of the ``.radarx`` accessors. This test
fails when a function is added to radarx without an example in the notebooks,
so that they cannot silently fall behind. Every name must also be listed in the
"Function index" of the first notebook.

To fix a failure, add a short example to the notebook section of the topic
(see the index for the sections) and add the name to the function index.
"""

import importlib
import inspect
import pkgutil
import re
from pathlib import Path

import pytest

import radarx  # noqa: F401  registers the accessors

NOTEBOOKS = Path(__file__).resolve().parents[1] / "docs" / "notebooks"
CORE = NOTEBOOKS / "Radar_Workflow.md"
ADVANCED = NOTEBOOKS / "Radar_Workflow_Advanced.md"

PACKAGES = [
    "radarx.retrieve",
    "radarx.grid",
    "radarx.io",
    "radarx.ml",
    "radarx.vis",
    "radarx.fundamentals",
    "radarx.core",
]  # radarx.testing only holds the deprecated IMD helpers (see EXCEPTIONS)

# Documented exceptions: the radarx IMD reader and its test-data helpers are
# deprecated (the files are read natively by xradar since its release after
# 0.12.0) and are removed after the next xradar release, so the notebooks show
# the xradar reader instead (see docs/notebooks/IMD_Radar_Data.md).
DEPRECATED_IMD = {
    "read_sweep",
    "read_volume",
    "to_cfradial2",
    "to_cfradial2_volumes",
    "fetch_imd_test_data",
    "display_fetched_files",
    "test_data_imd",
}
EXCEPTIONS = DEPRECATED_IMD

# Accessor attributes that are not retrievals (the plot accessor is shown with
# its classes and ``.radarx.plot.<kind>`` calls).
ACCESSOR_EXCEPTIONS = set()

CODE_CELL = re.compile(r"```\{code-cell\}[^\n]*\n(.*?)```", re.S)


IMPORT = re.compile(
    r"^[ \t]*from[ \t]+\S+[ \t]+import[ \t]+(?:\([^)]*\)|[^\n]*)|^[ \t]*import[ \t]+[^\n]*",
    re.M,
)


def code_of(path):
    """The text of the code cells of a MyST notebook, without the imports.

    A name that is only imported is not shown, so import statements are removed.
    """
    code = "\n".join(CODE_CELL.findall(path.read_text(encoding="utf-8")))
    return IMPORT.sub("", code)


def public_names():
    """``{name: module}`` of every public function, class and constant."""
    names = {}
    for package in PACKAGES:
        pkg = importlib.import_module(package)
        modules = [pkg]
        if hasattr(pkg, "__path__"):
            for info in pkgutil.iter_modules(pkg.__path__):
                if not info.name.startswith("_"):
                    modules.append(importlib.import_module(f"{package}.{info.name}"))
        for module in modules:
            for name in getattr(module, "__all__", []):
                obj = getattr(module, name, None)
                if inspect.ismodule(obj) or name.startswith("_"):
                    continue
                names.setdefault(name, module.__name__)
    return {k: v for k, v in sorted(names.items()) if k not in EXCEPTIONS}


def accessor_methods():
    from radarx.accessors import (
        RadarxDataArrayAccessor,
        RadarxDataSetAccessor,
        RadarxDataTreeAccessor,
    )

    methods = set()
    for cls in (RadarxDataArrayAccessor, RadarxDataSetAccessor, RadarxDataTreeAccessor):
        methods |= {n for n in dir(cls) if not n.startswith("_")}
    return sorted(methods - ACCESSOR_EXCEPTIONS)


def test_public_names_are_found():
    names = public_names()
    # a few anchors, so that an import problem cannot make the test pass trivially
    for name in ("dealias_velocity", "grid_cones", "read_sounding", "polar_patches"):
        assert name in names
    assert len(names) > 250


@pytest.mark.parametrize("path", [CORE, ADVANCED], ids=lambda p: p.name)
def test_notebooks_exist(path):
    assert path.exists()


def test_every_public_name_is_in_a_workflow_notebook():
    code = code_of(CORE) + "\n" + code_of(ADVANCED)
    missing = []
    for name, module in public_names().items():
        obj = getattr(importlib.import_module(module), name)
        # functions must be called; classes and constants only need to be used
        pattern = (
            rf"\b{re.escape(name)}\s*\("
            if inspect.isroutine(obj)
            else rf"\b{re.escape(name)}\b"
        )
        if not re.search(pattern, code):
            missing.append(name)
    assert (
        not missing
    ), "not shown in docs/notebooks/Radar_Workflow(.md|_Advanced.md): " + ", ".join(
        missing
    )


def test_every_accessor_method_is_in_a_workflow_notebook():
    code = code_of(CORE) + "\n" + code_of(ADVANCED)
    missing = [
        n
        for n in accessor_methods()
        if not re.search(rf"\.radarx\.{re.escape(n)}\b", code)
    ]
    assert (
        not missing
    ), "accessor methods not shown in the workflow notebooks: " + ", ".join(missing)


def test_every_public_name_is_in_the_function_index():
    text = CORE.read_text(encoding="utf-8")
    assert "## Function index" in text
    index = text.split("## Function index", 1)[1].split("\n## ", 1)[0]
    listed = set(re.findall(r"^\| `([A-Za-z_0-9]+)` \|", index, re.M))
    missing = [n for n in public_names() if n not in listed]
    assert not missing, "not in the Function index: " + ", ".join(missing)
    unknown = sorted(n for n in listed if n not in public_names())
    assert not unknown, "in the Function index but not public: " + ", ".join(unknown)
    accessors = set(re.findall(r"^\| `\.radarx\.([A-Za-z_0-9]+)` \|", index, re.M))
    assert accessors == set(
        accessor_methods()
    ), "accessor methods of the index differ from the accessors: " + ", ".join(
        sorted(accessors ^ set(accessor_methods()))
    )
