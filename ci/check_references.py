#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Check that public functions cite the published methods they implement.

For every public module of ``radarx.retrieve``, ``radarx.grid`` and
``radarx.io`` whose module docstring says that it implements a published
method (it has a numpy-style ``References`` section), every public function
of the module (the names in ``__all__``, or the public functions defined in
the module if there is no ``__all__``) must have a ``References`` section in
its own docstring. The check is offline and only looks at the source files
with :mod:`ast`; it does not import radarx or check the references against
Crossref (do that by hand with ``https://api.crossref.org/works/<doi>``,
anonymously).

Run from the repository root::

    python ci/check_references.py

The exit status is 1 if a function is missing its ``References`` section.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

PACKAGES = ("retrieve", "grid", "io")
_SECTION = re.compile(r"^\s*References\s*\n\s*-{3,}\s*$", re.MULTILINE)


def has_references(docstring):
    """Whether a docstring has a numpy-style ``References`` section."""
    return bool(docstring and _SECTION.search(docstring))


def public_names(tree):
    """Names listed in ``__all__`` (None if the module has none)."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            try:
                return set(ast.literal_eval(node.value))
            except ValueError:  # computed __all__, check every public function
                return None
    return None


def check_module(path):
    """Names of public functions of ``path`` without a References section."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if not has_references(ast.get_docstring(tree, clean=False)):
        return []
    names = public_names(tree)
    missing = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        if names is not None and node.name not in names:
            continue
        if not has_references(ast.get_docstring(node, clean=False)):
            missing.append(node.name)
    return missing


def main(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    failures = []
    for package in PACKAGES:
        for path in sorted((root / "radarx" / package).glob("*.py")):
            if path.name.startswith("_"):
                continue
            for name in check_module(path):
                failures.append(f"{path.relative_to(root)}: {name}")
    if failures:
        print("public functions without a References section in a module that")
        print("implements a published method:")
        for line in failures:
            print(f"  {line}")
        return 1
    print("all checked public functions have a References section")
    return 0


if __name__ == "__main__":
    sys.exit(main())
