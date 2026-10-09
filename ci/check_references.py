#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Check that public functions of published methods carry a References section.

Offline and static (the modules are parsed, not imported). For every module of
``radarx/retrieve``, ``radarx/grid`` and ``radarx/io`` whose module docstring
has a ``References`` section (it implements a published method), every public
top-level function (name without a leading underscore, listed in ``__all__``
when the module defines it, and not an accessor registered with
``accessor_method``) must have a numpy-style ``References`` section in its own
docstring, unless the docstring says "No published method" (a helper that
implements nothing from the literature). With ``--orphans`` the script also reports, per docstring, a
reference whose first-author surname is not mentioned in the text before the
section (cited nowhere) and an author-year citation in the text without an
entry (only citations of the form ``Surname (YYYY)`` or ``Surname et al.
YYYY`` are recognised).

Usage::

    python ci/check_references.py [--orphans] [files ...]

Exit status 1 if anything is reported.
"""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("retrieve", "grid", "io")
SECTION = re.compile(r"^\s*References\s*\n\s*-{3,}\s*$", re.M)
ENTRY = re.compile(
    r"^\s{0,4}([A-Z][\w\-\u00c0-\u017f]+),\s+(?:[A-Z]\.|[A-Z]\w*\.)", re.M
)
YEAR = r"(?:19|20)\d\d"


def has_references(doc):
    return bool(doc and SECTION.search(doc))


def split_references(doc):
    m = SECTION.search(doc)
    body, refs = doc[: m.start()], doc[m.end() :]
    stop = re.search(r"^\s*(Examples|Notes|See Also)\s*\n\s*-{3,}", refs, re.M)
    return body, refs[: stop.start()] if stop else refs


def public_names(tree):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "__all__":
                    return {ast.literal_eval(e) for e in node.value.elts}
    return None


def is_accessor(node):
    for d in node.decorator_list:
        func = d.func if isinstance(d, ast.Call) else d
        if getattr(func, "id", getattr(func, "attr", "")) == "accessor_method":
            return True
    return False


def orphans(doc):
    body, refs = split_references(doc)
    body = re.sub(r"\s+", " ", body)
    out = []
    for m in ENTRY.finditer(refs):
        surname = m.group(1)
        if surname not in body:
            out.append(f"reference of {surname} is not cited in the text")
    cited = set()
    for m in re.finditer(
        r"(?<!and )(?<!du )([A-Z][\w\-\u00c0-\u017f]+)(?: et al\.?,?| and ([A-Z][\w\-\u00c0-\u017f]+))? \(?"
        + YEAR,
        body,
    ):
        cited.add(m.group(1))
    listed = {m.group(1) for m in ENTRY.finditer(refs)}
    for name in sorted(cited - listed):
        if name not in ("Default", "Eq", "Eqs", "Sect", "Table", "Tables", "In"):
            out.append(f"{name} is cited in the text but has no entry")
    return out


def check(path, with_orphans):
    tree = ast.parse(path.read_text())
    mod_doc = ast.get_docstring(tree, clean=False)
    if not has_references(mod_doc):
        return []
    names = public_names(tree)
    problems = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        if names is not None and node.name not in names:
            continue
        if is_accessor(node):
            continue
        doc = ast.get_docstring(node, clean=False)
        if doc and "No published method" in doc:
            continue
        if not has_references(doc):
            problems.append(
                f"{path.relative_to(ROOT)}:{node.lineno}: "
                f"{node.name} has no References section"
            )
        elif with_orphans:
            for p in orphans(doc):
                problems.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}: " f"{node.name}: {p}"
                )
    return problems


def main(argv):
    with_orphans = "--orphans" in argv
    files = [Path(a) for a in argv if not a.startswith("--")]
    if not files:
        files = sorted(
            f
            for pkg in PACKAGES
            for f in (ROOT / "radarx" / pkg).glob("*.py")
            if not f.name.startswith("_")
        )
    problems = [p for f in files for p in check(Path(f).resolve(), with_orphans)]
    for p in problems:
        print(p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
