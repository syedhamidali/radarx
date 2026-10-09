"""Check that public functions of published-method modules carry References.

A module counts as implementing a published method when its module docstring
cites literature (an "et al." citation or a ``doi.org`` link).  For such a
module every public top-level function (listed in ``__all__`` when the module
defines it, otherwise every name not starting with an underscore) must have a
``References`` section in its own docstring, or the docstring must point to
the function it wraps with ``See Also`` or ``Same as`` wording and carry no
scientific content of its own (put such a function in the ``EXEMPT`` set).

The check is offline and only parses the source with ``ast``.

Usage: ``python ci/check_references.py`` (exit status 1 on failure).
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("retrieve", "grid", "io")
# "module.function" names that are exempt: thin wrappers, I/O helpers and
# functions that apply a documented standard definition without a method of
# their own (their docstrings point to the function that carries the
# references). Keep this list short and justified.
EXEMPT: set[str] = {
    "lightning.vertical_source_distribution",  # binning; flash rules: grid_lightning
    "lightning.cell_flash_rate",  # binning; flash rules: grid_lightning
    "wind_profile.layer_mean_wind",  # trapezoidal mean, no published method
    "wind_profile.storm_relative_wind",  # vector subtraction
    "aws_data.get_s3_client",
    "aws_data.list_available_files",
    "aws_data.download_file",
    "sounding.air_density",  # ideal-gas law, standard definition
    "sounding.open_sounding_file",
    "sounding.station_list",
    "sounding.nearest_station",
    "sounding.interpolate_profile",
    "sounding.isotherm_height",
    "sounding.mean_wind",
    "sounding.profile_to_grid",
    "surface.read_sticknet_locations",
    "surface.read_pips",
}
# Modules of other topic groups that have not yet been through the citation
# pass; remove an entry when its module is done.
PENDING_MODULES = {
    "disdrometer",
    "dsd",
    "dsd_bayes",
    "hid",
    "multidoppler",
    "qc",
    "rain_trajectories",
    "shear",
}

CITES_LITERATURE = re.compile(r"et al\.|doi\.org/")
HAS_REFERENCES = re.compile(r"^\s*References\s*\n\s*-{3,}", re.MULTILINE)


def public_functions(tree: ast.Module) -> list[ast.FunctionDef]:
    exported = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            try:
                exported = set(ast.literal_eval(node.value))
            except ValueError:
                exported = None
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if exported is not None:
        return [f for f in funcs if f.name in exported]
    return [f for f in funcs if not f.name.startswith("_")]


def main() -> int:
    missing: list[str] = []
    for pkg in PACKAGES:
        for path in sorted((ROOT / "radarx" / pkg).glob("*.py")):
            if path.name.startswith("_"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            module_doc = ast.get_docstring(tree) or ""
            if not CITES_LITERATURE.search(module_doc):
                continue
            for func in public_functions(tree):
                key = f"{path.stem}.{func.name}"
                if key in EXEMPT or path.stem in PENDING_MODULES:
                    continue
                if not HAS_REFERENCES.search(ast.get_docstring(func) or ""):
                    missing.append(f"radarx/{pkg}/{path.name}: {func.name}")
    if missing:
        print("Public functions without a 'References' section:")
        for line in missing:
            print("  " + line)
        return 1
    print("All public functions of published-method modules carry References.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
