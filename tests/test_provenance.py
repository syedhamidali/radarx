#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for the method provenance attributes, radarx.cite and radarx.methods."""

import ast
import importlib
import inspect
import json
import pkgutil
import textwrap
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import radarx
from radarx import accessors
from radarx._provenance import (
    HISTORY,
    METHOD,
    REFERENCES,
    VERSION,
    docstring_dois,
    load_registry,
    provenance,
    read_provenance,
    set_provenance,
)

from .grid.test_cone import GRID
from .grid.test_cone import _volume as cone_volume
from .test_dealias import synthetic_sweep
from .test_dealias import volume as dealias_volume
from .test_kdp import _sweep as kdp_sweep
from .test_kdp import _volume as kdp_volume
from .test_qc import _sweep as qc_sweep
from .test_vil import storm_volume

ROOT = Path(__file__).resolve().parents[1]
ATTRS = (METHOD, REFERENCES, VERSION)

# Public functions of radarx.retrieve and radarx.grid that return plain arrays,
# tuples of arrays or lists, so there is no xarray object to annotate.
EXCEPTIONS = {
    "hid.hid_classes": "list of (code, abbreviation, name) tuples",
    "grid.make_3d_grid": "tuple of NumPy arrays",
    "ugrid.gate_corners": "tuple of NumPy arrays",
}

# Accessor methods that do not wrap a function of radarx.retrieve or
# radarx.grid: they merge, read soundings (radarx.io) or draw figures.
ACCESSOR_EXCEPTIONS = {
    "assign": "merges products into a sweep or volume",
    "background": "radarx.io.sounding",
    "interpolate_profile": "radarx.io.sounding",
    "sounding": "radarx.io.sounding",
    "plot_cappi": "returns a figure",
    "plot_max_cappi": "returns a figure",
    "plot_ppi": "returns a figure",
    "plot_rhi": "returns a figure",
}

# Registry keys of references without a DOI and the text that marks them in a
# docstring.
KEY_MARKS = {
    "doviak-zrnic-1993": "Doviak, R. J., and D. S. Zrni",
    "rogers-yau-1989": "Rogers, R. R., and M. K. Yau",
    "smith-elmore-2004": "Smith, T. M., and K. L. Elmore",
    "maesaka-2012": "Maesaka, T., K. Iwanami",
    "ester-1996": "Ester, M., H.-P. Kriegel",
}


def public_functions():
    """Qualified names ``module.function`` of the public retrieve and grid functions."""
    found = {}
    for package in ("radarx.retrieve", "radarx.grid"):
        pkg = importlib.import_module(package)
        for info in pkgutil.iter_modules(pkg.__path__):
            if info.name.startswith("_"):
                continue
            module = importlib.import_module(f"{package}.{info.name}")
            names = getattr(
                module, "__all__", [n for n in dir(module) if not n.startswith("_")]
            )
            for name in names:
                func = getattr(module, name)
                if inspect.isfunction(func) and func.__module__ == module.__name__:
                    found[f"{info.name}.{name}"] = func
    return found


def check_attrs(obj):
    """The three attributes and a history line on every node of ``obj``."""
    nodes = [obj]
    if isinstance(obj, xr.DataTree):  # the root is the input's root
        nodes = list(obj.subtree)[1:]
    assert nodes
    for node in nodes:
        assert all(k in node.attrs for k in ATTRS), node.attrs
        assert node.attrs[METHOD]
        assert node.attrs[VERSION] == str(radarx.__version__)
        assert node.attrs[HISTORY]
        refs = [r for r in node.attrs[REFERENCES].split() if r]
        registry = load_registry()
        assert all(r in registry for r in refs)


def sweep_dataset():
    return qc_sweep()[0]


# static coverage


def test_every_public_function_has_provenance_or_an_exception():
    functions = public_functions()
    undecorated = sorted(
        name
        for name, func in functions.items()
        if not hasattr(func, "__radarx_provenance__") and name not in EXCEPTIONS
    )
    assert not undecorated
    assert set(EXCEPTIONS) <= set(functions)
    assert not [n for n in EXCEPTIONS if hasattr(functions[n], "__radarx_provenance__")]


def test_references_are_the_dois_of_the_docstring():
    for name, func in public_functions().items():
        info = getattr(func, "__radarx_provenance__", None)
        if info is None:
            continue
        dois = docstring_dois(inspect.getdoc(func))
        refs = info["references"]
        assert refs[: len(dois)] == dois, name
        extra = refs[len(dois) :]
        assert all(k in KEY_MARKS for k in extra), (name, extra)
        text = inspect.getdoc(func) or ""
        for key, mark in KEY_MARKS.items():
            assert (mark in text) == (key in extra), (name, key)
        assert info["method"], name


def test_every_doi_is_in_the_registry():
    registry = load_registry()
    lower = {k.lower() for k in registry}
    used = set()
    for func in public_functions().values():
        info = getattr(func, "__radarx_provenance__", None)
        if info:
            used.update(info["references"])
    assert used
    assert not [r for r in used if r.lower() not in lower]


def test_registry_entries_are_complete():
    registry = load_registry()
    assert len(registry) == len({k.lower() for k in registry})
    for key, entry in registry.items():
        assert entry["title"] and entry["authors"] and entry["type"], key
        if entry["type"] == "article":
            assert entry["journal"] and entry["year"] and entry["doi"], key
        if "doi" in entry and key != "radarx":
            assert entry["doi"] == key, key
        else:
            assert "." not in key.split("-")[0], key
    for key in KEY_MARKS:
        assert key in registry


def test_registry_matches_the_citation_file():
    cff = (ROOT / "CITATION.cff").read_text()
    entry = load_registry()["radarx"]
    assert f"doi: {entry['doi']}" in cff
    assert entry["doi"] == "10.5281/zenodo.14699306"
    assert entry["title"] in cff
    assert "family-names: Syed" in cff and entry["authors"] == ["Syed, H. A."]


def test_docstring_doi_parser():
    doc = textwrap.dedent(
        """
        Summary.

        References
        ----------
        .. [1] A, 2000. https://doi.org/10.1175/1520-
           0450(2000)<1:ABC>2.0.CO;2
        .. [2] B, 2001, https://doi.org/10.1000/xyz.
        .. [3] Book, no DOI.

        Examples
        --------
        https://doi.org/10.9999/not-a-reference
        """
    )
    assert docstring_dois(doc) == [
        "10.1175/1520-0450(2000)<1:ABC>2.0.CO;2",
        "10.1000/xyz",
    ]
    assert docstring_dois("No references here.") == []


# the helper


def test_set_provenance_on_every_kind_of_object():
    ds = xr.Dataset({"a": ("x", [1.0, 2.0])})
    da = ds["a"].copy()
    tree = xr.DataTree.from_dict({"/": xr.Dataset(), "s0": ds, "s1": ds})
    for obj in (ds, da, tree):
        out = set_provenance(obj, "A method", ["10.1/x"], {"w": 1.5, "mode": "m"}, "f")
        assert out is obj
    assert METHOD not in tree.attrs  # the root stays the input's root
    for node in (ds, da, tree["s0"], tree["s1"]):
        assert node.attrs[METHOD] == "A method"
        assert node.attrs[REFERENCES] == "10.1/x"
        assert node.attrs[HISTORY] == "f(w=1.5, mode='m')"


def test_chained_results_keep_both_methods():
    ds = xr.Dataset()
    set_provenance(ds, "First", ["10.1/a", "10.1/b"], function="one")
    set_provenance(ds, "Second", ["10.1/b", "10.1/c"], function="two")
    set_provenance(ds, "Second", ["10.1/c"], function="two")
    assert ds.attrs[METHOD] == "First | Second"
    assert ds.attrs[REFERENCES] == "10.1/a 10.1/b 10.1/c"
    assert ds.attrs[HISTORY] == "one()\ntwo()\ntwo()"
    (item,) = read_provenance(ds)
    assert item["method"] == ["First", "Second"]
    assert item["references"] == ["10.1/a", "10.1/b", "10.1/c"]


def test_history_has_no_time_stamp_and_no_arrays():
    ds, _ = qc_sweep()
    out = radarx.retrieve.echo_mask(ds, window=2.0, snr=None, weights={"rhohv": 1.0})
    line = out.attrs[HISTORY]
    assert line == "echo_mask(window=2.0)"
    again = radarx.retrieve.echo_mask(ds, window=2.0)
    assert again.attrs == out.attrs


def test_input_is_not_changed():
    ds, _ = qc_sweep()
    before = dict(ds.attrs)
    radarx.retrieve.echo_mask(ds)
    radarx.retrieve.apply_mask(ds)
    assert ds.attrs == before


def test_returned_input_is_copied_before_annotation():
    @provenance("Identity")
    def same(obj):
        return obj

    ds = xr.Dataset({"a": ("x", [1.0])})
    out = same(ds)
    assert METHOD in out.attrs and METHOD not in ds.attrs
    xr.testing.assert_equal(out, ds)


def test_only_the_outermost_call_annotates():
    @provenance("Inner")
    def inner():
        return xr.Dataset()

    @provenance("Outer")
    def outer():
        return inner()

    assert outer().attrs[METHOD] == "Outer"
    assert inner().attrs[METHOD] == "Inner"


def test_containers_and_plain_results():
    @provenance("Many")
    def many():
        return xr.Dataset(), [xr.DataArray([1.0])], {"k": xr.Dataset()}, np.ones(2)

    ds, das, mapping, arr = many()
    assert ds.attrs[METHOD] == das[0].attrs[METHOD] == mapping["k"].attrs[METHOD]
    assert isinstance(arr, np.ndarray)


# the public functions


def test_echo_mask_says_it_differs_from_the_papers():
    out = radarx.retrieve.echo_mask(sweep_dataset())
    assert "in the manner of" in out.attrs[METHOD]
    assert "own memberships and weights" in out.attrs[METHOD]
    assert "10.1175/JTECH2035.1" in out.attrs[REFERENCES]
    assert "10.1175/JTECH-D-15-0239.1" in out.attrs[REFERENCES]
    check_attrs(out)


def test_sweep_and_volume_results():
    sweep, *_ = kdp_sweep(nray=20, ng=300)
    dtree = kdp_volume()
    sweep_dop, truth = synthetic_sweep(nray=60, ngate=200)
    tree_dop, _ = dealias_volume(nray=60, ngate=200)
    shear = radarx.retrieve.llsd(sweep_dop["VRADH"].to_dataset(), "VRADH")
    results = {
        "estimate_kdp": radarx.retrieve.estimate_kdp(sweep),
        "estimate_kdp tree": radarx.retrieve.estimate_kdp(dtree),
        "echo_mask": radarx.retrieve.echo_mask(sweep_dataset()),
        "apply_mask": radarx.retrieve.apply_mask(sweep_dataset()),
        "dealias": radarx.retrieve.dealias_velocity(sweep_dop, nyquist_velocity=10.0),
        "dealias tree": radarx.retrieve.dealias_velocity(tree_dop),
        "llsd": shear,
        "azimuthal_shear": radarx.retrieve.azimuthal_shear(sweep_dop, "VRADH"),
        "radial_divergence": radarx.retrieve.radial_divergence(sweep_dop, "VRADH"),
        "vil": radarx.retrieve.vil(storm_volume()),
        "echo_top": radarx.retrieve.echo_top(storm_volume()),
    }
    for name, out in results.items():
        check_attrs(out)


def test_simple_dataset_functions():
    height = np.arange(0.0, 8000.0, 100.0)
    profile = xr.Dataset(
        {"u": ("height", height / 600.0), "v": ("height", np.zeros(height.size))},
        coords={"height": height},
    )
    diameter = xr.DataArray(np.linspace(0.5, 5.0, 10), dims="diameter")
    outs = [
        radarx.retrieve.bulk_shear(profile, 0, 6000),
        radarx.retrieve.layer_mean_wind(profile, 0, 6000),
        radarx.retrieve.bunkers_storm_motion(profile),
        radarx.retrieve.storm_relative_wind(profile, (3.0, 1.0)),
        radarx.retrieve.storm_relative_helicity(profile, (3.0, 1.0)),
        radarx.retrieve.terminal_fall_speed(diameter),
        radarx.retrieve.parsivel_bins(),
    ]
    for out in outs:
        check_attrs(out)
    assert outs[2].attrs[METHOD].startswith("Supercell motion of Bunkers")
    bunkers = docstring_dois(inspect.getdoc(radarx.retrieve.bunkers_storm_motion))
    assert bunkers and outs[2].attrs[REFERENCES].split() == bunkers


def test_grid_and_advection_functions():
    grid = radarx.grid.grid_cones(cone_volume(field="random"), "DBZH", **GRID)
    check_attrs(grid)
    assert "10.1175/2010JTECHA1402.1" in grid.attrs[REFERENCES]
    assert "doviak-zrnic-1993" in grid.attrs[REFERENCES]
    moved = radarx.retrieve.advect(grid, 5.0, 0.0, 60.0)
    # the grid's own provenance stays, the new method is added
    assert moved.attrs[METHOD].count(" | ") == 1
    assert moved.attrs[METHOD].startswith("Cone gridding")
    assert moved.attrs[HISTORY].splitlines()[-1] == "advect(u=5.0, v=0.0, dt=60.0)"
    check_attrs(moved)
    motion = radarx.retrieve.estimate_motion(grid, moved, "DBZH")
    check_attrs(motion)
    check_attrs(grid["DBZH"].radarx.advect(5.0, 0.0, 60.0))


def test_microphysics_and_thermodynamics_functions():
    params = xr.Dataset({"N0": 8000.0, "MU": 2.0, "LAMBDA": 2.5})
    levels = np.array([1000.0, 900.0, 800.0])
    sounding = xr.Dataset(
        {
            "temperature": ("level", [295.0, 288.0, 281.0]),
            "pressure": ("level", levels * 100.0),
            "dewpoint": ("level", [290.0, 283.0, 277.0]),
        }
    )
    outs = [
        radarx.retrieve.fall_speed(xr.DataArray([20.0, 30.0, 40.0], dims="x")),
        radarx.retrieve.dsd_spectrum(params),
        radarx.retrieve.potential_temperatures(sounding),
        radarx.retrieve.scattering_table(),
    ]
    for out in outs:
        check_attrs(out)


def test_method_without_published_source_has_empty_references():
    height = np.arange(0.0, 3000.0, 100.0)
    profile = xr.Dataset(
        {"u": ("height", height / 600.0), "v": ("height", 0 * height)},
        coords={"height": height},
    )
    out = radarx.retrieve.layer_mean_wind(profile, 0, 3000)
    assert out.attrs[REFERENCES] == ""
    check_attrs(out)


# accessors


def local_imports(tree, func):
    """Names that a function imports inside its body, with the imported objects."""
    local = {}
    package = func.__module__.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = importlib.import_module(
                "." * node.level + (node.module or ""), package
            )
            for alias in node.names:
                local[alias.asname or alias.name] = getattr(module, alias.name)
    return local


def call_target(node, local, func):
    """The object a call refers to, or None if it cannot be resolved by name."""
    target = node.func
    if isinstance(target, ast.Name):
        return local.get(target.id, func.__globals__.get(target.id))
    if not isinstance(target, ast.Attribute) or not isinstance(target.value, ast.Name):
        return None
    if target.value.id == "self":  # another accessor method
        return getattr(accessors.RadarxDataTreeAccessor, target.attr, None)
    base = local.get(target.value.id, func.__globals__.get(target.value.id))
    return getattr(base, target.attr, None)


def called_decorated(func, seen=None):
    """Does the accessor method call a function that records provenance?"""
    seen = set() if seen is None else seen
    if func in seen:
        return False
    seen.add(func)
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    local = local_imports(tree, func)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        obj = call_target(node, local, func)
        if hasattr(obj, "__radarx_provenance__"):
            return True
        if inspect.isfunction(obj) and obj.__module__.startswith("radarx.accessors"):
            if called_decorated(obj, seen):
                return True
    return False
    seen.add(func)
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    local = {}
    package = func.__module__.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = importlib.import_module(
                "." * node.level + (node.module or ""), package
            )
            for alias in node.names:
                local[alias.asname or alias.name] = getattr(module, alias.name)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name):
            obj = local.get(target.id, func.__globals__.get(target.id))
        elif (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):  # another accessor method
            obj = getattr(accessors.RadarxDataTreeAccessor, target.attr, None)
        elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            base = local.get(target.value.id, func.__globals__.get(target.value.id))
            obj = getattr(base, target.attr, None)
        else:
            continue
        if hasattr(obj, "__radarx_provenance__"):
            return True
        if inspect.isfunction(obj) and obj.__module__.startswith("radarx.accessors"):
            if called_decorated(obj, seen):
                return True
    return False


def accessor_methods():
    classes = (
        accessors.RadarxDataArrayAccessor,
        accessors.RadarxDataSetAccessor,
        accessors.RadarxDataTreeAccessor,
    )
    for cls in classes:
        for name, member in inspect.getmembers(cls):
            if name.startswith("_") or name == "plot":
                continue
            if inspect.isfunction(member):
                yield cls.__name__, name, member


def test_every_accessor_method_wraps_a_function_with_provenance():
    seen = 0
    for cls, name, member in accessor_methods():
        if name in ACCESSOR_EXCEPTIONS:
            continue
        seen += 1
        assert called_decorated(member), f"{cls}.{name}"
    assert seen > 40
    names = {name for _, name, _ in accessor_methods()}
    assert set(ACCESSOR_EXCEPTIONS) <= names


def test_accessor_methods_set_the_attributes():
    ds, *_ = kdp_sweep(nray=20, ng=300)
    check_attrs(ds.radarx.kdp())
    check_attrs(ds.radarx.echo_mask())
    check_attrs(ds.radarx.apply_mask())
    dtree = kdp_volume()
    check_attrs(dtree.radarx.kdp())
    tree_dop, _ = dealias_volume(nray=60, ngate=200)
    check_attrs(tree_dop.radarx.dealias())
    sweep_dop, _ = synthetic_sweep(nray=60, ngate=200)
    check_attrs(sweep_dop.radarx.dealias(nyquist_velocity=10.0))
    check_attrs(sweep_dop.radarx.llsd("VRADH"))
    check_attrs(sweep_dop.radarx.azimuthal_shear("VRADH"))
    check_attrs(sweep_dop.radarx.radial_divergence("VRADH"))
    volume = storm_volume()
    check_attrs(volume.radarx.vil())
    check_attrs(volume.radarx.echo_top())


# NetCDF


def test_attributes_survive_a_netcdf_round_trip(tmp_path):
    out = radarx.retrieve.echo_mask(sweep_dataset(), window=2.0)
    path = tmp_path / "mask.nc"
    out.to_netcdf(path)
    with xr.open_dataset(path) as back:
        for key in (METHOD, REFERENCES, VERSION, HISTORY):
            assert back.attrs[key] == out.attrs[key]
        assert len(back.attrs[REFERENCES].split()) == 5


def test_tree_attributes_survive_a_netcdf_round_trip(tmp_path):
    out = radarx.retrieve.estimate_kdp(kdp_volume())
    path = tmp_path / "kdp.nc"
    out.to_netcdf(path)
    back = xr.open_datatree(path)
    try:
        for node in back.subtree:
            if node.path == "/":
                continue
            assert node.attrs[METHOD] == out[node.path].attrs[METHOD]
            assert node.attrs[REFERENCES] == out[node.path].attrs[REFERENCES]
        assert "10.1175/1520-0426(1995)012<0643:AIFTFT>2.0.CO;2" in cite_dois(back)
    finally:
        back.close()


def cite_dois(obj):
    return "\n".join(radarx.cite(obj))


# cite and methods


def pipeline():
    sweep, *_ = kdp_sweep(nray=20, ng=300)
    mask = radarx.retrieve.echo_mask(sweep_dataset())
    kdp = radarx.retrieve.estimate_kdp(sweep)
    sweep_dop, _ = synthetic_sweep(nray=60, ngate=200)
    dealiased = radarx.retrieve.dealias_velocity(sweep_dop, nyquist_velocity=10.0)
    return mask, kdp, dealiased


def test_cite_pipeline_text():
    mask, kdp, dealiased = pipeline()
    refs = radarx.cite([mask, kdp, dealiased])
    assert refs[0].startswith("Syed, H. A.")
    assert "https://doi.org/10.5281/zenodo.14699306" in refs[0]
    text = "\n".join(refs)
    for doi in (
        "10.1175/JTECH2035.1",  # Gourley 2007, echo_mask
        "10.1175/1520-0426(1993)010<0798:TDDODV>2.0.CO;2",  # Jing and Wiener
        "10.1175/1520-0426(1995)012<0643:AIFTFT>2.0.CO;2",  # Hubbert and Bringi
    ):
        assert doi in text
    assert "Maesaka" in text  # a reference without DOI
    # Park et al. (2009) is cited by echo_mask and estimate_kdp, listed once
    assert sum("2008WAF2222205.1" in r for r in refs) == 1
    assert len(refs) == len(set(refs))
    assert len(refs) == 1 + len(
        {r for x in (mask, kdp, dealiased) for r in x.attrs[REFERENCES].split()}
    )


def test_cite_bibtex_and_keys():
    refs = radarx.cite(pipeline()[0], style="bibtex")
    assert refs[0].startswith("@misc{syed")
    assert all(r.startswith("@") and r.endswith("}") for r in refs)
    keys = [r.split("{", 1)[1].split(",", 1)[0] for r in refs]
    assert len(keys) == len(set(keys))
    assert any(k.startswith("gourley2007") for k in keys)
    assert "doi = {10.1175/JTECH2035.1}" in "\n".join(refs)


def test_cite_datatree_all_nodes_and_function():
    out = radarx.cite(radarx.retrieve.estimate_kdp(kdp_volume()))
    assert any("Hubbert" in r for r in out)
    assert len(out) == len(set(out))
    by_function = radarx.cite(radarx.retrieve.echo_mask)
    assert by_function == radarx.cite(radarx.retrieve.echo_mask(sweep_dataset()))


def test_cite_object_without_provenance_only_cites_radarx():
    assert len(radarx.cite(xr.Dataset())) == 1


def test_cite_unknown_reference_and_style():
    ds = set_provenance(xr.Dataset(), "Own", ["10.1234/unknown"])
    assert radarx.cite(ds)[1] == "https://doi.org/10.1234/unknown"
    with pytest.raises(ValueError):
        radarx.cite(ds, style="ris")


def test_methods_text():
    text = radarx.methods(radarx.retrieve.echo_mask)
    assert "Method: Fuzzy-logic echo score in the manner of" in text
    assert "Gourley et al. (2007)" in text
    result = radarx.methods(radarx.retrieve.echo_mask(sweep_dataset()))
    assert "Method: Fuzzy-logic echo score" in result and "Written by radarx" in result
    assert "no radarx provenance" in radarx.methods(xr.Dataset()).lower()


def test_registry_file_is_valid_json():
    path = ROOT / "radarx" / "data" / "references.json"
    assert isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
