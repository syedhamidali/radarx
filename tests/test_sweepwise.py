#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for the mechanism that runs sweep functions on stacks and dask arrays."""

import importlib
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from radarx import _sweepwise as sw

from .sweepwise_helpers import assert_lazy

NRAY, NGATE = 6, 8
SWEEP = ("azimuth", "range")


def field(shape=(NRAY, NGATE), dims=SWEEP, seed=0):
    values = np.random.default_rng(seed).uniform(1.0, 2.0, shape)
    coords = {
        "azimuth": np.arange(NRAY) * 60.0,
        "range": 1000.0 + 250.0 * np.arange(NGATE),
    }
    sizes = dict(zip(dims, shape))
    coords = {k: v for k, v in coords.items() if sizes.get(k) == v.size}
    return xr.DataArray(values, dims=dims, coords=coords, name="x")


def toy_core(batch):
    """Sweep result, ray sums, range maxima and a number per sweep."""
    out = []
    for context, (a, *rest) in batch:
        both = a if not rest or rest[0] is None else a + rest[0]
        out.append(
            (
                context * both,
                both.sum(axis=1),
                both.max(axis=0),
                both.sum(),
            )
        )
    return out


OUTPUTS = (
    sw.Output(np.float32),
    sw.Output(np.float64, "ray"),
    sw.Output(np.float64, "range"),
    sw.Output(np.float64, "scalar"),
)


def expected(a, b, context=2.0):
    both = a if b is None else a + b
    return context * both, both.sum(axis=-1), both.max(axis=-2), both.sum(axis=(-2, -1))


def test_ray_dimension():
    assert sw.ray_dimension(field()) == "azimuth"
    assert (
        sw.ray_dimension(field((2, NRAY, NGATE), ("time", "azimuth", "range")))
        == "azimuth"
    )
    rhi = field((NRAY, NGATE), ("elevation", "range"))
    assert sw.ray_dimension(rhi) == "elevation"
    assert sw.ray_dimension(field((NRAY, NGATE), ("ray", "range"))) == "ray"
    assert sw.extra_dims(field((2, NRAY, NGATE), ("time", "azimuth", "range"))) == (
        "time",
    )
    for dims in (("range",), ("time", "ray", "range"), ("azimuth", "gate")):
        with pytest.raises(ValueError, match="dimensions"):
            sw.ray_dimension(field((2,) * len(dims), dims))


def test_ray_coordinate():
    ds = field().to_dataset()
    np.testing.assert_array_equal(sw.ray_coordinate(ds, "azimuth"), ds.azimuth.values)
    stack = xr.concat([ds, ds], "time").assign(azimuth_t=("time", [0, 1]))
    assert sw.ray_coordinate(stack, "azimuth").shape == (NRAY,)
    bad = ds.assign_coords(azimuth=(("time", "azimuth"), np.zeros((2, NRAY))))
    with pytest.raises(ValueError, match="same rays"):
        sw.ray_coordinate(bad, "azimuth")


def test_plain_and_dask_detection():
    plain = field()
    assert sw.is_plain(plain)
    assert not sw.is_plain(field((2, NRAY, NGATE), ("time", "azimuth", "range")))
    pytest.importorskip("dask")
    assert sw.is_dask(plain.chunk())
    assert not sw.is_plain(plain.chunk())
    assert not sw.is_dask(plain)


def test_output_and_sweep_checks():
    with pytest.raises(ValueError, match="kind"):
        sw.Output(np.float32, "volume")
    assert "ray" in repr(sw.Output(np.float32, "ray"))
    with pytest.raises(ValueError, match="at least one field"):
        sw.Sweep([None, None])
    with pytest.raises(ValueError, match="at least one field"):
        sw.Gates([None])
    other = field((NRAY, NGATE), ("elevation", "range"))
    with pytest.raises(ValueError, match="dimensions"):
        sw.Sweep([field(), other])
    extra = field((2, NRAY, NGATE), ("time", "azimuth", "range"))
    with pytest.raises(ValueError, match="dimensions"):
        sw.Sweep([field(), extra])


@pytest.mark.parametrize("second", [True, False])
def test_plain_sweep_is_passed_directly(second):
    a, b = field(), field(seed=1)
    calls = []

    def core(batch):
        calls.append(len(batch))
        return toy_core(batch)

    sweep = sw.Sweep([a, b if second else None], context=2.0)
    assert sweep.plain
    res = sw.apply_sweeps(core, sweep, OUTPUTS)
    want = expected(a.values, b.values if second else None)
    for got, ref in zip(res, want):
        np.testing.assert_allclose(got.values, ref)
    assert calls == [1]
    assert res[0].dtype == np.float32
    assert res[0].dims == SWEEP and res[1].dims == ("azimuth",)
    assert res[2].dims == ("range",) and res[3].dims == ()
    assert "range" in res[0].coords


def test_transposed_field_keeps_its_dimension_order():
    a = field().transpose("range", "azimuth")
    res = sw.apply_sweeps(toy_core, sw.Sweep([a, None], context=2.0), OUTPUTS)
    assert res[0].dims == ("range", "azimuth")
    np.testing.assert_allclose(res[0].values, 2.0 * a.values)


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("one_by_one", [False, True])
def test_stack_over_two_extra_dimensions(chunked, one_by_one):
    dims = ("volume", "time", "azimuth", "range")
    a = field((2, 3, NRAY, NGATE), dims)
    b = field((3, NRAY, NGATE), ("time", "azimuth", "range"), seed=1)  # broadcast
    if chunked:
        pytest.importorskip("dask")
        a = a.chunk({"volume": 1, "time": 2, "azimuth": 4})
    sizes = []

    def core(batch):
        sizes.append(len(batch))
        return toy_core(batch)

    res = sw.apply_sweeps(core, sw.Sweep([a, b], context=2.0), OUTPUTS, one_by_one)
    assert res[0].dims == dims
    assert (res[1].dims, res[2].dims, res[3].dims) == (
        ("volume", "time", "azimuth"),
        ("volume", "time", "range"),
        ("volume", "time"),
    )
    assert res[0].dtype == np.float32
    want = expected(a.values, np.broadcast_to(b.values, a.shape))
    for got, ref in zip(res, want):
        np.testing.assert_allclose(got.compute().values, ref, rtol=1e-6)
    if chunked:
        assert_lazy(xr.Dataset({str(k): r for k, r in enumerate(res)}))
        assert res[0].chunks[-1] == (NGATE,) and res[0].chunks[-2] == (NRAY,)
    if one_by_one:
        assert set(sizes) == {1}


def test_none_field_in_the_middle_of_a_stack():
    a = field((2, NRAY, NGATE), ("time", "azimuth", "range"))
    seen = []

    def core(batch):
        seen.append([arrays[1] for _, arrays in batch])
        return [
            (arrays[0], arrays[0].sum(1), arrays[0].max(0), 0.0) for _, arrays in batch
        ]

    sw.apply_sweeps(core, sw.Sweep([a, None, a], context=1.0), OUTPUTS)
    assert seen == [[None, None]]


def test_empty_stack():
    a = field((0, NRAY, NGATE), ("time", "azimuth", "range"))
    res = sw.apply_sweeps(toy_core, sw.Sweep([a], context=2.0), OUTPUTS)
    assert res[0].shape == (0, NRAY, NGATE)
    assert res[3].shape == (0,)


def test_volume_of_plain_and_lazy_sweeps():
    pytest.importorskip("dask")
    a, b, c = field(seed=0), field(seed=1), field(seed=2)
    sizes = []

    def core(batch):
        sizes.append(len(batch))
        return toy_core(batch)

    sweeps = [
        sw.Sweep([a, None], context=1.0),
        sw.Sweep([b.chunk({"azimuth": 3}), None], context=2.0),
        sw.Sweep([c, None], context=3.0),
    ]
    res = sw.apply_sweeps_many(core, sweeps, OUTPUTS)
    assert sizes[0] == 2  # the two sweeps in memory go together
    assert res[1][0].chunks is not None
    assert res[0][0].chunks is None and res[2][0].chunks is None
    for r, x, ctx in zip(res, (a, b, c), (1.0, 2.0, 3.0)):
        np.testing.assert_allclose(r[0].compute().values, ctx * x.values)


def gate_core(batch):
    out = []
    for context, (a, *rest) in batch:
        both = a if not rest else a * rest[0]
        classes = np.repeat(both[..., None], 3, axis=-1) * np.arange(1, 4)
        out.append((context * both, classes))
    return out


GATE_OUTPUTS = (
    sw.Output(np.float32, "gate"),
    sw.Output(np.float64, "gate", new_dims={"class": 3}),
)


@pytest.mark.parametrize("chunked", [False, True])
def test_gates_broadcast_and_new_dimension(chunked):
    a = field((2, NRAY, NGATE), ("time", "azimuth", "range"))
    b = field((NRAY, NGATE), SWEEP, seed=1)
    if chunked:
        pytest.importorskip("dask")
        a = a.chunk({"time": 1, "range": 3})
    gates = sw.Gates([a, b], context=2.0)
    assert gates.plain is not chunked
    res = sw.apply_gates(gate_core, gates, GATE_OUTPUTS)
    want = a.values * b.values
    assert res[0].dims == a.dims and res[1].dims == (*a.dims, "class")
    np.testing.assert_allclose(res[0].compute().values, 2.0 * want, rtol=1e-6)
    np.testing.assert_allclose(
        res[1].compute().values, want[..., None] * np.arange(1, 4), rtol=1e-6
    )
    assert (res[0].chunks is not None) is chunked
    many = sw.apply_gates_many(
        gate_core, [gates, sw.Gates([b], context=1.0)], GATE_OUTPUTS
    )
    assert len(many) == 2 and many[1][0].dims == SWEEP


@pytest.mark.parametrize("chunked", [False, True])
def test_gates_along_a_core_dimension(chunked):
    z = field((4, 3, 5), ("time", "x", "z"))
    ceiling = field((4, 3), ("time", "x"), seed=1)
    if chunked:
        pytest.importorskip("dask")
        z = z.chunk({"time": 2, "z": 2})

    def core(batch):
        ((context, (values, limit)),) = batch
        assert values.shape[:-1] == limit.shape and values.shape[-1] == 5
        return [(values.sum(axis=-1) + limit,)]

    gates = sw.Gates([z, ceiling], core_dim="z")
    assert gates.dims == ("time", "x")
    (res,) = sw.apply_gates(core, gates, [sw.Output(np.float64, "gate")])
    assert res.dims == ("time", "x")
    np.testing.assert_allclose(res.compute().values, z.values.sum(-1) + ceiling.values)


def test_library_does_not_import_dask():
    code = (
        "import sys, radarx, radarx.retrieve, radarx.grid\n"
        "import xarray as xr, numpy as np\n"
        "from radarx.retrieve import llsd\n"
        "assert 'dask' not in sys.modules, 'dask was imported'\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


LABELS = {
    "Lazy, extra dimensions": "lazy",
    "Lazy": "lazy_only",
    "Computes": "computes",
    "Computes, cannot be lazy": "eager",
}


def documented():
    """Function names of the table in docs/usage.md by behaviour."""
    text = (Path(__file__).parents[1] / "docs" / "usage.md").read_text()
    found = {kind: set() for kind in LABELS.values()}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if line.startswith("|") and len(cells) == 3 and cells[0] in LABELS:
            module = cells[1].strip("`").replace("radarx.", "")
            names = re.findall(r"`(\w+)`", cells[2])
            found[LABELS[cells[0]]].update(f"{module}.{n}" for n in names)
    return found


def test_the_documentation_lists_the_declared_functions():
    declared = {kind: set() for kind in LABELS.values()}
    for key, (kind, _) in sw.DASK_SUPPORT.items():
        declared[kind].add(key)
    assert documented() == declared


def test_the_note_is_in_the_docstring():
    for key, (kind, why) in sw.DASK_SUPPORT.items():
        module, name = key.rsplit(".", 1)
        doc = getattr(importlib.import_module(f"radarx.{module}"), name).__doc__
        if kind in ("computes", "eager"):
            assert "computes the dask array" in doc, key
        else:
            assert "Dask arrays stay lazy" in doc, key
        assert len(re.findall(r"^ *Dask( and extra dimensions)?\. ", doc, re.M)) == 1


DOC = """Short.

    Parameters
    ----------
    x : int
        Something.

    Notes
    -----
    Existing note.

    Examples
    --------
    >>> pass
    """


def test_the_note_goes_to_the_notes_section():
    doc = sw._add_note(DOC, "Dask. New note.")
    assert doc.index("Existing note.") < doc.index("Dask. New note.")
    assert doc.index("Dask. New note.") < doc.index("Examples")
    assert doc.count("Notes\n") == 1


def test_a_missing_notes_section_is_made():
    no_notes = DOC.replace("Notes\n    -----\n    Existing note.\n\n    ", "")
    doc = sw._add_note(no_notes, "Dask. New note.")
    assert doc.index("Notes\n    -----\n    Dask. New note.") < doc.index("Examples")
    plain = sw._add_note(
        "Short.\n\n    Parameters\n    ----------\n    x : int\n", "Dask. N."
    )
    assert plain.rstrip().endswith("Dask. N.")
    bare = sw._add_note("Short.", "Dask. N.")
    assert "Notes\n    -----\n    Dask. N." in bare


def test_a_placeholder_ends_the_notes():
    doc = DOC.replace("Existing note.\n", "Existing note.\n    {references}\n")
    doc = sw._add_note(doc, "Dask. New note.")
    assert doc.index("Dask. New note.") < doc.index("{references}")


def test_dask_support_registers_and_checks():
    @sw.dask_support("lazy", "An exception.")
    def lazy(x):
        """Short.

        Notes
        -----
        Text.
        """

    @sw.dask_support("computes", "it needs all times.")
    def computes(x):
        """Short."""

    assert "An exception." in lazy.__doc__
    assert "computes the dask array: it needs all times." in computes.__doc__
    key = f"{__name__.replace('radarx.', '')}.lazy"
    assert sw.DASK_SUPPORT.pop(key)[0] == "lazy"
    sw.DASK_SUPPORT.pop(key.replace(".lazy", ".computes"))
    for kind, why in (("never", ""), ("computes", ""), ("eager", "")):
        with pytest.raises(ValueError):
            sw.dask_support(kind, why)
