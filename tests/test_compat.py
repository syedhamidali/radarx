#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the deprecated argument names and keyword-only arguments
===================================================================

The decorators of :mod:`radarx._compat` are tested on small functions, then
the old names are used on real functions: the result must equal the result
with the new name, and a ``DeprecationWarning`` must name both.
"""
import warnings

import numpy as np
import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx import _compat
from radarx._compat import keyword_only_after, renamed_arguments
from radarx.retrieve import (
    create_cappi,
    echo_mask,
    estimate_kdp,
    rotation_couplets,
)

from .provenance_helpers import without_provenance


@renamed_arguments(dbzh="dbz", rhohv_min="min_rhohv")
def _probe(obj, dbz=None, *, min_rhohv=0.9):
    return obj, dbz, min_rhohv


def test_old_keyword_maps_to_the_new_one_with_a_warning():
    with pytest.warns(DeprecationWarning, match="'dbzh'.*'dbz'") as record:
        out = _probe(1, dbzh="Z")
    assert out == (1, "Z", 0.9)
    assert "_probe()" in str(record[0].message)
    assert record[0].filename == __file__  # points at the caller


def test_several_old_keywords():
    with pytest.warns(DeprecationWarning) as record:
        out = _probe(1, dbzh="Z", rhohv_min=0.5)
    assert out == (1, "Z", 0.5)
    assert len(record) == 2


def test_new_names_and_positions_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _probe(1, "Z", min_rhohv=0.1) == (1, "Z", 0.1)
        assert _probe(1) == (1, None, 0.9)


def test_old_and_new_name_together_is_an_error():
    with pytest.raises(TypeError, match="'dbz' and its deprecated name 'dbzh'"):
        _probe(1, dbz="a", dbzh="b")


def test_signature_has_the_new_names_only():
    import inspect

    names = list(inspect.signature(_probe).parameters)
    assert names == ["obj", "dbz", "min_rhohv"]
    assert _probe.__radarx_renames__ == {"dbzh": "dbz", "rhohv_min": "min_rhohv"}
    assert (f"{__name__}._probe", "dbzh", "dbz") in _compat.RENAMED


def test_rename_must_fit_the_signature():
    with pytest.raises(ValueError, match="cannot rename 'a' to 'z'"):

        @renamed_arguments(a="z")
        def f(a, b):  # pragma: no cover
            return a

    with pytest.raises(ValueError, match="cannot rename 'b' to 'a'"):

        @renamed_arguments(b="a")
        def g(a, b):  # pragma: no cover
            return a


@keyword_only_after("n_threads", "tol")
def _kwonly(a, b=2, *, tol=0.1, n_threads=None):
    return a, b, tol, n_threads


def test_positional_use_of_a_keyword_only_argument_warns_and_works():
    with pytest.warns(DeprecationWarning, match="n_threads, tol as positional"):
        assert _kwonly(1, 2, 4, 0.5) == (1, 2, 0.5, 4)
    with pytest.warns(DeprecationWarning, match="n_threads as positional"):
        assert _kwonly(1, 2, 4) == (1, 2, 0.1, 4)


def test_keyword_use_does_not_warn_and_too_many_arguments_fail():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _kwonly(1, 2, tol=0.3, n_threads=2) == (1, 2, 0.3, 2)
    with pytest.raises(TypeError):
        with pytest.warns(DeprecationWarning):
            _kwonly(1, 2, 3, 4, 5)
    with pytest.raises(TypeError, match="multiple values for 'n_threads'"):
        _kwonly(1, 2, 3, n_threads=4)


# ---- real functions ---------------------------------------------------------


def _sweep(nray=40, ngate=120):
    rng = np.random.default_rng(0)
    dims = ("azimuth", "range")
    kdp = np.full((nray, ngate), 1.0)
    phi = 30.0 + np.cumsum(2.0 * kdp * 0.25, axis=1) + rng.normal(0, 0.3, kdp.shape)
    return xr.Dataset(
        {
            "DBZH": (dims, rng.uniform(20, 50, kdp.shape), {"units": "dBZ"}),
            "RHOHV": (dims, np.full(kdp.shape, 0.99), {"units": "1"}),
            "PHIDP": (dims, phi, {"units": "degree"}),
            "VRADH": (dims, rng.uniform(-10, 10, kdp.shape), {"units": "m s-1"}),
        },
        coords={
            "azimuth": ("azimuth", np.linspace(0.5, 359.5, nray)),
            "range": ("range", 500.0 + 250.0 * np.arange(ngate)),
            "elevation": ("azimuth", np.full(nray, 0.5)),
        },
    )


def test_estimate_kdp_old_names():
    ds = _sweep()
    new = estimate_kdp(ds, dbz="DBZH", min_rhohv=0.9)
    with pytest.warns(DeprecationWarning) as record:
        old = estimate_kdp(ds, dbzh="DBZH", rhohv_min=0.9)
    assert len(record) == 2
    xr.testing.assert_identical(without_provenance(new), without_provenance(old))


def test_accessor_passes_the_old_name_on():
    ds = _sweep()
    with pytest.warns(DeprecationWarning, match="'dbzh'"):
        old = ds.radarx.kdp(dbzh="DBZH")
    xr.testing.assert_identical(
        without_provenance(ds.radarx.kdp(dbz="DBZH")), without_provenance(old)
    )


def test_echo_mask_old_velocity_name():
    ds = _sweep()
    with pytest.warns(DeprecationWarning, match="'vrad'.*'field'"):
        old = echo_mask(ds, method="krause2016", vrad="VRADH", nodata=None)
    new = echo_mask(ds, method="krause2016", field="VRADH", nodata=None)
    xr.testing.assert_identical(without_provenance(new), without_provenance(old))


def test_first_argument_and_list_old_names():
    ds = _sweep()
    tree = xr.DataTree.from_dict({"/": xr.Dataset(), "sweep_0": ds})
    with pytest.warns(DeprecationWarning, match="'radar'.*'obj'"):
        with pytest.raises(ValueError, match="georeferenc"):
            create_cappi(radar=tree, height=1000.0)
    with pytest.warns(DeprecationWarning, match="'fields'.*'data_vars'"):
        with pytest.raises(ValueError, match="georeferenc"):
            create_cappi(tree, 1000.0, fields=["DBZH"])


def test_rotation_couplets_old_threshold_name():
    ds = _sweep()
    ds = ds.assign(azimuthal_shear=ds.VRADH * 1e-3)
    azimuth = np.deg2rad(ds.azimuth)
    ds = ds.assign_coords(x=ds.range * np.sin(azimuth), y=ds.range * np.cos(azimuth))
    with pytest.warns(DeprecationWarning, match="'min_reflectivity'.*'min_dbz'"):
        old = rotation_couplets(ds, min_reflectivity=30.0)
    new = rotation_couplets(ds, min_dbz=30.0)
    assert old.sizes == new.sizes


def test_hitschfeld_bordan_use_compiled_is_deprecated():
    from radarx.fundamentals.reflectivity import hitschfeld_bordan_correction

    dbz = np.full((3, 50), 30.0)
    rng = 500.0 + 250.0 * np.arange(50)
    kw = dict(a=3e-4, b=0.75)
    want = hitschfeld_bordan_correction(dbz, rng, engine="numpy", **kw)
    with pytest.warns(DeprecationWarning, match="use_compiled"):
        got = hitschfeld_bordan_correction(dbz, rng, use_compiled=False, **kw)
    np.testing.assert_array_equal(got, want)
    with pytest.warns(DeprecationWarning):
        auto = hitschfeld_bordan_correction(dbz, rng, use_compiled=True, **kw)
    np.testing.assert_allclose(auto, want, rtol=1e-12)
    with pytest.raises(ValueError, match="engine must be"):
        hitschfeld_bordan_correction(dbz, rng, engine="gpu", **kw)


def test_hitschfeld_bordan_compiled_engine_needs_the_kernel(monkeypatch):
    from radarx.fundamentals import reflectivity

    monkeypatch.setattr(reflectivity, "HAS_COMPILED_KERNEL", False)
    with pytest.raises(ImportError, match="not available"):
        reflectivity.hitschfeld_bordan_correction(
            np.full((2, 10), 30.0),
            np.arange(1, 11) * 250.0,
            a=3e-4,
            b=0.75,
            engine="compiled",
        )


def test_grid_radar_positional_options_are_deprecated():
    from radarx.grid import grid_radar

    tree = xr.DataTree.from_dict({"/": xr.Dataset(), "sweep_0": _sweep(8, 20)})
    sizes = (-1e3, 1e3), (-1e3, 1e3), (0, 1e3), 1000, 1000, 250, 0.2, 0.2, 1
    # a method that is checked first, so the call stops after the decorator
    with pytest.warns(DeprecationWarning, match="n_threads, fill_depth as positional"):
        with pytest.raises(ValueError, match="method must be"):
            grid_radar(tree, None, True, *sizes, "nope", 1, 100.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="method must be"):
            grid_radar(tree, None, True, *sizes, "nope", n_threads=1)
