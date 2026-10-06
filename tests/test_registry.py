#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for the accessor-method registry."""

import pytest
import xarray as xr

import radarx  # noqa: F401
from radarx import _registry, accessors


@pytest.fixture
def clean_registry(monkeypatch):
    monkeypatch.setattr(_registry, "_methods", {k: {} for k in _registry.KINDS})


def test_register_and_attach(clean_registry, monkeypatch):
    @_registry.accessor_method("dataset", "datatree", name="double_it")
    def _double(self, var):
        """Double one variable."""
        return self.xarray_obj[var] * 2

    assert set(_registry.registered("dataset")) == {"double_it"}
    assert _registry.registered("dataarray") == {}
    for cls in (accessors.RadarxDataSetAccessor, accessors.RadarxDataTreeAccessor):
        monkeypatch.delattr(cls, "double_it", raising=False)
    accessors._attach_registered_methods()
    ds = xr.Dataset({"a": ("x", [1.0, 2.0])})
    xr.testing.assert_equal(ds.radarx.double_it("a"), ds["a"] * 2)
    assert accessors.RadarxDataSetAccessor.double_it.__doc__ == "Double one variable."
    for cls in (accessors.RadarxDataSetAccessor, accessors.RadarxDataTreeAccessor):
        delattr(cls, "double_it")


def test_register_errors(clean_registry):
    with pytest.raises(ValueError, match="kinds"):
        _registry.accessor_method()
    with pytest.raises(ValueError, match="kinds"):
        _registry.accessor_method("grid")

    @_registry.accessor_method("dataset")
    def once(self):
        pass

    with pytest.raises(ValueError, match="already registered"):
        _registry.accessor_method("dataset", name="once")(lambda self: None)


def test_attach_refuses_to_shadow_methods(clean_registry):
    _registry.accessor_method("dataset", name="assign")(lambda self: None)
    with pytest.raises(ValueError, match="already defines 'assign'"):
        accessors._attach_registered_methods()
