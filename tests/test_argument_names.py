#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the argument conventions
==================================

One name per concept in every public function: ``obj`` for the first
argument, ``dbz`` for reflectivity, ``field`` for the radial velocity,
``min_<quantity>`` for thresholds, ``n_threads`` before ``engine`` (both
keyword-only). The accessor methods show the signature of the function they
call.
"""
import importlib
import inspect
import pkgutil

import pytest

import radarx
from radarx import _compat
from radarx._registry import method_signature

FIRST_ARGUMENT_OLD = {"ds", "dtree", "radar", "volume", "grid"}
# spellings that were replaced; none may be a parameter of a public function
OLD_NAMES = {
    "dbzh",
    "dbz_min",
    "rhohv_min",
    "rho_min",
    "snr_min",
    "zdr_min",
    "kdp_min",
    "beam_width",
    "min_reflectivity",
    "vrad",
}
# accessor methods that do not forward to one function
OWN_SIGNATURE = {"assign", "background", "interpolate_profile", "grid_radars"}
# the sounding accessor passes provider options on to the readers
WITH_KWARGS = {"sounding"}


def _public_functions():
    seen = {}
    for info in pkgutil.walk_packages(radarx.__path__, "radarx."):
        if info.name.rsplit(".", 1)[-1].startswith("_"):
            continue
        try:
            module = importlib.import_module(info.name)
        except ImportError:  # pragma: no cover - optional dependency
            continue
        for name in getattr(module, "__all__", ()):
            obj = getattr(module, name, None)
            if inspect.isfunction(obj):
                seen[f"{obj.__module__}.{obj.__name__}"] = obj
    return seen


FUNCTIONS = _public_functions()


def _params(func):
    return list(inspect.signature(func).parameters.values())


def test_the_audit_sees_the_public_api():
    assert len(FUNCTIONS) > 250


@pytest.mark.parametrize("key", sorted(FUNCTIONS))
def test_first_argument_is_not_a_container_nickname(key):
    params = _params(FUNCTIONS[key])
    if params:
        assert params[0].name not in FIRST_ARGUMENT_OLD, key


@pytest.mark.parametrize("key", sorted(FUNCTIONS))
def test_replaced_spellings_are_gone(key):
    names = {p.name for p in _params(FUNCTIONS[key])}
    assert not names & OLD_NAMES, key


@pytest.mark.parametrize("key", sorted(FUNCTIONS))
def test_n_threads_comes_before_engine_and_both_are_keyword_only(key):
    params = _params(FUNCTIONS[key])
    names = [p.name for p in params]
    for name in ("engine", "n_threads"):
        if name in names:
            kind = params[names.index(name)].kind
            assert kind is inspect.Parameter.KEYWORD_ONLY, f"{key}: {name}"
    if "engine" in names and "n_threads" in names:
        assert names.index("n_threads") < names.index("engine"), key


def test_every_alias_is_a_current_argument():
    assert len(_compat.RENAMED) > 90
    for qualified, old, new in _compat.RENAMED:
        if qualified.startswith("test"):  # the probes of the decorator tests
            continue
        module, _, name = qualified.rpartition(".")
        func = getattr(importlib.import_module(module), name)
        names = [p.name for p in _params(func)]
        assert new in names, qualified
        assert old not in names, qualified
        assert func.__radarx_renames__[old] == new


def _accessor_methods():
    classes = (
        radarx.accessors.RadarxDataArrayAccessor,
        radarx.accessors.RadarxDataSetAccessor,
        radarx.accessors.RadarxDataTreeAccessor,
    )
    for cls in classes:
        for name, member in inspect.getmembers(cls, inspect.isfunction):
            if not name.startswith("_"):
                yield cls.__name__, name, member


ACCESSORS = list(_accessor_methods())


def test_the_audit_sees_the_accessor_methods():
    assert len(ACCESSORS) > 150


@pytest.mark.parametrize("cls, name, member", ACCESSORS)
def test_accessor_methods_show_the_signature_of_their_function(cls, name, member):
    signature = inspect.signature(member)
    if name in OWN_SIGNATURE or name in WITH_KWARGS:
        kwargs = [
            p
            for p in signature.parameters.values()
            if p.kind is inspect.Parameter.VAR_KEYWORD
        ]
        assert bool(kwargs) == (name in WITH_KWARGS), f"{cls}.{name}"
        return
    target = member.__radarx_target__
    assert signature == method_signature(target), f"{cls}.{name}"
    assert list(signature.parameters)[0] == "self"
    # the same names as the function: nothing is lost or renamed
    assert list(signature.parameters)[1:] == [p.name for p in _params(target)][1:]


def test_accessor_methods_call_the_function_with_the_wrapped_object():
    from radarx.retrieve import vil

    method = radarx.accessors.RadarxDataSetAccessor.vil
    assert method.__radarx_target__ is vil
    assert method.__name__ == "vil"
    assert method.__doc__ and "radarx.retrieve.vil" in method.__doc__


def test_function_and_accessor_agree_on_the_threshold_names():
    method = radarx.accessors.RadarxDataSetAccessor.kdp
    names = list(inspect.signature(method).parameters)
    assert "dbz" in names and "min_rhohv" in names
    assert "dbzh" not in names and "rhohv_min" not in names
