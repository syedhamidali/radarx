#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Test configuration shared by the whole suite."""

# open-radar-data files that several test modules read, and the one the QVP
# time-series test starts from
SHARED_DATA = (
    "KLBB20160601_150025_V06",
    "swx_20120520_0641.nc",
    "corcsapr2cmacppiM1.c1.20181111.030003.nc",
    "RAW_NA_000_125_20080411181219",
)
RETRIES = 5


def pytest_configure(config):
    """Make the test-data downloads robust.

    The files come from GitHub, whose raw endpoint answers 504 (gateway
    time-out) when many workers ask at once, and on Windows two workers that
    fetch the same missing file at the same time fail (the temporary download
    cannot replace a file another worker has just created or still holds open).
    Every process therefore retries failed downloads, and the pytest-xdist
    controller fetches the shared files once, before the workers start. A
    download that still fails is left to the tests that need the file.
    """
    try:
        from open_radar_data import DATASETS
    except ImportError:
        return
    DATASETS.retry_if_failed = RETRIES
    if hasattr(config, "workerinput"):  # an xdist worker
        return
    for name in SHARED_DATA:
        try:
            DATASETS.fetch(name)
        except Exception:  # offline or server error: the tests report it
            pass
    try:
        from radarx.testing.test_data_imd import fetch_imd_test_data

        fetch_imd_test_data()
    except Exception:  # offline or server error: the tests report it
        pass


# Tests that take half a minute or more on a CI runner (measured with --durations): real
# volumes, Monte Carlo error estimates and the Bayesian DSD retrieval. They run in the full
# CI job (Linux, Python 3.12, NumPy 2); the other jobs set RADARX_SKIP_SLOW=1 ("0" or empty runs them) and skip them,
# which cuts their time by a third while each platform and NumPy version still runs the rest.
# An entry matches the tests whose id starts with it (so parametrized tests are included).
SLOW_TESTS = (
    "tests/test_dsd_bayes.py::test_real_nexrad_sweep",
    "tests/test_dtype_units.py::test_fields_are_float32_and_units_are_udunits[float64-dsd_bayesian]",
    "tests/test_dtype_units.py::test_fields_are_float32_and_units_are_udunits[float32-dsd_bayesian]",
    "tests/test_single_doppler.py::test_uncertainty_follows_the_input_errors",
    "tests/test_single_doppler.py::test_uncertainty_matches_the_error_on_synthetic_fields",
    "tests/test_single_doppler.py::test_uncertainty_is_reproducible_and_leaves_inputs_alone",
    "tests/test_single_doppler.py::test_real_volume",
    "tests/test_multidoppler.py::test_error_estimate_matches_the_inverse_hessian",
    "tests/test_multidoppler.py::test_lbfgsb_and_cg_reach_the_same_minimum",
    "tests/test_multidoppler.py::test_shift_winds_beats_advecting_the_radial_velocity",
    "tests/test_multidoppler.py::test_real_dual_doppler",
    "tests/test_unravel.py::test_options_and_variants",
    "tests/test_superobs.py::test_wrfda_round_trip",
    "tests/test_qvp.py::test_real_timeseries_melting_layer",
    "tests/test_ensemble.py::test_spread_matches_the_actual_error",
    "tests/test_tornado.py::test_tornet_inputs_dealias_and_kdp",
)


def pytest_collection_modifyitems(config, items):
    """Skip the slow tests when RADARX_SKIP_SLOW is set (see SLOW_TESTS)."""
    import os

    import pytest

    if os.environ.get("RADARX_SKIP_SLOW", "") in ("", "0"):
        return
    skip = pytest.mark.skip(
        reason="slow test: runs in the full CI job (RADARX_SKIP_SLOW)"
    )
    for item in items:
        if item.nodeid.startswith(SLOW_TESTS):
            item.add_marker(skip)
