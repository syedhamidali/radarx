#!/usr/bin/env python

"""Tests for `radarx` package."""

import radarx
import pytest
from unittest import mock


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_version_import_failure():
    """Test to handle the failure of importing radarx.version."""
    # Simulate the failure of importing radarx.version
    with mock.patch.dict("sys.modules", {"radarx.version": None}):
        # Reload radarx to trigger the exception handling in __init__.py
        import importlib

        importlib.reload(radarx)
        assert radarx.__version__ == "999"
