#!/usr/bin/env python
# Copyright (c) 2024-2025, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for Radarx Max-Cappi
==========================
"""
import pytest
from radarx.vis.maxcappi import plot_maxcappi
from open_radar_data import DATASETS
import xradar as xd


@pytest.fixture
def mock_dtree():
    """Fixture to create a mock radar DataTree."""
    file = DATASETS.fetch("swx_20120520_0641.nc")
    dtree = xd.io.open_cfradial1_datatree(file, sweep=[0, 1, 2, 3, 5, 7])
    return dtree


def test_plot_maxcappi_basic(mock_dtree, tmp_path):
    """Basic test for `plot_maxcappi`."""
    # Grid the radar data
    ds = mock_dtree.radarx.to_grid(
        data_vars=["corrected_reflectivity_horizontal"],
        pseudo_cappi=True,
        x_lim=(-50e3, 50e3),
        y_lim=(-50e3, 50e3),
        z_lim=(0, 5e3),
        x_step=2000,
        y_step=2000,
        z_step=1000,
    )

    # Create a valid directory for saving the plot
    save_dir = tmp_path / "plots"
    save_dir.mkdir()

    # Plot Max-CAPPI
    plot_maxcappi(
        ds=ds,
        data_var="corrected_reflectivity_horizontal",
        cmap="viridis",
        vmin=0,
        vmax=60,
        title="Test Max-CAPPI",
        add_map=False,
        colorbar=True,
        show_figure=False,
        savedir=str(save_dir),
    )

    # Dynamically construct the expected file name
    radar_name = ds.attrs.get("instrument_name", "Radar")
    time_str = ds["time"].dt.strftime("%Y%m%d%H%M%S").values.item()
    expected_file = save_dir / f"Test Max-CAPPI_{radar_name}_{time_str}.png"

    # Assert the plot was saved
    assert expected_file.exists(), f"Expected file {expected_file} was not created."


def test_plot_maxcappi_no_cartopy(monkeypatch):
    """Calling plot_maxcappi without cartopy raises a clear ImportError."""
    import radarx.vis.maxcappi as _mod

    monkeypatch.setattr(_mod, "_CARTOPY_AVAILABLE", False)
    with pytest.raises(ImportError, match="cartopy is required"):
        _mod.plot_maxcappi(None, "DBZH")


def test_import_radarx_does_not_require_cartopy(monkeypatch):
    """Importing radarx must not force a cartopy import."""
    import sys
    import types

    # Block cartopy by inserting a broken stub before any import
    stub = types.ModuleType("cartopy")
    stub.__spec__ = None

    def _raise(*a, **kw):
        raise ImportError("cartopy blocked for test")

    stub.__getattr__ = _raise
    monkeypatch.setitem(sys.modules, "cartopy", stub)
    monkeypatch.setitem(sys.modules, "cartopy.crs", None)
    monkeypatch.setitem(sys.modules, "cartopy.feature", None)
    monkeypatch.setitem(sys.modules, "cartopy.mpl", None)
    monkeypatch.setitem(sys.modules, "cartopy.mpl.gridliner", None)

    # Re-importing the vis.maxcappi module must succeed even with cartopy blocked
    import importlib

    import radarx.vis.maxcappi as _mod

    importlib.reload(_mod)
    assert not _mod._CARTOPY_AVAILABLE


if __name__ == "__main__":
    pytest.main(["-s", __file__])
