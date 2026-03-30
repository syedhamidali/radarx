#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx Plot Helpers
===================

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = ["plot_cappi", "plot_ppi", "plot_rhi"]

__doc__ = __doc__.format("\n   ".join(__all__))

import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _get_dataarray(ds, data_var):
    if isinstance(ds, xr.DataArray):
        return ds
    if data_var is None:
        raise ValueError("data_var must be provided when plotting an xarray.Dataset.")
    return ds[data_var]


def _get_figure_and_axis(ax=None, figsize=(7, 6)):
    if ax is not None:
        return ax.figure, ax
    return plt.subplots(figsize=figsize)


def _get_coord(da, name):
    if name in da.coords:
        return np.asarray(da.coords[name].values, dtype=float)
    raise ValueError(f"Coordinate '{name}' is required for this plot.")


def _get_time_token(ds):
    if "time" not in ds:
        return "unknown_time"

    value = np.asarray(ds["time"].values).reshape(-1)[0]
    value_da = xr.DataArray(value)
    if np.issubdtype(value_da.dtype, np.datetime64):
        return value_da.dt.strftime("%Y%m%d%H%M%S").item()
    return str(value)


def _get_radar_name(ds):
    radar_name = (
        ds.attrs.get("instrument_name") or ds.attrs.get("radar_name") or "Radar"
    )
    if isinstance(radar_name, bytes):
        radar_name = radar_name.decode()
    return str(radar_name)


def _finalize_plot(
    fig, ax, ds, title, savedir=None, dpi=100, show_figure=True, add_slogan=False
):
    if add_slogan:
        fig.text(
            0.5,
            0.02,
            "Plot by Radarx | Powered by Xradar",
            fontsize=9,
            fontname="Courier New",
            ha="center",
        )

    if savedir:
        filename = f"{title}_{_get_radar_name(ds)}_{_get_time_token(ds)}.png"
        filepath = os.path.join(savedir, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")

    if show_figure:
        plt.show()
    else:
        plt.close(fig)

    return ax


def _plot_plan_view(
    ds,
    data_var,
    title,
    cmap=None,
    vmin=None,
    vmax=None,
    colorbar=True,
    ax=None,
    dpi=100,
    savedir=None,
    show_figure=True,
    add_slogan=False,
    **kwargs,
):
    da = _get_dataarray(ds, data_var)
    x = _get_coord(da, "x") / 1000.0
    y = _get_coord(da, "y") / 1000.0

    kwargs.setdefault("shading", "nearest")

    fig, ax = _get_figure_and_axis(ax=ax)
    mesh = ax.pcolormesh(
        x, y, np.asarray(da.values), cmap=cmap, vmin=vmin, vmax=vmax, **kwargs
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title(title)

    if colorbar:
        colorbar_obj = fig.colorbar(mesh, ax=ax)
        colorbar_obj.set_label(da.attrs.get("units", ""))

    return _finalize_plot(
        fig,
        ax,
        ds if isinstance(ds, xr.Dataset) else da.to_dataset(name=da.name or data_var),
        title,
        savedir=savedir,
        dpi=dpi,
        show_figure=show_figure,
        add_slogan=add_slogan,
    )


def plot_ppi(
    ds,
    data_var,
    cmap=None,
    vmin=None,
    vmax=None,
    title=None,
    colorbar=True,
    ax=None,
    dpi=100,
    savedir=None,
    show_figure=True,
    add_slogan=False,
    **kwargs,
):
    """
    Plot a georeferenced plan-position indicator (PPI)-style view.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Sweep dataset containing 2D data on georeferenced ``x`` and ``y``
        coordinates.
    data_var : str
        Variable name to plot when ``ds`` is a dataset.
    """
    title = title or f"PPI {data_var}"
    return _plot_plan_view(
        ds,
        data_var,
        title=title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        colorbar=colorbar,
        ax=ax,
        dpi=dpi,
        savedir=savedir,
        show_figure=show_figure,
        add_slogan=add_slogan,
        **kwargs,
    )


def plot_cappi(
    ds,
    data_var,
    cmap=None,
    vmin=None,
    vmax=None,
    title=None,
    colorbar=True,
    ax=None,
    dpi=100,
    savedir=None,
    show_figure=True,
    add_slogan=False,
    **kwargs,
):
    """
    Plot a CAPPI dataset on the horizontal ``x``/``y`` plane.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        CAPPI dataset containing horizontal ``x`` and ``y`` coordinates.
    data_var : str
        Variable name to plot when ``ds`` is a dataset.
    """
    height = None
    if "z" in ds:
        height = float(np.asarray(ds["z"].values).reshape(-1)[0]) / 1000.0
    title = title or (
        f"CAPPI {height:.1f} km {data_var}"
        if height is not None
        else f"CAPPI {data_var}"
    )
    return _plot_plan_view(
        ds,
        data_var,
        title=title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        colorbar=colorbar,
        ax=ax,
        dpi=dpi,
        savedir=savedir,
        show_figure=show_figure,
        add_slogan=add_slogan,
        **kwargs,
    )


def plot_rhi(
    ds,
    data_var,
    cmap=None,
    vmin=None,
    vmax=None,
    title=None,
    colorbar=True,
    ax=None,
    dpi=100,
    savedir=None,
    show_figure=True,
    add_slogan=False,
    **kwargs,
):
    """
    Plot a range-height indicator (RHI)-style vertical section.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Sweep dataset containing georeferenced ``x``, ``y``, and ``z``
        coordinates.
    data_var : str
        Variable name to plot when ``ds`` is a dataset.
    """
    da = _get_dataarray(ds, data_var)
    x = _get_coord(da, "x")
    y = _get_coord(da, "y")
    z = _get_coord(da, "z")

    ground_range = np.hypot(x, y) / 1000.0
    height = z / 1000.0
    kwargs.setdefault("shading", "auto")

    fig, ax = _get_figure_and_axis(ax=ax)
    mesh = ax.pcolormesh(
        ground_range,
        height,
        np.asarray(da.values),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        **kwargs,
    )
    ax.set_xlabel("Ground Range (km)")
    ax.set_ylabel("Height (km)")
    ax.set_title(title or f"RHI {data_var}")

    if colorbar:
        colorbar_obj = fig.colorbar(mesh, ax=ax)
        colorbar_obj.set_label(da.attrs.get("units", ""))

    return _finalize_plot(
        fig,
        ax,
        ds if isinstance(ds, xr.Dataset) else da.to_dataset(name=da.name or data_var),
        title or f"RHI {data_var}",
        savedir=savedir,
        dpi=dpi,
        show_figure=show_figure,
        add_slogan=add_slogan,
    )
