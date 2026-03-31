#!/usr/bin/env python
# Copyright (c) 2024, radarx developers.
# Distributed under the MIT License. See LICENSE for more info.
# Most of the functions are borrowed from Xradar

"""
Radarx Accessors
================

To extend :py:class:`xarray:xarray.DataArray` and  :py:class:`xarray:xarray.Dataset`
radarx provides accessors which downstream libraries can hook into.

This module contains the functionality to create those accessors.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
from __future__ import annotations  # noqa: F401

__all__ = ["create_radarx_dataarray_accessor"]

__doc__ = __doc__.format("\n   ".join(__all__))

import xarray as xr
from .grid import grid_radar  # noqa
from .retrieve import create_cappi as retrieve_cappi  # noqa
from .vis import plot_cappi, plot_ppi, plot_rhi  # noqa

try:  # pragma: no cover
    from xarray import DataTree as RadarxDataTreeType

    register_datatree_accessor = xr.register_datatree_accessor
except (ImportError, AttributeError):  # pragma: no cover
    from datatree import DataTree as RadarxDataTreeType
    from datatree import register_datatree_accessor


def accessor_constructor(self, xarray_obj):  # pragma: no cover
    self._obj = xarray_obj  # pragma: no cover


def create_function(func):  # pragma: no cover
    def function(self):
        return func(self._obj)  # pragma: no cover

    return function  # pragma: no cover


def create_methods(funcs):  # pragma: no cover
    methods = {}
    for name, func in funcs.items():
        methods[name] = create_function(func)
    return methods  # pragma: no cover


def create_radarx_dataarray_accessor(name, funcs):  # pragma: no cover
    methods = {"__init__": accessor_constructor} | create_methods(funcs)
    cls_name = "".join([name.capitalize(), "Accessor"])
    accessor = type(cls_name, (object,), methods)
    return xr.register_dataarray_accessor(name)(accessor)  # pragma: no cover


class RadarxAccessor:
    """
    Common Datatree, Dataset, DataArray accessor functionality.
    """

    def __init__(
        self, xarray_obj: xr.Dataset | xr.DataArray | RadarxDataTreeType
    ) -> RadarxAccessor:
        self.xarray_obj = xarray_obj


@xr.register_dataset_accessor("radarx")
class RadarxDataSetAccessor(RadarxAccessor):
    """Dataset-level radarx plotting utilities."""

    def plot_max_cappi(
        self,
        data_var,
        cmap=None,
        vmin=None,
        vmax=None,
        title=None,
        lat_lines=None,
        lon_lines=None,
        add_map=True,
        projection=None,
        colorbar=True,
        range_rings=False,
        dpi=100,
        savedir=None,
        show_figure=True,
        add_slogan=False,
        **kwargs,
    ) -> xr.Dataset:
        """Plot a maximum CAPPI product from a 3D gridded radar dataset."""
        from .vis import plot_maxcappi

        radar = self.xarray_obj
        return radar.pipe(
            plot_maxcappi,
            data_var,
            cmap,
            vmin,
            vmax,
            title,
            lat_lines,
            lon_lines,
            add_map,
            projection,
            colorbar,
            range_rings,
            dpi,
            savedir,
            show_figure,
            add_slogan,
            **kwargs,
        )

    def plot_ppi(
        self,
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
    ) -> xr.Dataset:
        """Plot a georeferenced plan-position view using ``x`` and ``y``."""
        return self.xarray_obj.pipe(
            plot_ppi,
            data_var,
            cmap,
            vmin,
            vmax,
            title,
            colorbar,
            ax,
            dpi,
            savedir,
            show_figure,
            add_slogan,
            **kwargs,
        )

    def plot_rhi(
        self,
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
    ) -> xr.Dataset:
        """Plot a vertical cross-section using ground range and height."""
        return self.xarray_obj.pipe(
            plot_rhi,
            data_var,
            cmap,
            vmin,
            vmax,
            title,
            colorbar,
            ax,
            dpi,
            savedir,
            show_figure,
            add_slogan,
            **kwargs,
        )

    def plot_cappi(
        self,
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
    ) -> xr.Dataset:
        """Plot a CAPPI dataset on the horizontal plane."""
        return self.xarray_obj.pipe(
            plot_cappi,
            data_var,
            cmap,
            vmin,
            vmax,
            title,
            colorbar,
            ax,
            dpi,
            savedir,
            show_figure,
            add_slogan,
            **kwargs,
        )


@register_datatree_accessor("radarx")
class RadarxDataTreeAccessor(RadarxAccessor):
    """DataTree-level radarx retrieval and gridding utilities."""

    def to_grid(
        self,
        data_vars=None,
        pseudo_cappi=True,
        x_lim=(-100e3, 100e3),
        y_lim=(-100e3, 100e3),
        z_lim=(0, 10e3),
        x_step=1000,
        y_step=1000,
        z_step=250,
        x_smth=0.2,
        y_smth=0.2,
        z_smth=1,
    ):
        """Grid a georeferenced radar volume onto a Cartesian 3D domain."""
        dtree = grid_radar(
            self.xarray_obj,
            data_vars,
            pseudo_cappi,
            x_lim,
            y_lim,
            z_lim,
            x_step,
            y_step,
            z_step,
            x_smth,
            y_smth,
            z_smth,
        )
        return dtree

    def create_cappi(
        self,
        height,
        method="cartesian_idw",
        vertical_tolerance=None,
        apply_filter=False,
        *,
        fields=None,
        sweeps=None,
        x=None,
        y=None,
        x_res=1000.0,
        y_res=1000.0,
        padding=0.0,
    ):
        """
        Create a CAPPI from a georeferenced radar volume.

        Parameters
        ----------
        height : float
            Target CAPPI altitude in meters.
        method : {
            "cartesian_idw",
            "polar_vertical_interpolation",
            "height_window_composite",
        }, optional
            Retrieval algorithm to use. Legacy aliases ``"cartesian"``,
            ``"polar"``, and ``"pseudo_cappi"`` are accepted.
        vertical_tolerance : float or None, optional
            Maximum vertical distance above and below the requested CAPPI
            height, in meters, used by the selected retrieval method.
        apply_filter : bool, optional
            Apply built-in gate filtering when supported by the selected
            retrieval method.
        fields : list[str] or None, optional
            Radar variables to retrieve. If omitted, likely 2D radar fields are
            selected automatically.
        sweeps : list[str] or None, optional
            Sweep names to include. If omitted, all available sweep groups are
            used.
        x, y : array-like or None, optional
            Target Cartesian grid coordinates in meters. Used only with
            ``method="cartesian_idw"``.
        x_res, y_res : float, optional
            Cartesian output spacing in meters when ``x`` and ``y`` are not
            supplied. Used only with ``method="cartesian_idw"``.
        padding : float, optional
            Extra padding, in meters, applied to the Cartesian output domain.
            Used only with ``method="cartesian_idw"``.
        """
        return retrieve_cappi(
            self.xarray_obj,
            height=height,
            method=method,
            vertical_tolerance=vertical_tolerance,
            apply_filter=apply_filter,
            fields=fields,
            sweeps=sweeps,
            x=x,
            y=y,
            x_res=x_res,
            y_res=y_res,
            padding=padding,
        )

    def to_cappi(self, *args, **kwargs):
        """Convenience alias for :meth:`create_cappi`."""
        return self.create_cappi(*args, **kwargs)
