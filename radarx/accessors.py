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
from .vis import plot_maxcappi  # noqa


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
        self, xarray_obj: xr.Dataset | xr.DataArray | xr.DataTree
    ) -> RadarxAccessor:
        self.xarray_obj = xarray_obj


@xr.register_dataset_accessor("radarx")
class RadarxDataSetAccessor(RadarxAccessor):
    """Adds a number of radarx specific methods to xarray.DataArray objects."""

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
    ) -> xr.DataSet:
        """Plot Max Cappi"""
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


@xr.register_datatree_accessor("radarx")
class RadarxDataTreeAccessor(RadarxAccessor):
    """Adds a number of radarx specific methods to xarray.DataTree objects."""

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
