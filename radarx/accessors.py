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

from .grid import (
    grid_radar,  # noqa
    to_uxarray,  # noqa
)
from .retrieve import advect as retrieve_advect  # noqa
from .retrieve import create_cappi as retrieve_cappi  # noqa
from .retrieve import (  # noqa
    dealias_velocity,
    estimate_kdp,  # noqa
    melting_layer,
    qvp,
)
from .retrieve import dsd as retrieve_dsd  # noqa
from .retrieve import estimate_motion as retrieve_estimate_motion  # noqa
from .retrieve import interpolate_time as retrieve_interpolate_time  # noqa
from .retrieve import shear as _shear
from .vis import plot_cappi, plot_ppi, plot_rhi  # noqa
from .vis.interactive import (
    RadarxDataArrayPlotAccessor,
    RadarxDatasetPlotAccessor,
    RadarxDataTreePlotAccessor,
)

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


class _ShearMixin:
    """Azimuthal shear and radial divergence (LLSD) for sweeps and volumes."""

    def llsd(self, field="VRADH", window=(750.0, 2500.0), **kwargs):
        """
        Azimuthal shear and radial divergence by linear least-squares derivatives.

        Parameters
        ----------
        field : str, optional
            Dealiased radial velocity field. Default ``"VRADH"``.
        window : tuple of float, optional
            Window ``(range_m, azimuth_m)`` in metres. Default ``(750, 2500)``.
        **kwargs
            ``weights``, ``min_valid_fraction``, ``mask``, ``n_threads`` and
            ``engine``, see :func:`radarx.retrieve.llsd`.

        Returns
        -------
        xarray.Dataset or xarray.DataTree
            ``azimuthal_shear`` and ``radial_divergence`` in s⁻¹.

        See Also
        --------
        radarx.retrieve.llsd
        """
        return _shear.llsd(self.xarray_obj, field, window, **kwargs)

    def azimuthal_shear(self, field="VRADH", window=(750.0, 2500.0), **kwargs):
        """
        Azimuthal shear (s⁻¹) of the radial velocity by LLSD.

        Same parameters as :meth:`llsd`.

        Returns
        -------
        xarray.DataArray or xarray.DataTree
            Azimuthal shear on the sweep's coordinates.

        See Also
        --------
        radarx.retrieve.azimuthal_shear
        """
        return _shear.azimuthal_shear(self.xarray_obj, field, window, **kwargs)

    def radial_divergence(self, field="VRADH", window=(750.0, 2500.0), **kwargs):
        """
        Radial divergence (s⁻¹) of the radial velocity by LLSD.

        Same parameters as :meth:`llsd`.

        Returns
        -------
        xarray.DataArray or xarray.DataTree
            Radial divergence on the sweep's coordinates.

        See Also
        --------
        radarx.retrieve.radial_divergence
        """
        return _shear.radial_divergence(self.xarray_obj, field, window, **kwargs)


@xr.register_dataarray_accessor("radarx")
class RadarxDataArrayAccessor(RadarxAccessor):
    """DataArray-level radarx utilities."""

    @property
    def plot(self) -> RadarxDataArrayPlotAccessor:
        """
        Interactive hvplot-based plots, e.g. ``da.radarx.plot.ppi()``.

        Returns
        -------
        radarx.vis.interactive.RadarxDataArrayPlotAccessor
            Plot accessor; requires the optional hvplot dependencies.

        See Also
        --------
        radarx.vis.interactive
        """
        return RadarxDataArrayPlotAccessor(self.xarray_obj)

    def to_uxarray(self, variables=None):
        """
        Convert the sweep into a uxarray dataset with one face per gate.

        Parameters
        ----------
        variables : str or list of str, optional
            Variables to attach to the faces. By default all
            ``(azimuth, range)`` variables.

        Returns
        -------
        uxarray.UxDataset
            Dataset on the ``n_face`` dimension with a UGRID grid.

        See Also
        --------
        radarx.grid.to_uxarray
        """
        return to_uxarray(self.xarray_obj, variables=variables)

    def estimate_motion(self, other, field=None, **kwargs):
        """
        Storm motion from this gridded volume to a later one.

        Parameters
        ----------
        other : xarray.Dataset or xarray.DataArray
            The later gridded volume on the same grid.
        field : str, optional
            Field to track (Dataset only). Default: the reflectivity.
        **kwargs
            Passed to :func:`radarx.retrieve.estimate_motion`.

        Returns
        -------
        xarray.Dataset
            ``u`` and ``v`` storm motion (m/s) and the correlation ``quality``.

        See Also
        --------
        radarx.retrieve.estimate_motion
        """
        return retrieve_estimate_motion(self.xarray_obj, other, field, **kwargs)

    def advect(self, u, v=None, dt=None, **kwargs):
        """
        Move the gridded fields along the storm motion.

        Parameters
        ----------
        u : float, xarray.DataArray or xarray.Dataset
            Eastward motion (m/s), or the result of :meth:`estimate_motion`.
        v : float or xarray.DataArray, optional
            Northward motion (m/s).
        dt : float or timedelta, optional
            Time step in seconds (or pass ``time=`` as a target time).
        **kwargs
            Passed to :func:`radarx.retrieve.advect`.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            The advected fields.

        See Also
        --------
        radarx.retrieve.advect
        """
        return retrieve_advect(self.xarray_obj, u, v, dt, **kwargs)

    def interpolate_time(self, other, times, motion=None, **kwargs):
        """
        Advection-corrected time interpolation to a later gridded volume.

        Parameters
        ----------
        other : xarray.Dataset or xarray.DataArray
            The later gridded volume on the same grid.
        times : datetime-like or array-like
            Target times between the two volumes.
        motion : xarray.Dataset, optional
            Storm motion; estimated from the two volumes by default.
        **kwargs
            Passed to :func:`radarx.retrieve.interpolate_time`.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Fields with a new ``time`` dimension.

        See Also
        --------
        radarx.retrieve.interpolate_time
        """
        return retrieve_interpolate_time(
            self.xarray_obj, other, times, motion, **kwargs
        )


@xr.register_dataset_accessor("radarx")
class RadarxDataSetAccessor(_ShearMixin, RadarxAccessor):
    """Dataset-level radarx plotting utilities."""

    @property
    def plot(self) -> RadarxDatasetPlotAccessor:
        """
        Interactive hvplot-based plots, e.g. ``ds.radarx.plot.ppi()``.

        Returns
        -------
        radarx.vis.interactive.RadarxDatasetPlotAccessor
            Plot accessor; requires the optional hvplot dependencies.

        See Also
        --------
        radarx.vis.interactive
        """
        return RadarxDatasetPlotAccessor(self.xarray_obj)

    def to_uxarray(self, variables=None):
        """
        Convert the sweep into a uxarray dataset with one face per gate.

        Parameters
        ----------
        variables : str or list of str, optional
            Variables to attach to the faces. By default all
            ``(azimuth, range)`` variables.

        Returns
        -------
        uxarray.UxDataset
            Dataset on the ``n_face`` dimension with a UGRID grid.

        See Also
        --------
        radarx.grid.to_uxarray
        """
        return to_uxarray(self.xarray_obj, variables=variables)

    def estimate_motion(self, other, field=None, **kwargs):
        """
        Storm motion from this gridded volume to a later one.

        Parameters
        ----------
        other : xarray.Dataset or xarray.DataArray
            The later gridded volume on the same grid.
        field : str, optional
            Field to track (Dataset only). Default: the reflectivity.
        **kwargs
            Passed to :func:`radarx.retrieve.estimate_motion`.

        Returns
        -------
        xarray.Dataset
            ``u`` and ``v`` storm motion (m/s) and the correlation ``quality``.

        See Also
        --------
        radarx.retrieve.estimate_motion
        """
        return retrieve_estimate_motion(self.xarray_obj, other, field, **kwargs)

    def advect(self, u, v=None, dt=None, **kwargs):
        """
        Move the gridded fields along the storm motion.

        Parameters
        ----------
        u : float, xarray.DataArray or xarray.Dataset
            Eastward motion (m/s), or the result of :meth:`estimate_motion`.
        v : float or xarray.DataArray, optional
            Northward motion (m/s).
        dt : float or timedelta, optional
            Time step in seconds (or pass ``time=`` as a target time).
        **kwargs
            Passed to :func:`radarx.retrieve.advect`.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            The advected fields.

        See Also
        --------
        radarx.retrieve.advect
        """
        return retrieve_advect(self.xarray_obj, u, v, dt, **kwargs)

    def interpolate_time(self, other, times, motion=None, **kwargs):
        """
        Advection-corrected time interpolation to a later gridded volume.

        Parameters
        ----------
        other : xarray.Dataset or xarray.DataArray
            The later gridded volume on the same grid.
        times : datetime-like or array-like
            Target times between the two volumes.
        motion : xarray.Dataset, optional
            Storm motion; estimated from the two volumes by default.
        **kwargs
            Passed to :func:`radarx.retrieve.interpolate_time`.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Fields with a new ``time`` dimension.

        See Also
        --------
        radarx.retrieve.interpolate_time
        """
        return retrieve_interpolate_time(
            self.xarray_obj, other, times, motion, **kwargs
        )

    def kdp(self, phidp=None, rhohv=None, dbzh=None, **kwargs):
        """
        Process the differential phase and estimate KDP for this sweep.

        Parameters
        ----------
        phidp, rhohv, dbzh : str, optional
            Field names; by default the usual xradar/CfRadial names.
        **kwargs
            Options of :func:`radarx.retrieve.estimate_kdp`, e.g. ``method``.

        Returns
        -------
        xarray.Dataset
            ``PHIDP_processed`` (degrees), ``KDP`` (degrees/km) and
            ``PHIDP_OFFSET``.

        See Also
        --------
        radarx.retrieve.estimate_kdp
        """
        return estimate_kdp(self.xarray_obj, phidp, rhohv, dbzh, **kwargs)

    def dsd(self, method="constrained", **kwargs):
        """
        Retrieve gamma raindrop size distribution parameters.

        Parameters
        ----------
        method : {"constrained", "normalized"}, optional
            Retrieval method. Default ``"constrained"``.
        **kwargs
            Options of :func:`radarx.retrieve.dsd`, e.g. ``mask``, ``kdp``,
            ``band``.

        Returns
        -------
        xarray.Dataset
            ``N0``, ``NW``, ``D0``, ``DM``, ``MU``, ``LAMBDA``, ``RAIN_RATE``
            and ``LWC``.

        See Also
        --------
        radarx.retrieve.dsd
        """
        return retrieve_dsd(self.xarray_obj, method, **kwargs)

    def qvp(self, data_vars=None, **kwargs):
        """
        Quasi-vertical profile of this sweep.

        Parameters
        ----------
        data_vars : str or list of str, optional
            Variables to profile. By default all ``(azimuth, range)`` fields.
        **kwargs
            Options of :func:`radarx.retrieve.qvp` (``min_rhohv``,
            ``min_dbz``, ``min_count``, ``reduction``, ...).

        Returns
        -------
        xarray.Dataset
            Profiles on the ``height`` dimension.

        See Also
        --------
        radarx.retrieve.qvp
        """
        return qvp(self.xarray_obj, data_vars, **kwargs)

    def melting_layer(self, **kwargs):
        """
        Melting-layer top and bottom from quasi-vertical profiles.

        Parameters
        ----------
        **kwargs
            Options of :func:`radarx.retrieve.melting_layer`.

        Returns
        -------
        xarray.Dataset
            ``melting_layer_top``, ``melting_layer_bottom`` and
            ``melting_layer_peak`` heights.

        See Also
        --------
        radarx.retrieve.melting_layer
        """
        return melting_layer(self.xarray_obj, **kwargs)

    def dealias(self, field="VRADH", nyquist_velocity=None, **kwargs):
        """
        Dealias (unfold) the Doppler velocity of this sweep.

        Parameters
        ----------
        field : str, optional
            Radial velocity field. Default ``"VRADH"``.
        nyquist_velocity : float, optional
            Nyquist velocity in m/s. By default read from the sweep's
            ``nyquist_velocity`` (xradar).
        **kwargs
            Further options of :func:`radarx.retrieve.dealias_velocity`, e.g.
            ``reference`` or ``wind_profile``.

        Returns
        -------
        xarray.DataArray
            Dealiased velocity with the input's coordinates.

        See Also
        --------
        radarx.retrieve.dealias_velocity
        """
        return dealias_velocity(
            self.xarray_obj, field, nyquist_velocity=nyquist_velocity, **kwargs
        )

    def interpolate_profile(
        self,
        profile,
        variables=None,
        *,
        extrapolate=False,
        engine="auto",
        n_threads=None,
    ):
        """
        Add profile variables (temperature, pressure, wind, ...) at every gate or level.

        Parameters
        ----------
        profile : xarray.Dataset
            Profile on ``height``, e.g. from
            :func:`radarx.io.sounding.read_sounding`,
            :func:`radarx.io.sounding.era5_profile` or
            ``dtree.radarx.sounding()``.
        variables : list of str, optional
            Profile variables to add. Default: all.
        extrapolate : bool, optional
            Hold the end values beyond the profile. Default False (NaN).
        engine : {"auto", "compiled", "numpy"}, optional
            Kernel implementation.
        n_threads : int, optional
            Threads for the compiled kernel. Default: all cores.

        Returns
        -------
        xarray.Dataset
            The sweep or grid with the profile variables on the dimensions
            of its ``z`` (gate heights of a georeferenced sweep, or grid
            levels).

        See Also
        --------
        radarx.io.sounding.interpolate_profile
        """
        from .io.sounding import interpolate_profile

        ds = self.xarray_obj
        if "z" not in ds:
            raise ValueError(
                "the dataset needs 'z' heights; georeference the sweep first"
            )
        env = interpolate_profile(
            profile,
            ds["z"],
            variables,
            extrapolate=extrapolate,
            engine=engine,
            n_threads=n_threads,
        )
        return ds.assign({name: env[name] for name in env.data_vars})

    def background(
        self,
        profile=None,
        *,
        source="auto",
        time=None,
        time_interpolation="linear",
        engine="auto",
        n_threads=None,
    ):
        """
        Thermodynamic and wind background on a radarx grid.

        Parameters
        ----------
        profile : xarray.Dataset, optional
            A sounding to spread uniformly over the grid. By default the ERA5
            columns at every grid cell are used.
        source : {"auto", "arco", "cds", "gcs"}, optional
            ERA5 provider when ``profile`` is not given.
        time : str or datetime-like, optional
            ERA5 valid time. Default: the grid's ``time``.
        time_interpolation : {"linear", "nearest"}, optional
            ERA5 time interpolation.
        engine : {"auto", "compiled", "numpy"}, optional
            Kernel implementation.
        n_threads : int, optional
            Threads for the compiled kernel. Default: all cores.

        Returns
        -------
        xarray.Dataset
            See :func:`radarx.io.sounding.era5_column`.

        See Also
        --------
        radarx.io.sounding.era5_column, radarx.io.sounding.profile_to_grid
        """
        from .io.sounding import era5_column, profile_to_grid

        grid = self.xarray_obj
        if profile is not None:
            return profile_to_grid(profile, grid, engine=engine, n_threads=n_threads)
        return era5_column(
            grid,
            time,
            source=source,
            time_interpolation=time_interpolation,
            engine=engine,
            n_threads=n_threads,
        )

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
class RadarxDataTreeAccessor(_ShearMixin, RadarxAccessor):
    """DataTree-level radarx retrieval and gridding utilities."""

    @property
    def plot(self) -> RadarxDataTreePlotAccessor:
        """
        Interactive hvplot-based plots, e.g. ``dt.radarx.plot.ppi("DBZH")``.

        Returns
        -------
        radarx.vis.interactive.RadarxDataTreePlotAccessor
            Plot accessor; requires the optional hvplot dependencies.

        See Also
        --------
        radarx.vis.interactive
        """
        return RadarxDataTreePlotAccessor(self.xarray_obj)

    def sounding(self, source="era5", station=None, time=None, **kwargs):
        """
        Sounding or ERA5 profile for this radar volume.

        Uses the radar's ``latitude`` and ``longitude`` and the volume start
        time.

        Parameters
        ----------
        source : {"era5", "iem", "uwyo", "igra2"}, optional
            ``"era5"`` (default) for the ERA5 profile at the radar
            (:func:`radarx.io.sounding.era5_profile`; choose the provider with
            ``era5_source="arco"|"cds"|"gcs"``), otherwise the observed
            sounding from that archive at the nearest 00/12 UTC launch
            (:func:`radarx.io.sounding.read_sounding`).
        station : str, optional
            Radiosonde station. Default: the nearest station in that archive.
        time : str or datetime-like, optional
            Time to use instead of the volume start time.
        **kwargs
            Passed on to the reader.

        Returns
        -------
        xarray.Dataset
            Profile on ``height`` (m above sea level).

        See Also
        --------
        radarx.io.sounding
        """
        from .io.sounding import _sounding_for_volume

        if "era5_source" in kwargs:
            kwargs["source"] = kwargs.pop("era5_source")
        return _sounding_for_volume(
            self.xarray_obj, kind=source, station=station, time=time, **kwargs
        )

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
        method="cone",
        n_threads=None,
    ):
        """
        Grid the radar volume onto a Cartesian 3D domain.

        Parameters
        ----------
        data_vars : list of str, optional
            Fields to grid. By default all fields.
        pseudo_cappi : bool, optional
            Fill levels below the lowest sweep. Default True.
        x_lim, y_lim, z_lim : tuple of float, optional
            Grid extent in metres (``z`` above sea level).
        x_step, y_step, z_step : float, optional
            Grid spacing in metres.
        x_smth, y_smth, z_smth : float, optional
            Smoothing factors, for ``method="barnes"`` only.
        method : {"cone", "barnes"}, optional
            Interpolation method. Default ``"cone"``.
        n_threads : int, optional
            Threads for ``method="cone"``. Default: all cores.

        Returns
        -------
        xarray.Dataset
            Gridded fields on ``(z, y, x)``.

        See Also
        --------
        radarx.grid.grid_radar, radarx.grid.grid_cones
        """
        return grid_radar(
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
            method=method,
            n_threads=n_threads,
        )

    def kdp(self, phidp=None, rhohv=None, dbzh=None, **kwargs):
        """
        Process the differential phase and estimate KDP for every sweep.

        All rays of all sweeps are processed in one call of the compiled
        kernel.

        Parameters
        ----------
        phidp, rhohv, dbzh : str, optional
            Field names; by default the usual xradar/CfRadial names.
        **kwargs
            Options of :func:`radarx.retrieve.estimate_kdp`, e.g. ``method``.

        Returns
        -------
        xarray.DataTree
            The root of the volume and one node per sweep with processed
            ``PHIDP_processed`` (degrees), ``KDP`` (degrees/km) and ``PHIDP_OFFSET``.

        See Also
        --------
        radarx.retrieve.estimate_kdp
        """
        return estimate_kdp(self.xarray_obj, phidp, rhohv, dbzh, **kwargs)

    def dsd(self, method="constrained", **kwargs):
        """
        Retrieve gamma raindrop size distribution parameters for every sweep.

        All gates of all sweeps are processed in one call of the compiled
        kernel.

        Parameters
        ----------
        method : {"constrained", "normalized"}, optional
            Retrieval method. Default ``"constrained"``.
        **kwargs
            Options of :func:`radarx.retrieve.dsd`, e.g. ``mask``, ``kdp``,
            ``band``.

        Returns
        -------
        xarray.DataTree
            The root of the volume and one node per sweep with ``N0``,
            ``NW``, ``D0``, ``DM``, ``MU``, ``LAMBDA``, ``RAIN_RATE`` and
            ``LWC``.

        See Also
        --------
        radarx.retrieve.dsd
        """
        return retrieve_dsd(self.xarray_obj, method, **kwargs)

    def qvp(self, data_vars=None, **kwargs):
        """
        Quasi-vertical profile of one sweep of the volume.

        Parameters
        ----------
        data_vars : str or list of str, optional
            Variables to profile. By default all ``(azimuth, range)`` fields.
        **kwargs
            Options of :func:`radarx.retrieve.qvp`, e.g. ``sweep`` or
            ``elevation`` (default: the highest sweep), ``min_rhohv``,
            ``min_dbz``, ``min_count``, ``reduction``.

        Returns
        -------
        xarray.Dataset
            Profiles on the ``height`` dimension.

        See Also
        --------
        radarx.retrieve.qvp, radarx.retrieve.qvp_timeseries
        """
        return qvp(self.xarray_obj, data_vars, **kwargs)

    def dealias(self, field="VRADH", nyquist_velocity=None, **kwargs):
        """
        Dealias (unfold) the Doppler velocity of every sweep.

        Parameters
        ----------
        field : str, optional
            Radial velocity field. Default ``"VRADH"``.
        nyquist_velocity : float or dict, optional
            Nyquist velocity in m/s, or a dict by sweep name. By default read
            from each sweep's ``nyquist_velocity`` (xradar).
        **kwargs
            Further options of :func:`radarx.retrieve.dealias_velocity`, e.g.
            ``wind_profile``, ``sweep_continuity`` or ``name``.

        Returns
        -------
        xarray.DataTree
            Copy of the volume with the dealiased field in every sweep.

        See Also
        --------
        radarx.retrieve.dealias_velocity
        """
        return dealias_velocity(
            self.xarray_obj, field, nyquist_velocity=nyquist_velocity, **kwargs
        )

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
