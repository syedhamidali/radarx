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

from ._registry import forward
from .grid import (
    grid_radar,  # noqa
    to_uxarray,  # noqa
)
from .retrieve import advect as retrieve_advect  # noqa
from .retrieve import apply_mask as retrieve_apply_mask  # noqa
from .retrieve import create_cappi as retrieve_cappi  # noqa
from .retrieve import (  # noqa
    dealias_velocity,
    estimate_kdp,  # noqa
    melting_layer,
    qvp,
)
from .retrieve import echo_mask as retrieve_echo_mask  # noqa
from .retrieve import estimate_motion as retrieve_estimate_motion  # noqa
from .retrieve import interpolate_time as retrieve_interpolate_time  # noqa
from .retrieve import shear as _shear
from .retrieve.multidoppler import dual_doppler_analytic, multi_doppler
from .vis import plot_cappi, plot_maxcappi, plot_ppi, plot_rhi  # noqa
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


def _forwards(func):
    """Decorator for an accessor method that calls ``func`` on the wrapped object."""
    return lambda stub: forward(func, stub)


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

    @_forwards(_shear.llsd)
    def llsd(self):
        """
        Azimuthal shear and radial divergence by linear least-squares derivatives.

        Parameters
        ----------
        field : str, optional
            Dealiased radial velocity field. Default: the first of ``VRADH``,
            ``VRAD``, ``VRADV``, ``VEL``, ``VELOCITY``, ``velocity`` that the
            sweep holds.
        window : tuple of float, optional
            Window ``(range_m, azimuth_m)`` in metres. Default ``(750, 2500)``.

        Returns
        -------
        xarray.Dataset or xarray.DataTree
            ``azimuthal_shear`` and ``radial_divergence`` in s⁻¹.

        See Also
        --------
        radarx.retrieve.llsd
        """

    @_forwards(_shear.azimuthal_shear)
    def azimuthal_shear(self):
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

    @_forwards(_shear.radial_divergence)
    def radial_divergence(self):
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


class _AssignMixin:
    """Merge retrieval products into the sweep or volume they came from."""

    def assign(self, products):
        """
        Add retrieval products to this sweep or volume.

        The retrievals (``dealias``, ``kdp``, ``llsd``, ...) return their
        products only. ``assign`` puts them back next to the measured fields:
        product variables are added to the matching sweep, aligned to the
        sweep's coordinates (gates the products do not cover are NaN), and
        coordinates the sweep already has are kept from the sweep.

        Parameters
        ----------
        products : xarray.Dataset, xarray.DataArray or xarray.DataTree
            For a sweep, a Dataset or named DataArray of products. For a
            volume, a DataTree of products per sweep as returned by the
            retrievals; its root is not merged.

        Returns
        -------
        xarray.Dataset or xarray.DataTree
            A copy with the product variables added. Variables of the same
            name are replaced.

        Raises
        ------
        KeyError
            If a product sweep is not in the volume.
        TypeError
            If ``products`` does not fit this object.

        Examples
        --------
        >>> dtree = dtree.radarx.assign(dtree.radarx.dealias())  # doctest: +SKIP
        >>> dtree = dtree.radarx.assign(dtree.radarx.kdp())  # doctest: +SKIP
        >>> sweep = sweep.radarx.assign(sweep.radarx.llsd("VRADH_dealiased"))  # doctest: +SKIP
        """
        from .retrieve._products import assign_products

        return assign_products(self.xarray_obj, products)


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

    @_forwards(to_uxarray)
    def to_uxarray(self):
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

    @_forwards(retrieve_estimate_motion)
    def estimate_motion(self):
        """
        Storm motion from this gridded volume to a later one.

        Parameters
        ----------
        other : xarray.Dataset or xarray.DataArray
            The later gridded volume on the same grid.
        field : str, optional
            Field to track (Dataset only). Default: the reflectivity.

        Returns
        -------
        xarray.Dataset
            ``u`` and ``v`` storm motion (m/s) and the correlation ``quality``.

        See Also
        --------
        radarx.retrieve.estimate_motion
        """

    @_forwards(retrieve_advect)
    def advect(self):
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

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            The advected fields.

        See Also
        --------
        radarx.retrieve.advect
        """

    @_forwards(retrieve_interpolate_time)
    def interpolate_time(self):
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

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Fields with a new ``time`` dimension.

        See Also
        --------
        radarx.retrieve.interpolate_time
        """


@xr.register_dataset_accessor("radarx")
class RadarxDataSetAccessor(_AssignMixin, _ShearMixin, RadarxAccessor):
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

    @_forwards(to_uxarray)
    def to_uxarray(self):
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

    @_forwards(retrieve_estimate_motion)
    def estimate_motion(self):
        """
        Storm motion from this gridded volume to a later one.

        Parameters
        ----------
        other : xarray.Dataset or xarray.DataArray
            The later gridded volume on the same grid.
        field : str, optional
            Field to track (Dataset only). Default: the reflectivity.

        Returns
        -------
        xarray.Dataset
            ``u`` and ``v`` storm motion (m/s) and the correlation ``quality``.

        See Also
        --------
        radarx.retrieve.estimate_motion
        """

    @_forwards(retrieve_advect)
    def advect(self):
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

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            The advected fields.

        See Also
        --------
        radarx.retrieve.advect
        """

    @_forwards(retrieve_interpolate_time)
    def interpolate_time(self):
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

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Fields with a new ``time`` dimension.

        See Also
        --------
        radarx.retrieve.interpolate_time
        """

    @_forwards(estimate_kdp)
    def kdp(self):
        """
        Process the differential phase and estimate KDP for this sweep.

        Parameters
        ----------
        phidp, rhohv, dbz : str, optional
            Field names; by default the usual xradar/CfRadial names.

        Returns
        -------
        xarray.Dataset
            ``PHIDP_processed`` (degree), ``KDP`` (degree km-1) and
            ``PHIDP_OFFSET``.

        See Also
        --------
        radarx.retrieve.estimate_kdp
        """

    @_forwards(retrieve_echo_mask)
    def echo_mask(self):
        """
        Classify the gates of this sweep as meteorological or not.

        Returns
        -------
        xarray.Dataset
            ``ECHO_CLASS``, ``METEO_SCORE`` and ``METEO_MASK``.

        See Also
        --------
        radarx.retrieve.echo_mask, radarx.retrieve.apply_mask
        """

    @_forwards(retrieve_apply_mask)
    def apply_mask(self):
        """
        Set the non-meteorological gates of this sweep to NaN.

        Parameters
        ----------
        mask : xarray.Dataset or xarray.DataArray, optional
            Output of :func:`radarx.retrieve.echo_mask`; computed if not
            given.
        fields : str or list of str, optional
            Fields to mask. Default: all floating point gate fields.

        Returns
        -------
        xarray.Dataset
            Copy of the sweep with the masked fields.

        See Also
        --------
        radarx.retrieve.apply_mask, radarx.retrieve.echo_mask
        """

    @_forwards(qvp)
    def qvp(self):
        """
        Quasi-vertical profile of this sweep.

        Parameters
        ----------
        data_vars : str or list of str, optional
            Variables to profile. By default all ``(azimuth, range)`` fields.

        Returns
        -------
        xarray.Dataset
            Profiles on the ``height`` dimension.

        See Also
        --------
        radarx.retrieve.qvp
        """

    @_forwards(melting_layer)
    def melting_layer(self):
        """
        Melting-layer top and bottom from quasi-vertical profiles.

        Returns
        -------
        xarray.Dataset
            ``melting_layer_top``, ``melting_layer_bottom`` and
            ``melting_layer_peak`` heights (by default where the ρhv
            signature begins and ends) and, with an ``environment``, the
            0 °C and wet-bulb 0 °C heights and the offsets of the top.

        See Also
        --------
        radarx.retrieve.melting_layer
        """

    @_forwards(dealias_velocity)
    def dealias(self):
        """
        Dealias (unfold) the Doppler velocity of this sweep.

        Parameters
        ----------
        field : str, optional
            Radial velocity field. Default: the first of ``VRADH``, ``VRAD``,
            ``VRADV``, ``VEL``, ``VELOCITY``, ``velocity`` that the sweep holds.
        nyquist_velocity : float, optional
            Nyquist velocity in m/s. By default read from the sweep's
            ``nyquist_velocity`` (xradar).

        Returns
        -------
        xarray.DataArray
            Dealiased velocity ``f"{field}_dealiased"`` (or ``name``) with
            the input's coordinates. Add it to the sweep with
            :meth:`assign`.

        See Also
        --------
        radarx.retrieve.dealias_velocity
        """

    def interpolate_profile(
        self,
        profile,
        variables=None,
        *,
        extrapolate=False,
        n_threads=None,
        engine="auto",
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
        n_threads : int, optional
            Threads for the compiled kernel. Default: all cores.
        engine : {"auto", "compiled", "numpy"}, optional
            Kernel implementation.

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
        n_threads=None,
        engine="auto",
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
        n_threads : int, optional
            Threads for the compiled kernel. Default: all cores.
        engine : {"auto", "compiled", "numpy"}, optional
            Kernel implementation.

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

    @_forwards(multi_doppler)
    def multi_doppler(self):
        """
        Retrieve the 3D wind from the radial velocities of several radars.

        Parameters
        ----------
        background : xarray.Dataset, optional
            Background on the grid (``grid.radarx.background()``).

        Returns
        -------
        xarray.Dataset
            ``u``, ``v``, ``w`` and diagnostics on the grid.

        See Also
        --------
        radarx.retrieve.multi_doppler, radarx.retrieve.multi_doppler_input
        """

    @_forwards(dual_doppler_analytic)
    def dual_doppler_analytic(self):
        """
        Analytic dual-Doppler winds from the radial velocities of several radars.

        Parameters
        ----------
        background : xarray.Dataset, optional
            Background on the grid; only its air density and freezing level
            are used.

        Returns
        -------
        xarray.Dataset
            ``u``, ``v``, ``w``, ``u_error`` and ``v_error`` on the grid.

        See Also
        --------
        radarx.retrieve.dual_doppler_analytic, radarx.retrieve.multi_doppler
        """

    @_forwards(plot_maxcappi)
    def plot_max_cappi(self):
        """Plot a maximum CAPPI product from a 3D gridded radar dataset."""

    @_forwards(plot_ppi)
    def plot_ppi(self):
        """Plot a georeferenced plan-position view using ``x`` and ``y``."""

    @_forwards(plot_rhi)
    def plot_rhi(self):
        """Plot a vertical cross-section using ground range and height."""

    @_forwards(plot_cappi)
    def plot_cappi(self):
        """Plot a CAPPI dataset on the horizontal plane."""


@register_datatree_accessor("radarx")
class RadarxDataTreeAccessor(_AssignMixin, _ShearMixin, RadarxAccessor):
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

    @_forwards(grid_radar)
    def to_grid(self):
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
            Gridded fields on ``(z, y, x)`` with the volume's mean time as a
            scalar ``time`` coordinate.

        See Also
        --------
        radarx.grid.grid_radar, radarx.grid.grid_cones
        """

    @_forwards(estimate_kdp)
    def kdp(self):
        """
        Process the differential phase and estimate KDP for every sweep.

        All rays of all sweeps are processed in one call of the compiled
        kernel.

        Parameters
        ----------
        phidp, rhohv, dbz : str, optional
            Field names; by default the usual xradar/CfRadial names.

        Returns
        -------
        xarray.DataTree
            The root of the volume and one node per sweep with processed
            ``PHIDP_processed`` (degree), ``KDP`` (degree km-1) and ``PHIDP_OFFSET``.

        See Also
        --------
        radarx.retrieve.estimate_kdp
        """

    @_forwards(retrieve_echo_mask)
    def echo_mask(self):
        """
        Classify the gates of every sweep as meteorological or not.

        Returns
        -------
        xarray.DataTree
            One node per sweep with ``ECHO_CLASS``, ``METEO_SCORE`` and
            ``METEO_MASK``; all sweeps are classified in one kernel call.

        See Also
        --------
        radarx.retrieve.echo_mask, radarx.retrieve.apply_mask
        """

    @_forwards(retrieve_apply_mask)
    def apply_mask(self):
        """
        Set the non-meteorological gates of every sweep to NaN.

        Parameters
        ----------
        mask : xarray.DataTree, optional
            Output of :func:`radarx.retrieve.echo_mask`; computed if not
            given.
        fields : str or list of str, optional
            Fields to mask. Default: all floating point gate fields.

        Returns
        -------
        xarray.Dataset
            Copy of the volume with the masked fields.

        See Also
        --------
        radarx.retrieve.apply_mask, radarx.retrieve.echo_mask
        """

    @_forwards(qvp)
    def qvp(self):
        """
        Quasi-vertical profile of one sweep of the volume.

        Parameters
        ----------
        data_vars : str or list of str, optional
            Variables to profile. By default all ``(azimuth, range)`` fields.

        Returns
        -------
        xarray.Dataset
            Profiles on the ``height`` dimension.

        See Also
        --------
        radarx.retrieve.qvp, radarx.retrieve.qvp_timeseries
        """

    @_forwards(dealias_velocity)
    def dealias(self):
        """
        Dealias (unfold) the Doppler velocity of every sweep.

        Parameters
        ----------
        field : str, optional
            Radial velocity field. Default: the first of ``VRADH``, ``VRAD``,
            ``VRADV``, ``VEL``, ``VELOCITY``, ``velocity`` that the sweep holds.
        nyquist_velocity : float or dict, optional
            Nyquist velocity in m/s, or a dict by sweep name. By default read
            from each sweep's ``nyquist_velocity`` (xradar).

        Returns
        -------
        xarray.DataTree
            The root of the volume and one node per sweep with the dealiased
            velocity ``f"{field}_dealiased"`` (or ``name``). Add it to the
            volume with :meth:`assign`.

        See Also
        --------
        radarx.retrieve.dealias_velocity
        """

    @_forwards(retrieve_cappi)
    def create_cappi(self):
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

    @_forwards(retrieve_cappi)
    def to_cappi(self):
        """Convenience alias for :meth:`create_cappi`."""


def _attach_registered_methods():
    """Add the methods that feature modules registered (see ``radarx._registry``)."""
    from . import grid, io, retrieve, vis  # noqa: F401  feature modules register here
    from ._registry import registered

    for kind, cls in (
        ("dataarray", RadarxDataArrayAccessor),
        ("dataset", RadarxDataSetAccessor),
        ("datatree", RadarxDataTreeAccessor),
    ):
        for name, func in registered(kind).items():
            if hasattr(cls, name) and getattr(cls, name) is not func:
                raise ValueError(f"{cls.__name__} already defines {name!r}")
            setattr(cls, name, func)


_attach_registered_methods()
