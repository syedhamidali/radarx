#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx Interactive Plots
========================

Interactive radar plots built on `hvplot <https://hvplot.holoviz.org/>`_ and
`HoloViews <https://holoviews.org/>`_, following the accessor roadmap
discussed in `openradar/xradar#174
<https://github.com/openradar/xradar/issues/174>`_.

The functions are exposed through the ``.radarx.plot`` accessor on
:py:class:`xarray.DataArray`, :py:class:`xarray.Dataset` and
:py:class:`xarray.DataTree`::

    import radarx  # noqa: registers the accessors

    da.radarx.plot()  # range-azimuth (or time-range) view
    da.radarx.plot.ppi()  # plan-position view on x/y
    ds.radarx.plot.ppi()  # one PPI panel per variable
    dt.radarx.plot.ppi("DBZH")  # one PPI panel per sweep
    dt.radarx.plot.rhi("DBZH")
    dt.radarx.plot.cappi("DBZH", height=2000)
    grid.radarx.plot.max_cappi("DBZH")

Any other method falls through to ``obj.hvplot``, e.g.
``da.radarx.plot.hist()``. All plot methods accept ``backend`` (``"bokeh"`` or
``"matplotlib"``) and forward remaining keyword arguments to hvplot.

hvplot, holoviews and bokeh are optional dependencies; install them with
``pip install radarx[plot]``.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = [
    "hvplot_range_azimuth",
    "hvplot_ppi",
    "hvplot_mesh",
    "hvplot_centroids",
    "hvplot_rhi",
    "hvplot_cappi",
    "hvplot_max_cappi",
    "RadarxDataArrayPlotAccessor",
    "RadarxDatasetPlotAccessor",
    "RadarxDataTreePlotAccessor",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np
import xarray as xr

DEFAULT_CMAP = "ChaseSpectral"

# import speedup trick (borrowed from uxarray): importing hvplot is slow, so it
# is only done once a plot accessor is actually created.
_IMPORTED_HVPLOT = False


def _ensure_hvplot_imported():
    """Import ``hvplot.xarray`` and ``hvplot.pandas`` once, keeping the backend.

    Importing hvplot calls ``hvplot.extension()``, which would silently reset a
    backend the user had chosen before, so the active backend is restored.
    """
    global _IMPORTED_HVPLOT
    if _IMPORTED_HVPLOT:
        return
    try:
        from holoviews import Store
    except ImportError as err:
        raise ImportError(
            "Interactive radarx plots require hvplot and holoviews. "
            "Install them with: pip install 'radarx[plot]'"
        ) from err

    backend_orig = Store.current_backend
    try:
        import hvplot.pandas  # noqa: F401
        import hvplot.xarray  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "Interactive radarx plots require hvplot. "
            "Install it with: pip install 'radarx[plot]'"
        ) from err
    if backend_orig in Store.registry:
        Store.set_current_backend(backend_orig)
    import cmweather  # noqa: F401  registers radar colormaps

    _IMPORTED_HVPLOT = True


def _assign_backend(backend):
    """Switch the HoloViews backend to ``backend`` (``None`` keeps it)."""
    _ensure_hvplot_imported()
    import holoviews as hv

    if backend not in ("bokeh", "matplotlib", None):
        raise ValueError(
            "Unsupported backend. Expected one of ['bokeh', 'matplotlib'], "
            f"but received {backend!r}"
        )
    if backend is None or backend == hv.Store.current_backend:
        return
    import matplotlib as mpl

    # hv.extension("matplotlib") switches the active matplotlib backend
    # (e.g. to agg), which would break later plain matplotlib plots.
    mpl_backend = mpl.get_backend()
    hv.extension(backend)
    if backend == "matplotlib":
        mpl.use(mpl_backend)


def _radar_variables(ds):
    """Data variables with a ``range`` (sweep) or ``x``/``y`` (grid) dimension."""
    return [
        name
        for name, da in ds.data_vars.items()
        if "range" in da.dims or {"x", "y"} <= set(da.dims)
    ]


def _georeference(da):
    """Return ``da`` with Cartesian ``x``/``y``/``z`` gate coordinates in km.

    Uses existing ``x``/``y``/``z`` coordinates (e.g. from
    ``xradar.georeference``) and otherwise computes them from
    range/azimuth/elevation relative to the radar.
    """
    if not {"x", "y", "z"} <= set(da.coords):
        if "range" not in da.dims:
            raise ValueError(
                "Cannot georeference: expected 'x', 'y', 'z' coordinates or a "
                "polar sweep with a 'range' dimension."
            )
        from xradar.georeference import antenna_to_cartesian

        site_alt = float(da.coords["altitude"]) if "altitude" in da.coords else 0.0
        rng, az, el = (
            c.broadcast_like(da).transpose(*da.dims)
            for c in (da["range"], da["azimuth"], da["elevation"])
        )
        x, y, z = antenna_to_cartesian(
            rng.values, az.values, el.values, site_altitude=site_alt
        )
        dims = da.dims
        da = da.assign_coords(x=(dims, x), y=(dims, y), z=(dims, z))
    return da.assign_coords(
        {
            c: da[c]
            .copy(data=np.asarray(da[c].values, dtype=float) / 1e3)
            .assign_attrs(units="km")
            for c in ("x", "y", "z")
        }
    )


def _clim_title_defaults(da, kwargs, title):
    kwargs.setdefault("cmap", DEFAULT_CMAP)
    if "clim" not in kwargs:
        # radar moments often carry unmasked outliers; use 2-98 % limits
        kwargs.setdefault("robust", True)
    # hvplot centres the colour range on zero whenever data crosses zero,
    # which suits velocity but not reflectivity; pass symmetric=True to opt in
    kwargs.setdefault("symmetric", False)
    kwargs.setdefault("title", title)
    units = da.attrs.get("units")
    if units and "clabel" not in kwargs:
        kwargs["clabel"] = f"{da.name} ({units})" if da.name else units
    return kwargs


def _fixed_angle(da):
    for name in ("sweep_fixed_angle", "fixed_angle"):
        if name in da.coords and da.coords[name].size == 1:
            return float(da.coords[name])
    return None


def _ppi_title(da, kind="PPI"):
    angle = _fixed_angle(da)
    angle = f" {angle:.1f}°" if angle is not None else ""
    return f"{kind}{angle} {da.name or ''}".strip()


def _plan_axes(kwargs):
    kwargs.setdefault("aspect", "equal")
    kwargs.setdefault("xlabel", "x (km)")
    kwargs.setdefault("ylabel", "y (km)")
    return kwargs


def hvplot_range_azimuth(da, backend=None, **kwargs):
    """
    Plot a sweep in its native polar layout (range vs. azimuth/time).

    Parameters
    ----------
    da : xarray.DataArray
        Sweep moment with a ``range`` dimension and one ray dimension
        (``azimuth``, ``elevation`` or ``time``).
    backend : {"bokeh", "matplotlib"}, optional
        HoloViews plotting backend. ``None`` (default) keeps the active one.
    **kwargs : dict, optional
        Additional keyword arguments passed to hvplot (e.g. ``clim``, ``cmap``,
        ``frame_width``, ``rasterize``). By default the colour range uses the
        2nd-98th percentiles (``robust=True``) and is not forced symmetric.

    Returns
    -------
    holoviews.QuadMesh
        Range (km) on the x-axis against the ray dimension on the y-axis.

    Raises
    ------
    ValueError
        If ``da`` has no ``range`` dimension or no ray dimension.
    """
    _assign_backend(backend)
    ray_dim = next((d for d in da.dims if d != "range"), None)
    if "range" not in da.dims or ray_dim is None:
        raise ValueError("Expected a 2D sweep DataArray with a 'range' dimension.")
    da = da.assign_coords(range=da["range"] / 1e3)
    kwargs = _clim_title_defaults(da, kwargs, da.name or "")
    kwargs.setdefault("xlabel", "Range (km)")
    return da.hvplot.quadmesh(x="range", y=ray_dim, **kwargs)


def hvplot_ppi(da, backend=None, **kwargs):
    """
    Plot a sweep as a georeferenced plan-position indicator (PPI).

    Parameters
    ----------
    da : xarray.DataArray
        Sweep moment on polar ``azimuth``/``range`` dimensions, or a field
        already on ``x``/``y`` dimensions. Existing ``x``/``y``/``z``
        coordinates (e.g. from ``xradar.georeference``) are reused; otherwise
        they are computed from range, azimuth and elevation.
    backend : {"bokeh", "matplotlib"}, optional
        HoloViews plotting backend. ``None`` (default) keeps the active one.
    **kwargs : dict, optional
        Additional keyword arguments passed to hvplot (e.g. ``clim``, ``cmap``,
        ``frame_width``, ``rasterize``). By default the colour range uses the
        2nd-98th percentiles (``robust=True``) and is not forced symmetric.

    Returns
    -------
    holoviews.QuadMesh
        Plan view with ``x`` and ``y`` in km.

    Raises
    ------
    ValueError
        If ``da`` cannot be georeferenced.
    """
    _assign_backend(backend)
    if not ({"x", "y"} <= set(da.dims)):
        da = _georeference(da)
    else:
        da = da.assign_coords(x=da["x"] / 1e3, y=da["y"] / 1e3)
    kwargs = _plan_axes(_clim_title_defaults(da, kwargs, _ppi_title(da)))
    return da.hvplot.quadmesh(x="x", y="y", **kwargs)


def hvplot_mesh(da, backend=None, line_color="black", line_width=0.2, **kwargs):
    """
    Plot a PPI with the outline of every radar gate drawn.

    Useful to inspect the beam geometry and gate resolution. Best used on a
    subset of the sweep, as drawing every gate edge is slow.

    Parameters
    ----------
    da : xarray.DataArray
        Sweep moment; see :func:`hvplot_ppi`.
    backend : {"bokeh", "matplotlib"}, optional
        HoloViews plotting backend. ``None`` (default) keeps the active one.
    line_color : str, optional
        Colour of the gate outlines. Default is ``"black"``.
    line_width : float, optional
        Width of the gate outlines. Default is ``0.2``.
    **kwargs : dict, optional
        Additional keyword arguments passed to hvplot (e.g. ``clim``, ``cmap``,
        ``frame_width``, ``rasterize``). By default the colour range uses the
        2nd-98th percentiles (``robust=True``) and is not forced symmetric.

    Returns
    -------
    holoviews.QuadMesh
        Plan view with gate outlines.
    """
    import holoviews as hv

    plot = hvplot_ppi(da, backend=backend, **kwargs)
    if hv.Store.current_backend == "matplotlib":
        return plot.opts(edgecolors=line_color, linewidths=line_width)
    return plot.opts(line_color=line_color, line_width=line_width)


def hvplot_centroids(da, backend=None, **kwargs):
    """
    Plot the gate centres of a sweep as points coloured by value.

    Parameters
    ----------
    da : xarray.DataArray
        Sweep moment; see :func:`hvplot_ppi`. Gates with missing values are
        dropped.
    backend : {"bokeh", "matplotlib"}, optional
        HoloViews plotting backend. ``None`` (default) keeps the active one.
    **kwargs : dict, optional
        Additional keyword arguments passed to hvplot (e.g. ``clim``, ``cmap``,
        ``frame_width``, ``rasterize``). By default the colour range uses the
        2nd-98th percentiles (``robust=True``) and is not forced symmetric.

    Returns
    -------
    holoviews.Points
        One point per valid gate at its ``x``/``y`` location in km.
    """
    _assign_backend(backend)
    name = da.name or "value"
    da = _georeference(da.rename(name))
    import pandas as pd

    x, y, values = xr.broadcast(da["x"], da["y"], da)
    df = pd.DataFrame(
        {"x": x.values.ravel(), "y": y.values.ravel(), name: values.values.ravel()}
    ).dropna(subset=[name])
    kwargs = _plan_axes(_clim_title_defaults(da, kwargs, _ppi_title(da, "Gates")))
    kwargs.setdefault("size", 2)
    return df.hvplot.points(x="x", y="y", c=name, **kwargs)


def hvplot_rhi(da, backend=None, **kwargs):
    """
    Plot a range-height indicator (RHI) as ground range vs. height.

    Parameters
    ----------
    da : xarray.DataArray
        RHI sweep moment, typically on ``elevation``/``range`` dimensions.
    backend : {"bokeh", "matplotlib"}, optional
        HoloViews plotting backend. ``None`` (default) keeps the active one.
    **kwargs : dict, optional
        Additional keyword arguments passed to hvplot (e.g. ``clim``, ``cmap``,
        ``frame_width``, ``rasterize``). By default the colour range uses the
        2nd-98th percentiles (``robust=True``) and is not forced symmetric.

    Returns
    -------
    holoviews.QuadMesh
        Ground range (km) on the x-axis against height (km) on the y-axis.
    """
    _assign_backend(backend)
    da = _georeference(da)
    ground_range = np.hypot(da["x"], da["y"]).assign_attrs(units="km")
    da = da.assign_coords(ground_range=ground_range)
    angle = _fixed_angle(da)
    title = f"RHI{f' {angle:.1f}°' if angle is not None else ''} {da.name or ''}"
    kwargs = _clim_title_defaults(da, kwargs, title.strip())
    kwargs.setdefault("xlabel", "Ground range (km)")
    kwargs.setdefault("ylabel", "Height (km)")
    return da.hvplot.quadmesh(x="ground_range", y="z", **kwargs)


def hvplot_cappi(da, z=None, backend=None, **kwargs):
    """
    Plot a CAPPI on the horizontal ``x``/``y`` plane.

    Parameters
    ----------
    da : xarray.DataArray
        CAPPI field on ``x``/``y`` (e.g. from ``create_cappi``) or a 3D grid
        with a ``z`` dimension (e.g. from ``to_grid``).
    z : float, optional
        Height in metres to select from a 3D grid (nearest level). If omitted
        for a 3D grid, a height slider is shown.
    backend : {"bokeh", "matplotlib"}, optional
        HoloViews plotting backend. ``None`` (default) keeps the active one.
    **kwargs : dict, optional
        Additional keyword arguments passed to hvplot (e.g. ``clim``, ``cmap``,
        ``frame_width``, ``rasterize``). By default the colour range uses the
        2nd-98th percentiles (``robust=True``) and is not forced symmetric.

    Returns
    -------
    holoviews.QuadMesh or holoviews.DynamicMap
        Plan view, or a height slider over all levels of a 3D grid.
    """
    if "z" in da.dims:
        if z is not None:
            da = da.sel(z=z, method="nearest")
        else:
            kwargs.setdefault("groupby", "z")
    height = float(da["z"]) / 1e3 if "z" in da.coords and da["z"].size == 1 else None
    title = f"CAPPI {height:.1f} km {da.name}" if height is not None else None
    kwargs.setdefault("title", title or f"CAPPI {da.name}")
    return hvplot_ppi(da, backend=backend, **kwargs)


def hvplot_max_cappi(da, backend=None, **kwargs):
    """
    Plot a Max-CAPPI with side projections.

    The plan view shows the column maximum, flanked by the maximum along
    ``y`` (top, x-z) and along ``x`` (right, z-y). With the bokeh backend the
    axes are linked, so zooming the plan view also zooms the projections.

    Parameters
    ----------
    da : xarray.DataArray
        3D grid with ``z``, ``y`` and ``x`` dimensions (e.g. from ``to_grid``).
    backend : {"bokeh", "matplotlib"}, optional
        HoloViews plotting backend. ``None`` (default) keeps the active one.
    **kwargs : dict, optional
        Additional keyword arguments passed to hvplot (e.g. ``clim``, ``cmap``,
        ``frame_width``, ``rasterize``). By default the colour range uses the
        2nd-98th percentiles (``robust=True``) and is not forced symmetric.
        ``frame_width`` sets the size of the plan view in pixels (bokeh).

    Returns
    -------
    holoviews.Layout
        Top projection, plan view and right projection sharing one colour
        range.

    Raises
    ------
    ValueError
        If ``da`` is not a 3D grid.

    See Also
    --------
    radarx.vis.plot_maxcappi : Static matplotlib/cartopy Max-CAPPI.
    """
    import holoviews as hv

    _assign_backend(backend)
    if not {"x", "y", "z"} <= set(da.dims):
        raise ValueError("max_cappi expects a 3D grid with 'z', 'y', 'x' dimensions.")
    da = da.assign_coords(
        x=(da["x"] / 1e3).assign_attrs(units="km"),
        y=(da["y"] / 1e3).assign_attrs(units="km"),
        z=(da["z"] / 1e3).assign_attrs(units="km"),
    )
    kwargs = _clim_title_defaults(da, kwargs, f"Max-{da.name}")
    if kwargs.pop("robust", False):
        # one shared colour range for all three panels
        kwargs["clim"] = tuple(float(v) for v in np.nanpercentile(da, [2, 98]))
    size = kwargs.pop("frame_width", 400)
    clabel = kwargs.pop("clabel", None)
    side = {
        k: v
        for k, v in kwargs.items()
        if k in ("cmap", "clim", "rasterize", "symmetric")
    }
    plan = da.max("z").hvplot.quadmesh(
        x="x", y="y", xlabel="x (km)", ylabel="y (km)", colorbar=False, **kwargs
    )
    top = da.max("y").hvplot.quadmesh(
        x="x", y="z", colorbar=False, xlabel="", ylabel="Height (km)", title="", **side
    )
    right = da.max("x").hvplot.quadmesh(
        x="z", y="y", xlabel="Height (km)", ylabel="", title="", clabel=clabel, **side
    )
    side_size = size // 3
    plan.opts(frame_width=size, frame_height=size, backend="bokeh")
    top.opts(frame_width=size, frame_height=side_size, backend="bokeh")
    right.opts(frame_width=side_size, frame_height=size, backend="bokeh")
    plan.opts(aspect=1, backend="matplotlib")
    top.opts(aspect=3, backend="matplotlib")
    right.opts(aspect=1 / 3, backend="matplotlib")
    layout = _layout([top, hv.Empty(), plan, right], cols=2)
    return layout.opts(tight=True, hspace=0.05, vspace=0.05, backend="matplotlib")


def _layout(panels, cols):
    import holoviews as hv

    layout = hv.Layout(panels).cols(cols)
    return layout.opts(sublabel_format="", backend="matplotlib")


class _RadarxPlotAccessor:
    """Shared plumbing: lazy hvplot import and fallthrough to ``obj.hvplot``."""

    __slots__ = ("_obj",)

    def __init__(self, obj):
        _ensure_hvplot_imported()
        self._obj = obj

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._obj.hvplot, name)


class RadarxDataArrayPlotAccessor(_RadarxPlotAccessor):
    """
    Interactive plots of a single radar field, available as ``da.radarx.plot``.

    Calling the accessor picks a default view: range-azimuth for polar
    sweeps, a CAPPI for gridded fields and ``da.hvplot()`` otherwise. Any
    attribute not defined here is looked up on ``da.hvplot``.

    Parameters
    ----------
    obj : xarray.DataArray
        The radar field to plot.

    Examples
    --------
    >>> da.radarx.plot()  # doctest: +SKIP
    >>> da.radarx.plot.ppi(clim=(0, 60))  # doctest: +SKIP
    >>> da.radarx.plot.hist()  # falls through to da.hvplot.hist  # doctest: +SKIP
    """

    def __call__(self, backend=None, **kwargs):
        da = self._obj
        if "range" in da.dims and da.ndim == 2:
            return self.range_azimuth(backend=backend, **kwargs)
        if {"x", "y"} <= set(da.dims):
            return self.cappi(backend=backend, **kwargs)
        _assign_backend(backend)
        return da.hvplot(**kwargs)

    def range_azimuth(self, backend=None, **kwargs):
        """
        Plot the sweep in its native range-azimuth layout.

        Parameters
        ----------
        backend : {"bokeh", "matplotlib"}, optional
            HoloViews plotting backend. ``None`` (default) keeps the active one.
        **kwargs : dict, optional
            Additional keyword arguments passed to hvplot (e.g. ``clim``, ``cmap``,
            ``frame_width``, ``rasterize``). By default the colour range uses the
            2nd-98th percentiles (``robust=True``) and is not forced symmetric.

        Returns
        -------
        holoviews.QuadMesh

        See Also
        --------
        hvplot_range_azimuth
        """
        return hvplot_range_azimuth(self._obj, backend=backend, **kwargs)

    def ppi(self, backend=None, **kwargs):
        """
        Plot a georeferenced plan-position indicator.

        Parameters
        ----------
        backend : {"bokeh", "matplotlib"}, optional
            HoloViews plotting backend. ``None`` (default) keeps the active one.
        **kwargs : dict, optional
            Additional keyword arguments passed to hvplot (e.g. ``clim``, ``cmap``,
            ``frame_width``, ``rasterize``). By default the colour range uses the
            2nd-98th percentiles (``robust=True``) and is not forced symmetric.

        Returns
        -------
        holoviews.QuadMesh

        See Also
        --------
        hvplot_ppi
        """
        return hvplot_ppi(self._obj, backend=backend, **kwargs)

    def mesh(self, backend=None, **kwargs):
        """
        Plot a PPI with the outline of every gate drawn.

        Parameters
        ----------
        backend : {"bokeh", "matplotlib"}, optional
            HoloViews plotting backend. ``None`` (default) keeps the active one.
        **kwargs : dict, optional
            Additional keyword arguments passed to hvplot (e.g. ``clim``, ``cmap``,
            ``frame_width``, ``rasterize``). By default the colour range uses the
            2nd-98th percentiles (``robust=True``) and is not forced symmetric.

        Returns
        -------
        holoviews.QuadMesh

        See Also
        --------
        hvplot_mesh
        """
        return hvplot_mesh(self._obj, backend=backend, **kwargs)

    def centroids(self, backend=None, **kwargs):
        """
        Plot the gate centres as points coloured by value.

        Parameters
        ----------
        backend : {"bokeh", "matplotlib"}, optional
            HoloViews plotting backend. ``None`` (default) keeps the active one.
        **kwargs : dict, optional
            Additional keyword arguments passed to hvplot (e.g. ``clim``, ``cmap``,
            ``frame_width``, ``rasterize``). By default the colour range uses the
            2nd-98th percentiles (``robust=True``) and is not forced symmetric.

        Returns
        -------
        holoviews.Points

        See Also
        --------
        hvplot_centroids
        """
        return hvplot_centroids(self._obj, backend=backend, **kwargs)

    def rhi(self, backend=None, **kwargs):
        """
        Plot a range-height indicator.

        Parameters
        ----------
        backend : {"bokeh", "matplotlib"}, optional
            HoloViews plotting backend. ``None`` (default) keeps the active one.
        **kwargs : dict, optional
            Additional keyword arguments passed to hvplot (e.g. ``clim``, ``cmap``,
            ``frame_width``, ``rasterize``). By default the colour range uses the
            2nd-98th percentiles (``robust=True``) and is not forced symmetric.

        Returns
        -------
        holoviews.QuadMesh

        See Also
        --------
        hvplot_rhi
        """
        return hvplot_rhi(self._obj, backend=backend, **kwargs)

    def cappi(self, z=None, backend=None, **kwargs):
        """
        Plot the field on the horizontal plane.

        Parameters
        ----------
        z : float, optional
            Height in metres to select from a 3D grid; omit for a slider.
        backend : {"bokeh", "matplotlib"}, optional
            HoloViews plotting backend. ``None`` (default) keeps the active one.
        **kwargs : dict, optional
            Additional keyword arguments passed to hvplot (e.g. ``clim``, ``cmap``,
            ``frame_width``, ``rasterize``). By default the colour range uses the
            2nd-98th percentiles (``robust=True``) and is not forced symmetric.

        Returns
        -------
        holoviews.QuadMesh or holoviews.DynamicMap

        See Also
        --------
        hvplot_cappi
        """
        return hvplot_cappi(self._obj, z=z, backend=backend, **kwargs)

    def max_cappi(self, backend=None, **kwargs):
        """
        Plot a Max-CAPPI with side projections.

        Parameters
        ----------
        backend : {"bokeh", "matplotlib"}, optional
            HoloViews plotting backend. ``None`` (default) keeps the active one.
        **kwargs : dict, optional
            Additional keyword arguments passed to hvplot (e.g. ``clim``, ``cmap``,
            ``frame_width``, ``rasterize``). By default the colour range uses the
            2nd-98th percentiles (``robust=True``) and is not forced symmetric.

        Returns
        -------
        holoviews.Layout

        See Also
        --------
        hvplot_max_cappi
        """
        return hvplot_max_cappi(self._obj, backend=backend, **kwargs)


class RadarxDatasetPlotAccessor(_RadarxPlotAccessor):
    """
    Interactive plots of a sweep or gridded dataset, as ``ds.radarx.plot``.

    Methods taking ``variables`` return one panel per variable (a facet
    grid); when omitted, all radar fields in the dataset are used. Any
    attribute not defined here is looked up on ``ds.hvplot``.

    Parameters
    ----------
    obj : xarray.Dataset
        Sweep or gridded dataset to plot.

    Examples
    --------
    >>> ds.radarx.plot.ppi(["DBZH", "VRADH"])  # doctest: +SKIP
    >>> grid.radarx.plot.max_cappi("DBZH")  # doctest: +SKIP
    """

    def __call__(self, variables=None, backend=None, **kwargs):
        if "range" in self._obj.dims:
            return self.ppi(variables, backend=backend, **kwargs)
        return self.cappi(variables, backend=backend, **kwargs)

    def _facet(self, func, variables, cols, **kwargs):
        if isinstance(variables, str):
            return func(self._obj[variables], **kwargs)
        variables = variables or _radar_variables(self._obj)
        if not variables:
            raise ValueError("No radar variables found to plot.")
        da_kwargs = {}
        for name in ("sweep_fixed_angle", "fixed_angle"):
            if name in self._obj and self._obj[name].size == 1:
                da_kwargs[name] = self._obj[name]
        panels = [
            func(self._obj[v].assign_coords(da_kwargs), **kwargs) for v in variables
        ]
        return _layout(panels, cols)

    def ppi(self, variables=None, cols=2, backend=None, **kwargs):
        """
        Plot one PPI per variable.

        Parameters
        ----------
        variables : str or list of str, optional
            Variable(s) to plot. A single name returns a single panel; ``None``
            plots every radar field in the dataset.
        cols : int, optional
            Number of panels per row. Default is 2.
        backend : {"bokeh", "matplotlib"}, optional
            HoloViews plotting backend. ``None`` (default) keeps the active one.
        **kwargs : dict, optional
            Additional keyword arguments passed to hvplot (e.g. ``clim``, ``cmap``,
            ``frame_width``, ``rasterize``). By default the colour range uses the
            2nd-98th percentiles (``robust=True``) and is not forced symmetric.

        Returns
        -------
        holoviews.Layout or holoviews.Element
            One panel per variable, or a single element for a single variable.

        See Also
        --------
        hvplot_ppi
        """
        return self._facet(hvplot_ppi, variables, cols, backend=backend, **kwargs)

    def rhi(self, variables=None, cols=2, backend=None, **kwargs):
        """
        Plot one RHI per variable.

        Parameters
        ----------
        variables : str or list of str, optional
            Variable(s) to plot. A single name returns a single panel; ``None``
            plots every radar field in the dataset.
        cols : int, optional
            Number of panels per row. Default is 2.
        backend : {"bokeh", "matplotlib"}, optional
            HoloViews plotting backend. ``None`` (default) keeps the active one.
        **kwargs : dict, optional
            Additional keyword arguments passed to hvplot (e.g. ``clim``, ``cmap``,
            ``frame_width``, ``rasterize``). By default the colour range uses the
            2nd-98th percentiles (``robust=True``) and is not forced symmetric.

        Returns
        -------
        holoviews.Layout or holoviews.Element
            One panel per variable, or a single element for a single variable.

        See Also
        --------
        hvplot_rhi
        """
        return self._facet(hvplot_rhi, variables, cols, backend=backend, **kwargs)

    def cappi(self, variables=None, z=None, cols=2, backend=None, **kwargs):
        """
        Plot one CAPPI per variable.

        Parameters
        ----------
        variables : str or list of str, optional
            Variable(s) to plot. A single name returns a single panel; ``None``
            plots every radar field in the dataset.
        z : float, optional
            Height in metres to select from a 3D grid; omit for a slider.
        cols : int, optional
            Number of panels per row. Default is 2.
        backend : {"bokeh", "matplotlib"}, optional
            HoloViews plotting backend. ``None`` (default) keeps the active one.
        **kwargs : dict, optional
            Additional keyword arguments passed to hvplot (e.g. ``clim``, ``cmap``,
            ``frame_width``, ``rasterize``). By default the colour range uses the
            2nd-98th percentiles (``robust=True``) and is not forced symmetric.

        Returns
        -------
        holoviews.Layout or holoviews.Element
            One panel per variable, or a single element for a single variable.

        See Also
        --------
        hvplot_cappi
        """
        return self._facet(
            hvplot_cappi, variables, cols, z=z, backend=backend, **kwargs
        )

    def mesh(self, variable, backend=None, **kwargs):
        """
        Plot a PPI of one variable with gate outlines.

        Parameters
        ----------
        variable : str
            Variable to plot.
        backend : {"bokeh", "matplotlib"}, optional
            HoloViews plotting backend. ``None`` (default) keeps the active one.
        **kwargs : dict, optional
            Additional keyword arguments passed to hvplot (e.g. ``clim``, ``cmap``,
            ``frame_width``, ``rasterize``). By default the colour range uses the
            2nd-98th percentiles (``robust=True``) and is not forced symmetric.

        Returns
        -------
        holoviews.QuadMesh

        See Also
        --------
        hvplot_mesh
        """
        return hvplot_mesh(self._obj[variable], backend=backend, **kwargs)

    def centroids(self, variable, backend=None, **kwargs):
        """
        Plot the gate centres of one variable as points.

        Parameters
        ----------
        variable : str
            Variable to plot.
        backend : {"bokeh", "matplotlib"}, optional
            HoloViews plotting backend. ``None`` (default) keeps the active one.
        **kwargs : dict, optional
            Additional keyword arguments passed to hvplot (e.g. ``clim``, ``cmap``,
            ``frame_width``, ``rasterize``). By default the colour range uses the
            2nd-98th percentiles (``robust=True``) and is not forced symmetric.

        Returns
        -------
        holoviews.Points

        See Also
        --------
        hvplot_centroids
        """
        return hvplot_centroids(self._obj[variable], backend=backend, **kwargs)

    def max_cappi(self, variable, backend=None, **kwargs):
        """
        Plot a Max-CAPPI of one gridded variable.

        Parameters
        ----------
        variable : str
            Variable to plot.
        backend : {"bokeh", "matplotlib"}, optional
            HoloViews plotting backend. ``None`` (default) keeps the active one.
        **kwargs : dict, optional
            Additional keyword arguments passed to hvplot (e.g. ``clim``, ``cmap``,
            ``frame_width``, ``rasterize``). By default the colour range uses the
            2nd-98th percentiles (``robust=True``) and is not forced symmetric.

        Returns
        -------
        holoviews.Layout

        See Also
        --------
        hvplot_max_cappi
        """
        return hvplot_max_cappi(self._obj[variable], backend=backend, **kwargs)


class RadarxDataTreePlotAccessor:
    """
    Interactive plots across the sweeps of a volume, as ``dt.radarx.plot``.

    Methods return one panel per sweep (or a single panel when only one
    sweep is selected). Calling the accessor is the same as :meth:`ppi`.

    Parameters
    ----------
    dtree : xarray.DataTree
        Radar volume with ``sweep_*`` groups, e.g. from xradar.

    Examples
    --------
    >>> dt.radarx.plot.ppi("DBZH", sweeps=[0, 2])  # doctest: +SKIP
    >>> dt.radarx.plot.cappi("DBZH", height=2000)  # doctest: +SKIP
    """

    __slots__ = ("_obj",)

    def __init__(self, dtree):
        _ensure_hvplot_imported()
        self._obj = dtree

    def __call__(self, variable, sweeps=None, backend=None, **kwargs):
        return self.ppi(variable, sweeps=sweeps, backend=backend, **kwargs)

    def _sweeps(self, sweeps, mode=None):
        names = [n for n in self._obj.children if n.startswith("sweep")]
        if sweeps is not None:
            sweeps = [sweeps] if isinstance(sweeps, (str, int)) else sweeps
            names = [names[s] if isinstance(s, int) else s for s in sweeps]
        elif mode is not None:
            matching = [
                n
                for n in names
                if mode in str(self._obj[n].to_dataset().get("sweep_mode", "").values)
            ]
            names = matching or names
        if not names:
            raise ValueError("No sweep groups found in DataTree.")
        return names

    def _sweep_dataarray(self, name, variable):
        ds = self._obj[name].to_dataset()
        da = ds[variable]
        if "sweep_fixed_angle" in ds and ds["sweep_fixed_angle"].size == 1:
            da = da.assign_coords(sweep_fixed_angle=ds["sweep_fixed_angle"])
        for site in ("altitude",):
            if site not in da.coords and site in self._obj.ds:
                da = da.assign_coords({site: self._obj.ds[site]})
        return da

    def _facet(self, func, variable, sweeps, cols, mode=None, **kwargs):
        panels = [
            func(self._sweep_dataarray(name, variable), **kwargs)
            for name in self._sweeps(sweeps, mode=mode)
        ]
        return panels[0] if len(panels) == 1 else _layout(panels, cols)

    def ppi(self, variable, sweeps=None, cols=2, backend=None, **kwargs):
        """
        Plot one PPI per sweep.

        Parameters
        ----------
        variable : str
            Variable to plot.
        sweeps : int, str or list of int or str, optional
            Sweep groups to plot, by index (``0``) or name (``"sweep_0"``).
            Default is all sweeps.
        cols : int, optional
            Number of panels per row. Default is 2.
        backend : {"bokeh", "matplotlib"}, optional
            HoloViews plotting backend. ``None`` (default) keeps the active one.
        **kwargs : dict, optional
            Additional keyword arguments passed to hvplot (e.g. ``clim``, ``cmap``,
            ``frame_width``, ``rasterize``). By default the colour range uses the
            2nd-98th percentiles (``robust=True``) and is not forced symmetric.

        Returns
        -------
        holoviews.Layout or holoviews.Element
            One panel per sweep, or a single element for a single sweep.

        See Also
        --------
        hvplot_ppi
        """
        return self._facet(
            hvplot_ppi, variable, sweeps, cols, backend=backend, **kwargs
        )

    def rhi(self, variable, sweeps=None, cols=2, backend=None, **kwargs):
        """
        Plot one RHI per sweep.

        Parameters
        ----------
        variable : str
            Variable to plot.
        sweeps : int, str or list of int or str, optional
            Sweep groups to plot, by index (``0``) or name (``"sweep_0"``).
            Default is all sweeps with ``sweep_mode`` RHI, or all sweeps if
            there are none.
        cols : int, optional
            Number of panels per row. Default is 2.
        backend : {"bokeh", "matplotlib"}, optional
            HoloViews plotting backend. ``None`` (default) keeps the active one.
        **kwargs : dict, optional
            Additional keyword arguments passed to hvplot (e.g. ``clim``, ``cmap``,
            ``frame_width``, ``rasterize``). By default the colour range uses the
            2nd-98th percentiles (``robust=True``) and is not forced symmetric.

        Returns
        -------
        holoviews.Layout or holoviews.Element
            One panel per sweep, or a single element for a single sweep.

        See Also
        --------
        hvplot_rhi
        """
        return self._facet(
            hvplot_rhi, variable, sweeps, cols, mode="rhi", backend=backend, **kwargs
        )

    def mesh(self, variable, sweeps=0, cols=2, backend=None, **kwargs):
        """
        Plot PPIs with gate outlines.

        Parameters
        ----------
        variable : str
            Variable to plot.
        sweeps : int, str or list of int or str, optional
            Sweep groups to plot, by index (``0``) or name (``"sweep_0"``).
            Default is the first sweep.
        cols : int, optional
            Number of panels per row. Default is 2.
        backend : {"bokeh", "matplotlib"}, optional
            HoloViews plotting backend. ``None`` (default) keeps the active one.
        **kwargs : dict, optional
            Additional keyword arguments passed to hvplot (e.g. ``clim``, ``cmap``,
            ``frame_width``, ``rasterize``). By default the colour range uses the
            2nd-98th percentiles (``robust=True``) and is not forced symmetric.

        Returns
        -------
        holoviews.Layout or holoviews.Element
            One panel per sweep, or a single element for a single sweep.

        See Also
        --------
        hvplot_mesh
        """
        return self._facet(
            hvplot_mesh, variable, sweeps, cols, backend=backend, **kwargs
        )

    def centroids(self, variable, sweeps=0, cols=2, backend=None, **kwargs):
        """
        Plot gate centres as points.

        Parameters
        ----------
        variable : str
            Variable to plot.
        sweeps : int, str or list of int or str, optional
            Sweep groups to plot, by index (``0``) or name (``"sweep_0"``).
            Default is the first sweep.
        cols : int, optional
            Number of panels per row. Default is 2.
        backend : {"bokeh", "matplotlib"}, optional
            HoloViews plotting backend. ``None`` (default) keeps the active one.
        **kwargs : dict, optional
            Additional keyword arguments passed to hvplot (e.g. ``clim``, ``cmap``,
            ``frame_width``, ``rasterize``). By default the colour range uses the
            2nd-98th percentiles (``robust=True``) and is not forced symmetric.

        Returns
        -------
        holoviews.Layout or holoviews.Element
            One panel per sweep, or a single element for a single sweep.

        See Also
        --------
        hvplot_centroids
        """
        return self._facet(
            hvplot_centroids, variable, sweeps, cols, backend=backend, **kwargs
        )

    def cappi(self, variable, height, method="cartesian_idw", backend=None, **kwargs):
        """
        Retrieve a CAPPI at ``height`` and plot it.

        The volume is georeferenced first if needed.

        Parameters
        ----------
        variable : str
            Variable to retrieve and plot.
        height : float
            Target CAPPI altitude in metres.
        method : str, optional
            Retrieval method passed to :func:`radarx.retrieve.create_cappi`.
            Default is ``"cartesian_idw"``.
        backend : {"bokeh", "matplotlib"}, optional
            HoloViews plotting backend. ``None`` (default) keeps the active one.
        **kwargs : dict, optional
            Retrieval keywords of :func:`radarx.retrieve.create_cappi`
            (``vertical_tolerance``, ``apply_filter``, ``sweeps``, ``x``,
            ``y``, ``x_res``, ``y_res``, ``padding``); all remaining keywords
            are passed to hvplot.

        Returns
        -------
        holoviews.QuadMesh
            Plan view of the retrieved CAPPI.

        See Also
        --------
        hvplot_cappi
        """
        from ..retrieve import create_cappi

        retrieve_keys = (
            "vertical_tolerance",
            "apply_filter",
            "sweeps",
            "x",
            "y",
            "x_res",
            "y_res",
            "padding",
        )
        retrieve_kwargs = {k: kwargs.pop(k) for k in retrieve_keys if k in kwargs}
        dtree = self._obj
        first = dtree[self._sweeps(None)[0]]
        if "x" not in first.coords and "x" not in first.data_vars:
            dtree = dtree.xradar.georeference()
        ds = create_cappi(
            dtree, height=height, method=method, fields=[variable], **retrieve_kwargs
        )
        return hvplot_cappi(ds[variable], backend=backend, **kwargs)
