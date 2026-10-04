#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx UGRID
============

Represent a radar sweep as an unstructured grid in the
`UGRID <https://ugrid-conventions.github.io/ugrid-conventions/>`_ conventions
and wrap it in a `uxarray <https://uxarray.readthedocs.io/>`_ dataset.

Every radar gate becomes one quadrilateral face. Its corner nodes sit halfway
between neighbouring rays and gates, so faces describe the true gate
footprint, which grows with range. This follows the discussion in
`openradar/xradar#212 <https://github.com/openradar/xradar/issues/212>`_ and
`UXARRAY/uxarray#976 <https://github.com/UXARRAY/uxarray/issues/976>`_.

uxarray is an optional dependency (``pip install uxarray``).

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = ["gate_corners", "to_uxarray"]

__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np
import xarray as xr


def _edges(centers, full_circle=False):
    """
    Return cell edges halfway between sorted cell centres.

    Parameters
    ----------
    centers : numpy.ndarray
        1D cell centres, in order.
    full_circle : bool, optional
        Treat ``centers`` as angles on a closed 360 degree circle and return
        one edge per centre (the last edge closes back onto the first).

    Returns
    -------
    numpy.ndarray
        ``len(centers) + 1`` edges, or ``len(centers)`` for a full circle.
    """
    centers = np.asarray(centers, dtype=float)
    if full_circle:
        # gap to the previous centre; the first one wraps across 360 degrees
        gaps = np.diff(np.concatenate([[centers[-1] - 360.0], centers]))
        return np.mod(centers - gaps / 2, 360.0)
    mid = (centers[1:] + centers[:-1]) / 2
    first = centers[0] - (mid[0] - centers[0])
    last = centers[-1] + (centers[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def _is_full_circle(azimuth, tolerance=None):
    """True if sorted azimuths cover the whole circle without a gap."""
    az = np.sort(np.mod(np.asarray(azimuth, dtype=float), 360.0))
    if az.size < 3:
        return False
    gaps = np.diff(np.concatenate([az, [az[0] + 360.0]]))
    res = np.median(gaps)
    tolerance = 2 * res if tolerance is None else tolerance
    return bool(gaps.max() <= tolerance)


def _site(obj):
    """Radar site latitude, longitude and altitude from coordinates."""
    try:
        lat = float(obj["latitude"])
        lon = float(obj["longitude"])
    except KeyError as err:
        raise ValueError(
            "The sweep needs 'latitude' and 'longitude' of the radar site, e.g. "
            "from dtree['sweep_0'].to_dataset(inherit='all_coords')."
        ) from err
    alt = (
        float(obj["altitude"]) if "altitude" in obj.coords or "altitude" in obj else 0.0
    )
    return lat, lon, alt


def gate_corners(obj):
    """
    Compute the geographic corner nodes of every gate of a sweep.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        PPI sweep on ``azimuth`` and ``range`` dimensions with ``elevation``
        and the radar site ``latitude``/``longitude`` (``altitude``
        optional) as coordinates or variables.

    Returns
    -------
    node_lon, node_lat : numpy.ndarray
        Corner longitudes and latitudes, shape ``(n_ray_edges, n_range + 1)``,
        where ``n_ray_edges`` is ``n_ray`` for a full 360 degree sweep and
        ``n_ray + 1`` otherwise.
    order : numpy.ndarray
        Ray indices that sort the sweep by azimuth; faces follow this order.
    full_circle : bool
        Whether the sweep closes on itself.
    """
    import pyproj
    from xradar.georeference import antenna_to_cartesian

    if "range" not in obj.dims or "azimuth" not in obj.coords:
        raise ValueError("Expected a PPI sweep with 'azimuth' and 'range'.")
    lat, lon, alt = _site(obj)

    azimuth = np.mod(np.asarray(obj["azimuth"].values, dtype=float), 360.0)
    order = np.argsort(azimuth, kind="stable")
    azimuth = azimuth[order]
    elevation = np.broadcast_to(
        np.asarray(obj["elevation"].values, dtype=float), obj["azimuth"].shape
    )[order]
    full_circle = _is_full_circle(azimuth)

    az_edges = _edges(azimuth, full_circle=full_circle)
    if full_circle:
        el_edges = (elevation + np.roll(elevation, 1)) / 2
    else:
        el_edges = _edges(elevation)
    r_edges = np.clip(_edges(obj["range"].values), 0, None)

    r2, az2 = np.meshgrid(r_edges, az_edges)
    el2 = np.broadcast_to(el_edges[:, None], r2.shape)
    x, y, _ = antenna_to_cartesian(r2, az2, el2, site_altitude=alt)

    aeqd = pyproj.Proj(proj="aeqd", lat_0=lat, lon_0=lon, datum="WGS84")
    node_lon, node_lat = aeqd(x, y, inverse=True)
    return np.asarray(node_lon), np.asarray(node_lat), order, full_circle


def _face_node_connectivity(n_ray_edges, n_range, full_circle):
    """Counter-clockwise quad connectivity for a (ray, range) node lattice."""
    n_ray = n_ray_edges if full_circle else n_ray_edges - 1
    node = np.arange(n_ray_edges * (n_range + 1)).reshape(n_ray_edges, n_range + 1)
    nxt = np.roll(node, -1, axis=0) if full_circle else node[1:]
    cur = node[:n_ray]
    # ray i -> i + 1 turns clockwise (meteorological azimuth), so walk
    # inner(i) -> inner(i+1) -> outer(i+1) -> outer(i) for counter-clockwise
    faces = np.stack(
        [cur[:, :-1], nxt[:n_ray, :-1], nxt[:n_ray, 1:], cur[:, 1:]], axis=-1
    )
    return faces.reshape(-1, 4)


def to_uxarray(obj, variables=None):
    """
    Convert a radar sweep into a uxarray dataset with one face per gate.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        PPI sweep on ``azimuth`` and ``range`` dimensions; see
        :func:`gate_corners`.
    variables : str or list of str, optional
        Variables to attach to the faces. By default all data variables on
        the ``(azimuth, range)`` dimensions.

    Returns
    -------
    uxarray.UxDataset
        Dataset on the ``n_face`` dimension with a UGRID grid (``.uxgrid``).
        Faces are ordered by azimuth, then range; ``azimuth`` and ``range``
        of each face are kept as coordinates.

    Raises
    ------
    ImportError
        If uxarray is not installed.
    ValueError
        If ``obj`` is not a PPI sweep or lacks the radar site location.

    Examples
    --------
    >>> sweep = dtree["sweep_0"].to_dataset(inherit="all_coords")  # doctest: +SKIP
    >>> uxds = sweep.radarx.to_uxarray(["DBZH"])  # doctest: +SKIP
    >>> uxds["DBZH"].plot()  # gate polygons  # doctest: +SKIP
    """
    try:
        import uxarray as ux
    except ImportError as err:
        raise ImportError(
            "to_uxarray requires uxarray. Install it with: pip install uxarray"
        ) from err

    ds = obj.to_dataset() if isinstance(obj, xr.DataArray) else obj
    node_lon, node_lat, order, full_circle = gate_corners(ds)
    n_ray_edges, n_edges_range = node_lon.shape
    n_range = n_edges_range - 1

    faces = _face_node_connectivity(n_ray_edges, n_range, full_circle)
    grid = ux.Grid.from_topology(node_lon.ravel(), node_lat.ravel(), faces)

    ray_dim = "azimuth" if "azimuth" in ds.dims else ds["azimuth"].dims[0]
    if variables is None:
        variables = [
            name
            for name, da in ds.data_vars.items()
            if set(da.dims) == {ray_dim, "range"}
        ]
    elif isinstance(variables, str):
        variables = [variables]
    if not variables:
        raise ValueError("No (azimuth, range) variables to convert.")

    azimuth = ds["azimuth"].values[order]
    data_vars = {}
    for name in variables:
        da = ds[name].transpose(ray_dim, "range")
        values = np.asarray(da.values)[order].reshape(-1)
        data_vars[name] = ux.UxDataArray(
            values, dims=["n_face"], name=name, attrs=da.attrs, uxgrid=grid
        )
    uxds = ux.UxDataset(data_vars, uxgrid=grid)
    return uxds.assign_coords(
        azimuth=("n_face", np.repeat(azimuth, n_range)),
        range=("n_face", np.tile(ds["range"].values, azimuth.size)),
    )
