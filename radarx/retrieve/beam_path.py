#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Beam Path through a Refractivity Profile
========================================

True beam height and ground range of every gate of a sweep or volume, from the
refractivity of a sounding or reanalysis profile, as an alternative to the 4/3
Earth model. The ray tracer (Snell's law in a spherically stratified
atmosphere, after Bean and Dutton 1966), its equations, its limits and its
agreement with the 4/3 Earth model are described in
:mod:`radarx.fundamentals.refraction`; this module puts it on the gates of
radar data.

The refractivity :math:`N` comes from the profile: a ``refractivity`` variable,
else it is computed from ``pressure``, ``temperature`` and the water vapour
(``vapor_pressure``, ``specific_humidity``, ``dewpoint`` or
``relative_humidity``, in that order of preference) with
:func:`radarx.io.sounding.refractivity` (formula of Bean and Dutton, Eq. 1.17).
A profile on a ``height`` coordinate in m above sea level, as returned by the
readers of :mod:`radarx.io.sounding`, is the input; so is a pair
``(height, refractivity)`` of arrays.

Products
--------
``beam_height``
    Height of the beam centre above sea level (m).
``ground_range``
    Distance on the ground at sea level from the radar (m).
``gate_elevation``
    Elevation angle of the ray at the gate (degrees).
``k_e``
    Effective Earth radius factor that gives the traced height with the
    4/3 Earth model formula for that gate
    (:func:`radarx.fundamentals.refraction.k_e_from_rays`); its median over a
    sweep is the ``k_e`` for the gridding, merging and beam blockage
    functions, which keep the effective radius model.

The slant range of the gates is taken as the length of the ray (the apparent
range of the radar differs by at most about 200 m, see the fundamentals
module). The numerical work runs in a compiled, multithreaded C++ kernel with a
NumPy twin; the result stays lazy for a dask-backed sweep.

References
----------
.. [1] Bean, B. R., and E. J. Dutton, 1966: Radio Meteorology. NBS Monograph
   92, National Bureau of Standards, https://doi.org/10.6028/NBS.MONO.92

.. autosummary::
   :nosignatures:
   :toctree: generated/

   beam_path
"""

from __future__ import annotations

__all__ = ["beam_path"]

import numpy as np
import xarray as xr

from .._fields import as_sweep, node_dataset
from .._provenance import provenance
from .._registry import accessor_method
from ..fundamentals.constants import EARTH_RADIUS
from ..fundamentals.refraction import k_e_from_rays, trace_rays
from ..grid.cone import _sweep_names
from ._fieldtools import check_object
from ._products import product_tree

_RANGE = "range"
_HUMIDITY = ("vapor_pressure", "specific_humidity", "dewpoint", "relative_humidity")

_ATTRS = {
    "beam_height": {
        "long_name": "Beam centre height above sea level (ray traced)",
        "units": "m",
    },
    "ground_range": {
        "long_name": "Distance on the ground at sea level (ray traced)",
        "units": "m",
    },
    "gate_elevation": {
        "long_name": "Elevation of the ray at the gate",
        "units": "degree",
    },
    "k_e": {
        "long_name": "Effective Earth radius factor of the traced beam",
        "units": "1",
    },
}


def _vapor_pressure(profile, pressure, temperature, engine):
    """Water vapour pressure (Pa) of a profile from the first humidity variable."""
    from ..io import sounding as snd

    for name in _HUMIDITY:
        if name not in profile:
            continue
        if name == "vapor_pressure":
            return profile[name]
        if name == "specific_humidity":
            return snd._wrap(
                "vapor_pressure",
                "vapor_pressure",
                profile[name],
                pressure,
                engine=engine,
            )
        if name == "dewpoint":
            return snd.saturation_vapor_pressure(profile[name], engine=engine)
        return profile[name] * snd.saturation_vapor_pressure(temperature, engine=engine)
    raise ValueError(
        "the profile has no water vapour: give 'refractivity' or one of "
        f"{_HUMIDITY} besides 'pressure' and 'temperature'"
    )


def _profile_arrays(profile, form, engine):
    """``(height, refractivity)`` NumPy arrays of a profile."""
    if isinstance(profile, xr.Dataset):
        if "height" not in profile.coords:
            raise ValueError("the profile needs a 'height' coordinate (m)")
        if "refractivity" in profile:
            n = profile["refractivity"]
        else:
            from ..io.sounding import refractivity

            for name in ("pressure", "temperature"):
                if name not in profile:
                    raise ValueError(
                        f"the profile needs 'refractivity' or {name!r}, "
                        "'pressure' and 'temperature' with water vapour"
                    )
            p, t = profile["pressure"], profile["temperature"]
            e = _vapor_pressure(profile, p, t, engine)
            n = refractivity(p, t, e, form=form, engine=engine)
        return np.asarray(profile["height"].values), np.asarray(n.values)
    try:
        height, n = profile
    except (TypeError, ValueError):
        raise TypeError(
            "profile must be an xarray.Dataset on a 'height' coordinate or a "
            "pair (height, refractivity)"
        ) from None
    return np.asarray(height), np.asarray(n)


def _altitude(ds, radar_height):
    """Radar altitude (m): the argument, else the ``altitude`` of the data."""
    if radar_height is not None:
        return float(radar_height)
    if "altitude" in ds.variables:
        value = np.asarray(ds["altitude"].values, dtype=np.float64).ravel()
        value = value[np.isfinite(value)]
        if value.size:
            return float(value[0])
    raise ValueError(
        "the radar altitude is not in the data; pass radar_height (m above sea level)"
    )


def _elevation(ds):
    """Elevation of every ray as a DataArray (a scalar broadcast on the rays)."""
    for name in ("elevation", "sweep_fixed_angle"):
        if name in ds.variables:
            return ds[name], name
    raise ValueError(
        "the sweep needs an 'elevation' (one value per ray) or a "
        "'sweep_fixed_angle' variable"
    )


def _ray_chunks(ds, ray):
    """Chunk sizes of the rays when the sweep is a dask array, else None."""
    try:
        return ds.chunksizes.get(ray)
    except ValueError:  # pragma: no cover - chunks that differ between variables
        return None


def _sweep_products(ds, height, refractivity, options, radar_height):
    """The products of one sweep as a Dataset on its coordinates."""
    if _RANGE not in ds.coords:
        raise ValueError("the sweep needs a 'range' coordinate")
    step, earth_radius, n_threads, engine = options
    elev, _ = _elevation(ds)
    if elev.ndim == 0:
        # one elevation for the sweep: the products on (ray, range) like the fields
        ray = next(
            (d for d in ("azimuth", "time") if d in ds.dims),
            next((d for d in ds.dims if d != _RANGE), None),
        )
        if ray is not None:
            elev = xr.DataArray(np.full(ds.sizes[ray], float(elev.values)), dims=(ray,))
            chunks = _ray_chunks(ds, ray)
            if chunks is not None:
                elev = elev.chunk({ray: chunks})
    rng = ds[_RANGE].astype(np.float64)
    altitude = _altitude(ds, radar_height)
    traced = trace_rays(
        rng,
        elev.astype(np.float64).reset_coords(drop=True),
        height,
        refractivity,
        radar_height=altitude,
        step=step,
        earth_radius=earth_radius,
        n_threads=n_threads,
        engine=engine,
    )
    k_e = k_e_from_rays(
        rng.reset_coords(drop=True),
        elev.astype(np.float64).reset_coords(drop=True),
        traced["beam_height"],
        altitude,
    )
    out = xr.Dataset(
        {
            "beam_height": traced["beam_height"].astype(np.float32),
            "ground_range": traced["ground_range"].astype(np.float32),
            "gate_elevation": traced["gate_elevation"].astype(np.float32),
            "k_e": k_e.transpose(*traced["beam_height"].dims).astype(np.float32),
        }
    )
    dims = set(out.dims)
    coords = {
        n: c.variable.copy(deep=False)
        for n, c in ds.coords.items()
        if set(c.dims) <= dims and n not in out.coords
    }
    out = out.assign_coords(coords)
    for var, attrs in _ATTRS.items():
        out[var].attrs = dict(attrs)
    out.attrs = {"beam_path_step": float(step), "beam_path_radar_height": altitude}
    return out


@provenance(
    "Ray tracing of the radar beam through a refractivity profile "
    "(Snell's law in a spherically stratified atmosphere), after Bean and "
    "Dutton (1966)"
)
def beam_path(
    obj,
    profile,
    *,
    radar_height=None,
    form="bean_dutton",
    step=100.0,
    earth_radius=EARTH_RADIUS,
    n_threads=None,
    engine="auto",
):
    """
    True beam height and ground range of every gate from a refractivity profile.

    Integrates the ray of every beam elevation through the profile
    (:func:`radarx.fundamentals.refraction.trace_rays`) and returns the beam
    height, the ground range, the elevation of the ray and the equivalent
    effective Earth radius factor on the gates of the sweep. The equations
    and limits are in :mod:`radarx.fundamentals.refraction` and
    :mod:`radarx.retrieve.beam_path`.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        A sweep with ``range`` (m), ``elevation`` (one value per ray, else
        ``sweep_fixed_angle``) and the radar ``altitude`` (a coordinate, else
        pass ``radar_height``), or a volume with ``sweep_*`` groups. A
        dask-backed input gives a lazy result.
    profile : xarray.Dataset or tuple
        Profile on a ``height`` coordinate (m above sea level) with
        ``refractivity``, or with ``pressure`` (Pa), ``temperature`` (K) and a
        humidity variable (see :mod:`radarx.retrieve.beam_path`); or a pair
        ``(height, refractivity)`` of arrays. Not changed.
    radar_height : float, optional
        Radar altitude (m above sea level). Default: the ``altitude`` of the
        data.
    form : {"bean_dutton", "smith_weintraub", "itu"}, optional
        Constants of the refractivity formula when it is computed from
        pressure, temperature and humidity
        (:func:`radarx.io.sounding.refractivity`).
    step : float, optional
        Longest path step of the integration (m). Default 100.
    earth_radius : float, optional
        Radius of the spherical Earth (m). Default 6371 km.
    n_threads : int, optional
        Threads of the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation. ``"auto"`` (default) prefers the compiled kernel.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        On the coordinates of the input: ``beam_height`` (m above sea level),
        ``ground_range`` (m), ``gate_elevation`` (degrees) and ``k_e``, all
        float32. For a volume a DataTree with one such node per sweep and the
        root of the input. Merge it into the input with
        ``dtree.radarx.assign(products)``. The input is not changed.

    Raises
    ------
    ValueError
        For a profile without refractivity or water vapour, a sweep without
        ``range``, elevation or radar altitude, or an unknown option.
    TypeError
        If ``obj`` is not an xarray object or ``profile`` of the wrong type.
    ImportError
        If ``engine="compiled"`` and the kernel is not available.

    References
    ----------
    .. [1] Bean, B. R., and E. J. Dutton, 1966: Radio Meteorology. NBS
       Monograph 92, National Bureau of Standards,
       https://doi.org/10.6028/NBS.MONO.92

    Examples
    --------
    >>> sounding = read_sounding("72451", "2024-05-20 00:00")  # doctest: +SKIP
    >>> path = beam_path(sweep, sounding)  # doctest: +SKIP
    >>> k_e = float(path.k_e.median())  # doctest: +SKIP
    >>> paths = dtree.radarx.beam_path(sounding)  # doctest: +SKIP
    """
    check_object(obj)
    height, n = _profile_arrays(profile, form, engine)
    options = (step, earth_radius, n_threads, engine)
    obj = as_sweep(obj)
    if isinstance(obj, xr.DataTree):
        nodes = {
            name: _sweep_products(
                node_dataset(obj[name]), height, n, options, radar_height
            )
            for name in _sweep_names(obj)
        }
        return product_tree(obj, nodes)
    ds = (
        obj.to_dataset(name=obj.name or "field")
        if isinstance(obj, xr.DataArray)
        else obj
    )
    return _sweep_products(ds, height, n, options, radar_height)


@accessor_method("dataarray", "dataset", "datatree", name="beam_path")
def _beam_path_accessor(self, profile, **kwargs):
    """
    True beam height and ground range of every gate from a refractivity profile.

    See :func:`radarx.retrieve.beam_path` for the parameters.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``beam_height``, ``ground_range``, ``gate_elevation`` and ``k_e``.
    """
    return beam_path(self.xarray_obj, profile, **kwargs)
