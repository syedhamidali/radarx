#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Trapping Layers and Anomalous Propagation from a Refractivity Profile
=====================================================================

:func:`ducting_layers` finds the layers of a refractivity profile that can
trap the radar beam (ducts) and says which rays they trap, so that a sounding
or a reanalysis profile tells where the radar sees anomalous propagation
before the beam is traced with :func:`radarx.retrieve.beam_path`.

Equations
---------
The modified refractivity (Bean and Dutton 1966, Table 1.9, Section 4.4.4) [1]_

.. math::

    M(h) = N(h) + \\frac{h}{a}\\,10^{6}

adds the height term of the Earth's curvature to the refractivity :math:`N`
(N-units), with the height :math:`h` above sea level and the radius of the
Earth :math:`a`. A ray stays near the ground where :math:`dM/dh < 0`: a layer
whose gradient is below :math:`-10^6/a \\approx -157` N-units per km
(Eq. 4.23 of [1]_) is a *trapping layer*, a duct. In a layer from the height
:math:`h_b` (base) to :math:`h_t` (top) with :math:`M` decreasing, the
*M-deficit* :math:`\\Delta M = M(h_b) - M(h_t)` is its strength.

A ray that leaves a point at :math:`(n, r)` with the elevation :math:`\\theta`
obeys Snell's law, :math:`n r \\cos\\theta = \\mathrm{const}` (Eq. 3.1 of [1]_).
It turns over (:math:`\\theta = 0`) where :math:`n r` falls to this constant;
inside a trapping layer :math:`n r` decreases with the height, so a ray is
trapped if it turns over before the top, that is for the elevations below the
*angle of penetration* (Eq. 4.17 of [1]_)

.. math::

    \\cos\\theta_p = \\frac{n_t r_t}{n r},

with :math:`n_t r_t` at the top of the layer and :math:`n r` at the start of the
ray. For a radar at the base this is ``trapping_angle``; for a radar inside the
layer at the height ``radar_height`` it is ``radar_trapping_angle``. For small
angles :math:`\\theta_p \\approx \\sqrt{2\\,\\Delta M\\,10^{-6}}` (Eq. 4.22
of [1]_), the trapping angles of the book (a 100 m layer with a gradient of
-0.5 N-units per m: 8.3 mrad; 30 m: 4.5 mrad) and the exact angle are
compared in ``tests/test_ducts.py``, together with rays that are traced with
:func:`radarx.fundamentals.trace_rays` just below and just above the angle.
Rays above the angle leave the layer; below it they follow the layer for as
far as the profile is horizontally homogeneous.

Choices of radarx
-----------------
* The profile is linear between its levels, as for the ray tracer, so the
  trapping layers start and end at levels of the profile; a profile with
  noise at the resolution of the levels shows small layers (use
  ``min_deficit`` to leave them out).
* A layer that starts at the lowest level of the profile is ``surface_based``
  (Bean and Dutton: ground-based); the others are elevated trapping layers.
  The classification into surface ducts, surface-based and elevated ducts of
  the later literature (the height where :math:`M` returns to its value at the
  base) is not made; the layer thickness is the thickness of the trapping
  layer, not the thickness of the duct.
* The water vapour of the refractivity follows :func:`radarx.retrieve.beam_path`.

Limits
------
Horizontally homogeneous refractivity; geometric optics. Whether a duct
traps the wavelength of a given radar also depends on the thickness of the
duct: the longest trapped wavelength follows from the thickness and the
gradient (Eqs. 4.24 and 4.25 of [1]_), a thin layer traps only short
wavelengths. This is not applied here, so the layers are those of geometric
optics. The number of layers is at most ``max_layers`` per profile (``layer_count`` gives
the total).

The numerical work runs in a compiled, multithreaded C++ kernel (over the
profiles) with an equivalent NumPy implementation as fallback; the result
stays lazy for a dask-backed input.

References
----------
.. [1] Bean, B. R., and E. J. Dutton, 1966: Radio Meteorology. NBS Monograph
   92, National Bureau of Standards, https://doi.org/10.6028/NBS.MONO.92

.. autosummary::
   :nosignatures:
   :toctree: generated/

   modified_refractivity
   ducting_layers
"""

from __future__ import annotations

__all__ = ["modified_refractivity", "ducting_layers"]

import numpy as np
import xarray as xr

from .._provenance import provenance
from .._registry import accessor_method
from ..fundamentals.constants import EARTH_RADIUS
from . import _ducts_numpy as _ref
from .beam_path import _vapor_pressure

try:
    from . import _ducts

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _ducts = None
    HAS_COMPILED_KERNEL = False

_FIELDS = (
    ("layer_base", "m", "Height of the base of the trapping layer above sea level"),
    ("layer_top", "m", "Height of the top of the trapping layer above sea level"),
    ("layer_thickness", "m", "Thickness of the trapping layer"),
    (
        "m_deficit",
        "1",
        "Decrease of the modified refractivity across the layer (M-units)",
    ),
    (
        "trapping_angle",
        "degree",
        "Elevation below which a ray that leaves the base of the layer is trapped",
    ),
    (
        "radar_trapping_angle",
        "degree",
        "Elevation below which a ray from radar_height is trapped (radar inside the layer)",
    ),
    ("surface_based", "1", "1 if the layer begins at the lowest level of the profile"),
)


def _kernel(engine):
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled ducting kernel is not available")
    return _ducts if (HAS_COMPILED_KERNEL and engine != "numpy") else _ref


def _refractivity(profile, form, engine):
    """The refractivity (N-units) of a profile as a DataArray on ``height``."""
    if isinstance(profile, xr.DataArray):
        n = profile
    elif isinstance(profile, xr.Dataset):
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
            n = refractivity(
                p, t, _vapor_pressure(profile, p, t, engine), form=form, engine=engine
            )
    else:
        raise TypeError(
            "profile must be an xarray.Dataset with refractivity (or pressure, "
            "temperature and water vapour) or a refractivity DataArray, on a "
            "'height' dimension (m)"
        )
    if "height" not in n.dims:
        raise ValueError("the profile needs a 'height' dimension (m)")
    if "height" not in n.coords:
        raise ValueError("the profile needs a 'height' coordinate (m)")
    return n


def _heights(n):
    h = np.asarray(n["height"].values, dtype=np.float64)
    if h.size < 2:
        raise ValueError("the profile needs at least two levels")
    if not (np.all(np.isfinite(h)) and np.all(np.diff(h) > 0)):
        raise ValueError("the heights of the profile must be finite and ascending")
    return h


@provenance("Modified refractivity M = N + 1e6 h / a (Bean and Dutton 1966, Table 1.9)")
def modified_refractivity(profile, *, earth_radius=EARTH_RADIUS, form="bean_dutton"):
    """
    Modified refractivity of a profile.

    Parameters
    ----------
    profile : xarray.Dataset or xarray.DataArray
        Refractivity on a ``height`` coordinate (m above sea level): a
        DataArray in N-units, or a Dataset with ``refractivity`` or with
        ``pressure`` (Pa), ``temperature`` (K) and one of ``vapor_pressure``,
        ``specific_humidity``, ``dewpoint`` or ``relative_humidity``, as for
        :func:`radarx.retrieve.beam_path`. Any other dimensions are kept.
    earth_radius : float, optional
        Radius of the Earth (m). Default: the radarx value.
    form : str, optional
        Formula of the refractivity from the thermodynamic variables
        (:func:`radarx.io.sounding.refractivity`). Default ``"bean_dutton"``.

    Returns
    -------
    xarray.DataArray
        :math:`M = N + 10^6 h / a` (M-units), on the dimensions of the profile.
        It decreases with height in trapping layers.

    Raises
    ------
    ValueError
        For a missing variable or coordinate, or an ``earth_radius`` that is
        not positive.
    TypeError
        If ``profile`` is not an xarray object.

    Notes
    -----
    Bean and Dutton (1966, Table 1.9). The input is not changed.

    References
    ----------
    .. [1] Bean, B. R., and E. J. Dutton, 1966: Radio Meteorology. NBS
       Monograph 92, National Bureau of Standards,
       https://doi.org/10.6028/NBS.MONO.92

    Examples
    --------
    >>> import numpy as np, xarray as xr
    >>> h = np.array([0.0, 1000.0])
    >>> n = xr.DataArray([320.0, 281.0], dims="height", coords={"height": h})
    >>> modified_refractivity(n).round(1).values.tolist()
    [320.0, 437.96]
    """
    if not earth_radius > 0:
        raise ValueError("earth_radius must be positive")
    n = _refractivity(profile, form, "auto")
    m = n + n["height"].astype(np.float64) * (1.0e6 / earth_radius)
    m.name = "modified_refractivity"
    m.attrs = {
        "long_name": "Modified refractivity",
        "units": "1",
        "comment": "M = N + 1e6 h / a",
    }
    return m


@provenance(
    "Trapping layers (ducts) of a refractivity profile from the modified "
    "refractivity and Snell's law, after Bean and Dutton (1966)"
)
def ducting_layers(
    profile,
    *,
    radar_height=None,
    max_layers=4,
    min_deficit=0.0,
    earth_radius=EARTH_RADIUS,
    form="bean_dutton",
    n_threads=None,
    engine="auto",
):
    """
    Layers of a refractivity profile that trap the radar beam.

    A trapping layer is a run of levels where the modified refractivity
    decreases with height; its angle of penetration is the elevation below
    which a ray is trapped. The equations, the choices of radarx and the
    limits are in :mod:`radarx.retrieve.ducts`.

    Parameters
    ----------
    profile : xarray.Dataset or xarray.DataArray
        Refractivity on a ``height`` coordinate (m above sea level, strictly
        ascending): a DataArray in N-units, or a Dataset with
        ``refractivity`` or with ``pressure`` (Pa), ``temperature`` (K) and
        water vapour, see :func:`radarx.retrieve.modified_refractivity`. Any
        other dimensions (time, station, latitude, longitude) are batch
        dimensions: every profile is searched on its own. Levels with NaN are
        left out of their profile. Not changed. A dask-backed input gives a
        lazy result.
    radar_height : float, optional
        Height of the radar (m above sea level). Fills
        ``radar_trapping_angle`` for the layers that contain the radar.
    max_layers : int, optional
        Number of layers kept per profile, lowest first. Default 4.
    min_deficit : float, optional
        Layers whose M-deficit is not above this value (M-units) are left out.
        Default 0.
    earth_radius : float, optional
        Radius of the Earth (m). Default: the radarx value.
    form : str, optional
        Formula of the refractivity from the thermodynamic variables.
    n_threads : int, optional
        Threads of the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation. ``"auto"`` (default) prefers the compiled kernel.

    Returns
    -------
    xarray.Dataset
        On the batch dimensions of ``profile`` with a ``layer`` dimension
        (0 for the lowest layer; NaN for unused slots): ``layer_base``,
        ``layer_top``, ``layer_thickness`` (m), ``m_deficit`` (M-units),
        ``trapping_angle`` and ``radar_trapping_angle`` (degrees),
        ``surface_based`` (1 or 0), and ``layer_count``, the number of layers
        found, which can exceed ``max_layers``.

    Raises
    ------
    ValueError
        For a profile without a ``height`` coordinate or with heights that are
        not ascending, ``max_layers`` below 1, a negative ``min_deficit``, an
        ``earth_radius`` that is not positive or an unknown ``engine``.
    TypeError
        If ``profile`` is not an xarray object.
    ImportError
        If ``engine="compiled"`` and the kernel is not available.

    Notes
    -----
    Source: Bean and Dutton (1966), see the module documentation. The
    wavelength of the radar is not taken into account.

    References
    ----------
    .. [1] Bean, B. R., and E. J. Dutton, 1966: Radio Meteorology. NBS
       Monograph 92, National Bureau of Standards,
       https://doi.org/10.6028/NBS.MONO.92

    Examples
    --------
    >>> import numpy as np, xarray as xr
    >>> h = np.array([0.0, 100.0, 1000.0])
    >>> n = xr.DataArray([330.0, 280.0, 250.0], dims="height", coords={"height": h})
    >>> out = ducting_layers(n)
    >>> int(out.layer_count), float(out.layer_top.isel(layer=0))
    (1, 100.0)
    """
    kernel = _kernel(engine)
    if not earth_radius > 0:
        raise ValueError("earth_radius must be positive")
    if int(max_layers) < 1:
        raise ValueError("max_layers must be at least 1")
    if min_deficit < 0:
        raise ValueError("min_deficit must not be negative")
    n = _refractivity(profile, form, engine)
    height = _heights(n)
    radar = np.nan if radar_height is None else float(radar_height)
    max_layers = int(max_layers)
    n_threads = int(n_threads or 0)

    def block(values):
        values = np.asarray(values, dtype=np.float64)
        shape = values.shape[:-1]
        fields, count = kernel.trapping_layers(
            height,
            np.ascontiguousarray(values.reshape(-1, height.size)),
            radar,
            float(earth_radius),
            max_layers,
            float(min_deficit),
            n_threads,
        )
        fields = np.asarray(fields).copy()
        fields[..., 4:6] = np.degrees(fields[..., 4:6])
        return fields.reshape(shape + (max_layers, len(_FIELDS))), np.asarray(
            count
        ).reshape(shape)

    fields, count = xr.apply_ufunc(
        block,
        n.astype(np.float64).copy(deep=False).drop_vars("height", errors="ignore"),
        input_core_dims=[["height"]],
        output_core_dims=[["layer", "_field"], []],
        dask="parallelized",
        output_dtypes=[np.float64, np.int64],
        dask_gufunc_kwargs={
            "output_sizes": {"layer": max_layers, "_field": len(_FIELDS)},
            "allow_rechunk": True,
        },
        keep_attrs=False,
    )
    out = xr.Dataset()
    for k, (name, units, long_name) in enumerate(_FIELDS):
        out[name] = fields.isel(_field=k, drop=True)
        out[name].attrs = {"long_name": long_name, "units": units}
    out["layer_count"] = count
    out["layer_count"].attrs = {
        "long_name": "Number of trapping layers in the profile",
        "units": "1",
    }
    out = out.assign_coords(layer=np.arange(max_layers))
    out.attrs = {"earth_radius": float(earth_radius), "min_deficit": float(min_deficit)}
    return out


@accessor_method("dataset", name="ducting_layers", wraps=ducting_layers)
def _ducting_layers_accessor(self):
    """
    Layers of this refractivity profile that trap the radar beam.

    See :func:`radarx.retrieve.ducting_layers` for the parameters.

    Returns
    -------
    xarray.Dataset
        ``layer_base``, ``layer_top``, ``layer_thickness``, ``m_deficit``,
        ``trapping_angle``, ``radar_trapping_angle``, ``surface_based`` on a
        ``layer`` dimension and ``layer_count``.
    """
