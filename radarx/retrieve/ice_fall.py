#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Fall Speeds of Hail and Melting Snow
====================================

Terminal fall speeds for the particles above and in the melting layer, as the
counterpart of :func:`radarx.retrieve.terminal_fall_speed` for raindrops. The
same laws are options of the trajectory model,
``rain_trajectories(..., fall_speed=("hail", "heymsfield2020"))`` and
``fall_speed=("melting_snow", liquid_fraction, a, b)``, so that the size
sorting of hail and of melting particles can be followed with the wind.

Hail
----
:func:`hail_fall_speed` gives the terminal velocity of hail from its maximum
dimension :math:`D` (cm) with the power laws of the corrigendum of Heymsfield
et al. (2020) to Heymsfield et al. (2018):

* ``"heymsfield2020"`` (default): :math:`V_t = 7.6 D^{0.89}` m s\\ :sup:`-1`
  for :math:`D < D_x` and :math:`V_t = 8.4 D^{0.67}` for :math:`D > D_x`
  (Eqs. 2a and 2b, Table 2), a fit to hail collected at the ground and to
  wind-tunnel measurements of 3D-printed replicas of hailstones, for
  diameters above 0.5 cm. The two branches cross at 1.576 cm, where the law
  is switched; the text of the corrigendum prints 1.5 cm and its Table 2
  1.59 cm.
* ``"graupel_hail_obs"``: :math:`7.2 D^{0.84}`, the fit to the observations of
  graupel and hail (Table 2).
* ``"heymsfield2020_p10"``, ``"_p25"``, ``"_p50"``, ``"_p75"``, ``"_p90"``:
  the power laws of the 10th to 90th percentile of the velocities at a given
  size (Table 3), to represent their natural variability.
* ``"milbrandt_yau2005"``: :math:`V_t = \\gamma\\, 206.89\\, D_m^{0.6384}` with the
  diameter :math:`D_m` in m, the hail law of the multimoment bulk scheme of
  Milbrandt and Yau (2005, Table 2 of Part II, after Ferrier 1994) and the one
  law here with a density dependence, :math:`\\gamma = (\\rho_0/\\rho)^{1/2}`
  (Part I, Eq. A1), applied when ``air_density`` is given (with
  :math:`\\rho_0 = 1.204` kg m\\ :sup:`-3`, the density of the other fall speeds of
  radarx) and in the trajectory model with its profile. It gives 10.9 m
  s\\ :sup:`-1` for 1 cm and 26.5 m s\\ :sup:`-1` for 4 cm at sea level
  (Heymsfield et al.: 7.6 and 21.3 for the same sizes). The paper states no
  size range and does not say where its coefficient comes from (it cites Ferrier
  for them), and the fit has not been compared with measurements here.

The laws are fits to size, not physical models of the drag: they hold for
the maximum dimension, not for the diameter of the sphere of the same volume,
and the corrigendum gives no air-density dependence, so none is applied to
its laws (a hailstone falls faster aloft than these laws say; use
``"milbrandt_yau2005"`` for a law with one). The validity range of the
percentile laws is not stated there; 0.5 cm is used for all. Diameters below
it give a warning, because the laws fall off from the measured speeds there.
Earlier versions: the single law of Heymsfield et al. (2018) was in error and
is replaced by the corrigendum (the corrected laws are used here); Knight and
Heymsfield (1983) and Heymsfield and Wright (2014), the other forms named in
the literature, are not implemented.

Melting snow
------------
:func:`melting_snow_fall_speed` mixes the fall speed of the dry snowflake
:math:`V_s` and of the raindrop :math:`V_r` of the same mass,

.. math::

    V_m = V_s + (V_r - V_s)\\,\\Psi(\\ell),\\qquad
    \\Psi(\\ell) = \\alpha_\\Psi\\,\\ell + (1 - \\alpha_\\Psi)\\,\\ell^7,
    \\quad \\alpha_\\Psi = 0.246,

with the liquid water fraction :math:`\\ell` between 0 and 1. This is Eqs. (14)
and (15) of Frick et al. (2013), who took the form from the wind-tunnel
experiments of Mitra et al. (1990, their Fig. 2): the speed rises slowly
and linearly with :math:`\\ell` for small :math:`\\ell` and fast for :math:`\\ell
> 0.7`; the coefficient :math:`\\alpha_\\Psi` fits their Fig. 2. The equations
are those of Frick et al.; Mitra et al. has not been compared. The speed
:math:`V_s` is not given by a relation: Frick et al. take it from
Khvorostyanov and Curry (2005), which has not been compared. Pass a
power law :math:`V_s = a D^b` of the diameter :math:`D` (mm) of the raindrop
of the same mass (``snow=(a, b)``, m s\\ :sup:`-1`), from the mass-diameter and
fall-speed relations of the snow of your case; there is no default. The speed
of the raindrop is the radarx law (``rain``: ``"drag_balance"`` or
``"atlas1973"``). No density correction is applied to either
law. The liquid fraction is a number or an array, constant along a trajectory.

References
----------
.. [1] Heymsfield, A., M. Szakall, A. Jost, I. Giammanco, R. Wright, and J.
   Brimelow, 2020: Corrigendum (to: A comprehensive observational study of
   graupel and hail terminal velocity, mass flux, and kinetic energy). *J.
   Atmos. Sci.*, **77** (1), 405-412,
   https://doi.org/10.1175/JAS-D-19-0185.1
.. [2] Frick, C., A. Seifert, and H. Wernli, 2013: A bulk parametrization of
   melting snowflakes with explicit liquid water fraction for the COSMO
   model. *Geosci. Model Dev.*, **6** (6), 1925-1939,
   https://doi.org/10.5194/gmd-6-1925-2013
.. [3] Milbrandt, J. A., and M. K. Yau, 2005: A multimoment bulk
   microphysics parameterization. Part I: Analysis of the role of the spectral
   shape parameter. *J. Atmos. Sci.*, **62** (9), 3051-3064,
   https://doi.org/10.1175/JAS3534.1; Part II: A proposed three-moment closure
   and scheme description. *J. Atmos. Sci.*, **62** (9), 3065-3081,
   https://doi.org/10.1175/JAS3535.1

.. autosummary::
   :nosignatures:
   :toctree: generated/

   hail_fall_speed
   melting_snow_fall_speed
"""

from __future__ import annotations

__all__ = ["hail_fall_speed", "melting_snow_fall_speed"]

import warnings

import numpy as np
import xarray as xr

from .._provenance import provenance
from .._registry import accessor_method
from . import _ice_fall, _raindrop_fall


def _diameters(diameter):
    d = diameter if isinstance(diameter, xr.DataArray) else xr.DataArray(diameter)
    values = np.asarray(d.values, dtype=np.float64)
    if np.any(values[np.isfinite(values)] <= 0.0):
        raise ValueError("diameter must be positive (mm)")
    return d, values


@provenance(
    "Terminal fall speed of hail from the size relations of Heymsfield et al. (2020)"
)
def hail_fall_speed(diameter, law="heymsfield2020", *, air_density=None):
    """
    Terminal fall speed of hail from its maximum dimension.

    Parameters
    ----------
    diameter : xarray.DataArray or array-like
        Maximum dimension of the hailstone in mm.
    law : str, optional
        ``"heymsfield2020"`` (default), ``"graupel_hail_obs"`` or
        ``"heymsfield2020_p10"``, ``"_p25"``, ``"_p50"``, ``"_p75"``,
        ``"_p90"`` (percentiles) or ``"milbrandt_yau2005"``, see
        :mod:`radarx.retrieve.ice_fall`.
    air_density : float or xarray.DataArray, optional
        Air density (kg m-3) for the laws that depend on it (only
        ``"milbrandt_yau2005"``). Without it the speed is that at the sea-level
        density of radarx, 1.204 kg m-3.

    Returns
    -------
    xarray.DataArray
        Terminal fall speed in m s-1, broadcast like ``diameter``. It is a
        power law of the diameter and increases with it; the default law is
        continuous at the crossing of its two branches.

    Warns
    -----
    UserWarning
        If a diameter is below 5 mm, where the laws are not fitted.

    Raises
    ------
    ValueError
        For a diameter that is not positive, an unknown ``law``, or an
        ``air_density`` that is not positive or is given for a law without a
        density dependence.

    Notes
    -----
    See the module documentation for the validity range, the choice of the
    break of the default law at 1.576 cm and the air-density dependence. The
    inputs are not changed.

    References
    ----------
    .. [1] Heymsfield, A., M. Szakall, A. Jost, I. Giammanco, R. Wright, and
       J. Brimelow, 2020: Corrigendum (to: A comprehensive observational
       study of graupel and hail terminal velocity, mass flux, and kinetic
       energy). *J. Atmos. Sci.*, **77** (1), 405-412,
       https://doi.org/10.1175/JAS-D-19-0185.1
    .. [2] Milbrandt, J. A., and M. K. Yau, 2005: A multimoment bulk
       microphysics parameterization. Part II: A proposed three-moment closure
       and scheme description. *J. Atmos. Sci.*, **62** (9), 3065-3081,
       https://doi.org/10.1175/JAS3535.1

    Examples
    --------
    >>> round(float(hail_fall_speed(20.0)), 1)
    13.4
    >>> round(float(hail_fall_speed(20.0, "milbrandt_yau2005", air_density=0.6)), 1)
    24.1
    """
    d, values = _diameters(diameter)
    speed = _ice_fall.hail_speed(values, law)[0]
    exponent = _ice_fall.density_exponent(law)
    if air_density is not None:
        if exponent == 0.0:
            raise ValueError(
                f"the hail law {law!r} has no air density dependence; "
                "air_density is for 'milbrandt_yau2005'"
            )
        rho = (
            air_density
            if isinstance(air_density, xr.DataArray)
            else xr.DataArray(np.asarray(air_density, dtype=np.float64))
        )
        rho_values = np.asarray(rho.values, dtype=np.float64)
        if np.any(rho_values[np.isfinite(rho_values)] <= 0.0):
            raise ValueError("air_density must be positive (kg m-3)")
        factor = (_raindrop_fall.RHO0 / rho.astype(np.float64)) ** exponent
    below = np.isfinite(values) & (values < _ice_fall.HAIL_MIN_DIAMETER)
    if below.any():
        warnings.warn(
            f"{int(below.sum())} diameter(s) are below {_ice_fall.HAIL_MIN_DIAMETER:g} mm, "
            "where the hail laws are not fitted",
            UserWarning,
            stacklevel=2,
        )
    out = d.copy(data=speed)
    if air_density is not None:
        out = out * factor
    out = out.rename("terminal_fall_speed")
    out.attrs = {
        "long_name": "Terminal fall speed of hail",
        "units": "m s-1",
        "law": str(law),
    }
    return out


@provenance(
    "Terminal fall speed of melting snow: dry snow and rain speeds mixed with "
    "the liquid fraction, after Mitra et al. (1990) as in Frick et al. (2013)"
)
def melting_snow_fall_speed(diameter, liquid_fraction, *, snow, rain="drag_balance"):
    """
    Terminal fall speed of melting snow from its liquid water fraction.

    Parameters
    ----------
    diameter : xarray.DataArray or array-like
        Diameter (mm) of the raindrop that has the mass of the particle.
    liquid_fraction : float or xarray.DataArray
        Liquid water fraction, from 0 (dry snow) to 1 (raindrop).
    snow : tuple of float
        ``(a, b)`` of the speed of the dry snowflake of the same mass,
        :math:`V_s = a D^b` (m s-1, :math:`D` in mm). There is no default.
    rain : {"drag_balance", "atlas1973"}, optional
        Law of the raindrop of the same mass
        (:func:`radarx.retrieve.terminal_fall_speed`).

    Returns
    -------
    xarray.DataArray
        Terminal fall speed in m s-1, broadcast over ``diameter`` and
        ``liquid_fraction``. It equals the dry snow law at 0, the raindrop at
        1 and is between them otherwise.

    Raises
    ------
    ValueError
        For a diameter that is not positive, a liquid fraction outside
        [0, 1], ``a`` not positive, ``b`` not above -1 or an unknown ``rain``.

    Notes
    -----
    The mixing law is that of Frick et al. (2013); the dry snow speed has to
    be given (see the module documentation). No density correction is
    applied. The inputs are not changed.

    References
    ----------
    .. [1] Frick, C., A. Seifert, and H. Wernli, 2013: A bulk parametrization
       of melting snowflakes with explicit liquid water fraction for the COSMO
       model. *Geosci. Model Dev.*, **6** (6), 1925-1939,
       https://doi.org/10.5194/gmd-6-1925-2013

    Examples
    --------
    >>> v = melting_snow_fall_speed([2.0], 0.5, snow=(1.0, 0.3))
    >>> bool(v.item() > 1.0 * 2.0**0.3)
    True
    """
    d, values = _diameters(diameter)
    a, b = (float(v) for v in snow)
    if not (a > 0.0 and b > -1.0):
        raise ValueError("snow needs a > 0 and b > -1")
    ell = (
        liquid_fraction
        if isinstance(liquid_fraction, xr.DataArray)
        else xr.DataArray(liquid_fraction)
    )
    ell_values = np.asarray(ell.values, dtype=np.float64)
    if np.any(
        (ell_values[np.isfinite(ell_values)] < 0.0)
        | (ell_values[np.isfinite(ell_values)] > 1.0)
    ):
        raise ValueError("liquid_fraction must be between 0 and 1")
    if rain not in _raindrop_fall.LAWS:
        raise ValueError(f"rain must be one of {_raindrop_fall.LAWS}, not {rain!r}")
    d, ell = xr.broadcast(d, ell)
    dv = np.asarray(d.values, dtype=np.float64)
    weight = _ice_fall.psi(np.asarray(ell.values, dtype=np.float64))
    vs = a * np.power(dv, b)
    vr = _raindrop_fall.sea_level_speed(dv, rain)[0]
    out = d.copy(data=(1.0 - weight) * vs + weight * vr).rename("terminal_fall_speed")
    out.attrs = {
        "long_name": "Terminal fall speed of melting snow",
        "units": "m s-1",
    }
    return out


@accessor_method("dataarray", name="hail_fall_speed", wraps=hail_fall_speed)
def _hail_fall_speed_accessor(self):
    """
    Terminal fall speed of hail from the maximum dimension (mm) in the array.

    See :func:`radarx.retrieve.hail_fall_speed` for the parameters.

    Returns
    -------
    xarray.DataArray
        Terminal fall speed in m s-1.
    """


@accessor_method(
    "dataarray", name="melting_snow_fall_speed", wraps=melting_snow_fall_speed
)
def _melting_snow_fall_speed_accessor(self):
    """
    Terminal fall speed of melting snow; the array holds the diameters (mm).

    See :func:`radarx.retrieve.melting_snow_fall_speed` for the parameters.

    Returns
    -------
    xarray.DataArray
        Terminal fall speed in m s-1.
    """
