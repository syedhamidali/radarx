#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Soundings and ERA5 Profiles
===========================

Vertical profiles of temperature, humidity, pressure and wind from radiosonde
archives and from the ERA5 reanalysis, all returned in the same xarray form,
plus helpers that put them on radar gates, QVP heights or radarx grids.

Profile format
--------------

Every reader returns an :py:class:`xarray.Dataset` on a ``height`` dimension
(geometric height above mean sea level, m, ascending, without duplicates)
with the variables

=======================  ==================================  ============
variable                 CF standard name                    units
=======================  ==================================  ============
``pressure``             ``air_pressure``                    Pa
``geopotential_height``  ``geopotential_height``             m
``temperature``          ``air_temperature``                 K
``dewpoint``             ``dew_point_temperature``           K
``relative_humidity``    ``relative_humidity``               1
``specific_humidity``    ``specific_humidity``               kg kg-1
``u``, ``v``             ``eastward_wind``, ``northward_wind``  m s-1
``wind_speed``           ``wind_speed``                      m s-1
``wind_direction``       ``wind_from_direction``             degree
``w`` (ERA5 only)        ``upward_air_velocity``             m s-1
=======================  ==================================  ============

and the scalar coordinates ``time`` (launch or valid time), ``latitude`` and
``longitude``. Missing values are NaN. ``attrs`` hold ``source``,
``station``, ``station_name`` and ``launch_time``.

Radiosonde and ERA5 heights are geopotential heights :math:`H`. They are
converted to geometric heights, as used by radar beam heights, with
:math:`z = R_e H / (R_e - H)` (spherical Earth of radius
:math:`R_e = 6371.0088` km, gravity falling off with the inverse square of
the distance from the centre and equal to :math:`g_0 = 9.80665` m s-2 at
the surface). The difference is about 16 m at 10 km and 63 m at 20 km.
The formula is an elementary integral of :math:`g(z) = g_0 [R_e/(R_e+z)]^2`
(:math:`H = \\Phi/g_0 = R_e z/(R_e + z)`), a standard definition and not taken
from a particular paper. :math:`g_0 = 9.80665` m s-2 is the conventional
standard gravity (defined value, 3rd CGPM 1901). :math:`R_e` is the mean
radius :math:`(2a + b)/3` of the GRS80 ellipsoid (:math:`a = 6378137` m,
:math:`b = 6356752.3141` m, Moritz 2000 [7]; the mean is 6371008.77 m,
rounded here to 6371008.8 m). The ellipsoid parameters and the CGPM value
are quoted from the standards and not checked against the source documents.

Sources
-------

Observed soundings (:func:`read_sounding`):

- ``"iem"``: Iowa Environmental Mesonet RAOB archive, JSON service
  ``https://mesonet.agron.iastate.edu/json/raob.py`` (mandatory and
  significant levels; fast and reliable).
- ``"uwyo"``: University of Wyoming upper-air archive,
  ``https://weather.uwyo.edu/wsgi/sounding`` (CSV; high-resolution BUFR
  profiles when available, so slower).
- ``"igra2"``: NOAA NCEI Integrated Global Radiosonde Archive version 2
  (Durre et al. 2006 [3]; Durre et al. 2018 [8]; dataset doi [9]). The whole station file is downloaded once
  (about 100 MB zipped for a long-running station) and cached.

ERA5 (:func:`era5_profile`, :func:`era5_column`):

- ``"arco"``: the analysis-ready, cloud-optimised (ARCO) ERA5 time series on
  pressure levels served by ECMWF through the Copernicus Climate Data Store
  (dataset ``reanalysis-era5-pressure-levels-timeseries``, doi
  10.24381/af48f136 [6]). Needs a CDS account (``~/.cdsapirc``) and ``cdsapi``.
  It has 13 pressure levels (1000-50 hPa), 6-hourly times (00, 06, 12,
  18 UTC) and serves the nearest grid point or a small area.
- ``"cds"``: the full ERA5 hourly data on 37 pressure levels from the CDS
  (``reanalysis-era5-pressure-levels``, doi 10.24381/cds.bd0915c6 [5];
  ERA5 itself: Hersbach et al. 2020 [4]). Needs a
  CDS account; requests are queued (typically about a minute).
- ``"gcs"``: Google's ARCO-ERA5 Zarr store
  ``gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3``
  (37 levels, hourly), read anonymously over its public HTTPS endpoint
  ``https://storage.googleapis.com/gcp-public-data-arco-era5/...`` (fsspec
  HTTP, no Google credentials or gcsfs). Opened lazily with xarray; each field is
  stored as one global chunk per hour (about 100 MB), so a point profile
  reads a few hundred MB per hour.
- ``"auto"`` (default): hourly ERA5 on 37 levels, from the CDS (``"cds"``)
  when CDS credentials are configured, otherwise from Google's ARCO-ERA5
  (``"gcs"``). The 6-hourly ECMWF ARCO time series is used only when asked
  for with ``source="arco"``.

Downloads are cached under ``pooch.os_cache("radarx")/soundings`` (or the
``RADARX_CACHE_DIR`` environment variable).

Computation
-----------

Vertical interpolation, horizontal (bilinear) and time interpolation of ERA5
columns, level crossings (isotherms, wet-bulb zero), layer means, wind
rotation and the thermodynamic functions run in a compiled C++ kernel
(``radarx.io._sounding``, multithreaded); an identical NumPy implementation
is used when the kernel is not built (``engine="numpy"``).

Thermodynamics and constants (provenance of every number)
---------------------------------------------------------

All thermodynamic functions are elementwise and run in the kernels
(``_sounding.cpp`` and its NumPy twin ``_sounding_numpy.py``, which carry the
same constants). Formulas and constants, with what is and is not traceable:

- Saturation vapour pressure over liquid water,
  :math:`e_s(T) = 611.2 \\exp[17.67\\,t / (t + 243.5)]` Pa with
  :math:`t = T - 273.15` in degC, and its inverse (the dew point,
  :math:`t_d = 243.5\\,\\ln(e/611.2) / [17.67 - \\ln(e/611.2)]`). This is
  Bolton (1980) [1], equation (10), whose original form is
  :math:`e_s = 6.112 \\exp[17.67 t/(t + 243.5)]` hPa (coefficients 6.112 hPa,
  17.67 and 243.5 degC); the equation number and the 611.2 Pa conversion of
  the prefactor follow the common citation of that equation (the coefficients
  agree with a published formula collection of NCAR/EOL; the equation number
  is not checked against the paper).
  The validity range of the fit is not restated here. It is applied at all
  temperatures, i.e. also below 0 degC as an over-liquid (not over-ice) value;
  this is a radarx choice.
- Latent heat of vaporization, :math:`L_v(T) = (2.501 - 0.00237\\,t)
  \\times 10^6` J kg-1 (t in degC). Attributed to Bolton (1980) [1]; the
  coefficients and the equation number are not checked against the paper.
- Mixing ratio :math:`r = \\epsilon e / (p - e)`, specific humidity
  :math:`q = \\epsilon e / [p - (1 - \\epsilon) e]`, its inverse
  :math:`e = q p / [\\epsilon + (1 - \\epsilon) q]`, relative humidity
  :math:`e_s(T_d)/e_s(T)` and the virtual temperature
  :math:`T_v = T [1 + (1/\\epsilon - 1) q]` with air density
  :math:`\\rho = p/(R_d T_v)`: standard textbook definitions (no
  paper-specific coefficients).
- :math:`R_d = 287.04749`, :math:`R_v = 461.52311` J kg-1 K-1 (so
  :math:`\\epsilon = R_d/R_v = 0.62198`): radarx's gas-constant values. They
  match :math:`R^*/M` for dry air and water to about 1e-5 relative (with
  :math:`R^* = 8.3144626` J mol-1 K-1 and molar masses of about
  28.9655 and 18.0153 g mol-1), but the source of these exact digits is not
  documented.
- :math:`c_{{pd}} = 1005.7` and :math:`c_{{pv}} = 1875.0` J kg-1 K-1 in the
  wet-bulb equation: radarx's values; neither number could be traced to an
  equation or table of [1] or [2].
- Wet-bulb temperature: the isobaric wet-bulb temperature, the root of
  :math:`(c_{{pd}} + r c_{{pv}})(T - T_w) = L_v(T_w)\\,(r_s(T_w) - r)` (an
  enthalpy balance of an isobaric evaporation to saturation, a textbook
  definition), found by a bracketed Newton iteration to 1e-7 K (radarx
  choice). It is *not* the pseudo-adiabatic (Normand) wet-bulb temperature
  that Davies-Jones (2008) [2] computes by inverting Bolton's
  equivalent-potential-temperature formula; the earlier statement that the
  two differ by "a few tenths of a kelvin at most" is a radarx estimate that
  [2] does not state.
- 0 degC, isotherm and wet-bulb-zero levels, layer means: linear interpolation
  in height between levels, the upper-most crossing by default, trapezoidal
  rule for layer means (radarx choices, not from a paper). Pressure is
  interpolated linearly in its logarithm, as are missing geopotential heights
  (:func:`interpolate_profile`).
- Wind: the meteorological convention, direction the wind blows *from*
  clockwise from north, :math:`u = -s \\sin\\phi`, :math:`v = -s \\cos\\phi`
  (meteorological convention). One knot is 0.514444 m s-1 (the exact value is
  1852/3600 = 0.514444...).

Data sources and formats read here:

- IGRA version 2 [3], [8], fixed-column text (the format document of the
  archive; header record starting with ``#``, pressure in Pa, geopotential
  height in m, temperature in tenths of degC, dew-point depression in tenths
  of degC, wind direction in degree, wind speed in tenths of m s-1, missing
  values -9999 and -8888). The column positions were taken from the archive's
  format description (not checked against the document).
  Durre et al. (2018) [8] describe IGRA2; Durre et al. (2006) [3] describe
  version 1, which IGRA2 supersedes.
- IEM RAOB JSON and University of Wyoming CSV: web-service formats without a
  citable paper; the archives are named in the sources above.
- ERA5 [4] on pressure levels, from the CDS datasets [5] and [6] or Google's
  ARCO-ERA5 copy of the ERA5 data (no peer-reviewed description is cited).

Interface for multi-Doppler winds
---------------------------------

:func:`era5_column` and :func:`profile_to_grid` return the same background
on a radarx grid (the ``z``, ``y``, ``x`` coordinates and ``crs_wkt`` of
:func:`radarx.grid.grid_radar` output): ``u``, ``v`` (wind components along
the grid ``x`` and ``y`` axes), ``w``, ``temperature``, ``pressure``,
``specific_humidity``, ``dewpoint``, ``relative_humidity``, ``air_density``
on ``(z, y, x)``, and ``freezing_level``, ``wet_bulb_zero_height`` and
``wind_rotation`` on ``(y, x)``.

References
----------
.. [1] Bolton, D., 1980: The computation of equivalent potential temperature.
   Mon. Wea. Rev., 108, 1046-1053,
   https://doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2
.. [2] Davies-Jones, R., 2008: An efficient and accurate method for computing
   the wet-bulb temperature along pseudoadiabats. Mon. Wea. Rev., 136,
   2764-2785, https://doi.org/10.1175/2007MWR2224.1
.. [3] Durre, I., R. S. Vose, and D. B. Wuertz, 2006: Overview of the
   Integrated Global Radiosonde Archive. J. Climate, 19, 53-68,
   https://doi.org/10.1175/JCLI3594.1
.. [4] Hersbach, H., and Coauthors, 2020: The ERA5 global reanalysis.
   Quart. J. Roy. Meteor. Soc., 146, 1999-2049,
   https://doi.org/10.1002/qj.3803
.. [5] Copernicus Climate Change Service, 2018: ERA5 hourly data on pressure
   levels from 1940 to present. Copernicus Climate Change Service (C3S)
   Climate Data Store (CDS), https://doi.org/10.24381/cds.bd0915c6
.. [6] Copernicus Climate Change Service, 2026: ERA5 time-series data on
   pressure levels from 1940 to present. ECMWF,
   https://doi.org/10.24381/af48f136
.. [7] Moritz, H., 2000: Geodetic Reference System 1980. J. Geodesy, 74,
   128-133, https://doi.org/10.1007/s001900050278
.. [8] Durre, I., X. Yin, R. S. Vose, S. Applequist, and J. Arnfield, 2018:
   Enhancing the data coverage in the Integrated Global Radiosonde Archive.
   J. Atmos. Oceanic Technol., 35, 1753-1770,
   https://doi.org/10.1175/JTECH-D-17-0223.1
.. [9] Durre, I., X. Yin, R. S. Vose, S. Applequist, and J. Arnfield, 2016:
   Integrated Global Radiosonde Archive (IGRA), Version 2. NOAA National
   Centers for Environmental Information, https://doi.org/10.7289/V5X63K0Q

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = [
    "read_sounding",
    "open_sounding_file",
    "nearest_station",
    "station_list",
    "era5_profile",
    "era5_column",
    "profile_to_grid",
    "interpolate_profile",
    "isotherm_height",
    "wet_bulb_zero_height",
    "mean_wind",
    "saturation_vapor_pressure",
    "dewpoint_from_vapor_pressure",
    "dewpoint_from_specific_humidity",
    "specific_humidity_from_dewpoint",
    "relative_humidity_from_dewpoint",
    "wet_bulb_temperature",
    "air_density",
    "geopotential_to_height",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import csv
import datetime as _dt
import functools
import hashlib
import io
import json
import os
import re
from pathlib import Path

import numpy as np
import xarray as xr

from . import _sounding_numpy

try:
    from . import _sounding

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _sounding = None
    HAS_COMPILED_KERNEL = False

G0 = 9.80665
T0 = 273.15

IEM_URL = "https://mesonet.agron.iastate.edu/json/raob.py?ts={ts}&station={station}"
UWYO_URL = (
    "https://weather.uwyo.edu/wsgi/sounding?datetime={date}%20{hour:02d}:00:00"
    "&id={station}&src=UNKNOWN&type=TEXT:CSV"
)
IGRA2_URL = "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access"
# Google ARCO-ERA5 (gs://gcp-public-data-arco-era5/...) through its public HTTPS endpoint
ARCO_ERA5_ZARR = (
    "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/"
    "full_37-1h-0p25deg-chunk-1.zarr-v3"
)
CDS_PRESSURE_LEVELS = "reanalysis-era5-pressure-levels"
CDS_ARCO_TIMESERIES = "reanalysis-era5-pressure-levels-timeseries"
GCS_PAD = 8.0  # degrees cached around a Google ARCO-ERA5 request (radarx choice)
# the 37 standard ERA5 pressure levels [hPa] of the CDS pressure-level dataset (Copernicus
# Climate Change Service 2018, doi 10.24381/cds.bd0915c6)
ERA5_LEVELS = (
    [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300]
    + [350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875]
    + [900, 925, 950, 975, 1000]
)

KNOT = 0.514444  # m s-1 per knot (1852 m / 3600 s = 0.5144444...)

ATTRS = {
    "height": {
        "standard_name": "altitude",
        "long_name": "geometric height above mean sea level",
        "units": "m",
        "positive": "up",
    },
    "pressure": {
        "standard_name": "air_pressure",
        "long_name": "pressure",
        "units": "Pa",
    },
    "geopotential_height": {
        "standard_name": "geopotential_height",
        "long_name": "geopotential height",
        "units": "m",
    },
    "temperature": {
        "standard_name": "air_temperature",
        "long_name": "temperature",
        "units": "K",
    },
    "dewpoint": {
        "standard_name": "dew_point_temperature",
        "long_name": "dew point temperature",
        "units": "K",
    },
    "relative_humidity": {
        "standard_name": "relative_humidity",
        "long_name": "relative humidity with respect to liquid water",
        "units": "1",
    },
    "specific_humidity": {
        "standard_name": "specific_humidity",
        "long_name": "specific humidity",
        "units": "kg kg-1",
    },
    "u": {
        "standard_name": "eastward_wind",
        "long_name": "eastward wind component",
        "units": "m s-1",
    },
    "v": {
        "standard_name": "northward_wind",
        "long_name": "northward wind component",
        "units": "m s-1",
    },
    "w": {
        "standard_name": "upward_air_velocity",
        "long_name": "vertical wind component",
        "units": "m s-1",
    },
    "wind_speed": {
        "standard_name": "wind_speed",
        "long_name": "wind speed",
        "units": "m s-1",
    },
    "wind_direction": {
        "standard_name": "wind_from_direction",
        "long_name": "direction the wind blows from",
        "units": "degree",
    },
    "wet_bulb_temperature": {
        "standard_name": "wet_bulb_temperature",
        "long_name": "isobaric wet-bulb temperature",
        "units": "K",
    },
    "air_density": {
        "standard_name": "air_density",
        "long_name": "density of moist air",
        "units": "kg m-3",
    },
    "vapor_pressure": {
        "standard_name": "water_vapor_partial_pressure_in_air",
        "long_name": "water vapour pressure",
        "units": "Pa",
    },
    "saturation_vapor_pressure": {
        "long_name": "saturation vapour pressure over liquid water",
        "units": "Pa",
    },
    "freezing_level": {
        "long_name": "height of the 0 degC isotherm (top crossing)",
        "units": "m",
    },
    "wet_bulb_zero_height": {
        "long_name": "height of the 0 degC wet-bulb isotherm (top crossing)",
        "units": "m",
    },
}

GRID_WIND_ATTRS = {
    "u": {
        "standard_name": "x_wind",
        "long_name": "wind component along the grid x axis",
        "units": "m s-1",
    },
    "v": {
        "standard_name": "y_wind",
        "long_name": "wind component along the grid y axis",
        "units": "m s-1",
    },
    "wind_rotation": {
        "long_name": "direction of true north, clockwise from the grid y axis",
        "units": "degree",
    },
}

PROFILE_VARS = (
    "pressure",
    "geopotential_height",
    "temperature",
    "dewpoint",
    "relative_humidity",
    "specific_humidity",
    "u",
    "v",
    "w",
    "wind_speed",
    "wind_direction",
)


# ---------------------------------------------------------------------------
# kernel dispatch


def _kernel(engine):
    """The compiled kernel module or its NumPy twin for ``engine``."""
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled sounding kernel is not available")
    if HAS_COMPILED_KERNEL and engine != "numpy":
        return _sounding
    return _sounding_numpy


def _f64(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def _thermo(op, *arrays, engine="auto", n_threads=None):
    """Run an elementwise kernel on broadcast NumPy arrays, keeping the shape."""
    arrays = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in arrays))
    shape = arrays[0].shape
    k = _kernel(engine)
    out = k.thermo(op, [_f64(a).ravel() for a in arrays], n_threads=int(n_threads or 0))
    return np.asarray(out).reshape(shape)


def _as_dataarray(*args):
    """xarray-aligned DataArrays of the inputs (scalars and arrays allowed)."""
    das = [
        a if isinstance(a, xr.DataArray) else xr.DataArray(np.asarray(a)) for a in args
    ]
    return xr.broadcast(*das)


def _wrap(op, name, *args, engine="auto", n_threads=None):
    das = _as_dataarray(*args)
    out = _thermo(op, *(d.values for d in das), engine=engine, n_threads=n_threads)
    return das[0].copy(data=out).rename(name).assign_attrs(ATTRS.get(name, {}))


# ---------------------------------------------------------------------------
# thermodynamics (xarray in, xarray out)


def saturation_vapor_pressure(temperature, *, engine="auto", n_threads=None):
    """
    Saturation vapour pressure over liquid water.

    :math:`e_s = 611.2 \\exp[17.67 (T - 273.15) / (T - 29.65)]` Pa, Bolton
    (1980), eq. (10); written in kelvin, :math:`T - 29.65 = t + 243.5`
    with :math:`t = T - 273.15` in degC. Coefficients: 6.112 hPa
    (converted to 611.2 Pa), 17.67 and 243.5 degC, as in the original
    equation (equation number not re-checked against the paper). Used at all
    temperatures over liquid water (radarx choice; there is no over-ice
    branch).

    Parameters
    ----------
    temperature : xarray.DataArray or array-like
        Temperature in K.
    engine : {"auto", "compiled", "numpy"}, optional
        Kernel implementation. Default prefers the compiled kernel.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.

    Returns
    -------
    xarray.DataArray
        Saturation vapour pressure in Pa.

    References
    ----------
    Bolton, D., 1980: The computation of equivalent potential temperature.
    Mon. Wea. Rev., 108, 1046-1053,
    https://doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2
    """
    return _wrap(
        "esat",
        "saturation_vapor_pressure",
        temperature,
        engine=engine,
        n_threads=n_threads,
    )


def dewpoint_from_vapor_pressure(vapor_pressure, *, engine="auto", n_threads=None):
    """
    Dew point from water vapour pressure (Bolton 1980 eq. (10) inverted).

    :math:`t_d = 243.5 \\ln(e/611.2) / [17.67 - \\ln(e/611.2)]` degC, the
    algebraic inverse of :func:`saturation_vapor_pressure` (Bolton 1980;
    coefficients and equation number as there).

    Parameters
    ----------
    vapor_pressure : xarray.DataArray or array-like
        Vapour pressure in Pa.
    engine, n_threads : optional
        See :func:`saturation_vapor_pressure`.

    Returns
    -------
    xarray.DataArray
        Dew point in K (NaN for non-positive vapour pressure).

    References
    ----------
    Bolton, D., 1980: The computation of equivalent potential temperature.
    Mon. Wea. Rev., 108, 1046-1053,
    https://doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2
    """
    return _wrap(
        "dewpoint", "dewpoint", vapor_pressure, engine=engine, n_threads=n_threads
    )


def dewpoint_from_specific_humidity(
    specific_humidity, pressure, *, engine="auto", n_threads=None
):
    """
    Dew point from specific humidity and pressure.

    The vapour pressure is :math:`e = q p / (\\epsilon + (1 - \\epsilon) q)`
    with :math:`\\epsilon = R_d / R_v = 0.62198` (standard definition of
    specific humidity; :math:`R_d = 287.04749`, :math:`R_v = 461.52311`
    J kg-1 K-1 are radarx's values, see :mod:`radarx.io.sounding`); the dew
    point follows from Bolton (1980) eq. (10) inverted.

    Parameters
    ----------
    specific_humidity : xarray.DataArray or array-like
        Specific humidity in kg kg-1.
    pressure : xarray.DataArray or array-like
        Pressure in Pa.
    engine, n_threads : optional
        See :func:`saturation_vapor_pressure`.

    Returns
    -------
    xarray.DataArray
        Dew point in K.

    References
    ----------
    Bolton, D., 1980: The computation of equivalent potential temperature.
    Mon. Wea. Rev., 108, 1046-1053,
    https://doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2
    """
    e = _wrap(
        "vapor_pressure",
        "vapor_pressure",
        specific_humidity,
        pressure,
        engine=engine,
        n_threads=n_threads,
    )
    return dewpoint_from_vapor_pressure(e, engine=engine, n_threads=n_threads)


def specific_humidity_from_dewpoint(
    dewpoint, pressure, *, engine="auto", n_threads=None
):
    """
    Specific humidity from dew point and pressure.

    :math:`q = \\epsilon e / (p - (1 - \\epsilon) e)` (standard definition)
    with :math:`e = e_s(T_d)` from Bolton (1980) eq. (10).

    Parameters
    ----------
    dewpoint : xarray.DataArray or array-like
        Dew point in K.
    pressure : xarray.DataArray or array-like
        Pressure in Pa.
    engine, n_threads : optional
        See :func:`saturation_vapor_pressure`.

    Returns
    -------
    xarray.DataArray
        Specific humidity in kg kg-1.

    References
    ----------
    Bolton, D., 1980: The computation of equivalent potential temperature.
    Mon. Wea. Rev., 108, 1046-1053,
    https://doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2
    """
    e = saturation_vapor_pressure(dewpoint, engine=engine, n_threads=n_threads)
    return _wrap(
        "specific_humidity",
        "specific_humidity",
        e,
        pressure,
        engine=engine,
        n_threads=n_threads,
    )


def relative_humidity_from_dewpoint(
    temperature, dewpoint, *, engine="auto", n_threads=None
):
    """
    Relative humidity with respect to liquid water, :math:`e_s(T_d) / e_s(T)`.

    The ratio of vapour pressure (the saturation value at the dew point) to
    saturation vapour pressure, a standard definition; both values use the
    Bolton (1980) eq. (10) fit (see :func:`saturation_vapor_pressure`).

    Parameters
    ----------
    temperature, dewpoint : xarray.DataArray or array-like
        Temperature and dew point in K.
    engine, n_threads : optional
        See :func:`saturation_vapor_pressure`.

    Returns
    -------
    xarray.DataArray
        Relative humidity (fraction, units ``1``).

    References
    ----------
    Bolton, D., 1980: The computation of equivalent potential temperature.
    Mon. Wea. Rev., 108, 1046-1053,
    https://doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2
    """
    es_t = saturation_vapor_pressure(temperature, engine=engine, n_threads=n_threads)
    es_d = saturation_vapor_pressure(dewpoint, engine=engine, n_threads=n_threads)
    return (
        (es_d / es_t)
        .rename("relative_humidity")
        .assign_attrs(ATTRS["relative_humidity"])
    )


def wet_bulb_temperature(
    pressure, temperature, dewpoint, *, engine="auto", n_threads=None
):
    """
    Isobaric wet-bulb temperature.

    Solves :math:`(c_{pd} + r c_{pv})(T - T_w) = L_v(T_w)\\,(r_s(T_w) - r)` for
    :math:`T_w` between the dew point and the temperature with a bracketed
    Newton iteration (to 1e-7 K), where :math:`r` is the mixing ratio,
    :math:`r_s` the saturation mixing ratio over liquid water and
    :math:`L_v(T) = (2.501 - 0.00237 (T - 273.15)) \\times 10^6` J kg-1
    (attributed to Bolton 1980; the equation number, formerly given as (2),
    and the coefficients are not checked against the paper), with
    :math:`r_s` from the Bolton (1980) eq. (10) vapour pressure. The energy
    balance is the textbook isobaric wet-bulb definition. :math:`c_{pd} =
    1005.7` and :math:`c_{pv} = 1875.0` J kg-1 K-1 are radarx's values and
    are not traced to a table of the cited papers. It is not the
    pseudo-adiabatic wet-bulb temperature of Davies-Jones (2008), which
    inverts Bolton's equivalent potential temperature; the size of the
    difference is not stated by Davies-Jones (2008). The Newton tolerance of 1e-7 K is a radarx choice.

    Parameters
    ----------
    pressure : xarray.DataArray or array-like
        Pressure in Pa.
    temperature, dewpoint : xarray.DataArray or array-like
        Temperature and dew point in K.
    engine, n_threads : optional
        See :func:`saturation_vapor_pressure`.

    Returns
    -------
    xarray.DataArray
        Wet-bulb temperature in K.

    References
    ----------
    Bolton, D., 1980: The computation of equivalent potential temperature.
    Mon. Wea. Rev., 108, 1046-1053,
    https://doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2

    Davies-Jones, R., 2008: An efficient and accurate method for computing the
    wet-bulb temperature along pseudoadiabats. Mon. Wea. Rev., 136,
    2764-2785, https://doi.org/10.1175/2007MWR2224.1
    """
    return _wrap(
        "wet_bulb",
        "wet_bulb_temperature",
        pressure,
        temperature,
        dewpoint,
        engine=engine,
        n_threads=n_threads,
    )


def air_density(
    pressure, temperature, specific_humidity=0.0, *, engine="auto", n_threads=None
):
    """
    Density of moist air, :math:`\\rho = p / (R_d T_v)`.

    The virtual temperature is :math:`T_v = T (1 + (1/\\epsilon - 1) q)`
    (:math:`1/\\epsilon - 1 = 0.6078`). Both relations are standard textbook
    definitions (ideal-gas law for moist air); no paper-specific coefficient
    is used and no reference is cited. :math:`R_d` is radarx's value (see
    :mod:`radarx.io.sounding`).

    Parameters
    ----------
    pressure : xarray.DataArray or array-like
        Pressure in Pa.
    temperature : xarray.DataArray or array-like
        Temperature in K.
    specific_humidity : xarray.DataArray or array-like, optional
        Specific humidity in kg kg-1. Default 0 (dry air).
    engine, n_threads : optional
        See :func:`saturation_vapor_pressure`.

    Returns
    -------
    xarray.DataArray
        Air density in kg m-3.
    """
    return _wrap(
        "density",
        "air_density",
        pressure,
        temperature,
        specific_humidity,
        engine=engine,
        n_threads=n_threads,
    )


def geopotential_to_height(geopotential, *, engine="auto", n_threads=None):
    """
    Geometric height from geopotential.

    :math:`H = \\Phi / g_0` is the geopotential height and
    :math:`z = R_e H / (R_e - H)` the geometric height on a spherical Earth
    (:math:`g_0 = 9.80665` m s-2, :math:`R_e = 6371.0088` km). The relation
    follows from gravity decreasing with the inverse square of the distance
    from the Earth's centre (a standard derivation); :math:`g_0` is the
    conventional standard gravity and :math:`R_e` the GRS80 mean radius
    (Moritz 2000), both quoted from the standards (not checked against them).

    Parameters
    ----------
    geopotential : xarray.DataArray or array-like
        Geopotential in m2 s-2 (multiply a geopotential height by
        ``9.80665`` first).
    engine, n_threads : optional
        See :func:`saturation_vapor_pressure`.

    Returns
    -------
    xarray.DataArray
        Geometric height in m.

    References
    ----------
    Moritz, H., 2000: Geodetic Reference System 1980. J. Geodesy, 74,
    128-133, https://doi.org/10.1007/s001900050278
    """
    out = _wrap("height", "height", geopotential, engine=engine, n_threads=n_threads)
    return out.assign_attrs(ATTRS["height"])


# ---------------------------------------------------------------------------
# building profiles


def _to_datetime64(time):
    """``time`` (str, datetime, numpy or xarray scalar) as naive UTC datetime64[s]."""
    if isinstance(time, xr.DataArray):
        time = time.values
    if isinstance(time, _dt.datetime) and time.tzinfo is not None:
        time = time.astimezone(_dt.UTC).replace(tzinfo=None)
    if isinstance(time, str) and time.endswith("Z"):
        time = time[:-1]
    out = np.datetime64(np.asarray(time).ravel()[0] if np.ndim(time) else time, "s")
    if np.isnat(out):
        raise ValueError(f"cannot interpret time {time!r}")
    return out


def _dt_from64(t):
    return _dt.datetime.fromisoformat(str(np.datetime64(t, "s")))


def _dedupe(height, columns):
    """Sort by height and merge levels with the same height (mean of valid values)."""
    ok = np.isfinite(height)
    height = height[ok]
    columns = {k: v[ok] for k, v in columns.items()}
    uniq, inverse = np.unique(height, return_inverse=True)
    out = {}
    for name, values in columns.items():
        valid = np.isfinite(values)
        total = np.bincount(inverse, np.where(valid, values, 0.0), uniq.size)
        count = np.bincount(inverse, valid.astype(float), uniq.size)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[name] = np.where(count > 0, total / count, np.nan)
    return uniq, out


def _fill_height(height, pressure):
    """Fill missing heights by interpolating them in log-pressure."""
    height = height.copy()
    missing = ~np.isfinite(height) & np.isfinite(pressure) & (pressure > 0)
    known = np.isfinite(height) & np.isfinite(pressure) & (pressure > 0)
    if missing.any() and known.sum() >= 2:
        lp = -np.log(pressure[known])
        order = np.argsort(lp)
        lpm = -np.log(pressure[missing])
        inside = (lpm >= lp[order][0]) & (lpm <= lp[order][-1])
        filled = np.interp(lpm, lp[order], height[known][order])
        height[np.nonzero(missing)[0][inside]] = filled[inside]
    return height


def _profile_dataset(
    *,
    pressure=None,
    geopotential_height=None,
    height=None,
    temperature=None,
    dewpoint=None,
    relative_humidity=None,
    specific_humidity=None,
    u=None,
    v=None,
    w=None,
    wind_speed=None,
    wind_direction=None,
    time=None,
    latitude=np.nan,
    longitude=np.nan,
    attrs=None,
    engine="auto",
):
    """Assemble a profile Dataset from SI NumPy columns (any may be None)."""
    cols = dict(
        pressure=pressure,
        geopotential_height=geopotential_height,
        height=height,
        temperature=temperature,
        dewpoint=dewpoint,
        relative_humidity=relative_humidity,
        specific_humidity=specific_humidity,
        u=u,
        v=v,
        w=w,
    )
    n = max(np.size(c) for c in cols.values() if c is not None)
    cols = {
        k: (np.full(n, np.nan) if c is None else np.asarray(c, dtype=np.float64))
        for k, c in cols.items()
    }
    if u is None and wind_speed is not None and wind_direction is not None:
        spd = np.asarray(wind_speed, dtype=np.float64)
        rad = np.radians(np.asarray(wind_direction, dtype=np.float64))
        cols["u"] = -spd * np.sin(rad)
        cols["v"] = -spd * np.cos(rad)
    if height is None:
        gph = _fill_height(cols["geopotential_height"], cols["pressure"])
        cols["geopotential_height"] = gph
        cols["height"] = _thermo("height", gph * G0, engine=engine)
    z = cols.pop("height")
    z, cols = _dedupe(z, cols)

    p, t, td, q = (
        cols[k] for k in ("pressure", "temperature", "dewpoint", "specific_humidity")
    )
    if np.isnan(td).all() and np.isfinite(q).any():
        e = _thermo("vapor_pressure", q, p, engine=engine)
        td = _thermo("dewpoint", e, engine=engine)
    if np.isnan(td).all() and np.isfinite(cols["relative_humidity"]).any():
        e = cols["relative_humidity"] * _thermo("esat", t, engine=engine)
        td = _thermo("dewpoint", e, engine=engine)
    if np.isnan(q).all():
        e = _thermo("esat", td, engine=engine)
        q = _thermo("specific_humidity", e, p, engine=engine)
    rh = cols["relative_humidity"]
    if np.isnan(rh).all():
        rh = _thermo("esat", td, engine=engine) / _thermo("esat", t, engine=engine)
    cols.update(dewpoint=td, specific_humidity=q, relative_humidity=rh)
    cols["wind_speed"] = np.hypot(cols["u"], cols["v"])
    cols["wind_direction"] = np.mod(
        np.degrees(np.arctan2(-cols["u"], -cols["v"])), 360.0
    )
    cols["wind_direction"][~(cols["wind_speed"] > 0)] = np.where(
        cols["wind_speed"][~(cols["wind_speed"] > 0)] == 0, 0.0, np.nan
    )

    data_vars = {}
    for name in PROFILE_VARS:
        values = cols.get(name)
        if values is None or (
            name in ("w", "geopotential_height") and np.isnan(values).all()
        ):
            continue
        data_vars[name] = ("height", values, dict(ATTRS[name]))
    coords = {
        "height": ("height", z, dict(ATTRS["height"])),
        "latitude": (
            (),
            float(latitude),
            {"standard_name": "latitude", "units": "degrees_north"},
        ),
        "longitude": (
            (),
            float(longitude),
            {"standard_name": "longitude", "units": "degrees_east"},
        ),
    }
    if time is not None:
        coords["time"] = ((), np.datetime64(time, "ns"), {"standard_name": "time"})
    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs = {"Conventions": "CF-1.8", "featureType": "profile"}
    ds.attrs.update(attrs or {})
    return ds


# ---------------------------------------------------------------------------
# parsers (text -> profile Dataset)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _parse_iem(text, station=None, time=None, engine="auto"):
    """IEM RAOB JSON (``json/raob.py``) to a profile Dataset."""
    data = json.loads(text)
    profiles = data.get("profiles", [])
    if station is not None:
        profiles = [p for p in profiles if p.get("station") == station] or profiles
    if time is not None:
        want = _to_datetime64(time)
        profiles = [p for p in profiles if _to_datetime64(p["valid"]) == want] or []
    if not profiles or not profiles[0].get("profile"):
        raise ValueError("no sounding found in the IEM response")
    prof = profiles[0]
    rows = prof["profile"]

    def col(key):
        return np.array([_num(r.get(key)) for r in rows])

    meta = _station_meta(prof["station"], "iem")
    valid = _to_datetime64(prof["valid"])
    return _profile_dataset(
        pressure=col("pres") * 100.0,
        geopotential_height=col("hght"),
        temperature=col("tmpc") + T0,
        dewpoint=col("dwpc") + T0,
        wind_speed=col("sknt") * KNOT,
        wind_direction=col("drct"),
        time=valid,
        latitude=meta.get("latitude", np.nan),
        longitude=meta.get("longitude", np.nan),
        attrs={
            "source": "Iowa Environmental Mesonet RAOB archive",
            "source_url": "https://mesonet.agron.iastate.edu/archive/raob/",
            "station": prof["station"],
            "station_name": meta.get("name", ""),
            "launch_time": str(valid),
        },
        engine=engine,
    )


def _parse_uwyo(text, station="", engine="auto"):
    """University of Wyoming CSV (``wsgi/sounding ... TEXT:CSV``) to a profile."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines or "pressure" not in lines[0].lower():
        raise ValueError("no sounding found in the University of Wyoming response")
    reader = csv.reader(lines)
    header = [h.strip().lower() for h in next(reader)]
    rows = list(reader)

    def col(*keys):
        for key in keys:
            for i, h in enumerate(header):
                if h.startswith(key):
                    return np.array(
                        [_num(r[i]) if i < len(r) else np.nan for r in rows]
                    )
        return None

    times = [r[0] for r in rows if r and r[0].strip()]
    launch = _to_datetime64(times[0].strip().replace(" ", "T")) if times else None
    lat, lon = col("latitude"), col("longitude")
    meta = _station_meta(str(station), "uwyo") if station else {}
    return _profile_dataset(
        pressure=col("pressure") * 100.0,
        geopotential_height=col("geopotential height"),
        temperature=col("temperature") + T0,
        dewpoint=col("dew point") + T0,
        wind_speed=col("wind speed"),
        wind_direction=col("wind direction"),
        time=launch,
        latitude=(
            lat[0] if lat is not None and lat.size else meta.get("latitude", np.nan)
        ),
        longitude=(
            lon[0] if lon is not None and lon.size else meta.get("longitude", np.nan)
        ),
        attrs={
            "source": "University of Wyoming upper-air archive",
            "source_url": "https://weather.uwyo.edu/upperair/sounding.shtml",
            "station": str(station),
            "station_name": meta.get("name", ""),
            "launch_time": str(launch),
        },
        engine=engine,
    )


def _igra2_header(line):
    """
    Fields of an IGRA2 header record (fixed columns of the format document).

    The column positions follow the IGRA version 2 format description
    distributed with the archive (Durre et al. 2006, 2018; dataset doi
    10.7289/V5X63K0Q); they are not checked against that document. Latitude and longitude are in ten-thousandths of a degree.
    """
    return {
        "id": line[1:12].strip(),
        "time": np.datetime64(
            f"{line[13:17]}-{line[18:20]}-{line[21:23]}T{int(line[24:26]) % 99:02d}:00"
        ),
        "reltime": line[27:31].strip(),
        "latitude": _num(line[55:62]) / 1e4,
        "longitude": _num(line[63:71]) / 1e4,
    }


def _iter_igra2(lines):
    """Yield (header dict, data lines) for every sounding in IGRA2 text."""
    header, body = None, []
    for line in lines:
        if line.startswith("#"):
            if header is not None:
                yield header, body
            header, body = line, []
        elif header is not None:
            body.append(line)
    if header is not None:
        yield header, body


def _parse_igra2_sounding(header, body, engine="auto"):
    h = _igra2_header(header)

    def field(a, b, scale=1.0):
        vals = np.array([_num(ln[a:b]) for ln in body])
        vals[(vals == -9999) | (vals == -8888)] = np.nan
        return vals * scale

    # IGRA2 data records: pressure in Pa, geopotential height in m, temperature
    # and dew-point depression in tenths of degC, wind speed in tenths of m s-1;
    # -9999 / -8888 are missing (columns as in the archive's format description)
    t = field(22, 27, 0.1)
    launch = h["time"]
    if h["reltime"].isdigit() and not h["reltime"].startswith("99"):
        hh, mm = int(h["reltime"][:2]), int(h["reltime"][2:])
        launch = np.datetime64(str(h["time"])[:10]) + np.timedelta64(hh * 60 + mm, "m")
        if launch > h["time"] + np.timedelta64(12, "h"):
            launch -= np.timedelta64(1, "D")
    meta = _station_meta(h["id"], "igra2")
    return _profile_dataset(
        pressure=field(9, 15),
        geopotential_height=field(16, 21),
        temperature=t + T0,
        dewpoint=t - field(34, 39, 0.1) + T0,
        wind_direction=field(40, 45),
        wind_speed=field(46, 51, 0.1),
        time=h["time"],
        latitude=h["latitude"],
        longitude=h["longitude"],
        attrs={
            "source": "Integrated Global Radiosonde Archive version 2 (NOAA NCEI)",
            "source_url": "https://doi.org/10.7289/V5X63K0Q",
            "station": h["id"],
            "station_name": meta.get("name", ""),
            "launch_time": str(launch),
        },
        engine=engine,
    )


def _parse_igra2(lines, time=None, engine="auto"):
    """The IGRA2 sounding at ``time`` (or the first) from IGRA2 text lines."""
    want = None if time is None else _to_datetime64(time)
    for header, body in _iter_igra2(lines):
        if want is None or _igra2_header(header)["time"] == want:
            return _parse_igra2_sounding(header, body, engine=engine)
    raise ValueError(f"no IGRA2 sounding found for {time}")


def _parse_sharppy(text, engine="auto"):
    """SHARPpy / SPC text (``%TITLE%`` ... ``%RAW%`` ... ``%END%``) to a profile."""
    lines = text.splitlines()
    station, time = "", None
    if "%TITLE%" in text:
        title = lines[[ln.strip() for ln in lines].index("%TITLE%") + 1].split()
        if title:
            station = title[0]
        if len(title) > 1:
            m = re.match(r"(\d{2})(\d{2})(\d{2})/(\d{2})(\d{2})", title[1])
            if m:
                yy, mo, dd, hh, mi = (int(g) for g in m.groups())
                year = 2000 + yy if yy < 70 else 1900 + yy
                time = np.datetime64(f"{year:04d}-{mo:02d}-{dd:02d}T{hh:02d}:{mi:02d}")
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "%RAW%") + 1
    rows = []
    for ln in lines[start:]:
        if ln.strip() == "%END%":
            break
        if ln.strip():
            rows.append([_num(x) for x in ln.split(",")])
    data = np.array(rows, dtype=np.float64)
    data[data <= -9998] = np.nan
    meta = _station_meta(station, "iem") if station else {}
    return _profile_dataset(
        pressure=data[:, 0] * 100.0,
        geopotential_height=data[:, 1],
        temperature=data[:, 2] + T0,
        dewpoint=data[:, 3] + T0,
        wind_direction=data[:, 4],
        wind_speed=data[:, 5] * KNOT,
        time=time,
        latitude=meta.get("latitude", np.nan),
        longitude=meta.get("longitude", np.nan),
        attrs={
            "source": "SHARPpy/SPC sounding file",
            "station": station,
            "station_name": meta.get("name", ""),
            "launch_time": str(time),
        },
        engine=engine,
    )


_CSV_ALIASES = {
    "pressure": ("pressure", "pres", "p", "pressure_hpa"),
    "height": ("height", "hght", "z", "height_m", "geopotential_height"),
    "temperature": ("temperature", "temp", "tmpc", "t", "temperature_c"),
    "dewpoint": ("dewpoint", "dwpt", "dwpc", "td", "dewpoint_c"),
    "wind_direction": ("wind_direction", "drct", "wdir", "direction"),
    "wind_speed": ("wind_speed", "wspd", "speed"),
    "sknt": ("sknt", "wind_speed_kt", "speed_kt"),
    "u": ("u",),
    "v": ("v",),
}


def _parse_csv(text, engine="auto"):
    """Generic CSV with named columns (hPa, m, degC, degree, m/s or knots)."""
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    reader = csv.reader(lines)
    header = [h.strip().lower() for h in next(reader)]
    rows = list(reader)

    def col(name):
        for alias in _CSV_ALIASES[name]:
            if alias in header:
                i = header.index(alias)
                values = np.array([_num(r[i]) if i < len(r) else np.nan for r in rows])
                values[values <= -9998] = np.nan
                return values
        return None

    p, h, t, td = col("pressure"), col("height"), col("temperature"), col("dewpoint")
    if t is None or (p is None and h is None):
        raise ValueError("CSV needs a temperature column and pressure or height")
    spd = col("wind_speed")
    if spd is None and col("sknt") is not None:
        spd = col("sknt") * KNOT
    return _profile_dataset(
        pressure=None if p is None else p * 100.0,
        geopotential_height=h if h is not None else np.full(t.size, np.nan),
        temperature=t + T0,
        dewpoint=None if td is None else td + T0,
        u=col("u"),
        v=col("v"),
        wind_speed=spd,
        wind_direction=col("wind_direction"),
        attrs={"source": "CSV file"},
        engine=engine,
    )


def open_sounding_file(path, format="auto", *, time=None, station=None, engine="auto"):
    """
    Read a sounding from a local file.

    Parameters
    ----------
    path : str or os.PathLike
        File to read.
    format : {"auto", "csv", "sharppy", "uwyo", "igra2", "iem"}, optional
        File format. ``"auto"`` (default) detects it from the content:

        - ``"sharppy"``: SHARPpy/SPC text with ``%TITLE%`` and ``%RAW%``
          sections; columns pressure (hPa), height (m), temperature,
          dew point (degC), wind direction (degree), wind speed (knots).
        - ``"uwyo"``: University of Wyoming CSV (``TEXT:CSV``).
        - ``"igra2"``: IGRA2 station text file (or its ``.zip``); the
          sounding at ``time`` is read, by default the first.
        - ``"iem"``: IEM RAOB JSON.
        - ``"csv"``: a CSV file with a header naming the columns (case
          insensitive): ``pressure``/``pres`` (hPa), ``height``/``hght``
          (geopotential height, m), ``temperature``/``tmpc`` (degC),
          ``dewpoint``/``dwpc`` (degC), ``wind_direction``/``drct``
          (degree) with ``wind_speed`` (m/s) or ``sknt`` (knots), or ``u``,
          ``v`` (m/s). Values <= -9998 are missing.
    time : str or datetime-like, optional
        Sounding to read from a multi-sounding file (IGRA2, IEM).
    station : str, optional
        Station identifier, used for metadata (UWyo, IEM).
    engine : {"auto", "compiled", "numpy"}, optional
        Kernel for the derived thermodynamic variables.

    Returns
    -------
    xarray.Dataset
        Profile on ``height`` (see :mod:`radarx.io.sounding`).
    """
    path = Path(path)
    if path.suffix == ".zip":
        return _read_igra2_zip(path, time, engine=engine)
    text = path.read_text(errors="replace")
    if format == "auto":
        stripped = text.lstrip()
        if stripped.startswith("{"):
            format = "iem"
        elif "%RAW%" in text:
            format = "sharppy"
        elif re.match(r"#\w{11} \d{4} ", stripped):
            format = "igra2"
        elif "geopotential height" in stripped.splitlines()[0].lower():
            format = "uwyo"
        else:
            format = "csv"
    if format == "iem":
        return _parse_iem(text, station=station, time=time, engine=engine)
    if format == "sharppy":
        return _parse_sharppy(text, engine=engine)
    if format == "igra2":
        return _parse_igra2(text.splitlines(), time=time, engine=engine)
    if format == "uwyo":
        return _parse_uwyo(text, station=station or "", engine=engine)
    if format == "csv":
        return _parse_csv(text, engine=engine)
    raise ValueError(f"unknown format {format!r}")


# ---------------------------------------------------------------------------
# stations


@functools.lru_cache(maxsize=1)
def _stations():
    """The shipped station table as a dict of NumPy arrays."""
    path = Path(__file__).parent / "data" / "sounding_stations.csv"
    with open(path, newline="") as f:
        rows = list(csv.reader(ln for ln in f if not ln.startswith("#")))
    header, rows = rows[0], rows[1:]
    table = {h: np.array([r[i] for r in rows]) for i, h in enumerate(header)}
    for key in ("latitude", "longitude", "elevation"):
        table[key] = table[key].astype(np.float64)
    for key in ("first_year", "last_year"):
        table[key] = table[key].astype(int)
    return table


_ID_COLUMN = {"iem": "iem_id", "uwyo": "wmo_id", "igra2": "igra2_id"}


def _station_index(station):
    """Row of ``station`` (IGRA2, WMO or IEM/ICAO identifier) in the table."""
    table = _stations()
    s = str(station).strip().upper()
    for column in ("igra2_id", "wmo_id", "iem_id"):
        hit = np.nonzero(table[column] == s)[0]
        if hit.size:
            return int(hit[0])
    if len(s) == 3:  # e.g. "JAN" for KJAN
        hit = np.nonzero(table["iem_id"] == "K" + s)[0]
        if hit.size:
            return int(hit[0])
    return None


def _station_meta(station, source=None):
    i = _station_index(station)
    if i is None:
        return {}
    t = _stations()
    return {k: t[k][i].item() for k in t}


def _station_id(station, source):
    """Identifier of ``station`` as used by ``source``."""
    meta = _station_meta(station, source)
    sid = meta.get(_ID_COLUMN[source], "")
    if not sid:
        if source == "iem":
            return str(station).upper()
        if source == "uwyo" and str(station).isdigit():
            return str(station)
        raise ValueError(f"no {source} identifier known for station {station!r}")
    return sid


def station_list():
    """
    Radiosonde stations known to radarx.

    The table merges the IGRA2 station list (NOAA NCEI) with the Iowa
    Environmental Mesonet RAOB network, matched on the WMO station number,
    for stations reporting after 1990.

    Returns
    -------
    xarray.Dataset
        On a ``station`` dimension: ``igra2_id``, ``wmo_id``, ``iem_id``,
        ``latitude``, ``longitude``, ``elevation`` (m), ``name``,
        ``first_year`` and ``last_year``.
    """
    t = _stations()
    ds = xr.Dataset({k: ("station", v) for k, v in t.items()})
    ds["latitude"].attrs = {"standard_name": "latitude", "units": "degrees_north"}
    ds["longitude"].attrs = {"standard_name": "longitude", "units": "degrees_east"}
    ds["elevation"].attrs = {"long_name": "station elevation", "units": "m"}
    return ds


def nearest_station(latitude, longitude, time=None, *, source=None, n=1):
    """
    Radiosonde stations nearest to a location.

    Parameters
    ----------
    latitude, longitude : float
        Location in degrees (e.g. the radar site).
    time : str or datetime-like, optional
        Only stations whose record covers this year.
    source : {"iem", "uwyo", "igra2"}, optional
        Only stations with an identifier for this archive.
    n : int, optional
        Number of stations to return. Default 1.

    Returns
    -------
    xarray.Dataset
        The :func:`station_list` rows of the ``n`` nearest stations with
        their great-circle ``distance`` (m), nearest first.
    """
    ds = station_list()
    keep = np.ones(ds.sizes["station"], dtype=bool)
    if source is not None:
        keep &= ds[_ID_COLUMN[source]].values != ""
    if time is not None:
        year = _dt_from64(_to_datetime64(time)).year
        keep &= (ds["first_year"].values <= year) & (ds["last_year"].values >= year)
    ds = ds.isel(station=np.nonzero(keep)[0])
    lat1, lon1 = np.radians(latitude), np.radians(longitude)
    lat2, lon2 = np.radians(ds["latitude"].values), np.radians(ds["longitude"].values)
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    distance = 2 * 6371008.8 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    order = np.argsort(distance)[:n]
    out = ds.isel(station=order)
    out["distance"] = (
        "station",
        distance[order],
        {"long_name": "great-circle distance", "units": "m"},
    )
    return out


# ---------------------------------------------------------------------------
# downloads


def _cache_dir(*parts):
    root = os.environ.get("RADARX_CACHE_DIR")
    if root is None:
        import pooch

        root = pooch.os_cache("radarx")
    path = Path(root, "soundings", *parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download(url, fname, subdir, cache=True):
    """Download ``url`` into the cache (once) and return the local path."""
    import pooch

    path = _cache_dir(subdir)
    target = path / fname
    if target.exists() and not cache:
        target.unlink()
    return Path(
        pooch.retrieve(url, known_hash=None, fname=fname, path=path, progressbar=False)
    )


def read_sounding(station, time, source="iem", *, cache=True, engine="auto"):
    """
    Download an observed radiosonde sounding.

    Parameters
    ----------
    station : str or int
        Station identifier: IEM/ICAO (``"KJAN"``, ``"JAN"``), WMO number
        (``72235``) or IGRA2 identifier (``"USM00072235"``); it is translated
        for the chosen archive with the shipped station table.
    time : str or datetime-like
        Nominal launch time (UTC), usually 00 or 12 UTC.
    source : {"iem", "uwyo", "igra2"}, optional
        Archive. Default ``"iem"`` (fastest).
    cache : bool, optional
        Reuse a previous download. Default True.
    engine : {"auto", "compiled", "numpy"}, optional
        Kernel for the derived thermodynamic variables.

    Returns
    -------
    xarray.Dataset
        Profile on ``height`` (see :mod:`radarx.io.sounding`).

    References
    ----------
    Durre, I., R. S. Vose, and D. B. Wuertz, 2006: Overview of the Integrated
    Global Radiosonde Archive. J. Climate, 19, 53-68,
    https://doi.org/10.1175/JCLI3594.1

    Examples
    --------
    >>> ds = read_sounding("KJAN", "2022-03-31T00:00")  # doctest: +SKIP
    """
    t = _dt_from64(_to_datetime64(time))
    if source == "iem":
        sid = _station_id(station, "iem")
        url = IEM_URL.format(ts=t.strftime("%Y%m%d%H%M"), station=sid)
        path = _download(url, f"iem_{sid}_{t:%Y%m%d%H%M}.json", "iem", cache)
        try:
            return _parse_iem(path.read_text(), station=sid, engine=engine)
        except ValueError:
            path.unlink(missing_ok=True)
            raise
    if source == "uwyo":
        sid = _station_id(station, "uwyo")
        url = UWYO_URL.format(date=t.strftime("%Y-%m-%d"), hour=t.hour, station=sid)
        path = _download(url, f"uwyo_{sid}_{t:%Y%m%d%H}.csv", "uwyo", cache)
        try:
            return _parse_uwyo(path.read_text(), station=sid, engine=engine)
        except ValueError:
            path.unlink(missing_ok=True)
            raise
    if source == "igra2":
        sid = _station_id(station, "igra2")
        return _read_igra2_remote(sid, t, cache, engine)
    raise ValueError(f"unknown source {source!r}; use 'iem', 'uwyo' or 'igra2'")


def _read_igra2_zip(path, time, engine="auto"):
    """The sounding at ``time`` (or the first) from a zipped IGRA2 station file."""
    import zipfile

    with zipfile.ZipFile(path) as z:
        with z.open(z.namelist()[0]) as f:
            lines = io.TextIOWrapper(f, encoding="ascii", errors="replace")
            return _parse_igra2(lines, time=time, engine=engine)


def _read_igra2_remote(sid, t, cache, engine):
    """IGRA2: try the current-year file first, then the period-of-record file."""
    import urllib.error

    errors = []
    candidates = [
        (
            f"{IGRA2_URL}/data-y2d/{sid}-data-beg{t.year}.txt.zip",
            f"{sid}-data-beg{t.year}.txt.zip",
        ),
        (f"{IGRA2_URL}/data-por/{sid}-data.txt.zip", f"{sid}-data.txt.zip"),
    ]
    for url, fname in candidates:
        try:
            path = _download(url, fname, "igra2", cache)
        except Exception as err:  # noqa: BLE001 - try the next file
            errors.append(err)
            continue
        try:
            return _read_igra2_zip(path, np.datetime64(t, "s"), engine=engine)
        except ValueError as err:
            errors.append(err)
    raise ValueError(f"no IGRA2 sounding for {sid} at {t}: {errors}") from (
        errors[-1]
        if errors and not isinstance(errors[-1], urllib.error.URLError)
        else None
    )


# ---------------------------------------------------------------------------
# ERA5


def _cds_available():
    """Whether cdsapi is installed and CDS credentials are configured."""
    try:
        import cdsapi  # noqa: F401
    except ImportError:
        return False
    if os.environ.get("CDSAPI_KEY") or os.environ.get("ECMWF_DATASTORES_KEY"):
        return True
    rc = os.environ.get("CDSAPI_RC", str(Path.home() / ".cdsapirc"))
    return Path(rc).exists() or (Path.home() / ".ecmwfdatastoresrc").exists()


def _resolve_source(source, point):
    if source == "auto":
        return "cds" if _cds_available() else "gcs"
    if source not in ("arco", "cds", "gcs"):
        raise ValueError(
            f"unknown ERA5 source {source!r}; use 'auto', 'arco', 'cds' or 'gcs'"
        )
    return source


def _time_bracket(time, step_hours, method):
    """Times to read and their weights for linear or nearest time interpolation."""
    t = _to_datetime64(time)
    step = np.timedelta64(step_hours, "h")
    t0 = t.astype("datetime64[h]")
    t0 = t0 - (t0.astype(np.int64) % step_hours) * np.timedelta64(1, "h")
    t0 = t0.astype("datetime64[s]")
    w = float((t - t0) / np.timedelta64(1, "s")) / (step_hours * 3600.0)
    if method == "nearest":
        return (np.array([t0 + step if w >= 0.5 else t0]), np.array([1.0]))
    if method != "linear":
        raise ValueError("time_interpolation must be 'linear' or 'nearest'")
    if w == 0.0:
        return np.array([t0]), np.array([1.0])
    return np.array([t0, t0 + step]), np.array([1.0 - w, w])


_ERA5_NAMES = {
    "geopotential": ("z", "geopotential"),
    "temperature": ("t", "temperature"),
    "specific_humidity": ("q", "specific_humidity"),
    "u": ("u", "u_component_of_wind"),
    "v": ("v", "v_component_of_wind"),
    "omega": ("w", "vertical_velocity"),
}


def _standardize_era5(ds):
    """Rename an ERA5 download to (time, level, latitude, longitude) with ascending axes."""
    rename = {}
    for name in ("valid_time", "pressure_level", "pressureLevel"):
        if name in ds.dims or name in ds.coords:
            rename[name] = {"valid_time": "time"}.get(name, "level")
    ds = ds.rename(rename)
    out = {}
    for std, keys in _ERA5_NAMES.items():
        for key in keys:
            if key in ds:
                out[std] = ds[key]
                break
    ds = xr.Dataset(out)
    for dim in ("latitude", "longitude"):
        if dim not in ds.dims:
            ds = ds.expand_dims(dim)
    ds = ds.drop_vars(
        [c for c in ds.coords if c not in ("time", "level", "latitude", "longitude")]
    )
    lon = ds["longitude"].values
    ds = ds.assign_coords(longitude=((lon + 180.0) % 360.0) - 180.0)
    ds = ds.sortby(["latitude", "longitude"]).sortby("level", ascending=False)
    return ds.transpose("time", "level", "latitude", "longitude")


def _cache_key(*parts):
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:16]


def _era5_fields(source, box, times, cache=True):
    """
    ERA5 pressure-level fields covering ``box`` at ``times``.

    Returns an (time, level, latitude, longitude) Dataset with geopotential,
    temperature, specific_humidity, u, v and omega.
    """
    times = [np.datetime64(t, "s") for t in times]
    if source == "gcs":
        # The store has one global chunk per field and hour, so reading a
        # larger area costs nothing extra: cache a region GCS_PAD degrees
        # around the request per hour and reuse it for nearby points and grids.
        south, north, west, east = box
        parts = []
        for t in times:
            part = _gcs_cached(box, t) if cache else None
            if part is None:
                region = (
                    max(south - GCS_PAD, -90.0),
                    min(north + GCS_PAD, 90.0),
                    west - GCS_PAD,
                    east + GCS_PAD,
                )
                part = _standardize_era5(_era5_gcs(region, [t])).load()
                if cache:
                    stamp = str(t).replace(":", "").replace("-", "")
                    name = "era5_gcs_{}_{:+.2f}_{:+.2f}_{:+.2f}_{:+.2f}.nc".format(
                        stamp, *region
                    )
                    part.to_netcdf(_cache_dir("era5") / name)
            parts.append(part)
        ds = xr.concat(parts, "time") if len(parts) > 1 else parts[0]
        ds = ds.sel(
            latitude=slice(south - 1e-6, north + 1e-6),
            longitude=slice(west - 1e-6, east + 1e-6),
        )
    else:
        key = _cache_key(source, [round(b, 3) for b in box], [str(t) for t in times])
        path = _cache_dir("era5") / f"era5_{source}_{key}.nc"
        if cache and path.exists():
            return xr.load_dataset(path)
        raw = (
            _era5_cds(box, times, path)
            if source == "cds"
            else _era5_arco(box, times, path)
        )
        ds = _standardize_era5(raw).load()
        if cache:
            ds.sel(time=np.array(times, dtype="datetime64[ns]")).to_netcdf(path)
    ds = ds.sel(time=np.array(times, dtype="datetime64[ns]"))
    ds.attrs["era5_source"] = source
    return ds


def _gcs_cached(box, t):
    """A cached Google ARCO-ERA5 region at time ``t`` that contains ``box``."""
    south, north, west, east = box
    stamp = str(np.datetime64(t, "s")).replace(":", "").replace("-", "")
    for path in sorted(_cache_dir("era5").glob(f"era5_gcs_{stamp}_*.nc")):
        try:
            s, n, w, e = (float(v) for v in path.stem.split("_")[3:7])
        except ValueError:
            continue
        if s <= south and n >= north and w <= west and e >= east:
            return xr.load_dataset(path)
    return None


def _era5_gcs(box, times):
    """Lazy read of the Google ARCO-ERA5 Zarr store, selecting only the box and times."""
    south, north, west, east = box
    store = xr.open_zarr(
        ARCO_ERA5_ZARR,
        chunks=None,
        consolidated=True,
    )
    names = [keys[1] for keys in _ERA5_NAMES.values()]
    store = store[names].sel(time=np.array(times, dtype="datetime64[ns]"))
    lat = store["latitude"].values
    lon = store["longitude"].values  # 0 ... 359.75
    jlat = np.nonzero((lat >= south - 1e-6) & (lat <= north + 1e-6))[0]
    step = 0.25
    i0 = int(np.floor((west % 360.0) / step))
    n = int(np.ceil((east - west) / step)) + 1
    ilon = (i0 + np.arange(n)) % lon.size
    sub = store.isel(latitude=jlat, longitude=ilon)
    return sub.load()


def _cds_retrieve(dataset, request, path):
    import cdsapi

    tmp = path.with_suffix(".download.nc")
    cdsapi.Client(quiet=True, progress=False).retrieve(dataset, request, str(tmp))
    ds = xr.load_dataset(tmp)
    tmp.unlink(missing_ok=True)
    return ds


def _era5_cds(box, times, path):
    south, north, west, east = box
    dts = [_dt_from64(t) for t in times]
    request = {
        "product_type": ["reanalysis"],
        "variable": [keys[1] for keys in _ERA5_NAMES.values()],
        "year": sorted({f"{d:%Y}" for d in dts}),
        "month": sorted({f"{d:%m}" for d in dts}),
        "day": sorted({f"{d:%d}" for d in dts}),
        "time": sorted({f"{d:%H}:00" for d in dts}),
        "pressure_level": [str(p) for p in ERA5_LEVELS],
        "area": [north, west, south, east],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    return _cds_retrieve(CDS_PRESSURE_LEVELS, request, path)


def _era5_arco(box, times, path):
    south, north, west, east = box
    dts = [_dt_from64(t) for t in times]
    request = {
        "variable": [keys[1] for keys in _ERA5_NAMES.values()],
        "date": [f"{min(dts):%Y-%m-%d}/{max(dts):%Y-%m-%d}"],
        "data_format": "netcdf",
    }
    if north - south < 0.25 and east - west < 0.25:
        request["location"] = {
            "latitude": round(0.5 * (south + north), 4),
            "longitude": round(0.5 * (west + east), 4),
        }
    else:
        request["area"] = [north, west, south, east]
    return _cds_retrieve(CDS_ARCO_TIMESERIES, request, path)


def _era5_box(lat, lon, pad):
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    q = 0.25
    south = np.floor((np.nanmin(lat) - pad) / q) * q
    north = np.ceil((np.nanmax(lat) + pad) / q) * q
    west = np.floor((np.nanmin(lon) - pad) / q) * q
    east = np.ceil((np.nanmax(lon) + pad) / q) * q
    if north == south:
        north += q
    if east == west:
        east += q
    return (max(south, -90.0), min(north, 90.0), west, east)


def _era5_columns(source, lat, lon, time, time_interpolation, cache, engine, n_threads):
    """
    ERA5 columns at points: geometric heights (npts, nlev) ascending and a dict
    of (npts, nlev) arrays: pressure, temperature, specific_humidity, u, v, w.
    """
    point = np.size(lat) == 1
    src = _resolve_source(source, point)
    step = 6 if src == "arco" else 1
    times, weights = _time_bracket(time, step, time_interpolation)
    if src == "arco" and point:
        box = (float(lat), float(lat), float(lon), float(lon))
    else:
        box = _era5_box(lat, lon, 0.25)
    fields = _era5_fields(src, box, times, cache=cache)
    names = ["geopotential", "temperature", "specific_humidity", "u", "v", "omega"]
    level = fields["level"].values.astype(np.float64) * 100.0  # Pa, descending
    flat = fields["latitude"].values.astype(np.float64)
    flon = fields["longitude"].values.astype(np.float64)
    arr = np.stack([fields[n].values for n in names], axis=1).astype(np.float64)
    pres = np.broadcast_to(level[None, None, :, None, None], arr[:, :1].shape)
    arr = np.concatenate([arr, pres], axis=1)  # (ntime, nvar, nlev, nlat, nlon)
    if flat.size == 1:  # single grid point (ARCO location request)
        arr = np.repeat(arr, 2, axis=3)
        flat = np.array([flat[0] - 0.125, flat[0] + 0.125])
    if flon.size == 1:
        arr = np.repeat(arr, 2, axis=4)
        flon = np.array([flon[0] - 0.125, flon[0] + 0.125])
    qlat = np.atleast_1d(np.asarray(lat, dtype=np.float64)).ravel()
    qlon = np.atleast_1d(np.asarray(lon, dtype=np.float64)).ravel()
    qlon = ((qlon - flon[0]) % 360.0) + flon[0]
    if src == "arco" and point:  # the service returns the nearest grid point
        qlat = np.full_like(qlat, flat.mean())
        qlon = np.full_like(qlon, flon.mean())
    k = _kernel(engine)
    cols = np.asarray(
        k.bilinear_columns(
            _f64(arr),
            _f64(weights),
            _f64(flat),
            _f64(flon),
            _f64(qlat),
            _f64(qlon),
            n_threads=int(n_threads or 0),
        )
    )  # (nvar, npts, nlev)
    z = _thermo("height", cols[0], engine=engine, n_threads=n_threads)
    out = dict(
        zip(
            ["temperature", "specific_humidity", "u", "v", "omega", "pressure"],
            cols[1:],
        )
    )
    rho = _thermo(
        "density",
        out["pressure"],
        out["temperature"],
        out["specific_humidity"],
        engine=engine,
        n_threads=n_threads,
    )
    out["w"] = -out.pop("omega") / (rho * G0)
    order = np.argsort(np.nanmean(z, axis=0))  # ascending height
    z = z[:, order]
    out = {k_: v[:, order] for k_, v in out.items()}
    meta = {
        "source": {
            "arco": "ERA5 time series on pressure levels (ECMWF ARCO, Copernicus CDS)",
            "cds": "ERA5 hourly data on pressure levels (Copernicus CDS)",
            "gcs": "ERA5 (Google ARCO-ERA5 Zarr)",
        }[src],
        "era5_source": src,
        "source_url": {
            "arco": "https://doi.org/10.24381/af48f136",
            "cds": "https://doi.org/10.24381/cds.bd0915c6",
            "gcs": ARCO_ERA5_ZARR,
        }[src],
        "era5_times": ", ".join(str(t) for t in times),
        "era5_time_weights": ", ".join(f"{w:.4f}" for w in weights),
        "references": "Hersbach et al. (2020), https://doi.org/10.1002/qj.3803",
    }
    if src == "arco" and point:
        meta["era5_grid_point"] = f"{flat.mean():.2f}N {flon.mean():.2f}E (nearest)"
    return z, out, meta


def era5_profile(
    latitude,
    longitude,
    time,
    *,
    source="auto",
    time_interpolation="linear",
    cache=True,
    engine="auto",
    n_threads=None,
):
    """
    ERA5 profile at a point.

    Parameters
    ----------
    latitude, longitude : float
        Location in degrees.
    time : str or datetime-like
        Valid time (UTC).
    source : {"auto", "arco", "cds", "gcs"}, optional
        ERA5 provider (see :mod:`radarx.io.sounding`). ``"auto"`` (default)
        uses hourly ERA5 on 37 levels: the full CDS dataset when CDS
        credentials are configured, otherwise the anonymous Google ARCO-ERA5
        Zarr store. ``"arco"`` selects the 6-hourly ECMWF ARCO time series.
    time_interpolation : {"linear", "nearest"}, optional
        Interpolate linearly between the two ERA5 times bracketing ``time``
        (default) or take the nearest. ERA5 is hourly; the ECMWF ARCO time
        series is 6-hourly.
    cache : bool, optional
        Reuse cached downloads. Default True.
    engine : {"auto", "compiled", "numpy"}, optional
        Kernel implementation.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.

    Returns
    -------
    xarray.Dataset
        Profile on ``height`` (see :mod:`radarx.io.sounding`) with ``w``
        from the pressure velocity :math:`\\omega` as
        :math:`w = -\\omega / (\\rho g_0)` (hydrostatic). Horizontally the
        profile is bilinearly interpolated to the point (``"cds"``,
        ``"gcs"``) or taken at the nearest grid point (``"arco"``).

    References
    ----------
    Hersbach, H., and Coauthors, 2020: The ERA5 global reanalysis.
    Quart. J. Roy. Meteor. Soc., 146, 1999-2049,
    https://doi.org/10.1002/qj.3803
    """
    z, cols, meta = _era5_columns(
        source, latitude, longitude, time, time_interpolation, cache, engine, n_threads
    )
    t = _to_datetime64(time)
    meta.update(
        station=f"ERA5 {float(latitude):.3f}N {float(longitude):.3f}E",
        station_name="ERA5",
        launch_time=str(t),
    )
    return _profile_dataset(
        height=z[0],
        pressure=cols["pressure"][0],
        geopotential_height=None,
        temperature=cols["temperature"][0],
        specific_humidity=cols["specific_humidity"][0],
        u=cols["u"][0],
        v=cols["v"][0],
        w=cols["w"][0],
        time=t,
        latitude=latitude,
        longitude=longitude,
        attrs=meta,
        engine=engine,
    )


# ---------------------------------------------------------------------------
# profile helpers


def _profile_columns(profile, variables):
    """(ncol, nlev) heights, (nvar, ncol, nlev) values and the column dims."""
    if "height" not in profile.dims:
        raise ValueError("the profile needs a 'height' dimension")
    other = (
        [d for d in profile[variables[0]].dims if d != "height"] if variables else []
    )
    for name in variables:
        extra = set(profile[name].dims) - {"height"} - set(other)
        if extra:
            raise ValueError(f"{name} has unexpected dimensions {extra}")
    z = profile["height"]
    if z.ndim == 1:
        zt = z.broadcast_like(profile[variables[0]]) if other else z
    else:
        zt = z
    zt = zt.transpose(*other, "height")
    shape = tuple(profile.sizes[d] for d in other)
    ncol = int(np.prod(shape)) if other else 1
    zarr = _f64(zt.values.reshape(ncol, -1))
    vals = np.stack(
        [
            profile[n]
            .broadcast_like(zt)
            .transpose(*other, "height")
            .values.reshape(ncol, -1)
            for n in variables
        ]
    )
    return zarr, _f64(vals), other, shape


def interpolate_profile(
    profile,
    heights,
    variables=None,
    *,
    extrapolate=False,
    engine="auto",
    n_threads=None,
):
    """
    Interpolate a profile to arbitrary heights.

    Values are interpolated linearly in height between the nearest valid
    levels, pressure linearly in its logarithm; wind speed and direction are
    recomputed from the interpolated ``u`` and ``v``.

    Parameters
    ----------
    profile : xarray.Dataset
        Profile on ``height`` (m above sea level), e.g. from
        :func:`read_sounding` or :func:`era5_profile`. Profiles with extra
        dimensions (columns) are interpolated column by column; ``heights``
        must then share those dimensions.
    heights : xarray.DataArray, xarray.Dataset or array-like
        Target heights in m above sea level, of any shape: a sweep's gate
        heights ``ds["z"]``, QVP heights, or a grid's ``z``. A Dataset
        contributes its ``z``.
    variables : list of str, optional
        Variables to interpolate. Default: all profile variables.
    extrapolate : bool, optional
        Hold the lowest/highest valid value beyond the profile. Default
        False (NaN outside).
    engine : {"auto", "compiled", "numpy"}, optional
        Kernel implementation.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.

    Returns
    -------
    xarray.Dataset
        The variables on the dimensions and coordinates of ``heights``.

    Examples
    --------
    >>> sweep = dtree["sweep_0"].to_dataset()  # doctest: +SKIP
    >>> env = interpolate_profile(profile, sweep["z"])  # doctest: +SKIP
    """
    if isinstance(heights, xr.Dataset):
        heights = heights["z"]
    if not isinstance(heights, xr.DataArray):
        heights = xr.DataArray(np.asarray(heights, dtype=np.float64))
    if variables is None:
        variables = [
            n
            for n in profile.data_vars
            if "height" in profile[n].dims and n not in ("wind_speed", "wind_direction")
        ]
    else:
        variables = [
            n for n in variables if n not in ("wind_speed", "wind_direction")
        ] + (
            ["u", "v"]
            if {"wind_speed", "wind_direction"} & set(variables)
            and not {"u", "v"} <= set(variables)
            else []
        )
    zcols, vals, other, shape = _profile_columns(profile, variables)
    hdims = [d for d in heights.dims if d not in other]
    missing = [d for d in other if d not in heights.dims]
    if missing:
        heights = heights.expand_dims({d: profile.sizes[d] for d in missing})
    ht = heights.transpose(*other, *hdims)
    ncol = zcols.shape[0]
    target = _f64(ht.values.reshape(ncol, -1))
    log_var = np.array([n == "pressure" for n in variables], dtype=bool)
    k = _kernel(engine)
    res = np.asarray(
        k.interp_vertical(
            zcols,
            vals,
            target,
            log_var,
            extrapolate=extrapolate,
            n_threads=int(n_threads or 0),
        )
    )
    out_shape = tuple(ht.shape)
    coords = {name: c for name, c in heights.coords.items()}
    data_vars = {
        n: (ht.dims, res[i].reshape(out_shape), dict(profile[n].attrs))
        for i, n in enumerate(variables)
    }
    out = xr.Dataset(data_vars, coords=coords)
    if "u" in out and "v" in out:
        out["wind_speed"] = np.hypot(out["u"], out["v"]).assign_attrs(
            ATTRS["wind_speed"]
        )
        out["wind_direction"] = (
            np.mod(np.degrees(np.arctan2(-out["u"], -out["v"])), 360.0)
        ).assign_attrs(ATTRS["wind_direction"])
    out = out.transpose(*heights.dims)
    out.attrs = dict(profile.attrs)
    return out


def isotherm_height(
    profile,
    temperature=T0,
    *,
    variable="temperature",
    which="highest",
    engine="auto",
    n_threads=None,
):
    """
    Height at which the temperature falls through an isotherm.

    The crossing is located between the two valid levels that bracket it and
    interpolated linearly in height.

    Parameters
    ----------
    profile : xarray.Dataset
        Profile on ``height``; extra dimensions are treated as columns.
    temperature : float, optional
        Isotherm in K. Default 273.15 (0 degC); e.g. 263.15 and 253.15 for
        -10 and -20 degC.
    variable : str, optional
        Temperature variable. Default ``"temperature"``.
    which : {"highest", "lowest"}, optional
        With several crossings (inversions, warm noses), the top one (the top
        of the highest warm layer, default) or the lowest one.
    engine, n_threads : optional
        Kernel implementation and threads.

    Returns
    -------
    xarray.DataArray
        Height above sea level in m; NaN where the profile does not cross the
        isotherm (e.g. below freezing at all levels).
    """
    if which not in ("highest", "lowest"):
        raise ValueError("which must be 'highest' or 'lowest'")
    zcols, vals, other, shape = _profile_columns(profile, [variable])
    k = _kernel(engine)
    res = np.asarray(
        k.level_crossing(
            zcols,
            _f64(vals[0]),
            float(temperature),
            which == "highest",
            n_threads=int(n_threads or 0),
        )
    ).reshape(shape)
    coords = {
        c: profile.coords[c]
        for c in profile.coords
        if set(profile.coords[c].dims) <= set(other)
    }
    return xr.DataArray(
        res,
        dims=other,
        coords=coords,
        name="isotherm_height",
        attrs={
            "long_name": f"height of the {temperature - T0:g} degC isotherm ({which} crossing)",
            "units": "m",
            "isotherm": float(temperature),
        },
    )


def wet_bulb_zero_height(profile, *, which="highest", engine="auto", n_threads=None):
    """
    Height of the 0 degC wet-bulb temperature.

    The isobaric wet-bulb temperature (see :func:`wet_bulb_temperature`) is
    computed at every level from ``pressure``, ``temperature`` and
    ``dewpoint`` and its 0 degC crossing located as in
    :func:`isotherm_height`. This is the isobaric wet-bulb temperature, not
    the pseudo-adiabatic one of Davies-Jones (2008); the choice of the top
    crossing is a radarx convention.

    Parameters
    ----------
    profile : xarray.Dataset
        Profile on ``height``; extra dimensions are treated as columns.
    which : {"highest", "lowest"}, optional
        Crossing to return. Default the top one.
    engine, n_threads : optional
        Kernel implementation and threads.

    Returns
    -------
    xarray.DataArray
        Height above sea level in m (NaN without a crossing).

    References
    ----------
    Bolton, D., 1980: The computation of equivalent potential temperature.
    Mon. Wea. Rev., 108, 1046-1053,
    https://doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2

    Davies-Jones, R., 2008: An efficient and accurate method for computing the
    wet-bulb temperature along pseudoadiabats. Mon. Wea. Rev., 136,
    2764-2785, https://doi.org/10.1175/2007MWR2224.1
    """
    tw = wet_bulb_temperature(
        profile["pressure"],
        profile["temperature"],
        profile["dewpoint"],
        engine=engine,
        n_threads=n_threads,
    )
    out = isotherm_height(
        profile.assign(wet_bulb_temperature=tw),
        T0,
        variable="wet_bulb_temperature",
        which=which,
        engine=engine,
        n_threads=n_threads,
    )
    return out.rename("wet_bulb_zero_height").assign_attrs(
        ATTRS["wet_bulb_zero_height"]
    )


def mean_wind(
    profile, bottom, top, *, above_ground=False, engine="auto", n_threads=None
):
    """
    Layer-mean wind.

    The height-weighted mean of ``u`` and ``v`` over ``[bottom, top]``,
    integrating the profile linearly interpolated between valid levels
    (trapezoidal rule).

    Parameters
    ----------
    profile : xarray.Dataset
        Profile on ``height`` with ``u`` and ``v``.
    bottom, top : float
        Layer limits in m above sea level, or above the lowest valid level
        when ``above_ground=True`` (e.g. 0 and 6000 for the 0-6 km mean wind).
    above_ground : bool, optional
        Interpret the limits relative to the lowest valid wind level.
    engine, n_threads : optional
        Kernel implementation and threads.

    Returns
    -------
    xarray.Dataset
        ``u``, ``v``, ``wind_speed`` and ``wind_direction`` of the mean wind
        vector; NaN if valid winds do not span the layer.
    """
    zcols, vals, other, shape = _profile_columns(profile, ["u", "v"])
    k = _kernel(engine)
    if above_ground:
        ok = np.isfinite(vals[0]) & np.isfinite(zcols)
        ground = np.where(
            ok.any(axis=1), np.nanmin(np.where(ok, zcols, np.inf), axis=1), np.nan
        )
        res = np.full((2, zcols.shape[0]), np.nan)
        for c in range(zcols.shape[0]):  # usually one column
            if np.isfinite(ground[c]):
                res[:, c] = np.asarray(
                    k.layer_mean(
                        zcols[c : c + 1],
                        _f64(vals[:, c : c + 1]),
                        float(bottom + ground[c]),
                        float(top + ground[c]),
                        n_threads=int(n_threads or 0),
                    )
                )[:, 0]
    else:
        res = np.asarray(
            k.layer_mean(
                zcols, vals, float(bottom), float(top), n_threads=int(n_threads or 0)
            )
        )
    u, v = (r.reshape(shape) for r in res)
    coords = {
        c: profile.coords[c]
        for c in profile.coords
        if set(profile.coords[c].dims) <= set(other)
    }
    out = xr.Dataset(
        {
            "u": (other, u, dict(ATTRS["u"])),
            "v": (other, v, dict(ATTRS["v"])),
            "wind_speed": (other, np.hypot(u, v), dict(ATTRS["wind_speed"])),
            "wind_direction": (
                other,
                np.mod(np.degrees(np.arctan2(-u, -v)), 360.0),
                dict(ATTRS["wind_direction"]),
            ),
        },
        coords=coords,
    )
    ref = "above the lowest wind level" if above_ground else "above sea level"
    out.attrs = {"layer": f"{bottom:g}-{top:g} m {ref}", "cell_methods": "height: mean"}
    return out


# ---------------------------------------------------------------------------
# backgrounds on a radarx grid


def _grid_crs(grid):
    import pyproj

    if "crs_wkt" in grid.coords or "crs_wkt" in grid:
        return pyproj.CRS.from_cf(grid["crs_wkt"].attrs)
    lat0 = float(grid["latitude"])
    lon0 = float(grid["longitude"])
    return pyproj.CRS.from_dict(
        {"proj": "aeqd", "lat_0": lat0, "lon_0": lon0, "datum": "WGS84"}
    )


def _grid_lonlat_rotation(grid):
    """Cell longitude, latitude and the direction of true north from grid +y (deg)."""
    import pyproj

    crs = _grid_crs(grid)
    to_geo = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    to_grid = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    X, Y = np.meshgrid(
        grid["x"].values.astype(np.float64), grid["y"].values.astype(np.float64)
    )
    lon, lat = to_geo.transform(X, Y)
    dlat = 1e-4
    xn, yn = to_grid.transform(lon, np.minimum(lat + dlat, 90.0))
    angle = np.degrees(np.arctan2(xn - X, yn - Y))
    return np.asarray(lon), np.asarray(lat), angle


def _grid_time(grid):
    if "time" in grid.coords or "time" in grid:
        t = grid["time"].values
        return np.asarray(t).ravel()[0]
    raise ValueError("pass time=..., the grid has no 'time'")


def _background(zcols, cols, grid, attrs, engine, n_threads):
    """
    Common assembly of a grid background from columns.

    zcols: (ncol, nlev) with ncol == 1 (horizontally uniform) or one column per
    grid cell (row-major y, x); cols: dict of (ncol, nlev) arrays.
    """
    k = _kernel(engine)
    nt = int(n_threads or 0)
    z = grid["z"].values.astype(np.float64)
    ny, nx = grid.sizes["y"], grid.sizes["x"]
    ncell = ny * nx
    ncol = zcols.shape[0]
    names = ["pressure", "temperature", "specific_humidity", "u", "v", "w"]
    vals = _f64(np.stack([cols[n] for n in names]))
    target = _f64(np.broadcast_to(z, (ncol, z.size)))
    log_var = np.array([n == "pressure" for n in names], dtype=bool)
    res = np.asarray(k.interp_vertical(_f64(zcols), vals, target, log_var, False, nt))
    res = np.broadcast_to(res, (len(names), ncell, z.size)) if ncol == 1 else res
    out = {n: res[i] for i, n in enumerate(names)}  # (ncell, nz)

    lon, lat, angle = _grid_lonlat_rotation(grid)
    ang = np.broadcast_to(angle.reshape(ncell, 1), (ncell, z.size))
    rot = np.asarray(
        k.rotate_wind(
            _f64(out["u"]).ravel(), _f64(out["v"]).ravel(), _f64(ang).ravel(), nt
        )
    )
    out["u"], out["v"] = rot[0].reshape(ncell, z.size), rot[1].reshape(ncell, z.size)

    e = _thermo(
        "vapor_pressure",
        out["specific_humidity"],
        out["pressure"],
        engine=engine,
        n_threads=n_threads,
    )
    out["dewpoint"] = _thermo("dewpoint", e, engine=engine, n_threads=n_threads)
    out["relative_humidity"] = e / _thermo(
        "esat", out["temperature"], engine=engine, n_threads=n_threads
    )
    out["air_density"] = _thermo(
        "density",
        out["pressure"],
        out["temperature"],
        out["specific_humidity"],
        engine=engine,
        n_threads=n_threads,
    )

    tcol = _f64(cols["temperature"])
    frz = np.asarray(k.level_crossing(_f64(zcols), tcol, T0, True, nt))
    ecol = _thermo(
        "vapor_pressure",
        cols["specific_humidity"],
        cols["pressure"],
        engine=engine,
        n_threads=n_threads,
    )
    tdcol = _thermo("dewpoint", ecol, engine=engine, n_threads=n_threads)
    twcol = _thermo(
        "wet_bulb", cols["pressure"], tcol, tdcol, engine=engine, n_threads=n_threads
    )
    wbz = np.asarray(k.level_crossing(_f64(zcols), _f64(twcol), T0, True, nt))
    frz = np.broadcast_to(frz, (ncell,)) if ncol == 1 else frz
    wbz = np.broadcast_to(wbz, (ncell,)) if ncol == 1 else wbz

    def cube(a):
        return np.ascontiguousarray(
            np.moveaxis(a.reshape(ny, nx, z.size), -1, 0), dtype=np.float32
        )

    data_vars = {}
    for name in [
        "u",
        "v",
        "w",
        "temperature",
        "pressure",
        "specific_humidity",
        "dewpoint",
        "relative_humidity",
        "air_density",
    ]:
        a = dict(GRID_WIND_ATTRS.get(name, ATTRS[name]))
        data_vars[name] = (("z", "y", "x"), cube(out[name]), a)
    data_vars["u"][2][
        "comment"
    ] = "earth-relative wind rotated to the grid axes using wind_rotation"
    data_vars["freezing_level"] = (
        ("y", "x"),
        frz.reshape(ny, nx),
        dict(ATTRS["freezing_level"]),
    )
    data_vars["wet_bulb_zero_height"] = (
        ("y", "x"),
        wbz.reshape(ny, nx),
        dict(ATTRS["wet_bulb_zero_height"]),
    )
    data_vars["wind_rotation"] = (
        ("y", "x"),
        angle,
        dict(GRID_WIND_ATTRS["wind_rotation"]),
    )
    keep = {
        c: grid.coords[c]
        for c in grid.coords
        if set(grid.coords[c].dims) <= {"z", "y", "x"}
    }
    bg = xr.Dataset(data_vars, coords=keep)
    bg.attrs = dict(attrs)
    bg.attrs["Conventions"] = "CF-1.8"
    return bg


def era5_column(
    grid,
    time=None,
    *,
    source="auto",
    time_interpolation="linear",
    cache=True,
    engine="auto",
    n_threads=None,
):
    """
    ERA5 background on a radarx grid.

    ERA5 columns are interpolated bilinearly from the 0.25 degree grid to every
    grid cell, linearly in time between the two ERA5 times bracketing
    ``time``, and linearly in height (pressure in log) to the grid levels.
    Earth-relative winds are rotated to the grid axes.

    Parameters
    ----------
    grid : xarray.Dataset
        A radarx grid (``z``, ``y``, ``x`` in m and ``crs_wkt``), e.g. from
        :func:`radarx.grid.grid_radar` or ``dtree.radarx.to_grid()``.
    time : str or datetime-like, optional
        Valid time. Default: the grid's ``time``.
    source : {"auto", "arco", "cds", "gcs"}, optional
        ERA5 provider. ``"auto"`` (default) uses the full CDS dataset when CDS
        credentials are configured, otherwise the Google ARCO-ERA5 store.
    time_interpolation : {"linear", "nearest"}, optional
        Time interpolation. Default ``"linear"``.
    cache : bool, optional
        Reuse cached downloads. Default True.
    engine : {"auto", "compiled", "numpy"}, optional
        Kernel implementation.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.

    Returns
    -------
    xarray.Dataset
        On the grid's ``(z, y, x)``: ``u``, ``v`` (components along the grid
        ``x`` and ``y`` axes), ``w`` (:math:`-\\omega / (\\rho g_0)`),
        ``temperature``, ``pressure``, ``specific_humidity``, ``dewpoint``,
        ``relative_humidity`` and ``air_density`` (:func:`air_density`), and
        on ``(y, x)``: ``freezing_level``, ``wet_bulb_zero_height`` (top
        crossings, m above sea level) and ``wind_rotation`` (direction of
        true north clockwise from the grid ``y`` axis, degrees; grid
        components are :math:`u_x = u \\cos\\alpha + v \\sin\\alpha`,
        :math:`v_y = -u \\sin\\alpha + v \\cos\\alpha`). Levels outside the
        ERA5 column are NaN.

    References
    ----------
    Hersbach, H., and Coauthors, 2020: The ERA5 global reanalysis.
    Quart. J. Roy. Meteor. Soc., 146, 1999-2049,
    https://doi.org/10.1002/qj.3803
    """
    time = _grid_time(grid) if time is None else time
    lon, lat, _ = _grid_lonlat_rotation(grid)
    z, cols, meta = _era5_columns(
        source,
        lat.ravel(),
        lon.ravel(),
        time,
        time_interpolation,
        cache,
        engine,
        n_threads,
    )
    meta["valid_time"] = str(_to_datetime64(time))
    return _background(z, cols, grid, meta, engine, n_threads)


def profile_to_grid(profile, grid, *, engine="auto", n_threads=None):
    """
    Horizontally uniform background on a radarx grid from one profile.

    Gives the same variables as :func:`era5_column`, so a sounding and ERA5
    can be used interchangeably. The profile's earth-relative wind is rotated
    to the grid axes at every cell.

    Parameters
    ----------
    profile : xarray.Dataset
        Profile on ``height`` (e.g. from :func:`read_sounding`).
    grid : xarray.Dataset
        A radarx grid (``z``, ``y``, ``x`` and ``crs_wkt``).
    engine, n_threads : optional
        Kernel implementation and threads.

    Returns
    -------
    xarray.Dataset
        See :func:`era5_column`.
    """
    zc = profile["height"].values.astype(np.float64)[None, :]

    def col(name):
        if name in profile:
            return profile[name].values.astype(np.float64)[None, :]
        return np.full_like(zc, np.nan)

    cols = {
        n: col(n)
        for n in ["pressure", "temperature", "specific_humidity", "u", "v", "w"]
    }
    attrs = {k: v for k, v in profile.attrs.items() if k not in ("featureType",)}
    return _background(zc, cols, grid, attrs, engine, n_threads)


# ---------------------------------------------------------------------------
# helpers for the accessors


def _synoptic_time(time, hours=12):
    """Nearest 00/12 UTC (``hours`` apart) to ``time``."""
    t = _to_datetime64(time)
    step = np.timedelta64(hours, "h")
    day = t.astype("datetime64[D]").astype("datetime64[s]")
    k = np.round((t - day) / step)
    return day + int(k) * step


def _volume_site_time(dtree):
    """Radar latitude, longitude, altitude and the volume start time."""
    root = dtree.root.to_dataset()
    names = sorted(n for n in dtree.children if n.startswith("sweep"))
    first = None
    if names:
        try:
            first = dtree[names[0]].to_dataset(inherit="all_coords")
        except (TypeError, ValueError):  # pragma: no cover - older xarray
            first = dtree[names[0]].to_dataset()
    site = {}
    for key in ("latitude", "longitude", "altitude"):
        for src in (first, root):
            if src is not None and key in src:
                site[key] = float(src[key].values)
                break
    if "latitude" not in site or "longitude" not in site:
        raise ValueError("the volume needs the radar 'latitude' and 'longitude'")
    time = None
    if first is not None and "time" in first:
        time = np.asarray(first["time"].values).min()
    elif "time_coverage_start" in root:
        time = root["time_coverage_start"].values
    elif "time_coverage_start" in dtree.attrs:
        time = dtree.attrs["time_coverage_start"]
    if time is None:
        raise ValueError("cannot find the volume time; pass time=...")
    return site, _to_datetime64(time)


def _sounding_for_volume(dtree, kind="era5", station=None, time=None, **kwargs):
    """Profile for a radar volume: ERA5 at the site or the nearest sounding."""
    site, vtime = _volume_site_time(dtree)
    time = vtime if time is None else _to_datetime64(time)
    if kind == "era5":
        prof = era5_profile(site["latitude"], site["longitude"], time, **kwargs)
    elif kind in _ID_COLUMN:
        if station is None:
            near = nearest_station(
                site["latitude"], site["longitude"], time, source=kind
            )
            station = str(near[_ID_COLUMN[kind]].values[0])
        prof = read_sounding(station, _synoptic_time(time), source=kind, **kwargs)
    else:
        raise ValueError(
            f"unknown source {kind!r}; use 'era5', 'iem', 'uwyo' or 'igra2'"
        )
    prof.attrs["radar_latitude"] = site["latitude"]
    prof.attrs["radar_longitude"] = site["longitude"]
    if "altitude" in site:
        prof.attrs["radar_altitude"] = site["altitude"]
    prof.attrs["radar_time"] = str(time)
    return prof
