# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Parcel Thermodynamics: CAPE, CIN, Lifted Index, LCL, LFC and EL
===============================================================

Diagnostics of a lifted air parcel for soundings and ERA5 columns
(:func:`radarx.io.sounding.read_sounding`, :func:`radarx.io.sounding.era5_profile`,
:func:`radarx.io.sounding.era5_column`) or any Dataset with pressure,
temperature and humidity on a vertical dimension. All columns (times, grid
points) are computed in one call of a compiled C++ kernel
(``radarx.io._parcel``, multithreaded over columns) with an identical NumPy
implementation as the fallback and test oracle.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   parcel_diagnostics

Definitions
-----------
The thermodynamics are those of Bolton (1980) [1]_, in the forms used by
software that cites the paper (the equations were not checked against its
text): the saturation vapour pressure over liquid water :math:`e_s(T)` (eq. 10),
the potential temperature
:math:`\\theta = T (1000 / p)^{0.2854 (1 - 0.28 r)}` (eq. 6, :math:`p` in
hPa, :math:`r` the mixing ratio in kg kg-1), the temperature of the lifting
condensation level

.. math::

   T_L = \\frac{1}{1/(T_D - 56) + \\ln(T/T_D)/800} + 56 \\quad\\text{(eq. 15)}

and the equivalent potential temperature

.. math::

   \\theta_E = T \\left(\\frac{1000}{p}\\right)^{0.2854 (1 - 0.28 r)}
   \\exp\\left[r (1 + 0.81 r) \\left(\\frac{3376}{T_L} - 2.54\\right)\\right]
   \\quad\\text{(eq. 43)}.

A parcel rises along the dry adiabat of eq. (6), with the mixing ratio of its
start, up to the lifting condensation level, whose pressure follows from the
conservation of :math:`\\theta` between the start and :math:`T_L`; above it the
parcel is saturated and keeps :math:`\\theta_E` (eq. 43 with :math:`T_L = T`
and :math:`r = r_s(T, p)`), which is the pseudo-adiabat of Bolton: the
temperature at each pressure is found by bisection. The ratio of the gas
constants is :math:`\\epsilon` = Rd/Rv with the radarx values of
:mod:`radarx.io.sounding`.

The parcel is lifted from

- ``"surface-based"``: the lowest level;
- ``"mixed-layer"``: the lowest level with the pressure-weighted mean of the
  potential temperature and the mixing ratio of the lowest 100 hPa
  (``depth``);
- ``"most-unstable"``: the level of the largest :math:`\\theta_E` in the lowest
  300 hPa (``depth``).

These are the usual definitions; the layer depths are the customary values
(not from Bolton 1980).

The buoyancy is :math:`B = g (T_{v,p} - T_{v,e}) / T_{v,e}` with the virtual
temperature :math:`T_v = T (1 + r/\\epsilon)/(1 + r)` of the parcel and the
environment (the virtual temperature correction of Doswell and Rasmussen
(1994) [2]_; ``virtual=False`` leaves it out, which lowers the CAPE of the
test soundings of radarx by 6 and 7 %). The parcel carries no condensate
(pseudo-adiabatic). :math:`B` is linear in height between the levels, the
levels of the profile and the LCL are the nodes, and the zero crossings are
located exactly.

- ``cape`` = :math:`\\int B\\, dz` over the positive buoyancy between the LFC and
  the EL (J kg-1);
- ``cin`` = :math:`\\int B\\, dz` over the negative buoyancy from the start of
  the parcel to the LFC (J kg-1, zero or negative);
- ``lcl_*``: lifting condensation level;
- ``lfc_*``: the first level at or above the LCL where the parcel is warmer
  than the environment;
- ``el_*``: the highest level where the buoyancy falls to zero above the LFC;
  if the parcel is still buoyant at the top of the profile that top (the CAPE
  is then cut off there);
- ``lifted_index`` = environment temperature minus parcel temperature at
  500 hPa (K, Galway 1956 [3]_; actual temperatures, not virtual), negative
  for unstable air.

Without an LFC (the parcel never becomes buoyant, or the LCL lies above the
profile) ``cape`` is 0 and ``cin``, ``lfc_*`` and ``el_*`` are NaN. Heights are
those of the profile (m above sea level for the sounding and ERA5 readers of
:mod:`radarx.io.sounding`).

References
----------
.. [1] Bolton, D., 1980: The computation of equivalent potential temperature.
   *Mon. Wea. Rev.*, **108** (7), 1046-1053,
   https://doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2
.. [2] Doswell, C. A., and E. N. Rasmussen, 1994: The effect of neglecting the
   virtual temperature correction on CAPE calculations. *Wea. Forecasting*,
   **9** (4), 625-629,
   https://doi.org/10.1175/1520-0434(1994)009<0625:TEONTV>2.0.CO;2
.. [3] Galway, J. G., 1956: The lifted index as a predictor of latent
   instability. *Bull. Amer. Meteor. Soc.*, **37** (10), 528-529,
   https://doi.org/10.1175/1520-0477-37.10.528
"""

from __future__ import annotations

__all__ = ["parcel_diagnostics"]

import numpy as np
import xarray as xr

from .._provenance import provenance
from .._registry import accessor_method
from . import _parcel_numpy
from .sounding import (
    dewpoint_from_specific_humidity,
    dewpoint_from_vapor_pressure,
    saturation_vapor_pressure,
)

try:
    from . import _parcel

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _parcel = None
    HAS_COMPILED_KERNEL = False

PARCELS = {"surface-based": 0, "mixed-layer": 1, "most-unstable": 2}
DEFAULT_DEPTH = {"mixed-layer": 10000.0, "most-unstable": 30000.0}  # Pa

_ATTRS = {
    "cape": (
        "atmosphere_convective_available_potential_energy_wrt_surface",
        "J kg-1",
        "Convective available potential energy",
    ),
    "cin": (
        "atmosphere_convective_inhibition_wrt_surface",
        "J kg-1",
        "Convective inhibition (zero or negative)",
    ),
    "lifted_index": (
        None,
        "K",
        "Lifted index: environment minus parcel temperature at 500 hPa",
    ),
    "lcl_pressure": (None, "Pa", "Pressure of the lifting condensation level"),
    "lcl_height": (None, "m", "Height of the lifting condensation level"),
    "lfc_pressure": (None, "Pa", "Pressure of the level of free convection"),
    "lfc_height": (None, "m", "Height of the level of free convection"),
    "el_pressure": (None, "Pa", "Pressure of the equilibrium level"),
    "el_height": (None, "m", "Height of the equilibrium level"),
    "parcel_pressure": (None, "Pa", "Pressure at which the parcel starts"),
    "parcel_temperature": (None, "K", "Temperature at which the parcel starts"),
    "parcel_dewpoint": (None, "K", "Dew point at which the parcel starts"),
}


def _kernel(engine):
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled parcel kernel is not available")
    return _parcel if HAS_COMPILED_KERNEL and engine != "numpy" else _parcel_numpy


def _f64(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def _dewpoint(profile, pressure, temperature, dewpoint, engine, n_threads):
    """Dew point (K) of the profile from the first humidity variable found."""
    if dewpoint in profile:
        return profile[dewpoint]
    if "specific_humidity" in profile:
        return dewpoint_from_specific_humidity(
            profile["specific_humidity"],
            profile[pressure],
            engine=engine,
            n_threads=n_threads,
        )
    if "relative_humidity" in profile:
        rh = profile["relative_humidity"]
        if float(rh.max()) > 1.5:
            rh = rh / 100.0
        e = rh * saturation_vapor_pressure(
            profile[temperature], engine=engine, n_threads=n_threads
        )
        return dewpoint_from_vapor_pressure(e, engine=engine, n_threads=n_threads)
    raise ValueError(
        f"the profile needs {dewpoint!r}, 'specific_humidity' or 'relative_humidity'"
    )


@provenance(
    "Parcel CAPE, CIN, LCL, LFC, EL and lifted index, thermodynamics of Bolton (1980)"
)
def parcel_diagnostics(
    profile,
    parcel="surface-based",
    *,
    depth=None,
    virtual=True,
    dim="height",
    height=None,
    pressure="pressure",
    temperature="temperature",
    dewpoint="dewpoint",
    engine="auto",
    n_threads=None,
):
    """
    CAPE, CIN, lifted index, LCL, LFC and EL of a lifted parcel.

    See the module description for the definitions.

    Parameters
    ----------
    profile : xarray.Dataset
        ``pressure`` (Pa), ``temperature`` (K) and ``dewpoint`` (K), or
        ``specific_humidity`` (kg kg-1), or ``relative_humidity`` (fraction),
        on the vertical dimension ``dim`` with a height coordinate; other
        dimensions (time, ``y``, ``x``) are separate columns. Levels must
        rise (pressure falls) with height; levels that are not are skipped.
    parcel : {"surface-based", "mixed-layer", "most-unstable"}, optional
        The parcel to lift. Default ``"surface-based"``.
    depth : float, optional
        Depth (Pa) of the layer of the mixed-layer mean or of the search for
        the most unstable parcel. Default 10000 (100 hPa) and 30000 (300 hPa).
    virtual : bool, optional
        Use the virtual temperature in the buoyancy. Default True.
    dim : str, optional
        Vertical dimension. Default ``"height"`` (``"z"`` for ERA5 grids).
    height : str or xarray.DataArray, optional
        Heights (m) if not the ``dim`` coordinate.
    pressure, temperature, dewpoint : str, optional
        Variable names.
    engine : {"auto", "compiled", "numpy"}, optional
        Kernel implementation.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.

    Returns
    -------
    xarray.Dataset
        ``cape``, ``cin`` (J kg-1), ``lifted_index`` (K), ``lcl_pressure``,
        ``lfc_pressure``, ``el_pressure`` (Pa), ``lcl_height``, ``lfc_height``,
        ``el_height`` (m), and the start of the parcel: ``parcel_pressure``,
        ``parcel_temperature``, ``parcel_dewpoint``, on the dimensions of the
        profile without ``dim``.

    Raises
    ------
    ValueError
        For an unknown parcel, a missing variable or pressures not in Pa.

    References
    ----------
    .. [1] Bolton, D., 1980: The computation of equivalent potential
       temperature. *Mon. Wea. Rev.*, **108** (7), 1046-1053,
       https://doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2
    .. [2] Doswell, C. A., and E. N. Rasmussen, 1994: The effect of neglecting
       the virtual temperature correction on CAPE calculations. *Wea.
       Forecasting*, **9** (4), 625-629,
       https://doi.org/10.1175/1520-0434(1994)009<0625:TEONTV>2.0.CO;2

    Examples
    --------
    >>> prof = radarx.io.era5_profile(33.9, -88.3, "2022-03-31T00:00")  # doctest: +SKIP
    >>> radarx.io.parcel_diagnostics(prof, "most-unstable").cape  # doctest: +SKIP
    """
    if parcel not in PARCELS:
        raise ValueError(f"parcel must be one of {tuple(PARCELS)}, not {parcel!r}")
    for name in (pressure, temperature):
        if name not in profile:
            raise ValueError(f"the profile needs {name!r}")
    if depth is None:
        depth = DEFAULT_DEPTH.get(parcel, 10000.0)
    if not depth > 0:
        raise ValueError("depth must be positive")
    p = profile[pressure]
    if float(p.max()) < 2000.0:
        raise ValueError("pressure must be in Pa (the largest value is below 2000)")
    td = _dewpoint(profile, pressure, temperature, dewpoint, engine, n_threads)
    if height is not None:
        z = height if isinstance(height, xr.DataArray) else profile[height]
    elif dim in profile.coords:
        z = profile[dim]
    else:
        raise ValueError(f"no coordinate {dim!r} to use as height; pass height=")
    arrays = xr.broadcast(p, profile[temperature], td, z)
    if dim not in arrays[0].dims:
        raise ValueError(f"dimension {dim!r} not found in {arrays[0].dims}")
    other = [d for d in arrays[0].dims if d != dim]
    nz = arrays[0].sizes[dim]
    shape = tuple(arrays[0].sizes[d] for d in other)
    cols = [
        _f64(a.transpose(*other, dim).values.reshape(-1, nz)) for a in arrays
    ]  # pressure, temperature, dew point, height
    pc, tc, dc, zc = cols
    # heights must increase along the column
    if nz > 1:
        first = np.nanmedian(np.diff(zc, axis=1))
        if np.isfinite(first) and first < 0:
            pc, tc, dc, zc = (c[:, ::-1] for c in (pc, tc, dc, zc))
            pc, tc, dc, zc = (_f64(c) for c in (pc, tc, dc, zc))
    res = np.asarray(
        _kernel(engine).parcel(
            zc,
            pc,
            tc,
            dc,
            PARCELS[parcel],
            float(depth),
            bool(virtual),
            int(n_threads or 0),
        )
    )
    coords = {
        k: v
        for k, v in arrays[0].coords.items()
        if dim not in v.dims and set(v.dims) <= set(other)
    }
    data = {}
    for row, name in enumerate(_parcel_numpy.ROWS):
        standard_name, units, long_name = _ATTRS[name]
        attrs = {"long_name": long_name, "units": units}
        if standard_name:
            attrs["standard_name"] = standard_name
        data[name] = (other, res[row].reshape(shape), attrs)
    out = xr.Dataset(data, coords=coords)
    out.attrs.update(
        parcel=parcel,
        parcel_layer_depth=float(depth) if parcel != "surface-based" else 0.0,
        virtual_temperature_correction=int(bool(virtual)),
    )
    return out


@accessor_method("dataset", name="parcel_diagnostics")
def _parcel_accessor(self, parcel="surface-based", **kwargs):
    """
    CAPE, CIN, lifted index, LCL, LFC and EL of a profile or ERA5 columns.

    Parameters
    ----------
    parcel : {"surface-based", "mixed-layer", "most-unstable"}, optional
        The parcel to lift.
    **kwargs
        Options of :func:`radarx.io.parcel.parcel_diagnostics`.

    Returns
    -------
    xarray.Dataset
        See :func:`radarx.io.parcel.parcel_diagnostics`.
    """
    return parcel_diagnostics(self.xarray_obj, parcel, **kwargs)
