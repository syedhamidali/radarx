#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Target Classification of Cloud Radar Echoes
===========================================

What a Ka or W band cloud radar echo is made of (cloud droplets, drizzle or rain,
ice, melting ice, insects) from the reflectivity, the temperature of the air and,
where available, the linear depolarization ratio and the dual-frequency ratio.

The categories follow the target categorization of Cloudnet (Illingworth et
al. 2007), which combines a cloud radar, a lidar and a model: the phase
of an echo is told from the temperature, the melting layer is the layer in which
the temperature passes 0 degrees, and drizzle is the echo of large liquid
droplets. Cloudnet needs a lidar to find the cloud droplets and the supercooled
liquid layers, which a radar alone does not see; this classification uses the
radar only, so it has the classes a radar can tell, and no class for supercooled
liquid (it is ice here), for aerosol, or for a mixture of droplets and drizzle.
CloudnetPy (Tukiainen et al. 2020) is the open implementation of Cloudnet.

========  ===================================================================
Code      Class
========  ===================================================================
0         no echo
1         cloud droplets: liquid water (warm), reflectivity below ``min_drizzle_dbz``
2         drizzle or rain: liquid water, reflectivity at or above ``min_drizzle_dbz``
3         ice (below the melting zone, cold)
4         melting ice (within the melting zone of temperature)
5         insects (needs the depolarization ratio)
6         large ice, snow aggregates (ice with a dual-frequency ratio of at least ``min_large_dfr``)
7         echo of unknown phase (no temperature)
========  ===================================================================

Rules
-----
The rules are applied to every gate in this order, with the temperature :math:`T`
in degrees Celsius.

1. A gate without a reflectivity, or below ``min_dbz``, has no echo.
2. Without a temperature: unknown phase.
3. :math:`T` in the melting zone (``melting_temperature``, default 0 to 2 degrees
   Celsius): melting ice.
4. :math:`T` above the melting zone (liquid water): insects if the depolarization
   ratio is at least ``min_insect_ldr`` and the reflectivity is below
   ``max_insect_dbz``; else drizzle or rain at or above ``min_drizzle_dbz``; else cloud droplets.
5. :math:`T` below the melting zone (ice): large ice if the dual-frequency ratio
   is at least ``min_large_dfr``; else ice.

Thresholds are radarx choices, and not values of the papers:

* melting zone of 0 to 2 degrees Celsius. Cloudnet places the melting layer from the
  wet-bulb temperature; pass the wet-bulb temperature as ``temperature`` to do
  the same. A melting layer of about 400 m has a temperature range of 2 to 3
  degrees Celsius.
* ``min_drizzle_dbz`` of -20 dBZ, between the reflectivity of cloud droplets at W band and
  that of drizzle (at most about -20 dBZ for droplets of 10-20 micrometre radius,
  and drizzle of 50 micrometre radius or more above it).
* ``min_insect_ldr`` of -20 dB and ``max_insect_dbz`` of 10 dBZ: insects
  depolarize more strongly than droplets, drizzle and rain (below -25 dB).
* ``min_large_dfr`` of 3 dB, where the dual-frequency ratio of
  :mod:`radarx.retrieve.dual_frequency` of ice shows the Mie effects of
  snow aggregates (particle sizes of about 1.5 mm and larger for Ka and W band).

A radar without a lidar cannot tell supercooled liquid in a cold cloud from
ice, and cloud droplets from the weak drizzle of a deep stratocumulus. Use the
result as a first look at the echoes, not as a replacement for Cloudnet.

Implementation
--------------
A compiled, multithreaded C++ kernel applies the rules (``radarx.retrieve._cloud_class``),
with the equivalent NumPy implementation as fallback and test oracle. The function
works on :class:`xarray.DataArray`, :class:`xarray.Dataset` and
:class:`xarray.DataTree` objects (sweeps and time-height sections) and keeps dask
arrays lazy.

References
----------
.. [1] Illingworth, A. J., et al., 2007: Cloudnet: Continuous evaluation of cloud
       profiles in seven operational models using ground-based observations.
       *Bull. Amer. Meteor. Soc.*, **88**, 883-898,
       https://doi.org/10.1175/BAMS-88-6-883
.. [2] Tukiainen, S., E. O'Connor, and A. Korpinen, 2020: CloudnetPy: A Python
       package for processing cloud remote sensing data. *J. Open Source Softw.*,
       **5**, 2123, https://doi.org/10.21105/joss.02123

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = ["classify_cloud_targets"]

__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

import numpy as np
import xarray as xr

from .._provenance import provenance
from .._registry import accessor_method
from ._fieldtools import MissingField, choose_engine, detach, map_sweeps
from .cloud_attenuation import _gate_height, _reflectivity

try:
    from . import _cloud_class

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _cloud_class = None
    HAS_COMPILED_KERNEL = False

_RANGE = "range"
_FREEZING = 273.15
_TEMPERATURE_NAMES = ("temperature", "air_temperature", "T")

CLASSES = {
    0: "no_echo",
    1: "cloud_droplets",
    2: "drizzle_or_rain",
    3: "ice",
    4: "melting_ice",
    5: "insects",
    6: "large_ice",
    7: "unknown_phase",
}


def _classify_numpy(dbz, temperature, ldr, dfr, params):
    """NumPy reference of the kernel ``_cloud_class.classify`` (flat arrays, same order of rules)."""
    min_dbz, melt_lo, melt_hi, drizzle, insect_ldr, insect_dbz, large_dfr = params
    with np.errstate(invalid="ignore"):
        echo = np.isfinite(dbz)
        if np.isfinite(min_dbz):
            echo &= dbz >= min_dbz
        known = np.isfinite(temperature)
        melting = known & (temperature >= melt_lo) & (temperature <= melt_hi)
        warm = known & (temperature > melt_hi)
        cold = known & (temperature < melt_lo)
        insect = np.zeros(dbz.shape, dtype=bool)
        if np.isfinite(insect_ldr):
            insect = np.isfinite(ldr) & (ldr >= insect_ldr) & (dbz < insect_dbz)
        large = np.zeros(dbz.shape, dtype=bool)
        if np.isfinite(large_dfr):
            large = np.isfinite(dfr) & (dfr >= large_dfr)
        liquid = np.where(insect, 5, np.where(dbz >= drizzle, 2, 1))
        out = np.select(
            [~echo, ~known, melting, warm, cold],
            [0, 7, 4, liquid, np.where(large, 6, 3)],
            default=0,
        )
    return out.astype(np.int8)


def _classify(dbz, temperature, ldr, dfr, params, use_compiled, n_threads):
    arrays = [
        np.ascontiguousarray(a, dtype=np.float64).reshape(-1)
        for a in (dbz, temperature, ldr, dfr)
    ]
    params = np.ascontiguousarray(params, dtype=np.float64)
    if use_compiled:
        out = _cloud_class.classify(*arrays, params, int(n_threads or 0))
    else:
        out = _classify_numpy(*arrays, params)
    return out.reshape(np.shape(dbz))


def _temperature_kelvin(values, units):
    """Temperature in K from values in K or degrees Celsius (by the units, else by size)."""
    units = str(units or "").lower()
    if units in (
        "degc",
        "degree_celsius",
        "degrees_celsius",
        "celsius",
        "c",
        "degree_c",
    ):
        return values + _FREEZING
    if units in ("k", "kelvin"):
        return values
    return xr.where(values < 150.0, values + _FREEZING, values)


def _field(obj, spec, what):
    """An optional field given by name (a Dataset) or as a DataArray."""
    if isinstance(spec, xr.DataArray):
        return spec
    if isinstance(obj, xr.Dataset):
        if spec not in obj:
            raise MissingField(f"{spec!r} is not in the dataset")
        return obj[spec]
    raise ValueError(f"pass the {what} as a DataArray when obj is a DataArray")


def _gate_temperature(ref, obj, temperature, height, radar_height):
    """Temperature (K) on the gates from a field, a profile or a number."""
    if isinstance(temperature, xr.Dataset):
        if "height" not in temperature.coords or "temperature" not in temperature:
            raise ValueError(
                "a temperature profile needs a 'height' coordinate and 'temperature' (K)"
            )
        prof_h = np.asarray(temperature["height"].values, dtype=np.float64)
        if prof_h.ndim != 1 or prof_h.size < 2 or not np.all(np.diff(prof_h) > 0):
            raise ValueError(
                "the profile height must be one-dimensional and increasing"
            )
        prof_t = _temperature_kelvin(
            temperature["temperature"], temperature["temperature"].attrs.get("units")
        )
        h = _gate_height(ref, obj, height, radar_height)
        return xr.apply_ufunc(
            lambda x: np.interp(x, prof_h, np.asarray(prof_t.values, dtype=np.float64)),
            h,
            dask="parallelized",
            output_dtypes=[np.float64],
        )
    if isinstance(temperature, (int, float, np.integer, np.floating)):
        return xr.DataArray(
            float(temperature) + _FREEZING if temperature < 150 else float(temperature)
        )
    if isinstance(temperature, xr.DataArray):
        return _temperature_kelvin(temperature, temperature.attrs.get("units"))
    if isinstance(obj, xr.Dataset) and isinstance(temperature, str):
        if temperature not in obj:
            raise MissingField(f"{temperature!r} is not in the dataset")
        da = obj[temperature]
        return _temperature_kelvin(da, da.attrs.get("units"))
    if isinstance(obj, xr.Dataset) and temperature is None:
        for name in _TEMPERATURE_NAMES:
            if name in obj:
                da = obj[name]
                return _temperature_kelvin(da, da.attrs.get("units"))
    raise MissingField(
        "no temperature: pass a field name, a DataArray, a profile Dataset or a number"
    )


def _flag_attrs():
    return {
        "long_name": "Target classification of the cloud radar echoes",
        "flag_values": np.array(list(CLASSES), dtype=np.int8),
        "flag_meanings": " ".join(CLASSES.values()),
        "units": "1",
    }


@provenance(
    "Radar-only target classification of cloud radar echoes by reflectivity and temperature, "
    "after the categories of Cloudnet",
)
def classify_cloud_targets(
    obj,
    temperature=None,
    *,
    dbz=None,
    ldr=None,
    dfr=None,
    height=None,
    radar_height=None,
    melting_temperature=(0.0, 2.0),
    min_dbz=None,
    min_drizzle_dbz=-20.0,
    min_insect_ldr=-20.0,
    max_insect_dbz=10.0,
    min_large_dfr=3.0,
    n_threads=None,
    engine="auto",
):
    """
    Classify the echoes of a cloud radar into droplets, drizzle or rain, ice, melting ice and insects.

    See :mod:`radarx.retrieve.cloud_classification` for the classes and the rules.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        The reflectivity (DataArray), a sweep or time-height section with it,
        or a volume with ``sweep_*`` groups (sweeps without a reflectivity are
        skipped).
    temperature : str, float, xarray.DataArray or xarray.Dataset, optional
        Air temperature of the gates: the name of a field (``temperature``,
        ``air_temperature`` or ``T`` by default), a DataArray on the gates, a profile
        Dataset on a ``height`` coordinate (m above sea level) with
        ``temperature`` (interpolated to the heights of the gates, see ``height``), or a
        number (a constant temperature). Kelvin, or degrees Celsius if the
        ``units`` attribute says so (without ``units``, values below 150 are Celsius).
        Use the wet-bulb temperature to place the melting layer as Cloudnet does.
    dbz : str or xarray.DataArray, optional
        The reflectivity in dBZ. Default: ``obj`` itself for a DataArray, else
        the first of ``DBZH``, ``DBZ``, ``reflectivity``, ``Z``, ``Zh``.
    ldr : str or xarray.DataArray, optional
        Linear depolarization ratio in dB, to find insects. Default ``None``: no
        insect class.
    dfr : str or xarray.DataArray, optional
        Dual-frequency ratio in dB (:func:`radarx.retrieve.dual_frequency_ratio`)
        to tell large ice from ice. Default ``None``: no large-ice class.
    height : str or xarray.DataArray, optional
        Height of the gates in m above sea level (a DataArray with a ``range``
        dimension), for a temperature profile. Default: the ``beam_height``,
        ``height`` or ``z`` of the data, else the 4/3 Earth model from the
        ``elevation`` (or ``zenith_angle``).
    radar_height : float, optional
        Radar altitude in m above sea level for the 4/3 Earth model.
    melting_temperature : tuple of float, optional
        Lower and upper bound in degrees Celsius of the melting zone. Default
        ``(0.0, 2.0)``.
    min_dbz : float, optional
        Smallest reflectivity (dBZ) of an echo. Default ``None``: every finite value.
    min_drizzle_dbz : float, optional
        Smallest reflectivity (dBZ) of drizzle or rain. Default -20.
    min_insect_ldr : float, optional
        Smallest depolarization ratio (dB) of insects; ``None`` turns the insect
        rule off. Default -20.
    max_insect_dbz : float, optional
        Largest reflectivity (dBZ) of insects. Default 10.
    min_large_dfr : float, optional
        Smallest dual-frequency ratio (dB) of large ice; ``None`` turns the
        rule off. Default 3.
    n_threads : int, optional
        Threads of the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation. ``"auto"`` (default) prefers the compiled kernel.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        ``TARGET_CLASS`` (int8, codes in the table of the module and in the
        ``flag_values`` and ``flag_meanings`` attributes) on the gates of the
        reflectivity. A volume gives a DataTree with the root of the input and
        one node per sweep.

    Raises
    ------
    ValueError
        For an unknown ``engine``, a melting zone that is not an increasing pair,
        or a profile without ``height``.
    KeyError
        If the reflectivity or the temperature is not found.
    ImportError
        If ``engine="compiled"`` and the kernel is not available.

    References
    ----------
    Illingworth, A. J., et al., 2007: Cloudnet: Continuous evaluation of cloud
    profiles in seven operational models using ground-based observations.
    *Bull. Amer. Meteor. Soc.*, **88**, 883-898,
    https://doi.org/10.1175/BAMS-88-6-883

    Tukiainen, S., E. O'Connor, and A. Korpinen, 2020: CloudnetPy: A Python
    package for processing cloud remote sensing data. *J. Open Source Softw.*,
    **5**, 2123, https://doi.org/10.21105/joss.02123

    Examples
    --------
    >>> cls = classify_cloud_targets(section, "temperature", ldr="ldr")  # doctest: +SKIP
    >>> cls = section.radarx.classify_cloud_targets(model_profile, dfr=dfr)  # doctest: +SKIP
    """
    use_compiled = choose_engine(engine, HAS_COMPILED_KERNEL, "cloud classification")
    try:
        lo, hi = (float(v) for v in melting_temperature)
    except (TypeError, ValueError):
        raise ValueError(
            "melting_temperature must be a pair (lower, upper) in degrees Celsius"
        ) from None
    if not lo < hi:
        raise ValueError("melting_temperature must increase")

    def number(value):
        return np.nan if value is None else float(value)

    params = np.array(
        [
            number(min_dbz),
            lo + _FREEZING,
            hi + _FREEZING,
            float(min_drizzle_dbz),
            number(min_insect_ldr),
            float(max_insect_dbz),
            number(min_large_dfr),
        ]
    )

    def one(sweep, name=None):
        ref = _reflectivity(sweep, dbz)
        temp = _gate_temperature(ref, sweep, temperature, height, radar_height)
        ldr_da = None if ldr is None else _field(sweep, ldr, "depolarization ratio")
        dfr_da = None if dfr is None else _field(sweep, dfr, "dual-frequency ratio")
        nan = xr.full_like(ref, np.nan, dtype=np.float64)
        ldr_da = nan if ldr_da is None else ldr_da
        dfr_da = nan if dfr_da is None else dfr_da

        def block(z, t, lr, df):
            z, t, lr, df = np.broadcast_arrays(z, t, lr, df)
            return _classify(z, t, lr, df, params, use_compiled, n_threads)

        out = xr.apply_ufunc(
            block,
            *detach(ref, temp, ldr_da, dfr_da),
            dask="parallelized",
            output_dtypes=[np.int8],
        )
        out = out.transpose(*[d for d in ref.dims if d in out.dims], ...)
        out = out.astype(np.int8).rename("TARGET_CLASS")
        out.attrs = _flag_attrs()
        return out

    if isinstance(obj, xr.DataTree):
        return map_sweeps(obj, one, "a reflectivity and a temperature")
    if not isinstance(obj, (xr.DataArray, xr.Dataset)):
        raise TypeError("obj must be an xarray DataArray, Dataset or DataTree")
    return one(obj)


@accessor_method(
    "dataarray",
    "dataset",
    "datatree",
    name="classify_cloud_targets",
    wraps=classify_cloud_targets,
)
def _classify_cloud_targets_accessor(self):
    """
    Classify the echoes of a cloud radar into droplets, drizzle or rain, ice, melting ice and insects.

    See :func:`radarx.retrieve.classify_cloud_targets` for the parameters.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        ``TARGET_CLASS`` (int8).
    """
