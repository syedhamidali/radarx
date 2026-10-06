#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Vertically Pointing Radars and Wind Profilers
=============================================

Readers for profiling instruments into one xarray layout: an
:py:class:`xarray.Dataset` on ``(time, height)``, ``height`` being the
height above sea level (m, the ``height_agl`` coordinate keeps the height
above the instrument), with the scalar coordinates ``latitude``,
``longitude`` and ``altitude`` of the instrument. Profiler winds (``u``,
``v``) on this layout feed :mod:`radarx.retrieve.wind_profile` directly
(``bulk_shear``, ``storm_relative_helicity``, ...), one result per time.

Formats
-------
- :func:`read_mrr`: Micro Rain Radar (METEK MRR-2) averaged profiles as
  netCDF written by IMProToo (``MRR_*`` variables on ``time`` and
  ``MRR rangegate``), e.g. the UAH MAPNet RaDAPS MRR of PERiLS 2022 (Pangle
  and Knupp 2022a). The files have no location; pass it.
- :func:`read_wind_profiler`: Radiometrics 915-MHz radar wind profiler
  consensus winds (``u``, ``v``, ``w``, ``qcTag``, beam moments on
  ``time`` and ``height``, missing values 999.9), e.g. the UAH MAPNet RaDAPS
  profiler of PERiLS 2022 (Pangle and Knupp 2022b).

References
----------
Pangle, P., and K. Knupp, 2022a: PERiLS_2022: UAH MAPNet Micro Rain Radar
(MRR) Data. Version 1.0. UCAR/NCAR Earth Observing Laboratory,
https://doi.org/10.26023/PB1C-EW31-970C

Pangle, P., and K. Knupp, 2022b: PERiLS_2022: UAH MAPNet RaDAPS 915MHz Radar
Wind Profiler (RWP) Data. Version 1.0. UCAR/NCAR Earth Observing Laboratory,
https://doi.org/10.26023/F13E-70W4-5N0J

.. autosummary::
   :nosignatures:
   :toctree: generated/

   read_mrr
   read_wind_profiler
"""

from __future__ import annotations

__all__ = ["read_mrr", "read_wind_profiler"]

import numpy as np
import xarray as xr

_MRR_VARS = {
    "MRR_Capital_Z": (
        "DBZ",
        {
            "standard_name": "equivalent_reflectivity_factor",
            "long_name": "Radar reflectivity factor (attenuation corrected)",
            "units": "dBZ",
        },
    ),
    "MRR_Small_z": (
        "DBZ_ATTENUATED",
        {"long_name": "Attenuated radar reflectivity factor", "units": "dBZ"},
    ),
    "MRR_PIA": ("PIA", {"long_name": "Path-integrated attenuation", "units": "dB"}),
    "MRR_RR": (
        "RAIN_RATE",
        {"long_name": "Rain rate", "units": "mm h-1"},
    ),
    "MRR_LWC": (
        "LWC",
        {"long_name": "Liquid water content", "units": "g m-3"},
    ),
    "MRR_W": (
        "FALL_VELOCITY",
        {
            "long_name": "Mean Doppler (fall) velocity, positive downward",
            "units": "m s-1",
        },
    ),
}
_MRR_SPECTRA = {
    "MRR_D": (
        "drop_diameter",
        {"long_name": "Drop diameter of the bin", "units": "mm"},
    ),
    "MRR_N": (
        "drop_number_density",
        {"long_name": "Drop number density", "units": "m-3 mm-1"},
    ),
    "MRR_F": (
        "spectral_reflectivity",
        {"long_name": "Spectral reflectivity", "units": "dB"},
    ),
}
_LOC_ATTRS = {
    "latitude": {"standard_name": "latitude", "units": "degrees_north"},
    "longitude": {"standard_name": "longitude", "units": "degrees_east"},
    "altitude": {
        "standard_name": "altitude",
        "long_name": "Instrument altitude above sea level",
        "units": "m",
    },
}


def _location(latitude, longitude, altitude):
    return {
        k: xr.DataArray(float(v) if v is not None else np.nan, attrs=_LOC_ATTRS[k])
        for k, v in (
            ("latitude", latitude),
            ("longitude", longitude),
            ("altitude", altitude),
        )
    }


def _height_coords(agl, altitude):
    alt = float(altitude) if altitude is not None else np.nan
    base = alt if np.isfinite(alt) else 0.0
    return {
        "height": (
            "height",
            agl + base,
            {
                "standard_name": "altitude",
                "long_name": (
                    "Height above sea level"
                    if np.isfinite(alt)
                    else "Height above the instrument (altitude unknown)"
                ),
                "units": "m",
            },
        ),
        "height_agl": (
            "height",
            agl,
            {"long_name": "Height above the instrument", "units": "m"},
        ),
    }


def _epoch(seconds):
    return (np.asarray(seconds, dtype=np.float64) * 1e9).astype("datetime64[ns]")


def read_mrr(path, *, latitude=None, longitude=None, altitude=None, spectra=False):
    """
    Read Micro Rain Radar profiles (IMProToo netCDF).

    Parameters
    ----------
    path : str or os.PathLike
        IMProToo netCDF file.
    latitude, longitude, altitude : float, optional
        Instrument location (degrees, m above sea level). ``height`` is above
        sea level when ``altitude`` is given, else above the instrument.
    spectra : bool, optional
        Also return the drop-size spectra (``drop_diameter``,
        ``drop_number_density``, ``spectral_reflectivity`` on
        ``(time, height, bin)``). Default False.

    Returns
    -------
    xarray.Dataset
        ``DBZ``, ``DBZ_ATTENUATED``, ``PIA``, ``RAIN_RATE``, ``LWC`` and
        ``FALL_VELOCITY`` on ``(time, height)``.

    References
    ----------
    Pangle, P., and K. Knupp, 2022: PERiLS_2022: UAH MAPNet Micro Rain Radar
    (MRR) Data. Version 1.0. UCAR/NCAR Earth Observing Laboratory,
    https://doi.org/10.26023/PB1C-EW31-970C
    """
    with xr.open_dataset(path, decode_times=False) as raw:
        gate = raw["MRR rangegate"].values
        if gate.ndim == 2:
            agl = gate[0]
            if not np.allclose(gate, agl[None, :], equal_nan=True):
                raise ValueError("MRR range gates change with time; not supported")
        else:
            agl = gate
        agl = agl.astype(np.float64)
        time = _epoch(raw["time"].values)
        coords = {"time": ("time", time)}
        coords.update(_height_coords(agl, altitude))
        coords.update(_location(latitude, longitude, altitude))
        out = xr.Dataset(coords=coords)
        for name, (new, attrs) in _MRR_VARS.items():
            if name in raw:
                out[new] = (
                    ("time", "height"),
                    raw[name].values.astype(np.float64),
                    attrs,
                )
        if spectra:
            for name, (new, attrs) in _MRR_SPECTRA.items():
                if name in raw:
                    out[new] = (
                        ("time", "height", "bin"),
                        raw[name].values.astype(np.float64),
                        attrs,
                    )
        out.attrs.update(
            {k: str(v) for k, v in raw.attrs.items()}, instrument="Micro Rain Radar"
        )
    return out


def read_wind_profiler(path, *, min_qc=None, beams=False):
    """
    Read Radiometrics radar wind profiler consensus winds (netCDF).

    Parameters
    ----------
    path : str or os.PathLike
        Profiler netCDF file (``u``, ``v``, ``w``, ``qcTag``, ``epochTime``,
        ``height``, ``latitude``, ``longitude``, ``altitude``).
    min_qc : float, optional
        Set winds whose quality tag ``qcTag`` (higher is better) is below this
        to NaN.
    beams : bool, optional
        Also return the beam moments (``Vel_i``, ``SNR_i``, ``SW_i``) on
        ``(time, height, beam)``. Default False.

    Returns
    -------
    xarray.Dataset
        ``u``, ``v``, ``w``, ``wind_speed``, ``wind_direction`` and
        ``quality`` on ``(time, height)``; missing values (999.9) are NaN.

    References
    ----------
    Pangle, P., and K. Knupp, 2022: PERiLS_2022: UAH MAPNet RaDAPS 915MHz Radar
    Wind Profiler (RWP) Data. Version 1.0. UCAR/NCAR Earth Observing
    Laboratory, https://doi.org/10.26023/F13E-70W4-5N0J
    """

    def clean(a):
        a = np.asarray(a, dtype=np.float64)
        return np.where(np.abs(a) >= 999.0, np.nan, a)

    with xr.open_dataset(path, decode_times=False) as raw:
        lat = float(np.ravel(raw["latitude"].values)[0])
        lon = float(np.ravel(raw["longitude"].values)[0])
        alt = float(np.ravel(raw["altitude"].values)[0])
        agl = raw["height"].values.astype(np.float64)
        tname = "epochTime" if "epochTime" in raw else "time"
        coords = {"time": ("time", _epoch(raw[tname].values))}
        coords.update(_height_coords(agl, alt))
        coords.update(_location(lat, lon, alt))
        u, v, w = (clean(raw[k].values) for k in ("u", "v", "w"))
        qc = clean(raw["qcTag"].values) if "qcTag" in raw else np.full(u.shape, np.nan)
        if min_qc is not None:
            bad = ~(qc >= min_qc)
            u[bad] = v[bad] = w[bad] = np.nan
        dims = ("time", "height")
        out = xr.Dataset(
            {
                "u": (dims, u, {"standard_name": "eastward_wind", "units": "m s-1"}),
                "v": (dims, v, {"standard_name": "northward_wind", "units": "m s-1"}),
                "w": (
                    dims,
                    w,
                    {"standard_name": "upward_air_velocity", "units": "m s-1"},
                ),
                "wind_speed": (
                    dims,
                    np.hypot(u, v),
                    {"standard_name": "wind_speed", "units": "m s-1"},
                ),
                "wind_direction": (
                    dims,
                    np.degrees(np.arctan2(-u, -v)) % 360.0,
                    {"standard_name": "wind_from_direction", "units": "degree"},
                ),
                "quality": (
                    dims,
                    qc,
                    {
                        "long_name": "Consensus quality tag (higher is better)",
                        "units": "1",
                    },
                ),
            },
            coords=coords,
        )
        if beams:
            n = 1
            while f"Vel_{n}" in raw:
                n += 1
            if n > 1:
                bd = ("time", "height", "beam")
                for key, name, units in (
                    ("Vel", "beam_velocity", "m s-1"),
                    ("SNR", "beam_snr", "dB"),
                    ("SW", "beam_spectrum_width", "m s-1"),
                ):
                    if f"{key}_1" in raw:
                        out[name] = (
                            bd,
                            np.stack(
                                [clean(raw[f"{key}_{i}"].values) for i in range(1, n)],
                                -1,
                            ),
                            {"units": units},
                        )
                if "beam_azimuths" in raw:
                    out = out.assign_coords(
                        beam_azimuth=(
                            "beam",
                            raw["beam_azimuths"].values[: n - 1].astype(float),
                            {"units": "degree"},
                        )
                    )
        out.attrs.update({k: str(v) for k, v in raw.attrs.items()})
        out.attrs["instrument"] = "radar wind profiler"
    return out
