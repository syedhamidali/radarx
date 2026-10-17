#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx Volume Matching of Spaceborne and Ground Radars
=======================================================

Matches the volumes of a spaceborne radar, the Dual-frequency Precipitation Radar
(DPR) of the GPM core satellite, and of a ground radar (GR), after the volume
matching of the GPM ground validation network of Schwaller and Morris (2011): the
measurements of both radars are averaged over the volume that they share and are
compared, for the calibration of the ground radar, the check of the satellite
retrievals and the study of the sampling of the two.

Geometry
--------
The DPR scans across the track, ``nray = 49`` rays at ``nscan`` scans, and its
footprint is about 5 km wide at the surface (Seto et al. 2021). The files give the
position of every ray at the height of the earth ellipsoid (``Latitude``,
``Longitude``). The ray has the zenith angle :math:`\\theta` (``localZenithAngle``, up
to about 18 degrees), so that the part of the ray at the height :math:`h` above the
ellipsoid is displaced horizontally by

.. math::

    \\delta = h \\tan\\theta

toward the nadir track of the satellite (the parallax). :func:`dpr_parallax` gives the
displacement; the direction away from the track is that of the increasing ray
number on the side of the nadir ray (the ray with the smallest zenith angle), taken
from the positions of the rays.

The ground radar sweep :math:`k` samples the height :math:`h_k(s)` at the ground
distance :math:`s` (the effective Earth radius model, 4/3 by default, of
:func:`radarx.fundamentals.beam_height_at_ground_range`). The match volume of a
footprint and a sweep is the volume in which

1. the position of the footprint centre is that of the ray at the height of the beam
   at the same place, :math:`P_0 - h \\tan\\theta\\,\\hat u` with the ellipsoid
   position :math:`P_0` and the unit vector :math:`\\hat u` away from the nadir track
   (the beam height depends on the position and the position on the height: the two
   are iterated until the height changes by less than 1 cm, a few steps for sweeps
   below 30 degrees),
2. the GR gates are inside the horizontal footprint disc (radius
   ``footprint_diameter / 2``), and
3. the DPR range bins are inside the vertical extent of the GR beam at the slant
   range :math:`r` of the footprint, :math:`h \\pm (r\\,\\Delta/2)\\cos\\varepsilon`
   with the half-power beamwidth :math:`\\Delta` and the elevation :math:`\\varepsilon`.

Both sides are averaged. A reflectivity (or any quantity in dB) is averaged in linear
units and converted back; other quantities with the arithmetic mean. The standard
deviation within the volume and the fraction of the volume that has data are returned
with every mean, so that partly filled volumes can be rejected (``min_fraction``).

One compiled, multithreaded C++ kernel does the work for all sweeps and footprints at
once: the gates of a sweep are sorted into a grid of buckets, so that a footprint reads
only the buckets that its disc touches. A NumPy and SciPy implementation gives the same
sums.

Choices of radarx
-----------------
The paper is the basis of the geometry and of the idea of the match volume (Schwaller and
Morris 2011); its code is not used. Details that the paper does not fix, or that radarx
has not checked against it, are radarx's own:

- The horizontal footprint is a circle with ``footprint_diameter`` (default 5 km, the
  nominal horizontal resolution of the DPR, Seto et al. 2021) at every ray, at every
  height. The elongation of the footprint at the edge of the swath (about 5 %) and the
  change of its size with height are not modelled.
- The vertical extent of the beam is the half-power width (``beamwidth``, default
  1 degree, or the ``radar_beam_width_h`` of the volume). The gates are those of the sweep
  at the footprint: the vertical position of every gate inside the disc is not tested.
- ``weighting="uniform"`` (default) gives all gates and bins the same weight.
  ``"gaussian"`` weights the gates by :math:`\\exp(-\\ln 2\\,(d/R)^2)` with the distance
  :math:`d` from the footprint centre and the footprint radius :math:`R`, and the DPR bins
  likewise by their distance from the beam centre in units of the half extent of the
  beam: the half-power points of both patterns are the limits of the volume.
- A mean needs ``min_fraction`` (default 0.5) of the gates in the disc or of the bins
  in the beam to have a value. The fraction counts the gates without echo (NaN) as
  missing: mask or fill the fields before the call as the comparison needs (a clear
  gate is 0 mm/h in a rain rate, and below the minimum detectable signal in a
  reflectivity of the DPR).
- The footprints are those between ``min_range`` and ``max_range`` (defaults 15 and
  100 km) from the radar at the ellipsoid. The heights of the radar and the DPR are
  not corrected for the geoid.
- The ground radar is not corrected for attenuation, the DPR echo is the attenuation
  corrected ``zFactorFinal``. The difference of the frequencies (S, C, X band against Ku
  and Ka) is not converted either.
- Missing values of the DPR (below ``-1000``, ``-9999.9`` in the files) are set to NaN.

DSD parameters
--------------
``paramDSD`` of the DPR gives :math:`10\\log_{10} N_w` and :math:`D_m` of a normalized
gamma drop size distribution (Seto et al. 2021, Eq. 2) with the shape parameter
:math:`\\mu = 3` fixed in the algorithm. radarx uses the same conventions for the
comparison: :math:`D_m` is the mass-weighted mean diameter (mm) and :math:`N_w` the
normalized intercept (m-3 mm-1) of the :math:`D_m` form of Testud et al. (2001), with
:math:`\\mu = 3` (the default of :func:`radarx.retrieve.dsd`). :func:`dpr_dsd` splits
the array of the file into ``DM`` (mm), ``NW`` and ``DBNW`` (dB); a matched ``DBNW`` is
the 10 log10 of the linear mean of :math:`N_w`. The Nw of a ground radar retrieved with
another :math:`\\mu` is not comparable with the DPR (a ratio of :math:`D_m` and
:math:`N_w` that depends on the shape): retrieve with ``mu=3`` for the comparison.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = ["dpr_parallax", "dpr_dsd", "match_dpr", "match_statistics"]

__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

import numpy as np
import xarray as xr

from .._gridcf import aeqd_crs
from .._provenance import provenance
from .._registry import accessor_method
from .._sweepwise import dask_support
from ..fundamentals.geometry import beam_height_at_ground_range
from ..grid import cone
from ._fieldtools import DBZ_NAMES, MissingField, choose_engine, find_field

try:
    from . import _satellite_match

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover
    _satellite_match = None
    HAS_COMPILED_KERNEL = False

from . import _satellite_match_numpy

EARTH_RADIUS = cone.EARTH_RADIUS
_LON_NAMES = ("Longitude", "longitude", "lon")
_LAT_NAMES = ("Latitude", "latitude", "lat")
_ZENITH_NAMES = ("localZenithAngle", "zenith", "local_zenith_angle")
_HEIGHT_NAMES = ("height", "Height")
_TIME_NAMES = ("time", "scan_time")
_DSD_DIMS = ("DSD_params", "nDSD")
_DB_UNITS = ("dBZ", "dB", "dBz")
_MISSING = -1000.0
_WEIGHTINGS = ("uniform", "gaussian")
_MAX_ITERATIONS = 50
_TOLERANCE = 0.01  # m, of the beam height


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


@provenance("Parallax of a DPR ray after Schwaller and Morris (2011)")
def dpr_parallax(height, zenith):
    """
    Horizontal displacement of a DPR ray at a height above the ellipsoid.

    The ray is geolocated at the ellipsoid; the part of it at the height
    :math:`h` lies :math:`h \\tan\\theta` closer to the nadir track of the satellite
    (the satellite is above the track and the ray descends away from it).

    Parameters
    ----------
    height : float or xarray.DataArray
        Height above the ellipsoid (m).
    zenith : float or xarray.DataArray
        Local zenith angle of the ray (degrees; the sign is not used).

    Returns
    -------
    float or xarray.DataArray
        Displacement toward the nadir track (m), non-negative for positive heights.

    References
    ----------
    Schwaller, M. R., and K. R. Morris, 2011: A ground validation network for the
    Global Precipitation Measurement mission. *J. Atmos. Oceanic Technol.*, **28**,
    301-319, https://doi.org/10.1175/2010JTECHA1403.1

    Examples
    --------
    >>> float(radarx.retrieve.dpr_parallax(5000.0, 17.0))  # doctest: +SKIP
    1528.6
    """
    return height * np.tan(np.radians(np.abs(zenith)))


def _dsd_dim(param):
    for name in _DSD_DIMS:
        if name in param.dims:
            return name
    two = [d for d in param.dims if param.sizes[d] == 2]
    if len(two) == 1:
        return two[0]
    raise ValueError(
        f"cannot find the dimension of the two DSD parameters in {param.dims}; "
        "pass dim"
    )


@provenance("Dm and Nw of the DPR DSD parameters (Seto et al. 2021 convention)")
def dpr_dsd(param_dsd, dim=None):
    """
    Dm and Nw of the DSD parameters of the DPR (``paramDSD``).

    ``paramDSD`` holds :math:`10\\log_{10} N_w` (:math:`N_w` in m-3 mm-1) and
    :math:`D_m` (mm), in this order, of the normalized gamma distribution with the
    shape :math:`\\mu = 3` of the DPR algorithm. The missing values of the file (below
    ``-1000``) become NaN.

    Parameters
    ----------
    param_dsd : xarray.DataArray
        ``paramDSD`` with a dimension of two elements (``dim``).
    dim : str, optional
        Name of that dimension. Default: ``"DSD_params"`` or ``"nDSD"``, or the only
        dimension of size two.

    Returns
    -------
    xarray.Dataset
        ``DM`` (mm), ``DBNW`` (dB, 10 log10 of Nw in m-3 mm-1) and ``NW``
        (m-3 mm-1), on the other dimensions of ``param_dsd``. The shape parameter
        is ``attrs["mu"] = 3``.

    Raises
    ------
    ValueError
        If ``dim`` has no two elements, or none can be found.

    References
    ----------
    Seto, S., T. Iguchi, R. Meneghini, J. Awaka, T. Kubota, T. Masaki, and N.
    Takahashi, 2021: The precipitation rate retrieval algorithms for the GPM
    Dual-frequency Precipitation Radar. *J. Meteor. Soc. Japan*, **99**, 205-237,
    https://doi.org/10.2151/jmsj.2021-011

    Testud, J., S. Oury, R. A. Black, P. Amayenc, and X. Dou, 2001: The concept of
    "normalized" distribution to describe raindrop spectra: A tool for cloud physics
    and cloud remote sensing. *J. Appl. Meteor.*, **40**, 1118-1140,
    https://doi.org/10.1175/1520-0450(2001)040<1118:TCONDT>2.0.CO;2

    Examples
    --------
    >>> dsd = radarx.retrieve.dpr_dsd(dpr["paramDSD"])  # doctest: +SKIP
    """
    if not isinstance(param_dsd, xr.DataArray):
        raise TypeError("param_dsd must be an xarray.DataArray")
    dim = dim or _dsd_dim(param_dsd)
    if dim not in param_dsd.dims or param_dsd.sizes[dim] != 2:
        raise ValueError(f"{dim!r} must be a dimension of two elements")
    clean = param_dsd.where(param_dsd > _MISSING)
    dbnw = clean.isel({dim: 0}, drop=True)
    dm = clean.isel({dim: 1}, drop=True)
    out = xr.Dataset(
        {
            "DM": dm.rename("DM"),
            "DBNW": dbnw.rename("DBNW"),
            "NW": (10.0 ** (0.1 * dbnw)).rename("NW"),
        }
    )
    out["DM"].attrs = {
        "long_name": "Mass-weighted mean diameter of the DPR",
        "units": "mm",
    }
    out["DBNW"].attrs = {
        "long_name": "10 log10 of the normalized intercept parameter of the DPR",
        "units": "dB",
    }
    out["NW"].attrs = {
        "long_name": "Normalized intercept parameter of the DPR",
        "units": "m-3 mm-1",
    }
    out.attrs["mu"] = 3.0
    return out


def _fields(spec, ds, default_names, what):
    """List of ``(name, mode)`` for the fields to match; mode 1 is dB."""
    if spec is None:
        spec = [find_field(ds, None, default_names, what)]
    elif isinstance(spec, str):
        spec = [spec]
    items = list(spec.items()) if isinstance(spec, dict) else [(n, None) for n in spec]
    out = []
    for name, mode in items:
        if name not in ds:
            raise MissingField(f"{name!r} is not in the dataset")
        if mode is None:
            mode = "db" if str(ds[name].attrs.get("units", "")) in _DB_UNITS else "mean"
        if mode not in ("db", "mean"):
            raise ValueError(f"the averaging of {name!r} must be 'db' or 'mean'")
        out.append((name, 1 if mode == "db" else 0))
    if not out:
        raise ValueError(f"no {what} field given")
    return out


def _gr_volume(gr):
    """Sweeps, site and root dataset of the ground radar."""
    if isinstance(gr, xr.DataTree):
        sweeps, site, _ = cone._load_volume(gr)
        return sweeps, site, gr.root.to_dataset()
    if isinstance(gr, xr.Dataset):
        if "range" not in gr.dims:
            raise ValueError("a ground radar sweep needs a 'range' dimension")
        for key in ("latitude", "longitude"):
            if key not in gr:
                raise ValueError(f"the sweep needs the radar site {key!r}")
        site = {
            k: float(gr[k]) for k in ("latitude", "longitude", "altitude") if k in gr
        }
        return [gr], site, gr
    raise TypeError("gr must be an xarray DataTree volume or a sweep Dataset")


def _beamwidth(root, beamwidth):
    if beamwidth is not None:
        value = float(beamwidth)
    else:
        value = 1.0
        for key in ("radar_beam_width_h", "radar_beam_width_v"):
            if key in root and np.isfinite(float(root[key])):
                value = float(root[key])
                break
    if not 0.0 < value < 45.0:
        raise ValueError("beamwidth must be between 0 and 45 degrees")
    return value


def _sweep_gates(ds, names, altitude, k_e):
    """Flat gate positions ``(x, y)`` and values ``(nv, n)`` of a sweep."""
    from xradar.georeference import antenna_to_cartesian

    arrays = [cone._sweep_arrays(ds, name) for name, _ in names]
    _, azimuth, elevation, rng = arrays[0]
    gx, gy, _ = antenna_to_cartesian(
        rng[None, :],
        azimuth[:, None],
        elevation[:, None],
        site_altitude=altitude,
        **cone._antenna_kwargs(k_e),
    )
    values = np.stack([a[0].ravel() for a in arrays])
    return gx.ravel(), gy.ravel(), values


def _swath_array(ds, candidates, what, dims, ndim):
    da = ds[find_field(ds, None, candidates, what)]
    keep = dims[:ndim]
    extra = [d for d in da.dims if d not in dims]
    if extra:
        raise ValueError(
            f"{da.name!r} has the dimensions {tuple(da.dims)}: select one element of "
            f"{extra} first"
        )
    missing = [d for d in keep if d not in da.dims]
    if missing:
        raise ValueError(f"{da.name!r} needs the dimensions {keep}, not {da.dims}")
    return da.transpose(*keep)


def _clean(values):
    values = np.asarray(values, dtype=np.float64)
    return np.where(values > _MISSING, values, np.nan)


def _outward(x0, y0, zenith):
    """Unit vectors away from the nadir track, shape of ``x0`` (scan, ray)."""
    if x0.shape[1] < 2:
        raise ValueError("the swath needs at least two rays")
    dx = np.gradient(x0, axis=1)
    dy = np.gradient(y0, axis=1)
    norm = np.hypot(dx, dy)
    norm = np.where(norm > 0, norm, np.nan)
    nadir = int(np.argmin(np.nanmean(np.abs(zenith), axis=0)))
    sign = np.where(np.arange(x0.shape[1]) < nadir, -1.0, 1.0)
    return dx / norm * sign, dy / norm * sign


def _beam_position(x0, y0, ux, uy, tan_z, elevation, altitude, reff):
    """Position, beam height and ground distance of the ray inside the beam."""
    h = beam_height_at_ground_range(np.hypot(x0, y0), elevation, altitude, reff)
    for _ in range(_MAX_ITERATIONS):
        x, y = x0 - h * tan_z * ux, y0 - h * tan_z * uy
        new = beam_height_at_ground_range(np.hypot(x, y), elevation, altitude, reff)
        done = np.nanmax(np.abs(new - h), initial=0.0) < _TOLERANCE
        h = new
        if done:
            break
    x, y = x0 - h * tan_z * ux, y0 - h * tan_z * uy
    return x, y, h, np.hypot(x, y)


def _kernel(use_compiled, name):
    return getattr(_satellite_match if use_compiled else _satellite_match_numpy, name)


def _to_dataarray(array, dims, name, long_name, units, **attrs):
    return xr.DataArray(
        array,
        dims=dims,
        name=name,
        attrs={"long_name": long_name, "units": units, **attrs},
    )


def _finalize(mean, sd, count, total, modes, min_fraction):
    """Back-transform the means and mask the volumes that are too empty."""
    db = (modes == 1)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        logged = np.where(mean > 0, 10.0 * np.log10(mean), np.nan)
        mean = np.where(db, logged, mean)
        fraction = count / np.where(total > 0, total, np.nan)[None, :]
    bad = ~(fraction >= min_fraction) | (count == 0)
    return (
        np.where(bad, np.nan, mean),
        np.where(bad, np.nan, sd),
        np.where(np.isnan(fraction), 0.0, fraction),
    )


# --------------------------------------------------------------------------
# the matching
# --------------------------------------------------------------------------


@dask_support(
    "computes",
    "the gates of every sweep and the footprints of the swath are read into memory.",
)
@provenance(
    "Volume matching of a spaceborne radar and a ground radar after Schwaller and "
    "Morris (2011)",
    extra_refs=("doviak-zrnic-1993",),
)
def match_dpr(
    obj,
    dpr,
    *,
    gr_fields=None,
    dpr_fields=None,
    dims=("scan", "ray", "bin"),
    footprint_diameter=5000.0,
    beamwidth=None,
    min_range=15000.0,
    max_range=100000.0,
    min_fraction=0.5,
    weighting="uniform",
    k_e=4.0 / 3.0,
    max_time_difference=None,
    n_threads=None,
    engine="auto",
):
    """
    Match the volumes of a ground radar and of the GPM DPR.

    For every DPR footprint between ``min_range`` and ``max_range`` from the radar and
    every sweep of the ground radar, the fields of both radars are averaged over their
    common volume, with the parallax of the DPR ray taken into account. See the module
    documentation for the geometry and for the choices of radarx.

    Parameters
    ----------
    obj : xarray.DataTree or xarray.Dataset
        Ground radar volume (sweeps with ``azimuth``, ``elevation``, ``range`` and the
        radar site ``latitude``, ``longitude`` and ``altitude``), or one sweep.
    dpr : xarray.Dataset
        The DPR swath with the dimensions ``dims`` (scan, ray, bin): ``Longitude`` and
        ``Latitude`` of the rays (scan, ray; at the ellipsoid), ``localZenithAngle``
        (scan, ray; degrees), ``height`` of the range bins above the ellipsoid
        (scan, ray, bin; m) and the fields to match, e.g. ``zFactorFinal``. The
        names are those of the products 2A-DPR, 2A-Ku and 2A-Ka (``PRE`` group
        for ``localZenithAngle`` and ``height``, ``SLV`` for ``zFactorFinal`` and
        ``paramDSD``); ``longitude``, ``latitude`` and ``zenith`` are accepted too. An
        optional ``time`` (scan) gives ``time_difference``.
    gr_fields : str, list of str or dict, optional
        Fields of the ground radar to match. Default: the reflectivity. A dict maps
        the name to ``"db"`` (mean in linear units, for dBZ, ZDR and other
        quantities in dB) or ``"mean"`` (arithmetic mean); a name alone takes
        ``"db"`` where the ``units`` are dB or dBZ.
    dpr_fields : str, list of str or dict, optional
        Fields of the DPR to match, like ``gr_fields``. Default: ``zFactorFinal``.
    dims : tuple of str, optional
        Names of the scan, ray and bin dimensions of ``dpr``.
    footprint_diameter : float, optional
        Diameter of the DPR footprint (m). Default 5000.
    beamwidth : float, optional
        Half-power beamwidth of the ground radar (degrees). Default: the
        ``radar_beam_width_h`` of the volume, else 1.
    min_range, max_range : float, optional
        Footprints from ``min_range`` to ``max_range`` (m) from the radar, measured at
        the ellipsoid. Default 15 km and 100 km.
    min_fraction : float, optional
        Smallest fraction of the gates of the disc and of the bins of the beam that
        must have a value for a mean. Default 0.5.
    weighting : {"uniform", "gaussian"}, optional
        Weights of the gates and bins in the volume. Default ``"uniform"``.
    k_e : float, optional
        Effective Earth radius factor. Default 4/3.
    max_time_difference : float, optional
        Where both times are known, volumes more than this many seconds apart are set
        to NaN. Default: no limit.
    n_threads : int, optional
        Threads of the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled kernel.

    Returns
    -------
    xarray.Dataset
        On the dimensions ``footprint`` and ``sweep``: ``gr_<name>`` and
        ``dpr_<name>`` (the means), ``..._std`` (standard deviation of the values in
        the volume, in the units of the field) and ``..._fraction`` (fraction of the
        gates or bins that have a value), ``gr_gates`` and ``dpr_bins`` (number of
        gates and bins in the volume), ``match_longitude``, ``match_latitude``,
        ``match_height`` (m, the centre of the volume), ``ground_range`` (m),
        ``valid`` (True where every mean has a value) and ``time_difference`` (s,
        ground radar minus DPR, where both times are known). Coordinates: ``scan`` and
        ``ray`` of the footprint, its ``longitude`` and ``latitude`` and the
        ``elevation`` of the sweep.

    Raises
    ------
    TypeError
        For an input that is not an xarray object.
    KeyError
        If a field or a variable of the swath is missing, or no sweep has the fields.
    ValueError
        For unknown options, a swath with the wrong dimensions, or a footprint that
        is not a positive diameter.
    ImportError
        If ``engine="compiled"`` and the compiled kernel is not available.

    References
    ----------
    Schwaller, M. R., and K. R. Morris, 2011: A ground validation network for the
    Global Precipitation Measurement mission. *J. Atmos. Oceanic Technol.*, **28**,
    301-319, https://doi.org/10.1175/2010JTECHA1403.1

    Seto, S., T. Iguchi, R. Meneghini, J. Awaka, T. Kubota, T. Masaki, and N.
    Takahashi, 2021: The precipitation rate retrieval algorithms for the GPM
    Dual-frequency Precipitation Radar. *J. Meteor. Soc. Japan*, **99**, 205-237,
    https://doi.org/10.2151/jmsj.2021-011

    Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather Observations*,
    2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no DOI).

    Examples
    --------
    >>> m = radarx.retrieve.match_dpr(vol, dpr, dpr_fields="zFactorFinal")  # doctest: +SKIP
    >>> radarx.retrieve.match_statistics(m.gr_DBZH, m.dpr_zFactorFinal)  # doctest: +SKIP
    """
    use_compiled = choose_engine(engine, HAS_COMPILED_KERNEL, "satellite matching")
    if not isinstance(dpr, xr.Dataset):
        raise TypeError("dpr must be an xarray.Dataset")
    if weighting not in _WEIGHTINGS:
        raise ValueError(f"weighting must be one of {_WEIGHTINGS}, not {weighting!r}")
    if not float(footprint_diameter) > 0.0:
        raise ValueError("footprint_diameter must be positive")
    if not 0.0 <= float(min_fraction) <= 1.0:
        raise ValueError("min_fraction must be between 0 and 1")
    if not 0.0 <= float(min_range) < float(max_range):
        raise ValueError("min_range and max_range must satisfy 0 <= min < max")
    k_e = cone._check_k_e(k_e)
    reff = k_e * EARTH_RADIUS
    gaussian = weighting == "gaussian"
    radius = 0.5 * float(footprint_diameter)

    sweeps, site, root = _gr_volume(obj)
    altitude = site.get("altitude", 0.0)
    width = _beamwidth(root, beamwidth)
    gr_names = _fields(gr_fields, sweeps[0], DBZ_NAMES, "ground radar")
    gr_modes = np.array([m for _, m in gr_names], dtype=np.int64)
    used = [
        ds
        for ds in sweeps
        if all(n in ds and "range" in ds[n].dims for n, _ in gr_names)
    ]
    if not used:
        raise KeyError(
            f"no sweep has the fields {[n for n, _ in gr_names]} of the ground radar"
        )
    dpr_names = _fields(dpr_fields, dpr, ("zFactorFinal",), "DPR")
    dpr_modes = np.array([m for _, m in dpr_names], dtype=np.int64)
    scan_dim, ray_dim, bin_dim = dims

    # DPR geometry
    lon = _clean(_swath_array(dpr, _LON_NAMES, "longitude", dims, 2).values)
    lat = _clean(_swath_array(dpr, _LAT_NAMES, "latitude", dims, 2).values)
    zenith = _clean(_swath_array(dpr, _ZENITH_NAMES, "zenith angle", dims, 2).values)
    height = _clean(_swath_array(dpr, _HEIGHT_NAMES, "bin height", dims, 3).values)
    import pyproj

    to_xy = pyproj.Transformer.from_crs(
        pyproj.CRS.from_epsg(4326),
        aeqd_crs(site["latitude"], site["longitude"]),
        always_xy=True,
    )
    x0, y0 = to_xy.transform(lon, lat)
    ux, uy = _outward(x0, y0, zenith)
    ground = np.hypot(x0, y0)
    with np.errstate(invalid="ignore"):
        chosen = np.isfinite(zenith) & (ground >= min_range) & (ground <= max_range)
    scan_i, ray_i = np.nonzero(chosen)
    nf, ns = scan_i.size, len(used)
    x0, y0 = x0[scan_i, ray_i], y0[scan_i, ray_i]
    ux, uy = ux[scan_i, ray_i], uy[scan_i, ray_i]
    tan_z = np.tan(np.radians(np.abs(zenith[scan_i, ray_i])))

    qx = np.full((nf, ns), np.nan)
    qy, qh, qs, qext = (np.full((nf, ns), np.nan) for _ in range(4))
    elevations = np.empty(ns)
    for k, ds in enumerate(used):
        el = float(np.nanmedian(ds["elevation"].values))
        elevations[k] = el
        x, y, h, s = _beam_position(x0, y0, ux, uy, tan_z, el, altitude, reff)
        slant = (reff + h) * np.sin(s / reff) / np.cos(np.radians(el))
        qx[:, k], qy[:, k], qh[:, k], qs[:, k] = x, y, h, s
        qext[:, k] = slant * np.radians(width) / 2.0 * np.cos(np.radians(el))

    # ground radar side
    gates = [_sweep_gates(ds, gr_names, altitude, k_e) for ds in used]
    offsets = np.concatenate([[0], np.cumsum([g[0].size for g in gates])]).astype(
        np.int64
    )
    gx = np.concatenate([g[0] for g in gates])
    gy = np.concatenate([g[1] for g in gates])
    gvalues = np.concatenate([g[2] for g in gates], axis=1)
    sweep_of = np.broadcast_to(np.arange(ns), (nf, ns)).ravel()
    threads = int(n_threads or 0)
    g_mean, g_sd, g_count, g_total = _kernel(use_compiled, "match_gates")(
        gx,
        gy,
        offsets,
        gvalues,
        gr_modes,
        qx.ravel(),
        qy.ravel(),
        sweep_of.astype(np.int64),
        radius,
        gaussian,
        threads,
    )

    # DPR side
    footprint_height = height[scan_i, ray_i]
    dvalues = (
        np.stack(
            [
                _clean(_swath_array(dpr, (name,), name, dims, 3).values)[scan_i, ray_i]
                for name, _ in dpr_names
            ]
        )
        if nf
        else np.empty((len(dpr_names), 0, height.shape[2]))
    )
    ray_of = np.repeat(np.arange(nf, dtype=np.int64), ns)
    d_mean, d_sd, d_count, d_total = _kernel(use_compiled, "match_bins")(
        footprint_height,
        dvalues,
        dpr_modes,
        ray_of,
        (qh - qext).ravel(),
        (qh + qext).ravel(),
        qh.ravel(),
        gaussian,
        threads,
    )

    g_mean, g_sd, g_frac = _finalize(
        g_mean, g_sd, g_count, g_total, gr_modes, min_fraction
    )
    d_mean, d_sd, d_frac = _finalize(
        d_mean, d_sd, d_count, d_total, dpr_modes, min_fraction
    )

    # time
    difference = np.full((nf, ns), np.nan)
    tname = find_field(dpr, None, _TIME_NAMES, "scan time", required=False)
    if tname is not None and scan_dim in dpr[tname].dims and nf:
        t_dpr = dpr[tname].transpose(scan_dim).values.astype("datetime64[ns]")[scan_i]
        for k, ds in enumerate(used):
            t_gr = cone._mean_time(ds)
            if not np.isnat(t_gr):
                with np.errstate(invalid="ignore"):
                    difference[:, k] = (t_gr - t_dpr) / np.timedelta64(1, "s")
    if max_time_difference is not None:
        far = np.abs(difference) > float(max_time_difference)
        far = far.ravel()
        g_mean[:, far] = np.nan
        d_mean[:, far] = np.nan

    # output
    shape = (nf, ns)
    to_ll = pyproj.Transformer.from_crs(
        aeqd_crs(site["latitude"], site["longitude"]),
        pyproj.CRS.from_epsg(4326),
        always_xy=True,
    )
    mlon, mlat = to_ll.transform(qx, qy)
    dim2 = ("footprint", "sweep")
    out = xr.Dataset(
        coords={
            "scan": ("footprint", scan_i, {"long_name": "scan index of the footprint"}),
            "ray": ("footprint", ray_i, {"long_name": "ray index of the footprint"}),
            "longitude": ("footprint", lon[scan_i, ray_i], {"units": "degrees_east"}),
            "latitude": ("footprint", lat[scan_i, ray_i], {"units": "degrees_north"}),
            "elevation": ("sweep", elevations, {"units": "degree"}),
        }
    )
    for side, names, mean, sd, frac, total in (
        ("gr", gr_names, g_mean, g_sd, g_frac, g_total),
        ("dpr", dpr_names, d_mean, d_sd, d_frac, d_total),
    ):
        source = used[0] if side == "gr" else dpr
        for i, (name, mode) in enumerate(names):
            units = source[name].attrs.get("units", "")
            what = "linear mean" if mode else "mean"
            out[f"{side}_{name}"] = _to_dataarray(
                mean[i].reshape(shape),
                dim2,
                f"{side}_{name}",
                f"{what} of {name} in the match volume ({side})",
                units,
                averaging="db" if mode else "mean",
            )
            out[f"{side}_{name}_std"] = _to_dataarray(
                sd[i].reshape(shape),
                dim2,
                f"{side}_{name}_std",
                f"standard deviation of {name} in the match volume ({side})",
                units,
            )
            out[f"{side}_{name}_fraction"] = _to_dataarray(
                frac[i].reshape(shape),
                dim2,
                f"{side}_{name}_fraction",
                f"fraction of the volume with a value of {name} ({side})",
                "1",
            )
        out[f"{side}_{'gates' if side == 'gr' else 'bins'}"] = _to_dataarray(
            total.reshape(shape).astype(np.int64),
            dim2,
            f"{side}_{'gates' if side == 'gr' else 'bins'}",
            "number of "
            + ("gates" if side == "gr" else "range bins")
            + " in the match volume",
            "1",
        )
    ok = np.isfinite(g_mean).all(axis=0) & np.isfinite(d_mean).all(axis=0)
    out["valid"] = _to_dataarray(
        ok.reshape(shape),
        dim2,
        "valid",
        "every mean of the match volume has a value",
        "1",
    )
    out["match_longitude"] = _to_dataarray(
        mlon, dim2, "match_longitude", "longitude of the match volume", "degrees_east"
    )
    out["match_latitude"] = _to_dataarray(
        mlat, dim2, "match_latitude", "latitude of the match volume", "degrees_north"
    )
    out["match_height"] = _to_dataarray(
        qh,
        dim2,
        "match_height",
        "height of the centre of the beam above sea level",
        "m",
    )
    out["ground_range"] = _to_dataarray(
        qs,
        dim2,
        "ground_range",
        "ground distance of the match volume from the radar",
        "m",
    )
    out["time_difference"] = _to_dataarray(
        difference,
        dim2,
        "time_difference",
        "mean time of the sweep minus time of the scan",
        "s",
    )
    out.attrs.update(
        {
            "footprint_diameter": float(footprint_diameter),
            "beamwidth": width,
            "min_fraction": float(min_fraction),
            "weighting": weighting,
            "k_e": k_e,
            "radar_latitude": site["latitude"],
            "radar_longitude": site["longitude"],
            "radar_altitude": altitude,
        }
    )
    return out


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


@provenance("Agreement statistics of matched fields")
def match_statistics(reference, estimate, *, threshold=None, dim=None):
    """
    Agreement of two matched fields: bias, error, correlation and detection scores.

    Pairs where either value is missing are left out. The differences are
    ``estimate - reference``; for matched reflectivities in dBZ, they are in dB.

    Parameters
    ----------
    reference, estimate : xarray.DataArray
        Matched values with the same shape (e.g. the ground radar and the DPR of
        :func:`match_dpr`; for a rain rate, the radar and the IMERG of
        :func:`match_imerg`).
    threshold : float, optional
        Value from which a pair counts as an event (reflectivity or rain rate).
        Adds the detection scores.
    dim : str or list of str, optional
        Dimensions to reduce. Default: all.

    Returns
    -------
    xarray.Dataset
        ``count``, ``mean_reference``, ``mean_estimate``, ``bias`` (mean of
        estimate minus reference), ``ratio`` (mean estimate / mean reference),
        ``rmse``, ``mae`` and ``correlation`` (Pearson), and with ``threshold`` the
        probability of detection ``pod``, the false alarm ratio ``far``, the critical
        success index ``csi`` and the frequency bias ``frequency_bias``. A score
        without events is NaN.

    Raises
    ------
    TypeError
        If an input is not a DataArray.
    ValueError
        If the shapes differ.
    """
    for da in (reference, estimate):
        if not isinstance(da, xr.DataArray):
            raise TypeError("reference and estimate must be xarray.DataArray")
    if reference.sizes != estimate.sizes:
        raise ValueError("reference and estimate must have the same dimensions")
    estimate = xr.DataArray(
        estimate.transpose(*reference.dims).values,
        dims=reference.dims,
        coords=reference.coords,
    )
    ok = reference.notnull() & estimate.notnull()
    ref, est = reference.where(ok), estimate.where(ok)
    kw = {"dim": dim, "skipna": True}
    n = ok.sum(dim=dim)
    mean_r, mean_e = ref.mean(**kw), est.mean(**kw)
    diff = est - ref
    cov = ((ref - mean_r) * (est - mean_e)).mean(**kw)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / (ref.std(**kw) * est.std(**kw))
    out = xr.Dataset(
        {
            "count": n,
            "mean_reference": mean_r,
            "mean_estimate": mean_e,
            "bias": diff.mean(**kw),
            "ratio": mean_e / mean_r.where(mean_r != 0),
            "rmse": np.sqrt((diff**2).mean(**kw)),
            "mae": abs(diff).mean(**kw),
            "correlation": corr.where(n > 1),
        }
    )
    if threshold is not None:
        event_r, event_e = (ref >= threshold) & ok, (est >= threshold) & ok
        hits = (event_r & event_e).sum(dim=dim)
        false = (~event_r & event_e).sum(dim=dim)
        miss = (event_r & ~event_e).sum(dim=dim)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["pod"] = (hits / (hits + miss)).where(hits + miss > 0)
            out["far"] = (false / (hits + false)).where(hits + false > 0)
            out["csi"] = (hits / (hits + false + miss)).where(hits + false + miss > 0)
            out["frequency_bias"] = ((hits + false) / (hits + miss)).where(
                hits + miss > 0
            )
    return out


# --------------------------------------------------------------------------
# accessors
# --------------------------------------------------------------------------


@accessor_method("dataset", "datatree", name="match_dpr", wraps=match_dpr)
def _match_dpr_accessor(self):
    """
    Match the volumes of this ground radar volume and of the GPM DPR.

    See :func:`radarx.retrieve.match_dpr` for the parameters.

    Returns
    -------
    xarray.Dataset
        The matched means, deviations and fractions on ``footprint`` and ``sweep``.
    """
