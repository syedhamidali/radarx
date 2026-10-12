#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Beam Blockage and Terrain Clutter from a Digital Elevation Model
=================================================================

In mountainous country the lowest beams of a radar run into the ground. The
terrain removes part of the beam power, which lowers the reflectivity behind it
(and removes it altogether behind a high ridge), and the beam centre inside or
just above the terrain returns ground clutter. :func:`beam_blockage` finds both
for every gate of a sweep or volume from a digital elevation model (DEM).

Beam path
---------
The beam centre follows the effective Earth radius model of Doviak and Zrnic
(1993, Eqs. 2.28), with the effective radius :math:`k_e a`
(:math:`k_e = 4/3` by default), as in :mod:`radarx.fundamentals.geometry` and
:mod:`radarx.grid`. The radar sits at the altitude :math:`H_0` above sea level,
which is part of the Earth radius in the square root. At slant range :math:`r`
and elevation :math:`\\theta_e`

.. math::

    b = \\sqrt{r^2 + (k_e a + H_0)^2 + 2 r (k_e a + H_0) \\sin\\theta_e}, \\qquad
    h = b - k_e a, \\qquad
    s = k_e a \\arcsin\\frac{r \\cos\\theta_e}{b},

where :math:`h` is the height of the beam centre above sea level and :math:`s`
the distance on the ground. The gate lies at the distance :math:`s` from the
radar in the direction of the azimuth, which gives its latitude and longitude
(the direct problem on a sphere of radius 6371 km) or, for a DEM in projected
coordinates, its position :math:`x_0 + s \\sin\\phi`, :math:`y_0 + s \\cos\\phi`.
The terrain height :math:`t` at that position is interpolated bilinearly in the
DEM.

Partial beam blockage
---------------------
Bech et al. (2003) describe the beam cross section by a circle of radius
:math:`a = r\\beta/2`, where :math:`\\beta` is the half-power beamwidth, and the
terrain by the height difference :math:`y = t - h` between the terrain and the
beam centre. The partial beam blockage (PBB) is the part of the cross section
below the terrain: none if :math:`y \\le -a`, all of it if :math:`y \\ge a` and
in between (their Eqs. A1-A4)

.. math::

    \\mathrm{PBB} = \\frac{1}{\\pi}\\left[\\frac{y}{a}\\sqrt{1 - \\frac{y^2}{a^2}}
    + \\arcsin\\frac{y}{a} + \\frac{\\pi}{2}\\right].

This is ``pattern="uniform"``, a beam of uniform power inside the circle. The
paper notes that a Gaussian beam is "slightly less idealized, but more
realistic" (its Fig. A1 compares the two), and Lang et al. (2009) apply the
method of Bech et al. to "the intersection of a 1 degree Gaussian main lobe
with the DEM" and take the "fraction of the beam's power that is blocked". The
default ``pattern="gaussian"`` does that. The power pattern is Gaussian in the
angle from the beam axis with the half-power width :math:`\\beta`, that is the
standard deviation :math:`\\sigma = a/\\sqrt{2\\ln 2}` in the length units of the
cross section, and the power below the terrain is the normal distribution
function

.. math::

    \\mathrm{PBB} = \\tfrac12\\,\\mathrm{erfc}\\!\\left(\\frac{-y}{\\sqrt2\\,\\sigma}\\right).

Neither paper writes this formula down, so it is derived here from the
description and **not checked against the paper**. The values it gives at
:math:`y = 0, \\pm a` are 0.5, 0.88 and 0.12. A beam that grazes the terrain
(:math:`y = 0`) has a PBB of 0.5 for both patterns. ``two_way=True`` uses the
square of the pattern, which is narrower by :math:`\\sqrt 2` (the weighting of
the received power of a target in the beam) and is a radarx option, not in the
papers.

Cumulative beam blockage
------------------------
Terrain at one range shades everything behind it in the same direction.
Bech et al. (2003) describe the correction of the NEXRAD precipitation
processing (Fulton et al. 1998) as applied "to all pixels further out in range
of the same blocked radar ray, neglecting diffraction below the shadow
boundary", and Lang et al. (2009) correct the reflectivity with the "estimated
cumulative beam blockage fraction". The cumulative beam blockage (CBB) of a
gate is the maximum of the PBB along the ray up to and including the gate,

.. math::

    \\mathrm{CBB}(r_k) = \\max_{j \\le k} \\mathrm{PBB}(r_j).

The papers I could read give no formula for the CBB; the maximum along the ray
is the definition that follows from the description above and is the one used
here.

Terrain clutter
---------------
A gate is flagged as terrain clutter (``TERRAIN_CLUTTER`` = 1) where the beam
centre is not more than ``clutter_margin`` above the terrain surface,
:math:`h - t \\le` margin. With the default margin of 0 these are the gates
whose beam centre is inside the terrain, which have a PBB of at least 0.5. A
larger margin (e.g. half the beam radius) also flags gates that graze the
surface. The margin is a choice of the user.

Choices of radarx
-----------------
* **Terrain on the beam axis.** The terrain height is taken along the axis of
  the beam, as in Bech et al. (2003). The terrain across the width of the beam
  is not sampled, so a narrow ridge at the side of the beam is not seen, and
  the terrain height at a gate is a point value of the DEM.
* **Sampling along the ray.** The terrain is sampled at the gate centres. A
  ridge narrower than the gate spacing can fall between two gates: with
  ``oversample`` = n the terrain is also sampled at n - 1 points between the
  gates, which enter the CBB (the PBB of a gate stays the value at its
  centre).
* **Missing terrain.** The PBB is NaN where the DEM has no value or does not
  reach (and the clutter flag 0). The CBB keeps the maximum of the gates before
  it, and is NaN before the first gate with a terrain height. Fill sea cells of
  the DEM with 0 beforehand (``dem.fillna(0)``).
* **Atmosphere.** A constant refractivity gradient, ``k_e`` = 4/3. The paper of
  Bech et al. shows that the blockage changes much with the gradient in
  superrefractive conditions; pass ``k_e`` from
  :func:`radarx.fundamentals.geometry.k_e_from_refractivity` for them.
* **No Gaussian truncation.** The Gaussian pattern is not cut off, so a PBB of
  1e-9 appears where the terrain is far below the beam; use a threshold, not
  ``> 0``, to find blocked gates.
* **Beam width.** The vertical width ``radar_beam_width_v`` of the metadata
  (else ``radar_beam_width_h``), or ``beamwidth``. There is no default: a width
  that the data do not give has to be passed.
* **QPE and quality control.** The fields are not applied to any other field.
  Pass them as a mask, for example ``dbz.where(bb.CBB < 0.5)``, or use ``CBB``
  as a quality field.

Digital elevation model
-----------------------
``dem`` is an :class:`xarray.DataArray` of the terrain height in metres above
sea level on a regular or irregular grid, either with ``latitude``/``longitude``
(or ``lat``/``lon``) coordinates in degrees, or with projected ``x``/``y``
coordinates in metres together with ``crs``. radarx does not download terrain.
Tiles of SRTM or of the Copernicus DEM (GLO-30) can be read with ``rioxarray``
(``rioxarray.open_rasterio(path).squeeze("band", drop=True)``), which is not a
dependency of radarx; a band dimension of length 1 is dropped here. The part of
the DEM that the radar can reach is cut out, so a large or dask-backed DEM is
fine. A DEM with a cell size much coarser than the gate spacing smooths the
ridges; the ground distance and the azimuth in the projected case are measured
from the site in the plane of the projection, so use a projection that keeps
the distance near the site (UTM within a few degrees of the central meridian).

The numerical work runs in a compiled, multithreaded C++ kernel (over rays);
if it is not available an equivalent NumPy implementation is used. The result
stays lazy when the input is a dask array.

References
----------
.. [1] Bech, J., B. Codina, J. Lorente, and D. Bebbington, 2003: The
   sensitivity of single polarization weather radar beam blockage correction
   to variability in the vertical refractivity gradient. *J. Atmos. Oceanic
   Technol.*, **20** (6), 845-855,
   https://doi.org/10.1175/1520-0426(2003)020<0845:TSOSPW>2.0.CO;2
.. [2] Lang, T. J., S. W. Nesbitt, and L. D. Carey, 2009: On the correction of
   partial beam blockage in polarimetric radar data. *J. Atmos. Oceanic
   Technol.*, **26** (5), 943-957,
   https://doi.org/10.1175/2008JTECHA1133.1
.. [3] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
   Observations*, 2nd ed. Academic Press (book, no DOI). Eqs. (2.28).

The NEXRAD scheme of Fulton et al. (1998) follows the description of Bech et
al.; Fulton et al. has not been compared. Of Lang et al. (2009), the description
of the DEM method (section 2b) is used; its polarimetric methods have not been
compared.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   beam_blockage
"""

from __future__ import annotations

__all__ = ["beam_blockage"]

import numpy as np
import xarray as xr
from scipy.special import erfc

from .._fields import as_sweep, node_dataset
from .._provenance import provenance
from .._registry import accessor_method
from .._sweepwise import dask_support
from ..fundamentals.constants import EARTH_RADIUS
from ..grid.cone import _sweep_names
from ._fieldtools import check_object, choose_engine
from ._products import product_tree

try:
    from . import _beam_blockage

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _beam_blockage = None
    HAS_COMPILED_KERNEL = False

PATTERNS = {"gaussian": 0, "uniform": 2}
_GAUSSIAN_TWO_WAY = 1
_RANGE = "range"
_DEG = np.pi / 180.0
_LAT_NAMES = ("latitude", "lat")
_LON_NAMES = ("longitude", "lon")
_Y_NAMES = ("y", "northing")
_X_NAMES = ("x", "easting")
_BEAMWIDTH_NAMES = ("radar_beam_width_v", "radar_beam_width_h")
_MAX_SAMPLES = 2_000_000  # samples per block of the NumPy implementation


# --------------------------------------------------------------------------
# NumPy reference of the kernel
# --------------------------------------------------------------------------


def _bilinear(z, ya, xa, yq, xq):
    """Bilinear interpolation on ascending axes; NaN outside them or next to NaN."""
    ny, nx = z.shape
    inside = (yq >= ya[0]) & (yq <= ya[-1]) & (xq >= xa[0]) & (xq <= xa[-1])
    i = np.clip(np.searchsorted(ya, yq, side="right") - 1, 0, ny - 2)
    j = np.clip(np.searchsorted(xa, xq, side="right") - 1, 0, nx - 2)
    ty = (yq - ya[i]) / (ya[i + 1] - ya[i])
    tx = (xq - xa[j]) / (xa[j + 1] - xa[j])
    z00, z01 = z[i, j], z[i, j + 1]
    z10, z11 = z[i + 1, j], z[i + 1, j + 1]
    lo = z00 + (z01 - z00) * tx
    hi = z10 + (z11 - z10) * tx
    return np.where(inside, lo + (hi - lo) * ty, np.nan)


def _blocked_fraction(y, beam_radius, pattern):
    """Part of the beam cross section below the terrain, ``y`` above the beam centre."""
    x = y / beam_radius
    if pattern == PATTERNS["uniform"]:
        xc = np.clip(x, -1.0, 1.0)
        return (xc * np.sqrt(1.0 - xc * xc) + np.arcsin(xc) + 0.5 * np.pi) / np.pi
    sigma = (np.sqrt(0.5) if pattern == _GAUSSIAN_TWO_WAY else 1.0) / np.sqrt(
        2.0 * np.log(2.0)
    )
    return 0.5 * erfc(-x / (sigma * np.sqrt(2.0)))


def _locate(s, az_deg, geo):
    """DEM coordinates (y, x) of the points at ground distance ``s`` and azimuth."""
    sin_az = np.sin(az_deg * _DEG)
    cos_az = np.cos(az_deg * _DEG)
    if not geo["geographic"]:
        return geo["site_a"] + s * cos_az, geo["site_b"] + s * sin_az
    delta = s / geo["earth_radius"]
    sd, cd = np.sin(delta), np.cos(delta)
    sin_site = np.sin(geo["site_a"] * _DEG)
    cos_site = np.cos(geo["site_a"] * _DEG)
    sin_lat = np.clip(sin_site * cd + cos_site * sd * cos_az, -1.0, 1.0)
    lat = np.arcsin(sin_lat) / _DEG
    lon = (
        geo["site_b"]
        + np.arctan2(sin_az * sd * cos_site, cd - sin_site * np.sin(lat * _DEG)) / _DEG
    )
    lon = lon + 360.0 * np.floor((geo["lon_center"] - lon) / 360.0 + 0.5)
    return lat, lon


def _blockage_numpy(
    rng, az, el, dem, ya, xa, geo, k_e, beamwidth, pattern, oversample, margin
):
    """
    NumPy implementation of the kernel ``_beam_blockage.blockage`` (same results).

    ``rng`` (gates), ``az`` and ``el`` (rays) in metres and degrees. Returns the
    PBB, the CBB and the terrain height (rays, gates) and the clutter flag.
    """
    nr, ng = az.size, rng.size
    reff = k_e * geo["earth_radius"]
    centre = reff + geo["altitude"]
    half_width = 0.5 * beamwidth * _DEG
    step = np.empty(ng)
    step[1:] = np.diff(rng)
    step[0] = rng[1] - rng[0] if ng > 1 else 0.0
    # sample m of a gate lies m / oversample of a gate length before its centre
    back = np.arange(oversample - 1, -1, -1)
    r = rng[:, None] - step[:, None] * back[None, :] / oversample  # (ng, oversample)
    valid = r > 0.0
    r = np.where(valid, r, np.nan)
    pbb = np.full((nr, ng), np.nan)
    cbb = np.full((nr, ng), np.nan)
    terrain = np.full((nr, ng), np.nan)
    flag = np.zeros((nr, ng), dtype=np.uint8)
    per_ray = ng * oversample
    block = max(1, _MAX_SAMPLES // per_ray)
    with np.errstate(invalid="ignore", divide="ignore"):
        for q0 in range(0, nr, block):
            sl = slice(q0, min(nr, q0 + block))
            ok = np.isfinite(az[sl]) & np.isfinite(el[sl])
            a = np.where(ok, az[sl], 0.0)[:, None, None]
            e = np.where(ok, el[sl], 0.0)[:, None, None]
            rr = r[None, :, :]
            b = np.sqrt(
                rr * rr + centre * centre + 2.0 * rr * centre * np.sin(e * _DEG)
            )
            h = b - reff
            s = reff * np.arcsin(rr * np.cos(e * _DEG) / b)
            yq, xq = _locate(s, a, geo)
            t = _bilinear(dem, ya, xa, yq, xq)
            p = _blocked_fraction(t - h, rr * half_width, pattern)
            p = np.where(ok[:, None, None], p, np.nan)
            t = np.where(ok[:, None, None], t, np.nan)
            run = np.fmax.accumulate(p.reshape(p.shape[0], per_ray), axis=1)
            at_gate = (slice(None), slice(None), oversample - 1)
            pbb[sl] = p[at_gate]
            cbb[sl] = run.reshape(p.shape)[at_gate]
            terrain[sl] = t[at_gate]
            flag[sl] = (h[at_gate] - t[at_gate] <= margin).astype(np.uint8)
    return pbb, cbb, terrain, flag


# --------------------------------------------------------------------------
# DEM and geometry
# --------------------------------------------------------------------------


def _dem_crs(dem, crs):
    """The coordinate reference system of a DEM, from ``crs`` or its metadata."""
    from pyproj import CRS

    if crs is not None:
        return CRS.from_user_input(crs)
    spatial_ref = dem.coords["spatial_ref"] if "spatial_ref" in dem.coords else None
    for source in (spatial_ref.attrs if spatial_ref is not None else {}, dem.attrs):
        for key in ("crs_wkt", "crs"):
            if key in source:
                return CRS.from_user_input(source[key])
    return None


def _dim_of(dem, names):
    for name in names:
        if name in dem.dims:
            return name
    return None


def _dem_axes(dem, crs):
    """Axis names of a DEM and whether its coordinates are latitude and longitude."""
    if not isinstance(dem, xr.DataArray):
        raise TypeError("dem must be an xarray.DataArray of terrain heights (m)")
    dem = dem.squeeze(drop=True)
    if dem.ndim != 2:
        raise ValueError(
            f"dem must have two dimensions (latitude, longitude) or (y, x), not {dem.dims}"
        )
    lat, lon = _dim_of(dem, _LAT_NAMES), _dim_of(dem, _LON_NAMES)
    if lat is not None and lon is not None:
        return dem, lat, lon, True, None
    ydim, xdim = _dim_of(dem, _Y_NAMES), _dim_of(dem, _X_NAMES)
    if ydim is None or xdim is None:
        raise ValueError(
            "dem needs latitude and longitude, or y and x, coordinates; "
            f"it has the dimensions {dem.dims}"
        )
    system = _dem_crs(dem, crs)
    if system is None:
        raise ValueError(
            "a DEM with x and y coordinates needs its coordinate reference "
            "system: pass crs (for example 'EPSG:32632')"
        )
    return dem, ydim, xdim, bool(system.is_geographic), system


def _crop(axis, lo, hi):
    """Slice of an ascending axis that holds [lo, hi] with one cell to spare."""
    first = max(int(np.searchsorted(axis, lo, side="right")) - 2, 0)
    last = min(int(np.searchsorted(axis, hi, side="left")) + 1, axis.size)
    return slice(first, last)


def _prepare_dem(dem, crs, site, reach):
    """
    DEM values, ascending axes and site position in the DEM coordinates.

    ``reach`` is the largest ground distance of a gate (m); only the part of
    the DEM around the site that the radar reaches is loaded.
    """
    dem, ydim, xdim, geographic, system = _dem_axes(dem, crs)
    lat0, lon0, _ = site
    dem = dem.sortby([ydim, xdim])
    ya = np.asarray(dem[ydim].values, dtype=np.float64)
    xa = np.asarray(dem[xdim].values, dtype=np.float64)
    if geographic:
        site_a, site_b = lat0, lon0
        dlat = reach / EARTH_RADIUS / _DEG
        cos_lat = np.cos(np.deg2rad(min(abs(lat0) + dlat, 89.0)))
        dlon = min(dlat / cos_lat, 360.0)
        lon_center = 0.5 * (xa[0] + xa[-1])
        site_b = lon0 + 360.0 * np.floor((lon_center - lon0) / 360.0 + 0.5)
        yr = (site_a - dlat, site_a + dlat)
        xr_ = (site_b - dlon, site_b + dlon)
    else:
        from pyproj import Transformer

        to_dem = Transformer.from_crs("EPSG:4326", system, always_xy=True)
        site_b, site_a = to_dem.transform(lon0, lat0)
        yr = (site_a - reach, site_a + reach)
        xr_ = (site_b - reach, site_b + reach)
    rows, cols = _crop(ya, *yr), _crop(xa, *xr_)
    if rows.stop - rows.start < 2 or cols.stop - cols.start < 2:
        raise ValueError("the DEM does not reach the radar site")
    values = np.ascontiguousarray(
        dem.isel({ydim: rows, xdim: cols}).values, dtype=np.float64
    )
    return values, ya[rows], xa[cols], geographic, site_a, site_b


def _scalar(ds, names):
    """First finite value of the variables ``names`` of a dataset, else None."""
    for name in names:
        if name in ds.variables:
            value = np.asarray(ds[name].values, dtype=np.float64).ravel()
            value = value[np.isfinite(value)]
            if value.size:
                return float(value[0])
    return None


def _site_of(ds, site):
    """Latitude, longitude and altitude of the radar."""
    if site is not None:
        lat, lon, alt = (float(v) for v in site)
        return lat, lon, alt
    values = [_scalar(ds, (name,)) for name in ("latitude", "longitude", "altitude")]
    if None in values:
        raise ValueError(
            "the radar site (latitude, longitude, altitude) is not in the data; "
            "pass site=(latitude, longitude, altitude)"
        )
    return tuple(values)


def _beamwidth_of(beamwidth, sources):
    """Beam width (degrees): the argument or the first value of the metadata."""
    width = float(beamwidth) if beamwidth is not None else _scalar_from(sources)
    if not (np.isfinite(width) and 0.0 < width < 90.0):
        raise ValueError(
            f"the beam width must be between 0 and 90 degrees, not {width!r}"
        )
    return width


def _scalar_from(sources):
    for ds in sources:
        value = _scalar(ds, _BEAMWIDTH_NAMES)
        if value is not None:
            return value
    raise ValueError(
        "the data do not give the beam width (radar_beam_width_v or "
        "radar_beam_width_h); pass beamwidth in degrees"
    )


class _Sweep:
    """Geometry of one sweep: rays, ranges and the dataset that carries them."""

    def __init__(self, template, sources, site, beamwidth):
        if _RANGE not in template.coords:
            raise ValueError("the sweep needs a 'range' coordinate")
        if "azimuth" not in template.variables:
            raise ValueError("the sweep needs an 'azimuth' coordinate")
        az = template["azimuth"]
        if az.ndim != 1:
            raise ValueError("azimuth must be one-dimensional (one value per ray)")
        self.ray_dim = az.dims[0]
        self.template = template
        self.az = np.asarray(az.values, dtype=np.float64)
        self.rng = np.asarray(template[_RANGE].values, dtype=np.float64)
        if self.rng.size == 0 or not (
            np.all(np.isfinite(self.rng)) and np.all(self.rng > 0.0)
        ):
            raise ValueError("range must be finite and positive")
        elevation = _elevation_of(template, self.ray_dim, self.az.size)
        self.el = elevation
        self.site = _site_of(template, site)
        self.beamwidth = _beamwidth_of(beamwidth, [template] + list(sources))


def _elevation_of(template, ray_dim, n):
    for name in ("elevation", "sweep_fixed_angle"):
        if name in template.variables:
            el = template[name]
            if el.ndim == 0:
                return np.full(n, float(el.values))
            if el.dims == (ray_dim,):
                return np.asarray(el.values, dtype=np.float64)
            raise ValueError(f"{name!r} must be a scalar or have one value per ray")
    raise ValueError(
        "the sweep needs an 'elevation' (one value per ray) or a "
        "'sweep_fixed_angle' variable"
    )


# --------------------------------------------------------------------------
# xarray layer
# --------------------------------------------------------------------------

_ATTRS = {
    "PBB": {
        "long_name": "Partial beam blockage by terrain",
        "units": "1",
        "comment": "fraction of the beam power below the terrain at the gate",
    },
    "CBB": {
        "long_name": "Cumulative beam blockage by terrain",
        "units": "1",
        "comment": "maximum of the partial beam blockage along the ray",
    },
    "TERRAIN_HEIGHT": {
        "long_name": "Terrain height below the beam",
        "units": "m",
        "comment": "above sea level, interpolated from the DEM; NaN outside it",
    },
    "TERRAIN_CLUTTER": {
        "long_name": "Gate whose beam centre is at or near the terrain surface",
        "flag_values": np.array([0, 1], dtype=np.int8),
        "flag_meanings": "no_terrain_clutter terrain_clutter",
    },
}


def _block(sweep, dem_arrays, options, compiled):
    """The kernel (or its NumPy implementation) on the rays of one block."""
    values, ya, xa, geographic, site_a, site_b = dem_arrays
    k_e, pattern, oversample, margin, n_threads = options
    rng, beamwidth, altitude = sweep.rng, sweep.beamwidth, sweep.site[2]

    def run(az, el):
        if compiled:
            out = _beam_blockage.blockage(
                rng,
                az,
                el,
                values,
                ya,
                xa,
                geographic,
                site_a,
                site_b,
                altitude,
                k_e,
                EARTH_RADIUS,
                beamwidth,
                pattern,
                oversample,
                margin,
                int(n_threads or 0),
            )
        else:
            geo = {
                "geographic": geographic,
                "site_a": site_a,
                "site_b": site_b,
                "altitude": altitude,
                "earth_radius": EARTH_RADIUS,
                "lon_center": 0.5 * (xa[0] + xa[-1]),
            }
            out = _blockage_numpy(
                rng,
                az,
                el,
                values,
                ya,
                xa,
                geo,
                k_e,
                beamwidth,
                pattern,
                oversample,
                margin,
            )
        pbb, cbb, terrain, flag = out
        return (
            pbb.astype(np.float32),
            cbb.astype(np.float32),
            terrain.astype(np.float32),
            flag.astype(np.int8),
        )

    return run


def _ray_chunks(template, ray_dim):
    """Chunk sizes of the rays when the sweep is a dask array, else None."""
    return template.chunksizes.get(ray_dim)


def _blockage_fields(sweep, dem_arrays, options, compiled, pattern_name, two_way):
    """The products of one sweep as a Dataset on the coordinates of the sweep."""
    ray = sweep.ray_dim
    az = xr.DataArray(sweep.az, dims=(ray,))
    el = xr.DataArray(sweep.el, dims=(ray,))
    chunks = _ray_chunks(sweep.template, ray)
    if chunks is not None:
        az, el = az.chunk({ray: chunks}), el.chunk({ray: chunks})
    run = _block(sweep, dem_arrays, options, compiled)
    pbb, cbb, terrain, flag = xr.apply_ufunc(
        run,
        az,
        el,
        output_core_dims=[[_RANGE]] * 4,
        dask="parallelized",
        output_dtypes=[np.float32, np.float32, np.float32, np.int8],
        dask_gufunc_kwargs={"output_sizes": {_RANGE: sweep.rng.size}},
    )
    out = xr.Dataset(
        {
            "PBB": pbb,
            "CBB": cbb,
            "TERRAIN_HEIGHT": terrain,
            "TERRAIN_CLUTTER": flag,
        }
    )
    coords = {
        name: coord.variable.copy(deep=False)
        for name, coord in sweep.template.coords.items()
        if set(coord.dims) <= {ray, _RANGE}
    }
    out = out.assign_coords(coords)
    for name, attrs in _ATTRS.items():
        out[name].attrs = dict(attrs)
    out.attrs = {
        "beam_blockage_pattern": pattern_name + (" two way" if two_way else ""),
        "beam_blockage_beamwidth": sweep.beamwidth,
        "beam_blockage_k_e": options[0],
    }
    return out


def _reach(sweeps, k_e):
    """Largest ground distance of a gate of the sweeps (m)."""
    reach = 0.0
    for sweep in sweeps:
        reff = k_e * EARTH_RADIUS
        centre = reff + sweep.site[2]
        el = sweep.el[np.isfinite(sweep.el)] * _DEG
        r = sweep.rng.max()
        b = np.sqrt(r * r + centre * centre + 2.0 * r * centre * np.sin(el))
        s = reff * np.arcsin(r * np.cos(el) / b)
        reach = max(reach, float(s.max()) if s.size else 0.0)
    return 1.01 * reach + 1.0


@dask_support("lazy_only")
@provenance(
    "Partial and cumulative beam blockage of a Gaussian (or uniform) beam by "
    "terrain from a digital elevation model, after Bech et al. (2003)",
    extra_refs=("doviak-zrnic-1993",),
)
def beam_blockage(
    obj,
    dem,
    *,
    beamwidth=None,
    k_e=4.0 / 3.0,
    pattern="gaussian",
    two_way=False,
    clutter_margin=0.0,
    oversample=1,
    crs=None,
    site=None,
    n_threads=None,
    engine="auto",
):
    """
    Partial and cumulative beam blockage and terrain clutter of every gate.

    Places the gates of a sweep on a digital elevation model along the beam
    (effective Earth radius model), takes the part of the beam cross section
    below the terrain (Bech et al. 2003) and the maximum of it along the ray.
    The equations, the choices of radarx and how to get a DEM are in
    :mod:`radarx.retrieve.beam_blockage`.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or xarray.DataTree
        A sweep with ``azimuth`` (one value per ray), ``range`` (m) and
        ``elevation`` (one value per ray, else ``sweep_fixed_angle``) and the
        radar site ``latitude``, ``longitude`` and ``altitude``, or a volume
        with ``sweep_*`` groups. A DataArray (a field of the sweep) gives only
        the geometry: its coordinates are used. A dask-backed input gives a lazy
        result.
    dem : xarray.DataArray
        Terrain height (m above sea level) with ``latitude``/``longitude``
        (or ``lat``/``lon``) coordinates in degrees, or ``y``/``x`` projected
        coordinates in metres (then pass ``crs`` unless the DEM carries it, as
        a ``rioxarray`` DataArray does). A dimension of length 1 (``band``) is
        dropped. The DEM is not changed.
    beamwidth : float, optional
        Half-power beam width in degrees. Default: ``radar_beam_width_v``
        (else ``radar_beam_width_h``) of the data; an error if there is none.
    k_e : float, optional
        Effective Earth radius factor. Default 4/3.
    pattern : {"gaussian", "uniform"}, optional
        Power pattern of the beam: Gaussian (default) or the uniform circle of
        Bech et al. (2003).
    two_way : bool, optional
        Use the two-way (squared) Gaussian pattern, narrower by sqrt 2. Not
        with the uniform pattern. Default False.
    clutter_margin : float, optional
        A gate is flagged as terrain clutter where the beam centre is at most
        this height (m) above the terrain surface. Default 0.
    oversample : int, optional
        Samples of the terrain per gate length along the ray (at the gate
        centre and between the gates). Default 1.
    crs : str, int or pyproj.CRS, optional
        Coordinate reference system of a DEM with ``x``/``y`` coordinates (for
        example ``"EPSG:32632"``); a DEM in degrees needs none.
    site : tuple of float, optional
        ``(latitude, longitude, altitude)`` of the radar (degrees, degrees,
        m above sea level) if the data do not hold them.
    n_threads : int, optional
        Threads of the compiled kernel (per call; each dask chunk is one
        call). Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation. ``"auto"`` (default) prefers the compiled kernel.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        On the coordinates of the input: ``PBB`` (partial beam blockage, 0 to
        1), ``CBB`` (cumulative beam blockage, 0 to 1, never smaller than
        ``PBB`` and not decreasing along the ray), ``TERRAIN_HEIGHT`` (m above
        sea level) and ``TERRAIN_CLUTTER`` (int8 flag, 1 where the beam centre
        is at most ``clutter_margin`` above the terrain). NaN where the DEM has
        no value. For a volume a DataTree with one such node per sweep and the
        root of the input. Merge it into the input with
        ``dtree.radarx.assign(products)``, or use the fields as a mask. The
        input is not changed.

    Raises
    ------
    ValueError
        For an unknown option, a missing beam width or radar site, a DEM that
        has no usable coordinates or does not reach the radar.
    TypeError
        If ``obj`` is not an xarray object or ``dem`` not a DataArray.
    ImportError
        If ``engine="compiled"`` and the kernel is not available.

    Notes
    -----
    The Gaussian formula is derived from the description of the method and not
    checked against a paper (see the module documentation). The blockage is
    computed for a horizontal edge of terrain at the height found on the beam
    axis; the terrain across the beam width is not sampled.

    References
    ----------
    .. [1] Bech, J., B. Codina, J. Lorente, and D. Bebbington, 2003: The
       sensitivity of single polarization weather radar beam blockage
       correction to variability in the vertical refractivity gradient. *J.
       Atmos. Oceanic Technol.*, **20** (6), 845-855,
       https://doi.org/10.1175/1520-0426(2003)020<0845:TSOSPW>2.0.CO;2
    .. [2] Lang, T. J., S. W. Nesbitt, and L. D. Carey, 2009: On the correction
       of partial beam blockage in polarimetric radar data. *J. Atmos. Oceanic
       Technol.*, **26** (5), 943-957,
       https://doi.org/10.1175/2008JTECHA1133.1
    .. [3] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
       Observations*, 2nd ed. Academic Press (book, no DOI). Eqs. (2.28).

    Examples
    --------
    >>> dem = rioxarray.open_rasterio("dem.tif").squeeze("band", drop=True)  # doctest: +SKIP
    >>> bb = beam_blockage(sweep, dem)  # doctest: +SKIP
    >>> clean = sweep["DBZH"].where(bb["CBB"] < 0.5)  # doctest: +SKIP
    >>> blocked = dtree.radarx.beam_blockage(dem, k_e=1.2)  # doctest: +SKIP
    """
    check_object(obj)
    if pattern not in PATTERNS:
        raise ValueError(f"pattern must be one of {sorted(PATTERNS)}, not {pattern!r}")
    if two_way and pattern == "uniform":
        raise ValueError("two_way applies to the Gaussian pattern only")
    if not (np.isfinite(k_e) and k_e > 0):
        raise ValueError(f"k_e must be positive, not {k_e!r}")
    if int(oversample) != oversample or oversample < 1:
        raise ValueError(f"oversample must be a positive integer, not {oversample!r}")
    if not clutter_margin >= 0:
        raise ValueError(f"clutter_margin must not be negative, not {clutter_margin!r}")
    compiled = choose_engine(engine, HAS_COMPILED_KERNEL, "beam blockage")
    code = (
        _GAUSSIAN_TWO_WAY if (two_way and pattern == "gaussian") else PATTERNS[pattern]
    )
    options = (
        float(k_e),
        code,
        int(oversample),
        float(clutter_margin),
        n_threads,
    )

    obj = as_sweep(obj)
    if isinstance(obj, xr.DataTree):
        root = obj.root.to_dataset(inherit=False)
        sources = [root]
        if "radar_parameters" in obj.children:
            sources.append(obj["radar_parameters"].to_dataset(inherit=False))
        sweeps = {
            name: _Sweep(node_dataset(obj[name]), sources, site, beamwidth)
            for name in _sweep_names(obj)
        }
        dem_arrays = _prepare_dem(
            dem, crs, next(iter(sweeps.values())).site, _reach(sweeps.values(), k_e)
        )
        nodes = {
            name: _blockage_fields(sw, dem_arrays, options, compiled, pattern, two_way)
            for name, sw in sweeps.items()
        }
        return product_tree(obj, nodes)

    sweep = _Sweep(_as_dataset(obj), [], site, beamwidth)
    dem_arrays = _prepare_dem(dem, crs, sweep.site, _reach([sweep], k_e))
    return _blockage_fields(sweep, dem_arrays, options, compiled, pattern, two_way)


def _as_dataset(obj):
    """A sweep as a Dataset that holds its coordinates (a DataArray's too)."""
    if isinstance(obj, xr.DataArray):
        return obj.to_dataset(name=obj.name or "field")
    return obj


# --------------------------------------------------------------------------
# accessors
# --------------------------------------------------------------------------


@accessor_method(
    "dataarray", "dataset", "datatree", name="beam_blockage", wraps=beam_blockage
)
def _beam_blockage_accessor(self):
    """
    Partial and cumulative beam blockage and terrain clutter from a DEM.

    See :func:`radarx.retrieve.beam_blockage` for the parameters.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        ``PBB``, ``CBB``, ``TERRAIN_HEIGHT`` and ``TERRAIN_CLUTTER``.
    """
