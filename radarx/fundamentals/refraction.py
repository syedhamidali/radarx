#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Ray Tracing in a Refractive Atmosphere
======================================

The beam height and ground range of :mod:`radarx.fundamentals.geometry` use
the effective Earth radius model, a straight ray over an Earth of radius
:math:`k_e a` that stands for the bending of the ray in an atmosphere whose
refractive index decreases linearly with height. This module is the next step:
it integrates the ray through a measured profile of the radio refractivity
:math:`N(h)`, so that sub-refractive, super-refractive and ducting layers bend
the beam as they do in the profile, and returns the true beam height and ground
range of every range gate.

Equations
---------
The atmosphere is horizontally homogeneous and spherically stratified: the
refractive index :math:`n = 1 + 10^{-6} N` depends only on the distance
:math:`r = a + h` from the centre of the Earth. A ray obeys Snell's law in polar
coordinates, Eq. (3.1) of Bean and Dutton (1966) [1]_,

.. math::

    n\\, r \\cos\\theta = \\mathrm{const},

where :math:`\\theta` is the elevation angle of the ray against the local
horizontal. With the path length :math:`s`, the central angle :math:`\\varphi`
at the centre of the Earth and the geometry of the triangle of the Earth's
centre, the radar and the ray,

.. math::

    \\frac{dr}{ds} = \\sin\\theta,\\qquad
    \\frac{d\\varphi}{ds} = \\frac{\\cos\\theta}{r},\\qquad
    \\frac{d\\theta}{ds} = \\cos\\theta \\left(\\frac{1}{r}
    + \\frac{1}{n}\\frac{dn}{dh}\\right).

The last equation is the sum of the curvature of the Earth and the curvature
of the ray, :math:`-n^{-1}\\,dn/dh\\,\\cos\\theta` (Eq. 1.27 of [1]_), as seen
from the local horizontal. It follows from Snell's law by differentiation and
does not need the sign of :math:`\\theta` to be known, so rays that turn over in
a duct are followed. Where :math:`1/n \\, dn/dh \\le -1/r`, the ray curves at
least as much as the Earth ("trapping", about -157 N-units per km at the
surface of the radarx Earth radius).

The ground range returned is the distance on the ground at sea level,
:math:`a \\varphi`, and the beam height is :math:`r - a`, both for the radar at
the height ``radar_height``; the slant range of the gates is the length of the
ray, :math:`s`. The apparent range of a radar, which is :math:`\\int n\\,ds`
because the pulse travels at the speed of light in the air, differs from it
by at most about 200 m [1]_ (p. 51) and is not used here.

Refractivity
------------
:func:`radarx.io.sounding.refractivity` computes :math:`N` from pressure,
temperature and vapour pressure with the formula of Bean and Dutton [1]_,
Eq. (1.17), :math:`N = 77.6\\,p/T + 3.73\\times10^5\\,e/T^2` (hPa and K), whose
constants they take from Smith and Weintraub (1953) [2]_ (that paper was not
read). The three-term formula of Smith and Weintraub and that of ITU-R
P.453-9 are options there. The standard atmosphere of the 4/3 Earth model has a
constant gradient of -39.2 N-units per km (Section 3.6 of [1]_); the profile
may be any other.

The profile is interpolated linearly in height between its levels. Below the
lowest level and above the highest level the refractive index is constant (the
gradient is zero there). For a ray that leaves the radar below the first
level this is a radarx choice; so is the constant index above the top of a
sounding, where the refractivity is small and the gradient was not measured.

Integration
-----------
The three equations are integrated with the classical fourth-order Runge-Kutta
method (the compiled kernel ``radarx.fundamentals._refraction``, multithreaded
over elevations, with a NumPy twin) in equal steps of at most ``step``
(default 100 m) from gate to gate, so the values at the gates are not
interpolated. Rays with the same elevation share one path. The tests compare
the result with the closed form of a profile for which the ray is a
logarithmic spiral (:math:`n r = n_0 r_0`, :math:`\\theta` constant), with an
independent adaptive integration (SciPy), and with the 4/3 Earth model.

Agreement with the 4/3 Earth model
----------------------------------
For a constant gradient of :math:`dN/dh = -39.2` N-units km\\ :sup:`-1` (the
gradient of :math:`-1/(4a)`) the ray tracer reproduces the effective Earth
radius result (:func:`radarx.fundamentals.geometry.beam_center_height` and
:func:`~radarx.fundamentals.geometry.ground_range` with
``reff = 4/3 * EARTH_RADIUS``). For a radar at 0 or 1.5 km altitude, ranges up
to 200 km and elevations up to :math:`1^\\circ` the beam height differs by less
than 0.51 m and the ground range by less than 1.6e-4 of its value (up to
:math:`5^\\circ`: 1.7 m and 4.3e-4); the effective Earth radius factor of every
gate (:func:`k_e_from_rays`) is between 1.3332 and 1.3343. The difference is
what the effective radius model neglects: :math:`\\cos\\theta` and :math:`n` in
the curvature of the ray, and the ratio of the radii :math:`(a + h/k_e)/(a + h)`
in the horizontal distance. A sub-refractive profile (the refractivity
decreasing by less than 39.2 N-units per km) bends the beam down less than the
4/3 Earth, so that the beam is higher, and a super-refractive profile bends it
down more; the equivalent :math:`k_e` of a constant gradient is
:math:`1/(1 + a\\,dn/dh)` of the effective radius model (Eq. 1.29 of [1]_),
which the tracer reproduces to 0.2 %. Layers whose gradient is steeper than the
trapping value keep a ray that starts in them near the layer (a duct). The
numbers above and these properties are tested in
``tests/fundamentals/test_refraction.py``.

Limits
------
Horizontally homogeneous refractivity: a front or a sea breeze that changes the
profile along the beam is not represented. The ray is not stopped at the
surface; a ray that dips below the sea level keeps going, with the refractivity
of the first level. The geometric optics approximation holds when the
refractive index does not change appreciably over a wavelength [1]_; the
diffraction of a beam in a duct is not modelled. The refractive index does not
depend on the radar frequency (Section 1.3 of [1]_).

References
----------
.. [1] Bean, B. R., and E. J. Dutton, 1966: Radio Meteorology. NBS Monograph
   92, National Bureau of Standards, https://doi.org/10.6028/NBS.MONO.92
.. [2] Smith, E. K., and S. Weintraub, 1953: The constants in the equation for
   atmospheric refractive index at radio frequencies. Proc. IRE, **41** (8),
   1035-1037, https://doi.org/10.1109/JRPROC.1953.274297

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = ["trace_rays", "k_e_from_rays"]
__doc__ = __doc__.replace("{}", "\n   ".join(__all__))

import numpy as np
import xarray as xr

from .._provenance import provenance
from . import _refraction_numpy
from .constants import EARTH_RADIUS

try:
    from . import _refraction

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _refraction = None
    HAS_COMPILED_KERNEL = False


def _engine(engine):
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled refraction kernel is not available")
    return (
        _refraction
        if (HAS_COMPILED_KERNEL and engine != "numpy")
        else _refraction_numpy
    )


def _profile(height, refractivity):
    """Sorted, finite profile arrays ``(h, n)`` with ``n = 1 + 1e-6 N``."""
    h = np.asarray(height, dtype=np.float64).ravel()
    big_n = np.asarray(refractivity, dtype=np.float64).ravel()
    if h.shape != big_n.shape:
        raise ValueError("height and refractivity must have the same length")
    good = np.isfinite(h) & np.isfinite(big_n)
    h, big_n = h[good], big_n[good]
    if h.size < 2:
        raise ValueError("the profile needs at least two valid levels")
    order = np.argsort(h, kind="stable")
    h, big_n = h[order], big_n[order]
    if np.any(np.diff(h) <= 0):
        raise ValueError("the heights of the profile must be distinct")
    if np.abs(big_n).max() > 2000.0:
        raise ValueError(
            "the refractivity must be in N-units (about 0 to 400 in the "
            f"troposphere), got values from {big_n.min():g} to {big_n.max():g}"
        )
    return h, 1.0 + 1e-6 * big_n


def _as_range(range_m):
    if isinstance(range_m, xr.DataArray):
        if range_m.ndim != 1:
            raise ValueError("range_m must be one-dimensional")
        if range_m.dims[0] != "range":
            range_m = range_m.rename({range_m.dims[0]: "range"})
        return range_m.astype(np.float64)
    arr = np.atleast_1d(np.asarray(range_m, dtype=np.float64))
    if arr.ndim != 1:
        raise ValueError("range_m must be one-dimensional")
    return xr.DataArray(arr, dims="range")


def _as_elevation(elevation_deg):
    if isinstance(elevation_deg, xr.DataArray):
        return elevation_deg.astype(np.float64)
    arr = np.asarray(elevation_deg, dtype=np.float64)
    if arr.ndim > 1:
        raise ValueError(
            "pass an xarray.DataArray for an elevation of two or more dims"
        )
    return xr.DataArray(arr, dims=() if arr.ndim == 0 else "ray")


_ATTRS = {
    "beam_height": {
        "long_name": "height of the beam centre above sea level (ray traced)",
        "units": "m",
    },
    "ground_range": {
        "long_name": "distance on the ground at sea level (ray traced)",
        "units": "m",
    },
    "gate_elevation": {
        "long_name": "elevation angle of the ray at the gate against the local horizontal",
        "units": "degree",
    },
}


@provenance(
    "Ray tracing in a spherically stratified refractive atmosphere "
    "(Snell's law in polar coordinates), after Bean and Dutton (1966)"
)
def trace_rays(
    range_m,
    elevation_deg,
    height,
    refractivity,
    *,
    radar_height=0.0,
    step=100.0,
    earth_radius=EARTH_RADIUS,
    n_threads=None,
    engine="auto",
):
    """
    Beam height and ground range of range gates from a refractivity profile.

    Integrates the ray equations of :mod:`radarx.fundamentals.refraction`
    (Snell's law in a spherically stratified atmosphere) for every elevation
    and returns the ray at the range gates. It replaces
    :func:`~radarx.fundamentals.geometry.beam_center_height` and
    :func:`~radarx.fundamentals.geometry.ground_range` where the refractivity
    profile is known.

    Parameters
    ----------
    range_m : float, array-like or xarray.DataArray
        Gate ranges along the ray (path length) in m, one-dimensional (the
        dimension of a DataArray is renamed ``range``). Sorted internally.
    elevation_deg : float, array-like or xarray.DataArray
        Elevation of the beam at the radar in degrees: a number, a
        one-dimensional array (dimension ``ray``) or a DataArray of any
        dimensions (for example the elevation of every ray of a sweep).
        NaN gives NaN. Dask arrays give a lazy result.
    height : array-like
        Heights of the levels of the profile in m above sea level (same
        reference as ``radar_height``), at least two, distinct.
    refractivity : array-like
        Radio refractivity N in N-units at the levels (see
        :func:`radarx.io.sounding.refractivity`). The refractive index is
        ``1 + 1e-6 N``, linear between the levels and constant below and
        above them.
    radar_height : float, optional
        Height of the radar above sea level (m). Default 0.
    step : float, optional
        Longest path step of the integration (m). Default 100. Steps end at
        the levels of the profile; with the sharp inversion of the tests the
        height at 250 km range changes by less than 0.1 mm between steps of
        100 m and 2 m.
    earth_radius : float, optional
        Radius of the spherical Earth (m). Default the radarx value 6371 km
        (:data:`radarx.fundamentals.constants.EARTH_RADIUS`).
    n_threads : int, optional
        Threads of the compiled kernel (per call; each dask chunk is one
        call). Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation. ``"auto"`` (default) prefers the compiled kernel.

    Returns
    -------
    xarray.Dataset
        ``beam_height`` (m above sea level), ``ground_range`` (m, distance on
        the ground at sea level, ``a`` times the central angle) and
        ``gate_elevation`` (degrees, the elevation of the ray at the gate),
        on the dimensions of the elevation and ``range``. The inputs are not
        changed.

    Raises
    ------
    ValueError
        For a profile with fewer than two levels, repeated heights or a
        refractivity of more than 2000 N-units in magnitude, a non-positive ``step`` or
        an unknown ``engine``.
    ImportError
        If ``engine="compiled"`` and the kernel is not available.

    Notes
    -----
    The equations are those of Bean and Dutton [1]_ (Eqs. 1.27 and 3.1); the
    discretisation (Runge-Kutta in the path length), the constant refractive
    index outside the profile and the handling of the surface are radarx
    choices, see the module documentation.

    References
    ----------
    .. [1] Bean, B. R., and E. J. Dutton, 1966: Radio Meteorology. NBS
       Monograph 92, National Bureau of Standards,
       https://doi.org/10.6028/NBS.MONO.92

    Examples
    --------
    >>> import numpy as np
    >>> h = np.arange(0.0, 10001.0, 500.0)
    >>> ray = trace_rays(np.arange(1e3, 200e3, 1e3), 0.5, h, 313.0 - 39.2e-3 * h)
    >>> float(ray.beam_height.isel(range=-1)) > 1000.0
    True
    """
    kernel = _engine(engine)
    if not step > 0:
        raise ValueError("step must be positive")
    if not earth_radius > 0:
        raise ValueError("earth_radius must be positive")
    h, n = _profile(height, refractivity)
    rng = _as_range(range_m)
    elev = _as_elevation(elevation_deg)
    radar_height = float(radar_height)
    step = float(step)
    earth_radius = float(earth_radius)
    n_threads = int(n_threads or 0)

    order = np.argsort(rng.values, kind="stable")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    rng_sorted = np.ascontiguousarray(rng.values[order])

    def block(r_block, e_block):
        e = np.asarray(e_block, dtype=np.float64)
        unique, back = np.unique(e.ravel(), return_inverse=True)
        out = np.asarray(
            kernel.trace(
                rng_sorted, unique, h, n, radar_height, earth_radius, step, n_threads
            )
        )[:, inverse, :]
        out = out[back.reshape(-1)].reshape(e.shape + (rng_sorted.size, 3))
        return out[..., 0], out[..., 1], out[..., 2]

    # shallow copies: with several DataArray inputs xarray 2024.10 strips the
    # attributes of the coordinates of the caller's objects
    height_da, ground_da, angle_da = xr.apply_ufunc(
        block,
        rng.copy(deep=False),
        elev.copy(deep=False),
        input_core_dims=[["range"], []],
        output_core_dims=[["range"]] * 3,
        dask="parallelized",
        output_dtypes=[np.float64] * 3,
        dask_gufunc_kwargs={"allow_rechunk": True},
        keep_attrs=False,
    )
    out = xr.Dataset(
        {
            "beam_height": height_da,
            "ground_range": ground_da,
            "gate_elevation": angle_da,
        }
    )
    # the coordinates of the inputs with their attributes
    coords = {
        name: coord.variable.copy(deep=False)
        for source in (rng, elev)
        for name, coord in source.coords.items()
        if set(coord.dims) <= set(out.dims)
    }
    out = out.assign_coords(coords)
    for name, attrs in _ATTRS.items():
        out[name].attrs = dict(attrs)
    return out


@provenance(
    "Effective Earth radius factor that reproduces a traced beam height, from "
    "the effective Earth radius model of Bean and Dutton (1966)"
)
def k_e_from_rays(range_m, elevation_deg, beam_height, radar_height=0.0):
    """
    Effective Earth radius factor that gives a traced beam height.

    The beam height of the effective Earth radius model is
    :math:`h = \\sqrt{r^2 + A^2 + 2 r A \\sin\\theta} - k_e a` with
    :math:`A = k_e a + h_0` (Eq. 2.28b of Doviak and Zrnic 1993, as in
    :func:`~radarx.fundamentals.geometry.beam_center_height`). For given
    :math:`r`, :math:`\\theta`, :math:`h` and :math:`h_0` it is linear in
    :math:`k_e a`, and the equation is solved for it,

    .. math::

        k_e a = \\frac{r^2 + h_0^2 + 2 r h_0 \\sin\\theta - h^2}
        {2\\,(h - h_0 - r \\sin\\theta)}.

    Use it to find the single ``k_e`` that the gridding, merging and beam
    blockage functions (their ``k_e`` argument) need for the beams that the
    ray tracer :func:`trace_rays` computes, for example the median over the
    gates of a sweep. In a standard atmosphere it is 4/3; where the beam
    bends more it is larger, and for no bending (a straight ray in a flat
    atmosphere) it is infinite. Gates where the traced beam is not bent
    (the denominator is zero, or the range is zero) give NaN.

    Parameters
    ----------
    range_m : array-like or xarray.DataArray
        Slant range of the gates (m).
    elevation_deg : array-like or xarray.DataArray
        Elevation of the beam at the radar (degrees).
    beam_height : array-like or xarray.DataArray
        Traced height of the beam above sea level (m), for example the
        ``beam_height`` of :func:`trace_rays`.
    radar_height : float, optional
        Height of the radar above sea level (m). Default 0.

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        ``k_e``, broadcast like the inputs (a DataArray if any input is one).

    References
    ----------
    .. [1] Bean, B. R., and E. J. Dutton, 1966: Radio Meteorology. NBS
       Monograph 92, National Bureau of Standards,
       https://doi.org/10.6028/NBS.MONO.92

    Examples
    --------
    >>> from radarx.fundamentals.geometry import beam_center_height
    >>> h = beam_center_height(100e3, 0.5, reff=1.25 * 6371000.0)
    >>> round(float(k_e_from_rays(100e3, 0.5, h)), 3)
    1.25
    """
    r = range_m
    sin_e = np.sin(np.deg2rad(elevation_deg))
    h0 = float(radar_height)
    den = 2.0 * (beam_height - h0 - r * sin_e)
    num = r * r + h0 * h0 + 2.0 * r * h0 * sin_e - beam_height * beam_height
    with np.errstate(divide="ignore", invalid="ignore"):
        reff = num / den
        k = reff / EARTH_RADIUS
    if isinstance(k, xr.DataArray):
        k = k.where(np.isfinite(k) & (r > 0))
        k.name = "k_e"
        k.attrs = {"long_name": "effective Earth radius factor", "units": "1"}
        return k
    k = np.where(np.isfinite(k) & (np.asarray(r) > 0), k, np.nan)
    return k
