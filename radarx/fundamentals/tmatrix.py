#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
T-matrix Scattering by Raindrops
================================

Single-drop scattering of oblate raindrops at any radar wavelength, temperature, axis
ratio and canting angle, computed with the T-matrix method (the extended boundary
condition method of Waterman [1]_). The tables of :mod:`radarx.retrieve.dsd`
(S, C and X band, 7 degrees of canting) are built from this module, and the same
function gives the scattering for other wavelengths, canting widths and temperatures
without building a table.

The code is a C++ translation of the double-precision FORTRAN T-matrix code of
Mishchenko for axially symmetric particles, in the version distributed with
``pytmatrix`` (Leinonen [6]_), which radarx used before: the method is that of
Mishchenko and Travis [2]_ and Mishchenko et al. [3]_, the amplitude matrix of a
particle in a fixed orientation follows Mishchenko [4]_. The compiled kernel
``radarx.fundamentals._tmatrix`` is re-entrant, so the drops are computed in parallel,
and it keeps the T-matrix in double precision (the FORTRAN code stores it in
single precision). Against ``pytmatrix`` (the FORTRAN code) it agrees to
1e-7 of the largest element of the amplitude and phase matrices for drops of 0.5 to
8 mm at S, C and X band, with the same orders of the expansion. There is no NumPy
version of the method: without the compiled kernel the functions raise an
:class:`ImportError`.

Definitions
-----------
The amplitude matrix :math:`S` (2 x 2, in mm for lengths in mm) and the phase matrix
:math:`Z` (4 x 4) are those of Mishchenko [4]_ for the incident direction
(``thet0``, ``phi0``) and the scattered direction (``thet``, ``phi``). A drop is an
oblate spheroid with its symmetry axis at the Euler angles (``alpha``, ``beta``) of
Mishchenko [4]_; the canting of the axis is averaged by a quadrature over
:math:`\\beta` with the weight :math:`\\sin\\beta\\,\\exp(-(\\beta-\\bar\\beta)^2 / 2\\sigma^2)`
(a Gaussian distribution of the canting angle on the sphere, Huang et al. [5]_) and a
uniform quadrature over :math:`\\alpha`, as ``pytmatrix.orientation.orient_averaged_fixed``.
For horizontal incidence at 0 degrees elevation (``thet0 = thet = 90``),

.. math::

    \\sigma_{h,v} = 2\\pi (Z_{11} \\mp Z_{12} \\mp Z_{21} + Z_{22})\\ \\text{(backscatter)},\\quad
    K_{DP} = 10^{-3} \\frac{180}{\\pi} \\lambda\\, \\mathrm{Re}(S_{22} - S_{11})
    \\ \\text{(forward)},\\quad
    A_{h} = 4.343\\times10^{-3}\\, 2\\lambda\\, \\mathrm{Im}\\, S_{22},

with :math:`\\lambda` in mm, per drop per m3: the backscatter cross sections in mm2, the
specific differential phase in degrees per km and the specific attenuation in dB per km.
These are the quantities of the radarx drop-size tables.

Assumptions
-----------
- Oblate spheroids with the axis ratio of Brandes et al. [7]_ (a radarx choice, see
  :func:`axis_ratio_brandes`) or any axis ratio.
- The refractive index of liquid water comes from the Debye model of Ray [8]_ at the
  temperature of the rain (:func:`water_refractive_index`; the constants are not
  checked against the paper).
- The canting distribution is Gaussian with zero mean; 7 degrees is the radarx default
  (Huang et al. [5]_; the value is not checked against the paper). The width matters:
  at S band 20 degrees instead of 7 lowers the ZDR of a 4 mm drop by 0.65 dB, that of an
  8 mm drop by 1.7 dB, and K_DP by 27 %. Kumjian and Ryzhkov [9]_ assume 20 degrees,
  call 10 degrees more appropriate for typical rainfall (they chose 20 to enlarge the
  effect of evaporation on the correlation coefficient), and define S band as 10.9 cm
  and C band as 5.3 cm (radarx: 10.7 and 5.4 cm, at 2.8 and 5.6 GHz).
- The size parameter of the largest drop sets the cost: the order of the expansion is
  about :math:`x + 4.05 x^{1/3}` with :math:`x = 2\\pi a/\\lambda`, up to 99.

References
----------
.. [1] Waterman, P. C., 1971: Symmetry, unitarity, and geometry in electromagnetic
       scattering. *Phys. Rev. D*, **3**, 825-839,
       https://doi.org/10.1103/PhysRevD.3.825
.. [2] Mishchenko, M. I., and L. D. Travis, 1994: T-matrix computations of light
       scattering by large spheroidal particles. *Opt. Commun.*, **109**, 16-21,
       https://doi.org/10.1016/0030-4018(94)90731-5
.. [3] Mishchenko, M. I., L. D. Travis, and A. Macke, 1996: Scattering of light by
       polydisperse, randomly oriented, finite circular cylinders. *Appl. Opt.*,
       **35**, 4927-4940, https://doi.org/10.1364/AO.35.004927
.. [4] Mishchenko, M. I., 2000: Calculation of the amplitude matrix for a
       nonspherical particle in a fixed orientation. *Appl. Opt.*, **39**, 1026-1031,
       https://doi.org/10.1364/AO.39.001026
.. [5] Huang, G.-J., V. N. Bringi, and M. Thurai, 2008: Orientation angle
       distributions of drops after an 80-m fall using a 2D video disdrometer.
       *J. Atmos. Oceanic Technol.*, **25**, 1717-1723,
       https://doi.org/10.1175/2008JTECHA1075.1
.. [6] Leinonen, J., 2014: High-level interface to T-matrix scattering
       calculations: architecture, capabilities and limitations. *Opt. Express*,
       **22**, 1655-1660, https://doi.org/10.1364/OE.22.001655
.. [7] Brandes, E. A., G. Zhang, and J. Vivekanandan, 2002: Experiments in rainfall
       estimation with a polarimetric radar in a subtropical environment. *J. Appl.
       Meteor.*, **41**, 674-685,
       https://doi.org/10.1175/1520-0450(2002)041<0674:EIREWA>2.0.CO;2
.. [8] Ray, P. S., 1972: Broadband complex refractive indices of ice and water.
       *Appl. Opt.*, **11**, 1836-1844, https://doi.org/10.1364/AO.11.001836
.. [9] Kumjian, M. R., and A. V. Ryzhkov, 2010: The impact of evaporation on
       polarimetric characteristics of rain: Theoretical model and practical
       implications. *J. Appl. Meteor. Climatol.*, **49**, 1247-1267,
       https://doi.org/10.1175/2010JAMC2243.1

.. autosummary::

    raindrop_scattering
    spheroid_scattering
    orientation_quadrature
    water_refractive_index
    axis_ratio_brandes
"""

import numpy as np
import xarray as xr

from .._provenance import provenance

try:
    from . import _tmatrix

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _tmatrix = None
    HAS_COMPILED_KERNEL = False

__all__ = [
    "raindrop_scattering",
    "spheroid_scattering",
    "orientation_quadrature",
    "water_refractive_index",
    "axis_ratio_brandes",
]

BACKSCATTER = (90.0, 90.0, 0.0, 180.0)  # thet0, thet, phi0, phi: horizontal incidence
FORWARD = (90.0, 90.0, 0.0, 0.0)


def _require_kernel(engine="auto"):
    if engine == "numpy":
        raise ValueError(
            "the T-matrix method has no NumPy version: use engine='auto' or 'compiled'"
        )
    if engine not in ("auto", "compiled"):
        raise ValueError(f"engine must be 'auto' or 'compiled', not {engine!r}")
    if not HAS_COMPILED_KERNEL:
        raise ImportError(
            "radarx.fundamentals.tmatrix needs the compiled kernel "
            "radarx.fundamentals._tmatrix; install a radarx wheel or build the "
            "package with a C++17 compiler"
        )


def axis_ratio_brandes(diameter):
    """
    Vertical-to-horizontal axis ratio of a raindrop after Brandes et al. (2002).

    :math:`b/a = 0.9951 + 0.0251 D - 0.03644 D^2 + 0.005303 D^3 - 0.0002492 D^4`
    with the equal-volume diameter :math:`D` in mm, capped at 1 (a radarx choice).
    The polynomial agrees with Eq. 15 of Kumjian and Ryzhkov (2010), who print the
    second coefficient as 0.025 10; the equation number of Brandes et al. (2002) is not
    checked against the paper.

    Parameters
    ----------
    diameter : array_like
        Equal-volume diameter in mm.

    Returns
    -------
    numpy.ndarray
        The axis ratio (at most 1).

    References
    ----------
    .. [1] Brandes, E. A., G. Zhang, and J. Vivekanandan, 2002: Experiments in
           rainfall estimation with a polarimetric radar in a subtropical
           environment. *J. Appl. Meteor.*, **41**, 674-685,
           https://doi.org/10.1175/1520-0450(2002)041<0674:EIREWA>2.0.CO;2
    .. [2] Kumjian, M. R., and A. V. Ryzhkov, 2010: The impact of evaporation on
           polarimetric characteristics of rain: Theoretical model and practical
           implications. *J. Appl. Meteor. Climatol.*, **49**, 1247-1267,
           https://doi.org/10.1175/2010JAMC2243.1
    """
    d = np.asarray(diameter, dtype=float)
    r = 0.9951 + 0.0251 * d - 0.03644 * d**2 + 0.005303 * d**3 - 0.0002492 * d**4
    return np.minimum(r, 1.0)


def water_refractive_index(wavelength, temperature=20.0):
    """
    Complex refractive index of liquid water after Ray (1972), :math:`m = n - j k`.

    Debye-type model: static and high-frequency permittivities, relaxation wavelength,
    distribution parameter and ionic conductivity term as functions of temperature.
    The constants are not checked against the paper.

    Parameters
    ----------
    wavelength : float
        Wavelength in mm.
    temperature : float, optional
        Temperature in degrees Celsius. Default 20.

    Returns
    -------
    complex
        The refractive index, with a positive real and imaginary part (the
        convention of the T-matrix code, :math:`m = n + jk` with :math:`k > 0`).

    References
    ----------
    .. [1] Ray, P. S., 1972: Broadband complex refractive indices of ice and water.
           *Appl. Opt.*, **11**, 1836-1844, https://doi.org/10.1364/AO.11.001836
    """
    t = float(temperature)
    lam = float(wavelength) / 10.0  # cm
    eps_s = 78.54 * (
        1.0
        - 4.579e-3 * (t - 25.0)
        + 1.19e-5 * (t - 25.0) ** 2
        - 2.8e-8 * (t - 25.0) ** 3
    )
    eps_inf = 5.27137 + 0.0216474 * t - 0.00131198 * t**2
    alpha = -16.8129 / (t + 273.0) + 0.0609265
    lam_s = 0.00033836 * np.exp(2513.98 / (t + 273.0))  # cm
    sigma = 12.5664e8
    x = (lam_s / lam) ** (1.0 - alpha)
    s, c = np.sin(alpha * np.pi / 2.0), np.cos(alpha * np.pi / 2.0)
    den = 1.0 + 2.0 * x * s + x * x
    eps_r = eps_inf + (eps_s - eps_inf) * (1.0 + x * s) / den
    eps_i = (eps_s - eps_inf) * x * c / den + sigma * lam / 18.8496e10
    return complex(np.sqrt(complex(eps_r, eps_i)))


def _gautschi(z, w, n_iter):
    """Recurrence coefficients of the polynomials orthogonal for the discrete measure."""
    p = np.ones(z.shape)
    p /= np.sqrt(np.dot(p, p))
    p_prev = np.zeros(z.shape)
    wz = z * w
    a = np.empty(n_iter)
    b = np.empty(n_iter)
    for j in range(n_iter):
        p_norm = np.dot(w * p, p)
        a[j] = np.dot(wz * p, p) / p_norm
        b[j] = 0.0 if j == 0 else p_norm / np.dot(w * p_prev, p_prev)
        p_new = (z - a[j]) * p - b[j] * p_prev
        p_prev = p
        p = p_new
    return a, b[1:]


def orientation_quadrature(
    std=7.0, mean=0.0, n_alpha=5, n_beta=10, distribution="gaussian"
):
    """
    Euler angles and weights for the average over the orientation of a drop.

    The angle :math:`\\beta` between the symmetry axis and the vertical has the
    density :math:`\\sin\\beta\\, f(\\beta)` on :math:`[0, 180]` degrees, with a Gaussian
    :math:`f` (``std`` and ``mean`` in degrees) or a uniform one; its quadrature
    (``n_beta`` points) is the Gaussian quadrature of the discretized density, found
    from the eigenvalues of the Jacobi matrix (Golub and Welsch, as ``pytmatrix``). The
    angle :math:`\\alpha` takes ``n_alpha`` equally spaced values in [0, 360).

    Parameters
    ----------
    std : float, optional
        Standard deviation of the canting angle in degrees (Gaussian). Default 7.
    mean : float, optional
        Mean of the canting angle in degrees. Default 0.
    n_alpha, n_beta : int, optional
        Number of points of the two angles. Default 5 and 10.
    distribution : {"gaussian", "uniform"}, optional
        ``"uniform"`` gives the random orientation (``std`` and ``mean`` are ignored).

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_alpha * n_beta, 3)`` with ``alpha``, ``beta`` (degrees)
        and the weight of every orientation.
    """
    if distribution not in ("gaussian", "uniform"):
        raise ValueError("distribution must be 'gaussian' or 'uniform'")
    n = 4096
    dx = 180.0 / n
    z = np.linspace(0.5 * dx, 180.0 - 0.5 * dx, n)
    if distribution == "gaussian":
        if not std > 0:
            raise ValueError("std must be positive")
        w = dx * np.exp(-0.5 * ((z - mean) / std) ** 2) * np.sin(np.pi / 180.0 * z)
    else:
        w = dx * np.sin(np.pi / 180.0 * z)
    a, b = _gautschi(z, w, int(n_beta))
    jac = np.diag(a) + np.diag(np.sqrt(b), k=-1) + np.diag(np.sqrt(b), k=1)
    points, vecs = np.linalg.eigh(jac)
    order = points.argsort()
    points = points[order]
    weights = (vecs[0, :] ** 2 * w.sum())[order]
    alpha = np.linspace(0.0, 360.0, int(n_alpha) + 1)[:-1]
    out = np.empty((alpha.size * points.size, 3))
    out[:, 0] = np.repeat(alpha, points.size)
    out[:, 1] = np.tile(points, alpha.size)
    out[:, 2] = np.tile(weights, alpha.size)
    return out


def spheroid_scattering(
    radius,
    axis_ratio,
    wavelength,
    m,
    geometry,
    orientation,
    ddelt=1e-3,
    ndgs=2,
    rat=1.0,
    *,
    n_threads=None,
    engine="auto",
):
    """
    Amplitude and phase matrices of spheroids in given directions and orientations.

    Parameters
    ----------
    radius : array_like
        Equal-volume radius of every particle (``rat = 1``), in the unit of
        ``wavelength``.
    axis_ratio : array_like
        Ratio of the horizontal to the rotational axis of every particle (> 1 for
        oblate spheroids, < 1 for prolate), the same length as ``radius`` (a scalar
        is repeated).
    wavelength : float
        Wavelength, in the same unit as ``radius``.
    m : complex
        Complex refractive index with a positive imaginary part.
    geometry : array_like
        Shape ``(g, 4)`` or ``(4,)``: ``thet0``, ``thet``, ``phi0``, ``phi`` in
        degrees (zenith and azimuth of the incident and scattered directions).
    orientation : array_like
        Shape ``(k, 3)``: Euler angles ``alpha`` (0 to 360) and ``beta`` (0 to 180) in
        degrees and a weight; the result is the weighted mean over the orientations
        (see :func:`orientation_quadrature`; one row with weight 1 is a fixed
        orientation).
    ddelt : float, optional
        Accuracy of the expansion in the order and in the number of integration
        points. Default 1e-3 (the recommended value).
    ndgs : int, optional
        Number of integration points per unit of the order; 2 for compact particles,
        larger values (3 to 4) for aspherical ones. Default 2.
    rat : float, optional
        1 if ``radius`` is the equal-volume radius (default), another value for the
        equal-surface-area radius.
    n_threads : int, optional
        Threads (one particle per thread). Default: all cores.
    engine : {"auto", "compiled"}, optional
        There is only the compiled kernel; ``"numpy"`` raises a ``ValueError``.

    Returns
    -------
    S : numpy.ndarray
        Complex amplitude matrices, shape ``(n, g, 2, 2)`` (``S[..., 0, 0]`` is the
        vertical-vertical element).
    Z : numpy.ndarray
        Phase matrices, shape ``(n, g, 4, 4)``.
    nmax : numpy.ndarray
        Order of the multipole expansion used for every particle.

    Raises
    ------
    ImportError
        If the compiled kernel is not available.
    RuntimeError
        If the expansion does not converge (a size parameter that is too large).

    References
    ----------
    .. [1] Mishchenko, M. I., 2000: Calculation of the amplitude matrix for a
           nonspherical particle in a fixed orientation. *Appl. Opt.*, **39**,
           1026-1031, https://doi.org/10.1364/AO.39.001026
    .. [2] Mishchenko, M. I., and L. D. Travis, 1994: T-matrix computations of light
           scattering by large spheroidal particles. *Opt. Commun.*, **109**, 16-21,
           https://doi.org/10.1016/0030-4018(94)90731-5
    """
    _require_kernel(engine)
    radius = np.atleast_1d(np.asarray(radius, dtype=float))
    axis_ratio = np.broadcast_to(np.asarray(axis_ratio, dtype=float), radius.shape)
    geometry = np.atleast_2d(np.asarray(geometry, dtype=float))
    orientation = np.atleast_2d(np.asarray(orientation, dtype=float))
    m = complex(m)
    return _tmatrix.scatter_batch(
        np.ascontiguousarray(radius),
        np.ascontiguousarray(axis_ratio),
        float(rat),
        float(wavelength),
        m.real,
        m.imag,
        float(ddelt),
        int(ndgs),
        geometry,
        orientation,
        0 if n_threads is None else int(n_threads),
    )


@provenance(
    "T-matrix scattering by raindrops (Mishchenko FORTRAN code translated to C++)"
)
def raindrop_scattering(
    diameter,
    wavelength,
    temperature=20.0,
    m=None,
    axis_ratio=None,
    canting_std=7.0,
    n_alpha=5,
    n_beta=10,
    ndgs=4,
    ddelt=1e-3,
    *,
    n_threads=None,
    engine="auto",
):
    """
    Scattering by raindrops of given equal-volume diameters, as the radarx tables.

    Computes, for every diameter, the quantities of the table of
    :mod:`radarx.retrieve.dsd` at horizontal incidence: the backscatter cross sections
    at horizontal and vertical polarization, the real and imaginary parts of the
    copolar correlation term, the specific differential phase and the specific
    attenuation, all per drop per cubic metre (see the module documentation for the
    definitions). The drops are oblate spheroids whose axes are canted by a Gaussian
    distribution of the angle with the vertical.

    Parameters
    ----------
    diameter : array_like
        Equal-volume diameters in mm.
    wavelength : float
        Radar wavelength in mm (for example ``299.792458 / 2.8`` for S band at 2.8 GHz).
    temperature : float, optional
        Temperature of the water in degrees Celsius, for its refractive index.
        Default 20.
    m : complex, optional
        Complex refractive index (positive imaginary part). Default: Ray (1972) at
        ``temperature`` (:func:`water_refractive_index`).
    axis_ratio : callable or array_like, optional
        Vertical-to-horizontal axis ratio (at most 1) of every diameter, or a function
        of the diameter in mm. Default :func:`axis_ratio_brandes`.
    canting_std : float, optional
        Standard deviation of the canting angle in degrees. Default 7.
    n_alpha, n_beta : int, optional
        Points of the orientation quadrature (:func:`orientation_quadrature`).
    ndgs : int, optional
        Integration points per unit of the order, 4 as the radarx tables.
    ddelt : float, optional
        Accuracy of the T-matrix expansion. Default 1e-3.
    n_threads : int, optional
        Threads. Default: all cores.
    engine : {"auto", "compiled"}, optional
        There is only the compiled kernel; ``"numpy"`` raises a ``ValueError``.

    Returns
    -------
    xarray.Dataset
        On ``diameter`` (mm): ``sigma_h``, ``sigma_v``, ``copol_re``, ``copol_im``
        (mm2), ``kdp`` (degree km-1), ``ah`` and ``av`` (dB km-1) per drop per m3, and
        ``nmax``. The attributes hold the wavelength, temperature, refractive index and
        canting width.

    Raises
    ------
    ImportError
        If the compiled kernel is not available.

    References
    ----------
    .. [1] Mishchenko, M. I., and L. D. Travis, 1994: T-matrix computations of light
           scattering by large spheroidal particles. *Opt. Commun.*, **109**, 16-21,
           https://doi.org/10.1016/0030-4018(94)90731-5
    .. [2] Mishchenko, M. I., 2000: Calculation of the amplitude matrix for a
           nonspherical particle in a fixed orientation. *Appl. Opt.*, **39**,
           1026-1031, https://doi.org/10.1364/AO.39.001026
    .. [3] Brandes, E. A., G. Zhang, and J. Vivekanandan, 2002: Experiments in
           rainfall estimation with a polarimetric radar in a subtropical
           environment. *J. Appl. Meteor.*, **41**, 674-685,
           https://doi.org/10.1175/1520-0450(2002)041<0674:EIREWA>2.0.CO;2
    .. [4] Huang, G.-J., V. N. Bringi, and M. Thurai, 2008: Orientation angle
           distributions of drops after an 80-m fall using a 2D video disdrometer.
           *J. Atmos. Oceanic Technol.*, **25**, 1717-1723,
           https://doi.org/10.1175/2008JTECHA1075.1

    Examples
    --------
    >>> import numpy as np
    >>> from radarx.fundamentals.tmatrix import raindrop_scattering
    >>> d = raindrop_scattering(np.arange(0.5, 6.1, 0.5), 299.792458 / 2.8)  # doctest: +SKIP
    >>> zdr = 10 * np.log10(d.sigma_h / d.sigma_v)  # doctest: +SKIP
    """
    _require_kernel(engine)
    diameter = np.atleast_1d(np.asarray(diameter, dtype=float))
    if m is None:
        m = water_refractive_index(wavelength, temperature)
    if axis_ratio is None:
        vertical_over_horizontal = axis_ratio_brandes(diameter)
    elif callable(axis_ratio):
        vertical_over_horizontal = np.asarray(axis_ratio(diameter), dtype=float)
    else:
        vertical_over_horizontal = np.broadcast_to(
            np.asarray(axis_ratio, dtype=float), diameter.shape
        )
    orientation = orientation_quadrature(canting_std, 0.0, n_alpha, n_beta)
    s, z, nmax = spheroid_scattering(
        diameter / 2.0,
        1.0 / vertical_over_horizontal,
        wavelength,
        m,
        np.array([BACKSCATTER, FORWARD]),
        orientation,
        ddelt=ddelt,
        ndgs=ndgs,
        n_threads=n_threads,
        engine=engine,
    )
    zb, sf = z[:, 0], s[:, 1]
    two_pi = 2.0 * np.pi
    kdp = 1e-3 * (180.0 / np.pi) * wavelength * (sf[:, 1, 1] - sf[:, 0, 0]).real
    data = {
        "sigma_h": (
            two_pi * (zb[:, 0, 0] - zb[:, 0, 1] - zb[:, 1, 0] + zb[:, 1, 1]),
            "mm2",
        ),
        "sigma_v": (
            two_pi * (zb[:, 0, 0] + zb[:, 0, 1] + zb[:, 1, 0] + zb[:, 1, 1]),
            "mm2",
        ),
        "copol_re": (two_pi * (zb[:, 2, 2] + zb[:, 3, 3]), "mm2"),
        "copol_im": (two_pi * (zb[:, 3, 2] - zb[:, 2, 3]), "mm2"),
        "kdp": (kdp, "degree km-1"),
        "ah": (4.343e-3 * 2.0 * wavelength * sf[:, 1, 1].imag, "dB km-1"),
        "av": (4.343e-3 * 2.0 * wavelength * sf[:, 0, 0].imag, "dB km-1"),
    }
    out = xr.Dataset(
        {k: ("diameter", v, {"units": u}) for k, (v, u) in data.items()},
        coords={"diameter": ("diameter", diameter, {"units": "mm"})},
    )
    out["nmax"] = ("diameter", nmax)
    out.attrs.update(
        wavelength_mm=float(wavelength),
        temperature_celsius=float(temperature),
        refractive_index_real=complex(m).real,
        refractive_index_imag=complex(m).imag,
        canting_std_degrees=float(canting_std),
        method="T-matrix, horizontal incidence, Gaussian canting, per drop per m3",
    )
    return out
