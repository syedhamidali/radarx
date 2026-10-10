"""
Rayleigh Scattering Approximations
==================================

Functions for computing backscatter cross-sections, size parameters,
and absorption, scattering, and extinction coefficients under Rayleigh
scattering assumptions.

Conventions: the efficiencies ``Qa``, ``Qs``, ``Qe`` here are cross sections
divided by the geometric cross section ``pi a**2`` (``a`` = radius), with
size parameter ``x = 2 pi a / lambda``, and are built from the Rayleigh-limit
cross sections of Bringi and Chandrasekar [1]_. The absorption uses
``|Im(K)|``, so the refractive index may be given in either sign convention,
``m = n - j k`` (radar meteorology, [1]_ and
:mod:`radarx.fundamentals.attenuation`) or ``m = n + j k``. The ``dielectric`` argument of
:func:`backscatter_cross_section` is the dielectric factor ``|K|**2`` itself
(the literature values of |K|**2 are in [2]_).

References
----------
.. [1] Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler
       Weather Radar: Principles and Applications*. Cambridge University
       Press, https://doi.org/10.1017/CBO9780511541094 (book; ISBN
       0-521-62384-7). Eqs. (1.51b), (1.52), (1.57), (1.59), (2.132b),
       (2.133d), (2.134c); Sections 1.4, 1.5, 2.5.
.. [2] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
       Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
       DOI). Rayleigh approximation Eq. (3.6) and the |K|^2 values, p. 36.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = [
    "backscatter_cross_section",
    "normalized_backscatter_cross_section",
    "size_parameter",
    "absorption_coefficient",
    "scattering_coefficient",
    "extinction_coefficient",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np


def backscatter_cross_section(diameter, wavelength, dielectric=0.93):
    """
    Rayleigh backscatter cross-section for a water sphere.

    Implements ``pi**5 * dielectric * D**6 / lambda**4``, the Rayleigh
    backscatter cross section ``pi**5 |K|**2 D**6 / lambda**4`` with
    ``|K|**2 = |(eps - 1)/(eps + 2)|**2`` (about 0.93 for water), Eq. (1.51b)
    of [1]_ (p. 17; Eq. 3.6 of [2]_, p. 35-36). ``dielectric`` is ``|K|**2``
    itself (the literature value of [2]_), not ``|K|``. Before this release
    the argument was squared, so the default 0.93 gave 0.93 times the textbook
    cross section (0.3 dB low); values were also wrong for any other
    ``dielectric`` unless it was passed as ``|K|``.

    Parameters
    ----------
    diameter : float or array-like
        Drop diameter [m]
    wavelength : float
        Radar wavelength [m]
    dielectric : float
        Dielectric factor ``|K|**2`` (default: 0.93 for water)

    Returns
    -------
    float or array-like
        Backscatter cross-section [m^2]

    References
    ----------
    .. [1] Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler
           Weather Radar: Principles and Applications*. Cambridge University
           Press, https://doi.org/10.1017/CBO9780511541094 (book; ISBN
           0-521-62384-7).
    .. [2] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    """
    d = np.asarray(diameter)
    return (np.pi**5 * dielectric * d**6) / wavelength**4


def normalized_backscatter_cross_section(diameter, wavelength, dielectric=0.93):
    """
    Normalize Rayleigh backscatter cross-section by projected area.

    ``sigma_b / (pi D**2 / 4)``: the backscatter cross section divided by
    the geometric cross-section of the sphere, the normalization used for
    the "normalized radar cross section" of [1]_ (Section 2.5, Fig. 2.15;
    their normalized extinction cross section is ``sigma_ext / (pi a**2)``).
    ``dielectric`` is ``|K|**2`` as in :func:`backscatter_cross_section`.

    Parameters
    ----------
    diameter : float or array-like
        Drop diameter [m]
    wavelength : float
        Radar wavelength [m]
    dielectric : float, optional
        Dielectric factor ``|K|**2`` (default: 0.93)

    Returns
    -------
    float or array-like
        Normalized cross-section [unitless]

    References
    ----------
    .. [1] Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler
           Weather Radar: Principles and Applications*. Cambridge University
           Press, https://doi.org/10.1017/CBO9780511541094 (book; ISBN
           0-521-62384-7).
    """
    sigma = backscatter_cross_section(diameter, wavelength, dielectric)
    area = np.pi * (np.asarray(diameter) / 2.0) ** 2
    return sigma / area


def size_parameter(diameter, wavelength):
    """
    Calculate size parameter alpha = π * D / λ.

    Equals ``k0 * a = 2 pi a / lambda`` with radius ``a = D/2``, the "size"
    parameter ``rho_0`` of Section 2.5 of [1]_.

    Parameters
    ----------
    diameter : float or array-like
        Drop diameter [m]
    wavelength : float
        Radar wavelength [m]

    Returns
    -------
    float or array-like
        Size parameter (unitless)

    References
    ----------
    .. [1] Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler
           Weather Radar: Principles and Applications*. Cambridge University
           Press, https://doi.org/10.1017/CBO9780511541094 (book; ISBN
           0-521-62384-7).
    """
    return np.pi * np.asarray(diameter) / wavelength


def _size_parameter(radius, wavelength):
    """
    Calculate size parameter x = 2π * radius / wavelength.

    ``k0 a``, Section 2.5 of [1]_ (same quantity as :func:`size_parameter`
    with the diameter replaced by the radius).

    Parameters
    ----------
    radius : float or array-like
        Particle radius [m]
    wavelength : float
        Radar wavelength [m]

    Returns
    -------
    float or array-like
        Size parameter (unitless)

    References
    ----------
    .. [1] Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler
           Weather Radar: Principles and Applications*. Cambridge University
           Press, https://doi.org/10.1017/CBO9780511541094 (book; ISBN
           0-521-62384-7).
    """
    return 2 * np.pi * np.asarray(radius) / wavelength


def _complex_ratio(refractive_index):
    """
    Calculate the complex ratio (m^2 - 1) / (m^2 + 2).

    The factor ``K = (eps - 1) / (eps + 2)`` with ``eps = m**2``, Eq. (1.51b)
    and the text below it in [1]_ (its squared modulus is the "dielectric
    factor" ``|K|**2``).

    Parameters
    ----------
    refractive_index : complex
        Complex refractive index of the particle

    Returns
    -------
    complex or array-like of complex
        Complex ratio used in scattering calculations

    References
    ----------
    .. [1] Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler
           Weather Radar: Principles and Applications*. Cambridge University
           Press, https://doi.org/10.1017/CBO9780511541094 (book; ISBN
           0-521-62384-7).
    """
    m = refractive_index
    m2 = m * m
    return (m2 - 1) / (m2 + 2)


def absorption_coefficient(radius, wavelength, refractive_index):
    """
    Compute Rayleigh absorption coefficient.

    ``Qa = 4 x |Im(K)|`` with ``x = 2 pi a / lambda``. This is
    the absorption cross section ``sigma_a = 9 k0 V eps'' / |eps + 2|**2``,
    Eq. (1.59) of [1]_ (consistent with Eq. 2.134c), divided by ``pi a**2``
    (the ``4 x |Im(K)|`` form is derived here and is not printed in [1]_).
    ``Im(K)`` is negative for ``m = n - j k`` ([1]_, ``eps = eps' - j eps''``)
    and positive for ``m = n + j k``; the absolute value makes the result
    independent of the convention. Before this release only ``n + j k`` was
    accepted and ``n - j k`` silently gave 0.

    Parameters
    ----------
    radius : float or array-like
        Particle radius [m]
    wavelength : float
        Radar wavelength [m]
    refractive_index : complex
        Complex refractive index of the particle

    Returns
    -------
    float or array-like
        Absorption efficiency (Qa)

    References
    ----------
    .. [1] Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler
           Weather Radar: Principles and Applications*. Cambridge University
           Press, https://doi.org/10.1017/CBO9780511541094 (book; ISBN
           0-521-62384-7).
    """
    x = 2 * np.pi * radius / wavelength
    m = refractive_index
    m2 = m * m
    return 4 * x * np.abs(np.imag((m2 - 1) / (m2 + 2)))


def scattering_coefficient(radius, wavelength, refractive_index):
    """
    Compute Rayleigh scattering coefficient.

    ``Qs = (8/3) x**4 |K|**2``: the total scattering cross section
    ``sigma_s = (8 pi / 3) k0**4 a**6 |K|**2``, Eq. (2.132b) of [1]_ (same as
    Eq. 1.52), divided by ``pi a**2``.

    Parameters
    ----------
    radius : float or array-like
        Particle radius [m]
    wavelength : float
        Radar wavelength [m]
    refractive_index : complex
        Complex refractive index of the particle

    Returns
    -------
    float or array-like
        Scattering efficiency (Qs)

    References
    ----------
    .. [1] Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler
           Weather Radar: Principles and Applications*. Cambridge University
           Press, https://doi.org/10.1017/CBO9780511541094 (book; ISBN
           0-521-62384-7).
    """
    x = _size_parameter(radius, wavelength)
    ratio = _complex_ratio(refractive_index)
    return (8 / 3) * x**4 * np.abs(ratio) ** 2


def extinction_coefficient(radius, wavelength, refractive_index):
    """
    Compute Rayleigh extinction coefficient (Qa + Qs).

    ``Qe = Qa + Qs``, Eq. (1.57) of [1]_, clipped at zero. In the Rayleigh
    limit [1]_ notes the sum is correct only to order ``(D/lambda)**3``.

    Parameters
    ----------
    radius : float or array-like
        Particle radius [m]
    wavelength : float
        Radar wavelength [m]
    refractive_index : complex
        Complex refractive index of the particle

    Returns
    -------
    float or array-like
        Extinction efficiency (Qe)

    References
    ----------
    .. [1] Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler
           Weather Radar: Principles and Applications*. Cambridge University
           Press, https://doi.org/10.1017/CBO9780511541094 (book; ISBN
           0-521-62384-7).
    """
    return np.maximum(
        absorption_coefficient(radius, wavelength, refractive_index)
        + scattering_coefficient(radius, wavelength, refractive_index),
        0.0,
    )
