"""
Attenuation and Scattering Coefficients
=======================================

Functions related to absorption, scattering, and extinction cross sections for
spherical particles in the Rayleigh limit (particle diameter much smaller than
the wavelength).

The three ``*_coefficient`` functions return cross sections in m^2, not
dimensionless efficiencies (the docstrings used to say "unitless"; D^3/lambda
and D^6/lambda^4 have units of m^2). The same-named functions in
:mod:`radarx.fundamentals.scattering` take the radius and return efficiencies,
and :mod:`radarx.fundamentals` exports the ``scattering`` versions (they are
imported later). Sign convention: these functions use the engineering
convention of Bringi and Chandrasekar (2001), complex permittivity
``eps = eps' - j eps''`` and refractive index ``m = n - j k`` with ``k > 0``, for
which Im(K) < 0 for an absorbing particle [1]_. Passing ``m = n + j k``
gives a negative absorption coefficient.

References
----------
.. [1] Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler
       Weather Radar: Principles and Applications*. Cambridge University
       Press, https://doi.org/10.1017/CBO9780511541094 (book; ISBN
       0-521-62384-7). Eqs. (1.51b), (1.52), (1.56)-(1.59), (2.132), p. 17-18
       and Section 2.5.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = [
    "absorption_coefficient",
    "extinction_coefficient",
    "k_complex",
    "scattering_coefficient",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np


def k_complex(m):
    """
    Complex dielectric factor used in scattering calculations.

    ``K = (m**2 - 1) / (m**2 + 2)``, the Clausius-Mossotti factor whose
    squared modulus ``|K|**2`` is the "dielectric factor" of the Rayleigh
    backscatter law, Eq. (1.51b) and the text below it in [1]_.

    Parameters
    ----------
    m : complex
        Complex refractive index of the particle (``n - j k`` convention, see
        the module docstring).

    Returns
    -------
    complex
        Dielectric factor K.

    References
    ----------
    .. [1] Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler
           Weather Radar: Principles and Applications*. Cambridge University
           Press, https://doi.org/10.1017/CBO9780511541094 (book; ISBN
           0-521-62384-7).
    """
    return (m**2 - 1) / (m**2 + 2)


def absorption_coefficient(diameter, wavelength, m):
    """
    Absorption cross section Qa of a spherical particle (Rayleigh limit).

    Implements ``Qa = pi**2 * D**3 / lambda * Im(-K)`` with
    ``K = (m**2 - 1) / (m**2 + 2)``. This is the absorption cross section of
    Eq. (1.59) of [1]_, ``9 k0 V eps'' / |eps + 2|**2``, rewritten with the
    sphere volume ``V = pi D**3 / 6`` and ``k0 = 2 pi / lambda``; the
    equivalence was checked algebraically, it is not printed in this form in
    the book. The result is a cross section in m^2, not a dimensionless
    efficiency.

    Parameters
    ----------
    diameter : float or array-like
        Particle diameter [m]
    wavelength : float
        Radar wavelength [m]
    m : complex
        Complex refractive index (``n - j k`` convention)

    Returns
    -------
    float or array-like
        Absorption cross section [m^2]

    References
    ----------
    .. [1] Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler
           Weather Radar: Principles and Applications*. Cambridge University
           Press, https://doi.org/10.1017/CBO9780511541094 (book; ISBN
           0-521-62384-7).
    """
    Km_im = np.imag(-1 * k_complex(m))
    return (np.pi**2 * np.asarray(diameter) ** 3 / wavelength) * Km_im


def scattering_coefficient(diameter, wavelength, m):
    """
    Scattering cross section Qs of a spherical particle (Rayleigh limit).

    Implements ``Qs = 2 pi**5 D**6 |K|**2 / (3 lambda**4)``, which equals
    the total scattering cross section of Eq. (1.52) / (2.132b) of [1]_,
    ``3 k0**4 V**2 |K|**2 / (2 pi)``, with ``V = pi D**3 / 6`` and
    ``k0 = 2 pi / lambda`` (algebraic equivalence checked, not printed in
    this form). The result is a cross section in m^2, not a dimensionless
    efficiency; compare ``sigma_b = pi**5 |K|**2 D**6 / lambda**4``
    (Eq. 1.51b).

    Parameters
    ----------
    diameter : float or array-like
        Particle diameter [m]
    wavelength : float
        Radar wavelength [m]
    m : complex
        Complex refractive index

    Returns
    -------
    float or array-like
        Scattering cross section [m^2]

    References
    ----------
    .. [1] Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler
           Weather Radar: Principles and Applications*. Cambridge University
           Press, https://doi.org/10.1017/CBO9780511541094 (book; ISBN
           0-521-62384-7).
    """
    Km_abs = np.abs(k_complex(m))
    return (2 * np.pi**5 * np.asarray(diameter) ** 6 / (3 * wavelength**4)) * (
        Km_abs**2
    )


def extinction_coefficient(diameter, wavelength, m):
    """
    Extinction cross section Qe (absorption + scattering).

    ``Qe = Qa + Qs``, Eq. (1.57) of [1]_. In the Rayleigh limit the
    absorption term (order D**3) dominates the scattering term (order D**6)
    for absorbing particles.

    Parameters
    ----------
    diameter : float or array-like
        Particle diameter [m]
    wavelength : float
        Radar wavelength [m]
    m : complex
        Complex refractive index

    Returns
    -------
    float or array-like
        Extinction cross section [m^2]

    References
    ----------
    .. [1] Bringi, V. N., and V. Chandrasekar, 2001: *Polarimetric Doppler
           Weather Radar: Principles and Applications*. Cambridge University
           Press, https://doi.org/10.1017/CBO9780511541094 (book; ISBN
           0-521-62384-7).
    """
    Qa = absorption_coefficient(diameter, wavelength, m)
    Qs = scattering_coefficient(diameter, wavelength, m)
    return Qa + Qs
