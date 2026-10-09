"""
Radar Geometry Calculations
============================

Functions related to beam propagation, height, and sampling volume estimation.

The beam path uses the 4/3 effective Earth radius model [1]_ [2]_ (variability of the gradient: [3]_). The sample
volume convention is discussed in :mod:`radarx.fundamentals.beam` (issue
#173).

References
----------
.. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
       Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
       DOI). Eqs. (2.27)-(2.28d), pp. 21-22; Eqs. (4.13)-(4.16), pp. 74-75.
.. [2] Rinehart, R. E., 1991: *Radar for Meteorologists*, 2nd ed. (3rd
       printing 1994). Rinehart Publications, Grand Forks, ND, ISBN
       0-9608700-7-5 (book, no DOI). Chapter 3, pp. 42-43.
.. [3] Bech, J., B. Codina, J. Lorente, and D. Bebbington, 2003: The
       sensitivity of single polarization weather radar beam blockage
       correction to variability in the vertical refractivity gradient.
       *J. Atmos. Oceanic Technol.*, **20** (6), 845-855,
       https://doi.org/10.1175/1520-0426(2003)020<0845:TSOSPW>2.0.CO;2
       (cited for the variability of dn/dh only; not on disk, content not
       checked).

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = [
    "beam_center_height",
    "effective_radius",
    "half_power_radius",
    "sample_volume_gaussian",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np

from .constants import EARTH_RADIUS, EFFECTIVE_RADIUS_4_3


def effective_radius(dndh=-39e-6):
    """
    Compute effective Earth radius considering atmospheric refraction.

    Implements ``a_e = 1 / (1 / a + dn/dh)``, which is the equivalent-Earth
    relation ``a_e = a / (1 + a dn/dh)``, Eq. (2.27) of [1]_ (p. 21), with
    ``a = EARTH_RADIUS / 1000`` km (a radarx choice, see
    :mod:`radarx.fundamentals.constants`). The standard atmosphere has
    ``dn/dh = -1/(4a)`` giving ``a_e = 4/3 a`` [1]_ Eq. (2.28d); the default
    ``dn/dh = -39e-6 per km`` (= -39 N-units/km, the standard refraction value
    of Rinehart [2]_ pp. 42-43) gives ``a_e`` about 1.33 ``a``. The real
    gradient varies, see [3]_.

    Parameters
    ----------
    dndh : float
        Vertical gradient of the refractive index n per km, default
        ``-39e-6`` per km, i.e. -39 N-units per km (the old docstring unit
        "N-units/km" was inconsistent with the number).

    Returns
    -------
    float
        Effective radius of Earth [m]

    References
    ----------
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    .. [2] Rinehart, R. E., 1991: *Radar for Meteorologists*, 2nd ed. (3rd
           printing 1994). Rinehart Publications, Grand Forks, ND, ISBN
           0-9608700-7-5 (book, no DOI).
    .. [3] Bech, J., B. Codina, J. Lorente, and D. Bebbington, 2003: The
           sensitivity of single polarization weather radar beam blockage
           correction to variability in the vertical refractivity gradient.
           *J. Atmos. Oceanic Technol.*, **20** (6), 845-855,
           https://doi.org/10.1175/1520-0426(2003)020<0845:TSOSPW>2.0.CO;2
    """
    return (1.0 / ((1 / (EARTH_RADIUS / 1000.0)) + dndh)) * 1000.0


def beam_center_height(
    range_m, elevation_deg, radar_height=0.0, reff=EFFECTIVE_RADIUS_4_3
):
    """
    Calculate height of beam center above sea level.

    Implements ``h = sqrt(r**2 + reff**2 + 2 r reff sin(theta_e)) - reff``,
    Eq. (2.28b) of [1]_ (p. 21) with ``reff = k_e a``, plus ``radar_height``.
    The default ``reff`` is 4/3 times the mean Earth radius, Eq. (2.28d)
    [1]_. Valid for a constant refractivity gradient and small ``dh/ds``
    (the two limitations listed after Eq. 2.28d in [1]_).

    Parameters
    ----------
    range_m : float or array-like
        Slant range from radar [m]
    elevation_deg : float
        Elevation angle [degrees]
    radar_height : float
        Radar site altitude [m]
    reff : float
        Effective Earth radius [m]

    Returns
    -------
    float or array-like
        Beam center height [m]

    References
    ----------
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    """
    elev_rad = np.deg2rad(elevation_deg)
    term = np.sqrt(range_m**2 + reff**2 + 2 * range_m * reff * np.sin(elev_rad))
    return term - reff + radar_height


def sample_volume_gaussian(range_m, beamwidth_h_deg, beamwidth_v_deg, pulse_length_m):
    """
    Compute radar sample volume assuming Gaussian beam shape.

    Returns ``pi r**2 theta_h theta_v L / (16 ln 2)`` with ``L`` =
    ``pulse_length_m`` (issue #173). The effective volume of a Gaussian beam
    and rectangular pulse is ``V_e = pi r**2 theta phi h / (8 ln 2)`` with
    ``h = c tau / 2`` (Eqs. 4.13, 4.14, 4.16 of [1]_, pp. 74-75). Hence this
    function equals ``V_e`` only if ``pulse_length_m`` is the full pulse
    length in space ``c tau``; if the range depth ``c tau / 2`` is passed the
    result is ``V_e / 2``. See also [2]_ for the original Gaussian-beam
    derivation (content not checked here). The beamwidths are the half-power
    (3 dB, one-way) widths.

    Parameters
    ----------
    range_m : float or array-like
        Distance to sample volume [m]
    beamwidth_h_deg : float
        Horizontal beamwidth [degrees]
    beamwidth_v_deg : float
        Vertical beamwidth [degrees]
    pulse_length_m : float
        Pulse length in space ``c * tau`` [m] (NOT the range depth
        ``c * tau / 2``) for the result to equal the effective volume

    Returns
    -------
    float or array-like
        Effective sample volume [m³] (if ``pulse_length_m = c tau``)

    References
    ----------
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    .. [2] Probert-Jones, J. R., 1962: The radar equation in meteorology.
           *Q. J. R. Meteorol. Soc.*, **88** (378), 485-495,
           https://doi.org/10.1002/qj.49708837810
    """
    bwh_rad = np.deg2rad(beamwidth_h_deg)
    bwv_rad = np.deg2rad(beamwidth_v_deg)
    numerator = np.pi * range_m**2 * bwh_rad * bwv_rad * pulse_length_m
    return numerator / (16.0 * np.log(2.0))


def half_power_radius(range_m, beamwidth_half_deg):
    """
    Compute half-power beam radius.

    Returns ``range_m * theta / 2``: half of the cross-beam arc length
    subtended by the full half-power beamwidth ``theta`` (the argument is the
    full 3 dB width despite its name). Elementary geometry, no source; the
    beam shape is not used.

    Parameters
    ----------
    range_m : float or array-like
        Range from radar [m]
    beamwidth_half_deg : float
        Half-power beamwidth [degrees]

    Returns
    -------
    float or array-like
        Half-power radius [m]
    """
    return (range_m * np.deg2rad(beamwidth_half_deg)) / 2.0
