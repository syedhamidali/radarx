"""
Beam Geometry and Resolution
============================

Functions related to radar beamwidth and spatial resolution.

Sample-volume conventions (issue #173)
--------------------------------------
Three different "sample volumes" are implemented in radarx and none of them
is labelled in the code. For a Gaussian beam with half-power widths theta
and phi (radians) and a rectangular pulse of transmitted duration tau, the
Probert-Jones / Doviak and Zrnic effective (reflectivity-weighting) volume is

    V_e = pi r**2 theta phi h / (8 ln 2),   h = c tau / 2,

obtained from the weather radar equation, Eqs. (4.13), (4.14) and (4.16) of
[1]_ (pp. 74-75; the book prints it for a circular beam, theta_1**2, and the
unequal-width form is the same expression with theta_1**2 replaced by
theta phi); see also [2]_.

- :func:`compute_volume_resolution` returns ``(r theta)**2 * pulse_length``,
  a square-section geometric box of side ``r theta`` and length
  ``pulse_length``. It contains no ``ln 2`` and equals none of the standard
  definitions. ``pulse_length`` is used as given (a range extent in m).
- :func:`volume_resolution` returns
  ``r**2 theta phi pulse_length / (4 ln 2)``. With ``pulse_length = c tau / 2``
  this is ``(2 / pi) V_e`` (about 0.64 V_e); the origin of the ``4 ln 2``
  convention could not be traced to [1]_ or [2]_.
- :func:`radarx.fundamentals.geometry.sample_volume_gaussian` returns
  ``pi r**2 theta phi pulse_length / (16 ln 2)``, which equals ``V_e`` only if
  ``pulse_length`` is the full ``c tau`` (the pulse length in space); with
  ``pulse_length = c tau / 2`` it is ``V_e / 2``.

The behaviour is unchanged; use ``sample_volume_gaussian(..., c * tau)`` when
the effective volume is wanted.

References
----------
.. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
       Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
       DOI).
.. [2] Probert-Jones, J. R., 1962: The radar equation in meteorology.
       *Q. J. R. Meteorol. Soc.*, **88** (378), 485-495,
       https://doi.org/10.1002/qj.49708837810 (not on disk; cited as the
       origin of the Gaussian-beam effective volume, its content was not
       checked here).

.. autosummary::
   :nosignatures:
   :toctree: generated/

    {}
"""

__all__ = [
    "azimuthal_resolution",
    "beamwidth_to_radians",
    "compute_azimuth_resolution",
    "compute_beamwidth",
    "compute_volume_resolution",
    "volume_resolution",
]

__doc__ = __doc__.format("\n   ".join(__all__))


def compute_beamwidth(wavelength, antenna_diameter):
    """
    Compute the beamwidth of a radar.

    Returns ``1.22 * wavelength / antenna_diameter``. This is the angular
    radius of the first null of the Airy diffraction pattern of a uniformly
    illuminated circular aperture (the Rayleigh resolution angle), NOT the
    half-power (3 dB) beamwidth. For a tapered paraboloid illumination [1]_
    gives the 3 dB beamwidth as ``1.27 * wavelength / D`` rad, Eq. (3.2b)
    (p. 34), and the first-null width as ``3.27 * wavelength / D``. The
    first-null angle formula of [1]_ is Eq. (2.4) (Section 2.1); the digits
    printed in the text extraction used here could not be read reliably, so
    the 1.22 is not claimed to be Eq. (2.4). The previously cited "Eq. 3.5.2"
    could not be found in [1]_.

    Parameters
    ----------
    wavelength : float
        Radar wavelength [m].
    antenna_diameter : float
        Diameter of the radar antenna [m].

    Returns
    -------
    float
        Angle ``1.22 lambda / D`` in radians (first-null radius, see above).

    References
    ----------
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    """
    return 1.22 * wavelength / antenna_diameter


def compute_azimuth_resolution(range_m, beamwidth_rad):
    """
    Compute azimuthal resolution (cross-range resolution).

    Returns ``range_m * beamwidth_rad``, the arc length subtended by the
    beamwidth at range ``range_m`` (small-angle geometry). With the 3 dB
    beamwidth this is the cross-beam width of the resolution volume [1]_
    (Section 4.4). No equation number is claimed.

    Parameters
    ----------
    range_m : float
        Radar range [m].
    beamwidth_rad : float
        Beamwidth [radians].

    Returns
    -------
    float
        Azimuthal resolution [m].

    References
    ----------
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    """
    return range_m * beamwidth_rad


def compute_volume_resolution(range_m, beamwidth_rad, pulse_length):
    """
    Compute radar volume resolution.

    Returns ``(range_m * beamwidth_rad)**2 * pulse_length``: the volume of a
    box of square cross-section ``r theta`` by ``r theta`` and length
    ``pulse_length``. It is NOT the effective sample volume
    ``V_e = pi r**2 theta phi h / (8 ln 2)``, ``h = c tau / 2``, of Eqs.
    (4.13)-(4.16) of [1]_ (see the module docstring and issue #173). The
    previously cited "Eq. 3.5.7" could not be found in [1]_.

    Parameters
    ----------
    range_m : float
        Radar range [m].
    beamwidth_rad : float
        Beamwidth [radians].
    pulse_length : float
        Range extent of the volume [m], used as given (``c tau / 2`` for the
        usual resolution depth).

    Returns
    -------
    float
        Geometric box volume [m^3].

    References
    ----------
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    """
    az_res = compute_azimuth_resolution(range_m, beamwidth_rad)
    return az_res * az_res * pulse_length


def beamwidth_to_radians(beamwidth_deg):
    """
    Convert beamwidth from degrees to radians (``numpy.deg2rad``; unit
    conversion, no source needed).

    Parameters
    ----------
    beamwidth_deg : float
        Beamwidth in degrees.

    Returns
    -------
    float
        Beamwidth in radians.
    """
    import numpy as np

    return np.deg2rad(beamwidth_deg)


def azimuthal_resolution(range_m, beamwidth_deg):
    """
    Compute azimuthal resolution given beamwidth in degrees.

    Same as :func:`compute_azimuth_resolution` after converting the beamwidth
    to radians (small-angle arc length ``r * theta``).

    Parameters
    ----------
    range_m : float
        Radar range [m].
    beamwidth_deg : float
        Beamwidth [degrees].

    Returns
    -------
    float
        Azimuthal resolution [m].
    """
    bw_rad = beamwidth_to_radians(beamwidth_deg)
    return compute_azimuth_resolution(range_m, bw_rad)


def volume_resolution(range_m, bw_h_deg, bw_v_deg, pulse_length):
    """
    Compute radar volume resolution using horizontal and vertical beamwidth in degrees.

    Returns ``r**2 theta_h theta_v pulse_length / (4 ln 2)`` (angles in
    radians). Which volume this is, and the meaning of ``pulse_length``, are
    not defined by any source: with ``pulse_length = c tau / 2`` it equals
    ``(2 / pi)`` times the effective volume ``V_e = pi r**2 theta phi h /
    (8 ln 2)`` of Eqs. (4.13)-(4.16) of [1]_ (issue #173; see the module
    docstring). The ``4 ln 2`` form could not be traced to a source.

    Parameters
    ----------
    range_m : float
        Radar range [m].
    bw_h_deg : float
        Horizontal beamwidth [degrees].
    bw_v_deg : float
        Vertical beamwidth [degrees].
    pulse_length : float
        Range extent [m] of the volume, used as given.

    Returns
    -------
    float
        Radar sampling volume [m^3] in the (untraced) convention above.

    References
    ----------
    .. [1] Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather
           Observations*, 2nd ed. Academic Press, ISBN 0-12-221422-6 (book, no
           DOI).
    """
    import numpy as np

    bw_h_rad = np.deg2rad(bw_h_deg)
    bw_v_rad = np.deg2rad(bw_v_deg)
    return range_m**2 * bw_h_rad * bw_v_rad * pulse_length / (4 * np.log(2))
