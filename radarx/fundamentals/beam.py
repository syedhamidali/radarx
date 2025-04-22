"""
Beam Geometry and Resolution
============================

Functions related to radar beamwidth and spatial resolution.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   azimuthal_resolution
   beamwidth_to_radians
   compute_azimuth_resolution
   compute_beamwidth
   compute_volume_resolution
   volume_resolution

References
----------
- Doviak, R. J., & Zrnić, D. S. (1993). Doppler Radar and Weather Observations. Academic Press.
"""

__all__ = [
    "azimuthal_resolution",
    "beamwidth_to_radians",
    "compute_azimuth_resolution",
    "compute_beamwidth",
    "compute_volume_resolution",
    "volume_resolution",
]


def compute_beamwidth(wavelength, antenna_diameter):
    """
    Compute the beamwidth of a radar.

    Parameters
    ----------
    wavelength : float
        Radar wavelength [m].
    antenna_diameter : float
        Diameter of the radar antenna [m].

    Returns
    -------
    float
        Beamwidth in radians.

    References
    ----------
    - Doviak and Zrnić (1993), Eq. 3.5.2
    """
    return 1.22 * wavelength / antenna_diameter


def compute_azimuth_resolution(range_m, beamwidth_rad):
    """
    Compute azimuthal resolution (cross-range resolution).

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
    - Doviak and Zrnić (1993), Section 3.5
    """
    return range_m * beamwidth_rad


def compute_volume_resolution(range_m, beamwidth_rad, pulse_length):
    """
    Compute radar volume resolution.

    Parameters
    ----------
    range_m : float
        Radar range [m].
    beamwidth_rad : float
        Beamwidth [radians].
    pulse_length : float
        Pulse length [m].

    Returns
    -------
    float
        Radar sampling volume [m^3].

    References
    ----------
    - Doviak and Zrnić (1993), Eq. 3.5.7
    """
    az_res = compute_azimuth_resolution(range_m, beamwidth_rad)
    return az_res * az_res * pulse_length


def beamwidth_to_radians(beamwidth_deg):
    """
    Convert beamwidth from degrees to radians.

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

    Parameters
    ----------
    range_m : float
        Radar range [m].
    bw_h_deg : float
        Horizontal beamwidth [degrees].
    bw_v_deg : float
        Vertical beamwidth [degrees].
    pulse_length : float
        Pulse length [m].

    Returns
    -------
    float
        Radar sampling volume [m^3].
    """
    import numpy as np

    bw_h_rad = np.deg2rad(bw_h_deg)
    bw_v_rad = np.deg2rad(bw_v_deg)
    return range_m**2 * bw_h_rad * bw_v_rad * pulse_length / (4 * np.log(2))
