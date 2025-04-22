"""
Radar Geometry Calculations
============================

Functions related to beam propagation, height, and sampling volume estimation.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   effective_radius
   beam_center_height
   sample_volume_gaussian
   half_power_radius

References
----------
- Rinehart, R. E. (1997). Radar for Meteorologists.
- Bech et al. (2003). JAOT, Beam Blockage Corrections.
"""

import numpy as np
from .constants import EARTH_RADIUS, EFFECTIVE_RADIUS_4_3


def effective_radius(dndh=-39e-6):
    """
    Compute effective Earth radius considering atmospheric refraction.

    Parameters
    ----------
    dndh : float
        Vertical gradient of refractivity [N-units/km], default: -39e-6

    Returns
    -------
    float
        Effective radius of Earth [m]
    """
    return (1.0 / ((1 / (EARTH_RADIUS / 1000.0)) + dndh)) * 1000.0


def beam_center_height(
    range_m, elevation_deg, radar_height=0.0, reff=EFFECTIVE_RADIUS_4_3
):
    """
    Calculate height of beam center above sea level.

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
    """
    elev_rad = np.deg2rad(elevation_deg)
    term = np.sqrt(range_m**2 + reff**2 + 2 * range_m * reff * np.sin(elev_rad))
    return term - reff + radar_height


def sample_volume_gaussian(range_m, beamwidth_h_deg, beamwidth_v_deg, pulse_length_m):
    """
    Compute radar sample volume assuming Gaussian beam shape.

    Parameters
    ----------
    range_m : float or array-like
        Distance to sample volume [m]
    beamwidth_h_deg : float
        Horizontal beamwidth [degrees]
    beamwidth_v_deg : float
        Vertical beamwidth [degrees]
    pulse_length_m : float
        Pulse length [m]

    Returns
    -------
    float or array-like
        Sample volume [m³]
    """
    bwh_rad = np.deg2rad(beamwidth_h_deg)
    bwv_rad = np.deg2rad(beamwidth_v_deg)
    numerator = np.pi * range_m**2 * bwh_rad * bwv_rad * pulse_length_m
    return numerator / (16.0 * np.log(2.0))


def half_power_radius(range_m, beamwidth_half_deg):
    """
    Compute half-power beam radius.

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


__all__ = [
    "beam_center_height",
    "effective_radius",
    "half_power_radius",
    "sample_volume_gaussian",
]
