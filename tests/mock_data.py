"""
===================
Mock Data Generator
===================

# Example usage
elevation_angles = [0.5, 1.0, 1.5]
elevation_numbers = [0, 1, 2]
start_times = pd.date_range("2023-01-01 00:00:00", freq='1min', periods=3)

ds_mock_list = []
for elev_angle, elev_num, start_time in zip(elevation_angles, elevation_numbers, start_times):
    ds_mock = create_mock_radar_dataset(
        elevation_angle=elev_angle,
        elevation_number=elev_num,
        start_time=start_time
    )
    ds_mock_list.append(ds_mock)

# Combine the individual datasets into a list
for i, ds in enumerate(ds_mock_list):
    print(f"Sweep {i+1} dataset:")
    print(ds)
"""

import numpy as np
import xarray as xr
import pandas as pd


def create_mock_radar_dataset(
    radial_dim=360,
    bin_dim=831,
    elevation_angle=0.5,
    elevation_number=0,
    start_time=None,
    time_step="1s",
):
    """
    Create a mock radar dataset for testing purposes.

    Parameters:
    radial_dim: int - Number of radials (azimuths)
    bin_dim: int - Number of bins (range gates)
    elevation_angle: float - Elevation angle for the sweep
    elevation_number: int - Elevation number for the sweep
    start_time: pd.Timestamp - Start time for the sweep (esStartTime)
    time_step: str - Time step between radials (e.g., '1s')

    Returns:
    xr.Dataset - Mock radar dataset with the specified parameters.
    """

    if start_time is None:
        start_time = pd.Timestamp.now()

    # Generate radial azimuth and radial elevation
    radialAzim = np.linspace(0, 360, radial_dim, endpoint=False)
    radialElev = np.full(radial_dim, elevation_angle)

    # Create radialTime with a constant time for all radials
    radialTime = pd.date_range(start=start_time, periods=radial_dim, freq=time_step)

    # Create a sweep dimension (e.g., 1 for single sweep)
    sweep_dim = 1
    sweep_list = np.arange(1, sweep_dim + 1)

    # Create mock variables
    variables = {
        "siteLat": xr.DataArray(28.61, dims=()),  # Scalar value
        "siteLon": xr.DataArray(77.23, dims=()),  # Scalar value
        "siteAlt": xr.DataArray(200.0, dims=()),  # Scalar value
        "firstGateRange": xr.DataArray(600.0, dims=()),  # Scalar value
        "gateSize": xr.DataArray(300.0, dims=()),  # Scalar value
        "nyquist": xr.DataArray(25.0, dims=()),  # Scalar value
        "unambigRange": xr.DataArray(250.0, dims=()),  # Scalar value
        "elevationList": xr.DataArray(
            [elevation_angle], dims="sweep", coords={"sweep": sweep_list}
        ),
        "angleResolution": xr.DataArray(
            [1.0], dims="sweep", coords={"sweep": sweep_list}
        ),  # Proper angleResolution
    }

    # Add the variables that vary with each sweep
    variables.update(
        {
            "esStartTime": xr.DataArray(
                start_time, dims=()
            ),  # Single start time for the sweep
            "elevationNumber": xr.DataArray(
                [elevation_number], dims="sweep", coords={"sweep": sweep_list}
            ),
            "elevationAngle": xr.DataArray(
                [elevation_angle], dims="sweep", coords={"sweep": sweep_list}
            ),
        }
    )

    # Use "radial" and "bin" as dimension names to match the expected structure
    variables.update(
        {
            "radialAzim": xr.DataArray(radialAzim, dims="radial"),
            "radialElev": xr.DataArray(radialElev, dims="radial"),
            "radialTime": xr.DataArray(radialTime, dims="radial"),  # Correct radialTime
            "T": xr.DataArray(
                np.random.uniform(-127, 127, (radial_dim, bin_dim)),
                dims=["radial", "bin"],
            ),
            "Z": xr.DataArray(
                np.random.uniform(-127, 127, (radial_dim, bin_dim)),
                dims=["radial", "bin"],
            ),
            "V": xr.DataArray(
                np.random.uniform(-127, 127, (radial_dim, bin_dim)),
                dims=["radial", "bin"],
            ),
            "W": xr.DataArray(
                np.random.uniform(-127, 127, (radial_dim, bin_dim)),
                dims=["radial", "bin"],
            ),
            "ZDR": xr.DataArray(
                np.random.uniform(-127, 127, (radial_dim, bin_dim)),
                dims=["radial", "bin"],
            ),
            "KDP": xr.DataArray(
                np.random.uniform(-127, 127, (radial_dim, bin_dim)),
                dims=["radial", "bin"],
            ),
            "PHIDP": xr.DataArray(
                np.random.uniform(-127, 127, (radial_dim, bin_dim)),
                dims=["radial", "bin"],
            ),
            "SQI": xr.DataArray(
                np.random.uniform(-127, 127, (radial_dim, bin_dim)),
                dims=["radial", "bin"],
            ),
            "RHOHV": xr.DataArray(
                np.random.uniform(-127, 127, (radial_dim, bin_dim)),
                dims=["radial", "bin"],
            ),
            "HCLASS": xr.DataArray(
                np.random.uniform(-127, 127, (radial_dim, bin_dim)),
                dims=["radial", "bin"],
            ),
        }
    )

    # Create the dataset
    ds = xr.Dataset(variables)

    # Add global attributes
    ds.attrs["history"] = "Generated as mock data"
    ds.attrs["title"] = "Mock Radar Data"
    ds.attrs["Conventions"] = "CF/Radial instrument_parameters"

    return ds
