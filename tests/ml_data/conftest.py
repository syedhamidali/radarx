"""Synthetic sweeps and volumes for the ``ml/data`` label builders."""

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

ML_DATA = Path(__file__).resolve().parents[2] / "ml" / "data"
if str(ML_DATA) not in sys.path:
    sys.path.insert(0, str(ML_DATA))


def synthetic_sweep(
    nray=360,
    ngate=400,
    elevation=0.5,
    wind=(12.0, 8.0),
    nyquist=30.0,
    seed=0,
    t0="2022-03-30T23:46:00",
    polarimetric=True,
    velocity=True,
    azimuth_offset=0.5,
    sector=60.0,
    clutter=True,
):
    """
    A PPI sweep in the xradar layout.

    A rain region (40-55 dBZ, ZDR ~1.5 dB, RHOHV ~0.98, PHIDP increasing
    with range) between 20 and 70 km, a clutter patch near the radar
    (low RHOHV, noisy ZDR) and a uniform wind ``wind`` (u, v) as radial
    velocity, unaliased for the default Nyquist velocity. ``sector`` is the
    half width (degrees) of the rain region around 200 degrees azimuth.
    """
    rng = np.random.default_rng(seed)
    az = (np.arange(nray) + azimuth_offset) * 360.0 / nray
    r = 2125.0 + 250.0 * np.arange(ngate)
    A, R = np.meshgrid(az, r, indexing="ij")
    rain = (R > 20e3) & (R < 70e3) & (np.abs(((A - 200.0) + 180) % 360 - 180) <= sector)
    clutter = clutter & (R < 10e3) & (np.abs(((A - 40.0) + 180) % 360 - 180) < 20)
    dbz = np.where(rain, 40.0 + 15.0 * np.sin(R / 7e3) ** 2, np.nan)
    dbz = np.where(clutter, 50.0 + 10.0 * rng.standard_normal(A.shape), dbz)
    dbz = dbz + rng.normal(0, 0.5, A.shape)
    zdr = np.where(rain, 1.5, np.nan) + rng.normal(0, 0.2, A.shape)
    zdr = np.where(clutter, rng.normal(0, 4.0, A.shape), zdr)
    rho = np.where(rain, 0.985, np.nan) + rng.normal(0, 0.005, A.shape)
    rho = np.where(clutter, rng.uniform(0.3, 0.8, A.shape), rho)
    phi = np.where(rain, 30.0 + 1.0 * np.clip(R - 20e3, 0, None) / 1e3, np.nan)
    phi = phi + rng.normal(0, 2.0, A.shape)
    phi = np.where(clutter, rng.uniform(0, 360, A.shape), phi)
    u, v = wind
    el = np.deg2rad(elevation)
    vr = (u * np.sin(np.deg2rad(A)) + v * np.cos(np.deg2rad(A))) * np.cos(el)
    vr = np.where(np.isfinite(dbz), vr + rng.normal(0, 0.3, A.shape), np.nan)
    times = np.datetime64(t0, "ns") + (np.arange(nray) * 40e6).astype("timedelta64[ns]")
    data = {"DBZH": dbz}
    if polarimetric:
        data.update(ZDR=zdr, RHOHV=rho, PHIDP=phi)
    if velocity:
        data["VRADH"] = vr
    ds = xr.Dataset(
        {k: (("azimuth", "range"), v.astype("float32")) for k, v in data.items()},
        coords={
            "azimuth": ("azimuth", az, {"units": "degrees"}),
            "range": ("range", r.astype("float32"), {"units": "m"}),
            "elevation": ("azimuth", np.full(nray, elevation)),
            "time": ("azimuth", times),
            "latitude": 33.9,
            "longitude": -88.3,
            "altitude": 179.0,
        },
    )
    ds["sweep_fixed_angle"] = elevation
    ds["sweep_mode"] = "azimuth_surveillance"
    if velocity:
        ds = ds.assign_coords(nyquist_velocity=float(nyquist))
    return ds


def synthetic_volume(t0="2022-03-30T23:46:00", shift_km=0.0, seed=0, radar="KTST"):
    """A NEXRAD-like volume: two polarimetric and one Doppler-only sweeps."""
    sweeps = {}
    for i, (el, pol, vel) in enumerate(
        [(0.5, True, False), (0.5, False, True), (1.5, True, True)]
    ):
        ds = synthetic_sweep(
            elevation=el, polarimetric=pol, velocity=vel, seed=seed + i, t0=t0
        )
        if shift_km:
            ds = ds.roll(range=int(shift_km * 4), roll_coords=False)
        ds["sweep_number"] = i
        sweeps[f"sweep_{i}"] = ds.drop_vars(["latitude", "longitude", "altitude"])
    root = xr.Dataset(
        {"time_coverage_start": t0},
        coords={"latitude": 33.9, "longitude": -88.3, "altitude": 179.0},
        attrs={"instrument_name": radar},
    )
    return xr.DataTree.from_dict({"/": root, **sweeps})


@pytest.fixture
def sweep():
    return synthetic_sweep()


@pytest.fixture
def volume():
    return synthetic_volume()
