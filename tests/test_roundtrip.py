#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Round trips of the products through NetCDF and Zarr
===================================================

Every product that can be built from synthetic input is written to NetCDF and
to Zarr and read back, and the result must be identical (values, dtypes,
attributes and coordinates). Products whose input is a downloaded file are
left to the tests that fetch it.
"""

import numpy as np
import pytest
import xarray as xr
from packaging.version import Version

import radarx

from .grid.test_cone import GRID
from .grid.test_cone import _volume as cone_volume
from .grid.test_gates import GRID as GATES_GRID
from .grid.test_gates import _rhi_volume
from .test_advection import T0, _blobs
from .test_beam_blockage import bell, geographic_dem
from .test_beam_blockage import make_sweep as bb_sweep
from .test_cappi import _make_ppi_sweep
from .test_coldpool import column, station_network
from .test_couplets_vertical import COLUMN
from .test_couplets_vertical import volume as couplet_volume
from .test_dealias import synthetic_sweep
from .test_dealias import volume as dealias_volume
from .test_dealias_extras import VH, VL, dual_prf_sweep
from .test_disdrometer import _station
from .test_dsd import _sweep as dsd_sweep
from .test_dsd import _volume as dsd_volume
from .test_dsd_bayes import _volume as bayes_volume
from .test_ensemble import _rain_ds as ensemble_sweep
from .test_evaporation import _profile as evap_profile
from .test_evaporation import _series as evap_series
from .test_forward_operator import SITE as FO_SITE
from .test_forward_operator import synthetic_mpas, synthetic_wrf
from .test_hid import _random_volume as hid_volume
from .test_kdp import _sweep as kdp_sweep
from .test_kdp import _volume as kdp_volume
from .test_lagrangian import _point, _winds
from .test_lightning import DAY, _mask, _rate, _sources
from .test_lightning import _grid as lightning_grid
from .test_qc import _sweep as qc_sweep
from .test_qvp import _melting_profile, _profiles_dataset
from .test_qvp import _volume as qvp_volume
from .test_rain_trajectories import _source_grid
from .test_rain_trajectories import profile as rain_profile
from .test_shear_layer import column_volume
from .test_tc_winds import CENTER as TC_CENTER
from .test_tc_winds import small_tc_center, small_vortex
from .test_tornado import rankine
from .test_vil import storm_volume
from .test_wind_profile import vad_sweep

zarr = pytest.importorskip("zarr")

# xarray before 2025.1 cannot write Zarr format 3, which zarr 3 uses
NO_ZARR_3 = Version(xr.__version__) < Version("2025.1") and Version(
    zarr.__version__
) >= Version("3")

R = radarx.retrieve


def _hid(**kwargs):
    ds = hid_volume(n=1)[0].drop_vars("MET")
    return R.hid(ds, temperature="TEMP", **kwargs)


def _dealiased():
    sweep, _ = synthetic_sweep(nray=60, ngate=200)
    return sweep


def _grid():
    return radarx.grid.grid_cones(cone_volume(field="random"), "DBZH", **GRID)


def _wind():
    height = np.arange(0.0, 8000.0, 100.0)
    return xr.Dataset(
        {"u": ("height", height / 600.0), "v": ("height", 4.0 + 0 * height)},
        coords={"height": height},
    )


def _sounding():
    levels = np.array([1000.0, 900.0, 800.0])
    return xr.Dataset(
        {
            "temperature": ("level", [295.0, 288.0, 281.0]),
            "pressure": ("level", levels * 100.0),
            "dewpoint": ("level", [290.0, 283.0, 277.0]),
        }
    )


def _moved():
    return R.advect(_blobs(0, 8.0, -6.0), 8.0, -6.0, 300.0)


def _cappi_volume():
    return xr.DataTree.from_dict(
        {
            f"sweep_{k}": _make_ppi_sweep(el, "2024-01-01T00:00:00")
            for k, el in enumerate((1.0, 3.0, 5.0))
        }
    )


def _dsd_fit():
    return R.dsd(dsd_sweep()[0], "normalized", band="S")


def _nd():
    return R.number_concentration(R.disdrometer_qc(_station()[0]))


def _qc_station():
    return R.disdrometer_qc(_station()[0])


def _buoyancy_grid():
    x = np.arange(0.0, 10001.0, 500.0)
    y = np.arange(0.0, 6001.0, 1000.0)
    z = np.array([250.0, 500.0])
    b = (
        -2e-5 * xr.DataArray(x, dims="x", coords={"x": x})
        + 1e-5 * xr.DataArray(y, dims="y", coords={"y": y})
        + 0 * xr.DataArray(z, dims="z", coords={"z": z})
    )
    return b.transpose("z", "y", "x")


def _flashes():
    return _sources([(0, 0, 40), (10e3, 0, 25), (0, 10e3, 5), (-10e3, -10e3, 60)])[0]


def _rain_trajectories():
    return R.rain_trajectories(
        _source_grid().drop_vars("tt"),
        np.array([1.0, 2.0, 3.0]),
        profile=rain_profile(np.arange(0, 6001.0, 100.0), u=6.0, v=2.0),
        evaporation=False,
    )


def _buoyancy():
    z = np.arange(0.0, 3000.0, 100.0)
    return column(z, -0.03 * np.clip(1 - z / 2000.0, 0, None))


# Products that are not in PRODUCTS, with the reason. A product whose input
# is a download is left to the tests that fetch the file; the others need a
# multi-radar grid or a model volume that the tests build inline, and are not
# parametrised here yet.
NOT_COVERED = {
    "biology.biological_echo": "input is a volume and a downloaded ONNX model",
    "tornado.tornado_probability": "input is a volume and an ONNX model file",
    "tornado.tornet_inputs": "input is the module fixture of test_tornado",
    "single_doppler.single_doppler_winds": "input is a volume and an ONNX model",
    "grid.grid_radar": "test input is a downloaded NEXRAD volume",
    "grid.stack_data": "test input is a downloaded NEXRAD volume",
    "ugrid.to_uxarray": "needs the optional uxarray package",
    "multi.grid_radars": "needs several radar volumes",
    "multi.merge_radars": "needs a multi-radar grid",
    "multi.network_bias": "needs a multi-radar grid",
    "multidoppler.dual_doppler_analytic": "needs gridded volumes of several radars",
    "multidoppler.multi_doppler": "needs gridded volumes of several radars",
    "multidoppler.multi_doppler_input": "needs several radar volumes",
    "multidoppler.radar_geometry": "needs a grid with radar positions",
    "diabatic_lagrangian.diabatic_lagrangian": "needs winds, a sounding, a radar",
    "diabatic_lagrangian.polarimetric_precipitation": "needs a gridded radar",
    "diabatic_lagrangian.ziegler2013_precipitation": "needs a gridded radar",
    "disdrometer.match_radar": "needs a disdrometer record and a radar volume",
    "disdrometer.radar_at_location": "needs a radar volume at a site",
    "rain_trajectories.rain_source_points": "needs a target grid and a wind field",
    "rain_trajectories.surface_dsd": "needs trajectories and a source DSD",
    "rain_trajectories.trajectory_matched_times": "needs source and target grids",
}

PRODUCTS = {
    "estimate_kdp": lambda: R.estimate_kdp(kdp_sweep(nray=20, ng=300)[0]),
    "estimate_kdp (volume)": lambda: R.estimate_kdp(kdp_volume()),
    "echo_mask": lambda: R.echo_mask(qc_sweep()[0]),
    "apply_mask": lambda: R.apply_mask(qc_sweep()[0]),
    "hid (park)": lambda: _hid(method="park", band="S"),
    "hid (dolan)": lambda: _hid(method="dolan", band="C"),
    "hid (thompson)": lambda: _hid(method="thompson", band="S"),
    "dealias_velocity": lambda: R.dealias_velocity(_dealiased(), nyquist_velocity=10.0),
    "dealias_velocity (volume)": lambda: R.dealias_velocity(
        dealias_volume(nray=60, ngate=200)[0]
    ),
    "llsd": lambda: R.llsd(_dealiased(), "VRADH"),
    "azimuthal_shear": lambda: R.azimuthal_shear(_dealiased(), "VRADH"),
    "radial_divergence": lambda: R.radial_divergence(_dealiased(), "VRADH"),
    "beam_blockage": lambda: R.beam_blockage(
        bb_sweep(nray=6, ngate=60), geographic_dem(bell())
    ),
    "synthetic_vortex": small_vortex,
    "tc_center": small_tc_center,
    "gbvtd": lambda: R.gbvtd(
        small_vortex(), TC_CENTER, radii=np.array([15000.0, 20000.0, 25000.0])
    ),
    "tc_decompose": lambda: R.tc_decompose(
        R.gbvtd(small_vortex(), TC_CENTER, radii=np.array([15000.0, 20000.0, 25000.0]))
    ),
    "wrf_state": lambda: R.wrf_state(synthetic_wrf(nx=10, ny=8, nz=8)),
    "mpas_state": lambda: R.mpas_state(*synthetic_mpas(ncell=30, nz=6)),
    "forward_operator": lambda: R.forward_operator(
        R.wrf_state(synthetic_wrf(nx=10, ny=8, nz=8)), nrain="QNRAIN"
    ),
    "to_radar_grid": lambda: R.to_radar_grid(
        R.forward_operator(R.wrf_state(synthetic_wrf(nx=24, ny=20, nz=12))),
        FO_SITE,
        1.0,
        azimuth=[10.0, 80.0, 150.0],
        gate_range=[5000.0, 10000.0, 20000.0],
    ),
    "vil": lambda: R.vil(storm_volume()),
    "echo_top": lambda: R.echo_top(storm_volume()),
    "grid_cones": _grid,
    "grid_gates": lambda: radarx.grid.grid_gates(
        cone_volume(field="random"), "DBZH", **GRID, roi=1500.0
    ),
    "grid_rhi": lambda: radarx.grid.grid_rhi(
        _rhi_volume(lambda x, y, z: 20.0 + 0.001 * z),
        "DBZH",
        distance=np.arange(2e3, 20e3, 2e3),
        z=GATES_GRID["z"],
        roi=1000.0,
    ),
    "liquid_water_content": lambda: R.liquid_water_content(storm_volume()),
    "vil_density": lambda: R.vil_density(storm_volume()),
    "qvp": lambda: R.qvp(qvp_volume()),
    "melting_layer": lambda: R.melting_layer(
        _profiles_dataset([_melting_profile()] * 3)
    ),
    "advect": _moved,
    "estimate_motion": lambda: R.estimate_motion(
        _blobs(0, 8.0, -6.0), _blobs(300, 8.0, -6.0)
    ),
    "bulk_shear": lambda: R.bulk_shear(_wind(), 0, 6000),
    "layer_mean_wind": lambda: R.layer_mean_wind(_wind(), 0, 6000),
    "bunkers_storm_motion": lambda: R.bunkers_storm_motion(_wind()),
    "storm_relative_wind": lambda: R.storm_relative_wind(_wind(), (3.0, 1.0)),
    "storm_relative_helicity": lambda: R.storm_relative_helicity(_wind(), (3.0, 1.0)),
    "vad_profile": lambda: R.vad_profile(vad_sweep()),
    "terminal_fall_speed": lambda: R.terminal_fall_speed(
        xr.DataArray(np.linspace(0.5, 5.0, 10), dims="diameter")
    ),
    "parsivel_bins": lambda: R.parsivel_bins(),
    "dsd_spectrum": lambda: R.dsd_spectrum(
        xr.Dataset({"N0": 8000.0, "MU": 2.0, "LAMBDA": 2.5})
    ),
    "scattering_table": lambda: R.scattering_table(),
    "potential_temperatures": lambda: R.potential_temperatures(_sounding()),
    "fall_speed": lambda: R.fall_speed(xr.DataArray([20.0, 30.0, 40.0], dims="x")),
    "dsd": lambda: R.dsd(dsd_sweep()[0], "normalized", band="S"),
    "dsd (volume)": lambda: R.dsd(dsd_volume(), "normalized", band="S"),
    "evaporation": lambda: R.evaporation(evap_series(), evap_profile()),
    "integrate_evaporation": lambda: R.integrate_evaporation(
        evap_series(), evap_profile()
    ),
    "buoyancy": lambda: R.buoyancy(xr.DataArray([297.0, 300.0]), 300.0),
    "cold_pool_intensity": lambda: R.cold_pool_intensity(_buoyancy()),
    "cold_pool_intensity_from_surface": lambda: R.cold_pool_intensity_from_surface(
        xr.DataArray([-0.03, -0.05]), 1500.0
    ),
    "interpolate_time": lambda: R.interpolate_time(
        _blobs(0, 8.0, -6.0, threshold=None),
        _blobs(300, 8.0, -6.0, threshold=None),
        T0 + np.arange(0, 301, 150).astype("timedelta64[s]"),
    ),
    "create_cappi": lambda: R.create_cappi(
        _cappi_volume(), height=150.0, fields=["DBZH"], x_res=1000.0, y_res=1000.0
    ),
    "cold_pool_perturbation": lambda: R.cold_pool_perturbation(
        station_network(), slice("2022-03-31T00:00", "2022-03-31T00:30")
    ),
    "cold_pool_intensity_from_pressure": lambda: R.cold_pool_intensity_from_pressure(
        xr.DataArray([2.0, 3.0, 4.0], dims="x")
    ),
    "fit_gamma_moments": lambda: R.fit_gamma_moments(R.dsd_spectrum(_dsd_fit())),
    "radar_from_dsd": lambda: R.radar_from_dsd(R.dsd_spectrum(_dsd_fit()), "S"),
    "dsd_prior": lambda: R.dsd_prior(),
    "forward_grid": lambda: R.forward_grid("S"),
    "dsd_bayesian (volume)": lambda: R.dsd_bayesian(bayes_volume(), kdp="KDP"),
    "drop_evaporation_rate": lambda: R.drop_evaporation_rate(
        xr.DataArray([1.0, 2.0], dims="d"), 290.0, 90000.0, 0.01
    ),
    "trajectories": lambda: R.trajectories(
        _winds(5.0, -2.0, 0.5),
        time=T0,
        direction="forward",
        start=_point(2000.0, 15000.0, 1000.0),
    ),
    "disdrometer_qc": _qc_station,
    "number_concentration": lambda: R.number_concentration(_qc_station()),
    "process_disdrometer": lambda: R.process_disdrometer(_station()[0]),
    "baroclinic_generation": lambda: R.baroclinic_generation(_buoyancy_grid()),
    "qvp_timeseries": lambda: R.qvp_timeseries(
        [
            qvp_volume(elevations=(0.5, el), time=f"2026-01-01T00:{m:02d}")
            for m, el in ((0, 15.0), (5, 15.0), (10, 14.0))
        ],
        ["DBZH", "PHIDP"],
    ),
    "raupach_berne_correction": lambda: R.raupach_berne_correction(
        _station(ntime=4)[0]
    ),
    "dsd_moments": lambda: R.dsd_moments(_nd()),
    "fit_gamma": lambda: R.fit_gamma(_nd()),
    "cluster_flashes": lambda: R.cluster_flashes(_flashes()),
    "grid_lightning": lambda: R.grid_lightning(
        _flashes(), lightning_grid(), interval="1min"
    ),
    "rotation_couplets": lambda: R.rotation_couplets(rankine()),
    "rotation_couplets (min_tilts)": lambda: R.rotation_couplets(
        couplet_volume({"sweep_0": (0.5, [COLUMN]), "sweep_1": (1.5, [COLUMN])}),
        min_tilts=2,
    ),
    "azimuthal_shear_layer_max": lambda: R.azimuthal_shear_layer_max(
        column_volume(elevations=(0.5, 1.5, 3.0), n_az=90), min_reflectivity=20.0
    ),
    "correct_dual_prf": lambda: R.correct_dual_prf(
        dual_prf_sweep(nray=60, ngate=40)[0], nyquist_high=VH, nyquist_low=VL
    ),
    "dealias_confidence": lambda: R.dealias_confidence(
        R.dealias_velocity(_dealiased(), nyquist_velocity=10.0).to_dataset(),
        nyquist_velocity=10.0,
    ),
    "dealias_velocity (confidence)": lambda: R.dealias_velocity(
        _dealiased(), nyquist_velocity=10.0, confidence=True
    ),
    "lightning_jump": lambda: R.lightning_jump(
        _rate([12, 12, 13, 12, 13, 12, 12, 13, 12, 12, 13, 12, 12, 12] + [20, 30, 34])
    ),
    "vertical_source_distribution": lambda: R.vertical_source_distribution(
        _flashes(), np.arange(500.0, 15e3, 1000.0)
    ),
    "cell_flash_rate": lambda: R.cell_flash_rate(
        _flashes(),
        _mask(DAY + np.array([0, 60], dtype="timedelta64[s]")),
        z=np.arange(500.0, 15e3, 1000.0),
    ),
    "ziegler2013_profiles": lambda: R.ziegler2013_profiles(),
    "microphysical_rates": lambda: R.microphysical_rates(
        300.0, 90000.0, 0.01, 0.001, 0.001, 1.0e4, 0.0, 0.0
    ),
    "rain_trajectories": _rain_trajectories,
    "size_sorting": lambda: R.size_sorting(_rain_trajectories()),
    "rkw_ratio": lambda: R.rkw_ratio(xr.DataArray([10.0, 12.0]), 15.0),
    "perturb_radar": lambda: R.perturb_radar(ensemble_sweep(), members=3, snr="SNR"),
    "measurement_error": lambda: R.measurement_error(ensemble_sweep(), snr="SNR"),
    "ensemble_retrieval": lambda: R.ensemble_retrieval(
        ensemble_sweep(), "rain_rate_z", members=3, keep_members=True
    ),
    "ensemble_retrieval (hid)": lambda: R.ensemble_retrieval(
        ensemble_sweep(), "hid", members=3, options={"temperature": "TEMP", "band": "S"}
    ),
    "ensemble_statistics": lambda: R.ensemble_statistics(
        R.perturb_radar(ensemble_sweep(), members=3)
    ),
    "relation_scatter": lambda: R.relation_scatter("S"),
}


def _as_tree_or_dataset(obj):
    if isinstance(obj, xr.DataArray):
        return obj.to_dataset(name=obj.name or "product")
    return obj


def _open(path, engine, tree):
    if tree:
        return xr.open_datatree(path, engine=engine)
    return xr.open_dataset(path, engine=engine)


@pytest.mark.parametrize("name", list(PRODUCTS))
@pytest.mark.parametrize("engine", ["netcdf4", "zarr"])
def test_product_round_trip_is_identical(name, engine, tmp_path):
    if engine == "zarr" and NO_ZARR_3:
        pytest.skip("xarray before 2025.1 does not write Zarr 3")
    out = _as_tree_or_dataset(PRODUCTS[name]())
    tree = isinstance(out, xr.DataTree)
    path = tmp_path / ("product.zarr" if engine == "zarr" else "product.nc")
    if engine == "zarr":
        out.to_zarr(path)
    else:
        out.to_netcdf(path)
    back = _open(path, engine, tree)
    try:
        if not tree:
            back = back.load()
        xr.testing.assert_identical(back, out)
    finally:
        back.close()


def test_hid_codes_are_int8_without_a_fill_value(tmp_path):
    # a _FillValue of 0 would read class 0 (not classified) back as NaN float32
    out = PRODUCTS["hid (park)"]()
    assert out["HID"].dtype == np.int8
    assert "_FillValue" not in out["HID"].attrs
    out.to_netcdf(tmp_path / "hid.nc")
    with xr.open_dataset(tmp_path / "hid.nc") as back:
        assert back["HID"].dtype == np.int8


def test_not_covered_products_exist():
    from radarx._provenance import DECORATED

    assert set(NOT_COVERED) <= set(DECORATED)


@pytest.mark.parametrize(
    "name", ["estimate_kdp", "hid (park)", "grid_cones", "grid_gates", "grid_rhi"]
)
def test_chunked_product_round_trip_is_identical(name, tmp_path):
    pytest.importorskip("dask")
    out = PRODUCTS[name]()
    out.chunk().to_netcdf(tmp_path / "chunked.nc")
    with xr.open_dataset(tmp_path / "chunked.nc") as back:
        xr.testing.assert_identical(back.load(), out)
