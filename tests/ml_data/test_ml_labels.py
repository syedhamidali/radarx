"""Label builders of ``ml/data`` on synthetic sweeps with known answers."""

import numpy as np
import pytest
import xarray as xr
from mldata import labels
from mldata.polar import PolarGrid, gate_index, ray_index, resample_sweep, take

from radarx.retrieve import dealias_velocity, echo_mask, estimate_kdp, hid

from .conftest import synthetic_sweep

# --------------------------------------------------------------------------
# fixed polar grid
# --------------------------------------------------------------------------


@pytest.mark.parametrize("nray", [360, 720])
def test_ray_index_nearest_ray(nray):
    az = (np.arange(nray) + 0.3) * 360.0 / nray
    shuffled = np.random.default_rng(1).permutation(nray)
    idx = ray_index(az[shuffled], 360)
    assert np.all(idx >= 0)
    picked = az[shuffled][idx]
    centres = np.arange(360) + 0.5
    d = np.abs((picked - centres + 180) % 360 - 180)
    assert d.max() <= 0.5 * 360.0 / nray + 1e-9


def test_ray_index_gap_and_wrap():
    az = np.r_[np.arange(0.5, 90.0, 1.0), np.arange(180.5, 359.9, 1.0)]
    idx = ray_index(az, 360)
    assert np.all(idx[100:170] == -1)  # no rays between 90 and 180
    assert idx[0] >= 0 and idx[-1] >= 0
    # wrap: a single ray at 359.9 serves bin 0 (0.5 deg) too
    idx = ray_index(np.array([10.0, 359.9]), 360)
    assert idx[0] == 1
    assert ray_index(np.array([]), 4).tolist() == [-1] * 4


def test_gate_index_and_take():
    grid = PolarGrid(n_azimuth=4, first_gate=2125.0, gate_spacing=250.0, n_gates=6)
    native = 2125.0 + 250.0 * np.arange(3)
    g = gate_index(native, grid)
    assert g.tolist() == [0, 1, 2, -1, -1, -1]
    assert gate_index(np.array([]), grid).tolist() == [-1] * 6
    vals = np.arange(6.0).reshape(2, 3)
    out = take(vals, np.array([1, -1]), g, np.nan)
    assert out[0, :3].tolist() == [3.0, 4.0, 5.0]
    assert np.isnan(out[1]).all() and np.isnan(out[0, 3:]).all()


def test_resample_sweep_keeps_values_and_dtypes(sweep):
    sweep = sweep.assign(CLASS=(sweep.DBZH > 30).astype("int8"))
    grid = PolarGrid(n_gates=500)
    out = resample_sweep(sweep, ["DBZH", "CLASS"], grid)
    assert out["DBZH"].dims == ("azimuth", "range")
    assert out["DBZH"].dtype == np.float32 and out["CLASS"].dtype == np.int8
    # same native grid for the first 400 gates: values are copied exactly
    np.testing.assert_array_equal(out["DBZH"].values[:, :400], sweep["DBZH"].values)
    assert np.isnan(out["DBZH"].values[:, 400:]).all()
    assert (out["CLASS"].values[:, 400:] == 0).all()
    # float to class conversion fills NaN with 0
    out = resample_sweep(sweep, ["DBZH"], grid, dtypes={"DBZH": "int8"})
    assert out["DBZH"].dtype == np.int8


def test_resample_super_resolution_sweep():
    ds = synthetic_sweep(nray=720, ngate=50)
    out = resample_sweep(ds, ["VRADH"], PolarGrid(n_gates=50))
    assert out.sizes == {"azimuth": 360, "range": 50}
    # every output ray is one of the native rays
    native = {tuple(np.nan_to_num(r)) for r in ds["VRADH"].values}
    assert all(tuple(np.nan_to_num(r)) in native for r in out["VRADH"].values)


# --------------------------------------------------------------------------
# dealiasing labels
# --------------------------------------------------------------------------


@pytest.mark.parametrize("nyquist", [5.0, 8.0, 13.3])
def test_fold_velocity_roundtrip(nyquist):
    v = np.linspace(-60, 60, 1001)
    folded, k = labels.fold_velocity(v, nyquist)
    assert np.all(folded >= -nyquist) and np.all(folded < nyquist)
    np.testing.assert_allclose(folded + 2 * nyquist * k, v, atol=1e-9)
    assert k.dtype == np.int8
    da = xr.DataArray(np.r_[v, np.nan])
    f2, k2 = labels.fold_velocity(da, nyquist)
    assert isinstance(k2, xr.DataArray) and int(k2[-1]) == 0
    assert np.isnan(f2[-1])


def test_residual_jump_fraction(sweep):
    v = sweep["VRADH"].values
    assert labels.residual_jump_fraction(v, 30.0) == 0.0
    folded, _ = labels.fold_velocity(v, 6.0)
    assert labels.residual_jump_fraction(folded, 6.0) > 0.0
    assert labels.residual_jump_fraction(np.full((3, 3), np.nan), 1.0) == 0.0


def test_velocity_truth_sources(sweep):
    v = sweep["VRADH"]
    # observed field already unaliased
    truth, source = labels.velocity_truth(v, v, 30.0)
    assert source == labels.TRUTH_SOURCES["observed_unaliased"]
    np.testing.assert_array_equal(truth.values, v.values)
    # aliased measurement, correct dealiased field
    folded, _ = labels.fold_velocity(v, 8.0)
    truth, source = labels.velocity_truth(folded, v, 8.0)
    assert source == labels.TRUTH_SOURCES["radarx_dealiased"]
    assert labels.velocity_truth(folded, v, 8.0, sources=["observed_unaliased"]) is None
    # a dealiased field with fold lines left is rejected
    assert labels.velocity_truth(folded, folded, 8.0) is None
    # too few gates
    assert labels.velocity_truth(v, v, 30.0, min_gates=10**7) is None


def test_dealias_sample_with_radarx_baseline():
    ds = synthetic_sweep(nray=180, ngate=200, wind=(20.0, 5.0), sector=180)
    truth = ds["VRADH"]
    out = labels.dealias_sample(ds, truth, 9.0)
    valid = np.isfinite(truth.values)
    recon = out["VRADH_folded"] + 2 * 9.0 * out["FOLD"]
    np.testing.assert_allclose(recon.values[valid], truth.values[valid], atol=1e-4)
    assert np.all(np.abs(out["VRADH_folded"].values[valid]) <= 9.0)
    assert 0 < out.attrs["folded_fraction"] < 1
    assert "nyquist_velocity" not in out.variables
    # radarx recovers the uniform wind seen all around the radar
    err = np.abs(out["VRADH_radarx"].values - truth.values)[valid]
    assert np.mean(err < 0.1) > 0.99
    # the baseline function is replaceable
    out = labels.dealias_sample(ds, truth, 9.0, dealias=lambda s, vn: s["VRADH"])
    np.testing.assert_array_equal(
        out["VRADH_radarx"].values, out["VRADH_folded"].values
    )


def test_dealias_truth_from_radarx_on_aliased_sweep():
    ds = synthetic_sweep(nray=180, ngate=200, wind=(20.0, 5.0), sector=180)
    folded, _ = labels.fold_velocity(ds["VRADH"], 12.0)
    measured = ds.assign(VRADH=folded).assign_coords(nyquist_velocity=12.0)
    found = labels.velocity_truth(
        measured["VRADH"], dealias_velocity(measured, "VRADH"), 12.0
    )
    assert found is not None
    truth, source = found
    assert source == 1
    valid = np.isfinite(ds["VRADH"].values)
    assert np.mean(np.abs(truth.values - ds["VRADH"].values)[valid] < 0.1) > 0.99


# --------------------------------------------------------------------------
# blockage / inpainting
# --------------------------------------------------------------------------


def test_blockage_field_sectors():
    az = np.arange(360) + 0.5
    r = 2125.0 + 250.0 * np.arange(400)
    f = labels.blockage_field(az, r, np.random.default_rng(3), start_range=(20e3, 30e3))
    assert f.dtype == np.float32 and f.shape == (360, 400)
    assert f.min() >= 0 and f.max() <= 1 and f.max() > 0
    assert np.all(f[:, r < 20e3] == 0)  # nothing before the obstacles
    # once blocked, blocked to the end of the ray (non-decreasing)
    assert np.all(np.diff(f, axis=1) >= -1e-7)
    again = labels.blockage_field(
        az, r, np.random.default_rng(3), start_range=(20e3, 30e3)
    )
    np.testing.assert_array_equal(f, again)
    total = labels.blockage_field(
        az, r, np.random.default_rng(0), n_sectors=(2, 2), partial_probability=0.0
    )
    assert set(np.unique(total)) <= {0.0, 1.0}


def test_apply_blockage():
    dbz = np.array([[30.0, 30.0, 30.0]])
    f = np.array([[0.0, 0.5, 1.0]])
    out = labels.apply_blockage(dbz, f)
    np.testing.assert_allclose(out[0, :2], [30.0, 30.0 - 10 * np.log10(2)], atol=1e-5)
    assert np.isnan(out[0, 2])
    da = labels.apply_blockage(xr.DataArray(dbz, dims=("azimuth", "range")), f)
    assert isinstance(da, xr.DataArray) and da.dtype == np.float32


def test_inpaint_sample(sweep):
    grid = PolarGrid(n_gates=400)
    dbz = resample_sweep(sweep, ["DBZH"], grid)["DBZH"]
    out = labels.inpaint_sample(dbz, np.random.default_rng(5), partial_probability=0.0)
    blocked = out["BLOCKAGE"].values >= 0.99
    assert blocked.any()
    assert np.isnan(out["DBZH_blocked"].values[blocked]).all()
    free = ~blocked
    np.testing.assert_array_equal(
        out["DBZH_blocked"].values[free], out["DBZH"].values[free]
    )


# --------------------------------------------------------------------------
# qc / kdp / hid
# --------------------------------------------------------------------------


def test_qc_kdp_hid_samples(sweep):
    qc = echo_mask(sweep, min_size=1)
    s = labels.qc_sample(sweep, qc)
    assert s["ECHO_CLASS"].dtype == np.int8
    cls = s["ECHO_CLASS"].values
    # clutter patch is not meteorological, the rain mostly is
    r, a = sweep["range"].values, sweep["azimuth"].values
    clutter = (r[None, :] < 10e3) & (np.abs(a[:, None] - 40) < 15)
    rain = (r[None, :] > 25e3) & (r[None, :] < 65e3) & (np.abs(a[:, None] - 200) < 50)
    assert np.mean(cls[clutter] == 1) < 0.2
    assert np.mean(cls[rain] == 1) > 0.9

    kdp = estimate_kdp(sweep)
    k = labels.kdp_sample(sweep, kdp)
    # PHIDP rises by 1 deg/km: KDP = 0.5 deg/km in the rain
    assert abs(np.nanmedian(k["KDP"].values[rain]) - 0.5) < 0.1

    merged = sweep.assign(KDP=kdp["KDP"], METEO_MASK=qc["METEO_MASK"])
    profile = labels.standard_profile(4000.0)
    h = hid(merged, profile, mask="METEO_MASK", scores=False)
    temp = labels.gate_temperature(sweep, profile, 179.0)
    s = labels.hid_sample(merged, h, temp)
    assert s["HID"].dtype == np.int8 and s["METEO_MASK"].dtype == bool
    assert np.all(s["HID"].values[~s["METEO_MASK"].values] == 0)
    assert np.nanmax(s["TEMPERATURE"].values) < 30
    # without temperature and mask
    s = labels.hid_sample(sweep.assign(KDP=kdp["KDP"]), h)
    assert np.isnan(s["TEMPERATURE"].values).all()


def test_missing_optional_fields_are_nan():
    ds = synthetic_sweep(polarimetric=False, ngate=60, nray=36)
    qc = xr.Dataset(
        {
            "ECHO_CLASS": xr.zeros_like(ds.DBZH, dtype="int8"),
            "METEO_SCORE": xr.zeros_like(ds.DBZH),
        }
    )
    s = labels.qc_sample(ds, qc)
    assert np.isnan(s["ZDR"].values).all() and s["ZDR"].dtype == np.float32


def test_beam_height_and_profile():
    assert labels.beam_height(0.0, 0.5, 100.0) == pytest.approx(100.0)
    h = labels.beam_height(100e3, 0.5, 0.0)
    # 4/3 Earth: about 0.87 km from the elevation plus 0.59 km curvature
    assert 1300 < h < 1600
    p = labels.standard_profile(3000.0)
    t = p.temperature.interp(height=3000.0) - 273.15
    assert abs(float(t)) < 1e-6
    assert float(p.temperature.min()) == pytest.approx(273.15 - 60)


def test_gate_temperature_kelvin_and_celsius(sweep):
    p = labels.standard_profile(3000.0)
    t_k = labels.gate_temperature(sweep, p, 0.0)
    p_c = p.assign(temperature=(p.temperature - 273.15).assign_attrs(units="degC"))
    t_c = labels.gate_temperature(sweep, p_c, 0.0)
    np.testing.assert_allclose(t_k.values, t_c.values, atol=1e-4)
    assert t_k.attrs["units"] == "degC"


# --------------------------------------------------------------------------
# nowcasting
# --------------------------------------------------------------------------


def _blob_frames(u=10.0, v=-5.0, dt=300.0, n=3):
    x = y = (np.arange(128) - 63.5) * 1000.0
    X, Y = np.meshgrid(x, y)
    frames = []
    t0 = np.datetime64("2022-03-30T23:00:00", "ns")
    for i in range(n):
        cx, cy = -20e3 + u * dt * i, 10e3 + v * dt * i
        field = 55.0 * np.exp(-(((X - cx) / 12e3) ** 2 + ((Y - cy) / 6e3) ** 2))
        field = field + 30.0 * np.exp(
            -(
                ((X + 30e3 - u * dt * i) / 5e3) ** 2
                + ((Y + 30e3 - v * dt * i) / 9e3) ** 2
            )
        )
        field = np.where(field > 5.0, field, -10.0).astype("float32")
        frames.append(
            xr.DataArray(
                field,
                dims=("y", "x"),
                coords={"y": y, "x": x, "time": t0 + np.timedelta64(int(dt * i), "s")},
                name="DBZH",
            )
        )
    return frames


def test_nowcast_sample_motion_and_extrapolation():
    frames = _blob_frames()
    s = labels.nowcast_sample(frames, n_input=2, motion_options={"tile": None})
    assert s["DBZH"].dims == ("lead", "y", "x")
    assert s["lead"].values.tolist() == [-1, 0]
    assert s["lead_target"].values.tolist() == [1]
    assert s["DBZH_extrapolated"].dims == ("lead_target", "y", "x")
    np.testing.assert_array_equal(s["DBZH_target"][0].values, frames[2].values)
    assert s["target_time"].values[0] == frames[2]["time"].values
    assert float(s["u"].mean()) == pytest.approx(10.0, abs=1.0)
    assert float(s["v"].mean()) == pytest.approx(-5.0, abs=1.0)
    target = frames[2].values
    ext = s["DBZH_extrapolated"].isel(lead_target=0).values
    persistence = frames[1].values
    ok = np.isfinite(ext)
    err_ext = np.nanmean(np.abs(ext - target)[ok])
    err_per = np.nanmean(np.abs(persistence - target)[ok])
    assert err_ext < 0.5 * err_per
    assert s["time"].values == frames[1]["time"].values


def test_nowcast_sample_errors_and_no_motion():
    frames = _blob_frames()
    with pytest.raises(ValueError):
        labels.nowcast_sample(frames[:2], n_input=2)
    flat = [f * 0 - 10.0 for f in frames]
    assert labels.nowcast_sample(flat, n_input=2, motion_options={"tile": None}) is None


def test_coverage_and_grid_axes():
    x, y, z = labels.grid_axes({"size": 4, "spacing": 2000.0, "z": [500]})
    assert x.tolist() == [-3000.0, -1000.0, 1000.0, 3000.0]
    assert z.tolist() == [500.0]
    cov = labels.coverage(x, y, 2000.0)
    assert int(cov.sum()) == 4
