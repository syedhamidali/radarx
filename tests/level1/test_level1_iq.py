#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the IQ reader, the IQ constructor and the simulator
=============================================================

The simulator is the source of truth of the other Level I tests, so its
statistics are checked against the definition: the power, the white noise, the
mean velocity and width of the spectrum, the clutter and the aliasing.
"""
import numpy as np
import pytest
import xarray as xr
from level1_data import NYQUIST, PRT, WAVELENGTH

from radarx import level1

# ---- as_iq and read_iq -----------------------------------------------------


def samples(shape=(3, 8, 5), seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(-2000, 2000, shape), rng.integers(-2000, 2000, shape)


def test_as_iq_from_components_complex_and_dataarrays():
    i, q = samples()
    da = level1.as_iq(
        i,
        q,
        scale=1 / 32768,
        wavelength=WAVELENGTH,
        prt=PRT,
        range_start=500.0,
        gate_spacing=250.0,
    )
    assert da.dims == ("azimuth", "pulse", "range")
    assert da.dtype == np.complex64 and da.name == "IQ"
    np.testing.assert_allclose(da.values.real, i / 32768, rtol=1e-6)
    np.testing.assert_allclose(da.values.imag, q / 32768, rtol=1e-6)
    assert da.attrs["wavelength"] == WAVELENGTH and da.attrs["prt"] == PRT
    np.testing.assert_allclose(da.range, 500.0 + 250.0 * np.arange(5))
    assert da.range.attrs["units"] == "m"
    np.testing.assert_array_equal(da.pulse, np.arange(8))
    # complex input, double precision input keeps its precision
    wide = level1.as_iq(i.astype(float) + 1j * q, dims=("azimuth", "pulse", "range"))
    assert wide.dtype == np.complex128
    single = level1.as_iq(
        (i + 1j * q).astype(np.complex64), dims=("azimuth", "pulse", "range")
    )
    assert single.dtype == np.complex64
    # DataArrays keep their dimensions and coordinates
    di = xr.DataArray(
        i.astype(np.float32), dims=("ray", "pulse", "gate"), coords={"ray": [1, 2, 3]}
    )
    dq = xr.DataArray(q.astype(np.float32), dims=di.dims)
    out = level1.as_iq(di, dq, scale=0.5, name="IQV")
    assert out.dims == di.dims and out.name == "IQV" and out.dtype == np.complex64
    np.testing.assert_array_equal(out.ray, [1, 2, 3])
    both = level1.as_iq(xr.DataArray((i + 1j * q).astype(np.complex128), dims=di.dims))
    assert both.dtype == np.complex128
    from_array_q = level1.as_iq(di, q.astype(np.float32))
    assert from_array_q.dims == di.dims


def test_as_iq_errors():
    i, q = samples()
    with pytest.raises(ValueError, match="same shape"):
        level1.as_iq(i, q[:2])
    with pytest.raises(ValueError, match="same shape"):
        level1.as_iq(
            xr.DataArray(i, dims=("a", "b", "c")),
            xr.DataArray(q[:2], dims=("a", "b", "c")),
        )
    with pytest.raises(ValueError, match="pass q"):
        level1.as_iq(i)
    with pytest.raises(ValueError, match="axes"):
        level1.as_iq(i, q, dims=("pulse", "range"))
    with pytest.raises(ValueError, match="gate_spacing"):
        level1.as_iq(i[0, :, 0], q[0, :, 0], dims=("pulse",), gate_spacing=100.0)


def test_read_raw_interleaved_planar_with_offset(tmp_path):
    i, q = samples()
    shape = i.shape
    interleaved = np.stack([i, q], axis=-1).astype("<i2")
    path = tmp_path / "iq.raw"
    path.write_bytes(b"HDR!" + interleaved.tobytes())
    da = level1.read_iq(path, shape=shape, offset=4, wavelength=WAVELENGTH, prt=PRT)
    np.testing.assert_array_equal(da.values.real, i)
    np.testing.assert_array_equal(da.values.imag, q)
    assert da.dims == ("azimuth", "pulse", "range")
    planar = np.stack([i, q], axis=0).astype(">f4")
    path = tmp_path / "planar.bin"
    path.write_bytes(planar.tobytes())
    da = level1.read_iq(
        path,
        shape=shape,
        dtype=">f4",
        layout="planar",
        scale=2.0,
        dims=("ray", "pulse", "gate"),
    )
    np.testing.assert_allclose(da.values.real, 2.0 * i)
    assert da.dims == ("ray", "pulse", "gate")
    # one and two axes get the default names
    path = tmp_path / "one.raw"
    path.write_bytes(
        np.stack([i[0, :, 0], q[0, :, 0]], axis=-1).astype("<i2").tobytes()
    )
    assert level1.read_iq(path, shape=(8,)).dims == ("pulse",)
    path.write_bytes(np.stack([i[0], q[0]], axis=-1).astype("<i2").tobytes())
    assert level1.read_iq(path, shape=(8, 5)).dims == ("pulse", "range")


def test_read_raw_errors(tmp_path):
    path = tmp_path / "short.raw"
    path.write_bytes(np.zeros(10, dtype="<i2").tobytes())
    with pytest.raises(ValueError, match="shape"):
        level1.read_iq(path)
    with pytest.raises(ValueError, match="needs"):
        level1.read_iq(path, shape=(4, 4))
    with pytest.raises(ValueError, match="layout"):
        level1.read_iq(path, shape=(1, 1), layout="rows")
    with pytest.raises(FileNotFoundError):
        level1.read_iq(tmp_path / "missing.raw", shape=(1,))
    four = tmp_path / "four.raw"
    four.write_bytes(np.zeros(2 * 16, dtype="<i2").tobytes())
    with pytest.raises(ValueError, match="dims"):
        level1.read_iq(four, shape=(2, 2, 2, 2))


def test_read_npy(tmp_path):
    i, q = samples()
    cplx = (i + 1j * q).astype(np.complex64)
    np.save(tmp_path / "c.npy", cplx)
    da = level1.read_iq(tmp_path / "c.npy", wavelength=WAVELENGTH)
    np.testing.assert_array_equal(da.values, cplx)
    np.save(tmp_path / "r.npy", np.stack([i, q], axis=-1).astype(np.float32))
    da = level1.read_iq(tmp_path / "r.npy", dims=("azimuth", "pulse", "range"))
    np.testing.assert_array_equal(da.values, cplx)
    np.save(tmp_path / "bad.npy", i.astype(np.float32))
    with pytest.raises(ValueError, match="last axis"):
        level1.read_iq(tmp_path / "bad.npy")
    np.save(tmp_path / "four.npy", np.zeros((2, 2, 2, 2), dtype=complex))
    with pytest.raises(ValueError, match="dims"):
        level1.read_iq(tmp_path / "four.npy")


def test_read_netcdf_and_zarr(tmp_path):
    i, q = samples()
    dims = ("azimuth", "pulse", "range")
    ds = xr.Dataset(
        {"I": (dims, i.astype(np.float32)), "Q": (dims, q.astype(np.float32))}
    )
    nc = tmp_path / "iq.nc"
    ds.to_netcdf(nc)
    da = level1.read_iq(nc, wavelength=WAVELENGTH, prt=PRT)
    np.testing.assert_array_equal(da.values.real, i.astype(np.float32))
    np.testing.assert_array_equal(da.values.imag, q.astype(np.float32))
    renamed = ds.rename(I="in_phase", Q="quadrature")
    renamed.to_netcdf(tmp_path / "iq2.nc")
    da = level1.read_iq(tmp_path / "iq2.nc", variables=("in_phase", "quadrature"))
    assert da.dims == dims
    with pytest.raises(ValueError, match="no variables"):
        level1.read_iq(tmp_path / "iq2.nc")
    zarr = pytest.importorskip("zarr")  # noqa: F841
    store = tmp_path / "iq.zarr"
    ds.to_zarr(store, encoding={v: {"chunks": ds[v].shape} for v in ds.data_vars})
    lazy = level1.read_iq(store)
    np.testing.assert_array_equal(lazy.values.real, i.astype(np.float32))
    zstore = tmp_path / "cplx.zarr"
    cds = xr.Dataset({"IQ": (dims, (i + 1j * q).astype(np.complex64))})
    cds.to_zarr(zstore, encoding={"IQ": {"chunks": cds.IQ.shape}})
    lazy = level1.read_iq(zstore, wavelength=WAVELENGTH)
    np.testing.assert_array_equal(lazy.values, (i + 1j * q).astype(np.complex64))


# ---- simulate_iq -----------------------------------------------------------


def test_the_simulated_power_noise_and_shape():
    iq = level1.simulate_iq(
        np.zeros((6, 40)),
        width=1.0,
        power=50.0,
        noise=2.0,
        n_pulses=128,
        wavelength=WAVELENGTH,
        prt=PRT,
        seed=1,
    )
    assert iq.dims == ("azimuth", "pulse", "range")
    assert iq.shape == (6, 128, 40) and iq.dtype == np.complex64
    assert iq.attrs["wavelength"] == WAVELENGTH and iq.attrs["prt"] == PRT
    power = float((np.abs(iq) ** 2).mean())
    assert power == pytest.approx(52.0, rel=0.03)
    noise = level1.simulate_iq(
        np.zeros(2000), power=0.0, noise=2.0, n_pulses=16, dims=("range",), seed=2
    )
    assert float((np.abs(noise) ** 2).mean()) == pytest.approx(2.0, rel=0.03)
    assert abs(float(noise.values.real.mean())) < 0.05
    assert float(noise.values.real.var()) == pytest.approx(
        1.0, rel=0.05
    )  # half the power in I


def test_a_seed_gives_the_same_series_and_a_generator_continues():
    a = level1.simulate_iq([1.0, 2.0], seed=5, dims=("range",))
    b = level1.simulate_iq([1.0, 2.0], seed=5, dims=("range",))
    c = level1.simulate_iq([1.0, 2.0], seed=6, dims=("range",))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    rng = np.random.default_rng(5)
    d = level1.simulate_iq([1.0, 2.0], seed=rng, dims=("range",))
    e = level1.simulate_iq([1.0, 2.0], seed=rng, dims=("range",))
    np.testing.assert_array_equal(a, d)
    assert not np.array_equal(d, e)


def test_the_simulated_spectrum_has_the_requested_velocity_and_width():
    v0, w0 = 8.0, 3.0
    iq = level1.simulate_iq(
        np.full(3000, v0),
        width=w0,
        power=100.0,
        n_pulses=64,
        wavelength=WAVELENGTH,
        prt=PRT,
        dims=("range",),
        seed=3,
    )
    spectra = level1.doppler_spectra(iq, window="rect").SPECTRUM.values.astype(float)
    mean = spectra.mean(axis=0)
    v = (np.arange(64) - 32) * (WAVELENGTH / (2 * 64 * PRT))
    centre = (mean * v).sum() / mean.sum()
    spread = np.sqrt((mean * (v - centre) ** 2).sum() / mean.sum())
    assert centre == pytest.approx(v0, abs=0.1)
    assert spread == pytest.approx(w0, rel=0.05)


def test_the_sign_of_the_velocity_and_aliasing():
    for v, expected in (
        (10.0, 10.0),
        (-10.0, -10.0),
        (NYQUIST + 5.0, -NYQUIST + 5.0),
        (-NYQUIST - 5.0, NYQUIST - 5.0),
    ):
        iq = level1.simulate_iq(
            [v] * 50,
            width=0.5,
            power=100.0,
            n_pulses=64,
            wavelength=WAVELENGTH,
            prt=PRT,
            dims=("range",),
            seed=4,
        )
        pp = level1.pulse_pair_moments(iq)
        assert float(pp.VRADH.mean()) == pytest.approx(expected, abs=0.3)


def test_clutter_sits_at_zero_velocity_and_params_may_be_dataarrays():
    velocity = xr.DataArray(
        np.full((2, 300), 10.0),
        dims=("azimuth", "range"),
        coords={"azimuth": [10.0, 20.0]},
    )
    clutter = xr.DataArray(np.full((2, 300), 1000.0), dims=("azimuth", "range"))
    iq = level1.simulate_iq(
        velocity,
        width=1.0,
        power=1.0,
        clutter_power=clutter,
        clutter_width=0.1,
        n_pulses=64,
        wavelength=WAVELENGTH,
        prt=PRT,
        seed=0,
    )
    np.testing.assert_array_equal(iq.azimuth, [10.0, 20.0])
    pp = level1.pulse_pair_moments(iq)
    assert float(np.abs(pp.VRADH).mean()) < 1.0  # dominated by the clutter at zero
    assert float((np.abs(iq) ** 2).mean()) == pytest.approx(1001.0, rel=0.15)
    # a range-only simulation puts the pulse dimension before range
    one = level1.simulate_iq(np.zeros(4), dims=("range",), seed=0)
    assert one.dims == ("pulse", "range")
    other = level1.simulate_iq(xr.DataArray(np.zeros(4), dims="gate"), seed=0)
    assert other.dims == ("gate", "pulse")
    scalar = level1.simulate_iq(5.0, seed=0)
    assert scalar.dims == ("pulse",)


def test_simulate_errors():
    with pytest.raises(ValueError, match="n_pulses"):
        level1.simulate_iq(1.0, n_pulses=2)
    with pytest.raises(ValueError, match="positive"):
        level1.simulate_iq(1.0, wavelength=0.0)
    with pytest.raises(ValueError, match="positive"):
        level1.simulate_iq(1.0, prt=-1.0)
    with pytest.raises(ValueError, match="power"):
        level1.simulate_iq(1.0, power=-1.0)
    with pytest.raises(ValueError, match="noise"):
        level1.simulate_iq(1.0, noise=-1.0)
    with pytest.raises(ValueError, match="width"):
        level1.simulate_iq(1.0, width=-1.0)
    with pytest.raises(ValueError, match="dims"):
        level1.simulate_iq(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="dims"):
        level1.simulate_iq(np.zeros((2, 2)), dims=("a",))
