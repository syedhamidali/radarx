"""The converters keep DataArray coords and attrs and stay lazy with dask."""

import warnings

import numpy as np
import pytest
import xarray as xr

from radarx.fundamentals import common, reflectivity

dask_array = pytest.importorskip("dask.array")


def _dbz_array(chunked=False):
    da = xr.DataArray(
        np.array([[10.0, 20.0, 30.0], [0.0, 40.0, 50.0]]),
        dims=("azimuth", "range"),
        coords={"azimuth": [0.5, 1.5], "range": [250.0, 500.0, 750.0]},
        attrs={
            "units": "dBZ",
            "long_name": "Equivalent reflectivity factor",
            "standard_name": "equivalent_reflectivity_factor",
            "valid_range": [-30.0, 80.0],
            "source": "test",
        },
        name="DBZH",
    )
    return da.chunk({"azimuth": 1}) if chunked else da


@pytest.mark.parametrize("chunked", [False, True])
def test_dbz_to_z_keeps_the_dataarray(chunked):
    da = _dbz_array(chunked)
    out = common.dbz_to_z(da)
    assert isinstance(out, xr.DataArray)
    assert out.dims == da.dims
    xr.testing.assert_equal(out.coords.to_dataset(), da.coords.to_dataset())
    assert out.attrs["units"] == "mm6 m-3"
    assert out.attrs["long_name"] == "Reflectivity factor"
    assert out.attrs["source"] == "test"
    assert "standard_name" not in out.attrs and "valid_range" not in out.attrs
    assert out.name == "DBZH"
    assert (out.chunks is not None) == chunked
    np.testing.assert_allclose(out.values, 10 ** (da.values / 10))


def test_z_to_dbz_keeps_the_dataarray_and_stays_lazy():
    z = common.dbz_to_z(_dbz_array(chunked=True))
    out = common.z_to_dbz(z)
    assert isinstance(out, xr.DataArray)
    assert isinstance(out.data, dask_array.Array)
    assert out.attrs["units"] == "dBZ"
    np.testing.assert_allclose(out.values, _dbz_array().values)
    # Z = 0 gives -inf without a warning, also for a lazy array
    zero = xr.DataArray(np.array([0.0, 1.0]), dims="range").chunk()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = common.z_to_dbz(zero).values
    assert res[0] == -np.inf and res[1] == 0.0


@pytest.mark.parametrize(
    "func, source_units, target_units, expected",
    [
        (common.km_to_m, "km", "m", lambda v: v * 1000.0),
        (common.m_to_km, "m", "km", lambda v: v / 1000.0),
        (common.kts_to_mps, "knot", "m s-1", lambda v: v * 0.514444),
        (common.mps_to_kts, "m s-1", "knot", lambda v: v / 0.514444),
    ],
)
def test_unit_converters_keep_the_dataarray(func, source_units, target_units, expected):
    da = _dbz_array(chunked=True)
    da.attrs.update(units=source_units, long_name="Distance", standard_name="x")
    out = func(da)
    assert isinstance(out, xr.DataArray) and isinstance(out.data, dask_array.Array)
    assert out.attrs["units"] == target_units
    # the quantity is the same: long_name and standard_name stay
    assert out.attrs["long_name"] == "Distance"
    assert out.attrs["standard_name"] == "x"
    xr.testing.assert_equal(out.coords.to_dataset(), da.coords.to_dataset())
    np.testing.assert_allclose(out.values, expected(da.values))


def test_converters_keep_dask_arrays_lazy_and_plain_input_as_before():
    lazy = dask_array.from_array(np.array([10.0, 20.0]), chunks=1)
    assert isinstance(common.dbz_to_z(lazy), dask_array.Array)
    assert isinstance(common.km_to_m([1.0, 2.0]), np.ndarray)
    assert isinstance(common.dbz_to_z(10.0), np.floating)
    assert common.si_to_km(1500.0) == 1.5
    assert common.si_to_kilometers(2.0, unit="km") == 2.0
    assert np.allclose(common.dbz_to_z([10, 20]), [10.0, 100.0])


def test_aliases_keep_the_dataarray():
    da = _dbz_array()
    for func in (common.linearize_dbz, common.dbz_to_z):
        assert isinstance(func(da), xr.DataArray)
    for func in (common.dbz_from_z, common.z_to_dbz):
        assert isinstance(func(common.dbz_to_z(da)), xr.DataArray)
    for func in (common.km_to_si, common.kilometers_to_si, common.kilometers_to_meters):
        assert func(da).attrs["units"] == "m"
    for func in (common.si_to_km, common.m_to_km, common.meters_to_kilometers):
        assert func(da).attrs["units"] == "km"
    for func in (common.si_to_kts, common.mps_to_kts, common.mps_to_knots):
        assert func(da).attrs["units"] == "knot"
    for func in (common.kts_to_si, common.kts_to_mps, common.knots_to_mps):
        assert func(da).attrs["units"] == "m s-1"


@pytest.mark.parametrize(
    "func, kwargs",
    [
        (reflectivity.z_to_r_marshall_palmer, {}),
        (reflectivity.z_to_r_custom, {"a": 300.0, "b": 1.4}),
    ],
)
def test_rain_rate_keeps_the_dataarray(func, kwargs):
    da = _dbz_array(chunked=True)
    out = func(da, **kwargs)
    assert isinstance(out, xr.DataArray) and isinstance(out.data, dask_array.Array)
    assert out.attrs["units"] == "mm h-1"
    assert out.attrs["long_name"] == "Rain rate"
    assert "standard_name" not in out.attrs
    xr.testing.assert_equal(out.coords.to_dataset(), da.coords.to_dataset())
    np.testing.assert_allclose(out.values, func(_dbz_array().values, **kwargs))
