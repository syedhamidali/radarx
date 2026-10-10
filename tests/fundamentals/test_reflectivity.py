import numpy as np
import pytest

from radarx.fundamentals import reflectivity


def test_z_to_r_marshall_palmer():
    dbz = 40
    rr = reflectivity.z_to_r_marshall_palmer(dbz)
    assert rr > 0


def test_z_to_r_custom():
    dbz = np.array([30, 35, 40])
    rr = reflectivity.z_to_r_custom(dbz, a=250, b=1.5)
    assert rr.shape == dbz.shape
    assert np.all(rr > 0)


def test_dbz_attenuation_correction():
    dbz = np.array([10, 20, 30])
    with pytest.warns(DeprecationWarning, match="hitschfeld_bordan_correction"):
        corrected = reflectivity.dbz_attenuation_correction(dbz)
    assert np.all(corrected > dbz)


# --- Hitschfeld-Bordan -------------------------------------------------------

A, B = 1e-4, 0.8  # k = A Z^B in dB/km, illustrative values for the tests


def _storm(n=800, dr=100.0):
    """True reflectivity [dBZ]: a 10 km core of 40 dBZ in a 20 dBZ background."""
    rng = dr * (np.arange(n) + 1.0)
    dbz = np.full(n, 20.0)
    dbz[(rng > 20e3) & (rng < 30e3)] = 40.0
    return rng, dbz


def _attenuate(dbz, rng):
    """Forward model: two-way attenuation with k = A Z^B on a fine grid."""
    dr_km = np.gradient(rng) / 1000.0
    k = A * (10 ** (dbz / 10)) ** B
    one_way = np.cumsum(k * dr_km) - 0.5 * k * dr_km
    return dbz - 2.0 * one_way


def test_hitschfeld_bordan_recovers_the_unattenuated_profile():
    rng, truth = _storm()
    measured = _attenuate(truth, rng)
    assert truth[-1] - measured[-1] > 3.0  # the profile loses more than 3 dB
    corrected, pia = reflectivity.hitschfeld_bordan_correction(
        measured, rng, a=A, b=B, return_pia=True
    )
    np.testing.assert_allclose(corrected, truth, atol=0.02)
    np.testing.assert_allclose(corrected, measured + pia)
    # the correction grows through the core and stays constant behind it
    assert np.all(np.diff(pia) >= -1e-12)
    assert pia[0] < 0.01 < pia[-1]
    np.testing.assert_allclose(pia[-1], truth[-1] - measured[-1], atol=0.02)


def test_hitschfeld_bordan_without_attenuation_changes_nothing():
    rng, dbz = _storm(n=50)
    np.testing.assert_array_equal(
        reflectivity.hitschfeld_bordan_correction(dbz, rng, a=0.0, b=B), dbz
    )


def test_hitschfeld_bordan_caps_the_correction():
    rng, dbz = _storm()
    strong = reflectivity.hitschfeld_bordan_correction(
        dbz, rng, a=2e-2, b=B, return_pia=True
    )[1]
    assert np.all(np.isfinite(strong))
    assert strong.max() == pytest.approx(10.0)
    capped = reflectivity.hitschfeld_bordan_correction(
        dbz, rng, a=2e-2, b=B, max_correction_db=4.0, return_pia=True
    )[1]
    assert capped.max() == pytest.approx(4.0)


def test_hitschfeld_bordan_missing_gates():
    rng, dbz = _storm(n=200)
    gap = dbz.copy()
    gap[50:60] = np.nan
    zero = dbz.copy()
    zero[50:60] = -np.inf  # no echo: no attenuation either
    out_gap = reflectivity.hitschfeld_bordan_correction(gap, rng, a=A, b=B)
    out_zero = reflectivity.hitschfeld_bordan_correction(zero, rng, a=A, b=B)
    assert np.all(np.isnan(out_gap[50:60]))
    keep = np.ones(dbz.size, bool)
    keep[50:60] = False
    np.testing.assert_allclose(out_gap[keep], out_zero[keep])


@pytest.mark.skipif(
    not reflectivity.HAS_COMPILED_KERNEL, reason="the compiled kernel is not built"
)
def test_hitschfeld_bordan_kernel_agrees_with_numpy():
    rng = np.arange(1, 401) * 250.0
    rs = np.random.default_rng(3)
    dbz = rs.uniform(-10.0, 55.0, size=(37, rng.size))
    dbz[rs.random(dbz.shape) < 0.1] = np.nan
    dbz[0, :5] = -np.inf
    kwargs = dict(a=3e-4, b=0.75, max_correction_db=12.0, return_pia=True)
    compiled = reflectivity.hitschfeld_bordan_correction(dbz, rng, **kwargs)
    reference = reflectivity.hitschfeld_bordan_correction(
        dbz, rng, use_compiled=False, **kwargs
    )
    threaded = reflectivity.hitschfeld_bordan_correction(
        dbz, rng, n_threads=3, **kwargs
    )
    for got, want in zip(compiled, reference):
        np.testing.assert_allclose(got, want, rtol=1e-11, atol=1e-11, equal_nan=True)
    for got, want in zip(threaded, compiled):
        np.testing.assert_array_equal(got, want)
    assert (compiled[1][np.isfinite(dbz)] > 0).any()


def test_hitschfeld_bordan_dataarray_and_dask():
    xr = pytest.importorskip("xarray")
    dask_array = pytest.importorskip("dask.array")
    rng, truth = _storm(n=300, dr=250.0)
    block = np.stack([_attenuate(truth + shift, rng) for shift in (0.0, 5.0, 10.0)])
    da = xr.DataArray(
        block,
        dims=("azimuth", "range"),
        coords={"azimuth": [10.0, 20.0, 30.0], "range": rng},
        attrs={"units": "dBZ", "long_name": "Reflectivity"},
        name="DBZH",
    )
    out, pia = reflectivity.hitschfeld_bordan_correction(da, a=A, b=B, return_pia=True)
    assert isinstance(out, xr.DataArray) and out.dims == da.dims
    xr.testing.assert_equal(out.coords.to_dataset(), da.coords.to_dataset())
    assert out.attrs["units"] == "dBZ" and out.attrs["long_name"] == "Reflectivity"
    assert out.name == "DBZH" and pia.attrs["units"] == "dB"
    np.testing.assert_allclose(
        out.values, reflectivity.hitschfeld_bordan_correction(block, rng, a=A, b=B)
    )
    # lazy with dask, also when the range is chunked, and the same values
    lazy = da.chunk({"azimuth": 1, "range": 100})
    out_lazy = reflectivity.hitschfeld_bordan_correction(lazy, a=A, b=B)
    assert isinstance(out_lazy.data, dask_array.Array)
    np.testing.assert_allclose(out_lazy.values, out.values)
    # range not the last dimension
    swapped = reflectivity.hitschfeld_bordan_correction(
        da.transpose("range", "azimuth"), a=A, b=B
    )
    assert swapped.dims == ("range", "azimuth")
    np.testing.assert_allclose(swapped.transpose("azimuth", "range").values, out.values)
    # an array with the range on another axis
    arr = reflectivity.hitschfeld_bordan_correction(block.T, rng, a=A, b=B, axis=0)
    np.testing.assert_allclose(arr, out.values.T)


def test_hitschfeld_bordan_checks_its_input():
    rng, dbz = _storm(n=10)
    with pytest.raises(ValueError, match="range_m is required"):
        reflectivity.hitschfeld_bordan_correction(dbz, a=A, b=B)
    with pytest.raises(ValueError, match="10 values"):
        reflectivity.hitschfeld_bordan_correction(dbz, rng[:5], a=A, b=B)
    with pytest.raises(ValueError, match="increase"):
        reflectivity.hitschfeld_bordan_correction(dbz, rng[::-1], a=A, b=B)
    with pytest.raises(ValueError, match="b must be"):
        reflectivity.hitschfeld_bordan_correction(dbz, rng, a=A, b=0.0)
    with pytest.raises(ValueError, match="a must be"):
        reflectivity.hitschfeld_bordan_correction(dbz, rng, a=-1.0, b=B)
    with pytest.raises(ValueError, match="max_correction_db"):
        reflectivity.hitschfeld_bordan_correction(
            dbz, rng, a=A, b=B, max_correction_db=np.inf
        )
