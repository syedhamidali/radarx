# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for radarx probabilistic nowcasting
=========================================

A synthetic echo field is a smooth random texture that translates with a known
motion, so the extrapolation, the cascade and the motion have an analytic
truth. The ensemble tests check the properties of the model (unit variance of
the autoregressive levels, spread of the members, the rain mask, the
probability matching) and the agreement of the compiled kernel with the NumPy
implementation. The real-data test nowcasts three NEXRAD volumes.
"""

import numpy as np
import pytest
import xarray as xr
from scipy import fft as sfft
from scipy.ndimage import gaussian_filter, shift

import radarx  # noqa: F401
from radarx.retrieve import nowcast, nowcasting
from radarx.retrieve._nowcast_numpy import ar_step, ar_step_numpy
from radarx.retrieve.advection import estimate_motion
from radarx.retrieve.nowcasting import exceedance_probability, nowcast_model

from .provenance_helpers import without_provenance

ENGINES = ["numpy"] + (["compiled"] if nowcasting.HAS_COMPILED_KERNEL else [])
needs_kernel = pytest.mark.skipif(
    not nowcasting.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)
T0 = np.datetime64("2026-01-01T00:00:00", "ns")
N = 72
DX = 1000.0
DT = 300.0
X = np.arange(N) * DX
Y = np.arange(N) * DX
# the pattern moves by (U, V) m/s: the fixture shifts it by -(U, V) dt / DX cells
U, V = 8.0, -4.0


def _texture(seed=0, size=240):
    rng = np.random.default_rng(seed)
    big = gaussian_filter(rng.standard_normal((size, size)), 5, mode="wrap")
    big += 0.5 * gaussian_filter(rng.standard_normal((size, size)), 2, mode="wrap")
    return 30.0 + 12.0 * (big - big.mean()) / big.std()


BIG = _texture()


def _frame(k, u=U, v=V, nan_below=True, big=BIG):
    """Frame ``k`` of the translating texture (the pattern moves by (u, v))."""
    f = shift(big, (v * DT * k / DX, u * DT * k / DX), order=3, mode="wrap")
    f = f[80 : 80 + N, 80 : 80 + N]
    if nan_below:
        f = np.where(f >= 10.0, f, np.nan)
    ds = xr.Dataset(
        {"DBZH": (("y", "x"), f.astype("f4"), {"units": "dBZ"})},
        coords={"x": ("x", X, {"units": "m"}), "y": ("y", Y, {"units": "m"})},
    )
    return ds.assign(time=xr.DataArray(T0 + np.timedelta64(int(DT * k), "s")))


@pytest.fixture(scope="module")
def history():
    return [_frame(k) for k in range(3)]


def _truth(k):
    return _frame(k)["DBZH"].values


def _rmse(forecast, truth, floor=5.0):
    t = np.where(np.isnan(truth), floor, truth)
    inner = (slice(12, -12),) * 2
    return float(np.sqrt(np.nanmean((forecast[inner] - t[inner]) ** 2)))


# ---------------------------------------------------------------------------
# extrapolation
# ---------------------------------------------------------------------------


def test_extrapolation_recovers_the_translation(history):
    out = nowcast(history, 3, method="extrapolation")
    assert out.dims == ("lead_time", "y", "x")
    assert out.dtype == np.float32
    for i in range(3):
        assert _rmse(out.values[i], _truth(3 + i)) < 1.0
    assert out.attrs["units"] == "dBZ"
    assert out.attrs["nowcast_method"] == "extrapolation"


@pytest.mark.parametrize("engine", ENGINES)
def test_extrapolation_with_given_motion(history, engine):
    given = nowcast(history, 2, method="extrapolation", motion=(U, V), engine=engine)
    est = nowcast(history, 2, method="extrapolation", engine=engine)
    # the estimate differs from (U, V) by a few hundredths of a cell: only cells
    # at the echo threshold may differ by more than a little
    assert np.nanmean(np.abs(given.values - est.values) > 0.5) < 1e-3
    ds = estimate_motion(history[1], history[2])
    again = nowcast(history, 2, method="extrapolation", motion=ds, engine=engine)
    np.testing.assert_allclose(again.values, est.values, atol=1e-4)


def test_coordinates_and_times(history):
    out = nowcast(history, 4, method="extrapolation")
    lead = out["lead_time"].values
    np.testing.assert_array_equal(lead, np.arange(1, 5) * np.timedelta64(300, "s"))
    last = history[-1]["time"].values
    assert out["time"].values == last
    np.testing.assert_array_equal(out["valid_time"].values, last + lead)
    np.testing.assert_array_equal(out["x"].values, X)
    assert out.name == "DBZH"


def test_time_dimension_input_and_dataarray(history):
    stacked = xr.concat(history, dim="time").assign_coords(
        time=[f["time"].values for f in history]
    )
    from_dim = nowcast(stacked, 2, method="extrapolation")
    from_list = nowcast(history, 2, method="extrapolation")
    np.testing.assert_allclose(from_dim.values, from_list.values, atol=1e-5)
    arrays = [f["DBZH"].assign_coords(time=f["time"]) for f in history]
    from_da = nowcast(arrays, 2, method="extrapolation")
    np.testing.assert_allclose(from_da.values, from_list.values, atol=1e-5)
    # a Dataset with a time dimension also works as an accessor
    via = stacked.radarx.nowcast(2, method="extrapolation")
    np.testing.assert_allclose(via.values, from_list.values, atol=1e-5)


def test_dt_argument_and_uneven_times(history):
    bare = [f.drop_vars("time") for f in history]
    with pytest.raises(ValueError, match="no scalar 'time'"):
        nowcast(bare, 1, method="extrapolation")
    out = nowcast(bare, 1, method="extrapolation", dt=DT)
    ref = nowcast(history, 1, method="extrapolation")
    np.testing.assert_allclose(out.values, ref.values, atol=1e-5)
    assert "valid_time" not in out.coords
    shifted = [
        history[0],
        history[1],
        history[2].assign(time=history[2].time + np.timedelta64(120, "s")),
    ]
    with pytest.raises(ValueError, match="evenly spaced"):
        nowcast(shifted, 1, method="extrapolation")
    with pytest.raises(ValueError, match="dt must be positive"):
        nowcast(bare, 1, method="extrapolation", dt=-1.0)


def test_no_echo_gives_the_no_echo_value():
    empty = [
        _frame(k, nan_below=True).assign(DBZH=lambda d: d.DBZH * np.nan)
        for k in range(3)
    ]
    for method in ("extrapolation", "sprog", "steps"):
        with pytest.warns(RuntimeWarning, match="persistence"):
            out = nowcast(empty, 2, method=method, n_members=2, seed=0)
        assert np.all(out.values == 5.0)
    out = nowcast(empty, 2, method="sprog", motion=(1.0, 1.0))
    assert np.all(out.values[np.isfinite(out.values)] == 5.0)


# ---------------------------------------------------------------------------
# the cascade
# ---------------------------------------------------------------------------


def test_band_filters_add_up_to_one():
    w = nowcasting._band_filters((96, 100), 1000.0, 1000.0, 6)
    assert w.shape == (6, 96, 51)
    np.testing.assert_allclose(w.sum(axis=0), 1.0, atol=1e-12)
    assert w[0, 0, 0] == 1.0  # the mean belongs to the first level
    assert np.all(w >= 0)
    # the levels go from large scales to small ones
    k = np.hypot(np.fft.fftfreq(96)[:, None], np.fft.rfftfreq(100)[None, :])
    centroid = [(w[j] * k).sum() / w[j].sum() for j in range(6)]
    assert np.all(np.diff(centroid) > 0)


def test_levels_reconstruct_the_field(history):
    lagr = np.where(
        np.isfinite(_frame(0)["DBZH"].values), _frame(0)["DBZH"].values, 5.0
    )
    shape, offset = nowcasting._padded_shape(N, N, 0.25)
    w = nowcasting._band_filters(shape, DX, DX, 6)
    padded = np.full(shape, 5.0)
    padded[offset[0] : offset[0] + N, offset[1] : offset[1] + N] = lagr
    spec = sfft.rfft2(padded)
    back = sfft.irfft2((w * spec[None]).sum(axis=0), s=shape)
    np.testing.assert_allclose(back, padded, atol=1e-10)


def test_padded_shape_contains_the_grid():
    shape, (oy, ox) = nowcasting._padded_shape(70, 90, 0.25)
    assert shape[0] >= 70 + 2 * 18 and shape[1] >= 90 + 2 * 23
    assert oy + 70 <= shape[0] and ox + 90 <= shape[1]
    shape, off = nowcasting._padded_shape(70, 90, 0.0)
    assert shape[0] >= 70 and shape[1] >= 90 and min(off) >= 0


def test_yule_walker_parameters_and_innovation():
    # AR(2) with phi = (0.5, 0.3): rho1 = phi1 / (1 - phi2), rho2 = phi1 rho1 + phi2
    phi1, phi2 = 0.5, 0.3
    rho1 = phi1 / (1 - phi2)
    rho2 = phi1 * rho1 + phi2
    p1, p2, gain = nowcasting._ar_parameters(np.array([rho1]), np.array([rho2]), 2)
    np.testing.assert_allclose([p1[0], p2[0]], [phi1, phi2], atol=1e-12)
    # the innovation that gives the process unit variance
    rng = np.random.default_rng(0)
    z = np.zeros(200000)
    e = rng.standard_normal(z.size) * gain[0]
    for t in range(2, z.size):
        z[t] = phi1 * z[t - 1] + phi2 * z[t - 2] + e[t]
    assert abs(z[1000:].var() - 1.0) < 0.03
    # the published coefficient without the square root would give a variance
    assert abs(gain[0] ** 2 - (1 - phi1 * rho1 - phi2 * rho2)) < 1e-12
    # AR(1)
    p1, p2, gain = nowcasting._ar_parameters(np.array([0.8]), np.array([0.1]), 1)
    assert (p1[0], p2[0]) == (0.8, 0.0)
    assert abs(gain[0] - 0.6) < 1e-12


def test_correlations_are_limited():
    p1, p2, gain = nowcasting._ar_parameters(
        np.array([1.0, -0.5, 0.9]), np.array([1.0, 0.9, -0.99]), 2
    )
    assert np.all(np.isfinite(p1)) and np.all(np.isfinite(gain))
    # a stationary AR(2): roots of 1 - phi1 z - phi2 z**2 outside the unit circle
    for a, b in zip(p1, p2):
        roots = np.roots([-b, -a, 1.0])
        assert np.all(np.abs(roots) > 1.0)


@pytest.mark.parametrize("use_compiled", [False] + ([True] if needs_kernel else []))
def test_innovation_keeps_the_unit_variance_of_the_levels(history, use_compiled):
    """White noise through the innovation filter holds every level at variance 1."""
    data = np.where(np.isfinite(_truth(0)), _truth(0), 5.0)
    shape, offset = nowcasting._padded_shape(N, N, 0.25)
    w = nowcasting._band_filters(shape, DX, DX, 5)
    frames = np.stack([data] * 3)
    setup = nowcasting._cascade_setup(frames, w, offset, shape, 5.0, 2)
    # fixed parameters of a persistent process
    setup["phi1"] = np.full(5, 0.6)
    setup["phi2"] = np.full(5, 0.2)
    r1 = 0.6 / 0.8
    r2 = 0.6 * r1 + 0.2
    setup["gain"] = np.full(5, np.sqrt(1 - 0.6 * r1 - 0.2 * r2))
    amplitude = nowcasting._noise_amplitude(data, offset, shape)
    wh = nowcasting._innovation_filter(setup, w, amplitude)
    members = 24
    rngs = [np.random.default_rng(i) for i in range(members)]
    nh = shape[1] // 2 + 1
    z1 = np.zeros((members, 5, shape[0], nh), dtype=complex)
    z2 = np.zeros_like(z1)
    for _ in range(60):
        white = np.stack([g.standard_normal(shape) for g in rngs])
        ar_step(
            z1, z2, setup["phi1"], setup["phi2"], np.ones(5), wh,
            sfft.rfft2(white), use_compiled, 1,
        )  # fmt: skip
    n_pix = shape[0] * shape[1]
    var = [
        np.mean(
            [
                np.sum(setup["weights"] * np.abs(z1[m, j]) ** 2) / n_pix**2
                for m in range(members)
            ]
        )
        for j in range(5)
    ]
    np.testing.assert_allclose(var, 1.0, rtol=0.1)


@needs_kernel
def test_ar_step_kernel_matches_numpy():
    rng = np.random.default_rng(0)
    shape = (3, 4, 20, 11)
    z1 = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    z2 = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    wh = rng.random(shape[1:])
    noise = rng.standard_normal((3, 20, 11)) + 0j
    args = (rng.random(4), rng.random(4), rng.random(4), wh)
    for e in (noise, None):
        a1, a2 = z1.copy(), z2.copy()
        b1, b2 = z1.copy(), z2.copy()
        out_c = ar_step(a1, a2, *args, e, True, 2)
        out_n = ar_step_numpy(b1, b2, *args, e)
        np.testing.assert_allclose(out_c, out_n, rtol=1e-12)
        np.testing.assert_allclose(a1, b1, rtol=1e-12)
        np.testing.assert_allclose(a2, b2, rtol=1e-12)


@needs_kernel
def test_ar_step_kernel_checks_its_arguments():
    from radarx.retrieve import _nowcast

    z = np.zeros((1, 2, 4, 3), dtype=complex)
    ok = (
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        np.zeros((2, 4, 3)),
        np.empty(0, complex),
    )
    with pytest.raises(ValueError, match="4-D"):
        _nowcast.ar_step(np.zeros((2, 4, 3), complex), z, *ok, 1)
    with pytest.raises(ValueError, match="differ in shape"):
        _nowcast.ar_step(z, np.zeros((1, 2, 4, 4), complex), *ok, 1)
    with pytest.raises(ValueError, match="one value per level"):
        _nowcast.ar_step(z, z.copy(), np.zeros(3), *ok[1:], 1)
    with pytest.raises(ValueError, match="shape \\(level"):
        _nowcast.ar_step(z, z.copy(), *ok[:3], np.zeros((2, 4, 4)), ok[4], 1)
    with pytest.raises(ValueError, match="shape \\(member"):
        _nowcast.ar_step(z, z.copy(), *ok[:4], np.zeros((2, 4, 3), complex), 1)


# ---------------------------------------------------------------------------
# S-PROG and STEPS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["sprog", "steps"])
def test_cascade_forecast_follows_the_pattern(history, method):
    out = nowcast(history, 3, method=method, n_members=4, seed=3)
    if method == "steps":
        out = out.mean("member")
    ref = nowcast(history, 3, method="extrapolation")
    for i in range(3):
        err = _rmse(out.values[i], _truth(3 + i))
        persistence = _rmse(_truth(2), _truth(3 + i))
        assert err < 0.5 * persistence
        assert err < _rmse(ref.values[i], _truth(3 + i)) + 6.0


def test_sprog_loses_small_scales_and_steps_keeps_them(history):
    sprog = nowcast(history, 6, method="sprog", probability_matching=False)
    steps = nowcast(
        history, 6, method="steps", n_members=6, seed=0, probability_matching=False
    )

    def small_scale_variance(a):
        a = np.where(a > 5.0, a, 5.0)
        return float(np.var(a - gaussian_filter(a, 3)))

    first = small_scale_variance(sprog.values[0])
    last = small_scale_variance(sprog.values[-1])
    assert last < 0.9 * first
    members = [small_scale_variance(steps.values[m, -1]) for m in range(6)]
    assert np.mean(members) > last


def test_ensemble_members_differ_and_seed_reproduces(history):
    a = nowcast(history, 3, n_members=3, seed=7)
    b = nowcast(history, 3, n_members=3, seed=7)
    c = nowcast(history, 3, n_members=3, seed=8)
    assert a.dims == ("member", "lead_time", "y", "x")
    np.testing.assert_array_equal(a.values, b.values)
    assert not np.array_equal(a.values, c.values)
    assert not np.array_equal(a.values[0], a.values[1])
    np.testing.assert_array_equal(a["member"].values, [0, 1, 2])


def test_seed_gives_the_same_ensemble_for_any_number_of_threads(history):
    a = nowcast(history, 2, n_members=3, seed=2, n_threads=1)
    b = nowcast(history, 2, n_members=3, seed=2, n_threads=3)
    np.testing.assert_allclose(a.values, b.values, atol=1e-4)


def test_ensemble_does_not_depend_on_the_batching(history, monkeypatch):
    a = nowcast(history, 2, n_members=5, seed=4)
    monkeypatch.setattr(nowcasting, "_STATE_BYTES", 1.0)  # one member at a time
    b = nowcast(history, 2, n_members=5, seed=4)
    np.testing.assert_allclose(a.values, b.values, atol=1e-5)


@needs_kernel
@pytest.mark.parametrize("method", ["sprog", "steps"])
def test_engines_agree(history, method):
    kw = dict(method=method, n_members=3, seed=5)
    a = nowcast(history, 3, engine="compiled", **kw)
    b = nowcast(history, 3, engine="numpy", **kw)
    # float32 rounding of the output and the order of the sums
    np.testing.assert_allclose(a.values, b.values, atol=2e-3)


def test_engine_names():
    with pytest.raises(ValueError, match="engine must be"):
        nowcast([_frame(0), _frame(1)], 1, engine="gpu")
    if not nowcasting.HAS_COMPILED_KERNEL:  # pragma: no cover
        with pytest.raises(ImportError):
            nowcast([_frame(0), _frame(1)], 1, engine="compiled")


def test_ar_order_one_needs_two_frames(history):
    out = nowcast(history[1:], 2, method="sprog")
    assert out.shape[0] == 2
    with pytest.raises(ValueError, match="three frames"):
        nowcast(history[1:], 2, method="sprog", ar_order=2)
    one = nowcast(history, 2, method="sprog", ar_order=1)
    assert one.shape[0] == 2


def test_rain_mask_removes_echo_far_from_observed_echo(history):
    # one isolated cell: nothing may appear far from it
    f = np.full((N, N), np.nan)
    f[30:38, 30:38] = 40.0
    frames = []
    for k in range(3):
        g = history[0].copy()
        arr = shift(np.nan_to_num(f, nan=0.0), (0.0, 0.0), order=0)
        g["DBZH"] = (("y", "x"), np.where(arr > 0, arr, np.nan).astype("f4"))
        g = g.assign(time=T0 + np.timedelta64(int(DT * k), "s"))
        frames.append(g)
    out = nowcast(
        frames, 3, method="steps", n_members=3, seed=1, motion=(0.0, 0.0),
        mask_radius=6e3,
    )  # fmt: skip
    far = np.zeros((N, N), bool)
    far[:, :10] = True
    far[:, 50:] = True
    assert np.all(out.values[..., far] == 5.0)
    off = nowcast(
        frames, 3, method="steps", n_members=3, seed=1, motion=(0.0, 0.0),
        mask_radius=None,
    )  # fmt: skip
    assert np.any(off.values[..., far] > 5.0) or np.all(off.values[..., far] == 5.0)
    grown = nowcast(
        frames, 3, method="sprog", motion=(0.0, 0.0), mask_radius=2e3, mask_growth=20.0,
    )  # fmt: skip
    assert grown.shape[0] == 3


def test_probability_matching_restores_the_distribution():
    rng = np.random.default_rng(0)
    obs = rng.gamma(2.0, 5.0, (40, 50))
    field = rng.normal(0, 1, (4, 40, 50))
    seen = np.ones((40, 50), bool)
    seen[:5] = False
    ref = np.sort(obs[seen])
    out = nowcasting._match(field.copy(), seen, ref, False, 1)
    for m in range(4):
        np.testing.assert_array_equal(np.sort(out[m][seen]), ref)
        # the ranks of the forecast are kept
        np.testing.assert_array_equal(
            np.argsort(out[m][seen], kind="stable"),
            np.argsort(field[m][seen], kind="stable"),
        )
    np.testing.assert_array_equal(out[:, ~seen], field[:, ~seen])


def test_probability_matching_changes_the_forecast(history):
    on = nowcast(history, 4, method="sprog", probability_matching=True)
    off = nowcast(history, 4, method="sprog", probability_matching=False)
    assert not np.allclose(on.values, off.values)


def test_distance_map():
    echo = np.zeros((10, 10), bool)
    echo[5, 5] = True
    d = nowcasting._distance_map(echo, 1000.0, 2000.0)
    assert d[5, 5] == 0 and d[5, 6] == 2000.0 and d[6, 5] == 1000.0
    assert np.all(np.isinf(nowcasting._distance_map(np.zeros((3, 3), bool), 1.0, 1.0)))


def test_observed_area_limits_the_forecast(history):
    seen = np.ones((N, N), bool)
    seen[:, :20] = False
    obs = xr.DataArray(seen, dims=("y", "x"))
    out = nowcast(history, 2, method="sprog", observed=obs, motion=(0.0, 0.0))
    assert np.all(np.isnan(out.values[..., :20]))
    assert np.all(np.isfinite(out.values[..., 20:]))
    with pytest.raises(ValueError, match="observed must be on the grid"):
        nowcast(history, 2, observed=obs.isel(x=slice(5)))


# ---------------------------------------------------------------------------
# perturbed motion
# ---------------------------------------------------------------------------


def _centroid(a):
    a = np.where(a > 5.0, a - 5.0, 0.0)
    rows, cols = np.indices(a.shape)
    return (a * rows).sum() / a.sum(), (a * cols).sum() / a.sum()


def test_motion_perturbation_spreads_the_members(history):
    spec = {"parallel": (0.0, 1.0, 1.0), "perpendicular": (0.0, 1.0, 1.0)}
    steady = nowcast(history, 3, method="extrapolation", motion=(U, V))
    ens = nowcast(
        history, 3, method="steps", n_members=12, seed=1, motion=(U, V),
        motion_perturbation=spec, probability_matching=False, mask_radius=None,
    )  # fmt: skip
    base = _centroid(steady.values[-1])
    shifts = np.array(
        [np.subtract(_centroid(ens.values[m, -1]), base) for m in range(12)]
    )
    # the shift grows as sigma = 1 m/s * (1 + 0) times the lead (15 min): a few cells
    assert shifts.std() > 0.2
    plain = nowcast(
        history, 3, method="steps", n_members=12, seed=1, motion=(U, V),
        probability_matching=False, mask_radius=None,
    )  # fmt: skip
    plain_shifts = np.array(
        [np.subtract(_centroid(plain.values[m, -1]), base) for m in range(12)]
    )
    assert plain_shifts.std() < shifts.std()


def test_perturbed_motion_statistics():
    rng = np.random.default_rng(0)
    leads = np.array([5.0, 10.0, 20.0])
    par, perp = (0.1, 1.0, 0.5), (0.05, 1.0, 0.2)
    draws = np.array(
        [
            nowcasting._perturbed_motion(rng, 3.0, 4.0, par, perp, leads)
            for _ in range(4000)
        ]
    )  # (draw, lead, 2)
    along = np.array([0.6, 0.8])
    across = np.array([-0.8, 0.6])
    sig_par = par[0] * leads ** par[1] + par[2]
    sig_perp = perp[0] * leads ** perp[1] + perp[2]
    np.testing.assert_allclose((draws @ along).std(axis=0), sig_par, rtol=0.08)
    np.testing.assert_allclose((draws @ across).std(axis=0), sig_perp, rtol=0.08)
    # Laplace: excess kurtosis 3
    e = (draws @ along)[:, 0] / sig_par[0]
    assert 1.5 < np.mean(e**4) / np.mean(e**2) ** 2 - 3 < 4.5
    # zero mean motion: the x axis is the reference direction
    out = nowcasting._perturbed_motion(rng, 0.0, 0.0, par, perp, leads)
    assert out.shape == (3, 2)


def test_motion_perturbation_is_checked(history):
    with pytest.raises(ValueError, match="needs method='steps'"):
        nowcast(
            history, 1, method="sprog",
            motion_perturbation={"parallel": (0, 1, 1), "perpendicular": (0, 1, 1)},
        )  # fmt: skip
    for bad in (
        {"parallel": (1, 2, 3)},
        {"parallel": (1, 2), "perpendicular": (1, 2, 3)},
        5,
    ):
        with pytest.raises(ValueError, match="motion_perturbation"):
            nowcast(history, 1, motion_perturbation=bad, n_members=1)


# ---------------------------------------------------------------------------
# spatially varying motion, errors
# ---------------------------------------------------------------------------


def test_spatially_varying_motion(history):
    est = estimate_motion(history[1], history[2], tile=40e3)
    out = nowcast(history, 2, method="sprog", motion=est)
    assert np.isfinite(out.values).mean() > 0.7
    ens = nowcast(history, 2, n_members=2, seed=1, motion=est)
    assert ens.shape == (2, 2, N, N)


def test_unreliable_motion_warns_and_persists(history, monkeypatch):
    def no_motion(*args, **kwargs):
        return xr.Dataset({"u": np.nan, "v": np.nan, "quality": 0.0})

    monkeypatch.setattr(nowcasting, "estimate_motion", no_motion)
    with pytest.warns(RuntimeWarning, match="persistence"):
        out = nowcast(history, 1, method="extrapolation")
    again = nowcast(history, 1, method="extrapolation", motion=(0.0, 0.0))
    np.testing.assert_allclose(out.values, again.values)


def test_motion_with_nan_is_refused(history):
    with pytest.raises(ValueError, match="NaN"):
        nowcast(history, 1, motion=(np.nan, 1.0))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"method": "pysteps"}, "method must be"),
        ({"n_steps": 0}, "n_steps"),
        ({"n_steps": 2.5}, "n_steps"),
        ({"n_members": 0}, "n_members"),
        ({"ar_order": 3}, "ar_order"),
        ({"n_levels": 2}, "n_levels"),
        ({"mask_radius": -1.0}, "mask_radius"),
        ({"mask_growth": -1.0}, "mask_growth"),
        ({"zero_value": 12.0}, "zero_value"),
    ],
)
def test_options_are_checked(history, kwargs, message):
    kwargs = {"n_steps": 1, **kwargs}
    with pytest.raises(ValueError, match=message):
        nowcast(history, **kwargs)


def test_history_is_checked(history):
    with pytest.raises(ValueError, match="at least two"):
        nowcast(history[:1], 1)
    with pytest.raises(TypeError, match="list of Datasets"):
        nowcast([1, 2, 3], 1)
    with pytest.raises(ValueError, match="time dimension"):
        nowcast(history[0], 1)
    other = history[2].isel(x=slice(0, 60))
    with pytest.raises(ValueError, match="same grid"):
        nowcast([history[0], history[1], other], 1)


def test_input_is_not_changed(history):
    before = [f.copy(deep=True) for f in history]
    nowcast(history, 2, n_members=2, seed=0)
    for a, b in zip(history, before):
        xr.testing.assert_identical(a, b)


def test_provenance_and_attributes(history):
    out = nowcast(history, 2, n_members=2, seed=0)
    assert "10.1256/qj.04.100" in out.attrs["radarx_references"]
    assert (
        "10.1175/1520-0450(2003)042<0381:ADASSA>2.0.CO;2"
        in out.attrs["radarx_references"]
    )
    assert out.attrs["echo_threshold"] == 10.0
    assert out.attrs["no_echo_value"] == 5.0
    bare = without_provenance(out)
    assert "radarx_method" not in bare.attrs


def test_dask_input_is_computed(history):
    pytest.importorskip("dask")
    stacked = xr.concat(history, dim="time").assign_coords(
        time=[f["time"].values for f in history]
    )
    expected = nowcast(stacked, 2, method="sprog")
    chunked = nowcast(stacked.chunk({"x": 36, "y": 36}), 2, method="sprog")
    xr.testing.assert_allclose(chunked.compute(), expected)


# ---------------------------------------------------------------------------
# exceedance probability
# ---------------------------------------------------------------------------


def test_exceedance_probability():
    values = np.array([[10.0, 20.0, 30.0, 40.0], [30.0, np.nan, 30.0, np.nan]])
    ens = xr.DataArray(values.T[:, :, None], dims=("member", "k", "p"))
    prob = exceedance_probability(ens, 25.0)
    np.testing.assert_allclose(prob.values[:, 0], [0.5, 1.0])
    assert prob.dtype == np.float32
    assert prob.attrs["units"] == "1"
    allnan = xr.DataArray(np.full((3, 2), np.nan), dims=("member", "p"))
    assert np.isnan(exceedance_probability(allnan, 0.0)).all()
    # threshold is inclusive
    assert exceedance_probability(ens, 30.0).values[0, 0] == 0.5
    with pytest.raises(ValueError, match="no dimension"):
        exceedance_probability(ens, 1.0, member_dim="ensemble")
    via = ens.radarx.exceedance_probability(25.0)
    np.testing.assert_allclose(via.values, prob.values)


def test_ensemble_probability_is_calibrated_in_a_simple_case(history):
    ens = nowcast(history, 3, n_members=16, seed=0)
    prob = exceedance_probability(ens, 30.0)
    assert float(prob.min()) >= 0.0 and float(prob.max()) <= 1.0
    assert float(prob.isel(lead_time=0).max()) > 0.5


# ---------------------------------------------------------------------------
# irregular scan intervals
# ---------------------------------------------------------------------------


def _frame_at(seconds):
    """The translating texture at a time (s) after T0."""
    f = _frame(seconds / DT)
    return f.assign(time=T0 + np.timedelta64(int(seconds), "s"))


def test_frames_with_slightly_different_intervals_are_aligned_by_their_times():
    times = [0.0, 280.0, 580.0]  # intervals of 280 and 300 s
    frames = [_frame_at(t) for t in times]
    hist, planes, t_last, dt_s, steps = nowcasting._history(
        frames, None, None, "x", "y"
    )
    np.testing.assert_allclose(steps, [280.0, 300.0])
    assert dt_s == 290.0
    data = np.where(np.isfinite(planes), planes, 5.0)
    seen = np.ones((N, N), bool)
    lagr = nowcasting._lagrangian_history(
        data,
        U,
        V,
        DX,
        DX,
        5.0,
        seen,
        3,
        steps,
        {"use_compiled": True, "n_threads": None},
    )
    inner = (slice(18, -18),) * 2
    for k in (0, 1):
        both = (data[-1][inner] > 6) & (lagr[k][inner] > 6)
        assert np.abs(lagr[k][inner] - data[-1][inner])[both].mean() < 0.5
    # without the correction the frames would be off by a fraction of a cell
    wrong = nowcasting._lagrangian_history(
        data, U, V, DX, DX, 5.0, seen, 3, np.array([290.0, 290.0]),
        {"use_compiled": True, "n_threads": None},
    )  # fmt: skip
    assert (
        np.abs(wrong[1][inner] - data[-1][inner]).mean()
        > np.abs(lagr[1][inner] - data[-1][inner]).mean()
    )
    out = nowcast(frames, 2, method="sprog", motion=(U, V))
    assert out.shape[0] == 2
    with pytest.raises(ValueError, match="evenly spaced"):
        nowcast([frames[0], frames[1], _frame_at(1300.0)], 1)


@needs_kernel
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rank_match_kernel_matches_numpy(dtype):
    from radarx.retrieve import _nowcast
    from radarx.retrieve._nowcast_numpy import rank_match_numpy

    rng = np.random.default_rng(0)
    fields = rng.normal(size=(7, 400)).astype(dtype)
    fields[:, :50] = 0.0  # ties
    cells = np.sort(rng.choice(400, 150, replace=False))
    ref = np.sort(rng.gamma(2.0, 3.0, 150))
    a, b = fields.copy(), fields.copy()
    _nowcast.rank_match(a, cells, ref, 3)
    rank_match_numpy(b, cells, ref)
    np.testing.assert_array_equal(a, b)
    untouched = np.setdiff1d(np.arange(400), cells)
    np.testing.assert_array_equal(a[:, untouched], fields[:, untouched])
    with pytest.raises(ValueError, match="2-D"):
        _nowcast.rank_match(a[0], cells, ref, 1)
    with pytest.raises(ValueError, match="one value per cell"):
        _nowcast.rank_match(a, cells, ref[:5], 1)
    with pytest.raises(ValueError, match="out of range"):
        _nowcast.rank_match(a, cells + 1000, ref, 1)


# ---------------------------------------------------------------------------
# learned model (ONNX)
# ---------------------------------------------------------------------------


@pytest.fixture
def registry(monkeypatch, tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from radarx.ml import model as model_mod

    monkeypatch.setattr(model_mod, "_registered", {})
    monkeypatch.setenv("RADARX_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("RADARX_MODEL_REGISTRY", raising=False)
    return tmp_path


def _network(
    path, n_input=3, drift=0.5, frames=1, spatial=(None, None), outputs=("y",)
):
    """Output = the last input frame plus ``drift`` (and again, for 2 frames)."""
    import hashlib

    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", n_input, *spatial])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", frames, *spatial])
    inits = [
        helper.make_tensor("starts", TensorProto.INT64, [1], [-1]),
        helper.make_tensor("ends", TensorProto.INT64, [1], [np.iinfo(np.int64).max]),
        helper.make_tensor("axes", TensorProto.INT64, [1], [1]),
        helper.make_tensor("drift", TensorProto.FLOAT, [], [drift]),
    ]
    nodes = [
        helper.make_node("Slice", ["x", "starts", "ends", "axes"], ["last"]),
        helper.make_node("Add", ["last", "drift"], ["next"]),
    ]
    if frames == 2:
        nodes += [
            helper.make_node("Add", ["next", "drift"], ["after"]),
            helper.make_node("Concat", ["next", "after"], ["y"], axis=1),
        ]
    else:
        nodes += [helper.make_node("Identity", ["next"], ["y"])]
    graph = helper.make_graph(nodes, "drift", [x], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_nowcast_model_runs_the_network_recursively(registry, history):
    from radarx import ml

    path, sha = _network(registry / "drift.onnx", drift=0.5)
    ml.register_model("drift", path, sha, "MIT", "Doe (2026): drift")
    out = nowcast_model(history, "drift", 3, scale=2.0, offset=10.0)
    last = np.where(
        np.isfinite(history[-1]["DBZH"].values), history[-1]["DBZH"].values, 5.0
    )
    last = np.where(last >= 10.0, last, 5.0)
    # normalised (dBZ - 10) / 2, plus 0.5 per step, back: +1 dB per step
    for k in range(3):
        expected = np.where(last + (k + 1) >= 10.0, last + (k + 1), 5.0)
        # cells that were below the threshold start at the no-echo value
        echo = last >= 10.0
        np.testing.assert_allclose(out.values[k][echo], expected[echo], atol=1e-4)
    assert out.dims == ("lead_time", "y", "x") and out.dtype == np.float32
    assert out.attrs["ml_model"] == "drift" and out.attrs["ml_model_licence"] == "MIT"
    assert out.attrs["nowcast_method"] == "model"
    np.testing.assert_array_equal(
        out["lead_time"].values, np.arange(1, 4) * np.timedelta64(300, "s")
    )
    # a Model object and the first n_input frames of the history
    net = ml.load_model("drift")
    same = nowcast_model(history, net, 3, scale=2.0, offset=10.0)
    np.testing.assert_allclose(same.values, out.values)


def test_nowcast_model_with_a_network_that_returns_two_frames(registry, history):
    from radarx import ml

    path, sha = _network(registry / "two.onnx", drift=1.0, frames=2)
    ml.register_model("two", path, sha, "MIT", "x")
    out = nowcast_model(history, "two", 4, n_input=3)
    echo = history[-1]["DBZH"].values >= 10.0
    base = history[-1]["DBZH"].values
    for k in range(4):
        np.testing.assert_allclose(out.values[k][echo], base[echo] + k + 1, atol=1e-4)
    with pytest.raises(ValueError, match="use recurrent=True"):
        nowcast_model(history, "two", 3, recurrent=False)
    ok = nowcast_model(history, "two", 2, recurrent=False)
    assert ok.shape[0] == 2


def test_nowcast_model_checks_its_input(registry, history):
    from radarx import ml

    path, sha = _network(registry / "fixed.onnx", n_input=3, spatial=(40, 40))
    ml.register_model("fixed", path, sha, "MIT", "x")
    with pytest.raises(ValueError, match="size 72 on axis 2"):
        nowcast_model(history, "fixed", 1)
    path, sha = _network(registry / "four.onnx", n_input=4)
    ml.register_model("four", path, sha, "MIT", "x")
    with pytest.raises(ValueError, match="needs 4 frames"):
        nowcast_model(history, "four", 1)
    path, sha = _network(registry / "free.onnx", n_input=3)
    ml.register_model("free", path, sha, "MIT", "x")
    for kwargs, message in [
        ({"n_steps": 0}, "n_steps"),
        ({"scale": 0.0}, "scale"),
        ({"n_input": 0}, "n_input"),
        ({"zero_value": 20.0}, "zero_value"),
    ]:
        args = {"n_steps": 1, **kwargs}
        with pytest.raises(ValueError, match=message):
            nowcast_model(history, "free", **args)


def test_nowcast_model_rejects_other_layouts(registry, history):
    import hashlib

    import onnx
    from onnx import TensorProto, helper

    from radarx import ml

    # two inputs
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, ["N", 3, None, None])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, ["N", 3, None, None])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 3, None, None])
    graph = helper.make_graph(
        [helper.make_node("Add", ["a", "b"], ["y"])], "two", [a, b], [y]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = registry / "twoin.onnx"
    onnx.save(model, str(path))
    ml.register_model(
        "twoin", path, hashlib.sha256(path.read_bytes()).hexdigest(), "MIT", "x"
    )
    with pytest.raises(ValueError, match="2 inputs"):
        nowcast_model(history, "twoin", 1)
    # an output that is not (1, k, ny, nx)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3, None, None])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["x"], ["y"], keepdims=0)], "m", [x], [y]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = registry / "mean.onnx"
    onnx.save(model, str(path))
    ml.register_model(
        "mean", path, hashlib.sha256(path.read_bytes()).hexdigest(), "MIT", "x"
    )
    with pytest.raises(ValueError, match="the model returned"):
        nowcast_model(history, "mean", 1)


def test_noise_amplitude_has_no_cross_from_the_edges_of_the_grid():
    rng = np.random.default_rng(0)
    ramp = np.linspace(0.0, 30.0, 60)[None, :] + np.linspace(0.0, 20.0, 60)[:, None]
    field = gaussian_filter(rng.standard_normal((60, 60)), 3) * 5 + 20.0 + ramp
    shape, offset = nowcasting._padded_shape(60, 60, 0.25)
    amplitude = nowcasting._noise_amplitude(field, offset, shape)
    assert amplitude.shape == (shape[0], shape[1] // 2 + 1)
    assert np.all(amplitude >= 0)
    padded = np.zeros(shape)
    padded[offset[0] : offset[0] + 60, offset[1] : offset[1] + 60] = (
        field - field.mean()
    )
    plain = np.abs(sfft.rfft2(padded))

    def axes_share(a):  # energy on the two axes of the spectrum
        return (a[0, 1:] ** 2).sum() + (a[1:, 0] ** 2).sum()

    assert (
        axes_share(amplitude) / (amplitude**2).sum()
        < axes_share(plain) / (plain**2).sum()
    )
