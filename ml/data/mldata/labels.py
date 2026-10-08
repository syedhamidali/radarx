"""
Label builders: turn radarx retrievals on a sweep into training samples.

Every builder takes xarray sweeps (xradar layout, ``(azimuth, range)``) and
the radarx products for them and returns an :class:`xarray.Dataset` on the
same native coordinates with the variables of one task's schema
(:mod:`mldata.schema`); :func:`mldata.polar.resample_sweep` then puts it on
the fixed polar grid. Builders that need randomness (artificial folding and
blockage) take a :class:`numpy.random.Generator`, so a sample is reproducible
from its seed.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

EARTH_RADIUS = 6371000.0
#: 4/3 effective Earth radius for standard refraction.
EFFECTIVE_RADIUS = 4.0 / 3.0 * EARTH_RADIUS


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------


def _ray_dim(da):
    return [d for d in da.dims if d != "range"][0]


def _field(ds, name, template):
    """``ds[name]`` as float32, or an all-NaN array shaped like ``template``."""
    if name in ds:
        return ds[name].astype("float32")
    return xr.full_like(template, np.nan, dtype="float32").rename(name)


def beam_height(rng, elevation, altitude=0.0):
    """
    Height of the beam centre above sea level (4/3 Earth radius model).

    Parameters
    ----------
    rng : array-like
        Slant range in metres.
    elevation : array-like
        Elevation angle in degrees (broadcast against ``rng``).
    altitude : float
        Antenna altitude above sea level in metres.

    Returns
    -------
    numpy.ndarray
        Height above sea level in metres.
    """
    r = np.asarray(rng, dtype=np.float64)
    el = np.deg2rad(np.asarray(elevation, dtype=np.float64))
    ke = EFFECTIVE_RADIUS
    return np.sqrt(r**2 + ke**2 + 2.0 * r * ke * np.sin(el)) - ke + float(altitude)


def standard_profile(freezing_level, lapse_rate=6.5, top=20000.0, step=100.0):
    """
    Temperature profile with a constant lapse rate through a freezing level.

    A stand-in for a sounding when only the freezing level of a case is known
    (it is what :func:`radarx.retrieve.hid` mostly needs). The temperature is
    clipped at -60 degC aloft.

    Parameters
    ----------
    freezing_level : float
        Height of the 0 degC level above sea level in metres.
    lapse_rate : float, optional
        Lapse rate in K per km. Default 6.5.

    Returns
    -------
    xarray.Dataset
        ``temperature`` (K) on ``height`` (m above sea level).
    """
    height = np.arange(0.0, top + step, step)
    t_c = -lapse_rate * (height - float(freezing_level)) / 1000.0
    t_c = np.maximum(t_c, -60.0)
    return xr.Dataset(
        {"temperature": ("height", t_c + 273.15, {"units": "K"})},
        coords={"height": ("height", height, {"units": "m"})},
    )


def gate_temperature(ds, profile, altitude):
    """Temperature (degC) at every gate of sweep ``ds`` from a profile."""
    da = ds[[v for v in ds.data_vars if "range" in ds[v].dims][0]]
    ray = _ray_dim(da)
    el = ds["elevation"].values if "elevation" in ds.coords else ds["sweep_fixed_angle"]
    el = np.broadcast_to(np.asarray(el, dtype=float).reshape(-1, 1), (ds.sizes[ray], 1))
    h = beam_height(ds["range"].values[None, :], el, altitude)
    t = profile["temperature"]
    units = t.attrs.get("units", "K")
    t_c = np.asarray(t.values, dtype=float)
    if units in ("K", "kelvin") or (units == "" and np.nanmean(t_c) > 100):
        t_c = t_c - 273.15
    z = np.asarray(profile["height"].values, dtype=float)
    order = np.argsort(z)
    vals = np.interp(h, z[order], t_c[order], left=np.nan, right=np.nan)
    return xr.DataArray(
        vals.astype("float32"),
        dims=(ray, "range"),
        coords={ray: ds[ray], "range": ds["range"]},
        name="TEMPERATURE",
        attrs={"units": "degC", "long_name": "temperature at the gate"},
    )


# --------------------------------------------------------------------------
# qc, kdp, hid
# --------------------------------------------------------------------------


def qc_sample(sweep, qc):
    """
    Echo-classification sample: polarimetric inputs and the radarx classes.

    Parameters
    ----------
    sweep : xarray.Dataset
        Sweep with ``DBZH`` and (where present) ``ZDR``, ``RHOHV``,
        ``PHIDP``, no-data codes masked as NaN.
    qc : xarray.Dataset
        :func:`radarx.retrieve.echo_mask` output for the sweep.

    Returns
    -------
    xarray.Dataset
    """
    dbz = sweep["DBZH"].astype("float32")
    out = {n: _field(sweep, n, dbz) for n in ("DBZH", "ZDR", "RHOHV", "PHIDP")}
    out["ECHO_CLASS"] = qc["ECHO_CLASS"].astype("int8")
    out["METEO_SCORE"] = qc["METEO_SCORE"].astype("float32")
    return xr.Dataset(out)


def kdp_sample(sweep, kdp):
    """
    KDP sample: raw phase and moments, radarx processed phase and KDP.

    Parameters
    ----------
    sweep : xarray.Dataset
        Sweep with ``PHIDP`` (raw), ``RHOHV``, ``DBZH``, ``ZDR``.
    kdp : xarray.Dataset
        :func:`radarx.retrieve.estimate_kdp` output for the sweep.
    """
    dbz = sweep["DBZH"].astype("float32")
    out = {n: _field(sweep, n, dbz) for n in ("PHIDP", "RHOHV", "DBZH", "ZDR")}
    out["PHIDP_processed"] = kdp["PHIDP_processed"].astype("float32")
    out["KDP"] = kdp["KDP"].astype("float32")
    return xr.Dataset(out)


def hid_sample(sweep, hid, temperature=None):
    """
    Hydrometeor-classification sample.

    Parameters
    ----------
    sweep : xarray.Dataset
        Sweep with ``DBZH``, ``ZDR``, ``RHOHV``, ``KDP`` and ``METEO_MASK``.
    hid : xarray.Dataset
        :func:`radarx.retrieve.hid` output for the sweep.
    temperature : xarray.DataArray, optional
        Temperature (degC) per gate, e.g. from :func:`gate_temperature`.
    """
    dbz = sweep["DBZH"].astype("float32")
    out = {n: _field(sweep, n, dbz) for n in ("DBZH", "ZDR", "RHOHV", "KDP")}
    if temperature is None:
        temperature = xr.full_like(dbz, np.nan)
    out["TEMPERATURE"] = temperature.astype("float32")
    if "METEO_MASK" in sweep:
        out["METEO_MASK"] = sweep["METEO_MASK"].fillna(False).astype(bool)
    else:
        out["METEO_MASK"] = np.isfinite(dbz)
    out["HID"] = hid["HID"].astype("int8")
    out["HID_confidence"] = hid["HID_confidence"].astype("float32")
    return xr.Dataset(out)


# --------------------------------------------------------------------------
# dealiasing
# --------------------------------------------------------------------------


def fold_velocity(velocity, nyquist):
    """
    Fold velocities into the Nyquist interval ``[-nyquist, nyquist)``.

    Parameters
    ----------
    velocity : array-like or xarray.DataArray
        True radial velocity (m/s); NaN stays NaN.
    nyquist : float
        Nyquist velocity (m/s).

    Returns
    -------
    folded, fold
        The measured (aliased) velocity and the integer fold ``k`` with
        ``velocity = folded + 2 k nyquist`` (int8, 0 where NaN).
    """
    v = velocity
    k = np.floor((v + nyquist) / (2.0 * nyquist))
    folded = v - 2.0 * nyquist * k
    if isinstance(k, xr.DataArray):
        k = k.fillna(0).astype("int8")
    else:
        k = np.where(np.isfinite(k), k, 0).astype("int8")
    return folded, k


def residual_jump_fraction(velocity, nyquist):
    """
    Share of neighbouring valid gate pairs differing by more than ``nyquist``.

    Pairs along the ray and between adjacent rays (with wrap-around) are
    counted. A dealiased field without errors has (almost) none; fold lines
    left in a field show up as jumps of about twice the Nyquist velocity.
    """
    v = np.asarray(velocity, dtype=np.float64)
    bad = total = 0
    for a, b in ((v[:, 1:], v[:, :-1]), (np.roll(v, -1, axis=0), v)):
        ok = np.isfinite(a) & np.isfinite(b)
        bad += int(np.count_nonzero(np.abs(a - b)[ok] > nyquist))
        total += int(np.count_nonzero(ok))
    return bad / max(total, 1)


#: Origins of the true velocity of a dealiasing sample.
TRUTH_SOURCES = {"observed_unaliased": 0, "radarx_dealiased": 1}


def velocity_truth(
    observed,
    dealiased,
    nyquist,
    *,
    sources=("observed_unaliased", "radarx_dealiased"),
    max_jump_fraction=5e-4,
    min_gates=2000,
):
    """
    True radial velocity of a sweep, or None if it is not trustworthy.

    A sweep is *observed unaliased* when radarx's dealiasing leaves every
    gate in its measured fold: the measurement is then the truth. Otherwise
    the radarx-dealiased field is the truth (*radarx dealiased*). Either is
    accepted only when the field has (almost) no residual jumps larger than
    the Nyquist velocity, i.e. no fold lines.

    Parameters
    ----------
    observed, dealiased : xarray.DataArray
        Measured velocity and :func:`radarx.retrieve.dealias_velocity` of it.
    nyquist : float
        Nyquist velocity of the measurement (m/s).
    sources : sequence of str
        Accepted origins, see :data:`TRUTH_SOURCES`.
    max_jump_fraction : float
        Largest accepted :func:`residual_jump_fraction`.
    min_gates : int
        Minimum number of valid gates.

    Returns
    -------
    (xarray.DataArray, int) or None
        The truth and its :data:`TRUTH_SOURCES` code.
    """
    obs = observed.where(np.abs(observed) <= 1.01 * nyquist)
    truth = dealiased.where(np.isfinite(obs))
    valid = np.isfinite(truth.values)
    if int(valid.sum()) < min_gates:
        return None
    moved = np.abs(truth.values - obs.values)[valid] > nyquist
    source = "radarx_dealiased" if moved.any() else "observed_unaliased"
    if source not in sources:
        return None
    if residual_jump_fraction(truth.values, nyquist) > max_jump_fraction:
        return None
    return truth.astype("float32"), TRUTH_SOURCES[source]


def dealias_sample(sweep, truth, nyquist, dealias=None):
    """
    Dealiasing sample: the truth folded at an artificial Nyquist velocity.

    Parameters
    ----------
    sweep : xarray.Dataset
        Sweep with ``DBZH`` (optional) and coordinates.
    truth : xarray.DataArray
        True radial velocity on the sweep (see :func:`velocity_truth`).
    nyquist : float
        Artificial Nyquist velocity (m/s).
    dealias : callable, optional
        ``dealias(sweep_with_VRADH, nyquist) -> DataArray``, the baseline;
        by default :func:`radarx.retrieve.dealias_velocity`.

    Returns
    -------
    xarray.Dataset
        ``VRADH_folded``, ``DBZH``, ``VRADH``, ``FOLD`` and ``VRADH_radarx``,
        with the attribute ``folded_fraction``.
    """
    folded, k = fold_velocity(truth, float(nyquist))
    folded = folded.astype("float32")
    if dealias is None:
        from radarx.retrieve import dealias_velocity

        def dealias(ds, vn):
            return dealias_velocity(ds, "VRADH", vn)

    base = sweep.drop_vars(
        [v for v in sweep.data_vars if "range" in sweep[v].dims]
    ).assign(VRADH=folded)
    base = base.drop_vars("nyquist_velocity", errors="ignore")
    baseline = dealias(base, float(nyquist)).astype("float32")
    valid = np.isfinite(truth.values)
    out = xr.Dataset(
        {
            "VRADH_folded": folded,
            "DBZH": _field(sweep, "DBZH", folded),
            "VRADH": truth.astype("float32"),
            "FOLD": k.where(np.isfinite(truth), 0).astype("int8"),
            "VRADH_radarx": baseline.where(np.isfinite(truth)),
        }
    )
    out = out.drop_vars("nyquist_velocity", errors="ignore")
    out.attrs["folded_fraction"] = float(np.mean(k.values[valid] != 0))
    return out


# --------------------------------------------------------------------------
# blockage / inpainting
# --------------------------------------------------------------------------


def blockage_field(
    azimuth,
    rng,
    generator,
    *,
    n_sectors=(1, 3),
    width=(2.0, 20.0),
    start_range=(2000.0, 60000.0),
    partial_probability=0.5,
    partial_fraction=(0.3, 0.9),
):
    """
    Artificial beam blockage: sectors blocked from an obstacle range outward.

    Like a real obstacle, a blocked sector starts at some range and extends to
    the end of the ray. A sector blocks the beam totally (fraction 1) or
    partially (a power fraction drawn from ``partial_fraction``); overlapping
    sectors combine as ``1 - prod(1 - f)``.

    Parameters
    ----------
    azimuth, rng : array-like
        Azimuths (degrees) and ranges (m) of the grid.
    generator : numpy.random.Generator
        Random source.
    n_sectors : (int, int)
        Inclusive range of the number of sectors.
    width : (float, float)
        Range of the sector width (degrees).
    start_range : (float, float)
        Range of the obstacle distance (m).
    partial_probability : float
        Chance that a sector is partial rather than total.
    partial_fraction : (float, float)
        Range of the blocked power fraction of partial sectors.

    Returns
    -------
    numpy.ndarray
        Blocked fraction (float32, 0-1) on ``(azimuth, range)``.
    """
    az = np.asarray(azimuth, dtype=np.float64)
    r = np.asarray(rng, dtype=np.float64)
    passed = np.ones((az.size, r.size))
    n = int(generator.integers(n_sectors[0], n_sectors[1] + 1))
    for _ in range(n):
        centre = generator.uniform(0.0, 360.0)
        half = 0.5 * generator.uniform(*width)
        r0 = generator.uniform(*start_range)
        if generator.uniform() < partial_probability:
            frac = generator.uniform(*partial_fraction)
        else:
            frac = 1.0
        d = np.abs((az - centre + 180.0) % 360.0 - 180.0)
        inside = (d <= half)[:, None] & (r >= r0)[None, :]
        passed = np.where(inside, passed * (1.0 - frac), passed)
    return (1.0 - passed).astype("float32")


def apply_blockage(dbz, blockage, total=0.99):
    """
    Reflectivity seen through a blockage.

    Partially blocked gates lose ``-10 log10(1 - f)`` dB of power; gates
    blocked by at least ``total`` have no data (NaN).
    """
    f = np.asarray(blockage, dtype=np.float64)
    loss = 10.0 * np.log10(np.clip(1.0 - f, 1e-6, 1.0))
    out = np.where(f >= total, np.nan, np.asarray(dbz, dtype=np.float64) + loss)
    if isinstance(dbz, xr.DataArray):
        return dbz.copy(data=out.astype("float32"))
    return out.astype("float32")


def inpaint_sample(dbz, generator, **options):
    """
    Inpainting sample on the fixed grid from quality-controlled reflectivity.

    Parameters
    ----------
    dbz : xarray.DataArray
        Clean reflectivity on ``(azimuth, range)`` (NaN: no echo).
    generator : numpy.random.Generator
        Random source.
    **options
        Options of :func:`blockage_field`.

    Returns
    -------
    xarray.Dataset
        ``DBZH_blocked``, ``BLOCKAGE`` and ``DBZH``.
    """
    ray = _ray_dim(dbz)
    block = blockage_field(dbz[ray].values, dbz["range"].values, generator, **options)
    block = xr.DataArray(block, dims=(ray, "range"), coords=dbz.coords)
    return xr.Dataset(
        {
            "DBZH_blocked": apply_blockage(dbz, block),
            "BLOCKAGE": block,
            "DBZH": dbz.astype("float32"),
        }
    )


# --------------------------------------------------------------------------
# gridded tasks
# --------------------------------------------------------------------------


def grid_axes(cfg):
    """``(x, y, z)`` axes in metres of the nowcasting grid config."""
    n = int(cfg.get("size", 256))
    dx = float(cfg.get("spacing", 1000.0))
    x = (np.arange(n) - (n - 1) / 2.0) * dx
    z = np.asarray(cfg.get("z", [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]), float)
    return x, x.copy(), z


def composite_frame(dtree, cfg, n_threads=None):
    """
    Column-maximum reflectivity of a quality-controlled volume.

    The volume is cone-gridded (:func:`radarx.grid.grid_cones`) at the
    heights ``cfg["z"]``; the maximum over height is the composite. Cells
    within ``cfg["max_range"]`` of the radar without echo get
    ``cfg["no_echo"]`` (default -10 dBZ), cells beyond are NaN.

    Returns
    -------
    xarray.DataArray
        ``DBZH`` on ``(y, x)`` with a scalar ``time`` coordinate.
    """
    from radarx.grid import grid_cones

    x, y, z = grid_axes(cfg)
    grid = grid_cones(dtree, "DBZH", x, y, z, n_threads=n_threads)
    comp = grid["DBZH"].max("z", skipna=True)
    cover = coverage(x, y, float(cfg.get("max_range", 230e3)))
    comp = comp.where(~(cover & comp.isnull()), float(cfg.get("no_echo", -10.0)))
    comp = comp.where(cover).astype("float32")
    keep = [c for c in comp.coords if c in ("x", "y", "time")]
    return comp.reset_coords([c for c in comp.coords if c not in keep], drop=True)


def coverage(x, y, max_range):
    """True within ``max_range`` (m) of the radar at ``(0, 0)``."""
    xx, yy = np.meshgrid(np.asarray(x), np.asarray(y))
    return xr.DataArray(
        np.hypot(xx, yy) <= max_range,
        dims=("y", "x"),
        coords={"y": np.asarray(y), "x": np.asarray(x)},
    )


def nowcast_sample(frames, n_input=2, motion_options=None, n_threads=None):
    """
    Nowcasting sample from consecutive composite frames of one radar.

    The echo motion is estimated from the last two *input* frames only
    (:func:`radarx.retrieve.estimate_motion`, tiled), and the last input
    frame is extrapolated to each target frame time with
    :func:`radarx.retrieve.advect`: a physical baseline that never sees the
    targets.

    Parameters
    ----------
    frames : sequence of xarray.DataArray
        ``n_input + n_target`` composites on ``(y, x)`` with scalar ``time``,
        in time order (e.g. from :func:`composite_frame`).
    n_input : int
        Number of input frames (at least 2).
    motion_options : dict, optional
        Options of :func:`radarx.retrieve.estimate_motion`.

    Returns
    -------
    xarray.Dataset or None
        ``DBZH`` (inputs) on ``(lead, y, x)`` and ``frame_time``;
        ``DBZH_target`` and ``DBZH_extrapolated`` on ``(lead_target, y, x)``
        and ``target_time``; ``u``, ``v`` on ``(y, x)``. Leads count volumes
        relative to the last input (0). None if no reliable motion is found.
    """
    import warnings

    from radarx.retrieve import advect, estimate_motion

    if n_input < 2 or len(frames) <= n_input:
        raise ValueError("need at least 2 input frames and 1 target frame")
    opts = {"tile": 64e3, "floor": 10.0}
    opts.update(motion_options or {})
    observed = np.isfinite(frames[n_input - 1]) & np.isfinite(frames[n_input - 2])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        motion = estimate_motion(
            frames[n_input - 2],
            frames[n_input - 1],
            observed=observed,
            n_threads=n_threads,
            **opts,
        )
    if not np.all(np.isfinite(motion["u"].values)):
        return None
    leads = np.arange(len(frames)) - (n_input - 1)
    lead_in = xr.DataArray(leads[:n_input], dims="lead", name="lead")
    lead_out = xr.DataArray(leads[n_input:], dims="lead_target", name="lead_target")

    def stack(items, dim):
        return xr.concat([f.drop_vars("time") for f in items], dim=dim)

    last = frames[n_input - 1]
    extrap = [
        advect(last, motion, time=f["time"].values, n_threads=n_threads)
        for f in frames[n_input:]
    ]
    u = motion["u"].broadcast_like(last.drop_vars("time"))
    v = motion["v"].broadcast_like(last.drop_vars("time"))
    out = xr.Dataset(
        {
            "DBZH": stack(frames[:n_input], lead_in).astype("float32"),
            "DBZH_target": stack(frames[n_input:], lead_out).astype("float32"),
            "frame_time": (
                "lead",
                np.array([f["time"].values for f in frames[:n_input]]),
            ),
            "target_time": (
                "lead_target",
                np.array([f["time"].values for f in frames[n_input:]]),
            ),
            "u": u.astype("float32"),
            "v": v.astype("float32"),
            "DBZH_extrapolated": stack(extrap, lead_out).astype("float32"),
        }
    )
    out.attrs["motion_quality"] = float(motion["quality"])
    return out.assign_coords(time=last["time"].values)


def multidoppler_sample(volumes, cfg, time=None, motion=None, n_threads=None):
    """
    Multi-Doppler sample: gridded radial velocities and the retrieved wind.

    Parameters
    ----------
    volumes : sequence of xarray.DataTree
        Two or more quality-controlled volumes whose ``VRADH`` is already
        dealiased.
    cfg : dict
        Grid: ``x``, ``y``, ``z`` as ``[start, stop, step]`` in metres
        (``x``/``y`` relative to the first radar); optional ``weights`` for
        :func:`radarx.retrieve.multi_doppler`.
    time, motion
        Analysis time and storm motion for
        :func:`radarx.retrieve.multi_doppler_input` (both or neither).

    Returns
    -------
    xarray.Dataset
        ``VRADH``, ``DBZH``, ``beam_azimuth``, ``beam_elevation`` on
        ``(radar_index, z, y, x)``; ``u``, ``v``, ``w``, ``n_radars``,
        ``beam_crossing_angle`` on ``(z, y, x)``.
    """
    from radarx.retrieve import multi_doppler, multi_doppler_input

    def axis(key):
        start, stop, step = (float(a) for a in cfg[key])
        return np.arange(start, stop + 0.5 * step, step)

    grids = multi_doppler_input(
        volumes,
        x=axis("x"),
        y=axis("y"),
        z=axis("z"),
        time=time,
        motion=motion,
        n_threads=n_threads,
    )
    wind = multi_doppler(grids, weights=cfg.get("weights"), n_threads=n_threads)
    radar_vars = {
        "VRADH": "VRADH",
        "DBZH": "DBZH",
        "beam_azimuth": "azimuth",
        "beam_elevation": "elevation",
    }
    out = {}
    for name, src in radar_vars.items():
        if src in grids:
            out[name] = grids[src].astype("float32")
        else:
            out[name] = xr.full_like(grids["VRADH"], np.nan, dtype="float32")
    for name in ("u", "v", "w", "beam_crossing_angle"):
        out[name] = wind[name].astype("float32")
    out["n_radars"] = wind["n_radars"].astype("int8")
    ds = xr.Dataset(out).rename({"radar": "radar_index"})
    keep = {"x", "y", "z", "radar_index"}
    ds = ds.reset_coords([c for c in ds.coords if c not in keep], drop=True)
    names = [str(n) for n in grids["radar_name"].values]
    return ds.assign_coords(radar=("radar_index", names))
