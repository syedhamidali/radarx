# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Single-Doppler Wind Retrieval
=============================

Three-dimensional winds ``(u, v, w)`` from the radial velocity of a single
Doppler radar, its reflectivity and a background (sounding or ERA5).

One radar measures only the wind component along its beams, so the other
components must come from prior knowledge. Two kinds of prior are offered:

- **variational** (``model=None``): the variational cost of
  :func:`radarx.retrieve.multi_doppler` with the one radar's observation
  term, anelastic mass continuity, smoothness and the background (Gao et
  al. 1999). The cross-beam wind then comes from the background and from
  mass continuity only.
- **physics-informed network** (``model=...``): a convolutional network,
  trained on multi-Doppler retrievals and analytic flows with the same
  variational cost as a physics loss (``ml/models/single_doppler`` in the
  radarx repository), predicts the wind. Run through ONNX Runtime
  (``pip install radarx[ml]``). With ``refine=True`` (default) the
  prediction is then used as the background of the variational retrieval,
  which makes the result fit the observed radial velocities and the
  continuity equation.

The heavy work runs in compiled code: the network in ONNX Runtime, the
variational cost and its adjoint in the multithreaded kernel of
:func:`radarx.retrieve.multi_doppler`.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = ["single_doppler_winds"]

__doc__ = __doc__.format("\n   ".join(__all__))

import os
import warnings

import numpy as np
import xarray as xr

from .._registry import accessor_method

# -- input features of the network (shared with the training code) --------

FEATURES = (
    "radial_velocity",
    "radial_velocity_mask",
    "reflectivity",
    "reflectivity_mask",
    "beam_x",
    "beam_y",
    "beam_z",
    "background_u",
    "background_v",
    "height",
    "range",
)
FEATURE_VERSION = "1"
VR_SCALE = 30.0  # m s-1
DBZ_SCALE = 60.0  # dBZ
WIND_SCALE = 30.0  # m s-1
HEIGHT_SCALE = 12000.0  # m
RANGE_SCALE = 150000.0  # m
DEFAULT_SPACING = {"dx": 1000.0, "dy": 1000.0, "dz": 500.0}
DEFAULT_PAD_MULTIPLE = 4
REFINE_WEIGHTS = {"background": 0.05, "background_w": 0.05}
_TILE = 160  # horizontal tile (cells) of the network
_OVERLAP = 32


def _features(vr, dbz, coef, u_bg, v_bg, z, distance):
    """
    Network input features from NumPy arrays.

    Parameters
    ----------
    vr, dbz : numpy.ndarray
        Radial velocity (m s-1) and reflectivity (dBZ) on ``(z, y, x)``, NaN
        where missing.
    coef : numpy.ndarray
        Beam direction cosines ``(cos el sin az, cos el cos az, sin el)`` on
        ``(3, z, y, x)``.
    u_bg, v_bg : numpy.ndarray
        Background wind along the grid axes on ``(z, y, x)``.
    z : numpy.ndarray
        Heights of the levels (m).
    distance : numpy.ndarray
        Horizontal distance from the radar on ``(y, x)`` (m).

    Returns
    -------
    numpy.ndarray
        ``float32`` array of shape ``(len(FEATURES), z, y, x)``.
    """
    shape = vr.shape
    ok_vr = np.isfinite(vr)
    ok_dbz = np.isfinite(dbz)
    out = np.empty((len(FEATURES),) + shape, dtype=np.float32)
    out[0] = np.where(ok_vr, vr, 0.0) / VR_SCALE
    out[1] = ok_vr
    out[2] = np.where(ok_dbz, dbz, 0.0) / DBZ_SCALE
    out[3] = ok_dbz
    out[4:7] = np.nan_to_num(coef)
    out[7] = np.nan_to_num(u_bg) / WIND_SCALE
    out[8] = np.nan_to_num(v_bg) / WIND_SCALE
    out[9] = np.asarray(z, dtype=np.float64)[:, None, None] / HEIGHT_SCALE
    out[10] = np.asarray(distance)[None] / RANGE_SCALE
    return out


# -- model loading ---------------------------------------------------------


class _OnnxFile:
    """Minimal ``run`` interface around a local ONNX file."""

    def __init__(self, path, providers=None):
        try:
            import onnxruntime as ort
        except ImportError as err:  # pragma: no cover - depends on the env
            raise ImportError(
                "running a network needs onnxruntime: pip install radarx[ml]"
            ) from err
        self.path = os.fspath(path)
        self.session = ort.InferenceSession(
            self.path, providers=providers or ["CPUExecutionProvider"]
        )
        self.info = dict(self.session.get_modelmeta().custom_metadata_map)
        self.info.setdefault("name", os.path.basename(self.path))

    def run(self, inputs):
        names = [o.name for o in self.session.get_outputs()]
        return dict(zip(names, self.session.run(names, inputs)))


def _resolve_model(model, providers):
    """A model object with ``run`` and its metadata dictionary."""
    if hasattr(model, "run"):
        m = model
    else:
        try:
            from .. import ml  # radarx[ml]: model registry and ONNX Runtime
        except ImportError:
            ml = None
        if ml is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*not in the model registry")
                m = ml.load_model(model, providers=providers)
        elif isinstance(model, (str, os.PathLike)) and os.path.isfile(model):
            m = _OnnxFile(model, providers)
        else:
            raise ImportError(
                f"cannot load {model!r}: the radarx.ml model registry is not "
                "available; pass a local ONNX file or a model object"
            )
    info = {k: v for k, v in dict(getattr(m, "info", None) or {}).items() if v}
    session = getattr(m, "session", None)
    if session is not None and hasattr(session, "get_modelmeta"):
        for k, v in session.get_modelmeta().custom_metadata_map.items():
            info.setdefault(k, v)
    return m, info


def _model_spacing(info):
    return {k: float(info.get(k, v)) for k, v in DEFAULT_SPACING.items()}


# -- input handling --------------------------------------------------------


def _single_radar(grid_or_volume, x, y, z, radar, velocity, reflectivity, **kw):
    """One-radar Dataset on ``(radar, z, y, x)`` with beam geometry."""
    from .multidoppler import multi_doppler_input, radar_geometry

    obj = grid_or_volume
    if isinstance(obj, xr.DataTree):
        if x is None or y is None or z is None:
            raise ValueError("a radar volume needs the grid coordinates x, y and z")
        return multi_doppler_input(
            [obj], x, y, z, velocity=velocity, reflectivity=reflectivity, **kw
        )
    if not isinstance(obj, xr.Dataset):
        raise TypeError("expected an xarray.Dataset grid or an xarray.DataTree volume")
    ds = obj
    if velocity not in ds:
        raise ValueError(f"no {velocity!r} in the grid")
    if "radar" in ds.dims:
        if ds.sizes["radar"] > 1:
            if radar is None:
                raise ValueError(
                    "the grid has several radars: choose one with radar=<index>"
                )
            ds = ds.isel(radar=[int(radar)])
    else:
        ds = ds.expand_dims("radar")
        for name in ("radar_x", "radar_y", "radar_altitude"):
            if name in ds and "radar" not in ds[name].dims:
                ds[name] = ds[name].expand_dims("radar")
    for name in ("x", "y", "z"):
        if ds.sizes.get(name, 0) < 3:
            raise ValueError("the grid needs at least 3 points along x, y and z")
    if "azimuth" not in ds or "elevation" not in ds:
        ds = radar_geometry(ds)
    return ds


def _on_spacing(ds, background, spacing, velocity, reflectivity):
    """``ds`` and ``background`` interpolated to the network's grid spacing."""
    from .multidoppler import _spacing, radar_geometry

    grid = {
        "x": (_spacing(ds["x"]), spacing["dx"]),
        "y": (_spacing(ds["y"]), spacing["dy"]),
        "z": (_spacing(ds["z"]), spacing["dz"]),
    }
    if all(np.isclose(abs(a), b, rtol=1e-3) for a, b in grid.values()):
        return ds, background, False
    new = {}
    for name, (d, d_net) in grid.items():
        c = ds[name].values
        n = int(np.floor(abs(c[-1] - c[0]) / d_net)) + 1
        new[name] = c[0] + np.sign(d) * d_net * np.arange(n)
    keep = [v for v in (velocity, reflectivity) if v in ds]
    fine = ds[keep].interp(new)
    for name in ("radar_x", "radar_y", "radar_altitude"):
        fine[name] = ds[name]
    fine = radar_geometry(fine)
    bg = None
    if background is not None:
        bg = background[[v for v in ("u", "v") if v in background]]
        bg = bg.interp({k: v for k, v in new.items() if k in bg.dims})
    return fine, bg, True


def _full_background(background, ds):
    """Background with its 3-D fields (e.g. a profile on ``z``) on the full grid."""
    dims = ("z", "y", "x")
    template = xr.DataArray(
        np.zeros([ds.sizes[d] for d in dims]),
        dims=dims,
        coords={d: ds[d].values for d in dims},
    )
    out = background.copy()
    for name in ("u", "v", "w", "air_density"):
        if name in out and set(out[name].dims) <= set(dims):
            da = out[name].drop_vars(
                [c for c in out[name].coords if c not in dims], errors="ignore"
            )
            out[name] = da.broadcast_like(template).transpose(*dims)
    return out


def _beam_coefficients(ds):
    dims = ("z", "y", "x")
    el = np.radians(ds["elevation"].isel(radar=0).transpose(*dims).values)
    az = np.radians(ds["azimuth"].isel(radar=0).transpose(*dims).values)
    return np.stack([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])


def _grid_features(ds, background, velocity, reflectivity):
    """Network input features of a one-radar Dataset."""
    dims = ("z", "y", "x")
    shape = tuple(ds.sizes[d] for d in dims)
    vr = ds[velocity].isel(radar=0).transpose(*dims).values.astype(np.float64)
    if reflectivity in ds:
        dbz = ds[reflectivity]
        if "radar" in dbz.dims:
            dbz = dbz.isel(radar=0)
        dbz = dbz.transpose(*dims).values.astype(np.float64)
    else:
        dbz = np.full(shape, np.nan)
    ub = np.zeros(shape)
    vb = np.zeros(shape)
    if background is not None:
        bg = background.transpose(..., *[d for d in dims if d in background.dims])
        if "u" in bg:
            ub = np.broadcast_to(bg["u"].values, shape)
        if "v" in bg:
            vb = np.broadcast_to(bg["v"].values, shape)
    xx = ds["x"].values[None, :] - float(ds["radar_x"].values[0])
    yy = ds["y"].values[:, None] - float(ds["radar_y"].values[0])
    return _features(
        vr, dbz, _beam_coefficients(ds), ub, vb, ds["z"].values, np.hypot(xx, yy)
    )


def _pad_to(a, multiple):
    """Pad the last three axes of ``a`` (C, z, y, x) to multiples of ``multiple``."""
    pads = [(0, 0)] + [(0, (-n) % multiple) for n in a.shape[1:]]
    return np.pad(a, pads, mode="edge"), a.shape[1:]


def _tiles(n, tile, overlap):
    """Start indices of overlapping tiles covering ``range(n)``."""
    if n <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, n - tile, step)) + [n - tile]
    return sorted(set(starts))


def _blend(n, start, size, n_total, overlap):
    """1-D weight of a tile: cosine ramps where it overlaps its neighbours."""
    w = np.ones(size)
    ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(overlap) + 0.5) / overlap)
    if start > 0:
        w[:overlap] = ramp
    if start + size < n_total:
        w[-overlap:] = ramp[::-1]
    return w


def _predict(model, feats, pad_multiple, tile=_TILE, overlap=_OVERLAP):
    """Run the network on ``(C, z, y, x)`` features, in horizontal tiles."""
    names = getattr(model, "input_names", None) or ["features"]
    nz, ny, nx = feats.shape[1:]
    tile = max(int(tile), 2 * overlap)
    tile -= tile % pad_multiple
    out = np.zeros((3, nz, ny, nx))
    wsum = np.zeros((ny, nx))
    for j0 in _tiles(ny, tile, overlap):
        for i0 in _tiles(nx, tile, overlap):
            sub = feats[:, :, j0 : j0 + tile, i0 : i0 + tile]
            padded, shape = _pad_to(sub, pad_multiple)
            padded[[0, 1, 2, 3]] *= _valid_mask(padded.shape, shape)
            res = model.run({names[0]: padded[None].astype(np.float32)})
            wind = np.asarray(next(iter(res.values())))[0]
            wind = wind[:, : shape[0], : shape[1], : shape[2]]
            wy = (
                _blend(ny, j0, shape[1], ny, overlap)
                if ny > tile
                else np.ones(shape[1])
            )
            wx = (
                _blend(nx, i0, shape[2], nx, overlap)
                if nx > tile
                else np.ones(shape[2])
            )
            w2 = wy[:, None] * wx[None, :]
            out[:, :, j0 : j0 + shape[1], i0 : i0 + shape[2]] += wind * w2
            wsum[j0 : j0 + shape[1], i0 : i0 + shape[2]] += w2
    return out / wsum


def _valid_mask(padded_shape, shape):
    """1 inside the unpadded region, 0 in the padding (no observations there)."""
    m = np.zeros(padded_shape[1:], dtype=np.float32)
    m[: shape[0], : shape[1], : shape[2]] = 1.0
    return m


def _back_to_grid(wind, fine, ds):
    """Interpolate the network wind from its grid back to the grid of ``ds``."""
    da = xr.DataArray(
        wind,
        dims=("component", "z", "y", "x"),
        coords={"z": fine["z"].values, "y": fine["y"].values, "x": fine["x"].values},
    )
    back = da.interp(
        z=ds["z"].values,
        y=ds["y"].values,
        x=ds["x"].values,
        kwargs={"fill_value": None},
    )
    return back.values


# -- public API ------------------------------------------------------------


def single_doppler_winds(
    grid_or_volume,
    background=None,
    *,
    model=None,
    refine=True,
    x=None,
    y=None,
    z=None,
    radar=None,
    velocity="VRADH",
    reflectivity="DBZH",
    weights=None,
    fall_speed_correction=True,
    w_boundary="bottom",
    providers=None,
    engine="auto",
    n_threads=None,
    **input_kwargs,
):
    """
    Retrieve the three-dimensional wind from one Doppler radar.

    Without ``model`` this is the variational single-Doppler retrieval: the
    cost function of :func:`radarx.retrieve.multi_doppler` (Gao et al. 1999
    [1]_) with the observation term of the one radar, the anelastic mass
    continuity equation, smoothness and the background. Along the beams the
    wind follows the observations; across them it comes from the background
    and from mass continuity.

    With ``model`` a physics-informed network predicts the wind from the
    radial velocity, the reflectivity, the beam geometry and the background
    wind. It was trained against multi-Doppler retrievals of NEXRAD radar
    pairs and analytic flows, with the variational cost above (observation,
    mass continuity, smoothness and background terms of the one radar) as a
    physics loss; see ``ml/models/single_doppler/README.md`` in the radarx
    repository. With ``refine=True`` the prediction then serves as the
    background of the variational retrieval (weights
    ``{"background": 0.05, "background_w": 0.05}`` unless overridden), so
    that the final wind fits the observed radial velocities and the
    continuity equation while taking its cross-beam structure from the
    network. The network runs on its own grid spacing (``dx``, ``dy``,
    ``dz`` in the model's metadata, by default 1 km, 1 km, 500 m); other
    grids are interpolated to it and back.

    Parameters
    ----------
    grid_or_volume : xarray.Dataset or xarray.DataTree
        A gridded radar (``velocity``, optionally ``reflectivity``, on
        ``(z, y, x)`` or ``(radar, z, y, x)``, with ``radar_x``, ``radar_y``
        and ``radar_altitude``, e.g. from
        :func:`radarx.retrieve.multi_doppler_input`), or a radar volume,
        which is then gridded onto ``x``, ``y``, ``z`` with
        :func:`radarx.retrieve.multi_doppler_input`. Radial velocities must
        be dealiased.
    background : xarray.Dataset, optional
        Background on the grid from :func:`radarx.io.sounding.era5_column`
        or :func:`radarx.io.sounding.profile_to_grid` (``u``, ``v`` along the
        grid axes, optionally ``air_density`` and ``freezing_level``). Both
        methods need it to say anything about the cross-beam wind.
    model : str, path or object, optional
        The network: a model name of the ``radarx.ml`` registry, a local
        ONNX file, or an object with a ``run(dict) -> dict`` method (such as
        ``radarx.ml.load_model(...)``). The network takes the input
        ``features`` (``float32[N, C, Z, Y, X]``) and returns the wind
        (``float32[N, 3, Z, Y, X]``, m s-1). Default: no network
        (variational retrieval).
    refine : bool, optional
        Use the network's wind as the background of a variational retrieval.
        Default True. Ignored without ``model``.
    x, y, z : array-like, optional
        Grid coordinates (m) when ``grid_or_volume`` is a volume.
    radar : int, optional
        Index of the radar to use when the grid holds several.
    velocity, reflectivity : str, optional
        Field names. Default ``"VRADH"`` and ``"DBZH"``.
    weights : dict, optional
        Overrides of the weights of :func:`radarx.retrieve.multi_doppler`.
    fall_speed_correction : bool, optional
        Correct for the precipitation fall speed. Default True.
    w_boundary : str or sequence of str, optional
        Where ``w = 0`` (see :func:`radarx.retrieve.multi_doppler`).
        Default ``"bottom"``.
    providers : list of str, optional
        ONNX Runtime execution providers. Default: CPU.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation of the variational cost (and of the gridding).
    n_threads : int, optional
        Threads for the compiled kernels. Default: all cores.
    **input_kwargs
        Further options of :func:`radarx.retrieve.multi_doppler_input` when
        gridding a volume (``origin``, ``time``, ``motion``, ...).

    Returns
    -------
    xarray.Dataset
        ``u``, ``v`` (along the grid axes) and ``w`` on ``(z, y, x)`` with the
        grid's coordinates. The variational retrieval adds its diagnostics
        (``fall_speed``, ``mass_residual``, ``n_radars``, ``vr_residual``,
        ``cost``, ``cost_history``). With a network, ``u_network``,
        ``v_network`` and ``w_network`` hold its raw prediction and the
        attributes ``ml_model``, ``ml_model_version`` and
        ``ml_model_licence`` name it.

    References
    ----------
    .. [1] Gao, J., M. Xue, A. Shapiro, and K. K. Droegemeier, 1999: A
       variational method for the analysis of three-dimensional wind fields
       from two Doppler radars. *Mon. Wea. Rev.*, **127**, 2128-2142,
       https://doi.org/10.1175/1520-0493(1999)127<2128:AVMFTA>2.0.CO;2

    Examples
    --------
    >>> grid = radarx.retrieve.multi_doppler_input([kgwx], x, y, z)  # doctest: +SKIP
    >>> bg = radarx.io.sounding.era5_column(grid)  # doctest: +SKIP
    >>> wind = radarx.retrieve.single_doppler_winds(grid, bg)  # doctest: +SKIP
    >>> wind = radarx.retrieve.single_doppler_winds(
    ...     grid, bg, model="single_doppler.onnx")  # doctest: +SKIP
    """
    from .multidoppler import DEFAULT_WEIGHTS, multi_doppler

    kw = dict(n_threads=n_threads, engine=engine, **input_kwargs)
    ds = _single_radar(grid_or_volume, x, y, z, radar, velocity, reflectivity, **kw)
    if background is None:
        warnings.warn(
            "no background: the cross-beam wind is constrained by mass "
            "continuity only",
            stacklevel=2,
        )
    elif not isinstance(background, xr.Dataset):
        raise TypeError("background must be an xarray.Dataset")
    else:
        background = _full_background(background, ds)
    variational = dict(
        velocity=velocity,
        reflectivity=reflectivity,
        fall_speed_correction=fall_speed_correction,
        w_boundary=w_boundary,
        engine=engine,
        n_threads=n_threads,
    )
    if model is None:
        out = multi_doppler(ds, background, weights=weights, **variational)
        out.attrs["method"] = (
            "variational single-Doppler analysis (cost of Gao et al. 1999 with "
            "one radar)"
        )
        return out.squeeze("radar", drop=False)

    net, info = _resolve_model(model, providers)
    spacing = _model_spacing(info)
    pad = int(info.get("pad_multiple", DEFAULT_PAD_MULTIPLE))
    fine, fine_bg, regridded = _on_spacing(
        ds, background, spacing, velocity, reflectivity
    )
    feats = _grid_features(fine, fine_bg, velocity, reflectivity)
    wind = _predict(net, feats, pad)
    if regridded:
        wind = _back_to_grid(wind, fine, ds)
    dims = ("z", "y", "x")
    coords = {c: ds.coords[c] for c in ds.coords if set(ds.coords[c].dims) <= set(dims)}
    names = {"u": "x_wind", "v": "y_wind", "w": "upward_air_velocity"}
    network = xr.Dataset(coords=coords)
    for q, c in enumerate("uvw"):
        network[c] = (dims, wind[q].astype(np.float32))
    if refine:
        prior = network.copy()
        if background is not None:
            for name in ("air_density", "freezing_level"):
                if name in background:
                    prior[name] = background[name]
        w_ = dict(REFINE_WEIGHTS)
        w_.update(weights or {})
        unknown = set(w_) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown weights: {sorted(unknown)}")
        out = multi_doppler(
            ds, prior, weights=w_, first_guess=network, **variational
        ).squeeze("radar", drop=False)
        method = (
            "physics-informed network prior refined by the variational "
            "single-Doppler analysis (Gao et al. 1999 cost)"
        )
    else:
        out = network.copy()
        method = "physics-informed network"
    for c in "uvw":
        out[c].attrs = {
            "standard_name": names[c],
            "long_name": f"retrieved {'vertical air velocity' if c == 'w' else 'wind component along the grid ' + ('x' if c == 'u' else 'y') + ' axis'}",
            "units": "m s-1",
        }
        out[f"{c}_network"] = network[c].assign_attrs(
            long_name=f"{c} predicted by the network", units="m s-1"
        )
    out.attrs.update(
        method=method,
        ml_model=str(info.get("name", model if isinstance(model, str) else "")),
        ml_model_version=str(info.get("version", "")),
        ml_model_licence=str(info.get("licence", info.get("license", ""))),
    )
    return out


@accessor_method("dataset", "datatree", name="single_doppler_winds")
def _single_doppler_winds_accessor(self, background=None, **kwargs):
    """
    Retrieve the 3D wind from one Doppler radar.

    Parameters
    ----------
    background : xarray.Dataset, optional
        Background on the grid (``grid.radarx.background()``).
    **kwargs
        Options of :func:`radarx.retrieve.single_doppler_winds` (``model``,
        ``refine``, and ``x``, ``y``, ``z`` for a volume).

    Returns
    -------
    xarray.Dataset
        ``u``, ``v``, ``w`` on the grid.

    See Also
    --------
    radarx.retrieve.single_doppler_winds
    """
    return single_doppler_winds(self.xarray_obj, background, **kwargs)
