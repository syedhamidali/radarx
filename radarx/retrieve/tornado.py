#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tornado Detection
=================

Two independent tornado diagnostics on WSR-88D volumes:

``tornado_probability``
    The convolutional neural network (CNN) baseline of the TorNet benchmark
    (Veillette et al. 2025 [1]_), run with ONNX Runtime through
    :mod:`radarx.ml`. The network sees image chips of 120 rays (60° at 0.5°
    spacing) by 240 gates of 250 m of the two lowest tilts (0.5° and 0.9°) of
    six variables: reflectivity, dealiased radial velocity, specific
    differential phase, copolar correlation coefficient, differential
    reflectivity and spectrum width, plus a mask of range-folded gates (13
    channels), and the range of every gate and its inverse (CoordConv layers)
    (Veillette et al. 2025 [1]_, sections 2d and 3b). Each variable is scaled
    to about [-1, 1] with fixed bounds, and gates without data are set to -3.
    The network ends in 1 x 1 convolutions that give a tornado logit on a
    grid 16 times coarser than the chip; the chip's output is the maximum of
    that map. The chips tile the whole sweep with overlap; the per-gate
    probability is the logistic function of the logit map, taking the maximum
    where chips overlap.
``rotation_couplets``
    A physical comparison: compact maxima of the linear least-squares
    derivative (LLSD) azimuthal shear (:func:`radarx.retrieve.llsd`; Mahalik
    et al. 2019 [3]_) above 0.006 s⁻¹, with the velocity difference across
    each couplet. The threshold is the one that Veillette et al. (2025 [1]_,
    appendix on feature extraction) apply to the azimuthal shear field of
    their non-deep-learning baselines; radarx also attributes it to the NWS
    Tornado Probability Algorithm (Sandmæl et al. 2023 [2]_), not
    checked against that paper.

What is taken from the TorNet paper and what is not. Checked against the
arXiv preprint of Veillette et al. (arXiv:2401.16437v1, January 2024); the
published version (doi below) was not consulted, and its wording and
numbers may differ.

* Chip shape, tilts, variables, range-folded mask, the -3 fill of gates
  without data and range-folded gates, the CoordConv input of the range and
  its inverse, four convolution blocks (hence the factor 16), only the last
  frame of a sample and no azimuthal shear input ("only takes raw radar
  imagery") are as stated in sections 2d and 3b of the preprint. The preprint
  gives the chip as "80 km (240 250-m gates)", but 240 gates of 250 m are 60
  km.
* The preprint says the variables are "normalized to the range [0-1]". The
  released code and weights scale each channel with fixed minimum and maximum
  values to "approximate [-1,1]" (``normalize`` in
  ``tornet/models/keras/cnn_baseline.py`` and ``CHANNEL_MIN_MAX`` in
  ``tornet/data/constants.py`` of github.com/mit-ll/tornet, MIT licence); the
  ONNX model here uses the values stored in the Keras file.
* The preprint trains the network on chips, applies a global maximum over the
  chip and, in its section 4, runs the network on a full scan by removing the
  global maximum pooling (the network is fully convolutional), upsampling the
  likelihood field bilinearly. radarx instead tiles the sweep with 120 x 240
  chips (``stride``), repeats each coarse cell and takes the maximum where
  chips overlap: a radarx choice, not the procedure of the paper.
* The chip stride (a quarter of the chip), batch size, NEXRAD range grid
  (250 m gates from 2125 m, the ``min_range_m`` default of ``tornet`` code),
  the ZDR fill of -8 dB at gates without ZDR (read from the TorNet files, not
  stated in the paper) and the nearest-neighbour resampling of the sweeps to
  720 rays at 0.5° are radarx choices or readings of the released files.

The CNN was trained on the TorNet data set (August 2013 - August 2022
WSR-88D Level II data, 203 133 samples centred on storm cells identified by
SCIT, about 6.8 % of them confirmed tornadoes; Veillette et al. 2025 [1]_,
section 2c), whose velocities were dealiased with the method of Veillette et
al. (2023) [4]_ and whose KDP is the Level III product, upsampled to 0.5° by
nearest neighbour (section 2d). Here the inputs are prepared from the Level II
volume with radarx: velocities are dealiased with
:func:`radarx.retrieve.dealias_velocity` and KDP is estimated with
:func:`radarx.retrieve.estimate_kdp`; both differ slightly from the inputs the
network was trained on.

Model and data licences. The upstream Keras weights (Hugging Face
``tornet-ml/tornado_detector_baseline_v1``, MIT licence per the model card)
are converted to ONNX on first use
(needs the ``onnx`` and ``h5py`` packages once) and cached; inference needs
``onnxruntime`` (``pip install radarx[ml]``). The TorNet data are on Zenodo
(one record per year, listed in the README of github.com/mit-ll/tornet, e.g.
https://doi.org/10.5281/zenodo.12636522 for 2013 and the catalog; Creative
Commons Attribution 4.0 per DataCite) and are not part of radarx. Cite
Veillette et al. (2025) [1]_ when using the model.

References
----------
.. [1] Veillette, M. S., J. M. Kurdzo, P. M. Stepanian, J. Y. N. Cho, T.
   Reis, S. Samsi, J. McDonald, and N. Chisler, 2025: A benchmark dataset
   for tornado detection and prediction using full-resolution polarimetric
   weather radar data. *Artif. Intell. Earth Syst.*, **4** (1),
   https://doi.org/10.1175/AIES-D-24-0006.1
.. [2] Sandmæl, T. N., B. R. Smith, A. E. Reinhart, I. M. Schick, M. C. Ake,
   J. G. Madden, R. B. Steeves, S. S. Williams, K. L. Elmore, and T. C.
   Meyer, 2023: The Tornado Probability Algorithm: A probabilistic machine
   learning tornadic circulation detection algorithm. *Wea. Forecasting*,
   **38** (3), 445-466, https://doi.org/10.1175/WAF-D-22-0123.1
.. [3] Mahalik, M. C., B. R. Smith, K. L. Elmore, D. M. Kingfield, K. L.
   Ortega, and T. M. Smith, 2019: Estimates of gradients in radar moments
   using a linear least squares derivative technique. *Wea. Forecasting*,
   **34** (2), 415-434, https://doi.org/10.1175/WAF-D-18-0095.1
.. [4] Veillette, M. S., J. M. Kurdzo, P. M. Stepanian, J. McDonald, S.
   Samsi, and J. Y. N. Cho, 2023: A deep learning-based velocity dealiasing
   algorithm derived from the WSR-88D open radar product generator. *Artif.
   Intell. Earth Syst.*, **2** (3), https://doi.org/10.1175/AIES-D-22-0084.1

.. autosummary::
   :nosignatures:
   :toctree: generated/

   tornado_probability
   tornet_inputs
   rotation_couplets
"""

from __future__ import annotations

__all__ = ["tornado_probability", "tornet_inputs", "rotation_couplets"]

import numpy as np
import xarray as xr

from .._registry import accessor_method
from . import _onnx_models
from ._products import product_tree

#: Name of the default model, converted on first use.
DEFAULT_MODEL = "tornet-baseline-v1"

#: TorNet chip size (rays, gates): 120 0.5-degree rays by 240 gates of 250 m
#: (Veillette et al. 2025, section 2d; the preprint calls it "80 km" but 240 x
#: 250 m = 60 km).
CHIP = (120, 240)
_AZ_STEP = 0.5
_GATE = 250.0
#: First gate (m): ``min_range_m`` in tornet/data/preprocess.py of the TorNet
#: code (github.com/mit-ll/tornet).
_FIRST_GATE = 2125.0
#: ZDR (dB) of gates without ZDR data in the TorNet files: read from the
#: released files, not stated in Veillette et al. (2025).
_ZDR_FILL = -8.0

_VARIABLES = _onnx_models.TORNET_VARIABLES


# --------------------------------------------------------------------------
# input preparation
# --------------------------------------------------------------------------


def _sweeps(obj):
    """``(name, Dataset)`` of the sweeps of a volume, in order."""
    out = []
    for name, node in obj.children.items():
        if not name.startswith("sweep"):
            continue
        try:
            ds = node.to_dataset(inherit="all_coords")
        except (TypeError, ValueError):  # pragma: no cover - older xarray
            ds = node.to_dataset()
        out.append((name, ds))
    return out


def _fixed_angle(ds):
    if "sweep_fixed_angle" in ds:
        return float(np.asarray(ds["sweep_fixed_angle"]).ravel()[0])
    return float(np.nanmedian(np.asarray(ds["elevation"])))


def _flag_floor(da):
    """
    Largest flag value of a NEXRAD-coded field, or None.

    xradar decodes the NEXRAD Level II codes 0 (below threshold) and 1
    (range folded) to ``add_offset`` and ``add_offset + scale_factor``.
    """
    enc = da.encoding
    if "scale_factor" in enc and "add_offset" in enc:
        if np.issubdtype(np.dtype(enc.get("dtype", "f4")), np.integer):
            return float(enc["add_offset"]) + float(enc["scale_factor"])
    return None


def _clean(da):
    """Field with the NEXRAD flag codes masked; and the range-folded gates."""
    floor = _flag_floor(da)
    if floor is None:
        return da, xr.zeros_like(da, dtype=bool)
    scale = float(da.encoding["scale_factor"])
    folded = np.abs(da - floor) < 0.25 * scale
    return da.where(da > floor + 0.25 * scale), folded


def _nearest_rays(azimuth, target):
    """Index of the ray nearest to every target azimuth (circular)."""
    azimuth = np.mod(np.asarray(azimuth, dtype=np.float64), 360.0)
    order = np.argsort(azimuth)
    az = azimuth[order]
    ext = np.concatenate([az[-1:] - 360.0, az, az[:1] + 360.0])
    idx = np.clip(np.searchsorted(ext, target), 1, len(ext) - 1)
    left = target - ext[idx - 1] < ext[idx] - target
    pick = np.where(left, idx - 1, idx) - 1
    return order[np.mod(pick, len(az))]


def _regrid(da, azimuth, rng):
    """Nearest-neighbour values of a sweep field on the target grid."""
    ray_dim = [d for d in da.dims if d != "range"][0]
    values = da.transpose(ray_dim, "range").values
    rays = _nearest_rays(da["azimuth"].values, azimuth)
    src = np.asarray(da["range"].values, dtype=np.float64)
    step = src[1] - src[0] if src.size > 1 else _GATE
    gates = np.rint((rng - src[0]) / step).astype(np.int64)
    inside = (gates >= 0) & (gates < src.size)
    out = np.full((azimuth.size, rng.size), np.nan, dtype=np.float32)
    out[:, inside] = values[rays][:, gates[inside]]
    return out


def _nyquist(ds, field, nyquist_velocity, angle):
    if isinstance(nyquist_velocity, dict):
        nyquist_velocity = nyquist_velocity.get(round(angle, 1))
    if nyquist_velocity is not None:
        return float(nyquist_velocity)
    if "nyquist_velocity" in ds.variables:
        return float(np.nanmax(np.asarray(ds["nyquist_velocity"])))
    if "nyquist_velocity" in ds[field].attrs:
        return float(ds[field].attrs["nyquist_velocity"])
    return None


def _pick(sweeps, angle, tolerance, need):
    for name, ds in sweeps:
        if abs(_fixed_angle(ds) - angle) <= tolerance and all(
            any(f in ds for f in alts) for alts in need
        ):
            return name, ds
    return None, None


def _first(ds, names):
    for n in names:
        if n in ds:
            return n
    return None


def _doppler_velocity(dop, velocity, dealias, nyquist_velocity, angle):
    """Velocity of a Doppler sweep (flags removed, dealiased) and folded gates."""
    from .dealias import dealias_velocity

    vel, folded = _clean(dop[velocity])
    width_name = _first(dop, ("WRADH", "WRAD", "spectrum_width"))
    width = None
    if width_name:
        width, width_folded = _clean(dop[width_name])
        folded = folded | width_folded
    if dealias:
        nyq = _nyquist(dop, velocity, nyquist_velocity, angle)
        if nyq is None:
            raise ValueError(
                f"no Nyquist velocity at {angle}°: pass nyquist_velocity=, "
                "add a 'nyquist_velocity' coordinate, or dealias=False"
            )
        vel = dealias_velocity(
            dop.assign({velocity: vel}), velocity, nyquist_velocity=nyq
        )
    return vel, width, folded


def _tilt_fields(sweeps, angle, velocity, dealias, nyquist_velocity, kdp, tolerance):
    """The six TorNet variables of one tilt, its range-folded gates, its Doppler cut."""
    from .kdp import estimate_kdp

    _, dop = _pick(sweeps, angle, tolerance, [(velocity,)])
    _, surv = _pick(sweeps, angle, tolerance, [("RHOHV",), ("ZDR",)])
    if dop is None or surv is None:
        raise ValueError(
            f"no sweeps with {velocity!r} and RHOHV/ZDR within "
            f"{tolerance}° of {angle}°"
        )
    vel, width, folded = _doppler_velocity(
        dop, velocity, dealias, nyquist_velocity, angle
    )
    surv = surv.assign(
        {n: _clean(v)[0] for n, v in surv.data_vars.items() if v.ndim == 2}
    )
    dbz_name = _first(surv, ("DBZH", "DBZ", "reflectivity"))
    kdp_name = _first(surv, tuple(k for k in (kdp, "KDP") if k))
    kdp_da = surv[kdp_name] if kdp_name else estimate_kdp(surv)["KDP"]
    zdr = surv.get("ZDR")
    if zdr is not None:
        # TorNet holds -8 dB (the lowest coded ZDR) at gates without ZDR
        zdr = zdr.fillna(_ZDR_FILL)
    fields = {
        "DBZ": surv[dbz_name] if dbz_name else None,
        "VEL": vel,
        "KDP": kdp_da,
        "RHOHV": surv.get("RHOHV"),
        "ZDR": zdr,
        "WIDTH": width,
    }
    return fields, folded, dop


def tornet_inputs(
    volume,
    *,
    elevations=(0.5, 0.9),
    velocity="VRADH",
    dealias=True,
    nyquist_velocity=None,
    kdp=None,
    max_range=None,
    tolerance=0.25,
):
    """
    The inputs of the TorNet CNN prepared from a WSR-88D volume.

    Parameters
    ----------
    volume : xarray.DataTree
        A volume with ``sweep_*`` groups (xradar), e.g. NEXRAD Level II with
        split cuts: each tilt takes the polarimetric variables from the
        first sweep at that elevation holding them and the velocity and
        spectrum width from the first sweep holding velocity.
    elevations : tuple of float, optional
        Fixed angles (degrees) of the two tilts. Default ``(0.5, 0.9)``.
    velocity : str, optional
        Radial velocity field. Default ``"VRADH"``.
    dealias : bool, optional
        Dealias ``velocity`` with :func:`radarx.retrieve.dealias_velocity`
        (default). Pass ``False`` for velocities that are already dealiased.
    nyquist_velocity : float or dict, optional
        Nyquist velocity (m s⁻¹) for dealiasing, or a dict mapping the fixed
        angle (rounded to 0.1°) to it. By default it is read from the sweep
        (``nyquist_velocity`` variable or attribute).
    kdp : str, optional
        Specific differential phase field. By default ``KDP`` if present,
        otherwise it is estimated with :func:`radarx.retrieve.estimate_kdp`.
    max_range : float, optional
        Last range (m) of the output grid. Default: the farthest velocity gate.
    tolerance : float, optional
        Largest difference (degrees) between a sweep's fixed angle and the
        requested elevation. Default 0.25.

    Returns
    -------
    xarray.Dataset
        ``DBZ``, ``VEL``, ``KDP``, ``RHOHV``, ``ZDR``, ``WIDTH`` and
        ``range_folded_mask`` (float32) on ``(azimuth, range, tilt)``: 720
        rays at 0.25°, 0.75°, ..., 359.75° and 250 m gates from 2125 m, the
        grid of the TorNet data. Gates without data are NaN (``ZDR``: -8 dB,
        as in the TorNet files); flag codes of NEXRAD data are removed. Coordinates ``elevation`` (per tilt),
        ``time`` (of the first tilt) and the radar site.

    Raises
    ------
    ValueError
        If a tilt is missing, or dealiasing is requested without a Nyquist
        velocity.

    Notes
    -----
    The layout (two tilts of 0.5° and 0.9°, six variables, range-folded mask,
    ZDR fill) is that of the TorNet samples of Veillette et al. (2025) [1]_
    (sections 2d, 3b; arXiv preprint checked, see :mod:`radarx.retrieve.
    tornado`); dealiasing of the velocities with the method of Veillette et al.
    (2023) [2]_ in TorNet is replaced here by
    :func:`radarx.retrieve.dealias_velocity`, and the Level III KDP of TorNet
    by :func:`radarx.retrieve.estimate_kdp` unless ``KDP`` is present. The
    first gate at 2125 m and the 250 m gate spacing follow the TorNet code
    (``min_range_m`` in ``tornet/data/preprocess.py``); 720 rays at 0.5°
    spacing and the ``tolerance`` of 0.25° for the tilt angles are radarx
    choices. The ZDR fill of -8 dB (the lowest coded ZDR) is how the released
    TorNet files were read, not a statement in the paper.

    References
    ----------
    .. [1] Veillette, M. S., J. M. Kurdzo, P. M. Stepanian, J. Y. N. Cho, T.
       Reis, S. Samsi, J. McDonald, and N. Chisler, 2025: A benchmark dataset
       for tornado detection and prediction using full-resolution
       polarimetric weather radar data. *Artif. Intell. Earth Syst.*, **4**
       (1), https://doi.org/10.1175/AIES-D-24-0006.1
    .. [2] Veillette, M. S., J. M. Kurdzo, P. M. Stepanian, J. McDonald, S.
       Samsi, and J. Y. N. Cho, 2023: A deep learning-based velocity
       dealiasing algorithm derived from the WSR-88D open radar product
       generator. *Artif. Intell. Earth Syst.*, **2** (3),
       https://doi.org/10.1175/AIES-D-22-0084.1
    """
    if not hasattr(volume, "children"):
        raise TypeError("tornet_inputs needs an xarray.DataTree volume")
    sweeps = _sweeps(volume)
    azimuth = np.arange(_AZ_STEP / 2, 360.0, _AZ_STEP)
    tilts, rf_tilts, angles, times, ranges = [], [], [], [], []
    options = (velocity, dealias, nyquist_velocity, kdp, tolerance)
    for angle in elevations:
        fields, rf, dop = _tilt_fields(sweeps, angle, *options)
        stop = max_range or float(dop["range"].values[-1])
        rng = np.arange(_FIRST_GATE, stop + _GATE / 2, _GATE)
        ranges.append(rng)
        tilts.append(
            {
                k: (
                    _regrid(v, azimuth, rng)
                    if v is not None
                    else np.full((azimuth.size, rng.size), np.nan, np.float32)
                )
                for k, v in fields.items()
            }
        )
        rf_tilts.append(np.nan_to_num(_regrid(rf.astype(np.float32), azimuth, rng)))
        angles.append(angle)
        times.append(dop["time"].values.min() if "time" in dop.coords else None)
    n = min(r.size for r in ranges)
    rng = ranges[0][:n]
    data = {
        k: (
            ("azimuth", "range", "tilt"),
            np.stack([t[k][:, :n] for t in tilts], axis=-1),
        )
        for k in _VARIABLES
    }
    data["range_folded_mask"] = (
        ("azimuth", "range", "tilt"),
        np.stack([r[:, :n] for r in rf_tilts], axis=-1),
    )
    coords = {
        "azimuth": ("azimuth", azimuth, {"units": "degrees", "long_name": "azimuth"}),
        "range": ("range", rng, {"units": "meters", "long_name": "range"}),
        "elevation": ("tilt", np.asarray(angles), {"units": "degrees"}),
    }
    root = volume.to_dataset(inherit=False)
    for c in ("latitude", "longitude", "altitude"):
        if c in root.variables:
            coords[c] = root[c]
    if times[0] is not None:
        coords["time"] = times[0]
    out = xr.Dataset(data, coords=coords)
    for k in _VARIABLES:
        out[k].attrs = {"long_name": f"TorNet input {k}"}
    out["range_folded_mask"].attrs = {"long_name": "range-folded gates (1)"}
    return out


# --------------------------------------------------------------------------
# CNN inference
# --------------------------------------------------------------------------


def _coordinates(rng_chip):
    """
    TorNet's range channels (range and its inverse, in units of 1e5 m).

    TorNet labels each gate one gate (250 m) farther than the gate centre of
    the Level II data and computes the channels from the chip's range limits
    (outer edges of the labelled gates) as ``linspace(lower + 250, upper -
    250)`` with the scale 1e-5 (``compute_coordinates`` in
    ``tornet/data/preprocess.py`` of github.com/mit-ll/tornet); the same is
    done here. The paper describes the channels as a scaled radial coordinate
    and its inverse (CoordConv, Veillette et al. 2025, section 3b).
    """
    scale = 1e-5
    lower = (rng_chip[0] + _GATE - _GATE / 2 + _GATE) * scale
    upper = (rng_chip[-1] + _GATE + _GATE / 2 - _GATE) * scale
    r = np.linspace(lower, upper, rng_chip.size)
    r = np.maximum(r, _FIRST_GATE * scale)
    return np.stack([r, 1.0 / r], axis=-1).astype(np.float32)


def _starts(n, size, step, wrap):
    if wrap:
        return np.arange(0, n, step)
    if n <= size:
        return np.array([0])
    s = list(range(0, n - size + 1, step))
    if s[-1] != n - size:
        s.append(n - size)
    return np.array(s)


def _run_chips(model, inputs, chip, stride, batch_size):
    """Logits and logit maps of all chips tiling the sweep."""
    n_az, n_rng = inputs["DBZ"].shape[:2]
    n_rng_chip = min(chip[1], n_rng)
    az_starts = _starts(n_az, chip[0], stride[0], wrap=True)
    rng_starts = _starts(n_rng, n_rng_chip, stride[1], wrap=False)
    rng = inputs["range"]
    windows = [(a, r) for a in az_starts for r in rng_starts]
    logits, maps = [], []
    for b in range(0, len(windows), batch_size):
        batch = windows[b : b + batch_size]
        feed = {}
        for k in list(_VARIABLES) + ["range_folded_mask"]:
            arr = inputs[k]
            feed[k] = np.stack(
                [
                    np.take(arr, np.arange(a, a + chip[0]), axis=0, mode="wrap")[
                        :, r : r + n_rng_chip
                    ]
                    for a, r in batch
                ]
            ).astype(np.float32)
        coords = np.stack(
            [
                np.broadcast_to(
                    _coordinates(rng[r : r + n_rng_chip]), (chip[0], n_rng_chip, 2)
                )
                for _, r in batch
            ]
        )
        feed["coordinates"] = np.ascontiguousarray(coords, dtype=np.float32)
        out = model.run(feed)
        logits.append(np.asarray(out["logit"]).reshape(-1))
        maps.append(np.asarray(out["heatmap"]))
    return windows, np.concatenate(logits), np.concatenate(maps), n_rng_chip


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def tornado_probability(
    volume,
    model=None,
    *,
    stride=(30, 60),
    batch_size=16,
    providers=None,
    **kwargs,
):
    """
    Tornado probability from the TorNet CNN baseline (ONNX).

    Parameters
    ----------
    volume : xarray.DataTree or xarray.Dataset
        A WSR-88D volume (see :func:`tornet_inputs`), or a Dataset of
        prepared inputs as returned by :func:`tornet_inputs`.
    model : str or path or radarx.ml.Model, optional
        Model to run. Default ``"tornet-baseline-v1"``, the published TorNet
        weights converted to ONNX on first use. A path to an ONNX file with
        the same inputs and outputs, a name in the radarx model registry or
        a loaded model can be given instead.
    stride : tuple of int, optional
        Step (rays, gates) between the 120 x 240 chips that tile the sweep.
        Default ``(30, 60)``, a quarter of the chip.
    batch_size : int, optional
        Chips per inference call. Default 16.
    providers : list of str, optional
        ONNX Runtime execution providers. Default: CPU.
    **kwargs
        Passed to :func:`tornet_inputs` (``elevations``, ``velocity``,
        ``dealias``, ``nyquist_velocity``, ``kdp``, ``max_range``).

    Returns
    -------
    xarray.Dataset
        ``tornado_probability`` (0-1, float32) on the ``(azimuth, range)``
        grid of :func:`tornet_inputs`: for every gate the logistic function of
        the network's logit map (16 x 16 gates per cell) of the chips that
        cover it, the maximum over overlapping chips. ``chip_probability``
        on ``chip``: the TorNet output (logistic of the maximum logit) of
        every chip, with the azimuth and range of the chip centre. Attributes
        ``ml_model``, ``ml_model_version``, ``ml_model_licence`` and
        ``ml_model_citation``.

    Notes
    -----
    The network was trained on samples centred on storm cells, about 6.8 %
    of them confirmed tornadoes (Veillette et al. 2025 [1]_, section 2c), so
    its output is a likelihood on the TorNet sample distribution rather than
    a calibrated frequency for arbitrary gates. The paper (section 3c.4)
    shows that the raw likelihood is over-confident for mid-range values
    (0.2-0.7) and under-confident above 0.8, and fits an isotonic calibration
    (Brier score 0.0423 before and 0.0404 after on the test set); that
    calibration is not applied here.

    *Skill of the network (TorNet test set, from the paper, not measured
    here).* Table 2 (i) of the arXiv preprint ("confirmed tornadoes versus all
    nulls", test set of 31 467 samples, mean of five random seeds) gives for
    the CNN an area under the ROC curve of 0.8742 and a maximum critical
    success index of 0.3380 (the maxima taken over all thresholds); for
    comparison the operational TVS gives 0.6308 and 0.2002. Those numbers
    belong to chips of the TorNet test set, not to the full-scan output of
    this function, and the published version of the paper may differ. The
    model file used here is the released one (Hugging Face
    ``tornet-ml/tornado_detector_baseline_v1``, MIT licence), which is not
    necessarily one of the five seeds of the table.

    References
    ----------
    .. [1] Veillette, M. S., J. M. Kurdzo, P. M. Stepanian, J. Y. N. Cho, T.
       Reis, S. Samsi, J. McDonald, and N. Chisler, 2025: A benchmark dataset
       for tornado detection and prediction using full-resolution
       polarimetric weather radar data. *Artif. Intell. Earth Syst.*, **4**
       (1), https://doi.org/10.1175/AIES-D-24-0006.1

    Examples
    --------
    >>> out = radarx.retrieve.tornado_probability(dtree)  # doctest: +SKIP
    >>> out.tornado_probability.plot()  # doctest: +SKIP
    """
    if isinstance(volume, xr.Dataset):
        if kwargs:
            raise TypeError(f"unexpected arguments for prepared inputs: {kwargs}")
        inputs_ds = volume
    else:
        inputs_ds = tornet_inputs(volume, **kwargs)
    if min(stride) < 1:
        raise ValueError("stride must be positive")
    net = _onnx_models.load(model, DEFAULT_MODEL, providers)
    arrays = {k: inputs_ds[k].values for k in list(_VARIABLES) + ["range_folded_mask"]}
    arrays["range"] = inputs_ds["range"].values
    windows, logits, maps, n_rng_chip = _run_chips(
        net, arrays, CHIP, stride, batch_size
    )
    n_az, n_rng = arrays["DBZ"].shape[:2]
    best = np.full((n_az, n_rng), -np.inf)
    for (a, r), hm in zip(windows, maps):
        cell_az = -(-CHIP[0] // hm.shape[0])
        cell_rng = -(-n_rng_chip // hm.shape[1])
        up = np.repeat(np.repeat(hm, cell_az, 0), cell_rng, 1)[: CHIP[0], :n_rng_chip]
        rows = np.mod(np.arange(a, a + CHIP[0]), n_az)
        best[rows, r : r + n_rng_chip] = np.maximum(best[rows, r : r + n_rng_chip], up)
    prob = np.where(np.isfinite(best), _sigmoid(best), np.nan).astype(np.float32)
    attrs = _onnx_models.model_attrs(net, DEFAULT_MODEL)
    azimuth = inputs_ds["azimuth"].values
    rng = inputs_ds["range"].values
    chip_az = np.array(
        [np.mod(azimuth[a] + (CHIP[0] / 2 - 0.5) * _AZ_STEP, 360.0) for a, _ in windows]
    )
    chip_rng = np.array([rng[r : r + n_rng_chip].mean() for _, r in windows])
    out = xr.Dataset(
        {
            "tornado_probability": (
                ("azimuth", "range"),
                prob,
                {
                    "long_name": "tornado probability (TorNet CNN)",
                    "units": "1",
                    "comment": "logistic of the CNN logit map, maximum over chips",
                    **attrs,
                },
            ),
            "chip_probability": (
                "chip",
                _sigmoid(logits).astype(np.float32),
                {"long_name": "tornado probability of the chip", "units": "1"},
            ),
        },
        coords={
            "chip_azimuth": ("chip", chip_az, {"units": "degrees"}),
            "chip_range": ("chip", chip_rng, {"units": "meters"}),
            **{
                k: v
                for k, v in inputs_ds.coords.items()
                if "tilt" not in v.dims or k == "elevation"
            },
        },
        attrs={**attrs, "chip_size": list(CHIP), "stride": list(stride)},
    )
    return out


# --------------------------------------------------------------------------
# physical comparison: LLSD rotation couplets
# --------------------------------------------------------------------------


def _label_regions(candidate, az):
    """8-connected regions of ``candidate``; regions touching across north merge."""
    from scipy import ndimage

    labels, n = ndimage.label(candidate, structure=np.ones((3, 3)))
    order = np.argsort(np.mod(az, 360.0))
    first, last = order[0], order[-1]
    step = np.median(np.diff(az[order]))
    if n and abs(np.mod(az[first] - az[last], 360.0)) < 2 * step:
        for a, b in zip(labels[first], labels[last]):
            if a and b and a != b:
                labels[labels == b] = a
    return labels


def _delta_v(vel, az, rng, peak, diameter):
    """Largest velocity difference within ``diameter / 2`` of the peak gate."""
    ia, ir = peak
    r0, a0 = rng[ir], np.radians(az[ia])
    dtheta = np.radians(np.median(np.diff(np.sort(np.mod(az, 360.0)))))
    dr = rng[1] - rng[0] if rng.size > 1 else _GATE
    half_rays = int(np.ceil(diameter / 2 / max(r0 * dtheta, 1.0)))
    half_gates = int(np.ceil(diameter / 2 / dr))
    rays = np.mod(np.arange(ia - half_rays, ia + half_rays + 1), az.size)
    g0, g1 = max(ir - half_gates, 0), min(ir + half_gates + 1, rng.size)
    rr = rng[g0:g1][None, :]
    aa = np.radians(az[rays])[:, None]
    dist = np.hypot(
        rr * np.sin(aa) - r0 * np.sin(a0), rr * np.cos(aa) - r0 * np.cos(a0)
    )
    v = np.where(dist <= diameter / 2, vel[rays, g0:g1], np.nan)
    return np.nanmax(v) - np.nanmin(v) if np.isfinite(v).any() else np.nan


def _couplets_sweep(ds, field, threshold, window, min_area, diameter, min_reflectivity):
    from .shear import llsd

    da = ds[field]
    ray_dim = [d for d in da.dims if d != "range"][0]
    vel = da.transpose(ray_dim, "range").values.astype(np.float64)
    shear = (
        llsd(ds, field, window)["azimuthal_shear"]
        .transpose(ray_dim, "range")
        .values.astype(np.float64)
    )
    az = np.asarray(ds["azimuth"].values, dtype=np.float64)
    rng = np.asarray(ds["range"].values, dtype=np.float64)
    candidate = np.nan_to_num(shear) >= threshold
    if min_reflectivity is not None:
        dbz_name = _first(ds, ("DBZH", "DBZ", "reflectivity"))
        if dbz_name is not None:
            dbz = ds[dbz_name].transpose(ray_dim, "range").values
            candidate &= np.nan_to_num(dbz, nan=-99.0) >= min_reflectivity
    labels = _label_regions(candidate, az)
    dr = rng[1] - rng[0] if rng.size > 1 else _GATE
    dtheta = np.radians(np.median(np.diff(np.sort(np.mod(az, 360.0)))))
    gate_area = (rng * dtheta * dr)[None, :] * np.ones_like(shear)
    ids = np.unique(labels[labels > 0])
    rows = []
    x = ds["x"].transpose(ray_dim, "range").values if "x" in ds.coords else None
    y = ds["y"].transpose(ray_dim, "range").values if "y" in ds.coords else None
    for i in ids:
        sel = labels == i
        area = gate_area[sel].sum()
        if area < min_area:
            continue
        flat = np.where(sel, shear, -np.inf)
        ia, ir = np.unravel_index(np.argmax(flat), flat.shape)
        dv = _delta_v(vel, az, rng, (ia, ir), diameter)
        rows.append(
            (
                az[ia],
                rng[ir],
                shear[ia, ir],
                dv,
                area,
                x[ia, ir] if x is not None else np.nan,
                y[ia, ir] if y is not None else np.nan,
            )
        )
    rows.sort(key=lambda t: -t[2])
    cols = list(zip(*rows)) if rows else [[]] * 7
    arr = [np.asarray(c, dtype=np.float64) for c in cols]
    method = {
        "method": "LLSD azimuthal shear maxima",
        "source_field": field,
        "shear_threshold": threshold,
        "window_range_m": float(window[0]),
        "window_azimuth_m": float(window[1]),
        "diameter_m": float(diameter),
    }
    out = xr.Dataset(
        {
            "azimuthal_shear": (
                "couplet",
                arr[2],
                {"long_name": "peak azimuthal shear", "units": "s-1"},
            ),
            "delta_v": (
                "couplet",
                arr[3],
                {
                    "long_name": "velocity difference across the couplet",
                    "units": "m s-1",
                },
            ),
            "rotational_velocity": (
                "couplet",
                arr[3] / 2.0,
                {"long_name": "rotational velocity (delta_v / 2)", "units": "m s-1"},
            ),
            "area": (
                "couplet",
                arr[4] / 1e6,
                {"long_name": "area above the shear threshold", "units": "km2"},
            ),
        },
        coords={
            "azimuth": ("couplet", arr[0], {"units": "degrees"}),
            "range": ("couplet", arr[1], {"units": "meters"}),
            "x": ("couplet", arr[5], {"units": "meters"}),
            "y": ("couplet", arr[6], {"units": "meters"}),
        },
        attrs=method,
    )
    for c in ("time", "latitude", "longitude", "altitude", "sweep_fixed_angle"):
        if c in ds.variables and ds[c].ndim == 0:
            out.coords[c] = ds[c]
    return out


def rotation_couplets(
    obj,
    field="VRADH",
    *,
    threshold=0.006,
    window=(750.0, 2500.0),
    min_area=0.5e6,
    diameter=5000.0,
    min_reflectivity=None,
):
    """
    Rotation couplets: compact maxima of LLSD azimuthal shear.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataTree
        A PPI sweep with dealiased radial velocity, or a volume whose sweeps
        holding ``field`` are processed one by one.
    field : str, optional
        Dealiased radial velocity (m s⁻¹). Default ``"VRADH"``.
    threshold : float, optional
        Azimuthal shear (s⁻¹) above which gates belong to a candidate
        circulation. Default 0.006, the threshold that Veillette et al. (2025)
        [3]_ apply to the azimuthal shear field of their feature-based
        baselines (appendix); the same value is credited here to the NWS
        Tornado Probability Algorithm (Sandmæl et al. 2023 [1]_) on the 0.5°
        tilt, not checked against the paper.
    window : tuple of float, optional
        LLSD window ``(range_m, azimuth_m)``, see
        :func:`radarx.retrieve.llsd` and Mahalik et al. (2019) [2]_.
        Default ``(750, 2500)``, a radarx choice not checked against
        Mahalik et al. or Sandmæl et al.
    min_area : float, optional
        Smallest area (m²) of a region above the threshold. Default 0.5 km²,
        a radarx choice without a published source.
    diameter : float, optional
        Diameter (m) of the circle around the shear peak in which the
        largest velocity difference ``delta_v`` is measured. Default 5 km, a
        radarx choice without a published source.
    min_reflectivity : float, optional
        If given, only gates with at least this reflectivity (dBZ) count.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        One entry per couplet along ``couplet``, strongest shear first:
        ``azimuthal_shear`` (peak), ``delta_v``, ``rotational_velocity``
        (``delta_v / 2``) and ``area``, with the ``azimuth``, ``range`` (and
        ``x``, ``y`` if the sweep is georeferenced) of the peak. A DataTree
        input gives a DataTree with one such node per sweep that has
        ``field``.

    Notes
    -----
    A couplet is an 8-connected region of gates with LLSD azimuthal shear at
    or above ``threshold`` (and optionally ``min_reflectivity``); the peak
    shear, the velocity difference within ``diameter`` of the peak and the
    area are reported, and ``rotational_velocity`` is ``delta_v / 2`` (the
    usual definition of rotational velocity, not taken from the cited
    papers). This is a simplified physical diagnostic, not the Tornado
    Probability Algorithm of Sandmæl et al. (2023) [1]_, which is a machine-
    learning model on LLSD-derived features; the region extraction, minimum
    area, ``delta_v`` measurement and all sizes other than the 0.006 s⁻¹
    threshold are radarx choices.

    References
    ----------
    .. [1] Sandmæl, T. N., B. R. Smith, A. E. Reinhart, I. M. Schick, M. C.
       Ake, J. G. Madden, R. B. Steeves, S. S. Williams, K. L. Elmore, and
       T. C. Meyer, 2023: The Tornado Probability Algorithm: A probabilistic
       machine learning tornadic circulation detection algorithm. *Wea.
       Forecasting*, **38** (3), 445-466,
       https://doi.org/10.1175/WAF-D-22-0123.1
    .. [2] Mahalik, M. C., B. R. Smith, K. L. Elmore, D. M. Kingfield, K. L.
       Ortega, and T. M. Smith, 2019: Estimates of gradients in radar moments
       using a linear least squares derivative technique. *Wea. Forecasting*,
       **34** (2), 415-434, https://doi.org/10.1175/WAF-D-18-0095.1
    .. [3] Veillette, M. S., J. M. Kurdzo, P. M. Stepanian, J. Y. N. Cho, T.
       Reis, S. Samsi, J. McDonald, and N. Chisler, 2025: A benchmark dataset
       for tornado detection and prediction using full-resolution
       polarimetric weather radar data. *Artif. Intell. Earth Syst.*, **4**
       (1), https://doi.org/10.1175/AIES-D-24-0006.1
    """
    options = (field, threshold, window, min_area, diameter, min_reflectivity)
    if isinstance(obj, xr.Dataset):
        if field not in obj:
            raise KeyError(f"{field!r} is not in the sweep")
        return _couplets_sweep(obj, *options)
    if hasattr(obj, "children"):
        nodes = {
            name: _couplets_sweep(ds, *options)
            for name, ds in _sweeps(obj)
            if field in ds.data_vars
        }
        if not nodes:
            raise ValueError(f"No sweep contains {field!r}.")
        return product_tree(obj, nodes)
    raise TypeError("rotation_couplets needs an xarray.Dataset or xarray.DataTree")


@accessor_method("datatree", name="tornado_probability")
def _tornado_probability_accessor(self, model=None, **kwargs):
    return tornado_probability(self.xarray_obj, model, **kwargs)


@accessor_method("dataset", "datatree", name="rotation_couplets")
def _rotation_couplets_accessor(self, field="VRADH", **kwargs):
    return rotation_couplets(self.xarray_obj, field, **kwargs)


_tornado_probability_accessor.__doc__ = tornado_probability.__doc__
_rotation_couplets_accessor.__doc__ = rotation_couplets.__doc__
