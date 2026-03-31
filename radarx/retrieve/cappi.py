#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
CAPPI Retrieval
===============

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

__all__ = ["create_cappi"]

__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

_CARTESIAN_IDW_METHOD = "cartesian_idw"
_POLAR_VERTICAL_INTERPOLATION_METHOD = "polar_vertical_interpolation"
_HEIGHT_WINDOW_COMPOSITE_METHOD = "height_window_composite"
_DEFAULT_VERTICAL_TOLERANCES = {
    _CARTESIAN_IDW_METHOD: 3000.0,
    _POLAR_VERTICAL_INTERPOLATION_METHOD: None,
    _HEIGHT_WINDOW_COMPOSITE_METHOD: 500.0,
}
_COMMON_VELOCITY_FIELDS = (
    "velocity",
    "mean_doppler_velocity",
    "VRADH",
    "VEL",
)
_COMMON_REFLECTIVITY_FIELDS = (
    "reflectivity",
    "reflectivity_horizontal",
    "DBZH",
    "DBZ",
    "REF",
)
_METHOD_ALIASES = {
    "cartesian": _CARTESIAN_IDW_METHOD,
    _CARTESIAN_IDW_METHOD: _CARTESIAN_IDW_METHOD,
    "polar": _POLAR_VERTICAL_INTERPOLATION_METHOD,
    _POLAR_VERTICAL_INTERPOLATION_METHOD: _POLAR_VERTICAL_INTERPOLATION_METHOD,
    "pseudo_cappi": _HEIGHT_WINDOW_COMPOSITE_METHOD,
    _HEIGHT_WINDOW_COMPOSITE_METHOD: _HEIGHT_WINDOW_COMPOSITE_METHOD,
}


def _iter_sweeps(radar, sweeps=None):
    if sweeps is not None:
        return list(sweeps)
    return [name for name in radar.children if "sweep" in name]


def _node_to_dataset(node):
    try:
        return node.to_dataset(inherit="all_coords")
    except TypeError:  # pragma: no cover
        return node.to_dataset()


def _default_cappi_fields(ds):
    """Pick likely radar data variables, excluding metadata-like variables."""
    skip = {
        "x",
        "y",
        "z",
        "time",
        "range",
        "azimuth",
        "elevation",
        "latitude",
        "longitude",
        "altitude",
        "crs_wkt",
        "sweep_number",
        "sweep_fixed_angle",
        "sweep_mode",
        "follow_mode",
        "prt_mode",
    }
    fields = []
    for name, da in ds.data_vars.items():
        if name in skip or da.ndim != 2:
            continue
        if "range" in da.dims and any(
            dim in da.dims for dim in ("azimuth", "elevation", "time")
        ):
            fields.append(name)
    return fields


def _extract_scalar(ds, name):
    if name not in ds:
        return None

    values = np.asarray(ds[name].values)
    if values.size == 0:
        return None
    return values.reshape(-1)[0]


def _decode_if_bytes(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode()
    return value


def _get_reference_time_axis(ds, axis_length):
    if "time" not in ds:
        return np.arange(axis_length, dtype=int)

    values = np.asarray(ds["time"].values)
    if values.size == axis_length:
        return values.reshape(axis_length)

    scalar = values.reshape(-1)[0]
    return np.repeat(scalar, axis_length)


def _get_reference_metadata_value(ds, name, default=None):
    value = _extract_scalar(ds, name)
    if value is None:
        return default
    return _decode_if_bytes(value)


def _infer_field_name(ds, candidates):
    for candidate in candidates:
        if candidate in ds.data_vars:
            return candidate
    return None


def _collect_volume_points(
    radar,
    fields=None,
    sweeps=None,
    min_z=None,
    max_z=None,
):
    """
    Collect flattened gate coordinates and field values from georeferenced sweeps.
    """
    volume_xyz = []
    field_store = {}

    for sw in _iter_sweeps(radar, sweeps=sweeps):
        ds = _node_to_dataset(radar[sw])

        for coord in ("x", "y", "z"):
            if coord not in ds:
                raise ValueError(
                    f"{sw} is missing '{coord}'. Run radar.xradar.georeference() first."
                )

        if fields is None:
            these_fields = _default_cappi_fields(ds)
        else:
            these_fields = [field for field in fields if field in ds.data_vars]

        x = np.asarray(ds["x"].values).ravel()
        y = np.asarray(ds["y"].values).ravel()
        z = np.asarray(ds["z"].values).ravel()

        good_xyz = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if min_z is not None:
            good_xyz &= z >= min_z
        if max_z is not None:
            good_xyz &= z <= max_z

        if not np.any(good_xyz):
            continue

        sweep_xyz = np.column_stack([x[good_xyz], y[good_xyz], z[good_xyz]])
        volume_xyz.append(sweep_xyz)

        for field in these_fields:
            values = np.asarray(ds[field].values, dtype=float).ravel()[good_xyz]
            good_values = np.isfinite(values)
            if not np.any(good_values):
                continue

            field_store.setdefault(field, {"xyz": [], "values": []})
            field_store[field]["xyz"].append(sweep_xyz[good_values])
            field_store[field]["values"].append(values[good_values])

    if not volume_xyz:
        raise ValueError("No valid georeferenced gates found in requested volume.")

    packed_fields = {}
    for field, payload in field_store.items():
        if payload["xyz"]:
            packed_fields[field] = (
                np.vstack(payload["xyz"]),
                np.concatenate(payload["values"]),
            )

    if not packed_fields:
        raise ValueError("No valid radar fields were found for CAPPI retrieval.")

    return np.vstack(volume_xyz), packed_fields


def _build_target_grid(x, y):
    """
    Build a regular target grid from 1D x/y vectors.
    """
    if x is None or y is None:
        raise ValueError("Both x and y must be provided together.")

    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _make_grid_from_extent(
    xyz,
    x_res=1000.0,
    y_res=1000.0,
    padding=0.0,
):
    """
    Build default x/y vectors from volume extent.
    """
    xmin, ymin = np.nanmin(xyz[:, 0]), np.nanmin(xyz[:, 1])
    xmax, ymax = np.nanmax(xyz[:, 0]), np.nanmax(xyz[:, 1])

    xmin -= padding
    xmax += padding
    ymin -= padding
    ymax += padding

    x = np.arange(xmin, xmax + x_res, x_res, dtype=float)
    y = np.arange(ymin, ymax + y_res, y_res, dtype=float)
    return x, y


def _idw_interpolate_to_cappi(
    xyz_src,
    values,
    x_tgt,
    y_tgt,
    z_tgt,
    k=16,
    search_radius=5000.0,
    power=2.0,
    vertical_scale=3.0,
    min_neighbors=3,
):
    """
    Interpolate one field to a constant-z plane using 3D anisotropic IDW.
    """
    x_grid, y_grid = np.meshgrid(x_tgt, y_tgt)
    z_grid = np.full_like(x_grid, float(z_tgt), dtype=float)
    target_xyz = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])

    if values.size == 0:
        return np.full(x_grid.shape, np.nan, dtype=float)

    src_scaled = xyz_src.astype(float, copy=True)
    tgt_scaled = target_xyz.astype(float, copy=True)
    src_scaled[:, 2] *= vertical_scale
    tgt_scaled[:, 2] *= vertical_scale

    tree = cKDTree(src_scaled)
    dists, idxs = tree.query(
        tgt_scaled,
        k=k,
        distance_upper_bound=search_radius,
        workers=-1,
    )

    if k == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    out = np.full(target_xyz.shape[0], np.nan, dtype=float)
    source_count = len(values)

    for i in range(target_xyz.shape[0]):
        di = dists[i]
        ii = idxs[i]
        valid = np.isfinite(di) & (ii < source_count)
        if np.count_nonzero(valid) < min_neighbors:
            continue

        di = di[valid]
        ii = ii[valid]

        if np.any(di == 0.0):
            out[i] = values[ii[di == 0.0][0]]
            continue

        weights = 1.0 / np.power(di, power)
        out[i] = np.sum(weights * values[ii]) / np.sum(weights)

    return out.reshape(x_grid.shape)


def _cappi_reference_metadata(radar, sweeps=None):
    for sw in _iter_sweeps(radar, sweeps=sweeps):
        ds = _node_to_dataset(radar[sw])
        coords = {}
        for name in ("time", "latitude", "longitude", "altitude"):
            value = _extract_scalar(ds, name)
            if value is not None:
                coords[name] = value

        attrs = {
            key: value
            for key, value in ds.attrs.items()
            if key in {"instrument_name", "radar_name", "site_name"}
        }
        return coords, attrs

    return {}, {}


def _apply_velocity_texture_gate_filter(
    ds,
    velocity_field=None,
    reflectivity_field=None,
    texture_window=50,
    texture_threshold=None,
    reflectivity_min=-10.0,
    reflectivity_max=75.0,
):
    """
    Apply simple gate-level quality control using Doppler-velocity texture and
    reflectivity limits.
    """
    filtered = ds.copy()

    if velocity_field is not None and velocity_field in filtered:
        velocity_values = np.asarray(filtered[velocity_field].values, dtype=float)
        finite_velocity = velocity_values[np.isfinite(velocity_values)]
        if finite_velocity.size < 2:
            velocity_texture = None
        else:
            velocity_texture = (
                filtered[velocity_field]
                .rolling(
                    range=texture_window,
                    min_periods=1,
                    center=True,
                )
                .std()
            )

        if velocity_texture is not None and texture_threshold is None:
            texture_values = np.asarray(velocity_texture.values, dtype=float)
            finite_texture = texture_values[np.isfinite(texture_values)]

            if finite_texture.size == 0:
                texture_threshold = np.inf
            elif finite_texture.size == 1:
                texture_threshold = abs(float(finite_texture[0]))
            else:
                texture_threshold = abs(
                    float(np.nanvar(finite_texture)) + float(np.nanstd(finite_texture))
                )

        if velocity_texture is not None:
            filtered = filtered.where(velocity_texture < float(texture_threshold))

    if reflectivity_field is not None and reflectivity_field in filtered:
        filtered = filtered.where(
            (filtered[reflectivity_field] >= reflectivity_min)
            & (filtered[reflectivity_field] <= reflectivity_max)
        )

    return filtered


def _normalize_cappi_method(method):
    try:
        return _METHOD_ALIASES[method]
    except KeyError as exc:
        valid = ", ".join(sorted(_METHOD_ALIASES))
        raise ValueError(
            f"Unsupported CAPPI method '{method}'. Choose from: {valid}."
        ) from exc


def _resolve_vertical_tolerance(method, vertical_tolerance):
    if vertical_tolerance is None:
        return _DEFAULT_VERTICAL_TOLERANCES[method]
    return float(vertical_tolerance)


def _create_cappi_cartesian_idw(
    radar,
    height,
    fields=None,
    sweeps=None,
    x=None,
    y=None,
    x_res=1000.0,
    y_res=1000.0,
    padding=0.0,
    vertical_window=3000.0,
    k=16,
    search_radius=5000.0,
    power=2.0,
    vertical_scale=3.0,
    min_neighbors=3,
):
    """
    Create CAPPI on a Cartesian ``x``/``y`` plane using 3D IDW.

    Parameters
    ----------
    radar : xarray.DataTree
        Georeferenced radar volume. Each sweep must already contain ``x``, ``y``,
        and ``z`` coordinates.
    height : float
        Requested CAPPI altitude in meters.
    fields : list[str] or None, optional
        Data variables to interpolate. If omitted, likely radar fields are
        auto-selected.
    sweeps : list[str] or None, optional
        Sweep names to use. If omitted, all sweep groups are used.
    x, y : 1D array-like or None, optional
        Target horizontal grid coordinates. If omitted, they are built from the
        source extent using ``x_res`` and ``y_res``.
    x_res, y_res : float, optional
        Target grid spacing in meters when ``x`` and ``y`` are not supplied.
    padding : float, optional
        Extra padding around the source extent in meters.
    vertical_window : float, optional
        Only gates within ``height +/- vertical_window`` are considered.
    k : int, optional
        Number of nearest neighbors for IDW.
    search_radius : float, optional
        Maximum search radius in meters in the anisotropically scaled space.
    power : float, optional
        IDW power parameter.
    vertical_scale : float, optional
        Vertical distance multiplier. Values greater than 1 penalize vertical
        mismatch more strongly than horizontal mismatch.
    min_neighbors : int, optional
        Minimum valid neighbors required to produce a grid value.

    Returns
    -------
    xarray.Dataset
        CAPPI dataset on ``y``/``x`` coordinates with scalar ``z=height``.
    """
    if height is None:
        raise ValueError("height must be provided in meters.")

    volume_xyz, field_store = _collect_volume_points(
        radar,
        fields=fields,
        sweeps=sweeps,
        min_z=height - vertical_window,
        max_z=height + vertical_window,
    )

    if x is None or y is None:
        x, y = _make_grid_from_extent(
            volume_xyz,
            x_res=x_res,
            y_res=y_res,
            padding=padding,
        )
    else:
        x, y = _build_target_grid(x, y)

    data_vars = {}
    for field, (xyz_src, values) in field_store.items():
        cappi = _idw_interpolate_to_cappi(
            xyz_src=xyz_src,
            values=values,
            x_tgt=x,
            y_tgt=y,
            z_tgt=height,
            k=k,
            search_radius=search_radius,
            power=power,
            vertical_scale=vertical_scale,
            min_neighbors=min_neighbors,
        )
        attrs = {}
        ref_sweep = sweeps[0] if sweeps else _iter_sweeps(radar)[0]
        ref_ds = _node_to_dataset(radar[ref_sweep])
        attrs = ref_ds[field].attrs if field in ref_ds else {}
        data_vars[field] = (("y", "x"), cappi, attrs)

    ref_coords, ref_attrs = _cappi_reference_metadata(radar, sweeps=sweeps)

    ds_cappi = xr.Dataset(
        data_vars=data_vars,
        coords={
            "x": ("x", x),
            "y": ("y", y),
            "z": height,
            **ref_coords,
        },
        attrs={
            "product": "CAPPI",
            "height": float(height),
            "method": _CARTESIAN_IDW_METHOD,
            "interpolation": "3D anisotropic IDW",
            "vertical_window_m": float(vertical_window),
            "search_radius_m": float(search_radius),
            "idw_power": float(power),
            "vertical_scale": float(vertical_scale),
            "min_neighbors": int(min_neighbors),
            **ref_attrs,
        },
    )

    return ds_cappi


def _interp_to_reference(ds, field, az_ref, r_ref):
    """Interpolate a sweep field onto a reference azimuth/range grid."""
    az_src = np.asarray(ds.azimuth.values, dtype=float)
    data = np.asanyarray(ds[field].values)

    if np.ma.isMaskedArray(data):
        data = data.filled(np.nan)
    data = np.asarray(data, dtype=float)

    if data.shape[1] != len(r_ref):
        raise ValueError(
            f"Field '{field}' range dimension does not match reference grid."
        )

    order = np.argsort(az_src)
    az_sorted = az_src[order]
    data_sorted = data[order, :]

    if len(az_sorted) == len(az_ref) and np.allclose(az_sorted, az_ref):
        return data_sorted

    out = np.full((len(az_ref), data_sorted.shape[1]), np.nan, dtype=float)

    for j in range(data_sorted.shape[1]):
        column = data_sorted[:, j]
        valid = np.isfinite(column)

        if np.count_nonzero(valid) == 0:
            continue

        az_valid = az_sorted[valid]
        col_valid = column[valid]

        if az_valid.size == 1:
            out[:, j] = col_valid[0]
            continue

        az_periodic = np.concatenate([az_valid - 360.0, az_valid, az_valid + 360.0])
        col_periodic = np.concatenate([col_valid, col_valid, col_valid])

        sort_idx = np.argsort(az_periodic)
        az_periodic = az_periodic[sort_idx]
        col_periodic = col_periodic[sort_idx]

        az_periodic, unique_idx = np.unique(az_periodic, return_index=True)
        col_periodic = col_periodic[unique_idx]

        out[:, j] = np.interp(az_ref, az_periodic, col_periodic)

    return out


def _interp_vertical_column(z_col, v_col, height):
    """Linearly interpolate one vertical column to a target height."""
    z_col = np.asarray(z_col, dtype=float)
    v_col = np.asarray(v_col, dtype=float)

    valid = np.isfinite(z_col) & np.isfinite(v_col)
    if np.count_nonzero(valid) == 0:
        return np.nan, True

    z_valid = z_col[valid]
    v_valid = v_col[valid]

    order = np.argsort(z_valid)
    z_valid = z_valid[order]
    v_valid = v_valid[order]

    z_valid, unique_idx = np.unique(z_valid, return_index=True)
    v_valid = v_valid[unique_idx]

    if z_valid.size == 1:
        return np.nan, True

    if height < z_valid[0] or height > z_valid[-1]:
        return np.nan, True

    return float(np.interp(height, z_valid, v_valid)), False


def _create_cappi_polar_vertical_interpolation(
    radar,
    height,
    fields=None,
    sweeps=None,
    max_vertical_distance=None,
    vertical_interpolation="linear",
):
    """
    Create CAPPI in native polar coordinates using sweep-wise vertical interpolation.
    """
    sweeps = _iter_sweeps(radar, sweeps=sweeps)
    if not sweeps:
        raise ValueError("No sweep groups found in radar DataTree.")

    ref = _node_to_dataset(radar[sweeps[0]])
    az = np.asarray(ref.azimuth.values, dtype=float)
    r = np.asarray(ref.range.values, dtype=float)
    time_axis = _get_reference_time_axis(ref, len(az))

    if fields is None:
        fields = _default_cappi_fields(ref)

    if not fields:
        raise ValueError("No 2D azimuth/range fields available for CAPPI retrieval.")

    if vertical_interpolation not in {"linear", "nearest"}:
        raise ValueError("vertical_interpolation must be either 'linear' or 'nearest'.")

    z_stack = []
    field_stack = {field: [] for field in fields}

    for sw in sweeps:
        ds = _node_to_dataset(radar[sw])
        z_interp = _interp_to_reference(ds, "z", az, r)
        z_stack.append(z_interp)

        for field in fields:
            data = _interp_to_reference(ds, field, az, r)
            field_stack[field].append(data)

    z_3d = np.stack(z_stack, axis=0)

    max_available = np.nanmax(z_3d)
    min_available = np.nanmin(z_3d)
    if height > max_available or height < min_available:
        raise ValueError(
            f"Requested CAPPI height {height} m is outside available gate heights "
            f"({min_available:.1f} to {max_available:.1f} m)."
        )

    idx = np.argmin(np.abs(z_3d - height), axis=0)
    z_selected = np.take_along_axis(
        z_3d,
        idx[np.newaxis, :, :],
        axis=0,
    ).squeeze(0)

    if max_vertical_distance is None:
        dz = np.diff(np.sort(z_3d.ravel()))
        dz = dz[dz > 0]
        tol = 2.0 * np.median(dz) if dz.size else 1000.0
    else:
        tol = float(max_vertical_distance)

    mask = np.abs(z_selected - height) > tol
    data_vars = {}

    for field in fields:
        data_3d = np.stack(field_stack[field], axis=0)

        if vertical_interpolation == "nearest":
            cappi = np.take_along_axis(
                data_3d,
                idx[np.newaxis, :, :],
                axis=0,
            ).squeeze(0)
            cappi = np.ma.array(cappi, mask=mask)
        else:
            cappi = np.full(z_3d.shape[1:], np.nan, dtype=float)
            linear_mask = np.ones(z_3d.shape[1:], dtype=bool)

            for i in range(z_3d.shape[1]):
                for j in range(z_3d.shape[2]):
                    value, is_masked = _interp_vertical_column(
                        z_3d[:, i, j],
                        data_3d[:, i, j],
                        height,
                    )
                    cappi[i, j] = value
                    linear_mask[i, j] = is_masked

            if np.all(linear_mask):
                cappi = np.take_along_axis(
                    data_3d,
                    idx[np.newaxis, :, :],
                    axis=0,
                ).squeeze(0)
                linear_mask = mask.copy()

            cappi = np.ma.array(cappi, mask=linear_mask)

        data_vars[field] = (("time", "range"), cappi, ref[field].attrs)

    sweep_mode = _get_reference_metadata_value(
        ref, "sweep_mode", "azimuth_surveillance"
    )
    follow_mode = _get_reference_metadata_value(ref, "follow_mode", "none")
    prt_mode = _get_reference_metadata_value(ref, "prt_mode", "fixed")
    longitude = _extract_scalar(ref, "longitude")
    latitude = _extract_scalar(ref, "latitude")
    radar_altitude = _extract_scalar(ref, "altitude")

    data_vars.update(
        {
            "sweep_number": (("sweep",), np.array([0], dtype=int)),
            "fixed_angle": (("sweep",), np.array([0.0], dtype=float)),
            "sweep_mode": (("sweep",), np.array([sweep_mode], dtype=object)),
            "follow_mode": (("sweep",), np.array([follow_mode], dtype=object)),
            "prt_mode": (("sweep",), np.array([prt_mode], dtype=object)),
            "sweep_start_ray_index": (("sweep",), np.array([0], dtype=int)),
            "sweep_end_ray_index": (("sweep",), np.array([len(az) - 1], dtype=int)),
        }
    )

    ds_cappi = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": ("time", time_axis),
            "azimuth": ("time", az),
            "elevation": ("time", np.zeros_like(az, dtype=float)),
            "range": ("range", r),
            "sweep": ("sweep", np.array([0], dtype=int)),
            **(
                {
                    "longitude": longitude,
                    "latitude": latitude,
                    "altitude": float(height),
                }
                if longitude is not None and latitude is not None
                else {}
            ),
        },
        attrs={
            "product": "CAPPI",
            "height": float(height),
            "method": _POLAR_VERTICAL_INTERPOLATION_METHOD,
            "vertical_interpolation": vertical_interpolation,
            "max_vertical_distance": float(tol),
            **(
                {"radar_altitude": float(radar_altitude)}
                if radar_altitude is not None
                else {}
            ),
        },
    )

    return ds_cappi


def _create_cappi_height_window_composite(
    radar,
    height,
    fields=None,
    sweeps=None,
    height_window=500.0,
    apply_quality_control=False,
    velocity_field=None,
    reflectivity_field=None,
    texture_window=50,
    texture_threshold=None,
    reflectivity_min=-10.0,
    reflectivity_max=75.0,
):
    """
    Create a CAPPI-like horizontal composite by selecting gates within a
    vertical window around the target height and compositing the closest gates
    after sweep alignment.
    """
    sweeps = _iter_sweeps(radar, sweeps=sweeps)
    if not sweeps:
        raise ValueError("No sweep groups found in radar DataTree.")

    ref = _node_to_dataset(radar[sweeps[0]])
    az = np.asarray(ref.azimuth.values, dtype=float)
    r = np.asarray(ref.range.values, dtype=float)
    time_axis = _get_reference_time_axis(ref, len(az))

    if fields is None:
        fields = _default_cappi_fields(ref)

    if not fields:
        raise ValueError(
            "No 2D azimuth/range fields available for height-window compositing."
        )

    z_stack = []
    field_stack = {field: [] for field in fields}

    for sw in sweeps:
        ds = _node_to_dataset(radar[sw])

        if apply_quality_control:
            ds = _apply_velocity_texture_gate_filter(
                ds,
                velocity_field=velocity_field,
                reflectivity_field=reflectivity_field,
                texture_window=texture_window,
                texture_threshold=texture_threshold,
                reflectivity_min=reflectivity_min,
                reflectivity_max=reflectivity_max,
            )

        z_interp = _interp_to_reference(ds, "z", az, r)
        z_stack.append(z_interp)

        for field in fields:
            field_stack[field].append(_interp_to_reference(ds, field, az, r))

    z_3d = np.stack(z_stack, axis=0)
    distance = np.abs(z_3d - float(height))
    within_window = distance <= float(height_window)
    best_idx = np.argmin(
        np.where(within_window, distance, np.inf),
        axis=0,
    )

    sweep_mode = _get_reference_metadata_value(
        ref, "sweep_mode", "azimuth_surveillance"
    )
    follow_mode = _get_reference_metadata_value(ref, "follow_mode", "none")
    prt_mode = _get_reference_metadata_value(ref, "prt_mode", "fixed")
    longitude = _extract_scalar(ref, "longitude")
    latitude = _extract_scalar(ref, "latitude")
    radar_altitude = _extract_scalar(ref, "altitude")

    data_vars = {}
    for field in fields:
        field_3d = np.stack(field_stack[field], axis=0)
        composite = np.take_along_axis(
            field_3d,
            best_idx[np.newaxis, :, :],
            axis=0,
        ).squeeze(0)

        mask = ~np.take_along_axis(
            within_window,
            best_idx[np.newaxis, :, :],
            axis=0,
        ).squeeze(0)
        composite = np.ma.array(composite, mask=mask)
        data_vars[field] = (("time", "range"), composite, ref[field].attrs)

    data_vars.update(
        {
            "sweep_number": (("sweep",), np.array([0], dtype=int)),
            "fixed_angle": (("sweep",), np.array([0.0], dtype=float)),
            "sweep_mode": (("sweep",), np.array([sweep_mode], dtype=object)),
            "follow_mode": (("sweep",), np.array([follow_mode], dtype=object)),
            "prt_mode": (("sweep",), np.array([prt_mode], dtype=object)),
            "sweep_start_ray_index": (("sweep",), np.array([0], dtype=int)),
            "sweep_end_ray_index": (("sweep",), np.array([len(az) - 1], dtype=int)),
        }
    )

    ds_cappi = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": ("time", time_axis),
            "azimuth": ("time", az),
            "elevation": ("time", np.zeros_like(az, dtype=float)),
            "range": ("range", r),
            "sweep": ("sweep", np.array([0], dtype=int)),
            **(
                {
                    "longitude": longitude,
                    "latitude": latitude,
                    "altitude": float(height),
                }
                if longitude is not None and latitude is not None
                else {}
            ),
        },
        attrs={
            "product": "CAPPI",
            "height": float(height),
            "method": _HEIGHT_WINDOW_COMPOSITE_METHOD,
            "height_window": float(height_window),
            "apply_quality_control": bool(apply_quality_control),
            **(
                {"radar_altitude": float(radar_altitude)}
                if radar_altitude is not None
                else {}
            ),
        },
    )

    return ds_cappi


def create_cappi(
    radar,
    height,
    method=_CARTESIAN_IDW_METHOD,
    vertical_tolerance=None,
    apply_filter=False,
    *,
    fields=None,
    sweeps=None,
    x=None,
    y=None,
    x_res=1000.0,
    y_res=1000.0,
    padding=0.0,
):
    """
    Create a Constant Altitude Plan Position Indicator (CAPPI).

    Parameters
    ----------
    radar : xarray.DataTree
        Georeferenced radar volume containing one or more sweep groups.
    height : float
        Target CAPPI altitude in meters.
    method : {
        "cartesian_idw",
        "polar_vertical_interpolation",
        "height_window_composite",
    }, optional
        CAPPI retrieval method. ``"cartesian_idw"`` performs 3D anisotropic
        inverse-distance weighting on a regular ``x``/``y`` grid.
        ``"polar_vertical_interpolation"`` retains the native
        ``azimuth``/``range`` geometry and interpolates vertically across sweeps.
        ``"height_window_composite"`` forms a CAPPI-like composite by selecting
        the nearest gates within a prescribed vertical window after sweep
        alignment. Legacy aliases ``"cartesian"``, ``"polar"``, and
        ``"pseudo_cappi"`` are also accepted.
    vertical_tolerance : float or None, optional
        Maximum vertical distance above and below the requested CAPPI height,
        in meters, used by the selected retrieval method. If omitted, a
        method-specific default is used.
    apply_filter : bool, optional
        Apply built-in gate filtering when supported by the selected method.
        Currently this is used by ``method="height_window_composite"``.
    fields : list[str] or None, optional
        Radar variables to retrieve. If omitted, likely 2D radar fields are
        selected automatically.
    sweeps : list[str] or None, optional
        Sweep names to include in the retrieval. If omitted, all available
        sweep groups are used.
    x, y : array-like or None, optional
        Target Cartesian grid coordinates in meters. Used only with
        ``method="cartesian_idw"``.
    x_res, y_res : float, optional
        Cartesian output spacing in meters when ``x`` and ``y`` are not
        supplied. Used only with ``method="cartesian_idw"``.
    padding : float, optional
        Extra padding, in meters, applied to the Cartesian output domain.
        Used only with ``method="cartesian_idw"``.

    Returns
    -------
    xarray.Dataset
        CAPPI dataset in either Cartesian ``(y, x)`` or native polar
        ``(azimuth, range)`` geometry, depending on the selected method.
    """
    method = _normalize_cappi_method(method)
    vertical_tolerance = _resolve_vertical_tolerance(method, vertical_tolerance)

    if method == _CARTESIAN_IDW_METHOD:
        return _create_cappi_cartesian_idw(
            radar=radar,
            height=height,
            fields=fields,
            sweeps=sweeps,
            x=x,
            y=y,
            x_res=x_res,
            y_res=y_res,
            padding=padding,
            vertical_window=vertical_tolerance,
        )

    if method == _POLAR_VERTICAL_INTERPOLATION_METHOD:
        return _create_cappi_polar_vertical_interpolation(
            radar=radar,
            height=height,
            fields=fields,
            sweeps=sweeps,
            max_vertical_distance=vertical_tolerance,
        )

    if method == _HEIGHT_WINDOW_COMPOSITE_METHOD:
        ref_sweeps = _iter_sweeps(radar, sweeps=sweeps)
        ref_ds = _node_to_dataset(radar[ref_sweeps[0]])
        return _create_cappi_height_window_composite(
            radar=radar,
            height=height,
            fields=fields,
            sweeps=sweeps,
            height_window=vertical_tolerance,
            apply_quality_control=apply_filter,
            velocity_field=_infer_field_name(ref_ds, _COMMON_VELOCITY_FIELDS),
            reflectivity_field=_infer_field_name(ref_ds, _COMMON_REFLECTIVITY_FIELDS),
        )

    raise ValueError(f"Unsupported CAPPI method '{method}'.")
