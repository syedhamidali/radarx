---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3
  name: python3
  language: python
---

# Exporting results

A radarx result is an xarray object: a `DataTree` volume with the fields radarx
derived, or a gridded `Dataset`. `radarx.io` writes them to files that other
tools read and keeps the provenance (`radarx_method`, `radarx_references`,
`radarx_version`, `history`) wherever the format can hold it.

| function | format | for | `.radarx` method |
|---|---|---|---|
| `write_cfradial1` | CfRadial1 NetCDF | volume | `dtree.radarx.to_cfradial1` |
| `write_cfradial2` | CfRadial2 NetCDF | volume | `dtree.radarx.to_cfradial2` |
| `write_odim` | ODIM_H5 | volume | `dtree.radarx.to_odim` |
| `write_zarr` | Zarr | volume, grid, any result | `.radarx.to_zarr` |
| `write_netcdf_cf` | CF NetCDF with georeferencing | grid | `ds.radarx.to_netcdf_cf` |

The three radar formats are written by the writers of xradar, which know the
formats; radarx prepares a copy of the object around them and never changes the
input or copies the arrays. The preparation does what a result of radarx needs
and an xradar volume read from a file often does not get right:

- attributes that NetCDF cannot store are converted: `None` is dropped, `True`
  becomes `"true"` (as CfRadial writes it), lists and numpy values become what
  the format accepts;
- `_FillValue` is moved from the attributes to the encoding of floating
  variables and removed from integer variables, where it would turn the
  class 0 of the hydrometeor classes into NaN on reading;
- the integer packing a reader leaves in the encoding of a float field
  (`uint8` with `scale_factor`) is dropped by default, because NaN and values
  outside the packed range would be written as wrong numbers
  (`packing="keep"` keeps it);
- stale encodings (chunk layout of the source file, time units) are dropped.

The functions are named `write_*` because `radarx.io.to_cfradial2` is the
converter of the deprecated IMD reader.

```{code-cell} ipython3
import tempfile
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr
import xradar as xd
from open_radar_data import DATASETS

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.io import (
    write_cfradial1,
    write_cfradial2,
    write_netcdf_cf,
    write_odim,
    write_zarr,
)
from radarx.retrieve import create_cappi, echo_mask, estimate_kdp
from radarx.grid import grid_cones

warnings.filterwarnings("ignore")
work = Path(tempfile.mkdtemp())  # every file of this page goes here
```

## A volume with derived fields

Three sweeps of the KLBB WSR-88D (Lubbock, Texas) out to 80 km, with the
processed differential phase, `KDP` and the echo classification that radarx
derives. `.radarx.assign` merges the products into the sweeps and keeps their
method, references and history.

```{code-cell} ipython3
raw = xd.io.open_nexradlevel2_datatree(DATASETS.fetch("KLBB20160601_150025_V06"), sweep=[0, 2, 4])
raw = raw.map_over_datasets(lambda ds: ds.sel(range=slice(None, 80e3)) if "range" in ds.dims else ds)
raw = raw.xradar.georeference()
vol = raw.radarx.assign(estimate_kdp(raw)).radarx.assign(echo_mask(raw))
sweep = vol["sweep_0"]
print(list(sweep.data_vars))
print(sweep.attrs["radarx_method"])
print(sweep.attrs["history"])
```

## Zarr

Zarr stores the volume as it is, with lazy dask arrays written chunk by chunk,
and reads it back identical: values, dtypes, attributes (booleans and `None`
included) and the provenance of every node and variable.

```{code-cell} ipython3
path = vol.radarx.to_zarr(work / "volume.zarr")
back = xr.open_datatree(path, engine="zarr")
xr.testing.assert_identical(back, vol)
print(path.name, "reads back identical")
```

## CfRadial2

`write_cfradial2` keeps the tree: one NetCDF group per sweep. The writer of
xradar conforms every sweep to the CfRadial2 layout, so some content does not
survive: the sweep attributes, the variables without a `range` dimension
(`PHIDP_OFFSET`), the georeferenced `x`, `y`, `z`, and the ray dimension
becomes `time`. radarx therefore merges the provenance of the sweeps onto the
root attributes before writing.

```{code-cell} ipython3
path = vol.radarx.to_cfradial2(work / "volume_cfradial2.nc")
cf2 = xr.open_datatree(path)
print(sorted(set(sweep.variables) - set(cf2["sweep_0"].variables)))
print("root:", cf2.attrs["radarx_method"][:70], "...")
print("history:", cf2.attrs["history"].replace("\n", " | "))
print("on the field:", cf2["sweep_0"]["KDP"].attrs["radarx_references"][:50], "...")
```

## CfRadial1

CfRadial1 is one NetCDF dataset: the sweeps are stacked along `time` and padded
to the longest range, the sweep attributes do not exist and the fields are
ordinary variables of `(time, range)`. The provenance is on the global
attributes (merged from the sweeps) and on the fields; the booleans of the
attributes are the strings `"true"` and `"false"`.

```{code-cell} ipython3
path = vol.radarx.to_cfradial1(work / "volume_cfradial1.nc")
cf1 = xr.open_dataset(path)
print(dict(cf1.sizes))
print({k: cf1.attrs[k][:40] for k in ("radarx_method", "history", "avset_enabled")})
print(cf1["KDP"].dims, cf1["ECHO_CLASS"].dtype, cf1["PHIDP_OFFSET"].dims)
```

## ODIM_H5

`write_odim` writes one `datasetN` per sweep and one `dataM` per field on the
ray by gate grid, with `what/quantity` set to the name of the variable.
ODIM_H5 2.4.1 lists its quantities in Table 16: `DBZH`, `ZDR`, `RHOHV`,
`PHIDP`, `KDP` and `VRADH` are there, `PHIDP_processed`, `ECHO_CLASS` and
`METEO_SCORE` are not, and a reader other than xradar may skip them.
`fields=` writes a subset. `source` is the radar identifier that the
specification requires in `what/source` (`NOD:` and the node of the radar); the
one used here is written by hand for the example. The attributes of fields, sweeps and root are not
stored by the format, so the provenance is added to the `how` groups, where
the specification allows additional attributes. Booleans are written as
`uint8`.

```{code-cell} ipython3
path = vol.radarx.to_odim(work / "volume.h5", source="NOD:uskbb")
with h5py.File(path) as f:
    names = {
        f["dataset1"][k]["what"].attrs["quantity"].decode(): k
        for k in f["dataset1"]
        if k.startswith("data") and k != "data"
    }
    print(names)
    print(f["dataset1/how"].attrs["history"].decode().replace("\n", " | "))
    print(f["dataset1"][names["KDP"]]["how"].attrs["radarx_version"].decode())
    print("DBZH has a how group:", "how" in f["dataset1"][names["DBZH"]])
odim = xr.open_datatree(path, engine="odim")
kdp_back = odim["sweep_0"]["KDP"]
print(np.allclose(kdp_back, sweep["KDP"], equal_nan=True), kdp_back.dtype)
```

With `optional_how=True` (the default of `write_odim`) the angles and times of
every ray are written, so xradar reads back the original azimuths and times.
Without it the reader rebuilds a regular grid of rays.

## Which fields are written where

The same volume read back from each format, for the variables of the first
sweep.

```{code-cell} ipython3
with h5py.File(work / "volume.h5") as f:
    in_odim = set(names)
back_zarr = xr.open_datatree(work / "volume.zarr", engine="zarr")["sweep_0"]
present = {
    "Zarr": set(back_zarr.variables),
    "CfRadial2": set(cf2["sweep_0"].variables),
    "CfRadial1": set(cf1.variables),
    "ODIM_H5": in_odim,
}
own = sweep.to_dataset(inherit=False)
rows = [v for v in own.variables if v not in ("azimuth", "elevation", "time", "range")]
table = pd.DataFrame(
    {fmt: ["yes" if v in names_ else "no" for v in rows] for fmt, names_ in present.items()},
    index=rows,
)
table
```

The `x`, `y`, `z` gate positions come from `dtree.xradar.georeference()` and
are not part of CfRadial1, CfRadial2 or ODIM_H5; the per-ray `PHIDP_OFFSET` is
a CfRadial1 variable of `time` and has no place in the other two. CfRadial1
calls `sweep_fixed_angle` `fixed_angle`, and the scalar sweep metadata
(`sweep_mode`, `sweep_number`, ...) are variables of the CfRadial formats and
attributes of ODIM groups (`elangle` for the fixed angle).

## Grids

The Datasets of `grid_cones`, `grid_radars` and `create_cappi` are georeferenced
when they are built: 2-D `lat(y, x)` and `lon(y, x)` from the CRS, a `crs_wkt`
grid mapping variable with `grid_mapping` on every field, and `standard_name`,
`units` and `axis` on `x`, `y`, `z` and `time`. `write_netcdf_cf` writes
them as a CF file and adds what a file needs on top of that

- the grid mapping as a `crs` data variable (a scalar coordinate would be listed
  in the `coordinates` attribute of every field),
- the cell bounds of the axes,
- `Conventions`, and the same georeferencing for a grid that does not have it
  (a grid of a retrieval, or one with 1-D `lat(y)` and `lon(x)` from an older
  radarx), or another CRS with `crs=`.

```{code-cell} ipython3
grid = grid_cones(
    vol,
    "DBZH",
    x=np.arange(-60e3, 60e3 + 1, 2e3),
    y=np.arange(-60e3, 60e3 + 1, 2e3),
    z=np.arange(500.0, 6000.0, 500.0),
)
path = grid.radarx.to_netcdf_cf(work / "grid.nc")
cf = xr.open_dataset(path)
print(cf["DBZH"].attrs["grid_mapping"], cf["crs"].attrs["grid_mapping_name"])
print({k: cf[k].attrs.get("standard_name") for k in ("x", "y", "z", "time", "lat", "lon")})
print("lat:", grid["lat"].dims, "->", cf["lat"].dims, "; grid_mapping in the grid:", grid["DBZH"].attrs["grid_mapping"])
print(cf["x_bnds"].values[:2].tolist(), cf.attrs["Conventions"])
np.testing.assert_array_equal(cf["DBZH"], grid["DBZH"])
print(cf.attrs["radarx_method"], "|", cf.attrs["history"])
```

A CAPPI has no CRS. Its `x` and `y` are metres from the radar, which is how
xradar georeferences a sweep, so the CRS is an azimuthal equidistant
projection centred on the scalar `latitude` and `longitude` of the Dataset. Pass
`crs=` to say otherwise, or `crs=False` for none.

```{code-cell} ipython3
cappi = create_cappi(vol, height=1500.0, data_vars=["DBZH"], x_res=2000.0, y_res=2000.0)
path = write_netcdf_cf(cappi, work / "cappi.nc")
with xr.open_dataset(path) as back:
    print(back["crs"].attrs["latitude_of_projection_origin"], back["lat"].dims)
```

`write_zarr(grid, path, cf=True)` writes the same georeferencing to a Zarr
store.

```{code-cell} ipython3
path = write_zarr(grid, work / "grid.zarr", cf=True)
with xr.open_dataset(path, engine="zarr") as back:
    print(back["DBZH"].attrs["grid_mapping"], back["lat"].dims)
```

## Attributes and fill values

The preparation on a Dataset that has what NetCDF cannot store: a boolean, a
`None`, a numpy scalar, a list, a dictionary, and class codes with a
`_FillValue` attribute.

```{code-cell} ipython3
ds = xr.Dataset(
    {
        "HID": (("y", "x"), np.array([[0, 1, 2, 3]], dtype="int8"), {"_FillValue": 0}),
        "KDP": (("y", "x"), np.array([[0.5, np.nan, 1.5, 2.0]]), {"_FillValue": -9999.0}),
    },
    coords={"x": np.arange(4.0), "y": [0.0]},
    attrs={"qc": True, "unused": None, "bins": [1, 2, 3], "run": {"mode": "fast"}},
)
path = write_netcdf_cf(ds, work / "attrs.nc", crs=False)
with xr.open_dataset(path) as back:
    print(back["HID"].dtype, back["HID"].values.tolist(), "_FillValue" in back["HID"].attrs)
    print(back["KDP"].values.tolist())
    print({k: v for k, v in back.attrs.items() if k != "Conventions"})
```

## What cannot be written

- `radarx.io.write_cfradial1` and `write_cfradial2` need a `DataTree` with
  `sweep_*` nodes; `write_odim` also needs a radar identifier for
  `what/source` (`source="NOD:..."`; newer xradar builds it from the
  attributes of a volume that was read from ODIM, xradar 0.11 does not).
- The zarr package 3 needs xarray 2025.1 or newer; with xarray 2024.10
  install `zarr<3`.
- Only the CF NetCDF of a grid was checked with a validator (the CF checker
  4.1.0 against CF-1.8: no errors, a warning for a field without
  `long_name`). CfRadial1, CfRadial2 and ODIM_H5 files were read back with
  xarray, h5py and xradar, which the tests do for every format
  (`tests/test_export.py`); the CfRadial and ODIM_H5 specifications were not
  used to validate them.
