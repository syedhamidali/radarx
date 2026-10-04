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

# Wradlib

```{code-cell} ipython3
import numpy as np
import radarx as rx
import xarray as xr
import xradar as xd

# from osgeo import osr
import wradlib as wrl
from datetime import datetime
```

## Define Funcs

```{code-cell} ipython3
def filter_radar(ds):
    """Filters radar data for valid reflectivity values."""
    return ds.where((ds.reflectivity >= 0) & (ds.reflectivity <= 70))


def setup_grid(dtree, horiz_res=2000, vert_res=500, maxrange=275e3, maxalt=15e3):
    """Creates a 3D grid for wradlib."""
    # sitecoords = (
    #     dtree["sweep_0"].longitude.values,
    #     dtree["sweep_0"].latitude.values,
    #     dtree["sweep_0"].altitude.values,
    # )
    #
    #     proj = osr.SpatialReference()
    #     proj.ImportFromEPSG(4326)  # WGS 84 (EPSG:4326)
    #     trgxyz, trgshape = wrl.vpr.make_3d_grid(
    #         sitecoords, proj, maxrange, maxalt, horiz_res, vert_res)
    # ------------------
    #     This seems to crash RTD build, so trying something else
    #     see log https://app.readthedocs.org/api/v2/build/26513933.txt
    # -----------------

    x_lims = (-300e3, 250e3)
    y_lims = (-250e3, 300e3)

    lat, lon, x, y, z, trg_crs = rx.grid.make_3d_grid(
        dtree["sweep_0"].to_dataset(),
        x_lim=x_lims,
        y_lim=y_lims,
        x_step=horiz_res,
        y_step=horiz_res,
        z_lim=(0, maxalt),
        z_step=vert_res,
    )

    trgshape = len(z), len(y), len(x)
    trgxyz = wrl.util.gridaspoints(z, y, x)
    return trgxyz, trgshape


def process_sweeps(raw_dt, data_var="reflectivity"):
    """Processes sweep data and prepares for gridding."""
    swp_list, data_list = [], []

    for swp in raw_dt.match("sweep_*"):
        ds = raw_dt[swp].to_dataset()
        xyz = (
            xr.concat(
                [
                    ds.coords["x"].reset_coords(drop=True),
                    ds.coords["y"].reset_coords(drop=True),
                    ds.coords["z"].reset_coords(drop=True),
                ],
                "xyz",
            )
            .stack(npoints=("azimuth", "range"))
            .transpose(..., "xyz")
        )
        swp_list.append(xyz)
        data = ds[data_var].stack(npoints=("azimuth", "range"))
        data_list.append(data)

    xyz = xr.concat(swp_list, "npoints")
    data = xr.concat(data_list, "npoints")
    return xyz, data


def interpolate_to_grid(xyz, trgxyz, data, trgshape, maxrange, minelev, maxelev):
    """Interpolates radar data to a Cartesian 3D volume grid."""
    gridder = wrl.vpr.CAPPI(
        xyz.values,
        trgxyz,
        maxrange=maxrange,
        minelev=minelev,
        maxelev=maxelev,
    )
    vol = np.ma.masked_invalid(gridder(data.values).reshape(trgshape))
    return vol


def create_dataset(
    vol, trgxyz, trgshape, data, proj_crs, dtree, data_var="reflectivity"
):
    """Creates an Xarray dataset from gridded radar data."""
    trgx, trgy, trgz = (
        trgxyz[:, 0].reshape(trgshape)[0, 0, :],
        trgxyz[:, 1].reshape(trgshape)[0, :, 0],
        trgxyz[:, 2].reshape(trgshape)[:, 0, 0],
    )

    lon, lat = rx.utils.cartesian_to_geographic_aeqd(
        trgx,
        trgy,
        data.longitude.values,
        data.latitude.values,
        xd.georeference.get_earth_radius(proj_crs, data.latitude.values),
    )

    ds_wrl = xr.DataArray(
        data=vol,
        coords={"z": trgz, "y": trgy, "x": trgx},
        dims=("z", "y", "x"),
        name=data_var,
    ).to_dataset()

    ds_wrl["time"] = data.time.mean()
    ds_wrl.attrs = dtree.attrs
    ds_wrl["latitude"] = data["latitude"]
    ds_wrl["longitude"] = data["longitude"]
    ds_wrl["lon"] = xr.DataArray(lon, dims=["x"])
    ds_wrl["lat"] = xr.DataArray(lat, dims=["y"])
    ds_wrl = ds_wrl.set_coords(["lon", "lat"])
    return ds_wrl
```

# Run

```{code-cell} ipython3
import os

import pyart

# Created in part 1; recreate it when this notebook runs on its own
if not os.path.exists("KSGF20180612_083109_V06.nc"):
    radar = pyart.io.read_nexrad_archive(
        "s3://unidata-nexrad-level2/2018/06/12/KSGF/KSGF20180612_083109_V06"
    )
    pyart.io.write_cfradial("KSGF20180612_083109_V06.nc", radar)
```

```{code-cell} ipython3
def main():
    # Parameters
    filename = "KSGF20180612_083109_V06.nc"

    # Grid Setup
    h_res, v_res = 2000, 500
    maxrange, maxalt = 275e3, 15e3
    minelev, maxelev = 0.2, 21.0

    data_var = "reflectivity"

    # Read and process data
    dtree = xd.io.open_cfradial1_datatree(filename)
    dtree = rx.utils.combine_nexrad_sweeps(dtree)
    dtree = dtree.xradar.map_over_sweeps(filter_radar)
    dtree = dtree.xradar.georeference()

    tstart = datetime.now()

    trgxyz, trgshape = setup_grid(
        dtree, horiz_res=h_res, vert_res=v_res, maxrange=maxrange, maxalt=maxalt
    )

    raw_dt = dtree.xradar.map_over_sweeps(rx.utils.get_geocoords)
    proj_crs = xd.georeference.get_crs(raw_dt["sweep_0"].ds)

    xyz, data = process_sweeps(raw_dt, data_var=data_var)

    vol = interpolate_to_grid(xyz, trgxyz, data, trgshape, maxrange, minelev, maxelev)
    ds_wrl = create_dataset(
        vol, trgxyz, trgshape, data, proj_crs, dtree, data_var=data_var
    )

    print("Wradlib gridding took:", datetime.now() - tstart)
    display(ds_wrl)

    # Diagnostic plot
    ds_wrl.radarx.plot_max_cappi(
        data_var=data_var,
        vmin=-10,
        vmax=70,
        range_rings=True,
        add_map=True,
    )


if __name__ == "__main__":
    main()
```
