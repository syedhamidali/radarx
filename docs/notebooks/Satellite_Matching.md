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

# Ground radar and satellite: GPM DPR volume matching and IMERG

A ground radar (GR) and a spaceborne radar see the same storm from below and from above, with
different beams, at different times. To compare them, for the calibration of the ground radar,
the check of the retrievals of the Dual-frequency Precipitation Radar (DPR) of the GPM core
satellite or the study of the sampling of both, the measurements are averaged over the volume
that the two beams share (Schwaller and Morris 2011). radarx does this in two steps:

- `match_dpr` finds, for every DPR footprint near the radar and every sweep of the ground radar,
  the match volume with the parallax of the slanted DPR ray taken into account, and averages both
  radars over it. A compiled kernel does it for all sweeps and footprints in one call.
- `match_imerg` averages a radar rain rate onto the 0.1 degree grid and the 30 minute steps of the
  Integrated Multi-satellitE Retrievals for GPM (IMERG; Huffman et al. 2020).

`match_statistics` gives the bias, error, correlation and detection scores of either pair, and
`dpr_dsd` and `dpr_parallax` are the helpers for the DSD parameters and the geometry.

The DPR and IMERG files need an Earthdata login (NASA GES DISC), so this notebook builds
a swath and an IMERG field from a known storm: the answer is known and the notebook runs
offline. The DPR product 2A-DPR has the variables used here: `Longitude` and `Latitude` of the
rays at the ellipsoid, `localZenithAngle` and `height` of the range bins (group `PRE`) and
`zFactorFinal`, `paramDSD` (group `SLV`), with the dimensions scan, ray and bin.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import xarray as xr
from xradar.georeference import antenna_to_cartesian

import radarx  # noqa: F401  registers the .radarx accessors
from radarx._gridcf import aeqd_crs, lonlat
from radarx.retrieve import (
    dpr_dsd,
    dpr_parallax,
    dsd,
    match_dpr,
    match_imerg,
    match_statistics,
    radar_from_dsd,
)

SITE = {"latitude": 30.0, "longitude": -90.0, "altitude": 0.0}
```

## 1. A storm that both radars observe

The storm is a convective core that leans to the east with height (0.5 km per km) and loses
reflectivity above 8 km. The ground radar volume has eight sweeps; every gate takes the value of the
storm at its position. The DPR swath, with 49 rays and up to 17.5 degrees of zenith angle, flies
north 30 km west of the radar. A DPR range bin at the height $h$ lies $h \tan\theta$ closer to the
track than the position that the file gives for the ray (at the ellipsoid), so the bins take the
value of the storm at their true position.

```{code-cell} ipython3
def storm_dbz(x, y, h):
    """Reflectivity (dBZ) of the leaning storm at the east x, north y and height h (m)."""
    xc = 55e3 + 0.5 * h
    r2 = (x - xc) ** 2 + (y - 5e3) ** 2
    dbz = 52.0 * np.exp(-r2 / (2 * 9e3**2)) - 3.0 * np.maximum(h - 8e3, 0.0) / 1e3
    return np.where(dbz > 5.0, dbz, np.nan)


elevations = (0.5, 1.5, 3.0, 5.0, 8.0, 12.0, 16.0, 19.5)
azimuth = np.arange(0.0, 360.0, 1.0) + 0.5
rng = 125.0 + 250.0 * np.arange(600)
sweeps = {}
for i, el in enumerate(elevations):
    gx, gy, gz = antenna_to_cartesian(rng[None, :], azimuth[:, None], np.full((azimuth.size, 1), el))
    sweeps[f"/sweep_{i}"] = xr.Dataset(
        {"DBZH": (("azimuth", "range"), storm_dbz(gx, gy, gz), {"units": "dBZ"})},
        coords={"azimuth": azimuth, "elevation": ("azimuth", np.full(azimuth.size, el)), "range": rng},
    )
site = xr.Dataset(coords=SITE, data_vars={"radar_beam_width_h": 1.0})
volume = xr.DataTree.from_dict({"/": site, **sweeps})

# the DPR swath: 49 rays, 5 km between the scans, 407 km altitude
crs = aeqd_crs(SITE["latitude"], SITE["longitude"])
to_lonlat = pyproj.Transformer.from_crs(crs, pyproj.CRS.from_epsg(4326), always_xy=True)
zenith = (np.arange(49) - 24) * 17.5 / 24.0
tan = np.tan(np.radians(zenith))
track_x = -30e3
ray_x = track_x + 407e3 * tan
scan_y = np.arange(-100e3, 100e3, 5e3)
xx, yy = np.meshgrid(ray_x, scan_y)
lon, lat = to_lonlat.transform(xx, yy)
nbin = 176
height = ((nbin - 1 - np.arange(nbin)) * 125.0)[None, None, :] * np.cos(np.radians(zenith))[None, :, None]
height = np.broadcast_to(height, (scan_y.size, 49, nbin)).copy()
# the footprint (5 km circle) average at the true position of every bin
offset = np.linspace(-2000.0, 2000.0, 5)
ox, oy = np.meshgrid(offset, offset)
disc = np.hypot(ox, oy) <= 2500.0
true_x = xx[..., None] - height * tan[None, :, None]
mean_linear = np.zeros(height.shape)
for dx, dy in zip(ox[disc], oy[disc]):
    mean_linear += np.nan_to_num(10 ** (0.1 * storm_dbz(true_x + dx, yy[..., None] + dy, height)))
mean_linear /= disc.sum()
zfactor = np.where(mean_linear > 0, 10 * np.log10(np.where(mean_linear > 0, mean_linear, 1.0)), np.nan)
dpr = xr.Dataset(
    {
        "Longitude": (("scan", "ray"), lon),
        "Latitude": (("scan", "ray"), lat),
        "localZenithAngle": (("scan", "ray"), np.broadcast_to(zenith, lon.shape).copy()),
        "height": (("scan", "ray", "bin"), height),
        "zFactorFinal": (("scan", "ray", "bin"), zfactor, {"units": "dBZ"}),
    }
)
print(f"{scan_y.size} scans, {nbin} bins of 125 m; parallax of a 5 km bin on the outer ray: "
      f"{float(dpr_parallax(5000.0, zenith[-1])):.0f} m")
```

## 2. The match volumes

`match_dpr` returns one entry per footprint and sweep: the means of both radars, their standard
deviation in the volume, the fraction of the volume that has data, the centre of the volume
(`match_longitude`, `match_latitude`, `match_height`) and `valid`, true where every mean has a value.
Reflectivities are averaged in linear units and converted back to dBZ. Without the correction (the
zenith angle set to 0) the same call shows what the parallax costs.

```{code-cell} ipython3
kwargs = dict(gr_fields="DBZH", dpr_fields="zFactorFinal", min_range=20e3, max_range=100e3)
matched = match_dpr(volume, dpr, **kwargs)
flat = dpr.copy()
flat["localZenithAngle"] = flat["localZenithAngle"] * 0.0
uncorrected = match_dpr(volume, flat, **kwargs)
print(matched)
```

The agreement of the pairs where both radars see the storm. The beams of the ground radar and the
DPR sample different volumes, so the scatter is not zero even for a perfect parallax. How large is the
shift? Scaling it from 0 to 2 times the geometry of the swath, the error has its minimum near 1.

```{code-cell} ipython3
def agreement(m, min_elevation=3.0, min_zenith=8.0):
    ok = (
        m.valid
        & (m.gr_DBZH > 15)
        & (m.dpr_zFactorFinal > 15)
        & (m.elevation >= min_elevation)
        & (abs(m.ray - 24) * 17.5 / 24.0 >= min_zenith)
    )
    return match_statistics(m.gr_DBZH.where(ok), m.dpr_zFactorFinal.where(ok))


scales = np.linspace(0.0, 2.0, 11)
rmse = []
for scale in scales:
    scaled = dpr.copy()
    scaled["localZenithAngle"] = np.degrees(np.arctan(scale * np.tan(np.radians(dpr["localZenithAngle"]))))
    rmse.append(float(agreement(match_dpr(volume, scaled, **kwargs)).rmse))
for label, m in (("with parallax correction", matched), ("without", uncorrected)):
    s = agreement(m)
    print(f"{label}: {int(s['count'])} pairs, bias {float(s.bias):.2f} dB, RMSE {float(s.rmse):.2f} dB, "
          f"correlation {float(s.correlation):.3f}")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
axes[0].plot(scales, rmse, "o-")
axes[0].axvline(1.0, color="k", ls=":")
axes[0].set(xlabel="scale of the parallax shift", ylabel="RMSE (dB)", title="sweeps from 3 deg, rays 8 deg off nadir")
for ax, m, title in ((axes[1], uncorrected, "no parallax correction"), (axes[2], matched, "parallax corrected")):
    ok = m.valid & (m.gr_DBZH > 15) & (m.dpr_zFactorFinal > 15) & (m.elevation >= 3.0)
    ax.hexbin(m.gr_DBZH.where(ok).values.ravel(), m.dpr_zFactorFinal.where(ok).values.ravel(),
              gridsize=30, mincnt=1, bins="log", extent=(15, 55, 15, 55))
    ax.plot([15, 55], [15, 55], "k-", lw=0.8)
    ax.set(xlabel="ground radar (dBZ)", ylabel="DPR (dBZ)", title=title, aspect="equal")
```

The centre of a match volume is the footprint displaced by the parallax at the height of the beam.
The map shows, for the 8 degree sweep, where the ray is at the ellipsoid (grey) and where the volume
is (red), against the core of the storm at that height.

```{code-cell} ipython3
k = list(elevations).index(8.0)
to_xy = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), crs, always_xy=True)
fx, fy = to_xy.transform(matched.longitude.values, matched.latitude.values)
mx, my = to_xy.transform(matched.match_longitude.values[:, k], matched.match_latitude.values[:, k])
height_of_beam = float(np.nanmedian(matched.match_height.values[:, k]))
gx, gy = np.meshgrid(np.arange(20e3, 110e3, 500.0), np.arange(-40e3, 50e3, 500.0))
fig, ax = plt.subplots(figsize=(5.5, 4.6), layout="constrained")
ax.pcolormesh(gx / 1e3, gy / 1e3, storm_dbz(gx, gy, height_of_beam), cmap="Greys", vmin=0, vmax=55, shading="auto")
ax.plot(fx / 1e3, fy / 1e3, ".", color="tab:gray", ms=6, label="footprint at the ellipsoid")
ax.plot(mx / 1e3, my / 1e3, ".", color="tab:red", ms=6, label="match volume at the beam")
ax.set(xlim=(40, 100), ylim=(-25, 35), xlabel="east of the radar (km)", ylabel="north of the radar (km)",
       aspect="equal", title=f"8 deg sweep, beam at {height_of_beam / 1e3:.1f} km")
ax.legend(loc="upper left", fontsize=7)
```

The choices that Schwaller and Morris (2011) leave open, or that are not checked against the paper,
are radarx's own and are listed in the documentation of `radarx.retrieve.satellite_match`: the
footprint is a circle of `footprint_diameter` (5 km), the DPR bins are those within the half-power
width of the ground radar beam, a mean needs `min_fraction` of the volume to have data, and `weighting`
gives the gates and bins equal or Gaussian weights.

## 3. Dm and Nw: the convention

The DSD parameters of the DPR (`paramDSD`) are $10\log_{10} N_w$ and $D_m$ of the normalized gamma
distribution with the shape $\mu = 3$ fixed in the algorithm (Seto et al. 2021, Eq. 2): $N_w$ in
m$^{-3}$ mm$^{-1}$ and $D_m$ the mass-weighted mean diameter in mm, the $D_m$ form of Testud et al.
(2001). The DSD retrieval of radarx uses the same definitions with $\mu = 3$ as its default, so the two
are comparable as they are. `dpr_dsd` splits the array of the file. The check: the DSD of a
$D_m$ of 1.8 mm and a $10\log_{10} N_w$ of 34 is turned into S-band $Z_H$ and $Z_{DR}$ and the radarx
retrieval gives back the parameters. A ground radar retrieval with another $\mu$ is not comparable.

```{code-cell} ipython3
from math import gamma

dm_dpr, dbnw_dpr, mu = 1.8, 34.0, 3.0
d = np.arange(0.05, 8.0, 0.01)
f_mu = 6.0 / 4.0**4 * (4.0 + mu) ** (mu + 4.0) / gamma(mu + 4.0)
spectrum = xr.DataArray(
    10 ** (0.1 * dbnw_dpr) * f_mu * (d / dm_dpr) ** mu * np.exp(-(4.0 + mu) * d / dm_dpr),
    dims="diameter",
    coords={"diameter": d, "bin_width": ("diameter", np.full(d.size, 0.01))},
)
fields = radar_from_dsd(spectrum, band="S")
radar = xr.Dataset({"DBZH": ("r", [float(fields.DBZH)]), "ZDR": ("r", [float(fields.ZDR)])})
radar["DBZH"].attrs["units"] = "dBZ"
radar["ZDR"].attrs["units"] = "dB"
retrieved = dsd(radar, method="normalized", mu=3.0, band="S")
print(f"S band: ZH {float(fields.DBZH):.1f} dBZ, ZDR {float(fields.ZDR):.2f} dB")
print(f"DPR  Dm {dm_dpr:.2f} mm, 10 log10 Nw {dbnw_dpr:.1f}")
print(f"radarx Dm {float(retrieved.DM[0]):.2f} mm, 10 log10 Nw {10 * np.log10(float(retrieved.NW[0])):.1f}")

# the two parameters of a file in the layout (scan, ray, bin, nDSD) and the matching of a dB field
param = xr.DataArray(
    np.stack([np.full(dpr.height.shape, dbnw_dpr), np.full(dpr.height.shape, dm_dpr)], axis=-1),
    dims=("scan", "ray", "bin", "nDSD"),
)
params = dpr_dsd(param)
dpr["DBNW"] = params["DBNW"]
dpr["DM"] = params["DM"]
dpr["DBNW"].attrs["units"] = "dB"
both = match_dpr(volume, dpr, gr_fields="DBZH", dpr_fields=["DM", "DBNW"], **{k: v for k, v in kwargs.items() if k.endswith("range")})
print(f"matched Dm {float(both.dpr_DM.mean()):.2f} mm and 10 log10 Nw {float(both.dpr_DBNW.mean()):.1f} (averaged in linear units)")
```

## 4. The radar against IMERG

The radar rain rate comes from a grid of 1 km pixels (here the storm at 1 km above the
ground with the Z-R relation $Z = 200 R^{1.6}$, moving east at 40 km/h, one scan every 6 minutes)
and the IMERG field is a smoothed version of it, 20 percent weaker and displaced by 10 km. IMERG gives
the rate of the half hour that starts at its time label. `match_imerg` weights every scan by the
time it represents, averages the pixels of every IMERG cell and masks the cells that the radar
does not fill. The result has the radar and IMERG on the same cells and steps.

```{code-cell} ipython3
x = y = np.arange(-120e3, 120e3 + 1, 1000.0)
crs_attrs = crs.to_cf()
glon, glat = lonlat(crs_attrs, x, y)
times = np.datetime64("2024-05-01T00:03", "ns") + np.arange(10) * np.timedelta64(6, "m")


def rain_rate(x_km, y_km, shift_km):
    xc = 20.0 + shift_km
    dbz = 45.0 * np.exp(-((x_km - xc) ** 2 + (y_km - 5.0) ** 2) / (2 * 14.0**2))
    return np.where(dbz > 10.0, (10 ** (0.1 * dbz) / 200.0) ** (1 / 1.6), 0.0)


XK, YK = np.meshgrid(x / 1e3, y / 1e3)
rate = np.stack([rain_rate(XK, YK, 40.0 / 60.0 * (t - times[0]) / np.timedelta64(1, "m")) for t in times])
radar_rate = xr.DataArray(
    rate, dims=("time", "y", "x"), coords={"time": times, "x": x, "y": y},
    attrs={"units": "mm h-1"}, name="RAIN_RATE",
).assign_coords(crs_wkt=xr.DataArray(0, attrs=crs_attrs), lon=(("y", "x"), glon), lat=(("y", "x"), glat))

# an IMERG-like field: the half hours that start at 00:00 and 00:30, 20 percent weaker, 10 km further east
ilat = np.round(np.arange(28.55, 31.45, 0.1), 2)
ilon = np.round(np.arange(-91.45, -88.55, 0.1), 2)
ilon2, ilat2 = np.meshgrid(ilon, ilat)
ix, iy = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), crs, always_xy=True).transform(ilon2, ilat2)
steps = np.datetime64("2024-05-01T00:00", "ns") + np.arange(2) * np.timedelta64(30, "m")
imerg_values = np.stack([
    0.8 * np.mean([rain_rate(ix / 1e3 - 10.0, iy / 1e3, 40.0 / 60.0 * m) for m in range(15 * s, 15 * s + 30, 5)], axis=0)
    for s in range(2)
])
imerg = xr.DataArray(
    imerg_values.transpose(0, 2, 1), dims=("time", "lon", "lat"),
    coords={"time": steps, "lon": ilon, "lat": ilat}, attrs={"units": "mm/hr"}, name="precipitation",
)
both = match_imerg(radar_rate, imerg)
stats = match_statistics(both.radar, both.imerg, threshold=0.5)
print(both.radar.dims, both.radar.shape)
print(f"{int(stats['count'])} cells: bias {float(stats.bias):.2f} mm/h, ratio {float(stats.ratio):.2f}, "
      f"correlation {float(stats.correlation):.2f}, POD {float(stats.pod):.2f}, FAR {float(stats.far):.2f}")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
for ax, name in zip(axes[:2], ("radar", "imerg")):
    q = both[name].isel(time=0).plot(ax=ax, vmin=0, vmax=12, cmap="viridis", add_colorbar=False)
    ax.set_title(f"{name} 00:00-00:30 (mm/h)")
fig.colorbar(q, ax=axes[:2], shrink=0.8)
axes[2].scatter(both.radar.values.ravel(), both.imerg.values.ravel(), s=6)
axes[2].plot([0, 12], [0, 12], "k-", lw=0.8)
axes[2].set(xlabel="radar (mm/h)", ylabel="IMERG (mm/h)", title="cells")
```

The accumulation of the half hour (`output="accumulation"`) is the rate times 0.5 h, and
`time_label` sets whether the time of an IMERG value is the start (V07 files), the end or the centre of
its step. A cell needs `min_coverage` of its pixels and a pixel `min_time_coverage` of the half hour
to have radar data; fill the rain-free pixels inside the radar coverage with 0 before the call, as NaN
counts as outside the coverage.

## What is not here

The issue also lists the matching of the GPM Microwave Imager (GMI) and overlays of the lightning and
cloud-top fields of GOES on radar grids. They are not part of this notebook.

## References

- Huffman, G. J., et al., 2020: Integrated Multi-satellitE Retrievals for the Global Precipitation Measurement (GPM) mission (IMERG). *Satellite Precipitation Measurement*, Springer, 343-353, <https://doi.org/10.1007/978-3-030-24568-9_19>
- Schwaller, M. R., and K. R. Morris, 2011: A ground validation network for the Global Precipitation Measurement mission. *Journal of Atmospheric and Oceanic Technology*, **28**, 301-319, <https://doi.org/10.1175/2010JTECHA1403.1>
- Seto, S., T. Iguchi, R. Meneghini, J. Awaka, T. Kubota, T. Masaki, and N. Takahashi, 2021: The precipitation rate retrieval algorithms for the GPM Dual-frequency Precipitation Radar. *Journal of the Meteorological Society of Japan*, **99**, 205-237, <https://doi.org/10.2151/jmsj.2021-011>
- Testud, J., S. Oury, R. A. Black, P. Amayenc, and X. Dou, 2001: The concept of "normalized" distribution to describe raindrop spectra: A tool for cloud physics and cloud remote sensing. *Journal of Applied Meteorology*, **40**, 1118-1140, <https://doi.org/10.1175/1520-0450(2001)040<1118:TCONDT>2.0.CO;2>
