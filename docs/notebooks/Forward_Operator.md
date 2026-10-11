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

# Polarimetric forward operator for WRF and MPAS output

A forward operator turns the microphysics state of a numerical model into the
fields a radar would measure, so that a model run can be compared with radar
data in radar space instead of comparing retrieved quantities. This notebook
shows `forward_operator`, `to_radar_grid`, `wrf_state` and `mpas_state`:

- `wrf_state` and `mpas_state` turn a WRF `wrfout` Dataset, or an MPAS history or
  diagnostics Dataset on its unstructured mesh, into one standard state
  (`QRAIN`, `QSNOW`, `QGRAUP`, `QNRAIN`, `TEMPERATURE`, `AIR_DENSITY`, `HEIGHT`,
  `U`, `V`, `W`, `latitude`, `longitude`);
- `forward_operator` computes the reflectivity `DBZH`, the differential
  reflectivity `ZDR`, the specific differential phase `KDP`, the correlation
  coefficient `RHOHV` and the reflectivity-weighted fall speed on the model grid;
- `to_radar_grid` places the gates of a radar sweep in the model (4/3 effective
  Earth radius, `k_e`, with the site altitude in the Earth radius as in
  `radarx.fundamentals.geometry`) and interpolates the fields to them, with the
  radial velocity `VRADH` from the model wind and the fall speed.

Rain uses the T-matrix scattering tables of the DSD retrieval, so it agrees
with `radar_from_dsd` to rounding for the same size distribution. Snow, hail and
graupel are dry particles with Rayleigh scattering by oblate spheroids, with the
shape, orientation and density of the operators of
[Jung et al. (2008)](https://doi.org/10.1175/2007MWR2083.1), written with the
equations of the review of
[Ryzhkov et al. (2020)](https://doi.org/10.3390/atmos11040362) (the
operator of [Ryzhkov et al. 2011](https://doi.org/10.1175/2010JAMC2363.1) is
reproduced there; the 2011 paper itself has no open copy and was not read). A
threaded C++ kernel does the work per cell and per gate, with a NumPy fallback
that gives the same numbers.

What is **not** in the operator: the melting classes of Jung et al. (their paper
gives fitted amplitudes but not the size distribution of the mixture, so a
bright band is missing), attenuation, the beam pattern (a gate is the beam
centre), non-zero elevation in the orientation moments, resonance scattering of
large ice, and the rain size distribution of any double-moment scheme: that is
scheme-specific, was not readable in a source, and is therefore an explicit
assumption (`n0_rain`, `mu_rain`, or the number concentration). No WRF output
exists on the machine that built this page, so the model data below are
synthetic Datasets that follow the variable definitions of the WRF and MPAS
Registries.

```{code-cell} ipython3
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.special import gamma as gamma_fn

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.fundamentals.geometry import beam_center_height
from radarx.retrieve import (
    forward_operator,
    mpas_state,
    radar_from_dsd,
    scattering_table,
    to_radar_grid,
    wrf_state,
)
from radarx.retrieve.forward_operator import rayleigh_amplitudes

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="The input coordinates to pcolormesh")
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})
OKABE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
         "purple": "#CC79A7", "sky": "#56B4E9"}
```

## A WRF-like squall line

The Dataset below has the variables and dimensions of a `wrfout` file: `T`
(perturbation of the potential temperature over 300 K), `P` and `PB`, `PH` and
`PHB` on the staggered levels, the staggered winds `U`, `V`, `W`, `QVAPOR`,
`QRAIN`, `QSNOW`, `QGRAUP`, `QNRAIN` (number per kg, as in the WRF Registry) and
`XLAT`, `XLONG` on a rotated grid with a terrain slope. A convective line with
graupel aloft and heavy rain below sits in the middle, with a trailing
stratiform region to the west: snow above the 0 °C level and light rain below.
With real output, read the file with `xr.open_dataset("wrfout_d03_...")`.

```{code-cell} ipython3
G, KAPPA = 9.81, 287.0 / 1004.5


def standard_atmosphere(z):
    t = np.where(z < 11000.0, 288.15 - 6.5e-3 * z, 216.65)
    p = np.where(z < 11000.0, 101325.0 * (t / 288.15) ** 5.2559,
                 22632.0 * np.exp(-(z - 11000.0) * G / (287.0 * 216.65)))
    return t, p


def synthetic_wrf(nx=96, ny=72, nz=48, dx=3000.0, lat0=35.0, lon0=-97.5):
    """A squall-line-like Dataset with the variables of a wrfout file."""
    zi_flat = 17000.0 * np.linspace(0, 1, nz + 1) ** 1.4
    jj, ii = np.mgrid[0:ny, 0:nx]
    x, y = (ii - nx / 2) * dx, (jj - ny / 2) * dx
    rot = np.deg2rad(7.0)
    east, north = x * np.cos(rot) - y * np.sin(rot), x * np.sin(rot) + y * np.cos(rot)
    lat = lat0 + north / 111195.0
    lon = lon0 + east / (111195.0 * np.cos(np.deg2rad(lat)))
    terrain = 200.0 * ii / nx
    zint = zi_flat[:, None, None] + terrain[None] * (1 - zi_flat[:, None, None] / 17000.0)
    zmass = 0.5 * (zint[:-1] + zint[1:])
    temp, pres = standard_atmosphere(zmass)
    theta = temp * (1.0e5 / pres) ** KAPPA
    s, z = x[None] / 1000.0, zmass / 1000.0
    core = np.exp(-((s - 6.0) / 7.0) ** 2) * (1 - 0.15 * (y[None] / 1e5) ** 2)
    strat = np.exp(-((s + 45.0) / 40.0) ** 2) * (s < 0)
    warm = temp > 273.15
    qrain = np.where(warm, 5.0e-3 * core * np.exp(-z / 3.0) + 0.6e-3 * strat * (z < 3.8), 0.0)
    qgraup = np.where(~warm | (z < 5.5), 3.0e-3 * core * np.exp(-((z - 5.0) / 2.5) ** 2), 0.0)
    qsnow = np.where(~warm, 0.5e-3 * (strat + 0.4 * core) * np.exp(-((z - 7.0) / 3) ** 2), 0.0)
    dv = np.where(core > 0.2, 1.6e-3, 0.9e-3)  # mean volume diameter of the rain (m)
    qnrain = np.where(qrain > 0, 6.0 * qrain / (np.pi * 1000.0 * dv**3), 0.0)  # # kg-1
    qv = 0.01 * np.exp(-z / 2.5) * np.ones_like(z)
    wwind = 6.0 * core * np.sin(np.pi * np.clip(z / 12.0, 0, 1))

    def stag(a, axis):
        a = np.moveaxis(a, axis, 0)
        out = np.concatenate([a[:1] - 0.5 * (a[1:2] - a[:1]), 0.5 * (a[:-1] + a[1:]),
                              a[-1:] + 0.5 * (a[-1:] - a[-2:-1])])
        return np.moveaxis(out, 0, axis)

    d3 = ("Time", "bottom_top", "south_north", "west_east")
    d3z = ("Time", "bottom_top_stag", "south_north", "west_east")
    t1 = lambda a: a[None]  # noqa: E731
    return xr.Dataset({
        "T": (d3, t1(theta - 300.0)), "P": (d3, t1(np.zeros_like(pres))), "PB": (d3, t1(pres)),
        "PH": (d3z, t1(np.zeros_like(zint))), "PHB": (d3z, t1(G * zint)),
        "QVAPOR": (d3, t1(qv)), "QRAIN": (d3, t1(qrain)), "QSNOW": (d3, t1(qsnow)),
        "QGRAUP": (d3, t1(qgraup)), "QNRAIN": (d3, t1(qnrain)),
        "U": (("Time", "bottom_top", "south_north", "west_east_stag"), t1(stag(14.0 + 1.5 * z + 0 * s, 2))),
        "V": (("Time", "bottom_top", "south_north_stag", "west_east"), t1(stag(3.0 + 0.2 * z + 0 * s, 1))),
        "W": (d3z, t1(stag(wwind, 0))),
        "XLAT": (("Time", "south_north", "west_east"), t1(lat)),
        "XLONG": (("Time", "south_north", "west_east"), t1(lon)),
    }, attrs={"USE_THETA_M": 0})


wrfout = synthetic_wrf()
state = wrf_state(wrfout)
print(state)
```

`wrf_state` averages the staggered winds and the geopotential to the mass points,
rebuilds the temperature from `T`, `P` and `PB` (with the moist potential
temperature if the file has `USE_THETA_M = 1`) and the dry-air density from the
pressure and the temperature. The constants are those of WRF's
`module_model_constants.F`. The input Dataset is not changed.

## The operator on the model grid

The rain size distribution is not predicted by the scheme in this
single-moment-style example, so the rain intercept is an assumption. Here the
rain number concentration `QNRAIN` closes a gamma distribution of shape 2, and
`QGRAUP` is run as hail with the dry-hail values of Jung et al. (2008) (density
913 kg m$^{-3}$, axis ratio 0.75, 60° canting) to show the choice: nothing in
the model says that graupel is hail.

```{code-cell} ipython3
fields = forward_operator(state, nrain="QNRAIN", mu_rain=2.0, hail="QGRAUP")
print(fields.attrs["forward_operator_classes"], "| band", fields.attrs["forward_operator_band"])
print(f"DBZH {float(fields.DBZH.max()):.1f} dBZ at most, ZDR {float(fields.ZDR.min()):.2f} to "
      f"{float(fields.ZDR.max()):.2f} dB, KDP up to {float(fields.KDP.max()):.2f} deg/km, "
      f"RHOHV from {float(fields.RHOHV.min()):.3f}")
print(fields.attrs["radarx_method"])
print(fields.attrs["radarx_references"])
```

```{code-cell} ipython3
j = 36  # a west-east cross section through the line
xs = (np.arange(fields.sizes["west_east"]) - fields.sizes["west_east"] / 2) * 3.0
zc = state.HEIGHT.isel(Time=0, south_north=j).values / 1000.0
temp = state.TEMPERATURE.isel(Time=0, south_north=j).values
fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), layout="constrained", sharex=True, sharey=True)
panels = [("DBZH", "dBZ", -10, 65, "turbo"), ("ZDR", "dB", -0.5, 3.0, "viridis"),
          ("KDP", "deg/km", 0, 4, "viridis"), ("RHOHV", "", 0.97, 1.0, "viridis")]
X = np.broadcast_to(xs, zc.shape)
for ax, (name, unit, lo, hi, cmap) in zip(axes.ravel(), panels):
    pm = ax.pcolormesh(X, zc, fields[name].isel(Time=0, south_north=j).values, vmin=lo, vmax=hi,
                       cmap=cmap, shading="nearest")
    ax.contour(X, zc, temp, levels=[273.15], colors="k", linewidths=1.0)
    fig.colorbar(pm, ax=ax, label=unit)
    ax.set(title=name, ylim=(0, 15))
for ax in axes[1]:
    ax.set_xlabel("west-east distance (km)")
for ax in axes[:, 0]:
    ax.set_ylabel("height (km)")
plt.show()
```

The reflectivity has the convective core and the trailing stratiform region. In
the core the graupel (run as hail) lowers ZDR and RHOHV where it dominates, and
the rain below it has the high ZDR and KDP of large drops. The line is the 0 °C
level: with no melting class, snow stops at that level and rain starts there, so
the bright band of a real stratiform region is missing.

## Radar space

A radar 90 km southwest of the line scans at 0.5° and 3.0°. `to_radar_grid`
interpolates the linear moments (not dBZ) to the beam centre and forms
`DBZH`, `ZDR`, `RHOHV` after the sampling. The radial velocity is
$(u\sin\phi + v\cos\phi)\cos\theta + (w - v_t)\sin\theta$, with $v_t$ the
reflectivity-weighted fall speed. Gates outside the model (horizontally or above
its top) and gates with no echo are NaN.

```{code-cell} ipython3
site = (34.7, -98.4, 380.0)  # latitude, longitude, altitude (m)
sweeps = to_radar_grid(fields, site, [0.5, 3.0])
sweep = sweeps["sweep_0"].to_dataset()
print(sweeps)
```

```{code-cell} ipython3
fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), subplot_kw={"projection": "polar"}, layout="constrained")
for row, node in zip(axes, ("sweep_0", "sweep_1")):
    sw = sweeps[node].to_dataset()
    for ax, (name, unit, lo, hi, cmap) in zip(row, [("DBZH", "dBZ", 0, 65, "turbo"),
            ("ZDR", "dB", -0.5, 3.0, "viridis"), ("KDP", "deg/km", 0, 3, "viridis"),
            ("VRADH", "m/s", -25, 25, "RdBu_r")]):
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        pm = ax.pcolormesh(np.deg2rad(sw.azimuth), sw.range / 1000.0, sw[name].isel(Time=0).T,
                           vmin=lo, vmax=hi, cmap=cmap)
        fig.colorbar(pm, ax=ax, shrink=0.65, label=unit)
        ax.set_title(f"{name}, {float(sw.elevation[0]):.1f}°")
        ax.set_rlim(0, 150)
plt.show()
```

The 3° sweep sees the high part of the storm (graupel and snow) beyond about
30 km, where its beam is above the 0 °C level at 2.3 km; the 0.5° sweep sees the
rain out to 150 km. The beam height follows the 4/3 Earth radius
model with the site altitude in the Earth radius:

```{code-cell} ipython3
rng = np.array([20e3, 50e3, 100e3, 150e3])
for k_e in (4 / 3, 1.0):
    heights = beam_center_height(rng, 0.5, radar_height=site[2], reff=k_e * 6371000.0)
    print(f"k_e = {k_e:.2f}: beam centre at 0.5 deg", np.round(heights).astype(int), "m for ranges 20, 50, 100, 150 km")
by_acc = fields.radarx.to_radar_grid(site, 0.5, k_e=1.0)  # the accessor, with another k_e
print(f"mean DBZH at 0.5 deg: {float(sweep.DBZH.mean()):.1f} (4/3) against {float(by_acc.DBZH.mean()):.1f} (k_e = 1)")
```

## Checks against closed forms and the DSD forward model

**Rayleigh limit of exponential rain.** For small drops the reflectivity is
$\int D^6 N\,dD = 720\,N_0/\Lambda^7$ and ZDR tends to zero.

```{code-cell} ipython3
MASS = np.pi / 6.0 * 1.0e-6  # kg m-3 per mm3 m-3 of water


def rain_state(q, rho=1.1, temp=283.15):
    q = np.atleast_1d(np.asarray(q, dtype=float))
    return xr.Dataset({"QRAIN": ("x", q), "AIR_DENSITY": ("x", np.full(q.shape, rho)),
                       "TEMPERATURE": ("x", np.full(q.shape, temp))})


n0, lam = 8.0e6, 16.0  # m-4 and mm-1: drops below 0.6 mm
q = (8.0e3 * 6.0 / lam**4) * MASS / 1.1
out = forward_operator(rain_state([q]))
z_closed = 720.0 * (n0 * 1e-3) / lam**7
print(f"Z closed form {10 * np.log10(z_closed):.2f} dBZ, operator {float(out.DBZH[0]):.2f} dBZ, "
      f"ZDR {float(out.ZDR[0]):.3f} dB, RHOHV {float(out.RHOHV[0]):.4f}")
```

**Agreement with `radar_from_dsd`** on the same gamma size distribution, built
on the table diameters, at the three bands (the maximum differences over rain
contents from 0.02 to 8 g kg$^{-1}$):

```{code-cell} ipython3
q = np.array([2e-5, 2e-4, 1e-3, 4e-3, 8e-3])
for band in "SCX":
    tab = scattering_table(band, 10.0)
    d = tab.diameter.values
    w = np.zeros_like(d)
    w[:-1] += 0.5 * np.diff(d)
    w[1:] += 0.5 * np.diff(d)
    lam_q = (8.0e3 * 6.0 / (q * 1.1 / MASS)) ** 0.25
    spectrum = xr.DataArray(8.0e3 * np.exp(-lam_q[:, None] * d[None, :]), dims=("x", "diameter"),
                            coords={"diameter": d, "bin_width": ("diameter", w)})
    ref = radar_from_dsd(spectrum, band, 10.0)
    op = forward_operator(rain_state(q), band=band, temperature=None, rain_temperature=10.0)
    print(f"{band} band: max |DBZH| {float(abs(op.DBZH - ref.DBZH).max()):.1e} dB, "
          f"|ZDR| {float(abs(op.ZDR - ref.ZDR).max()):.1e} dB, "
          f"|KDP| {float(abs(op.KDP - ref.KDP).max()):.1e} deg/km, |RHOHV| {float(abs(op.RHOHV - ref.RHOHV).max()):.1e}")
```

**Invariants of rain:** the reflectivity increases with the rain content, ZDR stays
between -0.5 and 5 dB, KDP is not negative and RHOHV is at most 1.

```{code-cell} ipython3
qs = np.geomspace(5e-6, 2e-2, 200)
rain = forward_operator(rain_state(qs), min_dbz=-80.0)
print("DBZH increasing:", bool(np.all(np.diff(rain.DBZH.values) > 0)),
      "| ZDR range", np.round([float(rain.ZDR.min()), float(rain.ZDR.max())], 2),
      "| KDP min", float(rain.KDP.min()), "| RHOHV max", float(rain.RHOHV.max()))
```

**Ice amplitudes and closed forms of Jung et al. (2008).** The paper gives the
backscattering amplitudes of dry snow (100 kg m$^{-3}$) and dry hail
(913 kg m$^{-3}$) as numbers fitted to Rayleigh scattering at S band (its Eq. 9).
radarx computes them from the Rayleigh formulas of the 2020 review (Eqs. 3-5) with
the Maxwell Garnett mixing of ice in air and a dielectric constant of ice of
3.17, so that one code path serves every band and density. The table compares
them (amplitudes in mm per mm$^3$, along the transverse and the symmetry axis):

```{code-cell} ipython3
WL_S = 107.0687  # mm, the S-band wavelength of the scattering tables
jung = {"dry snow": (100.0, 0.194e-4, 0.191e-4), "dry hail": (913.0, 0.191e-3, 0.165e-3)}
print(f"{'class':>9} {'axis':>11} {'Jung et al.':>12} {'radarx':>12} {'difference':>11}")
for name, (rho, a_j, b_j) in jung.items():
    a_sym, a_tr = rayleigh_amplitudes(rho, 0.75, WL_S)
    for axis, jv, mine in (("transverse", a_j, a_tr), ("symmetry", b_j, a_sym)):
        print(f"{name:>9} {axis:>11} {jv:12.4e} {mine:12.4e} {100 * (mine / jv - 1):+10.2f}%")
```

With these, Eqs. 12 and 13 of the paper give $Z_H$ and $Z_V$ in closed form; the
operator is compared with them for ice alone:

```{code-cell} ipython3
def jung_closed_form(rho_i, n0, sigma, a, b, q, rho_air):
    s = np.deg2rad(sigma)
    e1, e2 = np.exp(-2 * s * s), np.exp(-8 * s * s)
    A, B, C = (3 + 4 * e1 + e2) / 8, (3 - 4 * e1 + e2) / 8, (1 - e2) / 8
    lam = (np.pi * rho_i * n0 / (rho_air * q)) ** 0.25 * 1e-3
    pref = 2880.0 * WL_S**4 * n0 * 1e-3 / (np.pi**4 * 0.93) * lam**-7
    zh = pref * (A * a**2 + B * b**2 + 2 * C * a * b)
    zv = pref * (B * a**2 + A * b**2 + 2 * C * a * b)
    return 10 * np.log10(zh), 10 * np.log10(zh / zv)


q = np.array([2e-5, 1.5e-4, 1e-3])
ice = xr.Dataset({"Q": ("x", q), "AIR_DENSITY": ("x", np.full(3, 0.8))})
for name, kw, args in (("snow", {"snow": "Q"}, (100.0, 3e6, 20.0, 0.194e-4, 0.191e-4)),
                       ("hail", {"snow": None, "hail": "Q"}, (913.0, 4e4, 60.0, 0.191e-3, 0.165e-3))):
    op = forward_operator(ice, rain=None, **kw)
    dbz, zdr = jung_closed_form(*args, q, 0.8)
    print(f"{name}: DBZH operator - closed form {np.round(op.DBZH.values - dbz, 3)} dB, "
          f"ZDR {np.round(op.ZDR.values - zdr, 3)} dB")
```

The snow is 0.22 dB higher in reflectivity and the hail 0.04 dB lower than the
closed forms; these are the 2.6 % and 0.4 % of the amplitudes above, squared.

**The compiled kernel and NumPy** give the same numbers; the kernel is faster
and threaded:

```{code-cell} ipython3
big = wrf_state(synthetic_wrf(nx=160, ny=120, nz=60))
timings = {}
for engine in ("numpy", "compiled"):
    try:
        t0 = time.perf_counter()
        f_big = forward_operator(big, nrain="QNRAIN", hail="QGRAUP", engine=engine)
        t1 = time.perf_counter()
        g_big = to_radar_grid(f_big, site, 0.5, engine=engine)
        t2 = time.perf_counter()
        timings[engine] = (t1 - t0, t2 - t1, f_big, g_big)
    except ImportError:
        pass
n_cells = big.QRAIN.size
for engine, (t_cells, t_gates, _, _) in timings.items():
    print(f"{engine:>8}: {n_cells / 1e6:.2f} M cells in {t_cells:.2f} s, "
          f"{timings[engine][3].DBZH.size / 1e3:.0f} k gates in {t_gates:.2f} s")
if len(timings) == 2:
    a, b = timings["numpy"], timings["compiled"]
    print("max difference DBZH on the model grid:", float(np.nanmax(abs(a[2].DBZH - b[2].DBZH))),
          "| on the radar grid:", float(np.nanmax(abs(a[3].DBZH - b[3].DBZH))))
```

## The assumptions about the rain size distribution

The scheme decides the rain size distribution and the model output often only
holds the mixing ratio, so the operator makes you name the assumption. The same
rain content gives different ZDR and KDP for different intercepts, shapes or a
number concentration; compare them on a ZDR-reflectivity diagram, the diagram a
radar user compares against.

```{code-cell} ipython3
qs = np.geomspace(2e-5, 1.2e-2, 60)
base = rain_state(qs)
cases = {
    "N0 = 8e6 m-4 (default), mu = 0": dict(),
    "N0 = 2e6 m-4, mu = 0": dict(n0_rain=2.0e6),
    "N0 = 3e7 m-4, mu = 0": dict(n0_rain=3.0e7),
    "N0 = 8e6 m-4, mu = 3": dict(mu_rain=3.0),
}
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
for (label, kw), color in zip(cases.items(), OKABE.values()):
    op = forward_operator(base, **kw)
    axes[0].plot(op.DBZH, op.ZDR, color=color, label=label)
    axes[1].plot(op.DBZH, op.KDP, color=color)
# a number concentration of 800 per kg-ish per metre-cubed closes the size distribution instead of N0
nt = 1.0e3 / 1.1  # 1000 m-3 in # kg-1
op = forward_operator(base.assign(QNRAIN=("x", np.full(qs.shape, nt))), nrain="QNRAIN", mu_rain=2.0)
axes[0].plot(op.DBZH, op.ZDR, color="k", ls="--", label="N_t = 1000 m-3, mu = 2")
axes[1].plot(op.DBZH, op.KDP, color="k", ls="--")
axes[0].set(xlabel="DBZH (dBZ)", ylabel="ZDR (dB)", xlim=(0, 60))
axes[1].set(xlabel="DBZH (dBZ)", ylabel="KDP (deg/km)", xlim=(20, 60), ylim=(0, 6))
axes[0].legend(fontsize=8)
plt.show()
```

## An MPAS-like unstructured mesh

MPAS history and diagnostics files hold the fields on `nCells` with `lonCell` and
`latCell` in radians and the layer interfaces `zgrid` (`nCells` by
`nVertLevelsP1`), which are often only in the static grid file. `mpas_state`
takes the output and the grid Dataset. The Dataset here is 1500 cells on a disc
of 400 km (the variable names and units are those of the MPAS-Atmosphere
Registry: `qr`, `qs`, `nr` in "nb kg-1", `rho` is the dry air density,
`uReconstructZonal`, `uReconstructMeridional`, `w` on the interfaces). The
horizontal neighbours of a gate come from a KD-tree on the sphere, so the same
call works on the unstructured mesh.

```{code-cell} ipython3
rng_ = np.random.default_rng(4)
ncell, nz = 1500, 30
dist_c = np.sqrt(rng_.uniform(0, 1, ncell)) * 400e3
bearing = rng_.uniform(0, 2 * np.pi, ncell)
lat_c = 35.0 + np.rad2deg(dist_c * np.cos(bearing) / 6371000.0)
lon_c = -97.5 + np.rad2deg(dist_c * np.sin(bearing) / (6371000.0 * np.cos(np.deg2rad(35.0))))
zi = 17000.0 * np.linspace(0, 1, nz + 1) ** 1.4
zgrid = np.repeat(zi[None], ncell, axis=0)
zm = 0.5 * (zgrid[:, :-1] + zgrid[:, 1:])
temp_m, pres_m = standard_atmosphere(zm)
exner = (pres_m / 1e5) ** KAPPA
shaft = np.exp(-(np.hypot(dist_c * np.cos(bearing) - 60e3, dist_c * np.sin(bearing) + 30e3) / 40e3) ** 2)[:, None]
qr = np.where(temp_m > 273.15, 3e-3 * shaft * np.exp(-zm / 3000.0), 0.0)
qs_ = np.where(temp_m <= 273.15, 0.6e-3 * shaft * np.exp(-((zm - 7000.0) / 3000.0) ** 2), 0.0)
dims = ("Time", "nCells", "nVertLevels")
mpas = xr.Dataset({
    "qv": (dims, (0.008 * np.exp(-zm / 2500.0))[None]), "qr": (dims, qr[None]), "qs": (dims, qs_[None]),
    "nr": (dims, np.where(qr > 0, 6.0 * qr / (np.pi * 1000.0 * 1.2e-3**3), 0.0)[None]),
    "rho": (dims, (pres_m / (287.0 * temp_m))[None]), "theta": (dims, (temp_m / exner)[None]),
    "exner": (dims, exner[None]), "pressure": (dims, pres_m[None]),
    "uReconstructZonal": (dims, np.full((1, ncell, nz), 10.0)),
    "uReconstructMeridional": (dims, np.full((1, ncell, nz), -5.0)),
    "w": (("Time", "nCells", "nVertLevelsP1"), np.full((1, ncell, nz + 1), 1.0)),
})
mpas_grid = xr.Dataset({"lonCell": ("nCells", np.deg2rad(lon_c)), "latCell": ("nCells", np.deg2rad(lat_c)),
                        "zgrid": (("nCells", "nVertLevelsP1"), zgrid)})
mstate = mpas_state(mpas, mpas_grid)
mfields = forward_operator(mstate, nrain="QNRAIN", mu_rain=1.0)
msweep = to_radar_grid(mfields, (35.0, -97.5, 300.0), 0.5, azimuth=np.arange(0.5, 360.0, 1.0),
                       gate_range=np.arange(500.0, 200001.0, 500.0))
print(mfields.DBZH.dims, f"| maximum {float(mfields.DBZH.max()):.1f} dBZ")
```

```{code-cell} ipython3
fig = plt.figure(figsize=(11, 4.6), layout="constrained")
ax = fig.add_subplot(1, 2, 1)
col_max = mfields.DBZH.isel(Time=0).max("nVertLevels")
sc = ax.scatter(mstate.longitude, mstate.latitude, c=col_max, s=9, cmap="turbo", vmin=0, vmax=50)
ax.set(xlabel="longitude", ylabel="latitude", title="column maximum DBZH on the mesh cells")
fig.colorbar(sc, ax=ax, label="dBZ")
ax = fig.add_subplot(1, 2, 2, projection="polar")
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
pm = ax.pcolormesh(np.deg2rad(msweep.azimuth), msweep.range / 1000.0, msweep.DBZH.isel(Time=0).T,
                   vmin=0, vmax=50, cmap="turbo")
fig.colorbar(pm, ax=ax, label="dBZ", shrink=0.8)
ax.set_title("DBZH at 0.5 degrees")
plt.show()
```

## Using real output

```python
import xarray as xr
from radarx.retrieve import forward_operator, to_radar_grid, wrf_state

state = wrf_state(xr.open_dataset("wrfout_d03_2022-03-30_12:00:00"))
fields = forward_operator(state, nrain="QNRAIN", mu_rain=2.0)   # name the rain assumption
site = (float(radar.latitude), float(radar.longitude), float(radar.altitude))  # the radar volume
model_volume = to_radar_grid(fields, site, [0.5, 1.5, 2.5])  # a DataTree with one sweep per angle
```

The result has the names of an xradar sweep (`DBZH`, `ZDR`, `KDP`, `RHOHV`,
`VRADH` on `azimuth` and `range`), so it goes into the same plots and
verification code as the radar data. Compare it field by field, or give the
model gates and the radar gates to `radarx.retrieve.dsd` or `hid` for a
comparison of the retrieved quantities.

## References

- Jung, Y., G. Zhang, and M. Xue, 2008: Assimilation of simulated polarimetric radar data for a convective storm using the ensemble Kalman filter. Part I: Observation operators for reflectivity and polarimetric variables. *Monthly Weather Review*, **136** (6), 2228-2245, <https://doi.org/10.1175/2007MWR2083.1>
- Ryzhkov, A. V., J. Snyder, J. T. Carlin, A. Khain, and M. Pinsky, 2020: What polarimetric weather radars offer to cloud modelers: Forward radar operators and microphysical/thermodynamic retrievals. *Atmosphere*, **11** (4), 362, <https://doi.org/10.3390/atmos11040362>
- Ryzhkov, A., M. Pinsky, A. Pokrovsky, and A. Khain, 2011: Polarimetric radar observation operator for a cloud model with spectral microphysics. *Journal of Applied Meteorology and Climatology*, **50** (4), 873-894, <https://doi.org/10.1175/2010JAMC2363.1> (not read)
- Maxwell Garnett, J. C., 1904: Colours in metal glasses and in metallic films. *Philosophical Transactions of the Royal Society of London A*, **203**, 385-420, <https://doi.org/10.1098/rsta.1904.0024>
- Doviak, R. J., and D. S. Zrnic, 1993: *Doppler Radar and Weather Observations*, 2nd ed. Academic Press.
