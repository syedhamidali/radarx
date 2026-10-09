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
---

# Rain Size Sorting and Raindrop Trajectories

Raindrops of different size fall at different speeds. In a sheared, evolving
wind they therefore leave a radar gate together but reach the ground apart in
space and time: large drops fall fast and arrive early, close to the point below
the gate, small drops fall slowly, are blown further by the wind aloft and
arrive late, if they survive the fall at all. This size sorting changes the drop
size distribution (DSD) between the radar beam and the surface (Kumjian and
Ryzhkov 2012; Dawson et al. 2015).

`radarx.retrieve.rain_trajectories` follows one drop per source point and size
bin from a radar gate, or any point of a 3-D grid, to the ground, with

- the terminal speed of Atlas et al. (1973), increased aloft by
  $(\rho_0/\rho)^{0.4}$ (Foote and du Toit 1969);
- 3-D, time-varying winds (a multi-Doppler analysis or a sounding) in a frame
  that moves with the storm;
- evaporation along the path with the single-drop law of
  `radarx.retrieve.drop_evaporation_rate` (the drops shrink and fall slower);
- optional turbulent dispersion (an ensemble of drops with random velocity
  perturbations).

One compiled, multithreaded kernel integrates all drops (source points $\times$
sizes $\times$ ensemble members) at once. Collisions and break-up are not
modelled. This notebook uses a synthetic environment only: a sheared wind
profile, a humid sounding and a moving rain cell.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.retrieve import (
    rain_source_points,
    rain_trajectories,
    size_sorting,
    surface_dsd,
    terminal_fall_speed,
    trajectory_matched_times,
)

plt.rcParams.update(
    {
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "font.size": 9.5,
    }
)
OKABE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00"}
```

## The environment

A wind that turns and strengthens with height (a shear of about 20 m s$^{-1}$
over 3 km), air that is dry near the ground and nearly saturated aloft, and a storm
that moves at 7 m s$^{-1}$ toward the east-northeast. Heights are above sea
level; the surface is at 0.

```{code-cell} ipython3
def environment(z, u, v):
    """A sounding-like profile: standard lapse rate, humidity rising with height."""
    temperature = 297.0 - 6.5e-3 * z
    return xr.Dataset(
        {
            "temperature": ("height", temperature),
            "pressure": ("height", 100500.0 * (temperature / 297.0) ** 5.2559),
            "relative_humidity": ("height", 0.97 - 0.42 * np.exp(-z / 1500.0)),
            "u": ("height", u),
            "v": ("height", v),
        },
        coords={"height": z},
    )


z = np.arange(0.0, 6001.0, 50.0)
profile = environment(
    z,
    2.0 + 24.0 * (1.0 - np.exp(-z / 2500.0)),
    1.0 + 10.0 * (1.0 - np.exp(-z / 2500.0)),
)
motion = (6.0, 3.0)  # storm motion (u, v), m s-1

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.4), constrained_layout=True)
ax1.plot(profile.u, z / 1e3, color=OKABE["blue"], label="$u$ (east)")
ax1.plot(profile.v, z / 1e3, color=OKABE["orange"], label="$v$ (north)")
ax1.plot(profile.u - motion[0], z / 1e3, "--", color=OKABE["blue"], lw=1)
ax1.plot(profile.v - motion[1], z / 1e3, "--", color=OKABE["orange"], lw=1)
ax1.axvline(0, color="0.4", lw=0.8)
ax1.set_ylim(0, 4)
ax1.set_xlabel("wind (m s$^{-1}$), dashed: storm-relative")
ax1.set_ylabel("height (km)")
ax1.legend(loc="upper left", frameon=False)
ax2.plot(100 * profile.relative_humidity, z / 1e3, color=OKABE["green"])
ax2.set_ylim(0, 4)
ax2.set_xlabel("relative humidity (%)")
ax2.set_ylabel("height (km)");
```

## Where do the drops go?

Drops of 0.5 to 6 mm start at 3 km above a point and are followed to the
ground. `size_sorting` expresses the end point of each size relative to the
2 mm drops of the same start: the displacement along the storm-relative
direction of the 2 mm drop and the difference in arrival time.

```{code-cell} ipython3
diameters = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
column = {"x": 0.0, "y": 0.0, "z": 3000.0}
traj = rain_trajectories(
    column, diameters, profile=profile, storm_motion=motion, store_path=2
).squeeze()
sort = size_sorting(traj, 2.0).squeeze()
print(
    "size (mm) | fall time (s) | landing diameter (mm) | mass evaporated | status"
)
for d, t, dl, f, s in zip(
    traj.diameter.values,
    traj.fall_time.values,
    traj.landing_diameter.values,
    traj.evaporated_mass_fraction.values,
    traj.status.values,
):
    state = {1: "reaches the ground", 2: "evaporates"}.get(int(s), "aloft")
    print(f"{d:9.1f} | {t:13.0f} | {dl:21.2f} | {100 * f:14.1f} % | {state}")
```

The small drops evaporate completely in the dry air near the ground, the larger
ones reach it a little smaller. In the storm-relative frame (the horizontal
position minus the distance the storm moved) the paths show the sorting:

```{code-cell} ipython3
colors = plt.cm.viridis(np.linspace(0.0, 0.92, diameters.size))
# storm-relative horizontal displacement along the storm-relative wind at 3 km
rel = np.array([float(profile.u.sel(height=3000.0)) - motion[0],
                float(profile.v.sel(height=3000.0)) - motion[1]])
axis = rel / np.linalg.norm(rel)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 3.7), constrained_layout=True)
for k, d in enumerate(diameters):
    p = traj.isel(diameter=k)
    x_rel = (p.path_x - motion[0] * p.path_time) * axis[0]
    x_rel = x_rel + (p.path_y - motion[1] * p.path_time) * axis[1]
    ax1.plot(x_rel / 1e3, p.path_z / 1e3, color=colors[k], label=f"{d:g} mm")
ax1.set_xlabel("storm-relative displacement (km)")
ax1.set_ylabel("height (km)")
ax1.set_title("paths in the storm frame")
ax1.legend(fontsize=7.5, ncol=2, frameon=False, loc="upper right")

ax2.plot(sort.diameter, sort.arrival_offset, "o-", color=OKABE["blue"], ms=4)
ax2.axhline(0, color="0.4", lw=0.8)
ax2.set_xlabel("drop diameter (mm)")
ax2.set_ylabel("arrival time relative to 2 mm (s)")
ax2.set_title("small drops arrive later")

ax3.bar(np.arange(diameters.size), 100 * traj.evaporated_mass_fraction, color=colors, width=0.7)
ax3.set_xticks(np.arange(diameters.size), [f"{d:g}" for d in diameters])
ax3.set_xlabel("drop diameter at 3 km (mm)")
ax3.set_ylabel("mass lost by evaporation (%)")
ax3.set_title("evaporation below 3 km");
```

## A rain cell: the DSD at the ground

A cell moves with the storm. Its DSD is a normalised gamma distribution
($\mu = 3$) with a mass-weighted mean diameter $D_m$ of 3 mm in the core and
1.8 mm at the edge, observed at 2.5 km every minute on a 2-km grid, as a
radar volume would. `rain_trajectories` takes the grid itself as the source
(any `x`, `y`, `z` coordinates would do, for example the gates of a sweep) and
`surface_dsd` collects the drops that land onto a surface grid, conserving the
number flux, so the concentration at the ground follows from the one aloft by
counting.

```{code-cell} ipython3
x = np.arange(-24e3, 24.1e3, 2e3)
times = np.arange(0.0, 1801.0, 60.0)  # s
bins = np.arange(0.5, 6.01, 0.25)
width = 0.25


def gamma_nd(dm, nw, mu=3.0):
    """Normalised gamma N(D) (m-3 mm-1), D in mm."""
    from scipy.special import gammaln

    f = 6.0 / 4.0**4 * np.exp((mu + 4.0) * np.log(4.0 + mu) - gammaln(mu + 4.0))
    s = bins / dm[..., None]
    return nw[..., None] * f * s**mu * np.exp(-(4.0 + mu) * s)


t_, y_, x_ = np.meshgrid(times, x, x, indexing="ij")
cx = -12e3 + motion[0] * t_  # the cell centre moves with the storm
cy = -9e3 + motion[1] * t_
r2 = ((x_ - cx) ** 2 + (y_ - cy) ** 2) / 5e3**2
core = np.exp(-r2)
dm = 1.8 + 1.2 * core
nw = 8000.0 * core
nd_src = xr.DataArray(
    gamma_nd(dm, nw),
    dims=("time", "y", "x", "diameter"),
    coords={"time": times, "y": x, "x": x, "diameter": bins},
    name="ND",
)
nd_src.attrs["units"] = "m-3 mm-1"


def moments(nd, d=bins, w=width):
    m3 = (nd * d**3 * w).sum("diameter")
    m4 = (nd * d**4 * w).sum("diameter")
    return (m4 / m3).where(m3 > 0), (nd * w).sum("diameter")


dm_aloft, nt_aloft = moments(nd_src)
```

```{code-cell} ipython3
src_pts = xr.Dataset(
    {"z": ((), 2500.0)},
    coords={"time": times, "y": x, "x": x},
)
cell = rain_trajectories(
    src_pts, bins, profile=profile, storm_motion=motion, time_step=10.0, max_time=1500.0
)
surface = surface_dsd(
    cell,
    nd_src,
    x=np.arange(-30e3, 45.1e3, 2e3),
    y=np.arange(-30e3, 45.1e3, 2e3),
    time=np.arange(0.0, 2700.0, 60.0),
    duration=60.0,
)
print(
    f"{float((cell.status == 1).mean()) * 100:.0f} % of the drops reach the ground, "
    f"{float((cell.status == 2).mean()) * 100:.0f} % evaporate"
)
```

At the surface the cell has moved on. The drops of different size no longer
coincide: the right panel shows the half-maximum contours of the concentration
of small (1-1.75 mm) and large (4-6 mm) drops. The large drops, which fall
fastest, stay closer to the core aloft; the small ones are blown further and
are partly lost by evaporation, which raises $D_m$ at the edge of the cell:

```{code-cell} ipython3
t_air, t_sfc = 600.0, 960.0  # s
sfc_dm = surface.DM.sel(time=t_sfc)
aloft = dm_aloft.sel(time=t_air).where(nt_aloft.sel(time=t_air) > 1.0)
fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), constrained_layout=True)
kw = dict(x="x", y="y", add_colorbar=False, vmin=1.8, vmax=3.0, cmap="viridis")
im = (aloft.assign_coords(x=aloft.x / 1e3, y=aloft.y / 1e3)).plot(ax=axes[0], **kw)
(sfc_dm.where(surface.NT.sel(time=t_sfc) > 1.0).assign_coords(
    x=surface.x / 1e3, y=surface.y / 1e3)).plot(ax=axes[1], **kw)
fig.colorbar(im, ax=axes[:2], location="bottom", shrink=0.6, label="$D_m$ (mm)")

# where the small and the large drops are at the ground
small = (surface.ND * width).sel(diameter=slice(1.0, 1.75)).sum("diameter").sel(time=t_sfc)
large = (surface.ND * width).sel(diameter=slice(4.0, 6.0)).sum("diameter").sel(time=t_sfc)
for g, c in ((small, OKABE["orange"]), (large, OKABE["blue"])):
    g = g.assign_coords(x=g.x / 1e3, y=g.y / 1e3)
    axes[2].contour(g.x, g.y, g / g.max(), levels=[0.5], colors=[c], linewidths=1.8)
axes[2].plot([], [], color=OKABE["orange"], label="1-1.75 mm")
axes[2].plot([], [], color=OKABE["blue"], label="4-6 mm")
peak = large.argmax(dim=("y", "x"))
xm, ym = float(large.x[peak["x"]]), float(large.y[peak["y"]])
axes[2].set_xlim(xm / 1e3 - 6, xm / 1e3 + 6)
axes[2].set_ylim(ym / 1e3 - 10, ym / 1e3 + 6)
axes[2].legend(title="half-maximum of the\nconcentration of drops of", loc="lower right",
               fontsize=8, frameon=False)
axes[0].set_title("$D_m$ at 2.5 km, t = 10 min")
axes[1].set_title("$D_m$ at the ground, t = 16 min")
axes[2].set_title("small and large drops at the ground")
for ax in axes:
    ax.set_xlabel("x (km)")
    ax.set_aspect("equal")
for ax in axes[:2]:
    ax.set_xlim(-24, 24)
    ax.set_ylim(-24, 24)
axes[0].set_ylabel("y (km)")
axes[1].set_ylabel("y (km)")
axes[2].set_ylabel("y (km)")
```

## From the ground to the source: disdrometers

For a disdrometer at the ground the question is reversed: where, and when, were
the drops of each size aloft? `rain_source_points` integrates the same
equations backward in time from the site. Given the DSD aloft, it also returns
the spectrum at the site, with the concentration ratio $n_{\rm ground}/n_{\rm
aloft}$ from the continuity equation along the path.

```{code-cell} ipython3
site = {"x": -2e3, "y": -6e3}  # a node of the surface grid
arrival = np.arange(900.0, 2401.0, 60.0)
back = rain_source_points(
    xr.Dataset({"x": ((), site["x"]), "y": ((), site["y"]), "z": ((), 0.0)},
               coords={"time": arrival}),
    bins,
    source_height=2500.0,
    profile=profile,
    storm_motion=motion,
    source_dsd=nd_src,
    time_step=10.0,
)
at_site = back.ND.transpose("time", "diameter")
fwd_site = surface.ND.sel(x=site["x"], y=site["y"], time=arrival)
```

The sources of one arrival time, relative to the site, line up along the
storm-relative wind aloft, ordered by size:

```{code-cell} ipython3
b = back.sel(time=1620.0)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.6), constrained_layout=True)
ok = b.status == 1
sc = ax1.scatter(
    ((b.source_x - site["x"]) / 1e3)[ok],
    ((b.source_y - site["y"]) / 1e3)[ok],
    c=b.diameter[ok],
    cmap="viridis",
    s=28,
    zorder=3,
)
ax1.plot(0, 0, "k+", ms=10, mew=1.5, zorder=4)
ax1.set_aspect("equal")
ax1.set_xlabel("source x - site x (km)")
ax1.set_ylabel("source y - site y (km)")
ax1.set_title("source of each size (+ = site)")
fig.colorbar(sc, ax=ax1, label="diameter at the ground (mm)", shrink=0.8)

ax2.plot(b.diameter[ok], (b.fall_time)[ok] / 60.0, "o-", color=OKABE["blue"], ms=4)
ax2.set_xlabel("diameter at the ground (mm)")
ax2.set_ylabel("time since the drops were at 2.5 km (min)")
ax2.set_title("fall time");
```

Reading the DSD aloft at those sources gives the surface spectrum. It agrees
with the number-flux conserving accumulation of the forward calculation, two
completely different ways to the same answer (the forward one smooths over the
2-km, 1-min grid and samples the cell once a minute):

```{code-cell} ipython3
groups = {"1-2 mm": (1.0, 2.0), "2-3.5 mm": (2.0, 3.5), "3.5-6 mm": (3.5, 6.1)}
fig, ax = plt.subplots(figsize=(7.5, 3.4), constrained_layout=True)
for (name, (lo, hi)), c in zip(groups.items(), (OKABE["orange"], OKABE["blue"], OKABE["red"])):
    sel = (bins >= lo) & (bins < hi)
    f = (fwd_site.isel(diameter=sel) * width).sum("diameter")
    bk = (at_site.isel(diameter=sel) * width).sum("diameter")
    peak = float(bk.max())
    ax.plot(arrival / 60, bk / peak, "-", color=c, label=f"backward, {name}")
    ax.plot(arrival / 60, f / peak, "o", color=c, ms=3.5, label=f"forward, {name}")
ax.set_xlabel("time (min)")
ax.set_ylabel("concentration / its maximum (backward)")
ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.28), frameon=False)
```

The three size groups reach the site with different timing and width because
their drops come from different places aloft (the map above), and the two
calculations agree.

## Constant wind and the trajectory-pair construction

In a wind $\mathbf u$ that does not change with height or time, without
evaporation, the drops of size $D$ released from a point of the moving echo
pattern land at a site at the time

$$ t(D) = t_g + \tau_D - \mathbf u \cdot \hat{\mathbf c}\,(\tau_D - \tau_{\rm ref}) / |\mathbf c| , $$

with the fall time $\tau_D$ and the storm motion $\mathbf c$. It is the
construction used to pair a radar gate with the spectra of a disdrometer.
`trajectory_matched_times` finds the same times by iterating on the release
time, and also works for winds that change with height and time and for
evaporating drops:

```{code-cell} ipython3
uniform = profile.copy(deep=True)
uniform["u"] = ("height", np.full(z.size, 9.0))
uniform["v"] = ("height", np.full(z.size, 4.0))
uniform["temperature"] = ("height", np.full(z.size, 288.0))
uniform["pressure"] = ("height", np.full(z.size, 95000.0))
uniform["relative_humidity"] = ("height", np.zeros(z.size))  # dry air: rho = p / (R T)
h, u, c = 2500.0, np.array([9.0, 4.0]), np.array(motion)
d_demo = np.array([1.0, 2.0, 3.0, 5.0])
rho = 95000.0 / (287.04 * 288.0)
tau = h / (terminal_fall_speed(d_demo, rho).values)
tau_ref = h / terminal_fall_speed(2.0, rho).values
closed = tau - (u @ c / np.linalg.norm(c)) * (tau - tau_ref) / np.linalg.norm(c)
gate = {"x": 20e3 - u[0] * tau_ref, "y": 5e3 - u[1] * tau_ref, "z": h, "time": 0.0}
found = trajectory_matched_times(
    gate, {"x": 20e3, "y": 5e3}, d_demo, storm_motion=motion, profile=uniform,
    evaporation=False,
)
print("diameter (mm)  closed form (s)  trajectories (s)")
for d, a, b_ in zip(d_demo, closed, found.arrival_time.values.ravel()):
    print(f"{d:13.1f}  {a:15.4f}  {b_:16.4f}")
```

## How accurate is the integration?

Drops are integrated with the fourth-order Runge-Kutta method (or Heun's method)
in steps of `time_step` seconds. The last step is shortened to end exactly on the
surface. Halving the step reduces the error of the landing point by $2^4$ (or
$2^2$) until round-off:

```{code-cell} ipython3
zf = np.arange(0.0, 8001.0, 1.0)  # smooth on every scale of the steps below
fine = environment(zf, 10.0 * np.sin(zf / 700.0), 4.0 * np.cos(zf / 1100.0))
src = {"x": 0.0, "y": 0.0, "z": 5000.0}
ref = rain_trajectories(src, [2.0, 4.0], profile=fine, time_step=0.5, max_time=2000.0)
steps = np.array([160.0, 80.0, 40.0, 20.0])
fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
for scheme, c in (("rk2", OKABE["orange"]), ("rk4", OKABE["blue"])):
    err = []
    for s in steps:
        r = rain_trajectories(
            src, [2.0, 4.0], profile=fine, time_step=s, max_time=2000.0, scheme=scheme
        )
        err.append(
            np.hypot(r.landing_x - ref.landing_x, r.landing_y - ref.landing_y).values.max()
        )
    ax.loglog(steps, err, "o-", color=c, label=scheme)
    order = 2 if scheme == "rk2" else 4
    ax.loglog(steps, 0.3 * err[0] * (steps / steps[0]) ** order, ":", color=c, lw=1,
              label=f"slope {order}")
ax.set_xlabel("time step (s)")
ax.set_ylabel("error of the landing point (m)")
ax.legend(frameon=False);
```

## References

- Atlas, D., R. C. Srivastava, and R. S. Sekhon, 1973: Doppler radar
  characteristics of precipitation at vertical incidence. *Rev. Geophys.*,
  **11** (1), 1–35, <https://doi.org/10.1029/RG011i001p00001>
- Dawson, D. T., E. R. Mansell, and M. R. Kumjian, 2015: Does wind shear cause
  hydrometeor size sorting? *J. Atmos. Sci.*, **72** (1), 340–348,
  <https://doi.org/10.1175/JAS-D-14-0084.1>
- Foote, G. B., and P. S. du Toit, 1969: Terminal velocity of raindrops aloft.
  *J. Appl. Meteor.*, **8** (2), 249–253,
  <https://doi.org/10.1175/1520-0450(1969)008<0249:TVORA>2.0.CO;2>
- Kumjian, M. R., and A. V. Ryzhkov, 2010: The impact of evaporation on
  polarimetric characteristics of rain: Theoretical model and practical
  implications. *J. Appl. Meteor. Climatol.*, **49** (6), 1247–1267,
  <https://doi.org/10.1175/2010JAMC2243.1>
- Kumjian, M. R., and A. V. Ryzhkov, 2012: The impact of size sorting on the
  polarimetric radar variables. *J. Atmos. Sci.*, **69** (6), 2042–2060,
  <https://doi.org/10.1175/JAS-D-11-0125.1>
