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

# Collisions, coalescence and breakup of raindrops in a rain shaft

`rain_trajectories` follows every drop size on its own path, so sizes cannot
collide. `radarx.retrieve.rain_shaft` is the collisional part: it takes the drop
size distribution (DSD) at the top of a column of still air and returns it at
depths below, with the drops colliding, coalescing and breaking up as they fall.
The collision kernel is the geometric sweep-out kernel; the outcome of a
collision is the parameterisation of Low and List (1982a, b): the coalescence
efficiency, then a filament, sheet or disk breakup with their fragment numbers
and size distributions. A compiled multithreaded kernel builds the tensor of
events and integrates the steady-state flux equation; the result conserves the
mass flux, that is the rain rate, to rounding.

This notebook checks the scheme against the tables of the two papers, runs a
Marshall-Palmer rain through a 4 km shaft, and shows what the choices of
radarx are. The equations, the places where the printed equations are not used
as printed, and the limits are in the documentation of
`radarx.retrieve.rain_collisions`.

```{code-cell} ipython3
import time

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.retrieve import _rain_collisions_numpy as ll82
from radarx.retrieve import _raindrop_fall, rain_shaft

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
OKABE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "black": "#222222"}
```

## The numbers of the paper

Part I of Low and List (1982a) gives, for ten pairs of colliding drops, the
collision kinetic energy and the coalescence efficiency; Part II (1982b) the
fractions of the breakup types and the mean number of fragments, with the
equations. The table below computes them from the equations with the energies
of the paper (the diameters are in cm, as there). The agreement is to the
printed digits except for the disks, whose numbers follow the logarithm of the
energy that the paper's fall-speed fit gives slightly differently.

```{code-cell} ipython3
pairs = [  # D_L, D_S (cm), CKE (J): Table 2 of Part I
    (0.18, 0.0395, 3.219e-7), (0.40, 0.0395, 8.426e-7), (0.44, 0.0395, 8.781e-7),
    (0.18, 0.0715, 8.935e-7), (0.18, 0.10, 9.484e-7), (0.30, 0.10, 41.00e-7),
    (0.36, 0.10, 53.53e-7), (0.46, 0.10, 64.78e-7), (0.36, 0.18, 85.50e-7),
    (0.46, 0.18, 124.5e-7),
]
paper_e = [0.507, 0.620, 0.633, 0.315, 0.244, 0.0842, 0.0, 0.0, 0.0, 0.0]  # Part I, Table 3
paper_f = [2.02, 1.79, 1.76, 3.67, 2.24, 5.06, 8.11, 11.47, 10.33, 14.20]  # Part II, Table 3
print(" D_L    D_S   E_coal (paper)   R_f   R_s   R_d   F_f   F_s   F_d   total F (paper)")
for (dl, ds, cke), e_paper, f_paper in zip(pairs, paper_e, paper_f):
    st = np.pi * ll82.SIGMA * ((dl * 1e-2) ** 2 + (ds * 1e-2) ** 2)
    sc = np.pi * ll82.SIGMA * ((dl * 1e-2) ** 3 + (ds * 1e-2) ** 3) ** (2 / 3)
    e = ll82.coalescence_efficiency(dl, ds, cke, sc, st - sc)
    rf, rs, rd = ll82.breakup_fractions_raw(cke, st)
    ff, fs, fd = ll82.mean_numbers(dl, ds, cke, st)
    total = (rf * ff + rs * fs + rd * fd) * (1 - e) + e
    print(f"{dl:5.2f} {ds:6.4f}   {e:5.3f}  ({e_paper:5.3f})   {rf:5.2f} {rs:5.2f} {rd:5.2f} "
          f"{ff:5.2f} {fs:5.2f} {fd:5.2f}   {total:6.2f}  ({f_paper:5.2f})")
```

The coalescence efficiency and the breakup types depend on the sizes through the
energies. This map shows the efficiency for the sizes of raindrops, with the
radarx fall speeds, and the probability of filament, sheet and disk breakup for
a drop of 4 mm hit by smaller ones.

```{code-cell} ipython3
d = np.geomspace(0.15, 6.0, 90)
v = _raindrop_fall.sea_level_speed(d)[0]
eff = np.full((d.size, d.size), np.nan)
for i in range(d.size):
    for j in range(i):
        cke, st, sc, dsig = ll82.energies(d[i] * 0.1, d[j] * 0.1, v[i], v[j])
        eff[i, j] = ll82.coalescence_efficiency(d[i] * 0.1, d[j] * 0.1, cke, sc, dsig)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9), constrained_layout=True)
mesh = ax1.pcolormesh(d, d, eff, cmap="viridis", shading="nearest", vmin=0, vmax=0.8)
ax1.set(xscale="log", yscale="log", xlabel="small drop (mm)", ylabel="large drop (mm)",
        title="coalescence efficiency")
ticks = [0.2, 0.5, 1.0, 2.0, 4.0]
ax1.set_xticks(ticks, [f"{t:g}" for t in ticks])
ax1.set_yticks(ticks, [f"{t:g}" for t in ticks])
ax1.minorticks_off()
fig.colorbar(mesh, ax=ax1)
dl = 0.4
small = d[d < 4.0]
fractions = []
for ds in small:
    cke, st, sc, dsig = ll82.energies(dl, ds * 0.1, float(_raindrop_fall.sea_level_speed(dl * 10)[0]),
                                      float(_raindrop_fall.sea_level_speed(ds)[0]))
    fractions.append(ll82.breakup_fractions(cke, st))
fractions = np.array(fractions)
for k, (name, color) in enumerate(zip(("filament", "sheet", "disk"), (OKABE["blue"], OKABE["orange"], OKABE["red"]))):
    ax2.plot(small, fractions[:, k], color=color, label=name)
ax2.set(xscale="log", xlabel="small drop (mm)", ylabel="share of the breakups", title="breakup type, large drop 4 mm")
ax2.set_xticks(ticks, [f"{t:g}" for t in ticks])
ax2.minorticks_off()
ax2.legend(frameon=False)
plt.show()
```

## A rain shaft

A Marshall-Palmer DSD of 50 mm h$^{-1}$ at the top, 40 pivot sizes from 0.2 to
7 mm, 4 km of still air at the sea-level density. Four runs: no collisions,
every collision coalesces, and the scheme of Low and List.

```{code-cell} ipython3
d = np.geomspace(0.2, 7.0, 40)
rate = 50.0
nd = xr.DataArray(8000.0 * np.exp(-4.1 * rate**-0.21 * d), dims="diameter",
                  coords={"diameter": ("diameter", d, {"units": "mm"})}, name="N")
depths = np.array([0.0, 500.0, 1000.0, 2000.0, 4000.0])
runs = {}
for name in ("none", "coalescence", "low_list1982"):
    t0 = time.time()
    runs[name] = rain_shaft(nd, depth=depths, collisions=name)
    print(f"{name:13s} {1000 * (time.time() - t0):5.0f} ms, rain rate "
          f"{float(runs[name].rain_rate.isel(depth=0)):.3f} -> {float(runs[name].rain_rate.isel(depth=-1)):.3f} mm/h")
width = np.diff(np.exp(np.concatenate([[2 * np.log(d[0]) - 0.5 * np.log(d[0] * d[1])],
                                        0.5 * np.log(d[:-1] * d[1:]),
                                        [2 * np.log(d[-1]) - 0.5 * np.log(d[-1] * d[-2])]])))
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)
for depth, ls in zip((0.0, 1000.0, 4000.0), ("-", "--", ":")):
    for name, color in (("coalescence", OKABE["blue"]), ("low_list1982", OKABE["red"])):
        axes[0].semilogy(d, runs[name].dsd.sel(depth=depth), ls, color=color,
                         label=f"{name}, {depth / 1000:.0f} km" if depth else None)
axes[0].semilogy(d, nd, color=OKABE["black"], lw=1, label="top")
axes[0].set(xlabel="diameter (mm)", ylabel="N(D) (m$^{-3}$ mm$^{-1}$)", ylim=(1e-2, 2e4))
axes[0].legend(frameon=False, fontsize=7.5)
for name, color in (("none", OKABE["black"]), ("coalescence", OKABE["blue"]), ("low_list1982", OKABE["red"])):
    out = runs[name]
    axes[1].plot(depths / 1e3, out.number_concentration, "o-", color=color, label=name)
    mass = (out.dsd * width * d**4).sum("diameter") / (out.dsd * width * d**3).sum("diameter")
    axes[2].plot(depths / 1e3, mass, "o-", color=color, label=name)
axes[1].set(xlabel="distance below the top (km)", ylabel="drops per m$^3$")
axes[2].set(xlabel="distance below the top (km)", ylabel="mass-weighted mean diameter (mm)")
axes[1].legend(frameon=False)
plt.show()
```

With coalescence only the drops grow and become few; with breakup the large
drops that coalescence makes break up again and the small fragments make the
number of drops rise first, so the distribution develops the small-drop and
large-drop peaks that the studies of the equilibrium of rain with the Low and
List kernel find (McFarquhar 2004; Prat and Barros 2007). The rise of the
smallest bin of the grid is the fragments below 0.2 mm, which are put on the
first pivot (the mass is conserved, the number is not): that is where the
parameterisation is an extrapolation, see the limits below.

## A two-moment shaft

A bin model is too expensive for every gate of a volume.
`radarx.retrieve.rain_shaft_bulk` follows two moments of the rain, the number
concentration and the water content, with the self-collection and breakup of
Seifert (2008, Appendix): the drops are a gamma distribution whose shape
parameter follows from the mean volume diameter $D_m$ (his Eq. 20; $\mu = 1$
at the equilibrium diameter of 1.1 mm), self-collection removes drops at the
rate $k_{rr} N_r L_r$ and breakup is a relaxation of $D_m$ toward 1.1 mm. The
figure compares it with the bin model of this notebook on the same tops. The
two agree on the direction and the scale of the change and on a mean volume
diameter between 1 and 1.3 mm after a few kilometres, not on the details: the
bin model with the breakup of Low and List first undershoots 1.1 mm and then
settles at about 1.26 mm, the two-moment scheme goes to 1.1 mm without
overshoot, and rain of small drops coalesces faster in the two-moment scheme.
The rain rates differ by a few percent because the two-moment scheme assumes a
gamma distribution with the shape parameter of Eq. (20) for the same water
content and number, which has other mass-weighted fall speeds than the
distribution of the bins; each conserves its own rain rate.

```{code-cell} ipython3
from radarx.retrieve import _rain_bulk_numpy as seifert
from radarx.retrieve import rain_shaft_bulk

depths = np.arange(0.0, 8001.0, 500.0)
d_bins = np.geomspace(0.15, 8.0, 60)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9), constrained_layout=True)
dm_axis = np.linspace(0.2, 5.0, 300)
ax1.plot(dm_axis, seifert.shape_parameter(dm_axis * 1e-3), color=OKABE["black"])
ax1.axvline(1.1, color="0.5", lw=0.8)
ax1.set(xlabel="mean volume diameter (mm)", ylabel="shape parameter", title="Eq. (20) of Seifert (2008)")
for (dm0, lwc, shape_mu), color in zip(((2.5, 2.0, 3.0), (4.0, 3.0, 6.0), (0.6, 1.0, 3.0)),
                                       (OKABE["blue"], OKABE["orange"], OKABE["green"])):
    lam = ((shape_mu + 3) * (shape_mu + 2) * (shape_mu + 1)) ** (1 / 3) / dm0
    shape = d_bins**shape_mu * np.exp(-lam * d_bins)
    mass = np.pi / 6 * 1e-3 * d_bins**3 * shape
    scale = lwc / np.sum(0.5 * (mass[1:] + mass[:-1]) * np.diff(d_bins))
    top = xr.DataArray(shape * scale, dims="diameter", coords={"diameter": d_bins})
    bins = rain_shaft(top, depth=depths)
    dm_bin = (6e-3 * bins.liquid_water_content / (np.pi * 1000 * bins.number_concentration)) ** (1 / 3) * 1e3
    bulk = rain_shaft_bulk(bins.isel(depth=0)[["number_concentration", "liquid_water_content"]], depth=depths)
    ax2.plot(depths / 1e3, dm_bin, color=color, label=f"bins, top {dm0:g} mm")
    ax2.plot(depths / 1e3, bulk.mean_volume_diameter, "--", color=color)
    print(f"top {dm0:g} mm, {lwc:g} g/m3: after 8 km bins {float(dm_bin[-1]):.2f} mm, bulk "
          f"{float(bulk.mean_volume_diameter[-1]):.2f} mm; rain rate bins {float(bins.rain_rate[-1]):.2f}, "
          f"bulk {float(bulk.rain_rate[-1]):.2f} mm/h")
ax2.axhline(1.1, color="0.5", lw=0.8)
ax2.set(xlabel="distance below the top (km)", ylabel="mean volume diameter (mm)",
        title="solid: bin model (Low and List), dashed: two moments")
ax2.legend(frameon=False, fontsize=8)
plt.show()
```

## Conservation, accuracy and cost

The rain rate is the flux of water and does not change with depth. The
integration is checked against a SciPy integration of the same equations in
the tests; the step is the shorter of `step` (metres) and a share `cfl` of the
distance in which the busiest drop size collides once, and the table shows how
the rain rate and the number of drops depend on that share.

```{code-cell} ipython3
for cfl in (1.0, 0.25, 0.05):
    t0 = time.time()
    out = rain_shaft(nd, depth=4000.0, cfl=cfl)
    print(f"cfl {cfl:4.2f}: {1000 * (time.time() - t0):6.0f} ms, drops {float(out.number_concentration):10.4f} m-3, "
          f"rain rate {float(out.rain_rate):.9f} mm/h")
```

Several columns are integrated at the same time and on all cores: every point
of a field of DSDs is one column (the gates of a sweep, the times of a
disdrometer). The tensor of events is built once per call, so the cost of one
call is the construction (0.1 s for 100 sizes) plus about 5 ms per column and
km for 40 sizes.

```{code-cell} ipython3
rates = xr.DataArray([5.0, 20.0, 50.0, 100.0, 200.0], dims="point", coords={"point": [5.0, 20.0, 50.0, 100.0, 200.0]})
field = xr.DataArray(8000.0 * np.exp(-4.1 * rates.values[:, None] ** -0.21 * d[None, :]),
                     dims=("point", "diameter"), coords={"point": rates.point, "diameter": d})
t0 = time.time()
many = field.radarx.rain_shaft(depth=[0.0, 3000.0])  # the accessor
print(f"{field.sizes['point']} columns in {1000 * (time.time() - t0):.0f} ms")
for rr in rates.values:
    top = many.sel(point=rr).isel(depth=0)
    bottom = many.sel(point=rr).isel(depth=1)
    print(f"R = {float(top.rain_rate):7.2f} mm/h: drops {float(top.number_concentration):8.1f} -> "
          f"{float(bottom.number_concentration):8.1f} m-3, rain rate after 3 km {float(bottom.rain_rate):7.2f}")
```

## What is, and is not, in the scheme

- The parameters come from the two papers as printed, except for four places
  where the printed equations are not used as printed (listed in the module
  documentation); the numbers of the tables above are reproduced.
- The large drop that is left after a breakup is not the Gaussian of the paper
  but a drop whose size makes the event conserve mass (as in McFarquhar 2004):
  for filaments it is within 1.3 % of the large colliding drop.
- The experiments cover large drops of 1.8 to 4.6 mm and small drops of 0.395 to
  1.8 mm. Other pairs use the fits as extrapolations. A collision efficiency of one
  overestimates the collisions of drops below 0.1 mm.
- No evaporation, wind or aerodynamic breakup; the density correction of the
  fall speed is applied if a profile or a top height is given.

## References

- Low, T. B., and R. List, 1982a: Collision, coalescence and breakup of
  raindrops. Part I: Experimentally established coalescence efficiencies and
  fragment size distributions in breakup. *J. Atmos. Sci.*, **39**, 1591–1606,
  <https://doi.org/10.1175/1520-0469(1982)039<1591:CCABOR>2.0.CO;2>
- Low, T. B., and R. List, 1982b: Collision, coalescence and breakup of
  raindrops. Part II: Parameterization of fragment size distributions.
  *J. Atmos. Sci.*, **39**, 1607–1619,
  <https://doi.org/10.1175/1520-0469(1982)039<1607:CCABOR>2.0.CO;2>
- McFarquhar, G. M., 2004: A new representation of collision-induced breakup of
  raindrops and its implications for the shapes of raindrop size distributions.
  *J. Atmos. Sci.*, **61**, 777–794,
  <https://doi.org/10.1175/1520-0469(2004)061<0777:ANROCB>2.0.CO;2>
- Prat, O. P., and A. P. Barros, 2007: A robust numerical solution of the
  stochastic collection-breakup equation for warm rain. *J. Appl. Meteor.
  Climatol.*, **46**, 1480–1497, <https://doi.org/10.1175/JAM2544.1>
- Seifert, A., 2008: On the parameterization of evaporation of raindrops as
  simulated by a one-dimensional rainshaft model. *J. Atmos. Sci.*, **65**,
  3608–3619, <https://doi.org/10.1175/2008JAS2586.1>
