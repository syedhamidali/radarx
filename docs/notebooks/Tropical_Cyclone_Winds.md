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

# Tropical-cyclone winds from one Doppler radar

A single Doppler radar measures only the wind along its beams. For a vortex such
as a tropical cyclone the beam crosses a ring around the centre at an angle that
changes along the ring, so the Doppler velocity on the ring still holds the
axisymmetric tangential wind, the axisymmetric radial wind and the low azimuthal
wavenumbers of the tangential wind. The *ground-based velocity track display*
(GBVTD, [Lee et al. 1999](https://doi.org/10.1175/1520-0493(1999)127<2419:TCKSRF>2.0.CO;2))
fits Fourier series to the velocity on such rings. `radarx.retrieve` has three
steps:

- `tc_center` finds the circulation centre and the radius of maximum wind: the
  centre that maximises the axisymmetric tangential wind of GBVTD, searched with
  a simplex from a grid of starting points and ranked as in
  [Bell and Lee (2012)](https://doi.org/10.1175/JAMC-D-11-0167.1);
- `gbvtd` retrieves, on rings around a centre, the axisymmetric tangential and
  radial wind, the wavenumber-1, 2 and 3 tangential wind and the along-beam mean
  wind;
- `tc_decompose` rebuilds the winds on a polar grid and splits them into the
  symmetric part and the asymmetric parts.
- `tc_ring_stats` gives the azimuthal mean and wavenumbers of any polar field (the
  reflectivity) on rings around the centre, and `tc_structure` analyses a radial
  profile of the wind or the reflectivity: the eyewall maximum, its width, secondary
  maxima and the moat, wind radii, the vorticity and the angular momentum.

`synthetic_vortex` and `vortex_profile` build an analytic vortex and forward-model
it to Doppler velocity, which is how the retrieval is tested below against a
known truth. The ring fits and the simplex searches run in a compiled,
multithreaded kernel; a NumPy implementation with the same steps gives the same
numbers, much more slowly.

```{note}
The equations and the centre finding were compared with the three papers of the
GBVTD series: Lee et al. (1999, Part I: the technique and its test on analytic
vortices), Lee and Marks (2000, Part II: the GBVTD-simplex centre finding) and
[Lee et al. (2000)](https://doi.org/10.1175/1520-0493(2000)129<3982:TCKSRF>2.0.CO;2)
(Part III: Typhoon Alex), and the ranking of the radii with Bell and Lee (2012).
The section "Checks against the papers" below reproduces the numbers of the
analytic experiments. Not checked: the interpolation of the sweep to constant
altitude that the papers use (radarx fits the gates directly), the figures of
the papers beyond the values stated in their texts and tables, and the
application to Typhoon Alex. Not implemented: the temporal part of Bell and Lee,
the modified GBVTD for the cross-beam mean wind
([Chen et al. 2013](https://doi.org/10.1175/JAMC-D-13-031.1)), and the extended GBVTD with
two radars, and the wavenumber-1 criterion of Part II (which the paper itself
rejects). The eyewall and ring diagnostics are in the section "Eyewall and ring
diagnostics" below. The module documentation `radarx.retrieve.tc_winds`
says which choices are radarx's own.
```

## The equations

The radar is at the origin, the centre at the distance $R_T$ in the direction
$\theta_T$, and a ring of radius $R$ has $\sin\alpha_{max} = R/R_T$. At a gate on
the ring the angle $\psi$ runs from the beam to the outward radial direction. The
horizontal projection of the Doppler velocity, $\hat V_d = V_d/\cos\phi$ for the
beam elevation $\phi$, is for a mean wind $V_M$ towards $\theta_M$, a tangential
wind $V_T$ (positive anticlockwise) and a radial wind $V_R$ (positive outward)

$$
\hat V_d = V_M\Big\{\cos(\theta_T-\theta_M)\Big[\tfrac{1-\cos\alpha_{max}}{2}\cos2\psi+\tfrac{1+\cos\alpha_{max}}{2}\Big]-\sin(\theta_T-\theta_M)\sin\alpha_{max}\sin\psi\Big\}-V_T\sin\psi+V_R\cos\psi .
$$

With $\hat V_d(\psi)=\sum_{n=0}^{4}(A_n\cos n\psi+B_n\sin n\psi)$ fitted on the ring
(Eq. 8 of Lee et al. 1999; the wavenumber $n$ of the tangential wind needs the
order $n+1$) and the closure that the asymmetric radial wind is zero (Eqs. 19 to 27),

$$
V_{TC0} = -B_1 - B_3 - V_M\sin(\theta_T-\theta_M)\sin\alpha_{max},\quad
V_{RC0} = A_1 + A_3,\quad
V_M\cos(\theta_T-\theta_M) = A_0 + A_2 + A_4,
$$

$$
V_{TC1} = -2(B_2+B_4),\quad V_{TS1} = A_2 - A_0 + A_4 + (A_0+A_2+A_4)\cos\alpha_{max},
$$

$$
V_{TC2} = -2B_3,\quad V_{TS2} = 2A_3,\quad V_{TC3} = -2B_4,\quad V_{TS3} = 2A_4 .
$$

Eq. 20 of the paper is printed with $+B_3$. Eqs. 15 and 18 of the same paper
give $-B_3$ (the $V_{TC2}$ of $B_1$ cancels only with the minus sign), which is
what the code uses and what Liou et al. (2005) and Chen et al. (2013) have.
The cross-beam mean wind $V_M\sin(\theta_T-\theta_M)$ cannot be found from one
radar and leaks into the axisymmetric tangential wind $V_{TC0}$ by
$V_M\sin\alpha_{max}$ (Fig. 11 of the paper); `gbvtd` removes it when `mean_wind`
(the storm motion is a common proxy) is given.

## An analytic vortex seen by a radar

The test vortex is the modified Rankine-combined vortex of Lee et al. (1999,
Eqs. 28 to 31; the exponent of the decay is the one of
[Chen et al. 2013](https://doi.org/10.1175/JAMC-D-13-031.1)): a maximum tangential
wind of 50 m/s at 20 km, a decay as $(R_{max}/R)^{0.6}$ outside, an axisymmetric
radial wind (outflow inside, inflow outside), wavenumber-1 and wavenumber-2
tangential asymmetries, a uniform mean wind and noise. The centre is 30 km east and
70 km north of the radar. The wind is projected on the beams of a 0.5° sweep with
the beam curvature of the 4/3 Earth model (the Doppler velocity is the projection
of the horizontal wind times the cosine of the local beam elevation).

```{code-cell} ipython3
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.retrieve import (
    gbvtd,
    synthetic_vortex,
    tc_center,
    tc_decompose,
    vortex_profile,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="The input coordinates to pcolormesh")

CENTER = (30000.0, 70000.0)  # m east and north of the radar
VMAX, RMW, XT = 50.0, 20000.0, 0.6
MEAN_WIND = (8.0, -4.0)  # m/s
RADIAL = (0.1, 3.0)  # c1 and c2 of Eqs. 30 and 31 of Lee et al. (1999), radii in km
ASYMMETRY = {1: (0.2, 45.0), 2: (0.1, 100.0)}  # relative amplitude, azimuth of the maximum

sweep = synthetic_vortex(
    center=CENTER, vmax=VMAX, rmw=RMW, xt=XT, radial=RADIAL, asymmetry=ASYMMETRY,
    mean_wind=MEAN_WIND, noise=2.0, seed=0, elevation=0.5,
)
sweep
```

```{code-cell} ipython3
geo = sweep.xradar.georeference()
fig, ax = plt.subplots(figsize=(6.2, 5.2), layout="constrained")
m = ax.pcolormesh(geo.x / 1e3, geo.y / 1e3, sweep.VRADH, cmap="RdBu_r", vmin=-50, vmax=50, shading="auto")
ax.plot(*np.array(CENTER) / 1e3, "k+", ms=14)
ax.plot(0, 0, "k^")
ax.set(xlim=(-10, 70), ylim=(20, 120), aspect="equal", xlabel="east (km)", ylabel="north (km)",
       title="Doppler velocity (m/s), radar at the triangle")
fig.colorbar(m, label="m/s")
plt.show()
```

## The centre and the radius of maximum wind

`tc_center` needs no first guess (it starts from the midpoint of the strongest
inbound and outbound velocities, the velocity dipole), but a best-track or
eye-fix position can be given as `first_guess=(x, y)` in metres or as
`first_guess_latlon`. For each ring radius it starts 16 simplex searches from a
4 by 4 grid around the guess (the first simplex is the equilateral triangle of
Lee and Marks 2000, 4 km from its centre, and a search ends when all its sides
are shorter than 50 m), keeps the converging ones within one standard deviation
of their mean, and ranks the radii with the NORM score of Bell and Lee (2012);
`selection="max_vt"` takes the ring with the highest axisymmetric wind, as Lee
and Marks do. The velocity must be dealiased first (`dealias_velocity`).

`max_wavenumber` is the highest Fourier order of the ring fit. The default, 2,
gives the vortex wavenumbers 0 and 1: the maximum winds of Table 4 of Lee and Marks
(2000) are those of this fit (see "Checks against the papers"), and it is the
choice of Murillo et al. (2011) as reported by Bell and Lee (2012). The maximum of
a noisy objective is biased, and a higher order lets more noise into the
tangential wind: over ten noise realisations of this vortex the order 2 has the
smallest median and rms error and the order 3 the largest.

```{code-cell} ipython3
kw = dict(min_radius=14000.0, max_radius=26000.0)
for order in (1, 2, 3):
    err = []
    for seed in range(10):
        s = synthetic_vortex(center=CENTER, vmax=VMAX, rmw=RMW, xt=XT, radial=RADIAL, asymmetry=ASYMMETRY,
                             mean_wind=MEAN_WIND, noise=2.0, seed=seed, elevation=0.5)
        c = tc_center(s, max_wavenumber=order, **kw)
        err.append(np.hypot(float(c.CENTER_X) - CENTER[0], float(c.CENTER_Y) - CENTER[1]))
    print(f"Fourier order {order}: centre error median {np.median(err):.0f} m, rms {np.sqrt(np.mean(np.square(err))):.0f} m, "
          f"largest {np.max(err):.0f} m over 10 noise realisations")
```

```{code-cell} ipython3
t0 = time.perf_counter()
found = tc_center(sweep, min_radius=8000.0, max_radius=40000.0)
print(f"{time.perf_counter() - t0:.2f} s")
cx, cy = float(found.CENTER_X), float(found.CENTER_Y)
print(f"centre ({cx:.0f}, {cy:.0f}) m, error {np.hypot(cx - CENTER[0], cy - CENTER[1]):.0f} m, "
      f"spread of the searches {float(found.CENTER_UNCERTAINTY):.0f} m")
print(f"radius of maximum wind {float(found.RMW) / 1e3:.1f} km (truth {RMW / 1e3:.0f}), "
      f"maximum tangential wind {float(found.VT_MAX):.1f} m/s (truth {VMAX:.0f})")
print(f"first guess from the dipole: ({float(found.FIRST_GUESS_X):.0f}, {float(found.FIRST_GUESS_Y):.0f}) m")
print(f"geographic centre: {float(found.CENTER_LATITUDE):.3f} N, {float(found.CENTER_LONGITUDE):.3f} E")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
r_km = found.radius / 1e3
axes[0].plot(r_km, found.VT0, "o-")
axes[0].axvline(RMW / 1e3, color="0.6", ls="--")
axes[0].set(xlabel="ring radius (km)", ylabel="mean axisymmetric VT (m/s)", title="of the converging centres")
axes[1].plot(r_km, found.NORM_SCORE, "o")
axes[1].set(xlabel="ring radius (km)", ylabel="NORM score", title="rings that qualify as RMW")
sc = axes[2].scatter(found.RING_CENTER_X / 1e3 - CENTER[0] / 1e3, found.RING_CENTER_Y / 1e3 - CENTER[1] / 1e3,
                     c=found.radius / 1e3, cmap="viridis")
axes[2].plot(0, 0, "k+", ms=14)
axes[2].set(xlabel="x error (km)", ylabel="y error (km)", title="consensus centre of every ring")
fig.colorbar(sc, ax=axes[2], label="radius (km)")
plt.show()
```

Away from the radius of maximum wind the best centre of a ring is not the
circulation centre (a ring that is shifted towards the eyewall has a larger mean
wind), which is why the ring of the radius of maximum wind is selected.

## GBVTD on rings

`gbvtd` takes the centre as `(x, y)` in metres, `center_latlon` or the result of
`tc_center`. The truth below is the analytic profile; the retrieval is
compared with the cross-beam mean wind removed (`mean_wind`) and without.

```{code-cell} ipython3
radii = np.arange(5000.0, 50001.0, 1000.0)
vt_true, vr_true = vortex_profile(radii, VMAX, RMW, XT, RADIAL)
away = np.abs(radii - RMW) > 2000.0  # the kink of the Rankine profile at the RMW is smoothed by the rings


def rms(a):
    return float(np.sqrt(np.nanmean(np.asarray(a) ** 2)))


plain = gbvtd(sweep, found, radii=radii)
fixed = gbvtd(sweep, found, radii=radii, mean_wind=MEAN_WIND)
print("mean axisymmetric tangential wind, rms error away from the RMW")
print(f"  mean wind aliased:  {rms((plain.VT0.values - vt_true)[away]):.2f} m/s")
print(f"  mean wind removed:  {rms((fixed.VT0.values - vt_true)[away]):.2f} m/s")
print(f"axisymmetric radial wind, rms error {rms((fixed.VR0.values - vr_true)[away]):.2f} m/s")
u_t, v_t = MEAN_WIND
along = (cx * u_t + cy * v_t) / np.hypot(cx, cy)
print(f"along-beam mean wind: fit {float(fixed.VM_ALONG.sel(radius=30000.0)):.2f} m/s, truth {along:.2f} m/s")
print(f"RMW {float(fixed.RMW) / 1e3:.2f} km, maximum {float(fixed.VT_MAX):.1f} m/s")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
r_km = radii / 1e3
axes[0].plot(r_km, vt_true, "k", label="truth")
axes[0].plot(r_km, plain.VT0, label="mean wind aliased")
axes[0].plot(r_km, fixed.VT0, label="mean wind removed")
axes[0].set(xlabel="radius (km)", ylabel="VT0 (m/s)", title="axisymmetric tangential wind")
axes[0].legend()
axes[1].plot(r_km, vr_true, "k", label="truth")
axes[1].plot(r_km, fixed.VR0, label="GBVTD")
axes[1].set(xlabel="radius (km)", ylabel="VR0 (m/s)", title="axisymmetric radial wind")
axes[1].legend()
axes[2].plot(r_km, 0.2 * vt_true, "k", label="truth wn 1")
axes[2].plot(r_km, fixed.VT1_AMPLITUDE, label="GBVTD wn 1")
axes[2].plot(r_km, 0.1 * vt_true, "k--", label="truth wn 2")
axes[2].plot(r_km, fixed.VT2_AMPLITUDE, label="GBVTD wn 2")
axes[2].set(xlabel="radius (km)", ylabel="amplitude (m/s)", title="tangential asymmetries")
axes[2].legend()
plt.show()
print("azimuth of the wavenumber-1 maximum (truth 45 deg), clockwise from north:",
      np.round(fixed.VT1_AZIMUTH.sel(radius=[15000.0, 25000.0, 35000.0]).values, 1))
```

The wavenumber-1 and 2 amplitudes are retrieved to within 10 percent of the
tangential wind, and the axisymmetric wind is too low by a few m/s at large radii
where the mean wind is removed. Both are the geometric distortion that GBVTD has
because the Fourier series is in the angle $\psi$ and not in the azimuth around
the centre (Lee et al. 2006): from the geometry, a wavenumber-1 tangential wind of
relative amplitude $A$ puts a wavenumber-0 term of the order of
$A\,V_{T0}\sin\alpha_{max}/2$ into the series in $\psi$ (derived here, not taken
from Lee et al.), 1.7 m/s at 40 km for $A = 0.2$ and a 33 m/s wind. A real asymmetric *radial* wind is assumed zero by the closure and
leaks into the wavenumber-1 tangential wind: add `radial_asymmetry={1: (3.0, 120.0)}`
to `synthetic_vortex` to see a few m/s of it.

## Sensitivity to the centre

GBVTD depends strongly on the centre. Lee and Marks (2000, Tables 1 and 2) move the
centre of an axisymmetric vortex with a radius of maximum wind of 20 km by 1 to 5 km
and find an apparent wavenumber 1 of 2 to 15 m/s and an apparent mean flow; for a
real wavenumber 1 of 10 m/s a displacement of 1 km is an error of 20 percent. The
tests of the next section reproduce their table. Here the centre of a vortex with a
real wavenumber 1 is moved along and across the beam by 1, 2 and 5 km:

```{code-cell} ipython3
sweep_a = synthetic_vortex(center=(0.0, 80000.0), vmax=VMAX, rmw=RMW, xt=XT, radial=RADIAL,
                           asymmetry={1: (0.2, 45.0)}, mean_wind=(10.0, 0.0), noise=2.0, seed=3)
c0 = np.array([0.0, 80000.0])
rows = []
for d in (0.0, 1000.0, 2000.0, 5000.0):
    for name, off in (("along the beam", (0.0, d)), ("across the beam", (d, 0.0))):
        if d == 0.0 and name == "across the beam":
            continue
        g = gbvtd(sweep_a, tuple(c0 + off), radii=radii, mean_wind=(10.0, 0.0))
        e0 = rms(((g.VT0.values - vt_true) / vt_true)[away])
        e1 = rms(((g.VT1_AMPLITUDE.values - 0.2 * vt_true) / (0.2 * vt_true))[away])
        rows.append((d / 1e3, name, 100 * e0, 100 * e1, float(g.RMW) / 1e3, float(g.VT_MAX)))
print(f"{'error (km)':>10s} {'direction':>16s} {'VT0 rms (%)':>12s} {'wn-1 rms (%)':>13s} {'RMW (km)':>9s} {'Vmax (m/s)':>11s}")
for row in rows:
    print(f"{row[0]:10.0f} {row[1]:>16s} {row[2]:12.1f} {row[3]:13.1f} {row[4]:9.2f} {row[5]:11.1f}")
```

The mean axisymmetric wind changes by a few percent, the wavenumber-1 amplitude by
17 to 29 percent (10 percent without a displacement), and the radius of maximum wind
and the peak wind move with the centre error. `tc_center` finds the centre of this
vortex to a few hundred metres; the spread of its searches is a measure of the
uncertainty, not a guarantee.

## Checks against the papers

The test vortex of Lee et al. (1999, section 3) is 50 m/s at 20 km with the radar 60 km
south of the centre; the experiments of Lee et al. (1999) and Lee and Marks (2000) are
repeated on it. The sweeps of the papers are interpolated to a 1 km map and to a
cylindrical grid, which smooths the sharp maximum of the Rankine profile (the paper
retrieves 48.2 m/s for the 50 m/s of the centre); radarx fits the gates and keeps more of
it.

**Table 1 of Part II.** The centre of the analysis is moved from the true centre
(0, 60) km; the amplitudes of the wavenumbers in the series in $\psi$, the mean flow
($A_0+A_2+A_4$) and the radius of the ring that has the highest wavenumber 0 are given
for the axisymmetric vortex without radial wind. The ring is the one of the paper.

```{code-cell} ipython3
paper_sweep = synthetic_vortex(center=(0.0, 60000.0), vmax=50.0, rmw=20000.0, elevation=0.0,
                               azimuth_step=0.5, range_step=250.0, max_range=140000.0)
# centre (km): wave 0, 1, 2, 3 (m/s), mean flow (m/s), radius (km) of Table 1 of Lee and Marks (2000)
table1 = {
    (0, 60): (48.2, 0.0, 0.0, 0.0, 0.0, 20), (1, 60): (47.3, 2.1, 0.3, 0.1, 2.4, 20),
    (2, 60): (45.5, 6.4, 0.8, 0.3, 4.5, 21), (3, 60): (43.9, 10.0, 1.1, 0.7, 6.2, 22),
    (4, 60): (42.6, 12.6, 1.5, 0.7, 7.5, 23), (5, 60): (41.3, 14.6, 2.7, 0.6, 8.5, 24),
    (0, 61): (48.0, 2.1, 0.7, 0.2, 0.0, 20), (0, 62): (47.9, 6.1, 1.9, 0.1, 0.0, 21),
    (0, 63): (47.6, 7.8, 3.0, 0.0, 0.0, 21), (0, 64): (47.0, 9.0, 4.0, 0.2, 0.0, 21),
    (0, 65): (46.2, 10.1, 4.8, 0.4, 0.0, 21),
}
print(f"{'centre':>9s}  {'paper: w0   w1   w2   w3  mean':>34s}  {'radarx: w0   w1   w2   w3  mean':>34s}")
for (x, y), row in table1.items():
    g = gbvtd(paper_sweep, (1000.0 * x, 1000.0 * y), radii=[1000.0 * row[5]], ring_width=1000.0)
    mine = [float(g[k].values[0]) for k in ("VT0", "VT1_AMPLITUDE", "VT2_AMPLITUDE", "VT3_AMPLITUDE", "VM_ALONG")]
    print(f"({x:2d}, {y:2d})   " + " ".join(f"{v:5.1f}" for v in row[:5]) + "    " + " ".join(f"{v:5.1f}" for v in mine))
```

The wavenumber-1, 2 and 3 amplitudes and the mean flow agree with the table to a few
tenths of a m/s (the apparent mean flow is zero when the centre is moved along the beam). The
wavenumber 0 is lower than the paper's for centres moved along the beam, by 1.6 to 4.8
m/s from 2 km on. The column is closer to the printed Eq. 20 ($+B_3$) than to the algebra of
Eqs. 15 and 18 that the code follows, but Fig. 4 of the same paper (below) is reproduced
only with the minus sign, and the maximum winds of Table 4 only with the order 2, so the
wavenumber-0 column of Table 1 stays unexplained.

**Table 2 of Part II** has the same quantities for maximum winds of 30 to 60 m/s: the apparent
wavenumber 1 is proportional to the maximum wind.

```{code-cell} ipython3
amp = {}
for vmax in (30.0, 40.0, 50.0, 60.0):
    sw = synthetic_vortex(center=(0.0, 60000.0), vmax=vmax, rmw=20000.0, elevation=0.0,
                          azimuth_step=0.5, range_step=250.0, max_range=140000.0)
    g = gbvtd(sw, (3000.0, 60000.0), radii=[22000.0], ring_width=1000.0)
    amp[vmax] = float(g.VT1_AMPLITUDE.values[0])
print("centre (3, 60) km, ring at 22 km, wavenumber 1 for 30, 40, 50, 60 m/s:",
      {k: round(v, 1) for k, v in amp.items()}, "paper 5.9, 7.8, 9.9, 11.7")
```

**Fig. 4 of Part II** starts the simplex at (-10, 50) km with an equilateral triangle of
circumradius 4 km. The maximum axisymmetric wind (rings of 15 to 25 km) at its vertices A, B,
C, at the reflection R1 of the worst vertex and at the expansion X1 is:

```{code-cell} ipython3
h = 0.5 * np.sqrt(3.0) * 4.0
A, B, C = (-10.0 - h, 48.0), (-10.0, 54.0), (-10.0 + h, 48.0)
mid = (0.5 * (B[0] + C[0]), 0.5 * (B[1] + C[1]))
vertices = {"A": A, "B": B, "C": C, "R1": (2 * mid[0] - A[0], 2 * mid[1] - A[1]),
            "X1": (3 * mid[0] - 2 * A[0], 3 * mid[1] - 2 * A[1])}
paper_vt = {"A": 21.35, "B": 29.60, "C": 34.15, "R1": 42.12, "X1": 44.81}
for name, (x, y) in vertices.items():
    g = gbvtd(paper_sweep, (1000.0 * x, 1000.0 * y), radii=np.arange(15000.0, 25001.0, 1000.0), ring_width=2000.0)
    print(f"{name:>2s} ({x:6.2f}, {y:5.2f}) km: radarx {float(g.VT0.max()):5.2f} m/s, paper {paper_vt[name]:5.2f} m/s")
```

**Tables 4 and 5 of Part II.** Eleven analytic vortices (the axisymmetric one with and
without radial wind, and asymmetries of 10 m/s at every radius for the wavenumbers 1, 2 and 3 with
the phases of the paper) are started from 36 points, 4 km apart, in a 20 km square around the
true centre; rings of 15 to 25 km, order 2, the ring with the highest wind (`selection="max_vt"`).
The paper's errors are 0.31 to 0.65 km, 0.34 km on average; with noise of 5 m/s 0.35 km.

```{code-cell} ipython3
cases = [("axisymmetric", None, False, 0.31, 48.9), ("with radial wind", None, True, 0.33, 48.8),
         ("wave 1 at 0", {1: 0.0}, False, 0.34, 48.8), ("wave 1 at 45", {1: 45.0}, False, 0.15, 47.0),
         ("wave 1 at 90", {1: 90.0}, False, 0.08, 46.3), ("wave 1 at 180", {1: 180.0}, False, 0.35, 49.0),
         ("wave 1 at 270", {1: 270.0}, False, 0.65, 51.4), ("wave 2 at 0", {2: 0.0}, False, 0.15, 52.6),
         ("wave 2 at 90", {2: 90.0}, False, 0.63, 45.1), ("wave 3 at 10", {3: 10.0}, False, 0.26, 47.8),
         ("wave 3 at 90", {3: 90.0}, False, 0.51, 50.9)]  # name, wave: {n: angle of the maximum}, radial wind, paper error (km), paper wave 0 (m/s)
for noise in (0.0, 5.0):
    errors = []
    print(f"noise {noise:.0f} m/s: {'case':>17s} {'error (km)':>11s} {'paper':>6s} {'wave 0':>7s} {'paper':>6s}")
    for name, wave, radial, paper_err, paper_w0 in cases:
        absolute = None if wave is None else {n: (10.0, 90.0 - angle) for n, angle in wave.items()}
        sw = synthetic_vortex(center=(0.0, 60000.0), vmax=50.0, rmw=20000.0, radial=RADIAL if radial else None,
                              absolute_asymmetry=absolute, elevation=0.0, azimuth_step=1.0, range_step=250.0,
                              max_range=140000.0, noise=noise, seed=0)
        res = tc_center(sw, first_guess=(0.0, 60000.0), radii=np.arange(15000.0, 25001.0, 1000.0), ring_width=2000.0,
                        max_wavenumber=2, guess_grid=6, guess_spacing=4000.0, search_radius=40000.0, selection="max_vt")
        err = np.hypot(float(res.CENTER_X), float(res.CENTER_Y) - 60000.0) / 1e3
        errors.append(err)
        print(f"{'':>17s}  {name:>17s} {err:11.2f} {paper_err:6.2f} {float(res.VT_MAX):7.1f} {paper_w0:6.1f}")
    print(f"{'':>17s}  mean error {np.mean(errors):.2f} km (paper 0.34 without and 0.35 with noise)")
```

The errors are about half of the paper's, with and without noise (the interpolation of the
paper adds error), and in the same direction, towards the radar. The maximum
winds follow the paper's to a few tenths of a m/s, but only with the order 2: the order 4 gives
49 m/s for the wavenumber-2 case at 0 degrees where the paper has 52.6.

**Section 7 of Part I.** With the series stopped at the wavenumber of the asymmetry (order
$n+1$), the wavenumber $n$ of 10 m/s at every radius is retrieved with an error below 20 percent
up to $\sin\alpha_{max}$ of 0.7, 0.5 and 0.33 for $n = 1, 2, 3$. The first radius (as $\sin\alpha_{max}$) at which the
amplitude of the wave in azimuth (after the distortion of the fit, the quantity of Fig. 13) is
off by more than 20 percent, for the phases of the asymmetry that go wrong first:

```{code-cell} ipython3
sines = np.arange(0.1, 0.96, 0.01)
for n, paper_limit in ((1, 0.7), (2, 0.5), (3, 0.33)):
    worst = np.zeros(sines.size)
    for angle in np.arange(0.0, 360.0 / n, 30.0):
        sw = synthetic_vortex(center=(0.0, 60000.0), vmax=50.0, rmw=20000.0, elevation=0.0, azimuth_step=1.0,
                              range_step=500.0, max_range=140000.0,
                              absolute_asymmetry={n: (10.0, 90.0 - angle)})
        g = gbvtd(sw, (0.0, 60000.0), radii=sines * 60000.0, ring_width=1000.0, max_wavenumber=n + 1)
        d = tc_decompose(g)
        bias = np.abs(d.AZIMUTHAL_AMPLITUDE.sel(wavenumber=n).values / 10.0 - 1.0)
        worst = np.maximum(worst, np.nan_to_num(bias, nan=9.0))
    print(f"wavenumber {n}: 20 percent error from sin(alpha_max) = {sines[np.argmax(worst > 0.2)]:.2f} (paper {paper_limit})")
```

**Section 6 of Part I.** A radial wind of the same wavenumber as the tangential wind cannot be
told apart from it by the closure. For 5 m/s in both, at the ring of 10 km, the retrieved
tangential wave ranges over the relative phases of the two winds as:

```{code-cell} ipython3
for n in (1, 2, 3):
    amplitude, phase_error = [], []
    for angle in np.arange(0.0, 360.0 / n, 15.0):  # angle of the maximum of the radial wind
        sw = synthetic_vortex(center=(0.0, 60000.0), vmax=50.0, rmw=20000.0, elevation=0.0, azimuth_step=0.5,
                              range_step=250.0, max_range=140000.0,
                              absolute_asymmetry={n: (5.0, 90.0)}, radial_asymmetry={n: (5.0, 90.0 - angle)})
        d = tc_decompose(gbvtd(sw, (0.0, 60000.0), radii=[10000.0], ring_width=1000.0))
        a = float(d.AZIMUTHAL_AMPLITUDE.sel(wavenumber=n).values[0])
        amplitude.append(a)
        if a > 1.0:  # the phase of a wave that vanishes is not defined
            p = (90.0 - float(d.AZIMUTHAL_PHASE.sel(wavenumber=n).values[0])) % (360.0 / n)
            phase_error.append((p + 180.0 / n) % (360.0 / n) - 180.0 / n)
    print(f"wavenumber {n}: amplitude {min(amplitude):.1f} to {max(amplitude):.1f} m/s (truth 5, paper 0 to 10), "
          f"phase error {min(phase_error):.0f} to {max(phase_error):.0f} degrees (paper +-{90 // n})")
```

## Symmetric and asymmetric winds

`tc_decompose` evaluates the Fourier coefficients at physical azimuths (the
exact relation between $\psi$ and the azimuth seen from the centre) and
returns the symmetric wind, the wavenumber-1 to 3 parts and their sum. It also
gives the winds in Cartesian components, the asymmetry relative to the symmetric
wind per radius, the radius of maximum wind per azimuth and, per radius, the
amplitude and azimuth of the waves of the sum in azimuth
(`AZIMUTHAL_AMPLITUDE` and `AZIMUTHAL_PHASE`), which are what Lee et al. (1999,
Figs. 13 to 15) compare with the analytic winds: the coefficients in $\psi$
(`VT1_AMPLITUDE`) are the quantities of Table 1 of Lee and Marks (2000).

```{code-cell} ipython3
dec = tc_decompose(gbvtd(sweep, found, radii=np.arange(2000.0, 60001.0, 1000.0), mean_wind=MEAN_WIND))
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), layout="constrained")
for ax, name, vmax in zip(axes, ["VT_SYMMETRIC", "VT_WAVENUMBER1", "VT_WAVENUMBER2", "VT"], [50, 12, 6, 50]):
    lo = 0 if name in ("VT_SYMMETRIC", "VT") else -vmax
    m = ax.pcolormesh(dec.X / 1e3, dec.Y / 1e3, dec[name], cmap="viridis" if lo == 0 else "RdBu_r",
                      vmin=lo, vmax=vmax, shading="auto")
    ax.set(aspect="equal", title=name.replace("_", " ").lower(), xlabel="east (km)")
    fig.colorbar(m, ax=ax, shrink=0.8, label="m/s")
axes[0].set_ylabel("north (km)")
plt.show()
ring = [15000.0, 25000.0, 35000.0]
print(f"asymmetry (rms of the asymmetric part over VT0) at 15, 25, 35 km: "
      f"{np.round(dec.ASYMMETRY.sel(radius=ring).values, 2)}")
print("wavenumber-1 amplitude in azimuth:",
      np.round(dec.AZIMUTHAL_AMPLITUDE.sel(wavenumber=1, radius=ring).values, 1),
      "truth", np.round(0.2 * vortex_profile(np.array(ring), VMAX, RMW, XT)[0], 1))
print("its azimuth (truth 45):", np.round(dec.AZIMUTHAL_PHASE.sel(wavenumber=1, radius=ring).values, 1))
print(f"radius of maximum wind per azimuth: {float(dec.RMW_AZIMUTH.min()) / 1e3:.1f} to "
      f"{float(dec.RMW_AZIMUTH.max()) / 1e3:.1f} km")
```

## Data gaps

The Fourier order that a ring supports depends on its largest angular gap: up to
30, 60 and 120 degrees allow the orders 4, 2 and 1 (Chen et al. 2013; 3 instead of
4 if `max_wavenumber=3`); a wider gap leaves the ring unretrieved (NaN).
Lee et al. (2000, Part III, Table 2) give 180, 90, 60 and 30 degrees for the
wavenumbers 0 to 3 and do not say whether the wavenumber is that of the vortex or the
order of the fit; the limits above agree with the second reading except for the order
1 (90 degrees in the table). `FIT_ORDER`, `DATA_GAP` and `N_POINTS` say what every
ring got; wavenumbers that the order does not resolve are NaN. Part III and Part I also
limit the order by the distance of the radar,
$\sin\alpha_{max}\le 1,\ 0.7,\ 0.5,\ 0.33$ for the wavenumbers 0 to 3 (the
wavenumbers are distorted by more than 20 percent beyond); `geometric_limit=True`
applies it.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(5.5, 3.4), layout="constrained")
for width, style in ((0.0, "k"), (40.0, "C0"), (90.0, "C1"), (140.0, "C3")):
    blank = [(180.0 - width / 2, 180.0 + width / 2)] if width else ()
    gapped = synthetic_vortex(center=CENTER, vmax=VMAX, rmw=RMW, xt=XT, mean_wind=MEAN_WIND,
                              asymmetry=ASYMMETRY, blank_sectors=blank, noise=2.0, seed=1)
    g = gbvtd(gapped, CENTER, radii=np.arange(8000.0, 40001.0, 2000.0), mean_wind=MEAN_WIND)
    order = int(np.nanmax(g.FIT_ORDER)) if g.FIT_ORDER.notnull().any() else None
    ax.plot(g.radius / 1e3, g.VT0, style, label=f"gap {width:.0f}°, order {order}")
ax.plot(radii / 1e3, vortex_profile(radii, VMAX, RMW, XT)[0], "0.7", lw=4, zorder=0, label="truth")
ax.set(xlabel="radius (km)", ylabel="VT0 (m/s)")
ax.legend(fontsize=8)
plt.show()
```

## Volumes, lazy data and the accessors

A volume (a `DataTree` with `sweep_*` nodes) is processed sweep by sweep and
returned as a tree; dask-backed data give lazy results; the functions are also
`.radarx` accessor methods.

```{code-cell} ipython3
root = xr.Dataset(coords={"latitude": 20.0, "longitude": 80.0, "altitude": 50.0})
leaf = sweep.drop_vars(["latitude", "longitude", "altitude"])
volume = xr.DataTree.from_dict({"/": root, "/sweep_0": leaf, "/sweep_1": leaf})
centres = volume.radarx.tc_center(min_radius=15000.0, max_radius=25000.0)  # a tree of results
winds = volume.radarx.gbvtd(centres, radii=np.arange(5000.0, 40001.0, 5000.0))  # one centre per sweep
lazy = sweep.chunk({"azimuth": 90}).radarx.gbvtd(CENTER, radii=np.arange(5000.0, 40001.0, 5000.0))
print(type(winds).__name__, list(winds.children), type(lazy.VT0.data).__name__)
```

## Compiled kernel and NumPy

```{code-cell} ipython3
small = dict(radii=np.array([18000.0, 20000.0, 22000.0]), guess_grid=2)
timing = {}
results = {}
for engine in ("compiled", "numpy"):
    t0 = time.perf_counter()
    results[engine] = tc_center(sweep, engine=engine, **small)
    timing[engine] = time.perf_counter() - t0
print({k: f"{v:.2f} s" for k, v in timing.items()})
print("same centre:", all(np.isclose(float(results["compiled"][k]), float(results["numpy"][k]), rtol=1e-6, atol=1e-3)
                         for k in ("CENTER_X", "CENTER_Y", "RMW", "VT_MAX")))
```

## Tropical Cyclone Tauktae (2021) seen by the Goa radar

The IMD C-band radar at Goa saw Tauktae on 15 May 2021, 00:58 UTC (the test data
of radarx, sweeps at 0.9 and 1.4 degrees). The position is the IBTrACS best track
interpolated to the time of the scan (NOAA NCEI IBTrACS v04r01, which has 3-hourly positions; the file has no best-track metadata). The
cyclone was then about 310 km away, so the beam is more than 10 km above the
surface, the Nyquist velocity is only 6.7 m/s and the velocity echoes do not reach
the core. `gbvtd` therefore returns nothing at the best-track position: no ring
has velocity data around enough of its circle (the largest gap must be at most 120
degrees). This is the result, and it shows what the function needs: velocity
coverage around the eyewall from a radar that is close enough to see it.

```{code-cell} ipython3
import pooch
from radarx.retrieve import dealias_velocity

registry = {
    "GOA210515005811-IMD-C.nc": "md5:45f692bbb9be9e5b3464bab7b52731f4",
    "GOA210515005811-IMD-C.nc.1": "md5:32504fa5035f61e49a384b6f9f515d7c",
}
fetcher = pooch.create(
    path=pooch.os_cache("radarx_data"),
    base_url="https://github.com/syedhamidali/pyscancf_examples/raw/main/data/goa_c/",
    registry=registry,
)
best_track = (12.85, 72.9)  # latitude, longitude, IBTrACS (Tauktae) interpolated to 00:58 UTC
goa = {}
for name in registry:
    ds = xr.open_dataset(fetcher.fetch(name), engine="imd")
    ds["VRADH"] = dealias_velocity(ds)
    goa[name] = ds.xradar.georeference()

lat0, lon0 = float(ds.latitude), float(ds.longitude)
bt_x = (best_track[1] - lon0) * 111.2 * np.cos(np.deg2rad(0.5 * (lat0 + best_track[0])))
bt_y = (best_track[0] - lat0) * 111.2
print(f"best track {np.hypot(bt_x, bt_y):.0f} km from the radar")
fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
for ax, (name, ds) in zip(axes, goa.items()):
    m = ax.pcolormesh(ds.x / 1e3, ds.y / 1e3, ds.VRADH, cmap="RdBu_r", vmin=-15, vmax=15, shading="auto")
    ax.plot(bt_x, bt_y, "k*", ms=14, label="best track")
    ax.set(xlim=(-450, 100), ylim=(-450, 100), aspect="equal", xlabel="east (km)",
           title=f"dealiased velocity, {float(ds.elevation.mean()):.1f}°")
ax.legend()
fig.colorbar(m, ax=axes, shrink=0.7, label="m/s")
plt.show()
```

```{code-cell} ipython3
ds = goa["GOA210515005811-IMD-C.nc"]
rings = gbvtd(ds, center_latlon=best_track, radii=np.arange(20000.0, 150001.0, 10000.0))
print("rings retrieved:", int(rings.VT0.notnull().sum()), "of", rings.sizes["radius"])
near = np.hypot(ds.x - bt_x * 1e3, ds.y - bt_y * 1e3) < 100e3
print(f"gates within 100 km of the best track: {int(near.sum())}, with a velocity: "
      f"{int((near & ds.VRADH.notnull()).sum())}")
guess = tc_center(ds, first_guess_latlon=best_track, min_radius=20000.0, max_radius=120000.0,
                  radius_step=5000.0, ring_width=5000.0, search_radius=60000.0, guess_spacing=10000.0)
print(f"tc_center returns {float(guess.VT_MAX):.1f} m/s as the largest tangential wind: not a vortex, "
      f"and the search is anchored at the first guess (limit {60} km)")
```

## Eyewall and ring diagnostics

`tc_ring_stats` fits a mean and the azimuthal wavenumbers 1 to 3 to any polar field on rings
around the centre; for the reflectivity it gives the radial profile of the eyewall and the
rainbands, their asymmetry and the fraction of every ring that has echo above a threshold.
`tc_structure` finds in a radial profile (`VT0` of `gbvtd`, or the `MEAN` of `tc_ring_stats`)
the primary maximum, its edges at half its prominence, the highest secondary maximum and the
moat between the two (the signature of concentric eyewalls, Willoughby et al. 1982), the
radii where the profile falls to given levels, and for the tangential wind the relative
vorticity $\zeta = r^{-1}\partial(rV)/\partial r$, the angular velocity and the angular
momentum (Kossin and Eastin 2001 describe the two regimes of the vorticity profile of the
eyewall). The thresholds are radarx's own. A vortex with the decay exponent 1 is a Rankine vortex, whose
vorticity is $2V_{max}/R_{max}$ inside the radius of maximum wind and zero outside; its retrieval gives

```{code-cell} ipython3
from radarx.retrieve import tc_ring_stats, tc_structure

rankine = synthetic_vortex(center=CENTER, vmax=VMAX, rmw=RMW, xt=1.0, azimuth_step=1.0, max_range=130000.0)
rankine_winds = gbvtd(rankine, CENTER, radii=np.arange(4000.0, 45001.0, 1000.0))
rankine_structure = tc_structure(rankine_winds, levels=[25.0])
print(f"maximum {float(rankine_structure.PEAK_VALUE):.1f} m/s at {float(rankine_structure.PEAK_RADIUS) / 1e3:.1f} km "
      f"(truth {VMAX:.0f} at {RMW / 1e3:.0f}), eyewall {float(rankine_structure.INNER_EDGE) / 1e3:.1f} to "
      f"{float(rankine_structure.OUTER_EDGE) / 1e3:.1f} km, 25 m/s at "
      f"{float(rankine_structure.OUTER_LEVEL_RADIUS[0]) / 1e3:.1f} km (truth {2 * RMW / 1e3:.0f}), vorticity at 10 km "
      f"{float(rankine_structure.VORTICITY.sel(radius=10000.0)) * 1e3:.2f}e-3 1/s (truth {2 * VMAX / RMW * 1e3:.2f}e-3)")
```

## Hurricane Harvey (2017) at landfall seen by Corpus Christi

Harvey made landfall near Rockport, Texas, at about 03 UTC on 26 August 2017. The eye was
about 60 km north-east of the NEXRAD radar at Corpus Christi (KCRP), close enough for the
rings of GBVTD to close. The volume of 03:04 UTC is one of the open AWS NEXRAD archive. The
Nyquist velocity (27.05 m/s) is read from the radial headers and the lowest Doppler cut is
dealiased with UNRAVEL (the winds are more than twice the Nyquist velocity). The National
Hurricane Center best track (HURDAT2, [Landsea and Franklin 2013](https://doi.org/10.1175/MWR-D-12-00254.1))
has at 03 UTC the position 28.0 N, 96.9 W, a maximum wind of 115 kt (59 m/s) and a radius of maximum
wind of 10 n mi (18.5 km); the radar sees the wind at 0.5 to 1 km above the surface, so the
numbers are close, not equal.

```{code-cell} ipython3
import xradar as xd
from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

from radarx.io.aws_data import download_file
from radarx.retrieve import dealias_velocity

kcrp = download_file("unidata-nexrad-level2", "2017/08/26/KCRP/KCRP20170826_030439_V06", ".")
tree = xd.io.open_nexradlevel2_datatree(kcrp)
with NEXRADLevel2File(kcrp) as nf:
    nyq = [h["msg_31_data_header"]["RAD"]["nyquist_vel"] / 100.0 for h in nf.msg_31_data_header]
sweeps = [n for n in tree.children if n.startswith("sweep")]
harvey = tree["sweep_1"].to_dataset(inherit="all_coords").sortby("azimuth")
harvey = harvey.assign_coords(nyquist_velocity=nyq[sweeps.index("sweep_1")])
harvey["VRADH"] = harvey["VRADH"].where(harvey["VRADH"] > -63.9)
harvey["DBZH"] = harvey["DBZH"].where(harvey["DBZH"] > -32.0)
harvey["VRADH"] = dealias_velocity(harvey, "VRADH", method="unravel")
best_track = dict(position=(28.0, -96.9), vmax=115 * 0.5144, rmw=10 * 1852.0)  # HURDAT2, 03 UTC

found = tc_center(harvey, first_guess_latlon=best_track["position"], min_radius=5000.0, max_radius=60000.0,
                  search_radius=40000.0)
lat0, lon0 = float(harvey.latitude), float(harvey.longitude)
bt_y = (best_track["position"][0] - lat0) * 111.2e3
bt_x = (best_track["position"][1] - lon0) * 111.2e3 * np.cos(np.deg2rad(0.5 * (lat0 + best_track["position"][0])))
print(f"centre {float(found.CENTER_LATITUDE):.2f} N, {-float(found.CENTER_LONGITUDE):.2f} W, "
      f"{np.hypot(float(found.CENTER_X) - bt_x, float(found.CENTER_Y) - bt_y) / 1e3:.1f} km from the best track "
      f"(uncertainty {float(found.CENTER_UNCERTAINTY):.0f} m); RMW {float(found.RMW) / 1e3:.1f} km (best track "
      f"{best_track['rmw'] / 1e3:.1f}), maximum {float(found.VT_MAX):.1f} m/s (best track {best_track['vmax']:.1f})")
```

```{code-cell} ipython3
hv_radii = np.arange(3000.0, 100001.0, 1000.0)
hv_winds = gbvtd(harvey, found, radii=hv_radii, max_wavenumber=4)
wind_profile = tc_structure(hv_winds, levels=[50.0])
dbz_rings = tc_ring_stats(harvey, found, field="DBZH", radii=hv_radii, max_wavenumber=2, threshold=35.0)
dbz_profile = tc_structure(dbz_rings, prominence=5.0)
print(f"wind: maximum {float(wind_profile.PEAK_VALUE):.1f} m/s at {float(wind_profile.PEAK_RADIUS) / 1e3:.1f} km, "
      f"eyewall {float(wind_profile.INNER_EDGE) / 1e3:.0f} to {float(wind_profile.OUTER_EDGE) / 1e3:.0f} km, "
      f"maximum vorticity {float(wind_profile.VORTICITY.max()) * 1e3:.1f}e-3 1/s at {float(wind_profile.VORTICITY.idxmax()) / 1e3:.0f} km")
print(f"reflectivity: maximum of the azimuthal mean {float(dbz_profile.PEAK_VALUE):.1f} dBZ at "
      f"{float(dbz_profile.PEAK_RADIUS) / 1e3:.1f} km, wavenumber 1 {float(dbz_rings.WAVE1_AMPLITUDE.sel(radius=20000.0)):.1f} dB "
      f"towards {float(dbz_rings.WAVE1_AZIMUTH.sel(radius=20000.0)):.0f} deg at 20 km")
```

```{code-cell} ipython3
polar = tc_decompose(hv_winds, azimuth=np.arange(0.0, 360.0, 5.0))
fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout="constrained")
ax = axes[0, 0]
geo = harvey.xradar.georeference()
m = ax.pcolormesh(geo.x / 1e3, geo.y / 1e3, geo.VRADH, cmap="RdBu_r", vmin=-60, vmax=60, shading="nearest")
ax.plot(float(found.CENTER_X) / 1e3, float(found.CENTER_Y) / 1e3, "k*", ms=12)
ax.plot(bt_x / 1e3, bt_y / 1e3, "wx", ms=10)
ax.set(xlim=(-60, 140), ylim=(-70, 130), aspect="equal", title="dealiased velocity, 0.5 deg (* centre, x best track)")
fig.colorbar(m, ax=ax, label="m/s")
ax = axes[0, 1]
ax.plot(hv_radii / 1e3, hv_winds.VT0, label="tangential")
ax.plot(hv_radii / 1e3, hv_winds.VR0, label="radial")
ax.axvline(best_track["rmw"] / 1e3, color="k", ls=":", label="best-track RMW")
ax.set(xlabel="radius (km)", ylabel="axisymmetric wind (m/s)", title="GBVTD")
ax.legend()
ax = axes[1, 0]
ax.plot(hv_radii / 1e3, dbz_rings.MEAN, label="mean")
ax.fill_between(hv_radii / 1e3, dbz_rings.MEAN - dbz_rings.STD, dbz_rings.MEAN + dbz_rings.STD, alpha=0.2)
ax.plot(hv_radii / 1e3, dbz_rings.WAVE1_AMPLITUDE, label="wavenumber 1")
ax.set(xlabel="radius (km)", ylabel="DBZH (dBZ)", title="reflectivity on rings")
ax.legend()
ax = axes[1, 1]
ax.plot(hv_radii / 1e3, wind_profile.VORTICITY * 1e3, label="vorticity (1e-3 1/s)")
ax.plot(hv_radii / 1e3, polar.ASYMMETRY, label="asymmetry")
ax.set(xlabel="radius (km)", title="axisymmetric vorticity and relative asymmetry")
ax.legend()
plt.show()
```

The retrieval stops at 62 km, the distance of the centre from the radar: a ring is only fitted
where the radar is outside it, so the wind radii of the gale and hurricane forces of the best track
(more than 60 km) are beyond what one radar at this distance gives. The axisymmetric wind falls
from its maximum of about 60 m/s at 22 km to 36 m/s at 60 km, and the vorticity has its maximum
inside the radius of maximum wind, in the eyewall ring of the second regime of Kossin and
Eastin (2001).

## References

- Bell, M. M., and W.-C. Lee, 2012: Objective tropical cyclone center tracking using single-Doppler radar. *J. Appl. Meteor. Climatol.*, **51** (5), 878-896, [doi:10.1175/JAMC-D-11-0167.1](https://doi.org/10.1175/JAMC-D-11-0167.1).
- Chen, X., K. Zhao, W.-C. Lee, B. J.-D. Jou, M. Xue, and P. R. Harasti, 2013: The improvement to the environmental wind and tropical cyclone circulation retrievals with the modified GBVTD (MGBVTD) technique. *J. Appl. Meteor. Climatol.*, **52** (11), 2493-2508, [doi:10.1175/JAMC-D-13-031.1](https://doi.org/10.1175/JAMC-D-13-031.1).
- Lee, W.-C., B. J.-D. Jou, P.-L. Chang, and S.-M. Deng, 1999: Tropical cyclone kinematic structure retrieved from single-Doppler radar observations. Part I: Interpretation of Doppler velocity patterns and the GBVTD technique. *Mon. Wea. Rev.*, **127** (10), 2419-2439, [doi:10.1175/1520-0493(1999)127<2419:TCKSRF>2.0.CO;2](https://doi.org/10.1175/1520-0493(1999)127<2419:TCKSRF>2.0.CO;2).
- Lee, W.-C., and F. D. Marks, 2000: Tropical cyclone kinematic structure retrieved from single-Doppler radar observations. Part II: The GBVTD-simplex center finding algorithm. *Mon. Wea. Rev.*, **128** (6), 1925-1936, [doi:10.1175/1520-0493(2000)128<1925:TCKSRF>2.0.CO;2](https://doi.org/10.1175/1520-0493(2000)128<1925:TCKSRF>2.0.CO;2).
- Lee, W.-C., B. J.-D. Jou, P.-L. Chang, and F. D. Marks, 2000: Tropical cyclone kinematic structure retrieved from single-Doppler radar observations. Part III: Evolution and structures of Typhoon Alex (1987). *Mon. Wea. Rev.*, **128** (12), 3982-4001, [doi:10.1175/1520-0493(2000)129<3982:TCKSRF>2.0.CO;2](https://doi.org/10.1175/1520-0493(2000)129<3982:TCKSRF>2.0.CO;2).
- Lee, W.-C., P. R. Harasti, M. Bell, B. J.-D. Jou, and M.-H. Chang, 2006: Doppler velocity signatures of idealized elliptical vortices. *Terr. Atmos. Ocean. Sci.*, **17** (2), 429-446, [doi:10.3319/TAO.2006.17.2.429(A)](https://doi.org/10.3319/TAO.2006.17.2.429(A)).
- Liou, Y.-C., T.-C. C. Wang, W.-C. Lee, and Y.-J. Chang, 2005: The extended GBVTD and the recovery of the tropical cyclones asymmetric structures. *Preprints, 32nd Conf. on Radar Meteorology*, Albuquerque, NM, Amer. Meteor. Soc., JP2J.8, [conference paper](https://ams.confex.com/ams/pdfpapers/95858.pdf).
- Murillo, S. T., W.-C. Lee, M. M. Bell, F. D. Marks Jr., P. P. Dodge, and G. M. Barnes, 2011: Intercomparison of ground-based velocity track display (GBVTD)-retrieved circulation centers and structures of Hurricane Danny (1997) from two coastal WSR-88Ds. *Mon. Wea. Rev.*, **139**, 153-174. (Cited by Bell and Lee 2012; not read.)
- Nelder, J. A., and R. Mead, 1965: A simplex method for function minimization. *Comput. J.*, **7** (4), 308-313, [doi:10.1093/comjnl/7.4.308](https://doi.org/10.1093/comjnl/7.4.308).
- Knapp, K. R., M. C. Kruk, D. H. Levinson, H. J. Diamond, and C. J. Neumann, 2010: The International Best Track Archive for Climate Stewardship (IBTrACS). *Bull. Amer. Meteor. Soc.*, **91** (3), 363-376, [doi:10.1175/2009BAMS2755.1](https://doi.org/10.1175/2009BAMS2755.1). (Best-track position of Tauktae, NOAA NCEI IBTrACS v04r01.)
- Kossin, J. P., and M. D. Eastin, 2001: Two distinct regimes in the kinematic and thermodynamic structure of the hurricane eye and eyewall. *J. Atmos. Sci.*, **58** (9), 1079-1090, [doi:10.1175/1520-0469(2001)058<1079:TDRITK>2.0.CO;2](https://doi.org/10.1175/1520-0469(2001)058<1079:TDRITK>2.0.CO;2).
- Landsea, C. W., and J. L. Franklin, 2013: Atlantic hurricane database uncertainty and presentation of a new database format. *Mon. Wea. Rev.*, **141** (10), 3576-3592, [doi:10.1175/MWR-D-12-00254.1](https://doi.org/10.1175/MWR-D-12-00254.1).
- Willoughby, H. E., J. A. Clos, and M. G. Shoreibah, 1982: Concentric eye walls, secondary wind maxima, and the evolution of the hurricane vortex. *J. Atmos. Sci.*, **39** (2), 395-411, [doi:10.1175/1520-0469(1982)039<0395:CEWSWM>2.0.CO;2](https://doi.org/10.1175/1520-0469(1982)039<0395:CEWSWM>2.0.CO;2).
