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

# T-matrix Scattering: Validation at Six Radar Bands

`radarx.fundamentals.raindrop_scattering` and `spheroid_scattering` are a C++
translation of the FORTRAN T-matrix code of Mishchenko. Matching the FORTRAN code
proves the translation, not the physics, so this notebook checks the kernel against
results that do not come from that code, at the wavelengths of the radars that
observe rain:

| Band | Frequency | Wavelength | Used by |
|---|---|---|---|
| S | 2.8 GHz | 107 mm | NEXRAD |
| C | 5.6 GHz | 53.5 mm | most national networks |
| X | 9.4 GHz | 31.9 mm | mobile and gap-filling radars |
| Ku | 13.6 GHz | 22.1 mm | GPM dual-frequency precipitation radar |
| Ka | 35.5 GHz | 8.4 mm | GPM, cloud and precipitation radars |
| W | 94 GHz | 3.19 mm | cloud radars |

The checks are the following. Each one is also a unit test
(`tests/fundamentals/test_tmatrix_physics.py`).

1. A sphere is solved exactly by the Mie theory. An independent Mie code (written
   below) must agree with the kernel for the amplitudes at every angle and for the
   extinction.
2. The optical theorem ties the forward amplitude to the energy removed from the
   beam. For a particle without absorption the extinction cross section must equal
   the integral of the scattering over all directions.
3. Reciprocity relates the scattering from direction $n_0$ to $n$ with that from
   $-n$ to $-n_0$.
4. Turning the particle and all directions together about the vertical changes nothing.
5. Particles much smaller than the wavelength follow the Rayleigh law with the
   depolarization factors of a spheroid, and depart from it with the square of the size.
6. The result must stop changing when the expansion is made more accurate.

```{code-cell} ipython3
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from radarx.fundamentals import (
    axis_ratio_brandes,
    raindrop_scattering,
    spheroid_scattering,
    water_refractive_index,
)

BANDS = {"S": 107.0, "C": 53.5, "X": 31.9, "Ku": 22.1, "Ka": 8.4, "W": 3.19}  # mm
pd.set_option("display.float_format", "{:.1e}".format)
```

## A sphere against the Mie theory

The Mie coefficients below follow Bohren and Huffman (1983), with the logarithmic
derivative computed by downward recurrence. They share no code with the T-matrix kernel.

```{code-cell} ipython3
def mie_coefficients(m, x):
    nmax = int(round(x + 4.05 * x ** (1 / 3) + 2))
    mx = m * x
    nmx = int(round(max(nmax, abs(mx)) + 16))
    d = np.zeros(nmx + 1, dtype=complex)
    for n in range(nmx, 0, -1):
        d[n - 1] = n / mx - 1.0 / (d[n] + n / mx)
    a = np.zeros(nmax, dtype=complex)
    b = np.zeros(nmax, dtype=complex)
    psi0, psi1 = np.cos(x), np.sin(x)
    chi0, chi1 = -np.sin(x), np.cos(x)
    xi1 = complex(psi1, -chi1)
    for n in range(1, nmax + 1):
        psi = (2 * n - 1) / x * psi1 - psi0
        chi = (2 * n - 1) / x * chi1 - chi0
        xi = complex(psi, -chi)
        a[n - 1] = ((d[n] / m + n / x) * psi - psi1) / ((d[n] / m + n / x) * xi - xi1)
        b[n - 1] = ((d[n] * m + n / x) * psi - psi1) / ((d[n] * m + n / x) * xi - xi1)
        psi0, psi1, chi0, chi1 = psi1, psi, chi1, chi
        xi1 = complex(psi1, -chi1)
    return a, b


def mie_amplitudes(m, x, theta):
    a, b = mie_coefficients(m, x)
    mu = np.cos(np.radians(np.atleast_1d(theta)))
    s1 = np.zeros(mu.shape, dtype=complex)
    s2 = np.zeros(mu.shape, dtype=complex)
    pi0, pi1 = np.zeros_like(mu), np.ones_like(mu)
    for n in range(1, len(a) + 1):
        tau = n * mu * pi1 - (n + 1) * pi0
        f = (2 * n + 1) / (n * (n + 1))
        s1 += f * (a[n - 1] * pi1 + b[n - 1] * tau)
        s2 += f * (a[n - 1] * tau + b[n - 1] * pi1)
        pi0, pi1 = pi1, ((2 * n + 1) * mu * pi1 - (n + 1) * pi0) / n
    return s1, s2


def mie_extinction(m, x):
    a, b = mie_coefficients(m, x)
    n = np.arange(1, len(a) + 1)
    return 2 / x**2 * np.sum((2 * n + 1) * (a + b).real)
```

The kernel returns the amplitude matrix $S$ and the phase matrix $Z$ of
Mishchenko (2000). For a sphere the element $Z_{11}$ is
$(|S_1|^2 + |S_2|^2) / (2k^2)$, and the extinction cross section follows from the
forward amplitude, $C_{ext} = (4\pi / k)\, \mathrm{Im}\, S(0)$. Drops of 0.5, 3 and 8 mm
are compared at five scattering angles at every band, with the refractive index of
water at 10 degrees Celsius.

```{code-cell} ipython3
angles = np.array([0.0, 30.0, 90.0, 150.0, 180.0])
geometry = np.array([[90.0, 90.0, 0.0, a] for a in angles])
rows = []
for band, wavelength in BANDS.items():
    m = water_refractive_index(wavelength, 10.0)
    k = 2 * np.pi / wavelength
    for diameter in (0.5, 3.0, 8.0):
        radius = diameter / 2
        s, z, nmax = spheroid_scattering(
            radius, 1.0, wavelength, m, geometry, [[0.0, 0.0, 1.0]]
        )
        s1, s2 = mie_amplitudes(m, k * radius, angles)
        z11 = 0.5 * (abs(s1) ** 2 + abs(s2) ** 2) / k**2
        extinction = 4 * np.pi / k * s[0, 0, 0, 0].imag
        reference = mie_extinction(m, k * radius) * np.pi * radius**2
        rows.append(
            dict(
                band=band,
                diameter=diameter,
                size_parameter=k * radius,
                order=int(nmax[0]),
                phase_function=np.abs(z[0, :, 0, 0] - z11).max() / z11.max(),
                extinction=abs(extinction / reference - 1),
            )
        )
mie = pd.DataFrame(rows)
mie
```

The largest departure is $2 \times 10^{-9}$ of the largest value of the phase function
and $10^{-9}$ of the extinction, and it grows only with the size parameter. The
backscatter cross section of the horizontally and vertically polarized wave of a
sphere is the same, and equals $4\pi|S_1(180^\circ)|^2/k^2$, so the Mie result also fixes the
quantity that the radar measures.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6.5, 3.6), layout="constrained")
for band, wavelength in BANDS.items():
    m = water_refractive_index(wavelength, 10.0)
    k = 2 * np.pi / wavelength
    diameter = np.linspace(0.2, 8.0, 40)
    sigma = raindrop_scattering(
        diameter, wavelength, temperature=10.0, canting_std=0.5, axis_ratio=1.0
    )
    s1 = np.array([mie_amplitudes(m, k * d / 2, [180.0])[0][0] for d in diameter])
    reference = 4 * np.pi * np.abs(s1) ** 2 / k**2
    # relative to the largest value of the band: the Mie solution has nulls at Ka and W band
    ax.semilogy(
        diameter, np.abs(sigma.sigma_h.values - reference) / reference.max() + 1e-16, label=band
    )
ax.set(
    xlabel="diameter (mm)",
    ylabel="|σ$_b$(T-matrix) − σ$_b$(Mie)| / max σ$_b$(Mie)",
    title="Backscatter cross section of spheres, all six bands",
)
ax.legend(ncols=3)
ax.grid(alpha=0.3)
```

The figure runs the same comparison through `raindrop_scattering`, which averages
over the canting of the drop (a sphere does not care) and returns the backscatter
cross section directly. The difference is divided by the largest cross section of
the band, because the backscatter of a sphere has nulls at Ka and W band (a drop of
about 4 mm at Ka band) where a relative difference is meaningless. Values near
$10^{-14}$ are rounding errors, and the steps are the points where the order of the
expansion increases.

## The optical theorem: scattering and extinction of a spheroid

For a particle that does not absorb, all the energy taken from the beam is scattered,
so the integral of the phase matrix over all directions must equal the extinction
computed from the forward amplitude. This is a check of the whole scattering matrix
of a *non-spherical* particle in an oblique orientation, and the Mie theory cannot
give it. The refractive index is that of water with an imaginary part of $10^{-12}$.
The two polarizations are the incident waves polarized along the first and second axis
of the basis; for each, the integral of $Z_{11} \pm Z_{12}$ is compared with the
extinction of the corresponding diagonal amplitude.

```{code-cell} ipython3
def sphere_quadrature(n_theta, n_phi):
    nodes, weights = np.polynomial.legendre.leggauss(n_theta)
    theta = np.degrees(np.arccos(nodes))
    phi = (np.arange(n_phi) + 0.5) * 360.0 / n_phi
    t, p = np.meshgrid(theta, phi, indexing="ij")
    w = np.outer(weights, np.full(n_phi, 2 * np.pi / n_phi))
    return t.ravel(), p.ravel(), w.ravel()


m_lossless = complex(1.33, 1e-12)
rows = []
for band, wavelength in BANDS.items():
    k = 2 * np.pi / wavelength
    for diameter, ratio in ((1.0, 1.1), (4.0, 1.4), (7.0, 1.8)):
        radius = diameter / 2
        theta, phi, w = sphere_quadrature(int(4 * np.pi * radius / wavelength) + 24, 48)
        geometry = np.vstack(
            [
                [90.0, 90.0, 0.0, 0.0],
                np.column_stack([np.full_like(theta, 90.0), theta, np.zeros_like(theta), phi]),
            ]
        )
        s, z, _ = spheroid_scattering(
            radius, ratio, wavelength, m_lossless, geometry, [[37.0, 63.0, 1.0]], ddelt=1e-4, ndgs=4
        )
        zs = z[0, 1:]
        scattering = (np.sum(w * (zs[:, 0, 0] + zs[:, 0, 1])), np.sum(w * (zs[:, 0, 0] - zs[:, 0, 1])))
        extinction = (4 * np.pi / k * s[0, 0, 0, 0].imag, 4 * np.pi / k * s[0, 0, 1, 1].imag)
        rows.append(
            dict(
                band=band,
                diameter=diameter,
                axis_ratio=ratio,
                first_axis=abs(scattering[0] / extinction[0] - 1),
                second_axis=abs(scattering[1] / extinction[1] - 1),
            )
        )
optical = pd.DataFrame(rows)
optical
```

The two sides agree to better than $2 \times 10^{-6}$ at every band and for oblate
spheroids of axis ratio up to 1.8 with the axis tilted 63 degrees from the vertical.
A drop with absorption removes more than it scatters: its extinction exceeds the
scattering by the absorbed energy, which is positive at every size and band.

## Reciprocity and rotation

Reciprocity of the amplitude matrix in the basis of Mishchenko (2000) reads
$S_{11}(n, n_0) = S_{11}(-n_0, -n)$, $S_{22}(n, n_0) = S_{22}(-n_0, -n)$ and
$S_{12}(n, n_0) = -S_{21}(-n_0, -n)$, where the sign comes from the azimuthal unit
vector changing sign under inversion. In the backscatter direction $-n_0 = n$, so
$S_{12} = -S_{21}$, the property that makes the radar measure a symmetric
quantity. Both are tested for drops in random orientations. The rotation test
turns the particle ($\alpha$) and both directions by the same azimuth.

```{code-cell} ipython3
rng = np.random.default_rng(5)
rows = []
for band, wavelength in BANDS.items():
    m = water_refractive_index(wavelength, 10.0)
    back, general, turned = 0.0, 0.0, 0.0
    for diameter in (1.0, 3.0, 6.0):
        ratio = 1.0 / float(axis_ratio_brandes(diameter))
        alpha, beta = rng.uniform(0, 360), rng.uniform(0, 180)
        s, _, _ = spheroid_scattering(
            diameter / 2, ratio, wavelength, m, [[90, 90, 0, 180]], [[alpha, beta, 1.0]]
        )
        back = max(back, abs(s[0, 0, 0, 1] + s[0, 0, 1, 0]) / np.abs(s).max())
        t0, t = rng.uniform(10, 170, 2)
        p0, p = rng.uniform(0, 180), rng.uniform(180, 360)
        kw = dict(ddelt=1e-6, ndgs=8)
        a, _, _ = spheroid_scattering(
            diameter / 2, ratio, wavelength, m, [[t0, t, p0, p]], [[alpha, beta, 1.0]], **kw
        )
        b, _, _ = spheroid_scattering(
            diameter / 2,
            ratio,
            wavelength,
            m,
            [[180 - t, 180 - t0, (p + 180) % 360, (p0 + 180) % 360]],
            [[alpha, beta, 1.0]],
            **kw,
        )
        a, b = a[0, 0], b[0, 0]
        general = max(
            general,
            max(
                abs(a[0, 0] - b[0, 0]),
                abs(a[1, 1] - b[1, 1]),
                abs(a[0, 1] + b[1, 0]),
                abs(a[1, 0] + b[0, 1]),
            )
            / np.abs(a).max(),
        )
        turn = 123.0
        c, _, _ = spheroid_scattering(
            diameter / 2, ratio, wavelength, m, [[70, 100, 20, 200]], [[40.0, 70.0, 1.0]]
        )
        e, _, _ = spheroid_scattering(
            diameter / 2,
            ratio,
            wavelength,
            m,
            [[70, 100, 20 + turn, 200 + turn]],
            [[40.0 + turn, 70.0, 1.0]],
        )
        turned = max(turned, np.abs(e - c).max() / np.abs(c).max())
    rows.append(dict(band=band, backscatter=back, general_directions=general, rotation=turned))
pd.DataFrame(rows)
```

Backscatter reciprocity holds to $10^{-8}$ at every band, and the general relation
to $10^{-6}$ when the expansion is converged tightly (with the default accuracy
the violation at W band grows to $10^{-3}$ for the largest drops in directions far from
backscatter, which is the accuracy of the expansion, shown in the last section). The
rotation changes the result by less than $10^{-6}$ (here by rounding error).

## The Rayleigh limit

For a particle whose size is much smaller than the wavelength, the field inside is
uniform and the backscatter cross section is
$\sigma_b = 4\pi k^4 |\alpha_i|^2$ with the polarizability

$$\alpha_i = \frac{V}{4\pi}\,\frac{\varepsilon - 1}{1 + L_i(\varepsilon - 1)}$$

along each axis. The depolarization factors of an oblate spheroid with the
axis ratio $r > 1$ and $e^2 = r^2 - 1$ are
$L_z = (1 + e^2)(1 - \arctan e / e) / e^2$ and $L_x = (1 - L_z) / 2$. For a
vertical symmetry axis and horizontal propagation the horizontally polarized wave
sees $L_x$ and the vertically polarized wave $L_z$. The first correction to this
law is of the order of $x^2$, with $x = 2\pi a / \lambda$, so the
relative departure must fall with the square of the size.

```{code-cell} ipython3
def depolarization(ratio):
    e2 = ratio**2 - 1.0
    e = np.sqrt(e2)
    lz = (1 + e2) / e2 * (1 - np.arctan(e) / e)
    return (1 - lz) / 2, lz


x_values = np.array([0.01, 0.02, 0.04, 0.08, 0.16])
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained", sharey=True)
for band, wavelength in BANDS.items():
    m = water_refractive_index(wavelength, 20.0)
    eps = m * m
    k = 2 * np.pi / wavelength
    for pol, ax in zip(("horizontal", "vertical"), axes):
        departure = []
        for x in x_values:
            radius = x / k
            ratio = 1.5
            _, z, _ = spheroid_scattering(radius, ratio, wavelength, m, [[90, 90, 0, 180]], [[0.0, 0.0, 1.0]])
            z = z[0, 0]
            sign = -1.0 if pol == "horizontal" else 1.0
            sigma = 2 * np.pi * (z[0, 0] + sign * z[0, 1] + sign * z[1, 0] + z[1, 1])
            lx, lz = depolarization(ratio)
            volume = 4 / 3 * np.pi * radius**3
            li = lx if pol == "horizontal" else lz
            rayleigh = 4 * np.pi * k**4 * abs(volume * (eps - 1) / (1 + li * (eps - 1)) / (4 * np.pi)) ** 2
            departure.append(abs(sigma / rayleigh - 1))
        ax.loglog(x_values, departure, "o-", label=band)
for pol, ax in zip(("horizontal", "vertical"), axes):
    ax.loglog(x_values, 0.4 * x_values**2, "k--", label="∝ $x^2$")
    ax.set(xlabel="size parameter $x$", title=f"{pol} polarization")
    ax.grid(alpha=0.3, which="both")
axes[0].set_ylabel("|σ$_b$ / σ$_b$(Rayleigh) − 1|")
axes[0].legend(ncols=2, fontsize=8)
```

The departure of every band and both polarizations falls parallel to the $x^2$
line (the dashed line only shows the slope). The size of the departure is set by
$|\varepsilon|\,x^2$, so it is smaller at the shorter wavelengths, where the
refractive index of water is smaller.

## Convergence of the expansion

The T-matrix method expands the field in multipoles up to an order
`nmax` that the code increases until the scattering cross section changes by less
than `ddelt`. The FORTRAN code recommends `ddelt = 1e-3`, which converges the total
scattering, but the radar uses the backscatter cross section and
the forward amplitude, which converge more slowly, in particular at the short
wavelengths where the vertically polarized backscatter of a large drop passes
through minima. The reference below uses `ddelt = 1e-7`.

```{code-cell} ipython3
diameter = np.round(np.arange(0.2, 8.01, 0.1), 2)
rows = []
tables = {}
for band, wavelength in BANDS.items():
    reference = raindrop_scattering(diameter, wavelength, ddelt=1e-7, ndgs=8)
    for ddelt in (1e-3, 1e-4, 1e-5):
        out = raindrop_scattering(diameter, wavelength, ddelt=ddelt)
        tables[band, ddelt] = out
        scale = float(np.abs(reference.kdp).max())
        rows.append(
            dict(
                band=band,
                ddelt=ddelt,
                sigma_h=float(np.abs(out.sigma_h / reference.sigma_h - 1).max()),
                sigma_v=float(np.abs(out.sigma_v / reference.sigma_v - 1).max()),
                ah=float(np.abs(out.ah / reference.ah - 1).max()),
                kdp_vs_largest=float(np.abs(out.kdp - reference.kdp).max() / scale),
                nmax=int(out.nmax.max()),
            )
        )
convergence = pd.DataFrame(rows).set_index(["band", "ddelt"])
convergence
```

With `ddelt = 1e-3` the vertically polarized backscatter of a drop at W band is wrong
by up to 25 % in a minimum, $K_{DP}$ at Ka and W band by 1 % of its largest value,
and at Ku band by 23 % of its own value at the diameter where it passes through zero. With `ddelt = 1e-5`, the default of
`raindrop_scattering`, the relative error of every quantity is below
$10^{-3}$ and the cost is about twice as much (see the timing below). The packaged scattering tables at S, C and X band keep
`ddelt = 1e-3` as the FORTRAN code and `pytmatrix` produced them (the error is below
$10^{-3}$ at these bands).

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
reference = raindrop_scattering(diameter, BANDS["W"], ddelt=1e-7, ndgs=8)
for ddelt in (1e-3, 1e-4, 1e-5):
    out = tables["W", ddelt]
    ax.semilogy(diameter, np.abs(out.sigma_v / reference.sigma_v - 1) + 1e-12, label=f"ddelt = {ddelt:g}")
ax.set(
    xlabel="diameter (mm)",
    ylabel="|σ$_v$ / σ$_v$(reference) − 1|",
    title="Vertically polarized backscatter at W band",
)
ax.legend()
ax.grid(alpha=0.3)
```

## Speed

The kernel computes one drop per thread. A table of 80 drops at every band takes:

```{code-cell} ipython3
diameter = np.linspace(0.2, 8.0, 80)
for n_threads in (1, None):
    start = time.perf_counter()
    for wavelength in BANDS.values():
        raindrop_scattering(diameter, wavelength, n_threads=n_threads)
    label = "one thread" if n_threads == 1 else "all cores"
    print(f"{label}: {time.perf_counter() - start:.2f} s for the six tables")
```

## What the checks establish

- The kernel is exact for the sphere (Mie, 1e-9) at the six bands, including the
  resonances of the large drops at Ka and W band.
- It conserves energy for the spheroid in an oblique orientation (optical theorem,
  1e-5), obeys reciprocity (1e-8 in backscatter), is rotation invariant (2e-7) and
  has the right Rayleigh limit with the correct first correction.
- The default accuracy of `raindrop_scattering` converges the radar quantities to
  $10^{-3}$ at every band. The `ddelt` of the FORTRAN code does not, at Ku, Ka and W.
- These checks do not test the *inputs*: the refractive index model of Ray (1972), the
  axis ratio of Brandes et al. (2002) and the width of the canting distribution are
  assumptions, discussed in `radarx.fundamentals.tmatrix`. The next notebook,
  `TMatrix_Radar_Bands`, applies the kernel to rain observed by radars of three bands.

## References

- Bohren, C. F., and D. R. Huffman, 1983 (reprinted 1998): *Absorption and Scattering of
  Light by Small Particles*. Wiley, <https://doi.org/10.1002/9783527618156>
- Mishchenko, M. I., 2000: Calculation of the amplitude matrix for a nonspherical
  particle in a fixed orientation. *Appl. Opt.*, **39**, 1026–1031,
  <https://doi.org/10.1364/AO.39.001026>
- Mishchenko, M. I., and L. D. Travis, 1998: Capabilities and limitations of a current
  FORTRAN implementation of the T-matrix method for randomly oriented, rotationally
  symmetric scatterers. *J. Quant. Spectrosc. Radiat. Transfer*, **60**, 309–324,
  <https://doi.org/10.1016/S0022-4073(98)00008-9>
- Brandes, E. A., G. Zhang, and J. Vivekanandan, 2002: Experiments in rainfall
  estimation with a polarimetric radar in a subtropical environment. *J. Appl.
  Meteor.*, **41**, 674–685,
  <https://doi.org/10.1175/1520-0450(2002)041<0674:EIREWA>2.0.CO;2>
- Ray, P. S., 1972: Broadband complex refractive indices of ice and water. *Appl. Opt.*,
  **11**, 1836–1844, <https://doi.org/10.1364/AO.11.001836>
