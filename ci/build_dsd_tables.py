#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Regenerate the raindrop scattering tables used by ``radarx.retrieve.dsd``.

The tables hold single-drop scattering properties of oblate raindrops at
S, C and X band for a few temperatures, computed with the T-matrix method
(Mishchenko and Travis 1998) through the ``pytmatrix`` interface
(Leinonen 2014). ``pytmatrix`` is only needed here, at table-build time;
radarx itself reads the CSV file this script writes.

Provenance and caveats are also summarised in
``radarx/retrieve/data/README_dsd.md``. Assumptions (documented in
:mod:`radarx.retrieve.dsd`):

- equal-volume diameters 0.05-8 mm in steps of 0.05 mm;
- oblate spheroids with the axis ratio of Brandes et al. (2002),
  b/a = 0.9951 + 0.0251 D - 0.03644 D^2 + 0.005303 D^3 - 0.0002492 D^4;
- canting angles Gaussian with zero mean and 7 degrees standard deviation
  (Huang et al. 2008), orientation-averaged with Gaussian quadrature;
- horizontal incidence (0 degree elevation): the tables therefore give the
  ZDR seen at low elevation; at 20 degrees elevation ZDR is about 12 % lower
  (a Rayleigh estimate, see :mod:`radarx.retrieve.dsd`), which these tables
  do not represent;
- complex refractive index of liquid water from the Debye model of
  Ray (1972), at 0, 10, 20 and 30 degrees Celsius (the constants in
  ``water_refractive_index`` were transcribed from that model and not
  re-checked against the paper, which is not on disk);
- radar frequencies 2.8, 5.6 and 9.4 GHz for S, C and X band, and the
  quadrature order ``ndgs = 4`` of ``pytmatrix``: radarx choices, not from the
  cited papers. The axis-ratio polynomial agrees with Eq. 15 of Kumjian and
  Ryzhkov (2010) (who print the second coefficient as 0.025 10); the equation
  number of Brandes et al. (2002) was not checked (paper not on disk), and the
  value of 7 degrees for the width of the canting distribution (Huang et al.
  2008) was not checked either.

Usage::

    python ci/build_dsd_tables.py [output.csv]

References
----------
Mishchenko, M. I., and L. D. Travis, 1998: Capabilities and limitations of a
current FORTRAN implementation of the T-matrix method for randomly oriented,
rotationally symmetric scatterers. *J. Quant. Spectrosc. Radiat. Transfer*,
**60** (3), 309-324, https://doi.org/10.1016/S0022-4073(98)00008-9

Leinonen, J., 2014: High-level interface to T-matrix scattering
calculations: architecture, capabilities and limitations. *Opt. Express*,
**22** (2), 1655-1660, https://doi.org/10.1364/OE.22.001655

Brandes, E. A., G. Zhang, and J. Vivekanandan, 2002: Experiments in rainfall
estimation with a polarimetric radar in a subtropical environment. *J. Appl.
Meteor.*, **41** (6), 674-685,
https://doi.org/10.1175/1520-0450(2002)041<0674:EIREWA>2.0.CO;2

Kumjian, M. R., and A. V. Ryzhkov, 2010: The impact of evaporation on
polarimetric characteristics of rain: Theoretical model and practical
implications. *J. Appl. Meteor. Climatol.*, **49** (6), 1247-1267,
https://doi.org/10.1175/2010JAMC2243.1

Huang, G.-J., V. N. Bringi, and M. Thurai, 2008: Orientation angle
distributions of drops after an 80-m fall using a 2D video disdrometer.
*J. Atmos. Oceanic Technol.*, **25** (9), 1717-1723,
https://doi.org/10.1175/2008JTECHA1075.1

Ray, P. S., 1972: Broadband complex refractive indices of ice and water.
*Appl. Opt.*, **11** (8), 1836-1844, https://doi.org/10.1364/AO.11.001836
"""

import sys
from pathlib import Path

import numpy as np

try:
    from pytmatrix import orientation, radar
    from pytmatrix import scatter as tm_scatter
    from pytmatrix.tmatrix import Scatterer
except ImportError:  # pragma: no cover - build-time tool
    sys.exit(
        "pytmatrix is needed to build the scattering tables "
        "(pip install pytmatrix, or build it from source with numpy.f2py)"
    )

# radar wavelengths [mm]: 2.8, 5.6 and 9.4 GHz (radarx choices, representative
# of S, C and X band; wavelength = c / f)
BANDS = {"S": 299.792458 / 2.8, "C": 299.792458 / 5.6, "X": 299.792458 / 9.4}
TEMPERATURES = (0.0, 10.0, 20.0, 30.0)  # degrees Celsius
DIAMETERS = np.round(np.arange(1, 161) * 0.05, 4)  # mm
CANTING_STD = 7.0  # degrees (Huang et al. 2008; value not checked, see above)
OUTPUT = Path(__file__).resolve().parents[1] / "radarx/retrieve/data/dsd_scattering.csv"


def axis_ratio(d):
    """
    Brandes et al. (2002) vertical-to-horizontal axis ratio, D in mm.

    The polynomial agrees with Eq. 15 of Kumjian and Ryzhkov (2010); it is
    capped at 1 (a radarx choice).
    """
    r = 0.9951 + 0.0251 * d - 0.03644 * d**2 + 0.005303 * d**3 - 0.0002492 * d**4
    return np.minimum(r, 1.0)


def water_refractive_index(wavelength_mm, temperature_c):
    """
    Complex refractive index of liquid water after Ray (1972).

    Debye-type model: static and high-frequency permittivities, relaxation
    wavelength, distribution parameter alpha and ionic conductivity term, as
    functions of temperature. The constants were transcribed here and not
    re-checked against the paper.
    """
    t = temperature_c
    lam = wavelength_mm / 10.0  # cm
    eps_s = 78.54 * (
        1.0
        - 4.579e-3 * (t - 25.0)
        + 1.19e-5 * (t - 25.0) ** 2
        - 2.8e-8 * (t - 25.0) ** 3
    )
    eps_inf = 5.27137 + 0.0216474 * t - 0.00131198 * t**2
    alpha = -16.8129 / (t + 273.0) + 0.0609265
    lam_s = 0.00033836 * np.exp(2513.98 / (t + 273.0))  # cm
    sigma = 12.5664e8
    x = (lam_s / lam) ** (1.0 - alpha)
    s, c = np.sin(alpha * np.pi / 2.0), np.cos(alpha * np.pi / 2.0)
    den = 1.0 + 2.0 * x * s + x * x
    eps_r = eps_inf + (eps_s - eps_inf) * (1.0 + x * s) / den
    eps_i = (eps_s - eps_inf) * x * c / den + sigma * lam / 18.8496e10
    return np.sqrt(complex(eps_r, eps_i))


def drop_properties(d, wavelength, m):
    """
    Orientation-averaged scattering properties of one drop of diameter d.

    Backscattering (sigma_h, sigma_v and the copolar correlation terms from
    the Z matrix) and forward scattering (K_DP, attenuation) from the T-matrix
    code at horizontal incidence.
    """
    s = Scatterer(
        radius=d / 2.0,
        wavelength=wavelength,
        m=m,
        axis_ratio=1.0 / axis_ratio(d),
        ndgs=4,
    )
    s.orient = orientation.orient_averaged_fixed
    s.or_pdf = orientation.gaussian_pdf(std=CANTING_STD)
    # backscattering, horizontal incidence
    s.set_geometry((90.0, 90.0, 0.0, 180.0, 0.0, 0.0))
    z = s.get_Z()
    sigma_h = radar.radar_xsect(s, True)
    sigma_v = radar.radar_xsect(s, False)
    copol_re = 2.0 * np.pi * (z[2, 2] + z[3, 3])
    copol_im = 2.0 * np.pi * (z[3, 2] - z[2, 3])
    # forward scattering
    s.set_geometry((90.0, 90.0, 0.0, 0.0, 0.0, 0.0))
    kdp = radar.Kdp(s)
    # specific attenuation per drop per m3: 10 log10(e) = 4.343, times the
    # extinction cross section [mm2] and the unit conversions (1e-6 m2 mm-2,
    # 1e3 m km-1) -> dB km-1
    ah = 4.343e-3 * tm_scatter.ext_xsect(s, h_pol=True)
    av = 4.343e-3 * tm_scatter.ext_xsect(s, h_pol=False)
    return sigma_h, sigma_v, copol_re, copol_im, kdp, ah, av


def main(path=OUTPUT):
    rows = []
    for band, wl in BANDS.items():
        for t in TEMPERATURES:
            m = water_refractive_index(wl, t)
            print(f"{band} band ({wl:.2f} mm), {t:.0f} C, m = {m:.4f}", flush=True)
            for d in DIAMETERS:
                rows.append((band, wl, t, d, *drop_properties(d, wl, m)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Single-drop scattering of raindrops, T-matrix (pytmatrix), "
        "built by ci/build_dsd_tables.py\n"
        "# axis ratio Brandes et al. (2002); Gaussian canting, 0 mean, "
        f"{CANTING_STD:g} deg std; horizontal incidence; water refractive "
        "index Ray (1972)\n"
        "# diameter: equal-volume diameter [mm]; sigma_h, sigma_v, copol_re, "
        "copol_im: backscatter cross sections and copolar correlation terms "
        "[mm2]\n"
        "# kdp [deg km-1], ah, av [dB km-1] per drop per m3\n"
        "band,wavelength,temperature,diameter,sigma_h,sigma_v,copol_re,"
        "copol_im,kdp,ah,av\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for band, wl, t, d, *vals in rows:
            f.write(
                f"{band},{wl:.4f},{t:g},{d:g},"
                + ",".join(f"{v:.6e}" for v in vals)
                + "\n"
            )
    print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
