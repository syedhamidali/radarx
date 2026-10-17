#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Regenerate the raindrop scattering tables used by ``radarx.retrieve.dsd``.

The tables hold single-drop scattering properties of oblate raindrops at
S, C and X band for a few temperatures, computed with the T-matrix method
(Mishchenko and Travis 1998) by :func:`radarx.fundamentals.tmatrix.raindrop_scattering`,
radarx's own C++ translation of the code of Mishchenko (the code that
``pytmatrix``, Leinonen 2014, wraps; ``pytmatrix`` is no longer needed). The full
table takes a few seconds. radarx itself reads the CSV file this script writes.

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
  ``water_refractive_index`` are not checked against the paper);
- radar frequencies 2.8, 5.6 and 9.4 GHz for S, C and X band, and the
  quadrature order ``ndgs = 4`` of the T-matrix code: radarx choices, not from the
  cited papers. The axis-ratio polynomial agrees with Eq. 15 of Kumjian and
  Ryzhkov (2010) (who print the second coefficient as 0.025 10); the equation
  number of Brandes et al. (2002) and the value of 7 degrees for the width of
  the canting distribution (Huang et al. 2008) are not checked.

Usage::

    python ci/build_dsd_tables.py [output.csv [canting_std_degrees]]

The second argument rebuilds the table for another width of the canting
distribution (7 by default, 10 is the value Kumjian and Ryzhkov 2010 call
appropriate for typical rain, see the sensitivity section of
:mod:`radarx.retrieve.dsd`). The header line of the CSV file records the canting
width and the frequencies; ``radarx.retrieve.scattering_table`` reads them from
there.

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

from radarx.fundamentals import tmatrix

# radar frequencies [GHz]: 2.8, 5.6 and 9.4 (radarx choices, representative of
# S, C and X band) and the wavelengths [mm] = c / f
FREQUENCIES_GHZ = {"S": 2.8, "C": 5.6, "X": 9.4}
BANDS = {band: 299.792458 / f for band, f in FREQUENCIES_GHZ.items()}
TEMPERATURES = (0.0, 10.0, 20.0, 30.0)  # degrees Celsius
DIAMETERS = np.round(np.arange(1, 161) * 0.05, 4)  # mm
CANTING_STD = 7.0  # degrees (Huang et al. 2008; value not checked, see above)
OUTPUT = Path(__file__).resolve().parents[1] / "radarx/retrieve/data/dsd_scattering.csv"


def main(path=OUTPUT, canting_std=CANTING_STD):
    """Write the table to ``path`` for Gaussian canting of ``canting_std`` degrees."""
    canting_std = float(canting_std)
    rows = []
    for band, wl in BANDS.items():
        for t in TEMPERATURES:
            m = tmatrix.water_refractive_index(wl, t)
            print(f"{band} band ({wl:.2f} mm), {t:.0f} C, m = {m:.4f}", flush=True)
            d = tmatrix.raindrop_scattering(
                DIAMETERS,
                wl,
                temperature=t,
                canting_std=canting_std,
                ndgs=4,
                ddelt=1e-3,  # the value of pytmatrix, with which the table was first built
            )
            for i, diameter in enumerate(DIAMETERS):
                rows.append(
                    (
                        band,
                        wl,
                        t,
                        diameter,
                        *(
                            float(d[v].values[i])
                            for v in (
                                "sigma_h",
                                "sigma_v",
                                "copol_re",
                                "copol_im",
                                "kdp",
                                "ah",
                                "av",
                            )
                        ),
                    )
                )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Single-drop scattering of raindrops, T-matrix "
        "(radarx.fundamentals.tmatrix), built by ci/build_dsd_tables.py\n"
        "# canting_std_deg="
        f"{canting_std:g}; "
        + "; ".join(f"frequency_ghz_{b}={f:g}" for b, f in FREQUENCIES_GHZ.items())
        + "; elevation_deg=0; axis_ratio=Brandes et al. (2002); "
        "canting=Gaussian, 0 mean; refractive_index=Ray (1972)\n"
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
    main(*sys.argv[1:3])
