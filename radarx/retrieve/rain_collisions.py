#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Collisions, Coalescence and Breakup of Raindrops in a Rain Shaft
================================================================

:func:`~radarx.retrieve.rain_trajectories` follows every drop size on its own
path and cannot let sizes collide: that needs all sizes of a volume together.
:func:`rain_shaft` is the collisional part. It takes the drop size
distribution (DSD) at the top of a column of still air and returns it at the
distances below, with the drops colliding, coalescing and breaking up as they
fall at their terminal speeds (the same fall-speed laws, with the same air
density correction, as the trajectory model). It conserves the mass flux of
rain, that is the rain rate, to rounding.

Equations
---------
The DSD is a set of pivot diameters :math:`D_k` (mm, the ``diameter`` axis)
with number concentrations :math:`N_k` (m\\ :sup:`-3`; the bins reach to the
geometric means of neighbouring pivots). Two drops of pivots :math:`i > j`
collide at the rate :math:`K_{ij} N_i N_j` with the geometric sweep-out kernel

.. math::

    K_{ij} = \\frac{\\pi}{4} (D_i + D_j)^2 |V_i - V_j|,

that is a collision efficiency of one: Low and List (1982a, p. 1596) state
that for the sizes of their experiments the collision efficiency is given by
the geometrical sweep-out, and Prat and Barros (2007) use one for drops of
more than 0.1 mm. Collisions inside one bin are not counted (they have no
speed difference). The collision either coalesces or breaks up the drops
(below), and every drop that is produced is put on the two pivots around its
mass so that the number of drops and the mass are conserved: every event
conserves the mass of the two drops to rounding. In steady state the flux
:math:`F_k = V_k N_k` obeys

.. math::

    \\frac{dF_k}{dz} = s(z) \\sum_{i > j} K_{ij} N_i N_j\\, G_{ijk},

with the depth :math:`z` below the top, the air density factor
:math:`s = (\\rho_0/\\rho)^{0.4}` of the fall speed (``density_correction``)
and :math:`G_{ijk}` the change of the number at pivot :math:`k` per event
(production minus the two drops lost). The classical fourth-order Runge-Kutta
method integrates it with steps of at most ``step`` metres and at most
``cfl`` times the distance over which a drop of the busiest size collides.

Outcome of a collision: Low and List (1982)
-------------------------------------------
``collisions="low_list1982"`` (default) follows Low and List (1982a, 1982b),
with diameters in cm and energies in J as there:

* the collision kinetic energy :math:`\\mathrm{CKE} = (\\rho\\pi/12) D_L^3 D_S^3
  /(D_L^3 + D_S^3)\\,(V_L - V_S)^2` (Part I, Eq. 3.1), the surface energies
  :math:`S_T = \\pi\\sigma (D_L^2 + D_S^2)`, :math:`S_C = \\pi\\sigma
  (D_L^3 + D_S^3)^{2/3}` and :math:`\\Delta S_\\sigma = S_T - S_C`, the total
  energy :math:`E_T = \\mathrm{CKE} + \\Delta S_\\sigma` (Eqs. 4.1-4.4) with
  :math:`\\sigma = 7.28\\times10^{-2}` N m\\ :sup:`-1`;
* the coalescence efficiency :math:`E_c = 0.778\\,(1 + D_S/D_L)^{-2}
  \\exp(-2.61\\times10^6\\,\\sigma E_T^2/S_C)` for :math:`E_T < 5\\,\\mu`\\ J, else
  zero (Part I, Eqs. 4.5-4.6);
* if the drops do not coalesce they break up as a filament, a sheet or a disk
  with the probabilities :math:`R_f = 1.11\\times10^{-4}\\,\\mathrm{CKE}^{-0.654}`
  (one below 0.893 :math:`\\mu`\\ J), :math:`R_s = 0.685\\,[1 - \\exp(-1.63
  (W_2 - 0.86))]` for the Weber number :math:`W_2 = \\mathrm{CKE}/S_T \\ge 0.86`,
  and :math:`R_d = 1 - R_f - R_s` (Part II, Eqs. 4.40-4.46);
* the mean number of fragments of the three types, Eqs. (3.3), (3.4), (3.7)
  and (3.9), at least two;
* the small fragments: for filaments the Gaussian of the small-drop remnant
  (Eqs. 4.12-4.13) and the log-normal bridge of Eqs. (4.14)-(4.23); for
  sheets and disks the log-normal cloud-droplet parts of Eqs.
  (4.27)-(4.32) and (4.37)-(4.39). The width of every log-normal follows from
  the mean number of fragments above 0.1 mm (bisection instead of the
  iteration of the paper).

Checks against the paper: the numbers of Tables 1 to 3 of Part II (mean
fragment numbers, breakup fractions, coalescence efficiencies and total
fragment numbers of the ten drop pairs) are reproduced to the printed digits
(the disk numbers to 0.15, because the collision energies of the paper come
from a different fall-speed fit); ``tests/test_rain_collisions.py`` holds the
comparison.

What is not taken from the paper
--------------------------------
* **The large remnant.** The Gaussians :math:`P_{f1}`, :math:`P_{s1}`,
  :math:`P_{d1}` of the paper that describe the large drop left after a
  breakup (Eqs. 4.1-4.10, 4.25-4.26, 4.33-4.36) are replaced by one drop
  whose size follows from the conservation of mass: it takes the mass that
  the small fragments do not carry. McFarquhar (2004, Section 2b) does the
  same after Brown (1997) because the parameterisation of the paper does not
  conserve mass. For the ten drop pairs of the paper the remnant of a filament
  is within 1.3 % of :math:`D_L`, that of a sheet 2.5 to 17 % smaller (where
  sheets occur) and that of a disk up to 5.4 % larger than :math:`D_L`
  (it is the coalesced drop, the log-normal fragments of the disk
  carrying almost no mass).
* **Types that cannot be built.** If the log-normal of a sheet or disk has no
  solution, or the large remnant would take less than a tenth of the mass,
  the probability of that type is given to the filament, whose distribution
  is available for every pair of 0.1 to 10 mm (tested). The probabilities are rescaled to one if the paper's
  :math:`R_f + R_s` exceeds one.
* **Reading of the printed equations.** The scan of Part II has four places
  that cannot be taken literally. (i) Eq. (3.3) prints
  :math:`9.67 (D_L - 0.170)^2`; the numbers of Table 1 require
  :math:`9.67 (D_L + 0.170)^2` (checked at four large drop sizes), which is
  used. (ii) Eq. (3.5) for :math:`D_{S0}` is not legible; the text defines it
  as the intersection of :math:`\\bar F_{f1}` and :math:`\\bar F_{f2}`, which is
  computed (the filament number is the smaller of the two). (iii) Eqs.
  (4.20)-(4.21) mix the two peak values with weights that jump at
  :math:`D_{S0}`; the weights are exchanged so that the peak is continuous.
  (iv) The sign of the exponent of :math:`D_S` in the exponential of
  Eq. (4.27) is ambiguous in the scan; the positive sign is used, as the
  negative one overflows. The lower limit of 0.893 :math:`\\mu`\\ J of
  Eq. (4.41) is that of Fig. 1 of Part II (the text prints 89.3 :math:`\\mu`\\ J,
  which would not give :math:`R_f = 1`).
* **Pressure.** The experiments of the paper were done at the pressure of the
  laboratory; the energies are computed with the sea-level fall speeds also
  aloft (Low and List and Fung 1982, cited in Part I, found no difference at
  50 kPa for one drop pair).

Limits
------
The experiments cover large drops of 1.8 to 4.6 mm and small drops of 0.395 to
1.8 mm. The parameterisation is applied to every pair of the ``diameter``
axis; outside that range the fits are extrapolated (Part II shows the fragment
numbers for large drops of 1.8 and 3.6 mm only), and McFarquhar (2004) reports
that extrapolated LL82 distributions can give unrealistic numbers of small
drops. A collision efficiency of one overestimates the collisions of drops
smaller than about 0.1 mm. Drops do not break up on their own (aerodynamic
breakup) and do not evaporate; the air is at rest. Bin size matters: the
number of pivots is limited to 256 (the event tensor has :math:`n^3`
entries, 8 MB for 100 pivots), and the numerical diffusion of the fixed
pivots is larger with a coarse grid; about 40 pivots from 0.2 to 7 mm
(geometric spacing) is a start.

The numerical work (the event tensor and the integration) runs in a compiled,
multithreaded C++ kernel (over the pairs of sizes and over columns) with an
equivalent NumPy implementation as fallback; the result stays lazy for a
dask-backed input.

References
----------
.. [1] Low, T. B., and R. List, 1982a: Collision, coalescence and breakup of
   raindrops. Part I: Experimentally established coalescence efficiencies and
   fragment size distributions in breakup. *J. Atmos. Sci.*, **39** (7),
   1591-1606, https://doi.org/10.1175/1520-0469(1982)039<1591:CCABOR>2.0.CO;2
.. [2] Low, T. B., and R. List, 1982b: Collision, coalescence and breakup of
   raindrops. Part II: Parameterization of fragment size distributions.
   *J. Atmos. Sci.*, **39** (7), 1607-1619,
   https://doi.org/10.1175/1520-0469(1982)039<1607:CCABOR>2.0.CO;2
.. [3] McFarquhar, G. M., 2004: A new representation of collision-induced
   breakup of raindrops and its implications for the shapes of raindrop size
   distributions. *J. Atmos. Sci.*, **61** (7), 777-794,
   https://doi.org/10.1175/1520-0469(2004)061<0777:ANROCB>2.0.CO;2
.. [4] Prat, O. P., and A. P. Barros, 2007: A robust numerical solution of the
   stochastic collection-breakup equation for warm rain. *J. Appl. Meteor.
   Climatol.*, **46** (9), 1480-1497, https://doi.org/10.1175/JAM2544.1

McFarquhar (2004), whose parameterisation of the fragments was read but not
implemented (the lower size limit of its log-normal distributions is not
given in the text read, and the coefficient of Eq. 23 in its Table 2 does not
agree with its Fig. 10), is cited for the mass-conserving remnant only.
Brown (1997), Seifert et al. (2005) and Hall (1980) were not read.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   rain_shaft
"""

from __future__ import annotations

__all__ = ["rain_shaft"]

import numpy as np
import xarray as xr

from .._provenance import provenance
from .._registry import accessor_method
from . import _rain_collisions_numpy as _ref
from . import _rain_trajectories_numpy as _traj
from . import rain_trajectories as _rt
from ._fieldtools import check_object

try:
    from . import _rain_collisions

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _rain_collisions = None
    HAS_COMPILED_KERNEL = False

COLLISIONS = {
    "none": _ref.NONE,
    "coalescence": _ref.COALESCENCE,
    "low_list1982": _ref.LOW_LIST,
}
MAX_PIVOTS = 256
_RHO0 = _rt.RHO0
_DIAMETER = "diameter"
_ATTRS = {
    "dsd": {
        "long_name": "Drop size distribution",
        "units": "m-3 mm-1",
        "comment": "number concentration per unit diameter at the depth below the top",
    },
    "number_concentration": {"long_name": "Total drop concentration", "units": "m-3"},
    "liquid_water_content": {"long_name": "Liquid water content", "units": "g m-3"},
    "rain_rate": {
        "long_name": "Rain rate",
        "units": "mm h-1",
        "comment": "conserved by the collisions",
    },
}


def _kernel(engine):
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled rain collision kernel is not available")
    return _rain_collisions if (HAS_COMPILED_KERNEL and engine != "numpy") else _ref


def _pivot_speeds(diameter, fall_speed):
    """Sea-level fall speeds (m s-1) at the pivots from a ``fall_speed`` specification."""
    kind, fc = _rt._fall(fall_speed)
    v = _traj._v0({"fall_kind": kind, "fc": fc}, diameter)[0]
    if not np.all(v > 0.0):
        raise ValueError(
            "the fall speed must be positive at every diameter; this law is zero "
            f"at {diameter[v <= 0.0].min():g} mm (use 'drag_balance' or start above 0.11 mm)"
        )
    return np.asarray(v, dtype=np.float64)


def _density_factor(profile, top_height, depth_max, density_correction):
    """Distances below the top (m) and the factor ``(rho0/rho)^0.4`` there."""
    if not density_correction or (profile is None and top_height is None):
        return np.array([0.0, max(depth_max, 1.0)]), np.ones(2)
    if top_height is None:
        raise ValueError(
            "a profile needs top_height, the height of the top of the shaft (m)"
        )
    top = float(top_height)
    if profile is None:
        z, t, lnp, _, _ = _rt._standard_atmosphere()
        rho = np.exp(lnp) / (_ref_gas_constant() * t)
    else:
        z, rho = _profile_density(profile)
    below = top - z
    order = np.argsort(below)
    below, rho = below[order], rho[order]
    # the shaft reaches from the top to the depth; levels above the top are not needed
    keep = np.concatenate([[True], np.diff(below) > 0.0])
    below, rho = below[keep], rho[keep]
    if below[0] > 0.0 or below[-1] < depth_max:
        reach = f"{top - below[-1]:g} to {top - below[0]:g} m"
        raise ValueError(
            f"the profile (heights {reach}) must cover the shaft, from {top:g} m "
            f"down to {top - depth_max:g} m"
        )
    return below, (_RHO0 / rho) ** 0.4


def _ref_gas_constant():
    return _traj.RD


def _profile_density(profile):
    """Heights and air density of a profile Dataset."""
    if not isinstance(profile, xr.Dataset) or "height" not in profile.coords:
        raise ValueError("profile must be a Dataset on a 'height' coordinate (m)")
    z = np.asarray(profile["height"].values, dtype=np.float64)
    if "air_density" in profile:
        rho = np.asarray(profile["air_density"].values, dtype=np.float64)
    else:
        for name in ("pressure", "temperature"):
            if name not in profile:
                raise ValueError(
                    f"the profile needs 'air_density' or 'pressure' and 'temperature' ({name!r} is missing)"
                )
        p = np.asarray(profile["pressure"].values, dtype=np.float64)
        t = np.asarray(profile["temperature"].values, dtype=np.float64)
        q = (
            np.asarray(profile["specific_humidity"].values, dtype=np.float64)
            if "specific_humidity" in profile
            else np.zeros_like(p)
        )
        rho = p / (_traj.RD * t * (1.0 + (1.0 / _traj.EPS - 1.0) * np.nan_to_num(q)))
    good = np.isfinite(z) & np.isfinite(rho) & (rho > 0.0)
    if good.sum() < 2:
        raise ValueError("the profile needs at least two levels with density")
    return z[good], rho[good]


def _bin_widths(diameter):
    """Widths (mm) of the bins around the pivots."""
    return np.diff(_ref.edges_cm(diameter)) * 10.0


@provenance(
    "Raindrop collisions, coalescence and breakup in a rain shaft: coalescence "
    "efficiency and fragment distributions of Low and List (1982) with a "
    "mass-conserving remnant, geometric collision kernel"
)
def rain_shaft(
    dsd,
    *,
    depth,
    collisions="low_list1982",
    fall_speed="drag_balance",
    density_correction=True,
    profile=None,
    top_height=None,
    step=10.0,
    cfl=0.25,
    n_threads=None,
    engine="auto",
):
    """
    Drop size distribution below the top of a rain shaft, with collisions.

    Evolves the DSD of a column of still air as the drops fall, collide,
    coalesce and break up; the equations, the scheme of Low and List (1982),
    the choices of radarx and the limits are in
    :mod:`radarx.retrieve.rain_collisions`.

    Parameters
    ----------
    dsd : xarray.DataArray
        Number concentration per unit diameter (m-3 mm-1) at the top of the
        shaft, on a ``diameter`` dimension (mm, positive, ascending, at least
        three pivots, at most 256) and any other dimensions (every point is
        one independent column, for example the gates of a sweep or the
        times of a disdrometer). Not changed. A dask-backed input gives a lazy
        result.
    depth : float or array-like
        Distance (m) below the top where the DSD is wanted: one value or an
        ascending series (not negative), the ``depth`` dimension.
    collisions : {"low_list1982", "coalescence", "none"}, optional
        Outcome of a collision. ``"low_list1982"`` (default): coalescence
        efficiency and breakup of Low and List (1982). ``"coalescence"``:
        every collision coalesces (classical accretion). ``"none"``: no
        collisions, which only changes the concentration with the density.
    fall_speed : str or tuple, optional
        Sea-level fall speed law of :func:`radarx.retrieve.rain_trajectories`
        (``"drag_balance"``, ``"atlas1973"``, ``(a, b, f)`` or
        ``("polynomial", c0, ...)``). It must be positive at all pivots.
    density_correction : bool, optional
        Multiply the fall speed by :math:`(\\rho_0/\\rho)^{0.4}` with the
        density of ``profile`` (or of a standard atmosphere if only
        ``top_height`` is given). Default True; without ``profile`` and
        ``top_height`` the air is at the sea-level density.
    profile : xarray.Dataset, optional
        Profile on ``height`` (m above sea level) with ``air_density`` or
        ``pressure`` and ``temperature`` (and ``specific_humidity``), see
        :mod:`radarx.io.sounding`. It must cover the shaft. Not changed.
    top_height : float, optional
        Height (m above sea level) of the top of the shaft; required with
        ``profile``.
    step : float, optional
        Longest step of the integration (m). Default 10.
    cfl : float, optional
        Longest step as a share of the distance over which the busiest drop
        size collides once. Default 0.25.
    n_threads : int, optional
        Threads of the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation. ``"auto"`` (default) prefers the compiled kernel.

    Returns
    -------
    xarray.Dataset
        On the dimensions of ``dsd`` with ``depth`` before ``diameter``:
        ``dsd`` (m-3 mm-1), ``number_concentration`` (m-3),
        ``liquid_water_content`` (g m-3) and ``rain_rate`` (mm h-1; it does
        not change with ``depth`` for any ``collisions``: the test of the mass
        conservation). Columns with a NaN or a negative value give NaN.

    Raises
    ------
    ValueError
        For an unsupported ``diameter`` axis, a negative or descending
        ``depth``, an unknown option, a fall speed that is not positive or a
        profile that does not cover the shaft.
    TypeError
        If ``dsd`` is not a DataArray.
    ImportError
        If ``engine="compiled"`` and the kernel is not available.

    Notes
    -----
    Sources: Low and List (1982a, b) for the coalescence efficiency and the
    breakup; McFarquhar (2004) for the mass-conserving large remnant; Prat and
    Barros (2007) for the collision efficiency of one. The reading of the
    printed equations, what was left out and the limits are in the module
    documentation.

    References
    ----------
    .. [1] Low, T. B., and R. List, 1982a: Collision, coalescence and breakup
       of raindrops. Part I: Experimentally established coalescence
       efficiencies and fragment size distributions in breakup. *J. Atmos.
       Sci.*, **39** (7), 1591-1606,
       https://doi.org/10.1175/1520-0469(1982)039<1591:CCABOR>2.0.CO;2
    .. [2] Low, T. B., and R. List, 1982b: Collision, coalescence and breakup
       of raindrops. Part II: Parameterization of fragment size distributions.
       *J. Atmos. Sci.*, **39** (7), 1607-1619,
       https://doi.org/10.1175/1520-0469(1982)039<1607:CCABOR>2.0.CO;2
    .. [3] McFarquhar, G. M., 2004: A new representation of collision-induced
       breakup of raindrops and its implications for the shapes of raindrop
       size distributions. *J. Atmos. Sci.*, **61** (7), 777-794,
       https://doi.org/10.1175/1520-0469(2004)061<0777:ANROCB>2.0.CO;2
    .. [4] Prat, O. P., and A. P. Barros, 2007: A robust numerical solution of
       the stochastic collection-breakup equation for warm rain. *J. Appl.
       Meteor. Climatol.*, **46** (9), 1480-1497,
       https://doi.org/10.1175/JAM2544.1

    Examples
    --------
    >>> import numpy as np, xarray as xr
    >>> d = np.geomspace(0.2, 7.0, 40)
    >>> nd = xr.DataArray(8000.0 * np.exp(-2.2 * d), dims="diameter", coords={"diameter": d})
    >>> out = rain_shaft(nd, depth=[1000.0, 2000.0])
    >>> bool(abs(out.rain_rate.isel(depth=1) / out.rain_rate.isel(depth=0) - 1.0) < 1e-9)
    True
    """
    check_object(dsd)
    if not isinstance(dsd, xr.DataArray):
        raise TypeError("dsd must be an xarray.DataArray with a 'diameter' dimension")
    if collisions not in COLLISIONS:
        raise ValueError(
            f"collisions must be one of {sorted(COLLISIONS)}, not {collisions!r}"
        )
    if _DIAMETER not in dsd.dims:
        raise ValueError("dsd needs a 'diameter' dimension (mm)")
    kernel = _kernel(engine)
    if not (step > 0 and cfl > 0):
        raise ValueError("step and cfl must be positive")
    diameter = np.asarray(dsd[_DIAMETER].values, dtype=np.float64)
    if diameter.size < 3 or diameter.size > MAX_PIVOTS:
        raise ValueError(
            f"the diameter axis needs 3 to {MAX_PIVOTS} pivots, not {diameter.size}"
        )
    if not (
        np.all(np.isfinite(diameter))
        and diameter[0] > 0.0
        and np.all(np.diff(diameter) > 0.0)
    ):
        raise ValueError("diameter must be positive and ascending")
    z_out = np.atleast_1d(np.asarray(depth, dtype=np.float64))
    if z_out.ndim != 1 or z_out.size == 0:
        raise ValueError("depth must be a number or a one-dimensional sequence")
    if not (
        np.all(np.isfinite(z_out)) and z_out[0] >= 0.0 and np.all(np.diff(z_out) >= 0.0)
    ):
        raise ValueError("depth must be ascending and not negative")
    v0 = _pivot_speeds(diameter, fall_speed)
    zs, ss = _density_factor(profile, top_height, float(z_out[-1]), density_correction)
    width = _bin_widths(diameter)
    n_threads = int(n_threads or 0)
    mode = COLLISIONS[collisions]
    tensors = kernel.build(diameter, v0, mode, n_threads)
    kern, gain = np.ascontiguousarray(tensors[0]), np.ascontiguousarray(tensors[1])
    dz_max, cfl = float(step), float(cfl)

    def block(nd):
        nd = np.asarray(nd, dtype=np.float64)
        shape = nd.shape[:-1]
        flat = nd.reshape(-1, diameter.size)
        bad = ~np.isfinite(flat).all(axis=1) | (flat < 0.0).any(axis=1)
        conc = np.where(bad[:, None], 0.0, flat) * width
        res = np.asarray(
            kernel.evolve(kern, gain, v0, conc, z_out, dz_max, cfl, zs, ss, n_threads)
        )
        res = res / width
        res[:, bad, :] = np.nan
        # (depth, column, diameter) -> (..., depth, diameter)
        return np.moveaxis(res, 0, 1).reshape(shape + (z_out.size, diameter.size))

    field = xr.apply_ufunc(
        block,
        dsd.astype(np.float64).copy(deep=False),
        input_core_dims=[[_DIAMETER]],
        output_core_dims=[["depth", _DIAMETER]],
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={
            "output_sizes": {"depth": z_out.size},
            "allow_rechunk": True,
        },
        keep_attrs=False,
    )
    field = field.assign_coords(depth=("depth", z_out))
    field["depth"].attrs = {
        "long_name": "Distance below the top of the shaft",
        "units": "m",
    }
    if _DIAMETER in dsd.coords:
        field = field.assign_coords(
            {_DIAMETER: dsd[_DIAMETER].variable.copy(deep=False)}
        )
    n_k = field * xr.DataArray(width, dims=_DIAMETER)
    vol = np.pi / 6.0 * (diameter**3)  # mm3
    out = xr.Dataset({"dsd": field.transpose(..., "depth", _DIAMETER)})
    out["number_concentration"] = n_k.sum(_DIAMETER)
    out["liquid_water_content"] = (n_k * xr.DataArray(vol * 1e-3, dims=_DIAMETER)).sum(
        _DIAMETER
    )
    out["rain_rate"] = (
        n_k
        * xr.DataArray(v0 * vol * 3.6e-3, dims=_DIAMETER)
        * xr.DataArray(_flux_ratio(zs, ss, z_out), dims="depth")
    ).sum(_DIAMETER)
    for name, attrs in _ATTRS.items():
        out[name].attrs = dict(attrs)
    out.attrs = {"collisions": collisions, "fall_speed_law": str(fall_speed)}
    return out


def _flux_ratio(zs, ss, z_out):
    """The factor ``s`` at the output depths: the fall speed relative to sea level."""
    return np.interp(z_out, zs, ss)


@accessor_method("dataarray", name="rain_shaft")
def _rain_shaft_accessor(self, **kwargs):
    """
    Drop size distribution below the top of a rain shaft, with collisions.

    See :func:`radarx.retrieve.rain_shaft` for the parameters; the accessor
    object is the DSD (``diameter`` dimension, m-3 mm-1).

    Returns
    -------
    xarray.Dataset
        ``dsd``, ``number_concentration``, ``liquid_water_content`` and
        ``rain_rate`` at the requested depths.
    """
    return rain_shaft(self.xarray_obj, **kwargs)
