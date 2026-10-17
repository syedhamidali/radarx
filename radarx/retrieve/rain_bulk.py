#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Two-Moment Rain Shaft with Self-Collection and Breakup
======================================================

:func:`rain_shaft_bulk` is the bulk counterpart of
:func:`radarx.retrieve.rain_shaft`. Instead of a size distribution on many
pivots it carries two moments of the rain, the number concentration
:math:`N_r` and the rainwater content :math:`L_r`, as the two-moment scheme of
Seifert (2008) does, so that it costs a few operations per level and can be
run on every gate of a volume. It follows the Appendix and Eqs. (19), (20) of
Seifert (2008) [1]_.

Equations
---------
The drops are a gamma distribution :math:`n(D) = N_0 D^\\mu e^{-\\lambda D}`
with the mean volume diameter :math:`D_m = (6 L_r / \\pi\\rho_w N_r)^{1/3}`.

* The shape parameter follows from :math:`D_m` (Eq. 20), a fit to a spectral
  bin model that includes collisions, breakup and evaporation, with
  :math:`D_\\mathrm{eq} = 1.1` mm, :math:`c_1 = 4` mm\\ :sup:`-1` and
  :math:`c_2 = 1` mm\\ :sup:`-1`:

  .. math::

      \\mu = \\begin{cases}
      6\\tanh^2[c_1 (D_m - D_\\mathrm{eq})] + 1, & D_m \\le D_\\mathrm{eq}\\\\
      30\\tanh^2[c_2 (D_m - D_\\mathrm{eq})] + 1, & D_m > D_\\mathrm{eq}
      \\end{cases}

  and the slope is :math:`\\lambda = [(\\mu+3)(\\mu+2)(\\mu+1)]^{1/3} / D_m`
  (Eq. 19).
* The fall speed of a drop is :math:`v(D) = a - b e^{-cD}` with
  :math:`a = 9.65` m s\\ :sup:`-1`, :math:`b = 9.8` m s\\ :sup:`-1` and
  :math:`c = 600` m\\ :sup:`-1` (Eq. A4), multiplied by the air density factor
  :math:`s = (\\rho_0/\\rho)^{0.4}` of the trajectory model. The number and the
  mass of the gamma distribution fall at

  .. math::

      v_k = a - b \\left(1 + \\frac{c}{\\lambda}\\right)^{-(\\mu + k + 1)},

  with :math:`k = 0` for the number and :math:`k = 3` for the mass (Eq. A10).
* Self-collection removes drops, :math:`\\partial N_r/\\partial t|_{sc} =
  -k_{rr} N_r L_r` with :math:`k_{rr} = 4.33` m\\ :sup:`3` kg\\ :sup:`-1`
  s\\ :sup:`-1` (Eq. A12), and breakup is a relaxation of the mean volume
  diameter toward :math:`D_\\mathrm{eq}`,

  .. math::

      \\left.\\frac{\\partial N_r}{\\partial t}\\right|_{br} =
      -\\{k_{br}(D_m - D_\\mathrm{eq}) + 1\\}
      \\left.\\frac{\\partial N_r}{\\partial t}\\right|_{sc},
      \\quad D_m > 0.3\\ \\mathrm{mm},

  zero otherwise, with :math:`k_{br} = 1` mm\\ :sup:`-1` (Eq. A13). The net
  change is :math:`k_{br}(D_m - D_\\mathrm{eq})\\,k_{rr} N_r L_r`: drops
  larger than 1.1 mm break up (the number grows), smaller ones coalesce.
  The rainwater content is not changed by either process.

In the steady state of a column of still air with the depth :math:`z` below the
top, the water flux :math:`F_L = s v_3 L_r` is constant and the flux of the
number of drops :math:`F_N = s v_0 N_r` obeys :math:`dF_N/dz = s\\,
\\partial N_r/\\partial t`. The state is :math:`F_N`; the pair
:math:`(N_r, L_r)` follows from the two fluxes by solving
:math:`F_N/F_L = (v_0/v_3)/m(D_m)` for :math:`D_m` (a bisection; the ratio
decreases with :math:`D_m`), with :math:`m` the mass of the drop of diameter
:math:`D_m`. The classical fourth-order Runge-Kutta method integrates the flux
with steps of at most ``step`` metres and at most ``cfl`` times the distance
over which the collisions change the number of drops. The rain rate
:math:`F_L/\\rho_w` does not change with the depth.

What is not taken from the paper
--------------------------------
* **The sign of Eq. (A12).** It prints
  :math:`\\partial N_r/\\partial t|_{sc} = k_{rr} N_r L_r` without the minus
  sign; self-collection removes drops, and Eq. (A13) needs the loss to give a
  relaxation toward :math:`D_\\mathrm{eq}`. The loss is used.
* **Density correction.** The paper gives no dependence of the constants on air
  density. The fall speeds and the collision rate are multiplied by
  :math:`s`, as in :func:`rain_shaft` (the collision rate is proportional to
  the speed difference); ``density_correction=False`` removes it.
* **Evaporation, accretion of cloud water and the size dependence of
  self-collection** of the full scheme of Seifert and Beheng (2001, 2006) are
  not included: the shaft is rain with collisions and breakup only, the case
  that :func:`rain_shaft` solves with a bin model.

Limits
------
The scheme is a fit: the relaxation toward 1.1 mm is a diagnostic of the
bin model of Seifert (2008) with his own breakup scheme, not of the
Low and List (1982) breakup of :func:`rain_shaft`. A steady shaft of the
bin model of this package with Low and List breakup reaches a mean volume
diameter of about 1.25 mm, the bulk scheme 1.1 mm, and the approach is slower in the bulk scheme; the notebook ``Rain_Collisions``
compares them. A column whose mean volume diameter at the top is outside 0.05
to 25 mm is NaN. The fall speed law :math:`a - b e^{-cD}` is negative below
26 :math:`\\mu`\\ m, so drizzle of a few tens of micrometres is not covered.

The numerical work runs in a compiled, multithreaded C++ kernel (over the
columns) with an equivalent NumPy implementation as fallback; the result stays
lazy for a dask-backed input.

References
----------
.. [1] Seifert, A., 2008: On the parameterization of evaporation of raindrops
   as simulated by a one-dimensional rainshaft model. *J. Atmos. Sci.*,
   **65** (11), 3608-3619, https://doi.org/10.1175/2008JAS2586.1

.. autosummary::
   :nosignatures:
   :toctree: generated/

   rain_shaft_bulk
"""

from __future__ import annotations

__all__ = ["rain_shaft_bulk"]

import numpy as np
import xarray as xr

from .._provenance import provenance
from .._registry import accessor_method
from . import _rain_bulk_numpy as _ref
from . import rain_collisions as _rc

try:
    from . import _rain_bulk

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _rain_bulk = None
    HAS_COMPILED_KERNEL = False

_ATTRS = {
    "number_concentration": {"long_name": "Raindrop concentration", "units": "m-3"},
    "liquid_water_content": {"long_name": "Rainwater content", "units": "g m-3"},
    "mean_volume_diameter": {
        "long_name": "Mean volume diameter of the rain",
        "units": "mm",
    },
    "shape_parameter": {
        "long_name": "Gamma shape parameter of the rain",
        "units": "1",
        "comment": "diagnosed from the mean volume diameter, Eq. (20) of Seifert (2008)",
    },
    "slope_parameter": {
        "long_name": "Gamma slope parameter of the rain",
        "units": "mm-1",
    },
    "rain_rate": {
        "long_name": "Rain rate",
        "units": "mm h-1",
        "comment": "conserved by self-collection and breakup",
    },
}


def _kernel(engine):
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled bulk rain shaft kernel is not available")
    return _rain_bulk if (HAS_COMPILED_KERNEL and engine != "numpy") else _ref


@provenance(
    "Two-moment rain shaft with self-collection and breakup of Seifert (2008): "
    "shape parameter from the mean volume diameter, relaxation toward 1.1 mm"
)
def rain_shaft_bulk(
    rain,
    *,
    depth,
    density_correction=True,
    profile=None,
    top_height=None,
    step=10.0,
    cfl=0.25,
    n_threads=None,
    engine="auto",
):
    """
    Number and water content of rain below the top of a shaft, two-moment.

    Evolves the number concentration and the rainwater content of a column of
    still air with self-collection and breakup; the equations, the choices of
    radarx and the limits are in :mod:`radarx.retrieve.rain_bulk`.

    Parameters
    ----------
    rain : xarray.Dataset
        ``number_concentration`` (m-3) and ``liquid_water_content`` (g m-3) at
        the top of the shaft, on any common dimensions (every point is one
        column, for example the gates of a sweep, the points of a grid or the
        times of a disdrometer). This is the form of the output of
        :func:`radarx.retrieve.rain_shaft`. Not changed. A dask-backed input
        gives a lazy result.
    depth : float or array-like
        Distance (m) below the top where the rain is wanted: one value (a
        scalar ``depth`` coordinate) or an ascending series (not negative),
        the ``depth`` dimension.
    density_correction : bool, optional
        Multiply the fall speeds and the collision rate by
        :math:`(\\rho_0/\\rho)^{0.4}`. Default True; without ``profile`` and
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
        Longest step as a share of the distance over which the collisions
        change the number of drops. Default 0.25.
    n_threads : int, optional
        Threads of the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation. ``"auto"`` (default) prefers the compiled kernel.

    Returns
    -------
    xarray.Dataset
        On the dimensions of ``rain`` with ``depth`` (if a series was given)
        last: ``number_concentration`` (m-3), ``liquid_water_content`` (g
        m-3), ``mean_volume_diameter`` (mm), ``shape_parameter``,
        ``slope_parameter`` (mm-1) and ``rain_rate`` (mm h-1, which does not
        change with the depth). Columns with a NaN, a value that is not
        positive or a mean volume diameter at the top outside 0.05 to 25 mm
        give NaN.

    Raises
    ------
    ValueError
        For a missing variable, a negative or descending ``depth``, an unknown
        ``engine`` or a profile that does not cover the shaft.
    TypeError
        If ``rain`` is not a Dataset.
    ImportError
        If ``engine="compiled"`` and the kernel is not available.

    Notes
    -----
    Source: Seifert (2008), see the module documentation for the equations,
    the interpreted sign of Eq. (A12) and the limits.

    References
    ----------
    .. [1] Seifert, A., 2008: On the parameterization of evaporation of
       raindrops as simulated by a one-dimensional rainshaft model. *J.
       Atmos. Sci.*, **65** (11), 3608-3619,
       https://doi.org/10.1175/2008JAS2586.1

    Examples
    --------
    >>> import xarray as xr
    >>> top = xr.Dataset({"number_concentration": 240.0, "liquid_water_content": 2.0})
    >>> out = rain_shaft_bulk(top, depth=[0.0, 4000.0])
    >>> bool(out.mean_volume_diameter.isel(depth=1) < out.mean_volume_diameter.isel(depth=0))
    True
    """
    if not isinstance(rain, xr.Dataset):
        raise TypeError(
            "rain must be an xarray.Dataset with number_concentration and liquid_water_content"
        )
    for name in ("number_concentration", "liquid_water_content"):
        if name not in rain:
            raise ValueError(f"rain needs the variable {name!r}")
    kernel = _kernel(engine)
    if not (step > 0 and cfl > 0):
        raise ValueError("step and cfl must be positive")
    z_out = np.atleast_1d(np.asarray(depth, dtype=np.float64))
    if z_out.ndim != 1 or z_out.size == 0:
        raise ValueError("depth must be a number or a one-dimensional sequence")
    if not (
        np.all(np.isfinite(z_out)) and z_out[0] >= 0.0 and np.all(np.diff(z_out) >= 0.0)
    ):
        raise ValueError("depth must be ascending and not negative")
    zs, ss = _rc._density_factor(
        profile, top_height, float(z_out[-1]), density_correction
    )
    n_threads = int(n_threads or 0)
    dz_max, cfl = float(step), float(cfl)
    number, water = xr.broadcast(
        rain["number_concentration"], rain["liquid_water_content"]
    )

    def block(n, w):
        n = np.asarray(n, dtype=np.float64)
        shape = n.shape
        res = np.asarray(
            kernel.evolve(
                np.ascontiguousarray(n.ravel()),
                np.ascontiguousarray(np.asarray(w, dtype=np.float64).ravel() * 1.0e-3),
                z_out,
                dz_max,
                cfl,
                zs,
                ss,
                n_threads,
            )
        )
        # (depth, column, 3) -> (..., depth, 3)
        return np.moveaxis(res, 0, 1).reshape(shape + (z_out.size, 3))

    field = xr.apply_ufunc(
        block,
        number.astype(np.float64).copy(deep=False),
        water.astype(np.float64).copy(deep=False),
        input_core_dims=[[], []],
        output_core_dims=[["depth", "_moment"]],
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {"depth": z_out.size, "_moment": 3}},
        keep_attrs=False,
    )
    field = field.assign_coords(depth=("depth", z_out))
    field["depth"].attrs = {
        "long_name": "Distance below the top of the shaft",
        "units": "m",
    }
    n_now, w_now, dm = (field.isel(_moment=k, drop=True) for k in range(3))
    mu = xr.apply_ufunc(_ref.shape_parameter, dm, dask="allowed")
    lam = xr.apply_ufunc(_ref.slope, dm, mu, dask="allowed")
    out = xr.Dataset(
        {
            "number_concentration": n_now,
            "liquid_water_content": w_now * 1.0e3,
            "mean_volume_diameter": dm * 1.0e3,
            "shape_parameter": mu,
            "slope_parameter": lam * 1.0e-3,
        }
    )
    flux_ratio = _rc._flux_ratio(zs, ss, z_out)
    vl = xr.apply_ufunc(lambda d: _ref.sedimentation(d)[1], dm, dask="allowed")
    out["rain_rate"] = w_now * vl * xr.DataArray(flux_ratio, dims="depth") * 3.6e3
    for name, attrs in _ATTRS.items():
        out[name].attrs = dict(attrs)
    out.attrs = {"fall_speed_law": "a - b exp(-c D), Seifert (2008) Eq. A4"}
    if np.ndim(depth) == 0:
        out = out.isel(depth=0)
    return out


@accessor_method("dataset", name="rain_shaft_bulk", wraps=rain_shaft_bulk)
def _rain_shaft_bulk_accessor(self):
    """
    Number and water content of rain below the top of a shaft, two-moment.

    See :func:`radarx.retrieve.rain_shaft_bulk` for the parameters; the
    accessor object is the Dataset with ``number_concentration`` and
    ``liquid_water_content`` at the top.

    Returns
    -------
    xarray.Dataset
        ``number_concentration``, ``liquid_water_content``,
        ``mean_volume_diameter``, ``shape_parameter``, ``slope_parameter``
        and ``rain_rate`` at the requested depths.
    """
