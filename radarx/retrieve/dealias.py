#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Doppler Velocity Dealiasing
===========================

Unfold Doppler radial velocities that exceed the Nyquist velocity.

A pulse Doppler radar measures radial velocity only within the Nyquist
interval :math:`[-V_n, V_n]`; a true velocity :math:`v` is reported as
:math:`v - 2 k V_n` for some integer fold :math:`k`. Dealiasing finds
:math:`k` for every gate.

radarx uses a region-based method that works directly on the polar sweep
grid, where each gate's neighbours are known by index. The method is
radarx's own combination of published ideas, not an implementation of one
paper: the criterion (a least-squares sum of squared velocity jumps between
neighbours, Jing and Wiener 1993 [1]), the reference wind (Eilts and Smith
1990 [2]), the volume continuity (James and Houze 2001 [3]) and the VAD fit
(Browning and Wexler 1968 [4]) are the published concepts; the region graph,
spanning tree, coordinate descent and the gate check are radarx's
implementation, and all thresholds below are radarx choices. The details of the
concepts quoted here are not checked against the papers.

1. **Regions.** Neighbouring gates (along the ray and between adjacent rays)
   whose velocities differ by less than ``threshold * Vn`` are joined with a
   union-find. Inside such a region the field is continuous, so all its gates
   share one fold. Velocity jumps of about :math:`2 V_n` (fold lines) separate
   regions. The default ``threshold=0.3`` is radarx's own choice. A thin
   neck of gates with in-between velocities (a vortex core, a gradient
   across a fold line) can join the two sides of a fold line through a chain
   of small steps and make one region of two blocks that are a fold apart.
   Only the gates joined to all four neighbours (their cores) connect
   regions, and only cores of at least 1000 gates; every other gate takes the
   region of the neighbour it matches best. Two blocks joined by a neck then
   stay two regions and the boundary between them decides their folds. A
   sweep without a core of that size is segmented by the joins alone
   (radarx choices).
2. **Region graph.** For every pair of touching regions, the number of
   boundary gate pairs and the summed velocity jump across the boundary are
   accumulated. Regions separated by short gaps of empty gates (along or
   across rays) are compared across the gap.
3. **Folds.** Integer folds per region minimise the summed squared velocity
   jump over all region boundaries, weighted by boundary length (the
   least-squares criterion of Jing and Wiener 1993 [1]). A maximum spanning tree
   (unambiguous boundaries first, then by length) gives the start; integer
   coordinate descent then moves single regions and whole blocks of
   consistently joined regions until no move lowers the cost, so groups of
   regions that are offset together are corrected too.
4. **Absolute fold.** The largest group of connected regions is matched to a
   reference velocity where one is available: a wind profile (sounding, VAD
   or model) as in Eilts and Smith (1990) [2], or the already dealiased
   sweep below, the volume continuity of James and Houze (2001) [3].
   Otherwise its mean velocity is brought closest to zero (radarx choice). A
   VAD fit of that group (Browning and Wexler 1968 [4]; the in-sweep fit uses
   20 gates on each side and needs at least 50 gates and an azimuth coverage
   above a minimum, radarx choices) then gives the reference for all other,
   disconnected groups.
   Finally, a gate that differs by more than :math:`V_n` from all of its
   neighbours is moved to the fold closest to their mean.
5. **Speed limit** (optional, ``max_speed``). Described under "Vortex cores"
   below.

All decisions use integer arithmetic, so the compiled C++ kernel
(multithreaded, all sweeps in one call) and the NumPy fallback give
identical folds.

**Confidence.** ``dealias_velocity(..., confidence=True)`` and
:func:`dealias_confidence` give, for every gate, how clearly the chosen fold
beats the second best one given the neighbouring gates (a radarx definition,
see the function). It is computed from the result alone.

**Dual-PRF radars.** :func:`correct_dual_prf` corrects the isolated dealiasing
errors that dual-PRF velocity data contain, with the local dealiasing step of
Holleman and Beekhuis (2003) [6]. It is a different problem from the
region-based dealiasing above, which unfolds the velocity of a single-PRF
sweep.

**Vortex cores.** The criterion above assumes that the velocity changes by
less than a Nyquist interval between neighbouring gates, which a compact
vortex signature breaks. A couplet with peaks of about :math:`\\pm 0.95 V_n`
beside a region that really is folded can join that region through a chain of
small steps, and the least-squares folds then shift the whole region, core
included, because most of its border pairs are fold lines; the core ends up at
speeds far beyond :math:`V_n`. The sweep alone cannot tell this from a strong
real wind, so nothing in the default method changes: a limit of 1.5
Nyquist velocities undoes the core on the lowest tilt of the KBMX volume of
27 April 2011 but also moves 1 100 gates of the lowest tilt of a squall line
at KGWX, 30 March 2022, where UNRAVEL agrees with the unmoved fold on 684 of
806 of them. The default result is checked instead. A block of at most 200 connected gates that the folds shifted
by the same single fold, and that holds 10 gates or more with speeds beyond
two Nyquist velocities (and beyond 50 m/s, so that the strong winds that a
radar with a small Nyquist velocity folds several times do not count), raises
a ``UserWarning`` that names the number of pockets, of gates and the largest
speed. These thresholds are radarx choices. ``max_speed`` is the
choice of the user: the largest radial speed the storm can have. A gate that
the folds put beyond it, in a cluster of at most 100 connected gates beyond the
limit and next to gates that are folded less and match it better without its
fold, goes back one fold at a time, so the core is undone from its edge, while
the plateau of a real fold, whose speed stays below the limit, keeps its fold.

**UNRAVEL.** ``dealias_velocity(..., method="unravel")`` implements the
algorithm of Louf et al. (2020) [5], which needs no reference velocity. The
implementation follows the paper; the reference implementation of the authors (open
source, ``github.com/vlouf/dealias``) was used to cross-check the order of the
steps and the values the paper does not give, and no code was taken from it.
A gate carries the flag -3 (missing), 0 (not processed), 1 (processed,
unchanged) or 2 (processed, unfolded); a gate is accepted when its unfolded
velocity is within ``alpha * Vn`` of a reference velocity (Eq. 3 of the paper),
and a gate that has been processed is changed only by the checks. The default
driver (Fig. 4 of the paper) does, for each sweep:

1. *Pre-processing.* A modified z-score, ``(v - median) / MAD``, in a running
   window of 10 gates along each ray; a gate with ``|z| > 2`` is unfolded
   against the mean of the window gates of the sign of the median, then against
   the mean of the others, and is made missing if neither is within the
   tolerance.
2. *Reference radials.* Two rays that hold enough gates (two thirds of the
   mean number per ray), no gate near the Nyquist velocity (``|v| >= 0.6 Vn``),
   and minima of the curve of the mean ``|v|`` along the azimuth; the first is
   the lowest minimum, the second the minimum at least 120 degrees from it that
   stands deepest below the adjacent maxima. If none qualifies the first
   condition, then the second, are dropped. The reference radials are assumed
   not to be aliased, and their neighbours are unfolded against them.
3. *First pass.* Azimuthal continuity in four sectors that start at the
   reference radials and end halfway between them, each gate against the mean
   of the three rays before it (at least two processed), ``alpha = 0.6``.
4. *Range and azimuthal continuity,* twice with a growing window (6 then 12):
   along the ray, onward then backward, against the mean of the processed
   gates in the window; across the rays from the first reference radial,
   clockwise then counterclockwise, against their median.
5. *Box.* Four passes against the median of the processed gates in a box
   around the gate, of (5, 5), (20, 10), (40, 10) and (80, 20) gates and rays.
6. *Closest reference.* The gates that remain take the fold nearest to the
   median of the processed gates around the closest processed gate.
7. *Checks.* The processed gates are brought within ``alpha * Vn`` of a
   least-squares line along each ray, and of the mean of the gates within one
   standard deviation of the mean in a box of (80, 20); for a volume, every
   sweep is then compared in the same way with the corrected sweep below it
   (``sweep_continuity``).

The Nyquist velocity, ``alpha`` (0.8) and the sizes and windows above are those
of the paper where it gives them (the first-pass ``alpha`` 0.6, ``beta`` 0.6,
the z-score window 10 and cutoff 2, the window 6 and the four boxes). It does
not give the tolerance of the pre-processing (0.3, as in the reference
implementation), the second window (12, ditto), the size of the box around the
closest reference (5 x 5), the box of the check (80, 20, ditto), the windows of
the 3-D check (10 gates, 3 rays), the number of near-Nyquist gates a reference
radial may hold (3, ditto) and the least-squares check's minimum of 10 gates;
these are radarx choices. Not implemented: the "long range" driver of the
paper (windows 12, 24, 48 and 96 alternating with 6), the box that precedes
the gate instead of surrounding it, and the use of a reflectivity gate filter.
A compiled kernel (sweeps in parallel, the passes of a sweep sequential, as the
paper's passes are) and a Python specification
(:mod:`radarx.retrieve._unravel_numpy`) give identical values. On the Doppler
cuts of KGWX volumes folded to a C-band Nyquist velocity of 13.3 m/s its errors
are lower than those of the region method on two of four volumes and higher on
the other two (see the notebook); the error rate of the paper's test is not
reproduced.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}

References
----------
.. [1] Jing, Z., and G. Wiener, 1993: Two-dimensional dealiasing of Doppler
   velocities. *J. Atmos. Oceanic Technol.*, **10** (6), 798-808,
   https://doi.org/10.1175/1520-0426(1993)010<0798:TDDODV>2.0.CO;2
.. [2] Eilts, M. D., and S. D. Smith, 1990: Efficient dealiasing of Doppler
   velocities using local environment constraints. *J. Atmos. Oceanic
   Technol.*, **7** (1), 118-128,
   https://doi.org/10.1175/1520-0426(1990)007<0118:EDODVU>2.0.CO;2
.. [3] James, C. N., and R. A. Houze, 2001: A real-time four-dimensional
   Doppler dealiasing scheme. *J. Atmos. Oceanic Technol.*, **18** (10),
   1674-1683, https://doi.org/10.1175/1520-0426(2001)018<1674:ARTFDD>2.0.CO;2
.. [4] Browning, K. A., and R. Wexler, 1968: The determination of kinematic
   properties of a wind field using Doppler radar. *J. Appl. Meteor.*, **7** (1),
   105-113, https://doi.org/10.1175/1520-0450(1968)007<0105:TDOKPO>2.0.CO;2
.. [5] Louf, V., A. Protat, R. C. Jackson, S. M. Collis, and J. Helmus, 2020:
   UNRAVEL: A robust modular velocity dealiasing technique for Doppler radar.
   *J. Atmos. Oceanic Technol.*, **37** (5), 741-758,
   https://doi.org/10.1175/JTECH-D-19-0020.1
.. [6] Holleman, I., and H. Beekhuis, 2003: Analysis and correction of dual
   PRF velocity data. *J. Atmos. Oceanic Technol.*, **20** (4), 443-453,
   https://doi.org/10.1175/1520-0426(2003)20<443:AACODP>2.0.CO;2
"""

from __future__ import annotations

__all__ = ["dealias_velocity", "correct_dual_prf", "dealias_confidence"]

__doc__ = __doc__.format("\n   ".join(__all__))

import copy
import functools
import warnings
from itertools import pairwise

import numpy as np
import xarray as xr

from .._compat import renamed_arguments
from .._fields import VELOCITY, as_sweep, find_field
from .._provenance import provenance
from .._registry import accessor_for
from .._sweepwise import (
    Output,
    Sweep,
    apply_sweeps,
    dask_support,
    is_plain,
    ray_coordinate,
    ray_dimension,
)
from ..fundamentals.geometry import beam_center_height
from ._products import product_tree

try:
    from . import _dealias

    HAS_COMPILED_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _dealias = None
    HAS_COMPILED_KERNEL = False

try:
    from . import _unravel

    HAS_UNRAVEL_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _unravel = None
    HAS_UNRAVEL_KERNEL = False

_SCALE = 1 << 20  # fixed point for velocity jumps in units of 2 Vn
# The constants below are radarx's own choices, not values from a paper.
_NYQUIST_TOLERANCE = 1.01  # values beyond this times Vn are flags, not data
_MAX_VOTE = 127  # folds beyond this are not physical
_VEL_Q = 256.0  # velocities in fixed point of 1/256 m/s
_TRIG_Q = 16384.0  # sin/cos of azimuth in fixed point
_VAD_WINDOW = 20  # range gates on each side for the in-sweep VAD fit
_VAD_MIN_GATES = 50
_VAD_MIN_SPREAD = 0.02  # azimuth coverage (determinant; full circle 0.25)
_MIN_NEIGHBOURS = 3
_MIN_COVERAGE = 180.0  # degrees of azimuth below which zero mean is doubtful
_DISAGREE_GATE = (
    1.25  # |difference between tilts| / Nyquist velocity of a gate in a block
)
_DISAGREE_BLOCK = 1.5  # median of that ratio in a block that is a fold apart
_DISAGREE_MIN_GATES = 1000  # connected gates of a block that is reported
_MAX_RAY_GAP = 10  # empty rays bridged between regions
_CORE_SIZE = 1000  # gates of a core that can seed a region (the same in the kernel)
_CORE_JOINS = 4  # joined neighbours of a core gate (4 of 4; 0 turns the rule off)
_GATE_PASSES = 2
_GUARD_PASSES = 100  # passes of the speed guard (it peels a core from its edge)
_GUARD_SIZE = (
    100  # gates of the largest cluster beyond the limit that the guard examines
)
_FAST_SPEED = 2.0  # speed (in Nyquist velocities) that a compact pocket must exceed
_FAST_FLOOR = 50.0  # and the speed in m/s below which a pocket is never reported
_FAST_GATES = 10  # gates beyond that speed in a pocket that is reported
_FAST_POCKET = 200  # gates of the largest pocket (one fold) that is reported


@dask_support(
    "lazy",
    "With ``sweep_continuity=True`` a volume of several sweeps is computed, because each sweep follows the one below it; pass ``sweep_continuity=False`` to keep the sweeps lazy.",
)
@renamed_arguments(radar="obj")
@provenance(
    "Velocity dealiasing: region-based (radarx's combination of published "
    "concepts) or UNRAVEL (Louf et al. 2020)"
)
def dealias_velocity(
    obj,
    field=None,
    nyquist_velocity=None,
    *,
    threshold=0.3,
    max_gap=20,
    reference=None,
    wind_profile=None,
    sweep_continuity=True,
    name=None,
    confidence=False,
    method="region",
    alpha=0.8,
    max_iterations=100,
    max_speed=None,
    n_threads=None,
    engine="auto",
    products_only=True,
):
    """
    Dealias (unfold) Doppler radial velocities.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataTree
        A PPI sweep on ``(azimuth, range)`` with ``azimuth`` and
        ``elevation`` coordinates, a single sweep node of a volume
        (``dtree["sweep_0"]``), or a volume with ``sweep_*`` groups, e.g. from
        xradar.
    field : str, optional
        Radial velocity field. Default: the first of ``VRADH``, ``VRAD``,
        ``VRADV``, ``VEL``, ``VELOCITY``, ``velocity`` that the sweep holds.
    nyquist_velocity : float or dict, optional
        Nyquist velocity in m/s. By default it is read from the sweep's
        ``nyquist_velocity`` variable or coordinate (xradar), or from the
        field's ``nyquist_velocity`` attribute. For a DataTree, a dict maps
        sweep names to values.
    threshold : float, optional
        Neighbouring gates belong to the same region when their velocities
        differ by less than ``threshold`` times the Nyquist velocity.
        Default 0.3.
    max_gap : int, optional
        Regions separated along a ray by up to this many empty gates (e.g.
        range-folded or censored gates) are still compared. Default 20.
        Across rays, gaps of up to 10 rays are bridged.
    reference : xarray.DataArray, optional
        Reference radial velocity on the sweep grid (Dataset input only),
        e.g. from a model. It fixes the absolute fold.
    wind_profile : xarray.Dataset, optional
        Horizontal wind profile with variables ``u`` (eastward) and ``v``
        (northward) in m/s on a ``height`` coordinate (metres above sea
        level), e.g. from a sounding, a VAD or a model. Its radial component
        is used as the reference velocity.
    sweep_continuity : bool, optional
        For a DataTree, process sweeps from the lowest elevation upward and
        use the dealiased sweep below as the reference for the next one
        (gaps are filled from ``wind_profile``). Default True.
    name : str, optional
        Name of the dealiased velocity. Default ``f"{field}_dealiased"``
        (e.g. ``"VRADH_dealiased"``), so that it never replaces the measured
        field when the products are merged into the sweep or volume with
        ``.radarx.assign(products)``.
    confidence : bool, optional
        Also return the continuity confidence of every gate (see
        :func:`dealias_confidence`) as ``f"{name}_confidence"``: 1 where the
        chosen fold agrees with the median of the neighbouring gates, falling
        to 0 where the second best fold fits them as well. Default False. A
        sweep then gives a Dataset instead of a DataArray; a volume gives the
        extra variable in every sweep node. Not available with
        ``products_only=False``.
    method : {"region", "unravel"}, optional
        ``"region"`` (default) is the region-based method of the module
        documentation. ``"unravel"`` is the UNRAVEL algorithm of Louf et al.
        (2020) [5], which needs no reference velocity: it ignores
        ``threshold``, ``max_gap`` and ``max_iterations``, does not take
        ``wind_profile`` or ``reference``, and uses ``sweep_continuity`` for
        its 3-D check between neighbouring elevation angles (see the module
        documentation for what was implemented). A volume is processed with
        the sweeps in parallel; the passes inside a sweep are sequential.
        It is much slower than the region method: on a WSR-88D volume of 14
        Doppler sweeps and 1.4 million valid gates, the compiled kernels take
        0.2 s for ``"region"`` and 9 to 12 s for ``"unravel"``, about 40 to 50
        times as long (the NumPy fallback of UNRAVEL is a Python loop over
        the gates and slower still).
    alpha : float, optional
        UNRAVEL's stringency (Eq. 3 of the paper): a gate is accepted when its
        unfolded velocity is within ``alpha * Vn`` of the reference. Default
        0.8, the paper's default (0.6 in its first pass, which is fixed).
        Only used with ``method="unravel"``.
    max_iterations : int, optional
        Maximum passes of the fold optimisation. Default 100.
    max_speed : float, optional
        Largest radial speed (m/s) that the storm can have. Default None: no
        limit. With a limit, a gate that the folds put beyond ``max_speed``
        goes back by one Nyquist interval at a time as long as that lowers the
        sum of its absolute velocity differences to the neighbouring gates
        that are folded less than it. Only a small cluster of gates beyond the
        limit (at most 100 connected gates) is examined, and a gate that is
        beside a less folded gate that it matches only with its fold keeps
        its fold, so the fold lines of a real fold survive. This protects the
        core of a vortex signature that was shifted together with a folded
        region next to it (see the module documentation). The limit is in m/s
        and not in units of the Nyquist velocity, because the wind of a
        folded region can be several times the Nyquist velocity of a short
        wavelength radar. It applies to every sweep of a volume; the winds are
        usually faster at the higher tilts, so dealias the lowest sweeps on
        their own with a limit. Only for ``method="region"``.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use. ``"auto"`` (default) prefers the compiled
        kernel and falls back to NumPy; both give identical folds.
    products_only : bool, optional
        ``True`` (default) returns the products only. ``False`` keeps the
        earlier behaviour for one more release and emits a
        ``FutureWarning``: a sweep gives the dealiased DataArray named
        ``field``, and a volume gives a copy of the input tree with the
        dealiased field (by default replacing ``field``) in every sweep.
        Use ``dtree.radarx.assign(dealias_velocity(dtree))`` instead.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        For a sweep, the dealiased velocity ``name`` with the input's
        coordinates and dtype. For a volume, a DataTree with the input's
        root and one node per sweep that has ``field``, holding the
        dealiased velocity ``name`` on the sweep's coordinates. A gate is NaN
        if it has no data, if its velocity is a flag value (``|v|`` above
        1.01 times the Nyquist velocity, e.g. the codes of range-folded
        gates), or, with ``method="unravel"``, if the pre-processing of the
        method removes it (see the Notes). A velocity at the Nyquist velocity
        or up to 1 % beyond it is data, in both methods: velocities are coded
        in steps of 0.5 m/s, so the largest code of a radar with a Nyquist
        velocity of 29.97 m/s is 30.0 m/s.

    Raises
    ------
    ValueError
        If the Nyquist velocity is unknown or not positive.
    ImportError
        If ``engine="compiled"`` and the compiled kernel is not available.

    Notes
    -----
    See the module documentation for the method, which combines the concepts
    of Jing and Wiener (1993) [1], Eilts and Smith (1990) [2], James and
    Houze (2001) [3] and Browning and Wexler (1968) [4] and is not an
    implementation of any one of these papers; the thresholds (``threshold``,
    ``max_gap``, the 10 ray gap, ``max_iterations``) are radarx choices.
    Without any reference, the
    absolute fold of the lowest sweep assumes that its mean radial velocity
    is close to zero, as for a horizontally uniform wind seen all around the
    radar; for echoes that cover only part of the circle, pass a
    ``wind_profile`` (or ``reference``). A warning is raised when no
    reference is available and the largest group of gates covers less than
    180 degrees of azimuth (a 60 degree sector in a 30 m/s wind with a
    10 m/s Nyquist velocity came out with no correct gate). The output
    attribute ``dealias_anchor`` names what fixed the absolute fold of the
    largest group: ``"wind_profile"``, ``"reference"``, ``"sweep_below"`` or
    ``"zero_mean"``; the other groups follow the same reference, else the
    azimuth wind (VAD) fitted to the largest group, else a zero mean.

    For a volume (a ``DataTree``) the dealiased neighbouring tilts are
    compared. Where two of them differ by more than 1.25 Nyquist velocities
    over a connected block of at least 1000 gates, with a median difference of
    at least 1.5 Nyquist velocities, one of the two has the wrong fold there:
    a ``UserWarning`` is raised and the number of gates is stored in the
    ``dealias_disagreement_gates`` attribute of both sweeps. The tilts are
    not corrected and the thresholds are radarx choices. Compare
    ``method="unravel"`` where this warning appears. On the lowest tilt of the
    Tuscaloosa-Birmingham tornado volume of KBMX (27 April 2011, 22:19 UTC) a
    block of 29 500 gates north of the radar was one fold away from the tilts
    above, because a thin neck of in-between velocities beside the vortex made
    one region of it and its neighbour; the regions are now cut at such necks
    (see "Regions" above).

    With the default ``max_speed=None`` the result is checked for the shifted
    vortex core of the module documentation: compact blocks (at most 200
    connected gates) that are shifted by one fold and hold 10 or more gates
    beyond two Nyquist velocities and 50 m/s raise a ``UserWarning``. The
    values are not changed. On the 0.5 degree cut of KBMX (27 April 2011) it
    reports one pocket of 13 gates with speeds up to 88 m/s, the core of the
    couplet, and ``max_speed=36`` gives peaks of -36 and +26 m/s there.

    Missing gates of ``method="unravel"``. The region method gives a value to
    every gate that has data. UNRAVEL makes a gate missing when its first step
    takes it for noise: the velocity is more than two median absolute
    deviations from the median of the 10 gates of its window along the ray, and
    neither unfolding by whole intervals brings it within 0.3 Vn of the mean of
    the window gates of either sign (steps 1 and the tolerance of the module
    documentation). That removes isolated noise, 1 to 4 % of the gates of the
    sweeps of a WSR-88D volume, and also gates where the velocity changes
    steeply along the ray without being noise, such as the core of a tornadic
    vortex. The gates are not recovered by the later passes.

    References
    ----------
    .. [1] Jing, Z., and G. Wiener, 1993: Two-dimensional dealiasing of
       Doppler velocities. *J. Atmos. Oceanic Technol.*, **10** (6), 798-808,
       https://doi.org/10.1175/1520-0426(1993)010<0798:TDDODV>2.0.CO;2
    .. [2] Eilts, M. D., and S. D. Smith, 1990: Efficient dealiasing of
       Doppler velocities using local environment constraints. *J. Atmos.
       Oceanic Technol.*, **7** (1), 118-128,
       https://doi.org/10.1175/1520-0426(1990)007<0118:EDODVU>2.0.CO;2
    .. [3] James, C. N., and R. A. Houze, 2001: A real-time four-dimensional
       Doppler dealiasing scheme. *J. Atmos. Oceanic Technol.*, **18** (10),
       1674-1683,
       https://doi.org/10.1175/1520-0426(2001)018<1674:ARTFDD>2.0.CO;2
    .. [4] Browning, K. A., and R. Wexler, 1968: The determination of
       kinematic properties of a wind field using Doppler radar. *J. Appl.
       Meteor.*, **7** (1), 105-113,
       https://doi.org/10.1175/1520-0450(1968)007<0105:TDOKPO>2.0.CO;2
    .. [5] Louf, V., A. Protat, R. C. Jackson, S. M. Collis, and J. Helmus,
       2020: UNRAVEL: A robust modular velocity dealiasing technique for
       Doppler radar. *J. Atmos. Oceanic Technol.*, **37** (5), 741-758,
       https://doi.org/10.1175/JTECH-D-19-0020.1

    Examples
    --------
    >>> vr = radarx.retrieve.dealias_velocity(ds, "VRADH", 26.0)  # doctest: +SKIP
    >>> vr.name  # doctest: +SKIP
    'VRADH_dealiased'
    >>> products = dtree.radarx.dealias("VRADH")  # doctest: +SKIP
    >>> dtree = dtree.radarx.assign(products)  # doctest: +SKIP
    """
    obj = as_sweep(obj)
    field = find_field(obj, VELOCITY, field)
    if method not in ("region", "unravel"):
        raise ValueError(f"method must be 'region' or 'unravel', not {method!r}")
    if method == "unravel":
        if wind_profile is not None or reference is not None:
            raise ValueError(
                "method='unravel' needs no reference velocity: "
                "do not pass wind_profile or reference"
            )
        if not products_only:
            raise ValueError("method='unravel' needs products_only=True")
        if not 0 < alpha <= 2:
            raise ValueError("alpha must be in (0, 2]")
        if max_speed is not None:
            raise ValueError("max_speed is only used by method='region'")
    if max_speed is not None and not max_speed > 0:
        raise ValueError("max_speed must be positive")
    if confidence and not products_only:
        raise ValueError("confidence=True needs products_only=True")
    if products_only:
        name = name or f"{field}_dealiased"
    else:
        warnings.warn(
            "dealias_velocity(..., products_only=False) returns the input with "
            "the dealiased field added and will be removed in the next "
            "release. Use the default products_only=True and merge the "
            "products with ds.radarx.assign(products) or "
            "dtree.radarx.assign(products).",
            FutureWarning,
            stacklevel=2,
        )
        name = name or field
    compiled = _use_compiled(engine)
    options = {
        "threshold": float(threshold),
        "max_gap": int(max_gap),
        "max_iterations": int(max_iterations),
        "max_speed": float(max_speed or 0.0),
        "n_threads": int(n_threads or 0),
        "compiled": compiled,
    }
    if method == "unravel":
        if engine == "compiled" and not HAS_UNRAVEL_KERNEL:
            raise ImportError("the compiled UNRAVEL kernel is not available")
        options["compiled"] = HAS_UNRAVEL_KERNEL and engine != "numpy"
        options["alpha"] = float(alpha)
    if isinstance(obj, xr.Dataset):
        sweep, result = _dealias_sweep(
            obj,
            field,
            _nyquist(obj, field, nyquist_velocity),
            reference,
            wind_profile,
            method,
            options,
        )
        result = result.rename(name)
        return _with_confidence(result, sweep, options) if confidence else result
    if reference is not None:
        raise ValueError("reference is only supported for a single sweep.")
    return _dealias_tree(
        obj,
        field,
        nyquist_velocity,
        wind_profile=wind_profile,
        sweep_continuity=sweep_continuity,
        name=name,
        products_only=products_only,
        confidence=confidence,
        method=method,
        **options,
    )


def _with_confidence(result, sweep, options):
    """Dataset of a dealiased DataArray and its confidence (see :func:`dealias_confidence`)."""
    opts = _confidence_options(
        1, _MIN_NEIGHBOURS, 1.0, options["n_threads"], options["compiled"]
    )
    conf = _confidence(result, sweep.azimuth, sweep.nyquist, opts)
    out = result.to_dataset()
    out[f"{result.name}_confidence"] = xr.DataArray(
        conf.data,
        dims=conf.dims,
        coords=result.coords,
        attrs=_confidence_attrs(opts, sweep.nyquist),
    )
    return out


def _confidence_core(batch, options):
    """Confidence of sweeps given as ``((azimuth, nyquist), (velocity,))``."""
    return [
        (
            _confidence_values(
                np.ascontiguousarray(arrays[0], dtype=np.float64),
                azimuth,
                nyquist,
                options,
            ),
        )
        for (azimuth, nyquist), arrays in batch
    ]


def _confidence(da, azimuth, nyquist, options):
    """Confidence of a field on (..., ray, range), lazy for dask arrays."""
    core = functools.partial(_confidence_core, options=options)
    (conf,) = apply_sweeps(
        core, Sweep([da], context=(azimuth, nyquist)), (Output(np.float32),)
    )
    return conf


_ANCHORS = ("zero_mean", "wind_profile", "reference", "sweep_below", "unravel")


def _dealias_core(batch, method, options):
    """
    Dealiased velocity of sweeps given as ``((sweep, wind, geometry), (velocity,
    reference))``: the velocity and the code of the anchor of each.
    """
    sweeps = [ctx[0].with_velocity(arrays[0]) for ctx, arrays in batch]
    if method == "unravel":
        folds = _unravel_folds_of(sweeps, [ctx[2] for ctx, _ in batch], options)
        codes = [_ANCHORS.index("unravel")] * len(sweeps)
    else:
        refs = [
            _fill(
                (
                    None
                    if arrays[1] is None
                    else np.ascontiguousarray(arrays[1], dtype=np.float64)
                ),
                ctx[1],
            )
            for ctx, arrays in batch
        ]
        solved = _region_folds(sweeps, **options)
        folds = _absolute_folds(sweeps, solved, refs, **options)
        if not options["max_speed"]:
            for sw, f in zip(sweeps, folds):
                _warn_fast_pockets(sw, f, stacklevel=8)
        codes = [
            _ANCHORS.index(
                _anchor_of(
                    sw,
                    sol[1],
                    ref,
                    "wind_profile" if arrays[1] is None else "reference",
                    "",
                )
            )
            for sw, sol, ref, (_, arrays) in zip(sweeps, solved, refs, batch)
        ]
    return [
        (sw.velocity + 2.0 * sw.nyquist * f, code)
        for sw, f, code in zip(sweeps, folds, codes)
    ]


def _fast_pockets(sweep, folds):
    """
    Compact pockets of one fold that hold speeds beyond ``_FAST_SPEED`` Vn.

    The speed is ``_FAST_SPEED`` Nyquist velocities, but at least
    ``_FAST_FLOOR`` m/s, so that the strong winds that a radar with a small
    Nyquist velocity folds several times do not count. A pocket is a block of
    at most ``_FAST_POCKET`` connected gates that are shifted by the same
    single fold (+1 or -1), and it is reported when ``_FAST_GATES`` of its
    gates or more are beyond the speed. Returns the number of pockets, of
    gates beyond the speed in them, the largest speed and the speed limit.
    """
    vel, nyq = sweep.velocity, sweep.nyquist
    valid = ~np.isnan(vel)
    fold = valid & (np.abs(folds) == 1)
    speed = np.abs(np.where(valid, vel, 0.0) + 2.0 * nyq * folds)
    limit = max(_FAST_SPEED * nyq, _FAST_FLOOR)
    beyond = fold & (speed > limit)
    if np.count_nonzero(beyond) < _FAST_GATES:
        return 0, 0, 0.0, limit
    label, count = _cluster_labels(fold, valid, sweep.links, same=folds)
    size = np.bincount(label[fold], minlength=count)
    fast = np.bincount(label[beyond], minlength=count)
    report = (fast >= _FAST_GATES) & (size <= _FAST_POCKET)
    inside = beyond & report[label]
    if not inside.any():
        return 0, 0, 0.0, limit
    return int(report.sum()), int(inside.sum()), float(speed[inside].max()), limit


def _warn_fast_pockets(sweep, folds, stacklevel, name=""):
    """Warn about compact pockets whose speed no storm is likely to reach."""
    pockets, gates, peak, limit = _fast_pockets(sweep, folds)
    if not pockets:
        return
    warnings.warn(
        f"dealias_velocity: {name}{pockets} compact pocket(s) shifted by one fold hold "
        f"{gates} gates with speeds beyond {limit:.0f} m/s, up to {peak:.0f} m/s. The core of a velocity couplet that was shifted with a "
        "folded region next to it looks like this. If the storm cannot be that "
        "fast, pass max_speed (m/s), the largest radial speed it can have.",
        UserWarning,
        stacklevel=stacklevel,
    )


def _dealias_sweep(ds, field, nyquist, reference, wind_profile, method, options):
    """The sweep and its dealiased velocity (lazy for dask or a stack)."""
    sweep = _Sweep(ds, field, nyquist, load=False)
    wind = None
    if wind_profile is not None:
        wind = _wind_reference(ds, sweep, wind_profile)
    ref = None if reference is None else reference.broadcast_like(sweep.da)
    geometry = _unravel_geometry(ds, sweep) if method == "unravel" else None
    core = functools.partial(_dealias_core, method=method, options=options)
    result, code = apply_sweeps(
        core,
        Sweep([sweep.da, ref], context=(sweep, wind, geometry)),
        (Output(sweep.dtype), Output(np.int8, "scalar")),
    )
    anchor = None
    if is_plain(sweep.da):
        anchor = _ANCHORS[int(code.values)]
    return sweep, sweep.decorate(result.data, anchor, method)


# --------------------------------------------------------------------------
# xarray layer
# --------------------------------------------------------------------------


def _use_compiled(engine):
    """Whether to run the compiled kernel for the requested ``engine``."""
    if engine not in ("auto", "compiled", "numpy"):
        raise ValueError(
            f"engine must be 'auto', 'compiled' or 'numpy', not {engine!r}"
        )
    if engine == "compiled" and not HAS_COMPILED_KERNEL:
        raise ImportError("the compiled dealiasing kernel is not available")
    return HAS_COMPILED_KERNEL and engine != "numpy"


def _nyquist(ds, field, override, sweep=None):
    """Nyquist velocity of a sweep: user value, xradar metadata or attribute."""
    value = override
    if isinstance(value, dict):
        value = value.get(sweep)
    if value is None and "nyquist_velocity" in ds.variables:
        value = np.nanmedian(np.asarray(ds["nyquist_velocity"].values, dtype=float))
    if value is None:
        value = ds[field].attrs.get("nyquist_velocity")
    if value is None:
        where = f" in {sweep!r}" if sweep else ""
        raise ValueError(
            f"Unknown Nyquist velocity{where}: pass nyquist_velocity=... or add a "
            "'nyquist_velocity' coordinate to the sweep."
        )
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Nyquist velocity must be positive, got {value}.")
    return value


def _ray_links(azimuth, max_gap=2.0):
    """Whether ray ``i`` touches ray ``i + 1`` (the last one: the first one)."""
    az = np.asarray(azimuth, dtype=np.float64)
    if az.size < 2:
        return np.zeros(az.size, dtype=np.uint8)
    step = np.abs((np.roll(az, -1) - az + 180.0) % 360.0 - 180.0)
    spacing = np.median(step[:-1]) if az.size > 2 else step[0]
    return (step <= max_gap * spacing).astype(np.uint8)


class _Sweep:
    """
    A sweep's geometry and its velocity as contiguous (ray, range) float64.

    The velocity is read when the sweep is made (``load=True``) or set later
    for the sweeps of a stack or of dask arrays (:meth:`with_velocity`).
    """

    def __init__(self, ds, field, nyquist, load=True):
        self.da = ds[field]
        self.ray = ray_dimension(self.da)
        self.nyquist = nyquist
        self.velocity = None
        self.links = _ray_links(ds["azimuth"].values)
        self.azimuth = ray_coordinate(ds, "azimuth")
        az = np.deg2rad(self.azimuth)
        self.ray_sin = np.ascontiguousarray(np.sin(az))
        self.ray_cos = np.ascontiguousarray(np.cos(az))
        if load:
            self.velocity = self._flagged(self.da.transpose(self.ray, "range").values)

    def _flagged(self, values):
        """Values beyond the Nyquist velocity are flags (e.g. range folded)."""
        vel = np.array(values, dtype=np.float64)
        vel[~(np.abs(vel) <= _NYQUIST_TOLERANCE * self.nyquist)] = np.nan
        return np.ascontiguousarray(vel)

    def with_velocity(self, values):
        """A copy of the sweep with the (ray, range) ``values`` as velocity."""
        sweep = copy.copy(self)
        sweep.velocity = self._flagged(values)
        return sweep

    def wrap(self, folds, anchor=None, method="region"):
        """Dealiased DataArray with the input's coordinates, dims and dtype."""
        out = self.velocity + 2.0 * self.nyquist * folds
        if self.da.dims[0] != self.ray:
            out = out.T
        return self.decorate(out, anchor, method)

    def decorate(self, values, anchor=None, method="region"):
        """The dealiased ``values`` (on the dims of the field) as a DataArray."""
        da = self.da
        dtype = da.dtype if np.issubdtype(da.dtype, np.floating) else np.float64
        attrs = dict(da.attrs)
        attrs.update(
            {
                "long_name": "Dealiased "
                + da.attrs.get("long_name", "radial velocity"),
                "nyquist_velocity": self.nyquist,
                "comment": "Nyquist folds unfolded with radarx.retrieve."
                "dealias_velocity ("
                + (
                    "UNRAVEL, Louf et al. 2020"
                    if method == "unravel"
                    else "region-based, Jing and Wiener 1993"
                )
                + ")",
            }
        )
        attrs.setdefault("units", "m s-1")
        attrs.setdefault(
            "standard_name", "radial_velocity_of_scatterers_away_from_instrument"
        )
        if anchor is not None:
            attrs["dealias_anchor"] = anchor
        return da.copy(data=values.astype(dtype, copy=False)).assign_attrs(attrs)

    @property
    def dtype(self):
        """Type of the dealiased velocity: that of the field if it is floating."""
        dtype = self.da.dtype
        return dtype if np.issubdtype(dtype, np.floating) else np.dtype(np.float64)


def _sweep_elevation(ds, sweep):
    """Elevation of every ray; the same for the sweeps of a stack."""
    elevation = ds["elevation"]
    if set(elevation.dims) - {sweep.ray}:
        raise ValueError(
            f"the elevation has the dimensions {elevation.dims}: a wind_profile "
            "needs the same elevation for every sweep of a stack"
        )
    return np.broadcast_to(elevation.values, ds[sweep.ray].shape)


def _beam_height(ds, sweep):
    """Gate heights above sea level (4/3 Earth) on (ray, range)."""
    rng = ds["range"].values.astype(np.float64)[None, :]
    elev = _sweep_elevation(ds, sweep)
    alt = float(ds["altitude"]) if "altitude" in ds.variables else 0.0
    return beam_center_height(rng, elev.astype(np.float64)[:, None], radar_height=alt)


def _wind_reference(ds, sweep, wind_profile):
    """Radial component of a ``u``/``v`` wind profile at every gate."""
    height = wind_profile["height"].values.astype(np.float64)
    order = np.argsort(height)
    z = _beam_height(ds, sweep)
    u = np.interp(z, height[order], wind_profile["u"].values[order], np.nan, np.nan)
    v = np.interp(z, height[order], wind_profile["v"].values[order], np.nan, np.nan)
    az = np.deg2rad(ds["azimuth"].values.astype(np.float64))[:, None]
    elev = _sweep_elevation(ds, sweep)
    cos_el = np.cos(np.deg2rad(elev.astype(np.float64)))[:, None]
    return (u * np.sin(az) + v * np.cos(az)) * cos_el


def _largest_component(comp):
    """Mask of the gates of the component with the most gates, or None."""
    valid = comp >= 0
    if not valid.any():
        return None
    return comp == np.argmax(np.bincount(comp[valid]))


def _azimuth_coverage(azimuth, mask):
    """Degrees of azimuth covered by the rays with gates in ``mask``."""
    az = np.sort(np.unique(np.mod(azimuth[mask.any(axis=1)], 360.0)))
    if az.size < 2:
        return 0.0
    gaps = np.diff(np.concatenate([az, [az[0] + 360.0]]))
    typical = np.median(gaps)
    return float(360.0 - gaps.max() + typical)


def _anchor_of(sweep, comp, reference, source, label):
    """
    Name of what fixes the absolute fold of the largest group of gates:
    ``source`` (``"wind_profile"``, ``"reference"`` or ``"sweep_below"``)
    when the reference has values there, else ``"zero_mean"``, which warns
    when the group spans less than 180 degrees of azimuth.
    """
    group = _largest_component(comp)
    if group is None:
        return "zero_mean"
    if source is not None and reference is not None:
        if np.isfinite(reference[group]).any():
            return source
    coverage = _azimuth_coverage(sweep.azimuth, group)
    if coverage < _MIN_COVERAGE:
        warnings.warn(
            f"dealias_velocity: {label}the largest group of gates covers only "
            f"{coverage:.0f} degrees of azimuth and no reference velocity is "
            "available, so its absolute fold assumes a zero mean radial "
            "velocity and can be wrong by multiples of 2 Vn; pass wind_profile "
            "or reference. The anchor is stored in the dealias_anchor "
            "attribute.",
            UserWarning,
            stacklevel=4,
        )
    return "zero_mean"


def _fill(ref, other):
    if ref is None:
        return other
    if other is None:
        return ref
    return np.where(np.isnan(ref), other, ref)


def _nearest_index(source, target, tolerance, period=None):
    """Index of the nearest ``source`` value for each ``target`` (-1 if too far)."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if period is not None:
        source = np.mod(source, period)
        target = np.mod(target, period)
    order = np.argsort(source, kind="stable")
    s = source[order]
    pos = np.searchsorted(s, target)
    lo = np.clip(pos - 1, 0, s.size - 1)
    hi = np.clip(pos, 0, s.size - 1)
    dist_lo = np.abs(target - s[lo])
    dist_hi = np.abs(target - s[hi])
    best = np.where(dist_hi < dist_lo, hi, lo)
    dist = np.minimum(dist_lo, dist_hi)
    if period is not None:  # across 0/360
        wrap_first = np.abs(target - s[0] - period)
        wrap_last = np.abs(target + period - s[-1])
        best = np.where(wrap_first < dist, 0, best)
        dist = np.minimum(dist, wrap_first)
        best = np.where(wrap_last < dist, s.size - 1, best)
        dist = np.minimum(dist, wrap_last)
    return np.where(dist <= tolerance, order[best], -1)


def _sweep_mapping(source, target):
    """Nearest ray and gate of sweep ``source`` for each ray and gate of ``target``."""
    az = np.asarray(target["azimuth"].values, dtype=np.float64)
    step = np.abs((np.diff(az) + 180.0) % 360.0 - 180.0)
    az_step = np.median(step) if step.size else 1.0
    rng = np.asarray(target["range"].values, dtype=np.float64)
    rng_step = np.median(np.diff(rng)) if rng.size > 1 else 1.0
    iray = _nearest_index(source["azimuth"].values, az, 1.5 * az_step, 360.0)
    igate = _nearest_index(source["range"].values, rng, 1.5 * rng_step)
    return iray.astype(np.int32), igate.astype(np.int32)


def _region_folds(sweeps, threshold, max_gap, max_iterations, n_threads, compiled, **_):
    """Steps 1-3 for every sweep: (relative folds, component ids)."""
    if compiled:
        return _dealias.region_folds(
            [s.velocity for s in sweeps],
            [s.links for s in sweeps],
            [s.nyquist for s in sweeps],
            threshold=threshold,
            max_gap=max_gap,
            max_ray_gap=_MAX_RAY_GAP,
            max_iterations=max_iterations,
            n_threads=n_threads,
        )
    return [
        _region_folds_numpy(
            s.velocity, s.links, s.nyquist, threshold, max_gap, max_iterations
        )
        for s in sweeps
    ]


def _absolute_folds(
    sweeps,
    solved,
    references,
    n_threads,
    compiled,
    previous=None,
    mapping=None,
    max_speed=0.0,
    **_,
):
    """Step 4 for every sweep: final integer folds (optionally chained)."""
    if compiled:
        none = np.zeros(0, dtype=np.int32)
        mapping = [(none, none) if m is None else m for m in mapping or []]
        return _dealias.absolute_folds(
            [s.velocity for s in sweeps],
            [k for k, _ in solved],
            [c for _, c in solved],
            [s.nyquist for s in sweeps],
            references,
            [s.ray_sin for s in sweeps],
            [s.ray_cos for s in sweeps],
            [s.links for s in sweeps],
            vad_window=_VAD_WINDOW,
            gate_passes=_GATE_PASSES,
            max_speed=max_speed,
            guard_size=_GUARD_SIZE,
            previous=previous or [],
            prev_ray=[m[0] for m in mapping] if mapping else [],
            prev_gate=[m[1] for m in mapping] if mapping else [],
            n_threads=n_threads,
        )
    out = []
    for i, (s, (k, c), ref) in enumerate(zip(sweeps, solved, references)):
        if previous and previous[i] >= 0:
            ref = _fill(
                _chained_reference(sweeps[previous[i]], out[previous[i]], *mapping[i]),
                ref,
            )
        out.append(
            _absolute_folds_numpy(
                s.velocity,
                k,
                c,
                s.nyquist,
                ref,
                s.ray_sin,
                s.ray_cos,
                s.links,
                max_speed,
            )
        )
    return out


def _chained_reference(source, folds, iray, igate):
    """Dealiased ``source`` sweep at the gates given by the nearest-index mapping."""
    unfolded = source.velocity + 2.0 * source.nyquist * folds
    ref = unfolded[np.clip(iray, 0, None)][:, np.clip(igate, 0, None)]
    ref[iray < 0] = np.nan
    ref[:, igate < 0] = np.nan
    return ref


def _tilt_disagreement(lower, upper, iray, igate, nyquist):
    """
    Sizes of the connected blocks where two tilts are about a fold apart.

    ``lower`` and ``upper`` are the dealiased velocities of two sweeps on the
    grid of ``upper`` (``iray`` and ``igate`` map the rays and gates of
    ``upper`` onto ``lower``). A block is a connected set of at least
    ``_DISAGREE_MIN_GATES`` gates where the velocities differ by more than
    ``_DISAGREE_GATE`` Nyquist velocities and the median difference is at
    least ``_DISAGREE_BLOCK``: one of the two tilts has the wrong fold there.
    """
    from scipy import ndimage

    ref = lower[np.clip(iray, 0, None)][:, np.clip(igate, 0, None)]
    ref[iray < 0] = np.nan
    ref[:, igate < 0] = np.nan
    with np.errstate(invalid="ignore"):
        ratio = np.abs(upper - ref) / nyquist
    labels, count = ndimage.label(np.nan_to_num(ratio, nan=0.0) > _DISAGREE_GATE)
    sizes = []
    for k in range(1, count + 1):
        block = labels == k
        n = int(block.sum())
        if n >= _DISAGREE_MIN_GATES and np.median(ratio[block]) >= _DISAGREE_BLOCK:
            sizes.append(n)
    return sizes


def _warn_tilt_disagreement(names, sweeps, results, datasets):
    """Warn where neighbouring tilts of a volume are a fold apart over a block."""
    for i, (a, b) in enumerate(pairwise(names)):
        values = []
        for k in (i, i + 1):
            v = np.asarray(results[k].values, dtype=np.float64)
            values.append(v if results[k].dims[0] == sweeps[k].ray else v.T)
        iray, igate = _sweep_mapping(datasets[a], datasets[b])
        sizes = _tilt_disagreement(
            values[0],
            values[1],
            iray,
            igate,
            min(sweeps[i].nyquist, sweeps[i + 1].nyquist),
        )
        if not sizes:
            continue
        total = int(sum(sizes))
        for k in (i, i + 1):
            results[k].attrs["dealias_disagreement_gates"] = max(
                total, int(results[k].attrs.get("dealias_disagreement_gates", 0))
            )
        warnings.warn(
            f"dealias_velocity: {a} and {b} differ by about one fold (2 Vn) over "
            f"{len(sizes)} connected block(s) of {total} gates; one of the two "
            "tilts has the wrong fold there. Compare method='unravel', or pass "
            "wind_profile or reference. The number of gates is stored in the "
            "dealias_disagreement_gates attribute.",
            UserWarning,
            stacklevel=4,
        )


def _dealias_tree(
    dtree,
    field,
    nyquist_velocity,
    wind_profile,
    sweep_continuity,
    name,
    products_only,
    confidence=False,
    method="region",
    **options,
):
    """Dealias every sweep of a volume that contains ``field``."""
    from ..grid.cone import _sweep_dataset, _sweep_names

    names = [n for n in _sweep_names(dtree) if field in dtree[n].data_vars]
    if not names:
        raise ValueError(f"No sweep contains {field!r}.")
    datasets = {n: _sweep_dataset(dtree, n) for n in names}
    elevation = {n: float(np.nanmedian(datasets[n]["elevation"].values)) for n in names}
    names.sort(key=lambda n: elevation[n])  # stable: split cuts keep their order
    if not all(is_plain(datasets[n][field]) for n in names):
        if not (sweep_continuity and len(names) > 1):
            return _dealias_lazy_tree(
                dtree,
                names,
                datasets,
                field,
                nyquist_velocity,
                wind_profile,
                name,
                products_only,
                confidence,
                method,
                options,
            )
        for n in names:  # the chain of sweeps needs the velocities
            if datasets[n][field].ndim != 2:
                raise ValueError(
                    "sweep_continuity=True needs sweeps on (azimuth, range); "
                    "pass sweep_continuity=False for sweeps with extra dimensions"
                )
            datasets[n] = datasets[n].assign({field: datasets[n][field].compute()})
    sweeps = [
        _Sweep(datasets[n], field, _nyquist(datasets[n], field, nyquist_velocity, n))
        for n in names
    ]
    if method == "unravel":
        folds = _unravel_folds(
            [datasets[n] for n in names], sweeps, sweep_continuity, options
        )
        results = [s.wrap(f, "unravel", "unravel") for s, f in zip(sweeps, folds)]
        _warn_tilt_disagreement(names, sweeps, results, datasets)
        return _tree_products(
            dtree, names, sweeps, results, name, products_only, confidence, options
        )
    # the expensive part, all sweeps in parallel
    solved = _region_folds(sweeps, **options)
    winds = [
        None if wind_profile is None else _wind_reference(datasets[n], s, wind_profile)
        for n, s in zip(names, sweeps)
    ]
    previous = mapping = None
    if sweep_continuity and len(names) > 1:
        # each sweep's absolute fold follows the dealiased sweep below it
        previous = list(range(-1, len(names) - 1))
        mapping = [None] + [
            _sweep_mapping(datasets[a], datasets[b]) for a, b in pairwise(names)
        ]
    folds = _absolute_folds(
        sweeps, solved, winds, previous=previous, mapping=mapping, **options
    )
    results = []
    for i, (n, s, f) in enumerate(zip(names, sweeps, folds)):
        if not options["max_speed"]:
            _warn_fast_pockets(s, f, stacklevel=6, name=f"{n}: ")
        ref = winds[i]
        source = "wind_profile"
        if previous is not None and previous[i] >= 0:
            ref = _fill(
                _chained_reference(
                    sweeps[previous[i]], folds[previous[i]], *mapping[i]
                ),
                ref,
            )
            source = "sweep_below"
        anchor = _anchor_of(s, solved[i][1], ref, source, f"{n}: ")
        results.append(s.wrap(f, anchor))
    _warn_tilt_disagreement(names, sweeps, results, datasets)
    return _tree_products(
        dtree, names, sweeps, results, name, products_only, confidence, options
    )


def _dealias_lazy_tree(
    dtree,
    names,
    datasets,
    field,
    nyquist_velocity,
    wind_profile,
    name,
    products_only,
    confidence,
    method,
    options,
):
    """Dealias the sweeps of a volume one by one, lazily for dask arrays."""
    sweeps, results = [], []
    for n in names:
        nyquist = _nyquist(datasets[n], field, nyquist_velocity, n)
        sweep, result = _dealias_sweep(
            datasets[n], field, nyquist, None, wind_profile, method, options
        )
        sweeps.append(sweep)
        results.append(result)
    return _tree_products(
        dtree, names, sweeps, results, name, products_only, confidence, options
    )


def _tree_products(
    dtree, names, sweeps, results, name, products_only, confidence, options
):
    """The result tree (or the input with the field added) of a dealiased volume."""
    if products_only:
        nodes = {}
        for n, s, r in zip(names, sweeps, results):
            r = r.rename(name)
            nodes[n] = _with_confidence(r, s, options) if confidence else r.to_dataset()
        return product_tree(dtree, nodes)
    out = dtree.copy()
    for n, result in zip(names, results):
        out[f"{n}/{name}"] = result.variable
    return out


def _unravel_geometry(ds, sweep):
    """Range, ground range, ground position and circle flag of a sweep."""
    rng = np.ascontiguousarray(ds["range"].values, dtype=np.float64)
    elevation = (
        float(np.nanmedian(ds["elevation"].values))
        if "elevation" in ds.variables
        else 0.0
    )
    ground = rng * np.cos(np.deg2rad(elevation))
    az = np.deg2rad(sweep.azimuth)
    x = np.ascontiguousarray(ground[None, :] * np.sin(az)[:, None])
    y = np.ascontiguousarray(ground[None, :] * np.cos(az)[:, None])
    return (rng, ground, x, y, _is_full_circle(sweep.azimuth))


def _unravel_solve(sweeps, prepared, p, options):
    """UNRAVEL on each sweep: the corrected velocities and the flags."""
    from . import _unravel_numpy as spec

    if options["compiled"]:
        solved = _unravel.unravel(
            [s.velocity for s in sweeps],
            [s.nyquist for s in sweeps],
            [s.azimuth for s in sweeps],
            [q[0] for q in prepared],
            [q[2] for q in prepared],
            [q[3] for q in prepared],
            [q[4] for q in prepared],
            p["alpha"],
            p["alpha_first"],
            p["alpha_pre"],
            p["z_cutoff"],
            p["pre_window"],
            p["beta"],
            list(p["windows"]),
            [tuple(b) for b in p["boxes"]],
            p["closest_half"],
            tuple(p["check_box"]),
            options["n_threads"],
        )
    else:
        solved = [
            spec.unravel_sweep(
                s.velocity, s.nyquist, s.azimuth, q[0], q[2], q[3], q[4], p
            )
            for s, q in zip(sweeps, prepared)
        ]
    return [np.array(o) for o, _ in solved], [np.array(f) for _, f in solved]


def _folds_of(outs, sweeps):
    """Integer folds (NaN where a gate is dropped) of corrected velocities."""
    folds = []
    for out, s in zip(outs, sweeps):
        with np.errstate(invalid="ignore"):
            folds.append(
                np.where(
                    np.isfinite(out),
                    np.rint((out - s.velocity) / (2.0 * s.nyquist)),
                    np.nan,
                )
            )
    return folds


def _unravel_folds_of(sweeps, prepared, options):
    """Folds of sweeps dealiased with UNRAVEL one by one (no 3-D check)."""
    from . import _unravel_numpy as spec

    p = dict(spec.DEFAULTS, alpha=options["alpha"])
    outs, _ = _unravel_solve(sweeps, prepared, p, options)
    return _folds_of(outs, sweeps)


def _unravel_folds(datasets, sweeps, three_d, options):
    """
    Integer folds (NaN where a gate is dropped) of sweeps dealiased with UNRAVEL.

    ``datasets`` are the sweeps ordered by elevation. Each sweep is run with
    the default driver of the paper (compiled kernel, or the Python
    specification of :mod:`radarx.retrieve._unravel_numpy`), then, if
    ``three_d`` and there are several, every sweep above the lowest is checked
    against the corrected sweep below it.
    """
    from . import _unravel_numpy as spec

    p = dict(spec.DEFAULTS, alpha=options["alpha"])
    prepared = [_unravel_geometry(ds, s) for ds, s in zip(datasets, sweeps)]
    outs, flags = _unravel_solve(sweeps, prepared, p, options)
    if three_d:
        for k in range(1, len(sweeps)):
            _check_below(k, outs, flags, sweeps, datasets, prepared, p, options)
    return _folds_of(outs, sweeps)


def _check_below(k, outs, flags, sweeps, datasets, prepared, p, options):
    """3-D check of sweep ``k`` against the corrected sweep ``k - 1`` (in place)."""
    from . import _unravel_numpy as spec

    s, below = sweeps[k], sweeps[k - 1]
    ground, ground_b = prepared[k][1], prepared[k - 1][1]
    step = np.abs((np.diff(below.azimuth) + 180.0) % 360.0 - 180.0)
    az_step = float(np.median(step)) if step.size else 1.0
    gate_step = float(np.median(np.diff(ground_b))) if ground_b.size > 1 else 1.0
    iray = _nearest_index(below.azimuth, s.azimuth, 1.5 * az_step, 360.0)
    igate = _nearest_index(ground_b, ground, 1.5 * gate_step)
    shape = outs[k].shape
    iray2 = np.ascontiguousarray(np.broadcast_to(iray[:, None], shape), dtype=np.int32)
    igate2 = np.ascontiguousarray(
        np.broadcast_to(igate[None, :], shape), dtype=np.int32
    )
    wg, wr = p["window_3d"]
    if options["compiled"]:
        new = _unravel.check_3d(
            outs[k],
            flags[k],
            outs[k - 1],
            flags[k - 1],
            iray2,
            igate2,
            s.nyquist,
            p["alpha"],
            wg,
            wr,
            prepared[k - 1][4],
        )
    else:
        new = outs[k].copy()
        spec.check_3d(
            new,
            flags[k],
            outs[k - 1],
            flags[k - 1],
            iray2,
            igate2,
            s.nyquist,
            p["alpha"],
            (wg, wr),
            prepared[k - 1][4],
        )
    outs[k] = np.asarray(new)
    done = np.isfinite(outs[k])
    flags[k] = np.where(done, np.where(outs[k] == s.velocity, 1, 2), -3).astype(np.int8)


# --------------------------------------------------------------------------
# NumPy reference implementation (identical folds to the C++ kernel)
# --------------------------------------------------------------------------


def _round_div(num, den):
    """round(num / den) with halves rounded up (integers, den > 0)."""
    return (2 * num + den) // (2 * den)


def _gate_links(nray, ngate, links):
    """Index pairs of neighbouring gates: along rays and between rays."""
    gate = np.arange(nray * ngate).reshape(nray, ngate)
    rays = np.flatnonzero(links)
    rays = rays[(rays + 1) % nray != rays]
    a = np.concatenate([gate[:, :-1].ravel(), gate[rays].ravel()])
    b = np.concatenate([gate[:, 1:].ravel(), gate[(rays + 1) % nray].ravel()])
    return a, b


def _gap_links(valid, ngate, max_gap):
    """Pairs of valid gates on one ray separated by 1 to ``max_gap`` empty gates."""
    g = np.flatnonzero(valid)
    step = np.diff(g)
    bridge = (step > 1) & (step <= max_gap + 1) & (g[:-1] // ngate == g[1:] // ngate)
    return g[:-1][bridge], g[1:][bridge]


def _ray_gap_links(valid, links, max_ray_gap):
    """Pairs of valid gates at one range separated by 1 to ``max_ray_gap`` empty rays."""
    nray, ngate = valid.shape
    gate = np.arange(nray * ngate).reshape(nray, ngate)
    a, b = [], []
    if nray <= 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    rays = np.arange(nray)
    link = links.astype(bool)
    open_ = valid & link[:, None]  # chain from ray r reaches ray r + d - 1 ... r + d
    for d in range(2, max_ray_gap + 2):
        if d >= nray:
            break
        mid = (rays + d - 1) % nray
        open_ = open_ & ~valid[mid] & link[mid][:, None]
        end = (rays + d) % nray
        hit = open_ & valid[end]
        r, j = np.nonzero(hit)
        a.append(gate[r, j])
        b.append(gate[end[r], j])
    if not a:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(a), np.concatenate(b)


def _segments(n, valid, v, ja, jb, core_joins):
    """
    Region id of every gate from the joined pairs ``ja`` - ``jb``.

    A gate with at least ``core_joins`` joined neighbours is a core gate.
    Cores connect through their joined pairs, and a core of at least
    ``_CORE_SIZE`` gates seeds a region. Every other gate takes, one ring
    per pass, the region of the assigned joined neighbour closest in
    velocity (the lower gate index on a tie), so a thin neck of in-between
    gates no longer merges the regions on its two sides. Gates that no seed
    reaches stay together as the joined pairs connect them. Without a seed,
    or with ``core_joins=0``, the regions are the connected joined gates.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    def connected(a, b):
        graph = coo_matrix((np.ones(a.size), (a, b)), shape=(n, n))
        return connected_components(graph, directed=False)[1]

    if core_joins <= 0:
        return connected(ja, jb)
    degree = np.bincount(ja, minlength=n) + np.bincount(jb, minlength=n)
    core = valid & (degree >= core_joins)
    both = core[ja] & core[jb]
    comp = connected(ja[both], jb[both])
    big = np.bincount(comp[core], minlength=comp.max() + 1) >= _CORE_SIZE
    comp = np.where(core & big[comp], comp, -1)
    if (comp < 0).all():
        return _segments(n, valid, v, ja, jb, 0)
    ea = np.concatenate([ja, jb])
    eb = np.concatenate([jb, ja])
    gap = np.abs(v[ea] - v[eb])
    # a loose gate joins its closest assigned neighbour, one ring per pass
    while True:
        sel = (comp[ea] < 0) & (comp[eb] >= 0)
        if not sel.any():
            break
        a, b, d = ea[sel], eb[sel], gap[sel]
        order = np.lexsort((b, d, a))
        a, b = a[order], b[order]
        head = np.r_[True, a[1:] != a[:-1]]
        comp[a[head]] = comp[b[head]]
    # what no core reaches stays together as before
    loose = valid & (comp < 0)
    if loose.any():
        keep = loose[ja] & loose[jb]
        comp[loose] = comp.max() + 1 + connected(ja[keep], jb[keep])[loose]
    return comp


def _region_folds_numpy(vel, links, nyq, threshold, max_gap, max_iterations):
    """Steps 1-3 (NumPy/SciPy): relative fold and component of every gate."""

    nray, ngate = vel.shape
    n = nray * ngate
    v = vel.ravel()
    valid = ~np.isnan(v)
    ga, gb = _gate_links(nray, ngate, links)
    both = valid[ga] & valid[gb]
    ga, gb = ga[both], gb[both]

    # 1. regions, numbered by their first gate
    join = np.abs(v[ga] - v[gb]) < threshold * nyq
    comp = _segments(n, valid, v, ga[join], gb[join], _CORE_JOINS)
    first = np.full(comp.max() + 1, n, dtype=np.int64)
    np.minimum.at(first, comp[valid], np.flatnonzero(valid))
    rank = np.argsort(np.argsort(first, kind="stable"), kind="stable")
    label = np.where(valid, rank[comp], -1)
    nreg = int(label.max()) + 1
    size = np.bincount(label[valid], minlength=nreg)

    # 2. region adjacency: boundary length and summed jump (fixed point),
    # also across short gaps along the ray
    gap_a, gap_b = _gap_links(valid, ngate, max_gap)
    ray_a, ray_b = _ray_gap_links(valid.reshape(nray, ngate), links, _MAX_RAY_GAP)
    ga = np.concatenate([ga, gap_a, ray_a])
    gb = np.concatenate([gb, gap_b, ray_b])
    la, lb = label[ga], label[gb]
    cross = la != lb
    ga, gb, la, lb = ga[cross], gb[cross], la[cross], lb[cross]
    d = np.where(la < lb, v[gb] - v[ga], v[ga] - v[gb])
    q = np.floor(d / (2.0 * nyq) * float(_SCALE) + 0.5).astype(np.int64)
    lo = np.minimum(la, lb).astype(np.int64)
    hi = np.maximum(la, lb).astype(np.int64)
    keys, inverse = np.unique(lo * (nreg + 1) + hi, return_inverse=True)
    edge_n = np.bincount(inverse, minlength=keys.size).astype(np.int64)
    edge_s = np.zeros(keys.size, dtype=np.int64)
    np.add.at(edge_s, inverse, q)
    edge_a = keys // (nreg + 1)
    edge_b = keys % (nreg + 1)

    # 3a. maximum spanning tree on boundary length, with fold offsets
    parent = np.arange(nreg)
    pot = np.zeros(nreg, dtype=np.int64)  # fold relative to parent

    def find(x):
        path = []
        while parent[x] != x:
            path.append(x)
            x = parent[x]
        acc = 0
        for node in reversed(path):  # nearest the root first
            acc += pot[node]
            pot[node] = acc
            parent[node] = x
        return x

    # confident boundaries first (mean jump within 0.3 of a whole number of
    # folds), then by length; ambiguous ones only join otherwise separate parts
    m_all = _round_div(edge_s, edge_n * _SCALE)
    ambiguous = 10 * np.abs(edge_s - m_all * edge_n * _SCALE) > 3 * edge_n * _SCALE
    for e in np.lexsort((-edge_n, ambiguous)):
        a, b = int(edge_a[e]), int(edge_b[e])
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        m = _round_div(int(edge_s[e]), int(edge_n[e]) * _SCALE)  # k_a - k_b
        pa = 0 if a == ra else int(pot[a])
        pb = 0 if b == rb else int(pot[b])
        delta = m - pa + pb  # k_ra - k_rb
        if ra < rb:
            parent[rb], pot[rb] = ra, -delta
        else:
            parent[ra], pot[ra] = rb, delta
    comp = np.array([find(r) for r in range(nreg)], dtype=np.int64)
    k = np.where(comp == np.arange(nreg), 0, pot).astype(np.int64)

    # 3b. integer least squares: coordinate descent on regions and blocks
    _icm(nreg, edge_a, edge_b, edge_n, edge_s, size, k, max_iterations)
    for _ in range(max_iterations):
        # joined by a boundary that agrees with the folds within 1/4 fold
        dk = k[edge_a] - k[edge_b]
        consistent = 4 * np.abs(edge_s - dk * edge_n * _SCALE) <= edge_n * _SCALE
        sid, nsup = _first_index_labels(nreg, edge_a[consistent], edge_b[consistent])
        if nsup == nreg:
            break
        ssize = np.bincount(sid, weights=size, minlength=nsup).astype(np.int64)
        A, B = sid[edge_a], sid[edge_b]
        cross = A != B
        t = edge_n * (k[edge_b] - k[edge_a]) * _SCALE + edge_s  # block jump A -> B
        A, B, t, en = A[cross], B[cross], t[cross], edge_n[cross]
        lo, hi = np.minimum(A, B), np.maximum(A, B)
        t = np.where(A < B, t, -t)
        keys, inverse = np.unique(lo * (nsup + 1) + hi, return_inverse=True)
        sn = np.zeros(keys.size, dtype=np.int64)
        ss = np.zeros(keys.size, dtype=np.int64)
        np.add.at(sn, inverse, en)
        np.add.at(ss, inverse, t)
        delta = np.zeros(nsup, dtype=np.int64)
        _icm(
            nsup,
            keys // (nsup + 1),
            keys % (nsup + 1),
            sn,
            ss,
            ssize,
            delta,
            max_iterations,
        )
        if not delta.any():
            break
        k += delta[sid]
        _icm(nreg, edge_a, edge_b, edge_n, edge_s, size, k, max_iterations)

    folds = np.zeros(n, dtype=np.int32)
    comps = np.full(n, -1, dtype=np.int32)
    folds[valid] = k[label[valid]]
    comps[valid] = comp[label[valid]]
    return folds.reshape(nray, ngate), comps.reshape(nray, ngate)


def _first_index_labels(n, a, b):
    """Connected components of ``n`` nodes, numbered by their smallest node."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    graph = coo_matrix((np.ones(a.size), (a, b)), shape=(n, n))
    ncomp, comp = connected_components(graph, directed=False)
    first = np.full(ncomp, n, dtype=np.int64)
    np.minimum.at(first, comp, np.arange(n))
    rank = np.argsort(np.argsort(first, kind="stable"), kind="stable")
    return rank[comp], ncomp


def _icm(nnode, edge_a, edge_b, edge_n, edge_s, size, k, max_iterations):
    """Integer Gauss-Seidel descent on the boundary least squares (in place)."""
    src = np.concatenate([edge_a, edge_b])
    dst = np.concatenate([edge_b, edge_a])
    nbn = np.concatenate([edge_n, edge_n])
    order = np.argsort(src, kind="stable")
    src, dst, nbn = src[order], dst[order], nbn[order]
    start = np.searchsorted(src, np.arange(nnode + 1))
    den = np.zeros(nnode, dtype=np.int64)
    sum_s = np.zeros(nnode, dtype=np.int64)
    np.add.at(den, edge_a, edge_n)
    np.add.at(den, edge_b, edge_n)
    np.add.at(sum_s, edge_a, edge_s)
    np.subtract.at(sum_s, edge_b, edge_s)
    nodes = [r for r in np.argsort(-size, kind="stable") if start[r] < start[r + 1]]
    for _ in range(max_iterations):
        changed = 0
        for r in nodes:
            s0, s1 = start[r], start[r + 1]
            nk = int(np.dot(nbn[s0:s1], k[dst[s0:s1]]))
            kn = _round_div(nk * _SCALE + int(sum_s[r]), int(den[r]) * _SCALE)
            if kn != k[r]:
                k[r] = kn
                changed += 1
        if not changed:
            break


def _quant(x, q):
    return np.floor(x * q + 0.5).astype(np.int64)


def _mode_votes(gc, votes, ncomp):
    """Most common vote per component: (components with votes, their vote)."""
    if not gc.size:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    pairs, counts = np.unique(np.stack([gc, votes], axis=1), axis=0, return_counts=True)
    # most gates, then smaller |vote|, then the negative one
    best = np.lexsort((pairs[:, 1], np.abs(pairs[:, 1]), -counts, pairs[:, 0]))
    pairs = pairs[best]
    head = np.r_[True, pairs[1:, 0] != pairs[:-1, 0]]
    return pairs[head, 0], pairs[head, 1]


def _votes(r, v, k, nyq):
    t = (r - v) / (2.0 * nyq)
    return np.clip(np.floor((t - k) + 0.5), -_MAX_VOTE, _MAX_VOTE).astype(np.int64)


def _zero_mean(gc, gv, gk, nyq, ncomp, shift, want):
    sel = want[gc]
    q = _quant(gv[sel] / (2.0 * nyq), float(_SCALE)) + gk[sel] * _SCALE
    total = np.zeros(ncomp, dtype=np.int64)
    np.add.at(total, gc[sel], q)
    count = np.bincount(gc[sel], minlength=ncomp).astype(np.int64)
    for c in np.flatnonzero(count):
        shift[c] = _round_div(-int(total[c]), int(count[c]) * _SCALE)


def _cross(a, b, c, d):
    return a * b - c * d


def _vad_fit(vel, folds, comp, anchor, rsin, rcos, nyq, window):
    """Per range gate VAD coefficients (a0, a1, a2) of the anchor, or None."""
    ngate = vel.shape[1]
    sel = comp == anchor
    s = np.broadcast_to(_quant(rsin, _TRIG_Q)[:, None], vel.shape)
    c = np.broadcast_to(_quant(rcos, _TRIG_Q)[:, None], vel.shape)
    u = _quant(np.where(sel, vel, 0.0), _VEL_Q) + folds.astype(np.int64) * _quant(
        2.0 * nyq, _VEL_Q
    )
    terms = [np.ones_like(u), s, c, s * s, s * c, c * c, u, u * s, u * c]
    acc = np.zeros((9, ngate + 1), dtype=np.int64)
    for q, term in enumerate(terms):
        acc[q, 1:] = np.cumsum(np.where(sel, term, 0).sum(axis=0))
    j = np.arange(ngate)
    lo = np.maximum(0, j - window)
    hi = np.minimum(ngate, j + window + 1)
    m = (acc[:, hi] - acc[:, lo]).astype(np.float64)
    T, V = _TRIG_Q, _VEL_Q
    n = m[0]
    s, c = m[1] / T, m[2] / T
    ss, sc, cc = m[3] / (T * T), m[4] / (T * T), m[5] / (T * T)
    bv, bs, bc = m[6] / V, m[7] / (V * T), m[8] / (V * T)
    with np.errstate(divide="ignore", invalid="ignore"):
        k1 = _cross(ss, cc, sc, sc)
        k2 = _cross(s, cc, sc, c)
        k3 = _cross(s, sc, ss, c)
        det = _cross(n, k1, s, k2) + c * k3
        good = (n >= _VAD_MIN_GATES) & (det / (n * n * n) >= _VAD_MIN_SPREAD)
        e1 = _cross(bs, cc, sc, bc)
        e2 = _cross(bs, sc, ss, bc)
        det0 = _cross(bv, k1, s, e1) + c * e2
        e3 = _cross(s, bc, bs, c)
        det1 = _cross(n, e1, bv, k2) + c * e3
        e4 = _cross(ss, bc, bs, sc)
        det2 = _cross(n, e4, s, e3) + bv * k3
        coef = np.stack([det0 / det, det1 / det, det2 / det], axis=1)
    fitted = np.flatnonzero(good)
    if not fitted.size:
        return None
    # unfitted gates: nearest fitted gate (ties: the one nearer the radar)
    p = np.searchsorted(fitted, j)
    after = fitted[np.minimum(p, fitted.size - 1)]
    before = fitted[np.maximum(p - 1, 0)]
    src = np.where(
        p == fitted.size,
        fitted[-1],
        np.where(
            (p == 0) | (after == j),
            after,
            np.where(after - j < j - before, after, before),
        ),
    )
    return coef[src]


def _neighbours(arrays, valid, links):
    """
    The 8 neighbours of every gate: for each of them ``(ok, *shifted)``, the
    mask of gates that have a valid neighbour there and the ``arrays`` moved
    onto the gate (0 where there is none). The neighbours are the gates beside
    it on the ray and the three on each linked ray beside the ray.
    """
    nray, ngate = valid.shape
    rays = np.arange(nray)
    prev_ray = (rays - 1) % nray
    next_ray = (rays + 1) % nray
    use_prev = (links[prev_ray] > 0) & (prev_ray != rays)
    use_next = (
        (links > 0)
        & (next_ray != rays)
        & (next_ray != np.where(use_prev, prev_ray, -1))
    )
    for source, use in ((rays, None), (prev_ray, use_prev), (next_ray, use_next)):
        for dj in (-1, 0, 1):
            if use is None and dj == 0:
                continue
            ok = np.zeros(valid.shape, dtype=bool)
            moved = [np.zeros_like(a) for a in arrays]
            lo, hi = max(0, -dj), min(ngate, ngate - dj)
            ok[:, lo:hi] = valid[source][:, lo + dj : hi + dj]
            if use is not None:
                ok &= use[:, None]
            for out, a in zip(moved, arrays):
                out[:, lo:hi] = a[source][:, lo + dj : hi + dj]
            yield (ok, *moved)


def _gate_check(vel, folds, comp, links, nyq, passes):
    """Refold gates that differ by more than Vn from all (>= 3) neighbours."""
    valid = comp >= 0
    twovn_q = int(_quant(2.0 * nyq, _VEL_Q))
    vn_q = int(_quant(nyq, _VEL_Q))
    vq = np.where(valid, _quant(np.where(valid, vel, 0.0), _VEL_Q), 0)
    f = folds.astype(np.int64)
    for _ in range(passes):
        u = vq + f * twovn_q
        cnt = np.zeros(vel.shape, dtype=np.int64)
        far = np.zeros(vel.shape, dtype=np.int64)
        total = np.zeros(vel.shape, dtype=np.int64)
        for ok, w in _neighbours((u,), valid, links):
            cnt += ok
            total += np.where(ok, w, 0)
            far += ok & (np.abs(w - u) > vn_q)
        change = valid & (cnt >= _MIN_NEIGHBOURS) & (far == cnt)
        num = total - cnt * u
        den = np.maximum(cnt, 1) * twovn_q
        d = np.where(change, (2 * num + den) // (2 * den), 0)
        if not d.any():
            break
        f = f + d
    return f.astype(np.int32)


def _cluster_labels(mask, valid, links, same=None):
    """
    Connected components of the gates of ``mask``: their labels and their number.

    With ``same``, neighbouring gates are only joined when they hold the same
    value of that array (the fold, for the pockets of one fold).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    index = np.arange(mask.size).reshape(mask.shape)
    arrays = (mask, index) if same is None else (mask, index, same)
    first, second = [], []
    for ok, inside, iw, *other in _neighbours(arrays, valid, links):
        joined = ok & mask & inside
        if other:
            joined &= other[0] == same
        first.append(index[joined])
        second.append(iw[joined])
    a, b = np.concatenate(first), np.concatenate(second)
    graph = coo_matrix((np.ones(a.size), (a, b)), shape=(mask.size,) * 2)
    count, label = connected_components(graph, directed=False)
    return label.reshape(mask.shape), count


def _cluster_sizes(mask, valid, links):
    """Gates in the cluster (connected gates of ``mask``) of every gate of the mask."""
    label, count = _cluster_labels(mask, valid, links)
    return np.where(mask, np.bincount(label.ravel(), minlength=count)[label], 0)


def _speed_guard(vel, folds, comp, links, nyq, max_speed):
    """
    Move gates that the folds put beyond ``max_speed`` back by whole intervals.

    Only a cluster of at most ``_GUARD_SIZE`` connected gates beyond the limit
    is examined. A folded gate of it goes to the fold one step nearer zero when
    that lowers the summed absolute velocity difference to its neighbours that
    are folded less than it (|fold| smaller). A gate that matches such a
    neighbour only with its fold keeps it. Passes repeat until nothing moves,
    so a core is undone from its edge.
    """
    valid = comp >= 0
    twovn_q = int(_quant(2.0 * nyq, _VEL_Q))
    cap_q = int(_quant(max_speed, _VEL_Q))
    vq = np.where(valid, _quant(np.where(valid, vel, 0.0), _VEL_Q), 0)
    f = folds.astype(np.int64)
    beyond = valid & (f != 0) & (np.abs(vq + f * twovn_q) > cap_q)
    small = _cluster_sizes(beyond, valid, links) <= _GUARD_SIZE
    for _ in range(_GUARD_PASSES):
        u = vq + f * twovn_q
        step = np.sign(f) * twovn_q
        net = np.zeros(vel.shape, dtype=np.int64)
        for ok, w, fw in _neighbours((u, f), valid, links):
            less = ok & (np.abs(fw) < np.abs(f))
            net += np.where(less, np.abs(u - w) - np.abs(u - step - w), 0)
        back = small & beyond & (np.abs(u) > cap_q) & (net > 0)
        if not back.any():
            break
        f = f - np.where(back, np.sign(f), 0)
    return f.astype(np.int32)


def _absolute_folds_numpy(vel, k, comp, nyq, ref, rsin, rcos, links, max_speed=0.0):
    """Step 4 (NumPy): anchor, in-sweep VAD reference, gate check and speed guard."""
    folds = np.zeros(vel.shape, dtype=np.int32)
    valid = comp >= 0
    if not valid.any():
        return folds
    ncomp = int(comp.max()) + 1
    gc = comp[valid].astype(np.int64)
    gk = k[valid].astype(np.int64)
    gv = vel[valid]
    csize = np.bincount(gc, minlength=ncomp)
    anchor = int(np.argmax(csize))
    shift = np.zeros(ncomp, dtype=np.int64)
    ext = None if ref is None else np.asarray(ref, dtype=np.float64)[valid]

    # 1. anchor: reference votes, else zero mean
    voted = np.zeros(ncomp, dtype=bool)
    sel = gc == anchor
    if ext is not None:
        sel &= ~np.isnan(ext)
    if ext is not None and sel.any():
        comps, best = _mode_votes(
            gc[sel], _votes(ext[sel], gv[sel], gk[sel], nyq), ncomp
        )
        shift[comps] = best
        voted[comps] = True
    else:
        want = np.zeros(ncomp, dtype=bool)
        want[anchor] = True
        _zero_mean(gc, gv, gk, nyq, ncomp, shift, want)
    first = np.zeros(vel.shape, dtype=np.int32)
    first[comp == anchor] = k[comp == anchor] + shift[anchor]

    # 2. other components: reference, else the anchor's VAD, else zero mean
    coef = _vad_fit(vel, first, comp, anchor, rsin, rcos, nyq, _VAD_WINDOW)
    refval = (
        np.full(vel.shape, np.nan) if ref is None else np.asarray(ref, dtype=np.float64)
    )
    if coef is not None:
        vad = (coef[None, :, 0] + coef[None, :, 1] * rsin[:, None]) + coef[
            None, :, 2
        ] * rcos[:, None]
        refval = np.where(np.isnan(refval), vad, refval)
    r = refval[valid]
    sel = (gc != anchor) & ~np.isnan(r)
    comps, best = _mode_votes(gc[sel], _votes(r[sel], gv[sel], gk[sel], nyq), ncomp)
    shift[comps] = best
    voted[comps] = True
    want = ~voted
    want[anchor] = False
    _zero_mean(gc, gv, gk, nyq, ncomp, shift, want)
    folds[valid] = gk + shift[gc]

    # 3. gate check, then the speed limit
    folds = _gate_check(vel, folds, comp, links, nyq, _GATE_PASSES)
    if max_speed > 0:
        folds = _speed_guard(vel, folds, comp, links, nyq, max_speed)
    return folds


# --------------------------------------------------------------------------
# local median, dealiasing confidence and dual-PRF error correction
# --------------------------------------------------------------------------

_DEALIASED = tuple(f"{n}_dealiased" for n in VELOCITY) + VELOCITY


def _is_full_circle(azimuth):
    """Whether the rays cover the whole circle (the first ray follows the last)."""
    az = np.sort(np.mod(np.asarray(azimuth, dtype=np.float64), 360.0))
    if az.size < 3:
        return False
    typical = np.median(np.diff(az))
    return bool(360.0 - (az[-1] - az[0]) <= 2.0 * typical)


def _local_median_numpy(vel, wrap, half_start, half_max, min_points, include_centre):
    """
    NumPy implementation of the compiled ``local_median`` kernel (same values).

    The window grows from half-width ``half_start`` to ``half_max``; each gate
    takes the median of the first window with at least ``min_points`` valid
    gates (``include_centre`` says whether the gate counts). Gates are
    gathered window by window for the gates still without a median.
    """
    nray, ngate = vel.shape
    out = np.full(vel.shape, np.nan)
    todo = np.argwhere(np.isfinite(vel))
    half_max = min(half_max, (nray - 1) // 2) if wrap else half_max
    for h in range(half_start, half_max + 1):
        if not len(todo):
            break
        rows, cols = todo[:, 0], todo[:, 1]
        stack = []
        for dr in range(-h, h + 1):
            r = rows + dr
            ok_r = np.ones(r.shape, dtype=bool)
            if wrap:
                r = r % nray
            else:
                ok_r = (r >= 0) & (r < nray)
                r = np.clip(r, 0, nray - 1)
            for dc in range(-h, h + 1):
                if not include_centre and dr == 0 and dc == 0:
                    continue
                c = cols + dc
                ok = ok_r & (c >= 0) & (c < ngate)
                stack.append(np.where(ok, vel[r, np.clip(c, 0, ngate - 1)], np.nan))
        window = np.stack(stack, axis=1)
        count = np.isfinite(window).sum(axis=1)
        done = count >= min_points
        if done.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(window[done], axis=1)
            out[rows[done], cols[done]] = med
        todo = todo[~done]
    return out


def _local_median(
    vel, wrap, half_start, half_max, min_points, include_centre, n_threads, compiled
):
    """Window medians of a (ray, range) array; see :func:`_local_median_numpy`."""
    if compiled:
        return _dealias.local_median(
            vel,
            bool(wrap),
            int(half_start),
            int(half_max),
            int(min_points),
            bool(include_centre),
            int(n_threads or 0),
        )
    return _local_median_numpy(
        vel, wrap, half_start, half_max, min_points, include_centre
    )


def _disagreement_numpy(values, wrap, half, threshold, min_neighbours):
    """
    NumPy implementation of the compiled ``neighbour_disagreement`` kernel.

    For each valid gate, the fraction of its valid neighbours (window of
    half-width ``half``, the gate left out) that differ from it by more than
    ``threshold``; NaN with fewer than ``min_neighbours`` valid neighbours.
    """
    nray, ngate = values.shape
    if wrap:
        half = min(half, (nray - 1) // 2)
    padded = np.full((nray, ngate + 2 * half), np.nan)
    padded[:, half : half + ngate] = values
    valid = np.zeros(values.shape, dtype=np.int64)
    far = np.zeros(values.shape, dtype=np.int64)
    for dr in range(-half, half + 1):
        if wrap:
            rows = np.roll(padded, -dr, axis=0)
        else:
            rows = np.full_like(padded, np.nan)
            lo, hi = max(0, -dr), min(nray, nray - dr)
            rows[lo:hi] = padded[lo + dr : hi + dr]
        for dc in range(-half, half + 1):
            if dr == 0 and dc == 0:
                continue
            neighbour = rows[:, half + dc : half + dc + ngate]
            ok = np.isfinite(neighbour)
            valid += ok
            far += ok & (np.abs(values - neighbour) > threshold)
    fraction = far / np.maximum(valid, 1)
    return np.where(np.isfinite(values) & (valid >= min_neighbours), fraction, np.nan)


def _confidence_values(values, azimuth, nyquist, options):
    """Confidence (1 - share of far neighbours) of dealiased velocities, float32."""
    wrap = _is_full_circle(azimuth)
    threshold = options["threshold"] * nyquist
    if options["compiled"]:
        fraction = _dealias.neighbour_disagreement(
            values,
            bool(wrap),
            int(options["window"]),
            float(threshold),
            int(options["min_neighbours"]),
            int(options["n_threads"] or 0),
        )
    else:
        fraction = _disagreement_numpy(
            values, wrap, options["window"], threshold, options["min_neighbours"]
        )
    return (1.0 - fraction).astype(np.float32)


def _confidence_attrs(options, nyquist):
    window = options["window"]
    return {
        "long_name": "confidence of the dealiased velocity",
        "units": "1",
        "valid_range": [0.0, 1.0],
        "comment": "share of the valid neighbouring gates that differ from the gate "
        "by no more than threshold * Vn, so that they prefer the chosen fold "
        "to the next one (a radarx definition, no paper defines it)",
        "window": f"{2 * window + 1} x {2 * window + 1} gates, centre left out",
        "min_neighbours": options["min_neighbours"],
        "threshold": options["threshold"],
        "nyquist_velocity": nyquist,
    }


def _confidence_options(window, min_neighbours, threshold, n_threads, compiled):
    if int(window) != window or window < 1:
        raise ValueError("window must be a positive integer (half-width in gates)")
    if int(min_neighbours) != min_neighbours or not 1 <= min_neighbours <= (
        (2 * window + 1) ** 2 - 1
    ):
        raise ValueError(
            "min_neighbours must be between 1 and the window size minus one"
        )
    if not threshold > 0:
        raise ValueError("threshold must be positive")
    return {
        "window": int(window),
        "min_neighbours": int(min_neighbours),
        "threshold": float(threshold),
        "n_threads": n_threads,
        "compiled": compiled,
    }


@dask_support("lazy")
@renamed_arguments(radar="obj")
@provenance(
    "Confidence of dealiased velocity from the neighbouring gates, a radarx definition"
)
def dealias_confidence(
    obj,
    field=None,
    nyquist_velocity=None,
    *,
    window=1,
    min_neighbours=3,
    threshold=1.0,
    name=None,
    n_threads=None,
    engine="auto",
):
    """
    Confidence of the fold chosen for every gate of a dealiased velocity field.

    A gate with the dealiased velocity :math:`v` could also have been unfolded
    to :math:`v \\pm 2 V_n`. A neighbouring gate with the velocity :math:`v_j`
    is closer to :math:`v` than to the next candidate fold if
    :math:`|v - v_j| < V_n`, and the confidence is the share of the valid
    neighbours that are:

    .. math::

        c = 1 - \\frac{\\#\\{j : |v - v_j| > \\mathrm{threshold}\\, V_n\\}}
                       {\\#\\{j\\}}

    with the neighbours :math:`j` the ``(2 window + 1)`` by ``(2 window + 1)``
    gates around the gate (rays wrap around north for full-circle sweeps), the
    gate itself left out, and at least ``min_neighbours`` of them valid.
    :math:`c = 1` where all neighbours support the chosen fold, :math:`c = 0`
    where none does (the case in which the gate check of
    :func:`dealias_velocity` refolds a gate), and a gate on a straight fold
    line has :math:`c = 5/8` with ``window=1``. This is a radarx definition:
    no paper defines a confidence for the dealiasing methods cited in
    :mod:`radarx.retrieve.dealias`. It is computed from the result alone, so it
    also applies to the output of another method or of
    :func:`correct_dual_prf`.

    What it says. It marks isolated gates and small clusters that disagree with
    their surroundings, and the two sides of the lines along which the folds
    change. It does not say that a region is right: a large region that is
    unfolded to the wrong fold as a whole has :math:`c = 1` inside, and is low
    only along its edge. Real shear and turbulence also lower :math:`c` where
    the velocity changes by more than :math:`V_n` between neighbouring gates.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataTree
        A sweep or a volume with the dealiased velocity and the ``azimuth``
        and ``range`` coordinates.
    field : str, optional
        Dealiased velocity. Default: the first of ``VRADH_dealiased``,
        ``VRAD_dealiased``, ... (the names :func:`dealias_velocity` gives),
        then ``VRADH``, ``VRAD``, ... that the sweep holds.
    nyquist_velocity : float or dict, optional
        Nyquist velocity (m s⁻¹), as in :func:`dealias_velocity`. By default
        it is read from the sweep, or from the ``nyquist_velocity`` attribute
        that :func:`dealias_velocity` writes on its result.
    window : int, optional
        Half-width of the neighbourhood in rays and gates. Default 1 (3 x 3,
        the neighbourhood of the gate check of :func:`dealias_velocity`).
    min_neighbours : int, optional
        Fewest valid neighbours for a confidence. Default 3, the number that
        the gate check also requires.
    threshold : float, optional
        Difference, in Nyquist velocities, beyond which a neighbour does not
        support the fold. Default 1: the neighbour is then closer to the next
        fold. Values below 1 count smaller differences as disagreement.
    name : str, optional
        Name of the result. Default ``f"{field}_confidence"``.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use; both give the same values.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        The confidence (float32, NaN where the gate has no data or too few
        neighbours) on the sweep's dimensions; a DataTree with one node per
        sweep that has ``field`` for a volume. Merge it with
        ``.radarx.assign``.

    Examples
    --------
    >>> dea = dtree.radarx.dealias("VRADH")  # doctest: +SKIP
    >>> conf = dtree.radarx.assign(dea).radarx.dealias_confidence()  # doctest: +SKIP
    """
    options = _confidence_options(
        window, min_neighbours, threshold, n_threads, _use_compiled(engine)
    )
    obj = as_sweep(obj)
    field = find_field(obj, _DEALIASED, field)
    if isinstance(obj, xr.Dataset):
        return _confidence_sweep(obj, field, nyquist_velocity, None, name, options)
    nodes = {}
    for sweep in _sweep_names_with(obj, field):
        ds = _sweep_dataset_of(obj, sweep)
        nodes[sweep] = _confidence_sweep(
            ds, field, nyquist_velocity, sweep, name, options
        ).to_dataset()
    if not nodes:
        raise ValueError(f"No sweep contains {field!r}.")
    return product_tree(obj, nodes)


def _sweep_names_with(dtree, field):
    from ..grid.cone import _sweep_names

    return [n for n in _sweep_names(dtree) if field in dtree[n].data_vars]


def _sweep_dataset_of(dtree, sweep):
    from ..grid.cone import _sweep_dataset

    return _sweep_dataset(dtree, sweep)


def _confidence_sweep(ds, field, nyquist, sweep, name, options):
    vn = _nyquist(ds, field, nyquist, sweep)
    da = ds[field]
    conf = _confidence(da, ray_coordinate(ds, "azimuth"), vn, options)
    return xr.DataArray(
        conf.data,
        dims=da.dims,
        coords=da.coords,
        name=name or f"{field}_confidence",
        attrs=_confidence_attrs(options, vn),
    )


def _prf_flags(ds, ray, prf, first_ray):
    """Boolean per ray, True for a high-PRF ray (see :func:`correct_dual_prf`)."""
    n = ds.sizes[ray]
    if prf is None:
        if first_ray not in ("high", "low"):
            raise ValueError(f"first_ray must be 'high' or 'low', not {first_ray!r}")
        return (np.arange(n) % 2 == 0) == (first_ray == "high")
    flags = ds[prf] if isinstance(prf, str) else prf
    values = np.asarray(flags.values if hasattr(flags, "values") else flags)
    if values.shape != (n,):
        raise ValueError(
            f"prf must have one value per ray ({n}), not shape {values.shape}"
        )
    return values.astype(bool)


def _dual_prf_core(batch, vh, vl, min_points, max_window, passes, n_threads, compiled):
    """Dual-PRF correction of sweeps given as ``((high, wrap), (velocity,))``."""
    half_max = (max_window - 1) // 2
    results = []
    for (high, wrap), arrays in batch:
        current = np.ascontiguousarray(arrays[0], dtype=np.float64)
        vu = np.where(high, vh, vl)[:, None]
        shift = np.zeros(current.shape, dtype=np.int32)
        for _ in range(passes):
            med = _local_median(
                current, wrap, 1, half_max, min_points, True, n_threads, compiled
            )
            with np.errstate(invalid="ignore"):
                outlier = np.isfinite(med) & (np.abs(current - med) > vu)
                k = np.floor((med - current) / (2.0 * vu) + 0.5)
            if not outlier.any():
                break
            k = np.where(outlier, k, 0.0)
            current = current + 2.0 * vu * k
            shift += k.astype(np.int32)
        results.append((current, shift))
    return results


def _dual_prf_sweep(
    ds,
    field,
    vh,
    vl,
    prf,
    first_ray,
    min_points,
    max_window,
    passes,
    name,
    n_threads,
    compiled,
):
    if field not in ds:
        raise KeyError(f"{field!r} is not in the sweep")
    da = ds[field]
    ray = ray_dimension(da)
    high = _prf_flags(ds, ray, prf, first_ray)
    wrap = _is_full_circle(ray_coordinate(ds, "azimuth"))
    core = functools.partial(
        _dual_prf_core,
        vh=vh,
        vl=vl,
        min_points=min_points,
        max_window=max_window,
        passes=passes,
        n_threads=n_threads,
        compiled=compiled,
    )
    dtype = da.dtype if np.issubdtype(da.dtype, np.floating) else np.float64
    out, shift = apply_sweeps(
        core,
        Sweep([da], context=(high, wrap)),
        (Output(dtype), Output(np.int32)),
    )
    attrs = dict(da.attrs)
    attrs.update(
        {
            "long_name": "Dual-PRF corrected "
            + da.attrs.get("long_name", "radial velocity"),
            "comment": "outliers of the dual-PRF velocity moved by multiples of twice the "
            "unambiguous velocity of their PRF towards the local median "
            "(Holleman and Beekhuis 2003, local dealiasing step)",
            "nyquist_high_prf": vh,
            "nyquist_low_prf": vl,
            "nyquist_extended": vh * vl / (vh - vl),
            "unfolding_factor": vl / (vh - vl),
            "min_points": min_points,
            "max_window": max_window,
            "passes": passes,
        }
    )
    if is_plain(da):
        attrs["outliers_corrected"] = int((shift.values != 0).sum())
    corrected = da.copy(data=out.data).assign_attrs(attrs)
    shifts = xr.DataArray(
        shift.data,
        dims=da.dims,
        coords=da.coords,
        attrs={
            "long_name": "multiples of twice the unambiguous velocity added by the dual-PRF correction",
            "units": "1",
        },
    )
    return xr.Dataset({name: corrected, f"{name}_shift": shifts})


def _per_sweep(value, sweep):
    return value.get(sweep) if isinstance(value, dict) else value


def _positive(value, what, sweep):
    if value is None:
        where = f" for {sweep!r}" if sweep else ""
        raise ValueError(f"{what} is needed{where}")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{what} must be positive, got {value}.")
    return value


@dask_support(
    "lazy",
    "The attribute ``outliers_corrected`` is written for a sweep in memory only.",
)
@renamed_arguments(radar="obj")
@provenance(
    "Dual-PRF velocity error correction by the local median (Holleman and Beekhuis 2003)"
)
def correct_dual_prf(
    obj,
    field=None,
    *,
    nyquist_high,
    nyquist_low,
    prf=None,
    first_ray="high",
    min_points=9,
    max_window=9,
    passes=2,
    name=None,
    n_threads=None,
    engine="auto",
):
    """
    Correct the dealiasing errors of dual-PRF radial velocities.

    A dual-PRF radar alternates two pulse repetition frequencies from ray to
    ray and unfolds the velocity of each ray with the difference of the two
    velocity estimates. The difference is noisy, so now and then a gate is
    unfolded to the wrong Nyquist interval of its own PRF and stands out as an
    isolated outlier whose error is a multiple of twice the unambiguous
    velocity of that ray's PRF. This function finds the outliers by their
    distance from the local median velocity and moves them back, the local
    dealiasing step of Holleman and Beekhuis (2003) [1]_.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataTree
        A sweep, or a volume whose sweeps holding ``field`` are processed one
        by one. The velocity is the output of the radar processor, already
        unfolded to the extended interval.
    field : str, optional
        Radial velocity field. Default: the first of ``VRADH``, ``VRAD``,
        ``VRADV``, ``VEL``, ``VELOCITY``, ``velocity`` that the sweep holds.
    nyquist_high, nyquist_low : float or dict
        Nyquist (unambiguous) velocities of the high and the low PRF in
        m s⁻¹, :math:`V_{uh} > V_{ul} > 0` (Eq. 1 of the paper,
        :math:`V_u = \\lambda\\,\\mathrm{PRF}/4`). For a volume a dict maps
        sweep names to values. The extended interval
        :math:`V_{ulh} = V_{uh} V_{ul} / (V_{uh} - V_{ul})` (Eq. 3) and the
        unfolding factor :math:`N = V_{ul} / (V_{uh} - V_{ul})` (Eq. 4) are
        written to the attributes.
    prf : str or array-like, optional
        Which rays used the high PRF: the name of a variable of the sweep, or
        an array with one value per ray, true (or 1) for the high PRF. Default
        None: the rays alternate, starting with ``first_ray``. This needs the
        rays to be in the order of acquisition.
    first_ray : {"high", "low"}, optional
        PRF of the first ray when ``prf`` is None. Default ``"high"``. The
        paper finds the PRF of each ray from the position of the outlier
        sidebands of the histogram of deviations (its Fig. 2) when the radar
        does not record it; that is not done here.
    min_points : int, optional
        Valid gates that the window of the local median must hold, including
        the gate. Default 9, the value of the paper.
    max_window : int, optional
        Largest edge of the window, an odd number of gates (rays and range).
        The window is 3 x 3 first and grows by 2 until it holds
        ``min_points`` valid gates (the paper says only that it is "enlarged
        step by step"); gates whose largest window is still short are left
        unchanged. Default 9, a radarx choice.
    passes : int, optional
        Number of passes. Each pass takes the medians of the field the previous
        pass left. Default 2: in the 300 scans of the paper (Table 2) one pass
        lowers the fraction of outliers from 1.1e-2 to 3.9e-4 and the second
        to 1.1e-4, a radarx choice.
    name : str, optional
        Name of the corrected velocity. Default ``f"{field}_dualprf"``; the
        number of shifts is ``f"{name}_shift"``.
    n_threads : int, optional
        Threads for the compiled kernel. Default: all cores.
    engine : {"auto", "compiled", "numpy"}, optional
        Implementation to use; both give the same values.

    Returns
    -------
    xarray.Dataset or xarray.DataTree
        The corrected velocity and ``f"{name}_shift"``, the multiples of twice
        the ray's unambiguous velocity that were added (0 for the unchanged
        gates), on the sweep's dimensions; a DataTree with one node per sweep
        that has ``field`` for a volume.

    Notes
    -----
    A gate is an outlier when its velocity differs from the median of the
    valid gates of its window (the gate included) by more than the
    unambiguous velocity of its ray's PRF. It is moved by the multiple
    :math:`k` of twice that velocity that brings it closest to the median,
    :math:`v + 2 k V_u`, so no smoothing is applied (the paper's last step
    description). The window is cut at the ends of a ray and wraps around
    north for full-circle sweeps.

    What is taken from the paper is the local dealiasing step:
    the median of at least nine valid gates in a 3 x 3 square enlarged when
    needed, the test against the unambiguous velocity and the correction by
    multiples of twice the unambiguous velocity of the ray's PRF, with several
    passes allowed. What is not implemented: the group-size filter (step 1,
    which removes clutter and noise; its minimum group size falls from 25 gates
    at zero range to 10 at the maximum range), the global dealiasing with a
    wind profile (step 2, for velocities beyond the extended interval; use
    :func:`dealias_velocity` with ``nyquist_velocity=V_ulh`` and a
    ``wind_profile`` before or after), and the reconstruction of the PRF
    labels. The window rule of the paper relies on the minimum group size of
    its first step; a fixed ``min_points`` is used here. The algorithm of Joe
    and May (2003, https://doi.org/10.1175/1520-0426(2003)20<429:CODPVE>2.0.CO;2)
    is not implemented. The paper tests the algorithm
    on one C-band radar with an unfolding factor of 3; where outliers cluster
    (strong shear, large spectrum width) several passes are needed, and the
    median of a window that holds genuine strong shear can move true gates.

    References
    ----------
    .. [1] Holleman, I., and H. Beekhuis, 2003: Analysis and correction of
       dual PRF velocity data. *J. Atmos. Oceanic Technol.*, **20** (4),
       443-453,
       https://doi.org/10.1175/1520-0426(2003)20<443:AACODP>2.0.CO;2

    Examples
    --------
    >>> fixed = radarx.retrieve.correct_dual_prf(  # doctest: +SKIP
    ...     sweep, "VRADH", nyquist_high=13.3, nyquist_low=10.0)
    """
    if int(passes) != passes or passes < 1:
        raise ValueError("passes must be a positive integer")
    if int(min_points) != min_points or min_points < 1:
        raise ValueError("min_points must be a positive integer")
    if int(max_window) != max_window or max_window < 3 or max_window % 2 == 0:
        raise ValueError("max_window must be an odd number of at least 3")
    compiled = _use_compiled(engine)
    obj = as_sweep(obj)
    field = find_field(obj, VELOCITY, field)
    name = name or f"{field}_dualprf"
    options = (min_points, max_window, int(passes), name, n_threads, compiled)

    def limits(sweep):
        vh = _positive(_per_sweep(nyquist_high, sweep), "nyquist_high", sweep)
        vl = _positive(_per_sweep(nyquist_low, sweep), "nyquist_low", sweep)
        if not vh > vl:
            raise ValueError("nyquist_high must be larger than nyquist_low")
        return vh, vl

    if isinstance(obj, xr.Dataset):
        vh, vl = limits(None)
        return _dual_prf_sweep(obj, field, vh, vl, prf, first_ray, *options)
    nodes = {}
    for sweep in _sweep_names_with(obj, field):
        vh, vl = limits(sweep)
        nodes[sweep] = _dual_prf_sweep(
            _sweep_dataset_of(obj, sweep), field, vh, vl, prf, first_ray, *options
        )
    if not nodes:
        raise ValueError(f"No sweep contains {field!r}.")
    return product_tree(obj, nodes)


accessor_for(correct_dual_prf, "dataset", "datatree")
accessor_for(dealias_confidence, "dataset", "datatree")
