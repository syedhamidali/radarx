# History

## Unreleased
- **FIX:** Require ``xradar>=0.11``. Without a lower bound, the conda-forge test environment for 0.4.0 resolved to xradar 0.0.5, which needs the old xarray-datatree package and fails with current xarray. ({pull}`86`) by [@syedhamidali](https://github.com/syedhamidali)
- **MNT:** Drop Python 3.10: radarx now requires Python >= 3.11 (conda-forge no longer builds compiled packages for 3.10). Wheels are built for Python 3.11-3.14. ({pull}`87`) by [@syedhamidali](https://github.com/syedhamidali)
- **MNT:** Remove the empty ``ruff.toml`` so ruff uses the ``pyproject.toml`` config again, fix its findings and unpin ruff in CI. ({pull}`106`) by [@syedhamidali](https://github.com/syedhamidali)
- **MNT:** Remove stale configuration (`tox.ini`, `rst2md.sh`, `requirements_dev.txt`), keep the development dependencies in the `dev` extra only, update the Makefile and contributing guide, and ignore the files written by the docs build and setuptools-scm. ({pull}`105`) by [@syedhamidali](https://github.com/syedhamidali)
- **MNT:** Test on Python 3.11-3.14 (NumPy 1 on 3.11 and 3.12 only). fast-barnes-py (Python < 3.13 only) is now optional and imported only for ``grid_radar(method="barnes")``; the unit-test environment keeps only radarx's own dependencies, separate from the notebook and docs environments, and the environments are no longer capped at Python < 3.13. ({pull}`108`) by [@syedhamidali](https://github.com/syedhamidali)
- **ADD:** Azimuthal shear and radial divergence by the linear least-squares derivative (LLSD) method: ``radarx.retrieve.llsd`` / ``azimuthal_shear`` / ``radial_divergence`` and ``ds.radarx.azimuthal_shear()`` (also on DataTree volumes). A plane is fitted to the Doppler velocity in a window of fixed physical size around every gate, using the measured ray azimuths and wrapping around north. A multithreaded C++ kernel with cumulative sums along range processes a NEXRAD volume in 0.09 s, with an identical NumPy fallback. ({pull}`107`) by [@syedhamidali](https://github.com/syedhamidali)
- **ADD:** Fast Doppler velocity dealiasing (``radarx.retrieve.dealias_velocity``, ``ds.radarx.dealias()``, ``dtree.radarx.dealias()``). Region-based unfolding on the polar sweep grid: union-find regions, integer least-squares folds over the region graph (region and block coordinate descent), absolute fold from the sweep below, a wind profile or an in-sweep VAD, and a final gate check. A multithreaded C++ kernel dealiases all sweeps of a volume in one call (NumPy fallback with identical results). On a strongly aliased NEXRAD volume (KGWX, 8.2 million gates) it matches Py-ART's region-based dealiasing on 99.9 % of gates in a fraction of a second, about 100 times faster. ({pull}`112`) by [@syedhamidali](https://github.com/syedhamidali)
- **ADD:** Advection correction of gridded radar volumes (``radarx.retrieve.estimate_motion``, ``advect``, ``interpolate_time``, also as ``ds.radarx.*``). Storm motion from FFT cross-correlation of two volumes (jointly observed area, reflectivity floor, Gaussian high-pass, Hann taper, normalised peak quality, parabolic sub-cell peak and iterative refinement that removes the window bias), optionally per tile as a smooth, spatially varying field. Fields are moved with a semi-Lagrangian scheme (bilinear or clipped cubic convolution) and an advected validity mask, in a multithreaded C++ kernel with an identical NumPy fallback; ``interpolate_time`` blends forward- and backward-advected volumes. On three consecutive NEXRAD volumes it cuts the 6.5-min forecast error from 6.3 to 3.7 dB and the time-interpolation error from 5.4 to 3.0 dB. (Gal-Chen 1982; Shapiro et al. 2010) ({pull}`PR`) by [@syedhamidali](https://github.com/syedhamidali)
- **ADD:** Advection correction of gridded radar volumes (``radarx.retrieve.estimate_motion``, ``advect``, ``interpolate_time``, also as ``ds.radarx.*``). Storm motion from FFT cross-correlation of two volumes (jointly observed area, reflectivity floor, Gaussian high-pass, Hann taper, normalised peak quality, parabolic sub-cell peak and iterative refinement that removes the window bias), optionally per tile as a smooth, spatially varying field. Fields are moved with a semi-Lagrangian scheme (bilinear or clipped cubic convolution) and an advected validity mask, in a multithreaded C++ kernel with an identical NumPy fallback; ``interpolate_time`` blends forward- and backward-advected volumes. On three consecutive NEXRAD volumes it cuts the 6.5-min forecast error from 6.3 to 3.7 dB and the time-interpolation error from 5.4 to 3.0 dB. (Gal-Chen 1982; Shapiro et al. 2010) ({pull}`111`) by [@syedhamidali](https://github.com/syedhamidali)

## 0.4.0 (2026-10-05)

This release makes cone gridding the default: a new gridder built for radar
geometry, with a compiled C++ kernel, that is much faster and more accurate
than Barnes interpolation. It also adds a uxarray (UGRID) conversion, speeds
up ``import radarx`` and fixes the SciPy/NumPy compatibility pin.

radarx now ships compiled wheels for Linux, macOS and Windows. Building from
source needs a C++ compiler for the fast kernel; without one, radarx falls
back to an equivalent NumPy implementation.

- **ADD:** Cone gridding (``radarx.grid.grid_cones``, now the default ``method="cone"`` of ``grid_radar`` / ``dtree.radarx.to_grid``). Each sweep is interpolated bilinearly at every output column from its four surrounding gates, using the measured ray azimuths and elevations, and each level is interpolated in height between the two sweeps that bracket it. No radius of influence or smoothing to tune; values never overshoot the data; cells not bracketed by two sweeps stay empty (``pseudo_cappi`` fills below the lowest sweep). A compiled C++ kernel (pybind11, multithreaded) does the work, with an identical NumPy fallback. On a NEXRAD volume it grids 43 million cells in 0.13 s (fast-barnes 5.5 s, Py-ART 63 s) and reconstructs a held-out tilt with about half the error of Barnes. ``method="barnes"`` remains available. ({pull}`79`) by [@syedhamidali](https://github.com/syedhamidali)
- **FIX:** Drop the ``scipy<=1.14.1`` pin. SciPy 1.14.1 does not support NumPy >= 2.3, so environments with a current NumPy got an incompatible SciPy and a warning on import. The test suite passes with SciPy 1.18 and NumPy 2.5. ({pull}`78`) by [@syedhamidali](https://github.com/syedhamidali)
- **ENH:** ``import radarx`` is about 25 % faster: matplotlib, cmweather, cartopy and boto3 are now imported only when a matplotlib plot or AWS helper is used. As a result, ``import radarx`` no longer registers the cmweather colormaps (e.g. ``ChaseSpectral``) for other libraries; radarx's own plots still register them, and ``import cmweather`` makes them available elsewhere. ({pull}`78`) by [@syedhamidali](https://github.com/syedhamidali)
- **ADD:** ``to_uxarray`` (``ds.radarx.to_uxarray()``) converts a PPI sweep into a uxarray dataset in the UGRID conventions, with one quadrilateral face per gate whose corners lie halfway to the neighbouring rays and gates. This gives true gate footprints and areas, area-weighted statistics and uxarray's subsetting and remapping on the native polar geometry (openradar/xradar#212, UXARRAY/uxarray#976). Install with ``pip install radarx[uxarray]``. ({pull}`77`) by [@syedhamidali](https://github.com/syedhamidali)
- **DOC:** New notebook: radar sweeps as unstructured grids with uxarray. ({pull}`77`) by [@syedhamidali](https://github.com/syedhamidali)
- **DOC:** Fix the fundamentals exercises: Example 3.9 and the multipath example passed one antenna gain to ``radar_equation``/``solve_peak_power``, which take separate transmit and receive gains; the multipath example also used a doubled phase difference, the wrong reflection sign and ``F**2`` instead of ``F**4``. ({pull}`76`) by [@syedhamidali](https://github.com/syedhamidali)
- **MNT:** Platform wheels are built with cibuildwheel; CI checks that the compiled kernel is present; workflows use read-only GitHub token permissions. ({pull}`79`) by [@syedhamidali](https://github.com/syedhamidali)
- **MNT:** ``docs/history.md`` uses ``merge=union`` so changelog entries from different branches no longer conflict. ({pull}`80`) by [@syedhamidali](https://github.com/syedhamidali)
- **DOC:** Updated README (features, optional extras, a runnable example, links, Python 3.10-3.13) and citation metadata: the all-versions Zenodo DOI ``10.5281/zenodo.14699306`` is used everywhere, and ``CITATION.cff`` follows CFF 1.2.0. Removed the CodeQL and codebeat badges. ({pull}`81`, {pull}`82`, {pull}`83`) by [@syedhamidali](https://github.com/syedhamidali)

## 0.3.1 (2026-10-04)
- **DOC:** Tagged documentation builds show the release version (the v0.3.0 docs showed ``0.3.1.dev0``). ({pull}`75`) by [@syedhamidali](https://github.com/syedhamidali)
- **FIX:** Declare ``boto3`` as a dependency. Without it, ``radarx.io`` silently failed to import on pip installs (``radarx.io`` was ``None``). ({pull}`75`) by [@syedhamidali](https://github.com/syedhamidali)

## 0.3.0 (2026-10-04)

This release adds CAPPI retrieval, interactive hvplot-based radar plots and
deprecates the radarx IMD reader in favour of xradar's IMD backend. The IMD
reader will be removed in the release following the first xradar version
that ships the IMD backend.

### Deprecations
- **DEP:** The radarx IMD reader (``read_sweep``, ``read_volume``, ``to_cfradial2``, ``to_cfradial2_volumes``) and ``radarx.testing.fetch_imd_test_data`` are deprecated and emit a ``FutureWarning`` naming the replacement. IMD data is read natively by xradar (releases after 0.12.0) via ``xr.open_dataset(file, engine="imd")``, ``xradar.io.open_imd_datatree`` and ``xradar.io.open_imd_volumes``. ({pull}`74`) by [@syedhamidali](https://github.com/syedhamidali)

### New features
- **ADD:** CAPPI retrieval with ``cartesian_idw``, ``polar_vertical_interpolation`` and ``height_window_composite`` methods (``dt.radarx.create_cappi``), plus matplotlib ``plot_ppi``, ``plot_rhi`` and ``plot_cappi`` helpers. ({pull}`73`) by [@syedhamidali](https://github.com/syedhamidali)
- **ADD:** Interactive hvplot/HoloViews plots via the ``.radarx.plot`` accessor on DataArray, Dataset and DataTree: range-azimuth, PPI, gate mesh, gate centroids, RHI, CAPPI and Max-CAPPI views, faceting over variables and sweeps, ``backend="bokeh"|"matplotlib"`` and fallthrough to ``.hvplot``, following the accessor roadmap in openradar/xradar#174. Large polar sweeps are rasterized with datashader by default. Install the optional dependencies with ``pip install radarx[plot]``. ({pull}`74`) by [@syedhamidali](https://github.com/syedhamidali)

### Fixes
- **FIX:** NEXRAD Level II data now comes from the ``unidata-nexrad-level2`` bucket; ``noaa-nexrad-level2`` no longer allows anonymous access (``radarx.io.aws_data``). ({pull}`74`) by [@syedhamidali](https://github.com/syedhamidali)
- **FIX:** ``combine_nexrad_sweeps`` works with recent xradar, which returns ``range`` without an index, and leaves gates beyond the short-range sweep missing instead of repeating its last gate. ({pull}`74`) by [@syedhamidali](https://github.com/syedhamidali)
- **FIX:** Gridding (``to_grid``, ``make_3d_grid``) keeps the radar site coordinates with newer xarray, which no longer inherits non-index root coordinates by default. ({pull}`74`) by [@syedhamidali](https://github.com/syedhamidali)
- **FIX:** ``to_cfradial2`` works with xradar 0.12, which renamed ``site_coords`` to ``site_as_coords``. ({pull}`74`) by [@syedhamidali](https://github.com/syedhamidali)
- **FIX:** Improved CAPPI/xradar interoperability by preserving sweep-style metadata and broader DataTree compatibility across supported xarray setups. ({pull}`73`) by [@syedhamidali](https://github.com/syedhamidali)
- **FIX:** cartopy is now an optional, lazy import for ``plot_maxcappi``. ({pull}`73`) by [@syedhamidali](https://github.com/syedhamidali)

### Documentation and maintenance
- **DOC:** Notebooks moved to MyST markdown in ``docs/notebooks`` (following xradar) and are executed during the docs build again, so the documentation shows figures; all notebooks are also run in CI. ({pull}`74`) by [@syedhamidali](https://github.com/syedhamidali)
- **DOC:** IMD notebook rewritten to read data with xradar; new Interactive Plots notebook. ({pull}`74`) by [@syedhamidali](https://github.com/syedhamidali)
- **MNT:** Simplified the CAPPI API around ``height``, ``method``, ``vertical_tolerance``, optional filtering and essential Cartesian grid controls. ({pull}`73`) by [@syedhamidali](https://github.com/syedhamidali)
- **MNT:** CI pins black and ruff, adds a Codacy configuration and builds the docs with myst-nb. ({pull}`73`, {pull}`74`) by [@syedhamidali](https://github.com/syedhamidali)

## 0.2.5 (2025-04-22)
- **ADD:** Added `fundamentals` module with core radar computation utilities. ({pull}`66`) by [@syedhamidali](https://github.com/syedhamidali)
- **CHG:** Refactored Doppler and scattering fundamentals to reduce complexity.
- **ADD:** Tests for PRF, beamwidth, radar range, Doppler shift, and SNR calculations. ({pull}`66`) by [@syedhamidali](https://github.com/syedhamidali)

## 0.2.4 (2025-01-20)
- **ADD:** Citation metadata for radarx. ({pull}`61`) by [@syedhamidali](https://github.com/syedhamidali)

## 0.2.3 (2025-01-19)
- **ADD:** Binder support setup. ({pull}`58`, {pull}`59`) by [@syedhamidali](https://github.com/syedhamidali)
- **MNT:** Update `max_cappi` to return figure object. ({pull}`60`) by [@syedhamidali](https://github.com/syedhamidali)

## 0.2.2 (2024-12-19)
- **MNT:** Enhancements for Binder support. ({pull}`58`, {pull}`59`) by [@syedhamidali](https://github.com/syedhamidali)

## 0.2.1 (2024-12-06)
- **REL** Version Release 0.2.1, ({pull}`56`) by [@syedhamidali](https://github.com/syedhamidali)
- **ADD:** Added wradlib gridding example. ({issue}`51`, {issue}`52`, {issue}`53`) and ({pull}`56`) by [@syedhamidali](https://github.com/syedhamidali)
- **ADD:** Added fast-barnes-py gridding for radar. ({issue}`51`, {issue}`52`, {issue}`53`) and ({pull}`55`) by [@syedhamidali](https://github.com/syedhamidali)

## 0.2.0 (2024-11-18)
This is the first version which uses DataTree from xarray. Thus, xarray is pinned to version >=2024.10.0.
- **REL:** Version Release 0.2.0. ({issue}`45`), ({pull}`46`) by [@syedhamidali](https://github.com/syedhamidali)

## 0.1.9 (2024-11-18)
This is the last version which uses datatree from xarray-contrib/datatree. Thus, xarray is pinned to version 2024.9.0.

## 0.1.8 (2024-09-18)
- **ADD:** Codecov in the workflow. ({pull}`25`) by [@syedhamidali](https://github.com/syedhamidali)
- **MNT:** Updated CI and fixed codecov. ({pull}`26`) by [@syedhamidali](https://github.com/syedhamidali)
- **ADD:** Added CircleCI. ({pull}`27`) by [@syedhamidali](https://github.com/syedhamidali)
- **FIX:** IMD fix. ({pull}`28`) by [@syedhamidali](https://github.com/syedhamidali)
- **FIX:** Notebook fix. ({pull}`29`) by [@syedhamidali](https://github.com/syedhamidali)
- **FIX:** Fixed IMD. ({pull}`31`) by [@syedhamidali](https://github.com/syedhamidali)
- **ENH:** Improved docstrings in `imd.py`. ({pull}`32`) by [@syedhamidali](https://github.com/syedhamidali)
- **MNT:** Changed Dependabot runs to weekly. ({pull}`33`) by [@syedhamidali](https://github.com/syedhamidali)

## 0.1.7 (2024-09-14)
- **ADD:** Function to convert data to CFRadial2. ({pull}`22`) by [@syedhamidali](https://github.com/syedhamidali)

## 0.1.6 (2024-09-10)
- **ENH:** Updated README. ({pull}`14`) by [@syedhamidali](https://github.com/syedhamidali)
- **FIX:** `read_sweep` fix. ({pull}`17`) by [@syedhamidali](https://github.com/syedhamidali)
- **FIX:** Fixed documentation issues. ({pull}`15`) by [@syedhamidali](https://github.com/syedhamidali)
- **ADD:** Created `dependabot.yml`. ({pull}`18`) by [@syedhamidali](https://github.com/syedhamidali)
- **ENH:** Added Dependabot. ({pull}`19`) by [@syedhamidali](https://github.com/syedhamidali)
- **MNT:** Updated README. ({pull}`20`) by [@syedhamidali](https://github.com/syedhamidali)

## 0.1.5 (2024-09-09)
- **FIX:** Versioning issue resolved. ({pull}`13`) by [@syedhamidali](https://github.com/syedhamidali)

## 0.1.4 (2024-09-08)
- **ENH:** Added environment `.yml` file. ({pull}`2`) by [@syedhamidali](https://github.com/syedhamidali)
- **ADD:** Links.rst added in docs. ({pull}`3`) by [@syedhamidali](https://github.com/syedhamidali)
- **FIX:** Fixed HTML title issue. ({pull}`4`) by [@syedhamidali](https://github.com/syedhamidali)
- **ENH:** Updated `README.md`. ({pull}`5`) by [@syedhamidali](https://github.com/syedhamidali)
- **FIX:** Addressed versioneer issues. ({pull}`6`, {pull}`7`) by [@syedhamidali](https://github.com/syedhamidali)
- **FIX:** Updated radarx logo. ({pull}`8`) by [@syedhamidali](https://github.com/syedhamidali)
- **MNT:** Converted RST to MD in documentation. ({pull}`10`) by [@syedhamidali](https://github.com/syedhamidali)
- **FIX:** history.md updates. ({pull}`11`) by [@syedhamidali](https://github.com/syedhamidali)
- **ADD:** Added logo to `README.md`. ({pull}`12`) by [@syedhamidali](https://github.com/syedhamidali)

## 0.1.3 (2024-09-08)
- **ENH:** Added environment `.yml` file. ({pull}`2`) by [@syedhamidali](https://github.com/syedhamidali)
- **ADD:** Added `Links.rst` in docs. ({pull}`3`) by [@syedhamidali](https://github.com/syedhamidali)
- **FIX:** Fixed HTML title issue. ({pull}`4`) by [@syedhamidali](https://github.com/syedhamidali)
- **ENH:** Updated `README.md`. ({pull}`5`) by [@syedhamidali](https://github.com/syedhamidali)
- **FIX:** Addressed versioneer issues. ({pull}`6`, {pull}`7`) by [@syedhamidali](https://github.com/syedhamidali)
- **FIX:** Updated radarx logo. ({pull}`8`) by [@syedhamidali](https://github.com/syedhamidali)
- **MNT:** Converted RST to MD in documentation. ({pull}`10`) by [@syedhamidali](https://github.com/syedhamidali)
- **FIX:** history.md updates. ({pull}`11`) by [@syedhamidali](https://github.com/syedhamidali)
- **ADD:** Added logo to `README.md`. ({pull}`12`) by [@syedhamidali](https://github.com/syedhamidali)

## 0.1.2 (2024-09-08)
- **FIX:** Created issue template. ({pull}`1`) by [@syedhamidali](https://github.com/syedhamidali)

## 0.1.1 (2024-09-07)
- Initial enhancements and setup completed.
- Full changelog: v0.1.0...v0.1.1

## 0.1.0 (2024-09-05)
- First commit and setup with pre-commit hooks.
