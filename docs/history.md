# History

## Unreleased
- **FIX:** Drop the ``scipy<=1.14.1`` pin. SciPy 1.14.1 does not support NumPy >= 2.3, so environments with a current NumPy got an incompatible SciPy and a warning on import. The test suite passes with SciPy 1.18 and NumPy 2.5. by [@syedhamidali](https://github.com/syedhamidali)
- **ENH:** ``import radarx`` is about 25 % faster: matplotlib, cmweather, cartopy and boto3 are now imported only when a matplotlib plot or AWS helper is used. by [@syedhamidali](https://github.com/syedhamidali)
- **ADD:** ``to_uxarray`` (``ds.radarx.to_uxarray()``) converts a PPI sweep into a uxarray dataset in the UGRID conventions, with one quadrilateral face per gate whose corners lie halfway to the neighbouring rays and gates. This gives true gate footprints and areas, area-weighted statistics and uxarray's subsetting and remapping on the native polar geometry (openradar/xradar#212, UXARRAY/uxarray#976). Install with ``pip install radarx[uxarray]``. ({pull}`77`) by [@syedhamidali](https://github.com/syedhamidali)
- **DOC:** New notebook: radar sweeps as unstructured grids with uxarray. ({pull}`77`) by [@syedhamidali](https://github.com/syedhamidali)
- **DOC:** Fix the fundamentals exercises: Example 3.9 and the multipath example passed one antenna gain to ``radar_equation``/``solve_peak_power``, which take separate transmit and receive gains; the multipath example also used a doubled phase difference, the wrong reflection sign and ``F**2`` instead of ``F**4``. ({pull}`76`) by [@syedhamidali](https://github.com/syedhamidali)

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
