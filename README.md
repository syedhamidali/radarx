# Radarx

![Radarx Logo](https://github.com/syedhamidali/radarx/raw/main/docs/_static/Radarx_Logo_micro.png)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14699306.svg)](https://doi.org/10.5281/zenodo.14699306)
[![Python Versions](https://img.shields.io/badge/Python-3.11%20|%203.12%20|%203.13%20|%203.14-blue)](https://www.python.org/downloads/)
[![PyPI Version](https://img.shields.io/pypi/v/radarx.svg)](https://pypi.org/project/radarx/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/radarx.svg?label=PyPI%20downloads)](https://pypi.org/project/radarx/)

[![Conda Version](https://img.shields.io/conda/vn/conda-forge/radarx.svg?logo=conda-forge&logoColor=white)](https://anaconda.org/conda-forge/radarx)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/radarx.svg?label=Conda%20downloads)](https://anaconda.org/conda-forge/radarx)

[![CI](https://github.com/syedhamidali/radarx/actions/workflows/ci.yml/badge.svg)](https://github.com/syedhamidali/radarx/actions/workflows/ci.yml)
[![Build distribution](https://github.com/syedhamidali/radarx/actions/workflows/upload_pypi.yml/badge.svg)](https://github.com/syedhamidali/radarx/actions/workflows/upload_pypi.yml)
[![Docs](https://readthedocs.org/projects/radarx/badge/?version=latest)](https://radarx.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/syedhamidali/radarx)](https://github.com/syedhamidali/radarx/blob/main/LICENSE)
![pre-commit enabled](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)

[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![CodeFactor](https://www.codefactor.io/repository/github/syedhamidali/radarx/badge)](https://www.codefactor.io/repository/github/syedhamidali/radarx)
[![codecov](https://codecov.io/gh/syedhamidali/radarx/graph/badge.svg?token=59WL4GNQOP)](https://codecov.io/gh/syedhamidali/radarx)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/092c74b48c0443aaa35cd292fa5aef54)](https://app.codacy.com/gh/syedhamidali/radarx/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)



Radarx is a Python library for weather radar data processing and visualization. It builds on [xradar](https://xradar.readthedocs.io/en/latest/), which reads radar data from many formats into [xarray](https://docs.xarray.dev/) [DataTree](https://docs.xarray.dev/en/stable/user-guide/hierarchical-data.html) structures, and adds gridding, retrievals and plotting through the `.radarx` accessor.

[![Project Status: Beta](https://img.shields.io/badge/status-beta-blue.svg)](https://www.repostatus.org/#beta)

> [!WARNING]
> **This project is currently in high development mode.**
> Features may change frequently, and some parts of the library may be incomplete or subject to change. Please proceed with caution.


## Key Features

- **Fast, accurate gridding**: Cone gridding interpolates within each sweep and then between sweeps, using the radar's own geometry. A compiled C++ kernel grids tens of millions of cells in a fraction of a second, with no smoothing radius to tune (`dtree.radarx.to_grid()`). Barnes interpolation is also available.
- **CAPPI retrieval**: Constant-altitude PPIs with several methods (`dtree.radarx.create_cappi()`).
- **Interactive plotting**: [hvplot](https://hvplot.holoviz.org/)-based range-azimuth, PPI, RHI, CAPPI and Max-CAPPI views on DataArrays, Datasets and DataTrees (`.radarx.plot`), plus matplotlib plots.
- **Unstructured grids**: Convert a sweep into a [uxarray](https://uxarray.readthedocs.io/) dataset with one cell per gate, for true gate footprints, area-weighted statistics and remapping (`.radarx.to_uxarray()`).
- **Radar fundamentals**: Functions for radar equations, beam geometry, Doppler, attenuation and more (`radarx.fundamentals`).
- **Cloud data access**: List and download NEXRAD and MRMS data from AWS (`radarx.io`).

## Installation

You can install `radarx` using conda from the `conda-forge` channel (recommended):

```bash
conda install -c conda-forge radarx
```

You can also install `radarx` via pip from PyPI:

```bash
python -m pip install radarx
```

Optional features need extra packages:

```bash
python -m pip install "radarx[plot]"     # interactive hvplot plots
python -m pip install "radarx[uxarray]"  # unstructured grids with uxarray
```

With conda, install them directly, e.g. `conda install -c conda-forge hvplot datashader` or `conda install -c conda-forge uxarray spatialpandas geopandas`.

Barnes gridding (`method="barnes"`) needs [fast-barnes-py](https://github.com/MeteoSwiss/fast-barnes-py), which supports Python < 3.13 only: `python -m pip install fast-barnes-py`. The default cone gridding needs no extra packages.

Alternatively, you can install it from source by cloning the repository
and running:

```bash
git clone https://github.com/syedhamidali/radarx.git
cd radarx
python -m pip install .
```

Building from source compiles the C++ gridding kernel if a C++ compiler is available; otherwise radarx falls back to an equivalent NumPy implementation.

## Usage

Read a radar volume with [xradar](https://xradar.readthedocs.io/en/latest/), then grid, retrieve and plot it with radarx:

```python
import xradar as xd
from open_radar_data import DATASETS

import radarx  # noqa: F401  registers the .radarx accessors

filename = DATASETS.fetch("KLBB20160601_150025_V06")
dtree = xd.io.open_nexradlevel2_datatree(filename)

# 3D grid with cone gridding (500 m spacing, 0.5 to 15 km height)
grid = dtree.radarx.to_grid(
    data_vars=["DBZH"],
    x_lim=(-150e3, 150e3), y_lim=(-150e3, 150e3), z_lim=(500, 15e3),
    x_step=500, y_step=500, z_step=500,
)

# interactive plots (requires radarx[plot])
dtree.radarx.plot.ppi("DBZH", sweeps=0)
grid.radarx.plot.max_cappi("DBZH")
```

See the [documentation](https://radarx.readthedocs.io/) for more examples, including CAPPIs, IMD data and unstructured grids.

> [!WARNING]
> The radarx IMD reader (`rx.io.read_sweep`, `rx.io.read_volume`, `rx.io.to_cfradial2`,
> `rx.io.to_cfradial2_volumes`) is **deprecated** and will be removed in a future release.
> IMD data is read natively by xradar (releases after 0.12.0): use
> `xr.open_dataset(file, engine="imd")` or `xd.io.open_imd_datatree(files)` instead.


## Documentation

For full documentation, see [radarx.readthedocs.io](https://radarx.readthedocs.io/).


## Contributing

Contributions are welcome! If you'd like to contribute, please follow
the steps below:

1.  Fork the repository.
2.  Create a new branch for your feature or bugfix.
3.  Write tests for your changes.
4.  Submit a pull request.

Please ensure that your code passes the pre-commit hooks and test suite
before submitting your PR.


## License

Radarx is licensed under the MIT License. See the
[LICENSE](https://github.com/syedhamidali/radarx/blob/main/LICENSE) file
for more details.


## Authors

-   Syed Hamid Ali

## Citation

If you use radarx, please cite this DOI. It covers all versions of radarx and always points to the latest release, so citations of every version are counted together. To record the exact version you used, add it to the citation (e.g. `version = {0.3.1}` in BibTeX).

> Syed, H. A. Radarx: An Xarray-based Python package for radar data processing. Zenodo. https://doi.org/10.5281/zenodo.14699306

```bibtex
@software{syed_radarx,
  author    = {Syed, Hamid Ali},
  title     = {Radarx: An Xarray-based Python package for radar data processing},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.14699306},
  url       = {https://doi.org/10.5281/zenodo.14699306},
}
```
