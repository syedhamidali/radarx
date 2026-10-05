# Radarx

![Radarx Logo](https://github.com/syedhamidali/radarx/raw/main/docs/_static/Radarx_Logo_micro.png)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14699312.svg)](https://doi.org/10.5281/zenodo.14699312)
[![Python Versions](https://img.shields.io/badge/Python-3.9%20|%203.10%20|%203.11%20|%203.12-blue)](https://www.python.org/downloads/)
[![PyPI Version](https://img.shields.io/pypi/v/radarx.svg)](https://pypi.org/project/radarx/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/radarx.svg?label=PyPI%20downloads)](https://pypi.org/project/radarx/)

[![Conda Version](https://img.shields.io/conda/vn/conda-forge/radarx.svg?logo=conda-forge&logoColor=white)](https://anaconda.org/conda-forge/radarx)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/radarx.svg?label=Conda%20downloads)](https://anaconda.org/conda-forge/radarx)

[![CI](https://github.com/syedhamidali/radarx/actions/workflows/ci.yml/badge.svg)](https://github.com/syedhamidali/radarx/actions/workflows/ci.yml)
[![Build distribution](https://github.com/syedhamidali/radarx/actions/workflows/upload_pypi.yml/badge.svg)](https://github.com/syedhamidali/radarx/actions/workflows/upload_pypi.yml)
[![RTD Version](https://readthedocs.org/projects/radarx/badge/?version=latest)](https://radarx.readthedocs.io/en/latest/?version=latest)
[![License](https://img.shields.io/github/license/syedhamidali/radarx)](https://github.com/syedhamidali/radarx/blob/main/LICENSE)
![pre-commit enabled](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)

<!-- [![Docs](https://readthedocs.org/projects/radarx/badge/?version=latest)](https://radarx.readthedocs.io/en/latest/) -->
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![CodeFactor](https://www.codefactor.io/repository/github/syedhamidali/radarx/badge)](https://www.codefactor.io/repository/github/syedhamidali/radarx)
[![codecov](https://codecov.io/gh/syedhamidali/radarx/graph/badge.svg?token=59WL4GNQOP)](https://codecov.io/gh/syedhamidali/radarx)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/092c74b48c0443aaa35cd292fa5aef54)](https://app.codacy.com/gh/syedhamidali/radarx/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)


<!-- [![Linux](https://img.shields.io/github/actions/workflow/status/syedhamidali/radarx/.github/workflows/tests.yaml?label=Linux)](https://github.com/syedhamidali/radarx/actions/workflows/tests.yaml)
[![macOS](https://img.shields.io/github/actions/workflow/status/syedhamidali/radarx/.github/workflows/tests.yaml?label=macOS)](https://github.com/syedhamidali/radarx/actions/workflows/tests.yaml)
[![Windows](https://img.shields.io/github/actions/workflow/status/syedhamidali/radarx/.github/workflows/tests_windows.yaml?label=Windows)](https://github.com/syedhamidali/radarx/actions/workflows/tests_windows.yaml) -->


Radarx is a Python library built for radar data processing and visualization. The library integrates tightly with [xradar](https://xradar.readthedocs.io/en/latest/) and leverages [xarray](http://xarray.pydata.org/) and [DataTree](https://xarray.pydata.org/en/stable/related-projects/datree.html) structures to enable easy and efficient manipulation of radar sweeps and volume data.

[![Project Status: Beta](https://img.shields.io/badge/status-beta-blue.svg)](https://www.repostatus.org/#beta)

> [!WARNING]
> **This project is currently in high development mode.**
> Features may change frequently, and some parts of the library may be incomplete or subject to change. Please proceed with caution.


## Key Features

- **Xradar Integration**: Uses [xradar](https://xradar.readthedocs.io/en/latest/) for reading radar data in different formats, providing a consistent interface for various radar types.
- **Interactive Plotting**: Optional [hvplot](https://hvplot.holoviz.org/)-based PPI, RHI, CAPPI and Max-CAPPI views via `ds.radarx.plot`.
- **Volume Scanning**: Utilities to process radar sweeps and group them into complete volume scans.
- **Data Gridding**: Provides tools for converting radar data to regular Cartesian grids, supporting complex radar geometries.
- **Xarray and DataTree Structured Data**: Radar data is returned as [xarray](http://xarray.pydata.org/) datasets, organized into [DataTree](https://xarray.pydata.org/en/stable/related-projects/datree.html) structures for easy navigation and analysis.


## Installation

You can install `radarx` using conda from the `conda-forge` channel (recommended):

```bash
conda install -c conda-forge radarx
```

You can also install `radarx` via pip from PyPI:

```bash
python -m pip install radarx
```

Alternatively, you can install it from source by cloning the repository
and running:

```bash
git clone https://github.com/syedhamidali/radarx.git
cd radarx
python -m pip install .
```

## Usage

Here's a simple example of how to use Radarx with [xradar](https://xradar.readthedocs.io/en/latest/)
to load radar data, grid it and plot it interactively:

```python
import xradar as xd
import radarx  # noqa: registers the ``.radarx`` accessors

# IMD radar data is read by xradar (releases after 0.12.0)
dtree = xd.io.open_imd_datatree(["radar_file.nc", "radar_file.nc.1", "radar_file.nc.2"])

# Grid the volume and plot an interactive Max-CAPPI (requires hvplot)
grid = dtree.radarx.to_grid(data_vars=["DBZH"])
grid.radarx.plot.max_cappi("DBZH")
```

> [!WARNING]
> The radarx IMD reader (`rx.io.read_sweep`, `rx.io.read_volume`, `rx.io.to_cfradial2`,
> `rx.io.to_cfradial2_volumes`) is **deprecated** and will be removed in the next release.
> IMD data is read natively by xradar (releases after 0.12.0): use
> `xr.open_dataset(file, engine="imd")` or `xd.io.open_imd_datatree(files)` instead.

Radarx leverages [xradar](https://xradar.readthedocs.io/en/latest/) to handle radar file formats and
integrates smoothly with [xarray](http://xarray.pydata.org/) and [DataTree](https://xarray.pydata.org/en/stable/related-projects/datree.html) for organizing and analyzing radar data.


## Xradar Integration

Radarx makes use of the powerful [xradar](https://xradar.readthedocs.io/en/latest/) library for radar data ingestion and format handling. This ensures that the package is flexible and can handle a variety of radar data formats, including ODIM_H5, Sigmet, and others. For more advanced users, [xradar](https://xradar.readthedocs.io/en/latest/) functionality can be directly accessed to extend Radarx\'s capabilities.


## Documentation

For full documentation, please visit the [Radarx
Documentation](https://github.com/syedhamidali/radarx).


## Contributing

Contributions are welcome! If you\'d like to contribute, please follow
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

>Syed, H. A. (2025). Radarx: An Xarray-based Python package for radar data processing (v0.2). Zenodo. https://doi.org/10.5281/zenodo.14699312

```python
@software{syed_2025_14699312,
  author       = {Syed, Hamid Ali},
  title        = {Radarx: An Xarray-based Python package for radar
                   data processing
                  },
  month        = jan,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v0.2},
  doi          = {10.5281/zenodo.14699312},
  url          = {https://doi.org/10.5281/zenodo.14699312},
  swhid        = {swh:1:dir:eb4e11846680cf6416be5940f36b363f74e1a3ec
                   ;origin=https://doi.org/10.5281/zenodo.14699306;vi
                   sit=swh:1:snp:f6755852f0e71678ed579651ec997ac4496f
                   3b30;anchor=swh:1:rel:ebd79cd3cf49a7e8a5e9b9576fb7
                   7f7c8bd0227e;path=syedhamidali-radarx-ec92870
                  },
}
```
