# Radarx

![Radarx Logo](https://github.com/syedhamidali/radarx/raw/main/docs/_static/Radarx_Logo_micro.png)

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
[![CodeQL](https://github.com/syedhamidali/radarx/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/syedhamidali/radarx/actions/workflows/github-code-scanning/codeql)
[![CodeFactor](https://www.codefactor.io/repository/github/syedhamidali/radarx/badge)](https://www.codefactor.io/repository/github/syedhamidali/radarx)
[![codebeat badge](https://codebeat.co/badges/9e6434e5-d40c-48d2-8f77-7e81241bd965)](https://codebeat.co/projects/github-com-syedhamidali-radarx-main)
[![codecov](https://codecov.io/gh/syedhamidali/radarx/graph/badge.svg?token=59WL4GNQOP)](https://codecov.io/gh/syedhamidali/radarx)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/092c74b48c0443aaa35cd292fa5aef54)](https://app.codacy.com/gh/syedhamidali/radarx/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)


<!-- [![Linux](https://img.shields.io/github/actions/workflow/status/syedhamidali/radarx/.github/workflows/tests.yaml?label=Linux)](https://github.com/syedhamidali/radarx/actions/workflows/tests.yaml)
[![macOS](https://img.shields.io/github/actions/workflow/status/syedhamidali/radarx/.github/workflows/tests.yaml?label=macOS)](https://github.com/syedhamidali/radarx/actions/workflows/tests.yaml)
[![Windows](https://img.shields.io/github/actions/workflow/status/syedhamidali/radarx/.github/workflows/tests_windows.yaml?label=Windows)](https://github.com/syedhamidali/radarx/actions/workflows/tests_windows.yaml) -->


Radarx is a Python library built for radar data processing and visualization. The library integrates tightly with [xradar](https://xradar.readthedocs.io/en/latest/) and leverages [xarray](http://xarray.pydata.org/) and [DataTree](https://xarray.pydata.org/en/stable/related-projects/datree.html) structures to enable easy and efficient manipulation of radar sweeps and volume data.

> [!WARNING]
> **This project is currently in high development mode.**
> Features may change frequently, and some parts of the library may be incomplete or subject to change. Please proceed with caution.


## Key Features

- **Xradar Integration**: Uses [xradar](https://xradar.readthedocs.io/en/latest/) for reading radar data in different formats, providing a consistent interface for various radar types.
- **IMD Radar Data Support**: Special support for reading and processing IMD radar data in NetCDF format.
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

Here's a simple example of how to use Radarx with [xradar]{.title-ref}
to load and process a volume scan:

```python
import radarx as rx

# List of radar files
files = [
    'radar_file1.nc',
    'radar_file2.nc',
    'radar_file3.nc'
]

# Read volume data using Radarx, with xradar integration
volume = rx.io.read_volume(files)

# Access a specific sweep or variable
dbz_data = volume['/volume_0']['DBZ']
```

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
