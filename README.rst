.. important::
    This project is currently in **high development mode**. Features may change frequently, and some parts of the library may be incomplete or subject to change. Please proceed with caution.

====================================================
Radarx: Xarray based toolkit for Radar Data Analysis
====================================================

.. image:: https://img.shields.io/pypi/v/radarx.svg
    :target: https://pypi.org/project/radarx/
    :alt: PyPI

.. image:: https://img.shields.io/github/license/syedhamidali/radarx
    :target: https://github.com/syedhamidali/radarx
    :alt: License

.. image:: https://img.shields.io/pypi/pyversions/radarx.svg
    :target: https://pypi.org/project/radarx/
    :alt: Python Version

Radarx is a Python library built for radar data processing, and visualization. The library integrates tightly with `xradar` and leverages `xarray` and `DataTree` structures to enable easy and efficient manipulation of radar sweeps and volume data.

=================
Key Features
=================

- **Xradar Integration**: Uses `xradar` for reading radar data in different formats, providing a consistent interface for various radar types.
- **IMD Radar Data Support**: Special support for reading and processing IMD radar data in NetCDF format.
- **Volume Scanning**: Utilities to process radar sweeps and group them into complete volume scans.
- **Data Gridding**: Provides tools for converting radar data to regular Cartesian grids, supporting complex radar geometries.
- **Xarray and DataTree Structured Data**: Radar data is returned as `xarray` datasets, organized into `DataTree` structures for easy navigation and analysis.

=================
Installation
=================

You can install Radarx via `pip` from PyPI:

.. code-block:: bash

    python -m pip install radarx

Alternatively, you can install it from source by cloning the repository and running:

.. code-block:: bash

    git clone https://github.com/syedhamidali/radarx.git
    cd radarx
    python -m pip install .

=================
Usage
=================

Here’s a simple example of how to use Radarx with `xradar` to load and process a volume scan:

.. code-block:: python

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

Radarx leverages `xradar` to handle radar file formats and integrates smoothly with `xarray` and `DataTree` for organizing and analyzing radar data.

=================
Xradar Integration
=================

Radarx makes use of the powerful `xradar` library for radar data ingestion and format handling. This ensures that the package is flexible and can handle a variety of radar data formats, including ODIM_H5, Sigmet, and others. For more advanced users, `xradar` functionality can be directly accessed to extend Radarx's capabilities.

=================
Documentation
=================

For full documentation, please visit the `Radarx Documentation <https://github.com/syedhamidali/radarx>`_.

=================
Contributing
=================

Contributions are welcome! If you'd like to contribute, please follow the steps below:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Write tests for your changes.
4. Submit a pull request.

Please ensure that your code passes the pre-commit hooks and test suite before submitting your PR.

=================
License
=================

Radarx is licensed under the MIT License. See the `LICENSE <https://github.com/syedhamidali/radarx/blob/main/LICENSE>`_ file for more details.

=================
Authors
=================

- Syed Hamid Ali
