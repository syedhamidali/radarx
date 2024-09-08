# Installation

## Stable release (PyPI)

To install `radarx` via PyPI, run this command in your terminal:

```bash
$ python -m pip install radarx
```

This is the preferred method to install `radarx`, as it will always install the most recent stable release.

If you don’t have [pip](https://pip.pypa.io) installed, this [Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/) can guide you through the process.

## Conda (via conda-forge)

You can also install `radarx` using conda from the `conda-forge` channel:

```bash
$ conda install -c conda-forge radarx
```

This will install `radarx` and any dependencies via the `conda` package manager. For more information on installing conda, you can visit [Conda's installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

## From sources

The sources for `radarx` can be downloaded from the [GitHub repository](https://github.com/syedhamidali/radarx).

You can either clone the public repository:

```bash
$ git clone git://github.com/syedhamidali/radarx
```

Or download the [tarball](https://github.com/syedhamidali/radarx/tarball/master):

```bash
$ curl -OJL https://github.com/syedhamidali/radarx/tarball/master
```

Once you have a copy of the source, you can install it with:

```bash
$ python setup.py install
```

## Development version

If you want to install the development version, you can clone the GitHub repository and install the development dependencies:

```bash
$ git clone git://github.com/syedhamidali/radarx
$ cd radarx
$ pip install -e .[dev]
```

This will install `radarx` in development mode, along with all dependencies required for contributing to the project, such as testing and linting tools.
```
