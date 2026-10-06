# Contributing

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at <https://github.com/syedhamidali/radarx/issues>.

If you are reporting a bug, please include:

-   Your operating system name and version.
-   Any details about your local setup that might be helpful in
    troubleshooting.
-   Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with \"bug\"
and \"help wanted\" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with
\"enhancement\" and \"help wanted\" is open to whoever wants to
implement it.

### Write Documentation

radarx could always use more documentation, whether as part of the
official radarx docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at
<https://github.com/syedhamidali/radarx/issues>.

If you are proposing a feature:

-   Explain in detail how it would work.
-   Keep the scope as narrow as possible, to make it easier to
    implement.
-   Remember that this is a volunteer-driven project, and that
    contributions are welcome :)

## Get Started!

Ready to contribute? Here\'s how to set up [radarx]{.title-ref} for
local development.

1.  Fork the [radarx]{.title-ref} repo on GitHub.

2.  Clone your fork locally:

    ```bash
    $ git clone git@github.com:your_name_here/radarx.git
    ```

3.  Create the development environment and install your local copy in
    editable mode (a C++ compiler builds the fast kernels; without one,
    radarx falls back to NumPy):

    ```bash
    $ cd radarx/
    $ mamba env create -f environment.yml
    $ mamba activate radarx-dev
    $ python -m pip install -e ".[dev]"
    $ pre-commit install
    ```

4.  Create a branch for local development:

    ```bash
    $ git checkout -b name-of-your-bugfix-or-feature
    ```

    Now you can make your changes locally.

5.  When you\'re done making changes, check that your changes pass
    ruff, black and the tests:

    ```bash
    $ make lint
    $ make test
    ```

6.  Commit your changes and push your branch to GitHub:

    ```bash
    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature
    ```

7.  Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1.  The pull request should include tests.
2.  If the pull request adds functionality, the docs should be updated.
    Put your new functionality into a function with a docstring, and add
    a changelog fragment `docs/changes/<PR number>.md` (see
    `docs/changes/README.md`).
3.  The pull request should work for all supported Python versions; the
    GitHub Actions checks on the pull request must pass.

## Adding a feature without touching shared files

New features plug in without editing shared files, so pull requests don't
conflict with each other:

- **Compiled kernels:** every `radarx/**/_*.cpp` file is built as the extension
  module of the same name (`setup.py` discovers them) and checked by
  `ci/check_compiled_kernel.py`. Keep an identical NumPy implementation as the
  fallback and test oracle.
- **Retrieval modules:** every public module in `radarx/retrieve/` is imported
  and its `__all__` re-exported by `radarx.retrieve` automatically.
- **Accessor methods:** register them in your module with
  `radarx._registry.accessor_method` instead of editing `radarx/accessors.py`.
- **Notebooks:** example notebooks named `docs/notebooks/<Capitalised_Name>.md`
  are listed in the user guide automatically.
- **Changelog:** add `docs/changes/<PR number>.md`.

## Tips

To run a subset of tests:

```bash
$ pytest tests/test_radarx.py
```

## Deploying

A reminder for the maintainers on how to release:

1. In a pull request titled `REL: X.Y.Z`, run
   `python ci/release_changelog.py X.Y.Z`. It moves the changelog fragments
   from `docs/changes/` into a new section of `docs/history.md` (add a short
   summary paragraph there if you like) and sets `version` and
   `date-released` in `CITATION.cff`. The README, the documentation and
   GitHub's "Cite this repository" button all take the citation from
   `CITATION.cff`.
2. After merging, publish a GitHub release with the tag `vX.Y.Z`. The version
   comes from the tag (setuptools-scm); the release workflow builds the wheels
   and uploads them to PyPI, and Zenodo archives the release under the same
   concept DOI.
3. Merge the conda-forge bot's version update on the radarx feedstock (check
   for new runtime dependencies).
