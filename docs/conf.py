#!/usr/bin/env python
#
# radarx documentation build configuration file, created by
# sphinx-quickstart.
#
import datetime as dt
import glob
import os
import sys
import types
import warnings

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as get_version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError
    from importlib_metadata import version as get_version

sys.path.insert(0, os.path.abspath(".."))


def _write_unreleased_changes():
    """Collect the changelog fragments ``changes/<PR>.md`` for history.md."""
    here = os.path.dirname(os.path.abspath(__file__))
    fragments = glob.glob(os.path.join(here, "changes", "[0-9]*.md"))
    fragments.sort(key=lambda path: int(os.path.basename(path).split(".")[0]))
    lines = []
    for path in fragments:
        with open(path) as f:
            lines += [line.rstrip() for line in f if line.strip()]
    with open(os.path.join(here, "changes", "unreleased.md"), "w") as f:
        f.write("\n".join(lines) + "\n" if lines else "No changes yet.\n")


_write_unreleased_changes()

# The notebooks read ERA5 from Google's ARCO-ERA5 store, which keeps every
# field as one global chunk per hour, so a cold read takes minutes. The docs
# build seeds radarx's cache with a small pre-extracted subset (KGWX region,
# 30-31 March 2022, 23 and 00 UTC) so the build fits the Read the Docs time
# limit; outside the docs build the notebooks read the store directly.
if "RADARX_CACHE_DIR" not in os.environ:
    import shutil
    import tempfile

    _cache = tempfile.mkdtemp(prefix="radarx-docs-cache-")
    shutil.copytree(
        os.path.join(os.path.dirname(__file__), "notebooks", "data", "era5"),
        os.path.join(_cache, "soundings", "era5"),
    )
    os.environ["RADARX_CACHE_DIR"] = _cache

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx_copybutton",
    "sphinx_favicon",
    "myst_nb",
]

# Enable additional MyST extensions
myst_enable_extensions = [
    "substitution",
    "colon_fence",  # For :: used in directives
    "dollarmath",  # For LaTeX math
    # "linkify",      # For automatic links
]

extlinks = {
    "issue": ("https://github.com/syedhamidali/radarx/issues/%s", "GH %s"),
    "pull": ("https://github.com/syedhamidali/radarx/pull/%s", "PR %s"),
}

mathjax_path = (
    "https://cdn.mathjax.org/mathjax/latest/MathJax.js?" "config=TeX-AMS-MML_HTMLorMML"
)

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "datatree": ("https://xarray-datatree.readthedocs.io/en/latest/", None),
    "pyproj": ("https://pyproj4.github.io/pyproj/stable/", None),
}

templates_path = ["_templates"]

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
    ".md": "myst-nb",
}

master_doc = "index"

project = "radarx"
copyright = "2022-2024, Hamid Ali Syed"
author = "Hamid Ali Syed"
html_title = project

# Get radarx version and modules
import radarx  # noqa

modules = []
for k, v in radarx.__dict__.items():
    if isinstance(v, types.ModuleType):
        if k not in ["_warnings", "version"]:
            modules.append(k)
            file = open(f"{k}.rst", mode="w")
            file.write(f".. automodule:: radarx.{k}\n")
            file.close()

# Create Library reference rst-file
reference = """
Library Reference
=================

.. toctree::
   :maxdepth: 4
"""

file = open("reference.rst", mode="w")
file.write(f"{reference}\n")
for mod in sorted(modules):
    file.write(f"   {mod}\n")
file.close()

rst_files = glob.glob("*.rst")
autosummary_generate = rst_files
autoclass_content = "both"

try:
    version = get_version("radarx")
except PackageNotFoundError:
    version = getattr(radarx, "__version__", "999")
# On Read the Docs, setuptools-scm can see the checkout as modified and
# report the next dev version (e.g. 0.3.1.dev0 for the v0.3.0 tag), so tag
# builds use the tag name itself.
if os.environ.get("READTHEDOCS_VERSION_TYPE") == "tag":
    version = os.environ.get("READTHEDOCS_GIT_IDENTIFIER", version).lstrip("v")
release = version

myst_substitutions = {
    "today": dt.datetime.utcnow().strftime("%Y-%m-%d"),
    "release": release,
}
myst_heading_anchors = 3

language = "en"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "links.rst",
    "**.ipynb_checkpoints",
    "notebooks/conftest.py",
    "notebooks/downloads",
    "changes",
]

pygments_style = "sphinx"

todo_include_todos = False

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- myst_nb specifics --
# Notebooks are MyST markdown (jupytext) without outputs; they are executed at
# build time.
nb_execution_mode = "auto"
nb_execution_kernel_name = "python3"
nb_execution_in_temp = True
nb_execution_timeout = 600
# fail the build instead of publishing a traceback
nb_execution_raise_on_error = True
# HoloViews also emits a comm payload for live kernels; static docs use the
# HTML output, so the unknown mime type is expected.
suppress_warnings = ["mystnb.unknown_mime_type"]

# -- Options for HTML output -------------------------------------------
html_theme = "pydata_sphinx_theme"
html_logo = "_static/radarx_logo_micro.svg"


def _custom_edit_url(
    github_user,
    github_repo,
    github_version,
    docpath,
    filename,
    default_edit_page_url_template,
):
    if filename.startswith("generated/"):
        modpath = os.sep.join(
            os.path.splitext(filename)[0].split("/")[-1].split(".")[:-1]
        )
        if modpath == "modules":
            modpath = "radarx"
        rel_modpath = os.path.join("..", modpath)
        if os.path.isdir(rel_modpath):
            docpath = modpath + "/"
            filename = "__init__.py"
        elif os.path.isfile(rel_modpath + ".py"):
            docpath = os.path.dirname(modpath)
            filename = os.path.basename(modpath) + ".py"
        else:
            warnings.warn(f"Not sure how to generate the API URL for: {filename}")
    return default_edit_page_url_template.format(
        github_user=github_user,
        github_repo=github_repo,
        github_version=github_version,
        docpath=docpath,
        filename=filename,
    )


html_context = {
    "github_url": "https://github.com",
    "github_user": "syedhamidali",
    "github_repo": "radarx",
    "github_version": "main",
    "doc_path": "docs",
    "edit_page_url_template": (
        "{{ radarx_custom_edit_url(github_user, github_repo, github_version, "
        "doc_path, file_name, default_edit_page_url_template) }}"
    ),
    "default_edit_page_url_template": (
        "https://github.com/{github_user}/{github_repo}/edit/"
        "{github_version}/{docpath}/{filename}"
    ),
    "radarx_custom_edit_url": _custom_edit_url,
}

html_theme_options = {
    "announcement": (
        "<p>radarx is in an early stage of development, please report any "
        "issues <a href='https://github.com/syedhamidali/radarx/issues'>"
        "here!</a></p>"
    ),
    "github_url": "https://github.com/syedhamidali/radarx",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/radarx",
            "icon": "fas fa-box",
        },
        {
            "type": "local",
            "name": "syedhamidali",
            "url": "https://syedha.com",
            "icon": "_static/Radarx_logo_micro.png",
        },
    ],
    "navbar_end": ["theme-switcher", "icon-links.html"],
    "use_edit_page_button": True,
}

html_static_path = ["_static"]

favicons = [
    {
        "rel": "icon",
        "sizes": "16x16",
        "href": "Radarx_logo_favicon.png",
    },
    {
        "rel": "icon",
        "sizes": "32x32",
        "href": "Radarx_logo_favicon.png",
    },
]

htmlhelp_basename = "radarxdoc"

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_special_with_doc = False
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "scalar": ":term:`scalar`",
    "sequence": ":term:`sequence`",
    "callable": ":py:func:`callable`",
    "file-like": ":term:`file-like <file-like object>`",
    "array-like": ":term:`array-like <array_like>`",
    "Path": "~~pathlib.Path",
}

man_pages = [(master_doc, "radarx", "radarx Documentation", [author], 1)]

texinfo_documents = [
    (
        master_doc,
        "radarx",
        "radarx Documentation",
        author,
        "radarx",
        "One line description of project.",
        "Miscellaneous",
    ),
]

rst_epilog = ""
with open("links.rst") as f:
    rst_epilog += f.read()
