#!/usr/bin/env python
#
# radarx documentation build configuration file, created by
# sphinx-quickstart.
#
import datetime as dt
import glob
import types
import os
import subprocess
import sys
import warnings
from importlib.metadata import version

sys.path.insert(0, os.path.abspath(".."))

# check readthedocs
on_rtd = os.environ.get("READTHEDOCS") == "True"

# processing on readthedocs
if on_rtd:
    # install radarx from checked out source
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    print(f"Installing commit {commit}")
    url = "https://github.com/syedhamidali/radarx.git"
    subprocess.check_call(
        ["python", "-m", "pip", "install", "--no-deps", f"git+{url}@{commit}"]
    )

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
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_favicon",
    "nbsphinx",
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

source_suffix = [".rst", ".md"]

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

version = version("radarx")
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
]

pygments_style = "sphinx"

todo_include_todos = False

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- nbsphinx specifics --
nbsphinx_execute = "always"
subprocess.check_call(["cp", "-rp", "../examples/notebooks", "."])

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
    "edit_page_url_template": "{{ radarx_custom_edit_url(github_user, github_repo, github_version, doc_path, file_name, default_edit_page_url_template) }}",
    "default_edit_page_url_template": "https://github.com/{github_user}/{github_repo}/edit/{github_version}/{docpath}/{filename}",
    "radarx_custom_edit_url": _custom_edit_url,
}

html_theme_options = {
    "announcement": "<p>radarx is in an early stage of development, please report any issues <a href='https://github.com/syedhamidali/radarx/issues'>here!</a></p>",
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
