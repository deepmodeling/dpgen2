# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from datetime import (
    date,
)

# -- Project information -----------------------------------------------------

project = "DPGEN2"
copyright = "2022-%d, DeepModeling" % date.today().year
author = "DeepModeling"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "deepmodeling_sphinx",
    "dargs.sphinx",
    "myst_parser",
    "sphinx_book_theme",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx.ext.autosummary",
    "sphinxarg.ext",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = []

autodoc_default_flags = ["members"]
autosummary_generate = True
master_doc = "index"


def run_apidoc(_):
    from sphinx.ext.apidoc import (
        main,
    )

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    module = os.path.join(cur_dir, "..", "dpgen2")
    main(
        [
            "-M",
            "--tocfile",
            "api",
            "-H",
            "DPGEN2 API",
            "-o",
            os.path.join(cur_dir, "api"),
            module,
            "--force",
        ]
    )


def setup(app):
    app.connect("builder-inited", run_apidoc)


intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "dargs": ("https://docs.deepmodeling.com/projects/dargs/en/latest/", None),
    "dflow": ("https://deepmodeling.com/dflow/", None),
    "dpdata": ("https://docs.deepmodeling.com/projects/dpdata/en/latest/", None),
}
