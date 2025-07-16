# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import pathlib

import eko

_HERE = pathlib.Path(__file__).absolute().parent

project = "geko"
copyright = "2025, Felix Hekhorn"
author = "Felix Hekhorn"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Extension configuration -------------------------------------------------
bibtex_bibfiles = ["refs.bib"]

# Example configuration for intersphinx: refer to the Python standard library.
# Thanks https://github.com/bskinn/sphobjinv
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "eko": (f"https://eko.readthedocs.io/en/v{eko.__version__}/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# A string to be included at the beginning of all files
rst_prolog = (_HERE / "_static" / "abbreviations.rst").read_text(encoding="utf-8")


# https://github.com/readthedocs/readthedocs.org/issues/1139#issuecomment-312626491
def run_apidoc(_):
    """Run apidoc."""
    # Starting from Sphinx 8.2 it is available as extension: https://github.com/sphinx-doc/sphinx/pull/13333
    import sys  # pylint: disable=import-outside-toplevel

    from sphinx.ext.apidoc import main  # pylint: disable=import-outside-toplevel

    sys.path.append(str(_HERE.parent))
    for pkg, docs_dest in dict(
        geko=_HERE / "modules" / "geko",
    ).items():
        package = _HERE.parent / "src" / pkg
        main(["--module-first", "-o", str(docs_dest), str(package)])
        (docs_dest / "modules.rst").unlink()


def setup(app):
    """Configure Sphinx."""
    app.setup_extension("sphinx.ext.autodoc")
    app.connect("builder-inited", run_apidoc)
