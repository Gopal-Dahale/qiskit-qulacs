# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""
Sphinx documentation builder
"""

import datetime
import os

# General options:
from pathlib import Path
from typing import Any, Dict, Optional

from importlib_metadata import version as metadata_version

project = "Qiskit-Qulacs"
author = "Gopal Ramesh Dahale"
copyright = (
    f"2023-{datetime.date.today().year}, {author}"  # pylint: disable=redefined-builtin
)


_rootdir = Path(__file__).parent.parent

# The full version, including alpha/beta/rc tags
version_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "qiskit_qulacs",
    "version.py",
)
version_dict: Optional[Dict[str, Any]] = {}
with open(version_path) as fp:
    exec(fp.read(), version_dict)
version = version_dict["__version__"]

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "jupyter_sphinx",
    "sphinx_autodoc_typehints",
    "reno.sphinxext",
    "nbsphinx",
    "myst_parser",
    "qiskit_sphinx_theme",
]
templates_path = ["_templates"]
numfig = True
numfig_format = {"table": "Table %s"}
language = "en"
pygments_style = "colorful"
add_module_names = False
modindex_common_prefix = ["qiskit_qulacs."]

# html theme options
html_theme = "qiskit-ecosystem"
html_static_path = ["_static"]
# html_logo = "_static/images/logo.png"
html_last_updated_fmt = "%Y/%m/%d"
html_title = f"{project} {version}"


# autodoc/autosummary options
autosummary_generate = True
autosummary_generate_overwrite = False
autoclass_content = "both"

# nbsphinx options (for tutorials)
nbsphinx_timeout = 180
nbsphinx_execute = "always"
nbsphinx_widgets_path = ""
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
