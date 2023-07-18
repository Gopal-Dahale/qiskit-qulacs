# See pyproject.toml for project configuration.
# This file exists for compatibility with legacy tools:
# https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html

import os
from typing import Any, Dict, Optional

import setuptools

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

version_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "qiskit_qulacs", "version.py")
)

version_dict: Optional[Dict[str, Any]] = {}
with open(version_path) as fp:
    exec(fp.read(), version_dict)
version = version_dict["__version__"]

setuptools.setup(
    name="qiskit_qulacs",
    description="Qiskit Qulacs to execute Qiskit "
    "programs using Qulacs as backend.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="qiskit qulacs quantum",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    python_requires=">=3.7",
    version=version,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)