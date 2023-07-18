"""Qiskit-Qulacs"""

from importlib_metadata import version as metadata_version, PackageNotFoundError

from .qulacs_provider import QulacsProvider


try:
    __version__ = metadata_version("qiskit_qulacs")
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    pass
