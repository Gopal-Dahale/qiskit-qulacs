"""QulacsProvider class."""
from qiskit.providers import ProviderV1 as Provider
from qiskit.providers.providerutils import filter_backends

from .qulacs_backend import QulacsBackend


class QulacsProvider(Provider):
    """QulacsProvider class."""

    @staticmethod
    def _get_backends():
        return [("qulacs_simulator", QulacsBackend)]

    def get_backend(self, name=None, **kwargs):
        return super().get_backend(name=name, **kwargs)

    def backends(self, name=None, filters=None, **kwargs):
        backends = []
        for backend_name, backend_cls in self._get_backends():
            if name is None or backend_name == name:
                backends.append(backend_cls(provider=self))
        return filter_backends(backends, filters=filters, **kwargs)

    def __str__(self):
        return "QulacsProvider"
