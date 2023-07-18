from qiskit.providers import ProviderV1 as Provider
from qiskit.providers.providerutils import filter_backends

from .qulacs_backend import QulacsBackend

class QulacsProvider(Provider):
    def __init__(self, token=None):
        super().__init__()
        self.token = token

    def backends(self, name=None, filters=None, **kwargs):
        backends = [QulacsBackend()]

        if name:
            backends = [
                backend for backend in backends if backend.name == name]

        return filter_backends(backends, filters=filters, **kwargs)