"""Tests for AWS Braket provider."""

from unittest import TestCase

from qiskit_qulacs.qulacs_backend import QulacsBackend
from qiskit_qulacs.qulacs_provider import QulacsProvider


class TestQulacsProvider(TestCase):
    """Tests QulacsProvider."""

    def setUp(self):
        self.provider = QulacsProvider()

    def test_provider_backends(self):
        """Tests provider."""
        backends = self.provider.backends()

        self.assertTrue(len(backends) > 0)
        for backend in backends:
            with self.subTest(f"{backend.name}"):
                self.assertIsInstance(backend, QulacsBackend)

    def test_get_backend(self):
        """get backend"""
        backend = self.provider.get_backend(name="qulacs_simulator")
        self.assertTrue(isinstance(backend, QulacsBackend))

    def test_str(self):
        """Test string repr"""
        self.assertEqual(str(self.provider), "QulacsProvider")
