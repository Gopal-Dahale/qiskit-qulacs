"""Tests for qulacs backend."""

from unittest import TestCase

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit
from qiskit.result import Result
from qiskit_aer import Aer

from qiskit_qulacs.qulacs_backend import QulacsBackend
from tests.utils import dicts_almost_equal

_EPS = 1e-10  # global variable used to chop very small numbers to zero


class TestQulacsBackend(TestCase):
    """Tests BraketBackend."""

    def setUp(self):
        self.aer_backend = Aer.get_backend("aer_simulator_statevector")

    def test_qulacs_backend_output(self):
        """Test qulacs backend output"""
        qulacs_backend = QulacsBackend()
        self.assertEqual(qulacs_backend.name(), "qulacs_simulator")

    def test_qulacs_backend_circuit(self):
        """Tests qulacs backend with circuit."""
        backend = QulacsBackend()
        circuits = []

        # Circuit 0
        q_c = QuantumCircuit(2)
        q_c.x(0)
        q_c.cx(0, 1)
        circuits.append(q_c)

        # Circuit 1
        q_c = QuantumCircuit(2)
        q_c.h(0)
        q_c.cx(0, 1)
        circuits.append(q_c)

        results = []
        for circuit in circuits:
            results.append(backend.run(circuit).result())

        # Result 0
        self.assertTrue(
            (
                np.linalg.norm(results[0].get_statevector() - np.array([0, 0, 0, 1]))
                < _EPS
            )
        )
        # Result 1
        _00 = np.abs(results[1].get_statevector()[0]) ** 2
        _11 = np.abs(results[1].get_statevector()[-1]) ** 2
        self.assertTrue(np.allclose([_00, _11], [0.5, 0.5]))

    def test_random_circuits(self):
        """Tests with random circuits."""
        qulacs_backend = QulacsBackend()

        for i in range(1, 10):
            with self.subTest(f"Random circuit with {i} qubits."):
                qiskit_circuit = random_circuit(i, 5, seed=42)
                transpiled_qiskit_circuit = transpile(qiskit_circuit, qulacs_backend)

                qulacs_result = (
                    qulacs_backend.run(transpiled_qiskit_circuit)
                    .result()
                    .get_statevector()
                )

                transpiled_qiskit_circuit.save_statevector()
                aer_result = (
                    self.aer_backend.run(transpiled_qiskit_circuit)
                    .result()
                    .get_statevector()
                    .data
                )

                self.assertTrue(np.linalg.norm(qulacs_result - aer_result) < _EPS)

    def test_single_shot(self):
        """Test single shot run."""
        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)

        shots = 1
        qulacs_backend = QulacsBackend()
        result = qulacs_backend.run(bell, shots=shots, seed_simulator=10).result()

        self.assertEqual(result.success, True)

    def test_against_reference(self):
        """Test data counts output for single circuit run against reference."""
        qulacs_backend = QulacsBackend()
        shots = 1024
        threshold = 0.04 * shots

        qc = QuantumCircuit(6)
        qc.h(range(6))
        qc.cx([0, 1, 2], [3, 4, 5])

        counts = (
            qulacs_backend.run(qc, shots=shots, seed_simulator=10).result().get_counts()
        )
        counts = {i: j / shots for i, j in counts.items()}

        qc.save_statevector()
        target = (
            self.aer_backend.run(qc, shots=shots, seed_simulator=10)
            .result()
            .get_counts()
        )
        error_msg = dicts_almost_equal(counts, target, threshold)

        if error_msg:
            msg = self._formatMessage(None, error_msg)
            raise self.failureException(msg)

    def test_options(self):
        """Test for options"""
        backend_options = {
            "shots": 3000,
            "seed_simulator": 42,
            "device": "CPU",
            "qco_enable": True,
            "qco_method": "greedy",
            "qco_max_block_size": 5,
        }

        with self.subTest("set_options"):
            qulacs_backend = QulacsBackend()
            qulacs_backend.set_options(**backend_options)
            self.assertEqual(qulacs_backend.options.get("shots"), 3000)
            self.assertEqual(qulacs_backend.options.get("seed_simulator"), 42)
            self.assertEqual(qulacs_backend.options.get("device"), "CPU")
            self.assertEqual(qulacs_backend.options.get("qco_enable"), True)
            self.assertEqual(qulacs_backend.options.get("qco_method"), "greedy")
            self.assertEqual(qulacs_backend.options.get("qco_max_block_size"), 5)

        with self.subTest("run"):
            bell = QuantumCircuit(2)
            bell.h(0)
            bell.cx(0, 1)

            qulacs_backend = QulacsBackend()
            result = qulacs_backend.run(
                [bell],
                qco_enable=True,
                qco_method="light",
            ).result()
            self.assertIsInstance(result, Result)
            np.testing.assert_allclose(
                result.get_statevector(),
                (1 / np.sqrt(2)) * np.array([1.0, 0.0, 0.0, 1.0]),
            )

    def test_repr(self):
        """Test string repr"""
        qulacs_backend = QulacsBackend()
        self.assertEqual(str(qulacs_backend), "qulacs_simulator")
