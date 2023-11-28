"""Tests for qulacs backend."""

from typing import Dict, List
from unittest import TestCase

import numpy as np
from qiskit import BasicAer, QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit

from qiskit_qulacs.qulacs_backend import QulacsBackend

_EPS = 1e-10  # global variable used to chop very small numbers to zero


def combine_dicts(
    dict1: Dict[str, float], dict2: Dict[str, float]
) -> Dict[str, List[float]]:
    """Combines dictionaries with different keys.

    Args:
                    dict1: first
                    dict2: second

    Returns:
                    merged dicts with list of keys
    """
    combined_dict: Dict[str, List[float]] = {}
    for key in dict1.keys():
        if key in combined_dict:
            combined_dict[key].append(dict1[key])
        else:
            combined_dict[key] = [dict1[key]]
    for key in dict2.keys():
        if key in combined_dict:
            combined_dict[key].append(dict2[key])
        else:
            combined_dict[key] = [dict2[key]]
    return combined_dict


class TestQulacsBackend(TestCase):
    """Tests BraketBackend."""

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
        backend = QulacsBackend()
        aer_backend = BasicAer.get_backend("statevector_simulator")

        for i in range(1, 10):
            with self.subTest(f"Random circuit with {i} qubits."):
                circuit = random_circuit(i, 5, seed=42)
                qulacs_result = backend.run(circuit).result().get_statevector()

                transpiled_aer_circuit = transpile(
                    circuit, backend=aer_backend, seed_transpiler=42
                )

                aer_result = (
                    aer_backend.run(transpiled_aer_circuit).result().get_statevector()
                )

                self.assertTrue(np.linalg.norm(qulacs_result - aer_result) < _EPS)
