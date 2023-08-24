"""Test Qulacs Estimator Gradients"""

from unittest import TestCase

import numpy as np
from ddt import data, ddt
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import EfficientSU2
from qiskit.circuit.library.standard_gates import RXXGate, RYYGate, RZXGate, RZZGate
from qiskit.quantum_info import SparsePauliOp

from qiskit_qulacs.qulacs_estimator import QulacsEstimator
from qiskit_qulacs.qulacs_estimator_gradient import QulacsEstimatorGradient

gradient_factories = [QulacsEstimatorGradient]


@ddt
class TestQulacsEstimatorGradient(TestCase):
    """Test Estimator Gradient"""

    @data(*gradient_factories)
    def test_gradient_operators(self, grad):
        """Test the estimator gradient for different operators"""
        estimator = QulacsEstimator()
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        gradient = grad(estimator)
        op = SparsePauliOp.from_list([("Z", 1)])
        correct_result = -1 / np.sqrt(2)
        param = [np.pi / 4]
        value = gradient.run([qc], [op], [param]).result().gradients[0]
        self.assertAlmostEqual(value[0], correct_result, 3)

    @data(*gradient_factories)
    def test_gradient_efficient_su2(self, grad):
        """Test the estimator gradient for EfficientSU2"""
        estimator = QulacsEstimator()
        qc = EfficientSU2(2, reps=1)
        op = SparsePauliOp.from_list([("ZI", 1)])
        gradient = grad(estimator)
        param_list = [
            [np.pi / 4 for param in qc.parameters],
            [np.pi / 2 for param in qc.parameters],
        ]
        correct_results = [
            [
                -0.35355339,
                -0.70710678,
                0,
                0.35355339,
                0,
                -0.70710678,
                0,
                0,
            ],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ]
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [op], [param]).result().gradients[0]
            np.testing.assert_allclose(gradients, correct_results[i], atol=1e-3)

    @data(*gradient_factories)
    def test_gradient_2qubit_gate(self, grad):
        """Test the estimator gradient for 2 qubit gates"""
        estimator = QulacsEstimator()
        for gate in [RXXGate, RYYGate, RZZGate, RZXGate]:
            param_list = [[np.pi / 4], [np.pi / 2]]
            correct_results = [
                [-0.70710678],
                [-1],
            ]
            op = SparsePauliOp.from_list([("ZI", 1)])
            for i, param in enumerate(param_list):
                a = Parameter("a")
                qc = QuantumCircuit(2)
                gradient = grad(estimator)

                if gate is RZZGate:
                    qc.h([0, 1])
                    qc.append(gate(a), [qc.qubits[0], qc.qubits[1]], [])
                    qc.h([0, 1])
                else:
                    qc.append(gate(a), [qc.qubits[0], qc.qubits[1]], [])
                gradients = gradient.run([qc], [op], [param]).result().gradients[0]
                np.testing.assert_allclose(gradients, correct_results[i], atol=1e-3)

    @data(*gradient_factories)
    def test_gradient_parameters(self, grad):
        """Test the estimator gradient for parameters"""
        estimator = QulacsEstimator()
        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc.rx(b, 0)
        gradient = grad(estimator)
        param_list = [[np.pi / 4, np.pi / 2]]
        correct_results = [
            [-0.70710678],
        ]
        op = SparsePauliOp.from_list([("Z", 1)])
        for i, param in enumerate(param_list):
            gradients = (
                gradient.run([qc], [op], [param], parameters=[[a]])
                .result()
                .gradients[0]
            )
            np.testing.assert_allclose(gradients, correct_results[i], atol=1e-3)

        # parameter order
        with self.subTest(msg="The order of gradients"):
            c = Parameter("c")
            qc = QuantumCircuit(1)
            qc.rx(a, 0)
            qc.rz(b, 0)
            qc.rx(c, 0)

            param_list = [[np.pi / 4, np.pi / 2, np.pi / 3]]
            correct_results = [
                [-0.35355339, 0.61237244, -0.61237244],
                [-0.61237244, 0.61237244, -0.35355339],
                [-0.35355339, -0.61237244],
                [-0.61237244, -0.35355339],
            ]
            param = [[a, b, c], [c, b, a], [a, c], [c, a]]
            op = SparsePauliOp.from_list([("Z", 1)])
            for i, p in enumerate(param):
                gradient = grad(estimator)
                gradients = (
                    gradient.run([qc], [op], param_list, parameters=[p])
                    .result()
                    .gradients[0]
                )
                np.testing.assert_allclose(gradients, correct_results[i], atol=1e-3)

    @data(*gradient_factories)
    def test_gradient_multi_arguments(self, grad):
        """Test the estimator gradient for multiple arguments"""
        estimator = QulacsEstimator()
        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc2 = QuantumCircuit(1)
        qc2.rx(b, 0)
        gradient = grad(estimator)
        param_list = [[np.pi / 4], [np.pi / 2]]
        correct_results = [
            [-0.70710678],
            [-1],
        ]
        op = SparsePauliOp.from_list([("Z", 1)])
        gradients = gradient.run([qc, qc2], [op] * 2, param_list).result().gradients
        np.testing.assert_allclose(gradients, correct_results, atol=1e-3)

        c = Parameter("c")
        qc3 = QuantumCircuit(1)
        qc3.rx(c, 0)
        qc3.ry(a, 0)
        param_list2 = [[np.pi / 4], [np.pi / 4, np.pi / 4], [np.pi / 4, np.pi / 4]]
        correct_results2 = [
            [-0.70710678],
            [-0.5],
            [-0.5, -0.5],
        ]
        gradients2 = (
            gradient.run(
                [qc, qc3, qc3], [op] * 3, param_list2, parameters=[[a], [c], None]
            )
            .result()
            .gradients
        )
        np.testing.assert_allclose(gradients2[0], correct_results2[0], atol=1e-3)
        np.testing.assert_allclose(gradients2[1], correct_results2[1], atol=1e-3)
        np.testing.assert_allclose(gradients2[2], correct_results2[2], atol=1e-3)

    @data(*gradient_factories)
    def test_gradient_validation(self, grad):
        """Test estimator gradient's validation"""
        estimator = QulacsEstimator()
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        gradient = grad(estimator)
        param_list = [[np.pi / 4], [np.pi / 2]]
        op = SparsePauliOp.from_list([("Z", 1)])
        with self.assertRaises(ValueError):
            gradient.run([qc], [op], param_list)
        with self.assertRaises(ValueError):
            gradient.run([qc, qc], [op, op], param_list, parameters=[[a]])
        with self.assertRaises(ValueError):
            gradient.run([qc, qc], [op], param_list, parameters=[[a]])
        with self.assertRaises(ValueError):
            gradient.run([qc], [op], [[np.pi / 4, np.pi / 4]])
