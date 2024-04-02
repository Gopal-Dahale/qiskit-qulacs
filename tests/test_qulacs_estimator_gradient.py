"""Test Qulacs Estimator Gradients"""

from unittest import TestCase

import numpy as np
import pytest
from ddt import data, ddt
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import EfficientSU2
from qiskit.circuit.library.standard_gates import RXXGate, RYYGate, RZXGate, RZZGate
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import ReverseEstimatorGradient

from qiskit_qulacs.qulacs_backend import QulacsBackend
from qiskit_qulacs.qulacs_estimator_gradient import QulacsEstimatorGradient

gradient_factories = [QulacsEstimatorGradient]


@ddt
class TestQulacsEstimatorGradient(TestCase):
    """Test Estimator Gradient"""

    @data(*gradient_factories)
    def test_gradient_operators(self, grad):
        """Test the estimator gradient for different operators"""
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        gradient = grad()
        op = SparsePauliOp.from_list([("Z", 1)])
        correct_result = -1 / np.sqrt(2)
        param = [np.pi / 4]
        value = gradient.run([qc], [op], [param]).result().gradients[0]
        self.assertAlmostEqual(value[0], correct_result, 3)

    @data(*gradient_factories)
    def test_gradient_efficient_su2(self, grad):
        """Test the estimator gradient for EfficientSU2"""
        qc = EfficientSU2(2, reps=1).decompose()
        op = SparsePauliOp.from_list([("ZI", 1)])
        gradient = grad()
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
        qulacs_backend = QulacsBackend()
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
                gradient = grad()

                if gate is RZZGate:
                    qc.h([0, 1])
                    qc.append(gate(a), [qc.qubits[0], qc.qubits[1]], [])
                    qc.h([0, 1])
                else:
                    qc.append(gate(a), [qc.qubits[0], qc.qubits[1]], [])

                tqc = transpile(qc, qulacs_backend)
                gradients = gradient.run([tqc], [op], [param]).result().gradients[0]
                np.testing.assert_allclose(gradients, correct_results[i], atol=1e-3)

    @data(*gradient_factories)
    def test_gradient_parameters(self, grad):
        """Test the estimator gradient for parameters"""
        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc.rx(b, 0)
        gradient = grad()
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
                gradient = grad()
                gradients = (
                    gradient.run([qc], [op], param_list, parameters=[p])
                    .result()
                    .gradients[0]
                )
                np.testing.assert_allclose(gradients, correct_results[i], atol=1e-3)

    @data(*gradient_factories)
    def test_gradient_multi_arguments(self, grad):
        """Test the estimator gradient for multiple arguments"""
        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc2 = QuantumCircuit(1)
        qc2.rx(b, 0)
        gradient = grad()
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
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        gradient = grad()
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

    @data(*gradient_factories)
    def test_gradient_with_parameter_vector(self, grad):
        """Tests that the gradient of a circuit with a parameter vector is calculated correctly."""
        qiskit_circuit = QuantumCircuit(1)

        theta_param = ParameterVector("θ", 2)
        theta_val = np.array([np.pi / 4, np.pi / 16])

        qiskit_circuit.rx(theta_param[0], 0)
        qiskit_circuit.rx(theta_param[1] * 4, 0)

        op = SparsePauliOp.from_list([("Z", 1)])

        est_grad = grad()
        have_gradient = (
            est_grad.run([qiskit_circuit], [op], [theta_val]).result().gradients[0]
        )

        want_gradient = [-1, -4]
        assert np.allclose(have_gradient, want_gradient)

    @data(*gradient_factories)
    def test_gradient_with_parameter_expressions(self, grad):
        """Tests that the gradient of a circuit with parameter expressions is calculated correctly."""
        qiskit_circuit = QuantumCircuit(1)

        theta_param = ParameterVector("θ", 3)
        theta_val = [3 * np.pi / 16, np.pi / 64]

        phi_param = Parameter("φ")
        phi_val = [np.pi / 8]

        # Apply an instruction with a regular parameter.
        qiskit_circuit.rx(phi_param, 0)
        # Apply an instruction with a parameter vector element.
        qiskit_circuit.rx(theta_param[0], 0)
        # Apply an instruction with a parameter expression involving one parameter.
        qiskit_circuit.rx(theta_param[1] + theta_param[1] + np.cos(theta_param[1]), 0)

        op = SparsePauliOp.from_list([("Z", 1)])
        est_grad = grad()
        qiskit_grad = ReverseEstimatorGradient()

        have_gradient = (
            est_grad.run([qiskit_circuit], [op], [theta_val + phi_val])
            .result()
            .gradients[0]
        )

        want_gradient = (
            qiskit_grad.run([qiskit_circuit], [op], [theta_val + phi_val])
            .result()
            .gradients[0]
        )

        self.assertTrue(np.allclose(have_gradient, want_gradient))


@ddt
class TestQulacsEstimatorGradientWarningsAndErrors(TestCase):
    """Test Estimator Gradient"""

    @data(*gradient_factories)
    def test_gradient_with_parameter_expression_having_two_paramters(self, grad):
        """Test gradient when two different parameters are passed in a single expression"""
        qiskit_circuit = QuantumCircuit(1)
        theta_param = ParameterVector("θ", 2)

        # Apply an instruction with a parameter expression involving two parameters.
        qiskit_circuit.rx(3 * theta_param[0] + theta_param[1], 0)

        op = SparsePauliOp.from_list([("Z", 1)])
        est_grad = grad()

        with pytest.raises(RuntimeError, match="Variable w.r.t should be given"):
            est_grad.run([qiskit_circuit], [op], [[0.2, 0.3]]).result()
