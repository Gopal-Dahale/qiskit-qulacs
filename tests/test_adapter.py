"""Tests for the Adapter class."""

from unittest import TestCase

import numpy as np
import pytest
import qiskit.circuit.library as lib
from ddt import data, ddt
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import PauliEvolutionGate, TwoLocal
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import Aer

from qiskit_qulacs.adapter import (
    convert_qiskit_to_qulacs_circuit,
    convert_sparse_pauliop_to_qulacs_obs,
)
from qiskit_qulacs.qulacs_backend import QulacsBackend
from qulacs import Observable, ParametricQuantumCircuit, PauliOperator, QuantumState

# FSim Gate with fixed parameters
# source: https://quantumai.google/reference/python/cirq/FSimGate
fsim_mat = np.array(
    [
        [1, 0, 0, 0],
        [0, np.cos(np.pi / 3), -1j * np.sin(np.pi / 3), 0],
        [0, -1j * np.sin(np.pi / 3), np.cos(np.pi / 3), 0],
        [0, 0, 0, np.exp(-1j * np.pi / 4)],
    ]
)

qiskit_standard_gates = [
    lib.IGate(),
    lib.SXGate(),
    lib.XGate(),
    lib.CXGate(),
    lib.RZGate(Parameter("λ")),
    lib.RGate(Parameter("ϴ"), Parameter("φ")),
    lib.C3SXGate(),
    lib.CCXGate(),
    lib.DCXGate(),
    lib.CHGate(),
    lib.CPhaseGate(Parameter("ϴ")),
    lib.CRXGate(Parameter("ϴ")),
    lib.CRYGate(Parameter("ϴ")),
    lib.CRZGate(Parameter("ϴ")),
    lib.CSwapGate(),
    lib.CSXGate(),
    lib.CUGate(Parameter("ϴ"), Parameter("φ"), Parameter("λ"), Parameter("γ")),
    lib.CU1Gate(Parameter("λ")),
    lib.CU3Gate(Parameter("ϴ"), Parameter("φ"), Parameter("λ")),
    lib.CYGate(),
    lib.CZGate(),
    lib.CCZGate(),
    lib.HGate(),
    lib.PhaseGate(Parameter("ϴ")),
    lib.RCCXGate(),
    lib.RC3XGate(),
    lib.RXGate(Parameter("ϴ")),
    lib.RXXGate(Parameter("ϴ")),
    lib.RYGate(Parameter("ϴ")),
    lib.RYYGate(Parameter("ϴ")),
    lib.RZZGate(Parameter("ϴ")),
    lib.RZXGate(Parameter("ϴ")),
    lib.XXMinusYYGate(Parameter("ϴ"), Parameter("φ")),
    lib.XXPlusYYGate(Parameter("ϴ"), Parameter("φ")),
    lib.ECRGate(),
    lib.SGate(),
    lib.SdgGate(),
    lib.CSGate(),
    lib.CSdgGate(),
    lib.SwapGate(),
    lib.iSwapGate(),
    lib.SXdgGate(),
    lib.TGate(),
    lib.TdgGate(),
    lib.UGate(Parameter("ϴ"), Parameter("φ"), Parameter("λ")),
    lib.U1Gate(Parameter("λ")),
    lib.U2Gate(Parameter("φ"), Parameter("λ")),
    lib.U3Gate(Parameter("ϴ"), Parameter("φ"), Parameter("λ")),
    lib.YGate(),
    lib.ZGate(),
]


def convert_and_check_statevector(testcase, qc, params=[]):
    """
    The function converts a Qiskit quantum circuit to a Qulacs circuit,
    obtains the statevectors from both frameworks, and checks if they are close.

    Args:
      testcase: instance of a unittest test case.
      qc: quantum circuit in Qiskit.
      params: list that contains the parameters to be assigned to the quantum circuit `qc`.
    """
    # convert qiskit's quantum circuit to qulacs
    qulacs_circuit_builder = convert_qiskit_to_qulacs_circuit(qc)
    qulacs_circuit = qulacs_circuit_builder(params)[0]

    # Obtaining statevector from qiskit
    if params:
        qc = qc.assign_parameters(params)
    qc.save_statevector()
    qiskit_sv = testcase.aer_backend.run(qc).result().get_statevector().data

    # Obtaining statevector from qulacs
    quantum_state = QuantumState(qulacs_circuit.get_qubit_count())
    qulacs_circuit.update_quantum_state(quantum_state)
    qulacs_sv = quantum_state.get_vector()

    testcase.assertTrue(np.allclose(qiskit_sv, qulacs_sv))


class TestAdapterConverter(TestCase):
    """Tests for the Adapter class."""

    def setUp(self):
        self.aer_backend = Aer.get_backend("aer_simulator_statevector")

    def test_state_preparation_01(self):
        """Tests state_preparation handling of Adapter"""
        qulacs_backend = QulacsBackend()

        input_state_vector = np.array([np.sqrt(3) / 2, np.sqrt(2) * complex(1, 1) / 4])

        qiskit_circuit = QuantumCircuit(1)
        qiskit_circuit.prepare_state(input_state_vector, 0)
        transpiled_qiskit_circuit = transpile(qiskit_circuit, qulacs_backend)

        convert_and_check_statevector(self, transpiled_qiskit_circuit)

    def test_state_preparation_00(self):
        """Tests state_preparation handling of Adapter"""
        qulacs_backend = QulacsBackend()
        input_state_vector = np.array([1 / np.sqrt(2), -1 / np.sqrt(2)])

        qiskit_circuit = QuantumCircuit(1)
        qiskit_circuit.prepare_state(input_state_vector, 0)
        transpiled_qiskit_circuit = transpile(qiskit_circuit, qulacs_backend)

        convert_and_check_statevector(self, transpiled_qiskit_circuit)

    def test_convert_parametric_qiskit_to_qulacs_circuit(self):
        """Tests convert_qiskit_to_qulacs_circuit works with parametric circuits."""

        theta = Parameter("θ")
        phi = Parameter("φ")
        lam = Parameter("λ")

        params = np.array([np.pi, np.pi / 2, np.pi / 3])

        qiskit_circuit = QuantumCircuit(1, 1)
        qiskit_circuit.rx(theta, 0)
        qiskit_circuit.ry(phi, 0)
        qiskit_circuit.rz(lam, 0)

        qulacs_circuit_builder = convert_qiskit_to_qulacs_circuit(qiskit_circuit)
        qulacs_circuit = qulacs_circuit_builder(params)[0]
        quantum_state = QuantumState(1)
        qulacs_circuit.update_quantum_state(quantum_state)
        qulacs_result = quantum_state.get_vector()

        # https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.html#qiskit.circuit.QuantumCircuit.parameters
        # Based on the above docs, the Paramters are sorted alphabetically.
        # Therefore θ, φ, λ should be sorted as θ, λ, φ.
        # so θ = params[0], λ = params[1], φ = params[2]
        # Also, the sign of the rotation is negative in qulacs.

        qulacs_circuit_ans = ParametricQuantumCircuit(1)
        qulacs_circuit_ans.add_parametric_RX_gate(0, -params[0])
        qulacs_circuit_ans.add_parametric_RY_gate(0, -params[2])
        qulacs_circuit_ans.add_parametric_RZ_gate(0, -params[1])
        quantum_state_ans = QuantumState(1)
        qulacs_circuit_ans.update_quantum_state(quantum_state_ans)
        qulacs_result_ans = quantum_state_ans.get_vector()

        self.assertTrue(np.allclose(qulacs_result, qulacs_result_ans))

    def test_longer_parameter_expression(self):
        """Tests parameter expression with arbitrary operations and length"""

        theta = Parameter("θ")
        phi = Parameter("φ")
        lam = Parameter("λ")

        values = [0.1, 0.2, 0.3]

        qc = QuantumCircuit(1, 1)
        qc.rx(phi * np.cos(theta) + lam, 0)

        convert_and_check_statevector(self, qc, values)

    def test_quantum_circuit_loaded_multiple_times_with_different_arguments(self):
        """Tests that a loaded quantum circuit can be called multiple times with
        different arguments."""

        theta = Parameter("θ")
        angle1 = 0.5
        angle2 = -0.5
        angle3 = 0

        qc = QuantumCircuit(3, 1)
        qc.rz(theta, [0])

        convert_and_check_statevector(self, qc, [angle1])
        convert_and_check_statevector(self, qc, [angle2])
        convert_and_check_statevector(self, qc, [angle3])

    def test_quantum_circuit_with_bound_parameters(self):
        """Tests loading a quantum circuit that already had bound parameters."""

        theta = Parameter("θ")

        qc = QuantumCircuit(3, 1)
        qc.rz(theta, [0])
        qc = qc.assign_parameters({theta: 0.5})

        convert_and_check_statevector(self, qc)

    def test_unused_parameters_are_ignored(self):
        """Tests that unused parameters are ignored during assignment."""
        a, b, c = [Parameter(var) for var in "abc"]
        v = ParameterVector("v", 2)

        qc = QuantumCircuit(1)
        qc.rz(a, 0)

        # convert qiskit's quantum circuit to qulacs
        qulacs_circuit_builder = convert_qiskit_to_qulacs_circuit(qc)
        qulacs_circuit = qulacs_circuit_builder([0.1, 0.2, 0.3, 0.4, 0.5])[0]

        # Obtaining statevector from qiskit
        qc = qc.assign_parameters([0.1])
        qc.save_statevector()
        qiskit_sv = self.aer_backend.run(qc).result().get_statevector().data

        # Obtaining statevector from qulacs
        quantum_state = QuantumState(qulacs_circuit.get_qubit_count())
        qulacs_circuit.update_quantum_state(quantum_state)
        qulacs_sv = quantum_state.get_vector()

        self.assertTrue(np.allclose(qiskit_sv, qulacs_sv))

    def test_unused_parameter_vector_items_are_ignored(self):
        """Tests that unused parameter vector items are ignored during assignment."""

        a, b, c = [Parameter(var) for var in "abc"]
        v = ParameterVector("v", 2)

        qc = QuantumCircuit(1)
        qc.rz(v[1], 0)

        # convert qiskit's quantum circuit to qulacs
        qulacs_circuit_builder = convert_qiskit_to_qulacs_circuit(qc)
        qulacs_circuit = qulacs_circuit_builder([0.1, 0.2, 0.3, 0.4, 0.5])[0]

        # Obtaining statevector from qiskit
        qc = qc.assign_parameters([0.1])
        qc.save_statevector()
        qiskit_sv = self.aer_backend.run(qc).result().get_statevector().data

        # Obtaining statevector from qulacs
        quantum_state = QuantumState(qulacs_circuit.get_qubit_count())
        qulacs_circuit.update_quantum_state(quantum_state)
        qulacs_sv = quantum_state.get_vector()

        self.assertTrue(np.allclose(qiskit_sv, qulacs_sv))

    def test_wires_two_different_quantum_registers(self):
        """Tests loading a circuit with the three-qubit operations supported by PennyLane."""

        three_wires = [0, 1, 2]

        qr1 = QuantumRegister(2)
        qr2 = QuantumRegister(1)

        qc = QuantumCircuit(qr1, qr2)
        qc.cswap(*three_wires)

        convert_and_check_statevector(self, qc)


class TestConverterGates(TestCase):
    """Tests for the Adapter class."""

    def setUp(self):
        self.aer_backend = Aer.get_backend("aer_simulator_statevector")

    def test_u_gate(self):
        """Tests adapter conversion of u gate"""
        qiskit_circuit = QuantumCircuit(1)
        qiskit_circuit.u(np.pi / 2, np.pi / 3, np.pi / 4, 0)
        convert_and_check_statevector(self, qiskit_circuit)

    def test_standard_gate_decomp(self):
        """Tests adapter decomposition of all standard gates to forms that can be translated"""
        qulacs_backend = QulacsBackend()

        for standard_gate in qiskit_standard_gates:
            qiskit_circuit = QuantumCircuit(standard_gate.num_qubits)
            qiskit_circuit.append(standard_gate, range(standard_gate.num_qubits))

            parameters = standard_gate.params
            if parameters:
                parameter_values = [
                    (137 / 61) * np.pi / i for i in range(1, len(parameters) + 1)
                ]
                parameter_bindings = dict(zip(parameters, parameter_values))
                qiskit_circuit = qiskit_circuit.assign_parameters(parameter_bindings)

            transpiled_qiskit_circuit = transpile(qiskit_circuit, qulacs_backend)

            with self.subTest(f"Circuit with {standard_gate.name} gate."):
                qulacs_job = qulacs_backend.run(transpiled_qiskit_circuit)
                qulacs_result = qulacs_job.result().get_statevector()

                transpiled_qiskit_circuit.save_statevector()
                qiskit_job = self.aer_backend.run(transpiled_qiskit_circuit)
                qiskit_result = qiskit_job.result().get_statevector().data

                self.assertTrue(np.allclose(qulacs_result, qiskit_result))

    def test_exponential_gate_decomp(self):
        """Tests adapter translation of exponential gates"""
        qulacs_backend = QulacsBackend()
        qiskit_circuit = QuantumCircuit(2)

        hamiltonian = SparsePauliOp(["ZZ", "XI"], [1.0, -0.1])
        evo = PauliEvolutionGate(hamiltonian, time=2)

        qiskit_circuit.append(evo, range(2))
        transpiled_qiskit_circuit = transpile(qiskit_circuit, qulacs_backend)

        qulacs_job = qulacs_backend.run(transpiled_qiskit_circuit)
        qulacs_result = qulacs_job.result().get_statevector()

        transpiled_qiskit_circuit.save_statevector()
        qiskit_job = self.aer_backend.run(transpiled_qiskit_circuit)
        qiskit_result = np.array(qiskit_job.result().get_statevector())

        self.assertTrue(np.allclose(qulacs_result, qiskit_result))

    def test_unitary_gate(self):
        """Test for unitary gate"""
        qiskit_circuit = QuantumCircuit(2)
        qiskit_circuit.unitary(fsim_mat, [0, 1])

        convert_and_check_statevector(self, qiskit_circuit)


class TestConverterWarningsAndErrors(TestCase):
    def test_params_not_passed(self):
        """Tests that a warning is raised if circuit has params but not params passed."""
        qc = QuantumCircuit(1)
        qc.rx(Parameter("θ"), 0)

        with pytest.raises(
            ValueError, match="The number of circuit parameters does not match"
        ):
            qulacs_circuit_builder = convert_qiskit_to_qulacs_circuit(qc)
            qulacs_circuit_builder()[0]

    def test_template_not_supported(self):
        """Tests that a warning is raised if an unsupported instruction was reached."""
        qc = TwoLocal(
            4,
            ["rx", "ry", "rz"],
            ["cz", "cx"],
            "linear",
            reps=1,
        )

        params = np.random.uniform(size=qc.num_parameters)

        with pytest.raises(
            ValueError, match="The Gate does not support trainable parameter"
        ):
            qulacs_circuit_builder = convert_qiskit_to_qulacs_circuit(qc)
            qulacs_circuit_builder(params)[0]

    def test_unsupported_gate(self):
        """Tests that a warning is raised if an unsupported gate was reached"""
        qc = QuantumCircuit(1)
        qc.rx(0.1, 0)
        qc.measure_all()

        with pytest.warns(UserWarning, match="not supported by Qiskit-Qulacs"):
            qulacs_circuit_builder = convert_qiskit_to_qulacs_circuit(qc)
            qulacs_circuit_builder()[0]


dummy_obs = [Observable(1), Observable(3), Observable(2)]

dummy_obs[0].add_operator(PauliOperator("I 0", 2.0))
dummy_obs[1].add_operator(PauliOperator("Z 0 Y 1 X 2", 1.0))

dummy_obs[2].add_operator(PauliOperator("Y 0 X 1", 3.0))
dummy_obs[2].add_operator(PauliOperator("X 0 Z 1", 7.0))


observable_factories = [
    (SparsePauliOp("I", coeffs=[2]), dummy_obs[0]),
    (SparsePauliOp("XYZ"), dummy_obs[1]),
    (SparsePauliOp(["XY", "ZX"], coeffs=[3, 7]), dummy_obs[2]),
]


@ddt
class TestConverterObservable(TestCase):
    """Tests for the Adapter class."""

    @data(*observable_factories)
    def test_convert_with_coefficients(self, ops):
        """Tests that a SparsePauliOp can be converted into a PennyLane operator with the default
        coefficients.
        """
        pauli_op, want_op = ops
        have_op = convert_sparse_pauliop_to_qulacs_obs(pauli_op)

        assert have_op.to_json() == want_op.to_json()
