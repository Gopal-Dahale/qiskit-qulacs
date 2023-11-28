"""Tests for the Adapter class."""
from unittest import TestCase

import numpy as np
from qiskit import BasicAer, QuantumCircuit, execute
from qiskit import extensions as ex
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate, TwoLocal
from qiskit.quantum_info import SparsePauliOp

from qiskit_qulacs.adapter import circuit_mapper, convert_qiskit_to_qulacs_circuit
from qiskit_qulacs.qulacs_backend import QulacsBackend
from qulacs import ParametricQuantumCircuit, QuantumState

_EPS = 1e-10  # global variable used to chop very small numbers to zero

qiskit_standard_gates = [
    ex.IGate(),
    ex.SXGate(),
    ex.XGate(),
    ex.CXGate(),
    ex.RZGate(Parameter("λ")),
    ex.RGate(Parameter("ϴ"), Parameter("φ")),
    ex.C3SXGate(),
    ex.CCXGate(),
    ex.DCXGate(),
    ex.CHGate(),
    ex.CPhaseGate(Parameter("ϴ")),
    ex.CRXGate(Parameter("ϴ")),
    ex.CRYGate(Parameter("ϴ")),
    ex.CRZGate(Parameter("ϴ")),
    ex.CSwapGate(),
    ex.CSXGate(),
    ex.CUGate(Parameter("ϴ"), Parameter("φ"), Parameter("λ"), Parameter("γ")),
    ex.CU1Gate(Parameter("λ")),
    ex.CU3Gate(Parameter("ϴ"), Parameter("φ"), Parameter("λ")),
    ex.CYGate(),
    ex.CZGate(),
    ex.CCZGate(),
    ex.HGate(),
    ex.PhaseGate(Parameter("ϴ")),
    ex.RCCXGate(),
    ex.RC3XGate(),
    ex.RXGate(Parameter("ϴ")),
    ex.RXXGate(Parameter("ϴ")),
    ex.RYGate(Parameter("ϴ")),
    ex.RYYGate(Parameter("ϴ")),
    ex.RZZGate(Parameter("ϴ")),
    ex.RZXGate(Parameter("ϴ")),
    ex.XXMinusYYGate(Parameter("ϴ"), Parameter("φ")),
    ex.XXPlusYYGate(Parameter("ϴ"), Parameter("φ")),
    ex.ECRGate(),
    ex.SGate(),
    ex.SdgGate(),
    ex.CSGate(),
    ex.CSdgGate(),
    ex.SwapGate(),
    ex.iSwapGate(),
    ex.SXdgGate(),
    ex.TGate(),
    ex.TdgGate(),
    ex.UGate(Parameter("ϴ"), Parameter("φ"), Parameter("λ")),
    ex.U1Gate(Parameter("λ")),
    ex.U2Gate(Parameter("φ"), Parameter("λ")),
    ex.U3Gate(Parameter("ϴ"), Parameter("φ"), Parameter("λ")),
    ex.YGate(),
    ex.ZGate(),
]


class TestAdapter(TestCase):
    """Tests for the Adapter class."""

    def test_state_preparation_01(self):
        """Tests state_preparation handling of Adapter"""

        input_state_vector = np.array([np.sqrt(3) / 2, np.sqrt(2) * complex(1, 1) / 4])

        qiskit_circuit = QuantumCircuit(1)
        qiskit_circuit.prepare_state(input_state_vector, 0)

        qulacs_circuit_builder, _ = convert_qiskit_to_qulacs_circuit(
            qiskit_circuit.decompose()
        )
        qulacs_circuit = qulacs_circuit_builder()
        quantum_state = QuantumState(1)
        qulacs_circuit.update_quantum_state(quantum_state)
        output_state_vector = quantum_state.get_vector()

        self.assertTrue(
            (np.linalg.norm(input_state_vector - output_state_vector)) < _EPS
        )

    def test_state_preparation_00(self):
        """Tests state_preparation handling of Adapter"""
        input_state_vector = np.array([1 / np.sqrt(2), -1 / np.sqrt(2)])

        qiskit_circuit = QuantumCircuit(1)
        qiskit_circuit.prepare_state(input_state_vector, 0)

        qulacs_circuit_builder, _ = convert_qiskit_to_qulacs_circuit(
            qiskit_circuit.decompose()
        )
        qulacs_circuit = qulacs_circuit_builder()
        quantum_state = QuantumState(1)
        qulacs_circuit.update_quantum_state(quantum_state)
        output_state_vector = quantum_state.get_vector()

        self.assertTrue(
            (np.linalg.norm(input_state_vector - output_state_vector)) < _EPS
        )

    def test_u_gate(self):
        """Tests adapter conversion of u gate"""
        qiskit_circuit = QuantumCircuit(1)
        aer_backend = BasicAer.get_backend("statevector_simulator")

        qiskit_circuit.u(np.pi / 2, np.pi / 3, np.pi / 4, 0)
        job = execute(qiskit_circuit, aer_backend)

        qulacs_circuit_builder, _ = convert_qiskit_to_qulacs_circuit(qiskit_circuit)
        qulacs_circuit = qulacs_circuit_builder()
        quantum_state = QuantumState(1)
        qulacs_circuit.update_quantum_state(quantum_state)
        qulacs_output = quantum_state.get_vector()

        qiskit_output = np.array(job.result().get_statevector())

        self.assertTrue(np.linalg.norm(qulacs_output - qiskit_output) < _EPS)

    def test_standard_gate_decomp(self):
        """Tests adapter decomposition of all standard gates to forms that can be translated"""
        aer_backend = BasicAer.get_backend("statevector_simulator")
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
                qiskit_circuit = qiskit_circuit.bind_parameters(parameter_bindings)

            if standard_gate.name not in ["cu"]:
                # parameters are not binding to cu. I am not sure why.

                with self.subTest(f"Circuit with {standard_gate.name} gate."):
                    qulacs_job = qulacs_backend.run(qiskit_circuit)
                    qulacs_result = qulacs_job.result().get_statevector()

                    qiskit_job = execute(qiskit_circuit, aer_backend)
                    qiskit_result = qiskit_job.result().get_statevector()

                    self.assertTrue(
                        np.linalg.norm(qulacs_result - qiskit_result) < _EPS
                    )

    def test_exponential_gate_decomp(self):
        """Tests adapter translation of exponential gates"""
        aer_backend = BasicAer.get_backend("statevector_simulator")
        qulacs_backend = QulacsBackend()
        qiskit_circuit = QuantumCircuit(2)

        hamiltonian = SparsePauliOp(["ZZ", "XI"], [1.0, -0.1])
        evo = PauliEvolutionGate(hamiltonian, time=2)

        qiskit_circuit.append(evo, range(2))

        qulacs_job = qulacs_backend.run(qiskit_circuit)
        qulacs_result = qulacs_job.result().get_statevector()

        qiskit_job = execute(qiskit_circuit, aer_backend)
        qiskit_result = qiskit_job.result().get_statevector()

        self.assertTrue(np.linalg.norm(qulacs_result - qiskit_result) < _EPS)

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

        qulacs_circuit_builder, _ = convert_qiskit_to_qulacs_circuit(qiskit_circuit)
        qulacs_circuit = qulacs_circuit_builder(params)
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

        self.assertTrue(np.linalg.norm(qulacs_result - qulacs_result_ans) < _EPS)
