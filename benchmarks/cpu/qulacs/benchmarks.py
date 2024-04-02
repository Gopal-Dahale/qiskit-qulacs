import numpy as np
import pytest

from qulacs import Observable, ParametricQuantumCircuit, PauliOperator, QuantumState
from qulacs.circuit import QuantumCircuitOptimizer as QCO

np.random.seed(0)
nqubits_list = range(4, 21)
nlayers = 3


def generate_circuit(nqubits):
    circuit = ParametricQuantumCircuit(nqubits)

    params = np.random.rand(nlayers + 1, nqubits, 3)

    for l in range(nlayers):
        for q in range(nqubits):
            circuit.add_parametric_RX_gate(q, params[l, q, 0])
            circuit.add_parametric_RY_gate(q, params[l, q, 1])
            circuit.add_parametric_RZ_gate(q, params[l, q, 2])
        for q in range(nqubits - 1):
            circuit.add_CNOT_gate(q, q + 1)

    # final rotation layer
    for q in range(nqubits):
        circuit.add_parametric_RX_gate(q, params[nlayers, q, 0])
        circuit.add_parametric_RY_gate(q, params[nlayers, q, 1])
        circuit.add_parametric_RZ_gate(q, params[nlayers, q, 2])

    return circuit, params


def generate_obs(nqubits):
    obs = Observable(nqubits)
    pauli_string = ""
    for q in range(nqubits):
        pauli_string += f"Z {q} "
    obs.add_operator(PauliOperator(pauli_string, 1.0))
    return obs


def execute_statevector(benchmark, circuit, params):
    qco = QCO()
    state = QuantumState(circuit.get_qubit_count())

    def evalfunc(circuit, state):
        qco.optimize_light(circuit)
        circuit.update_quantum_state(state)
        sv = state.get_vector()

    benchmark(evalfunc, circuit, state)


def execute_estimator(benchmark, circuit, obs, params):
    qco = QCO()
    state = QuantumState(circuit.get_qubit_count())

    def evalfunc(circuit, obs, params, state):
        qco.optimize_light(circuit)
        circuit.update_quantum_state(state)
        expval = obs.get_expectation_value(state)

    benchmark(evalfunc, circuit, obs, params, state)


def execute_estgradient(benchmark, circuit, obs, params):
    qco = QCO()
    state = QuantumState(circuit.get_qubit_count())

    def evalfunc(circuit, obs, params, state):
        qco.optimize_light(circuit)
        circuit.update_quantum_state(state)
        grads = circuit.backprop(obs)

    benchmark(evalfunc, circuit, obs, params, state)


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_statevector(benchmark, nqubits):
    benchmark.group = "qulacs_statevector"
    circuit, params = generate_circuit(nqubits)
    execute_statevector(benchmark, circuit, params)


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_estimator(benchmark, nqubits):
    benchmark.group = "qulacs_estimator"
    circuit, params = generate_circuit(nqubits)
    obs = generate_obs(nqubits)
    execute_estimator(benchmark, circuit, obs, params)


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_estgradient(benchmark, nqubits):
    benchmark.group = "qulacs_estgradient"
    circuit, params = generate_circuit(nqubits)
    obs = generate_obs(nqubits)
    execute_estgradient(benchmark, circuit, obs, params)
